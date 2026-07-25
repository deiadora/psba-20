"""Inference harness.

Backends are pluggable; only `generate_batch` must be implemented. Three are
shipped: vLLM (local, recommended for the Qwen2.5 organisms), HuggingFace
transformers (fallback), and an OpenAI-compatible HTTP client (for scorers and
for frontier-model exploratory audits).

Sampling policy (SPEC 8.2) is BOTH stochastic-repeated AND paraphrase-varied:

  stochastic-repeated  k samples at T=1.0 per item. Estimates the model's
                       choice DISTRIBUTION, which is the object of inference.
                       Cost: within-item samples are correlated, so k buys less
                       than it appears to; effective n is driven by item count.
  deterministic + paraphrase  one greedy sample per paraphrase. Removes
                       sampling noise but measures only the argmax, which is
                       insensitive to distributional shifts that do not cross
                       the mode. A loyalty that moves p from 0.30 to 0.45
                       is invisible to greedy decoding.

Neither alone is sufficient: the first cannot separate distributional shift from
surface artifact, the second cannot see sub-modal shift. The benchmark runs both
and reports them separately; the primary analysis uses the stochastic arm with
paraphrase as a random effect.

Logprobs (SPEC 9.5): where the backend exposes them, the harness records the
log-probability of each option letter token at the choice position. This adds
two things sampled proportions cannot give: (a) a continuous, zero-variance-free
readout on items where sampling has collapsed to one option, so an item at
behavioural ceiling can still show a graded shift; (b) a k-independent estimate,
cutting the samples needed for the same precision. Models that do not expose
logprobs fall back to sampled proportions only, and `logprob_available=False` is
recorded so the analysis can stratify rather than silently mix estimators.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

CHOICE_RE = re.compile(r"choice\s*[:：]\s*\(?([ABC])\)?", re.IGNORECASE)
REASON_RE = re.compile(r"reason\s*[:：]\s*(.*)", re.IGNORECASE | re.DOTALL)


@dataclass
class RunConfig:
    model_id: str
    role: str = "subject"                 # subject | baseline | scorer
    k: int = 15                           # stochastic samples per item
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 160
    greedy_arm: bool = True               # additionally run one T=0 sample
    seed: int = 20260725
    logprobs_top_k: int = 20
    batch_size: int = 64
    backend: str = "vllm"                 # vllm | hf | openai_compatible
    api_base: str | None = None
    extra: dict = field(default_factory=dict)


class Backend(Protocol):
    supports_logprobs: bool
    def generate_batch(self, prompts: list[str], cfg: RunConfig,
                       temperature: float, seed: int) -> list[dict]: ...


# --------------------------------------------------------------------------
class VLLMBackend:
    """Local open-weight inference. Recommended for the Qwen2.5 organisms.

    Throughput note for planning: a 7B model on one 80GB accelerator with
    continuous batching produces roughly 15-25 completions/second at ~140
    output tokens, i.e. the Pilot tier's ~154k generations in about 3 hours of
    wall clock. This is what makes the Pilot tier affordable without a grant.
    """
    supports_logprobs = True

    def __init__(self, model_id: str, **kw):
        from vllm import LLM  # imported lazily so the repo installs without GPU deps
        self.llm = LLM(model=model_id, **kw)

    def generate_batch(self, prompts, cfg, temperature, seed):
        from vllm import SamplingParams
        sp = SamplingParams(n=1, temperature=temperature, top_p=cfg.top_p,
                            max_tokens=cfg.max_new_tokens, seed=seed,
                            logprobs=cfg.logprobs_top_k)
        outs = self.llm.generate(prompts, sp)
        results = []
        for o in outs:
            c = o.outputs[0]
            results.append({"text": c.text, "logprobs": _extract_letter_logprobs(c)})
        return results


def _extract_letter_logprobs(completion) -> dict[str, float] | None:
    """Logprob of A/B/C at the first position where a letter is licensed.

    Implementation detail that matters: the letter must be read at the choice
    position, not anywhere in the completion, or the number is not a choice
    probability. We locate the first token whose top-k contains at least two of
    the three letters and read all three there.
    """
    lps = getattr(completion, "logprobs", None)
    if not lps:
        return None
    for step in lps:
        if step is None:
            continue
        toks = {}
        for _tid, info in step.items():
            tok = getattr(info, "decoded_token", None) or ""
            t = tok.strip().upper()
            if t in ("A", "B", "C"):
                toks[t] = float(getattr(info, "logprob", float("-inf")))
        if len(toks) >= 2:
            return {L: toks.get(L, float("-inf")) for L in "ABC"}
    return None


class HFBackend:
    supports_logprobs = True

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype)).to(device)
        self.device = device

    def generate_batch(self, prompts, cfg, temperature, seed):
        self.torch.manual_seed(seed)
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(
            **enc, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
            top_p=cfg.top_p, max_new_tokens=cfg.max_new_tokens,
            return_dict_in_generate=True, output_scores=True)
        texts = self.tok.batch_decode(out.sequences[:, enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
        return [{"text": t, "logprobs": None} for t in texts]


class OpenAICompatibleBackend:
    """For scorer models and frontier exploratory audits.

    Used for scorers precisely because scorers must come from a different model
    family than any subject model (SPEC 9.6); running them through a separate
    API path makes that separation auditable in the run log rather than a claim.
    """
    supports_logprobs = True

    def __init__(self, model_id: str, api_base: str, api_key_env: str = "PSBA_API_KEY"):
        import urllib.request
        self._urllib = urllib.request
        self.model_id, self.api_base = model_id, api_base.rstrip("/")
        self.key = os.environ.get(api_key_env, "")

    def generate_batch(self, prompts, cfg, temperature, seed):
        results = []
        for p in prompts:
            body = json.dumps({
                "model": self.model_id,
                "messages": [{"role": "user", "content": p}],
                "temperature": temperature, "top_p": cfg.top_p,
                "max_tokens": cfg.max_new_tokens, "seed": seed,
                "logprobs": True, "top_logprobs": min(cfg.logprobs_top_k, 20),
            }).encode()
            req = self._urllib.Request(
                f"{self.api_base}/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.key}"})
            for attempt in range(5):
                try:
                    with self._urllib.urlopen(req, timeout=120) as r:
                        d = json.loads(r.read())
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            ch = d["choices"][0]
            results.append({"text": ch["message"]["content"],
                            "logprobs": _extract_api_letter_logprobs(ch)})
        return results


def _extract_api_letter_logprobs(choice: dict) -> dict[str, float] | None:
    content = (choice.get("logprobs") or {}).get("content") or []
    for step in content:
        toks = {}
        for alt in step.get("top_logprobs", []):
            t = alt["token"].strip().upper()
            if t in ("A", "B", "C"):
                toks[t] = float(alt["logprob"])
        if len(toks) >= 2:
            return {L: toks.get(L, float("-inf")) for L in "ABC"}
    return None


def make_backend(cfg: RunConfig) -> Backend:
    if cfg.backend == "vllm":
        return VLLMBackend(cfg.model_id, **cfg.extra)
    if cfg.backend == "hf":
        return HFBackend(cfg.model_id, **cfg.extra)
    if cfg.backend == "openai_compatible":
        assert cfg.api_base, "api_base required"
        return OpenAICompatibleBackend(cfg.model_id, cfg.api_base, **cfg.extra)
    raise ValueError(f"unknown backend {cfg.backend}")


# --------------------------------------------------------------------------
def run_items(items: list[dict], cfg: RunConfig, backend: Backend | None = None,
              out_path: Path | None = None) -> list[dict]:
    """Emit one raw record per (item, sample). Nothing is scored here.

    Separating generation from scoring is deliberate: the scoring models must be
    re-runnable without re-querying the subject model, and scorer versions must
    be swappable without invalidating the behavioural data.
    """
    backend = backend or make_backend(cfg)
    records: list[dict] = []
    arms = [(i, cfg.temperature) for i in range(cfg.k)]
    if cfg.greedy_arm:
        arms.append(("greedy", 0.0))

    for sample_index, temp in arms:
        seed = cfg.seed + (sample_index if isinstance(sample_index, int) else 9999)
        for start in range(0, len(items), cfg.batch_size):
            chunk = items[start:start + cfg.batch_size]
            outs = backend.generate_batch([c["prompt"] for c in chunk], cfg, temp, seed)
            for it, o in zip(chunk, outs):
                records.append({
                    "item_id": it["item_id"],
                    "family_id": it["family_id"],
                    "template_id": it["template_id"],
                    "language": it["language"],
                    "principal_role": it["principal_role"],
                    "control_type": it["control_type"],
                    "framing": it["framing"],
                    "swap_strength": it.get("swap_strength", "full"),
                    "paraphrase_id": it["paraphrase_id"],
                    "option_display_order": it["option_display_order"],
                    "principal_aligned_label": it["principal_aligned_label"],
                    "neutral_label": it["neutral_label"],
                    "subject_model": cfg.model_id,
                    "subject_role": cfg.role,
                    "sample_index": sample_index,
                    "temperature": temp,
                    "seed": seed,
                    "raw_text": o["text"],
                    "letter_logprobs": o.get("logprobs"),
                    "logprob_available": o.get("logprobs") is not None,
                })
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records
