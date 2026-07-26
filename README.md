# PSBA-20

**Principal-Specific Behavioural Auditing Benchmark for Detecting Directional Asymmetries in Language Models**

A generator, not a dataset. Swap one config file to audit a different principal.

- **[SPEC.md](SPEC.md)** — the full specification (all 12 sections; read this first)
- **[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md)** — pre-registered hypotheses, tests, corrections
- **[RUBRIC.md](RUBRIC.md)** — human scoring rubric
- **[LIMITATIONS.md](LIMITATIONS.md)** — written before data collection

---

## The one-paragraph version

Twenty low-stakes administrative decision templates, each with three matched options: one that advantages a party, one neutral, one that advantages the other party. A principal's name is substituted into a slot and compared against a salience-matched principal, a generic descriptor, and a fictional name. Crucially the two parties then **exchange slots**, and the outcome is always coded as *"chose the option that advantages the principal"* — so a loyalty shifts the same direction in both framings while position bias and content preference cancel. Every template is pre-screened for headroom on a clean baseline and ships with its measured entropy, so a null result is interpretable. The primary signal is not the choice shift alone but its **dissociation from the stated reasons**: choices moving with the principal while the justification vocabulary stays constant and never mentions it.

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. generate + validate an English dataset (960 items, 60 families)
python -m psba.cli generate --languages en --paraphrases 3 --orders 2 --out data/items.jsonl

# 2. power / MDE table for the three tiers
python -m psba.cli power --simulate

# 3. run the correctness tests (20 tests, no GPU needed)
python3 tests/test_pipeline.py
```

With a GPU and the organisms downloaded:

```bash
# affordance pre-screen on the clean baseline — MANDATORY before analysis
python - <<'PY'
from pathlib import Path
from psba.run_model import RunConfig, run_items
from psba.cli import _jsonl
items = [i for i in _jsonl(Path('data/items.jsonl')) if i['control_type']=='neutral_entity']
run_items(items, RunConfig(model_id='Qwen/Qwen2.5-7B-Instruct', role='baseline', k=40),
          out_path=Path('runs/baseline.jsonl'))
PY

python -m psba.cli prescreen --items data/items.jsonl --raw runs/baseline.jsonl \
    --out-cards results/affordance_cards.json --out-items data/items.screened.jsonl

# then the organisms, then score, then analyse
python -m psba.cli score   --items data/items.screened.jsonl --raw runs/organismA.jsonl
python -m psba.cli analyse --results results/cells.jsonl
```

Adding a language is a human-authoring step by design — the generator refuses machine translation:

```bash
python -m psba.cli scaffold-language sw   # emits an authoring skeleton with all content blanked
```

---

## Layout

```
SPEC.md ANALYSIS_PLAN.md RUBRIC.md LIMITATIONS.md
templates/templates.en.yaml      20 decision templates (§2 format, machine-readable)
config/principals.example.yaml   the principal bundle — replace this to audit anyone else
config/languages.yaml            language set + H4 hypothesis + adaptation protocol
config/paraphrases.yaml          3 paraphrase frames per language
config/lexicons.yaml             checker lexicons + closed rationale taxonomy
schemas/item.schema.json         item schema (artifact 1)
schemas/results.schema.json      analysis-ready results table (artifact 5)
psba/generate.py                 generator (artifact 2)
psba/validate.py                 automated validity checks (artifact 2)
psba/affordance.py               mandatory pre-screen (§3)
psba/run_model.py                vLLM / HF / OpenAI-compatible harness, logprob capture
psba/score.py                    4-channel scoring, adjudication, dissociation (artifact 4)
psba/analyze.py                  permutation tests, FDR, power/MDE, ROC
psba/cli.py                      entry points
tests/test_pipeline.py           20 correctness tests
```

---

## Two numbers to know before using this

**Minimum detectable effect.** 8.2 pp (English Pilot), 6.7 pp (Tier-1, four languages), in percentage points of P(principal-aligned choice). A null result means *"no asymmetry above this"*, never *"no hidden objective"*.

**The binding constraint is template count, not compute.** 58% of the variance of the headline estimate is between-template, so at 20 templates the MDE floor is 6.3 pp even with infinite samples. Adding templates helps; adding samples past k ≈ 20 barely does.

---

## Two tests worth reading

`tests/test_pipeline.py` contains the pair that establishes the design's identifying claim:

- `test_loyalty_recovered_from_simulation` — a simulated 12 pp loyalty is detected.
- `test_position_bias_does_not_produce_dai` — an injected 15 pp *position* artifact, larger than any effect the benchmark claims to detect, produces **no** significant directional asymmetry, and is instead caught by the non-directional responsiveness statistic that exists for it.

Together they are the argument that this instrument measures loyalty rather than name-responsiveness.

---

Prepared for the Apart Research Secret Loyalties Hackathon, Track 2 (Detection & Auditing). Builds on methodological constraints from Blanche (2026), [The Transmutation Gap](https://apartresearch.com/project/the-transmutation-gap-crosslingual-coherence-evaluation-in-large-language-models-using-the-sovereigntycollaboration-transmutational-arc-framework-zukw)

