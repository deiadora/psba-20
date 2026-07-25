"""Command line entry points.

  python -m psba.cli generate      --languages en --paraphrases 3 --out data/items.jsonl
  python -m psba.cli validate      --items data/items.jsonl
  python -m psba.cli scaffold-language sw
  python -m psba.cli prescreen     --items data/items.jsonl --raw runs/baseline.jsonl
  python -m psba.cli score         --items data/items.jsonl --raw runs/organismA.jsonl
  python -m psba.cli analyse       --results results/cells.jsonl
  python -m psba.cli power
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from .affordance import apply_cards_to_items, screen_template
from .analyze import (PowerSpec, benjamini_hochberg, build_results_table,
                      cluster_permutation_test, gee_secondary, item_contrasts, mde,
                      simulate_power)
from .generate import GenConfig, generate, write_jsonl
from .score import (adjudicate, dissociation, extract_explanation,
                    score_choice, sample_calibration_subset)
from .validate import load_lexicons, validate_dataset

ROOT = Path(__file__).resolve().parent.parent
DEF = {
    "templates_dir": ROOT / "templates",
    "principals": ROOT / "config" / "principals.example.yaml",
    "languages": ROOT / "config" / "languages.yaml",
    "paraphrases": ROOT / "config" / "paraphrases.yaml",
    "lexicons": ROOT / "config" / "lexicons.yaml",
}


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def cmd_generate(a):
    cfg = GenConfig(templates_dir=Path(a.templates_dir), principals_path=Path(a.principals),
                    languages_path=Path(a.languages_config),
                    paraphrases_path=Path(a.paraphrases_config),
                    languages=a.languages, paraphrases_per_item=a.paraphrases,
                    orders_per_family=a.orders, smoke_test_fallback_en=a.smoke_test_fallback_en)
    items = generate(cfg)
    lex = load_lexicons(Path(a.lexicons))
    bundle = yaml.safe_load(Path(a.principals).read_text(encoding="utf-8"))
    report = validate_dataset(items, lex, bundle, enforce=not a.no_enforce)
    n = write_jsonl(items, Path(a.out))
    print(json.dumps({k: v for k, v in report.items() if k != "failure_summary"},
                     default=str, indent=1))
    print(f"wrote {n} items -> {a.out}")


def cmd_validate(a):
    items = list(_jsonl(Path(a.items)))
    lex = load_lexicons(Path(a.lexicons))
    bundle = yaml.safe_load(Path(a.principals).read_text(encoding="utf-8"))
    rep = validate_dataset(items, lex, bundle, enforce=not a.no_enforce)
    print(json.dumps({k: v for k, v in rep.items() if k != "failure_summary"},
                     default=str, indent=1))


def cmd_scaffold_language(a):
    """Emit an authoring skeleton for a new language.

    Deliberately NOT a translation. The file arrives with empty strings and the
    adaptation checklist inline, so a native author has to make the locale
    decisions the protocol requires rather than approve a machine draft.
    """
    src = yaml.safe_load((Path(a.templates_dir) / "templates.en.yaml").read_text(encoding="utf-8"))
    out = {"meta": dict(src["meta"], adaptation_level="adapted", source_language="en",
                        language=a.lang, authored_by="TODO native author",
                        checklist=["institutions locally plausible",
                                   "meaningfulness_score recorded per template",
                                   "benefit/neutral/opposing labels re-validated by two "
                                   "blind native speakers, NOT inherited from English",
                                   "option strings parallel imperatives, lengths matched",
                                   "back-translated and checked for stakes/affect drift"]),
           "templates": []}
    for t in src["templates"]:
        out["templates"].append({
            "id": t["id"], "name": t["name"], "domain": t["domain"],
            "rationale": t["rationale"], "source_context_en": t["decision_context"],
            "decision_context": "TODO: adapt, do not translate. Keep {{HOLDER}}/{{OTHER}} "
                                "attachment lines contiguous and final.",
            "options": {r: {"text": "TODO", "source_en": t["options"][r]["text"]}
                        for r in ("benefit", "neutral", "opposing")},
            "affordance_rationale": t["affordance_rationale"],
            "confounds": t["confounds"], "controls": t["controls"],
            "meaningfulness_score": None, "benefit_direction_agreement": None,
        })
    out_dir = Path(getattr(a, "out_dir", None) or a.templates_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"templates.{a.lang}.yaml"
    p.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"scaffold -> {p} (all TODOs must be filled by a native author before use)")


def _score_raw(items_by_id, raw_records):
    out = []
    for r in raw_records:
        it = items_by_id[r["item_id"]]
        cs = score_choice(r["raw_text"], it)
        lp = r.get("letter_logprobs")
        share = None
        if lp:
            import math
            mx = max(lp.values())
            exps = {k: math.exp(v - mx) for k, v in lp.items()}
            tot = sum(exps.values())
            share = exps[it["principal_aligned_label"]] / tot if tot else None
        out.append(dict(r, choice_label=cs.label, choice_status=cs.status,
                        choice_role=cs.role, choice_method=cs.method,
                        explanation=extract_explanation(r["raw_text"]),
                        logprob_aligned_share=share))
    return out


def cmd_prescreen(a):
    items = {i["item_id"]: i for i in _jsonl(Path(a.items))}
    raw = [r for r in _jsonl(Path(a.raw)) if r["control_type"] == "neutral_entity"]
    scored = _score_raw(items, raw)
    for s in scored:
        it = items[s["item_id"]]
        s["aligned_position"] = it["option_display_order"].index(
            "benefit" if it["framing"] == "forward" else "opposing")
    groups = defaultdict(list)
    for s in scored:
        groups[(s["template_id"], s["language"])].append(s)
    cards = {k: screen_template(k[0], k[1], v, a.baseline_model, a.temperature)
             for k, v in groups.items()}
    kept, dropped = apply_cards_to_items(list(items.values()), cards)
    Path(a.out_cards).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_cards).write_text(json.dumps(
        [c.as_dict() for c in cards.values()], indent=1), encoding="utf-8")
    write_jsonl(kept, Path(a.out_items))
    print(json.dumps({"cards": len(cards), "items_kept": len(kept), "dropped": dropped},
                     indent=1))
    for c in sorted(cards.values(), key=lambda c: c.template_id):
        print(f"  {c.template_id} {c.language}  p_aligned={c.p_aligned:.2f} "
              f"H={c.entropy_norm:.2f} r_split={c.split_half_reliability:.2f} "
              f"-> {c.screen_status}  {c.screen_notes[:70]}")


def cmd_score(a):
    items = {i["item_id"]: i for i in _jsonl(Path(a.items))}
    scored = _score_raw(items, _jsonl(Path(a.raw)))
    write_jsonl(scored, Path(a.out))
    print(f"scored {len(scored)} records -> {a.out}")
    print("NOTE: rationale coding requires two non-Qwen scorer models; run "
          "psba.score.code_explanations for each and psba.score.adjudicate to merge.")


def cmd_analyse(a):
    rows = list(_jsonl(Path(a.results)))
    cs = item_contrasts(rows, control_type=a.control)
    primary = cluster_permutation_test(cs, statistic="dai")
    ndr = cluster_permutation_test(cs, statistic="ndr")
    per_tpl = defaultdict(list)
    for c in cs:
        per_tpl[c.template_id].append(c)
    pv = {t: cluster_permutation_test(v, statistic="dai", n_perm=2000)["p_value"]
          for t, v in per_tpl.items() if len(v) >= 3}
    print(json.dumps({
        "primary_DAI": {k: v for k, v in primary.items() if k != "template_means"},
        "non_directional_responsiveness": {k: v for k, v in ndr.items() if k != "template_means"},
        "per_template_bh": benjamini_hochberg(pv) if pv else {},
        "gee_secondary": gee_secondary(rows, a.control).get("pvalues", "unavailable"),
    }, default=str, indent=1))


def cmd_power(a):
    tiers = {
        "Pilot-A (English only)": dict(n_templates=20, items_per_template=3, samples_per_cell=30),
        "Pilot-B (EN + 1 adapted)": dict(n_templates=20, items_per_template=6, samples_per_cell=30),
        "Tier-1 Full (4 languages)": dict(n_templates=20, items_per_template=12, samples_per_cell=40),
    }
    for name, kw in tiers.items():
        m1, m2 = mde(PowerSpec(**kw)), mde(PowerSpec(**kw, n_comparisons=20))
        print(f"{name:28} items={kw['n_templates']*kw['items_per_template']:4d} "
              f"gens/model={m1['generations_per_model']:8,d} "
              f"MDE={m1['mde_pp']:5.2f}pp  MDE(FDR m=20)={m2['mde_pp']:5.2f}pp")
    if a.simulate:
        s = PowerSpec(**tiers["Pilot-A (English only)"])
        for d in (0.0, 0.08, 0.10, 0.15):
            print(f"  simulated size/power at delta={d:.2f}: "
                  f"{simulate_power(s, d, n_sim=a.n_sim, n_perm=400):.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="psba")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--templates-dir", default=str(DEF["templates_dir"]))
        sp.add_argument("--principals", default=str(DEF["principals"]))
        # NB: distinct dests from the experimental factors below. An earlier
        # version bound --languages-config to dest="languages", which collided
        # with the language LIST on `generate` and produced a Path(list) error.
        sp.add_argument("--languages-config", dest="languages_config",
                        default=str(DEF["languages"]))
        sp.add_argument("--paraphrases-config", dest="paraphrases_config",
                        default=str(DEF["paraphrases"]))
        sp.add_argument("--lexicons", default=str(DEF["lexicons"]))
        sp.add_argument("--no-enforce", action="store_true")

    g = sub.add_parser("generate"); common(g)
    g.add_argument("--languages", nargs="*", default=["en"])
    g.add_argument("--paraphrases", type=int, default=3)
    g.add_argument("--orders", type=int, default=2)
    g.add_argument("--smoke-test-fallback-en", action="store_true")
    g.add_argument("--out", default="data/items.jsonl")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate"); common(v)
    v.add_argument("--items", required=True); v.set_defaults(func=cmd_validate)

    s = sub.add_parser("scaffold-language")
    s.add_argument("lang"); s.add_argument("--templates-dir", default=str(DEF["templates_dir"]))
    s.add_argument("--out-dir", default=None,
                   help="write the skeleton elsewhere; default is --templates-dir")
    s.set_defaults(func=cmd_scaffold_language)

    pr = sub.add_parser("prescreen")
    pr.add_argument("--items", required=True); pr.add_argument("--raw", required=True)
    pr.add_argument("--baseline-model", default="Qwen/Qwen2.5-7B-Instruct")
    pr.add_argument("--temperature", type=float, default=1.0)
    pr.add_argument("--out-cards", default="results/affordance_cards.json")
    pr.add_argument("--out-items", default="data/items.screened.jsonl")
    pr.set_defaults(func=cmd_prescreen)

    sc = sub.add_parser("score")
    sc.add_argument("--items", required=True); sc.add_argument("--raw", required=True)
    sc.add_argument("--out", default="results/scored.jsonl"); sc.set_defaults(func=cmd_score)

    an = sub.add_parser("analyse")
    an.add_argument("--results", required=True)
    an.add_argument("--control", default="matched_principal")
    an.set_defaults(func=cmd_analyse)

    po = sub.add_parser("power")
    po.add_argument("--simulate", action="store_true")
    po.add_argument("--n-sim", type=int, default=600)
    po.set_defaults(func=cmd_power)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
