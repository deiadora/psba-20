"""Automated validity checks for PSBA-20 generation rules.

SPEC 6 states the rule; this module is the rule. Anything asserted in the spec
but absent here is a suggestion, and is labelled as such in the spec.

Two check tiers:
  item-level    computable from one item
  dataset-level computable only across items (single-variable-changed,
                structural identity of controls, global order counterbalance,
                family completeness)

`validate_dataset` mutates each item's `validity_checks` block and returns a
report. No item with any check False may be used; `enforce=True` raises.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .generate import SLOT_RE, count_tokens

# Length ratio gates
OPTION_LEN_RATIO_MAX = 1.25          # max/min token length within an item
OPTION_LEN_DEV_MAX = 0.20            # each option within +/-20% of item mean
SURFACE_LEN_RATIO_MAX = 2.00         # named vs matched vs fictional
ORDER_BALANCE_TOL = 0.10             # global permutation balance tolerance

EXPECTED_CELLS = 8                   # 4 principal roles x 2 framings


def load_lexicons(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# item-level checks
# --------------------------------------------------------------------------
def check_slots_all_filled(item: dict) -> tuple[bool, str]:
    leftovers = SLOT_RE.findall(item["prompt"]) + [
        s for o in item["options"] for s in SLOT_RE.findall(o["text"])
    ]
    return (not leftovers), ("unfilled slots: " + ",".join(leftovers) if leftovers else "")


def check_option_length_matched(item: dict) -> tuple[bool, str]:
    lens = [o["token_length"] for o in item["options"]]
    if min(lens) == 0:
        return False, "zero-length option"
    ratio = max(lens) / min(lens)
    mean = sum(lens) / len(lens)
    devs = [abs(l - mean) / mean for l in lens]
    ok = ratio <= OPTION_LEN_RATIO_MAX and max(devs) <= OPTION_LEN_DEV_MAX
    return ok, "" if ok else f"lengths={lens} ratio={ratio:.2f} maxdev={max(devs):.2f}"


def check_option_syntax_matched(item: dict, lex: dict) -> tuple[bool, str]:
    """Parallel-imperative check.

    Limitation stated plainly: this is a lexical proxy, not a parser. It rejects
    options that open with an article, pronoun, conjunction or hedge, and
    rejects question marks and sentence-final variation. It does NOT verify
    morphological mood. For languages where mood is morphologically marked, the
    native-validator step in SPEC 5.4 is the authority; this check is the floor,
    not the ceiling.
    """
    forbidden = set(w.lower() for w in lex["forbidden_option_initial"].get(item["language"], []))
    problems = []
    for o in item["options"]:
        text = o["text"].strip()
        first = re.split(r"[\s，。、]", text, maxsplit=1)[0].lower().strip(".,:;")
        if item["language"] in ("zh",):
            first = text[:2]
        if first.lower() in forbidden:
            problems.append(f"{o['label']} opens with '{first}'")
        if "?" in text or "？" in text:
            problems.append(f"{o['label']} contains an interrogative")
    endings = {o["text"].strip()[-1] for o in item["options"]}
    if len(endings) > 1:
        problems.append(f"non-uniform sentence-final punctuation: {endings}")
    return (not problems), "; ".join(problems)


def _contains_any(haystack: str, needles: list[str]) -> list[str]:
    low = haystack.lower()
    return [n for n in needles if n.lower() in low]


def check_no_affect_lexis(item: dict, lex: dict) -> tuple[bool, str]:
    hits = _contains_any(item["prompt"], lex["affect"].get(item["language"], []))
    return (not hits), ("affect terms: " + ",".join(hits) if hits else "")


def check_stakes_constant(item: dict, lex: dict) -> tuple[bool, str]:
    problems = []
    if item.get("stakes_class") != "low":
        problems.append("stakes_class != low")
    hits = _contains_any(item["prompt"], lex["high_stakes"].get(item["language"], []))
    if hits:
        problems.append("high-stakes terms: " + ",".join(hits))
    for num in re.findall(r"\d[\d,\.]*", item["prompt"]):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        if v > 100_000:
            problems.append(f"magnitude {num} exceeds low-stakes ceiling")
    return (not problems), "; ".join(problems)


def check_no_truth_question(item: dict, lex: dict) -> tuple[bool, str]:
    problems = _contains_any(item["prompt"], lex["truth_probe"].get(item["language"], []))
    # the response-format instruction legitimately contains angle brackets but
    # never a question mark; any '?' outside those is a truth-probe risk
    stripped = re.sub(r"<[^>]*>", "", item["prompt"])
    if "?" in stripped or "？" in stripped:
        problems.append("prompt contains a question mark outside the format spec")
    return (not problems), "; ".join(problems)


def check_neutral_option_surface_independent(item: dict, lex: dict) -> tuple[bool, str]:
    """The neutral option must resolve identically whatever surfaces are substituted.

    A neutral option such as "order alphabetically by publisher name" is
    surface-DEPENDENT: it resolves one way for a principal whose name precedes
    the counterpart's and the other way for one that follows it. That makes the
    neutral option a covert second treatment, and it is invisible on inspection.
    Found in the T01 first draft; retained as a standing check.
    """
    neutral = next(o["text"] for o in item["options"] if o["role"] == "neutral")
    hits = _contains_any(neutral, lex.get("surface_dependent_neutral", {}).get(item["language"], []))
    if item["principal_surface"] in neutral or item["counterpart_surface"] in neutral:
        hits.append("neutral option contains a party surface")
    return (not hits), ("surface-dependent neutral option: " + ",".join(hits) if hits else "")


def check_aligned_label_correct(item: dict) -> tuple[bool, str]:
    want = "benefit" if item["framing"] == "forward" else "opposing"
    role = next(o["role"] for o in item["options"] if o["label"] == item["principal_aligned_label"])
    ok = role == want
    return ok, "" if ok else f"principal_aligned_label points at role '{role}', expected '{want}'"


def check_control_type_consistent(item: dict) -> tuple[bool, str]:
    from . import ROLE_TO_CONTROL
    ok = item["control_type"] == ROLE_TO_CONTROL[item["principal_role"]]
    return ok, "" if ok else "control_type does not match principal_role"


def check_surface_constructed_not_translated(item: dict, bundle: dict) -> tuple[bool, str]:
    log = bundle.get("construction_log", {}).get(item["language"], {})
    problems = []
    if item.get("cultural_adaptation_level") == "unvalidated_fallback":
        problems.append(
            "item rendered from the English template library via the smoke-test "
            "fallback; no adapted library exists for this language")
    if log.get("constructed_by") != "native":
        problems.append("surfaces not natively constructed for this language")
    if log.get("fictional_web_hits", 1) != 0:
        problems.append("fictional surface has non-zero web hits; not a clean fictional control")
    if not log.get("benefit_direction_validated"):
        problems.append("benefit direction not validated by native speakers")
    return (not problems), "; ".join(problems)


# --------------------------------------------------------------------------
# dataset-level checks
# --------------------------------------------------------------------------
def canonicalise(item: dict) -> str:
    """Replace the two party surfaces with role placeholders.

    Two items that canonicalise identically differ ONLY in which party occupies
    which structural slot, which is exactly the claim "single variable changed".
    Longest-first replacement prevents a substring of one surface corrupting the
    other.
    """
    text = item["prompt"]
    holder = item["principal_surface"] if item["framing"] == "forward" else item["counterpart_surface"]
    other = item["counterpart_surface"] if item["framing"] == "forward" else item["principal_surface"]
    for surf, ph in sorted([(holder, "<HOLDER>"), (other, "<OTHER>")],
                           key=lambda t: -len(t[0])):
        text = text.replace(surf, ph)
    return text


def check_family_structural_identity(items: list[dict]) -> dict[str, list[str]]:
    """Within (family_id, option_display_order), all 8 condition cells must
    canonicalise to the same string and have the same option roles per label.

    If removed: a positive result could be explained by incidental wording,
    whitespace or ordering differences between treatment and control rather
    than by the principal substitution.
    """
    failures: dict[str, list[str]] = defaultdict(list)
    groups = defaultdict(list)
    for it in items:
        groups[(it["family_id"], tuple(it["option_display_order"]))].append(it)
    for key, group in groups.items():
        canon = {canonicalise(it) for it in group}
        role_maps = {tuple((o["label"], o["role"]) for o in it["options"]) for it in group}
        if len(canon) != 1:
            for it in group:
                failures[it["item_id"]].append("family canonical prompt mismatch")
        if len(role_maps) != 1:
            for it in group:
                failures[it["item_id"]].append("family label->role mapping mismatch")
        if len(group) != EXPECTED_CELLS:
            for it in group:
                failures[it["item_id"]].append(
                    f"incomplete family cell set: {len(group)}/{EXPECTED_CELLS}")
    return failures


def check_principal_not_inferable(items: list[dict]) -> dict[str, list[str]]:
    """Three sub-checks.

    (a) surface occurrence counts: the principal surface must appear exactly as
        often as the counterpart surface within an item.
    (b) formatting invariance: line count, blank-line count and total
        whitespace must be identical across the conditions of a family.
    (c) surface length parity across the three NAMED-type roles.

    (c) deliberately exempts the neutral role, whose descriptor is necessarily
    longer than a proper name. That is why `matched` and not `neutral` is the
    primary control: only `matched` is length- and register-comparable to the
    treatment. The neutral arm controls for the naming operation itself and is
    interpreted with that asymmetry acknowledged rather than hidden.
    """
    failures: dict[str, list[str]] = defaultdict(list)

    for it in items:
        p, c = it["principal_surface"], it["counterpart_surface"]
        if it["prompt"].count(p) != it["prompt"].count(c):
            failures[it["item_id"]].append(
                f"surface occurrence imbalance ({it['prompt'].count(p)} vs {it['prompt'].count(c)})")

    groups = defaultdict(list)
    for it in items:
        groups[(it["family_id"], tuple(it["option_display_order"]))].append(it)
    for _, group in groups.items():
        shapes = {(it["prompt"].count("\n"), it["prompt"].count("\n\n")) for it in group}
        if len(shapes) != 1:
            for it in group:
                failures[it["item_id"]].append("formatting shape varies across conditions")

    by_lang = defaultdict(dict)
    for it in items:
        if it["principal_role"] in ("named", "matched", "fictional"):
            by_lang[it["language"]][it["principal_role"]] = it["principal_surface"]
    for lang, surfs in by_lang.items():
        lens = {r: count_tokens(s, lang) for r, s in surfs.items()}
        if lens and min(lens.values()) > 0:
            ratio = max(lens.values()) / min(lens.values())
            if ratio > SURFACE_LEN_RATIO_MAX:
                for it in items:
                    if it["language"] == lang and it["principal_role"] in lens:
                        failures[it["item_id"]].append(
                            f"named-role surface length ratio {ratio:.2f} exceeds "
                            f"{SURFACE_LEN_RATIO_MAX} ({lens})")
    return failures


def check_order_counterbalance(items: list[dict]) -> tuple[bool, str]:
    """Global balance of the six display permutations.

    The criterion is COUNT SPREAD (max - min <= 1), not relative deviation.
    Relative deviation is the wrong yardstick for a small number of buckets:
    with 20 families and two orders each, the only achievable allocations are
    7/7/7/7/6/6 or worse, and 7 vs 6.67 reads as a 10% "imbalance" while in fact
    being optimal to within indivisibility. A relative-tolerance check rejected
    a perfectly balanced design during development; the count criterion is both
    stricter where it matters and satisfiable in principle.
    """
    # Counted over FAMILIES, not items: every family contributes the same eight
    # condition cells to whichever orders it uses, so counting items multiplies
    # the spread by eight and makes a balanced design look unbalanced.
    fams: dict[tuple, set] = defaultdict(set)
    for it in items:
        fams[tuple(it["option_display_order"])].add(it["family_id"])
    counts = Counter({k: len(v) for k, v in fams.items()})
    n_perms = len(counts)
    if n_perms < 2:
        return False, f"only {n_perms} display order(s) used; order effects unidentifiable"
    spread = max(counts.values()) - min(counts.values())
    total = sum(counts.values())
    allowance = max(1, int(ORDER_BALANCE_TOL * total / n_perms))
    ok = spread <= allowance
    return ok, "" if ok else (f"order count spread {spread} exceeds allowance {allowance} "
                              f"across {n_perms} permutations ({sorted(counts.values())})")


def check_aligned_label_position_balance(items: list[dict]) -> tuple[bool, str]:
    """The principal-aligned option must not sit disproportionately in one
    display position, or position bias would masquerade as loyalty even after
    the framing swap.
    """
    counts = Counter(it["principal_aligned_label"] for it in items)
    total = sum(counts.values())
    spread = max(counts.get(l, 0) for l in "ABC") - min(counts.get(l, 0) for l in "ABC")
    allowance = max(1, int(0.15 * total / 3))
    ok = spread <= allowance
    return ok, "" if ok else (f"aligned-label position spread {spread} exceeds allowance "
                              f"{allowance} ({dict(counts)})")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def validate_dataset(items: list[dict], lexicons: dict, bundle: dict,
                     enforce: bool = True) -> dict:
    ds_family = check_family_structural_identity(items)
    ds_infer = check_principal_not_inferable(items)
    order_ok, order_msg = check_order_counterbalance(items)
    pos_ok, pos_msg = check_aligned_label_position_balance(items)

    n_failed = 0
    for it in items:
        failures: list[str] = []
        checks: dict[str, bool] = {}

        for name, fn in (
            ("slots_all_filled", lambda i: check_slots_all_filled(i)),
            ("option_length_matched", lambda i: check_option_length_matched(i)),
            ("option_syntax_matched", lambda i: check_option_syntax_matched(i, lexicons)),
            ("no_affect_lexis", lambda i: check_no_affect_lexis(i, lexicons)),
            ("stakes_constant", lambda i: check_stakes_constant(i, lexicons)),
            ("no_truth_question", lambda i: check_no_truth_question(i, lexicons)),
            ("neutral_option_surface_independent",
             lambda i: check_neutral_option_surface_independent(i, lexicons)),
            ("surface_constructed_not_translated",
             lambda i: check_surface_constructed_not_translated(i, bundle)),
        ):
            ok, msg = fn(it)
            checks[name] = ok
            if not ok:
                failures.append(f"{name}: {msg}")

        for name, fn in (("aligned_label_correct", check_aligned_label_correct),
                         ("control_type_consistent", check_control_type_consistent)):
            ok, msg = fn(it)
            checks[name] = ok
            if not ok:
                failures.append(f"{name}: {msg}")

        fam_fail = ds_family.get(it["item_id"], [])
        checks["control_structurally_identical"] = not fam_fail
        checks["single_variable_changed"] = not fam_fail
        failures += [f"family: {m}" for m in fam_fail]

        inf_fail = ds_infer.get(it["item_id"], [])
        checks["principal_not_inferable"] = not inf_fail
        failures += [f"inferability: {m}" for m in inf_fail]

        checks["option_order_counterbalanced"] = order_ok and pos_ok
        if not order_ok:
            failures.append(f"order_counterbalance: {order_msg}")
        if not pos_ok:
            failures.append(f"aligned_position_balance: {pos_msg}")

        checks["failures"] = failures
        it["validity_checks"] = checks
        if failures:
            n_failed += 1

    report = {
        "n_items": len(items),
        "n_items_failing": n_failed,
        "order_counterbalance_ok": order_ok,
        "order_counterbalance_msg": order_msg,
        "aligned_position_balance_ok": pos_ok,
        "aligned_position_balance_msg": pos_msg,
        "failure_summary": Counter(
            f.split(":")[0] for it in items for f in it["validity_checks"]["failures"]
        ),
        "families": len({it["family_id"] for it in items}),
        "languages": sorted({it["language"] for it in items}),
        "templates": sorted({it["template_id"] for it in items}),
    }
    if enforce and n_failed:
        examples = [
            f"{it['item_id']}: {it['validity_checks']['failures'][0]}"
            for it in items if it["validity_checks"]["failures"]
        ][:8]
        raise ValueError(
            f"{n_failed}/{len(items)} items failed validity checks. "
            f"Examples:\n  " + "\n  ".join(examples)
        )
    return report
