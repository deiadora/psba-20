"""PSBA-20: Principal-Specific Behavioural Auditing Benchmark.

A generator, not a dataset. See SPEC.md.
"""

__version__ = "1.0.0"
GENERATOR_VERSION = "1.0.0"

PRINCIPAL_ROLES = ("named", "matched", "neutral", "fictional")
ROLE_TO_CONTROL = {
    "named": "treatment",
    "matched": "matched_principal",
    "neutral": "neutral_entity",
    "fictional": "fictional_principal",
}
FRAMINGS = ("forward", "reversed")
OPTION_ROLES = ("benefit", "neutral", "opposing")
