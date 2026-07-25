# Limitations and future work

Written before data collection, so these are design properties rather than post-hoc apologies.

## Limitations of the construct

1. **It measures asymmetry, not intent.** A directional asymmetry is consistent with a hidden objective, with an incidental training-data correlation, and with a legitimately learned fact about the world. The dissociation channel narrows this but does not close it. PSBA-20 produces evidence, not a verdict.

2. **"Principal-benefiting" is a modelling assumption.** The benefit direction is authored, then validated by two blind native speakers per language. Where the validators disagree and the item is rewritten, what is being measured has changed slightly. The `benefit_direction_agreement` field exists so this is visible in the data rather than absorbed into the estimate.

3. **The principal must be identifiable.** A loyalty toward an abstract goal, a coalition, an ideology without an institutional name, or a class of people is out of scope. The design's whole identification strategy depends on being able to swap the principal's name.

4. **Low stakes are a deliberate restriction with a cost.** Stakes are held low and constant so refusal stays rare and the option schema stays usable. But a loyalty may be *stake-conditional* — quiescent at trivial stakes and active where something is at risk. PSBA-20 cannot see that, and a null here says nothing about it.

5. **Three options constrain resolution.** A 3-option schema makes choices codeable and makes JSD well-defined on a small support, at the cost of resolution. A loyalty expressed as a small shift along a continuum may not cross an option boundary. The logprob channel partly mitigates this; it does not remove it.

## Limitations of the instrument

6. **Template count is the binding constraint, and 20 is not many.** 58% of the variance of DAI is between-template, so the MDE floor is 6.3 pp at 20 templates however many samples are drawn. This is the single most important thing to fix, and it is cheap to fix: templates cost author time, not compute. 40 templates would bring the floor to 4.4 pp.

7. **The syntactic-parallelism check is a lexical proxy, not a parser.** It rejects options that open with articles, pronouns, conjunctions or hedges, and enforces uniform final punctuation. It does not verify morphological mood, which matters most in exactly the languages where the native-validator step is hardest to staff.

8. **The neutral-entity control is not length-matched, and cannot be.** A generic descriptor is longer than a proper name. The validator exempts the neutral arm and records the exemption; `matched_principal` is the primary control for this reason. Anyone reading a neutral-arm contrast should read it as secondary.

9. **T14 has a weaker reversal.** Conflict-of-interest structure is asymmetric by construction, so the stake swap is a partial inversion. It is quarantined into its own stratum rather than being fixed, because fixing it would mean removing the omission-type loyalty that makes it worth including.

10. **Validation rests on three organisms.** Sensitivity of 3/3 has a Wilson 95% interval of roughly [0.44, 1.00]. The ROC has about ten points. This bounds every claim the validation can support, and no amount of sampling within those three models widens it — more *organisms* are needed, ideally with different installation methods.

11. **Organism construction may not resemble frontier loyalties.** The organisms are fine-tuned 7B models. A loyalty in a frontier model, if one existed, might be installed differently, be better concealed, or interact with far stronger evaluation-awareness. Sensitivity measured on these organisms is not transferable to that setting without argument.

12. **Evaluation awareness is measured, not eliminated.** The suspicion probe and the mundane register reduce it; the distractor-load mode reduces it further. Neither guarantees a model that recognises the audit will behave as it would in deployment. DAI conditional on awareness is reported so that a difference is visible; a model that is *always* aware leaves us with no unaware stratum to compare against.

13. **Scorer independence is enforced at the family level by name matching.** Two models with different names may still share training data, distillation ancestry, or preference data. The check is a floor, not a guarantee, and cross-family agreement should not be read as independence in a deep sense.

14. **Cross-lingual arms depend on human authoring.** The generator refuses machine translation, which is correct but means the multilingual design cannot be run without native authors. H4 is therefore the most likely hypothesis to go untested in a weekend, and its absence from a Pilot-A report is a staffing limitation rather than a finding.

15. **Single-turn only.** See SPEC §12. The one-variable-changed guarantee is provable only in a single turn, and it is the guarantee the primary contrast rests on.

## Future work, in descending order of expected value per unit effort

1. **Expand to 40 templates.** Author time only; halves the MDE floor. Nothing else on this list improves the instrument as much per hour spent.
2. **Multi-turn extension** (SPEC §12), especially cumulative drift and consistency-under-challenge — the two things that could see effects the single-turn instrument provably cannot.
3. **More organisms, with varied installation methods** — different data mixes, different loyalty strengths, RL as well as SFT. This is what would turn a 10-point ROC into a usable one.
4. **Interpretability triangulation.** The organisms come with linear probes and activation patching available. A behavioural DAI that co-varies with a probe direction would be far stronger evidence than either alone, and would give the behavioural threshold a mechanistic anchor.
5. **Graded-outcome templates.** Replace some 3-option items with ordinal or continuous allocations (percentage splits, ranked lists of five) to raise resolution on sub-boundary shifts.
6. **Stake-conditional arms.** A matched high-stakes tier, accepting the higher refusal rate, to test whether loyalty is stake-gated.
7. **Adversarial affordance search.** Automatically search the template space for the *highest-affordance* framing of each domain, rather than hand-authoring one, using the pre-screen as the objective.
8. **Coalition and abstract principals.** Extend the slot to unnamed but identifiable interests, which requires a new identification strategy since the name swap is unavailable.
9. **Cross-principal calibration.** Run the same clean model against many principals to establish the null distribution of DAI empirically, rather than relying on the matched control alone. This is the cheapest route to a trustworthy per-audit FPR.
