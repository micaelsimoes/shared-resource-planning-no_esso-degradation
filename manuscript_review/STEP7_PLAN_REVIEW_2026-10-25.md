# Review of the Expert's Step 7 simulation plan

25 October 2026

This note assesses STEP7_SIMULATION_PLAN_2026-10-25.md and recommends responses to D1–D5. It is a proposed response for the author to forward, not a record of the author's final decisions or an instruction to launch simulations.

## Overall assessment

The proposed package is well focused. Support W182's re-audit and evidence inventory, E3 as the first inexpensive simulation, a staged E2 pilot, E1 in an economically active setting, and E6 from existing records. Deferring E5 and narrowing timing claims is defensible. E4 can remain deferred if the empirical claims are qualified accurately.

The main correction concerns E1: comparing a new no-ageing search against the historical aged search would mix ageing effects with code, numerical-setting, seed-cache and search-protocol effects. For a controlled planning comparison, rerun both arms under a common final implementation and protocol.

The manuscript source review covered edd7a63, but it did not establish code-to-equation equivalence. W182 should verify the newly exposed equations and current claims against the implementation. Newly printed equations need checking; their presence does not by itself show that the implemented physical model changed.

## Recommended responses to D1–D5

| Decision | Recommended response | Qualification |
|---|---|---|
| D1 — Package | E3, staged E2, matched E1 searches at 2×, the baseline-price neighbourhood screen, and E6 | Include a bounded alternative-start stage if W182 finds it practical. Re-estimate cost after matching both E1 arms. |
| D2 — E1 primary price | Accept 2× as the primary setting | State that it is a preselected setting where storage already has value. Retain and report the baseline screen, including null or unresolved results. |
| D3 — Aged E1 arm | Rerun on the same final code and with the same search protocol as the no-ageing arm | The existing aged plan may be an additional reference or a common seed candidate. It should not be the only controlled comparator. |
| D4 — Stochastic contribution | Defer E4-lite and narrow the demonstrated case-study claims | Retain “two-stage stochastic” where it accurately describes the general formulation. Do not imply that a multi-scenario search or a benefit of stochastic selection was demonstrated. |
| D5 — Order and manuscript | W182 first; then E3 and E2; resolve numerical problems before E1; extract E6 alongside | Regenerate the marked manuscript after final text and numerical updates. Follow the established execution workflow. |

These are recommendations for the author's decisions. No additional approval requirements or execution permissions are created by this note.

## 1. E1 needs matching beyond the code version

The current manuscript describes the historical doubled-price result as a variant-A search followed by a variant-B continuation, with existing evaluations seeding the caches. The proposed new no-ageing arm is a variant-A search from zero with a 20-new-evaluation budget.

Using the same code head alone would not make these procedures comparable. Please specify the same:

- Search variant or sequence of variants, poll rules and limits.
- Initial candidate/seed geometry and incumbent-selection rule.
- Evaluation budget and treatment of cached points.
- Production stopping settings and final evaluation criteria.

Evaluate the same seed investments under each ageing model; do not reuse one model's costs in the other. Decide the common seed set before the searches. If a shared A-then-B protocol is too expensive, a shared bounded variant-A protocol is acceptable, with its limits reported.

Re-evaluate the returned plans under both ageing models as proposed. The key contrast is F_A(x_N) − F_A(x_A), with both values assessed under baseline ageing and a consistent protocol.

The historical doubled-price incumbent remains uncertified by the interface-gap criterion. It should not serve as an unquestioned optimal aged reference. If the new aged result or cross-evaluation remains uncertified, apply the declared treatment and report any unresolved difference.

A negative contrast would indicate that the no-ageing search happened to find a plan performing better under the aged model. It would not establish that ignoring ageing is generally preferable. A positive contrast would likewise be evidence about the returned plans, not a global regret bound.

## 2. The baseline-price test should be described as a bounded screen

Accept the doubled-price case as primary. It provides a legitimate test where storage economics are active; this does not require an expectation that the resulting plans differ.

The reference unit's near-zero no-ageing margin and the observed fitted relationships motivate a limited baseline screen. They do not establish the outcome at all nodes, years or joint investments.

Please define precisely what the proposed “first poll” means. In the printed algorithm, a search starting at poll size four is not the same operation as evaluating the fourteen admissible unit neighbours around zero.

If the intention is to test those fourteen points, label the result a unit-neighbourhood screen. If none improves beyond the declared threshold, report that scope; do not claim a completed planning search or a globally preferred zero investment.

## 3. E2: distinguish the penalty pilot from start sensitivity

The eight-evaluation paired penalty pilot is a reasonable first stage. Its margins should be constructed from each configuration's own no-storage/reference pair.

Please adjust the proposed verdict rule:

- Apply the acceptance and comparison criterion appropriate to each new pair's own ranges, gaps and stopping statuses.
- Use the original baseline threshold as a descriptive reference only; it should not automatically determine verdicts for new or uncertified pairs.
- Report the range of paired margins across configurations separately from the numerical acceptance criterion within a pair.

The statement that local-solution differences will cancel because both arms share a configuration is a hypothesis to test. Shared settings do not guarantee cancellation.

W182 should determine whether a modest coordinated-start perturbation is practical. If so, include the proposed four additional evaluations. If it is deferred, explicitly describe E2 as sensitivity to initial penalties; it does not establish independence from starts.

Deferring a full centralized model is proportionate. A small equivalent-instance check could remain optional; a new all-block centralized implementation is not part of the minimum package.

## 4. E3: retain the experiment, separate predictions from evidence

One coordinated no-storage evaluation with matched NRF restrictions is a useful first addition. Verify final feasibility on both interface copies and retain consistent signs and accounting.

The fact that only four interface-hours have reverse flow does not bound the cost of prohibiting it. Flow magnitudes, representative-day/year weights, prices and binding constraints matter. W182 can inventory these quantities without new solves if the records permit.

Keep the predicted near-zero NRF cost as a hypothesis, not a reason to preserve the existing headline or interpretation. Recompute both conditional differences and their percentage denominators from the resulting records.

Also clarify comparison uncertainty. A three-start spread in a static NLP benchmark is not the same quantity as the consensus-gap/settling bar of an uncertified ADMM evaluation. Define an appropriate empirical reporting rule for that mixed comparison; a zero start spread is not proof of exactness.

The two reported differences should retain their conditional interpretation:

- Coordination under NRF: Q(P_NRF) − Q(C_NRF).
- Permitting reverse flow under coordination: Q(C_NRF) − Q(C_OPEN).

Interpret dispatch mechanisms from the new component and operating results. The explanation involving interface marginal values should not be held fixed irrespective of the observed decomposition.

## 5. E4 and E5: narrow the right claims

### E4

Deferring E4-lite is reasonable if the paper clearly distinguishes the general stochastic formulation from the experiments: the planning searches use a single operating scenario; the nine-scenario study evaluates fixed plans.

The current title does not contain “stochastic”. Retain technically accurate terminology in the formulation; qualify contribution and abstract statements that could imply demonstrated multi-scenario investment selection. No title change is required on this particular ground.

If E4-lite is later pursued, check its evaluation count. Three candidates under two configurations yield six cells. If two joint-scenario cells are reusable, that leaves three new expected-profile cells and one new joint cell: four new evaluations in total. The plan currently states four new expected-profile cells plus one joint cell; identify the extra cell if intentional.

The shortlist would establish candidate-ranking sensitivity, not the exact value of a stochastic solution.

### E5

Support deferring terminal-value-aware optimisation, with an explicit narrower claim:

- Salvage is a post-evaluation sensitivity.
- Dispatch and investment selection do not internalise it.
- The analysis does not establish the economically preferred installation year under a terminal-value-aware objective.

Report this as a partial response to the reviewer request, with a scope rationale. The fact that changing the objective requires reruns explains the effort; it does not demonstrate that the current post-evaluation treatment is equivalent.

## 6. E6, provenance and execution estimates

Support a zero-solve export from the final accepted reference-unit records. A day chosen for maximum contribution is a valid illustrative example if explicitly labelled; it is not a typical-day estimate. Show cost-component changes, units, time weights and capacity limits consistently.

Some schedules are described as uncommitted but hashed. Verify and preserve the exact inputs needed for reproducibility; do not describe all source material as committed until that is established.

Reconcile the E1 budget before freezing the matrix: a 20-new-evaluation search cap should not become 25 new search evaluations without an explicit extension. Count seed evaluations, repeated/final evaluations and cross-evaluations separately.

Re-estimate the campaign after the aged-arm rerun and any start-sensitivity stage. Runtime and concurrency figures are provisional. If search evaluations run concurrently, verify peak resource use and preserve deterministic candidate selection and decision ordering; do not assume serial per-evaluation timings scale directly.

## Expected next handover

W182 should return the current code/equation audit and evidence inventory, together with findings affecting these recommendations. The Expert can then produce the final case matrix with:

1. Matched E1 search protocols and reusable records.
2. The E2 pilot and a decision on a bounded alternative-start stage.
3. E3's comparison/status rules.
4. Corrected counts, runtime estimates and explicit finite limits.
5. The exact manuscript claims supported by each possible outcome.

No manuscript or campaign-spec edits, simulation launches, commits or messages to other chats were made in preparing this review.
