# P5.15 Step 4 — warm-continuation validation: design (not run)

**Planner design note, 2026-09-16.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 18 ("design only, zero solves:
draft a warm-continuation validation for Step 4 … Do not run it unless the Addendum 17 gate fails").
**Status: DESIGN ONLY. No run is authorized by this note.** It becomes a frozen spec only if the Addendum 17 gate
fails and the author authorizes it.

## 1. Why this validation exists

Step 4 evaluates many neighbouring investment candidates. A cold, Boyd-certified evaluation at C\* took **477 cycles
(≈4 h 50 min serially)**. If Addendum 17's primal-and-dual initialization does not make cold certification fast, the
outer method needs a cheaper evaluation. The obvious candidate is **warm continuation**: start a neighbour's ADMM from
the certified terminal state of an already-evaluated candidate.

The programme's own history says warm starts can mislead:
- **AB1:** the same balancing rule settled at ρ **131.7 warm** versus **3.47 cold**, a factor of 38; the adaptive
  operating point is a property of the trajectory it is placed on.
- **Track C1:** a warm cell stopped at 99.3 % of its bound, a cold cell at 58.0 %. Warm agreement was once read as a
  result before the stopping behaviour was checked.
- **`REVISION_CONTEXT.md`:** candidate evaluations warm-started from a **non-converged** state returned local results.
- **Addendum 16–17:** at C\* the storage direction is so flat that a certified stop pins cost but not EFC from above.

So warm continuation is admissible only if it is shown to reproduce cold certification **on the quantities Step 4
ranks by**, with a standing audit that would catch drift.

## 2. Candidates

Three adjacent rungs of the committed capacity ladder (`data/SRP1/Results/P514L/`, constant E/P ratio 4, uniform across
nodes 5/7/9, invested 2025):

| label | s (MVA) | e (MWh) | role |
|---|---|---|---|
| L | 0.9375 | 3.75 | lower neighbour |
| C\* | 0.96875 | 3.875 | certified reference (run 1) |
| U | 1.0 | 4.0 | upper neighbour |

The ladder runs were first-stage feasibility checks only, not ADMM evaluations. Before any run, confirm that all three
rungs are first-stage feasible under the current formulation (zero or bounded solves) and that the SoH floor stays
slack at L and U, by rerunning the Z2 price-taker check, which is LP only.

## 3. Arms

Every arm uses the **production oracle as it stands after the Addendum 17 gate**: case-file ρ defaults, balancing on,
freeze after 10 unchanged cycles, and the Boyd stop with 3 consecutive cycles. Every stop is **derived from the
trajectory**, not from the harness `stopped_by` field.

| arm | candidate | start | purpose |
|---|---|---|---|
| **C-L** | L | cold (the oracle's own initialization) | reference for L |
| **C-U** | U | cold | reference for U |
| **W-L←C\*** | L | warm from C\*'s certified terminal state | continuation downward |
| **W-U←C\*** | U | warm from C\*'s certified terminal state | continuation upward |
| **W-C\*←L** | C\* | warm from C-L's certified terminal state | history-dependence test at the certified point |

**"Warm state" is defined completely and recorded by hash:**
- the storage and interface consensus z;
- all agent copies;
- every dual (V, PF, ESS, per agent);
- the TSO proximal centres;
- each channel's ρ and γ **and freeze state**;
- IPOPT warm-start multipliers where production carries them.

The candidate-dependent capacities (s, e, the price-taker bound) are **recomputed** for the target candidate, never
inherited. **A warm start from a non-certified source is not permitted.**

## 4. Agreement criteria (pre-registered)

A warm arm **agrees** with its cold reference iff all of the following hold:
1. **Both certified and settled:** Boyd stop within the cap, all three channel terminal ratios **< 0.9**, and no ρ clamp.
   A stop at ≥ 0.9 on any channel is *stopped, not settled* and fails agreement outright.
2. **Cost within the rule-nine bar:** |Q_warm − Q_cold| ≤ terminal step (warm) + terminal step (cold). This is valid
   only because criterion 1 requires both runs to have settled.
3. **Ranking preserved:** sign(Q(L) − Q(C\*)) and sign(Q(U) − Q(C\*)) are identical under warm and cold, and each
   difference exceeds its combined bar. **A ranking whose difference is inside the bar is reported as indeterminate,
   not as agreement.**
4. **History independence at C\*:** W-C\*←L reproduces run 1's cost within the bar. It must also end with ρ values and
   a freeze state that do not depend on the source beyond what balancing re-selects, reported, not gated.
5. **Storage reported, not gated:** EFC/day is compared under the §6 claims-to-avoid discipline. It is a one-sided
   statement unless the Addendum 17 gate made EFC a certified quantity. A warm/cold EFC difference is never read as a
   fixed-point difference.

**Warm continuation is admitted for Step 4 only if every warm arm agrees on criteria 1–4.**

## 5. Cold-audit cadence in Step 4 (if admitted)

- **Always cold:** the first evaluation of any new incumbent, the final incumbent, and the runner-up.
- **Periodic:** every **5th** warm evaluation is repeated cold. Any disagreement on criteria 1–3 invalidates the warm
  chain back to the last agreeing audit, and those candidates are re-evaluated cold.
- **Triggered:** any warm evaluation whose cost difference to its source is within twice its bar (a near-tie), or whose
  source is more than one ladder step away, is evaluated cold.
- **Warm chains never extend from a warm evaluation that has not been cold-audited.**

## 6. Cost and sequencing

- **Arms:** four cold-or-warm arms plus one history test = 5 certifications.
- **Serial cost:** at ~39 s per cycle, and run 1's 477 cycles as an upper-bound cold length, up to ~24 h. Warm arms are
  expected to be much shorter, which is precisely what is under test.
- **With Step 3.6 parallelism:** at the targeted ~8–10 s per cycle, about 5–6 h.
- **Ordering:** run only after Step 3.6's gate, never concurrently with any numerical gate, one run at a time, attached,
  each launched only after its frozen spec and evaluator are committed.

## 7. Predictions to record in the frozen spec

- Warm arms certify in materially fewer cycles than cold. The size of that saving is the quantity of interest, not
  asserted.
- Cost agrees within the bar where both settle.
- The chief risk is **criterion 1**: a warm start inherits a frozen ρ, which may leave a channel stopped near its
  threshold rather than settled, as C1's warm cell did at 99.3 %.
- Storage EFC may differ between warm and cold by more than cost does (the flat-direction finding). That is expected
  and is not a disagreement under §4.
