# P5.15 Step 3.5 — polish-gap gate: pre-launch note

Planner note, written before the full run. Spec: `data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`,
item `4_step_3_5`. Script: `p515_s40_polish_gap.py` (`9422ba9b`); preparation report `WORKER_REPORT_S40_POLISH_GAP_PREP.md`.

**What runs.** The case-file configuration, which since `fb3de341` encodes the oracle (arm D), runs to Boyd certification
in-process. It must reproduce D bitwise — certification at cycle 139, `gross_operational_cost` 650,966,975.2943751, every
numeric trajectory field — or the script stops before polishing. That reproduction is also the τ = 0 determinism check
flagged as unverified in `P5_15_ADDENDUM21_EXPERT_REPORT.md` §3. Then all 48 network blocks are re-solved with the
unscaled base objective at fixed consensus (P5.7's design via `p56a_oracle`: interface quantities fixed at the midpoint
of the two sides' achieved values; the shared-ESS channel at its consensus variable z).

**Gate.** |Δ recourse| / recourse < 0.1 % on `gross_operational_cost` (settlement excluded, as certified). Not evaluated
if any block fails to solve.

**Expectation (spec v11):** the gate passes.

**Recorded risk, before the run.** In the 2-cycle smoke test, 13 of 48 polish solves were locally infeasible at fixed
consensus — all 12 TSO blocks and one DSO block. A 2-cycle state is far from converged (consensus residuals are 10²–10³
times tolerance), so this does not predict the certified point. But the programme's history records exact-consensus polish
breaking at incompletely converged states (`REVISION_CONTEXT.md` P5.7–P5.12 entries), and at certification the PF channel
closes near its tolerance. **If any block is infeasible at the certified point, the gate is not evaluated: the Planner
reports the failing blocks and stops for review**, per Step 3.5's own rule that the ρ/σ ratio is the lever and is not
touched without review.
