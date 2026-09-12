# P5.14-L — capacity ladder: the largest plan this oracle can evaluate

**Seven rungs, ESSO-initialization stage only, ADMM never started. 358 solves total,
~5 minutes. Guards clean.** Frozen spec
`data/SRP1/Results/P514L/frozen_ladder_v1_c1ea6393.json` (`c1ea6393`), with both go/no-go
branches and the prime suspect predeclared before any rung ran.

## The governing fact this serves

**The oracle cannot evaluate a plan large enough to exhibit the effect the paper claims.**
Every stabilization finding — `Q` well defined, ADMM converging, one local failure in
1,095 — was established in a regime where the storage is economically negligible. At the
first capacity where it is not, the evaluation aborts before ADMM starts.

One sub-result separates cleanly: **the cost model transferred and the reliability model
did not.** Cell 2 took 3,621 solves against 3,519 predicted — **2.9% over**. The 1-in-1,095
failure rate became **1-in-3**.

## 1. The ladder

| `s` (MVA/node) | `e` (MWh/node) | system-wide | outcome | solves |
|---|---|---|---|---|
| 0.25 | 1.00 | 0.75 / 3.0 | **PASS** | 51 |
| 0.50 | 2.00 | 1.50 / 6.0 | **PASS** | 51 |
| 0.75 | 3.00 | 2.25 / 9.0 | **PASS** | 51 |
| 0.875 | 3.50 | 2.63 / 10.5 | **PASS** | 51 |
| 0.9375 | 3.75 | 2.81 / 11.25 | **PASS** | 51 |
| **0.96875** | **3.875** | **2.91 / 11.6** | **PASS** | 51 |
| 1.00 | 4.00 | 3.0 / 12.0 | **FAIL — node 7** | 52 |

**A clean, monotone threshold.** Six consecutive passes and a single failure, with the
boundary bracketed to within **3.2%**: between 0.96875 and 1.00 MVA per node.

By the predeclared rule, a clean threshold is **a finding about the plan** — that network
(node 7, `case33_2`) cannot host 4.00 MWh at ratio 4, but can host 3.875 MWh. That is real
engineering information, not a solver artifact.

**Largest evaluable capacity established: `C* = 0.96875 MVA / 3.875 MWh` per node.**

## 2. The predeclared prime suspect is FALSIFIED

The hypothesis was the aggregate complementarity row
(`es_pch_hat_agg * es_pdch_hat_agg <= 1e-4`, `shared_energy_storage_data.py:618-620`),
which carries **no slack**, unlike the cohort rows at `:569`.

Terminal violations at node 7, `s = 1.00`, read at the terminal point via a declared
diagnostic re-solve:

| component | max violation |
|---|---|
| `rated_s_capacity_unit` | **6.845e-05** |
| `rated_s_capacity` | 2.282e-05 |
| **`energy_storage_operation_agg`** (the suspect) | **1e-08** |
| `energy_storage_complementarity` (cohort rows) | absent |
| all others | 0.0 |

The aggregate complementarity row is **satisfied to 1e-08**. The spec predeclared that this
falsifies the hypothesis, and it does. The dominant `6.845e-05` matches IPOPT's reported
constraint violation exactly, so the diagnostic is reading the right point.

**What carries the violation instead are two pure definitional identities** —
`es_s_rated_per_unit[y_inv,y] == es_s_investment[y_inv]` (`:442`) and
`es_s_rated[y] == sum(es_s_rated_per_unit)` (`:454`). Those cannot be infeasible on their
own: they are linear equalities in otherwise free variables. Their violation is where
IPOPT's restoration phase left residual, not a physical conflict.

So the two pieces of evidence are **compatible rather than contradictory**: a genuine
feasibility boundary near 1.00 MVA (the sharp threshold), with the unclosable residual
reported on definitional rows (the restoration artifact). What the ladder rules out is the
*specific* hypothesis that the unslacked aggregate complementarity row is the binding
obstruction.

## 3. The limit this result does NOT have

**The ladder tested the initialization stage only.** `C*` is the largest capacity whose
ESSO subproblems solve at initialization — **not** the largest that completes a full ADMM
evaluation. P5.12 established that a trajectory can initialize cleanly and fail later
(cycle 21). A full evaluation at `C*` remains unproven.

Also scope-limited to: ratio 4, single cohort invested 2025, `rel` 1e-4, `rho_pf` 300
adaptive, C3 active, this candidate shape. The threshold is granular to 3.2%.

## 4. The go/no-go, with what remains to decide it

The bounded error bar at `rel` 1e-4 is twice the tolerance, ~**164,000**, or **0.02%** of
an ~8.2e8 objective. The paper's claimed 2.01% incremental storage benefit is ~16.6e6 —
**about 100x the bounded bar**.

So **resolution is not the binding constraint at `C*`**: an effect would have to be under
1% of the paper's claimed magnitude to be unresolvable. What is not yet known is the
*actual* effect a plan at `C*` produces, and that needs **one cold evaluation at `C*`
paired against cell 2** — roughly 3,600 solves, about 35 minutes.

- **GO** if that effect sits comfortably above 164,000: the campaign proceeds at `C*`, and
  the paper's comparison is recoverable **at reduced scale** — which must be stated as
  reduced scale, never as the paper's plan.
- **NO-GO** if it does not: this tool as it stands cannot demonstrate the paper's numerical
  claims, and that goes to the manuscript rather than onto the blocker list.

**Not run. Reporting and stopping.**
