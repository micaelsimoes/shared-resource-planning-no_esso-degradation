# P5.15 Addendum 22 — handoff to the External Expert and the Author

**Planner report, 2026-09-18.** Self-contained; continues `P5_15_ADDENDUM21_EXPERT_REPORT.md`. Item-by-item detail:
`P5_15_S40_STEP3_CLOSURE_REPORT.md` (`96e9848d`).

- **Instance:** C\* (0.96875 MVA / 3.875 MWh per node 5/7/9, invested 2025).
- **Oracle:** arm D — certified at cycle 139, cost 650,966,975.2943751, bar 7,898.63.
- **Cost convention:** `gross_operational_cost`.

## 1. Verdict

**Step 3 did not close.** The Step 3.5 polish gap could not be evaluated at the oracle's certified point: 17 of 48 blocks
fail to solve at fixed consensus. Everything before it in Addendum 22's order is done. Three findings bear directly on
decisions Addendum 22 took:
- the node 7 congestion-relief reading is **not supported** by the data (§4);
- the cost differences between configurations are **determinate**, not stopping slack (§5);
- the persistent-worker path **does not deliver** its projected speed-up (§6).

| item | status | commit |
|---|---|---|
| spec v11, written before any work | done | `d2edf8b6` |
| (1a) cost decomposition | done — §5 | `11075352`, `d73a2b22`, `be80fe99` |
| (1b) node 7 cross-check and result table | done — §4 | `e4db9033`, `d73a2b22`, `be80fe99` |
| (2) clone replacement integrated; bitwise gate | **passed** | `b4ca1466`…`3d00e61f`, `8214be0d` |
| (2) 10-cycle timing re-measurement | **screening passes**, 97.0% | `9428c0b7`, `872ff8e0` |
| (2) persistent workers | built; **gate fails** on one non-numerical field; ~1.1× measured | `c6d60857`…`d8414fb7` |
| (3) D's configuration into the case file (Track B) | **done**; reproduces D over 139 cycles | `fb3de341`, `bc01238e` |
| (4) Step 3.5 polish gap | **not evaluated** — §2 | `51a5fb48`, `57d523d0` |
| (5) `REVISION_CONTEXT.md` head; Step 3 closed | not done | — |
| (6) Step 3.7 Anderson acceleration | held — §7 | — |

## 2. Step 3.5 — the gate cannot be evaluated as specified

### 2.1 What ran

The case-file configuration ran to certification in-process. Then every network block was re-solved with the unscaled
base objective at fixed consensus, following P5.7's design via `p56a_oracle`: interface quantities are fixed at the
midpoint of the two sides' achieved values, and the storage channel at its consensus variable.

### 2.2 Reproduction: passed twice — τ = 0 determinism confirmed

Both full runs reproduced arm D **bitwise** over the entire certified run: certification at cycle 139, cost
650,966,975.2943751, all 139 trajectory rows, 7,179 solves, 38 recovered failures. Addendum 21's handoff had flagged this
determinism as unverified; it is now established.

The first run stopped before polishing on one provenance field, a checklist key that exists only in override-configured
runs. That was my brief's omission, not a numerical difference; it was fixed and the run repeated.

### 2.3 Polish: 17 of 48 blocks fail

| blocks | IPOPT outcome |
|---|---|
| **all 12 TSO blocks** | Converged to a point of local infeasibility |
| 5 DSO blocks — DSO5 2025 Spring; DSO7 2025 Spring, Autumn; DSO9 2025 Spring, Autumn | iteration limit / restoration failed (numerical, not proven infeasible) |
| the other 31 DSO blocks | solved |

- **The risk was pre-registered.** I recorded it before the run (`P5_15_S40_POLISH_PRELAUNCH_NOTE.md`, `affdc19f`),
  after 13 of 48 blocks failed in the 2-cycle smoke test.
- **ρ/σ not touched.** As Step 3.5 requires, the ρ/σ ratio has not been changed.
- **Partial result on the 31 blocks that solved:** the base objective changes by +330,856 in total, 0.051% of the
  certified cost. Seventeen blocks went up and 14 down; the largest change was +246,696, on DSO5 2030 Spring. This covers
  no TSO block, so **it is not a gate value.**

### 2.4 Why — what is established and what is not

Zero-solve checks, preserved in `p515_s40_polish_failure_checks.py`:

| hypothesis | result |
|---|---|
| **H-V** — the midpoint of the two sides' interface voltages lies outside a bound | **Falsified.** 0 of 864 entries. The largest TSO/DSO voltage gap is 7e-5 pu. |
| **H-S** — the midpoint of the interface flows exceeds a branch rating | **Supported for 6 of 12 TSO blocks.** At node 7 the midpoint apparent flow exceeds the 100 MVA rating, by at most 5e-4 MVA, in periods of every spring and summer TSO block (2025/2030/2035). |
| the six **autumn and winter** TSO failures | **Unexplained** by either check. |

**Reading.** The certified point satisfies consensus only to the Boyd tolerance. Where a coupling constraint is **active**
at that point — node 7's interface at its rating — the midpoint can breach it by up to the primal residual, making the
fixed-consensus problem infeasible even though the ADMM point is certified. That accounts for half the TSO failures; the
other half needs the violated constraints read from the restoration-phase logs. **Exact-consensus polish with midpoint
fixing may be ill-posed at points with active coupling constraints.** The programme's earlier polish successes (P5.5-D1,
P5.7) were at points where node 7's interface did not bind.

## 3. What this means for Step 3's closing criterion

Step 3 closes when 3.0–3.5 pass. 3.0–3.4 have passed, and the oracle is fixed, certified and encoded in the case file.
3.5 is the paper's optimality evidence (R2.5), so it matters that the number is well-posed, not merely that one exists.
The choice of test is the Author's and the Expert's.

## 4. Node 7 — the Addendum 22 reporting instruction is not supported as stated

Addendum 22 asked for node 7's at-rating periods to be reported as the storage's congestion-relief value and the reason
for siting there. The period-by-period cross-check, identical across arms A–D, does not support that:

| test | observed | expected if independent |
|---|---|---|
| top-25 PF-residual entries falling in at-rating (≥ 99%) periods | **0** | ~2 |
| same at ≥ 95% | 1–2 (enrichment 0.43–0.85) | ~2.3 |
| binding periods with storage at ≥ 99% of its own rating | **1–2 of 23–27** | ~4 |
| storage dispatch in the top binding periods | **≈ 0.0004 MW** (idle) | — |

- **What holds:** node 7's interface is the only coupling constraint that binds, in 23 of 288 periods. Nodes 5 and 9 peak
  at 51% and 67% of their ratings.
- **What is not supported:**
  - that the storage relieves that congestion — it is essentially idle when the interface binds;
  - that the binding interface explains the PF residual tail — the coincidence is at or below chance.
- **New connection:** that same active rating is what makes half the TSO polish solves infeasible (§2.4).

## 5. The cost differences are real, and C and D reach theirs differently

All four runs (run 1, A, C, D) are the same candidate under four ADMM configurations, compared here only to diagnose the
differences. Each difference reconciles exactly to two components; every other priced component is identically zero, and
the reconciliation residual is ~1e-7.

| vs run 1 | generation cost | internal flexibility cost | net |
|---|---|---|---|
| A | +75,588 | −114,737 | −39,149 |
| C | −60,781 | +3,038 | −57,743 |
| **D (oracle)** | −44,476 | −27,715 | **−72,191** |

- **Not stopping slack.** Every difference exceeds its rule-nine bar, so these are different operating points of a
  nonconvex problem.
- **Offsetting moves.** Moves of order 10⁵ partly cancel to leave differences of order 10⁴.
- **C and D disagree in mechanism:** similar totals, but through **opposite-signed** flexibility moves.
- **The adopted rule** (one frozen configuration per campaign; costs never mixed across configurations) contains this for
  candidate ranking. It also means the manuscript cannot speak of a configuration-independent "system cost" for C\*.

## 6. Step 3.6 — the parallel path does not deliver

### 6.1 Clone replacement: done

Integrated, and bitwise identical to the legacy clone on every numeric artifact. Clones per cycle go from 12 to 0 on the
TSO path. The Worker's pristine-block-per-(year, day) design avoided the interface-load re-fix bug Addendum 22 asked to
be recorded: re-invoking the constructor would re-fix interface loads from the mutated consensus.

**Correction.** The "~4 s per cycle of TSO clones" I reported earlier was the **DSO node-7** failure-snapshot clone
(~3 s/cycle, still present, untouched by the change). The TSO removal saves ~1 s/cycle (12 × 0.088 s), within run-to-run
noise, so my recorded prediction of 31–33 s/cycle was not met: it rested on the misattributed figure.

### 6.2 Screening: passes

The 10-cycle measurement gives 97.0% absorbable overhead against a 70% bar, projecting 5.09× on 8 workers. Recorder off
vs on is bitwise identical, and the timing split covers all 564 solves. The earlier "indeterminate" verdict was a wiring
bug.

### 6.3 Persistent workers: built, gate fails narrowly, speed-up not delivered

- **Gate: fails on one field.** Every numeric artifact is identical and solves reconcile (153 = 153). But the parent's
  state sync writes implicit variable bounds back as explicit ones, so the pickled storage models differ in size. The
  same sync emits ~264k Pyomo domain warnings in two cycles.
- **Speed:**

  | per cycle | cycle 1 | cycle 2 |
  |---|---|---|
  | serial | 30.4 s | 33.5 s |
  | 8 workers | 26.3 s | 32.3 s |

  That is ~1.04–1.16×, against 5.09× projected.
- **Probable cause, not measured:** the parent re-syncs every block's full variable state each cycle. The screening
  counted that work as absorbable per-block overhead, but it lands on the parent, serially.

## 7. Step 3.7 — held

Authorized "after 3.5 and after the 3.6 bitwise gate"; neither has passed, so it has not started. Its own gate is
two-cycle identity with the flag off, then certification ≤ 80 cycles at C\* with cost within D's bar. That gate depends
on neither blocked item.

## 8. Options and the Planner's recommendation

1. **Step 3.5.**
   - (a) **Zero-solve:** read the violated constraints from the restoration-phase logs of the six unexplained TSO blocks
     and the five DSO failures. Cheap; completes the diagnosis.
   - (b) **Restate the test so it is well-posed at active coupling constraints.** Candidates:
     - fix the coupling quantities at the midpoint **projected** onto the active constraint set (e.g. scale P, Q back to
       the rating where the midpoint exceeds it);
     - fix one side's achieved values;
     - allow the fixed quantities a tolerance equal to the Boyd primal tolerance the point was certified to.
   - (c) Evaluate the gap at a different fixed point, e.g. P5.5-D1's convex-parent mapping.

   **Recommendation: (a) first, then (b) with projection.** It keeps P5.7's design and changes only the treatment of
   bounds the certified point satisfies to tolerance. **The decision is yours**, because the test is the paper's
   optimality evidence.
2. **Node 7.** Report only what holds: the one binding coupling constraint, 23 of 288 periods, with the utilization
   table. Drop the congestion-relief and siting-reason claims unless a counterfactual shows them, e.g. a Step 5
   evaluation without storage at node 7, or with it sited elsewhere, under the frozen oracle.
3. **Persistent workers.** Measure before fixing: per-phase timing on the parallel path over ~5 cycles, to locate the
   serial parent work. Fix the bound restore (small), and take the DSO node-7 clone removal (the same technique; ~3 s per
   cycle, removable without parallelism). Reconsider the architecture only if the parent sync is irreducible.
4. **Step 3.7.** Allow it to proceed now on the oracle: its gate is independent of 3.5 and of the worker gate, and
   Anderson acceleration's value can be read in parallel with the 3.5 decision.

## 9. Questions for the Expert and the Author

1. Which Step 3.5 test — midpoint as specified, projected, one-sided, toleranced, or a different fixed point — is the
   paper's optimality evidence at a point with active coupling constraints?
2. Do you accept the node 7 restatement in §8.2, and is a counterfactual siting evaluation wanted for Step 5?
3. Persistent workers: authorize parallel-path profiling plus the two small fixes, or pause the path?
4. Step 3.7: proceed now, or hold until 3.5 is resolved?
5. The cost result (§5): does "no configuration-independent system cost" need stating in the manuscript's method
   section?

## 10. Evidence index

| item | artifact | commit |
|---|---|---|
| prior handoff | `P5_15_ADDENDUM21_EXPERT_REPORT.md` | `25579b65` |
| spec v11 | `data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json` | `d2edf8b6` |
| cost decomposition, node 7 | `p515_s40_cost_decomposition.py`, `p515_s40_node7_crosscheck.py`, `P515S40/{cost_decomposition,node7_result}/`, `WORKER_REPORT_S40_ANALYSES.md` | `11075352`, `e4db9033`, `d73a2b22`, `be80fe99` |
| clone replacement and bitwise gate | `admm_parameters.py`, `network.py`, `shared_resources_planning.py`; `p515_s40_clone_capture_preflight.py`; `P515S40/clone_preflight{,_v2}/` | `b4ca1466`…`3d00e61f`, `8f5cff48`, `8214be0d` |
| 10-cycle timing | `p515_s36_step36_timing*.py`; `P515S36/step36_timing_10cyc{,_defectfix}/`; `WORKER_REPORT_S36_TIMING_10CYC.md` | `20b6fd23`, `9428c0b7`, `872ff8e0` |
| persistent workers | `admm_persistent_workers.py`; `p515_s40_persistent_workers_{checks,preflight}.py`; `P515S40/persistent_workers_*`; `WORKER_REPORT_S40_PERSISTENT_WORKERS.md` | `c6d60857`…`d8414fb7` |
| case file | `data/SRP1/SRP1_params.json`; `p515_s40_case_file_{oracle_checks,repro}.py`; `WORKER_REPORT_S40_CASE_FILE.md` | `fb3de341`, `ee976eff`, `bc01238e` |
| polish gap | `p515_s40_polish_gap{,_checks}.py`; `P5_15_S40_POLISH_PRELAUNCH_NOTE.md`; `P515S40/polish_gap{,_v2,_smoke,_checks}/`; `WORKER_REPORT_S40_POLISH_GAP_PREP.md` | `9422ba9b`, `affdc19f`, `51a5fb48`, `57d523d0` |
| polish-failure checks; closure report | `p515_s40_polish_failure_checks.py`, `P515S40/polish_failure_checks.json`; `P5_15_S40_STEP3_CLOSURE_REPORT.md` | `96e9848d` and the commit before it |

Every zero-solve claim above is backed by an armed `SolveProfileGuard`. Per-entry strides and `esso_capture/` are
hash-recorded; all other evidence is committed with sha256 manifests.
