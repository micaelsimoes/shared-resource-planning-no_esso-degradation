# P5.15 Addendum 22 — Step 3 closure: blocked at 3.5 (polish gap not evaluated)

**Planner report, 2026-09-18; stop for review.** Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 22. Frozen spec:
`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json` (`d2edf8b6`).

## 1. Closure status per item

| item | status | commit |
|---|---|---|
| (1a) cost decomposition, C/D vs run 1 | **done** — findings in §3 | `11075352`, `d73a2b22`, `be80fe99` |
| (1b) node 7 cross-check + result table | **done** — congestion-relief mechanism **not supported** (§4) | `e4db9033`, `d73a2b22`, `be80fe99` |
| (2) Step 3.6 clone replacement | **done**: integrated; bitwise gate PASSED | `b4ca1466`…`3d00e61f`, `8214be0d` |
| (2) Step 3.6 10-cycle re-measurement | **done**: screening PASSES, 97.0% absorbable, projected 5.09× | `9428c0b7`, `872ff8e0` |
| (2) Step 3.6 persistent workers | **built; gate FAILS by one non-numerical field; measured speed-up ~1.1×** (§5) | `c6d60857`…`d8414fb7` |
| (3) D's configuration into the case file | **done** — Track B closed; case file reproduces D | `fb3de341`, `ee976eff`, `bc01238e` |
| (4) Step 3.5 polish gap, gate < 0.1% | **NOT EVALUATED** — 17 of 48 blocks fail at fixed consensus (§2) | `51a5fb48`, `57d523d0` |
| (5) `REVISION_CONTEXT.md` head, Step 3 closed | **not done** — Step 3 cannot close without 3.5 | — |
| (6) Step 3.7 Anderson acceleration | not started (sequenced after 3.5 and the 3.6 gate) | — |

## 2. Step 3.5 — the polish gap could not be evaluated

**What ran.** The case-file configuration ran to certification in-process, then all 48 network blocks were re-solved with
the unscaled base objective at fixed consensus: P5.7's design via `p56a_oracle`, with interface quantities fixed at the
midpoint of the two sides' achieved values and the storage channel at its consensus variable.

**Reproduction: passed, twice.** Both full runs reproduced arm D bitwise — certification at cycle 139,
`gross_operational_cost` 650,966,975.2943751, all 139 trajectory rows, 7,179 solves, 38 recovered failures.
**τ = 0 determinism is confirmed over a full certified run.** The first run stopped before polishing on one provenance
field (a checklist key present only in override-configured runs); I had failed to carry that known exclusion into the
brief. Fixed and re-run (`51a5fb48`).

**Polish: 17 of 48 blocks failed, so the gate is not evaluated.**

| blocks | IPOPT outcome |
|---|---|
| all 12 TSO blocks | Converged to a point of local infeasibility |
| DSO5 2025 Spring; DSO7 2025 Spring, Autumn; DSO9 2025 Spring, Autumn | Maximum iterations exceeded / Restoration failed (numerical, not proven infeasible) |
| the other 31 DSO blocks | solved |

This was the risk recorded before the run (`P5_15_S40_POLISH_PRELAUNCH_NOTE.md`, `affdc19f`). Per Step 3.5's own rule, the
ρ/σ ratio has **not** been touched.

**The 31 blocks that solved** change the base objective by **+330,856 in total**, 0.051% of the certified cost; 17 went up
and 14 down, and the largest was +246,696 on DSO5 2030 Spring. This is a partial subset with no TSO block, so **it is
not a gate value and must not be read as one.**

**Two zero-solve hypothesis checks** (`p515_s40_polish_failure_checks.py`, guard armed, 0 solves):
- **Voltage midpoint (H-V): falsified.** No interface voltage midpoint lies outside its bounds. Thirty-nine entries sit
  within 1e-6 pu of the 1.1 pu bound, but the largest TSO/DSO voltage gap is 7e-5 pu and the midpoint stays inside.
- **Interface-rating midpoint (H-S): partly supported.** At node 7, fixing P and Q at the midpoint puts the apparent flow
  over the 100 MVA rating, by at most 5e-4 MVA (utilization 1.000005), in periods of **six** TSO blocks: all spring and
  summer blocks for 2025/2030/2035. **The six autumn and winter TSO failures are not explained by either check.**

**Reading, for review.** Exact-consensus polish fixes the coupling quantities at a point that satisfies consensus only to
the Boyd tolerance. Where a coupling constraint is **active** at the certified point — node 7's interface at its rating,
voltages at their bound — the midpoint can violate that constraint by up to the primal residual, making the fixed
problem infeasible even though the ADMM point is certified. That explains half of the TSO failures. The other half needs
the failing constraints read from the restoration-phase logs, which I have not done.

The test as specified may therefore be ill-posed at points with active coupling constraints. Whether to fix the
quantities at a projected rather than a midpoint value, fix one side's values, allow a tolerance on the fixed
quantities, or keep the test and change something else is a method decision — not mine to take.

## 3. Cost decomposition (item 1a)

All four runs are the same candidate under **four different ADMM configurations**; per Addendum 22 these costs are
compared here only to diagnose the differences.
- **Where the difference sits.** Each difference reconciles exactly to **generation cost + internal flexibility cost**.
  Every other priced component is identically zero, and the residual is about 1e-7.

  | vs run 1 | generation | flexibility | net |
  |---|---|---|---|
  | A | +75,588 | −114,737 | −39,149 |
  | C | −60,781 | +3,038 | −57,743 |
  | D (oracle) | −44,476 | −27,715 | −72,191 |

- **Not stopping slack.** Every difference is outside its rule-nine bar, so these are different operating points of a
  nonconvex problem.
- **C and D disagree in how they get there.** They reach similar totals through **opposite-signed** flexibility moves.
- **What this constrains.** The adopted rule (one frozen configuration per campaign; no mixing costs across
  configurations) contains this for ranking. It also constrains how the manuscript can speak of "the" system cost.

## 4. Node 7 (item 1b) — the Addendum 22 reporting instruction is not supported as stated

Addendum 22 asked that node 7's at-rating periods be reported as the storage's congestion-relief value and the reason for
siting there. The zero-solve cross-check, identical across arms A–D, does not support that reading:

| test | observed | expected if independent |
|---|---|---|
| top-25 PF-residual entries in at-rating (≥ 99%) periods | **0** | ~2 |
| same, at ≥ 95% | 1–2 (enrichment 0.43–0.85) | ~2.3 |
| binding periods with storage at ≥ 99% of its own rating | **1–2 of 23–27** | ~4 |
| storage dispatch in the top binding periods | **≈ 0.0004 MW** (idle) | — |

**What holds:** node 7's interface is the only one that binds, in 23 of 288 periods; nodes 5 and 9 never exceed 51% and
67% of their ratings. **What is not supported:** that the storage relieves that congestion, or that the binding interface
explains the PF residual tail. The reporting instruction needs restating before it enters the manuscript. One
connection is new: the same active rating is what breaks half the TSO polish solves (§2).

## 5. Step 3.6 — persistent workers

Built behind a default-off flag, and it preserves recovery, FrozenSMOPF snapshots, diagnostics and solve accounting
(counted in the child processes).
- **Bitwise gate: fails by one non-numerical field.** Every numeric artifact over two cycles is identical, and solves
  reconcile 153 = 153. But the parent's state sync writes implicit variable bounds back as explicit ones, so the pickled
  storage models differ in size. The same sync emits about 264k Pyomo domain warnings in two cycles.
- **Speed: not delivered.**

  | per cycle, wall time | cycle 1 | cycle 2 |
  |---|---|---|
  | serial | 30.4 s | 33.5 s |
  | 8 workers | 26.3 s | 32.3 s |

  That is about 1.04–1.16×, against 5.09× projected.
- **Probable cause, not measured:** the parent re-syncs every block's full variable state each cycle. That is serial
  work the screening counted as absorbable.
- **The matched full-length run was not launched**, since the two-cycle gate has not passed.

**Correction to an earlier claim of mine.** The "4.1 s per cycle TSO clone" I cited was the **DSO node-7** failure-snapshot
clone, which still costs about 3 s per cycle. The TSO clone removal saves about 1 s per cycle (12 × 0.088 s), within
run-to-run noise. My recorded prediction of 31–33 s per cycle is therefore not met: it was made from the misattributed
figure.

## 6. Case file (item 3) — Track B closed

Seven keys changed in `data/SRP1/SRP1_params.json`, each old → new: τ 1.0 → 0.0; ρ_ess 0.1125 → 0.01 (all networks and
ESSO); `balancing_exempt_until` absent → ESS {< 1.0 for 5 cycles}; `freeze_backstop_cycle` 60 → 200;
`minimum_consecutive_converged_cycles` 3 → 10; `num_max_iters` 25 → 300; `shared_ess_initialization` price-taker →
standalone.
- **Verification:** loaded parameters match D field by field; a case-file-only run reproduces D (two cycles in the check,
  and all 139 cycles in both polish-gap runs).
- **Superseded stages:** the s33e2, s34, s35ref, s35pt, s37 and s38 checklists now fail by design; they are listed with
  file:line in `WORKER_REPORT_S40_CASE_FILE.md`.

## 7. Questions for review

1. **Step 3.5:** the test as specified cannot be evaluated at the oracle's certified point, because the midpoint violates
   active coupling constraints (explained for 6 TSO blocks; 6 unexplained). Options:
   - (a) diagnose the remaining TSO failures from the restoration logs, zero solves;
   - (b) restate the test: projected rather than midpoint values, one side's values, or a tolerance on the fixed
     quantities;
   - (c) evaluate the gap on a different fixed point, e.g. P5.5-D1's convex-parent mapping.

   Which?
2. **Node 7:** how should the result be restated, given that the congestion-relief coincidence is at or below chance?
3. **Persistent workers:** fix the bound handling and profile the parent's sync, or reconsider the design given a
   measured ~1.1× against 5.09× projected? And should the DSO node-7 clone (about 3 s per cycle, removable) be taken
   first?
4. **Sequencing:** Step 3.7 was authorized after 3.5 and after the 3.6 gate, and neither has passed. Hold 3.7, or run it
   on the oracle now since its gate does not depend on either?

## Evidence

- Spec v11; `P5_15_S40_POLISH_PRELAUNCH_NOTE.md`.
- `data/SRP1/Results/P515S40/` subdirectories: `cost_decomposition/`, `node7_result/`, `clone_preflight{,_v2}/`,
  `persistent_workers_{checks,preflight_v4}/`, `case_file_{repro,oracle_load_check}`, `polish_gap{,_v2,_smoke,_checks}/`,
  `polish_failure_checks.json`.
- `data/SRP1/Results/P515S36/step36_timing_10cyc{,_defectfix}/`.
- Worker reports: `WORKER_REPORT_S40_*.md`, `WORKER_REPORT_S36_TIMING_10CYC.md`, `WORKER_REPORT_S36_CLONE_CAPTURE.md`.
- Zero-solve claims are backed by armed `SolveProfileGuard`s. Per-entry strides and `esso_capture/` are hash-recorded.
