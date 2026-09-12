# P5.12-W — oracle-reliability breadth probe

**Authorship note.** Planner-executed. The Worker session was terminated by an
external API spend limit during P5.12-G; this stage was run directly by the
Planner under the existing authorization. Sampling plan and materiality
thresholds were frozen to disk (`data/SRP1/Results/P512W/frozen_sampling_plan.json`,
SHA-256 `ba638f77c628f64c6252a21db5c0d72c5602677a9c2e277cb6c3d3aca426f7c8`)
**before** the first solve. No threshold, fixture or ordering was changed
afterwards.

## Phase 0 — fixture inventory (no solves)

Complete prepared solver fixtures with a preserved baseline exist **only for the
cycle-21 target block**, because P5.12-R deliberately captured that block alone.

| Candidate | Status | Reason |
|---|---|---|
| `TARGET_cycle21` (Arm A) | KNOWN POSITIVE CONTROL | already shown sensitive; reused, not rerun; excluded from all breadth counts |
| `TARGET_cycle20` | **REPLAYABLE** | complete pre-setup and prepared captures, embedded network/params, effective options, baseline NL hash, baseline log and `SolverResults` |
| `FROZEN_DSO_node7_case33_2_2025_Autumn_cycle7` | replayable only with enlarged budget | complete pre-solve model, RESCALED objective, full suffixes, `captured_outcome=optimal`; **missing** network/params objects and baseline NL hash → needs deterministic reconstruction plus a BASELINE verification solve (3 solves) |
| `FROZEN_TSO_case9_2025_Summer_cycle7` | replayable only with enlarged budget | same |
| Other 45 network blocks, all other cycles | NOT SCIENTIFICALLY REPLAYABLE | no pre-solve model state preserved; only appended per-block IPOPT logs (51 files). Replay would require re-running ADMM coordination, which this stage forbids |
| `cycle20_checkpoint` 51 block models | NOT SCIENTIFICALLY REPLAYABLE | post-solve end-of-cycle states, not pre-solve fixtures; would need trajectory reconstruction plus a baseline solve |
| `p511_selfconsistent_t0.pkl`, `P57/t0_state.pkl` | NOT SCIENTIFICALLY REPLAYABLE | whole-trajectory template states, not per-block fixtures with a preserved baseline |

**Consequence:** the 6-8 fixture target is unreachable from preserved artifacts.
One new fixture is replayable within the declared two-solve budget.

## Execution

2 new solves (LOW then HIGH), one fresh Python process each, fixture reloaded
independently from the frozen artifact. Gates passed in both: pre-setup digest
`393ce242…`, prepared digest `58e2a063…`, NL byte-identical to the frozen
prepared NL `fbe9e6e8…`, and the effective-option diff exactly
`{output_file, warm_start_bound_push}`. One solve per process, enforced.

## NEW fixture result — `TARGET_cycle20` (DSO:case33_3 | 2025 | Spring, cycle 20)

| | LOW `1e-6` | BASELINE `1e-5` | HIGH `1e-4` |
|---|---|---|---|
| termination | Optimal | Optimal | Optimal |
| iterations | 114 | 115 | 111 |
| objective (unscaled) | 1348.4251826018010 | 1348.4251823032162 | 1348.4252001948971 |
| dual infeasibility | 1.14e-03 | 1.13e-03 | 8.67e-03 |
| constraint violation | 7.28e-07 | 7.21e-07 | 6.95e-07 |
| complementarity | 4.56e-05 | 4.54e-05 | 4.82e-05 |
| safeguard (`z`) rows | 0 | 0 | 0 |
| restoration rows | 0 | 0 | 0 |

Against the frozen thresholds: relative objective difference `2.2e-10` (LOW) and
`1.3e-08` (HIGH), both far inside the `1e-6` equivalence bound. Iteration change
0.9% and 3.5%, far below the 50% path threshold; no restoration and no safeguard
in any run. Maximum scaled primal distance to baseline `4.05e-06` (LOW) and
`2.01e-04` (HIGH) — both in the declared indeterminate band (`1e-6`…`1e-3`),
therefore reported descriptively and **not** used for categorical classification.
No branch flip: that requires primal distance `>1e-3` **and** relative objective
difference `>1e-3`; neither holds.

**Classification: ROBUST.**

## Breadth counts (positive control excluded)

- NEW fixtures tested: **1**
- OUTCOME FLIP: **0**
- OPERATIONAL-OUTPUT / BRANCH FLIP: **0**
- PATH-ONLY sensitive: **0**
- ROBUST: **1** (`TARGET_cycle20`)
- Excluded: 2 frozen comparator blocks (budget), and all non-replayable candidates above.

## Known positive control (reported separately, not counted)

`TARGET_cycle21`: baseline `1e-5` → maxIterations (3000); LOW `1e-6` → Optimal
(89); HIGH `1e-4` → Optimal (151). Reused from P5.12-P; not rerun. The pipeline
used here is the same one that produced it and behaved consistently: identical
gate structure, identical single-option discipline, NL byte-equality to the
frozen capture in both stages.

## Interpretation

One robust new fixture is a data point, not breadth. No population inference is
made or authorized. The evidence is insufficient to distinguish "the cycle-21
target is special" from "fragility is common", because the preserved-artifact
base cannot supply a sample capable of separating those.

## Verdict

BREADTH PROBE INCONCLUSIVE
