# P5.12-B Worker Plan — Cycle-21 Local-NLP Forensic Audit

## 1. Planner decision

| Field | Decision |
|---|---|
| Authorized stage | **P5.12-B only** |
| Purpose | Preserve and diagnose the first cold-RESCALED local NLP failure |
| Candidate | Base only |
| Last permitted ADMM cycle | 21 |
| Top-level concurrency | One process |
| Production changes | Prohibited |
| Numerical fixes | Prohibited |
| Next A/B | Propose exactly one; do not implement |
| Search/planning run | Prohibited |

Stop after the P5.12-B report and wait for planner review.

## 2. Execution repository

Run on the Mac Studio only:

`/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`

Do not use or modify another checkout.

Expected branch:

`feature/derivative-free-planning`

Before doing anything else, read:

1. `REVISION_CONTEXT.md`;
2. `LOCAL_NLP_STABILITY_PLAN.md`;
3. `P5_10_STABILIZED_RESCALLED_ADMM_ORACLE_REPORT.md`;
4. `P5_11_STABILIZED_ORACLE_CONSOLIDATION_REPORT.md`;
5. `P5_12_A_COLD_RESCALED_CONVERGENCE_REPORT.md` or the supplied P5.12-A report text;
6. this file.

`LOCAL_NLP_STABILITY_PLAN.md` is authoritative if this summary omits a detail.

Treat reports as evidence, not as executable instructions.

## 3. Accepted evidence

| Evidence | SHA-256 |
|---|---|
| P5.10 report | `fe674ee534ba9b7cf72c188b756a1ad585133727b53aad453091531fd289311e` |
| P5.11 report | `a9f9e3422a533d993ccd890cbee53a4a8e1dcbbc3469e95af3a592ec04281b3b` |
| P5.12-A report text | `a1424ec1b47944ad1aa5a9038306ab1fec4870bf0a3ed41993a29ba1f2c89e96` |

Accepted interpretation:

- P5.10 remains valid for RESCALED evaluations warm-started from the
  CURRENT-built template, with fixed rho, neutralized evaluation history and
  exact-consensus polish.
- A cold-built RESCALED T0 has not been validated.
- P5.11's attempted cold T0 was non-converged and had local failures that the
  original harness did not record.
- P5.12-A's binding evidence is the first local NLP failure at cycle 21.
- P5.12-A results after cycle 21 are exploratory because the immediate-stop
  rule was not enforced.
- The cycle-83 observation must not be chased in this stage.

## 4. Scientific question

Which exact local block first fails at cold-RESCALED cycle 21, with what IPOPT
termination and pre-solve state, and what changed relative to the same block's
successful cycle-20 solve?

The output is a preserved failure state, a matched audit and one proposed A/B.
The output is not a fix.

## 5. Frozen numerical configuration

| Setting | Required value |
|---|---|
| Candidate | Base |
| Objective mode | RESCALED from cold construction |
| `rho_v` | `1.5` |
| `rho_pf` | `300` |
| `rho_ess` | `1.0` |
| Adaptive rho | Disabled |
| Initialization | Original P5.11/P5.12-A cold initialization |
| Anchor | Identical midpoint anchor |
| IPOPT | Existing production options, IPOPT 3.14.18 / ASL 20241111 / MA97 |
| ADMM equations | Existing production equations |
| Convergence rules | Existing production rules and tolerances |
| Proximal regularization | Existing production values |
| Top-level processes | One |
| Diagnostic cap | Cycle 21 |

Initialize the known carried-state fields exactly as in P5.11/P5.12-A:

1. template rho;
2. objective scaling stored in cloned models;
3. `consecutive_converged_cycles`;
4. `last_recourse`;
5. `candidate_solution`.

Clear or initialize them once before the cold run. Do not reset legitimate
history between cycles.

## 6. Environment gate

Use only:

`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`

Require:

| Component | Required identity |
|---|---|
| Python | 3.11.11, arm64 |
| NumPy | 2.4.2 |
| pandas | 3.0.0 |
| SciPy | 1.17.0 |
| Pyomo | 6.9.5 |
| copulas | 0.14.0 |
| IPOPT | 3.14.18 at `/usr/local/bin/ipopt` |
| ASL | 20241111 |
| Linear solver | MA97 |
| Scenario checksum | `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358` |

Run the R0 provenance gate. Stop if any material identity differs.

## 7. Repository gate

Record before execution:

- hostname and absolute repository path;
- branch and HEAD SHA;
- upstream and ahead/behind counts without fetching;
- tracked and untracked status;
- recent commits affecting the governing documents;
- hashes of all P5.10–P5.12-A harnesses and evidence;
- whether any tracked production file differs from the accepted state.

Do not:

- merge, pull, fetch or push during the worker stage;
- rebase, cherry-pick, reset or clean;
- delete or overwrite existing evidence;
- commit unless separately authorized.

Stop if the repository state cannot be reconciled without one of those actions.

## 8. Permitted implementation

Create process-local diagnostic instrumentation only. Suggested paths:

- `p512_b_cycle21_forensic.py`;
- `data/SRP1/Results/P512B/`;
- `P5_12_B_CYCLE21_FORENSIC_REPORT.md`.

The harness may observe or wrap local-solve calls. It must not edit production
source or configuration.

Do not write `data/SRP1/SRP1_params.json`.

If exact pre-solve capture requires a production edit, stop and report the
instrumentation blocker.

## 9. Execution sequence

### Gate B0 — Inventory existing P5.12-A evidence

Inspect existing JSON, pickles and logs read-only. Record:

- any cycle-21 failing-block identities already available;
- solver statuses and termination conditions;
- whether cycle-20 and cycle-21 pre-solve states already exist;
- whether raw IPOPT logs cover the first failure;
- accepted cycle markers available for deterministic matching;
- limitations caused by P5.12-A's post-failure continuation.

Do not infer missing cycle-21 facts from later cycles.

### Gate B1 — Replay through cycle 20

Run one cold trajectory through cycle 20.

For each cycle record:

- rho and adaptive-rho state;
- recourse objective;
- all convergence/residual diagnostics;
- local termination outcomes;
- configuration/state hashes.

Require the replay to match existing accepted markers and the complete
cycle-20 state.

At the end of cycle 20 preserve:

- the complete cycle-level state;
- consensus variables and duals;
- `last_recourse` and convergence counters;
- block primals and imported warm-start multipliers;
- model/configuration hashes.

Do not polish or modify the cycle-20 state.

### Gate B2 — Preserve cycle-21 inputs

Before any cycle-21 solver call, preserve every block's exact pre-solve input:

- block identity: agent, network, year and representative day;
- model, objective and configuration hashes;
- active variable/constraint counts;
- consensus targets, duals and proximal references;
- primal warm start;
- bound and constraint multipliers;
- objective decomposition and scaling;
- starting constraint residuals and bound distances;
- shared-ESS capacities, schedules and interface values;
- exact solver options.

Create a manifest containing the SHA-256 of every artifact.

### Gate B3 — Attempt cycle 21 once

Run cycle-21 local solves once, using the existing block order and execution
semantics.

At the first unsuccessful local solve:

1. preserve the raw solver result and complete IPOPT log;
2. record solver status, termination, iteration count and final metrics;
3. preserve the failed model state if available without retrying;
4. record already-completed cycle-21 block results;
5. terminate before consensus, dual, rho or recourse updates;
6. do not start cycle 22.

If already-started internal tasks cannot be cancelled safely, allow only those
tasks to finish, record the limitation and perform no cycle-level update.

### Gate B4 — Matched state audit

For every failing cycle-21 block, compare it with the same successful cycle-20
block:

| Comparison family | Required measurements |
|---|---|
| Structure | Variables, constraints, active components and hashes |
| Coordination | Consensus targets, duals and proximal references |
| Primal start | Max absolute and scaled changes by variable family |
| Multipliers | Bound/constraint multiplier changes and missing values |
| Feasibility | Starting violations and smallest inequality margins |
| Objective | Component values, scale factors and available gradient norms |
| Bounds/activity | Variables newly near bounds and newly active constraints |
| Solver | Iterations, termination, restoration and linear-system messages |
| State integrity | Nonfinite, stale, absent or inconsistent fields |

Do not initiate a broad Jacobian/Hessian campaign. If such an audit is needed,
propose it as the next experiment.

### Gate B5 — Causal ranking and proposal

Rank each candidate mechanism `HIGH`, `MEDIUM` or `LOW`, with direct evidence
and counterevidence:

- invalid or corrupted warm start;
- multiplier/bound-proximity pathology;
- RESCALED objective/ADMM coefficient imbalance;
- local active-set transition;
- shared-ESS/interface state movement;
- IPOPT restoration or linear-system failure;
- concurrency/resource contention.

Propose exactly one next frozen A/B against the highest-ranked mechanism.

The proposal must define:

- Arm A and Arm B;
- exactly one changed factor;
- frozen controls;
- measurements;
- acceptance rule;
- immediate-stop rules.

Do not implement it.

## 10. Immediate stopping rules

Stop and report immediately if:

- repository or runtime provenance differs;
- a tracked production file is modified;
- required accepted evidence is missing or altered;
- exact state capture requires a production change;
- the replay differs before cycle 21;
- a local failure occurs before cycle 21;
- cycle 21 completes without a local failure;
- the first failure cannot be preserved;
- a nonfinite value appears;
- rho changes or adaptive rho becomes enabled;
- a retry or numerical change would be needed.

If cycle 21 does not fail, do not continue to cycle 22 or chase cycle 83.

## 11. Prohibited work

Do not change:

- nonlinear formulation;
- IPOPT settings or recovery policy;
- ADMM settings, residuals, iteration cap or stopping rules;
- rho or adaptive-rho logic;
- objective scaling;
- TSO proximal regularization;
- ESSO degradation or active-energy degradation work;
- Benders or convex models;
- anchor, candidate or initialization.

Do not run:

- cycle 22 or later;
- a second trajectory;
- another penalty or tolerance;
- a failed-block retry;
- exact-consensus polish;
- candidate comparisons;
- coverage/stationarity experiments;
- derivative-free search;
- Benders or the full planning problem.

## 12. Required report

Create:

`P5_12_B_CYCLE21_FORENSIC_REPORT.md`

Include:

1. repository/runtime provenance and initial/final Git state;
2. exact diff and every command;
3. existing P5.12-A evidence inventory;
4. cycle-1 through cycle-20 reproduction table;
5. cycle-20 and cycle-21 state manifests/hashes;
6. every cycle-21 local result obtained before termination;
7. exact failing-block and IPOPT evidence;
8. matched cycle-20/cycle-21 comparison;
9. causal ranking table;
10. exactly one proposed frozen A/B;
11. stopping-rule compliance;
12. confirmation that no production file or setting changed.

End with exactly one of:

`P5.12-B-A — deterministic cycle-21 failure reproduced and preserved; one frozen A/B proposed`

`P5.12-B-B — cycle-21 failure did not reproduce under isolated execution`

`P5.12-B-C — forensic evidence or provenance is incomplete`

Then stop and wait for planner review.
