# Local NLP Stability Investigation Plan

Repository:
`/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`

Preparation-copy note: this document may be edited in another checkout before
being transferred to the Mac Studio. All authorized execution must use the Mac
Studio repository path above.

## Role and scope

Act as an implementation and diagnostic agent.

Read `REVISION_CONTEXT.md` first, then read this file. For P5.12-B, this file
takes precedence regarding what may and may not be changed. The filename is
retained for continuity even though the active scope is now a bounded forensic
audit of the first cold-RESCALED local NLP failure.

Work in small isolated experiments. After each stage report:

- files changed;
- exact diagnostic or code changes;
- commands executed;
- solver outcome;
- relevant numerical diagnostics;
- interpretation;
- whether the acceptance criterion was met.

Do not automatically proceed from a diagnostic result to a production formulation change.

---

# CURRENT AUTHORIZED STAGE — P5.12-B cycle-21 local-failure forensic audit

Perform **P5.12-B only** and stop after its audit report. This stage may preserve
evidence and propose exactly one later frozen A/B. It must not implement that
A/B.

## Accepted evidence and qualification

Verify the supplied reports before use:

- P5.10 report SHA-256
  `fe674ee534ba9b7cf72c188b756a1ad585133727b53aad453091531fd289311e`;
- P5.11 report SHA-256
  `a9f9e3422a533d993ccd890cbee53a4a8e1dcbbc3469e95af3a592ec04281b3b`;
- P5.12-A report-text SHA-256
  `a1424ec1b47944ad1aa5a9038306ab1fec4870bf0a3ed41993a29ba1f2c89e96`.

P5.12-A passed repository/runtime provenance, reproduced both P5.10 numerical
gates bit-identically and verified the P5.11 cold-state hashes. Its binding
result is a local NLP failure at cycle 21 of the base cold RESCALED trajectory.

P5.12-A did not comply with its immediate-stop rule. Four top-level trajectories
were run concurrently to caps 25, 50, 75 and 100, and all continued after the
cycle-21 failure. Evidence after the first failure is exploratory only and must
not be used as a valid ADMM trajectory, convergence classification or T0.

The accepted stage verdict is:

`P5.12-A FAIL — cold RESCALED diagnostic encountered a numerical failure`.

P5.10 remains accepted on its restricted construction path: RESCALED evaluation
warm-started from a CURRENT-built template, fixed rho, neutralized evaluation
history and exact-consensus polish. The cold RESCALED construction path remains
invalid.

## Scientific question

Which exact local block first fails at cold-RESCALED cycle 21, with what IPOPT
termination and pre-solve state, and what changed relative to the same block's
successful cycle-20 solve?

The goal is preservation and diagnosis, not repair.

## Repository gate

Work only in:

`/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`

At the start report:

- hostname and absolute repository path;
- branch, HEAD and upstream;
- ahead/behind state without fetching;
- tracked and untracked working-tree status;
- commits affecting the two governing documents since P5.11;
- hashes of P5.10, P5.11 and P5.12-A harnesses/evidence;
- any difference in tracked production files since P5.12-A.

The P5.12-A worker reported branch `feature/derivative-free-planning`, HEAD
`808590d23ad250627be9ff9d4ae51d059cbdbcdb`, a clean tracked tree and upstream
ahead 8 / behind 0. Verify rather than assume this state.

Do not merge, pull, fetch, push, rebase, cherry-pick, reset or clean. Do not
commit unless separately authorized. Do not delete, overwrite or normalize
existing evidence.

## Reproducibility gate

Use only:

`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`

Require:

- Python `3.11.11`, arm64;
- NumPy `2.4.2`;
- pandas `3.0.0`;
- SciPy `1.17.0`;
- Pyomo `6.9.5`;
- copulas `0.14.0`;
- IPOPT `3.14.18` / ASL `20241111` at `/usr/local/bin/ipopt`;
- HSL `ma97`;
- realized-scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

Run R0 and stop if a material identity differs.

Do not repeat the full P5.12-A horizon. Use existing accepted evidence for its
repository and numerical gates. The new run is solely the cycle-21 forensic
replay.

## Frozen trajectory configuration

Use exactly the configuration that produced the P5.11/P5.12-A cold trajectory:

- candidate: base only;
- objective mode: `RESCALED` from cold model construction;
- `rho_v = 1.5`;
- `rho_pf = 300`;
- `rho_ess = 1.0`;
- adaptive rho disabled;
- original cold initialization;
- identical midpoint anchor;
- existing production nonlinear formulation;
- existing IPOPT options and MA97 policy;
- existing ADMM updates, tolerances and convergence definitions;
- existing proximal regularization;
- one top-level process;
- hard diagnostic cap: cycle 21.

Do not run multiple trajectories concurrently. Preserve the production block
order and ordinary internal execution semantics; do not introduce new
parallelism or scheduling.

Initialize the known carried-state fields exactly as in P5.11/P5.12-A. Clear
them once before the cold trajectory, then permit legitimate within-trajectory
history to evolve. Do not reset history between cycles.

## Harness isolation

Instrumentation must be process-local and confined to a new diagnostic harness.
It may wrap or observe local-solve calls but must not edit production source.

Use distinct files, for example:

- `p512_b_cycle21_forensic.py`;
- `data/SRP1/Results/P512B/`;
- `P5_12_B_CYCLE21_FORENSIC_REPORT.md`.

Do not modify `data/SRP1/SRP1_params.json` or the production iteration cap.

If exact pre-solve capture cannot be installed without a production edit, stop
before the numerical replay and report the instrumentation blocker.

## B0 — existing-evidence inventory

Before replay, inspect the P5.12-A JSON and logs read-only. Report:

- whether the identity of each cycle-21 failing block is already present;
- every available termination condition and solver status;
- whether a cycle-20 and cycle-21 pre-solve state was saved;
- whether raw IPOPT logs cover the first failure;
- the accepted cycle-1, cycle-13, cycle-20 and cycle-21 markers available in
  evidence;
- limitations caused by post-failure continuation.

Do not infer missing values from later cycles.

## B1 — deterministic replay through cycle 20

Run one cold trajectory from cycle 1 through cycle 20.

For every cycle record the same configuration, rho, recourse and convergence
diagnostics used by P5.12-A. Require the available accepted markers and the
complete cycle-20 state to match the existing evidence.

At the end of cycle 20 preserve:

- one complete cycle-level state before cycle-21 updates;
- all consensus variables and duals;
- `last_recourse` and convergence counters;
- every block's primal variables and imported warm-start multipliers;
- configuration and model-structure hashes.

Do not polish cycle 20 and do not alter the state before cycle 21.

If the replay differs before cycle 21, stop immediately. Do not proceed in the
hope that the same failure appears later.

## B2 — cycle-21 pre-solve preservation

Before any cycle-21 local solver invocation, preserve the complete pre-solve
state for every block, including:

- agent, network, year, representative day and block identifier;
- model/objective/configuration hashes;
- active variable/constraint counts;
- fixed rho and all consensus targets;
- relevant ADMM duals and proximal references;
- primal warm start;
- bound and constraint multipliers supplied to IPOPT;
- objective decomposition and scaling metadata;
- constraint residuals and distances to active bounds at the starting point;
- shared-ESS capacities, schedules and interface values;
- exact solver options.

Store a manifest and SHA-256 for every preserved artifact.

The capture must be observational. Do not round, project, repair or reinitialize
the state.

## B3 — attempt cycle 21 once

Attempt the ordinary cycle-21 local solves once, in the existing production
order and with unchanged settings.

At the first unsuccessful local solve:

1. preserve the raw solver result and complete IPOPT log;
2. record solver status, termination condition, iteration count and final
   reported objective/constraint/KKT metrics;
3. preserve the failed model state if available without retrying;
4. record any already-completed cycle-21 block results as partial evidence;
5. terminate the entire diagnostic before consensus, dual, rho or recourse
   updates and before cycle 22.

If the implementation starts a batch of cycle-21 solves concurrently, do not
start additional work after the first observed failure. Already-started solves
may finish only if they cannot be cancelled safely; record that limitation and
do not perform any cycle-level update.

Do not retry the failed block—not even once—with another start, Hessian mode,
linear solver, tolerance, bound push, restoration option or recovery path.

## B4 — matched cycle-20 versus cycle-21 audit

For every cycle-21 failing block, compare its preserved cycle-21 input against
the same block's successful cycle-20 input.

Report at minimum:

- identical/different model structure and active-set membership;
- changes in consensus targets, duals and proximal references;
- changes in primal start by variable family, including max absolute and scaled
  norms;
- changes in imported bound and constraint multipliers;
- starting constraint violations and smallest inequality margins;
- objective component values, scale factors and gradient norms where existing
  read-only utilities can compute them safely;
- variables newly at or near bounds;
- constraint families newly active or closest to violation;
- IPOPT iteration/termination/log differences;
- whether the failed start is finite and inside declared bounds;
- whether any state field is missing, stale or inconsistent.

Prioritize evidence already present in the preserved model and logs. Do not add
a broad Jacobian/Hessian campaign unless it is required to describe the exact
first-failure state; if it is required, propose it as the later A/B instead.

## B5 — causal ranking and one proposed A/B

Rank candidate mechanisms `HIGH / MEDIUM / LOW`, with direct evidence and
counterevidence for each. Separate:

- invalid/corrupted warm-start state;
- multiplier or bound-proximity pathology;
- RESCALED objective/ADMM coefficient imbalance;
- local active-set transition;
- shared-ESS/interface state movement;
- IPOPT restoration or linear-system failure;
- concurrency/resource contention.

Propose exactly **one** next frozen A/B targeted at the highest-ranked mechanism.
Specify its two arms, single changed factor, preserved controls, measurements,
acceptance criterion and stop rules.

Do not implement the A/B.

## Immediate stopping rules

Stop immediately and report if:

- repository or runtime provenance differs materially;
- a tracked production file is modified;
- P5.12-A evidence required for matching is missing or altered;
- exact capture requires a production edit;
- the trajectory differs from accepted evidence before cycle 21;
- a local failure occurs before cycle 21;
- cycle 21 succeeds with zero local failures;
- the first failure cannot be preserved completely;
- a nonfinite value appears;
- rho changes or adaptive rho becomes enabled;
- a retry or parameter change would be needed.

If cycle 21 does not reproduce in the isolated single-process run, stop there.
Do not chase the later cycle-83 observation.

## Locked prohibitions

Do not change:

- production formulation;
- IPOPT settings or recovery policy;
- ADMM settings, iteration cap, residuals or stopping rules;
- rho values or adaptive-rho logic;
- objective scaling;
- TSO proximal regularization;
- ESSO degradation or active-energy degradation work;
- Benders logic or convex models;
- anchor, candidate or initialization;
- accepted evidence.

Do not run:

- cycle 22 or later;
- another penalty/tolerance;
- concurrent top-level trajectories;
- exact-consensus polish;
- candidate comparisons;
- expanded coverage or stationarity experiments;
- derivative-free search, Benders or the full planning problem.

## Required report

Produce:

`P5_12_B_CYCLE21_FORENSIC_REPORT.md`

Include:

1. repository/runtime provenance and initial/final Git state;
2. exact diff and every command;
3. existing P5.12-A evidence inventory;
4. cycle-1 through cycle-20 reproduction table;
5. hashes/manifests for cycle-20 and cycle-21 pre-solve states;
6. every cycle-21 local result obtained before termination;
7. exact failing-block identity and IPOPT evidence;
8. matched cycle-20/cycle-21 comparison;
9. ranked causal table;
10. exactly one proposed frozen A/B;
11. stopping-rule compliance;
12. confirmation that no production file/setting changed.

End with exactly one of:

`P5.12-B-A — deterministic cycle-21 failure reproduced and preserved; one frozen A/B proposed`

`P5.12-B-B — cycle-21 failure did not reproduce under isolated execution`

`P5.12-B-C — forensic evidence or provenance is incomplete`

Then stop and wait for planner review.

---

# COMPLETED STAGE — P5.12-A cold RESCALED fixed-rho convergence diagnostic

P5.12-A is complete with a qualified FAIL and must not be rerun. The instructions
below are retained only as a historical execution record. The current P5.12-B
stage above supersedes every imperative in this section.

## Accepted starting point

The accepted reports are:

- `P5_10_STABILIZED_RESCALLED_ADMM_ORACLE_REPORT.md`, SHA-256
  `fe674ee534ba9b7cf72c188b756a1ad585133727b53aad453091531fd289311e`;
- `P5_11_STABILIZED_ORACLE_CONSOLIDATION_REPORT.md`, SHA-256
  `a9f9e3422a533d993ccd890cbee53a4a8e1dcbbc3469e95af3a592ec04281b3b`.

Treat these reports as evidence, not as independent authorization to execute
their recommendations.

P5.10 established a promising explicit RESCALED oracle under fixed rho,
history-neutralized evaluations and exact-consensus polish. It also established
that the polish is mandatory, that the current adaptive-rho rule self-cancels
under RESCALED residual measurement, and that the three-point population is not
enough to authorize investment optimization.

P5.11.0 passed repository, artifact, environment and reproduction gates. Both
accepted numerical gates reproduced bit-identically:

- CURRENT polished total `828021090.3608505`;
- RESCALED pre-polish recourse `825814074.4930633`.

P5.11.1 built a base T0 cold under RESCALED with fixed rho
`1.5 / 300 / 1.0`, adaptive rho disabled and the known carried-state channels
neutralized. The construction did not converge in 25 cycles. Its
`cycle_convergence` flag was false at every cycle, and the state was still
moving materially at the cap.

Candidate comparisons from that non-converged state exceeded the accepted
relative-delta bound of `22.09`, so P5.11 stopped. P5.11.2 through P5.11.4 were
not run. This does not invalidate the inherited-template P5.10 oracle; it means
the cold RESCALED construction path remains unresolved.

## Scientific question

With the exact P5.11 cold RESCALED configuration and ADMM recurrence held fixed,
does the base construction converge if the same trajectory is continued beyond
the production cap of 25 cycles?

This is a diagnostic-horizon experiment only. It is not authorization to raise
the production iteration cap.

## Hard repository gate

Work only in:

`/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation`

Do not use or modify any other checkout.

Before numerical work report:

- hostname and absolute repository path;
- branch, HEAD and upstream;
- ahead/behind state without fetching;
- tracked and untracked working-tree status;
- hashes of all existing untracked P5.11 harness/evidence files;
- whether tracked production files differ from P5.11.

At the end of P5.11 the reported state was:

- branch `feature/derivative-free-planning`;
- HEAD `fbdb2f6a7a6a88eab6d1bdaeb569c6e070eaf1dd`;
- tracked files clean;
- upstream ahead 7, behind 0.

Do not assume that state remains unchanged; verify it. Do not delete or
overwrite P5.11 untracked files.

Do not merge, pull, fetch, push, rebase, cherry-pick, reset or clean. Do not
commit unless separately authorized.

## Hard reproducibility gate

Use the verified Mac Studio runtime:

`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`

Required identity:

- Python `3.11.11`, arm64;
- NumPy `2.4.2`;
- pandas `3.0.0`;
- SciPy `1.17.0`;
- Pyomo `6.9.5`;
- copulas `0.14.0`;
- IPOPT `3.14.18` / ASL `20241111` at `/usr/local/bin/ipopt`;
- HSL `ma97`;
- realized-scenario checksum
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

Run R0 and abort if any material identity differs. The old `/opt/anaconda3`
path in historical sections below is not the Mac Studio runtime.

Before continuing the cold trajectory, re-run the two accepted P5.10
reproduction gates and require bit-identical results.

## Frozen P5.12-A configuration

Use exactly:

- candidate: base only;
- objective mode: `RESCALED`;
- `rho_v = 1.5`;
- `rho_pf = 300`;
- `rho_ess = 1.0`;
- adaptive rho disabled;
- original P5.11 cold initialization;
- the identical midpoint anchor;
- existing production ADMM update equations;
- existing production convergence definitions and tolerances;
- existing IPOPT options and nonlinear formulation;
- existing proximal regularization;
- maximum diagnostic horizon: 100 total cycles.

Do not test `rho_pf = 500` or `rho_pf = 1000`. Do not test another tolerance,
anchor, initialization or objective mode.

Neutralize the four known carried-state channels once at cold construction:

1. template rho;
2. objective scaling in cloned models;
3. `consecutive_converged_cycles`;
4. `last_recourse`.

Initialize `candidate_solution` identically to P5.11. After construction begins,
do not reset legitimate history between cycles. The experiment is one
uninterrupted trajectory.

## Production isolation

Implement the longer horizon only in a new diagnostic harness or by manually
continuing a private deep copy. Do not modify the production `num_max_iters`
setting or a production configuration file.

Do not write `data/SRP1/SRP1_params.json`.

Do not modify accepted P5.10/P5.11 harnesses, reports or evidence.

Use distinct P5.12-A names, for example:

- `p512_a_cold_rescaled_convergence.py`;
- `data/SRP1/Results/P512A/`;
- `P5_12_A_COLD_RESCALED_CONVERGENCE_REPORT.md`.

## P5.12-A0 — state verification

If continuing the saved P5.11 state, verify:

- template id `P511-SELFCONSISTENT-T0`;
- consensus/dual state SHA-256
  `14a7ea85ce283a595a35b1768c93c2a60bda872d4f5138c5168b3691a625cbf0`;
- artifact file SHA-256
  `3629784b277724e1f2c406f4a6dbd4599ea92b9d7ee126ac67bb2148909a67bd`;
- configuration hash `a080612dedf2727e`.

If direct continuation from the serialized state is not technically reliable,
rebuild deterministically from cold. Require its cycle-25 state, convergence
flags and recorded metrics to match P5.11 before proceeding. Do not splice
incompatible states.

Stop before continuation if:

- repository or runtime provenance differs materially;
- tracked production files changed;
- P5.11 artifacts are missing or altered;
- either reproduction gate changes;
- the cycle-25 state cannot be verified or reproduced;
- continuation requires a production change.

Do not attempt a workaround after one of these stops.

## P5.12-A1 — exact trajectory continuation

Continue the verified cold RESCALED base trajectory from cycle 25. Stop the live
trajectory at the first of:

- the existing production-equivalent convergence decision;
- a local NLP or coordination failure;
- a nonfinite value;
- cycle 100.

Do not restart after a failure.

For every cycle record:

- cycle number;
- `cycle_convergence`;
- `consecutive_converged_cycles`;
- all primal and dual residuals;
- `stationarity_pf`;
- every convergence threshold and normalized criterion slack;
- consensus/interface disagreement;
- recourse objective and change from the previous cycle;
- consensus-state and dual-state step norms;
- start/end rho;
- every local NLP termination condition and IPOPT iteration count;
- failed-block count;
- wall-clock time.

Use the existing convergence decision exactly. Visual trends do not constitute
convergence.

## Detached checkpoint polish

At total cycles 25, 50, 75 and 100, or at the first converged cycle, deep-copy
the state and run exact-consensus polish on the copy only. The polish must not
alter the live trajectory.

For each checkpoint report:

- pre-polish interface disagreement;
- polish success/failure and failed blocks;
- polish correction;
- polished recourse and total objective;
- state and configuration hashes.

A successful polish of a non-converged checkpoint does not make it a valid T0.

## Descriptive terminal-window analysis

If the run reaches cycle 100, compare cycles 81-90 with cycles 91-100. For each
principal nonnegative residual and state-step norm compute:

`trend_ratio = median(cycles 91-100) / median(cycles 81-90)`.

Classify each metric descriptively as:

- decreasing: ratio `< 0.8`;
- approximately flat: ratio `0.8 .. 1.25`;
- increasing: ratio `> 1.25`.

Also report minima, maxima and their cycle indices. This classification does not
replace the existing convergence decision.

## Outcome classification

Classify `CONVERGED` only if the unchanged convergence logic terminates by cycle
100 and exact-consensus polish at that state succeeds with zero failed blocks.

Classify `UNRESOLVED` if cycle 100 is reached without convergence, all values
remain finite and no local block fails. Do not call the last state a
self-consistent T0.

Classify `FAILED` if any local block fails, a purportedly converged state cannot
be polished, values become nonfinite, continuation is corrupted or provenance
changes.

## Mandatory execution stops

Stop immediately if:

- a local NLP block fails;
- an objective, residual, consensus or dual value becomes nonfinite;
- rho changes;
- adaptive rho becomes enabled;
- a production formulation or setting changes;
- the trajectory fails to match P5.11 through cycle 25;
- legitimate state history is reset during continuation;
- another rho, tolerance, anchor or initialization would be required.

Do not try a second configuration or numerical fix.

## Required report

Produce:

`P5_12_A_COLD_RESCALED_CONVERGENCE_REPORT.md`

Include:

1. repository and runtime provenance;
2. initial/final branch, HEAD and working-tree state;
3. P5.10 reproduction-gate results;
4. P5.11 cycle-25 state verification;
5. exact code diff;
6. every command executed;
7. complete per-cycle trajectory;
8. convergence thresholds and normalized slacks;
9. detached checkpoint-polish results;
10. terminal-window trend ratios, if applicable;
11. exact stopping reason;
12. evidence that no production file or setting changed;
13. a recommendation for exactly one next stage.

End with exactly one of:

`P5.12-A PASS — cold RESCALED fixed-rho construction converged; planner may authorize a fresh T0 A/B`

`P5.12-A PARTIAL — cold RESCALED construction did not converge within the diagnostic horizon`

`P5.12-A FAIL — cold RESCALED diagnostic encountered a numerical or provenance failure`

Then stop and wait for planner review.

## Later roadmap — not authorized now

If P5.12-A passes, the next possible stage is a fresh inherited-T0 versus
genuinely converged RESCALED-T0 A/B. If that later A/B passes, expanded
candidate/depth coverage, interface-stationarity sensitivity and a separate
`rho_pf = 1000` certification may be considered in sequence.

If P5.12-A remains unresolved, planner review must choose exactly one of:

- one alternative fixed-penalty cold-construction diagnostic; or
- acceptance of the inherited-template oracle with an explicit
  construction-path limitation.

Adaptive-rho redesign and every investment-search run remain deferred in all
P5.12-A outcomes.

---

# RECOVERED ACCEPTED P5.6-C THROUGH P5.9 RESULTS

This summary restores accepted decisions pruned by the repeated
documentation-only merges. It is historical evidence, not active authorization.

## P5.6-C

Accepted verdict:

`P5.6-C-C — derivative-free search is not ready`.

T0/T4 relative landscapes were unstable: Spearman `-0.033`, with 6 of 9
improvement signs reversed. Fixed continuation repaired coverage but changed
direct-VALID controls by about `2.9e6`, so failure-only continuation would mix
incompatible surfaces.

## P5.6-D

Accepted verdict:

`P5.6-D-C — uniform refinement does not stabilize the investment landscape`.

Coverage reached 100% from `K=4`, but ordering remained unstable;
`tau_planning_refined = 811438.05`, the best candidate changed across every
tested depth transition, and a node-9 direction reversed between `K=8` and
`K=12`. No `K_STAR` was accepted.

## P5.7

Accepted verdict:

`P5.7-A — unique operational oracle can likely be recovered`.

Objective scaling was the primary branch-selection mechanism. The production
augmented objective divided the base term by `effective_scale` of roughly
`9.41e4 .. 1.16e5`; equivalent RESCALED local objectives recovered
`-8892463.76` versus `-383401.68` under CURRENT scaling. Initialization and
active-set differences were secondary. No production formulation changed.

## P5.8

Accepted verdict:

`P5.8-B — objective scaling improves stability but additional ADMM issues remain`.

RESCALED objectives were exact positive multiples of CURRENT on all 48 blocks
and improved local optimality/constraint residuals, but the initial replay still
drifted and frequently broke exact-consensus polish. The inherited
`consecutive_converged_cycles` channel and overly loose objective test were
identified. The anchor was shown to remain common across starts.

## P5.9

P5.9 established that voltage rho and ESS consensus were not binding, while the
adaptive active-power penalty rule reduced requested `rho_pf` values 300, 500
and 1000 to approximately 88.89, 98.77 and 131.69. Its CURRENT eight-depth
comparison still had two ranking flips and relative uncertainty `459695.59`
against signal `18085.03`. These findings motivated the explicit fixed-rho,
history-neutralized P5.10 oracle.

No stage in this recovered lineage authorized derivative-free investment search
or a production adaptive-rho change.

---

# HISTORICAL AUTHORIZED STAGE — P5.6-C landscape robustness and oracle coverage gate

P5.4-R completed canonical-environment revalidation. P5.5-D closed the rigorous lower-bound architecture decision and remains accepted as:

`P5.5-D-C — practical rigorous lower-bound architecture is unavailable`.

The continuous-convex/Benders route and MISOCP lower-bound route are retired as the planning architecture. Their code/evidence remains diagnostic/benchmark-only.

Current accepted nonlinear production baseline:

`06e921e5`

Accepted nonlinear decisions remain locked:

- active-energy ESS physics;
- H1 normalized complementarity with `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- net-P/Q-only nonlinear ADMM coordination;
- D2-P sensitivity-clean shared-S local-branch derivative bookkeeping;
- nonlinear AC SMOPF + original nonlinear ESSO retained as the physical feasibility/upper-bound model.

Active development branch:

`feature/derivative-free-planning`

P5.6-A built the deterministic nonlinear candidate-evaluation oracle and remains accepted as:

`P5.6-A PARTIAL — nonlinear oracle works but feasibility, purity, branch stability or computational cost remains unresolved`.

P5.6-B locked a declared direct policy and search specification but exposed unresolved **landscape representativeness and domain coverage**. It is accepted as:

`P5.6-B PARTIAL — oracle policy, anchor robustness, start policy or search resolution remains unresolved`.

P5.6-B settled:

- best certified base-investment incumbent `828021090.360850`;
- direct recurring template `T0`, id `a81f7f5191dd42dbf50d1726149b8909`, frozen by rule rather than fixed-point convergence;
- recurring anchor `MIDPOINT ONLY`;
- recurring direct start `T0 ONLY`;
- `(S,h)` coordinates with `h = E - 2S`;
- deterministic numerical repeatability floor of exactly zero and provisional `tau_numerical = 10.0`;
- deterministic MADS/pattern-search as the preferred search family in principle;
- practical upper-level parallelism ceiling of about four workers.

P5.6-B also found:

- T0→T4 base-template objective drift about `-2.059568e6` with no stabilization;
- direct T0 and cold starts have different failure sets;
- about 27% of the tested operational candidates fail under the locked direct T0 policy, including a budget-boundary crash under both direct starts;
- direct failures cannot be treated as physical infeasibility without further evidence;
- anchor mixing changes rankings and is therefore prohibited in the recurring objective.

P5.6-C is authorized as the **final pre-search gate**. It does not authorize the 200-evaluation investment-search campaign.

The active questions are:

1. does the relative investment landscape remain stable across deterministic templates T0/T2/T4?;
2. can a fixed, history-independent continuation fallback rescue false hidden-infeasible points without mixing incompatible objective surfaces?;
3. what planning-level robustness threshold `tau_planning` follows from cross-template relative-delta variation?;
4. what deterministic batch polling semantics preserve reproducibility under parallelism?;
5. is a trustworthy NOMAD/PyNOMAD/repository MADS implementation available, or should the first campaign use a simpler deterministic generalized pattern search?

# HISTORICAL P5.6-C REPRODUCIBILITY GATE

All paper-instance work must use:

`/opt/anaconda3/envs/opf_env_py311/bin/python`

Canonical SRP1 checksum:

`5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`

Every active harness that loads SRP1 must record:

- `sys.executable`;
- resolved conda environment;
- Python version;
- NumPy/pandas/SciPy versions;
- Pyomo version;
- IPOPT path/version and HSL solver when relevant;
- Gurobi/gurobipy version and licence status when relevant;
- realized scenario checksum.

Abort if the checksum differs. `srp_env` is noncanonical and its D3/D4 numerical evidence is historical only.

---

# ACCEPTED P5.4-R CANONICAL RESULTS

## R1 — nonlinear production regression

- DSO `36/36`;
- TSO `12/12`;
- ESSO `3/3`;
- primary failures / recoveries / persistent failures `0/0/0`;
- H1 complementarity violations `0/1728`;
- converter-capability violations `0/1728`;
- total network IPOPT iterations `3424`;
- mean / median / max `71.3 / 65.0 / 134`;
- runtime about `42 s`;
- representative equality Jacobians full row rank;
- zero zero-gradient ESS equality rows.

Earlier `3442` bootstrap iterations from `srp_env` are noncanonical.

## R2 — canonical D3

Canonical base distributed recourse:

`Q0 = 838496830.813414`

with `17` ADMM cycles and bit-identical repeated evaluations under the same canonical history.

Canonical cut threshold:

`tol_cut = 7.164e5`.

Canonical cut test:

- `2/8` decisive violations;
- worst `cut_gap ≈ -2.680808e6`;
- second decisive violation about `-1.315031e6`;
- all 18 aggregated S/E coefficients negative;
- predicted linear capacity effect remains below recourse-resolution scale.

Accepted verdict:

`P5.4-R-D3 FAIL — canonical nonlinear-recourse cuts are demonstrably unsafe`.

## R3 — canonical D4

Canonical cold base:

`838496830.81`.

Best observed recovered base branch:

`836586463.43`.

Improvement:

about `1.910367e6` (`0.228%`).

The branch fingerprint again localizes mainly to **TSO generation dispatch on Spring representative days**, partly offset by Summer changes.

Canonical hardened-cut test:

- `8/8` tested candidates decisively violate the hardened cut;
- gaps approximately `-0.96e6` to `-1.13e6`;
- all exceed canonical `tol_cut`;
- linear capacity contributions remain only O(`1e4`).

Accepted verdict:

`P5.4-R-D4 C — canonical hardened cuts remain demonstrably unsafe`.

Therefore the nonlinear-recourse derivative-cut/Benders machinery must not be used as a global lower-bound method.

## Environment sensitivity and call-history impurity are separate

Do not conflate:

- different environments generating different stochastic scenarios / different branches;
- persistent in-place data mutation causing call-history dependence.

The first is a reproducibility/environment issue. The second remains an independent software-hygiene issue to fix before nonlinear planning-UB certification.

---

# P5.5-A / P5.5-B STATUS

## P5.5-A

Historical verdict:

`P5.5-A PARTIAL — architecture is promising but unresolved convexity/bound-direction issues remain`.

Accepted inventory:

- master is an LP;
- TSO `case9`: one independent cycle;
- DSO `case33_1/2/3`: radial after preprocessing;
- one controllable continuous OLTC per DSO, range `[0.83,1.17]`;
- no phase shift/discrete tap logic;
- no controllable capacitor bank/switched shunt;
- retain conventional generation P/Q, RES P/Q/curtailment/PF control, active/reactive flexibility, shared ESS P/Q/SOC and interface P/Q;
- ordinary ESS absent in SRP1;
- lifted W-space AC SOC/QC is the intended convex family;
- no DC-OPF or LinDistFlow substitution.

## P5.5-B

Accepted verdict:

`P5.5-B PARTIAL — one or more lower-bound/cut/interface issues remain unresolved`.

This status reflects unmeasured real-model relaxation tightness and incomplete formal cut certification, not a rejected formulation.

### B1 — angle/QC safety accepted

Dormant ±30-degree angle constraints are not part of the production feasible set and are **not** lower-bound safe merely because dormant rules exist. Only class-A production constraints and class-B implications may tighten the certified LB model.

Retain `WijR >= 0`. Do not add ±30-degree limits.

### B2 — transformed OLTC accepted

Use:

`U_i = r^2 * W_ii`

`C_ij = r * W_ij^R`

`D_ij = r * W_ij^I`.

All transformer P/Q equations are affine in `U_i,C_ij,D_ij,W_jj`, verified against production to about `5.9e-15`.

Physical rank relation:

`C_ij^2 + D_ij^2 = U_i * W_jj`.

Convex relaxation:

`C_ij^2 + D_ij^2 <= U_i * W_jj`.

Continuous tap box:

`r_min^2 * W_ii <= U_i <= r_max^2 * W_ii`.

Because `W_ii > 0`, `r = sqrt(U_i/W_ii)` exists in `[r_min,r_max]`. SRP1 has no tap cost, tap-movement penalty, intertemporal coupling, discrete tap positions or phase shift. Therefore `r` and `r_sqr` are eliminated from the convex oracle. The old McCormick proposal is superseded.

### B3 — interface signs accepted; available-capacity coupling added by planner

DSO `REF` is the mathematical TSO import/export representation, not a separately costed physical generator.

The first centralized model must retain separate TSO/DSO/ESSO copies and replace ADMM consensus with exact affine equalities.

P5.5-C confirmed and implemented the planner correction that production passes ESSO **available** capacities into TSO/DSO network models. Centralization therefore includes exact capacity equalities:

`S_TSO_network = S_DSO_network = S_ESSO_available`

`E_TSO_network = E_DSO_network = E_ESSO_available`

with explicit unit/base conversion. Network SOC limits/anchors must use `E_available`, not rated E directly.

### B4 — ESSO lower-bound relaxation accepted

In the LB oracle drop:

- H1 hats/links and charge/discharge complementarity;
- non-negative complementarity objective penalty;
- ESSO degradation and SoH equalities;
- minimum-SoH restriction;
- associated non-negative slack penalties.

Retain:

`0 <= E_available <= E_rated`.

This is jointly affine and lower-bound safe. Weakness is acceptable; unknown-direction tightening is not.

### B5 — salvage planner correction

Current net recourse is gross operating cost minus terminal salvage, so salvage is a **credit**.

For a rigorous lower bound, subtract an **upper bound** on attainable salvage. Under non-negative configured salvage coefficients this maximum occurs at:

`E_available = E_rated`, equivalently `SoH = 1`.

P5.5-C verified the configured coefficient signs and derived:

`V_salvage_max(x) = sum gamma[node,cohort] * E_investment[node,cohort]`

then use:

`-V_salvage_max(x)`

in the planning lower-bound objective. The maximum-credit case is `SoH=1`; `soh_min` is not the safe lower-bound choice.

Salvage is excluded from operational `R(x)`. The safe affine salvage bound has no direct S coefficient.

### B6/B7 — Gurobi prototype evidence accepted, formal cut remains open

Use Gurobi in the canonical environment. Preferred interface:

`gurobi_persistent`.

Toy convex tests verified:

- fixing-row dual matches analytic derivative;
- `QCPDual=1` gives conic duals;
- `ObjVal`-anchored cut can be slightly unsafe at its own anchor;
- `ObjBound`-anchored cut passed a 12-point sweep.

For a scalar LB use `ObjBound`.

For a planning cut, do **not** assume:

`[ObjBound(x_k)-sigma] + g_k^T(x-x_k)`

is formally valid merely because `sigma` exceeds the scalar primal-dual gap. The desired formal contract is one dual-feasible affine function:

`L_k(x) = beta_k + g_k^T x`

with weak-duality proof:

`L_k(x) <= R(x)` for all admissible x.

P5.5-C investigated this on the real convex model and found no usable full-model dual certificate or reliable real-model derivative contract. The planning master remains blocked.

---

# ACCEPTED P5.5-C / P5.5-D RESULTS

## P5.5-C

Accepted verdict:

`P5.5-C PARTIAL`.

The centralized continuous conic oracle is implemented and useful diagnostically, but it has no usable full-model dual certificate and is far too loose for planning integration. The transformed OLTC and radial DSO SOC relaxations are not material gap sources; the meshed TSO is the dominant remaining AC-relaxation defect.

## P5.5-D

Accepted verdict:

`P5.5-D-C — practical rigorous lower-bound architecture is unavailable`.

Authoritative D-stage conclusions:

- the H1 convex hull in `(pch,pdch,S)` is already the active-sum triangle; ~`0.5 S` circulation is a convex-hull midpoint, not a missing continuous convex inequality;
- the lower-bound-safe fixed-mode screen reduces circulation from ~`0.5 S` to ~`0.01 S` but changes the full objective by only about `322532` (`0.049%`) and closes ~`0.42%` of the gap;
- the fixed-mode value remains ~`10.51%` below the polished nonlinear feasible-network reference; because `Q_MISOCP <= Q_fixed_mode`, the unrestricted MISOCP cannot repair the gap;
- no binary stage was run; the potential ~`1728` binary full formulation was rejected before branch-and-bound on a rigorous zero-binary screen;
- stronger TSO QC/SDP work is optional future benchmark work, not the active planning path.

---

# ACCEPTED P5.6-A RESULTS

P5.6-A is accepted as:

`P5.6-A PARTIAL — nonlinear oracle works but feasibility, purity, branch stability or computational cost remains unresolved`.

## A1 — complete nonlinear feasibility certificate

Canonical START-1 base candidate:

- max coordinated mismatch `5.551e-17`;
- original nonlinear ESSO max constraint violation `2.035e-13`;
- largest network constraint residual `1.093e-05 p.u.` at `DSO5|2030|Winter`, treated as the local IPOPT feasibility tolerance;
- gross operational cost `829291677.522120`;
- physical salvage `3439.659877`;
- net operational recourse `829288237.862242`;
- investment cost `50000.000000`;
- total planning objective `829338237.862242`.

The old P5.5-D `729.361e6` figure is superseded: P5.6-A proved that ~`95.5%` of the apparent 13% gain came from an invalid polish convention that erased TSO flexibility cost. Corrected polishing gives a genuine ~`9.21e6` (`1.0982%`) improvement over the ADMM-evaluated branch, mainly through slack/flexibility cleanup.

## A3/A4/A7 — purity, contract and caching

Per-evaluation deep copies, isolated solver-log paths and reset diagnostics close the A12 call-history mutation issue for the new oracle. The same base candidate is reproduced bit-identically across different call histories.

`evaluate_planning_candidate(...)` now checks first-stage feasibility before operational solves, executes nonlinear ADMM + original nonlinear ESSO + exact-consensus polish + full feasibility audit, returns explicit failure statuses, and caches only VALID completed results.

The canonical base lies at the minimum E/S ratio `E=2S`, so negative E-only search moves are first-stage infeasible.

## A5/A6 — start and runtime evidence

START-2 uses one fixed archived base-candidate template and is materially better/cheaper than cold on the tested candidates:

- START-1 cold base: `829338237.862242`, ~`562 s`, `17` ADMM cycles;
- START-2 template base: `828021090.3608505`, `2` ADMM cycles; one-off build included in the first ~`603 s` evaluation, then ~`100 s` on subsequent candidates;
- START-2 beats cold by roughly `0.64e6` to `1.32e6` on all three dual-start candidates;
- cold objective span across the three is ~`674337`; template span is ~`2024`.

Historical P5.6-A note, superseded by P5.6-B: the A report defined `Q_oracle` as the minimum over both starts, which would cost both solves. P5.6-B subsequently locked the recurring direct search start to `T0 ONLY`; cold is now final-certification / periodic-audit only. The ~`115 s` recurring successful-evaluation cost therefore refers to T0-only evaluation.

## A6 — interface-anchor issue

Midpoint polishing failed on one of four benchmark candidates. DSO-side anchoring repaired it. On the base candidate, DSO anchoring changes the total objective by `13053.97`, equal to `0.018 * tol_cut` but larger than the observed ~`2024` template-started candidate span.

Do not use the old Benders `tol_cut` as the derivative-free search-resolution threshold. P5.6-B subsequently established exact numerical repeatability and proposed `10.0`; P5.6-C renames this `tau_numerical = 10.0` and must derive a separate `tau_planning` from cross-template relative landscape variation.

---

# ACCEPTED P5.6-B RESULTS

P5.6-B is accepted as:

`P5.6-B PARTIAL — oracle policy, anchor robustness, start policy or search resolution remains unresolved`.

The verdict wording is historical; planner interpretation is more specific: midpoint-only anchor and T0 direct start are locked, while template **landscape representativeness**, false hidden-infeasibility coverage and planning-level branch uncertainty remain unresolved.

## B0/B1 — best incumbent and template chain

Current best rigorously feasible nonlinear planning incumbent at the canonical base investment:

- total planning objective `828021090.360850`;
- net operational recourse `827971090.360850`;
- gross operational cost `827974518.717105`;
- physical salvage `3428.356255`;
- investment cost `50000.000000`.

Template refinement on the same base candidate:

- T0 `828021090.360850`;
- T1 `827415318.563944`;
- T2 `826824028.845478`;
- T3 `826405022.193437`;
- T4 `825961521.882321`.

No stabilization is established. T0 is frozen by an explicit reproducibility rule, not because it is a fixed point. T0 id:

`a81f7f5191dd42dbf50d1726149b8909`.

Purity under T0 remains bit-identical across different call histories.

## B2 — midpoint anchor locked

On the 12-candidate benchmark population, midpoint and DSO anchors have identical success rate on operationally evaluated candidates (`8/11 = 72.7%`). Under T0 the DSO anchor rescues no midpoint failure. Anchor-induced objective changes range approximately `-13365.80 .. +11175.98` and can reverse candidate orderings.

Recurring search anchor:

`MIDPOINT ONLY`.

Both anchors remain available only for final incumbent certification.

## B3 — T0 direct start locked, coverage incomplete

Cold is never better than T0 on objective where both succeed, by about `1.32e6 .. 1.09e7`, so the recurring direct search start is:

`T0 ONLY`.

Cold is final-certification / periodic-audit only.

However, T0 and cold have different failure sets. At least one T0 `POLISH_FAILURE` candidate is VALID under cold, and the budget-boundary candidate crashes under both. Therefore operational failure is a hidden solver/branch-access issue unless physical infeasibility is separately proved.

## B4 — numerical threshold

Same-candidate repeatability under the locked T0+midpoint policy is exactly zero. P5.6-B proposed:

`tau_search = 10.0`.

For P5.6-C terminology this becomes:

`tau_numerical = 10.0`.

`tau_planning` is still pending and must be derived from cross-template **relative-to-base** landscape variation. Do not use historical Benders `tol_cut` as a derivative-free resolution criterion.

## B5/B6 — coordinates and search family

Use `(S,h)` with:

`h = E - 2S >= 0`, `E = 2S + h`.

This is a bijective change of variables on the active minimum-duration relation. Remaining max-duration, cumulative-capacity, lifetime and budget constraints are exact first-stage checks; do not project requested candidates silently.

Deterministic MADS/pattern search remains the selected family in principle. OrthoMADS is preferred conceptually, but no search is authorized until P5.6-C audits actual implementation availability and deterministic batch semantics.

## B7 — cost

Under T0 + midpoint:

- successful uncached evaluation ~`115 s`;
- blended observed cost ~`134 s`;
- final multi-start/both-anchor certification ~`740 s`;
- one-off T0 construction ~`518 s`;
- cache hit ~`0.040 s`.

Four workers are recommended. Eight workers risk resource contention and have already coincided with IPOPT process crashes.

---

# HISTORICAL SOLVER / ORACLE POLICY FOR P5.6-C

Use only the accepted nonlinear production solvers and settings. Do not retune IPOPT/MA97, ADMM, H1, proximal regularization, recovery logic or adaptive rho during C.

For direct recurring evaluations, the declared baseline policy is:

- template/start: `T0 ONLY`;
- T0 id: `a81f7f5191dd42dbf50d1726149b8909`;
- anchor: `MIDPOINT ONLY`;
- coordinates: `(S,h)`, `h = E - 2S`;
- numerical comparison threshold: `tau_numerical = 10.0`;
- cache key includes the exact template id and anchor convention.

Do not interpret a T0 failure as physical infeasibility. P5.6-C must test deterministic continuation coverage before any search uses an extreme barrier for these points.

Cold and alternate anchors remain available for final certification/diagnostics only unless P5.6-C explicitly changes the recurring policy.

---

# P5.6-C — LANDSCAPE ROBUSTNESS AND ORACLE COVERAGE GATE

This is the final gate before any derivative-free optimization run.

## C1 — correct the P5.6-B interpretation

Record that P5.6-B settled midpoint-only anchoring and T0 direct start. The unresolved issues are:

- cross-template investment-landscape representativeness;
- domain coverage / false hidden infeasibility;
- planning-level branch uncertainty;
- deterministic parallel poll semantics;
- practical MADS/pattern-search implementation path.

## C2 — cross-template landscape test

Use the fixed P5.6-B master-feasible benchmark population. Evaluate jointly reachable candidates under:

`T0`, `T2`, `T4`

with midpoint anchor only and the complete original nonlinear oracle contract.

For every candidate/template compute:

`Delta_Tk(x) = Q_Tk(x) - Q_Tk(base)`.

Across jointly VALID candidates report:

- Spearman rank correlation;
- Kendall rank correlation;
- complete ranking;
- maximum rank displacement;
- pairwise ordering reversals and worst reversal magnitude;
- `Delta_T0`, `Delta_T2`, `Delta_T4` per candidate;
- per-candidate delta range/std across templates;
- diagnostic affine fits between template objective levels.

Do not treat high correlation as sufficient if improvement over the base changes sign materially.

Classify LANDSCAPE-STABLE only if relative improvement signs/rankings are sufficiently preserved and the template effect is predominantly a common offset. If stable, keep T0. If unstable, do not launch MADS.

## C3 — deterministic continuation fallback

For each master-feasible target `x`, define the straight feasible first-stage path from canonical base `x0`:

`x(lambda) = (1-lambda)*x0 + lambda*x`.

Use fixed schedule only:

`lambda = 0.25, 0.50, 0.75, 1.00`.

Starting from the frozen T0/base state, solve each point in order. A complete VALID state may initialize the next point. Every point must run:

- original nonlinear ESSO;
- production ADMM;
- midpoint-only exact-consensus polish;
- full physical feasibility audit.

No adaptive lambda schedule and no anchor switching in C.

Test on:

- every T0/midpoint `POLISH_FAILURE` from B;
- the budget-boundary `SOLVER_CRASH` candidate;
- at least two direct-T0 VALID controls.

For controls compare direct and continuation objectives to determine whether continuation is a neutral coverage rescue or a materially different branch-selection surface.

## C4 — lock coverage policy

Compare:

A. direct T0 only;

B. direct T0, fixed continuation only on direct failure;

C. fixed continuation for every candidate.

Prefer B only if continuation improves coverage and gives compatible objectives on direct-valid controls. If continuation materially changes valid-control objectives, either use C for every candidate or return PARTIAL. Never mix incompatible surfaces silently.

Recompute success rate and ordinary/failed/blended cost under the chosen policy.

## C5 — define `tau_planning`

Keep:

`tau_numerical = 10.0`

unless new repeatability evidence requires otherwise.

Define separately:

`tau_planning`

from cross-template relative-to-base delta variation. This threshold describes robustness of investment improvements to known branch/template ambiguity. Do not include absolute common template offsets if rankings/deltas are stable.

Future search may use `tau_numerical` internally on the locked surface, but incumbent promotion and scientific claims must respect `tau_planning`.

## C6 — deterministic parallel polling

Future parallel search must be deterministic batch polling, never asynchronous first-finish opportunism:

1. generate ordered deterministic poll set;
2. first-stage feasibility screen;
3. remove duplicate/cache-hit points;
4. select batch deterministically;
5. evaluate up to four points in parallel;
6. wait for the complete selected batch;
7. choose the best VALID objective;
8. break ties within `tau_numerical` by deterministic poll index;
9. only then update incumbent/mesh.

This requirement preserves reproducibility under variable solver completion times.

## C7 — implementation inventory

Without installing anything, audit:

- NOMAD / PyNOMAD availability;
- repository pattern/MADS utilities;
- licence compatibility;
- hidden-constraint evaluator support;
- exact first-stage feasibility screening;
- deterministic direction/seed control;
- cache integration;
- deterministic batch/parallel evaluation.

Do not hand-code “OrthoMADS” casually. If a trustworthy MADS implementation is unavailable or disproportionate to integrate, a deterministic positive-spanning generalized pattern search is acceptable for the first campaign with limitations stated explicitly.

## C8 — no optimization run

Do not run the 200-evaluation campaign during P5.6-C.

## C9 — required verdict

End one:

`P5.6-C-A — derivative-free search is ready to launch`

or

`P5.6-C-B — derivative-free search is ready only on a restricted validated domain`

or

`P5.6-C-C — derivative-free search is not ready`.

For C-B, explicitly define the validated local domain from tested successful directions. For C-A, state the final recurring oracle/coverage policy, `tau_numerical`, `tau_planning`, deterministic poll semantics and selected implementation path.

Then:

`P5.6-C COMPLETE — ready for planner review before any optimization campaign`.

---

# LOCKED PRODUCTION DECISIONS DURING P5.6-C

Do not change:

- nonlinear AC SMOPF equations;
- active-energy ESS formulation;
- D2-P sensitivity-clean shared-S formulation;
- H1 dimensionless complementarity and `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`;
- net-P/Q-only nonlinear ADMM coordination;
- IPOPT tolerances/options;
- MA97/exact-Hessian policy;
- recovery/adaptive-rho/proximal policies;
- existing master/Benders equations;
- P5.5 convex/MISOCP diagnostic code.

Do not yet implement:

- derivative-free investment optimization;
- replacement outer planning loop;
- production MISOCP planning;
- distributed convex ADMM;
- QCP/Benders cut recovery;
- TSO SDP/QC strengthening.

---

# OLD P5.4-G STATUS

P5.4-G is permanently blocked for the old nonlinear-recourse derivative cuts. Do not run it.

Any later planner is a new architecture.

---

# DEFERRED ITEMS

Keep deferred during P5.6-C:

- physical complementarity tolerance `1e-5` / `1e-6` A/B;
- B1 exact `f_ref=0`;
- RES B2-R until defensible converter `Smax` data exists;
- nonlinear solver/ADMM retuning;
- distributed convex ADMM;
- full/production MISOCP planning;
- QCP/Benders cut recovery;
- TSO SDP/QC strengthening except as optional future diagnostic benchmark;
- actual derivative-free optimization run;
- surrogate-assisted planning unless later selected explicitly;
- any change to accepted nonlinear model mathematics.

# COMPLETED STAGE — P5.3 (historical execution record)

This historical section superseded the older frozen-cycle-only instructions during P5.3. It no longer defines the active scope; P5.4 above is authoritative.

The historical frozen-model work remains valid evidence and is summarized later in this file, but the primary P5.3 population is now the real current SRP1 positive-bootstrap cold initialization produced through the production candidate/model-construction path.

Current accepted production checkpoint:

`f77d829359ffd873367f556882546bc2dcc8ec99`

Current reduced-scenario identity:

- seed `2026`;
- combined scenario checksum:
  `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`.

P5.3 was a diagnostic/reformulation review. Its B3 direction has now received planner approval for controlled P5.4 productionization, not direct acceptance of the diagnostic wrapper.

## Current P5.3 invariants

Do not:

- tune IPOPT `tol`, `acceptable_tol`, `acceptable_iter`, `max_iter`, or pushes;
- switch production MA97/MA57 policy;
- change recovery classification;
- change ADMM rho rules or tolerances;
- change recourse-stationarity criteria;
- change ADMM objective scaling;
- change TSO proximal regularization;
- change Benders/local-cut logic;
- add generic feasibility slacks;
- productionize the P5.2 shared-ESS narrow band;
- cap shared-ESS `kappa`;
- change the stochastic scenario values during exact RES algebra A/B tests;
- add calendar degradation;
- modify terminal salvage;
- silently change ESS complementarity tolerance.

The active-power ESS network prototype described below is explicitly authorized as a diagnostic physical reformulation. Full ESSO cycling-degradation conversion is not part of P5.3-B3.

---

# P5.3 checkpoint after A/A2 — AUTHORITATIVE

P5.3-A and P5.3-A2 are complete. The detailed A/A-RES instructions retained below are historical execution records and **must not be rerun wholesale** unless a later B experiment explicitly requests a targeted derivative control.

Authoritative findings:

- full equality Jacobians are exactly row-rank deficient at the positive-bootstrap cold start because of `sess_snet_def` only;
- zero equality rows: 24 per DSO model, 72 per TSO model;
- full `sigma_min = 0` and full condition number is formally infinite;
- after removing only the exactly-zero rows, the tested reduced equality Jacobians have full row rank and no additional nullity;
- corrected reduced condition numbers are approximately `8.98e4` for DSOs and `1.42e3` for the TSO;
- the previous claim that TSO equality conditioning is worse is withdrawn;
- the previous near-zero `pij/qij` DSO column result is withdrawn as a derivative-audit artifact caused by uninitialized `r_sqr` rows being skipped;
- `sess_snet_def` remains HIGH risk: exact zero gradient, always-active equality, sole source of exact equality-rank deficiency, curvature up to about `18806`;
- `sess_comp` remains HIGH risk: tiny Jacobian and tiny `S_rated^2`-scaled margin;
- `sg_capability` remains HIGH risk: 3732 active rows, zero cold-start margin, gradients down to about `5.44e-5`;
- all 144 current SRP1 curtailable RES instances have `power_factor_control=True`; therefore `gen_pf_profile` is never instantiated and the old exact RES B2 cleanup is cancelled for SRP1;
- current RES synthetic `q_available` is zero, so `sg_available = P_available` and the capability circle collapses with stochastic active availability;
- RES `abs()` reflection is negligible; upper-support overshoot is material, reaching about `33.5%` above historical maxima in some season/type calls;
- the realized reduced SRP1 population has no values in `(0,1e-5]`, but 17 live values in `(1e-5,1e-4]`;
- cross-generator spatial correlation is not preserved by the current same-type pooled sampling assignment;
- DSO interface magnitude is intentionally freed in ADMM; the earlier claim of persistent DSO magnitude pinning is withdrawn;
- IPOPT internally scales the objective gradient but does not supply comparable constraint-row scaling.

## Historical B-series execution order (completed)

The user deliberately selected the following order because the RES capability experiment is easier and quicker to debug and validate than the active-power ESS refactor:

1. **P5.3-B1 — exact reference-angle gauge**;
2. **P5.3-B2-R — RES capability semantics and conditioning**;
3. **P5.3-B3 — active-power ESS structural prototype**.

B3 remains the highest expected structural-payoff experiment, but it is executed third for practical validation reasons.

Do not test more shared-ESS epsilon values or scalar `kappa` caps. Do not retune solver options.

---

# P5.3-A — Quantitative structural conditioning audit (COMPLETE — historical execution instructions)

## A1. Reproduce the real positive-bootstrap population

Generate the exact P5 iteration-2 positive-bootstrap candidate using production:

`_build_positive_bootstrap_candidate(...)`

and replay the real pre-solve initialization path.

Audit every network SMOPF immediately before IPOPT:

- 36 DSO models;
- 12 TSO models.

Retain ESSO models for structural context, but the primary derivative audit is the 48 network SMOPFs.

Do not reconstruct the bootstrap candidate manually.

The old frozen cycle-10 pickle may be used only as historical/control evidence.

## A2. Inspect the installed derivative environment first

Determine the safest available mechanism for evaluating the NLP Jacobian and related derivatives.

Prefer analytic Pyomo/PyNumero/NLP interfaces already available in the environment.

Do not introduce a large external dependency merely for this audit.

If an exact sparse Jacobian interface is unavailable, report that limitation before implementing a materially different numerical approach.

## A3. Constraint-family inventory

Enumerate every active network constraint component and classify it as:

- linear equality;
- linear inequality;
- nonlinear equality;
- nonlinear inequality;
- ranged nonlinear inequality.

For every family report:

- row count;
- polynomial degree where available;
- variables participating;
- typical variable magnitude;
- whether variables/row residuals can naturally be near zero;
- whether a row can be active at a zero-gradient point;
- whether stochastic data can switch the row on/off structurally.

Explicitly include at least:

- `voltage_mag_def`;
- `voltage_mag_sqr_def`;
- `voltage_setpoint_cons`;
- `voltage_product_real_def`;
- `voltage_product_imag_def`;
- `r_sqr_def`;
- `sg_capability`;
- `gen_pf_upper`;
- `gen_pf_lower`;
- `gen_pf_profile`;
- `flex_energy_balance_p`;
- `ess_pnet_def`;
- `ess_snet_def`;
- `ess_comp`;
- `ess_soc_def`;
- `sess_pnet_def`;
- `sess_snet_def`;
- `sess_comp`;
- `sess_soc_def`;
- `node_balance_p`;
- `node_balance_q`;
- `branch_flow_limit`;
- `branch_flow_limit_ji`;
- all distributed coordination/interface constraints added after base model construction.

## A4. Jacobian diagnostics at the cold start

For every active row, where technically available, compute:

- absolute constraint residual / violation;
- `||grad g||_2`;
- `||grad g||_inf`;
- smallest nonzero absolute Jacobian coefficient;
- largest absolute Jacobian coefficient;
- intra-row coefficient ratio;
- distance to the nearest inequality bound;
- `distance_to_bound / IPOPT_tol` using that model's actual configured `tol`.

Do not hide zero-gradient rows.

Report counts of rows with gradient norm below:

- `1e-12`;
- `1e-10`;
- `1e-8`;
- `1e-6`;
- `1e-4`.

These are diagnostic bins only.

Group by component, agent/network, year, and representative day.

Produce ranked top-N summaries for:

- smallest row norms;
- largest row norms;
- smallest inequality margins relative to IPOPT tolerance;
- largest intra-row derivative-scale ratios.

## A5. Jacobian column and near-dependence diagnostics

Report:

- zero/near-zero derivative columns;
- smallest/largest column norms;
- suspicious variable families.

Where practical, estimate:

- equality-Jacobian numerical rank;
- smallest singular values;
- largest singular value;
- a condition estimate.

At minimum do this for:

- one representative TSO model;
- one representative model from each DSO;
- every previously sensitive/failing positive-bootstrap state.

If full SVD is impractical, use a sparse extremal-singular-value or rank-revealing alternative and state the limitation.

Also identify pairs/groups of equality rows with nearly collinear normalized gradients.

## A6. Constraint-curvature audit

For nonlinear quadratic/bilinear rows report raw second-derivative/Hessian coefficient scales where practical.

Pay particular attention to:

- `sess_snet_def`;
- `ess_snet_def`;
- `sess_comp`;
- `ess_comp`;
- `sg_capability`;
- branch-current/apparent-power limits;
- squared-voltage/product definitions.

For shared ESS explicitly evaluate the bootstrap power ratings:

- `1.0635e-4 p.u.`;
- `2.1270e-4 p.u.`;
- `3.1905e-4 p.u.`;

with current:

`kappa = 1/S_rated`.

Distinguish clearly among:

- small Jacobian;
- large curvature;
- tiny inequality margin;
- exact rank degeneracy;
- near-linear dependence.

Do not collapse them into one generic "bad conditioning" label.

## A7. Objective-gradient scale

At the cold start report objective-gradient norms/coefficient ranges for major components that are present:

- physical/economic SMOPF objective;
- scenario-deviation regularization;
- proximal terms;
- ADMM augmentation.

Compare objective-gradient scale with suspicious constraint-Jacobian scales.

Do not change objective scaling.

---

# P5.3-A-RES — Stochastic RES and low-output audit (COMPLETE — historical execution instructions)

The current load and RES realizations come from historical-data-based copula/KDE models. Do not replace the copula model in this stage.

## RES1. Raw support and `abs()` post-processing

Instrument RES generation so the inverse-transformed samples are inspected **before**:

`np.abs(...)`.

For each network / season / RES type (PV, Wind) report:

- number of sampled hourly values;
- number/percentage negative before `abs`;
- minimum negative value;
- total/magnitude of positive generation created solely by reflection of negative samples;
- values above historical maximum;
- values below historical minimum;
- relevant quantiles before and after post-processing.

Do not alter `np.abs` during this audit.

## RES2. Tiny-generation population in the realized SMOPFs

After conversion to p.u. and after the actual one-scenario realization is selected, count available RES values in:

- exactly zero;
- `(0, 1e-6]`;
- `(1e-6, 1e-5]`;
- `(1e-5, 1e-4]`;
- `(1e-4, 1e-3]`;
- `>1e-3`.

For tiny values report exact:

- network;
- generator id/type;
- year;
- day;
- hour.

Cross-reference them with:

- `renewable_generation_is_unavailable`;
- `sg_capability`;
- `gen_pf_profile`;
- PF-control rows;
- local solve failures;
- unusually high IPOPT iteration counts where available.

Quantify how often the current structural switch at `EQUALITY_TOLERANCE` is exercised or nearly exercised.

Do not select a replacement threshold yet.

## RES3. Reactive-power/profile assumptions

For every curtailable RES generator report:

- `power_factor_control`;
- `min_pf`, `max_pf`;
- physical `pmax`, `qmin`, `qmax`;
- stochastic `pg_available`;
- stochastic `qg_available`;
- whether `qg_available` is identically zero;
- whether `gen_pf_profile` is instantiated;
- whether `sg_capability` is instantiated.

Explicitly identify rows where:

`q_available * pg == p_available * qg`

reduces to:

`p_available * qg == 0`

with very small positive `p_available`.

## RES4. Apparent-power capability interpretation

Determine whether current stochastic `sg_available = sqrt(pg_available^2 + qg_available^2)` is intentionally representing converter MVA rating or only stochastic renewable availability.

If `qg_available = 0`, the current capability radius collapses to `pg_available`.

Report whether the data/model contains a separate physical inverter MVA rating that could support a cleaner separation:

- stochastic active availability: `0 <= pg <= P_available`;
- converter capability: `pg^2 + qg^2 <= S_converter^2`.

Do not implement this physical change in the exact RES B2 experiment unless separately approved; it may change the feasible set.

## RES5. Spatial/scenario correlation observation

Audit how generated PV/wind profiles are assigned to individual generators.

Determine whether generator-to-generator spatial correlation is preserved or whether same-type generators effectively draw independently from a common synthetic pool.

Report this as a scenario-realism finding, separate from local NLP conditioning.

Do not change the sampling architecture in P5.3.

---

# P5.3-A-extra — Other structural checks

## Reference-angle/gauge audit

Confirm current reference-bus treatment of `f` and quantify any residual rotational/gauge degree of freedom.

Also audit DSO reference-bus `e` bounds and determine whether they effectively pin the coordinated interface voltage despite `enforce_vg = false`.

Do not change them in Phase A.

## Transformer auxiliary audit

Determine whether `r` and `r_sqr` variables constructed for non-transformer branches enter the generated NL problem.

If the writer eliminates them, classify as code cleanliness only.

If they reach IPOPT as unused/weakly-connected variables, classify as a conditioning issue.

## Branch-current audit

Quantify cancellation and derivative scales in current-limited branch rows based on:

`V_i^2 + V_j^2 - 2*W_ij_real`

multiplied by branch series-admittance magnitude squared.

Report whether low-impedance DSO branches create extreme coefficients or cancellation-sensitive constraints.

Do not reformulate branch currents unless this family ranks materially high.

---

# P5.3-A required output

Produce:

`P5_3_A_SMOPF_CONDITIONING_AUDIT.md`

Include a ranked `HIGH / MEDIUM / LOW` table for suspicious formulation families.

For every HIGH-risk item state:

- mathematical form;
- observed numerical evidence;
- physical role;
- exact failure mode: zero gradient, poor scale, near dependence, tiny margin, large curvature, etc.;
- possible exact reformulation;
- possible deliberate physical reformulation;
- expected numerical benefit;
- risk of changing the feasible set.

P5.3-A/A2 are complete. Do not rerun the full audit before B1/B2-R/B3; use only targeted derivative controls required by the specific experiment.

---

# P5.3-B1 — Exact reference-angle A/B

Proceed only if Phase A confirms that fixing the reference imaginary voltage does not violate an intentional interface convention.

Start from fresh accepted production models.

Do not carry any P5.2 narrow-band ESS change into this branch.

A — current production reference treatment.

B — exact gauge:

`f_ref = 0`.

Change only this fixing/bound condition.

Keep every other production equation and solver option unchanged.

Run the complete positive-bootstrap initialization.

Report:

- 51 final local outcomes;
- 48 network primary/recovery outcomes;
- IPOPT iterations;
- KKT metrics;
- objective values;
- failures by identity;
- equality-Jacobian rank/smallest-singular-value change on representative models;
- interface V/P/Q differences.

Acceptance:

- no material physical/economic change;
- no new failure family;
- gauge ambiguity removed;
- conditioning not worse.

Diagnostic only. Do not productionize automatically.

---

# P5.3-B2-R — RES capability semantics and conditioning

Run this **second**, after B1 and before B3.

Use a fresh accepted production baseline. Do not stack B1.

The old B2 exact fixed-PF/profile cleanup is cancelled for SRP1 because all 144 curtailable SRP1 RES instances have `power_factor_control=True`, so `gen_pf_profile` is never instantiated.

B2-R is a semantics-first diagnostic. It must not invent a converter rating.

## B2-R.1 Rating semantics first

For every curtailable SRP1 RES generator inspect:

- `pmax`;
- `pmin`;
- `qmax`;
- `qmin`;
- `min_pf`, `max_pf`;
- generator type;
- any explicit `S_rated`, inverter rating, converter rating, nameplate MVA field, or equivalent metadata;
- network JSON comments/metadata;
- historical operational-data units/meaning.

Determine whether the repository contains an **explicit defensible converter apparent-power rating**.

Do not infer `S_converter` merely because `pmax` and `qmax/qmin` exist unless their documented semantics make that inference unambiguous.

If no defensible converter MVA rating exists, STOP B2-R before formulation implementation and recommend the minimum data-model extension required.

Do not synthesize a rating from an arbitrary heuristic.

## B2-R.2 Explain the current RES feasible set

Current production effectively has:

`0 <= pg <= P_available`

plus:

`pg^2 + qg^2 <= S_available^2`

where:

`S_available = sqrt(P_available^2 + Q_available^2)`.

For the current synthetic SRP1 RES data:

`Q_available = 0`,

so:

`S_available = P_available`.

Explain mathematically, together with the active PF cone, what reactive-power capability remains at:

- `pg = P_available`;
- partial active dispatch;
- very low `P_available`;
- `P_available = 0`.

State whether the current behavior appears physically intentional or is likely a conflation of stochastic resource availability with converter/inverter MVA capability.

## B2-R.3 Conditional physical prototype

Proceed only if B2-R.1 finds an explicit defensible `S_converter`.

Create an isolated diagnostic formulation:

stochastic resource availability:

`0 <= pg <= P_available`

converter capability:

`pg^2 + qg^2 <= S_converter^2`

while retaining the existing PF-control inequalities and all stochastic scenario values unchanged.

This is a deliberate feasible-set change, not an exact algebraic rewrite.

Do not change:

- PF limits;
- RES-off threshold;
- stochastic samples;
- objective penalties;
- IPOPT settings;
- ADMM/planning settings.

## B2-R.4 Numerical and physical comparison

If the conditional prototype is implemented, compare production vs B2-R for:

- all 17 realized low-output rows in `(1e-5,1e-4] p.u.`;
- representative normal-output rows;
- all 48 network positive-bootstrap initialization solves;
- final 51 local outcomes where the unchanged ESSO initialization is relevant.

Report:

- `sg_capability` cold-start margin;
- row-gradient norm;
- whether the capability row is active at the cold start;
- primary/recovery/persistent failure counts;
- IPOPT iteration distribution and runtime;
- P/Q capability and dispatch;
- RES curtailment;
- local objective;
- interface V/P/Q;
- any new physical degrees of freedom introduced by separating availability from converter rating.

Determine whether the tiny active capability circle at low stochastic generation disappears when the converter radius is based on a fixed physical rating.

## B2-R.5 Stochastic-support recommendation — diagnostic only

Do not change the copula/KDE generator in B2-R.

Based on P5.3-A2, provide a separate recommendation on:

- handling of synthetic samples above physical/historical support;
- whether post-generation clipping to an explicit physical/nameplate maximum is justified;
- whether the marginal model itself should be bounded;
- how generator-site/spatial dependence should be represented in a later scenario-model revision;
- whether the current negligible `abs()` reflection still merits cleanup for physical clarity even though it is not a material numerical driver.

Keep stochastic-model recommendations separate from the NLP capability result.

## B2-R acceptance

Classify B2-R as one of:

- `PRODUCTIONIZE CANDIDATE` — explicit converter rating exists and the separated formulation improves physical semantics/conditioning without unacceptable regressions;
- `CONTINUE TESTING` — data/formulation is promising but evidence is incomplete;
- `DEFER — DATA MODEL REQUIRED` — no defensible converter rating exists;
- `REJECT` — reformulation is unsupported or creates unacceptable behavior.

Do not productionize automatically.

---

# P5.3-B3 — Active-power ESS structural prototype

Run this **third**, after B1 and B2-R. It remains the highest expected structural-payoff experiment, but is intentionally sequenced last because it is the largest refactor to debug and validate.

Use another **fresh accepted production baseline**.

Do not stack B1 or B2-R.

This is an authorized diagnostic physical reformulation, not an exact algebraic rewrite.

## B3.1 Trace all affected consumers first

Before changing a diagnostic branch, trace every network/coordination/ESSO consumer of:

- `sch`;
- `sdch`;
- `pch`;
- `pdch`;
- `pnet`;
- `qnet`;
- SOC;
- complementarity;
- converter limits;
- result processing;
- ADMM shared-ESS P/Q coupling;
- degradation/throughput.

Confirm that network-agent coordination is based on aggregate P/Q schedules and identify every place that would break if `sch/sdch` were removed.

## B3.2 Network SMOPF prototype

For ordinary and shared network ESS prototype:

`pnet = pch - pdch`

SOC from active power:

`SOC_t = SOC_{t-1} + eta_ch*pch*Delta_t - pdch*Delta_t/eta_dch`

Use the actual model time basis for `Delta_t`; do not assume it silently.

Converter capability:

`pnet^2 + qnet^2 <= S_rated^2`.

Reactive power remains converter loading but does not directly change stored battery energy.

Remove from the network diagnostic prototype:

- `sch`;
- `sdch`;
- `ess_snet_def`;
- `sess_snet_def`;
- `ess_pch_link`;
- `ess_pdch_link`;
- `sess_pch_link`;
- `sess_pdch_link`.

Replace the old apparent-power sum limit with explicit active-power bounds/sum constraints justified from the original device rating and intended physical behavior.

Complementarity acts on:

`pch * pdch`.

Do not silently change the configured complementarity tolerance.

Audit the new complementarity row's scale explicitly. If it remains a high-risk tiny bilinear inequality, report that rather than hiding it with arbitrary scaling.

Shared-ESS zero-capacity gating must remain safe.

## B3.3 Required physics tests

Demonstrate:

- pure Q: `pch = pdch = pnet = 0`, `qnet != 0` leaves SOC unchanged;
- pure charging changes SOC according to efficiency and time step;
- pure discharging changes SOC according to efficiency and time step;
- reactive power remains feasible up to converter capability;
- simultaneous active charging/discharging respects the intended complementarity semantics;
- zero-capacity shared ESS remains completely inactive;
- ordinary/shared P/Q sign conventions remain consistent with nodal balance and exported results.

## B3.4 Numerical tests

Run the complete positive-bootstrap **network initialization** and compare against current production:

- primary failures;
- recovery attempts;
- persistent failures;
- iterations;
- zero/near-zero Jacobian-row counts;
- smallest singular values / rank estimates where available;
- suspicious curvature coefficients;
- objective;
- P/Q schedules;
- SOC trajectories.

Do not enter ADMM with a partially converted ESS formulation unless all required ESSO/state-mapping dependencies are implemented consistently.

Do not rewrite ESSO cycling degradation in B3.

If the network prototype is favorable, produce a precise follow-on end-to-end plan covering:

- ESSO per-cohort `pch/pdch`;
- aggregate P/Q coordination;
- active/cell-side throughput;
- cycling degradation;
- SoH;
- sensitivities;
- state mapping;
- result exports.

Calendar degradation remains out of scope.

---

# Isolation and commit discipline

B1, B2-R, and B3 are independent A/B experiments.

Each begins from the same accepted current production baseline.

Do not accumulate B1 into B2-R, B2-R into B3, or otherwise stack favorable diagnostic changes during P5.3-B.

Diagnostic scripts/tests may be added, but production source files must be restored to the accepted baseline at the end of each diagnostic branch unless a later planner instruction explicitly authorizes a production commit.

Do not combine production edits from this stage into existing accepted commits.

---

# P5.3-B required report

Produce:

`P5_3_B_REFORMULATION_REPORT.md`

with separate sections for:

1. B1 — exact reference-angle gauge;
2. B2-R — RES capability semantics/conditioning, or the documented data-model stop if no defensible converter rating exists;
3. B3 — active-power ESS structural prototype.

For each candidate give one of:

`PRODUCTIONIZE / CONTINUE TESTING / REJECT / DEFER`.

The report must include, where applicable:

- before/after NLP dimensions;
- primary/recovery/persistent failure counts;
- IPOPT iteration/runtime comparisons;
- Jacobian rank/singular-value/curvature evidence;
- physical-equivalence statement for exact changes;
- deliberate-feasible-set-change statement for physical reformulations;
- interface V/P/Q effects;
- objective/dispatch effects;
- stochastic-data implications kept separate from NLP formulation implications.

Answer explicitly:

- Should `f_ref = 0` be adopted?
- Does B2-R confirm that current RES capability conflates stochastic availability with converter MVA rating?
- Is there sufficient data to reformulate RES converter capability safely?
- If B2-R is blocked by missing rating data, what exact data-model extension is required?
- What should be done about copula upper-support overshoot and missing spatial correlation?
- Does the active-power ESS formulation remove the exact equality-rank deficiency?
- Does it materially improve bootstrap NLP robustness?
- Does `pch * pdch` complementarity remain numerically problematic?
- Can `sch/sdch` be removed safely from the network SMOPF?
- Is the P5.2 narrow-band workaround still necessary after B3?
- Which nonlinear family is the highest remaining numerical risk after the successful reformulations?

Finish exactly:

`P5.3-B COMPLETE — reformulation experiments ready for planner decision`

Then stop.

---

# Historical evidence retained for reference

The following summarizes earlier local-NLP work. It remains valid evidence but does not constrain P5.3 where the current-stage instructions above explicitly supersede it.

## Original frozen reference failure

Historical frozen model:

`data/SRP1/Results/FrozenSMOPF/frozen_DSO_node7_case33_2_2025_Winter_cycle10.pkl`

Metadata:

- DSO node 7;
- `case33_2`;
- 2025 Winter;
- ADMM cycle 10;
- warm start;
- `rho_v = 1.5`;
- `rho_pf = 2.25`;
- `rho_ess = 3.375`.

Repeated baseline:

- primary exact-Hessian MA97: `internalSolverError / Error in step computation`;
- limited-memory recovery: `maxIterations`.

Tightening only explicit `vmag` or `vmag_sqr` bounds did not help.

Removing non-reference explicit `vmag` variables/equalities from the active DSO NLP converted the decisive frozen failure into a clean primary exact-Hessian success. This led to the accepted production `vmag_nodes` refactor.

## P2/P3 progression

The production `vmag_nodes` refactor passed multiple preserved local failures and live seed-2026 operational smoke tests.

Residual failures then localized around network-side shared-ESS operation. Audits identified the shared-ESS squared-magnitude equality as a major candidate because all derivatives vanish near zero dispatch.

Removing the row entirely fixed failures but materially violated the relation and was rejected.

Equivalent in-place scaling experiments proved that row normalization itself could clear decisive DSO and TSO exact-Hessian failures while preserving the physical equality.

## P4 accepted scaling

Shared ESS:

`kappa * ((sch-sdch)^2 - pnet^2 - qnet^2) = 0`, `kappa = 1/S_rated`.

Ordinary ESS uses the analogous immutable build-time scale.

The shared scale is mutable on reused models and preserves KKT multiplier consistency when capacity changes.

## P5 planning integration

Zero-investment operational evaluation converged cleanly, but the tiny positive-bootstrap candidate exposed new cold-start failures before ADMM.

Scalar caps on shared-ESS `kappa` produced strongly non-monotone convergence and relocated failures.

## P5.2 narrow-band evidence

Replacing the hard shared-ESS zero-gradient equality by a tiny two-sided ranged row gave the first full positive-bootstrap initialization with zero persistent failures. At `epsilon_rel = 1e-4`, all 48 network solves succeeded on the primary exact-Hessian path with zero recovery.

However, the nominal physical band was below the solver's effective feasibility resolution at tiny capacity, especially in TSO models. This is why P5.3 now prioritizes a broader structural formulation audit rather than productionizing the narrow-band workaround.
