# COWORK HANDOFF — Shared Resources Planning NLP Stability Investigation

> **SUPERSEDED (2026-09-13) by `PLANNER_BRIEF_2026-09-13.md` and its Addenda 1 and 2.**
> This document's framing — including its "programme closed" conclusion — no longer holds.
> The numerical programme was reopened when P5.15 Step 0 identified the ESSO `maxIterations`
> failures as a **warm-start policy defect**, not an intrinsic conditioning limit. Read
> `REVISION_CONTEXT.md` ("CURRENT SOURCE OF TRUTH") first, then the brief. Retained unedited
> below as a historical record of the state at handoff; do not act on its instructions.


## Purpose

This repository contains the Shared Resources Planning Tool and an ongoing
numerical-stability investigation of the nonlinear SMOPF operational oracle
used inside an ADMM-coordinated planning procedure.

The current investigation has progressed through the P5.12 stages.

Do not restart the investigation from first principles.

The repository contains the evidence, reports, harnesses and preserved solver
artifacts needed to reconstruct the current state.

---

## Roles

The intended reasoning structure is:

Planner
- owns technical/project decisions;
- reviews evidence independently;
- maintains REVISION_CONTEXT.md and LOCAL_NLP_STABILITY_PLAN.md;
- authorizes bounded implementation/experiments.

Advisor
- independent deep technical reviewer;
- mathematical/numerical/optimization questions only;
- read-only;
- should challenge assumptions;
- is not authoritative.

Worker
- implementation and experiment executor;
- bounded tasks only;
- runs code/tests/experiments;
- reports evidence;
- does not choose research direction.

Normal flow:

Planner -> Worker -> Planner

For mathematically subtle decisions:

Planner -> Advisor -> Planner -> Worker -> Planner

The user normally interacts only with the coordinating Planner.

---

## Canonical local numerical environment

Repository:
share-resource-planning-no_esso-degradation

Canonical Python:

/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python

Canonical IPOPT:

/usr/local/bin/ipopt

Production numerical work must preserve the existing provenance rules.

Do not silently use another Python, IPOPT, linear solver, scenario, or
environment.

The repository's provenance gates and current local instructions are
authoritative.

---

## Current accepted investigation state

### P5.12-R

Exact cycle-21 solver input successfully captured before solving.

Important result:

- target cycle-21 input can be reproduced without replaying the complete ADMM
  history;
- exact frozen fixture exists;
- target solve was proven not to have executed during capture.

### Arm A exact replay

The frozen cycle-21 input reproduced the historical failure exactly.

- 3000 IPOPT iterations;
- maxIterations;
- full trace reproduced;
- consumed NL byte-identical to frozen prepared NL.

Therefore the failure is deterministic from the frozen fixture.

### P5.12-K

Warm-start transfer bug hypothesis rejected.

Initial cycle-21 stationarity jump is caused by the ADMM consensus parameter
transition, not corrupted transferred multipliers.

No mechanism for the subsequent failure was established.

### P5.12-P

PATH SENSITIVITY SUPPORTED.

Using the byte-identical cycle-21 NLP:

warm_start_bound_push = 1e-6 -> Optimal, 89 iterations
warm_start_bound_push = 1e-5 -> maxIterations, 3000 iterations
warm_start_bound_push = 1e-4 -> Optimal, 151 iterations

Only warm_start_bound_push changed.

This establishes strong solver-path sensitivity for this frozen fixture.

It does NOT justify changing production settings.

### P5.12-T

ESCAPE SIGNATURE OBSERVED — MECHANISM UNRESOLVED.

All three paths enter a difficult regime.

The failed baseline differs because it never recovers.

Safeguard activity is a symptom, not the trigger.

No useful pre-failure predictor was identified.

### P5.12-G

NO ARM-A-UNIQUE LOCAL-GEOMETRY SIGNATURE.

Established:

- delta_c = 0 throughout all three trajectories;
- strong-form Jacobian-rank failure requiring IPOPT constraint
  regularization is unsupported;
- delta_x does not explain failure;
- no Arm-A-specific weak constraint-row family was found;
- no distinctive near-bound population was found;
- line-search rejection is not responsible for the tiny Arm-A accepted step;
- Tier-2 spectral/SVD analysis is currently not justified.

The exact fraction-to-boundary blocking variable remains unidentified because
d_x was not preserved.

Mechanism closure on the frozen cycle-21 fixture is currently deprioritized.

---

## Breadth / oracle-reliability investigation

The planning-relevant question is now:

Is the cycle-21 outcome sensitivity isolated, or does modelling-invariant
numerical initialization sensitivity affect other operational oracle
evaluations?

### Known positive control

TARGET_cycle21

Historical warm_start_bound_push = 1e-5.

1e-6 -> Optimal
1e-5 -> maxIterations
1e-4 -> Optimal

This fixture is a KNOWN POSITIVE CONTROL.

It must remain excluded from breadth numerator/denominator counts.

### TARGET_cycle20

Previously unknown fixture.

1e-6 / 1e-5 / 1e-4 all converge.

Classified robust.

The cycle-21 fragility is therefore not reproduced one cycle earlier in the
same block.

### DSO comparator

DSO case33_2, node 7
2025 Autumn
cycle 7

Baseline reconstruction was validated exceptionally strongly:

the reconstructed historical baseline IPOPT trace reproduced the preserved
trace byte-identically after normalization only of non-numerical fields.

This proves the preserved-comparator reconstruction method can be
scientifically valid.

All LOW / BASELINE / HIGH solves converged.

However, do NOT carry forward the simple label ROBUST without qualification.

Frozen objective equivalence threshold:
relative difference <= 1e-6.

Observed:

LOW vs baseline:
1.36e-5 relative objective difference.

HIGH vs baseline:
3.65e-6 relative objective difference.

Therefore objective equivalence lies in the predeclared INDETERMINATE band.

At the same time:

- interface vmag / pf_p / pf_q differences are approximately 1e-8;
- no outcome flip occurs;
- no branch flip is established;
- substantial primal movement is dominated by compensating internal qg
  redistribution;
- propagating operational outputs are effectively equivalent.

Carry this fixture as:

VALIDITY ROBUST
BRANCH EQUIVALENT
OBJECTIVE EQUIVALENCE INDETERMINATE

The larger objective difference corresponds descriptively to approximately
2.19 planning units under its planning weight.

Do not invent a new threshold from this result.

### TSO comparator

TSO case9
2025 Summer
cycle 7

Not yet tested in the breadth perturbation stage.

Important:

its historical warm_start_bound_push is 1e-6, NOT 1e-5.

Therefore its correct multiplicative test ladder is:

LOW      = 1e-7
BASELINE = 1e-6
HIGH     = 1e-5

This is the next proposed experiment.

---

## Current breadth verdict

BREADTH PROBE INCONCLUSIVE

Current previously-unknown evidence consists only of:

- TARGET_cycle20;
- DSO case33_2 comparator.

This is too small for a genuine breadth conclusion.

No population-frequency claim is permitted.

---

## Current next proposed action

The next proposed action is the bounded TSO comparator experiment:

TSO case9
2025 Summer
cycle 7

Three solves maximum:

BASELINE = warm_start_bound_push 1e-6
LOW      = 1e-7
HIGH     = 1e-5

The baseline solve is first a reconstruction gate.

LOW/HIGH may be interpreted only if baseline reconstruction reproduces the
strongest available historical evidence.

Each solve must run independently from a fresh reconstruction.

Only warm_start_bound_push plus unavoidable output_file redirection may
change.

Reuse all classification thresholds frozen during P5.12-W.

Do not execute this experiment until the current repository evidence and
governing documents have been audited.

---

## Important methodological constraints

Do not:

- tune IPOPT adaptively;
- run additional parameter values;
- modify production settings based on the cycle-21 result;
- resume investment search;
- infer causality from solver telemetry;
- launch Tier-2 SVD analysis without a new reason;
- create new ADMM trajectories merely to enlarge the breadth sample;
- change thresholds after seeing results;
- count TARGET_cycle21 as new breadth evidence.

Preserve strict distinction among:

1. solve-validity sensitivity;
2. objective-value reproducibility;
3. operational/interface-output reproducibility;
4. local branch/regime changes;
5. internal non-unique dispatch;
6. numerical-path-only sensitivity.

---

## Repository-state caveat

.claude/agents/planner.md and worker.md were recently modified by the user to
change model/effort configuration.

These are orchestration changes, not numerical project changes.

Going forward distinguish:

NUMERICAL FROZEN STATE

from:

ORCHESTRATION-ONLY APPROVED DRIFT.

Do not claim the entire Git working tree is unchanged when these files differ.

---

## Housekeeping facts to retain

Current IPOPT execution provides c/d constraint scaling.

This supersedes an older P5.3-era carried-forward statement that only
objective scaling was available.

Historical reports must not be silently rewritten.

Stage scripts/reports should use a consistent location convention going
forward, but do not reorganize historical evidence merely for cleanliness.

---

## First Cowork task

Before making ANY modification or running ANY solver:

1. Read fully:
   - CLAUDE.md
   - CLAUDE.local.md
   - REVISION_CONTEXT.md
   - LOCAL_NLP_STABILITY_PLAN.md
   - EXPERT_REVIEW.md
   - this COWORK_HANDOFF.md

2. Inspect:
   - current git branch and HEAD;
   - git status;
   - .claude/agents definitions;
   - P5.12-R/P/K/T/G/W/X reports and scripts;
   - preserved P512 result directories;
   - exact evidence supporting the TSO comparator.

3. Reconcile this handoff against the actual files.

4. Identify any discrepancy, stale statement or unsupported claim.

5. Do NOT modify anything.
6. Do NOT run any experiment.
7. Do NOT run IPOPT.

Return a:

COWORK HANDOFF ASSESSMENT

containing:

- repository state;
- investigation timeline;
- accepted findings;
- unresolved hypotheses;
- current breadth evidence;
- discrepancies found;
- whether the TSO comparator is scientifically ready;
- exact recommended next action;
- confidence and evidence basis.

Stop for user review.