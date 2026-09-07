# CLAUDE.md

Guidance for Claude Code when working in this repository. Read this file
first in every session before touching anything related to the local-NLP
stability initiative described below.

## Project

Shared Resources Planning Tool — an optimization research codebase for
planning TSO-DSO shared energy storage resources. It formulates nonlinear
SMOPF (sequential multi-period optimal power flow) subproblems in Pyomo,
solved with IPOPT + the MA97 linear solver, coordinated across a
transmission system operator (TSO) and multiple distribution system
operators (DSOs) via ADMM, inside an outer Benders-type investment
planning loop (`run_planning_problem()` in `shared_resources_planning.py`).

## Active initiative: local-NLP numerical-stability investigation

We are mid-execution of a staged investigation into local SMOPF solve
failures, governed by two documents that are the source of truth for
scope, findings, and rules. **Read both in full before doing any work on
this initiative — do not rely on this file's summary alone:**

- `REVISION_CONTEXT.md` — background, prior findings, why this
  investigation exists.
- `LOCAL_NLP_STABILITY_PLAN.md` — the authoritative staged plan (P1–P4.6).
  For any P4.x work, **this file's rules win over anything else,
  including this CLAUDE.md.**

## Current status (as of 2026-09-06, commit `0171f451`)

| Stage | Status | Report |
|---|---|---|
| P1–P3 | Diagnosis complete | `P3_AUDIT_REPORT.md`, `P3_5A_REPORT.md`–`P3_5D_REPORT.md` |
| P4.1 Lifecycle audit | Complete | folded into P4.2 diff, see `docs/P4_PROGRESS.md` |
| P4.2 Production implementation | Complete | `git show 0171f451 -- definitions.py model_construction_helpers.py network.py` |
| P4.3 Construction/equivalence validation | Complete — all invariants held | `P4_3_P4_4_REPORT.md` |
| P4.4 Frozen regression | Complete — required gate passed, 9/9 cases clean | `P4_3_P4_4_REPORT.md` |
| P4.5 Seed-2026 operational smoke | Complete — converged, improved vs. P2.10 baseline | `P4_5_REPORT.md` |
| **P4.6 Standard-ESS audit/extension gate** | **NOT STARTED — this is the next step** | — |
| Final consolidated P4 report | Not started (depends on P4.6) | — |

Full narrative detail, exact diffs, and file/script inventory:
`docs/P4_PROGRESS.md`.

## Standing rules — do not violate these

These prohibitions have applied throughout P1–P4.5 and remain in force for
P4.6 and the final report unless the planner (the user, in their reviewer
capacity) explicitly says otherwise in this session:

- Do **not** modify: `sess_comp`, shared-ESS or standard-ESS SOC, ESSO
  degradation/SoH/calendar ageing, active/apparent throughput
  definitions, generator/branch/voltage formulations, solver settings,
  ADMM settings, or Benders/local-cut logic.
- Do **not** modify standard/ordinary ESS equations unless P4.6's own
  validation explicitly earns that change (see `LOCAL_NLP_STABILITY_PLAN.md`
  §8) — and even then, only after an active standard-ESS test case is
  identified and the equivalence is verified, not assumed.
- Do **not** run the full planning problem (`run_planning_problem()`) —
  only the reduced distributed-operational-only smoke configuration
  (`SharedResourcesPlanning.run_operational_planning(type='distributed', ...)`),
  until a stage explicitly authorizes more.
- Every stage stops for planner review before the next stage starts. Each
  stage report ends with a specific required closing phrase (see the plan
  document and `docs/METHODOLOGY.md`) — never skip it, never paraphrase it.
- Never guess a real invocation/entry point when unsure — surface the
  candidate and its evidence, then wait for explicit confirmation before
  treating it as authoritative.
- Never fabricate or approximate a result. Every validation stage in this
  investigation runs against real production code and real (frozen or
  live) solves — see `docs/METHODOLOGY.md` for the exact discipline to
  follow.

## Methodology

This investigation follows a strict, consistent diagnostic discipline —
frozen-pickle regression with SHA-256 verification, calling real
production functions directly rather than reimplementing them, a
"required-first-gate-then-proceed" pattern for regression stages, and a
consistent stage-report template. Follow it for P4.6 and the final
report. Full detail: `docs/METHODOLOGY.md`.

## Environment

- Production code requires Pyomo + IPOPT + MA97. Check whether this
  Claude Code environment has that stack available (e.g. `python -c
  "import pyomo, pyomo.environ"`, and confirm IPOPT/MA97 are on PATH)
  before assuming you must hand scripts to the user to run.
  - If it's available: run P4.6's validation scripts yourself.
  - If it's not: write the script, verify it compiles
    (`python -m py_compile`), and ask the user to run it and share back
    the JSON report — this was the necessary workflow in the prior
    session (a sandboxed Cowork environment with no Pyomo/IPOPT), and it
    still works fine here if needed.
- Git: this repo lives at
  `/Users/micaelsimoes/PycharmProjects/shared-resources-planning`, on
  branch `admm_residual_balancing_tests`. The P4 kappa-scaling fix and
  all P3.5/P4 diagnostic scripts, reports, and small JSON/log evidence
  files are committed as of `0171f451`.
  - `data/` result/diagram output directories (multi-GB of pickles and
    `.xlsx` files) are intentionally **not** committed — do not `git add`
    them wholesale (no `git add -A` / `git add .`). Stage specific files
    by name, as the existing commits do.
  - `.env` exists in the working tree and must never be committed.

## Next step

Execute **P4.6** (standard-ESS audit/extension gate) exactly as specified
in `LOCAL_NLP_STABILITY_PLAN.md` §8 — audit first, and only implement a
production change if an active standard-ESS test case exists and
equivalence is verified; otherwise return a design recommendation and
wait for planner approval. Then produce the final consolidated P4 report
per §10 (sections A–F), ending with exactly one of:

```
P4 PASS — recommend planner approval for reduced planning baseline
P4 PARTIAL — planner review required before further execution
P4 FAIL — do not proceed
```

Then stop.
