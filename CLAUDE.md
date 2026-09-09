# CLAUDE.md

Guidance for Claude Code in this repository. Read this file first, then read
the two governing documents named below **in full** before doing any work on
the active initiative. This file is a pointer and a status board, not a
substitute for them.

## Project

Shared Resources Planning Tool — an optimization research codebase for
planning TSO-DSO shared energy storage resources. It formulates nonlinear
SMOPF (sequential multi-period optimal power flow) subproblems in Pyomo,
solved with IPOPT + the MA97 linear solver, coordinated across a transmission
system operator (TSO) and multiple distribution system operators (DSOs) via
ADMM.

## Governing documents — read both, in this order

- `REVISION_CONTEXT.md` — repository-wide source of truth. Its
  `CURRENT SOURCE OF TRUTH` section supersedes anything older recorded
  further down the same file.
- `LOCAL_NLP_STABILITY_PLAN.md` — the authoritative stage plan. The filename
  is legacy; the active scope moved from local-NLP repair to nonlinear-oracle
  stability long ago. **For any stage work, this file wins over this
  CLAUDE.md.**

Then read the reports for the two most recent stages: `P5_7_BRANCH_SELECTION_DIAGNOSIS_REPORT.md`
and `P5_8_ADMM_SCALING_VALIDATION_REPORT.md`.

## Where the work actually is

The active initiative is **not** local-NLP repair, and it is **not** P4.6.
Those are closed. It is: the nonlinear operational oracle returns a different
answer depending on how many times it is re-solved, so the investment
landscape it induces is not stable enough to optimize over.

Branch: `feature/derivative-free-planning`, from the accepted P5.5-D HEAD.

| Stage | Verdict | Report |
|---|---|---|
| P5.5-D | `P5.5-D-C` — practical rigorous lower-bound architecture is unavailable | `P5_5_CONVEX_PLANNING_ARCHITECTURE_REPORT.md` |
| P5.6-A / B | `PARTIAL` — oracle built; policy locked (T0-only start, midpoint-only anchor, `(S,h)` coordinates) | `P5_6_NONLINEAR_DERIVATIVE_FREE_PLANNING_REPORT.md` |
| P5.6-C | `P5.6-C-C` — derivative-free search is not ready | same |
| P5.6-D | `P5.6-D-C` — uniform refinement does not stabilize the investment landscape | same |
| P5.7 | `P5.7-A` — unique operational oracle can likely be recovered — **accepted** | `P5_7_BRANCH_SELECTION_DIAGNOSIS_REPORT.md` |
| **P5.8** | `P5.8-B` — objective scaling improves stability but additional ADMM issues remain — **DELIVERED, AWAITING PLANNER REVIEW** | `P5_8_ADMM_SCALING_VALIDATION_REPORT.md` |

**No new stage is authorized.** Do not start one.

The short version of the diagnosis: production forms each ADMM subproblem
objective as `base / effective_scale + consensus terms` with
`effective_scale` about `1.05e5`, so IPOPT's stationarity tolerance is five
orders of magnitude looser in base-objective units. P5.7 established that this
accounts for `96.5%` of the ADMM-to-polish gap and ruled out initialization,
IPOPT itself, and the formulation. P5.8 validated the rescaling — and found it
necessary but not sufficient: the per-step drift persists at `80-90%`, the
binding problem is now the stopping criterion
(`objective_tolerance = max(1e3, 1e-3 * recourse) = 827971`, i.e. `25x` the
`33031` planning signal), and rescaling breaks the downstream exact-consensus
polish because better local optimality comes with worse agreement.

## Standing prohibitions — in force until the planner says otherwise

- Do **not** modify production code. This includes the nonlinear AC SMOPF
  equations, the active-energy ESS formulation, H1 complementarity and
  `ESS_COMPLEMENTARITY_TOLERANCE = 1e-4`, D2-P shared-S bookkeeping, ESSO
  degradation/SoH/salvage, IPOPT tolerances and options, the MA97/exact-Hessian
  policy, ADMM tolerances, rho and adaptive-penalty settings, the interface
  anchor policy, and the master/Benders cut code.
- Do **not** run or implement derivative-free search, GPS, MADS, any
  investment search, surrogate optimization, the replacement outer planning
  loop, production MISOCP planning, distributed convex ADMM, QCP/Benders cut
  recovery, or TSO SDP/QC strengthening.
- Every stage stops for planner review before the next begins, and each stage
  report ends with its specific required closing phrase — never skip it, never
  paraphrase it. See `docs/METHODOLOGY.md`.
- Never guess an invocation or entry point. Surface the candidate and its
  evidence, then wait for confirmation.
- Never fabricate or approximate a result. Every validation runs against real
  production code and real solves.

## Environment — canonical runtime is a hard gate

The canonical interpreter on **this machine (Mac Studio, since 2026-09-09)**:

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python
```

The path `/opt/anaconda3/envs/opf_env_py311/bin/python` that still appears in
the stage reports and in many `p5*.py` harness docstrings is the old MacBook
Air's and is superseded. The reports are left as written because they record
the runtime that produced their evidence.

Canonical identity, all five asserted by `p54r_provenance.gate()`, which
aborts on any mismatch:

- SRP1 scenario checksum `5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358`;
- IPOPT resolved path `/usr/local/bin/ipopt`;
- IPOPT version `3.14.18`;
- IPOPT ASL build `20241111`;
- HSL linear solver `ma97` (HSL `5.5.0`).

Every harness that loads SRP1 must call that gate before doing any work.

**IPOPT must always be the locally installed `/usr/local/bin/ipopt`, never the
conda environment's.** The environment also contains
`conda-forge::ipopt 3.14.19` (ASL `20231111`), which shadows the canonical
binary on `PATH` whenever the environment is activated. Production is not
affected and is not to be changed: `network.py:487` and
`shared_energy_storage_data.py:859` both pass
`executable=solver_params.solver_path`, which `SolverParameters` reads from
`NLP_SOLVER_PATH` in `.env` with `require_path=True`, so `PATH` is never
consulted. The gate probes that same configured path and nothing else, asserts
it, and records the shadowing `PATH` binary separately as an explicitly unused
diagnostic.

The environment is captured in `environment.yml` (authoritative) and
`requirements.txt` (pip half only). It is half conda (46 packages) and half
pip (23); `copulas 0.14.0` is on the pip side and generates the scenario
realization the checksum is a checksum of. IPOPT and HSL are outside conda.

## Repository conventions

- `data/` result and diagram directories are multi-GB and intentionally **not**
  committed. Never `git add -A` or `git add .`; stage files by name.
- `.env` exists in the working tree and must never be committed.
- Diagnostic harnesses are `p5*.py` at the repository root, one per stage, each
  gated by `p54r_provenance.gate()`. Follow that pattern.
- Methodology — frozen-pickle regression with SHA-256 verification, calling
  real production functions rather than reimplementing them, and the stage
  report template — is in `docs/METHODOLOGY.md`.
