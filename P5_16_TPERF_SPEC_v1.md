# P5.16 T-perf — where the solve time goes (spec v1; author-authorized diagnostic, 2026-10-31)

Planner. Purpose: decide, from measurement, whether speed/memory work should target Pyomo overhead, IPOPT
conditioning (per-unit base, scaling), the W-auxiliary structure, or only parallelism — **before** the
convention-change re-evaluation campaign, so any formulation change rides the one re-evaluation already decided.
Not a production change; nothing here enters a table.

## 1. Part A — this machine (24 GB; interpreter `/opt/anaconda3/envs/opf_env_py311/bin/python`)

Inputs: the certified SRP1 blocks of the unit cell (`…/evals/ca8927e75d628bd1_n7_4h_e1/certified_models.pkl`,
sha256 `fe06ca48…`, hash-verified on load) — one TSO block and one DSO block (2025 Spring), plus their ESSO.
**Declared solves: exactly 6**, under `SolveProfileGuard` (bounded): for each of the TSO and DSO block, (i) a
warm-started re-solve at the certified point with production options (the ADMM-cycle case), (ii) a cold solve from
the production initialisation (the initialisation case), (iii) a warm re-solve with IPOPT `print_info_string yes`,
`print_level 5`, `nlp_scaling_method gradient-based` diagnostics on — the scaling case. ESSO: 0 solves (its cost is
known small; report its size only).

Measured per solve (all captured; capture paths asserted before the run): variables, rows, Jacobian and Hessian
nonzeros; wall time split — Pyomo NL write, IPOPT total, IPOPT linear-solver time and function-evaluation time
(from the IPOPT timing statistics, `print_timing_statistics yes`), result load; iteration count, restoration
phases, `mu_final`; peak RSS of the process; the gradient-scaling factors IPOPT applied (min/max per block) and the
count of variables at their bounds; the solver log in full.

## 2. Part B — the Studio (author runs one command; nothing else runs there meanwhile)

Same harness, same declared count, on one paper-scale TSO block and one DSO block built by
`p515_s44_scale_measurement.derive_case('paper', …)` at 25 scenarios (build only, then the three solves per block;
the build time and memory recorded too). Output root on the Studio `data/SRP1/Results/P516TPERF/studio/`; the
author copies it back. The harness refuses to run concurrently with itself and captures stderr.

## 3. Predictions (recorded before any run)

P1 NL write + result load ≥ 20 % of per-solve wall time on the paper-scale blocks (≥ 10 % on SRP1).
P2 Warm-started IPOPT iterations per block ≤ 30 on SRP1, ≤ 60 at paper scale; cold ≥ 3× warm.
P3 Gradient-scaling factors span ≥ 4 decades on the DSO block (per-unit base effect); ≤ 3 on the TSO.
P4 Linear-solver time ≥ 60 % of IPOPT time at paper scale (MA97 on the 25-scenario block).
P5 Peak RSS of one paper-scale DSO block solve ≥ 3 GiB above the build footprint.

## 4. Output

`data/SRP1/Results/P516TPERF/local/` (new), `manifest_sha256.json`, logs; report `P5_16_TPERF_REPORT.md` with a
**ranked lever table**: for each of (parallelism, NL-write avoidance, per-unit base / scaling, W-auxiliary
structure) the measured share of time it could address, the gate a change would need, and the risk. The Advisor
reads the report before any lever is adopted; a formulation lever, if adopted, enters the convention-change spec as
C6 with the Step 1b candidate-3′ gate shape (same dispatch to tolerance; iterations and time before/after).

## 5. Not permitted
Production code changes; more than the declared solves; touching the D0 inputs other than reading them; the dry-run
root; any ADMM cycle.
