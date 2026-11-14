# Worker Report — Gate G5 (reformulated ESSO, A1/A3/A4 agreement)

**Verdict: FAIL.**

A1 and A4 agree essentially bit-for-bit (relative difference `5.37e-12`, node 7 and node 9
identical). **A3 (`mu_strategy=adaptive`) disagrees with both A1 and A4 by `2.86e-4`
relative — 286x the gate's `1e-6` threshold — at both nodes.** The predeclared
discriminator (the barrier-complementarity leak, propagated through
`EPS_ESSO_THROUGHPUT * delta_throughput`) explains only **33.3%** of that disagreement;
a residual of **66.7%** of the observed difference remains unexplained by the leak
mechanism specified for this gate. Per the task's own predeclared rule ("if a
disagreement remains after accounting for Δleak → that is a genuine G5 failure"), this
is a genuine G5 failure, not a leak-explained one. All six solves converged `optimal`
with zero `maxIterations`/recovery events and rule-ten ratios of ~4.5% of threshold
(well-settled, not stopped), so the disagreement is not an artifact of under-convergence.

Authority: `PLANNER_BRIEF_2026-09-13.md`, Addendum 1 ("Gate G5 (new)") + Addendum 3.

## Task received

Run gate G5 only: on the reformulated ESSO (current working tree, including remedy (h)
`tol=1e-8`/`acceptable_tol=1e-7` and the H3 cohort-split rule), re-solve the node-7 and
node-9 instances under arms A1, A3, A4 (definitions per the Step 0 table). All three must
agree to 1e-6 relative. Report, per arm/node: `mu_final`, `s_obj`, the ratio detector
`max min(pch,pdch)/s_max`, the spurious throughput and its closed-form estimate,
objective, termination, iterations, max constraint violation; test the discriminator
(predicted vs. observed objective difference from the leak); report
`maxIterations`/recovery counts and rule-ten ratios; state the objective convention.
No production changes, no G1-G4, no ADMM campaign, no commit.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (Step 0 table, Addendum 1 "Gate G5 (new)", Addendum 3)
- `P5_15_EXPERT_HANDOFF.md`, `P5_15_GATE_HOLD_REPORT.md` (barrier-identity derivation,
  remedies ranking, why G1 was held)
- `WORKER_REPORT_H.md`, `WORKER_REPORT_H3.md` (remedy (h) and H3, both uncommitted in the
  working tree at task start — verified present via `import shared_energy_storage_data`)
- `shared_energy_storage_data.py` — full read of `_build_subproblem` (Variables,
  degradation chain, `energy_storage_operation_agg`, H3 rows, objective assembly at
  `feasibility_penalty`/`model.objective` — **`model.objective` is `feasibility_penalty`
  only; `salvage_value`/`salvage_credit` are separate expressions NOT included in the
  objective**), `_create_solver`, `_optimize`, `_run_solver_attempt`,
  `_is_recoverable_shared_ess_failure`, `_get_esso_complementarity_diagnostics`,
  `_complementarity_ratio_for_model`, `_parse_ipopt_barrier_terms`,
  `_update_model_with_candidate_solution`
- `shared_resources_planning.py:3169-3210` (`create_shared_energy_storage_model`, the
  production entry point whose pre-solve steps this harness mirrors)
- `p515_0_esso_warmstart_ab.py` (arm-table precedent, `max_constraint_violation`,
  `parse_ipopt_log_tail` — reused by import)
- `p515_1_esso_reform_smoke.py`, `p515_1b_eps_sensitivity_check.py`,
  `p515_h_tol_remedy_check.py` (the established reformulated-ESSO fixture and its
  construction path — reused by import)
- `p513_solve_profile_guard.py` (`SolveProfileGuard`)
- `helper_functions.py` (`fix_or_set`, `solver_result_succeeded`)
- `data/SRP1/SharedESS/SRP1_ESS_Params.json` (confirmed no case-file
  `warm_start_bound_frac`/`warm_start_slack_bound_frac` overrides — A1's "`_frac` at IPOPT
  defaults" needs no explicit deletion, unlike Step 0's pre-1a workaround)
- `git log`/`git status` (confirmed HEAD `8af3e242`, `shared_energy_storage_data.py`
  uncommitted-modified on top of it — this is remedy (h) + H3)

## Files modified

None under `data/`. No production code touched.

## Files created

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_g5_reformulated_gate.py`
  (new harness, `p5*` convention)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151/g5_gate_summary.json`
  (sha256 first 16: `4e522921c2cd129d`)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151/g5_gate_full.json`
  (sha256 first 16: `994a34da74def6c0`)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151/g5_gate_logs/node{7,9}_{A1,A3,A4}.txt`
  (6 IPOPT logs; no `_recovery` variants — confirms zero recovery solves)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/WORKER_REPORT_G5.md`
  (this file)

None of these paths pre-existed (confirmed via `ls data/SRP1/Results/P5151/` before writing);
none of the paths under `data/` was edited, only new files written.

## Instance

Reused, by import, the SAME reformulated node-7/node-9 fixture every P5.15 stage-1
diagnostic has used: `p56a_oracle.fresh_planning`, `create_admm_variables`, S=1.00 MVA /
E=2.00 MVAh candidate at nodes 7 and 9 (first representative year; node 5 zero
investment), `+/-10%` duty-cycle `p_req` (`_set_nonzero_charge_discharge_request`,
amplitude `0.10 * S`), `EPS_ESSO_THROUGHPUT = 1e-3` (case default, unchanged). This is the
right instance because it is the only reformulated-ESSO fixture this stage has validated
(Step 1 smoke test, the eps check, the tol-remedy check), so G5 tests the same
model/candidate the rest of P5.15 has been measuring rather than a new one.

- `baseline_checksum` (canonical `SRP1.json` scenario): `5a02b77ccbbbbbb8...` (full value in
  `g5_gate_summary.json`)
- `instance_hash_sha256`: `49fb0d26cd115c2dbe4b44ca85e4f2d316580f6a8126a151d6b20f4efeda330f`
- `shared_energy_storage_data.py` sha256 of the exact working-tree file this run executed
  against: `c793b713e675f773974b655263e8b4d4cbc6fcc51d1157870a3ff1cbf7811b7d` (HEAD
  `8af3e242286172efda0d02a67c55f586af94cd8c` plus uncommitted remedy (h) + H3, per
  `WORKER_REPORT_H.md` / `WORKER_REPORT_H3.md`)
- `ESSO_TOL_OVERRIDES` in force for every arm: `{'tol': 1e-8, 'acceptable_tol': 1e-7}`
  (remedy (h))

**Design note on the three arms per node.** Each arm solves an independent
`copy.deepcopy` of the SAME freshly-built, never-yet-solved per-node model (built once via
the production `build_subproblem` / `update_model_with_candidate_solution` /
`fix_or_set` sequence — the exact pre-solve steps of
`create_shared_energy_storage_model`, reused by direct call rather than reimplemented).
Only the solver policy differs between arms, per the brief's own table ("only solver
options differ"). Chaining A1/A3 from A4's own converged output was considered and
rejected: warm-starting from an already-optimal point converges trivially under any
policy and would not test whether the reformulated ESSO has multiple local optima
reachable from the same starting state under different solver policies — the actual
question G5 asks.

**Objective convention, stated on every table below**: `model.objective ==
model.feasibility_penalty` — the ESSO subproblem's own objective only (throughput
regularization `EPS_ESSO_THROUGHPUT * sum(pch+pdch)` plus, when `params.slacks=True`
(confirmed true in this fixture), `PENALTY_ESSO_SLACK * sum(slack_es_pnet_up +
slack_es_pnet_down)`). `salvage_value`/`salvage_credit` are separate `Expression`s **not**
included in `model.objective`. This harness never builds the TSO/DSO SMOPFs, so
`gross_operational_cost`/`net_operational_recourse` have no capture path here (same
convention as `p515_h_tol_remedy_check.py`, `p515_1b_eps_sensitivity_check.py`).

## Commands / experiments run

```
/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B p515_g5_reformulated_gate.py
```

Guard: `SolveProfileGuard` armed for the whole run, permitting solves only from
`shared_energy_storage_data.py:_run_solver_attempt`. Declared count **6** (2 nodes x 3
arms). **Observed: `permitted_solve=6`, `permitted_exec=6`, `blocked_solve=0`,
`blocked_exec=0`, `verify_failures=[]`.** Exactly the declared count, no more, no fewer.

A first run failed with `KeyError` inside the harness's own post-solve read (calling
`model.solutions.load_from(result)` a second time after `SED._optimize` had already
loaded it internally — Pyomo's `SolverResults` symbol map is single-use). Fixed in the
harness (read the already-loaded model directly instead of loading a second time); the
run reported below is the corrected one. The buggy run's empty/`objective:null` output
files were deleted before re-running (no committed artifact was touched; both files were
this harness's own first, unreported attempt).

## Results — per arm, per node (node 7 and node 9 are numerically identical throughout,
by construction: identical candidate, identical duty cycle)

| quantity | A1 (warm, pushes=1e-3) | A3 (A1 + adaptive) | A4 (cold) |
|---|---|---|---|
| termination | optimal | optimal | optimal |
| status | ok | ok | ok |
| iterations (primary) | 19 | 16 | 17 |
| recovery fired | no | no | no |
| `objective` (`feasibility_penalty`) | 0.023047848581294 | **0.023054447087831** | 0.023047848581170 |
| max constraint violation | 4.5806e-10 | 4.5084e-10 | 4.5757e-10 |
| worst constraint | `energy_storage_capacity_degradation[8]` | same | same |
| rule-ten ratio (constraint violation / `tol`=1e-8) | 0.04581 | 0.04508 | 0.04576 |
| `mu_final` (parsed, scaled `Complementarity`) | 9.096027e-10 | 1.870411e-09 | 9.096063e-10 |
| `s_obj` (parsed, scaled/unscaled `Objective`) | 0.1 (exact) | 0.1 (exact) | 0.1 (exact) |
| `eps` (`EPS_ESSO_THROUGHPUT`) | 1e-3 | 1e-3 | 1e-3 |
| `N_periods` (active cohort-periods) | 288 | 288 | 288 |
| ratio detector `max min(pch,pdch)/s_max` | 4.53511e-06 | 8.35485e-06 | 4.53511e-06 |
| closed-form bound `2N·mu/(2·s_obj·eps)` | 0.0026197 | 0.0053868 | 0.0026197 |
| spurious throughput measured `2Σmin(pch,pdch)` | 0.0026122 | **0.0048072** | 0.0026122 |
| measured/bound | 0.9972 | 0.8925 | 0.9972 |

`maxIterations` / recovery events, all 6 solves: **0** (expected: zero — matches).

## Discriminator test (predeclared)

`predicted_diff = EPS_ESSO_THROUGHPUT * (spurious_throughput_measured_a -
spurious_throughput_measured_b)`, compared against the observed
`objective_a - objective_b`. Identical at node 7 and node 9.

| pair | observed diff | predicted diff (leak) | residual | residual / observed | agree to 1e-6 rel? |
|---|---|---|---|---|---|
| A1 vs A3 | `-6.5985e-06` | `-2.1950e-06` | `-4.4035e-06` | **66.73%** | **NO** (rel. `2.862e-4`) |
| A1 vs A4 | `1.238e-13` | `1.237e-13` | `1.387e-16` | 0.11% | **YES** (rel. `5.37e-12`) |
| A3 vs A4 | `6.5985e-06` | `2.1950e-06` | `4.4035e-06` | **66.73%** | **NO** (rel. `2.862e-4`) |

**A1 vs A4**: the leak explains the (essentially zero) difference to within numerical
noise — this pair PASSES the gate outright, and the discriminator confirms the small
residual it does have is exactly what the barrier identity predicts.

**A1 vs A3 and A3 vs A4**: the leak explains only **33.3%** of the observed difference
(`predicted/observed = 0.3327`); a residual of **66.7%** (`4.40e-06`, roughly double the
predicted portion) is **unexplained by the leak mechanism this gate's discriminator
specifies**. Per the task's own predeclared rule, this is reported as a **genuine G5
failure**, not adjudicated away.

### A candidate mechanism for the residual — reasoned from code, NOT measured (no new solve run)

`model.objective` (`feasibility_penalty`) contains **two** arm-sensitive terms when
`params.slacks=True` (confirmed true here), not one:
`EPS_ESSO_THROUGHPUT * throughput` (the term the discriminator models) **and**
`PENALTY_ESSO_SLACK * (slack_es_pnet_up + slack_es_pnet_down)`
(`shared_energy_storage_data.py`, `energy_storage_operation_agg`: `es_pnet == agg_pnet +
slack_es_pnet_up - slack_es_pnet_down`). This is the same structural shape as the
pch/pdch pair the barrier identity describes — a linear equality with a non-negative
signed-slack decomposition — and `PENALTY_ESSO_SLACK / EPS_ESSO_THROUGHPUT = 1e3 / 1e-3 =
1e6`. If an analogous interior-point leak exists in `slack_es_pnet_up`/`_down` (both
legs simultaneously nonzero at the barrier residual scale, exactly as pch/pdch are), a
leak orders of magnitude smaller than the pch/pdch leak would still dominate the
objective difference by the same 1e6 factor, which would explain a residual of the
observed magnitude without requiring any deeper nonconvexity. **This is not measured**:
this run did not capture `slack_es_pnet_up`/`_down` values per arm (an oversight not
caught by the pre-run capture-path assertion, since the assertion checked the quantities
the task named, not this one), and no additional solve was run to check it (out of the
declared/authorized scope of 6 solves). It is reported as the most probable next
diagnostic step, not as an established explanation.

## Validation

- Guard-verified exact solve count: 6/6, zero blocked, zero extra (rules out both
  under-counting and an unrecorded recovery retry).
- Rule-ten ratios all ~0.045-0.046 (4.5-4.6% of the `tol=1e-8` threshold): every solve is
  genuinely well-settled, not stopped near its bound. The A1-vs-A3 disagreement is
  therefore not a stopping-slack artifact (rule ten/rule refinement, `CLAUDE.md`).
  A1-vs-A4's near-zero (`1.24e-13`) difference is consistent with their near-identical
  `mu_final` (`9.096027e-10` vs `9.096063e-10`, agree to 6 significant figures).
- Node 7 and node 9 reproduce identically at every arm (objective, iterations, `mu_final`,
  ratio detector all bit-identical between the two nodes) — an internal consistency check
  the harness did not need to force; it falls out of the two nodes carrying identical
  candidates and duty cycles by construction.
- `measured_to_bound_ratio` (`spurious_throughput_measured / spurious_throughput_bound`)
  is `<= 1` for every arm (0.997, 0.893, 0.997) — the closed-form bound held as an upper
  bound at every solve in this run (contrast with `WORKER_REPORT_H.md`'s finding that the
  bound was exceeded by 0.32% in a `tol=1e-6` control arm; all arms here run at `tol=1e-8`,
  where that report also found the bound holds).
- Capture-path preflight (rule eight) ran and passed BEFORE any solve: confirmed
  `SED.ESSO_TOL_OVERRIDES == {'tol': 1e-8, 'acceptable_tol': 1e-7}`, H3 function present,
  diagnostics function present, and that the production `mu_final`/`s_obj` parser
  succeeds on a pre-existing committed log from this same stage
  (`tol_check_logs/remedy_h/optim_log_node_7.txt`) before trusting it on new logs.
- **What is NOT validated**: the candidate `slack_es_pnet` mechanism above (code-reasoned
  only); whether A3's KKT point is a genuinely different local optimum of the underlying
  NLP or an artifact of the slack-leak interacting with `PENALTY_ESSO_SLACK`; whether this
  disagreement would still occur at material (C\*-scale) capacity rather than this 1.00
  MVA fixture.

## Unexpected findings

1. **A3 (`mu_strategy=adaptive`) does not agree with A1/A4 on the reformulated ESSO**,
   reproducing the qualitative pattern Addendum 1 recorded on the OLD (pre-reformulation)
   ESSO ("A3 found a different local optimum... evidence of multimodality"). The
   reformulation has NOT eliminated this: A1 and A4 (both monotone `mu_strategy`) agree to
   floating-point precision, but `mu_strategy=adaptive` alone is sufficient to land the
   solver at a measurably different (here, worse-objective) point, on the SAME instance,
   with only solver options differing.
2. The predeclared throughput-leak discriminator explains only a third of that
   disagreement; the residual is roughly double the predicted portion at both pairs
   involving A3 (a suspicious, possibly diagnostic 2x ratio, not investigated further).
3. `model.objective` contains a second arm-sensitive term
   (`PENALTY_ESSO_SLACK * slack_es_pnet_{up,down}`) that the discriminator, as specified
   by the dispatching task, does not model — flagged above as the most likely explanation
   for the residual, untested in this run.
4. My own harness bug (double `load_from` call) produced a first, silently-wrong run
   (`objective: null` for all arms) that I caught via the `KeyError` traceback rather than
   letting a null propagate into a result — reported per the project's failure-disclosure
   rule; the buggy run's own output files were deleted (not committed, not previously
   reported by anyone) before the corrected run.

## Remaining issues

- The genuine G5 failure is unresolved: whether the reformulated ESSO retains a real
  nonconvexity, or whether the disagreement is fully explained by an un-modeled
  `slack_es_pnet` leak (item 3 above), is not established by this task and needs either
  (a) a capture-path fix (record `slack_es_pnet_up/down` per arm) and a re-run under the
  SAME 6-solve budget, or (b) a small additional authorized experiment.
- Not tested at C\*-scale capacity (1.62 MVA or the 0.96875 MVA ladder point) — only at
  the 1.00 MVA fixture this stage has used throughout.
- H3 is confirmed inert on this fixture (single cohort per node, as in every prior P5.15-1
  check) and was not a factor in the observed disagreement.

## Questions for Planner

1. Is the `slack_es_pnet_up/down` leak hypothesis (item 3, "Unexpected findings") worth a
   small, explicitly-authorized follow-up (capture the two slack values per arm on the
   SAME instance, no new solve needed if folded into a re-run of this harness) before
   deciding whether G5's failure is a genuine nonconvexity or a second instance of the
   already-diagnosed leak mechanism at a different, more heavily-weighted, term?
2. Given A1 vs A4 PASS outright and only A3 disagrees, should the "adopted policy" from
   Addendum 1 (`mu_strategy=adaptive` not used as primary or recovery) be read as already
   containing the fix — i.e., is G5 to be read strictly (all three arms) or does the
   already-adopted production policy (which never uses A3) make the A1-vs-A4 agreement
   the operative result for closing Step 1?
