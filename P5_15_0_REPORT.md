# P5.15-0 — ESSO / DSO warm-start policy A/B (Step 0)

Worker report. Authorized by `PLANNER_BRIEF_2026-09-13.md`, Step 0. Diagnosis background:
`EXPERT_REVIEW_2_ACTION_PLAN.md` section 1. No production code was modified. Nothing was
committed (per instructions, commits are the Planner's responsibility).

Repository provenance at run time: HEAD `0d33cfcfd8715c334a9f18d101f91cefea294339`, branch
`feature/derivative-free-planning`. Interpreter
`/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`. IPOPT
`/usr/local/bin/ipopt` (version 3.14.18/3.14.20, both appear in the historical logs; the
solves in this stage report 3.14.18, consistent with `NLP_SOLVER_PATH` in `.env`). This is a
diagnostic stage; the frozen-artifact ceremony was not invoked (per the brief, "the
frozen-specification ceremony is not required for these diagnostics").

Harness: `p515_0_esso_warmstart_ab.py` (new file, the only file this task created). Evidence:
`data/SRP1/Results/P5150/p5150_report.json` (machine-readable, all numbers below are taken from
it) plus full IPOPT logs under `data/SRP1/Results/P5150/esso/logs/` and
`data/SRP1/Results/P5150/dso/logs/`.

---

## 0. Preflight (both fixtures)

**ESSO** — `data/SRP1/Results/P514N/esso_models_k10000.pkl`, a dict of 3 `ConcreteModel`
(nodes 5, 7, 9). For every node: `admm_objective` exists and `.active is True`;
`p_req`/`q_req`/`dual_p_req`/`dual_q_req`/`rho` all present; 288 (year,day,period) request
entries, all 288 nonzero on every one of `p_req`, `q_req`, `dual_p_req`, `dual_q_req`; `rho =
1.0`. All checks passed (`preflight.all_pass = True`). SHA-256 of the ordered
`(keys, p, q, dual_p, dual_q, rho)` tuple, JSON-serialized, per node:

| node | request-parameter hash (sha256) |
|---|---|
| 5 | `b4b7294f7a71e264d349655a9121d01bbc75f307ee5cc9e910bbd6cda66af5d8` |
| 7 | `5ee32516902f99998fddc87121f2ff4c44f83b44309c17b8ffcdc718baf3f9cd` |
| 9 | `62117b844bc66895c5658ebef6e12d1e10aa8f72a7c061723386f5ac43b8b3d6` |

**DSO** — `data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl` (the P5.12-R/P frozen
cycle-21 fixture). Metadata check: `boundary=pre_setup`, `cycle=21`,
`target=DSO:case33_3|2025|Spring`, `from_warm_start=True`; exactly one active objective
(`p58_rescaled_admm_objective`, the P5.8-rescaled form live at capture time). All checks
passed.

Production `params.solver_params` used for the ESSO solves (built via
`p56a_oracle.fresh_planning`, read-only, no solve performed before Part 1): `solver=ipopt`,
`solver_path=/usr/local/bin/ipopt`, `options={'tol': 1e-06, 'acceptable_tol': 1e-05,
'acceptable_iter': 5, 'linear_solver': 'ma57'}`, `recovery_options={'hessian_approximation':
'limited-memory', 'acceptable_tol': 0.0001, 'acceptable_iter': 1}` — read directly from
`data/SRP1/SharedESS/SRP1_ESS_Params.json`, unmodified.

---

## 1. Recorded production defect: `option_overrides` is clobbered for the ESSO's warm-start keys

Before running the arms, the harness performed one read-only probe (`clobber_probe` in the
report, no arm solved): it called production `shared_energy_storage_data._create_solver` with
`from_warm_start=True` and `option_overrides={'warm_start_mult_bound_push': 1e-3,
'warm_start_bound_push': 1e-3, 'warm_start_slack_bound_push': 1e-3}`. The solver object
returned had all three keys back at `1e-9`:

```
requested            : {'warm_start_mult_bound_push': 0.001, 'warm_start_bound_push': 0.001, 'warm_start_slack_bound_push': 0.001}
observed on solver    : {'warm_start_mult_bound_push': 1e-09, 'warm_start_bound_push': 1e-09,
                         'warm_start_slack_bound_push': 1e-09, 'warm_start_init_point': 'yes',
                         'warm_start_bound_frac': 1e-09, 'warm_start_slack_bound_frac': 1e-09}
```

Cause: `_create_solver` (`shared_energy_storage_data.py:898-911`) merges `option_overrides`
into the options dict and applies it to `solver.options` (lines 895-896), then — unconditionally,
whenever `from_warm_start=True` — re-assigns the same five keys to literal `1e-9` (lines
906-911). This runs *after* the merge, so any override on these five specific keys is silently
discarded whenever a warm-started ESSO solve is requested through `_run_solver_attempt(...,
option_overrides=...)`. `network._create_smopf_solver` does **not** have this defect (its
warm-start block at `network.py:529-534` reads the same *local* `options` dict that already
contains `option_overrides`, via `.get(key, ...)`, so an override there does take effect) — but
neither function offers a way to specify "leave at IPOPT's compiled default"; both always assign
concrete values to all five keys when warm-starting.

**Consequence for this harness.** Arms A1/A2/A3 (both families) could not be realized by
passing `option_overrides` alone. The harness instead calls the unmodified production
`_create_solver` / `_create_smopf_solver`, then edits the returned `solver.options` object
directly before calling `solver.solve(...)` — the same post-creation-override technique already
authorized and used by `p512_p_path_sensitivity.py` for one key on the DSO fixture. Every arm's
option dict is recorded before and after this adjustment in the JSON report
(`pre_adjust_options` / `post_adjust_options`), so the exact solver configuration used for every
solve is auditable.

This is a **production code defect worth Planner attention independent of the warm-start policy
question**: `option_overrides` is a documented parameter of `_run_solver_attempt` / `_optimize`,
and it silently fails for five specific keys under the single most common calling condition
(`from_warm_start=True`). See "Unexpected findings" below.

---

## 2. ESSO arms — evidence table

Instance: node 7, `data/SRP1/Results/P514N/esso_models_k10000.pkl` (P5.14-N perturbation arm,
`k=10000`). Independent deep copy of the pristine pickled model per arm; only solver options
(and, for A2, the exported suffixes) differ. Full IPOPT iteration tables:
`data/SRP1/Results/P5150/esso/logs/node7_<arm>.txt`.

| arm | termination | iterations | wall (s) | `admm_objective` | max constraint violation | rel. obj. diff vs A4 |
|---|---|---|---|---|---|---|
| A0 (current production) | `maxIterations` | 3000 | 3.70 | 1.366032314177866 | 1.715e-06 | — |
| A1 (pushes=1e-3, frac at IPOPT default) | `optimal` | 30 | 0.14 | -0.01198375471524498 | 2.916e-11 | 8.36e-11 |
| A2 (A1, primal-only warm start) | `optimal` | 37 | 0.15 | -0.011983754716242898 | 4.496e-13 | 2.89e-13 |
| A3 (A1 + `mu_strategy=adaptive`) | `optimal` | 25 | 0.14 | -0.01279096052738873 | 2.243e-12 | **6.74e-02** |
| A4 (cold start) | `optimal` | 59 | 0.19 | -0.011983754716246357 | 5.695e-13 | 0 (reference) |
| A5 (A0 + `max_iter=500`) | `maxIterations` | 500 | 0.69 | 1.366326462684692 | 6.454e-06 | — |

**Reproduction gate: PASSED.** A0 on node 7 hit `maxIterations` at exactly 3000 iterations,
reproducing the preserved `optim_log_node_7.txt` tail bit-for-bit in the fields that must match
(unscaled objective `1.3660323141778592e+00`, dual infeasibility `1.7746350293956035e+04`,
constraint violation `1.7154144709737720e-06`, same exit line).

**Success criterion (converged < 200 iters, objective within 1e-6 relative of A4, max violation
< 1e-6): A1 PASSES, A2 PASSES, A3 FAILS** (converges fast and cleanly, but to a *different*
local optimum: objective `-0.01279` vs A4's `-0.01198`, a 6.7% relative gap — this ESSO NLP is
nonconvex (the SoH chain is a 1095th-power term at this un-reformulated stage; see
`EXPERT_REVIEW_2_ACTION_PLAN.md` §1), and `mu_strategy=adaptive` changed the barrier path enough
to land in a different basin). **A5 fails by design** (cost-of-failure check only): capping
`max_iter` at 500 stops the run 2500 iterations sooner but at essentially the same non-converged
point (`admm_objective` differs from A0's 3000-iteration value by only ~2e-4 relative,
constraint violation actually *worse*, 6.45e-06 vs 1.72e-06) — i.e. `max_iter=500` alone is a
cheap early-failure detector, not a fix.

### Controls (nodes 5 and 9)

| node | arm | termination | iterations | `admm_objective` | max violation |
|---|---|---|---|---|---|
| 5 | A0 | `optimal` | 0 | -0.008019196655013524 | 6.96e-12 |
| 5 | A4 | `optimal` | 56 | -0.008099917964985357 | 7.93e-13 |
| 9 | A0 | **`maxIterations`** | 3000 | 1.3716665864047928 | 2.60e-06 |
| 9 | A4 | `optimal` | 59 | -0.008104095650270938 | 1.56e-12 |

**Node 5 is a genuine control**: its pickled state is already the solved optimum of its own
last cycle (A0 needs 0 IPOPT iterations to re-confirm it), and A4 finds the same point cold in
56 iterations (objective differs from A0 by ~1%, consistent with a slightly different, still
physically reasonable, stationary point one cycle apart — not investigated further, out of
scope).

**Node 9 is NOT a control — this corrects the brief's premise.** A0 on node 9 hits
`maxIterations` at exactly 3000 iterations, with the same signature as node 7 (dual infeasibility
frozen at ~1e3-1e4, `lg(mu)` stuck at -3.8, tiny constraint violation). Direct confirmation from
the untouched historical log: the tail of `optim_log_node_9.txt` at the repository root ends
`EXIT: Maximum Number of Iterations Exceeded.` with unscaled objective
`1.3716665864047899e+00`, matching this run's `1.3716665864047928` to 12 significant figures. So
the P5.14-N perturbation arm (`k=10000`) left **two** of its three captured nodes (7 and 9) in a
failed state, not one; only node 5 converged. This does not weaken H-WS — it is a second,
independent instance on which the same policy change (A4/A1/A2-style) converges cleanly (A4:
59 iterations, small violation) — but the "5 and 9 are controls" framing in the brief should be
corrected going forward.

---

## 3. DSO arms — evidence table

Instance: preserved cycle-21 fixture, `data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl`
(`DSO:case33_3|2025|Spring`, the same fixture P5.12-B/P512R/P512P used). Fresh, independent
reload from disk per arm (`pickle.load` repeated, not a shared in-memory deep copy), matching
the "independent reload" convention `p512_p_path_sensitivity.py` established for this fixture.
Full IPOPT iteration tables: `data/SRP1/Results/P5150/dso/logs/optim_log_case33_3_2025_Spring_<arm>.log`.

| arm | termination | iterations | wall (s) | objective (`p58_rescaled_admm_objective`) | max constraint violation | rel. obj. diff vs A4 |
|---|---|---|---|---|---|---|
| A0 (current production, `network.py:521-534`, resolves to 1e-5 pushes here) | `maxIterations` | 3000 | 16.54 | 1302.2135784698214 | 6.192e-02 | — |
| A1 (IPOPT compiled defaults, all five keys) | `optimal` | 85 | 0.80 | 1293.666817122966 | 7.135e-09 | 3.55e-09 |
| A2 (A1, primal-only warm start) | `optimal` | 89 | 0.85 | 1293.6668166211773 | 7.135e-09 | 3.16e-09 |
| A3 (A1 + `mu_strategy=adaptive`) | `optimal` | 37 | 0.54 | 1293.680074227944 | 6.175e-09 | **1.03e-05** |
| A4 (cold start) | `optimal` | 77 | 0.74 | 1293.6668125362894 | 7.135e-09 | 0 (reference) |

**Reproduction gate: PASSED.** A0 hits `maxIterations` at 3000 iterations, matching the
historical Arm A behaviour already established in P5.12-P (`data/SRP1/Results/P512P/`:
Arm A = `warm_start_bound_push=1e-5` = `maxIterations`, 3000 iterations). The resolved warm-start
pushes for A0 here are `1e-5` (fallback from this fixture's `bound_push=1e-5` production
option), matching P5.12-P's Arm A exactly — a second, independent confirmation of that
1e-5-pushes failure using the production entry point rather than the frozen forensic harness.

**Success criterion: A1 PASSES, A2 PASSES.** A3 misses the strict 1e-6 relative-objective bar
(1.03e-05, roughly 10x the threshold) though it is much closer to A4 than the ESSO case's A3 —
still a different point of the same augmented-Lagrangian NLP, reached via a different barrier
path; not recommended as a blanket policy for the same reason as the ESSO case (mu_strategy
changes which stationary point is found, on a problem that is not shown convex).

---

## 4. H-WS verdict

**ESSO: SUPPORTED.** On both genuine failing instances captured in the P5.14-N perturbation arm
(node 7 and, newly established here, node 9), the current production warm-start policy
(bound-multiplier import with all five pushes/fracs effectively at 1e-9) reproduces
`maxIterations` deterministically, while cold start (A4) and IPOPT-default-scale pushes with
either dual (A1) or primal-only (A2) warm start converge to the same point (A4) in well under
200 iterations with negligible constraint violation. `mu_strategy=adaptive` (A3) converges even
faster but to a materially different local optimum on the ESSO — evidence this NLP has multiple
local optima at this operating point, a separate finding, not a refutation of H-WS.

**DSO: SUPPORTED**, on the one preserved cycle-21 failing instance available, corroborating
P5.12-P's existing conclusion via the production entry point. P5.12-P already established
(2026-09-12, `data/SRP1/Results/P512P/`) that `warm_start_bound_push=1e-6` (variant 1) and
`1e-4` (variant 2) both converge on the same fixture where `1e-5` (Arm A / this run's A0) fails,
explicitly warning "no production parameter change is justified from this single frozen
instance. Two data points, one per side, on one captured failure would be tuning to that
instance." That caution stands: this stage adds three more configurations (A1 IPOPT defaults,
A2 primal-only, A4 cold) on the *same single* fixture, all of which converge — it strengthens
the case that the failure is warm-start-path-dependent rather than a property of the point
itself, but it remains **one fixture**. Unlike the ESSO, no second independent DSO failing
instance was available to test in this task.

---

## 5. Recommended production policy (per solver family)

Not adopted here — production adoption is Step 1's gate run, not authorized in this task.
Stated as the Planner requested, one concrete option block per family.

**ESSO** (`shared_energy_storage_data.py:_create_solver`, replacing the current hardcoded block
at lines 906-911, `from_warm_start=True` branch):

```
solver.options['warm_start_init_point'] = 'yes'
# leave warm_start_bound_push / warm_start_bound_frac / warm_start_slack_bound_push /
# warm_start_slack_bound_frac / warm_start_mult_bound_push UNSET -> IPOPT compiled default
# (0.001), matching Arm A1 in this report. Do not set mu_strategy.
```
(i.e. delete the five `solver.options[...] = 1e-9` lines; keep the `replace_warm_start_suffix`
calls — A2's primal-only variant showed no advantage over A1's dual warm start on either
instance, so there is no evidence to justify also dropping the bound-multiplier import.)

**Networks** (`network.py:_create_smopf_solver`, lines 529-534, `from_warm_start=True` branch):
same recommendation — do not assign any of the five `warm_start_*` keys unless the caller's
`solver_params.options`/`option_overrides` explicitly requested a value; let IPOPT's compiled
default apply otherwise. Concretely, replace the five unconditional `.get(key, options.get(fallback,
1e-6))` assignments with a loop that only sets a key if it is present in `options` (i.e. remove
the `1e-6` fallback default entirely), so a fixture with no explicit push configured gets IPOPT's
own default rather than production's historical `1e-6`.

**Both families:** do not adopt `mu_strategy=adaptive` as a blanket default (A3 landed away from
the reference point on both fixtures, worse on the ESSO). It may be worth a future, narrowly
scoped experiment as a *recovery-path* option only (after a first cold/default-push solve fails),
not as the primary policy — this is a suggestion, not a request for authorization.

### (a) `max_iter`

Recommend setting `max_iter = 500` for the ESSO (already tested as A5: on the still-failing A0
policy it stops 2500 iterations sooner at essentially the same non-converged point — cost of
failure is bounded without changing behaviour on any converging instance, since every converging
arm here finished in ≤ 89 iterations). Recommend an analogous cap for the networks; this task did
not run a network-specific `max_iter=500` cost-of-failure arm (not requested for the DSO part —
only A0-A4 were), so the cited evidence for that specific number is ESSO-only; the Planner may
want a one-arm network check before adopting the same number there.

### (b) `_is_recoverable_shared_ess_failure` / `_is_recoverable_network_failure` extension

Recommend extending both predicates to also fire on `maxIterations` and `infeasible` (currently:
`internalSolverError` only), with a cold retry (`from_warm_start=False`, matching this report's
A4 configuration) as the recovery action. Evidence: A4 converged cleanly on every instance tested
that A0 failed on — ESSO node 7 (59 iterations), ESSO node 9 (59 iterations), and the DSO cycle-21
fixture (77 iterations) — with objective agreement to ≤ 1e-6 relative against the corresponding
A1 in every case tested. A cold retry is therefore a plausible, cheap recovery path for both
failure modes on the evidence collected so far (three instances, all of the same warm-start
mechanism); it has not been tested against a failure with a different root cause (e.g. a genuine
feasibility problem), so `infeasible` recovery should be re-examined once such an instance is
available.

---

## 6. Numbers reused — instance, convention, error bar

All values in the two evidence tables above are single deterministic IPOPT solves (fixed
executable `/usr/local/bin/ipopt`, fixed options, fixed input state — no stochastic element in
the solver or in Python) run once each; IPOPT/ASL is deterministic given identical binary,
options and starting point, and the reproduction gates in both parts confirm this is holding here
(node-7 A0 reproduces the historical log's terminal objective/dual-infeasibility/constraint-
violation fields, and the DSO A0 reproduces P5.12-P's Arm A iteration count and termination
exactly). **Error bar: not applicable / zero, under the "exact bit-for-bit reproduction" precedent
established by P5.12-R/P for this repository, not a statistical estimate** — these are not
averaged or resampled quantities. Instance identity for every number: node id (5/7/9) +
`esso_models_k10000.pkl` for the ESSO table, `DSO:case33_3|2025|Spring` cycle 21 +
`cycle21_pre_setup/snapshot.pkl` for the DSO table, both path-qualified above. Convention: IPOPT's
own "unscaled" NLP-error-summary block for objective/dual-infeasibility/constraint-violation
(matches the historical logs' own convention); max-constraint-violation column is this harness's
own generic re-derivation (max over all active Pyomo constraints of `|body - rhs|` for equalities
or the bound-exceedance amount for inequalities, evaluated after `model.solutions.load_from`),
which is a distinct but consistent convention from IPOPT's own scaled/unscaled report — both are
recorded per arm in `p5150_report.json` (`ipopt_log_tail.nlp_error_summary_unscaled` vs
`max_constraint_violation`) so a reader can tell which convention any given number came from.

---

## Unexpected findings

1. **`option_overrides` is silently clobbered for the five ESSO warm-start keys** whenever
   `from_warm_start=True` (see section 1). This is a latent defect in a documented parameter of
   `_run_solver_attempt`/`_optimize`, independent of the warm-start *value* question this task
   was asked to test. Recommend the Planner treat this as its own small fix (remove or
   reorder the hardcoded block) regardless of what values Step 1 adopts.
2. **Node 9, not just node 7, is a genuine `maxIterations` failure** in the preserved P5.14-N
   perturbation-arm pickle (`esso_models_k10000.pkl`) — contradicting the brief's premise that
   nodes 5 and 9 are both controls. Confirmed independently from the untouched
   `optim_log_node_9.txt` at the repository root (tail: `EXIT: Maximum Number of Iterations
   Exceeded.`, matching unscaled objective to 12 significant figures). Only node 5 converged.
   This makes node 9 a second, useful, independent H-WS test instance (used above), but the
   Planner should not carry forward "node 9 converged" as an established fact from this arm.
3. **`mu_strategy=adaptive` is not solution-path-neutral on either NLP.** It lands on a
   materially different KKT point for the ESSO (6.7% relative objective gap from the reference)
   and a smaller but still threshold-exceeding one for the DSO fixture (1.03e-5 relative, vs the
   1e-6 bar). Neither NLP has been shown convex; this is consistent with, but not new proof of,
   the nonconvexity already on record in `EXPERT_REVIEW_2_ACTION_PLAN.md` §1.
4. Both historical append-only logs at the repository root (`optim_log_node_7.txt`,
   `optim_log_node_9.txt`, `optim_log_node_5.txt`) are large (15-35 MB) because
   `file_append='yes'` accumulates every solve across every arm/run that ever wrote to the
   default `node_id`-keyed filename with no redirect. Not this task's concern to fix, but noted
   since this harness deliberately redirected every one of its own solves to
   `data/SRP1/Results/P5150/.../*.txt|*.log` to avoid adding to those files.

## Remaining issues

- DSO H-WS rests on a single preserved failing fixture (cycle 21, `case33_3`); no second
  independent DSO failure instance was available to corroborate the way node 9 did for the ESSO.
- `max_iter=500` was tested as a cost-of-failure check only for the ESSO (A5); not tested on the
  network family.
- A3 (`mu_strategy=adaptive`) not being solution-path-neutral means any future exploration of it
  as a *recovery* option (not primary policy) would need its own equal-objective gate before
  being trusted, exactly as this task applied to A1/A2.
- This task did not attempt to reconcile IPOPT version reported in different logs (3.14.18 in
  this run and the historical node-7 tail; 3.14.20 in the historical log's very first header for
  an earlier cycle of the same file) — both point at `/usr/local/bin/ipopt`; not investigated
  further, out of scope for this task.

## Questions for Planner

1. Should the `option_overrides`-clobbering defect (finding 1) be scheduled as its own tiny fix
   ahead of or alongside Step 1, given it is independent of the warm-start *value* decision?
2. Given node 9's corrected status, should the Planner re-check whether any other artifact in the
   repository still asserts "node 9 converged" for the `k=10000` perturbation arm (e.g.
   `n1_k10000.json`'s per-node breakdown, if any) before this correction propagates further?
3. Do you want a one-arm `max_iter=500` cost-of-failure check on the network family before Step 1,
   given (a) of the max_iter recommendation was only demonstrated for the ESSO here?
