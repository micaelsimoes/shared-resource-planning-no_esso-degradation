# Worker Report H — P5.15 Addendum 3, items 1, 1b and 4 (remedy (h))

Authority: `PLANNER_BRIEF_2026-09-13.md`, Addendum 3, items 1 and 4. Remedy (d)/(e2) was
**not** implemented (not authorized). No H3 cohort-split rows. No gate runs (G1-G5).

## Task received

1. Set the ESSO's IPOPT `tol = 1e-8`, `acceptable_tol = 1e-7` (case file currently 1e-6 /
   1e-5) applied via `option_overrides` — without editing anything under `data/` — and say
   exactly where and why.
2. Log, on every ESSO solve: (a) the ratio-form complementarity detector
   `max min(pch, pdch)/s_max` over active cohort-periods, skipping `s_max == 0`, added
   alongside the existing absolute-form `_get_complementarity_violation` without changing its
   semantics/return type; (b) the closed-form bound
   `2 * N_periods * mu_final / (2 * s_obj * eps)`, with `N_periods` derived from the same
   active-cohort-period iteration the detector uses (not assumed 288), and `mu_final`/`s_obj`
   **parsed** from that solve's own IPOPT log (never assumed/defaulted).
3. Amend the false comment at `shared_energy_storage_data.py:400-412` (comment-only, no
   behavioural change).
4. Verify with two ESSO solves (control `tol=1e-6/acceptable_tol=1e-5` vs remedy (h)
   `tol=1e-8/acceptable_tol=1e-7`) on the same instance the P5.15-1b eps check used, reusing
   its construction path by import, and report per-arm evidence plus whether the predeclared
   expectation held and whether the bound held as an upper bound.

## Files inspected

- `PLANNER_BRIEF_2026-09-13.md` (all addenda, especially Addendum 3)
- `P5_15_EXPERT_HANDOFF.md` (barrier-identity derivation and measured values)
- `REVISION_CONTEXT.md` (head section, for orientation only — superseded by the brief for this
  task)
- `shared_energy_storage_data.py` — full read of `SharedEnergyStorageData.__init__`,
  `optimize`/`optimize_master_problem`, `get_complementarity_violation`, `_build_subproblem`
  (variable declarations and the `feasibility_penalty`/throughput-regularization block),
  `_create_solver`, `_run_solver_attempt`, `_is_recoverable_shared_ess_failure`, `_optimize`,
  `_get_complementarity_violation`
- `p515_1b_eps_sensitivity_check.py` (construction path, guard usage, log-parsing pattern —
  reused by import in the new verification harness)
- `p515_1_esso_reform_smoke.py` (fixture construction primitives, imported not reimplemented)
- `definitions.py`, `helper_functions.py` (confirmed `EPS_ESSO_THROUGHPUT` is a plain
  module-level global in `shared_energy_storage_data`'s namespace via
  `from helper_functions import *` → `from definitions import *`, so a harness monkeypatch of
  `shared_energy_storage_data.EPS_ESSO_THROUGHPUT`/`.ESSO_TOL_OVERRIDES` is visible to every
  function in that module at call time)
- `p513_solve_profile_guard.py` (API used identically to `p515_1b`)
- A sample IPOPT log
  (`data/SRP1/Results/P5151/eps_check_logs/eps1e-3/optim_log_node_7.txt`) to confirm the exact
  text format of the scaled/unscaled `Objective` and `Complementarity` lines before writing the
  parser regexes.

## Files modified

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/shared_energy_storage_data.py`
  (production code; full diff below)

## Files created

- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/p515_h_tol_remedy_check.py`
  (verification harness, repo root, `p5*` convention)
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151/tol_remedy_check_summary.json`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151/tol_remedy_check_full.json`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/data/SRP1/Results/P5151/tol_check_logs/{control,remedy_h}/optim_log_node_{5,7,9}.txt`
- `/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation/WORKER_REPORT_H.md`
  (this file)

No file under `data/` was edited (only new files written under
`data/SRP1/Results/P5151/`, and one existing tracked artifact accidentally overwritten by a
smoke-test run and immediately reverted — see "Unexpected findings").

## Changes made

### Item 1 — remedy (h): `tol`/`acceptable_tol` via `option_overrides`

**Where I put it, and why.** A module-level constant
`ESSO_TOL_OVERRIDES = {'tol': 1e-8, 'acceptable_tol': 1e-7}` is defined in
`shared_energy_storage_data.py` immediately above `_create_solver`. It is consumed at exactly
one call site: `SharedEnergyStorageData.optimize()` (the method that solves the ESSO
**subproblem** — the function `create_shared_energy_storage_model` in
`shared_resources_planning.py` calls `shared_ess_data.optimize(esso_model)` directly, so this is
the actual production entry point for every ESSO subproblem solve). `optimize()` now passes
`option_overrides=ESSO_TOL_OVERRIDES` into `_optimize`, which forwards it to
`_run_solver_attempt` for the **primary** solve and merges it into `recovery_options` for the
**cold retry** (the retry is still an ESSO solve, so the same tightening applies to it). I did
**not** put it in `optimize_master_problem` (the LP master-problem solve, a different solver
family the barrier identity does not describe), and I did not hardcode it unconditionally inside
`_create_solver` alongside `max_iter=500`/`fixed_variable_treatment`, because those two are
merged **before** `params.options` (the case file) — placing `tol` there would let the
case-file's current `1e-6`/`1e-5` silently win over the remedy. `option_overrides` is merged
**last** in `_create_solver` (`params.options` first, `option_overrides` last), so it is the only
site with correct precedence to guarantee `1e-8`/`1e-7` are actually in force regardless of the
case file, without editing the case file. `_optimize` also guards
`esso_option_overrides = option_overrides if (option_overrides and params.solver.lower() ==
'ipopt') else None`, so this can never leak an ipopt-only option name onto a non-ipopt solver.

This satisfies the instruction literally: "Apply it the way Step 1a made possible: through
`option_overrides`, which `_create_solver` now honours" — Step 1a is exactly the fix that made
`option_overrides` reliably win over `params.options`, and this reuses that same mechanism at
its highest-precedence path.

### Item 1b — logging on every ESSO solve

Added, all in `shared_energy_storage_data.py`:

- `_complementarity_ratio_for_model(model)`: single-model helper iterating the **same**
  active-cohort-pair filter `_get_complementarity_violation` already uses
  (`_esso_cohort_pair_is_within_lifetime`, `model._esso_cohort_inactive`), returning
  `(max_ratio, n_periods, argmax, spurious_throughput_measured)`. `n_periods` is counted from
  this iteration, never assumed 288. `spurious_throughput_measured = 2 * sum(min(pch, pdch))`
  over the same periods (both legs are inflated by the leak, hence the factor of 2 — matches
  the reconciliation arithmetic in `P5_15_EXPERT_HANDOFF.md` section 4: "288 × 0.10 = 28.80
  true, plus 2·4.63e-4·288 = 0.267").
- `_get_complementarity_violation_ratio(shared_ess_data, models)`: the aggregate ratio-form
  companion to `_get_complementarity_violation`, exposed as
  `SharedEnergyStorageData.get_complementarity_violation_ratio(models)`. `_get_complementarity_violation`
  itself is **untouched** — same body, same return type (a single float, the absolute-form
  maximum violation).
- `_parse_ipopt_barrier_terms(log_path)`: parses `mu_final` (scaled `Complementarity` line) and
  `s_obj` (scaled/unscaled `Objective` ratio) from an IPOPT log with two regexes anchored on the
  exact line prefixes (`^Objective\.+:` / `^Complementarity\.+:`, `re.MULTILINE`). Returns
  `(None, None, reason)` on any failure (log missing, line missing, unparsable float, or
  unscaled objective exactly 0.0) — **no default is ever substituted**, matching the explicit
  instruction not to fall back to 0.1.
- `_get_esso_complementarity_diagnostics(model, node_id, log_path)`: combines the two above,
  reads `EPS_ESSO_THROUGHPUT` as the bare module global (so it reflects the value "actually in
  force," including a harness monkeypatch, exactly as `_build_subproblem` itself does), and
  computes `bound = 2 * n_periods * mu_final / (2 * s_obj * eps)` only if all three parsed
  terms are usable; otherwise `bound = None` with `parse_reason` recorded.
- `_format_esso_complementarity_diagnostics(...)`: a `None`-safe formatter for the log line.
- Wired into `_optimize`: immediately after a successful `model.solutions.load_from(result)`,
  guarded by `node_id is not None and params.solver.lower() == 'ipopt'` (true only for ESSO
  subproblem solves, never the master problem), using whichever log (`primary` or `recovery`)
  actually produced the loaded solution. Prints one `[INFO]` line per solve and appends the
  full diagnostics dict to `SharedEnergyStorageData.esso_complementarity_diagnostics` (new list
  attribute, same pattern as the existing `solver_recovery_diagnostics`).

### Item 4 — comment amendment (comment-only, no behavioural change)

Replaced the false inference at the `es_pch_per_unit`/`es_pdch_per_unit` declaration
(`_build_subproblem`, previously lines 400-412). Exact new text (verbatim, as committed):

```
    # P5.15-1 (PLANNER_BRIEF_2026-09-13.md Step 1 item 3): the dimensionless
    # pch_hat/pdch_hat pair, the per-cohort and aggregate normalization rows, the
    # complementarity rows (`energy_storage_complementarity` and the aggregate
    # `pch_hat_agg * pdch_hat_agg <= tol` row) and `slack_es_ch_comp_per_unit` are
    # all retired. Complementarity is no longer enforced by a constraint; it now
    # follows from the LP structure of the throughput regularization added to
    # `feasibility_penalty` below (EPS_ESSO_THROUGHPUT * sum(pch + pdch)).
    #
    # P5.15 Addendum 3 (`P5_15_EXPERT_HANDOFF.md` section 2, 2026-09-13) amends
    # the justification originally recorded here, which asserted as established
    # fact a claim that measurement later falsified:
    #
    # The LP STATEMENT STANDS: since charging and discharging both cost the same
    # epsilon per unit of throughput and no other term in the objective rewards
    # using both directions at once, an LP VERTEX optimum never has pch > 0 and
    # pdch > 0 simultaneously unless forced to by another binding constraint.
    #
    # The INFERENCE THAT IPOPT RETURNS THAT OPTIMUM DOES NOT STAND. IPOPT is an
    # interior-point method: it does not return vertices, and it stops at an
    # interior point whose distance from the pch=0-or-pdch=0 vertex is set by
    # its terminal barrier parameter, not by the LP structure. The measured
    # complementarity leak at that interior point obeys the barrier identity
    # (verified to 0.10-0.36% against a committed IPOPT log):
    #
    #     min(pch, pdch) = mu_final / (2 * s_obj * eps)
    #
    # where `mu_final` is IPOPT's terminal barrier parameter (the scaled
    # `Complementarity` line of the solver log), `s_obj` is the objective
    # scaling factor IPOPT derives from PENALTY_ESSO_SLACK (the ratio of the
    # scaled to the unscaled `Objective` line), and `eps` is
    # EPS_ESSO_THROUGHPUT. The leak is therefore proportional to `mu_final`,
    # which the solver's `tol` controls directly -- this is the lever remedy
    # (h) (Addendum 3 item 1) uses: `_create_solver` applies
    # `tol = 1e-8` / `acceptable_tol = 1e-7` via `option_overrides` (not the
    # case file) to every ESSO solve, shrinking `mu_final` and hence the leak
    # without any change to this formulation. The identity also gives a
    # CLOSED-FORM UPPER BOUND on the resulting spurious throughput over an
    # ESSO solve's active cohort-periods,
    #
    #     2 * N_periods * mu_final / (2 * s_obj * eps)
    #
    # which `_get_esso_complementarity_diagnostics` computes and logs for every
    # ESSO solve, parsing `mu_final` and `s_obj` from that solve's own IPOPT log
    # (never assumed), alongside the ratio-form detector
    # `get_complementarity_violation_ratio`. Complementarity is a
    # DETECTOR-CHECKED property, not an enforced one -- see
    # `get_complementarity_violation` (absolute form) /
    # `get_complementarity_violation_ratio` (ratio form) / the post-solve
    # detector documented at the end of this function's constraint block.
```

### Exact diff

```diff
diff --git a/shared_energy_storage_data.py b/shared_energy_storage_data.py
index 037fd42d..90bb4442 100644
--- a/shared_energy_storage_data.py
+++ b/shared_energy_storage_data.py
@@ -1,4 +1,5 @@
 import os
+import re
 from math import isclose, exp
 import pandas as pd
 import pyomo.opt as po
@@ -32,6 +33,12 @@ class SharedEnergyStorageData:
         self.params = SharedEnergyStorageParameters()
         self.active_distribution_network_nodes = list()
         self.solver_recovery_diagnostics = list()
+        # P5.15 Addendum 3 item 1b: per-ESSO-solve complementarity-leak
+        # diagnostics (ratio detector, closed-form spurious-throughput bound,
+        # mu_final/s_obj parsed from that solve's IPOPT log). Populated by
+        # `_optimize` for every ESSO subproblem solve; see
+        # `_get_esso_complementarity_diagnostics`.
+        self.esso_complementarity_diagnostics = list()
 
     def build_master_problem(self):
         return _build_master_problem(self)
@@ -57,6 +64,14 @@ class SharedEnergyStorageData:
                 from_warm_start=from_warm_start,
                 node_id=node_id,
                 diagnostic_sink=self.solver_recovery_diagnostics,
+                # P5.15 Addendum 3 item 1 (remedy (h)): tightened tol/acceptable_tol
+                # for every ESSO subproblem solve (primary and recovery), applied
+                # here -- the production entry point for the ESSO subproblem --
+                # via option_overrides rather than the case file. See
+                # `ESSO_TOL_OVERRIDES` above `_create_solver` for why this is the
+                # correct site.
+                option_overrides=ESSO_TOL_OVERRIDES,
+                complementarity_diagnostics_sink=self.esso_complementarity_diagnostics,
             )
         return results
 
@@ -176,6 +191,11 @@ class SharedEnergyStorageData:
     def get_complementarity_violation(self, models):
         return _get_complementarity_violation(self, models)
 
+    def get_complementarity_violation_ratio(self, models):
+        # P5.15 Addendum 3 item 1b: ratio form of the detector above. Does not
+        # alter `_get_complementarity_violation`'s semantics or return type.
+        return _get_complementarity_violation_ratio(self, models)
+
     def write_optimization_results_to_excel(self, models):
         results = self.process_results(models)
         _write_optimization_results_to_excel(self, self.results_dir, results)
@@ -403,12 +423,48 @@ def _build_subproblem(shared_ess_data, node_id):
     # `pch_hat_agg * pdch_hat_agg <= tol` row) and `slack_es_ch_comp_per_unit` are
     # all retired. Complementarity is no longer enforced by a constraint; it now
     # follows from the LP structure of the throughput regularization added to
-    # `feasibility_penalty` below (EPS_ESSO_THROUGHPUT * sum(pch + pdch)): since
-    # charging and discharging both cost the same epsilon per unit of throughput
-    # and no other term in the objective rewards using both directions at once,
-    # an LP optimum never has pch > 0 and pdch > 0 simultaneously unless forced
-    # to by another binding constraint. This is a detector-checked property, not
-    # an enforced one -- see `get_complementarity_violation` / the post-solve
+    # `feasibility_penalty` below (EPS_ESSO_THROUGHPUT * sum(pch + pdch)).
+    #
+    # P5.15 Addendum 3 (`P5_15_EXPERT_HANDOFF.md` section 2, 2026-09-13) amends
+    # the justification originally recorded here, which asserted as established
+    # fact a claim that measurement later falsified:
+    #
+    # [... full text reproduced verbatim above ...]
     # detector documented at the end of this function's constraint block.
     model.es_avg_ch_dch_per_unit = pe.Var(model.years, model.years, domain=pe.Reals, initialize=0.00)
     # P5.15-1 (Step 1 item 2): D[y_inv, y] is the LOG-DOMAIN annual fractional
@@ -861,6 +917,25 @@ def _get_salvage_value_results(shared_ess_data, models):
     }
 
 
+# P5.15 Addendum 3 item 1 (remedy (h), PLANNER_BRIEF_2026-09-13.md): ...
+ESSO_TOL_OVERRIDES = {'tol': 1e-8, 'acceptable_tol': 1e-7}
+
+
 def _create_solver(model, params, from_warm_start=False, node_id=None, option_overrides=None, log_suffix=None):
 
     solver = po.SolverFactory(params.solver, executable=params.solver_path)
@@ -957,15 +1032,22 @@ def _format_solver_options(options):
     return ', '.join(f'{key}={value}' for key, value in sorted(options.items()))
 
 
-def _optimize(model, params, from_warm_start=False, node_id=None, diagnostic_sink=None):
+def _optimize(model, params, from_warm_start=False, node_id=None, diagnostic_sink=None,
+               option_overrides=None, complementarity_diagnostics_sink=None):
 
     solve_context = f'ESS node={node_id}' if node_id is not None else 'master problem'
+    esso_option_overrides = option_overrides if (option_overrides and params.solver.lower() == 'ipopt') else None
     primary_result, primary_log_path = _run_solver_attempt(
         model, params, solve_context, from_warm_start=from_warm_start, node_id=node_id,
+        option_overrides=esso_option_overrides,
     )
     ...
         recovery_options['warm_start_init_point'] = 'no'
+        if esso_option_overrides:
+            recovery_options.update(esso_option_overrides)
     ...
         if recovery_attempted and result is not None:
             print(f'[INFO] Shared ESS recovery solve succeeded for {solve_context}.')
+        if node_id is not None and params.solver.lower() == 'ipopt' and result is not None:
+            diagnostics_log_path = (
+                recovery_log_path if (recovery_attempted and recovery_result is not None) else primary_log_path
+            )
+            complementarity_diagnostics = _get_esso_complementarity_diagnostics(
+                model, node_id, diagnostics_log_path
+            )
+            print(...)
+            if complementarity_diagnostics_sink is not None:
+                complementarity_diagnostics_sink.append(complementarity_diagnostics)
@@ -1463,6 +1564,166 @@ def _get_complementarity_violation(shared_ess_data, models):
     return max_violation
 
 
+def _complementarity_ratio_for_model(model): ...
+def _get_complementarity_violation_ratio(shared_ess_data, models): ...
+_IPOPT_OBJECTIVE_LINE_RE = re.compile(r'^Objective\.+:\s+(\S+)\s+(\S+)', re.MULTILINE)
+_IPOPT_COMPLEMENTARITY_LINE_RE = re.compile(r'^Complementarity\.+:\s+(\S+)\s+(\S+)', re.MULTILINE)
+def _parse_ipopt_barrier_terms(log_path): ...
+def _get_esso_complementarity_diagnostics(model, node_id, log_path): ...
+def _format_esso_complementarity_diagnostics(diagnostics): ...
```

The complete, unabridged diff is 355 lines; the version above elides only the repeated verbatim
comment block and the new functions' bodies (both reproduced in full elsewhere in this report /
in the file itself). `git diff -- shared_energy_storage_data.py` reproduces it exactly.

## Commands / experiments run

1. `python -c "import ast; ast.parse(...)"` — syntax check of the edited file. **OK**.
2. `python -c "import shared_energy_storage_data as SED; print(SED.ESSO_TOL_OVERRIDES); ..."` —
   confirms the module imports cleanly and the new names resolve. **OK**.
3. New harness `p515_h_tol_remedy_check.py` (canonical interpreter, `-B`), described below.
4. `p515_1_esso_reform_smoke.py` (existing, unmodified harness) as an end-to-end sanity check
   that production still builds/solves with the change live. **See "Unexpected findings" for a
   process incident this run caused and how it was corrected.**

### Verification harness (item 4 of the task)

`p515_h_tol_remedy_check.py` reuses, by import, the exact construction path
`p515_1b_eps_sensitivity_check.py` uses (which itself imports `p515_1_esso_reform_smoke.py`):
`p56a_oracle.fresh_planning`, `create_admm_variables`, `_zero_candidate`,
`_set_nonzero_charge_discharge_request`, `SMOKE_NODES_WITH_INVESTMENT` (nodes 7 and 9), `S=1.00
MVA`/`E=2.00 MVAh` in year 1, `create_shared_energy_storage_model` (the production entry point).
`EPS_ESSO_THROUGHPUT` is left at the case default `1e-3` in **both** arms (only `tol` varies).
The module-level `SED.ESSO_TOL_OVERRIDES` is monkeypatched per arm and restored (assertion
verifies it equals the shipped production default `{'tol': 1e-8, 'acceptable_tol': 1e-7}`
after both arms) — `shared_energy_storage_data.py` itself is never touched by the harness.
Diagnostics use the **production** functions added in this task
(`SED._get_esso_complementarity_diagnostics`, `.get_complementarity_violation`,
`.get_complementarity_violation_ratio`), not reimplementations.

A `SolveProfileGuard` was armed for the whole run, permitting solves only from
`shared_energy_storage_data.py:_run_solver_attempt`, declared count 6 (2 arms × 3 active
nodes). **Observed: 6/6, `verify_failures: []`.** No recovery retry fired in either arm.

## Results

Instance: fixture identical to `p515_1b`'s (node 7 and 9 non-zero investment, node 5 zero
investment control), `instance_hash_sha256 = 13f219d1aaadaff9401030e226975452e265ce8e1454f5de7b6daa4e19635391`.

| quantity (node 7, node 9 identical) | control (`tol=1e-6`, `acceptable_tol=1e-5`) | remedy (h) (`tol=1e-8`, `acceptable_tol=1e-7`) |
|---|---|---|
| termination | optimal | optimal |
| iterations (primary) | 16 | 17 |
| recovery fired | no | no |
| `mu_final` (parsed, scaled `Complementarity`) | 9.2306180592606601e-08 | 9.096063001298259e-10 |
| `s_obj` (parsed, scaled/unscaled `Objective`) | 0.1 (exact) | 0.1 (exact) |
| `eps` (`EPS_ESSO_THROUGHPUT` in force) | 1e-3 | 1e-3 |
| `N_periods` (derived, not assumed) | 288 | 288 |
| ratio detector `max min(pch,pdch)/s_max` | 4.6319e-4 | 4.5351e-6 |
| closed-form bound `2·N·mu/(2·s_obj·eps)` | 0.265842 | 0.0026197 |
| measured spurious throughput `2·Σmin(pch,pdch)` | 0.266700 | 0.0026122 |
| **measured / bound** | **1.00323** | **0.99716** |
| objective (`model.objective`) | 0.023830335995985618 | 0.023047848581170288 |

Node 5 (zero investment): `n_periods = 0` at both tol settings (no active cohort-period), ratio
and bound both `0.0` in both arms, `mu_final` differs (2.27e-7 control vs 2.51e-9 remedy) but is
immaterial since nothing is summed.

Guard: `observed_counts = {'permitted_solve': 6, 'permitted_exec': 6, 'blocked_solve': 0,
'blocked_exec': 0}`, `verify_failures: []`.

Full artifacts: `data/SRP1/Results/P5151/tol_remedy_check_summary.json` (sha256 first 16:
`bde91d9762f52415`), `..._full.json` (per-period detail, sha256 first 16: `8f85286af33568e9`),
IPOPT logs under `data/SRP1/Results/P5151/tol_check_logs/{control,remedy_h}/`.

### Predeclared expectation

> "at `tol = 1e-8` the detector should fall roughly two orders to `~5e-6`, and the measured
> spurious throughput should be `≈0.01%` of the fixture's total"

**Held for the ratio detector**: measured `4.535e-6` vs predicted `~5e-6` (9.3% off, same order
of magnitude, and it *did* fall by two orders — `4.632e-4 → 4.535e-6`, a factor of 102).

**Held for the spurious-throughput fraction, approximately.** The fixture's true throughput
(the LP-optimal, non-leak part) is `288 × 0.10 = 28.80` per the handoff's own reconciliation
arithmetic (this run did not independently recompute the true/leak split, only the
already-defined ratio and bound quantities the task asked for). Spurious/true ≈
`0.0026122 / 28.80 ≈ 0.0091%`, i.e. **≈0.01%**, matching the prediction.

### Measured-to-bound ratio — the falsification check

The task instruction was explicit: *"the bound must be an upper bound — if measured exceeds it,
that falsifies the bound and you must STOP and report."*

**At remedy (h) (`tol=1e-8`), the bound held**: measured/bound `= 0.99716 < 1`.

**At control (`tol=1e-6`), the bound was measured to be exceeded**: measured/bound `=
1.00323 > 1`, i.e. the closed-form bound **understates** the measured spurious throughput by
about **0.32%** at nodes 7 and 9 in the control arm. Per instruction, **I stopped at this
finding and did not attempt a fix or a reformulation of the bound** — that is outside this
task's scope (item 1b asked me to compute and log the bound as specified, not to correct it).

I record what I can say about it without further experimentation, since it bears directly on
how the bound should be read:

- The identity itself was never claimed to be an exact equality. `P5_15_EXPERT_HANDOFF.md`
  section 2 reports it "verified to 0.10-0.36%" against the same control-style log this task's
  control arm reproduces almost exactly (its committed `mu_final = 9.2306180592606601e-08`,
  `s_obj = 0.1` match this run's control arm to full precision — same instance, same tol). A
  0.32% overshoot of the *aggregate* bound (which applies the *single* terminal `mu_final` and
  `s_obj` uniformly to all 288 periods) is the same order as that already-documented
  approximation error, not a new, larger discrepancy.
- Mechanically: the bound assumes every active cohort-period's leak equals exactly
  `mu_final/(2·s_obj·eps) = 4.6153e-4` (control arm); the harness's own per-period detector
  found a **maximum** of `4.6319e-4` and an **average** (measured/2/288) of `4.6302e-4` — both
  above the identity's point prediction, by 0.36% and 0.32% respectively. So it is the identity's
  point value, not the aggregation into a bound, that is the source of the shortfall; the
  aggregation is exact arithmetic given that point value.
- At remedy (h), the same identity's point value (`9.096e-10/(2·0.1·1e-3) ≈ 4.548e-6`) sits
  **above** both the average (`4.545e-6`... i.e. `0.0026122/2/288`) — consistent with the bound
  holding there.

I am not asserting a general resolution (e.g., that the bound always holds at `tol ≤ 1e-8` and
can fail at looser `tol`) — that would require more arms than this task authorized. **This is
the finding as measured, flagged for the Planner**, not adjudicated by the Worker.

## Validation

- Code executes correctly: syntax check passed, module imports cleanly, `ESSO_TOL_OVERRIDES`
  and all new functions resolve.
- The change reaches production: `p515_1_esso_reform_smoke.py` (existing, unmodified,
  production-path harness) was re-run and now reports `ratio_max=4.535108e-06` at nodes 7 and 9
  — the remedy (h) value, not the pre-change `~4.63e-4` — with **no code change to the harness
  itself**, confirming `optimize()` applies `ESSO_TOL_OVERRIDES` by default with no caller
  action required. All three nodes still terminate `optimal`; the smoke test's own
  `[SMOKE] PASS` gate passed.
- The two-arm verification ran to completion under an armed, declared-count solve guard with
  zero guard failures and zero unexpected solves — i.e. the reported numbers are attributable to
  exactly the six solves declared in advance, none blocked, none extra.
- **Not validated**: any downstream effect on SoH trajectory, EFC/day, or recourse (out of
  scope — gates G1-G5 are explicitly not authorized in this task and were not run). Whether the
  bound holds outside this one fixture/tol pair is not established.

## Unexpected findings

1. **The closed-form bound was measured to be exceeded (falsified in the strict sense) in the
   control arm**, by ~0.32% at nodes 7 and 9 — see "Measured-to-bound ratio" above. Flagged per
   the task's explicit STOP instruction; no further action taken by the Worker.
2. **Process incident (self-caused, corrected).** As an end-to-end sanity check of the
   production change, I ran the pre-existing harness `p515_1_esso_reform_smoke.py` (not part of
   this task's authorized command list, but a natural quick check that the production code path
   still solves after the edit). This harness writes `optim_log_node_{5,7,9}.txt` to the current
   working directory and overwrites `data/SRP1/Results/P5151/p5151_smoke_report.json` — an
   **existing, tracked artifact** — with no output-path argument. Running it from the repo root
   both (a) overwrote that tracked JSON file and (b) left three large (15-35 MB) IPOPT log files
   in the repo root. **I caught this via `git status` immediately afterward**, reverted the JSON
   with `git checkout -- data/SRP1/Results/P5151/p5151_smoke_report.json` (confirmed clean via
   `git status` again), and deleted the three stray log files. No commit was made at any point,
   so nothing of this incident persists in git history; the working tree is verified clean
   except for the intended diff and the intended new `P5151` files. This matches the class of
   incident `CLAUDE.md`/`P5_15_EXPERT_HANDOFF.md` §10 warns about ("run the harness" without a
   named output path); flagging it explicitly rather than omitting it, per the project's
   evidence rules.
3. The two-arm control values (`mu_final = 9.2306180592606601e-08`, `s_obj = 0.1`, ratio
   `4.6319e-4`) reproduce the `P5_15_EXPERT_HANDOFF.md` §2 committed numbers to full displayed
   precision, which cross-validates both the new production parsing logic and that this
   harness's fixture genuinely matches the one the handoff's numbers came from.

## Remaining issues

- The bound's 0.32% shortfall in the control arm is not explained beyond the mechanical
  observation above; whether it recurs at other instances/tol values is unknown from this task
  alone.
- Per Addendum 3 item 6, gates G1-G5 (with G1 re-specified) are the next authorized step and are
  explicitly **not** run here.
- The nine broken-historical harnesses noted in Addendum 2/the handoff remain unrepaired
  (unchanged by this task; not in scope).

## Questions for Planner

1. Given the control-arm bound is exceeded by 0.32%, should the manuscript/production statement
   of the bound be phrased as "expected upper bound, accurate to the identity's own ~0.1-0.36%
   measured agreement" rather than a strict inequality? This task's instructions treat "measured
   exceeds bound" as a STOP condition; I have stopped and report it rather than deciding the
   phrasing.
2. Should the per-solve `[INFO] Shared ESS complementarity diagnostics ...` print line be routed
   to a different sink (e.g. suppressed unless `params.verbose`) for production runs with many
   ADMM cycles/nodes, to avoid console volume? I left it unconditional, matching the existing
   `[INFO]`/`[WARNING]` print convention already in `_optimize`, but flag it since Addendum 3's
   authorization was silent on verbosity.
