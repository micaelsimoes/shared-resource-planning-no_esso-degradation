# Worker Report -- s32 Boyd mapping diagnostic (bounded, no production-code changes)

## Task received

Explain an inconsistency the Planner found in the committed cycle-1 s32
preflight (`data/SRP1/Results/P515S32/preflight/g_preflight.json`, commits
`2d765573` production / `7305a476` harness): under rho = gamma = 1, with
lambda^0 assumed zero, ||y_v|| should equal rho*||r_v|| = 0.101 but is
observed as 0.0375 (matching rho*||Delta z_v|| instead). Discriminate
hypotheses Ha-Hd by running a NEW, read-only-instrumented one-cycle diagnostic
through the real s32 arm machinery, into a NEW output directory. **Not**
authorized to launch the 150-cycle gate; no production-code edits permitted.

## Files inspected

- `data/SRP1/Results/P515S32/preflight/g_preflight.json`,
  `preflight_verification.json` (observation source).
- `WORKER_REPORT_S32_IMPL.md` (s32 implementation report -- F1/F4 mapping,
  zero-solve check 2's legacy-vs-Boyd term breakdown).
- `shared_resources_planning.py`:
  - `_run_operational_planning` cold-start initialization block (:2340-2429),
    in particular the pre-ADMM-loop call
    `planning_problem.update_interface_power_flow_variables(tso_model,
    dso_models, consensus_vars, dual_vars, results, admm_parameters,
    update_tn=True, update_dns=True)` at **:2421**, preceded by the comment
    "Initialize only the TSO-DSO interface coordination here. Shared-ESS
    dual variables must remain zero before the first consensus-ADMM cycle."
    (**:2419-2420**).
  - The ADMM main loop (:2444-2534): DSO solve -> `update_and_check_convergence`
    (`update_dns=True` only) -> TSO solve -> `update_and_check_convergence`
    (`update_tn=True` only) -> ESSO solve -> `update_and_check_convergence`
    (`update_sess=True` only) -> `get_admm_residual_metrics` (legacy,
    read-only) -> `get_admm_boyd_residual_metrics`.
  - `_update_interface_power_flow_variables` (**:6242-6322**): the vmag/pf
    "current"/"prev" refresh blocks are gated on `update_tn`/`update_dns`
    individually; the "Update Lambdas" block (**:6291-6322**) is gated ONLY
    on `update_tn and tso_succeeded and dso_succeeded` (**:6307, :6317**),
    with no check for whether the call originates from the pre-loop
    initialization or from inside the ADMM `for` loop.
  - `create_admm_variables` (**:3725-3820**): `dual_variables['vmag']` /
    `['pf']` are shaped `{'tso': {'current': dict()}, 'dso': {'current':
    dict()}}` (no `'prev'` key at all) and initialized to `0.0` everywhere;
    `dual_variables['ess']` likewise starts at `0.0`.
  - `get_admm_boyd_residual_metrics` (:5519-5750) -- read (not modified).
  - `update_and_check_convergence` (:2980-3009) -- read (not modified).
- `p515_s32_preflight.py` (structure to mirror: precheck ->
  `run_admm_arm(..., num_max_iters_override=1, apply_rho=False,
  full_diagnostics_in_rows=True)`).
- `p515_g_g1_g4_admm_gates.py`: `run_admm_arm`, `_construct_arm_planning`,
  `assert_s32_capture_paths` (source-text check on
  `get_admm_boyd_residual_metrics` -- see "Deviation" below),
  `SolveProfileGuard` usage (`N.PERMITTED`, `identity_holds` formula
  `51*len(rows)+51`).
- `p513_solve_profile_guard.py` (`SolveProfileGuard` API).
- `data/SRP1/SRP1_params.json` `admm.rho` block (all four networks' V/PF/ESS
  rho = 1.0 -- rules out a rho-mismatch explanation).

## Files modified

None in production, case-file, or existing-harness code. New files only:

- New: `p515_s32_mapping_diag.py` (diagnostic script).
- New (results): `data/SRP1/Results/P515S32/diag_mapping/*` (produced by
  `run_admm_arm` itself: `g_diag_mapping.json`, `heartbeat_diag_mapping.json`,
  `stdout_diag_mapping.log`, `leak_classification_diag_mapping.jsonl`,
  `network_failures_diag_mapping.jsonl`, `frozen_snapshots_diag_mapping.jsonl`,
  `esso_recovery_events_diag_mapping.jsonl`, `esso_models_diag_mapping.pkl`,
  `esso_capture/diag_mapping/*.jsonl`, empty `results/`) plus two new files
  this script writes itself (`mapping_diag_summary.json`,
  `raw_snapshots.pkl`), plus `manifest_sha256.json`.

`preflight/` was never touched (verified: no write calls target that path;
`_require_fresh_output_root`/`_refuse_overwrite` are only ever called against
`OUT_DIR = .../diag_mapping`).

## Changes made

Wrote `p515_s32_mapping_diag.py`. It:

1. Runs the capture-path precheck (`G.assert_s32_capture_paths`) on a fresh,
   never-solved planning object **before** installing any instrumentation
   (needed because that precheck reads `inspect.getsource(srp.
   get_admm_boyd_residual_metrics)` and asserts literal field names in the
   *production* source -- patching first would make it inspect the wrapper
   instead and fail spuriously; this was caught by a first run that raised
   `RuntimeError: S32 capture-path pre-flight FAILED` and fixed by moving the
   patch installation to run strictly after the precheck).
2. Monkeypatches (module-attribute reassignment on `shared_resources_planning`,
   this script only) three functions with wrappers that deep-copy snapshot
   `consensus_vars['vmag'|'pf'|'ess']` and `dual_vars['vmag'|'pf'|'ess']`
   immediately before and after calling the **saved original, unchanged**:
   - `_update_interface_power_flow_variables` -- logs every call, in order,
     with its `update_tn`/`update_dns` flags and a tag derived from them
     (`pre_admm_init`, `cycle1_dso`, `cycle1_tso`, `cycle1_esso`), plus the
     rho Params (`rho_v`, `rho_pf`, both TSO and DSO copies) read just before
     calling the original.
   - `get_admm_boyd_residual_metrics` -- logs its arguments (deep-copied) and
     return value.
   - `update_and_check_convergence` -- logs call order and `update_flags`
     (cross-check only; the `_update_interface_power_flow_variables` wrapper
     already gives the same information more directly, since
     `update_and_check_convergence` calls it unconditionally with the same
     flags).
3. Runs `G.run_admm_arm('diag_mapping', OUT_DIR, k_override=None,
   eval_id='p515s32_diag_mapping', num_max_iters_override=1, apply_rho=False,
   full_diagnostics_in_rows=True, post_run_hook=hook)` -- same candidate
   (C\*, via `_construct_arm_planning`'s default uniform assignment), same
   case-file rho (`apply_rho=False`), same smoke cap (1 cycle) as
   `p515_s32_preflight.py`. `OUT_DIR = data/SRP1/Results/P515S32/diag_mapping/`;
   refuses if it exists (hit twice during debugging, both times cleaned by
   removing the partial directory before re-running -- see Commands below).
4. Reconstructs, from the captured snapshots, per channel (V, PF-p, PF-q):
   lambda^0, r^0 (pre-init) and Delta-lambda over the pre-init round; r^1
   (cycle-1) and Delta-lambda over the cycle-1 round; y at Boyd-call time;
   and the entrywise identities requested by the method (Section 3 of the
   task). ESS is checked separately for a single, simpler quantity: the dual
   norms immediately before the very first `update_and_check_convergence`
   call (expected exactly zero).
5. Writes `mapping_diag_summary.json` (all norms/diffs/identities, the
   official production Boyd result for cycle 1, the call-order log, the ESS
   zero-check) and `raw_snapshots.pkl` (full deep-copied call/Boyd logs, for
   reproducibility).

A first pass at the analysis crashed with `KeyError: 5` because
`create_admm_variables` shapes `dual_vars[channel][agent]` as
`{'current': {...}}` **only** (no `'prev'` key at all for vmag/pf duals) and
`consensus_vars[channel][agent]` as `{'current': {...}, 'prev': {...}}` --
the first draft's flatteners omitted the `['current']` indexing level. Fixed
by adding it explicitly everywhere consensus/dual values are read (documented
in the script with a comment at the fix site).

A second issue: the PF-channel entrywise identity check
(`Delta-lambda == rho * r`) initially failed with a residual of ~30 (not
noise) because the PF dual update in
`_update_interface_power_flow_variables` carries an extra
`/interface_rating * dso_s_base` factor the V channel does not have
(`shared_resources_planning.py:6320`:
`dual_vars['pf']['dso']['current'][...] += rho_pf_dso * error_p_pf_req_dso /
interface_rating * dso_s_base`). This is a diagnostic-script bug, not a
production one -- fixed by adding a `_dso_pf_lambda_scale_series` helper
(reading `interface_rating`/`baseMVA` directly off the network objects, no
production reimplementation) and applying it only to the PF channels' entrywise
checks.

## Commands / experiments run

1. `python -c "import ast; ast.parse(...)"` on the diagnostic script --
   syntax OK (run after each edit).
2. `python p515_s32_mapping_diag.py` -- attempt 1: failed at the capture-path
   precheck (`inspect.getsource` saw the wrapper). No solves had occurred
   yet (the guard is installed only inside `run_admm_arm`, which had not
   been reached).
3. Fixed patch-installation order; `python p515_s32_mapping_diag.py` --
   attempt 2: the ADMM run itself succeeded (102/102 permitted solves,
   `local_solve_failures=0`), but the post-run analysis crashed with
   `KeyError: 5` (see above). Removed the partial `diag_mapping/` output
   directory and the two `data/SRP1/Results/P56A/evals/p515s32_diag_mapping*`
   scratch eval directories `run_admm_arm`/the precheck had created, since the
   script's own freshness check (`os.path.exists(OUT_DIR)`) would otherwise
   refuse to restart.
4. Fixed the `['current']` indexing bug; `python p515_s32_mapping_diag.py` --
   attempt 3: completed end to end (wall clock ~55 s, matching the
   preflight's own 53.7 s), but the PF entrywise identity check showed a
   ~30-unit residual (not the V channel, which was exact).
5. Fixed the PF `interface_rating`/`dso_s_base` scale factor; cleaned the
   partial output directories again; `python p515_s32_mapping_diag.py` --
   attempt 4 (final): `EXIT_CODE=0`. Solve-profile guard checked exactly
   against 102 (`permitted_solve=102, permitted_exec=102, blocked_solve=0,
   blocked_exec=0`). All entrywise identities hold to machine precision.
6. Wrote `manifest_sha256.json` over the full `diag_mapping/` tree (16 files,
   largest 2.97 MB -- well under the 20 MB threshold, so every file is
   committed directly rather than hash-recorded-only).

Interpreter used throughout: `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`
(per `CLAUDE.local.md`). Solver: production resolves `NLP_SOLVER_PATH` from
`.env` (`/usr/local/bin/ipopt`), unchanged by this task.

## Results

### Reproduction of the observation

The diagnostic's own run reproduces the preflight's committed numbers
exactly (same candidate, same case-file rho, same cold start):

| channel | r (official) | s_rho_part (official) | norm_y (official) |
|---|---|---|---|
| V | 0.10116633210621236 | 0.03749586499053396 | 0.03749586533889183 |
| PF | 3.6407372288890447 | 5.039837405154417 | 5.058518768214646 |
| ESS | 0.0052877172482701 | 0.0018403688080641093 | 0.0052877172482701 |

These match `g_preflight.json`/the Planner's table to full precision (V:
0.101166 / 0.0374959 / 0.0374959; ESS: `norm_y == r` exactly). The PF row's
"rho-part of s" in the Planner's table (5.03984) is `s_rho_part`, not the
full `s` (7.1274) -- consistent with `proximal_share = 1/sqrt(2)` for both V
and PF (rho = gamma = 1 before the cycle-1 penalty update, confirmed:
`rho_v_before = rho_pf_before = rho_ess_before = 1.0`).

### Call log: exactly 4 calls to `_update_interface_power_flow_variables`, in order

| index | tag | update_tn | update_dns |
|---|---|---|---|
| 0 | `pre_admm_init` | True | True |
| 1 | `cycle1_dso` | False | True |
| 2 | `cycle1_tso` | True | False |
| 3 | `cycle1_esso` | False | False |

3 calls to `update_and_check_convergence` (dso / tso / esso), matching
source exactly -- no hidden extra call.

### Snapshot table (raw, un-normalized "model" units; 864 entries per channel
= 3 active DSO nodes x 3 years x 4 days x 24 periods)

| channel | ‖lambda^0‖ | ‖r^0‖ (pre-init) | ‖Delta-lambda_preinit‖ | ‖r^1‖ (cycle-1) | ‖Delta-lambda_cycle1‖ | ‖y‖ at Boyd time |
|---|---|---|---|---|---|---|
| V | 0.0 | 31.3191 | 31.3191 | 34.9024 | 34.9024 | 12.9361 |
| PF-p | 0.0 | 815.619 | 488.267 | 474.143 | 304.540 | 385.819 |
| PF-q | 0.0 | 733.464 | 474.533 | 255.761 | 199.512 | 327.154 |
| ESS (tso/dso/esso duals, pre-loop) | 0.0 / 0.0 / 0.0 | -- (single round only, no pre-loop update) | | | | |

`rho_v_dso` and `rho_pf_dso` are both `1.0` at both the pre-init call and the
cycle-1 TSO call (confirmed by direct read of the Pyomo Params at call time,
not inferred).

### Verified identities (entrywise, max absolute difference over 864 entries)

| identity | V | PF-p | PF-q |
|---|---|---|---|
| Delta-lambda_cycle1 == rho * r^1 (PF: x (dso_s_base/interface_rating)) | 0.0 | 3.55e-15 | 3.55e-15 |
| Delta-lambda_preinit == rho * r^0 (PF: x (dso_s_base/interface_rating)) | 0.0 | 3.55e-15 | 3.55e-15 |
| y(Boyd time) == lambda^0 + Delta-lambda_preinit + Delta-lambda_cycle1 | 0.0 | 0.0 | 0.0 |
| x at Boyd time == x at lambda-update time | 0.0 | 0.0 | 0.0 |
| z at Boyd time == z at lambda-update time | 0.0 | 0.0 | 0.0 |
| lambda at Boyd time == lambda at lambda-update time | 0.0 | 0.0 | 0.0 |
| DSO-only call (`update_tn=False`) leaves dual_vars['vmag'\|'pf']['dso'] unchanged | 0.0 | 0.0 | 0.0 |
| ESSO-only call (`update_tn=False, update_dns=False`) leaves x, z unchanged | 0.0 | 0.0 | 0.0 |

The `3.55e-15` PF entries are floating-point noise (machine epsilon scale
relative to values in the hundreds), not a real discrepancy -- confirmed by
first observing a genuine ~30-unit residual before adding the missing
`dso_s_base/interface_rating` scale factor to the diagnostic's own formula
(see "Changes made"); after the fix the identity holds to machine precision.

**Every identity the method asked for holds exactly** (to machine precision).
The Boyd dual-update recursion, as implemented, is internally self-consistent
at every point checked -- lambda^0 is genuinely 0.0 (read directly from
`create_admm_variables`'s freshly-constructed dicts, before any update),
and `y = lambda^0 + sum of every Delta-lambda applied so far` holds exactly.

## Verdict per hypothesis

- **Ha -- CONFIRMED (root cause).** `dual_vars['vmag']`/`dual_vars['pf']`
  are updated **twice** before cycle 1's Boyd metrics are first computed: once
  by the pre-ADMM-loop initialization call at `shared_resources_planning.py:2421`
  (`update_tn=True, update_dns=True`, using the two *independent* initial
  local solves from `create_distribution_networks_models`/
  `create_transmission_network_model`, i.e. before any TSO-DSO coordination),
  and once by the ADMM loop's own cycle-1 TSO-side call
  (`update_tn=True, update_dns=False`, `:2483-2494` region). Both calls reach
  the SAME "Update Lambdas" block inside `_update_interface_power_flow_variables`
  (`:6291-6322`), which is gated only on `update_tn` (`:6307`, `:6317`) with
  no distinction between "pre-loop" and "in-loop" callers. lambda^0 (true
  zero, before either update) is confirmed exactly zero. So `y` at the
  cycle-1 Boyd call reflects **two** rounds of the dual-update recursion
  (`lambda^0 + Delta-lambda_preinit + Delta-lambda_cycle1`), while the `r`/`s`
  reported for "cycle 1" reflect only the **second** round. This is exactly
  why `‖y‖ != rho*‖r^{cycle1}‖`: the comparison implicitly assumed
  `y^0 = 0` relative to cycle 1, but `y^0` (relative to cycle 1) is already
  `rho*r^0` from the pre-loop round. The design comment at `:2419-2420`
  ("Shared-ESS dual variables must remain zero before the first
  consensus-ADMM cycle") shows this asymmetry between V/PF and ESS was a
  **deliberate** choice for ESS specifically -- it does not say the same for
  V/PF, and the pre-loop call for V/PF is not merely copying consensus
  values, it structurally performs a real lambda update through the identical
  code path used every subsequent cycle. Whether warm-starting V/PF's
  multipliers this way was an intentional design decision or an unexamined
  side effect of reusing `update_interface_power_flow_variables` (rather than
  a "copy-only" variant) for the pre-loop initialization is not something
  this diagnostic can determine from the code alone; the Planner should
  decide whether it is desired behavior.

- **Hb -- REFUTED.** Every cross-check requested by the method (x/z/lambda
  at Boyd-call time vs. at lambda-update time; the DSO-only call leaving
  lambda unchanged; the ESSO-only call leaving x/z unchanged) returns exactly
  0.0. `update_and_check_convergence` is called exactly 3 times per cycle
  (dso/tso/esso), matching source; nothing overwrites `consensus_vars` or
  `dual_vars` between the cycle-1 lambda update and the Boyd call.

- **Hc -- NOT the mechanism as literally stated, but adjacent.** There is no
  evidence of a reference-node-voltage-fixed-to-TSO-request construction
  tying x_DSO to z_TSO in a way that creates the mismatch (`x0`, `x1` above
  come from the DSO's own independently-solved local model, not a copy of the
  TSO's value). The real structural fact is Ha: the SAME lambda-update code
  path is legitimately invoked from two different call sites (once outside
  the counted-cycle loop, once inside it) before the first counted cycle's
  diagnostics are read.

- **Hd -- REFUTED.** The reproduced official Boyd result (this run's own
  `get_admm_boyd_residual_metrics` return value, captured via the wrapper
  with zero modification to the call) matches the committed preflight's
  `g_preflight.json` to full precision. The recorded norms are read from
  the same production dictionaries at the same point the formula states.

- **He.** The precise mechanism (both file:line and the exact identity that
  explains the numeric coincidence `‖y_v‖ == ‖Delta-lambda_cycle1,v‖` -- not
  literally an identity, since `‖a+b‖` need not equal `‖b‖`, but the
  vector `Delta-lambda_preinit,v` and `Delta-lambda_cycle1,v` are large and
  substantially opposed per-entry here, so their sum's norm (12.94, the
  observed `‖y_v‖` in raw units) is much smaller than either summand's norm
  alone (31.32, 34.90) -- confirmed above, not asserted) is recorded under
  Ha.

## Statement on internal consistency (method item 5)

**r, s and y as implemented ARE consistent with the actual dual-update
iteration** -- every Boyd identity checked (`Delta-lambda == rho*r`
entrywise, `y = lambda^0 + sum(Delta-lambda)`) holds to machine precision at
both update rounds. There is no bug in `get_admm_boyd_residual_metrics`,
`_update_interface_power_flow_variables`'s "Update Lambdas" block, or in the
"current"/"prev" bookkeeping the s32 implementation relies on. The
inconsistency the Planner found is fully explained by **which round(s) of
the (correctly-implemented) dual-update recursion the cycle-1 diagnostics
reflect**: `y` (cumulative) reflects two rounds; `r`/`s` (reported per
"cycle") reflect only the second. This is a **reporting/interpretation
mismatch relative to a hidden "iteration 0"**, not a formula error.

## Proposed minimal correction (text only -- NOT applied)

Two independent, non-exclusive options, for the Planner to choose between (or
neither, if the warm start is intended and the diagnostic-comparison
convention is simply documented instead):

1. **Document it.** Add a note wherever cycle-1 Boyd diagnostics are
   interpreted (e.g. in the s32 spec / gate report template) stating that
   the pre-ADMM-loop initialization performs one full, un-logged lambda
   update for V/PF (using the two independent initial local solves) before
   cycle 1's own update, so cycle-1's `y` is `lambda^0 + Delta-lambda_0 +
   Delta-lambda_1`, not `rho*r_1` alone. No code change.

2. **Remove the extra round**, if the Planner wants cycle 1 to be a true
   `y^0 = 0` start: split
   `planning_problem.update_interface_power_flow_variables(...,
   update_tn=True, update_dns=True)` at `:2421` into (a) refreshing
   `consensus_vars['vmag'|'pf']['tso'|'dso']['current'/'prev']` from the two
   independent initial solves (needed so cycle 1's local solves warm-start
   from a sensible interface value) and (b) **not** updating
   `dual_vars['vmag'|'pf']` at that call -- e.g. a new `update_lambda=False`
   flag on `_update_interface_power_flow_variables`'s "Update Lambdas" block,
   analogous to how the ESS path already keeps its duals at zero until the
   first in-loop `update_sess=True` call. This is a production-code change
   and is explicitly **not** authorized or applied by this task.

## Validation

- Syntax-checked after every edit (`ast.parse`).
- The diagnostic's own `run_admm_arm` call reproduced `local_solve_failures=0`,
  `network_failures.n_blocks=0`, and the official Boyd cycle-1 result
  matching the committed preflight to full float precision -- the arm ran
  correctly, not merely "without crashing".
- Solve-profile guard checked **exactly** against 102 (not just
  `identity_holds`): `permitted_solve=102`, `permitted_exec=102`,
  `blocked_solve=0`, `blocked_exec=0`.
- `sha256` manifest (`manifest_sha256.json`, 16 entries) generated over the
  full `diag_mapping/` tree after the final run; all files under 20 MB
  (largest: `raw_snapshots.pkl`, 2.97 MB), so none needed hash-only
  (non-committed) treatment.
- Confirmed `preflight/` was never written to: `OUT_DIR` is hard-coded to
  `.../diag_mapping`; both `_require_fresh_output_root`-equivalent
  (`os.path.exists(OUT_DIR)` guard) and every `_refuse_overwrite` call target
  paths under it.
- Distinguishing "code executes" from "claim verified": the numeric identity
  checks (not just "no exception raised") are what support the verdict above
  -- in particular, the PF entrywise check initially FAILED with a genuine
  ~30-unit residual (a real bug in the diagnostic script's own formula, not
  noise), which was diagnosed and fixed before concluding the identities
  hold.

## Unexpected findings

- The PF-channel dual update carries an extra `dso_s_base/interface_rating`
  scale factor relative to V (`shared_resources_planning.py:6320`) that the
  Boyd function's own `r`/`s` normalization does NOT mirror in the same way
  (`get_admm_boyd_residual_metrics` normalizes `r_pf` by `interface_rating`
  alone and `y_pf` by `s_base_dso` alone -- already documented as F4 in
  `WORKER_REPORT_S32_IMPL.md`, not new). This is a pre-existing, already-reviewed
  normalization convention, not a new finding, but it is the reason the PF
  entrywise check needed the extra scale factor this diagnostic script
  initially omitted.
- `assert_s32_capture_paths` (`p515_g_g1_g4_admm_gates.py:2020-2023`) is
  order-sensitive to monkeypatching: it uses `inspect.getsource` on the
  live `srp.get_admm_boyd_residual_metrics` attribute. Any future diagnostic
  that both runs this precheck and patches that function must run the
  precheck first, or the precheck will raise on the wrapper's source
  rather than the production function's.

## Remaining issues

- Whether the pre-loop V/PF lambda warm-start (Ha) is *intended* design or
  an unexamined side effect of code reuse is a judgment call for the
  Planner; this diagnostic establishes the mechanism and its exact numeric
  effect but does not decide intent.
- Not evaluated here (out of scope): whether this same "extra round" pattern
  recurs at every ADMM restart within a longer run (e.g. warm-started cycles
  after a `from_warm_start=True` continuation) -- the cold-start
  `initial_state is None` branch is the only one this one-cycle diagnostic
  exercises, matching the preflight's own scope.

## Questions for Planner

1. Is the V/PF pre-loop dual warm-start (Ha) intended? If yes, option 1
   (document) is sufficient. If no, option 2 (the `update_lambda` flag) is
   the minimal production change, to be scoped as its own authorized task.
2. Should this same check (lambda^0, pre-loop-round accounting) be added to
   the s32 gate's own per-cycle diagnostics for cycles beyond 1, or is cycle
   1 the only cycle where this ambiguity matters (since every subsequent
   cycle's `y` is a legitimate, fully-in-loop Boyd running sum)?
