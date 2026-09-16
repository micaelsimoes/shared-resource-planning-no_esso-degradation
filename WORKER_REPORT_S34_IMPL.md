# Worker Report -- P5.15 Step 3.4 (+3.3(b) folded in) implementation (s34)

## Task received

Implement frozen spec v4
(`data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json`, sha256
`966940a789db3093a57d5fdc23be8ccb172b762bf02d0cd09eeaaf0f52f544e9`,
predecessor sha256 in the spec verified against the actual hash of v3
`data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json`): D5 ESSO
AL scaling, a fixed common objective scale (sigma), the v4 rho/freeze
policy, and S_ref = 2.5 MVA. Add the `s34` harness arm. Run zero-solve
checks and a two-cycle preflight **with a detector gate**. **Did not**
launch the 150-cycle gate.

## Files inspected

- `data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json` (v4) and
  `data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json` (v3,
  predecessor; hash cross-checked against the v4 spec's own
  `predecessor.sha256` field -- matches).
- `PLANNER_BRIEF_2026-09-13.md` Addendum 15 item 5; commit `9f182b79` (the
  v4 spec commit, and this task's own starting `HEAD` / "pre-change"
  baseline for every zero-solve bit-identity check).
- `P5_15_S33_E2_GATE_REPORT.md`, `P5_15_EFC_ARCHAEOLOGY_NOTE.md` (sections 6
  and 7), `WORKER_REPORT_S33_E2_IMPL.md`, `WORKER_REPORT_S34_EFC_BENCHMARK.md`.
- `shared_resources_planning.py`: `_compute_common_admm_objective_scale`
  (:3063), its call site and the ADMM main-loop initialization/state
  plumbing in `_run_operational_planning` (:2309-2460, :2696-2850,
  :2956-3020), `update_transmission_model_to_admm` (:3925-4090),
  `update_distribution_models_to_admm` (:4106-4225), the ESSO objective
  construction in `update_shared_energy_storage_model_to_admm`
  (:4334-4405), `_update_tso_proximal_centres_after_solve` (:4263-4420, the
  `4400` call site), `get_admm_residual_metrics` (:5255-5330, the
  `5292/5298/5302` call sites), `get_admm_boyd_residual_metrics`
  (:5570-5670, the `5643/5646/5648` call sites),
  `_update_shared_energy_storage_variables` (:6505-6630, the
  `6596/6603/6607` call sites), `_get_admm_penalty_summary`/
  `_get_admm_gamma_summary`/`_update_admm_penalties` (:6310-6600).
- `admm_parameters.py` in full (`__init__`, `_read_parameters_from_file`).
- `data/SRP1/SRP1_params.json` `admm` block (pre-edit state confirmed:
  `rho`=1.0 on every network/channel, `penalty_update.freeze_after_cycle`=30,
  no `objective_scale`/`shared_ess_reference_rating_mva`/`esso_al_scale`/
  `freeze_after_unchanged_cycles`/`freeze_backstop_cycle` keys).
- `p515_g_g1_g4_admm_gates.py`: the `s32` (:1935-2200) and `s33e2`
  (:2214-2577) arm sections in full, as the template for `s34`; `run_admm_arm`
  and `_construct_arm_planning` (:1056-1345, generic infrastructure, NOT
  modified).
- `p515_s33e2_zero_solve_checks.py`, `p515_s33e2_preflight.py`,
  `p515_s32_zero_solve_checks.py` (`V1`, helpers reused by import: `_build_admm_ready_state`,
  `_reset_rho`, `_dummy_residual_metrics`, `_stub_optimize`, `S31C_FIXTURES`),
  `p515_s32v2_zero_solve_checks.py` (`V2`, helpers reused by import:
  `_fake_boyd_metrics_v2`, `_load_module_at_commit`).
- `data/SRP1/Results/P515S33_E2_run/` (`leak_classification_baseline.jsonl`,
  `g_baseline.json`) -- used READ ONLY, as the s33e2 detector-gate baseline
  and (via `write_boyd_terminal_s34`'s system-cost comparison) a matched-cycle
  reference; never re-run or overwritten.
- `data/SRP1/Results/P515S32_run/g_baseline.json` -- same, read only.

## Files modified

**Production (commit `ef2040d7`):**
- `admm_parameters.py`
- `shared_resources_planning.py`
- `data/SRP1/SRP1_params.json`
- New: `p515_s34_zero_solve_checks.py`
- New (results): `data/SRP1/Results/P515S34/zero_solve_checks/zero_solve_checks.json`,
  `manifest_sha256.json`

**Harness (this commit):**
- `p515_g_g1_g4_admm_gates.py` (new `s34` section only; every other arm's
  own code -- `s32`, `s33e2`, `ablation_*`, `g1`/`g1b`/`g2r`/`s30` -- is
  untouched, confirmed by the diff hunk headers below).
- New: `p515_s34_preflight.py`
- New (results): `data/SRP1/Results/P515S34/preflight/*` (24 files, incl.
  `esso_models_preflight.pkl`, `esso_capture/`, `leak_classification_preflight.jsonl`,
  the two v4-only sidecars, `boyd_terminal.json`) + `manifest_sha256.json`.
- This report.

## Discrepancy vs the prompt (reported, not resolved unilaterally)

The prompt's "Verified code facts" says *"the **11** call sites the spec
lists: 4051, 4170, 4243, 4400, 5292, 5298, 5302, 5643, 5646, 5648, 6596,
6603, 6607"* -- that list has **13** entries, not 11, and the spec's own
`changes_from_v3.d_ess_reference_rating.call_sites` text lists the same 13
sites explicitly (4 + 3 + 3 + 3). I treated the spec's 13-entry list as
authoritative (per "where this prompt and the spec differ, the spec
wins") and applied `S_ref` at all 13; `grep -c 'reference_mva=_admm_shared_ess_reference_mva('`
on the final source confirms exactly 13. No other discrepancy between the
spec and the prompt was found.

## Changes made

### 1. `admm_parameters.py`

- `__init__`: `penalty_update['freeze_after_unchanged_cycles'] = None` and
  `['freeze_backstop_cycle'] = None` added alongside the existing
  `freeze_after_cycle` (kept, commented as superseded-for-SRP1-but-still-wired);
  `self.objective_scale = None`, `.objective_scale_source = 'default'`,
  `.objective_scale_assert_factor = 3.0`; `self.shared_ess_reference_rating_mva = None`,
  `.shared_ess_reference_rating_source = 'default'`; `self.esso_al_scale =
  {'mode': 'fixed', 'value': 1.0, 'source': 'default'}`.
- `_read_parameters_from_file`: the two new freeze-cycle keys parsed the
  same way as `freeze_after_cycle` (optional non-negative int, `None`
  default), validated `>= 1`; `objective_scale` (optional positive float,
  records `objective_scale_source`), `objective_scale_assert_factor`
  (optional, default 3.0, validated `>= 1`); `shared_ess_reference_rating_mva`
  (optional positive float, records its own source); `esso_al_scale` accepts
  either a positive float (`mode='fixed'`) or the literal string
  `'sigma_over_median_block_weight'` (`mode='sigma_over_median_block_weight'`,
  `value=None`), raising `ValueError` on any other string. Every new key
  absent reproduces the `__init__` defaults exactly -- confirmed for CS1,
  CS7, HR1, OP1, OP2 (zero-solve check (h)).

### 2. `data/SRP1/SRP1_params.json`

Exactly the authorized edits: `admm.objective_scale = 93635360.0`,
`admm.shared_ess_reference_rating_mva = 2.5`, `admm.esso_al_scale =
"sigma_over_median_block_weight"`; `admm.rho.v.* = 0.0077`,
`admm.rho.pf.* = 0.198`, `admm.rho.ess.* = 0.05` (all four networks, `ess`
including `esso`); `admm.penalty_update.freeze_after_unchanged_cycles = 10`,
`.freeze_backstop_cycle = 60`. **`penalty_update.freeze_after_cycle` (the
v3 fixed-cycle-30 freeze) was REMOVED from the case file** -- superseded
by the v4 per-channel rule, per the task's explicit permission to remove
it; production keeps the mechanism wired (for any case study that might
still set it) but SRP1's own params file no longer triggers it. `git diff
9f182b79 --stat`: `18 insertions(+), 14 deletions(-)` (net edits to
`penalty_update`, `rho`, plus the three new top-level keys); every other
key (`tol`, `num_max_iters`, `minimum_consecutive_converged_cycles`,
`shared_ess_normalization_floor_mva`, `proximal_regularization`,
`previous_iteration`, `rho_previous_iter`) byte-identical.

### 3. `shared_resources_planning.py` (diff hunks, file:line as of the
production commit `ef2040d7`; 24 hunks total, `git diff 9f182b79 -- shared_resources_planning.py`)

1. **(b) Fixed sigma.** New `_resolve_common_admm_objective_scale(objective_scale_computed,
   admm_parameters)` (~3154-3187, right after `_compute_common_admm_objective_scale`):
   returns `(objective_scale_used, sigma_computed, sigma_fixed)`;
   `sigma_fixed is None` -> passthrough (today's behaviour); otherwise
   asserts `1/factor <= sigma_computed/sigma_fixed <= factor`, raising
   `ValueError` with both numbers and the ratio on failure. New
   `_compute_median_admm_block_weight(planning_problem)` (~3190-3216) and
   `_resolve_esso_al_scale(planning_problem, admm_parameters,
   objective_scale_used)` (~3219-3251) -- the latter returns
   `(al_scale_esso, apply_flag)`, `apply_flag=False` only when
   `esso_al_scale['source'] == 'default'` (absent). Call site
   (`_run_operational_planning`, ~2412-2418): `objective_scale_computed =
   _compute_common_admm_objective_scale(...)` is **still called
   unconditionally** (wired, per the spec); `objective_scale, sigma_computed,
   sigma_fixed = _resolve_common_admm_objective_scale(...)`;
   `al_scale_esso = _resolve_esso_al_scale(...)[0]`; `objective_scale`
   (whichever source) is passed to `update_distribution_models_to_admm`/
   `update_transmission_model_to_admm` exactly as before (no signature
   change to those two functions -- `effective_scale = objective_scale /
   block_weight` is untouched, per the spec's "placement unchanged").
2. **(a) D5 ESSO AL scaling.** `update_shared_energy_storage_model_to_admm`
   (~4334-4405) gains `al_scale_esso=1.0` kwarg; `apply_al_scale =
   (params.esso_al_scale.get('source') == 'case_file')`. When `apply_al_scale`:
   `models[node_id].admm_esso_al_scale = pe.Param(initialize=al_scale_esso)`
   is constructed and **only** the two dual terms and two rho/2 terms are
   multiplied by it (`obj += models[node_id].admm_esso_al_scale * (...)`,
   four call sites); the base objective (`obj = copy(models[node_id].objective.expr)`)
   is untouched either way. When NOT `apply_al_scale` (absent, default):
   the `admm_esso_al_scale` Param is **not even constructed**, and the four
   AL terms are added exactly as pre-3.4 (`obj += models[node_id].dual_p_req[...] * ...`,
   unscaled) -- bit-identical, confirmed by zero-solve check (b).
3. **(c) S_ref.** New `_admm_shared_ess_reference_mva(params)` (~3894-3903,
   right before the two normalization helpers) -- the single source every
   call site reads. `_shared_ess_admm_normalization_mva`/`_pu` (~3907-3919)
   gain an optional `reference_mva=None` kwarg: `reference_mva is not None`
   short-circuits to `float(reference_mva)` (mva) / `float(reference_mva)/s_base`
   (pu), else unchanged (`max(abs(rating), floor)`). **All 13** call sites
   (4068, 4187, 4260[multi-line], 4417, 5309, 5315, 5319, 5660, 5663, 5665,
   6613, 6620, 6624) pass `reference_mva=_admm_shared_ess_reference_mva(<the
   ADMMParameters instance in scope at that site>)` -- `params` at 4068/4187/4260/4417/6613/6620/6624,
   `planning_problem.params.admm` at 5309/5315/5319 (`get_admm_residual_metrics`
   has no `admm_parameters` argument), `admm_parameters` at 5660/5663/5665
   (`get_admm_boyd_residual_metrics` does). Grep count of the exact kwarg
   string on the final source: **13**.
4. **(d) Freeze policy v4.** New `_init_admm_freeze_state()` and
   `_admm_rho_at_clamp(rho_value, update_params, rel_tol=1e-9)` (~6350-6377,
   before `_update_admm_penalties`). `_update_admm_penalties` gains
   `freeze_state=None` kwarg (fresh state built internally if `None`,
   preserving any pre-v4 caller's behaviour exactly for the `legacy_frozen_global`
   path only -- see "Freeze precedence" below). Per-channel precedence
   (highest first): `channel_frozen_this_cycle = group_state['frozen'] or
   legacy_frozen_global or backstop_frozen_global` (`legacy_frozen_global`
   = the unchanged v3 `iter > freeze_after_cycle` test; `backstop_frozen_global`
   = new, `iter >= freeze_backstop_cycle`) > `not adaptive_penalty` >
   failure hold > increase/decrease/dead-band. When NOT frozen this cycle:
   streak bookkeeping (`unchanged_streak` resets to 0 and `ever_acted=True`
   on `increased`/`decreased`, else `+= 1`); if `freeze_after_unchanged_cycles
   is not None and ever_acted and unchanged_streak >= freeze_after_unchanged_cycles`,
   `frozen=True`, `reason='streak'`, `at_clamp = _admm_rho_at_clamp(before[group],
   update_params)` -- **takes effect starting the NEXT cycle** (this
   cycle's already-decided `action` is unaffected; verified in zero-solve
   check (e)). The scaling-application guard (~6529) is now just
   `if params.adaptive_penalty and allow_update:` (the old `and not frozen`
   dropped) -- safe because a frozen channel's `factors[group]` is already
   forced to `1.0` by the per-group precedence above, so applying the
   (no-op) scale/gamma-reset to it is a literal identity, not a relied-upon
   side effect (documented in the updated code comment). Return signature:
   `actions, before, after, before_gamma, after_gamma, rho_freeze_active,
   freeze_state` (was a 6-tuple ending in a single bool `frozen`; now
   `rho_freeze_active = all(freeze_state[g]['frozen'] for g in (v,pf,ess))`,
   and the mutated `freeze_state` dict is the new 7th element). Call site
   (`_run_operational_planning`, ~2716-2720) unpacks the 7-tuple and passes
   `freeze_state=freeze_state` (persisted across cycles the same way as
   `consecutive_converged_cycles`, initialized near it (~2358-2369) and
   included in both the early-failure and final `state` dicts).
5. **(e) Diagnostics.** New `_get_admm_efc_per_day_max(esso_model)`
   (~6310-6329, before `_get_admm_penalty_summary`) -- reads
   `es_avg_ch_dch_per_unit`/`es_e_rated_per_unit` directly off the
   already-solved ESSO models, same formula as
   `p514_n_instrumented_cstar.py`'s `capture_esso` (`avg / (2*rated)`),
   max across nodes and cohort-years; `None` if no cell available. Called
   once per cycle (~2720) right before the `admm_diagnostics.append({...})`
   dict literal, which gained exactly the keys the spec lists (~2830-2848):
   `sigma_fixed`, `sigma_computed`, `al_scale_esso`,
   `shared_ess_reference_rating_mva`, `freeze_after_unchanged_cycles`,
   `freeze_backstop_cycle`, `rho_frozen_{v,pf,ess}`,
   `rho_unchanged_streak_{v,pf,ess}`, `rho_at_clamp_{v,pf,ess}`,
   `efc_per_day_max`. Every pre-existing key (`rho_freeze_active`,
   `freeze_after_cycle`, `gamma_policy`, `gamma_tau`, all `boyd_*`,
   `gap_proxy_*`, ...) is untouched and still present.

**Not touched:** the Boyd r/s/eps formulas, the dead band, factors, clamp
values, the multiplier (lambda/z) updates, the proximal centring, the
settlement, the objective TERMS themselves (only the al_scale/sigma
MULTIPLIERS around them), `_run_operational_planning_hierarchical`,
`_run_operational_planning_without_coordination` (confirmed byte-identical
to commit `9f182b79` -- zero-solve check (j)).

### 4. `p515_g_g1_g4_admm_gates.py`: new arm `s34`

Same machinery as `s33e2` (module-level spec path/hash constants, rule-eleven
checklist function, rho/gamma/freeze trajectory helper, the
matched-cycle system-cost helper extended to s31c+s32+s33e2, the
`boyd_terminal.json` writer, the CLI dispatch branch). New:
`assert_s34_capture_paths` -- reuses `assert_s31c_capture_paths` directly
(**deliberately not** `assert_s32_capture_paths`/`assert_s33e2_capture_paths`,
both of which hard-assert their OWN spec's rho==1.0/`freeze_after_cycle==30`
values and would raise under v4's config; the still-required structural
checks those two functions perform -- Boyd field/`admm_diagnostics` key
presence, the mutable-gamma structural check, the interface-voltage writer
-- are reproduced directly in `assert_s34_capture_paths` instead), plus
every v4-only check (spec hash, `objective_scale`/`objective_scale_source`/
`objective_scale_assert_factor`, `esso_al_scale` presence and mode, S_ref,
initial rho v/pf/ess, `freeze_after_unchanged_cycles`/`freeze_backstop_cycle`,
`minimum_consecutive_converged_cycles`, and a capture-path callable check
for every new production/harness function this stage adds). New
`s34_capture_hooks` (a context manager, `capture_additions`): monkeypatches
`srp.get_admm_boyd_residual_metrics` (called exactly once per cycle, with
full `tso_model`/`dso_models`/`esso_model`/`consensus_vars`) to
**unconditionally** (no tolerance gate, unlike production's own
`[RECOURSE JUMP]` print) write two per-cycle JSONL sidecars: the SAME
block decomposition production's gated print computes (via the SAME
unmodified `srp._get_operational_recourse_block_components`/
`_get_operational_objective_component_blocks`), and per-entry shared-ESS
z/x on a recorded stride (1 = every cycle, ~1728 entries/cycle, "size
acceptable" per the spec's own escape clause) plus EFC/day per node. New
`write_boyd_terminal_s34` (calls `write_interface_settlement_detail_s31c`
first, then `write_interface_voltage_terminal`, then assembles
`boyd_terminal.json` with sigma_fixed/sigma_computed, al_scale_esso, S_ref,
initial rho, per-channel freeze cycle, `rho_at_clamp`, and the system-cost
comparison against s31c, s32 AND s33e2 at the spec's matched cycles). Output
root `data/SRP1/Results/P515S34_run` (refuses if it exists); label
`baseline`. No other arm's code was touched -- confirmed: `git diff 9f182b79
-- p515_g_g1_g4_admm_gates.py` is exactly TWO hunks: one new, self-contained
section inserted after `write_boyd_terminal_s33e2` (the `s32`/`s33e2`
sections above it are byte-for-byte unchanged), and one new `elif gate ==
's34':` branch at the CLI dispatch. The objective-component-blocks delta
bugfix (see "Commands / experiments run" below) was applied while the
new section was still uncommitted, so it is absorbed into the first hunk,
not a separate one.

### 5. `p515_s34_zero_solve_checks.py` (new)

Ten checks (a)-(j), `SolveProfileGuard(permitted=())` for the whole run.
Output: `data/SRP1/Results/P515S34/zero_solve_checks/` (new; refused if it
existed).

### 6. `p515_s34_preflight.py` (new)

Runs `run_admm_arm` (by import) for exactly 2 cycles through the `s34`
machinery, wrapped in `s34_capture_hooks`, into
`data/SRP1/Results/P515S34/preflight/` (fresh root). Checks per-cycle
field-completeness (every v3 field plus every v4 field), that cycle-1 rho
actions follow the adaptive/unfrozen v4 rule, and runs the **decisive D5
detector gate** against the committed s33e2 baseline
(`data/SRP1/Results/P515S33_E2_run/leak_classification_baseline.jsonl`,
read only).

## Commands / experiments run

1. `python -c "import ast; ast.parse(...)"` on every modified/new file --
   syntax OK after each edit.
2. `ADMMParameters().read_parameters_from_file(...)` against
   `data/SRP1/SRP1_params.json` and CS1/CS7/HR1/OP1/OP2's own params files
   -- confirmed the SRP1 edits load as intended and every OTHER case study
   loads with the unchanged defaults (also re-verified inside zero-solve
   check (h)).
3. `python p515_s34_zero_solve_checks.py` -- three iterations to green:
   (i) the ESSO models needed every `*_req`/`dual_*` entry zeroed before
   `pe.value(model.admm_objective)` (new `_zero_out_esso_requests` helper,
   mirroring the existing `_zero_out_admm_requests` for the TSO model);
   (ii) check (b)/(g)'s "absent" builds needed `esso_al_scale`/
   `shared_ess_reference_rating_mva` EXPLICITLY reset to the `__init__`
   defaults before building (SRP1's OWN case file now configures both, so
   `O.fresh_planning()` alone no longer produces the "absent" state); the
   DSO model's `p_ess_req` etc. are indexed by period ALONE (one shared
   ESS per DSO), not `(e, p)` like the TSO's -- fixed in check (g)'s
   zeroing loop; (iii) check (d)'s source-text scan for "every call site
   has `reference_mva=`" needed a regex-over-source approach (the naive
   per-line `in` check flagged the two legitimate multi-line call sites and
   the helper's own internal fallback call as false positives) and check
   (e)'s streak-trigger assertion had the OFF-BY-ONE-CYCLE expectation
   backwards (the trigger is detected and flagged at the END of the cycle
   whose action is still ordinary 'held'/'increased'/'decreased', not
   retroactively on that cycle's OWN action -- fixed the test assertion,
   not the production code, after confirming the production behaviour
   matches its own docstring). Final run: **10/10 checks pass**,
   `SolveProfileGuard.counts` all zero, `verify_failures=[]`.
4. `python p515_s34_preflight.py` (real IPOPT solves, 2 cycles, ~87 s wall)
   -- two iterations: (i) the `s34_capture_hooks` recourse-jump sidecar
   crashed on cycle 2 (`TypeError: unsupported operand type(s) for -:
   'dict' and 'dict'`) -- `_get_operational_objective_component_blocks`
   returns `{block_key: {component_name: value}}` (a dict of dicts), not
   `{block_key: value}` like `_get_operational_recourse_block_components`;
   fixed by flattening to `(block_key, component_name)` before
   differencing (harness-only fix, `p515_g_g1_g4_admm_gates.py`); (ii) the
   preflight's own `_field_completeness` incorrectly flagged
   `objective_change_abs`/`objective_tolerance`/`objective_change_ratio` as
   missing at cycle 1 (they are legitimately `None` there -- no previous
   recourse yet; s33e2's own precedent function records these WITHOUT a
   presence check, which my first draft had diverged from) -- fixed. Final
   run: **exit 0**, 153/153 solves, 0 local-solve failures, all fields
   complete both cycles, cycle-1 rule holds, **detector gate PASSES with
   margin** (see below).
5. Scratch eval directories under `data/SRP1/Results/P56A/evals/`
   (`p515s34_check_*`, `p515s34_preflight_*`) created by (2)/(3)/(4)'s
   internal `O.fresh_planning` calls were removed after each run;
   `data/SRP1/Results/P515S34_run/` (the real gate's output root) was
   **never created** -- confirmed by `ls`.

Interpreter: `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`.
Solver: production resolves `NLP_SOLVER_PATH` from `.env`
(`/usr/local/bin/ipopt`), unchanged by this task. **The exact gate command
(`python -u p515_g_g1_g4_admm_gates.py s34 > data/SRP1/Results/P515S34_launch.log 2>&1`)
was NOT executed**; `data/SRP1/Results/P515S34_run/` does not exist.

## D5 equivalence algebra (zero-solve check (a))

Production builds (`update_shared_energy_storage_model_to_admm`,
`apply_al_scale` branch):

```
new_obj = old_base + al_scale * AL
```

The LITERAL reading of "ESSO objective on the same sigma convention as the
networks" (dividing the base, as the TSO/DSO objectives ARE divided by
`effective_scale`) would instead be:

```
literal_obj = old_base / al_scale + AL
```

These are numerically DIFFERENT functions, but they differ by exactly the
POSITIVE constant factor `al_scale` (a fixed Pyomo `Param`, independent of
every decision variable):

```
al_scale * literal_obj = al_scale * (old_base/al_scale + AL)
                        = old_base + al_scale*AL
                        = new_obj
```

Multiplying an objective by a positive constant does not change its
argmin, so `new_obj` and `literal_obj` share the same minimiser even
though `new_obj != literal_obj` pointwise -- the "deviation in FORM, not
substance" the spec records. **Verified numerically** (not just
symbolically) on a built ESSO model at a hand-set nonzero request/dual
state, `al_scale=5.0`: `old_base_value=0.0` (the ESSO base objective's
build-default value at this zero-solve state, independent of `al_scale`
since the base is never divided -- see the check's own recorded value),
`al_only_value=-0.207451` computed by hand from the SAME unscaled
constraint expressions `update_shared_energy_storage_model_to_admm`'s
`else` branch would build, `production_new_obj_value=-1.037255 ==
old_base_value + al_scale*al_only_value` to machine precision, and
`al_scale * literal_reading_value == production_new_obj_value`
(`5.0 * -0.207451 = -1.037255`) to `1e-9` relative tolerance. Both the
`new_obj == base + al_scale*AL` identity and the `al_scale*literal ==
new_obj` identity hold exactly; `new_obj != literal_obj` (`-1.037255 !=
-0.207451`, as expected,
since `al_scale=5 != 1`).

## Zero-solve checks -- results (`p515_s34_zero_solve_checks.py`)

All 10 checks pass; `solve_profile_guard.counts = {permitted_solve: 0,
permitted_exec: 0, blocked_solve: 0, blocked_exec: 0}`, `verify_failures = []`.

| check | result | key evidence |
|---|---|---|
| (a) D5 equivalence | **pass** | see algebra above; `identity_holds_al_scale_times_literal_equals_new=True`, `new_obj_equals_base_plus_scaled_al=True`, `new_obj != literal_obj` (5x apart) |
| (b) esso_al_scale absent -> bit-identical | **pass** | with BOTH `esso_al_scale` and `shared_ess_reference_rating_mva` reset to their absent defaults, `objective_value_current_absent = objective_value_pre_change = -2.7923750000000003` (commit `9f182b79`) EXACTLY; `admm_esso_al_scale_param_constructed_when_absent=False` (the Param literally is not built) |
| (c) sigma assertion | **pass** | within factor 3 (`computed=2.34e8`, ratio 2.5): returns the fixed value unchanged; absent: computed value passed through; outside factor 3 (ratio 10): raises `ValueError` whose message contains both `9.363536e+07` and `9.363536e+08` |
| (d) S_ref at all 13 sites | **pass** | `n_call_sites_with_reference_mva_kwarg=13` (matches the spec's 13-entry list, not the prompt's "11"); unit tests: `mva_with_ref=2.5` (independent of rating/floor), `mva_without_ref=0.96875` (unchanged behaviour); end-to-end ESSO build: `implied_s_ref_from_built_objective=2.5` (recovered from the built constraint's own coefficient via a two-point finite difference) |
| (e) freeze rule v4 | **pass** | streak: cycle 1 `increased` (ever_acted=True, streak=0) -> cycles 2/3/4 `held` (streak 1/2/3) -> streak trigger fires at the END of cycle 4 (`frozen=True`, cycle 4's OWN action stays `'held'`) -> cycle 5 shows `'held (frozen after 3 unchanged cycles)'`; "never acted" channel: 7 cycles of dead-band never freezes via the streak (`ever_acted=False` throughout); backstop: freezes ALL THREE channels at cycle 5 regardless, label `'held (frozen backstop cycle 5)'`; clamp: rho forced to `penalty_update['max']=1e4`, streak-freeze triggers there, `at_clamp=True`, **no exception raised** |
| (f) update isolation | **pass** | `consensus_vars`/`dual_vars` bit-identical before/after; `rho_v` 1.0->1.5 (`increased`, factor = `penalty_update['increase_factor']`) |
| (g) TSO/DSO bit-identical when absent | **pass** | with `objective_scale` and `shared_ess_reference_rating_mva` both reset to `None`, the built TSO admm_objective and DSO admm_objective (hand-zeroed request/dual state) match commit `9f182b79` to `1e-12` relative tolerance |
| (h) other case params load | **pass** | CS1/CS7/HR1/OP1/OP2 all load `objective_scale=None` (source `default`), `shared_ess_reference_rating_mva=None`, `esso_al_scale={'mode':'fixed','value':1.0,'source':'default'}`, both freeze-v4 keys `None`; SRP1 loads `objective_scale=93635360.0` (source `case_file`), `shared_ess_reference_rating_mva=2.5`, `esso_al_scale={'mode':'sigma_over_median_block_weight',...,'source':'case_file'}`, `freeze_after_unchanged_cycles=10`, `freeze_backstop_cycle=60`, `freeze_after_cycle=None` (removed), rho v/pf/ess = 0.0077/0.198/0.05 |
| (i) fixtures | **pass** | 3/3 S31C fixtures + 2/2 s32 FrozenSMOPF pickles unpickle |
| (j) untouched paths | **pass** | `_run_operational_planning_hierarchical`/`_run_operational_planning_without_coordination` source-text IDENTICAL to commit `9f182b79`; `diff_hunk_count=24`, every hunk inside the permitted sites (enumerated above) |

## DETECTOR-GATE COMPARISON TABLE (decisive -- read this first)

From `data/SRP1/Results/P515S34/preflight/leak_classification_preflight.jsonl`
(THIS run's own ESSO solves) vs the committed
`data/SRP1/Results/P515S33_E2_run/leak_classification_baseline.jsonl`
(s33e2, read only), at the SAME (cycle, node) keys. Ratio = preflight /
s33e2-baseline; **> 1 with the "material worsening" flag** would mean the
quantity got worse (mu_final, complementarity ratio and spurious
throughput should all stay SMALL; an increase is a regression). Flag
threshold: **10x** (explicit Worker convention, not specified numerically
by the spec -- every ratio is reported regardless, so the Planner can
apply a different bound).

| cycle | node | mu_final ratio | s_obj ratio | complementarity_ratio_max ratio | spurious_throughput ratio | flagged |
|---|---|---|---|---|---|---|
| 001 | 5 | 0.896 | 1.000 | 1.000 | 0.968 | no |
| 001 | 7 | 1.010 | 1.000 | 0.996 | 0.972 | no |
| 001 | 9 | 1.004 | 1.000 | 0.995 | 0.979 | no |
| 002 | 5 | 1.189 | 1.000 | 0.630 | 0.824 | no |
| 002 | 7 | 1.249 | 1.000 | 0.646 | 0.825 | no |
| 002 | 9 | 1.146 | 1.000 | 0.613 | 0.831 | no |

**`any_material_worsening = False`. Every ratio is within ~25% of 1.0 or
better; complementarity ratio and spurious throughput are actually LOWER
(better) at cycle 2 than the s33e2 baseline at the same cycle.** `s_obj`
(the ESSO's own barrier-scale schedule constant, `1.000e-01` both sides,
production-set, independent of D5) is identical by construction, as
expected -- included for completeness per the spec's exact metric list.
**Gate PASSES**; the harness is reported as ready to commit; no revisit of
epsilon or the ESSO tolerance is triggered.

## Preflight results (`p515_s34_preflight.py`, two real ADMM cycles, C\* candidate)

`cycles_run=2`, `local_solve_failures=0`. Solve counts checked **exactly**
against 153 (`51*2+51`): `{'permitted_solve': 153, 'permitted_exec': 153,
'blocked_solve': 0, 'blocked_exec': 0}` (`identity_holds=True`).
`network_failures.n_blocks=0`. Every required field (every v3 Boyd/rho/gamma
field AND the new v4 fields: `sigma_fixed`, `sigma_computed`,
`al_scale_esso`, `shared_ess_reference_rating_mva`,
`freeze_after_unchanged_cycles`, `freeze_backstop_cycle`,
`rho_frozen_{v,pf,ess}`, `rho_unchanged_streak_{v,pf,ess}`,
`rho_at_clamp_{v,pf,ess}`, `efc_per_day_max`) populated on both cycles
(`field_completeness_all_ok=True`).

**Run-level constants (both cycles, as expected -- constant for the whole
run):** `sigma_fixed=9.363536e+07`, `sigma_computed=9.363536e+07`
(`ratio_to_fixed ~ 1.0000000031`, trivially within the assert factor 3 --
i.e. this C\* instance's computed sigma essentially COINCIDES with the
value the spec froze from it), `al_scale_esso=2.272110e+05` (= sigma /
median block weight, computed once, logged via `[ADMM ESSO AL SCALE]`),
`shared_ess_reference_rating_mva=2.5`.

**Cycle-1 Boyd values per channel** (r, s; action; rho before/after; gamma
before/after; frozen/streak/at_clamp):

| channel | r | s | action | rho before -> after | gamma before -> after | frozen | streak | at_clamp |
|---|---|---|---|---|---|---|---|---|
| V | 2.476e-01 | 2.001e-03 | increased | 0.0077 -> 0.01155 | 0.0077 -> 0.01155 | False | 0 | False |
| PF | 3.487e+00 | 1.445e+00 | held | 0.198 -> 0.198 | 0.198 -> 0.198 | False | 1 | False |
| ESS | 1.977e-01 | 1.398e-02 | increased | 0.05 -> 0.075 | 0.05 -> 0.075 | False | 0 | False |

**Cycle-2:**

| channel | r | s | action | rho before -> after | gamma before -> after | frozen | streak | at_clamp |
|---|---|---|---|---|---|---|---|---|
| V | 9.864e-02 | 3.209e-03 | held | 0.01155 -> 0.01155 | 0.01155 -> 0.01155 | False | 1 | False |
| PF | 2.816e+00 | 7.136e-01 | held | 0.198 -> 0.198 | 0.198 -> 0.198 | False | 2 | False |
| ESS | 1.084e-01 | 1.288e-02 | increased | 0.075 -> 0.1125 | 0.075 -> 0.1125 | False | 0 | False |

**Cycle-1 rho rule check:** every channel's action is a legal adaptive
value (`increased`/`held`), every channel unfrozen (`freeze_after_unchanged_cycles=10
>> 1`, `freeze_backstop_cycle=60 >> 1`) -- `cycle1_rho_rule_ok=True`.
Gamma tracks rho exactly on every channel/cycle (tied policy, unchanged
from v3).

**EFC/day:** `efc_per_day_max` cycle 1 = `2.875e-04`, cycle 2 = `6.110e-02`
(max across nodes 5/7/9); per-node at cycle 2: node5=0.06104,
node7=0.06099, node9=0.06110 -- roughly uniform across nodes, as expected
(uniform C\* assignment). **Not comparable to s33e2's cycle-1/2** (S_ref,
rho and the ESSO AL scale all changed; the spec's own prediction states
this explicitly) -- reported as a magnitude, not a match test.

**ESS per-entry step (cycle 1 -> cycle 2, from `ess_entry_stride_preflight.jsonl`,
stride 1 = every cycle):** 1,728 (node x year x day x power_type x period)
entries compared; `mean_abs_z_step = 6.582e-03`, `max_abs_z_step =
4.596e-02` (node 5, year 2035, day Spring, period 15, `p`, z: 0.0377 ->
0.0837). For comparison, s33e2's steady-state (post-cycle-31) per-entry
ESS step was `4.26e-5` (`P5_15_S33_E2_GATE_REPORT.md` section 3) --
**this cycle-1-to-2 step is ~150x s33e2's steady-state step**, though this
is an early-transient comparison (rho starts at 0.05 here vs 1.0 in
s33e2, and this is cycle 1->2, not a settled trajectory) and should be
read as a magnitude for the Planner, not a like-for-like rate comparison.

## Freeze-rule behaviour (as implemented and verified)

Per-channel precedence, highest first: (1) `channel_frozen_this_cycle`
(sticky once True -- triggered by the legacy v3 fixed-cycle freeze, the
v4 global backstop, or a v4 per-channel streak trigger from a PRIOR
cycle); (2) `not adaptive_penalty` -> `'fixed'`; (3) `not allow_update`
(failed local solve) -> `'held after solver failure'`, own label, ONLY
when not frozen; (4) ordinary increase/decrease/dead-band. When newly
triggered THIS cycle (legacy or backstop only -- a streak trigger is
always detected one cycle late, see below), `at_clamp` is recorded against
`before[group]` (the rho value BEFORE this cycle's update, i.e. the value
the channel is being frozen AT). The per-channel streak (`unchanged_streak`,
`ever_acted`) is updated ONLY when the channel is not already frozen this
cycle; the freeze decision from the streak is evaluated AFTER that
cycle's ordinary action, so it is visible in `freeze_state` from that
cycle's RETURN onward but does not retroactively change that cycle's own
`action` -- verified exactly in zero-solve check (e) and observed live in
the preflight (no channel reached the 10-cycle streak or the 60-cycle
backstop in 2 cycles, so `rho_frozen_*` is `False` throughout the
preflight, as expected).

## Deviations from the spec

None in substance. One reported FORM deviation, already flagged by the
spec itself as author-authorized: D5 is an AL-terms scaling, not a
base-objective division (`changes_from_v3.a_D5_esso_scaling.flagged_for_author`).
The prompt's "11 call sites" vs the spec's actual 13-entry list is a
prompt-vs-spec discrepancy, not a spec deviation (see above; resolved in
the spec's favour, all 13 sites updated).

## Validation

- Code executes correctly: confirmed (zero-solve checks + a real two-cycle
  preflight, both exited 0 and produced the expected JSON/JSONL outputs).
- Requested diagnostic works: confirmed -- every new capture field
  (sigma/S_ref/al_scale_esso/freeze diagnostics/EFC) is populated on real
  cycles; `boyd_terminal.json` carries sigma_fixed/sigma_computed,
  al_scale_esso, S_ref, initial rho, per-channel freeze cycle, `rho_at_clamp`,
  and the s31c/s32/s33e2 system-cost comparison; `assert_s34_capture_paths`
  passes on a fresh, never-solved planning object.
- The D5 detector-gate CLAIM (no material worsening vs s33e2) is
  confirmed by the table above, computed from this run's own real ESSO
  solves against the committed s33e2 baseline -- both read from disk, no
  fabricated or approximated values.
- Underlying numerical question (does the D5/sigma/S_ref/v4-freeze
  re-scaling release the storage arbitrage gradient s33e2 measured, and
  does the run converge under Boyd within 150 cycles) is **NOT** assessed
  here -- that requires the full 150-cycle gate, which this task explicitly
  does not launch.
- `git diff 9f182b79` hunk inspection (24 hunks in `shared_resources_planning.py`,
  5 in `admm_parameters.py`, all confined to the permitted sites) plus the
  zero-solve checks' independent source-text/byte-identity/numeric-value
  assertions confirm no unintended production modification.
- The exact gate command
  (`python -u p515_g_g1_g4_admm_gates.py s34 > data/SRP1/Results/P515S34_launch.log 2>&1`)
  was **not** executed; `data/SRP1/Results/P515S34_run/` does not exist.

## Unexpected findings

- The pre-existing zero-solve check scripts for s32/s32v2/s33e2 (`p515_s32_zero_solve_checks.py`,
  `p515_s32v2_zero_solve_checks.py`, `p515_s33e2_zero_solve_checks.py`)
  call `srp._update_admm_penalties(...)` and unpack a 6-tuple; the return
  arity is now 7 (the new `freeze_state`). These scripts would raise
  `ValueError: too many values to unpack` if literally re-run against the
  current code -- same category as the precedent already recorded in
  `WORKER_REPORT_S32V2_IMPL.md` and `WORKER_REPORT_S33_E2_IMPL.md`'s
  "Unexpected findings" (each prior stage broke the arity of the one
  before it the same way). Not treated as a defect; their committed JSON
  outputs remain valid, frozen evidence for the code as it existed when
  they ran; neither was re-run onto its own committed path by this task.
  My own `p515_s34_zero_solve_checks.py` imports these modules ONLY for
  their unpacking-independent helper functions (`_build_admm_ready_state`,
  `_reset_rho`, `_dummy_residual_metrics`, `_fake_boyd_metrics_v2`,
  `_load_module_at_commit`, `S31C_FIXTURES`, `_stub_optimize`), never
  their own `check_*` functions, so this does not affect the new checks.
- `_get_operational_objective_component_blocks` returns a dict-of-dicts
  (`{block_key: {component_name: value}}`), unlike its sibling
  `_get_operational_recourse_block_components` (`{block_key: scalar}`) --
  not documented in either function's one-line body, only discoverable by
  reading `_print_recourse_jump_diagnostics`'s own (separate) handling of
  it. Caught by the preflight's real second cycle, not by the zero-solve
  checks (which never exercise `s34_capture_hooks`) -- a gap in my own
  test coverage, noted for the Planner: the zero-solve checks did not
  cover the harness-level capture hooks at all (out of scope for
  "zero-solve checks on shared_resources_planning.py", but worth flagging
  since it was the ONLY bug the preflight caught that the zero-solve
  checks did not).
- SRP1's real case file now configuring `esso_al_scale` and
  `shared_ess_reference_rating_mva` by default meant every "absent" zero-solve
  check needed an EXPLICIT override back to the `ADMMParameters.__init__`
  defaults (`O.fresh_planning()` alone no longer produces the pre-3.4
  state for SRP1) -- a natural consequence of this being the FIRST task to
  actually configure these keys in a real case file, not a defect.

## Remaining issues

- The `s34` 150-cycle gate itself has not been run; only the Planner
  launches it. The exact command
  (`python -u p515_g_g1_g4_admm_gates.py s34 > data/SRP1/Results/P515S34_launch.log 2>&1`)
  is wired, and its rule-eleven checklist (spec v4 hash, `objective_scale`/
  `objective_scale_assert_factor`, `esso_al_scale`, S_ref, initial rho,
  freeze keys, `minimum_consecutive_converged_cycles`, and a capture path
  for every field the spec's `report_per_cycle_channel`/`report_terminal`/
  `capture_additions` require) is verified to pass, without running the
  campaign.
- The detector gate's "material worsening" threshold (10x) is an explicit
  Worker convention, not a number the spec or Planner specified; every
  underlying ratio is reported in the table above so a different threshold
  can be applied without re-running anything.
- `check_d_s_ref_all_sites`'s source-text scan (13/13) and
  `check_g_tso_dso_bit_identical`'s numeric-equality check are independent,
  but neither is a formal proof that no OTHER site was silently added; the
  end-to-end ESSO build (finite-difference recovery of `S_ref` from the
  built objective's own coefficient) is the stronger, independent check
  for the ESSO path specifically.
- The ESS per-entry step comparison against s33e2 (150x) is a magnitude
  only, not a rate comparison under matched conditions (see "Preflight
  results" above) -- the Planner's own `known_concerns`/`predictions` in
  the spec already anticipate this and ask for the full run's trajectory,
  not the 2-cycle preflight, to test the storage-response falsifier.

## Questions for Planner

- None blocking. If the detector-gate 10x threshold should instead be a
  specific number (e.g. matching some other stage's convention), that is a
  one-line change to `MATERIAL_WORSENING_RATIO` in `p515_s34_preflight.py`
  with no re-run needed to re-evaluate it against the SAME preflight
  evidence already captured.

## Provenance

- Production commit: `ef2040d7` -- `admm_parameters.py`,
  `shared_resources_planning.py`, `data/SRP1/SRP1_params.json`,
  `p515_s34_zero_solve_checks.py`,
  `data/SRP1/Results/P515S34/zero_solve_checks/{zero_solve_checks.json,manifest_sha256.json}`.
- Harness commit: this commit -- `p515_g_g1_g4_admm_gates.py`,
  `p515_s34_preflight.py`, `data/SRP1/Results/P515S34/preflight/*` (24
  files) + `manifest_sha256.json`, this report.
- Spec: `data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json`
  (v4), `data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json`
  (v3, predecessor).
- Pre-change baseline for every bit-identity check: commit `9f182b79`
  (this Worker's own starting `HEAD`).
- Interpreter: `/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python`.
- Solver: `/usr/local/bin/ipopt` (via `.env` `NLP_SOLVER_PATH`), unchanged.
