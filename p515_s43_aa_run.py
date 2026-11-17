"""
P5.15 Step 3.7 integration -- Task 2: the flag-ON Anderson acceleration (AA)
run harness.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 22 ("Step 3.7 -- Anderson
acceleration") and Addendum 23 amendments (i)-(vi); frozen spec v12
`data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json`,
`item4_step_3_7_anderson`; frozen spec v13
`data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json`,
`items_2_3` (AA code choices accepted).

================================================================================
WHAT THIS SCRIPT DOES
================================================================================
The case file (`data/SRP1/SRP1_params.json`) already encodes the D-oracle
configuration in full (Step 3 closure, Addendum 22/24) -- confirmed directly
(rho v/pf/ess, tau=0/gamma_policy=tied_to_rho, `balancing_exempt_until.ess`,
freeze policy, `minimum_consecutive_converged_cycles=10`,
`shared_ess_initialization=standalone`, `num_max_iters=300` all match D's own
overrides bit-for-bit). This harness therefore runs `p515_g_g1_g4_admm_gates.
run_admm_arm(label='s39_D', apply_rho=False, pre_solve_hook=<AA-on>)` --
EXACTLY the "case-file-alone" invocation `p515_s41_hull_polish.py`/
`p515_s42_exact_fix_rerun.py` already use and already validated bitwise
against D with the flag off -- with the ONLY configuration action being
`pre_solve_hook` setting `admm_parameters.anderson_acceleration['enabled'] =
True` (memory=5, regularization=1e-10 are already the `ADMMParameters`
defaults -- verified, not assumed, in the same hook). `apply_rho=False`,
`num_max_iters_override=300` (the case file's own cap, D's), cap 300, the
case file's own 10-consecutive-cycle certification bar, `full_diagnostics_in_
rows=True` (so every `aa_*` field production already adds to each cycle's
`admm_diagnostics` entry lands in `cycle_trajectory` -- asserted, not
assumed, by `_zero_solve_precheck` below before any solve).

Per this task's instruction, there is NO reproduction-vs-D check here (an
AA-on run is expected to differ from the plain-ADMM oracle D) -- the full
trajectory is recorded instead (standard `run_admm_arm` captures +
`aa_per_cycle.jsonl`, derived post-hoc from the SAME `aa_*` fields already
in `cycle_trajectory`, never recomputed).

After the run, IN-PROCESS (same live `models`/`planning`/`state`, zero extra
ADMM cycles), IF AND ONLY IF the run actually certified under the streak
criterion (`p515_g_g1_g4_admm_gates._derive_stopped_by_from_trajectory`, BY
IMPORT, unchanged -- corrects the harness defect `write_boyd_terminal_
s35ref`'s own naive `stopped_by` embeds) AND this is not a `--smoke-cycles`
run, evaluates the spec v12 `item4_step_3_7_anderson.gate` items that need
the certified models:
  (a) certification cycle (`report['cycles_run']`, the Planner's own usage
      in spec v13's prediction text, "certification at 139" for D) vs the
      <= 80 gate.
  (b) certified cost vs D: |Q - 650966975.2943751| <= 1.5e-4 * D.
  (c) cost decomposition vs D, reusing `p515_s40_cost_decomposition.py`'s
      OWN method (imported `PRICED_COMPONENT_KEYS`/`DETECTOR_COMPONENT_KEYS`
      constants, same dominant-two-components convention) applied to the
      (D, AA) pair -- that script's `main()` is not factored into a
      two-run-callable function, so the method (not the whole script) is
      re-applied here; every key LIST is the imported module constant.
  (d) the interval-hull polish gap at the AA-certified point, reusing
      `p515_s41_hull_polish.py`'s `_polish_all_blocks_hull` (which itself
      calls `hull_entries_with_esso`/`apply_hull_bounds`/`_hull_bounds_
      active`) BY IMPORT, UNCHANGED -- same gate (`|Delta|/cost < 0.1%`),
      same "all 48 blocks must solve" requirement, same reported quantities
      (settlement-excluded change, settlement remainder, non-degenerate
      active-bound counts).
  (e) optionally (`--persist-certified-models`) persists the certified
      TSO/DSO models, hash-recorded, reusing `p515_s42_exact_fix_rerun.py`'s
      `_persist_certified_models`, BY IMPORT, UNCHANGED, called BEFORE the
      hull polish mutates `models` in place.

If the run is a `--smoke-cycles` run, or did not certify (hit the cap
without ever reaching the required streak), items (a)-(e) are SKIPPED
CLEANLY (recorded as `post_certification_items_evaluated: False` with an
explicit reason) -- never attempted on an uncertified point.

================================================================================
EXACT FULL-RUN LAUNCH COMMAND (Planner launches; Worker does NOT)
================================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s43_aa_run.py --persist-certified-models \\
        > data/SRP1/Results/P515S43/aa_run_launch.log 2>&1

Preconditions (checked before anything is written; refuses loudly
otherwise): `.p515_g_gate.lock` absent; no OTHER process (excluding this
process's own ancestor chain) matching `p515_g_g1_g4_admm_gates.py`,
`p515_s39_`, or the broad `p515_s4` campaign pattern in `ps aux`; the output
root does not exist yet (write-once; `--suffix` redirects it); the
production files this task depends on (INCLUDING `data/SRP1/SRP1_params.json`,
the case file) are clean in git; D's committed reference report and
`component_levels_terminal.json` both exist. Run attached, alone (no
`screen`/`nohup`/backgrounding), both streams captured via the shell
redirection above -- NEVER `&`.

================================================================================
SMOKE TEST (the Worker runs THIS one)
================================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s43_aa_run.py --smoke-cycles 3 \\
        > data/SRP1/Results/P515S43/aa_run_smoke_launch.log 2>&1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- precondition/hash conventions, BY IMPORT
import p515_s40_cost_decomposition as CD  # noqa: E402 -- PRICED/DETECTOR_COMPONENT_KEYS, BY IMPORT
import p515_s41_hull_polish as HP  # noqa: E402 -- _polish_all_blocks_hull, BY IMPORT, UNCHANGED
import p515_s42_exact_fix_rerun as EF  # noqa: E402 -- _persist_certified_models, BY IMPORT, UNCHANGED
import p515_s43_aa_flagoff_gate as FG  # noqa: E402 -- AA_NEW_DIAGNOSTIC_FIELD_NAMES, BY IMPORT
import admm_anderson_acceleration as AA  # noqa: E402
import shared_energy_storage_data as SED  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_s40_polish_gap import (  # noqa: E402
    ARM_LABEL, FULL_NUM_MAX_ITERS, D_REFERENCE_PATH, D_CERTIFIED_COST,
    _build_floor_rows, _refuse_overwrite,
)

OUT_DIR_FULL = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S43', 'aa_run')
OUT_DIR_SMOKE = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S43', 'aa_run_smoke')

CERT_GATE_CYCLE_CAP = 80  # Addendum 23 amendment (v): "certification <= 80 cycles at C*"
COST_RELATIVE_TOLERANCE = 1.5e-4  # Addendum 23 amendment (v): "within 1.5e-4 relative of D"
COST_ABS_TOLERANCE = COST_RELATIVE_TOLERANCE * D_CERTIFIED_COST  # == 97645.05 (spec v12 wording)

# p515_s40_cost_decomposition.py section 4's own convention -- NOT exported
# as a module constant there (it is a local variable inside `main()`), so
# named here explicitly, with the source cited, rather than silently
# re-deriving it.
DOMINANT_TWO_COMPONENTS = ['generation_cost', 'flexibility_cost_internal']
# Planner decision before the full run (spec v12 item4 gate (c): the AA-vs-D difference must reconcile to the two known
# components). Every earlier decomposition (run 1, A, C, D; p515_s40_cost_decomposition.py) reconciled to ~1e-7 with
# every other priced component IDENTICALLY zero, so the pass condition is: |unaccounted residual| <= 1.0 (currency
# units) AND every non-dominant priced component's difference is exactly 0.0. The detector-penalty difference is
# reported (it was ~1e-4 between configurations) and counted in the accounted total, not gated.
RECONCILIATION_RESIDUAL_ABS_TOL = 1.0

FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = ('p515_g_g1_g4_admm_gates.py', 'p515_s39_', 'p515_s4')
PRODUCTION_FILES_TO_CHECK_CLEAN = tuple(CP.PRODUCTION_FILES_TO_CHECK_CLEAN) + (
    'admm_anderson_acceleration.py',)


def _check_preconditions(out_dir):
    failures = []
    lock_path = os.path.join(REPO, '.p515_g_gate.lock')
    if os.path.exists(lock_path):
        failures.append(f'lock file already exists: {lock_path}')

    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan process table: {error}')
        ps_output = ''
    excluded_pids = {str(p) for p in CP._ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        pid = fields[1] if len(fields) > 1 else None
        if pid in excluded_pids:
            continue
        if any(substring in line for substring in FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')

    if os.path.exists(out_dir):
        failures.append(f'output directory already exists (write-once): {out_dir}')

    try:
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--'] + list(PRODUCTION_FILES_TO_CHECK_CLEAN),
            capture_output=True, text=True, check=True, cwd=REPO).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not run git status: {error}')
        status = ''
    if status.strip():
        failures.append(f'production files (including the case file) are not clean in git:\n{status}')

    if not os.path.isfile(D_REFERENCE_PATH):
        failures.append(f"D's committed reference report missing: {D_REFERENCE_PATH}")
    d_component_levels_path = os.path.join(CD.RUNS['D']['dir'], 'component_levels_terminal.json')
    if not os.path.isfile(d_component_levels_path):
        failures.append(f"D's committed component_levels_terminal.json missing: {d_component_levels_path}")

    return failures


# ==============================================================================
#  the ONE configuration action: turn AA on (the case file already encodes
#  every other D-oracle setting -- verified here, not assumed)
# ==============================================================================
def _aa_on_pre_solve_hook(planning, sed, candidate, report):
    admm_params = planning.params.admm
    case_file_checks = {
        'rho_v_matches_D': all(float(v) == G.S39_RHO_V for v in admm_params.rho['v'].values()),
        'rho_pf_matches_D': all(float(v) == G.S39_RHO_PF for v in admm_params.rho['pf'].values()),
        'rho_ess_matches_D': all(float(v) == G.S39_RHO_ESS for v in admm_params.rho['ess'].values()),
        'tau_is_0': admm_params.proximal_regularization['tso'].get('tau') == float(G.S39_TAU),
        'gamma_policy_tied_to_rho': (
            admm_params.proximal_regularization['tso'].get('gamma_policy') == 'tied_to_rho'),
        'balancing_exempt_until_matches_D': (
            admm_params.penalty_update.get('balancing_exempt_until')
            == {'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}}),
        'balancing_exempt_channels_empty': (
            not admm_params.penalty_update.get('balancing_exempt_channels')),
        'freeze_after_unchanged_cycles_is_10': (
            admm_params.penalty_update.get('freeze_after_unchanged_cycles')
            == G.S39_FREEZE_AFTER_UNCHANGED_CYCLES),
        'freeze_backstop_cycle_is_200': (
            admm_params.penalty_update.get('freeze_backstop_cycle') == G.S39_FREEZE_BACKSTOP_CYCLE),
        'minimum_consecutive_converged_cycles_is_10': (
            admm_params.minimum_consecutive_converged_cycles == G.S39_REQUIRED_CONSECUTIVE_CYCLES),
        'shared_ess_initialization_is_standalone': (
            admm_params.shared_ess_initialization == 'standalone'),
    }
    missing = [k for k, v in case_file_checks.items() if not v]
    if missing:
        raise RuntimeError(
            f'S43 AA run: the case file does NOT encode the D-oracle configuration as '
            f'expected -- Step 3 closure assumption violated: {missing}')

    admm_params.anderson_acceleration = dict(admm_params.anderson_acceleration)
    admm_params.anderson_acceleration['enabled'] = True
    aa_checks = {
        'anderson_acceleration_enabled_is_true': admm_params.anderson_acceleration['enabled'] is True,
        'anderson_acceleration_memory_is_5': admm_params.anderson_acceleration.get('memory') == 5,
        'anderson_acceleration_regularization_is_1e-10': (
            admm_params.anderson_acceleration.get('regularization') == 1e-10),
    }
    missing2 = [k for k, v in aa_checks.items() if not v]
    if missing2:
        raise RuntimeError(f'S43 AA run: AA settings not as expected after override: {missing2}')

    report.setdefault('rule_eleven_checklist', {})['s43_case_file_oracle_checks'] = case_file_checks
    report['rule_eleven_checklist']['s43_anderson_acceleration_checks'] = aa_checks
    report['rule_eleven_checklist']['s43_anderson_acceleration_settings_in_force'] = dict(
        admm_params.anderson_acceleration)


# ==============================================================================
#  zero-solve precheck (rule eleven: assert the capture path BEFORE the run)
# ==============================================================================
def _zero_solve_precheck(precheck_eval_id):
    """Before any solve: (1) the pre_solve_hook applies cleanly and AA reads
    enabled on a freshly built (unsolved) planning object; (2) `admm_
    anderson_acceleration.build_iterate_layout` can be constructed on that
    same object (validates every static per-run quantity the AA module needs
    -- v_base, s_base, interface_rating, S_ref -- is available before the
    ADMM loop starts, not merely at some later cycle); (3) every `aa_*` field
    this task's sidecar needs is actually present in `shared_resources_
    planning.py`'s own `admm_diagnostics` dict literal (source-level, so a
    field-name typo here is caught before a 300-cycle run, not after).
    Armed `SolveProfileGuard(permitted=())`, not merely asserted."""
    import inspect
    import shared_resources_planning as srp

    guard = SolveProfileGuard(permitted=(), label='P5.15-S43 AA run zero-solve precheck').install()
    try:
        planning = G.O.fresh_planning(precheck_eval_id)
        report = {}
        _aa_on_pre_solve_hook(planning, planning.shared_ess_data, candidate=None, report=report)
        admm_params = planning.params.admm
        aa_enabled = AA.anderson_acceleration_enabled(admm_params)
        layout = AA.build_iterate_layout(planning, admm_params)
        checks = {
            'aa_enabled_after_hook': aa_enabled is True,
            'layout_has_entries': layout['n'] > 0,
            'layout_shared_ess_reference_rating_mva_matches_case_file': (
                layout['shared_ess_reference_rating_mva'] == admm_params.shared_ess_reference_rating_mva),
        }
        module_source = inspect.getsource(srp)
        for field in sorted(FG.AA_NEW_DIAGNOSTIC_FIELD_NAMES):
            checks[f'admm_diagnostics_key_{field}_present_in_source'] = (f"'{field}':" in module_source)
        missing = [k for k, v in checks.items() if not v]
        failures = guard.verify(0)
        checks['solve_profile_guard_verify_0_ok'] = (len(failures) == 0)
        if failures:
            missing.append('solve_profile_guard_verify_0_ok')
        return checks, missing, layout
    finally:
        guard.uninstall()


# ==============================================================================
#  cost decomposition vs D (item c) -- re-applies p515_s40_cost_decomposition.py's
#  own method/constants to the (D, AA) pair; that script's main() is not
#  factored into a callable two-run function.
# ==============================================================================
def _cost_decomposition_vs_d(aa_component_levels, aa_gross_cost, reference_dir=None):
    # P5.15 Addendum 25 item 2 (campaign harness post-certification step):
    # `reference_dir` names the D evaluation to reconcile against (the harness
    # passes the D evaluation of the SAME candidate). Default None = D's
    # committed run dir, exactly as before -- this script's own call is unchanged.
    d_dir = reference_dir if reference_dir is not None else CD.RUNS['D']['dir']
    with open(os.path.join(d_dir, 'component_levels_terminal.json')) as handle:
        d_cl = json.load(handle)
    d_gross = d_cl['recourse_components']['gross_operational_cost']
    tw_d = d_cl['totals_weighted']
    tw_aa = aa_component_levels['totals_weighted']
    all_keys = sorted(set(tw_d) | set(tw_aa))

    component_table = {}
    for key in all_keys:
        d_v, aa_v = tw_d.get(key), tw_aa.get(key)
        component_table[key] = {
            'D': d_v, 'AA': aa_v,
            'diff_AA_minus_D': (aa_v - d_v) if (d_v is not None and aa_v is not None) else None,
            'category': (
                'priced' if key in CD.PRICED_COMPONENT_KEYS else
                'detector_D' if key in CD.DETECTOR_COMPONENT_KEYS else
                'detector_D_total' if key == 'detector_penalty_total' else
                'other'),
        }

    headline_diff = aa_gross_cost - d_gross
    dominant_two_diff = sum(component_table[k]['diff_AA_minus_D'] for k in DOMINANT_TWO_COMPONENTS)
    detector_diff = component_table['detector_penalty_total']['diff_AA_minus_D']
    other_priced_diff = sum(
        component_table[k]['diff_AA_minus_D'] for k in CD.PRICED_COMPONENT_KEYS
        if k not in DOMINANT_TWO_COMPONENTS)
    accounted = dominant_two_diff + other_priced_diff + detector_diff
    unaccounted_residual = headline_diff - accounted
    tolerance = RECONCILIATION_RESIDUAL_ABS_TOL
    other_priced_nonzero = [k for k in CD.PRICED_COMPONENT_KEYS
                            if k not in DOMINANT_TWO_COMPONENTS
                            and component_table[k]['diff_AA_minus_D'] not in (0, 0.0)]

    return {
        'method_note': (
            "Re-applies p515_s40_cost_decomposition.py's own reconciliation method (same "
            "PRICED_COMPONENT_KEYS/DETECTOR_COMPONENT_KEYS constants, IMPORTED; same "
            "dominant-two-components convention, that script's section 4) to the (D, AA) "
            "pair -- that script's main() is hardcoded to four specific committed runs and "
            "is not factored into a directly-callable two-run function, so the METHOD (key "
            "lists, reconciliation formula) is re-applied here rather than the whole script "
            "imported; every key LIST used below is the imported module constant, never "
            "retyped."
        ),
        'd_dir': os.path.relpath(d_dir, REPO),
        'd_gross_operational_cost': d_gross,
        'aa_gross_operational_cost': aa_gross_cost,
        'headline_diff_AA_minus_D': headline_diff,
        'dominant_two_components': DOMINANT_TWO_COMPONENTS,
        'dominant_two_diff': dominant_two_diff,
        'other_priced_components_diff': other_priced_diff,
        'detector_penalty_total_diff': detector_diff,
        'accounted_total': accounted,
        'unaccounted_residual': unaccounted_residual,
        'unaccounted_residual_over_abs_headline_diff': (
            abs(unaccounted_residual) / abs(headline_diff) if headline_diff else None),
        'reconciliation_tolerance_used': tolerance,
        'reconciliation_tolerance_source': 'Planner, before the full run: |residual| <= 1.0 and other priced components identically 0',
        'other_priced_components_nonzero': other_priced_nonzero,
        'reconciles': bool(abs(unaccounted_residual) <= tolerance and not other_priced_nonzero),
        'component_table': component_table,
    }


# ==============================================================================
#  AA per-cycle sidecar (derived post-hoc from the SAME aa_* fields already in
#  cycle_trajectory -- no new computation)
# ==============================================================================
def _build_aa_per_cycle_sidecar(rows, path):
    _refuse_overwrite(path)
    with open(path, 'w') as handle:
        for row in rows:
            entry = {'cycle': row.get('cycle')}
            for field in sorted(FG.AA_NEW_DIAGNOSTIC_FIELD_NAMES):
                entry[field] = row.get(field)
            handle.write(json.dumps(entry, default=str) + '\n')


# ==============================================================================
#  post-run hook
# ==============================================================================
def _make_post_run_hook(out_dir, label, floor_rows_by_node, floor_sidecar_path,
                        recourse_jump_path, ess_stride_path, pf_stride_path,
                        exempt_until_state_path, persist_models, is_smoke, result_holder):
    def _hook(planning, sed, models, rows, report, out_dir=out_dir, label=label, state=None):
        report['s34_recourse_jump_sidecar_path'] = os.path.relpath(recourse_jump_path, REPO)
        report['s34_ess_entry_stride_sidecar_path'] = os.path.relpath(ess_stride_path, REPO)
        report['s35ref_soh_floor_sidecar_path'] = os.path.relpath(floor_sidecar_path, REPO)
        report['s38_pf_entry_stride_sidecar_path'] = os.path.relpath(pf_stride_path, REPO)
        report['s39_ess_exempt_until_state_sidecar_path'] = os.path.relpath(
            exempt_until_state_path, REPO)

        # Standard s39-style captures (component_levels_terminal.json,
        # interface_settlement_detail_s31c.json, interface_voltage_terminal.json,
        # boyd_terminal.json) -- BY IMPORT, unchanged, same call `_s39_hook` makes.
        G.write_boyd_terminal_s35ref(planning, sed, models, rows, report, out_dir, label,
                                     floor_rows_by_node=floor_rows_by_node,
                                     floor_sidecar_path=floor_sidecar_path)

        aa_sidecar_path = os.path.join(out_dir, 'aa_per_cycle.jsonl')
        _build_aa_per_cycle_sidecar(rows, aa_sidecar_path)
        result_holder['aa_per_cycle_sidecar_path'] = os.path.relpath(aa_sidecar_path, REPO)
        print(f'[S43-AA-RUN] wrote {aa_sidecar_path} ({len(rows)} rows)')

        state_dict = state or {}
        result_holder['peak_rss_ru_maxrss'] = state_dict.get('peak_rss_ru_maxrss')
        result_holder['peak_rss_platform_units'] = state_dict.get('peak_rss_platform_units')
        print(f"[S43-AA-RUN] peak_rss_ru_maxrss={result_holder['peak_rss_ru_maxrss']} "
              f"platform_units={result_holder['peak_rss_platform_units']!r}")

        stopped_by_info = G._derive_stopped_by_from_trajectory(
            rows, cap=len(rows) if is_smoke else FULL_NUM_MAX_ITERS,
            required_consecutive=G.S39_REQUIRED_CONSECUTIVE_CYCLES)
        result_holder['stopped_by_info'] = stopped_by_info
        certification_cycle = report.get('cycles_run')
        result_holder['certification_cycle'] = certification_cycle
        certified = (stopped_by_info['stopped_by'] == 'boyd')
        result_holder['certified'] = certified
        print(f'[S43-AA-RUN] cycles_run={certification_cycle} stopped_by={stopped_by_info["stopped_by"]!r} '
              f'certified={certified}')

        if is_smoke or not certified:
            result_holder['post_certification_items_evaluated'] = False
            result_holder['post_certification_skip_reason'] = (
                'smoke test (post-certification items intentionally not evaluated)' if is_smoke else
                f"run did not certify under the streak criterion (stopped_by="
                f"{stopped_by_info['stopped_by']!r}, cycles_run={certification_cycle}, "
                f"cap={FULL_NUM_MAX_ITERS})")
            print(f"[S43-AA-RUN] post-certification items SKIPPED: "
                  f"{result_holder['post_certification_skip_reason']}")
            return

        result_holder['post_certification_items_evaluated'] = True
        print('[S43-AA-RUN] certified -- evaluating spec v12 item4_step_3_7_anderson gate '
              'items (a)-(e) ...', flush=True)

        # -- item (a): certification cycle vs <= 80 ------------------------------
        gate_a = {
            'certification_cycle': certification_cycle, 'threshold': CERT_GATE_CYCLE_CAP,
            'pass': bool(certification_cycle is not None and certification_cycle <= CERT_GATE_CYCLE_CAP),
        }
        result_holder['gate_a_certification_cycle'] = gate_a
        print(f"[S43-AA-RUN] gate (a): certification_cycle={certification_cycle} <= "
              f"{CERT_GATE_CYCLE_CAP}: {gate_a['pass']}")

        # -- item (b): certified cost vs D ----------------------------------------
        q = report.get('gross_operational_cost')
        abs_diff = abs(q - D_CERTIFIED_COST) if q is not None else None
        gate_b = {
            'certified_cost': q, 'd_certified_cost': D_CERTIFIED_COST, 'abs_diff': abs_diff,
            'relative_tolerance': COST_RELATIVE_TOLERANCE, 'abs_tolerance': COST_ABS_TOLERANCE,
            'pass': bool(abs_diff is not None and abs_diff <= COST_ABS_TOLERANCE),
        }
        result_holder['gate_b_cost_vs_d'] = gate_b
        print(f"[S43-AA-RUN] gate (b): |Q-D|={abs_diff} <= {COST_ABS_TOLERANCE}: {gate_b['pass']}")

        # -- item (c): cost decomposition vs D ------------------------------------
        component_levels_path = os.path.join(out_dir, 'component_levels_terminal.json')
        with open(component_levels_path) as handle:
            aa_cl = json.load(handle)
        decomposition = _cost_decomposition_vs_d(aa_cl, q)
        result_holder['gate_c_cost_decomposition'] = decomposition
        print(f"[S43-AA-RUN] gate (c): headline_diff={decomposition['headline_diff_AA_minus_D']} "
              f"dominant_two_diff={decomposition['dominant_two_diff']} "
              f"unaccounted_residual={decomposition['unaccounted_residual']} "
              f"reconciles={decomposition['reconciles']}")

        # -- item (e), BEFORE polish mutates models: optionally persist certified models --
        if persist_models:
            print('[S43-AA-RUN] persisting certified models before hull polish ...', flush=True)
            result_holder['persisted_models'] = EF._persist_certified_models(models, out_dir)
        else:
            result_holder['persisted_models'] = None

        # -- item (d): interval-hull polish gap, BY IMPORT (HP), UNCHANGED -------
        if state is None or 'consensus_vars' not in state:
            raise RuntimeError('S43 AA run: state/consensus_vars not available to '
                               'post_run_hook -- cannot build the hull.')
        print('[S43-AA-RUN] hull-polishing 48 network blocks at the AA-certified point ...', flush=True)
        polish_started = time.time()
        polish, hull_bound_detail = HP._polish_all_blocks_hull(planning, models, state['consensus_vars'])
        polish['runtime_s'] = time.time() - polish_started
        result_holder['gate_d_hull_polish'] = polish
        result_holder['hull_bound_detail'] = hull_bound_detail
        gate = polish['gate']
        if gate is not None:
            print(f"[S43-AA-RUN] gate (d): relative={gate['relative_pct']}% "
                  f"(threshold {gate['threshold_pct']}%) pass={gate['pass']}", flush=True)
        else:
            print(f"[S43-AA-RUN] gate (d) NOT evaluated -- polish failed on blocks: "
                  f"{polish['failed_blocks']}", flush=True)
    return _hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-cycles', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--persist-certified-models', action='store_true', default=False)
    args = parser.parse_args()

    is_smoke = args.smoke_cycles is not None
    sfx = ('_' + args.suffix.strip('_')) if args.suffix.strip('_') else ''
    out_dir = (OUT_DIR_SMOKE if is_smoke else OUT_DIR_FULL) + sfx
    num_max_iters = args.smoke_cycles if is_smoke else FULL_NUM_MAX_ITERS
    run_eval_id = ('p515s43_aa_run_smoke_run' if is_smoke else 'p515s43_aa_run_run') + sfx
    precheck_eval_id = run_eval_id + '_precheck'
    zero_solve_precheck_eval_id = run_eval_id + '_zerosolve_precheck'

    failures = _check_preconditions(out_dir)
    if failures:
        for f in failures:
            print(f'[S43-AA-RUN PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    print('[S43-AA-RUN] preconditions passed (no lock, no forbidden process, fresh output '
          "dir, production files + case file clean in git, D's reference present).")

    zero_solve_checks, zero_solve_missing, layout = _zero_solve_precheck(zero_solve_precheck_eval_id)
    print(f'[S43-AA-RUN] zero-solve precheck: {zero_solve_checks}')
    if zero_solve_missing:
        print(f'[S43-AA-RUN] *** ZERO-SOLVE PRECHECK FAILED *** missing/broken: {zero_solve_missing}')
        raise SystemExit(1)
    print(f"[S43-AA-RUN] zero-solve precheck passed ({layout['n']} AA iterate entries; "
          f"S_ref={layout['shared_ess_reference_rating_mva']}).")

    G._acquire_exclusive_run_lock()
    started = time.time()

    _capture_checklist, floor_rows_by_node, floor_counts_by_node = _build_floor_rows(precheck_eval_id)
    print(f'[S43-AA-RUN] soh_floor_row_counts_by_node='
          f"{ {n: len(r) for n, r in floor_rows_by_node.items()} }")

    recourse_jump_path = os.path.join(out_dir, 'recourse_jump_sidecar_baseline.jsonl')
    ess_stride_path = os.path.join(out_dir, 'ess_entry_stride_baseline.jsonl')
    floor_sidecar_path = os.path.join(out_dir, 'soh_floor_sidecar_baseline.jsonl')
    pf_stride_path = os.path.join(out_dir, f'pf_entry_stride_{ARM_LABEL}.jsonl')
    exempt_until_state_path = os.path.join(out_dir, f'ess_exempt_until_state_{ARM_LABEL}.jsonl')
    for path in (recourse_jump_path, ess_stride_path, floor_sidecar_path, pf_stride_path,
                exempt_until_state_path):
        _refuse_overwrite(path)

    result_holder = {}
    hook = _make_post_run_hook(out_dir, ARM_LABEL, floor_rows_by_node, floor_sidecar_path,
                               recourse_jump_path, ess_stride_path, pf_stride_path,
                               exempt_until_state_path, args.persist_certified_models,
                               is_smoke, result_holder)

    print(f'[S43-AA-RUN] run CASE-FILE-ALONE (label={ARM_LABEL!r}, apply_rho=False) with '
          f'anderson_acceleration.enabled=True, num_max_iters_override={num_max_iters} '
          f"({'SMOKE' if is_smoke else 'FULL -- to certification or cap 300'}).")
    with G.s38_pf_capture_hooks(recourse_jump_path, ess_stride_path, floor_sidecar_path,
                                pf_stride_path, floor_rows_by_node, stride=1), \
         G.s39_exempt_until_capture_hooks(exempt_until_state_path):
        report, report_path = G.run_admm_arm(
            ARM_LABEL, out_dir, k_override=None, eval_id=run_eval_id,
            num_max_iters_override=num_max_iters, apply_rho=False,
            full_diagnostics_in_rows=True, post_run_hook=hook,
            pre_solve_hook=_aa_on_pre_solve_hook)

    print(f"[S43-AA-RUN] cycles_run={report['cycles_run']} recourse={report['recourse']} "
          f"wall={report['wall_clock_s']:.1f}s")

    payload = {
        'stage': 'P5.15 Step 3.7 integration -- Task 2: flag-ON Anderson acceleration run',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 22/23/24',
            'data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json',
            'data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'arm_label': ARM_LABEL,
        'mode': 'smoke' if is_smoke else 'full',
        'smoke_cycles': args.smoke_cycles,
        'num_max_iters_used': num_max_iters,
        'cap': FULL_NUM_MAX_ITERS,
        'required_consecutive_cycles': G.S39_REQUIRED_CONSECUTIVE_CYCLES,
        'out_dir': os.path.relpath(out_dir, REPO),
        'admm_report_path': os.path.relpath(report_path, REPO),
        'admm_cycles_run': report['cycles_run'],
        'admm_gross_operational_cost': report['gross_operational_cost'],
        'zero_solve_precheck': zero_solve_checks,
        'aa_per_cycle_sidecar_path': result_holder.get('aa_per_cycle_sidecar_path'),
        'peak_rss_ru_maxrss': result_holder.get('peak_rss_ru_maxrss'),
        'peak_rss_platform_units': result_holder.get('peak_rss_platform_units'),
        'stopped_by_info': result_holder.get('stopped_by_info'),
        'certification_cycle': result_holder.get('certification_cycle'),
        'certified': result_holder.get('certified'),
        'post_certification_items_evaluated': result_holder.get('post_certification_items_evaluated'),
        'post_certification_skip_reason': result_holder.get('post_certification_skip_reason'),
        'gate_a_certification_cycle': result_holder.get('gate_a_certification_cycle'),
        'gate_b_cost_vs_d': result_holder.get('gate_b_cost_vs_d'),
        'gate_c_cost_decomposition': result_holder.get('gate_c_cost_decomposition'),
        'gate_d_hull_polish': result_holder.get('gate_d_hull_polish'),
        'hull_bound_detail_written_separately': True,
        'persisted_models': result_holder.get('persisted_models'),
        'wall_clock_s': time.time() - started,
    }

    hull_bound_detail = result_holder.get('hull_bound_detail')
    if hull_bound_detail is not None:
        hull_bound_detail_path = os.path.join(out_dir, 'hull_bound_detail.json')
        _refuse_overwrite(hull_bound_detail_path)
        with open(hull_bound_detail_path, 'w') as handle:
            json.dump(hull_bound_detail, handle, indent=1, default=str)
        payload['hull_bound_detail_path'] = os.path.relpath(hull_bound_detail_path, REPO)
    else:
        payload['hull_bound_detail_path'] = None

    results_path = os.path.join(out_dir, 'aa_run_results.json')
    _refuse_overwrite(results_path)
    with open(results_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S43-AA-RUN] wrote {results_path}')

    manifest = {}
    for root, _dirs, files in os.walk(out_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = CP._sha256_file(fpath)
    manifest_path = os.path.join(out_dir, 'manifest_sha256.json')
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S43-AA-RUN] wrote {manifest_path}')

    if not result_holder.get('post_certification_items_evaluated'):
        print(f"[S43-AA-RUN] post-certification items not evaluated: "
              f"{result_holder.get('post_certification_skip_reason')}")
        return

    gate_a = result_holder.get('gate_a_certification_cycle') or {}
    gate_b = result_holder.get('gate_b_cost_vs_d') or {}
    gate_d = (result_holder.get('gate_d_hull_polish') or {}).get('gate')
    print(f"[S43-AA-RUN] GATE SUMMARY: (a) certification_cycle<=80: {gate_a.get('pass')} "
          f"(b) cost_within_tolerance: {gate_b.get('pass')} "
          f"(c) reconciles: "
          f"{(result_holder.get('gate_c_cost_decomposition') or {}).get('reconciles')} "
          f"(d) hull_polish_pass: {gate_d.get('pass') if gate_d else None}")


if __name__ == '__main__':
    main()
