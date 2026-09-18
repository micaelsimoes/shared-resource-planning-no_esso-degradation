"""
P5.15 Addendum 22 item (4) -- Step 3.5, the polish-gap gate on the oracle's
(arm D's) certified point.

Authority: `PLANNER_BRIEF_2026-09-13.md` Step 3.5 (~line 516) and Addendum 22
item `4_step_3_5`; frozen spec v11
`data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json`, item
`4_step_3_5`: "polish gap on D's certified point: re-solve every network
block with the unscaled base objective at fixed consensus (48 network
solves, no ADMM); gate: recourse change < 0.1% of the recourse; fail => stop
for review before touching the rho/sigma ratio."

================================================================================
READING: "fixed consensus" (Worker's determination, citations below)
================================================================================
Step 3.5's own text is: "re-solve every network block with the unscaled base
objective at fixed consensus (P5.7's test)". "P5.7's test" is
`p57_eval.py`/`p57_hypotheses.py`, which run their polish step by CALLING
`p56a_oracle.py`'s primitives verbatim: `common_coordinated_values`,
`apply_common_values`, `restore_base_objective`, `polish_networks`. Reading
those four functions (`p56a_oracle.py` lines 233-528) shows the mechanism:

  - Interface voltage / power flow (no explicit ADMM "z" for these channels
    in EITHER the P5.6-era model or the current P5.15 Boyd-residual mapping
    -- `get_admm_boyd_residual_metrics`'s own docstring, Advisor findings
    F1-F4, calls the TSO's own copy "z" and the DSO's own copy "x", i.e. an
    asymmetric Gauss-Seidel target, not a true two-sided average): the
    "converged consensus value" is taken as the MIDPOINT of the two sides'
    own achieved values, read directly off the solved Pyomo models
    (`t_model.pc_adn`/`qc_adn`, `d_model.pg_adn`/`qg_adn`, `vmag_sqr` on
    both sides) -- "the choice that minimises the largest correction asked
    of either side" (`common_coordinated_values` docstring). This is also
    exactly `p55d_d1_polish.py`'s (P5.5-D1's) `interface_values`/
    `apply_common_values` design (compare line-for-line); P5.7's helpers are
    a direct descendant of that stage, not an independent design. The two
    "designs" the task asks to distinguish are therefore THE SAME mechanism,
    with P5.7 (`p56a_oracle.py`, already imported into
    `p515_g_g1_g4_admm_gates.py` as `O` -- confirming it is
    structure-compatible with the CURRENT planning/model objects, not a
    stale P5.6-era interface) named as the authority Step 3.5 cites.
  - Shared-ESS dispatch: `consensus_vars['ess']['z']['current']` IS an
    explicit, genuine three-way consensus variable in the current ADMM
    (`create_admm_variables`, `shared_resources_planning.py` ~4041-4046) and
    is used directly, unchanged, by `common_coordinated_values`.

Fixing at this common value (`apply_common_values`) pins the model's own
`expected_interface_vmag`/`expected_interface_pf_p/q`/`expected_shared_ess_p/q`
Vars (the actual Boyd-residual "x", tied by hard equality constraints --
`tn_interface_expected_*_rule`/`dn_interface_expected_*_rule`,
`shared_resources_planning.py` ~3615-3693 -- to the physical variables
`apply_common_values` fixes directly) at a SINGLE number on both sides, i.e.
the primal residual r = x - z is set to exactly zero by construction, not
merely driven below the Boyd tolerance. The base objective is then
reactivated (no consensus penalty term at all, since consensus is now a hard
constraint), and every block is re-solved with `network.run_smopf`,
production's own SMOPF entry point -- exactly what Step 3.5 asks for:
"re-solve every network block with the unscaled base objective at fixed
consensus".

`common_coordinated_values`/`apply_common_values`/`per_block_base_objectives`/
`_tagged_holders` are imported BY IMPORT from `p56a_oracle.py` (already a
dependency of `p515_g_g1_g4_admm_gates.py`, imported there as `O`) and called
verbatim -- never reimplemented -- on the live `planning`/`models` a
case-file-alone `run_admm_arm` call returns, via `post_run_hook` (zero extra
solves besides the 48 polish solves themselves; `models` is the SAME dict
`run_operational_planning` returned, per `run_admm_arm`'s own docstring).

`p56a_oracle.restore_base_objective`/`polish_networks` are NOT reused as-is
(finding, confirmed by the smoke test): `restore_base_objective` predates
Step 3.4 (frozen spec v4), which made objective rescaling PRODUCTION
behaviour -- `run_admm_arm` wraps every ADMM cycle in `p58_rescale.
patched_admm_objectives()` unconditionally (`p515_g_g1_g4_admm_gates.py`'s
own `R = p58_rescale` import), which activates a THIRD Objective component
(`p58_rescale.RESCALED_OBJECTIVE`) that `restore_base_objective` does not
know about. Calling it alone after a real `run_admm_arm` run leaves BOTH the
base `objective` and the rescaled objective active; IPOPT's AMPL interface
then raises "more than one objective function ... AmplTNLP::set_active_
objective has not been called" on every block -- reproduced directly (see
Worker report) and confirmed to be the actual cause of every one of the
first smoke-test attempt's 48 polish failures, not an inherent property of a
far-from-converged fixed-consensus point. `_switch_to_base_objective` below
generalizes it (deactivate every Objective except `objective`, activate
`objective`, assert exactly one ends up active) and `_polish_networks_fixed_
consensus` is `polish_networks` with that one substitution -- otherwise
identical, same `_tagged_holders` loop, same `network.run_smopf` call.

Available shared-ESS S/E capacity is NOT separately reconciled here (unlike
`p55d_d1_polish.py`'s explicit capacity-shift step): Step 3.5's text names
only "network block[s] ... at fixed consensus", i.e. the three
augmented-Lagrangian channels (voltage, interface P/Q, shared-ESS P/Q), and
the current ADMM already republishes the SAME `sess_available_capacities`
into every block every cycle (unlike the P5.5-era design `p55d` was
correcting). Out of scope by the task's own wording; noted for the Planner.

================================================================================
BUILD
================================================================================
  1. `run_admm_arm(label='s39_D', ..., apply_rho=False, pre_solve_hook=None)`
     -- the SAME case-file-alone invocation `p515_s40_case_file_repro.py`
     already validated bitwise (2 cycles) -- to Boyd certification, THROUGH a
     `post_run_hook` that receives the live `models`/`state`/`planning`/`sed`
     before they are discarded.

     `num_max_iters_override=300` is REQUIRED, not a configuration change:
     `_construct_arm_planning` unconditionally sets
     `planning.params.admm.num_max_iters` to either the override or
     `N.CAP` (90, `p514_n_instrumented_cstar.py`) when the override is None
     -- so leaving it None would silently CAP the run at cycle 90, short of
     D's certification cycle 139. 300 is exactly D's own case-file
     `admm.num_max_iters` value (`data/SRP1/SRP1_params.json`), so this is
     "pass the case file's own cap through", not an override of it.

  2. Reproduction check (gating), inside the hook, BEFORE polishing: diffs
     the run's OWN report against D's committed
     `data/SRP1/Results/P515S39_D_run/g_s39_D.json`, using
     `p515_s40_clone_capture_preflight.py`'s comparator (`_diff`,
     `EXCLUDE_KEY_NAMES`, `EXCLUDE_DOTTED_SUFFIXES`,
     `INTENTIONAL_DIFF_SUFFIXES`) BY IMPORT. At the run's own cycle count N:
       - if N equals D's own `cycles_run` (139, the full run): the ENTIRE
         report is diffed against D's entire committed report (the same
         comprehensive comparison `p515_s40_case_file_repro.py` already
         used, just against the 139-cycle reference instead of the 2-cycle
         one).
       - if N is smaller (the smoke test): only `cycle_trajectory[:N]` and
         the handful of top-level fields PRODUCTION DERIVES FROM THE ROWS
         (`cycles_run`, `recourse`, `gross_operational_cost`,
         `converged_at_cycle`, `terminal_objective_change_abs`,
         `terminal_objective_tolerance`,
         `rule_ten_terminal_step_over_threshold`, `local_solve_failures` --
         copied verbatim from `run_admm_arm`'s own derivation, `_derive_top_
         level_from_rows` below) are compared, recomputed from D's OWN first
         N rows. Cycle-count-dependent artifacts (`esso_capture`,
         `solve_profile`, `network_failures_summary`,
         `esso_complementarity_diagnostics_by_round`, `esso_models_pickle`)
         are OUT OF SCOPE for a truncated comparison and are not compared --
         this is a stated scope, not silently skipped.
     If reproduction fails, the failure (first differing field, per-mode) is
     recorded in `polish_gap_results.json` written before the hook returns
     (`run_admm_arm` itself still writes `g_<label>.json` afterward --
     nothing is lost), the hook does NOT polish, and `main()` exits 1 after
     everything has been written. This IS the tau=0 determinism check.

  3-4. Polish + gate: see the docstring block above ("fixed consensus").
     Guarded (`SolveProfileGuard`, `('network.py',
     '_run_smopf_solver_attempt')`) separately from the ADMM guard (which is
     already uninstalled by the time `post_run_hook` runs) -- reports
     `permitted_solve` (48 + any recovery retries), `blocked_solve`
     (asserted 0).

  6. Write-once outputs under `data/SRP1/Results/P515S40/polish_gap/`
     (full) or `.../polish_gap_smoke/` (smoke), fresh `eval_id`s
     (`p515s40_polish_gap_run` / `p515s40_polish_gap_smoke_run`), a sha256
     manifest of everything this script itself writes.

================================================================================
EXACT LAUNCH COMMAND (the FULL run -- Planner launches; Worker does NOT)
================================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s40_polish_gap.py \\
        > data/SRP1/Results/P515S40/polish_gap_launch.log 2>&1

Preconditions identical to `p515_s39_preflight.py` / `p515_s40_case_file_repro.py`
(lock absent, no forbidden live process, fresh output root, production files
clean in git, D's committed reference report present) -- checked before
anything is written, refuses loudly otherwise. Run attached, alone, both
streams captured via the shell redirection above.

================================================================================
SMOKE TEST (hidden option; the Worker runs THIS one)
================================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s40_polish_gap.py --smoke-cycles 2 \\
        > data/SRP1/Results/P515S40/polish_gap_smoke_launch.log 2>&1
"""

import argparse
import io
import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p56a_oracle as O  # noqa: E402 -- "P5.7's test" primitives, BY IMPORT
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- comparator conventions, BY IMPORT
import shared_energy_storage_data as SED  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

ARM_LABEL = 's39_D'
FULL_NUM_MAX_ITERS = 300  # D's own case-file cap -- see module docstring item 1
D_REFERENCE_PATH = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S39_D_run', 'g_s39_D.json')
D_CERTIFICATION_CYCLE = 139
D_CERTIFIED_COST = 650966975.2943751

OUT_DIR_FULL = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', 'polish_gap')
OUT_DIR_SMOKE = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S40', 'polish_gap_smoke')

# Polish solves reach the solver through `network.py:_run_smopf_solver_attempt`
# (the SAME call site `p514_n_instrumented_cstar.PERMITTED` names for every
# other network solve in this campaign) -- no other call site is permitted
# during the polish window.
POLISH_PERMITTED = [('network.py', '_run_smopf_solver_attempt')]

GATE_THRESHOLD_PCT = 0.1

FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = tuple(CP.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS) + (
    'p515_s40_',)


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _check_preconditions(out_dir):
    failures = []
    lock_path = os.path.join(REPO, '.p515_g_gate.lock')
    if os.path.exists(lock_path):
        failures.append(f'lock file already exists: {lock_path}')

    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True,
                                   check=True).stdout
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
            ['git', 'status', '--porcelain', '--'] + list(CP.PRODUCTION_FILES_TO_CHECK_CLEAN),
            capture_output=True, text=True, check=True, cwd=REPO).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not run git status: {error}')
        status = ''
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')

    if not os.path.isfile(D_REFERENCE_PATH):
        failures.append(f"D's committed reference report missing: {D_REFERENCE_PATH}")

    return failures


def _derive_top_level_from_rows(rows):
    """Verbatim copy of `run_admm_arm`'s own top-level derivation
    (`p515_g_g1_g4_admm_gates.py` ~1307-1319), so a TRUNCATED reference (D's
    first N rows) reproduces exactly what a real N-cycle run of D would have
    reported at those fields -- not a re-derivation that could drift."""
    last = rows[-1] if rows else {}
    return {
        'cycles_run': len(rows),
        'recourse': last.get('recourse'),
        'gross_operational_cost': last.get('gross_operational_cost'),
        'converged_at_cycle': next((r['cycle'] for r in rows if r['cycle_convergence']), None),
        'terminal_objective_change_abs': last.get('objective_change_abs'),
        'terminal_objective_tolerance': last.get('objective_tolerance'),
        'rule_ten_terminal_step_over_threshold': (
            last.get('objective_change_abs') / last.get('objective_tolerance')
            if last.get('objective_change_abs') and last.get('objective_tolerance') else None),
        'local_solve_failures': sum(1 for r in rows if r.get('local_solves_ok') is False),
    }


def _reproduction_check(report):
    """Diff `report` against D's committed reference, scope depending on
    whether `report` is the full 139-cycle run or a truncated smoke run (see
    module docstring, build item 2). Returns a dict; never raises."""
    with open(D_REFERENCE_PATH) as handle:
        d_full = json.load(handle)

    n = len(report.get('cycle_trajectory') or [])
    d_rows_full = d_full.get('cycle_trajectory') or []
    is_full_mode = (n == len(d_rows_full))

    if is_full_mode:
        diffs = CP._diff(d_full, report, 'report')
        mode = 'full (entire report vs D\'s entire committed report)'
        scalar_checks = {
            'cycles_run': {'mine': report.get('cycles_run'), 'expected': D_CERTIFICATION_CYCLE,
                           'match': report.get('cycles_run') == D_CERTIFICATION_CYCLE},
            'gross_operational_cost': {
                'mine': report.get('gross_operational_cost'), 'expected': D_CERTIFIED_COST,
                'match': report.get('gross_operational_cost') == D_CERTIFIED_COST},
        }
    else:
        d_rows_truncated = d_rows_full[:n]
        row_diffs = CP._diff(d_rows_truncated, report.get('cycle_trajectory') or [],
                             'cycle_trajectory')
        derived_ref = _derive_top_level_from_rows(d_rows_truncated)
        derived_mine = {k: report.get(k) for k in derived_ref}
        scalar_diffs = CP._diff(derived_ref, derived_mine, 'derived_top_level')
        diffs = row_diffs + scalar_diffs
        mode = (f'truncated (cycle_trajectory[:{n}] + rows-derived top-level fields vs '
               f"D's first {n} committed rows; cycle-count-dependent artifacts "
               '(esso_capture, solve_profile, network_failures_summary, '
               'esso_complementarity_diagnostics_by_round, esso_models_pickle) '
               'out of scope for a truncated comparison)')
        scalar_checks = {}

    # P5.15 Step 3.5 (Planner, after the first full run stopped here on a single
    # provenance field): the `rule_eleven_checklist` subtree records HOW a run
    # was configured and verified (e.g. `s39_pre_solve_override_verification`,
    # present only when the s39 override hook ran), not what the run computed.
    # D was configured by overrides; this run is configured by the case file
    # alone (fb3de341), so that subtree necessarily differs. It is reported in
    # full below but does NOT gate. Every other field -- the whole trajectory,
    # costs, solve profile, failures, diagnostics -- still gates bitwise.
    provenance_diffs = [x for x in diffs if '.rule_eleven_checklist' in str(x.get('field', ''))
                        or str(x.get('field', '')).startswith('rule_eleven_checklist')]
    diffs = [x for x in diffs if x not in provenance_diffs]
    reproduces = (len(diffs) == 0)
    return {
        'provenance_diffs_reported_not_gating': provenance_diffs,
        'n_provenance_diffs': len(provenance_diffs),
        'mode': mode,
        'is_full_mode': is_full_mode,
        'n_cycles_compared': n,
        'd_reference_path': os.path.relpath(D_REFERENCE_PATH, REPO),
        'd_reference_cycles_run': d_full.get('cycles_run'),
        'excluded_field_names': sorted(CP.EXCLUDE_KEY_NAMES),
        'excluded_dotted_suffixes': sorted(CP.EXCLUDE_DOTTED_SUFFIXES),
        'intentional_difference_suffixes': sorted(CP.INTENTIONAL_DIFF_SUFFIXES),
        'scalar_checks': scalar_checks,
        'n_diffs': len(diffs),
        'diffs': diffs,
        'first_diffs': diffs[:20],
        'reproduces': reproduces,
    }


def _build_floor_rows(precheck_eval_id):
    """Zero-solve SoH floor-row identification, structural -- mirrors
    `p515_s40_case_file_repro.py`'s own helper of the same name verbatim."""
    precheck_planning = G.O.fresh_planning(precheck_eval_id)
    capture_checklist = G.assert_s31c_capture_paths(precheck_planning)
    active_nodes = list(precheck_planning.shared_ess_data.active_distribution_network_nodes)
    probe_esso_models = {node_id: SED._build_subproblem(precheck_planning.shared_ess_data, node_id)
                          for node_id in active_nodes}
    floor_rows_by_node, floor_counts_by_node = G._identify_soh_floor_rows(probe_esso_models)
    del probe_esso_models, precheck_planning
    return capture_checklist, floor_rows_by_node, floor_counts_by_node


def _switch_to_base_objective(model):
    """`p56a_oracle.restore_base_objective`, GENERALIZED (finding, see Worker
    report): deactivates EVERY Objective component on `model` except
    `objective` (the base), then activates `objective`.

    `p56a_oracle.restore_base_objective` only knows about `admm_objective`
    (P5.6/P5.7-era: only two Objective components ever existed). Step 3.4
    (frozen spec v4) made objective rescaling PRODUCTION behaviour:
    `run_admm_arm` wraps every ADMM cycle in `p58_rescale.
    patched_admm_objectives()` (`p515_g_g1_g4_admm_gates.py`'s own `R =
    p58_rescale` import, applied UNCONDITIONALLY -- not a P5.8-only
    diagnostic any more), which deactivates `admm_objective` and activates a
    THIRD component, `p58_rescale.RESCALED_OBJECTIVE`
    ('p58_rescaled_admm_objective' = `effective_scale * admm_objective`) --
    the objective IPOPT actually solves on every real cycle. Calling only
    `p56a_oracle.restore_base_objective` after a real `run_admm_arm` run
    leaves BOTH `objective` and `p58_rescaled_admm_objective` active; IPOPT's
    AMPL interface then raises "There is more than one objective function in
    the AMPL model, but AmplTNLP::set_active_objective has not been called"
    on every block. Reproduced directly (single-block probe, `tee=True`) and
    is the ROOT CAUSE of every one of the 48 smoke-test polish failures --
    NOT an inherent property of the far-from-converged 2-cycle fixed-
    consensus point.

    Asserts exactly one Objective is active afterward -- fails loudly, not
    silently, if this generalization is ever itself incomplete."""
    for obj in list(model.component_objects(pe.Objective, active=True, descend_into=False)):
        if obj.local_name != 'objective':
            obj.deactivate()
    model.objective.activate()
    active = [o.local_name for o in
             model.component_objects(pe.Objective, active=True, descend_into=False)]
    if active != ['objective']:
        raise RuntimeError(f"S40 polish gap: expected exactly one active objective "
                           f"(['objective']) after switching, got {active}")


def _polish_networks_fixed_consensus(planning, models):
    """`p56a_oracle.polish_networks`, with `_switch_to_base_objective` in
    place of `restore_base_objective` (see its docstring); otherwise
    identical -- same per-block loop (`p56a_oracle._tagged_holders`), same
    `network.run_smopf` call (production's own SMOPF entry point), same
    return shape `(blocks, all_solved)`."""
    blocks, all_solved = [], True
    for tag, holder in O._tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                network = holder.network[year][day]
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                _switch_to_base_objective(model)
                with redirect_stdout(io.StringIO()):
                    result = network.run_smopf(model, holder.params, print_header=False)
                ok = bool(srp._solver_result_succeeded(result))
                all_solved &= ok
                blocks.append({'agent': tag, 'year': year, 'day': day, 'solved': ok})
    return blocks, all_solved


def _polish_all_blocks(planning, models, consensus_vars):
    """The polish step itself (build items 3-4). Mutates `models` in place
    (no clone -- these models are discarded by the caller regardless, the
    same convention `p55d_d1_polish.py` uses). Returns the full polish
    record; never raises on a per-block solver failure (reported, not
    dropped -- gate is None when `all_solved` is False)."""
    before_recourse = planning.get_operational_recourse_components(models)
    before_per_block = O.per_block_base_objectives(planning, models)

    common = O.common_coordinated_values(planning, models, consensus_vars)
    O.apply_common_values(planning, models, common)

    guard = SolveProfileGuard(POLISH_PERMITTED, label='P5.15-S40 polish-gap').install()
    try:
        blocks, all_solved = _polish_networks_fixed_consensus(planning, models)
    finally:
        guard.uninstall()

    after_per_block = O.per_block_base_objectives(planning, models)

    per_block_records = []
    for b in blocks:
        key = f"{b['agent']}|{b['year']}|{b['day']}"
        before_v = before_per_block[key]['weighted_base_objective']
        after_v = after_per_block[key]['weighted_base_objective']
        per_block_records.append({
            'block': key, 'solved': b['solved'],
            'weighted_base_objective_before': before_v,
            'weighted_base_objective_after': after_v,
            'delta': after_v - before_v,
        })

    failed_blocks = [r['block'] for r in per_block_records if not r['solved']]

    gate = None
    after_recourse = None
    if all_solved:
        after_recourse = planning.get_operational_recourse_components(models)
        recourse_before = before_recourse['gross_operational_cost']
        recourse_after = after_recourse['gross_operational_cost']
        delta = recourse_after - recourse_before
        relative = abs(delta) / abs(recourse_before) if recourse_before else None
        gate = {
            'objective_convention': ('gross_operational_cost -- settlement-excluded '
                                     'system-cost recourse, the SAME convention the '
                                     'certified cost is reported in'),
            'recourse_before': recourse_before, 'recourse_after': recourse_after,
            'delta': delta, 'delta_sign': ('increase' if delta > 0 else
                                           'decrease' if delta < 0 else 'unchanged'),
            'relative_pct': (relative * 100.0) if relative is not None else None,
            'threshold_pct': GATE_THRESHOLD_PCT,
            'pass': (relative is not None and relative < GATE_THRESHOLD_PCT / 100.0),
        }

    per_block_sorted = sorted(per_block_records, key=lambda r: abs(r['delta']), reverse=True)

    return {
        'before_recourse_components': before_recourse,
        'after_recourse_components': after_recourse,
        'all_solved': all_solved,
        'failed_blocks': failed_blocks,
        'n_blocks': len(blocks),
        'per_block': per_block_records,
        'per_block_by_largest_abs_delta': per_block_sorted,
        'solve_profile': {
            'observed': dict(guard.counts),
            'n_blocks_dispatched': len(blocks),
            'retries_beyond_one_per_block': guard.counts['permitted_solve'] - len(blocks),
            'blocked_calls': guard.counts['blocked_solve'] + guard.counts['blocked_exec'],
        },
        'gate': gate,
    }


def _make_post_run_hook(out_dir, label, floor_rows_by_node, floor_sidecar_path,
                        recourse_jump_path, ess_stride_path, pf_stride_path,
                        exempt_until_state_path, result_holder):
    def _hook(planning, sed, models, rows, report, out_dir=out_dir, label=label,
              state=None):
        # provenance only -- matches p515_s40_case_file_repro.py's own `_hook`;
        # not read by write_boyd_terminal_s35ref itself.
        report['s34_recourse_jump_sidecar_path'] = os.path.relpath(recourse_jump_path, REPO)
        report['s34_ess_entry_stride_sidecar_path'] = os.path.relpath(ess_stride_path, REPO)
        report['s35ref_soh_floor_sidecar_path'] = os.path.relpath(floor_sidecar_path, REPO)
        report['s38_pf_entry_stride_sidecar_path'] = os.path.relpath(pf_stride_path, REPO)
        report['s39_ess_exempt_until_state_sidecar_path'] = os.path.relpath(
            exempt_until_state_path, REPO)

        # zero-solve terminal capture, same convention every s34/s35ref/s38/s39
        # arm uses -- boyd_terminal.json, interface_settlement_detail_s31c.json,
        # interface_voltage_terminal.json, component_levels_terminal.json.
        G.write_boyd_terminal_s35ref(planning, sed, models, rows, report, out_dir, label,
                                     floor_rows_by_node=floor_rows_by_node,
                                     floor_sidecar_path=floor_sidecar_path)

        repro = _reproduction_check(report)
        result_holder['reproduction'] = repro
        if not repro['reproduces']:
            result_holder['polish'] = None
            result_holder['stopped_before_polish'] = True
            print(f"[S40-POLISH-GAP] REPRODUCTION CHECK FAILED (mode={repro['mode']}): "
                  f"{repro['n_diffs']} diffs. STOPPING before polish. First diffs:")
            for d in repro['first_diffs']:
                print(f'  [FIRST DIFFS] {d}')
            return

        if state is None or 'consensus_vars' not in state:
            raise RuntimeError('S40 polish gap: state/consensus_vars not available to '
                               'post_run_hook -- cannot fix consensus.')
        print('[S40-POLISH-GAP] reproduction OK; polishing 48 network blocks '
              'at fixed consensus ...', flush=True)
        polish_started = time.time()
        polish = _polish_all_blocks(planning, models, state['consensus_vars'])
        polish['runtime_s'] = time.time() - polish_started
        result_holder['polish'] = polish
        result_holder['stopped_before_polish'] = False
        gate = polish['gate']
        if gate is not None:
            print(f"[S40-POLISH-GAP] gate: relative={gate['relative_pct']}% "
                  f"(threshold {gate['threshold_pct']}%) pass={gate['pass']}", flush=True)
        else:
            print(f"[S40-POLISH-GAP] gate NOT evaluated -- polish failed on blocks: "
                  f"{polish['failed_blocks']}", flush=True)
    return _hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-cycles', type=int, default=None, help=argparse.SUPPRESS)
    # Optional run suffix: a fresh output root AND fresh working-dir ids, so a
    # re-run never touches an earlier run's committed evidence.
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    is_smoke = args.smoke_cycles is not None
    sfx = ('_' + args.suffix.strip('_')) if args.suffix.strip('_') else ''
    out_dir = (OUT_DIR_SMOKE if is_smoke else OUT_DIR_FULL) + sfx
    num_max_iters = args.smoke_cycles if is_smoke else FULL_NUM_MAX_ITERS
    run_eval_id = ('p515s40_polish_gap_smoke_run' if is_smoke else 'p515s40_polish_gap_run') + sfx
    precheck_eval_id = run_eval_id + '_precheck'

    failures = _check_preconditions(out_dir)
    if failures:
        for f in failures:
            print(f'[S40-POLISH-GAP PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    print('[S40-POLISH-GAP] preconditions passed (no lock, no forbidden process, '
          "fresh output dir, production files clean, D's reference present).")

    G._acquire_exclusive_run_lock()
    os.makedirs(out_dir, exist_ok=True)

    started = time.time()
    _capture_checklist, floor_rows_by_node, floor_counts_by_node = _build_floor_rows(
        precheck_eval_id)
    print(f'[S40-POLISH-GAP] soh_floor_row_counts_by_node='
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
                               exempt_until_state_path, result_holder)

    print(f'[S40-POLISH-GAP] run CASE-FILE-ALONE (label={ARM_LABEL!r}, apply_rho=False, '
          f'pre_solve_hook=None), num_max_iters_override={num_max_iters} '
          f"({'SMOKE' if is_smoke else 'FULL -- to Boyd certification'}).")
    with G.s38_pf_capture_hooks(recourse_jump_path, ess_stride_path, floor_sidecar_path,
                                pf_stride_path, floor_rows_by_node, stride=1), \
         G.s39_exempt_until_capture_hooks(exempt_until_state_path):
        report, report_path = G.run_admm_arm(
            ARM_LABEL, out_dir, k_override=None, eval_id=run_eval_id,
            num_max_iters_override=num_max_iters, apply_rho=False,
            full_diagnostics_in_rows=True, post_run_hook=hook, pre_solve_hook=None)

    print(f"[S40-POLISH-GAP] cycles_run={report['cycles_run']} recourse={report['recourse']} "
          f"wall={report['wall_clock_s']:.1f}s")

    payload = {
        'stage': 'P5.15 Addendum 22 item (4) -- Step 3.5 polish-gap gate',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Step 3.5 and Addendum 22 item 4_step_3_5',
            'data/SRP1/Results/P515S40/frozen_s40_closure_spec_v11_0e9a37be.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'arm_label': ARM_LABEL,
        'mode': 'smoke' if is_smoke else 'full',
        'smoke_cycles': args.smoke_cycles,
        'num_max_iters_used': num_max_iters,
        'out_dir': os.path.relpath(out_dir, REPO),
        'admm_report_path': os.path.relpath(report_path, REPO),
        'admm_cycles_run': report['cycles_run'],
        'admm_gross_operational_cost': report['gross_operational_cost'],
        'admm_converged_at_cycle': report.get('converged_at_cycle'),
        'reproduction': result_holder.get('reproduction'),
        'stopped_before_polish': result_holder.get('stopped_before_polish'),
        'polish': result_holder.get('polish'),
        'wall_clock_s': time.time() - started,
    }

    results_path = os.path.join(out_dir, 'polish_gap_results.json')
    _refuse_overwrite(results_path)
    with open(results_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S40-POLISH-GAP] wrote {results_path}')

    manifest = {}
    for root, _dirs, files in os.walk(out_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = CP._sha256_file(fpath)
    manifest_path = os.path.join(out_dir, 'manifest_sha256.json')
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S40-POLISH-GAP] wrote {manifest_path}')

    if not result_holder.get('reproduction', {}).get('reproduces'):
        print('[S40-POLISH-GAP] *** REPRODUCTION CHECK FAILED -- stopped before polish. '
              f'See {results_path}. ***')
        sys.exit(1)

    gate = (result_holder.get('polish') or {}).get('gate')
    if gate is None:
        print('[S40-POLISH-GAP] *** GATE NOT EVALUATED -- a polish block failed. '
              f'See {results_path}. ***')
        sys.exit(1)

    print(f"[S40-POLISH-GAP] GATE: relative={gate['relative_pct']:.6f}% "
          f"(threshold {gate['threshold_pct']}%) PASS={gate['pass']}")
    if not gate['pass']:
        sys.exit(1)


if __name__ == '__main__':
    main()
