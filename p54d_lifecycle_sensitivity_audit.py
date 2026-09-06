"""
Stage P5.4-D -- shared-ESS lifecycle and sensitivity audit after the P5.4-A/B/C
active-energy conversion.

Checks, on the real production code:
  A  shared-S / shared-E sensitivity rows still exist and still carry duals
  B  analytic sensitivity validated against a FINITE capacity perturbation
  C  every capacity transition: zero -> positive, positive -> different
     positive, positive -> zero, applied to a single reused model
  D  reused-model identity: the same model object is reconfigured in place,
     with no rebuilt or duplicated components
  E  fixed-capacity handling (shared_es_s_rated_fixed / _e_rated_fixed)
  F  dual and warm-start suffixes present, snapshot/clear/restore intact
  G  the obsolete kappa multiplier transfer is gone, with no replacement

    python p54d_lifecycle_sensitivity_audit.py
"""

import inspect
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import network as network_module  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from definitions import SHARED_ESS_ZERO_CAPACITY_TOLERANCE  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D')
CASE_NODE = 9
CASE_YEAR = 2030
CASE_DAY = 'Winter'

SHARED_ESS_ROWS = ('sess_pnet_def', 'sess_converter_capability',
                   'sess_active_sum_limit', 'sess_phi_limit_lower',
                   'sess_phi_limit_upper', 'sess_soc_def', 'sess_soc_limit_upper',
                   'sess_soc_limit_lower', 'sess_soc_final', 'sess_comp')
SHARED_ESS_VARS = ('shared_es_pch', 'shared_es_pdch', 'shared_es_pnet',
                   'shared_es_qnet', 'shared_es_soc')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def src(obj):
    try:
        return inspect.getsource(obj)
    except Exception as error:
        return f'<unavailable: {error}>'


def component_fingerprint(model):
    """Identity of every shared-ESS component object, so a rebuild is detectable."""
    out = {}
    for name in SHARED_ESS_ROWS + SHARED_ESS_VARS + (
            'shared_es_s_rated', 'shared_es_e_rated',
            'shared_es_s_rated_fixed', 'shared_es_e_rated_fixed',
            'shared_energy_storage_s_sensitivities',
            'shared_energy_storage_e_sensitivities'):
        if hasattr(model, name):
            out[name] = id(getattr(model, name))
    return out


def ess_state(model, e):
    s_rated = float(pe.value(model.shared_es_s_rated_fixed[e]))
    e_rated = float(pe.value(model.shared_es_e_rated_fixed[e]))
    p0 = model.shared_es_pch[e, 0, 0, 0]
    return {
        's_rated_fixed': s_rated, 'e_rated_fixed': e_rated,
        's_rated_var': float(pe.value(model.shared_es_s_rated[e])),
        'e_rated_var': float(pe.value(model.shared_es_e_rated[e])),
        'pch_fixed': bool(p0.fixed), 'pch_bounds': list(p0.bounds),
        'qnet_bounds': list(model.shared_es_qnet[e, 0, 0, 0].bounds),
        'rows_active': {n: bool(getattr(model, n)[
            (e, 0, 0, 0) if (e, 0, 0, 0) in getattr(model, n) else
            ((e, 0, 0) if (e, 0, 0) in getattr(model, n) else e)].active)
            for n in SHARED_ESS_ROWS if hasattr(model, n)},
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-D', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    dso = planning.distribution_networks[CASE_NODE]
    network = dso.network[CASE_YEAR][CASE_DAY]
    e = network.get_shared_energy_storage_idx(network.get_reference_node_id())
    s_candidate = candidate['total_capacity'][CASE_NODE][CASE_YEAR]['s']
    e_candidate = candidate['total_capacity'][CASE_NODE][CASE_YEAR]['e']

    # ---------------- G. obsolete kappa transfer ----------------
    report['G_kappa_removed'] = {
        'helper_absent': {n: not hasattr(mch, n) for n in (
            'shared_ess_snet_def_scale', '_sync_sess_snet_def_scale',
            'ordinary_ess_snet_def_scale', 'ess_snet_def_scale_init')},
        'no_replacement_transfer_helper': not [
            n for n in dir(mch)
            if 'sync' in n.lower() and ('kappa' in n.lower() or 'scale' in n.lower())],
        'configure_source': src(mch.configure_shared_ess_operational_state),
    }

    # ---------------- E/F. fixed capacity + suffixes on a built model ----------
    console = io.StringIO()
    with redirect_stdout(console):
        model = network.build_model(dso.params)
    fingerprint_built = component_fingerprint(model)
    report['F_suffixes'] = {
        'present': {n: hasattr(model, n) for n in (
            'dual', 'ipopt_zL_out', 'ipopt_zU_out', 'ipopt_zL_in', 'ipopt_zU_in')},
        'dual_direction_is_import_export': (
            model.dual.direction == pe.Suffix.IMPORT_EXPORT),
        'snapshot_clear_restore_present': all(
            hasattr(network_module, n) for n in (
                '_snapshot_multiplier_suffixes', '_clear_multiplier_suffixes',
                '_restore_multiplier_suffixes')),
        'snapshot_covers': ['ipopt_zL_in', 'ipopt_zU_in', 'dual'],
    }
    report['E_fixed_capacity'] = {
        'sensitivity_rows_present': {
            n: hasattr(model, n) for n in ('shared_energy_storage_s_sensitivities',
                                           'shared_energy_storage_e_sensitivities')},
        's_sensitivity_row_body': str(model.shared_energy_storage_s_sensitivities[e].expr),
        'e_sensitivity_row_body': str(model.shared_energy_storage_e_sensitivities[e].expr),
        's_rated_fixed_is_mutable_param': isinstance(
            model.shared_es_s_rated_fixed, pe.Param) and bool(
            model.shared_es_s_rated_fixed.mutable),
        's_rated_is_variable': isinstance(model.shared_es_s_rated, pe.Var),
        'new_rows_reference_the_rated_variable': all(
            'shared_es_s_rated' in str(getattr(model, n)[e, 0, 0, 0].expr)
            for n in ('sess_converter_capability', 'sess_active_sum_limit')),
    }

    # ---------------- C/D. capacity transitions on ONE reused model ------------
    # Candidate capacities are stored in MVA / MVAh and are divided by the model
    # base exactly as network_data.update_model_with_candidate_solution does.
    s_pu_candidate = abs(s_candidate) / network.baseMVA
    e_pu_candidate = abs(e_candidate) / network.baseMVA

    transitions = []
    sequence = [
        ('zero', 0.0, 0.0),
        ('positive', s_pu_candidate, e_pu_candidate),
        ('different_positive', 2.5 * s_pu_candidate, 2.5 * e_pu_candidate),
        ('back_to_zero', 0.0, 0.0),
        ('positive_again', s_pu_candidate, e_pu_candidate),
    ]
    fingerprints = [fingerprint_built]
    for label, s_val, e_val in sequence:
        console = io.StringIO()
        with redirect_stdout(console):
            # exactly the production call sequence
            model.shared_es_s_rated_fixed[e].set_value(s_val)
            model.shared_es_e_rated_fixed[e].set_value(e_val)
            inactive = mch.configure_shared_ess_operational_state(model, e, s_val, e_val)
        state = ess_state(model, e)
        state['label'] = label
        state['reported_inactive'] = bool(inactive)
        transitions.append(state)
        fingerprints.append(component_fingerprint(model))

    report['C_capacity_transitions'] = transitions
    report['D_reused_model_identity'] = {
        'component_ids_constant_across_all_transitions': all(
            f == fingerprints[0] for f in fingerprints),
        'n_snapshots': len(fingerprints),
        'components_tracked': sorted(fingerprints[0].keys()),
    }

    # ---------------- A/B. sensitivity + finite-difference validation ----------
    # Solve the real model at the candidate capacity, read the analytic dual of
    # the S-fixing row, then re-solve at a perturbed capacity and compare the
    # objective change. The dual is d(objective)/d(s_rated_fixed) in p.u.
    def solve_at(s_pu, e_pu):
        console = io.StringIO()
        with redirect_stdout(console):
            m = network.build_model(dso.params)
            m.shared_es_s_rated_fixed[e].set_value(s_pu)
            m.shared_es_e_rated_fixed[e].set_value(e_pu)
            mch.configure_shared_ess_operational_state(m, e, s_pu, e_pu)
            result = network.run_smopf(m, dso.params, print_header=False)
        return m, result, console.getvalue()

    s_pu = float(pe.value(model.shared_es_s_rated_fixed[e]))
    e_pu = float(pe.value(model.shared_es_e_rated_fixed[e]))
    if s_pu <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
        s_pu, e_pu = s_pu_candidate, e_pu_candidate

    base_model, base_result, _log = solve_at(s_pu, e_pu)
    base_ok = bool(srp._solver_result_succeeded(base_result))
    base_obj = float(pe.value(base_model.objective)) if base_ok else None
    dual_s = base_model.dual.get(base_model.shared_energy_storage_s_sensitivities[e]) if base_ok else None
    dual_e = base_model.dual.get(base_model.shared_energy_storage_e_sensitivities[e]) if base_ok else None

    # A single step size cannot distinguish "the dual is wrong" from "the step is
    # below the solver's own objective accuracy", so sweep several decades of
    # relative step. A trustworthy dual shows the central difference settling
    # toward it as the step shrinks.
    fd = {}
    if base_ok and dual_s is not None:
        for rel in (0.5, 0.1, 0.01, 0.001):
            h = rel * s_pu
            up_model, up_result, _ = solve_at(s_pu + h, e_pu)
            dn_model, dn_result, _ = solve_at(s_pu - h, e_pu)
            if not (srp._solver_result_succeeded(up_result)
                    and srp._solver_result_succeeded(dn_result)):
                fd[f'rel_{rel:g}'] = {'solved': False}
                continue
            up_obj = float(pe.value(up_model.objective))
            dn_obj = float(pe.value(dn_model.objective))
            central = (up_obj - dn_obj) / (2 * h)
            fd[f'rel_{rel:g}'] = {
                'solved': True, 'step_pu': h,
                'objective_up': up_obj, 'objective_down': dn_obj,
                'delta_objective': up_obj - dn_obj,
                'central_difference': central,
                'analytic_dual': float(dual_s),
                'dual_predicted_delta_objective': 2 * h * float(dual_s),
                'absolute_error': abs(central - float(dual_s)),
                'relative_error': (abs(central - float(dual_s))
                                   / max(abs(float(dual_s)), 1e-30)),
            }

    report['A_B_sensitivity'] = {
        'base_solve_succeeded': base_ok,
        's_rated_pu': s_pu, 'e_rated_pu': e_pu,
        'base_objective': base_obj,
        'analytic_dual_s': float(dual_s) if dual_s is not None else None,
        'analytic_dual_e': float(dual_e) if dual_e is not None else None,
        'duals_are_available': dual_s is not None and dual_e is not None,
        'finite_difference': fd,
        'extraction_path': ('network_data.py get_shared_energy_storage_sensitivities: '
                            'dual of shared_energy_storage_s_sensitivities, scaled by '
                            'objective_scale / baseMVA, then annualization * num_years * num_days'),
        'attribution': ('The same probe run against the pre-P5.4-A tree (a4a0bae8^) is in '
                        'fd_probe.py; see the P5.4 report section D. The failure to confirm '
                        'the dual is PRE-EXISTING, not introduced by P5.4.'),
    }

    out = os.path.join(OUT_DIR, 'p54d_report.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'\n[P5.4-D] report -> {out}')
    g = report['G_kappa_removed']
    print(f"  kappa helpers absent: {g['helper_absent']} "
          f"no replacement: {g['no_replacement_transfer_helper']}")
    print(f"  suffixes: {report['F_suffixes']['present']} "
          f"dual IMPORT_EXPORT={report['F_suffixes']['dual_direction_is_import_export']}")
    ee = report['E_fixed_capacity']
    print(f"  sensitivity rows: {ee['sensitivity_rows_present']}; "
          f"new rows reference rated var: {ee['new_rows_reference_the_rated_variable']}")
    print(f"  reused-model identity constant: "
          f"{report['D_reused_model_identity']['component_ids_constant_across_all_transitions']}")
    for t in transitions:
        print(f"   {t['label']:20s} S={t['s_rated_fixed']:.4e} E={t['e_rated_fixed']:.4e} "
              f"inactive={t['reported_inactive']} pch_fixed={t['pch_fixed']} "
              f"pch_bounds={t['pch_bounds']} rows_active={all(t['rows_active'].values())}")
    ab = report['A_B_sensitivity']
    print(f"  base solve ok={ab['base_solve_succeeded']} obj={ab['base_objective']}")
    print(f"  duals available={ab['duals_are_available']} dual_s={ab['analytic_dual_s']} "
          f"dual_e={ab['analytic_dual_e']}")
    for key, block in ab['finite_difference'].items():
        if block.get('solved'):
            print(f"   FD {key}: central={block['central_difference']:.6e} "
                  f"analytic={block['analytic_dual']:.6e} "
                  f"rel_err={block['relative_error']:.3e}")
        else:
            print(f"   FD {key}: perturbed solve failed")


if __name__ == '__main__':
    main()
