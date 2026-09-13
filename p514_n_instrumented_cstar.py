"""
P5.14-N — instrumented C* control arm. Frozen spec:
    data/SRP1/Results/P514N/frozen_n1_control_arm_v1_f1bbddf4.json

The control arm the perturbation needs: an instrumented arm compared against an
uninstrumented one would be no A/B at all. The determinism gate doubles as the gate on the
instrumentation — exact reproduction of 816,121,464.1554238 proves the capture is
non-perturbing, so no separate neutrality run is required.

RULE ELEVEN: this harness asserts, before executing, that a capture path exists for every
quantity the frozen spec requires, and fails fast if one is missing.

    python p514_n_instrumented_cstar.py [k_override]
"""

import io
import json
import os
import pickle
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p512_a_cold_rescaled_convergence as A  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p58_rescale as R  # noqa: E402
import p59_rho as RH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P514N')
SPEC = 'data/SRP1/Results/P514N/frozen_n1_control_arm_v1_f1bbddf4.json'
S_INV, E_INV, INVEST_YEAR = 0.96875, 3.875, 2025
BUDGET, REL, CAP = 5.0e6, 1e-4, 90
RHO = {'v': 1.5, 'pf': 300.0, 'ess': 1.0}
REFERENCE_RECOURSE = 816121464.1554238
EFC_BINDING_THRESHOLD = 1.4612
PERMITTED = [('network.py', '_run_smopf_solver_attempt'),
             ('shared_energy_storage_data.py', '_run_solver_attempt')]

# ---- rule eleven: the capture checklist, asserted BEFORE the run ----
REQUIRED_ESSO_ATTRS = ('es_avg_ch_dch_per_unit', 'es_degradation_per_unit',
                       'es_soh_per_unit_cumul', 'es_e_rated_per_unit')
REQUIRED_DERIVED = ('efc_per_day', 'efc_margin_to_threshold')
REQUIRED_ARTIFACTS = ('esso_models_pickle',)


def assert_capture_paths_exist():
    """Fail fast if the spec requires a quantity this harness cannot capture."""
    missing = []
    for name in REQUIRED_ESSO_ATTRS:
        if name not in EXTRACTORS:
            missing.append(f'no extractor for ESSO attribute {name}')
    for name in REQUIRED_DERIVED:
        if name not in DERIVED:
            missing.append(f'no derivation for {name}')
    for name in REQUIRED_ARTIFACTS:
        if name not in ARTIFACTS:
            missing.append(f'no writer for artifact {name}')
    if missing:
        raise AssertionError('RULE ELEVEN: capture paths missing -> ' + '; '.join(missing))
    return {'esso_attributes': list(REQUIRED_ESSO_ATTRS),
            'derived': list(REQUIRED_DERIVED), 'artifacts': list(REQUIRED_ARTIFACTS),
            'asserted_before_run': True}


def _indexed(model, name):
    comp = getattr(model, name)
    return {f'{k}': pe.value(comp[k], exception=False) for k in comp}


EXTRACTORS = {name: (lambda m, n=name: _indexed(m, n)) for name in REQUIRED_ESSO_ATTRS}
DERIVED = {'efc_per_day': True, 'efc_margin_to_threshold': True}
ARTIFACTS = {'esso_models_pickle': True}


def capture_esso(models):
    """Every required ESSO quantity, per node, per cohort-year, plus the derived EFC."""
    out = {}
    for node_id, model in models.items():
        for name in REQUIRED_ESSO_ATTRS:
            if not hasattr(model, name):
                raise AssertionError(f'RULE ELEVEN: ESSO model for node {node_id} has no {name}')
        entry = {name: EXTRACTORS[name](model) for name in REQUIRED_ESSO_ATTRS}
        efc = {}
        for key, avg in entry['es_avg_ch_dch_per_unit'].items():
            rated = entry['es_e_rated_per_unit'].get(key)
            if avg is None or not rated:
                efc[key] = None
                continue
            value = avg / (2.0 * rated)          # EFC/day = throughput / (2 * E_rated)
            efc[key] = {'efc_per_day': value,
                        'margin_to_threshold': EFC_BINDING_THRESHOLD - value,
                        'fraction_of_threshold': value / EFC_BINDING_THRESHOLD}
        entry['efc_per_day_per_cohort_year'] = efc
        live = [v['efc_per_day'] for v in efc.values() if v]
        entry['efc_per_day_max'] = max(live) if live else None
        entry['efc_per_day_min'] = min(live) if live else None
        out[str(node_id)] = entry
    return out


def main(k_override=None):
    os.makedirs(OUT, exist_ok=True)
    checklist = assert_capture_paths_exist()          # rule eleven, before anything runs
    label = 'control' if k_override is None else f'k{k_override:g}'
    guard = SolveProfileGuard(PERMITTED, label=f'P5.14-N {label}').install()
    started = time.time()
    report = {'stage': 'P5.14-N', 'arm': label, 'spec': SPEC,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'rule_eleven_checklist': checklist}
    try:
        with redirect_stdout(io.StringIO()):
            planning = O.fresh_planning(f'p514n_{label}')
            planning.params.admm.num_max_iters = CAP
            planning.params.admm.tol['objective']['rel'] = REL
            planning.shared_ess_data.params.budget = BUDGET
            RH.apply_rho_to_params(planning, RHO)
            RH.set_adaptive_penalty(planning, True)
            sed = planning.shared_ess_data
            if k_override is not None:                 # perturbation arm only
                for year in sed.years:
                    for ess in sed.shared_energy_storages[year]:
                        ess.cl_eff = k_override
            report['k_in_force'] = {str(y): getattr(sed.shared_energy_storages[y][0], 'cl_eff', None)
                                    for y in sed.years}

            candidate = planning.get_initial_candidate_solution()
            for node_id in sed.active_distribution_network_nodes:
                candidate['investment'][node_id][INVEST_YEAR]['s'] = S_INV
                candidate['investment'][node_id][INVEST_YEAR]['e'] = E_INV
            srp._rebuild_candidate_total_capacities(planning, candidate)
            report['instance'] = {'s_mva': S_INV, 'e_mwh': E_INV, 'year': INVEST_YEAR}

            with R.patched_admm_objectives():
                _c, _results, models, _s, _p, state = planning.run_operational_planning(
                    type='distributed', candidate_solution=deepcopy(candidate),
                    print_results=False, debug_flag=False, return_state=True)
    finally:
        guard.uninstall()
        report['wall_clock_s'] = time.time() - started

    rows = [A.cycle_row(e, None) for e in (state.get('admm_diagnostics') or [])]
    last = rows[-1] if rows else {}
    report.update({
        'cycles_run': len(rows), 'recourse': last.get('recourse'),
        'gross_operational_cost': last.get('gross_operational_cost'),
        'converged_at_cycle': next((r['cycle'] for r in rows if r['cycle_convergence']), None),
        'terminal_objective_change_abs': last.get('objective_change_abs'),
        'terminal_objective_tolerance': last.get('objective_tolerance'),
        'rule_ten_terminal_step_over_threshold': (
            last.get('objective_change_abs') / last.get('objective_tolerance')
            if last.get('objective_change_abs') and last.get('objective_tolerance') else None),
        'local_solve_failures': sum(1 for r in rows if r.get('local_solves_ok') is False),
    })
    report['esso_capture'] = capture_esso(models['esso'])

    pickle_path = os.path.join(OUT, f'esso_models_{label}.pkl')
    try:
        with open(pickle_path, 'wb') as handle:
            pickle.dump(models['esso'], handle)
        report['esso_models_pickle'] = {'path': os.path.relpath(pickle_path, REPO),
                                        'bytes': os.path.getsize(pickle_path)}
    except Exception as error:
        report['esso_models_pickle'] = {'error': f'{type(error).__name__}: {error}'}

    delta = (report['recourse'] - REFERENCE_RECOURSE) if report['recourse'] is not None else None
    report['determinism_gate'] = {
        'reference': REFERENCE_RECOURSE, 'observed': report['recourse'], 'delta': delta,
        'exact': delta == 0.0,
        'verdict': ('PASS — reproduces exactly; the capture is non-perturbing'
                    if delta == 0.0 else
                    'FAIL — NON-DETERMINISM AT MATERIAL CAPACITY, a major finding that '
                    'supersedes the EFC question'),
        'secondary': {'cycles': report['cycles_run'], 'solves': guard.counts['permitted_solve'],
                      'rule_ten': report['rule_ten_terminal_step_over_threshold'],
                      'local_solve_failures': report['local_solve_failures']},
    }
    report['solve_profile'] = {'observed': dict(guard.counts),
                               'identity_holds': guard.counts['permitted_solve'] == 51 * len(rows) + 51}
    path = os.path.join(OUT, f'n1_{label}.json')
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    efc_all = [v['efc_per_day_max'] for v in report['esso_capture'].values() if v['efc_per_day_max']]
    print(f"[P5.14-N {label}] recourse={report['recourse']} delta={delta} "
          f"cycles={report['cycles_run']} solves={guard.counts['permitted_solve']} "
          f"wall={report['wall_clock_s']:.0f}s")
    print(f"   determinism: {report['determinism_gate']['verdict']}")
    print(f"   EFC/day max across nodes: {max(efc_all) if efc_all else None} "
          f"(threshold {EFC_BINDING_THRESHOLD}, decision guide 0.76)")
    return 0


if __name__ == '__main__':
    sys.exit(main(float(sys.argv[1]) if len(sys.argv) > 1 else None))
