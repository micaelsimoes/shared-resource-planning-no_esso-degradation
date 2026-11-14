"""
Stage P5.2-A2 -- narrow-band epsilon sensitivity on the outstanding
primary-path case.

Target: `case33_2 / node 7 / 2030 / Summer`, exact P5 positive-bootstrap cold
initialization state. In P5.2-A this case solved on the primary exact-Hessian
path under the hard equality, but under the narrow band with eps_rel = 1e-5 its
primary solve failed with `internalSolverError` and only the limited-memory
recovery converged it.

Question: does a modestly wider band restore a clean PRIMARY exact-Hessian
solve? Tested eps_rel values: 1e-5, 3e-5, 1e-4 (no others).

Everything else is held fixed: kappa = 1/S_rated (untouched production), the
same ranged formulation applied in place on the existing sess_snet_def
ConstraintData objects, same primal initial state, bounds, objective,
sess_comp, SOC, IPOPT options, MA97, exact-Hessian primary, recovery policy and
`from_warm_start=False`.

Unlike the earlier scripts this one splits the IPOPT log into per-attempt
blocks, so primary and recovery iterations/metrics are reported separately.

Diagnostic only. No production file is modified. The full 51-solve
initialization is NOT run.

    python p52a2_epsilon_sensitivity.py
"""

import io
import json
import os
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p51_small_capacity_scaling_diagnostic import (  # noqa: E402  (reuse)
    build_dso_initialization_models,
    clear_logs,
    primal_start_signature,
    read_logs,
    shared_ess_state,
    structural_signature,
)
from p52a_narrow_band_diagnostic import (  # noqa: E402  (reuse)
    apply_narrow_band,
    constraint_form_summary,
    physical_metrics,
)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P52A2')

TARGET = ('n7_2030_Summer', 'case33_2', 7, 2030, 'Summer')
CONTROLS = [
    ('n5_2025_Winter', 'case33_1', 5, 2025, 'Winter'),
    ('n7_2025_Winter', 'case33_2', 7, 2025, 'Winter'),
    ('n9_2030_Summer', 'case33_3', 9, 2030, 'Summer'),
]
EPSILONS = [1e-5, 3e-5, 1e-4]

BLOCK_KEYS = {
    'iterations': re.compile(r'Number of Iterations\.*:\s*(\d+)'),
    'objective': re.compile(r'Objective\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'primal_infeasibility': re.compile(r'Constraint violation\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'dual_infeasibility': re.compile(r'Dual infeasibility\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'complementarity': re.compile(r'Complementarity\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'overall_nlp_error': re.compile(r'Overall NLP error\.*:\s*([0-9eE+\-.]+)\s+([0-9eE+\-.]+)'),
    'exit_status': re.compile(r'EXIT: (.+?)\s*$', re.MULTILINE),
}


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


SUMMARY_ANCHOR = re.compile(r'Number of Iterations\.+:\s*(\d+)')


def parse_ipopt_blocks(text):
    """Return one dict per IPOPT attempt, in order (primary first, then any
    recovery attempt appended to the same log by file_append=yes).

    IMPORTANT: these logs are written with file_print_level=6, so the same
    metric names also appear on ~1000 per-iteration diagnostic lines. Parsing
    must therefore be anchored on the final-summary block, which is the only
    place `Number of Iterations....:` (dots + colon) appears. Everything for one
    attempt is read from the window that starts at that anchor.
    """
    anchors = list(SUMMARY_ANCHOR.finditer(text))
    blocks = []
    for i, anchor in enumerate(anchors):
        start = anchor.start()
        end = anchors[i + 1].start() if i + 1 < len(anchors) else len(text)
        window = text[start:end]
        block = {'iterations': anchor.group(1)}
        for key, pattern in BLOCK_KEYS.items():
            if key == 'iterations':
                continue
            match = pattern.search(window)
            if match:
                block[key] = match.groups() if len(match.groups()) > 1 else match.group(1)
        blocks.append(block)
    return blocks


def solve_with_split(distribution_network, model, year, day, log_prefix):
    network = distribution_network.network[year][day]
    params = distribution_network.params
    clear_logs(network, log_prefix)
    buffer = io.StringIO()
    started = time.time()
    with redirect_stdout(buffer):
        result = network.run_smopf(model, params, from_warm_start=False, print_header=False)
    runtime = time.time() - started
    console = buffer.getvalue()
    blocks = parse_ipopt_blocks(read_logs(network, log_prefix))

    primary_failed = 'Network primary solve did not converge' in console
    recovery_attempted = 'Retrying network solve once' in console
    recovery_succeeded = 'Network recovery solve succeeded' in console
    final_succeeded = bool(srp._solver_result_succeeded(result)) if result is not None else False

    return {
        'final_status': str(getattr(result.solver, 'status', None)) if result is not None else None,
        'final_termination': str(getattr(result.solver, 'termination_condition', None)) if result is not None else None,
        'final_succeeded': final_succeeded,
        'primary_failed': primary_failed,
        'recovery_attempted': recovery_attempted,
        'recovery_succeeded': recovery_succeeded,
        'clean_primary_success': final_succeeded and not recovery_attempted,
        'runtime_s': runtime,
        'n_ipopt_attempts': len(blocks),
        'primary_block': blocks[0] if blocks else None,
        'recovery_block': blocks[1] if len(blocks) > 1 else None,
        'console_tail': console[-1500:],
    }


def run_band_case(distribution_network, dso_models, name, year, day, epsilon_rel):
    network = distribution_network.network[year][day]
    idx = network.get_shared_energy_storage_idx(network.get_reference_node_id())
    base_model = dso_models[year][day]
    base_mva = network.baseMVA
    s_rated_pu = float(pe.value(base_model.shared_es_s_rated_fixed[idx]))
    kappa = float(pe.value(base_model.sess_snet_def_kappa[idx]))

    reference = {'primal': primal_start_signature(base_model),
                 'structure': structural_signature(base_model, idx),
                 'objective_at_start': float(pe.value(base_model.objective))}

    model = base_model.clone()
    band = apply_narrow_band(model, idx, epsilon_rel, s_rated_pu)
    pre = {'primal': primal_start_signature(model),
           'structure': structural_signature(model, idx),
           'objective_at_start': float(pe.value(model.objective)),
           'forms': constraint_form_summary(model, idx),
           'shared_ess': shared_ess_state(model, idx)}
    discipline = {
        'primal_start_identical': pre['primal'] == reference['primal'],
        'objective_at_start_identical': abs(pre['objective_at_start']
                                            - reference['objective_at_start']) < 1e-9,
        'index_tuples_unchanged': (pre['structure']['sess_snet_def_index_tuples_for_idx']
                                   == reference['structure']['sess_snet_def_index_tuples_for_idx']),
        'total_constraint_data_unchanged': (pre['structure']['total_constraint_data']
                                            == reference['structure']['total_constraint_data']),
        'no_replacement_component': not pre['structure']['has_replacement_component'],
        'rows_all_ranged': (pre['forms']['row_forms']['ranged'] > 0
                            and pre['forms']['row_forms']['equality'] == 0),
        'kappa_is_production': abs(pre['shared_ess']['kappa'] - kappa) < 1e-12,
        'sess_comp_rows_unchanged': (pre['structure']['sess_comp_rows']
                                     == reference['structure']['sess_comp_rows']),
    }

    solve = solve_with_split(distribution_network, model, year, day, f'optim_log_{name}')
    epsilon_abs = epsilon_rel * (s_rated_pu ** 2)
    physical = physical_metrics(model, idx, s_rated_pu, base_mva, epsilon_abs)
    if physical.get('available'):
        physical['band_utilization'] = (
            physical['max_abs_g_normalized_by_s_rated_sq'] / epsilon_rel)

    return {'network': name, 'year': year, 'day': day,
            'epsilon_rel': epsilon_rel, 's_rated_pu': s_rated_pu,
            's_rated_MVA': s_rated_pu * base_mva, 'kappa': kappa,
            'band': band, 'discipline_checks': discipline,
            'solve': solve, 'physical': physical}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.2-A2', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'epsilons_tested': EPSILONS,
              'target': {'network': TARGET[1], 'node': TARGET[2],
                         'year': TARGET[3], 'day': TARGET[4]}}

    quiet = io.StringIO()
    with redirect_stdout(quiet):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
    report['scenario_checksum'] = (
        re.findall(r'Scenario checksum: (\S+)', quiet.getvalue()) or [None])[-1]

    needed = sorted({TARGET[2]} | {c[2] for c in CONTROLS})
    dso_models = {}
    with redirect_stdout(quiet):
        for node_id in needed:
            dso_models[node_id] = build_dso_initialization_models(
                planning.distribution_networks[node_id], candidate['total_capacity'])

    # ---- target case across the epsilon ladder ----
    label, name, node_id, year, day = TARGET
    target_results = {}
    for epsilon in EPSILONS:
        print(f'[P5.2-A2] target {label} eps={epsilon:g} ...', flush=True)
        target_results[f'{epsilon:g}'] = run_band_case(
            planning.distribution_networks[node_id], dso_models[node_id],
            name, year, day, epsilon)
    report['target_results'] = target_results

    clean = [e for e in EPSILONS
             if target_results[f'{e:g}']['solve']['clean_primary_success']]
    report['epsilons_with_clean_primary_success'] = [f'{e:g}' for e in clean]
    report['recommended_epsilon'] = f'{min(clean):g}' if clean else None

    # ---- controls, only for epsilons that gave a clean primary success ----
    control_results = {}
    for epsilon in clean:
        per_eps = {}
        for clabel, cname, cnode, cyear, cday in CONTROLS:
            print(f'[P5.2-A2] control {clabel} eps={epsilon:g} ...', flush=True)
            per_eps[clabel] = run_band_case(
                planning.distribution_networks[cnode], dso_models[cnode],
                cname, cyear, cday, epsilon)
        control_results[f'{epsilon:g}'] = per_eps
    report['control_results'] = control_results
    report['controls_all_clean_primary'] = {
        key: all(v['solve']['clean_primary_success'] for v in per_eps.values())
        for key, per_eps in control_results.items()}

    out_path = os.path.join(OUT_DIR, 'p52a2_report.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'\n[P5.2-A2] report -> {out_path}')
    print(f"[P5.2-A2] checksum {report['scenario_checksum']}")
    print(f"\nTARGET {label} (S={target_results[f'{EPSILONS[0]:g}']['s_rated_MVA']:.5f} MVA, "
          f"kappa={target_results[f'{EPSILONS[0]:g}']['kappa']:.1f})")
    for epsilon in EPSILONS:
        r = target_results[f'{epsilon:g}']
        s, ph = r['solve'], r['physical']
        pb = s['primary_block'] or {}
        rb = s['recovery_block'] or {}
        print(f"  eps={epsilon:g}: clean_primary={s['clean_primary_success']} "
              f"final={s['final_status']}/{s['final_termination']} "
              f"primary_iters={pb.get('iterations')} recovery_iters={rb.get('iterations')} "
              f"rt={s['runtime_s']:.1f}s")
        print(f"           band_util={ph.get('band_utilization', float('nan')):.3e} "
              f"max|g|/S^2={ph.get('max_abs_g_normalized_by_s_rated_sq', float('nan')):.3e} "
              f"dS/S={ph.get('max_delta_S_over_s_rated', float('nan')):.3e} "
              f"near10%={ph.get('n_periods_within_10pct_of_boundary')} "
              f"active={ph.get('n_periods_active_at_boundary')}")
    print(f"\nclean primary success at: {report['epsilons_with_clean_primary_success'] or 'NONE'}")
    print(f"recommended (smallest successful): {report['recommended_epsilon']}")
    for key, per_eps in control_results.items():
        print(f"\ncontrols at eps={key}: all_clean_primary={report['controls_all_clean_primary'][key]}")
        for clabel, r in per_eps.items():
            s, ph = r['solve'], r['physical']
            pb = s['primary_block'] or {}
            print(f"  {clabel:16s} clean_primary={s['clean_primary_success']} "
                  f"{s['final_status']}/{s['final_termination']} iters={pb.get('iterations')} "
                  f"band_util={ph.get('band_utilization', float('nan')):.3e} "
                  f"dS/S={ph.get('max_delta_S_over_s_rated', float('nan')):.3e}")


if __name__ == '__main__':
    main()
