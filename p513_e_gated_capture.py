"""
P5.13-E — gated capture and evaluation for the C3 activation (gate v2).

Frozen gate: data/SRP1/Results/P513D/frozen_c3_impact_gate_v2_afc863a1.json

Captures run under `SolveProfileGuard`: solves reached through
`shared_energy_storage_data._run_solver_attempt` are counted and allowed; any other
solve raises. The declared profile is 3 solves and 3 process launches per capture, and
the count must match EXACTLY in either direction.

    python p513_e_gated_capture.py capture --label v2_pre
    python p513_e_gated_capture.py evaluate --pre <state_v2_pre_nodeN.json> --post <state_v2_post_nodeN.json>
"""

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p513_c_param_move_gate import model_state, sha256_file, sha256_text, git_head  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P513D')
GATE = 'data/SRP1/Results/P513D/frozen_c3_impact_gate_v2_afc863a1.json'
PERMITTED = [('shared_energy_storage_data.py', '_run_solver_attempt')]
EXPECTED_SOLVES_PER_CAPTURE = 3

N, D, R = 10000, 0.80, 0.50
K = N * D / (-math.log(R))
TWO_K = repr(2 * K)
TWO_CL_NOM = '20000'
RATIO_EXPECTED = N / K            # cl_nom / k
RATIO_TOL = 0.05
MAX_RELATIVE_MOVE = 0.20

VALUE_MOVERS = {
    'es_avg_ch_dch_per_unit', 'es_degradation_per_unit', 'es_degradation_per_unit_cumul',
    'es_e_available_per_unit', 'es_pch_hat_agg', 'es_pch_hat_per_unit', 'es_pch_per_unit',
    'es_pdch_hat_agg', 'es_pdch_hat_per_unit', 'es_pdch_per_unit', 'es_soh_per_unit',
    'es_soh_per_unit_cumul', 'slack_es_ch_comp_per_unit', 'slack_es_pnet_down',
    'slack_es_pnet_up', 'slack_es_soh_per_unit_cumul_down', 'slack_es_soh_per_unit_cumul_up',
    'slack_es_soh_per_unit_down', 'slack_es_soh_per_unit_up',
}


def capture(label):
    os.makedirs(OUT_DIR, exist_ok=True)
    guard = SolveProfileGuard(PERMITTED, label=f'P5.13-E capture {label}').install()
    try:
        console = io.StringIO()
        with redirect_stdout(console):
            import shared_resources_planning as srp
            from shared_resources_planning import SharedResourcesPlanning
            planning = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
            planning.read_planning_problem()
            candidate = srp._build_positive_bootstrap_candidate(
                planning, planning.params.benders.positive_bootstrap)
            consensus_vars, _dual = srp.create_admm_variables(planning)
            esso_models, _results = srp.create_shared_energy_storage_model(
                planning.shared_ess_data, consensus_vars, candidate['investment'])
        profile_failures = guard.verify(EXPECTED_SOLVES_PER_CAPTURE)
    finally:
        guard.uninstall()

    shared_ess_data = planning.shared_ess_data
    manifest = {
        'stage': 'P5.13-E', 'label': label, 'gate_spec': GATE,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': git_head(), 'harness_sha256': sha256_file(os.path.abspath(__file__)),
        'solve_profile': {'declared': EXPECTED_SOLVES_PER_CAPTURE,
                          'observed': dict(guard.counts),
                          'permitted_sites': dict(guard.permitted_sites),
                          'failures': profile_failures},
        'ageing_constants': {}, 'state_sha256': {}, 'nl_sha256': {},
    }
    for year in shared_ess_data.years:
        for idx, ess in enumerate(shared_ess_data.shared_energy_storages[year]):
            manifest['ageing_constants'][f'{year}|{idx}|bus{ess.bus}'] = {
                name: repr(getattr(ess, name, None))
                for name in ('t_cal', 'cl_nom', 'dod_nom', 'soh_min', 'cl_eff')}

    tmpdir = tempfile.mkdtemp(prefix='p513e_')
    for node_id in sorted(shared_ess_data.active_distribution_network_nodes):
        state_text = json.dumps(model_state(esso_models[node_id]), sort_keys=True, indent=1)
        manifest['state_sha256'][str(node_id)] = sha256_text(state_text)
        with open(os.path.join(OUT_DIR, f'state_{label}_node{node_id}.json'), 'w') as handle:
            handle.write(state_text)
        nl_path = os.path.join(tmpdir, f'esso_node{node_id}.nl')
        esso_models[node_id].write(nl_path, io_options={'symbolic_solver_labels': False})
        manifest['nl_sha256'][str(node_id)] = sha256_file(nl_path)

    out = os.path.join(OUT_DIR, f'manifest_{label}.json')
    with open(out, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[P5.13-E] {label}: solve profile {guard.counts} failures={profile_failures}')
    print(f'[P5.13-E] {label}: written to {out}')
    if profile_failures:
        raise SystemExit(f'SOLVE PROFILE VIOLATION: {profile_failures}')
    return out


def _split(state):
    """Separate structure from post-solve values."""
    structure, values = {}, {}
    for section, content in state.items():
        if section == 'vars':
            for name, rows in content.items():
                structure[f'vars.{name}'] = {k: [v[0], v[1], v[2], v[4]] for k, v in rows.items()}
                values[name] = {k: v[3] for k, v in rows.items()}
        else:
            for name, content_item in content.items():
                structure[f'{section}.{name}'] = content_item
    return structure, values


def evaluate(pre_path, post_path):
    with open(pre_path) as handle:
        pre = json.load(handle)
    with open(post_path) as handle:
        post = json.load(handle)

    pre_struct, pre_vals = _split(pre)
    post_struct, post_vals = _split(post)

    struct_diff = sorted(k for k in set(pre_struct) | set(post_struct)
                         if pre_struct.get(k) != post_struct.get(k))
    value_diff = sorted(k for k in set(pre_vals) | set(post_vals)
                        if pre_vals.get(k) != post_vals.get(k))

    deg = 'constraints.energy_storage_capacity_degradation'
    pre_rows = pre['constraints']['energy_storage_capacity_degradation']['rows']
    post_rows = post['constraints']['energy_storage_capacity_degradation']['rows']
    differing_rows = [r for r in pre_rows if pre_rows[r] != post_rows.get(r)]

    ratios, worst_move, worst_component = [], 0.0, None
    for name in value_diff:
        for key, pre_value in pre_vals[name].items():
            post_value = post_vals[name].get(key)
            try:
                a, b = float(pre_value), float(post_value)
            except (TypeError, ValueError):
                continue
            if abs(a) > 1e-12:
                move = abs(b - a) / abs(a)
                if move > worst_move:
                    worst_move, worst_component = move, f'{name}[{key}]'
                if name == 'es_degradation_per_unit':
                    ratios.append(b / a)
    ratio = sum(ratios) / len(ratios) if ratios else None

    failures = []
    if struct_diff != [deg]:
        failures.append(f'E1b structure: differing components {struct_diff} != [{deg}]')
    if len(differing_rows) != 6:
        failures.append(f'E1b: {len(differing_rows)} degradation rows differ, expected 6')
    unexpected = [v for v in value_diff if v not in VALUE_MOVERS]
    if unexpected:
        failures.append(f'E1c values: unexpected components moved: {unexpected}')
    if not all(TWO_K in post_rows[r] for r in differing_rows):
        failures.append('E2: a differing row does not carry 2k')
    if any(TWO_CL_NOM in row for row in post_rows.values()):
        failures.append('E2: a post row still carries 2*cl_nom')
    if repr(K) != '11541.560327111707':
        failures.append('E3: k does not reproduce the authorized value')
    if ratio is None or abs(ratio - RATIO_EXPECTED) > RATIO_TOL:
        failures.append(f'E1c ratio: es_degradation_per_unit ratio {ratio} outside '
                        f'{RATIO_EXPECTED:.4f} +/- {RATIO_TOL}')
    if worst_move > MAX_RELATIVE_MOVE:
        failures.append(f'E1c move: {worst_component} moved {worst_move:.3%} > {MAX_RELATIVE_MOVE:.0%}')

    verdict = {
        'stage': 'P5.13-E', 'gate_spec': GATE,
        'pre_file': os.path.basename(pre_path), 'post_file': os.path.basename(post_path),
        'calibration': {'N': N, 'D': D, 'R': R, 'k': K, 'two_k': 2 * K},
        'E1b_structural_differences': struct_diff,
        'E1b_degradation_rows_differing': len(differing_rows),
        'E1b_sample_pre': pre_rows[differing_rows[0]] if differing_rows else None,
        'E1b_sample_post': post_rows[differing_rows[0]] if differing_rows else None,
        'E1c_components_with_moved_values': value_diff,
        'E1c_unexpected_movers': unexpected,
        'E1c_degradation_ratio_observed': ratio,
        'E1c_degradation_ratio_expected': RATIO_EXPECTED,
        'E1c_worst_relative_move': {'component': worst_component, 'relative': worst_move},
        'failures': failures,
        'verdict': 'PASS' if not failures else 'FAIL',
    }
    out = os.path.join(OUT_DIR, 'c3_impact_verdict_v2.json')
    with open(out, 'w') as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)
    print(json.dumps(verdict, indent=2)[:2400])
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--label', required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--pre', required=True)
    ev.add_argument('--post', required=True)
    args = parser.parse_args()
    if args.mode == 'capture':
        capture(args.label)
        return 0
    return evaluate(args.pre, args.post)


if __name__ == '__main__':
    sys.exit(main())
