"""P5.15 Step 3.0 (Addendum 9) - baseline determinism comparison. Zero solves.

Compares the Step 3.0 repeat (data/SRP1/Results/P515S30) with the new-baseline G1 re-run
(data/SRP1/Results/P515G1B) BITWISE on: cycle count / convergence / recourse / solve count;
the per-cycle trajectory (recourse, objective step, V/PF/ESS primal and dual residuals);
ESSO es_soh_per_unit_cumul, es_D_per_unit, es_avg_ch_dch_per_unit for every node; the measured
detector, terminal lg(mu) and spurious throughput per ESSO solve; and the network-failure
classes (by network, year, day, cycle). A pickle hash is NOT used (not byte-stable).

Writes data/SRP1/Results/P515S30/s30_determinism_comparison.json and exits 0 iff every
compared quantity is bitwise identical.
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G1B')
REP = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S30')
TRAJ_FIELDS = ('recourse', 'gross_operational_cost', 'objective_change_abs', 'objective_tolerance',
               'primal_v', 'primal_v_mean', 'dual_v', 'dual_v_mean',
               'primal_pf', 'primal_pf_mean', 'dual_pf', 'dual_pf_mean',
               'primal_ess', 'primal_ess_mean', 'dual_ess', 'dual_ess_mean',
               'local_solves_ok', 'cycle_convergence')
ESSO_FIELDS = ('es_soh_per_unit_cumul', 'es_D_per_unit', 'es_avg_ch_dch_per_unit')
LEAK_FIELDS = ('complementarity_ratio_max', 'lg_mu_terminal', 's_obj', 'spurious_throughput_measured')


def _one(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one file for {pattern}, found {hits}')
    return hits[0]


def _jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    ref = json.load(open(_one(os.path.join(REF, 'g_*.json'))))
    rep = json.load(open(_one(os.path.join(REP, 'g_*.json'))))
    diffs = []

    def record(where, a, b):
        if a != b:
            diffs.append({'where': where, 'reference': a, 'repeat': b})

    for key in ('cycles_run', 'converged_at_cycle', 'recourse', 'terminal_objective_change_abs',
                'rule_ten_terminal_step_over_threshold', 'local_solve_failures'):
        record(key, ref.get(key), rep.get(key))
    record('solve_profile.permitted_solve', ref['solve_profile']['observed']['permitted_solve'],
           rep['solve_profile']['observed']['permitted_solve'])

    traj_ref, traj_rep = ref['cycle_trajectory'], rep['cycle_trajectory']
    record('cycle_trajectory.length', len(traj_ref), len(traj_rep))
    trajectory_entries = 0
    for row_ref, row_rep in zip(traj_ref, traj_rep):
        for field in TRAJ_FIELDS:
            trajectory_entries += 1
            record(f"cycle {row_ref.get('cycle')}.{field}", row_ref.get(field), row_rep.get(field))

    esso_entries = 0
    for node in ref['esso_capture']:
        for field in ESSO_FIELDS:
            for index, value in ref['esso_capture'][node][field].items():
                esso_entries += 1
                record(f'node {node}.{field}[{index}]', value, rep['esso_capture'][node][field].get(index))

    leak_ref = {(r['node_id'], r['cycle']): r for r in _jsonl(_one(os.path.join(REF, 'leak_classification_*.jsonl')))}
    leak_rep = {(r['node_id'], r['cycle']): r for r in _jsonl(_one(os.path.join(REP, 'leak_classification_*.jsonl')))}
    record('leak.solve_keys', sorted(map(str, leak_ref)), sorted(map(str, leak_rep)))
    leak_entries = 0
    for key, row in leak_ref.items():
        other = leak_rep.get(key, {})
        for field in LEAK_FIELDS:
            leak_entries += 1
            record(f'leak {key}.{field}', row.get(field), other.get(field))

    def failures(root):
        rows = _jsonl(_one(os.path.join(root, 'network_failures_*.jsonl')))
        return sorted((r.get('network_name'), r.get('year'), r.get('day'), str(r.get('cycle')), r.get('class')) for r in rows)
    record('network_failures', failures(REF), failures(REP))

    out = {
        'stage': 'P5.15 Step 3.0 baseline determinism',
        'reference': os.path.relpath(REF, REPO), 'repeat': os.path.relpath(REP, REPO),
        'compared': {'trajectory_entries': trajectory_entries, 'esso_entries': esso_entries,
                     'leak_entries': leak_entries},
        'n_differences': len(diffs), 'first_differences': diffs[:20],
        'bitwise_identical': not diffs,
        'note': 'Pickle hashes are not compared; pickled Pyomo models are not byte-stable.',
    }
    path = os.path.join(REP, 's30_determinism_comparison.json')
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    with open(path, 'w') as handle:
        json.dump(out, handle, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ('compared', 'n_differences', 'bitwise_identical')}, indent=1))
    if diffs:
        print('first differences:', json.dumps(diffs[:5], default=str, indent=1))
    return 0 if not diffs else 1


if __name__ == '__main__':
    sys.exit(main())
