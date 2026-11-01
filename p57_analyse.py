"""
Stage P5.7 -- compare the branches collected by `p57_d1_chain.py`.

Answers the two questions the stage asks of the diagnostic set:

  * the objective decomposition of every branch, by cost family and by agent;
  * WHICH VARIABLES MOVE between branches -- dispatch, flexibility, voltage
    slacks, ESS schedules, interface variables -- ranked, plus the active-set
    differences that hypothesis C asks for.

Reads only persisted evidence; solves nothing.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p57_analyse.py
"""

import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p57_eval as E  # noqa: E402
import p57_fingerprint as FP  # noqa: E402

CHAIN_PATH = os.path.join(E.OUT_DIR, 'p57_d1_chain.json')
OUT_PATH = os.path.join(E.OUT_DIR, 'p57_branch_comparison.json')

# the six solutions P5.7 asks for, as (label, step, which fingerprint)
REQUESTED = [('ADMM', 1, 'admm'), ('polish K=0', 1, 'polished'),
             ('H2', 2, 'polished'), ('H4', 4, 'polished'),
             ('H8', 8, 'polished'), ('H12', 12, 'polished')]

# the families P5.7 names explicitly, grouped the way the stage asks for them
GROUPS = {
    'generation dispatch': ('pg', 'qg'),
    'flexibility': ('flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down'),
    'voltage slacks': ('slack_v_sqr_up', 'slack_v_sqr_down'),
    'other slacks': ('slack_node_balance_p_up', 'slack_node_balance_p_down',
                     'slack_node_balance_q_up', 'slack_node_balance_q_down',
                     'slack_flow_ij_sqr', 'slack_flow_ji_sqr',
                     'slack_shared_es_soc_final_up',
                     'slack_shared_es_soc_final_down'),
    'ESS schedules': ('shared_es_pch', 'shared_es_pdch', 'shared_es_pnet',
                      'shared_es_qnet', 'shared_es_soc'),
    'TSO/DSO interface': ('pc', 'qc'),
    'voltage state': ('e', 'f', 'vmag', 'vmag_sqr'),
    'branch flows / OLTC': ('pij', 'qij', 'pji', 'qji', 'r'),
}


def _archive(step, which):
    key = 'fingerprint_admm' if which == 'admm' else 'fingerprint_polished'
    entry = step.get(key)
    if not entry:
        return None
    return os.path.join(REPO_ROOT, entry['archive'])


def _decompose(fingerprint):
    """Cost families and per-agent totals of one fingerprinted solution."""
    families, agents = defaultdict(float), defaultdict(float)
    total = 0.0
    for key, block in fingerprint['blocks'].items():
        agent = key.split('|')[0]
        agents[agent] += block['weighted_base_objective']
        total += block['weighted_base_objective']
        for name, value in (block.get('cost_families') or {}).items():
            if value is not None:
                families[name] += value
    return {'total_weighted_base_objective': total,
            'by_cost_family': dict(families), 'by_agent': dict(agents)}


def main():
    if not os.path.exists(CHAIN_PATH):
        print(f'[P5.7] ABORTED: {CHAIN_PATH} not found -- run p57_d1_chain.py')
        sys.exit(1)
    with open(CHAIN_PATH) as handle:
        chain = json.load(handle)

    by_step = {s['j']: s for s in chain['steps']}
    available = [(label, j, which) for label, j, which in REQUESTED
                 if j in by_step and _archive(by_step[j], which)]
    missing = [label for label, j, which in REQUESTED
               if (label, j, which) not in available]

    report = {'stage': 'P5.7 branch comparison',
              'provenance': chain.get('provenance'),
              'fixed_candidate': chain.get('fixed_candidate'),
              'reproduces_p56d': chain.get('reproduces_p56d'),
              'reproduction': chain.get('reproduction'),
              'solutions_requested': [r[0] for r in REQUESTED],
              'solutions_missing': missing,
              'objective_decomposition': {}, 'objective_path': [],
              'pairwise': {}}

    # ------------------------------------------------------- decomposition --
    for label, j, which in available:
        step = by_step[j]
        key = 'fingerprint_admm' if which == 'admm' else 'fingerprint_polished'
        report['objective_decomposition'][label] = {
            'step': j, 'kind': which,
            **_decompose(step[key]),
            'total_planning_objective': (step['total_objective']
                                         if which == 'polished' else None),
            'net_operational_recourse': (step['net_operational_recourse']
                                         if which == 'polished' else None),
        }

    reference = report['objective_decomposition'].get('polish K=0')
    for label, entry in report['objective_decomposition'].items():
        base = reference['total_weighted_base_objective'] if reference else None
        report['objective_path'].append({
            'solution': label, 'step': entry['step'], 'kind': entry['kind'],
            'weighted_base_objective': entry['total_weighted_base_objective'],
            'delta_vs_polish_K0': (entry['total_weighted_base_objective'] - base)
            if base is not None else None,
            'total_planning_objective': entry['total_planning_objective']})

    # ----------------------------------------------------------- pairwise --
    pairs = []
    for i in range(len(available) - 1):
        pairs.append((available[i], available[i + 1]))
    if len(available) > 2:
        pairs.append((available[1], available[-1]))   # polish K=0 -> deepest
        pairs.append((available[0], available[-1]))   # ADMM       -> deepest

    for a, b in pairs:
        name = f'{a[0]} -> {b[0]}'
        comparison = FP.compare(_archive(by_step[a[1]], a[2]),
                                _archive(by_step[b[1]], b[2]))
        rolled = FP.rollup(comparison)
        grouped = {}
        for group, names in GROUPS.items():
            entries = {n: rolled['variables'][n] for n in names
                       if n in rolled['variables']}
            if not entries:
                continue
            grouped[group] = {
                'max_abs_delta': max(e['max_abs_delta'] for e in entries.values()),
                'total_l1_delta': sum(e['total_l1_delta'] for e in entries.values()),
                'total_moved_1e-6': sum(e['total_moved_1e-6']
                                        for e in entries.values()),
                'total_n': sum(e['total_n'] for e in entries.values()),
                'per_family': entries}
        active = {n: e for n, e in rolled['constraint_activity'].items()
                  if e['total_symmetric_difference'] > 0}
        report['pairwise'][name] = {
            'variables_by_group': grouped,
            'variables_ranked_by_l1': sorted(
                ((n, e['total_l1_delta'], e['max_abs_delta'], e['worst_block'])
                 for n, e in rolled['variables'].items()),
                key=lambda row: -row[1])[:20],
            'active_set_changes': dict(sorted(
                active.items(),
                key=lambda kv: -kv[1]['total_symmetric_difference'])),
            'total_active_set_symmetric_difference': sum(
                e['total_symmetric_difference']
                for e in rolled['constraint_activity'].values()),
            'total_bound_activity_changes': sum(
                e['total_changed'] for e in rolled['bound_activity'].values()),
        }

    with open(OUT_PATH, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    # ------------------------------------------------------------- print ----
    print('[P5.7] objective path (weighted base objective, all 48 blocks)\n')
    print(f"      {'solution':14s} {'base objective':>20s} "
          f"{'vs polish K=0':>18s} {'planning objective':>22s}")
    for row in report['objective_path']:
        print(f"      {row['solution']:14s} {row['weighted_base_objective']:20.4f} "
              f"{(row['delta_vs_polish_K0'] or 0.0):18.4f} "
              f"{row['total_planning_objective'] or float('nan'):22.6f}")

    print('\n[P5.7] cost-family decomposition\n')
    labels = [r['solution'] for r in report['objective_path']]
    families = sorted({f for e in report['objective_decomposition'].values()
                       for f in e['by_cost_family']})
    print(f"      {'family':32s}" + ''.join(f'{l:>20s}' for l in labels))
    for family in families:
        row = ''.join(
            f"{report['objective_decomposition'][l]['by_cost_family'].get(family, 0.0):20.2f}"
            for l in labels)
        print(f'      {family:32s}{row}')

    for name, entry in report['pairwise'].items():
        print(f'\n[P5.7] === {name}')
        print(f"      active-set symmetric difference : "
              f"{entry['total_active_set_symmetric_difference']}")
        print(f"      variable bound-activity changes : "
              f"{entry['total_bound_activity_changes']}")
        for group, values in entry['variables_by_group'].items():
            print(f"      {group:22s} max|d| {values['max_abs_delta']:12.6e}   "
                  f"L1 {values['total_l1_delta']:12.6e}   "
                  f"moved {values['total_moved_1e-6']:6d}/{values['total_n']}")
        if entry['active_set_changes']:
            print('      active-set changes by constraint family:')
            for family, values in list(entry['active_set_changes'].items())[:8]:
                print(f"        {family:38s} entered {values['total_entered']:5d} "
                      f"left {values['total_left']:5d} "
                      f"(active {values['total_active_a']} -> "
                      f"{values['total_active_b']} of {values['total_n']})")

    print(f'\n[P5.7] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
