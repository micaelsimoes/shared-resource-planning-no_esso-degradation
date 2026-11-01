"""
Stage P5.8-A0 -- ESS and consensus tolerance screen.  Diagnostic only.

Three cases at the canonical base candidate, from the SAME frozen T0 template, so
the only difference between them is the ADMM tolerance setting:

  CASE A   current parameters, exactly as `data/SRP1/SRP1_params.json` has them
  CASE B   harmonized ESS: ess_mean 1e-2 -> 1e-3, ess 1e-1 -> 1e-2
  CASE C   harmonized ESS + minimum_consecutive_converged_cycles 1 -> 2

Production defaults are NOT changed: `SRP1_params.json` is never written, and the
override is applied to the per-evaluation deep copy only.

The stage question is whether the ESS tolerance is a secondary contributor to the
REFINEMENT DRIFT, so each case runs a four-generation chain rather than a single
evaluation -- a single evaluation cannot show drift.  Case A must reproduce the
accepted P5.6-D base chain, which is checked and reported.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p58_a0_tolerances.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p57_eval as E7  # noqa: E402
import p57_fingerprint as FP  # noqa: E402
import p58_eval as E  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(E.OUT_DIR, 'p58_a0_tolerances.json')
GENERATIONS = 4

CASES = {
    'A (current)': {},
    'B (harmonized ESS)': {
        'consensus': {'ess_mean': 1.0e-3, 'ess': 1.0e-2}},
    'C (harmonized ESS + 2 cycles)': {
        'consensus': {'ess_mean': 1.0e-3, 'ess': 1.0e-2},
        'minimum_consecutive_converged_cycles': 2},
}

P56D_BASE_CHAIN = {1: 828021090.360850, 2: 827415318.563944,
                   3: 826824028.845478, 4: 826405022.193437}
TAU_NUMERICAL = 10.0

# families the stage names explicitly
GROUPS = {'ESS schedule': ('shared_es_pch', 'shared_es_pdch', 'shared_es_pnet',
                           'shared_es_qnet', 'shared_es_soc'),
          'interface P/Q': ('flex_p_up', 'flex_p_down', 'flex_q_up',
                            'flex_q_down'),
          'generation dispatch': ('pg', 'qg')}


def movement(label_a, label_b):
    """Per-group movement between two archived solutions."""
    path_a = os.path.join(E.ARCHIVE_DIR, f'{label_a}.npz')
    path_b = os.path.join(E.ARCHIVE_DIR, f'{label_b}.npz')
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return None
    rolled = FP.rollup(FP.compare(path_a, path_b))
    out = {}
    for group, names in GROUPS.items():
        entries = [rolled['variables'][n] for n in names
                   if n in rolled['variables']]
        if not entries:
            continue
        out[group] = {
            'max_abs_delta': max(e['max_abs_delta'] for e in entries),
            'total_l1_delta': sum(e['total_l1_delta'] for e in entries),
            'total_moved_1e-6': sum(e['total_moved_1e-6'] for e in entries)}
    return out


def main():
    os.makedirs(E.OUT_DIR, exist_ok=True)
    os.makedirs(E.ARCHIVE_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.8-A0 tolerance screen', E.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.8] ABORTED\n{error}')
        sys.exit(1)

    x0 = dict(BC.population(planning_gate))['base']
    report = {'stage': 'P5.8-A0', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'candidate': 'canonical positive-bootstrap base x0',
              'template': 'T0 a81f7f5191dd42dbf50d1726149b8909 (shared by all '
                          'three cases, so only the tolerances differ)',
              'generations_per_case': GENERATIONS,
              'production_defaults_changed': False,
              'p56d_base_chain_reference': P56D_BASE_CHAIN,
              'cases': {}}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    print('[P5.8] obtaining T0 ...', flush=True)
    t0 = E7.t0_template()

    for case, overrides in CASES.items():
        tag = case.split()[0]
        print(f'\n[P5.8] ===== CASE {case} =====', flush=True)
        entry = {'overrides': overrides, 'generations': [], 'chain': []}
        report['cases'][case] = entry
        persist()
        state = t0
        for j in range(1, GENERATIONS + 1):
            record, new_state = E.evaluate(
                x0, template_state=state, eval_id=f'p58_a0_{tag}_j{j}',
                archive_label=f'a0_{tag}_j{j}',
                tolerance_overrides=overrides or None)
            row = {'generation': j, **E.summarize(record)}
            row['cycle_detail'] = record.get('admm_cycles_detail')
            entry['generations'].append(row)
            entry['chain'].append(record.get('total_objective'))
            if j == 1:
                entry['tolerances_in_force'] = record.get('tolerances_in_force')
                entry['tolerances_replaced'] = record.get('tolerances_replaced')
            persist()
            print(f"      gen {j}: {row['status']:16s} cycles={row['admm_cycles']} "
                  f"Q={record.get('total_objective')} "
                  f"admm->polish={row['admm_to_polish_improvement']}", flush=True)
            if record['status'] != O.STATUS_VALID:
                entry['stopped_at_generation'] = j
                break
            state = new_state

        chain = [q for q in entry['chain'] if q is not None]
        if len(chain) >= 2:
            deltas = [chain[i] - chain[i - 1] for i in range(1, len(chain))]
            entry['step_deltas'] = deltas
            entry['refinement_drift_gen1_to_last'] = chain[-1] - chain[0]
            entry['mean_step_delta'] = float(np.mean(deltas))
        for j in range(2, len(chain) + 1):
            move = movement(f'a0_{tag}_j{j - 1}__polished',
                            f'a0_{tag}_j{j}__polished')
            if move:
                entry.setdefault('movement_between_generations', {})[
                    f'{j - 1}->{j}'] = move
        persist()
        print(f"      drift gen1->gen{len(chain)}: "
              f"{entry.get('refinement_drift_gen1_to_last')}", flush=True)

    # case A must reproduce the accepted chain
    case_a = report['cases']['A (current)']['chain']
    reproduction = {}
    for j, value in enumerate(case_a, start=1):
        if value is None or j not in P56D_BASE_CHAIN:
            continue
        delta = value - P56D_BASE_CHAIN[j]
        reproduction[j] = {'p56d': P56D_BASE_CHAIN[j], 'p58': value,
                           'delta': delta,
                           'within_tau_numerical': abs(delta) <= TAU_NUMERICAL}
    report['case_A_reproduces_p56d'] = all(
        r['within_tau_numerical'] for r in reproduction.values())
    report['case_A_reproduction'] = reproduction
    persist()

    print('\n[P5.8] A0 summary')
    print(f"      {'case':32s} {'drift g1->g4':>16s} {'mean step':>14s} "
          f"{'cycles (g2..)':>14s}")
    for case, entry in report['cases'].items():
        cycles = [g['admm_cycles'] for g in entry['generations'][1:]]
        print(f"      {case:32s} "
              f"{entry.get('refinement_drift_gen1_to_last', float('nan')):16.2f} "
              f"{entry.get('mean_step_delta', float('nan')):14.2f} "
              f"{str(cycles):>14s}")
    print(f"\n[P5.8] case A reproduces the accepted P5.6-D chain: "
          f"{report['case_A_reproduces_p56d']}")
    print(f'[P5.8] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
