"""
Stage P5.8-C -- full ADMM replay, CURRENT versus RESCALED.

The comparison is controlled: both chains start from the SAME frozen T0 primal
state.  For the RESCALED chain the template's network subproblem objectives are
multiplied by their own `effective_scale` once, before the chain starts; because
production's warm-start path clones `initial_state['models']` and never rebuilds
the augmented objectives, every generation then runs the RESCALED formulation
from an identical initial point.  P5.7-B measured initialization sensitivity at
303 on 8.3e8, so what is left is the formulation.

The polish is production's own in both chains -- `p58_eval` removes the
diagnostic objective before polishing -- so the polished values remain directly
comparable to the accepted P5.6-D chain.

The CURRENT chain is not re-run here: it is the accepted P5.6-D base chain,
reproduced twice already (P5.7 to 4.8e-07, P5.8-A0 case A exactly).

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p58_c_replay.py
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p57_eval as E7  # noqa: E402
import p58_eval as E  # noqa: E402
import p58_rescale as R  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(E.OUT_DIR, 'p58_c_replay.json')
GENERATIONS = 8

# the accepted CURRENT chain, reproduced exactly by P5.8-A0 case A
CURRENT_CHAIN = {1: 828021090.360850, 2: 827415318.563944, 3: 826824028.845478,
                 4: 826405022.193437, 5: 825961521.882321, 6: 825531306.746985,
                 7: 825108709.695181, 8: 824795363.718628}
PLANNING_SIGNAL = 33031.0     # P5.6-D best-to-second candidate gap
TAU_NUMERICAL = 10.0


def main():
    os.makedirs(E.OUT_DIR, exist_ok=True)
    os.makedirs(E.ARCHIVE_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.8-C ADMM replay', E.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.8] ABORTED\n{error}')
        sys.exit(1)

    x0 = dict(BC.population(planning_gate))['base']
    report = {'stage': 'P5.8-C', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'candidate': 'canonical positive-bootstrap base x0',
              'generations': GENERATIONS,
              'current_chain_reference': CURRENT_CHAIN,
              'planning_signal_to_resolve': PLANNING_SIGNAL,
              'design': ('both chains start from the same frozen T0 primal '
                         'state; only the network subproblem objective scaling '
                         'differs'),
              'rescaled_chain': []}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    # an INDEPENDENT load, so the cached CURRENT template is never mutated
    print('[P5.8] loading a private copy of T0 ...', flush=True)
    with open(E7.T0_CACHE, 'rb') as handle:
        t0 = pickle.load(handle)
    applied = R.rescale_state_models(t0)
    report['blocks_rescaled'] = len([v for v in applied.values() if v])
    report['effective_scale_range'] = [min(v for v in applied.values() if v),
                                       max(v for v in applied.values() if v)]
    persist()
    print(f"[P5.8] rescaled {report['blocks_rescaled']} template subproblems, "
          f"effective_scale in {report['effective_scale_range']}\n", flush=True)

    state = t0
    for j in range(1, GENERATIONS + 1):
        record, new_state = E.evaluate(
            x0, template_state=state, eval_id=f'p58_c_rescaled_j{j}',
            archive_label=f'c_rescaled_j{j}', capture_admm=True)
        row = {'generation': j, **E.summarize(record)}
        row['cycle_detail'] = record.get('admm_cycles_detail')
        if record.get('fingerprint_polished'):
            row['archive_polished'] = record['fingerprint_polished']['archive']
        if record.get('fingerprint_admm'):
            row['archive_admm'] = record['fingerprint_admm']['archive']
        report['rescaled_chain'].append(row)
        persist()
        current = CURRENT_CHAIN.get(j)
        print(f"      gen {j}: {row['status']:16s} cycles={row['admm_cycles']} "
              f"ADMM recourse={row['admm_net_recourse_before_polish']}  "
              f"Q={record.get('total_objective')}  "
              f"(CURRENT Q={current})", flush=True)
        if row.get('failed_blocks'):
            print(f"           polish failed on {len(row['failed_blocks'])} "
                  f"blocks: {row['failed_blocks'][:6]}", flush=True)
        if new_state is None:
            report['stopped_at_generation'] = j
            break
        if record['status'] != O.STATUS_VALID:
            # the ADMM stage succeeded; only the downstream polish did not, so
            # the chain continues and the ADMM-side comparison stays complete
            report.setdefault('generations_without_a_polish', []).append(j)
        state = new_state

    report['rescaled_admm_prepolish_recourse'] = [
        r.get('admm_net_recourse_before_polish') for r in report['rescaled_chain']]
    chain = [r['total_objective'] for r in report['rescaled_chain']
             if r.get('total_objective') is not None]
    if len(chain) >= 2:
        deltas = [chain[i] - chain[i - 1] for i in range(1, len(chain))]
        report['rescaled_step_deltas'] = deltas
        report['rescaled_drift_gen1_to_last'] = chain[-1] - chain[0]
        current_list = [CURRENT_CHAIN[j] for j in sorted(CURRENT_CHAIN)][:len(chain)]
        current_deltas = [current_list[i] - current_list[i - 1]
                          for i in range(1, len(current_list))]
        report['current_step_deltas'] = current_deltas
        report['current_drift_gen1_to_last'] = current_list[-1] - current_list[0]
        report['refinements_to_reach_planning_signal'] = {
            'CURRENT': next((i + 2 for i, d in enumerate(current_deltas)
                             if abs(d) <= PLANNING_SIGNAL), None),
            'RESCALED': next((i + 2 for i, d in enumerate(deltas)
                              if abs(d) <= PLANNING_SIGNAL), None)}
        report['refinements_to_reach_tau_numerical'] = {
            'CURRENT': next((i + 2 for i, d in enumerate(current_deltas)
                             if abs(d) <= TAU_NUMERICAL), None),
            'RESCALED': next((i + 2 for i, d in enumerate(deltas)
                              if abs(d) <= TAU_NUMERICAL), None)}

    report['admm_to_polish_gap'] = {
        'RESCALED': [r['admm_to_polish_improvement']
                     for r in report['rescaled_chain']]}
    persist()

    print('\n[P5.8] C summary')
    print(f"      {'gen':>4} {'CURRENT Q':>20} {'RESCALED Q':>20} "
          f"{'CURRENT step':>15} {'RESCALED step':>15}")
    for j, row in enumerate(report['rescaled_chain'], start=1):
        cur = CURRENT_CHAIN.get(j)
        cs = (CURRENT_CHAIN[j] - CURRENT_CHAIN[j - 1]) if j > 1 and \
            j in CURRENT_CHAIN and (j - 1) in CURRENT_CHAIN else None
        rs = report.get('rescaled_step_deltas', [None] * 99)[j - 2] if j > 1 else None
        print(f"      {j:4d} {cur if cur else float('nan'):20.6f} "
              f"{row['total_objective'] or float('nan'):20.6f} "
              f"{cs if cs is not None else float('nan'):15.2f} "
              f"{rs if rs is not None else float('nan'):15.2f}")
    print(f'\n[P5.8] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
