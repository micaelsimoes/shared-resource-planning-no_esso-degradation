"""
Stage P5.7 hypothesis D, completion -- penalty continuation and the fixed-
consensus control, both at the base candidate.

The main harness ran D4 at the P5.6-D target `se|node9|2025|-10%`, where the
DIRECT polish already fails; its first penalty step therefore failed too and the
homotopy never started.  That is a defect of where the experiment was placed,
not a result, so it is re-run here at x0, where the direct polish succeeds.

Two sequences, both from ONE ADMM result, so the consensus is IDENTICAL
throughout and cannot contribute:

  penalty continuation -- polish four times with the objective multiplied by
      (1 / effective_scale) ** t for t = 1, 2/3, 1/3, 0, each warm-started from
      the previous.  t = 1 is the magnitude the ADMM subproblem gives the base
      objective; t = 0 is production's own polish.

  fixed-consensus control -- polish four times at t = 0, each warm-started from
      the previous.  Four solves, same count, no scaling change and no consensus
      change.  This separates "more solves" from "more ADMM updates".

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p57_d4_penalty.py
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56b_candidates as BC  # noqa: E402
import p57_eval as E  # noqa: E402
import p57_hypotheses as H  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(E.OUT_DIR, 'p57_d4_penalty.json')

# for reference, from the accepted P5.6-D base chain
CHAIN = {1: 828021090.360850, 2: 827415318.563944, 3: 826824028.845478,
         4: 826405022.193437}


def main():
    try:
        provenance, planning_gate = gate('P5.7 D4 penalty continuation', E.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.7] ABORTED\n{error}')
        sys.exit(1)

    x0 = dict(BC.population(planning_gate))['base']
    report = {'stage': 'P5.7-D4 penalty continuation', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'candidate': 'canonical base x0',
              'base_chain_reference': CHAIN,
              'note': ('both sequences reuse ONE ADMM result, so the consensus '
                       'is identical throughout and cannot contribute'),
              'penalty_continuation': [], 'fixed_consensus_control': []}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    print('[P5.7] obtaining T0 ...', flush=True)
    t0 = E.t0_template()
    ctx = H.prepare(x0, t0, 'p57_D4_base')
    report['admm'] = ctx['admm']
    persist()

    print('\n[P5.7] penalty (objective-scaling) continuation at x0', flush=True)
    values = None
    for t in (1.0, 2 / 3, 1 / 3, 0.0):
        result, values = H.polish_variant(
            ctx, f'D4b_penalty_t{t:.3f}'.replace('.', '_'),
            rescale=('admm_scale_pow', t), init_values=values)
        report['penalty_continuation'].append({
            'exponent_t': t, 'status': result['status'],
            'rescale_factor_min': result.get('rescale_factor_min'),
            'rescale_factor_max': result.get('rescale_factor_max'),
            'total_objective': result.get('total_objective'),
            'runtime_s': result['runtime_s']})
        persist()
        print(f"      t={t:.4f}  factor {result.get('rescale_factor_min')}"
              f" .. {result.get('rescale_factor_max')}  "
              f"{result['status']:16s} Q={result.get('total_objective')}",
              flush=True)
        if result['status'] != 'VALID':
            break

    print('\n[P5.7] fixed-consensus control: four polishes at production scale',
          flush=True)
    values = None
    for k in range(1, 5):
        result, values = H.polish_variant(ctx, f'D4b_control_{k}',
                                          rescale=None, init_values=values)
        report['fixed_consensus_control'].append({
            'solve': k, 'status': result['status'],
            'total_objective': result.get('total_objective'),
            'runtime_s': result['runtime_s']})
        persist()
        print(f"      solve {k}  {result['status']:16s} "
              f"Q={result.get('total_objective')}", flush=True)
        if result['status'] != 'VALID':
            break

    persist()
    print(f'\n[P5.7] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
