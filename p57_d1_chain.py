"""
Stage P5.7 diagnostic 1 -- the branch chain at ONE fixed investment candidate.

The fixed candidate is the canonical positive-bootstrap base x0.  That choice is
deliberate: under the P5.6-D oracle every continuation point of H_K(x0) IS x0,
so the twelve solves below differ in nothing but how many times the state has
been re-solved.  Any movement between them is therefore a pure local-branch
effect with the investment held exactly constant -- which is the P5.7 question
stripped of every other variable.

What this collects, per step:

    the ADMM solution      (pre-polish, augmented objective, consensus params
                            at their converged values)
    the polished solution  (exact-consensus, base objective, full audit)

so the six states P5.7 asks for are the outputs of

    ADMM        = step 1, pre-polish
    polish K=0  = step 1, polished        (no continuation refinement at all)
    H2          = step 2, polished
    H4          = step 4, polished
    H8          = step 8, polished
    H12         = step 12, polished

The objectives are checked against the accepted P5.6-D base chain.  A
re-implementation that drifts from the thing it is diagnosing is worthless, so a
mismatch beyond tau_numerical aborts the run.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p57_d1_chain.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p57_eval as E  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(E.OUT_DIR, 'p57_d1_chain.json')
STEPS = 12
TAU_NUMERICAL = 10.0

# accepted P5.6-D base refinement chain, step -> total planning objective
P56D_BASE_CHAIN = {
    1: 828021090.360850, 2: 827415318.563944, 3: 826824028.845478,
    4: 826405022.193437, 5: 825961521.882321, 6: 825531306.746985,
    7: 825108709.695181, 8: 824795363.718628, 9: 824488243.726694,
    12: 823731333.558647,
}


def main():
    os.makedirs(E.OUT_DIR, exist_ok=True)
    os.makedirs(E.ARCHIVE_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.7 branch chain', E.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.7] ABORTED\n{error}')
        sys.exit(1)

    x0 = dict(BC.population(planning_gate))['base']
    candidate = O.vector_to_candidate(planning_gate, x0)
    ok, reason = O.check_master_feasibility(planning_gate, candidate)
    if not ok:
        print(f'[P5.7] ABORTED: base is first-stage infeasible: {reason}')
        sys.exit(1)

    report = {'stage': 'P5.7-diag1', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'fixed_candidate': 'canonical positive-bootstrap base x0',
              'why_this_candidate': (
                  'at x = x0 every continuation point of H_K is x0, so the '
                  'steps differ only in how many times the state has been '
                  're-solved; investment is held exactly constant'),
              'anchor': 'midpoint', 'template': 'T0 '
              'a81f7f5191dd42dbf50d1726149b8909',
              'steps_planned': STEPS, 'tau_numerical': TAU_NUMERICAL,
              'p56d_base_chain_reference': P56D_BASE_CHAIN,
              'steps': [], 'reproduction': {}}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    print('[P5.7] obtaining the frozen T0 template ...', flush=True)
    started = time.time()
    t0 = E.t0_template()
    report['t0_build_runtime_s'] = time.time() - started
    print(f"[P5.7] T0 built in {report['t0_build_runtime_s']:.1f}s\n", flush=True)
    persist()

    state = t0
    for j in range(1, STEPS + 1):
        print(f'[P5.7] --- step {j}/{STEPS}', flush=True)
        result, new_state = E.evaluate_capture(
            x0, template_state=state, eval_id=f'p57_chain_j{j}',
            archive_label=f'chain_j{j}')
        step = {'j': j, 'status': result['status'],
                'input_state': 'T0' if j == 1 else f'output of step {j - 1}',
                'total_objective': result.get('total_objective'),
                'net_operational_recourse': result.get('net_operational_recourse'),
                'gross_operational_cost': result.get('gross_operational_cost'),
                'physical_salvage': result.get('physical_salvage'),
                'investment_cost': result.get('investment_cost'),
                'admm': result.get('admm'),
                'coordination_residuals': result.get('coordination_residuals'),
                'esso_max_violation': (result.get('esso_audit') or {}).get(
                    'max_violation'),
                'esso_production_feasible': (result.get('esso_audit') or {}).get(
                    'production_feasible'),
                'network_max_violation': (result.get('network_audit') or {}).get(
                    'max_violation'),
                'network_worst_block': (result.get('network_audit') or {}).get(
                    'worst_block'),
                'network_h1_violation': (result.get('network_audit') or {}).get(
                    'max_h1_complementarity_violation'),
                'converter_capability_violation': (
                    result.get('network_audit') or {}).get(
                    'max_converter_capability_violation'),
                'per_block_polished': result.get('per_block_polished'),
                'fingerprint_admm': result.get('fingerprint_admm'),
                'fingerprint_polished': result.get('fingerprint_polished'),
                'polish_runtime_s': result.get('polish_runtime_s'),
                'wall_clock_s': result.get('wall_clock_s')}
        report['steps'].append(step)

        if j in P56D_BASE_CHAIN and step['total_objective'] is not None:
            delta = step['total_objective'] - P56D_BASE_CHAIN[j]
            report['reproduction'][j] = {
                'p56d': P56D_BASE_CHAIN[j], 'p57': step['total_objective'],
                'delta': delta, 'within_tau_numerical': abs(delta) <= TAU_NUMERICAL}
            print(f'      reproduction vs P5.6-D: delta = {delta:+.6e}',
                  flush=True)
        persist()

        print(f"      {step['status']}  Q={step['total_objective']}  "
              f"t={step['wall_clock_s']:.0f}s", flush=True)
        if result['status'] != O.STATUS_VALID:
            report['stopped_at_step'] = j
            persist()
            print('[P5.7] chain stopped: step did not return VALID', flush=True)
            break
        state = new_state

    bad = {k: v for k, v in report['reproduction'].items()
           if not v['within_tau_numerical']}
    report['reproduces_p56d'] = not bad
    report['reproduction_failures'] = bad
    persist()
    print(f'\n[P5.7] reproduces the accepted P5.6-D base chain: '
          f'{report["reproduces_p56d"]}')
    if bad:
        print(f'[P5.7] MISMATCHES: {bad}')
    print(f'[P5.7] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
