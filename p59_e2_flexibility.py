"""
Stage P5.9-E2 -- flexibility behaviour against the frozen T0 anchor.

E1 established that neither the rescaling nor the rho override touches the
anchor (max |pc(T0) - pc(rescaled template)| = 0.0 exactly, same for qc).  What
remains is the narrow question the brief asks: does the RESCALED formulation
change how much flexibility is used against that frozen reference?

P5.7 §1 measured, under CURRENT, `flexibility_cost` GROWING by +3 040 929 from
polish K=0 to H12 while `generation_cost` fell by -7 310 095 -- refinement finds
a better dispatch and pays more flexibility to hold the interface away from a
reference frozen in T0.  This runs the same decomposition under both
formulations at the same generation, using production's own per-block
accessor via `p57_fingerprint`, weighted exactly as P5.7 weighted it.

No anchor policy is changed.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p59_e2_flexibility.py
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56b_candidates as BC  # noqa: E402
import p57_fingerprint as FP  # noqa: E402
import p59_eval as EV  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(EV.OUT_DIR, 'p59_e2_flexibility.json')
GENERATIONS = 4
RHO = {'v': 1.5, 'pf': 1000.0}


def families(record):
    """Weighted cost families summed over all 48 blocks."""
    fingerprint = record.get('fingerprint_polished')
    if not fingerprint:
        return None
    totals = {}
    for block in (fingerprint.get('blocks') or {}).values():
        for name, value in (block.get('cost_families') or {}).items():
            if value is not None:
                totals[name] = totals.get(name, 0.0) + value
    totals['total_weighted_base_objective'] = fingerprint.get(
        'total_weighted_base_objective')
    return totals


def run(x, label, rescale, rho):
    state, _ = EV.load_template(rescale=rescale, rho=rho)
    out = []
    for j in range(1, GENERATIONS + 1):
        case_id = f'{label}_j{j}'
        record, new_state = EV.evaluate(
            x, template_state=state, case_id=case_id, rho=rho,
            archive_label=case_id, capture_polished=True)
        out.append({'generation': j, 'status': record.get('status'),
                    'total_objective': record.get('total_objective'),
                    'families': families(record)})
        fam = out[-1]['families'] or {}
        print(f"      gen {j}: {record.get('status'):16s} "
              f"flex={fam.get('flexibility_cost')} "
              f"gen_cost={fam.get('generation_cost')}", flush=True)
        if new_state is None:
            break
        state = new_state
    return out


def growth(series, key):
    values = [r['families'][key] for r in series
              if r.get('families') and r['families'].get(key) is not None]
    if len(values) < 2:
        return None
    return {'first': values[0], 'last': values[-1], 'growth': values[-1] - values[0]}


def main():
    os.makedirs(EV.OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.9-E2 flexibility', EV.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.9] ABORTED\n{error}')
        sys.exit(1)

    x0 = dict(BC.population(planning_gate))['base']
    report = {'stage': 'P5.9-E2', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'anchor_policy_changed': False,
              'generations': GENERATIONS, 'rho': RHO,
              'p57_reference': {'flexibility_growth_K0_to_H12': 3040929.21,
                                'generation_change_K0_to_H12': -7310095.44}}

    def persist():
        with open(OUT_PATH, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    print('[P5.9-E2] CURRENT formulation', flush=True)
    report['CURRENT'] = run(x0, 'e2_current', rescale=False, rho=None)
    persist()
    print('[P5.9-E2] RESCALED + rho_pf=1000', flush=True)
    report['RESCALED'] = run(x0, 'e2_rescaled', rescale=True, rho=RHO)
    persist()

    report['comparison'] = {
        arm: {'flexibility': growth(report[arm], 'flexibility_cost'),
              'generation': growth(report[arm], 'generation_cost')}
        for arm in ('CURRENT', 'RESCALED')}
    persist()

    print('\n[P5.9-E2] flexibility and generation cost over '
          f'{GENERATIONS} generations')
    for arm, block in report['comparison'].items():
        for key, value in block.items():
            if value:
                print(f"      {arm:9} {key:12} {value['first']:18.2f} -> "
                      f"{value['last']:18.2f}   ({value['growth']:+15.2f})")
    print(f'\n[P5.9-E2] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
