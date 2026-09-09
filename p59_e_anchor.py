"""
Stage P5.9-E -- interface anchor revisit under the rescaled formulation.

NO ANCHOR POLICY IS CHANGED.  P5.8-E already settled that the anchor is common
across candidates and across start policies (cold vs T0-warm agreed to
1.003037e-07 p.u.), so this is explicitly not a priority and is run only after
A-D.  Two narrow questions remain, and only these are asked:

  1. does the RESCALED formulation move the anchor, or change how much
     flexibility is used against it?
  2. does the growth in `flexibility_cost` along a refinement chain -- P5.7 §1
     measured +3 040 929 over eleven generations against a reference frozen in
     T0 -- behave differently once the subproblems are properly scaled?

`_prepare_transmission_objectives_for_admm` fixes the TSO's ADN load `pc`/`qc`
at the DSO's then-current consensus interface power
(`shared_resources_planning.py:2905-2919`); the warm-start path clones
`initial_state['models']` (2173-2179) and never re-runs it, so the anchor is
inherited from T0 and never refreshed.  That is measured here, not assumed, and
nothing is done about it.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p59_e_anchor.py
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p57_eval as E7  # noqa: E402
import p58_e_anchor as E8A  # noqa: E402
import p59_eval as EV  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(EV.OUT_DIR, 'p59_e_anchor.json')
A_PATH = os.path.join(EV.OUT_DIR, 'p59_a_sweep.json')
D_PATH = os.path.join(EV.OUT_DIR, 'p59_d_replay.json')

# P5.7 §1, CURRENT formulation, polish K=0 -> H12
P57_FLEXIBILITY_GROWTH = 3040929.21
P57_GENERATION_SAVING = -7310095.44


def flexibility_series(chain):
    """flexibility_cost and generation_cost per generation, where recorded."""
    out = []
    for row in chain.get('generations', []):
        families = row.get('cost_families') or {}
        out.append({'generation': row.get('generation'),
                    'status': row.get('status'),
                    'flexibility_cost': families.get('flexibility_cost'),
                    'generation_cost': families.get('generation_cost'),
                    'gross_operational_cost': families.get('gross_operational_cost'),
                    'total_objective': row.get('total_objective')})
    return out


def growth(series, key):
    values = [r[key] for r in series if r.get(key) is not None]
    if len(values) < 2:
        return None
    return {'first': values[0], 'last': values[-1],
            'growth': values[-1] - values[0], 'n': len(values)}


def main():
    os.makedirs(EV.OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.9-E anchor revisit', EV.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.9] ABORTED\n{error}')
        sys.exit(1)

    with open(A_PATH) as handle:
        a_report = json.load(handle)
    best_rho = (a_report.get('best_case_row') or {}).get('rho_requested')

    report = {'stage': 'P5.9-E', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'anchor_policy_changed': False,
              'best_rho': best_rho,
              'p58_e_conclusion': ('anchor common across candidates and across '
                                   'start policies; not a priority'),
              'p57_reference': {'flexibility_growth_K0_to_H12': P57_FLEXIBILITY_GROWTH,
                                'generation_saving_K0_to_H12': P57_GENERATION_SAVING}}

    # ---- E1: does rescaling move the anchor at all? ------------------------
    with open(E7.T0_CACHE, 'rb') as handle:
        t0 = pickle.load(handle)
    anchor_t0 = E8A.anchor(t0['models']['tso'])
    report['anchor_t0_blocks'] = len(anchor_t0)

    state, info = EV.load_template(rescale=True, rho=best_rho)
    anchor_rescaled_template = E8A.anchor(state['models']['tso'])
    report['E1_rescaling_moves_anchor'] = E8A.compare(
        anchor_t0, anchor_rescaled_template)
    report['E1_note'] = ('rescaling and the rho override act on the objective '
                         'only; any nonzero difference here would mean one of '
                         'them touched the fixed anchor, which it must not')

    # ---- E2: flexibility behaviour along the chains, from D ----------------
    if os.path.exists(D_PATH):
        with open(D_PATH) as handle:
            d_report = json.load(handle)
        report['E2_flexibility'] = {}
        for name, entry in d_report.get('candidates', {}).items():
            block = {}
            for arm in ('CURRENT', 'RESCALED'):
                if arm not in entry:
                    continue
                series = flexibility_series(entry[arm])
                block[arm] = {
                    'series': series,
                    'flexibility_growth': growth(series, 'flexibility_cost'),
                    'generation_growth': growth(series, 'generation_cost')}
            report['E2_flexibility'][name] = block
    else:
        report['E2_flexibility'] = 'stage D report not present'

    with open(OUT_PATH, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print('[P5.9-E] anchor revisit')
    for name, value in report['E1_rescaling_moves_anchor'].items():
        print(f"      max |anchor(T0) - anchor(rescaled template)| {name}: "
              f"{value['max_over_blocks']:.6e}")
    if isinstance(report['E2_flexibility'], dict):
        for name, block in report['E2_flexibility'].items():
            print(f'\n      {name}')
            for arm, data in block.items():
                flex = data['flexibility_growth']
                gen = data['generation_growth']
                if flex:
                    print(f"        {arm:9} flexibility {flex['first']:.2f} -> "
                          f"{flex['last']:.2f}  ({flex['growth']:+.2f} over "
                          f"{flex['n']} generations)")
                if gen:
                    print(f"        {arm:9} generation  {gen['first']:.2f} -> "
                          f"{gen['last']:.2f}  ({gen['growth']:+.2f})")
    print(f'\n[P5.9-E] report -> {OUT_PATH}')


if __name__ == '__main__':
    main()
