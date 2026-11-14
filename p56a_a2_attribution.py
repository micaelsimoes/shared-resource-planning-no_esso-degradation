"""
Stage P5.6-A2 -- attribute the ~13% ADMM-to-polished difference.

P5.5-D1 reported a polished value 13.0% below the ADMM recourse and attributed
it to the ADMM minimising its augmented objective.  P5.6-A0.4 withdrew that
attribution as unproved.  This script attributes it from the recorded per-block
base-objective decomposition, which the oracle captures before and after
polishing within the SAME evaluation, so no second ADMM run is needed.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p56a_a2_attribution.py
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402

OUT_DIR = O.OUT_DIR
FAMILIES = ('generation_cost', 'flexibility_cost', 'load_curtailment_cost',
            'gen_curtailment_penalty', 'ess_utilization_cost_penalty',
            'slack_penalties', 'ess_complementarity_penalties')


def main():
    source = os.path.join(OUT_DIR, 'p56a_a1_certificate.json')
    if not os.path.exists(source):
        print(f'[A2] ABORTED: {source} not found -- run P5.6-A1 first.')
        sys.exit(1)
    with open(source) as handle:
        a1 = json.load(handle)

    admm, polished = a1['per_block_admm'], a1['per_block_polished']
    total_admm = sum(v['weighted_base_objective'] for v in admm.values())
    total_polished = sum(v['weighted_base_objective'] for v in polished.values())

    by_family = defaultdict(float)
    by_agent = defaultdict(float)
    by_agent_year_day = {}
    for key in admm:
        agent = key.split('|')[0]
        delta_block = (polished[key]['weighted_base_objective']
                       - admm[key]['weighted_base_objective'])
        by_agent[agent] += delta_block
        entry = {'admm': admm[key]['weighted_base_objective'],
                 'polished': polished[key]['weighted_base_objective'],
                 'delta': delta_block, 'families': {}}
        for family in FAMILIES:
            a_val = admm[key]['families'].get(family)
            p_val = polished[key]['families'].get(family)
            if a_val is None or p_val is None:
                continue
            entry['families'][family] = {'admm': a_val, 'polished': p_val,
                                         'delta': p_val - a_val}
            by_family[family] += p_val - a_val
        by_agent_year_day[key] = entry

    # the objective sums the first six families; complementarity is inside it too
    family_total = sum(by_family[f] for f in FAMILIES)
    ranked_blocks = sorted(by_agent_year_day.items(), key=lambda kv: kv[1]['delta'])
    spring_tso = sum(v['delta'] for k, v in by_agent_year_day.items()
                     if k.startswith('TSO|') and k.endswith('|Spring'))

    report = {
        'stage': 'P5.6-A2', 'provenance': a1.get('provenance'),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'p56a_a1_certificate.json (single evaluation, both snapshots)',
        'total_admm_base_objective': total_admm,
        'total_polished_base_objective': total_polished,
        'total_delta': total_polished - total_admm,
        'total_delta_relative': (total_polished - total_admm) / total_admm,
        'delta_by_family': dict(by_family),
        'delta_by_family_sum': family_total,
        'delta_by_agent': dict(by_agent),
        'ten_largest_negative_block_deltas': [
            {'block': k, **{kk: vv for kk, vv in v.items() if kk != 'families'}}
            for k, v in ranked_blocks[:10]],
        'tso_spring_total_delta': spring_tso,
        'tso_spring_share_of_total': (spring_tso / (total_polished - total_admm)
                                      if total_polished != total_admm else None),
        'per_block': by_agent_year_day,
    }

    out = os.path.join(OUT_DIR, 'p56a_a2_attribution.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'[A2] ADMM base objective      {total_admm:.4f}')
    print(f'[A2] polished base objective  {total_polished:.4f}')
    print(f'[A2] delta                    {total_polished - total_admm:.4f} '
          f'({100 * (total_polished - total_admm) / total_admm:.4f} %)')
    print('\n[A2] delta by objective family')
    for family, value in sorted(by_family.items(), key=lambda kv: kv[1]):
        share = 100 * value / (total_polished - total_admm)
        print(f'      {family:32s} {value:18.4f}   {share:8.3f} %')
    print(f'      {"SUM":32s} {family_total:18.4f}')
    print('\n[A2] delta by agent')
    for agent, value in sorted(by_agent.items(), key=lambda kv: kv[1]):
        print(f'      {agent:8s} {value:18.4f}')
    print('\n[A2] ten largest negative block deltas')
    for key, entry in ranked_blocks[:10]:
        biggest = sorted(entry['families'].items(), key=lambda kv: kv[1]['delta'])
        top = biggest[0] if biggest else ('-', {'delta': 0.0})
        print(f"      {key:22s} {entry['delta']:16.4f}   "
              f"dominant: {top[0]} {top[1]['delta']:.4f}")
    print(f"\n[A2] TSO Spring blocks total {spring_tso:.4f} "
          f"({100 * spring_tso / (total_polished - total_admm):.2f} % of the change)")
    print(f'\n[A2] report -> {out}')


if __name__ == '__main__':
    main()
