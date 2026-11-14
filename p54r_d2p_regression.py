"""
Stage P5.4-R1 -- canonical D2-P positive-bootstrap regression (provenance check).

Short gate only: confirms the accepted post-D2-P production formulation still
holds on the CANONICAL SRP1 scenarios. No formulation change.

Writes to data/SRP1/Results/P54R_D2P/ so the earlier noncanonical evidence under
P54H1/ and P54E/ is preserved untouched.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p54r_d2p_regression.py
"""

import json
import os
import statistics
import sys
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from definitions import ESS_COMPLEMENTARITY_TOLERANCE, SHARED_ESS_ZERO_CAPACITY_TOLERANCE  # noqa: E402
from p53b3_active_power_ess import jacobian_for, run_branch  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54R_D2P')
JAC_TAGS = {'case33_1/2030/Winter', 'case9/2025/Winter'}
EPS = ESS_COMPLEMENTARITY_TOLERANCE
SQRT_EPS = EPS ** 0.5


def main():
    try:
        provenance, _ = gate('P5.4-R1 canonical D2-P regression', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[R1] ABORTED\n{error}')
        sys.exit(1)

    report = {'stage': 'P5.4-R1', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    print('\n[R1] canonical positive-bootstrap initialization ...', flush=True)
    res = run_branch(prototype=False, capture_tags=JAC_TAGS)
    agg = res['aggregates']

    jac = {}
    for tag in sorted(JAC_TAGS):
        if tag in res['captured']:
            spec = jacobian_for(res['captured'][tag], tag)
            jac[tag] = {
                'n_equality_rows': spec['full'].get('n_rows'),
                'n_exactly_zero_rows': spec['full'].get('n_exactly_zero_rows'),
                'zero_row_components': spec.get('zero_row_components'),
                'sigma_min_full': spec['full'].get('sigma_min'),
                'reduced_condition_number': spec.get('reduced', {}).get('condition_number'),
                'full_row_rank': spec['full'].get('n_exactly_zero_rows') == 0,
            }

    cap, comp, circ = [], [], []
    for tag, rec in res['physics'].items():
        s = rec['s_rated']
        if s <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE:
            continue
        for i in range(len(rec['pnet'])):
            pnet, qnet = rec['pnet'][i], rec['qnet'][i]
            pch, pdch = rec['pch'][i], rec['pdch'][i]
            hc, hd = rec['pch_hat'][i], rec['pdch_hat'][i]
            cap.append(max((pnet ** 2 + qnet ** 2 - s ** 2) / s ** 2, 0.0))
            comp.append(max(hc * hd - EPS, 0.0))
            circ.append(min(pch, pdch) / s)

    report['R1_gate'] = {
        'per_agent': agg['per_agent'],
        'primary_failures': agg['primary_failures'],
        'recovery_attempts': agg['recovery_attempts'],
        'persistent_failures': agg['persistent_failures'],
        'persistent_failure_ids': agg['persistent_failure_ids'],
        'iterations': agg['iterations'],
        'runtime_s': agg['runtime_s'],
        'n_shared_ess_rows': len(cap),
        'max_converter_capability_violation': max(cap) if cap else None,
        'n_converter_capability_violations': sum(1 for v in cap if v > 0.0),
        'max_h1_complementarity_violation': max(comp) if comp else None,
        'n_h1_complementarity_violations': sum(1 for v in comp if v > 0.0),
        'max_p_circ_norm': max(circ) if circ else None,
        'mean_p_circ_norm': statistics.fmean(circ) if circ else None,
        'max_p_circ_over_sqrt_eps': (max(circ) / SQRT_EPS) if circ else None,
        'n_rows_above_1e-2_S': sum(1 for v in circ if v > 1e-2),
        'jacobian': jac,
    }
    g = report['R1_gate']
    g['requirements'] = {
        'dso_36_of_36': agg['per_agent'].get('dso', {}).get('succeeded') == 36,
        'tso_12_of_12': agg['per_agent'].get('tso', {}).get('succeeded') == 12,
        'esso_3_of_3': agg['per_agent'].get('esso', {}).get('succeeded') == 3,
        'zero_persistent_failures': agg['persistent_failures'] == 0,
        'zero_h1_complementarity_violations': g['n_h1_complementarity_violations'] == 0,
        'zero_converter_capability_violations': g['n_converter_capability_violations'] == 0,
    }
    g['all_requirements_met'] = all(g['requirements'].values())

    out = os.path.join(OUT_DIR, 'p54r_d2p_report.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f"\n[R1] {agg['per_agent']}")
    print(f"    primary={agg['primary_failures']} recovery={agg['recovery_attempts']} "
          f"persistent={agg['persistent_failures']} {agg['persistent_failure_ids']}")
    print(f"    iterations={agg['iterations']}")
    print(f"    runtime={agg['runtime_s']:.0f}s")
    print(f"    converter-capability violations: {g['n_converter_capability_violations']} "
          f"/ {g['n_shared_ess_rows']} (max {g['max_converter_capability_violation']:.4e})")
    print(f"    H1 complementarity violations  : {g['n_h1_complementarity_violations']} "
          f"/ {g['n_shared_ess_rows']} (max {g['max_h1_complementarity_violation']:.4e})")
    print(f"    max min(pch,pdch)/S = {g['max_p_circ_norm']:.4e} "
          f"({g['max_p_circ_over_sqrt_eps']:.4f} x sqrt(eps)); rows >1e-2*S: {g['n_rows_above_1e-2_S']}")
    for tag, v in jac.items():
        print(f"    {tag:24s} zero_grad_rows={v['n_exactly_zero_rows']} "
              f"sigma_min={v['sigma_min_full']:.4e} full_rank={v['full_row_rank']}")
    print(f"\n[R1] requirements: {json.dumps(g['requirements'], indent=1)}")
    print(f"[R1] ALL REQUIREMENTS MET: {g['all_requirements_met']}")
    print(f'[R1] report -> {out}')


if __name__ == '__main__':
    main()
