"""
Stage P5.6-A1 -- complete feasibility certificate for the D1 polished point.

P5.5-D1 established that the 48 nonlinear NETWORK SMOPFs are feasible at the
common interface values.  It did not audit the ORIGINAL nonlinear ESSO, so the
point could not be called a rigorous feasible upper-bound incumbent (P5.6-A0.3
withdraws that name).  This script closes the gap: it runs the full oracle
pipeline, which solves the original nonlinear ESSO with the investment and the
common coordinated P/Q schedules fixed, takes the PHYSICAL available capacities
from that solution, pushes them into every network copy, re-solves all 48
SMOPFs, and audits every constraint family on both sides.

Only if the complete original nonlinear system passes may the point be renamed
RIGOROUS FEASIBLE NONLINEAR UB INCUMBENT.

    /opt/anaconda3/envs/opf_env_py311/bin/python p56a_a1_certificate.py
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = O.OUT_DIR


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.6-A1 certificate', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[A1] ABORTED\n{error}')
        sys.exit(1)

    with redirect_stdout(io.StringIO()):
        base = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
    x = O.candidate_to_vector(planning, base)

    print('[A1] evaluating the canonical base candidate, cold start ...',
          flush=True)
    result = O.evaluate_planning_candidate(
        x, start_policy=O.START_COLD, eval_id='a1_certificate',
        use_cache=False, verbose=True)

    result['provenance'] = provenance
    result['stage'] = 'P5.6-A1'
    out = os.path.join(OUT_DIR, 'p56a_a1_certificate.json')
    with open(out, 'w') as handle:
        json.dump(result, handle, indent=1, default=str)

    print(f"\n[A1] status : {result['status']}")
    if 'admm' in result:
        a = result['admm']
        print(f"[A1] ADMM    : converged={a['converged']} cycles={a['cycles']} "
              f"runtime={a['runtime_s']:.1f}s")
        print(f"             gross={a.get('gross_operational_cost')} "
              f"net recourse={a.get('net_operational_recourse')}")
    if 'admm_coordination_residuals' in result:
        print('\n[A1] coordinated residuals, ADMM-converged (p.u.)')
        for k, v in result['admm_coordination_residuals'].items():
            print(f'      {k:22s} {v:.6e}')
    if 'coordination_residuals' in result:
        print('\n[A1] coordinated residuals, polished (p.u. / MVA / MVAh)')
        for k, v in result['coordination_residuals'].items():
            flag = '' if v < O.COORDINATION_TARGET else '   ABOVE TARGET'
            print(f'      {k:22s} {v:.6e}{flag}')
        print(f"      target                 {O.COORDINATION_TARGET:.0e}")
    if 'esso_audit' in result:
        e = result['esso_audit']
        print('\n[A1] ORIGINAL nonlinear ESSO audit')
        print(f"      max constraint violation        {e['max_violation']:.6e}"
              f"   (worst family {e['worst_family']})")
        print(f"      production feasibility violation {e['production_feasibility_violation']:.6e}"
              f"   tolerance {e['production_tolerance']:.0e}"
              f"   feasible={e['production_feasible']}")
        for node, entry in e['per_node'].items():
            worst = sorted(entry['families'].items(),
                           key=lambda kv: -kv[1]['max_violation'])[:3]
            print(f"      node {node}: max {entry['max_violation']:.3e}  "
                  + ', '.join(f'{n}={f["max_violation"]:.2e}' for n, f in worst))
    if 'network_audit' in result:
        n = result['network_audit']
        print('\n[A1] nonlinear NETWORK audit (48 SMOPFs)')
        print(f"      max constraint violation        {n['max_violation']:.6e}"
              f"   (worst block {n['worst_block']})")
        print(f"      max H1 complementarity violation {n['max_h1_complementarity_violation']:.6e}")
        print(f"      max converter-capability violation {n['max_converter_capability_violation']:.6e}")
    if result['status'] == O.STATUS_VALID:
        print('\n[A1] exact objective accounting')
        print(f"      gross operational cost      {result['gross_operational_cost']:.6f}")
        print(f"      actual physical salvage     {result['physical_salvage']:.6f}")
        print(f"      net operational recourse    {result['net_operational_recourse']:.6f}")
        print(f"      investment cost             {result['investment_cost']:.6f}")
        print(f"      TOTAL PLANNING OBJECTIVE    {result['total_objective']:.6f}")
        print('\n[A1] the point may be renamed RIGOROUS FEASIBLE NONLINEAR UB INCUMBENT')
    else:
        print(f"\n[A1] NOT certified: {result['status']}")
    print(f"\n[A1] wall clock {result['wall_clock_s']:.1f} s")
    print(f'[A1] report -> {out}')


if __name__ == '__main__':
    main()
