"""
Stage P5.5-C7 (diagnosis) -- locate the barrier's numerical trouble by solving
progressively larger sub-problems of the centralized convex relaxation.

The full model reaches primal ~ dual ~ 6.53e8 and then stalls without a dual
certificate.  That could be intrinsic to the formulation, or an artefact of size
and coupling.  Solving one block, then one year-day slice with its interface,
then one whole year, then everything, separates those two explanations.

A restricted model bounds a restricted problem.  None of the values here is a
bound on the planning recourse; they are solver-behaviour evidence only.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55c_c7_subsolves.py
"""

import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402
from p55c_c1_oracle import build_centralized_relaxation, model_size  # noqa: E402
from p55c_c7_solve import _grb, solve  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55C')

CASES = [
    ('dso5 alone, 2025 Winter', dict(only_agents=['DSO5'], only_years=[2025], only_days=['Winter'])),
    ('tso alone, 2025 Winter', dict(only_agents=['TSO'], only_years=[2025], only_days=['Winter'])),
    ('tso+dso5, 2025 Winter', dict(only_agents=['TSO', 'DSO5'], only_years=[2025], only_days=['Winter'])),
    ('all agents, 2025 Winter', dict(only_years=[2025], only_days=['Winter'])),
    ('all agents, 2025 all days', dict(only_years=[2025])),
    ('all agents, all years, Winter', dict(only_days=['Winter'])),
    ('full model', dict()),
]

VARIANTS = [
    ('default', {}),
    ('homogeneous', {'BarHomogeneous': 1}),
    ('homogeneous+focus', {'BarHomogeneous': 1, 'NumericFocus': 3}),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C7 subsolves', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C7d] ABORTED\n{error}')
        sys.exit(1)

    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    report = {'stage': 'P5.5-C7 subsolves', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'cases': []}

    for label, kwargs in CASES:
        parent = build_centralized_relaxation(planning, candidate, **kwargs)
        size = model_size(parent)
        entry = {'case': label, 'restriction': {k: list(v) for k, v in kwargs.items()},
                 'model_size': size, 'variants': []}
        print(f"\n[C7d] === {label}: {size['ac_blocks']} blocks, "
              f"{size['variables']} vars, {size['constraints']} cons", flush=True)
        for vname, extra in VARIANTS:
            with redirect_stdout(io.StringIO()) as log:
                try:
                    opt, _ = solve(parent, extra=extra)
                    failed = None
                except Exception as error:
                    opt, failed = None, f'{type(error).__name__}: {error}'
            text = log.getvalue()
            if opt is None:
                variant = {'variant': vname, 'params': extra, 'error': failed}
            else:
                bound = _grb(opt, 'ObjBound')
                value = _grb(opt, 'ObjVal')
                variant = {
                    'variant': vname, 'params': extra,
                    'gurobi_status': _grb(opt, 'Status'),
                    'ObjVal': value, 'ObjBound': bound,
                    'certified': bound is not None and math.isfinite(bound),
                    'BarIterCount': _grb(opt, 'BarIterCount'),
                    'Runtime': _grb(opt, 'Runtime'),
                    'MaxVio': _grb(opt, 'MaxVio'),
                    'qcp_dual_warning': 'failed to compute QCP dual' in text,
                    'numerical_trouble': 'Numerical trouble' in text,
                    'suboptimal': 'Sub-optimal termination' in text,
                }
            entry['variants'].append(variant)
            flag = ('CERTIFIED' if variant.get('certified') else
                    'no-bound' if 'error' not in variant else 'ERROR')
            print(f"    {vname:20s} {flag:10s} ObjVal={variant.get('ObjVal')} "
                  f"ObjBound={variant.get('ObjBound')} "
                  f"iters={variant.get('BarIterCount')}", flush=True)
            if variant.get('certified'):
                break
        report['cases'].append(entry)

    out = os.path.join(OUT_DIR, 'p55c_c7_subsolves.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[C7d] report -> {out}')


if __name__ == '__main__':
    main()
