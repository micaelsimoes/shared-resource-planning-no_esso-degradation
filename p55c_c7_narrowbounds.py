"""
Stage P5.5-C7 (diagnosis, fourth level) -- test the near-fixed-variable
hypothesis for the QCP dual failure.

Production boxes every quantity it wants to be "effectively zero" into
[0, EQUALITY_TOLERANCE] = [0, 1e-5] rather than fixing it: the flexibility
variables of non-flexible loads, curtailment when disabled, and so on.  A
log-barrier term on a box of width 1e-5 has enormous curvature, and a barrier
method has to drive those variables into a 1e-5-wide corridor while the rest of
the model lives at order 1.

That is the one structural difference that tracks the observed failure: the TSO
network has 3 loads and certifies cleanly, the DSO networks have 33 loads each
and never produce a QCP dual.  This script counts those variables and measures
what happens when they are fixed instead of boxed.

FIXING THEM IS NOT A PROPOSED CHANGE.  It removes up to 1e-5 of slack per
variable and so slightly SHRINKS the feasible set, which would break the
outer-relaxation direction.  It is run here only to confirm or refute the cause.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55c_c7_narrowbounds.py
"""

import io, json, math, os, sys
from collections import Counter
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
NARROW = 1e-4

CASES = [
    ('dso5 2025 Winter', dict(only_agents=['DSO5'], only_years=[2025], only_days=['Winter'])),
    ('tso 2025 Winter', dict(only_agents=['TSO'], only_years=[2025], only_days=['Winter'])),
    ('tso+dso5 2025 Winter', dict(only_agents=['TSO', 'DSO5'], only_years=[2025],
                                  only_days=['Winter'])),
    ('all agents 2025 Winter', dict(only_years=[2025], only_days=['Winter'])),
]


def census(parent):
    counts = Counter()
    for var in parent.component_objects(pe.Var, active=None, descend_into=True):
        for index in var:
            data = var[index]
            lo, hi = data.bounds
            if lo is not None and hi is not None and 0.0 < (hi - lo) <= NARROW:
                counts[var.local_name] += 1
    return dict(counts), sum(counts.values())


def fix_narrow(parent):
    n = 0
    for var in parent.component_objects(pe.Var, active=None, descend_into=True):
        for index in var:
            data = var[index]
            lo, hi = data.bounds
            if lo is not None and hi is not None and 0.0 < (hi - lo) <= NARROW:
                data.fix(lo)
                n += 1
    return n


def probe(parent, label, tag, report_case):
    with redirect_stdout(io.StringIO()) as log:
        try:
            opt, _ = solve(parent, extra={})
            failure = None
        except Exception as exc:
            opt, failure = None, f'{type(exc).__name__}: {exc}'
    text = log.getvalue()
    if opt is None:
        record = {'variant': tag, 'error': failure}
    else:
        duals_ok, dual_error = True, None
        try:
            for index in parent.capacity_fix_s:
                opt.get_linear_constraint_attr(parent.capacity_fix_s[index], 'Pi')
        except Exception as exc:
            duals_ok, dual_error = False, f'{type(exc).__name__}: {exc}'
        bound, value = _grb(opt, 'ObjBound'), _grb(opt, 'ObjVal')
        record = {
            'variant': tag, 'gurobi_status': _grb(opt, 'Status'),
            'ObjVal': value, 'ObjBound': bound,
            'relative_gap': (abs(value - bound) / max(abs(value), 1e-12)
                             if value is not None and bound is not None
                             and math.isfinite(bound) else None),
            'BarIterCount': _grb(opt, 'BarIterCount'),
            'qcp_dual_warning': 'failed to compute QCP dual' in text,
            'duals_readable': duals_ok, 'dual_error': dual_error,
        }
    report_case['variants'].append(record)
    print(f"    {tag:16s} status={record.get('gurobi_status')} "
          f"ObjVal={record.get('ObjVal')} ObjBound={record.get('ObjBound')} "
          f"qcp_warn={record.get('qcp_dual_warning')} "
          f"duals={record.get('duals_readable')}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C7 narrow bounds', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C7nb] ABORTED\n{error}')
        sys.exit(1)
    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    report = {'stage': 'P5.5-C7 narrow bounds', 'provenance': provenance,
              'narrow_width_threshold': NARROW,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'cases': []}

    for label, restriction in CASES:
        parent = build_centralized_relaxation(planning, candidate, **restriction)
        by_name, total = census(parent)
        size = model_size(parent)
        case = {'case': label, 'model_size': size,
                'narrow_variables_total': total,
                'narrow_variables_by_name': by_name,
                'narrow_fraction': total / max(size['variables'], 1),
                'variants': []}
        print(f"\n[C7nb] === {label}: {size['variables']} vars, "
              f"{total} boxed within {NARROW:g} "
              f"({100 * total / max(size['variables'], 1):.1f}%)", flush=True)
        for name, count in sorted(by_name.items(), key=lambda kv: -kv[1])[:6]:
            print(f"        {name:32s} {count}")
        probe(parent, label, 'boxed', case)
        fixed = fix_narrow(parent)
        case['n_fixed'] = fixed
        probe(parent, label, 'fixed', case)
        report['cases'].append(case)

    out = os.path.join(OUT_DIR, 'p55c_c7_narrowbounds.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[C7nb] report -> {out}')


if __name__ == '__main__':
    main()
