"""
Stage P5.5-C7 (remedy test) -- does WIDENING the near-fixed boxes restore the
dual certificate, and at what cost in bound quality?

`p55c_c7_narrowbounds.py` showed that the 1e-5-wide boxes production uses for
"effectively zero" quantities are what stops Gurobi computing a QCP dual: FIXING
them restored certification on tso+dso5.  Fixing shrinks the feasible set and so
is not admissible.  Widening enlarges it, which keeps every bound valid while
removing the same numerical obstruction.

Reported for each width: whether a dual certificate appears, the bound, and the
bound's distance from the unwidened primal value -- that distance is the price
paid for the certificate.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p55c_c7_widen.py
"""

import io, json, math, os, sys
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
    ('tso+dso5 2025 Winter', dict(only_agents=['TSO', 'DSO5'], only_years=[2025],
                                  only_days=['Winter'])),
    ('all agents 2025 Winter', dict(only_years=[2025], only_days=['Winter'])),
    ('all agents 2025 all days', dict(only_years=[2025])),
    ('full model', dict()),
]
WIDTHS = [None, 1e-4, 1e-3, 1e-2]
SETTINGS = [('default', {}), ('homogeneous', {'BarHomogeneous': 1})]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C7 widen', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C7w] ABORTED\n{error}')
        sys.exit(1)
    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    report = {'stage': 'P5.5-C7 widen', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'cases': []}

    for label, restriction in CASES:
        case = {'case': label, 'widths': []}
        print(f'\n[C7w] === {label}', flush=True)
        for width in WIDTHS:
            parent = build_centralized_relaxation(planning, candidate,
                                                  widen_narrow_bounds=width,
                                                  **restriction)
            size = model_size(parent)
            entry = {'width': width, 'model_size': size,
                     'n_widened': getattr(parent, 'widened_variables', 0),
                     'settings': []}
            for sname, extra in SETTINGS:
                with redirect_stdout(io.StringIO()) as log:
                    try:
                        opt, _ = solve(parent, extra=extra)
                        failure = None
                    except Exception as exc:
                        opt, failure = None, f'{type(exc).__name__}: {exc}'
                text = log.getvalue()
                if opt is None:
                    record = {'setting': sname, 'error': failure}
                else:
                    duals, dual_error = {}, None
                    try:
                        for index in parent.capacity_fix_s:
                            duals[f's|{index[0]}|{index[1]}'] = float(
                                opt.get_linear_constraint_attr(
                                    parent.capacity_fix_s[index], 'Pi'))
                        for index in parent.capacity_fix_e:
                            duals[f'e|{index[0]}|{index[1]}'] = float(
                                opt.get_linear_constraint_attr(
                                    parent.capacity_fix_e[index], 'Pi'))
                    except Exception as exc:
                        dual_error = f'{type(exc).__name__}: {exc}'
                    bound, value = _grb(opt, 'ObjBound'), _grb(opt, 'ObjVal')
                    record = {
                        'setting': sname, 'gurobi_status': _grb(opt, 'Status'),
                        'ObjVal': value, 'ObjBound': bound,
                        'certified': (bound is not None and math.isfinite(bound)
                                      and 'failed to compute QCP dual' not in text
                                      and dual_error is None),
                        'relative_gap': (abs(value - bound) / max(abs(value), 1e-12)
                                         if value is not None and bound is not None
                                         and math.isfinite(bound) else None),
                        'BarIterCount': _grb(opt, 'BarIterCount'),
                        'Runtime': _grb(opt, 'Runtime'),
                        'qcp_dual_warning': 'failed to compute QCP dual' in text,
                        'duals_readable': dual_error is None,
                        'dual_error': dual_error, 'duals': duals,
                    }
                entry['settings'].append(record)
                print(f"    width={str(width):8s} {sname:12s} "
                      f"status={record.get('gurobi_status')} "
                      f"ObjVal={record.get('ObjVal')} ObjBound={record.get('ObjBound')} "
                      f"CERT={record.get('certified')}", flush=True)
                if record.get('certified'):
                    break
            case['widths'].append(entry)
        report['cases'].append(case)

    out = os.path.join(OUT_DIR, 'p55c_c7_widen.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[C7w] report -> {out}')


if __name__ == '__main__':
    main()
