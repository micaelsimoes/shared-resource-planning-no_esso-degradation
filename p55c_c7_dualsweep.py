"""
Stage P5.5-C7 (diagnosis, third level) -- can Gurobi produce QCP duals for this
model at ANY accuracy setting?

Every run so far, certified or not, printed

    Warning: failed to compute QCP dual solution due to inaccurate barrier
             solution.  Try decreasing BarQCPConvTol for more accuracy

while the barrier itself reported OPTIMAL and reproduced the same primal value
to five digits.  So the primal relaxation solves; what fails is dual extraction.
This sweeps the accuracy parameters Gurobi names, on one DSO block and one TSO
block, and -- the point of the exercise -- actually tries to READ the duals of
the capacity-fixing rows rather than trusting ObjBound to be finite.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55c_c7_dualsweep.py
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
    ('dso5 2025 Winter', dict(only_agents=['DSO5'], only_years=[2025], only_days=['Winter'])),
    ('tso 2025 Winter', dict(only_agents=['TSO'], only_years=[2025], only_days=['Winter'])),
    ('tso+dso5 2025 Winter', dict(only_agents=['TSO', 'DSO5'], only_years=[2025],
                                  only_days=['Winter'])),
]

SETTINGS = [
    ('default', {}),
    ('barqcp1e-8', {'BarQCPConvTol': 1e-8}),
    ('barqcp1e-10', {'BarQCPConvTol': 1e-10}),
    ('barqcp1e-10+focus3', {'BarQCPConvTol': 1e-10, 'NumericFocus': 3}),
    ('barqcp1e-12+focus3+bar1e-12', {'BarQCPConvTol': 1e-12, 'BarConvTol': 1e-12,
                                     'NumericFocus': 3}),
    ('focus3+homog+barqcp1e-10', {'BarQCPConvTol': 1e-10, 'NumericFocus': 3,
                                  'BarHomogeneous': 1}),
]


def read_duals(opt, parent):
    """Actually read the capacity-fixing-row duals; report what happens."""
    values, error = {}, None
    try:
        for index in parent.capacity_fix_s:
            values[f's|{index[0]}|{index[1]}'] = float(
                opt.get_linear_constraint_attr(parent.capacity_fix_s[index], 'Pi'))
        for index in parent.capacity_fix_e:
            values[f'e|{index[0]}|{index[1]}'] = float(
                opt.get_linear_constraint_attr(parent.capacity_fix_e[index], 'Pi'))
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    return values, error


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C7 dual sweep', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C7ds] ABORTED\n{error}')
        sys.exit(1)
    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    report = {'stage': 'P5.5-C7 dual sweep', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'cases': []}

    for label, restriction in CASES:
        parent = build_centralized_relaxation(planning, candidate, **restriction)
        entry = {'case': label, 'model_size': model_size(parent), 'settings': []}
        print(f"\n[C7ds] === {label}: {entry['model_size']}", flush=True)
        for sname, extra in SETTINGS:
            with redirect_stdout(io.StringIO()) as log:
                try:
                    opt, _ = solve(parent, extra=extra)
                    failure = None
                except Exception as exc:
                    opt, failure = None, f'{type(exc).__name__}: {exc}'
            text = log.getvalue()
            if opt is None:
                record = {'setting': sname, 'params': extra, 'error': failure}
            else:
                duals, dual_error = read_duals(opt, parent)
                bound = _grb(opt, 'ObjBound')
                value = _grb(opt, 'ObjVal')
                record = {
                    'setting': sname, 'params': extra,
                    'gurobi_status': _grb(opt, 'Status'),
                    'ObjVal': value, 'ObjBound': bound,
                    'relative_gap': (abs(value - bound) / max(abs(value), 1e-12)
                                     if value is not None and bound is not None
                                     and math.isfinite(bound) else None),
                    'BarIterCount': _grb(opt, 'BarIterCount'),
                    'Runtime': _grb(opt, 'Runtime'),
                    'qcp_dual_warning': 'failed to compute QCP dual' in text,
                    'duals_readable': dual_error is None,
                    'dual_error': dual_error,
                    'n_duals': len(duals),
                    'duals_all_zero': (bool(duals) and
                                       all(abs(v) < 1e-12 for v in duals.values())),
                    'duals': duals,
                }
            entry['settings'].append(record)
            print(f"    {sname:28s} status={record.get('gurobi_status')} "
                  f"ObjVal={record.get('ObjVal')} ObjBound={record.get('ObjBound')} "
                  f"qcp_warn={record.get('qcp_dual_warning')} "
                  f"duals_readable={record.get('duals_readable')} "
                  f"all_zero={record.get('duals_all_zero')}", flush=True)
        report['cases'].append(entry)

    out = os.path.join(OUT_DIR, 'p55c_c7_dualsweep.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[C7ds] report -> {out}')


if __name__ == '__main__':
    main()
