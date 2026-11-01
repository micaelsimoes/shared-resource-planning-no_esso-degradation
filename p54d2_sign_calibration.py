"""
Stage P5.4-D2.4 (a) -- calibrate the Pyomo/IPOPT multiplier sign convention on
trivial parametric NLPs, before using it on the real model.

The envelope derivative of a parametric NLP needs the exact sign with which the
`dual` suffix (equality/inequality rows) and the `ipopt_zL_out` /
`ipopt_zU_out` suffixes (variable bounds) enter. That convention is asserted
nowhere in this repo, so it is measured here against problems whose exact
derivative is known analytically.

Three calibration problems, each solved with the SAME solver the production
path uses:

  P1  fixing row      min (x-5)^2  s.t.  x == c          Q(c) = (c-5)^2
  P2  upper bound     min (x-5)^2  s.t.  0 <= x <= u     Q(u) = (u-5)^2 for u<5
  P3  lower bound     min (x+5)^2  s.t.  l <= x <= 10    Q(l) = (l+5)^2 for l>-5

    python p54d2_sign_calibration.py
"""

import json
import os
import sys

import pyomo.environ as pe
import pyomo.opt as po

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D2')


def solver_from_production():
    """Use the same IPOPT executable the production solve path uses, read from
    the real production params object rather than hardcoded."""
    import io
    from contextlib import redirect_stdout
    from shared_resources_planning import SharedResourcesPlanning as SRP
    console = io.StringIO()
    with redirect_stdout(console):
        p = SRP('data/SRP1', 'SRP1.json')
        p.read_planning_problem()
    sp = p.distribution_networks[9].params.solver_params
    return sp.solver, sp.solver_path


def attach_suffixes(model):
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)
    model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)


def p1_fixing_row(solver, c):
    """min (x-5)^2 s.t. x == c.  Q(c) = (c-5)^2,  dQ/dc = 2(c-5)."""
    m = pe.ConcreteModel()
    m.x = pe.Var(initialize=0.0)
    m.c = pe.Param(initialize=c, mutable=True)
    # written in the same orientation as sess_s_sensitivities: param == var
    m.fix_row = pe.Constraint(expr=m.c == m.x)
    m.obj = pe.Objective(expr=(m.x - 5.0) ** 2, sense=pe.minimize)
    attach_suffixes(m)
    solver.solve(m, tee=False)
    return {'c': c, 'x': float(pe.value(m.x)), 'Q': float(pe.value(m.obj)),
            'analytic_dQ_dc': 2.0 * (c - 5.0),
            'dual_fix_row': float(m.dual.get(m.fix_row))}


def p2_upper_bound(solver, u):
    """min (x-5)^2 s.t. 0 <= x <= u, u < 5 so the UPPER bound is active.
    Q(u) = (u-5)^2, dQ/du = 2(u-5) < 0."""
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(0.0, u), initialize=0.0)
    m.obj = pe.Objective(expr=(m.x - 5.0) ** 2, sense=pe.minimize)
    attach_suffixes(m)
    solver.solve(m, tee=False)
    return {'u': u, 'x': float(pe.value(m.x)), 'Q': float(pe.value(m.obj)),
            'analytic_dQ_du': 2.0 * (u - 5.0),
            'zU': float(m.ipopt_zU_out.get(m.x, 0.0)),
            'zL': float(m.ipopt_zL_out.get(m.x, 0.0))}


def p3_lower_bound(solver, l):
    """min (x+5)^2 s.t. l <= x <= 10, l > -5 so the LOWER bound is active.
    Q(l) = (l+5)^2, dQ/dl = 2(l+5) > 0."""
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(l, 10.0), initialize=10.0)
    m.obj = pe.Objective(expr=(m.x + 5.0) ** 2, sense=pe.minimize)
    attach_suffixes(m)
    solver.solve(m, tee=False)
    return {'l': l, 'x': float(pe.value(m.x)), 'Q': float(pe.value(m.obj)),
            'analytic_dQ_dl': 2.0 * (l + 5.0),
            'zU': float(m.ipopt_zU_out.get(m.x, 0.0)),
            'zL': float(m.ipopt_zL_out.get(m.x, 0.0))}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    name, path = solver_from_production()
    solver = po.SolverFactory(name, executable=path)
    print(f'[D2.4a] solver = {name} @ {path}')

    report = {'stage': 'P5.4-D2.4a sign calibration',
              'solver': name, 'solver_path': path}

    report['P1_fixing_row'] = [p1_fixing_row(solver, c) for c in (2.0, 3.0, 7.0, 8.0)]
    report['P2_upper_bound'] = [p2_upper_bound(solver, u) for u in (1.0, 2.0, 3.0)]
    report['P3_lower_bound'] = [p3_lower_bound(solver, l) for l in (-2.0, 0.0, 2.0)]

    # --- infer the conventions ---
    p1 = report['P1_fixing_row']
    ratios = [r['dual_fix_row'] / r['analytic_dQ_dc'] for r in p1
              if abs(r['analytic_dQ_dc']) > 1e-9]
    report['fixing_row_convention'] = {
        'ratios_dual_over_analytic': ratios,
        'all_close_to_plus_1': all(abs(r - 1.0) < 1e-6 for r in ratios),
        'all_close_to_minus_1': all(abs(r + 1.0) < 1e-6 for r in ratios),
    }

    p2 = report['P2_upper_bound']
    r2 = [r['zU'] / r['analytic_dQ_du'] for r in p2 if abs(r['analytic_dQ_du']) > 1e-9]
    report['upper_bound_convention'] = {
        'ratios_zU_over_analytic': r2,
        'dQ_du_equals_plus_zU': all(abs(r - 1.0) < 1e-6 for r in r2),
        'dQ_du_equals_minus_zU': all(abs(r + 1.0) < 1e-6 for r in r2),
    }

    p3 = report['P3_lower_bound']
    r3 = [r['zL'] / r['analytic_dQ_dl'] for r in p3 if abs(r['analytic_dQ_dl']) > 1e-9]
    report['lower_bound_convention'] = {
        'ratios_zL_over_analytic': r3,
        'dQ_dl_equals_plus_zL': all(abs(r - 1.0) < 1e-6 for r in r3),
        'dQ_dl_equals_minus_zL': all(abs(r + 1.0) < 1e-6 for r in r3),
    }

    with open(os.path.join(OUT_DIR, 'p54d2_sign_calibration.json'), 'w') as h:
        json.dump(report, h, indent=1, default=str)

    print('\n  P1 fixing row (param == var), analytic dQ/dc vs dual:')
    for r in p1:
        print(f"    c={r['c']:5.1f} x={r['x']:8.4f} Q={r['Q']:9.4f} "
              f"analytic={r['analytic_dQ_dc']:+9.4f} dual={r['dual_fix_row']:+9.4f} "
              f"ratio={r['dual_fix_row']/r['analytic_dQ_dc']:+.6f}")
    print(f"  -> dual == +dQ/dc: {report['fixing_row_convention']['all_close_to_plus_1']}; "
          f"dual == -dQ/dc: {report['fixing_row_convention']['all_close_to_minus_1']}")

    print('\n  P2 active UPPER bound, analytic dQ/du vs zU:')
    for r in p2:
        print(f"    u={r['u']:5.1f} x={r['x']:8.4f} Q={r['Q']:9.4f} "
              f"analytic={r['analytic_dQ_du']:+9.4f} zU={r['zU']:+9.4f} zL={r['zL']:+9.4f} "
              f"ratio={r['zU']/r['analytic_dQ_du']:+.6f}")
    print(f"  -> dQ/du == +zU: {report['upper_bound_convention']['dQ_du_equals_plus_zU']}; "
          f"== -zU: {report['upper_bound_convention']['dQ_du_equals_minus_zU']}")

    print('\n  P3 active LOWER bound, analytic dQ/dl vs zL:')
    for r in p3:
        print(f"    l={r['l']:5.1f} x={r['x']:8.4f} Q={r['Q']:9.4f} "
              f"analytic={r['analytic_dQ_dl']:+9.4f} zU={r['zU']:+9.4f} zL={r['zL']:+9.4f} "
              f"ratio={r['zL']/r['analytic_dQ_dl']:+.6f}")
    print(f"  -> dQ/dl == +zL: {report['lower_bound_convention']['dQ_dl_equals_plus_zL']}; "
          f"== -zL: {report['lower_bound_convention']['dQ_dl_equals_minus_zL']}")


if __name__ == '__main__':
    main()
