"""
Stage P5.4-R4 -- verify Gurobi conic capability in the canonical environment.

Corrects the P5.5-A / A11 finding that "no SOCP-capable solver is available",
which was true of `srp_env` and wrong about the canonical environment.

Solves a small convex QCP/SOCP with a known analytic optimum and checks:
  * primal optimality
  * QCPDual = 1 accepted
  * linear-constraint dual
  * quadratic/conic-constraint dual
  * reported optimality gap

Nothing is installed. Also confirms the Pyomo interfaces, since the convex SMOPF
prototype will be written in Pyomo.

    /opt/anaconda3/envs/opf_env_py311/bin/python p54r_gurobi_conic_check.py
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54R_GUROBI')


def gurobipy_socp():
    """min t  s.t.  x + y >= 3,  x^2 + y^2 <= t^2,  t >= 0.

    Optimum: x = y = 3/2, t = 3/sqrt(2) = 2.1213203435596424.
    """
    import gurobipy as gp
    from gurobipy import GRB

    out = {'gurobipy_version': '.'.join(str(v) for v in gp.gurobi.version())}
    model = gp.Model('socp_check')
    model.Params.OutputFlag = 0
    model.Params.QCPDual = 1
    out['QCPDual_accepted'] = model.Params.QCPDual == 1

    x = model.addVar(lb=-10, ub=10, name='x')
    y = model.addVar(lb=-10, ub=10, name='y')
    t = model.addVar(lb=0, ub=10, name='t')
    lin = model.addConstr(x + y >= 3, name='lin')
    soc = model.addQConstr(x * x + y * y <= t * t, name='soc')
    model.setObjective(t, GRB.MINIMIZE)
    model.optimize()

    analytic = 3.0 / (2.0 ** 0.5)
    out.update({
        'status': int(model.Status),
        'status_is_optimal': model.Status == GRB.OPTIMAL,
        'objective': model.ObjVal,
        'analytic_optimum': analytic,
        'absolute_error': abs(model.ObjVal - analytic),
        'relative_error': abs(model.ObjVal - analytic) / analytic,
        'x': x.X, 'y': y.X, 't': t.X,
        'linear_constraint_dual_Pi': lin.Pi,
        'quadratic_constraint_dual_QCPi': soc.QCPi,
        'linear_dual_available': lin.Pi is not None,
        'quadratic_dual_available': soc.QCPi is not None,
        'is_qcp': bool(model.IsQCP),
        'num_qconstrs': model.NumQConstrs,
    })
    try:
        out['MIPGap_reported'] = model.MIPGap
    except Exception:
        out['MIPGap_reported'] = None
    # For a continuous convex QCP the meaningful gap is primal vs dual objective.
    try:
        out['ObjBound'] = model.ObjBound
        out['optimality_gap_abs'] = abs(model.ObjVal - model.ObjBound)
    except Exception:
        out['ObjBound'] = None
        out['optimality_gap_abs'] = None
    out['barrier_iterations'] = getattr(model, 'BarIterCount', None)
    model.dispose()
    return out


def licence_info():
    path = os.path.expanduser('~/gurobi.lic')
    fields = {}
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    if key.upper() not in ('KEY', 'PASSWORD'):
                        fields[key.upper()] = value
    return {'path': path, 'exists': os.path.exists(path),
            'type': fields.get('TYPE'), 'expiration': fields.get('EXPIRATION'),
            'licence_id': fields.get('LICENSEID'),
            'licence_version': fields.get('VERSION'),
            'hostname': fields.get('HOSTNAME')}


def pyomo_interfaces():
    import pyomo
    import pyomo.environ as pe
    out = {'pyomo_version': pyomo.__version__, 'interfaces': {}}
    for name in ('gurobi', 'gurobi_direct', 'gurobi_persistent'):
        try:
            out['interfaces'][name] = bool(
                pe.SolverFactory(name).available(exception_flag=False))
        except Exception as error:
            out['interfaces'][name] = f'error: {type(error).__name__}'
    try:
        from pyomo.core.kernel.conic import quadratic, rotated_quadratic  # noqa: F401
        out['pyomo_kernel_conic'] = True
    except Exception:
        out['pyomo_kernel_conic'] = False
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-R4', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'sys_executable': sys.executable,
              'note': 'verification only; nothing was installed'}
    report['licence'] = licence_info()
    report['pyomo'] = pyomo_interfaces()
    try:
        report['socp_check'] = gurobipy_socp()
        report['socp_check']['all_checks_passed'] = all([
            report['socp_check']['status_is_optimal'],
            report['socp_check']['QCPDual_accepted'],
            report['socp_check']['linear_dual_available'],
            report['socp_check']['quadratic_dual_available'],
            report['socp_check']['relative_error'] < 1e-6,
        ])
    except Exception as error:
        report['socp_check'] = {'error': f'{type(error).__name__}: {error}',
                                'all_checks_passed': False}

    out = os.path.join(OUT_DIR, 'p54r_gurobi_check.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    s = report['socp_check']
    print('[R4] Gurobi conic verification (canonical environment)')
    print(f"    gurobipy         : {s.get('gurobipy_version')}")
    lic = report['licence']
    print(f"    licence          : {lic['type']} id={lic['licence_id']} "
          f"version={lic['licence_version']} expires={lic['expiration']}")
    print(f"    Pyomo {report['pyomo']['pyomo_version']} interfaces: {report['pyomo']['interfaces']}")
    print(f"    pyomo.core.kernel.conic available: {report['pyomo']['pyomo_kernel_conic']}")
    print()
    print(f"    status optimal   : {s.get('status_is_optimal')}")
    print(f"    objective        : {s.get('objective'):.12f}")
    print(f"    analytic optimum : {s.get('analytic_optimum'):.12f}")
    print(f"    relative error   : {s.get('relative_error'):.3e}")
    print(f"    QCPDual accepted : {s.get('QCPDual_accepted')}")
    print(f"    linear dual  (Pi): {s.get('linear_constraint_dual_Pi')}")
    print(f"    conic dual (QCPi): {s.get('quadratic_constraint_dual_QCPi')}")
    print(f"    ObjBound         : {s.get('ObjBound')}  |gap|={s.get('optimality_gap_abs')}")
    print(f"    barrier iters    : {s.get('barrier_iterations')}")
    print(f"\n[R4] ALL CHECKS PASSED: {s.get('all_checks_passed')}")
    print(f'[R4] report -> {out}')


if __name__ == '__main__':
    main()
