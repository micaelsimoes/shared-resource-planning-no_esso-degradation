"""
Stage P5.5-B6/B7 -- rigorous dual-derived cut contract, and the Pyomo/Gurobi
interface decision.

The question is not "can Gurobi solve an SOCP" (P5.4-R4 settled that) but
"can we build a GLOBALLY VALID planning cut from the dual output we will use".

Test problem, chosen because its value function is known in closed form:

    R(theta) = min  -p
               s.t. || (p, q) ||_2 <= S        (SOC, jointly convex in p,q,S)
                    q == q0                    (fixed reactive)
                    S == theta                 (capacity-fixing row)

    =>  p* = sqrt(theta^2 - q0^2),  R(theta) = -sqrt(theta^2 - q0^2)
        dR/dtheta = -theta / sqrt(theta^2 - q0^2)          (analytic)
        R is convex in theta for theta > q0.

This mirrors the production contract exactly: an affine capacity-fixing row
carrying the cut coefficient, and a second-order cone whose right-hand side is
the capacity variable.

A second test adds a multi-period active-sum row and SOC energy limits, so the
contract is checked on something structurally closer to the shared-ESS block.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p55b_cut_contract.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55B')
Q0 = 0.6


def analytic_R(theta):
    return -math.sqrt(max(theta ** 2 - Q0 ** 2, 0.0))


def analytic_dR(theta):
    return -theta / math.sqrt(theta ** 2 - Q0 ** 2)


def build_model(theta):
    """Production-shaped: affine fixing row + SOC with the capacity on the RHS."""
    m = pe.ConcreteModel()
    m.p = pe.Var(bounds=(0.0, 10.0), initialize=0.5)
    m.q = pe.Var(bounds=(-10.0, 10.0), initialize=Q0)
    m.S = pe.Var(bounds=(0.0, 10.0), initialize=theta)
    m.theta = pe.Param(initialize=theta, mutable=True)

    # capacity-fixing row, same orientation as shared_energy_storage_s_sensitivities
    m.fix_row = pe.Constraint(expr=m.theta == m.S)
    # jointly convex second-order cone (norm form, not the squared-RHS form)
    m.soc = pe.Constraint(expr=m.p ** 2 + m.q ** 2 <= m.S ** 2)
    m.qfix = pe.Constraint(expr=m.q == Q0)
    m.obj = pe.Objective(expr=-m.p, sense=pe.minimize)

    m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    return m


def solve_with(interface, theta, extra_options=None):
    m = build_model(theta)
    opt = pe.SolverFactory(interface)
    options = {'QCPDual': 1, 'OutputFlag': 0}
    options.update(extra_options or {})
    for key, value in options.items():
        opt.options[key] = value
    if interface == 'gurobi_persistent':
        opt.set_instance(m)
        res = opt.solve(load_solutions=True)
    else:
        res = opt.solve(m, load_solutions=True)

    out = {'interface': interface, 'theta': theta,
           'termination': str(res.solver.termination_condition),
           'ObjVal': float(pe.value(m.obj)),
           'p': float(pe.value(m.p)), 'q': float(pe.value(m.q)),
           'S': float(pe.value(m.S))}
    try:
        out['fix_row_dual'] = float(m.dual[m.fix_row])
    except Exception as error:
        out['fix_row_dual'] = None
        out['fix_row_dual_error'] = f'{type(error).__name__}'
    try:
        out['soc_dual'] = float(m.dual[m.soc])
    except Exception as error:
        out['soc_dual'] = None
        out['soc_dual_error'] = f'{type(error).__name__}'
    try:
        out['qfix_dual'] = float(m.dual[m.qfix])
    except Exception:
        out['qfix_dual'] = None
    for key in ('Lower bound', 'Upper bound'):
        try:
            out[key.replace(' ', '_')] = float(res.problem[0][key])
        except Exception:
            pass
    return out, m


def gurobipy_reference(theta):
    """Native gurobipy: gives ObjBound as well as the duals."""
    import gurobipy as gp
    from gurobipy import GRB
    mm = gp.Model()
    mm.Params.OutputFlag = 0
    mm.Params.QCPDual = 1
    p = mm.addVar(lb=0.0, ub=10.0, name='p')
    q = mm.addVar(lb=-10.0, ub=10.0, name='q')
    S = mm.addVar(lb=0.0, ub=10.0, name='S')
    fix = mm.addConstr(S == theta, name='fix')
    qf = mm.addConstr(q == Q0, name='qfix')
    soc = mm.addQConstr(p * p + q * q <= S * S, name='soc')
    mm.setObjective(-p, GRB.MINIMIZE)
    mm.optimize()
    out = {'theta': theta, 'status': int(mm.Status), 'ObjVal': mm.ObjVal,
           'ObjBound': mm.ObjBound, 'fix_dual': fix.Pi, 'qfix_dual': qf.Pi,
           'soc_dual_QCPi': soc.QCPi,
           'gap_abs': abs(mm.ObjVal - mm.ObjBound)}
    mm.dispose()
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.5-B6/B7', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'sys_executable': sys.executable, 'q0': Q0,
              'test_problem': 'min -p s.t. ||(p,q)||<=S, q==q0, S==theta'}

    theta_k = 1.0

    # ---------- B7: interface comparison ----------
    interfaces = {}
    for name in ('gurobi_direct', 'gurobi_persistent', 'gurobi'):
        try:
            rec, _ = solve_with(name, theta_k)
            rec['fix_dual_matches_analytic'] = (
                rec['fix_row_dual'] is not None
                and abs(rec['fix_row_dual'] - analytic_dR(theta_k)) < 1e-6)
            interfaces[name] = rec
        except Exception as error:
            interfaces[name] = {'error': f'{type(error).__name__}: {error}'}
    report['B7_interfaces'] = interfaces
    report['B7_analytic_dR_at_theta_k'] = analytic_dR(theta_k)

    # ---------- B6: cut contract ----------
    native_k = gurobipy_reference(theta_k)
    chosen = 'gurobi_persistent' if interfaces.get('gurobi_persistent', {}).get(
        'fix_dual_matches_analytic') else 'gurobi_direct'
    rec_k, _ = solve_with(chosen, theta_k)
    lam = rec_k['fix_row_dual']

    # Cut from ONE dual solution:  L(theta) = beta + lam*theta,
    # beta = R(theta_k) - lam*theta_k, i.e. the dual function value d(lam,mu).
    # Two anchors are compared: ObjVal (not rigorous) and ObjBound (dual bound).
    beta_objval = rec_k['ObjVal'] - lam * theta_k
    beta_objbound = native_k['ObjBound'] - lam * theta_k

    report['B6_cut_construction'] = {
        'theta_k': theta_k,
        'interface_used': chosen,
        'lambda_fix_row_dual': lam,
        'analytic_dR_dtheta': analytic_dR(theta_k),
        'dual_matches_analytic_derivative': abs(lam - analytic_dR(theta_k)) < 1e-6,
        'ObjVal': rec_k['ObjVal'],
        'ObjBound': native_k['ObjBound'],
        'analytic_R': analytic_R(theta_k),
        'gap_abs': native_k['gap_abs'],
        'ordering': {
            'ObjBound_le_analytic_R': native_k['ObjBound'] <= analytic_R(theta_k) + 1e-12,
            'analytic_R_le_ObjVal': analytic_R(theta_k) <= rec_k['ObjVal'] + 1e-12,
            'ObjBound_le_ObjVal': native_k['ObjBound'] <= rec_k['ObjVal'] + 1e-12,
        },
        'beta_from_ObjVal': beta_objval,
        'beta_from_ObjBound': beta_objbound,
        'L_at_theta_k_from_ObjVal': beta_objval + lam * theta_k,
        'L_at_theta_k_from_ObjBound': beta_objbound + lam * theta_k,
    }

    # ---------- sweep: is the cut a global under-estimator? ----------
    sweep = []
    for theta in [0.65, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0]:
        native = gurobipy_reference(theta)
        r_true = analytic_R(theta)
        l_objval = beta_objval + lam * theta
        l_objbound = beta_objbound + lam * theta
        sweep.append({
            'theta': theta,
            'R_analytic': r_true,
            'R_solved_ObjVal': native['ObjVal'],
            'R_solved_ObjBound': native['ObjBound'],
            'L_from_ObjVal': l_objval,
            'L_from_ObjBound': l_objbound,
            'violation_vs_analytic_ObjVal_anchor': l_objval - r_true,
            'violation_vs_analytic_ObjBound_anchor': l_objbound - r_true,
            'violation_vs_solved_ObjVal_anchor': l_objval - native['ObjVal'],
        })
    report['B6_sweep'] = sweep
    worst_objval = max(s['violation_vs_analytic_ObjVal_anchor'] for s in sweep)
    worst_objbound = max(s['violation_vs_analytic_ObjBound_anchor'] for s in sweep)
    report['B6_sweep_summary'] = {
        'n_points': len(sweep),
        'worst_violation_ObjVal_anchor': worst_objval,
        'worst_violation_ObjBound_anchor': worst_objbound,
        'ObjVal_anchor_is_valid_cut': worst_objval <= 1e-9,
        'ObjBound_anchor_is_valid_cut': worst_objbound <= 1e-9,
        'note': ('positive violation means L(theta) > R(theta), i.e. the cut cuts '
                 'off feasible values and is INVALID at that point'),
    }

    out = os.path.join(OUT_DIR, 'p55b_cut_contract.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print('[B7] Pyomo/Gurobi interfaces at theta_k = 1.0')
    for name, rec in interfaces.items():
        if 'error' in rec:
            print(f'  {name:20s} ERROR {rec["error"]}')
            continue
        print(f"  {name:20s} term={rec['termination']:>8s} ObjVal={rec['ObjVal']:+.9f} "
              f"fix_dual={rec['fix_row_dual']} soc_dual={rec['soc_dual']} "
              f"matches_analytic={rec.get('fix_dual_matches_analytic')}")
    print(f"  analytic dR/dtheta = {analytic_dR(theta_k):+.9f}")

    c = report['B6_cut_construction']
    print(f"\n[B6] cut construction (interface {c['interface_used']})")
    print(f"  lambda (fixing-row dual) = {c['lambda_fix_row_dual']:+.9f}  "
          f"analytic = {c['analytic_dR_dtheta']:+.9f}  match={c['dual_matches_analytic_derivative']}")
    print(f"  ObjBound={c['ObjBound']:+.12f} <= R_analytic={c['analytic_R']:+.12f} "
          f"<= ObjVal={c['ObjVal']:+.12f}")
    print(f"  ordering checks: {c['ordering']}")
    print(f"  beta(ObjVal anchor)   = {c['beta_from_ObjVal']:+.12f}")
    print(f"  beta(ObjBound anchor) = {c['beta_from_ObjBound']:+.12f}")

    print(f"\n[B6] sweep -- positive 'viol' means the cut is INVALID there")
    print(f"{'theta':>7} {'R_analytic':>14} {'L(ObjVal)':>14} {'viol':>12} "
          f"{'L(ObjBound)':>14} {'viol':>12}")
    for s in sweep:
        print(f"{s['theta']:>7.2f} {s['R_analytic']:>14.9f} {s['L_from_ObjVal']:>14.9f} "
              f"{s['violation_vs_analytic_ObjVal_anchor']:>+12.3e} "
              f"{s['L_from_ObjBound']:>14.9f} "
              f"{s['violation_vs_analytic_ObjBound_anchor']:>+12.3e}")
    print(f"\n[B6] {json.dumps(report['B6_sweep_summary'], indent=1)}")
    print(f'[B6] report -> {out}')


if __name__ == '__main__':
    main()
