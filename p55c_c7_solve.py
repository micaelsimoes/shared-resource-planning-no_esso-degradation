"""
Stage P5.5-C7/C8/C9 -- solve the centralized convex lower-bound oracle on the
canonical positive-bootstrap base candidate, then measure how tight it is and
what it is worth as a lower bound.

C7  Solve with gurobi_persistent, QCPDual=1.  NonConvex is pinned to 0 so that
    Gurobi REFUSES the model unless every quadratic row is recognised as convex
    or conic -- that refusal is the check, not a nuisance.
C8  Tightness diagnostics: AC rank gap, OLTC rank gap, TSO cycle consistency via
    atan2, ESS simultaneous circulation, ESSO available/rated energy ratio.
C9  LB_rec(x) = ObjBound_R(x) - V_salvage_max(x) against the canonical nonlinear
    values.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55c_c7_solve.py
"""

import io
import json
import math
import os
import pickle
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

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55C')

# canonical nonlinear reference values (P5.4-R / P5.5-A)
Q_NONLINEAR_COLD = 838496830.813414
Q_NONLINEAR_BEST_RECOVERED = 836586463.43


def salvage_max(traces, candidate):
    """V_salvage_max(x) = sum_{node,cohort} gamma * E_investment, from C0.1."""
    total = 0.0
    terms = []
    for cohort in traces['C0_1_salvage']['cohorts']:
        node = cohort['node_id']
        year = cohort['investment_year']
        gamma = float(cohort['gamma'])
        e_inv = abs(candidate['investment'][node][year]['e'])
        total += gamma * e_inv
        terms.append({'node_id': node, 'investment_year': year, 'gamma': gamma,
                      'e_investment': e_inv, 'contribution': gamma * e_inv})
    return total, terms


# Numerics ladder.  The model is badly scaled by construction -- production's
# penalty constants span 1e-1 to 1e6 and the year/day weights multiply them by
# ~4.6e2 -- so the default barrier stalls.  None of these settings change the
# feasible set or the objective; they only change how the barrier is run, and
# the ladder stops at the first rung that returns a usable dual bound.
# Each rung is (name, objective_scale, gurobi parameters).  `objective_scale`
# divides the objective by a constant; the feasible set is untouched and the
# bound and duals are scaled back exactly, so this changes nothing about what is
# being bounded.
NUMERICS_LADDER = [
    ('unscaled', 1.0, {}),
    ('unscaled+homogeneous', 1.0, {'BarHomogeneous': 1}),
    ('scaled1e6', 1e6, {}),
    ('scaled1e6+homogeneous', 1e6, {'BarHomogeneous': 1}),
    ('scaled1e6+homogeneous+focus', 1e6, {'BarHomogeneous': 1, 'NumericFocus': 3}),
    ('scaled1e6+focus+tighttol', 1e6, {'NumericFocus': 3, 'BarQCPConvTol': 1e-9,
                                       'BarConvTol': 1e-9}),
    ('scaled1e6+nopresolve', 1e6, {'BarHomogeneous': 1, 'NumericFocus': 3,
                                   'Presolve': 0}),
    ('scaled1e8+homogeneous+focus', 1e8, {'BarHomogeneous': 1, 'NumericFocus': 3}),
]


def solve(parent, extra=None, time_limit=None):
    opt = pe.SolverFactory('gurobi_persistent')
    opt.set_instance(parent)
    opt.set_gurobi_param('QCPDual', 1)
    opt.set_gurobi_param('NonConvex', 0)      # refuse anything not convex/conic
    for key, value in (extra or {}).items():
        opt.set_gurobi_param(key, value)
    if time_limit:
        opt.set_gurobi_param('TimeLimit', time_limit)
    result = opt.solve(tee=True, load_solutions=True, save_results=False)
    return opt, result


def _grb(opt, attr, default=None):
    try:
        return getattr(opt._solver_model, attr)
    except Exception:
        return default


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C7 solve', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C7] ABORTED\n{error}')
        sys.exit(1)

    with open(os.path.join(OUT_DIR, 'p55c_c0_traces.json')) as handle:
        traces = json.load(handle)

    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    print('[C7] building the centralized convex relaxation ...', flush=True)
    parent = build_centralized_relaxation(planning, candidate)
    size = model_size(parent)
    print(f"[C7] {size['ac_blocks']} AC blocks, {size['variables']} variables, "
          f"{size['constraints']} constraints", flush=True)

    print('[C7] solving with gurobi_persistent (QCPDual=1, NonConvex=0) ...', flush=True)
    attempts = []
    opt = result = None
    scale = 1.0
    for rung, rung_scale, extra in NUMERICS_LADDER:
        print(f'\n[C7] ===== numerics rung: {rung} scale={rung_scale:g} {extra} =====',
              flush=True)
        if rung_scale != scale:
            parent = build_centralized_relaxation(planning, candidate,
                                                  objective_scale=rung_scale)
            scale = rung_scale
        try:
            opt, result = solve(parent, extra=extra)
        except Exception as error:
            attempts.append({'rung': rung, 'objective_scale': rung_scale,
                             'params': extra,
                             'error': f'{type(error).__name__}: {error}'})
            print(f'[C7] rung {rung} raised {type(error).__name__}: {error}')
            continue
        bound = _grb(opt, 'ObjBound')
        value = _grb(opt, 'ObjVal')
        attempts.append({'rung': rung, 'objective_scale': rung_scale, 'params': extra,
                         'gurobi_status': _grb(opt, 'Status'),
                         'ObjVal_scaled': value, 'ObjBound_scaled': bound,
                         'ObjVal': None if value is None else value * rung_scale,
                         'ObjBound': None if bound is None else bound * rung_scale,
                         'BarIterCount': _grb(opt, 'BarIterCount'),
                         'Runtime': _grb(opt, 'Runtime')})
        if bound is not None and math.isfinite(bound):
            print(f'[C7] rung {rung} produced a finite dual bound: {bound * rung_scale}')
            break
        print(f'[C7] rung {rung} gave no finite dual bound')
    if opt is None:
        report = {'stage': 'P5.5-C7', 'provenance': provenance, 'model_size': size,
                  'solve_failed': True, 'attempts': attempts}
        with open(os.path.join(OUT_DIR, 'p55c_c7_solve.json'), 'w') as handle:
            json.dump(report, handle, indent=1, default=str)
        print('\n[C7] SOLVE FAILED on every numerics rung')
        sys.exit(2)

    solver = result.solver
    obj_val = _grb(opt, 'ObjVal')
    obj_bound = _grb(opt, 'ObjBound')
    if obj_val is not None:
        obj_val *= scale
    if obj_bound is not None:
        obj_bound *= scale
    report = {
        'stage': 'P5.5-C7', 'provenance': provenance,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model_size': size,
        'numerics_ladder': attempts,
        'objective_scale_used': scale,
        'solver': {
            'status': str(solver.status),
            'termination_condition': str(solver.termination_condition),
            'gurobi_status': _grb(opt, 'Status'),
            'ObjVal': obj_val,
            'ObjBound': obj_bound,
            'MIPGap_or_relative_gap': (
                abs(obj_val - obj_bound) / max(abs(obj_val), 1e-12)
                if obj_val is not None and obj_bound is not None else None),
            'BarIterCount': _grb(opt, 'BarIterCount'),
            'IterCount': _grb(opt, 'IterCount'),
            'Runtime': _grb(opt, 'Runtime'),
            'MaxVio': _grb(opt, 'MaxVio'),
            'ConstrResidual': _grb(opt, 'ConstrResidual'),
            'NumQConstrs': _grb(opt, 'NumQConstrs'),
            'NumConstrs': _grb(opt, 'NumConstrs'),
            'NumVars': _grb(opt, 'NumVars'),
        },
    }

    # dual availability on the capacity-fixing rows
    duals = {}
    try:
        for index in parent.capacity_fix_s:
            duals[f's|{index[0]}|{index[1]}'] = scale * float(
                opt.get_linear_constraint_attr(parent.capacity_fix_s[index], 'Pi'))
        for index in parent.capacity_fix_e:
            duals[f'e|{index[0]}|{index[1]}'] = scale * float(
                opt.get_linear_constraint_attr(parent.capacity_fix_e[index], 'Pi'))
        report['capacity_fixing_duals'] = duals
        report['duals_available'] = True
    except Exception as error:
        report['duals_available'] = False
        report['dual_error'] = f'{type(error).__name__}: {error}'

    # ---------------------------------------------------------------- C9
    v_salvage, salvage_terms = salvage_max(traces, candidate)
    lb_rec = (obj_bound - v_salvage) if obj_bound is not None else None
    report['C9'] = {
        'V_salvage_max': v_salvage,
        'salvage_terms': salvage_terms,
        'ObjBound_R': obj_bound,
        'LB_rec': lb_rec,
        'UB_best_cold': Q_NONLINEAR_COLD,
        'UB_best_recovered': Q_NONLINEAR_BEST_RECOVERED,
    }
    if lb_rec is not None:
        for name, ub in (('cold', Q_NONLINEAR_COLD),
                         ('best_recovered', Q_NONLINEAR_BEST_RECOVERED)):
            report['C9'][f'gap_abs_vs_{name}'] = ub - lb_rec
            report['C9'][f'gap_rel_vs_{name}'] = (ub - lb_rec) / abs(ub)

    out = os.path.join(OUT_DIR, 'p55c_c7_solve.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    # persist the solved primal state so C8/C10 do not have to re-solve
    state = {'objective': obj_val, 'bound': obj_bound, 'blocks': {}}
    for key, blk in parent.blocks.items():
        entry = {}
        for var in blk.component_objects(pe.Var, active=None, descend_into=False):
            entry[var.local_name] = {str(i): (None if var[i].value is None
                                              else float(var[i].value))
                                     for i in var}
        state['blocks']['|'.join(str(k) for k in key)] = entry
    for name in ('S_rated', 'E_rated', 'S_available', 'E_available'):
        var = getattr(parent, name)
        state[name] = {f'{i[0]}|{i[1]}': float(var[i].value) for i in var}
    with open(os.path.join(OUT_DIR, 'p55c_c7_state.pkl'), 'wb') as handle:
        pickle.dump(state, handle)

    print(f"\n[C7] status                 = {report['solver']['termination_condition']}")
    print(f"[C7] ObjVal                 = {obj_val}")
    print(f"[C7] ObjBound               = {obj_bound}")
    print(f"[C7] barrier iterations     = {report['solver']['BarIterCount']}")
    print(f"[C7] runtime (s)            = {report['solver']['Runtime']}")
    print(f"[C7] max primal violation   = {report['solver']['MaxVio']}")
    print(f"[C7] duals available        = {report['duals_available']}")
    print(f"\n[C9] V_salvage_max          = {v_salvage:.6f}")
    print(f"[C9] LB_rec                 = {lb_rec}")
    if lb_rec is not None:
        print(f"[C9] gap vs cold            = {report['C9']['gap_abs_vs_cold']:.6f} "
              f"({report['C9']['gap_rel_vs_cold'] * 100:.4f} %)")
        print(f"[C9] gap vs best recovered  = "
              f"{report['C9']['gap_abs_vs_best_recovered']:.6f} "
              f"({report['C9']['gap_rel_vs_best_recovered'] * 100:.4f} %)")
    print(f'\n[C7] report -> {out}')


if __name__ == '__main__':
    main()
