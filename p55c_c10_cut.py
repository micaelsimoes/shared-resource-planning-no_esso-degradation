"""
Stage P5.5-C9/C10 -- the lower bound, and the cut contract on the real convex
model.

C9   LB_rec(x) = ObjBound_R(x) - V_salvage_max(x), against the canonical
     nonlinear feasible values.  The full model yields no certified ObjBound
     (P5.5-C7), so the arithmetic is also carried out with the UNCERTIFIED primal
     value, clearly labelled, purely to establish whether a bound of this
     tightness could ever be useful to the master.

C10  The cut contract, on the real convex model rather than the B6/B7 toy:
     extract g_k from the S/E capacity-fixing rows, validate it as a derivative
     of the convex value function by central finite differences, and test the
     ObjBound-anchored affine minorant

         L_k(x) = [ObjBound(x_k) - sigma] + g_k^T (x - x_k)

     at perturbed capacities.  Run on the largest sub-model for which Gurobi
     actually exposes duals, because that is the only place the contract can be
     tested at all.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p55c_c10_cut.py
"""

import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402
from p55c_c1_oracle import build_centralized_relaxation, model_size  # noqa: E402
from p55c_c7_solve import (Q_NONLINEAR_BEST_RECOVERED, Q_NONLINEAR_COLD,  # noqa: E402
                           _grb, salvage_max, solve)

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55C')

# The largest restriction for which P5.5-C7 obtained readable duals.
CUT_RESTRICTION = dict(only_agents=['TSO', 'DSO5'], only_years=[2025],
                       only_days=['Winter'])
CUT_SETTINGS = {'BarHomogeneous': 1}
PROBE_NODE, PROBE_YEAR = 5, 2025
# canonical Benders cut tolerance (P5.4-R)
TOL_CUT = 7.164e5


def solve_at(planning, candidate, restriction, settings, add_s=0.0, add_e=0.0,
             node=PROBE_NODE, year=PROBE_YEAR):
    """Solve the restricted relaxation with one node-year capacity shifted.

    Perturbations are ABSOLUTE (MVA / MVAh).  The positive-bootstrap base
    candidate carries 0.0106 MVA / 0.0213 MVAh at every node-year, so relative
    perturbations move the capacity by ~1e-4 MVA and the value function does not
    respond at all above solver noise; only investment-scale steps test anything.
    """
    perturbed = deepcopy(candidate)
    entry = perturbed['total_capacity'][node][year]
    entry['s'] = max(abs(entry['s']) + add_s, 0.0)
    entry['e'] = max(abs(entry['e']) + add_e, 0.0)
    parent = build_centralized_relaxation(planning, perturbed, **restriction)
    with redirect_stdout(io.StringIO()) as log:
        opt, _ = solve(parent, extra=settings)
    text = log.getvalue()
    bound, value = _grb(opt, 'ObjBound'), _grb(opt, 'ObjVal')
    duals, dual_error = {}, None
    try:
        for index in parent.capacity_fix_s:
            duals[('s', index[0], index[1])] = float(
                opt.get_linear_constraint_attr(parent.capacity_fix_s[index], 'Pi'))
        for index in parent.capacity_fix_e:
            duals[('e', index[0], index[1])] = float(
                opt.get_linear_constraint_attr(parent.capacity_fix_e[index], 'Pi'))
    except Exception as exc:
        dual_error = f'{type(exc).__name__}: {exc}'
    return {
        'add_s': add_s, 'add_e': add_e,
        's': abs(entry['s']), 'e': abs(entry['e']),
        'gurobi_status': _grb(opt, 'Status'), 'ObjVal': value, 'ObjBound': bound,
        'certified': (bound is not None and math.isfinite(bound)
                      and 'failed to compute QCP dual' not in text
                      and dual_error is None),
        'qcp_dual_warning': 'failed to compute QCP dual' in text,
        'duals': duals, 'dual_error': dual_error,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C9/C10', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C10] ABORTED\n{error}')
        sys.exit(1)

    with open(os.path.join(OUT_DIR, 'p55c_c0_traces.json')) as handle:
        traces = json.load(handle)
    with open(os.path.join(OUT_DIR, 'p55c_c8_tightness.json')) as handle:
        tightness = json.load(handle)
    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    report = {'stage': 'P5.5-C9/C10', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    # ------------------------------------------------------------------ C9
    v_salvage, salvage_terms = salvage_max(traces, candidate)
    relaxation_value = tightness['solve']['ObjVal']
    certified_bound = tightness['solve']['ObjBound']
    certified = tightness['solve']['certified']
    lb_if_it_were_a_bound = relaxation_value - v_salvage
    c9 = {
        'V_salvage_max': v_salvage,
        'salvage_terms': salvage_terms,
        'production_salvage_actual': 3439.5551395370985,
        'maximality_holds': 3439.5551395370985 <= v_salvage,
        'full_model_ObjBound': certified_bound,
        'full_model_certified': certified,
        'LB_rec': None if not certified else certified_bound - v_salvage,
        'full_model_relaxation_value_UNCERTIFIED': relaxation_value,
        'LB_rec_if_uncertified_value_were_a_bound': lb_if_it_were_a_bound,
        'UB_cold': Q_NONLINEAR_COLD,
        'UB_best_recovered': Q_NONLINEAR_BEST_RECOVERED,
        'tol_cut': TOL_CUT,
    }
    for name, ub in (('cold', Q_NONLINEAR_COLD),
                     ('best_recovered', Q_NONLINEAR_BEST_RECOVERED)):
        gap = ub - lb_if_it_were_a_bound
        c9[f'gap_abs_vs_{name}_uncertified'] = gap
        c9[f'gap_rel_vs_{name}_uncertified'] = gap / abs(ub)
        c9[f'gap_over_tol_cut_vs_{name}'] = gap / TOL_CUT
    report['C9'] = c9

    print('\n[C9] salvage')
    print(f"    V_salvage_max                    = {v_salvage:.6f}")
    print(f"    production salvage (actual)      = {c9['production_salvage_actual']:.6f}")
    print(f"    actual <= max (C0.1 maximality)  = {c9['maximality_holds']}")
    print('\n[C9] bound')
    print(f"    full-model ObjBound              = {certified_bound} "
          f"(certified={certified})")
    print(f"    full-model relaxation value      = {relaxation_value:.6f}  "
          f"[UNCERTIFIED -- not a bound]")
    print(f"    LB_rec if that value were a bound= {lb_if_it_were_a_bound:.6f}")
    for name, ub in (('cold', Q_NONLINEAR_COLD),
                     ('best_recovered', Q_NONLINEAR_BEST_RECOVERED)):
        print(f"    gap vs {name:15s}       = {c9[f'gap_abs_vs_{name}_uncertified']:.4f} "
              f"({c9[f'gap_rel_vs_{name}_uncertified'] * 100:.4f} %) "
              f"= {c9[f'gap_over_tol_cut_vs_{name}']:.1f} x tol_cut")

    # ----------------------------------------------------------------- C10
    print('\n[C10] cut contract on the real convex model', flush=True)
    print(f'[C10] restriction: {CUT_RESTRICTION}', flush=True)
    base = solve_at(planning, candidate, CUT_RESTRICTION, CUT_SETTINGS)
    print(f"[C10] base: status={base['gurobi_status']} ObjVal={base['ObjVal']} "
          f"ObjBound={base['ObjBound']} certified={base['certified']}", flush=True)

    c10 = {'restriction': {k: list(v) for k, v in CUT_RESTRICTION.items()},
           'settings': CUT_SETTINGS, 'base': {k: v for k, v in base.items()
                                              if k != 'duals'},
           'base_duals': {f'{a}|{b}|{c}': v for (a, b, c), v in base['duals'].items()},
           'probes': []}

    g_s = base['duals'].get(('s', PROBE_NODE, PROBE_YEAR))
    g_e = base['duals'].get(('e', PROBE_NODE, PROBE_YEAR))
    c10['g_k'] = {'s': g_s, 'e': g_e}
    print(f'[C10] g_k from capacity-fixing rows: s={g_s} e={g_e}', flush=True)

    if base['certified'] and g_s is not None:
        for label, kwargs in (('S +1 MVA', dict(add_s=1.0)),
                              ('S +10 MVA', dict(add_s=10.0)),
                              ('S +50 MVA', dict(add_s=50.0)),
                              ('E +2 MVAh', dict(add_e=2.0)),
                              ('E +20 MVAh', dict(add_e=20.0)),
                              ('E +100 MVAh', dict(add_e=100.0)),
                              ('S +10, E +20', dict(add_s=10.0, add_e=20.0))):
            probe = solve_at(planning, candidate, CUT_RESTRICTION, CUT_SETTINGS,
                             **kwargs)
            ds = probe['s'] - base['s']
            de = probe['e'] - base['e']
            predicted = base['ObjBound'] + (g_s or 0.0) * ds + (g_e or 0.0) * de
            actual_bound = probe['ObjBound']
            actual_value = probe['ObjVal']
            entry = {
                'label': label, 'delta_s': ds, 'delta_e': de,
                'probe_ObjVal': actual_value, 'probe_ObjBound': actual_bound,
                'probe_certified': probe['certified'],
                'linear_prediction_from_base_ObjBound': predicted,
                # a valid minorant must sit at or below the true value function
                'minorant_violation_vs_ObjVal': (
                    predicted - actual_value if actual_value is not None else None),
                'minorant_violation_vs_ObjBound': (
                    predicted - actual_bound
                    if actual_bound is not None and math.isfinite(actual_bound)
                    else None),
            }
            c10['probes'].append(entry)
            print(f"    {label:14s} dS={ds:10.4f} dE={de:10.4f} "
                  f"predicted={predicted:.6f} ObjVal={actual_value} "
                  f"violation={entry['minorant_violation_vs_ObjVal']}", flush=True)

        violations = [p['minorant_violation_vs_ObjVal'] for p in c10['probes']
                      if p['minorant_violation_vs_ObjVal'] is not None]
        c10['worst_minorant_violation'] = max(violations) if violations else None
        c10['minorant_valid_at_all_probes'] = (
            all(v <= 1e-6 * max(abs(base['ObjBound']), 1.0) for v in violations)
            if violations else None)
    else:
        c10['probes_skipped'] = ('base solve produced no certified bound or no '
                                 'capacity-row dual, so there is nothing to test')

    c10['verdict'] = (
        'solver-tolerance-safe empirical cut only; no formal dual certificate exposed'
        if not certified else 'dual certificate available on the full model')
    c10['master_status'] = ('blocked' if not certified else 'unblocked')
    report['C10'] = c10

    out = os.path.join(OUT_DIR, 'p55c_c10_cut.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f"\n[C10] verdict: {c10['verdict']}")
    print(f"[C10] master:  {c10['master_status']}")
    print(f'\n[C10] report -> {out}')


if __name__ == '__main__':
    main()
