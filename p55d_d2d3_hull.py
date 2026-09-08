"""
Stage P5.5-D2/D3 -- the continuous convex hull of the shared-ESS operating set,
and the H1-safe disjunctive outer approximation.

D2 asks whether the ~0.5 S simultaneous circulation observed in P5.5-C is a
missing convex constraint or an unavoidable property of convexification.  D3
builds the disjunction that a mixed-integer model would use, and proves it is an
OUTER approximation of the production H1 set so that a MISOCP optimum over it
remains a valid lower bound.

Production's set, for one shared ESS and one period, with capacity S:

    pch >= 0,  pdch >= 0
    pch + pdch <= S                                (sess_active_sum_limit_rule)
    pch_hat * pdch_hat <= eps                      (sess_comp_rule, H1 form)

with pch = S * pch_hat, pdch = S * pdch_hat, so the last row is exactly

    pch * pdch <= eps * S^2,        eps = ESS_COMPLEMENTARITY_TOLERANCE = 1e-4,
                                    delta = sqrt(eps) = 0.01.

Everything below is proved analytically and then checked numerically against the
production constants and, where a solved point is available, against the D1
polished nonlinear schedule.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55d_d2d3_hull.py
"""

import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from definitions import ESS_COMPLEMENTARITY_TOLERANCE  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55D')


def h1_feasible(pch, pdch, s, eps):
    return (pch >= -1e-15 and pdch >= -1e-15
            and pch + pdch <= s * (1 + 1e-12)
            and pch * pdch <= eps * s * s * (1 + 1e-12))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-D2/D3 hull', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[D2] ABORTED\n{error}')
        sys.exit(1)

    eps = float(ESS_COMPLEMENTARITY_TOLERANCE)
    delta = math.sqrt(eps)
    s = 1.0                       # everything below is homogeneous of degree 1 in S
    report = {'stage': 'P5.5-D2/D3', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'eps': eps, 'delta': delta}

    print(f'[D2] eps = ESS_COMPLEMENTARITY_TOLERANCE = {eps:g}, '
          f'delta = sqrt(eps) = {delta:g}\n')

    # ------------------------------------------------------------------ D2.1
    charge, discharge, midpoint = (s, 0.0), (0.0, s), (0.5 * s, 0.5 * s)
    d2_1 = {
        'charge_point_(S,0)_H1_feasible': h1_feasible(*charge, s, eps),
        'discharge_point_(0,S)_H1_feasible': h1_feasible(*discharge, s, eps),
        'midpoint_(S/2,S/2)_H1_feasible': h1_feasible(*midpoint, s, eps),
        'midpoint_product_over_eps_S2': (0.25 * s * s) / (eps * s * s),
        'statement': ('(S,0) and (0,S) are both H1-feasible, so BY DEFINITION OF '
                      'CONVEXITY every convex set containing both contains their '
                      'midpoint (S/2, S/2).  That midpoint is NOT H1-feasible: '
                      'its product is S^2/4, which exceeds eps*S^2 by a factor '
                      'of 1/(4*eps) = 2500.'),
    }
    report['D2_1_midpoint'] = d2_1
    print('[D2.1] the two valid operating modes and their midpoint')
    print(f"    (S,0)     H1-feasible : {d2_1['charge_point_(S,0)_H1_feasible']}")
    print(f"    (0,S)     H1-feasible : {d2_1['discharge_point_(0,S)_H1_feasible']}")
    print(f"    (S/2,S/2) H1-feasible : {d2_1['midpoint_(S/2,S/2)_H1_feasible']}"
          f"   (product is {d2_1['midpoint_product_over_eps_S2']:.0f} x eps*S^2)")

    # ------------------------------------------------------------------ D2.2
    # Exact disjunction: D = {(a,0): 0<=a<=S} union {(0,b): 0<=b<=S}
    # conv(D) = triangle {pch>=0, pdch>=0, pch+pdch<=S}
    rng = np.random.default_rng(20260908)
    n = 200000
    u = rng.random(n)
    v = rng.random(n)
    tri_mask = (u + v) <= 1.0
    tri = np.stack([u[tri_mask], v[tri_mask]], axis=1) * s
    # every triangle point is a convex combination of (0,0), (S,0), (0,S), all in D
    max_sum = float(np.max(tri[:, 0] + tri[:, 1]))
    d2_2 = {
        'D_subset_of_triangle': True,
        'D_subset_of_triangle_proof': ('each leg point (a,0) or (0,b) with '
                                       '0<=a,b<=S satisfies pch>=0, pdch>=0 and '
                                       'pch+pdch<=S'),
        'triangle_subset_of_conv_D': True,
        'triangle_subset_of_conv_D_proof': ('every (pch,pdch) with pch,pdch>=0 and '
                                            'pch+pdch<=S equals '
                                            '(pch/S)(S,0) + (pdch/S)(0,S) + '
                                            '(1-(pch+pdch)/S)(0,0), a convex '
                                            'combination of three points of D'),
        'conclusion': ('conv(exact charge/discharge disjunction) is EXACTLY the '
                       'triangle {pch>=0, pdch>=0, pch+pdch<=S}, i.e. the '
                       'production active-sum envelope is already that convex '
                       'hull -- there is nothing left to tighten.'),
        'sampled_triangle_points': int(tri.shape[0]),
        'max_sampled_pch_plus_pdch_over_S': max_sum / s,
    }
    report['D2_2_exact_disjunction_hull'] = d2_2
    print('\n[D2.2] convex hull of the EXACT disjunction')
    print(f"    {d2_2['conclusion']}")

    # ------------------------------------------------------------------ D2.3
    # conv(F_H1): D subset F_H1 subset triangle  =>  conv(F_H1) = triangle
    on_axis = np.concatenate([
        np.stack([np.linspace(0, s, 5000), np.zeros(5000)], axis=1),
        np.stack([np.zeros(5000), np.linspace(0, s, 5000)], axis=1)])
    d_in_h1 = all(h1_feasible(a, b, s, eps) for a, b in on_axis)
    h1_samples = tri[(tri[:, 0] * tri[:, 1]) <= eps * s * s]
    h1_in_triangle = bool(np.all(h1_samples[:, 0] + h1_samples[:, 1] <= s + 1e-12))
    d2_3 = {
        'D_subset_F_H1': bool(d_in_h1),
        'F_H1_subset_triangle': h1_in_triangle,
        'conclusion': ('D subset F_H1 subset triangle, and conv(D) = triangle, so '
                       'triangle = conv(D) subset conv(F_H1) subset triangle: '
                       'conv(F_H1) is EXACTLY the same triangle.  The bilinear '
                       'H1 row pch*pdch <= eps*S^2 contributes NOTHING to the '
                       'convex hull -- it is entirely redundant once convexified.'),
        'sampled_H1_points': int(h1_samples.shape[0]),
    }
    report['D2_3_H1_hull'] = d2_3
    print('\n[D2.3] convex hull of the ACTUAL H1 set (tolerance, not exact complementarity)')
    print(f"    {d2_3['conclusion']}")

    # ---------------------------------------------------- maximum circulation
    # max min(pch,pdch) over F_H1: at pch = pdch = c we need c^2 <= eps S^2,
    # so c <= delta*S; and 2c <= S is slack because delta = 0.01 << 0.5.
    circ_h1 = float(np.max(np.minimum(h1_samples[:, 0], h1_samples[:, 1]))) / s
    d2_max = {
        'max_circulation_over_S_in_F_H1_analytic': delta,
        'max_circulation_over_S_in_F_H1_sampled': circ_h1,
        'max_circulation_over_S_in_conv_F_H1': 0.5,
        'ratio_hull_over_H1': 0.5 / delta,
        'observed_in_P55C_continuous_relaxation': 0.4989262,
        'statement': ('The exact H1 set permits at most delta*S = 0.01 S of '
                      'simultaneous circulation.  Its convex hull permits 0.5 S, '
                      'attained exactly at the midpoint of the two valid modes.  '
                      'The 0.4989 S measured in P5.5-C is therefore not a symptom '
                      'of a missing constraint: it is the convex-hull midpoint, '
                      'reached because nothing in the relaxed objective opposes '
                      'it.'),
    }
    report['D2_4_maximum_circulation'] = d2_max
    print('\n[D2.4] how much circulation each set allows')
    print(f"    max min(pch,pdch)/S over F_H1          : {delta:.4f}  "
          f"(sampled {circ_h1:.6f})")
    print(f"    max min(pch,pdch)/S over conv(F_H1)    : 0.5000")
    print(f"    ratio                                  : {0.5 / delta:.0f} x")
    print(f"    observed in the P5.5-C relaxation      : 0.498926")

    # -------------------------------------------------- required D2 statement
    required = ('The observed approximately 0.5 S simultaneous circulation '
                'cannot be removed by a purely continuous convex constraint on '
                'pch/pdch/S without excluding convex combinations of valid '
                'physical operating modes.')
    report['D2_required_statement'] = required
    report['D2_required_statement_holds'] = True
    print(f'\n[D2] REQUIRED STATEMENT\n    "{required}"')

    # ------------------------------------------------------------------ D2.5
    # Multi-period: does the SOC recursion imply extra valid convex cuts?
    d2_5 = {
        'soc_recursion': 'soc[p] = soc[p-1] + eta_ch*pch[p]*dt - pdch[p]*dt/eta_dch',
        'day_balance': 'soc[last] = soc_init (soft: shared_ess.day_balance slack is enabled)',
        'implied_equality': ('summing the recursion over the day and imposing the '
                             'day balance gives eta_ch*sum(pch) = sum(pdch)/eta_dch, '
                             'i.e. sum(pdch) = eta_ch*eta_dch*sum(pch)'),
        'is_it_a_new_cut': False,
        'why_not': ('that equality is a linear consequence of rows already in the '
                    'model (the SOC recursion and the day-balance anchor), so '
                    'adding it changes no feasible set.  It also constrains only '
                    'the daily TOTALS, not min(pch,pdch) in any single period, so '
                    'it cannot bound circulation per period.'),
        'further_weakening': ('the day-balance anchor carries a slack '
                              '(params.slacks.shared_ess.day_balance is True), so '
                              'even the totals equality is soft.'),
        'decisive_objective_observation': (
            'P5.5-C established that production ZEROES penalty_ess_usage before '
            'the distributed solve, and get_primal_value -- the definition of the '
            'recourse being bounded -- is evaluated on the zeroed parameter.  The '
            'objective of the relaxed problem therefore charges NOTHING for '
            'pch + pdch.  Circulation is free, and the round-trip loss it causes '
            'is a benefit to the relaxation rather than a cost: it lets the '
            'storage absorb energy the network needs to shed.  This is why the '
            'relaxation sits at the hull midpoint rather than somewhere interior.'),
        'no_inequality_added': True,
    }
    report['D2_5_multiperiod'] = d2_5
    print('\n[D2.5] multi-period SOC recursion')
    print(f"    implied equality : {d2_5['implied_equality']}")
    print(f"    new valid cut?   : NO -- {d2_5['why_not']}")
    print(f"    and note         : {d2_5['decisive_objective_observation']}")

    # ------------------------------------------------------------------- D3
    # min(pch,pdch)^2 <= pch*pdch <= eps*S^2  =>  min(pch,pdch) <= delta*S
    viol = 0.0
    for a, b in h1_samples:
        viol = max(viol, min(a, b) - delta * s)
    d3 = {
        'proof': ('for pch,pdch >= 0, min(pch,pdch)^2 <= pch*pdch <= eps*S^2, '
                  'hence min(pch,pdch) <= sqrt(eps)*S = delta*S'),
        'consequence': ('every H1-feasible point has pdch <= delta*S (when pdch is '
                        'the min) or pch <= delta*S (when pch is the min), so '
                        'F_H1 is contained in the union F_charge OR F_discharge'),
        'F_charge': 'pdch <= delta*S, together with pch,pdch >= 0 and pch+pdch <= S',
        'F_discharge': 'pch  <= delta*S, together with pch,pdch >= 0 and pch+pdch <= S',
        'containment_F_H1_subset_union': True,
        'sampled_points_checked': int(h1_samples.shape[0]),
        'worst_sampled_violation_of_min_le_delta_S': float(viol),
        'exact_exclusivity_rejected': (
            'pch == 0 OR pdch == 0 is NOT used: production H1 permits '
            'simultaneous operation up to delta*S, so exact exclusivity would '
            'CUT OFF H1-feasible points and the result would not be a lower '
            'bound.'),
        'active_sum_retained': 'pch + pdch <= S is retained in both branches',
        'big_M': ('not used; the branch is imposed with a Gurobi indicator '
                  'constraint, so no arbitrary constant enters the formulation'),
        'lower_bound_validity': (
            'F_H1 subset F_disjunctive_outer, so minimising over the disjunctive '
            'set gives a value no larger than minimising over F_H1.  The MISOCP '
            'optimum therefore remains a valid lower bound to the nonlinear '
            'H1-constrained recourse, subject to the other already accepted '
            'relaxations (W-space rank, ESSO energy interval, dropped penalties).'),
    }
    report['D3_disjunction'] = d3
    print('\n[D3] H1-safe disjunctive outer approximation')
    print(f"    {d3['proof']}")
    print(f"    worst sampled violation of min(pch,pdch) <= delta*S : {viol:.3e} "
          f"over {h1_samples.shape[0]} H1 points")
    print(f"    F_H1 subset F_charge union F_discharge : "
          f"{d3['containment_F_H1_subset_union']}")

    # ------------------------------- confront the D1 polished nonlinear schedule
    schedule_path = os.path.join(OUT_DIR, 'p55d_d1_ess_schedule.json')
    if os.path.exists(schedule_path):
        with open(schedule_path) as handle:
            schedule = json.load(handle)
        worst_ratio, worst_key = 0.0, None
        for key, entry in schedule.items():
            pch, pdch = entry['pch'], entry['pdch']
            circ = min(abs(pch), abs(pdch))
            if circ > worst_ratio:
                worst_ratio, worst_key = circ, key
        report['D3_against_polished_nonlinear'] = {
            'source': 'p55d_d1_ess_schedule.json',
            'max_min_pch_pdch_pu': worst_ratio, 'worst_index': worst_key,
            'note': ('absolute p.u. circulation of the polished nonlinear point; '
                     'compare with delta*S for that ESS'),
        }
        print(f'\n[D3] polished nonlinear point: max min(pch,pdch) = '
              f'{worst_ratio:.6e} p.u. at {worst_key}')

    out = os.path.join(OUT_DIR, 'p55d_d2d3_hull.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[D2/D3] report -> {out}')


if __name__ == '__main__':
    main()
