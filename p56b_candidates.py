"""
The fixed P5.6-B benchmark population.

Twelve master-feasible candidates (the base plus eleven moves), none random, chosen to cover what B2 has to
separate: all three nodes, all three years, negative and positive fixed-ratio
moves, two duration-increase directions, the candidate that needed the
DSO-anchor fallback in P5.6-A, and one candidate near the first-stage budget
boundary.  The boundary candidate is built from the existing planning rules --
the base uses 5% of the investment budget, so scaling every investment by 19
uses 95% of it -- rather than by inventing a new rule.

The base sits exactly on E = phi_min * S with phi_min = 2, so E-only DECREASES
are first-stage infeasible; only E increases are available as duration moves.
"""

import io
import os
import sys
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_candidates as A  # noqa: E402
import p56a_oracle as O  # noqa: E402

# (label, kind, node|None for all, year|None for all, relative change)
#   kind 's'  : power only, raises E/S above the minimum
#   kind 'e'  : energy only -- a duration increase at fixed power
#   kind 'se' : both, preserving the E/S ratio
MOVES = [
    ('s|node5|2025|-10%', 's', 5, 2025, -0.10),
    ('se|node5|2025|-10%', 'se', 5, 2025, -0.10),
    ('se|node7|2030|-10%', 'se', 7, 2030, -0.10),
    ('se|node9|2025|-10%', 'se', 9, 2025, -0.10),   # needed the A-stage fallback
    ('se|node5|2030|+10%', 'se', 5, 2030, +0.10),
    ('se|node7|2035|+10%', 'se', 7, 2035, +0.10),
    ('se|node9|2025|+10%', 'se', 9, 2025, +0.10),
    ('e|node5|2025|+25%', 'e', 5, 2025, +0.25),     # duration up, fixed power
    ('e|node9|2030|+50%', 'e', 9, 2030, +0.50),     # duration up, fixed power
    ('se|ALL|-10%', 'se', None, None, -0.10),
    ('se|ALL|x19 (budget boundary)', 'se', None, None, +18.0),
]


def apply_move(base, kind, node, year, rel):
    x = {key: dict(value) for key, value in base.items()}
    components = ('s', 'e') if kind == 'se' else (kind,)
    for (n, y) in x:
        if node is not None and n != node:
            continue
        if year is not None and y != year:
            continue
        for component in components:
            x[(n, y)][component] = x[(n, y)][component] * (1.0 + rel)
    return x


def population(planning):
    """[(label, x)] -- the base candidate first, then the twelve moves."""
    base = A.base_vector(planning)
    out = [('base', base)]
    for label, kind, node, year, rel in MOVES:
        out.append((label, apply_move(base, kind, node, year, rel)))
    return out


def describe(planning, x):
    """Master-feasibility and the binding first-stage quantities."""
    candidate = O.vector_to_candidate(planning, x)
    feasible, reason = O.check_master_feasibility(planning, candidate)
    params = planning.shared_ess_data.params
    max_e = max(v['e'] for node in candidate['total_capacity'].values()
                for v in node.values())
    ratios = [candidate['investment'][n][y]['e'] / candidate['investment'][n][y]['s']
              for n in candidate['investment'] for y in candidate['investment'][n]
              if candidate['investment'][n][y]['s'] > 0]
    cost = O.investment_cost(planning, candidate)
    return {
        'master_feasible': feasible, 'reason': reason,
        'investment_cost': cost,
        'budget': params.budget,
        'budget_used_fraction': cost / params.budget,
        'max_cumulative_E': max_e,
        'max_capacity': params.max_capacity,
        'capacity_used_fraction': max_e / params.max_capacity,
        'min_ES_ratio_seen': min(ratios) if ratios else None,
        'max_ES_ratio_seen': max(ratios) if ratios else None,
        'phi_min': params.min_energy_to_power_ratio,
        'phi_max': params.max_energy_to_power_ratio,
    }
