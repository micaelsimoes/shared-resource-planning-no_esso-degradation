"""
Stage P5.6-B5 -- first-stage-feasible search coordinates.

The base candidate sits EXACTLY on E = phi_min * S at all nine (node, investment
year) pairs, so the minimum-duration constraint is active everywhere.  That makes
the choice of coordinates a correctness question, not a convenience: in native
S/E coordinates half the poll directions at the base are infeasible before any
model is solved.

This enumerates the feasible directions of each candidate representation at the
base and at a perturbed interior point, so the recommendation rests on counted
directions rather than on argument.

    /opt/anaconda3/envs/opf_env_py311/bin/python p56b_b5_coordinates.py
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_candidates as A  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p56b_candidates as B  # noqa: E402
import p56b_policy as P  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402


def native_directions(x):
    """+-s and +-e per (node, year): the 2n native coordinate directions."""
    for key in sorted(x):
        for component in ('s', 'e'):
            for sign in (+1.0, -1.0):
                yield (f'{"+" if sign > 0 else "-"}{component}|{key[0]}|{key[1]}',
                       key, component, sign)


def step_native(x, key, component, sign, step):
    out = {k: dict(v) for k, v in x.items()}
    out[key][component] = out[key][component] + sign * step
    return out


def step_sh(x, key, component, sign, step, phi_min):
    """A step in (S, h) coordinates, mapped back to (S, E).

    h = E - phi_min * S >= 0, so E = phi_min * S + h.  Moving S carries E with
    it; moving h changes duration at fixed power.
    """
    out = {k: dict(v) for k, v in x.items()}
    s = out[key]['s']
    h = out[key]['e'] - phi_min * s
    if component == 's':
        s = s + sign * step
    else:
        h = h + sign * step
    out[key]['s'] = s
    out[key]['e'] = phi_min * s + h
    return out


def count_feasible(planning, x, stepper, step, phi_min):
    total, feasible, blocked = 0, 0, []
    for name, key, component, sign in native_directions(x):
        total += 1
        trial = stepper(x, key, component, sign, step, phi_min) \
            if stepper is step_sh else stepper(x, key, component, sign, step)
        candidate = O.vector_to_candidate(planning, trial)
        ok, reason = O.check_master_feasibility(planning, candidate)
        if ok:
            feasible += 1
        else:
            blocked.append({'direction': name, 'reason': reason})
    return {'directions': total, 'feasible': feasible,
            'infeasible': total - feasible,
            'feasible_fraction': feasible / total,
            'blocked_examples': blocked[:4],
            'blocked_kinds': sorted({b['direction'][:2] for b in blocked})}


def main():
    try:
        provenance, planning = gate('P5.6-B5 coordinates', P.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[B5] ABORTED\n{error}')
        sys.exit(1)

    params = planning.shared_ess_data.params
    phi_min = params.min_energy_to_power_ratio
    phi_max = params.max_energy_to_power_ratio
    base = A.base_vector(planning)
    step = 0.10 * base[(5, 2025)]['s']          # a 10 % move on the base power

    # an interior point: duration raised everywhere, so E > phi_min * S
    interior = {k: {'s': v['s'], 'e': v['e'] * 1.25} for k, v in base.items()}

    report = {'stage': 'P5.6-B5', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'phi_min': phi_min, 'phi_max': phi_max,
              'max_capacity': params.max_capacity, 'budget': params.budget,
              'step_used': step,
              'first_stage_constraints': [
                  's[n,y] >= 0 and e[n,y] >= 0',
                  'e[n,y] >= phi_min * s[n,y]        (phi_min = 2)',
                  'e[n,y] <= phi_max * s[n,y]        (phi_max = 10)',
                  'cumulative E per node/year <= max_capacity   (cohort/calendar-life mapping)',
                  'expected discounted investment cost <= budget',
              ],
              'feasible_set_is_polyhedral': True,
              'points': {}}

    for label, x in (('base (E = phi_min * S active everywhere)', base),
                     ('interior (E = 1.25 * phi_min * S)', interior)):
        entry = {
            'OPTION_A_native_S_E': count_feasible(planning, x, step_native,
                                                  step, phi_min),
            'OPTION_B_power_plus_duration_slack': count_feasible(
                planning, x, step_sh, step, phi_min),
        }
        report['points'][label] = entry
        print(f'\n[B5] {label}')
        for option, data in entry.items():
            print(f"      {option:36s} feasible {data['feasible']:2d}/"
                  f"{data['directions']}  "
                  f"({100 * data['feasible_fraction']:.0f} %)  "
                  f"blocked kinds {data['blocked_kinds']}")

    out = os.path.join(P.OUT_DIR, 'p56b_b5_coordinates.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[B5] report -> {out}')


if __name__ == '__main__':
    main()
