"""
Canonical benchmark candidates for P5.6-A.

Every perturbation here is one of the P5.4-R/D3 candidates already defined and
solved under the canonical environment, rather than a newly invented point, so
the benchmark is comparable with the existing historical evidence.
"""

import io
import os
import sys
from contextlib import redirect_stdout

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

# (label, kind, node, year, relative change) -- all from P5.4-R/D3.
#
# `kind='se'` scales S and E together.  It is needed because the canonical base
# candidate sits EXACTLY at the minimum energy-to-power ratio of 2.0, so any
# E-only reduction is first-stage infeasible: P5.4-R/D3's `e|node9|2025|-x%`
# candidates violate `min_energy_to_power_ratio` and the oracle rejects them as
# INVALID_INVESTMENT before running anything.  That rejection is retained below
# as a deliberate negative test.
PERTURBATIONS = [
    ('s|node5|2025|-10%', 's', 5, 2025, -0.10),
    ('s|node9|2025|-5%', 's', 9, 2025, -0.05),
    ('s+e|node9|2025|-10%', 'se', 9, 2025, -0.10),
]

# A P5.4-R/D3 candidate that is first-stage INFEASIBLE, kept as a negative test.
INFEASIBLE_PERTURBATION = ('e|node9|2025|-10%', 'e', 9, 2025, -0.10)


def base_vector(planning):
    with redirect_stdout(io.StringIO()):
        base = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
    return O.candidate_to_vector(planning, base)


def perturbed(base, kind, node, year, rel):
    x = {key: dict(value) for key, value in base.items()}
    for component in (('s', 'e') if kind == 'se' else (kind,)):
        x[(node, year)][component] = x[(node, year)][component] * (1.0 + rel)
    return x


def benchmark_population(planning):
    """The fixed benchmark set: canonical base plus three D3 perturbations."""
    base = base_vector(planning)
    population = [('base (canonical positive bootstrap)', base)]
    for label, kind, node, year, rel in PERTURBATIONS:
        population.append((label, perturbed(base, kind, node, year, rel)))
    return population
