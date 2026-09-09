"""
Stage P5.5-C7 (diagnosis, second level) -- find which constraint family stops the
barrier certifying a dual bound on a DSO block.

`p55c_c7_subsolves.py` showed that a TSO-only model certifies in ten barrier
iterations while a DSO-only model of half the size does not certify at all.  So
the obstruction is a property of the distribution block, not of problem size or
of the TSO/DSO coupling.  This script takes one DSO block and knocks out one
suspect family at a time.

Every variant here MODIFIES the model and therefore bounds a different problem.
Nothing produced here is a bound on anything; the only output that matters is
whether the barrier certifies.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p55c_c7_dso_diag.py
"""

import io
import json
import math
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402
from p55c_c1_oracle import build_centralized_relaxation, model_size  # noqa: E402
from p55c_c7_solve import _grb, solve  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55C')

RESTRICTION = dict(only_agents=['DSO5'], only_years=[2025], only_days=['Winter'])


def blocks_of(parent):
    return list(parent.blocks.values())


# ---------------------------------------------------------------- knockouts
def knock_none(parent, planning):
    return 'as built (coordinated DSO block)'


def knock_degenerate_cones(parent, planning):
    """Replace `pg^2+qg^2 <= 0` rows by fixing pg = qg = 0.

    A second-order cone with a zero right-hand side has empty interior, which is
    exactly the situation in which a primal-dual interior point method has no
    strictly feasible point to work with.  Unavailable renewables produce one
    such row per generator and period.
    """
    n = 0
    for blk in blocks_of(parent):
        network, params = blk.network, blk.params
        for g in blk.generators:
            for p in blk.periods:
                avail = mch.sg_avail_init(blk, g, blk.s_o, p, network, params)
                if not network.generators[g].is_curtaillable() or avail > 1e-9:
                    continue
                index = (g, blk.s_m, blk.s_o, p)
                if index in blk.sg_capability:
                    blk.sg_capability[index].deactivate()
                blk.pg[index].fix(0.0)
                blk.qg[index].fix(0.0)
                n += 1
    return f'degenerate zero-RHS sg_capability cones removed ({n} rows)'


def knock_tap_box(parent, planning):
    """Pin every tap branch to its nominal ratio, removing the OLTC box."""
    n = 0
    for blk in blocks_of(parent):
        network = blk.network
        if not hasattr(blk, 'tap_branches'):
            continue
        for b in blk.tap_branches:
            ratio_sqr = network.branches[b].ratio ** 2
            f_idx = network.get_node_idx(network.branches[b].fbus)
            for p in blk.periods:
                blk.tap_lower[b, p].deactivate()
                blk.tap_upper[b, p].deactivate()
                n += 1
        blk.tap_pinned = pe.Constraint(
            blk.tap_branches, blk.periods,
            rule=lambda m, b, p: m.Ub[b, p] == (network.branches[b].ratio ** 2)
            * m.vmag_sqr[network.get_node_idx(network.branches[b].fbus),
                         m.s_m, m.s_o, p])
    return f'OLTC tap box replaced by the nominal ratio ({n} rows)'


def knock_rank_soc(parent, planning):
    """Drop the rank cone entirely (leaves a pure linear/affine network)."""
    n = 0
    for blk in blocks_of(parent):
        for index in blk.rank_soc:
            blk.rank_soc[index].deactivate()
            n += 1
    return f'rank_soc cones deactivated ({n} rows)'


def knock_sess_cone(parent, planning):
    """Drop the shared-ESS converter capability cone."""
    n = 0
    for blk in blocks_of(parent):
        for index in blk.sess_converter_capability:
            blk.sess_converter_capability[index].deactivate()
            n += 1
    return f'sess_converter_capability cones deactivated ({n} rows)'


def knock_branch_flow(parent, planning):
    """Drop the branch thermal-limit cones."""
    n = 0
    for blk in blocks_of(parent):
        for index in blk.branch_flow_limit:
            blk.branch_flow_limit[index].deactivate()
            n += 1
        if hasattr(blk, 'branch_flow_limit_ji'):
            for index in blk.branch_flow_limit_ji:
                blk.branch_flow_limit_ji[index].deactivate()
                n += 1
    return f'branch thermal-limit cones deactivated ({n} rows)'


def knock_voltage_slacks(parent, planning):
    n = 0
    for blk in blocks_of(parent):
        for name in ('slack_v_sqr_up', 'slack_v_sqr_down'):
            if not hasattr(blk, name):
                continue
            var = getattr(blk, name)
            for index in var:
                var[index].fix(0.0)
                n += 1
    return f'voltage slacks fixed to zero ({n} variables)'


KNOCKOUTS = [
    ('baseline', knock_none),
    ('no_degenerate_cones', knock_degenerate_cones),
    ('no_tap_box', knock_tap_box),
    ('no_sess_cone', knock_sess_cone),
    ('no_branch_flow_cone', knock_branch_flow),
    ('no_voltage_slacks', knock_voltage_slacks),
    ('no_rank_soc', knock_rank_soc),
]

VARIANTS = [('default', {}), ('homogeneous+focus', {'BarHomogeneous': 1,
                                                    'NumericFocus': 3})]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C7 DSO diagnosis', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C7dd] ABORTED\n{error}')
        sys.exit(1)

    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    report = {'stage': 'P5.5-C7 DSO diagnosis', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'restriction': {k: list(v) for k, v in RESTRICTION.items()},
              'knockouts': []}

    for name, knock in KNOCKOUTS:
        parent = build_centralized_relaxation(planning, candidate, **RESTRICTION)
        description = knock(parent, planning)
        entry = {'knockout': name, 'description': description,
                 'model_size': model_size(parent), 'variants': []}
        print(f'\n[C7dd] === {name}: {description}', flush=True)
        for vname, extra in VARIANTS:
            with redirect_stdout(io.StringIO()) as log:
                try:
                    opt, _ = solve(parent, extra=extra)
                    failure = None
                except Exception as error:
                    opt, failure = None, f'{type(error).__name__}: {error}'
            text = log.getvalue()
            if opt is None:
                variant = {'variant': vname, 'error': failure}
            else:
                bound = _grb(opt, 'ObjBound')
                variant = {
                    'variant': vname, 'ObjVal': _grb(opt, 'ObjVal'),
                    'ObjBound': bound,
                    'certified': bound is not None and math.isfinite(bound),
                    'BarIterCount': _grb(opt, 'BarIterCount'),
                    'Runtime': _grb(opt, 'Runtime'),
                    'qcp_dual_warning': 'failed to compute QCP dual' in text,
                    'numerical_trouble': 'Numerical trouble' in text,
                    'suboptimal': 'Sub-optimal termination' in text,
                }
            entry['variants'].append(variant)
            flag = 'CERTIFIED' if variant.get('certified') else 'no-bound'
            print(f"    {vname:20s} {flag:10s} ObjVal={variant.get('ObjVal')} "
                  f"ObjBound={variant.get('ObjBound')} "
                  f"iters={variant.get('BarIterCount')}", flush=True)
            if variant.get('certified'):
                break
        report['knockouts'].append(entry)

    out = os.path.join(OUT_DIR, 'p55c_c7_dso_diag.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[C7dd] report -> {out}')


if __name__ == '__main__':
    main()
