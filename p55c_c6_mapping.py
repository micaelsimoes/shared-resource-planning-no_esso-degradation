"""
Stage P5.5-C6 -- map a canonical nonlinear feasible state into the convex block
and verify the outer-relaxation direction.

This is the strongest implementation check available: if the relaxation is a
genuine outer relaxation, every nonlinear feasible point must map to a convex
feasible point, and the relaxed objective there must be <= the nonlinear
objective by exactly the dropped non-negative terms.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p55c_c6_mapping.py
"""

import io
import json
import os
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from convex_oracle import block_objective, build_ac_block  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55C')


def constraint_violation(con_data):
    """Signed violation of one constraint (0 if satisfied)."""
    try:
        body = float(pe.value(con_data.body))
    except Exception:
        return None
    worst = 0.0
    if con_data.has_lb():
        worst = max(worst, float(pe.value(con_data.lower)) - body)
    if con_data.has_ub():
        worst = max(worst, body - float(pe.value(con_data.upper)))
    return worst


OPERATIONAL_NAMES = ('pc', 'qc', 'pg', 'qg', 'flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down',
                     'slack_v_sqr_up', 'slack_v_sqr_down',
                     'shared_es_pch', 'shared_es_pdch', 'shared_es_pnet',
                     'shared_es_qnet', 'shared_es_soc',
                     'slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down')


def map_state(blk, model, network, s_m=0, s_o=0, clip=True):
    """Copy a solved nonlinear model's state into the convex block.

    With ``clip=False`` the nonlinear values are transferred verbatim, including
    the solver's tolerance-level bound violations; that point is what the exact
    objective accounting must be evaluated at.  With ``clip=True`` each value is
    projected onto the convex variable's own bounds, which is what makes the
    mapped point genuinely convex-feasible.  The projection itself is reported.
    """
    for i in blk.nodes:
        for p in blk.periods:
            e = float(pe.value(model.e[i, s_m, s_o, p]))
            f = float(pe.value(model.f[i, s_m, s_o, p]))
            blk.vmag_sqr[i, s_m, s_o, p].value = e * e + f * f
    for b in blk.branches:
        branch = network.branches[b]
        fi = network.get_node_idx(branch.fbus)
        ti = network.get_node_idx(branch.tbus)
        for p in blk.periods:
            ef, ff = (float(pe.value(model.e[fi, s_m, s_o, p])),
                      float(pe.value(model.f[fi, s_m, s_o, p])))
            et, ft = (float(pe.value(model.e[ti, s_m, s_o, p])),
                      float(pe.value(model.f[ti, s_m, s_o, p])))
            WijR = ef * et + ff * ft
            WijI = ff * et - ef * ft
            r = (float(pe.value(model.r[b, s_m, s_o, p]))
                 if branch.is_transformer else 1.0)
            blk.Cb[b, p].value = r * WijR
            blk.Db[b, p].value = r * WijI
            blk.Ub[b, p].value = r * r * (ef * ef + ff * ff)
    projection = {}
    for name in OPERATIONAL_NAMES:
        if not hasattr(blk, name) or not hasattr(model, name):
            continue
        src, dst = getattr(model, name), getattr(blk, name)
        if not isinstance(dst, pe.Var):
            continue                     # Params carry identical data by construction
        worst = 0.0
        for idx in dst:
            try:
                value = float(pe.value(src[idx]))
            except Exception:
                continue
            lo, hi = dst[idx].bounds
            shifted = value
            if lo is not None and shifted < lo:
                shifted = lo
            if hi is not None and shifted > hi:
                shifted = hi
            worst = max(worst, abs(shifted - value))
            dst[idx].value = shifted if clip else value
        if worst > 0.0:
            projection[name] = worst
    for e in blk.shared_energy_storages:
        blk.S_av[e].value = float(pe.value(model.shared_es_s_rated_fixed[e]))
        blk.E_av[e].value = float(pe.value(model.shared_es_e_rated_fixed[e]))
    return projection


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning = gate('P5.5-C6 mapping', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C6] ABORTED\n{error}')
        sys.exit(1)

    console = io.StringIO()
    with redirect_stdout(console):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    report = {'stage': 'P5.5-C6', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'cases': {}}

    year, day = list(planning.years)[0], list(planning.days)[0]
    cases = [('tso', planning.transmission_network, None),
             ('dso5', planning.distribution_networks[5], 5),
             ('dso7', planning.distribution_networks[7], 7),
             ('dso9', planning.distribution_networks[9], 9)]

    for tag, holder, node_id in cases:
        network = holder.network[year][day]
        params = holder.params
        print(f'\n[C6] {tag}: solving canonical nonlinear SMOPF ...', flush=True)
        with redirect_stdout(io.StringIO()):
            model = network.build_model(params)
            # capacity exactly as production sets it
            if node_id is None:
                for e_idx, sess in enumerate(network.shared_energy_storages):
                    nid = sess.bus
                    s_pu = abs(candidate['total_capacity'][nid][year]['s']) / network.baseMVA
                    e_pu = abs(candidate['total_capacity'][nid][year]['e']) / network.baseMVA
                    model.shared_es_s_rated_fixed[e_idx].set_value(s_pu)
                    model.shared_es_e_rated_fixed[e_idx].set_value(e_pu)
                    mch.configure_shared_ess_operational_state(model, e_idx, s_pu, e_pu)
            else:
                e_idx = network.get_shared_energy_storage_idx(network.get_reference_node_id())
                s_pu = abs(candidate['total_capacity'][node_id][year]['s']) / network.baseMVA
                e_pu = abs(candidate['total_capacity'][node_id][year]['e']) / network.baseMVA
                model.shared_es_s_rated_fixed[e_idx].set_value(s_pu)
                model.shared_es_e_rated_fixed[e_idx].set_value(e_pu)
                mch.configure_shared_ess_operational_state(model, e_idx, s_pu, e_pu)
            result = network.run_smopf(model, params, print_header=False)
        ok = bool(srp._solver_result_succeeded(result))
        if not ok:
            report['cases'][tag] = {'nonlinear_solved': False}
            print('    nonlinear solve FAILED')
            continue
        q_nl = float(pe.value(model.objective))

        blk = build_ac_block(network, params)

        # (a) exact accounting at the verbatim nonlinear point (no projection)
        map_state(blk, model, network, clip=False)
        q_raw = float(pe.value(block_objective(blk)))
        dropped = float(pe.value(mch.ess_complementarity_penalties_rule(
            model, 0, 0, network=network, params=params))) * (
            network.prob_market_scenarios[0] * network.prob_operation_scenarios[0])
        residual_raw = q_nl - q_raw - dropped

        # (b) convex-feasible point: project onto the convex variable bounds
        projection = map_state(blk, model, network, clip=True)

        # production's own residual on the same families, for attribution
        prod_worst = {}
        for con in model.component_objects(pe.Constraint, active=True):
            worst = 0.0
            for idx in con:
                v = constraint_violation(con[idx])
                if v is not None:
                    worst = max(worst, v)
            prod_worst[con.local_name] = worst

        # violations by family
        fams = defaultdict(lambda: {'n': 0, 'max_violation': 0.0, 'worst_index': None})
        for con in blk.component_objects(pe.Constraint, active=True):
            name = con.local_name
            for idx in con:
                v = constraint_violation(con[idx])
                if v is None:
                    continue
                fams[name]['n'] += 1
                if v > fams[name]['max_violation']:
                    fams[name]['max_violation'] = v
                    fams[name]['worst_index'] = str(idx)
        fams = {k: dict(v) for k, v in fams.items()}
        worst_overall = max((v['max_violation'] for v in fams.values()), default=0.0)

        # objective at the projected (convex-feasible) point
        q_relaxed_at_point = float(pe.value(block_objective(blk)))
        residual = q_nl - q_relaxed_at_point - dropped

        report['cases'][tag] = {
            'nonlinear_solved': True,
            'nonlinear_objective': q_nl,
            'relaxed_objective_at_raw_mapped_point': q_raw,
            'relaxed_objective_at_projected_point': q_relaxed_at_point,
            'dropped_complementarity_penalty': dropped,
            'accounting_residual_raw': residual_raw,
            'accounting_exact': abs(residual_raw) < 1e-9 * max(abs(q_nl), 1.0),
            'bound_projection_by_variable': projection,
            'objective_shift_from_projection': residual,
            'worst_violation_overall': worst_overall,
            'mapped_point_feasible': worst_overall < 1e-6,
            'violations_by_family': fams,
            'nonlinear_model_worst_violation_by_family': prod_worst,
            'nonlinear_model_worst_violation_overall': max(prod_worst.values(), default=0.0),
            'n_vars': sum(len(list(v.values()))
                          for v in blk.component_objects(pe.Var, active=None)),
            'n_constraints': sum(len(list(c.values()))
                                 for c in blk.component_objects(pe.Constraint, active=True)),
        }
        r = report['cases'][tag]
        print(f"    nonlinear objective            = {q_nl:.9f}")
        print(f"    relaxed obj @ raw mapped point = {q_raw:.9f}")
        print(f"    dropped complementarity term   = {dropped:.9e}")
        print(f"    EXACT accounting residual      = {residual_raw:.3e}  "
              f"exact={r['accounting_exact']}")
        print(f"    relaxed obj @ projected point  = {q_relaxed_at_point:.9f}"
              f"   (shift {residual:.3e})")
        if projection:
            print("    bound projection (max |shift|): " + ', '.join(
                f'{k}={v:.2e}' for k, v in sorted(
                    projection.items(), key=lambda kv: -kv[1])[:5]))
        print(f"    nonlinear model's own worst violation = "
              f"{r['nonlinear_model_worst_violation_overall']:.3e}")
        print(f"    worst mapped-point violation = {worst_overall:.3e}  "
              f"feasible={r['mapped_point_feasible']}")
        print(f"    model size: {r['n_vars']} vars, {r['n_constraints']} constraints")
        bad = sorted(((v['max_violation'], k) for k, v in fams.items()), reverse=True)[:5]
        for viol, name in bad:
            print(f"      {name:28s} max_violation={viol:.3e} "
                  f"(worst {fams[name]['worst_index']})")

    out = os.path.join(OUT_DIR, 'p55c_c6_mapping.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'\n[C6] report -> {out}')


if __name__ == '__main__':
    main()
