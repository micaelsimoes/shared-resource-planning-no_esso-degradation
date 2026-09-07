"""
Stage P5.4-H1.2/H1.3 -- zero-capacity lifecycle and derivative/rank audit for the
dimensionless charge/discharge complementarity.

H1 rewrites only the complementarity inequality, from the physically tiny
`pch * pdch <= eps * S^2` (RHS ~ 1e-12 at bootstrap capacity) to the O(1)
`pch_hat * pdch_hat <= eps` (RHS = 1e-4), linked by

    pch  - S_rated * pch_hat  == 0
    pdch - S_rated * pdch_hat == 0

which never divides by the rating and keeps a unit coefficient on the physical
variable. For positive capacity this is an exact reformulation.

Sections
  A  exact-reformulation check: the new row reproduces the old feasible set
  B  zero-capacity / reused-model lifecycle (H1.2)
  C  derivative and rank audit on representative DSO and TSO models (H1.3)

    python p54h1_normalized_complementarity_audit.py
"""

import inspect
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import numpy as np
import pyomo.environ as pe
from pyomo.core.expr.calculus.derivatives import Modes, differentiate

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from definitions import ESS_COMPLEMENTARITY_TOLERANCE, SHARED_ESS_ZERO_CAPACITY_TOLERANCE  # noqa: E402
from p53b3_active_power_ess import jacobian_for  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54H1')

LINK_ROWS = ('sess_pch_hat_link', 'sess_pdch_hat_link')
HAT_VARS = ('shared_es_pch_hat', 'shared_es_pdch_hat')
TRACKED = LINK_ROWS + HAT_VARS + (
    'sess_comp', 'sess_pnet_def', 'sess_converter_capability',
    'sess_active_sum_limit', 'sess_soc_def', 'shared_es_pch', 'shared_es_pdch',
    'shared_es_s_rated', 'shared_es_s_rated_fixed')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def src(obj):
    try:
        return inspect.getsource(obj)
    except Exception as error:
        return f'<unavailable: {error}>'


def grad_norm(expr, variables):
    """Analytic gradient of `expr` w.r.t. `variables` (reverse-mode AD)."""
    try:
        grads = differentiate(expr, wrt_list=list(variables), mode=Modes.reverse_numeric)
    except Exception as error:
        return {'error': f'{type(error).__name__}: {error}'}
    values = [float(g) for g in grads]
    return {'entries': values, 'l2_norm': float(np.linalg.norm(values)),
            'max_abs': float(max(abs(v) for v in values)) if values else 0.0}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-H1.2/H1.3', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'eps': ESS_COMPLEMENTARITY_TOLERANCE,
              'rule_sources': {
                  'sess_comp_rule': src(mch.sess_comp_rule),
                  'sess_pch_hat_link_rule': src(mch.sess_pch_hat_link_rule),
                  'sess_pdch_hat_link_rule': src(mch.sess_pdch_hat_link_rule),
                  'ess_comp_rule': src(mch.ess_comp_rule),
                  'ess_pch_hat_link_rule': src(mch.ess_pch_hat_link_rule)}}

    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    dso = planning.distribution_networks[9]
    network = dso.network[2030]['Winter']
    e = network.get_shared_energy_storage_idx(network.get_reference_node_id())
    s_pu = abs(candidate['total_capacity'][9][2030]['s']) / network.baseMVA
    e_pu = abs(candidate['total_capacity'][9][2030]['e']) / network.baseMVA

    console = io.StringIO()
    with redirect_stdout(console):
        model = network.build_model(dso.params)
        mch.configure_shared_ess_operational_state(model, e, s_pu, e_pu)

    # ---------------- A. exact-reformulation check ----------------
    # Drive the model to states straddling the OLD boundary pch*pdch = eps*S^2
    # and confirm the NEW row's margin has the same sign at each one.
    s_m, s_o, p = 0, 0, 0
    eps = ESS_COMPLEMENTARITY_TOLERANCE

    def place(pch, pdch):
        model.shared_es_pch[e, s_m, s_o, p].value = pch
        model.shared_es_pdch[e, s_m, s_o, p].value = pdch
        model.shared_es_pch_hat[e, s_m, s_o, p].value = pch / s_pu
        model.shared_es_pdch_hat[e, s_m, s_o, p].value = pdch / s_pu
        old_margin = eps * s_pu ** 2 - pch * pdch
        new_body = float(pe.value(model.sess_comp[e, s_m, s_o, p].body))
        new_margin = eps - new_body
        link_ch = float(pe.value(model.sess_pch_hat_link[e, s_m, s_o, p].body))
        link_dch = float(pe.value(model.sess_pdch_hat_link[e, s_m, s_o, p].body))
        return {'pch': pch, 'pdch': pdch,
                'old_margin_eps_S2_minus_product': old_margin,
                'new_row_body_hat_product': new_body,
                'new_margin_eps_minus_body': new_margin,
                'signs_agree': (old_margin > 0) == (new_margin > 0),
                'ratio_old_over_new_margin': (old_margin / new_margin)
                if abs(new_margin) > 0 else None,
                'link_ch_residual': link_ch, 'link_dch_residual': link_dch,
                'links_satisfied': abs(link_ch) < 1e-18 and abs(link_dch) < 1e-18}

    on_boundary = s_pu * eps ** 0.5   # pch = pdch = S*sqrt(eps) sits exactly on it
    equivalence = {
        'strictly_interior': place(0.5 * on_boundary, 0.5 * on_boundary),
        'on_old_boundary': place(on_boundary, on_boundary),
        'strictly_violating': place(2.0 * on_boundary, 2.0 * on_boundary),
        'asymmetric_interior': place(0.9 * s_pu, 1e-5 * s_pu),
        'asymmetric_violating': place(0.9 * s_pu, 1e-2 * s_pu),
    }
    report['A_exact_reformulation'] = {
        'boundary_power_pu': on_boundary,
        'sqrt_eps': eps ** 0.5,
        'points': equivalence,
        'all_signs_agree': all(v['signs_agree'] for v in equivalence.values()),
        'all_links_satisfied': all(v['links_satisfied'] for v in equivalence.values()),
        'note': ('the two margins differ by exactly the factor S_rated^2, which is '
                 'why the new row is resolvable and the old one was not'),
    }

    # ---------------- B. zero-capacity / reused-model lifecycle ----------------
    def fingerprint():
        return {n: id(getattr(model, n)) for n in TRACKED if hasattr(model, n)}

    def state(label):
        pch_hat = model.shared_es_pch_hat[e, s_m, s_o, p]
        pdch_hat = model.shared_es_pdch_hat[e, s_m, s_o, p]
        return {
            'label': label,
            's_rated_fixed': float(pe.value(model.shared_es_s_rated_fixed[e])),
            'pch_fixed': bool(model.shared_es_pch[e, s_m, s_o, p].fixed),
            'pch_hat_fixed': bool(pch_hat.fixed),
            'pdch_hat_fixed': bool(pdch_hat.fixed),
            'pch_hat_value': float(pe.value(pch_hat)),
            'pdch_hat_value': float(pe.value(pdch_hat)),
            'pch_hat_bounds': list(pch_hat.bounds),
            'link_rows_active': all(
                bool(getattr(model, n)[e, s_m, s_o, p].active) for n in LINK_ROWS),
            'comp_row_active': bool(model.sess_comp[e, s_m, s_o, p].active),
            'link_ch_residual': float(pe.value(
                model.sess_pch_hat_link[e, s_m, s_o, p].body)),
        }

    prints = [fingerprint()]
    lifecycle = []
    for label, sv, ev in (('zero', 0.0, 0.0),
                          ('positive', s_pu, e_pu),
                          ('different_positive', 2.5 * s_pu, 2.5 * e_pu),
                          ('back_to_zero', 0.0, 0.0),
                          ('positive_again', s_pu, e_pu)):
        console = io.StringIO()
        with redirect_stdout(console):
            model.shared_es_s_rated_fixed[e].set_value(sv)
            model.shared_es_e_rated_fixed[e].set_value(ev)
            mch.configure_shared_ess_operational_state(model, e, sv, ev)
        lifecycle.append(state(label))
        prints.append(fingerprint())

    report['B_lifecycle'] = {
        'transitions': lifecycle,
        'component_ids_constant': all(f == prints[0] for f in prints),
        'components_tracked': sorted(prints[0].keys()),
        'no_division_by_capacity_in_any_row': not any(
            '/' in str(getattr(model, n)[e, s_m, s_o, p].body) for n in LINK_ROWS),
        'comp_row_body_has_no_division': '/' not in str(
            model.sess_comp[e, s_m, s_o, p].body),
    }

    # ---------------- C. derivative and rank audit ----------------
    def audit(tag, net, params, s_val, e_val, idx):
        console = io.StringIO()
        with redirect_stdout(console):
            m = net.build_model(params)
            m.shared_es_s_rated_fixed[idx].set_value(s_val)
            m.shared_es_e_rated_fixed[idx].set_value(e_val)
            mch.configure_shared_ess_operational_state(m, idx, s_val, e_val)

        pch = m.shared_es_pch[idx, 0, 0, 0]
        pdch = m.shared_es_pdch[idx, 0, 0, 0]
        pch_hat = m.shared_es_pch_hat[idx, 0, 0, 0]
        pdch_hat = m.shared_es_pdch_hat[idx, 0, 0, 0]
        s_rated = m.shared_es_s_rated[idx]
        wrt = [pch, pdch, pch_hat, pdch_hat, s_rated]

        # cold start: everything at its initialized value (zero dispatch)
        link_ch = m.sess_pch_hat_link[idx, 0, 0, 0]
        link_dch = m.sess_pdch_hat_link[idx, 0, 0, 0]
        comp = m.sess_comp[idx, 0, 0, 0]

        out = {
            'tag': tag, 's_rated_pu': s_val,
            'link_ch_gradient': grad_norm(link_ch.body, wrt),
            'link_dch_gradient': grad_norm(link_dch.body, wrt),
            'comp_gradient_at_cold_start': grad_norm(comp.body, wrt),
            'comp_rhs': float(pe.value(comp.upper)),
            'comp_body_at_cold_start': float(pe.value(comp.body)),
            'wrt_order': ['pch', 'pdch', 'pch_hat', 'pdch_hat', 's_rated'],
        }
        # curvature: d2/dpch_hat dpdch_hat of the normalized product is exactly 1,
        # versus d2/dpch dpdch of the old product, also 1 -- but the OLD row's RHS
        # scaled as S^2 while the new one does not. Measure the gradient at a
        # representative interior point instead of only at the cold start.
        pch_hat.value = 0.5
        pdch_hat.value = 0.5
        pch.value = 0.5 * s_val
        pdch.value = 0.5 * s_val
        out['comp_gradient_at_half_rating'] = grad_norm(comp.body, wrt)
        out['comp_body_at_half_rating'] = float(pe.value(comp.body))
        out['old_row_rhs_would_have_been'] = ESS_COMPLEMENTARITY_TOLERANCE * s_val ** 2
        out['rhs_amplification_vs_old'] = (
            ESS_COMPLEMENTARITY_TOLERANCE / (ESS_COMPLEMENTARITY_TOLERANCE * s_val ** 2))
        pch_hat.value = 0.0
        pdch_hat.value = 0.0
        pch.value = 0.0
        pdch.value = 0.0

        # gradients w.r.t. the physical variables must stay +1 on the link rows
        out['link_ch_unit_coefficient_on_pch'] = abs(
            out['link_ch_gradient']['entries'][0] - 1.0) < 1e-15
        out['link_dch_unit_coefficient_on_pdch'] = abs(
            out['link_dch_gradient']['entries'][1] - 1.0) < 1e-15

        spec = jacobian_for(m, tag)
        full = spec['full']
        out['rank'] = {
            'n_equality_rows': full.get('n_rows'),
            'n_exactly_zero_rows': full.get('n_exactly_zero_rows'),
            'zero_row_components': spec.get('zero_row_components'),
            'n_derivative_failures': spec.get('n_derivative_failures'),
            'sigma_min_full': full.get('sigma_min'),
            'reduced_condition_number': spec.get('reduced', {}).get('condition_number'),
            'full_row_rank': full.get('n_exactly_zero_rows') == 0,
            'link_rows_among_zero_gradient': any(
                c in LINK_ROWS for c in spec.get('zero_row_components', {})),
        }
        return out

    audits = {}
    audits['dso/case33_1/2030/Winter'] = audit(
        'case33_1/2030/Winter',
        planning.distribution_networks[9].network[2030]['Winter'],
        planning.distribution_networks[9].params, s_pu, e_pu, e)

    tso_net = planning.transmission_network.network[2025]['Winter']
    tso_e = 0
    tso_node = planning.transmission_network.active_distribution_network_nodes[tso_e]
    tso_s = abs(candidate['total_capacity'][tso_node][2025]['s']) / tso_net.baseMVA
    tso_en = abs(candidate['total_capacity'][tso_node][2025]['e']) / tso_net.baseMVA
    audits['tso/case9/2025/Winter'] = audit(
        'case9/2025/Winter', tso_net, planning.transmission_network.params,
        tso_s, tso_en, tso_e)

    report['C_derivative_rank'] = audits

    out = os.path.join(OUT_DIR, 'p54h1_audit_report.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'\n[P5.4-H1] audit -> {out}')
    a = report['A_exact_reformulation']
    print(f"  exact reformulation: signs agree at all 5 probe points = {a['all_signs_agree']}; "
          f"links satisfied = {a['all_links_satisfied']}")
    for name, v in a['points'].items():
        print(f"   {name:22s} old_margin={v['old_margin_eps_S2_minus_product']:+.4e} "
              f"new_margin={v['new_margin_eps_minus_body']:+.4e} agree={v['signs_agree']}")
    b = report['B_lifecycle']
    print(f"  lifecycle: component ids constant = {b['component_ids_constant']}; "
          f"no division by capacity = {b['no_division_by_capacity_in_any_row']}")
    for t in b['transitions']:
        print(f"   {t['label']:20s} S={t['s_rated_fixed']:.4e} "
              f"hat_fixed=({t['pch_hat_fixed']},{t['pdch_hat_fixed']}) "
              f"hat_val=({t['pch_hat_value']:.1e},{t['pdch_hat_value']:.1e}) "
              f"bounds={t['pch_hat_bounds']} rows_active={t['link_rows_active']}/"
              f"{t['comp_row_active']} link_res={t['link_ch_residual']:.1e}")
    for tag, v in report['C_derivative_rank'].items():
        r = v['rank']
        print(f"  {tag}:")
        print(f"    link rows: |grad| ch={v['link_ch_gradient']['l2_norm']:.4e} "
              f"dch={v['link_dch_gradient']['l2_norm']:.4e}; "
              f"unit coeff on physical = "
              f"{v['link_ch_unit_coefficient_on_pch']}/{v['link_dch_unit_coefficient_on_pdch']}")
        print(f"    comp row: RHS={v['comp_rhs']:.4e} (old would be "
              f"{v['old_row_rhs_would_have_been']:.4e}, x{v['rhs_amplification_vs_old']:.4e}); "
              f"|grad| cold={v['comp_gradient_at_cold_start']['l2_norm']:.4e} "
              f"half-rating={v['comp_gradient_at_half_rating']['l2_norm']:.4e}")
        print(f"    rank: rows={r['n_equality_rows']} zero_grad_rows={r['n_exactly_zero_rows']} "
              f"owners={r['zero_row_components']} sigma_min={r['sigma_min_full']:.4e} "
              f"full_rank={r['full_row_rank']} link_rows_zero_grad={r['link_rows_among_zero_gradient']}")


if __name__ == '__main__':
    main()
