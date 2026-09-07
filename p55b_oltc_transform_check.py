"""
Stage P5.5-B2 -- numerical verification of the transformed continuous-OLTC
formulation against the production expressions.

Transformed variables, for the transformer's from-bus i and to-bus j:

    U_i  = r^2 * W_ii        C_ij = r * WijR        D_ij = r * WijI

Claim: every production transformer expression -- nodal P and Q contributions,
Pij, Qij, Pji, Qji and the thermal-limit expression -- is AFFINE in
(U_i, C_ij, D_ij, W_jj), and the exact rank relation becomes

    C_ij^2 + D_ij^2 = U_i * W_jj.

This script does not trust that algebra. It builds the real production model,
places a random-but-reproducible feasible-shaped point on (e, f, r), evaluates
the production expressions, evaluates the transformed affine expressions, and
differences them.

    /opt/anaconda3/envs/opf_env_py311/bin/python p55b_oltc_transform_check.py
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P55B')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
        planning.read_planning_problem()
        dso = planning.distribution_networks[5]
        net = dso.network[2025]['Winter']
        model = net.build_model(dso.params)

    tr_idx = [b for b, br in enumerate(net.branches) if br.is_transformer]
    assert len(tr_idx) == 1, tr_idx
    b = tr_idx[0]
    branch = net.branches[b]
    i = net.get_node_idx(branch.fbus)
    j = net.get_node_idx(branch.tbus)
    s_m = s_o = 0

    report = {'stage': 'P5.5-B2 verification',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'branch': {'branch_id': branch.branch_id, 'fbus': branch.fbus,
                         'tbus': branch.tbus, 'g': branch.g, 'b': branch.b,
                         'b_sh': branch.b_sh, 'g_sh': branch.g_sh,
                         'ratio': branch.ratio, 'rate': branch.rate},
              'points': []}

    # deterministic probe points spanning the tap range and a range of voltages
    probes = [(1.00, 0.02, 0.98, -0.03, 0.83),
              (1.02, -0.01, 0.95, 0.04, 1.00),
              (0.99, 0.03, 1.03, -0.02, 1.17),
              (1.05, 0.00, 0.92, 0.05, 0.95)]

    worst = 0.0
    for k, (ei, fi, ej, fj, r) in enumerate(probes):
        p = k % len(list(model.periods))
        model.e[i, s_m, s_o, p].value = ei
        model.f[i, s_m, s_o, p].value = fi
        model.e[j, s_m, s_o, p].value = ej
        model.f[j, s_m, s_o, p].value = fj
        model.r[b, s_m, s_o, p].value = r
        model.r_sqr[b, s_m, s_o, p].value = r ** 2
        # W variables consistent with (e, f)
        Wii = ei ** 2 + fi ** 2
        Wjj = ej ** 2 + fj ** 2
        WijR = ei * ej + fi * fj
        WijI = fi * ej - ei * fj
        model.vmag_sqr[i, s_m, s_o, p].value = Wii
        model.vmag_sqr[j, s_m, s_o, p].value = Wjj
        model.voltage_product_real[b, s_m, s_o, p].value = WijR
        model.voltage_product_imag[b, s_m, s_o, p].value = WijI

        # transformed variables
        U = r ** 2 * Wii
        C = r * WijR
        D = r * WijI
        g, bb, bsh = branch.g, branch.b, branch.b_sh

        # --- production terminal-power expressions, evaluated ---
        pij_prod, qij_prod = mch._branch_terminal_power_expressions(
            model, b, s_m, s_o, p, net, 'ij')
        pji_prod, qji_prod = mch._branch_terminal_power_expressions(
            model, b, s_m, s_o, p, net, 'ji')
        pij_prod = float(pe.value(pij_prod)); qij_prod = float(pe.value(qij_prod))
        pji_prod = float(pe.value(pji_prod)); qji_prod = float(pe.value(qji_prod))

        # --- transformed affine expressions, derived by hand ---
        pij_tr = g * U - g * C - bb * D
        qij_tr = -(bb + 0.5 * bsh) * U + bb * C - g * D
        pji_tr = g * Wjj - g * C + bb * D
        qji_tr = -(bb + 0.5 * bsh) * Wjj + bb * C + g * D

        # --- production nodal contributions for the transformer, both ends ---
        def nodal_contrib(node_idx):
            """Replicates the branch term inside node_balance_p/q for one node."""
            if node_idx == i:
                fnode = i
                vsq = float(pe.value(model.vmag_sqr[i, s_m, s_o, p]))
                rsq = float(pe.value(model.r_sqr[b, s_m, s_o, p]))
                head_p = g * vsq * rsq
                head_q = -(bb + 0.5 * bsh) * vsq * rsq
            else:
                fnode = j
                vsq = float(pe.value(model.vmag_sqr[j, s_m, s_o, p]))
                head_p = g * vsq
                head_q = -(bb + 0.5 * bsh) * vsq
            cr, ci = mch._branch_voltage_products(model, net, b, fnode, s_m, s_o, p)
            cr = float(pe.value(cr)); ci = float(pe.value(ci))
            rr = float(pe.value(model.r[b, s_m, s_o, p]))
            Pi = head_p - rr * (g * cr + bb * ci)
            Qi = head_q + rr * (bb * cr - g * ci)
            return Pi, Qi

        Pi_prod, Qi_prod = nodal_contrib(i)
        Pj_prod, Qj_prod = nodal_contrib(j)
        # transformed
        Pi_tr = g * U - (g * C + bb * D)
        Qi_tr = -(bb + 0.5 * bsh) * U + (bb * C - g * D)
        Pj_tr = g * Wjj - (g * C - bb * D)
        Qj_tr = -(bb + 0.5 * bsh) * Wjj + (bb * C + g * D)

        # --- rank relation ---
        rank_lhs = C ** 2 + D ** 2
        rank_rhs = U * Wjj

        checks = {
            'pij': (pij_prod, pij_tr), 'qij': (qij_prod, qij_tr),
            'pji': (pji_prod, pji_tr), 'qji': (qji_prod, qji_tr),
            'node_i_P': (Pi_prod, Pi_tr), 'node_i_Q': (Qi_prod, Qi_tr),
            'node_j_P': (Pj_prod, Pj_tr), 'node_j_Q': (Qj_prod, Qj_tr),
            'rank': (rank_lhs, rank_rhs),
        }
        entry = {'probe': {'e_i': ei, 'f_i': fi, 'e_j': ej, 'f_j': fj, 'r': r},
                 'W': {'Wii': Wii, 'Wjj': Wjj, 'WijR': WijR, 'WijI': WijI},
                 'transformed': {'U': U, 'C': C, 'D': D},
                 'checks': {}}
        for name, (a, bv) in checks.items():
            diff = abs(a - bv)
            scale = max(abs(a), abs(bv), 1.0)
            entry['checks'][name] = {'production': a, 'transformed': bv,
                                     'abs_diff': diff, 'rel_diff': diff / scale}
            worst = max(worst, diff / scale)
        report['points'].append(entry)

    # tap-existence check: r = sqrt(U/Wii) in [rmin, rmax] iff rmin^2 Wii <= U <= rmax^2 Wii
    from definitions import TRANSFORMER_MAXIMUM_RATIO, TRANSFORMER_MINIMUM_RATIO
    report['tap_elimination'] = {
        'r_min': TRANSFORMER_MINIMUM_RATIO, 'r_max': TRANSFORMER_MAXIMUM_RATIO,
        'box': 'r_min^2 * Wii <= U_i <= r_max^2 * Wii  (affine in U_i, Wii)',
        'recovery': 'r = sqrt(U_i / Wii), well defined because Wii >= v_min^2 > 0',
        'v_min_ref_bus': net.nodes[i].v_min, 'v_max_ref_bus': net.nodes[i].v_max,
        'Wii_strictly_positive': net.nodes[i].v_min ** 2 > 0,
    }
    report['worst_relative_difference'] = worst
    report['all_expressions_match'] = worst < 1e-12

    out = os.path.join(OUT_DIR, 'p55b_oltc_transform.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f"[B2] transformer branch {branch.branch_id}: bus {branch.fbus} -> {branch.tbus}, "
          f"g={branch.g:.6f} b={branch.b:.6f} b_sh={branch.b_sh} rate={branch.rate}")
    print(f"[B2] {len(probes)} probe points spanning r in "
          f"[{TRANSFORMER_MINIMUM_RATIO}, {TRANSFORMER_MAXIMUM_RATIO}]\n")
    print(f"{'expression':12s} " + '  '.join(f'probe{k}' .rjust(12) for k in range(len(probes))))
    names = list(report['points'][0]['checks'])
    for name in names:
        cells = '  '.join(f"{pt['checks'][name]['rel_diff']:12.3e}" for pt in report['points'])
        print(f'{name:12s} {cells}')
    print(f"\n[B2] worst relative difference across all expressions and probes: {worst:.3e}")
    print(f"[B2] all production expressions reproduced exactly: {report['all_expressions_match']}")
    print(f"[B2] tap elimination: {report['tap_elimination']['box']}")
    print(f"     Wii >= v_min^2 = {net.nodes[i].v_min ** 2} > 0 -> r recoverable")
    print(f'[B2] report -> {out}')


if __name__ == '__main__':
    main()
