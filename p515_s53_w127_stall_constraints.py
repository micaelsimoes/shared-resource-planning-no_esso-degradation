"""P5.15 W127 -- which constraints pin the interface copies at the F2 challenger pf stall. ZERO SOLVES.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any Pyomo / production import and verified at exactly
0 at the end. Model LOADS are permitted (task W127): the persisted terminal models are unpickled and only READ (values,
bounds, Params, the IPOPT suffixes `dual`, `ipopt_zL_out`, `ipopt_zU_out`); two finite-difference curvature probes set
one Var value temporarily and restore it bit-exactly (asserted). Nothing is built for solving and nothing is solved.

INSTANCE: campaign s53_w118_resettle_r2_f2_challenger (cell spec c7c9aea8), eval key 1fe91e86f11e76af..., candidate key
4032a138... (f2_challenger: 2030, n5 (0.25 MVA, 1.0 MWh), n7 (1.0, 3.0), n9 (0, 0), flexibility price x2); terminal
models of cycle 261 (cap). certified_models.pkl sha256 a8974231..., esso_models_s39_D.pkl sha256 0fb6cc1d... -- both
verified against the campaign manifest BEFORE unpickling. Contrast: SRP1 x = 0, campaign s53_w101_srp1_cont_x0, eval
d110bd1a5977df1e..., certified_models.pkl sha256 99ab1070... (verified against that campaign's manifest).
Block: 2030 Autumn, hour 4 = period index 3 (ESSO index y=1, d=2 with DAYS_ORDER Spring, Summer, Autumn, Winter).

DEFINITIONS (fixed before the run):
  active       a bound b (Var lb/ub, or a row's lower/upper) with |value - b| <= 1e-6 * max(1, |b|).
  KKT sign     established on the stored suffixes (check `kkt_check`, every block):  grad f = J^T lambda + zL + zU,
               zL >= 0 (lb), zU <= 0 (ub), f = the ACTIVE objective (p58_rescaled_admm_objective, EUR per
               representative day, variables in p.u. on baseMVA).  A row <= up that is active has lambda <= 0.
  lever cost   first-order change of the active objective per unit move s (+1 / -1) of one Var v, with every
               EQUALITY row re-satisfied at its multiplier and active inequalities / bounds left to go slack:
                   dJ = s * ( sum_{active inequality rows i} lambda_i * dc_i/dv  + zL_v + zU_v )
               FEASIBLE only if no active bound / inequality is violated by the move (else `blocked_by` lists them).
               It is a one-sided, first-order, single-multiplier-set quantity: at a degenerate point the true
               directional cost can be larger, and the quadratic AL terms (second order) are NOT in it.
  AL curvature d2J/dE2 of the active objective in one consensus Var E, by a central difference of the analytic
               gradient (h = 1e-7 p.u.); = admm_objective_scale * rho / R^2 for the pf term.
  dead-zone rate  per-cycle change of the pf AL gradient with both copies frozen = curvature_pf * |x - z|
               (dual update y += rho (x - z)/R, W126 verified slope / rho r = 1.00000004).
  EUR/MWh      rescaled EUR/pu divided by baseMVA (100, derived as settlement gradient / recorded hour-4 price).

Smoke (nothing written): add --dry.
Output (write-once, new directory): data/SRP1/Results/P515S53/w127_stall_constraints/
    w127_stall_constraints.json, launch.log (captured by the launcher), manifest_sha256.json (--manifest)
Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S53/w127_stall_constraints && set -o noclobber && \\
    nice -n 10 /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w127_stall_constraints.py \\
        > data/SRP1/Results/P515S53/w127_stall_constraints/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w127_stall_constraints.py --manifest
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W127 stall constraints zero-solve').install()

import numpy as np  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

THIS = os.path.abspath(__file__)
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S53', 'w127_stall_constraints')
OUT_JSON = os.path.join(OUT_DIR, 'w127_stall_constraints.json')
CH_CAMPAIGN = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w118_resettle', 'campaign_s53_w118_resettle_r2_f2_challenger')
CH_EVAL = os.path.join(CH_CAMPAIGN, 'evals', '1fe91e86f11e76af_f2_challenger')
X0_CAMPAIGN = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0')
X0_EVAL = os.path.join(X0_CAMPAIGN, 'evals', 'd110bd1a5977df1e_x0')
INPUTS = {
    'ch_models': (os.path.join(CH_EVAL, 'certified_models.pkl'), CH_CAMPAIGN,
                  'a8974231f50bd6a62cf2faf1fabbdc9199143d229fbbac8be6f03313a833566c'),
    'ch_esso': (os.path.join(CH_EVAL, 'esso_models_s39_D.pkl'), CH_CAMPAIGN,
                '0fb6cc1d830817868c30736accfada18ac3f53e115e4adb51313595e4d4be0ff'),
    'ch_detail': (os.path.join(CH_EVAL, 'interface_settlement_detail_s31c.json'), CH_CAMPAIGN, None),
    'x0_models': (os.path.join(X0_EVAL, 'certified_models.pkl'), X0_CAMPAIGN,
                  '99ab1070b0e61cc7818975ce898069d2060d2668bae67c166b58913e9923c33a'),
}
INSTANCE = {
    'challenger': {'campaign': 's53_w118_resettle_r2_f2_challenger', 'cell_spec': 'c7c9aea8',
                   'eval_key': '1fe91e86f11e76af', 'candidate_key_prefix': '4032a138', 'label': 'f2_challenger',
                   'terminal_cycle': 261},
    'x0': {'campaign': 's53_w101_srp1_cont_x0', 'eval_key': 'd110bd1a5977df1e811a1b3963afc565de6daa65f76dcc096ead3fdc70b08546',
           'candidate_key': '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57', 'certification_cycle': 181},
}
YEAR, DAY, P = 2030, 'Autumn', 3
ESSO_Y, ESSO_D = 1, 2          # years [2025, 2030, 2035]; DAYS_ORDER Spring, Summer, Autumn, Winter (asserted in W126)
NODES = (5, 7, 9)
TOL = 1e-6
FD_H = 1e-7
DSO_LEVERS = (('flex_p_up', +1), ('flex_p_down', -1), ('pg', -1), ('shared_es_pdch', +1), ('shared_es_pch', -1))
TSO_LEVERS = (('pg', -1), ('shared_es_pdch', -1), ('shared_es_pch', +1))
ROW_FAMILIES_DSO = ('node_balance_p', 'node_balance_q', 'expected_interface_pf_p_def', 'expected_interface_pf_q_def',
                    'expected_interface_vmag_def', 'voltage_magnitude_upper_cons', 'voltage_magnitude_lower_cons',
                    'branch_flow_limit', 'branch_flow_limit_ji', 'sg_capability', 'gen_pf_upper', 'gen_pf_lower',
                    'sess_pnet_def', 'sess_pch_hat_link', 'sess_pdch_hat_link', 'sess_converter_capability',
                    'sess_active_sum_limit', 'sess_soc_def', 'sess_soc_limit_upper', 'sess_soc_limit_lower', 'sess_comp',
                    'expected_shared_ess_p_def', 'r_sqr_def')
VAR_FAMILIES = ('pg', 'qg', 'flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down', 'vmag', 'r', 'shared_es_pnet',
                'shared_es_pch', 'shared_es_pdch', 'shared_es_soc', 'shared_es_qnet', 'interface_delta_p',
                'interface_delta_q', 'pc', 'expected_interface_pf_p', 'pij', 'pji')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[{_utc()}] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _verify_inputs():
    out = {}
    for key, (rel, campaign, declared) in INPUTS.items():
        man_rel = os.path.join(campaign, 'campaign_manifest_sha256.json')
        pinned = json.load(open(os.path.join(REPO, man_rel))).get(rel)
        now = _sha(os.path.join(REPO, rel))
        if pinned is None or pinned != now or (declared is not None and declared != now):
            raise RuntimeError(f'{rel}: sha256 {now} vs manifest {pinned} vs declared {declared}')
        out[rel] = {'sha256': now, 'manifest': man_rel, 'declared_in_task': declared}
    return out


def _f(x):
    return None if x is None else float(x)


def _at(val, b):
    return b is not None and val is not None and abs(val - b) <= TOL * max(1.0, abs(b))


def _hour_index(idx):
    return isinstance(idx, tuple) and len(idx) >= 1 and idx[-1] == P or idx == P


class Block:
    """Read-only KKT view of one persisted block (active objective, rows, suffixes)."""

    def __init__(self, m, pe, differentiate, Modes, identify_variables):
        self.m, self.pe = m, pe
        self._diff, self._modes, self._ident = differentiate, Modes, identify_variables
        objs = [o for o in m.component_data_objects(pe.Objective, active=True)]
        if len(objs) != 1:
            raise RuntimeError(f'{len(objs)} active objectives')
        self.obj = objs[0]
        self.vars = [v for v in m.component_data_objects(pe.Var) if not v.fixed]
        self.col = {id(v): i for i, v in enumerate(self.vars)}
        self.grad = self._grad(self.obj.expr)
        self.rows_of = {}          # id(var) -> list of (row, d row / d var)
        self.jtl = np.zeros(len(self.vars))
        self.n_rows, self.n_rows_no_dual = 0, 0
        for c in m.component_data_objects(pe.Constraint, active=True):
            self.n_rows += 1
            lam = m.dual.get(c)
            if lam is None:
                self.n_rows_no_dual += 1
                lam = 0.0
            vs = list(identify_variables(c.body, include_fixed=False))
            if not vs:
                continue
            for v, d in zip(vs, differentiate(c.body, wrt_list=vs, mode=Modes.reverse_numeric)):
                self.rows_of.setdefault(id(v), []).append((c, d))
                self.jtl[self.col[id(v)]] += d * lam
        self.zl = np.array([m.ipopt_zL_out.get(v, 0.0) for v in self.vars])
        self.zu = np.array([m.ipopt_zU_out.get(v, 0.0) for v in self.vars])

    def _grad(self, expr):
        g = np.zeros(len(self.vars))
        vs = list(self._ident(expr, include_fixed=False))
        for v, d in zip(vs, self._diff(expr, wrt_list=vs, mode=self._modes.reverse_numeric)):
            g[self.col[id(v)]] += d
        return g

    def kkt_check(self):
        r = self.grad - self.jtl - self.zl - self.zu
        return {'convention': 'grad f = J^T lambda + zL + zU', 'max_abs_residual': float(np.abs(r).max()),
                'max_abs_grad': float(np.abs(self.grad).max()),
                'relative': float(np.abs(r).max() / max(1.0, np.abs(self.grad).max())),
                'n_rows': self.n_rows, 'n_rows_without_dual': self.n_rows_no_dual,
                'alt_sign_max_abs_residual': float(np.abs(self.grad + self.jtl + self.zl + self.zu).max())}

    def dobj(self, v):
        return float(self.grad[self.col[id(v)]])

    def row_info(self, c):
        pe = self.pe
        body = float(pe.value(c.body))
        lo = _f(pe.value(c.lower)) if c.lower is not None else None
        up = _f(pe.value(c.upper)) if c.upper is not None else None
        return {'row': c.name, 'body': body, 'lower': lo, 'upper': up, 'equality': bool(c.equality),
                'dual': _f(self.m.dual.get(c)), 'active_lower': bool(_at(body, lo)), 'active_upper': bool(_at(body, up))}

    def var_info(self, v):
        val = v.value
        return {'var': v.name, 'value': _f(val), 'lb': _f(v.lb), 'ub': _f(v.ub), 'fixed': bool(v.fixed),
                'zL': _f(self.m.ipopt_zL_out.get(v, 0.0)), 'zU': _f(self.m.ipopt_zU_out.get(v, 0.0)),
                'at_lb': bool(_at(val, v.lb)), 'at_ub': bool(_at(val, v.ub))}

    def lever(self, v, s):
        """First-order one-sided cost of moving v by s (see module docstring)."""
        info = self.var_info(v)
        blocked, cost_terms = [], []
        if info['at_lb'] and s < 0:
            blocked.append(f'{v.name} at lb {v.lb}')
        if info['at_ub'] and s > 0:
            blocked.append(f'{v.name} at ub {v.ub}')
        dj = s * (info['zL'] + info['zU'])
        for c, d in self.rows_of.get(id(v), []):
            if c.equality:
                continue
            ri = self.row_info(c)
            if not (ri['active_lower'] or ri['active_upper']):
                continue
            if (ri['active_upper'] and s * d > 0) or (ri['active_lower'] and s * d < 0):
                blocked.append(f"{c.name} ({'upper' if ri['active_upper'] else 'lower'}, dual {ri['dual']:.6g})")
            dj += s * (ri['dual'] or 0.0) * d
            cost_terms.append({'row': c.name, 'dual': ri['dual'], 'd_row_d_var': float(d)})
        return {'var': v.name, 'direction': s, 'value': info['value'], 'lb': info['lb'], 'ub': info['ub'],
                'zL': info['zL'], 'zU': info['zU'], 'feasible': not blocked, 'blocked_by': blocked,
                'first_order_dJ_per_pu': float(dj), 'active_inequality_terms': cost_terms}

    def curvature(self, v):
        """d2J/dv2 by central difference of the analytic gradient; value restored bit-exactly."""
        pe = self.pe
        x0 = v.value
        vs = list(self._ident(self.obj.expr, include_fixed=False))
        k = [i for i, w in enumerate(vs) if w is v]
        if len(k) != 1:
            raise RuntimeError(f'{v.name} not in the objective')
        g = []
        for h in (FD_H, -FD_H):
            v.set_value(x0 + h, skip_validation=True)
            g.append(self._diff(self.obj.expr, wrt_list=[v], mode=self._modes.reverse_numeric)[0])
        v.set_value(x0, skip_validation=True)
        if v.value != x0:
            raise RuntimeError('value not restored')
        _ = pe
        return float((g[0] - g[1]) / (2 * FD_H))


def hour_tables(blk, families_rows):
    m, pe = blk.m, blk.pe
    vars_active, vars_listed = [], []
    for fam in VAR_FAMILIES:
        comp = m.component(fam)
        if comp is None or comp.ctype is not pe.Var:      # e.g. the DSO's `pc` is a Param
            continue
        for idx, v in comp.items():
            if not _hour_index(idx):
                continue
            vi = blk.var_info(v)
            vars_listed.append(vi)
            if vi['at_lb'] or vi['at_ub']:
                vars_active.append(vi)
    rows_active, rows_eq = [], []
    for fam in families_rows:
        comp = m.component(fam)
        if comp is None:
            continue
        for idx, c in comp.items():
            if not c.active or not _hour_index(idx):
                continue
            ri = blk.row_info(c)
            if ri['equality']:
                rows_eq.append(ri)
            elif ri['active_lower'] or ri['active_upper']:
                rows_active.append(ri)
    return {'vars_hour4': vars_listed, 'vars_hour4_active': vars_active, 'rows_hour4_active_inequality': rows_active,
            'rows_hour4_equality': rows_eq}


def family_summary(entries, key='var'):
    out = {}
    for e in entries:
        fam = e[key].split('[')[0]
        s = out.setdefault(fam, {'n': 0, 'n_at_lb': 0, 'n_at_ub': 0, 'n_active_upper': 0, 'n_active_lower': 0,
                                 'mult_min': None, 'mult_max': None})
        s['n'] += 1
        for f, k in (('at_lb', 'n_at_lb'), ('at_ub', 'n_at_ub'), ('active_upper', 'n_active_upper'),
                     ('active_lower', 'n_active_lower')):
            if e.get(f):
                s[k] += 1
        mult = e.get('dual') if key == 'row' else ((e['zL'] or 0.0) + (e['zU'] or 0.0))
        if mult is not None:
            s['mult_min'] = mult if s['mult_min'] is None else min(s['mult_min'], mult)
            s['mult_max'] = mult if s['mult_max'] is None else max(s['mult_max'], mult)
    return out


def dso_block(blk, tso_blk, dn, base_mva):
    m, pe = blk.m, blk.pe
    x = float(pe.value(m.expected_interface_pf_p[P]))
    z_req = float(pe.value(m.p_pf_req[P]))
    g_x = blk.dobj(m.expected_interface_pf_p[P])
    curv_pf = blk.curvature(m.expected_interface_pf_p[P])
    tabs = hour_tables(blk, ROW_FAMILIES_DSO + ('flex_energy_balance_p', 'sess_soc_final'))
    for fam in ('flex_energy_balance_p', 'sess_soc_final'):      # day-level rows (not hour-indexed)
        comp = m.component(fam)
        for c in (comp.values() if comp is not None else []):
            tabs.setdefault('rows_day_level', []).append(blk.row_info(c))
    levers = []
    for fam, s in DSO_LEVERS:
        comp = m.component(fam)
        if comp is None:
            continue
        for idx, v in comp.items():
            if not _hour_index(idx) or v.fixed:
                continue
            if fam == 'pg' and idx[0] == 0:          # reference generator = the interface itself
                continue
            levers.append(blk.lever(v, s))
    feas = [lv for lv in levers if lv['feasible']]
    cheapest = min(feas, key=lambda r: r['first_order_dJ_per_pu']) if feas else None
    flex_ub_sum = sum((v.ub or 0.0) - (v.value or 0.0) for idx, v in m.flex_p_up.items() if _hour_index(idx))
    ess_24 = None
    if m.component('shared_es_pnet') is not None and not all(v.fixed for v in m.shared_es_pnet.values()):
        ess_24 = {'pnet_pu': [float(m.shared_es_pnet[0, 0, 0, t].value) for t in sorted({k[-1] for k in m.shared_es_pnet.keys()})],
                  'soc_pu': [float(m.shared_es_soc[0, 0, 0, t].value) for t in sorted({k[-1] for k in m.shared_es_pnet.keys()})],
                  's_rated_pu': float(pe.value(m.shared_es_s_rated_fixed[0])),
                  'e_rated_pu': float(pe.value(m.shared_es_e_rated_fixed[0])),
                  'curvature_ess_al_hour4': blk.curvature(m.expected_shared_ess_p[P])}
    rate = curv_pf * abs(x - z_req)
    return {
        'node': dn, 'x_dso_copy_pu': x, 'z_tso_copy_req_pu': z_req, 'gap_x_minus_z_mw': (x - z_req) * base_mva,
        'dual_pf_p_req': float(pe.value(m.dual_pf_p_req[P])), 'rho_pf': float(pe.value(m.rho_pf)),
        'admm_objective_scale': float(pe.value(m.admm_objective_scale)),
        'pf_AL_gradient_on_x_eur_per_pu': g_x, 'pf_AL_gradient_on_x_eur_per_mwh': g_x / base_mva,
        'dual_expected_interface_pf_p_def': _f(m.dual.get(m.expected_interface_pf_p_def[P])),
        'settlement_gradient_eur_per_pu': float(blk._grad(m.interface_settlement.expr)[blk.col[id(m.pg[0, 0, 0, P])]]),
        'node_balance_p_dual_ref_bus': _f(m.dual.get(m.node_balance_p[0, 0, 0, P])),
        'node_balance_p_duals_all_buses': [_f(m.dual.get(m.node_balance_p[i, 0, 0, P])) for i in sorted({k[0] for k in m.node_balance_p.keys()})],
        'pf_curvature_eur_per_pu2': curv_pf, 'implied_R_pu': float(np.sqrt(blk.m.admm_objective_scale.value * pe.value(m.rho_pf) / curv_pf)),
        'dead_zone_rate_eur_per_pu_per_cycle': rate,
        'flex_p_up_headroom_hour4_pu': flex_ub_sum,
        'load_curtailment_vars_present': m.component('pc_curt_up') is not None,
        'levers_raise_import': levers, 'cheapest_feasible_lever': cheapest,
        'cycles_to_first_order_threshold_at_current_rate': (cheapest['first_order_dJ_per_pu'] / rate) if (cheapest and rate > 0) else None,
        'tables': tabs, 'var_family_summary_hour4_active': family_summary(tabs['vars_hour4_active']),
        'row_family_summary_hour4_active': family_summary(tabs['rows_hour4_active_inequality'], key='row'),
        'shared_ess_day': ess_24, 'kkt_check': blk.kkt_check()}


def tso_block(blk, base_mva):
    m, pe = blk.m, blk.pe
    per_dn = []
    for dn in m.active_distribution_networks:
        z = m.expected_interface_pf_p[dn, P]
        dv = m.interface_delta_p[dn, 0, 0, P]
        per_dn.append({'dn': dn, 'z_tso_copy_pu': float(z.value), 'x_dso_req_pu': float(pe.value(m.p_pf_req[dn, P])),
                       'pf_AL_gradient_on_z_eur_per_pu': blk.dobj(z),
                       'dual_expected_interface_pf_p_def': _f(m.dual.get(m.expected_interface_pf_p_def[dn, P])),
                       'settlement_gradient_on_delta_eur_per_pu': float(blk._grad(m.interface_settlement.expr)[blk.col[id(dv)]]),
                       'interface_delta_p': blk.var_info(dv), 'pc_anchor_fixed_pu': float(m.pc[dn, 0, 0, P].value),
                       'pf_curvature_eur_per_pu2': blk.curvature(z)})
    gens = []
    for g in sorted({k[0] for k in m.pg.keys()}):
        v = m.pg[g, 0, 0, P]
        vi = blk.var_info(v)
        vi['cost_gradient_eur_per_pu'] = blk.dobj(v)
        vi['pg_avail_param'] = float(pe.value(m.pg_avail[g, 0, P]))
        vi['class_by_objective_gradient'] = 'costed (conventional)' if vi['cost_gradient_eur_per_pu'] > 0 else (
            'zero-cost, available' if (v.ub or 0) > 0 else 'zero-cost, zero availability')
        gens.append(vi)
    tabs = hour_tables(blk, ROW_FAMILIES_DSO)
    levers = []
    for fam, s in TSO_LEVERS:
        for idx, v in m.component(fam).items():
            if _hour_index(idx) and not v.fixed:
                levers.append(blk.lever(v, s))
    feas = [lv for lv in levers if lv['feasible']]
    ess_curv = {int(e): blk.curvature(m.expected_shared_ess_p[e, P]) for e in m.shared_energy_storages
                if not m.shared_es_pnet[e, 0, 0, P].fixed}
    return {'per_dn': per_dn, 'generators_hour4': gens, 'levers_lower_injection': levers,
            'cheapest_feasible_lever': min(feas, key=lambda r: r['first_order_dJ_per_pu']) if feas else None,
            'ess_al_curvature_hour4': ess_curv,
            'node_balance_p_duals': [_f(m.dual.get(m.node_balance_p[i, 0, 0, P])) for i in sorted({k[0] for k in m.node_balance_p.keys()})],
            'tables': tabs, 'var_family_summary_hour4_active': family_summary(tabs['vars_hour4_active']),
            'row_family_summary_hour4_active': family_summary(tabs['rows_hour4_active_inequality'], key='row'),
            'kkt_check': blk.kkt_check()}


def esso_block(m, pe, differentiate, Modes, identify_variables):
    vs = list(identify_variables(m.objective.expr, include_fixed=False))
    gb = dict(zip([id(v) for v in vs], differentiate(m.objective.expr, wrt_list=vs, mode=Modes.reverse_numeric)))
    day = []
    for t in m.periods:
        v = m.es_pnet[ESSO_Y, ESSO_D, t]
        day.append({'t': int(t), 'pnet_mw': float(v.value), 'p_req_mw': float(pe.value(m.p_req[ESSO_Y, ESSO_D, t])),
                    'dual_p_req': float(pe.value(m.dual_p_req[ESSO_Y, ESSO_D, t])),
                    'base_objective_gradient_on_pnet': float(gb.get(id(v), 0.0))})
    rows = []
    target = {id(m.es_pnet[ESSO_Y, ESSO_D, P]), id(m.es_qnet[ESSO_Y, ESSO_D, P])}
    units = []
    for fam in ('es_pch_per_unit', 'es_pdch_per_unit'):
        for k, v in getattr(m, fam).items():
            if tuple(k[1:]) == (ESSO_Y, ESSO_D, P):
                target.add(id(v))
                units.append({'var': v.name, 'value': _f(v.value), 'base_objective_gradient': float(gb.get(id(v), 0.0))})
    for c in m.component_data_objects(pe.Constraint, active=True):
        if any(id(w) in target for w in identify_variables(c.body, include_fixed=False)):
            body = float(pe.value(c.body))
            lo = _f(pe.value(c.lower)) if c.lower is not None else None
            up = _f(pe.value(c.upper)) if c.upper is not None else None
            rows.append({'row': c.name, 'expr': str(c.expr)[:200], 'body': body, 'lower': lo, 'upper': up,
                         'dual': _f(m.dual.get(c)), 'active_upper': bool(_at(body, up)), 'active_lower': bool(_at(body, lo))})
    return {'s_rated_mva_by_year': [_f(v.value) for v in m.es_s_rated.values()],
            'e_rated_mwh_by_year': [_f(v.value) for v in m.es_e_rated.values()], 'rho': float(pe.value(m.rho)),
            'base_objective_vars': sorted({v.parent_component().name for v in vs}),
            'day_2030_autumn': day, 'hour4_units': units, 'hour4_rows': rows}


def run():
    t0 = time.time()
    dry = '--dry' in sys.argv
    if os.path.exists(OUT_JSON):
        raise RuntimeError(f'{OUT_JSON} exists; write-once')
    if not dry and not os.path.isdir(OUT_DIR):
        raise RuntimeError(f'{OUT_DIR} missing; the launcher creates it')
    inputs = _verify_inputs()
    _log('inputs sha256-verified against the campaign manifests: ' + ', '.join(v['sha256'][:8] for v in inputs.values()))

    import pyomo.environ as pe
    from pyomo.core.expr.calculus.derivatives import differentiate, Modes
    from pyomo.core.expr.visitor import identify_variables
    tools = (pe, differentiate, Modes, identify_variables)

    detail = json.load(open(os.path.join(REPO, INPUTS['ch_detail'][0])))
    price = {n: detail['interface_reporting_detail'][str(n)][str(YEAR)][DAY]['periods'][str(P)]['price_per_mwh'] for n in NODES}
    day_prices = [detail['interface_reporting_detail']['5'][str(YEAR)][DAY]['periods'][str(t)]['price_per_mwh'] for t in range(24)]

    result = {'stage': 'P5.15 W127 stall constraints (zero solves, model loads)', 'utc': _utc(),
              'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=REPO).stdout.strip(),
              'script_sha256': _sha(THIS), 'instance': INSTANCE, 'inputs': inputs,
              'block': {'year': YEAR, 'day': DAY, 'period_index': P, 'hour': P + 1},
              'definitions': {'active_tol': TOL, 'fd_h': FD_H, 'objective': 'active objective p58_rescaled_admm_objective '
                              '(EUR per representative day; p.u. variables); EUR/MWh = EUR/pu / baseMVA'},
              'hour4_price_eur_mwh': price, 'day_prices_eur_mwh_n5': day_prices}

    for label, key in (('challenger', 'ch_models'), ('x0', 'x0_models')):
        with open(os.path.join(REPO, INPUTS[key][0]), 'rb') as handle:
            payload = pickle.load(handle)
        _log(f'{label}: certified models unpickled')
        tso = Block(payload['tso'][YEAR][DAY], *tools)
        base = tso.dobj(tso.m.pg[0, 0, 0, P]) / price[5] if label == 'challenger' else result['challenger']['base_mva']
        out = {'base_mva': base, 'tso': tso_block(tso, base), 'dso': {}}
        for n in NODES:
            if n not in payload['dso']:
                continue
            blk = Block(payload['dso'][n][YEAR][DAY], *tools)
            out['dso'][str(n)] = dso_block(blk, tso, n, base)
            d = out['dso'][str(n)]
            _log(f"{label} n{n}: gap {d['gap_x_minus_z_mw']:+.4f} MW, AL grad {d['pf_AL_gradient_on_x_eur_per_pu']:.1f}, "
                 f"cheapest lever {d['cheapest_feasible_lever']['var'] if d['cheapest_feasible_lever'] else None} "
                 f"{d['cheapest_feasible_lever']['first_order_dJ_per_pu'] if d['cheapest_feasible_lever'] else None}, "
                 f"rate {d['dead_zone_rate_eur_per_pu_per_cycle']:.3f}/cycle, kkt rel {d['kkt_check']['relative']:.2e}")
        _log(f"{label} TSO: kkt rel {out['tso']['kkt_check']['relative']:.2e}, cheapest lever "
             f"{out['tso']['cheapest_feasible_lever']['var'] if out['tso']['cheapest_feasible_lever'] else None}")
        result[label] = out
        del payload

    with open(os.path.join(REPO, INPUTS['ch_esso'][0]), 'rb') as handle:
        esso = pickle.load(handle)
    result['challenger']['esso'] = {str(n): esso_block(esso[n], *tools) for n in (5, 7)}
    _log('ESSO models read')

    guard_failures = _GUARD.verify(0)
    result['solve_profile_guard'] = {'permitted': [], 'verify_0': guard_failures, 'counts': dict(_GUARD.counts)}
    result['wall_s'] = time.time() - t0
    if dry:
        _log(f'DRY: nothing written; guard verify(0) {guard_failures}; counts {dict(_GUARD.counts)}')
        return 0 if not guard_failures else 1
    with open(OUT_JSON, 'x') as handle:
        GRIO.dump(result, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    _log(f'wrote {OUT_JSON}; guard verify(0) {guard_failures}; counts {dict(_GUARD.counts)}; wall {result["wall_s"]:.1f} s')
    return 0 if not guard_failures else 1


def manifest():
    out = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if os.path.exists(out):
        raise RuntimeError(f'{out} exists; write-once')
    entries = {}
    for name in sorted(os.listdir(OUT_DIR)):
        entries[os.path.relpath(os.path.join(OUT_DIR, name), REPO)] = _sha(os.path.join(OUT_DIR, name))
    entries[os.path.relpath(THIS, REPO)] = _sha(THIS)
    for rel, _campaign, _declared in INPUTS.values():
        entries[rel + ' (hash-recorded input, not committed)'] = _sha(os.path.join(REPO, rel))
    with open(out, 'x') as handle:
        GRIO.dump(entries, handle, indent=1, sort_keys=True)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(manifest() if '--manifest' in sys.argv else run())
