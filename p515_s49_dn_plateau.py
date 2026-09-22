"""
P5.15 Addendum 32 Q4 follow-up (task W31) -- which DN-side row sets the bus-7 price plateau at x = 0, and the
throughput-weighted split of W25's flatness term. ZERO SOLVES, read-only, nothing built.

Armed `SolveProfileGuard(permitted=())` is installed before any Pyomo / production import; `verify(0)` is checked at
the end (exact count: 0 solves, 0 solver launches).

INSTANCE: x = 0 (candidate key 8435c718..., nodes 5/7/9 empty), campaign s48_x0_capture, eval key d2c96b14...; the
persisted certified models (`certified_models.pkl`, sha256 03b62593..., verified before use; not committed) are
unpickled and their terminal IPOPT primal / dual / bound-multiplier values read. Task B also reads the BASELINE
storage record (s47_recert n7_4h_e1, candidate key db77e154...) for the storage throughput, exactly as W25 did.

TASK A -- the DN price chain (formulas; B_dn = DN baseMVA = 100, dt = 1 h; every dual is converted by /B_dn):
  The active DSO objective is p58_rescaled_admm_objective. Its gradient on the physical variables is the
  PHYSICAL per-representative-day cost (verified per block: df/d pg[ref] / B_dn = pi, the market price of the
  interface settlement, and df/d flex_p_down / B_dn = c_flex[p]); so a raw dual / B_dn is EUR/MWh with no
  sigma / block-weight factor (unlike the TSO blocks, where W25 needs sigma / (w_block ...) for the ADMM terms only).
  KKT convention (W28): r_v = df/dv - sum_c lambda_c dc/dv - zL_v - zU_v, lambda = model.dual, zL >= 0, zU <= 0.
  (1) interface (DN reference bus 1, node 0; pg[0] = import from the TN, free):
        y0_p := dual(node_balance_p[0,p]) / B_dn = pi_p + lambda(expected_interface_pf_p_def[p]) / B_dn
      (stationarity of pg[0]); the second term is the ADMM consensus price correction (stationarity of
      expected_interface_pf_p: lambda_E = gradient of the ADMM terms).
  (2) TSO side: LMP_n,p (TSO bus n of the DN) = dual(node_balance_p[bus n]) / B_tso (W28). At ADMM consensus
      y0_p = LMP_n,p; the gap is reported per hour (it is the consensus residual, not a modelling term).
  (3) DN network: y_i,p := dual(node_balance_p[i,p]) / B_dn at load bus i; the DN loss / voltage factor is
      y_i,p / y0_p (reported, not decomposed further).
  (4) Load flexibility of load l at bus i (pc_node = pc + flex_p_up - flex_p_down; daily row
      flex_energy_balance_p[l]: sum_p flex_p_up = sum_p flex_p_down (+ slacks)):
        mu_l := dual(flex_energy_balance_p[l]) / B_dn                (a DAILY constant per load)
        stationarity of flex_p_up[l,p]:   y_i,p = mu_l + z_up          (z = (zL + zU)/B of the variable)
        stationarity of flex_p_down[l,p]: y_i,p = mu_l + c_flex_l,p - z_down
      (signs NOT assumed: both identities are recomputed from the model with numeric derivatives and checked to
      KKT_TOL_EUR; a flex variable is MARGINAL when |z| <= Z_MARG_EUR, then y_i,p = its row price within Z_MARG).
  Plateau test (per c_interface hour of the TSO, classified by W28): the marginal rows' prices -- mu_l + c_flex_l,p
  (P-down) or mu_l (P-up), built from the flex_energy_balance_p dual and the objective coefficient, NOT from y_i --
  are referred to the interface with the DN factor (row price / (y_i/y0)) and compared with y0 and LMP_n,p.
  KKT bracket (every flex row, marginal or not): rows with y_i >= row price (P-up with z_up > 0, P-down with
  z_down < 0) give lower bounds, the others upper bounds; referred to the interface; y0 must lie inside.
  Alternatives tested in the same hours: (ii) DN-local storage (existence read from the DN case files and from the
  model index sets), (iii) DN renewable curtailment (DN generator whose bus price is within Z_MARG of its own
  objective gradient, 0 here -> it would set the price);
  otherwise 'bracketed by flex rows'.
  cost_flex relation (tested, not asserted): mu_l compared with y_i in the hours where flex_p_up[l] is marginal (the
  hours INTO which shifted energy is consumed at the margin) and with the day's lowest y_i.

TASK B -- throughput-weighted flatness split (W25 T4 baseline, r2 and undiscounted weightings):
  W25: flatness_tw = -(1/T) sum_b w_b T_b [S_pi,b - S_LMP0,b], T_b = sum_p (eff_ch pch_b,p + pdch_b,p / eff_dch) dt
  / 2 (baseline ESSO terminal capture, cycle 112, eff 0.97 / 0.96), T = sum_b w_b T_b, S = top-4 minus bottom-4
  (own hours), LMP0 = bus-7 LMP at x = 0. W28 per block: S_pi,b - S_LMP,b = H_b - sum_c D_c,b (hour selection and
  the components of LMP7 - pi: reference_generator_limit (the c_interface / generator-limit term), losses,
  congestion, voltage_bounds, voltage_setpoint, angle, admm_interface_voltage, other, residual). Hence
      flatness_tw = -(1/T) sum_b w_b T_b H_b + sum_c (1/T) sum_b w_b T_b D_c,b.
  Both W25's (b) and W28 use LMP7 at x = 0 (W25 from record a0_c7 x0 via its identity; W28 from the s48 x0_capture
  duals, a bitwise-identical-trajectory reproduction): the split is exact up to the difference of the two LMP series
  (<= 1.7e-6 EUR/MWh per hour, W28) -- the residual is reported.

Output (write-once): data/SRP1/Results/P515S49/dn_plateau/{dn_plateau.json, dn_plateau.md, launch.log (shell),
manifest_sha256.json (`--manifest`, after the run)}.
EXACT COMMAND (repo root; attached; both streams; noclobber):
    mkdir data/SRP1/Results/P515S49/dn_plateau && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_dn_plateau.py \\
        > data/SRP1/Results/P515S49/dn_plateau/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_dn_plateau.py --manifest
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W31 DN plateau (zero solves)').install()

import numpy as np  # noqa: E402
import pyomo.environ as pe  # noqa: E402
from pyomo.core.expr.calculus.derivatives import Modes, differentiate  # noqa: E402
from pyomo.core.expr.visitor import identify_variables  # noqa: E402

STAGE = 'P5.15 Addendum 32 Q4 follow-up W31 -- DN row setting the bus-7 price plateau; throughput-weighted flatness split (zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 32 Q4', 'Planner task W31']
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S49', 'dn_plateau')
PICKLE_REL = os.path.join('data', 'SRP1', 'Results', 'P515S48', 'x0_capture', 'evals', 'd2c96b1480402a3b_x0',
                          'certified_models.pkl')
PICKLE_SHA = '03b62593a23f748c819f18dce52c88c6a9802b8af3c52033ae09a3b88d10afce'
X0_KEY = '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57'
EVAL_KEY = 'd2c96b1480402a3b61aca4abc188e41c6009eb582d8e6ccdd380e51651f996c7'
W28 = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S48', 'x0_capture_analysis', 'x0_capture_analysis.json'),
       'commit': 'dd0a86b3'}
W25 = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'tso_marginal_cost', 'tso_marginal_cost.json'),
       'commit': 'b7aca555'}
W29 = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S49', 'flex_pq_split'), 'commit': 'a6adf0d8'}
BASELINE_REC = 'data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1'
BASELINE_KEY = 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a'
BASELINE_CYCLE = 112
CASE_REL = 'data/SRP1/SRP1.json'
DN_CASE = {5: 'case33_1', 7: 'case33_2', 9: 'case33_3'}
YEARS = (2025, 2030, 2035)
DAYS = ('Spring', 'Summer', 'Autumn', 'Winter')
DETAIL_YEAR = 2030
NODES = (5, 7, 9)
NP = 24
H = 4
DT_H = 1.0

# Tolerances, stated before the committed run (set after a development peek at DN7 2030 Spring, disclosed):
BOUND_TOL_PU = 1e-6            # PRIMAL status only (informative): within 1e-6 p.u. (0.1 kW) of a bound
Z_MARG_EUR = 0.1               # a flex variable / DN generator is MARGINAL (sets the price at its bus) when its net
                               # bound multiplier |zL + zU| / B_dn <= 0.1 EUR/MWh. Revised after development run 1
                               # (disclosed): a primal-margin rule (1e-6 p.u.) misclassified variables sitting ~1e-6..2e-5
                               # p.u. from a bound with IPOPT barrier multipliers 0.07..0.09 EUR/MWh (tol 1e-5) as
                               # interior, and DN generators ~1e-5 p.u. below availability (multiplier ~86 EUR/MWh) as
                               # curtailed. The multiplier rule is the economic definition: y_i = row price + z exactly.
KKT_TOL_EUR = 1e-3             # |stationarity residual| / B_dn of a flex variable / pg[0] / expected_interface_pf_p
ROW_MATCH_TOL_EUR = 1e-3       # |y_i,p - (row price +/- z)| for every flex variable (KKT identity incl. multiplier)
CONSENSUS_TOL_EUR = 0.05       # |LMP_n,p (TSO) - y0_p (DN)|: the ADMM consensus residual (peek: max ~0.01)
PRICE_TOL = 1e-9               # df/d pg[0] / B_dn vs W25's market price (same data twice)
W25_REPRO_TOL = 1e-9           # Task B with W25's own series vs W25's committed term
SPLIT_RESID_TOL = 1e-4         # Task B: W28-based split vs W25's term (LMP series differ by <= 1.7e-6)
OBJ_CONV = ('Objective convention: prices / duals in EUR/MWh (DN: raw dual / B_dn, the active objective being the '
            'physical per-representative-day cost on these variables; TSO: dual / baseMVA, W28); Task B in W25 T4 '
            'units: EUR per MWh of full cell cycle, Q(x) = gross_operational_cost settlement-EXCLUDED, block weights '
            'w_r2 = num_years * num_days / 1.02^(y - 2025) or w_undisc = num_years * num_days.')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W31] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True).stdout.strip()


def _committed_unmodified(rel, commit):
    return (_git(['rev-parse', f'{commit}:{rel}']) == _git(['hash-object', rel])
            and not _git(['status', '--porcelain', '--', rel]))


def _fam(c):
    return c.parent_component().name


def _topbottom(series):
    order = sorted(range(len(series)), key=lambda k: (-series[k], k))
    order_b = sorted(range(len(series)), key=lambda k: (series[k], k))
    return sorted(order[:H]), sorted(order_b[:H])


def _spread(series):
    t, b = _topbottom(series)
    return sum(series[k] for k in t) / H - sum(series[k] for k in b) / H


# ======================================================================================================================
#  inputs + capture-path checklist (before any analysis)
# ======================================================================================================================
def resolve_inputs():
    checks, info = {}, {}
    ppath = os.path.join(REPO, PICKLE_REL)
    got = _sha(ppath) if os.path.isfile(ppath) else None
    checks['pickle_sha256_matches'] = got == PICKLE_SHA
    rec = json.load(open(os.path.join(REPO, os.path.dirname(PICKLE_REL), 'evaluation_record.json')))
    checks['pickle_eval_is_x0'] = rec.get('candidate_key') == X0_KEY
    checks['w28_json_committed_unmodified'] = _committed_unmodified(W28['path'], W28['commit'])
    checks['w25_json_committed_unmodified'] = _committed_unmodified(W25['path'], W25['commit'])
    brec = json.load(open(os.path.join(REPO, BASELINE_REC, 'evaluation_record.json')))
    checks['baseline_record_candidate_key'] = brec.get('candidate_key') == BASELINE_KEY
    checks['baseline_record_cycles_112'] = int(brec.get('cycles_run', -1)) == BASELINE_CYCLE
    cap = os.path.join(REPO, BASELINE_REC, 'esso_capture', 's39_D', f'node7_cycle{BASELINE_CYCLE:03d}.jsonl')
    checks['baseline_esso_capture_present'] = os.path.isfile(cap)
    info.update({'pickle': {'path': PICKLE_REL, 'sha256': got, 'committed': False},
                 'eval_record_candidate_key': rec.get('candidate_key'), 'eval_key_expected': EVAL_KEY,
                 'w28_json': {'path': W28['path'], 'commit': W28['commit'], 'sha256': _sha(W28['path'])},
                 'w25_json': {'path': W25['path'], 'commit': W25['commit'], 'sha256': _sha(W25['path'])},
                 'baseline_record': {'dir': BASELINE_REC, 'candidate_key': brec.get('candidate_key'),
                                     'cycles_run': brec.get('cycles_run'),
                                     'esso_capture': os.path.relpath(cap, REPO),
                                     'esso_capture_sha256': _sha(cap) if os.path.isfile(cap) else None,
                                     'committed': False}})
    return info, checks


def capture_checklist(P, w28, w25):
    items = {}
    for n in NODES:
        for y in YEARS:
            for d in DAYS:
                m = (((P.get('dso') or {}).get(n) or {}).get(y) or {}).get(d)
                k = f'dso{n}_{y}_{d}'
                items[f'{k}_present'] = m is not None
                if m is None:
                    continue
                items[f'{k}_active_objective_p58'] = [o.name for o in m.component_data_objects(pe.Objective, active=True)] == ['p58_rescaled_admm_objective']
                for a in ('flex_energy_balance_p', 'flex_p_up', 'flex_p_down', 'node_balance_p', 'pc_node', 'pg',
                          'expected_interface_pf_p_def', 'expected_interface_pf_p', 'ipopt_zL_out', 'ipopt_zU_out',
                          'dual', 'interface_settlement', 'es_soc', 'shared_es_e_rated_fixed'):
                    items[f'{k}_has_{a}'] = hasattr(m, a)
                items[f'{k}_flex_balance_duals'] = all(c in m.dual for c in m.flex_energy_balance_p.values())
                items[f'{k}_node_balance_duals'] = all(c in m.dual for c in m.node_balance_p.values())
            items[f'tso_{y}'] = all(((P.get('tso') or {}).get(y) or {}).get(d) is not None for d in DAYS)
    for y in YEARS:
        for d in DAYS:
            b = (w28.get('blocks') or {}).get(f'{y}_{d}') or {}
            items[f'w28_{y}_{d}_per_hour_price_setter'] = len(b.get('per_hour') or []) == NP and all(
                'price_setter' in h and 'lmp_by_bus' in h for h in b.get('per_hour') or [])
            items[f'w28_{y}_{d}_spread_components'] = 'component_spreads_D' in (b.get('spread') or {})
    items['w25_T4_baseline_terms'] = all(k in w25['T4']['baseline'] for k in ('r2', 'undiscounted', 'eff_ch', 'eff_dch'))
    items['w25_T2_per_day_12'] = len(w25['T2']['per_day']) == 12
    return items


# ======================================================================================================================
#  Task A
# ======================================================================================================================
def dn_block(m, B, lmp_tso, setter, pi_w25):
    obj = next(iter(m.component_data_objects(pe.Objective, active=True)))
    dual, zL, zU = m.dual, m.ipopt_zL_out, m.ipopt_zU_out
    nloads = len({k[0] for k in m.flex_p_up.keys()})
    # load -> node from pc_node expressions
    load_node = {}
    for i in m.nodes:
        for p in (0,):
            for v in identify_variables(m.pc_node[i, 0, 0, p].expr, include_fixed=True):
                if _fam(v) == 'flex_p_up':
                    load_node[v.index()[0]] = i
    ref = 0
    gen_node = {}
    for i in m.nodes:
        for v in identify_variables(m.pg_node[i, 0, 0, 0].expr, include_fixed=True):
            if _fam(v) == 'pg':
                gen_node[v.index()[0]] = i
    # variables of interest and the rows touching them
    gens_all = sorted({k[0] for k in m.pg.keys()} - {0})
    voi = [m.pg[0, 0, 0, p] for p in range(NP)] + [m.expected_interface_pf_p[p] for p in range(NP)]
    voi += [m.pg[gi, 0, 0, p] for gi in gens_all for p in range(NP) if not m.pg[gi, 0, 0, p].fixed]
    for l in range(nloads):
        for p in range(NP):
            voi += [m.flex_p_up[l, 0, 0, p], m.flex_p_down[l, 0, 0, p]]
    ids = {id(v): k for k, v in enumerate(voi)}
    jt = np.zeros(len(voi))
    for c in m.component_data_objects(pe.Constraint, active=True):
        vs = [v for v in identify_variables(c.body, include_fixed=False) if id(v) in ids]
        if not vs:
            continue
        lam = float(dual.get(c, 0.0))
        for v, dv in zip(vs, differentiate(c.body, wrt_list=vs, mode=Modes.reverse_numeric)):
            jt[ids[id(v)]] += lam * float(dv)
    ovs = [v for v in identify_variables(obj.expr, include_fixed=False) if id(v) in ids]
    gf = np.zeros(len(voi))
    for v, dv in zip(ovs, differentiate(obj.expr, wrt_list=ovs, mode=Modes.reverse_numeric)):
        gf[ids[id(v)]] += float(dv)
    z = np.array([float(zL.get(v, 0.0)) + float(zU.get(v, 0.0)) for v in voi])
    resid = (gf - jt - z) / B
    g = lambda v: gf[ids[id(v)]] / B  # noqa: E731
    zz = lambda v: z[ids[id(v)]] / B  # noqa: E731
    rr = lambda v: resid[ids[id(v)]]  # noqa: E731

    y = {(i, p): float(dual[m.node_balance_p[i, 0, 0, p]]) / B for i in m.nodes for p in range(NP)}
    mu = {l: float(dual[m.flex_energy_balance_p[l, 0, 0]]) / B for l in range(nloads)}
    slack_flex = max(abs(pe.value(v)) for v in list(m.slack_flex_p_balance_up.values()) + list(m.slack_flex_p_balance_down.values()))

    def status(v):
        val, lb, ub = pe.value(v), v.lb, v.ub
        if ub is not None and lb is not None and ub - lb <= 2 * BOUND_TOL_PU:
            return 'degenerate'
        if lb is not None and val - lb <= BOUND_TOL_PU:
            return 'lb'
        if ub is not None and ub - val <= BOUND_TOL_PU:
            return 'ub'
        return 'interior'

    # DN generators other than the interface (renewables, curtailment penalty read)
    pen = pe.value(m.penalty_gen_curtailment)
    gens = sorted({k[0] for k in m.pg.keys()} - {0})
    hours = []
    worst_kkt = max(abs(r) for r in resid)
    max_price_dev = 0.0
    max_row_dev = 0.0
    for p in range(NP):
        pi = g(m.pg[0, 0, 0, p])
        max_price_dev = max(max_price_dev, abs(pi - pi_w25[p]))
        lamE = float(dual[m.expected_interface_pf_p_def[p]]) / B
        y0 = y[(ref, p)]
        loads = []
        for l in range(nloads):
            i = load_node[l]
            up, dn = m.flex_p_up[l, 0, 0, p], m.flex_p_down[l, 0, 0, p]
            c = g(dn)
            su, sd = status(up), status(dn)
            row_price = mu[l] + c
            zu_, zd_ = zz(up), zz(dn)
            loads.append({'load': l, 'node': i, 'y_i': y[(i, p)], 'dn_factor_y_i_over_y0': y[(i, p)] / y0 if y0 else None,
                          'mu_l': mu[l], 'c_flex': c, 'flex_up_MW': pe.value(up) * B, 'flex_up_ub_MW': up.ub * B,
                          'flex_down_MW': pe.value(dn) * B, 'flex_down_ub_MW': dn.ub * B,
                          'up_status': su, 'down_status': sd, 'z_up': zu_, 'z_down': zd_,
                          'up_marginal': abs(zu_) <= Z_MARG_EUR, 'down_marginal': abs(zd_) <= Z_MARG_EUR,
                          'kkt_up': rr(up), 'kkt_down': rr(dn),
                          'down_row_price_mu_plus_c': row_price,
                          'down_identity_dev': y[(i, p)] - (row_price - zz(dn)),
                          'up_identity_dev': y[(i, p)] - (mu[l] + zz(up))})
            max_row_dev = max(max_row_dev, abs(loads[-1]['down_identity_dev']), abs(loads[-1]['up_identity_dev']))
        down_int = [L for L in loads if L['down_marginal']]
        up_int = [L for L in loads if L['up_marginal']]
        curtailed = []
        for gi in gens:
            v = m.pg[gi, 0, 0, p]
            if v.fixed:
                continue
            # marginal iff the price at its bus equals its own marginal cost (objective gradient) within Z_MARG: the
            # pg bound multiplier alone is not enough (the capability / power-factor rows also carry the limit)
            yb, own = y[(gen_node[gi], p)], g(v)
            if abs(yb - own) <= Z_MARG_EUR:
                curtailed.append({'gen': gi, 'pg_MW': pe.value(v) * B, 'ub_MW': v.ub * B, 'bus_price': yb,
                                  'own_marginal_cost': own, 'kkt_resid': rr(v)})
        mechs = []
        if down_int:
            mechs.append('flex_p_down_marginal (price = mu_l + c_flex_l,p)')
        if up_int:
            mechs.append('flex_p_up_marginal (price = mu_l)')
        if curtailed:
            mechs.append('DN RES marginal (price = 0)')
        mech = ' + '.join(mechs) if mechs else 'bracketed by flex rows (no marginal row within Z_MARG)'
        # interface-referred row prices of the marginal loads: row price / (y_i / y0)
        referred = [L['down_row_price_mu_plus_c'] / L['dn_factor_y_i_over_y0'] for L in down_int] + \
                   [L['mu_l'] / L['dn_factor_y_i_over_y0'] for L in up_int]
        # KKT bracket of y0 from every flex row, referred to the interface (lower: rows with y_i >= row price, i.e.
        # up with z_up > 0, down with z_down < 0; upper: the opposite); nearest row price
        lo, hi, near = [], [], []
        for L in loads:
            f = L['dn_factor_y_i_over_y0']
            for r, zv, sgn in ((L['mu_l'], L['z_up'], 1.0), (L['down_row_price_mu_plus_c'], L['z_down'], -1.0)):
                eff = sgn * zv          # y_i - row price
                (lo if eff >= 0 else hi).append(r / f)
                near.append(abs(eff))
        br_lo, br_hi = max(lo, default=None), min(hi, default=None)
        hours.append({
            'hour': p + 1, 'tso_price_setter': setter[p], 'pi': pi, 'lmp_tso_bus': lmp_tso[p], 'y0_dn_ref': y0,
            'consensus_gap_lmp_tso_minus_y0': lmp_tso[p] - y0, 'lambda_E_over_B': lamE,
            'interface_identity_resid': y0 - (pi + lamE),
            'kkt_pg0': rr(m.pg[0, 0, 0, p]), 'kkt_E': rr(m.expected_interface_pf_p[p]),
            'mechanism': mech, 'n_down_interior': len(down_int), 'n_up_interior': len(up_int),
            'bracket_lo_referred': br_lo, 'bracket_hi_referred': br_hi,
            'bracket_width': (br_hi - br_lo) if (br_lo is not None and br_hi is not None) else None,
            'y0_inside_bracket': (br_lo is None or y0 >= br_lo - 1e-9) and (br_hi is None or y0 <= br_hi + 1e-9),
            'nearest_row_price_distance_at_bus': min(near) if near else None,
            'n_down_marginal_mu_plus_c': len(down_int), 'n_up_marginal_mu': len(up_int),
            'n_down_at_ub': sum(1 for L in loads if L['down_status'] == 'ub'),
            'n_up_at_ub': sum(1 for L in loads if L['up_status'] == 'ub'),
            'down_interior_row_price_min': min((L['down_row_price_mu_plus_c'] for L in down_int), default=None),
            'down_interior_row_price_max': max((L['down_row_price_mu_plus_c'] for L in down_int), default=None),
            'up_marginal_mu_min': min((L['mu_l'] for L in up_int), default=None),
            'up_marginal_mu_max': max((L['mu_l'] for L in up_int), default=None),
            'down_interior_referred_to_interface_mean': (sum(referred) / len(referred)) if referred else None,
            'down_interior_referred_to_interface_max_abs_dev_from_y0': max((abs(r - y0) for r in referred), default=None),
            'lmp_tso_minus_referred_mean': (lmp_tso[p] - sum(referred) / len(referred)) if referred else None,
            'c_flex_mean': sum(L['c_flex'] for L in loads) / len(loads),
            'dn_res_curtailed': curtailed, 'loads': loads})
    # cost_flex relation: mu_l vs y_i in the hours where flex_p_up[l] is interior, and vs the day's min y_i
    mu_rel = []
    for l in range(nloads):
        i = load_node[l]
        up_hours = [h['hour'] for h in hours if h['loads'][l]['up_marginal']]
        dn_hours = [h['hour'] for h in hours if h['loads'][l]['down_marginal']]
        yi = [y[(i, p)] for p in range(NP)]
        mu_rel.append({'load': l, 'node': i, 'mu_l': mu[l],
                       'up_interior_hours': up_hours,
                       'y_i_in_up_interior_hours': [yi[h - 1] for h in up_hours],
                       'up_ub_hours': [h['hour'] for h in hours if h['loads'][l]['up_status'] == 'ub'],
                       'down_interior_hours': dn_hours,
                       'down_ub_hours': [h['hour'] for h in hours if h['loads'][l]['down_status'] == 'ub'],
                       'day_min_y_i': min(yi), 'day_max_y_i': max(yi),
                       'mu_minus_day_min_y_i': mu[l] - min(yi),
                       'flex_up_MWh_day': sum(h['loads'][l]['flex_up_MW'] for h in hours),
                       'flex_down_MWh_day': sum(h['loads'][l]['flex_down_MW'] for h in hours)})
    local_storage = {'es_soc_index_size': len(m.es_soc), 'energy_storages_set_size': len(list(m.energy_storages))
                     if hasattr(m, 'energy_storages') else None,
                     'shared_es_e_rated_fixed_MWh': [pe.value(v) * B for v in m.shared_es_e_rated_fixed.values()],
                     'shared_es_s_rated_fixed_MVA': [pe.value(v) * B for v in m.shared_es_s_rated_fixed.values()],
                     'shared_es_soc_max_abs': max(abs(pe.value(v)) for v in m.shared_es_soc.values()),
                     'sess_soc_def_max_abs_dual_eur': max(abs(float(dual.get(c, 0.0))) for c in m.sess_soc_def.values()) / B}
    return {'n_loads': nloads, 'load_node': load_node, 'penalty_gen_curtailment': pen, 'mu': mu,
            'max_abs_slack_flex_balance_pu': slack_flex,
            'max_abs_kkt_resid_eur': float(worst_kkt), 'max_price_dev_vs_w25': max_price_dev,
            'max_down_interior_row_dev_eur': max_row_dev, 'hours': hours, 'mu_relation': mu_rel,
            'local_storage': local_storage}


def dn_case_storage():
    out = {}
    for n, case in DN_CASE.items():
        for y in YEARS:
            d = json.load(open(os.path.join(REPO, 'data', 'SRP1', case, f'{case}_{y}.json')))
            out[f'{case}_{y}'] = {'has_energy_storages_key': 'energy_storages' in d,
                                  'n_energy_storages': len(d.get('energy_storages') or []),
                                  'baseMVA': d.get('baseMVA'), 'n_loads': len(d.get('loads') or [])}
    return out


# ======================================================================================================================
#  Task B
# ======================================================================================================================
def task_b(w25, w28):
    b25 = w25['T4']['baseline']
    eff_ch, eff_dch = b25['eff_ch'], b25['eff_dch']
    cap = os.path.join(REPO, BASELINE_REC, 'esso_capture', 's39_D', f'node7_cycle{BASELINE_CYCLE:03d}.jsonl')
    case = json.load(open(os.path.join(REPO, CASE_REL)))
    years, days = list(case['Years'].keys()), list(case['Days'].keys())
    rows = {}
    for line in open(cap):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (years[int(r['y'])], days[int(r['d'])], int(r['p']))
        if key in rows:
            raise RuntimeError(f'duplicate capture row {key}')
        rows[key] = r
    per_day = {(str(r['year']), r['day']): r for r in w25['T2']['per_day']}
    comps = list(next(iter(w28['blocks'].values()))['spread']['component_spreads_D'].keys())
    out = {'formula': ('flatness_tw = -(1/T) sum_b w_b T_b (S_pi,b - S_LMP,b) = -(1/T) sum_b w_b T_b H_b + sum_c '
                       '(1/T) sum_b w_b T_b D_c,b; T_b = sum_p (eff_ch pch + pdch/eff_dch) dt / 2; T = sum_b w_b T_b'),
           'eff_ch': eff_ch, 'eff_dch': eff_dch, 'capture_rows': len(rows), 'per_block': []}
    Tb = {}
    for y in years:
        for d in days:
            Tb[(y, d)] = sum(eff_ch * rows[(y, d, p)]['pch'] + rows[(y, d, p)]['pdch'] / eff_dch
                             for p in range(NP)) * DT_H / 2.0
    for y in years:
        for d in days:
            bw = w28['blocks'][f'{y}_{d}']
            pdw = per_day[(y, d)]
            s = bw['spread']
            ci_top = sum(1 for h in s['lmp7_top4_hours'] if bw['per_hour'][h - 1]['price_setter']['kind'] == 'c_interface')
            ci_bot = sum(1 for h in s['lmp7_bottom4_hours'] if bw['per_hour'][h - 1]['price_setter']['kind'] == 'c_interface')
            out['per_block'].append({
                'block': f'{y}_{d}', 'T_b_mwh_cycle': Tb[(y, d)], 'w_r2': pdw['w_r2'], 'w_undisc': pdw['w_undisc'],
                'w25_market_spread': _spread(pdw['price_series']), 'w25_lmp0_spread': _spread(pdw['lmp_series']),
                'w28_flatness': s['flatness_market_minus_lmp7'], 'w28_H': s['hour_selection_H'],
                'w28_D': s['component_spreads_D'], 'lmp7_top4_hours': s['lmp7_top4_hours'],
                'lmp7_bottom4_hours': s['lmp7_bottom4_hours'], 'c_interface_hours_in_top4': ci_top,
                'c_interface_hours_in_bottom4': ci_bot,
                'w25_vs_w28_lmp_max_abs': max(abs(a - b) for a, b in zip(pdw['lmp_series'], bw['lmp7'])),
                'w25_vs_w28_same_top_bottom_hours': _topbottom(pdw['lmp_series']) == _topbottom(bw['lmp7'])})
    for wk in ('r2', 'undiscounted'):
        wkey = 'w_r2' if wk == 'r2' else 'w_undisc'
        T = sum(r[wkey] * r['T_b_mwh_cycle'] for r in out['per_block'])
        w25_term = next(t['eur_per_mwh_cycle'] for t in b25[wk]['terms'] if t['term'].startswith('(b) marginal-cost flatness'))
        repro_w25 = -sum(r[wkey] * r['T_b_mwh_cycle'] * (r['w25_market_spread'] - r['w25_lmp0_spread'])
                         for r in out['per_block']) / T
        terms = {'hour_selection_H': -sum(r[wkey] * r['T_b_mwh_cycle'] * r['w28_H'] for r in out['per_block']) / T}
        for c in comps:
            terms[f'D_{c}'] = sum(r[wkey] * r['T_b_mwh_cycle'] * r['w28_D'][c] for r in out['per_block']) / T
        total = sum(terms.values())
        direct = -sum(r[wkey] * r['T_b_mwh_cycle'] * r['w28_flatness'] for r in out['per_block']) / T
        dw = {'weights': wkey, 'T_mwh_cycle': T, 'T_w25_committed': b25[wk]['T_cell_throughput_full_cycle_mwh'],
              'w25_committed_flatness_term': w25_term, 'reproduced_with_w25_series': repro_w25,
              'reproduced_with_w25_series_minus_committed': repro_w25 - w25_term,
              'split_terms_signed_as_contributions_to_captured': terms, 'split_sum': total,
              'split_sum_minus_w25_committed_residual': total - w25_term,
              'w28_direct_flatness_tw': direct, 'split_sum_minus_w28_direct': total - direct,
              'per_year': {}}
        for y in years:
            sub = [r for r in out['per_block'] if r['block'].startswith(y)]
            Ty = sum(r[wkey] * r['T_b_mwh_cycle'] for r in sub)
            dw['per_year'][y] = {'T_share': Ty / T,
                                 'contribution_to_all_years_term': -sum(r[wkey] * r['T_b_mwh_cycle'] * r['w28_flatness'] for r in sub) / T,
                                 'D_reference_generator_limit_contribution': sum(r[wkey] * r['T_b_mwh_cycle'] * r['w28_D']['reference_generator_limit'] for r in sub) / T,
                                 'H_contribution': -sum(r[wkey] * r['T_b_mwh_cycle'] * r['w28_H'] for r in sub) / T}
        out[wk] = dw
    return out


# ======================================================================================================================
#  report
# ======================================================================================================================
def _f(x, nd=2):
    return '' if x is None else f'{x:,.{nd}f}'


def write_md(path, res):
    L = []
    a = L.append
    a(f"# {res['stage']}")
    a('')
    inp = res['inputs']
    a(f"Instance: x = 0, candidate key `{X0_KEY}`, eval key `{EVAL_KEY}`; models `{inp['pickle']['path']}` sha256 "
      f"`{inp['pickle']['sha256']}` (verified; not committed). Task B: BASELINE record `{BASELINE_REC}` (candidate key "
      f"`{BASELINE_KEY}`, terminal cycle {BASELINE_CYCLE}), ESSO capture sha256 `{inp['baseline_record']['esso_capture_sha256']}`. "
      f"W28 JSON {inp['w28_json']['commit']} sha256 `{inp['w28_json']['sha256']}`; W25 JSON {inp['w25_json']['commit']} sha256 "
      f"`{inp['w25_json']['sha256']}`.")
    a('')
    a(f"Solve profile: armed SolveProfileGuard(permitted=()); counts {res['solve_profile_guard']['counts']}; verify(0) "
      f"failures {res['solve_profile_guard']['verify_failures']}. all_checks_pass = {res['all_checks_pass']}; failing: {res['failing_checks']}.")
    a('')
    a(OBJ_CONV)
    a('')
    a('## Method (formulas)')
    a('')
    a(res['method']['task_a'])
    a('')
    a(res['method']['task_b'])
    a('')
    a('## Checks')
    a('')
    a('| check | value |')
    a('|---|---|')
    for k, v in res['checks'].items():
        a(f'| {k} | {v} |')
    a('')
    a('## Task A -- DN-local storage (ii)')
    a('')
    a('DN case files (energy_storages key read by network.py): ' + '; '.join(
        f"{k}: {v['n_energy_storages']}" for k, v in res['task_a']['dn_case_storage'].items()) + '.')
    ls = {f"dso{n}": res['task_a']['blocks'][f'{n}_{DETAIL_YEAR}_Spring']['local_storage'] for n in NODES}
    a('Model index sets (2030 Spring): ' + '; '.join(f"{k}: es_soc size {v['es_soc_index_size']}, shared_es E "
                                                     f"{v['shared_es_e_rated_fixed_MWh']} MWh, S {v['shared_es_s_rated_fixed_MVA']} MVA, "
                                                     f"max |sess_soc_def dual| {v['sess_soc_def_max_abs_dual_eur']:.2e} EUR/MWh"
                                                     for k, v in ls.items()) + '.')
    a('')
    a('## Task A -- mechanism per hour, all blocks (counts of hours)')
    a('')
    a('| DN | year | day | c_interface hours | P-down marginal (mu+c) | P-up marginal (mu) | DN RES marginal (0) | bracketed only | max bracket width, bracketed-only hours | max |consensus gap| c_int | max flex identity dev incl. z | max KKT resid (EUR/MWh) | slack flex bal (pu) |')
    a('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for key, blk in res['task_a']['blocks'].items():
        n, y, d = key.split('_')
        ci = [h for h in blk['hours'] if h['tso_price_setter'] == 'c_interface']
        c_dn = sum(1 for h in ci if 'flex_p_down_marginal' in h['mechanism'])
        c_up = sum(1 for h in ci if 'flex_p_up_marginal' in h['mechanism'])
        c_res = sum(1 for h in ci if 'DN RES' in h['mechanism'])
        brk = [h for h in ci if h['mechanism'].startswith('bracketed')]
        wmax = max((h['bracket_width'] for h in brk if h['bracket_width'] is not None), default=None)
        gap = max((abs(h['consensus_gap_lmp_tso_minus_y0']) for h in ci), default=0.0)
        a(f"| {n} | {y} | {d} | {len(ci)} | {c_dn} | {c_up} | {c_res} | {len(brk)} | {_f(wmax, 3)} | {gap:.4f} | "
          f"{blk['max_down_interior_row_dev_eur']:.2e} | {blk['max_abs_kkt_resid_eur']:.2e} | "
          f"{blk['max_abs_slack_flex_balance_pu']:.1e} |")
    a('')
    for n in NODES:
        for d in DAYS:
            blk = res['task_a']['blocks'][f'{n}_{DETAIL_YEAR}_{d}']
            a(f'### DN {n} ({DN_CASE[n]}) {DETAIL_YEAR} {d}: hour by hour')
            a('')
            a('| h | TSO setter | pi | LMP TSO bus | y0 DN ref | gap | lambda_E/B | mechanism | #P-down marginal | #P-down at ub (primal) | #P-up marginal | #P-up at ub (primal) | mu+c of marginal P-down (min..max) | mu of marginal P-up (min..max) | marginal rows referred to interface (mean) | LMP - referred | bracket [lo, hi] referred | nearest row distance | c_flex mean |')
            a('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
            for h in blk['hours']:
                a(f"| {h['hour']} | {h['tso_price_setter']} | {_f(h['pi'])} | {_f(h['lmp_tso_bus'], 3)} | {_f(h['y0_dn_ref'], 3)} | "
                  f"{h['consensus_gap_lmp_tso_minus_y0']:+.4f} | {_f(h['lambda_E_over_B'])} | {h['mechanism'].split(' (')[0]} | "
                  f"{h['n_down_interior']} | {h['n_down_at_ub']} | {h['n_up_interior']} | {h['n_up_at_ub']} | "
                  f"{_f(h['down_interior_row_price_min'])}..{_f(h['down_interior_row_price_max'])} | "
                  f"{_f(h['up_marginal_mu_min'])}..{_f(h['up_marginal_mu_max'])} | "
                  f"{_f(h['down_interior_referred_to_interface_mean'], 3)} | "
                  f"{'' if h['lmp_tso_minus_referred_mean'] is None else format(h['lmp_tso_minus_referred_mean'], '+.4f')} | "
                  f"[{_f(h['bracket_lo_referred'], 3)}, {_f(h['bracket_hi_referred'], 3)}] | {h['nearest_row_price_distance_at_bus']:.4f} | {_f(h['c_flex_mean'])} |")
            a('')
            mus = [r['mu_l'] for r in blk['mu_relation']]
            a(f"mu_l (daily shadow value of shifted energy, EUR/MWh): min {min(mus):.2f}, max {max(mus):.2f}, mean {sum(mus)/len(mus):.2f}. "
              f"cost_flex relation: " + res['task_a']['mu_relation_summary'][f'{n}_{DETAIL_YEAR}_{d}']['text'])
            a('')
    a('## Task A -- plateau vs cost_flex (all blocks, c_interface hours)')
    a('')
    a(res['task_a']['cost_flex_test']['text'])
    a('')
    a('## Task B -- throughput-weighted flatness split (W25 T4 baseline)')
    a('')
    tb = res['task_b']
    a(tb['formula'])
    a('')
    a('| block | T_b (MWh-cycle) | w_r2 | market spread | LMP7 spread | flatness | H | D_ref_gen_limit (c_interface) | D_losses | D_voltage | D_congestion | c_int hours top4/bottom4 | W25 vs W28 LMP max abs | same hours |')
    a('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in tb['per_block']:
        D = r['w28_D']
        a(f"| {r['block']} | {r['T_b_mwh_cycle']:.3f} | {r['w_r2']:.2f} | {r['w25_market_spread']:.3f} | {r['w25_lmp0_spread']:.3f} | "
          f"{r['w28_flatness']:.3f} | {r['w28_H']:.3f} | {D['reference_generator_limit']:.3f} | {D['losses']:.3f} | "
          f"{D['voltage_bounds']:.3f} | {D['congestion']:.3f} | {r['c_interface_hours_in_top4']}/{r['c_interface_hours_in_bottom4']} | "
          f"{r['w25_vs_w28_lmp_max_abs']:.1e} | {r['w25_vs_w28_same_top_bottom_hours']} |")
    a('')
    for wk in ('r2', 'undiscounted'):
        d = tb[wk]
        a(f"**Weighting {d['weights']}**: T = {d['T_mwh_cycle']:.4f} (W25 {d['T_w25_committed']:.4f}); W25 committed (b) = "
          f"{d['w25_committed_flatness_term']:.6f}; reproduced with W25's own series {d['reproduced_with_w25_series']:.6f} "
          f"(diff {d['reproduced_with_w25_series_minus_committed']:.1e}).")
        a('')
        a('| term (contribution to captured, EUR/MWh-cycle) | value |')
        a('|---|---|')
        for k, v in d['split_terms_signed_as_contributions_to_captured'].items():
            a(f'| {k} | {v:+.6f} |')
        a(f"| **sum** | **{d['split_sum']:+.6f}** |")
        a(f"| residual (sum - W25 committed) | {d['split_sum_minus_w25_committed_residual']:+.2e} |")
        a('')
        a('Per year (contribution to the all-years term): ' + '; '.join(
            f"{y}: T share {v['T_share']:.3f}, flatness {v['contribution_to_all_years_term']:+.3f} (H {v['H_contribution']:+.3f}, "
            f"D_ref_gen_limit {v['D_reference_generator_limit_contribution']:+.3f})" for y, v in d['per_year'].items()))
        a('')
    a(tb['relationship'])
    a('')
    with open(path, 'x') as h:
        h.write('\n'.join(L) + '\n')


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    t0 = time.time()
    out_dir = os.environ.get('W31_SCRATCH_OUT') or os.path.join(REPO, OUT_REL)   # scratch: development only
    if not os.path.isdir(out_dir):
        raise SystemExit(f'output directory {OUT_REL} must be created by the launch command')
    for f in ('dn_plateau.json', 'dn_plateau.md'):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'{f} exists (write-once)')
    inputs, checks = resolve_inputs()
    if not all(checks.values()):
        raise SystemExit(f'input checks failed: {[k for k, v in checks.items() if not v]}')
    w28 = json.load(open(W28['path']))
    w25 = json.load(open(W25['path']))
    _log(f'loading {PICKLE_REL}')
    with open(os.path.join(REPO, PICKLE_REL), 'rb') as h:
        P = pickle.load(h)
    cl = capture_checklist(P, w28, w25)
    bad = [k for k, v in cl.items() if not v]
    _log(f'capture-path checklist: {len(cl)} items, {len(bad)} False: {bad}')
    if bad:
        raise SystemExit('capture-path checklist failed')
    case9 = json.load(open(os.path.join(REPO, 'data', 'SRP1', 'case9', f'case9_{DETAIL_YEAR}.json')))
    B_tso = float(case9['baseMVA'])
    per_day25 = {(str(r['year']), r['day']): r for r in w25['T2']['per_day']}

    blocks = {}
    for n in NODES:
        B_dn = float(json.load(open(os.path.join(REPO, 'data', 'SRP1', DN_CASE[n], f'{DN_CASE[n]}_2030.json')))['baseMVA'])
        for y in YEARS:
            for d in DAYS:
                t = time.time()
                bw = w28['blocks'][f'{y}_{d}']
                setter = [h['price_setter']['kind'] for h in bw['per_hour']]
                lmp_w28 = [h['lmp_by_bus'][str(n)] for h in bw['per_hour']]
                mt = P['tso'][y][d]
                node_of_bus = {int(nd['bus_i']): k for k, nd in enumerate(case9['nodes'])}
                lmp_t = [float(mt.dual[mt.node_balance_p[node_of_bus[n], 0, 0, p]]) / B_tso for p in range(NP)]
                blk = dn_block(P['dso'][n][y][d], B_dn, lmp_t, setter, per_day25[(str(y), d)]['price_series'])
                blk['B_dn'] = B_dn
                blk['lmp_tso_vs_w28_max_abs'] = max(abs(a - b) for a, b in zip(lmp_t, lmp_w28))
                blocks[f'{n}_{y}_{d}'] = blk
                ci = [h for h in blk['hours'] if h['tso_price_setter'] == 'c_interface']
                mechs = defaultdict(int)
                for h in ci:
                    mechs[h['mechanism'].split(' ')[0]] += 1
                _log(f"DN{n} {y} {d}: c_interface {len(ci)} {dict(mechs)} kkt {blk['max_abs_kkt_resid_eur']:.2e} "
                     f"rowdev {blk['max_down_interior_row_dev_eur']:.2e} gap "
                     f"{max((abs(h['consensus_gap_lmp_tso_minus_y0']) for h in ci), default=0):.4f} "
                     f"({time.time() - t:.1f}s)")

    # checks
    chk = dict(checks)
    chk['capture_path_checklist_all_true'] = not bad
    allh = [(k, h) for k, b in blocks.items() for h in b['hours']]
    chk['dn_objective_gradient_pg0_equals_market_price'] = max(b['max_price_dev_vs_w25'] for b in blocks.values()) <= PRICE_TOL * 1e3
    chk['tso_lmp_matches_w28'] = max(b['lmp_tso_vs_w28_max_abs'] for b in blocks.values()) <= 1e-9
    chk['kkt_resid_below_tol_all_dn_blocks'] = max(b['max_abs_kkt_resid_eur'] for b in blocks.values()) <= KKT_TOL_EUR
    chk['interface_identity_y0_eq_pi_plus_lambdaE'] = max(abs(h['interface_identity_resid']) for _k, h in allh) <= KKT_TOL_EUR
    chk['flex_identities_y_eq_row_price_plus_multiplier_all_slots'] = max(b['max_down_interior_row_dev_eur'] for b in blocks.values()) <= ROW_MATCH_TOL_EUR
    chk['y0_inside_flex_kkt_bracket_all_hours'] = all(h['y0_inside_bracket'] for _k, h in allh)
    ci_all = [(k, h) for k, h in allh if h['tso_price_setter'] == 'c_interface']
    chk['consensus_gap_below_tol_in_c_interface_hours'] = max(abs(h['consensus_gap_lmp_tso_minus_y0']) for _k, h in ci_all) <= CONSENSUS_TOL_EUR
    chk['flex_balance_slacks_negligible'] = max(b['max_abs_slack_flex_balance_pu'] for b in blocks.values()) <= 1e-6
    chk['no_dn_local_storage_in_models'] = all(b['local_storage']['es_soc_index_size'] == 0 for b in blocks.values())
    chk['shared_storage_zero_rated_at_x0'] = all(max(b['local_storage']['shared_es_e_rated_fixed_MWh'] + b['local_storage']['shared_es_s_rated_fixed_MVA']) == 0.0 for b in blocks.values())

    # mechanism statistics in c_interface hours
    mech_count = defaultdict(int)
    for _k, h in ci_all:
        mech_count[h['mechanism']] += 1
    node7_2030 = [(k, h) for k, h in ci_all if k.startswith(f'7_{DETAIL_YEAR}')]
    mech7 = defaultdict(int)
    for _k, h in node7_2030:
        mech7[h['mechanism']] += 1

    # cost_flex relation
    mu_summary = {}
    diffs_up, diffs_min = [], []
    for k, b in blocks.items():
        d_up = [yv - r['mu_l'] for r in b['mu_relation'] for yv in r['y_i_in_up_interior_hours']]
        d_min = [r['mu_minus_day_min_y_i'] for r in b['mu_relation']]
        diffs_up += d_up
        diffs_min += d_min
        uph = sorted({h for r in b['mu_relation'] for h in r['up_interior_hours']})
        dnh = sorted({h for r in b['mu_relation'] for h in r['down_interior_hours']})
        mu_summary[k] = {'n_up_interior_slots': len(d_up),
                         'max_abs_y_minus_mu_in_up_interior_slots': max((abs(x) for x in d_up), default=None),
                         'mu_minus_day_min_y_i_range': [min(d_min), max(d_min)],
                         'hours_with_any_up_interior': uph, 'hours_with_any_down_interior': dnh,
                         'text': (f"flex_p_up marginal in {len(d_up)} load-hours (hours {uph}); there y_i - mu_l max abs "
                                  f"{max((abs(x) for x in d_up), default=float('nan')):.2e} EUR/MWh; mu_l - day-min y_i in "
                                  f"[{min(d_min):.2f}, {max(d_min):.2f}]; flex_p_down marginal in hours {dnh}.")}
    # plateau vs cost_flex (tested): over c_interface hours where some P-down row is marginal, compare the TSO bus LMP
    # with (a) the marginal rows referred to the interface, (b) mean mu + mean c_flex of the marginal P-down loads,
    # (c) the day's lowest TSO bus LMP + the same c_flex ("off-peak price + cost_flex")
    pts, pts_up, brk = [], [], []
    for k, h in ci_all:
        blk = blocks[k]
        day_min = min(x['lmp_tso_bus'] for x in blk['hours'])
        dn_m = [L for L in h['loads'] if L['down_marginal']]
        if dn_m:
            cm = sum(L['c_flex'] for L in dn_m) / len(dn_m)
            mm = sum(L['mu_l'] for L in dn_m) / len(dn_m)
            pts.append({'block': k, 'hour': h['hour'], 'lmp': h['lmp_tso_bus'], 'referred': h['down_interior_referred_to_interface_mean'],
                        'mu_plus_c': mm + cm, 'offpeak_plus_c': day_min + cm, 'mu': mm, 'c': cm, 'day_min_lmp': day_min})
        elif h['n_up_marginal_mu']:
            up_m = [L for L in h['loads'] if L['up_marginal']]
            pts_up.append({'block': k, 'hour': h['hour'], 'lmp': h['lmp_tso_bus'], 'referred': h['down_interior_referred_to_interface_mean'],
                           'mu': sum(L['mu_l'] for L in up_m) / len(up_m), 'day_min_lmp': day_min})
        if h['mechanism'].startswith('bracketed'):
            brk.append({'block': k, 'hour': h['hour'], 'lmp': h['lmp_tso_bus'], 'lo': h['bracket_lo_referred'],
                        'hi': h['bracket_hi_referred'], 'width': h['bracket_width'],
                        'nearest': h['nearest_row_price_distance_at_bus']})
    mabs = lambda xs: (max(xs), sum(xs) / len(xs)) if xs else (None, None)  # noqa: E731
    d_ref = mabs([abs(q['lmp'] - q['referred']) for q in pts + pts_up])
    d_mc = mabs([abs(q['lmp'] - q['mu_plus_c']) for q in pts])
    d_off = mabs([abs(q['lmp'] - q['offpeak_plus_c']) for q in pts])
    d_upmu = mabs([abs(q['lmp'] - q['mu']) for q in pts_up])
    w_br = mabs([q['width'] for q in brk if q['width'] is not None])
    near_br = mabs([q['nearest'] for q in brk])
    cost_flex_test = {
        'n_c_interface_hours_all_blocks': len(ci_all), 'mechanism_counts_all_blocks': dict(mech_count),
        'mechanism_counts_node7_2030': dict(mech7),
        'p_down_marginal_hours': pts, 'p_up_marginal_only_hours': pts_up, 'bracketed_only_hours': brk,
        'lmp_minus_referred_marginal_rows_max_mean': d_ref, 'lmp_minus_mu_plus_c_max_mean': d_mc,
        'lmp_minus_offpeak_plus_c_max_mean': d_off, 'lmp_minus_mu_p_up_only_max_mean': d_upmu,
        'bracket_width_bracketed_only_max_mean': w_br, 'nearest_row_distance_bracketed_only_max_mean': near_br,
        'text': (f"{len(ci_all)} c_interface (TSO-classified) DN-hours over the 36 DN blocks (3 DNs x 12; each TSO hour "
                 f"counted once per DN). Mechanism counts: {dict(mech_count)}. Hours with a marginal P-down row: {len(pts)}; "
                 f"with only marginal P-up rows: {len(pts_up)}; bracketed only: {len(brk)}. |LMP_TSO - marginal rows "
                 f"referred to the interface| max/mean {d_ref}. P-down hours: |LMP - (mu + c_flex)| max/mean {d_mc} "
                 f"(no DN factor); |LMP - (day-min LMP + c_flex)| max/mean {d_off} ('off-peak price + cost_flex'). "
                 f"P-up-only hours: |LMP - mu| max/mean {d_upmu}. Bracketed-only hours: bracket width max/mean {w_br}, "
                 f"nearest row distance at the bus max/mean {near_br}.")}

    task_b_res = task_b(w25, w28)
    task_b_res['relationship'] = (
        'Relationship: W25 (b) is defined on LMP7 at x = 0 (LMP0), not on the storage LMP at x (W25 kept LMP(x) only for '
        'the ideal-at-LMP(x) diagnostic, not in (b)); W28 decomposes the SAME quantity per block from the x = 0 '
        'capture models (s48_x0_capture, trajectory bitwise identical to a0_c7 x0). The throughput weights are the '
        'BASELINE storage schedule (W25 T4). The split is therefore exact by construction (identity S_pi - S_LMP = '
        'H - sum_c D_c holds per block to 1e-13), with the only residual being W25-vs-W28 LMP series differences '
        '(<= 1.7e-6 EUR/MWh per hour) -- reported above.')
    for wk in ('r2', 'undiscounted'):
        d = task_b_res[wk]
        chk[f'task_b_{wk}_T_matches_w25'] = abs(d['T_mwh_cycle'] - d['T_w25_committed']) <= 1e-9 * d['T_mwh_cycle']
        chk[f'task_b_{wk}_w25_series_repro'] = abs(d['reproduced_with_w25_series_minus_committed']) <= W25_REPRO_TOL
        chk[f'task_b_{wk}_split_residual_below_tol'] = abs(d['split_sum_minus_w25_committed_residual']) <= SPLIT_RESID_TOL
        chk[f'task_b_{wk}_split_sums_to_direct'] = abs(d['split_sum_minus_w28_direct']) <= 1e-9
    chk['task_b_same_top_bottom_hours_all_blocks'] = all(r['w25_vs_w28_same_top_bottom_hours'] for r in task_b_res['per_block'])

    g_fail = GUARD.verify(0)
    chk['solve_profile_guard_verified_0'] = not g_fail and GUARD.counts == {'permitted_solve': 0, 'permitted_exec': 0,
                                                                           'blocked_solve': 0, 'blocked_exec': 0}
    failing = [k for k, v in chk.items() if not v]
    res = {
        'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': _git(['rev-parse', 'HEAD']), 'script': 'p515_s49_dn_plateau.py',
        'script_sha256': _sha(os.path.join(REPO, 'p515_s49_dn_plateau.py')),
        'script_git_status': _git(['status', '--porcelain', '--', 'p515_s49_dn_plateau.py']),
        'instance': {'candidate_key_x0': X0_KEY, 'eval_key': EVAL_KEY, 'baseline_candidate_key': BASELINE_KEY},
        'inputs': inputs, 'objective_convention': OBJ_CONV,
        'tolerances': {'BOUND_TOL_PU': BOUND_TOL_PU, 'KKT_TOL_EUR': KKT_TOL_EUR, 'ROW_MATCH_TOL_EUR': ROW_MATCH_TOL_EUR,
                       'CONSENSUS_TOL_EUR': CONSENSUS_TOL_EUR, 'W25_REPRO_TOL': W25_REPRO_TOL,
                       'SPLIT_RESID_TOL': SPLIT_RESID_TOL,
                       'note': 'set before the committed run, after a development peek at DN7 2030 Spring (disclosed)'},
        'method': {'task_a': __doc__.split('TASK A')[1].split('TASK B')[0].strip(),
                   'task_b': 'TASK B' + __doc__.split('TASK B')[1].split('Output (write-once)')[0].rstrip()},
        'out_dir': os.path.relpath(out_dir, REPO),
        'capture_path_checklist_asserted_before_analysis': {'n_items': len(cl), 'false': bad},
        'checks': chk, 'all_checks_pass': not failing, 'failing_checks': failing,
        'task_a': {'dn_case_storage': dn_case_storage(), 'cost_flex_test': cost_flex_test,
                   'mu_relation_summary': mu_summary, 'blocks': blocks},
        'task_b': task_b_res,
        'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_failures': g_fail},
        'wall_clock_s': time.time() - t0,
    }
    res = _jsonable(res)
    with open(os.path.join(out_dir, 'dn_plateau.json'), 'x') as h:
        json.dump(res, h, indent=1)
    write_md(os.path.join(out_dir, 'dn_plateau.md'), res)
    _log(f"cost_flex test: {cost_flex_test['text']}")
    for wk in ('r2', 'undiscounted'):
        d = task_b_res[wk]
        _log(f"Task B {wk}: W25 {d['w25_committed_flatness_term']:.6f} split {d['split_sum']:.6f} residual "
             f"{d['split_sum_minus_w25_committed_residual']:.2e} terms "
             f"{ {k: round(v, 4) for k, v in d['split_terms_signed_as_contributions_to_captured'].items()} }")
    _log(f'checks failing: {failing}; guard {GUARD.counts}; wall {time.time() - t0:.1f}s')


def write_manifest(out_dir):
    man = {}
    for fname in sorted(os.listdir(out_dir)):
        if fname == 'manifest_sha256.json':
            continue
        man[os.path.relpath(os.path.join(out_dir, fname), REPO)] = _sha(os.path.join(out_dir, fname))
    man[os.path.relpath(os.path.join(REPO, 'p515_s49_dn_plateau.py'), REPO)] = _sha(os.path.join(REPO, 'p515_s49_dn_plateau.py'))
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        json.dump(man, handle, indent=2)
    return man


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        print(json.dumps(write_manifest(os.path.join(REPO, OUT_REL)), indent=2))
    else:
        main()
