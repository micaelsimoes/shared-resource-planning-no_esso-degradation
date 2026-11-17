"""P5.15 S55 -- centralised single-day check, STEP 2 (author GO 2026-11-16). Exactly 8 IPOPT solves.

Frozen spec: data/SRP1/Results/P515S55/central_check/frozen_central_check_spec_v1_<sha8>.json (passed with --spec; its
sha256 must start with the <sha8> in its name, and it must be committed clean, as must this script).

For each 2025 representative day (Spring, Summer, Autumn, Winter), single scenario, plan without storage (x = 0):
  merged model  production centralized_coordination.combine_networks(TN, DNs) day network, Network.build_model(params)
                with params = deepcopy(merged.params); IPOPT options per spec (tol, compl_inf_tol, linear_solver
                overridden; every other inherited option unchanged); recovery and tier-2 recovery OFF
  loads         every pc, qc Var fixed at its demand load.pd / load.qd (ruling 1; the DSO models' treatment)
  primary       solve from the model's own initial values (Network.run_smopf, from_warm_start=False)
  warm          a second, freshly built merged model whose free primal Vars are set from the coordinated certified models
                of that day (eval d110bd1a, certified_models.pkl, one guarded load, sha256 pinned); no duals; same
                options; Network.run_smopf, from_warm_start=False (IPOPT starts from the set values)
Before the warm solve: the merged objective terms and the maximum constraint violation are evaluated AT the mapped
coordinated point (mapping gate: TN generation + DN flexibility there must reproduce the coordinated day cost).

Reported per day and start: status (termination condition, IPOPT exit text), iterations, wall time; C = TN generation +
DN flexibility (value(total_gen_cost) + value(total_flex_cost)); slack penalties and the full objective beside it; the
difference to the coordinated cost (component_levels_terminal.json of ref:7aa017f0, unweighted) in EUR and percent; the
hour-by-hour interface P, Q and voltage magnitude per DN: merged = power on the DN branches leaving the TN bus
(production _get_branch_power_flow) and sqrt(e^2 + f^2) at the TN bus; coordinated = TSO pc_adn / qc_adn / vmag_adn and
DSO pg_adn / qg_adn / vmag_adn.

GUARDS. SolveProfileGuard(permitted=[(this file, '_solve_one')]) installed before any project import and verified at
exactly 8 solves and 8 executions; pickle allow-list gate: exactly one load, of the pinned certified_models.pkl;
single-instance lock; output directory must not exist; stdout and stderr captured by the caller (attached run).

Usage: NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s55_central_check.py --spec SPEC --out-dir DIR
       [--trial]   (--trial: skips the committed-clean checks and the solves -- builds, maps and evaluates only)
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, REPO)
SCRIPT_REL = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=((SCRIPT_REL, '_solve_one'),), label='P5.15 S55 centralised check').install()

import pickle  # noqa: E402

PKL_REL = ('data/SRP1/Results/P515S53/w101_srp1_continuation/campaign_s53_w101_srp1_cont_x0/evals/'
           'd110bd1a5977df1e_x0/certified_models.pkl')
PKL_SHA = '99ab1070b0e61cc7818975ce898069d2060d2668bae67c166b58913e9923c33a'
_ORIG_LOAD = pickle.load
PK = {'delegated_loads': 0, 'refused': 0}


def _gated_load(file, *a, **k):
    path = os.path.relpath(os.path.abspath(getattr(file, 'name', '')), REPO)
    if path != PKL_REL or PK['delegated_loads'] >= 1 or file.tell() != 0:
        PK['refused'] += 1
        raise RuntimeError(f'pickle.load refused: {path} (only one load of {PKL_REL} is permitted)')
    data = file.read()
    if hashlib.sha256(data).hexdigest() != PKL_SHA:
        PK['refused'] += 1
        raise RuntimeError('pickle.load refused: sha256 of the bytes read differs from the pin')
    PK['delegated_loads'] += 1
    import io
    return _ORIG_LOAD(io.BytesIO(data), *a, **k)


def _refuse_loads(*_a, **_k):
    PK['refused'] += 1
    raise RuntimeError('pickle.loads refused')


pickle.load, pickle.loads = _gated_load, _refuse_loads

import pyomo.environ as pe  # noqa: E402

from centralized_coordination import combine_networks  # noqa: E402
from network import _get_branch_power_flow  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

YEAR = 2025
DAYS = ('Spring', 'Summer', 'Autumn', 'Winter')
CL_REL = ('data/SRP1/Results/P515S53/w101_srp1_continuation/campaign_s53_w101_srp1_cont_x0/evals/'
          'd110bd1a5977df1e_x0/component_levels_terminal.json')
NODE_VARS = ('e', 'f', 'slack_v_sqr_down', 'slack_v_sqr_up', 'vmag_sqr')
BRANCH_VARS = ('voltage_product_real', 'voltage_product_imag', 'r', 'r_sqr', 'pij', 'qij', 'pji', 'qji')
GEN_VARS = ('pg', 'qg')
LOAD_VARS = ('flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down', 'slack_flex_p_balance_up',
             'slack_flex_p_balance_down', 'slack_flex_q_balance_up', 'slack_flex_q_balance_down')
FIXED_AT_DEMAND = ('pc', 'qc')


def _sha(path):
    with open(path, 'rb') as h:
        return hashlib.sha256(h.read()).hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True).stdout


def committed_clean(rel):
    return bool(_git('ls-files', rel).strip()) and not _git('status', '--porcelain', '--', rel).strip()


def _by_str(d, key):
    for k in (key, str(key)):
        if k in d:
            return d[k]
    try:
        return d[int(key)]
    except (KeyError, ValueError, TypeError):
        raise KeyError(key)


# ------------------------------------------------------------------------------------------------ merged-model helpers
def configure(params, spec):
    p = deepcopy(params)
    p.solver_params.options = dict(p.solver_params.options or {})
    p.solver_params.options.update(spec['solver']['option_overrides'])
    p.solver_params.recovery_enabled = False
    p.solver_params.recovery_tier2_enabled = False
    return p


def fix_loads(m, md):
    n = 0
    for c in m.loads:
        load = md.loads[c]
        for s_m in m.scenarios_market:
            for s_o in m.scenarios_operation:
                for p in m.periods:
                    m.pc[c, s_m, s_o, p].fix(load.pd[s_o][p])
                    m.qc[c, s_m, s_o, p].fix(load.qd[s_o][p])
                    n += 2
    return n


def _find(seq, attr, value):
    for i, x in enumerate(seq):
        if getattr(x, attr) == value:
            return i
    raise KeyError(f'{attr}={value}')


def source_maps(md, tn_day, dn_days):
    """merged element index -> (source key, source element index), by the merge's old_* identifiers."""
    def src(new_id):
        s = str(new_id)
        if s.startswith('TN_'):
            return 'tso', tn_day
        dn = int(s.split('_')[1])
        return dn, dn_days[dn]

    maps = {'node': {}, 'branch': {}, 'gen': {}, 'load': {}}
    for i, n in enumerate(md.nodes):
        key, net = src(n.bus_i)
        maps['node'][i] = (key, net.get_node_idx(n.old_bus_i))
    for i, b in enumerate(md.branches):
        key, net = src(b.branch_id)
        maps['branch'][i] = (key, _find(net.branches, 'branch_id', b.old_branch_id))
    for i, g in enumerate(md.generators):
        key, net = src(g.gen_id)
        maps['gen'][i] = (key, _find(net.generators, 'gen_id', g.old_gen_id))
    for i, ld in enumerate(md.loads):
        key, net = src(ld.load_id)
        maps['load'][i] = (key, _find(net.loads, 'load_id', ld.old_load_id))
    return maps


def interface_angles(tm, tn_day, dn_ids):
    """theta[dn][(s_m, s_o, p)]: the coordinated TSO's voltage angle at the TN bus of DN dn. Each DSO model measures
    angles from its own reference bus, so the DN's e, f are rotated by this angle when mapped into the merged model
    (where the former DN reference bus IS the TN bus). Branch voltage products inside a DN are rotation invariant."""
    th = {}
    for dn in dn_ids:
        i = tn_day.get_node_idx(dn)
        th[dn] = {(s_m, s_o, p): math.atan2(tm.f[i, s_m, s_o, p].value, tm.e[i, s_m, s_o, p].value)
                  for s_m in tm.scenarios_market for s_o in tm.scenarios_operation for p in tm.periods}
    return th


def warm_start(m, maps, src_models, theta):
    kinds = [(NODE_VARS, 'node'), (BRANCH_VARS, 'branch'), (GEN_VARS, 'gen'), (LOAD_VARS, 'load')]
    report = {}
    handled = set(FIXED_AT_DEMAND)
    for names, kind in kinds:
        for name in names:
            handled.add(name)
            v = getattr(m, name, None)
            if v is None:
                continue
            r = {'set': 0, 'skipped_fixed': 0, 'unmapped': 0, 'outside_bounds': 0, 'max_outside_amount': 0.0,
                 'rotated': 0}
            for idx in v:
                var = v[idx]
                if var.fixed:
                    r['skipped_fixed'] += 1
                    continue
                idx_t = idx if isinstance(idx, tuple) else (idx,)
                key, se = maps[kind][idx_t[0]]
                sv = getattr(src_models[key], name, None)
                sidx = (se,) + idx_t[1:]
                if sv is None or sidx not in sv or sv[sidx].value is None:
                    r['unmapped'] += 1
                    continue
                val = sv[sidx].value
                if name in ('e', 'f') and key != 'tso':
                    sm = src_models[key]
                    e0, f0 = sm.e[sidx].value, sm.f[sidx].value
                    t = theta[key][idx_t[1:]]
                    val = (e0 * math.cos(t) - f0 * math.sin(t)) if name == 'e' else (e0 * math.sin(t) + f0 * math.cos(t))
                    r['rotated'] += 1
                out_amt = max((var.lb - val) if var.lb is not None else 0.0, (val - var.ub) if var.ub is not None else 0.0)
                if out_amt > 0.0:
                    r['outside_bounds'] += 1
                    r['max_outside_amount'] = max(r['max_outside_amount'], out_amt)
                var.set_value(val, skip_validation=True)
                SET_IDS.add(id(var))
                r['set'] += 1
            report[name] = r
    others = sorted(v.name for v in m.component_objects(pe.Var, active=True) if v.name not in handled and len(v))
    return report, others


PARAM_ALIGN_EXCLUDED = ('interface_settlement_weight',)   # the merged model has no interface: settlement is 0


def align_scalar_params(m, src_models):
    """Objective identity (spec 'objective' clause): every scalar mutable Param of the merged model takes the value the
    coordinated certified models carry (the ADMM path sets e.g. penalty_gen_curtailment = 0 and
    penalty_shared_ess_usage = 0, which a plain build_model leaves at their defaults). Refuses when the coordinated
    models disagree among themselves on a Param the merged model has."""
    rep = {'aligned': {}, 'equal': [], 'absent_in_coordinated': [], 'conflict': {}, 'excluded': {}}
    for prm in m.component_objects(pe.Param, active=True):
        if prm.is_indexed() or not prm.mutable:
            continue
        name = prm.name
        mine = pe.value(prm)
        if name in PARAM_ALIGN_EXCLUDED:
            rep['excluded'][name] = mine
            continue
        vals = {}
        for key, sm in src_models.items():
            sp = getattr(sm, name, None)
            if sp is not None and not sp.is_indexed():
                vals[str(key)] = pe.value(sp)
        if not vals:
            rep['absent_in_coordinated'].append(name)
            continue
        uniq = set(vals.values())
        if len(uniq) > 1:
            rep['conflict'][name] = vals
            continue
        target = uniq.pop()
        if target != mine:
            prm.set_value(target)
            rep['aligned'][name] = [mine, target]
        else:
            rep['equal'].append(name)
    return rep


def coverage(m, mapping_report):
    """Every free Var that appears in an active constraint or the objective must have been set by the mapping."""
    from pyomo.core.expr.visitor import identify_variables
    used = {}
    for c in m.component_data_objects(pe.Constraint, active=True):
        for v in identify_variables(c.body, include_fixed=False):
            used[id(v)] = v
    for v in identify_variables(m.objective.expr, include_fixed=False):
        used[id(v)] = v
    missing = {}
    for v in used.values():
        if id(v) not in SET_IDS:
            comp = v.parent_component().name
            missing[comp] = missing.get(comp, 0) + 1
    return {'n_free_vars_used': len(used), 'unset_used_by_component': missing}


SET_IDS = set()


def ref_generator_limits(dms, dn_days):
    """The merge drops each DN reference generator, so its P/Q limits are not in the merged model: were they active in
    the coordinated models? Minimum distance to a bound over the day, MW / MVAr."""
    out = {}
    for dn, dm in dms.items():
        g = dn_days[dn].get_reference_gen_idx()
        base = dn_days[dn].baseMVA
        dist = {}
        for name in ('pg', 'qg'):
            v = getattr(dm, name)
            d = []
            for p in dm.periods:
                x = v[g, 0, 0, p]
                d.append(min((x.value - x.lb) if x.lb is not None else float('inf'),
                             (x.ub - x.value) if x.ub is not None else float('inf')) * base)
            dist[name + '_min_distance_to_bound'] = min(d)
            x0 = v[g, 0, 0, dm.periods.first() if hasattr(dm.periods, 'first') else 0]
            dist[name + '_bounds'] = [x0.lb * base if x0.lb is not None else None,
                                      x0.ub * base if x0.ub is not None else None]
        out[str(dn)] = dist
    return out


def max_violation(m, top=5):
    viol = []
    for c in m.component_data_objects(pe.Constraint, active=True):
        body = pe.value(c.body, exception=False)
        if body is None:
            continue
        lo = pe.value(c.lower) if c.has_lb() else None
        up = pe.value(c.upper) if c.has_ub() else None
        v = max((lo - body) if lo is not None else 0.0, (body - up) if up is not None else 0.0, 0.0)
        if v > 0.0:
            viol.append((v, c.name))
    viol.sort(reverse=True)
    by_comp = {}
    for v, n in viol:
        comp = n.split('[')[0]
        by_comp[comp] = max(by_comp.get(comp, 0.0), v)
    return {'max': viol[0][0] if viol else 0.0, 'top': viol[:top], 'max_by_component': by_comp,
            'n_violated_gt_1e-6': sum(1 for v, _n in viol if v > 1e-6)}


def costs(m):
    out = {k: float(pe.value(getattr(m, k))) for k in ('total_gen_cost', 'total_flex_cost', 'total_load_curt_cost',
                                                        'total_gen_curt_penalty', 'total_slack_penalties',
                                                        'total_ess_utilization_cost_penalty',
                                                        'total_ess_complementarity_penalties')}
    out['C_gen_plus_flex'] = out['total_gen_cost'] + out['total_flex_cost']
    out['objective'] = float(pe.value(m.objective))
    return out


def interface_merged(md, m, params, dn_ids):
    out = {}
    for dn in dn_ids:
        tbus = f'TN_{dn}'
        node_idx = md.get_node_idx(tbus)
        brs = [b for b in md.branches if b.fbus == tbus or b.tbus == tbus]
        brs = [b for b in brs if str(b.branch_id).startswith(f'ADN_{dn}_')]
        P, Q, V = [], [], []
        for p in m.periods:
            pp = qq = 0.0
            for b in brs:
                other = b.tbus if b.fbus == tbus else b.fbus
                pij, qij = _get_branch_power_flow(md, params, b, tbus, other, m, 0, 0, p)
                pp += pij
                qq += qij
            P.append(pp)
            Q.append(qq)
            V.append(math.sqrt(pe.value(m.e[node_idx, 0, 0, p]) ** 2 + pe.value(m.f[node_idx, 0, 0, p]) ** 2))
        out[str(dn)] = {'branches': [str(b.branch_id) for b in brs], 'P_mw': P, 'Q_mvar': Q, 'V_pu': V}
    return out


def interface_coordinated(tm, dms, tn_day, dn_days):
    base = tn_day.baseMVA
    out = {}
    for k, dn in enumerate(tn_day.active_distribution_network_nodes):
        dm, dd = dms[dn], dn_days[dn]
        out[str(dn)] = {
            'tso_pc_adn_mw': [pe.value(tm.pc_adn[k, 0, 0, p]) * base for p in tm.periods],
            'tso_qc_adn_mvar': [pe.value(tm.qc_adn[k, 0, 0, p]) * base for p in tm.periods],
            'tso_vmag_adn_pu': [pe.value(tm.vmag_adn[k, 0, 0, p]) for p in tm.periods],
            'dso_pg_adn_mw': [pe.value(dm.pg_adn[0, 0, p]) * dd.baseMVA for p in dm.periods],
            'dso_qg_adn_mvar': [pe.value(dm.qg_adn[0, 0, p]) * dd.baseMVA for p in dm.periods],
            'dso_vmag_adn_pu': [pe.value(dm.vmag_adn[0, 0, p]) for p in dm.periods]}
    return out


def compare_interface(merged, coord):
    out = {}
    for dn, mv in merged.items():
        c = coord[dn]
        out[dn] = {
            'max_abs_dP_vs_tso_mw': max(abs(a - b) for a, b in zip(mv['P_mw'], c['tso_pc_adn_mw'])),
            'max_abs_dP_vs_dso_mw': max(abs(a - b) for a, b in zip(mv['P_mw'], c['dso_pg_adn_mw'])),
            'max_abs_dQ_vs_tso_mvar': max(abs(a - b) for a, b in zip(mv['Q_mvar'], c['tso_qc_adn_mvar'])),
            'max_abs_dQ_vs_dso_mvar': max(abs(a - b) for a, b in zip(mv['Q_mvar'], c['dso_qg_adn_mvar'])),
            'max_abs_dV_vs_tso_pu': max(abs(a - b) for a, b in zip(mv['V_pu'], c['tso_vmag_adn_pu'])),
            'hourly_dP_vs_dso_mw': [a - b for a, b in zip(mv['P_mw'], c['dso_pg_adn_mw'])]}
    return out


def _solve_one(md, m, params):
    """The ONE permitted solve site (SolveProfileGuard). Production Network.run_smopf, from the model's current values."""
    n_rec = len(getattr(md, 'ipopt_solve_records', []) or [])
    t0 = time.perf_counter()
    result = md.run_smopf(m, params, from_warm_start=False, print_header=True)
    wall = time.perf_counter() - t0
    recs = list(getattr(md, 'ipopt_solve_records', []) or [])[n_rec:]
    st = {'wall_s': wall, 'ipopt_records': recs,
          'termination_condition': str(result.solver.termination_condition) if result is not None else None,
          'solver_status': str(result.solver.status) if result is not None else None,
          'solution_loaded': result is not None}
    return st


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--trial', action='store_true')
    a = ap.parse_args()
    t_start = time.perf_counter()
    pre = []
    spec_sha = _sha(a.spec)
    tag = os.path.basename(a.spec).rsplit('_', 1)[-1].split('.')[0]
    if not spec_sha.startswith(tag):
        pre.append(f'spec sha256 {spec_sha[:8]} does not match its name tag {tag}')
    if not a.trial:
        for rel in (SCRIPT_REL, a.spec):
            if not committed_clean(rel):
                pre.append(f'{rel} not committed clean')
    if os.environ.get('NLP_SOLVER_PATH') != '/usr/local/bin/ipopt':
        pre.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}, not /usr/local/bin/ipopt")
    if os.path.exists(a.out_dir):
        pre.append(f'{a.out_dir} exists')
    if _sha(PKL_REL) != PKL_SHA:
        pre.append('certified_models.pkl sha256 differs from the pin')
    lock = os.path.join(os.path.dirname(os.path.abspath(a.out_dir)), '.central_check.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        pre.append(f'another run holds {lock}')
        lock = None
    if pre:
        print('[central check] PRECONDITION FAILED:', pre, flush=True)
        if lock:
            os.remove(lock)
        sys.exit(1)
    try:
        run(a, spec_sha, t_start)
    finally:
        if lock and os.path.exists(lock):
            os.remove(lock)


def run(a, spec_sha, t_start):
    with open(a.spec, encoding='utf-8') as h:
        spec = json.load(h)
    with open(CL_REL, encoding='utf-8') as h:
        cl = json.load(h)
    comparator = {}
    for b in cl['blocks'].values():
        if str(b['year']) == str(YEAR):
            d = comparator.setdefault(b['day'], {'gen_plus_flex': 0.0, 'detector_penalty_total': 0.0})
            d['gen_plus_flex'] += b['unweighted']['generation_cost'] + b['unweighted']['flexibility_cost_internal']
            d['detector_penalty_total'] += b['unweighted']['detector_penalty_total']
    # capture checklist, asserted before any solve (CLAUDE.md rule eleven)
    capture = {'status': '_solve_one', 'iterations_exit': 'ipopt_solve_records', 'wall_time': '_solve_one',
               'cost': 'costs', 'comparator': CL_REL, 'interface_merged': 'interface_merged',
               'interface_coordinated': 'interface_coordinated', 'warm_mapping': 'warm_start',
               'warm_point_violation': 'max_violation'}
    for k, fn in capture.items():
        assert fn == CL_REL or fn in globals() or fn == 'ipopt_solve_records', k
    os.makedirs(a.out_dir)
    planning = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
    planning.read_planning_problem()
    tn, dns = planning.transmission_network, planning.distribution_networks
    merged = combine_networks(tn, dns)
    params = configure(merged.params, spec)
    with open(PKL_REL, 'rb') as h:
        models = pickle.load(h)
    res = {'stage': spec['stage'], 'spec': {'path': a.spec, 'sha256': spec_sha}, 'script_sha256': _sha(SCRIPT_REL),
           'git_head': _git('rev-parse', 'HEAD').strip(), 'utc_start': datetime.now(timezone.utc).isoformat(),
           'trial': a.trial, 'solver_options_in_force': dict(params.solver_params.options),
           'recovery_enabled': params.solver_params.recovery_enabled,
           'recovery_tier2_enabled': params.solver_params.recovery_tier2_enabled,
           'comparator': {'path': CL_REL, 'sha256': _sha(CL_REL), 'days': comparator},
           'certified_models': {'path': PKL_REL, 'sha256': PKL_SHA}, 'days': {}}
    gates = spec['pre_solve_gates']
    ctx, failed = {}, []
    # ---- phase A (no solve): every day's warm model is built, mapped and gated BEFORE the first solve
    for day in DAYS:
        md = merged.network[YEAR][day]
        tn_day = tn.network[YEAR][day]
        dn_days = {dn: dns[dn].network[YEAR][day] for dn in dns}
        tm = _by_str(_by_str(models['tso'], YEAR), day)
        dms = {dn: _by_str(_by_str(_by_str(models['dso'], dn), YEAR), day) for dn in dns}
        src = {'tso': tm, **dms}
        maps = source_maps(md, tn_day, dn_days)
        # alignment of the pickled models with the networks read here (element order and demand)
        align = {'tso_sizes': [len(tm.nodes), len(tn_day.nodes), len(tm.generators), len(tn_day.generators),
                               len(tm.branches), len(tn_day.branches)],
                 'dso_load_pd_equal': all(
                     abs(pe.value(dms[dn].pc[c, 0, 0, p]) - dn_days[dn].loads[c].pd[0][p]) == 0.0
                     for dn in dns for c in dms[dn].loads for p in dms[dn].periods)}
        align['ok'] = (align['tso_sizes'][0::2] == align['tso_sizes'][1::2]) and align['dso_load_pd_equal']
        day_res = {'alignment': align, 'comparator': comparator[day],
                   'coordinated_ref_generator_limits': ref_generator_limits(dms, dn_days),
                   'interface_coordinated': interface_coordinated(tm, dms, tn_day, dn_days)}
        wm = md.build_model(params)
        r = {'n_load_values_fixed_at_demand': fix_loads(wm, md), 'param_alignment': align_scalar_params(wm, src)}
        rep, others = warm_start(wm, maps, src, interface_angles(tm, tn_day, list(dns)))
        r['warm_mapping'] = rep
        r['vars_not_mapped_components'] = others
        r['coverage'] = coverage(wm, rep)
        r['at_mapped_point'] = {'costs': costs(wm), 'max_violation': max_violation(wm)}
        g = {'alignment_ok': align['ok'],
             'no_param_conflict': not r['param_alignment']['conflict'],
             'coverage_complete': not r['coverage']['unset_used_by_component'],
             'mapped_point_cost_matches_comparator': abs(r['at_mapped_point']['costs']['C_gen_plus_flex']
                                                         - comparator[day]['gen_plus_flex']) <= gates['mapped_cost_abs_eur'],
             'mapped_point_max_violation_ok': r['at_mapped_point']['max_violation']['max'] <= gates['mapped_max_violation']}
        r['gates'] = g
        failed += [f'{day}:{k}' for k, ok in g.items() if not ok]
        day_res['warm'] = r
        res['days'][day] = day_res
        ctx[day] = (md, wm, src)
        print(f'[central check] {day} pre-solve gates: {g}', flush=True)
    if failed:
        res['refused_before_any_solve'] = failed
        print(f'[central check] REFUSED before any solve: {failed}', flush=True)
    # ---- phase B: the 8 solves (primary from the default start, then the gated warm model)
    for day in ([] if (failed or a.trial) else DAYS):
        md, wm, src = ctx[day]
        day_res = res['days'][day]
        pm = md.build_model(params)
        rp = {'n_load_values_fixed_at_demand': fix_loads(pm, md), 'param_alignment': align_scalar_params(pm, src),
              'at_default_point': {'max_violation': max_violation(pm)}}
        day_res['primary'] = rp
        for start, m, r in (('primary', pm, rp), ('warm', wm, day_res['warm'])):
            md.logs_dir = os.path.join(a.out_dir, 'logs', start)
            os.makedirs(md.logs_dir, exist_ok=True)
            st = _solve_one(md, m, params)
            r['solve'] = st
            if st['solution_loaded']:
                r['costs'] = costs(m)
                c = r['costs']['C_gen_plus_flex']
                cc = comparator[day]['gen_plus_flex']
                r['diff_eur'] = c - cc
                r['diff_pct'] = 100.0 * (c - cc) / cc
                r['diff_in_bars'] = (c - cc) / (spec['resolution']['bar_rel_of_C'] * cc)
                r['max_violation_after'] = max_violation(m)
                r['interface_merged'] = interface_merged(md, m, params, list(dns))
                r['interface_vs_coordinated'] = compare_interface(r['interface_merged'], day_res['interface_coordinated'])
                lim = day_res['coordinated_ref_generator_limits']
                r['dropped_ref_generator_limits_exceeded'] = {
                    dn: {'P_hours_outside': [h for h, x in enumerate(iv['P_mw'])
                                             if (lim[dn]['pg_bounds'][0] is not None and x < lim[dn]['pg_bounds'][0] - 1e-6)
                                             or (lim[dn]['pg_bounds'][1] is not None and x > lim[dn]['pg_bounds'][1] + 1e-6)],
                         'Q_hours_outside': [h for h, x in enumerate(iv['Q_mvar'])
                                             if (lim[dn]['qg_bounds'][0] is not None and x < lim[dn]['qg_bounds'][0] - 1e-6)
                                             or (lim[dn]['qg_bounds'][1] is not None and x > lim[dn]['qg_bounds'][1] + 1e-6)]}
                    for dn, iv in r['interface_merged'].items()}
            print(f"[central check] {day} {start}: {st['termination_condition']} "
                  f"{(st['ipopt_records'] or [{}])[-1].get('exit')} wall {st['wall_s']:.1f}s "
                  f"diff {r.get('diff_eur')} EUR ({r.get('diff_pct')} %)", flush=True)
    res['total_wall_s'] = time.perf_counter() - t_start
    expected = 0 if (a.trial or failed) else 8
    res['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(expected, expected)}
    res['pickle'] = dict(PK)
    res['guards_ok'] = (not res['guard']['verify_failures']) and PK == {'delegated_loads': 1, 'refused': 0}
    out = os.path.join(a.out_dir, 'central_check_results.json')
    with open(out, 'x', encoding='utf-8') as h:
        json.dump(res, h, indent=1, default=str)
    print('[central check] wrote', out, 'guards_ok', res['guards_ok'], flush=True)
    sys.exit(3 if failed else (0 if res['guards_ok'] else 2))


if __name__ == '__main__':
    main()
