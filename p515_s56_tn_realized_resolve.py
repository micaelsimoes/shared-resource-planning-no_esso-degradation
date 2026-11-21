"""P5.15 S56 -- D0: TN realized-exchange re-solve (Planner task after audit D). Exactly 80 IPOPT solves.

Question: the multi-scenario ADMM formulation pins the TN to ONE scenario-common interface exchange (Addendum 38 (A))
while each DN's per-scenario draw differs from it (audit D, `interface_consistency.json`); the deviations are settled
financially only (DSO covariance settlement, row 18). How large is what the pinned approximation omits on the TN side?
NO FORMULATION CHANGE: production builders and solver only; the TN models are re-solved with fixed ADN interface loads.

Frozen spec: data/SRP1/Results/P515S56/tn_realized_resolve/frozen_tn_realized_spec_v1_<sha8>.json (passed with --spec;
its sha256 must start with the <sha8> in its name, and it must be committed clean, as must this script).

Cells (the saved 3 x 3 pair terminals, campaign s53_w91_3x3_pair): x0 (f6e9cd53...) and unit (c82522f4..., node 7,
0.25 MVA / 1.0 MWh). Instance: data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json, read with the reader the
3 x 3 harness used (`p515_s44_scale_measurement.read_planning_from_derived_case`, production SharedResourcesPlanning),
combined scenario checksum checked against the campaign spec's declaration.

Per cell and arm, the 20 (year, day) TSO block models are built by production `NetworkData.build_model()` on a deep copy
of the instance, with the candidate applied as production does (`update_data_with_candidate_solution` /
`update_model_with_candidate_solution` with the candidate's total capacity, then the terminal published AVAILABLE
capacity per node and year from `evaluation_record.json`, the value the TSO block carried at the terminal cycle), and
production `_prepare_transmission_objectives_for_admm` (penalty_shared_ess_usage 0, penalty_gen_curtailment 0,
cost_load_curtailment at its ADMM value); the interface settlement weight is then set to 0 (the central check's
alignment: the settlement is a transfer and is constant here, every interface quantity being fixed). NO ADMM
augmented-Lagrangian / proximal terms, NO interface-voltage pin, NO row 18 (the TSO never carries it), NO expected-value
Vars. The ADN setup is the one `create_transmission_network_model` applies (slack_v_sqr at the ADN bus fixed at 0, pc/qc
bounds removed, the ADN flexibility legs fixed at 0), with:
  Arm A (control): pc/qc of each ADN load fixed in every scenario at the TSO's terminal scenario-common exchange
                   (`interface_profiles.per_scenario` TSO side = pc + interface_delta, asserted scenario-common);
                   interface_delta_p/q fixed at 0 in every scenario.
  Arm B          : pc/qc fixed per scenario at the DN's realized pg_adn / qg_adn of that scenario (DSO block
                   `interface_profiles.per_scenario`, key '<s_m>_<s_o>' -- the formulation's own pairing); otherwise A.
  Both           : shared-ESS pnet/qnet (the scenario-free copy, `sess_na_scenario`) fixed at the TSO block's terminal
                   schedule (`shared_ess_schedule`).
The TN node balance (`compute_node_load`) reads pc[c, s_m, s_o, p] per scenario, so per-scenario fixed values are
carried without any code change (mch `compute_node_load`, `interface_pf_p_transmission_def`).

Objective solved = production `objective_function_rule` = total_gen_cost + total_flex_cost + total_load_curt_cost +
total_gen_curt_penalty (weight 0) + total_ess_utilization_cost_penalty (shared weight 0) + total_slack_penalties +
total_ess_complementarity_penalties (shared-ESS day-balance slack) + 0 x interface_settlement; checked at the built point.

Statistics (spec 'statistics'): C = total_gen_cost + total_flex_cost (EUR per representative day, probability-weighted
over the 9 scenarios); weighted = sum_b w_b C_b, w_b = production `_get_admm_block_weight` (years x days x
1.02^-(y-2025)), the Q weighting. cov_b = sum_DN DSO-block covariance settlement (settlement 'deviation', recomputed
from the per-scenario profiles and the block's own price array and checked). D1_b = (C_B - C_A) - cov_b.

GUARDS. SolveProfileGuard(permitted=[(this file, '_solve_one')]) installed before any project import and verified at
exactly 80 solves and 80 executions (0 under --trial or a pre-solve refusal); pickle.load/loads blocked for the whole run
(no pickle is read); single-instance lock; refuses if the campaign lock exists or a campaign / concurrency-gate process
or another copy runs; output directory must not exist; stdout and stderr captured by the caller (attached run).

Usage: NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s56_tn_realized_resolve.py --spec SPEC \\
           --out-dir DIR [--trial]
       (--trial: skips the committed-clean checks and the solves -- reads, builds and gates only)
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT_REL = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=((SCRIPT_REL, '_solve_one'),), label='P5.15 S56 D0 TN realized re-solve').install()

import pickle  # noqa: E402

PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked (D0 reads no pickle)')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pyomo.environ as pe  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import model_construction_helpers as mch  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

EVAL_ROOT = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals')
CAMPAIGN_SPEC = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair',
                             'campaign_spec_s53_w91_3x3_pair_231558f0.json')
CAMPAIGN_LOCK = os.path.join(REPO, '.p515_s44_campaign.lock')
CONCURRENT_SCRIPTS = ('p515_s44_campaign_harness.py', 'p515_s56_concurrency_gate.py', SCRIPT_REL,
                      'p515_s56_central_ms_build.py')
SOLVE_TOL = {'gen_bound_pu': 1e-5, 'branch_rel': 1e-4, 'vmag_sqr': 1e-6}
CHECKS = []


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [D0] {msg}', flush=True)


def _ck(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:400]}' if detail is not None and not ok else ''))
    return bool(ok)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout


def committed_clean(rel):
    return bool(_git('ls-files', '--', rel).strip()) and not _git('status', '--porcelain', '--', rel).strip()


def _input_record(path):
    tracked = bool(_git('ls-files', '--', path).strip())
    clean = (not _git('status', '--porcelain', '--', path).strip()) if tracked else None
    return {'path': path, 'sha256': _sha(path), 'tracked': tracked, 'clean': clean}


def concurrent_processes():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me = os.getpid()
    found = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(
                os.path.basename(t) in CONCURRENT_SCRIPTS for t in tok[2:4]):
            found.append(ln.strip()[:300])
    return found


class _Stages:
    @contextmanager
    def stage(self, name):
        t0 = time.time()
        _log(f'stage: {name}')
        yield
        _log(f'stage done ({time.time() - t0:.1f} s): {name}')


def configure(params, spec):
    p = deepcopy(params)
    p.solver_params.options = dict(p.solver_params.options or {})
    p.solver_params.options.update(spec['solver']['option_overrides'])
    p.solver_params.recovery_enabled = False
    p.solver_params.recovery_tier2_enabled = False
    return p


def _year_key(keys, year):
    for k in keys:
        if int(k) == int(year):
            return k
    raise KeyError(year)


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1.0)


# ------------------------------------------------------------------------------------------------- inputs (no solve)
def load_cell(cell):
    edir = os.path.join(EVAL_ROOT, cell['eval_dir'])
    man = json.load(open(os.path.join(edir, 'child_manifest_sha256.json')))
    rec = {}
    for fname in ('multiscenario_terminal.json', 'evaluation_record.json', 'component_levels_terminal.json'):
        p = os.path.join(edir, fname)
        r = _input_record(p)
        r['pinned_sha256_child_manifest'] = man.get(p)
        r['pinned_ok'] = man.get(p) == r['sha256']
        _ck(f"{cell['label']}: {fname} sha256 == child manifest pin", r['pinned_ok'], r)
        rec[fname] = r
    rec['child_manifest_sha256.json'] = _input_record(os.path.join(edir, 'child_manifest_sha256.json'))
    ms = json.load(open(os.path.join(edir, 'multiscenario_terminal.json')))
    er = json.load(open(os.path.join(edir, 'evaluation_record.json')))
    cl = json.load(open(os.path.join(edir, 'component_levels_terminal.json')))
    _ck(f"{cell['label']}: candidate_canonical == spec", er['candidate_canonical'] == cell['candidate_canonical'],
        er['candidate_canonical'])
    _ck(f"{cell['label']}: status certified, terminal == certification cycle",
        er['status'] == 'certified' and (er.get('certification') or {}).get('certification_cycle') == er['cycles_run'],
        {'status': er['status'], 'cycles_run': er['cycles_run']})
    return {'edir': edir, 'inputs': rec, 'ms': ms, 'er': er, 'cl': cl}


def block_data(cell_label, ms, tn, year, day, dn_ids):
    """Per (year, day): the TSO common exchange, the DN realized exchanges, probabilities, prices, the DSO covariance
    (saved and recomputed), the shared-ESS schedule. Gated, zero solves."""
    yk = str(int(year))
    tb = ms['blocks'][f'TSO|{yk}|{day}']
    net = tn.network[year][day]
    scen = [(s_m, s_o) for s_m in range(len(net.prob_market_scenarios))
            for s_o in range(len(net.prob_operation_scenarios))]
    omega = {f'{s_m}_{s_o}': net.prob_market_scenarios[s_m] * net.prob_operation_scenarios[s_o] for s_m, s_o in scen}
    tag = f'{cell_label} {yk} {day}'
    ok = _ck(f'{tag}: 9 scenarios; TSO per-scenario probabilities == network product',
             tb['n_scenarios'] == 9 and len(scen) == 9 and all(
                 abs(tb['per_scenario_costs'][k]['probability'] - omega[k]) <= 1e-15 for k in omega),
             {'n': tb['n_scenarios']})
    w_saved = tb['admm_block_weight']
    w_prod = srp._get_admm_block_weight(tn, year, day)
    ok &= _ck(f'{tag}: admm_block_weight saved == production', w_saved == w_prod, [w_saved, w_prod])
    tso_p, tso_q, dso_p, dso_q, cov, cov_re, pi_tn_vs_dn, prices = {}, {}, {}, {}, {}, {}, 0.0, None
    for dn in dn_ids:
        tp = tb['interface_profiles']['per_scenario'][str(dn)]
        sp = max(abs(a - b) for k in tp for a, b in zip(tp['0_0']['p'], tp[k]['p']))
        sq = max(abs(a - b) for k in tp for a, b in zip(tp['0_0']['q'], tp[k]['q']))
        ok &= _ck(f'{tag} DN{dn}: TSO exchange scenario-common (max spread P, Q <= 1e-9)', sp <= 1e-9 and sq <= 1e-9,
                  [sp, sq])
        tso_p[dn], tso_q[dn] = list(tp['0_0']['p']), list(tp['0_0']['q'])
        db = ms['blocks'][f'DSO|{dn}|{yk}|{day}']
        ok &= _ck(f'{tag} DN{dn}: DSO probabilities == TSO', all(
            abs(db['per_scenario_costs'][k]['probability'] - omega[k]) <= 1e-15 for k in omega))
        dsp = db['interface_profiles']['per_scenario']
        dso_p[dn] = {k: list(dsp[k]['p']) for k in omega}
        dso_q[dn] = {k: list(dsp[k]['q']) for k in omega}
        pi = db['dispersion']['pi_by_market_by_hour']          # [t][s_m], EUR/MWh, the DSO block's own price array
        dnet_prices = [[float(x) for x in row] for row in pi]
        tn_pi = [[net.cost_energy_p[s_m][t] for s_m in range(len(net.prob_market_scenarios))] for t in range(24)]
        pi_tn_vs_dn = max(pi_tn_vs_dn, max(abs(a - b) for r1, r2 in zip(dnet_prices, tn_pi) for a, b in zip(r1, r2)))
        prices = tn_pi
        pbar = [sum(omega[k] * dso_p[dn][k][t] for k in omega) for t in range(24)]
        c = sum(omega[f'{s_m}_{s_o}'] * sum(pi[t][s_m] * (dso_p[dn][f'{s_m}_{s_o}'][t] - pbar[t]) for t in range(24))
                for s_m, s_o in scen)
        cov_re[dn] = c
        cov[dn] = db['settlement']['deviation']
        ok &= _ck(f'{tag} DN{dn}: DSO covariance recomputed == saved settlement deviation (abs <= 1e-6 max(1,|x|))',
                  abs(c - cov[dn]) <= 1e-6 * max(1.0, abs(cov[dn])), [c, cov[dn]])
    ok &= _ck(f'{tag}: TN price array == every DN block price array (max abs <= 1e-9)', pi_tn_vs_dn <= 1e-9,
              pi_tn_vs_dn)
    sess = {}
    for dn in dn_ids:
        sch = tb['shared_ess_schedule'][str(dn)]
        ok &= _ck(f'{tag} DN{dn}: shared-ESS schedule on the na scenario [0, 0]', list(sch['na_scenario']) == [0, 0])
        sess[dn] = {'pnet_mw': list(sch['pnet_mw']), 'qnet_mvar': list(sch['qnet_mvar'])}
    tso_cost_saved = {
        'block_recourse_production_local': tb['block_recourse_production_local'],
        'per_scenario_generation_conventional_cost': {k: tb['per_scenario_costs'][k]['generation_conventional_cost']
                                                      for k in omega},
        'per_scenario_obj': {k: tb['per_scenario_costs'][k]['obj'] for k in omega},
        'per_scenario_res_curtailed_mwh': {k: tb['per_scenario_costs'][k]['generation_renewable_curtailed']['s']
                                           for k in omega}}
    return {'ok': ok, 'omega': omega, 'w': w_prod, 'base': net.baseMVA, 'tso_p': tso_p, 'tso_q': tso_q, 'dso_p': dso_p, 'dso_q': dso_q,
            'cov_saved': cov, 'cov_recomputed': cov_re, 'cov_block': sum(cov.values()), 'sess': sess,
            'prices': prices, 'tso_cost_saved': tso_cost_saved}


# ------------------------------------------------------------------------------------------------------ model build
def build_arm(tn, candidate, avail, cell_label, arm, bdata, dn_ids):
    """Production build of the 20 TSO blocks for one (cell, arm); returns (models, per-block gate records)."""
    models = tn.build_model()
    tn.update_model_with_candidate_solution(models, candidate['total_capacity'])
    tn.update_model_with_candidate_solution(models, avail)       # terminal published available capacity (ADMM value)
    srp._prepare_transmission_objectives_for_admm(tn, models)
    recs = {}
    for year in tn.years:
        for day in tn.days:
            m, net = models[year][day], tn.network[year][day]
            bd = bdata[(str(int(year)), day)]
            s_base = net.baseMVA
            m.interface_settlement_weight.set_value(0.0)
            s_m0, s_o0 = mch.sess_na_scenario(m)
            n_fixed = 0
            for dn_pos, dn in enumerate(net.active_distribution_network_nodes):
                node_idx = net.get_node_idx(dn)
                c = net.get_adn_load_idx(dn)
                for s_m in m.scenarios_market:
                    for s_o in m.scenarios_operation:
                        key = f'{s_m}_{s_o}'
                        for p in m.periods:
                            if tn.params.slacks.grid_operation.voltage:
                                m.slack_v_sqr_down[node_idx, s_m, s_o, p].fix(0.0)
                                m.slack_v_sqr_up[node_idx, s_m, s_o, p].fix(0.0)
                            pv = (bd['tso_p'][dn][p] if arm == 'A' else bd['dso_p'][dn][key][p]) / s_base
                            qv = (bd['tso_q'][dn][p] if arm == 'A' else bd['dso_q'][dn][key][p]) / s_base
                            for var, val in ((m.pc[c, s_m, s_o, p], pv), (m.qc[c, s_m, s_o, p], qv)):
                                var.setlb(None)
                                var.setub(None)
                                var.fix(val)
                                n_fixed += 1
                            for leg in (m.flex_p_up, m.flex_p_down, m.flex_q_up, m.flex_q_down):
                                leg[c, s_m, s_o, p].fix(0.0)
                            m.interface_delta_p[dn_pos, s_m, s_o, p].fix(0.0)
                            m.interface_delta_q[dn_pos, s_m, s_o, p].fix(0.0)
                e = net.get_shared_energy_storage_idx(dn)
                for p in m.periods:
                    m.shared_es_pnet[e, s_m0, s_o0, p].fix(bd['sess'][dn]['pnet_mw'][p] / s_base)
                    m.shared_es_qnet[e, s_m0, s_o0, p].fix(bd['sess'][dn]['qnet_mvar'][p] / s_base)
            recs[(str(int(year)), day)] = model_gates(m, net, tn.params, cell_label, arm, year, day, n_fixed, dn_ids)
    return models, recs


OBJ_PARTS = ('total_gen_cost', 'total_flex_cost', 'total_load_curt_cost', 'total_gen_curt_penalty',
             'total_ess_utilization_cost_penalty', 'total_slack_penalties', 'total_ess_complementarity_penalties')


def model_gates(m, net, params, cell_label, arm, year, day, n_fixed, dn_ids):
    tag = f'{cell_label} {arm} {int(year)} {day}'
    absent = [n for n in ('row18_deviation_charge', 'scenario_voltage_pin', 'expected_interface_pf_p',
                          'expected_interface_vmag', 'dual_pf_p_req', 'p_pf_req', 'vmag_req') if hasattr(m, n)]
    delta_ok = all(v.fixed and v.value == 0.0 for v in m.interface_delta_p.values()) and all(
        v.fixed and v.value == 0.0 for v in m.interface_delta_q.values())
    obj = pe.value(m.objective)
    rule = pe.value(mch.objective_function_rule(m, params))
    parts = sum(pe.value(getattr(m, k)) for k in OBJ_PARTS)
    scal = {prm.name: pe.value(prm) for prm in m.component_objects(pe.Param, active=True)
            if not prm.is_indexed() and prm.mutable}
    g = {'n_pc_qc_fixed': n_fixed, 'n_pc_qc_expected': 2 * len(dn_ids) * 9 * 24,
         'absent_admm_pin_row18_components_present': absent, 'interface_delta_all_fixed_zero': delta_ok,
         'objective_eq_rule_at_built_point': _rel(obj, rule) <= 1e-12,
         'objective_eq_sum_of_parts_at_built_point': _rel(obj, parts) <= 1e-12,
         'settlement_weight_zero': pe.value(m.interface_settlement_weight) == 0.0,
         'penalty_gen_curtailment_zero': pe.value(m.penalty_gen_curtailment) == 0.0,
         'penalty_shared_ess_usage_zero': pe.value(m.penalty_shared_ess_usage) == 0.0}
    ok = (g['n_pc_qc_fixed'] == g['n_pc_qc_expected'] and not absent and delta_ok
          and g['objective_eq_rule_at_built_point'] and g['objective_eq_sum_of_parts_at_built_point']
          and g['settlement_weight_zero'] and g['penalty_gen_curtailment_zero'] and g['penalty_shared_ess_usage_zero'])
    _ck(f'{tag}: model gates', ok, g)
    g['scalar_mutable_params'] = scal
    g['ok'] = ok
    return g


# --------------------------------------------------------------------------------------------------- solve + extract
def _solve_one(net, m, params):
    """The ONE permitted solve site (SolveProfileGuard). Production Network.run_smopf from the model's own values."""
    n_rec = len(getattr(net, 'ipopt_solve_records', []) or [])
    t0 = time.perf_counter()
    result = net.run_smopf(m, params, from_warm_start=False, print_header=True)
    wall = time.perf_counter() - t0
    recs = list(getattr(net, 'ipopt_solve_records', []) or [])[n_rec:]
    return {'wall_s': wall, 'ipopt_records': recs,
            'termination_condition': str(result.solver.termination_condition) if result is not None else None,
            'solver_status': str(result.solver.status) if result is not None else None,
            'solution_loaded': result is not None}


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


SLACK_VARS = ('slack_v_sqr_down', 'slack_v_sqr_up', 'slack_node_balance_p_up', 'slack_node_balance_p_down',
              'slack_node_balance_q_up', 'slack_node_balance_q_down', 'slack_flow_ij_sqr', 'slack_flow_ji_sqr',
              'slack_flex_p_balance_up', 'slack_flex_p_balance_down', 'slack_shared_es_soc_final_up',
              'slack_shared_es_soc_final_down')


def _at_bound(v, side):
    """True when a non-fixed, non-degenerate pg (controllable units only, at the call site) sits within
    SOLVE_TOL['gen_bound_pu'] of its lower/upper bound."""
    lb, ub, x = v.lb, v.ub, v.value
    if v.fixed or x is None or lb is None or ub is None or ub - lb <= 1e-9:
        return False
    return (ub - x if side == 'ub' else x - lb) <= SOLVE_TOL['gen_bound_pu']


def extract(m, net, params, bd, dn_ids):
    """Everything reported per solved block (zero solves)."""
    base = net.baseMVA
    out = {'costs': {k: float(pe.value(getattr(m, k))) for k in OBJ_PARTS}}
    out['costs']['C_gen_plus_flex'] = out['costs']['total_gen_cost'] + out['costs']['total_flex_cost']
    out['costs']['objective'] = float(pe.value(m.objective))
    out['per_scenario'] = {f'{s_m}_{s_o}': {'gen_cost': float(pe.value(m.gen_cost_scenario[s_m, s_o])),
                                            'flex_cost': float(pe.value(m.flex_cost_scenario[s_m, s_o]))}
                           for s_m in m.scenarios_market for s_o in m.scenarios_operation}
    out['detector_components'] = srp._get_local_detector_components(m, net, params)
    sl = {}
    for name in SLACK_VARS:
        v = getattr(m, name, None)
        if v is None:
            continue
        vals = [x.value for x in v.values() if x.value is not None]
        sl[name] = {'max': max(vals) if vals else None, 'min': min(vals) if vals else None}
    out['slack_var_extremes'] = sl
    out['max_violation'] = max_violation(m)
    ctrl = [g for g in m.generators if net.generators[g].is_controllable()]
    curt = [g for g in m.generators if net.generators[g].is_curtaillable()]
    has_dual = hasattr(m, 'dual') and len(m.dual) > 0
    hours = {}
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            key = f'{s_m}_{s_o}'
            w = bd['omega'][key]
            rows = []
            for p in m.periods:
                r = {'gen_ctrl_mw': sum(m.pg[g, s_m, s_o, p].value for g in ctrl) * base,
                     'gen_cost_eur': sum(net.cost_energy_p[s_m][p] * base * m.pg[g, s_m, s_o, p].value for g in ctrl),
                     'res_curt_mw': sum((pe.value(m.pg_avail[g, s_o, p]) - m.pg[g, s_m, s_o, p].value) for g in curt)
                     * base,
                     'gens_at_ub': [int(g) for g in ctrl if _at_bound(m.pg[g, s_m, s_o, p], 'ub')],
                     'gens_at_lb': [int(g) for g in ctrl if _at_bound(m.pg[g, s_m, s_o, p], 'lb')],
                     'branches_binding': [int(b) for b in m.branches if net.branches[b].status and net.branches[b].rate
                                          and pe.value(m.flow_ij_sqr[b, s_m, s_o, p])
                                          >= (net.branches[b].rate / base) ** 2 * (1 - SOLVE_TOL['branch_rel'])],
                     'nodes_at_vmax': [int(i) for i in m.nodes if net.nodes[i].v_max ** 2
                                       - m.vmag_sqr[i, s_m, s_o, p].value <= SOLVE_TOL['vmag_sqr']],
                     'nodes_at_vmin': [int(i) for i in m.nodes if m.vmag_sqr[i, s_m, s_o, p].value
                                       - net.nodes[i].v_min ** 2 <= SOLVE_TOL['vmag_sqr']],
                     'adn': {}}
                for dn in dn_ids:
                    i = net.get_node_idx(dn)
                    a = {'pc_mw': m.pc[net.get_adn_load_idx(dn), s_m, s_o, p].value * base,
                         'vmag_pu': math.sqrt(m.vmag_sqr[i, s_m, s_o, p].value)}
                    if has_dual:
                        dp = m.dual.get(m.node_balance_p[i, s_m, s_o, p])
                        dq = m.dual.get(m.node_balance_q[i, s_m, s_o, p])
                        a['dual_p_raw'] = dp
                        a['dual_q_raw'] = dq
                        a['mc_p_eur_per_mwh'] = (dp / (w * base)) if dp is not None else None
                        a['mc_q_eur_per_mvarh'] = (dq / (w * base)) if dq is not None else None
                    r['adn'][str(dn)] = a
                rows.append(r)
            hours[key] = rows
    out['hours'] = hours
    out['duals_available'] = has_dual
    return out


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
        for rel in (SCRIPT_REL, os.path.relpath(os.path.abspath(a.spec), REPO)):
            if not committed_clean(rel):
                pre.append(f'{rel} not committed clean')
    if os.environ.get('NLP_SOLVER_PATH') != '/usr/local/bin/ipopt':
        pre.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}, not /usr/local/bin/ipopt")
    if os.path.exists(a.out_dir):
        pre.append(f'{a.out_dir} exists')
    if os.path.exists(CAMPAIGN_LOCK):
        pre.append(f'campaign lock {CAMPAIGN_LOCK} exists')
    others = concurrent_processes()
    if others:
        pre.append(f'concurrent processes: {others}')
    lock = os.path.join(os.path.dirname(os.path.abspath(a.out_dir)), '.tn_realized_resolve.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        pre.append(f'another run holds {lock}')
        lock = None
    if pre:
        print('[D0] PRECONDITION FAILED:', pre, flush=True)
        if lock:
            os.remove(lock)
        return 1
    try:
        return run(a, spec_sha, t_start)
    finally:
        if lock and os.path.exists(lock):
            os.remove(lock)


def run(a, spec_sha, t_start):
    with open(a.spec, encoding='utf-8') as h:
        spec = json.load(h)
    # capture checklist, asserted before anything is read or solved (CLAUDE.md rule eleven): every quantity the spec
    # 'reported' list names has a capture path in this file
    capture = {'C_A': 'extract', 'C_B': 'extract', 'status_iterations_time': '_solve_one',
               'slack_detector_terms': 'extract', 'max_constraint_violation': 'max_violation',
               'dso_covariance_per_block': 'block_data', 'D1': 'statistics', 'C_A_vs_saved_tso_cost': 'statistics',
               'bus_duals_adn': 'extract', 'curtailment_binding_limits_per_hour': 'extract'}
    missing = [k for k in spec['reported'] if k not in capture]
    assert not missing, f'spec quantities without a capture path: {missing}'
    for k, fn in capture.items():
        assert fn in globals(), (k, fn)
    os.makedirs(a.out_dir)
    expected_solves = 0 if a.trial else spec['guards']['expected_solves']
    res = {'stage': spec['stage'], 'spec': {'path': a.spec, 'sha256': spec_sha}, 'script': SCRIPT_REL,
           'script_sha256': _sha(SCRIPT_REL), 'git_head': _git('rev-parse', 'HEAD').strip(),
           'utc_start': datetime.now(timezone.utc).isoformat(), 'trial': a.trial,
           'interpreter': sys.executable, 'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'),
           'objective_convention': spec['objective']['reported_C'], 'weighting': spec['statistics']['weighting'],
           'cells': {}}
    modules = {p: _input_record(p) for p in ('network.py', 'network_data.py', 'model_construction_helpers.py',
                                             'shared_resources_planning.py', 'p515_s44_scale_measurement.py',
                                             spec['instance']['case_path'], CAMPAIGN_SPEC)}
    res['modules'] = modules
    inst = spec['instance']
    _ck('instance case sha256 == spec', _sha(inst['case_path']) == inst['case_sha256'])
    cspec = json.load(open(CAMPAIGN_SPEC))
    di = cspec['configuration']['derived_instance']
    _ck('instance == the campaign spec declaration (path, sha256, checksum)',
        di['case_path'] == inst['case_path'] and di['case_sha256'] == inst['case_sha256']
        and di['scenario_checksum'] == inst['scenario_checksum'])
    planning = S.read_planning_from_derived_case({'derived_case': {'path': inst['case_path']}}, a.out_dir, _Stages())
    checksum = (planning.scenario_metadata or {}).get('combined_scenario_checksum')
    _ck('combined scenario checksum == declaration', checksum == inst['scenario_checksum'], checksum)
    sed = planning.shared_ess_data
    dn_ids = list(planning.transmission_network.active_distribution_network_nodes)
    _ck('TN ADN nodes == [5, 7, 9]', dn_ids == [5, 7, 9], dn_ids)
    params_solve = configure(planning.transmission_network.params, spec)
    res['solver_options_in_force'] = dict(params_solve.solver_params.options)
    res['recovery'] = [params_solve.solver_params.recovery_enabled, params_solve.solver_params.recovery_tier2_enabled]

    # ---- phase A (zero solves): inputs, block data, candidate application and model gates for every (cell, arm)
    ctx = {}
    for cell in spec['cells']:
        lab = cell['label']
        cd = load_cell(cell)
        pc = deepcopy(planning)
        tn = pc.transmission_network
        cand = pc.get_initial_candidate_solution()
        iy = _year_key(pc.shared_ess_data.years, cell['candidate_canonical']['investment_year'])
        for n, (s_val, e_val) in cell['candidate_canonical']['nodes'].items():
            cand['investment'][int(n)][iy]['s'] = float(s_val)
            cand['investment'][int(n)][iy]['e'] = float(e_val)
        srp._rebuild_candidate_total_capacities(pc, cand)
        tn.update_data_with_candidate_solution(cand['total_capacity'])
        avail = {}
        for n in dn_ids:
            pub = cd['er']['storage_per_node'][str(n)]['published_available_capacity_terminal']
            avail[n] = {y: {'s': pub[str(int(y))]['s_available'], 'e': pub[str(int(y))]['e_available']}
                        for y in tn.years}
        bdata = {}
        for year in tn.years:
            for day in tn.days:
                bdata[(str(int(year)), day)] = block_data(lab, cd['ms'], tn, year, day, dn_ids)
        cov_total = sum(bd['w'] * bd['cov_block'] for bd in bdata.values())
        saved_dso = cd['ms']['summary']['settlement_identity']['dso_deviation_part']
        _ck(f'{lab}: sum_b w_b cov_b == summary DSO deviation part (rel <= 1e-9)',
            _rel(cov_total, sum(saved_dso.values())) <= 1e-9, [cov_total, sum(saved_dso.values())])
        arms_gate = {}
        for arm in ('A', 'B'):
            models, recs = build_arm(tn, cand, avail, lab, arm, bdata, dn_ids)
            arms_gate[arm] = recs
            del models
        ctx[lab] = {'cell': cell, 'cd': cd, 'tn': tn, 'cand': cand, 'avail': avail, 'bdata': bdata}
        res['cells'][lab] = {
            'eval_dir': cd['edir'], 'inputs': cd['inputs'], 'candidate_canonical': cell['candidate_canonical'],
            'candidate_total_capacity': {str(n): {str(y): v for y, v in d.items()}
                                         for n, d in cand['total_capacity'].items()},
            'available_capacity_applied': {str(n): {str(y): v for y, v in d.items()} for n, d in avail.items()},
            'dso_covariance_weighted_total': cov_total, 'dso_covariance_saved_summary': saved_dso,
            'certification_bar_eur': cd['er']['bar']['value'],
            'terminal_gross_operational_cost': cd['er']['terminal_gross_operational_cost'],
            'pre_solve_model_gates': {arm: {f'{k[0]}|{k[1]}': v for k, v in r.items()} for arm, r in arms_gate.items()},
            'blocks': {}}
    failed = [c['check'] for c in CHECKS if not c['ok']]
    if failed:
        res['refused_before_any_solve'] = failed
        _log(f'REFUSED before any solve: {len(failed)} failed checks')
    # ---- phase B: the 80 solves
    for cell in ([] if (failed or a.trial) else spec['cells']):
        lab = cell['label']
        c = ctx[lab]
        tn, bdata = c['tn'], c['bdata']
        for arm in ('A', 'B'):
            models, recs = build_arm(tn, c['cand'], c['avail'], lab, arm, bdata, dn_ids)
            for year in tn.years:
                for day in tn.days:
                    key = (str(int(year)), day)
                    net, m = tn.network[year][day], models[year][day]
                    blk = res['cells'][lab]['blocks'].setdefault(f'{key[0]}|{day}', {})
                    r = {'model_gates_ok': recs[key]['ok']}
                    net.logs_dir = os.path.join(a.out_dir, 'logs', lab, arm)
                    os.makedirs(net.logs_dir, exist_ok=True)
                    st = _solve_one(net, m, params_solve)
                    r['solve'] = st
                    if st['solution_loaded']:
                        try:
                            r.update(extract(m, net, tn.params, bdata[key], dn_ids))
                        except Exception:  # noqa: BLE001 -- a capture failure is recorded, never hides the solve
                            r['extract_error'] = traceback.format_exc()
                            _ck(f'{lab} {arm} {key[0]} {day}: extraction', False, r['extract_error'][-400:])
                    blk[arm] = r
                    ex = (st['ipopt_records'] or [{}])[-1]
                    _log(f"{lab} {arm} {key[0]} {day}: {st['termination_condition']} {ex.get('exit')} "
                         f"it {ex.get('iterations')} wall {st['wall_s']:.1f}s "
                         f"C {r.get('costs', {}).get('C_gen_plus_flex')}")
                    models[year][day] = None
            del models
        for year in tn.years:
            for day in tn.days:
                key = (str(int(year)), day)
                res['cells'][lab]['blocks'][f'{key[0]}|{day}']['inputs_used'] = {
                    'w': bdata[key]['w'], 'omega': bdata[key]['omega'], 'cov_block': bdata[key]['cov_block'],
                    'cov_saved_by_dn': {str(k): v for k, v in bdata[key]['cov_saved'].items()},
                    'cov_recomputed_by_dn': {str(k): v for k, v in bdata[key]['cov_recomputed'].items()},
                    'tso_cost_saved': bdata[key]['tso_cost_saved'],
                    'tso_p_common_mw': {str(k): v for k, v in bdata[key]['tso_p'].items()},
                    'prices_eur_per_mwh_by_hour_by_market': bdata[key]['prices']}
    if not (failed or a.trial):
        try:
            res['statistics'] = statistics(res, ctx, dn_ids)
        except Exception:  # noqa: BLE001 -- recorded; the per-block records are still written
            res['statistics_error'] = traceback.format_exc()
            _ck('statistics computed', False, res['statistics_error'][-400:])
    res['total_wall_s'] = time.perf_counter() - t_start
    gv = GUARD.verify(expected_solves, expected_solves)
    res['guard'] = {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                    'expected_solves': expected_solves, 'verify_failures': gv}
    res['pickle'] = dict(PK)
    res['guards_ok'] = (not gv) and PK == {'load': 0, 'loads': 0}
    res['checks'] = CHECKS
    res['failed_checks'] = [c['check'] for c in CHECKS if not c['ok']]
    res['utc_end'] = datetime.now(timezone.utc).isoformat()
    out = os.path.join(a.out_dir, 'tn_realized_resolve_results.json')
    with open(out, 'x', encoding='utf-8') as h:
        GRIO.dump(res, h, indent=1, default=GRIO.json_default_item)
    manifest = {'outputs': {os.path.basename(out): _sha(out)},
                'inputs': {**{f"{lab}/{k}": v for lab, cc in res['cells'].items() for k, v in cc['inputs'].items()},
                           **{f'module/{k}': v for k, v in modules.items()}},
                'spec': {'path': a.spec, 'sha256': spec_sha},
                'logs_dir_files': {os.path.relpath(os.path.join(dp, f), a.out_dir): _sha(os.path.join(dp, f))
                                   for dp, _d, fs in os.walk(os.path.join(a.out_dir, 'logs')) for f in fs},
                'note': 'the stdout/stderr launch logs are written while the script runs; hashed in the Worker report'}
    with open(os.path.join(a.out_dir, 'manifest_sha256.json'), 'x', encoding='utf-8') as h:
        GRIO.dump(manifest, h, indent=1)
    _log(f"wrote {out}; guards_ok {res['guards_ok']}; failed checks {len(res['failed_checks'])}")
    if failed:
        return 3
    return 0 if res['guards_ok'] else 2


# ------------------------------------------------------------------------------------------------------- statistics
def statistics(res, ctx, dn_ids):
    out = {'per_cell': {}}
    for lab, cc in res['cells'].items():
        bd_all = ctx[lab]['bdata']
        rows = {}
        tot = {'C_A_w': 0.0, 'C_B_w': 0.0, 'dC_w': 0.0, 'cov_w': 0.0, 'D1_w': 0.0, 'V_w': 0.0, 'D1prime_w': 0.0,
               'first_order_dC_w': 0.0, 'objA_w': 0.0, 'objB_w': 0.0, 'C_A_minus_saved_gen_w': 0.0,
               'C_A_minus_saved_local_w': 0.0}
        n_opt = {'A': 0, 'B': 0}
        complete = True
        for bkey, blk in cc['blocks'].items():
            y, d = bkey.split('|')
            bd = bd_all[(y, d)]
            w = bd['w']
            A, B = blk.get('A', {}), blk.get('B', {})
            for arm, r in (('A', A), ('B', B)):
                ex = ((r.get('solve') or {}).get('ipopt_records') or [{}])[-1]
                if (r.get('solve') or {}).get('termination_condition') == 'optimal' and \
                        ex.get('exit') == 'Optimal Solution Found.':
                    n_opt[arm] += 1
            if 'costs' not in A or 'costs' not in B:
                complete = False
                rows[bkey] = {'complete': False}
                continue
            cA, cB = A['costs']['C_gen_plus_flex'], B['costs']['C_gen_plus_flex']
            # V_b: the DN realized draw change valued at the scenario price (= cov_b + the consensus-residual part)
            V = sum(bd['omega'][k] * sum(bd['prices'][t][int(k.split('_')[0])]
                                         * (bd['dso_p'][dn][k][t] - bd['tso_p'][dn][t]) for t in range(24))
                    for dn in dn_ids for k in bd['omega'])
            fo = None
            if A.get('duals_available'):
                fo = 0.0
                base = bd['base']
                for k in bd['omega']:
                    for t in range(24):
                        for dn in dn_ids:
                            a_ = A['hours'][k][t]['adn'][str(dn)]
                            b_ = B['hours'][k][t]['adn'][str(dn)]
                            if a_.get('dual_p_raw') is not None:
                                fo += a_['dual_p_raw'] * (b_['pc_mw'] - a_['pc_mw']) / base
                            if a_.get('dual_q_raw') is not None:
                                dq = (bd['dso_q'][dn][k][t] - bd['tso_q'][dn][t]) / base
                                fo += a_['dual_q_raw'] * dq
            saved_gen = ctx[lab]['cd']['cl']['blocks'][f'TSO|{y}|{d}']['unweighted']['generation_cost']
            saved_local = bd['tso_cost_saved']['block_recourse_production_local']
            curtA = sum(h['res_curt_mw'] for k in A['hours'] for h in A['hours'][k]) / 9.0
            curtB = sum(h['res_curt_mw'] for k in B['hours'] for h in B['hours'][k]) / 9.0
            hrsA = sum(1 for k in A['hours'] for h in A['hours'][k] if h['res_curt_mw'] > 1e-3)
            hrsB = sum(1 for k in B['hours'] for h in B['hours'][k] if h['res_curt_mw'] > 1e-3)
            row = {'w': w, 'C_A': cA, 'C_B': cB, 'dC': cB - cA, 'cov': bd['cov_block'], 'D1': (cB - cA) - bd['cov_block'],
                   'V_price_valued_draw_change': V, 'D1prime': (cB - cA) - V, 'first_order_dC_from_A_duals': fo,
                   'obj_A': A['costs']['objective'], 'obj_B': B['costs']['objective'],
                   'saved_tso_generation_cost': saved_gen, 'saved_tso_block_local': saved_local,
                   'C_A_minus_saved_gen': cA - saved_gen, 'obj_A_minus_saved_local': A['costs']['objective'] - saved_local,
                   'res_curt_mwh_expected_A': curtA, 'res_curt_mwh_expected_B': curtB,
                   'curtailment_scenario_hours_A': hrsA, 'curtailment_scenario_hours_B': hrsB,
                   'detector_A': A['detector_components']['detector_penalty_total'],
                   'detector_B': B['detector_components']['detector_penalty_total'],
                   'slack_penalties_A': A['costs']['total_slack_penalties'],
                   'slack_penalties_B': B['costs']['total_slack_penalties'],
                   'max_slack_var_B': max((v['max'] for v in B['slack_var_extremes'].values() if v['max'] is not None),
                                          default=None),
                   'max_violation_A': A['max_violation']['max'], 'max_violation_B': B['max_violation']['max']}
            rows[bkey] = row
            tot['C_A_w'] += w * cA
            tot['C_B_w'] += w * cB
            tot['dC_w'] += w * (cB - cA)
            tot['cov_w'] += w * bd['cov_block']
            tot['D1_w'] += w * row['D1']
            tot['V_w'] += w * V
            tot['D1prime_w'] += w * row['D1prime']
            tot['first_order_dC_w'] += w * (fo or 0.0)
            tot['objA_w'] += w * row['obj_A']
            tot['objB_w'] += w * row['obj_B']
            tot['C_A_minus_saved_gen_w'] += w * row['C_A_minus_saved_gen']
            tot['C_A_minus_saved_local_w'] += w * row['obj_A_minus_saved_local']
        # where B differs most: top scenario-hours by |w * omega * (gen cost B - gen cost A)|
        diffs = []
        for bkey, blk in cc['blocks'].items():
            A, B = blk.get('A', {}), blk.get('B', {})
            if 'hours' not in A or 'hours' not in B:
                continue
            y, d = bkey.split('|')
            bd = bd_all[(y, d)]
            for k in A['hours']:
                for t in range(24):
                    ha, hb = A['hours'][k][t], B['hours'][k][t]
                    diffs.append({'block': bkey, 'scenario': k, 'hour': t,
                                  'w_omega_dgen_cost_eur': bd['w'] * bd['omega'][k] * (hb['gen_cost_eur'] - ha['gen_cost_eur']),
                                  'dgen_ctrl_mw': hb['gen_ctrl_mw'] - ha['gen_ctrl_mw'],
                                  'd_draw_mw': sum(hb['adn'][str(dn)]['pc_mw'] - ha['adn'][str(dn)]['pc_mw'] for dn in dn_ids),
                                  'res_curt_mw_A': ha['res_curt_mw'], 'res_curt_mw_B': hb['res_curt_mw'],
                                  'gens_at_ub_A': ha['gens_at_ub'], 'gens_at_ub_B': hb['gens_at_ub'],
                                  'branches_binding_A': ha['branches_binding'], 'branches_binding_B': hb['branches_binding'],
                                  'mc_p_A': {dn: ha['adn'][dn].get('mc_p_eur_per_mwh') for dn in ha['adn']},
                                  'mc_p_B': {dn: hb['adn'][dn].get('mc_p_eur_per_mwh') for dn in hb['adn']},
                                  'price': bd['prices'][t][int(k.split('_')[0])]})
        diffs.sort(key=lambda r: -abs(r['w_omega_dgen_cost_eur']))
        changed_binding = sum(1 for r in diffs if r['gens_at_ub_A'] != r['gens_at_ub_B']
                              or r['branches_binding_A'] != r['branches_binding_B'])
        out['per_cell'][lab] = {'blocks': rows, 'totals_weighted': tot, 'complete': complete,
                                'n_optimal': n_opt, 'top_scenario_hours_by_weighted_gen_cost_change': diffs[:25],
                                'n_scenario_hours': len(diffs), 'n_scenario_hours_binding_set_changed': changed_binding,
                                'n_scenario_hours_curtailment_A': sum(1 for r in diffs if r['res_curt_mw_A'] > 1e-3),
                                'n_scenario_hours_curtailment_B': sum(1 for r in diffs if r['res_curt_mw_B'] > 1e-3)}
    pc = out['per_cell']
    if 'x0' in pc and 'unit' in pc:
        out['pair_D1_x0_minus_unit'] = pc['x0']['totals_weighted']['D1_w'] - pc['unit']['totals_weighted']['D1_w']
    return out


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
