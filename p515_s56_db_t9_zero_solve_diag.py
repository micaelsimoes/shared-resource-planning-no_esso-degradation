"""P5.15 S56 -- ZERO-SOLVE diagnostics after the D_b re-solve (spec 04dd2bbe) and the T9 attribution (spec c6c244c7).

Written after the runs, after viewing their results: exploratory, no recorded predictions. Never solves
(SolveProfileGuard(permitted=()) installed before any project import, verified at 0); pickle.load/loads blocked.

Part A -- why every Arm B (realized DN exchanges) TN re-solve ended "Converged to a point of local infeasibility".
  Reads the committed run record (db_tn_resolve_results.json) and its IPOPT logs, and rebuilds the Arm B TN models
  of both instances with the same production builder (p515_s56_tn_realized_resolve_v2.build_arm_v2, zero solves).
  Records, per block:
    (a) per scenario-hour the total realized DN exchange sum_DN P_DSO[s, t] (MW; > 0 = the DNs draw from the TN) and the
        TSO scenario-common total sum_DN P_TSO[t]; the scenario-hours with a net DN EXPORT (total < 0);
    (b) the TN sink inventory of the built model: every bus carrying a load and whether it is an ADN bus; the lower
        bound of every TN generator's pg (MW) in every scenario-period; the free slack Var components that enter the
        node balance (none unless params.slacks.node_balance); node shunt conductances gs; branch r, x, g_sh, rate;
        params.branch_limit_type;
    (c) a loss upper bound U = sum_branches r_b (rate_b / v_min)^2 x baseMVA (MW): with every bus voltage >= v_min and
        every branch apparent flow <= rate, the series current is <= rate / v_min, so the series loss of a branch is <=
        r (rate / v_min)^2 (line-charging currents ignored: an indication, not a proof);
    (d) the IPOPT terminal block of each Arm B log (unscaled constraint violation, exit message).
  With no TN load outside the ADN buses and every generator bounded below by 0, a net DN export can leave the TN only
  as series / shunt losses; (c) says whether losses can in principle absorb it.

Part B -- the T9 attribution: the model difference between the v2 build (cohort follow-up 'before' tree) and the
  after build, for the plan of t_ii_b and t_iv_peak (srp1 p2025_2030, independent_units, node 7), read from the
  committed symbolic NL row/col lists of the cohort follow-up T1 run (init and admm stages): the row and column names
  present in one build only (by family), and whether the common rows and columns appear in the same order.

Usage (from a clean export root, as the stage scripts):
  LP_SOLVER_PATH=... NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s56_db_t9_zero_solve_diag.py \\
      --db-run DIR --out-dir NEW_DIR
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 D_b / T9 zero-solve diagnostics (never solves)').install()

import pickle  # noqa: E402

PK = {'load': 0, 'loads': 0}


def _blk(kind):
    def f(*_a, **_k):
        PK[kind] += 1
        raise RuntimeError(f'pickle.{kind} blocked')
    return f


pickle.load, pickle.loads = _blk('load'), _blk('loads')

import argparse  # noqa: E402
import collections  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s56_db_tn_resolve as DB  # noqa: E402  (installs its own guard: removed right after import)

DB.GUARD.uninstall()
pickle.load, pickle.loads = _blk('load'), _blk('loads')
H, V2 = DB.H, DB.V2

import pyomo.environ as pe  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

NL = 'data/SRP1/Results/P515S56/cohort_followup_v1/run/nl/{tree}/esso/srp1__p2025_2030__inobj0__indep/' \
     'srp1__p2025_2030__inobj0__indep__{stage}__n7.sym.{kind}.gz'


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ipopt_terminal(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    out = {'path': path, 'sha256': _sha(path)}
    m = re.search(r'Constraint violation\.+:\s+(\S+)\s+(\S+)', txt.split('Number of Iterations....:')[-1])
    out['constraint_violation_scaled_unscaled'] = [float(m.group(1)), float(m.group(2))] if m else None
    m = re.search(r'EXIT: (.*)', txt)
    out['exit'] = m.group(1).strip() if m else None
    m = re.search(r'Number of Iterations\.+: (\d+)', txt)
    out['iterations'] = int(m.group(1)) if m else None
    return out


def part_a(db_run, out_dir):
    rec = json.load(open(os.path.join(db_run, 'db_tn_resolve_results.json')))
    spec = json.load(open(rec['spec']['path']))
    out = {'run_record': {'path': os.path.join(db_run, 'db_tn_resolve_results.json'),
                          'sha256': _sha(os.path.join(db_run, 'db_tn_resolve_results.json'))}, 'blocks': {}}
    for inst in spec['instances']:
        lab = inst['label']
        ic = rec['instances'][lab]
        cd = DB.load_arm(inst)
        planning = H.S.read_planning_from_derived_case({'derived_case': {'path': inst['case_path']}},
                                                       os.path.join(out_dir, f'read_{lab}'), H._Stages())
        tn = planning.transmission_network
        dn_ids = list(tn.active_distribution_network_nodes)
        cand = planning.get_initial_candidate_solution()
        srp._rebuild_candidate_total_capacities(planning, cand)
        tn.update_data_with_candidate_solution(cand['total_capacity'])
        avail = {n: {y: {'s': ic['available_capacity_applied'][str(n)][str(y)]['s'],
                         'e': ic['available_capacity_applied'][str(n)][str(y)]['e']} for y in tn.years} for n in dn_ids}
        bdata = {(str(int(y)), d): H.block_data(lab, cd['ms'], tn, y, d, dn_ids) for y in tn.years for d in tn.days}
        models, _recs, _iso = V2.build_arm_v2(tn, cand, avail, lab, 'B', bdata, dn_ids)
        for y in tn.years:
            for d in tn.days:
                bk = f'{int(y)}|{d}'
                m, net = models[y][d], tn.network[y][d]
                iu = ic['blocks'][bk]['inputs_used']
                tso, dso = iu['tso_p_common_mw'], iu['dso_p_realized_mw']
                tot = {k: [sum(dso[str(dn)][k][t] for dn in dn_ids) for t in range(24)] for k in iu['omega']}
                tso_tot = [sum(tso[str(dn)][t] for dn in dn_ids) for t in range(24)]
                export = sorted(((k, t, v) for k, row in tot.items() for t, v in enumerate(row) if v < 0.0),
                                key=lambda x: x[2])
                base = net.baseMVA
                load_buses = sorted({int(ld.bus) for ld in net.loads})
                pg_lb = [m.pg[g, s_m, s_o, p].lb for g in m.generators for s_m in m.scenarios_market
                         for s_o in m.scenarios_operation for p in m.periods if not m.pg[g, s_m, s_o, p].fixed]
                slack_free = {}
                for v in m.component_objects(pe.Var):
                    if v.name.startswith('slack'):
                        free = [x for x in v.values() if not x.fixed]
                        slack_free[v.name] = {'n': len(list(v.values())), 'n_free': len(free),
                                              'ub_set': sorted({x.ub for x in free if x.ub is not None})[:5]}
                vmin = min(nd.v_min for nd in net.nodes)
                br = [{'f': b.fbus, 't': b.tbus, 'r': b.r, 'x': b.x, 'g_sh': b.g_sh, 'b_sh': b.b_sh, 'rate_mva': b.rate,
                       'status': bool(b.status),
                       'loss_bound_mw': b.r * (b.rate / base / vmin) ** 2 * base if b.status else 0.0} for b in net.branches]
                arm_b_log = os.path.join(db_run, 'logs', lab, 'B', f'optim_log_case9_{int(y)}_{d}.log')
                out['blocks'][f'{lab}|{bk}'] = {
                    'min_total_realized_dn_exchange_mw': min(min(r) for r in tot.values()),
                    'min_total_tso_common_exchange_mw': min(tso_tot),
                    'n_scenario_hours_net_dn_export': len(export),
                    'scenarios_with_net_export': sorted({k for k, _t, _v in export}),
                    'net_export_scenario_hours_mw': [[k, t, v] for k, t, v in export],
                    'tn_load_buses': load_buses, 'adn_buses': dn_ids,
                    'all_tn_load_buses_are_adn': set(load_buses) <= set(dn_ids),
                    'tn_generator_pg_lower_bound_max_mw': max(pg_lb) * base if pg_lb else None,
                    'tn_generator_pg_lower_bound_min_mw': min(pg_lb) * base if pg_lb else None,
                    'slack_vars': slack_free,
                    'node_balance_slacks_configured': bool(tn.params.slacks.node_balance.active_power),
                    'node_shunt_gs_max': max(abs(nd.gs) for nd in net.nodes),
                    'branch_limit_type': str(tn.params.branch_limit_type), 'v_min': vmin, 'branches': br,
                    'loss_upper_bound_mw': sum(b['loss_bound_mw'] for b in br),
                    'arm_B_ipopt_terminal': ipopt_terminal(arm_b_log) if os.path.exists(arm_b_log) else None}
    return out


def part_b():
    fam = lambda n: re.sub(r'\[.*', '', n)  # noqa: E731
    out = {}
    for stage in ('init', 'admm'):
        for kind in ('row', 'col'):
            pb, pa = NL.format(tree='before', stage=stage, kind=kind), NL.format(tree='after', stage=stage, kind=kind)
            b = gzip.open(pb, 'rt').read().split()
            a = gzip.open(pa, 'rt').read().split()
            sb, sa = set(b), set(a)
            out[f'{stage}_{kind}'] = {
                'before': {'path': pb, 'sha256': _sha(pb), 'n': len(b)}, 'after': {'path': pa, 'sha256': _sha(pa), 'n': len(a)},
                'only_before_by_family': dict(collections.Counter(fam(x) for x in sb - sa)),
                'only_after_by_family': dict(collections.Counter(fam(x) for x in sa - sb)),
                'common': len(sb & sa),
                'common_same_order': [x for x in b if x in sa] == [x for x in a if x in sb]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db-run', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out_dir):
        print(f'REFUSE: {a.out_dir} exists', flush=True)
        return 1
    os.makedirs(a.out_dir)
    res = {'stage': 'P5.15 S56 D_b / T9 zero-solve diagnostics (exploratory, after the runs)', 'script':
           os.path.basename(__file__), 'script_sha256': _sha(os.path.abspath(__file__)),
           'utc_start': datetime.now(timezone.utc).isoformat(), 'run_directory': HERE,
           'provenance_file_sha256': _sha('EXPORT_PROVENANCE.json') if os.path.exists('EXPORT_PROVENANCE.json') else None}
    for name, fn in (('part_a_db_infeasibility', lambda: part_a(a.db_run, a.out_dir)), ('part_b_t9_model_diff', part_b)):
        try:
            res[name] = fn()
        except Exception:  # noqa: BLE001
            res[f'{name}_error'] = traceback.format_exc()
            print(res[f'{name}_error'], flush=True)
    gv = GUARD.verify(0)
    res['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': gv, 'db_module_guard_counts': dict(DB.GUARD.counts)}
    res['pickle'] = dict(PK)
    res['ok'] = (not gv) and PK == {'load': 0, 'loads': 0} and not any(k.endswith('_error') for k in res)
    path = os.path.join(a.out_dir, 'db_t9_zero_solve_diag.json')
    with open(path, 'x', encoding='utf-8') as h:
        GRIO.dump(res, h, indent=1, default=GRIO.json_default_item)
    with open(os.path.join(a.out_dir, 'manifest_sha256.json'), 'x', encoding='utf-8') as h:
        GRIO.dump({'files': {os.path.relpath(os.path.join(dp, f), a.out_dir): _sha(os.path.join(dp, f))
                             for dp, _d, fs in os.walk(a.out_dir) for f in fs}}, h, indent=1)
    print(f"wrote {path}; ok {res['ok']}", flush=True)
    return 0 if res['ok'] else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
