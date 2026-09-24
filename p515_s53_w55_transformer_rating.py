"""P5.15 W55 -- DN HV/MV interface-transformer rating in force (ZERO SOLVES, NO MODEL CONSTRUCTION).

Resolves W53 finding 4 (commit 51bd342d): "the DN7 transformer sits at 1.0 while importing about 99.6 MW, which
suggests the effective limit is not the 200 MVA in the case file".

Read-only. An armed SolveProfileGuard(permitted=()) is installed before Pyomo is imported and verified at exactly 0;
Network.build_model / NetworkData.build_model / SharedEnergyStorageData.build_subproblem / build_master_problem are
replaced by raising blockers (declared count 0). Nothing is built: the constraint is read from the PERSISTED certified
terminal models of s48_x0_capture (the models the certified SRP1 x = 0 record was computed from), and the Network
object is the one bound into the branch_flow_limit rule partial of each persisted model (so the rating reported is
the rating the constraint was built with, not a re-read of the case file).

Inputs (sha256 verified before analysis):
  * data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/certified_models.pkl (03b62593..., as recorded by
    W53 and the s48 child manifest);
  * data/SRP1/Results/P515S52/campaign_s52_pilot_nopersist/evals/{7d53b6f2..._x0,711fce9a..._n7_4h_e1}/results/
    SRP1_distributed_terminal.xlsx (sha256 as recorded in the W53 JSON);
  * data/SRP1/SRP1.json and data/SRP1/case33_{1,2,3}/case33_*_<year>.json (git-tracked, must be clean).

Formulas (preserved here, operationally):
  rating_pu            = branch.rate / network.baseMVA                         (model_construction_helpers.py:1632)
  constraint           flow_ij_sqr[k] <= rating_pu**2 + EQUALITY_TOLERANCE     (model_construction_helpers.py:1637)
  flow_ij_sqr[k]       = pij[k]**2 + qij[k]**2  for a transformer under MIXED   (model_construction_helpers.py:1355-1364)
  S_MVA                = sqrt(flow_ij_sqr) * baseMVA
  loading              = sqrt(flow_ij_sqr) / rating_pu                          (network.py:1525/1565 -> 'Flow_ij, [%]')
  at_rating            loading >= 1 - AT_RATING_TOL
  workbook consistency 'Flow_ij, [%]'(1->2)  vs  'S, [MVA]'(1->2) / case-file rating_MVA

Usage (canonical interpreter; attached; both streams captured):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w55_transformer_rating.py \
      > data/SRP1/Results/P515S53/w55_transformer_rating_launch.log 2>&1
OUTPUT (write-once, refuses to overwrite): data/SRP1/Results/P515S53/w55_transformer_rating/
  {transformer_rating.json, manifest_sha256.json}
"""
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W55 transformer rating (zero solves)').install()

import openpyxl  # noqa: E402
import pyomo.environ as pe  # noqa: E402
from pyomo.core.expr.visitor import identify_variables  # noqa: E402

from definitions import BRANCH_LIMIT_MIXED, EQUALITY_TOLERANCE  # noqa: E402

STAGE = 'P5.15 W55 -- DN interface-transformer rating in force (zero solves, no model construction)'
_R = os.path.join('data', 'SRP1', 'Results')
OUT_REL = os.path.join(_R, 'P515S53', 'w55_transformer_rating')
MODELS = {'path': os.path.join(_R, 'P515S48', 'x0_capture', 'evals', 'd2c96b1480402a3b_x0', 'certified_models.pkl'),
          'sha256': '03b62593a23f748c819f18dce52c88c6a9802b8af3c52033ae09a3b88d10afce',
          'instance': 'srp1_x0', 'eval_key': 'd2c96b1480402a3b61aca4abc188e41c6009eb582d8e6ccdd380e51651f996c7',
          'candidate_canonical': {'investment_year': 2025, 'nodes': {'5': [0.0, 0.0], '7': [0.0, 0.0],
                                                                     '9': [0.0, 0.0]}}}
WORKBOOKS = {
    'pilot_x0': {'path': os.path.join(_R, 'P515S52', 'campaign_s52_pilot_nopersist', 'evals', '7d53b6f21b686a44_x0',
                                      'results', 'SRP1_distributed_terminal.xlsx'),
                 'sha256': 'ecbb11f43ac62272a40dc1973b0bb3bc3ab914f89d91ff180d51df2b6b2148ee',
                 'candidate_canonical': {'investment_year': 2025, 'nodes': {'5': [0.0, 0.0], '7': [0.0, 0.0],
                                                                            '9': [0.0, 0.0]}}},
    'pilot_unit': {'path': os.path.join(_R, 'P515S52', 'campaign_s52_pilot_nopersist', 'evals',
                                        '711fce9aa74d6878_n7_4h_e1', 'results', 'SRP1_distributed_terminal.xlsx'),
                   'sha256': '1a28ceb214dc44c274d2d046ed87d85db8bbf6117dd464c0a9b5492a3afe499d',
                   'candidate_canonical': {'investment_year': 2025, 'nodes': {'5': [0.0, 0.0], '7': [0.25, 1.0],
                                                                              '9': [0.0, 0.0]}}}}
W53_JSON = os.path.join(_R, 'P515S53', 'curtailment_audit', 'curtailment_audit.json')
SRP1_JSON = os.path.join('data', 'SRP1', 'SRP1.json')
AT_RATING_TOL = 1e-3          # as W53's TRAFO_AT_RATING_TOL
BOUND_REL_TOL = 0.0           # the constructed upper bound must equal rating_pu**2 + EQUALITY_TOLERANCE exactly
WB_CONSISTENCY_TOL = 1e-9     # |Flow_ij - S_MVA / rating_MVA|, both derived from the same solution


class ConstructionBlockers:
    def __init__(self):
        import network as network_module
        import network_data as network_data_module
        import shared_energy_storage_data as sed
        self.targets = [(network_module.Network, 'build_model'), (network_data_module.NetworkData, 'build_model'),
                        (sed.SharedEnergyStorageData, 'build_subproblem'),
                        (sed.SharedEnergyStorageData, 'build_master_problem')]
        self.blocked_calls, self.originals = [], {}

    def install(self):
        def blocker(name):
            def _raise(*a, **k):
                self.blocked_calls.append(name)
                raise RuntimeError(f'W55: model construction blocked ({name})')
            return _raise
        for cls, attr in self.targets:
            self.originals[(cls, attr)] = getattr(cls, attr)
            setattr(cls, attr, blocker(f'{cls.__name__}.{attr}'))
        return self


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _tracked_clean(path):
    tracked = bool(_git(['ls-files', '--', path]).strip())
    clean = tracked and not _git(['status', '--porcelain', '--', path]).strip()
    return tracked, clean


def case_files():
    cfg = json.load(open(SRP1_JSON))
    out = {'SRP1.json': {'path': SRP1_JSON, 'sha256': _sha(SRP1_JSON), 'tracked_clean': _tracked_clean(SRP1_JSON)},
           'dn_mapping': {str(dn['connection_node_id']): dn['name'] for dn in cfg['DistributionNetworks']},
           'files': {}}
    for node, name in out['dn_mapping'].items():
        d = os.path.join('data', 'SRP1', name)
        for fn in sorted(os.listdir(d)):
            if fn.startswith(name + '_') and fn.endswith('.json') and fn[len(name) + 1:-5].isdigit():
                p = os.path.join(d, fn)
                data = json.load(open(p))
                out['files'][p] = {'node': node, 'year': int(fn[len(name) + 1:-5]), 'baseMVA': data['baseMVA'],
                                   'transformers': data['transformers'], 'sha256': _sha(p),
                                   'tracked_clean': _tracked_clean(p)}
    return out


def analyse_models(cases):
    assert _sha(MODELS['path']) == MODELS['sha256'], 'certified_models.pkl sha256 mismatch'
    with open(MODELS['path'], 'rb') as h:
        payload = pickle.load(h)
    blocks, hours, checks = [], [], []
    for node in sorted(payload['dso']):
        for y in sorted(payload['dso'][node]):
            for day, m in payload['dso'][node][y].items():
                fcn = m.branch_flow_limit._rule._fcn
                net, params = fcn.keywords['network'], fcn.keywords['params']
                fcn_ji = m.branch_flow_limit_ji._rule._fcn
                (s_m, s_o), = [(a, b) for a in m.scenarios_market for b in m.scenarios_operation]
                trafos = [i for i, b in enumerate(net.branches) if b.is_transformer]
                assert len(trafos) == 1, (node, y, day, trafos)
                k = trafos[0]
                br = net.branches[k]
                case_path = os.path.join('data', 'SRP1', net.name, f'{net.name}_{y}.json')
                case_tr = cases['files'][case_path]['transformers']
                rating_pu = br.rate / net.baseMVA
                expected_upper = rating_pu ** 2 + EQUALITY_TOLERANCE
                body_vars = {v.parent_component().local_name for v in identify_variables(m.flow_ij_sqr[k, s_m, s_o, 0].expr)}
                b = {'node': node, 'year': y, 'day': day, 'network_name': net.name, 'network_year': net.year,
                     'network_day': net.day, 'case_file': case_path, 'case_file_transformers': case_tr,
                     'network_baseMVA': net.baseMVA, 'branch_index': k, 'branch_id': br.branch_id,
                     'fbus': br.fbus, 'tbus': br.tbus, 'branch_rate_MVA': br.rate, 'ratio': br.ratio,
                     'branch_limit_type': params.branch_limit_type,
                     'apparent_power_limited_branches': list(m.apparent_power_limited_branches),
                     'n_branches': len(net.branches), 'rating_pu': rating_pu,
                     'constraint_upper_expected': expected_upper,
                     'flow_ij_sqr_expr': str(m.flow_ij_sqr[k, s_m, s_o, 0].expr), 'flow_ij_sqr_vars': sorted(body_vars),
                     'ji_rule_same_network_object': fcn_ji.keywords['network'] is net}
                uppers, lowers, s_max, n_at = set(), set(), 0.0, 0
                for t in m.periods:
                    con = m.branch_flow_limit[k, s_m, s_o, t]
                    con_ji = m.branch_flow_limit_ji[k, s_m, s_o, t]
                    uppers.update([pe.value(con.upper), pe.value(con_ji.upper)])
                    lowers.update([None if c.lower is None else pe.value(c.lower) for c in (con, con_ji)])
                    body = pe.value(con.body)
                    pij, qij = pe.value(m.pij[k, s_m, s_o, t]), pe.value(m.qij[k, s_m, s_o, t])
                    s_mva = math.sqrt(max(body, 0.0)) * net.baseMVA
                    loading = math.sqrt(max(body, 0.0)) / rating_pu
                    s_ji = math.sqrt(max(pe.value(con_ji.body), 0.0)) * net.baseMVA
                    s_max = max(s_max, s_mva)
                    if loading >= 1.0 - AT_RATING_TOL:
                        n_at += 1
                        hours.append({'node': node, 'year': y, 'day': day, 'hour': t, 'S_ij_MVA': s_mva,
                                      'P_ij_MW': pij * net.baseMVA, 'Q_ij_Mvar': qij * net.baseMVA,
                                      'S_ji_MVA': s_ji, 'loading': loading,
                                      'pg_adn_MW': pe.value(m.pg_adn[s_m, s_o, t]) * net.baseMVA,
                                      'upper_pu2': con.upper, 'body_pu2': body,
                                      'dual_raw': m.dual.get(con) if hasattr(m, 'dual') else None,
                                      'dual_ji_raw': m.dual.get(con_ji) if hasattr(m, 'dual') else None})
                b.update({'constraint_uppers_seen': sorted(uppers), 'constraint_lowers_seen': [str(x) for x in lowers],
                          'max_S_ij_MVA': s_max, 'hours_at_rating': n_at})
                blocks.append(b)
                checks.append({
                    'block': f'{node}|{y}|{day}',
                    'network_is_case_of_node': net.name == cases['dn_mapping'][str(node)],
                    'network_year_day_match': (int(net.year) == int(y) and net.day == day),
                    'branch_is_case_transformer': (len(case_tr) == 1 and br.fbus == case_tr[0]['fbus']
                                                   and br.tbus == case_tr[0]['tbus']),
                    'rate_equals_case_rating': br.rate == float(case_tr[0]['rating']),
                    'baseMVA_equals_case': net.baseMVA == float(cases['files'][case_path]['baseMVA']),
                    'mixed_and_only_trafo_apparent': (params.branch_limit_type == BRANCH_LIMIT_MIXED
                                                      and list(m.apparent_power_limited_branches) == [k]),
                    'body_is_pij2_plus_qij2': body_vars == {'pij', 'qij'},
                    'upper_exact_both_directions': uppers == {expected_upper},
                    'no_lower': lowers == {None},
                    'ji_same_network': b['ji_rule_same_network_object']})
    return blocks, hours, checks


def cross_check_w53(hours):
    w53 = json.load(open(W53_JSON))
    at = {(h['node'], h['year'], h['day'], h['hour']) for h in hours}
    rows, keys = set(), set()
    for e in w53['srp1_x0']['entries_above_tol']:
        for r in e.get('active_rows_network_hour', {}).get('branch', []):
            rows.add((e['network'], r['row'], r['is_transformer']))
            if r['is_transformer']:
                keys.add((int(e['network'][3:]), int(e['year']), e['day'], int(e['hour'])))
    return {'w53_sha256': _sha(W53_JSON), 'w53_distinct_active_branch_rows': sorted(map(list, rows)),
            'w53_transformer_active_network_hours': len(keys),
            'w53_transformer_active_hours_all_at_rating_here': keys <= at,
            'w53_hours_not_at_rating_here': sorted(map(list, keys - at))}


def analyse_workbook(label, spec, rating_by_year):
    assert _sha(spec['path']) == spec['sha256'], f'{label} workbook sha256 mismatch'
    wb = openpyxl.load_workbook(spec['path'], read_only=True, data_only=True)
    flow, pf = {}, {}
    for sheet, store, labels in (('Branch Loading', flow, ('Flow_ij, [%]',)),
                                 ('Power Flows', pf, ('P, [MW]', 'Q, [MVAr]', 'S, [MVA]'))):
        for row in wb[sheet].iter_rows(min_row=2, values_only=True):
            if (row[0] == 'DSO' and str(row[1]) == '7' and row[3] == 1 and row[4] == 2 and row[7] in labels
                    and row[8] != 'Expected'):
                store[(row[7], int(row[5]), row[6], str(row[8]), str(row[9]))] = list(row[10:34])
    wb.close()
    diffs, n_at, max_flow, max_s, examples = [], 0, 0.0, 0.0, []
    for (lab, y, d, sm, so), vals in flow.items():
        svals = pf[('S, [MVA]', y, d, sm, so)]
        pvals, qvals = pf[('P, [MW]', y, d, sm, so)], pf[('Q, [MVAr]', y, d, sm, so)]
        r = rating_by_year[y]
        for t, v in enumerate(vals):
            diffs.append(abs(v - svals[t] / r))
            max_flow, max_s = max(max_flow, v), max(max_s, svals[t])
            if v >= 1.0 - AT_RATING_TOL:
                n_at += 1
                if len(examples) < 12:
                    examples.append({'year': y, 'day': d, 's_m': sm, 's_o': so, 'hour': t, 'Flow_ij': v,
                                     'S_MVA': svals[t], 'P_MW': pvals[t], 'Q_Mvar': qvals[t]})
    return {'path': spec['path'], 'sha256': spec['sha256'], 'candidate_canonical': spec['candidate_canonical'],
            'rating_MVA_by_year_from_case_file': rating_by_year, 'n_rows_flow_ij': len(flow),
            'n_cells': len(diffs), 'max_abs_Flow_ij_minus_S_over_case_rating': max(diffs),
            'max_Flow_ij': max_flow, 'max_S_ij_MVA': max_s, 'cells_at_rating': n_at, 'examples_at_rating': examples}


def main():
    t0 = time.time()
    out_dir = os.path.join(REPO, OUT_REL)
    if os.path.exists(out_dir):
        raise SystemExit(f'refusing to overwrite existing output {OUT_REL}')
    blockers = ConstructionBlockers().install()
    cases = case_files()
    blocks, hours, checks = analyse_models(cases)
    xw53 = cross_check_w53(hours)
    rating7 = {v['year']: float(v['transformers'][0]['rating']) for p, v in cases['files'].items() if v['node'] == '7'}
    wbs = {k: analyse_workbook(k, s, rating7) for k, s in WORKBOOKS.items()}
    guard_fail = GUARD.verify(0)
    all_checks = {
        'case_files_tracked_clean': all(v['tracked_clean'] == (True, True) for v in cases['files'].values())
                                    and cases['SRP1.json']['tracked_clean'] == (True, True),
        'per_block_checks_all_true': all(all(v for k, v in c.items() if k != 'block') for c in checks),
        'w53_transformer_hours_all_at_rating_here': xw53['w53_transformer_active_hours_all_at_rating_here'],
        'workbook_Flow_ij_equals_S_over_case_rating': all(w['max_abs_Flow_ij_minus_S_over_case_rating']
                                                          <= WB_CONSISTENCY_TOL for w in wbs.values()),
        'no_model_construction_call': not blockers.blocked_calls,
        'solve_profile_guard_verified_0': not guard_fail}
    summary = {}
    for b in blocks:
        s = summary.setdefault(str(b['node']), {'network_name': b['network_name'], 'rates_MVA': set(),
                                                 'baseMVA': set(), 'max_S_ij_MVA': 0.0, 'hours_at_rating': 0})
        s['rates_MVA'].add(b['branch_rate_MVA']); s['baseMVA'].add(b['network_baseMVA'])
        s['max_S_ij_MVA'] = max(s['max_S_ij_MVA'], b['max_S_ij_MVA']); s['hours_at_rating'] += b['hours_at_rating']
    for s in summary.values():
        s['rates_MVA'], s['baseMVA'] = sorted(s['rates_MVA']), sorted(s['baseMVA'])
    res = {'stage': STAGE, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
           'git_head': _git(['rev-parse', 'HEAD']).strip(), 'script': os.path.basename(__file__),
           'script_sha256': _sha(__file__), 'script_git_status': _git(['status', '--porcelain', '--', __file__]).strip(),
           'interpreter': sys.executable, 'EQUALITY_TOLERANCE': EQUALITY_TOLERANCE, 'AT_RATING_TOL': AT_RATING_TOL,
           'models_input': MODELS, 'case_files': cases, 'summary_by_dn_node': summary,
           'blocks': blocks, 'at_rating_hours_srp1_x0': hours, 'block_checks': checks, 'w53_cross_check': xw53,
           'pilot_workbooks_dso7_branch_1_2': wbs, 'checks': all_checks, 'all_checks_pass': all(all_checks.values()),
           'model_construction_blocked_calls': blockers.blocked_calls,
           'solve_profile_guard': {'counts': GUARD.counts, 'verify_0_failures': guard_fail},
           'wall_clock_s': time.time() - t0}
    os.makedirs(out_dir)
    jp = os.path.join(out_dir, 'transformer_rating.json')
    with open(jp, 'w') as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'w') as f:
        json.dump({'transformer_rating.json': _sha(jp)}, f, indent=1)
    print(json.dumps({'summary_by_dn_node': summary, 'checks': all_checks, 'w53_cross_check': {
        k: v for k, v in xw53.items() if k != 'w53_hours_not_at_rating_here'},
        'pilot': {k: {kk: w[kk] for kk in ('max_Flow_ij', 'max_S_ij_MVA', 'cells_at_rating',
                                             'max_abs_Flow_ij_minus_S_over_case_rating')} for k, w in wbs.items()},
        'guard': GUARD.counts, 'guard_verify_0_failures': guard_fail}, indent=1, default=str))
    if not res['all_checks_pass']:
        raise SystemExit('W55: a check failed')


if __name__ == '__main__':
    main()
