"""
P5.15 Addendum 28 (task W19) -- zero-solve reports Z1-Z4. ZERO SOLVES.

Armed `SolveProfileGuard(permitted=())` is installed before any model import and
`verify(0)` is checked at the end (exact count: 0 solves, 0 solver launches).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 28 (W19); frozen spec v16
`data/SRP1/Results/P515S46/frozen_s46_ageing_spec_v16_f4295086.json`, section
`zero_solve_items`.

INSTANCE. Every figure refers to the candidate
    node 7 at 0.25 MVA / 1.0 MWh, investment year 2025, nodes 5 and 9 empty
    candidate_key db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a
(recomputed here with `p515_s44_campaign_harness.canonical_candidate/candidate_key`
and checked against the records), and to x = 0 (the a0_c7:x0 evaluation) where
Q(0) is needed. Both are committed, certified Phase A evaluations run under the
frozen oracle (case file, AA keep_memory, cap 500).

WHAT EACH ITEM COMPUTES (the formulas are also written into the JSON / Markdown)
  Z1  Branch endpoints. From the SRP1 data read by production
      (`p56a_oracle.load_baseline`) and from block models BUILT (never solved) by
      production (`Network.build_model`): the branches `network.py:
      get_interface_branch_rating` sums, the rows that limit them, where the
      shared storage enters the DSO and the TSO models; plus the terminal
      interface flows of the certified record to show which row is at its bound.
  Z2  Ageing trajectory of the node-7 unit, from the committed A1a record and its
      terminal SoH-floor sidecar; phi_cal and k read from the production object;
      the SoH chain recomputed from EFC/day with production's law and compared.
  Z3  Discount re-weighting of V = Q(0) - Q(x) per representative year, from the
      per-block component levels of the two certified records.
  Z4  Price spread of the market prices the models actually use, and the
      captured spread implied by the value per MWh and the EFC of Z2.

Output (write-once, new directory):
  data/SRP1/Results/P515S46/zero_solve_reports/
      zero_solve_reports.json, zero_solve_reports.md, launch.log, manifest_sha256.json

Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S46/zero_solve_reports
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s46_zero_solve_reports.py \\
        > data/SRP1/Results/P515S46/zero_solve_reports/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s46_zero_solve_reports.py --manifest
"""

import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)   # p56a_oracle.load_baseline reads 'data/SRP1' relative to the repository

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W19 zero-solve reports Z1-Z4').install()

import p515_s44_campaign_harness as H  # noqa: E402

STAGE = 'P5.15 Addendum 28 W19 -- zero-solve reports Z1-Z4 (zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 28 (W19)',
             'data/SRP1/Results/P515S46/frozen_s46_ageing_spec_v16_f4295086.json (zero_solve_items)']
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S46', 'zero_solve_reports')
OUT_ROOT = os.path.join(REPO, OUT_REL)
RESULTS_NAME = 'zero_solve_reports.json'
MD_NAME = 'zero_solve_reports.md'
MANIFEST_NAME = 'manifest_sha256.json'

SPEC_REL = 'data/SRP1/Results/P515S46/frozen_s46_ageing_spec_v16_f4295086.json'
A1A_REL = 'data/SRP1/Results/P515S45/campaign_s45_a1a/evals/7eb1ce62c2509f54_n7_4h_e1'
A0X_REL = 'data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7eb1ce62c2509f54_n7_p0_25_e1_0'
X0_REL = 'data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0'
PHASE_A_TABLES_REL = 'data/SRP1/Results/P515S45/phase_a_tables/phase_a_tables.json'
PHASE_A_IDENTITY = 'a0_c7:n7_p0.25_e1.0'
CASE_REL = 'data/SRP1/SRP1.json'
MARKET_REL = 'data/SRP1/MarketData/SRP1_market_data.xlsx'
ESS_PARAMS_REL = 'data/SRP1/SharedESS/SRP1_ESS_Params.json'

NODE = 7
S_MVA, E_MWH = 0.25, 1.0
EXPECTED_KEY = 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a'
RATES = (0.0, 0.02, 0.05, 0.08)
PRODUCTION_RATE = 0.02
EXPERT_PREDICTION_Z3 = {'0.0': '+ at most ~10 %', '0.05': '~ -12 %', '0.08': '~ -21 %'}
EXPERT_PREDICTION_Z4 = 'captured ~55-60 EUR/MWh per full cycle'
# Z3 reconstruction tolerance (stated before the run): the r = 2 % per-block sum must
# reproduce the certified value 261,807.50 to within 1.0 EUR (double-precision sums of
# 48 blocks of ~1e7 EUR carry ~1e-7 EUR of rounding; 1 EUR is a generous, stated bound).
Z3_RECONSTRUCTION_TOL_EUR = 1.0
# Z1: "at rating" utilization threshold (the Addendum-21/22 convention, >= 99 %)
AT_RATING = 0.99
# Z1: a slot is 'on the row bound' when |S| is within this of sqrt(1 + 1e-5) x 100 MVA (stated before the run)
ON_BOUND_TOL_MVA = 1e-3
DURATION_H = 4        # Z4 top-h / bottom-h window: the unit's E/P = 1.0 / 0.25 = 4 h

# gross (settlement-excluded) per-block cost = the objective components of
# `objective_function_rule` as captured by `_s31_block_components`, the settlement
# excluded; the reconciliation against recourse_components.gross_operational_cost is CHECKED.
GROSS_COMPONENTS = ('generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost',
                    'res_curtailment_penalty', 'ess_usage_cost', 'detector_penalty_total')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _last_jsonl(rel):
    last = None
    with open(os.path.join(REPO, rel)) as handle:
        for line in handle:
            if line.strip():
                last = line
    return json.loads(last)


def _lines(rel, needle):
    """1-based line numbers of every line of `rel` containing `needle` (citation helper)."""
    with open(os.path.join(REPO, rel)) as handle:
        return [i + 1 for i, line in enumerate(handle) if needle in line]


def _one_line(rel, needle):
    found = _lines(rel, needle)
    if len(found) != 1:
        raise RuntimeError(f'citation needle not unique in {rel}: {needle!r} -> {found}')
    return f'{rel}:{found[0]}'


def _fmt(x, nd=2):
    return f'{x:,.{nd}f}'


# ======================================================================================================================
#  instance
# ======================================================================================================================
def instance_section():
    cand_map = {n: ((S_MVA, E_MWH) if n == NODE else (0.0, 0.0)) for n in H.ACTIVE_NODES}
    canon = H.canonical_candidate(cand_map)
    key = H.candidate_key(canon)
    x0_canon = H.canonical_candidate({n: (0.0, 0.0) for n in H.ACTIVE_NODES})
    x0_key = H.candidate_key(x0_canon)
    recs = {name: _load(os.path.join(rel, 'evaluation_record.json'))
            for name, rel in (('a1a', A1A_REL), ('a0_c7', A0X_REL), ('x0', X0_REL))}
    checks = {
        'candidate_key_recomputed_equals_expected': key == EXPECTED_KEY,
        'a1a_record_key_matches': recs['a1a']['candidate_key'] == key,
        'a0_c7_record_key_matches': recs['a0_c7']['candidate_key'] == key,
        'x0_record_key_matches': recs['x0']['candidate_key'] == x0_key,
        'all_three_records_certified': all(r['status'] == 'certified' for r in recs.values()),
    }
    info = {
        'candidate_canonical': canon, 'candidate_key': key,
        'x0_canonical': x0_canon, 'x0_candidate_key': x0_key,
        'records': {name: {'path': rel, 'campaign_id': recs[name]['campaign_id'],
                           'candidate_label': recs[name]['candidate_label'],
                           'candidate_key': recs[name]['candidate_key'],
                           'eval_key': recs[name]['eval_key'], 'status': recs[name]['status'],
                           'certified_cost_gross_eur': recs[name]['certified_cost'],
                           'bar_eur': recs[name]['bar']['value'],
                           'terminal_objective_change_abs': recs[name]['rule_ten']['terminal_objective_change_abs'],
                           'terminal_step_over_threshold': recs[name]['rule_ten']['terminal_step_over_threshold']}
                    for name, rel in (('a1a', A1A_REL), ('a0_c7', A0X_REL), ('x0', X0_REL))},
    }
    return checks, info, recs


# ======================================================================================================================
#  Z1 -- branch endpoints
# ======================================================================================================================
def z1(planning):
    from pyomo.core.expr.visitor import identify_variables
    import pyomo.environ as pe
    import definitions as D
    checks, out = {}, {}
    dn = planning.distribution_networks[NODE]
    tn = planning.transmission_network

    cite = {
        'get_interface_branch_rating': _one_line('network.py', 'def get_interface_branch_rating(self):'),
        'selection_rule': _one_line('network.py', 'if branch.fbus == ref_node_id or branch.tbus == ref_node_id:'),
        'rating_sum': _one_line('network.py', 'interface_branch_rating += branch.rate'),
        'branch_flow_limit_rule': _one_line('model_construction_helpers.py', 'def branch_flow_limit_rule('),
        'branch_flow_limit_ji_rule': _one_line('model_construction_helpers.py', 'def branch_flow_limit_ji_rule('),
        'apparent_power_limit_for_transformers_under_MIXED': _one_line(
            'model_construction_helpers.py', 'def branch_uses_apparent_power_limit('),
        'branch_flow_limit_row_wired': _one_line('network.py', 'model.branch_flow_limit = pe.Constraint('),
        'branch_flow_limit_ji_row_wired': _one_line('network.py', 'model.branch_flow_limit_ji = pe.Constraint('),
        'dso_interface_p_def_storage_at_reference_bus': _one_line(
            'model_construction_helpers.py', 'def interface_pf_p_distribution_def('),
        'dso_interface_p_returns_pg_minus_storage': _one_line('model_construction_helpers.py',
                                                              'return m.pg[ref_gen_idx, s_m, s_o, p] - shared_ess_p'),
        'shared_storage_in_node_balance': _one_line('model_construction_helpers.py',
                                                    'Pd += model.shared_es_pnet[e, s_m, s_o, p]'),
        'tso_adn_interface_load_def': _one_line('model_construction_helpers.py',
                                                'def interface_pf_p_transmission_def('),
        'get_interface_branch_rating_uses_in_srp': [
            f'shared_resources_planning.py:{n}' for n in _lines('shared_resources_planning.py',
                                                               'get_interface_branch_rating()')],
    }
    out['citations'] = cite
    out['selection_rule_text'] = (
        'network.py get_interface_branch_rating: DSO only; ref = get_reference_node_id(); sums branch.rate over '
        'every IN-SERVICE branch (branch.status) whose fbus OR tbus equals the reference bus. It is used by '
        'production ONLY as the per-unit normalization of the PF consensus residual / augmented-Lagrangian terms '
        '(the shared_resources_planning.py sites listed) -- it is not itself a constraint row.')

    # ---- data: every (year, day) of the node-7 DSO ----
    per_block, all_single = {}, True
    ref_ids, branch_sets = set(), set()
    for year in dn.years:
        for day in dn.days:
            net = dn.network[year][day]
            ref = net.get_reference_node_id()
            adj = [(b, br) for b, br in enumerate(net.branches)
                   if br.status and (br.fbus == ref or br.tbus == ref)]
            ref_idx = net.get_node_idx(ref)
            loads_at_ref = [c for c, ld in enumerate(net.loads) if ld.bus == ref]
            gens_at_ref = [(g, gen.bus, gen.pmin, gen.pmax, gen.qmin, gen.qmax)
                           for g, gen in enumerate(net.generators) if gen.bus == ref]
            sess_buses = [e.bus for e in net.shared_energy_storages]
            per_block[f'{year}|{day}'] = {
                'reference_bus': ref, 'baseMVA': net.baseMVA,
                'adjacent_in_service_branches': [
                    {'branch_index': b, 'branch_id': br.branch_id, 'from_bus': br.fbus, 'to_bus': br.tbus,
                     'rate_mva': br.rate, 'is_transformer': br.is_transformer} for b, br in adj],
                'get_interface_branch_rating_mva': net.get_interface_branch_rating(),
                'loads_at_reference_bus': loads_at_ref,
                'reference_bus_gs_bs': [net.nodes[ref_idx].gs, net.nodes[ref_idx].bs],
                'generators_at_reference_bus_pu': gens_at_ref,
                'shared_storage_buses': sess_buses,
            }
            all_single = all_single and len(adj) == 1
            ref_ids.add(ref)
            branch_sets.add(tuple((br.fbus, br.tbus, br.rate) for _b, br in adj))
    out['dso_node7_data_per_block'] = per_block
    checks['z1_single_adjacent_branch_every_block'] = all_single
    checks['z1_same_reference_bus_and_branch_every_block'] = len(ref_ids) == 1 and len(branch_sets) == 1
    checks['z1_rating_equals_that_branch_rate_every_block'] = all(
        v['get_interface_branch_rating_mva'] == sum(a['rate_mva'] for a in v['adjacent_in_service_branches'])
        for v in per_block.values())
    checks['z1_no_load_no_shunt_at_reference_bus'] = all(
        not v['loads_at_reference_bus'] and v['reference_bus_gs_bs'] == [0.0, 0.0] for v in per_block.values())
    checks['z1_dso_shared_storage_at_reference_bus'] = all(
        v['shared_storage_buses'] == [v['reference_bus']] for v in per_block.values())
    first = per_block[next(iter(per_block))]
    br0 = first['adjacent_in_service_branches'][0]
    ref_bus = first['reference_bus']

    # ---- built (never solved) block models: which rows limit that branch, where the storage enters ----
    dso_rows, tso_rows = {}, {}
    for year in dn.years:
        for day in dn.days:
            net = dn.network[year][day]
            m = net.build_model(dn.params)
            b = br0['branch_index']
            ref_idx = net.get_node_idx(ref_bus)
            to_idx = net.get_node_idx(br0['to_bus'])
            row_ok = True
            for p in m.periods:
                c_ij = m.branch_flow_limit[b, 0, 0, p]
                c_ji = m.branch_flow_limit_ji[b, 0, 0, p]
                v_ij = sorted(v.name for v in identify_variables(c_ij.body))
                v_ji = sorted(v.name for v in identify_variables(c_ji.body))
                nb_ref = [v.name for v in identify_variables(m.node_balance_p[ref_idx, 0, 0, p].body)]
                nb_to = [v.name for v in identify_variables(m.node_balance_p[to_idx, 0, 0, p].body)]
                adn = [v.name for v in identify_variables(m.pg_adn[0, 0, p].expr)]
                row_ok = row_ok and (
                    v_ij == sorted([f'pij[{b},0,0,{p}]', f'qij[{b},0,0,{p}]'])
                    and v_ji == sorted([f'pji[{b},0,0,{p}]', f'qji[{b},0,0,{p}]'])
                    and abs(pe.value(c_ij.upper) - ((br0['rate_mva'] / net.baseMVA) ** 2 + D.EQUALITY_TOLERANCE)) < 1e-15
                    and any(n.startswith('shared_es_pnet[') for n in nb_ref)
                    and not any(n.startswith('shared_es_pnet[') for n in nb_to)
                    and sorted(n.split('[')[0] for n in adn) == ['pg', 'shared_es_pnet'])
            dso_rows[f'{year}|{day}'] = {
                'apparent_power_limited_branches': list(m.apparent_power_limited_branches),
                'row_names': [f'branch_flow_limit[{b},s_m,s_o,p]', f'branch_flow_limit_ji[{b},s_m,s_o,p]'],
                'row_upper_pu2': pe.value(m.branch_flow_limit[b, 0, 0, 0].upper),
                'all_periods_structure_ok': row_ok}
            tnet = tn.network[year][day]
            mt = tnet.build_model(tn.params)
            i7 = tnet.get_node_idx(NODE)
            dn_idx = tn.active_distribution_network_nodes.index(NODE)
            t_ok = True
            for p in mt.periods:
                nb7 = [v.name for v in identify_variables(mt.node_balance_p[i7, 0, 0, p].body)]
                t_ok = t_ok and any(n.startswith('shared_es_pnet[') for n in nb7) and any(
                    n.startswith('pc[') for n in nb7)
            tso_sess = [e.bus for e in tnet.shared_energy_storages]
            tso_rows[f'{year}|{day}'] = {
                'tso_shared_storage_buses': tso_sess,
                'tso_node_balance_at_bus7_holds_storage_and_adn_load_all_periods': t_ok,
                'tso_branches_adjacent_to_bus7': [
                    {'from_bus': x.fbus, 'to_bus': x.tbus, 'rate_mva': x.rate}
                    for x in tnet.branches if x.status and (x.fbus == NODE or x.tbus == NODE)],
                'tso_apparent_power_limited_branches': list(mt.apparent_power_limited_branches),
                'tso_branch_limit_type': tn.params.branch_limit_type,
                'dn_index_of_node7': dn_idx}
    out['dso_built_model_rows'] = dso_rows
    out['tso_built_model_rows'] = tso_rows
    checks['z1_dso_rows_structure_ok_every_block_every_period'] = all(
        v['all_periods_structure_ok'] for v in dso_rows.values())
    checks['z1_tso_storage_at_bus7_with_adn_load_every_block'] = all(
        v['tso_node_balance_at_bus7_holds_storage_and_adn_load_all_periods'] and NODE in v['tso_shared_storage_buses']
        for v in tso_rows.values())

    # ---- terminal evidence: which row is at its bound (the certified records) ----
    bound_mva = br0['rate_mva'] * math.sqrt(1.0 + D.EQUALITY_TOLERANCE * (first['baseMVA'] / br0['rate_mva']) ** 2)
    terminal = {}
    for name, rel in (('a1a_candidate', A1A_REL), ('x0', X0_REL)):
        det = _load(os.path.join(rel, 'interface_settlement_detail_s31c.json'))['interface_reporting_detail']
        per_node = {}
        for n in ('5', '7', '9'):
            rating = planning.distribution_networks[int(n)].network[2025]['Spring'].get_interface_branch_rating()
            rows = []
            for y, dd in det[n].items():
                for day, blk in dd.items():
                    for p, v in blk['periods'].items():
                        s = math.hypot(v['p_int_dso_expected_mw'], v['q_int_dso_expected_mvar'])
                        rows.append({'year': y, 'day': day, 'period': int(p), 'p_mw': v['p_int_dso_expected_mw'],
                                     'q_mvar': v['q_int_dso_expected_mvar'], 's_mva': s, 'util': s / rating})
            at = [r for r in rows if r['util'] >= AT_RATING]
            per_node[n] = {'rating_mva': rating, 'n_slots': len(rows), 'max_util': max(r['util'] for r in rows),
                           'n_at_rating_ge_0_99': len(at)}
            if n == str(NODE):
                per_node[n]['at_rating_slots'] = sorted(at, key=lambda r: (r['year'], r['day'], r['period']))
                per_node[n]['max_s_mva'] = max(r['s_mva'] for r in rows)
                per_node[n]['max_p_mw_in_at_rating_slots'] = max(r['p_mw'] for r in at) if at else None
                per_node[n]['max_abs_s_minus_row_bound_in_at_rating_slots_mva'] = (
                    max(abs(r['s_mva'] - bound_mva) for r in at) if at else None)
                on_bound = [r for r in rows if abs(r['s_mva'] - bound_mva) <= ON_BOUND_TOL_MVA]
                per_node[n]['n_on_row_bound'] = len(on_bound)
                per_node[n]['on_row_bound_tolerance_mva'] = ON_BOUND_TOL_MVA
                per_node[n]['max_s_minus_row_bound_mva'] = max(r['s_mva'] for r in rows) - bound_mva
                per_node[n]['at_rating_utilization_min'] = min(r['util'] for r in at) if at else None
        terminal[name] = per_node
    out['terminal_interface_flows'] = terminal
    out['row_bound_mva'] = bound_mva
    out['row_bound_formula'] = ('|S_ij| at the branch_flow_limit row bound = sqrt((rate/baseMVA)^2 + '
                                'EQUALITY_TOLERANCE) * baseMVA = sqrt(1 + 1e-5) * 100 MVA')
    a7, x7 = terminal['a1a_candidate'][str(NODE)], terminal['x0'][str(NODE)]
    slots_a = {(r['year'], r['day'], r['period']) for r in a7['at_rating_slots']}
    slots_x = {(r['year'], r['day'], r['period']) for r in x7['at_rating_slots']}
    xmap = {(r['year'], r['day'], r['period']): r for r in x7['at_rating_slots']}
    out['candidate_vs_x0_at_rating'] = {
        'same_slot_set': slots_a == slots_x, 'n_candidate': len(slots_a), 'n_x0': len(slots_x),
        'max_abs_delta_p_mw_on_common_slots': max(
            (abs(r['p_mw'] - xmap[(r['year'], r['day'], r['period'])]['p_mw'])
             for r in a7['at_rating_slots'] if (r['year'], r['day'], r['period']) in xmap), default=None)}
    checks['z1_node7_only_node_at_rating'] = (a7['n_at_rating_ge_0_99'] > 0
                                             and terminal['a1a_candidate']['5']['n_at_rating_ge_0_99'] == 0
                                             and terminal['a1a_candidate']['9']['n_at_rating_ge_0_99'] == 0)
    checks['z1_node7_has_slots_on_the_branch_row_bound'] = a7['n_on_row_bound'] > 0
    checks['z1_node7_never_exceeds_the_branch_row_bound_by_more_than_1e-5_mva'] = a7['max_s_minus_row_bound_mva'] < 1e-5
    # reference generator: pg_ref = p_int + pnet_storage (bus-1 balance). The DSO's own terminal storage
    # dispatch comes from the ESS stride sidecar -- NOT committed (13 MB) but hash-recorded in the
    # committed child manifest; its sha256 is verified against that manifest before use.
    gen_ref = first['generators_at_reference_bus_pu'][0]
    pmax_mw, qmax_mvar = gen_ref[3] * first['baseMVA'], gen_ref[5] * first['baseMVA']
    stride_rel = os.path.join(A1A_REL, 'ess_entry_stride_baseline.jsonl')
    manifest = _load(os.path.join(A1A_REL, 'child_manifest_sha256.json'))
    recorded = None
    for k_, v_ in (manifest.get('files', manifest) if isinstance(manifest, dict) else {}).items():
        if k_.endswith('7eb1ce62c2509f54_n7_4h_e1/ess_entry_stride_baseline.jsonl'):
            recorded = v_ if isinstance(v_, str) else v_.get('sha256')
    actual = H.sha256_file(os.path.join(REPO, stride_rel))
    checks['z1_ess_stride_sidecar_hash_matches_committed_child_manifest'] = recorded == actual
    stride = _last_jsonl(stride_rel)
    x_dso = {(e['year'], e['day'], e['power_type']): e['x']['dso'] for e in stride['entries']
             if e['node_id'] == NODE}
    gen_rows = []
    for r in a7['at_rating_slots']:
        pn = x_dso[(r['year'], r['day'], 'p')][r['period']]
        qn = x_dso[(r['year'], r['day'], 'q')][r['period']]
        gen_rows.append({'year': r['year'], 'day': r['day'], 'period': r['period'],
                         'p_int_mw': r['p_mw'], 'storage_pnet_dso_mw_positive_charge': pn,
                         'pg_ref_mw': r['p_mw'] + pn, 'q_int_mvar': r['q_mvar'],
                         'storage_qnet_dso_mvar': qn, 'qg_ref_mvar': r['q_mvar'] + qn,
                         'storage_abs_s_over_rating': math.hypot(pn, qn) / S_MVA})
    out['reference_generator_bound_margin'] = {
        'pmax_mw': pmax_mw, 'qmax_mvar': qmax_mvar, 'ess_stride_sidecar': stride_rel,
        'ess_stride_sidecar_sha256': actual, 'ess_stride_sidecar_committed_manifest_sha256': recorded,
        'ess_stride_cycle': stride['cycle'],
        'formula': 'pg_ref = p_int + shared_es_pnet (DSO copy, x_dso; positive = charging), from the bus-1 balance',
        'max_pg_ref_mw_in_at_rating_slots': max(g['pg_ref_mw'] for g in gen_rows),
        'max_abs_qg_ref_mvar_in_at_rating_slots': max(abs(g['qg_ref_mvar']) for g in gen_rows),
        'max_storage_abs_s_over_rating_in_at_rating_slots': max(g['storage_abs_s_over_rating'] for g in gen_rows),
        'n_at_rating_slots_storage_ge_0_99_of_own_rating': sum(g['storage_abs_s_over_rating'] >= 0.99 for g in gen_rows),
        'n_at_rating_slots_storage_le_0_01_of_own_rating': sum(g['storage_abs_s_over_rating'] <= 0.01 for g in gen_rows),
        'at_rating_slots_storage_ge_0_99_of_own_rating': [g for g in gen_rows if g['storage_abs_s_over_rating'] >= 0.99],
        'per_slot': gen_rows}
    checks['z1_reference_generator_bound_inactive'] = (
        max(g['pg_ref_mw'] for g in gen_rows) < pmax_mw - 0.01
        and max(abs(g['qg_ref_mvar']) for g in gen_rows) < qmax_mvar - 0.01)

    out['one_line'] = (
        f"Node-7 DN (case33_2): get_interface_branch_rating sums ONE branch -- branch_id {br0['branch_id']} "
        f"(index {br0['branch_index']}), from bus {br0['from_bus']} (the DN reference bus) to bus {br0['to_bus']}, "
        f"transformer, {br0['rate_mva']:g} MVA; the binding row is its apparent-power limit at the bus-{br0['from_bus']} "
        f"end, branch_flow_limit[{br0['branch_index']},*] (pij^2 + qij^2 <= 1 + 1e-5 p.u.^2); the shared storage "
        f"injects at DN bus {ref_bus} (the reference bus, the from-bus of that branch) in the DSO model and at TN bus "
        f"{NODE} in the TSO model.")
    out['nuance'] = (
        f"There is no multi-branch sum: bus {ref_bus} has exactly one in-service branch, no load and no shunt, so "
        f"the DSO interface flow pg_adn = pg_ref - shared_es_pnet equals, by the bus-{ref_bus} node balance, the "
        f"from-end flow of branch {br0['from_bus']}-{br0['to_bus']} identically. The storage term appears in the "
        f"bus-{ref_bus} balance and in pg_adn and cancels between them; it appears in no row of bus "
        f"{br0['to_bus']} or beyond and not in the branch row. The injection therefore sits AT the upstream "
        f"terminal of the single constrained branch (the TN side of the cut), not upstream of a separate element: "
        f"the constrained flow is the DN's own net draw, which the storage cannot change -- only DN-side resources "
        f"(flexibility, DERs) can. In the TSO the storage is a separate injection at TN bus {NODE}, the same bus as "
        f"the ADN interface load; TN branches at bus 7 were NOT evaluated here (no TN branch flow is captured in the "
        f"records used).")
    out['confirm_or_refute'] = (
        'CONFIRMED, with the precise wording: the binding element (branch 1-2, 100 MVA, its bus-1 end) is on the '
        'DN side of the storage\'s injection bus (bus 1 = its from-bus), and the interface flow is by construction '
        'net of the storage, so the storage cannot relieve it.')
    return checks, out


# ======================================================================================================================
#  Z2 -- ageing trajectory
# ======================================================================================================================
def z2(planning, recs):
    checks, out = {}, {}
    sed = planning.shared_ess_data
    years = list(sed.years)
    idx = sed.get_shared_energy_storage_idx(NODE)
    ess = {str(y): sed.shared_energy_storages[y][idx] for y in years}
    phi = {y: e.phi_cal for y, e in ess.items()}
    k = {y: e.cl_eff for y, e in ess.items()}
    soh_min = {y: e.soh_min for y, e in ess.items()}
    num_years = {str(y): sed.years[y] for y in years}
    days = dict(sed.days)
    rel = 'shared_energy_storage_data.py'
    soh_chain = _one_line(rel, '== prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y]) * (phi_cal ** num_years))')
    chain_line = int(soh_chain.split(':')[-1])
    # the text `phi_cal = shared_energy_storage.phi_cal` occurs twice (the model build and a later
    # reporting function); the one consumed by the SoH row is the last occurrence BEFORE that row.
    phi_reads = _lines(rel, 'phi_cal = shared_energy_storage.phi_cal')
    cite = {
        'phi_cal_read': f'{rel}:{max(n for n in phi_reads if n < chain_line)}',
        'phi_cal_read_all_occurrences': [f'{rel}:{n}' for n in phi_reads],
        'soh_chain_row': soh_chain,
        'D_row': _one_line(rel, "model.es_D_per_unit[y_inv, y] * (2 * shared_energy_storage.cl_eff * model.es_e_investment_fixed[y_inv])"),
        'throughput_row': _one_line(rel, 'avg_ch_dch += (num_days / 365.00) * (eff_ch * pch * dt + pdch * dt / eff_dch)'),
        'available_energy_row': _one_line(rel, 'model.es_e_available_per_unit[y_inv, y] == model.es_e_rated_per_unit[y_inv, y] * model.es_soh_per_unit_cumul[y_inv, y]'),
        'soh_floor_row': _one_line(rel, 'model.es_soh_per_unit_cumul[y_inv, y] >= shared_energy_storage.soh_min)'),
        'published_available_capacity': _one_line(rel, 'def get_updated_capacities(self, model):'),
        'phi_cal_source_default': _one_line('shared_energy_storage_parameters.py', 'self.calendar_retention_per_year = 1.00'),
        'phi_cal_bound_to_object': _one_line('shared_energy_storage_parameters.py',
                                             'shared_energy_storage.phi_cal = self.calendar_retention_per_year'),
        'cl_eff_bound_to_object': _one_line('shared_energy_storage_parameters.py',
                                            'shared_energy_storage.cl_eff = self.effective_cycle_constant()'),
        'tso_uses_e_available': [f'shared_resources_planning.py:{n}' for n in _lines(
            'shared_resources_planning.py',
            "shared_es_e_rated_fixed[shared_ess_idx].set_value(sess_estimated_capacity[year]['e_available'] / s_base)")],
        'efc_capture_formula': _one_line('p515_g_g1_g4_admm_gates.py',
                                         'efc_per_day = (float(avg) / (2.0 * float(rated))) if (avg is not None and rated) else None'),
    }
    ess_params = _load(ESS_PARAMS_REL)
    out['citations'] = cite
    out['phi_cal_as_consumed'] = phi
    out['phi_cal_in_case_file'] = ('ageing.calendar_retention_per_year is ABSENT from '
                                   f'{ESS_PARAMS_REL}; the default 1.0 applies' if
                                   'calendar_retention_per_year' not in ess_params.get('ageing', {})
                                   else ess_params['ageing']['calendar_retention_per_year'])
    out['k_cl_eff_as_consumed'] = k
    out['k_calibration_in_case_file'] = ess_params['ageing'].get('calibration')
    out['soh_min'] = soh_min
    out['num_years_per_block'] = num_years
    out['days'] = days
    checks['z2_phi_cal_is_1'] = all(v == 1.0 for v in phi.values())
    k_cal = ess_params['ageing']['calibration']
    k_expected = k_cal['cycles_n'] * k_cal['reference_dod_d'] / (-math.log(k_cal['eol_retention_r']))
    checks['z2_k_equals_C3_calibration'] = all(abs(v - k_expected) < 1e-9 for v in k.values())

    rec = recs['a1a']
    st = rec['storage_per_node'][str(NODE)]
    floor = _last_jsonl(os.path.join(A1A_REL, 'soh_floor_sidecar_baseline.jsonl'))
    floor7 = [e for e in floor['entries'] if e['node_id'] == NODE]
    k0 = k[str(years[0])]
    n0 = num_years[str(years[0])]
    phi0 = phi[str(years[0])]
    rows, prev, max_dev = [], 1.0, 0.0
    efc_total = 0.0
    for yi, y in enumerate(years):
        key = f'(0, {yi})'
        efc = st['efc_per_day_per_cohort_year'][key]
        soh_rec = st['terminal_soh_per_active_cohort_year'][key]
        d_val = 365.0 * n0 * efc / k0          # = 365 * num_years * avg_ch_dch / (2 k E), avg = 2 E EFC
        soh_calc = prev * math.exp(-d_val) * phi0 ** n0
        e_avail_pub = st['published_available_capacity_terminal'][str(y)]['e_available']
        max_dev = max(max_dev, abs(soh_calc - soh_rec))
        fl = [e for e in floor7 if e['y_inv'] == '0' and e['y'] == str(yi)][0]
        rows.append({'cohort': years[0], 'year_block': y, 'cohort_year_index': key,
                     'efc_per_day': efc, 'D': d_val, 'soh_start_of_block': prev,
                     'soh_used_for_available_energy_end_of_block': soh_rec, 'soh_recomputed': soh_calc,
                     'e_available_mwh_published': e_avail_pub, 'e_rated_mwh': E_MWH,
                     'e_available_over_rated': e_avail_pub / E_MWH,
                     'floor_row_active': fl['active'], 'floor_row_dual': fl['dual'],
                     'sidecar_soh': fl['es_soh_per_unit_cumul'], 'sidecar_efc': fl['efc_per_day'],
                     'efc_block_total_cycles': 365.0 * n0 * efc})
        efc_total += 365.0 * n0 * efc
        prev = soh_rec
    out['trajectory'] = rows
    out['inactive_cohorts'] = [e for e in floor7 if e['y_inv'] != '0']
    out['efc_horizon_total_cycles'] = efc_total
    out['formula'] = (
        'EFC/day[y_inv,y] = avg_ch_dch / (2 E_rated), avg_ch_dch = sum_d (n_d/365) sum_p (eta_ch pch + pdch/eta_dch) dt '
        '(throughput on the CELL side, E_rated = nameplate); D[y_inv,y] = 365 * num_years * avg_ch_dch / (2 k E) = '
        '365 * num_years * EFC/day / k; SoH[y_inv,y] = SoH[y_inv,y-1] * exp(-D[y_inv,y]) * phi_cal^num_years '
        '(SoH[y_inv,y_inv-1] := 1); E_available[y] = sum_{y_inv} E_rated[y_inv,y] * SoH[y_inv,y], published to the '
        'TSO and DSO blocks of year y as shared_es_e_rated_fixed.')
    out['end_of_block_statement'] = (
        'YES -- available energy in block y uses the END-of-block SoH: SoH[0,y] already includes exp(-D[0,y]), the '
        'degradation caused by block y\'s own throughput (for 2025: 0.8345, not 1.0).')
    checks['z2_soh_chain_recomputes_within_1e-9'] = max_dev < 1e-9
    checks['z2_available_equals_rated_times_soh'] = all(
        abs(r['e_available_mwh_published'] - E_MWH * r['soh_used_for_available_energy_end_of_block']) < 1e-9
        for r in rows)
    checks['z2_sidecar_matches_record'] = all(
        abs(r['sidecar_soh'] - r['soh_used_for_available_energy_end_of_block']) < 1e-15
        and abs(r['sidecar_efc'] - r['efc_per_day']) < 1e-15 for r in rows)
    checks['z2_no_floor_row_active'] = not any(r['floor_row_active'] for r in rows)
    out['max_abs_soh_recompute_deviation'] = max_dev
    return checks, out


# ======================================================================================================================
#  Z3 -- discount re-weighting
# ======================================================================================================================
def _per_year(levels, planning):
    tn = planning.transmission_network
    years = list(tn.years)
    y0 = int(years[0])
    per_year = {str(y): {'weighted_r2': 0.0, 'undiscounted': 0.0} for y in years}
    max_w_dev = 0.0
    for key, blk in levels['blocks'].items():
        y, day = blk['year'], blk['day']
        n_years = tn.years[int(y)]
        n_days = tn.days[day]
        w_expected = n_years * n_days / ((1.0 + tn.discount_factor) ** (int(y) - y0))
        max_w_dev = max(max_w_dev, abs(blk['admm_block_weight'] - w_expected))
        g_w = sum(blk['weighted'][c] for c in GROSS_COMPONENTS)
        g_u = sum(blk['unweighted'][c] for c in GROSS_COMPONENTS)
        per_year[y]['weighted_r2'] += g_w
        per_year[y]['undiscounted'] += g_u * n_years * n_days
    total_w = sum(v['weighted_r2'] for v in per_year.values())
    return per_year, total_w, max_w_dev


def z3(planning, recs):
    checks, out = {}, {}
    tn = planning.transmission_network
    years = list(tn.years)
    y0 = int(years[0])
    cite = {'block_weight': _one_line('shared_resources_planning.py', 'def _get_admm_block_weight(network_data, year, day):'),
            'annualization': _one_line('shared_resources_planning.py',
                                       'annualization = 1.0 / ((1.0 + network_data.discount_factor) ** (int(year) - int(years[0])))'),
            'weight_return': _one_line('shared_resources_planning.py',
                                       'return float(network_data.years[year]) * float(network_data.days[day]) * annualization'),
            'discount_factor_source': f'{CASE_REL} "DiscountFactor"'}
    out['citations'] = cite
    out['production_convention'] = (
        'block weight = num_years[y] * num_days[d] / (1 + r)^(y - y_first), r = DiscountFactor = 0.02: ONE discount '
        'factor per representative year, applied to all num_years (5) years of that block (NOT an annuity over '
        'the 5 years); y_first = 2025 so the 2025 block is undiscounted.')
    out['discount_factor_in_model'] = tn.discount_factor
    checks['z3_model_discount_factor_is_0_02'] = tn.discount_factor == PRODUCTION_RATE

    lv = {name: _load(os.path.join(rel, 'component_levels_terminal.json'))
          for name, rel in (('a1a', A1A_REL), ('a0_c7', A0X_REL), ('x0', X0_REL))}
    # the A0 and A1a copies of the candidate are the same key; their per-block levels must be bit-identical
    checks['z3_a1a_and_a0_c7_candidate_blocks_bit_identical'] = (
        json.dumps(lv['a1a']['blocks'], sort_keys=True) == json.dumps(lv['a0_c7']['blocks'], sort_keys=True))
    q = {}
    for name in ('a1a', 'x0'):
        per_year, total_w, w_dev = _per_year(lv[name], planning)
        gross = lv[name]['recourse_components']['gross_operational_cost']
        q[name] = {'per_year': per_year, 'sum_blocks_weighted_r2': total_w,
                   'recorded_gross_operational_cost': gross,
                   'recorded_certified_cost_in_evaluation_record': recs[name]['certified_cost'],
                   'abs_sum_minus_recorded': abs(total_w - gross), 'max_block_weight_deviation': w_dev}
        checks[f'z3_{name}_blocks_reconcile_to_gross_within_0.01_eur'] = abs(total_w - gross) < 0.01
        checks[f'z3_{name}_block_weights_match_formula'] = w_dev < 1e-9
        checks[f'z3_{name}_undiscounted_times_factor_equals_weighted'] = all(
            abs(v['undiscounted'] / ((1 + PRODUCTION_RATE) ** (int(y) - y0)) - v['weighted_r2'])
            < 1e-6 * abs(v['weighted_r2']) for y, v in per_year.items())
    out['Q_per_year'] = q

    v_year = {y: {'V_weighted_r2': q['x0']['per_year'][y]['weighted_r2'] - q['a1a']['per_year'][y]['weighted_r2'],
                  'V_undiscounted': q['x0']['per_year'][y]['undiscounted'] - q['a1a']['per_year'][y]['undiscounted']}
              for y in (str(x) for x in years)}
    out['V_per_year'] = v_year
    table = _load(PHASE_A_TABLES_REL)
    t1_row = [r for r in table['T1'] if isinstance(r, dict) and r.get('identity') == PHASE_A_IDENTITY] \
        if isinstance(table['T1'], list) else None
    if not t1_row:
        rows = table['T1'].get('rows') if isinstance(table['T1'], dict) else None
        t1_row = [r for r in (rows or []) if r.get('identity') == PHASE_A_IDENTITY]
    if len(t1_row) != 1:
        raise RuntimeError(f'Phase A T1 row {PHASE_A_IDENTITY} not found exactly once')
    t1 = t1_row[0]
    v_cert = t1['value_eur']
    i_x = t1['I_x_eur']
    checks['z3_T1_row_candidate_key_matches'] = t1.get('candidate_key') == EXPECTED_KEY
    v_r2 = sum(v['V_weighted_r2'] for v in v_year.values())
    v_r2_from_undisc = sum(v['V_undiscounted'] / (1 + PRODUCTION_RATE) ** (int(y) - y0) for y, v in v_year.items())
    out['reconstruction'] = {
        'certified_value_eur': v_cert, 'certified_value_source': f"{PHASE_A_TABLES_REL} T1 identity "
                                                                  f"{PHASE_A_IDENTITY} value_eur",
        'V_r2_from_weighted_blocks': v_r2, 'V_r2_from_undiscounted_reweighted': v_r2_from_undisc,
        'abs_diff_weighted': abs(v_r2 - v_cert), 'abs_diff_reweighted': abs(v_r2_from_undisc - v_cert),
        'tolerance_eur': Z3_RECONSTRUCTION_TOL_EUR}
    checks['z3_r2_reconstruction_within_tolerance'] = (abs(v_r2 - v_cert) <= Z3_RECONSTRUCTION_TOL_EUR
                                                       and abs(v_r2_from_undisc - v_cert) <= Z3_RECONSTRUCTION_TOL_EUR)
    bar_sum = recs['a1a']['bar']['value'] + recs['x0']['bar']['value']
    rows = []
    for r in RATES:
        per = {y: v['V_undiscounted'] / (1 + r) ** (int(y) - y0) for y, v in v_year.items()}
        val = sum(per.values())
        # conservative resolution: the bar is on the whole weighted Q; re-weighting can scale an error
        # sitting in year y by (1.02/(1+r))^(y-y0); the worst case over y bounds it.
        scale = max(((1 + PRODUCTION_RATE) / (1 + r)) ** (int(y) - y0) for y in v_year)
        rows.append({'rate': r, 'V_per_year': per, 'V': val, 'V_over_V_2pct': None,
                     'value_to_cost': val / i_x, 'resolution_bar_eur_conservative': bar_sum * scale})
    v2 = [x for x in rows if x['rate'] == PRODUCTION_RATE][0]['V']
    for x in rows:
        x['V_over_V_2pct'] = x['V'] / v2
        x['change_vs_2pct_pct'] = 100.0 * (x['V'] / v2 - 1.0)
        x['expert_prediction'] = EXPERT_PREDICTION_Z3.get(str(x['rate']))
    out['I_x_eur'] = i_x
    out['I_x_source'] = f"{PHASE_A_TABLES_REL} T1 identity {PHASE_A_IDENTITY} I_x_eur (paid in 2025, factor 1.0, held fixed)"
    out['bar_sum_eur'] = bar_sum
    out['bar_note'] = ('resolution of V(2 %) = bar(x) + bar(x0) (max |gross step| over the last 10 cycles of each '
                       'run); the per-year split of the bar is not recorded, so V(r)\'s bar is bounded '
                       'conservatively by bar_sum * max_y ((1.02/(1+r))^(y-2025)).')
    out['table'] = rows
    out['formula'] = ('Q_y = sum over the TSO block and all DSO blocks of year y of the gross settlement-excluded '
                      'per-block cost (sum of ' + ' + '.join(GROSS_COMPONENTS) + '); undiscounted Q_y = sum of '
                      'unweighted * num_years * num_days; V_y = Q_y(0) - Q_y(x); V(r) = sum_y V_y_undiscounted / '
                      '(1 + r)^(y - 2025); value_to_cost(r) = V(r) / I(x).')
    return checks, out


# ======================================================================================================================
#  Z4 -- price spread
# ======================================================================================================================
def z4(planning, z2_out, z3_out):
    import numpy as np
    checks, out = {}, {}
    tn = planning.transmission_network
    years = list(tn.years)
    y0 = int(years[0])
    cite = {'market_reader': _one_line('shared_resources_planning.py', 'def _read_market_data_from_file(planning_problem):'),
            'price_array': _one_line('shared_resources_planning.py',
                                     'planning_problem.cost_energy_p[year][day] = np.array(energy_selected_profiles * energy_growth_cumul)'),
            'tso_prices_bound': _one_line('shared_resources_planning.py',
                                          'transmission_network.network[year][day].cost_energy_p = planning_problem.cost_energy_p[year][day]'),
            'dso_prices_bound': _one_line('shared_resources_planning.py',
                                          'distribution_network.network[year][day].cost_energy_p = planning_problem.cost_energy_p[year][day]'),
            'generation_cost_uses_price': _one_line('model_construction_helpers.py',
                                                    'gen_cost_scenario += c_p[p] * network.baseMVA * model.pg[g, s_m, s_o, p]'),
            'market_file': f'{MARKET_REL} (base profiles; SRP1.json "MarketData"), synthetic scenario selection '
                           f'seeded by SRP1.json "RandomSeed", growth factor from its "Growth Factors" sheet'}
    out['citations'] = cite
    same_obj, max_net_dev = True, 0.0
    for year in years:
        for day in tn.days:
            ref = planning.cost_energy_p[year][day]
            same_obj = same_obj and tn.network[year][day].cost_energy_p is ref
            for nid, dn in planning.distribution_networks.items():
                same_obj = same_obj and dn.network[year][day].cost_energy_p is ref
                max_net_dev = max(max_net_dev, float(np.max(np.abs(np.asarray(dn.network[year][day].cost_energy_p)
                                                                    - np.asarray(ref)))))
    checks['z4_tso_and_dso_price_arrays_are_the_planning_object'] = same_obj
    n_scen = {str(y): {d: int(np.asarray(planning.cost_energy_p[y][d]).shape[0]) for d in tn.days} for y in years}
    checks['z4_one_market_scenario'] = all(v == 1 for d in n_scen.values() for v in d.values())
    # cross-check against the prices the certified run recorded at every node-7 slot
    det = _load(os.path.join(A1A_REL, 'interface_settlement_detail_s31c.json'))['interface_reporting_detail']
    max_rec_dev = 0.0
    for y, dd in det[str(NODE)].items():
        for day, blk in dd.items():
            arr = np.asarray(planning.cost_energy_p[int(y)][day])[0]
            for p, v in blk['periods'].items():
                max_rec_dev = max(max_rec_dev, abs(v['price_per_mwh'] - float(arr[int(p)])))
    out['max_abs_dev_vs_prices_recorded_in_certified_run'] = max_rec_dev
    checks['z4_prices_equal_those_recorded_by_the_certified_run'] = max_rec_dev < 1e-9

    h = DURATION_H
    rows = []
    for year in years:
        for day in tn.days:
            arr = np.asarray(planning.cost_energy_p[year][day])[0].astype(float)
            srt = np.sort(arr)
            rows.append({'year': year, 'day': day, 'num_days': tn.days[day], 'num_years': tn.years[year],
                         'weight_undiscounted': tn.years[year] * tn.days[day],
                         'weight_model_r2': tn.years[year] * tn.days[day] / (1 + tn.discount_factor) ** (int(year) - y0),
                         'min': float(srt[0]), 'max': float(srt[-1]), 'mean': float(arr.mean()),
                         'max_minus_min': float(srt[-1] - srt[0]),
                         f'top{h}_mean_minus_bottom{h}_mean': float(srt[-h:].mean() - srt[:h].mean()),
                         'prices': [float(x) for x in arr]})

    def wavg(field, wkey, subset=None):
        rs = [r for r in rows if subset is None or subset(r)]
        return sum(r[field] * r[wkey] for r in rs) / sum(r[wkey] for r in rs)

    f_mm, f_th = 'max_minus_min', f'top{h}_mean_minus_bottom{h}_mean'
    summary = {
        'all_years_day_weighted_undiscounted': {'max_minus_min': wavg(f_mm, 'weight_undiscounted'),
                                                f_th: wavg(f_th, 'weight_undiscounted'),
                                                'mean_price': wavg('mean', 'weight_undiscounted')},
        'all_years_model_block_weight_r2': {'max_minus_min': wavg(f_mm, 'weight_model_r2'),
                                            f_th: wavg(f_th, 'weight_model_r2'),
                                            'mean_price': wavg('mean', 'weight_model_r2')},
        'per_year_day_weighted': {str(y): {'max_minus_min': wavg(f_mm, 'weight_undiscounted', lambda r, y=y: r['year'] == y),
                                           f_th: wavg(f_th, 'weight_undiscounted', lambda r, y=y: r['year'] == y),
                                           'mean_price': wavg('mean', 'weight_undiscounted', lambda r, y=y: r['year'] == y)}
                                  for y in years},
    }
    out['per_representative_day'] = rows
    out['spread_summary'] = summary
    out['spread_formula'] = (f'per representative (year, day): max_p c_p - min_p c_p, and mean of the top-{h} hourly '
                             f'prices minus mean of the bottom-{h} (h = E/P = 4 h); averaged with weight num_years * '
                             f'num_days (undiscounted) and, separately, with the model block weight num_years * '
                             f'num_days / 1.02^(y-2025).')

    # captured spread
    efc_rows = z2_out['trajectory']
    efc_total = z2_out['efc_horizon_total_cycles']
    v2 = [x for x in z3_out['table'] if x['rate'] == PRODUCTION_RATE][0]['V']
    v0 = [x for x in z3_out['table'] if x['rate'] == 0.0][0]['V']
    efc_disc = sum(r['efc_block_total_cycles'] / (1 + PRODUCTION_RATE) ** (int(r['year_block']) - y0) for r in efc_rows)
    captured = {
        'formula': ('captured spread [EUR/MWh per full cycle] = (V / E) / EFC_horizon, EFC_horizon = sum_y 365 * '
                    'num_years * EFC/day_y (full equivalent cycles of the nameplate E over the 15 years; Z2 EFC, '
                    'cell-side throughput / (2 E))'),
        'E_mwh': E_MWH, 'EFC_horizon_cycles': efc_total,
        'EFC_horizon_discounted_2pct_cycles': efc_disc,
        'a_literal_V2pct_over_undiscounted_EFC': (v2 / E_MWH) / efc_total,
        'b_consistent_undiscounted_V0_over_undiscounted_EFC': (v0 / E_MWH) / efc_total,
        'c_consistent_discounted_V2pct_over_discounted_EFC': (v2 / E_MWH) / efc_disc,
        'per_year_undiscounted': {
            r['year_block']: {'V_undiscounted': z3_out['V_per_year'][str(r['year_block'])]['V_undiscounted'],
                              'EFC_block_cycles': r['efc_block_total_cycles'],
                              'captured_eur_per_mwh_cycle': (z3_out['V_per_year'][str(r['year_block'])]['V_undiscounted']
                                                             / E_MWH) / r['efc_block_total_cycles'],
                              'soh_end_of_block': r['soh_used_for_available_energy_end_of_block']}
            for r in efc_rows},
        'expert_prediction': EXPERT_PREDICTION_Z4,
    }
    out['captured_spread'] = captured
    return checks, out


# ======================================================================================================================
#  Markdown
# ======================================================================================================================
def _markdown(res):
    L = []
    inst = res['instance']
    L.append('# P5.15 Addendum 28 W19 -- zero-solve reports Z1-Z4')
    L.append('')
    L.append(f"Instance: node 7, 0.25 MVA / 1.0 MWh, 2025; nodes 5, 9 empty. candidate_key "
             f"`{inst['candidate_key']}`; x = 0 key `{inst['x0_candidate_key']}`.")
    L.append(f"Records: A1a `{A1A_REL}`, A0 duplicate `{A0X_REL}`, x0 `{X0_REL}` (all certified, frozen oracle).")
    L.append(f"Solve profile: armed SolveProfileGuard(permitted=()); counts {res['solve_profile_guard']['counts']}; "
             f"verify(0) failures {res['solve_profile_guard']['verify_0_failures']}.")
    L.append(f"Git HEAD {res['git_HEAD']}. all_pass = {res['all_pass']}; failed checks: {res['failed_checks']}.")
    L.append('')
    L.append('Objective convention: Q(x) = GROSS operational cost, settlement-EXCLUDED (the oracle cost convention); '
             'terminal salvage is ~0 on these runs (gross = net). V = Q(0) - Q(x). EUR.')
    L.append('')
    z = res['Z1']
    L.append('## Z1 -- branch endpoints')
    L.append('')
    L.append(z['one_line'])
    L.append('')
    L.append('**Nuance.** ' + z['nuance'])
    L.append('')
    L.append('**Verdict.** ' + z['confirm_or_refute'])
    L.append('')
    a7 = z['terminal_interface_flows']['a1a_candidate']['7']
    L.append(f"Evidence (terminal, certified A1a record): node 7 at or above {AT_RATING:.0%} of 100 MVA in "
             f"{a7['n_at_rating_ge_0_99']}/{a7['n_slots']} slots; max |S| = {a7['max_s_mva']:.7f} MVA against the row bound "
             f"sqrt(1 + 1e-5) x 100 = {z['row_bound_mva']:.7f} MVA (max deviation in the at-rating slots "
             f"{a7['max_abs_s_minus_row_bound_in_at_rating_slots_mva']:.2e} MVA; minimum utilization among them "
             f"{a7['at_rating_utilization_min']:.6f}; {a7['n_on_row_bound']} slots within "
             f"{a7['on_row_bound_tolerance_mva']:g} MVA of the bound). Nodes 5/9 max utilization "
             f"{z['terminal_interface_flows']['a1a_candidate']['5']['max_util']:.3f} / "
             f"{z['terminal_interface_flows']['a1a_candidate']['9']['max_util']:.3f}. Same at-rating slot set at x = 0: "
             f"{z['candidate_vs_x0_at_rating']['same_slot_set']} (max |dP| on those slots "
             f"{z['candidate_vs_x0_at_rating']['max_abs_delta_p_mw_on_common_slots']:.2e} MW). Reference generator "
             f"(pmax {z['reference_generator_bound_margin']['pmax_mw']:g} MW) is not at its bound: max pg_ref in those "
             f"slots {z['reference_generator_bound_margin']['max_pg_ref_mw_in_at_rating_slots']:.4f} MW. Storage in "
             f"those slots (DSO copy): at >= 99 % of its own 0.25 MVA rating in "
             f"{z['reference_generator_bound_margin']['n_at_rating_slots_storage_ge_0_99_of_own_rating']}, at <= 1 % in "
             f"{z['reference_generator_bound_margin']['n_at_rating_slots_storage_le_0_01_of_own_rating']} of "
             f"{a7['n_at_rating_ge_0_99']} (per-slot values in the JSON).")
    L.append('')
    L.append('Citations: ' + '; '.join(f'{k} {v}' for k, v in z['citations'].items()))
    L.append('')
    z = res['Z2']
    L.append('## Z2 -- ageing trajectory (n7 0.25/1.0, 2025)')
    L.append('')
    y0 = str(list(z['phi_cal_as_consumed'])[0])
    L.append(f"phi_cal as consumed = {z['phi_cal_as_consumed'][y0]} ({z['citations']['phi_cal_read']}, applied at "
             f"{z['citations']['soh_chain_row']}; {z['phi_cal_in_case_file']}). k = cl_eff = "
             f"{z['k_cl_eff_as_consumed'][y0]:.9f} (C3: 10000 x 0.80 / -ln 0.50; {z['citations']['D_row']}). "
             f"soh_min = {z['soh_min'][y0]}.")
    L.append('')
    L.append('| cohort | block | EFC/day | D | SoH start | SoH used (end of block) | recomputed | E_avail MWh | floor active |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for r in z['trajectory']:
        L.append(f"| {r['cohort']} | {r['year_block']} | {r['efc_per_day']:.6f} | {r['D']:.6f} | "
                 f"{r['soh_start_of_block']:.6f} | {r['soh_used_for_available_energy_end_of_block']:.6f} | "
                 f"{r['soh_recomputed']:.6f} | {r['e_available_mwh_published']:.6f} | {r['floor_row_active']} |")
    L.append('')
    L.append(f"EFC over the horizon = {z['efc_horizon_total_cycles']:.2f} full cycles. Cohorts 2030/2035 carry zero "
             f"capacity (SoH fixed 1.0, not a result).")
    L.append('')
    L.append('Formula: ' + z['formula'])
    L.append('')
    L.append('**End-of-block:** ' + z['end_of_block_statement'])
    L.append('')
    z = res['Z3']
    L.append('## Z3 -- discount re-weighting')
    L.append('')
    L.append('Production convention: ' + z['production_convention'] + f" ({z['citations']['block_weight']}-"
             f"{z['citations']['weight_return'].split(':')[-1]}).")
    L.append('')
    L.append('Formula: ' + z['formula'])
    L.append('')
    L.append('| year | Q_y(0) weighted 2 % | Q_y(x) weighted 2 % | V_y weighted 2 % | V_y undiscounted |')
    L.append('|---|---|---|---|---|')
    for y, v in z['V_per_year'].items():
        L.append(f"| {y} | {_fmt(z['Q_per_year']['x0']['per_year'][y]['weighted_r2'])} | "
                 f"{_fmt(z['Q_per_year']['a1a']['per_year'][y]['weighted_r2'])} | {_fmt(v['V_weighted_r2'])} | "
                 f"{_fmt(v['V_undiscounted'])} |")
    rc = z['reconstruction']
    L.append('')
    L.append(f"Reconstruction check: V(2 %) from blocks = {_fmt(rc['V_r2_from_weighted_blocks'], 6)}, re-weighted from "
             f"undiscounted = {_fmt(rc['V_r2_from_undiscounted_reweighted'], 6)}; certified "
             f"{_fmt(rc['certified_value_eur'], 6)}; |diff| {rc['abs_diff_weighted']:.2e} / "
             f"{rc['abs_diff_reweighted']:.2e} EUR (tolerance {rc['tolerance_eur']} EUR).")
    L.append('')
    L.append(f"I(x) = {_fmt(z['I_x_eur'])} EUR (2025, held fixed). Resolution of V(2 %): bar(x) + bar(x0) = "
             f"{_fmt(z['bar_sum_eur'])} EUR.")
    L.append('')
    L.append('| r | V(r) | V(r)/V(2 %) | change | value-to-cost | bar (conservative) | expert |')
    L.append('|---|---|---|---|---|---|---|')
    for x in z['table']:
        L.append(f"| {x['rate']:.0%} | {_fmt(x['V'])} | {x['V_over_V_2pct']:.4f} | {x['change_vs_2pct_pct']:+.2f} % | "
                 f"{x['value_to_cost']:.4f} | {_fmt(x['resolution_bar_eur_conservative'])} | "
                 f"{x['expert_prediction'] or '-'} |")
    L.append('')
    L.append(z['bar_note'])
    L.append('')
    z = res['Z4']
    L.append('## Z4 -- price spread')
    L.append('')
    s = z['spread_summary']
    k4 = f'top{DURATION_H}_mean_minus_bottom{DURATION_H}_mean'
    L.append('Formula: ' + z['spread_formula'])
    L.append('')
    L.append('| scope | max - min | top-4 - bottom-4 | mean price |')
    L.append('|---|---|---|---|')
    L.append(f"| all years, day-weighted (undiscounted) | {s['all_years_day_weighted_undiscounted']['max_minus_min']:.2f} | "
             f"{s['all_years_day_weighted_undiscounted'][k4]:.2f} | {s['all_years_day_weighted_undiscounted']['mean_price']:.2f} |")
    L.append(f"| all years, model block weight (2 %) | {s['all_years_model_block_weight_r2']['max_minus_min']:.2f} | "
             f"{s['all_years_model_block_weight_r2'][k4]:.2f} | {s['all_years_model_block_weight_r2']['mean_price']:.2f} |")
    for y, v in s['per_year_day_weighted'].items():
        L.append(f"| {y}, day-weighted | {v['max_minus_min']:.2f} | {v[k4]:.2f} | {v['mean_price']:.2f} |")
    L.append('')
    L.append('| year | day | days | min | max | max - min | top-4 - bottom-4 |')
    L.append('|---|---|---|---|---|---|---|')
    for r in z['per_representative_day']:
        L.append(f"| {r['year']} | {r['day']} | {r['num_days']} | {r['min']:.2f} | {r['max']:.2f} | "
                 f"{r['max_minus_min']:.2f} | {r[k4]:.2f} |")
    c = z['captured_spread']
    L.append('')
    L.append('Captured spread. ' + c['formula'])
    L.append('')
    L.append(f"- EFC_horizon = {c['EFC_horizon_cycles']:.2f} cycles (discounted at 2 %: {c['EFC_horizon_discounted_2pct_cycles']:.2f})")
    L.append(f"- (a) V(2 %) / EFC_horizon = {c['a_literal_V2pct_over_undiscounted_EFC']:.2f} EUR/MWh per full cycle")
    L.append(f"- (b) V(0 %) / EFC_horizon = {c['b_consistent_undiscounted_V0_over_undiscounted_EFC']:.2f}")
    L.append(f"- (c) V(2 %) / EFC_horizon discounted = {c['c_consistent_discounted_V2pct_over_discounted_EFC']:.2f}")
    for y, v in c['per_year_undiscounted'].items():
        L.append(f"- {y}: V_y undiscounted {_fmt(v['V_undiscounted'])} / {v['EFC_block_cycles']:.1f} cycles = "
                 f"{v['captured_eur_per_mwh_cycle']:.2f} EUR/MWh per cycle (SoH end of block {v['soh_end_of_block']:.4f})")
    L.append(f"- expert: {c['expert_prediction']}")
    L.append('')
    L.append('Citations: ' + '; '.join(f'{k} {v}' for k, v in z['citations'].items()))
    L.append('')
    L.append('## Checks')
    L.append('')
    for k, v in sorted(res['checks'].items()):
        L.append(f'- {k}: {v}')
    L.append('')
    return '\n'.join(L)


# ======================================================================================================================
def _manifest():
    files = {}
    for name in sorted(os.listdir(OUT_ROOT)):
        if name == MANIFEST_NAME:
            continue
        path = os.path.join(OUT_ROOT, name)
        files[os.path.relpath(path, REPO)] = {'sha256': H.sha256_file(path), 'bytes': os.path.getsize(path)}
    manifest = {'stage': STAGE, 'generated_utc': _utc(), 'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
                'instance_candidate_key': EXPECTED_KEY,
                'script': {'path': os.path.basename(__file__), 'sha256': H.sha256_file(os.path.abspath(__file__))},
                'files': files}
    path = os.path.join(OUT_ROOT, MANIFEST_NAME)
    if os.path.exists(path):
        raise SystemExit(f'refusing to overwrite the manifest: {path}')
    with open(path, 'w') as handle:
        json.dump(manifest, handle, indent=1)
    print(f'[W19] manifest: {len(files)} files -> {os.path.relpath(path, REPO)}', flush=True)


def main():
    started = _utc()
    print(f'[W19] instance: node {NODE} {S_MVA} MVA / {E_MWH} MWh, 2025; candidate_key {EXPECTED_KEY}; '
          f'x0 record {X0_REL}', flush=True)
    if not os.path.isdir(OUT_ROOT):
        raise SystemExit(f'output directory must be created by the launcher: {OUT_ROOT}')
    results_path = os.path.join(OUT_ROOT, RESULTS_NAME)
    md_path = os.path.join(OUT_ROOT, MD_NAME)
    for p in (results_path, md_path):
        if os.path.exists(p):
            raise SystemExit(f'refusing to overwrite (write-once): {p}')

    inputs = [SPEC_REL, PHASE_A_TABLES_REL, CASE_REL, MARKET_REL, ESS_PARAMS_REL,
              'data/SRP1/SRP1_params.json', 'data/SRP1/case33_2/case33_2_params.json',
              'data/SRP1/case9/case9_params.json']
    for rel in (A1A_REL, A0X_REL, X0_REL):
        inputs += [os.path.join(rel, f) for f in ('evaluation_record.json', 'component_levels_terminal.json',
                                                  'interface_settlement_detail_s31c.json',
                                                  'soh_floor_sidecar_baseline.jsonl')]
    inputs.append(os.path.join(A1A_REL, 'child_manifest_sha256.json'))
    code = ['network.py', 'model_construction_helpers.py', 'shared_resources_planning.py',
            'shared_energy_storage_data.py', 'shared_energy_storage_parameters.py', 'p56a_oracle.py',
            'p515_s44_campaign_harness.py', 'p515_g_g1_g4_admm_gates.py', 'definitions.py',
            os.path.basename(__file__)]
    tracked = set(_git(['ls-files', '--', *inputs, *code]).splitlines())
    res = {
        'stage': STAGE, 'authority': AUTHORITY, 'started_utc': started,
        'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
        'git_status_porcelain_inputs_and_code': _git(['status', '--porcelain', '--', *inputs, *code]).splitlines(),
        'inputs_sha256': {rel: {'sha256': H.sha256_file(os.path.join(REPO, rel)), 'tracked': rel in tracked}
                          for rel in inputs},
        'code_sha256': {rel: {'sha256': H.sha256_file(os.path.join(REPO, rel)), 'tracked': rel in tracked}
                        for rel in code},
        'objective_convention': ('Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); '
                                 'V(x) = Q(0) - Q(x); terminal salvage reported by the records ~0 (gross = net here)'),
    }
    checks, errors = {}, {}
    checks['inputs_all_tracked'] = all(v['tracked'] for v in res['inputs_sha256'].values())
    try:
        c, info, recs = instance_section()
        checks.update(c)
        res['instance'] = info
        import p56a_oracle as O
        planning = O.load_baseline()['planning']
        res['baseline_checksum'] = O.load_baseline()['checksum']
        for name, fn in (('Z1', lambda: z1(planning)), ('Z2', lambda: z2(planning, recs)),
                         ('Z3', lambda: z3(planning, recs))):
            try:
                c, out = fn()
                checks.update(c)
                res[name] = out
            except Exception:  # noqa: BLE001 -- recorded; the run then fails
                errors[name] = traceback.format_exc()
                print(errors[name], file=sys.stderr, flush=True)
        if 'Z2' in res and 'Z3' in res:
            try:
                c, out = z4(planning, res['Z2'], res['Z3'])
                checks.update(c)
                res['Z4'] = out
            except Exception:  # noqa: BLE001
                errors['Z4'] = traceback.format_exc()
                print(errors['Z4'], file=sys.stderr, flush=True)
        else:
            errors['Z4'] = 'not run: Z2 or Z3 failed'
    except Exception:  # noqa: BLE001
        errors['setup'] = traceback.format_exc()
        print(errors['setup'], file=sys.stderr, flush=True)

    guard_failures = GUARD.verify(0)
    res.update({'checks': checks, 'failed_checks': sorted(k for k, v in checks.items() if not v),
                'errors': errors,
                'solve_profile_guard': {'permitted': [], 'counts': dict(GUARD.counts),
                                        'verify_0_failures': guard_failures},
                'ended_utc': _utc()})
    res['all_pass'] = bool(checks) and not res['failed_checks'] and not errors and not guard_failures
    H._write_once_json(results_path, res)
    if not errors:
        with open(md_path, 'w') as handle:
            handle.write(_markdown(res))
    GUARD.uninstall()
    print(f"[W19] checks={len(checks)} failed={res['failed_checks']} errors={sorted(errors)} "
          f"guard={dict(GUARD.counts)} verify0_failures={guard_failures} all_pass={res['all_pass']}", flush=True)
    if not errors:
        z3r = res['Z3']['reconstruction']
        print(f"[W19] Z3 reconstruction |diff| = {z3r['abs_diff_weighted']:.3e} EUR (tol {z3r['tolerance_eur']})",
              flush=True)
    if not res['all_pass']:
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        GUARD.uninstall()
        _manifest()
        sys.exit(0)
    main()
