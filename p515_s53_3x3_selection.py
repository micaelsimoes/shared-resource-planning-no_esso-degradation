"""P5.15 Addendum 40 ruling 3 (task W52) -- 3 x 3 selection rule and R prediction.
ZERO SOLVES, no model construction, read-only (production data reader only).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 3; frozen spec v23
`data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json` key `ruling3_3x3`.

WHAT THIS SCRIPT DOES
  0. Reuses W22 (`p515_s47_market_spreads.py`, commit 046d4d00) BY IMPORT: the spread definition
     (`day_spreads`, `weights`, `horizon`, `instance_spreads`), the data-only production read
     (`production_read` with `Blocks`: Network.build_model, NetworkData.build_model,
     SharedEnergyStorageData.build_subproblem / .build_master_problem replaced by RAISING blockers, declared
     count 0; declared stop point SharedEnergyStorageData.read_parameters_from_file, 1 hit per read, 2 reads),
     and production's energy growth factor (`energy_growth_factor`). W22's module-level OUT_DIR is re-pointed
     to this script's output directory before the reads, so nothing is written under P515S47.
     The operational data (loads' pd, generators' pg) are read by production BEFORE the stop point
     (`_read_planning_problem`: DSO and TSO `read_network_data` precede `_compute_scenario_metadata`); the
     planning object is captured by a pass-through recorder on SharedResourcesPlanning.read_planning_problem
     (records `self`, then calls the original unchanged). Instances: SRP1 (`data/SRP1/SRP1.json`) and the
     committed paper-scale case (`.../P515S44/scale_measurement/paper_build/case/SRP1__paper.json`, 5 years
     x 3, 4 days, 5 market x 5 operation scenarios); both scenario checksums are hard requirements.
  1. MARKET. For every 3-subset S of the five market-scenario indices, per (year, day):
     pm_S[s] = pm[s] / sum_{s' in S} pm[s'] (probability renormalization), mean_c_S = sum_{s in S} pm_S[s] c[s],
     and its 4 h spread (W19 Z4 / W22: mean of the 4 highest minus mean of the 4 lowest hourly prices).
     Horizon value = day-weighted average, w = num_years[y] * num_days[d] (undiscounted); also with the model
     block weight w / 1.02^(y - 2025) and growth-normalized (each (y, d) spread / (1+g)^(y-2025) before
     weighting). TARGET = W22's committed five-scenario mean-profile spread (undiscounted), 91.7277...
     SELECTION RULE (stated before the run): rank by |spread_u(S) - TARGET|, ascending; tie (within 1e-9)
     broken by |spread_r2(S) - TARGET_r2| (TARGET_r2 = five-scenario mean-profile spread, model weight), then
     by the lexicographic order of the index tuple. The full ranking of the 10 subsets is recorded.
  2. OPERATION. Metric (stated before the run; PRIMARY): system horizon RES availability per operation-scenario
     index s_o = day-weighted (w undiscounted) average over (year, day) of the daily available RES energy
     sum_{networks} sum_{PV/Wind generators} sum_{hours} pg[s_o, h] * baseMVA  [MWh/day], networks = the TSO
     (case9) and the three DSOs, as read by production (before any ADN/ESS node is added). SECONDARY (reported,
     not used for selection): the same with load (sum of pd = pc + pc_flex), RES share of load = RES / load
     (ratio of horizon averages), net load = load - RES, per network, and with the model block weight.
     SELECTION: the five indices ranked by the primary metric; selected = (lowest, median = 3rd, highest).
     Distinguishability evidence: relative range (max - min) / mean of RES and of load across the five; the
     per-(year, day) rank of each index on block RES and Kendall's W (coefficient of concordance) across the
     20 blocks, W = 12 * S / (m^2 (n^3 - n)), m = 20 blocks, n = 5 indices, S = sum_i (R_i - mean R)^2,
     R_i = sum over blocks of index i's rank (1 = lowest RES).
  3. R PREDICTION for the selected set (as W22): R_u = spread_u(S*) / 97.99086039739976 (SRP1, Z4,
     undiscounted); R_r2 = spread_r2(S*) / 97.01788674187569 (SRP1, Z4, model block weight, production
     DiscountFactor 0.02); R_gn = growth-normalized spread(S*) / growth-normalized SRP1 spread (undiscounted
     weights). The same three for the five-scenario set must reproduce W22's committed values to 1e-9.
  4. CONTEXT: bias of the reduction on the first-order arbitrage value = spread(S*) / spread(5) - 1 in each form.
  5. Realization check (index level, dummy 100-row frames, no prices generated): production draws scenarios
     with DataFrame.sample(n=NumScenarios, random_state=seed) and no seed depends on the scenario count, so a
     case derived with 3 scenarios holds the first 3 rows of the 5-scenario draw iff sample(n=3) is a prefix
     of sample(n=5) for every seed production uses; checked for every market (energy, flexibility) and every
     operational (load pc/qc/pc_flex/pc_flex_up/pc_flex_down, generator pg/qg) seed of the paper case.

Output (write-once, new directory) data/SRP1/Results/P515S53/selection_3x3/:
    selection_3x3.json, selection_3x3.md, launch.log, production_stdout_capture.log, manifest_sha256.json,
    production_read/ (production's scenario plots; hashed in the manifest only, not committed).

Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S53/selection_3x3
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_3x3_selection.py \\
        > data/SRP1/Results/P515S53/selection_3x3/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_3x3_selection.py --manifest
"""
import itertools
import json
import os
import subprocess
import sys
import time
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')

import p515_s47_market_spreads as W22  # noqa: E402  (W22 precedent, reused by import)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

STAGE = 'P5.15 Addendum 40 ruling 3 W52 -- 3 x 3 selection rule and R prediction (spec v23; zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 3',
             'data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json ruling3_3x3']
SPEC_REL = 'data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'selection_3x3')
OUT_DIR = os.path.join(REPO, OUT_REL)
RESULTS_NAME = 'selection_3x3.json'
MD_NAME = 'selection_3x3.md'
MANIFEST_NAME = 'manifest_sha256.json'
STDOUT_CAPTURE = 'production_stdout_capture.log'

W22_JSON_REL = 'data/SRP1/Results/P515S47/market_spreads/market_spreads.json'
W22_COMMIT = '046d4d00'
F_TH = W22.F_TH
F_MM = W22.F_MM
K = 3                                   # scenarios kept per dimension
TIE_TOL = 1e-9
REPRO_TOL = 1e-9
Y0 = 2025
DECLARED_STOPS = W22.DECLARED_STOPS     # 2: one per read (SRP1, paper)
PREFIX_POOL = 100                       # n_samples of production's synthetic pools (market and operational)
OP_LOAD_LABELS = ('pc', 'qc', 'pc_flex', 'pc_flex_up', 'pc_flex_down')   # network.py _update_network_with_operational_data
OP_GEN_LABELS = ('pg', 'qg')

MARKET_RULE = ('rank the 10 three-subsets S of the market-scenario indices {1..5} by |spread_u(S) - TARGET| '
               'ascending, spread_u = horizon (undiscounted day-weighted) 4 h spread of the probability-renormalized '
               'mean price profile, TARGET = W22 five-scenario mean-profile spread (undiscounted); ties (|diff| equal '
               f'within {TIE_TOL}) broken by |spread_r2(S) - TARGET_r2|, then by the lexicographic order of S; select '
               'the first.')
OP_RULE = ('rank the five operation-scenario indices by the PRIMARY metric (system horizon RES availability, '
           'MWh/day, undiscounted day weights, TSO + 3 DSOs, PV + Wind pg * baseMVA summed over 24 h) ascending; '
           'select (lowest, median = 3rd of 5, highest). Secondary metrics are reported, not used.')


class Recorder:
    """Pass-through on SharedResourcesPlanning.read_planning_problem: records `self`, calls the original."""

    def __init__(self):
        from shared_resources_planning import SharedResourcesPlanning
        self.cls = SharedResourcesPlanning
        self.original = SharedResourcesPlanning.read_planning_problem
        self.captured = []

    def install(self):
        rec = self

        def recording(planning_self, *a, **k):
            rec.captured.append(planning_self)
            return rec.original(planning_self, *a, **k)
        self.cls.read_planning_problem = recording

    def uninstall(self):
        self.cls.read_planning_problem = self.original


def _refuse_concurrent():
    ps = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True).stdout
    me = os.getpid()
    others = [l for l in ps.splitlines() if 'p515_s53_3x3_selection.py' in l and '--manifest' not in l
              and int(l.split()[0]) != me and 'python' in l]
    if others:
        raise RuntimeError(f'another copy is running: {others}')


# ======================================================================================
#  operational data extraction (data already read by production; nothing is built)
# ======================================================================================
def extract_operation(planning):
    import numpy as np
    from definitions import GEN_RES_SOLAR, GEN_RES_WIND, GEN_CURTAILLABLE_TYPES
    holders = [('TSO_case9', planning.transmission_network)] + [
        (f'DSO_node{k}_{v.name}', v) for k, v in sorted(planning.distribution_networks.items())]
    out, seeds, other_curt = {}, [], []
    for name, nd in holders:
        out[name] = {'num_oper_scenarios': nd.num_oper_scenarios, 'blocks': {}}
        for y in sorted(nd.years):
            for d in nd.days:
                net = nd.network[y][d]
                n = net.num_oper_scenarios
                pv, wind, load = np.zeros(n), np.zeros(n), np.zeros(n)
                n_pv = n_wind = 0
                gen_ids = []
                for g in net.generators:
                    if g.gen_type in GEN_CURTAILLABLE_TYPES:
                        gen_ids.append(g.gen_id)
                    if g.gen_type == GEN_RES_SOLAR:
                        pv += (np.asarray(g.pg, float) * net.baseMVA).sum(axis=1)
                        n_pv += 1
                    elif g.gen_type == GEN_RES_WIND:
                        wind += (np.asarray(g.pg, float) * net.baseMVA).sum(axis=1)
                        n_wind += 1
                    elif g.gen_type in GEN_CURTAILLABLE_TYPES:
                        other_curt.append((name, y, d, g.gen_id, g.gen_type))
                for ld in net.loads:
                    arr = np.asarray(ld.pd, float) * net.baseMVA
                    if arr.shape != (n, net.num_instants):
                        raise RuntimeError(f'{name} {y}/{d} load {ld.load_id}: pd shape {arr.shape}')
                    load += arr.sum(axis=1)
                out[name]['blocks'][(y, d)] = {'pv': pv, 'wind': wind, 'res': pv + wind, 'load': load,
                                               'n_pv': n_pv, 'n_wind': n_wind, 'n_loads': len(net.loads),
                                               'num_instants': net.num_instants, 'baseMVA': net.baseMVA}
                seeds.append({'net': name, 'year': y, 'day': d, 'seed': net.random_seed,
                              'load_ids': [ld.load_id for ld in net.loads], 'gen_ids': gen_ids})
    return out, seeds, other_curt


def operation_analysis(ops, years, days):
    import numpy as np
    blocks = [(y, d) for y in sorted(years) for d in days]
    ns = {v['num_oper_scenarios'] for v in ops.values()}
    if len(ns) != 1:
        raise RuntimeError(f'networks disagree on num_oper_scenarios: {ns}')
    n = ns.pop()
    w = {b: W22.weights(years, days, b[0], b[1], Y0) for b in blocks}   # (undiscounted, model r2)

    def hz(get, which):
        num = sum(w[b][which] * get(b) for b in blocks)
        return num / sum(w[b][which] for b in blocks)

    nets = list(ops)
    sys_blk = {b: {q: sum(ops[k]['blocks'][b][q] for k in nets) for q in ('pv', 'wind', 'res', 'load')}
               for b in blocks}
    per_index = []
    for s in range(n):
        rec = {'index': s + 1}
        for which, tag in ((0, 'u'), (1, 'r2')):
            for q in ('pv', 'wind', 'res', 'load'):
                rec[f'system_{q}_MWh_per_day_{tag}'] = float(hz(lambda b, q=q: sys_blk[b][q][s], which))
            rec[f'system_res_share_of_load_{tag}'] = rec[f'system_res_MWh_per_day_{tag}'] / rec[f'system_load_MWh_per_day_{tag}']
            rec[f'system_net_load_MWh_per_day_{tag}'] = rec[f'system_load_MWh_per_day_{tag}'] - rec[f'system_res_MWh_per_day_{tag}']
        rec['per_network_u'] = {k: {q: float(hz(lambda b, k=k, q=q: ops[k]['blocks'][b][q][s], 0))
                                    for q in ('res', 'load')} for k in nets}
        for k in nets:
            r = rec['per_network_u'][k]
            r['res_share_of_load'] = r['res'] / r['load'] if r['load'] else None
        per_index.append(rec)
    primary = 'system_res_MWh_per_day_u'
    ranked = sorted(per_index, key=lambda r: (r[primary], r['index']))
    selected = [ranked[0]['index'], ranked[n // 2]['index'], ranked[-1]['index']]

    def order_by(key):
        return [r['index'] for r in sorted(per_index, key=lambda r: (r[key], r['index']))]
    orders = {k: order_by(k) for k in ('system_res_MWh_per_day_u', 'system_res_MWh_per_day_r2',
                                       'system_res_share_of_load_u', 'system_net_load_MWh_per_day_u',
                                       'system_load_MWh_per_day_u', 'system_pv_MWh_per_day_u',
                                       'system_wind_MWh_per_day_u')}
    per_net_orders = {k: [r['index'] for r in sorted(per_index, key=lambda r: (r['per_network_u'][k]['res'], r['index']))]
                      for k in nets}

    def rel_range(key):
        vals = [r[key] for r in per_index]
        return (max(vals) - min(vals)) / (sum(vals) / len(vals))
    # per-block ranks on block system RES (1 = lowest) and Kendall's W
    block_ranks = {}
    for b in blocks:
        v = sys_blk[b]['res']
        order = sorted(range(n), key=lambda s: (v[s], s))
        rk = [0] * n
        for pos, s in enumerate(order):
            rk[s] = pos + 1
        block_ranks[f'{b[0]}/{b[1]}'] = rk
    m = len(blocks)
    R = [sum(block_ranks[k][s] for k in block_ranks) for s in range(n)]
    Rbar = sum(R) / n
    S = sum((r - Rbar) ** 2 for r in R)
    kendall_w = 12.0 * S / (m ** 2 * (n ** 3 - n))
    sel_low, sel_mid, sel_high = selected
    consistency = {
        'blocks_where_selected_lowest_is_lowest': sum(1 for k in block_ranks if block_ranks[k][sel_low - 1] == 1),
        'blocks_where_selected_highest_is_highest': sum(1 for k in block_ranks if block_ranks[k][sel_high - 1] == n),
        'blocks_where_selected_median_is_3rd': sum(1 for k in block_ranks if block_ranks[k][sel_mid - 1] == 3),
        'blocks_where_selected_low_below_selected_high': sum(
            1 for k in block_ranks if block_ranks[k][sel_low - 1] < block_ranks[k][sel_high - 1]),
        'num_blocks': m}
    per_block_system = {f'{b[0]}/{b[1]}': {q: [float(x) for x in sys_blk[b][q]] for q in ('res', 'load')}
                        for b in blocks}
    per_block_network = {k: {f'{b[0]}/{b[1]}': {q: [float(x) for x in ops[k]['blocks'][b][q]] for q in ('res', 'load')}
                             for b in blocks} for k in nets}
    return {'num_operation_scenarios': n, 'primary_metric': primary, 'rule': OP_RULE,
            'ranking_primary_ascending': [{'rank': i + 1, **{kk: r[kk] for kk in r if kk != 'per_network_u'}}
                                          for i, r in enumerate(ranked)],
            'per_index': per_index, 'selected_low_mid_high': selected,
            'orders_ascending': orders, 'per_network_res_orders_ascending': per_net_orders,
            'relative_range': {k: rel_range(k) for k in ('system_res_MWh_per_day_u', 'system_load_MWh_per_day_u',
                                                        'system_res_share_of_load_u', 'system_net_load_MWh_per_day_u')},
            'per_block_rank_on_system_res': block_ranks, 'rank_sums': R, 'kendall_W': kendall_w,
            'kendall_W_note': ('W = 1: every block ranks the indices identically; W ~ 0: no agreement. Under independent '
                               f'per-block draws the expectation is about 1/m = {1 / m:.3f}.'),
            'selected_consistency_across_blocks': consistency,
            'network_counts': {k: {'n_pv': ops[k]['blocks'][blocks[0]]['n_pv'], 'n_wind': ops[k]['blocks'][blocks[0]]['n_wind'],
                                   'n_loads': ops[k]['blocks'][blocks[0]]['n_loads'],
                                   'baseMVA': ops[k]['blocks'][blocks[0]]['baseMVA'],
                                   'num_instants': ops[k]['blocks'][blocks[0]]['num_instants']} for k in nets},
            'per_block_system_MWh_per_day': per_block_system, 'per_block_network_MWh_per_day': per_block_network}


# ======================================================================================
#  market subsets
# ======================================================================================
def subset_rows(rd, subset):
    """Per (year, day) rows for the renormalized mean profile of `subset` (0-based indices) and the
    renormalized average of the subset's scenario spreads."""
    import numpy as np
    years, days = rd['years'], rd['days']
    idx = list(subset)
    mp, av = [], []
    for y in sorted(years):
        pm = np.asarray(rd['pm'][y], float)
        pms = pm[idx] / pm[idx].sum()
        for d in days:
            c = rd['prices'][(y, d)]
            wu, wr = W22.weights(years, days, y, d, Y0)
            base = {'year': y, 'day': d, 'weight_undiscounted': wu, 'weight_model_r2': wr}
            mean_c = pms @ c[idx]
            mp.append({**base, **W22.day_spreads(mean_c), 'pm_renormalized': pms.tolist()})
            sp = [W22.day_spreads(c[s]) for s in idx]
            av.append({**base, F_TH: float(sum(pms[i] * sp[i][F_TH] for i in range(len(idx)))),
                       F_MM: float(sum(pms[i] * sp[i][F_MM] for i in range(len(idx))))})
    return mp, av


def gn_h(rows, g, wkey='weight_undiscounted'):
    """W22's growth-normalized horizon average (norm_h in p515_s47_market_spreads.main)."""
    return (sum(r[F_TH] / (1 + g) ** (r['year'] - Y0) * r[wkey] for r in rows) / sum(r[wkey] for r in rows))


def market_analysis(paper_rd, srp_rows, g, target_u, target_r2):
    ns = paper_rd['num_market_scenarios']
    srp_gn = gn_h(srp_rows['s1'], g)
    cands = []
    for S in list(itertools.combinations(range(ns), K)) + [tuple(range(ns))]:
        mp, av = subset_rows(paper_rd, S)
        su = W22.horizon(mp, F_TH, 'weight_undiscounted')
        sr = W22.horizon(mp, F_TH, 'weight_model_r2')
        sg = gn_h(mp, g)
        rec = {'subset': [s + 1 for s in S], 'spread_u': su, 'spread_r2': sr, 'spread_growth_normalized_u': sg,
               'abs_diff_u_to_target': abs(su - target_u), 'signed_diff_u_to_target': su - target_u,
               'abs_diff_r2_to_target_r2': abs(sr - target_r2),
               'avg_of_scenario_spreads_u': W22.horizon(av, F_TH, 'weight_undiscounted'),
               'avg_of_scenario_spreads_r2': W22.horizon(av, F_TH, 'weight_model_r2'),
               'max_minus_min_mean_profile_u': W22.horizon(mp, F_MM, 'weight_undiscounted'),
               'mean_price_u': W22.horizon(mp, 'mean', 'weight_undiscounted'),
               'R_u': su / W22.SRP1_Z4_SPREAD, 'R_r2': sr / W22.SRP1_Z4_SPREAD_R2, 'R_gn': sg / srp_gn,
               'per_year_spread_u': {str(y): W22.horizon(mp, F_TH, 'weight_undiscounted', lambda r, y=y: r['year'] == y)
                                     for y in sorted(paper_rd['years'])},
               'per_block_spread': {f'{r["year"]}/{r["day"]}': r[F_TH] for r in mp}}
        cands.append(rec)
    full = cands.pop()
    ranked = sorted(cands, key=lambda r: (r['abs_diff_u_to_target'], r['abs_diff_r2_to_target_r2'], r['subset']))
    for i, r in enumerate(ranked):
        r['rank'] = i + 1
    ties = [r['subset'] for r in ranked[1:] if abs(r['abs_diff_u_to_target'] - ranked[0]['abs_diff_u_to_target']) <= TIE_TOL]
    return {'rule': MARKET_RULE, 'target_u': target_u, 'target_r2': target_r2, 'srp1_growth_normalized_spread': srp_gn,
            'ranking': ranked, 'selected': ranked[0]['subset'], 'ties_at_top': ties,
            'tie_break_used': bool(ties),
            'margin_winner_to_runner_up_abs_diff': ranked[1]['abs_diff_u_to_target'] - ranked[0]['abs_diff_u_to_target'],
            'full_five': full}


# ======================================================================================
#  realization check (index level)
# ======================================================================================
def prefix_check(paper_rd, seeds):
    import pandas as pd
    from helper_functions import derive_random_seed
    pool = pd.DataFrame({'i': range(PREFIX_POOL)})

    def is_prefix(k):
        a = list(pool.sample(n=K, random_state=k).index)
        b = list(pool.sample(n=5, random_state=k).index)
        return b[:K] == a
    market_seed = derive_random_seed(paper_rd['random_seed'], 'market')
    m_total = m_ok = 0
    for y in sorted(paper_rd['years']):
        for d in paper_rd['days']:
            for kind in ('energy', 'flexibility'):
                m_total += 1
                m_ok += is_prefix(derive_random_seed(market_seed, 'selection', kind, y, str(d)))
    o_total = o_ok = 0
    fails = []
    for s in seeds:
        labels = [('load', str(lid), lab) for lid in s['load_ids'] for lab in OP_LOAD_LABELS] + \
                 [('generator', str(gid), lab) for gid in s['gen_ids'] for lab in OP_GEN_LABELS]
        for lab in labels:
            o_total += 1
            ok = is_prefix(derive_random_seed(s['seed'], *lab))
            o_ok += ok
            if not ok and len(fails) < 20:
                fails.append([s['net'], s['year'], s['day'], *lab])
    return {'note': ('index-level only, dummy frame of the pool size; seeds recomputed with production\'s '
                     'derive_random_seed and the labels of _read_market_data_from_file and '
                     '_update_network_with_operational_data (network.random_seed read from the production objects). '
                     f'Checks whether DataFrame.sample(n={K}) is the first {K} rows of DataFrame.sample(n=5) for the '
                     'same seed. No seed depends on the scenario count.'),
            'market_seeds_checked': m_total, 'market_prefix_true': m_ok,
            'operation_seeds_checked': o_total, 'operation_prefix_true': o_ok, 'first_failures': fails,
            'implication': None}


# ======================================================================================
#  markdown
# ======================================================================================
def _f(x, nd=2):
    return f'{x:,.{nd}f}'


def markdown(res):
    M, O, P = res['market'], res['operation'], res['R_prediction']
    L = [f'# {STAGE}', '', f'Started {res["started_utc"]}; HEAD {res["git_HEAD"]}; script sha256 {res["script_sha256"]}.',
         '', 'Authority: ' + '; '.join(AUTHORITY) + f' (spec sha256 {res["spec_sha256"]}).', '',
         '**Zero solves** (armed SolveProfileGuard(permitted=()), verify(0) failures: '
         f'{res["solve_profile_guard"]["verify_0_failures"]}; counts {res["solve_profile_guard"]["counts"]}). '
         f'**No model construction** (blocked calls: {res["model_construction_blocked_calls"]}; blockers '
         f'{res["model_construction_blockers"]}). Stop point hits: {res["stop_point"]["hits"]} (declared {DECLARED_STOPS}).',
         '', f'Instances: ' + '; '.join(f'{k}: {v["case"]} checksum {v["scenario_checksum"][:16]}... match '
                                          f'{v["checksum_matches"]}' for k, v in res['instances'].items()), '',
         '## 1. Market scenarios (3 of 5)', '', 'Spread definition: ' + W22.SPREAD_FORMULA, '',
         'Renormalization: pm_S[s] = pm[s] / sum_(s in S) pm[s] (= 1/3 each, pm = [0.2] x 5).', '',
         'Rule: ' + M['rule'], '',
         f'TARGET (W22 five-scenario mean profile, undiscounted) = {M["target_u"]!r}; TARGET_r2 = {M["target_r2"]!r}.', '',
         '| rank | subset | 4 h spread undisc. | spread - target | 4 h spread r2 | growth-norm. | avg of scen. spreads | R_u | R_r2 | R_gn |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for r in M['ranking']:
        L.append(f'| {r["rank"]} | {r["subset"]} | {r["spread_u"]:.4f} | {r["signed_diff_u_to_target"]:+.4f} | '
                 f'{r["spread_r2"]:.4f} | {r["spread_growth_normalized_u"]:.4f} | {r["avg_of_scenario_spreads_u"]:.4f} | '
                 f'{r["R_u"]:.4f} | {r["R_r2"]:.4f} | {r["R_gn"]:.4f} |')
    F = M['full_five']
    L.append(f'| (all 5) | {F["subset"]} | {F["spread_u"]:.4f} | {F["signed_diff_u_to_target"]:+.4f} | {F["spread_r2"]:.4f} | '
             f'{F["spread_growth_normalized_u"]:.4f} | {F["avg_of_scenario_spreads_u"]:.4f} | {F["R_u"]:.4f} | '
             f'{F["R_r2"]:.4f} | {F["R_gn"]:.4f} |')
    L += ['', f'**Selected market scenarios: {M["selected"]}.** Tie at the top: {M["ties_at_top"] or "none"} '
          f'(tie-break used: {M["tie_break_used"]}); margin to runner-up in |diff|: {M["margin_winner_to_runner_up_abs_diff"]:.4f} EUR/MWh.', '']
    L += ['## 2. Operation scenarios (3 of 5)', '', 'Rule: ' + O['rule'], '',
          'Network counts: ' + json.dumps(O['network_counts']), '',
          '| rank | index | RES MWh/day (u) | RES r2 | PV | Wind | load MWh/day (u) | RES share of load | net load |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in O['ranking_primary_ascending']:
        L.append(f'| {r["rank"]} | {r["index"]} | {_f(r["system_res_MWh_per_day_u"])} | {_f(r["system_res_MWh_per_day_r2"])} | '
                 f'{_f(r["system_pv_MWh_per_day_u"])} | {_f(r["system_wind_MWh_per_day_u"])} | '
                 f'{_f(r["system_load_MWh_per_day_u"])} | {r["system_res_share_of_load_u"]:.4f} | '
                 f'{_f(r["system_net_load_MWh_per_day_u"])} |')
    L += ['', f'**Selected operation scenarios (low, mid, high RES): {O["selected_low_mid_high"]}.**', '',
          'Orders (ascending) by metric: ' + json.dumps(O['orders_ascending']), '',
          'Per-network RES orders (ascending): ' + json.dumps(O['per_network_res_orders_ascending']), '',
          'Relative range (max - min) / mean across the five: ' + json.dumps({k: round(v, 5) for k, v in O['relative_range'].items()}), '',
          f'Per-block ranks on system RES: Kendall W = {O["kendall_W"]:.4f} ({O["kendall_W_note"]}); rank sums {O["rank_sums"]}; '
          'consistency of the selection: ' + json.dumps(O['selected_consistency_across_blocks']), '',
          '| block | ranks of indices 1..5 (1 = lowest RES) |', '|---|---|']
    for k, v in O['per_block_rank_on_system_res'].items():
        L.append(f'| {k} | {v} |')
    L += ['', 'Distinguishability finding: ' + res['operation_finding'], '']
    L += ['## 3. R prediction for the selected 3 x 3 set -- PREDICTION, recorded before any 3 x 3 run', '',
          P['formula'], '', '| form | selected set | five-scenario set (W22) | selected / five - 1 |', '|---|---|---|---|']
    for k in ('R_u', 'R_r2', 'R_gn'):
        L.append(f'| {k} | {P["selected"][k]:.4f} | {P["five"][k]:.4f} | {P["bias_vs_five"][k]:+.4%} |')
    L += ['', 'PREDICTION: ' + P['prediction'], '', '## 4. Comparison with the five-scenario mean profile', '',
          res['comparison'], '', '## 5. Realization check (how a 3 x 3 case is drawn by production)', '',
          json.dumps(res['prefix_check'], indent=1), '', '## Checks', '', '| check | result |', '|---|---|']
    for k, v in res['checks'].items():
        L.append(f'| {k} | {v} |')
    L += ['', f'ok = {res["ok"]}', '']
    return '\n'.join(L)


# ======================================================================================
#  main
# ======================================================================================
def write_manifest():
    entries = {}
    for dirpath, _dirs, files in os.walk(OUT_DIR):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, OUT_DIR)
            if rel == MANIFEST_NAME:
                continue
            entries[rel] = W22._sha256_file(p)
    path = os.path.join(OUT_DIR, MANIFEST_NAME)
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    with open(path, 'w') as handle:
        json.dump({'created_utc': W22._utc(), 'directory': OUT_REL, 'files': dict(sorted(entries.items())),
                   'script': {'p515_s53_3x3_selection.py': W22._sha256_file(os.path.abspath(__file__)),
                              'p515_s47_market_spreads.py (imported)': W22._sha256_file(os.path.abspath(W22.__file__))}},
                  handle, indent=2)
    print(f'[manifest] {len(entries)} files -> {path}')


def main():
    if '--manifest' in sys.argv:
        write_manifest()
        return 0
    _refuse_concurrent()
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in (RESULTS_NAME, MD_NAME, STDOUT_CAPTURE, MANIFEST_NAME):
        if os.path.exists(os.path.join(OUT_DIR, name)):
            raise RuntimeError(f'write-once: {name} exists in {OUT_REL}')
    if os.path.exists(os.path.join(OUT_DIR, 'production_read')):
        raise RuntimeError('write-once: production_read exists')
    started, t0 = W22._utc(), time.time()
    head = W22._git(['rev-parse', 'HEAD']).strip()
    print(f'[{STAGE}] start {started}; HEAD {head}', flush=True)
    W22.OUT_DIR = OUT_DIR          # W22.production_read writes production's plots under OUT_DIR/production_read
    guard = SolveProfileGuard(permitted=(), label=STAGE).install()
    import numpy as np
    import p56a_oracle as O
    blocks = W22.Blocks()
    blocks.install()
    recorder = Recorder()
    recorder.install()
    spec = json.load(open(os.path.join(REPO, SPEC_REL)))
    res = {'stage': STAGE, 'authority': AUTHORITY, 'started_utc': started, 'git_HEAD': head,
           'script_sha256': W22._sha256_file(os.path.abspath(__file__)),
           'imported_w22_script_sha256': W22._sha256_file(os.path.abspath(W22.__file__)), 'w22_commit': W22_COMMIT,
           'spec': SPEC_REL, 'spec_sha256': W22._sha256_file(os.path.join(REPO, SPEC_REL)),
           'spec_ruling3_3x3': spec['ruling3_3x3']}
    checks = {}
    try:
        with open(os.path.join(OUT_DIR, STDOUT_CAPTURE), 'w') as sink:
            print('[read] SRP1 ...', flush=True)
            srp_rd = W22.production_read(W22.SRP1_CASE_REL, 'srp1', sink)
            srp_planning = recorder.captured[-1]
            del srp_planning
            print('[read] paper ...', flush=True)
            paper_rd = W22.production_read(W22.PAPER_CASE_REL, 'paper', sink)
            paper_planning = recorder.captured[-1]
        print('[extract] operational data ...', flush=True)
        ops, seeds, other_curt = extract_operation(paper_planning)
        recorder.captured.clear()
        del paper_planning
        g = W22.energy_growth_factor()
        print('[prefix] realization check ...', flush=True)
        prefix = prefix_check(paper_rd, seeds)
    finally:
        recorder.uninstall()
        blocks.uninstall()
        guard.uninstall()
    failures = guard.verify(0)
    res['solve_profile_guard'] = {'permitted': [], 'counts': guard.counts, 'verify_0_failures': failures}
    res['model_construction_blockers'] = [f'{c.__name__}.{a}' for c, a in blocks.targets]
    res['model_construction_blocked_calls'] = blocks.blocked_calls
    res['stop_point'] = {'target': f'{blocks.stop_target[0].__name__}.{blocks.stop_target[1]}',
                         'hits': blocks.stop_hits, 'declared': DECLARED_STOPS,
                         'site': W22._one_line('shared_resources_planning.py', 'shared_ess_data.read_parameters_from_file()'),
                         'operational_data_read_before_stop': [
                             W22._one_line('shared_resources_planning.py', 'distribution_network.read_network_data()'),
                             W22._one_line('shared_resources_planning.py', 'transmission_network.read_network_data()')]}
    checks['solve_guard_verify_0'] = not failures
    checks['no_model_construction_call'] = len(blocks.blocked_calls) == 0
    checks['stop_point_hits_exactly_declared'] = blocks.stop_hits == DECLARED_STOPS
    checks['no_curtailable_generator_other_than_pv_wind'] = not other_curt

    expected = {'srp1': (O.CANONICAL_CHECKSUM, 'p56a_oracle.CANONICAL_CHECKSUM'),
                'paper': (W22.PAPER_CHECKSUM_RECORDED, W22.PAPER_CHECKSUM_SOURCE)}
    inst = {}
    for k, rd in (('srp1', srp_rd), ('paper', paper_rd)):
        inst[k] = {x: rd[x] for x in ('case', 'case_sha256', 'years', 'days', 'num_market_scenarios', 'random_seed',
                                      'discount_factor', 'scenario_checksum', 'market_scenario_checksum', 'pm')}
        inst[k]['expected_checksum'], inst[k]['expected_checksum_source'] = expected[k]
        inst[k]['checksum_matches'] = rd['scenario_checksum'] == expected[k][0]
        checks[f'{k}_scenario_checksum_matches'] = inst[k]['checksum_matches']
        checks[f'{k}_prices_bound_to_every_block_are_planning_object'] = rd['binding']['tso_and_dso_price_array_is_planning_object']
        checks[f'{k}_network_pm_is_planning_prob_market_scenarios'] = rd['binding']['pm_is_planning_object']
        checks[f'{k}_no_congestion_management_network'] = all(
            v != W22.OBJ_CONGESTION_MANAGEMENT for v in rd['binding']['obj_types'].values())
    checks['paper_pm_all_0_2'] = all(np.allclose(v, [0.2] * 5) for v in paper_rd['pm'].values())
    res['instances'] = inst

    # ---- W22 identity: the five-scenario figures must reproduce W22's committed values ----
    w22 = json.load(open(os.path.join(REPO, W22_JSON_REL)))
    w22v = w22['ratios']['values']
    target_u = w22v['paper_mean_profile_4h_spread_undiscounted']
    target_r2 = w22['paper']['summary']['mean_profile']['horizon_model_r2'][F_TH]
    srp_rows, srp_sum = W22.instance_spreads(srp_rd)
    checks['srp1_recomputed_spread_equals_Z4_97.99_to_1e-9'] = abs(
        srp_sum['s1']['horizon_undiscounted'][F_TH] - W22.SRP1_Z4_SPREAD) <= REPRO_TOL
    checks['srp1_recomputed_spread_r2_equals_Z4_97.02_to_1e-9'] = abs(
        srp_sum['s1']['horizon_model_r2'][F_TH] - W22.SRP1_Z4_SPREAD_R2) <= REPRO_TOL
    del srp_sum

    M = market_analysis(paper_rd, srp_rows, g, target_u, target_r2)
    F = M['full_five']
    checks['five_set_spread_u_reproduces_W22_to_1e-9'] = abs(F['spread_u'] - target_u) <= REPRO_TOL
    checks['five_set_R_u_reproduces_W22_to_1e-9'] = abs(F['R_u'] - w22v['R = mean_profile / SRP1 (undiscounted)']) <= REPRO_TOL
    checks['five_set_R_r2_reproduces_W22_to_1e-9'] = abs(F['R_r2'] - w22v['R_r2 = mean_profile_r2 / SRP1_r2']) <= REPRO_TOL
    checks['five_set_R_gn_reproduces_W22_to_1e-9'] = abs(
        F['R_gn'] - w22v['supplementary: growth-normalized (2025 level) R = paper mean_profile / SRP1, undiscounted weights']) <= REPRO_TOL
    checks['ten_market_subsets_ranked'] = len(M['ranking']) == 10
    res['w22_reference'] = {'json': W22_JSON_REL, 'json_sha256': W22._sha256_file(os.path.join(REPO, W22_JSON_REL)),
                            'target_u': target_u, 'target_r2': target_r2,
                            'R_u': w22v['R = mean_profile / SRP1 (undiscounted)'],
                            'R_r2': w22v['R_r2 = mean_profile_r2 / SRP1_r2'],
                            'R_gn': w22v['supplementary: growth-normalized (2025 level) R = paper mean_profile / SRP1, undiscounted weights']}
    res['energy_growth_factor_g'] = g
    res['market'] = M

    OP = operation_analysis(ops, paper_rd['years'], paper_rd['days'])
    res['operation'] = OP
    checks['five_operation_indices_ranked'] = len(OP['ranking_primary_ascending']) == 5
    checks['operation_selection_three_distinct'] = len(set(OP['selected_low_mid_high'])) == 3
    rr = OP['relative_range']
    sel = OP['selected_low_mid_high']
    share_order = OP['orders_ascending']['system_res_share_of_load_u']
    res['operation_finding'] = (
        f'Relative range across the five indices: RES {rr["system_res_MWh_per_day_u"]:.4%}, load '
        f'{rr["system_load_MWh_per_day_u"]:.4%}, RES share of load {rr["system_res_share_of_load_u"]:.4%}, net load '
        f'{rr["system_net_load_MWh_per_day_u"]:.4%}. Kendall W of the per-block RES ranks = {OP["kendall_W"]:.4f} '
        f'(about {1 / 20:.3f} expected with no agreement). Primary-metric order {OP["orders_ascending"]["system_res_MWh_per_day_u"]}; '
        f'RES-share order {share_order}; net-load order {OP["orders_ascending"]["system_net_load_MWh_per_day_u"]}; '
        f'load order {OP["orders_ascending"]["system_load_MWh_per_day_u"]}. The selected (low, mid, high) = {sel} is '
        f'{"the same" if [share_order[0], share_order[2], share_order[-1]] == sel else "NOT the same"} as the '
        f'(low, mid, high) of the RES-share order ({[share_order[0], share_order[2], share_order[-1]]}). '
        'Operation-scenario indices are not coherent RES states: production draws every load and every generator row '
        'with its own seed per network and per (year, day) block (network.py _update_network_with_operational_data; '
        'network_data.py realization seed per block), so an index is a bundle of independent per-element draws and '
        'its horizon RES level is an aggregate of those draws.')

    # ---- R prediction ----
    sel_m = next(r for r in M['ranking'] if r['subset'] == M['selected'])
    P = {'formula': (f'R_u = spread_u(S*) / {W22.SRP1_Z4_SPREAD!r} (SRP1 Z4, undiscounted day weights); R_r2 = '
                     f'spread_r2(S*) / {W22.SRP1_Z4_SPREAD_R2!r} (model block weight w / 1.02^(y-2025)); R_gn = '
                     f'[sum w_u {F_TH}(y,d) / (1+g)^(y-2025) / sum w_u](S*) / same for SRP1, g = {g!r}. spread(S*) is '
                     'the horizon 4 h spread of the probability-renormalized mean price profile of the selected '
                     'market scenarios. The operation-scenario selection does not enter R (R is a price-only measure).'),
         'selected_market_scenarios': M['selected'], 'selected_operation_scenarios': sel,
         'selected': {'spread_u': sel_m['spread_u'], 'spread_r2': sel_m['spread_r2'],
                      'spread_growth_normalized_u': sel_m['spread_growth_normalized_u'],
                      'R_u': sel_m['R_u'], 'R_r2': sel_m['R_r2'], 'R_gn': sel_m['R_gn']},
         'five': {'spread_u': F['spread_u'], 'spread_r2': F['spread_r2'],
                  'spread_growth_normalized_u': F['spread_growth_normalized_u'],
                  'R_u': F['R_u'], 'R_r2': F['R_r2'], 'R_gn': F['R_gn']}}
    P['bias_vs_five'] = {k: P['selected'][k] / P['five'][k] - 1 for k in ('R_u', 'R_r2', 'R_gn')}
    P['prediction'] = (f'for the 3 x 3 instance (market {M["selected"]}, operation {sel}), the first-order arbitrage value '
                       f'per MWh relative to SRP1 is R_u = {sel_m["R_u"]:.4f}, R_r2 = {sel_m["R_r2"]:.4f} (the form of the '
                       f'confirmed R = 0.937), R_gn = {sel_m["R_gn"]:.4f}; recorded before any 3 x 3 run exists.')
    res['R_prediction'] = P
    res['comparison'] = (
        f'Selected-set renormalized mean-profile 4 h spread {sel_m["spread_u"]:.4f} vs five-scenario {F["spread_u"]:.4f} EUR/MWh '
        f'(undiscounted; difference {sel_m["spread_u"] - F["spread_u"]:+.4f}, {P["bias_vs_five"]["R_u"]:+.4%}); r2 '
        f'{sel_m["spread_r2"]:.4f} vs {F["spread_r2"]:.4f} ({P["bias_vs_five"]["R_r2"]:+.4%}); growth-normalized '
        f'{sel_m["spread_growth_normalized_u"]:.4f} vs {F["spread_growth_normalized_u"]:.4f} ({P["bias_vs_five"]["R_gn"]:+.4%}). '
        f'On the first-order (spread-proportional) model the reduction is expected to bias the arbitrage value '
        f'{"UP" if sel_m["spread_r2"] > F["spread_r2"] else "DOWN"} relative to the full 5 x 5 instance by about '
        f'{abs(P["bias_vs_five"]["R_r2"]):.2%} (r2 form). Bracket: average of the selected scenarios\' own spreads '
        f'{sel_m["avg_of_scenario_spreads_u"]:.4f} vs five-scenario {F["avg_of_scenario_spreads_u"]:.4f} '
        f'({sel_m["avg_of_scenario_spreads_u"] / F["avg_of_scenario_spreads_u"] - 1:+.4%}). This covers price only; the '
        'operation-scenario reduction (and any interaction with network constraints) is not captured by R.')

    nat = next(r for r in M['ranking'] if r['subset'] == [1, 2, 3])
    prefix['implication'] = (
        f'market prefix {prefix["market_prefix_true"]}/{prefix["market_seeds_checked"]}, operation prefix '
        f'{prefix["operation_prefix_true"]}/{prefix["operation_seeds_checked"]}. If all true, a case derived with '
        f'NumMarketScenarios = {K} and num_operation_scenarios = {K} (the p515_s44_scale_measurement.derive_case route used '
        f'for the 2 x 2 pilot) holds exactly indices [1, 2, 3] in both dimensions, NOT the selected market {M["selected"]} x '
        f'operation {sel}. Indices [1, 2, 3] rank {nat["rank"]} of 10 on the market rule (spread_u {nat["spread_u"]:.4f}, '
        f'R_r2 {nat["R_r2"]:.4f}). Realizing the selected set needs a selection mechanism the derive route does not have.')
    checks['prefix_check_ran_market'] = prefix['market_seeds_checked'] == 2 * len(paper_rd['years']) * len(paper_rd['days'])
    checks['prefix_check_ran_operation'] = prefix['operation_seeds_checked'] > 0
    res['prefix_check'] = prefix

    res['checks'] = checks
    ok = all(checks.values())
    res['ok'] = ok
    res['elapsed_s'] = time.time() - t0
    post = W22._git(['status', '--porcelain'])
    res['post_git_status_tracked_changes'] = [l for l in post.splitlines() if not l.startswith('??')]
    res['ended_utc'] = W22._utc()
    res = W22._jsonable(res)
    with open(os.path.join(OUT_DIR, RESULTS_NAME), 'w') as handle:
        json.dump(res, handle, indent=1)
    with open(os.path.join(OUT_DIR, MD_NAME), 'w') as handle:
        handle.write(markdown(res))
    print('[market ranking] ' + json.dumps([(r['rank'], r['subset'], round(r['spread_u'], 4),
                                             round(r['signed_diff_u_to_target'], 4)) for r in M['ranking']]))
    print(f'[market selected] {M["selected"]}; ties {M["ties_at_top"]}')
    print('[operation ranking] ' + json.dumps([(r['rank'], r['index'], round(r['system_res_MWh_per_day_u'], 3),
                                                round(r['system_load_MWh_per_day_u'], 3))
                                               for r in OP['ranking_primary_ascending']]))
    print(f'[operation selected] {sel}; Kendall W {OP["kendall_W"]:.4f}')
    print('[R prediction] ' + json.dumps({k: P[k] for k in ('selected', 'five', 'bias_vs_five')}))
    print('[prefix] ' + prefix['implication'])
    print('[checks] ' + json.dumps(checks, indent=1))
    print(f'[{STAGE}] ok={ok}; guard counts {guard.counts}; blocked {blocks.blocked_calls}; stop hits {blocks.stop_hits}; '
          f'elapsed {res["elapsed_s"]:.1f} s', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
