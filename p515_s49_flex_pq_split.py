"""P5.15 Addenda 33-34 (task W29) -- flexibility split (congestion relief vs economic shifting) and the P/Q split
of the storage's value and the DSOs' flexibility saving. ZERO SOLVES, no model construction, read-only.

Armed `SolveProfileGuard(permitted=())` is installed before any Pyomo / production import; `verify(0)` is checked at
the end (exact count: 0 solves, 0 solver launches). Nothing is built or solved: the certified models persisted by
the s48_x0_capture evaluation (`certified_models.pkl`, sha256 03b62593...) are unpickled ONCE and their terminal
IPOPT primal / dual values are read; every other input is a record file.

Authority: PLANNER_BRIEF_2026-09-13.md Addenda 33-34; frozen spec v19 data/SRP1/Results/P515S49/
frozen_s49_spec_v19_f8adcc97.json `zero_solve_items` Z33_flex_split and Z34_pq_split; Planner task W29.

INSTANCES (every artifact records them by candidate key)
  x = 0     : s48_x0_capture:x0, candidate_key 8435c718..., certified, 132 cycles, persisted models (the ONLY
              persisted terminal models used). Baseline ageing declaration (C2 + phi_cal 0.985 + soh_min 0.70);
              Q(0) is ageing-independent (W21 2466401d).
  UNIT      : s47_recert:n7_4h_e1 (S2 record), node 7 0.25 MVA / 1.0 MWh, 2025, nodes 5/9 empty,
              candidate_key db77e154..., baseline C2 + phi_cal 0.985 + soh_min 0.70. RECORDS ONLY -- no terminal
              models exist for it (not persisted); S3's duplicate record (s47_a1a_baseline, same eval key) is
              compared field by field.

DEFINITIONS AND FORMULAS (also written into the JSON and the Markdown)
  Blocks b = (year, day); weights w_r2 = admm_block_weight = num_years * num_days / 1.02^(y - 2025) (the convention
  of Q) and w_u = num_years * num_days (undiscounted); dt = 1 h.
  Flexibility cost of a DSO block (production objective term, model_construction_helpers.flexibility_cost):
      FC_b = sum_{c in fl_reg loads, not TSO ADN} sum_p c_flex[p] * baseMVA * (flex_p_down[c,p] + flex_q_down[c,p])
  read EXACTLY from each persisted DSO block as the linear representation of flex_cost_scenario[0,0] (coefficient
  of flex_p_down[c,p] = coefficient of flex_q_down[c,p] = c_flex[p] * baseMVA -- checked); flex_p_up / flex_q_up
  carry no cost. Split per slot p: FC_P = sum_c coef * flex_p_down, FC_Q = sum_c coef * flex_q_down.
  Node-7 binding set S = the (year, day, period) slots where the node-7 DSO interface apparent flow is >= 99 % of
  its 100 MVA rating (W19 Z1 rule and slot set, dd3afa6e; recomputed here on both records and required equal).
  Congestion relief (CR) = the part of FC in slots in S; economic shifting (ES) = the rest. The same slot set is
  applied to all three DSOs (the node-7 binding periods; DSOs 5 and 9 have no binding interface of their own).
  UNIT (block totals only): FC_b(x) = unweighted flexibility_cost_internal of the S2 record. For a block with no
  slot in S, CR_b(x) = 0 exactly. For a block with slots in S only an interval is available:
      CR_b(x) in [max(-eps_b, FC_b(x) - UB_ES_b), min(FC_b(x) + eps_b, UB_CR_b)],
      UB_CR_b = sum_{p in S_b} sum_c coef * (ub(flex_p_down) + ub(flex_q_down)), UB_ES_b the same over p not in S_b,
      eps_b = sum_c sum_p coef * 2 * RELAX (IPOPT bound relaxation 1e-8 p.u. per variable, default options),
  ub read from the x = 0 models; they come from the DN data (network.py flexibility bounds) and do not depend on
  the investment candidate.
  P/Q at the unit: FC_Q,b(x) in [-eps_q,b, UB_Q,b], UB_Q,b = sum_c sum_p coef * ub(flex_q_down) (the reactive
  flexibility bound is EQUALITY_TOLERANCE by construction, network.py); FC_P,b(x) = FC_b(x) - FC_Q,b(x).
  Storage marginal values (x = 0, EXACT duals of the persisted terminal TSO blocks):
      LMP_P,b[p] = dual(node_balance_p[bus 7]) / baseMVA  [EUR/MWh], LMP_Q,b[p] = dual(node_balance_q[bus 7]) /
      baseMVA [EUR/MVArh]; sign: + objective cost of one more unit of LOAD at the bus (W25/W27 convention).
  With the unit (records only): the W25 identity (b7aca555) LMP_P = pi + sigma*lambda_dso_p/(w*s_base*R*baseMVA)
  and its reactive analogue LMP_Q = sigma*lambda_dso_q/(w*s_base*R*baseMVA) (no reactive settlement / price term),
  both VERIFIED on every persisted x = 0 TSO block against the terminal pf_entry_stride before use.
  Storage value split (ATTRIBUTED, first order at the x = 0 marginal values; the storage's load-convention net
  powers pnet, qnet from the unit's terminal records):
      A_P = sum_b w_b sum_p LMP_P,b[p] * (-pnet[p]) * dt   (pnet = pch - pdch, ESSO terminal capture; = W25 A0)
      A_Q = sum_b w_b sum_p LMP_Q,b[p] * (-qnet[p]) * dt   (qnet = TSO copy of the ESS consensus, ess stride)
      remainder = value - A_P - A_Q (system response incl. the DSO saving, and ADMM stopping)
  value = Q(0) - Q(x), gross_operational_cost, settlement excluded.
  Voltage: an entry is AT the 1.1 pu bound when v_max - v <= 1e-6 pu (the production summary convention of
  interface_voltage_terminal.json); counts are also given at 1e-5, 1e-4, 1e-3 pu. x = 0 from the models (TSO
  vmag_sqr, the voltage_magnitude_upper_cons multiplier and the complementarity product s * |z|, s = the row's
  slack in pu^2), cross-checked against the x = 0 record's interface_voltage_terminal.json; the unit from its record.

Output (write-once, new directory) data/SRP1/Results/P515S49/flex_pq_split/:
    flex_pq_split.json, flex_pq_split.md, launch.log, manifest_sha256.json

EXACT COMMAND (repo root; attached; alone; both streams captured):
    mkdir data/SRP1/Results/P515S49/flex_pq_split && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_flex_pq_split.py \\
        > data/SRP1/Results/P515S49/flex_pq_split/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_flex_pq_split.py --manifest
Development runs only: W29_SCRATCH_OUT=<dir outside the repo> (recorded in the output).
"""
import gc
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W29 flexibility and P/Q splits (zero solves)').install()

import pyomo.environ as pe  # noqa: E402
from pyomo.repn import generate_standard_repn  # noqa: E402

STAGE = 'P5.15 Addenda 33-34 W29 -- flexibility (congestion relief / economic shifting) and P/Q splits (zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addenda 33-34',
             'data/SRP1/Results/P515S49/frozen_s49_spec_v19_f8adcc97.json zero_solve_items Z33_flex_split, Z34_pq_split',
             'Planner task W29, 2026-09-22']
SPEC_REL = 'data/SRP1/Results/P515S49/frozen_s49_spec_v19_f8adcc97.json'
SPEC_SHA_PREFIX = 'f8adcc97'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S49', 'flex_pq_split')
OUT_DIR = os.environ.get('W29_SCRATCH_OUT') or os.path.join(REPO, OUT_REL)
RESULTS_NAME = 'flex_pq_split.json'
MD_NAME = 'flex_pq_split.md'
MANIFEST_NAME = 'manifest_sha256.json'
LOCK_NAME = '.w29.lock'

X0_REL = 'data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0'
X0_CAMPAIGN_MANIFEST = 'data/SRP1/Results/P515S48/x0_capture/campaign_manifest_sha256.json'
PICKLE_REL = X0_REL + '/certified_models.pkl'
PICKLE_SHA = '03b62593a23f748c819f18dce52c88c6a9802b8af3c52033ae09a3b88d10afce'
UNIT_REL = 'data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1'
UNIT_DUP_REL = 'data/SRP1/Results/P515S47/campaign_s47_a1a_baseline/evals/bd504ecf5a288d44_n7_4h_e1'
KEY_X0 = '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57'
KEY_UNIT = 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a'
A0_X0_REL = 'data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0'   # W25's x = 0 record
CASE_REL = 'data/SRP1/SRP1.json'
MARKET_REL = 'data/SRP1/MarketData/SRP1_market_data.xlsx'
CASE9_FMT = 'data/SRP1/case9/case9_{y}.json'
W19_JSON_REL = 'data/SRP1/Results/P515S46/zero_solve_reports/zero_solve_reports.json'
W19_COMMIT = 'dd3afa6e'
W25_JSON_REL = 'data/SRP1/Results/P515S47/tso_marginal_cost/tso_marginal_cost.json'
W25_COMMIT = 'b7aca555'
SNAP_REL = UNIT_REL + '/results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl'
SNAP_YEAR, SNAP_DAY, SNAP_STRIDE_CYCLE = '2025', 'Summer', 6

NODE = 7
ADN = (5, 7, 9)
DT_H = 1.0
FIRST_YEAR = 2025
AT_RATING = 0.99                       # W19 Z1 convention
S_RATED_MVA = 0.25                     # the unit's converter rating (checked against the record)

# Tolerances, stated before the committed run:
TOL_BOUND_PU = 1e-6                    # "at the 1.1 pu bound" (interface_voltage_terminal summary convention)
TOL_LADDER_PU = (1e-6, 1e-5, 1e-4, 1e-3)
RELAX = 1e-8                           # IPOPT default bound_relax_factor (not set by production), p.u. per variable
IDENT_TOL = 1e-4                       # EUR/MWh (EUR/MVArh): KKT identities on the persisted blocks (W25's tolerance)
REPRO_REL = 1e-9                       # model flexibility cost vs the record's component level (relative)
REPRO_ABS = 1e-6                       # EUR, absolute floor for the same
COEF_TOL = 1e-9                        # EUR/p.u.: P and Q coefficient equality / uniformity across loads
VOLT_XCHK = 1e-9                       # pu: model voltage vs the x = 0 record's interface_voltage_terminal
W25_A0_TOL = 0.1                       # EUR: recomputed A_P vs W25's committed A0 (duals vs identity series)
DELTA_INTERIOR_MW = 1e-3               # interface_delta margin to its bound for the identity to hold
SUM_TOL_EUR = 0.01

OBJ_CONV = ('Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); '
            'value = Q(0) - Q(x) (positive = the storage lowers operating cost); terminal salvage 0 on both records '
            '(gross = net); EUR; model block weights num_years * num_days / 1.02^(y - 2025) (r2) unless stated.')
YEARS = (2025, 2030, 2035)
DAYS = ('Spring', 'Summer', 'Autumn', 'Winter')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(msg, flush=True)


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _lines(rel, needle):
    with open(os.path.join(REPO, rel)) as handle:
        return [i + 1 for i, line in enumerate(handle) if needle in line]


def _one_line(rel, needle):
    found = _lines(rel, needle)
    if len(found) != 1:
        raise RuntimeError(f'citation needle not unique in {rel}: {needle!r} -> {found}')
    return f'{rel}:{found[0]}'


def _fmt(x, nd=2):
    return f'{x:,.{nd}f}'


def _pct(xs, q):
    s = sorted(xs)
    if not s:
        return None
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ======================================================================================================================
#  capture-path checklist (CLAUDE.md: assert the capture path before executing)
# ======================================================================================================================
def _manifest_hashes(rel):
    man = _load(rel)
    flat = {}

    def _collect(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and len(v) == 64 and k.startswith('data/'):
                    flat[k] = v
                elif isinstance(v, dict) and 'sha256' in v and k.startswith('data/'):
                    flat[k] = v['sha256']
                else:
                    _collect(v)
    _collect(man)
    return flat


def capture_checklist():
    chk, info = {}, {'files': {}}
    rec_x0 = _load(X0_REL + '/evaluation_record.json')
    rec_u = _load(UNIT_REL + '/evaluation_record.json')
    rec_d = _load(UNIT_DUP_REL + '/evaluation_record.json')
    cyc_x0, cyc_u = int(rec_x0['cycles_run']), int(rec_u['cycles_run'])
    needed = {
        'x0': (X0_REL, [PICKLE_REL, X0_REL + '/evaluation_record.json', X0_REL + '/component_levels_terminal.json',
                        X0_REL + '/interface_settlement_detail_s31c.json', X0_REL + '/interface_voltage_terminal.json',
                        X0_REL + '/pf_entry_stride_s39_D.jsonl', X0_REL + '/ess_entry_stride_baseline.jsonl']),
        'unit': (UNIT_REL, [UNIT_REL + '/evaluation_record.json', UNIT_REL + '/component_levels_terminal.json',
                            UNIT_REL + '/interface_settlement_detail_s31c.json',
                            UNIT_REL + '/interface_voltage_terminal.json', UNIT_REL + '/pf_entry_stride_s39_D.jsonl',
                            UNIT_REL + '/ess_entry_stride_baseline.jsonl',
                            UNIT_REL + f'/esso_capture/s39_D/node{NODE}_cycle{cyc_u:03d}.jsonl', SNAP_REL]),
        'unit_duplicate': (UNIT_DUP_REL, [UNIT_DUP_REL + '/evaluation_record.json',
                                          UNIT_DUP_REL + '/component_levels_terminal.json',
                                          UNIT_DUP_REL + '/interface_voltage_terminal.json',
                                          UNIT_DUP_REL + '/ess_entry_stride_baseline.jsonl']),
        'a0_x0': (A0_X0_REL, [A0_X0_REL + '/evaluation_record.json', A0_X0_REL + '/component_levels_terminal.json']),
    }
    static = [SPEC_REL, CASE_REL, MARKET_REL, W19_JSON_REL, W25_JSON_REL] + [CASE9_FMT.format(y=y) for y in YEARS]
    all_paths = [p for _, (_, ps) in needed.items() for p in ps] + static
    tracked = set(_git(['ls-files', '--', *all_paths]).splitlines())
    for name, (rel, paths) in needed.items():
        man_rel = rel + '/child_manifest_sha256.json'
        flat = _manifest_hashes(man_rel)
        chk[f'{name}_child_manifest_tracked'] = man_rel in set(_git(['ls-files', '--', man_rel]).splitlines())
        for p in paths:
            exists = os.path.isfile(os.path.join(REPO, p))
            sha = _sha256(os.path.join(REPO, p)) if exists else None
            settled = ('git' if p in tracked else
                       ('committed child manifest hash' if flat.get(p) == sha and sha is not None else 'UNSETTLED'))
            info['files'][p] = {'sha256': sha, 'tracked': p in tracked, 'settled_by': settled,
                                'child_manifest_sha256': flat.get(p)}
            chk[f'exists::{p}'] = exists
            chk[f'settled::{p}'] = settled != 'UNSETTLED'
    for p in static:
        exists = os.path.isfile(os.path.join(REPO, p))
        info['files'][p] = {'sha256': _sha256(os.path.join(REPO, p)) if exists else None, 'tracked': p in tracked,
                            'settled_by': 'git' if p in tracked else 'UNSETTLED'}
        chk[f'tracked::{p}'] = p in tracked
    camp = _manifest_hashes(X0_CAMPAIGN_MANIFEST)
    chk['x0_campaign_manifest_tracked'] = X0_CAMPAIGN_MANIFEST in set(_git(['ls-files', '--', X0_CAMPAIGN_MANIFEST]).splitlines())
    chk['pickle_sha256_is_03b62593'] = info['files'][PICKLE_REL]['sha256'] == PICKLE_SHA
    chk['pickle_sha256_in_committed_campaign_manifest'] = camp.get(PICKLE_REL) == PICKLE_SHA
    chk['spec_sha256_prefix_f8adcc97'] = info['files'][SPEC_REL]['sha256'].startswith(SPEC_SHA_PREFIX)
    chk['x0_candidate_key'] = rec_x0['candidate_key'] == KEY_X0
    chk['unit_candidate_key'] = rec_u['candidate_key'] == KEY_UNIT
    chk['unit_duplicate_candidate_key'] = rec_d['candidate_key'] == KEY_UNIT
    chk['x0_certified'] = rec_x0['status'] == 'certified'
    chk['unit_certified'] = rec_u['status'] == 'certified'
    chk['unit_duplicate_certified'] = rec_d['status'] == 'certified'
    cand = rec_u['candidate_canonical']
    chk['unit_is_node7_0p25_1p0_2025_nodes_5_9_empty'] = (
        cand['investment_year'] == 2025 and list(cand['nodes']['7']) == [S_RATED_MVA, 1.0]
        and list(cand['nodes']['5']) == [0.0, 0.0] and list(cand['nodes']['9']) == [0.0, 0.0])
    chk['x0_all_nodes_empty'] = all(list(v) == [0.0, 0.0] for v in rec_x0['candidate_canonical']['nodes'].values())
    chk['unit_duplicate_same_cost_and_cycles'] = (rec_d['certified_cost'] == rec_u['certified_cost']
                                                  and rec_d['cycles_run'] == rec_u['cycles_run'])
    ab = rec_u['configuration'].get('ess_ageing_baseline', {})
    info['unit_ageing_declaration'] = ab
    info['x0_ageing_declaration'] = rec_x0['configuration'].get('ess_ageing_baseline', {})
    for name, rec, rel in (('x0', rec_x0, X0_REL), ('unit', rec_u, UNIT_REL), ('unit_duplicate', rec_d, UNIT_DUP_REL)):
        info[name] = {'path': rel, 'campaign_id': rec['campaign_id'], 'candidate_label': rec['candidate_label'],
                      'candidate_key': rec['candidate_key'], 'candidate_canonical': rec['candidate_canonical'],
                      'eval_key': rec.get('eval_key'), 'status': rec['status'], 'cycles_run': int(rec['cycles_run']),
                      'certified_cost_gross_eur': rec['certified_cost'],
                      'terminal_net_operational_recourse': rec.get('terminal_net_operational_recourse'),
                      'bar_eur': rec['bar']['value'],
                      'terminal_objective_change_abs': rec['rule_ten']['terminal_objective_change_abs'],
                      'terminal_step_over_threshold': rec['rule_ten']['terminal_step_over_threshold']}
    info['cycles'] = {'x0': cyc_x0, 'unit': cyc_u}
    return chk, info


def model_checklist(payload):
    """Capture-path assertion on the unpickled models, BEFORE any analysis."""
    chk = {}
    chk['payload_has_tso_and_dso'] = set(payload) == {'tso', 'dso'}
    chk['dso_nodes_5_7_9'] = sorted(payload['dso']) == list(ADN)
    ok_d = ok_t = True
    for y in YEARS:
        for d in DAYS:
            mt = payload['tso'][y][d]
            ok_t = ok_t and all(hasattr(mt, a) for a in ('node_balance_p', 'node_balance_q', 'vmag_sqr',
                                                         'voltage_magnitude_upper_cons', 'dual', 'interface_delta_p',
                                                         'interface_delta_q', 'admm_common_objective_scale',
                                                         'admm_block_weight', 'shared_es_pnet', 'shared_es_qnet'))
            ok_t = ok_t and len(mt.dual) > 0
            for n in ADN:
                md = payload['dso'][n][y][d]
                ok_d = ok_d and all(hasattr(md, a) for a in ('flex_cost_scenario', 'flex_p_up', 'flex_p_down',
                                                             'flex_q_up', 'flex_q_down', 'pij', 'qij', 'vmag_sqr',
                                                             'voltage_magnitude_upper_cons', 'dual'))
                ok_d = ok_d and list(md.flex_cost_scenario.keys()) == [(0, 0)]
    chk['tso_blocks_have_duals_balances_voltage_rows'] = ok_t
    chk['dso_blocks_have_flex_cost_expression_flex_vars_single_scenario'] = ok_d
    return chk


# ======================================================================================================================
#  case / market data
# ======================================================================================================================
def case_blocks():
    case = _load(CASE_REL)
    blocks = []
    for y in YEARS:
        for d in DAYS:
            ny, nd = float(case['Years'][str(y)]), float(case['Days'][d])
            blocks.append({'year': y, 'day': d, 'num_years': ny, 'num_days': nd, 'w_u': ny * nd,
                           'w_r2': ny * nd / (1.0 + float(case['DiscountFactor'])) ** (y - FIRST_YEAR)})
    return case, blocks


def growth_factors():
    import pandas as pd
    g = pd.read_excel(os.path.join(REPO, MARKET_REL), sheet_name='Growth Factors')
    out = {str(r['Growth factors']): float(r['Value, [%]']) for _, r in g.iterrows()}
    return out


def citations():
    srp = 'shared_resources_planning.py'
    mch = 'model_construction_helpers.py'
    net = 'network.py'
    return {
        'market_workbook_read_growth_factors_sheet': _one_line(srp, "'growth_factors': pd.read_excel(filename, sheet_name='Growth Factors'),"),
        'flexibility_growth_factor_read': _one_line(srp, "flexibility_growth_factor = float(growth_factors[growth_factors['Growth factors'] == 'Flexibility']"),
        'flexibility_growth_cumul': _one_line(srp, 'flexibility_growth_cumul = (1 + flexibility_growth_factor) ** (year - initial_year)'),
        'flexibility_profile_sampled_per_year_day': _one_line(srp, "flexibility_selected_profiles = synthetic_profiles['flexibility'][day].sample("),
        'cost_flex_built': _one_line(srp, 'planning_problem.cost_flex[year][day] = np.array(flexibility_selected_profiles * flexibility_growth_cumul)'),
        'cost_flex_bound_to_dso_blocks': _one_line(srp, 'distribution_network.network[year][day].cost_flex = planning_problem.cost_flex[year][day]'),
        'flexibility_cost_def': _one_line(mch, 'def flexibility_cost(model, network, s_m, s_o, params):'),
        'flexibility_cost_term_p_down_plus_q_down_same_c_flex': _one_line(mch, 'flex_cost += c_flex[p] * network.baseMVA * ('),
        'tso_adn_interface_loads_excluded': _one_line(mch, '# excluded from this charge -- it priced movement of the'),
        'flex_p_day_balance_rule': _one_line(mch, 'def flex_energy_balance_p_rule(m, c, s_m, s_o, network, params):'),
        'flex_q_day_balance_NOT_wired': _one_line(net, '# model.flex_energy_balance_q = pe.Constraint('),
        'flex_p_day_balance_wired': _one_line(net, 'model.flex_energy_balance_p = pe.Constraint('),
        'reactive_flex_up_bound_is_EQUALITY_TOLERANCE': _one_line(net, 'load.flexibility.reactive_power.upward = np.ones(pc_flex_up.shape) * EQUALITY_TOLERANCE'),
        'reactive_flex_down_bound_is_EQUALITY_TOLERANCE': _one_line(net, 'load.flexibility.reactive_power.downward = np.ones(pc_flex_down.shape) * EQUALITY_TOLERANCE'),
        'qc_flex_down_bounds': _one_line(mch, 'def qc_flex_down_bounds(m, c, s_m, s_o, p, network, params):'),
        'EQUALITY_TOLERANCE': _one_line('definitions.py', 'EQUALITY_TOLERANCE = '),
        'shared_storage_in_node_balance_load_convention': _one_line(mch, 'Qd += model.shared_es_qnet[e, s_m, s_o, p]'),
        'dso_interface_q_def_storage_at_reference_bus': _one_line(mch, 'def interface_pf_q_distribution_def(m, s_m, s_o, p, network):'),
        'voltage_upper_row': _one_line(mch, 'def voltage_magnitude_upper_cons_rule(m, i, s_m, s_o, p, network, params):'),
        'tso_adn_bus_voltage_slacks_fixed_0': _one_line(srp, 'tso_model[year][day].slack_v_sqr_up[adn_node_idx, s_m, s_o, p].fix(0.00)'),
        'sess_converter_capability_pq_circle': _one_line(mch, 'def sess_converter_capability_rule(m, e, s_m, s_o, p):'),
        'esso_objective_feasibility_penalty_only': _one_line('shared_energy_storage_data.py', 'expr=model.feasibility_penalty,'),
    }


# ======================================================================================================================
#  record readers
# ======================================================================================================================
def stride_terminal(rel, cycle, fname):
    out, last = None, None
    with open(os.path.join(REPO, rel, fname)) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            last = row['cycle']
            if row['cycle'] == cycle:
                out = row
    return out, last


def ess_cycle(rel, cycle):
    out = {}
    with open(os.path.join(REPO, rel, 'ess_entry_stride_baseline.jsonl')) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['cycle'] == cycle:
                for e in row['entries']:
                    out[(e['node_id'], str(e['year']), e['day'], e['power_type'])] = e
    return out


def esso_capture(rel, cycle):
    rows = {}
    with open(os.path.join(REPO, rel, 'esso_capture', 's39_D', f'node{NODE}_cycle{cycle:03d}.jsonl')) as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (YEARS[int(r['y'])], DAYS[int(r['d'])], int(r['p']))
            if key in rows:
                raise RuntimeError(f'duplicate ESSO capture row {key}')
            rows[key] = r
    return rows


def binding_slots(rel):
    det = _load(rel + '/interface_settlement_detail_s31c.json')['interface_reporting_detail'][str(NODE)]
    rating = None
    s = []
    for y, dd in det.items():
        for day, blk in dd.items():
            for p, v in blk['periods'].items():
                mva = math.hypot(v['p_int_dso_expected_mw'], v['q_int_dso_expected_mvar'])
                s.append((int(y), day, int(p), mva))
    return s


# ======================================================================================================================
#  x = 0 models
# ======================================================================================================================
def dso_block(md, B):
    """Exact per-slot flexibility decomposition of one DSO block."""
    expr = md.flex_cost_scenario[0, 0].expr
    repn = generate_standard_repn(expr, compute_values=True)
    if repn.quadratic_vars or repn.nonlinear_expr is not None or abs(repn.constant) > 0:
        raise RuntimeError('flex_cost_scenario is not purely linear')
    coef_p, coef_q = {}, {}
    other = []
    for v, c in zip(repn.linear_vars, repn.linear_coefs):
        comp = v.parent_component().name
        idx = v.index()
        if comp == 'flex_p_down':
            coef_p[(idx[0], idx[3])] = float(c)
        elif comp == 'flex_q_down':
            coef_q[(idx[0], idx[3])] = float(c)
        else:
            other.append(comp)
    loads = sorted({k[0] for k in coef_p})
    periods = sorted(md.periods)
    pq_dev = max(abs(coef_p[k] - coef_q.get(k, float('nan'))) for k in coef_p)
    same_keys = set(coef_p) == set(coef_q)
    prof = [coef_p[(loads[0], p)] / B for p in periods]
    uni_dev = max(abs(coef_p[(c, p)] / B - prof[p]) for c in loads for p in periods)
    rows = []
    min_q = min_p = float('inf')
    max_q_ub = 0.0
    for p in periods:
        r = {'period': p, 'c_flex': prof[p], 'p_up_mw': 0.0, 'p_down_mw': 0.0, 'q_up_mvar': 0.0, 'q_down_mvar': 0.0,
             'cost_p': 0.0, 'cost_q': 0.0, 'ub_cost_p': 0.0, 'ub_cost_q': 0.0, 'eps_p': 0.0, 'eps_q': 0.0,
             'p_up_mw_all_loads': 0.0, 'p_down_mw_all_loads': 0.0}
        for c in md.loads:
            r['p_up_mw_all_loads'] += pe.value(md.flex_p_up[c, 0, 0, p]) * B
            r['p_down_mw_all_loads'] += pe.value(md.flex_p_down[c, 0, 0, p]) * B
        for c in loads:
            pu, pd = pe.value(md.flex_p_up[c, 0, 0, p]), pe.value(md.flex_p_down[c, 0, 0, p])
            qu, qd = pe.value(md.flex_q_up[c, 0, 0, p]), pe.value(md.flex_q_down[c, 0, 0, p])
            r['p_up_mw'] += pu * B
            r['p_down_mw'] += pd * B
            r['q_up_mvar'] += qu * B
            r['q_down_mvar'] += qd * B
            r['cost_p'] += coef_p[(c, p)] * pd
            r['cost_q'] += coef_q[(c, p)] * qd
            r['ub_cost_p'] += coef_p[(c, p)] * md.flex_p_down[c, 0, 0, p].ub
            r['ub_cost_q'] += coef_q[(c, p)] * md.flex_q_down[c, 0, 0, p].ub
            r['eps_p'] += coef_p[(c, p)] * RELAX
            r['eps_q'] += coef_q[(c, p)] * RELAX
            min_q = min(min_q, qd, qu)
            min_p = min(min_p, pd, pu)
            max_q_ub = max(max_q_ub, md.flex_q_down[c, 0, 0, p].ub, md.flex_q_up[c, 0, 0, p].ub)
        rows.append(r)
    fb = [c for c in md.loads if not md.flex_p_down[c, 0, 0, 0].fixed]
    return {'rows': rows, 'n_loads_in_cost': len(loads), 'n_loads_total': len(list(md.loads)),
            'other_vars_in_cost': sorted(set(other)), 'pq_coef_max_abs_dev': pq_dev, 'pq_same_index_set': same_keys,
            'coef_uniform_across_loads_max_abs_dev_eur_per_mwh': uni_dev, 'c_flex': prof,
            'total_from_expression': pe.value(md.flex_cost_scenario[0, 0]),
            'total_flex_cost': pe.value(md.total_flex_cost),
            'min_flex_value_pu': {'p': min_p, 'q': min_q}, 'max_q_flex_ub_pu': max_q_ub,
            'n_unfixed_flex_loads': len(fb)}


def dso_voltage_context(md):
    n_up = n_lo = 0
    slack_sum = 0.0
    nrows = 0
    for k in md.voltage_magnitude_upper_cons:
        i, _, _, p = k
        v = math.sqrt(max(pe.value(md.vmag_sqr[i, 0, 0, p]), 0.0))
        vmax = math.sqrt(md.voltage_magnitude_upper_cons[k].upper)
        nrows += 1
        if vmax - v <= 1e-4:
            n_up += 1
        slack_sum += pe.value(md.slack_v_sqr_up[i, 0, 0, p]) + pe.value(md.slack_v_sqr_down[i, 0, 0, p])
    for k in md.voltage_magnitude_lower_cons:
        i, _, _, p = k
        v = math.sqrt(max(pe.value(md.vmag_sqr[i, 0, 0, p]), 0.0))
        vmin = math.sqrt(md.voltage_magnitude_lower_cons[k].lower)
        if v - vmin <= 1e-4:
            n_lo += 1
    return {'n_rows': nrows, 'n_within_1e-4_of_vmax': n_up, 'n_within_1e-4_of_vmin': n_lo,
            'sum_voltage_slacks_pu2': slack_sum}


def tso_block(mt, case9, pi, stride_x0, blk):
    B = float(case9['baseMVA'])
    order = [n['bus_i'] for n in case9['nodes']]
    sigma = pe.value(mt.admm_common_objective_scale)
    w = pe.value(mt.admm_block_weight)
    out = {'sigma': sigma, 'w_block': w, 'lmp_p': {}, 'lmp_q': {}, 'ident_p_err': 0.0, 'ident_q_err': 0.0,
           'voltage': [], 'delta_min_margin_pu': float('inf')}
    for dn, b in enumerate(ADN):
        i = order.index(b)
        lp, lq = [], []
        for p in range(24):
            yp = mt.dual[mt.node_balance_p[i, 0, 0, p]] / B
            yq = mt.dual[mt.node_balance_q[i, 0, 0, p]] / B
            ep = stride_x0[(b, str(blk['year']), blk['day'], 'p', p)]
            eq = stride_x0[(b, str(blk['year']), blk['day'], 'q', p)]
            R = ep['interface_rating'] / B
            predp = pi[p] + sigma * ep['lambda_dso'] / (w * ep['s_base_dso'] * R * B)
            predq = sigma * eq['lambda_dso'] / (w * eq['s_base_dso'] * R * B)
            out['ident_p_err'] = max(out['ident_p_err'], abs(yp - predp))
            out['ident_q_err'] = max(out['ident_q_err'], abs(yq - predq))
            lp.append(yp)
            lq.append(yq)
            for var in (mt.interface_delta_p[dn, 0, 0, p], mt.interface_delta_q[dn, 0, 0, p]):
                out['delta_min_margin_pu'] = min(out['delta_min_margin_pu'],
                                                 pe.value(var) - var.lb, var.ub - pe.value(var))
        out['lmp_p'][b] = lp
        out['lmp_q'][b] = lq
    for k in mt.voltage_magnitude_upper_cons:
        i, _, _, p = k
        row = mt.voltage_magnitude_upper_cons[k]
        vsq = pe.value(mt.vmag_sqr[i, 0, 0, p])
        v_sqrt = math.sqrt(max(vsq, 0.0))
        v_var = pe.value(mt.vmag[i, 0, 0, p]) if (i, 0, 0, p) in mt.vmag else None
        v = v_var if v_var is not None else v_sqrt     # the vmag Var where it exists (as the record), else sqrt(vmag_sqr)
        vmax = math.sqrt(row.upper)
        s_row = row.upper - pe.value(row.body)
        z = mt.dual.get(row)
        lo_row = mt.voltage_magnitude_lower_cons[k] if k in mt.voltage_magnitude_lower_cons else None
        out['voltage'].append({
            'bus': order[i], 'period': p, 'v_pu': v, 'v_max_pu': vmax, 'dist_to_vmax_pu': vmax - v,
            'v_min_pu': math.sqrt(lo_row.lower) if lo_row is not None else None,
            'dist_to_vmin_pu': (v - math.sqrt(lo_row.lower)) if lo_row is not None else None,
            'upper_row_slack_pu2': s_row, 'upper_row_dual': z,
            'complementarity_product': (abs(z) * s_row) if z is not None else None,
            'slack_v_sqr_up_fixed': bool(mt.slack_v_sqr_up[i, 0, 0, p].fixed),
            'slack_v_sqr_up_value': pe.value(mt.slack_v_sqr_up[i, 0, 0, p]),
            'lmp_q_at_bus': mt.dual[mt.node_balance_q[i, 0, 0, p]] / B,
            'vmag_var_pu': v_var, 'sqrt_vmag_sqr_pu': v_sqrt})
    return out


# ======================================================================================================================
#  helpers for aggregation
# ======================================================================================================================
def _sum_rows(rows, keys):
    return {k: sum(r[k] for r in rows) for k in keys}


VOL_KEYS = ('p_up_mw', 'p_down_mw', 'q_up_mvar', 'q_down_mvar', 'cost_p', 'cost_q')


def split_x0(dso_rows, blocks, S):
    """dso_rows[(node, y, d)] -> per-slot rows. Returns per DSO: CR/ES x components, weighted r2 and undiscounted."""
    out = {}
    for n in ADN:
        agg = {}
        for wkey in ('w_r2', 'w_u'):
            g = {grp: {k: 0.0 for k in VOL_KEYS} for grp in ('CR', 'ES')}
            per_year = {y: {grp: {k: 0.0 for k in VOL_KEYS} for grp in ('CR', 'ES')} for y in YEARS}
            for b in blocks:
                for r in dso_rows[(n, b['year'], b['day'])]:
                    grp = 'CR' if (b['year'], b['day'], r['period']) in S else 'ES'
                    for k in VOL_KEYS:
                        g[grp][k] += b[wkey] * r[k] * (DT_H if not k.startswith('cost') else 1.0)
                        per_year[b['year']][grp][k] += b[wkey] * r[k] * (DT_H if not k.startswith('cost') else 1.0)
            for grp in ('CR', 'ES'):
                g[grp]['cost_total'] = g[grp]['cost_p'] + g[grp]['cost_q']
                for y in YEARS:
                    per_year[y][grp]['cost_total'] = per_year[y][grp]['cost_p'] + per_year[y][grp]['cost_q']
            tot = g['CR']['cost_total'] + g['ES']['cost_total']
            agg[wkey] = {'CR': g['CR'], 'ES': g['ES'], 'total_cost': tot,
                         'CR_share_of_cost': (g['CR']['cost_total'] / tot) if tot else None,
                         'per_year': {str(y): per_year[y] for y in YEARS}}
        out[str(n)] = agg
    return out


def split_unit(fc_unit, dso_rows, blocks, S):
    """Block totals at the unit; exact where a block has no slot in S, interval otherwise."""
    out = {}
    for n in ADN:
        res = {}
        for wkey in ('w_r2', 'w_u'):
            cr_lo = cr_hi = 0.0
            es_exact = 0.0
            fc_tot = 0.0
            fcq_lo = fcq_hi = 0.0
            per_block = []
            for b in blocks:
                rows = dso_rows[(n, b['year'], b['day'])]
                fc = fc_unit[(n, b['year'], b['day'])]
                in_s = [r for r in rows if (b['year'], b['day'], r['period']) in S]
                out_s = [r for r in rows if (b['year'], b['day'], r['period']) not in S]
                eps = sum(r['eps_p'] + r['eps_q'] for r in rows)
                ub_cr = sum(r['ub_cost_p'] + r['ub_cost_q'] for r in in_s)
                ub_es = sum(r['ub_cost_p'] + r['ub_cost_q'] for r in out_s)
                ubq = sum(r['ub_cost_q'] for r in rows)
                epsq = sum(r['eps_q'] for r in rows)
                if in_s:
                    lo, hi = max(-eps, fc - ub_es), min(fc + eps, ub_cr)
                else:
                    lo = hi = 0.0
                    es_exact += b[wkey] * fc
                cr_lo += b[wkey] * lo
                cr_hi += b[wkey] * hi
                fc_tot += b[wkey] * fc
                fcq_lo += b[wkey] * (-epsq)
                fcq_hi += b[wkey] * ubq
                per_block.append({'year': b['year'], 'day': b['day'], 'fc_unweighted': fc, 'n_slots_in_S': len(in_s),
                                  'CR_interval_unweighted': [lo, hi], 'UB_CR': ub_cr, 'UB_ES': ub_es, 'eps': eps,
                                  'UB_Q': ubq, 'eps_q': epsq})
            res[wkey] = {'FC_total': fc_tot, 'CR_interval': [cr_lo, cr_hi],
                         'ES_interval': [fc_tot - cr_hi, fc_tot - cr_lo],
                         'ES_exact_part_blocks_without_S': es_exact,
                         'FC_Q_interval': [fcq_lo, fcq_hi], 'FC_P_interval': [fc_tot - fcq_hi, fc_tot - fcq_lo],
                         'per_block': per_block}
        out[str(n)] = res
    return out


# ======================================================================================================================
#  markdown
# ======================================================================================================================
def _markdown(res):
    L = []
    a = L.append
    inst = res['instances']
    a(f'# {STAGE}\n')
    a(f'Instances: x = 0 `{inst["x0"]["candidate_key"]}` (`{inst["x0"]["path"]}`, {inst["x0"]["campaign_id"]}, '
      f'status {inst["x0"]["status"]}, {inst["x0"]["cycles_run"]} cycles, Q gross {_fmt(inst["x0"]["certified_cost_gross_eur"])}, '
      f'bar {_fmt(inst["x0"]["bar_eur"])}, terminal step / threshold {inst["x0"]["terminal_step_over_threshold"]:.4f}; '
      f'persisted models `{PICKLE_REL}` sha256 {PICKLE_SHA}); UNIT node 7 0.25 MVA / 1.0 MWh, 2025, nodes 5/9 empty '
      f'`{inst["unit"]["candidate_key"]}` (`{inst["unit"]["path"]}`, {inst["unit"]["campaign_id"]}, status '
      f'{inst["unit"]["status"]}, {inst["unit"]["cycles_run"]} cycles, Q gross {_fmt(inst["unit"]["certified_cost_gross_eur"])}, '
      f'bar {_fmt(inst["unit"]["bar_eur"])}, terminal step / threshold {inst["unit"]["terminal_step_over_threshold"]:.4f}; '
      f'RECORDS ONLY, no persisted models). Baseline = C2 + phi_cal 0.985 + soh_min 0.70.\n')
    g = res['solve_profile_guard']
    a(f'Solve profile: armed SolveProfileGuard(permitted=()); counts {g["counts"]}; verify(0) failures '
      f'{g["verify_0_failures"]}. Git HEAD {res["git_HEAD"]}. all_pass = {res.get("all_pass")}; failed checks: '
      f'{res.get("failed_checks")}.\n')
    a(OBJ_CONV + '\n')

    z = res['Z33a']
    a('## Z33 (a) -- the cost_flex profile actually applied\n')
    a(z['statement'] + '\n')
    a('Citations: ' + '; '.join(f'{k} {v}' for k, v in res['citations'].items()) + '\n')
    a('| year | day | growth cumul (1+g)^(y-2025) | min | max | mean | top-4 mean | bottom-4 mean | base (applied / cumul) mean |')
    a('|---|---|---|---|---|---|---|---|---|')
    for r in z['per_block']:
        a(f'| {r["year"]} | {r["day"]} | {r["growth_cumul"]:.6f} | {r["min"]:.3f} | {r["max"]:.3f} | {r["mean"]:.3f} | '
          f'{r["top4_mean"]:.3f} | {r["bottom4_mean"]:.3f} | {r["base_mean"]:.3f} |')
    a('')
    a('| year | day-weighted mean c_flex (EUR/MWh) | min | max |')
    a('|---|---|---|---|')
    for y, r in z['per_year'].items():
        a(f'| {y} | {r["mean_day_weighted"]:.3f} | {r["min"]:.3f} | {r["max"]:.3f} |')
    a(f'| all (undiscounted day weights) | {z["all_years_mean_day_weighted"]:.3f} | {z["all_min"]:.3f} | {z["all_max"]:.3f} |')
    a('\nHourly profiles (24 values per block) are in the JSON (`Z33a.per_block[*].hourly`).\n')

    zb = res['Z33b']
    a('## Z33 (b) -- congestion relief (node-7 binding slots) vs economic shifting\n')
    a(zb['slot_set_statement'] + '\n')
    a('**x = 0 (exact, per slot from the persisted DSO blocks; weighted r2, EUR; volumes MWh / MVArh, weighted by the '
      'same block weights).** P-up and Q-up carry no cost (the objective prices only flex_p_down and flex_q_down).\n')
    a('| DSO | group | slots | P-up MWh | P-down MWh | Q-up MVArh | Q-down MVArh | cost P-down | cost Q-down | cost total | share of DSO flex cost |')
    a('|---|---|---|---|---|---|---|---|---|---|---|')
    for n in ADN:
        s = zb['x0'][str(n)]['w_r2']
        for grp, lab, ns in (('CR', 'congestion relief', zb['n_slots_S']), ('ES', 'economic shifting', zb['n_slots_total'] - zb['n_slots_S'])):
            v = s[grp]
            share = v['cost_total'] / s['total_cost'] if s['total_cost'] else float('nan')
            a(f'| {n} | {lab} | {ns} | {_fmt(v["p_up_mw"])} | {_fmt(v["p_down_mw"])} | {v["q_up_mvar"]:.4f} | '
              f'{v["q_down_mvar"]:.4f} | {_fmt(v["cost_p"])} | {v["cost_q"]:.4f} | {_fmt(v["cost_total"])} | {share:.4f} |')
    a('')
    a(zb['x0_reading'] + '\n')
    a('**UNIT (block totals only -- no per-slot flexibility is recorded for it; weighted r2, EUR).** CR is exact (= 0) '
      'in the blocks without a binding slot; in the 7 blocks with binding slots only the interval stated in the '
      'formula is available.\n')
    a('| DSO | FC(0) | FC(x) | saving FC(0) - FC(x) | CR(0) exact | CR(x) interval | saving on CR interval | ES(0) exact | ES(x) interval | saving on ES interval | ES(x) exact part (blocks without S) |')
    a('|---|---|---|---|---|---|---|---|---|---|---|')
    for n in ADN:
        s0 = zb['x0'][str(n)]['w_r2']
        su = zb['unit'][str(n)]['w_r2']
        c0, e0 = s0['CR']['cost_total'], s0['ES']['cost_total']
        a(f'| {n} | {_fmt(s0["total_cost"])} | {_fmt(su["FC_total"])} | {_fmt(s0["total_cost"] - su["FC_total"])} | '
          f'{_fmt(c0)} | [{_fmt(su["CR_interval"][0])}, {_fmt(su["CR_interval"][1])}] | '
          f'[{_fmt(c0 - su["CR_interval"][1])}, {_fmt(c0 - su["CR_interval"][0])}] | {_fmt(e0)} | '
          f'[{_fmt(su["ES_interval"][0])}, {_fmt(su["ES_interval"][1])}] | '
          f'[{_fmt(e0 - su["ES_interval"][1])}, {_fmt(e0 - su["ES_interval"][0])}] | {_fmt(su["ES_exact_part_blocks_without_S"])} |')
    a('')
    a('Saving restricted to the blocks WITHOUT a binding slot (exact, all economic shifting, r2): ' + '; '.join(
        f'DSO {n} {_fmt(zb["saving_blocks_without_S"][str(n)])}' for n in ADN) + '. Saving in the 7 blocks WITH binding '
      'slots (block totals, r2): ' + '; '.join(f'DSO {n} {_fmt(zb["saving_blocks_with_S"][str(n)])}' for n in ADN) + '.\n')
    a(zb['resolution_statement'] + '\n')
    a(zb['not_computable'] + '\n')

    z4 = res['Z34']
    a('## Z34 (a) -- P/Q split of the storage value and of each DSO\'s flexibility saving\n')
    a(z4['a']['definition'] + '\n')
    a('| quantity | r2 (EUR) | undiscounted (EUR) | status |')
    a('|---|---|---|---|')
    for r in z4['a']['storage_rows']:
        a(f'| {r["quantity"]} | {_fmt(r["r2"])} | {_fmt(r["undisc"])} | {r["status"]} |')
    a('')
    a('| DSO | saving (exact, records) | FC_P(0) exact | FC_Q(0) exact | FC_Q(x) interval | saving_Q interval | saving_P interval |')
    a('|---|---|---|---|---|---|---|')
    for r in z4['a']['dso_rows']:
        a(f'| {r["dso"]} | {_fmt(r["saving"])} | {_fmt(r["fcp0"])} | {r["fcq0"]:.4f} | [{r["fcq_x"][0]:.4f}, {r["fcq_x"][1]:.4f}] | '
          f'[{r["saving_q"][0]:.4f}, {r["saving_q"][1]:.4f}] | [{_fmt(r["saving_p"][0])}, {_fmt(r["saving_p"][1])}] |')
    a('')
    a(z4['a']['reading'] + '\n')

    a('## Z34 (b) -- the storage\'s Q dispatch as a fraction of its 0.25 MVA rating\n')
    b = z4['b']
    a(b['definition'] + '\n')
    a('| series | min | median | p90 | p99 | max | slots > 1 % | slots > 10 % | slots > 50 % | absorbing (q > 0) | injecting (q < 0) |')
    a('|---|---|---|---|---|---|---|---|---|---|---|')
    for k, s in b['distributions'].items():
        a(f'| {k} | {s["min"]:.5f} | {s["median"]:.5f} | {s["p90"]:.5f} | {s["p99"]:.5f} | {s["max"]:.5f} | '
          f'{s["n_gt_0p01"]} | {s["n_gt_0p1"]} | {s["n_gt_0p5"]} | {s.get("n_absorbing", "")} | {s.get("n_injecting", "")} |')
    a('')
    a(b['reading'] + '\n')

    a('## Z34 (c) -- interface voltage-bound activity (entries at the 1.1 pu bound)\n')
    c = z4['c']
    a(c['definition'] + '\n')
    a('| run | bus | entries (of 288) within 1e-6 pu of v_max | 1e-5 | 1e-4 | 1e-3 | min distance to v_max (pu) | within 1e-6 of v_min |')
    a('|---|---|---|---|---|---|---|---|')
    for run in ('x0_models', 'x0_record', 'unit_record'):
        for bus in ADN:
            r = c[run]['per_bus'][str(bus)]
            a(f'| {run} | {bus} | {r["n_1e-06"]} | {r["n_1e-05"]} | {r["n_0.0001"]} | {r["n_0.001"]} | '
              f'{r["min_dist_vmax"]:.3e} | {r["n_vmin_1e-06"]} |')
    a('')
    a(c['slots_statement'] + '\n')
    a(c['multiplier_statement'] + '\n')
    a(c['other_buses_statement'] + '\n')
    a(c['dso_context_statement'] + '\n')

    a('## Z34 (d) -- Addendum 34 conclusion\n')
    a(z4['d']['statement'] + '\n')

    a('## Checks\n')
    for k, v in sorted(res['checks'].items()):
        a(f'- {k}: {v}')
    return '\n'.join(L) + '\n'


# ======================================================================================================================
#  main
# ======================================================================================================================
def _manifest():
    files = {}
    for name in sorted(os.listdir(OUT_DIR)):
        if name in (MANIFEST_NAME, LOCK_NAME):
            continue
        path = os.path.join(OUT_DIR, name)
        files[os.path.relpath(path, REPO)] = {'sha256': _sha256(path), 'bytes': os.path.getsize(path)}
    res = _load(os.path.join(OUT_DIR, RESULTS_NAME))
    manifest = {'stage': STAGE, 'generated_utc': _utc(), 'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
                'instance_candidate_keys': {'x0': res['instances']['x0']['candidate_key'],
                                            'unit': res['instances']['unit']['candidate_key']},
                'script': {'path': os.path.basename(__file__), 'sha256': _sha256(os.path.abspath(__file__))},
                'files': files,
                'evidence_inputs_sha256': {k: v['sha256'] for k, v in res['instances']['files'].items()}}
    path = os.path.join(OUT_DIR, MANIFEST_NAME)
    if os.path.exists(path):
        raise SystemExit(f'refusing to overwrite the manifest: {path}')
    with open(path, 'w') as handle:
        json.dump(manifest, handle, indent=1)
    _log(f'[W29] manifest: {len(files)} files -> {os.path.relpath(path, REPO)}')


def main():
    started = _utc()
    if not os.path.isdir(OUT_DIR):
        raise SystemExit(f'output directory must be created by the launcher: {OUT_DIR}')
    res_path, md_path = os.path.join(OUT_DIR, RESULTS_NAME), os.path.join(OUT_DIR, MD_NAME)
    for p in (res_path, md_path):
        if os.path.exists(p):
            raise SystemExit(f'refusing to overwrite (write-once): {p}')
    lock = os.path.join(OUT_DIR, LOCK_NAME)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # refuses a concurrent copy of this script
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    _log(f'[W29] start {started}; out {OUT_DIR}')
    res = {'stage': STAGE, 'authority': AUTHORITY, 'started_utc': started, 'out_dir': OUT_DIR,
           'git_HEAD': _git(['rev-parse', 'HEAD']).strip(), 'objective_convention': OBJ_CONV,
           'tolerances_stated_before_run': {
               'TOL_BOUND_PU': TOL_BOUND_PU, 'TOL_LADDER_PU': TOL_LADDER_PU, 'RELAX': RELAX, 'IDENT_TOL': IDENT_TOL,
               'REPRO_REL': REPRO_REL, 'REPRO_ABS': REPRO_ABS, 'COEF_TOL': COEF_TOL, 'VOLT_XCHK': VOLT_XCHK,
               'W25_A0_TOL': W25_A0_TOL, 'DELTA_INTERIOR_MW': DELTA_INTERIOR_MW, 'SUM_TOL_EUR': SUM_TOL_EUR,
               'AT_RATING': AT_RATING}}
    checks, errors = {}, {}
    try:
        # ---------------- capture checklist (fail fast) ----------------
        c, inst = capture_checklist()
        checks.update(c)
        res['instances'] = inst
        missing = sorted(k for k, v in c.items() if not v)
        _log(f'[W29] capture checklist: {len(c)} items, failed {missing}')
        if missing:
            raise RuntimeError(f'capture checklist failed before execution: {missing}')
        cyc0, cycu = inst['cycles']['x0'], inst['cycles']['unit']
        case, blocks = case_blocks()
        bmap = {(b['year'], b['day']): b for b in blocks}
        gf = growth_factors()
        res['growth_factors_workbook'] = gf
        res['citations'] = citations()
        z4 = _load(W19_JSON_REL)['Z4']
        prices = {(int(r['year']), r['day']): list(r['prices']) for r in z4['per_representative_day']}

        # ---------------- binding slot set ----------------
        w19 = _load(W19_JSON_REL)['Z1']
        S_w19 = {(int(r['year']), r['day'], int(r['period'])) for r in
                 w19['terminal_interface_flows']['a1a_candidate'][str(NODE)]['at_rating_slots']}
        rating = w19['terminal_interface_flows']['a1a_candidate'][str(NODE)]['rating_mva']
        sets = {}
        for name, rel in (('x0', X0_REL), ('unit', UNIT_REL)):
            rows = binding_slots(rel)
            sets[name] = {(y, d, p) for (y, d, p, s) in rows if s / rating >= AT_RATING}
            checks[f'binding_set_{name}_288_slots'] = len(rows) == 288
        checks['binding_set_x0_equals_W19'] = sets['x0'] == S_w19
        checks['binding_set_unit_equals_W19'] = sets['unit'] == S_w19
        S = S_w19
        _log(f'[W29] binding set |S| = {len(S)}; x0 == W19 {sets["x0"] == S_w19}; unit == W19 {sets["unit"] == S_w19}')

        # ---------------- terminal strides ----------------
        pf0_row, last0 = stride_terminal(X0_REL, cyc0, 'pf_entry_stride_s39_D.jsonl')
        pfu_row, lastu = stride_terminal(UNIT_REL, cycu, 'pf_entry_stride_s39_D.jsonl')
        checks['pf_stride_x0_last_cycle_is_terminal'] = last0 == cyc0 and pf0_row is not None
        checks['pf_stride_unit_last_cycle_is_terminal'] = lastu == cycu and pfu_row is not None
        st0 = {(e['node_id'], str(e['year']), e['day'], e['power_type'], e['period']): e for e in pf0_row['entries']}
        stu = {(e['node_id'], str(e['year']), e['day'], e['power_type'], e['period']): e for e in pfu_row['entries']}
        del pf0_row, pfu_row
        ess_u = ess_cycle(UNIT_REL, cycu)
        ess_d = ess_cycle(UNIT_DUP_REL, cycu)
        ess_snap = ess_cycle(UNIT_REL, SNAP_STRIDE_CYCLE)
        esso = esso_capture(UNIT_REL, cycu)
        checks['esso_capture_288_rows_single_cohort'] = len(esso) == 288 and all(int(r['y_inv']) == 0 for r in esso.values())
        checks['esso_capture_s_max_is_0p25'] = all(abs(r['s_max'] - S_RATED_MVA) <= 1e-12 for r in esso.values())

        # ---------------- x = 0 models (loaded ONCE) ----------------
        _log(f'[W29] loading {PICKLE_REL}')
        with open(os.path.join(REPO, PICKLE_REL), 'rb') as handle:
            payload = pickle.load(handle)
        mc = model_checklist(payload)
        checks.update(mc)
        miss = sorted(k for k, v in mc.items() if not v)
        _log(f'[W29] model capture checklist: {len(mc)} items, failed {miss}')
        if miss:
            raise RuntimeError(f'model capture checklist failed: {miss}')
        cl0 = _load(X0_REL + '/component_levels_terminal.json')['blocks']
        cla0 = _load(A0_X0_REL + '/component_levels_terminal.json')['blocks']
        dso_rows, dso_meta, dso_volt = {}, {}, {}
        worst = {'pq': 0.0, 'uni': 0.0, 'repro': 0.0, 'repro_expr_vs_total': 0.0, 'cross_dso': 0.0,
                 'a0_vs_s48': 0.0, 'min_q': float('inf'), 'min_p': float('inf'), 'max_q_ub': 0.0}
        other_vars = set()
        profiles = {}
        for n in ADN:
            for y in YEARS:
                for d in DAYS:
                    md = payload['dso'][n][y][d]
                    B = 100.0
                    blk = dso_block(md, B)
                    key = f'DSO|{n}|{y}|{d}'
                    rec_fc = cl0[key]['unweighted']['flexibility_cost_internal']
                    tot = sum(r['cost_p'] + r['cost_q'] for r in blk['rows'])
                    worst['repro'] = max(worst['repro'], abs(tot - rec_fc) / max(abs(rec_fc), 1.0))
                    worst['repro_expr_vs_total'] = max(worst['repro_expr_vs_total'],
                                                       abs(tot - blk['total_from_expression']),
                                                       abs(blk['total_from_expression'] - blk['total_flex_cost']))
                    worst['a0_vs_s48'] = max(worst['a0_vs_s48'], abs(cla0[key]['unweighted']['flexibility_cost_internal'] - rec_fc))
                    worst['pq'] = max(worst['pq'], blk['pq_coef_max_abs_dev'] if blk['pq_same_index_set'] else float('inf'))
                    worst['uni'] = max(worst['uni'], blk['coef_uniform_across_loads_max_abs_dev_eur_per_mwh'])
                    worst['min_q'] = min(worst['min_q'], blk['min_flex_value_pu']['q'])
                    worst['min_p'] = min(worst['min_p'], blk['min_flex_value_pu']['p'])
                    worst['max_q_ub'] = max(worst['max_q_ub'], blk['max_q_flex_ub_pu'])
                    other_vars |= set(blk['other_vars_in_cost'])
                    if (y, d) in profiles:
                        worst['cross_dso'] = max(worst['cross_dso'], max(abs(a_ - b_) for a_, b_ in zip(profiles[(y, d)], blk['c_flex'])))
                    else:
                        profiles[(y, d)] = blk['c_flex']
                    dso_rows[(n, y, d)] = blk['rows']
                    dso_meta[key] = {k: v for k, v in blk.items() if k != 'rows'}
                    dso_volt[key] = dso_voltage_context(md)
            gc.collect()
        checks['flex_cost_only_flex_p_down_and_flex_q_down'] = other_vars == set()
        checks['flex_cost_P_and_Q_coefficients_equal_every_load_period'] = worst['pq'] <= COEF_TOL
        checks['flex_cost_coefficient_uniform_across_loads'] = worst['uni'] <= COEF_TOL
        checks['flex_cost_profile_identical_across_the_three_DSOs'] = worst['cross_dso'] <= COEF_TOL
        checks['flex_cost_per_slot_sum_reproduces_record_component_level'] = worst['repro'] <= REPRO_REL or worst['repro'] * 1.0 <= REPRO_ABS
        checks['flex_cost_per_slot_sum_equals_model_expression_and_total'] = worst['repro_expr_vs_total'] <= REPRO_ABS
        checks['x0_s48_flex_levels_equal_a0_x0_record_W25'] = worst['a0_vs_s48'] <= REPRO_ABS
        checks['q_flex_bounds_are_2e-5_pu'] = abs(worst['max_q_ub'] - 2e-5) <= 1e-12
        checks['flex_values_not_below_relaxed_bound'] = min(worst['min_q'], worst['min_p']) >= -RELAX * 1.001
        res['x0_dso_model_checks'] = {'worst': worst, 'per_block_meta': dso_meta}
        _log(f'[W29] DSO blocks: worst {worst}')

        tso = {}
        volt_rows0 = []
        for y in YEARS:
            case9 = _load(CASE9_FMT.format(y=y))
            for d in DAYS:
                t = tso_block(payload['tso'][y][d], case9, prices[(y, d)], st0, {'year': y, 'day': d})
                tso[(y, d)] = t
                for r in t['voltage']:
                    volt_rows0.append(dict(r, year=y, day=d))
        ident_p = max(t['ident_p_err'] for t in tso.values())
        ident_q = max(t['ident_q_err'] for t in tso.values())
        dmin = min(t['delta_min_margin_pu'] for t in tso.values())
        checks['tso_identity_P_on_every_persisted_x0_block'] = ident_p <= IDENT_TOL
        checks['tso_identity_Q_on_every_persisted_x0_block'] = ident_q <= IDENT_TOL
        checks['tso_interface_delta_interior_x0_models'] = dmin * 100.0 >= DELTA_INTERIOR_MW
        checks['tso_sigma_and_w_equal_record_weights'] = all(
            abs(t['w_block'] - bmap[k]['w_r2']) <= 1e-9 * 500 for k, t in tso.items())
        # unit TSO snapshot: sign / units of the ESS stride Q (load convention, MVAr = p.u. x baseMVA)
        with open(os.path.join(REPO, SNAP_REL), 'rb') as handle:
            snap = pickle.load(handle)
        ms = snap['model']
        snap_dev = 0.0
        for p in range(24):
            for e_idx, bus in enumerate(ADN):
                q_m = pe.value(ms.shared_es_qnet[e_idx, 0, 0, p]) * 100.0
                p_m = pe.value(ms.shared_es_pnet[e_idx, 0, 0, p]) * 100.0
                if bus == NODE:
                    snap_dev = max(snap_dev,
                                   abs(q_m - ess_snap[(NODE, SNAP_YEAR, SNAP_DAY, 'q')]['x']['tso'][p]),
                                   abs(p_m - ess_snap[(NODE, SNAP_YEAR, SNAP_DAY, 'p')]['x']['tso'][p]))
                else:
                    snap_dev = max(snap_dev, abs(q_m), abs(p_m))
        checks['ess_stride_q_is_tso_shared_es_qnet_x_baseMVA_load_convention'] = snap_dev <= 1e-9
        res['ess_stride_sign_check'] = {'snapshot': SNAP_REL, 'stride_cycle': SNAP_STRIDE_CYCLE,
                                        'max_abs_dev_mw_mvar': snap_dev,
                                        'snapshot_metadata': {k: str(v) for k, v in snap['metadata'].items()}}
        del snap, ms
        del payload
        gc.collect()
        _log(f'[W29] TSO: identity P {ident_p:.3e}, Q {ident_q:.3e}; delta min margin {dmin * 100:.3f} MW; '
             f'snapshot sign check {snap_dev:.2e}')

        # ---------------- Z33 (a) ----------------
        cf_rows = []
        for b in blocks:
            h = profiles[(b['year'], b['day'])]
            cum = (1.0 + gf['Flexibility']) ** (b['year'] - FIRST_YEAR)
            s = sorted(h)
            cf_rows.append({'year': b['year'], 'day': b['day'], 'growth_cumul': cum, 'hourly': h,
                            'base_hourly': [v / cum for v in h], 'min': min(h), 'max': max(h), 'mean': sum(h) / 24,
                            'top4_mean': sum(s[-4:]) / 4, 'bottom4_mean': sum(s[:4]) / 4,
                            'base_mean': sum(h) / 24 / cum, 'w_u': b['w_u']})
        per_year = {}
        for y in YEARS:
            sub = [r for r in cf_rows if r['year'] == y]
            per_year[str(y)] = {'mean_day_weighted': sum(r['w_u'] * r['mean'] for r in sub) / sum(r['w_u'] for r in sub),
                                'min': min(r['min'] for r in sub), 'max': max(r['max'] for r in sub)}
        allmean = sum(r['w_u'] * r['mean'] for r in cf_rows) / sum(r['w_u'] for r in cf_rows)
        cites = res['citations']
        res['Z33a'] = {
            'per_block': cf_rows, 'per_year': per_year, 'all_years_mean_day_weighted': allmean,
            'all_min': min(r['min'] for r in cf_rows), 'all_max': max(r['max'] for r in cf_rows),
            'growth_factor_flexibility': gf['Flexibility'], 'growth_factor_energy': gf['Energy'],
            'statement': (
                f'Applied c_flex = (coefficient of flex_p_down[c, p] in flex_cost_scenario[0, 0]) / baseMVA, read from '
                f'every persisted x = 0 DSO block (3 DSOs x 12 blocks); identical across the three DSOs (max dev '
                f'{worst["cross_dso"]:.1e}) and across all priced loads (max dev {worst["uni"]:.1e}). Production builds it '
                f'as cost_flex[year][day] = (one sampled synthetic flexibility-price profile per (year, day), market '
                f'scenario count 1) x (1 + g_flex)^(year - 2025) ({cites["cost_flex_built"]}; sampling '
                f'{cites["flexibility_profile_sampled_per_year_day"]}; cumul {cites["flexibility_growth_cumul"]}; g read '
                f'at {cites["flexibility_growth_factor_read"]} from {MARKET_REL} sheet "Growth Factors": Flexibility '
                f'{gf["Flexibility"]}, Energy {gf["Energy"]}) and binds it to every DSO block '
                f'({cites["cost_flex_bound_to_dso_blocks"]}). Range {min(r["min"] for r in cf_rows):.2f} to '
                f'{max(r["max"] for r in cf_rows):.2f} EUR/MWh; day-weighted mean {allmean:.2f} (2025 '
                f'{per_year["2025"]["mean_day_weighted"]:.2f}, 2030 {per_year["2030"]["mean_day_weighted"]:.2f}, 2035 '
                f'{per_year["2035"]["mean_day_weighted"]:.2f}). '
                f'P and Q carry the SAME price: flexibility_cost ({cites["flexibility_cost_def"]}) charges '
                f'c_flex[p] * baseMVA * (flex_p_down + flex_q_down) ({cites["flexibility_cost_term_p_down_plus_q_down_same_c_flex"]}); '
                f'confirmed on the built objective: coefficient(flex_p_down[c,p]) = coefficient(flex_q_down[c,p]) for '
                f'every priced load and period (max abs dev {worst["pq"]:.1e} EUR per p.u.). flex_p_up and flex_q_up '
                f'are NOT priced. BUT the reactive flexibility is structurally absent: its bounds are set to '
                f'EQUALITY_TOLERANCE (1e-5 p.u.) for every load ({cites["reactive_flex_up_bound_is_EQUALITY_TOLERANCE"]}, '
                f'{cites["reactive_flex_down_bound_is_EQUALITY_TOLERANCE"]}), i.e. ub(flex_q_*) = 2e-5 p.u. = 2 kVAr '
                f'per load in the built models (checked: max ub {worst["max_q_ub"]:.1e} p.u.), and the Q day-balance '
                f'row is not wired ({cites["flex_q_day_balance_NOT_wired"]}); the P day balance is '
                f'({cites["flex_p_day_balance_wired"]}).')}

        # ---------------- Z33 (b) ----------------
        split0 = split_x0(dso_rows, blocks, S)
        clu = _load(UNIT_REL + '/component_levels_terminal.json')['blocks']
        cld = _load(UNIT_DUP_REL + '/component_levels_terminal.json')['blocks']
        fc_u = {}
        dup_dev = 0.0
        for n in ADN:
            for b in blocks:
                key = f'DSO|{n}|{b["year"]}|{b["day"]}'
                fc_u[(n, b['year'], b['day'])] = clu[key]['unweighted']['flexibility_cost_internal']
                dup_dev = max(dup_dev, abs(clu[key]['unweighted']['flexibility_cost_internal']
                                           - cld[key]['unweighted']['flexibility_cost_internal']))
        checks['unit_duplicate_S3_flex_levels_identical'] = dup_dev == 0.0
        splitu = split_unit(fc_u, dso_rows, blocks, S)
        blocks_with_S = sorted({(y, d) for (y, d, _) in S})
        sav_wo, sav_w = {}, {}
        for n in ADN:
            fc0b = {(b['year'], b['day']): sum(r['cost_p'] + r['cost_q'] for r in dso_rows[(n, b['year'], b['day'])]) for b in blocks}
            sav_wo[str(n)] = sum(b['w_r2'] * (fc0b[(b['year'], b['day'])] - fc_u[(n, b['year'], b['day'])])
                                 for b in blocks if (b['year'], b['day']) not in blocks_with_S)
            sav_w[str(n)] = sum(b['w_r2'] * (fc0b[(b['year'], b['day'])] - fc_u[(n, b['year'], b['day'])])
                                for b in blocks if (b['year'], b['day']) in blocks_with_S)
        w25 = _load(W25_JSON_REL)
        w25_rows = {r['agent']: r for r in w25['T3']['baseline']['rows_r2'] if r['year'] == 'all'}
        for n in ADN:
            s_ = split0[str(n)]['w_r2']['total_cost'] - splitu[str(n)]['w_r2']['FC_total']
            checks[f'dso{n}_saving_equals_W25_T3_within_0.01'] = abs(s_ - w25_rows[f'DSO{n}']['flexibility_cost_internal']) <= SUM_TOL_EUR
        bar_sum = inst['x0']['bar_eur'] + inst['unit']['bar_eur']
        n_all = 12 * 24
        eps_tot = {str(n): sum(b['w_r2'] * sum(r['eps_p'] + r['eps_q'] for r in dso_rows[(n, b['year'], b['day'])]
                                               if (b['year'], b['day'], r['period']) in S)
                               for b in blocks) for n in ADN}
        x0_reading = (
            'x = 0: the flexibility cost in the node-7 binding slots is ' + ', '.join(
                f'{split0[str(n)]["w_r2"]["CR"]["cost_total"]:.2f}' for n in ADN) +
            ' EUR (DSO 5, 7, 9; r2), i.e. zero to within IPOPT\'s bound relaxation (|.| <= eps_S = ' + ', '.join(
                f'{eps_tot[str(n)]:.2f}' for n in ADN) + ' EUR over the 23 slots, the values being slightly negative P-down / Q-down at '
            'the relaxed lower bound). No DSO buys downward flexibility in those hours. In the binding slots the DSOs '
            'move load UP (unpriced P-up): P-up in S = ' + ', '.join(
                f'{split0[str(n)]["w_r2"]["CR"]["p_up_mw"]:,.0f}' for n in ADN) + ' MWh (weighted) = ' + ', '.join(
                f'{split0[str(n)]["w_r2"]["CR"]["p_up_mw"] / (split0[str(n)]["w_r2"]["CR"]["p_up_mw"] + split0[str(n)]["w_r2"]["ES"]["p_up_mw"]):.3f}'
                for n in ADN) + ' of each DSO\'s total P-up; the priced P-down leg of the same day-balanced shift '
            'lies entirely outside S. Under the stated definition the whole flexibility cost at x = 0 is economic '
            'shifting.')
        res['Z33b'] = {
            'x0_reading': x0_reading, 'eps_total_r2': eps_tot,
            'slot_set': sorted([list(s) for s in S]), 'n_slots_S': len(S), 'n_slots_total': n_all,
            'blocks_with_S': [list(b) for b in blocks_with_S],
            'x0': split0, 'unit': splitu, 'saving_blocks_without_S': sav_wo, 'saving_blocks_with_S': sav_w,
            'slot_set_statement': (
                f'Binding set S = the {len(S)} of {n_all} (year, day, hour) slots where the node-7 DSO interface '
                f'apparent flow sqrt(P^2 + Q^2) (interface_settlement_detail, DSO side) is >= {AT_RATING:.0%} of its '
                f'{rating:.0f} MVA rating -- W19 Z1\'s slot set ({W19_COMMIT}), recomputed here on the x = 0 record '
                f'(equal: {sets["x0"] == S_w19}) and on the unit record (equal: {sets["unit"] == S_w19}). S lies in '
                f'{len(blocks_with_S)} blocks: ' + ', '.join(f'{y} {d}' for y, d in blocks_with_S) + '; hours 9-13. '
                'The same slots are applied to DSOs 5 and 9 (whose own interfaces are never near rating: W19 max '
                'utilisation 0.51 / 0.67), so for them "congestion relief" means "flexibility bought in the node-7 '
                'binding hours".'),
            'resolution_statement': (
                f'Resolution: value and savings are differences of two ADMM runs; bar(x0) + bar(unit) = '
                f'{_fmt(bar_sum)} EUR (x0 bar {_fmt(inst["x0"]["bar_eur"])}, terminal step / threshold '
                f'{inst["x0"]["terminal_step_over_threshold"]:.4f}; unit bar {_fmt(inst["unit"]["bar_eur"])}, terminal '
                f'step / threshold {inst["unit"]["terminal_step_over_threshold"]:.4f}). The bar bounds the TOTAL '
                'recourse; its split across agents/blocks is not recorded, so a per-DSO or per-group saving smaller '
                'than the bar is not shown to be determinate by this bar (each per-DSO saving here is of the same '
                'order as the bar).'),
            'not_computable': (
                'NOT COMPUTABLE for the unit without its terminal DSO models: the per-slot flexibility (hence the exact '
                'CR/ES split inside the 7 blocks that contain binding slots, and the per-slot P-up / P-down / Q '
                'volumes). Searched: the unit record directory (every *.json / *.jsonl key containing "flex": only '
                'component_levels_terminal / evaluation_record block totals, interface_settlement_detail '
                'flexibility_volumes_per_dso = TSO interface_delta totals, recourse_jump_sidecar per-block totals), '
                'results/FrozenSMOPF (cycle-7 pre-solve snapshots only, not terminal), esso_capture (ESSO rows only). '
                'What it would take: the unit\'s terminal DSO blocks -- a re-evaluation of the unit with '
                'persist_certified_models (one ~1 h evaluation; solves, not authorized here).')}
        _log('[W29] Z33b x0 CR shares: ' + ', '.join(
            f'{n}: {split0[str(n)]["w_r2"]["CR_share_of_cost"]:.4f}' for n in ADN))

        # ---------------- Z34 (a) storage P/Q value ----------------
        recu = _load(UNIT_REL + '/evaluation_record.json')
        rec0 = _load(X0_REL + '/evaluation_record.json')
        value = rec0['certified_cost'] - recu['certified_cost']
        checks['value_is_259427.77_2dp'] = abs(round(value, 2) - 259427.77) <= 0.005 + 1e-9
        rep_u = _load(UNIT_REL + '/interface_settlement_detail_s31c.json')['interface_reporting_detail']
        dmin_u = float('inf')
        for n in ADN:
            for b in blocks:
                for p in range(24):
                    per = rep_u[str(n)][str(b['year'])][b['day']]['periods'][str(p)]
                    rat = stu[(n, str(b['year']), b['day'], 'p', p)]['interface_rating']
                    dmin_u = min(dmin_u, rat - abs(per['delta_p_mw']['0_0']), rat - abs(per['delta_q_mvar']['0_0']))
        checks['unit_interface_delta_interior_every_block_period'] = dmin_u >= DELTA_INTERIOR_MW
        lmpx_p, lmpx_q = {}, {}
        for b in blocks:
            sig = tso[(b['year'], b['day'])]['sigma']
            wb = b['w_r2']
            lp, lq = [], []
            for p in range(24):
                ep = stu[(NODE, str(b['year']), b['day'], 'p', p)]
                eq = stu[(NODE, str(b['year']), b['day'], 'q', p)]
                R = ep['interface_rating'] / 100.0
                lp.append(prices[(b['year'], b['day'])][p] + sig * ep['lambda_dso'] / (wb * ep['s_base_dso'] * R * 100.0))
                lq.append(sig * eq['lambda_dso'] / (wb * eq['s_base_dso'] * R * 100.0))
            lmpx_p[(b['year'], b['day'])] = lp
            lmpx_q[(b['year'], b['day'])] = lq

        def _A(wkey):
            acc = {k: 0.0 for k in ('AP0', 'APx', 'AP0_tso', 'AQ0', 'AQx', 'AQ0_z', 'AQ0_dso', 'AQ0_esso', 'qabs_mvarh',
                                    'q_mvarh_signed')}
            map_dev = 0.0
            for b in blocks:
                w = b[wkey]
                yk, dk = b['year'], b['day']
                t = tso[(yk, dk)]
                eq = ess_u[(NODE, str(yk), dk, 'q')]
                epp = ess_u[(NODE, str(yk), dk, 'p')]
                for p in range(24):
                    r = esso[(yk, dk, p)]
                    pnet = r['pch'] - r['pdch']
                    map_dev = max(map_dev, abs(r['pnet'] - epp['x']['esso'][p]))
                    acc['AP0'] += w * t['lmp_p'][NODE][p] * (-pnet) * DT_H
                    acc['APx'] += w * lmpx_p[(yk, dk)][p] * (-pnet) * DT_H
                    acc['AP0_tso'] += w * t['lmp_p'][NODE][p] * (-epp['x']['tso'][p]) * DT_H
                    acc['AQ0'] += w * t['lmp_q'][NODE][p] * (-eq['x']['tso'][p]) * DT_H
                    acc['AQx'] += w * lmpx_q[(yk, dk)][p] * (-eq['x']['tso'][p]) * DT_H
                    acc['AQ0_z'] += w * t['lmp_q'][NODE][p] * (-eq['z'][p]) * DT_H
                    acc['AQ0_dso'] += w * t['lmp_q'][NODE][p] * (-eq['x']['dso'][p]) * DT_H
                    acc['AQ0_esso'] += w * t['lmp_q'][NODE][p] * (-eq['x']['esso'][p]) * DT_H
                    acc['qabs_mvarh'] += w * abs(eq['x']['tso'][p]) * DT_H
                    acc['q_mvarh_signed'] += w * eq['x']['tso'][p] * DT_H
            acc['map_dev'] = map_dev
            return acc
        A2, Au = _A('w_r2'), _A('w_u')
        checks['esso_capture_pnet_equals_ess_stride_esso_p'] = A2['map_dev'] <= 1e-9
        w25_A0 = w25['T4']['baseline']['r2']['A_realized_at_lmp_x0_eur']
        checks['A_P_at_x0_duals_reproduces_W25_A0_within_0.1_eur'] = abs(A2['AP0'] - w25_A0) <= W25_A0_TOL
        # value undiscounted
        gross = ('generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost', 'res_curtailment_penalty',
                 'ess_usage_cost', 'detector_penalty_total')
        value_r2_levels = value_u = 0.0
        for key, b0 in cl0.items():
            bu = clu[key]
            parts = key.split('|')
            yk, dk = int(parts[-2]), parts[-1]
            for comp in gross:
                value_r2_levels += b0['weighted'][comp] - bu['weighted'][comp]
                value_u += (b0['unweighted'][comp] - bu['unweighted'][comp]) * bmap[(yk, dk)]['w_u']
        checks['value_from_component_levels_equals_record_difference'] = abs(value_r2_levels - value) <= SUM_TOL_EUR
        # DSO saving P/Q
        dso_rows_out = []
        for n in ADN:
            s0 = split0[str(n)]['w_r2']
            su = splitu[str(n)]['w_r2']
            fcp0 = s0['CR']['cost_p'] + s0['ES']['cost_p']
            fcq0 = s0['CR']['cost_q'] + s0['ES']['cost_q']
            saving = s0['total_cost'] - su['FC_total']
            fq = su['FC_Q_interval']
            sq = [fcq0 - fq[1], fcq0 - fq[0]]
            dso_rows_out.append({'dso': n, 'saving': saving, 'fcp0': fcp0, 'fcq0': fcq0, 'fcq_x': fq, 'saving_q': sq,
                                 'saving_p': [saving - sq[1], saving - sq[0]]})
        lmpq7 = [v for t in tso.values() for v in t['lmp_q'][NODE]]
        lmpp7 = [v for t in tso.values() for v in t['lmp_p'][NODE]]
        storage_rows = [
            {'quantity': 'value = Q(0) - Q(x) (records)', 'r2': value, 'undisc': value_u, 'status': 'exact'},
            {'quantity': 'A_P: P dispatch (ESSO -pnet) at LMP_P,7(x=0) (terminal TSO duals)', 'r2': A2['AP0'],
             'undisc': Au['AP0'], 'status': 'attributed (first order)'},
            {'quantity': 'A_Q: Q dispatch (TSO copy -qnet) at LMP_Q,7(x=0) (terminal TSO duals)', 'r2': A2['AQ0'],
             'undisc': Au['AQ0'], 'status': 'attributed (first order)'},
            {'quantity': 'remainder = value - A_P - A_Q (system response incl. the DSO saving; ADMM stopping)',
             'r2': value - A2['AP0'] - A2['AQ0'], 'undisc': value_u - Au['AP0'] - Au['AQ0'], 'status': 'residual'},
            {'quantity': '  check: A_P with LMP_P(x) (identity on the unit record)', 'r2': A2['APx'], 'undisc': Au['APx'],
             'status': 'attributed'},
            {'quantity': '  check: A_Q with LMP_Q(x) (identity on the unit record)', 'r2': A2['AQx'], 'undisc': Au['AQx'],
             'status': 'attributed'},
            {'quantity': '  check: A_Q at LMP_Q(x=0) with the consensus z / DSO copy / ESSO copy', 'r2': A2['AQ0_z'],
             'undisc': Au['AQ0_z'], 'status': f'DSO copy {A2["AQ0_dso"]:.4f}, ESSO copy {A2["AQ0_esso"]:.4f} (r2)'},
            {'quantity': '  |Q| throughput of the storage, MVArh (TSO copy)', 'r2': A2['qabs_mvarh'],
             'undisc': Au['qabs_mvarh'], 'status': f'signed (absorbing +) {A2["q_mvarh_signed"]:.3f} MVArh (r2)'},
        ]
        res['Z34'] = {'a': {
            'definition': (
                'Storage: value = Q(0) - Q(x) (exact, records). Attributed split at the x = 0 marginal values (the exact '
                'terminal duals of the persisted x = 0 TSO blocks at bus 7, where the storage injects in the TSO model; '
                'in the DSO model its P and Q enter the reference-bus balance and the interface definition and cancel '
                f'-- {cites["dso_interface_q_def_storage_at_reference_bus"]}, W19 Z1): A_P = sum_b w_b sum_p '
                'LMP_P,7 * (pdch - pch) (ESSO terminal capture; W25\'s A0), A_Q = sum_b w_b sum_p LMP_Q,7 * (-qnet) '
                '(qnet = the TSO copy of the ESS Q consensus in the unit\'s terminal ess stride, load convention, MVAr; '
                'sign/units verified against the unit\'s persisted cycle-7 TSO snapshot). LMP_P,7 = '
                'dual(node_balance_p[7]) / baseMVA, LMP_Q,7 = dual(node_balance_q[7]) / baseMVA (EUR per MWh / MVArh '
                'of load at bus 7). remainder = value - A_P - A_Q. DSO flexibility saving: saving = FC(0) - FC(x) '
                '(exact, records); FC_P(0) and FC_Q(0) exact from the x = 0 models; at the unit FC_Q(x) is bounded by '
                'the structural reactive-flexibility bound (formula in the header), which bounds saving_Q and '
                'saving_P = saving - saving_Q. EXACT: value, saving, FC_P(0), FC_Q(0), the duals. ATTRIBUTED: A_P, '
                'A_Q (first-order pricing of the dispatch at fixed marginal values). BOUNDED (not exact): the unit\'s '
                'P/Q flexibility split.'),
            'storage_rows': storage_rows, 'dso_rows': dso_rows_out,
            'lmp_q7_x0_summary_eur_per_mvarh': {'min': min(lmpq7), 'max': max(lmpq7),
                                                'mean_abs': sum(abs(v) for v in lmpq7) / len(lmpq7)},
            'lmp_p7_x0_summary_eur_per_mwh': {'min': min(lmpp7), 'max': max(lmpp7),
                                              'mean': sum(lmpp7) / len(lmpp7)},
            'W25_A0_committed': w25_A0,
            'lmp_series_x0': {f'{y}_{d}': {'lmp_p7': t['lmp_p'][NODE], 'lmp_q7': t['lmp_q'][NODE],
                                           'lmp_q5': t['lmp_q'][5], 'lmp_q9': t['lmp_q'][9]} for (y, d), t in tso.items()},
            'lmp_series_unit_identity': {f'{y}_{d}': {'lmp_p7': lmpx_p[(y, d)], 'lmp_q7': lmpx_q[(y, d)]}
                                         for (y, d) in lmpx_p},
            'identity_errors_x0_models': {'P_max_abs_eur_per_mwh': ident_p, 'Q_max_abs_eur_per_mvarh': ident_q,
                                          'tso_interface_delta_min_margin_mw': dmin * 100.0,
                                          'unit_interface_delta_min_margin_mw': dmin_u}}}
        tot_q_ub = sum(r['fcq_x'][1] for r in dso_rows_out)
        res['Z34']['a']['reading'] = (
            f'Storage: A_P = {_fmt(A2["AP0"])} ({A2["AP0"] / value:.4f} of the value; W25 committed A0 '
            f'{_fmt(w25_A0)}), A_Q = {A2["AQ0"]:,.4f} EUR ({A2["AQ0"] / value:.2e} of the value), remainder '
            f'{_fmt(value - A2["AP0"] - A2["AQ0"])}. The reactive marginal value at bus 7 at x = 0 is '
            f'{min(lmpq7):.4f} to {max(lmpq7):.4f} EUR/MVArh (mean |.| '
            f'{sum(abs(v) for v in lmpq7) / len(lmpq7):.4f}) against LMP_P,7 mean {sum(lmpp7) / len(lmpp7):.2f} '
            f'EUR/MWh. DSOs: FC_Q(0) = ' + ', '.join(f'{r["fcq0"]:.4f}' for r in dso_rows_out) +
            f' EUR (DSO 5, 7, 9) and |FC_Q(x)| <= ' + ', '.join(f'{r["fcq_x"][1]:.2f}' for r in dso_rows_out) +
            f' EUR by the structural bound (sum {tot_q_ub:.2f}). Because FC_Q(0) is already zero (at the relaxed '
            'bound), the storage cannot SAVE reactive flexibility cost: saving_Q <= ' + ', '.join(
                f'{r["saving_q"][1]:.2f}' for r in dso_rows_out) + ' EUR, hence saving_P >= saving - that, i.e. at '
            'least ' + ', '.join(f'{r["saving_p"][0]:,.2f}' for r in dso_rows_out) + ' EUR of the DSO savings '
            '(DSO 5, 7, 9) is active-power (P-down) flexibility. The other side of the interval (the unit buying MORE '
            'reactive flexibility, saving_Q < 0) is bounded only by the structural bound, which is wide '
            '(2e-5 p.u. per load-hour, priced over the horizon) and so uninformative; without the unit\'s DSO models '
            'the Q part is not determined beyond: saving_Q <= ~0.')

        # ---------------- Z34 (b) Q dispatch fraction ----------------
        def _dist(vals, signed=None):
            out = {'min': min(vals), 'median': _pct(vals, 0.5), 'p90': _pct(vals, 0.9), 'p99': _pct(vals, 0.99),
                   'max': max(vals), 'n_gt_0p01': sum(v > 0.01 for v in vals), 'n_gt_0p1': sum(v > 0.1 for v in vals),
                   'n_gt_0p5': sum(v > 0.5 for v in vals), 'n': len(vals)}
            if signed is not None:
                out['n_absorbing'] = sum(v > 0 for v in signed)
                out['n_injecting'] = sum(v < 0 for v in signed)
            return out
        series = {}
        per_slot = []
        dup_q = 0.0
        for b in blocks:
            eq = ess_u[(NODE, str(b['year']), b['day'], 'q')]
            epp = ess_u[(NODE, str(b['year']), b['day'], 'p')]
            eqd = ess_d[(NODE, str(b['year']), b['day'], 'q')]
            for p in range(24):
                dup_q = max(dup_q, abs(eq['z'][p] - eqd['z'][p]))
                per_slot.append({'year': b['year'], 'day': b['day'], 'period': p, 'in_S': (b['year'], b['day'], p) in S,
                                 'q_z_mvar': eq['z'][p], 'q_tso': eq['x']['tso'][p], 'q_dso': eq['x']['dso'][p],
                                 'q_esso': eq['x']['esso'][p], 'p_z_mw': epp['z'][p],
                                 'q_frac': abs(eq['z'][p]) / S_RATED_MVA,
                                 's_frac': math.hypot(epp['z'][p], eq['z'][p]) / S_RATED_MVA,
                                 'lmp_q7_x0': tso[(b['year'], b['day'])]['lmp_q'][NODE][p]})
        checks['unit_duplicate_S3_terminal_q_identical'] = dup_q == 0.0
        for lab, key in (('|q| / S (consensus z)', 'q_z_mvar'), ('|q| / S (TSO copy)', 'q_tso'),
                         ('|q| / S (DSO copy)', 'q_dso'), ('|q| / S (ESSO copy)', 'q_esso')):
            series[lab] = _dist([abs(r[key]) / S_RATED_MVA for r in per_slot], [r[key] for r in per_slot])
        series['|q| / S (z) in the 23 binding slots'] = _dist([r['q_frac'] for r in per_slot if r['in_S']],
                                                             [r['q_z_mvar'] for r in per_slot if r['in_S']])
        series['sqrt(p^2+q^2) / S (z, converter loading)'] = _dist([r['s_frac'] for r in per_slot])
        qmax = max(per_slot, key=lambda r: r['q_frac'])
        at_circle = [r for r in per_slot if r['s_frac'] >= 0.99]
        res['Z34']['b'] = {
            'definition': (f'Terminal cycle {cycu} of the unit\'s ess_entry_stride_baseline.jsonl, node 7, power_type '
                           '"q" (MVAr, load convention: + = absorbing; per copy x.tso / x.dso / x.esso and the consensus '
                           f'z); fraction = |q| / {S_RATED_MVA} MVA, per slot (288 = 12 blocks x 24 h). esso_capture '
                           'holds no Q field (pch, pdch, pnet only), so the stride is the only per-slot Q source. The '
                           f'unit\'s S3 duplicate gives identical terminal z (max dev {dup_q:.1e}).'),
            'distributions': series, 'per_slot': per_slot,
            'argmax_slot': {k: qmax[k] for k in ('year', 'day', 'period', 'q_z_mvar', 'q_frac', 's_frac', 'in_S')},
            'reading': (f'Q is NOT idle: median |q|/S = {series["|q| / S (consensus z)"]["median"]:.4f}, max '
                        f'{series["|q| / S (consensus z)"]["max"]:.4f} ({qmax["year"]} {qmax["day"]} h{qmax["period"]}), '
                        f'{series["|q| / S (consensus z)"]["n_gt_0p01"]} of 288 slots above 1 % of rating, '
                        f'{series["|q| / S (consensus z)"]["n_absorbing"]} absorbing / '
                        f'{series["|q| / S (consensus z)"]["n_injecting"]} injecting; converter loading max '
                        f'{series["sqrt(p^2+q^2) / S (z, converter loading)"]["max"]:.4f}; {len(at_circle)} slots at '
                        f'>= 99 % converter loading (|q|/S there {min(r["q_frac"] for r in at_circle) if at_circle else float("nan"):.4f} to '
                        f'{max(r["q_frac"] for r in at_circle) if at_circle else float("nan"):.4f}: Q shares the P-Q circle with P). Its value at the bus-7 '
                        f'reactive marginal cost is A_Q = {A2["AQ0"]:.4f} EUR (r2) over the horizon.'
                        if series['|q| / S (consensus z)']['max'] > 0.01 else
                        f'Q is idle (max |q|/S = {series["|q| / S (consensus z)"]["max"]:.2e} <= 1 %).')}

        # ---------------- Z34 (c) voltage ----------------
        def _vcount(rows, dist_key='dist_to_vmax_pu', vmin_key='dist_to_vmin_pu'):
            out = {}
            for bus in ADN:
                sub = [r for r in rows if r['bus'] == bus]
                d_ = {f'n_{t}': sum(r[dist_key] <= t for r in sub) for t in TOL_LADDER_PU}
                d_['n'] = len(sub)
                d_['min_dist_vmax'] = min(r[dist_key] for r in sub)
                d_['n_vmin_1e-06'] = sum((r[vmin_key] is not None and r[vmin_key] <= TOL_BOUND_PU) for r in sub)
                d_['slots_at_bound'] = sorted([[r['year'], r['day'], r['period']] for r in sub if r[dist_key] <= TOL_BOUND_PU])
                out[str(bus)] = d_
            return out
        adn_rows0 = [r for r in volt_rows0 if r['bus'] in ADN]
        vt0 = _load(X0_REL + '/interface_voltage_terminal.json')
        vtu = _load(UNIT_REL + '/interface_voltage_terminal.json')
        vtd = _load(UNIT_DUP_REL + '/interface_voltage_terminal.json')
        m0 = {(r['bus'], r['year'], r['day'], r['period']): r for r in adn_rows0}
        xdev = 0.0
        for e in vt0['entries']:
            r = m0[(e['node_id'], int(e['year']), e['day'], int(e['period']))]
            xdev = max(xdev, abs(r['v_pu'] - e['tso_pu']), abs(r['v_max_pu'] - e['v_max_pu']))
        checks['x0_model_voltages_equal_x0_record_interface_voltage_terminal'] = xdev <= VOLT_XCHK
        vv_dev = max(abs(r['vmag_var_pu'] - r['sqrt_vmag_sqr_pu']) for r in adn_rows0)
        checks['unit_duplicate_S3_voltage_entries_identical'] = vtu['entries'] == vtd['entries']

        def _rec_rows(vt):
            return [{'bus': e['node_id'], 'year': int(e['year']), 'day': e['day'], 'period': int(e['period']),
                     'dist_to_vmax_pu': e['v_max_pu'] - e['tso_pu'], 'dist_to_vmin_pu': e['tso_pu'] - e['v_min_pu'],
                     'v_pu': e['tso_pu'], 'dso_pu': e['dso_pu']} for e in vt['entries']]
        rows_x0rec, rows_u = _rec_rows(vt0), _rec_rows(vtu)
        vc = {'x0_models': {'per_bus': _vcount(adn_rows0)}, 'x0_record': {'per_bus': _vcount(rows_x0rec)},
              'unit_record': {'per_bus': _vcount(rows_u)}}
        # change x0 -> unit at bus 7 and the at-bound set changes
        su0 = {(r['bus'], r['year'], r['day'], r['period']) for r in rows_x0rec if r['dist_to_vmax_pu'] <= TOL_BOUND_PU}
        suu = {(r['bus'], r['year'], r['day'], r['period']) for r in rows_u if r['dist_to_vmax_pu'] <= TOL_BOUND_PU}
        mu0 = {(r['bus'], r['year'], r['day'], r['period']): r['v_pu'] for r in rows_x0rec}
        dv7 = [r['v_pu'] - mu0[(7, r['year'], r['day'], r['period'])] for r in rows_u if r['bus'] == NODE]
        at0 = [r for r in adn_rows0 if r['dist_to_vmax_pu'] <= TOL_BOUND_PU]
        prods = [r['complementarity_product'] for r in at0 if r['complementarity_product'] is not None]
        duals_at = [abs(r['upper_row_dual']) for r in at0 if r['upper_row_dual'] is not None]
        lmpq_at = [r['lmp_q_at_bus'] for r in at0]
        near7 = [r for r in adn_rows0 if r['bus'] == NODE and r['dist_to_vmax_pu'] <= 1e-4]
        vc_near7 = [{k: r[k] for k in ('year', 'day', 'period', 'v_pu', 'dist_to_vmax_pu', 'upper_row_dual',
                                       'complementarity_product', 'lmp_q_at_bus')} for r in near7]
        other = [r for r in volt_rows0 if r['bus'] not in ADN]
        other_at = [r for r in other if r['dist_to_vmax_pu'] <= TOL_BOUND_PU]
        dctx = {k: v for k, v in dso_volt.items()}
        n_dso_up = sum(v['n_within_1e-4_of_vmax'] for v in dctx.values())
        n_dso_lo = sum(v['n_within_1e-4_of_vmin'] for v in dctx.values())
        n_dso_rows = sum(v['n_rows'] for v in dctx.values())
        slack_dso = sum(v['sum_voltage_slacks_pu2'] for v in dctx.values())
        vc['x0_models']['at_bound_rows'] = at0
        vc['x0_models']['other_tso_buses_at_vmax'] = [{k: r[k] for k in ('bus', 'year', 'day', 'period', 'v_pu', 'upper_row_dual')} for r in other_at]
        vc['unit_record']['bus7_v_change_vs_x0'] = {'min': min(dv7), 'max': max(dv7),
                                                   'mean_abs': sum(abs(v) for v in dv7) / len(dv7)}
        vc['at_bound_set_x0_record'] = sorted([list(k) for k in su0])
        vc['at_bound_set_unit'] = sorted([list(k) for k in suu])
        vc['entered_bound_with_unit'] = sorted([list(k) for k in suu - su0])
        vc['left_bound_with_unit'] = sorted([list(k) for k in su0 - suu])
        vc['dso_internal_context_x0'] = dctx
        vc['bus7_within_1e-4_rows_x0'] = vc_near7
        per_dso_ctx = {str(n): sum(v['n_within_1e-4_of_vmax'] for k, v in dctx.items() if k.startswith(f'DSO|{n}|'))
                       for n in ADN}
        vc['definition'] = (
            f'At the bound = v_max - v <= {TOL_BOUND_PU:g} pu (the convention of the production summary '
            '"n_entries_at_bound_within_1e-6_pu" in interface_voltage_terminal.json), v = TSO-side voltage magnitude '
            'at the ADN bus; also counted at 1e-5 / 1e-4 / 1e-3 pu. The TSO voltage rows at the ADN buses are HARD '
            f'(their slacks are fixed at 0: {cites["tso_adn_bus_voltage_slacks_fixed_0"]}; checked on the models). '
            'x = 0 from the persisted models (the vmag Var at the ADN buses; sqrt(vmag_sqr) elsewhere) and, as a cross-check, from the x = 0 record '
            f'(max dev {xdev:.1e} pu; the model\'s vmag Var is used, max |vmag - sqrt(vmag_sqr)| {vv_dev:.1e} pu); '
            'the unit from its record (S3 duplicate identical).')
        vc['slots_statement'] = (
            'At the bound (1e-6 pu): x = 0 -- ' + '; '.join(
                f'bus {b_} {vc["x0_models"]["per_bus"][str(b_)]["n_1e-06"]}' for b_ in ADN) +
            ' ; unit -- ' + '; '.join(f'bus {b_} {vc["unit_record"]["per_bus"][str(b_)]["n_1e-06"]}' for b_ in ADN) +
            f'. With the unit {len(suu - su0)} entries enter and {len(su0 - suu)} leave the at-bound set. Bus-7 '
            f'voltage change x = 0 -> unit: {min(dv7):+.2e} to {max(dv7):+.2e} pu (mean |.| '
            f'{sum(abs(v) for v in dv7) / len(dv7):.2e}); bus-7 minimum distance to 1.1 pu: x = 0 '
            f'{vc["x0_models"]["per_bus"]["7"]["min_dist_vmax"]:.2e}, unit {vc["unit_record"]["per_bus"]["7"]["min_dist_vmax"]:.2e} pu. '
            'Per-slot lists are in the JSON.')
        vc['multiplier_statement'] = (
            f'Multipliers at x = 0 on the {len(at0)} at-bound ADN entries: |dual(voltage_magnitude_upper_cons)| '
            f'{(min(duals_at) if duals_at else float("nan")):.3g} to {(max(duals_at) if duals_at else float("nan")):.3g} '
            '(EUR per p.u.^2 of v^2 per representative-day hour); the complementarity product s * |z| is '
            f'{(min(prods) if prods else float("nan")):.3g} to {(max(prods) if prods else float("nan")):.3g} '
            f'(median {(_pct(prods, 0.5) if prods else float("nan")):.3g}) -- a near-common value across rows, '
            'consistent with rows held at the interior-point barrier distance from the bound; their activity is '
            'resolved only to that level. Economically: the reactive marginal value at those entries '
            f'is {(min(lmpq_at) if lmpq_at else float("nan")):.4f} to {(max(lmpq_at) if lmpq_at else float("nan")):.4f} '
            'EUR/MVArh (dual(node_balance_q) / baseMVA at the same bus and hour). Bus 7 (the storage bus): '
            f'{len(near7)} entries within 1e-4 pu of 1.1, with |dual| '
            f'{(min(abs(r["upper_row_dual"]) for r in near7) if near7 else float("nan")):.3g} to '
            f'{(max(abs(r["upper_row_dual"]) for r in near7) if near7 else float("nan")):.3g} and s * |z| '
            f'{(min(r["complementarity_product"] for r in near7) if near7 else float("nan")):.3g} to '
            f'{(max(r["complementarity_product"] for r in near7) if near7 else float("nan")):.3g}.')
        vc['other_buses_statement'] = (
            f'Other TSO buses (non-ADN) within 1e-6 pu of v_max at x = 0: {len(other_at)} of {len(other)} entries '
            f'(buses {sorted({r["bus"] for r in other_at})}).')
        vc['dso_context_statement'] = (
            f'Context (x = 0, all DSO buses, soft voltage rows): {n_dso_up} of {n_dso_rows} DSO bus-hour entries '
            f'within 1e-4 pu of v_max (DSO 5 {per_dso_ctx["5"]}, DSO 7 {per_dso_ctx["7"]}, DSO 9 {per_dso_ctx["9"]}), '
            f'{n_dso_lo} within 1e-4 pu of v_min; sum of DSO voltage slacks {slack_dso:.3e} p.u.^2. In the DSO model '
            'the storage\'s Q enters only the reference-bus balance and the interface definition, where it cancels '
            '(W19 Z1 for P; the Q rows are the same form), so it reaches DN voltages only through the TSO bus-7 '
            'voltage, whose value to the system is priced by LMP_Q,7.')
        res['Z34']['c'] = vc

        # ---------------- Z34 (d) ----------------
        q_idle = series['|q| / S (consensus z)']['max'] <= 0.01
        bus7_binds = (vc['x0_models']['per_bus']['7']['n_1e-06'] > 0 or vc['unit_record']['per_bus']['7']['n_1e-06'] > 0)
        res['Z34']['d'] = {
            'criteria_stated_before_run': ('Q idle := max |q|/S <= 1 % over the 288 slots; the bus-7 voltage bound '
                                           'binds := any bus-7 entry within 1e-6 pu of v_max at x = 0 or with the unit.'),
            'q_idle': q_idle, 'bus7_voltage_bound_binds': bus7_binds,
            'statement': (
                f'Q idle: {q_idle} (max |q|/S {series["|q| / S (consensus z)"]["max"]:.4f}). Bus-7 1.1 pu bound binds: '
                f'{bus7_binds} (x = 0: {vc["x0_models"]["per_bus"]["7"]["n_1e-06"]} entries, unit: '
                f'{vc["unit_record"]["per_bus"]["7"]["n_1e-06"]}). The 1.1 pu bound is reached only at the OTHER two '
                f'interfaces (bus 5: x = 0 {vc["x0_models"]["per_bus"]["5"]["n_1e-06"]}, unit '
                f'{vc["unit_record"]["per_bus"]["5"]["n_1e-06"]}; bus 9: x = 0 {vc["x0_models"]["per_bus"]["9"]["n_1e-06"]}, '
                f'unit {vc["unit_record"]["per_bus"]["9"]["n_1e-06"]}), where there is no storage in this instance. '
                f'At bus 7 the reactive marginal value is {min(lmpq7):.4f} to {max(lmpq7):.4f} EUR/MVArh (vs LMP_P,7 '
                f'mean {sum(lmpp7) / len(lmpp7):.2f} EUR/MWh), so the storage\'s Q -- dispatched, not idle -- is worth '
                f'A_Q = {A2["AQ0"]:.2f} EUR over the horizon ({A2["AQ0"] / value:.1e} of the value, below the '
                f'resolution bar {_fmt(bar_sum)}). The DSOs\' reactive flexibility cannot be displaced: it is '
                'structurally absent (bounds = EQUALITY_TOLERANCE) and unused at x = 0, so saving_Q <= ~0 and the '
                'saving is P flexibility. Reading (criteria above): the storage\'s voltage support is effectively '
                'unmonetized because no voltage constraint binds at its bus (bus 7 never within 1e-6 pu of 1.1; the '
                f'reactive price there is at most {max(abs(v) for v in lmpq7):.2f} EUR/MVArh in magnitude), and the DN side has no reactive flexibility for it to '
                'displace -- network value exists only where a constraint binds.')}
        _log(f'[W29] Z34: A_P {A2["AP0"]:.2f} A_Q {A2["AQ0"]:.4f}; q max frac {series["|q| / S (consensus z)"]["max"]:.4f}; '
             f'bus7 binds {bus7_binds}')
    except Exception:  # noqa: BLE001 -- recorded; the run then fails
        errors['run'] = traceback.format_exc()
        print(errors['run'], file=sys.stderr, flush=True)

    failures = GUARD.verify(0)
    res.update({'checks': checks, 'failed_checks': sorted(k for k, v in checks.items() if not v), 'errors': errors,
                'solve_profile_guard': {'permitted': [], 'counts': dict(GUARD.counts), 'verify_0_failures': failures},
                'ended_utc': _utc()})
    res['all_pass'] = bool(checks) and not res['failed_checks'] and not errors and not failures

    def _default(o):
        if isinstance(o, tuple):
            return list(o)
        if hasattr(o, 'item'):
            return o.item()
        return str(o)
    with open(res_path, 'x') as handle:
        json.dump(res, handle, indent=1, default=_default)
    if not errors:
        with open(md_path, 'x') as handle:
            handle.write(_markdown(res))
    os.remove(lock)
    GUARD.uninstall()
    _log(f"[W29] checks={len(checks)} failed={res['failed_checks']} errors={sorted(errors)} "
         f"guard={dict(GUARD.counts)} verify0_failures={failures} all_pass={res['all_pass']}")
    if not res['all_pass']:
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        GUARD.uninstall()
        _manifest()
        sys.exit(0)
    main()
