"""P5.15 Addendum 31 item (2) (task W25) -- TSO bus-7 marginal cost, value attribution and the
captured-spread decomposition. ZERO SOLVES, no model construction, read-only.

Armed `SolveProfileGuard(permitted=())` is installed before any production import; `verify(0)` is
checked at the end (exact count: 0 solves, 0 solver launches).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 31 ("Author's sanity check (cycle economics)",
zero-solve item) and the Planner's W25 task (T1-T5).

INSTANCES (every artifact records them by candidate key)
  x = 0      : a0_c7:x0 record, candidate_key 8435c718...  (Q(0), ageing-independent per W21 2466401d)
  BASELINE   : s47_recert:n7_4h_e1 (S2 record, C2 + phi_cal 0.985 + soh_min 0.70),
               node 7 0.25 MVA / 1.0 MWh, 2025, nodes 5/9 empty, candidate_key db77e154...
  SENSITIVITY: s45_a1a:n7_4h_e1 (C3 / soh_min 0.50 record), same candidate key -- reported in
               SEPARATE tables labelled "sensitivity set" (Addendum 30: never shares a table with the
               baseline).

WHAT IS COMPUTED (formulas are also written into the JSON and the Markdown)
  T1  How the market price enters the TSO objective and how bus 7 connects to it: file:line
      citations resolved at run time, the case9 generator data, and structural facts read from the
      run's own persisted TSO block (x0 run, 2025 Summer, cycle-7 pre-solve snapshot).
  T2  The TSO bus-7 nodal marginal cost at x = 0. The terminal duals are NOT persisted. They are
      recovered from an exact first-order (KKT) identity of the TSO ADMM block, derived from the code
      and VERIFIED on the one persisted, solved TSO block of each run (the cycle-7 pre-solve snapshot,
      which carries the cycle-6 solve's IPOPT duals):
          y_b[p] / baseMVA = pi[p] + sigma * lambda_dso_b[p] / (w_block * s_base * R_b * baseMVA)
      y_b = dual of node_balance_p at the ADN bus b (Pyomo sign: +objective coefficient, S36),
      pi = market price, sigma = common ADMM objective scale, w_block = admm_block_weight,
      lambda_dso = the DSO's post-update PF dual of the same cycle (pf_entry_stride, pre-AA),
      R_b = interface rating / s_base (p.u.). Valid when interface_delta_p is strictly inside its
      bounds (checked for every terminal block/period from the run's interface_reporting_detail).
      4 h spread per representative day = mean of the 4 highest - mean of the 4 lowest (W19 Z4).
  T3  value = Q(0) - Q(x) split by agent (TSO, DSO 5/7/9) x component x year, from
      component_levels_terminal.json 'weighted' (model block weights, 2 %), with the exact-sum check.
  T4  Captured-spread decomposition (per full cycle of cell-side throughput, both weightings).
  T5  value / Q(0), sigma_Q / Q(0), bar / Q(0) with log10.

Output (write-once, new directory) data/SRP1/Results/P515S47/tso_marginal_cost/:
    tso_marginal_cost.json, tso_marginal_cost.md, launch.log, manifest_sha256.json

Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S47/tso_marginal_cost
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_tso_marginal_cost.py \\
        > data/SRP1/Results/P515S47/tso_marginal_cost/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_tso_marginal_cost.py --manifest
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W25 TSO bus-7 marginal cost (zero solves)').install()

import pyomo.environ as pe  # noqa: E402

STAGE = 'P5.15 Addendum 31 item (2) W25 -- TSO bus-7 marginal cost, value attribution, captured spread (zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 31 (author sanity check, zero-solve item)',
             'Planner task W25 (T1-T5), 2026-09-21']
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'tso_marginal_cost')
# W25_SCRATCH_OUT (optional, test runs only): write to a scratch directory instead; the value is recorded.
OUT_DIR = os.environ.get('W25_SCRATCH_OUT') or os.path.join(REPO, OUT_REL)
RESULTS_NAME = 'tso_marginal_cost.json'
MD_NAME = 'tso_marginal_cost.md'
MANIFEST_NAME = 'manifest_sha256.json'
LOCK_NAME = '.w25.lock'

RECORDS = {
    'x0': 'data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0',
    'baseline': 'data/SRP1/Results/P515S47/campaign_s47_recert/evals/bd504ecf5a288d44_n7_4h_e1',
    'sensitivity_c3': 'data/SRP1/Results/P515S45/campaign_s45_a1a/evals/7eb1ce62c2509f54_n7_4h_e1',
}
RECORD_LABEL = {'x0': 'x = 0 (a0_c7:x0)',
                'baseline': 'BASELINE (s47_recert:n7_4h_e1; C2 + phi_cal 0.985 + soh_min 0.70)',
                'sensitivity_c3': 'SENSITIVITY SET (s45_a1a:n7_4h_e1; C3, soh_min 0.50)'}
EXPECTED_KEY = {'x0': '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57',
                'baseline': 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a',
                'sensitivity_c3': 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a'}
EXPECTED_VALUE = {'baseline': 259427.77, 'sensitivity_c3': 261807.50}   # task text, 2-decimal check
SNAPSHOT = 'results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl'
SNAPSHOT_YEAR, SNAPSHOT_DAY, SNAPSHOT_CYCLE = '2025', 'Summer', 7
CASE_REL = 'data/SRP1/SRP1.json'
PARAMS_REL = 'data/SRP1/SRP1_params.json'
CASE9_REL = 'data/SRP1/case9/case9_2025.json'
Z4_JSON_REL = 'data/SRP1/Results/P515S46/zero_solve_reports/zero_solve_reports.json'
Z4_COMMIT = 'dd3afa6e'
W22_JSON_REL = 'data/SRP1/Results/P515S47/market_spreads/market_spreads.json'
W22_COMMIT = '046d4d00'
PHASE_A_TABLES_REL = 'data/SRP1/Results/P515S45/phase_a_tables/phase_a_tables.json'
PHASE_A_COMMIT = 'c1b64278'
SRP1_Z4_SPREAD = 97.99086039739976
SRP1_Z4_SPREAD_R2 = 97.01788674187569

NODE = 7
ADN_NODES = (5, 7, 9)
E_MWH = 1.0
H = 4                     # 4 h window = E/P = 1.0 / 0.25
DT_H = 1.0                # 24 periods per representative day
DISCOUNT = 0.02
FIRST_YEAR = 2025

# Tolerances, stated before the committed run:
#  - KKT identity on the persisted solved blocks: |y/baseMVA - (pi + sigma*lambda/(w*sb*R*B))| <= 1e-4 EUR/MWh
#    (IPOPT's converged duals; 1e-4 EUR/MWh is 1e-6 of the ~100 EUR/MWh price level).
#  - interface_delta_p "strictly interior": margin to its bound >= 1e-3 MW at every terminal block/period.
#  - exact-sum checks (T3): |sum of parts - (Q(0) - Q(x))| <= 0.01 EUR (W19's reconciliation bound).
#  - Z4 reproduction of the market spread: 1e-9 EUR/MWh.
#  - ESSO index mapping / EFC recompute: 1e-9 (relative) -- the same data written twice by production.
KKT_TOL = 1e-4
DELTA_INTERIOR_MW = 1e-3
SUM_TOL_EUR = 0.01
REPRO_TOL = 1e-9
EFC_REL_TOL = 1e-9
GEN_INTERIOR_PU = 1e-4    # a CONV generator is "interior" when both bound margins exceed 1e-4 p.u.

GROSS_COMPONENTS = ('generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost',
                    'res_curtailment_penalty', 'ess_usage_cost', 'detector_penalty_total')
DETECTOR_SUBCOMPONENTS = ('voltage_slack', 'node_balance_slack', 'branch_flow_slack',
                          'flexibility_p_day_balance_slack', 'local_ess_day_balance_slack',
                          'shared_ess_day_balance_slack')
OBJ_CONV = ('Objective convention: Q(x) = gross_operational_cost, settlement-EXCLUDED (the oracle cost convention); '
            'value = Q(0) - Q(x) (positive = the storage lowers operating cost); terminal salvage is 0 on these '
            'records (gross = net); EUR; model block weights num_years * num_days / 1.02^(y - 2025) unless stated.')

SPREAD_FORMULA = (f'per representative (year, day) and hourly series c (24 values, EUR/MWh): '
                  f'top{H}_minus_bottom{H} = mean of the {H} highest c_p - mean of the {H} lowest (W19 Z4, '
                  f'p515_s46_zero_solve_reports.py @ {Z4_COMMIT}); horizon average = sum w * spread / sum w with '
                  f'w = num_years * num_days (undiscounted) or w = num_years * num_days / 1.02^(y - 2025) (r2).')
LMP_FORMULA = ('LMP_b[y,d,p] [EUR/MWh] = y_b / baseMVA = pi[y,d,p] + sigma * lambda_dso_b[y,d,p] / '
               '(w_block[y,d] * s_base * R_b * baseMVA); y_b = IPOPT dual of node_balance_p at TSO bus b '
               '(Pyomo sign: +objective coefficient of an interior injection, per p.u.-hour of the active '
               'objective, which is the PHYSICAL per-representative-day objective plus sigma/w_block times the '
               'ADMM terms -- p58_rescaled_admm_objective); lambda_dso = dual_vars[pf][dso] after the cycle\'s '
               'update (pf_entry_stride, captured before Anderson acceleration); R_b = interface rating / s_base '
               '(p.u.); sigma = admm_common_objective_scale; w_block = admm_block_weight. Derivation: '
               'stationarity of the TSO block in interface_delta_p (free; enters node_balance_p[b] with -1 via '
               'compute_node_load, the settlement with -pi*baseMVA, expected_interface_pf_p_def with -1) and in '
               'expected_interface_pf_p (free; ADMM terms), with lambda_tso = -lambda_dso (production '
               'antisymmetry, admm_anderson_acceleration.py) and the post-update identity lambda_tso(c)/s_base = '
               'lambda_param + rho*(E - p_req)/R (p_req = this cycle\'s DSO value: DSO solves first). Single market and '
               'operation scenario: the scenario-deviation penalty has zero gradient; TSO proximal gamma_pf = 0 '
               '(checked on the snapshot).')


def _utc():
    return datetime.now(timezone.utc).isoformat()


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


def _find_key(obj, key):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found += _find_key(v, key)
    elif isinstance(obj, list):
        for v in obj:
            found += _find_key(v, key)
    return found


def _topbottom(series, h=H):
    s = sorted(series)
    return sum(s[-h:]) / h - sum(s[:h]) / h


def _fmt(x, nd=2):
    return f'{x:,.{nd}f}'


# ======================================================================================================================
#  case data (light reads, no production objects)
# ======================================================================================================================
def case_structure():
    case = _load(CASE_REL)
    years = list(case['Years'].keys())
    days = list(case['Days'].keys())
    blocks = []
    for y in years:
        for d in days:
            n_y, n_d = float(case['Years'][y]), float(case['Days'][d])
            blocks.append({'year': y, 'day': d, 'num_years': n_y, 'num_days': n_d,
                           'w_undisc': n_y * n_d,
                           'w_r2': n_y * n_d / (1.0 + DISCOUNT) ** (int(y) - FIRST_YEAR)})
    return case, years, days, blocks


def market_prices(blocks):
    """Per-(year, day) market prices: the committed Z4 per-day arrays (W19 verified them equal to the
    prices recorded by the certified runs); reproduced spread must equal Z4's to REPRO_TOL."""
    z4 = _load(Z4_JSON_REL)['Z4']
    prices = {}
    for r in z4['per_representative_day']:
        prices[(str(r['year']), r['day'])] = list(r['prices'])
    num = sum(b['w_undisc'] * _topbottom(prices[(b['year'], b['day'])]) for b in blocks)
    den = sum(b['w_undisc'] for b in blocks)
    num2 = sum(b['w_r2'] * _topbottom(prices[(b['year'], b['day'])]) for b in blocks)
    den2 = sum(b['w_r2'] for b in blocks)
    repro = {'undiscounted': num / den, 'r2': num2 / den2,
             'z4_committed_undiscounted': z4['spread_summary']['all_years_day_weighted_undiscounted'][f'top{H}_mean_minus_bottom{H}_mean'],
             'z4_committed_r2': z4['spread_summary']['all_years_model_block_weight_r2'][f'top{H}_mean_minus_bottom{H}_mean']}
    return prices, repro


# ======================================================================================================================
#  record readers
# ======================================================================================================================
def _stride_terminal(rel_dir, cycle_terminal):
    """Stream pf_entry_stride_s39_D.jsonl; return node-7/5/9 P entries at the terminal cycle and the cycle
    before it, plus cycle 6 (for the snapshot check)."""
    path = os.path.join(REPO, rel_dir, 'pf_entry_stride_s39_D.jsonl')
    keep = {cycle_terminal: None, cycle_terminal - 1: None, SNAPSHOT_CYCLE - 1: None}
    n_lines, last_cycle = 0, None
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            n_lines += 1
            last_cycle = row['cycle']
            if row['cycle'] in keep:
                ent = {}
                for e in row['entries']:
                    if e['power_type'] != 'p':
                        continue
                    ent[(e['node_id'], e['year'], e['day'], e['period'])] = {
                        'lambda_dso': e['lambda_dso'], 'x_dso': e['x_dso'], 'z_tso': e['z_tso_current'],
                        'interface_rating': e['interface_rating'], 's_base': e['s_base_dso'], 'rho_pf': e['rho_pf']}
                keep[row['cycle']] = ent
    return keep, n_lines, last_cycle


def _ess_stride_terminal(rel_dir, cycle_terminal):
    path = os.path.join(REPO, rel_dir, 'ess_entry_stride_baseline.jsonl')
    out = None
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['cycle'] == cycle_terminal:
                out = {}
                for e in row['entries']:
                    if e['node_id'] == NODE and e['power_type'] == 'p':
                        out[(e['year'], e['day'])] = {'z': e['z'], 'x': e['x']}
    return out


def _esso_capture(rel_dir, cycle_terminal, years, days):
    path = os.path.join(REPO, rel_dir, 'esso_capture', 's39_D', f'node{NODE}_cycle{cycle_terminal:03d}.jsonl')
    rows = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (years[int(r['y'])], days[int(r['d'])], int(r['p']))
            if key in rows:
                raise RuntimeError(f'duplicate ESSO capture row {key} in {path} (more than one cohort?)')
            rows[key] = {'pch': r['pch'], 'pdch': r['pdch'], 'pnet': r['pnet'], 'y_inv': r['y_inv']}
    return rows, os.path.relpath(path, REPO)


# ======================================================================================================================
#  pre-run capture checklist (CLAUDE.md rule: assert the capture path before executing)
# ======================================================================================================================
def capture_checklist():
    chk, info = {}, {}
    tracked = set(_git(['ls-files', '--', *RECORDS.values(), CASE_REL, PARAMS_REL, CASE9_REL, Z4_JSON_REL,
                        W22_JSON_REL, PHASE_A_TABLES_REL]).splitlines())
    for name, rel in RECORDS.items():
        rec = _load(os.path.join(rel, 'evaluation_record.json'))
        cyc = int(rec['cycles_run'])
        man_rel = os.path.join(rel, 'child_manifest_sha256.json')
        manifest = _load(man_rel)
        man_files = manifest.get('files', manifest) if isinstance(manifest, dict) else {}
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
        _collect(man_files)
        needed = {
            'evaluation_record': os.path.join(rel, 'evaluation_record.json'),
            'component_levels_terminal': os.path.join(rel, 'component_levels_terminal.json'),
            'interface_settlement_detail': os.path.join(rel, 'interface_settlement_detail_s31c.json'),
            'pf_entry_stride': os.path.join(rel, 'pf_entry_stride_s39_D.jsonl'),
            'tso_snapshot_cycle7': os.path.join(rel, SNAPSHOT),
        }
        if name != 'x0':
            needed['ess_entry_stride'] = os.path.join(rel, 'ess_entry_stride_baseline.jsonl')
            needed['esso_capture_terminal'] = os.path.join(rel, 'esso_capture', 's39_D',
                                                           f'node{NODE}_cycle{cyc:03d}.jsonl')
        files = {}
        for label, frel in needed.items():
            exists = os.path.isfile(os.path.join(REPO, frel))
            is_tracked = frel in tracked
            sha = _sha256(os.path.join(REPO, frel)) if exists else None
            in_manifest = frel in flat
            files[label] = {'path': frel, 'exists': exists, 'tracked': is_tracked, 'sha256': sha,
                            'child_manifest_sha256': flat.get(frel),
                            'settled_by': ('git' if is_tracked else
                                           ('committed child manifest hash' if in_manifest and flat[frel] == sha
                                            else 'UNSETTLED'))}
            chk[f'{name}_{label}_exists'] = exists
            chk[f'{name}_{label}_settled_by_git_or_committed_manifest_hash'] = files[label]['settled_by'] != 'UNSETTLED'
        chk[f'{name}_child_manifest_tracked'] = man_rel in set(_git(['ls-files', '--', man_rel]).splitlines())
        chk[f'{name}_candidate_key_as_expected'] = rec['candidate_key'] == EXPECTED_KEY[name]
        chk[f'{name}_certified'] = rec['status'] == 'certified'
        info[name] = {'path': rel, 'label': RECORD_LABEL[name], 'campaign_id': rec['campaign_id'],
                      'candidate_label': rec['candidate_label'], 'candidate_key': rec['candidate_key'],
                      'candidate_canonical': rec['candidate_canonical'], 'eval_key': rec['eval_key'],
                      'status': rec['status'], 'cycles_run': cyc,
                      'certified_cost_gross_eur': rec['certified_cost'],
                      'terminal_net_operational_recourse': rec.get('terminal_net_operational_recourse'),
                      'bar_eur': rec['bar']['value'],
                      'terminal_objective_change_abs': rec['rule_ten']['terminal_objective_change_abs'],
                      'terminal_step_over_threshold': rec['rule_ten']['terminal_step_over_threshold'],
                      'case_file_sha256_in_child': rec.get('case_file_sha256_in_child'),
                      'files': files}
    for rel in (CASE_REL, PARAMS_REL, CASE9_REL, Z4_JSON_REL, W22_JSON_REL, PHASE_A_TABLES_REL):
        chk[f'tracked::{rel}'] = rel in tracked
    info['static_inputs'] = {rel: {'sha256': _sha256(os.path.join(REPO, rel)), 'tracked': rel in tracked}
                             for rel in (CASE_REL, PARAMS_REL, CASE9_REL, Z4_JSON_REL, W22_JSON_REL,
                                         PHASE_A_TABLES_REL)}
    params_sha = info['static_inputs'][PARAMS_REL]['sha256']
    chk['case_params_file_is_the_one_every_record_ran_with'] = all(
        info[n]['case_file_sha256_in_child'] == params_sha for n in RECORDS)
    return chk, info


# ======================================================================================================================
#  T1 -- how the market price enters the TSO objective
# ======================================================================================================================
def t1_citations():
    mch = 'model_construction_helpers.py'
    srp = 'shared_resources_planning.py'
    cite = {
        'generation_cost_prices_every_controllable_generator_at_c_p': _one_line(mch, 'def generation_cost(model, network, s_m, s_o, params):'),
        'controllable_types': _one_line('definitions.py', 'GEN_CONTROLLABLE_TYPES = '),
        'total_generation_cost_probability_weights': _one_line(mch, 'def total_generation_cost_rule(model, network):'),
        'interface_energy_settlement_tso_minus_pi_pc_adn': _one_line(mch, 'def interface_energy_settlement(model, network):'),
        'settlement_in_objective': _one_line(mch, 'obj += model.interface_settlement_weight * model.interface_settlement'),
        'settlement_weight_set_1_admm_tso': _one_line(srp, 'def _prepare_transmission_objectives_for_admm(transmission_network, model):'),
        'tso_prices_bound_per_block': _one_line(srp, 'transmission_network.network[year][day].cost_energy_p = planning_problem.cost_energy_p[year][day]'),
        'tso_market_probabilities_bound': _one_line(srp, 'transmission_network.network[year][day].prob_market_scenarios = planning_problem.prob_market_scenarios[year]'),
        'market_probabilities_uniform': _one_line(srp, 'planning_problem.prob_market_scenarios[year] = [(1 / planning_problem.num_market_scenarios)] * planning_problem.num_market_scenarios'),
        'price_array_built': _one_line(srp, 'planning_problem.cost_energy_p[year][day] = np.array(energy_selected_profiles * energy_growth_cumul)'),
        'compute_node_load_adn_delta_and_shared_storage': _one_line(mch, 'def compute_node_load(model, i, s_m, s_o, p, network, params):'),
        'node_balance_p_rule': _one_line(mch, 'def node_balance_p_rule(model, i, s_m, s_o, p, network, params):'),
        'tso_interface_pc_adn_def': _one_line(mch, 'def interface_pf_p_transmission_def(m, dn, s_m, s_o, p, network, params):'),
        'tso_pc_fixed_delta_freed_admm': _one_line(srp, '# Fix Pc and Qc (base profiles), free pc_adn and qc_adn'),
        'tso_admm_objective_pf_terms': _one_line(srp, 'def update_transmission_model_to_admm(planning_problem, model, params, objective_scale):'),
        'tso_dual_pf_param_set_from_dual_vars_over_s_base': _one_line(srp, "model[year][day].dual_pf_p_req[dn, p].set_value(dual_pf['current'][node_id][year][day]['p'][p] / s_base)"),
        'pf_dual_update_rule': _one_line(srp, 'def _update_interface_power_flow_variables(planning_problem, tso_model, dso_models, interface_vars, dual_vars, results, params, update_tn=True, update_dns=True):'),
        'aa_pf_dual_antisymmetry': _one_line('admm_anderson_acceleration.py', "dual_vars['pf']['tso']['current'][node_id][year][day][pt][p] = -y"),
        'admm_block_weight': _one_line(srp, 'def _get_admm_block_weight(network_data, year, day):'),
        'esso_operational_objective_feasibility_penalty_only': _one_line('shared_energy_storage_data.py', 'expr=model.feasibility_penalty,'),
        'dso_interface_p_def_storage_at_reference_bus': _one_line(mch, 'def interface_pf_p_distribution_def('),
        'objective_scale_fixed_in_case_file': _one_line(PARAMS_REL, '"objective_scale":'),
    }
    esso_price_refs = _lines('shared_energy_storage_data.py', 'cost_energy_p')
    return cite, esso_price_refs


def t1_case9():
    c9 = _load(CASE9_REL)
    gens = [{'gen_id': g['gen_id'], 'bus': g['bus'], 'type': g['type'], 'Pmin': g['Pmin'], 'Pmax': g['Pmax'],
             'priced_at_market_price': g['type'] in ('CONV', 'REF')} for g in c9['generators']]
    loads = [{'load_id': ld['load_id'], 'bus': ld['bus']} for ld in c9['loads']]
    ref = [n['bus_i'] for n in c9['nodes'] if int(n['type']) == 3]
    return {'baseMVA': c9['baseMVA'], 'generators': gens, 'loads': loads, 'reference_bus': ref,
            'node_order_bus_i': [n['bus_i'] for n in c9['nodes']]}


# ======================================================================================================================
#  snapshot check (the one persisted SOLVED TSO block of each run)
# ======================================================================================================================
def snapshot_check(name, prices, stride6, case9):
    rel = os.path.join(RECORDS[name], SNAPSHOT)
    with open(os.path.join(REPO, rel), 'rb') as handle:
        payload = pickle.load(handle)
    meta = payload['metadata']
    m = payload['model']
    duals = {k.name: v for k, v in m.dual.items()}
    B = float(case9['baseMVA'])
    sigma = pe.value(m.admm_common_objective_scale)
    w_block = pe.value(m.admm_block_weight)
    eff = pe.value(m.admm_objective_scale)
    pi = prices[(SNAPSHOT_YEAR, SNAPSHOT_DAY)]
    node_idx = {b: case9['node_order_bus_i'].index(b) for b in case9['node_order_bus_i']}
    out = {'path': rel, 'metadata': {k: str(v) for k, v in meta.items()},
           'active_objectives': [o.name for o in m.component_objects(pe.Objective, active=True)],
           'admm_common_objective_scale_sigma': sigma, 'admm_block_weight': w_block,
           'admm_objective_scale_eff': eff, 'eff_equals_sigma_over_w': abs(eff - sigma / w_block) <= 1e-6 * eff,
           'interface_settlement_weight': pe.value(m.interface_settlement_weight),
           'rho_pf': pe.value(m.rho_pf),
           'tso_proximal_gamma_pf': (pe.value(m.prox_gamma_pf) if hasattr(m, 'prox_gamma_pf') else None),
           'per_bus': {}}
    kkt_err, anti_err = [], []
    for dn, node in enumerate(ADN_NODES):
        rows = []
        for p in range(24):
            s = stride6[(node, SNAPSHOT_YEAR, SNAPSHOT_DAY, p)]
            r_pu = s['interface_rating'] / B
            y = duals[f'node_balance_p[{node_idx[node]},0,0,{p}]'] / B
            pred = pi[p] + sigma * s['lambda_dso'] / (w_block * s['s_base'] * r_pu * B)
            lam_param_raw = pe.value(m.dual_pf_p_req[dn, p]) * B      # TSO's dual_pf in raw units (x s_base)
            dvar = m.interface_delta_p[dn, 0, 0, p]
            pcv = m.pc[[i for i, ld in enumerate(case9['loads']) if ld['bus'] == node][0], 0, 0, p]
            rows.append({'period': p, 'price_pi': pi[p], 'dual_over_baseMVA': y, 'identity_prediction': pred,
                         'abs_err': abs(y - pred), 'lambda_dso_cycle6': s['lambda_dso'],
                         'tso_dual_pf_param_cycle7_raw': lam_param_raw,
                         'antisymmetry_abs_dev': abs(lam_param_raw + s['lambda_dso']),
                         'interface_delta_p_pu': pe.value(dvar), 'delta_lb': dvar.lb, 'delta_ub': dvar.ub,
                         'delta_fixed': bool(dvar.fixed), 'pc_fixed': bool(pcv.fixed), 'pc_value_pu': pe.value(pcv)})
            kkt_err.append(abs(y - pred))
            anti_err.append(abs(lam_param_raw + s['lambda_dso']))
        out['per_bus'][str(node)] = rows
    gen_rows = []
    for gi, g in enumerate(case9['generators']):
        if g['type'] not in ('CONV', 'REF'):
            continue
        for p in range(24):
            v = m.pg[gi, 0, 0, p]
            margin = min(pe.value(v) - v.lb, v.ub - pe.value(v))
            y = duals[f'node_balance_p[{node_idx[g["bus"]]},0,0,{p}]'] / B
            gen_rows.append({'gen_id': g['gen_id'], 'bus': g['bus'], 'period': p, 'pg_pu': pe.value(v),
                             'bound_margin_pu': margin, 'interior': margin > GEN_INTERIOR_PU,
                             'dual_over_baseMVA': y, 'price_pi': pi[p], 'abs_dual_minus_pi': abs(y - pi[p])})
    interior = [r for r in gen_rows if r['interior']]
    out['conv_generator_buses'] = {
        'n_rows': len(gen_rows), 'n_interior': len(interior),
        'max_abs_dual_minus_pi_interior': max((r['abs_dual_minus_pi'] for r in interior), default=None),
        'max_abs_dual_minus_pi_at_bound': max((r['abs_dual_minus_pi'] for r in gen_rows if not r['interior']),
                                              default=None),
        'rows': gen_rows}
    out['kkt_identity_max_abs_err_eur_per_mwh'] = max(kkt_err)
    out['antisymmetry_max_abs_dev_raw'] = max(anti_err)
    out['delta_all_free_and_interior'] = all(
        (not r['delta_fixed']) and r['delta_lb'] + 1e-9 < r['interface_delta_p_pu'] < r['delta_ub'] - 1e-9
        for rows in out['per_bus'].values() for r in rows)
    out['pc_all_fixed'] = all(r['pc_fixed'] for rows in out['per_bus'].values() for r in rows)
    del payload, m, duals
    gc.collect()
    return out


# ======================================================================================================================
#  T2 -- terminal bus marginal costs
# ======================================================================================================================
def terminal_lmp(name, prices, blocks, case9, sigma, stride_t, cycle_t):
    B = float(case9['baseMVA'])
    rep = _load(os.path.join(RECORDS[name], 'interface_settlement_detail_s31c.json'))['interface_reporting_detail']
    lmp, spreads, delta_rows, price_dev = {}, [], [], 0.0
    for b in blocks:
        yk, dk = b['year'], b['day']
        for node in ADN_NODES:
            series = []
            for p in range(24):
                s = stride_t[(node, yk, dk, p)]
                r_pu = s['interface_rating'] / B
                val = prices[(yk, dk)][p] + sigma * s['lambda_dso'] / (b['w_r2'] * s['s_base'] * r_pu * B)
                series.append(val)
                per = rep[str(node)][yk][dk]['periods'][str(p)]
                price_dev = max(price_dev, abs(per['price_per_mwh'] - prices[(yk, dk)][p]))
                dlt = per['delta_p_mw']['0_0']
                delta_rows.append({'node': node, 'year': yk, 'day': dk, 'period': p, 'delta_p_mw': dlt,
                                   'rating_mw': s['interface_rating'],
                                   'margin_mw': s['interface_rating'] - abs(dlt)})
            lmp[(node, yk, dk)] = series
    return lmp, delta_rows, price_dev


def spread_table(series_by_block, blocks):
    rows = []
    for b in blocks:
        s = series_by_block[(b['year'], b['day'])]
        rows.append({'year': b['year'], 'day': b['day'], 'w_undisc': b['w_undisc'], 'w_r2': b['w_r2'],
                     'min': min(s), 'max': max(s), 'mean': sum(s) / len(s), f'top{H}_minus_bottom{H}': _topbottom(s)})
    agg = {}
    for wk in ('w_undisc', 'w_r2'):
        agg[wk] = sum(r[wk] * r[f'top{H}_minus_bottom{H}'] for r in rows) / sum(r[wk] for r in rows)
    per_year = {}
    for y in sorted({b['year'] for b in blocks}):
        sub = [r for r in rows if r['year'] == y]
        per_year[y] = sum(r['w_undisc'] * r[f'top{H}_minus_bottom{H}'] for r in sub) / sum(r['w_undisc'] for r in sub)
    return rows, agg, per_year


# ======================================================================================================================
#  T3 -- attribution
# ======================================================================================================================
def _agent(block_key):
    parts = block_key.split('|')
    return ('TSO', parts[1]) if parts[0] == 'TSO' else (f'DSO{parts[1]}', parts[2])


def attribution(name_x, blocks):
    lv0 = _load(os.path.join(RECORDS['x0'], 'component_levels_terminal.json'))
    lvx = _load(os.path.join(RECORDS[name_x], 'component_levels_terminal.json'))
    rec0 = _load(os.path.join(RECORDS['x0'], 'evaluation_record.json'))
    recx = _load(os.path.join(RECORDS[name_x], 'evaluation_record.json'))
    chk = {}
    if set(lv0['blocks']) != set(lvx['blocks']):
        raise RuntimeError('block sets differ between records')
    wmap = {(b['year'], b['day']): b for b in blocks}
    comps = GROSS_COMPONENTS + DETECTOR_SUBCOMPONENTS
    table = {}          # (agent, year) -> comp -> value (r2 weighted)
    table_u = {}        # undiscounted
    max_w_dev, max_det_dev = 0.0, 0.0
    for key in lv0['blocks']:
        b0, bx = lv0['blocks'][key], lvx['blocks'][key]
        agent, year = _agent(key)
        day = key.split('|')[-1]
        wb = wmap[(year, day)]
        max_w_dev = max(max_w_dev, abs(b0['admm_block_weight'] - wb['w_r2']), abs(bx['admm_block_weight'] - wb['w_r2']))
        for lv in (b0, bx):
            max_det_dev = max(max_det_dev, abs(sum(lv['unweighted'][c] for c in DETECTOR_SUBCOMPONENTS)
                                               - lv['unweighted']['detector_penalty_total']))
        t = table.setdefault((agent, year), {c: 0.0 for c in comps})
        tu = table_u.setdefault((agent, year), {c: 0.0 for c in comps})
        for c in comps:
            t[c] += b0['weighted'][c] - bx['weighted'][c]
            tu[c] += (b0['unweighted'][c] - bx['unweighted'][c]) * wb['w_undisc']
    value = rec0['certified_cost'] - recx['certified_cost']
    total = sum(v[c] for v in table.values() for c in GROSS_COMPONENTS)
    chk[f'T3_{name_x}_block_weights_match_formula'] = max_w_dev <= 1e-9 * 500
    chk[f'T3_{name_x}_detector_subcomponents_sum_to_detector_total'] = max_det_dev <= 1e-6
    chk[f'T3_{name_x}_parts_sum_to_value_within_0.01_eur'] = abs(total - value) <= SUM_TOL_EUR
    chk[f'T3_{name_x}_value_matches_task_text_2dp'] = abs(round(value, 2) - EXPECTED_VALUE[name_x]) <= 0.005 + 1e-9
    chk[f'T3_{name_x}_gross_equals_net_both_records'] = all(
        abs(r['certified_cost'] - r['terminal_net_operational_recourse']) <= 1e-6 for r in (rec0, recx))
    agents = ['TSO', 'DSO5', 'DSO7', 'DSO9']
    years = sorted({y for (_, y) in table})

    def _rows(tab):
        rows = []
        for a in agents:
            for y in years:
                v = tab[(a, y)]
                rows.append({'agent': a, 'year': y, **{c: v[c] for c in comps},
                             'gross_total': sum(v[c] for c in GROSS_COMPONENTS)})
            rows.append({'agent': a, 'year': 'all', **{c: sum(tab[(a, y)][c] for y in years) for c in comps},
                         'gross_total': sum(tab[(a, y)][c] for y in years for c in GROSS_COMPONENTS)})
        for y in years + ['all']:
            ys = years if y == 'all' else [y]
            rows.append({'agent': 'ALL', 'year': y,
                         **{c: sum(tab[(a, yy)][c] for a in agents for yy in ys) for c in comps},
                         'gross_total': sum(tab[(a, yy)][c] for a in agents for yy in ys for c in GROSS_COMPONENTS)})
        return rows
    out = {'record_x': RECORDS[name_x], 'record_x0': RECORDS['x0'],
           'candidate_key_x': recx['candidate_key'], 'candidate_key_x0': rec0['candidate_key'],
           'Q0_gross_eur': rec0['certified_cost'], 'Qx_gross_eur': recx['certified_cost'],
           'value_eur': value, 'sum_of_parts_eur': total, 'abs_sum_minus_value_eur': abs(total - value),
           'bar_x_eur': recx['bar']['value'], 'bar_x0_eur': rec0['bar']['value'],
           'resolution_bar_sum_eur': recx['bar']['value'] + rec0['bar']['value'],
           'rows_r2': _rows(table), 'rows_undiscounted': _rows(table_u),
           'formula': ('contribution[agent, year, component] = sum over that agent\'s blocks of year y of '
                       '(Q0 - Qx) of the component, "weighted" field of component_levels_terminal.json '
                       '(= unweighted x admm_block_weight, num_years*num_days/1.02^(y-2025)); undiscounted rows use '
                       'unweighted x num_years x num_days. Gross = generation_cost + flexibility_cost_internal + '
                       'load_curtailment_cost + res_curtailment_penalty + ess_usage_cost + detector_penalty_total; '
                       'detector_penalty_total = sum of the six detector subcomponents (shown separately, not added '
                       'twice).')}
    return chk, out


# ======================================================================================================================
#  T4 -- captured-spread decomposition
# ======================================================================================================================
def decomposition(name_x, blocks, years, days, prices, lmp0, lmpx, attr, eff_ch, eff_dch):
    rec = _load(os.path.join(RECORDS[name_x], 'evaluation_record.json'))
    cyc = int(rec['cycles_run'])
    rows, esso_rel = _esso_capture(RECORDS[name_x], cyc, years, days)
    ess = _ess_stride_terminal(RECORDS[name_x], cyc)
    chk = {}
    chk[f'T4_{name_x}_esso_capture_288_rows_single_cohort'] = (len(rows) == 288 and
                                                               all(r['y_inv'] == 0 for r in rows.values()))
    map_dev = max(abs(rows[(b['year'], b['day'], p)]['pnet'] - ess[(b['year'], b['day'])]['x']['esso'][p])
                  for b in blocks for p in range(24))
    chk[f'T4_{name_x}_esso_capture_index_mapping_matches_ess_stride'] = map_dev <= 1e-9
    # EFC recompute with the production efficiencies (validates eff_ch/eff_dch in force and the mapping)
    efc_rec = rec['storage_per_node'][str(NODE)]['efc_per_day_per_cohort_year']
    efc_dev = 0.0
    efc_calc = {}
    for yi, y in enumerate(years):
        avg = 0.0
        for d in days:
            nd = next(b['num_days'] for b in blocks if b['year'] == y and b['day'] == d)
            avg += (nd / 365.0) * sum(eff_ch * rows[(y, d, p)]['pch'] * DT_H + rows[(y, d, p)]['pdch'] * DT_H / eff_dch
                                      for p in range(24))
        efc = avg / (2.0 * E_MWH)
        efc_calc[y] = efc
        ref = efc_rec[f'(0, {yi})']
        efc_dev = max(efc_dev, abs(efc - ref) / abs(ref))
    chk[f'T4_{name_x}_efc_recomputed_with_production_efficiencies'] = efc_dev <= EFC_REL_TOL

    def _conv(wkey):
        T = A0 = Ax = Atso0 = A_ll0 = ideal_pi_tw = ideal_lmp_tw = ideal_lmpx_tw = 0.0
        eloss_ch = eloss_dch = 0.0
        ch_cell = dch_cell = 0.0
        p_ch_num = p_dch_num = e_ch_grid = e_dch_grid = 0.0
        for b in blocks:
            w = b[wkey]
            yk, dk = b['year'], b['day']
            pch = [rows[(yk, dk, p)]['pch'] for p in range(24)]
            pdch = [rows[(yk, dk, p)]['pdch'] for p in range(24)]
            xt = ess[(yk, dk)]['x']['tso']
            l0 = lmp0[(NODE, yk, dk)]
            lx = lmpx[(NODE, yk, dk)]
            Tb = sum(eff_ch * pch[p] + pdch[p] / eff_dch for p in range(24)) * DT_H / 2.0
            T += w * Tb
            ideal_pi_tw += w * Tb * _topbottom(prices[(yk, dk)])
            ideal_lmp_tw += w * Tb * _topbottom(l0)
            ideal_lmpx_tw += w * Tb * _topbottom(lx)
            A0 += w * sum(l0[p] * (pdch[p] - pch[p]) for p in range(24)) * DT_H
            Ax += w * sum(lx[p] * (pdch[p] - pch[p]) for p in range(24)) * DT_H
            Atso0 += w * sum(l0[p] * (-xt[p]) for p in range(24)) * DT_H
            A_ll0 += w * sum(l0[p] * (pdch[p] / eff_dch - eff_ch * pch[p]) for p in range(24)) * DT_H
            eloss_ch += w * sum(l0[p] * pch[p] * (1.0 - eff_ch) for p in range(24)) * DT_H
            eloss_dch += w * sum(l0[p] * pdch[p] * (1.0 / eff_dch - 1.0) for p in range(24)) * DT_H
            ch_cell += w * sum(eff_ch * pch[p] for p in range(24)) * DT_H
            dch_cell += w * sum(pdch[p] / eff_dch for p in range(24)) * DT_H
            p_ch_num += w * sum(l0[p] * pch[p] for p in range(24)) * DT_H
            p_dch_num += w * sum(l0[p] * pdch[p] for p in range(24)) * DT_H
            e_ch_grid += w * sum(pch) * DT_H
            e_dch_grid += w * sum(pdch) * DT_H
        # components of value from T3 (same weighting)
        rows_t3 = attr['rows_r2'] if wkey == 'w_r2' else attr['rows_undiscounted']
        dq = {r['agent']: r['gross_total'] for r in rows_t3 if r['year'] == 'all'}
        dq_tso_gen = next(r['generation_cost'] for r in rows_t3 if r['agent'] == 'TSO' and r['year'] == 'all')
        V = dq['ALL']
        dq_dso = dq['DSO5'] + dq['DSO7'] + dq['DSO9']
        dq_tso = dq['TSO']
        s_pi_dw = sum(b[wkey] * _topbottom(prices[(b['year'], b['day'])]) for b in blocks) / sum(b[wkey] for b in blocks)
        s_lmp_dw = sum(b[wkey] * _topbottom(lmp0[(NODE, b['year'], b['day'])]) for b in blocks) / sum(b[wkey] for b in blocks)
        p_ch_bar = p_ch_num / e_ch_grid
        p_dch_bar = p_dch_num / e_dch_grid
        terms = [
            ('available: market 4 h spread, day-weighted', s_pi_dw),
            ('seasonal/throughput weighting: day-weighted minus throughput-weighted market spread',
             -(s_pi_dw - ideal_pi_tw / T)),
            ('(b) marginal-cost flatness: market minus bus-7 LMP(x=0) 4 h spread, throughput-weighted',
             -(ideal_pi_tw - ideal_lmp_tw) / T),
            ('dispatch timing & depth: ideal top-4/bottom-4 at LMP(x=0) minus the lossless value of the '
             'realized schedule', -(ideal_lmp_tw - A_ll0) / T),
            ('(a) round-trip efficiency losses valued at LMP(x=0) at the hours they occur', -(A_ll0 - A0) / T),
            ('(d) system remainder: value minus the realized arbitrage at LMP(x=0) (price response of the '
             'system to the storage, ADMM stopping); split below into TSO and DN-side parts', (V - A0) / T),
        ]
        split = [
            ('   of which TSO: TSO cost reduction minus the realized arbitrage at LMP(x=0)', (dq_tso - A0) / T),
            ('   of which (c) DN-side: DSO cost change (value contributions of DSO 5/7/9; + = DSO costs fall)', dq_dso / T),
        ]
        captured = V / T
        total = sum(v for _, v in terms)
        return {
            'weighting': wkey, 'T_cell_throughput_full_cycle_mwh': T,
            'captured_eur_per_mwh_cycle': captured, 'sum_of_terms': total, 'abs_sum_minus_captured': abs(total - captured),
            'terms': [{'term': k, 'eur_per_mwh_cycle': v} for k, v in terms],
            'system_remainder_split': [{'term': k, 'eur_per_mwh_cycle': v} for k, v in split],
            'grouped_4way': {
                '(a) efficiency': terms[4][1], '(b) marginal-cost flatness': terms[2][1],
                '(c) DN-side (part of the system remainder)': split[1][1],
                '(d) residual = weighting + timing/depth + TSO part of the system remainder':
                    terms[1][1] + terms[3][1] + split[0][1]},
            'value_minus_A_lmp_x0_eur': V - A0, 'value_minus_A_lmp_x0_over_resolution_bar':
                (V - A0) / (attr['bar_x_eur'] + attr['bar_x0_eur']),
            'resolution_eur_per_mwh_cycle': (attr['bar_x_eur'] + attr['bar_x0_eur']) / T,
            'resolution_note': ('bar(x) + bar(x0) divided by T (bars are r2-weighted gross steps; used for both '
                                'weightings as the stopping-slack scale). A term smaller than this is indeterminate.'),
            'value_eur': V, 'dQ_TSO_eur': dq_tso, 'dQ_TSO_generation_cost_eur': dq_tso_gen, 'dQ_DSO_eur': dq_dso,
            'A_realized_at_lmp_x0_eur': A0, 'A_realized_at_lmp_x_eur': Ax,
            'A_realized_at_lmp_trapezoid_eur': 0.5 * (A0 + Ax),
            'A_tso_copy_at_lmp_x0_eur': Atso0, 'A_lossless_at_lmp_x0_eur': A_ll0,
            'efficiency_loss_charge_eur': eloss_ch, 'efficiency_loss_discharge_eur': eloss_dch,
            'ideal_top4_bottom4_at_pi_throughput_weighted_eur': ideal_pi_tw,
            'ideal_top4_bottom4_at_lmp_x0_throughput_weighted_eur': ideal_lmp_tw,
            'ideal_top4_bottom4_at_lmp_x_throughput_weighted_eur': ideal_lmpx_tw,
            'market_spread_day_weighted': s_pi_dw, 'lmp_x0_spread_day_weighted': s_lmp_dw,
            'cell_energy_in_mwh': ch_cell, 'cell_energy_out_mwh': dch_cell,
            'grid_energy_charged_mwh': e_ch_grid, 'grid_energy_discharged_mwh': e_dch_grid,
            'charge_weighted_lmp_x0': p_ch_bar, 'discharge_weighted_lmp_x0': p_dch_bar,
            'analytic_efficiency_loss_per_cell_cycle_at_weighted_prices':
                p_ch_bar * (1.0 / eff_ch - 1.0) + p_dch_bar * (1.0 - eff_dch),
            'system_remainder_with_lmp_x_eur_per_mwh_cycle': (V - Ax) / T,
            'system_remainder_with_trapezoid_eur_per_mwh_cycle': (V - 0.5 * (A0 + Ax)) / T,
        }
    out = {'record': RECORDS[name_x], 'candidate_key': rec['candidate_key'], 'label': RECORD_LABEL[name_x],
           'esso_capture_file': esso_rel, 'terminal_cycle': cyc,
           'esso_capture_vs_ess_stride_max_abs_dev_mw': map_dev, 'efc_recomputed': efc_calc,
           'efc_recorded': efc_rec, 'efc_max_rel_dev': efc_dev,
           'eff_ch': eff_ch, 'eff_dch': eff_dch, 'round_trip': eff_ch * eff_dch,
           'r2': _conv('w_r2'), 'undiscounted': _conv('w_undisc'),
           'formula': (
               'Throughput T = sum_b w_b sum_p (eff_ch*pch + pdch/eff_dch)*dt / 2 (cell-side MWh of full cycles; '
               'W19 Z4 denominator E*EFC); captured = value / T. Grid injection at bus 7 = pdch - pch (ESSO '
               'terminal capture). A(LMP) = sum w * LMP*(pdch - pch)*dt; A_lossless = sum w * LMP*(pdch/eff_dch - '
               'eff_ch*pch)*dt; efficiency loss = A_lossless - A = sum w*LMP*[pch*(1-eff_ch) + pdch*(1/eff_dch-1)]*dt. '
               'Ideal = sum_b w_b * T_b * top4_minus_bottom4(series_b), T_b the block\'s daily throughput. '
               'value = dQ_TSO + dQ_DSO (T3); system remainder (V - A)/T = (dQ_TSO - A)/T + dQ_DSO/T. Terms telescope: '
               'day-weighted market spread - weighting - flatness - timing/depth - efficiency + system remainder = '
               'value / T exactly.')}
    for wk in ('r2', 'undiscounted'):
        chk[f'T4_{name_x}_{wk}_terms_telescope_to_captured'] = out[wk]['abs_sum_minus_captured'] <= 1e-9 * 100
    return chk, out


# ======================================================================================================================
#  markdown
# ======================================================================================================================
def _markdown(res):
    L = []
    a = L.append
    inst = res['instances']
    a(f'# {STAGE}\n')
    a('Instances (candidate keys): x = 0 `{}`; node 7 0.25 MVA / 1.0 MWh, 2025, nodes 5/9 empty `{}`.'.format(
        inst['x0']['candidate_key'], inst['baseline']['candidate_key']))
    for n in RECORDS:
        a(f'- {inst[n]["label"]}: `{inst[n]["path"]}` (status {inst[n]["status"]}, cycles {inst[n]["cycles_run"]}, '
          f'Q gross {_fmt(inst[n]["certified_cost_gross_eur"])}, bar {_fmt(inst[n]["bar_eur"])}, '
          f'terminal step / threshold {inst[n]["terminal_step_over_threshold"]:.4f})')
    g = res['solve_profile_guard']
    a(f'\nSolve profile: armed SolveProfileGuard(permitted=()); counts {g["counts"]}; verify(0) failures '
      f'{g["verify_0_failures"]}. Git HEAD {res["git_HEAD"]}. all_pass = {res.get("all_pass")}; failed checks: '
      f'{res.get("failed_checks")}.\n')
    a(OBJ_CONV + '\n')

    t1 = res['T1']
    a('## T1 -- how the market price enters the TSO objective\n')
    for line in t1['statements']:
        a(f'- {line}')
    a('\nCitations: ' + '; '.join(f'{k} {v}' for k, v in t1['citations'].items()) + '\n')
    a('Case9 generators: ' + '; '.join(f'gen {gg["gen_id"]} bus {gg["bus"]} {gg["type"]}'
                                     f'{" (priced at pi)" if gg["priced_at_market_price"] else " (no cost term)"}'
                                     for gg in t1['case9']['generators']) +
      '. Loads: ' + ', '.join(f'bus {ld["bus"]}' for ld in t1['case9']['loads']) + ' (exactly the three ADN interfaces).\n')

    t2 = res['T2']
    a('## T2 -- TSO bus-7 nodal marginal cost at x = 0\n')
    a(f'**Recoverability.** {t2["recoverability"]}\n')
    a(f'Formula: {LMP_FORMULA}\n')
    a('Verification on the persisted solved TSO block of each run (2025 Summer; duals of the cycle-6 solve carried '
      'by the cycle-7 pre-solve snapshot; buses 5, 7, 9 x 24 periods):\n')
    a('| run | max abs(identity error) EUR/MWh | antisymmetry max dev (raw) | delta free & interior | pc fixed | '
      'sigma | w_block | CONV interior rows | max abs(dual - pi) interior CONV |')
    a('|---|---|---|---|---|---|---|---|---|')
    for n, sc in t2['snapshot_checks'].items():
        cg = sc['conv_generator_buses']
        a(f'| {n} | {sc["kkt_identity_max_abs_err_eur_per_mwh"]:.3e} | {sc["antisymmetry_max_abs_dev_raw"]:.3e} | '
          f'{sc["delta_all_free_and_interior"]} | {sc["pc_all_fixed"]} | {sc["admm_common_objective_scale_sigma"]:,.0f} | '
          f'{sc["admm_block_weight"]:.4f} | {cg["n_interior"]}/{cg["n_rows"]} | '
          f'{cg["max_abs_dual_minus_pi_interior"] if cg["max_abs_dual_minus_pi_interior"] is None else format(cg["max_abs_dual_minus_pi_interior"], ".3e")} |')
    a('')
    a(f'Terminal validity: interface_delta_p margin to its bound, minimum over every block/period/node of the x = 0 '
      f'record = {t2["x0_delta_min_margin_mw"]:.3f} MW (required >= {DELTA_INTERIOR_MW} MW). Prices in the run '
      f'vs Z4 arrays: max abs dev {t2["x0_price_max_abs_dev"]:.3e}.\n')
    a(f'Spread definition: {SPREAD_FORMULA}\n')
    a('Units / sign: the dual is EUR per p.u.-hour of the representative-day objective; divided by baseMVA = 100 it is '
      'EUR/MWh; positive = the cost of serving one more MWh of load at bus 7 (equivalently the value of one more '
      'MWh injected there). The storage (load convention: charging positive) sees -LMP per MWh injected.\n')
    a('| year | day | market 4 h spread | bus-7 LMP 4 h spread (x = 0) | difference | market mean | LMP mean | '
      'max abs(LMP - pi) | LMP spread change over the last cycle |')
    a('|---|---|---|---|---|---|---|---|---|')
    for r in t2['per_day']:
        a(f'| {r["year"]} | {r["day"]} | {r["market_spread"]:.3f} | {r["lmp_spread"]:.3f} | '
          f'{r["market_spread"] - r["lmp_spread"]:+.3f} | {r["market_mean"]:.3f} | {r["lmp_mean"]:.3f} | '
          f'{r["max_abs_lmp_minus_pi"]:.3f} | {r["lmp_spread_change_last_cycle"]:+.2e} |')
    a('')
    a('| scope | market 4 h spread | bus-7 LMP 4 h spread (x = 0) | ratio LMP / market |')
    a('|---|---|---|---|')
    for k, v in t2['aggregates'].items():
        a(f'| {k} | {v["market"]:.3f} | {v["lmp_x0"]:.3f} | {v["ratio"]:.5f} |')
    a('')
    a('Bus-7 LMP 4 h spread with the storage in place (same identity, terminal cycle of each record), horizon '
      'average: ' + '; '.join(f'{k}: undiscounted {v["w_undisc"]:.3f}, r2 {v["w_r2"]:.3f}'
                               for k, v in t2['lmp_with_storage_spreads'].items()) + '\n')
    a(f'**Reading.** {t2["reading"]}\n')

    for name in ('baseline', 'sensitivity_c3'):
        at = res['T3'][name]
        title = 'BASELINE' if name == 'baseline' else 'SENSITIVITY SET (C3; never compared in one table with the baseline)'
        a(f'## T3 -- attribution of value = Q(0) - Q(x): {title}\n')
        a(f'Record x `{at["record_x"]}` (candidate_key `{at["candidate_key_x"]}`) against x = 0 `{at["record_x0"]}` '
          f'(candidate_key `{at["candidate_key_x0"]}`). Q(0) = {_fmt(at["Q0_gross_eur"])}, Q(x) = {_fmt(at["Qx_gross_eur"])}, '
          f'value = {_fmt(at["value_eur"])}; sum of parts = {_fmt(at["sum_of_parts_eur"])} '
          f'(|diff| {at["abs_sum_minus_value_eur"]:.2e} EUR). Resolution bar(x) + bar(x0) = {_fmt(at["resolution_bar_sum_eur"])}.\n')
        a(OBJ_CONV + ' Weighted at 2 % (the model block weights; this is the convention of Q).\n')
        a('| agent | year | generation_cost | flexibility_cost_internal | load_curtailment | res_curtailment | '
          'ess_usage | detector_penalty_total | gross total |')
        a('|---|---|---|---|---|---|---|---|---|')
        for r in at['rows_r2']:
            a(f'| {r["agent"]} | {r["year"]} | {_fmt(r["generation_cost"])} | {_fmt(r["flexibility_cost_internal"])} | '
              f'{_fmt(r["load_curtailment_cost"])} | {_fmt(r["res_curtailment_penalty"])} | {_fmt(r["ess_usage_cost"])} | '
              f'{_fmt(r["detector_penalty_total"])} | {_fmt(r["gross_total"])} |')
        a('')
        a('Detector subcomponents (all agents, all years, r2): ' + ', '.join(
            f'{c} {_fmt(next(r[c] for r in at["rows_r2"] if r["agent"] == "ALL" and r["year"] == "all"))}'
            for c in DETECTOR_SUBCOMPONENTS) + '. Undiscounted rows are in the JSON.\n')

    for name in ('baseline', 'sensitivity_c3'):
        de = res['T4'][name]
        title = 'BASELINE' if name == 'baseline' else 'SENSITIVITY SET (C3)'
        a(f'## T4 -- captured spread decomposition: {title}\n')
        a(f'Record `{de["record"]}` (candidate_key `{de["candidate_key"]}`), ESSO terminal capture `{de["esso_capture_file"]}`. '
          f'Efficiencies in force: eff_ch = {de["eff_ch"]}, eff_dch = {de["eff_dch"]} (round trip {de["round_trip"]:.4f}); '
          f'confirmed by recomputing the recorded EFC/day (max rel dev {de["efc_max_rel_dev"]:.1e}).\n')
        a(f'Formula: {de["formula"]}\n')
        for wk in ('r2', 'undiscounted'):
            d = de[wk]
            a(f'**Weighting {wk}** ({OBJ_CONV.split(";")[0]}; per MWh of full cell cycle). T = {d["T_cell_throughput_full_cycle_mwh"]:.2f} MWh-cycles, '
              f'value {_fmt(d["value_eur"])}, captured {d["captured_eur_per_mwh_cycle"]:.3f} EUR/MWh-cycle; resolution '
              f'(bars / T) {d["resolution_eur_per_mwh_cycle"]:.3f}.\n')
            a('| term | EUR/MWh-cycle |')
            a('|---|---|')
            for t in d['terms']:
                a(f'| {t["term"]} | {t["eur_per_mwh_cycle"]:+.3f} |')
            a(f'| **= captured (value / T)** | **{d["captured_eur_per_mwh_cycle"]:.3f}** |')
            for t in d['system_remainder_split']:
                a(f'| {t["term"]} (not added again) | {t["eur_per_mwh_cycle"]:+.3f} |')
            a('')
            a('Four-way grouping: ' + '; '.join(f'{k} {v:+.3f}' for k, v in d['grouped_4way'].items()) +
              f'. Charge-weighted LMP {d["charge_weighted_lmp_x0"]:.2f}, discharge-weighted LMP {d["discharge_weighted_lmp_x0"]:.2f} EUR/MWh; '
              f'analytic efficiency loss per cell cycle at these prices p_ch (1/eff_ch - 1) + p_dch (1 - eff_dch) = '
              f'{d["analytic_efficiency_loss_per_cell_cycle_at_weighted_prices"]:.3f} EUR/MWh. System remainder with LMP(x) '
              f'{d["system_remainder_with_lmp_x_eur_per_mwh_cycle"]:+.3f}, with the trapezoid {d["system_remainder_with_trapezoid_eur_per_mwh_cycle"]:+.3f}. '
              f'value - A(LMP x=0) = {_fmt(d["value_minus_A_lmp_x0_eur"])} EUR = {d["value_minus_A_lmp_x0_over_resolution_bar"]:+.3f} x the resolution bar. '
              f'dQ_TSO {_fmt(d["dQ_TSO_eur"])} (generation cost {_fmt(d["dQ_TSO_generation_cost_eur"])}); A at LMP(x=0) '
              f'{_fmt(d["A_realized_at_lmp_x0_eur"])} (TSO copy {_fmt(d["A_tso_copy_at_lmp_x0_eur"])}); dQ_DSO {_fmt(d["dQ_DSO_eur"])}.\n')

    t5 = res['T5']
    a('## T5 -- planning-signal magnitude (baseline points)\n')
    a(OBJ_CONV + '\n')
    a('| quantity | EUR | / Q(0) | log10 |')
    a('|---|---|---|---|')
    for r in t5['rows']:
        a(f'| {r["quantity"]} | {_fmt(r["eur"])} | {r["ratio"]:.3e} | {r["log10"]:.3f} |')
    a(f'\n{t5["reading"]}\n')

    rr = res['R_restatement']
    a('## Paper-scale R on the marginal-cost spread\n')
    a(rr['statement'] + '\n')
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
                'instance_candidate_keys': {n: res['instances'][n]['candidate_key'] for n in RECORDS},
                'script': {'path': os.path.basename(__file__), 'sha256': _sha256(os.path.abspath(__file__))},
                'files': files,
                'evidence_inputs_sha256': {
                    **{f['path']: f['sha256'] for n in RECORDS for f in res['instances'][n]['files'].values()},
                    **{k: v['sha256'] for k, v in res['instances']['static_inputs'].items()}}}
    path = os.path.join(OUT_DIR, MANIFEST_NAME)
    if os.path.exists(path):
        raise SystemExit(f'refusing to overwrite the manifest: {path}')
    with open(path, 'w') as handle:
        json.dump(manifest, handle, indent=1)
    print(f'[W25] manifest: {len(files)} files -> {os.path.relpath(path, REPO)}', flush=True)


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
    print(f'[W25] start {started}; records {RECORDS}', flush=True)
    res = {'stage': STAGE, 'authority': AUTHORITY, 'started_utc': started,
           'git_HEAD': _git(['rev-parse', 'HEAD']).strip(), 'objective_convention': OBJ_CONV,
           'out_dir': OUT_DIR,
           'spread_formula': SPREAD_FORMULA, 'lmp_formula': LMP_FORMULA,
           'tolerances_stated_before_run': {'KKT_TOL_eur_per_mwh': KKT_TOL, 'DELTA_INTERIOR_MW': DELTA_INTERIOR_MW,
                                            'SUM_TOL_EUR': SUM_TOL_EUR, 'REPRO_TOL': REPRO_TOL,
                                            'EFC_REL_TOL': EFC_REL_TOL, 'GEN_INTERIOR_PU': GEN_INTERIOR_PU}}
    checks, errors = {}, {}
    try:
        # ---- capture checklist BEFORE any computation (fail fast) ----
        c, inst = capture_checklist()
        checks.update(c)
        res['instances'] = inst
        missing = [k for k, v in c.items() if not v]
        print(f'[W25] capture checklist: {len(c)} items, failed {missing}', flush=True)
        if missing:
            raise RuntimeError(f'capture checklist failed before execution: {missing}')

        case, years, days, blocks = case_structure()
        prices, repro = market_prices(blocks)
        checks['Z4_market_spread_reproduced_undiscounted'] = abs(repro['undiscounted'] - SRP1_Z4_SPREAD) <= REPRO_TOL
        checks['Z4_market_spread_reproduced_r2'] = abs(repro['r2'] - SRP1_Z4_SPREAD_R2) <= REPRO_TOL
        case9 = t1_case9()
        params = _load(PARAMS_REL)
        scale_vals = _find_key(params, 'objective_scale')
        res['case_file_objective_scale'] = scale_vals

        # ---- T1 ----
        cite, esso_price_refs = t1_citations()
        checks['T1_esso_model_file_has_no_market_price_reference'] = esso_price_refs == []
        checks['T1_tso_only_loads_are_the_three_adn_interfaces'] = sorted(ld['bus'] for ld in case9['loads']) == list(ADN_NODES)
        checks['T1_no_priced_generator_at_bus7'] = not any(g['bus'] == NODE and g['priced_at_market_price'] for g in case9['generators'])

        # ---- strides + snapshot checks ----
        stride = {}
        snap = {}
        for n in RECORDS:
            cyc = inst[n]['cycles_run']
            keep, n_lines, last_cycle = _stride_terminal(RECORDS[n], cyc)
            checks[f'T2_{n}_pf_stride_last_cycle_is_terminal'] = (last_cycle == cyc and n_lines == cyc)
            stride[n] = keep
            snap[n] = snapshot_check(n, prices, keep[SNAPSHOT_CYCLE - 1], case9)
            checks[f'T2_{n}_snapshot_identity_within_{KKT_TOL}'] = snap[n]['kkt_identity_max_abs_err_eur_per_mwh'] <= KKT_TOL
            checks[f'T2_{n}_snapshot_delta_free_interior_pc_fixed'] = snap[n]['delta_all_free_and_interior'] and snap[n]['pc_all_fixed']
            checks[f'T2_{n}_snapshot_eff_equals_sigma_over_w'] = snap[n]['eff_equals_sigma_over_w']
            checks[f'T2_{n}_snapshot_settlement_weight_1_proximal_off'] = (
                snap[n]['interface_settlement_weight'] == 1.0 and snap[n]['tso_proximal_gamma_pf'] in (None, 0.0))
            checks[f'T2_{n}_sigma_equals_case_file_objective_scale'] = (
                len(scale_vals) == 1 and snap[n]['admm_common_objective_scale_sigma'] == float(scale_vals[0]))
            print(f'[W25] {n}: snapshot identity max err {snap[n]["kkt_identity_max_abs_err_eur_per_mwh"]:.3e}, '
                  f'antisym {snap[n]["antisymmetry_max_abs_dev_raw"]:.3e}', flush=True)
        res['stride_reading'] = {n: {'terminal_cycle': inst[n]['cycles_run']} for n in RECORDS}

        # ---- T2 ----
        lmp, lmp_prev, delta_min, price_dev = {}, {}, {}, {}
        for n in RECORDS:
            sigma = snap[n]['admm_common_objective_scale_sigma']
            cyc = inst[n]['cycles_run']
            lmp[n], drows, price_dev[n] = terminal_lmp(n, prices, blocks, case9, sigma, stride[n][cyc], cyc)
            lmp_prev[n], _, _ = terminal_lmp(n, prices, blocks, case9, sigma, stride[n][cyc - 1], cyc - 1)
            delta_min[n] = min(r['margin_mw'] for r in drows)
            checks[f'T2_{n}_terminal_interface_delta_interior_every_block_period'] = delta_min[n] >= DELTA_INTERIOR_MW
            checks[f'T2_{n}_run_prices_equal_z4_arrays'] = price_dev[n] <= 1e-9
        per_day = []
        for b in blocks:
            s0 = lmp['x0'][(NODE, b['year'], b['day'])]
            sp = lmp_prev['x0'][(NODE, b['year'], b['day'])]
            pi = prices[(b['year'], b['day'])]
            per_day.append({'year': b['year'], 'day': b['day'], 'w_undisc': b['w_undisc'], 'w_r2': b['w_r2'],
                            'market_spread': _topbottom(pi), 'lmp_spread': _topbottom(s0),
                            'market_mean': sum(pi) / 24, 'lmp_mean': sum(s0) / 24,
                            'max_abs_lmp_minus_pi': max(abs(s0[p] - pi[p]) for p in range(24)),
                            'lmp_spread_change_last_cycle': _topbottom(s0) - _topbottom(sp),
                            'lmp_series': s0, 'price_series': pi})
        agg = {}
        for wk, lab in (('w_undisc', 'all years, day-weighted (undiscounted)'), ('w_r2', 'all years, model block weight (2 %)')):
            m_ = sum(r[wk] * r['market_spread'] for r in per_day) / sum(r[wk] for r in per_day)
            l_ = sum(r[wk] * r['lmp_spread'] for r in per_day) / sum(r[wk] for r in per_day)
            agg[lab] = {'market': m_, 'lmp_x0': l_, 'ratio': l_ / m_}
        for y in years:
            sub = [r for r in per_day if r['year'] == y]
            m_ = sum(r['w_undisc'] * r['market_spread'] for r in sub) / sum(r['w_undisc'] for r in sub)
            l_ = sum(r['w_undisc'] * r['lmp_spread'] for r in sub) / sum(r['w_undisc'] for r in sub)
            agg[f'{y}, day-weighted'] = {'market': m_, 'lmp_x0': l_, 'ratio': l_ / m_}
        other_nodes = {}
        for node in (5, 9):
            ser = {(b['year'], b['day']): lmp['x0'][(node, b['year'], b['day'])] for b in blocks}
            _, a_, _ = spread_table(ser, blocks)
            other_nodes[str(node)] = a_
        with_storage = {}
        for n in ('baseline', 'sensitivity_c3'):
            ser = {(b['year'], b['day']): lmp[n][(NODE, b['year'], b['day'])] for b in blocks}
            _, a_, _ = spread_table(ser, blocks)
            with_storage[n] = a_
        und = agg['all years, day-weighted (undiscounted)']
        res['T2'] = {
            'recoverability': (
                'The terminal bus-7 duals are NOT persisted by any of the three records (searched, W25 exploration: '
                'the file listing of the three record directories incl. results/ and esso_capture/, and the key names '
                'of every *.json / *.jsonl / *.log of the x0 directory for dual / lmp / lambda / marginal / shadow; '
                'the persisted pickles are the ESSO models (esso_models_s39_D.pkl) and the cycle-7 2025-Summer TSO '
                '(Summer) and DSO-node-7 (Autumn) pre-solve snapshots; esso_capture/ holds ESSO rows only, empty at '
                'x = 0). They ARE recoverable without a solve through an exact '
                'first-order identity of the TSO ADMM block (formula below), verified on that snapshot in every run '
                'to the stated tolerance, and applied to the terminal PF duals recorded every cycle in '
                'pf_entry_stride_s39_D.jsonl (hash-settled by the committed child manifest).'),
            'snapshot_checks': snap,
            'x0_delta_min_margin_mw': delta_min['x0'], 'delta_min_margin_mw_per_record': delta_min,
            'x0_price_max_abs_dev': price_dev['x0'],
            'per_day': per_day, 'aggregates': agg,
            'other_adn_buses_x0_lmp_spreads': other_nodes,
            'lmp_with_storage_spreads': with_storage,
            'reading': (
                f'At x = 0 the 4 h spread of the bus-7 marginal cost is {und["lmp_x0"]:.3f} EUR/MWh against the '
                f'market\'s {und["market"]:.3f} (ratio {und["ratio"]:.4f}, undiscounted day weights; per year '
                + ', '.join(f'{y} {agg[f"{y}, day-weighted"]["ratio"]:.4f}' for y in years)
                + f'). The largest hourly deviation |LMP - pi| on a representative day is '
                f'{max(r["max_abs_lmp_minus_pi"] for r in per_day):.2f} EUR/MWh; the largest change of a daily '
                f'LMP spread over the last ADMM cycle is {max(abs(r["lmp_spread_change_last_cycle"]) for r in per_day):.2e} '
                f'EUR/MWh (settled). With the storage in place the spread is '
                f'{with_storage["baseline"]["w_undisc"]:.3f} (baseline) / {with_storage["sensitivity_c3"]["w_undisc"]:.3f} '
                f'(C3), undiscounted. Which TSO rows make the bus-7 marginal cost depart from pi (branch limits, '
                f'voltage limits, losses) is NOT identifiable from the persisted data: the only persisted solved TSO '
                f'block is 2025 Summer at cycle 6 (max |LMP_7 - pi| there '
                f'{max(abs(r["dual_over_baseMVA"] - r["price_pi"]) for r in snap["x0"]["per_bus"][str(NODE)]):.3f} EUR/MWh).')}
        print(f'[W25] T2: market {und["market"]:.4f} vs LMP(x0) {und["lmp_x0"]:.4f} ratio {und["ratio"]:.5f}', flush=True)

        # ---- T1 statements (after the snapshot facts are known) ----
        sx = snap['x0']
        res['T1'] = {
            'citations': cite, 'esso_model_file_cost_energy_p_lines': esso_price_refs, 'case9': case9,
            'statements': [
                'The TSO pays the market price on its generation: generation_cost = sum over controllable generators '
                '(CONV, REF, controllable RES) of c_p[s_m][p] * baseMVA * pg, with c_p = network.cost_energy_p[s_m] '
                '(the per-(year, day) market price array of scenario s_m); total_generation_cost weights each '
                'scenario by prob_market[s_m] * prob_operation[s_o]. On SRP1 there is one market and one operation '
                f'scenario (weight 1). In case9 (2025 file) the priced generators are '
                + ', '.join(f'{g["type"]} gen {g["gen_id"]} at bus {g["bus"]}' for g in case9['generators'] if g['priced_at_market_price'])
                + '; unpriced: '
                + ', '.join(f'{g["type"]} gen {g["gen_id"]} at bus {g["bus"]}' for g in case9['generators'] if not g['priced_at_market_price'])
                + f'. Generators at bus {NODE}: '
                + str([g['gen_id'] for g in case9['generators'] if g['bus'] == NODE]) + '.',
                'The TSO earns the market price on the interface energy it delivers: interface_energy_settlement = '
                '-sum prob * c_p[p] * baseMVA * pc_adn[dn] for every ADN bus (5, 7, 9), weight 1 in the ADMM path '
                f'(snapshot: interface_settlement_weight = {sx["interface_settlement_weight"]}); the DSO pays +c_p * pg_adn. '
                'The settlement is excluded from the reported Q(x).',
                'Bus 7 connects to it through node_balance_p[7]: load = pc (FIXED at the build-time consensus) + '
                'interface_delta_p (FREE within +/- the interface rating in p.u.) + shared_es_pnet (load convention). '
                'The TSO\'s only loads are the three ADN interfaces.',
                'The storage does NOT face the market price directly: the ESSO objective is its feasibility penalty '
                '(plus ADMM terms) and shared_energy_storage_data.py contains no reference to cost_energy_p. Its '
                'economic signal is the consensus duals of the TSO and DSO copies. In the TSO its injection enters '
                'node_balance_p[7] only, so the TSO values it at the bus-7 dual y_7; in the DSO it enters the '
                'reference-bus balance and pg_adn and cancels (W19 Z1), so the DSO copy carries no price. The storage '
                'therefore sees the bus-7 marginal cost, and y_7 = pi + (interface consensus correction) by the '
                'identity of T2 -- the market price reaches it only through the bus-7 nodal balance.']}

        # ---- T3 ----
        res['T3'] = {}
        for n in ('baseline', 'sensitivity_c3'):
            c, out = attribution(n, blocks)
            checks.update(c)
            res['T3'][n] = out
            print(f'[W25] T3 {n}: value {out["value_eur"]:.2f}, parts {out["sum_of_parts_eur"]:.2f}', flush=True)

        # ---- T4 ----
        from shared_energy_storage import SharedEnergyStorage
        _ses = SharedEnergyStorage()
        eff_ch, eff_dch = float(_ses.eff_ch), float(_ses.eff_dch)
        overrides = _lines('shared_energy_storage_data.py', 'eff_ch =')
        res['efficiency_source'] = {
            'eff_ch': eff_ch, 'eff_dch': eff_dch,
            'defined_at': [_one_line('shared_energy_storage.py', 'self.eff_ch = '),
                           _one_line('shared_energy_storage.py', 'self.eff_dch = ')],
            'assignments_named_eff_ch_in_shared_energy_storage_data_py': overrides,
            'note': ('SharedEnergyStorage class defaults; the only file-driven eff_ch/eff_dch assignment in the '
                     'production modules is network.py (ordinary ESS). Confirmed empirically: the recorded EFC/day '
                     'is recomputed from the ESSO capture with these values (check T4_*_efc_recomputed...).')}
        res['T4'] = {}
        for n in ('baseline', 'sensitivity_c3'):
            c, out = decomposition(n, blocks, years, days, prices, lmp['x0'], lmp[n], res['T3'][n], eff_ch, eff_dch)
            checks.update(c)
            res['T4'][n] = out
            print(f'[W25] T4 {n}: captured r2 {out["r2"]["captured_eur_per_mwh_cycle"]:.3f}, '
                  f'undiscounted {out["undiscounted"]["captured_eur_per_mwh_cycle"]:.3f}', flush=True)
        checks['T4_sensitivity_c3_reproduces_W19_captured_52.11'] = abs(
            res['T4']['sensitivity_c3']['r2']['captured_eur_per_mwh_cycle']
            - _load(Z4_JSON_REL)['Z4']['captured_spread']['c_consistent_discounted_V2pct_over_discounted_EFC']) <= 1e-6

        # ---- T5 ----
        pa = _load(PHASE_A_TABLES_REL)['T3']
        q0 = inst['x0']['certified_cost_gross_eur']
        vb = res['T3']['baseline']['value_eur']
        rows5 = []
        for label, eur in (
                ('value (baseline smallest node-7 unit)', vb),
                ('sigma_Q = s (Phase A T3 regression residual std, dof 13)', pa['s_eur']),
                ('sigma_Q = residual rms (Phase A T3)', pa['residual_rms_eur']),
                ('sigma_Q = residual max abs (Phase A T3)', pa['residual_max_abs_eur']),
                ('bar(x0)', inst['x0']['bar_eur']),
                ('bar(x) baseline', inst['baseline']['bar_eur']),
                ('bar(x) + bar(x0) (resolution of the baseline value)', inst['x0']['bar_eur'] + inst['baseline']['bar_eur'])):
            rows5.append({'quantity': label, 'eur': eur, 'ratio': eur / q0, 'log10': math.log10(eur / q0)})
        res['T5'] = {'Q0_gross_eur': q0, 'rows': rows5, 'phase_a_tables_commit': PHASE_A_COMMIT,
                     'reading': (f'value / Q(0) = {vb / q0:.3e} = 10^{math.log10(vb / q0):.2f}: the planning signal is '
                                 f'{-math.log10(vb / q0):.2f} orders of magnitude below the recourse, i.e. between '
                                 f'{math.floor(-math.log10(vb / q0))} and {math.ceil(-math.log10(vb / q0))} '
                                 f'(1 part in {q0 / vb:,.0f}); '
                                 f'sigma_Q / Q(0) = 10^{math.log10(pa["residual_rms_eur"] / q0):.2f} to '
                                 f'10^{math.log10(pa["residual_max_abs_eur"] / q0):.2f}; bar(x)+bar(x0) / Q(0) = '
                                 f'10^{math.log10((inst["x0"]["bar_eur"] + inst["baseline"]["bar_eur"]) / q0):.2f}; '
                                 f'the value exceeds sigma_Q by {vb / pa["residual_max_abs_eur"]:.1f}x to '
                                 f'{vb / pa["residual_rms_eur"]:.1f}x.')}

        # ---- R restatement ----
        w22 = _load(W22_JSON_REL)['ratios']['values']
        ratio_u = agg['all years, day-weighted (undiscounted)']['ratio']
        per_year_ratio = {y: agg[f'{y}, day-weighted']['ratio'] for y in years}
        R_m = w22['R = mean_profile / SRP1 (undiscounted)']
        lo, hi = min(per_year_ratio.values()), max(per_year_ratio.values())
        res['R_restatement'] = {
            'W22_R_market_spread': R_m,
            'SRP1_lmp_to_market_spread_ratio_undiscounted': ratio_u,
            'SRP1_lmp_to_market_spread_ratio_per_year': per_year_ratio,
            'formula': ('R_LMP = (paper bus-7 LMP spread) / (SRP1 bus-7 LMP spread) = R_market * f_paper / f_SRP1, '
                        'f = LMP 4 h spread / market 4 h spread (undiscounted day weights); f_paper is unknown '
                        'without paper-scale TSO duals.'),
            'conditional_range_if_f_paper_within_SRP1_per_year_range': [R_m * lo / ratio_u, R_m * hi / ratio_u],
            'statement': (
                f'The storage sees the bus-7 marginal cost (T1), whose SRP1 4 h spread is f_SRP1 = {ratio_u:.4f} of the '
                f'market spread (per year ' + ', '.join(f'{y} {v:.4f}' for y, v in per_year_ratio.items()) + '). '
                'Recomputing W22\'s paper-scale R on the marginal-cost spread needs the paper-scale bus-7 duals at an '
                'ADMM equilibrium; no paper-scale ADMM evaluation has reached one (searched: '
                'data/SRP1/Results/P515S44/scale_measurement -- paper_cycle_snapoff_r1/r2 are one-cycle timing runs, '
                'both watchdog-aborted, exit 97; the "paper_plan" records under P515S44 are SRP1-scale runs of the '
                'paper\'s plan), so it CANNOT be computed without solves and is not computed. '
                f'R on the market spread (W22 @ {W22_COMMIT}: {R_m:.4f}) equals R on the marginal-cost spread only if '
                f'f_paper = f_SRP1. Restated prediction: R_LMP = {R_m:.4f} x f_paper / {ratio_u:.4f}. Conditional '
                f'range, ASSUMING f_paper lies within SRP1\'s per-year range [{lo:.4f}, {hi:.4f}]: R_LMP in '
                f'[{R_m * lo / ratio_u:.4f}, {R_m * hi / ratio_u:.4f}] (an assumption, not a measurement). '
                'Cheapest solve-bearing route: none extra -- the planned paper-scale x = 0 evaluation, run with the '
                'PF entry-stride capture (lambda_dso per cycle) and one TSO snapshot for the identity check, yields f_paper '
                'at zero additional solves; the identity must be re-derived there for 5 x 5 scenarios (probability '
                'weights and the scenario-deviation penalty enter the stationarity conditions).')}
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
        return str(o)
    with open(res_path, 'x') as handle:
        json.dump(res, handle, indent=1, default=_default)
    if not errors:
        with open(md_path, 'x') as handle:
            handle.write(_markdown(res))
    os.remove(lock)
    GUARD.uninstall()
    print(f"[W25] checks={len(checks)} failed={res['failed_checks']} errors={sorted(errors)} "
          f"guard={dict(GUARD.counts)} verify0_failures={failures} all_pass={res['all_pass']}", flush=True)
    if not res['all_pass']:
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        GUARD.uninstall()
        _manifest()
        sys.exit(0)
    main()
