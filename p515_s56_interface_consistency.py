"""P5.15 S56 -- audit D: scenario-wise TN-DN interface consistency (ZERO SOLVES; no unpickling).

Question (author / Expert, before any multi-scenario search): does the executed multi-scenario operational formulation
produce physically consistent TN-DN exchanges in each joint scenario?

What this script does:
  1. CODE ANCHORS. Verifies that every file:line the report cites still holds the quoted text, and records the sha256 of
     every cited module, so the trace in the report is pinned to the source it was read from.
  2. NETWORK FACTS (raw case files, the files production reads): baseMVA of the TN and of each DN; base kV of the TN
     ADN bus and of the DN reference bus; the DN branch(es) at the reference bus (the interface transformer) and its
     rating; the DN reference generator's bounds -- for the five 3 x 3 years.
  3. PAIRED DISCREPANCIES from the saved multi-scenario terminal captures (`multiscenario_terminal.json`, written by
     `p515_s44_campaign_harness.multiscenario_terminal_capture` at the TERMINAL point of each run off the run's own
     final models; per-scenario interface V / P / Q from production `Network.process_results_interface`, i.e.
       TSO side:  P = pc_adn[dn, s, t] * baseMVA  (`interface_pf_p_transmission_def`: pc + interface_delta_p),
                  V = |V| of the TN ADN bus;
       DSO side:  P = pg_adn[s, t] * baseMVA      (`interface_pf_p_distribution_def`: pg_ref - shared-ESS pnet),
                  V = |V| of the DN reference bus;
     both positive in the DN-import direction). Every capture read is sha256-checked against the child manifest that
     pins it. For each (DN, year, day, scenario key s = '<s_m>_<s_o>', hour t):
       dP = P_TSO[s,t] - P_DSO[s,t]       dQ likewise       dV = V_TSO[s,t] - V_DSO[s,t]   (p.u., same base kV)
       decomposition  dP = c_t - d[s,t],  c_t = Pbar_TSO[t] - Pbar_DSO[t] (the expected-value consensus residual the
                      ADMM drives to zero), d[s,t] = P_DSO[s,t] - Pbar_DSO[t] (the DSO's deviation from its own
                      commitment, the quantity row 18 prices); checked numerically (P_TSO[s] = Pbar_TSO, common)
       shared ESS     dP_ess[t] = pnet_TSO-copy[t] - pnet_DSO-copy[t] (both scenario-free); dP_total = dP + dP_ess
                      (TN withdrawal at the bus pc_adn + pnet_TN minus DN reference injection pg_adn + pnet_DN)
     Metrics: max |.| (with argmax), probability-weighted RMS
       rms = sqrt( sum_b sum_s omega_s sum_t x^2 / (N_b * 24) )                 (representative days equal)
       rms_w = sqrt( sum_b w_b sum_s omega_s sum_t x^2 / (sum_b w_b * 24) )     (w_b = admm_block_weight, Q weighting)
     per DN and pooled over the three DNs; plus the TN-side and DN-side scenario spreads max_s - min_s.
  4. ENERGY NOT SUPPLIED BY ANY TN MODEL: u[s,t] = P_DSO[s,t] - P_TSO[s,t] = -dP (MW, 1-hour periods => MWh):
       E_plus = sum_s omega_s sum_t max(u, 0)   drawn by the DN in scenario s, not carried by the TN model
       E_minus = sum_s omega_s sum_t max(-u, 0) carried by the TN model in scenario s, not taken by the DN
     per representative day (unweighted sum over blocks) and block-weighted (sum_b w_b E_b; w_b includes the 1.02^-k
     discount, so "discounted MWh"); valued at the scenario price pi[s_m, t] (the DSO block's own market price array,
     `dispersion.pi_by_market_by_hour`). Reconciled against production's own totals:
       sum_b w_b sum_s omega_s sum_t pi u == T_TSO + sum T_DSO  (`settlement_identity.residual_interface_settlement_total`)
       sum_b w_b sum_s omega_s sum_t pi d == the DSO covariance (`settlement_identity.dso_deviation_part`)
     and per scenario against each block's `settlement_at_scenario_price` (both sides), which fixes signs, units, prices.
  5. WORKBOOK CROSS-CHECK (the 3 x 3 pair and the continuation): the terminal workbook's 'Interface PF' sheet (written
     by `_write_interface_results_to_excel` from the same production results path) against the JSON, per row.
  6. ACCEPTANCE-TEST INPUTS: the per-cycle record's `max_abs_d_p_mw` at the terminal cycle vs the capture, and the
     terminal = certified cycle facts from `evaluation_record.json`.

Guards: SolveProfileGuard(permitted=()) installed before any repository import and verified at exactly 0; `pickle.load`
and `pickle.loads` are blocked for the whole run (counted, verified 0).

Output: NEW directory data/SRP1/Results/P515S56/interface_consistency/ (refuses if it exists, or if another copy is
running). Launch (attached, both streams captured into a sibling log):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_interface_consistency.py \\
        > data/SRP1/Results/P515S56/interface_consistency_launch.log 2>&1
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 audit D interface consistency (never solves)').install()

import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('audit D: pickle.load is blocked (no unpickling in this stage)')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('audit D: pickle.loads is blocked (no unpickling in this stage)')


pickle.load = _blocked_load
pickle.loads = _blocked_loads

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'interface_consistency')
if '--dry-run-out' in sys.argv:
    OUT_REL = os.path.abspath(sys.argv[sys.argv.index('--dry-run-out') + 1])
    if OUT_REL.startswith(REPO + os.sep):
        raise SystemExit('--dry-run-out must be outside the repository')
OUT_DIR = os.path.join(REPO, OUT_REL)
RESULTS_NAME = 'interface_consistency.json'
MANIFEST_NAME = 'manifest_sha256.json'
STARTED = time.time()
CHECKS = []

R = os.path.join('data', 'SRP1', 'Results')
# (label, eval dir, role, workbook cross-check)
EVALS = [
    ('3x3_pair_x0', os.path.join(R, 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals', 'f6e9cd53fdbb8ee8_x0'),
     'primary: W91 3 x 3 pair, x = 0 (the 48.3 MW record)', True),
    ('3x3_pair_n7_4h_e1', os.path.join(R, 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals',
                                       'c82522f470b35b58_n7_4h_e1'),
     'primary: W91 3 x 3 pair, node-7 plan (storage present)', True),
    ('3x3_x0_continuation_r2', os.path.join(R, 'P515S53', 'w98_continuation', 'campaign_s53_w98_x0_continuation_r2',
                                            'evals', '25b92ae0f1f2c02e_x0_cont'),
     'W98 x = 0 continuation r2 (3 x 3)', True),
    ('3x3_smoke_boff_x0', os.path.join(R, 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_smoke_boff', 'evals',
                                       'f6e9cd53fdbb8ee8_x0'), 'smoke (3 cycles, not certified)', False),
    ('3x3_smoke_bon_x0', os.path.join(R, 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_smoke_bon', 'evals',
                                      'f6e9cd53fdbb8ee8_x0'), 'smoke (3 cycles, not certified)', False),
    ('2x2_pilot_x0', os.path.join(R, 'P515S52', 'campaign_s52_pilot_nopersist', 'evals', '7d53b6f21b686a44_x0'),
     'S52 2 x 2 pilot, x = 0', False),
    ('2x2_pilot_n7_4h_e1', os.path.join(R, 'P515S52', 'campaign_s52_pilot_nopersist', 'evals',
                                        '711fce9aa74d6878_n7_4h_e1'), 'S52 2 x 2 pilot, node-7 plan', False),
    ('2x2_pilot_repro_x0', os.path.join(R, 'P515S52', 'campaign_s52_pilot_repro_nopersist', 'evals',
                                        '7d53b6f21b686a44_x0'), 'S52 2 x 2 pilot repro (2 cycles, not certified)', False),
    ('2x2_alpha_x0_a0p00', os.path.join(R, 'P515S53', 'alpha_row', 'campaign_s53_alpha_row_v25', 'evals',
                                        'b123cd978794d690_x0_a0p00'), 'alpha row 2 x 2, alpha 0', False),
    ('2x2_alpha_x0_a0p10', os.path.join(R, 'P515S53', 'alpha_row', 'campaign_s53_alpha_row_v25', 'evals',
                                        '7516903c91153a29_x0_a0p10'), 'alpha row 2 x 2, alpha 0.10', False),
    ('2x2_alpha_x0_a0p25', os.path.join(R, 'P515S53', 'alpha_row', 'campaign_s53_alpha_row_v25', 'evals',
                                        '62b46280a65f7744_x0_a0p25'), 'alpha row 2 x 2, alpha 0.25', False),
    ('2x2_alpha_x0_a0p50', os.path.join(R, 'P515S53', 'alpha_row', 'campaign_s53_alpha_row_v25', 'evals',
                                        '7d53b6f21b686a44_x0_a0p50'), 'alpha row 2 x 2, alpha 0.50', False),
    ('2x2_alpha_x0_a1p00', os.path.join(R, 'P515S53', 'alpha_row', 'campaign_s53_alpha_row_v25', 'evals',
                                        '1bb2d63a07273887_x0_a1p00'), 'alpha row 2 x 2, alpha 1.00', False),
    ('2x2_alpha_n7_a0p50', os.path.join(R, 'P515S53', 'alpha_row', 'campaign_s53_alpha_row_v25', 'evals',
                                        '711fce9aa74d6878_n7_4h_e1_a0p50'), 'alpha row 2 x 2, node-7 plan, 0.50',
     False),
    ('2x2_alpha_smoke_x0', os.path.join(R, 'P515S53', 'alpha_row', 'campaign_s53_alpha_row_smoke', 'evals',
                                        '7d53b6f21b686a44_x0_a0p50'), 'alpha row smoke (2 cycles)', False),
    ('2x2_alpha_smoke_r2_x0', os.path.join(R, 'P515S53', 'alpha_row', 'campaign_s53_alpha_row_smoke_r2', 'evals',
                                           '7d53b6f21b686a44_x0_a0p50'), 'alpha row smoke r2 (2 cycles)', False),
]

DN_CASE = {5: 'case33_1', 7: 'case33_2', 9: 'case33_3'}
TN_CASE = 'case9'
YEARS_3X3 = (2025, 2028, 2031, 2034, 2037)

# (file, line, substring that must appear on that line, what the report cites it for)
CODE_ANCHORS = [
    ('shared_resources_planning.py', 4267, "interface_pf_p = consensus_vars['pf']['dso']['current'][adn_node_id][year][day]['p'][p] / s_base",
     'TSO ADN load anchor pc = the DSO consensus value, no scenario index'),
    ('shared_resources_planning.py', 4274, 'fix_or_set(tso_model[year][day].pc[adn_load_idx, s_m, s_o, p], interface_pf_p)',
     'pc fixed, same value in every (s_m, s_o)'),
    ('shared_resources_planning.py', 4300, 'if (s_m, s_o) == sess_na_scenario(tso_model[year][day]):',
     'only the first scenario pair copy of interface_delta is freed'),
    ('shared_resources_planning.py', 4315, 'expected_interface_pf_p_def = pe.Constraint',
     'TSO expected interface P = sum_s pi_s pc_adn'),
    ('shared_resources_planning.py', 4333, 'premium_alpha=0.0,', 'TSO: no row 18 (pin only)'),
    ('shared_resources_planning.py', 4356, "consensus_vars['pf']['tso']['current'][adn_node_id][year][day]['p'][p] = interface_pf_p",
     'TSO consensus value = expected'),
    ('shared_resources_planning.py', 4412, 'expected_interface_pf_p_def = pe.Constraint',
     'DSO expected interface P = sum_s pi_s pg_adn'),
    ('shared_resources_planning.py', 4428, 'premium_alpha=premium_alpha,', 'DSO: row 18 at the run alpha'),
    ('shared_resources_planning.py', 5195, 'constraint_p_req = (model[year][day].expected_interface_pf_p[dn, p] - model[year][day].p_pf_req[dn, p])',
     'TSO augmented Lagrangian on the EXPECTED interface P'),
    ('shared_resources_planning.py', 5407, 'dso_model[year][day].e[ref_node_idx, s_m, s_o, p].fixed = False',
     'DN reference-bus voltage magnitude free per scenario'),
    ('shared_resources_planning.py', 5467, 'constraint_p_req = (dso_model[year][day].expected_interface_pf_p[p] - dso_model[year][day].p_pf_req[p])',
     'DSO augmented Lagrangian on the EXPECTED interface P'),
    ('shared_resources_planning.py', 6147, "model[year][day].p_pf_req[dn, p].set_value(pf_req['dso']['current']",
     'TSO target = DSO expected (consensus value)'),
    ('shared_resources_planning.py', 6476, "model[year][day].p_pf_req[p].set_value(pf_req['tso']['current']",
     'DSO target = TSO expected (consensus value)'),
    ('shared_resources_planning.py', 6881, "tso_v = consensus_vars['vmag']['tso']['current'][node_id][year][day][p]",
     'legacy primal residual reads the consensus (expected) values'),
    ('shared_resources_planning.py', 7250, "x_dso_pf = consensus_vars['pf']['dso']['current'][node_id][year][day][power_type][p]",
     'Boyd residual reads the consensus (expected) values'),
    ('shared_resources_planning.py', 3385, 'cycle_convergence = boyd_all_pass and local_solves_ok',
     'the production stopping test is the Boyd test'),
    ('shared_resources_planning.py', 8501, 'p_req = pe.value(tso_model[year][day].expected_interface_pf_p[dn, p]) * s_base',
     'consensus update: TSO expected'),
    ('shared_resources_planning.py', 8524, 'p_req = pe.value(dso_model[year][day].expected_interface_pf_p[p]) * s_base',
     'consensus update: DSO expected'),
    ('shared_resources_planning.py', 8550, "error_p_pf_req_tso = interface_vars['pf']['tso']['current']",
     'dual update on expected TSO - expected DSO'),
    ('shared_resources_planning.py', 918, 'def _get_local_interface_dispersion(model, network):',
     'DSO dispersion d = pg_adn - own expected (the 48.3 MW quantity)'),
    ('shared_resources_planning.py', 13458, 'def _check_interface_nodes_base_voltage_consistency(planning_problem):',
     'TN ADN bus and DN reference bus base kV checked equal (5 %)'),
    ('shared_resources_planning.py', 13478, 'adn_load.pd = np.zeros(adn_load.pd.shape)',
     'TN ADN load data zeroed; its pc is the interface anchor'),
    ('model_construction_helpers.py', 841, 'def sess_na_scenario(m):', 'the scenario-free alias pair'),
    ('model_construction_helpers.py', 1259, 'pc_adn += m.interface_delta_p[dn, s_m0, s_o0, p]',
     'TSO interface P = pc + scenario-free delta'),
    ('model_construction_helpers.py', 1297, 'return m.pg[ref_gen_idx, s_m, s_o, p] - shared_ess_p',
     'DSO interface P = reference generator - shared ESS (per scenario)'),
    ('model_construction_helpers.py', 1392, 'Pd += model.interface_delta_p[dn, s_m0, s_o0, p]',
     'TN node balance carries the scenario-free delta'),
    ('model_construction_helpers.py', 1419, 'Pd += model.shared_es_pnet[e, s_m0, s_o0, p]',
     'shared ESS in the node balance, scenario-free'),
    ('model_construction_helpers.py', 1729, 'def interface_energy_settlement(model, network):',
     'settlement at the scenario price, both sides'),
    ('model_construction_helpers.py', 1773, 'def interface_energy_settlement_contracted(model, network):',
     'contracted part (excluded from Q)'),
    ('model_construction_helpers.py', 1861, 'def row18_interface_deviation_p_rule(m, s_m, s_o, p):',
     'row 18 deviation definition'),
    ('model_construction_helpers.py', 1917, 'if n_scenarios == 1:', 'nothing wired at one scenario'),
    ('model_construction_helpers.py', 1940, 'model.scenario_voltage_pin_weight = pe.Param(initialize=PENALTY_SCENARIO_DEVIATION, mutable=True)',
     'voltage pin weight'),
    ('model_construction_helpers.py', 1979, 'charge += (probability * model.row18_alpha * model.row18_premium[p] * network.baseMVA',
     'row 18 charge'),
    ('model_construction_helpers.py', 2005, 'not (not network.is_transmission and network.generators[g].gen_type == GEN_REFERENCE)',
     'DN reference generator carries no generation cost (priced only by the settlement)'),
    ('model_construction_helpers.py', 2416, 'def dn_interface_expected_pf_p_def(m, p, network):', 'DSO expectation'),
    ('model_construction_helpers.py', 2473, 'def tn_interface_expected_pf_p_def(m, dn, p, network):', 'TSO expectation'),
    ('network.py', 83, 'def get_interface_branch_rating(self):', 'interface rating = DN branches at the reference bus'),
    ('network.py', 422, 'model.interface_delta_p = pe.Var(model.adn_nodes, model.scenarios_market, model.scenarios_operation',
     'interface_delta declared per scenario (copies other than the alias stay fixed at 0)'),
    ('network.py', 440, 'model.pc_adn = pe.Expression(', 'TSO interface expression'),
    ('network.py', 444, 'model.pg_adn = pe.Expression(', 'DSO interface expression'),
    ('network.py', 1900, 'def _process_results_interface(network, model):', 'per-scenario interface V/P/Q results'),
    ('definitions.py', 75, 'PENALTY_SCENARIO_DEVIATION = 9e4', 'pin weight value'),
    ('p515_s44_campaign_harness.py', 3803, 'def _interface_profiles(model, network, kind):',
     'the capture of the per-scenario profiles used here'),
]
CITED_MODULES = sorted({a[0] for a in CODE_ANCHORS} | {'settling_criterion_v6.py', 'settling_criterion_v2.py',
                                                       'settling_criterion_v5.py', 'p513_solve_profile_guard.py',
                                                       'gate_result_io.py', SCRIPT})


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _ck(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:400]}' if detail is not None and not ok else ''))


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def _input_record(path):
    tracked = bool(_git(['ls-files', '--', path]).strip())
    clean = not _git(['status', '--porcelain', '--', path]).strip() if tracked else None
    return {'path': path, 'sha256': _sha(path), 'tracked': tracked, 'clean': clean}


def _other_copy_running():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me = os.getpid()
    found = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(os.path.basename(t) == SCRIPT for t in tok[2:4]):
            found.append(ln.strip())
    return found


# ======================================================================================================================
#  1. code anchors
# ======================================================================================================================
def code_anchors():
    out = []
    cache = {}
    for fname, line, text, claim in CODE_ANCHORS:
        if fname not in cache:
            with open(fname) as handle:
                cache[fname] = handle.read().splitlines()
        actual = cache[fname][line - 1] if line - 1 < len(cache[fname]) else None
        ok = actual is not None and text in actual
        out.append({'file': fname, 'line': line, 'expected_substring': text, 'claim': claim, 'ok': ok,
                    'actual_line': actual})
    _ck(f'code anchors: all {len(out)} cited lines hold the quoted text', all(a['ok'] for a in out),
        [a for a in out if not a['ok']])
    return out


# ======================================================================================================================
#  2. network facts (raw case files)
# ======================================================================================================================
def network_facts():
    facts = {}
    for year in YEARS_3X3:
        tn_path = os.path.join('data', 'SRP1', TN_CASE, f'{TN_CASE}_{year}.json')
        tn = json.load(open(tn_path))
        tn_kv = {n['bus_i']: n['baseKV'] for n in tn['nodes']}
        yf = {'tn_case': tn_path, 'tn_sha256': _sha(tn_path), 'tn_baseMVA': tn['baseMVA'],
              'tn_loads': tn.get('loads'), 'dn': {}}
        for node, case in DN_CASE.items():
            p = os.path.join('data', 'SRP1', case, f'{case}_{year}.json')
            d = json.load(open(p))
            ref = [n for n in d['nodes'] if int(n['type']) == 3]
            ref_id = ref[0]['bus_i']
            kv = {n['bus_i']: n['baseKV'] for n in d['nodes']}
            at_ref = []
            for kind in ('lines', 'transformers'):
                for b in d.get(kind, []):
                    if b.get('status', 1) and (b['fbus'] == ref_id or b['tbus'] == ref_id):
                        other = b['tbus'] if b['fbus'] == ref_id else b['fbus']
                        at_ref.append({'kind': kind, 'fbus': b['fbus'], 'tbus': b['tbus'],
                                       'rating_mva': b.get('rating', b.get('rate')), 'ratio': b.get('ratio'),
                                       'baseKV_from_to': [kv[b['fbus']], kv[b['tbus']]], 'other_bus': other})
            refgen = [g for g in d['generators'] if g['type'] == 'REF']
            yf['dn'][str(node)] = {'case': p, 'sha256': _sha(p), 'baseMVA': d['baseMVA'], 'ref_bus': ref_id,
                                   'ref_bus_baseKV': ref[0]['baseKV'], 'tn_bus_baseKV': tn_kv[node],
                                   'branches_at_ref_bus': at_ref, 'ref_generator': refgen}
        facts[str(year)] = yf
    ok_base = all(f['tn_baseMVA'] == 100.0 and all(v['baseMVA'] == 100.0 for v in f['dn'].values())
                  for f in facts.values())
    ok_kv = all(v['ref_bus_baseKV'] == v['tn_bus_baseKV'] for f in facts.values() for v in f['dn'].values())
    ok_tr = all(len(v['branches_at_ref_bus']) == 1 and v['branches_at_ref_bus'][0]['kind'] == 'transformers'
                for f in facts.values() for v in f['dn'].values())
    _ck('network facts: TN and every DN baseMVA = 100 on the five 3 x 3 years', ok_base)
    _ck('network facts: DN reference bus base kV == TN ADN bus base kV (every DN, every year)', ok_kv)
    _ck('network facts: exactly one branch at each DN reference bus, a transformer (inside the DN model)', ok_tr)
    return facts


# ======================================================================================================================
#  3/4. paired discrepancies and energy not supplied
# ======================================================================================================================
def _new_acc():
    return {'max_abs': 0.0, 'argmax': None, 'sum_sq': 0.0, 'sum_sq_w': 0.0, 'n_blocks': 0, 'w_sum': 0.0}


def _acc_add(acc, x, where, omega, w):
    if abs(x) > acc['max_abs']:
        acc['max_abs'] = abs(x)
        acc['argmax'] = {**where, 'value': x}
    acc['sum_sq'] += omega * x * x
    acc['sum_sq_w'] += w * omega * x * x


def _acc_finish(acc, n_blocks, w_sum):
    return {'max_abs': acc['max_abs'], 'argmax': acc['argmax'],
            'rms': math.sqrt(acc['sum_sq'] / (n_blocks * 24)) if n_blocks else None,
            'rms_blockweighted': math.sqrt(acc['sum_sq_w'] / (w_sum * 24)) if w_sum else None}


QTY = ('dP_mw', 'dQ_mvar', 'dV_pu', 'c_p_mw', 'd_p_mw', 'c_q_mvar', 'd_q_mvar', 'c_v_pu', 'dP_ess_mw', 'dQ_ess_mvar',
       'dP_total_mw')


def analyse(label, m, rec):
    blocks = m['blocks']
    tso_keys = sorted(k for k in blocks if k.startswith('TSO|'))
    per_dn = {}
    pooled = {q: _new_acc() for q in QTY}
    pooled_nb, pooled_w = 0, 0.0
    spreads = {'tn_p_mw': 0.0, 'tn_q_mvar': 0.0, 'tn_v_pu': 0.0, 'dn_p_mw': 0.0, 'dn_q_mvar': 0.0, 'dn_v_pu': 0.0}
    energy = {}
    worst = {'decomposition_identity_mw': 0.0, 'tn_scenario_vs_committed_p_mw': 0.0,
             'dso_settlement_rel': 0.0, 'tso_settlement_rel': 0.0, 'dispersion_d_vs_capture_mw': 0.0,
             'prob_tso_vs_dso': 0.0}
    priced_u_weighted = 0.0
    priced_d_weighted = 0.0
    nodes = sorted({k.split('|')[1] for k in blocks if k.startswith('DSO|')}, key=int)
    tso_settle_rebuilt = {}
    for node in nodes:
        accs = {q: _new_acc() for q in QTY}
        nb, wsum = 0, 0.0
        e = {'E_plus_mwh_sum_over_blocks': 0.0, 'E_minus_mwh_sum_over_blocks': 0.0,
             'E_plus_mwh_weighted': 0.0, 'E_minus_mwh_weighted': 0.0,
             'E_dn_import_mwh_sum_over_blocks': 0.0, 'E_dn_import_mwh_weighted': 0.0,
             'C_plus_eur_sum_over_blocks': 0.0, 'C_minus_eur_sum_over_blocks': 0.0,
             'C_plus_eur_weighted': 0.0, 'C_minus_eur_weighted': 0.0, 'C_net_eur_weighted': 0.0,
             'E_plus_total_incl_ess_mwh_weighted': 0.0, 'E_minus_total_incl_ess_mwh_weighted': 0.0,
             'E_abs_d_mwh_weighted': 0.0, 'per_block_max': None}
        per_block_rows = []
        for tkey in tso_keys:
            _, year, day = tkey.split('|')
            dkey = f'DSO|{node}|{year}|{day}'
            tb, db = blocks[tkey], blocks[dkey]
            w = float(db['admm_block_weight'])
            worst['prob_tso_vs_dso'] = max(worst['prob_tso_vs_dso'], abs(w - float(tb['admm_block_weight'])))
            tprof = tb['interface_profiles']['per_scenario'][node]
            tcom = tb['interface_profiles']['committed'][node]
            dprof = db['interface_profiles']['per_scenario']
            dcom = db['interface_profiles']['committed']
            pi = db['dispersion']['pi_by_market_by_hour']          # [t][s_m]
            dper = db['dispersion']['per_scenario_d']
            tess = tb['shared_ess_schedule'].get(node)
            dess_keys = list(db['shared_ess_schedule'].keys())
            dess = db['shared_ess_schedule'][dess_keys[0]] if len(dess_keys) == 1 else None
            if tess is None or dess is None:
                raise RuntimeError(f'{label} {dkey}: shared ESS schedule not found on both sides '
                                   f'(TSO keys {list(tb["shared_ess_schedule"])}, DSO keys {dess_keys})')
            scen = sorted(dprof.keys())
            if sorted(tprof.keys()) != scen:
                raise RuntimeError(f'{label} {dkey}: scenario key sets differ TSO {sorted(tprof)} vs DSO {scen}')
            b_eplus = b_eminus = 0.0
            b_rms_acc = 0.0
            for s in scen:
                s_m = int(s.split('_')[0])
                omega = float(db['per_scenario_costs'][s]['probability'])
                worst['prob_tso_vs_dso'] = max(worst['prob_tso_vs_dso'],
                                               abs(omega - float(tb['per_scenario_costs'][s]['probability'])))
                tp, tq, tv = tprof[s]['p'], tprof[s]['q'], tprof[s]['v']
                dp, dq, dv = dprof[s]['p'], dprof[s]['q'], dprof[s]['v']
                settle_dso = 0.0
                for t in range(24):
                    price = float(pi[t][s_m])
                    dP = tp[t] - dp[t]
                    dQ = tq[t] - dq[t]
                    dV = tv[t] - dv[t]
                    c_p = tcom['p_mw'][t] - dcom['p_mw'][t]
                    d_p = dp[t] - dcom['p_mw'][t]
                    c_q = tcom['q_mvar'][t] - dcom['q_mvar'][t]
                    d_q = dq[t] - dcom['q_mvar'][t]
                    c_v = tcom['v_pu'][t] - dcom['v_pu'][t]
                    dP_ess = tess['pnet_mw'][t] - dess['pnet_mw'][t]
                    dQ_ess = tess['qnet_mvar'][t] - dess['qnet_mvar'][t]
                    vals = {'dP_mw': dP, 'dQ_mvar': dQ, 'dV_pu': dV, 'c_p_mw': c_p, 'd_p_mw': d_p, 'c_q_mvar': c_q,
                            'd_q_mvar': d_q, 'c_v_pu': c_v, 'dP_ess_mw': dP_ess, 'dQ_ess_mvar': dQ_ess,
                            'dP_total_mw': dP + dP_ess}
                    where = {'node': int(node), 'year': year, 'day': day, 'scenario': s, 'hour': t}
                    for q, x in vals.items():
                        _acc_add(accs[q], x, where, omega, w)
                        _acc_add(pooled[q], x, where, omega, w)
                    b_rms_acc += omega * dP * dP
                    worst['decomposition_identity_mw'] = max(worst['decomposition_identity_mw'],
                                                             abs(dP - (c_p - d_p) - (tp[t] - tcom['p_mw'][t])))
                    worst['tn_scenario_vs_committed_p_mw'] = max(worst['tn_scenario_vs_committed_p_mw'],
                                                                 abs(tp[t] - tcom['p_mw'][t]))
                    worst['dispersion_d_vs_capture_mw'] = max(worst['dispersion_d_vs_capture_mw'],
                                                              abs(d_p - float(dper[s]['d_p_mw'][t])))
                    u = -dP
                    u_tot = -(dP + dP_ess)
                    up, um = max(u, 0.0), max(-u, 0.0)
                    b_eplus += omega * up
                    b_eminus += omega * um
                    e['E_plus_mwh_sum_over_blocks'] += omega * up
                    e['E_minus_mwh_sum_over_blocks'] += omega * um
                    e['E_plus_mwh_weighted'] += w * omega * up
                    e['E_minus_mwh_weighted'] += w * omega * um
                    e['E_plus_total_incl_ess_mwh_weighted'] += w * omega * max(u_tot, 0.0)
                    e['E_minus_total_incl_ess_mwh_weighted'] += w * omega * max(-u_tot, 0.0)
                    e['E_dn_import_mwh_sum_over_blocks'] += omega * dp[t]
                    e['E_dn_import_mwh_weighted'] += w * omega * dp[t]
                    e['E_abs_d_mwh_weighted'] += w * omega * abs(d_p)
                    e['C_plus_eur_sum_over_blocks'] += omega * price * up
                    e['C_minus_eur_sum_over_blocks'] += omega * price * um
                    e['C_plus_eur_weighted'] += w * omega * price * up
                    e['C_minus_eur_weighted'] += w * omega * price * um
                    e['C_net_eur_weighted'] += w * omega * price * u
                    priced_u_weighted += w * omega * price * u
                    priced_d_weighted += w * omega * price * d_p
                    settle_dso += price * dp[t]
                    key_t = (year, day, s)
                    tso_settle_rebuilt[key_t] = tso_settle_rebuilt.get(key_t, 0.0) - price * tp[t]
                ref = float(db['per_scenario_costs'][s]['settlement_at_scenario_price'])
                worst['dso_settlement_rel'] = max(worst['dso_settlement_rel'],
                                                  abs(settle_dso - ref) / max(abs(ref), 1.0))
            # spreads (max_s - min_s per hour)
            for t in range(24):
                for side, prof, pfx in (('tn', tprof, 'tn'), ('dn', dprof, 'dn')):
                    for comp, nm in (('p', 'p_mw'), ('q', 'q_mvar'), ('v', 'v_pu')):
                        vals_t = [prof[s][comp][t] for s in scen]
                        spreads[f'{pfx}_{nm}'] = max(spreads[f'{pfx}_{nm}'], max(vals_t) - min(vals_t))
            per_block_rows.append({'year': year, 'day': day, 'weight': w,
                                   'rms_dP_mw': math.sqrt(b_rms_acc / 24), 'E_plus_mwh': b_eplus,
                                   'E_minus_mwh': b_eminus})
            nb += 1
            wsum += w
        pooled_nb += nb
        pooled_w += wsum
        e['per_block_max'] = max(per_block_rows, key=lambda r: r['rms_dP_mw'])
        e['E_plus_share_of_dn_import_weighted'] = (e['E_plus_mwh_weighted'] / e['E_dn_import_mwh_weighted']
                                                   if e['E_dn_import_mwh_weighted'] else None)
        per_dn[node] = {q: _acc_finish(accs[q], nb, wsum) for q in QTY}
        per_dn[node]['n_blocks'] = nb
        per_dn[node]['per_block'] = per_block_rows
        energy[node] = e
    overall = {q: _acc_finish(pooled[q], pooled_nb, pooled_w) for q in QTY}
    # TSO settlement per scenario (summed over DNs) against the capture
    for tkey in tso_keys:
        _, year, day = tkey.split('|')
        for s, sc in blocks[tkey]['per_scenario_costs'].items():
            ref = float(sc['settlement_at_scenario_price'])
            got = tso_settle_rebuilt[(year, day, s)]
            worst['tso_settlement_rel'] = max(worst['tso_settlement_rel'], abs(got - ref) / max(abs(ref), 1.0))
    si = m['summary']['settlement_identity']
    dso_dev_total = sum(float(v) for v in si['dso_deviation_part'].values())
    recon = {
        'priced_u_weighted_recomputed': priced_u_weighted,
        'T_TSO_plus_T_DSO_production': si['residual_interface_settlement_total'],
        'rel_diff_u': abs(priced_u_weighted - si['residual_interface_settlement_total'])
        / max(abs(si['residual_interface_settlement_total']), 1.0),
        'priced_d_weighted_recomputed': priced_d_weighted,
        'dso_covariance_production': dso_dev_total,
        'rel_diff_d': abs(priced_d_weighted - dso_dev_total) / max(abs(dso_dev_total), 1.0),
        'contracted_leftover_production': si['contracted_total'],
        'row18_charge_weighted_production': m['summary']['all_dso']['row18_charge_weighted'],
        'convention': ('u = P_DSO - P_TSO (MW, DN-import positive); T_TSO + T_DSO = sum_b w_b sum_s omega_s sum_t '
                       'pi[s_m,t] u[s,t] (model_construction_helpers.interface_energy_settlement, TSO -pi*pc_adn, '
                       'DSO +pi*pg_adn); the DSO covariance = sum_b w_b sum_s omega_s sum_t pi d (sum_s omega_s d = 0)')}
    tests = {
        'decomposition dP == c - d + (P_TSO[s] - Pbar_TSO) holds (max abs, MW)': worst['decomposition_identity_mw'] <= 1e-9,
        'TN interface P scenario-common: max |P_TSO[s] - Pbar_TSO| <= 1e-9 MW': worst['tn_scenario_vs_committed_p_mw'] <= 1e-9,
        'd recomputed == captured per_scenario_d (<= 1e-9 MW)': worst['dispersion_d_vs_capture_mw'] <= 1e-9,
        'DSO per-scenario settlement rebuilt == capture (rel <= 1e-9)': worst['dso_settlement_rel'] <= 1e-9,
        'TSO per-scenario settlement rebuilt with DSO prices == capture (rel <= 1e-9)': worst['tso_settlement_rel'] <= 1e-9,
        'T_TSO + T_DSO reconciled (rel <= 1e-9)': recon['rel_diff_u'] <= 1e-9,
        'DSO covariance reconciled (rel <= 1e-9)': recon['rel_diff_d'] <= 1e-9,
        'TSO and DSO block weights and scenario probabilities equal': worst['prob_tso_vs_dso'] <= 1e-12,
    }
    for name, ok in tests.items():
        _ck(f'{label}: {name}', ok, worst)
    # reproduce the W91 dispersion summary
    summ = m['summary']['all_dso']
    repro = {'max_abs_d_mw_recomputed': overall['d_p_mw']['max_abs'],
             'max_abs_d_mw_capture_summary': summ['max_abs_mw_over_all_dso_blocks'],
             'argmax': overall['d_p_mw']['argmax'],
             'per_dn_capture': {n: v['max_abs_mw'] for n, v in m['summary']['per_dso'].items()},
             'per_dn_recomputed': {n: per_dn[n]['d_p_mw']['max_abs'] for n in per_dn}}
    _ck(f'{label}: max |d| recomputed == capture summary max_abs_mw_over_all_dso_blocks',
        abs(repro['max_abs_d_mw_recomputed'] - repro['max_abs_d_mw_capture_summary']) <= 1e-9, repro)
    # per-cycle record at the terminal cycle
    pc_last = None
    pcr = os.path.join(rec['_eval_dir'], 'per_cycle_record.jsonl')
    if os.path.isfile(pcr):
        with open(pcr) as handle:
            lines = [json.loads(ln) for ln in handle if ln.strip()]
        if lines:
            pc_last = {k: lines[-1].get(k) for k in ('cycle', 'max_abs_d_p_mw', 'cycle_convergence', 'boyd_all_pass',
                                                     'consecutive_converged_cycles', 'boyd_pf_primal_ratio',
                                                     'boyd_v_primal_ratio', 'E_abs_d_p_mwh_weighted')}
    return {'per_dn': per_dn, 'overall': overall, 'spreads_max_over_dn_block_hour': spreads, 'energy': energy,
            'reconciliation': recon, 'worst_identities': worst, 'tests': tests, 'w91_dispersion_reproduction': repro,
            'per_cycle_record_terminal': pc_last}


def workbook_crosscheck(eval_dir, m):
    import openpyxl
    path = os.path.join(eval_dir, 'results', 'SRP1_distributed_terminal.xlsx')
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb['Interface PF']
    blocks = m['blocks']
    qmap = {'Vmag, [p.u.]': 'v', 'P, [MW]': 'p', 'Q, [MVAr]': 'q'}
    n, worst = 0, {'p': 0.0, 'q': 0.0, 'v': 0.0}
    skipped = 0
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            continue
        node, op, year, day, qty, s_m, s_o = row[:7]
        if not isinstance(s_m, int) or qty not in qmap:
            skipped += 1
            continue
        s = f'{s_m}_{s_o}'
        comp = qmap[qty]
        if op == 'TSO':
            ref = blocks[f'TSO|{year}|{day}']['interface_profiles']['per_scenario'][str(node)][s][comp]
        else:
            ref = blocks[f'DSO|{node}|{year}|{day}']['interface_profiles']['per_scenario'][s][comp]
        vals = row[7:31]
        for t in range(24):
            worst[comp] = max(worst[comp], abs(float(vals[t]) - float(ref[t])))
        n += 1
    wb.close()
    return {'workbook': path, 'rows_compared': n, 'rows_skipped_expected_or_other': skipped,
            'max_abs_diff_vs_json': worst}


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    started = datetime.now(timezone.utc).isoformat()
    others = _other_copy_running()
    if others:
        _log(f'REFUSE: another copy of {SCRIPT} is running: {others}')
        return 2
    if os.path.exists(OUT_DIR):
        _log(f'REFUSE: {OUT_REL} exists')
        return 2
    os.makedirs(OUT_DIR)

    anchors = code_anchors()
    modules = {p: _input_record(p) for p in CITED_MODULES}
    facts = network_facts()

    inputs, results = {}, {}
    for label, edir, role, xcheck in EVALS:
        _log(f'== {label} ({edir})')
        mpath = os.path.join(edir, 'multiscenario_terminal.json')
        rpath = os.path.join(edir, 'evaluation_record.json')
        cman = os.path.join(edir, 'child_manifest_sha256.json')
        man = json.load(open(cman))
        rec_in = {}
        for fname in ('multiscenario_terminal.json', 'evaluation_record.json', 'per_cycle_record.jsonl'):
            p = os.path.join(edir, fname)
            r = _input_record(p)
            r['pinned_sha256_child_manifest'] = man.get(p)
            r['pinned_ok'] = man.get(p) == r['sha256']
            rec_in[fname] = r
            _ck(f'{label}: {fname} sha256 == child manifest pin', r['pinned_ok'], r)
        rec_in['child_manifest_sha256.json'] = _input_record(cman)
        inputs[label] = rec_in
        m = json.load(open(mpath))
        rec = json.load(open(rpath))
        rec['_eval_dir'] = edir
        n_scen = m['blocks']['TSO|2025|Spring']['n_scenarios']
        a = analyse(label, m, rec)
        cert = rec.get('certification') or {}
        a['run'] = {'role': role, 'eval_dir': edir, 'candidate_label': rec.get('candidate_label'),
                    'candidate_canonical': rec.get('candidate_canonical'), 'candidate_key': rec.get('candidate_key'),
                    'eval_key': rec.get('eval_key'), 'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
                    'certification_cycle': cert.get('certification_cycle'), 'n_scenarios': n_scen,
                    'alpha_in_force': m['summary'].get('alpha_in_force'),
                    'capture_point': 'terminal (multiscenario_terminal_capture, off the final models)',
                    'terminal_equals_certified': (rec.get('status') == 'certified'
                                                  and cert.get('certification_cycle') == rec.get('cycles_run')),
                    'gross_operational_cost_terminal': (m['summary'].get('recourse_components') or {}).get(
                        'gross_operational_cost')}
        if xcheck:
            _log(f'   workbook cross-check {label} ...')
            xc = workbook_crosscheck(edir, m)
            xc['workbook_sha256'] = _sha(xc['workbook'])
            xc['pinned_ok'] = man.get(xc['workbook']) == xc['workbook_sha256']
            _ck(f'{label}: terminal workbook sha256 == child manifest pin', xc['pinned_ok'])
            _ck(f'{label}: workbook Interface PF == JSON per-scenario profiles (max abs <= 1e-9)',
                xc['rows_compared'] > 0 and max(xc['max_abs_diff_vs_json'].values()) <= 1e-9, xc)
            a['workbook_crosscheck'] = xc
        results[label] = a

    guard_fail = GUARD.verify(0)
    _ck('own SolveProfileGuard(permitted=()) verify(0) == []', guard_fail == [], guard_fail)
    _ck('pickle.load / pickle.loads never reached', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    payload = {
        'schema': 'p515_s56_interface_consistency_v1', 'script': SCRIPT, 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - STARTED,
        'interpreter': sys.executable, 'git_head': _git(['rev-parse', 'HEAD']).strip(),
        'solve_claim': ('ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) armed for the whole run, '
                        'verify(0)); NO UNPICKLING (pickle.load/loads blocked, counted 0)'),
        'formula': {
            'pairing': ("scenario key s = '<s_m>_<s_o>' on both sides (the formulation's own index; the TSO's s_o is "
                        "the TN operation scenario, the DSO's s_o the DN's -- no constraint ties them; the TN interface "
                        "P/Q is scenario-common, so dP/dQ do not depend on the pairing)"),
            'dP': 'P_TSO[s,t] - P_DSO[s,t] (MW; P_TSO = pc_adn*baseMVA, P_DSO = pg_adn*baseMVA; DN-import positive)',
            'dV': 'V_TSO[s,t] - V_DSO[s,t] (p.u.; TN ADN bus vs DN reference bus; equal base kV, checked)',
            'c': 'Pbar_TSO[t] - Pbar_DSO[t] (expected-value consensus residual)',
            'd': 'P_DSO[s,t] - Pbar_DSO[t] (DSO deviation from its own commitment; production _get_local_interface_dispersion)',
            'dP_total': 'dP + (pnet_TSO-copy - pnet_DSO-copy) (shared ESS at the same bus, both scenario-free)',
            'rms': 'sqrt(sum_b sum_s omega_s sum_t x^2 / (N_b*24)), representative days equal',
            'rms_blockweighted': 'sqrt(sum_b w_b sum_s omega_s sum_t x^2 / (sum_b w_b*24)), w_b = admm_block_weight',
            'u': '-dP = P_DSO - P_TSO; E_plus = sum_s omega_s sum_t max(u,0) MWh per representative day; '
                 '*_weighted = sum_b w_b E_b (w_b = years x days x 1.02^-(y-2025): discounted horizon MWh)',
            'C': 'sum omega_s pi[s_m,t] max(+-u,0) (EUR at the scenario price, DSO block price array)',
            'spreads': 'max over (DN, block, hour) of max_s x - min_s x, per side'},
        'code_anchors': anchors, 'modules': modules, 'network_facts': facts,
        'results': results, 'inputs': inputs,
        'checks': CHECKS, 'failed_checks': [c['check'] for c in CHECKS if not c['ok']],
        'all_checks_ok': all(c['ok'] for c in CHECKS),
        'guards': {'own': {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                           'verify_0': guard_fail}, 'pickle_counts': dict(PICKLE_COUNTS)},
    }
    out_path = os.path.join(OUT_DIR, RESULTS_NAME)
    with open(out_path, 'x') as handle:
        GRIO.dump(payload, handle, indent=1, default=GRIO.json_default_item)
    manifest = {'outputs': {RESULTS_NAME: _sha(out_path)},
                'inputs': {**{f'{lab}/{k}': v for lab, d in inputs.items() for k, v in d.items()},
                           **{f'module/{k}': v for k, v in modules.items()}},
                'note': ('the launch log (sibling interface_consistency_launch.log) is written while the script runs; '
                         'its sha256 is recorded in the Worker report')}
    with open(os.path.join(OUT_DIR, MANIFEST_NAME), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1)
    _log(f'all_checks_ok = {payload["all_checks_ok"]}; failed = {payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
