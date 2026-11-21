"""P5.15 S56 -- reuse audit J (ZERO SOLVES, read-only): the 2x flexibility-price multiplier, the derivative-free search
moves available for a staged (node, year) domain, and search-effort comparability.

Questions (Planner task, S56, 2026-11-21):
  Q1  the flexibility-price multiplier the published 2x cells used (harness option `flex_price_multiplier`): where it is
      applied, exactly which arrays / objective terms it scales and which it leaves alone (TSO, DSO, ESSO, row-18
      premium, flexibility penalties and slacks, the uncoordinated benchmark's 1 EUR/MWh tie-breaker), and whether it
      composes with the 3 x 3 multi-scenario read and with the EV injection hook (`p515_s56_ev_injection.py`).
  Q2  the search moves of variant A (`p515_s47_phase_b_record.run_mads`), variant B (`p515_s53_f2_certificate`), the E1
      "B+" launcher (`p515_s55_e1_launcher.py`) and the R3 neighbourhood of `p516_lattice.py`, recorded by exact-text
      anchors (file, line) so the move table in the Worker report is checkable; plus three executable facts (pure
      calls): the 7-variable hard-wiring of PB, the Halton prime table length, the harness's single-year candidate.
  Q3  evaluations per poll / per search from the committed search records (s47, s51, s53; W173), and the size of a
      unit neighbourhood on the 18-variable staged domain z in Z^{2 |N| |Y|} (3 nodes x 2025/2030/2035) around
      x = 0, the reference unit (node 7 0.25 MVA / 1 MWh, 2025) and the 2030 plan (node 5 0.25/0.5 + node 7 1/3.5,
      2030), after admissibility decided by PRODUCTION's first-stage check
      (`shared_resources_planning._check_candidate_first_stage_feasibility`, total capacities rebuilt by production's
      `_rebuild_candidate_total_capacities`) on a real `SharedEnergyStorageData` read by the production readers.
      Three neighbourhoods: (i) the 2n coordinate poll +-e_i (36 raw points), (ii) the full l_inf-1 box (3^18 - 1 raw
      offsets; enumerated as the product of per-cohort admissible offsets, every product point decided by the
      production check), (iii) the R3 neighbourhood of `p516_lattice.GeneralLattice.neighbourhood` (every listed
      point re-decided by the production check). Informational: two-cohort "redistribution" moves (one R3 decrease
      on one cohort + one R3 increase on another; NOT an existing neighbourhood) and the size of the whole staged
      admissible domain (enumerated, decided by the production check).

GUARDS: SolveProfileGuard(permitted=()) installed before any repository import and verified at exactly 0; the imported
PB module's own parent guard verified at 0; pickle.load / pickle.loads blocked and counted (0). No model is built.

Writes ONLY to a NEW directory (default data/SRP1/Results/P515S56/reuse_audit_J/): reuse_audit_J.json and
manifest_sha256.json. Refuses if the directory holds anything other than launch.log, or if another copy is running.
Launch (repo root, canonical interpreter, attached, both streams captured):
    mkdir data/SRP1/Results/P515S56/reuse_audit_J && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_reuse_audit_J.py \\
        > data/SRP1/Results/P515S56/reuse_audit_J/launch.log 2>&1
Exit: 0 written and every check holds; 3 written with failed checks listed; 2 refused before computing.
"""
import os
import pickle
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 reuse audit J (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('reuse audit J: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('reuse audit J: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import types  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from itertools import product  # noqa: E402

import numpy as np  # noqa: E402

import definitions as DEF  # noqa: E402
import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402  (stdlib-only at import)
import p515_s47_phase_b_record as PB  # noqa: E402  (installs its own parent guard, permitted=())
import p516_lattice as GL  # noqa: E402
import shared_energy_storage_data as ESSD  # noqa: E402
import shared_resources_planning as SRP  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'reuse_audit_J')
OUT_JSON = 'reuse_audit_J.json'
OUT_MANIFEST = 'manifest_sha256.json'
ALLOWED_PREEXISTING = {'launch.log'}

CASE_SRP1 = os.path.join('data', 'SRP1', 'SRP1.json')
CASE_3X3 = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'SRP1__s56_srp1grid_3x3.json')
EV_PROFILES = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'SRP1__s56_srp1grid_3x3_ev_profiles.json')
DATA_DIR = os.path.join('data', 'SRP1')
SEARCH_RECORDS = {
    's47_phase_b (variant A, m = 1)': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b',
                                                   'campaign_results.json'),
    's51_f2_phase_b (variant A, m = 2)': os.path.join('data', 'SRP1', 'Results', 'P515S51', 'campaign_s51_f2_phase_b',
                                                      'campaign_results.json'),
    's53_f2_certificate_r1 (variant B, m = 2)': os.path.join('data', 'SRP1', 'Results', 'P515S53',
                                                             'campaign_s53_f2_certificate_r1', 'campaign_results.json'),
}
W173_JSON = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w173_search_counts', 'w173_search_counts.json')
G_CC4_JSON = os.path.join('data', 'SRP1', 'Results', 'P516CC', 'g_cc4', 'g_cc4_results.json')
PUBLISHED_2X_RECORD_GLOBS = {
    's51_f2_phase_b': 'data/SRP1/Results/P515S51/campaign_s51_f2_phase_b/evals/*_m2/evaluation_record.json',
    's55_e2_unit_2x_rho0p5': 'data/SRP1/Results/P515S55/e2_rho/campaign_s55_step7_e2_unit_2x_rho0p5/evals/*/evaluation_record.json',
}
GATE_ARTIFACTS = {
    'W33 flexibility-price multiplier zero-solve checks (K1-K6)': (
        'data/SRP1/Results/P515S49/flex_price_checks/flex_price_checks.json', 'all_ok'),
    'W33 flexibility-price multiplier 2-cycle gate': ('data/SRP1/Results/P515S49/flex_price_gate/gate.json', 'gate_pass'),
    'PB launcher checks (synthetic cache, faked evaluate)': (
        'data/SRP1/Results/P515S47/phase_b_record_checks_r2/phase_b_record_checks.json', None),
    'G-CC-4 general lattice gate': (G_CC4_JSON, 'all_pass'),
    'W183b E1 dry run': ('data/SRP1/Results/P515S55/w183b_e1_dryrun/w183b_e1_dryrun.json', None),
    'W183g(b) E1 v2 dry run': ('data/SRP1/Results/P515S55/w183g_b_e1_v2_dryrun/w183g_b_e1_v2_dryrun.json', None),
    'W183i(c) E1 v3 resume gate': ('data/SRP1/Results/P515S55/w183i_c_e1_v3_resume/w183i_c_e1_v3_resume_gate.json',
                                   None),
    'W173 search counts': (W173_JSON, None),
}

YEARS = (2025, 2030, 2035)
NODES = (5, 7, 9)
INCUMBENTS = {  # (node, year) -> (zP, zE); P = 0.25 zP MVA, E = 0.5 zE MWh
    'x0': {},
    'reference_unit_n7_0.25_1_y2025': {(7, 2025): (1, 2)},
    'plan_2030_n5_0.25_0.5__n7_1_3.5': {(5, 2030): (1, 1), (7, 2030): (4, 7)},
}
PLAN_2030_I_EXPECTED = 992268.11   # printed I(x) of the 2030 plan (W182g / S55 count), checked to the cent
M_VALUES_INFORMATIONAL = (1.5, 1.75, 3.0)

# ======================================================================================================================
#  exact-text anchors (file, text) -> line numbers; never typed by hand
# ======================================================================================================================
H_ = 'p515_s44_campaign_harness.py'
MCH = 'model_construction_helpers.py'
SRP_ = 'shared_resources_planning.py'
NET = 'network.py'
EV = 'p515_s56_ev_injection.py'
PBF = 'p515_s47_phase_b_record.py'
S53F = 'p515_s53_f2_certificate.py'
E1F = 'p515_s55_e1_launcher.py'
GLF = 'p516_lattice.py'
ANCHORS = {
    # ---- Q1: the multiplier ------------------------------------------------------------------------------------------
    'h_flex_vars': (H_, "FLEX_PRICE_VARS = ('flex_p_down', 'flex_q_down')"),
    'h_scaled_fn': (H_, 'def _scaled_cost_flex(array, m):'),
    'h_scaled_arith': (H_, 'return np.asarray(array, dtype=np.float64) * m'),
    'h_apply_fn': (H_, 'def apply_flex_price_multiplier(planning, m):'),
    'h_apply_tso_untouched_doc': (H_, "charge excludes the ADN-interface loads, its only loads on SRP1) and `planning.cost_flex` are left untouched;"),
    'h_apply_dso_loop_replace': (H_, 'net.cost_flex = scaled'),
    'h_apply_holder_rebound': (H_, 'dn.cost_flex = rebound'),
    'h_check_tso_unchanged': (H_, "'tso_arrays_unchanged': all(tso.network[y][d].cost_flex is tso_arrays[(y, d)]"),
    'h_check_planning_unchanged': (H_, "'planning_cost_flex_unchanged': all(planning.cost_flex[y][d] is a"),
    'h_check_energy_unchanged': (H_, "'dso_energy_prices_unchanged': all(planning.distribution_networks[n].network[y][d].cost_energy_p is a"),
    'h_precondition_min_cost': (H_, "precondition[str(node_id)] = {'obj_type_is_min_cost': params.obj_type == OBJ_MIN_COST,"),
    'h_hook_call_last': (H_, 'if flex_price_multiplier is not None:  # W33: last, so every check above ran on the unmodified planning'),
    'h_hook_apply': (H_, 'applied_fp = apply_flex_price_multiplier(planning, flex_price_multiplier)  # raises on any mismatch'),
    'h_key_payload': (H_, "payload['flex_price_multiplier'] = flex_m"),
    'h_key_rule': (H_, 'def flex_price_multiplier_in_key(value):'),
    'h_postrun_readback_fn': (H_, 'def flex_price_readback_run_models(dso_models, planning, original, m):'),
    'h_canonical_one_year': (H_, "return {'investment_year': int(investment_year),"),
    'mch_flex_cost_fn': (MCH, 'def flexibility_cost(model, network, s_m, s_o, params):'),
    'mch_flex_cost_cflex': (MCH, 'c_flex = network.cost_flex[s_m]'),
    'mch_flex_cost_skip_tso_adn': (MCH, 'if load_is_tso_adn_interface(network, load):'),
    'mch_flex_cost_term': (MCH, 'flex_cost += c_flex[p] * network.baseMVA * ('),
    'mch_flex_cost_vars': (MCH, 'model.flex_p_down[c, s_m, s_o, p] + model.flex_q_down[c, s_m, s_o, p]'),
    'mch_flex_cost_expression': (MCH, 'model.flex_cost_scenario = pe.Expression(model.scenarios_market, model.scenarios_operation, rule=partial(flex_cost_rule, network=network, params=params))'),
    'mch_obj_min_cost': (MCH, 'obj = model.total_gen_cost + model.total_flex_cost + model.total_load_curt_cost + model.total_gen_curt_penalty'),
    'mch_obj_congestion': (MCH, 'obj = model.total_gen_curt_penalty + model.total_load_curt_penalty + model.total_flex_penalty'),
    'mch_flex_usage_penalty': (MCH, 'penalty = model.penalty_flex_usage'),
    'mch_day_balance_slack_penalty': (MCH, 'total += base * PENALTY_FLEXIBILITY * (model.slack_flex_p_balance_up[c, s_m, s_o] + model.slack_flex_p_balance_down[c, s_m, s_o])'),
    'mch_adn_flex_cost_definitional': (MCH, 'def adn_interface_flexibility_cost(model, network, s_m, s_o, params):'),
    'mch_pibar_energy_price': (MCH, 'contribution = network.prob_market_scenarios[s_m] * network.cost_energy_p[s_m][p]'),
    'mch_row18_pibar': (MCH, 'pibar = expected_market_price(network, p)'),
    'mch_row18_charge': (MCH, 'charge += (probability * model.row18_alpha * model.row18_premium[p] * network.baseMVA'),
    'mch_settlement_energy_price': (MCH, 'c_p = network.cost_energy_p[s_m]'),
    'srp_cost_flex_generated': (SRP_, 'planning_problem.cost_flex[year][day] = np.array(flexibility_selected_profiles * flexibility_growth_cumul)'),
    'srp_cost_flex_dso_bind': (SRP_, 'distribution_network.network[year][day].cost_flex = planning_problem.cost_flex[year][day]'),
    'srp_cost_flex_tso_bind': (SRP_, 'transmission_network.network[year][day].cost_flex = planning_problem.cost_flex[year][day]'),
    'srp_cost_flex_report_sheet': (SRP_, 'cost_flexibility = planning_problem.cost_flex[year][day]'),
    'srp_checksum_flex': (SRP_, "('market', 'flexibility', year, day),"),
    'net_report_cost_flex': (NET, 'def _compute_cost_flexibility_per_scenario(network, model, params, s_m, s_o):'),
    'net_q_flex_constant_up': (NET, 'load.flexibility.reactive_power.upward = np.ones(pc_flex_up.shape) * EQUALITY_TOLERANCE'),
    'ub_cflex_expectation': ('uncoordinated_benchmark.py', 'c_flex = sum(dnet.prob_market_scenarios[s] * dnet.cost_flex[s][p]'),
    'w93_tie_breaker': ('p515_s53_w93_uncoordinated_benchmark.py', "'decision': {'passive_dso': 1.0, 'price_taker_dso': 0.0, 'tso': 0.0},"),
    'ub_tie_breaker_is_curtailment_penalty': ('uncoordinated_benchmark.py', 'block.penalty_gen_curtailment.set_value(float(curtailment_penalty))'),
    'ev_hook_point': (EV, 'self.original = SRP._compute_scenario_metadata'),
    'ev_inject_in_place': (EV, 'arr[...] = m'),
    'ev_assert_block_market_arrays': (EV, "for fld, tag in (('cost_energy_p', 'energy'), ('cost_flex', 'flex')):"),
    's49_k5_tso_esso_identical': ('p515_s49_flex_price_checks.py', '(b) m != 1: every TSO and ESSO block digest identical'),
    # ---- Q2: search code ---------------------------------------------------------------------------------------------
    'pb_n_vars': (PBF, 'N_VARS = 2 * len(ACTIVE_NODES) + 1  # 7: (zP, zE) per node + the common year index'),
    'pb_lattice_dim_check': (PBF, 'if len(self.nodes) * 2 + 1 != N_VARS:'),
    'pb_primes': (PBF, 'PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)'),
    'pb_halton_t0': (PBF, 'HALTON_T0 = PRIMES[N_VARS - 1]'),
    'pb_poll_design': (PBF, "POLL_DESIGN = 'orthomads_n_plus_1_neg'"),
    'pb_delta0': (PBF, 'DELTA_0 = 4'),
    'pb_delta_min': (PBF, 'DELTA_MIN = 1'),
    'pb_completion_cap': (PBF, 'COMPLETION_CAP = 30'),
    'pb_max_new': (PBF, 'MAX_NEW_EVALUATIONS = 20'),
    'pb_barrier_poll': (PBF, 'BARRIER_STOP_PER_POLL = 2'),
    'pb_barrier_overall': (PBF, 'BARRIER_STOP_OVERALL = 3'),
    'pb_box_offsets': (PBF, 'for d in product((-1, 0, 1), repeat=N_VARS):'),
    'pb_success_double': (PBF, 'delta *= 2'),
    'pb_failure_halve': (PBF, 'delta = max(DELTA_MIN, delta // 2)'),
    'pb_tie_rule': (PBF, "best = min(improvers, key=lambda c: (c['F_eur'], c['I_x_eur'], c['label']))"),
    'pb_barrier_classify': (PBF, 'if f_x is None or not math.isfinite(f_x):'),
    'pb_resolution': (PBF, 'return max(bar_x + bar_inc, sigma_q)'),
    'pb_cache_hit': (PBF, "entry['disposition'] = 'cache_hit'"),
    'pb_year_coordinate_ordinal': (PBF, 'zY = index into the instance years (2025, 2030, 2035; the'),
    'pb_one_year_A1': (PBF, 'A1  ACCEPTED: 7 variables with one common year'),
    's53_2n': (S53F, 'N_DIRECTIONS = 2 * N_VARS                  # 14: +-h_j'),
    's53_min_feasible': (S53F, 'MIN_FEASIBLE_POLL_POINTS = N_VARS + 1      # 8'),
    's53_max_new': (S53F, 'MAX_NEW_EVALUATIONS = 60'),
    's53_snap_levels': (S53F, "SNAP_KEY_LEVELS = ('l1_to_rounded', 'linf_to_rounded', 'l1_to_incumbent', 'same_year_as_incumbent', 'lower_I',"),
    's53_unit_mesh_always': (S53F, 'POLL (Delta stays 1: the STEP4 5.2 / ruling A3 doubling is NOT applied in this continuation)'),
    's53_coord_names_7': (S53F, "COORD_NAMES = ('zP5', 'zE5', 'zP7', 'zE7', 'zP9', 'zE9', 'zY')"),
    'e1_advance_window': (E1F, 'ADVANCE_WINDOW = 10'),
    'e1_n_max': (E1F, 'N_MAX_NEW_EVALUATIONS = 30'),
    'e1_max_polls': (E1F, 'MAX_POLLS = 10'),
    'e1_loop': (E1F, 'def run_continuation_e1(lattice, cache, key_of, incumbent, evaluate_fn, sigma_q, k0=HALTON_K0,'),
    'e1_uses_pb_lattice': (E1F, 'lattice = PB.Lattice(years, costs)'),
    'gl_dim': (GLF, 'self.dim = 2 * len(self.nodes) * len(self.years)'),
    'gl_activation_pairs': (GLF, 'ACTIVATION_PAIRS = ((1, 1), (1, 2))'),
    'gl_completion_cap': (GLF, 'COMPLETION_CAP = 60   # Addendum 88 R3.4'),
    'gl_n_max': (GLF, 'N_MAX = 120           # Addendum 88 R3.4'),
    'gl_coord_steps': (GLF, "for coord, delta in (('zP', +1), ('zP', -1), ('zE', +1), ('zE', -1)):"),
    'gl_deactivate_smallest': (GLF, "raw.append(('b', f'deactivate:n{n}:y{y}', self.with_pair(z_inc, n, y, 0, 0)))"),
    'gl_shift': (GLF, "raw.append(('c', f'shift:n{n}:y{y}->y{target}' + (':merged' if (tp, te) != (0, 0) else ''), new))"),
    'gl_size_bound': (GLF, 'return 2 * len(self.cohorts) + 5 * n_active'),
    'gl_resolution_by_bound': (GLF, 'def resolution_by_bound(points, f_lb, margin, ub):'),
    'gl_key_schema_v2': (GLF, "KEY_SCHEMA = 'v2'"),
    'srp_master_check': (SRP_, 'def _check_candidate_first_stage_feasibility(planning_problem, candidate_solution):'),
    'srp_rebuild_totals': (SRP_, 'def _rebuild_candidate_total_capacities(planning_problem, candidate_solution):'),
}

CHECKS = []


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _ck(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {detail}' if detail is not None and not ok else ''))


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def _input_record(path):
    return {'path': path, 'sha256': _sha(path), 'tracked': bool(_git(['ls-files', '--', path]).strip()),
            'clean': not _git(['status', '--porcelain', '--', path]).strip()}


def _other_copy_running():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me, found = os.getpid(), []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(os.path.basename(t) == SCRIPT for t in tok[2:4]):
            found.append(ln.strip())
    return found


def locate_anchors():
    out, missing, cache = {}, [], {}
    for key, (path, text) in ANCHORS.items():
        if path not in cache:
            with open(path) as handle:
                cache[path] = handle.read().splitlines()
        lines = [i + 1 for i, ln in enumerate(cache[path]) if text in ln]
        if not lines:
            missing.append(key)
        out[key] = {'file': path, 'text': text, 'lines': lines}
    return out, missing


def _count_text(path, text):
    with open(path) as handle:
        return handle.read().count(text)


# ======================================================================================================================
#  Q1 -- the multiplier
# ======================================================================================================================
def q1_network_files(case):
    """Loads and fl_reg flags of every network file at the instance years, and each network's params."""
    adn_nodes = sorted(int(dn['connection_node_id']) for dn in case['DistributionNetworks'])
    nets = [('TSO', case['TransmissionNetwork'])] + [(f"DSO{dn['connection_node_id']}", dn)
                                                       for dn in case['DistributionNetworks']]
    out = {'adn_nodes': adn_nodes, 'networks': {}}
    for tag, spec in nets:
        name = spec['name']
        with open(os.path.join(DATA_DIR, name, spec['params_file'])) as handle:
            params = json.load(handle)
        per_year = {}
        for y in YEARS:
            path = os.path.join(DATA_DIR, name, f'{name}_{y}.json')
            with open(path) as handle:
                data = json.load(handle)
            loads = data['loads']
            fl = [ld for ld in loads if bool(ld['fl_reg'])]
            per_year[str(y)] = {'file': path, 'sha256': _sha(path), 'n_loads': len(loads), 'n_fl_reg': len(fl),
                                'fl_reg_buses': sorted(int(ld['bus']) for ld in fl)}
        out['networks'][tag] = {'name': name, 'params_file': spec['params_file'],
                                'obj_type': params.get('obj_type'), 'fl_reg': params.get('fl_reg'),
                                'per_year': per_year}
    return out


def q1_market_composition(case_3x3):
    """Production market read (`_read_market_data_from_file`) of the 3 x 3 case on a bare namespace (no network, no
    model): the SP cost_flex / cost_energy_p draws; their probability-weighted means (the audit-A formula,
    p515_s56_expected_profiles.compute_means) against the stored EV profiles; and the harness's own scaling
    (`H._scaled_cost_flex`) applied before vs after the mean."""
    pp = types.SimpleNamespace(
        data_dir=DATA_DIR, market_data_file=case_3x3['MarketData'],
        random_seed=int(case_3x3['RandomSeed']), years={int(y): v for y, v in case_3x3['Years'].items()},
        days=case_3x3['Days'], num_market_scenarios=int(case_3x3['NumMarketScenarios']),
        prob_market_scenarios={}, cost_energy_p={}, cost_flex={})
    buf = io.StringIO()
    with redirect_stdout(buf):
        SRP._read_market_data_from_file(pp)
    with open(EV_PROFILES) as handle:
        ev = json.load(handle)
    blocks, n_mean_equal, n_mean_equal_energy = {}, 0, 0
    m_results = {str(m): {'n_bitwise_equal': 0, 'max_rel_diff': 0.0} for m in (2.0,) + M_VALUES_INFORMATIONAL}
    for y in sorted(pp.years):
        w = np.asarray(pp.prob_market_scenarios[y], dtype=float)
        for d in pp.days:
            A = pp.cost_flex[y][d]
            Ae = pp.cost_energy_p[y][d]
            mean = np.sum(w[:, None] * A, axis=0).reshape(1, -1)
            mean_e = np.sum(w[:, None] * Ae, axis=0).reshape(1, -1)
            stored = np.asarray(ev['arrays'][f'market|flex|{y}|{d}'], dtype=np.float64).reshape(1, -1)
            stored_e = np.asarray(ev['arrays'][f'market|energy|{y}|{d}'], dtype=np.float64).reshape(1, -1)
            eq = mean.tobytes() == stored.tobytes()
            eq_e = mean_e.tobytes() == stored_e.tobytes()
            n_mean_equal += int(eq)
            n_mean_equal_energy += int(eq_e)
            rec = {'shape': list(A.shape), 'dtype': str(A.dtype), 'probabilities': [float(v) for v in w],
                   'mean_equals_stored_ev_bitwise': eq, 'energy_mean_equals_stored_ev_bitwise': eq_e,
                   'flex_min': float(A.min()), 'flex_max': float(A.max()), 'flex_mean_of_mean': float(mean.mean())}
            for m in (2.0,) + M_VALUES_INFORMATIONAL:
                scaled_then_mean = np.sum(w[:, None] * H._scaled_cost_flex(A, m), axis=0).reshape(1, -1)
                mean_then_scaled = H._scaled_cost_flex(mean, m)
                same = scaled_then_mean.tobytes() == mean_then_scaled.tobytes()
                rel = float(np.max(np.abs(scaled_then_mean - mean_then_scaled) / np.abs(mean_then_scaled)))
                m_results[str(m)]['n_bitwise_equal'] += int(same)
                m_results[str(m)]['max_rel_diff'] = max(m_results[str(m)]['max_rel_diff'], rel)
                if m == 2.0:
                    rec['m2_mean_of_scaled_equals_scaled_mean_bitwise'] = same
            blocks[f'{y}|{d}'] = rec
    return {'method': ('shared_resources_planning._read_market_data_from_file on a namespace carrying the 3 x 3 case '
                       'fields (MarketData, RandomSeed, Years, Days, NumMarketScenarios); mean = sum_s w_s A[s] '
                       '(audit A formula); scaling = p515_s44_campaign_harness._scaled_cost_flex'),
            'case': CASE_3X3, 'n_blocks': len(blocks), 'n_flex_mean_equal_stored_ev': n_mean_equal,
            'n_energy_mean_equal_stored_ev': n_mean_equal_energy, 'composition_by_m': m_results,
            'per_block': blocks, 'production_stdout_lines': len(buf.getvalue().splitlines())}


def q1_published_records():
    out = {}
    for tag, pattern in PUBLISHED_2X_RECORD_GLOBS.items():
        files = sorted(f for f in glob.glob(pattern) if _git(['ls-files', '--', f]).strip())
        if not files:
            out[tag] = {'pattern': pattern, 'n_files': 0}
            continue
        n_all_ok = 0
        for f in files:  # every committed record of the cell set: applied checks and terminal read-back
            with open(f) as handle:
                r = json.load(handle)
            a = r.get('flex_price_applied_in_child') or {}
            t = r.get('flex_price_readback_terminal') or {}
            n_all_ok += int(r.get('flex_price_multiplier') == 2.0 and bool(a) and all(a.get('checks', {}).values())
                            and t.get('all_match') is True and t.get('tso_arrays_still_the_planning_arrays') is True)
        path = files[0]
        with open(path) as handle:
            rec = json.load(handle)
        applied = rec.get('flex_price_applied_in_child') or {}
        rb = applied.get('readback_pre_run') or {}
        post = rec.get('flex_price_readback_terminal') or {}
        out[tag] = {'pattern': pattern, 'n_files': len(files), 'n_files_all_flex_checks_ok': n_all_ok,
                    'example': path, 'example_sha256': _sha(path),
                    'flex_price_multiplier': rec.get('flex_price_multiplier'),
                    'flex_price_label': rec.get('flex_price_label'),
                    'where_applied': applied.get('where_applied'),
                    'checks': applied.get('checks'),
                    'readback_pre_run': {k: rb.get(k) for k in ('n_blocks', 'n_flex_coefficients', 'n_bitwise_exact',
                                                                'max_rel_dev', 'all_match')},
                    'profile_identical_across_dso': applied.get('profile_identical_across_dso'),
                    'post_run_readback_keys_present': sorted(k for k in rec if 'readback' in k and 'flex' in k),
                    'post_run_readback': {k: post.get(k) for k in ('n_blocks', 'n_coefficients',
                                                                   'n_bitwise_equal_m_times_original', 'all_match',
                                                                   'tso_arrays_still_the_planning_arrays')} if post else None,
                    'all_record_flex_fields_ok': (bool(applied) and all(applied.get('checks', {}).values())
                                                  and bool(post) and post.get('all_match') is True
                                                  and post.get('tso_arrays_still_the_planning_arrays') is True)}
    return out


def q1_multiscenario_scan():
    """Committed campaign specs: entries carrying both a derived (multi-scenario) instance and a multiplier != 1."""
    specs = [f for f in _git(['ls-files', 'data']).splitlines() if os.path.basename(f).startswith('campaign_spec')
             and f.endswith('.json')]
    n_derived, hits = 0, []
    for f in specs:
        with open(f) as handle:
            spec = json.load(handle)
        derived = (spec.get('configuration') or {}).get('derived_instance')
        if not derived:
            continue
        n_derived += 1
        for e in spec.get('candidates') or spec.get('entries') or []:
            m = e.get('flex_price_multiplier') if isinstance(e, dict) else None
            if m is not None and m != 1.0:
                hits.append({'spec': f, 'entry': e.get('label'), 'm': m})
    return {'scope': ('every git-tracked data/**/campaign_spec*.json; a spec counts as multi-scenario when '
                      'configuration.derived_instance is set'), 'n_specs_scanned': len(specs),
            'n_specs_with_derived_instance': n_derived, 'n_entries_with_m_not_1': len(hits), 'hits': hits[:20]}


# ======================================================================================================================
#  Q3 -- staged domain, production admissibility
# ======================================================================================================================
def production_ess(case):
    """A real SharedEnergyStorageData set up as `_read_planning_problem` sets it (srp ~9044-9066): params and cost
    workbook through the production readers, `create_shared_energy_storages` on a namespace carrying the years and the
    active DN nodes. No network, no model."""
    years = {int(y): int(v) for y, v in case['Years'].items()}
    nodes = [int(dn['connection_node_id']) for dn in case['DistributionNetworks']]
    d = ESSD.SharedEnergyStorageData()
    d.data_dir = DATA_DIR
    d.years = years
    d.discount_factor = float(case['DiscountFactor'])
    d.params_file = case['SharedEnergyStorage']['params_file']
    d.read_parameters_from_file()
    pp_ns = types.SimpleNamespace(
        years=years, transmission_network=types.SimpleNamespace(active_distribution_network_nodes=nodes),
        distribution_networks={n: types.SimpleNamespace(name=dn['name'])
                               for n, dn in zip(nodes, case['DistributionNetworks'])})
    d.create_shared_energy_storages(pp_ns)
    d.data_file = case['SharedEnergyStorage']['data_file']
    d.read_shared_energy_storage_data_from_file()
    d.active_distribution_network_nodes = nodes
    return types.SimpleNamespace(shared_ess_data=d), nodes, list(years)


def make_production_check(pp, nodes, years):
    def candidate(z):
        inv = {n: {y: {'s': GL.GRANULE_P_MVA * z.get((n, y), (0, 0))[0],
                       'e': GL.GRANULE_E_MWH * z.get((n, y), (0, 0))[1]} for y in years} for n in nodes}
        cand = {'investment': inv, 'total_capacity': {n: {} for n in nodes}}
        SRP._rebuild_candidate_total_capacities(pp, cand)
        return cand

    def check(z):
        cand = candidate(z)
        ok, why = SRP._check_candidate_first_stage_feasibility(pp, cand)
        return ok, why
    return check, candidate


def as_dict(z_tuple, lat):
    return {(n, y): lat.pair(z_tuple, n, y) for n, y in lat.cohorts if lat.pair(z_tuple, n, y) != (0, 0)}


def as_tuple(zd, lat):
    z = lat.x0()
    for (n, y), (p, e) in zd.items():
        z = lat.with_pair(z, n, y, p, e)
    return z


def _reason_classes(why):
    cls = set()
    for r in [s for s in why.split('; ') if s]:
        if r.startswith('negative'):
            cls.add('negative')
        elif 'E/S ratio' in r:
            cls.add('duration_ratio')
        elif 'maximum energy capacity' in r:
            cls.add('cumulative_cap')
        elif 'budget' in r:
            cls.add('budget')
        else:
            cls.add('other:' + r[:40])
    return sorted(cls)


def neighbourhood_2n(lat, check, z_inc):
    raw, adm, rej = 0, [], Counter()
    for i in range(lat.dim):
        for s in (+1, -1):
            z = list(z_inc)
            z[i] += s
            z = tuple(z)
            raw += 1
            ok, why = check(as_dict_raw(z, lat))
            if ok:
                adm.append(lat.label(z))
            else:
                for c in _reason_classes(why):
                    rej[c] += 1
    return {'n_raw': raw, 'n_admissible': len(adm), 'admissible': adm, 'rejected_by_class': dict(rej)}


def as_dict_raw(z, lat):
    """Like as_dict but keeps negative entries (the production check must see them)."""
    return {(n, y): lat.pair(z, n, y) for n, y in lat.cohorts if lat.pair(z, n, y) != (0, 0)}


def cohort_locally_admissible(zp, ze):
    return zp >= 0 and ze >= 0 and ((zp == 0 and ze == 0) or (zp > 0 and zp <= ze <= 2 * zp))


def neighbourhood_box(lat, check, z_inc, pp_years):
    """Every z with ||z - z_inc||_inf <= 1, z != z_inc. Product of per-cohort offsets (9 each); a cohort whose own
    offset violates sign / P=0<=>E=0 / duration is rejected by production on that cohort's own reasons whatever the
    others are (the per-cohort rows of `_check_candidate_first_stage_feasibility`), which is CHECKED here per cohort
    and offset by running production on z_inc with only that cohort changed. Every product of locally admissible
    offsets is then decided by production (cumulative cap and budget couple the cohorts)."""
    per_cohort, local_check_ok = [], True
    local_detail = []
    for n, y in lat.cohorts:
        zp0, ze0 = lat.pair(z_inc, n, y)
        opts = []
        for dp, de in product((-1, 0, 1), repeat=2):
            zp, ze = zp0 + dp, ze0 + de
            local = cohort_locally_admissible(zp, ze)
            zd = as_dict_raw(lat.with_pair(z_inc, n, y, zp, ze), lat)
            _ok, why = check(zd)
            own = [r for r in why.split('; ') if r and f'node {n}, year {y}' in r
                   and ('negative' in r or 'E/S ratio' in r)]
            if bool(own) == local:
                local_check_ok = False
                local_detail.append({'cohort': [n, y], 'pair': [zp, ze], 'local': local, 'own_reasons': own})
            if local:
                opts.append((zp, ze))
        per_cohort.append(opts)
    n_raw_product = 1
    for o in per_cohort:
        n_raw_product *= len(o)
    adm, rej = [], Counter()
    for combo in product(*per_cohort):
        z = z_inc
        for (n, y), (zp, ze) in zip(lat.cohorts, combo):
            z = lat.with_pair(z, n, y, zp, ze)
        if z == z_inc:
            continue
        ok, why = check(as_dict(z, lat))
        if ok:
            adm.append(z)
        else:
            for c in _reason_classes(why):
                rej[c] += 1
    by_n_changed = Counter(sum(1 for a, b in zip(z, z_inc) if a != b) for z in adm)
    by_n_cohorts_changed = Counter(sum(1 for c in lat.cohorts if lat.pair(z, *c) != lat.pair(z_inc, *c)) for z in adm)
    return {'n_raw_offsets': 3 ** lat.dim - 1, 'n_locally_admissible_products_excl_incumbent': n_raw_product - 1,
            'per_cohort_local_option_counts': {f'n{n}_y{y}': len(o) for (n, y), o in zip(lat.cohorts, per_cohort)},
            'local_rule_matches_production_per_cohort': local_check_ok, 'local_rule_mismatches': local_detail[:10],
            'n_admissible': len(adm), 'rejected_products_by_class': dict(rej),
            'admissible_by_n_coordinates_changed': {str(k): v for k, v in sorted(by_n_changed.items())},
            'admissible_by_n_cohorts_changed': {str(k): v for k, v in sorted(by_n_cohorts_changed.items())}}, adm


def neighbourhood_r3(lat, check, z_inc):
    nb = lat.neighbourhood(z_inc)
    agree = all(check(as_dict(tuple(p['z']), lat))[0] for p in nb['points'])
    dropped_prod = [check(as_dict_raw(tuple(p['z']), lat))[0] for p in nb['dropped']]
    by_part = Counter(p['part'] for p in nb['points'])
    by_move = Counter(p['move'].split(':')[0] for p in nb['points'])
    return {'A_active_cohorts': nb['A'], 'size_bound': nb['size_bound'], 'n_raw': nb['n_raw'],
            'n_raw_by_part': nb['n_raw_by_part'], 'n_admissible': nb['n_admissible'],
            'admissible_by_part': dict(by_part), 'admissible_by_move': dict(by_move),
            'n_dropped': nb['n_dropped'], 'dropped_by_reason_class': nb['dropped_by_reason_class'],
            'n_duplicates': len(nb['duplicates']), 'over_completion_cap_60': nb['over_completion_cap'],
            'every_listed_point_admissible_by_production': agree,
            'every_dropped_point_rejected_by_production': not any(dropped_prod),
            'points': [{'move': p['move'], 'label': p['label'], 'I_x_eur': p['I_x_eur']} for p in nb['points']]}


def r3_primitive_moves(lat, z, n, y):
    """R3 single-cohort primitives on cohort (n, y), split by direction (informational composition only)."""
    zp, ze = lat.pair(z, n, y)
    dec, inc = [], []
    if (zp, ze) != (0, 0):
        dec += [('zP-1', (zp - 1, ze)), ('zE-1', (zp, ze - 1))]
        inc += [('zP+1', (zp + 1, ze)), ('zE+1', (zp, ze + 1))]
        if (zp, ze) in GL.SMALLEST_COHORTS:
            dec.append(('deactivate', (0, 0)))
    else:
        inc += [('activate_0.25_0.5', (1, 1)), ('activate_0.25_1', (1, 2))]
    return dec, inc


def redistribution_moves(lat, check, z_inc):
    """INFORMATIONAL, not an existing neighbourhood: one R3 decrease on cohort i and one R3 increase on cohort j != i."""
    seen, adm, raw = set(), [], 0
    for ci in lat.cohorts:
        dec, _ = r3_primitive_moves(lat, z_inc, *ci)
        for _nd, (dp, de) in dec:
            z1 = lat.with_pair(z_inc, ci[0], ci[1], dp, de)
            for cj in lat.cohorts:
                if cj == ci:
                    continue
                _, inc = r3_primitive_moves(lat, z_inc, *cj)
                for _ni, (ip, ie) in inc:
                    z2 = lat.with_pair(z1, cj[0], cj[1], ip, ie)
                    raw += 1
                    if z2 in seen:
                        continue
                    seen.add(z2)
                    if check(as_dict_raw(z2, lat))[0]:
                        adm.append(z2)
    same_node = sum(1 for z in adm if len({n for (n, _y) in [c for c in lat.cohorts
                                                             if lat.pair(z, *c) != lat.pair(z_inc, *c)]}) == 1)
    return {'definition': ('one R3 decrease (zP-1, zE-1, or deactivation of a smallest cohort) on cohort i and one R3 '
                           'increase (zP+1, zE+1, or activation to 0.25/0.5 or 0.25/1) on cohort j != i; NOT an '
                           'existing neighbourhood'), 'n_raw': raw, 'n_distinct': len(seen),
            'n_admissible': len(adm), 'n_admissible_within_one_node': same_node,
            'n_admissible_across_nodes': len(adm) - same_node}


def staged_domain(lat, check, pp_ess):
    """Every admissible staged point: per node, cohort triples with cumulative zE <= floor(max_cap / 0.5) (all cohorts
    are alive in 2035 on this grid -- asserted), then node triples with closed-form cost <= B + 1 EUR, each decided by
    the production check. Cross-checks: the common-year single-cohort subset = 1,705 (x = 0 incl.; the s47/s51/S55
    domain), and the <= 1 cohort per node subset = G-CC-4's 6,694."""
    zE_cap = int(round(lat.inst.max_capacity_mwh / GL.GRANULE_E_MWH))
    all_alive_last = all(lat.years[-1] in lat.alive_blocks[y] for y in lat.years)
    opts = [(0, 0)] + [(p, e) for e in range(1, zE_cap + 1) for p in range(1, zE_cap + 1) if p <= e <= 2 * p]
    unit = {}
    for y in lat.years:
        unit[y] = (sum(a for a, _b in lat._coef[y]) * GL.GRANULE_P_MVA, sum(b for _a, b in lat._coef[y]) * GL.GRANULE_E_MWH)
    B = lat.inst.budget_eur
    per_node = []
    for combo in product(opts, repeat=len(lat.years)):
        if sum(e for _p, e in combo) > zE_cap:
            continue
        c = sum(unit[y][0] * p + unit[y][1] * e for (p, e), y in zip(combo, lat.years))
        if c <= B + 1.0:
            per_node.append((c, combo))
    per_node.sort()
    costs = np.array([c for c, _ in per_node])
    n_candidates, adm = 0, []
    for i, (c1, t1) in enumerate(per_node):
        if c1 > B + 1.0:
            break
        for c2, t2 in per_node:
            if c1 + c2 > B + 1.0:
                break
            j_end = int(np.searchsorted(costs, B + 1.0 - c1 - c2, side='right'))
            for c3, t3 in per_node[:j_end]:
                n_candidates += 1
                zd = {}
                for n, trip in zip(lat.nodes, (t1, t2, t3)):
                    for y, pair in zip(lat.years, trip):
                        if pair != (0, 0):
                            zd[(n, y)] = pair
                if check(zd)[0]:
                    adm.append(zd)
    n_common_single = sum(1 for zd in adm if len({y for (_n, y) in zd}) <= 1)
    n_le1_per_node = sum(1 for zd in adm if all(sum(1 for (n2, _y) in zd if n2 == n) <= 1 for n in lat.nodes))
    n_staged = sum(1 for zd in adm if any(sum(1 for (n2, _y) in zd if n2 == n) >= 2 for n in lat.nodes))
    by_n_cohorts = Counter(len(zd) for zd in adm)
    return {'cumulative_cap_granules_per_node': zE_cap, 'all_cohorts_alive_in_last_block': all_alive_last,
            'per_cohort_options_incl_zero': len(opts), 'per_node_triples_cap_ok_cost_le_B_plus_1': len(per_node),
            'n_candidates_checked_by_production': n_candidates, 'n_admissible_incl_x0': len(adm),
            'n_common_year_single_cohort_incl_x0': n_common_single,
            'n_at_most_one_cohort_per_node_incl_x0': n_le1_per_node,
            'n_with_a_node_holding_two_or_more_cohorts': n_staged,
            'by_number_of_active_cohorts': {str(k): v for k, v in sorted(by_n_cohorts.items())}}


def q3_search_records():
    out = {}
    for tag, path in SEARCH_RECORDS.items():
        with open(path) as handle:
            d = json.load(handle)
        polls = []
        for p in d['poll_history']:
            disp = Counter('duplicate' if str(c['disposition']).startswith('duplicate') else c['disposition']
                           for c in p['candidates'])
            parts = Counter(c['poll_part'] for c in p['candidates'])
            polls.append({'poll_index': p['poll_index'], 'halton_t': p.get('halton_t'), 'delta': p['poll_size_delta'],
                          'incumbent': p['incumbent']['label'], 'n_points_listed': len(p['candidates']),
                          'by_part': dict(parts), 'dispositions': dict(disp),
                          'n_new_evaluations_listed': p['n_new_evaluations'], 'n_cache_hits': p['n_cache_hits'],
                          'n_evaluated_in_batches': sum(len(b) for b in p['batches']),
                          'batch_sizes': [len(b) for b in p['batches']],
                          'completion_n_feasible': (p.get('completion') or {}).get('n_feasible'),
                          'decision': p['decision']})
        out[tag] = {'path': path, 'sha256': _sha(path), 'termination': d['termination']['reason'],
                    'n_polls': d['n_polls'], 'n_new_evaluations': d['n_new_evaluations'],
                    'n_cache_hits_total': sum(p['n_cache_hits'] for p in polls),
                    'n_evaluated_total_in_batches': sum(p['n_evaluated_in_batches'] for p in polls),
                    'final_incumbent': (d.get('final_incumbent') or {}).get('label'),
                    'wall_clock_s': d.get('wall_clock_s'), 'polls': polls}
    return out


def gate_artifacts():
    out = {}
    for name, (path, flag) in GATE_ARTIFACTS.items():
        rec = {'path': path, 'exists': os.path.exists(path)}
        if rec['exists']:
            rec['sha256'] = _sha(path)
            rec['tracked'] = bool(_git(['ls-files', '--', path]).strip())
            with open(path) as handle:
                d = json.load(handle)
            if flag:
                rec[flag] = d.get(flag)
            else:
                for k in ('all_ok', 'all_pass', 'gate_pass', 'all_checks_ok', 'ok', 'integrity'):
                    if k in d:
                        v = d[k]
                        rec[k] = v if not isinstance(v, (dict, list)) else (
                            {kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))}
                            if isinstance(v, dict) else len(v))
        out[name] = rec
    return out


# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = args.out
    started = datetime.now(timezone.utc).isoformat()
    others = _other_copy_running()
    if others:
        _log(f'REFUSE: another copy of {SCRIPT} is running: {others}')
        return 2
    if os.path.exists(out_dir):
        present = set(os.listdir(out_dir))
        if not present <= ALLOWED_PREEXISTING:
            _log(f'REFUSE: output directory {out_dir} exists and holds {sorted(present - ALLOWED_PREEXISTING)}')
            return 2
    else:
        os.makedirs(out_dir)
    anchors, missing = locate_anchors()
    if missing:
        _log(f'REFUSE: anchors not found: {missing}')
        return 2
    with open(CASE_3X3) as handle:
        case_3x3 = json.load(handle)
    with open(CASE_SRP1) as handle:
        case_srp1 = json.load(handle)
    inputs = {p: _input_record(p) for p in sorted({a['file'] for a in anchors.values()} | {
        CASE_3X3, CASE_SRP1, EV_PROFILES, W173_JSON, G_CC4_JSON, SCRIPT, 'definitions.py', 'shared_energy_storage_data.py',
        'p513_solve_profile_guard.py', 'gate_result_io.py', *SEARCH_RECORDS.values()})}

    # ---------------------------------------------------------------- Q1
    _log('Q1: network files, params, constants')
    q1 = {}
    q1['networks_3x3_case'] = q1_network_files(case_3x3)
    nets = q1['networks_3x3_case']
    adn = set(nets['adn_nodes'])
    tso = nets['networks']['TSO']
    _ck('Q1 TSO: every fl_reg load sits at an ADN interface bus (2025/2030/2035), so flexibility_cost charges none',
        all(set(v['fl_reg_buses']) <= adn for v in tso['per_year'].values()),
        {y: v['fl_reg_buses'] for y, v in tso['per_year'].items()})
    _ck('Q1 every network obj_type COST and fl_reg True (the harness precondition OBJ_MIN_COST + fl_reg)',
        all(v['obj_type'] == 'COST' and v['fl_reg'] is True for v in nets['networks'].values()),
        {k: (v['obj_type'], v['fl_reg']) for k, v in nets['networks'].items()})
    dso_fl = {k: {y: v['n_fl_reg'] for y, v in n['per_year'].items()} for k, n in nets['networks'].items() if k != 'TSO'}
    q1['dso_flex_loads'] = dso_fl
    q1['expected_flex_coefficients_per_dso_block_one_scenario'] = {
        k: {y: n * 24 * 2 for y, n in v.items()} for k, v in dso_fl.items()}
    q1['constants'] = {'PENALTY_FLEXIBILITY': DEF.PENALTY_FLEXIBILITY,
                       'PENALTY_FLEXIBILITY_USAGE': DEF.PENALTY_FLEXIBILITY_USAGE,
                       'EQUALITY_TOLERANCE': DEF.EQUALITY_TOLERANCE,
                       'H.FLEX_PRICE_VARS': list(H.FLEX_PRICE_VARS), 'H.FLEX_PRICE_LABEL': H.FLEX_PRICE_LABEL,
                       'H.FLEX_PRICE_READBACK_REL_TOL': H.FLEX_PRICE_READBACK_REL_TOL}
    ess_modules = ['shared_energy_storage_data.py', 'shared_energy_storage.py', 'shared_energy_storage_parameters.py',
                   'shared_ess_price_taker.py']
    q1['cost_flex_text_occurrences'] = {m: _count_text(m, 'cost_flex') for m in ess_modules + [MCH, SRP_, NET,
                                                                                             'uncoordinated_benchmark.py']}
    _ck('Q1 no ESSO module references cost_flex (the ESSO prices no flexibility)',
        all(q1['cost_flex_text_occurrences'][m] == 0 for m in ess_modules),
        {m: q1['cost_flex_text_occurrences'][m] for m in ess_modules})
    _log('Q1: production market read of the 3 x 3 case and the 2x composition')
    q1['market_composition_3x3'] = q1_market_composition(case_3x3)
    mc = q1['market_composition_3x3']
    _ck('Q1 SP cost_flex draws are (3 x 24) float64 per (year, day), 12 blocks',
        mc['n_blocks'] == 12 and all(b['shape'] == [3, 24] and b['dtype'] == 'float64' for b in mc['per_block'].values()))
    _ck('Q1 the probability-weighted mean of the production draws == the stored EV flex means bitwise (12/12)',
        mc['n_flex_mean_equal_stored_ev'] == 12, mc['n_flex_mean_equal_stored_ev'])
    _ck('Q1 the same for the energy price (12/12) (the read reproduces the instance)',
        mc['n_energy_mean_equal_stored_ev'] == 12, mc['n_energy_mean_equal_stored_ev'])
    _ck('Q1 m = 2: mean(2 x draws) == 2 x mean(draws) bitwise in every block (EV-at-2x = mean of SP-at-2x exactly)',
        mc['composition_by_m']['2.0']['n_bitwise_equal'] == 12, mc['composition_by_m']['2.0'])
    q1['published_2x_records'] = q1_published_records()
    for tag, rec in q1['published_2x_records'].items():
        _ck(f'Q1 published 2x records {tag}: every record m = 2, every applied check True (TSO / planning / energy '
            f'arrays unchanged), terminal read-back all_match and TSO arrays still the planning arrays',
            rec.get('n_files', 0) > 0 and rec.get('n_files_all_flex_checks_ok') == rec.get('n_files')
            and rec.get('all_record_flex_fields_ok'),
            {k: rec.get(k) for k in ('n_files', 'n_files_all_flex_checks_ok', 'flex_price_multiplier', 'checks')})
    q1['multiscenario_scan'] = q1_multiscenario_scan()

    # ---------------------------------------------------------------- Q2 (executable facts)
    _log('Q2: executable facts')
    q2 = {}
    try:
        PB.halton_point(17, n=18)
        q2['pb_halton_point_n18'] = 'no error'
    except Exception as exc:  # recorded, the point of the probe
        q2['pb_halton_point_n18'] = f'{type(exc).__name__}: {exc}'
    q2['pb_primes_len'] = len(PB.PRIMES)
    try:
        PB.Lattice(YEARS, {y: {'power': 1.0, 'energy': 1.0} for y in YEARS}, nodes=(5, 7, 9, 11))
        q2['pb_lattice_4_nodes'] = 'no error'
    except Exception as exc:
        q2['pb_lattice_4_nodes'] = f'{type(exc).__name__}: {exc}'
    q2['harness_canonical_candidate_keys'] = sorted(H.canonical_candidate({5: (0.0, 0.0), 7: (0.25, 1.0),
                                                                           9: (0.0, 0.0)}, investment_year=2025))
    _ck('Q2 PB Halton table cannot serve 18 variables (IndexError expected)',
        q2['pb_halton_point_n18'].startswith('IndexError'), q2['pb_halton_point_n18'])
    _ck('Q2 PB.Lattice refuses any node set other than 3 nodes (N_VARS = 7 hard-wired)',
        q2['pb_lattice_4_nodes'].startswith('ValueError'), q2['pb_lattice_4_nodes'])
    _ck('Q2 harness canonical candidate = {investment_year, nodes} (one year per candidate)',
        q2['harness_canonical_candidate_keys'] == ['investment_year', 'nodes'], q2['harness_canonical_candidate_keys'])
    q2['gate_artifacts'] = gate_artifacts()
    q2['p516_lattice_users'] = [f for f in _git(['ls-files', '*.py']).splitlines()
                                if f != GLF and ('p516_lattice' in open(f).read() or 'GeneralLattice' in open(f).read())]
    q2['e1_search_roots_on_disk'] = sorted(d for d in glob.glob('data/SRP1/Results/P515S55/e1_2x*'))

    # ---------------------------------------------------------------- Q3
    _log('Q3: committed search records')
    q3 = {'search_records': q3_search_records()}
    with open(W173_JSON) as handle:
        w173 = json.load(handle)
    q3['w173_item2_box_sizes'] = {k: {kk: (v.get(kk) if not isinstance(v.get(kk), dict) else
                                           {a: b for a, b in v.get(kk).items() if not isinstance(b, (dict, list))})
                                      for kk in ('incumbent', 'a_final_poll', 'b_box')}
                                  for k, v in w173['item2'].items() if isinstance(v, dict)}
    _log('Q3: production ESS data (3 x 3 case) and the general lattice')
    pp, nodes, years = production_ess(case_3x3)
    _ck('Q3 instance nodes / years', tuple(nodes) == NODES and tuple(years) == YEARS, [nodes, years])
    check, candidate = make_production_check(pp, nodes, years)
    inst = GL.load_instance_from_case_dict(case_3x3, DATA_DIR, case_source={'path': CASE_3X3, 'sha256': _sha(CASE_3X3)})
    lat = GL.GeneralLattice(inst)
    prm = pp.shared_ess_data.params
    q3['production_params'] = {'budget': prm.budget, 'max_capacity': prm.max_capacity,
                               'min_energy_to_power_ratio': prm.min_energy_to_power_ratio,
                               'max_energy_to_power_ratio': prm.max_energy_to_power_ratio,
                               'discount_factor': pp.shared_ess_data.discount_factor,
                               'prob_market_scenarios_workbook': list(pp.shared_ess_data.prob_market_scenarios),
                               't_cal': [pp.shared_ess_data.shared_energy_storages[y][0].t_cal for y in years],
                               'alive_blocks_general_lattice': {str(k): list(v) for k, v in lat.alive_blocks.items()}}
    # I(x) cross-check: production I (re-derived from the check's own formula) vs the general lattice
    i_2030 = lat.investment_cost(as_tuple(INCUMBENTS['plan_2030_n5_0.25_0.5__n7_1_3.5'], lat))
    _ck('Q3 the 2030 plan I(x) == 992,268.11 EUR (general lattice, master term order)', round(i_2030, 2) == PLAN_2030_I_EXPECTED,
        i_2030)
    q3['neighbourhoods'] = {}
    for name, zd in INCUMBENTS.items():
        z_inc = as_tuple(zd, lat)
        ok, why = check(zd)
        _ck(f'Q3 incumbent {name} admissible by production', ok, why)
        nb2n = neighbourhood_2n(lat, check, z_inc)
        box, _adm = neighbourhood_box(lat, check, z_inc, years)
        r3 = neighbourhood_r3(lat, check, z_inc)
        red = redistribution_moves(lat, check, z_inc)
        _ck(f'Q3 {name}: per-cohort local rule == production per-cohort reasons (box enumeration is exact)',
            box['local_rule_matches_production_per_cohort'], box['local_rule_mismatches'])
        _ck(f'Q3 {name}: every R3 point admissible by production, every dropped R3 point rejected',
            r3['every_listed_point_admissible_by_production'] and r3['every_dropped_point_rejected_by_production'])
        # the 7-variable common-year lattice (variant A / B / E1) at the same plan, when it is a common-year plan
        ys = {y for (_n, y) in zd}
        pb7 = None
        if len(ys) <= 1:
            w2_costs = None
            with open(PB.INVESTMENT_COST_RESULTS['path']) as handle:
                w2_costs = PB.unit_costs_from_w2(json.load(handle))
            plat = PB.Lattice(YEARS, w2_costs)
            yi = YEARS.index(next(iter(ys))) if ys else 0
            z7 = []
            for n in NODES:
                p_, e_ = next(((p, e) for (nn, _y), (p, e) in zd.items() if nn == n), (0, 0))
                z7 += [p_, e_]
            z7 = tuple(z7) + (yi,)
            comp = plat.completion(z7)
            pb7 = {'z7': list(z7), 'label': plat.label(z7), 'n_box_feasible_7var': comp['n_feasible'],
                   'over_cap_30': comp['over_cap'], 'n_raw_offsets_7var': comp['n_raw_offsets'],
                   'rejected_raw_offsets_by_class': comp['rejected_raw_offsets_by_class']}
        q3['neighbourhoods'][name] = {'incumbent': {f'n{n}_y{y}': [p, e] for (n, y), (p, e) in zd.items()},
                                      'label': lat.label(z_inc), 'I_x_eur': lat.investment_cost(z_inc),
                                      'budget_slack_eur': prm.budget - lat.investment_cost(z_inc),
                                      'coordinate_2n': nb2n, 'box_linf1': box, 'r3_p516': r3,
                                      'informational_redistribution_pairs': red, 'pb_7var_box_same_plan': pb7}
    _log('Q3: whole staged admissible domain (production-decided)')
    q3['staged_domain'] = staged_domain(lat, check, pp)
    sd = q3['staged_domain']
    _ck('Q3 all cohorts alive in 2035 on this grid (cumulative cap = sum over cohorts)', sd['all_cohorts_alive_in_last_block'])
    _ck('Q3 staged domain: common-year single-cohort subset == 1,705 (x = 0 incl.; the s47/s51 domain)',
        sd['n_common_year_single_cohort_incl_x0'] == 1705, sd['n_common_year_single_cohort_incl_x0'])
    with open(G_CC4_JSON) as handle:
        gcc4 = json.load(handle)
    g_le1 = gcc4['predictions_vs_outcomes']['L1']['outcome']['general_le1_year_per_node_admissible']
    _ck('Q3 staged domain: <= 1 cohort per node subset == G-CC-4 general_le1_year_per_node_admissible',
        sd['n_at_most_one_cohort_per_node_incl_x0'] == g_le1, [sd['n_at_most_one_cohort_per_node_incl_x0'], g_le1])

    # ---------------------------------------------------------------- guards, write
    guard_fail = GUARD.verify(0)
    pb_fail = PB.PARENT_GUARD.verify(0)
    _ck('own SolveProfileGuard(permitted=()) verify(0) == []', not guard_fail, guard_fail)
    _ck('PB parent guard verify(0) == []', not pb_fail, pb_fail)
    _ck('pickle load/loads 0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    payload = {
        'schema': 'p515_s56_reuse_audit_J_v1', 'script': SCRIPT, 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'interpreter': sys.executable,
        'git_head': _git(['rev-parse', 'HEAD']).strip(),
        'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) for the whole run, verify(0))',
        'instance': {'case': CASE_3X3, 'sha256': _sha(CASE_3X3), 'nodes': list(NODES), 'years': list(YEARS),
                     'note': 'admissibility is instance-independent across SRP1 and the 3 x 3 case (same ESS files)'},
        'staged_domain_definition': ('z in Z^18: (zP, zE) per (node, year), node-major then year (p516_lattice); P = 0.25 '
                                     'zP MVA, E = 0.5 zE MWh; admissible iff production '
                                     '_check_candidate_first_stage_feasibility holds on the candidate whose total '
                                     'capacities production _rebuild_candidate_total_capacities builds'),
        'Q1': q1, 'Q2': q2, 'Q3': q3, 'anchors': anchors, 'inputs': inputs, 'checks': CHECKS,
        'failed_checks': [c['check'] for c in CHECKS if not c['ok']],
        'all_checks_ok': all(c['ok'] for c in CHECKS),
        'guards': {'own': {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                           'verify_0': guard_fail},
                   'pb_parent': {'label': PB.PARENT_GUARD.label, 'counts': dict(PB.PARENT_GUARD.counts),
                                 'verify_0': pb_fail},
                   'pickle_counts': dict(PICKLE_COUNTS)},
    }
    json_path = os.path.join(out_dir, OUT_JSON)
    with open(json_path, 'w') as handle:
        GRIO.dump(payload, handle, indent=1, sort_keys=False)
    manifest = {'outputs': {OUT_JSON: _sha(json_path)}, 'inputs': inputs,
                'note': 'launch.log is written while the script runs; its sha256 is recorded in the Worker report'}
    with open(os.path.join(out_dir, OUT_MANIFEST), 'w') as handle:
        GRIO.dump(manifest, handle, indent=1)
    _log(f'all_checks_ok = {payload["all_checks_ok"]}; failed = {payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] else 3


if __name__ == '__main__':
    sys.exit(main())
