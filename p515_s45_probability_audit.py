"""P5.15 Addendum 27 item 5b (Worker task W3) -- PROBABILITY AUDIT. ZERO SOLVES.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27 ("Paper scale -- bounded task authorized ...
(b) ... preceded by a zero-solve audit of every consumer of
`shared_ess_data.prob_market_scenarios` on the oracle path"); P5_15_S44_SELECTION_REPORT.md
section 4 ("the ESS workbook's investment-cost scenario probabilities overwrite
`prob_market_scenarios`. This one is uninvestigated.").

QUESTION: is any OPERATIONAL term (TSO/DSO/ESSO objective or constraint) or the oracle's
aggregated cost Q(x) (`gross_operational_cost`, `net_operational_recourse`) weighted by the ESS
workbook's investment-cost scenario probabilities instead of the networks' market-scenario
probabilities -- at SRP1, and at paper scale?

WHAT THIS SCRIPT DOES (all zero-solve; a `SolveProfileGuard(permitted=())` is installed before
any production import, stays installed for the whole run, and `verify(0)` is checked exactly):

  S. STATIC TRACE.
     S1. Oracle-path module set: AST import closure (repo-root modules only) of the three
         oracle entry scripts p56a_oracle.py, p515_g_g1_g4_admm_gates.py,
         p515_s44_campaign_harness.py (function-level imports included).
     S2. Search: every *.py under the repository (all depths, .git excluded) for the primary
         patterns `prob_market_scenarios`, `prob_operation_scenarios`, `cost_investment`
         (the investment-cost scenario data the workbook probabilities pair with) and the broad
         patterns `prob_`, `market_scenarios`, `probability`, `omega`. Per-file hit lines are
         recorded.
     S3. Every primary-pattern hit in a PRODUCTION module (repo-root .py that is not a
         diagnostic harness) or an oracle-path harness module is mapped to its innermost
         enclosing function (AST) and classified by the declared RULES table below. The script
         FAILS if any such hit has no rule, or if any rule matches no hit (the table is proven
         complete and non-stale against the code as it is at run time).
  N. SRP1 NUMERICAL CHECK at C* (0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, year 2025), through
     the oracle's construction path: production's reader on data/SRP1/SRP1.json (checksum
     verified == p56a_oracle.CANONICAL_CHECKSUM, then injected as the oracle baseline exactly as
     p515_s44_scale_measurement.py does, with results/diagrams/logs redirected into this
     script's output directory so nothing is written outside it), `p56a_oracle.WORK_DIR`
     redirected likewise, then `p515_g_g1_g4_admm_gates._construct_arm_planning` (D arm,
     apply_rho=False), then production's own initialization sequence of
     `_run_operational_planning` (create_admm_variables, create_distribution_networks_models,
     create_transmission_network_model, create_shared_energy_storage_model,
     _prepare_*_objectives_for_admm, update_shared_energy_storage_model_to_admm) with each
     agent's `.optimize` replaced ON THIS PLANNING INSTANCE ONLY by an interceptor that records
     the call and returns "no result" (never a solver, never a fabricated solution) -- the
     precedent of p515_s44_addendum26_confirmations.py item 3. Declared intercepts, checked
     exactly: dso 3, tso 1, esso_init 1.
     Prints every probability on every object and the weight each scenario term actually
     carries in the BUILT Pyomo objects (expression-tree weights of every total_* aggregate,
     implied weights recovered from linear coefficients of the objective and of the
     expected-value constraints, the ESSO objective's coefficient set and variable index sets).
  P. PAPER SCALE, DATA ONLY (no model build): production's reader on the committed derived case
     file data/SRP1/Results/P515S44/scale_measurement/paper_build/case/SRP1__paper.json
     (~0.5 GB, ~13 s in the scale measurement), with Network.build_model,
     NetworkData.build_model, SharedEnergyStorageData.build_subproblem and
     .build_master_problem replaced by blockers that RAISE for the whole paper phase (so no
     model can be constructed). Records the probabilities every object would carry.

Usage (attached; both streams captured by the caller into the launch log):
    python -u p515_s45_probability_audit.py            # the run (write-once)
    python -u p515_s45_probability_audit.py --manifest # sha256 manifest of the output dir
"""
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')   # production plots scenarios while reading data

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402  (pyomo only)

STAGE = 'P5.15 Addendum 27 item 5b (W3) probability audit -- zero solves'
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S45', 'probability_audit')
AUDIT_JSON = 'probability_audit.json'
STDOUT_CAPTURE = 'production_stdout_capture.log'
MANIFEST_NAME = 'manifest_sha256.json'
LAUNCH_LOG = 'probability_audit_launch.log'
DATA_DIR = os.path.join(REPO, 'data', 'SRP1')
ESS_XLSX_REL = 'data/SRP1/SharedESS/SRP1_ESS.xlsx'
PAPER_CASE_REL = 'data/SRP1/Results/P515S44/scale_measurement/paper_build/case/SRP1__paper.json'
PAPER_CHECKSUM_RECORDED = '1e8bdd3e5233442a87fbe44b78281c8ae61b09c09700388fed20ab9684d18aef'
PAPER_CHECKSUM_SOURCE = ('data/SRP1/Results/P515S44/scale_measurement/paper_build/'
                         'build_child_stdout.log:33 ("[INFO] Scenario checksum: ...")')
PRIOR_COST_FILE_COMMIT = '2cada62b^'   # the file the Addendum 27 corrected cost file replaced

INVEST_YEAR = 2025
C_STAR = {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)}
C_STAR_LABEL = '0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025 (spec v14 C_star)'
EVAL_ID = 'p515s45_probability_audit'
DECLARED_INTERCEPTS = {'dso': 3, 'tso': 1, 'esso_init': 1}

ORACLE_ENTRY_SCRIPTS = ('p56a_oracle.py', 'p515_g_g1_g4_admm_gates.py', 'p515_s44_campaign_harness.py')
PRIMARY_PATTERNS = {
    'prob_market_scenarios': r'prob_market_scenarios',
    'prob_operation_scenarios': r'prob_operation_scenarios',
    'cost_investment': r'cost_investment',
}
BROAD_PATTERNS = {
    'prob_': r'prob_',
    'market_scenarios': r'market_scenarios',
    'probability': r'probability',
    'omega': r'omega',
}
# Diagnostic harnesses (not production modules) at the repo root.
HARNESS_RE = re.compile(r'^(p\d|audit_|validate_|debug\.py$|main\.py$)')

# ------------------------------------------------------------------------------------------
# Classification codes (Planner's (a)-(d) plus two declared additions):
#   A  assignment / source of a probability vector (not a consumer)
#   a  operational term: a TSO/DSO/ESSO objective term or constraint of the oracle's models
#   b  cost aggregation forming the oracle's Q(x) (gross_operational_cost and/or
#      net_operational_recourse)
#   c  Benders master / investment cost / first-stage only (not the oracle's Q(x))
#   d  dead or unused on the oracle path (not reached, or reached only on another path)
#   e  diagnostic / reporting ON the oracle path that is neither an objective term nor Q(x)
# 'source' = which probability vector the site reads: NETWORK (network.prob_*_scenarios, the
# market/operation scenario probabilities), WORKBOOK (shared_ess_data.prob_market_scenarios
# after the overwrite = the ESS workbook's investment-cost scenario probabilities), PLANNING
# (planning_problem.prob_market_scenarios).
# ------------------------------------------------------------------------------------------
RULES = {
    # ---- shared_energy_storage_data.py (the ESSO data object) ----
    ('shared_energy_storage_data.py', '__init__'): (
        'A', 'WORKBOOK', 'initializer `dict()`; rebound by _read_planning_problem then by the workbook reader'),
    ('shared_energy_storage_data.py', '_read_shared_energy_storage_data_from_file'): (
        'A', 'WORKBOOK', 'THE OVERWRITE: rebinds shared_ess_data.prob_market_scenarios to the workbook '
                         "sheet 'Scenarios' row 0 (numScenarios, p1..pN); cost_investment is read with the "
                         'same num_scenarios -> the two are paired by construction'),
    ('shared_energy_storage_data.py', '_build_master_problem'): (
        'c', 'WORKBOOK', 'Benders master: scenario set, budget row, investment-cost objective '
                         '(omega_m x cost_investment[..][s_m]); not built on the oracle path'),
    ('shared_energy_storage_data.py', '_get_expected_energy_investment_cost'): (
        'b', 'WORKBOOK', 'E_workbook[c_inv_energy(year)] -- the unit cost of the terminal SALVAGE value '
                         '(salvage_value Expression of each ESSO model -> get_salvage_value -> '
                         'terminal_salvage_value -> net_operational_recourse = gross - salvage). Enters the '
                         'NET recourse only, never gross_operational_cost, never an objective (the ESSO '
                         'objective is feasibility_penalty + AL terms). Also feeds '
                         '_get_salvage_value_sensitivities (Benders cut, c) and _get_salvage_value_results '
                         '(reporting, d). Pairing is investment-cost probability x investment-cost '
                         'scenario, i.e. correct by design (salvage cost_basis '
                         'EXPECTED_INSTALLATION_ENERGY_COST)'),
    ('shared_energy_storage_data.py', '_get_investment_cost_and_rated_capacity'): (
        'c', 'WORKBOOK', 'master-result reporting of expected investment cost'),
    ('shared_energy_storage_data.py', '_write_ess_costs_to_excel'): (
        'd', 'WORKBOOK', 'Excel results writer (investment costs); oracle calls print_results=False'),
    # ---- shared_resources_planning.py ----
    ('shared_resources_planning.py', '__init__'): (
        'A', 'PLANNING', 'planning_problem.prob_market_scenarios initializer dict()'),
    ('shared_resources_planning.py', '_read_market_data_from_file'): (
        'A', 'PLANNING', 'planning_problem.prob_market_scenarios[year] = [1/NumMarketScenarios]*NumMarketScenarios'),
    ('shared_resources_planning.py', '_read_planning_problem'): (
        'A', 'NETWORK/PLANNING/WORKBOOK', 'network.prob_market_scenarios = planning[year] (obj COST) or [1.0] '
                                          '(CONGESTION_MANAGEMENT); shared_ess_data.prob_market_scenarios = '
                                          'planning dict (later rebound by the workbook reader)'),
    ('shared_resources_planning.py', '_get_local_detector_components'): (
        'e', 'NETWORK', 'D-row detector levels (detector_penalty_total, economic_recourse_* fields); not '
                        'gross/net'),
    ('shared_resources_planning.py', '_check_candidate_first_stage_feasibility'): (
        'c', 'WORKBOOK', 'first-stage budget check (expected investment cost); p56a_oracle.check_master_feasibility'),
    ('shared_resources_planning.py', '_build_positive_bootstrap_candidate'): (
        'c', 'WORKBOOK', 'bootstrap candidate sizing from expected unit costs; not used when a candidate is given'),
    ('shared_resources_planning.py', '_scenario_probability'): (
        'a', 'NETWORK', 'helper = network pm*po; used by the TSO/DSO scenario-deviation penalties (objective '
                        'terms, skipped when |S_m|x|S_o| = 1, ACTIVE at paper scale), the tracking penalty '
                        '(no-coordination path, d) and scenario-dispersion reporting (d)'),
    ('shared_resources_planning.py', '_get_local_slack_penalty_components'): (
        'e', 'NETWORK', 'slack component diagnostics in the ADMM loop'),
    ('shared_resources_planning.py', '_get_tso_voltage_slack_state'): (
        'e', 'NETWORK', 'TSO voltage-slack diagnostic in the ADMM loop'),
    ('shared_resources_planning.py', '_get_expected_network_shared_ess_charge_discharge_mw'): (
        'e', 'NETWORK', 'residual-metric diagnostic (get_admm_residual_metrics)'),
    ('shared_resources_planning.py', '_write_operational_planning_main_info_per_operator'): (
        'd', 'NETWORK', 'Excel writer'),
    ('shared_resources_planning.py', '_write_interface_results_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('shared_resources_planning.py', '_write_shared_energy_storages_results_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('shared_resources_planning.py', '_write_network_voltage_results_per_operator'): ('d', 'NETWORK', 'Excel writer'),
    ('shared_resources_planning.py', '_write_network_consumption_results_per_operator'): ('d', 'NETWORK', 'Excel writer'),
    ('shared_resources_planning.py', '_write_network_generation_results_per_operator'): ('d', 'NETWORK', 'Excel writer'),
    ('shared_resources_planning.py', '_write_network_branch_results_per_operator'): ('d', 'NETWORK', 'Excel writer'),
    ('shared_resources_planning.py', '_write_network_branch_loading_results_per_operator'): ('d', 'NETWORK', 'Excel writer'),
    ('shared_resources_planning.py', '_write_network_power_flow_results_per_operator'): ('d', 'NETWORK', 'Excel writer'),
    ('shared_resources_planning.py', '_write_network_energy_storages_results_per_operator'): ('d', 'NETWORK', 'Excel writer'),
    # ---- network.py ----
    ('network.py', '__init__'): ('A', 'NETWORK', 'initializers list()'),
    ('network.py', '_build_model'): (
        'a', 'NETWORK', 'scenario index sets of every TSO/DSO block: scenarios_market = range(len(network.pm)), '
                        'scenarios_operation = range(len(network.po))'),
    ('network.py', '_read_network_operational_data_from_file'): (
        'd', 'NETWORK', 'defined, never called (no call site in any .py of the repository)'),
    ('network.py', '_process_results_summary_detail'): ('d', 'NETWORK', 'results processing (not on oracle path)'),
    ('network.py', '_compute_total_load'): ('d', 'NETWORK', 'results processing'),
    ('network.py', '_compute_total_generation'): ('d', 'NETWORK', 'results processing'),
    ('network.py', '_compute_conventional_generation'): ('d', 'NETWORK', 'results processing'),
    ('network.py', '_compute_renewable_generation'): ('d', 'NETWORK', 'results processing'),
    ('network.py', '_compute_losses'): ('d', 'NETWORK', 'results processing'),
    ('network.py', '_compute_generation_curtailment'): ('d', 'NETWORK', 'results processing'),
    ('network.py', '_compute_load_curtailment'): ('d', 'NETWORK', 'results processing'),
    ('network.py', '_compute_flexibility_used'): ('d', 'NETWORK', 'results processing'),
    # ---- network_data.py ----
    ('network_data.py', '_read_network_data'): (
        'A', 'NETWORK', 'network.prob_operation_scenarios = [1/num_oper_scenarios]*num_oper_scenarios'),
    ('network_data.py', '_write_main_info_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('network_data.py', '_write_market_cost_values_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('network_data.py', '_write_network_voltage_results_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('network_data.py', '_write_network_consumption_results_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('network_data.py', '_write_network_generation_results_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('network_data.py', '_write_network_branch_results_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('network_data.py', '_write_network_branch_loading_results_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('network_data.py', '_write_network_branch_power_flow_results_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('network_data.py', '_write_network_energy_storage_results_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    ('network_data.py', '_write_shared_network_energy_storage_results_to_excel'): ('d', 'NETWORK', 'Excel writer'),
    # ---- model_construction_helpers.py (TSO/DSO block objective and constraints) ----
    ('model_construction_helpers.py', 'interface_energy_settlement'): (
        'a', 'NETWORK', 'interface settlement objective term (weight 1 in ADMM); subtracted back out of Q(x)'),
    ('model_construction_helpers.py', 'total_generation_cost_rule'): ('a+b', 'NETWORK', 'objective aggregate; in gross via get_primal_value'),
    ('model_construction_helpers.py', 'total_flex_cost_rule'): ('a+b', 'NETWORK', 'objective aggregate; in gross'),
    ('model_construction_helpers.py', 'total_load_curtailment_cost_rule'): ('a+b', 'NETWORK', 'objective aggregate; in gross'),
    ('model_construction_helpers.py', 'total_gen_curtailment_penalty_rule'): ('a+b', 'NETWORK', 'objective aggregate (weight param 0 in ADMM); in gross'),
    ('model_construction_helpers.py', 'total_ess_utilization_cost_penalty_rule'): ('a+b', 'NETWORK', 'objective aggregate (shared weight 0 in ADMM); in gross'),
    ('model_construction_helpers.py', 'total_slack_penalties_rule'): ('a+b', 'NETWORK', 'objective aggregate; in gross'),
    ('model_construction_helpers.py', 'total_ess_complementarity_penalties_rule'): ('a+b', 'NETWORK', 'objective aggregate; in gross'),
    ('model_construction_helpers.py', 'total_load_curtailment_penalty_rule'): (
        'd', 'NETWORK', 'built only for obj_type CONGESTION_MANAGEMENT; all four SRP1 networks are COST'),
    ('model_construction_helpers.py', 'total_flex_penalty_rule'): (
        'd', 'NETWORK', 'built only for obj_type CONGESTION_MANAGEMENT; all four SRP1 networks are COST'),
    ('model_construction_helpers.py', 'dn_interface_expected_vmag_def'): ('a', 'NETWORK', 'expected-value constraint (DSO interface V)'),
    ('model_construction_helpers.py', 'dn_interface_expected_pf_p_def'): ('a', 'NETWORK', 'expected-value constraint (DSO interface P)'),
    ('model_construction_helpers.py', 'dn_interface_expected_pf_q_def'): ('a', 'NETWORK', 'expected-value constraint (DSO interface Q)'),
    ('model_construction_helpers.py', 'dn_interface_expected_sess_p_def'): ('a', 'NETWORK', 'expected shared-ESS P the ADMM exchanges with the ESSO (DSO)'),
    ('model_construction_helpers.py', 'dn_interface_expected_sess_q_def'): ('a', 'NETWORK', 'expected shared-ESS Q (DSO)'),
    ('model_construction_helpers.py', 'tn_interface_expected_vmag_def'): ('a', 'NETWORK', 'expected-value constraint (TSO interface V)'),
    ('model_construction_helpers.py', 'tn_interface_expected_pf_p_def'): ('a', 'NETWORK', 'expected-value constraint (TSO interface P)'),
    ('model_construction_helpers.py', 'tn_interface_expected_pf_q_def'): ('a', 'NETWORK', 'expected-value constraint (TSO interface Q)'),
    ('model_construction_helpers.py', 'tn_interface_expected_sess_p_def'): ('a', 'NETWORK', 'expected shared-ESS P the ADMM exchanges with the ESSO (TSO)'),
    ('model_construction_helpers.py', 'tn_interface_expected_sess_q_def'): ('a', 'NETWORK', 'expected shared-ESS Q (TSO)'),
    # ---- other production modules ----
    ('hierarchical_coordination.py', '_get_pq_map_vertices'): ('d', 'NETWORK', 'hierarchical PQ-map path only'),
    ('hierarchical_coordination.py', '_get_pq_initial_solution'): ('d', 'NETWORK', 'hierarchical PQ-map path only'),
    ('convex_oracle.py', 'block_objective'): ('d', 'NETWORK', 'module not in the oracle-path import closure'),
    ('shared_ess_price_taker.py', '_check_single_market_and_operation_scenario'): (
        'd', 'NETWORK', "price-taker initialization guard; SRP1_params.json admm.shared_ess_initialization = 'standalone'"),
    # ---- oracle-path harness modules ----
    ('p56a_oracle.py', 'investment_cost'): ('c', 'WORKBOOK', 'I(x) transcription of the master investment cost'),
    ('p515_g_g1_g4_admm_gates.py', '_s31_scenario_weighted'): ('e', 'NETWORK', 'component-level capture helper'),
}
# cost_investment sites: every one must pair cost_investment[..][s] with the WORKBOOK weights.
COST_INVESTMENT_RULES = {
    ('shared_energy_storage_data.py', '__init__'): 'A initializer',
    ('shared_energy_storage_data.py', '_read_shared_energy_storage_data_from_file'): 'A read from workbook, num_scenarios of the Scenarios sheet',
    ('shared_energy_storage_data.py', '_build_master_problem'): 'c paired with workbook omega_m (same loop index)',
    ('shared_energy_storage_data.py', '_get_expected_energy_investment_cost'): 'b(net, salvage) paired with workbook probability (same enumerate index)',
    ('shared_energy_storage_data.py', '_get_investment_cost_and_rated_capacity'): 'c paired with workbook omega_market',
    ('shared_resources_planning.py', '_check_candidate_first_stage_feasibility'): 'c paired with workbook probability',
    ('shared_resources_planning.py', '_build_positive_bootstrap_candidate'): 'c paired with workbook probability',
    ('p56a_oracle.py', 'investment_cost'): 'c paired with workbook probability',
}


# ======================================================================================
#  utilities
# ======================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args, binary=False):
    out = subprocess.run(['git'] + args, cwd=REPO, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, float) or isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return repr(obj)


def _preflight():
    ps = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True).stdout
    me = os.getpid()
    copies = []
    for line in ps.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or int(parts[0]) == me:
            continue
        cmd = parts[1]
        if ('p515_s45_probability_audit.py' in cmd and 'python' in cmd.split()[0]
                and '--manifest' not in cmd):
            copies.append(line.strip())
    if copies:
        raise RuntimeError(f'refusing to run concurrently with another copy of itself: {copies}')
    harness = [l.strip() for l in ps.splitlines()
               if re.search(r'p51\d\w*\.py|p514_\w*\.py', l) and 'p515_s45_probability_audit.py' not in l]
    lock = os.path.join(REPO, '.p515_s44_campaign.lock')
    lock_rec = None
    if os.path.exists(lock):
        with open(lock) as handle:     # read only; never touched
            lock_rec = handle.read()
    status = _git(['status', '--porcelain'])
    return {'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
            'git_status_porcelain_sha256': hashlib.sha256(status.encode()).hexdigest(),
            'git_status_porcelain_tracked_changes': [l for l in status.splitlines() if not l.startswith('??')],
            'git_status_porcelain_lines': len(status.splitlines()),
            '_status_lines': status.splitlines(),
            'campaign_lock_present': lock_rec is not None, 'campaign_lock_content': lock_rec,
            'other_harness_processes_running (authorized concurrency, Planner W3 task)': harness}


def _files_modified_since(t0):
    """Every file under the repo (outside .git and outside OUT_DIR) modified at/after t0 --
    recorded, not asserted: a concurrent campaign writes its own directories meanwhile."""
    out = []
    out_dir = os.path.abspath(OUT_DIR)
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d != '.git']
        if os.path.abspath(dirpath).startswith(out_dir):
            continue
        for f in filenames:
            p = os.path.join(dirpath, f)
            try:
                if os.stat(p).st_mtime >= t0:
                    out.append(os.path.relpath(p, REPO))
            except OSError:
                pass
    return sorted(out)


# ======================================================================================
#  S. static trace
# ======================================================================================
def _module_closure():
    root_modules = {f[:-3] for f in os.listdir(REPO) if f.endswith('.py')}
    seen, todo = set(), [s[:-3] for s in ORACLE_ENTRY_SCRIPTS]
    edges = {}
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        tree = ast.parse(open(os.path.join(REPO, mod + '.py')).read())
        deps = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                deps.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                deps.add(node.module.split('.')[0])
        deps &= root_modules
        edges[mod] = sorted(deps)
        todo.extend(deps - seen)
    return sorted(m + '.py' for m in seen), edges


def _function_spans(path):
    tree = ast.parse(open(path).read())
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, node.end_lineno, node.name))
    return spans


def _enclosing(spans, line):
    best = None
    for start, end, name in spans:
        if start <= line <= end and (best is None or start >= best[0]):
            best = (start, end, name)
    return best[2] if best else '<module>'


def _all_py_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d != '.git']
        for f in filenames:
            if f.endswith('.py'):
                out.append(os.path.relpath(os.path.join(dirpath, f), REPO))
    return sorted(out)


def static_trace():
    closure, edges = _module_closure()
    files = _all_py_files()
    root_files = [f for f in files if os.sep not in f]
    production = sorted(f for f in root_files if not HARNESS_RE.match(f))
    classified_scope = sorted(set(production) | {f for f in closure if f in root_files})

    search = {'primary': {}, 'broad_counts': {}, 'broad_lines_in_oracle_path_modules': {}}
    for key, pattern in PRIMARY_PATTERNS.items():
        rx = re.compile(pattern)
        search['primary'][key] = {}
        for f in files:
            lines = open(os.path.join(REPO, f), errors='replace').read().splitlines()
            hits = [i + 1 for i, l in enumerate(lines) if rx.search(l)]
            if hits:
                search['primary'][key][f] = hits
    for key, pattern in BROAD_PATTERNS.items():
        rx = re.compile(pattern)
        search['broad_counts'][key] = {}
        search['broad_lines_in_oracle_path_modules'][key] = {}
        for f in files:
            lines = open(os.path.join(REPO, f), errors='replace').read().splitlines()
            hits = [i + 1 for i, l in enumerate(lines) if rx.search(l)]
            if hits:
                search['broad_counts'][key][f] = len(hits)
                if f in closure or f in production:
                    search['broad_lines_in_oracle_path_modules'][key][f] = [
                        f'{n}: {lines[n - 1].strip()[:200]}' for n in hits]

    table, unruled, cost_table, cost_unruled = [], [], [], []
    used_rules, used_cost_rules = set(), set()
    for f in classified_scope:
        path = os.path.join(REPO, f)
        spans = _function_spans(path)
        lines = open(path).read().splitlines()
        for n, text in enumerate(lines, start=1):
            prob_hit = re.search(r'prob_market_scenarios|prob_operation_scenarios', text)
            cost_hit = re.search(r'cost_investment', text)
            if not (prob_hit or cost_hit):
                continue
            func = _enclosing(spans, n)
            if prob_hit:
                rule = RULES.get((f, func))
                if text.lstrip().startswith('#') or (text.lstrip().startswith(("'", '"', '`'))):
                    rule = ('doc', '-', 'comment/docstring text')
                entry = {'site': f'{f}:{n}', 'function': func, 'text': text.strip()[:220],
                         'reads_shared_ess_data_attr': bool(re.search(
                             r'(shared_ess_data|esso|sed)\.prob_market_scenarios', text)),
                         'on_oracle_path_module': f in closure}
                if rule is None:
                    unruled.append(entry)
                else:
                    if rule[0] != 'doc':
                        used_rules.add((f, func))
                    entry.update(classification=rule[0], source=rule[1], note=rule[2])
                    table.append(entry)
            if cost_hit:
                crule = COST_INVESTMENT_RULES.get((f, func))
                centry = {'site': f'{f}:{n}', 'function': func, 'text': text.strip()[:220]}
                if crule is None:
                    cost_unruled.append(centry)
                else:
                    used_cost_rules.add((f, func))
                    centry['classification'] = crule
                    cost_table.append(centry)
    stale = sorted(f'{k[0]}::{k[1]}' for k in RULES if k not in used_rules)
    stale_cost = sorted(f'{k[0]}::{k[1]}' for k in COST_INVESTMENT_RULES if k not in used_cost_rules)
    # the enclosing-function claims that carry the verdict, re-derived from the code
    srp_src = open(os.path.join(REPO, 'shared_resources_planning.py')).read()
    sed_src = open(os.path.join(REPO, 'shared_energy_storage_data.py')).read()
    tree = ast.parse(sed_src)
    fn = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    sub = ast.get_source_segment(sed_src, fn['_build_subproblem'])
    salvage = ast.get_source_segment(sed_src, fn['_build_terminal_salvage_value_expression'])
    structural = {
        'esso_subproblem_mentions_prob_or_scenario': bool(re.search(r'prob_|scenario', sub)),
        'esso_subproblem_objective_expr': re.search(r'model\.objective = pe\.Objective\((.*?)\)\n', sub, re.S).group(1).strip(),
        'esso_salvage_expression_calls_expected_energy_investment_cost':
            '_get_expected_energy_investment_cost' in salvage,
        'recourse_components_net_equals_gross_minus_salvage':
            'net_operational_recourse = gross_operational_cost - terminal_salvage_value' in srp_src,
        'read_network_operational_data_from_file_call_sites': sum(
            len(re.findall(r'_read_network_operational_data_from_file\(', open(os.path.join(REPO, f), errors='replace').read()))
            for f in files) - 1,
        'oracle_calls_run_operational_planning_with_print_results_False':
            'print_results=False' in open(os.path.join(REPO, 'p515_g_g1_g4_admm_gates.py')).read(),
    }
    return {'oracle_path_module_closure': closure, 'import_edges': edges,
            'production_modules': production, 'classified_scope': classified_scope,
            'all_py_files_searched': len(files), 'search': search,
            'read_site_table': table, 'unruled_primary_hits': unruled, 'stale_rules': stale,
            'cost_investment_table': cost_table, 'cost_investment_unruled': cost_unruled,
            'cost_investment_stale_rules': stale_cost, 'structural_checks': structural}


# ======================================================================================
#  workbook, read independently of production
# ======================================================================================
def workbook_scenarios():
    import pandas as pd
    out = {}
    blobs = {'working_copy': open(os.path.join(REPO, ESS_XLSX_REL), 'rb').read(),
             f'{PRIOR_COST_FILE_COMMIT}': _git(['show', f'{PRIOR_COST_FILE_COMMIT}:{ESS_XLSX_REL}'], binary=True)}
    for label, data in blobs.items():
        df = pd.read_excel(io.BytesIO(data), sheet_name='Scenarios', header=None)
        n = int(df.iloc[0, 1])
        probs = [float(df.iloc[0, i + 2]) for i in range(n)]
        cost = {}
        for sheet in ('Investment Cost, Power', 'Investment Cost, Energy'):
            dfc = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None)
            years = [int(v) for v in dfc.iloc[0, 1:]]
            cost[sheet] = {str(y): [float(dfc.iloc[i + 1, j + 1]) for i in range(n)]
                           for j, y in enumerate(years) if y in (2025, 2028, 2030, 2031, 2034, 2035, 2037)}
        out[label] = {'sha256': hashlib.sha256(data).hexdigest(),
                      'Scenarios_sheet_row0': [_jsonable(v) for v in df.iloc[0].tolist()],
                      'num_scenarios': n, 'probabilities': probs, 'sum': sum(probs),
                      'unit_costs_selected_years': cost}
    return out


# ======================================================================================
#  pyomo inspection helpers
# ======================================================================================
def _flatten_sum(e):
    name = type(e).__name__
    if name in ('SumExpression', 'LinearExpression', 'NPV_SumExpression'):
        for a in e.args:
            yield from _flatten_sum(a)
    else:
        yield e


def _is_named(x):
    f = getattr(x, 'is_named_expression_type', None)
    return bool(f and f())


def aggregate_weights(component, pe):
    """Weight on each scenario sub-Expression inside a total_* aggregate Expression."""
    from pyomo.core.expr.numvalue import native_numeric_types
    out, unparsed = [], []
    for t in _flatten_sum(component.expr):
        if type(t) in native_numeric_types:
            if t != 0:
                unparsed.append(f'constant {t!r}')
            continue
        if _is_named(t):
            out.append({'sub': t.parent_component().name, 'index': list(t.index()), 'weight': 1.0})
            continue
        args = getattr(t, 'args', ())
        if len(args) == 2:
            done = False
            for num, named in ((args[0], args[1]), (args[1], args[0])):
                is_num = type(num) in native_numeric_types or (hasattr(num, 'is_constant') and num.is_constant())
                if is_num and _is_named(named):
                    out.append({'sub': named.parent_component().name, 'index': list(named.index()),
                                'weight': float(pe.value(num))})
                    done = True
                    break
            if done:
                continue
        unparsed.append(type(t).__name__)
    return out, unparsed


def block_probability_audit(model, network, params, pe, repn_fn, srp):
    """Probabilities on the network object + the weights actually built into the block."""
    import model_construction_helpers as mch
    rec = {'prob_market_scenarios': list(network.prob_market_scenarios),
           'prob_operation_scenarios': list(network.prob_operation_scenarios),
           'len_scenarios_market_set': len(model.scenarios_market),
           'len_scenarios_operation_set': len(model.scenarios_operation),
           'has_scenario_deviation_penalty': hasattr(model, 'scenario_deviation_penalty'),
           'obj_type': params.obj_type}
    aggs = {}
    for name in ('total_gen_cost', 'total_flex_cost', 'total_load_curt_cost', 'total_gen_curt_penalty',
                 'total_ess_utilization_cost_penalty', 'total_slack_penalties',
                 'total_ess_complementarity_penalties', 'total_load_curt_penalty', 'total_flex_penalty'):
        comp = getattr(model, name, None)
        if comp is None:
            aggs[name] = 'not built'
            continue
        weights, unparsed = aggregate_weights(comp, pe)
        aggs[name] = {'weights': weights, 'unparsed_terms': unparsed}
    rec['aggregate_weights'] = aggs
    # implied weight from the objective's linear coefficients: pg of cost-bearing generators
    repn = repn_fn(model.objective.expr, compute_values=True, quadratic=False)
    implied_pg = set()
    cost_gens = {g for g in model.generators
                 if network.generators[g].is_controllable()
                 and not (not network.is_transmission and network.generators[g].gen_type == mch.GEN_REFERENCE)}
    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        comp = var.parent_component().name
        idx = var.index()
        if comp == 'pg' and idx[0] in cost_gens:
            s_m, p = idx[1], idx[3]
            c = network.cost_energy_p[s_m][p] * network.baseMVA
            if c != 0.0:
                implied_pg.add(round(float(coef) / float(c), 12))
    rec['implied_weight_from_objective_coef_pg_over_cp_baseMVA'] = sorted(implied_pg)
    rec['interface_settlement_weight'] = float(pe.value(model.interface_settlement_weight))
    # expected-value constraints: -coef(var)/coef(expected var), per variable component, with
    # the variable's index (the scenario position is visible in the index)
    implied_cons = {}
    for con in model.component_objects(pe.Constraint, active=True):
        vals = {}
        for cd in con.values():
            if not cd.active:
                continue
            r = repn_fn(cd.body, compute_values=True, quadratic=False)
            exp_vars = [(v, c) for v, c in zip(r.linear_vars, r.linear_coefs)
                        if v.parent_component().name.startswith('expected_')]
            if len(exp_vars) != 1 or not r.is_linear():
                break
            ce = float(exp_vars[0][1])
            for v, c in zip(r.linear_vars, r.linear_coefs):
                if v.parent_component().name.startswith('expected_'):
                    continue
                vals.setdefault(v.parent_component().name, set()).add(round(-float(c) / ce, 12))
        if vals:
            implied_cons[con.name] = {k: sorted(v) for k, v in vals.items()}
    rec['implied_weight_in_expected_value_constraints'] = implied_cons
    return rec


# ======================================================================================
#  N. SRP1 numerical check
# ======================================================================================
class _Interceptor:
    def __init__(self):
        self.calls = []

    def network(self, holder, kind):
        def _stub(model, *args, **kwargs):
            self.calls.append(kind)
            return {year: {day: None for day in holder.days} for year in holder.years}
        return _stub

    def esso(self, sed):
        def _stub(models, *args, **kwargs):
            self.calls.append('esso_coordination' if kwargs.get('cycle') is not None else 'esso_init')
            return {node_id: None for node_id in sed.active_distribution_network_nodes}
        return _stub

    def counts(self):
        out = {k: 0 for k in DECLARED_INTERCEPTS}
        for c in self.calls:
            out[c] = out.get(c, 0) + 1
        return out


def _probability_objects(planning):
    sed = planning.shared_ess_data
    rec = {'planning.num_market_scenarios': planning.num_market_scenarios,
           'planning.prob_market_scenarios': {str(y): list(v) for y, v in planning.prob_market_scenarios.items()},
           'shared_ess_data.prob_market_scenarios': list(sed.prob_market_scenarios),
           'shared_ess_data.prob_market_scenarios_type': type(sed.prob_market_scenarios).__name__,
           'shared_ess_data.prob_market_scenarios_is_planning_dict': sed.prob_market_scenarios is planning.prob_market_scenarios,
           'shared_ess_data.cost_investment_num_scenarios': {k: len(v) for k, v in sed.cost_investment.items()},
           'networks': {}}
    holders = [('TSO', planning.transmission_network)] + [
        (f'DSO{n}', planning.distribution_networks[n]) for n in sorted(planning.distribution_networks)]
    for label, holder in holders:
        per = {}
        for year in holder.years:
            for day in holder.days:
                net = holder.network[year][day]
                per[f'{year}/{day}'] = {
                    'prob_market_scenarios': list(net.prob_market_scenarios),
                    'prob_operation_scenarios': list(net.prob_operation_scenarios),
                    'pm_is_planning_prob_market_scenarios[year] (same list object)':
                        net.prob_market_scenarios is planning.prob_market_scenarios[year],
                    'num_market_price_rows (cost_energy_p)': len(net.cost_energy_p),
                    'combination_weights_pm_x_po_sum': sum(a * b for a in net.prob_market_scenarios
                                                           for b in net.prob_operation_scenarios)}
        rec['networks'][label] = {'name': holder.name, 'obj_type': holder.params.obj_type,
                                  'num_oper_scenarios': holder.num_oper_scenarios, 'blocks': per}
    return rec


def srp1_numerical(stdout_sink):
    import pyomo.environ as pe
    from pyomo.repn import generate_standard_repn
    import shared_resources_planning as srp
    import shared_energy_storage_data as SED
    import p56a_oracle as O
    import p515_g_g1_g4_admm_gates as G
    from shared_resources_planning import SharedResourcesPlanning

    steps = {}
    read_dir = os.path.join(OUT_DIR, 'srp1_read')
    planning0 = SharedResourcesPlanning(DATA_DIR, 'SRP1.json')
    planning0.results_dir = os.path.join(read_dir, 'Results')
    planning0.diagrams_dir = os.path.join(read_dir, 'Diagrams')
    planning0.logs_dir = os.path.join(planning0.results_dir, 'Logs')
    with redirect_stdout(stdout_sink):
        planning0.read_planning_problem()
    checksum = planning0.scenario_metadata['combined_scenario_checksum']
    if checksum != O.CANONICAL_CHECKSUM:
        raise RuntimeError(f'SRP1 scenario checksum {checksum} != canonical {O.CANONICAL_CHECKSUM}')
    if O._BASELINE is not None:
        raise RuntimeError('oracle baseline already loaded; refusing to inject')
    O._BASELINE = {'planning': planning0, 'checksum': checksum}
    O.WORK_DIR = os.path.join(OUT_DIR, 'evals')
    steps['read'] = {'case': 'data/SRP1/SRP1.json', 'case_sha256': _sha256_file(os.path.join(DATA_DIR, 'SRP1.json')),
                     'scenario_checksum': checksum, 'equals_canonical': True,
                     'substitution_declared': ('production reader on the canonical case with results/diagrams/'
                                               'logs dirs redirected into the output dir, injected as '
                                               'p56a_oracle._BASELINE (p515_s44_scale_measurement.py '
                                               'precedent); O.WORK_DIR redirected into the output dir')}

    report = {}
    with redirect_stdout(stdout_sink):
        planning, sed, cand = G._construct_arm_planning(
            'p515s45_prob_audit', os.path.join(OUT_DIR, 'arm_construct'), report,
            investment_map=C_STAR, eval_id=EVAL_ID, num_max_iters_override=500, apply_rho=False)
    steps['construct'] = {'instance': report.get('instance'), 'candidate_label': C_STAR_LABEL,
                          'candidate_investment': _jsonable(cand['investment']),
                          'candidate_sha256': hashlib.sha256(json.dumps(_jsonable(cand['investment']),
                                                                        sort_keys=True).encode()).hexdigest(),
                          'apply_rho': False, 'budget_in_force': sed.params.budget}
    probs = _probability_objects(planning)

    # investment side (off Q): I(x), expected unit costs (production helper vs independent)
    years = list(sed.years)
    inv = {'oracle_investment_cost_I_x_at_C_star': O.investment_cost(planning, cand),
           'first_stage_feasibility': list(O.check_master_feasibility(planning, cand)),
           'expected_energy_unit_cost_production_helper': {
               str(y): SED._get_expected_energy_investment_cost(sed, y) for y in years},
           'expected_energy_unit_cost_independent': {
               str(y): sum(p * sed.cost_investment['energy'][s][y]
                           for s, p in enumerate(sed.prob_market_scenarios)) for y in years}}

    interceptor = _Interceptor()
    tn = planning.transmission_network
    tn.optimize = interceptor.network(tn, 'tso')
    for dn in planning.distribution_networks.values():
        dn.optimize = interceptor.network(dn, 'dso')
    sed.optimize = interceptor.esso(sed)
    with redirect_stdout(stdout_sink):
        cv, _dv = srp.create_admm_variables(planning)
        dso_models, _ = srp.create_distribution_networks_models(
            planning.distribution_networks, cv, cand['total_capacity'],
            parallel_execution=planning.parallel_execution)
        tso_model, _ = srp.create_transmission_network_model(planning, cv, cand['total_capacity'])
        esso_models, _ = srp.create_shared_energy_storage_model(sed, cv, cand['investment'])
    intercepts = {'declared': DECLARED_INTERCEPTS, 'observed': interceptor.counts()}
    intercepts['exact_match'] = intercepts['observed'] == DECLARED_INTERCEPTS
    if not intercepts['exact_match']:
        raise RuntimeError(f'intercept count mismatch: {intercepts}')

    def all_blocks():
        for year in tn.years:
            for day in tn.days:
                yield 'TSO', tn, tso_model[year][day], year, day
        for node in sorted(planning.distribution_networks):
            dn = planning.distribution_networks[node]
            for year in dn.years:
                for day in dn.days:
                    yield f'DSO{node}', dn, dso_models[node][year][day], year, day

    blocks_before = {}
    for label, holder, model, year, day in all_blocks():
        blocks_before[f'{label}/{year}/{day}'] = block_probability_audit(
            model, holder.network[year][day], holder.params, pe, generate_standard_repn, srp)
    with redirect_stdout(stdout_sink):
        srp._prepare_distribution_objectives_for_admm(planning.distribution_networks, dso_models)
        srp._prepare_transmission_objectives_for_admm(tn, tso_model)
    blocks_after = {}
    for label, holder, model, year, day in all_blocks():
        rec = block_probability_audit(model, holder.network[year][day], holder.params, pe,
                                      generate_standard_repn, srp)
        rec['q_aggregation_block_weight (_get_admm_block_weight = N_y x N_d x annualization)'] = \
            srp._get_admm_block_weight(holder, year, day)
        blocks_after[f'{label}/{year}/{day}'] = rec

    # ESSO
    params = planning.params.admm
    esso = {}
    scale = params.objective_scale
    with redirect_stdout(stdout_sink):
        al_scale = srp._resolve_esso_al_scale(planning, params, scale)[0] if scale is not None else None
    import definitions as DEF
    esso_constants = {'PENALTY_ESSO_SLACK': DEF.PENALTY_ESSO_SLACK, 'EPS_ESSO_THROUGHPUT': DEF.EPS_ESSO_THROUGHPUT}
    salvage_factors = {}
    idx0 = sed.get_shared_energy_storage_idx(sed.active_distribution_network_nodes[0])
    for y_inv, year_inv in enumerate(years):
        ess = sed.shared_energy_storages[year_inv][idx0]
        age, remaining, frac = SED._get_remaining_calendar_life(sed, y_inv, ess)
        salvage_factors[str(year_inv)] = {
            'age_at_terminal_years': age, 'remaining_life_years': remaining,
            'remaining_life_fraction': frac,
            'expected_energy_unit_cost_workbook_weighted': SED._get_expected_energy_investment_cost(sed, year_inv),
            'terminal_discount': SED._get_terminal_discount_factor(sed),
            'salvage_params': {'enabled': sed.params.salvage_value.enabled,
                               'energy_recovery_fraction': sed.params.salvage_value.energy_recovery_fraction,
                               'recycling_floor_fraction': sed.params.salvage_value.recycling_floor_fraction},
            'min_soh': ess.soh_min}
    for node in sed.active_distribution_network_nodes:
        m = esso_models[node]
        scen_components = sorted(c.name for c in m.component_objects(descend_into=True)
                                 if 'scenario' in c.name.lower() or 'prob' in c.name.lower())
        r = generate_standard_repn(m.objective.expr, compute_values=True, quadratic=False)
        sv = generate_standard_repn(m.salvage_value.expr, compute_values=True, quadratic=False)
        sv_coefs = {}
        for v, c in zip(sv.linear_vars, sv.linear_coefs):
            sv_coefs.setdefault(v.parent_component().name, []).append([list(v.index()), float(c)])
        esso[str(node)] = {
            'has_scenarios_market_attr': hasattr(m, 'scenarios_market'),
            'components_named_like_scenario_or_prob': scen_components,
            'base_objective_is_linear': r.is_linear(),
            'base_objective_distinct_linear_coefficients': sorted({round(float(c), 15) for c in r.linear_coefs}),
            'base_objective_variable_components': sorted({v.parent_component().name for v in r.linear_vars}),
            'base_objective_variable_index_arities': sorted({len(v.index()) if isinstance(v.index(), tuple) else 1
                                                             for v in r.linear_vars}),
            'salvage_value_linear_coefficients (net recourse only; not an objective term)': sv_coefs,
            'salvage_value_constant': float(sv.constant) if sv.constant is not None else None,
            'salvage_factors_per_cohort': salvage_factors,
        }
    esso_admm = {}
    if al_scale is not None:
        with redirect_stdout(stdout_sink):
            srp.update_shared_energy_storage_model_to_admm(planning, esso_models, params, al_scale_esso=al_scale)
        from pyomo.core.expr.visitor import identify_variables, identify_mutable_parameters
        for node in sed.active_distribution_network_nodes:
            m = esso_models[node]
            e = m.admm_objective.expr
            esso_admm[str(node)] = {
                'objective_active': m.objective.active, 'admm_objective_active': m.admm_objective.active,
                'variable_components': sorted({v.parent_component().name for v in identify_variables(e)}),
                'mutable_param_components': sorted({p.parent_component().name for p in identify_mutable_parameters(e)}),
                'al_scale_esso': al_scale}
    return {'steps': steps, 'interceptor_check': intercepts, 'probability_objects': probs,
            'investment_side': inv, 'network_blocks_before_admm_prepare': blocks_before,
            'network_blocks_after_admm_prepare': blocks_after, 'esso_base': esso,
            'esso_admm_objective': esso_admm, 'objective_scale_used_for_esso_al': scale,
            'esso_objective_constants_definitions_py': esso_constants}


# ======================================================================================
#  P. paper scale, data only
# ======================================================================================
def paper_data_only(stdout_sink):
    import network as network_module
    import network_data as network_data_module
    import shared_energy_storage_data as SED
    from shared_resources_planning import SharedResourcesPlanning

    blocked = []
    originals = {}

    def blocker(name):
        def _raise(*a, **k):
            blocked.append(name)
            raise RuntimeError(f'paper-scale phase: model construction blocked ({name})')
        return _raise
    targets = [(network_module.Network, 'build_model'), (network_data_module.NetworkData, 'build_model'),
               (SED.SharedEnergyStorageData, 'build_subproblem'),
               (SED.SharedEnergyStorageData, 'build_master_problem')]
    for cls, attr in targets:
        originals[(cls, attr)] = getattr(cls, attr)
        setattr(cls, attr, blocker(f'{cls.__name__}.{attr}'))
    try:
        case_path = os.path.join(REPO, PAPER_CASE_REL)
        rel = os.path.relpath(case_path, DATA_DIR)
        planning = SharedResourcesPlanning(DATA_DIR, rel)
        planning.name = 'SRP1'
        read_dir = os.path.join(OUT_DIR, 'paper_read')
        planning.results_dir = os.path.join(read_dir, 'Results')
        planning.diagrams_dir = os.path.join(read_dir, 'Diagrams')
        planning.logs_dir = os.path.join(planning.results_dir, 'Logs')
        with redirect_stdout(stdout_sink):
            planning.read_planning_problem()
    finally:
        for (cls, attr), fn in originals.items():
            setattr(cls, attr, fn)
    checksum = planning.scenario_metadata['combined_scenario_checksum']
    probs = _probability_objects(planning)
    sed = planning.shared_ess_data
    nm = planning.num_market_scenarios
    wb = list(sed.prob_market_scenarios)
    counterfactual = {
        'what_each_operational_combination_gets (network pm x po)': 1.0 / (nm * probs['networks']['TSO']['num_oper_scenarios']),
        'what_market_scenario_s_m_would_get_if_the_workbook_vector_were_used': {
            str(s): (wb[s] if s < len(wb) else 'IndexError: workbook vector has only %d entries' % len(wb))
            for s in range(nm)},
        'network_market_weight': 1.0 / nm,
        'max_abs_weight_difference_on_existing_indices': max(abs(wb[s] - 1.0 / nm) for s in range(min(nm, len(wb)))),
        'note': ('counterfactual only -- no site found by the static trace uses the workbook vector on an '
                 'operational term; shown to size what such a defect WOULD mean'),
    }
    salvage = {}
    years = list(sed.years)
    idx0 = sed.get_shared_energy_storage_idx(sed.active_distribution_network_nodes[0])
    for y_inv, year_inv in enumerate(years):
        ess = sed.shared_energy_storages[year_inv][idx0]
        age, remaining, frac = SED._get_remaining_calendar_life(sed, y_inv, ess)
        salvage[str(year_inv)] = {'age_at_terminal_years': age, 'remaining_life_fraction': frac,
                                  'expected_energy_unit_cost_workbook_weighted':
                                      SED._get_expected_energy_investment_cost(sed, year_inv)}
    return {'case': PAPER_CASE_REL, 'case_sha256': _sha256_file(os.path.join(REPO, PAPER_CASE_REL)),
            'scenario_checksum': checksum,
            'scenario_checksum_equals_scale_measurement_record': checksum == PAPER_CHECKSUM_RECORDED,
            'scale_measurement_checksum_source': PAPER_CHECKSUM_SOURCE,
            'model_construction_blockers': [f'{c.__name__}.{a}' for c, a in targets],
            'blocked_calls': blocked, 'probability_objects': probs, 'counterfactual': counterfactual,
            'salvage_factors_per_cohort (data-level; net recourse only)': salvage,
            'terminal_discount': SED._get_terminal_discount_factor(sed)}


# ======================================================================================
#  verdict (computed from the evidence above)
# ======================================================================================
def verdict(static, srp1, paper):
    table = static['read_site_table']
    workbook_operational = [r for r in table if r['source'] == 'WORKBOOK' and r['classification'] in ('a', 'a+b')]
    workbook_gross = [r for r in table if r['source'] == 'WORKBOOK' and 'b' in r['classification']]
    # SRP1 numerics
    wb = srp1['probability_objects']['shared_ess_data.prob_market_scenarios']
    net_weights = set()
    for rec in srp1['network_blocks_after_admm_prepare'].values():
        for agg in rec['aggregate_weights'].values():
            if isinstance(agg, dict):
                net_weights.update(abs(w['weight']) for w in agg['weights'])
        net_weights.update(abs(v) for v in rec['implied_weight_from_objective_coef_pg_over_cp_baseMVA'])
        for per_var in rec['implied_weight_in_expected_value_constraints'].values():
            for vals in per_var.values():
                net_weights.update(abs(v) for v in vals)
    esso_coefs = set()
    for rec in srp1['esso_base'].values():
        esso_coefs.update(rec['base_objective_distinct_linear_coefficients'])
    return {
        'static_workbook_vector_on_operational_term_sites': [r['site'] for r in workbook_operational],
        'static_workbook_vector_in_Q_sites': [r['site'] for r in workbook_gross],
        'srp1_workbook_vector': wb,
        'srp1_network_abs_weights_found_in_built_models (signed values per block in network_blocks_*; '
        'a -1 is the structural sign of shared_es_pnet in the DSO pg_adn expression times weight 1)': sorted(net_weights),
        'srp1_any_workbook_probability_in_network_weights': any(
            abs(w - p) < 1e-12 for w in net_weights for p in wb if abs(p - 1.0) > 1e-12),
        'srp1_esso_base_objective_coefficients': sorted(esso_coefs),
        'srp1_any_workbook_probability_in_esso_objective_coefficients': any(
            abs(c - p) < 1e-15 for c in esso_coefs for p in wb),
        'srp1_overwrite_visible_at_srp1': wb != [1.0],
        'paper_network_pm': sorted({tuple(b['prob_market_scenarios']) for n in paper['probability_objects']['networks'].values()
                                    for b in n['blocks'].values()}),
        'paper_network_po': sorted({tuple(b['prob_operation_scenarios']) for n in paper['probability_objects']['networks'].values()
                                    for b in n['blocks'].values()}),
        'paper_workbook_vector': paper['probability_objects']['shared_ess_data.prob_market_scenarios'],
    }


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
            entries[rel] = _sha256_file(p)
    entries_script = {'p515_s45_probability_audit.py': _sha256_file(os.path.abspath(__file__))}
    path = os.path.join(OUT_DIR, MANIFEST_NAME)
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    with open(path, 'w') as handle:
        json.dump({'created_utc': _utc(), 'directory': os.path.relpath(OUT_DIR, REPO),
                   'files': dict(sorted(entries.items())), 'script': entries_script}, handle, indent=2)
    print(f'[manifest] {len(entries)} files -> {path}')


def main():
    if '--manifest' in sys.argv:
        write_manifest()
        return 0
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, AUDIT_JSON)
    if os.path.exists(out_path):
        raise RuntimeError(f'write-once: {out_path} exists')
    started = _utc()
    import time as _time
    t0 = _time.time()
    pre = _preflight()
    print(f'[{STAGE}] start {started}; HEAD {pre["git_HEAD"]}')
    guard = SolveProfileGuard(permitted=(), label=STAGE).install()
    result = {'stage': STAGE, 'started_utc': started, 'script_sha256': _sha256_file(os.path.abspath(__file__)),
              'preflight': pre}
    ok = True
    sink_path = os.path.join(OUT_DIR, STDOUT_CAPTURE)
    try:
        with open(sink_path, 'w') as sink:
            for name, fn in (('static_trace', static_trace), ('workbook_independent_read', workbook_scenarios)):
                print(f'[{name}] ...')
                try:
                    result[name] = fn()
                except Exception as error:
                    ok = False
                    result[name] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            st = result.get('static_trace', {})
            if st.get('unruled_primary_hits') or st.get('stale_rules') or st.get('cost_investment_unruled') \
                    or st.get('cost_investment_stale_rules'):
                ok = False
                print('[static_trace] FAIL: table incomplete or stale', file=sys.stderr)
            for name, fn in (('srp1_numerical', srp1_numerical), ('paper_scale_data_only', paper_data_only)):
                print(f'[{name}] ...')
                try:
                    result[name] = fn(sink)
                except Exception as error:
                    ok = False
                    result[name] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
                    print(traceback.format_exc(), file=sys.stderr)
    finally:
        guard.uninstall()
    failures = guard.verify(0)
    result['solve_profile_guard'] = {'permitted': [], 'counts': guard.counts, 'verify_0_failures': failures}
    if failures:
        ok = False
    if ok:
        try:
            result['verdict_inputs'] = verdict(result['static_trace'], result['srp1_numerical'],
                                               result['paper_scale_data_only'])
        except Exception as error:
            ok = False
            result['verdict_inputs'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    post_status = _git(['status', '--porcelain'])
    pre_lines = set(pre.pop('_status_lines'))
    post_lines = set(post_status.splitlines())
    result['post_git_status_porcelain_sha256'] = hashlib.sha256(post_status.encode()).hexdigest()
    result['post_git_status_tracked_changes'] = [l for l in post_status.splitlines() if not l.startswith('??')]
    result['git_status_lines_added_during_run'] = sorted(post_lines - pre_lines)
    result['git_status_lines_removed_during_run'] = sorted(pre_lines - post_lines)
    result['files_modified_outside_out_dir_during_run (includes the concurrent campaign)'] = \
        _files_modified_since(t0)
    result['ended_utc'] = _utc()
    result['ok'] = ok
    with open(out_path, 'w') as handle:
        json.dump(_jsonable(result), handle, indent=1)
    print(f'[{STAGE}] ok={ok}; guard counts {guard.counts}; wrote {out_path}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
