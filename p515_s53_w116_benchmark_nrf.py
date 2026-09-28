"""
P5.15 Addendum 57 Decision 1 (Planner task W116) -- the UNCOORDINATED BENCHMARK, SPEC v3: the NO-REVERSE-FLOW (NRF) arms,
the report-only FEASIBILITY SWEEP of the unconstrained (spec-v2) arms, and the v3 report. Stage harness.

WRITTEN AND CHECKED WITHOUT ANY SOLVE (W116: code, zero-solve checks and freeze; NO benchmark stage is run by W116).
The spec, its zero-solve checks and the v2 -> v3 key diff are `p515_s53_w116_benchmark_spec_v3.py`.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 57 Decision 1 ((a) the sweep, report-only; (b) every uncoordinated arm
under p_int >= 0 at each interface, the claim against the best NRF arm; the zero-solve reverse-flow count of the
coordinated Q181 solution beside it), Addendum 49 (everything else as it stands: three starts, the consistency
re-evaluation, the passive tie-breakers 0.1 / 10, curtailment per arm) and Addendum 56 (net curtailment the frozen
primary; positive / negative parts and raw MWh beside it).

INSTANCE (every artifact): SRP1, x = 0, candidate key 8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57;
COORDINATED = the settled cycle-181 cell d110bd1a5977df1e_x0 (certified_models.pkl sha256 99ab1070..., Q181 =
653,873,702.1876609), not re-run -- `p515_s53_w93_uncoordinated_benchmark.COORDINATED`, imported unchanged.

OBJECTIVE CONVENTION (every Q): gross_operational_cost, settlement EXCLUDED (production
`_get_operational_recourse_components`), evaluation tie-breaker 0 in every arm, salvage reported separately -- the v2
convention (`uncoordinated_benchmark.evaluate_common_q`).

THE NRF DEFINITION (uncoordinated_benchmark.NO_REVERSE_FLOW_ROW; sign and source lines there): every DSO arm block carries
    uncoord_no_reverse_flow[s_m, s_o, p]:  pg_adn[s_m, s_o, p] >= 0     (import only: TN -> DN)
built by `build_dso_arm_models(no_reverse_flow=True)`. Passive keeps flexibility at 0 and minimum curtailment (decision
tie-breaker 1 EUR/MWh, value-independence at 0.1 / 10); price-taker keeps its own objective (tie-breaker 0). The TSO arm
is v2's, unchanged: interface P/Q fixed to the DSO schedule, voltages bounded (`build_tso_arm_model`, not edited).

STAGES (each its own attached process; output write-once under data/SRP1/Results/P515S53/w116_benchmark_nrf/<run_id>/):
  sweep --arm A              REPORT-ONLY. The spec-v2 (unconstrained) arm A, COLD start only: the 36 DSO blocks, then
                             every one of the 12 TSO blocks at the DSO schedule, CONTINUING past a failing TSO block.
                             For each failing block ONE elastic E3 solve (p515_s53_w114_passive_infeasibility, imported:
                             `build_elastic_copy(block, 'rows', ('uncoord_interface_p_fixed',))` + `elastic_solve`,
                             interface P elastic, Q and physics hard): per block and hour whether the TN can accept the DN
                             schedule and, where not, by how much (|accepted - target| per node, MW).
                             SOLVE ACCOUNTING (cannot be exact in advance: how many blocks fail is what the sweep
                             measures). Declared: an UPPER BOUND, 36 x 3 + 12 x (3 + 1) = 156 launches per arm (3 =
                             production's attempts per network solve: primary, recovery, tier 2 -- network._run_smopf; 1 =
                             the E3 launch of a failing TSO block), and an EXACT PER-BLOCK count: before and after every
                             block the armed guard's delta must equal the attempts production itself recorded for that
                             block (`_drain_network_ipopt_solve_records`, one record per `_run_smopf_solver_attempt`),
                             1 <= attempts <= 3, and exactly 1 for an E3 launch; the cumulative total is verified exactly
                             after every block (too few fails as loudly as too many). A DSO block that fails every tier
                             ends the sweep (no DSO schedule exists to test).
  nrf-arm --arm A --start S  The NRF arm: 48 solves (36 DSO + 12 TSO), then the consistency re-evaluation 36 -> 84; the
                             declared sequential pass adds 48 -> 132 if triggered. Counts exact at every phase boundary as
                             under spec v2 (a production retry raises the count and fails the stage -- v2's convention,
                             kept). A in {passive, price_taker}, S in {cold, warm_from_certified, perturbed}.
  nrf-passive-tie-breaker --value V   48 solves (passive NRF, cold, DSO tie-breaker V in {0.1, 10}); no consistency step.
  report                     ZERO solves. claim = min(passive_NRF, price_taker_NRF) - Q181 with the decomposition and the
                             bands; curtailment per arm (net primary, positive / negative parts, raw MWh); the sweep's
                             "the TN cannot accept the DNs' exchange in n of 12 blocks (h hours)" per arm; the coordinated
                             reverse-flow count (from the W116 zero-solve checks); the frozen predictions against their
                             outcomes. Reads spec v2's lambda-look / common-Q gate / coupling check outputs READ-ONLY (sha
                             verified; Addendum 57: those results stand).

Guards: `SolveProfileGuard` armed BEFORE any production import; permitted sites: nrf-arm / variant
(('uncoordinated_benchmark.py', '_solve_block'),); sweep adds (p515_s53_w114_passive_infeasibility.py, 'elastic_solve');
report (). W114's module installs its own guard at import; the sweep imports it AFTER arming its own and uninstalls
W114's at once (LIFO), recording that. Each run refuses unless: no campaign / legacy / W93 / W116 lock, no forbidden live
process (the p515_s4* campaign children, the G1-G4 gates, any other p515_s53_* process), production and these files clean
in git, its output directory absent, its IPOPT log dir absent, and the frozen spec v3 binds (`frozen_spec_binding_failures`).

COMMANDS: the frozen spec v3 lists them (attached, alone, one at a time, both streams to a new log, noclobber):
    mkdir -p data/SRP1/Results/P515S53/w116_benchmark_nrf/launch_logs
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w116_benchmark_nrf.py --stage sweep --arm passive > data/SRP1/Results/P515S53/w116_benchmark_nrf/launch_logs/sweep_passive_cold.log 2>&1
    ... (--stage sweep --arm price_taker; --stage nrf-arm --arm A --start S; --stage nrf-passive-tie-breaker --value V;
         --stage report; log name = <run_id>.log)
Exit 0 = completed with every gate of the stage passed; 1 = a gate or a solve failed; 2 = refused (precondition).
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s44_campaign_harness as H  # noqa: E402 -- standard library only at import
import gate_result_io as GRIO  # noqa: E402 -- the one gate-result writer (W100)
import p515_s53_w93_uncoordinated_benchmark as BENCH  # noqa: E402 -- stdlib + H + GRIO at import; no guard

# ======================================================================================================================
#  FROZEN CONFIGURATION (declared before any run; recorded in every artifact and in the frozen spec v3)
# ======================================================================================================================
STAGE = ('P5.15 Addendum 57 W116 -- uncoordinated benchmark spec v3: no-reverse-flow arms, the feasibility sweep of the '
         'unconstrained arms, the coordinated reverse-flow count')
AUTHORITY = BENCH.AUTHORITY + [
    'PLANNER_BRIEF_2026-09-13.md Addendum 56 (curtailment reporting: net primary, positive / negative parts, raw MWh)',
    'PLANNER_BRIEF_2026-09-13.md Addendum 57 Decision 1 ((a) sweep, report-only; (b) no-reverse-flow arms, claim vs the '
    'best NRF arm; zero-solve reverse-flow count of the coordinated Q181 solution)',
    'TASKS.md Addendum 57 order (W116)', 'Planner task W116 (benchmark spec v3)']
SCRIPT_NAME = os.path.basename(__file__)
OUT_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w116_benchmark_nrf')
LAUNCH_LOGS_REL = os.path.join(OUT_ROOT_REL, 'launch_logs')
CHECKS_DIR_REL = os.path.join(OUT_ROOT_REL, 'w116_zero_solve_checks')
REVERSE_FLOW_OUTPUT_REL = os.path.join(CHECKS_DIR_REL, 'reverse_flow_count_q181.json')
FROZEN_SPEC_GLOB = 'frozen_s53_benchmark_spec_v*_*.json'
FROZEN_SPEC_MIN_VERSION = 3
LOCK_PATH = os.path.join(REPO, '.p515_s53_w116_benchmark.lock')
EVAL_ID_PREFIX = 'p515s53w116_'       # isolated IPOPT log dir: p56a_oracle.WORK_DIR/<EVAL_ID_PREFIX><run_id>/logs
W114_SCRIPT = 'p515_s53_w114_passive_infeasibility.py'
EXTRA_CLEAN_FILES = BENCH.EXTRA_CLEAN_FILES + (SCRIPT_NAME, W114_SCRIPT)
FROZEN_SPEC_BOUND_FILES = BENCH.FROZEN_SPEC_BOUND_FILES + (SCRIPT_NAME, W114_SCRIPT)

# spec v2 (predecessor) and its stage outputs that v3 reads READ-ONLY (Addendum 57: "The lambda-look results stand as
# recorded ... common-Q gate bitwise"); committed in 2ee7a124 (W113), sha256 from the committed W113 evidence manifest.
V2_SPEC = {'path': os.path.join(BENCH.OUT_ROOT_REL, 'frozen_s53_benchmark_spec_v2_bb659122.json'),
           'sha256': 'bb659122af30b70a878c039c63d6458bf2c87d2a8971222f76f5c7cbb7576112',
           'committed_in': '59c3b995077164dd8fa0bce00bcb4527e3377063', 'version': 2}
V2_STAGE_OUTPUTS = {
    'lambda_look': {'path': os.path.join(BENCH.OUT_ROOT_REL, 'lambda_look', 'lambda_look.json'),
                    'sha256': 'a80c473835f24d1dd6e3be183ec9dd5e2c946c185d5c43a447d67647b46f53ca'},
    'common_q_gate': {'path': os.path.join(BENCH.OUT_ROOT_REL, 'common_q_gate', 'common_q_gate.json'),
                      'sha256': '8e1a60f6c7e6e4b9e421004782b031a6813930a1a615f7d8a3e5b89bd9767f5e'},
    'tso_coupling_check': {'path': os.path.join(BENCH.OUT_ROOT_REL, 'tso_coupling_check', 'tso_coupling_check.json'),
                           'sha256': 'bedf6d0aa22c152ee8391a58cf94406f5dab212611b0bb31f40c6a360d4ca1d5'},
}
V2_STAGE_OUTPUTS_COMMITTED_IN = '2ee7a124'
# W113's cold-start records of the spec-v2 arms (committed 2ee7a124 / 209f4829): the sweep's reproduction reference
# (reported, not gating) and the source of its expected counts before the first failing block.
W113_COLD = {
    'passive': {'per_solve_record': {'path': os.path.join(BENCH.OUT_ROOT_REL, 'arm_passive_cold', 'per_solve_record.jsonl'),
                                     'sha256': 'b7458de38383f04d0781981f9e96fc49f5c668562ee3c3838676f71ac3d054e9'},
                'failure': {'path': os.path.join(BENCH.OUT_ROOT_REL, 'arm_passive_cold', 'failure.json'),
                            'sha256': 'dcc53e5927ae8ff0acde2ceeade29056ca8c87a6069a35835165e373208e30d5'}},
    'price_taker': {'per_solve_record': {'path': os.path.join(BENCH.OUT_ROOT_REL, 'arm_price_taker_cold',
                                                              'per_solve_record.jsonl'),
                                         'sha256': 'c39414d94a73e45abadf5eec02cc517db79e91e90745e5b7c898e0c6a6ee6f94'},
                    'failure': {'path': os.path.join(BENCH.OUT_ROOT_REL, 'arm_price_taker_cold', 'failure.json'),
                                'sha256': '9777d0b8f245390fec6db971a96cf00f453eb5178e19e1272c156a3ed417e809'}},
}

NRF_ARM_RUN_IDS = [f'nrf_arm_{a}_{s}' for a in ('passive', 'price_taker')
                   for s in ('cold', 'warm_from_certified', 'perturbed')]
NRF_VARIANT_RUN_IDS = ['nrf_passive_tie_breaker_0p1', 'nrf_passive_tie_breaker_10']
SWEEP_RUN_IDS = ['sweep_passive_cold', 'sweep_price_taker_cold']
REPORT_RUN_ID = 'report_v3'
PERMITTED_ARM_SITES = (('uncoordinated_benchmark.py', '_solve_block'),)
PERMITTED_SWEEP_SITES = (('uncoordinated_benchmark.py', '_solve_block'), (W114_SCRIPT, 'elastic_solve'))

# production's attempts per network solve: primary, recovery (tier 1), tier 2 (network._run_smopf)
MAX_ATTEMPTS_PER_NETWORK_SOLVE = 3
E3_LAUNCHES_PER_FAILING_BLOCK = 1
E3_VARIANT_LABEL = 'E3_interface_P_elastic_Q_hard'
E3_ROWS = ('uncoord_interface_p_fixed',)
SRP1_DECLARED_V3 = {
    'dso_blocks': 36, 'tso_blocks': 12, 'nrf_rows_per_dso_block': 24,
    'nrf_arm_solves': 48, 'nrf_reevaluation_solves': 36, 'nrf_sequential_pass_solves': 48, 'nrf_variant_solves': 48,
    'max_attempts_per_network_solve': MAX_ATTEMPTS_PER_NETWORK_SOLVE,
    'e3_launches_per_failing_tso_block': E3_LAUNCHES_PER_FAILING_BLOCK,
    'sweep_upper_bound_per_arm': 36 * MAX_ATTEMPTS_PER_NETWORK_SOLVE
    + 12 * (MAX_ATTEMPTS_PER_NETWORK_SOLVE + E3_LAUNCHES_PER_FAILING_BLOCK),
}
# A move / slack at or below this is a numerical zero in the sweep's per-hour table (W114's SLACK_REPORT_THRESHOLD,
# 1e-6 model units = p.u. on the TN base) -- expressed in MW on the TN base below.
SWEEP_ACCEPT_THRESHOLD_PU = 1e-6
# The reverse-flow count: PRIMARY is strict (p_int < 0, Addendum 57's wording); SECONDARY separates numerical zeros
# (p_int < -1e-6 p.u. of the DN base, the same 1e-6 model-unit threshold). Both reported.
REVERSE_FLOW_MATERIAL_TOL_PU = 1e-6
PERIOD_HOURS = 1.0                    # SRP1: 24 one-hour periods per representative day (asserted where used)

NRF_DEFINITION = {
    'row': 'uncoordinated_benchmark.NO_REVERSE_FLOW_ROW = uncoord_no_reverse_flow[s_m, s_o, p]: pg_adn[s_m, s_o, p] >= 0',
    'bounded_quantity': ('pg_adn[s_m, s_o, p], production\'s per-scenario DSO interface expression '
                         '(network.py L444; model_construction_helpers.interface_pf_p_distribution_def L1287-1297 = '
                         'pg[ref_gen, s_m, s_o, p] - the scenario-free shared-ESS net power at the reference bus); a ROW '
                         'because pg_adn is an Expression; expected_interface_pf_p[p] = E_s[pg_adn] (the Var the TSO '
                         'arm is fixed to) inherits the bound'),
    'sign': ('pg_adn > 0 = power from the TN INTO the DN (import); < 0 = reverse flow (export). Source: the DN reference '
             'generator is the upstream grid and a generator injects into its bus (compute_node_gen L1425-1433, load '
             'convention compute_node_load L1402-1419); the DSO settlement pays +pi_t baseMVA pg_adn '
             '(interface_energy_settlement L1729-1756, the cost of importing); the TSO copy pc_adn is the TN load at '
             'the ADN bus (interface_pf_p_transmission_def L1241-1262) and the consensus pairs E[pg_adn] with E[pc_adn] '
             '(dn_/tn_interface_expected_pf_p_def L2416-2424 / L2473-2475). Checked from the Q181 models by the W116 '
             'zero-solve checks (TN energy balance: sum pg_TN - sum pc_adn = TN losses >= 0 every hour)'),
    'where': 'every DSO arm block (3 DNs x 3 years x 4 days), every market / operation scenario, every period; '
             '24 rows per block at SRP1; declared per block in the build record and counted by the structural check',
    'not_in': 'the coordinated path, production, and the TSO arm (unchanged: interface P/Q fixed, voltages bounded)',
    'consistency_reevaluation': ('deactivated with the voltage / thermal limit rows (the reference generator is freed '
                                 'there) and evaluated as a hard DN limit (kind no_reverse_flow, excess p.u., under '
                                 'hard_tol) -- a violation at the TN\'s actual voltage triggers the declared sequential '
                                 'pass, in which the DSO re-solves WITH the rows at that voltage'),
    'arm_economies': {'passive': 'flexibility fixed at 0; decision tie-breaker 1 EUR/MWh (minimum curtailment the sole '
                                 'term); value-independence at 0.1 and 10',
                      'price_taker': 'production local pricing (settlement weight 1), decision tie-breaker 0, under '
                                     'the bound'},
    'authority': 'Addendum 57 Decision 1(b)',
}

_GUARD = None
_LOG_T0 = time.time()


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W116 +{time.time() - _LOG_T0:8.1f}s] {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


def _install_guard(label, permitted):
    global _GUARD
    from p513_solve_profile_guard import SolveProfileGuard
    _GUARD = SolveProfileGuard(permitted, label=f'P5.15 W116 {label}').install()
    return _GUARD


def _check_guard(expected, where):
    failures = _GUARD.verify(expected)
    if failures:
        raise RuntimeError(f'SolveProfileGuard at {where}: expected exactly {expected}: {failures}; '
                           f'counts {_GUARD.counts}')
    return {'where': where, 'expected': expected, 'counts': dict(_GUARD.counts), 'verified': True}


def _load_verified_json(entry):
    path = BENCH._verified_path(entry)
    with open(path) as handle:
        return json.load(handle)


def _new_planning(O, run_id):
    eval_id = EVAL_ID_PREFIX + run_id
    if os.path.exists(os.path.join(O.WORK_DIR, eval_id)):
        raise RuntimeError(f'refusing: IPOPT log dir already exists: {os.path.join(O.WORK_DIR, eval_id)}')
    planning = O.fresh_planning(eval_id)
    return planning, os.path.join(O.WORK_DIR, eval_id, 'logs')


# ======================================================================================================================
#  preconditions, lock, spec binding, provenance
# ======================================================================================================================
def _latest_frozen_spec(root_rel=None):
    import glob
    best = None
    for path in glob.glob(os.path.join(_abs(root_rel or OUT_ROOT_REL), FROZEN_SPEC_GLOB)):
        match = re.fullmatch(r'frozen_s53_benchmark_spec_v(\d+)_([0-9a-f]{8})\.json', os.path.basename(path))
        if match and (best is None or int(match.group(1)) > best[1]):
            best = (path, int(match.group(1)), match.group(2))
    return best


def _frozen_spec_identity():
    latest = _latest_frozen_spec()
    if latest is None:
        return None
    return {'path': os.path.relpath(latest[0], REPO), 'version': latest[1], 'sha256': H.sha256_file(latest[0])}


def frozen_spec_binding_failures(root_rel=None):
    """A v3 stage runs only under its frozen spec: the highest-version spec under the v3 output root exists, has
    version >= 3, content sha256 starting with its <hash8>, names this output root, binds exactly
    FROZEN_SPEC_BOUND_FILES at their on-disk sha256, and carries the COORDINATED models / reference Q."""
    latest = _latest_frozen_spec(root_rel)
    if latest is None:
        return [f'no frozen benchmark spec {FROZEN_SPEC_GLOB} under {root_rel or OUT_ROOT_REL}']
    path, version, hash8 = latest
    failures = []
    if version < FROZEN_SPEC_MIN_VERSION:
        failures.append(f'frozen spec version {version} < {FROZEN_SPEC_MIN_VERSION}')
    sha = H.sha256_file(path)
    if not sha.startswith(hash8):
        failures.append(f'frozen spec {path}: content sha256 {sha} does not start with its name hash {hash8}')
    with open(path) as handle:
        spec = json.load(handle)
    if spec.get('output_root') != OUT_ROOT_REL:
        failures.append(f"frozen spec output_root {spec.get('output_root')!r} != {OUT_ROOT_REL!r}")
    pins = spec.get('code_sha256_binding') or {}
    if sorted(pins) != sorted(FROZEN_SPEC_BOUND_FILES):
        failures.append(f'frozen spec binds {sorted(pins)} != FROZEN_SPEC_BOUND_FILES {sorted(FROZEN_SPEC_BOUND_FILES)}')
    for name, pinned in pins.items():
        got = H.sha256_file(_abs(name))
        if got != pinned:
            failures.append(f'{name}: sha256 {got} != frozen spec pin {pinned}')
    coordinated = spec.get('coordinated') or {}
    if (coordinated.get('certified_models') or {}).get('sha256') != BENCH.COORDINATED['certified_models']['sha256']:
        failures.append('frozen spec coordinated certified_models sha256 != COORDINATED')
    if (coordinated.get('reference_q') or {}).get('hex') != float(BENCH.COORDINATED['certified_gross']).hex():
        failures.append('frozen spec coordinated reference Q != COORDINATED certified_gross')
    return failures


def check_preconditions(run_dir):
    failures = H.check_campaign_preconditions(run_dir, extra_clean_files=EXTRA_CLEAN_FILES)
    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan the process table: {error}')
        ps_output = ''
    excluded = {str(pid) for pid in H._ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[1] in excluded:
            continue
        if any(s in line for s in BENCH.EXTRA_FORBIDDEN_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')
    for lock in (LOCK_PATH, BENCH.LOCK_PATH):
        if os.path.exists(lock):
            failures.append(f'benchmark lock exists: {lock}')
    failures.extend(frozen_spec_binding_failures())
    return failures


def acquire_lock(run_id):
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as handle:
        json.dump({'pid': os.getpid(), 'run_id': run_id, 'started_utc': _utc()}, handle)


def release_lock():
    if os.path.exists(LOCK_PATH):
        os.remove(LOCK_PATH)


def provenance(extra=None):
    canonical, key = BENCH.x0_candidate_key()
    record = {
        'stage': STAGE, 'authority': AUTHORITY, 'utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'script': SCRIPT_NAME, 'script_sha256': H.sha256_file(os.path.abspath(__file__)),
        'module_sha256': {name: H.sha256_file(_abs(name)) for name in FROZEN_SPEC_BOUND_FILES},
        'interpreter': sys.executable, 'nlp_solver_path': H._resolve_solver_path_from_dotenv(),
        'instance': {'problem': 'SRP1', 'label': BENCH.X0['label'], 'canonical_candidate': canonical,
                     'candidate_key': key, 'candidate_key_declared': BENCH.X0['candidate_key'],
                     'candidate_key_matches': key == BENCH.X0['candidate_key']},
        'coordinated_cell': {k: v for k, v in BENCH.COORDINATED.items() if k != 'description'},
        'frozen_benchmark_spec': _frozen_spec_identity(),
        'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED (production '
                                 '_get_operational_recourse_components); every block priced for the evaluation with '
                                 "production's ADMM-subproblem pricing and the curtailment tie-breaker at the "
                                 'EVALUATION value 0; salvage reported separately'),
        'no_reverse_flow_definition': NRF_DEFINITION,
        'tie_breaker': BENCH.TIE_BREAKER, 'arm_network_compl_inf_tol': BENCH.ARM_NETWORK_COMPL_INF_TOL,
        'perturbation': BENCH.PERTURBATION, 'q_min_over_starts_note': BENCH.Q_MIN_OVER_STARTS_NOTE,
        'tolerances': {'consistency': BENCH.CONSISTENCY_TOL, 'sweep_accept_threshold_pu': SWEEP_ACCEPT_THRESHOLD_PU,
                       'reverse_flow_material_tol_pu': REVERSE_FLOW_MATERIAL_TOL_PU},
    }
    if extra:
        record.update(extra)
    return record


# ======================================================================================================================
#  the reverse-flow count (zero solves; pure reads) -- the coordinated Q181 models (W116 checks) and every NRF arm
# ======================================================================================================================
REVERSE_FLOW_DEFINITION = (
    'Addendum 57: per DSO interface entry (DN node, year, representative day, market scenario s_m, operation scenario '
    's_o, period p) p_int = pg_adn[s_m, s_o, p] x DN baseMVA [MW] -- the sign convention of the NRF rows (> 0 import '
    'TN -> DN, < 0 REVERSE flow, export DN -> TN). REVERSE entry: p_int < 0 (strict; the primary count, the '
    'Addendum\'s wording); MATERIAL reverse entry: p_int < -1e-6 p.u. of the DN base (separates numerical zeros). '
    'Counts: interface-hours (node x block x hour x scenario entries), probability-weighted (sum of omega), and '
    'day-weighted (sum of years x days x omega = interface-hours over the horizon, undiscounted). Energy of the reverse '
    'entries, |p_int| x 1 h: MWh per representative day (omega-weighted), day-weighted MWh (years x days), and block-'
    'weighted (years x days x discount -- the EUR-at-1-EUR/MWh convention of the curtailment figures). The TSO copy '
    'pc_adn is counted beside it (secondary), with the largest DSO - TSO difference (the consensus residual).')


def reverse_flow_count(planning, models, *, keep_entries=50):
    """The reverse-flow count of `models` ({'dso': ..., 'tso': optional}); REVERSE_FLOW_DEFINITION. Zero solves."""
    import pyomo.environ as pe
    import shared_resources_planning as srp
    fields = ('count', 'count_probability_weighted', 'count_day_weighted', 'energy_mwh_rep_day',
              'energy_mwh_day_weighted', 'energy_mwh_block_weighted')

    def zero():
        return {f: 0.0 if f != 'count' else 0 for f in fields}

    totals = {'strict': zero(), 'material': zero()}
    per_node = {}
    entries_all = []
    n_entries = 0
    min_entry = None
    at_zero = 0
    tso_side = {'strict': 0, 'material': 0, 'max_abs_dso_minus_tso_mw': 0.0} if models.get('tso') is not None else None
    tn = planning.transmission_network
    adn_nodes = list(tn.active_distribution_network_nodes)
    for node_id in sorted(planning.distribution_networks):
        dn = planning.distribution_networks[node_id]
        per_node[node_id] = {'strict': zero(), 'material': zero()}
        for year in dn.years:
            for day in dn.days:
                network = dn.network[year][day]
                block = models['dso'][node_id][year][day]
                if len(block.periods) != 24:
                    raise RuntimeError(f'reverse_flow_count: {len(block.periods)} periods, expected 24 one-hour periods')
                base = network.baseMVA
                day_weight = float(dn.years[year]) * float(dn.days[day])
                block_weight = srp._get_admm_block_weight(dn, year, day)
                t_block = None if tso_side is None else models['tso'][year][day]
                t_base = None if t_block is None else tn.network[year][day].baseMVA
                dn_idx = None if t_block is None else adn_nodes.index(node_id)
                for s_m in block.scenarios_market:
                    for s_o in block.scenarios_operation:
                        omega = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
                        for p in block.periods:
                            n_entries += 1
                            v_pu = float(pe.value(block.pg_adn[s_m, s_o, p]))
                            mw = v_pu * base
                            if abs(v_pu) <= REVERSE_FLOW_MATERIAL_TOL_PU:
                                at_zero += 1
                            entry = {'node_id': node_id, 'year': str(year), 'day': str(day), 's_m': s_m, 's_o': s_o,
                                     'period': p, 'hour': p + 1, 'p_int_mw': mw, 'p_int_pu': v_pu, 'omega': omega}
                            if t_block is not None:
                                t_mw = float(pe.value(t_block.pc_adn[dn_idx, s_m, s_o, p])) * t_base
                                entry['p_int_tso_mw'] = t_mw
                                tso_side['max_abs_dso_minus_tso_mw'] = max(tso_side['max_abs_dso_minus_tso_mw'],
                                                                           abs(mw - t_mw))
                                if t_mw < 0.0:
                                    tso_side['strict'] += 1
                                if t_mw < -REVERSE_FLOW_MATERIAL_TOL_PU * t_base:
                                    tso_side['material'] += 1
                            if min_entry is None or mw < min_entry['p_int_mw']:
                                min_entry = entry
                            for kind, hit in (('strict', v_pu < 0.0), ('material', v_pu < -REVERSE_FLOW_MATERIAL_TOL_PU)):
                                if not hit:
                                    continue
                                for target in (totals[kind], per_node[node_id][kind]):
                                    target['count'] += 1
                                    target['count_probability_weighted'] += omega
                                    target['count_day_weighted'] += day_weight * omega
                                    target['energy_mwh_rep_day'] += omega * abs(mw) * PERIOD_HOURS
                                    target['energy_mwh_day_weighted'] += day_weight * omega * abs(mw) * PERIOD_HOURS
                                    target['energy_mwh_block_weighted'] += block_weight * omega * abs(mw) * PERIOD_HOURS
                            if v_pu < 0.0:
                                entries_all.append(entry)
    entries_all.sort(key=lambda e: e['p_int_mw'])
    return {'definition': REVERSE_FLOW_DEFINITION, 'n_entries': n_entries, 'totals': totals,
            'per_node': {str(k): v for k, v in per_node.items()},
            'n_entries_at_zero_within_material_tol': at_zero,
            'min_entry': min_entry, 'reverse_entries_most_negative_first': entries_all[:keep_entries],
            'n_reverse_entries_listed': min(len(entries_all), keep_entries),
            'tso_side_secondary': tso_side,
            'material_tol_pu': REVERSE_FLOW_MATERIAL_TOL_PU,
            'any_reverse_flow_strict': totals['strict']['count'] > 0,
            'any_reverse_flow_material': totals['material']['count'] > 0}


# ======================================================================================================================
#  NRF arm (stage nrf-arm / nrf-passive-tie-breaker): v2's stage_arm with the DSO arm under no_reverse_flow=True
# ======================================================================================================================
def stage_nrf_arm(run_dir, run_id, *, arm, start, dso_tie_breaker, consistency):
    srp, UB, O, R = BENCH._production()
    planning, logs_dir = _new_planning(O, run_id)
    solver_options = BENCH.apply_arm_solver_options(srp, planning)
    candidate = BENCH._x0_candidate(srp, planning)
    checks = BENCH.common_capture_checklist(srp, UB, R, planning)
    certified = BENCH._load_pickle_verified(BENCH.COORDINATED['certified_models'])
    esso = BENCH._load_pickle_verified(BENCH.COORDINATED['esso_models'])
    checks.update(BENCH._model_capture_checks(certified, 'coordinated', need_duals=False))
    reference = UB.coordinated_reference_structure(planning, certified)
    warm = UB.extract_model_values(planning, certified) if start != UB.START_COLD else None
    coordinated_schedule = UB.get_dso_interface_schedule(planning, certified['dso'])
    del certified
    gc.collect()
    declared = UB.declared_solve_count(planning)
    d = SRP1_DECLARED_V3
    checks['declared_arm_equals_srp1_literal'] = declared['total'] == d['nrf_arm_solves']
    checks['declared_reevaluation_equals_srp1_literal'] = declared['dso'] == d['nrf_reevaluation_solves']
    checks['declared_pass_equals_srp1_literal'] = declared['total'] == d['nrf_sequential_pass_solves']
    checks['esso_salvage_is_zero_at_x0'] = planning.shared_ess_data.get_salvage_value(esso) == 0.0
    checks['tie_breaker_decision_declared'] = dso_tie_breaker is not None
    checks['no_reverse_flow_row_declared_in_module'] = getattr(UB, 'NO_REVERSE_FLOW_ROW', None) == 'uncoord_no_reverse_flow'
    BENCH._assert_checklist(checks, f'{run_id} (before any solve)')
    _check_guard(0, f'{run_id} before solves')
    sink = BENCH.SolveSink(run_dir)
    expected = 0
    phases = []

    # phase A -- the NRF arm
    arm_out = UB.run_operational_planning_uncoordinated(
        planning, candidate, arm=arm, dso_curtailment_penalty=dso_tie_breaker,
        tso_curtailment_penalty=BENCH.TIE_BREAKER['decision']['tso'], reference_structure=reference, start=start,
        warm_values=warm, perturbation=BENCH.PERTURBATION if start == UB.START_PERTURBED else None,
        record_callback=sink, no_reverse_flow=True)
    expected += declared['total']
    phases.append(_check_guard(expected, f'{run_id} phase A (NRF arm)'))
    nrf_blocks = {label: rec.get('no_reverse_flow_rows') for label, rec in arm_out['structure'].items()
                  if label.startswith('DSO|')}
    if arm_out['build']['dso'].get('no_reverse_flow') is not True or any(
            v != d['nrf_rows_per_dso_block'] for v in nrf_blocks.values()) or len(nrf_blocks) != d['dso_blocks']:
        raise RuntimeError(f'{run_id}: the NRF rows are not present as declared on every DSO block: {nrf_blocks}')
    models = {'tso': arm_out['models']['tso'], 'dso': arm_out['models']['dso'], 'esso': esso}
    evaluation = UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=BENCH.TIE_BREAKER['evaluation'],
                                      require_unchanged=False)
    nrf_held = reverse_flow_count(planning, models)
    BENCH._checkpoint(run_dir, {'phase': 'A', 'gross_operational_cost': evaluation['gross_operational_cost'],
                                'solves': expected, 'guard': dict(_GUARD.counts),
                                'reverse_entries_strict': nrf_held['totals']['strict']['count'],
                                'min_p_int_mw': (nrf_held['min_entry'] or {}).get('p_int_mw')})
    _log(f"{run_id}: phase A Q = {evaluation['gross_operational_cost']!r}; NRF check: min p_int "
         f"{(nrf_held['min_entry'] or {}).get('p_int_mw')!r} MW, strict reverse entries "
         f"{nrf_held['totals']['strict']['count']}, at zero {nrf_held['n_entries_at_zero_within_material_tol']}")
    result = {
        'run': run_id, 'arm': arm, 'start': start, 'no_reverse_flow': True,
        'decision_tie_breaker': {'dso': dso_tie_breaker, 'tso': BENCH.TIE_BREAKER['decision']['tso']},
        'evaluation_tie_breaker': BENCH.TIE_BREAKER['evaluation'],
        'capture_path_checklist': checks, 'solver_options': solver_options, 'ipopt_logs_dir': logs_dir,
        'declared_solves': {'arm': declared, 'reevaluation': declared['dso'] if consistency else 0,
                            'sequential_pass_if_triggered': declared['total'] if consistency else 0},
        'structure': arm_out['structure'], 'build': {k: BENCH._compact_build(v) for k, v in arm_out['build'].items()},
        'start_records': arm_out['start_records'],
        'phase_A': {'evaluation': BENCH._compact_evaluation(evaluation),
                    'curtailment_signed_parts': BENCH.curtailment_signed_parts(planning, models,
                                                                               evaluation['curtailment']),
                    'no_reverse_flow': {'rows_per_dso_block': nrf_blocks, 'dso_reverse_flow_count': nrf_held},
                    'dso_interface_schedule': arm_out['dso_interface_schedule'],
                    'tso_interface_schedule': arm_out['tso_interface_schedule'],
                    'tso_vs_dso_schedule_max_abs': BENCH._schedule_difference_tso_dso(arm_out),
                    'dso_schedule_vs_coordinated_max_abs': BENCH._schedule_difference(
                        arm_out['dso_interface_schedule'], coordinated_schedule),
                    'solve_summary': sink.summary(f'{arm}:{start}')},
    }
    arm_cost = evaluation['gross_operational_cost']
    arm_cost_source = 'phase_A'

    if consistency:
        mismatch = UB.interface_voltage_mismatch(planning, arm_out['models']['tso'], arm_out['models']['dso'])
        reeval = UB.reevaluate_dso_at_actual_voltage(planning, arm_out['models']['dso'], mismatch['v_actual_dn_pu'],
                                                     tolerances=BENCH.CONSISTENCY_TOL, record_callback=sink)
        expected += declared['dso']
        phases.append(_check_guard(expected, f'{run_id} phase B (consistency re-evaluation)'))
        nrf_violations = [h for b in reeval['blocks'].values() for h in b['violations']['hard']
                          if h.get('kind') == 'no_reverse_flow']
        BENCH._checkpoint(run_dir, {'phase': 'B', 'trigger': reeval['trigger_sequential_pass'],
                                    'max_abs_dv_dn_pu': mismatch['max_abs_dv_dn_pu'], 'solves': expected,
                                    'n_no_reverse_flow_violations': len(nrf_violations)})
        result['phase_B_consistency'] = {
            'max_abs_dv_dn_pu': mismatch['max_abs_dv_dn_pu'],
            'max_abs_dv_dn_pu_per_node_hour': mismatch['max_abs_dv_dn_pu_per_node_hour'],
            'dv_entries': mismatch['entries'],
            'reevaluation': reeval,
            'no_reverse_flow_violations_at_actual_voltage': {
                'n': len(nrf_violations),
                'max_excess_pu': max((h['excess_pu2'] for h in nrf_violations), default=0.0)},
            'trigger_rule': ('v2\'s rule (any hard > hard_tol, any thermal > thermal_tol, any soft excess > '
                             'soft_excess_tol); W116: the no-reverse-flow rows are evaluated as hard DN limits'),
            'solve_summary': sink.summary('consistency:'),
        }
        if reeval['trigger_sequential_pass']:
            shift = UB.pin_dso_interface_voltage(planning, arm_out['models']['dso'], mismatch['v_actual_dn_pu'])
            UB.solve_dso_models(planning, arm_out['models']['dso'], phase='sequential_pass:dso', record_callback=sink)
            new_schedule = UB.get_dso_interface_schedule(planning, arm_out['models']['dso'])
            moved = UB.set_tso_interface_targets(planning, arm_out['models']['tso'], new_schedule)
            UB.solve_tso_model(planning, arm_out['models']['tso'], phase='sequential_pass:tso', record_callback=sink)
            expected += declared['total']
            phases.append(_check_guard(expected, f'{run_id} phase C (sequential pass)'))
            evaluation_c = UB.evaluate_common_q(planning, models,
                                                evaluation_curtailment_penalty=BENCH.TIE_BREAKER['evaluation'],
                                                require_unchanged=False)
            mismatch_c = UB.interface_voltage_mismatch(planning, arm_out['models']['tso'], arm_out['models']['dso'])
            nrf_held_c = reverse_flow_count(planning, models)
            BENCH._checkpoint(run_dir, {'phase': 'C', 'gross_operational_cost': evaluation_c['gross_operational_cost'],
                                        'solves': expected,
                                        'reverse_entries_strict': nrf_held_c['totals']['strict']['count']})
            result['phase_C_sequential_pass'] = {
                'dso_voltage_shift_max_pu': shift, 'tso_target_move_max': moved,
                'evaluation': BENCH._compact_evaluation(evaluation_c),
                'curtailment_signed_parts': BENCH.curtailment_signed_parts(planning, models,
                                                                           evaluation_c['curtailment']),
                'no_reverse_flow': {'dso_reverse_flow_count': nrf_held_c},
                'effect_on_q_eur': evaluation_c['gross_operational_cost'] - evaluation['gross_operational_cost'],
                'max_abs_dv_dn_pu_after_pass': mismatch_c['max_abs_dv_dn_pu'],
                'dso_interface_schedule': new_schedule,
                'solve_summary': sink.summary('sequential_pass:'),
            }
            arm_cost = evaluation_c['gross_operational_cost']
            arm_cost_source = 'phase_C_sequential_pass'
    result['arm_cost'] = {'gross_operational_cost': arm_cost, 'source': arm_cost_source,
                          'objective_convention': evaluation['objective_convention'],
                          'curtailment_rows_are_phase_A': True}
    result['solve_profile_guard'] = {'phases': phases, 'final': _check_guard(expected, f'{run_id} end')}
    result['solve_summary_all'] = sink.summary()
    BENCH._write_json_once(os.path.join(run_dir, f'{run_id}.json'), provenance(result))
    return 0


# ======================================================================================================================
#  sweep (report-only): the spec-v2 arm, cold, continuing past failing TSO blocks; E3 per failing block
# ======================================================================================================================
SWEEP_DEFINITION = (
    'Addendum 57 Decision 1(a), report-only. The spec-v2 (UNCONSTRAINED: no_reverse_flow=False) arm, COLD start: '
    'build_dso_arm_models + check_arm_structures + apply_start(cold); the 36 DSO blocks solved '
    '(uncoordinated_benchmark._solve_block, production retry tiers); build_tso_arm_model(fixed_interface, voltage '
    'unpinned) at the DSO schedule + check_arm_structures + apply_start(cold); then EVERY TSO block solved, a failing '
    'block (ArmSolveFailure after production\'s tiers) recorded and followed by ONE elastic E3 solve on a clone() of that '
    'block (p515_s53_w114_passive_infeasibility.build_elastic_copy(block, \'rows\', (\'uncoord_interface_p_fixed\',)) + '
    'elastic_solve: pyomo core.add_slack_variables on the interface-P rows only, Q rows and every physical row hard, '
    'objective = the unweighted sum of the slacks, one IPOPT attempt, obj_scaling_factor 1), then the next block. Per '
    'TSO block and hour: ACCEPTED if the block solved (the fixed rows then hold; residual recorded), else from E3: the '
    'accepted interface P per node (expected_interface_pf_p at the E3 solution) against the DN target, move = accepted - '
    'target [MW], and the hour is NOT ACCEPTED iff some node\'s |move| > 1e-6 p.u. of the TN base (W114\'s threshold); '
    'deficit = sum over nodes of |move| [MW] (the L1 distance the TN needs), signed sum beside it. An E3 that does not '
    'terminate optimal leaves that block\'s hours UNDETERMINED (the block still counts as not accepting; h is then a '
    'lower bound, the undetermined hours listed). E3 minimises the L1 slack of the whole block; the TN arm block has no '
    'active row whose free variables span two periods (x = 0: the shared ESS are zero-rated; asserted zero-solve by the '
    'W116 checks, V5), so the per-hour minima are separable. The '
    'manuscript statement per arm: "without any interface rule the TN cannot accept the DNs\' exchange in n of 12 blocks '
    '(h hours)". No consistency re-evaluation, no Q: the sweep decides nothing.')


class BlockSolveAccount:
    """Exact per-block solve accounting against the armed guard (the sweep): before and after every network block the
    guard's delta (solves AND process launches) must equal the attempts production itself recorded for that block,
    1 <= attempts <= MAX_ATTEMPTS_PER_NETWORK_SOLVE; an E3 launch is exactly 1; the cumulative count is verified
    exactly after every block (`_check_guard`), and against the declared upper bound at the end."""

    def __init__(self, guard, upper_bound):
        self.guard = guard
        self.upper_bound = int(upper_bound)
        self.cumulative = 0
        self.ledger = []

    def _snap(self):
        return self.guard.counts['permitted_solve'], self.guard.counts['permitted_exec']

    def _settle(self, before, n_launches, *, label, kind, low, high):
        solves, execs = self._snap()
        d_solve, d_exec = solves - before[0], execs - before[1]
        problems = []
        if not (low <= n_launches <= high):
            problems.append(f'{n_launches} recorded attempts outside [{low}, {high}]')
        if d_solve != n_launches or d_exec != n_launches:
            problems.append(f'guard delta solve {d_solve} / exec {d_exec} != recorded attempts {n_launches}')
        if problems:
            raise RuntimeError(f'solve accounting FAILED at {label} ({kind}): ' + '; '.join(problems))
        self.cumulative += n_launches
        if self.cumulative > self.upper_bound:
            raise RuntimeError(f'solve accounting: cumulative {self.cumulative} exceeds the declared upper bound '
                               f'{self.upper_bound}')
        check = _check_guard(self.cumulative, f'after {label} ({kind})')
        row = {'sequence': len(self.ledger) + 1, 'block': label, 'kind': kind, 'attempts_recorded': n_launches,
               'guard_delta_solve': d_solve, 'guard_delta_exec': d_exec, 'cumulative': self.cumulative,
               'guard_verified': check['verified']}
        self.ledger.append(row)
        return row

    def network_solve(self, UB, call, *, label, kind):
        """`call()` -> (result, record) of UB._solve_block; returns (record, failure or None)."""
        before = self._snap()
        failure = None
        try:
            _result, record = call()
        except UB.ArmSolveFailure as error:
            failure = error
            record = error.record
        self._settle(before, int(record['n_attempts']), label=label, kind=kind, low=1,
                     high=MAX_ATTEMPTS_PER_NETWORK_SOLVE)
        return record, failure

    def elastic_solve(self, call, *, label):
        before = self._snap()
        out = call()
        self._settle(before, int(out['n_attempts']), label=label, kind='E3', low=E3_LAUNCHES_PER_FAILING_BLOCK,
                     high=E3_LAUNCHES_PER_FAILING_BLOCK)
        return out

    def summary(self):
        by_kind = {}
        for row in self.ledger:
            k = by_kind.setdefault(row['kind'], {'n_blocks': 0, 'launches': 0, 'n_retried': 0})
            k['n_blocks'] += 1
            k['launches'] += row['attempts_recorded']
            k['n_retried'] += int(row['attempts_recorded'] > 1)
        return {'declared_upper_bound': self.upper_bound, 'observed_total': self.cumulative,
                'within_upper_bound': self.cumulative <= self.upper_bound, 'by_kind': by_kind,
                'rule': ('exact per block: guard delta == production-recorded attempts (1..3 per network solve; '
                         'exactly 1 per E3); cumulative verified exactly after every block; total <= upper bound'),
                'ledger': self.ledger}


def installed_solve_guard():
    """The SolveProfileGuard whose wrapper is the current `OptSolver.solve` (the top of the stack), or None."""
    from pyomo.opt.base.solvers import OptSolver
    for cell in getattr(OptSolver.solve, '__closure__', None) or ():
        contents = cell.cell_contents
        if type(contents).__name__ == 'SolveProfileGuard':
            return contents
    return None


def _import_w114(expected_top_guard=None):
    """W114's module, imported AFTER this stage's guard is armed. Its import installs W114's own (permitting) guard on
    top of ours; it is uninstalled at once (LIFO, only if it is the guard in force), and the guard in force afterwards
    must be `expected_top_guard` (default: this stage's)."""
    import p515_s53_w114_passive_infeasibility as W
    expected_top_guard = _GUARD if expected_top_guard is None else expected_top_guard
    note = {'w114_module_sha256': H.sha256_file(_abs(W114_SCRIPT)), 'w114_guard_installed_at_import': W._GUARD is not None,
            'w114_guard_uninstalled': False}
    if W._GUARD is not None and installed_solve_guard() is W._GUARD:
        counts = dict(W._GUARD.counts)
        W._GUARD.uninstall()
        note.update({'w114_guard_uninstalled': True, 'w114_guard_counts_at_uninstall': counts,
                     'w114_guard_counts_all_zero': all(v == 0 for v in counts.values())})
    top = installed_solve_guard()
    note['guard_in_force_after_import'] = None if top is None else top.label
    note['guard_in_force_is_expected'] = top is expected_top_guard
    variants = {label: (kind, rows) for label, kind, rows in W.ELASTIC_VARIANTS}
    note['e3_variant'] = variants.get(E3_VARIANT_LABEL)
    if variants.get(E3_VARIANT_LABEL) != ('rows', E3_ROWS):
        raise RuntimeError(f'W114 E3 variant is not {E3_ROWS}: {variants.get(E3_VARIANT_LABEL)}')
    if top is not expected_top_guard or not note.get('w114_guard_counts_all_zero', True):
        raise RuntimeError(f'W114 import left the guard state unexpected: {note}')
    return W, note


def _w113_cold_records(arm):
    ref = W113_COLD[arm]
    path = BENCH._verified_path(ref['per_solve_record'])
    recs = {}
    with open(path) as handle:
        for line in handle:
            r = json.loads(line)
            recs[r['block']] = r
    failure = _load_verified_json(ref['failure'])
    recs[failure['record']['block']] = failure['record']
    return recs


def _hour_rows_accepted(label, year, day, sched_block, periods):
    rows = []
    for p in periods:
        target = sum(sched_block[n]['p_mw'][p] for n in sched_block)
        rows.append({'block': label, 'year': str(year), 'day': str(day), 'hour': p + 1, 'accepted': True,
                     'determined': True, 'deficit_l1_mw': 0.0, 'deficit_signed_mw': 0.0,
                     'dn_target_total_mw': target, 'tn_accepted_total_mw': None,
                     'dn_net_export_total': target < 0.0})
    return rows


def _hour_rows_from_e3(label, year, day, moves, periods, threshold_mw, determined):
    rows = []
    for p in periods:
        per_node = {}
        l1 = signed = target = accepted_total = 0.0
        worst = 0.0
        missing = False
        for node, series in moves.items():
            m = series[p]
            if m['move_p_mw'] is None:
                missing = True
                continue
            per_node[node] = {'target_p_mw': m['target_p_mw'], 'accepted_p_mw': m['accepted_p_mw'],
                              'move_p_mw': m['move_p_mw'], 'v_pu_tn': m['v_pu_tn']}
            l1 += abs(m['move_p_mw'])
            signed += m['move_p_mw']
            worst = max(worst, abs(m['move_p_mw']))
            target += m['target_p_mw']
            accepted_total += m['accepted_p_mw']
        hour_determined = determined and not missing
        rows.append({'block': label, 'year': str(year), 'day': str(day), 'hour': p + 1,
                     'accepted': (worst <= threshold_mw) if hour_determined else None,
                     'determined': hour_determined, 'deficit_l1_mw': l1 if hour_determined else None,
                     'deficit_signed_mw': signed if hour_determined else None, 'max_node_move_mw': worst,
                     'dn_target_total_mw': target, 'tn_accepted_total_mw': accepted_total,
                     'dn_net_export_total': target < 0.0, 'per_node': per_node})
    return rows


def stage_sweep(run_dir, run_id, *, arm):
    srp, UB, O, R = BENCH._production()
    W, w114_note = _import_w114()
    planning, logs_dir = _new_planning(O, run_id)
    solver_options = BENCH.apply_arm_solver_options(srp, planning)
    candidate = BENCH._x0_candidate(srp, planning)
    checks = BENCH.common_capture_checklist(srp, UB, R, planning)
    certified = BENCH._load_pickle_verified(BENCH.COORDINATED['certified_models'])
    checks.update(BENCH._model_capture_checks(certified, 'coordinated', need_duals=False))
    reference = UB.coordinated_reference_structure(planning, certified)
    del certified
    gc.collect()
    w113 = _w113_cold_records(arm)
    tn = planning.transmission_network
    n_dso = len(planning.distribution_networks) * len(tn.years) * len(tn.days)
    n_tso = len(tn.years) * len(tn.days)
    upper = n_dso * MAX_ATTEMPTS_PER_NETWORK_SOLVE + n_tso * (MAX_ATTEMPTS_PER_NETWORK_SOLVE
                                                              + E3_LAUNCHES_PER_FAILING_BLOCK)
    checks['sweep_upper_bound_equals_srp1_literal'] = upper == SRP1_DECLARED_V3['sweep_upper_bound_per_arm']
    checks['sweep_block_counts_srp1'] = (n_dso, n_tso) == (SRP1_DECLARED_V3['dso_blocks'], SRP1_DECLARED_V3['tso_blocks'])
    checks['w114_e3_variant_is_interface_p_rows'] = w114_note['e3_variant'] == ('rows', E3_ROWS)
    checks['w114_functions_callable'] = all(callable(getattr(W, n, None)) for n in (
        'build_elastic_copy', 'elastic_solve', 'read_slacks', 'interface_moves', 'tn_state', 'compare_to_w113'))
    checks['w113_cold_records_verified'] = len(w113) > 0
    checks['e3_row_is_fixed_interface_p_row'] = E3_ROWS == (UB.FIXED_INTERFACE_P_ROW,)
    BENCH._assert_checklist(checks, f'{run_id} (before any solve)')
    _check_guard(0, f'{run_id} before solves')
    sink = BENCH.SolveSink(run_dir)
    account = BlockSolveAccount(_GUARD, upper)
    decision = BENCH.TIE_BREAKER['decision'][f'{arm}_dso']
    phase_dso, phase_tso = f'sweep:{arm}:cold:dso', f'sweep:{arm}:cold:tso'

    # the 36 DSO blocks (the spec-v2 arm: no_reverse_flow False), cold
    dso_models, dso_build = UB.build_dso_arm_models(planning, candidate['total_capacity'], arm=arm,
                                                    curtailment_penalty=decision, no_reverse_flow=False)
    structure = UB.check_arm_structures(planning, {'dso': dso_models}, reference, dso_build_record=dso_build)
    start_records = UB.apply_start(planning, {'tso': None, 'dso': dso_models}, start=UB.START_COLD, agents=('DSO',))
    reproduction = {}
    for node_id in sorted(planning.distribution_networks):
        dn = planning.distribution_networks[node_id]
        for year in dn.years:
            for day in dn.days:
                label = UB.block_label('DSO', node_id, year, day)
                record, failure = account.network_solve(
                    UB, lambda: UB._solve_block(planning, dn, dn.network[year][day], dso_models[node_id][year][day],
                                                kind='DSO', node_id=node_id, year=year, day=day, phase=phase_dso,
                                                record_callback=sink), label=label, kind='DSO')
                if label in w113:
                    reproduction[label] = W.compare_to_w113(record, w113[label])['identical_bitwise']
                if failure is not None:
                    raise RuntimeError(f'{run_id}: DSO block {label} failed every production tier; the sweep needs '
                                       f'the DSO schedule and ends here ({failure})')
    dso_schedule = UB.get_dso_interface_schedule(planning, dso_models)
    dso_reverse = reverse_flow_count(planning, {'dso': dso_models})       # the unconstrained DNs' export entries
    BENCH._checkpoint(run_dir, {'phase': 'DSO', 'solves': account.cumulative, 'guard': dict(_GUARD.counts),
                                'dso_reverse_entries_strict': dso_reverse['totals']['strict']['count']})

    # the TSO arm at that schedule, every block, continuing past failures
    tso_model, tso_build = UB.build_tso_arm_model(planning, candidate['total_capacity'], dso_schedule,
                                                  curtailment_penalty=BENCH.TIE_BREAKER['decision']['tso'],
                                                  coupling=UB.TSO_COUPLING_FIXED,
                                                  pin_interface_voltage=UB.TSO_ARM_PIN_INTERFACE_VOLTAGE)
    structure.update(UB.check_arm_structures(planning, {'tso': tso_model}, reference, tso_build_record=tso_build,
                                             tso_coupling=UB.TSO_COUPLING_FIXED))
    start_records.update(UB.apply_start(planning, {'tso': tso_model, 'dso': None}, start=UB.START_COLD,
                                        agents=('TSO',)))
    adn_nodes = list(tn.active_distribution_network_nodes)
    per_block, hour_rows = {}, []
    for year in tn.years:
        for day in tn.days:
            network = tn.network[year][day]
            block = tso_model[year][day]
            label = UB.block_label('TSO', None, year, day)
            threshold_mw = SWEEP_ACCEPT_THRESHOLD_PU * network.baseMVA
            sched_block = {n: dso_schedule[n][year][day] for n in adn_nodes}
            record, failure = account.network_solve(
                UB, lambda: UB._solve_block(planning, tn, network, block, kind='TSO', node_id=None, year=year,
                                            day=day, phase=phase_tso, record_callback=sink), label=label, kind='TSO')
            if label in w113:
                reproduction[label] = W.compare_to_w113(record, w113[label])['identical_bitwise']
            entry = {'accepted': failure is None, 'n_attempts': record['n_attempts'],
                     'termination_condition': record['termination_condition'], 'summary': record['summary'],
                     'dn_net_exchange_mw_per_hour': [sum(sched_block[n]['p_mw'][p] for n in adn_nodes)
                                                     for p in block.periods]}
            entry['n_hours_dn_net_export'] = sum(1 for v in entry['dn_net_exchange_mw_per_hour'] if v < 0.0)
            if failure is None:
                tso_sched = UB.get_tso_interface_schedule(planning, tso_model)
                entry['fixed_row_residual_max_mw'] = max(
                    abs(tso_sched[n][year][day]['p_mw'][p] - sched_block[n]['p_mw'][p])
                    for n in adn_nodes for p in block.periods)
                rows = _hour_rows_accepted(label, year, day, sched_block, block.periods)
            else:
                e3_label = f'E3_{year}_{day}'
                c, targets, bound_map, _classification = W.build_elastic_copy(block, 'rows', E3_ROWS)
                e3 = account.elastic_solve(lambda: W.elastic_solve(planning, tn, network, c, e3_label), label=label)
                optimal = e3['termination_condition'] == 'optimal' and e3['solution_loaded']
                slacks = W.read_slacks(c, targets, network, bound_map, allow_none=not optimal)
                moves = W.interface_moves(c, network, sched_block)
                entry['e3'] = {'label': e3_label, 'solve': e3, 'determined': optimal,
                               'elastic_objective_value': W._val(
                                   c.component('_core_add_slack_variables')._slack_objective),
                               'slacks': slacks, 'interface_accepted_vs_target': moves,
                               'tn_state': W.tn_state(c, network) if e3['solution_loaded'] else None}
                rows = _hour_rows_from_e3(label, year, day, moves, block.periods, threshold_mw, optimal)
                del c
                gc.collect()
            entry['hours_not_accepted'] = [r['hour'] for r in rows if r['accepted'] is False]
            entry['hours_undetermined'] = [r['hour'] for r in rows if not r['determined']]
            per_block[label] = entry
            hour_rows.extend(rows)
            BENCH._checkpoint(run_dir, {'phase': 'TSO', 'block': label, 'accepted': entry['accepted'],
                                        'hours_not_accepted': entry['hours_not_accepted'],
                                        'solves': account.cumulative})
            _log(f"{run_id}: {label} {'ACCEPTED' if entry['accepted'] else 'NOT ACCEPTED'}"
                 + ('' if entry['accepted'] else f" -- E3 {entry['e3']['solve']['termination_condition']}, hours "
                                                  f"{entry['hours_not_accepted']} undetermined "
                                                  f"{entry['hours_undetermined']}"))
    failing = [label for label, e in per_block.items() if not e['accepted']]
    n_hours = sum(1 for r in hour_rows if r['accepted'] is False)
    n_undetermined = sum(1 for r in hour_rows if not r['determined'])
    statement = (f"without any interface rule the TN cannot accept the DNs' exchange in {len(failing)} of "
                 f"{len(per_block)} blocks ({n_hours} hours"
                 + (f'; {n_undetermined} hours undetermined, so h is a lower bound' if n_undetermined else '') + ')')
    accounting = account.summary()
    final = _check_guard(account.cumulative, f'{run_id} end')
    result = provenance({
        'run': run_id, 'arm': arm, 'start': 'cold', 'no_reverse_flow': False, 'report_only': True,
        'decision_tie_breaker': {'dso': decision, 'tso': BENCH.TIE_BREAKER['decision']['tso']},
        'capture_path_checklist': checks, 'solver_options': solver_options, 'ipopt_logs_dir': logs_dir,
        'w114_import': w114_note, 'structure': structure,
        'build': {'dso': BENCH._compact_build(dso_build), 'tso': BENCH._compact_build(tso_build)},
        'start_records': start_records, 'dso_interface_schedule': dso_schedule,
        'dso_reverse_flow_count_unconstrained': dso_reverse,
        'sweep': {'definition': SWEEP_DEFINITION, 'per_block': per_block, 'per_block_hour': hour_rows,
                  'n_blocks': len(per_block), 'n_blocks_tn_cannot_accept': len(failing), 'failing_blocks': failing,
                  'n_hours_tn_cannot_accept': n_hours, 'n_hours_undetermined': n_undetermined,
                  'h_is_lower_bound': n_undetermined > 0, 'statement': statement,
                  'deficit_l1_mw_total_rep_days': sum(r['deficit_l1_mw'] or 0.0 for r in hour_rows),
                  'accept_threshold_pu_tn_base': SWEEP_ACCEPT_THRESHOLD_PU},
        'reproduction_vs_w113_cold_bitwise': {'per_block': reproduction,
                                              'n_compared': len(reproduction),
                                              'n_identical': sum(1 for v in reproduction.values() if v),
                                              'gating': False},
        'solve_accounting': accounting, 'solve_profile_guard': final, 'solve_summary_all': sink.summary(),
    })
    BENCH._write_json_once(os.path.join(run_dir, f'{run_id}.json'), result)
    _log(f'{run_id}: {statement}; launches {account.cumulative} (upper bound {upper})')
    return 0


# ======================================================================================================================
#  report (zero solves)
# ======================================================================================================================
REPORT_CAPTURE_PATHS_V3 = {
    'common_q_gate_status_v2': ('common_q_gate', ('status',)),
    'common_q_reference_q181_v2': ('common_q_gate', ('gate', 'gross_certified')),
    'coordinated_curtailment_recomputed_v2': ('common_q_gate', ('evaluation_at_0', 'curtailment', 'totals')),
    'coordinated_curtailment_signed_parts_v2': ('common_q_gate', ('curtailment_signed_parts', 'totals')),
    'lambda_t_prediction_usable_v2': ('lambda_look', ('prediction_usable',)),
    'lambda_t_vs_pi_t_summary_v2': ('lambda_look', ('summary_coordinated',)),
    'coupling_check_tn_cost_totals_v2': ('tso_coupling_check', ('tn_cost_weighted_totals',)),
    'nrf_arm_cost': ('nrf_arm_*', ('arm_cost', 'gross_operational_cost')),
    'nrf_arm_cost_source': ('nrf_arm_*', ('arm_cost', 'source')),
    'nrf_arm_curtailment_totals': ('nrf_arm_*', ('phase_A', 'evaluation', 'curtailment', 'totals')),
    'nrf_arm_curtailment_signed_parts': ('nrf_arm_*', ('phase_A', 'curtailment_signed_parts', 'totals')),
    'nrf_arm_no_reverse_flow_held': ('nrf_arm_*', ('phase_A', 'no_reverse_flow', 'dso_reverse_flow_count', 'totals')),
    'nrf_arm_consistency_max_abs_dv': ('nrf_arm_*', ('phase_B_consistency', 'max_abs_dv_dn_pu')),
    'nrf_arm_consistency_trigger': ('nrf_arm_*', ('phase_B_consistency', 'reevaluation', 'trigger_sequential_pass')),
    'nrf_arm_consistency_nrf_violations': ('nrf_arm_*', ('phase_B_consistency',
                                                         'no_reverse_flow_violations_at_actual_voltage')),
    'nrf_variant_dso_interface_schedule': ('nrf_passive_tie_breaker_*', ('phase_A', 'dso_interface_schedule')),
    'nrf_variant_cost': ('nrf_passive_tie_breaker_*', ('arm_cost', 'gross_operational_cost')),
    'nrf_variant_curtailment_totals': ('nrf_passive_tie_breaker_*', ('phase_A', 'evaluation', 'curtailment', 'totals')),
    'nrf_variant_curtailment_signed_parts': ('nrf_passive_tie_breaker_*', ('phase_A', 'curtailment_signed_parts',
                                                                           'totals')),
    'sweep_n_blocks': ('sweep_*', ('sweep', 'n_blocks_tn_cannot_accept')),
    'sweep_n_hours': ('sweep_*', ('sweep', 'n_hours_tn_cannot_accept')),
    'sweep_per_block_hour': ('sweep_*', ('sweep', 'per_block_hour')),
    'sweep_statement': ('sweep_*', ('sweep', 'statement')),
    'sweep_solve_accounting': ('sweep_*', ('solve_accounting', 'observed_total')),
    'coordinated_reverse_flow_count': ('reverse_flow_count_q181', ('count', 'totals')),
}
REPORT_PRODUCERS_V3 = {'nrf_arm_*': 'stage_nrf_arm', 'nrf_passive_tie_breaker_*': 'stage_nrf_arm',
                       'sweep_*': 'stage_sweep', 'common_q_gate': 'v2 (read-only)', 'lambda_look': 'v2 (read-only)',
                       'tso_coupling_check': 'v2 (read-only)',
                       'reverse_flow_count_q181': 'p515_s53_w116_benchmark_spec_v3.v4_reverse_flow_count (checks)'}


def report_capture_check(outputs):
    present, absent = {}, []
    for quantity, (source, path) in REPORT_CAPTURE_PATHS_V3.items():
        ids = [k for k in outputs if k.startswith(source[:-1])] if source.endswith('*') else [source]
        if not ids:
            absent.append(f'{quantity}@{source}')
            present[f'{quantity}@{source}'] = False
        for run_id in ids:
            ok, _value = BENCH._dig(outputs.get(run_id), path)
            present[f'{quantity}@{run_id}'] = ok
            if not ok:
                absent.append(f'{quantity}@{run_id}')
    return {'n_checked': len(present), 'absent': absent, 'all_present': not absent}


def _score_predictions(spec, per_arm, claim, sweeps, nrf_failed):
    preds = (spec or {}).get('predictions_recorded_before_any_run_v3') or {}
    out = {}
    feas = preds.get('nrf_arms_feasible_at_every_tso_block')
    if feas is not None:
        out['nrf_arms_feasible_at_every_tso_block'] = {
            'prediction': feas, 'nrf_runs_failed': nrf_failed,
            'outcome': 'held' if not nrf_failed and len(per_arm) == 2 else ('failed' if nrf_failed else 'not scoreable')}
    sign = preds.get('nrf_claim_sign')
    if sign is not None:
        out['nrf_claim_sign'] = {'prediction': sign, 'claim': None if not claim.get('computed') else {
            'benefit_eur': claim['benefit_eur'], 'verdict': claim['verdict'], 'determinate': claim['determinate']}}
    sweep = preds.get('sweep_n_of_12')
    if sweep is not None:
        out['sweep_n_of_12'] = {'prediction': sweep, 'observed': {
            arm: None if rec is None else {'n': rec['sweep']['n_blocks_tn_cannot_accept'],
                                           'h': rec['sweep']['n_hours_tn_cannot_accept'],
                                           'failing_blocks': rec['sweep']['failing_blocks']}
            for arm, rec in sweeps.items()}}
    out['scoring_note'] = 'outcomes stated beside the frozen predictions; the Planner scores the reasoned ones'
    return out


def stage_report(run_dir):
    root = _abs(OUT_ROOT_REL)

    def load(run_id):
        path = os.path.join(root, run_id, f'{run_id}.json')
        if not os.path.exists(path):
            return None
        with open(path) as handle:
            return json.load(handle)

    def failed(run_id):
        return os.path.exists(os.path.join(root, run_id, 'failure.json'))

    v2 = {name: _load_verified_json(entry) for name, entry in V2_STAGE_OUTPUTS.items()}
    gate, look, coupling = v2['common_q_gate'], v2['lambda_look'], v2['tso_coupling_check']
    arms = {run_id: load(run_id) for run_id in NRF_ARM_RUN_IDS}
    variants = {run_id: load(run_id) for run_id in NRF_VARIANT_RUN_IDS}
    sweeps = {run_id: load(run_id) for run_id in SWEEP_RUN_IDS}
    reverse = None
    reverse_path = _abs(REVERSE_FLOW_OUTPUT_REL)
    if os.path.exists(reverse_path):
        with open(reverse_path) as handle:
            reverse = json.load(handle)
    missing = [k for k, v in list(arms.items()) + list(variants.items()) + list(sweeps.items()) if v is None]
    if reverse is None:
        missing.append('reverse_flow_count_q181')
    nrf_failed = [k for k in NRF_ARM_RUN_IDS + NRF_VARIANT_RUN_IDS if failed(k)]
    q_coord = BENCH.COORDINATED['certified_gross']
    band_coord = BENCH.COORDINATED_REPRODUCIBILITY_BAND_REL * q_coord
    per_arm = {}
    for arm in ('passive', 'price_taker'):
        costs = {s: arms[f'nrf_arm_{arm}_{s}']['arm_cost']['gross_operational_cost']
                 for s in ('cold', 'warm_from_certified', 'perturbed') if arms.get(f'nrf_arm_{arm}_{s}') is not None}
        if costs:
            best_start = min(costs, key=costs.get)
            per_arm[arm] = {'q_by_start': costs, 'q_best': costs[best_start], 'best_start': best_start,
                            'multimodality_band_eur': max(costs.values()) - min(costs.values()),
                            'n_starts': len(costs),
                            'arm_cost_source_by_start': {s: arms[f'nrf_arm_{arm}_{s}']['arm_cost']['source']
                                                         for s in costs}}
    if gate is None or gate.get('status') != 'PASS':
        claim = {'computed': False, 'reason': 'common-Q gate (spec v2, read-only) has not PASSED'}
    elif len(per_arm) == 2 and all(p['n_starts'] == 3 for p in per_arm.values()):
        best_arm = min(per_arm, key=lambda a: per_arm[a]['q_best'])
        benefit = per_arm[best_arm]['q_best'] - q_coord
        larger_band = max(per_arm[best_arm]['multimodality_band_eur'], band_coord)
        claim = {
            'computed': True, 'objective_convention': 'gross_operational_cost, settlement excluded',
            'definition': 'benefit = min(Q_passive_NRF, Q_price_taker_NRF) - Q181 (Addendum 57 Decision 1(b))',
            'measures': 'dynamic coordination against a static interface limit (no reverse flow)',
            'q_min_over_starts_note': BENCH.Q_MIN_OVER_STARTS_NOTE, 'best_uncoordinated_arm': f'{best_arm}_NRF',
            'benefit_eur': benefit, 'benefit_relative': benefit / q_coord,
            'bands_eur': {'best_arm_multimodality': per_arm[best_arm]['multimodality_band_eur'],
                          'coordinated_reproducibility_0.011pct': band_coord, 'dso_band_step5': None},
            'larger_band_eur': larger_band, 'determinate': abs(benefit) > larger_band,
            'verdict': ('coordination beats the best NRF arrangement' if benefit > larger_band else
                        ('the best NRF arrangement beats coordination' if benefit < -larger_band else
                         'inside the band')),
            'decomposition': {
                'passive_NRF_minus_price_taker_NRF_eur': per_arm['passive']['q_best'] - per_arm['price_taker']['q_best'],
                'price_taker_NRF_minus_coordinated_eur': per_arm['price_taker']['q_best'] - q_coord,
                'passive_NRF_minus_coordinated_eur': per_arm['passive']['q_best'] - q_coord},
            'reverse_flow_caveat': None,
        }
        if reverse is not None and reverse['count']['totals']['strict']['count'] > 0:
            claim['reverse_flow_caveat'] = ('the coordinated solution has reverse-flow interface-hours (see '
                                            'coordinated_reverse_flow_count): part of the measured benefit is the '
                                            'value of allowing reverse flow')
    else:
        claim = {'computed': False, 'reason': f'NRF arm runs missing or failed: missing {missing}; failed {nrf_failed}'}
    value_independence = {}
    base = arms.get('nrf_arm_passive_cold')
    for run_id, rec in variants.items():
        if base is None or rec is None:
            continue
        value_independence[run_id] = {
            'dso_tie_breaker': rec['decision_tie_breaker']['dso'],
            'interface_schedule_max_abs_difference_vs_1': BENCH._schedule_difference(
                BENCH._keys_to_str(rec['phase_A']['dso_interface_schedule']),
                BENCH._keys_to_str(base['phase_A']['dso_interface_schedule'])),
            'q_difference_vs_1_eur': (rec['arm_cost']['gross_operational_cost']
                                      - base['phase_A']['evaluation']['gross_operational_cost'])}
    consistency = {run_id: {'max_abs_dv_dn_pu': (rec.get('phase_B_consistency') or {}).get('max_abs_dv_dn_pu'),
                            'trigger': ((rec.get('phase_B_consistency') or {}).get('reevaluation') or {}).get(
                                'trigger_sequential_pass'),
                            'nrf_violations': (rec.get('phase_B_consistency') or {}).get(
                                'no_reverse_flow_violations_at_actual_voltage'),
                            'pass_effect_eur': (rec.get('phase_C_sequential_pass') or {}).get('effect_on_q_eur')}
                   for run_id, rec in arms.items() if rec is not None}
    curtailment = BENCH.curtailment_table(gate, arms, variants)
    for run_id, row in curtailment['arms_and_variants_phase_A'].items():
        rec = arms.get(run_id) or variants.get(run_id)
        row['row_phase'] = 'phase_A'
        row['arm_cost_source'] = rec['arm_cost']['source']
    sweep_summary = {run_id: None if rec is None else {
        'statement': rec['sweep']['statement'], 'n_blocks_tn_cannot_accept': rec['sweep']['n_blocks_tn_cannot_accept'],
        'n_hours_tn_cannot_accept': rec['sweep']['n_hours_tn_cannot_accept'],
        'n_hours_undetermined': rec['sweep']['n_hours_undetermined'], 'failing_blocks': rec['sweep']['failing_blocks'],
        'per_block_hours_not_accepted': {label: e['hours_not_accepted'] for label, e in rec['sweep']['per_block'].items()},
        'solve_accounting_total': rec['solve_accounting']['observed_total']} for run_id, rec in sweeps.items()}
    spec_identity = _frozen_spec_identity()
    spec = None
    if spec_identity is not None:
        with open(_abs(spec_identity['path'])) as handle:
            spec = json.load(handle)
    predictions = _score_predictions(spec, per_arm, claim,
                                     {'passive': sweeps.get('sweep_passive_cold'),
                                      'price_taker': sweeps.get('sweep_price_taker_cold')}, nrf_failed)
    capture = report_capture_check({**v2, **arms, **variants, **sweeps, 'reverse_flow_count_q181': reverse})
    guard = _check_guard(0, 'report end')
    result = provenance({
        'run': REPORT_RUN_ID, 'solve_profile_guard': guard, 'missing_inputs': missing, 'failed_runs': nrf_failed,
        'report_capture_check': capture,
        'v2_inputs_read_only': {name: {**entry, 'committed_in': V2_STAGE_OUTPUTS_COMMITTED_IN}
                                for name, entry in V2_STAGE_OUTPUTS.items()},
        'coordinated': {'q': q_coord, 'q_hex': float(q_coord).hex(), 'reproducibility_band_eur': band_coord,
                        'source': ('settled cycle-181 x = 0 cell d110bd1a5977df1e_x0 (not re-run): '
                                   + BENCH.COORDINATED['certified_gross_source'])},
        'coordinated_reverse_flow_count': None if reverse is None else {
            'source': {'path': REVERSE_FLOW_OUTPUT_REL, 'sha256': H.sha256_file(reverse_path)},
            'totals': reverse['count']['totals'], 'per_node': reverse['count']['per_node'],
            'tso_side_secondary': reverse['count']['tso_side_secondary'],
            'statement': ('part of the measured benefit is the value of allowing reverse flow'
                          if reverse['count']['totals']['strict']['count'] > 0 else
                          'no reverse-flow interface-hour in the coordinated solution')},
        'curtailment_table': curtailment, 'common_q_gate_status_v2': gate.get('status'),
        'lambda_look_prediction_usable_v2': look.get('prediction_usable'),
        'lambda_look_summary_v2': look.get('summary_coordinated'),
        'tso_coupling_check_v2': {'tn_cost_weighted_totals': coupling['tn_cost_weighted_totals'],
                                  'fixed_minus_penalty_tn_cost_weighted': coupling[
                                      'fixed_minus_penalty_tn_cost_weighted']},
        'per_arm_nrf': per_arm, 'claim': claim, 'passive_tie_breaker_value_independence_nrf': value_independence,
        'consistency_nrf': consistency, 'sweep': sweep_summary, 'predictions_scored': predictions,
    })
    BENCH._write_json_once(os.path.join(run_dir, f'{REPORT_RUN_ID}.json'), result)
    return 0 if (claim.get('computed') and capture['all_present']) else 1


# ======================================================================================================================
#  main
# ======================================================================================================================
def _run_id(args):
    if args.stage == 'sweep':
        return f'sweep_{args.arm}_cold'
    if args.stage == 'nrf-arm':
        return f'nrf_arm_{args.arm}_{args.start}'
    if args.stage == 'nrf-passive-tie-breaker':
        return 'nrf_passive_tie_breaker_' + {0.1: '0p1', 10.0: '10'}[args.value]
    if args.stage == 'report':
        return REPORT_RUN_ID
    raise ValueError(args.stage)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--stage', required=True, choices=('sweep', 'nrf-arm', 'nrf-passive-tie-breaker', 'report'))
    parser.add_argument('--arm', choices=('passive', 'price_taker'))
    parser.add_argument('--start', choices=('cold', 'warm_from_certified', 'perturbed'))
    parser.add_argument('--value', type=float, choices=BENCH.TIE_BREAKER['passive_value_independence_variants'])
    args = parser.parse_args(argv)
    if args.stage in ('sweep', 'nrf-arm') and args.arm is None:
        parser.error(f'--stage {args.stage} needs --arm')
    if args.stage == 'nrf-arm' and args.start is None:
        parser.error('--stage nrf-arm needs --start')
    if args.stage not in ('sweep', 'nrf-arm') and args.arm is not None:
        parser.error('--arm only with --stage sweep / nrf-arm')
    if args.stage != 'nrf-arm' and args.start is not None:
        parser.error('--start only with --stage nrf-arm (the sweep is cold by declaration)')
    if args.stage == 'nrf-passive-tie-breaker' and args.value is None:
        parser.error('--stage nrf-passive-tie-breaker needs --value')
    if args.stage != 'nrf-passive-tie-breaker' and args.value is not None:
        parser.error('--value only with --stage nrf-passive-tie-breaker')
    return args


def main(argv=None):
    args = parse_args(argv)
    run_id = _run_id(args)
    run_dir = os.path.join(_abs(OUT_ROOT_REL), run_id)
    failures = check_preconditions(run_dir)
    if failures:
        print(f'REFUSING TO RUN {run_id}:', *failures, sep='\n  ', flush=True)
        return 2
    permitted = {'sweep': PERMITTED_SWEEP_SITES, 'nrf-arm': PERMITTED_ARM_SITES,
                 'nrf-passive-tie-breaker': PERMITTED_ARM_SITES, 'report': ()}[args.stage]
    _install_guard(run_id, permitted)
    acquire_lock(run_id)
    try:
        os.makedirs(run_dir, exist_ok=False)
        _log(f'{STAGE} -- {run_id} (permitted solve sites {permitted}); output {os.path.relpath(run_dir, REPO)}')
        try:
            if args.stage == 'sweep':
                code = stage_sweep(run_dir, run_id, arm=args.arm)
            elif args.stage == 'nrf-arm':
                code = stage_nrf_arm(run_dir, run_id, arm=args.arm, start=args.start,
                                     dso_tie_breaker=BENCH.TIE_BREAKER['decision'][f'{args.arm}_dso'],
                                     consistency=True)
            elif args.stage == 'nrf-passive-tie-breaker':
                code = stage_nrf_arm(run_dir, run_id, arm='passive', start='cold', dso_tie_breaker=float(args.value),
                                     consistency=False)
            else:
                code = stage_report(run_dir)
        except Exception as error:  # noqa: BLE001 -- recorded, then non-zero exit
            traceback.print_exc()
            BENCH._write_json_once(os.path.join(run_dir, 'failure.json'), provenance({
                'run': run_id, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
                'solve_profile_guard_counts': dict(_GUARD.counts) if _GUARD is not None else None,
                'record': getattr(error, 'record', None)}))
            code = 1
        BENCH._write_manifest(run_dir)
        _log(f'{run_id}: exit {code}; guard counts {dict(_GUARD.counts)}')
        return code
    finally:
        if _GUARD is not None:
            _GUARD.uninstall()
        release_lock()


if __name__ == '__main__':
    sys.exit(main())
