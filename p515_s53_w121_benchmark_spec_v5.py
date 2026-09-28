"""P5.15 Addendum 57 Decision 1, Planner task W121 -- BENCHMARK SPEC v5 and its ZERO-SOLVE CHECKS. NO BENCHMARK STAGE RUNS
HERE (no sweep, no NRF arm, no tie-breaker, no report).

A `SolveProfileGuard(permitted=())` is armed BEFORE any production import; importing the v4 script
(`p515_s53_w119_benchmark_spec_v4`, whose checks are re-run here as they apply to v5) arms W119's, W116's, W111's and
W106's zero-solve guards on top of it; W114's module (imported by V5) installs a permitting guard at import, which is
uninstalled at once. Every guard is verified at 0 at the end. The NL files the checks write go through Pyomo's NL writer
(`block.write(format=nl)`, the writer OptSolver.solve calls in production) into a temporary directory -- neither
OptSolver.solve nor a process launch is reached (the guards verify it).

THE DEFECT (W120, 4858b3a5): `uncoordinated_benchmark.build_consistency_reevaluation_block` fixed the DN reference-bus e at
the TN's actual interface voltage while its relaxation loop cleared the bounds of UNFIXED e/f only; the fixed e kept its
setpoint bounds vg +/- SMALL_TOLERANCE = [0.9999, 1.0001], and at the TN's 1.1 p.u. Pyomo's NL writer raised
InfeasibleConstraintException on e[0,0,0,0] (case33_1, 2025, Spring, the first re-evaluation block) before any IPOPT
launch. PLANNER RULING (W121): an implementation bug; the reference bus's vg +/- tolerance band is the DSO's voltage
SETPOINT, not a physical DN limit; make the fixed value admissible; keep every genuine DN limit enforced or checked exactly
as before. THE FIX (W121 code commit): the fixed reference e and f have their bounds cleared (recorded) before they are
fixed; nothing else in the re-evaluation changes.

WHAT v5 IS: v4 a50ed4c3 (predecessor, not edited) with: the NRF arm and tie-breaker stages under new run ids (suffix _r2;
v4's nrf_arm_passive_cold and its P56A IPOPT dir are consumed); the sweeps DONE under v4 (W120), carried by reference and
sha and not re-run; the binding re-pinned (uncoordinated_benchmark.py and the harness); the re-evaluation fix, the
Planner ruling, the aborted-entry accounting, the Planner predictions scored so far, the remaining commands in order,
and the v4 -> v5 key diff recorded. Everything else identical to v4 (V9 asserts it).

MODE --freeze-spec: writes <root>/frozen_s53_benchmark_spec_v5_<hash8>.json once (refuses unless the highest existing
  benchmark spec version anywhere under data/ is exactly 4, v4 has its committed sha256, the bound files, this script and
  the v4 script are clean in git, the only stage outputs under the root are W120's (the two sweeps and the consumed
  nrf_arm_passive_cold, sha-verified), and no _r2 output or IPOPT log dir exists).
MODE --checks (default): write-once under <root>/w121_zero_solve_checks<suffix>/.
  V0  preconditions snapshot (v3's V0)                     V1  the settled models' hash (v3's V1)
  V2  the NRF rows present / absent (v3's V2)              V3  the TSO arm unchanged against v2 (v3's V3)
  V4  the reverse-flow count recomputed, equal to the committed W116 file (v4's V4)
  V5  the sweep's solve accounting (v3's V5; the sweep code is unchanged, the sweeps are done)
  V6  W100's repository-wide boolean-typing test           V7  committed eval keys unchanged, HEAD tree (v4's V7)
  V8  the frozen spec v5 binds; NEGATIVE: altered pin, name-hash mismatch, v4 / v3 specs refused, v5 over v4
  V9  v4 -> v5 key diff: only the declared keys change; the embedded diff equals the recomputed one
  V10 no production file changed (v3's V10)                V11 the report's capture paths (v3's V11)
  V12 the stage wiring on the v5 spec (v3's V12: every command parses to its _r2 run id and log)
  V13 per-block accounting controls and W113 replay (v4's V13)
  V14 spec v5 content: predictions and rulings carried, scores so far recomputed, the report reads the sweeps sha-verified
  V15 THE FIX (Planner tests 1-3): at the observed TN voltage the OLD function (git blob at the v4 pin) makes the NL writer
      raise InfeasibleConstraintException and the NEW one writes cleanly, on every DSO block of both NRF arms; the
      re-evaluation clones old vs new differ ONLY in the reference e/f bounds; the DN physical voltage limits (the
      reference bus's included) are still checked -- planted violations reported and triggering the sequential pass;
      the fixed value equals the TN interface voltage bitwise (through production's interface_voltage_mismatch)
  V16 phase A and the TSO path unchanged against the v4 code (function-source identity of every uncoordinated_benchmark
      function but the fixed one; harness functions changed only as declared; phase-A DSO / TSO arm models digest-equal
      old vs new; W120's phase-A records replayed through the old and the new account identically)
  V17 every other FIXED variable of the re-evaluation clone (Planner test 5): per family, bounds vs fixed value, which
      are setpoints and which physical; IPOPT's recorded variable-bound violation of the W120 / sweep DSO solutions
  V18 the aborted-entry accounting: W120's case (guard solve 49 / exec 48, 48 attributed) reported as "entered but
      aborted before record" and raising if the stage tries to continue (stubs + the committed W120 records)

COMMANDS (repo root, canonical interpreter, attached, both streams, noclobber):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w121_benchmark_spec_v5.py --freeze-spec > data/SRP1/Results/P515S53/w116_benchmark_nrf/launch_logs/w121_freeze_spec_v5.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w121_benchmark_spec_v5.py --checks > data/SRP1/Results/P515S53/w116_benchmark_nrf/launch_logs/w121_zero_solve_checks.log 2>&1
Exit 0 = done / all checks pass; 1 = a check failed; 2 = refused.
"""

import argparse
import ast
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402 -- the guard first

_GUARD = SolveProfileGuard((), label='P5.15 W121 zero-solve').install()

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402 -- stdlib at import
import p515_s53_w93_uncoordinated_benchmark as BENCH  # noqa: E402 -- stdlib + H + GRIO at import
import p515_s53_w116_benchmark_nrf as NRF  # noqa: E402 -- stdlib + H + GRIO + BENCH at import; no guard
import p515_s53_w119_benchmark_spec_v4 as V4  # noqa: E402 -- arms W119's, W116's, W111's and W106's guards at import

V3 = V4.V3
W111 = V4.W111
W106 = V4.W106
PY = V3.PY
THIS = os.path.basename(__file__)
V4_SCRIPT = V4.THIS
OUT_ROOT_REL = NRF.OUT_ROOT_REL
CHECKS_DIR_REL = os.path.join(OUT_ROOT_REL, 'w121_zero_solve_checks')
LAUNCH_LOGS_REL = NRF.LAUNCH_LOGS_REL
HARNESS = NRF.SCRIPT_NAME
UB_FILE = 'uncoordinated_benchmark.py'
P56A_WORK_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals')
RULING_LABEL = 'Planner ruling (W121)'
V4_SPEC = {'path': os.path.join(OUT_ROOT_REL, 'frozen_s53_benchmark_spec_v4_a50ed4c3.json'),
           'sha256': 'a50ed4c35a4cd06593e34a49ec8d84943626e479ab3517fef764a45b8dea1af7', 'version': 4,
           'committed_in': 'e4bffcad1ede49bbea0a42f2e6ccd2e1c6b75f62'}
# the v4-bound code: uncoordinated_benchmark.py and the harness at W120's commit carry exactly v4's pins (checked)
V4_CODE_COMMIT = NRF.W120_COMMIT
W120_FAILURE = NRF.CONSUMED_RUNS_V4['nrf_arm_passive_cold']['failure']
W120_RECORDS = NRF.CONSUMED_RUNS_V4['nrf_arm_passive_cold']['per_solve_record']
W120_LOGS = {run_id: os.path.join(LAUNCH_LOGS_REL, f'{run_id}.log')
             for run_id in ('sweep_passive_cold', 'sweep_price_taker_cold', 'nrf_arm_passive_cold')}
W120_MANIFEST = {'path': os.path.join(OUT_ROOT_REL, 'w120_evidence_manifest_sha256.json'),
                 'sha256': None}      # the manifest is read for the log shas; its own sha is recorded, not pinned
W120_GUARD_COUNTS = {'permitted_solve': 49, 'permitted_exec': 48, 'blocked_solve': 0, 'blocked_exec': 0}
# the observed TN voltage in W120's error (the DN reference e of case33_1 2025 Spring, hour 1, DN p.u.); V15 re-reads it
# from the committed failure.json and asserts equality
W120_OBSERVED_V_DN_PU = 1.1000000042330327
W120_FAILURE_BLOCK = ('case33_1', '2025', 'Spring')
INFORMATIONAL_FILES = V4.INFORMATIONAL_FILES + (THIS,)
# V9: the top-level keys v5 may differ from v4 in (every other v4 key is carried identically and asserted)
V5_CHANGED_OR_ADDED_TOP_KEYS = (
    'version', 'predecessor', 'predecessor_note', 'stage', 'authority', 'frozen_utc', 'git_head', 'code_sha256_binding',
    'code_sha256_informational', 'stages_v3_addendum_57_order', 'zero_solve_checks',
    'remaining_stage_commands_v5', 'sweeps_done_v5', 'consumed_runs_v5', 'reevaluation_fix_v5', 'planner_rulings_w121',
    'predictions_scored_so_far_v5', 'solve_accounting_v5', 'solve_counts_remaining_v5', 'v4_to_v5_key_diff')
V5_STAGE_FIELDS = {'sweep': ('status', 'result_w120'),
                   'nrf-arm': ('run_id', 'command', 'status', 'replaces_v4_run_id'),
                   'nrf-passive-tie-breaker': ('run_id', 'command', 'status', 'replaces_v4_run_id'),
                   'report': ('status',)}
V5_PINS_CHANGED = sorted([HARNESS, UB_FILE])
# V16: what W121 may change in the two bound files
UB_FUNCTIONS_CHANGED = {'build_consistency_reevaluation_block'}
HARNESS_DEFS_CHANGED = {'DeclaredBlockAccount', 'stage_report', '_run_id', 'main'}
HARNESS_ASSIGNMENTS_CHANGED = {'STAGE', 'AUTHORITY', 'FROZEN_SPEC_MIN_VERSION', 'NRF_ARM_RUN_IDS', 'NRF_VARIANT_RUN_IDS'}
HARNESS_ASSIGNMENTS_ADDED = {'RUN_ID_SUFFIX_V5', 'W120_COMMIT', 'SWEEPS_DONE_W120', 'CONSUMED_RUNS_V4',
                             'ABORTED_ENTRY_STATUS'}
DECLARED_ACCOUNT_METHODS_CHANGED = {'open_phase', 'settle_record', 'close_phase', 'summary'}
DECLARED_ACCOUNT_METHODS_ADDED = {'aborted_entry'}
_T0 = time.time()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W121 +{time.time() - _T0:8.1f}s] {msg}', flush=True)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _git_blob(commit, rel):
    return subprocess.run(['git', 'show', f'{commit}:{rel}'], cwd=REPO, capture_output=True, check=True).stdout


def _entry(rel):
    return {'path': rel, 'sha256': _sha(rel)}


# ======================================================================================================================
#  stage outputs under the shared root
# ======================================================================================================================
def _v4_ids():
    suffix = NRF.RUN_ID_SUFFIX_V5
    return ([r[:-len(suffix)] for r in NRF.NRF_ARM_RUN_IDS], [r[:-len(suffix)] for r in NRF.NRF_VARIANT_RUN_IDS])


def stage_outputs_v5():
    """Every stage run directory, stage log or P56A IPOPT log dir that exists for any v4 or v5 run id; the expected set
    is exactly W120's (the two sweeps and the consumed nrf_arm_passive_cold)."""
    v4_arms, v4_variants = _v4_ids()
    run_ids = (NRF.SWEEP_RUN_IDS + NRF.NRF_ARM_RUN_IDS + NRF.NRF_VARIANT_RUN_IDS + [NRF.REPORT_RUN_ID] + v4_arms
               + v4_variants)
    present = []
    for r in run_ids:
        for rel in (os.path.join(OUT_ROOT_REL, r), os.path.join(LAUNCH_LOGS_REL, f'{r}.log'),
                    os.path.join(P56A_WORK_DIR_REL, NRF.EVAL_ID_PREFIX + r)):
            if os.path.exists(_abs(rel)):
                present.append(rel)
    expected = []
    for r in NRF.SWEEP_RUN_IDS + list(NRF.CONSUMED_RUNS_V4):
        expected += [os.path.join(OUT_ROOT_REL, r), os.path.join(LAUNCH_LOGS_REL, f'{r}.log'),
                     os.path.join(P56A_WORK_DIR_REL, NRF.EVAL_ID_PREFIX + r)]
    return {'present': sorted(present), 'expected_w120': sorted(expected),
            'unexpected': sorted(set(present) - set(expected)), 'missing_w120': sorted(set(expected) - set(present)),
            'exactly_w120': sorted(present) == sorted(expected)}


def _w120_committed_shas():
    """The W120 files the spec carries, sha-verified against W120's committed evidence manifest."""
    manifest = json.load(open(_abs(W120_MANIFEST['path'])))
    blob = _git_blob(V4_CODE_COMMIT, W120_MANIFEST['path'])
    out = {'manifest': {'path': W120_MANIFEST['path'], 'sha256': _sha(W120_MANIFEST['path']),
                        'equals_committed_blob': json.loads(blob) == manifest}}
    bad = []
    for rel, sha in manifest.items():
        if _sha(rel) != sha:
            bad.append(rel)
    for run_id, entry in NRF.SWEEPS_DONE_W120.items():
        for key in ('result', 'per_solve_record', 'manifest'):
            if manifest.get(entry[key]['path']) != entry[key]['sha256']:
                bad.append(f'{run_id}:{key} pin != W120 manifest')
    for key in ('failure', 'per_solve_record'):
        e = NRF.CONSUMED_RUNS_V4['nrf_arm_passive_cold'][key]
        if manifest.get(e['path']) != e['sha256']:
            bad.append(f'consumed:{key} pin != W120 manifest')
    hr = NRF.CONSUMED_RUNS_V4['nrf_arm_passive_cold']['ipopt_logs_hash_record']
    if manifest.get(hr['path']) != hr['sha256']:
        bad.append('ipopt hash record pin != W120 manifest')
    out['mismatches'] = bad
    out['all_verified'] = not bad and out['manifest']['equals_committed_blob']
    out['files'] = manifest
    return out


# ======================================================================================================================
#  the frozen spec v5
# ======================================================================================================================
def _argv_of(command):
    return command.split(f' -u {HARNESS} ', 1)[1].split(' > ', 1)[0]


def _stages_v5(v4_stages):
    stages = copy.deepcopy(v4_stages)
    for st in stages:
        if st['stage'] == 'sweep':
            st['status'] = ('DONE under spec v4 (W120, commit 4858b3a5, exit 0) -- NOT re-run; the report reads the '
                            'committed result sha-verified (p515_s53_w116_benchmark_nrf.SWEEPS_DONE_W120)')
            st['result_w120'] = NRF.SWEEPS_DONE_W120[st['run_id']]
        elif st['stage'] in ('nrf-arm', 'nrf-passive-tie-breaker'):
            old = st['run_id']
            new = old + NRF.RUN_ID_SUFFIX_V5
            st['replaces_v4_run_id'] = old
            st['run_id'] = new
            st['command'] = V3._stage_command(_argv_of(st['command']), new)
            st['status'] = ('to run under spec v5' + (
                ' -- the v4 run id was consumed by W120 (stopped in phase B on the re-evaluation defect fixed in W121)'
                if old in NRF.CONSUMED_RUNS_V4 else ''))
        elif st['stage'] == 'report':
            st['status'] = 'to run under spec v5, last (zero solves)'
    return stages


def _remaining(stages):
    rows = [{'order': i + 1, 'run_id': st['run_id'], 'stage': st['stage'], 'command': st['command'],
             'declared_solves': st['declared_solves']}
            for i, st in enumerate(s for s in stages if s['stage'] != 'sweep')]
    return {'commands_in_order': rows,
            'rules': ('run from the repository root with the canonical interpreter; attached, alone, one at a time, '
                      'both streams to a new log (noclobber); the next stage only after the previous one has exited '
                      'and been read; a stage exit 1 or 2 stops the sequence for the Planner'),
            'not_in_the_list': 'the two sweeps (DONE under v4, W120; not re-run)'}


def _sweeps_done():
    out = {}
    manifest = json.load(open(_abs(W120_MANIFEST['path'])))
    for run_id, entry in NRF.SWEEPS_DONE_W120.items():
        rec = NRF._load_verified_json(entry['result'])
        sw = rec['sweep']
        log = W120_LOGS[run_id]
        out[run_id] = {**copy.deepcopy(entry), 'launch_log': {'path': log, 'sha256': manifest.get(log)},
                       'statement': sw['statement'], 'n_blocks': sw['n_blocks'],
                       'n_blocks_tn_cannot_accept': sw['n_blocks_tn_cannot_accept'],
                       'failing_blocks': sw['failing_blocks'], 'n_hours_tn_cannot_accept': sw['n_hours_tn_cannot_accept'],
                       'n_hours_undetermined': sw['n_hours_undetermined'],
                       'deficit_l1_mw_total_rep_days': sw['deficit_l1_mw_total_rep_days'],
                       'launches_observed': rec['solve_accounting']['observed_total'],
                       'launch_upper_bound': rec['solve_accounting']['declared_upper_bound'],
                       'reproduction_vs_w113_cold': {k: rec['reproduction_vs_w113_cold_bitwise'][k]
                                                     for k in ('n_compared', 'n_identical')},
                       'frozen_spec_at_run': rec['frozen_benchmark_spec'], 'not_rerun': True}
    return out


def _consumed():
    manifest = json.load(open(_abs(W120_MANIFEST['path'])))
    failure = NRF._load_verified_json(W120_FAILURE)
    checkpoint = json.loads(open(_abs(os.path.join(OUT_ROOT_REL, 'nrf_arm_passive_cold',
                                                   'phase_checkpoints.jsonl'))).readline())
    log = W120_LOGS['nrf_arm_passive_cold']
    return {'nrf_arm_passive_cold': {
        **copy.deepcopy(NRF.CONSUMED_RUNS_V4['nrf_arm_passive_cold']),
        'launch_log': {'path': log, 'sha256': manifest.get(log)},
        'error': failure['error'], 'guard_counts_at_failure': failure['solve_profile_guard_counts'],
        'attributed_at_failure': failure['solve_accounting']['attempts_attributed_total'],
        'phase_A_observed': {'gross_operational_cost': checkpoint['gross_operational_cost'],
                             'objective_convention': 'gross_operational_cost, settlement excluded',
                             'blocks': 48, 'attempts': checkpoint['solves'], 'min_p_int_mw': checkpoint['min_p_int_mw'],
                             'note': ('phase A closed exactly (48 blocks, 48 attempts, no retry, all 12 TSO blocks '
                                      'feasible); NOT a benchmark result: the stage failed and is re-run as '
                                      'nrf_arm_passive_cold_r2')},
        'p56a_ipopt_dir': os.path.join(P56A_WORK_DIR_REL, NRF.CONSUMED_RUNS_V4['nrf_arm_passive_cold'][
            'ipopt_log_dir_eval_id']),
        'kept_as_evidence': True, 'not_reused': True}}


def _ref_bus_limits_from_case_files():
    """The DN reference bus (type 3) limits per DN case and planning year, read from the committed case JSON files the
    production reader parses (network._read_network_from_json_file: node.v_min = Vmin, node.v_max = Vmax, generator.vg =
    Vg). V15 asserts the loaded planning holds the same values."""
    srp1 = json.load(open(_abs(os.path.join('data', 'SRP1', 'SRP1.json'))))
    out = {}
    for dn in srp1['DistributionNetworks']:
        name = dn['name']
        for year in srp1['Years']:
            rel = os.path.join('data', 'SRP1', name, f'{name}_{year}.json')
            case = json.load(open(_abs(rel)))
            refs = [n for n in case['nodes'] if int(n['type']) == 3]
            gens = [g for g in case['generators'] if g['type'] == 'REF']
            out[f'{name}|{year}'] = {'case_file': {'path': rel, 'sha256': _sha(rel)},
                                     'connection_node_id': dn['connection_node_id'],
                                     'reference_nodes': [{'bus_i': n['bus_i'], 'Vmin': n['Vmin'], 'Vmax': n['Vmax'],
                                                          'baseKV': n['baseKV']} for n in refs],
                                     'reference_generator_vg': [g['Vg'] for g in gens]}
    return out


def _source_lines(module, name):
    obj = getattr(module, name)
    lines, start = inspect.getsourcelines(obj)
    return {'function': f'{module.__name__}.{name}', 'lines': [start, start + len(lines) - 1],
            'source_sha256': hashlib.sha256(''.join(lines).encode()).hexdigest()}


def reevaluation_fix_v5():
    import model_construction_helpers as mch
    import uncoordinated_benchmark as UB
    import definitions as D
    limits = _ref_bus_limits_from_case_files()
    values = {(n['Vmin'], n['Vmax']) for v in limits.values() for n in v['reference_nodes']}
    vgs = {g for v in limits.values() for g in v['reference_generator_vg']}
    failure = NRF._load_verified_json(W120_FAILURE)
    v_obs = W120_OBSERVED_V_DN_PU
    return {
        'label': RULING_LABEL,
        'defect': {
            'found_by': 'W120 (commit 4858b3a5, spec v4), stage nrf_arm_passive_cold, phase B, first block',
            'function': 'uncoordinated_benchmark.build_consistency_reevaluation_block (v4 pin b608873b)',
            'mechanism': ('the reference-bus e was FIXED at the TN\'s actual interface voltage (skip_validation=True) '
                          'while the relaxation loop cleared the bounds of UNFIXED e / f only; the fixed e kept its '
                          'setpoint bounds vg +/- SMALL_TOLERANCE = [0.9999, 1.0001]; Pyomo\'s NL writer (repn/ampl.py '
                          'cache_fixed_var, absolute TOL 1e-8) raised on the first fixed Var outside its bounds'),
            'error': failure['error'], 'evidence': dict(W120_FAILURE),
            'why_missed': ('the W93 and W116 checks built the clone at voltages inside the setpoint band (the block\'s '
                           'own voltage; the midpoint of the band); the path had never executed at a voltage outside '
                           'it')},
        'ruling': {
            'label': RULING_LABEL,
            'text': ('an implementation bug. Addendum 49\'s convention is "DN at the TN\'s actual interface voltage". '
                     'The reference bus\'s vg +/- tolerance band is the DSO\'s voltage SETPOINT, not a physical DN '
                     'limit. Make the fixed reference-bus voltage admissible at the TN\'s value (bounds cleared the '
                     'way the loop does for the unfixed e / f, applied also to the fixed reference e and f). Keep every '
                     'genuine DN limit enforced or checked exactly as before: the DN\'s voltage-magnitude limits at '
                     'every bus, including the reference bus\'s vmin / vmax, remain hard limits checked afterwards; a '
                     'violation still triggers the one sequential pass (Addendum 49)'),
            'do_not_change': ['the convention (DN at the TN\'s actual interface voltage)',
                              'which voltage the DN is fixed at', 'the NRF rows\' treatment (deactivated in the clone, '
                              'checked as hard DN limits)', 'anything in phase A']},
        'fix': {
            'function': 'uncoordinated_benchmark.build_consistency_reevaluation_block',
            'change': ('for every (s_m, s_o, p): the reference-bus e[ref, s_m, s_o, p] and f[ref, s_m, s_o, p] have '
                       'their (lb, ub) recorded under record[\'reference_voltage_setpoint_bounds_cleared\'], set to '
                       '(None, None), and are then fixed at the SAME values as before (e = float(v_actual_dn_pu[p]), '
                       'f = 0.0, skip_validation=True). Nothing else in the function, and no other function, changes'),
            'code_commit': 'the git_head of this spec (W121 code commit)',
            'setpoint_sources': [_source_lines(mch, 'e_bounds'), _source_lines(mch, 'f_bounds')],
            'setpoint_values': {'e_ref_dn': 'vg +/- SMALL_TOLERANCE', 'SMALL_TOLERANCE': D.SMALL_TOLERANCE,
                                'f_ref': '+/- EQUALITY_TOLERANCE', 'EQUALITY_TOLERANCE': D.EQUALITY_TOLERANCE,
                                'vg_reference_generator_srp1': sorted(vgs)}},
        'dn_reference_bus_physical_limits': {
            'what': ('the reference bus\'s voltage-magnitude limits are node.v_min / node.v_max of the DN case (read '
                     'from Vmin / Vmax, network._read_network_from_json_file). At a BUS_REF node '
                     '_voltage_magnitude_slack_enabled is False (node.type != BUS_REF is required), so vmag_sqr_bounds '
                     'returns the HARD pair (v_min^2, v_max^2) and vmag_bounds (v_min, v_max); the soft rows '
                     'voltage_magnitude_lower/upper_cons exist there too with their slack Vars bounded at (0, 0)'),
            'sources': [_source_lines(mch, n) for n in ('vmag_sqr_bounds', 'vmag_bounds', '_voltage_magnitude_slack_enabled',
                                                        'voltage_slack_down_bounds', 'voltage_slack_up_bounds',
                                                        'voltage_magnitude_lower_cons_rule',
                                                        'voltage_magnitude_upper_cons_rule', 'vmag_sqr_def')],
            'srp1_values_from_case_files': limits,
            'srp1_reference_vmin_vmax_distinct': sorted([list(v) for v in values]),
            'srp1_reference_vmag_sqr_hard_band': [0.9 ** 2, 1.1 ** 2] if values == {(0.9, 1.1)} else None,
            'how_checked_in_the_reevaluation': (
                'build_consistency_reevaluation_block records every bus\'s ORIGINAL vmag_sqr bounds '
                '(record[\'original_vmag_sqr_bounds\'], the reference bus included) before relaxing them to [0, inf), '
                'and deactivates the voltage / thermal limit rows; consistency_violations (UNCHANGED) reports a HARD '
                'item for vmag_sqr outside those original bounds, a SOFT item per violated voltage row (excess over the '
                'arm\'s own slack), the thermal rows, the reference generator\'s original P/Q bounds and the NRF rows; '
                'the declared trigger fires the one sequential pass on any hard > hard_tol, thermal > thermal_tol or '
                'soft excess > soft_excess_tol (1e-6 p.u.^2 each, BENCH.CONSISTENCY_TOL). With e and f fixed, '
                'vmag_sqr_def (active: vmag_sqr is free) makes vmag_sqr[ref] = e^2 + f^2 at the solution, so the '
                'reference bus\'s physical limit is checked on the TN\'s voltage itself'),
            'at_the_w120_voltage': {
                'v_dn_pu': v_obs, 'vmag_sqr': v_obs ** 2, 'hard_band_upper': 1.1 ** 2,
                'excess_pu2': v_obs ** 2 - 1.1 ** 2, 'hard_tol_pu2': BENCH.CONSISTENCY_TOL['hard_tol_pu2'],
                'triggers': (v_obs ** 2 - 1.1 ** 2) > BENCH.CONSISTENCY_TOL['hard_tol_pu2'],
                'reading': ('the TN interface voltage W120 reported is 1.1 p.u. + 4.2e-9; the DN reference bus\'s '
                            'physical upper limit is 1.1 p.u.: at that voltage the re-evaluation REPORTS a hard item of '
                            '~9.3e-9 p.u.^2 at the reference bus, below hard_tol -- no trigger from that alone')}},
        'unchanged_by_the_fix': ['consistency_violations', 'reevaluate_dso_at_actual_voltage',
                                 'interface_voltage_mismatch', 'pin_dso_interface_voltage', 'the NRF rows\' treatment',
                                 'the trigger rule and tolerances', 'phase A (stage_nrf_arm source identical)'],
        'other_fixed_variables_in_the_clone': {
            'reference_voltage_setpoint': ('e / f at the reference bus -- SETPOINT bounds; cleared by the fix (e was '
                                           'the defect; f = 0.0 lies inside +/- EQUALITY_TOLERANCE but is cleared '
                                           'the same way, as ruled)'),
            'dso_decisions_at_the_arm_solution': (
                'DSO_DECISION_VAR_FAMILIES and the P / Q of every non-reference generator -- PHYSICAL bounds (flexibility, '
                'curtailment, ESS power / SoC, generator limits); NOT cleared. Fixed at the arm\'s IPOPT solution: '
                'IPOPT 3.14 runs with honor_original_bounds = no and bound_relax_factor = 1e-8, so a returned value may '
                'lie up to ~1e-8 x max(1, |bound|) outside its bound, against Pyomo\'s absolute 1e-8 -- REPORTED (V17 '
                'measures the margin), not changed'),
            'zero_slacks': 'DSO_REEVALUATION_ZERO_SLACKS fixed at 0.0 -- their bounds contain 0 by construction (V17)',
            'fixed_before_the_clone': ('Vars the arm block already fixes (e.g. passive flexibility at 0); they passed '
                                       'phase A\'s NL writer; V17 lists them'),
            'measured_in': 'the W121 zero-solve checks V17'},
    }


def planner_rulings_w121():
    return {'label': RULING_LABEL,
            'reevaluation_defect': 'implementation bug; fixed as ruled (see reevaluation_fix_v5.ruling)',
            'run_ids': ('new run ids for every stage not yet run cleanly: the suffix _r2 on all six NRF arm stages and '
                        'both tie-breakers (uniformity); nrf_arm_passive_cold and its P56A working dir are consumed'),
            'sweeps': 'done: carried by reference and sha (4858b3a5), not re-run',
            'accounting': ('a solve that entered the guard and died before any record is reported explicitly as '
                           '"entered but aborted before record"; it keeps raising if the stage tries to continue; it is '
                           'never silently absorbed'),
            'carried': ['the Planner predictions (W119) and their scores so far', 'the W119 rulings', 'the v4 -> v5 key '
                        'diff (v4_to_v5_key_diff)']}


def predictions_scored_so_far(v4):
    sweeps = {arm: NRF._load_verified_json(NRF.SWEEPS_DONE_W120[f'sweep_{arm}_cold']['result'])
              for arm in ('passive', 'price_taker')}
    scored = NRF._score_planner_predictions_w119(v4['predictions_recorded_before_any_run_v4'], {}, {'computed': False},
                                                 sweeps, [])
    rows = scored.get('sweep_n_of_12_unconstrained_arms_cold') or {}
    return {'source': 'p515_s53_w116_benchmark_nrf._score_planner_predictions_w119 on the committed W120 sweep results '
                      '(sha-verified), zero solves',
            'scored': scored,
            'sweep_passive': (rows.get('passive') or {}).get('outcome'),
            'sweep_price_taker': (rows.get('price_taker') or {}).get('outcome'),
            'nrf_claim': 'not yet scoreable (NRF arms not run)',
            'nrf_arms_feasible_at_every_tso_block': ('not yet scoreable; observation only (not a score): the consumed '
                                                     'W120 passive cold phase A solved all 12 TSO blocks'),
            'both_sweep_predictions_held': ((rows.get('passive') or {}).get('outcome') == 'held'
                                            and (rows.get('price_taker') or {}).get('outcome') == 'held')}


def solve_accounting_v5():
    return {
        'ruling': ('Planner task W121: make the accounting report a solve that entered the guard and died before any '
                   'record explicitly as "entered but aborted before record"; keep it raising if the stage otherwise '
                   'tries to continue; never silently absorbed'),
        'w120_case': {'guard_counts': W120_GUARD_COUNTS, 'attributed': 48,
                      'reading': ('1 solve entered OptSolver.solve (guard solve 49) and died in the Pyomo NL writer '
                                  'before any process launch (exec 48); production produced no record for it')},
        'implemented_in': ('p515_s53_w116_benchmark_nrf.DeclaredBlockAccount.aborted_entry (status '
                           f'{NRF.ABORTED_ENTRY_STATUS!r}: solves entered, launches, phase, block in progress); in '
                           'summary() (key entered_but_aborted_before_record) and in failure.json (key '
                           'solve_accounting_entered_but_aborted_before_record) with a log line; settle_record raises '
                           'on a later record (guard delta above the attempts, text names it), open_phase and '
                           'close_phase raise on it'),
        'per_block_rule_otherwise': 'solve_accounting_v4 (unchanged)',
        'checked_by': 'the W121 zero-solve checks V18 (stubs + the committed W120 records)'}


def solve_counts_remaining(v4):
    c = v4['solve_counts']
    arms, variants = c['nrf_arm_runs'], c['nrf_variant_runs']
    lo = arms * c['nrf_arm_per_run_launch_range'][0] + variants * c['nrf_variant_per_run_launch_range'][0]
    hi = arms * c['nrf_arm_per_run_launch_range_if_triggered'][1] + variants * c['nrf_variant_per_run_launch_range'][1]
    return {'stages': 'six NRF arms (_r2), two tie-breakers (_r2), report',
            'nrf_arm_per_run_blocks_exact': c['nrf_arm_per_run_blocks_exact'],
            'nrf_arm_per_run_blocks_if_triggered': c['nrf_arm_per_run_blocks_if_triggered'],
            'nrf_variant_per_run_blocks_exact': c['nrf_variant_per_run_blocks_exact'], 'report': 0,
            'total_blocks_exact': c['total_nrf_blocks_exact'],
            'total_blocks_if_every_arm_triggers': c['total_nrf_blocks_if_every_arm_triggers'],
            'total_launch_range': [lo, hi],
            'rule': 'solve_accounting_v4 per declared block + solve_accounting_v5 (aborted entries reported)',
            'sweeps': 'done (W120): 51 (passive) and 72 (price-taker) launches, not re-run'}


def build_spec_v5(v4):
    v5 = copy.deepcopy(v4)
    stages = _stages_v5(v4['stages_v3_addendum_57_order'])
    v5.update({
        'version': 5,
        'predecessor': {'path': V4_SPEC['path'], 'sha256': V4_SPEC['sha256'], 'version': 4,
                        'committed_in': V4_SPEC['committed_in'], 'git_head_at_freeze': v4['git_head']},
        'predecessor_note': (
            'v5 = v4 after W120 stopped nrf_arm_passive_cold in phase B on a code defect (the re-evaluation clone kept '
            'the DN reference e\'s setpoint bounds while fixing it at the TN voltage), fixed in W121 as the Planner '
            'ruled. Changed: the NRF arm and tie-breaker stage entries (run ids with suffix _r2, commands, status); the '
            'sweep entries (status: DONE, the W120 results by reference and sha); the binding (uncoordinated_benchmark.py '
            'and the harness re-pinned); zero_solve_checks. Added: remaining_stage_commands_v5, sweeps_done_v5, '
            'consumed_runs_v5, reevaluation_fix_v5, planner_rulings_w121, predictions_scored_so_far_v5, '
            'solve_accounting_v5, solve_counts_remaining_v5, v4_to_v5_key_diff. Everything else identical to v4 (V9 '
            'asserts it). v4 is not edited'),
        'stage': NRF.STAGE, 'authority': NRF.AUTHORITY, 'frozen_utc': _utc(), 'git_head': _git(['rev-parse', 'HEAD']),
        'code_sha256_binding': {name: _sha(name) for name in NRF.FROZEN_SPEC_BOUND_FILES},
        'code_sha256_informational': {name: W106._informational_pin(name) for name in INFORMATIONAL_FILES},
        'stages_v3_addendum_57_order': stages,
        'zero_solve_checks': {'command': (f'set -o noclobber && {PY} -u {THIS} --checks > '
                                          f'{LAUNCH_LOGS_REL}/w121_zero_solve_checks.log 2>&1'),
                              'output': CHECKS_DIR_REL,
                              'note': ('run AFTER this freeze; V8 verifies this spec binds; V9 diffs it against v4; '
                                       'V15-V18 the fix, phase-A identity, the fixed-variable scan and the aborted-entry '
                                       'accounting'),
                              'v4_checks': v4['zero_solve_checks']},
        'remaining_stage_commands_v5': _remaining(stages),
        'sweeps_done_v5': _sweeps_done(),
        'consumed_runs_v5': _consumed(),
        'reevaluation_fix_v5': reevaluation_fix_v5(),
        'planner_rulings_w121': planner_rulings_w121(),
        'predictions_scored_so_far_v5': predictions_scored_so_far(v4),
        'solve_accounting_v5': solve_accounting_v5(),
        'solve_counts_remaining_v5': solve_counts_remaining(v4),
    })
    # the key diff of everything above against v4, embedded (the diff key itself is the one addition it cannot list)
    v5['v4_to_v5_key_diff'] = {'note': 'W111.key_diff(v4, v5 without this key); V9 recomputes it',
                               'paths': W111.key_diff(v4, json.loads(GRIO.dumps(v5)))}
    return v5


def freeze_spec():
    dirty = _git(['status', '--porcelain', '--'] + list(NRF.FROZEN_SPEC_BOUND_FILES) + [THIS, V4_SCRIPT])
    if dirty:
        print(f'REFUSING to freeze: bound files not clean in git:\n{dirty}', flush=True)
        return 2
    highest = W106._highest_existing_version()
    if highest != 4:
        print(f'REFUSING to freeze: highest existing benchmark spec version is {highest}, expected exactly 4', flush=True)
        return 2
    outputs = stage_outputs_v5()
    if not outputs['exactly_w120']:
        print(f'REFUSING to freeze: stage outputs under the root are not exactly W120\'s: {outputs}', flush=True)
        return 2
    w120 = _w120_committed_shas()
    if not w120['all_verified']:
        print(f'REFUSING to freeze: W120 evidence does not verify: {w120["mismatches"]}', flush=True)
        return 2
    v4 = W111._load_verified_json(V4_SPEC)
    v5 = build_spec_v5(v4)
    text = GRIO.dumps(v5, indent=1, sort_keys=True) + '\n'
    data = text.encode()
    sha = hashlib.sha256(data).hexdigest()
    path = _abs(os.path.join(OUT_ROOT_REL, f'frozen_s53_benchmark_spec_v5_{sha[:8]}.json'))
    with open(path, 'xb') as handle:
        handle.write(data)
    if H.sha256_file(path) != sha:
        raise RuntimeError('frozen spec sha mismatch after write')
    diff = W111.key_diff(v4, json.loads(text))
    failures = _all_guard_failures()
    _log(f'frozen spec v5: {os.path.relpath(path, REPO)} sha256 {sha}; predecessor v4 {V4_SPEC["sha256"]}; git_head '
         f'{v5["git_head"]}; guard verify(0) {failures}')
    _log(f"predictions scored so far: {v5['predictions_scored_so_far_v5']['sweep_passive']} (passive sweep), "
         f"{v5['predictions_scored_so_far_v5']['sweep_price_taker']} (price-taker sweep)")
    top = sorted({x['path'].split('.')[0].split('[')[0] for x in diff})
    _log(f'v4 -> v5 key diff: {len(diff)} paths; top-level keys changed / added / removed: {top}')
    for x in diff:
        _log(f"  {x['kind']:8s} {x['path']}")
    for row in v5['remaining_stage_commands_v5']['commands_in_order']:
        _log(f"  stage {row['order']}: {row['command']}")
    return 0 if not any(failures.values()) else 1


def _all_guard_failures():
    return {'w121': _GUARD.verify(0), 'w119_import': V4._GUARD.verify(0), 'w116_import': V3._GUARD.verify(0),
            'w111_import': W111._GUARD.verify(0), 'w106_import': W106._GUARD.verify(0)}


# ======================================================================================================================
#  checks -- helpers
# ======================================================================================================================
def _renamed(res, new_id):
    res = dict(res)
    res['id_as_written'] = res.get('id')
    res['id'] = new_id
    return res


def _latest_spec():
    latest = NRF._latest_frozen_spec()
    with open(latest[0]) as handle:
        return latest, json.load(handle)


def _load_old_ub():
    """uncoordinated_benchmark.py at the v4 pin (W120's commit), imported from a temporary copy."""
    v4 = W111._load_verified_json(V4_SPEC)
    return V3._load_module_from_git(V4_CODE_COMMIT, UB_FILE, '_ub_at_spec_v4', v4['code_sha256_binding'][UB_FILE])


def _write_nl(block, tmpdir, tag):
    """Pyomo's NL writer on `block` into `tmpdir` (the writer OptSolver.solve calls; no solve, no process launch).
    Returns {'written', 'error_type', 'error', 'bytes'}; the file is removed."""
    from pyomo.opt import ProblemFormat
    from pyomo.common.errors import InfeasibleConstraintException
    path = os.path.join(tmpdir, f'{tag}.nl')
    out = {'written': False, 'error_type': None, 'error': None, 'bytes': None}
    try:
        block.write(path, format=ProblemFormat.nl, io_options={'symbolic_solver_labels': False})
        out['written'] = True
        out['bytes'] = os.path.getsize(path)
    except InfeasibleConstraintException as error:
        out['error_type'] = 'InfeasibleConstraintException'
        out['error'] = str(error)
    except Exception as error:  # noqa: BLE001 -- any other writer failure is recorded as such
        out['error_type'] = type(error).__name__
        out['error'] = str(error)
    finally:
        for name in os.listdir(tmpdir):
            if name.startswith(tag):
                os.remove(os.path.join(tmpdir, name))
    return out


def _arm_models_warm(UB, planning, candidate, warm, arm):
    """The NRF arm's DSO blocks exactly as stage nrf-arm builds them, at the warm_from_certified start (every free Var at
    the settled Q181 solution -- an IPOPT solution under the campaign's bound options)."""
    decision = BENCH.TIE_BREAKER['decision'][f'{arm}_dso']
    models, build = UB.build_dso_arm_models(planning, candidate['total_capacity'], arm=arm, curtailment_penalty=decision,
                                            no_reverse_flow=True)
    starts = UB.apply_start(planning, {'tso': None, 'dso': models}, start=UB.START_WARM, warm_values=warm,
                            agents=('DSO',))
    return models, build, starts


def _blocks(planning):
    for node_id in sorted(planning.distribution_networks):
        dn = planning.distribution_networks[node_id]
        for year in dn.years:
            for day in dn.days:
                yield node_id, year, day, dn, dn.network[year][day]


def _var_rows(block):
    import pyomo.environ as pe
    return {v.name: (v.lb, v.ub, v.fixed, v.value, str(v.domain))
            for v in block.component_data_objects(pe.Var, descend_into=True)}


# ======================================================================================================================
#  V15 -- the fix
# ======================================================================================================================
def v15_reevaluation_fix(UB, planning, candidate, certified, warm, old):
    import pyomo.environ as pe
    import model_construction_helpers as mch
    from definitions import BUS_REF
    failure = NRF._load_verified_json(W120_FAILURE)
    observed_text = failure['error'].split('fixed value ', 1)[1].split(' ', 1)[0]
    v_obs = W120_OBSERVED_V_DN_PU
    tmp = tempfile.mkdtemp(prefix='w121_nl_')
    tol = BENCH.CONSISTENCY_TOL
    out = {'observed_v_dn_pu': v_obs, 'observed_v_equals_w120_failure_text': float(observed_text) == v_obs,
           'w120_error': failure['error']}
    try:
        # ---- test 1: the negative control, every DSO block of both NRF arms, at the observed TN voltage
        per_arm = {}
        w120_block_msg = None
        for arm in UB.DSO_ARMS:
            models, _build, _starts = _arm_models_warm(UB, planning, candidate, warm, arm)
            rows = {}
            for node_id, year, day, _dn, network in _blocks(planning):
                block = models[node_id][year][day]
                label = UB.block_label('DSO', node_id, year, day)
                v = [v_obs] * len(block.periods)
                c_old, _r_old = old.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v)
                w_old = _write_nl(c_old, tmp, 'old')
                c_new, r_new = UB.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v)
                w_new = _write_nl(c_new, tmp, 'new')
                rows[label] = {'old': {k: w_old[k] for k in ('written', 'error_type', 'error')},
                               'new': {k: w_new[k] for k in ('written', 'error_type', 'bytes')},
                               'n_setpoint_bounds_cleared': len(r_new['reference_voltage_setpoint_bounds_cleared'])}
                if arm == UB.ARM_PASSIVE and (network.name, str(year), str(day)) == W120_FAILURE_BLOCK:
                    w120_block_msg = {'label': label, 'old_error': w_old['error'],
                                      'equals_w120_error': ('InfeasibleConstraintException: ' + str(w_old['error'])
                                                            == failure['error'])}
                del c_old, c_new
            per_arm[arm] = {'n_blocks': len(rows),
                            'old_raises_infeasible_constraint_every_block': all(
                                r['old']['error_type'] == 'InfeasibleConstraintException'
                                and "variable 'e[0," in (r['old']['error'] or '') for r in rows.values()),
                            'new_writes_cleanly_every_block': all(r['new']['written'] for r in rows.values()),
                            'setpoint_bounds_cleared_per_block': sorted({r['n_setpoint_bounds_cleared']
                                                                         for r in rows.values()}),
                            'blocks': rows}
            del models
            gc.collect()
        out['test1_negative_control_observed_voltage'] = {
            'per_arm': per_arm, 'w120_block': w120_block_msg,
            'passed': (all(p['n_blocks'] == 36 and p['old_raises_infeasible_constraint_every_block']
                           and p['new_writes_cleanly_every_block'] and p['setpoint_bounds_cleared_per_block'] == [48]
                           for p in per_arm.values())
                       and w120_block_msg is not None and w120_block_msg['equals_w120_error'])}

        # the passive NRF arm again for tests 2-3 and the old-vs-new clone comparison
        models, _build, _starts = _arm_models_warm(UB, planning, candidate, warm, UB.ARM_PASSIVE)
        # ---- test 3: the fixed value equals the TN interface voltage bitwise, through production's own path
        #      (interface_voltage_mismatch on the settled Q181 TSO blocks' own interface voltages; then those Vars set
        #      to the observed W120 value -- restored bitwise afterwards)
        tn = planning.transmission_network
        adn = list(tn.active_distribution_network_nodes)
        saved = {(y, d, dn, p): certified['tso'][y][d].expected_interface_vmag[dn, p].value
                 for y in tn.years for d in tn.days for dn in range(len(adn))
                 for p in certified['tso'][y][d].periods}
        path_rows = {}
        try:
            for case in ('settled_q181_tn_voltage', 'observed_w120_tn_voltage'):
                if case == 'observed_w120_tn_voltage':
                    for (y, d, dn, p) in saved:
                        certified['tso'][y][d].expected_interface_vmag[dn, p].set_value(v_obs)
                mismatch = UB.interface_voltage_mismatch(planning, certified['tso'], models)
                n_eq = n_tn_eq = n_old_ok = n_new_ok = n_fixed = n_outside_band = 0
                max_ulp_tn = 0
                min_v, max_v = math.inf, -math.inf
                for node_id, year, day, _dn, network in _blocks(planning):
                    block = models[node_id][year][day]
                    v_actual = mismatch['v_actual_dn_pu'][node_id][year][day]
                    ref_idx = network.get_node_idx(network.get_reference_node_id())
                    c_new, _r = UB.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v_actual)
                    for p in block.periods:
                        e = c_new.e[ref_idx, 0, 0, p]
                        f = c_new.f[ref_idx, 0, 0, p]
                        n_fixed += int(e.fixed and f.fixed and f.value == 0.0)
                        band = block.e[ref_idx, 0, 0, p].bounds
                        n_outside_band += int(not (band[0] <= float(e.value) <= band[1]))
                        n_eq += int(float(e.value).hex() == float(v_actual[p]).hex())
                        v_tn_pu = float(pe.value(certified['tso'][year][day].expected_interface_vmag[
                            adn.index(node_id), p]))
                        n_tn_eq += int(float(e.value).hex() == v_tn_pu.hex())
                        max_ulp_tn = max(max_ulp_tn, abs(float(e.value) - v_tn_pu) / math.ulp(v_tn_pu))
                        min_v, max_v = min(min_v, float(e.value)), max(max_v, float(e.value))
                    n_new_ok += int(_write_nl(c_new, tmp, 'new')['written'])
                    c_old, _ro = old.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v_actual)
                    n_old_ok += int(_write_nl(c_old, tmp, 'old')['written'])
                    del c_new, c_old
                bases = sorted({(tn.network[y][d].get_node_base_kv(n),
                                 planning.distribution_networks[n].network[y][d].get_node_base_kv(
                                     planning.distribution_networks[n].network[y][d].get_reference_node_id()))
                                for n in adn for y in tn.years for d in tn.days})
                path_rows[case] = {'n_e_and_f_fixed_f_zero': n_fixed,
                                   'n_entries_outside_the_arm_setpoint_band': n_outside_band,
                                   'n_fixed_e_bitwise_equal_v_actual_dn_pu': n_eq,
                                   'n_fixed_e_bitwise_equal_tso_interface_vmag_pu': n_tn_eq,
                                   'max_ulp_fixed_e_vs_tso_interface_vmag_pu': max_ulp_tn,
                                   'v_range_dn_pu': [min_v, max_v], 'base_kv_tn_dn_pairs': bases,
                                   'old_nl_written_blocks': n_old_ok, 'new_nl_written_blocks': n_new_ok,
                                   'max_abs_dv_dn_pu_mismatch': mismatch['max_abs_dv_dn_pu']}
        finally:
            for (y, d, dn, p), value in saved.items():
                certified['tso'][y][d].expected_interface_vmag[dn, p].set_value(value)
        restored = all(certified['tso'][y][d].expected_interface_vmag[dn, p].value == value
                       for (y, d, dn, p), value in saved.items())
        n_entries = 36 * 24
        out['test3_fixed_value_equals_tn_interface_voltage'] = {
            'definition': ('v_actual_dn_pu = production interface_voltage_mismatch: TSO expected_interface_vmag[dn, p] '
                           'x TN node base kV / DN reference base kV; the clone fixes e[ref] = float(v_actual[p])'),
            'cases': path_rows, 'tso_values_restored_bitwise': restored,
            'passed': (restored and all(r['n_fixed_e_bitwise_equal_v_actual_dn_pu'] == n_entries
                                        and r['n_e_and_f_fixed_f_zero'] == n_entries and r['new_nl_written_blocks'] == 36
                                        for r in path_rows.values())
                       and path_rows['observed_w120_tn_voltage']['old_nl_written_blocks'] == 0
                       and path_rows['observed_w120_tn_voltage']['n_entries_outside_the_arm_setpoint_band'] == n_entries),
            'note': ('old_nl_written_blocks is the pre-fix function at the same voltages (reported); at the settled '
                     'Q181 TN voltages too the old clone fails the writer in every block whose TN voltage leaves '
                     '[0.9999, 1.0001]')}

        # ---- the clones old vs new differ ONLY in the reference e / f bounds (every DN limit as before)
        cmp_rows = {}
        for node_id, year, day, _dn, network in _blocks(planning):
            block = models[node_id][year][day]
            label = UB.block_label('DSO', node_id, year, day)
            ref_idx = network.get_node_idx(network.get_reference_node_id())
            v = [v_obs] * len(block.periods)
            c_old, r_old = old.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v)
            c_new, r_new = UB.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v)
            a, b = _var_rows(c_old), _var_rows(c_new)
            differs = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            expected = sorted([c_new.e[ref_idx, 0, 0, p].name for p in block.periods]
                              + [c_new.f[ref_idx, 0, 0, p].name for p in block.periods])
            only_bounds = all(a[k][2:] == b[k][2:] and b[k][:2] == (None, None) for k in differs if k in a and k in b)
            d_old, d_new = V3.model_digest(c_old), V3.model_digest(c_new)
            r_new_wo = {k: val for k, val in r_new.items() if k != 'reference_voltage_setpoint_bounds_cleared'}
            cleared = r_new['reference_voltage_setpoint_bounds_cleared']
            setpoint_ok = all(
                (tuple(cleared[c_new.e[ref_idx, 0, 0, p].name]) == tuple(block.e[ref_idx, 0, 0, p].bounds)
                 and tuple(cleared[c_new.f[ref_idx, 0, 0, p].name]) == tuple(block.f[ref_idx, 0, 0, p].bounds))
                for p in block.periods)
            cmp_rows[label] = {'vars_differing': len(differs), 'differ_exactly_in_ref_e_f': differs == expected,
                               'difference_is_bounds_only_cleared': only_bounds,
                               'rows_params_objectives_expressions_digest_equal': all(
                                   d_old['sha256'][k] == d_new['sha256'][k] for k in ('con', 'param', 'obj', 'expr')),
                               'records_equal_but_the_new_key': r_old == r_new_wo,
                               'cleared_bounds_are_the_arm_setpoint_bounds': setpoint_ok,
                               'example_cleared': {k: cleared[k] for k in list(cleared)[:2]}}
            del c_old, c_new
        out['clone_old_vs_new'] = {
            'blocks': cmp_rows,
            'passed': len(cmp_rows) == 36 and all(
                r['differ_exactly_in_ref_e_f'] and r['difference_is_bounds_only_cleared']
                and r['rows_params_objectives_expressions_digest_equal'] and r['records_equal_but_the_new_key']
                and r['cleared_bounds_are_the_arm_setpoint_bounds'] for r in cmp_rows.values()),
            'consistency_violations_source_identical': (inspect.getsource(UB.consistency_violations)
                                                        == inspect.getsource(old.consistency_violations)),
            'reevaluate_dso_at_actual_voltage_source_identical': (inspect.getsource(UB.reevaluate_dso_at_actual_voltage)
                                                                  == inspect.getsource(
                                                                      old.reevaluate_dso_at_actual_voltage))}
        out['clone_old_vs_new']['passed'] = bool(out['clone_old_vs_new']['passed']
                                                 and out['clone_old_vs_new']['consistency_violations_source_identical']
                                                 and out['clone_old_vs_new'][
                                                     'reevaluate_dso_at_actual_voltage_source_identical'])

        # ---- test 2: the DN physical voltage limits are still checked (planted violations reported and triggering)
        node_id, year, day, dn_data, network = next(
            (n, y, d, dn, net) for n, y, d, dn, net in _blocks(planning)
            if (net.name, str(y), str(d)) == W120_FAILURE_BLOCK)
        block = models[node_id][year][day]
        ref_idx = network.get_node_idx(network.get_reference_node_id())
        ref_node = network.nodes[ref_idx]
        limits = {'block': UB.block_label('DSO', node_id, year, day), 'network': network.name,
                  'reference_node_bus_i': ref_node.bus_i, 'reference_node_is_bus_ref': ref_node.type == BUS_REF,
                  'v_min': ref_node.v_min, 'v_max': ref_node.v_max,
                  'slack_enabled_at_reference': mch._voltage_magnitude_slack_enabled(ref_node, dn_data.params),
                  'vmag_sqr_bounds_rule': mch.vmag_sqr_bounds(None, ref_idx, 0, 0, 0, network, dn_data.params)}
        every_dn = {}
        for n, y, d, dn2, net in _blocks(planning):
            r_idx = net.get_node_idx(net.get_reference_node_id())
            nd = net.nodes[r_idx]
            every_dn[f'{net.name}|{y}|{d}'] = (nd.type == BUS_REF, nd.v_min, nd.v_max,
                                               mch._voltage_magnitude_slack_enabled(nd, dn2.params),
                                               net.generators[net.get_reference_gen_idx()].vg)
        limits['every_block_reference_type_vmin_vmax_slack_vg'] = sorted({str(v) for v in every_dn.values()})

        def reeval_at(v_ref, plant_other=None):
            c, rec = UB.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=[v_ref] * len(block.periods))
            for p in block.periods:          # the power-flow identity the solve enforces: vmag_sqr = e^2 + f^2, vmag
                e = float(c.e[ref_idx, 0, 0, p].value)
                c.vmag_sqr[ref_idx, 0, 0, p].set_value(e * e, skip_validation=True)
                if (ref_idx, 0, 0, p) in c.vmag:
                    c.vmag[ref_idx, 0, 0, p].set_value(abs(e), skip_validation=True)
            target = None
            if plant_other is not None:
                target = next(v for v in c.vmag_sqr.values() if v.index()[0] != ref_idx and v.index()[3] == 5)
                lb, ub = rec['original_vmag_sqr_bounds'][target.name]
                target.set_value((ub ** 0.5 + plant_other) ** 2, skip_validation=True)
            viol = UB.consistency_violations(c, rec, block, hard_tol=tol['hard_tol_pu2'],
                                             soft_excess_tol=tol['soft_excess_tol_pu2'], thermal_tol=tol['thermal_tol_pu2'])
            ref_hard = [h for h in viol['hard'] if h.get('kind') == 'vmag_sqr_band'
                        and h['var'].startswith(f'vmag_sqr[{ref_idx},')]
            ref_soft = [s for s in viol['soft'] if s['row'].split('[', 1)[1].startswith(f'{ref_idx},')]
            nl = _write_nl(c, tmp, 'planted')
            ref_bounds = sorted({tuple(rec['original_vmag_sqr_bounds'][c.vmag_sqr[ref_idx, 0, 0, p].name])
                                 for p in block.periods})
            res = {'v_ref_dn_pu': v_ref, 'trigger_sequential_pass': viol['trigger_sequential_pass'],
                   'n_hard': len(viol['hard']), 'n_hard_at_reference': len(ref_hard),
                   'max_hard_excess_at_reference_pu2': max((h['excess_pu2'] for h in ref_hard), default=0.0),
                   'n_soft_rows_at_reference': len(ref_soft), 'max_hard_excess_pu2': viol['max_hard_excess_pu2'],
                   'max_soft_excess_pu2': viol['max_soft_excess_pu2'],
                   'reference_original_vmag_sqr_bounds': ref_bounds, 'nl_written': nl['written'],
                   'planted_other_bus': None if target is None else target.name,
                   'planted_other_bus_flagged': None if target is None else any(
                       h.get('var') == target.name for h in viol['hard'])}
            del c
            return res

        above = reeval_at(1.12)
        below = reeval_at(0.88)
        observed = reeval_at(v_obs)
        other = reeval_at(1.0, plant_other=0.05)
        band = [ref_node.v_min ** 2, ref_node.v_max ** 2]
        out['test2_dn_physical_limits_still_checked'] = {
            'reference_bus_physical_limits': limits,
            'planted_above_v_max_at_reference_1p12': above, 'planted_below_v_min_at_reference_0p88': below,
            'observed_w120_voltage_1p1_plus_4p2e-9': observed, 'planted_other_bus_plus_0p05_above_v_max': other,
            'passed': bool(
                limits['reference_node_is_bus_ref'] and limits['slack_enabled_at_reference'] is False
                and tuple(limits['vmag_sqr_bounds_rule']) == (ref_node.v_min ** 2, ref_node.v_max ** 2)
                and len(limits['every_block_reference_type_vmin_vmax_slack_vg']) == 1
                and above['trigger_sequential_pass'] and above['n_hard_at_reference'] == 24
                and abs(above['max_hard_excess_at_reference_pu2'] - (1.12 ** 2 - band[1])) < 1e-12
                and above['n_soft_rows_at_reference'] == 24 and above['nl_written']
                and below['trigger_sequential_pass'] and below['n_hard_at_reference'] == 24
                and abs(below['max_hard_excess_at_reference_pu2'] - (band[0] - 0.88 ** 2)) < 1e-12
                and above['reference_original_vmag_sqr_bounds'] == [tuple(band)]
                and observed['n_hard_at_reference'] == 24 and observed['trigger_sequential_pass'] is False
                and abs(observed['max_hard_excess_at_reference_pu2'] - (v_obs ** 2 - band[1])) < 1e-15
                and other['trigger_sequential_pass'] and other['planted_other_bus_flagged']
                and other['n_hard_at_reference'] == 0)}
        del models
        gc.collect()
    finally:
        shutil.rmtree(tmp)
    parts = ('test1_negative_control_observed_voltage', 'test3_fixed_value_equals_tn_interface_voltage',
             'clone_old_vs_new', 'test2_dn_physical_limits_still_checked')
    out['passed'] = bool(out['observed_v_equals_w120_failure_text'] and all(out[k]['passed'] for k in parts))
    return {'id': 'V15_reevaluation_fix', **out}


# ======================================================================================================================
#  V16 -- phase A and the TSO path unchanged against the v4 code
# ======================================================================================================================
def _top_level_sources(text):
    """{name: source segment} for every top-level def / class / assignment target of a module's text; class methods as
    'Class.method'."""
    tree = ast.parse(text)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(text, node)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        out[f'{node.name}.{sub.name}'] = ast.get_source_segment(text, sub)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[f'={target.id}'] = ast.get_source_segment(text, node)
    return out


def v16_phase_a_and_tso_unchanged(UB, planning, candidate, certified, old):
    import pyomo.environ as pe
    v4 = W111._load_verified_json(V4_SPEC)
    # uncoordinated_benchmark.py: every callable's source identical except the fixed function; no name added / removed
    changed = sorted(n for n, f in vars(UB).items() if callable(f) and getattr(f, '__module__', None) == UB.__name__
                     and hasattr(old, n) and inspect.getsource(f) != inspect.getsource(getattr(old, n)))
    added = sorted(n for n in vars(UB) if not n.startswith('__') and not hasattr(old, n))
    removed = sorted(n for n in vars(old) if not n.startswith('__') and not hasattr(UB, n))
    constants_equal = all(getattr(old, n) == v for n, v in vars(UB).items()
                          if not n.startswith('_') and n.isupper() and hasattr(old, n))
    tso_sources = {n: inspect.getsource(getattr(UB, n)) == inspect.getsource(getattr(old, n))
                   for n in V3.TSO_PATH_FUNCTIONS}
    phase_a_functions = ('run_operational_planning_uncoordinated', 'build_dso_arm_models', '_no_reverse_flow_rule',
                         'check_arm_structures', 'check_arm_block_structure', 'solve_dso_models', 'solve_tso_model',
                         '_solve_block', 'apply_start', 'build_tso_arm_model', 'evaluate_common_q',
                         'get_dso_interface_schedule', 'get_tso_interface_schedule', 'curtailment_report')
    phase_a_sources = {n: inspect.getsource(getattr(UB, n)) == inspect.getsource(getattr(old, n))
                       for n in phase_a_functions}
    # the harness: AST source segments at the v4 pin vs now
    blob = _git_blob(V4_CODE_COMMIT, HARNESS)
    old_sha = hashlib.sha256(blob).hexdigest()
    a, b = _top_level_sources(blob.decode()), _top_level_sources(open(_abs(HARNESS)).read())
    h_changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    h_added = sorted(set(b) - set(a))
    h_removed = sorted(set(a) - set(b))
    defs_changed = {k for k in h_changed if not k.startswith('=') and '.' not in k}
    methods_changed = {k.split('.', 1)[1] for k in h_changed if k.startswith('DeclaredBlockAccount.')}
    methods_added = {k.split('.', 1)[1] for k in h_added if k.startswith('DeclaredBlockAccount.')}
    assigns_changed = {k[1:] for k in h_changed if k.startswith('=')}
    assigns_added = {k[1:] for k in h_added if k.startswith('=')}
    other_added = {k for k in h_added if not k.startswith('=') and not k.startswith('DeclaredBlockAccount.')}
    harness = {'v4_harness_sha256': old_sha, 'v4_pin': v4['code_sha256_binding'][HARNESS],
               'defs_changed': sorted(defs_changed), 'declared_account_methods_changed': sorted(methods_changed),
               'declared_account_methods_added': sorted(methods_added), 'assignments_changed': sorted(assigns_changed),
               'assignments_added': sorted(assigns_added), 'other_added': sorted(other_added), 'removed': h_removed,
               'stage_nrf_arm_identical': a.get('stage_nrf_arm') == b.get('stage_nrf_arm'),
               'stage_sweep_identical': a.get('stage_sweep') == b.get('stage_sweep'),
               'block_solve_account_identical': a.get('BlockSolveAccount') == b.get('BlockSolveAccount'),
               'declared_block_networks_identical': a.get('declared_block_networks') == b.get('declared_block_networks')}
    harness_ok = (old_sha == harness['v4_pin'] and defs_changed <= HARNESS_DEFS_CHANGED
                  and methods_changed <= DECLARED_ACCOUNT_METHODS_CHANGED
                  and methods_added == DECLARED_ACCOUNT_METHODS_ADDED
                  and assigns_changed <= HARNESS_ASSIGNMENTS_CHANGED and assigns_added == HARNESS_ASSIGNMENTS_ADDED
                  and not other_added and not h_removed and harness['stage_nrf_arm_identical']
                  and harness['stage_sweep_identical'] and harness['block_solve_account_identical']
                  and harness['declared_block_networks_identical'])
    # phase-A models: the passive NRF DSO arm (36 blocks, cold build) and the TSO arm (12 blocks) digest-equal old vs new
    decision = BENCH.TIE_BREAKER['decision']
    new_dso, new_b = UB.build_dso_arm_models(planning, candidate['total_capacity'], arm='passive',
                                             curtailment_penalty=decision['passive_dso'], no_reverse_flow=True)
    old_dso, old_b = old.build_dso_arm_models(planning, candidate['total_capacity'], arm='passive',
                                              curtailment_penalty=decision['passive_dso'], no_reverse_flow=True)
    dso_equal = {UB.block_label('DSO', n, y, d): V3.model_digest(new_dso[n][y][d]) == V3.model_digest(old_dso[n][y][d])
                 for n, y, d, _dn, _net in _blocks(planning)}
    build_equal = (json.dumps(new_b, sort_keys=True, default=str) == json.dumps(old_b, sort_keys=True, default=str))
    del new_dso, old_dso
    gc.collect()
    targets = UB.get_dso_interface_schedule(planning, certified['dso'])
    kw = dict(curtailment_penalty=decision['tso'], coupling=UB.TSO_COUPLING_FIXED,
              pin_interface_voltage=UB.TSO_ARM_PIN_INTERFACE_VOLTAGE)
    new_tso, _nb = UB.build_tso_arm_model(planning, candidate['total_capacity'], targets, **kw)
    old_tso, _ob = old.build_tso_arm_model(planning, candidate['total_capacity'], targets, **kw)
    tso_equal = {UB.block_label('TSO', None, y, d): V3.model_digest(new_tso[y][d]) == V3.model_digest(old_tso[y][d])
                 for y in planning.transmission_network.years for d in planning.transmission_network.days}
    del new_tso, old_tso
    gc.collect()
    # the phase-A accounting: W120's committed phase-A records through the v4 and the v5 account -- identical ledgers
    replay = _replay_phase_a_old_new(UB, planning)
    ub = {'v4_ub_sha256': v4['code_sha256_binding'][UB_FILE], 'functions_changed': changed, 'names_added': added,
          'names_removed': removed, 'module_constants_equal': constants_equal,
          'tso_path_function_sources_identical': tso_sources, 'phase_a_function_sources_identical': phase_a_sources}
    passed = (set(changed) == UB_FUNCTIONS_CHANGED and not added and not removed and constants_equal
              and all(tso_sources.values()) and all(phase_a_sources.values()) and harness_ok
              and len(dso_equal) == 36 and all(dso_equal.values()) and build_equal
              and len(tso_equal) == 12 and all(tso_equal.values()) and replay['passed'])
    return {'id': 'V16_phase_a_and_tso_path_unchanged_vs_v4', 'passed': bool(passed), 'code_commit_v4': V4_CODE_COMMIT,
            'uncoordinated_benchmark': ub, 'harness': harness, 'harness_ok': harness_ok,
            'phase_a_passive_nrf_dso_blocks_digest_equal': dso_equal, 'phase_a_dso_build_records_equal': build_equal,
            'tso_arm_blocks_digest_equal': tso_equal, 'phase_a_accounting_replay': replay,
            'digest_definition': inspect.getdoc(V3.model_digest)}


def _load_old_harness_account():
    """The v4 harness's DeclaredBlockAccount class, executed from the blob in an isolated namespace with this module's
    BlockSolveAccount-compatible dependencies (the class text only -- no module-level code of the old harness runs)."""
    blob = _git_blob(V4_CODE_COMMIT, HARNESS).decode()
    tree = ast.parse(blob)
    wanted = {'BlockSolveAccount', 'DeclaredBlockAccount'}
    namespace = {'MAX_ATTEMPTS_PER_NETWORK_SOLVE': NRF.MAX_ATTEMPTS_PER_NETWORK_SOLVE,
                 'ATTEMPT_TIERS': NRF.ATTEMPT_TIERS, 'E3_LAUNCHES_PER_FAILING_BLOCK': NRF.E3_LAUNCHES_PER_FAILING_BLOCK}
    guard_box = {}

    def _check_guard(expected, where):
        failures = guard_box['g'].verify(expected)
        if failures:
            raise RuntimeError(f'SolveProfileGuard at {where}: expected exactly {expected}: {failures}')
        return {'where': where, 'expected': expected, 'counts': dict(guard_box['g'].counts), 'verified': True}
    namespace['_check_guard'] = _check_guard
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), f'<{HARNESS}@v4>', 'exec'), namespace)  # noqa: S102
    return namespace['DeclaredBlockAccount'], guard_box


def _replay_phase_a_old_new(UB, planning):
    path = BENCH._verified_path(W120_RECORDS)
    with open(path) as handle:
        records = [json.loads(line) for line in handle]
    block_networks = NRF.declared_block_networks(UB, planning)
    dso = [k for k in block_networks if k.startswith('DSO|')]
    tso = [k for k in block_networks if k.startswith('TSO|')]
    OldAccount, box = _load_old_harness_account()
    saved = NRF._GUARD
    out = {}
    try:
        ledgers = {}
        for tag in ('v4', 'v5'):
            g = V4._FakeGuard()
            if tag == 'v4':
                box['g'] = g
                acc = OldAccount(g, 396, block_networks)
            else:
                NRF._GUARD = g
                acc = NRF.DeclaredBlockAccount(g, 396, block_networks)
            acc.open_phase('A_nrf_arm', dso + tso, ('passive:cold:dso', 'passive:cold:tso'))
            for r in records:
                g.launch(r['n_attempts'])
                acc.settle_record(r)
            closed = acc.close_phase('A_nrf_arm')
            summary = acc.summary()
            ledgers[tag] = {'ledger': acc.ledger, 'closed': closed,
                            'summary_without_v5_key': {k: v for k, v in summary.items()
                                                       if k != 'entered_but_aborted_before_record'},
                            'v5_key': summary.get('entered_but_aborted_before_record', 'absent')}
        out['n_records'] = len(records)
        out['ledgers_identical'] = ledgers['v4']['ledger'] == ledgers['v5']['ledger']
        out['phase_close_identical'] = ledgers['v4']['closed'] == ledgers['v5']['closed']
        out['summaries_identical_but_the_v5_key'] = (ledgers['v4']['summary_without_v5_key']
                                                     == ledgers['v5']['summary_without_v5_key'])
        out['v5_key_none_on_a_clean_phase'] = ledgers['v5']['v5_key'] is None
        out['attempts_attributed'] = ledgers['v5']['closed']['attempts_attributed']
        out['passed'] = bool(len(records) == 48 and out['ledgers_identical'] and out['phase_close_identical']
                             and out['summaries_identical_but_the_v5_key'] and out['v5_key_none_on_a_clean_phase']
                             and out['attempts_attributed'] == 48)
    finally:
        NRF._GUARD = saved
    return out


# ======================================================================================================================
#  V17 -- every other fixed variable of the re-evaluation clone
# ======================================================================================================================
def v17_fixed_variable_scan(UB, planning, candidate, warm, old):
    import pyomo.environ as pe
    from pyomo.core.expr.visitor import identify_variables
    from pyomo.repn import ampl as pyomo_ampl
    tol_writer = float(pyomo_ampl.TOL)
    v_obs = W120_OBSERVED_V_DN_PU
    decisions = set(UB.DSO_DECISION_VAR_FAMILIES)
    zero = set(UB.DSO_REEVALUATION_ZERO_SLACKS)
    per_arm = {}
    for arm in UB.DSO_ARMS:
        models, _build, starts = _arm_models_warm(UB, planning, candidate, warm, arm)
        fam = {}
        old_outside = {}
        for node_id, year, day, _dn, network in _blocks(planning):
            block = models[node_id][year][day]
            ref_idx = network.get_node_idx(network.get_reference_node_id())
            ref_gen = network.get_reference_gen_idx()
            fixed_in_arm = {v.name for v in block.component_data_objects(pe.Var, descend_into=True) if v.fixed}
            v = [v_obs] * len(block.periods)
            c, rec = UB.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v)
            referenced = set()
            for con in c.component_data_objects(pe.Constraint, active=True, descend_into=True):
                referenced.update(id(x) for x in identify_variables(con.body, include_fixed=True))
            for obj in c.component_data_objects(pe.Objective, active=True, descend_into=True):
                referenced.update(id(x) for x in identify_variables(obj.expr, include_fixed=True))
            for var in c.component_data_objects(pe.Var, descend_into=True):
                if not var.fixed:
                    continue
                family = var.parent_component().name
                idx = var.index()
                if family in ('e', 'f') and isinstance(idx, tuple) and idx[0] == ref_idx:
                    cls = 'reference_voltage_setpoint'
                elif var.name in fixed_in_arm:
                    cls = 'fixed_before_the_clone'
                elif family in zero:
                    cls = 'zero_slack'
                elif family in decisions or (family in ('pg', 'qg') and idx[0] != ref_gen):
                    cls = 'dso_decision_at_arm_solution'
                else:
                    cls = 'other'
                value = float(var.value)
                lb, ub = var.lb, var.ub
                excess = max((lb - value) if lb is not None else 0.0, (value - ub) if ub is not None else 0.0, 0.0)
                bmax = max([abs(x) for x in (lb, ub) if x is not None], default=0.0)
                row = fam.setdefault(f'{cls}:{family}', {
                    'class': cls, 'family': family, 'n_fixed': 0, 'n_in_active_rows_or_objective': 0,
                    'n_outside_bounds': 0, 'n_outside_bounds_above_writer_tol': 0, 'max_excess': 0.0,
                    'max_abs_finite_bound': 0.0, 'n_with_abs_bound_above_1': 0, 'n_unbounded': 0,
                    'value_range': [math.inf, -math.inf]})
                row['n_fixed'] += 1
                row['n_in_active_rows_or_objective'] += int(id(var) in referenced)
                row['n_outside_bounds'] += int(excess > 0.0)
                row['n_outside_bounds_above_writer_tol'] += int(excess > tol_writer)
                row['max_excess'] = max(row['max_excess'], excess)
                row['max_abs_finite_bound'] = max(row['max_abs_finite_bound'], bmax)
                row['n_with_abs_bound_above_1'] += int(bmax > 1.0)
                row['n_unbounded'] += int(lb is None and ub is None)
                row['value_range'] = [min(row['value_range'][0], value), max(row['value_range'][1], value)]
            del c
            # the OLD clone: which fixed Vars lie outside their bounds by more than the writer's tolerance
            c_old, _ro = old.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v)
            for var in c_old.component_data_objects(pe.Var, descend_into=True):
                if var.fixed:
                    value = float(var.value)
                    ex = max((var.lb - value) if var.lb is not None else 0.0,
                             (value - var.ub) if var.ub is not None else 0.0, 0.0)
                    if ex > tol_writer:
                        key = var.parent_component().name
                        old_outside[key] = old_outside.get(key, 0) + 1
            del c_old
        per_arm[arm] = {'families': dict(sorted(fam.items())),
                        'old_clone_fixed_outside_writer_tol_by_family': old_outside,
                        'warm_n_outside_arm_bounds_max': max(r['warm']['n_outside_arm_bounds'] for r in starts.values()),
                        'new_clone_fixed_outside_writer_tol_total': sum(r['n_outside_bounds_above_writer_tol']
                                                                        for r in fam.values()),
                        'classes_present': sorted({r['class'] for r in fam.values()})}
        del models
        gc.collect()
    ipopt = _ipopt_bound_violation_records(tol_writer)
    options = _ipopt_bound_options_w120()
    relax = 1e-8      # IPOPT 3.14 default bound_relax_factor (not set in the arm options; see ipopt_bound_options)
    hazards = {}
    for arm, p in per_arm.items():
        for key, row in p['families'].items():
            if row['class'] != 'dso_decision_at_arm_solution':
                continue
            potential = relax * max(1.0, row['max_abs_finite_bound'])
            if potential > tol_writer:
                hazards.setdefault(row['family'], {'max_abs_finite_bound': row['max_abs_finite_bound'],
                                                   'n_fixed_with_abs_bound_above_1': row['n_with_abs_bound_above_1'],
                                                   'n_in_active_rows_or_objective': row['n_in_active_rows_or_objective'],
                                                   'potential_ipopt_relaxation': potential,
                                                   'max_excess_observed_warm': {}})['max_excess_observed_warm'][arm] = (
                    row['max_excess'])
    passed = (all(p['new_clone_fixed_outside_writer_tol_total'] == 0
                  and p['old_clone_fixed_outside_writer_tol_by_family'] == {'e': 36 * 24}
                  and 'other' not in p['classes_present'] for p in per_arm.values())
              and options['sha_verified'])
    return {'id': 'V17_fixed_variable_scan', 'passed': bool(passed), 'pyomo_nl_writer_tol_abs': tol_writer,
            'physical_bound_hazard': {
                'rule': ('a DSO decision fixed at an IPOPT arm solution may lie up to bound_relax_factor x max(1, |b|) '
                         'outside a PHYSICAL bound b (honor_original_bounds = no); the NL writer raises above its '
                         'absolute TOL. Families where that potential exceeds TOL are listed (reported, not changed: '
                         'physical bounds)'),
                'families': hazards, 'ipopt_bound_options': options},
            'interface_q_note': ('no interface-Q Var is fixed in the clone: the reference generator\'s P / Q are FREED '
                                 '(original bounds recorded and checked as hard) and expected_interface_pf_q is free'),
            'observed_v_dn_pu': v_obs,
            'classification': ('reference_voltage_setpoint = e / f at the reference bus (SETPOINT: cleared by the fix); '
                               'dso_decision_at_arm_solution = DSO_DECISION_VAR_FAMILIES and non-reference pg / qg '
                               '(PHYSICAL bounds, fixed at the arm solution: REPORTED); zero_slack = '
                               'DSO_REEVALUATION_ZERO_SLACKS at 0.0; fixed_before_the_clone = fixed in the arm block '
                               'already; other = none expected'),
            'values_source': ('the NRF arm DSO blocks at the warm_from_certified start: every free Var at the settled '
                              'Q181 IPOPT solution (not an arm solution; the arm solutions do not exist zero-solve)'),
            'per_arm': per_arm, 'ipopt_recorded_variable_bound_violation': ipopt}


def _ipopt_bound_options_w120():
    """The IPOPT options list of W120's phase-A log of the DSO block case33_1 2025 Spring (the block whose re-evaluation
    failed), sha-verified against W120's IPOPT log hash record: bound_relax_factor / honor_original_bounds are NOT set,
    so IPOPT 3.14.18's defaults (1e-8 / no) are in force."""
    rel = os.path.join(P56A_WORK_DIR_REL, 'p515s53w116_nrf_arm_passive_cold', 'logs', 'optim_log_case33_1_2025_Spring.log')
    record = NRF._load_verified_json(NRF.CONSUMED_RUNS_V4['nrf_arm_passive_cold']['ipopt_logs_hash_record'])
    text = open(_abs(rel)).read()
    head = text.split('*' * 20, 1)[0]
    version = next((line.strip() for line in text.splitlines() if line.startswith('This is Ipopt version')), None)
    return {'log': rel, 'sha_verified': record.get(rel) == _sha(rel),
            'bound_relax_factor_set': 'bound_relax_factor' in head,
            'honor_original_bounds_set': 'honor_original_bounds' in head,
            'ipopt_version_line': version,
            'defaults_in_force': 'IPOPT 3.14 defaults: bound_relax_factor = 1e-8, honor_original_bounds = no',
            'final_variable_bound_violation_line': next((line.strip() for line in text.splitlines()
                                                         if line.startswith('Variable bound violation')), None)}


def _ipopt_bound_violation_records(tol_writer):
    """IPOPT's own final 'Variable bound violation' (unscaled; the returned point against the ORIGINAL bounds,
    honor_original_bounds = no) of every DSO solve in the committed W120 records: the margin by which fixing an arm
    solution could trip the NL writer's absolute tolerance."""
    sources = {'nrf_arm_passive_cold_phase_A': W120_RECORDS,
               **{f'{run_id}_dso': NRF.SWEEPS_DONE_W120[run_id]['per_solve_record'] for run_id in NRF.SWEEP_RUN_IDS}}
    out = {}
    for name, entry in sources.items():
        path = BENCH._verified_path(entry)
        vals = []
        with open(path) as handle:
            for line in handle:
                r = json.loads(line)
                if r['kind'] != 'DSO':
                    continue
                last = r['attempts'][-1].get('final_summary') or {}
                vals.append((last.get('unscaled') or {}).get('variable_bound_violation'))
        finite = [v for v in vals if v is not None]
        out[name] = {'n_dso_records': len(vals), 'n_parsed': len(finite), 'max': max(finite, default=None),
                     'n_above_writer_tol': sum(1 for v in finite if v > tol_writer),
                     'margin_below_writer_tol': None if not finite else tol_writer - max(finite)}
    return out


# ======================================================================================================================
#  V18 -- the aborted-entry accounting
# ======================================================================================================================
def v18_aborted_entry_accounting(UB, planning):
    nets = {'DSO|5|2025|Spring': ('case33_1', '2025', 'Spring'), 'DSO|5|2025|Summer': ('case33_1', '2025', 'Summer')}
    labels = list(nets)
    saved = NRF._GUARD
    res = {}

    def rec(label, n=1, phase='consistency:reevaluation'):
        name, year, day = nets[label]
        return {'block': label, 'kind': 'DSO', 'phase': phase, 'n_attempts': n, 'succeeded': True,
                'attempts': [{'attempt': t, 'network': name, 'year': int(year), 'day': day}
                             for t in NRF.ATTEMPT_TIERS[:n]]}

    def expect_raise(fn, *contains):
        try:
            fn()
            return False, None
        except RuntimeError as error:
            return all(c in str(error) for c in contains), str(error)[:400]

    try:
        # stubs: the W120 shape -- a phase, one aborted entry (solve entered, no launch, no record)
        g = V4._FakeGuard()
        NRF._GUARD = g
        acc = NRF.DeclaredBlockAccount(g, 100, nets)
        acc.open_phase('B', labels, ('consistency:reevaluation',))
        clean = acc.summary()['entered_but_aborted_before_record']
        g.launch(1, exec_n=0)
        reported = acc.summary()['entered_but_aborted_before_record']
        res['clean_phase_reports_none'] = clean is None
        res['aborted_entry_reported_explicitly'] = (
            reported is not None and reported['status'] == NRF.ABORTED_ENTRY_STATUS
            and reported['solves_entered'] == 1 and reported['process_launches'] == 0 and reported['launched'] is False
            and reported['phase'] == 'B' and reported['block_in_progress'] == labels[0] and reported['absorbed'] is False)
        res['aborted_entry_record'] = reported
        ok, msg = expect_raise(lambda: (g.launch(1), acc.settle_record(rec(labels[0]))), NRF.ABORTED_ENTRY_STATUS,
                               'guard delta')
        res['NEGATIVE_a_later_record_raises_naming_it'] = ok
        res['later_record_error'] = msg
        g = V4._FakeGuard()
        NRF._GUARD = g
        acc = NRF.DeclaredBlockAccount(g, 100, nets)
        acc.open_phase('B', labels, ('consistency:reevaluation',))
        g.launch(1)
        acc.settle_record(rec(labels[0]))
        g.launch(1, exec_n=0)
        ok, msg = expect_raise(lambda: acc.close_phase('B'), NRF.ABORTED_ENTRY_STATUS, 'SolveProfileGuard')
        res['NEGATIVE_close_phase_raises_naming_it'] = ok
        res['close_error'] = msg
        g = V4._FakeGuard()
        NRF._GUARD = g
        acc = NRF.DeclaredBlockAccount(g, 100, nets)
        g.launch(1, exec_n=0)
        ok, msg = expect_raise(lambda: acc.open_phase('B', labels, ('consistency:reevaluation',)),
                               NRF.ABORTED_ENTRY_STATUS, 'SolveProfileGuard')
        res['NEGATIVE_open_phase_raises_naming_it'] = ok
        g = V4._FakeGuard()
        NRF._GUARD = g
        acc = NRF.DeclaredBlockAccount(g, 100, nets)
        acc.open_phase('B', labels, ('consistency:reevaluation',))
        g.launch(1)
        launched = acc.summary()['entered_but_aborted_before_record']
        res['launched_but_unrecorded_reported'] = (launched is not None and launched['launched'] is True
                                                   and launched['process_launches'] == 1)
        # the committed W120 records: phase A settles (48), phase B opens, one aborted entry -> W120's guard counts
        path = BENCH._verified_path(W120_RECORDS)
        with open(path) as handle:
            records = [json.loads(line) for line in handle]
        block_networks = NRF.declared_block_networks(UB, planning)
        dso = [k for k in block_networks if k.startswith('DSO|')]
        tso = [k for k in block_networks if k.startswith('TSO|')]
        g = V4._FakeGuard()
        NRF._GUARD = g
        acc = NRF.DeclaredBlockAccount(g, 396, block_networks)
        acc.open_phase('A_nrf_arm', dso + tso, ('passive:cold:dso', 'passive:cold:tso'))
        for r in records:
            g.launch(r['n_attempts'])
            acc.settle_record(r)
        acc.close_phase('A_nrf_arm')
        acc.open_phase('B_consistency_reevaluation', dso, ('consistency:reevaluation',))
        g.launch(1, exec_n=0)
        summary = acc.summary()
        w120 = summary['entered_but_aborted_before_record']
        first_b = next(k for k, v in block_networks.items() if v == W120_FAILURE_BLOCK)
        failure = NRF._load_verified_json(W120_FAILURE)
        res['w120_replay'] = {
            'n_records': len(records), 'guard_counts': dict(g.counts), 'attributed': summary['attempts_attributed_total'],
            'aborted_entry': w120, 'first_phase_b_block': first_b,
            'guard_counts_equal_w120_failure': dict(g.counts) == failure['solve_profile_guard_counts'] == W120_GUARD_COUNTS,
            'w120_ledger_did_not_name_it': 'entered_but_aborted_before_record' not in failure['solve_accounting'],
            'passed': bool(w120 is not None and w120['solves_entered'] == 1 and w120['process_launches'] == 0
                           and w120['phase'] == 'B_consistency_reevaluation' and w120['block_in_progress'] == first_b
                           and dso[0] == first_b and summary['attempts_attributed_total'] == 48
                           and dict(g.counts) == failure['solve_profile_guard_counts'])}
    finally:
        NRF._GUARD = saved
    main_src = inspect.getsource(NRF.main)
    wiring = {'failure_json_carries_the_aborted_entry': "'solve_accounting_entered_but_aborted_before_record'" in main_src,
              'failure_json_carries_the_ledger': "'solve_accounting': _ACCOUNT.summary()" in main_src,
              'logged': "_log(f'{run_id}: solve accounting: {_ACCOUNT.aborted_entry()}')" in main_src,
              'stage_does_not_catch_inside': 'except' not in inspect.getsource(NRF.stage_nrf_arm)}
    passed = (res['clean_phase_reports_none'] and res['aborted_entry_reported_explicitly']
              and res['NEGATIVE_a_later_record_raises_naming_it'] and res['NEGATIVE_close_phase_raises_naming_it']
              and res['NEGATIVE_open_phase_raises_naming_it'] and res['launched_but_unrecorded_reported']
              and res['w120_replay']['passed'] and all(wiring.values()))
    return {'id': 'V18_aborted_entry_accounting', 'passed': bool(passed), 'controls': res, 'wiring': wiring,
            'status_text': NRF.ABORTED_ENTRY_STATUS}


# ======================================================================================================================
#  the re-run checks, adapted to v5
# ======================================================================================================================
def v8_spec_binding():
    failures = NRF.frozen_spec_binding_failures()
    latest = NRF._latest_frozen_spec()
    negatives = {}
    if latest is not None:
        path, version, _hash8 = latest
        tmp = tempfile.mkdtemp(prefix='w121_spec_neg_')
        try:
            spec = json.load(open(path))
            spec['code_sha256_binding'][HARNESS] = '0' * 64
            data = (GRIO.dumps(spec, indent=1, sort_keys=True) + '\n').encode()
            sha = hashlib.sha256(data).hexdigest()
            name = os.path.join(tmp, f'frozen_s53_benchmark_spec_v{version}_{sha[:8]}.json')
            with open(name, 'wb') as handle:
                handle.write(data)
            negatives['altered_pin_refused'] = any(HARNESS in f for f in NRF.frozen_spec_binding_failures(tmp))
            os.remove(name)
            bad = os.path.join(tmp, f'frozen_s53_benchmark_spec_v{version}_00000000.json')
            shutil.copyfile(path, bad)
            negatives['name_hash_mismatch_refused'] = any('does not start with its name hash' in f
                                                          for f in NRF.frozen_spec_binding_failures(tmp))
            os.remove(bad)
            shutil.copyfile(_abs(V4_SPEC['path']), os.path.join(tmp, os.path.basename(V4_SPEC['path'])))
            v4_fail = NRF.frozen_spec_binding_failures(tmp)
            negatives['v4_spec_refused_version_and_pins'] = (any('version 4 < 5' in f for f in v4_fail)
                                                             and any(HARNESS in f for f in v4_fail)
                                                             and any(UB_FILE in f for f in v4_fail))
            shutil.copyfile(path, os.path.join(tmp, os.path.basename(path)))
            both = NRF._latest_frozen_spec(tmp)
            negatives['v5_selected_over_v4_in_shared_root'] = (both is not None and both[1] == 5
                                                               and NRF.frozen_spec_binding_failures(tmp) == [])
            for f in os.listdir(tmp):
                os.remove(os.path.join(tmp, f))
            shutil.copyfile(_abs(V4.V3_SPEC['path']), os.path.join(tmp, os.path.basename(V4.V3_SPEC['path'])))
            v3_fail = NRF.frozen_spec_binding_failures(tmp)
            negatives['v3_spec_refused'] = any('version 3 < 5' in f for f in v3_fail)
        finally:
            shutil.rmtree(tmp)
    passed = (latest is not None and latest[1] == 5 and not failures and bool(negatives) and all(negatives.values())
              and NRF.FROZEN_SPEC_MIN_VERSION == 5)
    return {'id': 'V8_frozen_spec_v5_binds', 'passed': bool(passed),
            'spec': None if latest is None else {'path': os.path.relpath(latest[0], REPO), 'version': latest[1],
                                                 'sha256': H.sha256_file(latest[0])},
            'frozen_spec_min_version': NRF.FROZEN_SPEC_MIN_VERSION, 'binding_failures': failures,
            'NEGATIVE': negatives}


def v9_spec_diff():
    v4 = W111._load_verified_json(V4_SPEC)
    latest, v5 = _latest_spec()
    if latest[1] != 5:
        return {'id': 'V9_v4_v5_key_diff', 'passed': False, 'reason': f'latest spec under the root is {latest}'}
    diff = W111.key_diff(v4, v5)
    top = sorted({x['path'].split('.')[0].split('[')[0] for x in diff})
    v5_wo = {k: v for k, v in v5.items() if k != 'v4_to_v5_key_diff'}
    embedded_equal = (v5['v4_to_v5_key_diff']['paths'] == W111.key_diff(v4, v5_wo))
    pins4, pins5 = v4['code_sha256_binding'], v5['code_sha256_binding']
    stage_rows, stages_ok = [], len(v4['stages_v3_addendum_57_order']) == len(v5['stages_v3_addendum_57_order'])
    for s4, s5 in zip(v4['stages_v3_addendum_57_order'], v5['stages_v3_addendum_57_order']):
        differs = sorted(k for k in set(s4) | set(s5) if s4.get(k) != s5.get(k))
        allowed = V5_STAGE_FIELDS[s4['stage']]
        ok = set(differs) <= set(allowed)
        if s4['stage'] in ('nrf-arm', 'nrf-passive-tie-breaker'):
            ok = ok and (s5['run_id'] == s4['run_id'] + NRF.RUN_ID_SUFFIX_V5
                         and s5['replaces_v4_run_id'] == s4['run_id']
                         and _argv_of(s5['command']) == _argv_of(s4['command'])
                         and s5['command'] == s4['command'].replace(f"/{s4['run_id']}.log", f"/{s5['run_id']}.log"))
        if s4['stage'] == 'report':
            ok = ok and s5['command'] == s4['command'] and s5['run_id'] == s4['run_id']
        stages_ok = stages_ok and ok
        stage_rows.append({'run_id_v4': s4['run_id'], 'run_id_v5': s5['run_id'], 'stage': s4['stage'],
                           'fields_differing': differs, 'ok': ok})
    remaining = [r['run_id'] for r in v5['remaining_stage_commands_v5']['commands_in_order']]
    checks = {
        'v4_sha256_unchanged': _sha(V4_SPEC['path']) == V4_SPEC['sha256'],
        'v4_file_clean_in_git': _git(['status', '--porcelain', '--', V4_SPEC['path']]) == '',
        'v5_predecessor_is_v4': (v5.get('predecessor') or {}).get('sha256') == V4_SPEC['sha256'],
        'changed_top_keys_within_declared_set': set(top) <= set(V5_CHANGED_OR_ADDED_TOP_KEYS),
        'every_other_v4_key_identical': all(v4[k] == v5.get(k) for k in v4 if k not in V5_CHANGED_OR_ADDED_TOP_KEYS),
        'no_v4_key_removed': all(k in v5 for k in v4),
        'binding_differs_only_in_harness_and_ub_pins': (sorted(pins4) == sorted(pins5) and sorted(
            k for k in pins4 if pins4[k] != pins5[k]) == V5_PINS_CHANGED),
        'stage_entries_differ_only_in_declared_fields': stages_ok,
        'remaining_commands_in_order': remaining == NRF.NRF_ARM_RUN_IDS + NRF.NRF_VARIANT_RUN_IDS + [NRF.REPORT_RUN_ID],
        'remaining_commands_equal_stage_entries': all(
            r['command'] == next(s['command'] for s in v5['stages_v3_addendum_57_order'] if s['run_id'] == r['run_id'])
            for r in v5['remaining_stage_commands_v5']['commands_in_order']),
        'embedded_key_diff_equals_recomputed': embedded_equal,
        'solve_counts_identical': v4['solve_counts'] == v5['solve_counts'],
        'predictions_v3_v4_carried_identical': all(v4[k] == v5[k] for k in ('predictions_recorded_before_any_run_v3',
                                                                            'predictions_recorded_before_any_run_v4')),
        'rulings_w119_carried_identical': v4['planner_rulings_w119'] == v5['planner_rulings_w119'],
        'output_root_same': v5.get('output_root') == v4.get('output_root') == OUT_ROOT_REL,
        'shared_root_outputs_exactly_w120': stage_outputs_v5()['exactly_w120'],
    }
    return {'id': 'V9_v4_v5_key_diff', 'passed': all(checks.values()), 'checks': checks, 'v4': dict(V4_SPEC),
            'v5': {'path': os.path.relpath(latest[0], REPO), 'sha256': H.sha256_file(latest[0])},
            'n_paths': len(diff), 'changed_top_keys': top, 'stage_entries': stage_rows,
            'top_level': [x for x in diff if '.' not in x['path'] and '[' not in x['path']], 'diff': diff,
            'stage_outputs': stage_outputs_v5()}


def v14_spec_v5_content():
    v4 = W111._load_verified_json(V4_SPEC)
    latest, spec = _latest_spec()
    recomputed = predictions_scored_so_far(v4)
    scoring, scoring_ok = V4._synthetic_scoring(spec)
    report_src = inspect.getsource(NRF.stage_report)
    w120 = _w120_committed_shas()
    fix = spec.get('reevaluation_fix_v5') or {}
    checks = {
        'scores_so_far_recorded_equal_recomputed': spec.get('predictions_scored_so_far_v5') == json.loads(
            GRIO.dumps(recomputed)),
        'both_sweep_predictions_held': recomputed['both_sweep_predictions_held'] is True,
        'planner_predictions_carried': (spec['predictions_recorded_before_any_run_v4']
                                        == v4['predictions_recorded_before_any_run_v4']),
        'worker_expectations_carried': (spec['predictions_recorded_before_any_run_v3']
                                        == v4['predictions_recorded_before_any_run_v3']),
        'report_scores_planner_predictions': "'predictions_recorded_before_any_run_v4'" in inspect.getsource(
            NRF._score_predictions),
        'scoring_exercised_synthetic': scoring_ok,
        'report_reads_sweeps_sha_verified': ("_load_verified_json(SWEEPS_DONE_W120[run_id]['result'])" in report_src),
        'report_reads_r2_arm_ids': 'RUN_ID_SUFFIX_V5' in report_src,
        'w120_evidence_verified': w120['all_verified'],
        'sweeps_done_recorded_equal_harness_pins': all(
            spec['sweeps_done_v5'][r][k] == NRF.SWEEPS_DONE_W120[r][k]
            for r in NRF.SWEEP_RUN_IDS for k in ('result', 'per_solve_record', 'manifest', 'committed_in')),
        'consumed_recorded': set(spec.get('consumed_runs_v5') or {}) == set(NRF.CONSUMED_RUNS_V4),
        'ruling_recorded': (fix.get('ruling') or {}).get('label') == RULING_LABEL,
        'reference_limits_recorded': ((fix.get('dn_reference_bus_physical_limits') or {}).get(
            'srp1_reference_vmin_vmax_distinct') == [[0.9, 1.1]]),
        'accounting_v5_recorded': NRF.ABORTED_ENTRY_STATUS in (spec.get('solve_accounting_v5') or {}).get(
            'implemented_in', ''),
    }
    return {'id': 'V14_spec_v5_content', 'passed': all(checks.values()), 'checks': checks,
            'spec': {'path': os.path.relpath(latest[0], REPO), 'sha256': H.sha256_file(latest[0])},
            'scores_so_far': {k: recomputed[k] for k in ('sweep_passive', 'sweep_price_taker', 'nrf_claim',
                                                         'nrf_arms_feasible_at_every_tso_block')},
            'synthetic_scoring_outcomes': scoring, 'w120_evidence_mismatches': w120['mismatches']}


def run_checks(suffix=''):
    out_dir = _abs(CHECKS_DIR_REL + suffix)
    if os.path.exists(out_dir):
        print(f'REFUSING: output exists (write-once): {out_dir}', flush=True)
        return 2
    os.makedirs(out_dir)
    results, timings, extra = [], {}, {}

    def record(fn, *args, new_id=None):
        t = time.time()
        try:
            res = fn(*args)
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            res = {'id': new_id or getattr(fn, '__name__', 'check'), 'passed': False,
                   'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        payload = res[0] if isinstance(res, tuple) else res
        if new_id is not None and payload.get('id') != new_id:
            payload = _renamed(payload, new_id)
        timings[payload.get('id', 'check')] = time.time() - t
        _log(f"{payload.get('id')}: {'PASS' if payload.get('passed') is True else 'FAIL'}")
        results.append(payload)
        return res

    record(V3.v0_preconditions, new_id='V0_preconditions')
    record(V3.v1_model_hash, new_id='V1_model_hash')
    record(v8_spec_binding)
    record(v9_spec_diff)
    record(V3.v10_production_unchanged, new_id='V10_no_production_change')
    try:
        w106_c6 = W106.c6_eval_keys()
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        w106_c6 = {'id': 'C6_committed_eval_keys_unchanged', 'passed': False, 'error': f'{type(error).__name__}: {error}'}
    extra['C6_w106_as_written_informational'] = {k: w106_c6.get(k) for k in ('id', 'passed', 'committed_specs', 'error')}
    record(V4.v7_eval_keys_head_tree, w106_c6, new_id='V7_committed_eval_keys_unchanged_head_tree')
    import shared_resources_planning as srp          # production imports only after the guards (armed at import)
    import uncoordinated_benchmark as UB
    import p56a_oracle as O
    extra['p56a_work_dir_matches'] = os.path.relpath(O.WORK_DIR, REPO) == P56A_WORK_DIR_REL
    t = time.time()
    planning = O.load_baseline()['planning']
    timings['load_baseline_planning_s'] = time.time() - t
    candidate = BENCH._x0_candidate(srp, planning)
    t = time.time()
    certified = BENCH._load_pickle_verified(BENCH.COORDINATED['certified_models'])
    timings['unpickle_settled_certified_models_s'] = time.time() - t
    reference = UB.coordinated_reference_structure(planning, certified)
    warm = UB.extract_model_values(planning, certified)
    old, old_sha, old_tmp = _load_old_ub()
    extra['old_ub'] = {'commit': V4_CODE_COMMIT, 'sha256': old_sha}
    try:
        record(V4.v4_reverse_flow_count, UB, srp, planning, certified, out_dir, new_id='V4_reverse_flow_count_q181')
        record(V4.v13_nrf_block_accounting, UB, planning, new_id='V13_nrf_per_block_accounting')
        record(v18_aborted_entry_accounting, UB, planning)
        record(v15_reevaluation_fix, UB, planning, candidate, certified, warm, old)
        record(v17_fixed_variable_scan, UB, planning, candidate, warm, old)
        record(v16_phase_a_and_tso_unchanged, UB, planning, candidate, certified, old)
    finally:
        sys.modules.pop('_ub_at_spec_v4', None)
        shutil.rmtree(old_tmp)
    v2res = record(V3.v2_nrf_rows, UB, srp, planning, candidate, reference, certified, warm,
                   new_id='V2_nrf_rows_present_and_absent')
    digests = v2res[1] if isinstance(v2res, tuple) else None
    if digests is not None:
        record(V3.v3_tso_unchanged, UB, planning, candidate, reference, certified, digests,
               new_id='V3_tso_arm_unchanged_vs_v2')
    else:
        results.append({'id': 'V3_tso_arm_unchanged_vs_v2', 'passed': False, 'reason': 'V2 did not return digests'})
    del warm
    gc.collect()
    record(V3.v5_sweep_accounting, UB, planning, candidate, certified, new_id='V5_sweep_solve_accounting')
    del certified
    gc.collect()
    record(V3.v11_report_capture, os.path.join(out_dir, os.path.basename(NRF.REVERSE_FLOW_OUTPUT_REL)),
           new_id='V11_report_capture_paths')
    record(V3.v12_stage_wiring, new_id='V12_stage_wiring')
    record(v14_spec_v5_content)
    record(V3.v6_typing_test, out_dir, new_id='V6_w100_bool_typing_test')
    guards = _all_guard_failures()
    passed = (all(r.get('passed') is True for r in results) and not any(guards.values())
              and extra['p56a_work_dir_matches'])
    payload = {'stage': 'P5.15 W121 zero-solve checks (benchmark spec v5)', 'utc': _utc(),
               'git_head': _git(['rev-parse', 'HEAD']), 'script_sha256': _sha(THIS), 'harness_sha256': _sha(HARNESS),
               'uncoordinated_benchmark_sha256': _sha(UB_FILE), 'v4_script_sha256': _sha(V4_SCRIPT),
               'solve_profile_guard': {'permitted': [], 'verify_0': guards, 'counts_w121': dict(_GUARD.counts),
                                       'counts_w119_import': dict(V4._GUARD.counts),
                                       'counts_w116_import': dict(V3._GUARD.counts),
                                       'counts_w111_import': dict(W111._GUARD.counts),
                                       'counts_w106_import': dict(W106._GUARD.counts)},
               'passed': bool(passed), 'n_checks': len(results),
               'failed': [r.get('id') for r in results if r.get('passed') is not True],
               'timings_s': timings, 'results': results, 'output_suffix': suffix, **extra}
    with open(os.path.join(out_dir, 'w121_zero_solve_checks.json'), 'x') as handle:
        GRIO.dump(payload, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    manifest = {}
    for root, _dirs, files in os.walk(out_dir):
        for name in sorted(files):
            manifest[os.path.relpath(os.path.join(root, name), REPO)] = H.sha256_file(os.path.join(root, name))
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1, sort_keys=True)
    _log(f"W121 checks: {'ALL PASS' if passed else 'FAIL ' + str(payload['failed'])}; guards {guards}; counts "
         f'W121 {dict(_GUARD.counts)}')
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--freeze-spec', action='store_true')
    group.add_argument('--checks', action='store_true')
    parser.add_argument('--output-suffix', default='', choices=('', '_r2', '_r3'),
                        help='checks output directory suffix (write-once; earlier runs are kept as evidence)')
    args = parser.parse_args(argv)
    try:
        return freeze_spec() if args.freeze_spec else run_checks(args.output_suffix)
    finally:
        W106._GUARD.uninstall()        # LIFO: W106 (top), W111, W116 (v3 script), W119 (v4 script), W121
        W111._GUARD.uninstall()
        V3._GUARD.uninstall()
        V4._GUARD.uninstall()
        _GUARD.uninstall()


if __name__ == '__main__':
    sys.exit(main())
