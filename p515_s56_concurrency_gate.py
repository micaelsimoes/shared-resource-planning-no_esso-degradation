"""
P5.15 S56 (design note v1, D6) -- the SERIAL-vs-CONCURRENT BITWISE EQUIVALENCE GATE for candidate-level poll
parallelism.

Authority: S56_DESIGN_NOTE_v1.md section "Parallelism (D6)" ("Equivalence gate, bitwise. Over 3 cycles, compare one
candidate run alone and the same candidate inside a full batch, in separate roots, on the per-cycle record"); the
author's request ("compare representative serial and parallel evaluations using the same inputs and numerical
criteria"); Planner task (S56 concurrency gate); TASKS.md S56 order.

WHAT IT RUNS. The EXISTING campaign harness `p515_s44_campaign_harness.py`, unchanged (sha256 pinned), through its own
public functions (`freeze_campaign_spec`, `check_campaign_preconditions`, `acquire_campaign_lock`, `evaluate`,
`release_campaign_lock`), on the published SRP1 single-scenario configuration as the current production declares it
(the W187 screen's `x0_A_1x` configuration block: case file with AA keep_memory declared, ESS ageing BASELINE declared,
convergence-depth tail declared ENABLED compl_inf_tol 1e-6, 1x flexibility price, no overrides, no options), with the
harness's own cycle cap (`spec['cap']` -> `run_admm_arm(num_max_iters_override=cap)`, the mechanism of the W90/W91
3-cycle smokes) set to 3:
  root A (serial):     concurrency 1, ONE evaluation: the reference unit (node 7, 0.25 MVA / 1 MWh, 2025);
  root B (concurrent): concurrency 5, FIVE evaluations in one batch: the reference unit (first) + x = 0 +
                       n5 0.25/0.5 + n9 0.25/1 + n7 0.5/1 (all 2025; every one a label of the frozen s47 Phase B domain).
A then B, sequentially, in ONE attached process, each under the harness's campaign lock (acquired and released per
root), never concurrently with another campaign.

THE GATE (bitwise; the frozen gate spec lists every compared and every excluded field, declared before the run):
  G0 test condition: A ran at max concurrency 1, B at max concurrency 5; every child exit 0, status not_certified,
     cycles_run 3.
  G1 per-cycle record of the shared candidate: `g_s39_D.json` cycle_trajectory (every field of every row),
     `per_cycle_record.jsonl`, `aa_per_cycle.jsonl` -- no exclusions.
  G2 every IPOPT solve of the shared candidate: `network_ipopt_solve_records.jsonl` (every field except log_path and
     log_bytes) and, from every IPOPT log of its working dir (network + ESSO), the final-statistics block of every solve
     (iterations, objective, dual infeasibility, constraint / bound violation, complementarity, NLP error, function
     evaluation counts, EXIT message) minus the timing lines.
  G3 the evaluation record of the shared candidate, minus the declared excluded fields (wall times, PIDs, paths,
     timestamps, root identity, process memory, hashes of files that embed paths).
  G4 solve counts: each of the 6 evaluations' armed, bounded child guard (`run_admm_arm`'s SolveProfileGuard) reconciles
     to 51 x (cap 3 + 1) = 204 + every attempted retry; the parent never solves (SolveProfileGuard(permitted=()),
     verify(0)).
Equality is bitwise: floats compared by their IEEE-754 bytes, types exact, key sets equal, list lengths equal.
Supplementary (reported, NOT gated): every other JSON/JSONL artifact of the shared candidate's eval dir with root
tokens normalised; the full IPOPT log text with root tokens normalised and timing lines removed; an informational
read-only cross-check of cycles 1-3 against committed evaluations with the same eval key.

MEASURED (reported against the recorded predictions P2, P3): per-cycle wall of the shared candidate in A and B
(production's own "Iteration k: X s" lines) and of every B child; peak RSS per child (`peak_rss.
child_python_process_ru_maxrss`, wait4 ru_maxrss); footprint per child (`ri_lifetime_max_phys_footprint`, the W91
ChildFootprintMonitor, copied here, not imported); machine memory available during B (vm_stat rule of the S47 launcher,
sampled every 5 s).

MODES (repo root, canonical interpreter, attached, ALONE, both streams captured, never detached):
  --freeze   ZERO SOLVES. Creates data/SRP1/Results/P515S56/concurrency_gate_v1/ (refuses if it exists) and writes
             frozen_concurrency_gate_spec_v1_<sha8>.json: roots, candidates (eval keys from scratch freezes of the
             two campaign specs in a temporary directory outside the repo), cap, compared / excluded fields,
             solve counts, predictions, commands, the capture-path checklist and the comparator self-test on committed
             artifacts.
               set -o noclobber && NLP_SOLVER_PATH=/usr/local/bin/ipopt \\
                 /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_concurrency_gate.py --freeze \\
                 > data/SRP1/Results/P515S56/concurrency_gate_v1_freeze_launch.log 2>&1
  --run --spec-sha256 S   The two roots, then the comparison; writes gate_result.json (gate_result_io) and
             manifest_sha256.json into the gate directory (write-once).
               set -o noclobber && NLP_SOLVER_PATH=/usr/local/bin/ipopt \\
                 /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_concurrency_gate.py --run \\
                 --spec-sha256 <sha256 of the frozen gate spec> \\
                 > data/SRP1/Results/P515S56/concurrency_gate_v1/run_launch.log 2>&1
Exit codes: 0 gate PASS; 2 gate FAIL (a bitwise mismatch: investigated, never re-run or tuned); 1 precondition,
harness, guard or capture failure.

Objective convention (for every value quoted): Q = gross_operational_cost (contracted interface settlement excluded,
salvage reported separately); the gate compares records bitwise and quotes no objective difference.
"""
import argparse
import ctypes
import hashlib
import json
import os
import pickle
import re
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 concurrency gate parent (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('S56 concurrency gate: pickle.load called -- the parent never unpickles')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('S56 concurrency gate: pickle.loads called -- the parent never unpickles')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer)
import p515_s44_campaign_harness as H  # noqa: E402  (stdlib only at import)

pickle.load, pickle.loads = _blocked_load, _blocked_loads

# ======================================================================================================================
#  constants (all frozen into the gate spec)
# ======================================================================================================================
SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRING = 'p515_s56_concurrency_gate'
GIB = 1 << 30
CANONICAL_PYTHON = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
SOLVER_PATH = '/usr/local/bin/ipopt'
HARNESS_EXPECTED_SHA256 = 'd9a944f220e791b82147f08d0e606126ad454b8b6b2994024f94e106798fb191'
GATE_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'concurrency_gate_v1')
SPEC_PREFIX = 'frozen_concurrency_gate_spec_v1_'
RUN_LOG_NAME = 'run_launch.log'
RESULT_FILE = 'gate_result.json'
MANIFEST_FILE = 'manifest_sha256.json'
P56A_WORK_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals')  # p56a_oracle.WORK_DIR (not imported)
STAGE = ('P5.15 S56 D6 -- serial-vs-concurrent bitwise equivalence gate for candidate-level poll parallelism '
         '(SRP1 single scenario, published configuration, 3-cycle cap; root A concurrency 1 x 1, root B concurrency 5 '
         'x 5)')
AUTHORITY = [
    'S56_DESIGN_NOTE_v1.md, section Parallelism (D6): bitwise equivalence gate over 3 cycles, one candidate alone vs '
    'the same candidate inside a full batch, in separate roots, on the per-cycle record',
    "author's request (S56 order, 2026-11-21): compare representative serial and parallel evaluations using the same "
    'inputs and numerical criteria',
    'Planner task: S56 concurrency gate (Advisor: the gate must be bitwise; predictions P1-P3 recorded before the run)',
    'TASKS.md, S56 order',
]
CAP = 3
REQUIRED_CONSECUTIVE_CYCLES = 10
SOLVES_PER_CYCLE = 51          # (1 TSO + 3 DSO) x 3 years x 4 days + 3 ESSO nodes (run_admm_arm's own derivation)
BASE_SOLVES_PER_EVALUATION = SOLVES_PER_CYCLE * (CAP + 1)   # 204: initialisation round + 3 ADMM cycles
CONFIGURATION = {
    'name': ('S56 D6 CONCURRENCY GATE -- the published SRP1 single-scenario configuration as current production '
             'declares it (the W187 screen x0_A_1x block): the case file (AA keep_memory declared), the ESS ageing '
             'baseline BASELINE (C2 + phi_cal 0.985 + soh_min 0.70) declared, the convergence-depth tight tail '
             'DECLARED ENABLED (compl_inf_tol 1e-6), 1x flexibility price'),
    'arm_label': 's39_D',
    'case_file_anderson_acceleration': {'enabled': True, 'memory': 5, 'regularization': 1e-10,
                                        'reject_policy': 'keep_memory'},
    'convergence_depth_tail': {'enabled': True, 'compl_inf_tol': 1e-06},
    'ess_ageing_baseline': {'calendar_life_years': 15, 'calendar_retention_per_year': 0.985,
                            'calibration': {'cycles_n': 10000, 'eol_retention_r': 0.8, 'reference_dod_d': 0.8,
                                            'status': 'ACTIVE'},
                            'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8, 'minimum_soh': 0.7},
    'ess_ageing_baseline_label': 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)',
    'note': ('identical in both roots; no overrides, no model variant, no flexibility-price variant, no NRF, no rho0 '
             'scale, no post-certification; cap 3 (cannot certify: 10 consecutive passing cycles required)'),
    'overrides': {},
}
REFERENCE_CONFIGURATION_SPEC = {   # the configuration block CONFIGURATION reproduces (checked key by key at freeze)
    'path': os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e1_screen_1x', 'campaign_s55_step7_x0_a_1x',
                         'campaign_spec_s55_step7_x0_a_1x_64aac615.json'),
    'keys': ('apply_rho', 'arm_label', 'case_file', 'case_file_anderson_acceleration', 'case_file_sha256',
             'convergence_depth_tail', 'ess_ageing_baseline', 'ess_ageing_baseline_label', 'ess_params_file',
             'full_diagnostics_in_rows', 'overrides'),
}
CANDIDATES = {   # label (= the s47 Phase B domain label) -> {node: (s_mva, e_mwh)}, investment year 2025
    'y2025__n7_p0.25_e1': {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)},     # the reference unit (shared)
    'x0': {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)},
    'y2025__n5_p0.25_e0.5': {5: (0.25, 0.5), 7: (0.0, 0.0), 9: (0.0, 0.0)},
    'y2025__n9_p0.25_e1': {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.25, 1.0)},
    'y2025__n7_p0.5_e1': {5: (0.0, 0.0), 7: (0.5, 1.0), 9: (0.0, 0.0)},
}
SHARED_LABEL = 'y2025__n7_p0.25_e1'
DOMAIN_SPEC_REL = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b',
                               'campaign_spec_s47_phase_b_8cfa264e.json')
ROOTS = {
    'A': {'campaign_id': 's56_cgate_v1_rootA_serial', 'concurrency': 1, 'labels': [SHARED_LABEL],
          'role': 'serial: the shared candidate alone'},
    'B': {'campaign_id': 's56_cgate_v1_rootB_concurrent', 'concurrency': 5,
          'labels': [SHARED_LABEL, 'x0', 'y2025__n5_p0.25_e0.5', 'y2025__n9_p0.25_e1', 'y2025__n7_p0.5_e1'],
          'role': 'concurrent: the shared candidate inside a full batch of five'},
}
ORDER = ('A', 'B')
MEMORY_PER_SLOT_BYTES = 11 * GIB // 4       # 2.75 GiB (the S47 launcher rule, as the Step-7 launcher)
MEMORY_RULE = ('hw.memsize - (wired + anonymous + compressor-occupied pages) x page size >= concurrency x 2.75 GiB '
               '(the S47 launcher rule); gating before each root')
MEMORY_SAMPLE_S = 5.0

# ---- compared files and exclusions (declared before the run) ---------------------------------------------------------
G1_FILES = ('g_s39_D.json#cycle_trajectory', 'per_cycle_record.jsonl', 'aa_per_cycle.jsonl')
G2_RECORDS_FILE = 'network_ipopt_solve_records.jsonl'
G2_RECORDS_EXCLUDED = {
    '[*].log_path': 'path (the IPOPT log path names the working dir, which carries the campaign id)',
    '[*].log_bytes': ('byte offsets into the log file: the log embeds its own path (output_file option line) and the '
                      'IPOPT timing lines, so offsets differ with the root name and with timings'),
}
IPOPT_FINAL_START = 'Number of Iterations....:'
IPOPT_TIMING_PREFIXES = ('Total seconds in IPOPT', 'Total CPU secs', 'Total CPU seconds', 'Total wall-clock secs',
                         'Total wall clock')
G3_EXCLUDED = {
    'campaign_id': 'root identity (one campaign id per root)',
    'campaign_spec_path': 'path',
    'campaign_spec_sha256': 'root identity (one frozen campaign spec per root)',
    'campaign_lock_seen_by_child': 'PID, root identity and timestamp of the campaign lock',
    'child_pid': 'PID',
    'parent_pid': 'PID',
    'eval_dir': 'path',
    'report_path': 'path',
    'per_cycle_record_path': 'path',
    'post_certification_path': 'path',
    'working_dir_ids': 'path (the working-dir names carry the campaign id)',
    'aa_per_cycle.path': 'path',
    'network_failures_summary.path': 'path',
    'convergence_depth_capture.network_ipopt_solve_records_path': 'path',
    'convergence_depth_capture.convergence_depth_tail_state_path': 'path',
    'convergence_depth_capture.network_ipopt_solve_records_sha256': (
        'hash of a file that embeds paths (log_path) and log byte offsets; the file is compared field by field in G2'),
    'convergence_depth_per_round_append.events_path': 'path',
    'convergence_depth_per_round_append.records_append_path': 'path',
    'convergence_depth_per_round_append.records_append_sha256': 'as network_ipopt_solve_records_sha256 (same content)',
    'convergence_depth_per_round_append.end_of_run_records_sha256': 'as network_ipopt_solve_records_sha256',
    'wall_time_s': 'wall times',
    'peak_rss': ('process memory measurements (ru_maxrss): not a wall time, PID, path or timestamp -- ADDED to the '
                 "Planner's list, stated here before the run; reported under P3 instead"),
}
SUPPLEMENTARY_SKIP_FILES = ('launch.json', 'wait4_rusage.json', 'child_manifest_sha256.json', 'exit_code.txt',
                            'heartbeat_s39_D.json', 'evaluation_record.json', 'per_cycle_record.jsonl',
                            'aa_per_cycle.jsonl', G2_RECORDS_FILE)
SUPPLEMENTARY_GENERIC_KEYS = ('pid', 'ppid', 'child_pid', 'parent_pid', 'log_path', 'log_bytes')
SUPPLEMENTARY_GENERIC_SUBSTRINGS = ('utc', 'timestamp', 'wall', 'rss', 'footprint', 'elapsed', 'runtime', 'seconds',
                                    '_path', 'path_', 'sha256')
HISTORICAL_CROSS_CHECKS = {   # informational, read-only, NOT gated
    'x0': {'eval_dir': os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e1_screen_1x', 'campaign_s55_step7_x0_a_1x',
                                    'evals', '5cfe69a615ae3708_x0_a_1x'),
           'note': 'W187 screen x0_A_1x (same harness d9a944f2, production f6c23977), cycles 1-3'},
    SHARED_LABEL: {'eval_dir': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'tight_tail_w86',
                                            'campaign_s53_w86_tail_recert', 'evals', 'ca8927e75d628bd1_n7_4h_e1'),
                   'note': 'W86 tail re-certification n7_4h_e1 (older harness and production), cycles 1-3'},
}
SELFTEST_DIRS = {
    'x0_A_1x': os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e1_screen_1x', 'campaign_s55_step7_x0_a_1x',
                            'evals', '5cfe69a615ae3708_x0_a_1x'),
    'n7_N_1x': os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e1_screen_1x',
                            'campaign_s55_step7_y2025__n7_p0_25_e0_5_n_1x', 'evals',
                            'a4bce87e2c2851e9_y2025__n7_p0_25_e0_5_n_1x'),
}
PREDICTIONS = {
    'P1': {'text': "the shared candidate's per-cycle record is bitwise identical in A and B", 'source': 'Advisor',
           'evaluated_as': 'G1 and G2 and G3 hold (every compared field bitwise equal)'},
    'P2': {'text': 'contention 1.12-1.22x per cycle in B vs A', 'source': 'Advisor',
           'evaluated_as': ('primary: sum over cycles 1-3 of the shared candidate\'s production "Iteration k: X s" '
                            'wall in B / the same sum in A, held iff in [1.12, 1.22]; per-cycle ratios reported beside'),
           'range': [1.12, 1.22]},
    'P3': {'text': 'peak RSS per SRP1 child 2.3-3.5 GiB', 'source': 'Advisor',
           'evaluated_as': ('every one of the 6 children: evaluation_record peak_rss.child_python_process_ru_maxrss '
                            '(bytes, macOS) / 2^30 in [2.3, 3.5]'),
           'range_gib': [2.3, 3.5]},
}
FAILURE_POLICY = ('a bitwise mismatch is investigated (which field, which cycle), never re-run or tuned; the first '
                  'differing field per compared file is recorded in gate_result.json')


# ======================================================================================================================
#  small helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')} [S56-CGATE] {msg}", flush=True)


def _abs(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def _rel(path):
    return os.path.relpath(path, REPO) if os.path.isabs(path) else path


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _sha_or_none(rel):
    return _sha(rel) if os.path.isfile(_abs(rel)) else None


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _read_jsonl(path):
    with open(_abs(path)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git_tracked_clean(rel):
    tracked = bool(H._git(['ls-files', '--', rel]).strip())
    clean = not H._git(['status', '--porcelain', '--', rel]).strip()
    return tracked, clean


def _committed_clean(rel):
    tracked, clean = _git_tracked_clean(rel)
    return tracked and clean


def _write_once_json(path, obj):
    text = GRIO.dumps(obj, indent=1, sort_keys=True)
    with open(path, 'x') as handle:
        handle.write(text)
    return H.sha256_file(path)


def guards_report():
    failures = PARENT_GUARD.verify(0)
    return {'parent_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': failures},
            'pickle': {'counts': dict(PICKLE_COUNTS),
                       'blocked': pickle.load is _blocked_load and pickle.loads is _blocked_loads},
            'all_zero': (not failures and not any(PICKLE_COUNTS.values())
                         and pickle.load is _blocked_load and pickle.loads is _blocked_loads)}


def _own_process_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and OWN_PROCESS_SUBSTRING in parts[1] and 'python' in parts[1]:
            hits.append(line.strip())
    return hits


def solver_check():
    res = H._resolve_solver_path_from_dotenv()
    path = res.get('NLP_SOLVER_PATH')
    sha = H.sha256_file(path) if path and os.path.isfile(path) else None
    return {'resolved': res, 'expected': SOLVER_PATH, 'sha256': sha,
            'ok': path == SOLVER_PATH and res.get('source') == 'environment' and sha is not None}


def memory_available():
    """The S47 / Step-7 launcher measure: hw.memsize - (wired + anonymous + compressor-occupied pages) x page size."""
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    page = int(text.splitlines()[0].split('page size of')[1].split('bytes')[0].strip())
    raw = {}
    for line in text.splitlines()[1:]:
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().rstrip('.')
            if v.isdigit():
                raw[k.strip().strip('"')] = int(v)
    need = ('Pages wired down', 'Anonymous pages', 'Pages occupied by compressor')
    total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, check=True).stdout)
    if any(raw.get(k) is None for k in need):
        return {'utc': _utc(), 'available_bytes': None, 'missing': [k for k in need if raw.get(k) is None]}
    avail = total - sum(raw[k] for k in need) * page
    return {'utc': _utc(), 'available_bytes': avail, 'available_gib': avail / GIB, 'hw_memsize_gib': total / GIB,
            'page_size': page, 'pages': {k: raw.get(k) for k in need + ('Pages free', 'Pages inactive')}}


def memory_preflight(concurrency):
    m = memory_available()
    required = int(concurrency) * MEMORY_PER_SLOT_BYTES
    m.update({'concurrency': int(concurrency), 'required_gib': required / GIB, 'rule': MEMORY_RULE,
              'pass': m.get('available_bytes') is not None and m['available_bytes'] >= required})
    return m


class MemorySampler(threading.Thread):
    """Machine memory available (the preflight measure) every MEMORY_SAMPLE_S seconds while a root runs."""

    def __init__(self, period_s=MEMORY_SAMPLE_S):
        super().__init__(name='memory-sampler', daemon=True)
        self.period_s = period_s
        self._halt = threading.Event()
        self.samples = []
        self.errors = []

    def run(self):
        while True:
            try:
                m = memory_available()
                self.samples.append({'utc': m['utc'], 't': time.time(), 'available_bytes': m.get('available_bytes')})
            except Exception as error:  # noqa: BLE001 -- recorded
                self.errors.append(f'{type(error).__name__}: {error}')
            if self._halt.wait(self.period_s):
                return

    def stop(self):
        self._halt.set()
        self.join(timeout=self.period_s + 30)
        vals = [s['available_bytes'] for s in self.samples if isinstance(s.get('available_bytes'), int)]
        return {'measure': 'hw.memsize - (wired + anonymous + compressor-occupied) x page size (vm_stat)',
                'period_s': self.period_s, 'n_samples': len(self.samples), 'errors': self.errors,
                'min_available_gib': (min(vals) / GIB) if vals else None,
                'max_available_gib': (max(vals) / GIB) if vals else None,
                'first_available_gib': (vals[0] / GIB) if vals else None,
                'last_available_gib': (vals[-1] / GIB) if vals else None,
                'samples': self.samples, 'thread_alive_after_stop': self.is_alive()}


# ======================================================================================================================
#  the child footprint monitor -- COPIED from p515_s53_w90_3x3_campaign.py (W91; not imported: that launcher arms its
#  own guard chain at import). Unchanged logic; the libproc handle is opened here.
# ======================================================================================================================
FOOTPRINT_POLL_S = 1.0
FOOTPRINT_LAUNCH_POLL_S = 0.2
RUSAGE_INFO_V4 = 4
_RUSAGE_V4_FIELDS = (
    'ri_user_time', 'ri_system_time', 'ri_pkg_idle_wkups', 'ri_interrupt_wkups', 'ri_pageins', 'ri_wired_size',
    'ri_resident_size', 'ri_phys_footprint', 'ri_proc_start_abstime', 'ri_proc_exit_abstime', 'ri_child_user_time',
    'ri_child_system_time', 'ri_child_pkg_idle_wkups', 'ri_child_interrupt_wkups', 'ri_child_pageins',
    'ri_child_elapsed_abstime', 'ri_diskio_bytesread', 'ri_diskio_byteswritten', 'ri_cpu_time_qos_default',
    'ri_cpu_time_qos_maintenance', 'ri_cpu_time_qos_background', 'ri_cpu_time_qos_utility', 'ri_cpu_time_qos_legacy',
    'ri_cpu_time_qos_user_initiated', 'ri_cpu_time_qos_user_interactive', 'ri_billed_system_time',
    'ri_serviced_system_time', 'ri_logical_writes', 'ri_lifetime_max_phys_footprint', 'ri_instructions', 'ri_cycles',
    'ri_billed_energy', 'ri_serviced_energy', 'ri_interval_max_phys_footprint', 'ri_runnable_time')
FOOTPRINT_MEASURE = (
    "ri_lifetime_max_phys_footprint of the evaluation child (the harness's --child Python process; its IPOPT "
    'subprocesses are NOT included -- the same scope as child_python_process_ru_maxrss), read by this launcher with '
    'proc_pid_rusage(pid, RUSAGE_INFO_V4); FINAL value = the read on the exited, not-yet-reaped child, taken on the '
    'kqueue EVFILT_PROC NOTE_EXIT event; identity: pid from the harness launch.json, parent pid == this launcher, '
    'command line holds --child and the eval key (W91 ChildFootprintMonitor, copied)')
try:
    _LIBPROC = ctypes.CDLL('/usr/lib/libproc.dylib')
    _LIBPROC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
except OSError:
    _LIBPROC = None


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [('ri_uuid', ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _RUSAGE_V4_FIELDS]


def rusage_v4(pid):
    if _LIBPROC is None:
        return None
    info = _RUsageInfoV4()
    if _LIBPROC.proc_pid_rusage(int(pid), RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return {k: int(getattr(info, k)) for k in ('ri_phys_footprint', 'ri_lifetime_max_phys_footprint',
                                               'ri_interval_max_phys_footprint', 'ri_resident_size',
                                               'ri_proc_start_abstime', 'ri_proc_exit_abstime')}


class ChildFootprintMonitor(threading.Thread):
    """The footprint peak of ONE evaluation child, from this launcher (the child's parent). Reads only."""

    def __init__(self, launch_json_path, eval_key, poll_s=FOOTPRINT_POLL_S):
        super().__init__(name=f'footprint-{str(eval_key)[:8]}', daemon=True)
        self.launch_json_path, self.eval_key, self.poll_s = launch_json_path, eval_key, poll_s
        self._halt = threading.Event()
        self.result = {'measure': FOOTPRINT_MEASURE, 'launch_json': _rel(launch_json_path), 'eval_key': eval_key,
                       'poll_s': poll_s, 'pid': None, 'identity': None, 'identity_verified': False,
                       'registered_exit_event': False, 'n_live_reads': 0,
                       'live_max_phys_footprint_sampled_bytes': None,
                       'lifetime_max_phys_footprint_last_live_bytes': None, 'exit_event_received': False,
                       'final_read_after_exit': False, 'exit_read': None, 'lifetime_max_phys_footprint_bytes': None,
                       'capture_exact': False, 'errors': []}

    def _identity(self, pid):
        import psutil
        try:
            proc = psutil.Process(pid)
            cmd = proc.cmdline()
            ppid = proc.ppid()
            return {'ppid': ppid, 'launcher_pid': os.getpid(), 'ppid_is_launcher': ppid == os.getpid(),
                    'cmdline_has_child_flag': '--child' in cmd, 'cmdline_has_eval_key': self.eval_key in cmd}
        except psutil.Error as error:
            return {'error': f'{type(error).__name__}: {error}'}

    def _live(self, info):
        r = self.result
        r['n_live_reads'] += 1
        r['live_max_phys_footprint_sampled_bytes'] = max(r['live_max_phys_footprint_sampled_bytes'] or 0,
                                                         info['ri_phys_footprint'])
        r['lifetime_max_phys_footprint_last_live_bytes'] = info['ri_lifetime_max_phys_footprint']

    def _final(self, info):
        r = self.result
        r['final_read_after_exit'] = True
        r['exit_read'] = dict(info)
        r['lifetime_max_phys_footprint_bytes'] = info['ri_lifetime_max_phys_footprint']

    def run(self):
        r = self.result
        try:
            pid = None
            while pid is None:
                try:
                    with open(self.launch_json_path) as handle:
                        pid = int(json.load(handle)['pid'])
                except (OSError, ValueError, KeyError, TypeError):
                    if self._halt.wait(FOOTPRINT_LAUNCH_POLL_S):
                        r['errors'].append('halted before a readable launch.json appeared')
                        return
            r['pid'] = pid
            first = rusage_v4(pid)
            ident = self._identity(pid)
            r['identity'] = ident
            r['identity_verified'] = first is not None and all(
                ident.get(k) is True for k in ('ppid_is_launcher', 'cmdline_has_child_flag', 'cmdline_has_eval_key'))
            if not r['identity_verified']:
                r['errors'].append('child identity not verified')
                return
            start = first['ri_proc_start_abstime']
            kq = select.kqueue()
            try:
                kq.control([select.kevent(pid, filter=select.KQ_FILTER_PROC,
                                          flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                                          fflags=select.KQ_NOTE_EXIT)], 0)
                r['registered_exit_event'] = True
                while True:
                    info = rusage_v4(pid)
                    if info is None or info['ri_proc_start_abstime'] != start:
                        r['errors'].append('child gone before an exit read (reaped, or pid reused)')
                        break
                    if info['ri_proc_exit_abstime']:
                        self._final(info)
                        break
                    self._live(info)
                    if kq.control(None, 1, self.poll_s):
                        r['exit_event_received'] = True
                        z = rusage_v4(pid)
                        if z is not None and z['ri_proc_start_abstime'] == start and z['ri_proc_exit_abstime']:
                            self._final(z)
                        else:
                            r['errors'].append('exit event received but the child was already reaped')
                        break
                    if self._halt.is_set():
                        r['errors'].append('halted while the child was alive')
                        break
            finally:
                kq.close()
        except Exception as error:  # noqa: BLE001 -- recorded; the capture is then not exact
            r['errors'].append(f'{type(error).__name__}: {error}')
        finally:
            if r['lifetime_max_phys_footprint_bytes'] is None:
                r['lifetime_max_phys_footprint_bytes'] = r['lifetime_max_phys_footprint_last_live_bytes']
            r['capture_exact'] = bool(r['identity_verified'] and r['final_read_after_exit']
                                      and isinstance(r['lifetime_max_phys_footprint_bytes'], int)
                                      and r['lifetime_max_phys_footprint_bytes'] > 0)
            if r['lifetime_max_phys_footprint_bytes'] is not None:
                r['lifetime_max_phys_footprint_gib'] = r['lifetime_max_phys_footprint_bytes'] / GIB

    def stop(self):
        self._halt.set()
        self.join(timeout=self.poll_s + 10)
        self.result['thread_alive_after_stop'] = self.is_alive()
        return self.result


def footprint_capture_selftest():
    """ZERO SOLVES: the footprint capture exercised on a real child of this process (W91's self-test, copied):
    a child touches 256 MiB, frees it, sleeps, exits; reaped by os.wait4 WNOHANG polling. Negative control: a monitor
    given another eval key refuses."""
    alloc = 256 * (1 << 20)
    marker = 's56_fp_selftest_' + hashlib.sha256(f'{os.getpid()}_{time.time()}'.encode()).hexdigest()[:16]
    code = ('import time\nb = bytearray(%d)\nfor k in range(0, len(b), 4096):\n    b[k] = 1\ndel b\ntime.sleep(2.0)\n'
            % alloc)
    d = tempfile.mkdtemp(prefix='s56_fp_selftest_')
    out = {'allocation_bytes': alloc}
    try:
        mon = ChildFootprintMonitor(os.path.join(d, 'launch.json'), marker, poll_s=0.5)
        neg = ChildFootprintMonitor(os.path.join(d, 'launch.json'), marker + '_not_this_child', poll_s=0.5)
        mon.start()
        neg.start()
        proc = subprocess.Popen([sys.executable, '-c', code, '--child', marker], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(os.path.join(d, 'launch.json'), 'x') as handle:
            json.dump({'pid': proc.pid}, handle)
        t0, reaped, status = time.time(), False, None
        while not reaped and time.time() - t0 < 120:
            time.sleep(1.0)
            rp, status, _ru = os.wait4(proc.pid, os.WNOHANG)
            reaped = rp == proc.pid
        proc.returncode = os.waitstatus_to_exitcode(status) if reaped else None
        res, nres = mon.stop(), neg.stop()
        checks = {'reaped_by_wait4_exit_0': reaped and proc.returncode == 0,
                  'identity_verified': res['identity_verified'] is True,
                  'final_read_after_exit': res['final_read_after_exit'] is True,
                  'lifetime_max_ge_allocation': (res['lifetime_max_phys_footprint_bytes'] or 0) >= alloc,
                  'capture_exact': res['capture_exact'] is True,
                  'negative_control_refuses_other_eval_key': nres['identity_verified'] is False
                  and nres['capture_exact'] is False}
        out.update({'lifetime_max_phys_footprint_bytes': res['lifetime_max_phys_footprint_bytes'], 'checks': checks,
                    'all_hold': all(checks.values())})
    except Exception as error:  # noqa: BLE001 -- the self-test FAILS, recorded
        out.update({'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
                    'all_hold': False})
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out


# ======================================================================================================================
#  the comparator (bitwise)
# ======================================================================================================================
def _leaf_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack('<d', a) == struct.pack('<d', b)
    return a == b


def _show(v):
    """A JSON-safe, string-flag-free display of a compared value (gate_result_io refuses "True"/"False" strings)."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return 'str:' + v[:200]
    return 'json:' + json.dumps(v, sort_keys=True, default=str)[:200]


def _norm_path(path):
    return re.sub(r'\[\d+\]', '[*]', path)


def _is_excluded(path, excluded):
    p = _norm_path(path)
    for ex in excluded:
        if p == ex or p.startswith(ex + '.') or p.startswith(ex + '['):
            return True
    return False


def deep_diff(a, b, excluded=(), path='', out=None, excluded_hits=None, generic=None):
    """Every difference between a and b as (path, repr_a, repr_b), in document order. `excluded`: declared paths
    (dot-separated keys, [*] for any list index); a difference inside one is recorded in `excluded_hits` instead.
    `generic`: optional callable(key) -> True when a key is excluded by a generic rule (supplementary only)."""
    out = [] if out is None else out
    excluded_hits = [] if excluded_hits is None else excluded_hits

    def _strip(p):
        return p[1:] if p.startswith('.') else p

    def rec(x, y, p, key=None):
        sp = _strip(p)
        if (excluded and _is_excluded(sp, excluded)) or (generic is not None and key is not None and generic(key)):
            if not _equal_tree(x, y):
                excluded_hits.append(_norm_path(sp))
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in list(x.keys()) + [k for k in y.keys() if k not in x]:
                cp = f'{p}.{k}'
                if k not in x or k not in y:
                    if (excluded and _is_excluded(_strip(cp), excluded)) or (generic is not None and generic(k)):
                        excluded_hits.append(_norm_path(_strip(cp)))
                    else:
                        out.append((_strip(cp), 'MISSING' if k not in x else _show(x[k]),
                                    'MISSING' if k not in y else _show(y[k])))
                    continue
                rec(x[k], y[k], cp, k)
        elif isinstance(x, list) and isinstance(y, list):
            for i in range(min(len(x), len(y))):
                rec(x[i], y[i], f'{p}[{i}]', key)
            if len(x) != len(y):   # after the element differences, so the first difference is the earliest row
                out.append(((sp + '.' if sp else '') + '__len__', len(x), len(y)))
        elif not _leaf_equal(x, y):
            out.append((sp, _show(x), _show(y)))

    rec(a, b, path)
    return out, excluded_hits


def _equal_tree(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_equal_tree(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_equal_tree(u, v) for u, v in zip(x, y))
    return _leaf_equal(x, y)


def _summ(diffs, excluded_hits, n=25):
    return {'identical': not diffs, 'n_differences': len(diffs), 'first_differences': diffs[:n],
            'excluded_fields_that_differed': sorted(set(excluded_hits))}


def _first_cycle_field(diffs):
    """For a per-cycle file (list of rows indexed from cycle 1), the first differing (row index, field)."""
    if not diffs:
        return None
    m = re.match(r'^\[(\d+)\]\.?(.*)$', diffs[0][0])
    return {'path': diffs[0][0], 'row_index': int(m.group(1)) if m else None, 'field': m.group(2) if m else None,
            'value_A': diffs[0][1], 'value_B': diffs[0][2]}


def per_cycle_files(eval_dir):
    g = _load(os.path.join(eval_dir, 'g_s39_D.json'))
    return {'g_s39_D.json#cycle_trajectory': g['cycle_trajectory'],
            'per_cycle_record.jsonl': _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl')),
            'aa_per_cycle.jsonl': _read_jsonl(os.path.join(eval_dir, 'aa_per_cycle.jsonl'))}


def ipopt_final_blocks(path):
    """Per solve, the final-statistics block (from 'Number of Iterations....:' to the EXIT line inclusive, blank and
    timing lines removed), plus every EXIT line and the number of solve starts, of one IPOPT log."""
    blocks, exits, starts, cur = [], [], 0, None
    with open(path, 'rb') as handle:
        for raw in handle:
            line = raw.decode('latin-1').rstrip('\r\n')
            if line.startswith('This is Ipopt version'):
                starts += 1
            if line.startswith('EXIT:'):
                exits.append(line)
            if line.startswith(IPOPT_FINAL_START):
                cur = [line]
                continue
            if cur is not None:
                if line.startswith('EXIT:'):
                    cur.append(line)
                    blocks.append(cur)
                    cur = None
                elif line.strip() and not line.startswith(IPOPT_TIMING_PREFIXES):
                    cur.append(line)
    summary = []
    for b in blocks:
        it = next((ln.split(':', 1)[1].strip() for ln in b if ln.startswith(IPOPT_FINAL_START)), None)
        obj = next((ln.split(':', 1)[1].split() for ln in b if ln.startswith('Objective....')), None)
        summary.append({'iterations': int(it) if it is not None and it.isdigit() else it,
                        'objective_scaled': obj[0] if obj else None, 'objective_unscaled': obj[1] if obj else None,
                        'exit': b[-1]})
    return {'n_solve_starts': starts, 'exit_lines': exits, 'final_blocks': blocks, 'summary': summary,
            'unterminated_block': cur is not None}


def ipopt_full_text_normalised(path, tokens):
    h = hashlib.sha256()
    n = 0
    with open(path, 'rb') as handle:
        for raw in handle:
            line = raw.decode('latin-1')
            if line.lstrip().startswith(IPOPT_TIMING_PREFIXES):
                continue
            for t, rep in tokens:
                line = line.replace(t, rep)
            h.update(line.encode('latin-1'))
            n += 1
    return h.hexdigest(), n


def logs_dir_of(run_id):
    return _abs(os.path.join(P56A_WORK_DIR_REL, run_id, 'logs'))


def compare_ipopt_logs(logs_a, logs_b, tokens_a=None, tokens_b=None, full_text=False):
    fa = sorted(os.listdir(logs_a)) if os.path.isdir(logs_a) else None
    fb = sorted(os.listdir(logs_b)) if os.path.isdir(logs_b) else None
    out = {'logs_a': _rel(logs_a), 'logs_b': _rel(logs_b), 'files_a': None if fa is None else len(fa),
           'files_b': None if fb is None else len(fb), 'same_file_names': fa is not None and fa == fb,
           'per_file_differences': [], 'n_solves_compared': 0, 'n_files_compared': 0, 'summary_by_file': {}}
    if fa is None or fb is None:
        out['identical'] = False
        out['error'] = 'a logs directory is missing'
        return out
    for name in sorted(set(fa) & set(fb)):
        pa, pb = ipopt_final_blocks(os.path.join(logs_a, name)), ipopt_final_blocks(os.path.join(logs_b, name))
        out['n_files_compared'] += 1
        out['n_solves_compared'] += len(pa['final_blocks'])
        out['summary_by_file'][name] = pa['summary']
        diff = []
        for key in ('n_solve_starts', 'exit_lines', 'unterminated_block'):
            if pa[key] != pb[key]:
                diff.append({'what': key, 'A': _show(pa[key]), 'B': _show(pb[key])})
        if len(pa['final_blocks']) != len(pb['final_blocks']):
            diff.append({'what': 'n_final_blocks', 'A': len(pa['final_blocks']), 'B': len(pb['final_blocks'])})
        for i, (ba, bb) in enumerate(zip(pa['final_blocks'], pb['final_blocks'])):
            if ba != bb:
                j = next((k for k in range(min(len(ba), len(bb))) if ba[k] != bb[k]), min(len(ba), len(bb)))
                diff.append({'what': 'final_block', 'solve_index_in_file': i,
                             'first_differing_line_A': ba[j] if j < len(ba) else None,
                             'first_differing_line_B': bb[j] if j < len(bb) else None})
                break
        if diff:
            out['per_file_differences'].append({'file': name, 'differences': diff})
    for name in sorted(set(fa) ^ set(fb)):
        out['per_file_differences'].append({'file': name, 'differences': [{'what': 'present in one root only'}]})
    out['identical'] = out['same_file_names'] and not out['per_file_differences']
    if full_text:
        ft = {'identical_files': 0, 'differing_files': []}
        for name in sorted(set(fa) & set(fb)):
            ha, na = ipopt_full_text_normalised(os.path.join(logs_a, name), tokens_a or [])
            hb, nb = ipopt_full_text_normalised(os.path.join(logs_b, name), tokens_b or [])
            if ha == hb:
                ft['identical_files'] += 1
            else:
                ft['differing_files'].append({'file': name, 'lines_A': na, 'lines_B': nb})
        out['full_text_normalised_supplementary'] = ft
    out.pop('summary_by_file') if not full_text else None
    return out


def compare_eval_dirs(dir_a, dir_b, run_id_a, run_id_b, tokens_a=None, tokens_b=None, supplementary=True,
                      max_cycles=None):
    """The gate's G1-G3 (+ supplementary) between two evaluation directories of the same eval key."""
    res = {'dir_a': _rel(dir_a), 'dir_b': _rel(dir_b)}
    g1 = {}
    pa, pb = per_cycle_files(dir_a), per_cycle_files(dir_b)
    for name in G1_FILES:
        xa, xb = pa[name], pb[name]
        if max_cycles is not None:
            xa, xb = xa[:max_cycles], xb[:max_cycles]
        d, ex = deep_diff(xa, xb)
        g1[name] = {**_summ(d, ex), 'n_rows_A': len(pa[name]), 'n_rows_B': len(pb[name]),
                    'first_differing_cycle_field': _first_cycle_field(d)}
    res['G1_per_cycle_record'] = {'files': g1, 'holds': all(v['identical'] for v in g1.values())}
    if max_cycles is not None:   # the historical cross-check compares cycles only
        return res
    ra, rb = _read_jsonl(os.path.join(dir_a, G2_RECORDS_FILE)), _read_jsonl(os.path.join(dir_b, G2_RECORDS_FILE))
    d, ex = deep_diff(ra, rb, excluded=tuple(G2_RECORDS_EXCLUDED))
    g2_records = {**_summ(d, ex), 'n_records_A': len(ra), 'n_records_B': len(rb)}
    g2_logs = compare_ipopt_logs(logs_dir_of(run_id_a), logs_dir_of(run_id_b), tokens_a, tokens_b,
                                 full_text=supplementary)
    res['G2_ipopt_solves'] = {'network_ipopt_solve_records': g2_records, 'ipopt_log_final_blocks': g2_logs,
                              'holds': g2_records['identical'] and g2_logs['identical']}
    rec_a, rec_b = _load(os.path.join(dir_a, 'evaluation_record.json')), _load(os.path.join(dir_b,
                                                                                           'evaluation_record.json'))
    d, ex = deep_diff(rec_a, rec_b, excluded=tuple(G3_EXCLUDED))
    res['G3_evaluation_record'] = {**_summ(d, ex), 'holds': not d}
    if supplementary:
        res['supplementary_not_gated'] = supplementary_compare(dir_a, dir_b, tokens_a or [], tokens_b or [])
    return res


def _normalise_strings(obj, tokens):
    if isinstance(obj, str):
        for t, rep in tokens:
            obj = obj.replace(t, rep)
        return obj
    if isinstance(obj, dict):
        return {_normalise_strings(k, tokens): _normalise_strings(v, tokens) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalise_strings(v, tokens) for v in obj]
    return obj


def _generic_excluded_key(key):
    k = str(key).lower()
    return k in SUPPLEMENTARY_GENERIC_KEYS or any(s in k for s in SUPPLEMENTARY_GENERIC_SUBSTRINGS)


def supplementary_compare(dir_a, dir_b, tokens_a, tokens_b):
    out = {'json_files': {}, 'other_files_sha_equal': {}, 'note': (
        'NOT GATED. Root tokens (campaign id, campaign spec sha256) normalised in every string; keys excluded by the '
        'generic rule: ' + ', '.join(SUPPLEMENTARY_GENERIC_KEYS) + ' or containing '
        + ', '.join(SUPPLEMENTARY_GENERIC_SUBSTRINGS))}
    names_a = sorted(f for f in os.listdir(dir_a) if os.path.isfile(os.path.join(dir_a, f)))
    names_b = sorted(f for f in os.listdir(dir_b) if os.path.isfile(os.path.join(dir_b, f)))
    out['file_names_equal'] = names_a == names_b
    out['only_in_A'] = sorted(set(names_a) - set(names_b))
    out['only_in_B'] = sorted(set(names_b) - set(names_a))
    for name in sorted(set(names_a) & set(names_b)):
        if name in SUPPLEMENTARY_SKIP_FILES:
            continue
        fa, fb = os.path.join(dir_a, name), os.path.join(dir_b, name)
        if name.endswith('.json') or name.endswith('.jsonl'):
            try:
                xa = _load(fa) if name.endswith('.json') else _read_jsonl(fa)
                xb = _load(fb) if name.endswith('.json') else _read_jsonl(fb)
                d, ex = deep_diff(_normalise_strings(xa, tokens_a), _normalise_strings(xb, tokens_b),
                                  generic=_generic_excluded_key)
                out['json_files'][name] = _summ(d, ex, n=10)
            except Exception as error:  # noqa: BLE001 -- recorded
                out['json_files'][name] = {'error': f'{type(error).__name__}: {error}'}
        else:
            out['other_files_sha_equal'][name] = H.sha256_file(fa) == H.sha256_file(fb)
    out['all_json_identical'] = all(v.get('identical') for v in out['json_files'].values())
    return out


# ======================================================================================================================
#  measurements
# ======================================================================================================================
def _parse_iter_line(line):
    m = re.search(r'- Iteration (\d+): ([0-9.]+) s', line)
    return (int(m.group(1)), float(m.group(2))) if m else None


def cycle_walls_v2(eval_dir):
    """Production's own per-cycle wall: the '- Iteration k: X s' lines of the run's stdout (2 decimals); and the
    interval between consecutive 'drained' events of the convergence-depth append (secondary)."""
    walls = {}
    with open(os.path.join(eval_dir, 'stdout_s39_D.log'), errors='replace') as handle:
        for line in handle:
            if '- Iteration ' in line and line.rstrip().endswith(' s'):
                p = _parse_iter_line(line)
                if p:
                    walls[p[0]] = p[1]
    drained = {}
    ev_path = os.path.join(eval_dir, 'convergence_depth_append_events.jsonl')
    if os.path.isfile(ev_path):
        for e in _read_jsonl(ev_path):
            if e.get('event') == 'drained':
                drained[int(e['round'])] = datetime.fromisoformat(e['utc']).timestamp()
    return {'production_iteration_wall_s': {str(k): walls[k] for k in sorted(walls)},
            'drained_event_interval_s': {str(k): drained[k] - drained[k - 1] for k in sorted(drained)
                                         if k - 1 in drained}}


def contention(a_walls, b_walls):
    if 'production_iteration_wall_s' not in (a_walls or {}) or 'production_iteration_wall_s' not in (b_walls or {}):
        return {'error': 'per-cycle walls unavailable', 'primary_sum_ratio_B_over_A': None,
                'per_cycle_ratio_B_over_A': {}}
    a = {int(k): v for k, v in a_walls['production_iteration_wall_s'].items()}
    b = {int(k): v for k, v in b_walls['production_iteration_wall_s'].items()}
    cycles = [k for k in range(1, CAP + 1) if k in a and k in b]
    per = {str(k): b[k] / a[k] for k in cycles if a[k] > 0}
    primary = (sum(b[k] for k in cycles) / sum(a[k] for k in cycles)) if len(cycles) == CAP else None
    return {'cycles': cycles, 'A_s': {str(k): a[k] for k in cycles}, 'B_s': {str(k): b[k] for k in cycles},
            'per_cycle_ratio_B_over_A': per, 'primary_sum_ratio_B_over_A': primary}


# ======================================================================================================================
#  spec construction
# ======================================================================================================================
def production_files_sha():
    files = sorted(set(H.PRODUCTION_FILES_TO_CHECK_CLEAN) | {SCRIPT_NAME, 'gate_result_io.py',
                                                             'p513_solve_profile_guard.py',
                                                             H.ESS_PARAMS_FILE_REL, 'data/SRP1/SRP1.json'})
    return {f: _sha_or_none(f) for f in files}


def scratch_freeze(scratch):
    """Freeze both campaign specs into a temporary directory (outside the repo) to derive the exact entries."""
    out = {}
    for rk in ORDER:
        r = ROOTS[rk]
        cands = [(lab, CANDIDATES[lab], {}) for lab in r['labels']]
        path, sha, spec = H.freeze_campaign_spec(os.path.join(scratch, f"campaign_{r['campaign_id']}"),
                                                 r['campaign_id'], cands, CONFIGURATION, CAP, r['concurrency'],
                                                 AUTHORITY, required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
                                                 extra={'scratch': True})
        out[rk] = spec
    return out


def entry_view(e):
    return {k: e[k] for k in ('label', 'canonical', 'key', 'eval_key', 'eval_dir', 'working_dir_ids',
                              'effective_anderson_acceleration', 'overrides', 'post_certification') if k in e}


def domain_membership():
    d = _load(DOMAIN_SPEC_REL)
    dom = d['extra']['domain_I_x_eur']
    by_label = {e['label']: e for e in d['candidates']}
    out = {}
    for lab, cand in CANDIDATES.items():
        canon = H.canonical_candidate(cand)
        e = by_label.get(lab)
        out[lab] = {'in_s47_domain': lab in dom, 'I_x_eur': dom.get(lab),
                    's47_canonical_equals': e is not None and e['canonical'] == canon,
                    's47_candidate_key_equals': e is not None and e['key'] == H.candidate_key(canon)}
    return {'domain_spec': {'path': DOMAIN_SPEC_REL, 'sha256': _sha(DOMAIN_SPEC_REL)}, 'per_label': out,
            'all_admissible': all(v['in_s47_domain'] and v['s47_canonical_equals'] for v in out.values())}


def reference_configuration_check(spec_cfg):
    ref = _load(REFERENCE_CONFIGURATION_SPEC['path'])['configuration']
    keys = REFERENCE_CONFIGURATION_SPEC['keys']
    per = {k: json.dumps(ref.get(k), sort_keys=True) == json.dumps(spec_cfg.get(k), sort_keys=True) for k in keys}
    return {'reference': {'path': REFERENCE_CONFIGURATION_SPEC['path'],
                          'sha256': _sha(REFERENCE_CONFIGURATION_SPEC['path'])},
            'per_key_equal': per, 'all_equal': all(per.values())}


def exclusion_coverage_check():
    """Every root-specific leaf of the committed record of the SAME configuration and harness (W187 x0_A_1x): a key
    naming a PID / time / memory, or a string carrying the campaign id, an absolute repo path or the P56A working dir,
    must lie inside a declared G3 exclusion."""
    rec_rel = os.path.join(SELFTEST_DIRS['x0_A_1x'], 'evaluation_record.json')
    rec = _load(rec_rel)
    cid = rec['campaign_id']
    uncovered, covered = [], []

    def walk(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from walk(v, f'{p}.{k}' if p else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from walk(v, f'{p}[{i}]')
        else:
            yield p, o

    for p, v in walk(rec, ''):
        key = p.split('.')[-1].split('[')[0].lower()
        numeric = isinstance(v, (int, float)) and not isinstance(v, bool)
        flagged = ((key in ('pid', 'ppid', 'child_pid', 'parent_pid') and numeric)
                   or (any(s in key for s in ('wall', 'rss')) and numeric)
                   or ('utc' in key and isinstance(v, str))
                   or (isinstance(v, str) and (cid in v or REPO in v or 'P56A' in v)))
        if flagged:
            (covered if _is_excluded(p, tuple(G3_EXCLUDED)) else uncovered).append(_norm_path(p))
    return {'record': {'path': rec_rel, 'sha256': _sha(rec_rel)}, 'n_flagged_covered': len(set(covered)),
            'uncovered': sorted(set(uncovered)), 'holds': not uncovered}


def capture_path_checklist():
    """Rule eleven, before anything runs: a capture path exists for every quantity this gate requires, exercised on
    committed artifacts written by the SAME harness (sha256 d9a944f2) for the same configuration (W187 x0_A_1x)."""
    d = _abs(SELFTEST_DIRS['x0_A_1x'])
    rec = _load(os.path.join(d, 'evaluation_record.json'))
    run_id = rec['working_dir_ids']['run']
    checks, detail = {}, {}
    try:
        pc = per_cycle_files(d)
        checks['G1_three_per_cycle_files_parse_with_rows'] = all(len(v) > 0 for v in pc.values())
        detail['G1_rows'] = {k: len(v) for k, v in pc.items()}
    except Exception as error:  # noqa: BLE001
        checks['G1_three_per_cycle_files_parse_with_rows'] = False
        detail['G1_error'] = f'{type(error).__name__}: {error}'
    recs = _read_jsonl(os.path.join(d, G2_RECORDS_FILE))
    checks['G2_solve_records_have_iterations_exit_mu'] = bool(recs) and all(
        {'iterations', 'exit', 'mu_final', 'log_path', 'log_bytes'} <= set(r) for r in recs)
    logs = logs_dir_of(run_id)
    names = sorted(os.listdir(logs)) if os.path.isdir(logs) else []
    net = [n for n in names if n.startswith('optim_log_case')]
    esso = [n for n in names if n.startswith('optim_log_esso_node')]
    pnet = ipopt_final_blocks(os.path.join(logs, net[0])) if net else None
    pesso = ipopt_final_blocks(os.path.join(logs, esso[0])) if esso else None
    checks['G2_network_log_final_blocks_parse'] = bool(pnet and pnet['final_blocks'] and pnet['summary'][0][
        'objective_unscaled'] and pnet['summary'][0]['exit'].startswith('EXIT:') and not pnet['unterminated_block'])
    checks['G2_esso_log_final_blocks_parse'] = bool(pesso and pesso['final_blocks'] and pesso['summary'][0][
        'objective_unscaled'] and pesso['summary'][0]['exit'].startswith('EXIT:'))
    checks['G2_one_final_block_per_solve_start'] = bool(pnet and pnet['n_solve_starts'] == len(pnet['final_blocks']))
    detail['logs'] = {'dir': _rel(logs), 'n_network_logs': len(net), 'n_esso_logs': len(esso),
                      'first_network_log_solves': pnet and len(pnet['final_blocks']),
                      'first_network_log_starts': pnet and pnet['n_solve_starts']}
    checks['G3_record_has_solve_profile_and_peak_rss'] = (
        isinstance((rec.get('solve_profile') or {}).get('observed'), dict)
        and isinstance((rec.get('peak_rss') or {}).get('child_python_process_ru_maxrss'), int))
    walls = cycle_walls_v2(d)
    checks['per_cycle_wall_from_production_stdout'] = len(walls['production_iteration_wall_s']) == rec['cycles_run']
    checks['per_cycle_wall_from_drained_events'] = len(walls['drained_event_interval_s']) == rec['cycles_run']
    detail['walls_first3'] = {k: walls['production_iteration_wall_s'].get(k) for k in ('1', '2', '3')}
    m = memory_available()
    checks['memory_available_measure'] = isinstance(m.get('available_bytes'), int)
    detail['memory_now'] = m
    fp = footprint_capture_selftest()
    checks['footprint_capture_selftest'] = fp.get('all_hold') is True
    detail['footprint_selftest'] = fp
    checks['wait4_rusage_written_by_harness'] = os.path.isfile(os.path.join(d, 'wait4_rusage.json'))
    checks['harness_sha256_is_the_one_that_wrote_these'] = H.sha256_file(H.HARNESS_PATH) == HARNESS_EXPECTED_SHA256
    return {'source_eval_dir': _rel(d), 'checks': checks, 'detail': detail, 'all_hold': all(checks.values())}


def comparator_selftest():
    """ZERO SOLVES, on committed artifacts: a directory against itself must be identical in G1-G3; two different
    evaluations must differ in G1 (first differing cycle and field reported)."""
    a = _abs(SELFTEST_DIRS['x0_A_1x'])
    b = _abs(SELFTEST_DIRS['n7_N_1x'])
    ra = _load(os.path.join(a, 'evaluation_record.json'))['working_dir_ids']['run']
    rb = _load(os.path.join(b, 'evaluation_record.json'))['working_dir_ids']['run']
    same = compare_eval_dirs(a, a, ra, ra, supplementary=False)
    diff = compare_eval_dirs(a, b, ra, rb, supplementary=False)
    checks = {'self_G1_identical': same['G1_per_cycle_record']['holds'],
              'self_G2_identical': same['G2_ipopt_solves']['holds'],
              'self_G3_identical': same['G3_evaluation_record']['holds'],
              'self_G2_compared_solves_positive': same['G2_ipopt_solves']['ipopt_log_final_blocks'][
                  'n_solves_compared'] > 0,
              'other_G1_differs': not diff['G1_per_cycle_record']['holds'],
              'other_G2_differs': not diff['G2_ipopt_solves']['holds'],
              'other_G3_differs': not diff['G3_evaluation_record']['holds']}
    return {'pair_same': [SELFTEST_DIRS['x0_A_1x']] * 2, 'pair_other': [SELFTEST_DIRS['x0_A_1x'],
                                                                        SELFTEST_DIRS['n7_N_1x']],
            'checks': checks, 'all_hold': all(checks.values()),
            'other_first_differences': {
                'G1': {k: v['first_differing_cycle_field'] for k, v in diff['G1_per_cycle_record']['files'].items()},
                'G3_first': diff['G3_evaluation_record']['first_differences'][:3],
                'G2_n_solves_compared_self': same['G2_ipopt_solves']['ipopt_log_final_blocks']['n_solves_compared']}}


def commands(spec_name_hint):
    py = CANONICAL_PYTHON
    return {
        'freeze': (f'set -o noclobber && NLP_SOLVER_PATH={SOLVER_PATH} {py} -u {SCRIPT_NAME} --freeze > '
                   f'data/SRP1/Results/P515S56/concurrency_gate_v1_freeze_launch.log 2>&1'),
        'run': (f'set -o noclobber && NLP_SOLVER_PATH={SOLVER_PATH} {py} -u {SCRIPT_NAME} --run --spec-sha256 '
                f'<sha256 of {spec_name_hint}> > {GATE_DIR_REL}/{RUN_LOG_NAME} 2>&1'),
        'cwd': REPO,
        'attached': 'attached, alone, both streams captured, never detached; A then B in the same process',
    }


def freeze(_args):
    started = time.time()
    gate_dir = _abs(GATE_DIR_REL)
    if os.path.exists(gate_dir):
        _log(f'REFUSING: the gate directory exists: {GATE_DIR_REL}')
        return 1
    failures = []
    if os.path.realpath(sys.executable) != os.path.realpath(CANONICAL_PYTHON):
        failures.append(f'not the canonical interpreter: {sys.executable}')
    if H.sha256_file(H.HARNESS_PATH) != HARNESS_EXPECTED_SHA256:
        failures.append('harness sha256 differs from the expected d9a944f2')
    sol = solver_check()
    if not sol['ok']:
        failures.append(f'solver check: {sol}')
    alive = _own_process_alive()
    if alive:
        failures.append(f'another copy is alive: {alive}')
    if failures:
        for f in failures:
            _log(f'[FREEZE REFUSED] {f}')
        return 1
    scratch = tempfile.mkdtemp(prefix='s56_cgate_scratch_')
    try:
        specs = scratch_freeze(scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    roots = {}
    for rk in ORDER:
        s = specs[rk]
        roots[rk] = {**ROOTS[rk], 'campaign_root': os.path.join(GATE_DIR_REL, f"campaign_{ROOTS[rk]['campaign_id']}"),
                     'entries': [entry_view(e) for e in s['candidates']],
                     'campaign_spec_configuration': s['configuration'],
                     'cap': s['cap'], 'required_consecutive_cycles': s['required_consecutive_cycles'],
                     'expected_solves': {'base_per_evaluation': BASE_SOLVES_PER_EVALUATION,
                                         'base_total': BASE_SOLVES_PER_EVALUATION * len(s['candidates']),
                                         'rule': '51 per round x (cap 3 + 1) per evaluation + every attempted retry '
                                                 '(per-event identity of run_admm_arm)'}}
    shared = {rk: next(e for e in roots[rk]['entries'] if e['label'] == SHARED_LABEL) for rk in ORDER}
    shared_same = {k: shared['A'][k] == shared['B'][k] for k in ('label', 'canonical', 'key', 'eval_key', 'eval_dir',
                                                                 'effective_anderson_acceleration', 'overrides')}
    cfg_same = json.dumps(specs['A']['configuration'], sort_keys=True) == json.dumps(specs['B']['configuration'],
                                                                                      sort_keys=True)
    _log('scratch freezes done; running the reference-configuration, domain, exclusion-coverage, capture-path and '
         'comparator checks (zero solves)')
    refcfg = reference_configuration_check(specs['A']['configuration'])
    dom = domain_membership()
    cover = exclusion_coverage_check()
    capture = capture_path_checklist()
    selftest = comparator_selftest()
    hist = {lab: {**v, 'eval_key_in_dir_name': os.path.basename(v['eval_dir'])[:16],
                  'eval_key_prefix_equal': next(e['eval_key'] for e in roots['B']['entries'] if e['label'] == lab)[
                      :16] == os.path.basename(v['eval_dir'])[:16],
                  'present': os.path.isdir(_abs(v['eval_dir']))} for lab, v in HISTORICAL_CROSS_CHECKS.items()}
    checks = {'shared_entry_identical_in_both_roots': all(shared_same.values()),
              'configuration_identical_in_both_roots': cfg_same,
              'configuration_equals_w187_x0_A_1x_block': refcfg['all_equal'],
              'candidates_admissible_in_s47_domain': dom['all_admissible'],
              'g3_exclusions_cover_every_root_specific_leaf': cover['holds'],
              'capture_path_checklist': capture['all_hold'],
              'comparator_selftest': selftest['all_hold']}
    spec = {
        'schema': 'p515_s56_concurrency_gate_spec_v1', 'version': 1, 'predecessor': None,
        'stage': STAGE, 'authority': AUTHORITY, 'frozen_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'script': {'path': SCRIPT_NAME, 'sha256': _sha(SCRIPT_NAME)},
        'harness': {'path': _rel(H.HARNESS_PATH), 'sha256': H.sha256_file(H.HARNESS_PATH), 'unchanged': True},
        'pins_sha256': production_files_sha(), 'interpreter': CANONICAL_PYTHON, 'solver': sol,
        'thread_caps': dict(H.THREAD_CAP_ENV), 'PYTHONHASHSEED_in_parent_env': os.environ.get('PYTHONHASHSEED'),
        'cap': {'value': CAP, 'mechanism': ("the harness's own spec['cap'] -> run_admm_arm(num_max_iters_override="
                                            "cap) (p515_s44_campaign_harness.py, child), as in the W90/W91 3-cycle "
                                            "smokes (campaign_s53_w91_3x3_smoke_bon spec: cap 3)"),
                'consequence': 'cycles_run 3, status not_certified (10 consecutive passing cycles are required)'},
        'configuration_passed_to_freeze_campaign_spec': CONFIGURATION,
        'configuration_reference_check': refcfg,
        'objective_convention': ('Q = gross_operational_cost (contracted interface settlement excluded; salvage '
                                 'reported separately); no objective difference is quoted -- records are compared '
                                 'bitwise'),
        'candidates': {lab: {'nodes': {str(n): list(v) for n, v in c.items()}, 'investment_year': 2025}
                       for lab, c in CANDIDATES.items()},
        'shared_candidate': {'label': SHARED_LABEL, 'description': 'the reference unit: node 7, 0.25 MVA / 1 MWh, 2025',
                             'entry': shared['A'], 'identical_in_both_roots': shared_same},
        'domain_membership': dom, 'roots': roots, 'order': list(ORDER),
        'gate': {
            'G0_test_condition': ('A: evaluate() max_concurrent_observed == 1; B: == 5 (all five children launched in '
                                  'the first poll); every child exit 0, status not_certified, cycles_run 3'),
            'G1_per_cycle_record': {'files': list(G1_FILES), 'excluded': {}},
            'G2_ipopt_solves': {'records_file': G2_RECORDS_FILE, 'records_excluded': G2_RECORDS_EXCLUDED,
                                'logs': ('every file of <P56A work dir>/<working_dir_ids.run>/logs (network '
                                         'optim_log_case*.log and ESSO optim_log_esso_node*_cycle*.txt): same file '
                                         'names; per file the number of solve starts, every EXIT line and, per solve, '
                                         'the final-statistics block from "Number of Iterations....:" to the EXIT '
                                         'line inclusive with blank lines and lines starting with '
                                         f'{list(IPOPT_TIMING_PREFIXES)} removed, compared as text')},
            'G3_evaluation_record': {'file': 'evaluation_record.json', 'excluded': G3_EXCLUDED,
                                     'exclusion_coverage_check': cover},
            'G4_solve_counts': ('each of the 6 records: solve_profile.base_solves == 204, rounds == 4, '
                                'reconciliation_supported and identity_holds True, observed.permitted_solve == '
                                'expected_solves, blocked 0; parent SolveProfileGuard(permitted=()) verify(0) and '
                                'pickle loads 0'),
            'equality': ('bitwise: floats by IEEE-754 bytes (struct <d), exact types, equal key sets, equal list '
                         'lengths; strings and ints by =='),
            'verdict': 'GATE PASS iff G0 and G1 and G2 and G3 and G4 hold',
        },
        'supplementary_not_gated': {
            'json_artifacts': ('every other *.json / *.jsonl of the shared eval dir except '
                               f'{list(SUPPLEMENTARY_SKIP_FILES)}; root tokens normalised; generic key rule '
                               f'{list(SUPPLEMENTARY_GENERIC_KEYS)} + substrings '
                               f'{list(SUPPLEMENTARY_GENERIC_SUBSTRINGS)}'),
            'ipopt_full_text': 'every log, root tokens normalised, timing lines removed, sha256 compared',
            'historical_cross_check_cycles_1_3': hist,
        },
        'measurements': {
            'per_cycle_wall': ('production\'s "- Iteration k: X s" lines of stdout_s39_D.log (2 decimals), per cycle, '
                               'every child; secondary: the interval between consecutive "drained" events of '
                               'convergence_depth_append_events.jsonl'),
            'contention': ('shared candidate: B/A per cycle and sum over cycles 1-3 (primary); every B child against '
                           'the shared candidate in A, reported'),
            'peak_rss': ('evaluation_record peak_rss.child_python_process_ru_maxrss (bytes, macOS ru_maxrss); '
                         'wait4 ru_maxrss of the reaped child; solver_subprocesses_max_ru_maxrss'),
            'footprint': FOOTPRINT_MEASURE,
            'memory_available_during_B': (f'the preflight measure sampled every {MEMORY_SAMPLE_S} s from just before '
                                          'the B launch to its return; min / max / first / last'),
            'memory_preflight': MEMORY_RULE,
        },
        'predictions_recorded_before_run': PREDICTIONS,
        'failure_policy': FAILURE_POLICY,
        'not_permitted': ['any change to production code or the harness', 'any run other than the two roots',
                          'tuning', 're-running on failure'],
        'writes': {'gate_dir': GATE_DIR_REL,
                   'campaign_roots': [roots[rk]['campaign_root'] for rk in ORDER],
                   'p56a_working_dirs (harness-fixed location, new names)': [
                       os.path.join(P56A_WORK_DIR_REL, e['working_dir_ids'][w]) for rk in ORDER
                       for e in roots[rk]['entries'] for w in ('precheck', 'run')],
                   'results': [os.path.join(GATE_DIR_REL, RESULT_FILE), os.path.join(GATE_DIR_REL, MANIFEST_FILE)],
                   'note': 'nothing a committed report cites is written'},
        'commands': commands(SPEC_PREFIX + '<sha8>.json'),
        'freeze_checks': checks, 'freeze_checks_all_hold': all(checks.values()),
        'capture_path_checklist': capture, 'comparator_selftest': selftest,
        'freeze_wall_s': time.time() - started,
    }
    os.makedirs(gate_dir)
    text = GRIO.dumps(spec, indent=1, sort_keys=True)
    digest = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(gate_dir, f'{SPEC_PREFIX}{digest[:8]}.json')
    with open(path, 'x') as handle:
        handle.write(text)
    if H.sha256_file(path) != digest:
        _log('spec hash mismatch after write')
        return 1
    for k, v in checks.items():
        _log(f'  freeze check {k}: {v}')
    _log(f'shared entry: eval key {shared["A"]["eval_key"]} eval dir {shared["A"]["eval_dir"]}')
    for rk in ORDER:
        _log(f"root {rk}: {roots[rk]['campaign_root']} concurrency {roots[rk]['concurrency']} entries "
             f"{[(e['label'], e['eval_key'][:16]) for e in roots[rk]['entries']]}")
    g = guards_report()
    _log(f'guards {g}')
    _log(f'FROZEN: {_rel(path)} sha256 {digest}')
    return 0 if (all(checks.values()) and g['all_zero']) else 1


# ======================================================================================================================
#  run
# ======================================================================================================================
def run_root(rk, gspec, gate_dir):
    r = gspec['roots'][rk]
    root = _abs(r['campaign_root'])
    failures = []
    failures += H.check_campaign_preconditions(root, extra_clean_files=(SCRIPT_NAME, _rel(gspec['_path'])))
    mem = memory_preflight(r['concurrency'])
    if not mem['pass']:
        failures.append(f"memory preflight REFUSED: available {mem.get('available_gib')} GiB < required "
                        f"{mem.get('required_gib')} GiB")
    if failures:
        return {'launched': False, 'failures': failures, 'memory_preflight': mem}
    cands = [(lab, CANDIDATES[lab], {}) for lab in r['labels']]
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, r['campaign_id'], cands, CONFIGURATION, CAP, r['concurrency'], AUTHORITY,
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra={'gate_spec': {'path': _rel(gspec['_path']), 'sha256': gspec['_sha256']}, 'root': rk,
               'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME),
               'objective_convention': gspec['objective_convention']})
    got = [entry_view(e) for e in spec['candidates']]
    entries_equal = json.dumps(got, sort_keys=True) == json.dumps(r['entries'], sort_keys=True)
    cfg_equal = json.dumps(spec['configuration'], sort_keys=True) == json.dumps(r['campaign_spec_configuration'],
                                                                                 sort_keys=True)
    if not (entries_equal and cfg_equal and spec['cap'] == CAP and spec['concurrency'] == r['concurrency']):
        return {'launched': False, 'campaign_spec': {'path': _rel(spec_path), 'sha256': spec_sha},
                'failures': [f'frozen campaign spec differs from the gate spec declaration: entries_equal '
                             f'{entries_equal} configuration_equal {cfg_equal}'], 'memory_preflight': mem}
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha)
    _log(f"[ROOT {rk}] lock {lock}; {len(r['labels'])} evaluation(s) at concurrency {r['concurrency']}, cap {CAP}; "
         f"spec {_rel(spec_path)}")
    entries = {e['label']: e for e in spec['candidates']}
    monitors = {lab: ChildFootprintMonitor(os.path.join(root, 'evals', entries[lab]['eval_dir'], 'launch.json'),
                                           entries[lab]['eval_key']) for lab in r['labels']}
    sampler = MemorySampler()
    sampler.start()
    for mon in monitors.values():
        mon.start()
    t0 = time.time()
    records, batch, err = None, {}, None
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate(list(r['labels']), ctx)
        batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    except BaseException as error:  # noqa: BLE001 -- recorded; a harness fault
        err = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    finally:
        footprints = {lab: mon.stop() for lab, mon in monitors.items()}
        mem_during = sampler.stop()
        released = H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[ROOT {rk}] campaign lock released: {released}')
    wall = time.time() - t0
    per = {}
    for lab in r['labels']:
        e = entries[lab]
        ed = os.path.join(root, 'evals', e['eval_dir'])
        rec = next((x for x in (records or []) if (x or {}).get('candidate_label') == lab), None) or {}
        sp = rec.get('solve_profile') or {}
        try:
            walls = cycle_walls_v2(ed)
        except Exception as error:  # noqa: BLE001
            walls = {'error': f'{type(error).__name__}: {error}'}
        w4 = _load(os.path.join(ed, 'wait4_rusage.json')) if os.path.isfile(os.path.join(ed, 'wait4_rusage.json')) \
            else {}
        per[lab] = {'eval_dir': _rel(ed), 'eval_key': e['eval_key'], 'working_dir_run': e['working_dir_ids']['run'],
                    'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
                    'exit_code': (rec.get('parent_view') or {}).get('exit_code'),
                    'gross_operational_cost_terminal': rec.get('terminal_gross_operational_cost'),
                    'solve_profile': sp, 'peak_rss': rec.get('peak_rss'),
                    'peak_rss_child_gib': ((rec.get('peak_rss') or {}).get('child_python_process_ru_maxrss') or 0)
                    / GIB if rec.get('peak_rss') else None,
                    'wait4': w4, 'wait4_ru_maxrss_gib': ((w4.get('rusage') or {}).get('ru_maxrss') or 0) / GIB
                    if w4 else None,
                    'footprint': footprints.get(lab), 'wall_time_s': rec.get('wall_time_s'), 'cycle_walls': walls,
                    'local_solve_failures': rec.get('local_solve_failures')}
    return {'launched': True, 'campaign_root': _rel(root), 'campaign_spec': {'path': _rel(spec_path),
                                                                             'sha256': spec_sha},
            'entries_equal_declaration': entries_equal, 'configuration_equal_declaration': cfg_equal,
            'memory_preflight': mem, 'memory_during': mem_during, 'batch_info': batch, 'root_wall_s': wall,
            'harness_error': err, 'per_evaluation': per}


def g4_check(per, base):
    out = {}
    for lab, p in per.items():
        sp = p.get('solve_profile') or {}
        obs = sp.get('observed') or {}
        out[lab] = {'base_solves': sp.get('base_solves'), 'rounds': sp.get('rounds'),
                    'retry_solves_credited': sp.get('retry_solves_credited'),
                    'observed_permitted_solve': obs.get('permitted_solve'), 'expected_solves': sp.get('expected_solves'),
                    'holds': (sp.get('base_solves') == base and sp.get('rounds') == CAP + 1
                              and sp.get('reconciliation_supported') is True and sp.get('identity_holds') is True
                              and obs.get('permitted_solve') == sp.get('expected_solves')
                              and obs.get('blocked_solve') == 0 and obs.get('blocked_exec') == 0)}
    return out


def run(args):
    started = time.time()
    gate_dir = _abs(GATE_DIR_REL)
    failures = []
    hits = [f for f in os.listdir(gate_dir) if f.startswith(SPEC_PREFIX)] if os.path.isdir(gate_dir) else []
    match = [f for f in hits if H.sha256_file(os.path.join(gate_dir, f)) == args.spec_sha256]
    if len(match) != 1:
        _log(f'REFUSING: no unique gate spec with sha256 {args.spec_sha256} in {GATE_DIR_REL} ({hits})')
        return 1
    spec_rel = os.path.join(GATE_DIR_REL, match[0])
    gspec = _load(spec_rel)
    gspec['_path'], gspec['_sha256'] = spec_rel, args.spec_sha256
    allowed = {match[0], RUN_LOG_NAME}
    extra_files = sorted(set(os.listdir(gate_dir)) - allowed)
    if extra_files:
        failures.append(f'the gate dir must hold only the spec and {RUN_LOG_NAME}; also holds {extra_files}')
    if not _committed_clean(spec_rel):
        failures.append('the gate spec is not committed and clean')
    if not _committed_clean(SCRIPT_NAME):
        failures.append('this script is not committed and clean')
    if gspec['script']['sha256'] != _sha(SCRIPT_NAME):
        failures.append('this script differs from the one the spec pinned')
    if H.sha256_file(H.HARNESS_PATH) != gspec['harness']['sha256'] or gspec['harness']['sha256'] != \
            HARNESS_EXPECTED_SHA256:
        failures.append('harness sha256 differs from the pin')
    pins_now = production_files_sha()
    pin_diff = sorted(f for f, s in gspec['pins_sha256'].items() if f != SCRIPT_NAME and pins_now.get(f) != s)
    if pin_diff:
        failures.append(f'pinned files differ: {pin_diff}')
    if os.path.realpath(sys.executable) != os.path.realpath(CANONICAL_PYTHON):
        failures.append(f'not the canonical interpreter: {sys.executable}')
    sol = solver_check()
    if not sol['ok'] or sol['sha256'] != gspec['solver']['sha256']:
        failures.append(f'solver differs from the frozen one: {sol}')
    if not gspec.get('freeze_checks_all_hold'):
        failures.append('the frozen spec records a failed freeze check')
    alive = _own_process_alive()
    if alive:
        failures.append(f'another copy of this script is alive: {alive}')
    for rk in ORDER:
        if os.path.exists(_abs(gspec['roots'][rk]['campaign_root'])):
            failures.append(f"campaign root exists: {gspec['roots'][rk]['campaign_root']}")
        for e in gspec['roots'][rk]['entries']:
            for w in ('precheck', 'run'):
                if os.path.exists(_abs(os.path.join(P56A_WORK_DIR_REL, e['working_dir_ids'][w]))):
                    failures.append(f"working dir exists: {e['working_dir_ids'][w]}")
    mem_b_upfront = memory_preflight(ROOTS['B']['concurrency'])   # so A never runs when B could not follow it
    _log(f"memory now {mem_b_upfront.get('available_gib')} GiB; B requires {mem_b_upfront.get('required_gib')} GiB")
    if not mem_b_upfront['pass']:
        failures.append(f"memory preflight for root B REFUSED before root A: available "
                        f"{mem_b_upfront.get('available_gib')} GiB < required {mem_b_upfront.get('required_gib')} GiB")
    _log('capture-path checklist (zero solves, before any launch)')
    capture = capture_path_checklist()
    if not capture['all_hold']:
        failures.append(f"capture-path checklist fails: {[k for k, v in capture['checks'].items() if not v]}")
    if failures:
        for f in failures:
            _log(f'[RUN PRECONDITION FAILED] {f}')
        return 1
    head = H._git(['rev-parse', 'HEAD'])
    _log(f'every precondition holds; HEAD {head}; gate spec {spec_rel}')
    roots_out = {}
    for rk in ORDER:
        _log(f"[ROOT {rk}] {gspec['roots'][rk]['role']}: launching")
        roots_out[rk] = run_root(rk, gspec, gate_dir)
        ro = roots_out[rk]
        if not ro.get('launched'):
            _log(f"[ROOT {rk}] NOT LAUNCHED: {ro.get('failures')}")
            break
        for lab, p in ro['per_evaluation'].items():
            _log(f"[ROOT {rk}] {lab}: exit {p['exit_code']} status {p['status']} cycles {p['cycles_run']} "
                 f"walls {p['cycle_walls'].get('production_iteration_wall_s') if isinstance(p['cycle_walls'], dict) else None} "
                 f"peak RSS {p['peak_rss_child_gib']} GiB footprint "
                 f"{(p['footprint'] or {}).get('lifetime_max_phys_footprint_gib')} GiB")
        if ro.get('harness_error'):
            _log(f"[ROOT {rk}] HARNESS ERROR: {ro['harness_error']['error']}")
            break
    both = all(roots_out.get(rk, {}).get('launched') and not roots_out[rk].get('harness_error') for rk in ORDER)
    gate = {'evaluated': False}
    if both:
        _log('comparing the shared candidate (G1-G3, supplementary) -- zero solves')
        pa = roots_out['A']['per_evaluation'][SHARED_LABEL]
        pb = roots_out['B']['per_evaluation'][SHARED_LABEL]
        tok_a = [(roots_out['A']['campaign_spec']['sha256'], '<SPEC_SHA256>'),
                 (gspec['roots']['A']['campaign_id'], '<CAMPAIGN_ID>'),
                 (roots_out['A']['campaign_spec']['sha256'][:8], '<SPEC_SHA8>')]
        tok_b = [(roots_out['B']['campaign_spec']['sha256'], '<SPEC_SHA256>'),
                 (gspec['roots']['B']['campaign_id'], '<CAMPAIGN_ID>'),
                 (roots_out['B']['campaign_spec']['sha256'][:8], '<SPEC_SHA8>')]
        try:
            cmp_ = compare_eval_dirs(_abs(pa['eval_dir']), _abs(pb['eval_dir']), pa['working_dir_run'],
                                     pb['working_dir_run'], tok_a, tok_b, supplementary=True)
        except Exception as error:  # noqa: BLE001 -- a comparator fault is a harness fault, recorded
            cmp_ = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        g0 = {'A_max_concurrent_observed': roots_out['A']['batch_info'].get('max_concurrent_observed'),
              'B_max_concurrent_observed': roots_out['B']['batch_info'].get('max_concurrent_observed'),
              'children': {f'{rk}:{lab}': {'exit_code': p['exit_code'], 'status': p['status'],
                                           'cycles_run': p['cycles_run']}
                           for rk in ORDER for lab, p in roots_out[rk]['per_evaluation'].items()}}
        g0['holds'] = (g0['A_max_concurrent_observed'] == 1 and g0['B_max_concurrent_observed'] == 5
                       and all(c['exit_code'] == 0 and c['status'] == 'not_certified' and c['cycles_run'] == CAP
                               for c in g0['children'].values()))
        g4 = {rk: g4_check(roots_out[rk]['per_evaluation'], BASE_SOLVES_PER_EVALUATION) for rk in ORDER}
        guards_mid = guards_report()
        g4_holds = all(v['holds'] for rk in ORDER for v in g4[rk].values()) and guards_mid['all_zero']
        hist = {}
        for lab, h in HISTORICAL_CROSS_CHECKS.items():
            try:
                if os.path.isdir(_abs(h['eval_dir'])):
                    cur = roots_out['B']['per_evaluation'][lab]['eval_dir']
                    c = compare_eval_dirs(_abs(h['eval_dir']), _abs(cur), None, None, supplementary=False,
                                          max_cycles=CAP)
                    hist[lab] = {'historical': h, 'G1_cycles_1_3': c['G1_per_cycle_record']}
                else:
                    hist[lab] = {'historical': h, 'present': False}
            except Exception as error:  # noqa: BLE001 -- informational
                hist[lab] = {'historical': h, 'error': f'{type(error).__name__}: {error}'}
        g1 = (cmp_.get('G1_per_cycle_record') or {}).get('holds') is True
        g2 = (cmp_.get('G2_ipopt_solves') or {}).get('holds') is True
        g3 = (cmp_.get('G3_evaluation_record') or {}).get('holds') is True
        gate = {'evaluated': True, 'G0_test_condition': g0, 'G1_holds': g1, 'G2_holds': g2, 'G3_holds': g3,
                'G4_solve_counts': g4, 'G4_holds': g4_holds, 'comparison': cmp_,
                'GATE_PASS': bool(g0['holds'] and g1 and g2 and g3 and g4_holds),
                'historical_cross_check_not_gated': hist}
    # measurements and predictions
    meas, preds = {}, {}
    if both:
        wa = roots_out['A']['per_evaluation'][SHARED_LABEL]['cycle_walls']
        wb = roots_out['B']['per_evaluation'][SHARED_LABEL]['cycle_walls']
        cont = contention(wa, wb)
        others = {lab: contention(wa, p['cycle_walls']) for lab, p in roots_out['B']['per_evaluation'].items()}
        rss = {f'{rk}:{lab}': p['peak_rss_child_gib'] for rk in ORDER for lab, p in
               roots_out[rk]['per_evaluation'].items()}
        fp = {f'{rk}:{lab}': {'gib': (p['footprint'] or {}).get('lifetime_max_phys_footprint_gib'),
                              'capture_exact': (p['footprint'] or {}).get('capture_exact')}
              for rk in ORDER for lab, p in roots_out[rk]['per_evaluation'].items()}
        meas = {'contention_shared_candidate': cont, 'contention_every_B_child_vs_A_shared': others,
                'peak_rss_child_gib': rss, 'footprint_child_gib': fp,
                'memory_available_during_B': {k: v for k, v in roots_out['B']['memory_during'].items()
                                              if k != 'samples'},
                'memory_preflight': {rk: roots_out[rk]['memory_preflight'] for rk in ORDER},
                'root_wall_s': {rk: roots_out[rk]['root_wall_s'] for rk in ORDER}}
        p2v = cont['primary_sum_ratio_B_over_A']
        lo, hi = PREDICTIONS['P2']['range']
        rlo, rhi = PREDICTIONS['P3']['range_gib']
        preds = {'P1': {**PREDICTIONS['P1'], 'outcome': {'G1': gate['G1_holds'], 'G2': gate['G2_holds'],
                                                          'G3': gate['G3_holds']},
                        'held': bool(gate['G1_holds'] and gate['G2_holds'] and gate['G3_holds'])},
                 'P2': {**PREDICTIONS['P2'], 'outcome': p2v, 'per_cycle': cont['per_cycle_ratio_B_over_A'],
                        'held': (p2v is not None and lo <= p2v <= hi)},
                 'P3': {**PREDICTIONS['P3'], 'outcome': rss,
                        'held': all(v is not None and rlo <= v <= rhi for v in rss.values())}}
    g = guards_report()
    result = {'schema': 'p515_s56_concurrency_gate_result_v1', 'stage': STAGE, 'utc': _utc(), 'git_head_at_run': head,
              'gate_spec': {'path': spec_rel, 'sha256': args.spec_sha256},
              'script': {'path': SCRIPT_NAME, 'sha256': _sha(SCRIPT_NAME)},
              'harness': {'path': _rel(H.HARNESS_PATH), 'sha256': H.sha256_file(H.HARNESS_PATH)},
              'objective_convention': gspec['objective_convention'],
              'shared_candidate': gspec['shared_candidate']['label'],
              'capture_path_checklist_at_run': capture, 'roots': roots_out, 'gate': gate,
              'measurements': meas, 'predictions': preds, 'guards': g, 'failure_policy': FAILURE_POLICY,
              'wall_clock_s': time.time() - started,
              'note_manifest': f'{RUN_LOG_NAME} is excluded from the manifest (still being written); hash it after exit'}
    rsha = _write_once_json(os.path.join(gate_dir, RESULT_FILE), result)
    manifest = {}
    for directory, _dirs, files in os.walk(gate_dir):
        for fname in sorted(files):
            if fname in (RUN_LOG_NAME, MANIFEST_FILE):
                continue
            fpath = os.path.join(directory, fname)
            manifest[_rel(fpath)] = H.sha256_file(fpath)
    for rk in ORDER:
        for p in (roots_out.get(rk, {}).get('per_evaluation') or {}).values():
            ld = logs_dir_of(p['working_dir_run'])
            if os.path.isdir(ld):
                for fname in sorted(os.listdir(ld)):
                    manifest[_rel(os.path.join(ld, fname))] = H.sha256_file(os.path.join(ld, fname))
    _write_once_json(os.path.join(gate_dir, MANIFEST_FILE), {'files_sha256': manifest, 'gate_result_sha256': rsha})
    _log(f'gate_result.json sha256 {rsha}')
    if gate.get('evaluated'):
        _log(f"G0 {gate['G0_test_condition']['holds']} G1 {gate['G1_holds']} G2 {gate['G2_holds']} G3 "
             f"{gate['G3_holds']} G4 {gate['G4_holds']} -> GATE_PASS {gate['GATE_PASS']}")
        for k, v in preds.items():
            _log(f"{k}: held {v['held']} outcome {v['outcome']}")
    _log(f'guards {g}')
    if not both or not g['all_zero'] or not gate.get('evaluated') or 'error' in gate.get('comparison', {}):
        return 1
    return 0 if gate['GATE_PASS'] else 2


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--spec-sha256', default=None)
    args = parser.parse_args()
    if args.run and not args.spec_sha256:
        parser.error('--run needs --spec-sha256')
    _log(f'{STAGE}; pid {os.getpid()}; interpreter {sys.executable}')
    code = freeze(args) if args.freeze else run(args)
    _log(f'exit {code}')
    return code


if __name__ == '__main__':
    sys.exit(main())
