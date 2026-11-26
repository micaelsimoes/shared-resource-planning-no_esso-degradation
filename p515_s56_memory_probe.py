"""P5.15 S56 critical path step 3 -- MEMORY PROBE of the corrected stochastic (SP) model: build + exactly 2 ADMM cycles
on each candidate planning grid, run serially, with the peak RSS of the whole process tree sampled by this launcher.

Authority: TASKS.md "Working mode and EV priority", critical path step 3 (engineering measurement mode: a script with
pass/fail checks; no frozen prediction spec). The decision quantity is the peak tree RSS against 20.7 GiB (Audit B's
measured 3 x 3 footprint); the grid itself is the author's decision.

GRIDS (the SP instances, no EV injection, no dependence re-pairing):
  five_block  data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json   (2025/2028/2031/2034/2037, end 2040;
              scenario checksum 0248a5c8...; the instance of step 0 and the base of the dependence instance)
  srp1grid    data/SRP1/Results/P515S56/instances_v1/SRP1__s56_srp1grid_3x3.json  (2025/2030/2035, 12 blocks;
              scenario checksum 73d0a230...)
CONFIGURATION (both grids; existing harness keys only, nothing edited): the step 4 run configuration
(`p515_s56_ev_harness_tests._spec_like`: case-file AA declaration, ESS ageing baseline, convergence-depth tail) +
the derived SP instance; entry options: interface_coupling = COUPLING_V1_NOQ (per-scenario coupling F1, V-1; AA forced
off by production under it), no interface_deviation_premium (production default alpha 0.0), s56_cohort_formulation
'independent_units_all_node_years', s56_terminal_value_in_objective True, s56_esso_slack_penalty 1e4,
flex_price_multiplier 2.0, s56_staged_investment = the mixed-duration staged plan of the EV harness tests
(`p515_s56_ev_harness_tests.STAGED[grid]`, already mapped onto each grid's years). Cap 2, concurrency 1.

SOLVES. This launcher never solves: SolveProfileGuard(permitted=()) installed before any repository import (plus the
guard the imported test module installs), both verified at exactly 0. The harness child's own armed guard counts every
solve; its record's solve_profile must equal the count DECLARED HERE before launch:
  ((1 + n_dso) x n_years x n_days + n_esso_nodes) x (cycles + 1)
  five_block ((1 + 3) x 5 x 4 + 3) x 3 = 249;   srp1grid ((1 + 3) x 3 x 4 + 3) x 3 = 153
(the formula is re-derived from the case file before launch and must give the declared constant).

MEMORY. A sampler thread in this launcher reads, every SAMPLE_S = 1 s, every process of the tree (this launcher, the
harness child, the IPOPT processes): RSS (psutil) and phys_footprint (proc_pid_rusage RUSAGE_INFO_V4), plus vm_stat
(free, inactive, speculative, swapouts, pageouts, compressor). Every sample is written to <grid>/samples.jsonl.
WATCHDOG (declared): tree RSS >= 28 GiB, or swapouts since launch >= 1 GiB -> every descendant of this launcher is sent
SIGTERM (SIGKILL after 15 s); the run is then recorded as stopped ("> 28 GiB" for the RSS trigger).

ISOLATION. Runs from a detached CLONE of the main checkout at the export commit (`--export`), not the live working tree
(another Worker edits the harness there). A plain `git archive` export cannot host the campaign harness: its
freeze_campaign_spec and check_campaign_preconditions call git (HEAD, the case file's last commit, git status of the
production files). The clone is verified clean (git status --porcelain, tracked files) and every root-level *.py is
compared byte for byte with `git show <commit>:<file>`; this script is overlaid (untracked) from the commit that holds it.
EXPORT_PROVENANCE.json at the clone root records both commits and the hashes.

MODES
  --export --commit C --overlay-commit O --dest D     (from the main checkout; zero solves, no project import;
                                                       O = WORKTREE only for a trial clone: --run refuses it)
  --trial --grid G                                    (from the clone; zero solves: freeze into a trial root, capture-
                                                       path assertion, sampler for 5 s, record-reading on the committed
                                                       EV smoke record)
  --run --grid G                                      (from the clone; THE SOLVED RUN; attached, alone, stdout and stderr
                                                       captured by the caller)
  --manifest --dir D                                  (from the main checkout: sha256 of every file under D, with the
                                                       EV-harness commit-selection rule)
Usage (clone root, canonical interpreter, attached, both streams captured, one grid at a time):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt LP_SOLVER_PATH=<clp> <python> -u p515_s56_memory_probe.py --run --grid five_block \\
      > data/SRP1/Results/P515S56/memory_probe_v1/five_block_launch_stdout.log \\
      2> data/SRP1/Results/P515S56/memory_probe_v1/five_block_launch_stderr.log
Objective convention (cost figures copied from the harness's cost breakdown): EUR of total discounted cost; gross =
production's gross_operational_cost (coupled entry: settlement excluded in full), net = gross - TV credit,
total = I(x) + net. Two cycles are not a converged point: costs are carried for identification only.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 memory probe launcher (this process never solves)').install()

import argparse  # noqa: E402
import ctypes  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
SCHEMA = 'p515_s56_memory_probe_v1'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'memory_probe_v1')
PROVENANCE_FILE = 'EXPORT_PROVENANCE.json'
MAIN_CHECKOUT = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'
WORKTREE_OVERLAY = 'WORKTREE'   # --overlay-commit WORKTREE: the script from the working tree (a trial clone only)
SOLVER_PATH = '/usr/local/bin/ipopt'
GIB = float(2 ** 30)
DECISION_GIB = 20.7                  # Audit B's measured 3 x 3 footprint (the Planner's threshold)
STOP_TREE_RSS_BYTES = 28 * 2 ** 30   # watchdog: "> 28 GiB"
STOP_SWAPOUT_BYTES = 1 * 2 ** 30     # watchdog: swapouts since launch
SAMPLE_S = 1.0
OUTSIDE_EVERY = 30                   # samples between reads of the machine's other large processes
OUTSIDE_MIN_BYTES = 2 ** 30
KILL_GRACE_S = 15.0
CAP = 2
GRIDS = {
    'five_block': {'years': [2025, 2028, 2031, 2034, 2037], 'n_days': 4, 'checksum_prefix': '0248a5c8',
                   'declared_solves': ((1 + 3) * 5 * 4 + 3) * (CAP + 1)},   # 249
    'srp1grid': {'years': [2025, 2030, 2035], 'n_days': 4, 'checksum_prefix': '73d0a230',
                 'declared_solves': ((1 + 3) * 3 * 4 + 3) * (CAP + 1)},     # 153
}
LABEL = 'staged_sp'
COHORT_MODE = 'independent_units_all_node_years'
SLACK_PENALTY = 1e4
FLEX_M = 2.0


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [memprobe] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args, cwd=HERE, binary=False):
    out = subprocess.run(['git', *args], cwd=cwd, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def _dump_new(path, obj):
    import gate_result_io as GRIO
    with open(path, 'x') as handle:
        GRIO.dump(obj, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)


# ======================================================================================================================
#  memory readings: vm_stat, proc_pid_rusage footprint, the process-tree sampler and watchdog
# ======================================================================================================================
def vmstat():
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    page = int(text.splitlines()[0].split('page size of')[1].split('bytes')[0].strip())
    raw = {}
    for line in text.splitlines()[1:]:
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().rstrip('.')
            if v.isdigit():
                raw[k.strip().strip('"')] = int(v)
    free, inactive = raw.get('Pages free'), raw.get('Pages inactive')
    return {'utc': _utc(), 't': time.time(), 'page_size': page,
            'free_gib': free * page / GIB, 'inactive_gib': inactive * page / GIB,
            'free_plus_inactive_gib': (free + inactive) * page / GIB,
            'speculative_gib': raw.get('Pages speculative', 0) * page / GIB,
            'wired_gib': raw.get('Pages wired down', 0) * page / GIB,
            'anonymous_gib': raw.get('Anonymous pages', 0) * page / GIB,
            'compressor_occupied_gib': raw.get('Pages occupied by compressor', 0) * page / GIB,
            'swapouts_pages': raw.get('Swapouts'), 'swapins_pages': raw.get('Swapins'),
            'pageouts_pages': raw.get('Pageouts')}


def swapusage():
    try:
        return subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f'unavailable: {error}'


RUSAGE_INFO_V4 = 4
_RUSAGE_V4_FIELDS = (   # = p515_s56_concurrency_gate (W91 ChildFootprintMonitor), copied
    'ri_user_time', 'ri_system_time', 'ri_pkg_idle_wkups', 'ri_interrupt_wkups', 'ri_pageins', 'ri_wired_size',
    'ri_resident_size', 'ri_phys_footprint', 'ri_proc_start_abstime', 'ri_proc_exit_abstime', 'ri_child_user_time',
    'ri_child_system_time', 'ri_child_pkg_idle_wkups', 'ri_child_interrupt_wkups', 'ri_child_pageins',
    'ri_child_elapsed_abstime', 'ri_diskio_bytesread', 'ri_diskio_byteswritten', 'ri_cpu_time_qos_default',
    'ri_cpu_time_qos_maintenance', 'ri_cpu_time_qos_background', 'ri_cpu_time_qos_utility', 'ri_cpu_time_qos_legacy',
    'ri_cpu_time_qos_user_initiated', 'ri_cpu_time_qos_user_interactive', 'ri_billed_system_time',
    'ri_serviced_system_time', 'ri_logical_writes', 'ri_lifetime_max_phys_footprint', 'ri_instructions', 'ri_cycles',
    'ri_billed_energy', 'ri_serviced_energy', 'ri_interval_max_phys_footprint', 'ri_runnable_time')
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
    return {k: int(getattr(info, k)) for k in ('ri_phys_footprint', 'ri_lifetime_max_phys_footprint')}


def _kind(cmd, name):
    text = ' '.join(cmd) if cmd else name
    if 'ipopt' in (name or '').lower() or 'ipopt' in text.lower().split(' ')[0]:
        return 'ipopt'
    if '--child' in (cmd or []):
        return 'harness_child'
    if SCRIPT in text:
        return 'launcher'
    return 'other'


class TreeSampler(threading.Thread):
    """Every SAMPLE_S: RSS and phys_footprint of this launcher and all its descendants, plus vm_stat. Watchdog on the
    declared triggers. Every sample appended to `samples_path` (one JSON line each)."""

    def __init__(self, samples_path, period_s=SAMPLE_S):
        super().__init__(name='tree-sampler', daemon=True)
        import psutil
        self.psutil = psutil
        self.me = psutil.Process(os.getpid())
        self.samples_path = samples_path
        self.period_s = period_s
        self._halt = threading.Event()
        self.base_vm = vmstat()
        self.n = 0
        self.errors = []
        self.peak = None            # the sample with the largest tree RSS
        self.peak_fp = None         # the sample with the largest tree footprint
        self.peak_single = None     # the largest single process RSS seen
        self.max_swapout_delta_bytes = 0
        self.max_compressor_gib = 0.0
        self.min_free_plus_inactive = None
        self.cmd_cache = {}
        self.first_seen = {}        # pid -> {kind, create_time, first_seen_t}
        self.first_ipopt = None
        self.child = None
        self.stopped = None         # watchdog record
        self.max_n_ipopt = 0
        self.child_lifetime_max_footprint = None   # the harness child's ri_lifetime_max_phys_footprint, last live read
        self.outside = []           # every OUTSIDE_EVERY samples: machine processes outside the tree with RSS >= 1 GiB

    def _outside(self, t, tree_pids):
        rows = []
        for p in self.psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
            try:
                if p.info['pid'] in tree_pids or p.info['memory_info'].rss < OUTSIDE_MIN_BYTES:
                    continue
                rows.append({'pid': p.info['pid'], 'name': p.info['name'], 'rss_gib': p.info['memory_info'].rss / GIB,
                             'cmd_head': ' '.join((p.info['cmdline'] or [])[:3])[:160]})
            except Exception:  # noqa: BLE001
                continue
        self.outside.append({'t': t, 'processes': rows})

    def _cmd(self, p):
        if p.pid not in self.cmd_cache:
            try:
                self.cmd_cache[p.pid] = (p.cmdline(), p.name(), p.create_time())
            except self.psutil.Error:
                self.cmd_cache[p.pid] = ([], '?', None)
        return self.cmd_cache[p.pid]

    def _sample(self):
        t = time.time()
        procs = [self.me]
        try:
            procs += self.me.children(recursive=True)
        except self.psutil.Error as error:
            self.errors.append(f'children: {error}')
        rows = []
        for p in procs:
            try:
                rss = p.memory_info().rss
            except self.psutil.Error:
                continue
            cmd, name, ctime = self._cmd(p)
            kind = _kind(cmd, name)
            ru = rusage_v4(p.pid) or {}
            rows.append({'pid': p.pid, 'kind': kind, 'name': name, 'rss': rss,
                         'footprint': ru.get('ri_phys_footprint'),
                         'lifetime_max_footprint': ru.get('ri_lifetime_max_phys_footprint')})
            if p.pid not in self.first_seen:
                self.first_seen[p.pid] = {'kind': kind, 'name': name, 'create_time': ctime, 'first_seen_t': t}
                if kind == 'ipopt' and self.first_ipopt is None:
                    self.first_ipopt = {'pid': p.pid, 'create_time': ctime, 'first_seen_t': t}
                if kind == 'harness_child' and self.child is None:
                    self.child = {'pid': p.pid, 'create_time': ctime, 'first_seen_t': t, 'cmdline_head': cmd[:4]}
        for r in rows:
            if r['kind'] == 'harness_child' and r['lifetime_max_footprint']:
                self.child_lifetime_max_footprint = max(self.child_lifetime_max_footprint or 0,
                                                        r['lifetime_max_footprint'])
        if self.n % OUTSIDE_EVERY == 0:
            self._outside(t, {r['pid'] for r in rows})
        vm = vmstat()
        tree_rss = sum(r['rss'] for r in rows)
        tree_fp = sum(r['footprint'] or 0 for r in rows)
        n_ipopt = sum(1 for r in rows if r['kind'] == 'ipopt')
        big = max(rows, key=lambda r: r['rss']) if rows else None
        swap_delta = ((vm['swapouts_pages'] or 0) - (self.base_vm['swapouts_pages'] or 0)) * vm['page_size']
        s = {'t': t, 'utc': vm['utc'], 'n_procs': len(rows), 'n_ipopt': n_ipopt, 'tree_rss': tree_rss,
             'tree_footprint': tree_fp,
             'largest': {k: big[k] for k in ('pid', 'kind', 'rss', 'footprint')} if big else None,
             'by_kind_rss': {k: sum(r['rss'] for r in rows if r['kind'] == k) for k in
                             ('launcher', 'harness_child', 'ipopt', 'other')},
             'free_plus_inactive_gib': vm['free_plus_inactive_gib'], 'free_gib': vm['free_gib'],
             'inactive_gib': vm['inactive_gib'], 'compressor_occupied_gib': vm['compressor_occupied_gib'],
             'swapouts_pages': vm['swapouts_pages'], 'pageouts_pages': vm['pageouts_pages'],
             'swapout_delta_bytes': swap_delta}
        self.n += 1
        self.max_n_ipopt = max(self.max_n_ipopt, n_ipopt)
        if self.peak is None or tree_rss > self.peak['tree_rss']:
            self.peak = dict(s, processes=rows, vm_stat=vm)
        if self.peak_fp is None or tree_fp > self.peak_fp['tree_footprint']:
            self.peak_fp = {k: s[k] for k in ('t', 'utc', 'tree_rss', 'tree_footprint', 'free_plus_inactive_gib')}
        if big and (self.peak_single is None or big['rss'] > self.peak_single['rss']):
            self.peak_single = dict(big, t=t, utc=vm['utc'])
        self.max_swapout_delta_bytes = max(self.max_swapout_delta_bytes, swap_delta)
        self.max_compressor_gib = max(self.max_compressor_gib, vm['compressor_occupied_gib'])
        fpi = vm['free_plus_inactive_gib']
        self.min_free_plus_inactive = fpi if self.min_free_plus_inactive is None else min(self.min_free_plus_inactive,
                                                                                          fpi)
        with open(self.samples_path, 'a') as handle:
            handle.write(json.dumps(s, sort_keys=True) + '\n')
        return s

    def _stop_tree(self, reason, s):
        _log(f'WATCHDOG: {reason}; stopping the process tree')
        rec = {'reason': reason, 'utc': _utc(), 'tree_rss_gib': s['tree_rss'] / GIB,
               'swapout_delta_gib': s['swapout_delta_bytes'] / GIB, 'signalled': [], 'killed': []}
        try:
            kids = self.me.children(recursive=True)
        except self.psutil.Error:
            kids = []
        for p in kids:
            try:
                p.send_signal(signal.SIGTERM)
                rec['signalled'].append(p.pid)
            except self.psutil.Error:
                pass
        _gone, alive = self.psutil.wait_procs(kids, timeout=KILL_GRACE_S)
        for p in alive:
            try:
                p.kill()
                rec['killed'].append(p.pid)
            except self.psutil.Error:
                pass
        self.stopped = rec

    def run(self):
        while True:
            try:
                s = self._sample()
                if self.stopped is None:
                    if s['tree_rss'] >= STOP_TREE_RSS_BYTES:
                        self._stop_tree('> 28 GiB (tree RSS reached the 28 GiB watchdog)', s)
                    elif s['swapout_delta_bytes'] >= STOP_SWAPOUT_BYTES:
                        self._stop_tree('swapouts since launch reached 1 GiB', s)
            except Exception as error:  # noqa: BLE001 -- recorded
                self.errors.append(f'{type(error).__name__}: {error}')
            if self._halt.wait(self.period_s):
                return

    def stop(self):
        self._halt.set()
        self.join(timeout=self.period_s + KILL_GRACE_S + 30)
        pk = self.peak or {}
        return {
            'measure': ('tree = this launcher + every descendant (harness child, IPOPT processes); RSS = psutil '
                        'memory_info().rss; footprint = proc_pid_rusage(RUSAGE_INFO_V4).ri_phys_footprint; sampled '
                        f'every {self.period_s} s; vm_stat each sample'),
            'period_s': self.period_s, 'n_samples': self.n, 'errors': self.errors[:50], 'n_errors': len(self.errors),
            'thread_alive_after_stop': self.is_alive(),
            'peak_tree_rss_bytes': pk.get('tree_rss'), 'peak_tree_rss_gib': (pk.get('tree_rss') or 0) / GIB,
            'peak_tree_rss_utc': pk.get('utc'), 'peak_tree_rss_by_kind_gib': {
                k: v / GIB for k, v in (pk.get('by_kind_rss') or {}).items()},
            'peak_tree_rss_processes': pk.get('processes'),
            'vm_stat_at_peak': pk.get('vm_stat'),
            'peak_tree_footprint': {**(self.peak_fp or {}),
                                    'tree_footprint_gib': ((self.peak_fp or {}).get('tree_footprint') or 0) / GIB},
            'peak_single_process': dict(self.peak_single or {},
                                        rss_gib=((self.peak_single or {}).get('rss') or 0) / GIB),
            'max_swapout_delta_bytes': self.max_swapout_delta_bytes,
            'max_compressor_occupied_gib': self.max_compressor_gib,
            'min_free_plus_inactive_gib': self.min_free_plus_inactive,
            'max_concurrent_ipopt': self.max_n_ipopt,
            'harness_child': self.child, 'first_ipopt': self.first_ipopt,
            'harness_child_lifetime_max_phys_footprint_gib': (
                self.child_lifetime_max_footprint / GIB if self.child_lifetime_max_footprint else None),
            'outside_tree_processes_ge_1gib': {'every_n_samples': OUTSIDE_EVERY, 'reads': self.outside,
                                               'max_rss_gib_seen': max([q['rss_gib'] for o in self.outside
                                                                        for q in o['processes']] or [0.0])},
            'watchdog': {'tree_rss_stop_gib': STOP_TREE_RSS_BYTES / GIB,
                         'swapout_stop_gib': STOP_SWAPOUT_BYTES / GIB, 'stopped': self.stopped},
            'base_vm_stat': self.base_vm,
        }


def top_processes(n=12):
    """The n largest-RSS processes of the machine (the cleanliness record before / after a run)."""
    import psutil
    rows = []
    for p in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
        try:
            rows.append({'pid': p.info['pid'], 'name': p.info['name'], 'rss_gib': p.info['memory_info'].rss / GIB,
                         'cmd_head': ' '.join((p.info['cmdline'] or [])[:3])[:160]})
        except Exception:  # noqa: BLE001
            continue
    return sorted(rows, key=lambda r: -r['rss_gib'])[:n]


# ======================================================================================================================
#  --export (main checkout; zero solves; no project import)
# ======================================================================================================================
def export(a):
    repo = HERE
    if os.path.exists(a.dest):
        print(f'REFUSE: {a.dest} exists', flush=True)
        return 1
    commit = _git('rev-parse', a.commit, cwd=repo).strip()
    overlay_commit = (WORKTREE_OVERLAY if a.overlay_commit == WORKTREE_OVERLAY
                      else _git('rev-parse', a.overlay_commit, cwd=repo).strip())
    if subprocess.run(['git', 'cat-file', '-e', f'{commit}:{SCRIPT}'], cwd=repo, capture_output=True).returncode == 0:
        print(f'REFUSE: {SCRIPT} is tracked at {commit} (the overlay is an addition only)', flush=True)
        return 2
    if overlay_commit == WORKTREE_OVERLAY:
        with open(os.path.abspath(__file__), 'rb') as handle:
            script_bytes = handle.read()
    else:
        script_bytes = _git('show', f'{overlay_commit}:{SCRIPT}', cwd=repo, binary=True)
    # a detached clone (reads the main repository; writes only under dest), checked out at the export commit
    subprocess.run(['git', 'clone', '--quiet', '--shared', '--no-checkout', repo, a.dest], check=True)
    subprocess.run(['git', '-c', 'advice.detachedHead=false', 'checkout', '--quiet', '--detach', commit], cwd=a.dest,
                   check=True)
    head = _git('rev-parse', 'HEAD', cwd=a.dest).strip()
    status = _git('status', '--porcelain', cwd=a.dest)
    # byte identity of every root-level tracked file with `git show <commit>:<file>` (= the git archive bytes)
    root_tracked = [p for p in _git('ls-tree', '--name-only', commit, cwd=repo).splitlines()
                    if os.path.isfile(os.path.join(a.dest, p))]
    mismatch, root_files = [], {}
    for rel in root_tracked:
        blob = _git('show', f'{commit}:{rel}', cwd=repo, binary=True)
        sha_blob = hashlib.sha256(blob).hexdigest()
        sha_file = _sha(os.path.join(a.dest, rel))
        root_files[rel] = sha_file
        if sha_blob != sha_file:
            mismatch.append(rel)
    dst = os.path.join(a.dest, SCRIPT)
    with open(dst, 'xb') as handle:
        handle.write(script_bytes)
    inputs = {}
    for rel in (os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'SRP1__s53_3x3.json'),
                os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'instance_record.json'),
                os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'SRP1__s56_srp1grid_3x3.json'),
                os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1', 'instance_record.json')):
        inputs[rel] = {'sha256_in_clone': _sha(os.path.join(a.dest, rel)),
                       'sha256_at_commit': hashlib.sha256(_git('show', f'{commit}:{rel}', cwd=repo,
                                                               binary=True)).hexdigest()}
    prov = {'export_commit': commit, 'head_in_clone': head, 'overlay_commit': overlay_commit, 'utc': _utc(),
            'method': ('git clone --shared --no-checkout <main checkout> <dest>; git checkout --detach <commit>; '
                       f'{SCRIPT} written from `git show <overlay commit>:{SCRIPT}` (untracked in the clone)'),
            'why_not_git_archive': ('the campaign harness calls git in freeze_campaign_spec (rev-parse HEAD, the case '
                                    'file and ESS params last commit) and check_campaign_preconditions (git status of '
                                    'the production files); a git archive export has no repository'),
            'git_status_porcelain_after_checkout': status, 'clean': status.strip() == '',
            'root_tracked_files_sha256': root_files, 'root_files_differing_from_git_show': mismatch,
            'overlay': {SCRIPT: {'sha256': _sha(dst), 'from_commit': overlay_commit}},
            'results_inputs': inputs,
            'export_script': SCRIPT, 'export_script_sha256_in_main_checkout': _sha(os.path.abspath(__file__))}
    with open(os.path.join(a.dest, PROVENANCE_FILE), 'x', encoding='utf-8') as handle:
        json.dump(prov, handle, indent=1, sort_keys=True)
    ok = head == commit and status.strip() == '' and not mismatch and all(
        v['sha256_in_clone'] == v['sha256_at_commit'] for v in inputs.values())
    print(f'export {commit} -> {a.dest}: head {head}, clean {status.strip() == ""}, {len(root_files)} root files, '
          f'{len(mismatch)} differing, overlay {SCRIPT} from {overlay_commit}; ok {ok}', flush=True)
    return 0 if ok else 3


# ======================================================================================================================
#  run-side helpers (the clone)
# ======================================================================================================================
def provenance_check(trial):
    out = {'cwd': os.getcwd(), 'here': HERE, 'interpreter': sys.executable,
           'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH'), 'lp_solver_path_env': os.environ.get('LP_SOLVER_PATH'),
           'script_sha256': _sha(os.path.abspath(__file__))}
    bad = []
    path = os.path.join(HERE, PROVENANCE_FILE)
    if not os.path.exists(path):
        bad.append(f'{PROVENANCE_FILE} missing (not a provenance-recorded clone)')
        return out, bad
    with open(path) as handle:
        prov = json.load(handle)
    out['export'] = {k: prov.get(k) for k in ('export_commit', 'overlay_commit', 'utc', 'method')}
    head = _git('rev-parse', 'HEAD').strip()
    out['git_head'] = head
    out['git_status_tracked'] = _git('status', '--porcelain', '--untracked-files=no')
    if head != prov.get('export_commit'):
        bad.append(f'HEAD {head} != export commit {prov.get("export_commit")}')
    if out['git_status_tracked'].strip():
        bad.append(f'tracked files modified in the clone: {out["git_status_tracked"]}')
    if prov.get('overlay', {}).get(SCRIPT, {}).get('sha256') != out['script_sha256']:
        bad.append(f'{SCRIPT} sha256 differs from the export record')
    for rel, sha in prov.get('root_tracked_files_sha256', {}).items():
        if _sha(os.path.join(HERE, rel)) != sha:
            bad.append(f'{rel}: sha256 differs from the export record')
    if os.environ.get('NLP_SOLVER_PATH') != SOLVER_PATH:
        bad.append(f'NLP_SOLVER_PATH is {os.environ.get("NLP_SOLVER_PATH")!r}, not {SOLVER_PATH}')
    if os.path.realpath(HERE) == os.path.realpath(MAIN_CHECKOUT):
        bad.append('running from the main checkout')
    if prov.get('overlay_commit') == WORKTREE_OVERLAY:
        bad.append('the overlay was taken from the working tree (trial clones only)')
    out['harness_sha256'] = _sha(os.path.join(HERE, 'p515_s44_campaign_harness.py'))
    return out, bad


def derive_solves(case_rel):
    with open(os.path.join(HERE, case_rel)) as handle:
        case = json.load(handle)
    n_years, n_days = len(case['Years']), len(case['Days'])
    n_dso = len(case['DistributionNetworks'])
    n_esso = n_dso   # one shared ESS per DN connection node (nodes 5, 7, 9)
    return {'n_years': n_years, 'n_days': n_days, 'n_dso': n_dso, 'n_esso_nodes': n_esso, 'cycles': CAP,
            'years': sorted(int(y) for y in case['Years']),
            'formula': '((1 + n_dso) x n_years x n_days + n_esso_nodes) x (cycles + 1)',
            'value': ((1 + n_dso) * n_years * n_days + n_esso) * (CAP + 1)}


def build_entry(T, H, grid):
    derived = T.sp_declaration(grid)
    cfg, required = T._spec_like(derived, cap=CAP)
    cfg['name'] = (f'S56 memory probe v1 ({grid}): corrected SP, staged mixed-duration plan, V-1 per-scenario coupling, '
                   f'alpha 0, cohort all node-years, TV, slack 1e4, flex x2, cap 2')
    plan = T.STAGED[grid]
    options = {H.S56_COHORT_FORMULATION_KEY: COHORT_MODE, H.S56_TV_IN_OBJECTIVE_KEY: True,
               H.S56_ESSO_SLACK_PENALTY_KEY: SLACK_PENALTY, H.S56_STAGED_INVESTMENT_KEY: json.loads(json.dumps(plan)),
               'flex_price_multiplier': FLEX_M, 'interface_coupling': dict(T.COUPLING_V1_NOQ)}
    return derived, cfg, required, plan, options


def lattice_key_check(H, entry, derived):
    import p516_lattice as L
    case_rel = H.s56_case_rel(derived)
    lat = H.s56_lattice(case_rel)
    z = lat.z_of_canonical(entry['canonical'])
    key = lat.key(z)
    return {'case_rel': case_rel, 'lattice_key': key, 'entry_candidate_key': entry['key'],
            'equal': key == entry['key'], 'lattice_label': lat.label(z), 'key_schema': L.KEY_SCHEMA,
            'c5_reasons': lat.reasons(z)}


def read_record(H, rec, ed, grid, declared_solves):
    """Readings and pass/fail checks from the evaluation record and its eval dir (zero solves)."""
    per_cycle = []
    pc_path = os.path.join(ed, 'per_cycle_record.jsonl')
    if os.path.exists(pc_path):
        with open(pc_path) as handle:
            per_cycle = [json.loads(t) for t in handle if t.strip()]
    exit_code = None
    ec_path = os.path.join(ed, 'exit_code.txt')
    if os.path.exists(ec_path):
        with open(ec_path) as handle:
            exit_code = handle.read().strip()
    pv = rec.get('parent_view') or {}
    sp_ = rec.get('solve_profile') or {}
    obs = sp_.get('observed') or {}
    applied = rec.get('s56_ev_plan_applied_in_child') or {}
    in_force = applied.get('in_force') or {}
    probe = applied.get('probe_readback') or {}
    staged = applied.get('staged') or {}
    flex = rec.get('flex_price_applied_in_child') or {}
    flex_t = rec.get('flex_price_readback_terminal') or {}
    ica = rec.get('interface_coupling_applied_in_child') or {}
    cb = rec.get('s56_cost_breakdown') or {}
    nfs = (rec.get('network_failures_summary') or {}).get('classes') or {}
    checks = {
        'exit_0': {'ok': pv.get('exit_code') == 0 or (pv.get('exit_code') is None and exit_code == '0'),
                   'detail': {'parent_view_exit_code': pv.get('exit_code'), 'exit_code_txt': exit_code,
                              'status': rec.get('status')}},
        'cycles_run_2': {'ok': rec.get('cycles_run') == CAP, 'detail': rec.get('cycles_run')},
        'solve_count_exact': {'ok': (obs.get('permitted_solve') == declared_solves
                                     and obs.get('permitted_exec') == declared_solves
                                     and obs.get('blocked_solve') == 0 and obs.get('blocked_exec') == 0
                                     and sp_.get('base_solves') == declared_solves
                                     and sp_.get('retry_solves_credited') == 0 and sp_.get('identity_holds') is True),
                              'detail': {'declared': declared_solves, 'observed': obs,
                                         'base_solves': sp_.get('base_solves'),
                                         'retry_solves_credited': sp_.get('retry_solves_credited'),
                                         'identity_holds': sp_.get('identity_holds')}},
        'no_local_solve_failure': {'ok': rec.get('local_solve_failures') == 0
                                   and all(v == 0 for v in nfs.values())
                                   and (rec.get('network_failures_summary') or {}).get('n_esso_recovery_events') == 0,
                                   'detail': {'local_solve_failures': rec.get('local_solve_failures'),
                                              'network_failures_classes': nfs,
                                              'n_esso_recovery_events': (rec.get('network_failures_summary') or {}).get(
                                                  'n_esso_recovery_events')}},
        'readback_cohort_mode': {'ok': in_force.get('cohort_formulation') == COHORT_MODE and bool(probe) and all(
            v.get('model_cohort_formulation') == COHORT_MODE and v.get('row_inventory_ok') is True
            for v in probe.values()), 'detail': {'in_force': in_force.get('cohort_formulation'), 'per_node': {
                n: (v.get('model_cohort_formulation'), v.get('row_inventory_ok')) for n, v in probe.items()}}},
        'readback_kappa_tv': {'ok': in_force.get('salvage_value_in_objective') is True and bool(probe) and all(
            v.get('tv_ok') is True and (v.get('tv') or {}).get('has_kappa') is True for v in probe.values())
            and (cb.get('checks') or {}).get('kappa_equals_a_over_sigma_where_built') is True
            and (cb.get('checks') or {}).get('kappa_built_iff_tv_declared') is True,
            'detail': {'in_objective': in_force.get('salvage_value_in_objective'),
                       'per_node_tv_ok': {n: v.get('tv_ok') for n, v in probe.items()},
                       'cost_breakdown_checks': cb.get('checks')}},
        'readback_slack_penalty_1e4': {'ok': in_force.get('esso_slack_penalty') == SLACK_PENALTY and bool(probe) and all(
            v.get('slack_ok') is True for v in probe.values()),
            'detail': {'in_force': in_force.get('esso_slack_penalty'),
                       'per_node': {n: v.get('slack_coefficients') for n, v in probe.items()}}},
        'readback_flex_x2': {'ok': rec.get('flex_price_multiplier') == FLEX_M
                             and flex.get('flex_price_multiplier') == FLEX_M
                             and all(v is True for v in (flex.get('checks') or {}).values()) and bool(flex.get('checks'))
                             and (flex.get('readback_pre_run') or {}).get('all_match') is True
                             and flex_t.get('all_match') is True,
                             'detail': {'checks': flex.get('checks'),
                                        'pre_run_n_coefficients': (flex.get('readback_pre_run') or {}).get(
                                            'n_flex_coefficients'),
                                        'terminal_all_match': flex_t.get('all_match'),
                                        'terminal_n_coefficients': flex_t.get('n_coefficients')}},
        'readback_staged_plan': {'ok': staged.get('written_equals_canonical') is True and applied.get('all_ok') is True
                                 and applied.get('settings_ok') is True,
                                 'detail': {'written_equals_canonical': staged.get('written_equals_canonical'),
                                            'all_ok': applied.get('all_ok'), 'written': staged.get('written')}},
        'per_scenario_coupling_active': {
            'ok': (ica.get('in_force') == {'per_scenario_interface_coupling': True,
                                           'per_scenario_voltage_mode': 'per_scenario', 'row18_charge_reactive': False}
                   and (ica.get('checks') or {}).get('in_force_equals_declaration') is True
                   and len(per_cycle) == CAP and all(c.get('interface_coupling_active') is True
                                                     and c.get('interface_coupling_aa_forced_off') is True
                                                     for c in per_cycle)),
            'detail': {'in_force': ica.get('in_force'), 'checks': ica.get('checks'),
                       'per_cycle': [{k: c.get(k) for k in ('cycle', 'interface_coupling_active',
                                                            'interface_coupling_voltage_mode',
                                                            'interface_coupling_aa_forced_off')} for c in per_cycle]}},
        'hook_order': {'ok': rec.get('s56_hook_order') == ['esso_settings_and_staged_write', 'flex_price_multiplier'],
                       'detail': rec.get('s56_hook_order')},
        'cost_breakdown_written': {'ok': cb.get('status') == 'written' and cb.get('harness_checks_ok') is True,
                                   'detail': {'status': cb.get('status'), 'checks': cb.get('checks')}},
    }
    readings = {
        'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
        'cycle_wall_s': [c.get('cycle_wall_s') for c in per_cycle],
        'rss_bytes_per_cycle_child': [c.get('rss_bytes') for c in per_cycle],
        'wall_time_s': rec.get('wall_time_s'), 'peak_rss_record': rec.get('peak_rss'),
        'solve_profile': sp_,
        'cost_breakdown_summary': {k: cb.get(k) for k in (
            'investment_cost_eur', 'gross_operational_cost_eur', 'terminal_value_credit_eur',
            'net_operational_recourse_eur', 'total_planning_cost_eur', 'flexibility_cost_reactive_down_eur',
            'flexibility_active_down_eur', 'esso_slack_totals', 'leak_detector', 'exit_certificate_e6_summary')},
        'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost'),
        'terminal_net_operational_recourse': rec.get('terminal_net_operational_recourse'),
        'interface_coupling_applied_in_child': ica,
        's56_ev_plan_in_force': in_force,
    }
    return checks, readings


def _guards(extra):
    out = {'own': {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_0': GUARD.verify(0)}}
    for label, g in extra:
        out[label] = {'label': g.label, 'counts': dict(g.counts), 'verify_0': g.verify(0)}
    return out, all(not v['verify_0'] for v in out.values())


# ======================================================================================================================
#  --trial / --run (the clone)
# ======================================================================================================================
def run(a, trial):
    grid = a.grid
    g = GRIDS[grid]
    out_dir = os.path.join(HERE, OUT_REL, grid + ('_trial' if trial else ''))
    if os.path.exists(out_dir):
        raise SystemExit(f'REFUSED: {os.path.relpath(out_dir, HERE)} exists (write-once)')
    prov, bad = provenance_check(trial)
    if bad and not trial:
        raise SystemExit(f'REFUSED before launch (zero solves): {bad}')
    os.makedirs(out_dir)
    t0 = time.time()
    res = {'schema': SCHEMA, 'mode': 'trial' if trial else 'run', 'grid': grid, 'started_utc': _utc(),
           'provenance': prov, 'provenance_failures': bad, 'error': None,
           'declared': {'cap': CAP, 'concurrency': 1, 'label': LABEL, 'solves': g['declared_solves'],
                        'years': g['years'], 'decision_threshold_gib': DECISION_GIB,
                        'watchdog': {'tree_rss_stop_gib': STOP_TREE_RSS_BYTES / GIB,
                                     'swapout_stop_gib': STOP_SWAPOUT_BYTES / GIB, 'sample_s': SAMPLE_S}}}
    checks = {}
    extra_guards = []
    sampler = None
    try:
        import p515_s44_campaign_harness as H
        import p515_s56_ev_harness_tests as T   # installs its own permitted=() guard (verified at 0 below)
        extra_guards.append(('p515_s56_ev_harness_tests', T.GUARD))
        derived, cfg, required, plan, options = build_entry(T, H, grid)
        sol = derive_solves(derived['case_path'])
        res['solves_derivation'] = sol
        checks['declared_solves_equal_case_derivation'] = {
            'ok': sol['value'] == g['declared_solves'] and sol['years'] == g['years'] and sol['n_days'] == g['n_days'],
            'detail': sol}
        checks['instance_checksum'] = {'ok': derived['scenario_checksum'].startswith(g['checksum_prefix']),
                                       'detail': derived['scenario_checksum']}
        res['configuration'] = {'derived_instance': derived, 'cfg': cfg, 'required_consecutive_cycles': required,
                                'staged_plan': plan, 'entry_options': options,
                                'alpha': 'interface_deviation_premium not declared: production default alpha 0.0',
                                'not_declared': {'s56_ev_profiles': 'SP instance (no EV injection)'}}
        root = os.path.join(out_dir, f'campaign_s56memprobe_v1_{grid}' + ('_trial' if trial else ''))
        failures = H.check_campaign_preconditions(root)
        res['preconditions'] = failures
        if failures:
            raise RuntimeError(f'refused by the harness preconditions (zero solves): {failures}')
        spec_path, csha, cspec = H.freeze_campaign_spec(
            root, f's56memprobe_v1_{grid}' + ('_trial' if trial else ''), [(LABEL, None, options)], cfg, CAP, 1,
            ['TASKS.md "Working mode and EV priority", critical path step 3 (memory probe); engineering measurement'],
            required_consecutive_cycles=required,
            extra={'campaign_script': SCRIPT, 'campaign_script_sha256': prov['script_sha256'],
                   'declared_solves_per_evaluation': g['declared_solves'], 'mode': res['mode']})
        entry = cspec['candidates'][0]
        res['campaign'] = {'root': os.path.relpath(root, HERE), 'spec': os.path.relpath(spec_path, HERE),
                           'spec_sha256': csha,
                           'entry': {k: entry.get(k) for k in ('label', 'key', 'eval_key', 'eval_dir', 'canonical',
                                                                's56_investment', 'interface_coupling',
                                                                'flex_price_multiplier')}}
        lk = lattice_key_check(H, entry, derived)
        res['staged_key'] = lk
        checks['staged_key_equals_general_lattice_key'] = {'ok': lk['equal'] and not lk['c5_reasons'], 'detail': lk}
        s56plan = H.s56_child_declaration(cspec, entry)
        checklist = H.assert_s56_ev_plan_capture_path(s56plan)
        checks['capture_path_asserted_before_any_solve'] = {
            'ok': all(v for v in checklist.values() if isinstance(v, bool)), 'detail': checklist}
        res['vm_stat_before'] = vmstat()
        res['swapusage_before'] = swapusage()
        res['top_processes_before'] = top_processes()
        sampler = TreeSampler(os.path.join(out_dir, 'samples.jsonl'))
        sampler.start()
        if trial:
            time.sleep(5)
            src = os.path.join(HERE, 'data', 'SRP1', 'Results', 'P515S56', 'ev_harness_v1', 'smoke',
                               'campaign_s56evh_v1_smoke', 'evals')
            unit = [d for d in sorted(os.listdir(src)) if d.endswith('_unit')][0]
            with open(os.path.join(src, unit, 'evaluation_record.json')) as handle:
                rec = json.load(handle)
            c2, r2 = read_record(H, rec, os.path.join(src, unit), grid, 153)
            res['trial_record_reading'] = {'source': os.path.relpath(os.path.join(src, unit), HERE),
                                           'checks': {k: v['ok'] for k, v in c2.items()},
                                           'cycle_wall_s': r2['cycle_wall_s']}
        else:
            lock = H.acquire_campaign_lock(cspec['campaign_id'], csha)
            _log(f'{grid}: lock {lock}; spec {os.path.relpath(spec_path, HERE)}; declared solves {g["declared_solves"]}')
            records = None
            try:
                ctx = H.CampaignContext(root, spec_path, csha, cspec, log=_log)
                records = H.evaluate([LABEL], ctx)
            finally:
                res['lock_released'] = H.release_campaign_lock(expected_pid=os.getpid())
            rec = (records or [{}])[0]
            ed = os.path.join(root, 'evals', entry['eval_dir'])
            res['eval_dir'] = os.path.relpath(ed, HERE)
            c2, r2 = read_record(H, rec, ed, grid, g['declared_solves'])
            checks.update(c2)
            res['readings'] = r2
    except Exception as error:  # noqa: BLE001 -- recorded (a harness fault or a refusal)
        res['error'] = f'{type(error).__name__}: {error}'
        res['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if sampler is not None:
            res['memory'] = sampler.stop()
        res['vm_stat_after'] = vmstat()
        res['swapusage_after'] = swapusage()
        time.sleep(0 if trial else 20)
        res['vm_stat_after_20s'] = vmstat()
        res['top_processes_after'] = top_processes()
    mem = res.get('memory') or {}
    if mem:
        child = mem.get('harness_child') or {}
        first = mem.get('first_ipopt') or {}
        res['build'] = {
            'definition': ('harness child process creation -> creation of the first IPOPT process in the tree (the '
                           'initialisation solve round starts); includes interpreter start, imports, the derived-'
                           'instance read, model build, the configuration hook and its read-back probes'),
            'build_s': (first['create_time'] - child['create_time']) if (first.get('create_time') and
                                                                         child.get('create_time')) else None,
            'child_create_time': child.get('create_time'), 'first_ipopt_create_time': first.get('create_time')}
        stopped = (mem.get('watchdog') or {}).get('stopped')
        if not trial:
            checks['watchdog_not_triggered'] = {'ok': stopped is None, 'detail': stopped}
        peak = mem.get('peak_tree_rss_gib')
        res['decision'] = {
            'quantity': 'peak tree RSS (GiB = 2^30 bytes) against 20.7 GiB',
            'peak_tree_rss_gib': peak if stopped is None or not stopped['reason'].startswith('>') else '> 28 GiB',
            'threshold_gib': DECISION_GIB,
            'within_threshold': (peak is not None and peak <= DECISION_GIB) if stopped is None else False}
    gd, zero = _guards(extra_guards)
    res['guards'] = gd
    res['launcher_zero_solve_verified'] = zero
    checks['launcher_zero_solves'] = {'ok': zero, 'detail': gd}
    res['checks'] = checks
    failed = sorted(k for k, v in checks.items() if not v['ok'])
    res['summary'] = {'n_checks': len(checks), 'n_pass': len(checks) - len(failed), 'failed': failed,
                      'all_pass': bool(checks) and not failed and res['error'] is None}
    res['wall_s'] = time.time() - t0
    res['finished_utc'] = _utc()
    _dump_new(os.path.join(out_dir, f'memory_probe_{grid}{"_trial" if trial else ""}.json'), res)
    for k in sorted(checks):
        _log(f'{"PASS" if checks[k]["ok"] else "FAIL"} {k}' + ('' if checks[k]['ok']
                                                                else f' -- {str(checks[k]["detail"])[:500]}'))
    _log(f'[{grid}] {"PASS" if res["summary"]["all_pass"] else "FAIL"}: {res["summary"]["n_pass"]}/'
         f'{res["summary"]["n_checks"]}; error {res["error"]}; peak tree RSS {mem.get("peak_tree_rss_gib")} GiB; '
         f'decision {res.get("decision")}')
    return 0 if res['summary']['all_pass'] else 2


# ======================================================================================================================
#  --manifest (main checkout)
# ======================================================================================================================
COMMIT_EXCLUDED_PARTS = ('/scratch', '/planning_read/', '/results/', '/esso_capture/')   # = p515_s56_ev_harness_tests
COMMIT_EXCLUDED_SUFFIXES = ('.pkl', '.xlsx', '.pdf', '.png', '.nl')
COMMIT_MAX_BYTES = 1_300_000


def committed_by_rule(rel, size):
    if any(part in '/' + rel for part in COMMIT_EXCLUDED_PARTS) or rel.endswith(COMMIT_EXCLUDED_SUFFIXES):
        return False
    return size <= COMMIT_MAX_BYTES


def manifest(a):
    base = os.path.join(HERE, a.dir)
    target = os.path.join(base, 'manifest_sha256.json')
    if os.path.exists(target):
        raise SystemExit(f'REFUSED: {target} exists (write-once)')
    files = {}
    for root, _dirs, names in os.walk(base):
        for name in sorted(names):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, HERE)
            size = os.path.getsize(path)
            files[rel] = {'sha256': _sha(path), 'bytes': size, 'committed': committed_by_rule(rel, size)}
    obj = {'schema': SCHEMA + '_manifest', 'utc': _utc(), 'git_head': _git('rev-parse', 'HEAD').strip(),
           'selection_rule': {'excluded_parts': list(COMMIT_EXCLUDED_PARTS),
                              'excluded_suffixes': list(COMMIT_EXCLUDED_SUFFIXES), 'max_bytes': COMMIT_MAX_BYTES,
                              'source': 'p515_s56_ev_harness_tests.committed_by_rule (no size exemption)'},
           'n_files': len(files), 'n_committed': sum(1 for v in files.values() if v['committed']),
           'files': dict(sorted(files.items()))}
    with open(target, 'x') as handle:
        json.dump(obj, handle, indent=1, sort_keys=True)
    _log(f'manifest: {len(files)} files ({obj["n_committed"]} committed) -> {os.path.relpath(target, HERE)} '
         f'({_sha(target)})')
    return 0


def main():
    ap = argparse.ArgumentParser()
    for flag in ('--export', '--trial', '--run', '--manifest'):
        ap.add_argument(flag, action='store_true')
    ap.add_argument('--grid', choices=tuple(GRIDS))
    ap.add_argument('--commit')
    ap.add_argument('--overlay-commit')
    ap.add_argument('--dest')
    ap.add_argument('--dir')
    a = ap.parse_args()
    if a.export and a.commit and a.overlay_commit and a.dest:
        return export(a)
    if (a.trial or a.run) and a.grid:
        os.chdir(HERE)
        return run(a, trial=a.trial)
    if a.manifest and a.dir:
        return manifest(a)
    ap.error('one mode with its arguments')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
