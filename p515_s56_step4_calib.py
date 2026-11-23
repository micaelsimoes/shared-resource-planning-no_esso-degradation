"""P5.15 S56 step 4 CALIBRATION RE-RUN -- arm V-1 only (F1 + per-scenario V, alpha 0, x0) on instance A (2025|Winter)
and instance B ({2025, 2035} x Summer), each to a declared cycle cap (A 140, B 320) with production's certification
stop DISABLED for the arm, under ONE frozen spec (`frozen_step4_calib_spec_v3_<sha8>.json`, derived from spec v2
7ca210fb by `p515_s56_step4_calib_spec_v3_derive.py`).

Authority: TASKS.md "Expert rulings on handback 3 (2026-11-23)" item 1 (decision 1 (a) approved; caps, decision rule,
interface V upper-bound multiplier capture); the Planner's task of 2026-11-23 (calibration re-run).

EACH RUN = ONE harness campaign (one entry, concurrency 1, a NEW absolute root under step4_calib_v1/), A first, then B
(the harness campaign lock is one per checkout). The entry is spec v2's V-1 entry (options `SM.arm_options('V1')`)
plus the two harness keys of 091b7745:
  `s56_certification_stop_disabled` True   the certificate length is raised to 10**9 after production's tail-baseline
                                           call (the only effect: the loop runs to the cap);
  `s56_interface_v_multiplier_capture` True  per-cycle interface V upper-limit multipliers, both sides (read-only).
Neither key enters the eval key; the V-1 eval key equals step 4 main's.

STOPS (declared in the spec; the launcher's monitor thread raises SIGINT in this process -- the harness's own interrupt
path: it SIGTERMs and reaps the child, records written so far are kept):
  * PREFIX: cycles 1..46 (A) / 1..107 (B) must equal step 4 main's V-1 run (spec v2, committed 5f1db156) bitwise, the
    timing fields of spec v2's M3 list excluded (declared in advance): s56_cycle_capture.jsonl and
    interface_duals_per_cycle.jsonl cycle lines, network_ipopt_solve_records rounds 0..P (append file), checked LIVE
    as each line is written; per_cycle_record.jsonl lines 1..P checked after the run. A difference stops the run.
  * PROJECTION (B only): once cycle 20's capture line is written, mean s/cycle = (t(20) - t(1)) / 19 (t = the time the
    monitor observed the line), projection = (t(1) - t_launch) + mean x (cap - 1); > 4.5 h stops the run.
  * a harness fault (exception, guard mismatch, child exit != 0).
Solves: this process never solves (SolveProfileGuard(permitted=()), verified at 0); the child's armed guard counts every
solve and its record's solve_profile reconciles it (A 7, B 11 per round x (cycles + 1) + retries).

MODES:
  --check --spec S                        zero solves: every pre-launch assertion (capture paths, references, hashes)
  --run --spec S --instance A|B           one campaign, attached; stdout / stderr captured by the caller
Usage: NLP_SOLVER_PATH=/usr/local/bin/ipopt <canonical python> -u p515_s56_step4_calib.py --run --spec S --instance A \
       > <stdout log> 2> <stderr log>
Objective convention (every Q): EUR of total discounted cost; V-1's primary gross_operational_cost = settlement
excluded in full (pin excluded, salvage separate); e_b(k) = Q_b(k) - w_b x C_central (step 3 gross EUR/day x w_b).
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 step 4 calibration launcher (this process never solves)'
                          ).install()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402  (parent side: no model imports)
import p515_s56_step4_smoke as SM  # noqa: E402  (helpers only; installs its own permitted=() guard)
import p515_s56_step4_main as L  # noqa: E402  (helpers only; installs its own permitted=() guard)

SOLVER_PATH = '/usr/local/bin/ipopt'
RES = os.path.join('data', 'SRP1', 'Results', 'P515S56')
CALIB_DIR = os.path.join(RES, 'step4_calib_v1')
SPEC_PREFIX = 'frozen_step4_calib_spec_v3_'
DERIVE_SCRIPT = 'p515_s56_step4_calib_spec_v3_derive.py'
CHECKS_SCRIPT = 'p515_s56_step4_calib_checks.py'
PHASE = 'calib'
INSTANCES = ('A', 'B')
ARM = 'V1'
POLL_S = 5.0
VMSTAT_EVERY_S = 3600.0
GiB = 1024.0 ** 3


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [S56 step4 calib] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout


def _load(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path, obj):
    with open(path, 'x', encoding='utf-8') as handle:
        GRIO.dump(obj, handle, indent=1, default=GRIO.json_default_item)


def label_of(arm=ARM):
    return f'x0_{arm}_{PHASE}'


def load_spec(path):
    spec = _load(path)
    sha = _sha(path)
    tag = os.path.basename(path).rsplit('_', 1)[-1].split('.')[0]
    if not os.path.basename(path).startswith(SPEC_PREFIX) or not sha.startswith(tag):
        raise RuntimeError(f'spec {path}: name / sha256 {sha[:8]} mismatch')
    return spec, sha, os.path.relpath(os.path.abspath(path), REPO)


def calib_options(spec):
    """Spec v2's V-1 options (the smoke's construction) plus the two harness keys; must equal the spec's arm."""
    cg = SM.coupling_gates_literals()
    options = SM.arm_options(ARM, cg)
    options[H.S56_CERTIFICATION_STOP_DISABLED_KEY] = True
    options[H.S56_INTERFACE_V_MULTIPLIER_CAPTURE_KEY] = True
    if options != spec['i_arms_instances_cap']['arms'][ARM]['options']:
        raise RuntimeError(f'spec V-1 options differ from the launcher construction: {options}')
    return options


def reference_eval_dir(spec, inst):
    ref = spec['prefix_gate']['reference'][inst]
    return os.path.join(REPO, ref['eval_dir'])


def other_copies_of_this_launcher():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    mine = H._ancestor_pids()
    out = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 2 or int(tok[0]) in mine:
            continue
        if any(os.path.basename(t) == SCRIPT for t in tok[1:6]) and '--run' in tok:
            out.append(ln.strip()[:300])
    return out


# ======================================================================================================================
#  prefix comparison (canonical text with the declared keys removed)
# ======================================================================================================================
def _strip(obj, keys):
    if isinstance(obj, dict):
        return {k: _strip(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip(v, keys) for v in obj]
    return obj


def canon(obj, keys):
    return json.dumps(_strip(obj, keys), sort_keys=True)


def canon_sha(obj, keys):
    return hashlib.sha256(canon(obj, keys).encode()).hexdigest()


def _differing_keys(a, b, keys):
    a, b = _strip(a, keys), _strip(b, keys)
    if isinstance(a, dict) and isinstance(b, dict):
        return sorted(k for k in set(a) | set(b) if json.dumps(a.get(k), sort_keys=True)
                      != json.dumps(b.get(k), sort_keys=True))
    return ['<whole>']


def _jsonl_lines(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(t) for t in handle if t.strip()]


def reference_items(ref_dir, prefix, keys):
    """{file: [(item id, canonical sha256)]} of the reference's prefix, in file order."""
    out = {}
    for name in ('s56_cycle_capture.jsonl', 'interface_duals_per_cycle.jsonl'):
        out[name] = [(x['cycle'], canon_sha(x, keys)) for x in _jsonl_lines(os.path.join(ref_dir, name))
                     if not x.get('header') and x.get('cycle') is not None and x['cycle'] <= prefix]
    recs = [x for x in _jsonl_lines(os.path.join(ref_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE))
            if x.get('round') is not None and x['round'] <= prefix]
    out[H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE] = [((x['round'], i), canon_sha(x, keys)) for i, x in enumerate(recs)]
    return out


def reference_line(ref_dir, name, item_id, prefix):
    if name == H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE:
        recs = [x for x in _jsonl_lines(os.path.join(ref_dir, name)) if x.get('round') is not None
                and x['round'] <= prefix]
        return recs[item_id[1]]
    return next(x for x in _jsonl_lines(os.path.join(ref_dir, name)) if not x.get('header')
                and x.get('cycle') == item_id)


class TailReader:
    """Incremental reader of a growing jsonl file: complete lines only."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.buf = b''

    def new_lines(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, 'rb') as handle:
            handle.seek(self.offset)
            data = handle.read()
        self.offset += len(data)
        self.buf += data
        parts = self.buf.split(b'\n')
        self.buf = parts[-1]
        return [json.loads(p) for p in parts[:-1] if p.strip()]


class Monitor(threading.Thread):
    """Live prefix gate, the B projection stop and hourly vm_stat (module docstring). Never solves."""

    def __init__(self, inst, eval_dir, ref_dir, ref_items, prefix, cap, keys, projection, rec):
        super().__init__(daemon=True)
        self.inst, self.eval_dir, self.ref_dir, self.prefix, self.cap, self.keys = (
            inst, eval_dir, ref_dir, prefix, cap, keys)
        self.ref = ref_items
        self.projection = projection
        self.rec = rec
        self.stop_event = threading.Event()
        self.readers = {n: TailReader(os.path.join(eval_dir, n)) for n in ref_items}
        self.pos = {n: 0 for n in ref_items}
        self.compared = {n: 0 for n in ref_items}
        self.mismatches = []
        self.t_obs = {}
        self.t_launch = None
        self.stop_fired = None
        self.last_vmstat = time.time()
        self.vmstat_files = []
        self.errors = []

    def _fire(self, reason):
        if self.stop_fired is None:
            self.stop_fired = {'utc': _utc(), 'reason': reason}
            _log(f'STOP ({self.inst}): {reason}')
            os.kill(os.getpid(), signal.SIGINT)

    def _check_file(self, name):
        for x in self.readers[name].new_lines():
            if name == H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE:
                if x.get('round') is None or x['round'] > self.prefix:
                    continue
            else:
                if x.get('header') or x.get('cycle') is None:
                    continue
                if name == H.S56_CYCLE_CAPTURE_FILE:
                    self.t_obs.setdefault(x['cycle'], time.time())
                if x['cycle'] > self.prefix:
                    continue
            i = self.pos[name]
            self.pos[name] += 1
            if i >= len(self.ref[name]):
                self.mismatches.append({'file': name, 'position': i, 'why': 'more prefix items than the reference'})
                continue
            ref_id, ref_sha = self.ref[name][i]
            my_id = (x['round'], i) if name == H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE else x['cycle']
            self.compared[name] += 1
            if my_id != ref_id or canon_sha(x, self.keys) != ref_sha:
                try:
                    keys = _differing_keys(reference_line(self.ref_dir, name, ref_id, self.prefix), x, self.keys)
                except Exception as error:  # noqa: BLE001
                    keys = [f'<could not reload the reference line: {error}>']
                self.mismatches.append({'file': name, 'position': i, 'reference_id': ref_id, 'id': my_id,
                                        'differing_top_level_keys': keys})

    def _vmstat(self, tag):
        try:
            L.MAIN_DIR = CALIB_DIR
            before = set(os.listdir(os.path.join(CALIB_DIR, 'vmstat'))) if os.path.isdir(
                os.path.join(CALIB_DIR, 'vmstat')) else set()
            L.vmstat(SimpleNamespace(spec=None, tag=tag))
            after = set(os.listdir(os.path.join(CALIB_DIR, 'vmstat')))
            self.vmstat_files.extend(sorted(after - before))
        except Exception as error:  # noqa: BLE001
            self.errors.append(f'vmstat: {error}')

    def run(self):
        while not self.stop_event.is_set():
            try:
                if self.t_launch is None and os.path.exists(os.path.join(self.eval_dir, 'launch.json')):
                    self.t_launch = os.path.getmtime(os.path.join(self.eval_dir, 'launch.json'))
                for name in self.ref:
                    self._check_file(name)
                if self.mismatches:
                    self._fire(f'prefix differs: {self.mismatches[:3]}')
                p = self.projection
                if p is not None and p.get('evaluated') is None and p['at_cycle'] in self.t_obs and 1 in self.t_obs \
                        and self.t_launch is not None:
                    mean = (self.t_obs[p['at_cycle']] - self.t_obs[1]) / (p['at_cycle'] - 1)
                    proj = (self.t_obs[1] - self.t_launch) + mean * (self.cap - 1)
                    p.update({'evaluated': _utc(), 'mean_s_per_cycle_2_to_20': mean,
                              't1_minus_launch_s': self.t_obs[1] - self.t_launch, 'projected_s': proj,
                              'projected_h': proj / 3600.0, 'fires': proj > p['limit_s']})
                    _log(f"projection ({self.inst}): mean {mean:.1f} s/cycle, projected {proj / 3600.0:.2f} h "
                         f"(limit {p['limit_s'] / 3600.0:.1f} h)")
                    if proj > p['limit_s']:
                        self._fire(f"projection {proj / 3600.0:.2f} h > {p['limit_s'] / 3600.0:.1f} h")
                if time.time() - self.last_vmstat >= VMSTAT_EVERY_S:
                    self.last_vmstat = time.time()
                    self._vmstat(f'{self.inst}_hourly')
            except Exception as error:  # noqa: BLE001 -- recorded; the monitor keeps going
                self.errors.append(f'{type(error).__name__}: {error}')
            self.stop_event.wait(POLL_S)

    def summary(self):
        return {'prefix': self.prefix, 'compared_live': dict(self.compared),
                'reference_counts': {n: len(v) for n, v in self.ref.items()},
                'mismatches': self.mismatches[:50], 'n_mismatches': len(self.mismatches),
                'stop_fired': self.stop_fired, 'projection': self.projection,
                't_launch_epoch': self.t_launch,
                't_obs_capture_line_epoch': {str(k): v for k, v in sorted(self.t_obs.items())},
                'vmstat_files': self.vmstat_files, 'errors': self.errors}


def post_run_prefix_gate(eval_dir, ref_dir, prefix, keys):
    """After the run: every prefix item of the four files (the three live files again, in full, and
    per_cycle_record.jsonl lines 1..P)."""
    out = {'prefix': prefix, 'excluded_keys': sorted(keys), 'files': {}}
    held = True
    for name in ('per_cycle_record.jsonl', H.S56_CYCLE_CAPTURE_FILE, 'interface_duals_per_cycle.jsonl',
                 H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE):
        a = _jsonl_lines(os.path.join(ref_dir, name))
        b = _jsonl_lines(os.path.join(eval_dir, name)) if os.path.exists(os.path.join(eval_dir, name)) else []
        if name == H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE:
            a = [x for x in a if x.get('round') is not None and x['round'] <= prefix]
            b = [x for x in b if x.get('round') is not None and x['round'] <= prefix]
        else:
            a = [x for x in a if not x.get('header') and x.get('cycle') is not None and x['cycle'] <= prefix]
            b = [x for x in b if not x.get('header') and x.get('cycle') is not None and x['cycle'] <= prefix]
        diffs = []
        for i, (x, y) in enumerate(zip(a, b)):
            if canon(x, keys) != canon(y, keys):
                diffs.append({'position': i, 'id': x.get('cycle', x.get('round')),
                              'differing_top_level_keys': _differing_keys(x, y, keys)})
        keys_only = sorted({k for x in a for k in x} ^ {k for y in b for k in y})
        ok = len(a) == len(b) and not diffs and len(a) > 0
        held = held and ok
        out['files'][name] = {'n_reference': len(a), 'n_run': len(b), 'n_differ': len(diffs),
                              'first_differences': diffs[:10], 'keys_in_one_file_only': keys_only, 'equal': ok}
    out['held'] = held
    return out


# ======================================================================================================================
#  pre-launch assertions (zero solves)
# ======================================================================================================================
def prelaunch(spec, inst):
    """Every capture path the spec requires, the references and their hashes. Returns (checks, refusals)."""
    import p515_s56_step4_main_evaluate as E   # imports only (zero-solve module)
    import p515_s56_step4_postdiag as PD       # imports only (zero-solve module)
    refusals = []
    checks = {}
    try:
        checks['main_launcher_capture_path_checklist'] = L.capture_path_checklist()
    except Exception as error:  # noqa: BLE001
        refusals.append(f'main launcher capture-path checklist: {error}')
    try:
        tail_on = bool((spec['i_arms_instances_cap']['configuration'].get('convergence_depth_tail') or {}).get('enabled'))
        checks['certification_stop_path'] = H.assert_s56_certification_stop_path(
            {'tail_enabled_for_this_run': tail_on}, int(spec['i_arms_instances_cap']['required_consecutive_cycles']))
    except Exception as error:  # noqa: BLE001
        refusals.append(f'certification-stop path: {error}')
    try:
        checks['interface_v_multiplier_path'] = H.assert_s56_interface_v_multiplier_path()
    except Exception as error:  # noqa: BLE001
        refusals.append(f'interface-V multiplier path: {error}')
    # post hoc quantities (spec 'captures'): their sources in the per-cycle capture and the functions that compute them
    fields = set(H.S56_CYCLE_CAPTURE_LINE_FIELDS_BY_ARM[H.S56_CAPTURE_ARM_COUPLED])
    post_hoc = {
        'e_b_source_q_blocks': 'q_blocks' in fields,
        'o1_price_source_settlement_parts': 'q_blocks' in fields,
        'o2_dual_split_sources': {'primal_expectation', 'primal_deviation', 'dual_expectation_v'} <= fields,
        'o2_dual_split_function': callable(getattr(PD, 'channel_parts', None)),
        'v_base_function': callable(getattr(PD, 'v_base_dso', None)) and callable(getattr(E, 'v_base', None)),
        'mismatch_dP_dQ_dV_function': callable(getattr(E, 'capture_scenario_mismatch', None)),
        'settling_bar_function': callable(getattr(E, 'settling_bar', None)),
        'tail_depth_rule': callable(getattr(L, 'tail_depth_rounds', None)),
        'interface_dual_deviation_fields': 'lambda_dev_' in open('interface_dual_capture.py', encoding='utf-8').read(),
        'terminal_step_to_threshold': "'rule_ten': {" in open('p515_s44_campaign_harness.py', encoding='utf-8').read(),
        'peak_rss_record': "'peak_rss'" in open('p515_s44_campaign_harness.py', encoding='utf-8').read(),
        'cycle_wall_s_per_cycle': 'cycle_wall_s' in H.PER_CYCLE_RECORD_FIELDS,
        'probabilities_terminal_file': H.MULTISCENARIO_TERMINAL_FILE == 'multiscenario_terminal.json',
    }
    checks['post_hoc_capture_paths'] = post_hoc
    bad = sorted(k for k, v in post_hoc.items() if v is not True)
    if bad:
        refusals.append(f'post hoc capture paths missing: {bad}')
    # references (prefix gate) and cited artifacts
    ref = spec['prefix_gate']['reference'][inst]
    ref_dir = reference_eval_dir(spec, inst)
    for name, sha in ref['files_sha256'].items():
        p = os.path.join(ref_dir, name)
        if not os.path.exists(p) or _sha(p) != sha:
            refusals.append(f'prefix reference {inst}/{name} missing or hash differs')
    for key, item in spec['cites'].items():
        if isinstance(item, dict) and 'sha256' in item and _sha(item['path']) != item['sha256']:
            refusals.append(f'cite {key}: {item["path"]} sha256 differs from the spec')
    if _sha('p515_s44_campaign_harness.py') != spec['harness']['sha256']:
        refusals.append('harness sha256 differs from the spec')
    nl = _load(spec['nl_identity_check']['compare']['path'])
    checks['nl_identity_check_holds'] = nl.get('holds') is True
    if nl.get('holds') is not True:
        refusals.append('the zero-solve NL identity / option check does not hold')
    return checks, refusals


def check(args):
    spec, sha, rel = load_spec(args.spec)
    out = {}
    for inst in INSTANCES:
        checks, refusals = prelaunch(spec, inst)
        out[inst] = {'refusals': refusals, 'n_checks': len(checks)}
    _log(f'--check {rel}: ' + json.dumps(out) + f'; guard {GUARD.counts}')
    return 0 if all(not v['refusals'] for v in out.values()) and not GUARD.verify(0) else 2


# ======================================================================================================================
#  one campaign
# ======================================================================================================================
def run(args):
    t0 = time.time()
    spec, spec_sha, spec_rel = load_spec(args.spec)
    inst = args.instance
    cap = int(spec['cap'][PHASE][inst])
    prefix = int(spec['prefix_gate']['prefix_cycles'][inst])
    keys = tuple(spec['prefix_gate']['excluded_keys_declared_in_advance'])
    tag = f"{spec['roots']['root_tag']}_{PHASE}_{inst}"
    root = spec['roots']['campaign_roots'][PHASE][inst]
    rec = {'schema': 'p515_s56_step4_calib_launcher_v1', 'phase': PHASE, 'instance': inst, 'arm': ARM, 'cap': cap,
           'prefix_cycles': prefix, 'utc_start': _utc(), 'pid': os.getpid(), 'spec': {'path': spec_rel,
                                                                                    'sha256': spec_sha},
           'script': SCRIPT, 'script_sha256': _sha(SCRIPT), 'git_head': _git('rev-parse', 'HEAD').strip(),
           'interpreter': sys.executable, 'nlp_solver_path': os.environ.get('NLP_SOLVER_PATH'), 'refusals': [],
           'error': None}
    os.makedirs(CALIB_DIR, exist_ok=True)
    rec_path = os.path.join(CALIB_DIR, f'{tag}_launcher.json')
    if os.path.exists(rec_path):
        _log(f'REFUSED: {rec_path} exists (write-once)')
        return 1
    ref = rec['refusals']
    if root != os.path.join(REPO, CALIB_DIR, f"campaign_s56s4calib_{spec['roots']['root_tag']}_{PHASE}_{inst}"):
        ref.append(f'spec root {root!r} is not the absolute version-tagged root under {CALIB_DIR}')
    if inst == 'B':
        a_rec = os.path.join(CALIB_DIR, f"{spec['roots']['root_tag']}_{PHASE}_A_launcher.json")
        if not os.path.exists(a_rec):
            ref.append('instance B runs only after instance A (no A launcher record)')
        else:
            ar = _load(a_rec)
            if not ((ar.get('result_summary') or {}).get('all_exit_0') is True
                    and (ar.get('prefix_gate_post_run') or {}).get('held') is True
                    and (ar.get('monitor') or {}).get('stop_fired') is None):
                ref.append('instance B runs only after instance A exited 0 with its prefix gate held and no stop')
    if os.environ.get('NLP_SOLVER_PATH') != SOLVER_PATH:
        ref.append(f"NLP_SOLVER_PATH is {os.environ.get('NLP_SOLVER_PATH')!r}")
    for rel in (SCRIPT, spec_rel, 'p515_s44_campaign_harness.py', 'p515_s56_step4_main.py', 'p515_s56_step4_smoke.py',
                DERIVE_SCRIPT, CHECKS_SCRIPT):
        if not SM.committed_clean(rel):
            ref.append(f'{rel} not committed clean')
    others = other_copies_of_this_launcher()
    if others:
        ref.append(f'another copy of this launcher is running: {others}')
    if os.path.exists(root):
        ref.append(f'campaign root exists: {root}')
    try:
        checks, prefs = prelaunch(spec, inst)
        rec['prelaunch_checks'] = checks
        ref.extend(prefs)
    except Exception as error:  # noqa: BLE001
        ref.append(f'prelaunch assertions raised: {error}')
    try:
        options = calib_options(spec)
    except Exception as error:  # noqa: BLE001
        ref.append(str(error))
        options = None
    try:
        seed_src, seed_name = L.smoke_init_record(inst, ARM)
    except Exception as error:  # noqa: BLE001
        ref.append(f'smoke initialisation record {inst}/{ARM}: {error}')
        seed_src = seed_name = None
    if ref:
        _log(f'REFUSED before launch (zero solves): {ref}')
        rec['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
        write_json(rec_path, rec)
        return 3
    i_block = spec['i_arms_instances_cap']
    cfg = deepcopy(i_block['configuration'])
    cfg['name'] = (f"S56 step 4 calibration re-run: instance {inst} ({', '.join(i_block['instances'][inst]['blocks'])}), "
                   f"arm V-1, certification stop disabled, cap {cap}; x0, alpha 0")
    cfg['derived_instance'] = H.validate_derived_instance(i_block['instances'][inst]['derived_instance'])
    failures = H.check_campaign_preconditions(root, extra_clean_files=(SCRIPT, spec_rel, 'p515_s56_step4_main.py',
                                                                       'p515_s56_step4_smoke.py', DERIVE_SCRIPT,
                                                                       CHECKS_SCRIPT))
    if failures:
        rec['refusals'] = failures
        _log(f'REFUSED by the harness preconditions (zero solves): {failures}')
        rec['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
        write_json(rec_path, rec)
        return 3
    with open(os.path.join(CALIB_DIR, f'{tag}_launcher_pid.txt'), 'x') as handle:
        handle.write(f'{os.getpid()}\n')
    L.MAIN_DIR = CALIB_DIR
    L.vmstat(SimpleNamespace(spec=None, tag=f'{inst}_before'))
    spec_path, csha, cspec = H.freeze_campaign_spec(
        root, f"s56s4calib_{spec['roots']['root_tag']}_{PHASE}_{inst}", [(label_of(), dict(SM.X0), options)], cfg,
        cap, 1, spec['authority'], required_consecutive_cycles=i_block['required_consecutive_cycles'],
        extra={'calib_spec': {'path': spec_rel, 'sha256': spec_sha}, 'phase': PHASE, 'instance': inst, 'arm': ARM,
               'campaign_script': SCRIPT, 'campaign_script_sha256': _sha(SCRIPT),
               'objective_convention': spec['objective_convention']})
    entry = cspec['candidates'][0]
    rec['campaign'] = {'root': root, 'spec': {'path': os.path.relpath(spec_path, REPO), 'sha256': csha},
                       'entry': {'label': entry['label'], 'eval_key': entry['eval_key'], 'eval_dir': entry['eval_dir']}}
    ref_spec_eval_key = spec['prefix_gate']['reference'][inst]['eval_key']
    rec['eval_key_equals_step4_main_v1'] = entry['eval_key'] == ref_spec_eval_key
    idir = os.path.join(root, H.INIT_IDENTITY_DIR_NAME)
    os.makedirs(idir)
    dst = os.path.join(idir, seed_name)
    shutil.copyfile(seed_src, dst)
    rec['init_identity_seeded'] = {'from_smoke_arm': ARM, 'file': seed_name, 'sha256_src': _sha(seed_src),
                                   'sha256_dst': _sha(dst)}
    ref_dir = reference_eval_dir(spec, inst)
    ref_items = reference_items(ref_dir, prefix, keys)
    rec['reference_prefix_items'] = {n: len(v) for n, v in ref_items.items()}
    projection = None
    pj = spec['stops']['projection']
    if inst in pj['instances']:
        projection = {'at_cycle': int(pj['at_cycle']), 'limit_s': float(pj['limit_h']) * 3600.0, 'evaluated': None}
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    monitor = Monitor(inst, eval_dir, ref_dir, ref_items, prefix, cap, keys, projection, rec)
    lock = H.acquire_campaign_lock(cspec['campaign_id'], csha)
    _log(f'{inst}: lock {lock}; campaign spec {os.path.relpath(spec_path, REPO)}; entry {entry["label"]} eval key '
         f'{entry["eval_key"][:16]} (= step 4 main V-1: {rec["eval_key_equals_step4_main_v1"]}); cap {cap}; prefix '
         f'{prefix}; reference items {rec["reference_prefix_items"]}')
    records = None
    monitor.start()
    try:
        ctx = H.CampaignContext(root, spec_path, csha, cspec, log=_log)
        records = H.evaluate([label_of()], ctx)
        rec['batch_info'] = getattr(H.evaluate, 'last_batch_info', None)
    except BaseException as error:  # noqa: BLE001 -- a harness fault or a declared stop (SIGINT), recorded
        rec['error'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        _log(f'HARNESS FAULT / STOP: {rec["error"]["error"]}')
    finally:
        rec['lock_released'] = H.release_campaign_lock(expected_pid=os.getpid())
        time.sleep(POLL_S)
        monitor.stop_event.set()
        monitor.join(timeout=60)
    rec['monitor'] = monitor.summary()
    r = (records or [None])[0] or {}
    rec['results'] = [{'arm': ARM, 'status': r.get('status'), 'cycles_run': r.get('cycles_run'),
                       'exit_code': (r.get('parent_view') or {}).get('exit_code'),
                       'wall_s_parent_view': (r.get('parent_view') or {}).get('wall_s'),
                       'solve_profile': r.get('solve_profile'), 'peak_rss': r.get('peak_rss'),
                       'eval_dir': r.get('eval_dir')}]
    rec['result_summary'] = {'all_exit_0': bool(records) and rec['results'][0]['exit_code'] == 0,
                             'reached_cap': r.get('cycles_run') == cap}
    try:
        rec['prefix_gate_post_run'] = post_run_prefix_gate(eval_dir, ref_dir, prefix, keys)
    except Exception as error:  # noqa: BLE001
        rec['prefix_gate_post_run'] = {'held': False, 'error': f'{type(error).__name__}: {error}',
                                       'traceback': traceback.format_exc()}
    try:
        L.vmstat(SimpleNamespace(spec=None, tag=f'{inst}_after'))
    except Exception as error:  # noqa: BLE001
        rec['vmstat_after_error'] = str(error)
    rec['guard'] = {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0)}
    rec['wall_s'] = time.time() - t0
    rec['utc_end'] = _utc()
    write_json(rec_path, rec)
    ok = (rec['error'] is None and rec['result_summary']['all_exit_0'] and rec['result_summary']['reached_cap']
          and rec['prefix_gate_post_run'].get('held') is True and monitor.stop_fired is None
          and not rec['guard']['verify_failures'])
    _log(f"{inst}: exit {rec['results'][0]['exit_code']} cycles {r.get('cycles_run')} status {r.get('status')}; "
         f"prefix post-run held {rec['prefix_gate_post_run'].get('held')}; live mismatches "
         f"{rec['monitor']['n_mismatches']}; stop {monitor.stop_fired}; wall {rec['wall_s']:.0f}s -> {rec_path}")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--spec', required=True)
    ap.add_argument('--instance', choices=INSTANCES)
    a = ap.parse_args()
    if a.check:
        return check(a)
    if a.run and a.instance:
        return run(a)
    ap.error('--check --spec S | --run --spec S --instance A|B')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
