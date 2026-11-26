"""P5.15 Addendum 65, Planner task W159 -- THE CLOSING READS BEFORE THE STEP 6 TABLES FREEZE.
ZERO SOLVES, NO MODEL LOADS. A NEW FILE: no harness, scorer, builder or production module is edited.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 65 ("Closing reads before the freeze (zero-solve)": (a) the certifying
spec of every cell the tables use; (b) T6 shows both uncoordinated arms; (c) the banner, threading and version reads,
and the HSL_MA97 bit-compatibility statement checked against the author's coinhsl-2023.11.17 archive).

WHAT IT READS (nothing is written anywhere except the new output directory below)
  (a) T1-T10 of data/SRP1/Results/P515S53/w157_step6_tables_a64/w157_step6_tables.json (b913ea94; checked against its
      committed manifest). For every evaluation a table uses, the run record is located among the COMMITTED campaign
      specs (the spec whose campaign root holds evals/<eval_dir>/evaluation_record.json, tracked) and the AS-RUN
      configuration is read from the evaluation record itself: case file / ESS params sha256 seen in the child, the
      Anderson-acceleration dict in force, the convergence-depth tail in force, the ageing model as read back from the
      built models (k, phi_cal, the SoH floor row, the SoH point), the model variant, the flexibility-price multiplier,
      the settling rule module / version, the certification status and cycle, the launch time. Each is classified
      against the CURRENT CONFIGURATION, defined operationally as `inputs_in_force_now` of the v6 stage spec 96c23404.
      The 0.50-era records behind T8's "eps_AE (0.50, superseded)" column are read from the inputs that
      data/SRP1/Results/P515S46/ageing_mechanism/ageing_mechanism.json (b5eca2a2) lists. T6's uncoordinated arms are
      benchmark runs (spec v5 bca69f97), not settling certificates; what their records state is reported.
  (b) T6 in the W157 JSON and Markdown against report_v3.json (spec v5 bca69f97; commit 8d42dfb8) and the per-arm run
      records: which fields of each NRF arm the table carries and which it lacks, with the source of each missing one.
  (c) 1 threading -- the launch commands of the v6 / ext v3 / A64 specs (environment assignments), the shell profile
        and Claude shell-snapshot files, .env (ONLY whether thread variables are named; nothing of its content is
        recorded), launchctl's environment, `git grep` of tracked Python for thread variables, this process's
        environment, and the thread caps every table cell's child recorded (launch.json, evaluation record, the
        first line of child_stdout.log);
      2 linear-solver banners -- one v6 TSO log, one v6 DSO log and one v6 ESSO log of cell d_4a82a64a (and, as a
        second sample at the end of the window, the TSO and ESSO cycle-1 logs of e_soh050): path, sha256 now, the
        launch-manifest entry, line number and exact banner; plus a tally of every banner in every log the
        d_4a82a64a launch manifest lists (each file's sha256 checked against the manifest);
      3 versions now -- sys.version, pyomo.version.version, sw_vers, conda-meta/history (every entry), the newest
        file mtime in the environment, `softwareupdate --history`, /usr/local/bin/ipopt (mtime, sha256 against the v6
        spec pin, `--version`) and the libraries it links; compared with the start of the first v6 cell
        (run_d_4a82a64a) and the end of the last launch (e_soh050);
      4 HSL_MA97 -- the two CoinHSL 2023.11.17 archives in ~/Downloads are LISTED and searched IN MEMORY (tarfile;
        nothing is extracted to disk); the ipopt3.14.18hsl5.5.0 package archive is read in memory to compare its
        libcoinhsl with the installed one; the installed library's link map (otool -L) and symbols (nm) are read.
      `ipopt --version` is a subprocess call outside Pyomo: it prints the version and exits, it is not a solve and the
      Pyomo-level guard neither intercepts nor counts it.

GUARDS. `pickle.load` / `pickle.loads` are blocked BEFORE any project import and verified at 0; an armed
`SolveProfileGuard(permitted=())` is installed before any other project import and verified at exactly 0 at the end.
The only project modules imported are p513_solve_profile_guard and gate_result_io (the shared writer).

MODE (repo root, canonical interpreter; attached, both streams captured; outputs opened 'x', never overwritten):
    mkdir -p data/SRP1/Results/P515S53/w159_closing_reads && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w159_closing_reads.py \\
        > data/SRP1/Results/P515S53/w159_closing_reads/launch.log 2>&1
Exit: 0 = written, every integrity check holds and the guards are at 0; 3 = written, an integrity check failed
(listed); 1 = harness fault or precondition (nothing written). FINDINGS (an old-configuration figure, a missing T6
field, an environment change) never change the exit code: they are the result.
"""
import glob
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W159: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W159: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W159 closing reads (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import pyomo.version  # noqa: E402 -- read for (c)3 (already imported by the guard's pyomo imports)

TAG = 'W159'
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(S53, 'w159_closing_reads')
OUT_JSON = 'w159_closing_reads.json'
OUT_MD = 'w159_closing_reads.md'
OUT_MAN = 'manifest_sha256.json'
SCRIPT_REL = os.path.basename(__file__)

W157_DIR = os.path.join(S53, 'w157_step6_tables_a64')
W157_JSON = os.path.join(W157_DIR, 'w157_step6_tables.json')
W157_MD = os.path.join(W157_DIR, 'w157_step6_tables.md')
W157_MAN = os.path.join(W157_DIR, 'manifest_sha256.json')
V6_SPEC = os.path.join(S53, 'w142_resettle_v6', 'frozen_s53_resettle_spec_v6_96c23404.json')
EXT_SPEC = os.path.join(S53, 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json')
A64_SPEC = os.path.join(S53, 'w155_a64_cells', 'frozen_s53_a64_cells_spec_v1_44a2dce8.json')
V4_SPEC = os.path.join(S53, 'w137_resettle_v4', 'frozen_s53_resettle_spec_v4_e11fbc89.json')
V5_SPEC = os.path.join(S53, 'w139_resettle_v5', 'frozen_s53_resettle_spec_v5_ab32ffc9.json')
BENCH_SPEC = os.path.join(S53, 'w116_benchmark_nrf', 'frozen_s53_benchmark_spec_v5_bca69f97.json')
BENCH_REPORT = os.path.join(S53, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json')
BENCH_REPORT_MAN = os.path.join(S53, 'w116_benchmark_nrf', 'report_v3', 'manifest_sha256.json')
W145_JSON = os.path.join(S53, 'w145_banded_fit', 'w145_banded_fit.json')
W153D_JSON = os.path.join(S53, 'w153_step5_rows', 'w153_discount_row.json')
W149_JSON = os.path.join(S53, 'w149_h_f9eae48f_prices', 'w149_interface_price_read.json')
W127_MAN = os.path.join(S53, 'w127_stall_constraints', 'manifest_sha256.json')
AGEING_MECH = os.path.join('data', 'SRP1', 'Results', 'P515S46', 'ageing_mechanism', 'ageing_mechanism.json')
ESS_PARAMS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
HARNESS = 'p515_s44_campaign_harness.py'

FIRST_V6 = {'cell': 'd_4a82a64a',
            'launch_json': os.path.join(S53, 'w142_resettle_v6', 'campaign_s53_w142_resettle_v6_d_4a82a64a', 'evals',
                                        '14b00a04ffbcfd33_d_4a82a64a', 'launch.json'),
            'manifest': os.path.join(S53, 'w142_resettle_v6', 'run_d_4a82a64a_launch_manifest_sha256.json'),
            'p56a': os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals',
                                 'p515s44_s53_w142_resettle_v6_d_4a82a64a_14b00a04ffbcfd33_run', 'logs')}
LAST_LAUNCH = {'cell': 'e_soh050',
               'campaign_root': os.path.join(S53, 'w155_a64_cells', 'campaign_s53_w155_a64_r2_e_soh050'),
               'launch_json': os.path.join(S53, 'w155_a64_cells', 'campaign_s53_w155_a64_r2_e_soh050', 'evals',
                                           '3f01b9eaaab13c82_e_soh050', 'launch.json'),
               'campaign_results': os.path.join(S53, 'w155_a64_cells', 'campaign_s53_w155_a64_r2_e_soh050',
                                                'campaign_results.json'),
               'launch_log': os.path.join(S53, 'w155_a64_cells', 'run_e_soh050_launch.log'),
               'manifest': os.path.join(S53, 'w155_a64_cells', 'run_e_soh050_launch_manifest_sha256.json'),
               'p56a': os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals',
                                    'p515s44_s53_w155_a64_r2_e_soh050_3f01b9eaaab13c82_run', 'logs')}

THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_THREAD_LIMIT')
HOME = os.path.expanduser('~')
SHELL_FILES = [os.path.join(HOME, n) for n in ('.zshrc', '.zprofile', '.zshenv', '.zlogin', '.zlogout', '.bash_profile',
                                               '.bashrc', '.profile')] + \
              ['/etc/zshenv', '/etc/zprofile', '/etc/zshrc', '/etc/zlogin', '/etc/profile', '/etc/bashrc']
SNAPSHOT_DIR = os.path.join(HOME, '.claude', 'shell-snapshots')
CONDA_ENV = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311'
CONDA_HISTORY = os.path.join(CONDA_ENV, 'conda-meta', 'history')
IPOPT = '/usr/local/bin/ipopt'
IPOPT_LIBS = ['/usr/local/lib/libipopt.3.dylib', '/usr/local/lib/libcoinhsl.2.dylib',
              '/usr/local/lib/libcoinmumps.3.dylib', '/usr/local/lib/libipoptamplinterface.3.dylib',
              '/usr/local/lib/libcoinasl.2.dylib']
HSL_SRC_ARCHIVE_CANDIDATES = [os.path.join(HOME, 'coinhsl-2023.11.17.tar.gz'),
                              os.path.join(HOME, 'Downloads', 'coinhsl-2023.11.17.tar.gz')]
HSL_BIN_ARCHIVE = os.path.join(HOME, 'Downloads', 'CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5.tar.gz')
IPOPT_PKG_ARCHIVE = os.path.join(HOME, 'Downloads', 'ipopt3.14.18hsl5.5.0arm64.tar.gz')
THIRDPARTY_HSL = os.path.join(HOME, 'ThirdParty-HSL')
STATEMENT_PATTERNS = [r'bit[- ]?compatib', r'bit[- ]for[- ]bit', r'reproducib', r'number of threads',
                      r'independent of the number', r'regardless of the number', r'determinis']
BANNER_RE = re.compile(r'^This is Ipopt version (\S+), running with linear solver (\S+)\.\s*$')

T_PREFIX = {'T1': 'claims', 'T2': 'cells', 'T3': 'break-even fit', 'T4': 'year ladder', 'T5': 'Phase B',
            'T6': 'benchmark', 'T7': 'discount', 'T8': 'ageing arms', 'T9': 'dead zone', 'T10': 'A64 rows'}

# campaign-root directory prefix -> the stage spec the campaign ran under (checked against the campaign spec text
# where that text names it)
STAGE_BY_DIR = [
    (os.path.join(S53, 'w142_resettle_v6') + os.sep, 'v6', 'frozen_s53_resettle_spec_v6_96c23404'),
    (os.path.join(S53, 'w142_resettle_ext_v6') + os.sep, 'ext v3 (rule v6)', 'frozen_s53_resettle_ext_spec_v3_84775dc4'),
    (os.path.join(S53, 'w155_a64_cells') + os.sep, 'A64 v1 (rule v6)', 'frozen_s53_a64_cells_spec_v1_44a2dce8'),
    (os.path.join(S53, 'w137_resettle_v4') + os.sep, 'v4', 'frozen_s53_resettle_spec_v4_e11fbc89'),
    (os.path.join(S53, 'w139_resettle_v5') + os.sep, 'v5', 'frozen_s53_resettle_spec_v5_ab32ffc9'),
    (os.path.join(S53, 'w118_resettle', 'campaign_s53_w118_resettle_r2_'), 'W118 r2 (rule v2)',
     'frozen_s53_resettle_spec_v2_fc791891'),
    (os.path.join(S53, 'w118_resettle', 'campaign_s53_w118_resettle_'), 'W118 r1', 'frozen_s53_resettle_spec_v1_d902a85c'),
    (os.path.join(S53, 'w101_srp1_continuation') + os.sep, 'W101 continuation (settling_criterion v1 rule)',
     'frozen_s53_spec_v39_8a612429'),
    (os.path.join('data', 'SRP1', 'Results', 'P515S46') + os.sep, 'S46 ageing batch (Addenda 28-29)', None),
    (os.path.join('data', 'SRP1', 'Results', 'P515S45') + os.sep, 'S45 A1a (Addendum 26-27 era)', None),
]


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} [{TAG}] {msg}", flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _run(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {'cmd': cmd, 'rc': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except Exception as exc:  # noqa: BLE001
        return {'cmd': cmd, 'rc': None, 'stdout': '', 'stderr': f'{type(exc).__name__}: {exc}'}


def _git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=REPO).stdout


def _tracked(path):
    return subprocess.run(['git', 'ls-files', '--error-unmatch', path], capture_output=True, cwd=REPO).returncode == 0


def _committed_clean(path):
    return _tracked(path) and _git('status', '--porcelain', '--', path).strip() == ''


def _iso_from_epoch(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def _epoch_from_iso(s):
    return datetime.fromisoformat(s).timestamp()


def _mtime(path):
    st = os.stat(path)
    return {'mtime_epoch': st.st_mtime, 'mtime_utc': _iso_from_epoch(st.st_mtime)}


INPUTS = {}
CHECKS = {}


def _input(path, manifest=None):
    """Record a committed input's sha256; with `manifest`, check it against that committed manifest's entry."""
    rec = {'sha256': _sha(path), 'committed_clean': _committed_clean(path)}
    if manifest is not None:
        man = _load(manifest)
        rec['manifest'] = manifest
        rec['manifest_committed_clean'] = _committed_clean(manifest)
        rec['manifest_entry'] = man.get(path)
        rec['matches_manifest'] = man.get(path) == rec['sha256']
    INPUTS[path] = rec
    return rec


# ======================================================================================================================
#  (a) certifying spec and as-run configuration of every evaluation a table uses
# ======================================================================================================================
def current_configuration():
    v6 = _load(V6_SPEC)
    now = v6['inputs_in_force_now']
    cfg = now['configuration_now']
    ess = _load(ESS_PARAMS)
    ess_sha = _sha(ESS_PARAMS)
    ab = cfg['ess_ageing_baseline']
    k = ab['calibration']['cycles_n'] * ab['calibration']['reference_dod_d'] / (-math.log(ab['calibration']['eol_retention_r']))
    return {
        'source': f'{V6_SPEC} inputs_in_force_now (stage spec v6 96c23404)',
        'case_file_sha256': now['case_file']['sha256'],
        'cost_file_sha256': now['cost_file']['sha256'],
        'ess_params_sha256': now['ess_params_file']['sha256'],
        'ess_params_file_now_sha256': ess_sha,
        'ess_params_file_now_equals_pin': ess_sha == now['ess_params_file']['sha256'],
        'anderson_acceleration': cfg['case_file_anderson_acceleration'],
        'tail': cfg['convergence_depth_tail'],
        'ageing_baseline': ab,
        'ageing_baseline_label': cfg['ess_ageing_baseline_label'],
        'k_expected': k,
        'energy_to_power_factor_bounds_h': [ess.get('min_energy_to_power_factor'), ess.get('max_energy_to_power_factor')],
        'minimum_soh_in_file': ess.get('minimum_soh'),
    }


def index_campaign_specs():
    """eval_key -> [(campaign spec path, campaign root, eval_dir, record path, record tracked)] over every COMMITTED
    campaign spec under data/SRP1/Results."""
    idx = {}
    files = [f for f in _git('ls-files', 'data/SRP1/Results').split('\n') if re.search(r'/campaign_spec_[^/]*\.json$', f)]
    for f in files:
        try:
            sp = _load(f)
        except Exception:  # noqa: BLE001
            continue
        root = os.path.dirname(f)
        for c in sp.get('candidates', []) or []:
            ek = c.get('eval_key')
            if not ek:
                continue
            rec = os.path.join(root, 'evals', c.get('eval_dir', ''), 'evaluation_record.json')
            idx.setdefault(ek, []).append({'campaign_spec': f, 'campaign_root': root, 'eval_dir': c.get('eval_dir'),
                                           'record': rec, 'record_exists': os.path.exists(rec),
                                           'record_tracked': os.path.exists(rec) and _tracked(rec)})
    return idx, len(files)


def stage_of(path):
    for prefix, label, spec in STAGE_BY_DIR:
        if path.startswith(prefix):
            return label, spec
    return None, None


def _readback_ageing(rec):
    """The ageing model as read back from the built models (variant cells) or as verified pre-run (baseline cells)."""
    out = {}
    mv = rec.get('model_variant_readback_pre_run')
    if isinstance(mv, dict) and mv.get('per_node'):
        vals = {}
        for node, r in mv['per_node'].items():
            rb = r.get('readback', {})
            vals[node] = {'k': rb.get('k'), 'phi_cal_in_model': rb.get('phi_cal_in_model'),
                          'soh_floor_row_lower': rb.get('floor_row_lower'),
                          'available_energy_soh_point': rb.get('available_energy_soh_point')}
        out['source'] = 'model_variant_readback_pre_run (read back from the built models)'
        out['per_node'] = vals
        out['ageing_enabled'] = (mv.get('expected') or {}).get('ageing_enabled')
    ev = rec.get('ess_ageing_verified_pre_run')
    if isinstance(ev, dict) and ev.get('per_ess'):
        out['ess_ageing_verified_pre_run'] = {
            'loaded_minimum_soh': (ev.get('loaded') or {}).get('minimum_soh'),
            'loaded_calendar_retention_per_year': (ev.get('loaded') or {}).get('calendar_retention_per_year'),
            'loaded_calibration': (ev.get('loaded') or {}).get('calibration'),
            'per_ess_soh_min': sorted({e.get('soh_min') for e in ev['per_ess']}),
            'per_ess_phi_cal': sorted({e.get('phi_cal') for e in ev['per_ess']}),
            'per_ess_k': sorted({e.get('cl_eff') for e in ev['per_ess']}),
            'file_sha256': ev.get('file_sha256')}
    if not out:
        txt = json.dumps(rec)
        out['source'] = 'no structured ageing read-back in the record; values found by text search of the record'
        out['text_search'] = {kw: sorted(set(re.findall(rf'"{kw}": ([0-9.eE+-]+|true|false|null)', txt)))
                              for kw in ('floor_row_lower', 'minimum_soh', 'eol_retention_r', 'calendar_retention_per_year',
                                         'phi_cal_in_model', 'k')}
    return out


def _effective_ageing(ag):
    """(k set, phi set, floor set, soh point set, enabled) from the read-back."""
    ks, phis, floors, points = set(), set(), set(), set()
    if 'per_node' in ag:
        for v in ag['per_node'].values():
            ks.add(v['k']); phis.add(v['phi_cal_in_model']); floors.add(v['soh_floor_row_lower'])
            points.add(v['available_energy_soh_point'])
        enabled = ag.get('ageing_enabled')
    elif 'ess_ageing_verified_pre_run' in ag:
        e = ag['ess_ageing_verified_pre_run']
        ks, phis, floors = set(e['per_ess_k']), set(e['per_ess_phi_cal']), set(e['per_ess_soh_min'])
        points = {'end (baseline; no variant)'}
        enabled = True
    else:
        return None
    return {'k': sorted(x for x in ks if x is not None), 'phi_cal': sorted(x for x in phis if x is not None),
            'soh_floor': sorted(x for x in floors if x is not None), 'soh_point': sorted(points), 'enabled': enabled}


def _calibration_label(k_list, enabled):
    if enabled is False:
        return 'no ageing (ageing disabled)'
    if not k_list:
        return 'not read back'
    labels = []
    for k in k_list:
        r = math.exp(-10000 * 0.8 / k)
        labels.append(f'k {k:,.2f} (10,000 cycles at DoD 0.8 to retention r = {r:.2f})')
    return '; '.join(labels)


def _settling_rule(rec):
    for key in ('settling_resettle', 'settling_continuation'):
        s = rec.get(key)
        if isinstance(s, dict):
            r = s.get('settling_rule') or s.get('declaration', {}).get('settling_rule') or s
            return {'record_key': key, 'module': r.get('module'), 'class': r.get('class') or r.get('function'),
                    'version': r.get('version')}
    return {'record_key': None}


def _ess_params_at_head(campaign_spec_path):
    """The ESS params file as committed at the campaign spec's git_head (a git object read; the record itself does not
    carry the file's sha256, so the working-tree state at run time is NOT established by this read)."""
    if not campaign_spec_path or not os.path.exists(campaign_spec_path):
        return None
    head = _load(campaign_spec_path).get('git_head')
    if not head:
        return {'git_head': None}
    raw = subprocess.run(['git', 'show', f'{head}:{ESS_PARAMS}'], capture_output=True, cwd=REPO).stdout
    try:
        d = json.loads(raw)
    except Exception:  # noqa: BLE001
        return {'git_head': head, 'readable': False}
    ag = d.get('ageing') or {}
    return {'git_head': head, 'sha256_of_committed_blob': _sha_bytes(raw),
            'min_energy_to_power_factor': d.get('min_energy_to_power_factor'),
            'max_energy_to_power_factor': d.get('max_energy_to_power_factor'),
            'minimum_soh': ag.get('minimum_soh'),
            'calibration': {k: v for k, v in (ag.get('calibration') or {}).items() if not k.startswith('_')},
            'calendar_retention_per_year': ag.get('calendar_retention_per_year', 'absent from the file'),
            'note': 'file as committed at the campaign git_head; the run-time working tree is not recorded'}


def as_run(rec_path):
    rec = _load(rec_path)
    eval_dir = os.path.dirname(rec_path)
    launch = os.path.join(eval_dir, 'launch.json')
    stdout = os.path.join(eval_dir, 'child_stdout.log')
    lj = _load(launch) if os.path.exists(launch) else {}
    first = ''
    if os.path.exists(stdout):
        with open(stdout, errors='replace') as fh:
            first = fh.readline().rstrip('\n')
    caps_line = re.search(r"caps=(\{[^}]*\})", first)
    tail = rec.get('convergence_depth_tail_applied_in_child')
    cfg = rec.get('configuration') or {}
    ag = _readback_ageing(rec)
    return {
        'record': rec_path,
        'record_sha256': _sha(rec_path),
        'record_committed_clean': _committed_clean(rec_path),
        'campaign_spec_path': rec.get('campaign_spec_path'),
        'campaign_spec_sha256': rec.get('campaign_spec_sha256'),
        'candidate_canonical': rec.get('candidate_canonical'),
        'candidate_key': rec.get('candidate_key'),
        'status': rec.get('status'),
        'certification_cycle': rec.get('certification_cycle'),
        'cycles_run': rec.get('cycles_run'),
        'case_file_sha256_in_child': rec.get('case_file_sha256_in_child') or cfg.get('case_file_sha256'),
        'ess_params_sha256_in_child': rec.get('ess_params_sha256_in_child'),
        'ess_params_sha256_declared': (cfg.get('ess_params_file') or {}).get('sha256'),
        'anderson_acceleration_in_child': rec.get('anderson_acceleration_effective_in_child'),
        'tail_in_force': None if not isinstance(tail, dict) else {
            'enabled_in_force': tail.get('enabled_in_force'), 'after': tail.get('after'), 'ok': tail.get('ok')},
        'tail_record_present': isinstance(tail, dict),
        'ageing_readback': ag,
        'ageing_effective': _effective_ageing(ag),
        'model_variant': rec.get('model_variant'),
        'model_variant_label': rec.get('model_variant_label'),
        'flex_price_multiplier': rec.get('flex_price_multiplier'),
        'flex_price_label': rec.get('flex_price_label'),
        'overrides': cfg.get('overrides'),
        'settling_rule': _settling_rule(rec),
        'started_utc': lj.get('started_utc'),
        'wall_time_s': rec.get('wall_time_s'),
        'thread_caps_in_child_env_launch_json': lj.get('thread_caps_in_child_env'),
        'thread_caps_seen_by_child_record': rec.get('thread_caps_seen_by_child'),
        'child_stdout_first_line_caps': caps_line.group(1) if caps_line else None,
        'nlp_solver_path_in_child': rec.get('nlp_solver_path_in_child'),
    }


def classify(ar, cur):
    """Current configuration, current + a declared design variant, or old (with the differences)."""
    diffs, variants = [], []
    if ar['case_file_sha256_in_child'] != cur['case_file_sha256']:
        diffs.append(f"case file sha {str(ar['case_file_sha256_in_child'])[:8]} != {cur['case_file_sha256'][:8]}")
    ess_sha = ar['ess_params_sha256_in_child'] or ar['ess_params_sha256_declared']
    if ess_sha != cur['ess_params_sha256']:
        diffs.append(f"ESS params sha {'not recorded' if ess_sha is None else ess_sha[:8]} != {cur['ess_params_sha256'][:8]}")
    aa = ar['anderson_acceleration_in_child'] or {}
    if not (aa.get('enabled') is True and aa.get('reject_policy') == cur['anderson_acceleration']['reject_policy']):
        diffs.append(f'AA in child {aa}')
    t = ar['tail_in_force']
    if not (t and t.get('enabled_in_force') is True and (t.get('after') or {}).get('compl_inf_tol') == cur['tail']['compl_inf_tol']):
        diffs.append('tight tail not in force (no convergence_depth_tail record: production default is tail OFF)'
                     if t is None else f'tail {t}')
    eff = ar['ageing_effective']
    ab = cur['ageing_baseline']
    nodes = ((ar['candidate_canonical'] or {}).get('nodes') or {})
    no_storage = bool(nodes) and all(not any(v) for v in nodes.values())
    if eff is None and no_storage:
        variants.append('no storage in the candidate: the ageing model does not enter Q')
    elif eff is None:
        diffs.append('ageing model not read back in the record')
    else:
        k_ok = eff['k'] and all(abs(k - cur['k_expected']) < 1e-6 for k in eff['k'])
        phi_ok = eff['phi_cal'] == [ab['calendar_retention_per_year']]
        floor_ok = eff['soh_floor'] == [ab['minimum_soh']]
        point_ok = all(p.startswith('end') for p in eff['soh_point'])
        en_ok = eff['enabled'] is True
        mv = ar['model_variant']
        if not (k_ok and phi_ok and point_ok and en_ok):
            what = (f"calibration {_calibration_label(eff['k'], eff['enabled'])}, phi_cal {eff['phi_cal']}, "
                    f"SoH point {eff['soh_point']}")
            (variants if mv else diffs).append(f'model variant {mv}: {what}' if mv else f'ageing: {what}')
        if not floor_ok:
            msg = f"SoH floor row {eff['soh_floor']} (current {ab['minimum_soh']})"
            declared_soh = 'soh' in json.dumps(ar.get('model_variant_label') or '').lower() or \
                           'e_soh050' in (ar['record'] or '')
            (variants if declared_soh else diffs).append(msg + (' -- declared (the soh_min row)' if declared_soh else ''))
        elif mv and k_ok and phi_ok and point_ok and en_ok:
            variants.append(f'model variant {mv} (identical in effect to the baseline)')
    fm = ar['flex_price_multiplier']
    if fm not in (None, 1.0, 1):
        variants.append(f'flexibility price multiplier {fm} (declared)')
    if ar['overrides']:
        diffs.append(f"overrides {ar['overrides']}")
    real_variants = [v for v in variants if not v.startswith('no storage')]
    cls = 'OLD CONFIGURATION' if diffs else ('current + declared design variant' if real_variants else 'current')
    return cls, diffs, variants


def collect_uses(w157, w145, w153d, w149_rows):
    """table usage: list of (table, row label, cell name, eval_key or None)."""
    t = w157['tables']
    cells = t['cells']
    uses = []

    def cell_key(name):
        c = cells.get(name)
        return None if c is None else c.get('eval_key')

    for c in t['claims']:
        for side in ('ref_cell', 'other_cell'):
            uses.append(('T1', c['claim_id'], c[side], cell_key(c[side])))
    for name, c in cells.items():
        uses.append(('T2', name, name, c.get('eval_key')))
    for lb, p in w145['provenance'].items():
        nm = p.get('cell') or ('ref:bd504ecf' if 'bd504ecf' in (p.get('source') or '') else None)
        uses.append(('T3', lb, nm, p.get('eval_key') or cell_key(nm)))
    uses.append(('T3', 'Q(0) reference', 'ref:7aa017f0', cell_key('ref:7aa017f0')))
    for y, r in t['year_ladder']['per_year'].items():
        uses.append(('T4', y, f'W118 yl_y{y}', r['eval_key']))
    uses.append(('T4', 'Q181 in M = I + Q - Q181', 'ref:7aa017f0', cell_key('ref:7aa017f0')))
    for r in t['phase_b']:
        ek = r.get('eval_key') or cell_key(r['cell'])
        uses.append(('T5', r['cell'] + (' (SUPERSEDED row)' if r.get('superseded') else ''), r['cell'], ek))
    uses.append(('T5', 'x = 0 reference', 'ref:7aa017f0', cell_key('ref:7aa017f0')))
    uses.append(('T6', 'coordinated (settled x = 0)', 'ref:7aa017f0', cell_key('ref:7aa017f0')))
    inst = w153d['result']['instance']
    for role in ('x0', 'unit'):
        uses.append(('T7', f'discount instance {role}', inst[role]['name'], inst[role]['eval_key']))
    for r in t['ageing']['rows']:
        uses.append(('T8', f"{r['arm']} (value, value - I, floor year, eps_AE 0.70)", r['cell'], cell_key(r['cell'])))
    uses.append(('T8', 'Q(0) of value = Q(0) - Q(x)', 'ref:7aa017f0', cell_key('ref:7aa017f0')))
    for r in t['dead_zone']['cells']:
        uses.append(('T9', r['cell'], r['cell'], cell_key(r['cell'])))
    w149_map = {'f2_challenger': 'ref:e28de4ac', 'f2_incumbent': 'ref:5ca4f86c'}
    for r in w149_rows:
        nm = w149_map.get(r['cell'], r['cell'])
        uses.append(('T9', f"W149 row {r['cell']}", nm, cell_key(nm)))
    for r in t['a64']['rows']:
        for nm, c in (r.get('cells') or {}).items():
            uses.append(('T10', r['claim_id'], nm, c.get('eval_key')))
        for side in ('ref_cell', 'other_cell'):
            nm = r.get(side)
            if nm in ('bd504ecf', '7aa017f0'):
                uses.append(('T10', r['claim_id'], f'ref:{nm}', cell_key(f'ref:{nm}')))
    return uses


def part_a(w157, cur):
    w145 = _load(W145_JSON)
    w153d = _load(W153D_JSON)
    w149 = _load(W149_JSON)
    uses = collect_uses(w157, w145, w153d, w149['dead_zone_table'])
    idx, n_specs = index_campaign_specs()
    cells_t2 = w157['tables']['cells']
    by_key = {}
    for tbl, row, name, ek in uses:
        by_key.setdefault(ek, {'names': set(), 'rows': []})
        by_key[ek]['names'].add(str(name))
        by_key[ek]['rows'].append(f'{tbl}: {row}')
    evals, unresolved = {}, []
    for ek, u in by_key.items():
        if ek is None:
            unresolved.append({'eval_key': None, 'names': sorted(u['names']), 'rows': u['rows']})
            continue
        cands = idx.get(ek, [])
        ran = [c for c in cands if c['record_exists'] and c['record_tracked']]
        if len(ran) != 1:
            unresolved.append({'eval_key': ek, 'names': sorted(u['names']), 'candidates': cands})
            continue
        run = ran[0]
        ar = as_run(run['record'])
        stage, stage_spec = stage_of(run['campaign_root'] + os.sep)
        spec_text = open(run['campaign_spec']).read()
        named_specs = sorted(set(re.findall(r'frozen_s53_[a-z0-9_]+?_v\d+_[0-9a-f]{8}', spec_text)))
        cls, diffs, variants = classify(ar, cur)
        t2 = None
        for nm in u['names']:
            if nm in cells_t2:
                t2 = cells_t2[nm].get('certifying_spec')
        t2_stage = None
        if isinstance(t2, dict):
            if t2.get('series') == 'frozen_s53_resettle_ext_spec':
                t2_stage = 'ext v3 (rule v6)'
            elif t2.get('mode') == 'v6 from records':
                t2_stage = 'v5'  # the run is the v5 run; the certificate is v6 from records
            else:
                t2_stage = {4: 'v4', 5: 'v5', 6: 'v6'}.get(t2.get('version'))
        spec_sha_now = _sha(run['campaign_spec'])
        evals[ek] = {'names': sorted(u['names']), 'table_rows': u['rows'], 'run': run,
                     'other_specs_listing_this_eval_key': [c['campaign_spec'] for c in cands if c is not run],
                     'stage': stage, 'stage_spec_by_directory': stage_spec,
                     'stage_specs_named_in_campaign_spec': named_specs,
                     'note_named_specs': ('frozen specs the campaign-spec text names (report-only: the per-cell '
                                          'spec names the spec its cell table / rule was imported from)'),
                     'T2_certifying_spec': t2, 'T2_certifying_spec_implies_stage': t2_stage,
                     'stage_agrees_with_T2': None if t2_stage is None else t2_stage == stage,
                     'campaign_spec_sha256_now': spec_sha_now,
                     'campaign_spec_sha256_matches_record': spec_sha_now == ar['campaign_spec_sha256'],
                     'as_run': ar, 'classification': cls,
                     'differences_from_current': diffs, 'declared_design_variants': variants}
    # the 0.50-era records behind T8's superseded eps_AE column
    am = _load(AGEING_MECH)
    _input(AGEING_MECH)
    era050 = {}
    s46_results = next(p for p in am['inputs_sha256'] if p.endswith('campaign_s46_ageing/campaign_results.json'))
    q0_dir = _load(s46_results)['baseline_inputs']['Q0_eval_dir']
    q0_rec = os.path.join(q0_dir, 'evaluation_record.json')
    era_paths = [p for p in am['inputs_sha256'] if p.endswith('evaluation_record.json')] + [q0_rec]
    for p in era_paths:
        ar = as_run(p)
        cls, diffs, variants = classify(ar, cur)
        stage, _ = stage_of(p)
        pin = am['inputs_sha256'].get(p)
        era050[p] = {'role': ('Q(0) of the 0.50-era values (S46 campaign_results baseline_inputs.Q0_eval_dir; not '
                              'itself listed in ageing_mechanism inputs, reached through the pinned campaign_results)')
                     if p == q0_rec else 'ageing-batch point listed in ageing_mechanism.json inputs',
                     'ess_params_at_campaign_git_head': _ess_params_at_head(ar['campaign_spec_path']),
                     'sha256_now': ar['record_sha256'], 'sha256_pinned_by_ageing_mechanism': pin,
                     'matches_pin': True if pin is None else ar['record_sha256'] == pin, 'stage': stage, 'as_run': ar,
                     'classification': cls, 'differences_from_current': diffs, 'declared_design_variants': variants}
    eps050 = {r['arm']: r.get('eps_AE_050_superseded') for r in w157['tables']['ageing']['rows']}
    # the T6 uncoordinated arms: benchmark runs (not settling certificates)
    rep = _load(BENCH_REPORT)
    arms = {}
    for d in sorted(glob.glob(os.path.join(S53, 'w116_benchmark_nrf', 'nrf_arm_*_r2'))):
        j = os.path.join(d, os.path.basename(d) + '.json')
        if not os.path.exists(j):
            continue
        a = _load(j)
        so = a.get('solver_options') or {}
        ob = so.get('options_before') or {}
        arms[os.path.basename(d)] = {
            'record': j, 'sha256': _sha(j), 'committed_clean': _committed_clean(j),
            'instance_candidate_key': (a.get('instance') or {}).get('candidate_key'),
            'arm_network_compl_inf_tol': a.get('arm_network_compl_inf_tol'),
            'solver_options_compl_inf_tol_applied': so.get('compl_inf_tol'),
            'linear_solver_by_block_type': sorted({v.get('linear_solver') for v in ob.values() if isinstance(v, dict)}),
            'nlp_solver_path': a.get('nlp_solver_path'), 'interpreter': a.get('interpreter'),
            'case_or_ess_params_sha256_recorded': bool(re.search(cur['case_file_sha256'][:8] + '|' +
                                                                 cur['ess_params_sha256'][:8], json.dumps(a))),
            'gross_operational_cost': (a.get('arm_cost') or {}).get('gross_operational_cost')}
    old_items = []
    for ek, e in evals.items():
        if e['classification'] == 'OLD CONFIGURATION':
            old_items.append({'what': f"{', '.join(e['names'])} (eval {ek[:8]})", 'differences': e['differences_from_current'],
                              'table_rows': e['table_rows']})
    era_cfg = {}
    for p, e in era050.items():
        eff = e['as_run']['ageing_effective'] or {}
        era_cfg[p] = {'role': e['role'], 'ess_params_at_campaign_git_head': e['ess_params_at_campaign_git_head'],
                      'model_variant': e['as_run']['model_variant'], 'calibration': _calibration_label(eff.get('k', []), eff.get('enabled')),
                      'phi_cal': eff.get('phi_cal'), 'soh_floor': eff.get('soh_floor'), 'soh_point': eff.get('soh_point'),
                      'tail_in_force': e['as_run']['tail_in_force'], 'certified_at': e['as_run']['certification_cycle'],
                      'started_utc': e['as_run']['started_utc'], 'classification': e['classification'],
                      'candidate': e['as_run']['candidate_canonical']}
    old_items.append({
        'what': "T8 column 'eps_AE (0.50, superseded)' -- the Addendum 28 ageing-batch mechanism table "
                f"{AGEING_MECH} (b5eca2a2), formula elasticity = ln(value / value_C3) / ln(AE / AE_C3)",
        'values_in_T8': eps050,
        'records': era_cfg,
        'table_rows': [f'T8: {arm} -- column eps_AE (0.50, superseded)' for arm, v in eps050.items() if v is not None],
        'labelled_superseded_in_T8': True})
    superseded = [{'what': 'T5 row pb_y2025_n5 (W118 r2 certificate, rule v2)',
                   'note': next((r['note'] for r in w157['tables']['phase_b'] if r.get('superseded')), None)}]
    rule_classes = {}
    for ek, e in evals.items():
        r = e['as_run']['settling_rule']
        key = f"{e['stage']} | {r.get('module')} v{r.get('version')}"
        rule_classes.setdefault(key, []).append(', '.join(e['names']))
    CHECKS['a_every_table_eval_key_resolved_to_one_committed_run_record'] = not unresolved
    CHECKS['a_every_run_record_committed_clean'] = all(e['as_run']['record_committed_clean'] for e in evals.values())
    CHECKS['a_0_50_era_records_match_the_ageing_mechanism_pins'] = all(e['matches_pin'] for e in era050.values())
    CHECKS['a_stage_by_directory_agrees_with_T2_certifying_spec'] = all(
        e['stage_agrees_with_T2'] in (True, None) for e in evals.values())
    CHECKS['a_campaign_spec_sha256_matches_every_record'] = all(
        e['campaign_spec_sha256_matches_record'] for e in evals.values())
    return {'current_configuration': cur, 'n_committed_campaign_specs_indexed': n_specs, 'n_table_uses': len(uses),
            'evaluations': evals, 'unresolved': unresolved, 'era_050_records': era050,
            'benchmark_arm_records_T6': arms, 'old_configuration_items': old_items,
            'superseded_certificates_displayed': superseded, 'settling_rule_classes': rule_classes,
            'scope': ('every eval key named by T1-T10 of the W157 a64 JSON (claims, cells, the W145 fit points, the '
                      'year-ladder cells, Phase B rows, the coordinated x = 0, the W153 discount instance, the ageing '
                      'rows, the dead-zone and W149 rows, the A64 rows) and the records ageing_mechanism.json lists; '
                      f'run records located among {n_specs} committed campaign specs under data/SRP1/Results')}


# ======================================================================================================================
#  (b) T6 carries both uncoordinated arms?
# ======================================================================================================================
def part_b(w157):
    b = w157['tables']['benchmark']
    md = open(W157_MD).read()
    sec = md.split('## T6', 1)[1].split('\n## ', 1)[0]
    rep = _load(BENCH_REPORT)
    pa = rep['per_arm_nrf']
    fields = []

    def row(field, in_json, in_md, source, value):
        fields.append({'field': field, 'in_T6_json': in_json, 'in_T6_markdown': in_md, 'source': source, 'value': value})

    for arm in ('passive', 'price_taker'):
        a = b['arms'].get(arm, {})
        row(f'{arm}: Q (best of 3 starts)', 'Q_best' in a, f"{pa[arm]['q_best']:,.2f}" in sec,
            f'report_v3.json per_arm_nrf.{arm}.q_best', pa[arm]['q_best'])
        row(f'{arm}: best start', 'best_start' in a, pa[arm]['best_start'] in sec,
            f'report_v3.json per_arm_nrf.{arm}.best_start', pa[arm]['best_start'])
        row(f'{arm}: multimodality band', 'multimodality_band' in a,
            f"{pa[arm]['multimodality_band_eur']:,.3f}" in sec,
            f'report_v3.json per_arm_nrf.{arm}.multimodality_band_eur', pa[arm]['multimodality_band_eur'])
        row(f'{arm}: Q by start (cold / warm_from_certified / perturbed)', 'q_by_start' in a,
            all(re.search(re.escape(st) + r'[^|\n]{0,24}' + re.escape(f'{v:,.2f}'), sec)
                for st, v in pa[arm]['q_by_start'].items()),
            f'report_v3.json per_arm_nrf.{arm}.q_by_start', pa[arm]['q_by_start'])
    dec = rep['claim']['decomposition']
    row('claim (best arm, price-taker): benefit, relative, multiple, determinate', True,
        '+90,896,608.40' in sec, 'report_v3.json claim.benefit_eur / benefit_relative / determinate',
        {'benefit_eur': rep['claim']['benefit_eur'], 'benefit_relative': rep['claim']['benefit_relative'],
         'larger_band_eur': rep['claim']['larger_band_eur'], 'determinate': rep['claim']['determinate']})
    row('passive arm against coordinated: Q_passive - Q181', 'decomposition' in b and
        'passive_NRF_minus_coordinated_eur' in b.get('decomposition', {}),
        f"{dec['passive_NRF_minus_coordinated_eur']:,.2f}" in sec,
        'report_v3.json claim.decomposition.passive_NRF_minus_coordinated_eur', dec['passive_NRF_minus_coordinated_eur'])
    row('passive minus price-taker: Q_passive - Q_price_taker', 'decomposition' in b,
        f"{dec['passive_NRF_minus_price_taker_NRF_eur']:,.2f}" in sec,
        'report_v3.json claim.decomposition.passive_NRF_minus_price_taker_NRF_eur',
        dec['passive_NRF_minus_price_taker_NRF_eur'])
    row('passive arm: relative benefit and multiple of its own band', False, False,
        'NOT RECORDED in report_v3.json (only the best-arm claim carries benefit_relative and determinate); derivable '
        'from per_arm_nrf.passive.q_best, coordinated.q and the bands by the claim definition', None)
    row('coordinated band (reproducibility 0.011 %)', 'coordinated_reproducibility_band' in b, '71,926.11' in sec,
        'report_v3.json claim.bands_eur.coordinated_reproducibility_0.011pct',
        rep['claim']['bands_eur']['coordinated_reproducibility_0.011pct'])
    rf = rep['coordinated_reverse_flow_count']['totals']['material']
    row('reverse-flow interface-hours in the coordinated solution (and MWh)', 'reverse_flow_interface_hours' in b,
        'Reverse-flow interface-hours' in sec, 'report_v3.json coordinated_reverse_flow_count.totals.material',
        {'count': rf['count'], 'energy_mwh_block_weighted': rf['energy_mwh_block_weighted']})
    row('NRF arms: reverse flow excluded by the arm definition (pg_adn >= 0 rows)', False, False,
        'report_v3.json no_reverse_flow_definition (Addendum 57 Decision 1(b)); per-arm JSON no_reverse_flow', None)
    row('NRF arms: consistency re-evaluation NRF violations (n, max excess pu) per start', False, False,
        'report_v3.json consistency_nrf.<arm run>.nrf_violations',
        {k: v['nrf_violations'] for k, v in rep['consistency_nrf'].items()})
    for sw in ('sweep_passive_cold', 'sweep_price_taker_cold'):
        s = rep['sweep'][sw]
        row(f'{sw}: blocks / hours the TN cannot accept', sw in b.get('sweep', {}),
            f"{sw}: {s['n_blocks_tn_cannot_accept']}/12 blocks, {s['n_hours_tn_cannot_accept']} h" in sec,
            f'report_v3.json sweep.{sw}', {'n_blocks': s['n_blocks_tn_cannot_accept'],
                                            'n_hours': s['n_hours_tn_cannot_accept']})
        row(f'{sw}: which blocks fail', False, all(fb in sec for fb in s['failing_blocks']),
            f'report_v3.json sweep.{sw}.failing_blocks', s['failing_blocks'])
    blocks = {}
    for d in sorted(glob.glob(os.path.join(S53, 'w116_benchmark_nrf', 'nrf_arm_*_r2'))):
        j = os.path.join(d, os.path.basename(d) + '.json')
        a = _load(j)
        bc = ((a.get('phase_C_sequential_pass') or {}).get('evaluation') or {}).get('block_components')
        blocks[os.path.basename(d)] = {'path': j, 'field': 'phase_C_sequential_pass.evaluation.block_components',
                                       'n_blocks': len(bc) if isinstance(bc, dict) else None}
    row('per-block decomposition of each arm (12 TSO + 36 DSO blocks)', False, False,
        'per-arm run records nrf_arm_<arm>_<start>_r2.json phase_C_sequential_pass.evaluation.block_components '
        '(the arm cost source is phase_C_sequential_pass); not in report_v3.json', blocks)
    row('curtailment by agent per arm and start', False, False, 'report_v3.json curtailment_table.arms_and_variants_phase_A',
        None)
    both = all(r['in_T6_markdown'] for r in fields if r['field'].startswith(('passive: Q (best', 'price_taker: Q (best')))
    missing = [r for r in fields if not r['in_T6_markdown']]
    _input(BENCH_REPORT, BENCH_REPORT_MAN)
    _input(BENCH_SPEC)
    return {'T6_carries_both_arms_Q_and_band': both, 'fields': fields, 'missing_from_T6_markdown': missing,
            'report_v3_commit': (_git('log', '--format=%h', '-1', '--', BENCH_REPORT).strip()),
            'w157_builder_edited': False}


# ======================================================================================================================
#  (c) provenance reads
# ======================================================================================================================
def _thread_var_hits(text):
    return {v: bool(re.search(rf'\b{v}\b', text)) for v in THREAD_VARS}


def part_c1(evals, era050):
    out = {}
    cmds = {}
    for label, path in (('v6', V6_SPEC), ('ext v3', EXT_SPEC), ('A64 v1', A64_SPEC)):
        sp = _load(path)
        lc = sp.get('launch_commands') or {}
        names = set()
        for c in lc.values():
            pre = c.split(' -u ')[0] if isinstance(c, str) else ''
            names |= set(re.findall(r'(?:^|\s|&&\s*)([A-Z_][A-Z0-9_]*)=', pre))
        cmds[label] = {'n_commands': len(lc), 'env_assignments_before_interpreter': sorted(names)}
    out['launch_commands_env_assignments'] = cmds
    shells = {}
    for f in SHELL_FILES:
        if os.path.exists(f):
            try:
                txt = open(f, errors='replace').read()
                shells[f] = {'exists': True, **_mtime(f), 'thread_vars_named': _thread_var_hits(txt)}
            except PermissionError as exc:
                shells[f] = {'exists': True, 'error': str(exc)}
        else:
            shells[f] = {'exists': False}
    out['shell_profiles'] = shells
    snaps = {}
    for f in sorted(glob.glob(os.path.join(SNAPSHOT_DIR, '*'))):
        txt = open(f, errors='replace').read()
        snaps[f] = {**_mtime(f), 'thread_vars_named': _thread_var_hits(txt)}
    out['claude_shell_snapshots'] = {'dir': SNAPSHOT_DIR, 'files': snaps,
                                     'note': ('only the snapshot files present now are readable; the campaign-time '
                                              'snapshots are not preserved unless listed here with an earlier mtime')}
    env_path = os.path.join(REPO, '.env')
    if os.path.exists(env_path):
        txt = open(env_path, errors='replace').read()
        out['dot_env'] = {'exists': True, 'thread_vars_named': _thread_var_hits(txt),
                          'any_line_naming_THREAD': bool(re.search(r'THREAD', txt)),
                          'note': 'only these booleans are recorded; no content, no hash'}
        del txt
    else:
        out['dot_env'] = {'exists': False}
    lc = {}
    for v in THREAD_VARS:
        r = _run(['launchctl', 'getenv', v])
        lc[v] = {'rc': r['rc'], 'value': r['stdout'].strip() or None}
    out['launchctl_getenv'] = lc
    gg = _git('grep', '-n', '-E', '|'.join(THREAD_VARS[:5]), '--', '*.py')
    lines = [ln for ln in gg.split('\n') if ln]
    out['git_grep_tracked_python'] = {
        'pattern': '|'.join(THREAD_VARS[:5]), 'n_lines': len(lines),
        'files': sorted({ln.split(':', 1)[0] for ln in lines}),
        'production_modules_not_p5': sorted({ln.split(':', 1)[0] for ln in lines if not os.path.basename(ln.split(':', 1)[0]).startswith('p5')}),
        'harness_lines': [ln for ln in lines if ln.startswith(HARNESS + ':')]}
    src = open(HARNESS).read().split('\n')
    out['harness_thread_caps'] = {
        'file': HARNESS, 'sha256_now': _sha(HARNESS),
        'THREAD_CAP_ENV_definition_lines': [f'{i + 1}: {s}' for i, s in enumerate(src)
                                            if 'THREAD_CAP_ENV = {' in s or re.match(r"\s+'(OMP|MKL|OPENBLAS|VECLIB|NUMEXPR)_", s)],
        'child_env_update_lines': [f'{i + 1}: {s.strip()}' for i, s in enumerate(src) if 'child_env.update(THREAD_CAP_ENV)' in s],
        'child_refuses_without_caps_lines': [f'{i + 1}: {s.strip()}' for i, s in enumerate(src) if 'CHILD REFUSES: thread caps' in s]}
    out['this_process_environment'] = {v: os.environ.get(v) for v in THREAD_VARS}
    per = {}
    for ek, e in list(evals.items()) + [(p, {'names': [p], 'as_run': e['as_run']}) for p, e in era050.items()]:
        ar = e['as_run']
        per[', '.join(e['names'])] = {
            'launch_json_OMP': (ar['thread_caps_in_child_env_launch_json'] or {}).get('OMP_NUM_THREADS'),
            'record_seen_by_child_OMP': (ar['thread_caps_seen_by_child_record'] or {}).get('OMP_NUM_THREADS'),
            'child_stdout_caps': ar['child_stdout_first_line_caps'],
            'all_five_caps_1_in_record': (ar['thread_caps_seen_by_child_record'] or {}) == {
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}}
    out['per_evaluation_child_thread_caps'] = per
    out['all_table_evaluations_ran_with_OMP_NUM_THREADS_1'] = all(
        v['record_seen_by_child_OMP'] == '1' for v in per.values())
    return out


def _banner_lines(path, first_only=False):
    hits = []
    with open(path, errors='replace') as fh:
        for i, line in enumerate(fh, 1):
            if line.startswith('This is Ipopt version'):
                m = BANNER_RE.match(line)
                hits.append((i, line.rstrip('\n'), m.group(1) if m else None, m.group(2) if m else None))
                if first_only:
                    break
    return hits


def part_c2():
    out = {'samples': [], 'tally': {}}
    for spec, names in ((FIRST_V6, ['optim_log_case9_2025_Summer.log', 'optim_log_case33_2_2025_Summer.log',
                                    'optim_log_esso_node5_cycle001.txt']),
                        (LAST_LAUNCH, ['optim_log_case9_2025_Summer.log', 'optim_log_esso_node7_cycle001.txt'])):
        man = _load(spec['manifest'])
        _input(spec['manifest'])
        for n in names:
            p = os.path.join(spec['p56a'], n)
            rec = {'cell': spec['cell'], 'path': p, 'exists': os.path.exists(p),
                   'launch_manifest': spec['manifest'], 'manifest_entry': man.get(p)}
            if rec['exists']:
                rec['sha256_now'] = _sha(p)
                rec['matches_launch_manifest'] = rec['sha256_now'] == rec['manifest_entry']
                hits = _banner_lines(p)
                rec['first_banner_line_number'] = hits[0][0] if hits else None
                rec['first_banner_text'] = hits[0][1] if hits else None
                rec['n_banners'] = len(hits)
                rec['linear_solvers_in_all_banners'] = sorted({h[3] for h in hits})
                rec['ipopt_versions_in_all_banners'] = sorted({h[2] for h in hits})
                rec.update(_mtime(p))
            out['samples'].append(rec)
    man = _load(FIRST_V6['manifest'])
    tally = {'network': {}, 'esso': {}, 'n_files': 0, 'n_hash_match': 0, 'n_missing': 0, 'versions': set(),
             'files_without_banner': []}
    for p, h in sorted(man.items()):
        if '/P56A/' not in p:
            continue
        tally['n_files'] += 1
        if not os.path.exists(p):
            tally['n_missing'] += 1
            continue
        if _sha(p) == h:
            tally['n_hash_match'] += 1
        kind = 'esso' if '/optim_log_esso_' in p else 'network'
        hits = _banner_lines(p)
        if not hits:
            tally['files_without_banner'].append(p)
        for _i, _t, ver, ls in hits:
            tally[kind][ls] = tally[kind].get(ls, 0) + 1
            tally['versions'].add(ver)
    tally['versions'] = sorted(tally['versions'])
    out['tally_d_4a82a64a_all_manifest_logs'] = tally
    CHECKS['c2_sample_logs_exist_and_match_their_launch_manifest'] = all(
        s.get('exists') and s.get('matches_launch_manifest') for s in out['samples'])
    CHECKS['c2_every_d_4a82a64a_manifest_log_present_and_hash_matching'] = (
        tally['n_missing'] == 0 and tally['n_hash_match'] == tally['n_files'])
    return out


def _parse_conda_history(path):
    entries = []
    cur = None
    for line in open(path, errors='replace'):
        m = re.match(r'^==> (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) <==', line)
        if m:
            cur = {'timestamp_local': m.group(1),
                   'epoch': time.mktime(time.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')), 'lines': 0, 'cmd': None}
            entries.append(cur)
        elif cur is not None:
            cur['lines'] += 1
            if line.startswith('# cmd:'):
                cur['cmd'] = line.strip()
    for e in entries:
        e['utc'] = _iso_from_epoch(e['epoch'])
    return entries


def part_c3():
    out = {'read_at_utc': _utc(), 'interpreter': sys.executable, 'sys_version': sys.version,
           'pyomo_version': pyomo.version.version, 'platform': sys.platform}
    out['sw_vers'] = _run(['sw_vers'])['stdout']
    out['uname'] = _run(['uname', '-a'])['stdout'].strip()
    lj = _load(FIRST_V6['launch_json'])
    t0 = _epoch_from_iso(lj['started_utc'])
    cr = _load(LAST_LAUNCH['campaign_results'])
    hb = _load(os.path.join(LAST_LAUNCH['campaign_root'], 'campaign_heartbeat.json'))
    last_line = open(LAST_LAUNCH['launch_log']).read().rstrip('\n').split('\n')[-1][:8]
    t1 = _epoch_from_iso(cr['utc']) if 'utc' in cr else _epoch_from_iso(hb['utc'])
    out['window'] = {'first_v6_cell_start_utc': lj['started_utc'], 'first_v6_source': FIRST_V6['launch_json'],
                     'last_launch_end_utc': _iso_from_epoch(t1),
                     'last_launch_source': f"{LAST_LAUNCH['campaign_results']} utc (heartbeat {hb['utc']}; launch "
                                           f"log last line stamped {last_line} UTC)",
                     'last_launch_start_utc': _load(LAST_LAUNCH['launch_json'])['started_utc']}
    _input(FIRST_V6['launch_json'])
    _input(LAST_LAUNCH['campaign_results'])
    hist = _parse_conda_history(CONDA_HISTORY)
    out['conda_history'] = {'path': CONDA_HISTORY, **_mtime(CONDA_HISTORY), 'n_entries': len(hist),
                            'entries': [{k: e[k] for k in ('timestamp_local', 'utc', 'lines', 'cmd')} for e in hist],
                            'last_entry_local': hist[-1]['timestamp_local'] if hist else None,
                            'timezone_note': f'history timestamps are local time ({time.tzname}); converted with the '
                                             'system timezone rules',
                            'entries_at_or_after_first_v6_start': [e['timestamp_local'] for e in hist if e['epoch'] >= t0]}
    newest = (0, None)
    n_after = []
    for root, dirs, files in os.walk(CONDA_ENV):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in files:
            if f.endswith('.pyc'):
                continue
            p = os.path.join(root, f)
            try:
                m = os.lstat(p).st_mtime
            except OSError:
                continue
            if m > newest[0]:
                newest = (m, p)
            if m >= t0:
                n_after.append(p)
    out['conda_env_files'] = {'newest_mtime_utc': _iso_from_epoch(newest[0]), 'newest_file': newest[1],
                              'n_files_modified_at_or_after_first_v6_start': len(n_after),
                              'files_modified_at_or_after_first_v6_start': n_after[:50],
                              'scope': f'every file under {CONDA_ENV} except __pycache__ and *.pyc (lstat mtime)'}
    py_real = os.path.realpath(sys.executable)
    out['python_binary'] = {'path': sys.executable, 'realpath': py_real, **_mtime(py_real)}
    su = _run(['softwareupdate', '--history'], timeout=180)
    rows = []
    for line in su['stdout'].split('\n'):
        m = re.search(r'(\d\d)/(\d\d)/(\d{4}), (\d\d:\d\d:\d\d)\s*$', line)
        if m:
            ts = f'{m.group(3)}-{m.group(2)}-{m.group(1)} {m.group(4)}'
            ep = time.mktime(time.strptime(ts, '%Y-%m-%d %H:%M:%S'))
            rows.append({'line': line.strip(), 'local': ts, 'utc': _iso_from_epoch(ep), 'at_or_after_first_v6': ep >= t0,
                         'epoch': ep})
    out['softwareupdate_history'] = {'rc': su['rc'], 'n_rows': len(rows),
                                     'rows': [{k: r[k] for k in ('line', 'local', 'utc', 'at_or_after_first_v6')} for r in rows],
                                     'date_format_assumed': 'dd/mm/yyyy, HH:MM:SS local (as printed)',
                                     'rows_at_or_after_first_v6': [r['line'] for r in rows if r['epoch'] >= t0]}
    sv = '/System/Library/CoreServices/SystemVersion.plist'
    out['system_version_plist'] = {'path': sv, **_mtime(sv)}
    v6 = _load(V6_SPEC)
    pin = v6['pins']['solver']
    ip = {'path': IPOPT, **_mtime(IPOPT), 'sha256': _sha(IPOPT), 'v6_spec_pin_sha256': pin.get('sha256')}
    ip['matches_v6_pin'] = ip['sha256'] == pin.get('sha256')
    ip['version_output'] = _run([IPOPT, '--version'])
    ip['version_note'] = ('`ipopt --version` prints the version and exits; it is a plain subprocess call, not a Pyomo '
                          'solve, so the guard neither intercepts nor counts it')
    ip['modified_at_or_after_first_v6'] = ip['mtime_epoch'] >= t0
    out['ipopt'] = ip
    libs = {}
    for lib in IPOPT_LIBS:
        if os.path.exists(lib):
            libs[lib] = {**_mtime(lib), 'sha256': _sha(lib)}
            libs[lib]['modified_at_or_after_first_v6'] = libs[lib]['mtime_epoch'] >= t0
    out['ipopt_libraries'] = libs
    changes = []
    if out['conda_history']['entries_at_or_after_first_v6_start']:
        changes.append(f"conda history entries: {out['conda_history']['entries_at_or_after_first_v6_start']}")
    if n_after:
        changes.append(f'{len(n_after)} files in the conda environment modified at or after the first v6 start')
    if out['softwareupdate_history']['rows_at_or_after_first_v6']:
        changes.append(f"softwareupdate rows: {out['softwareupdate_history']['rows_at_or_after_first_v6']}")
    if ip['modified_at_or_after_first_v6'] or not ip['matches_v6_pin']:
        changes.append('ipopt binary modified or differs from the v6 pin')
    for lib, r in libs.items():
        if r['modified_at_or_after_first_v6']:
            changes.append(f'{lib} modified at or after the first v6 start')
    out['changes_at_or_after_first_v6_start'] = changes
    out['environment_unchanged_since_first_v6_start_by_these_reads'] = not changes
    return out


def _tar_members(path):
    with tarfile.open(path, 'r:gz') as tf:
        return [m for m in tf.getmembers()]


def _tar_read(path, name):
    with tarfile.open(path, 'r:gz') as tf:
        f = tf.extractfile(name)
        return None if f is None else f.read()


def _search_text(text, patterns):
    hits = []
    for i, line in enumerate(text.split('\n'), 1):
        for pat in patterns:
            if re.search(pat, line, re.IGNORECASE):
                hits.append({'line': i, 'pattern': pat, 'text': line.rstrip()})
                break
    return hits


def part_c4():
    out = {'patterns': STATEMENT_PATTERNS}
    src = next((p for p in HSL_SRC_ARCHIVE_CANDIDATES if os.path.exists(p)), None)
    out['source_archive_candidates'] = {p: os.path.exists(p) for p in HSL_SRC_ARCHIVE_CANDIDATES}
    out['source_archive'] = src
    doc_re = re.compile(r'(?i)(\.pdf|\.txt|\.md|\.html?|\.tex|\.ps|readme|changelog|licen[cs]e|/doc)')
    if src:
        mem = _tar_members(src)
        out['source_archive_sha256'] = _sha(src)
        out['source_archive_n_members'] = len(mem)
        out['source_archive_doc_like_members'] = [m.name for m in mem if m.isfile() and doc_re.search(m.name)]
        out['source_archive_hsl_ma97_members'] = [m.name for m in mem if 'ma97' in m.name.lower()]
        out['source_archive_pdf_members'] = [m.name for m in mem if m.name.lower().endswith('.pdf')]
        searched = {}
        for m in mem:
            if not m.isfile():
                continue
            if doc_re.search(m.name) or 'ma97' in m.name.lower():
                b = _tar_read(src, m.name)
                txt = b.decode('utf-8', errors='replace')
                searched[m.name] = {'n_lines': txt.count('\n'), 'hits': _search_text(txt, STATEMENT_PATTERNS)}
                if m.name.endswith('hsl_ma97d.f90'):
                    out['hsl_ma97d_f90'] = {'member': m.name, 'n_lines': txt.count('\n'),
                                            'version_lines': [ln for ln in txt.split('\n')[:12] if 'Version' in ln],
                                            'num_threads_source_lines': [
                                                f'{i}: {ln.strip()}' for i, ln in enumerate(txt.split('\n'), 1)
                                                if 'omp_get_max_threads' in ln][:6]}
                    MA97_LINES.update({i: ln for i, ln in enumerate(txt.split('\n'), 1)})
        out['source_archive_searched_members'] = searched
        out['source_archive_all_hits'] = [{'member': k, **h} for k, v in searched.items() for h in v['hits']]
    if os.path.exists(HSL_BIN_ARCHIVE):
        mem = _tar_members(HSL_BIN_ARCHIVE)
        out['binary_archive'] = HSL_BIN_ARCHIVE
        out['binary_archive_sha256'] = _sha(HSL_BIN_ARCHIVE)
        out['binary_archive_n_members'] = len(mem)
        out['binary_archive_doc_like_members'] = [m.name for m in mem if m.isfile() and doc_re.search(m.name)]
        out['binary_archive_pdf_members'] = [m.name for m in mem if m.name.lower().endswith('.pdf')]
        hits = []
        for m in mem:
            if m.isfile() and (doc_re.search(m.name) or m.name.endswith(('hsl_ma97d.h', 'hsl_ma97s.h'))):
                txt = (_tar_read(HSL_BIN_ARCHIVE, m.name) or b'').decode('utf-8', errors='replace')
                hits += [{'member': m.name, **h} for h in _search_text(txt, STATEMENT_PATTERNS)]
        out['binary_archive_hits'] = hits
        lib = next((m.name for m in mem if m.name.endswith('lib/libcoinhsl.dylib') and m.isfile()), None)
        out['binary_archive_libcoinhsl_member'] = lib
        if lib:
            out['binary_archive_libcoinhsl_sha256'] = _sha_bytes(_tar_read(HSL_BIN_ARCHIVE, lib))
    # what the installed IPOPT links
    hsl = '/usr/local/lib/libcoinhsl.2.dylib'
    inst = {'otool_L_ipopt': _run(['otool', '-L', IPOPT])['stdout'],
            'otool_L_libcoinhsl': _run(['otool', '-L', hsl])['stdout'],
            'libcoinhsl_sha256': _sha(hsl)}
    nm_all = _run(['nm', hsl])['stdout'].split('\n')
    nm_u = _run(['nm', '-u', hsl])['stdout'].split('\n')
    inst['nm_symbols_matching_ma97'] = sum(1 for s in nm_all if 'ma97' in s.lower())
    inst['nm_symbols_openmp'] = [s.strip() for s in nm_all if re.search(r'GOMP_|\bomp_|_omp_|__kmpc', s)][:20]
    inst['nm_undefined_openmp'] = [s.strip() for s in nm_u if re.search(r'GOMP_|\bomp_|_omp_|__kmpc', s)][:20]
    inst['links_libgomp_or_libomp'] = bool(re.search(r'lib(g|i)?omp', inst['otool_L_libcoinhsl']))
    raw = open(hsl, 'rb').read()
    refs = sorted({int(x) for x in re.findall(rb'At line (\d+) of file coinhsl/hsl_ma97/hsl_ma97d\.f90', raw)})
    inst['runtime_message_line_refs_hsl_ma97d_f90'] = refs
    inst['source_paths_named'] = sorted({x.decode() for x in re.findall(rb'coinhsl/hsl_ma97/[A-Za-z0-9_/]+\.f90', raw)})
    if MA97_LINES:
        inst['archive_source_lines_at_those_refs'] = {r: MA97_LINES.get(r, '').strip() for r in refs}
    del raw
    if os.path.exists(IPOPT_PKG_ARCHIVE):
        mem = _tar_members(IPOPT_PKG_ARCHIVE)
        inst['ipopt_package_archive'] = IPOPT_PKG_ARCHIVE
        inst['ipopt_package_archive_sha256'] = _sha(IPOPT_PKG_ARCHIVE)
        inst['ipopt_package_archive'] = {'path': IPOPT_PKG_ARCHIVE, **_mtime(IPOPT_PKG_ARCHIVE),
                                         'members': [m.name for m in mem][:40]}
        for nm, local in (('lib/libcoinhsl.2.dylib', hsl), ('bin/ipopt', IPOPT)):
            b = _tar_read(IPOPT_PKG_ARCHIVE, nm)
            inst[f'package_{nm}_sha256'] = None if b is None else _sha_bytes(b)
            inst[f'package_{nm}_equals_installed'] = (b is not None and _sha_bytes(b) == _sha(local))
    tp = os.path.join(THIRDPARTY_HSL, '.libs', 'libcoinhsl.2.dylib')
    if os.path.exists(tp):
        inst['thirdparty_hsl_build'] = {
            'path': tp, **_mtime(tp), 'sha256': _sha(tp), 'equals_installed': _sha(tp) == inst['libcoinhsl_sha256'],
            'otool_L': _run(['otool', '-L', tp])['stdout'],
            'coinhsl_symlink': os.path.realpath(os.path.join(THIRDPARTY_HSL, 'coinhsl'))}
    out['installed_ipopt_hsl'] = inst
    return out


MA97_LINES = {}


# ======================================================================================================================
#  main
# ======================================================================================================================
def _md(doc):
    a, c = doc['a'], doc['c']
    L = []
    w = L.append
    w('# P5.15 W159 — closing reads before the Step 6 tables freeze (Addendum 65)')
    w('')
    w(f"Generated {doc['utc']} by `{SCRIPT_REL}` (sha256 `{doc['script_sha256'][:12]}`), git HEAD `{doc['git_head'][:12]}`; "
      'zero solves (guard verified at 0), pickle blocked (0 calls). Every value below is read from records; the JSON '
      f'`{OUT_JSON}` carries the full evidence.')
    w('')
    w('## (a) Certifying spec and as-run configuration of every evaluation T1–T10 use')
    w('')
    cur = a['current_configuration']
    w(f"Current configuration (operational definition: `{cur['source']}`): case file `{cur['case_file_sha256'][:8]}`, "
      f"ESS params `{cur['ess_params_sha256'][:8]}` (E/P bounds {cur['energy_to_power_factor_bounds_h']} h), AA "
      f"{cur['anderson_acceleration']['reject_policy']}, tight tail compl_inf_tol {cur['tail']['compl_inf_tol']}, "
      f"{cur['ageing_baseline_label']} (k {cur['k_expected']:,.2f}).")
    w('')
    w('| evaluation | stage (certifying spec) | rule | status / k* | started (UTC) | classification | variant / differences | tables |')
    w('|---|---|---|---|---|---|---|---|')
    for ek, e in sorted(a['evaluations'].items(), key=lambda kv: (kv[1]['stage'] or '', kv[1]['names'])):
        ar = e['as_run']
        r = ar['settling_rule']
        tbls = sorted({x.split(':')[0] for x in e['table_rows']}, key=lambda s: int(s[1:]))
        dv = '; '.join(e['differences_from_current'] + e['declared_design_variants']) or '—'
        w(f"| {', '.join(e['names'])} (`{ek[:8]}`) | {e['stage']} | {r.get('module')} | {ar['status']} / "
          f"{ar['certification_cycle']} | {(ar['started_utc'] or '')[:19]} | {e['classification']} | {dv} | {', '.join(tbls)} |")
    w('')
    if a['unresolved']:
        w(f"**Unresolved eval keys:** {a['unresolved']}")
        w('')
    w('### Old-configuration figures still in a manuscript table')
    w('')
    for it in a['old_configuration_items']:
        w(f"- **{it['what']}**")
        if 'records' in it:
            for p, r in it['records'].items():
                eh = r['ess_params_at_campaign_git_head'] or {}
                w(f"  - `{p}` ({r['role']}): ESS params at git_head `{str(eh.get('git_head'))[:8]}`: calibration "
                  f"{eh.get('calibration')}, minimum_soh {eh.get('minimum_soh')}, E/P "
                  f"[{eh.get('min_energy_to_power_factor')}, {eh.get('max_energy_to_power_factor')}] h, calendar "
                  f"retention {eh.get('calendar_retention_per_year')}")
                w(f"    read back from the record: variant {r['model_variant']}; calibration {r['calibration']}; phi_cal {r['phi_cal']}; "
                  f"SoH floor {r['soh_floor']}; SoH point {r['soh_point']}; tail in force {r['tail_in_force']}; "
                  f"certified at {r['certified_at']}; started {r['started_utc']}; classification {r['classification']}")
            w(f"  - T8 values: {it['values_in_T8']}")
        else:
            w(f"  - differences: {it['differences']}")
        w(f"  - rows: {it['table_rows']}")
    w('')
    w('### Superseded certificate displayed')
    w('')
    for s in a['superseded_certificates_displayed']:
        w(f"- {s['what']}: {s['note']}")
    w('')
    w('### T6 uncoordinated arms (benchmark runs, spec v5 bca69f97; not settling certificates)')
    w('')
    for k, v in a['benchmark_arm_records_T6'].items():
        w(f"- `{k}`: instance `{(v['instance_candidate_key'] or '')[:8]}`, arm network compl_inf_tol "
          f"{v['arm_network_compl_inf_tol']}, linear solvers {v['linear_solver_by_block_type']}, case / ESS params "
          f"sha recorded: {v['case_or_ess_params_sha256_recorded']}")
    w('')
    b = doc['b']
    w('## (b) T6 and the two uncoordinated arms')
    w('')
    w(f"T6 carries Q and band of both arms: **{b['T6_carries_both_arms_Q_and_band']}**. Fields:")
    w('')
    w('| field | in T6 JSON | in T6 Markdown | source |')
    w('|---|---|---|---|')
    for f in b['fields']:
        w(f"| {f['field']} | {f['in_T6_json']} | {f['in_T6_markdown']} | {f['source']} |")
    w('')
    w('## (c) Provenance reads')
    w('')
    c1 = c['c1']
    w('### 1. Threading')
    w('')
    w(f"- Launch-command environment assignments: {c1['launch_commands_env_assignments']}")
    w(f"- Shell profiles naming a thread variable: "
      f"{[f for f, v in c1['shell_profiles'].items() if v.get('exists') and any((v.get('thread_vars_named') or {}).values())] or 'none'}"
      f" (checked: {[f for f, v in c1['shell_profiles'].items() if v.get('exists')]})")
    w(f"- Claude shell snapshots: {[(os.path.basename(f), v['mtime_utc'][:19], any(v['thread_vars_named'].values())) for f, v in c1['claude_shell_snapshots']['files'].items()]}")
    w(f"- .env names a thread variable: {c1['dot_env'].get('thread_vars_named')}")
    lcg = {k: v['value'] for k, v in c1['launchctl_getenv'].items()}
    w(f"- launchctl getenv: {lcg}")
    w(f"- Tracked Python naming thread variables: production (non-p5) {c1['git_grep_tracked_python']['production_modules_not_p5']}; "
      f"harness lines {c1['harness_thread_caps']['THREAD_CAP_ENV_definition_lines'][:2]} … "
      f"{c1['harness_thread_caps']['child_env_update_lines']} {c1['harness_thread_caps']['child_refuses_without_caps_lines']}")
    w(f"- This process (the current shell's export): {c1['this_process_environment']}")
    w(f"- Every table evaluation's child saw OMP_NUM_THREADS = '1': **{c1['all_table_evaluations_ran_with_OMP_NUM_THREADS_1']}**")
    w('')
    c2 = c['c2']
    w('### 2. Linear-solver banners')
    w('')
    w('| cell | log | sha256 | matches launch manifest | first banner line | banner | all banners |')
    w('|---|---|---|---|---:|---|---|')
    for s in c2['samples']:
        w(f"| {s['cell']} | `{s['path']}` | `{s.get('sha256_now', '')[:16]}` | {s.get('matches_launch_manifest')} | "
          f"{s.get('first_banner_line_number')} | {s.get('first_banner_text')} | {s.get('n_banners')} × "
          f"{s.get('linear_solvers_in_all_banners')} |")
    t = c2['tally_d_4a82a64a_all_manifest_logs']
    w('')
    w(f"All {t['n_files']} logs the d_4a82a64a launch manifest lists: {t['n_hash_match']} hash-match, {t['n_missing']} "
      f"missing; banners network {t['network']}, ESSO {t['esso']}; versions {t['versions']}; files without a banner "
      f"{len(t['files_without_banner'])}.")
    w('')
    c3 = c['c3']
    w('### 3. Versions now and environment changes')
    w('')
    w(f"- Read {c3['read_at_utc']}: Python `{c3['sys_version']}`; Pyomo {c3['pyomo_version']}; "
      f"sw_vers `{' '.join(c3['sw_vers'].split())}`")
    w(f"- Window: first v6 cell start {c3['window']['first_v6_cell_start_utc']}; last launch end "
      f"{c3['window']['last_launch_end_utc']}")
    w(f"- conda-meta/history: {c3['conda_history']['n_entries']} entr(y/ies); last {c3['conda_history']['last_entry_local']} "
      f"(local); entries at or after the first v6 start: {c3['conda_history']['entries_at_or_after_first_v6_start']}")
    w(f"- Environment files: newest mtime {c3['conda_env_files']['newest_mtime_utc']}; modified at or after the first "
      f"v6 start: {c3['conda_env_files']['n_files_modified_at_or_after_first_v6_start']}")
    w(f"- softwareupdate --history rows at or after the first v6 start: {c3['softwareupdate_history']['rows_at_or_after_first_v6']}; "
      f"all rows: {[r['line'] for r in c3['softwareupdate_history']['rows']]}")
    w(f"- /usr/local/bin/ipopt: mtime {c3['ipopt']['mtime_utc']}, sha256 `{c3['ipopt']['sha256'][:16]}` matches the v6 "
      f"pin: {c3['ipopt']['matches_v6_pin']}; `--version`: `{c3['ipopt']['version_output']['stdout'].strip()}`")
    w(f"- Changes at or after the first v6 start found by these reads: {c3['changes_at_or_after_first_v6_start'] or 'none'}")
    w('')
    c4 = c['c4']
    w('### 4. HSL_MA97 bit-compatibility statement')
    w('')
    w(f"- Source archive: `{c4.get('source_archive')}` ({c4.get('source_archive_n_members')} members); doc-like members "
      f"{c4.get('source_archive_doc_like_members')}; PDF members {c4.get('source_archive_pdf_members')}")
    w(f"- Binary archive: `{c4.get('binary_archive')}`; doc-like members {c4.get('binary_archive_doc_like_members')}; "
      f"PDF members {c4.get('binary_archive_pdf_members')}")
    w(f"- Pattern hits in the source archive: {[(h['member'], h['line'], h['text']) for h in c4.get('source_archive_all_hits', [])]}")
    w(f"- Pattern hits in the binary archive: {[(h['member'], h['line'], h['text']) for h in c4.get('binary_archive_hits', [])]}")
    inst = c4['installed_ipopt_hsl']
    w(f"- Installed libcoinhsl `{inst['libcoinhsl_sha256'][:16]}`: ma97 symbols {inst['nm_symbols_matching_ma97']}; "
      f"OpenMP symbols {inst['nm_symbols_openmp']}; links libgomp/libomp {inst['links_libgomp_or_libomp']}; equals the "
      f"ipopt3.14.18hsl5.5.0 package member: {inst.get('package_lib/libcoinhsl.2.dylib_equals_installed')}; runtime "
      f"message line refs into hsl_ma97d.f90 {inst['runtime_message_line_refs_hsl_ma97d_f90']} -> archive source lines "
      f"{inst.get('archive_source_lines_at_those_refs')}")
    tp = inst.get('thirdparty_hsl_build')
    if tp:
        w(f"- ~/ThirdParty-HSL build `{tp['sha256'][:16]}` (mtime {tp['mtime_utc'][:10]}) equals installed: {tp['equals_installed']}")
    w('')
    w('## Checks')
    w('')
    for k, v in doc['checks'].items():
        w(f'- {k}: {v}')
    w('')
    return '\n'.join(L) + '\n'


def main():
    t_start = time.time()
    if not os.path.isdir(OUT_DIR):
        raise SystemExit(f'{OUT_DIR} must exist (the launch command creates it for launch.log)')
    extra = sorted(set(os.listdir(OUT_DIR)) - {'launch.log'})
    if extra:
        raise SystemExit(f'{OUT_DIR} already holds {extra}: refusing (outputs are never overwritten)')
    _log(f'script {SCRIPT_REL} sha256 {_sha(SCRIPT_REL)} committed clean {_committed_clean(SCRIPT_REL)}')
    w157_rec = _input(W157_JSON, W157_MAN)
    _input(W157_MD, W157_MAN)
    for p in (V6_SPEC, EXT_SPEC, A64_SPEC, V4_SPEC, V5_SPEC, W145_JSON, W153D_JSON, W149_JSON, W127_MAN, ESS_PARAMS, HARNESS):
        _input(p)
    w157 = _load(W157_JSON)
    cur = current_configuration()
    _log('(a) certifying spec and as-run configuration')
    a = part_a(w157, cur)
    _log(f"(a) {len(a['evaluations'])} evaluations, {len(a['unresolved'])} unresolved, "
         f"old-configuration items {len(a['old_configuration_items'])}")
    _log('(b) T6 arms')
    b = part_b(w157)
    _log(f"(b) both arms Q and band in T6: {b['T6_carries_both_arms_Q_and_band']}; fields missing from the T6 Markdown "
         f"{len(b['missing_from_T6_markdown'])}")
    _log('(c)1 threading')
    c1 = part_c1(a['evaluations'], a['era_050_records'])
    _log('(c)2 banners')
    c2 = part_c2()
    _log('(c)3 versions')
    c3 = part_c3()
    _log('(c)4 HSL_MA97')
    c4 = part_c4()
    CHECKS['inputs_committed_clean'] = all(v['committed_clean'] for v in INPUTS.values())
    CHECKS['inputs_with_manifest_match'] = all(v.get('matches_manifest', True) for v in INPUTS.values())
    CHECKS['w157_tables_json_matches_its_manifest'] = w157_rec['matches_manifest']
    guard = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    CHECKS['zero_solve_guard_verified_0'] = guard['verify_0_failures'] == []
    CHECKS['pickle_load_and_loads_0'] = PICKLE_COUNTS == {'load': 0, 'loads': 0}
    failed = [k for k, v in CHECKS.items() if v is not True]
    doc = {'stage': 'P5.15 Addendum 65 W159 -- closing reads before the Step 6 tables freeze (zero solves)',
           'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 65 (closing reads (a), (b), (c)); Planner task W159',
           'utc': _utc(), 'git_head': _git('rev-parse', 'HEAD').strip(), 'script': SCRIPT_REL,
           'script_sha256': _sha(SCRIPT_REL), 'inputs': INPUTS, 'a': a, 'b': b,
           'c': {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4}, 'checks': dict(CHECKS), 'failed_checks': failed,
           'guard': guard, 'pickle_counts': dict(PICKLE_COUNTS), 'wall_s': time.time() - t_start,
           'exit_code': 0 if not failed else 3}
    out_json = os.path.join(OUT_DIR, OUT_JSON)
    out_md = os.path.join(OUT_DIR, OUT_MD)
    GRIO.check(doc)
    md = _md(doc)
    with open(out_json, 'x') as fh:
        GRIO.dump(doc, fh, indent=1, sort_keys=True)
    with open(out_md, 'x') as fh:
        fh.write(md)
    man = {p: _sha(p) for p in (out_json, out_md, SCRIPT_REL)}
    man.update({p: v['sha256'] for p, v in INPUTS.items()})
    with open(os.path.join(OUT_DIR, OUT_MAN), 'x') as fh:
        GRIO.dump(man, fh, indent=1, sort_keys=True)
    for k, v in CHECKS.items():
        _log(f'check {k}: {v}')
    _log(f"guard {guard}; pickle {PICKLE_COUNTS}; wrote {out_json}, {out_md}, {os.path.join(OUT_DIR, OUT_MAN)}; "
         f"exit {doc['exit_code']}; wall {doc['wall_s']:.1f} s")
    return doc['exit_code']


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
