"""
P5.15 W185 (E2) -- per-cell diagnostics of the eight E2 stage-1 certified evaluations (STEP7_SIMULATION_PLAN_2026-10-25
section 4, "feasibility checks per cell"; Addendum 85 ruling 6 / Addendum 86: one guarded-read authorization covers as
many guarded reads as the task needs, each logged with its run; W183i (d): the run's own persisted certified_models.pkl
is the capture path). ZERO SOLVES, NO MODEL BUILD, EXACTLY EIGHT PICKLE LOADS (one per declared (path, sha256)).
Precedent and code source: p515_s55_w184_e3_diagnostics.py (W184), copied here, not imported (importing it would arm
its own one-file pickle gate).

INSTANCES (the eight E2 cells, in the launcher's order; plan x = 0 or the node-7 reference unit 0.25 MVA / 1 MWh, 2025)
  e2_x0_1x_rho0p5, e2_unit_1x_rho0p5, e2_x0_1x_rho2p0, e2_unit_1x_rho2p0,
  e2_x0_2x_rho0p5, e2_unit_2x_rho0p5, e2_x0_2x_rho2p0, e2_unit_2x_rho2p0
  each identified by its eval key (asserted = the eval dir prefix and the evaluation_record) and its candidate key.

OBJECTIVE CONVENTION. Q = certified gross_operational_cost, contracted interface settlement excluded (Addendum 83
ruling 2). Values are quoted only to identify the instance; nothing here recomputes Q.

PER CELL
  (1) closure slack. Recorded: component_levels_terminal unweighted shared_ess_day_balance_slack per block (EUR) is
      base * PENALTY_SHARED_ESS_BALANCE * (s+ + s-) [p.u.] (model_construction_helpers.shared_ess_day_balance_slack_
      penalty), so the closure level S = s+ + s- in MWh = value / PENALTY_SHARED_ESS_BALANCE (dt = 1 h). From the
      pickle: every slack Var slack_shared_es_soc_final_up/down x baseMVA, its lb/ub x baseMVA, fixed status, and the
      sess_soc_final row residual. Limit (W180d (ii)): each slack within IPOPT's bound relaxation of its lower bound 0,
      i.e. >= -1e-8 p.u. x baseMVA x 1 h = -1e-6 MWh, and |slack| <= 1e-6 MWh; and <= its ub (0.05 E^Av + 1e-5 p.u.).
  (2) converter-circle residual pnet^2 + qnet^2 - S^2 (MVA^2) over ACTIVE sess_converter_capability rows only
      (blocks x hours); max, sign, and the row body - upper.
  (3) max |P^E| / S^Av = max |shared_es_pnet| / shared_es_s_rated_fixed over entries with S > 0 (and the apparent
      sqrt(pnet^2 + qnet^2) / S beside it).
  (4) SOC against [0.10 E^Av, 0.90 E^Av] (definitions.ENERGY_STORAGE_MIN/MAX_ENERGY_STORED), E^Av =
      shared_es_e_rated_fixed as the model holds it, compared with the record's published_available_capacity_terminal;
      min margins and violation counts (strict < 0, and < -1e-6 MWh), over active rows.
  (5) interface residual ratios: evaluation_record terminal_ratios_per_channel (primal, dual, max), boyd_terminal
      binding_test_per_channel, rule_ten; per-hour consensus residual max |r| per DSO node (settlement detail).
  (6) settlement: recourse_components interface_settlement_tso / _dso / _total; identity
      t_tso_plus_t_dso_terminal == -sum(priced residual weighted) and its residual.
  (7) "ESSO sigma slack": no quantity of that name exists in the code (W183i (d) Blocked 2) -- not applicable. The ESSO
      models (esso_models_s39_D.pkl) are not on the allow-list and are not read.
  At x = 0 cells the storage is zero-capacity (fixed variables, inactive rows): what the model holds is reported.

GUARDS
  SolveProfileGuard(permitted=()) installed before any project import, verified at exactly 0 at the end.
  pickle.load / loads / Unpickler replaced for the whole run by an allow-list gate of exactly the eight (path, sha256)
  pairs below (sha256 taken from each cell's COMMITTED campaign_manifest_sha256.json; checked before any load against
  the committed manifest (git show HEAD:), the working-tree manifest, post_certification.json and the file's bytes);
  the bytes hashed are the bytes unpickled; each path at most once; exactly 8 delegated loads (0 with --no-pickle,
  len(--cells) in a trial). Each load is logged with its run. Negative controls (must refuse, never delegate):
    NC_loads_bytes          pickle.loads(bytes)
    NC_load_in_memory       pickle.load(BytesIO)
    NC_unpickler            pickle.Unpickler(...)
    NC_non_listed           pickle.load of a non-listed pickle (cell 1's esso_models_s39_D.pkl; refused before read)
    NC_wrong_sha256         a gate of the same class holding cell 1's path with a wrong sha256 refuses the real file
    NC_second_load[cell]    a second pickle.load of each allowed file, after its load
  pyomo Model.__init__ blocked and counted (0 required).
  Refuses: an output directory that is missing or holds anything but launch.log; a concurrent copy; a missing capture
  path (asserted before any reading); any recorded input whose sha256 differs from its committed campaign manifest.

OUTPUT (default data/SRP1/Results/P515S55/w185_e2_diagnostics/): w185_e2_diagnostics.json, manifest_sha256.json;
`--post-run` writes manifest_post_run_sha256.json over the outputs + launch.log. `--out-dir DIR` redirects (trials);
`--cells a,b` restricts the cells (trials only); `--no-pickle` skips the loads (trials only).

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S55/w185_e2_diagnostics && set -o noclobber && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w185_e2_diagnostics.py \\
      > data/SRP1/Results/P515S55/w185_e2_diagnostics/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w185_e2_diagnostics.py --post-run
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W185 E2 diagnostics (never solves)').install()

import gc  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import math  # noqa: E402

P55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
E2 = os.path.join(P55, 'e2_rho')
# (cell, eval-dir basename, certified_models.pkl sha256 from the committed campaign manifest), launcher order
CELL_TABLE = (
    ('e2_x0_1x_rho0p5', '64f3605028f1cf87_e2_x0_1x_rho0p5',
     '64ec67cc0a59bf57fd2cdf0ae53216a7683062dfea83949fbde18218220e0614'),
    ('e2_unit_1x_rho0p5', 'a895b40ec0b7ccdb_e2_unit_1x_rho0p5',
     'a668d74abc6f1737bc9aee970ff5b403d0c828be3a36bb58b1460230579ac4ab'),
    ('e2_x0_1x_rho2p0', 'f4df38b251a68808_e2_x0_1x_rho2p0',
     'b332511ca7a8b197d4889d6f3089ad5958ef5061b994fcea02fb6d0a341011dc'),
    ('e2_unit_1x_rho2p0', '13145d7ed63e9f2d_e2_unit_1x_rho2p0',
     'bada14e6dacfd860176e1c3b7a8119dfb641359732edaddd7ebc4f138e4e371e'),
    ('e2_x0_2x_rho0p5', '14a161e63a4ee34f_e2_x0_2x_rho0p5',
     '1c19b627c38644481e1c265c5c0c787de9718a13c099bc575186b310d389492d'),
    ('e2_unit_2x_rho0p5', '3e35c66844450f94_e2_unit_2x_rho0p5',
     '88b6b4741d7c07b3445dad0b561e161268415396c8bce7497ed80f5c853ea073'),
    ('e2_x0_2x_rho2p0', '16614c6680f4d3b4_e2_x0_2x_rho2p0',
     '191e3c7ecd03474c4640ee336fe1380ec39624d2305a89442c2149d8b3997bfd'),
    ('e2_unit_2x_rho2p0', '43074041a7942052_e2_unit_2x_rho2p0',
     'abfbcad0cfe11447cd1e6209c8a504e3debd2f10a3dbee01c2477aad0352a83d'),
)
CELLS = {}
for _c, _ev, _sha in CELL_TABLE:
    _root = os.path.join(E2, f'campaign_s55_step7_{_c}')
    _eval = os.path.join(_root, 'evals', _ev)
    CELLS[_c] = {'root': _root, 'eval': _eval, 'pickle': os.path.join(_eval, 'certified_models.pkl'),
                 'pickle_sha256': _sha, 'esso_pickle': os.path.join(_eval, 'esso_models_s39_D.pkl'),
                 'eval_key_prefix': _ev.split('_', 1)[0],
                 'plan': 'x0' if '_x0_' in _c else 'unit'}
ALLOW_LIST = [(v['pickle'], v['pickle_sha256']) for v in CELLS.values()]
assert len(ALLOW_LIST) == 8 and len({p for p, _ in ALLOW_LIST}) == 8 and len({s for _, s in ALLOW_LIST}) == 8


class PickleRefused(RuntimeError):
    pass


class AllowListPickleGate:
    """Global replacement for pickle.load / pickle.loads: an allow-list of (path, sha256); each path at most once."""

    def __init__(self, allowed, label, original_loads, log=None):
        self.allowed = {os.path.realpath(os.path.join(REPO, p)): s for p, s in allowed}
        self.label = label
        self._orig_loads = original_loads
        self._log = log
        self.counts = {'load_attempts': 0, 'loads_attempts': 0, 'unpickler_attempts': 0, 'refused': 0,
                       'delegated': 0}
        self.refusals = []
        self.loaded = []
        self._loaded_paths = set()

    def _refuse(self, kind, reason):
        self.counts['refused'] += 1
        self.refusals.append({'kind': kind, 'reason': reason})
        raise PickleRefused(f'{self.label}: {kind} refused -- {reason}')

    def load(self, file, *args, **kwargs):
        self.counts['load_attempts'] += 1
        name = getattr(file, 'name', None)
        if not isinstance(name, str):
            self._refuse('pickle.load', 'the handle has no file path (not on the allow-list)')
        path = os.path.realpath(name)
        if path not in self.allowed:
            self._refuse('pickle.load', f'{os.path.relpath(path, REPO)} is not on the allow-list')
        if path in self._loaded_paths:
            self._refuse('pickle.load', f'{os.path.relpath(path, REPO)} has already been loaded once in this run '
                                        f'(each allow-listed file at most once)')
        if file.tell() != 0:
            self._refuse('pickle.load', f'handle not at position 0 (tell {file.tell()})')
        data = file.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != self.allowed[path]:
            self._refuse('pickle.load', f'sha256 of the bytes read {digest} != allow-listed {self.allowed[path]}')
        self.counts['delegated'] += 1
        self._loaded_paths.add(path)
        entry = {'n': self.counts['delegated'], 'path': os.path.relpath(path, REPO),
                 'sha256_of_bytes_unpickled': digest, 'size_bytes': len(data)}
        self.loaded.append(entry)
        if self._log:
            self._log(f'{self.label}: delegated load #{entry["n"]} {entry["path"]} sha256 {digest} '
                      f'({len(data)} bytes)')
        return self._orig_loads(data, *args, **kwargs)

    def loads(self, *_a, **_k):
        self.counts['loads_attempts'] += 1
        self._refuse('pickle.loads', 'byte-string loads are not on the allow-list')


def _log(msg):
    from datetime import datetime as _dt, timezone as _tz
    print(f"[{_dt.now(_tz.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


_ORIGINAL_PICKLE = {'load': pickle.load, 'loads': pickle.loads, 'Unpickler': pickle.Unpickler}
PICKLE_GATE = AllowListPickleGate(ALLOW_LIST, 'W185 pickle gate', _ORIGINAL_PICKLE['loads'], log=_log)


class _BlockedUnpickler(pickle.Unpickler):
    def __init__(self, *_a, **_k):
        PICKLE_GATE.counts['unpickler_attempts'] += 1
        PICKLE_GATE._refuse('pickle.Unpickler', 'Unpickler is not on the allow-list')


pickle.load, pickle.loads, pickle.Unpickler = PICKLE_GATE.load, PICKLE_GATE.loads, _BlockedUnpickler

from pyomo.core.base.PyomoModel import Model as _PyomoModel  # noqa: E402

BUILD_COUNTS = {'pyomo_model_init': 0}


def _blocked_model_init(self, *_a, **_k):
    BUILD_COUNTS['pyomo_model_init'] += 1
    raise RuntimeError('W185: a Pyomo model build was attempted -- no model builds are permitted')


_PyomoModel.__init__ = _blocked_model_init

import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pyomo.environ as pe  # noqa: E402

import gate_result_io as GRIO  # noqa: E402  (the shared writer)
import definitions as DEFS  # noqa: E402  (constants only)

SCRIPT_REL = os.path.basename(__file__)
DEFAULT_OUT_DIR = os.path.join(P55, 'w185_e2_diagnostics')
SRP1_JSON = os.path.join('data', 'SRP1', 'SRP1.json')
CASE_DIRS = {'TSO': 'case9', 5: 'case33_1', 7: 'case33_2', 9: 'case33_3'}
SOC_MIN_FRAC, SOC_MAX_FRAC = DEFS.ENERGY_STORAGE_MIN_ENERGY_STORED, DEFS.ENERGY_STORAGE_MAX_ENERGY_STORED
PENALTY_SHARED_ESS_BALANCE = DEFS.PENALTY_SHARED_ESS_BALANCE
BOUND_RELAX_FACTOR_DEFAULT = 1e-8   # IPOPT 3.14 registered default (W180d (ii); not set anywhere in production)
CONSTR_VIOL_TOL_DEFAULT = 1e-4
DT_H = 1.0                          # 24 periods per representative day (W180d (ii))
RELAX_PU = min(CONSTR_VIOL_TOL_DEFAULT, BOUND_RELAX_FACTOR_DEFAULT * 1.0)   # at lb = 0
CLOSURE_LIMIT_MWH = 1e-6            # the task's per-slack limit = RELAX_PU x 100 MVA x 1 h
IDENTITY_FLAG_EUR = 1e-3
UNIT_CELL_NODE = 7
REC_FILES = ('component_levels_terminal.json', 'evaluation_record.json', 'boyd_terminal.json',
             'interface_settlement_detail_s31c.json', 'post_certification.json')
PROD_SOURCE_NAMES = {   # names this script reads, asserted present in production source
    'network.py': ['model.shared_es_pnet = pe.Var', 'model.shared_es_qnet = pe.Var', 'model.shared_es_soc = pe.Var',
                   'model.shared_es_s_rated_fixed = pe.Param', 'model.shared_es_e_rated_fixed = pe.Param',
                   'model.slack_shared_es_soc_final_up = pe.Var', 'model.slack_shared_es_soc_final_down = pe.Var',
                   'model.sess_converter_capability = pe.Constraint', 'model.sess_soc_limit_upper = pe.Constraint',
                   'model.sess_soc_limit_lower = pe.Constraint', 'model.sess_soc_final = pe.Constraint'],
    'model_construction_helpers.py': [
        'return (m.shared_es_pnet[e, s_m0, s_o0, p] ** 2',
        '+ m.shared_es_qnet[e, s_m0, s_o0, p] ** 2) <= m.shared_es_s_rated_fixed[e] ** 2',
        'soc_min = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_MIN_ENERGY_STORED',
        'soc_max = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_MAX_ENERGY_STORED',
        'total += base * PENALTY_SHARED_ESS_BALANCE * (model.slack_shared_es_soc_final_up[e, s_m0, s_o0] + '
        'model.slack_shared_es_soc_final_down[e, s_m0, s_o0])',
        'slack_ub = 0.0 if inactive else e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE'],
    'definitions.py': ['ENERGY_STORAGE_MAX_ENERGY_STORED = 0.90', 'ENERGY_STORAGE_MIN_ENERGY_STORED = 0.10',
                       'PENALTY_SHARED_ESS_BALANCE = 1e3'],
    'p515_s42_exact_fix_rerun.py': ["payload = {'tso': models['tso'], 'dso': models['dso']}"],
}

INPUTS = {}
_TRACKED = None


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, check=True, capture_output=True).stdout


def _status(path):
    global _TRACKED
    if _TRACKED is None:
        _TRACKED = set(_git(['ls-files', '-z']).decode('utf-8').split('\0'))
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel not in _TRACKED:
        return 'NOT TRACKED' + (' (gitignored)' if subprocess.run(['git', 'check-ignore', '-q', rel], cwd=REPO
                                                                  ).returncode == 0 else '')
    dirty = _git(['status', '--porcelain', '--', rel]).decode().strip()
    return 'committed' if not dirty else 'committed, MODIFIED in working tree'


def _reg(path, sha=None):
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel not in INPUTS:
        INPUTS[rel] = {'sha256': sha or _sha(path), 'status': _status(path), 'size_bytes': os.path.getsize(path)}
    return INPUTS[rel]


def _json(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _maxabs(values):
    values = [abs(v) for v in values if v is not None]
    return max(values) if values else None


def _val(x):
    v = pe.value(x, exception=False)
    return None if v is None else float(v)


def _by_str(mapping, key):
    hits = [k for k in mapping if str(k) == str(key)]
    if len(hits) != 1:
        raise KeyError(f'{key!r}: {len(hits)} matching keys in {sorted(map(str, mapping))}')
    return mapping[hits[0]]


def _sign(v):
    return None if v is None else (1 if v > 0 else (-1 if v < 0 else 0))


def _refuse(check, detail):
    _log(f'REFUSED (check: {check}): {detail}')
    raise SystemExit(f'REFUSED (check: {check}): {detail}')


def other_copies():
    r = subprocess.run(['ps', '-Ao', 'pid=,args='], capture_output=True, text=True).stdout.splitlines()
    me = os.getpid()
    out = []
    for ln in r:
        parts = ln.strip().split(None, 1)
        if len(parts) < 2 or int(parts[0]) == me:
            continue
        argv = parts[1].split()
        if argv and 'python' in os.path.basename(argv[0]) and any(a.endswith(SCRIPT_REL) for a in argv[1:3]):
            out.append(ln.strip())
    return out


# ============================================================================================ pre-run checks
def capture_checklist(cells):
    """Rule eleven: every capture path exists before any reading (fail fast)."""
    need = {}
    for c in cells:
        v = CELLS[c]
        for f in REC_FILES:
            need[f'{c} {f}'] = os.path.join(v['eval'], f)
        need[f'{c} campaign_manifest_sha256.json'] = os.path.join(v['root'], 'campaign_manifest_sha256.json')
        need[f'{c} certified_models.pkl'] = v['pickle']
        need[f'{c} esso_models_s39_D.pkl (negative control target, never read)'] = v['esso_pickle']
    need['SRP1.json'] = SRP1_JSON
    for k, d in CASE_DIRS.items():
        for y in ('2025', '2030', '2035'):
            need[f'case {k} {y}'] = os.path.join('data', 'SRP1', d, f'{d}_{y}.json')
    for f in PROD_SOURCE_NAMES:
        need[f'production source {f}'] = f
    missing = sorted(k for k, p in need.items() if not os.path.isfile(p))
    keys = {}
    if not missing:
        for f, names in PROD_SOURCE_NAMES.items():
            src = open(f, encoding='utf-8').read()
            for n in names:
                keys[f'{f} contains {n!r}'] = n in src
        for c in cells:
            ev = CELLS[c]['eval']
            cl = json.load(open(os.path.join(ev, 'component_levels_terminal.json')))
            keys[f'{c} component_levels 48 blocks with the closure slacks'] = (
                len(cl['blocks']) == 48 and all(
                    k in b['unweighted'] and k in b['weighted'] for b in cl['blocks'].values()
                    for k in ('shared_ess_day_balance_slack', 'local_ess_day_balance_slack',
                              'flexibility_p_day_balance_slack')))
            er = json.load(open(os.path.join(ev, 'evaluation_record.json')))
            keys[f'{c} evaluation_record eval key prefix = dir'] = er.get('eval_key', '').startswith(
                CELLS[c]['eval_key_prefix'])
            keys[f'{c} evaluation_record terminal_ratios_per_channel'] = set(
                er.get('terminal_ratios_per_channel') or {}) == {'v', 'pf', 'ess'}
            keys[f'{c} evaluation_record rule_ten'] = 'terminal_step_over_threshold' in (er.get('rule_ten') or {})
            keys[f'{c} evaluation_record storage_per_node node 7 published capacity'] = bool(
                ((er.get('storage_per_node') or {}).get('7') or {}).get('published_available_capacity_terminal'))
            keys[f'{c} evaluation_record recourse_components settlement'] = all(
                k in (er.get('recourse_components') or {}) for k in
                ('interface_settlement_tso', 'interface_settlement_dso', 'interface_settlement_total'))
            bt = json.load(open(os.path.join(ev, 'boyd_terminal.json')))
            keys[f'{c} boyd_terminal binding_test_per_channel'] = set(
                bt.get('binding_test_per_channel') or {}) == {'v', 'pf', 'ess'}
            isd = json.load(open(os.path.join(ev, 'interface_settlement_detail_s31c.json')))
            keys[f'{c} settlement detail consensus residual per dso'] = set(
                isd['interface_consensus_residual_per_dso']) == {'5', '7', '9'}
            keys[f'{c} settlement detail t_tso_plus_t_dso_terminal'] = 't_tso_plus_t_dso_terminal' in isd
            pc = json.load(open(os.path.join(ev, 'post_certification.json')))
            keys[f'{c} post_certification persisted_models sha256'] = bool(
                (pc.get('persisted_models') or {}).get('sha256'))
    failed = sorted(k for k, v in keys.items() if not v)
    return {'ok': not missing and not failed, 'n_paths': len(need), 'missing_paths': missing,
            'n_key_checks': len(keys), 'failed_key_checks': failed, 'paths': need}


def verify_cell_pins(c):
    """The committed campaign manifest (git show HEAD:) first; then working tree, post_certification, bytes."""
    v = CELLS[c]
    man_rel = os.path.join(v['root'], 'campaign_manifest_sha256.json')
    committed = json.loads(_git(['show', f'HEAD:{man_rel}']).decode('utf-8'))
    working = json.load(open(man_rel))
    _reg(man_rel)
    pc = json.load(open(os.path.join(v['eval'], 'post_certification.json')))
    actual = _sha(v['pickle'])
    pk = {'pin_in_script': v['pickle_sha256'], 'committed_campaign_manifest': committed.get(v['pickle']),
          'working_tree_campaign_manifest': working.get(v['pickle']),
          'post_certification': (pc.get('persisted_models') or {}).get('sha256'),
          'sha256_actual_before_load': actual, 'size_bytes': os.path.getsize(v['pickle']),
          'git_status': _status(v['pickle'])}
    pk['ok'] = (actual == v['pickle_sha256'] == pk['committed_campaign_manifest'] ==
                pk['working_tree_campaign_manifest'] == pk['post_certification'])
    rec = {}
    for f in REC_FILES:
        p = os.path.join(v['eval'], f)
        a = _sha(p)
        rec[f] = {'committed_manifest': committed.get(p), 'actual': a, 'status': _status(p),
                  'ok': a == committed.get(p) and _status(p) == 'committed'}
    return {'manifest': man_rel, 'manifest_status': _status(man_rel), 'committed_equals_working': committed == working,
            'pickle': pk, 'recorded': rec,
            'ok': pk['ok'] and committed == working and all(r['ok'] for r in rec.values())}


# ============================================================================================ part A (recorded)
def part_a(c, cl, er, bt, isd):
    # (1) closure slack, recorded
    per = {}
    for name, b in cl['blocks'].items():
        u = b['unweighted']['shared_ess_day_balance_slack']
        per[name] = {'penalty_eur_unweighted': u, 'S_mwh': u / PENALTY_SHARED_ESS_BALANCE,
                     'local_ess_day_balance_slack_eur': b['unweighted']['local_ess_day_balance_slack']}
    s = [v['S_mwh'] for v in per.values()]
    imax = max(per, key=lambda k: abs(per[k]['S_mwh']))
    closure = {'definition': ('S = s+ + s- [MWh] = unweighted shared_ess_day_balance_slack [EUR] / '
                              'PENALTY_SHARED_ESS_BALANCE (= base x 1e3 x (s+ + s-) p.u.; dt 1 h)'),
               'n_blocks': len(s), 'n_blocks_nonzero': sum(1 for x in s if x != 0.0),
               'S_mwh_min': min(s), 'S_mwh_max': max(s), 'S_mwh_max_abs': abs(per[imax]['S_mwh']),
               'S_mwh_max_abs_at': imax, 'sign_of_max_abs': _sign(per[imax]['S_mwh']),
               'n_blocks_positive': sum(1 for x in s if x > 0), 'n_blocks_negative': sum(1 for x in s if x < 0),
               'local_ess_day_balance_slack_max_abs_eur': _maxabs(v['local_ess_day_balance_slack_eur']
                                                                  for v in per.values()),
               'weighted_total_eur': cl['totals_weighted']['shared_ess_day_balance_slack'],
               'per_block': per}
    # (5) ratios
    resid = {}
    for node, pr in isd['interface_consensus_residual_per_dso'].items():
        vals = [v['residual_mw'] for v in pr['periods'].values()]
        worst = max(pr['periods'].items(), key=lambda kv: abs(kv[1]['residual_mw']))
        resid[node] = {'n_hours': len(vals), 'max_abs_residual_mw': _maxabs(vals), 'at': worst[0],
                       'sum_priced_residual_weighted_eur': sum(
                           v['priced_residual_pi_baseMVA_residual_weighted'] for v in pr['periods'].values())}
    trc = er['terminal_ratios_per_channel']
    boyd = {ch: {k: v.get(k) for k in ('r', 's', 'eps_pri', 'eps_dual', 'primal_ratio', 'dual_ratio', 'channel_pass')}
            for ch, v in bt['binding_test_per_channel'].items()}
    ratios = {'terminal_ratios_per_channel': trc,
              'max_ratio_per_channel': {ch: trc[ch]['max'] for ch in trc},
              'max_ratio_overall': max(trc[ch]['max'] for ch in trc),
              'max_ratio_channel': max(trc, key=lambda ch: trc[ch]['max']),
              'boyd_binding_test_per_channel': boyd,
              'all_channels_pass': all(v['channel_pass'] is True for v in boyd.values()),
              'first_pass_cycle_per_channel': er.get('first_pass_cycle_per_channel'),
              'rule_ten': er['rule_ten'],
              'interface_consensus_residual_per_dso': resid}
    # (6) settlement
    rc = er['recourse_components']
    sum_priced = sum(v['sum_priced_residual_weighted_eur'] for v in resid.values())
    t_dso = rc['interface_settlement_dso']
    settlement = {
        'T_TSO_eur': rc['interface_settlement_tso'], 'T_DSO_eur': t_dso,
        'total_eur': rc['interface_settlement_total'],
        'total_recomputed_eur': rc['interface_settlement_tso'] + sum(t_dso.values()),
        'total_recomputed_minus_recorded_eur': rc['interface_settlement_tso'] + sum(t_dso.values())
        - rc['interface_settlement_total'],
        't_tso_plus_t_dso_terminal_detail': isd['t_tso_plus_t_dso_terminal'],
        'identity': {'definition': 't_tso_plus_t_dso_terminal == -sum(priced residual weighted)',
                     'lhs': isd['t_tso_plus_t_dso_terminal'], 'minus_sum_priced_residual': -sum_priced,
                     'residual_eur': isd['t_tso_plus_t_dso_terminal'] + sum_priced,
                     'abs_residual_eur': abs(isd['t_tso_plus_t_dso_terminal'] + sum_priced)},
        'settlement_remainder': er.get('settlement_remainder'),
        'excluded_from_Q': True}
    return {'closure_recorded': closure, 'ratios': ratios, 'settlement': settlement,
            'esso_feasibility_violation_D3_recorded': cl.get('esso_feasibility_violation_D3')}


# ============================================================================================ part B (pickle)
def _ess_entry(model, e, base):
    sms, sos, periods = list(model.scenarios_market), list(model.scenarios_operation), list(model.periods)
    sm0, so0 = sms[0], sos[0]
    S = _val(model.shared_es_s_rated_fixed[e])
    E = _val(model.shared_es_e_rated_fixed[e])
    fixed = {n: {'fixed': 0, 'free': 0} for n in ('shared_es_pnet', 'shared_es_qnet', 'shared_es_soc',
                                                  'shared_es_pch', 'shared_es_pdch')}
    for sm in sms:
        for so in sos:
            for p in periods:
                for n in fixed:
                    fixed[n]['fixed' if getattr(model, n)[e, sm, so, p].fixed else 'free'] += 1
    cc, lo, up = model.sess_converter_capability, model.sess_soc_limit_lower, model.sess_soc_limit_upper
    circ_active, circ_all, body_up_active, pratio, aratio, pabs, qabs, socs = [], [], [], [], [], [], [], []
    soc_lo_margin, soc_up_margin = [], []
    n_cc, n_cc_active, n_soc, n_soc_active = 0, 0, 0, 0
    for p in periods:
        pn = _val(model.shared_es_pnet[e, sm0, so0, p])
        qn = _val(model.shared_es_qnet[e, sm0, so0, p])
        soc = _val(model.shared_es_soc[e, sm0, so0, p])
        pabs.append(abs(pn) * base)
        qabs.append(abs(qn) * base)
        socs.append(soc * base)
        r = (pn ** 2 + qn ** 2 - S ** 2) * base ** 2
        circ_all.append(r)
        key = (e, sm0, so0, p)
        if key in cc:
            n_cc += 1
            if cc[key].active:
                n_cc_active += 1
                circ_active.append((r, str(p)))
                body_up_active.append((_val(cc[key].body) - _val(cc[key].upper)) * base ** 2)
        if S > 0:
            pratio.append(abs(pn) / S)
            aratio.append(math.sqrt(pn ** 2 + qn ** 2) / S)
        for row, kind in ((lo, 'lo'), (up, 'up')):
            if key in row:
                n_soc += 1
                if row[key].active:
                    n_soc_active += 1
                    if kind == 'lo':
                        soc_lo_margin.append(((soc - SOC_MIN_FRAC * E) * base, str(p)))
                    else:
                        soc_up_margin.append(((SOC_MAX_FRAC * E - soc) * base, str(p)))
    # closure slacks and row
    slacks = {}
    for n in ('slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down'):
        var = getattr(model, n, None)
        if var is None:
            slacks[n] = {'present': False}
            continue
        vd = var[e, sm0, so0]
        val, lb, ub = _val(vd), vd.lb, vd.ub
        slacks[n] = {'present': True, 'value_mwh': val * base * DT_H, 'fixed': bool(vd.fixed),
                     'lb_mwh': None if lb is None else lb * base * DT_H,
                     'ub_mwh': None if ub is None else ub * base * DT_H,
                     'relaxed_floor_mwh': -RELAX_PU * base * DT_H if lb == 0.0 else None,
                     'below_relaxed_floor': (val < -RELAX_PU * (1 + 1e-9)) if lb == 0.0 else None,
                     'above_ub': (ub is not None and val > ub),
                     'abs_le_1e-6_mwh': abs(val * base * DT_H) <= CLOSURE_LIMIT_MWH * (1 + 1e-9)}
    fin = model.sess_soc_final
    fkey = (e, sm0, so0)
    fin_row = None
    if fkey in fin:
        row = fin[fkey]
        body = _val(row.body)
        lo_b, up_b = _val(row.lower), _val(row.upper)
        fin_row = {'active': bool(row.active), 'body_minus_lower_mwh': None if lo_b is None else (body - lo_b) * base,
                   'body_minus_upper_mwh': None if up_b is None else (body - up_b) * base}
    s_final = _val(model.shared_es_soc[e, sm0, so0, periods[-1]])
    imax_c = max(circ_active, key=lambda t: t[0]) if circ_active else None
    out = {
        's_rated_fixed_mva': S * base, 'e_rated_fixed_mwh': E * base, 'inactive_zero_capacity': not (S > 0 and E > 0),
        'fixed_vs_free_counts': fixed,
        # (2)
        'circle_rows_constructed': n_cc, 'circle_rows_active': n_cc_active,
        'circle_residual_mva2_max_active': imax_c[0] if imax_c else None,
        'circle_residual_at_period': imax_c[1] if imax_c else None,
        'circle_residual_sign_of_max_active': _sign(imax_c[0]) if imax_c else None,
        'circle_residual_mva2_min_active': min(t[0] for t in circ_active) if circ_active else None,
        'circle_n_active_positive': sum(1 for t in circ_active if t[0] > 0),
        'circle_row_body_minus_upper_mva2_max_active': max(body_up_active) if body_up_active else None,
        'circle_residual_mva2_max_all_rows': max(circ_all),
        'circle_residual_pu2_max_active': (imax_c[0] / base ** 2) if imax_c else None,
        'circle_residual_over_S2_max_active': (imax_c[0] / (S * base) ** 2) if imax_c and S > 0 else None,
        # (3)
        'pnet_abs_max_mw': max(pabs), 'qnet_abs_max_mvar': max(qabs),
        'pnet_over_s_max': max(pratio) if pratio else None,
        'apparent_over_s_max': max(aratio) if aratio else None,
        'pnet_over_s_note': None if pratio else 'S = 0: ratio undefined; max |pnet| reported instead',
        # (4)
        'soc_mwh_min': min(socs), 'soc_mwh_max': max(socs),
        'soc_rows_constructed': n_soc, 'soc_rows_active': n_soc_active,
        'soc_minus_10pct_E_min_mwh': min(t[0] for t in soc_lo_margin) if soc_lo_margin else None,
        'soc_minus_10pct_E_min_at_period': min(soc_lo_margin)[1] if soc_lo_margin else None,
        'soc_90pct_E_minus_soc_min_mwh': min(t[0] for t in soc_up_margin) if soc_up_margin else None,
        'soc_90pct_E_minus_soc_min_at_period': min(soc_up_margin)[1] if soc_up_margin else None,
        'soc_violations_strict': sum(1 for t in soc_lo_margin + soc_up_margin if t[0] < 0),
        'soc_violations_beyond_1e-6_mwh': sum(1 for t in soc_lo_margin + soc_up_margin if t[0] < -1e-6),
        'soc_inactive_rows_margins_all_periods': None if (soc_lo_margin or soc_up_margin) else {
            'soc_minus_10pct_E_min_mwh': min(sc - SOC_MIN_FRAC * E * base for sc in socs),
            'soc_90pct_E_minus_soc_min_mwh': min(SOC_MAX_FRAC * E * base - sc for sc in socs)},
        # (1)
        'closure_slacks': slacks,
        'closure_S_mwh': sum(v['value_mwh'] for v in slacks.values() if v.get('present')),
        'soc_final_mwh': s_final * base, 'soc_final_minus_half_E_mwh': (s_final - 0.5 * E) * base,
        'sess_soc_final_row': fin_row,
    }
    return out


def part_b(models, srp1, base, er):
    years, days = list(srp1['Years']), list(srp1['Days'])
    dn_order = [dn['connection_node_id'] for dn in srp1['DistributionNetworks']]
    blocks = {}
    for y in years:
        for d in days:
            tm = _by_str(_by_str(models['tso'], y), d)
            blocks[f'TSO|{y}|{d}'] = {'base_mva': base[('TSO', y)], 'year': y, 'net': 'TSO',
                                      'entries': {str(e): _ess_entry(tm, e, base[('TSO', y)])
                                                  for e in tm.shared_energy_storages},
                                      'own_ess': len(list(tm.energy_storages)) if hasattr(tm, 'energy_storages')
                                      else None}
            for node in dn_order:
                dm = _by_str(_by_str(_by_str(models['dso'], node), y), d)
                blocks[f'DSO{node}|{y}|{d}'] = {'base_mva': base[(node, y)], 'year': y, 'net': f'DSO{node}',
                                                'entries': {str(e): _ess_entry(dm, e, base[(node, y)])
                                                            for e in dm.shared_energy_storages},
                                                'own_ess': len(list(dm.energy_storages))
                                                if hasattr(dm, 'energy_storages') else None}
    flat = [(b, e, v) for b, blk in blocks.items() for e, v in blk['entries'].items()]
    act = [(b, e, v) for b, e, v in flat if not v['inactive_zero_capacity']]
    # capacity vs the record (node 7 published available capacity per year)
    pub = ((er.get('storage_per_node') or {}).get(str(UNIT_CELL_NODE)) or {}).get(
        'published_available_capacity_terminal') or {}
    cap_cmp = []
    for b, e, v in act:
        y = blocks[b]['year']
        p = pub.get(str(y)) or {}
        cap_cmp.append({'block': b, 'entry': e, 's_model_mva': v['s_rated_fixed_mva'],
                        'e_model_mwh': v['e_rated_fixed_mwh'], 's_available_record': p.get('s_available'),
                        'e_available_record': p.get('e_available'),
                        'abs_diff_s': None if p.get('s_available') is None else abs(v['s_rated_fixed_mva']
                                                                                    - p['s_available']),
                        'abs_diff_e': None if p.get('e_available') is None else abs(v['e_rated_fixed_mwh']
                                                                                    - p['e_available'])})
    nets_with_storage = sorted({blocks[b]['net'] for b, _e, _v in act})

    def _mx(key, src):
        vals = [(v[key], f'{b} e{e}') for b, e, v in src if v[key] is not None]
        return max(vals) if vals else (None, None)

    def _mn(key, src):
        vals = [(v[key], f'{b} e{e}') for b, e, v in src if v[key] is not None]
        return min(vals) if vals else (None, None)
    circ = _mx('circle_residual_mva2_max_active', act)
    pr = _mx('pnet_over_s_max', act)
    ar = _mx('apparent_over_s_max', act)
    slo = _mn('soc_minus_10pct_E_min_mwh', act)
    sup = _mn('soc_90pct_E_minus_soc_min_mwh', act)
    slack_vals = [(s['value_mwh'], f'{b} e{e} {n}') for b, e, v in flat
                  for n, s in v['closure_slacks'].items() if s.get('present')]
    slack_vals_act = [(s['value_mwh'], f'{b} e{e} {n}') for b, e, v in act
                      for n, s in v['closure_slacks'].items() if s.get('present')]
    smax_abs = max(slack_vals_act, key=lambda t: abs(t[0])) if slack_vals_act else (None, None)
    by_net = {}
    for net in nets_with_storage:
        sub = [(b, e, v) for b, e, v in act if blocks[b]['net'] == net]
        cm = _mx('circle_residual_mva2_max_active', sub)
        by_net[net] = {
            'n_entries': len(sub),
            'circle_residual_mva2_max_active': cm[0], 'circle_at': cm[1],
            'circle_residual_pu2_max_active': _mx('circle_residual_pu2_max_active', sub)[0],
            'circle_residual_over_S2_max_active': _mx('circle_residual_over_S2_max_active', sub)[0],
            'circle_n_active_positive': sum(v['circle_n_active_positive'] for _b, _e, v in sub),
            'circle_rows_active': sum(v['circle_rows_active'] for _b, _e, v in sub),
            'pnet_over_s_max': _mx('pnet_over_s_max', sub)[0], 'apparent_over_s_max': _mx('apparent_over_s_max', sub)[0],
            'soc_minus_10pct_E_min_mwh': _mn('soc_minus_10pct_E_min_mwh', sub)[0],
            'soc_90pct_E_minus_soc_min_mwh': _mn('soc_90pct_E_minus_soc_min_mwh', sub)[0],
            'soc_violations_strict': sum(v['soc_violations_strict'] for _b, _e, v in sub),
            'soc_violations_beyond_1e-6_mwh': sum(v['soc_violations_beyond_1e-6_mwh'] for _b, _e, v in sub),
            'soc_rows_active': sum(v['soc_rows_active'] for _b, _e, v in sub),
            'closure_slack_mwh_min': min((s_['value_mwh'] for _b, _e, v in sub for s_ in v['closure_slacks'].values()
                                          if s_.get('present')), default=None),
            'closure_slack_mwh_max': max((s_['value_mwh'] for _b, _e, v in sub for s_ in v['closure_slacks'].values()
                                          if s_.get('present')), default=None)}
    summary = {
        'n_block_models': len(blocks), 'n_shared_ess_entries': len(flat),
        'by_network_with_storage': by_net,
        'n_entries_with_storage': len(act), 'networks_with_storage': nets_with_storage,
        'n_entries_zero_capacity': len(flat) - len(act),
        'own_ess_entries_total': sum(blk['own_ess'] or 0 for blk in blocks.values()),
        'capacity_vs_record': {'n': len(cap_cmp),
                               'max_abs_diff_s_mva': _maxabs(x['abs_diff_s'] for x in cap_cmp),
                               'max_abs_diff_e_mwh': _maxabs(x['abs_diff_e'] for x in cap_cmp),
                               'per_entry': cap_cmp},
        # (1)
        'closure': {
            'n_slack_vars': len(slack_vals), 'n_slack_vars_storage_entries': len(slack_vals_act),
            'slack_mwh_min_storage': min(slack_vals_act)[0] if slack_vals_act else None,
            'slack_mwh_max_storage': max(slack_vals_act)[0] if slack_vals_act else None,
            'slack_mwh_max_abs_storage': abs(smax_abs[0]) if smax_abs[0] is not None else None,
            'slack_mwh_max_abs_at': smax_abs[1], 'sign_of_max_abs': _sign(smax_abs[0]),
            'n_below_relaxed_floor': sum(1 for _b, _e, v in flat for s in v['closure_slacks'].values()
                                         if s.get('present') and s['below_relaxed_floor']),
            'n_above_ub': sum(1 for _b, _e, v in flat for s in v['closure_slacks'].values()
                              if s.get('present') and s['above_ub']),
            'n_abs_gt_1e-6_mwh': sum(1 for _b, _e, v in flat for s in v['closure_slacks'].values()
                                     if s.get('present') and not s['abs_le_1e-6_mwh']),
            'ub_mwh_storage_min': min((s['ub_mwh'] for _b, _e, v in act for s in v['closure_slacks'].values()
                                       if s.get('present')), default=None),
            'zero_capacity_slacks_all_exactly_0_fixed_bounds_0_0': all(
                s['value_mwh'] == 0.0 and s['fixed'] and s['lb_mwh'] == 0.0 and s['ub_mwh'] == 0.0
                for _b, _e, v in flat if v['inactive_zero_capacity'] for s in v['closure_slacks'].values()
                if s.get('present')),
            'soc_final_row_max_abs_residual_mwh_active': _maxabs(
                x for _b, _e, v in act if v['sess_soc_final_row'] and v['sess_soc_final_row']['active']
                for x in (v['sess_soc_final_row']['body_minus_lower_mwh'],)),
            'soc_final_minus_half_E_max_abs_mwh': _maxabs(v['soc_final_minus_half_E_mwh'] for _b, _e, v in act),
        },
        # (2)
        'circle': {'rows_constructed': sum(v['circle_rows_constructed'] for _b, _e, v in flat),
                   'rows_active': sum(v['circle_rows_active'] for _b, _e, v in flat),
                   'residual_mva2_max_active': circ[0], 'at': circ[1], 'sign_of_max': _sign(circ[0]),
                   'residual_pu2_max_active': _mx('circle_residual_pu2_max_active', act)[0],
                   'residual_over_S2_max_active': _mx('circle_residual_over_S2_max_active', act)[0],
                   'n_active_positive': sum(v['circle_n_active_positive'] for _b, _e, v in act),
                   'row_body_minus_upper_mva2_max_active': _mx('circle_row_body_minus_upper_mva2_max_active', act)[0],
                   'zero_capacity_residual_mva2_max_all_rows': max(
                       (v['circle_residual_mva2_max_all_rows'] for _b, _e, v in flat if v['inactive_zero_capacity']),
                       default=None)},
        # (3)
        'pe_over_sav': {'pnet_over_s_max': pr[0], 'at': pr[1], 'apparent_over_s_max': ar[0], 'apparent_at': ar[1],
                        'pnet_abs_max_mw_all': max(v['pnet_abs_max_mw'] for _b, _e, v in flat),
                        'qnet_abs_max_mvar_all': max(v['qnet_abs_max_mvar'] for _b, _e, v in flat),
                        'note': None if act else 'no entry with S > 0: ratio undefined; max |pnet| reported'},
        # (4)
        'soc': {'rows_constructed': sum(v['soc_rows_constructed'] for _b, _e, v in flat),
                'rows_active': sum(v['soc_rows_active'] for _b, _e, v in flat),
                'soc_minus_10pct_E_min_mwh': slo[0], 'lower_at': slo[1],
                'soc_90pct_E_minus_soc_min_mwh': sup[0], 'upper_at': sup[1],
                'violations_strict': sum(v['soc_violations_strict'] for _b, _e, v in act),
                'violations_beyond_1e-6_mwh': sum(v['soc_violations_beyond_1e-6_mwh'] for _b, _e, v in act),
                'soc_mwh_abs_max_zero_capacity': max(
                    (max(abs(v['soc_mwh_min']), abs(v['soc_mwh_max'])) for _b, _e, v in flat
                     if v['inactive_zero_capacity']), default=None)},
        'fixed_vs_free_totals': {n: {k: sum(v['fixed_vs_free_counts'][n][k] for _b, _e, v in flat)
                                     for k in ('fixed', 'free')}
                                 for n in ('shared_es_pnet', 'shared_es_qnet', 'shared_es_soc', 'shared_es_pch',
                                           'shared_es_pdch')},
    }
    return {'summary': summary, 'per_block': blocks}


def _nc(fn, label, gate=None):
    gate = gate or PICKLE_GATE
    before = gate.counts['delegated']
    try:
        fn()
    except PickleRefused as exc:
        return {'label': label, 'ok': gate.counts['delegated'] == before, 'refusal': str(exc)}
    return {'label': label, 'ok': False, 'refusal': None}


def _load_twice_nc(path):
    def f():
        with open(path, 'rb') as fh:
            pickle.load(fh)
    return f


def flags_for(c, a, b):
    """Every value outside its limit (empty list = none)."""
    fl = []
    cr = a['closure_recorded']
    if cr['S_mwh_max_abs'] > 2 * CLOSURE_LIMIT_MWH * (1 + 1e-9):
        fl.append(f"(1) recorded closure |S| {cr['S_mwh_max_abs']:.3e} MWh > 2 x 1e-6 (two slacks) at "
                  f"{cr['S_mwh_max_abs_at']}")
    if b:
        s = b['summary']
        cl = s['closure']
        for k in ('n_below_relaxed_floor', 'n_above_ub', 'n_abs_gt_1e-6_mwh'):
            if cl[k]:
                fl.append(f'(1) closure slack {k} = {cl[k]}')
        if s['circle']['n_active_positive']:
            fl.append(f"(2) converter circle positive on {s['circle']['n_active_positive']} of "
                      f"{s['circle']['rows_active']} active rows; max {s['circle']['residual_mva2_max_active']:.3e} "
                      f"MVA^2 ({s['circle']['residual_pu2_max_active']:.3e} p.u.^2; "
                      f"{s['circle']['residual_over_S2_max_active']:.3e} of S^2) at {s['circle']['at']}")
        if s['pe_over_sav']['pnet_over_s_max'] is not None and s['pe_over_sav']['pnet_over_s_max'] > 1.0:
            fl.append(f"(3) max |P^E|/S^Av {s['pe_over_sav']['pnet_over_s_max']} > 1")
        if s['soc']['violations_strict']:
            fl.append(f"(4) SOC limit violated on {s['soc']['violations_strict']} active rows (strict; "
                      f"{s['soc']['violations_beyond_1e-6_mwh']} beyond 1e-6 MWh); min margins "
                      f"{s['soc']['soc_minus_10pct_E_min_mwh']} / {s['soc']['soc_90pct_E_minus_soc_min_mwh']} MWh")
    r = a['ratios']
    if r['max_ratio_overall'] > 1.0 or not r['all_channels_pass']:
        fl.append(f"(5) terminal ratio max {r['max_ratio_overall']} on {r['max_ratio_channel']}; channels pass "
                  f"{r['all_channels_pass']}")
    st = a['settlement']
    if st['identity']['abs_residual_eur'] > IDENTITY_FLAG_EUR:
        fl.append(f"(6) settlement identity residual {st['identity']['abs_residual_eur']} EUR > {IDENTITY_FLAG_EUR}")
    if abs(st['total_recomputed_minus_recorded_eur']) > IDENTITY_FLAG_EUR:
        fl.append(f"(6) T_TSO + sum T_DSO - recorded total = {st['total_recomputed_minus_recorded_eur']} EUR")
    return fl


# ============================================================================================ main
def main():
    args = sys.argv[1:]
    no_pickle = '--no-pickle' in args
    out_dir = DEFAULT_OUT_DIR
    if '--out-dir' in args:
        out_dir = args[args.index('--out-dir') + 1]
    cells = list(CELLS)
    if '--cells' in args:
        cells = args[args.index('--cells') + 1].split(',')
        bad = [c for c in cells if c not in CELLS]
        if bad:
            _refuse('cells known', bad)
    trial = no_pickle or cells != list(CELLS) or os.path.abspath(out_dir) != os.path.abspath(DEFAULT_OUT_DIR)
    if (no_pickle or cells != list(CELLS)) and os.path.abspath(out_dir) == os.path.abspath(DEFAULT_OUT_DIR):
        _refuse('trial options only with --out-dir', 'the real output directory takes the full run only')
    out_json = os.path.join(out_dir, 'w185_e2_diagnostics.json')
    out_man = os.path.join(out_dir, 'manifest_sha256.json')
    out_post = os.path.join(out_dir, 'manifest_post_run_sha256.json')
    if '--post-run' in args:
        rels = [out_json, out_man, os.path.join(out_dir, 'launch.log')]
        for r in rels:
            if not os.path.exists(r):
                raise SystemExit(f'post-run: {r} missing')
        if os.path.exists(out_post):
            raise SystemExit(f'post-run: {out_post} exists (write-once)')
        man = {r: _sha(r) for r in rels}
        with open(out_post, 'x', encoding='utf-8') as h:
            GRIO.dump(man, h, indent=1, sort_keys=True)
            h.write('\n')
        print('post-run manifest:', json.dumps(man, indent=1))
        return 0
    started = datetime.now(timezone.utc).isoformat()
    if not os.path.isdir(out_dir):
        _refuse('output dir exists', f'{out_dir} does not exist')
    extra = sorted(set(os.listdir(out_dir)) - {'launch.log'})
    if extra:
        _refuse('output dir fresh', f'{out_dir} holds {extra}')
    oc = other_copies()
    if oc:
        _refuse('no concurrent copy', oc)
    head = _git(['rev-parse', 'HEAD']).decode().strip()
    dirty = _git(['status', '--porcelain', '--untracked-files=no']).decode().splitlines()
    script_status = _status(SCRIPT_REL)
    _log(f'repo HEAD {head}; tracked changes: {dirty}; script {script_status}; trial {trial}; no_pickle {no_pickle}; '
         f'cells {cells}; out {out_dir}')

    chk = capture_checklist(cells)
    if not chk['ok']:
        _refuse('capture-path checklist', f"missing {chk['missing_paths']}; failed {chk['failed_key_checks']}")
    _log(f"capture-path checklist: {chk['n_paths']} paths, {chk['n_key_checks']} key checks, all present")
    pins = {}
    for c in cells:
        pins[c] = verify_cell_pins(c)
        if not pins[c]['ok']:
            _refuse(f'{c}: pickle sha256 = pin = committed manifest = working manifest = post_certification; '
                    f'recorded inputs = committed manifest', json.dumps(pins[c]))
        _log(f"{c}: pins verified; pickle sha256 before load {pins[c]['pickle']['sha256_actual_before_load']} "
             f"(= committed campaign manifest = post_certification); {len(REC_FILES)} recorded inputs = manifest")

    srp1 = _json(SRP1_JSON)
    base = {}
    for k, dname in CASE_DIRS.items():
        for y in srp1['Years']:
            base[(k, y)] = float(_json(os.path.join('data', 'SRP1', dname, f'{dname}_{y}.json'))['baseMVA'])

    # negative controls before any load
    first = list(CELLS)[0]
    ncs = {
        'NC_loads_bytes': _nc(lambda: pickle.loads(b'\x80\x04N.'), 'pickle.loads(bytes)'),
        'NC_load_in_memory': _nc(lambda: pickle.load(io.BytesIO(b'\x80\x04N.')), 'pickle.load(BytesIO)'),
        'NC_unpickler': _nc(lambda: pickle.Unpickler(io.BytesIO(b'\x80\x04N.')), 'pickle.Unpickler(...)'),
        'NC_non_listed': _nc(_load_twice_nc(CELLS[first]['esso_pickle']),
                             f"pickle.load({CELLS[first]['esso_pickle']}) (not on the allow-list)"),
    }
    wrong_gate = AllowListPickleGate([(CELLS[first]['pickle'], '0' * 64)], 'W185 NC wrong-sha256 gate',
                                     _ORIGINAL_PICKLE['loads'])

    def _wrong():
        with open(CELLS[first]['pickle'], 'rb') as fh:
            wrong_gate.load(fh)
    ncs['NC_wrong_sha256'] = _nc(_wrong, f"{CELLS[first]['pickle']} against a wrong allow-listed sha256 (gate of the "
                                         f"same class)", gate=wrong_gate)
    ncs['NC_wrong_sha256']['ok'] = (ncs['NC_wrong_sha256']['ok'] and wrong_gate.counts['delegated'] == 0
                                    and wrong_gate.counts['refused'] == 1 and 'sha256 of the bytes read' in
                                    (ncs['NC_wrong_sha256']['refusal'] or ''))
    ncs['NC_non_listed']['ok'] = ncs['NC_non_listed']['ok'] and 'not on the allow-list' in (
        ncs['NC_non_listed']['refusal'] or '')
    _log(f"negative controls before the loads: {[(k, v['ok']) for k, v in ncs.items()]}")
    if not all(v['ok'] for v in ncs.values()):
        _refuse('negative controls before the loads', json.dumps(ncs))

    per_cell = {}
    modules_before = set(sys.modules)
    for c in cells:
        v = CELLS[c]
        ev = v['eval']
        cl = _json(os.path.join(ev, 'component_levels_terminal.json'))
        er = _json(os.path.join(ev, 'evaluation_record.json'))
        bt = _json(os.path.join(ev, 'boyd_terminal.json'))
        isd = _json(os.path.join(ev, 'interface_settlement_detail_s31c.json'))
        _reg(os.path.join(ev, 'post_certification.json'))
        cand = er['candidate_canonical']['nodes']
        is_x0 = all(list(x) == [0.0, 0.0] for x in cand.values())
        if is_x0 != (v['plan'] == 'x0') or not er['eval_key'].startswith(v['eval_key_prefix']):
            _refuse('instance identity', [c, er['eval_key'], cand])
        a = part_a(c, cl, er, bt, isd)
        b = None
        if not no_pickle:
            _log(f"GUARDED READ {c}: loading {v['pickle']} sha256 {v['pickle_sha256']} (declared load "
                 f"{PICKLE_GATE.counts['delegated'] + 1} of {len(cells)} of this run, HEAD {head}, out {out_dir})")
            with open(v['pickle'], 'rb') as fh:
                models = pickle.load(fh)
            _log(f'{c}: loaded; payload keys {sorted(models)}')
            b = part_b(models, srp1, base, er)
            del models
            gc.collect()
            ncs[f'NC_second_load[{c}]'] = _nc(_load_twice_nc(v['pickle']), f'second pickle.load of {v["pickle"]}')
            _log(f"{c}: NC second load refused {ncs[f'NC_second_load[{c}]']['ok']}")
        fl = flags_for(c, a, b)
        notes = []
        if b:
            cvr = b['summary']['capacity_vs_record']
            if (cvr['max_abs_diff_e_mwh'] or 0) > 0 or (cvr['max_abs_diff_s_mva'] or 0) > 0:
                notes.append(f"capacity held by the models differs from the record's published_available_capacity_"
                             f"terminal: max |dE| {cvr['max_abs_diff_e_mwh']} MWh, max |dS| "
                             f"{cvr['max_abs_diff_s_mva']} MVA (not a limit; reported)")
        per_cell[c] = {
            'instance': {'eval_dir': ev, 'eval_key': er['eval_key'], 'candidate_key': er['candidate_key'],
                         'candidate_canonical': er['candidate_canonical'], 'plan': v['plan'],
                         'status': er['status'], 'certified_cost_Q': er['certified_cost'],
                         'certification_cycle': er.get('certification_cycle'), 'cycles_run': er.get('cycles_run'),
                         'rho0_scale': er.get('rho0_scale'), 'certified_models_sha256': v['pickle_sha256']},
            'A_recorded': a, 'B_pickle': b,
            'esso_sigma_slack': {'status': 'not applicable',
                                 'reason': ('no quantity of that name exists in the code (W183i (d) Blocked 2); the '
                                            'ESSO models are not in certified_models.pkl and esso_models_s39_D.pkl '
                                            'is not on this allow-list')},
            'flags': fl, 'notes': notes}
        _log(f"{c}: flags {fl if fl else 'none'}; notes {notes if notes else 'none'}")
    project_modules_by_unpickling = sorted(
        m for m in set(sys.modules) - modules_before
        if getattr(sys.modules[m], '__file__', None)
        and os.path.dirname(os.path.abspath(sys.modules[m].__file__)) == REPO)

    gfail = GUARD.verify(0)
    expected_loads = 0 if no_pickle else len(cells)
    expected_refusals = 4 + (0 if no_pickle else len(cells))
    pickle_ok = (PICKLE_GATE.counts['delegated'] == expected_loads
                 and PICKLE_GATE.counts['refused'] == expected_refusals
                 and PICKLE_GATE.counts['unpickler_attempts'] == 1
                 and pickle.load == PICKLE_GATE.load and pickle.loads == PICKLE_GATE.loads
                 and pickle.Unpickler is _BlockedUnpickler)
    build_ok = BUILD_COUNTS['pyomo_model_init'] == 0
    ncs_ok = all(v['ok'] for v in ncs.values())
    guards = {
        'solve_profile_guard': {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_0_failures': gfail},
        'pickle_gate': {'allow_list': [{'path': p, 'sha256': s} for p, s in ALLOW_LIST],
                        'counts': dict(PICKLE_GATE.counts), 'expected_delegated_loads': expected_loads,
                        'expected_refusals': expected_refusals, 'refusals': PICKLE_GATE.refusals,
                        'loaded': PICKLE_GATE.loaded, 'ok': pickle_ok},
        'nc_wrong_sha256_gate': {'counts': dict(wrong_gate.counts), 'refusals': wrong_gate.refusals},
        'build_guard': {'blocked': 'pyomo.core.base.PyomoModel.Model.__init__', 'counts': dict(BUILD_COUNTS),
                        'ok': build_ok},
        'negative_controls': ncs, 'negative_controls_ok': ncs_ok,
        'project_modules_imported_by_unpickling': project_modules_by_unpickling,
    }
    if gfail or not pickle_ok or not build_ok or not ncs_ok:
        _log(f'GUARD FAULT: {json.dumps(guards, default=str)}')
        raise SystemExit('guard fault')

    result = {
        'schema': 'p515_s55_w185_e2_diagnostics_v1',
        'stage': ('P5.15 W185 (E2): per-cell plan-section-4 diagnostics of the eight E2 cells -- recorded outputs and '
                  'one guarded read of each run\'s certified_models.pkl (zero solves, no build)'),
        'trial': trial, 'no_pickle': no_pickle, 'cells': cells, 'started_utc': started, 'git_HEAD': head,
        'tracked_changes_at_start': dirty,
        'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'git_status': script_status},
        'interpreter': sys.executable,
        'objective_convention': ('Q = certified gross_operational_cost: gross operational cost, contracted interface '
                                 'settlement excluded (Addendum 83 ruling 2); quoted for instance identity only'),
        'limits': {'closure_per_slack_mwh': CLOSURE_LIMIT_MWH,
                   'closure_relaxed_floor_definition': ('IPOPT bound_relax_factor 1e-8 (default, unset) x '
                                                        'max(|lb|, 1) at lb 0, capped by constr_viol_tol 1e-4: '
                                                        '1e-8 p.u. x baseMVA x 1 h'),
                   'closure_ub_definition': 'e_capacity x ESS_DAY_BALANCE_SLACK_FRACTION (0.05) + 1e-5 p.u.',
                   'circle': 'pnet^2 + qnet^2 - S^2 <= 0 on active rows', 'pe_over_sav': '<= 1',
                   'soc': f'[{SOC_MIN_FRAC} E^Av, {SOC_MAX_FRAC} E^Av]',
                   'ratios': 'terminal ratio <= 1 per channel (channel_pass)',
                   'settlement_identity_flag_eur': IDENTITY_FLAG_EUR},
        'pre_run_checks': {'capture_checklist': {k: v for k, v in chk.items() if k != 'paths'}, 'pins': pins},
        'base_mva': {f'{k[0]}|{k[1]}': v for k, v in base.items()},
        'per_cell': per_cell,
        'flags_all_cells': {c: per_cell[c]['flags'] for c in cells},
        'notes_all_cells': {c: per_cell[c]['notes'] for c in cells},
        'guards': guards,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    with open(out_json, 'x', encoding='utf-8') as fh:
        GRIO.dump(result, fh, indent=1, sort_keys=False)
        fh.write('\n')
    manifest = {'outputs': {out_json: _sha(out_json)},
                'script': {SCRIPT_REL: {'sha256': _sha(SCRIPT_REL), 'git_status': script_status}},
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)}, 'n_inputs': len(INPUTS)}
    with open(out_man, 'x', encoding='utf-8') as fh:
        GRIO.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')
    summ = {}
    for c in cells:
        pc = per_cell[c]
        a, b = pc['A_recorded'], pc['B_pickle']
        summ[c] = {'Q': pc['instance']['certified_cost_Q'], 'k*': pc['instance']['certification_cycle'],
                   'closure_rec_S_mwh_minmax': [a['closure_recorded']['S_mwh_min'], a['closure_recorded']['S_mwh_max']],
                   'ratio_max': a['ratios']['max_ratio_per_channel'],
                   'consensus_max_mw': {n: r['max_abs_residual_mw']
                                        for n, r in a['ratios']['interface_consensus_residual_per_dso'].items()},
                   'settlement_total': a['settlement']['total_eur'],
                   'identity_residual': a['settlement']['identity']['residual_eur'], 'flags': pc['flags']}
        if b:
            s = b['summary']
            summ[c].update({'closure_slack': {k: s['closure'][k] for k in
                                              ('slack_mwh_min_storage', 'slack_mwh_max_storage', 'n_below_relaxed_floor',
                                               'n_abs_gt_1e-6_mwh', 'n_above_ub')},
                            'circle': {k: s['circle'][k] for k in ('rows_active', 'residual_mva2_max_active',
                                                                   'residual_over_S2_max_active',
                                                                   'sign_of_max', 'n_active_positive')},
                            'by_network': s['by_network_with_storage'],
                            'pe_over_sav': s['pe_over_sav']['pnet_over_s_max'],
                            'soc': {k: s['soc'][k] for k in ('rows_active', 'soc_minus_10pct_E_min_mwh',
                                                             'soc_90pct_E_minus_soc_min_mwh', 'violations_strict')}})
    print('=== per-cell summary', json.dumps(summ, indent=1, default=str))
    print('=== guards', json.dumps({k: guards[k] for k in ('solve_profile_guard', 'pickle_gate', 'build_guard')},
                                   indent=1, default=str))
    _log(f'wrote {out_json} sha256 {_sha(out_json)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
