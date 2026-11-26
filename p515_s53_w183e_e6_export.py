"""
P5.15 Addendum 83, ruling 4 and W183 order item W183e -- the E6 data export (STEP7_SIMULATION_PLAN_2026-10-25.md
section 7, amended in section 11). ZERO SOLVES, NO MODEL BUILD, EXACTLY ONE PICKLE LOAD (one declared file).

SOURCE CELLS (E.21 rows 1 and 2; W101 continuations; single scenario, 1 market x 1 operation)
  row 1  `ref:7aa017f0`  x = 0                          eval d110bd1a5977df1e_x0        k* 181
  row 2  `ref:bd504ecf`  node 7 0.25 MVA / 1 MWh, 2025  eval 3f084f2ffaeef2b7_n7_4h_e1  k* 172 (the reference unit)
  Row identity is checked against W180b's committed row map (frozen eval keys).

OBJECTIVE CONVENTION. Q = certified `gross_operational_cost`: gross operational cost, contracted interface settlement
excluded (Addendum 83 ruling 2). At one market scenario the deviation settlement and the commitment premium are
identically 0. No salvage credit is involved (terminal_salvage_value = 0 on both rows).

DECLARED DAY-SELECTION RULE (plan section 7): the (year, day) block with the largest value contribution
Q(0) - Q(x) in the certified decomposition, labelled an ILLUSTRATIVE day, not a typical-day estimate.
  Per-block Q (one network block): the sum over the components that make up `gross_operational_cost` in
  `component_levels_terminal.json` (`blocks.<b>.weighted`): generation_cost + flexibility_cost_internal +
  load_curtailment_cost + res_curtailment_penalty + ess_usage_cost + detector_penalty_total. "weighted" = unweighted x
  `_get_admm_block_weight` (year count x day count x discount annualisation), the weight `get_primal_value` uses.
  The (year, day) value = the sum over the four network blocks (TSO + DSO 5, 7, 9) of that (year, day).
  Identity asserted per row: sum over the 48 blocks == recourse_components.gross_operational_cost (residual reported).
  Value contribution c(y, d) = Q_row1(y, d) - Q_row2(y, d)  (EUR, block-weighted, discounted).
  Ranking: all 12 blocks, by c descending; the declared day is rank 1. The unweighted (one representative day)
  contribution is reported beside it, not used.
  Resolution of the ranking (CLAUDE.md "Report a difference with its resolution", local measure only): for each row,
  the range (max - min) of the (year, day) block value over that row's certification window (settling_decision
  `window`) from `recourse_blocks_all.jsonl`; bar(c) = range_row1 + range_row2; the rank-1 vs rank-2 margin is
  "not explained by stopping slack" iff margin > bar(rank 1) + bar(rank 2). Terminal steps are reported beside.

EXPORT CONTENTS (no figure)
  (a) declared day: hourly ESS charge / discharge / net active power and reactive power for every copy available --
      ESSO (committed `esso_capture/s39_D/node7_cycle172.jsonl`: pch, pdch, pnet; stride x.esso: p, q), the TSO and DSO
      network copies (pickle: shared_es_pch / pdch / pnet / qnet / soc, expected_shared_ess_p / q; stride x.tso / x.dso),
      the consensus z (stride) -- SoC of the TSO and DSO copies from the pickle (the only SoC variable: the ESSO model
      has no intra-day SoC), and the 10 % / 90 % limits of E^Av (record value, and the pickle's
      shared_es_e_rated_fixed x baseMVA). Derived SoC (labelled DERIVED, not model variables): recursion from the ESSO
      pch / pdch, and from the consensus z net power. Market price per hour (row 2 committed settlement detail) as
      context.
  (b) per (year, day) block: change against x = 0 (row 2 minus row 1) of the TN conventional cost (TSO
      generation_cost) and of the DN flexibility activation cost (DSO flexibility_cost_internal, per DN and total),
      weighted and unweighted, with the remaining Q components; secondary, labelled: the TN-side signed interface
      delta (interface_delta_p, MW, summed over hours) per DN from the committed settlement detail -- an interface
      quantity, NOT the DN-internal activation volume (which no committed record of row 1 carries).
  (c) E^Av per representative year and node (both rows) with the 0.70 floor (minimum_soh x E_rated), SoH, margin to
      the floor and the floor rows active at terminal.
  Supporting: node 7 hourly TSO / DSO copy schedule and SoC for all 12 blocks of row 2 (the same single read).

CROSS-CHECKS (every quantity available from two sources; max absolute differences reported, none gated)
  stride x.tso / x.dso (p, q) vs pickle expected_shared_ess_p / q x baseMVA, all 12 node-7 blocks; expected vs pnet;
  pickle SoC vs the SoC recursion from the pickle's own pch / pdch (eff_ch 0.97, eff_dch 0.96, start 0.5 E);
  pickle SoC vs the SoC recursion from the RECORDED net power of each copy (stride x.tso, x.dso; split
  pch = max(pnet, 0), pdch = max(-pnet, 0)); ESSO capture pnet vs stride x.esso; ESSO capture pch - pdch vs pnet;
  E^Av: pickle shared_es_e_rated_fixed x baseMVA vs evaluation_record published_available_capacity_terminal vs
  ageing_trajectory_terminal; EFC per day: recomputed from the ESSO capture vs the record; per-block Q and its TSO
  generation / DSO flexibility components: component_levels_terminal vs recourse_blocks_all line k*; block weights
  vs the SRP1.json formula; row 1 node-7 storage entries (x = 0) all zero.

GUARDS
  `SolveProfileGuard(permitted=())` installed before any project import and verified at exactly 0.
  Pickle: `pickle.load` / `pickle.loads` / `pickle.Unpickler` replaced for the whole run by an allow-list gate of ONE
  (path, sha256) = row 2's `certified_models.pkl`, sha256 d0a3d512... (pinned in post_certification.json). The gate
  refuses any handle without a path, any path off the list, any byte-string `loads`, any `Unpickler`, any load after
  the first, and any file whose bytes (read from the handle at position 0) do not hash to the listed sha256; the
  hashed bytes are the bytes unpickled. Exactly 1 delegated load is required (0 in --dry-run); fewer or more fails.
  Model builds: `pyomo.core.base.PyomoModel.Model.__init__` blocked and counted (0 required). Unpickling resolves
  production rule functions by name (importing production modules is permitted; building or solving is not).
  Negative controls (each must raise PickleRefused, none may delegate): NC1 a gate whose allow-list carries a wrong
  sha256 for the same file; NC2 `pickle.loads` of a byte string; NC3 `pickle.load` of an in-memory handle; NC4 (real
  run) a second `pickle.load` of the allowed file after the single load.
  Refuses: an output directory that is missing or holds anything but launch.log; a concurrent copy; a missing capture
  path (checklist asserted before any reading); a JSONL whose sha256 differs from its pin or from its blob at
  741e8ddf; a pickle whose sha256 differs from the pin or from post_certification.json. JSON written through
  `gate_result_io`.

OUTPUT (default data/SRP1/Results/P515S55/e6_export/): e6_export.json, manifest_sha256.json; `--post-run` then writes
manifest_post_run_sha256.json over the outputs + launch.log. `--out-dir DIR` redirects (trials); `--dry-run` skips the
pickle load (expected loads 0) and refuses the default directory.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir -p data/SRP1/Results/P515S55/e6_export && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w183e_e6_export.py \\
      > data/SRP1/Results/P515S55/e6_export/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w183e_e6_export.py --post-run
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183e E6 export (never solves)').install()

import hashlib  # noqa: E402
import io  # noqa: E402

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W101 = os.path.join(P53, 'w101_srp1_continuation')
R1_EVAL = os.path.join(W101, 'campaign_s53_w101_srp1_cont_x0', 'evals', 'd110bd1a5977df1e_x0')
R2_EVAL = os.path.join(W101, 'campaign_s53_w101_srp1_cont_n7_4h_e1', 'evals', '3f084f2ffaeef2b7_n7_4h_e1')
PICKLE_REL = os.path.join(R2_EVAL, 'certified_models.pkl')
PICKLE_SHA256 = 'd0a3d512fa58f41053befdb784bb842ad57ec0d71c989aef776a0aa668e6030d'


class PickleRefused(RuntimeError):
    pass


class AllowListPickleGate:
    """Global replacement for pickle.load / pickle.loads: an allow-list of (path, sha256); exactly one load."""

    def __init__(self, allowed, label, original_loads):
        self.allowed = {os.path.realpath(os.path.join(REPO, p)): s for p, s in allowed}
        self.label = label
        self._orig_loads = original_loads
        self.counts = {'load_attempts': 0, 'loads_attempts': 0, 'unpickler_attempts': 0, 'refused': 0,
                       'delegated': 0}
        self.refusals = []
        self.loaded = []

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
        if self.counts['delegated'] >= 1:
            self._refuse('pickle.load', 'the single declared load has already been made (exactly 1 permitted)')
        if file.tell() != 0:
            self._refuse('pickle.load', f'handle not at position 0 (tell {file.tell()})')
        data = file.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != self.allowed[path]:
            self._refuse('pickle.load', f'sha256 of the bytes read {digest} != allow-listed {self.allowed[path]}')
        self.counts['delegated'] += 1
        self.loaded.append({'path': os.path.relpath(path, REPO), 'sha256_of_bytes_unpickled': digest,
                            'size_bytes': len(data)})
        return self._orig_loads(data, *args, **kwargs)

    def loads(self, *_a, **_k):
        self.counts['loads_attempts'] += 1
        self._refuse('pickle.loads', 'byte-string loads are not on the allow-list')


_ORIGINAL_PICKLE = {'load': pickle.load, 'loads': pickle.loads, 'Unpickler': pickle.Unpickler}
PICKLE_GATE = AllowListPickleGate([(PICKLE_REL, PICKLE_SHA256)], 'W183e pickle gate', _ORIGINAL_PICKLE['loads'])


class _BlockedUnpickler(pickle.Unpickler):
    def __init__(self, *_a, **_k):
        PICKLE_GATE.counts['unpickler_attempts'] += 1
        PICKLE_GATE._refuse('pickle.Unpickler', 'Unpickler is not on the allow-list')


pickle.load, pickle.loads, pickle.Unpickler = PICKLE_GATE.load, PICKLE_GATE.loads, _BlockedUnpickler

from pyomo.core.base.PyomoModel import Model as _PyomoModel  # noqa: E402

BUILD_COUNTS = {'pyomo_model_init': 0}


def _blocked_model_init(self, *_a, **_k):
    BUILD_COUNTS['pyomo_model_init'] += 1
    raise RuntimeError('W183e: a Pyomo model build was attempted -- no model builds are permitted')


_PyomoModel.__init__ = _blocked_model_init

import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pyomo.environ as pe  # noqa: E402

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer)

SCRIPT_REL = os.path.basename(__file__)
DEFAULT_OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e6_export')
W180B = os.path.join(P53, 'w180b_row16_class', 'w180b_row16_class.json')
JSONL_COMMIT = '741e8ddfdfb96f61d8505c9d83dbfcbeb902f303'
ROWS = {
    1: {'cell': 'ref:7aa017f0', 'label': 'x0', 'eval': R1_EVAL, 'k_star': 181,
        'stride_sha256': '0012cef5f7ad0722bbacdcf07141d3fa39b9c154c5dadbc7b4f27e31246295b9'},
    2: {'cell': 'ref:bd504ecf', 'label': 'n7_4h_e1', 'eval': R2_EVAL, 'k_star': 172,
        'stride_sha256': '47de0cb1eff168ec759108e1558f10768e21e5e496197938294edaa8c3eb0f4e'},
}
ESSO_CAPTURE = os.path.join(R2_EVAL, 'esso_capture', 's39_D', 'node7_cycle172.jsonl')
ESSO_CAPTURE_SHA256 = 'ac8bc61e6798cba31c1541c91df39d15311a0b4991be5ae9f547175c83db8a2a'
SRP1_JSON = os.path.join('data', 'SRP1', 'SRP1.json')
CASE_FILES = {('TSO', y): os.path.join('data', 'SRP1', 'case9', f'case9_{y}.json') for y in ('2025', '2030', '2035')}
CASE_FILES.update({('DSO7', y): os.path.join('data', 'SRP1', 'case33_2', f'case33_2_{y}.json')
                   for y in ('2025', '2030', '2035')})
STORAGE_NODE = 7
Q_COMPONENTS = ('generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost', 'res_curtailment_penalty',
                'ess_usage_cost', 'detector_penalty_total')
EFF_CH, EFF_DCH = 0.97, 0.96            # shared_energy_storage.SharedEnergyStorage defaults (never overwritten)
SOC_INIT_FRAC, SOC_MIN_FRAC, SOC_MAX_FRAC = 0.50, 0.10, 0.90   # definitions.ENERGY_STORAGE_* constants
MIN_SOH_DECLARED = 0.70
PROD_SOURCE_NAMES = {   # names the pickle extraction reads, asserted present in the production builder source
    'network.py': ['model.shared_es_pch = pe.Var', 'model.shared_es_pdch = pe.Var', 'model.shared_es_pnet = pe.Var',
                   'model.shared_es_qnet = pe.Var', 'model.shared_es_soc = pe.Var',
                   'model.shared_es_e_rated_fixed = pe.Param', 'model.shared_es_s_rated_fixed = pe.Param',
                   'model.slack_shared_es_soc_final_up = pe.Var'],
    'shared_resources_planning.py': ['tso_model[year][day].expected_shared_ess_p = pe.Var',
                                     'dso_model[year][day].expected_shared_ess_p = pe.Var'],
    'definitions.py': ['ENERGY_STORAGE_MAX_ENERGY_STORED = 0.90', 'ENERGY_STORAGE_MIN_ENERGY_STORED = 0.10',
                       'ENERGY_STORAGE_RELATIVE_INIT_SOC = 0.50'],
    'shared_energy_storage.py': ['self.eff_ch = 0.97', 'self.eff_dch = 0.96'],
    'model_construction_helpers.py': ['delta = (eff_ch * m.shared_es_pch[e, s_m0, s_o0, p] * dt',
                                      '- m.shared_es_pdch[e, s_m0, s_o0, p] * dt / eff_dch)'],
}

INPUTS = {}
_TRACKED = None


def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


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
        return 'UNCOMMITTED'
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


def _jsonl(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def _maxabs(values):
    values = [abs(v) for v in values]
    return max(values) if values else None


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
def _refuse(check, detail):
    _log(f'REFUSED (check: {check}): {detail}')
    raise SystemExit(f'REFUSED (check: {check}): {detail}')


def capture_checklist():
    """Rule eleven: every capture path the export needs exists before any reading (fail fast)."""
    need = {}
    for no, r in ROWS.items():
        for f in ('component_levels_terminal.json', 'evaluation_record.json', 'settling_decision.json',
                  'recourse_blocks_all.jsonl', 'ess_entry_stride_baseline.jsonl',
                  'interface_settlement_detail_s31c.json', 'post_certification.json'):
            need[f'row{no} {f}'] = os.path.join(r['eval'], f)
    need['row2 certified_models.pkl'] = PICKLE_REL
    need['row2 ESSO capture k*'] = ESSO_CAPTURE
    need['W180b row map'] = W180B
    need['SRP1.json'] = SRP1_JSON
    for k, p in CASE_FILES.items():
        need[f'case file {k[0]} {k[1]}'] = p
    for f in PROD_SOURCE_NAMES:
        need[f'production source {f}'] = f
    missing = sorted(k for k, p in need.items() if not os.path.isfile(p))
    keys = {}
    if not missing:
        for no, r in ROWS.items():
            cl = json.load(open(os.path.join(r['eval'], 'component_levels_terminal.json')))
            b0 = next(iter(cl['blocks'].values()))
            keys[f'row{no} component_levels 48 blocks with every Q component, weighted and unweighted'] = (
                len(cl['blocks']) == 48 and all(c in b['weighted'] and c in b['unweighted']
                                                for b in cl['blocks'].values() for c in Q_COMPONENTS)
                and 'admm_block_weight' in b0)
            keys[f'row{no} component_levels cycles_run == k*'] = cl.get('cycles_run') == r['k_star']
            er = json.load(open(os.path.join(r['eval'], 'evaluation_record.json')))
            keys[f'row{no} evaluation_record published_available_capacity_terminal per node'] = all(
                'published_available_capacity_terminal' in er['storage_per_node'][n] for n in ('5', '7', '9'))
            keys[f'row{no} evaluation_record ageing_trajectory_terminal'] = 'nodes' in er.get(
                'ageing_trajectory_terminal', {})
            sd = json.load(open(os.path.join(r['eval'], 'settling_decision.json')))
            keys[f'row{no} settling_decision certified at k* with a window'] = (
                sd.get('status') == 'certified' and sd.get('k_star') == r['k_star'] and len(sd.get('window', [])) == 2)
            isd = json.load(open(os.path.join(r['eval'], 'interface_settlement_detail_s31c.json')))
            keys[f'row{no} settlement detail node 7 per-period price and interface delta'] = all(
                'price_per_mwh' in v and 'delta_p_mw' in v
                for v in isd['interface_reporting_detail']['7']['2025']['Spring']['periods'].values())
        src = {f: open(f, encoding='utf-8').read() for f in PROD_SOURCE_NAMES}
        for f, names in PROD_SOURCE_NAMES.items():
            for n in names:
                keys[f'production source {f}: {n!r}'] = n in src[f]
        with open(ESSO_CAPTURE, encoding='utf-8') as fh:
            lines = [json.loads(x) for x in fh if x.strip()]
        keys['ESSO capture 288 lines at cycle 172 with pch, pdch, pnet'] = (
            len(lines) == 288 and all(str(x.get('cycle')) == '172' and all(k in x for k in ('pch', 'pdch', 'pnet'))
                                      for x in lines))
    bad_keys = sorted(k for k, ok in keys.items() if not ok)
    return {'ok': not missing and not bad_keys, 'n_paths': len(need), 'missing_paths': missing,
            'n_key_checks': len(keys), 'failed_key_checks': bad_keys, 'paths': need}


def verify_jsonl_pins():
    out = {}
    head = _git(['rev-parse', 'HEAD']).decode().strip()
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor', JSONL_COMMIT, head], cwd=REPO).returncode == 0
    for no, r in ROWS.items():
        path = os.path.join(r['eval'], 'ess_entry_stride_baseline.jsonl')
        work = _sha(path)
        blob_741 = hashlib.sha256(_git(['show', f'{JSONL_COMMIT}:{path}'])).hexdigest()
        blob_head = hashlib.sha256(_git(['show', f'HEAD:{path}'])).hexdigest()
        out[f'row{no}'] = {'path': path, 'pinned': r['stride_sha256'], 'working_file': work,
                           'blob_at_741e8ddf': blob_741, 'blob_at_HEAD': blob_head,
                           'ok': work == blob_741 == blob_head == r['stride_sha256']}
        _reg(path, work)
    out['741e8ddf_is_ancestor_of_HEAD'] = ancestor
    out['ok'] = ancestor and all(v['ok'] for k, v in out.items() if k.startswith('row'))
    return out


def verify_pickle_pin():
    pc = json.load(open(os.path.join(R2_EVAL, 'post_certification.json')))
    _reg(os.path.join(R2_EVAL, 'post_certification.json'))
    pinned_pc = pc['persisted_models']['sha256']
    pinned_path = pc['persisted_models']['path']
    actual = _sha(PICKLE_REL)
    res = {'path': PICKLE_REL, 'pinned_in_task': PICKLE_SHA256, 'pinned_in_post_certification': pinned_pc,
           'post_certification_path_equals': os.path.normpath(pinned_path) == os.path.normpath(PICKLE_REL),
           'size_pinned': pc['persisted_models']['size_bytes'], 'size_actual': os.path.getsize(PICKLE_REL),
           'sha256_actual_before_load': actual}
    res['ok'] = (actual == PICKLE_SHA256 == pinned_pc and res['post_certification_path_equals']
                 and res['size_pinned'] == res['size_actual'])
    _reg(PICKLE_REL, actual)
    return res


def verify_row_identity(er, w180b_rows):
    out = {}
    for no, r in ROWS.items():
        w = next(x for x in w180b_rows if x['no'] == no)
        out[f'row{no}'] = {'cell_W180b': w['cell'], 'cell_declared': r['cell'],
                           'eval_key_W180b': w['frozen']['eval_key'], 'eval_key_record': er[no]['eval_key'],
                           'k_star_W180b': w['frozen']['k_star'], 'certification_cycle': er[no]['certification_cycle'],
                           'status_record': er[no]['status'],
                           'candidate_canonical': er[no]['candidate_canonical'],
                           'candidate_key': er[no]['candidate_key']}
        o = out[f'row{no}']
        o['ok'] = (o['cell_W180b'] == r['cell'] and o['eval_key_W180b'] == o['eval_key_record']
                   and o['k_star_W180b'] == r['k_star'] == o['certification_cycle'] and o['status_record'] == 'certified')
    out['ok'] = all(v['ok'] for v in out.values())
    return out


# ============================================================================================ records
def block_q(block):
    return sum(block['weighted'][c] for c in Q_COMPONENTS)


def block_q_unweighted(block):
    return sum(block['unweighted'][c] for c in Q_COMPONENTS)


def yd_order(srp1):
    return [(y, d) for y in srp1['Years'] for d in srp1['Days']]


def per_yd(cl, fn):
    out = {}
    for b in cl['blocks'].values():
        out[(b['year'], b['day'])] = out.get((b['year'], b['day']), 0.0) + fn(b)
    return out


def rb_yd_values(line):
    """(year, day) -> sum of the four network block values of one recourse_blocks_all line."""
    out = {}
    for b in line['blocks']:
        if b['agent'] in ('TSO', 'DSO'):
            out[(b['year'], b['day'])] = out.get((b['year'], b['day']), 0.0) + b['value']
    return out


def rb_key(b):
    return f"TSO|{b['year']}|{b['day']}" if b['agent'] == 'TSO' else f"DSO|{b['node_id']}|{b['year']}|{b['day']}"


def day_selection(srp1, cl, rb, sd):
    identity = {}
    for no in ROWS:
        s = sum(block_q(b) for b in cl[no]['blocks'].values())
        g = cl[no]['recourse_components']['gross_operational_cost']
        identity[f'row{no}'] = {'sum_of_48_block_Q': s, 'gross_operational_cost': g, 'residual': s - g,
                                'relative_residual': (s - g) / g}
    q = {no: per_yd(cl[no], block_q) for no in ROWS}
    qu = {no: per_yd(cl[no], block_q_unweighted) for no in ROWS}
    weights = {}
    for b in cl[2]['blocks'].values():
        weights.setdefault((b['year'], b['day']), set()).add(b['admm_block_weight'])
    for b in cl[1]['blocks'].values():
        weights.setdefault((b['year'], b['day']), set()).add(b['admm_block_weight'])
    # resolution from recourse_blocks_all over each row's certification window
    res = {}
    for no in ROWS:
        w0, w1 = sd[no]['window']
        by_cycle = {ln['cycle']: rb_yd_values(ln) for ln in rb[no] if w0 - 1 <= ln['cycle'] <= w1}
        missing = [c for c in range(w0 - 1, w1 + 1) if c not in by_cycle]
        res[no] = {'window': [w0, w1], 'missing_cycles': missing, 'per_yd': {}}
        for yd in yd_order(srp1):
            vals = [by_cycle[c][yd] for c in range(w0, w1 + 1) if c in by_cycle]
            res[no]['per_yd'][yd] = {'range_over_window': max(vals) - min(vals),
                                     'terminal_step_abs': abs(by_cycle[w1][yd] - by_cycle[w1 - 1][yd])}
    ranking = []
    for (y, d) in yd_order(srp1):
        c = q[1][(y, d)] - q[2][(y, d)]
        bar = res[1]['per_yd'][(y, d)]['range_over_window'] + res[2]['per_yd'][(y, d)]['range_over_window']
        ranking.append({'year': y, 'day': d, 'block_weight': sorted(weights[(y, d)]),
                        'Q_row1_x0_block': q[1][(y, d)], 'Q_row2_x_block': q[2][(y, d)],
                        'value_contribution_Q0_minus_Qx': c,
                        'value_contribution_unweighted_one_day': qu[1][(y, d)] - qu[2][(y, d)],
                        'bar_sum_of_window_ranges': bar,
                        'range_row1_window': res[1]['per_yd'][(y, d)]['range_over_window'],
                        'range_row2_window': res[2]['per_yd'][(y, d)]['range_over_window'],
                        'terminal_step_row1': res[1]['per_yd'][(y, d)]['terminal_step_abs'],
                        'terminal_step_row2': res[2]['per_yd'][(y, d)]['terminal_step_abs']})
    ranking.sort(key=lambda e: (-e['value_contribution_Q0_minus_Qx'], e['year'], e['day']))
    for i, e in enumerate(ranking, 1):
        e['rank'] = i
    un_rank = sorted(ranking, key=lambda e: -e['value_contribution_unweighted_one_day'])
    for i, e in enumerate(un_rank, 1):
        e['rank_by_unweighted_one_day_contribution'] = i
    for a, b in zip(ranking, ranking[1:]):
        m = a['value_contribution_Q0_minus_Qx'] - b['value_contribution_Q0_minus_Qx']
        pair_bar = a['bar_sum_of_window_ranges'] + b['bar_sum_of_window_ranges']
        a['margin_to_next_rank'] = m
        a['margin_to_next_rank_not_explained_by_stopping_slack'] = m > pair_bar
    top, second = ranking[0], ranking[1]
    margin = top['value_contribution_Q0_minus_Qx'] - second['value_contribution_Q0_minus_Qx']
    bar = top['bar_sum_of_window_ranges'] + second['bar_sum_of_window_ranges']
    total_c = sum(e['value_contribution_Q0_minus_Qx'] for e in ranking)
    return {
        'rule': ('declared (STEP7 plan section 7): the (year, day) block with the largest value contribution '
                 'Q(0) - Q(x) in the certified decomposition; labelled an illustrative day, not a typical-day '
                 'estimate; horizon totals remain representative-day-weighted'),
        'convention': ('Q = certified gross_operational_cost, contracted interface settlement excluded; block values '
                       'block-weighted (year count x day count x discount annualisation, _get_admm_block_weight); '
                       'value contribution = Q_row1(x = 0) - Q_row2(x), EUR; positive = cheaper with storage'),
        'per_block_Q_definition': f'sum over {list(Q_COMPONENTS)} of component_levels_terminal blocks.<b>.weighted, '
                                  'summed over TSO + DSO 5/7/9 of the (year, day)',
        'identity_check': identity,
        'block_weights_equal_across_networks_and_rows': all(len(v) == 1 for v in weights.values()),
        'ranking': ranking,
        'sum_of_contributions': total_c,
        'Q_row1_minus_Q_row2': (cl[1]['recourse_components']['gross_operational_cost']
                                - cl[2]['recourse_components']['gross_operational_cost']),
        'declared_day': {'year': top['year'], 'day': top['day'], 'label': 'illustrative day (not a typical-day estimate)',
                         'value_contribution': top['value_contribution_Q0_minus_Qx'],
                         'share_of_total_value': top['value_contribution_Q0_minus_Qx'] / total_c},
        'margin_rank1_minus_rank2': margin,
        'resolution': {'bar_rank1_plus_rank2': bar, 'margin_over_bar': margin / bar if bar else None,
                       'not_explained_by_stopping_slack': margin > bar,
                       'windows': {f'row{no}': res[no]['window'] for no in ROWS},
                       'missing_cycles_in_windows': {f'row{no}': res[no]['missing_cycles'] for no in ROWS},
                       'note': ('bar = sum over both rows of the per-(year, day) block range over each row\'s '
                                'certification window (recourse_blocks_all.jsonl); a local, stopping-slack measure '
                                'only (CLAUDE.md refinement of 2026-09-13)')},
    }


def panel_b(srp1, cl, rb, isd):
    rows_out = []
    xcheck = {'tso_generation_cost': [], 'dso_flexibility_cost': [], 'block_q_value': []}
    rb_last = {no: rb[no][-1] for no in ROWS}
    for no in ROWS:
        assert rb_last[no]['cycle'] == ROWS[no]['k_star']
        ocb = {rb_key(b): b for b in rb_last[no]['objective_component_blocks']}
        vb = {rb_key(b): b['value'] for b in rb_last[no]['blocks'] if b['agent'] in ('TSO', 'DSO')}
        for key, b in cl[no]['blocks'].items():
            xcheck['block_q_value'].append(block_q(b) - vb[key])
            if b['kind'] == 'TSO':
                xcheck['tso_generation_cost'].append(b['weighted']['generation_cost'] - ocb[key]['generation_cost'])
            else:
                xcheck['dso_flexibility_cost'].append(b['weighted']['flexibility_cost_internal']
                                                      - ocb[key]['flexibility_cost'])
    nodes = ('5', '7', '9')
    for (y, d) in yd_order(srp1):
        e = {'year': y, 'day': d}
        for wk in ('weighted', 'unweighted'):
            t = {no: cl[no]['blocks'][f'TSO|{y}|{d}'][wk] for no in ROWS}
            ds = {no: {n: cl[no]['blocks'][f'DSO|{n}|{y}|{d}'][wk] for n in nodes} for no in ROWS}
            sec = {'tn_conventional_cost_tso_generation_cost': {
                'row1_x0': t[1]['generation_cost'], 'row2_x': t[2]['generation_cost'],
                'change_x_minus_x0': t[2]['generation_cost'] - t[1]['generation_cost']}}
            flex = {}
            for n in nodes:
                a, b_ = ds[1][n]['flexibility_cost_internal'], ds[2][n]['flexibility_cost_internal']
                flex[n] = {'row1_x0': a, 'row2_x': b_, 'change_x_minus_x0': b_ - a}
            a = sum(ds[1][n]['flexibility_cost_internal'] for n in nodes)
            b_ = sum(ds[2][n]['flexibility_cost_internal'] for n in nodes)
            flex['total'] = {'row1_x0': a, 'row2_x': b_, 'change_x_minus_x0': b_ - a}
            sec['dn_flexibility_activation_cost_dso_flexibility_cost_internal'] = flex
            other = {}
            for comp in Q_COMPONENTS:
                if comp in ('flexibility_cost_internal',):
                    continue
                v1 = (t[1][comp] if comp != 'generation_cost' else 0.0) + sum(ds[1][n][comp] for n in nodes)
                v2 = (t[2][comp] if comp != 'generation_cost' else 0.0) + sum(ds[2][n][comp] for n in nodes)
                label = 'dso_generation_cost' if comp == 'generation_cost' else f'{comp}_tso_plus_dso'
                other[label] = {'change_x_minus_x0': v2 - v1}
            tflex1 = t[1]['flexibility_cost_internal']
            tflex2 = t[2]['flexibility_cost_internal']
            other['tso_flexibility_cost_internal'] = {'change_x_minus_x0': tflex2 - tflex1}
            sec['other_Q_components'] = other
            fn = block_q if wk == 'weighted' else block_q_unweighted
            q1 = fn(cl[1]['blocks'][f'TSO|{y}|{d}']) + sum(fn(cl[1]['blocks'][f'DSO|{n}|{y}|{d}']) for n in nodes)
            q2 = fn(cl[2]['blocks'][f'TSO|{y}|{d}']) + sum(fn(cl[2]['blocks'][f'DSO|{n}|{y}|{d}']) for n in nodes)
            sec['Q_block'] = {'row1_x0': q1, 'row2_x': q2, 'change_x_minus_x0': q2 - q1}
            sec['components_sum_to_Q_change_residual'] = (
                sec['tn_conventional_cost_tso_generation_cost']['change_x_minus_x0'] + flex['total']['change_x_minus_x0']
                + sum(v['change_x_minus_x0'] for v in other.values()) - (q2 - q1))
            e[wk] = sec
        iface = {}
        for n in nodes:
            per = {}
            for no in ROWS:
                periods = isd[no]['interface_reporting_detail'][n][y][d]['periods']
                vals = [sum(periods[str(p)]['delta_p_mw'].values()) for p in range(24)]
                per[f'row{no}'] = {'sum_signed_mwh': sum(vals), 'sum_abs_mwh': sum(abs(v) for v in vals)}
            per['change_sum_signed_x_minus_x0'] = per['row2']['sum_signed_mwh'] - per['row1']['sum_signed_mwh']
            per['change_sum_abs_x_minus_x0'] = per['row2']['sum_abs_mwh'] - per['row1']['sum_abs_mwh']
            iface[n] = per
        e['secondary_interface_delta_p_tn_side'] = iface
        rows_out.append(e)
    return {
        'convention': ('change = row 2 (x) minus row 1 (x = 0), EUR; "weighted" = block-weighted and discounted '
                       '(the Q decomposition), "unweighted" = one representative day (scenario-probability-weighted '
                       'only); a negative change is a saving'),
        'dn_flexibility_activation_measure': ('DSO flexibility_cost_internal (EUR) -- the activation COST; the '
                                              'activation volume (MWh) of row 1 is carried by no committed record'),
        'secondary_label': ('secondary_interface_delta_p_tn_side: sum over the 24 hours of the TN-side signed interface '
                            'reparametrisation interface_delta_p (MW x 1 h) per DN, from the committed '
                            'interface_settlement_detail_s31c.json -- an INTERFACE quantity, not the DN-internal '
                            'activation volume; unweighted (one day)'),
        'blocks': rows_out,
        'horizon_totals_weighted': {
            'tn_conventional_cost_change': sum(r['weighted']['tn_conventional_cost_tso_generation_cost'][
                                                   'change_x_minus_x0'] for r in rows_out),
            'dn_flexibility_cost_change_total': sum(r['weighted'][
                'dn_flexibility_activation_cost_dso_flexibility_cost_internal']['total']['change_x_minus_x0']
                for r in rows_out),
            'Q_change': sum(r['weighted']['Q_block']['change_x_minus_x0'] for r in rows_out)},
        'cross_check_vs_recourse_blocks_all_line_k_star': {
            k: {'n': len(v), 'max_abs_diff_eur': _maxabs(v)} for k, v in xcheck.items()},
    }


def panel_c(er):
    out = {}
    xcheck = []
    for no in ROWS:
        rec = er[no]
        min_soh = rec['ess_ageing_baseline']['minimum_soh']
        row = {'minimum_soh_record': min_soh, 'nodes': {}}
        traj = rec['ageing_trajectory_terminal']['nodes']
        for n, s in rec['storage_per_node'].items():
            e_rated = s['e_mwh']
            node = {'e_rated_mwh': e_rated, 's_rated_mva': s['s_mva'], 'has_storage': s['has_storage'],
                    'soh_floor_rows_active_at_terminal': s['soh_floor_rows_active_at_terminal'], 'years': {}}
            cells = {c['block_year']: c for c in traj[n]['cells']}
            for y, cap in s['published_available_capacity_terminal'].items():
                ea = cap['e_available']
                floor = min_soh * e_rated
                node['years'][y] = {'e_available_mwh': ea, 's_available_mva': cap['s_available'],
                                    'soh': (ea / e_rated) if e_rated else None,
                                    'floor_e_mwh_0p70_x_e_rated': floor,
                                    'margin_above_floor_mwh': (ea - floor) if e_rated else None,
                                    'soc_lower_10pct_mwh': SOC_MIN_FRAC * ea, 'soc_upper_90pct_mwh': SOC_MAX_FRAC * ea,
                                    'efc_per_day': cells.get(y, {}).get('efc_per_day')}
                if y in cells:
                    xcheck.append(ea - cells[y]['e_available'])
            row['nodes'][n] = node
        out[f'row{no}'] = row
    out['minimum_soh_declared'] = MIN_SOH_DECLARED
    out['minimum_soh_record_equals_declared'] = all(out[f'row{no}']['minimum_soh_record'] == MIN_SOH_DECLARED
                                                    for no in ROWS)
    out['cross_check_published_vs_ageing_trajectory_e_available'] = {'n': len(xcheck), 'max_abs_diff_mwh':
                                                                     _maxabs(xcheck)}
    out['convention'] = ('E^Av = published_available_capacity_terminal.e_available (END-of-block SoH x E_rated; '
                         'available_energy_soh_point "end"); floor = minimum_soh x E_rated = 0.70 x E_rated')
    return out


# ============================================================================================ stride / ESSO
def stride_index(line):
    return {(e['node_id'], e['year'], e['day'], e['power_type']): e for e in line['entries']}


def soc_recursion(pch, pdch, e_mwh):
    soc, prev = [], SOC_INIT_FRAC * e_mwh
    for c, dc in zip(pch, pdch):
        prev = prev + EFF_CH * c - dc / EFF_DCH
        soc.append(prev)
    return soc


def soc_from_net(pnet, e_mwh):
    return soc_recursion([max(v, 0.0) for v in pnet], [max(-v, 0.0) for v in pnet], e_mwh)


def esso_section(srp1, capture, stride2, er2):
    years, days = list(srp1['Years']), list(srp1['Days'])
    by = {}
    for x in capture:
        by[(years[x['y']], days[x['d']], x['p'])] = x
    diffs_pnet, diffs_def = [], []
    for (y, d) in yd_order(srp1):
        xs = stride2[(STORAGE_NODE, y, d, 'p')]['x']['esso']
        for p in range(24):
            c = by[(y, d, p)]
            diffs_pnet.append(c['pnet'] - xs[p])
            diffs_def.append(c['pch'] - c['pdch'] - c['pnet'])
    traj = {c['block_year']: c for c in er2['ageing_trajectory_terminal']['nodes'][str(STORAGE_NODE)]['cells']}
    efc = {}
    for y in years:
        avg = sum((srp1['Days'][d] / 365.0) * sum(EFF_CH * by[(y, d, p)]['pch'] + by[(y, d, p)]['pdch'] / EFF_DCH
                                                  for p in range(24)) for d in days)
        e_rated = traj[y]['e_rated']
        efc[y] = {'avg_ch_dch_recomputed': avg, 'avg_ch_dch_record': traj[y]['avg_ch_dch'],
                  'efc_per_day_recomputed': avg / (2.0 * e_rated), 'efc_per_day_record': traj[y]['efc_per_day'],
                  'diff_efc': avg / (2.0 * e_rated) - traj[y]['efc_per_day']}
    return by, {
        'source': ESSO_CAPTURE,
        'index_mapping': 'capture y -> SRP1.json Years order, d -> Days order, p -> period (validated by the pnet match)',
        'pnet_vs_stride_x_esso': {'n': len(diffs_pnet), 'max_abs_diff_mw': _maxabs(diffs_pnet)},
        'pch_minus_pdch_minus_pnet': {'n': len(diffs_def), 'max_abs_mw': _maxabs(diffs_def)},
        'efc_per_day_recomputed_vs_record': efc,
        'efc_formula': ('avg_ch_dch(y) = sum_d (num_days_d / 365) sum_p (0.97 pch + pdch / 0.96) x 1 h '
                        '(shared_energy_storage_data, energy_storage_charging_discharging); EFC/day = avg / (2 E_rated)'),
    }


# ============================================================================================ pickle
def _first(s):
    items = list(s)
    if len(items) != 1:
        raise RuntimeError(f'expected exactly one scenario, got {len(items)}')
    return items[0]


def _val(c):
    v = pe.value(c, exception=False)
    return None if v is None else float(v)


def _by_str(mapping, key):
    """The value whose key equals `key` as a string (pickled dicts may key years by int)."""
    hits = [k for k in mapping if str(k) == str(key)]
    if len(hits) != 1:
        raise KeyError(f'{key!r}: {len(hits)} matching keys in {sorted(map(str, mapping))}')
    return mapping[hits[0]]


def extract_copy(model, e, base, kind):
    sm, so = _first(model.scenarios_market), _first(model.scenarios_operation)
    periods = list(model.periods)
    out = {'periods': periods, 'scenario': [sm, so], 'base_mva': base,
           'e_rated_fixed_mwh': _val(model.shared_es_e_rated_fixed[e]) * base,
           's_rated_fixed_mva': _val(model.shared_es_s_rated_fixed[e]) * base}
    for name, key in (('shared_es_pch', 'pch_mw'), ('shared_es_pdch', 'pdch_mw'), ('shared_es_pnet', 'pnet_mw'),
                      ('shared_es_qnet', 'qnet_mvar'), ('shared_es_soc', 'soc_mwh')):
        var = getattr(model, name)
        out[key] = [_val(var[e, sm, so, p]) * base for p in periods]
    if kind == 'TSO':
        out['expected_p_mw'] = [_val(model.expected_shared_ess_p[e, p]) * base for p in periods]
        out['expected_q_mvar'] = [_val(model.expected_shared_ess_q[e, p]) * base for p in periods]
    else:
        out['expected_p_mw'] = [_val(model.expected_shared_ess_p[p]) * base for p in periods]
        out['expected_q_mvar'] = [_val(model.expected_shared_ess_q[p]) * base for p in periods]
    if hasattr(model, 'slack_shared_es_soc_final_up'):
        out['day_end_slack_up_mwh'] = _val(model.slack_shared_es_soc_final_up[e, sm, so]) * base
        out['day_end_slack_down_mwh'] = _val(model.slack_shared_es_soc_final_down[e, sm, so]) * base
    return out


def copy_checks(cp, stride_p, stride_q, agent):
    e = cp['e_rated_fixed_mwh']
    rec = soc_recursion(cp['pch_mw'], cp['pdch_mw'], e)
    rec_net = soc_from_net(stride_p, e)
    return {
        f'stride_x_{agent}_p_vs_pickle_expected_p': _maxabs([a - b for a, b in zip(stride_p, cp['expected_p_mw'])]),
        f'stride_x_{agent}_q_vs_pickle_expected_q': _maxabs([a - b for a, b in zip(stride_q, cp['expected_q_mvar'])]),
        'pickle_expected_p_vs_pnet': _maxabs([a - b for a, b in zip(cp['expected_p_mw'], cp['pnet_mw'])]),
        'pickle_expected_q_vs_qnet': _maxabs([a - b for a, b in zip(cp['expected_q_mvar'], cp['qnet_mvar'])]),
        'pickle_pnet_vs_pch_minus_pdch': _maxabs([n - (c - d) for n, c, d in zip(cp['pnet_mw'], cp['pch_mw'],
                                                                                 cp['pdch_mw'])]),
        'pickle_soc_vs_recursion_from_pickle_pch_pdch_mwh': _maxabs([a - b for a, b in zip(cp['soc_mwh'], rec)]),
        f'pickle_soc_vs_recursion_from_recorded_net_power_stride_x_{agent}_mwh': _maxabs(
            [a - b for a, b in zip(cp['soc_mwh'], rec_net)]),
        'soc_min_minus_10pct_e_mwh': min(cp['soc_mwh']) - SOC_MIN_FRAC * e,
        'soc_90pct_e_minus_max_mwh': SOC_MAX_FRAC * e - max(cp['soc_mwh']),
        'day_end_soc_minus_half_e_mwh': cp['soc_mwh'][-1] - SOC_INIT_FRAC * e,
        'max_min_pch_pdch_mw': max(min(c, d) for c, d in zip(cp['pch_mw'], cp['pdch_mw'])),
    }


def pickle_section(srp1, models, base, stride2, er2):
    dn_order = [dn['connection_node_id'] for dn in srp1['DistributionNetworks']]
    e_tso = dn_order.index(STORAGE_NODE)
    dso_key = next(k for k in models['dso'] if int(k) == STORAGE_NODE)
    tso_dict, dso_dict = models['tso'], models['dso'][dso_key]
    pub = er2['storage_per_node'][str(STORAGE_NODE)]['published_available_capacity_terminal']
    blocks, checks, eav = {}, {}, []
    other_nodes_maxabs = []
    for (y, d) in yd_order(srp1):
        tm, dm = _by_str(_by_str(tso_dict, y), d), _by_str(_by_str(dso_dict, y), d)
        n_tso_ess = len(list(tm.shared_energy_storages))
        tso = extract_copy(tm, e_tso, base[('TSO', y)], 'TSO')
        dso = extract_copy(dm, 0, base[('DSO7', y)], 'DSO')
        sp = stride2[(STORAGE_NODE, y, d, 'p')]
        sq = stride2[(STORAGE_NODE, y, d, 'q')]
        blocks[f'{y}|{d}'] = {'tso_copy': tso, 'dso_copy': dso}
        checks[f'{y}|{d}'] = {'tso_copy': copy_checks(tso, sp['x']['tso'], sq['x']['tso'], 'tso'),
                              'dso_copy': copy_checks(dso, sp['x']['dso'], sq['x']['dso'], 'dso'),
                              'n_tso_shared_ess': n_tso_ess}
        for cp in (tso, dso):
            eav.append({'block': f'{y}|{d}', 'e_pickle': cp['e_rated_fixed_mwh'], 'e_record': pub[y]['e_available'],
                        's_pickle': cp['s_rated_fixed_mva'], 's_record': pub[y]['s_available']})
        sm, so = _first(tm.scenarios_market), _first(tm.scenarios_operation)
        for i in range(n_tso_ess):
            if i == e_tso:
                continue
            other_nodes_maxabs.append(abs(_val(tm.shared_es_e_rated_fixed[i])))
            for name in ('shared_es_pnet', 'shared_es_qnet', 'shared_es_soc'):
                var = getattr(tm, name)
                other_nodes_maxabs.extend(abs(_val(var[i, sm, so, p])) for p in tm.periods)
    agg = {}
    for blk in checks.values():
        for side in ('tso_copy', 'dso_copy'):
            for k, v in blk[side].items():
                agg.setdefault(f'{side}.{k}', []).append(v)
    summary = {}
    for k, vals in agg.items():
        if k.endswith(('soc_min_minus_10pct_e_mwh', 'soc_90pct_e_minus_max_mwh')):
            summary[k] = {'min_over_12_blocks': min(vals)}
        elif k.endswith('day_end_soc_minus_half_e_mwh'):
            summary[k] = {'max_abs_over_12_blocks': _maxabs(vals)}
        else:
            summary[k] = {'max_over_12_blocks': max(vals)}
    e_diff = [x['e_pickle'] - x['e_record'] for x in eav]
    s_diff = [x['s_pickle'] - x['s_record'] for x in eav]
    tso_vs_dso_soc = _maxabs([a - b for blk in blocks.values()
                              for a, b in zip(blk['tso_copy']['soc_mwh'], blk['dso_copy']['soc_mwh'])])
    return blocks, {
        'tso_shared_ess_index_of_node_7': e_tso,
        'tso_index_mapping': 'SRP1.json DistributionNetworks order (planning_problem.distribution_networks insertion '
                             'order, _add_shared_energy_storage_to_transmission_network); validated for node 7 by the '
                             'stride match; nodes 5/9 carry zero capacity (immaterial)',
        'dso_key': str(dso_key), 'dso_shared_ess_index': 0,
        'per_block': checks, 'summary_over_12_blocks': summary,
        'tso_copy_soc_vs_dso_copy_soc_max_abs_mwh': tso_vs_dso_soc,
        'e_av_pickle_vs_record': {'n': len(eav), 'max_abs_diff_e_mwh': _maxabs(e_diff),
                                  'max_abs_diff_s_mva': _maxabs(s_diff), 'entries': eav,
                                  'note': ('network copies hold the E^Av that the ESSO published BEFORE the cycle-k* '
                                           'network solves (shared_resources_planning.get_updated_capacities at the '
                                           'end of cycle k*-1); the record holds the terminal ESSO value '
                                           '(hypothesis for any difference, not verified)')},
        'tso_other_shared_ess_max_abs': _maxabs(other_nodes_maxabs),
    }


def declared_day_table(y, d, blocks, stride2, capture_by, isd2, er2):
    blk = blocks[f'{y}|{d}']
    tso, dso = blk['tso_copy'], blk['dso_copy']
    sp, sq = stride2[(STORAGE_NODE, y, d, 'p')], stride2[(STORAGE_NODE, y, d, 'q')]
    ea = er2['storage_per_node'][str(STORAGE_NODE)]['published_available_capacity_terminal'][y]['e_available']
    pch = [capture_by[(y, d, p)]['pch'] for p in range(24)]
    pdch = [capture_by[(y, d, p)]['pdch'] for p in range(24)]
    soc_esso = soc_recursion(pch, pdch, ea)
    soc_z = soc_from_net(sp['z'], ea)
    periods = isd2['interface_reporting_detail'][str(STORAGE_NODE)][y][d]['periods']
    hours = []
    for p in range(24):
        c = capture_by[(y, d, p)]
        hours.append({
            'period': p, 'hour_ending': p + 1,
            'market_price_eur_per_mwh': periods[str(p)]['price_per_mwh'],
            'esso': {'pch_mw': c['pch'], 'pdch_mw': c['pdch'], 'pnet_mw': c['pnet'],
                     'p_mw_stride': sp['x']['esso'][p], 'q_mvar_stride': sq['x']['esso'][p]},
            'tso_copy': {'pch_mw': tso['pch_mw'][p], 'pdch_mw': tso['pdch_mw'][p], 'pnet_mw': tso['pnet_mw'][p],
                         'qnet_mvar': tso['qnet_mvar'][p], 'soc_mwh': tso['soc_mwh'][p],
                         'expected_p_mw': tso['expected_p_mw'][p], 'expected_q_mvar': tso['expected_q_mvar'][p],
                         'p_mw_stride': sp['x']['tso'][p], 'q_mvar_stride': sq['x']['tso'][p]},
            'dso_copy': {'pch_mw': dso['pch_mw'][p], 'pdch_mw': dso['pdch_mw'][p], 'pnet_mw': dso['pnet_mw'][p],
                         'qnet_mvar': dso['qnet_mvar'][p], 'soc_mwh': dso['soc_mwh'][p],
                         'expected_p_mw': dso['expected_p_mw'][p], 'expected_q_mvar': dso['expected_q_mvar'][p],
                         'p_mw_stride': sp['x']['dso'][p], 'q_mvar_stride': sq['x']['dso'][p]},
            'consensus_z': {'p_mw': sp['z'][p], 'q_mvar': sq['z'][p]},
            'soc_DERIVED': {'from_esso_pch_pdch_mwh': soc_esso[p], 'from_consensus_z_net_mwh': soc_z[p]},
        })
    return {
        'year': y, 'day': d, 'node': STORAGE_NODE, 'label': 'illustrative day (not a typical-day estimate)',
        'units': 'MW, MVAr, MWh (1 h periods); pnet = pch - pdch, positive = charging (sess_pnet_rule)',
        'soc_limits': {'e_available_record_mwh': ea, 'lower_10pct_record_mwh': SOC_MIN_FRAC * ea,
                       'upper_90pct_record_mwh': SOC_MAX_FRAC * ea,
                       'e_rated_fixed_pickle_tso_mwh': tso['e_rated_fixed_mwh'],
                       'lower_10pct_pickle_tso_mwh': SOC_MIN_FRAC * tso['e_rated_fixed_mwh'],
                       'upper_90pct_pickle_tso_mwh': SOC_MAX_FRAC * tso['e_rated_fixed_mwh'],
                       'e_rated_fixed_pickle_dso_mwh': dso['e_rated_fixed_mwh'],
                       'start_and_target_end_soc_mwh': SOC_INIT_FRAC * tso['e_rated_fixed_mwh']},
        'day_end_slack_mwh': {'tso_copy': [tso.get('day_end_slack_up_mwh'), tso.get('day_end_slack_down_mwh')],
                              'dso_copy': [dso.get('day_end_slack_up_mwh'), dso.get('day_end_slack_down_mwh')],
                              'order': '[up, down]'},
        'soc_DERIVED_label': ('DERIVED, not model variables: SoC recursion (eff_ch 0.97, eff_dch 0.96, start 0.5 E^Av '
                              'record) from the ESSO pch / pdch, and from the consensus z net power split as '
                              'pch = max(pnet, 0), pdch = max(-pnet, 0); the ESSO model has no intra-day SoC'),
        'derived_soc_vs_pickle_max_abs_mwh': {
            'from_esso_vs_tso_copy': _maxabs([a - b for a, b in zip(soc_esso, tso['soc_mwh'])]),
            'from_z_vs_tso_copy': _maxabs([a - b for a, b in zip(soc_z, tso['soc_mwh'])])},
        'hours': hours,
    }


# ============================================================================================ negative controls
def nc_wrong_sha():
    wrong = PICKLE_SHA256[:-1] + ('0' if PICKLE_SHA256[-1] != '0' else '1')
    gate = AllowListPickleGate([(PICKLE_REL, wrong)], 'W183e NC1 gate (wrong sha256)', _ORIGINAL_PICKLE['loads'])
    try:
        with open(PICKLE_REL, 'rb') as fh:
            gate.load(fh)
        return {'raised': False, 'delegated': gate.counts['delegated'], 'ok': False}
    except PickleRefused as exc:
        return {'raised': True, 'message': str(exc), 'delegated': gate.counts['delegated'],
                'ok': gate.counts['delegated'] == 0}


def nc_expect_refusal(fn, label):
    before = PICKLE_GATE.counts['delegated']
    try:
        fn()
        return {'control': label, 'raised': False, 'ok': False}
    except PickleRefused as exc:
        return {'control': label, 'raised': True, 'message': str(exc),
                'ok': PICKLE_GATE.counts['delegated'] == before}


def _second_load():
    with open(PICKLE_REL, 'rb') as fh:
        pickle.load(fh)


# ============================================================================================ main
def main():
    args = sys.argv[1:]
    dry = '--dry-run' in args
    out_dir = DEFAULT_OUT_DIR
    if '--out-dir' in args:
        out_dir = args[args.index('--out-dir') + 1]
    out_json = os.path.join(out_dir, 'e6_export.json')
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
        return
    started = datetime.now(timezone.utc).isoformat()
    if dry and os.path.normpath(out_dir) == os.path.normpath(DEFAULT_OUT_DIR):
        _refuse('dry run not into the default output dir', out_dir)
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
    _log(f'repo HEAD {head}; tracked changes: {dirty}; script {script_status}; dry_run {dry}; out_dir {out_dir}')

    chk = capture_checklist()
    if not chk['ok']:
        _refuse('capture-path checklist', f"missing {chk['missing_paths']}; failed {chk['failed_key_checks']}")
    _log(f"capture-path checklist: {chk['n_paths']} paths, {chk['n_key_checks']} key checks, all present")
    jp = verify_jsonl_pins()
    if not jp['ok']:
        _refuse('JSONL sha256 = pin = blob at 741e8ddf = blob at HEAD', json.dumps(jp))
    _log(f"JSONL pins verified: row1 {jp['row1']['working_file'][:8]}, row2 {jp['row2']['working_file'][:8]} "
         f"(= blobs at 741e8ddf and HEAD)")
    pk = verify_pickle_pin()
    if not pk['ok']:
        _refuse('pickle sha256 = pin = post_certification pin', json.dumps(pk))
    _log(f"pickle sha256 verified before load: {pk['sha256_actual_before_load']}")
    if _sha(ESSO_CAPTURE) != ESSO_CAPTURE_SHA256:
        _refuse('ESSO capture sha256 = W182f pin', ESSO_CAPTURE)

    srp1 = _json(SRP1_JSON)
    base = {}
    for k, p in CASE_FILES.items():
        base[k] = float(_json(p)['baseMVA'])
    cl = {no: _json(os.path.join(r['eval'], 'component_levels_terminal.json')) for no, r in ROWS.items()}
    er = {no: _json(os.path.join(r['eval'], 'evaluation_record.json')) for no, r in ROWS.items()}
    sd = {no: _json(os.path.join(r['eval'], 'settling_decision.json')) for no, r in ROWS.items()}
    rb = {no: _jsonl(os.path.join(r['eval'], 'recourse_blocks_all.jsonl')) for no, r in ROWS.items()}
    isd = {no: _json(os.path.join(r['eval'], 'interface_settlement_detail_s31c.json')) for no, r in ROWS.items()}
    w180b = _json(W180B)
    ident = verify_row_identity(er, w180b['rows'])
    if not ident['ok']:
        _refuse('row identity vs W180b', json.dumps(ident))
    stride_last = {}
    for no, r in ROWS.items():
        with open(os.path.join(r['eval'], 'ess_entry_stride_baseline.jsonl'), encoding='utf-8') as fh:
            last = None
            for line in fh:
                if line.strip():
                    last = line
        stride_last[no] = json.loads(last)
        if stride_last[no]['cycle'] != r['k_star']:
            _refuse('stride last line is cycle k*', f"row {no}: {stride_last[no]['cycle']}")
    stride = {no: stride_index(stride_last[no]) for no in ROWS}
    capture = _jsonl(ESSO_CAPTURE)

    # block weights vs the SRP1.json formula
    yrs = list(srp1['Years'])
    wdiff = []
    for no in ROWS:
        for b in cl[no]['blocks'].values():
            f = srp1['Years'][b['year']] * srp1['Days'][b['day']] / (1.0 + srp1['DiscountFactor']) ** (
                int(b['year']) - int(yrs[0]))
            wdiff.append(b['admm_block_weight'] - f)

    sel = day_selection(srp1, cl, rb, sd)
    dy, dd = sel['declared_day']['year'], sel['declared_day']['day']
    _log(f"declared day: {dy} {dd}, contribution {sel['declared_day']['value_contribution']:.2f} EUR; margin to rank 2 "
         f"{sel['margin_rank1_minus_rank2']:.2f}; bar {sel['resolution']['bar_rank1_plus_rank2']:.2f}")
    pb = panel_b(srp1, cl, rb, isd)
    pc = panel_c(er)
    capture_by, esso = esso_section(srp1, capture, stride[2], er[2])
    row1_node7 = [v for (n, _y, _d, _t), e in stride[1].items() if n == STORAGE_NODE
                  for v in (e['z'] + e['x']['tso'] + e['x']['dso'] + e['x']['esso'])]

    # negative controls before the load
    ncs = {'NC1_wrong_sha256_on_allow_list': nc_wrong_sha(),
           'NC2_pickle_loads_bytes': nc_expect_refusal(lambda: pickle.loads(b'\x80\x04N.'), 'pickle.loads(bytes)'),
           'NC3_pickle_load_in_memory_handle': nc_expect_refusal(lambda: pickle.load(io.BytesIO(b'\x80\x04N.')),
                                                                  'pickle.load(BytesIO)')}
    _log(f"negative controls before load: {[(k, v['ok']) for k, v in ncs.items()]}")
    if not all(v['ok'] for v in ncs.values()):
        _refuse('negative controls before the load', json.dumps(ncs))

    pick, table, blocks_supporting = None, None, None
    expected_loads = 0 if dry else 1
    modules_before = set(sys.modules)
    if not dry:
        _log(f'loading {PICKLE_REL} (single declared load)')
        with open(PICKLE_REL, 'rb') as fh:
            models = pickle.load(fh)
        _log(f"loaded; payload keys {sorted(models)}")
        need = ('shared_es_pch', 'shared_es_pdch', 'shared_es_pnet', 'shared_es_qnet', 'shared_es_soc',
                'shared_es_e_rated_fixed', 'shared_es_s_rated_fixed', 'expected_shared_ess_p', 'expected_shared_ess_q')
        tm0 = _by_str(_by_str(models['tso'], yrs[0]), list(srp1['Days'])[0])
        missing = [n for n in need if not hasattr(tm0, n)]
        if missing:
            _refuse('pickle models carry the capture names', missing)
        blocks_supporting, pick = pickle_section(srp1, models, base, stride[2], er[2])
        table = declared_day_table(dy, dd, blocks_supporting, stride[2], capture_by, isd[2], er[2])
        del models
        ncs['NC4_second_load_of_the_allowed_file'] = nc_expect_refusal(_second_load, 'second pickle.load')
        _log(f"NC4: {ncs['NC4_second_load_of_the_allowed_file']['ok']}")
    project_modules_imported = sorted(
        m for m in sys.modules
        if getattr(sys.modules[m], '__file__', None)
        and os.path.dirname(os.path.abspath(sys.modules[m].__file__)) == REPO)
    modules_by_unpickling = sorted(
        m for m in set(sys.modules) - modules_before
        if getattr(sys.modules[m], '__file__', None)
        and os.path.dirname(os.path.abspath(sys.modules[m].__file__)) == REPO)

    gfail = GUARD.verify(0)
    expected_refusals_global = 2 + (0 if dry else 1)
    pickle_ok = (PICKLE_GATE.counts['delegated'] == expected_loads
                 and PICKLE_GATE.counts['refused'] == expected_refusals_global
                 and PICKLE_GATE.counts['unpickler_attempts'] == 0
                 and pickle.load == PICKLE_GATE.load and pickle.loads == PICKLE_GATE.loads)
    build_ok = BUILD_COUNTS['pyomo_model_init'] == 0
    ncs_ok = all(v['ok'] for v in ncs.values())
    guards = {
        'solve_profile_guard': {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_0_failures': gfail},
        'pickle_gate': {'allow_list': [{'path': PICKLE_REL, 'sha256': PICKLE_SHA256}],
                        'counts': dict(PICKLE_GATE.counts), 'expected_delegated_loads': expected_loads,
                        'expected_refusals': expected_refusals_global, 'refusals': PICKLE_GATE.refusals,
                        'loaded': PICKLE_GATE.loaded, 'ok': pickle_ok},
        'build_guard': {'blocked': 'pyomo.core.base.PyomoModel.Model.__init__', 'counts': dict(BUILD_COUNTS),
                        'ok': build_ok},
        'negative_controls': ncs, 'negative_controls_ok': ncs_ok,
        'project_modules_imported': project_modules_imported,
        'project_modules_imported_by_unpickling': modules_by_unpickling,
    }
    if gfail or not pickle_ok or not build_ok or not ncs_ok:
        _log(f'GUARD FAULT: {json.dumps(guards, default=str)}')
        raise SystemExit('guard fault')

    cross = {
        'block_weight_vs_SRP1_formula_max_abs': _maxabs(wdiff),
        'day_selection_identity': sel['identity_check'],
        'panel_b_vs_recourse_blocks_all': pb['cross_check_vs_recourse_blocks_all_line_k_star'],
        'panel_c_published_vs_ageing_trajectory': pc['cross_check_published_vs_ageing_trajectory_e_available'],
        'esso': {k: esso[k] for k in ('pnet_vs_stride_x_esso', 'pch_minus_pdch_minus_pnet')},
        'efc_per_day_max_abs_diff': _maxabs([v['diff_efc'] for v in esso['efc_per_day_recomputed_vs_record'].values()]),
        'efc_stride_line_vs_record_max': (stride_last[2]['efc_per_day_per_node'][str(STORAGE_NODE)]
                                          - er[2]['storage_per_node'][str(STORAGE_NODE)]['efc_per_day_max']),
        'row1_x0_node7_storage_entries_max_abs_mw': _maxabs(row1_node7),
        'n_row1_node7_values': len(row1_node7),
    }
    if pick is not None:
        cross['pickle'] = {k: pick[k] for k in ('summary_over_12_blocks', 'tso_copy_soc_vs_dso_copy_soc_max_abs_mwh',
                                                'tso_other_shared_ess_max_abs')}
        cross['pickle']['e_av_pickle_vs_record'] = {k: pick['e_av_pickle_vs_record'][k]
                                                    for k in ('n', 'max_abs_diff_e_mwh', 'max_abs_diff_s_mva')}
        cross['declared_day_derived_soc_vs_pickle'] = table['derived_soc_vs_pickle_max_abs_mwh']

    result = {
        'schema': 'p515_s53_w183e_e6_export_v1',
        'stage': ('P5.15 Addendum 83 ruling 4 / W183e: E6 data export (zero solves, no build, one guarded pickle read '
                  'of row 2 certified_models.pkl)'),
        'dry_run': dry, 'started_utc': started, 'git_HEAD': head, 'tracked_changes_at_start': dirty,
        'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'git_status': script_status},
        'interpreter': sys.executable,
        'objective_convention': ('Q = certified gross_operational_cost: gross operational cost, contracted interface '
                                 'settlement excluded; one market scenario (deviation settlement and commitment '
                                 'premium identically 0); terminal salvage 0 on both rows'),
        'instance': {f'row{no}': {'cell': r['cell'], 'label': r['label'], 'eval_dir': r['eval'], 'k_star': r['k_star'],
                                  'eval_key': er[no]['eval_key'], 'candidate_key': er[no]['candidate_key'],
                                  'candidate_canonical': er[no]['candidate_canonical'],
                                  'certified_cost_Q': er[no]['certified_cost']} for no, r in ROWS.items()},
        'pre_run_checks': {'capture_checklist': {k: v for k, v in chk.items() if k != 'paths'},
                           'jsonl_pins': jp, 'pickle_pin': pk, 'row_identity': ident},
        'base_mva': {f'{k[0]}|{k[1]}': v for k, v in base.items()},
        'constants': {'eff_ch': EFF_CH, 'eff_dch': EFF_DCH, 'soc_init_frac': SOC_INIT_FRAC,
                      'soc_min_frac': SOC_MIN_FRAC, 'soc_max_frac': SOC_MAX_FRAC, 'period_hours': 1.0,
                      'sources': 'shared_energy_storage.py defaults; definitions.py; asserted in the capture checklist'},
        'day_selection': sel,
        'panel_a_declared_day': table,
        'panel_b_per_block_change_vs_x0': pb,
        'panel_c_e_available': pc,
        'esso_side': esso,
        'pickle_extraction': pick,
        'supporting_node7_all_blocks_row2': blocks_supporting,
        'cross_checks': cross,
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

    # console summary
    print('=== day selection (Q gross, contracted settlement excluded; block-weighted EUR)')
    for e in sel['ranking']:
        print(f"  {e['rank']:>2} {e['year']} {e['day']:<7} c = {e['value_contribution_Q0_minus_Qx']:>12.2f}  "
              f"one-day {e['value_contribution_unweighted_one_day']:>9.2f} (rank {e['rank_by_unweighted_one_day_contribution']})"
              f"  bar {e['bar_sum_of_window_ranges']:>9.2f}  w {e['block_weight']}  next-rank determinate "
              f"{e.get('margin_to_next_rank_not_explained_by_stopping_slack')}")
    print(json.dumps({k: sel[k] for k in ('declared_day', 'margin_rank1_minus_rank2', 'resolution', 'identity_check',
                                          'sum_of_contributions', 'Q_row1_minus_Q_row2')}, indent=1, default=str))
    print('=== panel b horizon totals', json.dumps(pb['horizon_totals_weighted']))
    print('=== cross-checks', json.dumps(cross, indent=1, default=str))
    if table is not None:
        print(f'=== declared day {dy} {dd} node 7 (MW / MWh)')
        print('  h  price   esso_pnet  tso_pnet  dso_pnet   z_p     tso_soc  dso_soc  derived_esso_soc')
        for h in table['hours']:
            print(f"  {h['hour_ending']:>2} {h['market_price_eur_per_mwh']:7.2f} {h['esso']['pnet_mw']:9.5f} "
                  f"{h['tso_copy']['pnet_mw']:9.5f} {h['dso_copy']['pnet_mw']:9.5f} {h['consensus_z']['p_mw']:9.5f} "
                  f"{h['tso_copy']['soc_mwh']:8.5f} {h['dso_copy']['soc_mwh']:8.5f} "
                  f"{h['soc_DERIVED']['from_esso_pch_pdch_mwh']:8.5f}")
        print(json.dumps(table['soc_limits'], indent=1), json.dumps(table['day_end_slack_mwh']))
    print('=== panel c')
    for no in ROWS:
        for n, node in pc[f'row{no}']['nodes'].items():
            if node['has_storage']:
                print(f'  row {no} node {n}', json.dumps(node['years']))
    print('guards:', json.dumps(guards, indent=1, default=str))
    _log(f'wrote {out_json} and {out_man}')


if __name__ == '__main__':
    main()
