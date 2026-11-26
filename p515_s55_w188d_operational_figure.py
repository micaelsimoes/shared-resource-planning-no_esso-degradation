"""
P5.15 Addendum 89, Planner order W188 (d) -- operational figure DATA (the expert chooses the figure; no figure here).
ZERO SOLVES, NO MODEL BUILD, EXACTLY TWO GUARDED PICKLE LOADS (one per allow-listed file; 0 in --dry-run).

PART 1 -- E6 RE-DERIVATION (W183e, `P5_15_W183E_E6_EXPORT.md`; declared day 2025 Spring, panel (b) in EUR)
  The E6 numbers are re-derived HERE, by this script's own code, from the committed records of the two E.21 cells, and
  only then compared with `data/SRP1/Results/P515S55/e6_export/e6_export.json` (read as the comparison target, never
  as an input to a re-derived value):
    row 1  `ref:7aa017f0`  x = 0                    eval d110bd1a5977df1e_x0        k* 181
    row 2  `ref:bd504ecf`  node 7 0.25 MVA / 1 MWh  eval 3f084f2ffaeef2b7_n7_4h_e1  k* 172 (the reference unit)
  Records read per row: component_levels_terminal.json, evaluation_record.json, settling_decision.json,
  recourse_blocks_all.jsonl, interface_settlement_detail_s31c.json; plus data/SRP1/SRP1.json.
  Definitions (as declared in the W183e harness docstring and STEP7 plan section 7; restated operationally):
    Q_block(b)       = sum over C = (generation_cost, flexibility_cost_internal, load_curtailment_cost,
                       res_curtailment_penalty, ess_usage_cost, detector_penalty_total) of
                       component_levels_terminal.blocks[b].weighted[C]   ("unweighted" likewise for the one-day value)
    Q(y, d)          = sum of Q_block over the four network blocks TSO|y|d, DSO|5|y|d, DSO|7|y|d, DSO|9|y|d
    c(y, d)          = Q_row1(y, d) - Q_row2(y, d)                         (EUR; block-weighted, discounted)
    rank             = order of c descending (ties: year, day); declared day = rank 1; share = c(rank 1) / sum c
    range_row(y, d)  = max - min over cycles w0..w1 of the (y, d) sum of recourse_blocks_all.blocks[].value
                       (TSO + DSO agents), [w0, w1] = settling_decision.window of the row
    terminal step    = |value(w1) - value(w1 - 1)| of that sum
    bar(y, d)        = range_row1 + range_row2; next rank determinate iff c(r) - c(r+1) > bar(r) + bar(r+1)
    panel (b)        = per (y, d) and per measure (weighted, unweighted): row 2 minus row 1 of the TSO generation_cost
                       (TN conventional cost), of each DSO flexibility_cost_internal (DN flexibility cost) and their
                       total, of the remaining Q components (DSO generation_cost; load curtailment, RES curtailment,
                       ESS usage, detector penalties over TSO + DSO; TSO flexibility_cost_internal) and of Q;
                       secondary: per DN, the 24-hour sum of the TN-side interface_delta_p (signed and absolute) from
                       interface_settlement_detail_s31c.json
    panel (c)        = per row, node and year: E^Av = storage_per_node[n].published_available_capacity_terminal[y]
                       .e_available, SoH = E^Av / E_rated, floor = minimum_soh x E_rated, margin, 10 / 90 % SoC limits,
                       EFC/day (ageing_trajectory_terminal cell of that year)
  Every compared entry is written to e6_rederivation_comparison.csv (path, re-derived, e6_export, |diff|, relative).

PART 2 -- SoC EXPORT (from the persisted certified models; two guarded pickle reads)
  Allow-list (path, sha256), each pinned in its cell's post_certification.json persisted_models:
    row 1  .../d110bd1a5977df1e_x0/certified_models.pkl         99ab1070b0e61cc7818975ce898069d2060d2668bae67c166b58913e9923c33a
    row 2  .../3f084f2ffaeef2b7_n7_4h_e1/certified_models.pkl   d0a3d512fa58f41053befdb784bb842ad57ec0d71c989aef776a0aa668e6030d
  Row 2 (reference unit, node 7): for the four days of 2025 and of 2035, the hourly SoC of the TSO copy (TSO shared ESS
  index of node 7 = its position in SRP1.json DistributionNetworks) and of the DSO-7 copy (index 0), with pch, pdch,
  pnet, qnet; the model's own SoC limits (pe.value of sess_soc_limit_lower / _upper bounds, per period) and the record's
  10 / 90 % of E^Av; the TSO - DSO copy agreement. Units: MW, MWh (values x baseMVA of the copy's case file).
  CONTACT (declared before the run): an hour is a contact with a limit when |SoC - limit| <= CONTACT_TOL_MWH = 1e-4 MWh
  (the model's limit of that copy); the signed minimum distances are reported beside the counts.
  Row 1 (x = 0): NO zeros are exported. The model is read to state the zero: for every shared ESS of every TSO and DSO
  copy of all 12 blocks, e_rated_fixed and s_rated_fixed, the number of SoC entries, how many are FIXED, the max |value|,
  and the number of ACTIVE sess_soc_* rows (configure_shared_ess_operational_state fixes the variables at 0 and
  deactivates the rows at zero capacity). The same check is applied to row 2's zero-capacity nodes 5 and 9.

PART 3 -- SoH PER BLOCK UNDER THE SEVEN CALIBRATIONS OF TABLE 6 (records only; no pickle)
  Table 6 = `tab:ageing_calibrations` of main.tex at Overleaf a87e195 (clone manuscript/6a67305f25e8348fb71380c3,
  md5 39f6fd86...), rows mapped to the frozen ageing table (frozen_step6_tables_v1_590088fe.json tables.ageing.rows,
  plus cells_appended_w160.e_soh050 for the 0.50 floor row) and through each cell's eval_key to its committed
  evaluation_record.json. Per calibration and representative-year block (5-year blocks 2025 / 2030 / 2035; the model
  carries ONE SoH per cohort and block, not per day): soh_prev_end, soh_end, soh_mid_closed_form,
  soh_used_for_available_energy, e_available, efc_per_day, D (ageing_trajectory_terminal.nodes['7'].cells), the floor
  row's activity at terminal (storage_per_node['7'].soh_floor_rows_active_at_terminal) and the calibration identity
  read from the record: model_variant_applied_in_child.after.per_ess (bus 7, 2025: cl_nom, dod_nom, cl_eff = k,
  phi_cal, soh_min), model_variant.eol_retention_r, model_variant_readback_pre_run.per_node['7'].readback
  (k, phi_cal_in_model, available_energy_soh_point, floor_row_lower = the applied SoH^Min), ageing_enabled.
  Cross-checks (report only): k = cycles_n x dod / (-ln R); the printed Table 6 entries; AE and EFC of the frozen table
  recomputed as pv(.) = sum_y w_y v_y / sum_y w_y, w_y = 1 / 1.02**(y - 2025) (p515_s46_ageing_mechanism.pv, the
  definition quoted in the W142 campaign DEFINITIONS); the frozen binding year; the baseline arm (e_c2_calfade) against
  the W101 reference-unit record (3f084f2f).

GUARDS
  SolveProfileGuard(permitted=()) installed before any project import, verified at exactly 0.
  Pickle: pickle.load / loads / Unpickler replaced for the whole run by an allow-list gate of TWO (path, sha256) pairs;
  refused: a handle without a path, a path off the list, any byte-string loads, any Unpickler, a second load of a file
  already loaded, a handle not at position 0, bytes whose sha256 differs from the listed one (the hashed bytes are the
  bytes unpickled). Exactly 2 delegated loads required (0 in --dry-run). Negative controls (each must raise
  PickleRefused and delegate nothing): NC1 a gate whose allow-list carries a wrong sha256 for row 2's file; NC2
  pickle.loads(bytes); NC3 pickle.load(BytesIO); NC4 an off-list pickle (e_c2_calfade's certified_models.pkl) refused
  on its path, before any byte is read; NC5 (real run) a second load of row 2's file; NC6 (real run) a second load of
  row 1's file.
  Builds: pyomo.core.base.PyomoModel.Model.__init__ blocked and counted (0 required).
  Refuses: an output dir that is missing or holds anything but launch.log; a concurrent copy; a missing capture path
  (checklist before any reading); a pickle whose sha256 differs from its pin; an E6 export that differs from its
  committed blob. JSON through gate_result_io.

OUTPUT (default data/SRP1/Results/P515S55/w188d_operational_figure/): w188d_operational_figure.json, the CSVs listed in
it, manifest_sha256.json; `--post-run` writes manifest_post_run_sha256.json over every output + launch.log.
`--out-dir DIR` redirects (trials); `--dry-run` skips the two loads (expected 0) and refuses the default directory.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir -p data/SRP1/Results/P515S55/w188d_operational_figure && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w188d_operational_figure.py \\
      > data/SRP1/Results/P515S55/w188d_operational_figure/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w188d_operational_figure.py --post-run
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W188d operational figure data (never solves)').install()

import csv  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import math  # noqa: E402

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W101 = os.path.join(P53, 'w101_srp1_continuation')
R1_EVAL = os.path.join(W101, 'campaign_s53_w101_srp1_cont_x0', 'evals', 'd110bd1a5977df1e_x0')
R2_EVAL = os.path.join(W101, 'campaign_s53_w101_srp1_cont_n7_4h_e1', 'evals', '3f084f2ffaeef2b7_n7_4h_e1')
PICKLES = {
    1: (os.path.join(R1_EVAL, 'certified_models.pkl'),
        '99ab1070b0e61cc7818975ce898069d2060d2668bae67c166b58913e9923c33a'),
    2: (os.path.join(R2_EVAL, 'certified_models.pkl'),
        'd0a3d512fa58f41053befdb784bb842ad57ec0d71c989aef776a0aa668e6030d'),
}
W142 = os.path.join(P53, 'w142_resettle_ext_v6')
OFF_LIST_PICKLE = os.path.join(W142, 'campaign_s53_w142_resettle_ext_v6_e_c2_calfade', 'evals',
                               'bbd82994a7ddff94_e_c2_calfade', 'certified_models.pkl')


class PickleRefused(RuntimeError):
    pass


class AllowListPickleGate:
    """Global replacement for pickle.load / pickle.loads: an allow-list of (path, sha256); one load per listed file."""

    def __init__(self, allowed, label, original_loads):
        self.allowed = {os.path.realpath(os.path.join(REPO, p)): s for p, s in allowed}
        self.label = label
        self._orig_loads = original_loads
        self.counts = {'load_attempts': 0, 'loads_attempts': 0, 'unpickler_attempts': 0, 'refused': 0,
                       'delegated': 0}
        self.refusals = []
        self.loaded = []
        self._done = set()

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
        if path in self._done:
            self._refuse('pickle.load', f'{os.path.relpath(path, REPO)} already loaded once (one load per file)')
        if file.tell() != 0:
            self._refuse('pickle.load', f'handle not at position 0 (tell {file.tell()})')
        data = file.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != self.allowed[path]:
            self._refuse('pickle.load', f'sha256 of the bytes read {digest} != allow-listed {self.allowed[path]}')
        self._done.add(path)
        self.counts['delegated'] += 1
        self.loaded.append({'path': os.path.relpath(path, REPO), 'sha256_of_bytes_unpickled': digest,
                            'size_bytes': len(data)})
        return self._orig_loads(data, *args, **kwargs)

    def loads(self, *_a, **_k):
        self.counts['loads_attempts'] += 1
        self._refuse('pickle.loads', 'byte-string loads are not on the allow-list')


_ORIGINAL_PICKLE = {'load': pickle.load, 'loads': pickle.loads, 'Unpickler': pickle.Unpickler}
PICKLE_GATE = AllowListPickleGate([PICKLES[1], PICKLES[2]], 'W188d pickle gate', _ORIGINAL_PICKLE['loads'])


class _BlockedUnpickler(pickle.Unpickler):
    def __init__(self, *_a, **_k):
        PICKLE_GATE.counts['unpickler_attempts'] += 1
        PICKLE_GATE._refuse('pickle.Unpickler', 'Unpickler is not on the allow-list')


pickle.load, pickle.loads, pickle.Unpickler = PICKLE_GATE.load, PICKLE_GATE.loads, _BlockedUnpickler

from pyomo.core.base.PyomoModel import Model as _PyomoModel  # noqa: E402

BUILD_COUNTS = {'pyomo_model_init': 0}


def _blocked_model_init(self, *_a, **_k):
    BUILD_COUNTS['pyomo_model_init'] += 1
    raise RuntimeError('W188d: a Pyomo model build was attempted -- no model builds are permitted')


_PyomoModel.__init__ = _blocked_model_init

import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pyomo.environ as pe  # noqa: E402

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer)

SCRIPT_REL = os.path.basename(__file__)
DEFAULT_OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w188d_operational_figure')
E6_JSON = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e6_export', 'e6_export.json')
E6_COMMIT = 'd794f11b'
FROZEN = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
MANUSCRIPT = os.path.join('manuscript', '6a67305f25e8348fb71380c3', 'main.tex')
MANUSCRIPT_MD5 = '39f6fd86f90b0fbe7c3cc335b1bb793c'
SRP1_JSON = os.path.join('data', 'SRP1', 'SRP1.json')
ROWS = {
    1: {'cell': 'ref:7aa017f0', 'label': 'x0', 'eval': R1_EVAL, 'k_star': 181},
    2: {'cell': 'ref:bd504ecf', 'label': 'n7_4h_e1', 'eval': R2_EVAL, 'k_star': 172},
}
DN_NODES = ('5', '7', '9')
DN_CASE = {'5': 'case33_1', '7': 'case33_2', '9': 'case33_3'}
YEARS = ('2025', '2030', '2035')
SOC_YEARS = ('2025', '2035')
STORAGE_NODE = 7
Q_COMPONENTS = ('generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost', 'res_curtailment_penalty',
                'ess_usage_cost', 'detector_penalty_total')
SOC_MIN_FRAC, SOC_MAX_FRAC, SOC_INIT_FRAC = 0.10, 0.90, 0.50
CONTACT_TOL_MWH = 1e-4
FLOOR_ACTIVE_TOL = 1e-6      # the s35ref capture's activity rule (|SoH - soh_min| <= 1e-6), report-only here
SOC_ROW_NAMES = ('sess_soc_def', 'sess_soc_limit_upper', 'sess_soc_limit_lower', 'sess_soc_final')
OP_VAR_NAMES = ('shared_es_pch', 'shared_es_pdch', 'shared_es_pnet', 'shared_es_qnet', 'shared_es_soc')

# Table 6 (tab:ageing_calibrations, main.tex l. 1324-1330 at a87e195) as printed, mapped to its arm record.
# (label, arm, frozen cell, eval dir, printed (N, delta, R) or None, printed k or None (infinity), printed phi,
#  printed SoH point or None, printed SoH^Min or None)
CALIBRATIONS = [
    ('Baseline (cycling + calendar)', 'C2_calfade', 'e_c2_calfade',
     os.path.join(W142, 'campaign_s53_w142_resettle_ext_v6_e_c2_calfade', 'evals', 'bbd82994a7ddff94_e_c2_calfade'),
     (10000, 0.80, 0.80), 35851, 0.985, 'end', 0.70),
    ('Cycling only', 'C2', 'e_c2',
     os.path.join(W142, 'campaign_s53_w142_resettle_ext_v6_e_c2', 'evals', 'ba969ad21c1831ab_e_c2'),
     (10000, 0.80, 0.80), 35851, 1.000, 'end', 0.70),
    ('Retention 50 %', 'C3_unit', 'e_c3_unit',
     os.path.join(W142, 'campaign_s53_w142_resettle_ext_v6_e_c3_unit', 'evals', '409740fb51f27cab_e_c3_unit'),
     (10000, 0.80, 0.50), 11542, 1.000, 'end', 0.70),
    ('Retention 50 %, mid-block', 'C3_midblock', 'e_c3_midblock',
     os.path.join(W142, 'campaign_s53_w142_resettle_ext_v6_e_c3_midblock', 'evals', '8b1eace91df86cd5_e_c3_midblock'),
     (10000, 0.80, 0.50), 11542, 1.000, 'mid', 0.70),
    ('Datasheet 8 000 cycles', 'C4', 'e_c4',
     os.path.join(W142, 'campaign_s53_w142_resettle_ext_v6_e_c4', 'evals', 'c59d7006b4b9c0a2_e_c4'),
     (8000, 1.00, 0.70), 22429, 1.000, 'end', 0.70),
    ('No ageing', 'no_ageing', 'e_no_ageing',
     os.path.join(W142, 'campaign_s53_w142_resettle_ext_v6_e_no_ageing', 'evals', '1b10e9e161523f46_e_no_ageing'),
     None, None, 1.000, None, None),
    ('Baseline, 0.50 floor', 'soh050', 'e_soh050',
     os.path.join(P53, 'w155_a64_cells', 'campaign_s53_w155_a64_r2_e_soh050', 'evals', '3f01b9eaaab13c82_e_soh050'),
     (10000, 0.80, 0.80), 35851, 0.985, 'end', 0.50),
]
# Table 8 (tab:res_ageing, main.tex l. 1548-1561) as printed: binding year, AE, EFC/day (report-only cross-check)
TABLE8_PRINTED = {'e_c2_calfade': ('2035', 0.793, 0.86), 'e_c2': ('never', 0.888, 1.19),
                  'e_c3_unit': ('2035', 0.786, 0.76), 'e_c3_midblock': ('2035', 0.836, 0.76),
                  'e_c4': ('never', 0.834, 1.13), 'e_no_ageing': ('n/a', 1.000, 1.29),
                  'e_soh050': ('never', 0.778, 1.08)}

INPUTS = {}
_TRACKED = None
OUTPUT_FILES = []


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


def _reg(path, sha=None, status=None):
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel not in INPUTS:
        INPUTS[rel] = {'sha256': sha or _sha(path), 'status': status or _status(path),
                       'size_bytes': os.path.getsize(path)}
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


def _refuse(check, detail):
    _log(f'REFUSED (check: {check}): {detail}')
    raise SystemExit(f'REFUSED (check: {check}): {detail}')


def _yd_order(srp1):
    return [(y, d) for y in srp1['Years'] for d in srp1['Days']]


def _by_str(mapping, key):
    hits = [k for k in mapping if str(k) == str(key)]
    if len(hits) != 1:
        raise KeyError(f'{key!r}: {len(hits)} matching keys in {sorted(map(str, mapping))}')
    return mapping[hits[0]]


# ============================================================================================ pre-run checks
def capture_checklist():
    """Rule eleven: every capture path the export needs exists before any reading (fail fast)."""
    need = {}
    for no, r in ROWS.items():
        for f in ('component_levels_terminal.json', 'evaluation_record.json', 'settling_decision.json',
                  'recourse_blocks_all.jsonl', 'interface_settlement_detail_s31c.json', 'post_certification.json'):
            need[f'row{no} {f}'] = os.path.join(r['eval'], f)
        need[f'row{no} certified_models.pkl'] = PICKLES[no][0]
    for c in CALIBRATIONS:
        need[f'{c[2]} evaluation_record.json'] = os.path.join(c[3], 'evaluation_record.json')
    need['E6 export'] = E6_JSON
    need['frozen step-6 tables'] = FROZEN
    need['manuscript main.tex (clone)'] = MANUSCRIPT
    need['SRP1.json'] = SRP1_JSON
    for y in YEARS:
        need[f'case9 {y}'] = os.path.join('data', 'SRP1', 'case9', f'case9_{y}.json')
        for n, case in DN_CASE.items():
            need[f'{case} {y}'] = os.path.join('data', 'SRP1', case, f'{case}_{y}.json')
    need['production source network.py'] = 'network.py'
    need['production source model_construction_helpers.py'] = 'model_construction_helpers.py'
    missing = sorted(k for k, p in need.items() if not os.path.isfile(p))
    keys = {}
    if not missing:
        for no, r in ROWS.items():
            cl = json.load(open(os.path.join(r['eval'], 'component_levels_terminal.json')))
            keys[f'row{no} component_levels: 48 blocks, every Q component weighted and unweighted, block weight'] = (
                len(cl['blocks']) == 48 and all(all(c in b['weighted'] and c in b['unweighted'] for c in Q_COMPONENTS)
                                                and 'admm_block_weight' in b for b in cl['blocks'].values()))
            keys[f'row{no} component_levels cycles_run == k*'] = cl.get('cycles_run') == r['k_star']
            sd = json.load(open(os.path.join(r['eval'], 'settling_decision.json')))
            keys[f'row{no} settling_decision certified at k* with a window'] = (
                sd.get('status') == 'certified' and sd.get('k_star') == r['k_star'] and len(sd.get('window', [])) == 2)
            er = json.load(open(os.path.join(r['eval'], 'evaluation_record.json')))
            keys[f'row{no} evaluation_record published capacity per node and ageing trajectory'] = (
                all('published_available_capacity_terminal' in er['storage_per_node'][n] for n in DN_NODES)
                and 'nodes' in er.get('ageing_trajectory_terminal', {}))
            pc = json.load(open(os.path.join(r['eval'], 'post_certification.json')))
            keys[f'row{no} post_certification persisted_models pin = allow-list sha256 and path'] = (
                pc['persisted_models']['sha256'] == PICKLES[no][1]
                and os.path.normpath(pc['persisted_models']['path']) == os.path.normpath(PICKLES[no][0]))
            isd = json.load(open(os.path.join(r['eval'], 'interface_settlement_detail_s31c.json')))
            keys[f'row{no} settlement detail: per DN, year, day, 24 periods with delta_p_mw'] = all(
                len(isd['interface_reporting_detail'][n][y][d]['periods']) == 24
                and all('delta_p_mw' in v for v in isd['interface_reporting_detail'][n][y][d]['periods'].values())
                for n in DN_NODES for y in YEARS for d in ('Spring', 'Summer', 'Autumn', 'Winter'))
        need_fields = ('block_year', 'soh_prev_end', 'soh_end', 'soh_mid_closed_form', 'soh_used_for_available_energy',
                       'e_available', 'efc_per_day', 'D', 'phi_cal_in_model', 'cl_eff', 'e_rated', 'n_years')
        for c in CALIBRATIONS:
            er = json.load(open(os.path.join(c[3], 'evaluation_record.json')))
            cells = er.get('ageing_trajectory_terminal', {}).get('nodes', {}).get('7', {}).get('cells', [])
            keys[f'{c[2]} node-7 ageing cells for 2025/2030/2035 with every SoH field'] = (
                [x.get('block_year') for x in cells] == list(YEARS)
                and all(all(f in x for f in need_fields) for x in cells))
            keys[f'{c[2]} model_variant_applied_in_child.after.per_ess carries bus 7 / 2025'] = any(
                e.get('bus') == 7 and str(e.get('year')) == '2025'
                for e in ((er.get('model_variant_applied_in_child') or {}).get('after') or {}).get('per_ess', []))
            keys[f'{c[2]} model_variant_readback_pre_run node 7 readback'] = bool(
                ((er.get('model_variant_readback_pre_run') or {}).get('per_node', {}).get('7') or {}).get('readback'))
            keys[f'{c[2]} status certified'] = er.get('status') == 'certified'
        fz = json.load(open(FROZEN))
        cells = dict(fz['tables']['cells'], **fz['tables']['cells_appended_w160'])
        for c in CALIBRATIONS:
            keys[f'frozen cell {c[2]} present with eval_key'] = c[2] in cells and bool(cells[c[2]].get('eval_key'))
        keys['frozen ageing table rows for the six w142 arms'] = (
            sorted(r['cell'] for r in fz['tables']['ageing']['rows'])
            == sorted(c[2] for c in CALIBRATIONS if c[2] != 'e_soh050'))
        src = {f: open(f, encoding='utf-8').read() for f in ('network.py', 'model_construction_helpers.py')}
        for n in ('model.sess_soc_limit_upper = pe.Constraint', 'model.sess_soc_limit_lower = pe.Constraint',
                  'model.shared_es_soc = pe.Var', 'model.shared_es_e_rated_fixed = pe.Param'):
            keys[f'network.py: {n!r}'] = n in src['network.py']
        for n in ('soc_min = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_MIN_ENERGY_STORED',
                  'soc_max = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_MAX_ENERGY_STORED',
                  'def configure_shared_ess_operational_state(', 'entry.fix(0.0)', 'entry.deactivate()'):
            keys[f'model_construction_helpers.py: {n!r}'] = n in src['model_construction_helpers.py']
    bad_keys = sorted(k for k, ok in keys.items() if not ok)
    return {'ok': not missing and not bad_keys, 'n_paths': len(need), 'missing_paths': missing,
            'n_key_checks': len(keys), 'failed_key_checks': bad_keys, 'paths': need}


def verify_pickle_pins():
    out = {}
    for no, (path, pin) in PICKLES.items():
        pc = json.load(open(os.path.join(ROWS[no]['eval'], 'post_certification.json')))
        actual = _sha(path)
        out[f'row{no}'] = {'path': path, 'pinned_in_task_or_script': pin,
                           'pinned_in_post_certification': pc['persisted_models']['sha256'],
                           'size_pinned': pc['persisted_models']['size_bytes'], 'size_actual': os.path.getsize(path),
                           'sha256_actual_before_load': actual}
        o = out[f'row{no}']
        o['ok'] = actual == pin == o['pinned_in_post_certification'] and o['size_pinned'] == o['size_actual']
        _reg(path, actual)
    out['ok'] = all(v['ok'] for v in out.values())
    return out


def verify_e6_blob():
    work = _sha(E6_JSON)
    blob = hashlib.sha256(_git(['show', f'{E6_COMMIT}:{E6_JSON}'])).hexdigest()
    head = hashlib.sha256(_git(['show', f'HEAD:{E6_JSON}'])).hexdigest()
    _reg(E6_JSON, work)
    return {'path': E6_JSON, 'working_file': work, 'blob_at_d794f11b': blob, 'blob_at_HEAD': head,
            'ok': work == blob == head}


# ============================================================================================ part 1: re-derivation
def _q(block, measure):
    return sum(block[measure][c] for c in Q_COMPONENTS)


def rederive_day_selection(srp1, cl, sd, rb):
    order = _yd_order(srp1)
    identity, qyd, qyd_u, wts = {}, {}, {}, {}
    for no in ROWS:
        tot = 0.0
        qyd[no], qyd_u[no] = {}, {}
        for key, b in cl[no]['blocks'].items():
            yd = (b['year'], b['day'])
            v = _q(b, 'weighted')
            tot += v
            qyd[no][yd] = qyd[no].get(yd, 0.0) + v
            qyd_u[no][yd] = qyd_u[no].get(yd, 0.0) + _q(b, 'unweighted')
            wts.setdefault(yd, set()).add(b['admm_block_weight'])
        g = cl[no]['recourse_components']['gross_operational_cost']
        identity[f'row{no}'] = {'sum_of_48_block_Q': tot, 'gross_operational_cost': g, 'residual': tot - g,
                                'relative_residual': (tot - g) / g}
    rng = {}
    for no in ROWS:
        w0, w1 = sd[no]['window']
        series = {}
        for ln in rb[no]:
            if w0 - 1 <= ln['cycle'] <= w1:
                s = {}
                for b in ln['blocks']:
                    if b['agent'] in ('TSO', 'DSO'):
                        s[(b['year'], b['day'])] = s.get((b['year'], b['day']), 0.0) + b['value']
                series[ln['cycle']] = s
        rng[no] = {'window': [w0, w1], 'missing': [c for c in range(w0 - 1, w1 + 1) if c not in series], 'yd': {}}
        for yd in order:
            vals = [series[c][yd] for c in range(w0, w1 + 1) if c in series]
            rng[no]['yd'][yd] = (max(vals) - min(vals), abs(series[w1][yd] - series[w1 - 1][yd]))
    ranking = []
    for yd in order:
        e = {'year': yd[0], 'day': yd[1], 'block_weight': sorted(wts[yd]),
             'Q_row1_x0_block': qyd[1][yd], 'Q_row2_x_block': qyd[2][yd],
             'value_contribution_Q0_minus_Qx': qyd[1][yd] - qyd[2][yd],
             'value_contribution_unweighted_one_day': qyd_u[1][yd] - qyd_u[2][yd],
             'range_row1_window': rng[1]['yd'][yd][0], 'range_row2_window': rng[2]['yd'][yd][0],
             'terminal_step_row1': rng[1]['yd'][yd][1], 'terminal_step_row2': rng[2]['yd'][yd][1]}
        e['bar_sum_of_window_ranges'] = e['range_row1_window'] + e['range_row2_window']
        ranking.append(e)
    ranking.sort(key=lambda e: (-e['value_contribution_Q0_minus_Qx'], e['year'], e['day']))
    for i, e in enumerate(ranking, 1):
        e['rank'] = i
    for i, e in enumerate(sorted(ranking, key=lambda e: -e['value_contribution_unweighted_one_day']), 1):
        e['rank_by_unweighted_one_day_contribution'] = i
    for a, b in zip(ranking, ranking[1:]):
        a['margin_to_next_rank'] = a['value_contribution_Q0_minus_Qx'] - b['value_contribution_Q0_minus_Qx']
        a['margin_to_next_rank_not_explained_by_stopping_slack'] = (
            a['margin_to_next_rank'] > a['bar_sum_of_window_ranges'] + b['bar_sum_of_window_ranges'])
    total = sum(e['value_contribution_Q0_minus_Qx'] for e in ranking)
    top, second = ranking[0], ranking[1]
    margin = top['value_contribution_Q0_minus_Qx'] - second['value_contribution_Q0_minus_Qx']
    bar = top['bar_sum_of_window_ranges'] + second['bar_sum_of_window_ranges']
    return {
        'identity_check': identity,
        'block_weights_equal_across_networks_and_rows': all(len(v) == 1 for v in wts.values()),
        'ranking': ranking, 'sum_of_contributions': total,
        'Q_row1_minus_Q_row2': (cl[1]['recourse_components']['gross_operational_cost']
                                - cl[2]['recourse_components']['gross_operational_cost']),
        'declared_day': {'year': top['year'], 'day': top['day'], 'value_contribution': top['value_contribution_Q0_minus_Qx'],
                         'share_of_total_value': top['value_contribution_Q0_minus_Qx'] / total},
        'margin_rank1_minus_rank2': margin,
        'resolution': {'bar_rank1_plus_rank2': bar, 'margin_over_bar': margin / bar,
                       'not_explained_by_stopping_slack': margin > bar,
                       'windows': {f'row{no}': rng[no]['window'] for no in ROWS},
                       'missing_cycles_in_windows': {f'row{no}': rng[no]['missing'] for no in ROWS}},
    }


def rederive_panel_b(srp1, cl, rb, isd):
    blocks = []
    for (y, d) in _yd_order(srp1):
        e = {'year': y, 'day': d}
        for m in ('weighted', 'unweighted'):
            t = {no: cl[no]['blocks'][f'TSO|{y}|{d}'][m] for no in ROWS}
            dn = {no: {n: cl[no]['blocks'][f'DSO|{n}|{y}|{d}'][m] for n in DN_NODES} for no in ROWS}
            sec = {'tn_conventional_cost_tso_generation_cost': {
                'row1_x0': t[1]['generation_cost'], 'row2_x': t[2]['generation_cost'],
                'change_x_minus_x0': t[2]['generation_cost'] - t[1]['generation_cost']}}
            flex = {}
            for n in DN_NODES:
                a, b = dn[1][n]['flexibility_cost_internal'], dn[2][n]['flexibility_cost_internal']
                flex[n] = {'row1_x0': a, 'row2_x': b, 'change_x_minus_x0': b - a}
            a = sum(dn[1][n]['flexibility_cost_internal'] for n in DN_NODES)
            b = sum(dn[2][n]['flexibility_cost_internal'] for n in DN_NODES)
            flex['total'] = {'row1_x0': a, 'row2_x': b, 'change_x_minus_x0': b - a}
            sec['dn_flexibility_activation_cost_dso_flexibility_cost_internal'] = flex
            other = {'dso_generation_cost': {'change_x_minus_x0': (
                sum(dn[2][n]['generation_cost'] for n in DN_NODES) - sum(dn[1][n]['generation_cost'] for n in DN_NODES))}}
            for comp in ('load_curtailment_cost', 'res_curtailment_penalty', 'ess_usage_cost', 'detector_penalty_total'):
                v1 = t[1][comp] + sum(dn[1][n][comp] for n in DN_NODES)
                v2 = t[2][comp] + sum(dn[2][n][comp] for n in DN_NODES)
                other[f'{comp}_tso_plus_dso'] = {'change_x_minus_x0': v2 - v1}
            other['tso_flexibility_cost_internal'] = {'change_x_minus_x0': (t[2]['flexibility_cost_internal']
                                                                            - t[1]['flexibility_cost_internal'])}
            sec['other_Q_components'] = other
            q1 = sum(t[1][c] for c in Q_COMPONENTS) + sum(dn[1][n][c] for n in DN_NODES for c in Q_COMPONENTS)
            q2 = sum(t[2][c] for c in Q_COMPONENTS) + sum(dn[2][n][c] for n in DN_NODES for c in Q_COMPONENTS)
            sec['Q_block'] = {'row1_x0': q1, 'row2_x': q2, 'change_x_minus_x0': q2 - q1}
            sec['components_sum_to_Q_change_residual'] = (
                sec['tn_conventional_cost_tso_generation_cost']['change_x_minus_x0'] + flex['total']['change_x_minus_x0']
                + sum(v['change_x_minus_x0'] for v in other.values()) - (q2 - q1))
            e[m] = sec
        iface = {}
        for n in DN_NODES:
            per = {}
            for no in ROWS:
                periods = isd[no]['interface_reporting_detail'][n][y][d]['periods']
                vals = [sum(periods[str(p)]['delta_p_mw'].values()) for p in range(24)]
                per[f'row{no}'] = {'sum_signed_mwh': sum(vals), 'sum_abs_mwh': sum(abs(v) for v in vals)}
            per['change_sum_signed_x_minus_x0'] = per['row2']['sum_signed_mwh'] - per['row1']['sum_signed_mwh']
            per['change_sum_abs_x_minus_x0'] = per['row2']['sum_abs_mwh'] - per['row1']['sum_abs_mwh']
            iface[n] = per
        e['secondary_interface_delta_p_tn_side'] = iface
        blocks.append(e)
    xc = {'tso_generation_cost': [], 'dso_flexibility_cost': [], 'block_q_value': []}
    for no in ROWS:
        last = rb[no][-1]
        if last['cycle'] != ROWS[no]['k_star']:
            _refuse('recourse_blocks_all last line is cycle k*', f"row {no}: {last['cycle']}")

        def key(b):
            return f"TSO|{b['year']}|{b['day']}" if b['agent'] == 'TSO' else f"DSO|{b['node_id']}|{b['year']}|{b['day']}"
        ocb = {key(b): b for b in last['objective_component_blocks']}
        val = {key(b): b['value'] for b in last['blocks'] if b['agent'] in ('TSO', 'DSO')}
        for k, b in cl[no]['blocks'].items():
            xc['block_q_value'].append(_q(b, 'weighted') - val[k])
            if b['kind'] == 'TSO':
                xc['tso_generation_cost'].append(b['weighted']['generation_cost'] - ocb[k]['generation_cost'])
            else:
                xc['dso_flexibility_cost'].append(b['weighted']['flexibility_cost_internal'] - ocb[k]['flexibility_cost'])
    return {
        'blocks': blocks,
        'horizon_totals_weighted': {
            'tn_conventional_cost_change': sum(b['weighted']['tn_conventional_cost_tso_generation_cost']
                                               ['change_x_minus_x0'] for b in blocks),
            'dn_flexibility_cost_change_total': sum(b['weighted']['dn_flexibility_activation_cost_dso_flexibility_cost_'
                                                                  'internal']['total']['change_x_minus_x0'] for b in blocks),
            'Q_change': sum(b['weighted']['Q_block']['change_x_minus_x0'] for b in blocks)},
        'cross_check_vs_recourse_blocks_all_line_k_star': {k: {'n': len(v), 'max_abs_diff_eur': _maxabs(v)}
                                                           for k, v in xc.items()},
    }


def rederive_panel_c(er):
    out, xc = {}, []
    for no in ROWS:
        rec = er[no]
        smin = rec['ess_ageing_baseline']['minimum_soh']
        traj = rec['ageing_trajectory_terminal']['nodes']
        row = {'minimum_soh_record': smin, 'nodes': {}}
        for n in DN_NODES:
            s = rec['storage_per_node'][n]
            er_ = s['e_mwh']
            cells = {c['block_year']: c for c in traj[n]['cells']}
            node = {'e_rated_mwh': er_, 's_rated_mva': s['s_mva'], 'has_storage': s['has_storage'],
                    'soh_floor_rows_active_at_terminal': s['soh_floor_rows_active_at_terminal'], 'years': {}}
            for y in YEARS:
                cap = s['published_available_capacity_terminal'][y]
                ea = cap['e_available']
                node['years'][y] = {
                    'e_available_mwh': ea, 's_available_mva': cap['s_available'],
                    'soh': ea / er_ if er_ else None, 'floor_e_mwh_0p70_x_e_rated': smin * er_,
                    'margin_above_floor_mwh': ea - smin * er_ if er_ else None,
                    'soc_lower_10pct_mwh': SOC_MIN_FRAC * ea, 'soc_upper_90pct_mwh': SOC_MAX_FRAC * ea,
                    'efc_per_day': cells[y]['efc_per_day'] if y in cells else None}
                if y in cells:
                    xc.append(ea - cells[y]['e_available'])
            row['nodes'][n] = node
        out[f'row{no}'] = row
    out['minimum_soh_declared'] = 0.70
    out['minimum_soh_record_equals_declared'] = all(out[f'row{no}']['minimum_soh_record'] == 0.70 for no in ROWS)
    out['cross_check_published_vs_ageing_trajectory_e_available'] = {'n': len(xc), 'max_abs_diff_mwh': _maxabs(xc)}
    return out


def _flatten(obj, prefix, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f'{prefix}.{k}' if prefix else str(k), out)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, dict) and 'year' in x and 'day' in x for x in obj):
            for x in obj:
                _flatten({k: v for k, v in x.items() if k not in ('year', 'day')}, f"{prefix}[{x['year']}|{x['day']}]",
                         out)
        else:
            for i, x in enumerate(obj):
                _flatten(x, f'{prefix}[{i}]', out)
    else:
        out[prefix] = obj


def compare_with_e6(mine, e6):
    rows, missing_in_e6 = [], []
    for section, theirs in (('day_selection', e6['day_selection']),
                            ('panel_b_per_block_change_vs_x0', e6['panel_b_per_block_change_vs_x0']),
                            ('panel_c_e_available', e6['panel_c_e_available'])):
        a, b = {}, {}
        _flatten(mine[section], section, a)
        _flatten(theirs, section, b)
        for path, va in a.items():
            if path not in b:
                missing_in_e6.append(path)
                continue
            vb = b[path]
            if isinstance(va, bool) or isinstance(vb, bool) or va is None or vb is None or isinstance(va, str):
                rows.append({'path': path, 'rederived': va, 'e6_export': vb, 'abs_diff': None, 'rel_diff': None,
                             'equal': va == vb, 'kind': 'exact'})
            else:
                d = abs(va - vb)
                rows.append({'path': path, 'rederived': va, 'e6_export': vb, 'abs_diff': d,
                             'rel_diff': d / abs(vb) if vb else (0.0 if d == 0 else None), 'equal': va == vb,
                             'kind': 'numeric'})
        not_rederived = sorted(p for p, v in b.items() if p not in a and isinstance(v, (int, float))
                               and not isinstance(v, bool))
        rows_nr = [{'path': p, 'rederived': None, 'e6_export': b[p], 'abs_diff': None, 'rel_diff': None,
                    'equal': None, 'kind': 'e6 numeric entry not re-derived'} for p in not_rederived]
        rows.extend(rows_nr)
    num = [r for r in rows if r['kind'] == 'numeric']
    exact = [r for r in rows if r['kind'] == 'exact']
    nonzero = [r for r in num if r['abs_diff'] != 0]
    summary = {
        'n_compared_numeric': len(num), 'n_compared_exact': len(exact),
        'n_numeric_bitwise_equal': sum(1 for r in num if r['equal']),
        'n_numeric_nonzero_diff': len(nonzero),
        'max_abs_diff': max((r['abs_diff'] for r in num), default=None),
        'max_rel_diff': max((r['rel_diff'] for r in num if r['rel_diff'] is not None), default=None),
        'largest_nonzero': sorted(nonzero, key=lambda r: -r['abs_diff'])[:10],
        'n_exact_unequal': sum(1 for r in exact if not r['equal']),
        'exact_unequal': [r for r in exact if not r['equal']],
        'n_rederived_paths_absent_in_e6': len(missing_in_e6), 'rederived_paths_absent_in_e6': missing_in_e6,
        'n_e6_numeric_entries_not_rederived': sum(1 for r in rows if r['kind'] == 'e6 numeric entry not re-derived'),
    }
    return rows, summary


# ============================================================================================ part 2: pickles
def _first(s):
    items = list(s)
    if len(items) != 1:
        raise RuntimeError(f'expected exactly one scenario, got {len(items)}')
    return items[0]


def _val(c):
    v = pe.value(c, exception=False)
    return None if v is None else float(v)


def _entries_of(component, e):
    for idx in component:
        first = idx[0] if isinstance(idx, tuple) else idx
        if first == e:
            yield component[idx]


def zero_capacity_state(model, e, label):
    """The model's own statement of an inactive shared ESS: capacity params, fixed operational variables, rows."""
    out = {'copy': label, 'ess_index': e, 'e_rated_fixed_pu': _val(model.shared_es_e_rated_fixed[e]),
           's_rated_fixed_pu': _val(model.shared_es_s_rated_fixed[e])}
    for name in OP_VAR_NAMES:
        ents = list(_entries_of(getattr(model, name), e))
        vals = [abs(_val(x) or 0.0) for x in ents]
        out[f'{name}_n'] = len(ents)
        out[f'{name}_n_fixed'] = sum(1 for x in ents if x.fixed)
        out[f'{name}_max_abs_pu'] = max(vals) if vals else None
        out[f'{name}_n_value_none'] = sum(1 for x in ents if x.value is None)
    n_rows = n_active = 0
    for name in SOC_ROW_NAMES:
        ents = list(_entries_of(getattr(model, name), e))
        n_rows += len(ents)
        n_active += sum(1 for x in ents if x.active)
    out['soc_rows_n'] = n_rows
    out['soc_rows_n_active'] = n_active
    out['inactive_by_model'] = (out['e_rated_fixed_pu'] == 0.0 and out['s_rated_fixed_pu'] == 0.0
                                and all(out[f'{n}_n_fixed'] == out[f'{n}_n'] and out[f'{n}_max_abs_pu'] == 0.0
                                        for n in OP_VAR_NAMES)
                                and n_active == 0)
    return out


def extract_active(model, e, base, kind):
    sm, so = _first(model.scenarios_market), _first(model.scenarios_operation)
    periods = list(model.periods)
    out = {'e_rated_fixed_mwh': _val(model.shared_es_e_rated_fixed[e]) * base,
           's_rated_fixed_mva': _val(model.shared_es_s_rated_fixed[e]) * base, 'base_mva': base, 'periods': periods}
    for name, key in (('shared_es_pch', 'pch_mw'), ('shared_es_pdch', 'pdch_mw'), ('shared_es_pnet', 'pnet_mw'),
                      ('shared_es_qnet', 'qnet_mvar'), ('shared_es_soc', 'soc_mwh')):
        var = getattr(model, name)
        out[key] = [_val(var[e, sm, so, p]) * base for p in periods]
    lo, up, act = [], [], []
    for p in periods:
        cl_ = model.sess_soc_limit_lower[e, sm, so, p]
        cu_ = model.sess_soc_limit_upper[e, sm, so, p]
        lo.append(_val(cl_.lower) * base)
        up.append(_val(cu_.upper) * base)
        act.append(bool(cl_.active and cu_.active))
    out['soc_lower_model_mwh'], out['soc_upper_model_mwh'], out['soc_limit_rows_active'] = lo, up, act
    out['soc_vars_fixed'] = sum(1 for p in periods if model.shared_es_soc[e, sm, so, p].fixed)
    if hasattr(model, 'slack_shared_es_soc_final_up'):
        out['day_end_slack_up_mwh'] = _val(model.slack_shared_es_soc_final_up[e, sm, so]) * base
        out['day_end_slack_down_mwh'] = _val(model.slack_shared_es_soc_final_down[e, sm, so]) * base
    out['kind'] = kind
    return out


def read_row2(models, srp1, base):
    dn_order = [str(dn['connection_node_id']) for dn in srp1['DistributionNetworks']]
    e7 = dn_order.index(str(STORAGE_NODE))
    blocks, zero_checks = {}, []
    for y in YEARS:
        for d in srp1['Days']:
            tm = _by_str(_by_str(models['tso'], y), d)
            caps = {i: _val(tm.shared_es_e_rated_fixed[i]) for i in tm.shared_energy_storages}
            nonzero = sorted(i for i, v in caps.items() if v and abs(v) > 0)
            if nonzero != [e7]:
                _refuse('row 2: exactly one TSO shared ESS with capacity, at node 7\'s index', f'{y} {d}: {caps}')
            for i in tm.shared_energy_storages:
                if i != e7:
                    zc = zero_capacity_state(tm, i, f'TSO|{y}|{d}')
                    zc.update({'row': 2, 'node': dn_order[i]})
                    zero_checks.append(zc)
            for n in DN_NODES:
                dm = _by_str(_by_str(_by_str(models['dso'], n), y), d)
                idx = list(dm.shared_energy_storages)
                if n != str(STORAGE_NODE):
                    for i in idx:
                        zc = zero_capacity_state(dm, i, f'DSO{n}|{y}|{d}')
                        zc.update({'row': 2, 'node': n})
                        zero_checks.append(zc)
            if y in SOC_YEARS:
                dm7 = _by_str(_by_str(_by_str(models['dso'], str(STORAGE_NODE)), y), d)
                if list(dm7.shared_energy_storages) != [0]:
                    _refuse('row 2: DSO-7 copy has one shared ESS at index 0', list(dm7.shared_energy_storages))
                blocks[(y, d)] = {'tso': extract_active(tm, e7, base[('TSO', y)], 'TSO'),
                                  'dso': extract_active(dm7, 0, base[('7', y)], 'DSO7')}
    return e7, blocks, zero_checks


def read_row1(models, srp1):
    dn_order = [str(dn['connection_node_id']) for dn in srp1['DistributionNetworks']]
    checks = []
    for y in YEARS:
        for d in srp1['Days']:
            tm = _by_str(_by_str(models['tso'], y), d)
            for i in tm.shared_energy_storages:
                zc = zero_capacity_state(tm, i, f'TSO|{y}|{d}')
                zc.update({'row': 1, 'node': dn_order[i]})
                checks.append(zc)
            for n in DN_NODES:
                dm = _by_str(_by_str(_by_str(models['dso'], n), y), d)
                for i in dm.shared_energy_storages:
                    zc = zero_capacity_state(dm, i, f'DSO{n}|{y}|{d}')
                    zc.update({'row': 1, 'node': n})
                    checks.append(zc)
    return checks


def soc_tables(blocks, er2, e6):
    pub = er2['storage_per_node'][str(STORAGE_NODE)]['published_available_capacity_terminal']
    hourly, summary, limits = [], [], []
    e6_sup = e6.get('supporting_node7_all_blocks_row2') or {}
    e6_diffs = []
    for (y, d), cp in blocks.items():
        ea = pub[y]['e_available']
        rec_lo, rec_up = SOC_MIN_FRAC * ea, SOC_MAX_FRAC * ea
        t, s = cp['tso'], cp['dso']
        for p in range(24):
            hourly.append({
                'year': y, 'day': d, 'period': p, 'hour_ending': p + 1,
                'soc_tso_copy_mwh': t['soc_mwh'][p], 'soc_dso_copy_mwh': s['soc_mwh'][p],
                'soc_tso_minus_dso_mwh': t['soc_mwh'][p] - s['soc_mwh'][p],
                'pch_tso_mw': t['pch_mw'][p], 'pdch_tso_mw': t['pdch_mw'][p], 'pnet_tso_mw': t['pnet_mw'][p],
                'qnet_tso_mvar': t['qnet_mvar'][p],
                'pch_dso_mw': s['pch_mw'][p], 'pdch_dso_mw': s['pdch_mw'][p], 'pnet_dso_mw': s['pnet_mw'][p],
                'qnet_dso_mvar': s['qnet_mvar'][p],
                'soc_lower_model_tso_mwh': t['soc_lower_model_mwh'][p], 'soc_upper_model_tso_mwh': t['soc_upper_model_mwh'][p],
                'soc_lower_model_dso_mwh': s['soc_lower_model_mwh'][p], 'soc_upper_model_dso_mwh': s['soc_upper_model_mwh'][p],
                'soc_lower_record_mwh': rec_lo, 'soc_upper_record_mwh': rec_up,
                'contact_lower_tso_flag': int(abs(t['soc_mwh'][p] - t['soc_lower_model_mwh'][p]) <= CONTACT_TOL_MWH),
                'contact_upper_tso_flag': int(abs(t['soc_upper_model_mwh'][p] - t['soc_mwh'][p]) <= CONTACT_TOL_MWH),
                'contact_lower_dso_flag': int(abs(s['soc_mwh'][p] - s['soc_lower_model_mwh'][p]) <= CONTACT_TOL_MWH),
                'contact_upper_dso_flag': int(abs(s['soc_upper_model_mwh'][p] - s['soc_mwh'][p]) <= CONTACT_TOL_MWH),
            })
        row = {'year': y, 'day': d, 'e_available_record_mwh': ea,
               'e_rated_fixed_tso_copy_mwh': t['e_rated_fixed_mwh'], 'e_rated_fixed_dso_copy_mwh': s['e_rated_fixed_mwh'],
               'soc_start_model_tso_mwh': SOC_INIT_FRAC * t['e_rated_fixed_mwh'],
               'soc_start_model_dso_mwh': SOC_INIT_FRAC * s['e_rated_fixed_mwh']}
        for side, c in (('tso', t), ('dso', s)):
            soc = c['soc_mwh']
            row[f'soc_min_{side}_mwh'] = min(soc)
            row[f'soc_max_{side}_mwh'] = max(soc)
            row[f'min_soc_minus_lower_model_{side}_mwh'] = min(a - b for a, b in zip(soc, c['soc_lower_model_mwh']))
            row[f'min_upper_model_minus_soc_{side}_mwh'] = min(b - a for a, b in zip(soc, c['soc_upper_model_mwh']))
            row[f'n_contact_lower_{side}'] = sum(1 for a, b in zip(soc, c['soc_lower_model_mwh'])
                                                 if abs(a - b) <= CONTACT_TOL_MWH)
            row[f'n_contact_upper_{side}'] = sum(1 for a, b in zip(soc, c['soc_upper_model_mwh'])
                                                 if abs(b - a) <= CONTACT_TOL_MWH)
            row[f'day_end_soc_minus_start_{side}_mwh'] = soc[-1] - SOC_INIT_FRAC * c['e_rated_fixed_mwh']
            row[f'energy_charged_{side}_mwh'] = sum(c['pch_mw'])
            row[f'energy_discharged_{side}_mwh'] = sum(c['pdch_mw'])
            row[f'max_min_pch_pdch_{side}_mw'] = max(min(a, b) for a, b in zip(c['pch_mw'], c['pdch_mw']))
            row[f'soc_limit_rows_active_{side}'] = sum(c['soc_limit_rows_active'])
            row[f'soc_vars_fixed_{side}'] = c['soc_vars_fixed']
            row[f'model_lower_minus_10pct_e_rated_fixed_{side}_maxabs'] = _maxabs(
                [lo - SOC_MIN_FRAC * c['e_rated_fixed_mwh'] for lo in c['soc_lower_model_mwh']])
            row[f'model_upper_minus_90pct_e_rated_fixed_{side}_maxabs'] = _maxabs(
                [u - SOC_MAX_FRAC * c['e_rated_fixed_mwh'] for u in c['soc_upper_model_mwh']])
            row[f'day_end_slack_up_{side}_mwh'] = c.get('day_end_slack_up_mwh')
            row[f'day_end_slack_down_{side}_mwh'] = c.get('day_end_slack_down_mwh')
        row['soc_tso_minus_dso_maxabs_mwh'] = _maxabs([a - b for a, b in zip(t['soc_mwh'], s['soc_mwh'])])
        row['pnet_tso_minus_dso_maxabs_mw'] = _maxabs([a - b for a, b in zip(t['pnet_mw'], s['pnet_mw'])])
        row['e_rated_fixed_tso_minus_record_mwh'] = t['e_rated_fixed_mwh'] - ea
        row['e_rated_fixed_dso_minus_record_mwh'] = s['e_rated_fixed_mwh'] - ea
        sup = e6_sup.get(f'{y}|{d}')
        if sup:
            dd = []
            for side, key in (('tso', 'tso_copy'), ('dso', 'dso_copy')):
                for f in ('soc_mwh', 'pch_mw', 'pdch_mw', 'pnet_mw', 'qnet_mvar'):
                    dd.extend(a - b for a, b in zip(cp[side][f], sup[key][f]))
            row['vs_e6_supporting_same_pickle_maxabs'] = _maxabs(dd)
            e6_diffs.extend(dd)
        summary.append(row)
    for y in SOC_YEARS:
        ea = pub[y]['e_available']
        bl = [cp for (yy, _d), cp in blocks.items() if yy == y]
        limits.append({'row': 2, 'node': STORAGE_NODE, 'year': y, 'e_rated_mwh': 1.0, 'e_available_record_mwh': ea,
                       'soh_end_of_block_record': ea / 1.0,
                       'soc_lower_10pct_record_mwh': SOC_MIN_FRAC * ea, 'soc_upper_90pct_record_mwh': SOC_MAX_FRAC * ea,
                       'soc_start_and_target_end_50pct_record_mwh': SOC_INIT_FRAC * ea,
                       'e_rated_fixed_model_min_over_copies_and_days_mwh': min(min(c['tso']['e_rated_fixed_mwh'],
                                                                                   c['dso']['e_rated_fixed_mwh']) for c in bl),
                       'e_rated_fixed_model_max_over_copies_and_days_mwh': max(max(c['tso']['e_rated_fixed_mwh'],
                                                                                   c['dso']['e_rated_fixed_mwh']) for c in bl),
                       'soc_lower_model_min_mwh': min(min(c[s]['soc_lower_model_mwh']) for c in bl for s in ('tso', 'dso')),
                       'soc_upper_model_max_mwh': max(max(c[s]['soc_upper_model_mwh']) for c in bl for s in ('tso', 'dso'))})
    return hourly, summary, limits, {'n': len(e6_diffs), 'max_abs': _maxabs(e6_diffs)}


# ============================================================================================ part 3: SoH
def _pv(values):
    w = [1.0 / 1.02 ** (int(y) - 2025) for y in YEARS]
    return sum(a * v for a, v in zip(w, values)) / sum(w)


def soh_tables(fz):
    cells_fz = dict(fz['tables']['cells'], **fz['tables']['cells_appended_w160'])
    ageing_rows = {r['cell']: r for r in fz['tables']['ageing']['rows']}
    per_block, identity = [], []
    traj_by_cell = {}
    for (label, arm, cell, ev, printed_nrd, printed_k, printed_phi, printed_point, printed_min) in CALIBRATIONS:
        er = _json(os.path.join(ev, 'evaluation_record.json'))
        mv = er.get('model_variant') or {}
        after = [e for e in er['model_variant_applied_in_child']['after']['per_ess']
                 if e.get('bus') == 7 and str(e.get('year')) == '2025'][0]
        rb = er['model_variant_readback_pre_run']['per_node']['7']['readback']
        rbt = (er.get('model_variant_readback_terminal') or {}).get('per_node', {}).get('7', {}).get('readback') or {}
        traj = er['ageing_trajectory_terminal']
        cells = traj['nodes']['7']['cells']
        traj_by_cell[cell] = cells
        active = er['storage_per_node']['7'].get('soh_floor_rows_active_at_terminal') or []
        soh_min = rb['floor_row_lower']
        enabled = bool(traj.get('ageing_enabled'))
        r_eol = mv.get('eol_retention_r')
        k_formula = after['cl_nom'] * after['dod_nom'] / (-math.log(r_eol)) if r_eol else None
        fzc = cells_fz[cell]
        ar = ageing_rows.get(cell, {})
        floor_years = [y for i, y in enumerate(YEARS) if f'(0, {i})' in active]
        ae = _pv([c['soh_used_for_available_energy'] for c in cells])
        efc = _pv([c['efc_per_day'] for c in cells])
        t8 = TABLE8_PRINTED[cell]
        ident = {
            'calibration_table6': label, 'arm': arm, 'cell': cell, 'eval_dir': ev, 'eval_key': er['eval_key'],
            'eval_key_frozen': fzc.get('eval_key'), 'eval_key_matches_frozen': er['eval_key'] == fzc.get('eval_key'),
            'status': er['status'], 'k_star': er.get('certification_cycle'), 'k_star_frozen': fzc.get('k_star'),
            'candidate_canonical_node7': json.dumps(er['candidate_canonical']['nodes'].get('7')),
            'investment_year': er['candidate_canonical'].get('investment_year'),
            'ageing_enabled': enabled,
            'cycle_count_N_entered': after['cl_nom'], 'dod_delta_entered': after['dod_nom'],
            'end_retention_R_entered': r_eol,
            'k_record_cl_eff': after['cl_eff'], 'k_readback_in_model': rb.get('k'),
            'k_formula_N_delta_over_minus_ln_R': k_formula,
            'k_formula_minus_record': (k_formula - after['cl_eff']) if k_formula else None,
            'calendar_retention_phi_cal_record': after['phi_cal'], 'phi_cal_in_model_readback': rb.get('phi_cal_in_model'),
            'soh_point_readback': rb.get('available_energy_soh_point'), 'soh_point_trajectory': traj.get('available_energy_soh_point'),
            'soh_min_applied_floor_row_lower': soh_min,
            'soh_min_terminal_readback': rbt.get('floor_row_lower'),
            'table6_N_delta_R_printed': json.dumps(list(printed_nrd)) if printed_nrd else 'n/a',
            'table6_k_printed': printed_k if printed_k is not None else 'infinity',
            'table6_phi_printed': printed_phi, 'table6_soh_point_printed': printed_point or 'n/a',
            'table6_soh_min_printed': printed_min if printed_min is not None else 'n/a',
            'table6_k_matches_record_rounded': (printed_k == round(after['cl_eff'])) if (printed_k and enabled) else (not enabled),
            'table6_phi_matches_record': abs(printed_phi - after['phi_cal']) < 1e-12,
            'table6_point_matches_record': (printed_point == rb.get('available_energy_soh_point')) if printed_point else (not enabled),
            'table6_soh_min_matches_record': (abs(printed_min - soh_min) < 1e-12) if printed_min is not None else (not enabled),
            'floor_rows_active_at_terminal': json.dumps(active), 'binding_year_record': floor_years[0] if floor_years else 'never',
            'binding_year_frozen': (ar.get('floor_year_070') if cell != 'e_soh050' else None),
            'binding_year_table8_printed': t8[0],
            'AE_recomputed_pv': ae, 'AE_frozen': ar.get('AE'), 'AE_table8_printed': t8[1],
            'EFC_recomputed_pv': efc, 'EFC_frozen': ar.get('EFC'), 'EFC_table8_printed': t8[2],
            'value_minus_I_frozen_eur': ar.get('value_minus_I'), 'certified_Q_record': er.get('certified_cost'),
        }
        ident['AE_recomputed_minus_frozen'] = (ae - ar['AE']) if ar.get('AE') is not None else None
        ident['EFC_recomputed_minus_frozen'] = (efc - ar['EFC']) if ar.get('EFC') is not None else None
        ident['AE_rounded_3_matches_table8'] = round(ae, 3) == t8[1]
        ident['EFC_rounded_2_matches_table8'] = round(efc, 2) == t8[2]
        identity.append(ident)
        for i, c in enumerate(cells):
            per_block.append({
                'calibration_table6': label, 'arm': arm, 'cell': cell, 'eval_key16': er['eval_key'][:16],
                'node': 7, 'cohort_investment_year': c['investment_year'], 'block_year': c['block_year'],
                'block_years_covered': f"{c['block_year']}-{int(c['block_year']) + c['n_years'] - 1}",
                'n_years': c['n_years'], 'e_rated_mwh': c['e_rated'],
                'soh_start_of_block': c['soh_prev_end'], 'soh_end_of_block': c['soh_end'],
                'soh_mid_closed_form': c['soh_mid_closed_form'],
                'soh_used_for_available_energy': c['soh_used_for_available_energy'],
                'e_available_mwh': c['e_available'], 'efc_per_day': c['efc_per_day'],
                'cycling_degradation_D': c['D'], 'phi_cal_in_model': c['phi_cal_in_model'], 'k_cl_eff': c['cl_eff'],
                'ageing_enabled': int(enabled), 'soh_point': traj.get('available_energy_soh_point'),
                'soh_min_applied': soh_min, 'floor_row_active_at_terminal_flag': int(f'(0, {i})' in active),
                'soh_end_minus_soh_min': c['soh_end'] - soh_min,
            })
    w101 = _json(os.path.join(R2_EVAL, 'evaluation_record.json'))
    w_cells = w101['ageing_trajectory_terminal']['nodes']['7']['cells']
    diffs = []
    for a, b in zip(traj_by_cell['e_c2_calfade'], w_cells):
        for f in ('soh_prev_end', 'soh_end', 'soh_mid_closed_form', 'soh_used_for_available_energy', 'e_available',
                  'efc_per_day', 'D'):
            diffs.append(a[f] - b[f])
    baseline_vs_w101 = {'fields': 7, 'blocks': len(w_cells), 'max_abs_diff': _maxabs(diffs),
                        'certified_Q_calfade': next(i for i in identity if i['cell'] == 'e_c2_calfade')['certified_Q_record'],
                        'certified_Q_w101_row2': w101.get('certified_cost')}
    return per_block, identity, baseline_vs_w101


# ============================================================================================ output helpers
def _write_csv(out_dir, name, rows):
    path = os.path.join(out_dir, name)
    if not rows:
        raise RuntimeError(f'{name}: no rows')
    cols = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, 'x', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: (int(v) if isinstance(v, bool) else repr(v) if isinstance(v, float)
                        else ('' if v is None else v)) for k, v in r.items()})
    OUTPUT_FILES.append(path)
    return {'path': path, 'n_rows': len(rows), 'columns': cols}


# ============================================================================================ negative controls
def nc_wrong_sha():
    path, pin = PICKLES[2]
    wrong = pin[:-1] + ('0' if pin[-1] != '0' else '1')
    gate = AllowListPickleGate([(path, wrong)], 'W188d NC1 gate (wrong sha256)', _ORIGINAL_PICKLE['loads'])
    try:
        with open(path, 'rb') as fh:
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


def _load_path(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# ============================================================================================ main
def main():
    args = sys.argv[1:]
    dry = '--dry-run' in args
    out_dir = DEFAULT_OUT_DIR
    if '--out-dir' in args:
        out_dir = args[args.index('--out-dir') + 1]
    out_json = os.path.join(out_dir, 'w188d_operational_figure.json')
    out_man = os.path.join(out_dir, 'manifest_sha256.json')
    out_post = os.path.join(out_dir, 'manifest_post_run_sha256.json')
    if '--post-run' in args:
        man_in = json.load(open(out_man))
        rels = sorted(man_in['outputs']) + [out_man, os.path.join(out_dir, 'launch.log')]
        for r in rels:
            if not os.path.exists(r):
                raise SystemExit(f'post-run: {r} missing')
        if os.path.exists(out_post):
            raise SystemExit(f'post-run: {out_post} exists (write-once)')
        bad = [r for r, s in man_in['outputs'].items() if _sha(r) != s]
        if bad:
            raise SystemExit(f'post-run: outputs changed since the manifest: {bad}')
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
    pk = verify_pickle_pins()
    if not pk['ok']:
        _refuse('pickle sha256 = pin = post_certification pin', json.dumps(pk))
    _log(f"pickle pins verified before load: row1 {pk['row1']['sha256_actual_before_load'][:8]}, "
         f"row2 {pk['row2']['sha256_actual_before_load'][:8]}")
    e6b = verify_e6_blob()
    if not e6b['ok']:
        _refuse('e6_export.json = blob at d794f11b = blob at HEAD', json.dumps(e6b))
    ms = hashlib.md5(open(MANUSCRIPT, 'rb').read()).hexdigest()
    _reg(MANUSCRIPT, status='clone (not in this repository)')
    if ms != MANUSCRIPT_MD5:
        _refuse('manuscript main.tex md5 = a87e195', ms)

    srp1 = _json(SRP1_JSON)
    base = {}
    for y in YEARS:
        base[('TSO', y)] = float(_json(os.path.join('data', 'SRP1', 'case9', f'case9_{y}.json'))['baseMVA'])
        for n, case in DN_CASE.items():
            base[(n, y)] = float(_json(os.path.join('data', 'SRP1', case, f'{case}_{y}.json'))['baseMVA'])
    cl = {no: _json(os.path.join(r['eval'], 'component_levels_terminal.json')) for no, r in ROWS.items()}
    er = {no: _json(os.path.join(r['eval'], 'evaluation_record.json')) for no, r in ROWS.items()}
    sd = {no: _json(os.path.join(r['eval'], 'settling_decision.json')) for no, r in ROWS.items()}
    rb = {no: _jsonl(os.path.join(r['eval'], 'recourse_blocks_all.jsonl')) for no, r in ROWS.items()}
    isd = {no: _json(os.path.join(r['eval'], 'interface_settlement_detail_s31c.json')) for no, r in ROWS.items()}
    for no, r in ROWS.items():
        if er[no]['certification_cycle'] != r['k_star'] or er[no]['status'] != 'certified':
            _refuse('row identity (certified at k*)', f'row {no}')

    # ---- part 1: re-derive, then compare
    mine = {'day_selection': rederive_day_selection(srp1, cl, sd, rb),
            'panel_b_per_block_change_vs_x0': rederive_panel_b(srp1, cl, rb, isd),
            'panel_c_e_available': rederive_panel_c(er)}
    ds = mine['day_selection']
    _log(f"re-derived declared day {ds['declared_day']['year']} {ds['declared_day']['day']} "
         f"{ds['declared_day']['value_contribution']:.2f} EUR, share {ds['declared_day']['share_of_total_value']:.4f}, "
         f"total {ds['sum_of_contributions']:.2f}")
    with open(E6_JSON, encoding='utf-8') as fh:
        e6 = json.load(fh)
    cmp_rows, cmp_summary = compare_with_e6(mine, e6)
    _log(f"E6 comparison: {cmp_summary['n_compared_numeric']} numeric ({cmp_summary['n_numeric_bitwise_equal']} bitwise "
         f"equal, max |diff| {cmp_summary['max_abs_diff']}), {cmp_summary['n_compared_exact']} exact "
         f"({cmp_summary['n_exact_unequal']} unequal)")

    # ---- part 3: SoH (records only)
    fz = _json(FROZEN)
    soh_rows, soh_ident, base_vs_w101 = soh_tables(fz)
    _log(f'SoH table: {len(soh_rows)} rows over {len(soh_ident)} calibrations')

    # ---- negative controls before the loads
    ncs = {'NC1_wrong_sha256_on_allow_list': nc_wrong_sha(),
           'NC2_pickle_loads_bytes': nc_expect_refusal(lambda: pickle.loads(b'\x80\x04N.'), 'pickle.loads(bytes)'),
           'NC3_pickle_load_in_memory_handle': nc_expect_refusal(lambda: pickle.load(io.BytesIO(b'\x80\x04N.')),
                                                                  'pickle.load(BytesIO)'),
           'NC4_off_list_pickle': nc_expect_refusal(lambda: _load_path(OFF_LIST_PICKLE),
                                                    f'pickle.load({OFF_LIST_PICKLE})')}
    _log(f"negative controls before loads: {[(k, v['ok']) for k, v in ncs.items()]}")
    if not all(v['ok'] for v in ncs.values()):
        _refuse('negative controls before the loads', json.dumps(ncs))

    expected_loads = 0 if dry else 2
    modules_before = set(sys.modules)
    soc = None
    zero_checks = []
    load_log = []
    if not dry:
        # row 2 first (the reference unit), then row 1 (x = 0); one model set in memory at a time
        _log(f'LOAD 1 of 2: {PICKLES[2][0]} (row 2, reference unit)')
        models = _load_path(PICKLES[2][0])
        load_log.append({'order': 1, 'row': 2, 'path': PICKLES[2][0], 'utc': datetime.now(timezone.utc).isoformat(),
                         'payload_keys': sorted(models)})
        e7, blocks, zc2 = read_row2(models, srp1, base)
        zero_checks.extend(zc2)
        del models
        gc.collect()
        _log(f'row 2 read: TSO index of node 7 = {e7}; {len(blocks)} blocks; {len(zc2)} zero-capacity copies checked')
        ncs['NC5_second_load_row2'] = nc_expect_refusal(lambda: _load_path(PICKLES[2][0]), 'second load of row 2')
        _log(f'LOAD 2 of 2: {PICKLES[1][0]} (row 1, x = 0)')
        models = _load_path(PICKLES[1][0])
        load_log.append({'order': 2, 'row': 1, 'path': PICKLES[1][0], 'utc': datetime.now(timezone.utc).isoformat(),
                         'payload_keys': sorted(models)})
        zc1 = read_row1(models, srp1)
        zero_checks.extend(zc1)
        del models
        gc.collect()
        _log(f'row 1 read: {len(zc1)} shared-ESS copies checked')
        ncs['NC6_second_load_row1'] = nc_expect_refusal(lambda: _load_path(PICKLES[1][0]), 'second load of row 1')
        _log(f"NC5 {ncs['NC5_second_load_row2']['ok']}, NC6 {ncs['NC6_second_load_row1']['ok']}")
        hourly, summ, limits, vs_e6 = soc_tables(blocks, er[2], e6)
        soc = {'tso_index_of_node7': e7, 'hourly': hourly, 'summary': summ, 'limits': limits, 'vs_e6_supporting': vs_e6}
    project_modules_by_unpickling = sorted(
        m for m in set(sys.modules) - modules_before
        if getattr(sys.modules[m], '__file__', None)
        and os.path.dirname(os.path.abspath(sys.modules[m].__file__)) == REPO)

    gfail = GUARD.verify(0)
    expected_refusals = 3 + (0 if dry else 2)   # NC2, NC3, NC4 (+ NC5, NC6); NC1 runs on its own gate
    pickle_ok = (PICKLE_GATE.counts['delegated'] == expected_loads
                 and PICKLE_GATE.counts['refused'] == expected_refusals
                 and PICKLE_GATE.counts['unpickler_attempts'] == 0
                 and pickle.load == PICKLE_GATE.load and pickle.loads == PICKLE_GATE.loads)
    build_ok = BUILD_COUNTS['pyomo_model_init'] == 0
    ncs_ok = all(v['ok'] for v in ncs.values())
    guards = {
        'solve_profile_guard': {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_0_failures': gfail},
        'pickle_gate': {'allow_list': [{'path': p, 'sha256': s} for p, s in (PICKLES[1], PICKLES[2])],
                        'counts': dict(PICKLE_GATE.counts), 'expected_delegated_loads': expected_loads,
                        'expected_refusals': expected_refusals, 'refusals': PICKLE_GATE.refusals,
                        'loaded': PICKLE_GATE.loaded, 'load_log': load_log, 'ok': pickle_ok},
        'build_guard': {'blocked': 'pyomo.core.base.PyomoModel.Model.__init__', 'counts': dict(BUILD_COUNTS),
                        'ok': build_ok},
        'negative_controls': ncs, 'negative_controls_ok': ncs_ok,
        'project_modules_imported_by_unpickling': project_modules_by_unpickling,
    }
    if gfail or not pickle_ok or not build_ok or not ncs_ok:
        _log(f'GUARD FAULT: {json.dumps(guards, default=str)}')
        raise SystemExit('guard fault')

    # ---- outputs
    files = {}
    files['e6_rederivation_comparison'] = _write_csv(out_dir, 'e6_rederivation_comparison.csv', cmp_rows)
    files['e6_rederived_day_ranking'] = _write_csv(out_dir, 'e6_rederived_day_ranking.csv', [
        {k: (v[0] if k == 'block_weight' else v) for k, v in e.items()} for e in ds['ranking']])
    pb_rows = []
    for b in mine['panel_b_per_block_change_vs_x0']['blocks']:
        for m in ('weighted', 'unweighted'):
            s = b[m]
            r = {'year': b['year'], 'day': b['day'], 'measure': m,
                 'tn_conventional_cost_change_eur': s['tn_conventional_cost_tso_generation_cost']['change_x_minus_x0']}
            for n in DN_NODES + ('total',):
                r[f'dn_flexibility_cost_change_{n}_eur'] = s['dn_flexibility_activation_cost_dso_flexibility_cost_internal'][n]['change_x_minus_x0']
            for k, v in s['other_Q_components'].items():
                r[f'{k}_change_eur'] = v['change_x_minus_x0']
            r['Q_change_eur'] = s['Q_block']['change_x_minus_x0']
            r['Q_row1_x0_eur'] = s['Q_block']['row1_x0']
            r['Q_row2_x_eur'] = s['Q_block']['row2_x']
            r['components_minus_Q_change_residual_eur'] = s['components_sum_to_Q_change_residual']
            pb_rows.append(r)
    files['e6_rederived_panel_b'] = _write_csv(out_dir, 'e6_rederived_panel_b.csv', pb_rows)
    pc_rows = []
    for no in ROWS:
        for n, node in mine['panel_c_e_available'][f'row{no}']['nodes'].items():
            for y, v in node['years'].items():
                pc_rows.append(dict({'row': no, 'cell': ROWS[no]['cell'], 'node': n, 'year': y,
                                     'e_rated_mwh': node['e_rated_mwh'], 's_rated_mva': node['s_rated_mva']}, **v))
    files['e6_rederived_panel_c'] = _write_csv(out_dir, 'e6_rederived_panel_c.csv', pc_rows)
    files['soh_per_block_calibrations'] = _write_csv(out_dir, 'soh_per_block_calibrations.csv', soh_rows)
    files['soh_calibration_identity'] = _write_csv(out_dir, 'soh_calibration_identity.csv', soh_ident)
    if soc is not None:
        files['soc_hourly_reference_unit'] = _write_csv(out_dir, 'soc_hourly_reference_unit_2025_2035.csv', soc['hourly'])
        files['soc_block_summary'] = _write_csv(out_dir, 'soc_block_summary_reference_unit.csv', soc['summary'])
        files['soc_limits_per_year'] = _write_csv(out_dir, 'soc_limits_per_year_reference_unit.csv', soc['limits'])
        files['zero_capacity_model_check'] = _write_csv(out_dir, 'zero_capacity_model_check.csv', zero_checks)

    x0 = [z for z in zero_checks if z['row'] == 1]
    r2z = [z for z in zero_checks if z['row'] == 2]
    statement_x0 = None
    if soc is not None:
        statement_x0 = {
            'n_shared_ess_copies_checked': len(x0),
            'all_inactive_by_model': all(z['inactive_by_model'] for z in x0),
            'n_soc_entries': sum(z['shared_es_soc_n'] for z in x0),
            'n_soc_entries_fixed': sum(z['shared_es_soc_n_fixed'] for z in x0),
            'max_abs_soc_pu': max(z['shared_es_soc_max_abs_pu'] for z in x0),
            'max_abs_e_rated_fixed_pu': max(abs(z['e_rated_fixed_pu']) for z in x0),
            'max_abs_s_rated_fixed_pu': max(abs(z['s_rated_fixed_pu']) for z in x0),
            'n_soc_rows': sum(z['soc_rows_n'] for z in x0), 'n_soc_rows_active': sum(z['soc_rows_n_active'] for z in x0),
            'statement': ('x = 0: every shared-ESS copy (TSO indices for nodes 5/7/9 and each DSO copy, all 12 blocks) '
                          'has e_rated_fixed = s_rated_fixed = 0 and every pch/pdch/pnet/qnet/SoC entry FIXED at 0, with '
                          'the SoC rows deactivated (configure_shared_ess_operational_state, zero capacity); SoC is '
                          'identically 0 by construction -- no hourly zeros are exported'),
            'row2_zero_capacity_nodes_5_9': {'n_copies': len(r2z), 'all_inactive_by_model': all(z['inactive_by_model'] for z in r2z)},
        }
    result = {
        'schema': 'p515_s55_w188d_operational_figure_v1',
        'stage': 'P5.15 Addendum 89 W188 (d): operational figure data (zero solves, no build, two guarded pickle reads)',
        'dry_run': dry, 'started_utc': started, 'git_HEAD': head, 'tracked_changes_at_start': dirty,
        'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'git_status': script_status},
        'interpreter': sys.executable,
        'objective_convention': ('Q = certified gross_operational_cost: gross operational cost, contracted interface '
                                 'settlement excluded; one market scenario; terminal salvage 0 on both E.21 rows'),
        'instance': {f'row{no}': {'cell': r['cell'], 'eval_dir': r['eval'], 'k_star': r['k_star'],
                                  'eval_key': er[no]['eval_key'], 'candidate_key': er[no]['candidate_key'],
                                  'candidate_canonical': er[no]['candidate_canonical'],
                                  'certified_cost_Q': er[no]['certified_cost']} for no, r in ROWS.items()},
        'pre_run_checks': {'capture_checklist': {k: v for k, v in chk.items() if k != 'paths'},
                           'pickle_pins': pk, 'e6_blob': e6b, 'manuscript_md5': ms},
        'base_mva': {f'{k[0]}|{k[1]}': v for k, v in base.items()},
        'constants': {'soc_min_frac': SOC_MIN_FRAC, 'soc_max_frac': SOC_MAX_FRAC, 'soc_init_frac': SOC_INIT_FRAC,
                      'contact_tol_mwh': CONTACT_TOL_MWH, 'pv_weights': 'w_y = 1 / 1.02**(y - 2025)'},
        'part1_e6_rederivation': {'rederived': mine, 'comparison_summary': cmp_summary},
        'part2_soc': None if soc is None else {
            'tso_index_of_node7': soc['tso_index_of_node7'], 'block_summary': soc['summary'], 'limits': soc['limits'],
            'vs_e6_supporting_same_pickle': soc['vs_e6_supporting'], 'x0_statement': statement_x0,
            'contact_rule': f'|SoC - model limit| <= {CONTACT_TOL_MWH} MWh, per copy, per hour'},
        'part3_soh': {'identity': soh_ident, 'baseline_vs_w101_reference_unit': base_vs_w101,
                      'note': ('SoH is carried per cohort and 5-year block (representative year), not per day; '
                               'soh_used_for_available_energy is the end-of-block SoH except under the mid-block arm')},
        'files': {k: {'path': v['path'], 'n_rows': v['n_rows']} for k, v in files.items()},
        'guards': guards,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    with open(out_json, 'x', encoding='utf-8') as fh:
        GRIO.dump(result, fh, indent=1, sort_keys=False)
        fh.write('\n')
    OUTPUT_FILES.append(out_json)
    manifest = {'outputs': {p: _sha(p) for p in OUTPUT_FILES},
                'script': {SCRIPT_REL: {'sha256': _sha(SCRIPT_REL), 'git_status': script_status}},
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)}, 'n_inputs': len(INPUTS)}
    with open(out_man, 'x', encoding='utf-8') as fh:
        GRIO.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')

    # ---- console summary
    print('=== part 1: E6 re-derivation vs e6_export.json')
    print(json.dumps({k: v for k, v in cmp_summary.items() if k not in ('rederived_paths_absent_in_e6',)},
                     indent=1, default=str))
    for e in ds['ranking']:
        print(f"  {e['rank']:>2} {e['year']} {e['day']:<7} c = {e['value_contribution_Q0_minus_Qx']:>12.2f} "
              f"bar {e['bar_sum_of_window_ranges']:>9.2f} next-determinate "
              f"{e.get('margin_to_next_rank_not_explained_by_stopping_slack')}")
    print('  horizon', json.dumps(mine['panel_b_per_block_change_vs_x0']['horizon_totals_weighted']))
    print('=== part 3: SoH per block')
    for r in soh_rows:
        print(f"  {r['calibration_table6']:<32} {r['block_year']} start {r['soh_start_of_block']:.6f} end "
              f"{r['soh_end_of_block']:.6f} used {r['soh_used_for_available_energy']:.6f} floor-active "
              f"{r['floor_row_active_at_terminal_flag']}")
    for i in soh_ident:
        print(f"  {i['cell']:<14} k {i['k_record_cl_eff']:.3f} phi {i['calendar_retention_phi_cal_record']} point "
              f"{i['soh_point_readback']} min {i['soh_min_applied_floor_row_lower']} AE {i['AE_recomputed_pv']:.6f} "
              f"(frozen {i['AE_frozen']}) EFC {i['EFC_recomputed_pv']:.6f} (frozen {i['EFC_frozen']}) binding "
              f"{i['binding_year_record']} (frozen {i['binding_year_frozen']}, T8 {i['binding_year_table8_printed']})")
    print('  baseline vs W101', json.dumps(base_vs_w101))
    if soc is not None:
        print('=== part 2: SoC')
        for r in soc['summary']:
            print(f"  {r['year']} {r['day']:<7} E^Av {r['e_available_record_mwh']:.6f} TSO [{r['soc_min_tso_mwh']:.6f}, "
                  f"{r['soc_max_tso_mwh']:.6f}] contacts lo/up {r['n_contact_lower_tso']}/{r['n_contact_upper_tso']} "
                  f"DSO contacts {r['n_contact_lower_dso']}/{r['n_contact_upper_dso']} TSO-DSO "
                  f"{r['soc_tso_minus_dso_maxabs_mwh']:.2e} vs E6 {r.get('vs_e6_supporting_same_pickle_maxabs')}")
        print('  limits', json.dumps(soc['limits'], indent=1))
        print('  x0', json.dumps(statement_x0, indent=1))
    print('guards:', json.dumps(guards, indent=1, default=str))
    _log(f'wrote {len(OUTPUT_FILES)} outputs and {out_man}')


if __name__ == '__main__':
    main()
