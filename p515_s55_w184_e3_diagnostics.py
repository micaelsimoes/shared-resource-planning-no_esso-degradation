"""
P5.15 W184 (E3) -- diagnostics of the C_NRF certified evaluation (Addendum 85 W184 line and ruling 6; Addendum 86;
W183i (d) Blocked 2, option (a): the persisted models are the capture path). ZERO SOLVES, NO MODEL BUILD, EXACTLY ONE
PICKLE LOAD (one declared file, by (path, sha256)).

INSTANCE
  C_NRF  cell e3_cnrf_x0_1x, eval key 35b275180d897e4c..., candidate key 8435c718... (x = 0), 1x, nrf_option True,
         Step-7 resettle v6, certified at k* 206 (run commit d7d4875e).
  C_OPEN E.21 row 1, eval key d110bd1a5977df1e... (W101 x0, certified at 181) -- recorded outputs only, for comparison.

OBJECTIVE CONVENTION. Q = certified gross_operational_cost, contracted interface settlement excluded (Addendum 83
ruling 2); one market scenario (deviation settlement and commitment premium identically 0); terminal salvage 0.

PART A -- recorded outputs (JSON; no pickle)
  A1 curtailment check (W183i ruling (b)): Table 12 (main.tex, tab:res_benchmark) and E.23 (tab:appE_benchmark) rows
     read verbatim; the static columns re-derived from report_v3 phase A (net, mwh_day_weighted / 1000, by agent); the
     coordinated C_OPEN curtailment recomputed from row 1's component_levels_terminal.json with the formula
       GWh(agent) = sum over that agent's blocks of unweighted res_curtailment_definitional_at_weight_1
                    x Years[y] x Days[d] (SRP1.json, undiscounted) / 1000
     and compared with report_v3 curtailment_table.coordinated (settled cycle 181, recomputed from models, net,
     mwh_day_weighted); C_NRF by the same formula.
  A2 closure slack per block (component_levels_terminal: shared_ess_day_balance_slack, local_ess_day_balance_slack,
     flexibility_p_day_balance_slack; weighted and unweighted) -- both cells.
  A3 interface residual ratios (evaluation_record terminal_ratios_per_channel, rule_ten; boyd_terminal
     binding_test_per_channel) and the per-hour interface consensus residual (interface_settlement_detail_s31c
     interface_consensus_residual_per_dso: max |residual_mw| per node) -- both cells.
  A4 settlement (recourse_components; the S31C identity T_TSO + T_DSO == -sum priced residual weighted) -- both cells.
  A5 rule-ten terminal step / threshold -- both cells.
  A6 the four C_OPEN reverse-flow hours (W182f Z7) read on both copies in both cells' settlement detail.

PART B -- the single guarded read of C_NRF's certified_models.pkl ({'tso': {y: {d: model}}, 'dso': {node: {y: {d}}}})
  B1 per block model (12 TSO + 36 DSO): shared-ESS capacity Params (S, E) x baseMVA, fixed status of the operational
     variables, active-row counts of sess_converter_capability / sess_soc_limit_*; converter-circle residual
     pnet^2 + qnet^2 - S^2 (MVA^2; max, its sign); max |pnet| / S (undefined where S = 0: max |pnet| reported); SOC
     against [0.10 E, 0.90 E] (definitions.ENERGY_STORAGE_MIN/MAX_ENERGY_STORED); the network's own ESS (es_*) if any.
  B2 NRF rows from the models (count, active, min body) -- cross-check of the run's terminal capture.
  B3 interface P at the four C_OPEN reverse-flow hours on both copies: TSO pc_adn[dn, s_m, s_o, p] and
     expected_interface_pf_p[dn, p]; DSO pg_adn[s_m, s_o, p] and expected_interface_pf_p[p]; x baseMVA (MW,
     DN-import positive); the NRF row body there; against the run's capture and the recorded settlement detail.
  "ESSO sigma slack": no quantity of that name exists in the code (W183i (d) Blocked 2) -- reported not applicable.
  The ESSO models (esso_models_s39_D.pkl) are not on the allow-list and are not read.

GUARDS
  SolveProfileGuard(permitted=()) installed before any project import, verified at exactly 0 at the end.
  pickle.load / loads / Unpickler replaced for the whole run by an allow-list gate of ONE (path, sha256) = C_NRF's
  certified_models.pkl, sha256 b4fcc6a8... as recorded in the run's campaign_manifest_sha256.json and
  post_certification.json (both checked before the load); the bytes hashed are the bytes unpickled; exactly 1
  delegated load (0 with --no-pickle). Negative controls (must refuse, never delegate): NC2 loads(bytes), NC3
  load(BytesIO), NC4 a second load of the allowed file. pyomo Model.__init__ blocked and counted (0 required).
  Each load is logged with its run (Addendum 85 ruling 6).
  Refuses: an output directory that is missing or holds anything but launch.log; a concurrent copy; a missing
  capture path (asserted before any reading); any pinned input whose sha256 differs from its pin.

OUTPUT (default data/SRP1/Results/P515S55/w184_e3_diagnostics/): w184_e3_diagnostics.json, manifest_sha256.json;
`--post-run` writes manifest_post_run_sha256.json over the outputs + launch.log. `--out-dir DIR` redirects (trials).

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S55/w184_e3_diagnostics && set -o noclobber && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w184_e3_diagnostics.py \\
      > data/SRP1/Results/P515S55/w184_e3_diagnostics/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w184_e3_diagnostics.py --post-run
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W184 E3 diagnostics (never solves)').install()

import hashlib  # noqa: E402
import io  # noqa: E402
import re  # noqa: E402

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
P55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
E3_ROOT = os.path.join(P55, 'e3_cnrf', 'campaign_s55_step7_e3_cnrf_x0_1x')
E3_EVAL = os.path.join(E3_ROOT, 'evals', '35b275180d897e4c_e3_cnrf_x0_1x')
ROW1_EVAL = os.path.join(P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0', 'evals',
                         'd110bd1a5977df1e_x0')
PICKLE_REL = os.path.join(E3_EVAL, 'certified_models.pkl')
PICKLE_SHA256 = 'b4fcc6a81efeb53c720bb4a3f624b22e408e01a649cc845632f5e74ca059bfbc'
E3_EVAL_KEY = '35b275180d897e4c123c37839b91f50b01454bb24085685d26e9d2b33346e12f'
C_OPEN_EVAL_KEY = 'd110bd1a5977df1e811a1b3963afc565de6daa65f76dcc096ead3fdc70b08546'


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
PICKLE_GATE = AllowListPickleGate([(PICKLE_REL, PICKLE_SHA256)], 'W184 pickle gate', _ORIGINAL_PICKLE['loads'])


class _BlockedUnpickler(pickle.Unpickler):
    def __init__(self, *_a, **_k):
        PICKLE_GATE.counts['unpickler_attempts'] += 1
        PICKLE_GATE._refuse('pickle.Unpickler', 'Unpickler is not on the allow-list')


pickle.load, pickle.loads, pickle.Unpickler = PICKLE_GATE.load, PICKLE_GATE.loads, _BlockedUnpickler

from pyomo.core.base.PyomoModel import Model as _PyomoModel  # noqa: E402

BUILD_COUNTS = {'pyomo_model_init': 0}


def _blocked_model_init(self, *_a, **_k):
    BUILD_COUNTS['pyomo_model_init'] += 1
    raise RuntimeError('W184: a Pyomo model build was attempted -- no model builds are permitted')


_PyomoModel.__init__ = _blocked_model_init

import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import pyomo.environ as pe  # noqa: E402

import gate_result_io as GRIO  # noqa: E402  (the shared writer)

SCRIPT_REL = os.path.basename(__file__)
DEFAULT_OUT_DIR = os.path.join(P55, 'w184_e3_diagnostics')
CELLS = {'C_NRF': E3_EVAL, 'C_OPEN': ROW1_EVAL}
PINS = {   # inputs whose identity a value depends on (sha256 pinned; checked before reading)
    os.path.join(E3_ROOT, 'campaign_results.json'): '0db2edf9af390131bf88a12aae9ff5200894663530a3e9aafea289e00e5bf627',
    os.path.join(P53, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json'):
        'bf6dbf7247a922020c58e7b9502f63539c0bbf048447c3e33bbbf202c152694a',
    os.path.join(P53, 'w182f_records', 'w182f_records.json'):
        '405c79c9e146600421a76443692fb23eb69d0d35d90358de14c899d66ddda085',
    os.path.join(ROW1_EVAL, 'component_levels_terminal.json'):
        '033c1e7eac919969c0cdfbfbb8969484b49b2e83b09a36cd171e7c3068a3a3d1',
    os.path.join(ROW1_EVAL, 'evaluation_record.json'):
        'c08fa4335c7e266d6d61439fe36d28d56cc7ead31b4956d0f2c6d1fe2b0e8956',
    os.path.join('manuscript', '6a67305f25e8348fb71380c3', 'main.tex'):
        'cec2e2982dfa1509e6e066a208521fb1eea40d5c71f1cdf71a9bf00a12b3520b',   # Overleaf a87e195 (round 8)
}
REPORT_V3 = os.path.join(P53, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json')
W182F = os.path.join(P53, 'w182f_records', 'w182f_records.json')
MAIN_TEX = os.path.join('manuscript', '6a67305f25e8348fb71380c3', 'main.tex')
APPE_TEX = os.path.join(P55, 'w183h_round8', 'appE_generated_w183h.tex')
SRP1_JSON = os.path.join('data', 'SRP1', 'SRP1.json')
CASE_DIRS = {'TSO': 'case9', 5: 'case33_1', 7: 'case33_2', 9: 'case33_3'}
SOC_MIN_FRAC, SOC_MAX_FRAC = 0.10, 0.90
PROD_SOURCE_NAMES = {   # names this script reads, asserted present in production source
    'network.py': ['model.shared_es_pnet = pe.Var', 'model.shared_es_qnet = pe.Var', 'model.shared_es_soc = pe.Var',
                   'model.shared_es_s_rated_fixed = pe.Param', 'model.shared_es_e_rated_fixed = pe.Param',
                   'model.sess_converter_capability = pe.Constraint', 'model.sess_soc_limit_upper = pe.Constraint',
                   'model.sess_soc_limit_lower = pe.Constraint', 'model.es_pnet = pe.Var',
                   'model.ess_converter_capability = pe.Constraint', 'model.ess_soc_limits = pe.Constraint',
                   'model.pc_adn = pe.Expression', 'model.pg_adn = pe.Expression'],
    'model_construction_helpers.py': ['return (m.shared_es_pnet[e, s_m0, s_o0, p] ** 2',
                                      '+ m.shared_es_qnet[e, s_m0, s_o0, p] ** 2) <= m.shared_es_s_rated_fixed[e] ** 2'],
    'definitions.py': ['ENERGY_STORAGE_MAX_ENERGY_STORED = 0.90', 'ENERGY_STORAGE_MIN_ENERGY_STORED = 0.10'],
    'shared_resources_planning.py': ["INTERFACE_NO_REVERSE_FLOW_ROW = 'interface_no_reverse_flow_p'",
                                     'return m.pc_adn[dn, s_m, s_o, p] >= 0.0', 'return m.pg_adn[s_m, s_o, p] >= 0.0',
                                     'p_int_tso_expected = pe.value(local_tso_model.expected_interface_pf_p[dn, p]) '
                                     '* s_base'],
    'p515_s42_exact_fix_rerun.py': ["payload = {'tso': models['tso'], 'dso': models['dso']}"],
}
Q_PARTS = ('generation_cost', 'flexibility_cost_internal', 'load_curtailment_cost', 'res_curtailment_penalty',
           'ess_usage_cost', 'detector_penalty_total')

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


def _text(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return f.read()


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
def capture_checklist():
    """Rule eleven: every capture path exists before any reading (fail fast)."""
    need = {}
    for cell, ev in CELLS.items():
        for f in ('component_levels_terminal.json', 'evaluation_record.json', 'boyd_terminal.json',
                  'interface_settlement_detail_s31c.json'):
            need[f'{cell} {f}'] = os.path.join(ev, f)
    need['C_NRF campaign_results.json'] = os.path.join(E3_ROOT, 'campaign_results.json')
    need['C_NRF campaign_manifest_sha256.json'] = os.path.join(E3_ROOT, 'campaign_manifest_sha256.json')
    need['C_NRF post_certification.json'] = os.path.join(E3_EVAL, 'post_certification.json')
    need['C_NRF certified_models.pkl'] = PICKLE_REL
    for p in PINS:
        need[f'pinned {p}'] = p
    need['E.23 generated tex (round 8)'] = APPE_TEX
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
        res = json.load(open(os.path.join(E3_ROOT, 'campaign_results.json')))
        nt = (res.get('outcome') or {}).get('nrf_terminal') or {}
        keys['campaign_results nrf_terminal written'] = nt.get('status') == 'written'
        keys['campaign_results binding hours by the expected-value rule'] = bool(
            (nt.get('binding_hours_expected_value_rule') or {}).get('hours'))
        keys['campaign_results eval key = E3'] = res.get('eval_key') == E3_EVAL_KEY
        for cell, ev in CELLS.items():
            cl = json.load(open(os.path.join(ev, 'component_levels_terminal.json')))
            keys[f'{cell} component_levels 48 blocks with the closure slacks and curtailment'] = (
                len(cl['blocks']) == 48 and all(
                    k in b['unweighted'] and k in b['weighted'] for b in cl['blocks'].values()
                    for k in ('shared_ess_day_balance_slack', 'local_ess_day_balance_slack',
                              'flexibility_p_day_balance_slack', 'res_curtailment_definitional_at_weight_1')))
            er = json.load(open(os.path.join(ev, 'evaluation_record.json')))
            keys[f'{cell} evaluation_record terminal_ratios_per_channel'] = bool(er.get('terminal_ratios_per_channel'))
            keys[f'{cell} evaluation_record rule_ten'] = 'terminal_step_over_threshold' in (er.get('rule_ten') or {})
            isd = json.load(open(os.path.join(ev, 'interface_settlement_detail_s31c.json')))
            keys[f'{cell} settlement detail consensus residual per dso'] = set(
                isd['interface_consensus_residual_per_dso']) == {'5', '7', '9'}
        rv3 = json.load(open(REPORT_V3))
        keys['report_v3 coordinated curtailment net by agent'] = 'net' in rv3['curtailment_table']['coordinated'][
            'settled_cycle_181_signed_parts_from_models']
        keys['report_v3 coordinated_cell is row 1'] = rv3['coordinated_cell']['eval_key'] == C_OPEN_EVAL_KEY
    failed = sorted(k for k, v in keys.items() if not v)
    return {'ok': not missing and not failed, 'n_paths': len(need), 'missing_paths': missing,
            'n_key_checks': len(keys), 'failed_key_checks': failed, 'paths': need}


def verify_pins():
    out = {}
    for p, s in PINS.items():
        actual = _sha(p)
        out[p] = {'pin': s, 'actual': actual, 'ok': actual == s, 'status': _status(p)}
    return {'ok': all(v['ok'] for v in out.values()), 'files': out}


def verify_pickle_pin():
    man = json.load(open(os.path.join(E3_ROOT, 'campaign_manifest_sha256.json')))
    pc = json.load(open(os.path.join(E3_EVAL, 'post_certification.json')))
    actual = _sha(PICKLE_REL)
    rec_man = man.get(PICKLE_REL)
    rec_pc = (pc.get('persisted_models') or {}).get('sha256')
    return {'pin': PICKLE_SHA256, 'campaign_manifest': rec_man, 'post_certification': rec_pc,
            'sha256_actual_before_load': actual, 'size_bytes': os.path.getsize(PICKLE_REL),
            'git_status': _status(PICKLE_REL),
            'ok': actual == PICKLE_SHA256 == rec_man == rec_pc}


# ============================================================================================ part A
def _day_weight(srp1, y, d):
    return srp1['Years'][str(y)] * srp1['Days'][str(d)]


def curtailment_gwh(cl, srp1):
    out = {'TSO': 0.0, 'DSO': 0.0}
    rep = {'TSO': 0.0, 'DSO': 0.0}
    for b in cl['blocks'].values():
        u = b['unweighted']['res_curtailment_definitional_at_weight_1']
        out[b['kind']] += u * _day_weight(srp1, b['year'], b['day'])
        rep[b['kind']] += u
    return {a: {'gwh_day_weighted': out[a] / 1000.0, 'mwh_day_weighted': out[a], 'mwh_rep_day_sum': rep[a]}
            for a in out}


def _table_row(tex, label, first_cell):
    """The LaTeX row beginning with `first_cell` inside the table labelled `label` (verbatim)."""
    i = tex.index('\\label{' + label + '}')
    j = tex.index('\\end{table}', i)
    for line in tex[i:j].splitlines():
        if line.strip().startswith(first_cell):
            return line.strip()
    return None


def part_a1_curtailment(srp1, cl, rv3):
    tex, appe = _text(MAIN_TEX), _text(APPE_TEX)
    t12 = {'coordinated': _table_row(tex, 'tab:res_benchmark', 'Coordinated'),
           'price_taker': _table_row(tex, 'tab:res_benchmark', 'Price-taker NRF'),
           'passive': _table_row(tex, 'tab:res_benchmark', 'Passive NRF')}
    e23 = {'coordinated': _table_row(appe, 'tab:appE_benchmark', 'Coordinated')}

    def _cells(row):
        return [c.strip().rstrip('\\').strip() for c in row.split('&')] if row else None
    t12_cells = {k: _cells(v) for k, v in t12.items()}
    e23_cells = {k: _cells(v) for k, v in e23.items()}
    coord_curt_cols_t12 = t12_cells['coordinated'][-2:] if t12_cells['coordinated'] else None
    coord_curt_cols_e23 = e23_cells['coordinated'][-2:] if e23_cells['coordinated'] else None
    # the static columns: report_v3 phase A, net, mwh_day_weighted / 1000, by agent, printed at one decimal
    arms = rv3['curtailment_table']['arms_and_variants_phase_A']
    static = {}
    for key, arm in (('price_taker', 'nrf_arm_price_taker_warm_from_certified_r2'),
                     ('passive', 'nrf_arm_passive_perturbed_r2')):
        net = arms[arm]['signed_parts']['net']['by_agent']
        gwh = {a: net[a]['mwh_day_weighted'] / 1000.0 for a in ('TSO', 'DSO')}
        printed = t12_cells[key][-2:]
        static[key] = {'arm': arm, 'gwh_net_mwh_day_weighted_over_1000': gwh,
                       'one_decimal': [f"{gwh['TSO']:.1f}", f"{gwh['DSO']:.1f}"], 'table12_printed': printed,
                       'reproduces_table12': [f"{gwh['TSO']:.1f}", f"{gwh['DSO']:.1f}"] == printed}
    coord = rv3['curtailment_table']['coordinated']['settled_cycle_181_signed_parts_from_models']['net']['by_agent']
    c_open = curtailment_gwh(cl['C_OPEN'], srp1)
    c_nrf = curtailment_gwh(cl['C_NRF'], srp1)
    cmp_ = {a: {'recomputed_mwh_day_weighted': c_open[a]['mwh_day_weighted'],
                'report_v3_coordinated_mwh_day_weighted': coord[a]['mwh_day_weighted'],
                'abs_diff_mwh': abs(c_open[a]['mwh_day_weighted'] - coord[a]['mwh_day_weighted']),
                'recomputed_mwh_rep_day_sum': c_open[a]['mwh_rep_day_sum'],
                'report_v3_mwh_rep_day_sum': coord[a]['mwh_rep_day_sum'],
                'abs_diff_rep_day_sum': abs(c_open[a]['mwh_rep_day_sum'] - coord[a]['mwh_rep_day_sum'])}
            for a in ('TSO', 'DSO')}
    reproduces_v3 = all(v['abs_diff_mwh'] <= 1e-9 * max(1.0, abs(v['report_v3_coordinated_mwh_day_weighted']))
                        and v['abs_diff_rep_day_sum'] <= 1e-12 for v in cmp_.values())
    table12_has_numbers = bool(coord_curt_cols_t12) and not all(c == '---' for c in coord_curt_cols_t12)
    return {
        'formula': ('GWh(agent) = sum over the agent\'s blocks of component_levels_terminal unweighted '
                    'res_curtailment_definitional_at_weight_1 (MWh per representative day, net = signed, the '
                    'production definitional quantity) x Years[y] x Days[d] (SRP1.json, undiscounted) / 1000; the '
                    'static columns: report_v3 phase A signed_parts.net.by_agent.mwh_day_weighted / 1000'),
        'table12_rows_verbatim': t12, 'table12_cells': t12_cells,
        'e23_coordinated_row_verbatim': e23['coordinated'],
        'table12_coordinated_curtailment_columns': coord_curt_cols_t12,
        'e23_coordinated_curtailment_columns': coord_curt_cols_e23,
        'table12_coordinated_columns_carry_numbers': table12_has_numbers,
        'static_columns_reproduced': static,
        'c_open_recomputed_vs_report_v3_coordinated': cmp_,
        'c_open_reproduces_report_v3_coordinated': reproduces_v3,
        'C_OPEN_gwh': {a: c_open[a]['gwh_day_weighted'] for a in c_open},
        'C_NRF_gwh': {a: c_nrf[a]['gwh_day_weighted'] for a in c_nrf},
        'C_NRF_minus_C_OPEN_mwh': {a: c_nrf[a]['mwh_day_weighted'] - c_open[a]['mwh_day_weighted'] for a in c_nrf},
        'note': ('Table 12 and E.23 print "---" for the coordinated curtailment, so there is no printed coordinated '
                 'value to reproduce; the comparison is made against the committed coordinated record that the '
                 'W116 benchmark computed (report_v3 curtailment_table.coordinated, settled cycle 181, recomputed '
                 'from models) under the same net / mwh_day_weighted definition as the static columns'),
    }


def part_a2_closure(cl):
    out = {}
    for cell, c in cl.items():
        per = {}
        for name, b in c['blocks'].items():
            per[name] = {k: {'unweighted': b['unweighted'][k], 'weighted': b['weighted'][k]}
                         for k in ('shared_ess_day_balance_slack', 'local_ess_day_balance_slack',
                                   'flexibility_p_day_balance_slack')}
        summ = {}
        for k in ('shared_ess_day_balance_slack', 'local_ess_day_balance_slack', 'flexibility_p_day_balance_slack'):
            vals = [v[k]['weighted'] for v in per.values()]
            uw = [v[k]['unweighted'] for v in per.values()]
            summ[k] = {'sum_weighted': sum(vals), 'max_abs_weighted': _maxabs(vals),
                       'max_abs_unweighted': _maxabs(uw), 'n_blocks_nonzero': sum(1 for v in uw if v != 0.0)}
        out[cell] = {'summary': summ, 'per_block': per,
                     'totals_weighted_detector': {k: c['totals_weighted'][k] for k in
                                                  ('voltage_slack', 'node_balance_slack', 'branch_flow_slack',
                                                   'flexibility_p_day_balance_slack', 'local_ess_day_balance_slack',
                                                   'shared_ess_day_balance_slack', 'detector_penalty_total')},
                     'esso_feasibility_violation_D3': c.get('esso_feasibility_violation_D3')}
    return out


def part_a3_a5_ratios_settlement(er, bt, isd):
    out = {}
    for cell in CELLS:
        e, b, s = er[cell], bt[cell], isd[cell]
        resid = {}
        for node, per in s['interface_consensus_residual_per_dso'].items():
            vals = [v['residual_mw'] for v in per['periods'].values()]
            pr = [v['priced_residual_pi_baseMVA_residual_weighted'] for v in per['periods'].values()]
            worst = max(per['periods'].items(), key=lambda kv: abs(kv[1]['residual_mw']))
            resid[node] = {'n_hours': len(vals), 'max_abs_residual_mw': _maxabs(vals), 'at': worst[0],
                           'sum_priced_residual_weighted_eur': sum(pr)}
        rc = e['recourse_components']
        sum_priced = sum(v['sum_priced_residual_weighted_eur'] for v in resid.values())
        out[cell] = {
            'eval_key': e['eval_key'], 'candidate_key': e['candidate_key'], 'certified_cost_Q': e['certified_cost'],
            'certification_cycle': e.get('certification_cycle'), 'cycles_run': e.get('cycles_run'),
            'terminal_ratios_per_channel': e['terminal_ratios_per_channel'],
            'first_pass_cycle_per_channel': e.get('first_pass_cycle_per_channel'),
            'boyd_binding_test_per_channel': {ch: {k: v.get(k) for k in ('r', 's', 'eps_pri', 'eps_dual',
                                                                       'primal_ratio', 'dual_ratio', 'channel_pass')}
                                              for ch, v in b['binding_test_per_channel'].items()},
            'rule_ten': e['rule_ten'],
            'interface_consensus_residual_per_dso': resid,
            'settlement': {k: rc.get(k) for k in ('interface_settlement_tso', 'interface_settlement_dso',
                                                  'interface_settlement_total', 'interface_settlement_contracted_total',
                                                  'interface_settlement_deviation_total',
                                                  'interface_settlement_covariance_total',
                                                  'gross_operational_cost', 'gross_operational_cost_including_settlement',
                                                  'terminal_salvage_value', 'net_operational_recourse')},
            't_tso_plus_t_dso_terminal': s['t_tso_plus_t_dso_terminal'],
            'settlement_identity': {'definition': 't_tso_plus_t_dso_terminal == -sum(priced residual weighted)',
                                    'lhs': s['t_tso_plus_t_dso_terminal'], 'minus_sum_priced_residual': -sum_priced,
                                    'abs_diff': abs(s['t_tso_plus_t_dso_terminal'] + sum_priced)},
            'settlement_remainder': e.get('settlement_remainder'),
        }
    return out


def part_a6_hours(isd, z7_hours):
    out = {}
    for cell in CELLS:
        ird = isd[cell]['interface_reporting_detail']
        rows = []
        for h in z7_hours:
            v = ird[h['node']][h['year']][h['day']]['periods'][str(h['period'])]
            rows.append({'hour': f"{h['node']}|{h['year']}|{h['day']}|p{h['period']} (h{h['hour_1_based']})",
                         'p_int_tso_expected_mw': v['p_int_tso_expected_mw'],
                         'p_int_dso_expected_mw': v['p_int_dso_expected_mw'],
                         'delta_p_mw': v.get('delta_p_mw'), 'anchor_p_mw': v.get('anchor_p_mw'),
                         'price_per_mwh': v.get('price_per_mwh')})
        out[cell] = rows
    return out


# ============================================================================================ part B
def _ess_block(model, base):
    """Shared ESS (and the network's own ESS, if any) in one network block model."""
    sms, sos, periods = list(model.scenarios_market), list(model.scenarios_operation), list(model.periods)
    out = {'n_scenarios': [len(sms), len(sos)], 'n_periods': len(periods), 'shared': {}, 'own': {}}
    for e in model.shared_energy_storages:
        s = _val(model.shared_es_s_rated_fixed[e])
        E = _val(model.shared_es_e_rated_fixed[e])
        circ, pratio, pabs, socmin, socmax, socs = [], [], [], [], [], []
        fixed = {n: [0, 0] for n in ('shared_es_pnet', 'shared_es_qnet', 'shared_es_soc', 'shared_es_pch',
                                     'shared_es_pdch')}
        for sm in sms:
            for so in sos:
                for p in periods:
                    for n in fixed:
                        v = getattr(model, n)[e, sm, so, p]
                        fixed[n][0 if v.fixed else 1] += 1
                    pn = _val(model.shared_es_pnet[e, sm, so, p])
                    qn = _val(model.shared_es_qnet[e, sm, so, p])
                    soc = _val(model.shared_es_soc[e, sm, so, p])
                    circ.append((pn ** 2 + qn ** 2 - s ** 2) * base ** 2)
                    pabs.append(abs(pn) * base)
                    if s > 0:
                        pratio.append(abs(pn) / s)
                    socs.append(soc * base)
                    socmin.append((soc - SOC_MIN_FRAC * E) * base)
                    socmax.append((SOC_MAX_FRAC * E - soc) * base)
        cc = model.sess_converter_capability
        lo, up = model.sess_soc_limit_lower, model.sess_soc_limit_upper
        idx = [i for i in cc if i[0] == e]
        body_resid = [(_val(cc[i].body) - _val(cc[i].upper)) * base ** 2 for i in idx]
        imax = max(range(len(circ)), key=lambda k: circ[k])
        out['shared'][str(e)] = {
            's_rated_fixed_mva': s * base, 'e_rated_fixed_mwh': E * base,
            'fixed_vs_free_counts': {n: {'fixed': v[0], 'free': v[1]} for n, v in fixed.items()},
            'circle_residual_mva2_max': circ[imax], 'circle_residual_mva2_min': min(circ),
            'circle_residual_sign_of_max': (1 if circ[imax] > 0 else (-1 if circ[imax] < 0 else 0)),
            'circle_rows_constructed': len(idx), 'circle_rows_active': sum(1 for i in idx if cc[i].active),
            'circle_row_body_minus_upper_mva2_max': max(body_resid) if body_resid else None,
            'pnet_abs_max_mw': max(pabs), 'qnet_abs_max_mvar': max(
                abs(_val(model.shared_es_qnet[e, sm, so, p])) * base for sm in sms for so in sos for p in periods),
            'pnet_over_s_max': max(pratio) if pratio else None,
            'pnet_over_s_note': None if pratio else 'S = 0: ratio undefined; max |pnet| reported instead',
            'soc_mwh_min': min(socs), 'soc_mwh_max': max(socs),
            'soc_minus_10pct_E_min_mwh': min(socmin), 'soc_90pct_E_minus_soc_min_mwh': min(socmax),
            'soc_rows_constructed': sum(1 for i in lo if i[0] == e) + sum(1 for i in up if i[0] == e),
            'soc_rows_active': sum(1 for i in lo if i[0] == e and lo[i].active)
            + sum(1 for i in up if i[0] == e and up[i].active)}
    if hasattr(model, 'energy_storages'):
        for e in model.energy_storages:
            cc = getattr(model, 'ess_converter_capability', None)
            sl = getattr(model, 'ess_soc_limits', None)
            rc = [(_val(cc[i].body) - _val(cc[i].upper)) * base ** 2 for i in cc if i[0] == e] if cc is not None else []
            lo = [(_val(sl[i].body) - _val(sl[i].lower)) * base for i in sl if i[0] == e] if sl is not None else []
            hi = [(_val(sl[i].upper) - _val(sl[i].body)) * base for i in sl if i[0] == e] if sl is not None else []
            out['own'][str(e)] = {'circle_row_body_minus_upper_mva2_max': max(rc) if rc else None,
                                  'soc_minus_lower_min_mwh': min(lo) if lo else None,
                                  'upper_minus_soc_min_mwh': min(hi) if hi else None, 'n_rows_circle': len(rc)}
    return out


def _nrf_rows(model, base):
    c = getattr(model, 'interface_no_reverse_flow_p', None)
    if c is None:
        return {'present': False}
    bodies = [_val(c[i].body) * base for i in c]
    return {'present': True, 'n_rows': len(bodies), 'n_active': sum(1 for i in c if c[i].active),
            'min_body_mw': min(bodies), 'n_body_le_1e-3_mw': sum(1 for b in bodies if b <= 1e-3),
            'n_body_lt_minus_1e-3_mw': sum(1 for b in bodies if b < -1e-3)}


def part_b(models, srp1, base, z7_hours, capture_hours):
    years, days = list(srp1['Years']), list(srp1['Days'])
    dn_order = [dn['connection_node_id'] for dn in srp1['DistributionNetworks']]
    blocks, nrf = {}, {}
    for y in years:
        for d in days:
            tm = _by_str(_by_str(models['tso'], y), d)
            blocks[f'TSO|{y}|{d}'] = _ess_block(tm, base[('TSO', y)])
            nrf[f'TSO|{y}|{d}'] = _nrf_rows(tm, base[('TSO', y)])
            for node in dn_order:
                dm = _by_str(_by_str(_by_str(models['dso'], node), y), d)
                blocks[f'DSO{node}|{y}|{d}'] = _ess_block(dm, base[(node, y)])
                nrf[f'DSO{node}|{y}|{d}'] = _nrf_rows(dm, base[(node, y)])
    # summaries
    sh = [(b, e, v) for b, blk in blocks.items() for e, v in blk['shared'].items()]
    own = [(b, e, v) for b, blk in blocks.items() for e, v in blk['own'].items()]
    imax = max(sh, key=lambda t: t[2]['circle_residual_mva2_max'])
    summary = {
        'n_block_models': len(blocks), 'n_shared_ess_entries': len(sh), 'n_own_ess_entries': len(own),
        'shared_s_rated_fixed_mva_max': max(v['s_rated_fixed_mva'] for _b, _e, v in sh),
        'shared_e_rated_fixed_mwh_max': max(v['e_rated_fixed_mwh'] for _b, _e, v in sh),
        'circle_residual_mva2_max': imax[2]['circle_residual_mva2_max'], 'circle_residual_at': f'{imax[0]} e{imax[1]}',
        'circle_residual_sign_of_max': imax[2]['circle_residual_sign_of_max'],
        'circle_rows_constructed': sum(v['circle_rows_constructed'] for _b, _e, v in sh),
        'circle_rows_active': sum(v['circle_rows_active'] for _b, _e, v in sh),
        'pnet_abs_max_mw': max(v['pnet_abs_max_mw'] for _b, _e, v in sh),
        'qnet_abs_max_mvar': max(v['qnet_abs_max_mvar'] for _b, _e, v in sh),
        'pnet_over_s_max': _maxabs([v['pnet_over_s_max'] for _b, _e, v in sh]),
        'n_entries_with_s_zero': sum(1 for _b, _e, v in sh if v['s_rated_fixed_mva'] == 0.0),
        'soc_mwh_abs_max': max(max(abs(v['soc_mwh_min']), abs(v['soc_mwh_max'])) for _b, _e, v in sh),
        'soc_minus_10pct_E_min_mwh': min(v['soc_minus_10pct_E_min_mwh'] for _b, _e, v in sh),
        'soc_90pct_E_minus_soc_min_mwh': min(v['soc_90pct_E_minus_soc_min_mwh'] for _b, _e, v in sh),
        'soc_rows_constructed': sum(v['soc_rows_constructed'] for _b, _e, v in sh),
        'soc_rows_active': sum(v['soc_rows_active'] for _b, _e, v in sh),
        'fixed_vs_free_totals': {n: {k: sum(v['fixed_vs_free_counts'][n][k] for _b, _e, v in sh)
                                     for k in ('fixed', 'free')}
                                 for n in ('shared_es_pnet', 'shared_es_qnet', 'shared_es_soc', 'shared_es_pch',
                                           'shared_es_pdch')},
    }
    nrf_tot = {}
    for k, v in nrf.items():
        net = k.split('|')[0]
        a = nrf_tot.setdefault(net, {'n_rows': 0, 'n_active': 0, 'min_body_mw': None, 'n_body_le_1e-3_mw': 0,
                                     'n_body_lt_minus_1e-3_mw': 0, 'n_blocks_with_row': 0})
        if v.get('present'):
            a['n_blocks_with_row'] += 1
            a['n_rows'] += v['n_rows']
            a['n_active'] += v['n_active']
            a['n_body_le_1e-3_mw'] += v['n_body_le_1e-3_mw']
            a['n_body_lt_minus_1e-3_mw'] += v['n_body_lt_minus_1e-3_mw']
            a['min_body_mw'] = v['min_body_mw'] if a['min_body_mw'] is None else min(a['min_body_mw'], v['min_body_mw'])
    # the four hours on both copies
    hours = []
    cap = {(str(h['node']), str(h['year']), str(h['day']), int(h['hour'])): h for h in capture_hours}
    for h in z7_hours:
        node, y, d, p = int(h['node']), h['year'], h['day'], int(h['period'])
        tm = _by_str(_by_str(models['tso'], y), d)
        dm = _by_str(_by_str(_by_str(models['dso'], node), y), d)
        bt, bd = base[('TSO', y)], base[(node, y)]
        dn = dn_order.index(node)
        pp = [pp_ for pp_ in tm.periods if int(pp_) == p][0]
        sm, so = list(tm.scenarios_market)[0], list(tm.scenarios_operation)[0]
        dsm, dso_ = list(dm.scenarios_market)[0], list(dm.scenarios_operation)[0]
        tso_p = _val(tm.pc_adn[dn, sm, so, pp]) * bt
        dso_p = _val(dm.pg_adn[dsm, dso_, pp]) * bd
        tso_e = _val(tm.expected_interface_pf_p[dn, pp]) * bt
        dso_e = _val(dm.expected_interface_pf_p[pp]) * bd
        tr, dr = tm.interface_no_reverse_flow_p[dn, sm, so, pp], dm.interface_no_reverse_flow_p[dsm, dso_, pp]
        c = cap.get((str(node), str(y), str(d), p))
        hours.append({
            'hour': f'{node}|{y}|{d}|p{p} (h{h["hour_1_based"]})', 'c_open_p_int_dso_mw_z7': h['p_int_dso_mw'],
            'tso_copy': {'pc_adn_mw': tso_p, 'expected_interface_pf_p_mw': tso_e,
                         'nrf_row_body_mw': _val(tr.body) * bt, 'nrf_row_active': bool(tr.active)},
            'dso_copy': {'pg_adn_mw': dso_p, 'expected_interface_pf_p_mw': dso_e,
                         'nrf_row_body_mw': _val(dr.body) * bd, 'nrf_row_active': bool(dr.active)},
            'tso_minus_dso_mw': tso_p - dso_p,
            'capture_p_int_tso_expected_mw': c and c['p_int_tso_expected_mw'],
            'capture_p_int_dso_expected_mw': c and c['p_int_dso_expected_mw'],
            'pickle_minus_capture_tso_mw': (tso_e - c['p_int_tso_expected_mw']) if c else None,
            'pickle_minus_capture_dso_mw': (dso_e - c['p_int_dso_expected_mw']) if c else None,
            'among_capture_binding_hours': c is not None})
    return {'summary': summary, 'per_block': blocks, 'nrf_rows_by_network': nrf_tot, 'nrf_rows_per_block': nrf,
            'four_hours_both_copies': hours,
            'esso_sigma_slack': {'status': 'not applicable',
                                 'reason': ('no quantity of that name exists in the code (W183i (d) Blocked 2); the '
                                            'ESSO models are not in certified_models.pkl and esso_models_s39_D.pkl '
                                            'is not on this allow-list')}}


def _second_load():
    with open(PICKLE_REL, 'rb') as fh:
        pickle.load(fh)


def nc_expect_refusal(fn, label):
    before = PICKLE_GATE.counts['delegated']
    try:
        fn()
    except PickleRefused as exc:
        return {'label': label, 'ok': PICKLE_GATE.counts['delegated'] == before, 'refusal': str(exc)}
    return {'label': label, 'ok': False, 'refusal': None}


# ============================================================================================ main
def main():
    args = sys.argv[1:]
    no_pickle = '--no-pickle' in args
    out_dir = DEFAULT_OUT_DIR
    if '--out-dir' in args:
        out_dir = args[args.index('--out-dir') + 1]
    out_json = os.path.join(out_dir, 'w184_e3_diagnostics.json')
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
    _log(f'repo HEAD {head}; tracked changes: {dirty}; script {script_status}; no_pickle {no_pickle}; out {out_dir}')

    chk = capture_checklist()
    if not chk['ok']:
        _refuse('capture-path checklist', f"missing {chk['missing_paths']}; failed {chk['failed_key_checks']}")
    _log(f"capture-path checklist: {chk['n_paths']} paths, {chk['n_key_checks']} key checks, all present")
    pins = verify_pins()
    if not pins['ok']:
        _refuse('pinned inputs sha256', json.dumps({k: v for k, v in pins['files'].items() if not v['ok']}))
    pk = verify_pickle_pin()
    if not pk['ok']:
        _refuse('pickle sha256 = pin = campaign manifest = post_certification', json.dumps(pk))
    _log(f"pins verified ({len(PINS)}); pickle sha256 before load {pk['sha256_actual_before_load']} "
         f"(= campaign manifest = post_certification)")

    srp1 = _json(SRP1_JSON)
    base = {}
    for k, dname in CASE_DIRS.items():
        for y in srp1['Years']:
            base[(k, y)] = float(_json(os.path.join('data', 'SRP1', dname, f'{dname}_{y}.json'))['baseMVA'])
    cl = {c: _json(os.path.join(ev, 'component_levels_terminal.json')) for c, ev in CELLS.items()}
    er = {c: _json(os.path.join(ev, 'evaluation_record.json')) for c, ev in CELLS.items()}
    bt = {c: _json(os.path.join(ev, 'boyd_terminal.json')) for c, ev in CELLS.items()}
    isd = {c: _json(os.path.join(ev, 'interface_settlement_detail_s31c.json')) for c, ev in CELLS.items()}
    res = _json(os.path.join(E3_ROOT, 'campaign_results.json'))
    rv3 = _json(REPORT_V3)
    z7 = _json(W182F)['Z7_row1_reverse_flow']
    if er['C_NRF']['eval_key'] != E3_EVAL_KEY or er['C_OPEN']['eval_key'] != C_OPEN_EVAL_KEY:
        _refuse('instance identity', [er['C_NRF']['eval_key'], er['C_OPEN']['eval_key']])
    z7_hours = [{'node': str(e['node_id']), 'year': str(e['year']), 'day': str(e['day']), 'period': int(e['period']),
                 'hour_1_based': e.get('hour'), 'p_int_dso_mw': e['p_int_dso_mw']}
                for e in z7['entries_most_negative_first']]
    nt = res['outcome']['nrf_terminal']
    capture_hours = nt['binding_hours_expected_value_rule']['hours']

    a1 = part_a1_curtailment(srp1, cl, rv3)
    _log(f"A1 curtailment: Table 12 coordinated columns {a1['table12_coordinated_curtailment_columns']}; static "
         f"reproduced {[v['reproduces_table12'] for v in a1['static_columns_reproduced'].values()]}; C_OPEN vs "
         f"report_v3 coordinated {a1['c_open_reproduces_report_v3_coordinated']}; C_NRF GWh {a1['C_NRF_gwh']}")
    a2 = part_a2_closure(cl)
    a35 = part_a3_a5_ratios_settlement(er, bt, isd)
    a6 = part_a6_hours(isd, z7_hours)

    ncs = {'NC2_pickle_loads_bytes': nc_expect_refusal(lambda: pickle.loads(b'\x80\x04N.'), 'pickle.loads(bytes)'),
           'NC3_pickle_load_in_memory_handle': nc_expect_refusal(lambda: pickle.load(io.BytesIO(b'\x80\x04N.')),
                                                                  'pickle.load(BytesIO)')}
    _log(f"negative controls before load: {[(k, v['ok']) for k, v in ncs.items()]}")
    if not all(v['ok'] for v in ncs.values()):
        _refuse('negative controls before the load', json.dumps(ncs))
    pb = None
    expected_loads = 0 if no_pickle else 1
    modules_before = set(sys.modules)
    if not no_pickle:
        _log(f'GUARDED READ: loading {PICKLE_REL} sha256 {PICKLE_SHA256} (the single declared load of this run, '
             f'HEAD {head}, out {out_dir})')
        with open(PICKLE_REL, 'rb') as fh:
            models = pickle.load(fh)
        _log(f"loaded; payload keys {sorted(models)}; gate {PICKLE_GATE.loaded}")
        pb = part_b(models, srp1, base, z7_hours, capture_hours)
        del models
        ncs['NC4_second_load_of_the_allowed_file'] = nc_expect_refusal(_second_load, 'second pickle.load')
        _log(f"NC4: {ncs['NC4_second_load_of_the_allowed_file']['ok']}")
    project_modules_by_unpickling = sorted(
        m for m in set(sys.modules) - modules_before
        if getattr(sys.modules[m], '__file__', None)
        and os.path.dirname(os.path.abspath(sys.modules[m].__file__)) == REPO)

    gfail = GUARD.verify(0)
    expected_refusals = 2 + (0 if no_pickle else 1)
    pickle_ok = (PICKLE_GATE.counts['delegated'] == expected_loads
                 and PICKLE_GATE.counts['refused'] == expected_refusals
                 and PICKLE_GATE.counts['unpickler_attempts'] == 0
                 and pickle.load == PICKLE_GATE.load and pickle.loads == PICKLE_GATE.loads)
    build_ok = BUILD_COUNTS['pyomo_model_init'] == 0
    ncs_ok = all(v['ok'] for v in ncs.values())
    guards = {
        'solve_profile_guard': {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_0_failures': gfail},
        'pickle_gate': {'allow_list': [{'path': PICKLE_REL, 'sha256': PICKLE_SHA256}],
                        'counts': dict(PICKLE_GATE.counts), 'expected_delegated_loads': expected_loads,
                        'expected_refusals': expected_refusals, 'refusals': PICKLE_GATE.refusals,
                        'loaded': PICKLE_GATE.loaded, 'ok': pickle_ok},
        'build_guard': {'blocked': 'pyomo.core.base.PyomoModel.Model.__init__', 'counts': dict(BUILD_COUNTS),
                        'ok': build_ok},
        'negative_controls': ncs, 'negative_controls_ok': ncs_ok,
        'project_modules_imported_by_unpickling': project_modules_by_unpickling,
    }
    if gfail or not pickle_ok or not build_ok or not ncs_ok:
        _log(f'GUARD FAULT: {json.dumps(guards, default=str)}')
        raise SystemExit('guard fault')

    result = {
        'schema': 'p515_s55_w184_e3_diagnostics_v1',
        'stage': ('P5.15 W184 (E3): C_NRF diagnostics -- recorded outputs and one guarded read of the run\'s '
                  'certified_models.pkl (zero solves, no build)'),
        'no_pickle': no_pickle, 'started_utc': started, 'git_HEAD': head, 'tracked_changes_at_start': dirty,
        'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'git_status': script_status},
        'interpreter': sys.executable,
        'objective_convention': ('Q = certified gross_operational_cost: gross operational cost, contracted interface '
                                 'settlement excluded; one market scenario; terminal salvage 0 on both cells'),
        'instance': {c: {'eval_dir': CELLS[c], 'eval_key': er[c]['eval_key'], 'candidate_key': er[c]['candidate_key'],
                         'candidate_canonical': er[c]['candidate_canonical'], 'certified_cost_Q': er[c]['certified_cost'],
                         'certification_cycle': er[c].get('certification_cycle')} for c in CELLS},
        'pre_run_checks': {'capture_checklist': {k: v for k, v in chk.items() if k != 'paths'},
                           'pins': pins, 'pickle_pin': pk},
        'base_mva': {f'{k[0]}|{k[1]}': v for k, v in base.items()},
        'A1_curtailment_check': a1,
        'A2_closure_slack': a2,
        'A3_A5_ratios_settlement_rule_ten': a35,
        'A6_four_hours_recorded_settlement_detail': a6,
        'B_pickle': pb,
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
    print('=== A1', json.dumps({k: a1[k] for k in ('table12_coordinated_curtailment_columns',
                                                  'e23_coordinated_curtailment_columns',
                                                  'c_open_reproduces_report_v3_coordinated', 'C_OPEN_gwh', 'C_NRF_gwh',
                                                  'C_NRF_minus_C_OPEN_mwh')}, indent=1))
    print('=== A2', json.dumps({c: v['summary'] for c, v in a2.items()}, indent=1))
    print('=== A3-A5', json.dumps({c: {k: v[k] for k in ('terminal_ratios_per_channel', 'rule_ten',
                                                        'interface_consensus_residual_per_dso', 'settlement_identity')}
                                  for c, v in a35.items()}, indent=1))
    print('=== A6', json.dumps(a6, indent=1))
    if pb:
        print('=== B summary', json.dumps(pb['summary'], indent=1))
        print('=== B NRF rows', json.dumps(pb['nrf_rows_by_network'], indent=1))
        print('=== B four hours', json.dumps(pb['four_hours_both_copies'], indent=1))
    print('=== guards', json.dumps({k: guards[k] for k in ('solve_profile_guard', 'pickle_gate', 'build_guard')},
                                   indent=1, default=str))
    _log(f'wrote {out_json} sha256 {_sha(out_json)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
