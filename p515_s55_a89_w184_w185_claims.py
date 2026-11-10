"""
P5.15 Addendum 89 ruling 2 -- the claims W184 (E3) and W185 (E2) buy, computed from their committed records.
ZERO SOLVES, NO MODEL BUILD, NO PICKLE. Every number is read by eval key / path with its sha256; nothing is typed.

CONTEXT. W184 (one cell, e3_cnrf_x0_1x) and W185 (eight cells, e2_*) ran on 2026-10-26 under spec v1 (production
6f43479e), before Addendum 87's hold; Addendum 89 (2026-11-10) orders them as if not run. Its predictions P1 / P2 were
recorded on 2026-11-10, AFTER the runs: they are checked here against existing records, not as blind predictions.

PART 1 -- W184 -> the Section 4.4 sentence (ruling 2), on the price-taker denominator Q(P_NRF):
  X        = (Q(P_NRF) - Q(C_NRF)) / Q(P_NRF)     de-confounded saving (reverse flow prohibited on both sides)
  enabled  = (Q(P_NRF) - Q(C_OPEN)) / Q(P_NRF)    saving with reverse flow enabled on the coordinated exchange
  share    = (Q(C_NRF) - Q(C_OPEN)) / (Q(P_NRF) - Q(C_OPEN))   the four reverse-flow hours' share of the gap
  Sources: Q(P_NRF) = report_v3 per_arm_nrf.price_taker.q_best (= frozen tables benchmark.arms.price_taker.Q_best,
  = step7_e3_report difference_1.Q_P_NRF, bitwise); Q(C_OPEN) = frozen tables cells[eval_key d110bd1a...] (E.21
  row 1; = benchmark.coordinated_Q = report_v3 coordinated.q = step7_e3_report difference_2.Q_C_OPEN, bitwise);
  Q(C_NRF) = e3 campaign_results outcome.cell_report.Q_end (= step7_e3_report Q_C_NRF, bitwise).
  Resolutions: the numerator of `share` by rule v6 (certified pair): threshold max{3 x max(b_NRF, b_OPEN), 2 tau};
  the gaps by the declared procedure of W184 / the frozen benchmark (ADMM measure = max(band, the between-
  configuration spread at C*), static measure = the three-start spread). A bound construction for `share` is
  reported beside it: [(d - thr) / (G + m), (d + thr) / (G - m)], thr and m taken as bars (not a statistical CI).
  P1 (brief, Addendum 89): NRF raises Q(0) by < 10 % of the price-taker gap of 90,896.6 kEUR; X > 11 %; share < 1 %;
  margin stop if X < 9 %.  The manuscript's 12.2 % is checked against `enabled` (main.tex, Overleaf a87e195).

PART 2 -- W185 -> the spread table (ruling 2). For each price (1x, 2x): margin M = Q(0) - Q(unit) - I(unit) and the
  value V = Q(0) - Q(unit) at the baseline (E.21 rows 1, 2 / 34, 35, frozen tables by eval key) and at rho0 x 0.5 and
  x 2.0 (step7_e2_report_after_8of8 cells; = each campaign_results.json, bitwise); spread = max - min over the three;
  against each arm's rule-v6 threshold and the baseline's (recomputed and checked equal to the report's); H2 per pair
  (report, re-derived); rho read-back 8/8 (report and each campaign_results outcome.rho_read_back).
  P2: on every cell the certified Q lies within 3b of the frozen cell's window. The frozen tables carry (Q, band
  width) only; the window endpoints are read from the summaries behind them: rows 34, 35 from S6
  (w142_summary_after_34_l_195156fa.json, an input of the frozen tables by sha256), rows 1, 2 from the W101
  three-reference summary (the v6-from-records window); each is checked equal to the frozen (Q, band width), bitwise.
  Distance d = max(0, lo - Q_new, Q_new - hi). b read two ways: the frozen cell's band and the new cell's band.
  The frozen-JSON-only reading is reported beside: Q_frozen lies in its window of width b_f, so
  max(0, |dQ| - b_f) <= d <= |dQ|.  Sign / determinacy class of every decisive difference vs the baseline.

PART 3 -- code heads. For each of the nine runs: the cell spec's pins (production head, harness, hooks, launcher),
  the stage spec's pins, git_head_at_run; at that HEAD, the sha256 of every pinned production module and of the three
  run files read from the git object store (git show HEAD:path) against the pins; ancestry of 6f43479e; the commits
  after 6f43479e that touch production modules (tracked root-level *.py not matching ^p\\d), with their dates, against
  the runs' end times (f6c23977, 2026-10-31, is expected first).

GUARDS. SolveProfileGuard(permitted=()) installed before any project import, verified at exactly 0. pickle.load /
  loads / Unpickler replaced by refusers for the whole run (0 attempts required; one negative control must refuse).
  pyomo Model.__init__ blocked (0 required). Refuses: an output directory that is missing or holds anything but
  launch.log; a concurrent copy; a missing capture path (asserted before any reading); a pinned input whose sha256
  differs from its pin; any cross-check that fails.

OUTPUT (default data/SRP1/Results/P515S55/a89_w184_w185_claims/): a89_w184_w185_claims.json, the generated
  a89_w184_w185_claims_tables.md (the sentence and tables, numbers from the JSON), manifest_sha256.json;
  `--post-run` writes manifest_post_run_sha256.json over the outputs + launch.log. `--out-dir DIR` redirects (trials).

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S55/a89_w184_w185_claims && set -o noclobber && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_a89_w184_w185_claims.py \\
      > data/SRP1/Results/P515S55/a89_w184_w185_claims/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_a89_w184_w185_claims.py --post-run
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 A89 W184/W185 claims (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0, 'Unpickler': 0}


class PickleRefused(RuntimeError):
    pass


def _pickle_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise PickleRefused('A89 claims: pickle.load refused (no pickle is permitted)')


def _pickle_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise PickleRefused('A89 claims: pickle.loads refused (no pickle is permitted)')


class _BlockedUnpickler(pickle.Unpickler):
    def __init__(self, *_a, **_k):
        PICKLE_COUNTS['Unpickler'] += 1
        raise PickleRefused('A89 claims: pickle.Unpickler refused (no pickle is permitted)')


pickle.load, pickle.loads, pickle.Unpickler = _pickle_load, _pickle_loads, _BlockedUnpickler

from pyomo.core.base.PyomoModel import Model as _PyomoModel  # noqa: E402

BUILD_COUNTS = {'pyomo_model_init': 0}


def _blocked_model_init(self, *_a, **_k):
    BUILD_COUNTS['pyomo_model_init'] += 1
    raise RuntimeError('A89 claims: a Pyomo model build was attempted -- no model builds are permitted')


_PyomoModel.__init__ = _blocked_model_init

import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402  (the shared writer)

SCRIPT_REL = os.path.basename(__file__)
P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
P55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
DEFAULT_OUT_DIR = os.path.join(P55, 'a89_w184_w185_claims')

FROZEN = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
E3_REPORT = os.path.join(P55, 'e3_cnrf', 'step7_e3_report.json')
E3_STAGE_SPEC = os.path.join(P55, 'e3_cnrf', 'frozen_s55_step7_e3_spec_v1_bd2cfa4d.json')
E3_ROOT = os.path.join(P55, 'e3_cnrf', 'campaign_s55_step7_e3_cnrf_x0_1x')
E2_REPORT = os.path.join(P55, 'e2_rho', 'step7_e2_report_after_8of8.json')
E2_STAGE_SPEC = os.path.join(P55, 'e2_rho', 'frozen_s55_step7_e2_spec_v1_7631a6d7.json')
E2_DIR = os.path.join(P55, 'e2_rho')
REPORT_V3 = os.path.join(P53, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json')
W101_REF = os.path.join(P53, 'w101_srp1_continuation', 'w101_three_reference_summary.json')
S6 = os.path.join(P53, 'w142_resettle_v6', 'w142_summary_after_34_l_195156fa.json')
INVEST = os.path.join('data', 'SRP1', 'Results', 'P515S45', 'investment_cost', 'investment_cost_results.json')
MAIN_TEX = os.path.join('manuscript', '6a67305f25e8348fb71380c3', 'main.tex')
APPE_TEX = os.path.join(P55, 'w183h_round8', 'appE_generated_w183h.tex')
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'

PINS = {   # sha256 of each primary input, as recorded by the manifest named beside it
    FROZEN: '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4',      # name-embedded; e3/e2 manifests
    E3_REPORT: '0bb49f415a9ae331af5870373933efe882c42acf4dd8de0e4156f30963aa5ecb',   # step7_e3_report_manifest
    E2_REPORT: '628136efe0ed4aa082608ce49b44535948bcc52153e085e4aa9a943eff88b4d1',   # step7_e2_report_after_8of8_manifest
    os.path.join(E3_ROOT, 'campaign_results.json'):
        '0db2edf9af390131bf88a12aae9ff5200894663530a3e9aafea289e00e5bf627',          # step7_e3_report source
    REPORT_V3: 'bf6dbf7247a922020c58e7b9502f63539c0bbf048447c3e33bbbf202c152694a',   # frozen tables inputs BENCH
    S6: 'ba15d547133b5c291f4b5a87036482222dc370f29b22dfb9c3a739c2c9c8732c',          # frozen tables inputs S6
    W101_REF: '87234aedd7f9d63d11106a3433220856f4c058a178f05962dbf83f163f0acc29',    # its _manifest_sha256.json
    INVEST: '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b',      # e2 report records_read
    E3_STAGE_SPEC: 'bd2cfa4d5503cdc440f01c9e2370b00ce22dd13833ef9e4d8c3d04233a465b46',
    E2_STAGE_SPEC: '7631a6d7c28cfdde5fd3d7fa4ca6db0da29b3a784aa2b08491465a897ae66e66',
    MAIN_TEX: 'cec2e2982dfa1509e6e066a208521fb1eea40d5c71f1cdf71a9bf00a12b3520b',    # W184 pin, Overleaf a87e195
    APPE_TEX: '66bd3dd0a4d7b623232d95c420d1ba50cdb4bb0c075cfc0093af38e34b3a3d14',    # w183h_round8 manifest
}
PIN_MANIFESTS = {   # (manifest file, key) whose recorded sha256 must equal the pin
    W101_REF: (os.path.join(P53, 'w101_srp1_continuation', 'w101_three_reference_summary_manifest_sha256.json'),
               W101_REF),
    APPE_TEX: (os.path.join(P55, 'w183h_round8', 'manifest_sha256.json'), 'appE_generated_w183h.tex'),
}

C_OPEN_KEY = 'd110bd1a5977df1e811a1b3963afc565de6daa65f76dcc096ead3fdc70b08546'   # E.21 row 1
E3_KEY = '35b275180d897e4c123c37839b91f50b01454bb24085685d26e9d2b33346e12f'
BASELINE = {'1x': {'x0': 'd110bd1a5977df1e811a1b3963afc565de6daa65f76dcc096ead3fdc70b08546',
                   'unit': '3f084f2ffaeef2b78c9c6930d7f2134e66b61fbbefecef19a5ce0db8b1e9d153', 'rows': (1, 2),
                   'mult': None},   # the base price: the cell record carries no multiplier (None)
            '2x': {'x0': '354b1bdae2b86bce8638689641c7f32fee4c8c0aa896f1d50e9be6312ebee10d',
                   'unit': 'dca08628128736042fb1d1fa1a933b0b93f6d4cfea9ef8b4279e3c45d5a6139a', 'rows': (34, 35),
                   'mult': 2.0}}
WINDOW_SOURCE = {   # eval key -> (summary, path inside it)
    BASELINE['1x']['x0']: (W101_REF, ('reports', 'x0')),
    BASELINE['1x']['unit']: (W101_REF, ('reports', 'n7_4h_e1')),
    BASELINE['2x']['x0']: (S6, ('reports', 'h_50dea31c')),
    BASELINE['2x']['unit']: (S6, ('reports', 'h_74eda68d')),
}
UNIT_LABEL = 'n7_p0.25_e1.0'
E2_CELLS = [f'e2_{p}_{m}_rho{s}' for m in ('1x', '2x') for s in ('0p5', '2p0') for p in ('x0', 'unit')]
SCALE = {'0p5': 0.5, '2p0': 2.0}
PROD_HEAD = '6f43479ed8ece7a4a5878942ca38952ce1cae1bc'
EXPECTED_PINS = {'production_head': PROD_HEAD,
                 'harness': 'd9a944f220e791b82147f08d0e606126ad454b8b6b2994024f94e106798fb191',
                 'hooks': '9318ade3e6d0d5fec25d0c276c4d82fba5bb163c7da61ec10ee6f49aaf2fa1d2',
                 'launcher': '3b990d19bbde8e9fb6bf75d2e5447c55116d4a3d4d21c174e53776dffc561757'}
RUN_FILES = {'harness': 'p515_s44_campaign_harness.py', 'hooks': 'p515_s55_step7_resettle_hooks.py',
             'launcher': 'p515_s55_step7_launcher.py'}
LATER_PROD_COMMIT = 'f6c23977'
P1_LITERALS = ('raises Q(0) by less than 10 % of the price-taker gap of\n  90,896.6 k€', 'stays above 11 %',
               'carry less than 1 % of the gap', 'a de-confounded saving below 9 % stops for review')
P1 = {'raise_over_gap_lt': 0.10, 'X_gt': 0.11, 'share_lt': 0.01, 'X_stop_below': 0.09, 'gap_keur_1dp': 90896.6}

INPUTS = {}
_TRACKED = None


def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, check=True, capture_output=True).stdout


def _git_rc(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True).returncode


def _status(path):
    global _TRACKED
    if _TRACKED is None:
        _TRACKED = set(_git(['ls-files', '-z']).decode('utf-8').split('\0'))
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel not in _TRACKED:
        return 'NOT TRACKED' + (' (gitignored)' if _git_rc(['check-ignore', '-q', rel]) == 0 else '')
    dirty = _git(['status', '--porcelain', '--', rel]).decode().strip()
    return 'committed' if not dirty else 'committed, MODIFIED in working tree'


def _reg(path):
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel not in INPUTS:
        INPUTS[rel] = {'sha256': _sha(path), 'status': _status(path), 'size_bytes': os.path.getsize(path)}
    return INPUTS[rel]


def _json(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _text(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return f.read()


def _refuse(check, detail):
    _log(f'REFUSED (check: {check}): {detail}')
    raise SystemExit(f'REFUSED (check: {check}): {detail}')


def _require(cond, check, detail=''):
    if not cond:
        _refuse(check, detail)


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


def _dig(d, path):
    for k in path:
        d = d[k]
    return d


def _e2_root(cell):
    return os.path.join(E2_DIR, f'campaign_s55_step7_{cell}')


def _campaign_spec_path(root):
    hits = sorted(f for f in os.listdir(root) if f.startswith('campaign_spec_') and f.endswith('.json'))
    if len(hits) != 1:
        _refuse('one campaign spec per cell root', f'{root}: {hits}')
    return os.path.join(root, hits[0])


def _launch_json(root):
    ev = os.path.join(root, 'evals')
    dirs = sorted(os.listdir(ev))
    if len(dirs) != 1:
        _refuse('one eval dir per cell root', f'{ev}: {dirs}')
    return os.path.join(ev, dirs[0], 'launch.json')


# ============================================================================================ pre-run checks
def capture_checklist():
    """Rule eleven: every capture path and every field this script reads exists before any computation."""
    need = {p: p for p in PINS}
    for p, (m, _k) in PIN_MANIFESTS.items():
        need[f'manifest of {p}'] = m
    need['brief'] = BRIEF
    roots = {'e3_cnrf_x0_1x': E3_ROOT}
    roots.update({c: _e2_root(c) for c in E2_CELLS})
    for c, root in roots.items():
        need[f'{c} campaign_results.json'] = os.path.join(root, 'campaign_results.json')
        need[f'{c} evals/'] = os.path.join(root, 'evals')
    missing = sorted(k for k, p in need.items() if not os.path.exists(p))
    keys = {}
    if not missing:
        fz = json.load(open(FROZEN))
        cells = list(fz['tables']['cells'].values())
        for price, b in BASELINE.items():
            for role in ('x0', 'unit'):
                hits = [c for c in cells if c.get('eval_key') == b[role]]
                keys[f'frozen tables cells: exactly one cell with eval key {b[role][:16]} ({price} {role})'] = (
                    len(hits) == 1 and all(k in hits[0] for k in ('Q', 'band_width', 'status', 'k_star')))
        keys['frozen tables benchmark arms.price_taker.Q_best / coordinated_Q / benefit'] = all(
            k in fz['tables']['benchmark'] for k in ('arms', 'coordinated_Q', 'benefit', 'larger_band', 'multiple'))
        keys['frozen tables constants TAU'] = 'TAU' in fz['constants']
        e3 = json.load(open(E3_REPORT))
        keys['e3 report difference_1 / difference_2 / Q_C_NRF'] = all(
            k in e3 for k in ('difference_1', 'difference_2', 'Q_C_NRF', 'cell_report'))
        e2 = json.load(open(E2_REPORT))
        keys['e2 report: 8 cells, 4 pairs, H2_per_pair, rho read-back per cell'] = (
            set(e2['cells']) == set(E2_CELLS) and len(e2['pairs']) == 4 and len(e2['H2_per_pair']) == 4
            and set(e2['rho_read_back_holds_per_cell']) == set(E2_CELLS))
        for k_, (src, path) in WINDOW_SOURCE.items():
            s = json.load(open(src))
            r = _dig(s, path)
            keys[f'window source {os.path.basename(src)} {"/".join(path)} has band [lo, hi], eval key, window'] = (
                isinstance(r.get('band'), list) and len(r['band']) == 2 and r.get('eval_key') == k_
                and 'window' in r)
        rv3 = json.load(open(REPORT_V3))
        keys['report_v3 per_arm_nrf.price_taker.q_best and coordinated.q'] = (
            'q_best' in rv3['per_arm_nrf']['price_taker'] and 'q' in rv3['coordinated'])
        inv = json.load(open(INVEST))
        keys[f'investment cost candidates.{UNIT_LABEL}.I_new_eur'] = 'I_new_eur' in inv['candidates'][UNIT_LABEL]
        for c, root in roots.items():
            res = json.load(open(os.path.join(root, 'campaign_results.json')))
            keys[f'{c} campaign_results git_head_at_run, utc, outcome.cell_report'] = (
                'git_head_at_run' in res and 'utc' in res and 'cell_report' in res.get('outcome', {}))
            spec = json.load(open(_campaign_spec_path(root)))
            ex = spec.get('extra', {})
            keys[f'{c} campaign spec pins (production head, run files, hooks, launcher, harness)'] = (
                'production' in ex.get('pins', {}) and 'run_files_sha256' in ex.get('pins', {})
                and 'hooks_module' in ex and 'campaign_script_sha256' in ex and 'harness' in spec)
            keys[f'{c} launch.json started_utc'] = 'started_utc' in json.load(open(_launch_json(root)))
            if c.startswith('e2_'):
                keys[f'{c} outcome.rho_read_back.holds'] = 'holds' in res['outcome'].get('rho_read_back', {})
        brief = open(BRIEF, encoding='utf-8').read()
        a89 = brief[brief.index('# Addendum 89'):]
        for lit in P1_LITERALS:
            keys[f'brief Addendum 89 P1 contains {lit!r}'] = lit in a89
        keys['main.tex carries the 12.2 % sentence and Table 12 price-taker row'] = (
            '(12.2\\,\\%) relative to the price-taker benchmark' in open(MAIN_TEX, encoding='utf-8').read()
            and 'Price-taker NRF &' in open(MAIN_TEX, encoding='utf-8').read())
        keys['appE tex carries tab:appE_evaluations'] = '\\label{tab:appE_evaluations}' in open(
            APPE_TEX, encoding='utf-8').read()
    failed = sorted(k for k, v in keys.items() if not v)
    return {'ok': not missing and not failed, 'n_paths': len(need), 'missing_paths': missing,
            'n_key_checks': len(keys), 'failed_key_checks': failed}


def verify_pins():
    out = {}
    for p, s in PINS.items():
        actual = _sha(p)
        rec = None
        if p in PIN_MANIFESTS:
            m, k = PIN_MANIFESTS[p]
            man = json.load(open(m))
            rec = man.get(k) if isinstance(man.get(k), str) else (man.get('outputs') or {}).get(k)
            if rec is None:
                for v in man.values():
                    if isinstance(v, dict) and isinstance(v.get(k), str):
                        rec = v[k]
        out[p] = {'pin': s, 'actual': actual, 'manifest_record': rec,
                  'ok': actual == s and (rec is None or rec == s) and (p not in PIN_MANIFESTS or rec == s),
                  'status': _status(p)}
    return {'ok': all(v['ok'] for v in out.values()), 'files': out}


# ============================================================================================ helpers
def rule_v6(b0, bu, margin, tau, both_certified=True):
    thr = max(3.0 * max(b0, bu), 2.0 * tau)
    det = both_certified and abs(margin) >= thr
    return {'threshold': thr, 'binding_term': '3 x larger bar' if 3.0 * max(b0, bu) >= 2.0 * tau else '2 tau',
            'margin_over_threshold': abs(margin) / thr, 'verdict': 'determinate' if det else 'within bar',
            'sign': (margin > 0) - (margin < 0)}


def _f(x, nd=2):
    return f'{x:,.{nd}f}'


def _pct(x, nd):
    return f'{100.0 * x:.{nd}f}'


# ============================================================================================ part 1
def part1(fz, e3, rv3, tau, main_tex):
    bench = fz['tables']['benchmark']
    cells = list(fz['tables']['cells'].values())
    row1 = [c for c in cells if c['eval_key'] == C_OPEN_KEY]
    _require(len(row1) == 1, 'E.21 row 1 by eval key', len(row1))
    row1 = row1[0]
    res = _json(os.path.join(E3_ROOT, 'campaign_results.json'))
    cr = res['outcome']['cell_report']
    _require(res['eval_key'] == E3_KEY == cr['eval_key'] == e3['eval_key'], 'E3 instance identity')
    _require(cr['status'] == 'certified' and row1['status'] == 'certified', 'both cells certified')
    qp = rv3['per_arm_nrf']['price_taker']['q_best']
    qo = row1['Q']
    qn = cr['Q_end']
    xchk = {
        'Q_P_NRF report_v3 == frozen benchmark arms.price_taker.Q_best': qp == bench['arms']['price_taker']['Q_best'],
        'Q_P_NRF report_v3 == step7_e3_report difference_1.Q_P_NRF': qp == e3['difference_1']['Q_P_NRF'],
        'Q_C_OPEN frozen row 1 == benchmark coordinated_Q': qo == bench['coordinated_Q'],
        'Q_C_OPEN frozen row 1 == report_v3 coordinated.q': qo == rv3['coordinated']['q'],
        'Q_C_OPEN frozen row 1 == step7_e3_report difference_2.Q_C_OPEN': qo == e3['difference_2']['Q_C_OPEN'],
        'Q_C_NRF campaign_results == step7_e3_report Q_C_NRF': qn == e3['Q_C_NRF'],
        'benchmark instance is x = 0 (candidate key)': (bench['instance']['candidate_key'] == row1['candidate_key']
                                                         == cr['candidate_key']),
        'E3 cell is NRF (nrf_option True)': cr.get('nrf_option') is True,
    }
    _require(all(xchk.values()), 'part 1 cross-checks', [k for k, v in xchk.items() if not v])
    g_nrf = qp - qn            # coordination under NRF
    g_open = qp - qo           # coordination with reverse flow enabled (the price-taker gap)
    d = qn - qo                # NRF imposed on the coordinated exchange
    X = g_nrf / qp
    enabled = g_open / qp
    share = d / g_open
    xchk2 = {
        'gap Q(P_NRF) - Q(C_OPEN) == frozen benchmark.benefit (|diff| <= 1e-6 EUR)': abs(g_open - bench['benefit'])
        <= 1e-6,
        'difference Q(C_NRF) - Q(C_OPEN) == step7_e3_report difference_2.difference_eur (|diff| <= 1e-6)':
            abs(d - e3['difference_2']['difference_eur']) <= 1e-6,
        'difference Q(P_NRF) - Q(C_NRF) == step7_e3_report difference_1.difference_eur (|diff| <= 1e-6)':
            abs(g_nrf - e3['difference_1']['difference_eur']) <= 1e-6,
    }
    _require(all(xchk2.values()), 'part 1 difference cross-checks', [k for k, v in xchk2.items() if not v])
    b_nrf = cr['band_width']
    b_open = row1['band_width']
    rv = rule_v6(b_open, b_nrf, d, tau)
    _require(abs(rv['threshold'] - e3['difference_2']['resolution']['threshold']) <= 1e-6
             and rv['verdict'] == e3['difference_2']['resolution']['verdict'], 'rule v6 recomputed = e3 report')
    spread_cstar = rv3['coordinated']['reproducibility_band_eur']
    static_spread = rv3['per_arm_nrf']['price_taker']['multimodality_band_eur']
    m_nrf = max(b_nrf, spread_cstar)
    m_open = bench['larger_band']
    _require(m_open == spread_cstar, 'benchmark larger_band == report_v3 coordinated reproducibility band')
    _require(abs(m_nrf - e3['difference_1']['admm_cell_measure']) <= 1e-9, 'W184 ADMM measure recomputed')
    share_lo = (d - rv['threshold']) / (g_open + m_open)
    share_hi = (d + rv['threshold']) / (g_open - m_open)
    # manuscript
    lines = main_tex.splitlines()
    pct_lines = [{'line': i + 1, 'text': ln.strip()[:240]} for i, ln in enumerate(lines) if '12.2\\' in ln]
    t12_sec = main_tex[main_tex.index('\\label{tab:res_benchmark}'):]
    t12_sec = t12_sec[:t12_sec.index('\\end{tabular}')]
    t12 = [ln for ln in t12_sec.splitlines() if ln.strip().startswith('Price-taker NRF &')]
    _require(len(t12) == 1, 'one Table 12 (tab:res_benchmark) price-taker row in main.tex', len(t12))
    nums = re.findall(r'([\d\\,]+\.\d)\s*\(([\d.]+)\)', t12[0])
    t12_vals = [(float(a.replace('\\,', '').replace(',', '')), float(b)) for a, b in nums]
    manuscript = {
        'main_tex': MAIN_TEX, 'lines_with_12.2_percent': pct_lines,
        'table12_price_taker_row': t12[0].strip(), 'table12_parsed': t12_vals,
        'enabled_one_decimal': round(100.0 * enabled, 1),
        'enabled_equals_manuscript_12.2': round(100.0 * enabled, 1) == 12.2,
        'gap_keur_one_decimal': round(g_open / 1000.0, 1),
        'gap_over_Q_C_OPEN_one_decimal': round(100.0 * g_open / qo, 1),
        'table12_gap_and_13.9_reproduced': (len(t12_vals) >= 2 and t12_vals[1][0] == round(g_open / 1000.0, 1)
                                            and t12_vals[1][1] == round(100.0 * g_open / qo, 1)),
        'table12_Q_P_NRF_reproduced': len(t12_vals) >= 1 and t12_vals[0][0] == round(qp / 1000.0, 1),
        'X_one_decimal': round(100.0 * X, 1),
        'X_and_enabled_coincide_at_one_decimal': round(100.0 * X, 1) == round(100.0 * enabled, 1),
        'X_and_enabled_coincide_at_two_decimals': round(100.0 * X, 2) == round(100.0 * enabled, 2),
        'reading': ('the manuscript\'s 12.2 % (abstract, highlight, conclusions: "90.9 MEUR (12.2 %) relative to the '
                    'price-taker benchmark") is (Q(P_NRF) - Q(C_OPEN)) / Q(P_NRF); Table 12\'s 13.9 % is the same '
                    'gap over Q(C_OPEN)'),
    }
    p1 = {
        'recorded': '2026-11-10 (Addendum 89), after the run of 2026-10-26: checked against existing records',
        'gap_figure_from_records_keur_1dp': round(g_open / 1000.0, 1),
        'gap_figure_matches_P1_90896.6': round(g_open / 1000.0, 1) == P1['gap_keur_1dp'],
        'raise_of_Q0_over_gap': share, 'raise_lt_10pct': share < P1['raise_over_gap_lt'],
        'X': X, 'X_gt_11pct': X > P1['X_gt'],
        'share': share, 'share_lt_1pct': share < P1['share_lt'],
        'margin_stop_X_lt_9pct': X < P1['X_stop_below'],
        'note': ('the raise of Q(0) by NRF and the four hours\' share are one quantity, (Q(C_NRF) - Q(C_OPEN)) / '
                 '(Q(P_NRF) - Q(C_OPEN)), read against two thresholds (10 % and 1 %); its attribution to the four '
                 'hours rests on H3c (the binding set is exactly those four hours), not on an hour-by-hour split'),
    }
    p1['holds'] = (p1['gap_figure_matches_P1_90896.6'] and p1['raise_lt_10pct'] and p1['X_gt_11pct']
                   and p1['share_lt_1pct'] and not p1['margin_stop_X_lt_9pct'])
    sentence_1dp = (f'With reverse flow prohibited on the coordinated exchange as on the static arrangements, '
                    f'coordination saves {_pct(X, 1)} % ({_f(g_nrf / 1e6, 1)} M€); with reverse flow enabled, '
                    f'{_pct(enabled, 1)} % ({_f(g_open / 1e6, 1)} M€), both relative to the price-taker benchmark. '
                    f'The four reverse-flow interface-hours account for {_f(d / 1e3, 1)} k€, {_pct(share, 3)} % of '
                    f'the {_f(g_open / 1e3, 1)} k€ gap.')
    sentence_3dp = (f'With reverse flow prohibited on the coordinated exchange as on the static arrangements, '
                    f'coordination saves {_pct(X, 3)} %; with reverse flow enabled, {_pct(enabled, 3)} % (price-taker '
                    f'denominator {_f(qp / 1e3, 1)} k€); the four reverse-flow hours carry {_pct(share, 4)} % of the gap.')
    return {
        'objective_convention': 'Q = certified gross_operational_cost, contracted interface settlement excluded; '
                                'one market scenario; terminal salvage 0 (gross = net on all three)',
        'instances': {
            'P_NRF': {'Q': qp, 'source': f'{REPORT_V3} per_arm_nrf.price_taker.q_best',
                      'best_start': rv3['per_arm_nrf']['price_taker']['best_start'],
                      'candidate_key': bench['instance']['candidate_key']},
            'C_OPEN': {'Q': qo, 'source': f'{FROZEN} tables.cells[eval_key {C_OPEN_KEY[:16]}] (E.21 row 1, '
                                          f'{row1.get("name")})', 'eval_key': C_OPEN_KEY,
                       'band_width': b_open, 'k_star': row1['k_star'], 'status': row1['status']},
            'C_NRF': {'Q': qn, 'source': f'{E3_ROOT}/campaign_results.json outcome.cell_report.Q_end',
                      'eval_key': E3_KEY, 'band_width': b_nrf, 'k_star': cr['k_star'], 'status': cr['status'],
                      'branch': cr['branch']}},
        'cross_checks': {**xchk, **xchk2},
        'differences_eur': {'Q_P_NRF_minus_Q_C_NRF': g_nrf, 'Q_P_NRF_minus_Q_C_OPEN': g_open,
                            'Q_C_NRF_minus_Q_C_OPEN': d},
        'ratios': {'X_deconfounded': X, 'enabled': enabled, 'share_of_gap': share,
                   'definitions': {'X_deconfounded': '(Q(P_NRF) - Q(C_NRF)) / Q(P_NRF)',
                                   'enabled': '(Q(P_NRF) - Q(C_OPEN)) / Q(P_NRF)',
                                   'share_of_gap': '(Q(C_NRF) - Q(C_OPEN)) / (Q(P_NRF) - Q(C_OPEN))'}},
        'resolution': {
            'share_numerator_rule_v6': {**rv, 'bands': [b_nrf, b_open], 'tau': tau},
            'gap_nrf_declared_procedure': {'admm_measure': m_nrf, 'static_three_start_spread': static_spread,
                                           'over_admm_measure': g_nrf / m_nrf,
                                           'resolved': g_nrf > 3 * m_nrf and g_nrf > 3 * static_spread,
                                           'source': 'W184: max(C_NRF band, spread at C*); static three-start spread'},
            'gap_open_frozen_benchmark': {'larger_band': m_open, 'multiple': bench['multiple'],
                                          'determinate': bench['determinate'],
                                          'multiple_recomputed': g_open / m_open},
            'share_bound_construction': {'lo': share_lo, 'hi': share_hi,
                                         'definition': '[(d - thr) / (G + m), (d + thr) / (G - m)], thr = the rule-v6 '
                                                       'threshold of d, m = the gap\'s larger ADMM measure, taken as '
                                                       'bars (a bound construction, not a statistical interval)'}},
        'manuscript': manuscript,
        'P1': p1,
        'sentence_one_decimal': sentence_1dp,
        'sentence_three_decimals': sentence_3dp,
    }


# ============================================================================================ part 2
def part2(fz, e2, tau, inv):
    cells = list(fz['tables']['cells'].values())
    I_unit = e2['baseline']['1x']['I_unit']
    _require(I_unit == e2['baseline']['2x']['I_unit'], 'one I(unit) at both prices')
    I_w2 = inv['candidates'][UNIT_LABEL]['I_new_eur']
    _require(abs(I_unit - I_w2) <= 1e-6, 'I(unit) = W2 table', [I_unit, I_w2])
    windows = {}
    for k_, (src, path) in WINDOW_SOURCE.items():
        r = _dig(_json(src), path)
        fc = [c for c in cells if c['eval_key'] == k_]
        _require(len(fc) == 1, 'frozen cell by eval key', k_)
        fc = fc[0]
        lo, hi = r['band']
        q_src = r.get('Q_k_star', r.get('Q_cert_new_Q_k_star'))
        chk = {'eval_key': r['eval_key'] == k_, 'width_equals_frozen_band_width': (hi - lo) == fc['band_width'],
               'source_band_width_equals_frozen': r['band_width'] == fc['band_width'],
               'source_Q_k_star_equals_frozen_Q': q_src == fc['Q'], 'frozen_Q_in_window': lo <= fc['Q'] <= hi,
               'k_star_equal': r['k_star'] == fc['k_star']}
        _require(all(chk.values()), 'frozen window identity', {k_: chk})
        windows[k_] = {'lo': lo, 'hi': hi, 'width': hi - lo, 'cycles': r['window'], 'Q_frozen': fc['Q'],
                       'b_frozen': fc['band_width'], 'k_star': fc['k_star'], 'status': fc['status'],
                       'branch': fc.get('branch'), 'cell': fc['cell'],
                       'source': f'{src} {"/".join(path)}', 'checks': chk}
    # E.21 rows (appE tex) -- k* and band at one decimal
    tex = _text(APPE_TEX)
    sec = tex[tex.index('\\label{tab:appE_evaluations}'):]
    sec = sec[:sec.index('\\end{longtable}')]
    rows_tex = {}
    for ln in sec.splitlines():
        m = re.match(r'^\s*(\d+) & ', ln)
        if m and int(m.group(1)) in (1, 2, 34, 35):
            f = [x.strip() for x in ln.split('&')]
            rows_tex[int(m.group(1))] = {'k_star': int(f[6]), 'band_keur': float(f[8]), 'text': ln.strip()}
    e21 = {}
    for price, b in BASELINE.items():
        for role, row in zip(('x0', 'unit'), b['rows']):
            w = windows[b[role]]
            t = rows_tex.get(row)
            e21[row] = {'eval_key': b[role], 'tex': t,
                        'matches_frozen': (t is not None and t['k_star'] == w['k_star']
                                           and t['band_keur'] == round(w['b_frozen'] / 1000.0, 1))}
    _require(all(v['matches_frozen'] for v in e21.values()), 'E.21 rows 1, 2, 34, 35 = frozen cells', e21)
    # new cells
    new = {}
    for c in E2_CELLS:
        rc = e2['cells'][c]
        path = os.path.join(_e2_root(c), 'campaign_results.json')
        _require(_sha(path) == rc['source']['results_sha256'], 'campaign_results sha256 = e2 report source', c)
        res = _json(path)
        cr = res['outcome']['cell_report']
        price = c.split('_')[2]
        s = SCALE[c.split('rho')[1]]
        chk = {'Q': cr['Q_end'] == rc['Q'], 'band': cr['band'] == rc['band'] and cr['band_width'] == rc['band_width'],
               'eval_key': cr['eval_key'] == rc['eval_key'] == res['eval_key'], 'status': cr['status'] == rc['status'],
               'rho0_scale': cr['rho0_scale'] == s, 'flex_price_multiplier': cr['flex_price_multiplier'] ==
               BASELINE[price]['mult'],
               'candidate': cr['candidate_key'] == [x for x in cells if x['eval_key'] ==
                                                    BASELINE[price]['x0' if '_x0_' in c else 'unit']][0][
                   'candidate_key'],
               'rho_read_back_holds_both': (res['outcome']['rho_read_back']['holds'] is True
                                            and e2['rho_read_back_holds_per_cell'][c] is True),
               'rho_n_checks_equal': res['outcome']['rho_read_back']['n_checks'] == rc['rho_read_back_n_checks']}
        _require(all(chk.values()), 'e2 cell = campaign_results', {c: chk})
        new[c] = {'Q': cr['Q_end'], 'band': cr['band'], 'b_new': cr['band_width'], 'status': cr['status'],
                  'branch': cr['branch'], 'k_star': cr['k_star'], 'eval_key': cr['eval_key'], 'price': price,
                  'rho0_scale': s, 'role': 'x0' if '_x0_' in c else 'unit',
                  'rho_read_back': {'holds': res['outcome']['rho_read_back']['holds'],
                                    'n_checks': res['outcome']['rho_read_back']['n_checks'],
                                    'n_mismatches': len(res['outcome']['rho_read_back']['mismatches'])}}
    # P2 (a): Q within 3b of the frozen window
    p2_cells = {}
    for c, v in new.items():
        w = windows[BASELINE[v['price']][v['role']]]
        dist = max(0.0, w['lo'] - v['Q'], v['Q'] - w['hi'])
        dq = v['Q'] - w['Q_frozen']
        p2_cells[c] = {'Q_new': v['Q'], 'window_frozen': [w['lo'], w['hi']], 'frozen_cell': w['cell'],
                       'dQ_vs_frozen_Q': dq, 'distance_to_frozen_window': dist,
                       'inside_frozen_window': dist == 0.0,
                       'b_frozen': w['b_frozen'], 'b_new': v['b_new'],
                       'within_3b_frozen': dist <= 3 * w['b_frozen'], 'within_3b_new': dist <= 3 * v['b_new'],
                       'distance_over_3b_frozen': dist / (3 * w['b_frozen']),
                       'distance_over_3b_new': dist / (3 * v['b_new']),
                       'frozen_json_only_bounds': {'lo': max(0.0, abs(dq) - w['b_frozen']), 'hi': abs(dq),
                                                   'within_3b_frozen_certain': abs(dq) <= 3 * w['b_frozen'],
                                                   'within_3b_new_certain': abs(dq) <= 3 * v['b_new']}}
    # spread table
    spread = {}
    for price, b in BASELINE.items():
        bx = windows[b['x0']]
        bu = windows[b['unit']]
        rows = []
        mb = bx['Q_frozen'] - bu['Q_frozen'] - I_unit
        rvb = rule_v6(bx['b_frozen'], bu['b_frozen'], mb, tau,
                      bx['status'] == 'certified' and bu['status'] == 'certified')
        eb = e2['baseline'][price]
        _require(abs(mb - eb['margin']) <= 1e-6 and abs(rvb['threshold'] - eb['threshold']) <= 1e-6
                 and rvb['verdict'] == eb['verdict'] and rvb['sign'] == eb['sign'], 'baseline = e2 report', price)
        rows.append({'arm': 'baseline (rho0 x 1)', 'rows': list(b['rows']), 'Q_x0': bx['Q_frozen'],
                     'Q_unit': bu['Q_frozen'], 'V': bx['Q_frozen'] - bu['Q_frozen'], 'margin': mb, **rvb,
                     'bands': [bx['b_frozen'], bu['b_frozen']]})
        for s in ('0p5', '2p0'):
            cx, cu = new[f'e2_x0_{price}_rho{s}'], new[f'e2_unit_{price}_rho{s}']
            m = cx['Q'] - cu['Q'] - I_unit
            rv = rule_v6(cx['b_new'], cu['b_new'], m, tau,
                         cx['status'] == 'certified' and cu['status'] == 'certified')
            pr = [p for p in e2['pairs'] if p['price'] == price and p['rho0_scale'] == SCALE[s]]
            _require(len(pr) == 1, 'one e2 pair', (price, s))
            pr = pr[0]
            h2_shift = m - mb
            h2 = abs(h2_shift) <= rvb['threshold'] and rv['verdict'] == rvb['verdict'] and rv['sign'] == rvb['sign']
            _require(abs(m - pr['margin']) <= 1e-6 and abs(rv['threshold'] - pr['resolution']['threshold']) <= 1e-6
                     and rv['verdict'] == pr['verdict'] and h2 == pr['H2']['holds']
                     and h2 == e2['H2_per_pair'][f'{price}_rho{SCALE[s]}'], 'pair = e2 report', (price, s))
            rows.append({'arm': f'rho0 x {SCALE[s]}', 'cells': [f'e2_x0_{price}_rho{s}', f'e2_unit_{price}_rho{s}'],
                         'Q_x0': cx['Q'], 'Q_unit': cu['Q'], 'V': cx['Q'] - cu['Q'], 'margin': m, **rv,
                         'bands': [cx['b_new'], cu['b_new']], 'shift_vs_baseline': h2_shift, 'H2': h2,
                         'branches': [cx['branch'], cu['branch']]})
        ms = [r['margin'] for r in rows]
        sp = max(ms) - min(ms)
        thr = {r['arm']: r['threshold'] for r in rows}
        spread[price] = {'rows': rows, 'margin_min': min(ms), 'margin_max': max(ms), 'spread': sp,
                         'V_spread': max(r['V'] for r in rows) - min(r['V'] for r in rows),
                         'spread_over_threshold': {a: sp / t for a, t in thr.items()},
                         'spread_lt_every_threshold': all(sp < t for t in thr.values()),
                         'sign_kept': len({r['sign'] for r in rows}) == 1,
                         'class_kept': len({r['verdict'] for r in rows}) == 1,
                         'V_sign_kept': len({(r['V'] > 0) - (r['V'] < 0) for r in rows}) == 1}
    rho = {c: v['rho_read_back'] for c, v in new.items()}
    n_rho = sum(1 for v in rho.values() if v['holds'] is True and v['n_mismatches'] == 0)
    p2 = {
        'recorded': '2026-11-10 (Addendum 89), after the runs of 2026-10-26: checked against existing records',
        'a_within_3b_frozen_all': all(v['within_3b_frozen'] for v in p2_cells.values()),
        'a_within_3b_new_all': all(v['within_3b_new'] for v in p2_cells.values()),
        'b_sign_and_class_kept_all': all(v['sign_kept'] and v['class_kept'] for v in spread.values()),
        'c_rho_read_back': f'{n_rho}/{len(rho)}',
        'sign_stop': not all(v['sign_kept'] for v in spread.values()),
    }
    p2['holds_reading_b_frozen'] = (p2['a_within_3b_frozen_all'] and p2['b_sign_and_class_kept_all']
                                    and n_rho == len(rho))
    p2['holds_reading_b_new'] = p2['a_within_3b_new_all'] and p2['b_sign_and_class_kept_all'] and n_rho == len(rho)
    return {'objective_convention': 'Q = certified gross_operational_cost, contracted interface settlement excluded; '
                                    'margin = Q(0) - Q(unit) - I(unit); V = Q(0) - Q(unit)',
            'I_unit': I_unit, 'I_unit_w2': I_w2, 'tau': tau, 'frozen_windows': windows, 'e21_rows_tex': e21,
            'new_cells': new, 'P2_cells': p2_cells, 'spread': spread, 'rho_read_back': rho, 'P2': p2}


# ============================================================================================ part 3
def part3(e3stage, e2stage):
    out = {'stage_specs': {}, 'runs': {}}
    for name, sp, path in (('e3', e3stage, E3_STAGE_SPEC), ('e2', e2stage, E2_STAGE_SPEC)):
        p = sp['pins']
        out['stage_specs'][name] = {
            'path': path, 'git_head_at_freeze': sp['git_head'],
            'production_head': p['all']['production']['head'], 'harness': p['harness_sha256'],
            'hooks': p['hooks_sha256'], 'launcher': p['code_sha256'][RUN_FILES['launcher']],
            'equal_expected': (p['all']['production']['head'] == EXPECTED_PINS['production_head']
                               and p['harness_sha256'] == EXPECTED_PINS['harness']
                               and p['hooks_sha256'] == EXPECTED_PINS['hooks']
                               and p['code_sha256'][RUN_FILES['launcher']] == EXPECTED_PINS['launcher']
                               and p['code_sha256'][RUN_FILES['harness']] == EXPECTED_PINS['harness']
                               and p['code_sha256'][RUN_FILES['hooks']] == EXPECTED_PINS['hooks'])}
    prod_modules = e3stage['pins']['all']['production']['modules_sha256']
    _require(prod_modules == e2stage['pins']['all']['production']['modules_sha256'], 'one production pin set')
    roots = {'e3_cnrf_x0_1x': (E3_ROOT, E3_STAGE_SPEC)}
    roots.update({c: (_e2_root(c), E2_STAGE_SPEC) for c in E2_CELLS})
    blob_cache = {}

    def blob_sha(head, f):
        if (head, f) not in blob_cache:
            r = subprocess.run(['git', 'show', f'{head}:{f}'], cwd=REPO, capture_output=True)
            blob_cache[(head, f)] = _sha_bytes(r.stdout) if r.returncode == 0 else None
        return blob_cache[(head, f)]

    end_times = []
    for c, (root, stage_path) in roots.items():
        res = _json(os.path.join(root, 'campaign_results.json'))
        sp_path = _campaign_spec_path(root)
        spec = _json(sp_path)
        lj = _json(_launch_json(root))
        ex = spec['extra']
        head = res['git_head_at_run']
        pins = {'production_head': ex['pins']['production']['head'], 'harness': spec['harness']['sha256'],
                'harness_run_files': ex['pins']['run_files_sha256'][RUN_FILES['harness']],
                'hooks': ex['hooks_module']['sha256'],
                'hooks_run_files': ex['pins']['run_files_sha256'][RUN_FILES['hooks']],
                'launcher': ex['campaign_script_sha256'],
                'launcher_run_files': ex['pins']['run_files_sha256'][RUN_FILES['launcher']]}
        pins_ok = all(pins[k] == EXPECTED_PINS[k.split('_run_files')[0]] for k in pins)
        at_head_run = {k: blob_sha(head, f) == EXPECTED_PINS[k] for k, f in RUN_FILES.items()}
        prod_diff = sorted(f for f, s in prod_modules.items() if blob_sha(head, f) != s)
        spec_modules_equal = ex['pins']['production']['modules_sha256'] == prod_modules
        anc = _git_rc(['merge-base', '--is-ancestor', PROD_HEAD, head]) == 0
        out['runs'][c] = {
            'campaign_spec': sp_path, 'campaign_spec_sha256_in_results': res['campaign_spec_sha256'],
            'campaign_spec_sha256_actual': _sha(sp_path), 'stage_spec': res['stage_spec'],
            'stage_spec_sha256_actual': _sha(stage_path), 'git_head_at_run': head,
            'started_utc': lj['started_utc'], 'ended_utc': res['utc'], 'pins': pins, 'pins_equal_expected': pins_ok,
            'run_files_at_head_equal_pins': at_head_run,
            'production_modules_at_head_differing_from_pin': prod_diff,
            'campaign_spec_production_modules_equal_stage_pin': spec_modules_equal,
            'production_head_is_ancestor_of_run_head': anc,
            'all_ok': (pins_ok and all(at_head_run.values()) and not prod_diff and spec_modules_equal and anc
                       and res['campaign_spec_sha256'] == _sha(sp_path)
                       and res['stage_spec']['sha256'] == _sha(stage_path))}
        end_times.append(res['utc'])
    prod_at_6f = sorted(f for f, s in prod_modules.items() if blob_sha(PROD_HEAD, f) != s)
    # production modules by the pin's definition at HEAD
    tracked = _git(['ls-files']).decode().splitlines()
    prod_now = sorted(f for f in tracked if '/' not in f and f.endswith('.py') and not re.match(r'^p\d', f))
    union = sorted(set(prod_now) | set(prod_modules))
    log = _git(['log', '--format=%H|%cI|%s', f'{PROD_HEAD}..HEAD', '--'] + union).decode().splitlines()
    commits = []
    for ln in log:
        h, t, s = ln.split('|', 2)
        files = _git(['diff-tree', '--no-commit-id', '--name-only', '-r', h, '--'] + union).decode().split()
        commits.append({'commit': h, 'committer_date': t, 'subject': s[:160], 'production_files': files})
    commits.reverse()
    first = commits[0] if commits else None
    last_end = max(end_times)
    out['production_head_modules_equal_pin'] = not prod_at_6f
    out['production_head_commit'] = _git(['log', '-1', '--format=%H|%cI|%s', PROD_HEAD]).decode().strip()[:200]
    out['production_definition'] = e3stage['pins']['all']['production']['definition']
    out['production_modules_new_since_pin'] = sorted(set(prod_now) - set(prod_modules))
    out['commits_touching_production_after_6f43479e'] = commits
    out['first_later_production_commit'] = first
    out['first_later_is_f6c23977'] = bool(first and first['commit'].startswith(LATER_PROD_COMMIT))
    out['last_run_end_utc'] = last_end
    out['first_later_after_every_run'] = bool(first and datetime.fromisoformat(first['committer_date']) >
                                              datetime.fromisoformat(last_end))
    out['all_runs_ok'] = all(v['all_ok'] for v in out['runs'].values())
    out['n_runs'] = len(out['runs'])
    out['scope'] = ('pins read from each cell\'s campaign spec and the two stage specs; code at each run\'s '
                    'git_head_at_run read from the git object store and hashed; the run records carry no hash of the '
                    'files as executed, so "the working tree equalled HEAD at launch" rests on the spec\'s '
                    'uncommitted_changes = [] at freeze and the Worker reports\' pre-launch clean-tree observations')
    return out


# ============================================================================================ tables (markdown)
def tables_md(p1, p2, p3):
    L = []
    L.append('<!-- generated by p515_s55_a89_w184_w185_claims.py from a89_w184_w185_claims.json; do not edit -->')
    L.append('')
    L.append('### Part 1 -- W184, the Section 4.4 sentence (Q gross, settlement excluded)')
    L.append('')
    L.append('| quantity | € | source |')
    L.append('|---|---|---|')
    for k in ('P_NRF', 'C_OPEN', 'C_NRF'):
        i = p1['instances'][k]
        L.append(f"| Q({k}) | {_f(i['Q'])} | {i['source']} |")
    d = p1['differences_eur']
    r = p1['ratios']
    res = p1['resolution']
    L.append('')
    L.append('| ratio | definition | value | % (3 dp) | resolution |')
    L.append('|---|---|---|---|---|')
    L.append(f"| X (de-confounded) | {r['definitions']['X_deconfounded']} | {r['X_deconfounded']:.6f} | "
             f"{_pct(r['X_deconfounded'], 3)} | gap {_f(d['Q_P_NRF_minus_Q_C_NRF'])} € = "
             f"{res['gap_nrf_declared_procedure']['over_admm_measure']:.1f} x the ADMM measure "
             f"{_f(res['gap_nrf_declared_procedure']['admm_measure'])} € (static spread "
             f"{res['gap_nrf_declared_procedure']['static_three_start_spread']:.4f} €) |")
    L.append(f"| enabled | {r['definitions']['enabled']} | {r['enabled']:.6f} | {_pct(r['enabled'], 3)} | gap "
             f"{_f(d['Q_P_NRF_minus_Q_C_OPEN'])} € = {res['gap_open_frozen_benchmark']['multiple_recomputed']:.1f} x "
             f"the larger band {_f(res['gap_open_frozen_benchmark']['larger_band'])} € (frozen benchmark, "
             f"determinate {res['gap_open_frozen_benchmark']['determinate']}) |")
    nv = res['share_numerator_rule_v6']
    sb = res['share_bound_construction']
    L.append(f"| four hours' share | {r['definitions']['share_of_gap']} | {r['share_of_gap']:.6e} | "
             f"{_pct(r['share_of_gap'], 4)} | numerator {_f(d['Q_C_NRF_minus_Q_C_OPEN'])} € {nv['verdict']}, "
             f"{nv['margin_over_threshold']:.2f} x the rule-v6 threshold {_f(nv['threshold'])} €; bound "
             f"construction [{_pct(sb['lo'], 4)}, {_pct(sb['hi'], 4)}] % |")
    L.append('')
    L.append(f"Drafted sentence (manuscript precision): \"{p1['sentence_one_decimal']}\"")
    L.append('')
    L.append(f"At three decimals: \"{p1['sentence_three_decimals']}\"")
    L.append('')
    m = p1['manuscript']
    L.append(f"Manuscript: enabled at one decimal = {m['enabled_one_decimal']} % (equals the manuscript's 12.2 %: "
             f"{m['enabled_equals_manuscript_12.2']}); gap {m['gap_keur_one_decimal']} k€ = "
             f"{m['gap_over_Q_C_OPEN_one_decimal']} % of Q(C_OPEN) (Table 12 row reproduced: "
             f"{m['table12_gap_and_13.9_reproduced']}); X and enabled coincide at one decimal: "
             f"{m['X_and_enabled_coincide_at_one_decimal']}, at two: {m['X_and_enabled_coincide_at_two_decimals']}.")
    L.append('')
    q = p1['P1']
    L.append('| P1 clause | threshold | outcome | holds |')
    L.append('|---|---|---|---|')
    L.append(f"| price-taker gap figure | 90,896.6 k€ | {q['gap_figure_from_records_keur_1dp']} k€ | "
             f"{q['gap_figure_matches_P1_90896.6']} |")
    L.append(f"| NRF raises Q(0) by < 10 % of the gap | 10 % | {_pct(q['raise_of_Q0_over_gap'], 4)} % | "
             f"{q['raise_lt_10pct']} |")
    L.append(f"| de-confounded saving > 11 % | 11 % | {_pct(q['X'], 3)} % | {q['X_gt_11pct']} |")
    L.append(f"| four hours' share < 1 % | 1 % | {_pct(q['share'], 4)} % | {q['share_lt_1pct']} |")
    L.append(f"| margin stop if saving < 9 % | 9 % | {_pct(q['X'], 3)} % | stop: {q['margin_stop_X_lt_9pct']} |")
    L.append(f"| **P1** | | | **{q['holds']}** |")
    L.append('')
    L.append('### Part 2 -- W185, the spread table (Q gross, settlement excluded; margin = Q(0) - Q(unit) - I(unit), '
             f"I(unit) = {_f(p2['I_unit'])} €)")
    L.append('')
    L.append('| price | arm | Q(0) | Q(unit) | V = Q(0) - Q(unit) | margin | bands (0 / unit) | rule-v6 threshold | '
             '|margin| / thr | verdict, sign | shift vs baseline | H2 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for price, s in p2['spread'].items():
        for r_ in s['rows']:
            sh = _f(r_['shift_vs_baseline']) if 'shift_vs_baseline' in r_ else '--'
            h2 = str(r_['H2']) if 'H2' in r_ else '--'
            L.append(f"| {price} | {r_['arm']} | {_f(r_['Q_x0'])} | {_f(r_['Q_unit'])} | {_f(r_['V'])} | "
                     f"{_f(r_['margin'])} | {_f(r_['bands'][0])} / {_f(r_['bands'][1])} | {_f(r_['threshold'])} | "
                     f"{r_['margin_over_threshold']:.2f} | {r_['verdict']}, {'+' if r_['sign'] > 0 else '-'} | {sh} | "
                     f"{h2} |")
    L.append('')
    L.append('| price | margin range | spread (max - min, 3 arms) | spread / threshold: baseline, x0.5, x2.0 | '
             'spread < every threshold | sign kept | class kept |')
    L.append('|---|---|---|---|---|---|---|')
    for price, s in p2['spread'].items():
        rat = ', '.join(f'{v:.3f}' for v in s['spread_over_threshold'].values())
        L.append(f"| {price} | [{_f(s['margin_min'])}, {_f(s['margin_max'])}] | {_f(s['spread'])} | {rat} | "
                 f"{s['spread_lt_every_threshold']} | {s['sign_kept']} | {s['class_kept']} |")
    L.append('')
    L.append('| cell | Q new | frozen cell, window [lo, hi] | distance | b frozen | d / 3b_frozen | b new | '
             'd / 3b_new | frozen-JSON-only bounds on d | branch | rho read-back |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for c, v in p2['P2_cells'].items():
        nc = p2['new_cells'][c]
        fb = v['frozen_json_only_bounds']
        L.append(f"| {c} | {_f(v['Q_new'])} | {v['frozen_cell']} [{_f(v['window_frozen'][0])}, "
                 f"{_f(v['window_frozen'][1])}] | {_f(v['distance_to_frozen_window'])} | {_f(v['b_frozen'])} | "
                 f"{v['distance_over_3b_frozen']:.3f} | {_f(v['b_new'])} | {v['distance_over_3b_new']:.3f} | "
                 f"[{_f(fb['lo'])}, {_f(fb['hi'])}] | {nc['branch']} | {nc['rho_read_back']['holds']} "
                 f"({nc['rho_read_back']['n_checks']} checks, {nc['rho_read_back']['n_mismatches']} mismatches) |")
    L.append('')
    pp = p2['P2']
    L.append('| P2 clause | reading b = frozen band | reading b = new band |')
    L.append('|---|---|---|')
    L.append(f"| certified Q within 3b of the frozen window, every cell | {pp['a_within_3b_frozen_all']} | "
             f"{pp['a_within_3b_new_all']} |")
    L.append(f"| every decisive difference keeps sign and determinacy class | {pp['b_sign_and_class_kept_all']} | "
             f"{pp['b_sign_and_class_kept_all']} |")
    L.append(f"| rho read-back | {pp['c_rho_read_back']} | {pp['c_rho_read_back']} |")
    L.append(f"| sign stop | {pp['sign_stop']} | {pp['sign_stop']} |")
    L.append(f"| **P2** | **{pp['holds_reading_b_frozen']}** | **{pp['holds_reading_b_new']}** |")
    L.append('')
    L.append('### Part 3 -- code heads')
    L.append('')
    L.append('| run | git HEAD at run | started / ended (UTC) | pins = expected | run files at HEAD = pins | '
             'production modules at HEAD = pin | 6f43479e ancestor |')
    L.append('|---|---|---|---|---|---|---|')
    for c, v in p3['runs'].items():
        L.append(f"| {c} | `{v['git_head_at_run'][:8]}` | {v['started_utc'][:19]} / {v['ended_utc'][:19]} | "
                 f"{v['pins_equal_expected']} | {all(v['run_files_at_head_equal_pins'].values())} | "
                 f"{not v['production_modules_at_head_differing_from_pin']} | "
                 f"{v['production_head_is_ancestor_of_run_head']} |")
    L.append('')
    for n, s in p3['stage_specs'].items():
        L.append(f"- stage spec {n} ({s['path']}): production `{s['production_head'][:8]}`, harness "
                 f"`{s['harness'][:8]}`, hooks `{s['hooks'][:8]}`, launcher `{s['launcher'][:8]}`; equal expected: "
                 f"{s['equal_expected']}")
    L.append(f"- production modules at `6f43479e` equal the pin: {p3['production_head_modules_equal_pin']}")
    L.append(f"- commits after `6f43479e` touching production modules ({len(p3['commits_touching_production_after_6f43479e'])}):")
    for cm in p3['commits_touching_production_after_6f43479e']:
        L.append(f"  - `{cm['commit'][:8]}` {cm['committer_date']}: {', '.join(cm['production_files'])}")
    L.append(f"- first is `{LATER_PROD_COMMIT}`: {p3['first_later_is_f6c23977']}; after every run (last run ended "
             f"{p3['last_run_end_utc'][:19]}): {p3['first_later_after_every_run']}")
    L.append(f"- production modules tracked now but not in the pin: {p3['production_modules_new_since_pin']}")
    return '\n'.join(L) + '\n'


# ============================================================================================ main
def main():
    args = sys.argv[1:]
    out_dir = DEFAULT_OUT_DIR
    if '--out-dir' in args:
        out_dir = args[args.index('--out-dir') + 1]
    out_json = os.path.join(out_dir, 'a89_w184_w185_claims.json')
    out_md = os.path.join(out_dir, 'a89_w184_w185_claims_tables.md')
    out_man = os.path.join(out_dir, 'manifest_sha256.json')
    out_post = os.path.join(out_dir, 'manifest_post_run_sha256.json')
    if '--post-run' in args:
        rels = [out_json, out_md, out_man, os.path.join(out_dir, 'launch.log')]
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
    _log(f'repo HEAD {head}; tracked changes: {dirty}; script {script_status}; out {out_dir}')

    chk = capture_checklist()
    if not chk['ok']:
        _refuse('capture-path checklist', f"missing {chk['missing_paths']}; failed {chk['failed_key_checks']}")
    _log(f"capture-path checklist: {chk['n_paths']} paths, {chk['n_key_checks']} key checks, all present")
    pins = verify_pins()
    if not pins['ok']:
        _refuse('pinned inputs sha256', json.dumps({k: v for k, v in pins['files'].items() if not v['ok']}))
    _log(f'pins verified ({len(PINS)})')

    fz = _json(FROZEN)
    tau = fz['constants']['TAU']
    e3 = _json(E3_REPORT)
    e2 = _json(E2_REPORT)
    rv3 = _json(REPORT_V3)
    inv = _json(INVEST)
    _require(abs(2 * tau - e3['difference_2']['H3a_positive_and_within_resolution_lt_2tau']['two_tau']) <= 1e-9,
             'tau = e3 report two_tau / 2')
    main_tex = _text(MAIN_TEX)
    _text(BRIEF)
    p1 = part1(fz, e3, rv3, tau, main_tex)
    _log(f"part 1: X {p1['ratios']['X_deconfounded']:.6f}, enabled {p1['ratios']['enabled']:.6f}, share "
         f"{p1['ratios']['share_of_gap']:.6e}; P1 holds {p1['P1']['holds']}")
    p2 = part2(fz, e2, tau, inv)
    _log(f"part 2: spread 1x {p2['spread']['1x']['spread']:.2f}, 2x {p2['spread']['2x']['spread']:.2f}; P2 "
         f"(b frozen) {p2['P2']['holds_reading_b_frozen']}, (b new) {p2['P2']['holds_reading_b_new']}")
    p3 = part3(_json(E3_STAGE_SPEC), _json(E2_STAGE_SPEC))
    _log(f"part 3: {p3['n_runs']} runs, all ok {p3['all_runs_ok']}; first later production commit "
         f"{(p3['first_later_production_commit'] or {}).get('commit')}")

    nc = {'pickle.loads(bytes) refused': False}
    try:
        pickle.loads(b'\x80\x04N.')
    except PickleRefused:
        nc['pickle.loads(bytes) refused'] = True
    gfail = GUARD.verify(0)
    pickle_ok = (PICKLE_COUNTS == {'load': 0, 'loads': 1, 'Unpickler': 0} and nc['pickle.loads(bytes) refused']
                 and pickle.load is _pickle_load and pickle.loads is _pickle_loads)
    build_ok = BUILD_COUNTS['pyomo_model_init'] == 0
    guards = {'solve_profile_guard': {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_0_failures': gfail},
              'pickle_block': {'counts': dict(PICKLE_COUNTS), 'expected': {'load': 0, 'loads': 1, 'Unpickler': 0},
                               'note': 'the one loads attempt is the negative control', 'negative_control': nc,
                               'ok': pickle_ok},
              'build_guard': {'blocked': 'pyomo.core.base.PyomoModel.Model.__init__', 'counts': dict(BUILD_COUNTS),
                              'ok': build_ok}}
    if gfail or not pickle_ok or not build_ok:
        _log(f'GUARD FAULT: {json.dumps(guards, default=str)}')
        raise SystemExit('guard fault')
    result = {
        'schema': 'p515_s55_a89_w184_w185_claims_v1',
        'stage': 'P5.15 Addendum 89 ruling 2: the W184 / W185 claims from their committed records (zero solves)',
        'started_utc': started, 'git_HEAD': head, 'tracked_changes_at_start': dirty,
        'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'git_status': script_status},
        'interpreter': sys.executable,
        'timing_note': ('W184 / W185 ran 2026-10-26 under spec v1 (production 6f43479e); Addendum 89 and its P1 / P2 '
                        'were recorded 2026-11-10, after the runs: P1 / P2 are checked against existing records, not '
                        'blind predictions'),
        'pre_run_checks': {'capture_checklist': chk, 'pins': pins},
        'tau': tau,
        'part1_w184_section_4_4': p1,
        'part2_w185_spread_table': p2,
        'part3_code_heads': p3,
        'guards': guards,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    with open(out_json, 'x', encoding='utf-8') as fh:
        GRIO.dump(result, fh, indent=1, sort_keys=False)
        fh.write('\n')
    with open(out_md, 'x', encoding='utf-8') as fh:
        fh.write(tables_md(p1, p2, p3))
    manifest = {'outputs': {out_json: _sha(out_json), out_md: _sha(out_md)},
                'script': {SCRIPT_REL: {'sha256': _sha(SCRIPT_REL), 'git_status': script_status}},
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)}, 'n_inputs': len(INPUTS)}
    with open(out_man, 'x', encoding='utf-8') as fh:
        GRIO.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')
    print(open(out_md, encoding='utf-8').read())
    print('=== guards', json.dumps(guards, indent=1, default=str))
    _log(f'wrote {out_json} sha256 {_sha(out_json)}; {out_md} sha256 {_sha(out_md)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
