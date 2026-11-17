"""
P5.15 Addendum 79 order, Planner task W180b -- item (b), "Row 16". ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION
IMPORT. Static reading of committed records and of the Overleaf clone at e5322f8 (read only: `git rev-parse`,
`git show`, file reads; no fetch, pull, checkout or status).

QUESTION. For every row of Table E.21 (`tab:appE_evaluations`) of main.tex at Overleaf e5322f8 (61 rows), establish
  (1) the rule version under which its printed certificate (or uncertified status) was ISSUED;
  (2) whether its record carries a v6 DECISION -- the v6 run itself (stage specs v6 96c23404, ext v3 84775dc4, A64 v1
      44a2dce8), or v6 applied to the committed record of the issuing run (W142 item 1 `item1_v6_on_every_record`,
      W142 item 2 `d_c52e1670_v6_from_records_certificate.json`, W168 for x = 0) -- and whether it is a reference
      continuation (W163 N49: the W101 continuations x0, n7_4h_e1, C*);
  (3) whether the printed Status / Rule / k* are consistent with (1) and (2).
Prediction (expert, Addendum 79 (b)): 0 further rows -- besides row 16 -- print a certificate issued under an earlier
rule version without a v6 decision on its record.

DEFINITIONS (operational; the formula is preserved here and in the output JSON `definitions`).
  issued_rule_version: the integer version of the settling-rule class named by the stop_rule.class of the stage spec
      under which the printed decision was made (settling_criterion.SettlingRule -> 1, settling_criterion_vN.* -> N);
      for d_c52e1670 the certificate itself is v6 ("v6 from records", W142 item 2).
  v6 decision: one of
      'v6 run'                       the issuing run is a v6-stage run (its campaign_results G15 is the v6 decision);
      'v6 from records (W142 item 2)' d_c52e1670's v6 certificate written from its committed v5 run;
      'v6 on record (W142 item 1)'   W142 item 1 entry whose floor-replay instance (candidate_key, eval dir key prefix)
                                     equals the frozen cell's (candidate_key, eval_key[:16]);
      none                           no entry found in the searched sources (listed in `search_scope`).
  v6 decision "decided" = status in {certified, uncertified}; "not decided" = any other status string.
  expected Status: v6 certified -> 'cert.' (+ ', osc.'/', mon.' when the branch is printed); v6 uncertified -> the W153
      cause formula (p515_s53_w153_step5_zero_solve_rows.py l. 645) applied to the v6 decision's fields:
      'gap clause' if gap_refusals else 'lapse reset' if lapse_events else 'growth test' if 'swings_growing' in reasons
      else 'other'; v6 not decided and frozen `superseded` -> 'superseded'.
  expected Rule: 'run' for kind 'v6 run'; 'record' for the two from-records kinds when decided; '---' when not decided.
  expected k*/end: v6 k_star (certified) or k_cap (uncertified); for a not-decided row, the issuing certificate's k*.
  earlier_rule_certificate_without_v6_decision (the prediction's quantity): issued_rule_version < 6 AND the printed
      Status is a certificate or a superseded certificate AND NOT (v6 decision certified at the printed k*).

Guards: `SolveProfileGuard(permitted=())` installed before any project import and verified at exactly 0 at the end;
`pickle.load` / `pickle.loads` blocked for the whole run and counted (0 expected). No production module imported
(asserted). Refuses to run unless the output directory exists and holds nothing but launch.log, and refuses to run
concurrently with another copy of itself. Booleans are written through the shared writer `gate_result_io`.

Output: data/SRP1/Results/P515S53/w180b_row16_class/w180b_row16_class.json and manifest_sha256.json (every input with
sha256 and git-tracked status; the output JSON; this script). `--post-run` afterwards writes
manifest_post_run_sha256.json over the outputs and launch.log.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w180b_row16_class && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w180b_row16_class.py \\
      > data/SRP1/Results/P515S53/w180b_row16_class/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w180b_row16_class.py --post-run
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W180b row-16 classification at Overleaf e5322f8 (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W180b: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W180b: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import ast  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer that refuses string flags)

SCRIPT_REL = os.path.basename(__file__)
P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(P53, 'w180b_row16_class')
OUT_JSON = os.path.join(OUT_DIR, 'w180b_row16_class.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
OUT_POST = os.path.join(OUT_DIR, 'manifest_post_run_sha256.json')

CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
CLONE_HEAD = 'e5322f85dced8034312c9afb97c19001bce6bfa0'
MAIN_SHA = 'b3282295ece4cc3b0f9bc4ff5cd00ad2d86a7e60650221bfd2d0bd00fa61c492'

FROZEN = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FROZEN_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
GEN_APPE = os.path.join('manuscript_review', 'gen_appE.py')
T2_CSV = os.path.join('manuscript_review', 'export_W160', 'T2.csv')
W142_V6R = os.path.join(P53, 'w142_resettle_v6', 'v6_from_records', 'w142_v6_from_records.json')
W142_CERT_D = os.path.join(P53, 'w142_resettle_v6', 'v6_from_records', 'd_c52e1670_v6_from_records_certificate.json')
W142_FLOOR = os.path.join(P53, 'w142_resettle_v6', 'floor_replay', 'w142_floor_replay.json')
W168 = os.path.join(P53, 'w168_x0_v6_replay', 'w168_x0_v6_replay_decision.json')
W163 = os.path.join(P53, 'w163_paragraphs_v4_check', 'w163_paragraphs_v4_figure_check.json')
W101_SUMMARY = os.path.join(P53, 'w101_srp1_continuation', 'w101_three_reference_summary.json')
W118_SUMMARY = os.path.join(P53, 'w118_resettle', 'w118_resettle_summary.json')
W153_SRC = 'p515_s53_w153_step5_zero_solve_rows.py'
SPEC = {
    'v39_W101': os.path.join(P53, 'frozen_s53_spec_v39_8a612429.json'),
    'v2_W118': os.path.join(P53, 'w118_resettle', 'frozen_s53_resettle_spec_v2_fc791891.json'),
    'v4': os.path.join(P53, 'w137_resettle_v4', 'frozen_s53_resettle_spec_v4_e11fbc89.json'),
    'v5': os.path.join(P53, 'w139_resettle_v5', 'frozen_s53_resettle_spec_v5_ab32ffc9.json'),
    'v6': os.path.join(P53, 'w142_resettle_v6', 'frozen_s53_resettle_spec_v6_96c23404.json'),
    'ext_v3': os.path.join(P53, 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json'),
    'a64_v1': os.path.join(P53, 'w155_a64_cells', 'frozen_s53_a64_cells_spec_v1_44a2dce8.json'),
}
V6_RUN_SPECS = ('v6', 'ext_v3', 'a64_v1')
EXPECTED_ROW = 16
EXPECTED_ROW_CELL = 'pb_y2025_n5'
FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions',
                             'settling_criterion')

# ------------------------------------------------------------------------------------------------------ input registry
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


def _git(args, cwd=REPO):
    return subprocess.run(['git'] + list(args), cwd=cwd, check=True, capture_output=True).stdout


def _tracked(path):
    global _TRACKED
    if _TRACKED is None:
        _TRACKED = set(_git(['ls-files', '-z']).decode('utf-8').split('\0'))
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel.startswith(CLONE + os.sep):
        return 'overleaf clone (read only)'
    return 'committed' if rel in _TRACKED else 'UNCOMMITTED'


def _reg(path):
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel not in INPUTS:
        INPUTS[rel] = {'sha256': _sha(path), 'status': _tracked(path)}
    return INPUTS[rel]


def _read(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return f.read()


def _read_json(path):
    return json.loads(_read(path))


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


# ------------------------------------------------------------------------------------------------------- manuscript
def read_manuscript():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).decode().strip()
    blob = _git(['show', f'{CLONE_HEAD}:main.tex'], cwd=CLONE)
    blob_sha = _sha_bytes(blob)
    wc = os.path.join(CLONE, 'main.tex')
    wc_sha = _reg(wc)['sha256']
    INPUTS[f'{CLONE}@{CLONE_HEAD}:main.tex (git show)'] = {'sha256': blob_sha, 'status': 'overleaf clone blob'}
    text = blob.decode('utf-8')
    return text, {'clone': CLONE, 'clone_head': head, 'declared_head': CLONE_HEAD, 'head_ok': head == CLONE_HEAD,
                  'blob_sha256': blob_sha, 'declared_sha256': MAIN_SHA, 'blob_sha_ok': blob_sha == MAIN_SHA,
                  'working_copy_sha256': wc_sha, 'working_copy_equals_blob': wc_sha == blob_sha,
                  'n_lines': len(text.split('\n')),
                  'read_only_operations': ['git rev-parse HEAD', f'git show {CLONE_HEAD}:main.tex', 'read main.tex']}


def row_cells(line):
    t = line.strip()
    if t.endswith('\\\\'):
        t = t[:-2]
    return [c.strip() for c in re.split(r'(?<!\\)&', t)]


def strip_markers(s):
    marks = re.findall(r'\$\^\{([a-z])\}\$', s)
    clean = re.sub(r'\$\^\{([a-z])\}\$', '', s).strip()
    return clean, marks


def parse_e21(lines):
    i0 = next(i for i, ln in enumerate(lines) if r'\label{tab:appE_evaluations}' in ln)
    i1 = next(i for i in range(i0, len(lines)) if r'\end{longtable}' in lines[i])
    rows = []
    for i in range(i0, i1):
        if not re.match(r'^\d+ & ', lines[i].strip()):
            continue
        c = row_cells(lines[i])
        if len(c) != 14:
            raise RuntimeError(f'E.21 l. {i + 1}: {len(c)} cells')
        st, marks = strip_markers(c[4])
        rows.append({'no': int(c[0]), 'line': i + 1, 'plan': c[1], 'year': c[2], 'condition': c[3],
                     'status_printed': c[4], 'status': st, 'marks': marks, 'k0': c[5], 'kstar_end': c[6],
                     'range': c[7], 'band': c[8], 'gap': c[9], 'slack': c[10], 'nc': c[11], 'replay': c[12],
                     'rule': c[13]})
    caption = next(lines[i] for i in range(i0 - 3, i1) if '\\caption{' in lines[i])
    return rows, caption, (i0 + 1, i1 + 1)


def decode_plan(txt):
    t = txt.strip()
    if 'boldsymbol{x}=\\boldsymbol{0}' in t:
        return {}
    out = {}
    for part in t.split(';'):
        m = re.fullmatch(r'([0-9, ]+):\s*([0-9.]+)/([0-9.]+)(\s+each)?', part.strip())
        if not m:
            raise ValueError(f'plan text not decodable: {txt!r}')
        for n in m.group(1).split(','):
            out[n.strip()] = (float(m.group(2)), float(m.group(3)))
    return out


def cand_plan(c):
    return {n: (float(v[0]), float(v[1])) for n, v in c['candidate_canonical']['nodes'].items() if v[0] or v[1]}


def dec_places(s):
    return len(s.split('.')[1]) if '.' in s else 0


def at_print(printed, value, scale=1.0):
    """True if |printed - value/scale| <= half a unit of the printed last digit (+1e-9); None if not comparable."""
    p = printed.strip()
    if value is None or p in ('---', ''):
        return None
    v = value / scale
    return abs(float(p) - v) <= 0.5 * 10 ** (-dec_places(p)) + 1e-9


# --------------------------------------------------------------------------------------------------- generator (ast)
def gen_appE_static():
    src = _read(GEN_APPE)
    tree = ast.parse(src)
    calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'append' and isinstance(n.func.value, ast.Name) and n.func.value.id == 'G'),
                   key=lambda n: n.lineno)
    consts = {}
    for n in tree.body:   # module-level string constants (e.g. X0), resolved without executing the generator
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
            consts[n.targets[0].id] = n.value.value
    order = []
    for c in calls:
        for row in c.args[0].elts[1].elts:
            order.append(tuple(e.value if isinstance(e, ast.Constant) else consts[e.id] for e in row.elts))
    superseded = None
    rule_src = status_src = None
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SUPERSEDED' for t in n.targets):
            superseded = sorted(ast.literal_eval(n.value))
        if isinstance(n, ast.FunctionDef) and n.name == 'rule_col':
            rule_src = ast.get_source_segment(src, n)
        if isinstance(n, ast.FunctionDef) and n.name == 'status':
            status_src = ast.get_source_segment(src, n)
    return order, {'path': GEN_APPE, 'executed': False, 'SUPERSEDED': superseded, 'rule_col_source': rule_src,
                   'status_source': status_src}


# ------------------------------------------------------------------------------------------------------- evidence
def rule_version_of_spec(spec):
    cls = (spec.get('stop_rule') or {}).get('class') or ''
    m = re.fullmatch(r'settling_criterion(?:_v(\d+))?\.\w+', cls)
    if not m:
        raise RuntimeError(f'stop_rule.class not recognised: {cls!r}')
    return (int(m.group(1)) if m.group(1) else 1), cls


def g15(cr):
    g = (cr.get('gate_detail') or {}).get('G15') or {}
    return {'decision_status': g.get('decision_status'), 'k_star': g.get('k_star'), 'k_cap': g.get('k_cap'),
            'cycles_run': g.get('cycles_run'), 'stopped_by': g.get('stopped_by')}


def w153_cause(gap_refusals, lapse_events, reasons):
    """The W153 cause formula, verbatim in logic (p515_s53_w153_step5_zero_solve_rows.py l. 645)."""
    gaps = gap_refusals or []
    lapses = lapse_events or []
    reasons = reasons or []
    return ('gap clause' if gaps else 'lapse reset' if lapses else 'growth test' if 'swings_growing' in reasons
            else 'other')


def assess_row(r, v6, superseded, issued_version, issuing_k_star):
    """Expected Status / Rule / k* from the v6 decision, the consistency of the printed entries, and the prediction's
    flag. Pure: used on every row and, unchanged, by the mutation self-test."""
    decided = v6 is not None and v6['status'] in ('certified', 'uncertified')
    if v6 is None:
        exp_status, exp_rule, exp_k = None, None, None
    elif v6['status'] == 'certified':
        exp_status, exp_rule, exp_k = 'cert.', ('run' if v6['kind'] == 'v6 run' else 'record'), v6['k_star']
    elif v6['status'] == 'uncertified':
        exp_status, exp_rule, exp_k = v6['cause_w153'], ('run' if v6['kind'] == 'v6 run' else 'record'), v6['k_cap']
    else:
        exp_status, exp_rule, exp_k = ('superseded' if superseded else None), '---', issuing_k_star
    printed_cert = r['status'].startswith('cert.')
    status_ok = (r['status'] == exp_status) if exp_status != 'cert.' else printed_cert
    branch_printed = {'cert., osc.': 'oscillatory', 'cert., mon.': 'monotone'}.get(r['status'])
    branch_ok = (branch_printed is None) or (v6 is not None and v6.get('branch') == branch_printed)
    k_ok = str(exp_k) == r['kstar_end']
    rule_ok = r['rule'] == exp_rule
    certd = bool(v6 and v6['status'] == 'certified')
    rng_ok = at_print(r['range'], v6.get('range_over_tau')) if certd else None
    band_ok = at_print(r['band'], v6.get('band_width'), 1000.0) if certd else None
    marks_ok = ('s' in r['marks']) == superseded
    v6_cert_at_printed = bool(certd and str(v6['k_star']) == r['kstar_end'])
    prints_cert = printed_cert or r['status'] == 'superseded'
    flagged = bool(issued_version < 6 and prints_cert and not v6_cert_at_printed)
    entries_ok = bool(status_ok and branch_ok and k_ok and rule_ok and marks_ok and rng_ok is not False
                      and band_ok is not False)
    return {'decided': decided, 'exp_status': exp_status, 'exp_rule': exp_rule, 'exp_k': exp_k,
            'status_ok': status_ok, 'branch_ok': branch_ok, 'k_ok': k_ok, 'rule_ok': rule_ok, 'rng_ok': rng_ok,
            'band_ok': band_ok, 'marks_ok': marks_ok, 'v6_cert_at_printed': v6_cert_at_printed, 'flagged': flagged,
            'entries_ok': entries_ok}


def mutation_self_test(out_rows):
    """Negative controls: each planted defect must change the verdict it targets; the unmutated rows must not."""
    import copy
    by = {x['no']: x for x in out_rows}

    def run(no, mut_printed=None, mut_v6=None, mut_sup=None, mut_ver=None):
        x = by[no]
        r = copy.deepcopy(x['printed'])
        v6 = copy.deepcopy(x['v6_decision'])
        sup = x['frozen']['superseded']
        ver = x['issuing']['issued_rule_version']
        if mut_printed:
            r.update(mut_printed)
        if mut_v6:
            v6.update(mut_v6)
        if mut_sup is not None:
            sup = mut_sup
        if mut_ver is not None:
            ver = mut_ver
        return assess_row(r, v6, sup, ver, x['issuing']['G15']['k_star'])

    cases = {
        'M1 row 17 (v2 cert): v6 on record planted "not decided" -> flagged':
            run(17, mut_v6={'status': 'not decided within the recorded cycles 1..172', 'k_star': None})['flagged'] is True,
        'M2 row 17: v6 certified at a different cycle (171) -> flagged and k* inconsistent':
            (lambda a: a['flagged'] is True and a['k_ok'] is False)(run(17, mut_v6={'k_star': 171})),
        'M3 row 16 restored to the 48099ab print ("cert., osc." / "record") -> flagged, entries inconsistent':
            (lambda a: a['flagged'] is True and a['entries_ok'] is False)(
                run(16, mut_printed={'status': 'cert., osc.', 'marks': ['s'], 'rule': 'record'})),
        'M4 row 5 (v6 run) printed Rule "record" -> rule inconsistent, not flagged (issued under v6)':
            (lambda a: a['rule_ok'] is False and a['flagged'] is False)(run(5, mut_printed={'rule': 'record'})),
        'M5 row 12 (v4 cert) printed Rule "run" -> rule inconsistent':
            run(12, mut_printed={'rule': 'run'})['rule_ok'] is False,
        'M6 row 40 (v2 uncertified) v6 cause changed (no gap refusals, lapse) -> status inconsistent, not flagged':
            (lambda a: a['status_ok'] is False and a['flagged'] is False)(run(40, mut_v6={'cause_w153': 'lapse reset'})),
        'M7 row 3 issued v6 -> not flagged even if v6 decision removed':
            run(3, mut_v6={'status': 'not decided', 'k_star': None})['flagged'] is False,
        'M8 row 18 printed band 4.4 against v6 band 3.70 k EUR -> band inconsistent':
            run(18, mut_printed={'band': '4.4'})['band_ok'] is False,
        'C0 every row unmutated: flag set == {16}':
            {no for no in by if run(no)['flagged']} == {16},
    }
    return {'cases': cases, 'all_pass': all(cases.values())}


def issuing_summary_q(w142_key, q6, w118, w101):
    """Bitwise identity of the v6-on-record Q(k*) with the issuing stage's own summary Q(k*), where one exists:
    W118 summary reports.<key>.Q_k_star; W101 summary reports.<x0|n7_4h_e1>.Q_cert_new_Q_k_star. None otherwise."""
    if q6 is None:
        return None
    if w142_key in w118['reports']:
        q = w118['reports'][w142_key].get('Q_k_star')
        return None if q is None else q == q6
    k101 = {'x0': 'x0', 'unit_n7_4h_e1': 'n7_4h_e1'}.get(w142_key)
    if k101:
        return w101['reports'][k101].get('Q_cert_new_Q_k_star') == q6
    return None


def main():
    global OUT_DIR, OUT_JSON, OUT_MANIFEST, OUT_POST
    if '--out-dir' in sys.argv:                    # trial runs only (a scratch directory holding only launch.log)
        OUT_DIR = sys.argv[sys.argv.index('--out-dir') + 1]
        OUT_JSON = os.path.join(OUT_DIR, 'w180b_row16_class.json')
        OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
        OUT_POST = os.path.join(OUT_DIR, 'manifest_post_run_sha256.json')
    if '--post-run' in sys.argv:
        rels = [OUT_JSON, OUT_MANIFEST, os.path.join(OUT_DIR, 'launch.log')]
        for r in rels:
            if not os.path.exists(r):
                raise SystemExit(f'post-run: {r} missing')
        if os.path.exists(OUT_POST):
            raise SystemExit(f'post-run: {OUT_POST} exists (write-once)')
        man = {r: _sha(r) for r in rels}
        with open(OUT_POST, 'x', encoding='utf-8') as h:
            GRIO.dump(man, h, indent=1, sort_keys=True)
            h.write('\n')
        print('post-run manifest:', json.dumps(man, indent=1))
        return

    started = datetime.now(timezone.utc).isoformat()
    if not os.path.isdir(OUT_DIR):
        raise SystemExit(f'REFUSED (check: output dir exists): {OUT_DIR} does not exist (mkdir it; it must hold only launch.log)')
    extra = sorted(set(os.listdir(OUT_DIR)) - {'launch.log'})
    if extra:
        raise SystemExit(f'REFUSED (check: output dir fresh): {OUT_DIR} holds {extra}')
    oc = other_copies()
    if oc:
        raise SystemExit(f'REFUSED (check: no concurrent copy): {oc}')

    # ---------------------------------------------------------------- capture-path checklist (fail fast, before work)
    required = [FROZEN, GEN_APPE, T2_CSV, W142_V6R, W142_CERT_D, W142_FLOOR, W168, W163, W101_SUMMARY, W118_SUMMARY,
                W153_SRC] + list(SPEC.values())
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'REFUSED (check: capture paths exist): missing {missing}')

    text, ms = read_manuscript()
    if not (ms['head_ok'] and ms['blob_sha_ok']):
        raise SystemExit(f'REFUSED (check: clone HEAD / main.tex sha): {ms}')
    lines = text.split('\n')
    rows, caption, e21_span = parse_e21(lines)
    nos = [r['no'] for r in rows]
    if nos != list(range(1, 62)):
        raise SystemExit(f'E.21 rows not 1..61: {nos}')

    order, gen = gen_appE_static()
    t2 = {r['cell']: r for r in csv.DictReader(open(T2_CSV, encoding='utf-8'))}
    _reg(T2_CSV)
    fz = _read_json(FROZEN)
    if INPUTS[FROZEN]['sha256'] != FROZEN_SHA:
        raise SystemExit('frozen JSON sha mismatch')
    FZ = dict(fz['tables']['cells'])
    for k, v in fz['tables']['cells_appended_w160'].items():
        if k in FZ:
            raise SystemExit(f'cell {k} in both cells and cells_appended_w160')
        FZ[k] = v
    if len(order) != 61 or {o[0] for o in order} != set(t2) or set(t2) != set(FZ):
        raise SystemExit('gen_appE order / T2 / frozen cell sets differ')

    specs = {k: _read_json(p) for k, p in SPEC.items()}
    spec_rule = {k: rule_version_of_spec(s) for k, s in specs.items()}
    w142 = _read_json(W142_V6R)
    item1 = w142['item1_v6_on_every_record']
    certD = _read_json(W142_CERT_D)
    floor = _read_json(W142_FLOOR)
    w168 = _read_json(W168)
    w163 = _read_json(W163)
    w101 = _read_json(W101_SUMMARY)
    w118 = _read_json(W118_SUMMARY)
    _read(W153_SRC)
    w153_line = next((i + 1 for i, ln in enumerate(open(W153_SRC, encoding='utf-8'))
                      if "cause = ('gap clause' if gaps else 'lapse reset' if lapses else 'growth test' if 'swings_growing' in reasons" in ln), None)

    # W142 record index by instance identity (candidate_key, eval-dir key prefix)
    rec_index = {}
    for key, rep in floor['reports'].items():
        inst = rep['instance']
        ev16 = os.path.basename(rep['eval_dir']).split('_')[0]
        rec_index.setdefault((inst['candidate_key'], ev16), []).append(key)
    if set(floor['reports']) != set(item1):
        raise SystemExit('W142 floor replay reports and item 1 records differ')

    # W163 N49: reference continuations, mapped to eval keys through the W101 summary
    n49 = next(c for c in w163['checks_in_v4'] if c.get('id') == 'N49')
    n25 = next(c for c in w163['checks_in_v4'] if c.get('id') == 'N25')
    refcont_eval = {w101['reports'][k]['eval_key']: k for k in n49['value']}

    # v6-run stage cells -> campaign roots
    v6run_cells = {}
    for sk in V6_RUN_SPECS:
        for c, cs in specs[sk]['cells'].items():
            v6run_cells.setdefault(c, []).append((sk, cs['campaign_root']))

    out_rows = []
    for r, (cell, g_plan, g_year, g_cond) in zip(rows, order):
        f = FZ[cell]
        t = t2[cell]
        ev = {}
        # ---- identity of the printed row
        ident = {
            'cell_by_generator_order': cell,
            'printed_plan_equals_generator_text': r['plan'] == g_plan,
            'printed_year_equals_generator_text': r['year'] == g_year,
            'printed_plan_equals_frozen_candidate': decode_plan(r['plan']) == cand_plan(f),
            'printed_year_equals_frozen_candidate': (r['year'] == '---') if not cand_plan(f) else
            (r['year'] == str(f['candidate_canonical']['investment_year'])),
            'printed_kstar_end_equals_T2': r['kstar_end'] == t['k* / end'],
            'frozen_status_equals_T2': f['status'] == t['status'],
            'frozen_eval_key': f['eval_key'], 'frozen_candidate_key': f['candidate_key'],
        }
        ident['ok'] = all(v for k, v in ident.items() if isinstance(v, bool))
        cs = f.get('certifying_spec')
        superseded = bool(f.get('superseded'))

        # ---- matching W142 item-1 records by identity
        keys = rec_index.get((f['candidate_key'], f['eval_key'][:16]), [])
        if len(keys) > 1:
            raise SystemExit(f'{cell}: more than one W142 record matches {keys}')
        w142_key = keys[0] if keys else None
        w142_rec = item1.get(w142_key) if w142_key else None

        # ---- (1) issuing rule and the issuing run
        issuing = {}
        if cs and cs.get('mode') == 'v6 from records':
            src_spec = 'v5' if cs.get('source_run_stage_spec') == 'frozen_s53_resettle_spec_v5_ab32ffc9' else None
            if src_spec is None:
                raise SystemExit(f'{cell}: unknown source run spec {cs}')
            issuing = {'kind': 'v6 from records certificate', 'issued_rule_version': 6,
                       'certificate_record': cs['certificate_record'],
                       'certificate_k_star': certD['k_star'], 'certificate_label': certD['label'],
                       'source_run_spec': SPEC[src_spec], 'source_run_rule_version': spec_rule[src_spec][0]}
            root = specs[src_spec]['cells'][cell]['campaign_root']
        else:
            sk = None
            if cs is None:
                if cell in v6run_cells and len(v6run_cells[cell]) == 1:
                    sk = v6run_cells[cell][0][0]
                elif w142_rec is not None:
                    path = w142_rec['committed_run']['spec']['path']
                    sk = next(k for k, p in SPEC.items() if p == path)
            else:
                path = (cs.get('stage_spec') or {}).get('path')
                if path is None and cs.get('series') == 'frozen_s53_resettle_spec' and cs.get('version') == 6:
                    path = SPEC['v6']
                sk = next((k for k, p in SPEC.items() if p == path), None)
            if sk is None:
                raise SystemExit(f'{cell}: issuing spec not resolved ({cs})')
            issuing = {'kind': 'stage spec', 'spec_key': sk, 'spec_path': SPEC[sk],
                       'stop_rule_class': spec_rule[sk][1], 'issued_rule_version': spec_rule[sk][0],
                       'frozen_certifying_spec': cs}
            if cell in (specs[sk].get('cells') or {}):
                root = specs[sk]['cells'][cell]['campaign_root']
                issuing['campaign_root_source'] = f'{SPEC[sk]} cells.{cell}.campaign_root'
            elif w142_rec is not None and w142_rec['committed_run'].get('campaign_spec'):
                root = os.path.dirname(w142_rec['committed_run']['campaign_spec'])
                issuing['campaign_root_source'] = f'{W142_V6R} item1_v6_on_every_record.{w142_key}.committed_run.campaign_spec'
            else:
                root = None
        crp = os.path.join(root, 'campaign_results.json') if root else None
        if crp is None or not os.path.exists(crp):
            raise SystemExit(f'{cell}: issuing campaign_results not found ({crp})')
        cr = _read_json(crp)
        issuing['campaign_results'] = crp
        issuing['campaign_results_eval_key_equals_frozen'] = cr.get('eval_key') == f['eval_key']
        issuing['campaign_results_stage_spec'] = (cr.get('stage_spec') or {}).get('path')
        exp_spec_path = issuing.get('spec_path') or issuing.get('source_run_spec')
        issuing['campaign_results_stage_spec_equals_issuing_spec'] = issuing['campaign_results_stage_spec'] == exp_spec_path
        issuing['G15'] = g15(cr)
        ev['issuing_run'] = crp

        # ---- (2) v6 decision on the record
        v6 = None
        if issuing.get('issued_rule_version') == 6 and issuing['kind'] == 'stage spec' and issuing['spec_key'] in V6_RUN_SPECS:
            crr = cr.get('cell_report') or {}
            if not issuing['campaign_results_eval_key_equals_frozen']:
                raise SystemExit(f'{cell}: v6 run eval key differs from frozen')
            v6 = {'kind': 'v6 run', 'status': issuing['G15']['decision_status'], 'k_star': issuing['G15']['k_star'],
                  'k_cap': issuing['G15']['k_cap'], 'branch': crr.get('branch'),
                  'range_over_tau': crr.get('range_over_tau'), 'band_width': crr.get('band_width'),
                  'n_vetoes': None, 'gap_refusals_n': len(crr.get('gap_refusals') or []),
                  'lapse_events_n': len(crr.get('lapse_events') or []), 'reasons': crr.get('reasons'),
                  'cause_w153': (w153_cause(crr.get('gap_refusals'), crr.get('lapse_events'), crr.get('reasons'))
                                 if issuing['G15']['decision_status'] == 'uncertified' else None),
                  'evidence': f'{crp} gate_detail.G15, cell_report'}
        elif issuing['kind'] == 'v6 from records certificate':
            v6 = {'kind': 'v6 from records (W142 item 2)', 'status': 'certified', 'k_star': certD['k_star'],
                  'k_cap': None, 'branch': certD['branch'], 'range_over_tau': certD['range_over_tau'],
                  'band_width': certD['band_width'], 'n_vetoes': len(certD.get('vetoes') or []),
                  'evidence': f'{W142_CERT_D} k_star, branch, range_over_tau, band_width',
                  'w142_item1_record': w142_key,
                  'w142_item1_agrees': (w142_rec is not None and w142_rec['v6']['status'] == 'certified'
                                        and w142_rec['v6']['k_star'] == certD['k_star']),
                  'Q_k_star_equals_frozen_Q': certD['Q_k_star'] == f.get('Q')}
        elif w142_rec is not None:
            d6 = w142_rec['v6']
            ab = floor['reports'][w142_key]['arms']['AB']
            v6 = {'kind': 'v6 on record (W142 item 1)', 'w142_record': w142_key, 'status': d6['status'],
                  'k_star': d6['k_star'], 'k_cap': d6['k_cap'], 'branch': d6['branch'],
                  'range_over_tau': d6['range_over_tau'], 'band_width': d6['band_width'], 'n_vetoes': d6['n_vetoes'],
                  'v6_equals_replay_arm_AB': w142_rec['v6_equals_replay_arm_AB'],
                  'floor_replay_AB': {'status': ab.get('status'), 'reasons': ab.get('reasons'),
                                      'gap_refusals_n': len(ab.get('gap_refusals') or []),
                                      'lapse_events_n': len(ab.get('lapse_events') or []),
                                      'vetoes': ab.get('vetoes')},
                  'cause_w153': (w153_cause(ab.get('gap_refusals'), ab.get('lapse_events'), ab.get('reasons'))
                                 if d6['status'] == 'uncertified' else None),
                  'identity': {'candidate_key_equal': floor['reports'][w142_key]['instance']['candidate_key'] == f['candidate_key'],
                               'eval_dir': floor['reports'][w142_key]['eval_dir'],
                               'Q_k_star_equals_frozen_Q': ((d6['Q_k_star'] == f['Q']) if (d6.get('Q_k_star') is not None
                                                            and f.get('Q') is not None) else None),
                               'frozen_cell_has_Q': f.get('Q') is not None,
                               'Q_k_star_equals_issuing_summary_Q': issuing_summary_q(w142_key, d6.get('Q_k_star'), w118, w101),
                               'band_width_equals_frozen': d6['band_width'] == f.get('band_width')},
                  'committed_run': w142_rec['committed_run'], 'certifying_spec_w142': w142_rec['certifying_spec'],
                  'evidence': f'{W142_V6R} item1_v6_on_every_record.{w142_key}.v6; {W142_FLOOR} reports.{w142_key}.arms.AB'}
            if issuing['kind'] == 'stage spec' and issuing['spec_key'] not in V6_RUN_SPECS:
                cr_rule = w142_rec['committed_run']['criterion']
                issuing['w142_committed_run_criterion'] = cr_rule
                issuing['w142_committed_run_criterion_version_agrees'] = cr_rule.startswith(f"v{issuing['issued_rule_version']}")
            if w142_key == 'x0':
                v6['w168'] = {'status': w168['decision_primary']['status'], 'k_star': w168['decision_primary']['k_star'],
                              'equals_w142': w168['w142_crosscheck']['equal'],
                              'evidence': f'{W168} decision_primary, w142_crosscheck'}
        ev['v6_decision'] = v6['evidence'] if v6 else None

        # ---- reference continuation (W163 N49)
        refcont = refcont_eval.get(f['eval_key'])
        refc = {'is_reference_continuation_W163_N49': refcont is not None, 'w101_cell': refcont,
                'replay': n49['value'].get(refcont) if refcont else None,
                'continues_earlier_run_per_E21_note': r['no'] in (1, 2, 40, 41)}

        # ---- (3) expected printed entries and consistency
        A = assess_row(r, v6, superseded, issuing['issued_rule_version'], issuing['G15']['k_star'])
        decided, exp_status, exp_rule, exp_k = A['decided'], A['exp_status'], A['exp_rule'], A['exp_k']
        status_ok, branch_ok, k_ok, rule_ok = A['status_ok'], A['branch_ok'], A['k_ok'], A['rule_ok']
        rng_ok, band_ok, marks_ok = A['rng_ok'], A['band_ok'], A['marks_ok']
        v6_cert_at_printed, flagged = A['v6_cert_at_printed'], A['flagged']
        issuing_ok = bool(issuing['campaign_results_eval_key_equals_frozen']
                          and issuing['campaign_results_stage_spec_equals_issuing_spec'])
        consistent = bool(ident['ok'] and issuing_ok and A['entries_ok'])

        if v6 is None:
            klass = 'NO v6 DECISION FOUND'
        elif v6['kind'] == 'v6 run':
            klass = f"v6 run, {v6['status']}"
        elif v6['kind'].startswith('v6 from records'):
            klass = 'v6-from-records certificate (issued under v6)'
        elif not decided:
            klass = f"v{issuing['issued_rule_version']} certificate, NOT decided by v6 on its record"
        else:
            klass = (f"v{issuing['issued_rule_version']} {'certificate' if v6['status'] == 'certified' else 'uncertified'}"
                     f", v6 on record {v6['status']} at {exp_k}")
        if refc['is_reference_continuation_W163_N49']:
            klass += ' [reference continuation]'

        out_rows.append({
            'no': r['no'], 'line': r['line'], 'cell': cell, 'printed': r, 'T2': {k: t[k] for k in (
                'status', 'branch', 'k* / end', 'cause (uncertified)', 'certifying spec', 'eval key')},
            'frozen': {'status': f['status'], 'k_star': f.get('k_star'), 'end_cycle': f.get('end_cycle'),
                       'certifying_spec': cs, 'superseded': superseded, 'eval_key': f['eval_key']},
            'identity': ident, 'issuing': issuing, 'v6_decision': v6, 'reference_continuation': refc,
            'expected': {'status': exp_status, 'rule': exp_rule, 'kstar_end': exp_k},
            'checks': {'issuing_run_identity_ok': issuing_ok, 'status_ok': status_ok, 'branch_ok': branch_ok, 'kstar_ok': k_ok, 'rule_ok': rule_ok,
                       'superseded_marker_ok': marks_ok, 'range_at_print_ok': rng_ok, 'band_at_print_ok': band_ok,
                       'consistent': consistent},
            'v6_decided': decided, 'v6_certificate_at_printed_kstar': v6_cert_at_printed,
            'earlier_rule_certificate_without_v6_decision': flagged,
            'class': klass, 'evidence_paths': ev,
        })

    # -------------------------------------------------------------------------------------------- summary, prediction
    selftest = mutation_self_test(out_rows)
    flagged = [x['no'] for x in out_rows if x['earlier_rule_certificate_without_v6_decision']]
    further = [n for n in flagged if n != EXPECTED_ROW]
    inconsistent = [x['no'] for x in out_rows if not x['checks']['consistent']]
    by_rule = {}
    for x in out_rows:
        by_rule.setdefault(x['printed']['rule'], []).append(x['no'])
    by_class = {}
    for x in out_rows:
        by_class.setdefault(x['class'], []).append(x['no'])
    by_issued = {}
    for x in out_rows:
        by_issued.setdefault(f"v{x['issuing']['issued_rule_version']}", []).append(x['no'])
    r16 = next(x for x in out_rows if x['no'] == EXPECTED_ROW)
    row16 = {'cell': r16['cell'], 'cell_is_expected': r16['cell'] == EXPECTED_ROW_CELL,
             'status_printed': r16['printed']['status_printed'], 'prints_superseded_s': r16['printed']['status'] == 'superseded' and 's' in r16['printed']['marks'],
             'rule_printed': r16['printed']['rule'], 'rule_is_dashes': r16['printed']['rule'] == '---',
             'kstar_printed': r16['printed']['kstar_end'], 'kstar_is_167': r16['printed']['kstar_end'] == '167',
             'issuing_certificate': r16['issuing']['G15'], 'issued_rule_version': r16['issuing']['issued_rule_version'],
             'v6_on_record': {k: r16['v6_decision'][k] for k in ('status', 'n_vetoes', 'floor_replay_AB')},
             'caption_s_legend': ('$^{s}$: superseded by evaluation~15; certified under an earlier version of the rule, whose '
                                  'cycle, range and band are shown, and not decided under the rule of '
                                  '\\ref{app:admm_updated_certification} within its recorded cycles') in caption}
    row16['ok'] = bool(row16['cell_is_expected'] and row16['prints_superseded_s'] and row16['rule_is_dashes']
                       and row16['kstar_is_167'] and row16['issuing_certificate']['k_star'] == 167
                       and row16['v6_on_record']['status'] == 'not decided within the recorded cycles 1..167'
                       and row16['caption_s_legend'])
    uncertified_rows = [x['no'] for x in out_rows if not x['printed']['status'].startswith('cert.') and x['printed']['status'] != 'superseded']
    run_rows = by_rule.get('run', [])
    record_rows = by_rule.get('record', [])
    handling = {
        'uncertified_rows': {
            'rows': uncertified_rows,
            'how': ('an uncertified row prints no certificate, so it cannot count toward the prediction; it is still '
                    'classified: the issuing rule of its uncertified decision, and whether v6 decided it uncertified at '
                    'the printed end cycle (v6 run: G15 decision_status/k_cap; record: W142 item 1 status/k_cap), with '
                    'the printed cause compared to the W153 cause formula applied to the v6 decision fields'),
            'all_v6_uncertified_at_printed_end': all(x['v6_decision'] and x['v6_decision']['status'] == 'uncertified'
                                                     and str(x['v6_decision']['k_cap']) == x['printed']['kstar_end']
                                                     for x in out_rows if x['no'] in uncertified_rows)},
        'run_rows': {
            'rows': run_rows,
            'how': ('Rule "run": the issuing run is itself a v6-rule run (stage spec v6 96c23404, ext v3 84775dc4 or '
                    'A64 v1 44a2dce8, stop_rule.class settling_criterion_v6.SettlingRuleV6); its campaign_results '
                    'eval_key must equal the frozen cell\'s and its G15 decision is the v6 decision; no record-based '
                    're-decision is needed or looked for'),
            'all_issued_v6_by_a_v6_run': all(x['v6_decision'] and x['v6_decision']['kind'] == 'v6 run'
                                             for x in out_rows if x['no'] in run_rows)},
        'record_rows': {'rows': record_rows,
                        'all_have_v6_certificate_at_printed_kstar_or_v6_uncertified_at_printed_end':
                            all((x['v6_certificate_at_printed_kstar'] or (x['v6_decision']['status'] == 'uncertified'
                                 and str(x['v6_decision']['k_cap']) == x['printed']['kstar_end'])) for x in out_rows
                                if x['no'] in record_rows)},
    }
    prediction = {
        'text': ('Addendum 79 (b), expert: 0 further rows -- besides row 16 -- print a certificate issued under an earlier '
                 'rule version without a v6 decision on its record (every "record" row has a v6-from-records '
                 'certificate or is a reference continuation)'),
        'rows_flagged': flagged, 'further_rows_besides_16': further, 'held': (further == [] and flagged == [EXPECTED_ROW]),
    }
    search_scope = {
        'rows_examined': 'all 61 rows of Table E.21 in main.tex at Overleaf e5322f8 (git show blob), lines '
                         f'{e21_span[0]}-{e21_span[1]}',
        'v6_decisions_searched': [
            f'{W142_V6R} item1_v6_on_every_record (all {len(item1)} records), matched by floor-replay instance identity',
            f'{W142_CERT_D} (W142 item 2)', f'{W168} decision_primary (x = 0)',
            'campaign_results.json G15 of every cell of the v6 stage spec (34), ext v3 (7), A64 v1 (4) through the '
            'specs\' cells.<c>.campaign_root'],
        'issuing_runs_read': 'campaign_results.json of the issuing run of every row (61 files; paths per row)',
        'reference_continuations': f'{W163} checks_in_v4 N49, mapped through {W101_SUMMARY} reports.<k>.eval_key',
        'not_searched': ['W118 v1 (r1) campaigns; v3 (W132) and other v4/v5 cells not printed in E.21; the 3 x 3 '
                         'evaluations (not in E.21); pickled models (blocked)'],
    }
    definitions = {
        'issued_rule_version': 'version of stop_rule.class of the issuing stage spec (settling_criterion.SettlingRule -> 1, '
                               'settling_criterion_vN.* -> N); d_c52e1670: 6 (W142 item 2 certificate)',
        'w153_cause_formula': f"{W153_SRC} l. {w153_line}: 'gap clause' if gap_refusals else 'lapse reset' if "
                              "lapse_events else 'growth test' if 'swings_growing' in reasons else 'other'",
        'earlier_rule_certificate_without_v6_decision': 'issued_rule_version < 6 AND printed Status begins "cert." or is '
                                                        '"superseded" AND NOT (v6 decision certified at the printed k*)',
        'expected_rule': "'run' if v6 decision kind == 'v6 run'; 'record' if a decided from-records kind; '---' if not decided",
        'at_print': '|printed - value/scale| <= half a unit of the printed last digit (+1e-9); band scale 1000 (k EUR)',
    }

    result = {
        'schema': 'p515_s53_w180b_row16_class_v1',
        'stage': 'P5.15 Addendum 79 W180b -- E.21 rows: rule version of the printed decision, v6 decision on record, '
                 'consistency of Status/Rule/k*, at Overleaf e5322f8',
        'started_utc': started, 'git_HEAD': _git(['rev-parse', 'HEAD']).decode().strip(), 'interpreter': sys.executable,
        'objective_convention': 'no objective value is reported; Q compared bitwise only as an identity check '
                                '(gross_operational_cost, settlement excluded)',
        'manuscript': ms, 'E21_span_lines': e21_span, 'generator': gen,
        'spec_rule_versions': {k: {'path': SPEC[k], 'class': v[1], 'version': v[0]} for k, v in spec_rule.items()},
        'definitions': definitions, 'search_scope': search_scope,
        'W163_N49': n49['value'], 'W163_N25_last_cycles': n25['value'].get('last_cycles'),
        'rows': out_rows,
        'summary': {'n_rows': len(out_rows), 'by_printed_rule': by_rule, 'by_issued_rule_version': by_issued,
                    'by_class': by_class, 'inconsistent_rows': inconsistent,
                    'identity_ok_all': all(x['identity']['ok'] for x in out_rows)},
        'row16': row16, 'handling': handling, 'prediction_b': prediction,
        'mutation_self_test': selftest,
    }
    bad = sorted(m for m in sys.modules if m.startswith(FORBIDDEN_MODULE_PREFIXES))
    gfail = GUARD.verify(0)
    if bad or gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads']:
        raise SystemExit(f'guard fault: production modules {bad}; guard {gfail}; pickle {PICKLE_COUNTS}')
    result['guards'] = {'solve_profile_guard': {'label': GUARD.label, 'counts': dict(GUARD.counts),
                                                'verify_0_failures': gfail},
                        'pickle_counts': dict(PICKLE_COUNTS),
                        'pickle_blocked': pickle.load is _blocked_load and pickle.loads is _blocked_loads,
                        'production_modules_imported': bad}
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(OUT_JSON, 'x', encoding='utf-8') as fh:
        GRIO.dump(result, fh, indent=1, sort_keys=False)
        fh.write('\n')
    manifest = {'outputs': {OUT_JSON: _sha(OUT_JSON)}, 'script': {SCRIPT_REL: _sha(SCRIPT_REL)},
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)}, 'n_inputs': len(INPUTS),
                'n_uncommitted_inputs': sum(1 for v in INPUTS.values() if v['status'] == 'UNCOMMITTED')}
    with open(OUT_MANIFEST, 'x', encoding='utf-8') as fh:
        GRIO.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')

    # ---------------------------------------------------------------------------------------------- console summary
    print(f"clone HEAD {ms['clone_head']} ok {ms['head_ok']}; main.tex blob sha ok {ms['blob_sha_ok']}; working copy "
          f"= blob {ms['working_copy_equals_blob']}; {ms['n_lines']} lines; E.21 lines {e21_span}")
    print(f"identity ok on all 61 rows: {result['summary']['identity_ok_all']}")
    print('per row: no | cell | printed status / k* / rule | issued v | v6 decision | expected | consistent | flagged')
    for x in out_rows:
        v6 = x['v6_decision']
        v6s = f"{v6['kind']}: {v6['status']} k*={v6.get('k_star')} cap={v6.get('k_cap')}" if v6 else 'NONE'
        print(f"  {x['no']:>2} {x['cell']:<16} {x['printed']['status_printed']:<22} {x['printed']['kstar_end']:>4} "
              f"{x['printed']['rule']:<6} v{x['issuing']['issued_rule_version']} | {v6s} | exp {x['expected']} | "
              f"{x['checks']['consistent']} | {x['earlier_rule_certificate_without_v6_decision']}")
    print('by printed rule:', {k: len(v) for k, v in by_rule.items()})
    print('by issued rule version:', {k: len(v) for k, v in by_issued.items()})
    print('by class:')
    for k, v in by_class.items():
        print(f'  {k}: {len(v)} {v}')
    print(f'inconsistent rows: {inconsistent}')
    print(f"row 16: {row16['ok']} ({row16['status_printed']}, rule {row16['rule_printed']}, k* {row16['kstar_printed']}, "
          f"v6 {row16['v6_on_record']['status']}, vetoes {row16['v6_on_record']['n_vetoes']}; caption legend {row16['caption_s_legend']})")
    print(f"handling: uncertified {handling['uncertified_rows']['rows']} all v6-uncertified at printed end "
          f"{handling['uncertified_rows']['all_v6_uncertified_at_printed_end']}; run rows {len(run_rows)} all v6 runs "
          f"{handling['run_rows']['all_issued_v6_by_a_v6_run']}; record rows {record_rows}")
    print(f"PREDICTION (b): flagged {flagged}; further besides 16 {further}; held {prediction['held']}")
    print(f"inputs {len(INPUTS)} ({manifest['n_uncommitted_inputs']} uncommitted)")
    print(f"guards: {result['guards']}")
    print('mutation self-test:')
    for k, v in selftest['cases'].items():
        print(f'  {v}  {k}')
    print(f'output {OUT_JSON} sha256 {_sha(OUT_JSON)}')
    if not selftest['all_pass']:
        raise SystemExit('mutation self-test FAILED (outputs written; see JSON mutation_self_test)')


if __name__ == '__main__':
    main()
