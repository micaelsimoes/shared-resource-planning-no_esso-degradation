"""P5.15 Addendum 80, Planner task W181a -- ITEMS (a) AND (b) OF THE W181 ORDER: IDENTITY OF OVERLEAF cd57efc WITH THE
ROUND-7 DELIVERABLES, AND APPENDIX E REGENERATED AT cd57efc. A SHORT CHECK, NOT A RE-AUDIT (Addendum 80 ruling 5).
ZERO SOLVES, NO MODEL LOADS, READ ONLY (the manuscript clone, manuscript_review/, the frozen JSON and the specs are only
read; the clone is never fetched, pulled, checked out or modified).

WHAT IT DOES.
  (a) Identity. The clone's HEAD (expected cd57efc, one commit after e5322f8), the sha256 and line counts of main.tex and
      response_to_reviewers_draft.tex at HEAD and at e5322f8; byte-identity of main.tex with
      manuscript_review/main_e5322f8_round7.tex and of the letter with manuscript_review/response_to_reviewers_v8.tex (and,
      at e5322f8, with main_48099ab_round6.tex / response_to_reviewers_v7.tex); the clone diff e5322f8 -> HEAD
      (git diff, read-only) parsed hunk by hunk and compared, in their +/- lines and in order, with the expert's
      manuscript_review/main_e5322f8_round7.diff (6 hunks) and manuscript_review/letter_v7_v8.diff (2 hunks); hunk
      ranges and context lines are compared and reported beside.
  (b) Appendix E. Runs the committed (patched, fc2fad77) generator manuscript_review/gen_appE.py IN-PROCESS with an
      explicit output path INSIDE this stage's output directory (never its cwd default appE.tex, never onto
      manuscript_review/appendixE_evaluation_record.tex), and the committed verifier manuscript_review/verify_appE.py
      in-process (it regenerates into its own tempfile.mkdtemp() directory); the verifier output is captured and its
      PASS / FAIL lines counted. The generator output must occur exactly once in the clone's main.tex, line-aligned (the
      line range is located by content), and equal the committed appendixE_evaluation_record.tex and the verifier's own
      regeneration. Then the W180a entry check (sections 0-5 and 7-11 below are copied from
      p515_s53_w180a_appendixE_check.py, which is not imported, edited or run; the row-16 convention of Addendum 79
      ruling 1 is kept) of Tables E.20-E.24 at HEAD against the frozen JSON frozen_step6_tables_v1_590088fe.json, the same
      check at e5322f8 as a cross-check, and the entry-by-entry diff of the parsed tables e5322f8 -> HEAD (cells
      separately from captions, notes, group headers and other lines) and the line diff of the whole Appendix E block,
      every hunk located and classified (cell / caption / note / group header / prose / comment).
  (c) Mutation test of both comparisons, in memory only: single-entry edits to a copy of the parsed HEAD tables, each of
      which must be caught at the edited table / row / column (frozen check) and as exactly one changed entry (diff);
      plus caption / note edits for the diff.
The prose ratings of W180a (its section 12) are NOT run here: the five round-7 sentences are W181b's item (c).

OBJECTIVE CONVENTION. Q = gross_operational_cost, settlement excluded (the frozen primary); net = Q - terminal salvage
credit; value = Q(0) - Q(x); F = Q + I. EUR in the JSON, k EUR in the manuscript.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import and verified at exactly 0 at the end;
pickle.load / pickle.loads blocked for the whole run (counters verified 0). git reads only (rev-parse, status, ls-files,
log, show, diff), also in the manuscript clone. Refuses to run beside another copy of itself. Every JSON output goes
through the shared writer gate_result_io (refuses string flags).

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir data/SRP1/Results/P515S53/w181a_identity_appendixE && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w181a_identity_appendixE.py \\
        > data/SRP1/Results/P515S53/w181a_identity_appendixE/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w181a_identity_appendixE/w181a_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w181a_identity_appendixE/w181a_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w181a_identity_appendixE.py --post-run
Write-once: every output is opened 'x' in a directory that holds nothing but its launch log.
Exit: 0 = written, every integrity check holds and the guards are at 0 (MISMATCHES and failed predictions are FINDINGS
and never change the exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard fault
(nothing written).
"""
import argparse
import contextlib
import difflib
import gc
import hashlib
import io
import json
import os
import pickle
import re
import runpy
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W181a identity and Appendix E at Overleaf cd57efc '
                                              '(never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W181a: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W181a: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer that refuses string flags)

pickle.load, pickle.loads = _blocked_load, _blocked_loads

SCRIPT_REL = os.path.basename(__file__)
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(S53, 'w181a_identity_appendixE')
OUT_NAMES = {'json': 'w181a_identity_appendixE.json', 'md': 'w181a_identity_appendixE.md', 'man': 'manifest_sha256.json',
             'log': 'launch.log', 'gen': 'appE_generated_w181a.tex', 'verify': 'verify_appE_output.txt',
             'typing_json': 'w181a_bool_typing_test.json', 'typing_log': 'w181a_bool_typing_test.log',
             'post': 'manifest_post_run_sha256.json'}
LAUNCH_LOG_RX = re.compile(r'launch(_attempt\d+)?\.log')

CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
CLONE_HEAD = 'cd57efcb54399cb652a0ac2dcd64e5d3a78d89fc'
MAIN_SHA = 'ff7b045edd17136fce7a37d9ef7ec16c8515ab4f8df69f3c33cc021127bbed35'
MAIN_LINES = 2713
LETTER_REL_CLONE = 'response_to_reviewers_draft.tex'
LETTER_SHA = '06cc24604c54955964c6000edf48de2aba34408d6a6e499e55122ac705d5cb2c'
LETTER_LINES = 292
PREV_REV = 'e5322f8'
PREV_REV_FULL = 'e5322f85dced8034312c9afb97c19001bce6bfa0'
PREV_SHA = 'b3282295ece4cc3b0f9bc4ff5cd00ad2d86a7e60650221bfd2d0bd00fa61c492'   # W180a's audited main.tex
PREV_LINES = 2710
PREV_LETTER_SHA = '34a5754ba6ccb53e29d64fad38fddc921c3048479fc5c31eef39664d9a0efa38'  # = response_to_reviewers_v7.tex
EXPECTED_DIFF_FILES = ['main.tex', LETTER_REL_CLONE]
EXPECTED_HUNKS = {'main.tex': 6, LETTER_REL_CLONE: 2}
FZ_REL = os.path.join(S53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FZ_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
GEN_REL = os.path.join('manuscript_review', 'gen_appE.py')
VERIFY_REL = os.path.join('manuscript_review', 'verify_appE.py')
APPX_REL = os.path.join('manuscript_review', 'appendixE_evaluation_record.tex')
EXPORT_DIR = os.path.join('manuscript_review', 'export_W160')
EXPORT_FILES = ('MANIFEST.md', 'README.md', 'README_v2_note.md', 'T1.csv', 'T1.tex', 'T10.csv', 'T10.tex', 'T2.csv',
                'T2.tex', 'T5.csv', 'T5.tex', 'T6.csv', 'T6.tex', 'T9.csv', 'T9.tex', 'paragraphs_v2.md')

INPUTS = {
    'FROZEN_JSON': FZ_REL,
    'REVIEW_COPY': os.path.join('manuscript_review', 'main_e5322f8_round7.tex'),
    'REVIEW_COPY_PREV': os.path.join('manuscript_review', 'main_48099ab_round6.tex'),
    'LETTER_V8': os.path.join('manuscript_review', 'response_to_reviewers_v8.tex'),
    'LETTER_V7': os.path.join('manuscript_review', 'response_to_reviewers_v7.tex'),
    'MAIN_DIFF': os.path.join('manuscript_review', 'main_e5322f8_round7.diff'),
    'LETTER_DIFF': os.path.join('manuscript_review', 'letter_v7_v8.diff'),
    'APPX_COMMITTED': APPX_REL,
    'GEN': GEN_REL,
    'VERIFY': VERIFY_REL,
    'STEP6_ROUND7': 'STEP6_ROUND7_CORRECTIONS.md',
    'W180A_SCRIPT': 'p515_s53_w180a_appendixE_check.py',
    'W180A_GENERATED': os.path.join(S53, 'w180a_appendixE', 'appE_generated_w180a.tex'),
    'SPEC_V4': os.path.join(S53, 'w137_resettle_v4', 'frozen_s53_resettle_spec_v4_e11fbc89.json'),
    'SPEC_V5': os.path.join(S53, 'w139_resettle_v5', 'frozen_s53_resettle_spec_v5_ab32ffc9.json'),
    'SPEC_V6': os.path.join(S53, 'w142_resettle_v6', 'frozen_s53_resettle_spec_v6_96c23404.json'),
    'SPEC_EXT_V3': os.path.join(S53, 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json'),
    'SPEC_W118_V2': os.path.join(S53, 'w118_resettle', 'frozen_s53_resettle_spec_v2_fc791891.json'),
    'SPEC_A64_V1': os.path.join(S53, 'w155_a64_cells', 'frozen_s53_a64_cells_spec_v1_44a2dce8.json'),
    'SPEC_W101_X0': os.path.join(S53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0',
                                 'campaign_spec_s53_w101_srp1_cont_x0_d1d5c3bc.json'),
    'SPEC_W101_UNIT': os.path.join(S53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_n7_4h_e1',
                                   'campaign_spec_s53_w101_srp1_cont_n7_4h_e1_434f46fc.json'),
    'W142_V6_FROM_RECORDS': os.path.join(S53, 'w142_resettle_v6', 'v6_from_records', 'w142_v6_from_records.json'),
}
for _f in EXPORT_FILES:
    INPUTS[f'EXPORT_{_f}'] = os.path.join(EXPORT_DIR, _f)

# ---- sections 0-5 and 7-11 below are COPIED VERBATIM from p515_s53_w180a_appendixE_check.py (W180a, lines 191-863
# ---- and 892-1216; that script is neither imported, edited nor run). They carry W179a's entry check with W180a's
# ---- row-16 convention (Addendum 79 ruling 1), marked 'W180a ADAPTATION' in check_e21. W180a's section 6 (needles,
# ---- W163 walk) and section 12 (prose ratings) are not used here and are not copied.
# ======================================================================================================================
#  0. small helpers
# ======================================================================================================================
def _log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(rel):
    with open(os.path.join(REPO, rel), 'rb') as h:
        return _sha_bytes(h.read())


def _git(*args, cwd=REPO):
    return subprocess.run(['git', *args], capture_output=True, text=True, cwd=cwd).stdout.strip()


def _committed_clean(rel):
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', rel], capture_output=True, text=True,
                             cwd=REPO).returncode == 0
    return tracked and _git('status', '--porcelain', '--', rel) == ''


def _json(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        return json.load(h)


def _text(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        return h.read()


def dec_places(s):
    s = s.strip()
    return len(s.split('.')[1]) if '.' in s else 0


def parse_num(s):
    """'$-$14.5' -> -14.5; '653\\,873.7' -> 653873.7; '---' -> None."""
    t = s.strip().replace('$-$', '-').replace('\\,', '').replace('{,}', '').replace(',', '').replace('$', '')
    if t in ('---', ''):
        return None
    return float(t)


def at_print(printed, value, scale=1.0):
    """MATCH if |printed - value/scale| <= half a unit of the printed last digit (+1e-9). Returns (ok, value_at_dp)."""
    if value is None:
        return printed.strip() == '---', None
    p = parse_num(printed)
    if p is None:
        return False, None
    dp = dec_places(printed.strip().replace('$-$', '-').replace('\\,', ''))
    v = value / scale
    return abs(p - v) <= 0.5 * 10 ** (-dp) + 1e-9, round(v, dp)


def strip_markers(s):
    marks = re.findall(r'\$\^\{(\\Delta|[a-z])\}\$', s)
    clean = re.sub(r'\$\^\{(\\Delta|[a-z])\}\$', '', s).strip()
    return clean, [m.replace('\\Delta', 'Delta') for m in marks]


def table_body(lines, label, end_token):
    """Line indices (0-based) from the \\label line to the first end_token after it."""
    i0 = next(i for i, ln in enumerate(lines) if f'\\label{{{label}}}' in ln)
    i1 = next(i for i in range(i0, len(lines)) if end_token in lines[i])
    return i0, i1


def row_cells(line):
    t = line.strip()
    if t.endswith('\\\\'):
        t = t[:-2]
    return [c.strip() for c in re.split(r'(?<!\\)&', t)]


# ======================================================================================================================
#  1. plan decoding
# ======================================================================================================================
def decode_plan(txt):
    """'5: 0.25/0.5; 7: 1/3.5' -> {'5': (0.25, 0.5), '7': (1.0, 3.5)}; '5, 9: 0.25/0.5 each' -> both; x = 0 -> {}."""
    t = txt.strip()
    if 'boldsymbol{x}=\\boldsymbol{0}' in t:
        return {}
    out = {}
    for part in t.split(';'):
        part = part.strip()
        m = re.fullmatch(r'([0-9, ]+):\s*([0-9.]+)/([0-9.]+)(\s+each)?', part)
        if not m:
            raise ValueError(f'plan text not decodable: {txt!r}')
        for n in m.group(1).split(','):
            out[n.strip()] = (float(m.group(2)), float(m.group(3)))
    return out


def cand_plan(c):
    return {n: (float(v[0]), float(v[1])) for n, v in c['candidate_canonical']['nodes'].items() if v[0] or v[1]}


def plan_str(p):
    return '; '.join(f'{n}: {v[0]:g}/{v[1]:g}' for n, v in sorted(p.items())) or 'x = 0'


# ======================================================================================================================
#  2. conditions (from each cell's own stage spec)
# ======================================================================================================================
PROD_AGEING = ('on', 0.985, 0.8, 'end', 0.7)
COND_LABEL = {
    'base': (1.0, PROD_AGEING, 'production'),
    '$m=1.5$': (1.5, PROD_AGEING, 'production'),
    '$m=1.75$': (1.75, PROD_AGEING, 'production'),
    '$m=2$': (2.0, PROD_AGEING, 'production'),
    'no ageing': (1.0, ('off',), 'ext_arm'),
    'cycling only': (1.0, ('on', 1.0, 0.8, 'end', 0.7), 'ext_arm'),
    'baseline': (1.0, ('on', 0.985, 0.8, 'end', 0.7), 'ext_arm'),
    'datasheet': (1.0, ('on', 1.0, 0.7, 'end', 0.7), 'ext_arm'),
    'retention 50\\,\\%': (1.0, ('on', 1.0, 0.5, 'end', 0.7), 'ext_arm'),
    'ret.\\ 50\\,\\%, mid-block': (1.0, ('on', 1.0, 0.5, 'mid', 0.7), 'ext_arm'),
    'baseline, 0.50 floor': (1.0, ('on', 0.985, 0.8, 'end', 0.5), 'a64_floor'),
    'baseline, 0.70 floor': (1.0, ('on', 0.985, 0.8, 'end', 0.7), 'a64_floor'),
}


def _prod_ageing_from(conf):
    ab = conf['ess_ageing_baseline']
    return ('on', ab['calendar_retention_per_year'], ab['calibration']['eol_retention_r'], 'end', ab['minimum_soh'])


def cell_conditions(J, cells):
    """{cell: {'m', 'ageing', 'family', 'source'}} from the stage spec each cell ran under."""
    out = {}
    v6, v4, v5 = J['SPEC_V6'], J['SPEC_V4'], J['SPEC_V5']
    ext, w118, a64 = J['SPEC_EXT_V3'], J['SPEC_W118_V2'], J['SPEC_A64_V1']
    for c in cells:
        rec = None
        if c in v6['cells']:
            sc = v6['cells'][c]
            rec = {'m': float(sc['m_flex_price_multiplier']), 'ageing': _prod_ageing_from(sc['configuration']),
                   'family': 'production', 'source': f"{INPUTS['SPEC_V6']} cells.{c}"}
        elif c == 'd_c52e1670' or c == 'b_4649234b':
            sc = v5['cells'][c]
            rec = {'m': float(sc['m_flex_price_multiplier']), 'ageing': _prod_ageing_from(sc['configuration']),
                   'family': 'production', 'source': f"{INPUTS['SPEC_V5']} cells.{c}"}
        elif c in ('b_0dd237f0', 'b_2a0ba8b2'):
            sc = v4['cells'][c]
            rec = {'m': float(sc['m_flex_price_multiplier']), 'ageing': _prod_ageing_from(sc['configuration']),
                   'family': 'production', 'source': f"{INPUTS['SPEC_V4']} cells.{c}"}
        elif c in ext['cells']:
            sc = ext['cells'][c]
            mv = sc.get('model_variant')
            conf = sc['configuration']
            if isinstance(mv, dict) and mv.get('ageing_enabled') is not None:
                ag = ('on', mv['calendar_retention_per_year'], mv['eol_retention_r'], mv['available_energy_soh_point'],
                      conf['ess_ageing_baseline']['minimum_soh']) if mv['ageing_enabled'] else ('off',)
                fam = 'ext_arm'
            else:
                ag, fam = _prod_ageing_from(conf), 'production'
            mm = sc.get('m_flex_price_multiplier')
            rec = {'m': float(mm) if mm is not None else 1.0, 'ageing': ag, 'family': fam,
                   'source': f"{INPUTS['SPEC_EXT_V3']} cells.{c}.model_variant / configuration (no multiplier field: base)"}
        elif c in a64['cells']:
            sc = a64['cells'][c]
            mm = sc['flex_price_multiplier']
            mv = sc.get('model_variant')
            conf = sc['configuration']
            if isinstance(mv, dict) and mv.get('ageing_enabled') is not None:
                ag = ('on', mv['calendar_retention_per_year'], mv['eol_retention_r'], mv['available_energy_soh_point'],
                      sc['minimum_soh'])
                fam = 'a64_floor'
            else:
                ag, fam = _prod_ageing_from(conf), 'production'
            rec = {'m': float(mm) if mm is not None else 1.0, 'ageing': ag, 'family': fam,
                   'source': f"{INPUTS['SPEC_A64_V1']} cells.{c}.flex_price_multiplier / model_variant / minimum_soh"}
        elif c.startswith('pb_') or c.startswith('yl_') or c in ('ref:5ca4f86c', 'ref:e28de4ac'):
            key = {'ref:5ca4f86c': 'f2_incumbent', 'ref:e28de4ac': 'f2_challenger'}.get(c, c)
            sc = w118['cells'][key]
            mm = sc['flex_price_multiplier']
            rec = {'m': float(mm) if mm is not None else 1.0, 'ageing': _prod_ageing_from(sc['configuration']),
                   'family': 'production', 'source': f"{INPUTS['SPEC_W118_V2']} cells.{key}.flex_price_multiplier ({mm})"}
        elif c in ('ref:7aa017f0', 'ref:bd504ecf'):
            key = 'SPEC_W101_X0' if c == 'ref:7aa017f0' else 'SPEC_W101_UNIT'
            sc = J[key]
            if 'flex' in json.dumps(sc):
                raise RuntimeError(f'{INPUTS[key]} mentions a flexibility field; read it explicitly')
            rec = {'m': 1.0, 'ageing': _prod_ageing_from(sc['configuration']), 'family': 'production',
                   'source': f"{INPUTS[key]} configuration (no flexibility-price field in the spec: base)"}
        if rec is None:
            raise RuntimeError(f'no spec found for cell {c}')
        rec['ageing'] = tuple(float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x for x in rec['ageing'])
        out[c] = rec
    return out


def cond_matches(label, rec):
    m, ag, fam = COND_LABEL[label]
    ok_m = abs(rec['m'] - m) < 1e-12
    ok_ag = tuple(rec['ageing']) == tuple(float(x) if isinstance(x, float) else x for x in ag)
    ok_fam = rec['family'] == fam
    return ok_m and ok_ag and ok_fam, {'m': ok_m, 'ageing': ok_ag, 'family': ok_fam}


# ======================================================================================================================
#  3. Table E.21 (evaluations)
# ======================================================================================================================
def parse_e21(lines):
    i0, i1 = table_body(lines, 'tab:appE_evaluations', '\\end{longtable}')
    rows, group = [], None
    for i in range(i0, i1):
        ln = lines[i]
        g = re.search(r'\\textit\{\(([a-f])\)', ln)
        if g:
            group = g.group(1)
        if re.match(r'^\d+ & ', ln.strip()):
            cells = row_cells(ln)
            if len(cells) != 14:
                raise RuntimeError(f'E.21 l. {i + 1}: {len(cells)} cells')
            rows.append({'line': i + 1, 'group': group, 'cells': cells})
    note = next(lines[i] for i in range(i0, i1) if '\\textit{Note:}' in lines[i])
    return rows, note


def status_text(c):
    if c['status'] == 'certified':
        return {'oscillatory': 'cert., osc.', 'monotone': 'cert., mon.', None: 'cert.'}[c.get('branch')]
    cause = c.get('cause_uncertified') or ''
    return 'growth test' if cause.startswith('growth test') else cause


def expected_rule(c, cond, v6spec):
    """(label, reason) from the certifying spec the frozen cell records."""
    cs = c.get('certifying_spec')
    name = c['cell']
    if name in ('ref:7aa017f0', 'ref:bd504ecf'):
        return 'record', 'reference certified under criterion v1 (W101); v6 applied to its record by W142 item 1'
    if name in ('ref:5ca4f86c', 'ref:e28de4ac'):
        return 'record', 'W118 r2 run (criterion v2), uncertified; v6 applied to its record by W142 item 1'
    if cs is None:
        if c['status'] == 'uncertified' and name in v6spec['cells']:
            return 'run', 'uncertified cell of the v6 stage (spec 96c23404): the rule ran in the run'
        return None, 'no certifying spec and not a v6-stage cell'
    if cs.get('mode') == 'v6 from records':
        return 'record', 'v6 applied to the committed v5 run (W142 v6_from_records)'
    series = cs.get('series')
    path = (cs.get('stage_spec') or {}).get('path', '')
    if series == 'frozen_s53_resettle_spec' and cs.get('version') == 6:
        return 'run', 'v6 stage spec 96c23404'
    if series == 'frozen_s53_resettle_ext_spec':
        return 'run', 'ext spec v3 84775dc4 (criterion v6)'
    if series == 'A64 v1':
        return 'run', 'A64 spec v1 44a2dce8 (criterion v6)'
    if series == 'W118 r2' or 'resettle_spec_v4' in path or 'resettle_spec_v5' in path:
        return 'record', f'certified under an earlier criterion ({series} {os.path.basename(path)}); v6 applied to its record by W142 item 1'
    return None, f'unclassified spec {cs}'


W142_KEY = {'b_0dd237f0': 'b_0dd237f0@v4', 'b_2a0ba8b2': 'b_2a0ba8b2@v4', 'b_4649234b': 'b_4649234b@v5',
            'd_c52e1670': 'd_c52e1670@v5', 'ref:5ca4f86c': 'f2_incumbent', 'ref:e28de4ac': 'f2_challenger',
            'ref:7aa017f0': 'x0', 'ref:bd504ecf': 'unit_n7_4h_e1', 'yl_y2030': 'yl_y2030', 'yl_y2035': 'yl_y2035',
            'pb_y2025_n5': 'pb_y2025_n5', 'pb_y2025_n7': 'pb_y2025_n7', 'pb_y2025_n9': 'pb_y2025_n9',
            'pb_y2030_n5': 'pb_y2030_n5', 'pb_y2030_n7': 'pb_y2030_n7', 'pb_y2030_n9': 'pb_y2030_n9'}


def check_e21(rows, ALL, COND, J):
    out, idmap, problems = [], {}, []
    for r in rows:
        no, plan_t, year_t, cond_t, status_t, k0_t, ks_t, rng_t, band_t, gap_t, slack_t, nc_t, rep_t, rule_t = r['cells']
        no = int(no)
        status_clean, marks = strip_markers(status_t)
        plan = decode_plan(plan_t)
        year = None if year_t == '---' else int(year_t)
        cands = []
        for name, c in ALL.items():
            if cand_plan(c) != plan:
                continue
            if plan and c['candidate_canonical']['investment_year'] != year:
                continue
            ok, _parts = cond_matches(cond_t, COND[name])
            if ok:
                cands.append(name)
        disamb = None
        if len(cands) > 1:
            sup = [x for x in cands if ALL[x].get('superseded') is True]
            if 's' in marks and len(sup) == 1:
                cands, disamb = sup, 'printed $^{s}$ -> the superseded cell'
            else:
                nsup = [x for x in cands if ALL[x].get('superseded') is not True]
                if 's' not in marks and len(nsup) == 1:
                    cands, disamb = nsup, 'no $^{s}$ -> the non-superseded cell'
        if len(cands) != 1:
            problems.append(f'E.21 row {no}: plan/year/condition match {cands}')
            out.append({'no': no, 'line': r['line'], 'cell': None, 'candidates': cands})
            continue
        name = cands[0]
        c = ALL[name]
        idmap[no] = name
        cert = c['status'] == 'certified'
        ent = {}

        def put(col, printed, ok, value, note=None, frozen=True):
            st = ('MATCH' if ok else 'MISMATCH') if frozen else ('NOT_FROZEN_AGREES' if ok else 'NOT_FROZEN_DISAGREES')
            ent[col] = {'printed': printed, 'status': st, 'frozen' if frozen else 'derived': value}
            if note:
                ent[col]['note'] = note
        put('plan', plan_t, True, plan_str(cand_plan(c)), 'decoded plan = candidate_canonical (mapping key)')
        put('year', year_t, True, None if not plan else c['candidate_canonical']['investment_year'], 'mapping key')
        put('condition', cond_t, True, {'m': COND[name]['m'], 'ageing': list(COND[name]['ageing']),
                                         'family': COND[name]['family']},
            f"not a frozen field; source: {COND[name]['source']} (mapping key)", frozen=False)
        exp_status = expected_status_w180a(c)  # W180a ADAPTATION (Addendum 79 ruling 1; W179a: status_text(c))
        put('status', status_clean, status_clean == exp_status, exp_status)
        exp_marks = []
        if c.get('flag_non_clean_turning_point') is True:
            exp_marks.append('f')
        if c.get('at_or_above_0.95_tau_twin_of'):
            exp_marks.append('t')
        if c.get('superseded') is True:
            exp_marks.append('s')
        put('markers', marks, sorted(marks) == sorted(exp_marks), exp_marks)
        k0 = c.get('k0_run')
        ok, v = at_print(k0_t, k0)
        put('k0', k0_t, ok, k0)
        ks = c.get('k_star') if cert else c.get('end_cycle')
        ok, v = at_print(ks_t, ks)
        put('kstar_or_end', ks_t, ok, ks)
        if cert:
            ok, v = at_print(rng_t, c.get('range_over_tau'))
            put('range_over_tau', rng_t, ok, v)
            ok, v = at_print(band_t, c.get('band_width'), 1000.0)
            put('band_k', band_t, ok, v)
            put('gap_k', gap_t, gap_t == '---', None)
            put('slack_k', slack_t, slack_t == '---', None)
        else:
            put('range_over_tau', rng_t, rng_t == '---', None)
            put('band_k', band_t, band_t == '---', None, 'band of an uncertified cell not printed')
            ok, v = at_print(gap_t, abs(c['gap']), 1000.0)
            put('gap_k', gap_t, ok, v, '|gap| (priced interface-consensus gap at the cap)')
            ok, v = at_print(slack_t, abs(c['slack']), 1000.0)
            put('slack_k', slack_t, ok, v)
        nc = c.get('non_clean_after_N')
        ok, v = at_print(nc_t, None if nc is None else len(nc))
        put('non_clean_after_N', nc_t, ok, nc)
        ok, v = at_print(rep_t, c.get('replay_bitwise_through'))
        put('replay', rep_t, ok, c.get('replay_bitwise_through'))
        lab, why = expected_rule_w180a(c, COND[name], J)  # W180a ADAPTATION (Addendum 79 ruling 1; W179a: expected_rule)
        put('rule', rule_t, rule_t == lab, lab, f'derived from certifying_spec: {why} (not a frozen field)', frozen=False)
        out.append({'no': no, 'line': r['line'], 'group': r['group'], 'cell': name, 'disambiguation': disamb,
                    'entries': ent})
    return out, idmap, problems


# ======================================================================================================================
#  4. Table E.20 (comparisons)
# ======================================================================================================================
VERD = {'det.': 'determinate', 'w.r.': 'within resolution', 'w.b.': 'within the uncertified bar'}
CALIB = {'No ageing': 'no ageing', 'Cycling only': 'cycling only', 'Baseline': 'baseline',
         'Datasheet 8\\,000 cycles': 'datasheet', 'Retention 50\\,\\%': 'retention 50\\,\\%',
         'Retention 50\\,\\%, mid-block': 'ret.\\ 50\\,\\%, mid-block', 'Baseline, 0.50 floor': 'baseline, 0.50 floor'}
CALIB_DIFF = {'No ageing $-$ ret.\\ 50\\,\\%': 'no ageing', 'Cycling only $-$ ret.\\ 50\\,\\%': 'cycling only',
              'Baseline $-$ ret.\\ 50\\,\\%': 'baseline', 'Datasheet $-$ ret.\\ 50\\,\\%': 'datasheet',
              'Ret.\\ 50\\,\\%, mid-block $-$ ret.\\ 50\\,\\%': 'ret.\\ 50\\,\\%, mid-block'}


def parse_e20(lines):
    i0, i1 = table_body(lines, 'tab:appE_claims', '\\end{longtable}')
    rows, group = [], None
    for i in range(i0, i1):
        ln = lines[i]
        g = re.search(r'\\textit\{\(([a-d])\)', ln)
        if g:
            group = g.group(1)
        if re.match(r'^\d+ & ', ln.strip()):
            cells = row_cells(ln)
            if len(cells) != 10:
                raise RuntimeError(f'E.20 l. {i + 1}: {len(cells)} cells')
            rows.append({'line': i + 1, 'group': group, 'cells': cells})
    note_i = next(i for i in range(i0, i1) if '\\textit{Note:}' in lines[i])
    return rows, lines[note_i], note_i + 1


def claim_index(T):
    idx = {}
    for c in T['claims']:
        idx.setdefault((c['ref_cell'], c['other_cell']), []).append(('T1', c))
    for c in T['a64']['rows']:
        ref = {'7aa017f0': 'ref:7aa017f0', 'bd504ecf': 'ref:bd504ecf'}.get(c['ref_cell'], c['ref_cell'])
        idx.setdefault((ref, c['other_cell']), []).append(('T10', c))
    return idx


def check_desc(group, desc, a, b, ALL, COND, idrow):
    """Check the comparison description against the two cells; returns (ok, detail)."""
    d, marks = strip_markers(desc)
    d = d.replace(' (reference unit)', '')
    ca, cb = ALL[a], ALL[b]
    det = {}
    if group == 'a':
        m = re.fullmatch(r'(.+?)(?:, (2030|2035))?', d)
        plan, year = decode_plan(m.group(1)), int(m.group(2) or 2025)
        det = {'ref_is_x0': cand_plan(ca) == {} and COND[a]['m'] == 1.0,
               'other_plan': cand_plan(cb) == plan, 'other_year': cb['candidate_canonical']['investment_year'] == year,
               'other_base_price': COND[b]['m'] == 1.0}
    elif group == 'b':
        m = re.fullmatch(r'(.+?), \$m=([0-9.]+)\$', d)
        mm = float(m.group(2))
        if '$\\to$' in m.group(1):
            p1, p2 = m.group(1).split('$\\to$')
            n = p1.split(':')[0].strip()
            plan1 = decode_plan(p1.strip())
            plan2 = {n: decode_plan(f'{n}: {p2.strip()}')[n]}
            det = {'ref_plan': cand_plan(ca) == plan1, 'other_plan': cand_plan(cb) == plan2,
                   'price_ref': COND[a]['m'] == mm, 'price_other': COND[b]['m'] == mm, 'Delta_marker': 'Delta' in marks,
                   'years_2025': ca['candidate_canonical']['investment_year'] == 2025 == cb['candidate_canonical']['investment_year']}
        else:
            plan = decode_plan(m.group(1))
            det = {'ref_is_x0': cand_plan(ca) == {}, 'other_plan': cand_plan(cb) == plan, 'price_ref': COND[a]['m'] == mm,
                   'price_other': COND[b]['m'] == mm, 'other_year_2025': cb['candidate_canonical']['investment_year'] == 2025}
    elif group == 'c':
        m = re.fullmatch(r'(.+?)(?:, (2025))?', d)
        plan, year = decode_plan(m.group(1)), int(m.group(2) or 2030)
        det = {'ref_is_returned_plan': cand_plan(ca) == {'5': (0.25, 0.5), '7': (1.0, 3.5)}
               and ca['candidate_canonical']['investment_year'] == 2030,
               'other_plan': cand_plan(cb) == plan, 'other_year': cb['candidate_canonical']['investment_year'] == year,
               'price_2': COND[a]['m'] == 2.0 and COND[b]['m'] == 2.0,
               'n_marker_iff_not_unit_neighbour': ('n' in marks) == (not is_unit_neighbour(cand_plan(ca), cand_plan(cb)))}
    elif group == 'd':
        unit = {'7': (0.25, 1.0)}
        det['both_unit_2025'] = (cand_plan(cb) == unit and cb['candidate_canonical']['investment_year'] == 2025)
        if d in CALIB:
            det['ref_is_x0_base'] = cand_plan(ca) == {} and COND[a]['m'] == 1.0
            det['other_calibration'] = cond_matches(CALIB[d], COND[b])[0]
        elif d in CALIB_DIFF:
            det['ref_is_ret50'] = cond_matches('retention 50\\,\\%', COND[a])[0] and cand_plan(ca) == unit
            det['other_calibration'] = cond_matches(CALIB_DIFF[d], COND[b])[0]
        elif d == 'Baseline: 0.50 $-$ 0.70 floor':
            det['ref_is_unit_070'] = cand_plan(ca) == unit and tuple(COND[a]['ageing']) == PROD_AGEING
            det['other_is_050_floor'] = cond_matches('baseline, 0.50 floor', COND[b])[0]
        else:
            det['decodable'] = False
    return all(det.values()), det


def is_unit_neighbour(p, q):
    """l_inf distance 1 on the lattice (power step 0.25 MVA, energy step 0.5 MWh) over (P5, E5, P7, E7, P9, E9)."""
    steps = []
    for n in ('5', '7', '9'):
        a, b = p.get(n, (0.0, 0.0)), q.get(n, (0.0, 0.0))
        steps += [abs(a[0] - b[0]) / 0.25, abs(a[1] - b[1]) / 0.5]
    return max(steps) <= 1.0 + 1e-9


def check_e20(rows, idmap, ALL, COND, T):
    idx = claim_index(T)
    out, used, problems = [], [], []
    for r in rows:
        no, desc, ev_t, dg_t, thr_t, xg_t, vg_t, dn_t, xn_t, vn_t = r['cells']
        no = int(no)
        ev_a, ev_b = [x.strip() for x in ev_t.split('/')]
        (na, ma), (nb, mb) = strip_markers(ev_a), strip_markers(ev_b)
        a, b = idmap.get(int(na)), idmap.get(int(nb))
        cands = idx.get((a, b), [])
        if r['group'] == 'a':
            cands = [x for x in cands if x[1].get('form', 'F') == 'F' and not x[1]['claim_id'].startswith('CHECK')
                     and not x[1]['claim_id'].endswith(':net')]
        if len(cands) != 1:
            problems.append(f'E.20 row {no}: pair ({a}, {b}) -> {[x[1]["claim_id"] for x in cands]}')
            out.append({'no': no, 'line': r['line'], 'pair': [a, b], 'claim': None})
            continue
        src, cl = cands[0]
        used.append((src, cl['claim_id']))
        ent = {}

        def put(col, printed, ok, value, note=None, frozen=True):
            st = ('MATCH' if ok else 'MISMATCH') if frozen else ('NOT_FROZEN_AGREES' if ok else 'NOT_FROZEN_DISAGREES')
            ent[col] = {'printed': printed, 'status': st, 'frozen' if frozen else 'derived': value}
            if note:
                ent[col]['note'] = note
        dok, ddet = check_desc(r['group'], desc, a, b, ALL, COND, no)
        put('description', desc, dok, ddet, 'not a frozen field as a whole: decoded against the two cells (plan, year from '
            'candidate_canonical; price / calibration from each cell\'s stage spec)', frozen=False)
        for lab, nm, mk in (('eval_ref_markers', a, ma), ('eval_other_markers', b, mb)):
            exp = (['u'] if ALL[nm]['status'] == 'uncertified' else []) + (['f'] if ALL[nm].get('flag_non_clean_turning_point') is True else [])
            put(lab, mk, sorted(mk) == sorted(exp), exp)
        ok, v = at_print(dg_t, cl['d_gross'], 1000.0)
        put('d_gross_k', dg_t, ok, v)
        ok, v = at_print(thr_t, cl['gross_threshold_or_bar'], 1000.0)
        put('threshold_or_bar_k', thr_t, ok, v)
        ok, v = at_print(xg_t, cl['gross_multiple'])
        put('x_gross', xg_t, ok, v)
        put('verdict_gross', vg_t, VERD.get(vg_t) == cl['gross_verdict'], cl['gross_verdict'])
        if src == 'T10':
            put('d_net_k', dn_t, dn_t == '---' and 'd_net' not in cl, None, 'T10 carries no net form (not evaluated net)')
            put('x_net', xn_t, xn_t == '---', None)
            put('verdict_net', vn_t, vn_t == '---', None)
            net_src = None
        else:
            net_src = cl
            if cl['claim_id'].startswith('G:') and cl['claim_id'].endswith(':gross'):
                net_id = cl['claim_id'][:-len(':gross')] + ':net'
                net_src = next(x for x in T['claims'] if x['claim_id'] == net_id)
                used.append(('T1', net_id))
                ent['g_net_consistency'] = {
                    'status': 'MATCH' if (net_src['d_net'] == cl['d_net'] and net_src['d_gross'] == cl['d_gross']
                                          and net_src['net_multiple'] == cl['net_multiple']) else 'MISMATCH',
                    'frozen': {'net_claim': net_id, 'd_net_equal': net_src['d_net'] == cl['d_net'],
                               'd_gross_equal': net_src['d_gross'] == cl['d_gross']}}
            ok, v = at_print(dn_t, net_src['d_net'], 1000.0)
            put('d_net_k', dn_t, ok, v)
            ok, v = at_print(xn_t, net_src['net_multiple'])
            put('x_net', xn_t, ok, v)
            put('verdict_net', vn_t, VERD.get(vn_t) == net_src['net_verdict_W153']
                and net_src['net_verdict_W153'] == net_src['net_verdict_recomputed_here'], net_src['net_verdict_W153'])
            put('net_same_threshold', thr_t, abs(net_src['net_threshold_or_bar_W153'] - cl['gross_threshold_or_bar']) < 1e-6,
                net_src['net_threshold_or_bar_W153'], 'caption: net "against the same threshold or bar"')
        out.append({'no': no, 'line': r['line'], 'group': r['group'], 'pair': [a, b], 'source': src,
                    'claim': cl['claim_id'], 'net_claim': None if net_src is None else net_src['claim_id'], 'entries': ent})
    return out, used, problems


# ======================================================================================================================
#  5. Tables E.22-E.24
# ======================================================================================================================
def tabular_rows(lines, label):
    i0, i1 = table_body(lines, label, '\\end{tabular}')
    mid = next(i for i in range(i0, i1) if lines[i].strip() == '\\midrule')
    rows = []
    for i in range(mid + 1, i1):
        t = lines[i].strip()
        if t in ('\\midrule', '\\bottomrule') or not t.endswith('\\\\'):
            continue
        rows.append({'line': i + 1, 'cells': row_cells(t)})
    return rows, (i0, i1)


def check_e22(lines, idmap, ALL, T):
    rows, (i0, i1) = tabular_rows(lines, 'tab:appE_neighbours')
    pb = {r['cell']: r for r in T['phase_b']}
    claims = {c['other_cell']: c for c in T['claims'] if c['claim_id'].startswith('C:')}
    out = []
    for r in rows:
        nodes_t, year_t, ev_t, d_t, thr_t, x_t = r['cells']
        cell = idmap.get(int(ev_t))
        c = ALL[cell]
        nodes = sorted(n.strip() for n in nodes_t.replace(' and ', ',').split(','))
        plan_ok = cand_plan(c) == {n: (0.25, 0.5) for n in nodes} and c['candidate_canonical']['investment_year'] == int(year_t)
        ent = {'nodes_year': {'printed': f'{nodes_t} {year_t}', 'status': 'MATCH' if plan_ok else 'MISMATCH',
                              'frozen': plan_str(cand_plan(c)) + f" {c['candidate_canonical']['investment_year']}"}}
        srcs = []
        if cell in pb:
            srcs.append(('phase_b', pb[cell]['M_gross'], pb[cell]['v6_threshold'], pb[cell]['v6_multiple'], pb[cell]['v6_verdict']))
        if cell in claims:
            cl = claims[cell]
            srcs.append(('claims ' + cl['claim_id'], cl['d_gross'], cl['gross_threshold_or_bar'], cl['gross_multiple'], cl['gross_verdict']))
        for sname, dv, tv, xv, vv in srcs:
            a1, v1 = at_print(d_t, dv, 1000.0)
            a2, v2 = at_print(thr_t, tv, 1000.0)
            a3, v3 = at_print(x_t, xv)
            ent[f'{sname}'] = {'printed': [d_t, thr_t, x_t], 'status': 'MATCH' if (a1 and a2 and a3 and vv == 'determinate') else 'MISMATCH',
                               'frozen': [v1, v2, v3, vv]}
        if not srcs:
            ent['source'] = {'status': 'MISMATCH', 'frozen': 'no phase_b row or C claim for this cell'}
        out.append({'line': r['line'], 'eval': int(ev_t), 'cell': cell, 'entries': ent})
    note = next(lines[i] for i in range(i0, i1 + 8) if 'superseded first certificate' in lines[i])
    m = re.search(r'gave ([0-9.]+)~k\\euro\{\} \(([0-9.]+)\$\\times\$\)', note)
    s = pb['pb_y2025_n5']
    a1, v1 = at_print(m.group(1), s['M_gross'], 1000.0)
    a2, v2 = at_print(m.group(2), s['v6_multiple'])
    note_chk = {'printed': [m.group(1), m.group(2)], 'status': 'MATCH' if a1 and a2 and idmap.get(16) == 'pb_y2025_n5' else 'MISMATCH',
                'frozen': [v1, v2], 'cell': 'pb_y2025_n5 (E.21 row 16)'}
    return out, note_chk


def check_e23(lines, T):
    rows, (i0, i1) = tabular_rows(lines, 'tab:appE_benchmark')
    B = T['benchmark']
    arms = B['w160_additions']['arms_in_full']
    START = {'cold': 'cold', 'perturbed': 'perturbed', 'warm': 'warm_from_certified'}
    out, arm = [], None
    for r in rows:
        c = r['cells']
        if c[0].startswith('Coordinated'):
            ok, v = at_print(c[2], B['coordinated_Q'], 1000.0)
            out.append({'line': r['line'], 'row': 'coordinated', 'entries': {
                'Q_k': {'printed': c[2], 'status': 'MATCH' if ok else 'MISMATCH', 'frozen': v},
                'other_columns': {'printed': c[3:], 'status': 'MATCH' if all(x == '---' for x in c[3:]) else 'MISMATCH',
                                  'frozen': 'not applicable'}}})
            continue
        if c[0]:
            arm = {'Passive NRF': 'passive', 'Price-taker NRF': 'price_taker'}[c[0]]
        st_t, marks = strip_markers(c[1])
        st = START[st_t]
        A = arms[arm]
        cn = A['consistency_nrf_by_start'][st]
        cu = A['curtailment_phase_A_by_start'][st]['by_agent']
        ent = {}
        for col, printed, val, scale in (('Q_k', c[2], A['Q_by_start'][st], 1000.0),
                                         ('nrf_n', c[3], cn['nrf_violations_n'], 1.0),
                                         ('nrf_max_excess_pu', c[4], cn['nrf_violations_max_excess_pu'], 1.0),
                                         ('consistency_pass_k', c[5], cn['pass_effect_eur'], 1000.0),
                                         ('curt_TN_GWh', c[6], cu['TSO']['mwh_day_weighted'], 1000.0),
                                         ('curt_DN_GWh', c[7], cu['DSO']['mwh_day_weighted'], 1000.0)):
            ok, v = at_print(printed, val, scale)
            ent[col] = {'printed': printed, 'status': 'MATCH' if ok else 'MISMATCH', 'frozen': v}
        exp_b = A['best_start'] == st
        ent['best_marker'] = {'printed': marks, 'status': 'MATCH' if (('b' in marks) == exp_b) else 'MISMATCH',
                              'frozen': A['best_start']}
        out.append({'line': r['line'], 'row': f'{arm}/{st}', 'entries': ent})
    # note
    note = next(lines[i] for i in range(i1, i1 + 8) if '\\textit{Note:}' in lines[i])
    nchk = {}
    m = re.search(r'spread over the starts ([0-9.]+)~k\\euro\{\} \(passive\) and ([0-9.]+)~k\\euro\{\} \(price-taker\); '
                  r'reproducibility band of the coordinated evaluation ([0-9.]+)~k\\euro', note)
    for lab, p, v in (('spread_passive', m.group(1), arms['passive']['multimodality_band']),
                      ('spread_price_taker', m.group(2), arms['price_taker']['multimodality_band']),
                      ('coordinated_band', m.group(3), B['coordinated_reproducibility_band'])):
        ok, vv = at_print(p, v, 1000.0)
        nchk[lab] = {'printed': p, 'status': 'MATCH' if ok else 'MISMATCH', 'frozen': vv}
    m = re.search(r'in 1 of 12 blocks \(6~h; 2035 Summer\) under the passive arrangement and in 8 of 12 \(25~h; ([^)]*)\)', note)
    sp = arms['passive']['sweep_unconstrained_cold']
    sq = arms['price_taker']['sweep_unconstrained_cold']
    nchk['sweep_passive'] = {'printed': '1 of 12; 6 h; 2035 Summer', 'frozen': [sp['n_blocks_tn_cannot_accept'],
                             sp['n_hours_tn_cannot_accept'], sp['failing_blocks']],
                             'status': 'MATCH' if (m and sp['n_blocks_tn_cannot_accept'] == 1 and sp['n_hours_tn_cannot_accept'] == 6
                                                   and sp['failing_blocks'] == ['TSO|-|2035|Summer']) else 'MISMATCH'}
    blocks_printed = [x.strip() for x in m.group(1).split(',')] if m else []
    blocks_frozen = [b.replace('TSO|-|', '').replace('|', ' ') for b in sq['failing_blocks']]
    nchk['sweep_price_taker'] = {'printed': ['8 of 12', '25 h', blocks_printed],
                                 'frozen': [sq['n_blocks_tn_cannot_accept'], sq['n_hours_tn_cannot_accept'], blocks_frozen],
                                 'status': 'MATCH' if (sq['n_blocks_tn_cannot_accept'] == 8 and sq['n_hours_tn_cannot_accept'] == 25
                                                       and blocks_printed == blocks_frozen) else 'MISMATCH'}
    return out, nchk


def check_e24(lines, idmap, ALL, T, e21_rows):
    rows, _ = tabular_rows(lines, 'tab:appE_gap')
    dz = {r['cell']: r for r in T['dead_zone']['cells']}
    e21_by_no = {r['no']: r for r in e21_rows}
    out = []
    for r in rows:
        ev_t, plan_t, year_t, cond_t, cause_t, gap_t, s5, s7, s9, res_t, lapse_t = r['cells']
        no = int(ev_t)
        cell = idmap.get(no)
        z = dz.get(cell)
        ent = {}
        e21 = e21_by_no[no]['entries'] if no in e21_by_no and 'entries' in e21_by_no[no] else {}
        same = (e21.get('plan', {}).get('printed') == plan_t and e21.get('year', {}).get('printed') == year_t
                and e21.get('condition', {}).get('printed') == cond_t and e21.get('status', {}).get('printed') == cause_t)
        ent['plan_year_cond_cause_as_E21'] = {'printed': [plan_t, year_t, cond_t, cause_t], 'status': 'MATCH' if same else 'MISMATCH',
                                              'frozen': f'E.21 row {no} ({cell})'}
        if z is None:
            ent['dead_zone_row'] = {'status': 'MISMATCH', 'frozen': None}
            out.append({'line': r['line'], 'eval': no, 'cell': cell, 'entries': ent})
            continue
        ok, v = at_print(gap_t, z['t_sum_at_cap'], 1000.0)
        ent['gap_k'] = {'printed': gap_t, 'status': 'MATCH' if ok else 'MISMATCH', 'frozen': v}
        sh = z.get('share_by_node')
        for n, p in (('5', s5), ('7', s7), ('9', s9)):
            ok, v = at_print(p, None if sh is None else 100.0 * sh[n])
            ent[f'share_n{n}_pct'] = {'printed': p, 'status': 'MATCH' if ok else 'MISMATCH', 'frozen': v}
        ok, v = at_print(res_t, z.get('pf_primal_last'))
        ent['pf_primal_last'] = {'printed': res_t, 'status': 'MATCH' if ok else 'MISMATCH', 'frozen': v}
        lap = [e['cycle'] for e in (z.get('lapse_events') or [])]
        lp = [] if lapse_t == '---' else [int(x) for x in lapse_t.split(',')]
        ent['lapse_resets'] = {'printed': lapse_t, 'status': 'MATCH' if lp == lap else 'MISMATCH', 'frozen': lap,
                               'note': None if 'lapse_events' in z else 'lapse_events not recorded for this row'}
        out.append({'line': r['line'], 'eval': no, 'cell': cell, 'entries': ent})
    return out




# ======================================================================================================================
#  7. W180a: the row-16 convention (Addendum 79 ruling 1) -- the only adaptation of the W179a entry check
# ======================================================================================================================
def expected_status_w180a(c):
    """Addendum 79 ruling 1: a superseded cell prints Status "superseded" (its ^s marker kept)."""
    return 'superseded' if c.get('superseded') is True else status_text(c)


def expected_rule_w180a(c, cond, J):
    """Addendum 79 ruling 1: a superseded cell prints Rule "---" -- derived here, not assumed: the rule (v6) applied to
    its committed record (W142 item 1) must NOT decide it; otherwise W179a's derivation."""
    if c.get('superseded') is True:
        k = W142_KEY.get(c['cell'])
        v6 = ((J['W142_V6_FROM_RECORDS']['item1_v6_on_every_record'].get(k) or {}).get('v6') or {}) if k else {}
        st = v6.get('status')
        if st is not None and st != 'certified':
            return '---', (f'superseded cell (frozen superseded = true); the rule applied to its record (W142 item 1 {k}) '
                           f'is {st!r} with {v6.get("n_vetoes")} veto(es): not decided under the rule, so no rule label '
                           '(Addendum 79 ruling 1)')
        return None, f'superseded cell whose record v6 status is {st!r}: no "---" derivation'
    return expected_rule(c, cond, J['SPEC_V6'])


# ======================================================================================================================
#  8. W180a: generator, verifier, bitwise identity
# ======================================================================================================================
def run_generator(out_full):
    if os.path.exists(out_full):
        raise RuntimeError(f'{out_full} exists: the generator would overwrite it')
    argv0 = list(sys.argv)
    buf = io.StringIO()
    sys.argv = [os.path.join(REPO, GEN_REL), out_full]
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(REPO, GEN_REL), run_name='__main__')
    finally:
        sys.argv = argv0
    gc.collect()
    with open(out_full, 'rb') as h:
        data = h.read()
    return data, buf.getvalue()


def run_verifier():
    argv0 = list(sys.argv)
    buf = io.StringIO()
    err, tmp_out = None, None
    sys.argv = [os.path.join(REPO, VERIFY_REL)]
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(REPO, VERIFY_REL), run_name='__main__')
        tmp_out = sys.argv[1] if len(sys.argv) > 1 else None   # the verifier sets sys.argv to its tempdir target
    except Exception as e:  # recorded, never absorbed: the integrity check fails on it
        err = repr(e)
    finally:
        sys.argv = argv0
    gc.collect()
    text = buf.getvalue()
    lines = text.split('\n')
    tmp_sha = None
    if tmp_out and os.path.exists(tmp_out):
        with open(tmp_out, 'rb') as h:
            tmp_sha = _sha_bytes(h.read())
    return {'output': text, 'n_pass': sum(1 for ln in lines if ln.startswith('PASS ')),
            'n_fail': sum(1 for ln in lines if ln.startswith('FAIL ')), 'fail_lines': [ln for ln in lines if ln.startswith('FAIL ')],
            'exception': err, 'tempdir_output': tmp_out, 'tempdir_output_sha256': tmp_sha}


def locate_bytes(hay, needle):
    hits, i = [], hay.find(needle)
    while i >= 0:
        hits.append(i)
        i = hay.find(needle, i + 1)
    out = []
    for s in hits:
        e = s + len(needle)
        start_line = hay[:s].count(b'\n') + 1
        n_lines = needle.count(b'\n')
        out.append({'byte_start': s, 'byte_end': e, 'start_line': start_line, 'end_line': start_line + n_lines - 1,
                    'n_lines': n_lines, 'starts_at_line_start': s == 0 or hay[s - 1:s] == b'\n',
                    'ends_at_line_end': needle.endswith(b'\n')})
    return out


def appendix_block(lines):
    """[i0, i1) of the Appendix E block: from the '% ====' rule above '% APPENDIX E -- EVALUATION RECORD' to the line
    before '\\section*{Declaration'."""
    ih = next(i for i, ln in enumerate(lines) if ln.startswith('% APPENDIX E -- EVALUATION RECORD'))
    i0 = ih - 1 if lines[ih - 1].startswith('% =====') else ih
    i1 = next(i for i in range(ih, len(lines)) if lines[i].startswith('\\section*{Declaration'))
    return i0, i1


# ======================================================================================================================
#  9. W180a: the entry check (W179a sections 3-5 with the row-16 convention) and its tallies
# ======================================================================================================================
def table_check(lines, ALL, COND, J, T):
    e21_rows, e21_note = parse_e21(lines)
    e21_out, idmap, p21 = check_e21(e21_rows, ALL, COND, J)
    e20_rows, e20_note, e20_note_line = parse_e20(lines)
    e20_out, used, p20 = check_e20(e20_rows, idmap, ALL, COND, T)
    e22_out, e22_note = check_e22(lines, idmap, ALL, T)
    e23_out, e23_note = check_e23(lines, T)
    e24_out = check_e24(lines, idmap, ALL, T, e21_out)
    summ, flat = {}, []
    for name, rows, extra in (('E.20', e20_out, None), ('E.21', e21_out, None), ('E.22', e22_out, {'note_superseded': e22_note}),
                              ('E.23', e23_out, e23_note), ('E.24', e24_out, None)):
        s = {'rows': len(rows), 'entries': 0, 'MATCH': 0, 'MISMATCH': 0, 'NOT_FROZEN_AGREES': 0, 'NOT_FROZEN_DISAGREES': 0}
        items = [(r.get('no') or r.get('eval') or r.get('row'), k, e) for r in rows for k, e in r.get('entries', {}).items()]
        items += [('note', k, e) for k, e in (extra or {}).items()]
        for rid, k, e in items:
            s['entries'] += 1
            s[e['status']] = s.get(e['status'], 0) + 1
            flat.append({'table': name, 'row': rid, 'key': k, 'status': e['status'], 'printed': e.get('printed'),
                         'frozen_or_derived': e.get('frozen', e.get('derived'))})
        summ[name] = s
    bad = [f for f in flat if f['status'] in ('MISMATCH', 'NOT_FROZEN_DISAGREES')]
    return {'summary': summ, 'flat': flat, 'bad': bad, 'mapping_problems': p20 + p21, 'idmap': idmap, 'used': used,
            'E20': e20_out, 'E21': e21_out, 'E22': e22_out, 'E22_note': e22_note, 'E23': e23_out, 'E23_note': e23_note,
            'E24': e24_out, 'e21_note': e21_note, 'e20_note_line': e20_note_line}


def rounding_detail(R):
    """W179a's rounding census of E.20-E.24 (k EUR amounts; ratio columns of E.20-E.22; E.24 Resid.), plus the E.24
    shares (%) reported beside."""
    kdp, rdp, resid_dp, share_dp = [], [], [], []
    for r in R['E20']:
        for k in ('d_gross_k', 'threshold_or_bar_k', 'd_net_k'):
            p = r.get('entries', {}).get(k, {}).get('printed')
            if p and p != '---':
                kdp.append(dec_places(p.replace('$-$', '-')))
        for k in ('x_gross', 'x_net'):
            p = r.get('entries', {}).get(k, {}).get('printed')
            if p and p != '---':
                rdp.append(dec_places(p))
    for r in R['E21']:
        for k in ('band_k', 'gap_k', 'slack_k'):
            p = r.get('entries', {}).get(k, {}).get('printed')
            if p and p != '---':
                kdp.append(dec_places(p))
        p = r.get('entries', {}).get('range_over_tau', {}).get('printed')
        if p and p != '---':
            rdp.append(dec_places(p))
    for r in R['E22']:
        for k, e in r['entries'].items():
            if k != 'nodes_year':
                kdp += [dec_places(e['printed'][0].replace('$-$', '-')), dec_places(e['printed'][1])]
                rdp.append(dec_places(e['printed'][2]))
    for r in R['E23']:
        for k in ('Q_k', 'consistency_pass_k'):
            p = r.get('entries', {}).get(k, {}).get('printed')
            if p and p != '---':
                kdp.append(dec_places(p.replace('$-$', '-').replace('\\,', '')))
    for r in R['E24']:
        p = r['entries'].get('gap_k', {}).get('printed')
        if p and p != '---':
            kdp.append(dec_places(p.replace('$-$', '-')))
        p = r['entries'].get('pf_primal_last', {}).get('printed')
        if p and p != '---':
            resid_dp.append(dec_places(p))
        for n in '579':
            p = r['entries'].get(f'share_n{n}_pct', {}).get('printed')
            if p and p != '---':
                share_dp.append(dec_places(p))
    return {'k_EUR_dp': sorted(set(kdp)), 'n_k': len(kdp), 'ratio_dp_E20_E22': sorted(set(rdp)), 'n_ratio': len(rdp),
            'E24_resid_ratio_dp': sorted(set(resid_dp)), 'n_E24_resid': len(resid_dp),
            'E24_share_pct_dp_beside': sorted(set(share_dp)), 'n_E24_share': len(share_dp)}


# ======================================================================================================================
#  10. W180a: raw parse of the five tables and the entry-by-entry diff
# ======================================================================================================================
TABLES = (('E.20', 'tab:appE_claims'), ('E.21', 'tab:appE_evaluations'), ('E.22', 'tab:appE_neighbours'),
          ('E.23', 'tab:appE_benchmark'), ('E.24', 'tab:appE_gap'))
COLS = {'E.20': ['No.', 'Comparison', 'Eval.', 'Delta gross (kEUR)', 'Threshold or bar (kEUR)', 'x gross', 'Verdict gross',
                 'Delta net (kEUR)', 'x net', 'Verdict net'],
        'E.21': ['No.', 'Plan', 'Year', 'Condition', 'Status', 'k0', 'k*/end', 'Range', 'Band', '|Gap|', '|Slack|', 'N.c.',
                 'Replay', 'Rule'],
        'E.22': ['Node(s)', 'Year', 'Eval.', 'Delta (kEUR)', 'Threshold (kEUR)', 'x'],
        'E.23': ['Arrangement', 'Start', 'Q (kEUR)', 'NRF n', 'NRF largest (p.u.)', 'Consistency pass (kEUR)',
                 'Curtailment TN (GWh)', 'Curtailment DN (GWh)'],
        'E.24': ['Eval.', 'Plan', 'Year', 'Cond.', 'Cause', 'Gap (kEUR)', 'Share n.5', 'Share n.7', 'Share n.9', 'Resid.',
                 'Lapse resets']}


def table_region(lines, label):
    il = next(i for i, ln in enumerate(lines) if f'\\label{{{label}}}' in ln)
    i0 = max(i for i in range(il + 1) if '\\begin{longtable}' in lines[i] or '\\begin{table}' in lines[i])
    env = 'longtable' if '\\begin{longtable}' in lines[i0] else 'table'
    i1 = next(i for i in range(il, len(lines)) if f'\\end{{{env}}}' in lines[i])
    return i0, i1


def raw_tables(lines):
    out = {}
    for t, label in TABLES:
        i0, i1 = table_region(lines, label)
        if t in ('E.20', 'E.21'):
            data_idx = [i for i in range(i0, i1 + 1) if re.match(r'^\d+ & ', lines[i].strip())]
        else:
            data_idx = [r['line'] - 1 for r in tabular_rows(lines, label)[0]]
        rows, caption, notes, groups, other, arm = {}, [], [], [], [], ''
        for i in range(i0, i1 + 1):
            ln = lines[i]
            if i in data_idx:
                cells = row_cells(ln)
                if t == 'E.22':
                    key = cells[2]
                elif t == 'E.23':
                    arm = cells[0] or arm
                    key = f'{arm}|{strip_markers(cells[1])[0]}'
                else:
                    key = cells[0]
                if key in rows:
                    raise RuntimeError(f'{t}: duplicate row key {key!r}')
                rows[key] = {'line': i + 1, 'cells': cells}
            elif '\\caption' in ln:
                caption.append({'line': i + 1, 'text': ln.strip()})
            elif '\\textit{Note:}' in ln:
                notes.append({'line': i + 1, 'text': ln.strip()})
            elif re.search(r'\\textit\{\([a-f]\)', ln):
                groups.append({'line': i + 1, 'text': ln.strip()})
            else:
                other.append({'line': i + 1, 'text': ln.strip()})
        out[t] = {'label': label, 'region': [i0 + 1, i1 + 1], 'rows': rows, 'caption': caption, 'notes': notes,
                  'group_headers': groups, 'other_lines': other}
    return out


def token_diff(a, b):
    ta, tb = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    return [{'op': op, 'old': ' '.join(ta[i1:i2]), 'new': ' '.join(tb[j1:j2])}
            for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal']


def diff_tables(A, B):
    ch = {'cells': [], 'row_set_changes': [], 'captions': [], 'notes': [], 'group_headers': [], 'other_lines': []}
    n_cells = 0
    for t in A:
        ra, rb = A[t]['rows'], B[t]['rows']
        if list(ra) != list(rb):
            ch['row_set_changes'].append({'table': t, 'old_keys': list(ra), 'new_keys': list(rb)})
        for k in ra:
            if k not in rb:
                continue
            ca, cb = ra[k]['cells'], rb[k]['cells']
            if len(ca) != len(cb):
                ch['row_set_changes'].append({'table': t, 'row': k, 'n_cells': [len(ca), len(cb)]})
                continue
            for j, (x, y) in enumerate(zip(ca, cb)):
                n_cells += 1
                if x != y:
                    ch['cells'].append({'table': t, 'row': k, 'col': j, 'column': COLS[t][j], 'old': x, 'new': y,
                                        'line_old': ra[k]['line'], 'line_new': rb[k]['line']})
        for part in ('caption', 'notes', 'group_headers', 'other_lines'):
            la = [x['text'] for x in A[t][part]]
            lb = [x['text'] for x in B[t][part]]
            if la == lb:
                continue
            sm = difflib.SequenceMatcher(a=la, b=lb, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == 'equal':
                    continue
                old = '\n'.join(la[i1:i2])
                new = '\n'.join(lb[j1:j2])
                ch['captions' if part == 'caption' else part].append({
                    'table': t, 'op': op, 'lines_old': [x['line'] for x in A[t][part][i1:i2]],
                    'lines_new': [x['line'] for x in B[t][part][j1:j2]], 'old': old, 'new': new,
                    'token_changes': token_diff(old, new)})
    ch['n_cells_compared'] = n_cells
    return ch


def block_diff(la, a0, lb, b0, lines_b):
    """Line diff of the two Appendix E blocks; every hunk with absolute line numbers and its location."""
    sm = difflib.SequenceMatcher(a=la, b=lb, autojunk=False)
    regions = {t: table_region(lines_b, label) for t, label in TABLES}
    hunks = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            continue
        nb = list(range(b0 + j1 + 1, b0 + j2 + 1))
        anchor = b0 + (j1 if j2 > j1 else max(j1 - 1, 0))
        sect = next((lines_b[i].strip() for i in range(anchor, b0 - 1, -1)
                     if lines_b[i].lstrip().startswith(('\\subsection', '\\section'))), 'block header')
        in_table = [t for t, (r0, r1) in regions.items() if r0 <= anchor <= r1]
        touched = [x for x in la[i1:i2] + lb[j1:j2] if x.strip()]
        kind = ('comment' if touched and all(x.lstrip().startswith('%') for x in touched) else
                f'table {in_table[0]}' if in_table else 'prose')
        hunks.append({'op': op, 'lines_old': [a0 + i1 + 1, a0 + i2], 'lines_new': [b0 + j1 + 1, b0 + j2], 'section': sect,
                      'kind': kind, 'old': la[i1:i2], 'new': lb[j1:j2],
                      'token_changes': token_diff('\n'.join(la[i1:i2]), '\n'.join(lb[j1:j2]))})
    return hunks


# ======================================================================================================================
#  11. W180a: mutation test of both comparisons (in memory only)
# ======================================================================================================================
def mutate_cell(lines, raw, table, row, col, fn):
    L = list(lines)
    r = raw[table]['rows'][row]
    i = r['line'] - 1
    cells = list(r['cells'])
    old = cells[col]
    cells[col] = fn(old)
    indent = re.match(r'^\s*', L[i]).group(0)
    L[i] = indent + ' & '.join(cells) + ' \\\\'
    return L, old, cells[col]


def mutate_text(lines, raw, table, part, needle, repl):
    L = list(lines)
    hits = [x['line'] - 1 for x in raw[table][part] if needle in x['text']]
    if len(hits) != 1:
        raise RuntimeError(f'mutation needle {needle!r} in {table} {part}: {len(hits)} hits')
    L[hits[0]] = L[hits[0]].replace(needle, repl, 1)
    return L, needle, repl


def _bump_last_digit(s):
    m = re.search(r'(\d)(?!.*\d)', s)
    d = int(m.group(1))
    return s[:m.start()] + str((d + 1) % 10) + s[m.end():]



# ======================================================================================================================
#  12. W181a: unified-diff parsing and the identity comparison (item (a))
# ======================================================================================================================
HUNK_RX = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$')


def parse_udiff(text, default_file=None):
    """{file: [hunk]} from a unified diff. A hunk is read by its header counts (so a removed line that starts with
    '-- ' is never mistaken for a file header). File names: 'diff --git a/X b/X' -> X; else default_file."""
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines = lines[:-1]
    files, cur, i, n = {}, default_file, 0, len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith('diff --git '):
            cur = ln.split(' b/', 1)[1]
            i += 1
            continue
        m = HUNK_RX.match(ln)
        if not m:
            i += 1          # file headers (---, +++), index, mode lines
            continue
        os_, ol, ns, nl_ = int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)
        h = {'header': ln, 'old_start': os_, 'old_len': ol, 'new_start': ns, 'new_len': nl_,
             'header_context': m.group(5).strip(), 'lines': []}
        ro, rn = ol, nl_
        i += 1
        while i < n and (ro > 0 or rn > 0):
            t = lines[i]
            if t.startswith('\\'):
                h['lines'].append(('\\', t))
            elif t.startswith('-'):
                h['lines'].append(('-', t[1:]))
                ro -= 1
            elif t.startswith('+'):
                h['lines'].append(('+', t[1:]))
                rn -= 1
            else:            # ' ' context (or an empty line where an editor stripped the single space)
                h['lines'].append((' ', t[1:] if t.startswith(' ') else t))
                ro -= 1
                rn -= 1
            i += 1
        h['counts_consistent'] = ro == 0 and rn == 0
        while i < n and lines[i].startswith('\\'):
            h['lines'].append(('\\', lines[i]))
            i += 1
        files.setdefault(cur, []).append(h)
    return files


def compare_hunks(got, exp):
    """Per hunk, in order: +/- lines equal (the test); ranges and context lines reported beside."""
    per = []
    for k in range(max(len(got), len(exp))):
        g = got[k] if k < len(got) else None
        e = exp[k] if k < len(exp) else None
        if g is None or e is None:
            per.append({'hunk': k + 1, 'present_in_clone_diff': g is not None, 'present_in_expert_diff': e is not None,
                        'pm_lines_equal': False})
            continue
        gpm = [x for x in g['lines'] if x[0] in '+-']
        epm = [x for x in e['lines'] if x[0] in '+-']
        gctx = [x for x in g['lines'] if x[0] == ' ']
        ectx = [x for x in e['lines'] if x[0] == ' ']
        per.append({'hunk': k + 1, 'clone_header': g['header'], 'expert_header': e['header'],
                    'ranges_equal': (g['old_start'], g['old_len'], g['new_start'], g['new_len'])
                    == (e['old_start'], e['old_len'], e['new_start'], e['new_len']),
                    'n_minus': sum(1 for x in gpm if x[0] == '-'), 'n_plus': sum(1 for x in gpm if x[0] == '+'),
                    'pm_lines_equal': gpm == epm, 'context_lines_equal': gctx == ectx,
                    'full_body_equal': [x for x in g['lines'] if x[0] != '\\'] == [x for x in e['lines'] if x[0] != '\\'],
                    'counts_consistent': g['counts_consistent'] and e['counts_consistent'],
                    'first_pm_mismatch': next(({'i': j, 'clone': a, 'expert': b} for j, (a, b) in enumerate(zip(gpm, epm))
                                               if a != b), None) if gpm != epm else None})
    all_pm_in_order = ([x for h in got for x in h['lines'] if x[0] in '+-']
                       == [x for h in exp for x in h['lines'] if x[0] in '+-'])
    return {'n_hunks_clone': len(got), 'n_hunks_expert': len(exp), 'per_hunk': per,
            'every_hunk_pm_equal': len(got) == len(exp) and all(p['pm_lines_equal'] for p in per),
            'concatenated_pm_lines_equal_in_order': all_pm_in_order,
            'every_hunk_range_equal': len(got) == len(exp) and all(p.get('ranges_equal') for p in per),
            'every_hunk_context_equal': len(got) == len(exp) and all(p.get('context_lines_equal') for p in per)}


# ======================================================================================================================
#  13. W181a: classification of the block hunks
# ======================================================================================================================
PART_CLASS = {'rows': 'cell', 'caption': 'caption', 'notes': 'note', 'group_headers': 'group header',
              'other_lines': 'table other'}


def line_classes(raw, lines):
    """{1-based line: class} for every line of a table region (from the raw parse), else by content."""
    m = {}
    for t, d in raw.items():
        for k, r in d['rows'].items():
            m[r['line']] = (t, 'cell')
        for part in ('caption', 'notes', 'group_headers', 'other_lines'):
            for x in d[part]:
                m[x['line']] = (t, PART_CLASS[part])
    return m


def classify_hunk(h, old_lines, new_lines, cls_old, cls_new):
    out = set()
    for (a, b), L, C in (((h['lines_old'][0], h['lines_old'][1]), old_lines, cls_old),
                         ((h['lines_new'][0], h['lines_new'][1]), new_lines, cls_new)):
        for ln in range(a, b + 1):
            txt = L[ln - 1]
            if not txt.strip():
                continue
            if txt.lstrip().startswith('%'):
                out.add('comment')
            elif ln in C:
                out.add(f'{C[ln][0]} {C[ln][1]}')
            else:
                out.add('prose')
    return sorted(out)


# ======================================================================================================================
#  14. post-run, guards, other copies, summary
# ======================================================================================================================
def guards_state():
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    g = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    pk = {'counts': dict(PICKLE_COUNTS), 'pickle_load_and_loads_blocked': blocked}
    pk['ok'] = blocked and PICKLE_COUNTS == {'load': 0, 'loads': 0}
    return g, pk, (not g['verify_0_failures']) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('man', 'json', 'md', 'gen', 'verify', 'typing_json', 'typing_log')]
    rels += [os.path.join(out_dir, f) for f in sorted(os.listdir(os.path.join(REPO, out_dir))) if LAUNCH_LOG_RX.fullmatch(f)]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W181a post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W181a post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}: " + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def other_copies():
    r = subprocess.run(['ps', '-Ao', 'pid=,args='], capture_output=True, text=True).stdout.splitlines()
    me = os.getpid()
    out = []
    for ln in r:
        parts = ln.strip().split(None, 1)
        if len(parts) < 2 or int(parts[0]) == me:
            continue
        argv = parts[1].split()
        if argv and 'python' in os.path.basename(argv[0]) and any(a.endswith(SCRIPT_REL) for a in argv[1:3]) \
                and '--post-run' not in argv:
            out.append(ln.strip())
    return out


def _cell(s, n=160):
    s = str(s).replace('|', '\\|').replace('\n', ' ')
    return s if len(s) <= n else s[:n - 1] + '…'


def md_summary(o):
    L = [f"# W181a — identity and Appendix E at Overleaf {o['clone_head'][:7]}", '',
         f"main.tex sha256 `{o['identity']['head']['main.tex']['sha256']}`; letter sha256 "
         f"`{o['identity']['head'][LETTER_REL_CLONE]['sha256']}`; frozen JSON `{FZ_SHA[:8]}`; script `{SCRIPT_REL}`. "
         f"Zero solves; guards {'0/0 verified' if o['guards_ok'] else 'FAILED'}.", '', '## Predictions', '']
    for p in o['predictions']:
        L.append(f"- {p['prediction']}: **{p['outcome']}** — {_cell(p['detail'], 900)}")
    I = o['identity']
    L += ['', '## (a) Identity', '', '| revision | file | sha256 | lines | equals |', '|---|---|---|---|---|']
    for rev in ('head', 'prev'):
        for f, d in I[rev].items():
            L.append(f"| {I['revs'][rev][:7]} | {f} | `{d['sha256'][:16]}…` | {d['lines']} | {d['equals']} |")
    L += ['', f"Files in `git diff {PREV_REV} {o['clone_head'][:7]}`: {I['diff_files']}", '']
    for f, c in I['diff_comparison'].items():
        L.append(f"- {f}: clone {c['n_hunks_clone']} hunks, expert {c['n_hunks_expert']} hunks; every hunk +/- equal "
                 f"{c['every_hunk_pm_equal']}; concatenated +/- in order {c['concatenated_pm_lines_equal_in_order']}; "
                 f"ranges equal {c['every_hunk_range_equal']}; context equal {c['every_hunk_context_equal']}")
        for p in c['per_hunk']:
            L.append(f"  - hunk {p['hunk']}: {_cell(p.get('clone_header'), 60)} | -{p.get('n_minus')} +{p.get('n_plus')} | "
                     f"+/- equal {p['pm_lines_equal']}, range equal {p.get('ranges_equal')}, context equal {p.get('context_lines_equal')}")
    g = o['generator']
    L += ['', '## (b) Generator / verifier / bitwise', '',
          f"- generator stdout `{g['stdout'].strip()}`; output `{g['output']}` sha256 `{g['sha256']}` ({g['n_lines']} lines)",
          f"- verifier: {o['verifier']['n_pass']} PASS, {o['verifier']['n_fail']} FAIL; exception {o['verifier']['exception']}; "
          f"tempdir output sha256 `{o['verifier']['tempdir_output_sha256']}`",
          f"- occurrences in main.tex at {o['clone_head'][:7]}: {o['bitwise']['main_tex_occurrences']}",
          f"- = committed {APPX_REL}: {o['bitwise']['equals_committed_appendix']}; = verifier regeneration: "
          f"{o['bitwise']['equals_verifier_regeneration']}",
          f"- Appendix E block (comment rule to the line before Declaration): at {o['clone_head'][:7]} lines "
          f"{o['bitwise']['block_head_lines']}; at {PREV_REV} lines {o['bitwise']['block_prev_lines']}",
          '', f"## Table entries at {o['clone_head'][:7]} vs the frozen JSON", '',
          '| table | rows | entries | MATCH | MISMATCH | not a frozen field, agrees | not a frozen field, disagrees |',
          '|---|---|---|---|---|---|---|']
    for t, s in o['table_check_head']['summary'].items():
        L.append(f"| {t} | {s['rows']} | {s['entries']} | {s['MATCH']} | {s['MISMATCH']} | {s['NOT_FROZEN_AGREES']} | {s['NOT_FROZEN_DISAGREES']} |")
    L += ['', 'Every MISMATCH / disagreement:', '']
    bad = o['table_check_head']['bad']
    L += [f"- {b}" for b in bad] if bad else ['- none']
    L += ['', f"Cross-check: {PREV_REV} under the same convention, non-agreeing entries: "
          f"{o['table_check_prev']['bad']}", '', f"## Entry diff {PREV_REV} -> {o['clone_head'][:7]}", '',
          f"Table cells compared: {o['table_diff']['n_cells_compared']}; changed: {len(o['table_diff']['cells'])}", '']
    for c in o['table_diff']['cells']:
        L.append(f"- cell {c['table']} row {c['row']} col {c['col']} ({c['column']}): `{c['old']}` -> `{c['new']}`")
    for part in ('row_set_changes', 'captions', 'notes', 'group_headers', 'other_lines'):
        L.append(f"- {part}: {len(o['table_diff'][part])}")
        for c in o['table_diff'][part]:
            L.append(f"  - {c.get('table')} l. {c.get('lines_old')} -> {c.get('lines_new')}: {_cell(c.get('token_changes'), 700)}")
    L += ['', '## Appendix E block line diff (hunks, classified)', '',
          '| # | class | section | lines old | lines new | token changes |', '|---|---|---|---|---|---|']
    for i, h in enumerate(o['block_diff'], 1):
        L.append(f"| {i} | {', '.join(h['class'])} | {_cell(h['section'], 60)} | {h['lines_old']} | {h['lines_new']} | "
                 f"{_cell(h['token_changes'], 500)} |")
    L += ['', f"Generator output W180a -> W181a: {len(o['generator_vs_w180a']['hunks'])} hunk(s), same +/- lines as the "
          f"block diff: {o['generator_vs_w180a']['same_as_block_diff']}", '',
          '## Mutation test', '', '| id | edit | caught by frozen check | caught by diff |', '|---|---|---|---|']
    for m in o['mutation_test']['cases']:
        L.append(f"| {m['id']} | {_cell(m['edit'], 110)} | {m['caught_frozen_check']} | {m['caught_diff']} |")
    L += ['', '## Integrity checks', '']
    L += [f"- {k}: {v}" for k, v in o['checks'].items()]
    return '\n'.join(L) + '\n'


# ======================================================================================================================
#  15. main
# ======================================================================================================================
def _nlines(b):
    return b.count(b'\n') + (0 if b.endswith(b'\n') or not b else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--trial-out-dir', default=None, help='trial runs only: an output directory outside the results')
    args = ap.parse_args()
    out_dir = args.trial_out_dir or OUT_DIR
    tag = 'W181a'
    if args.post_run:
        post_run(out_dir)
    t0 = time.time()
    pre = []
    others = other_copies()
    if others:
        pre.append(f'another copy of {SCRIPT_REL} is running: {others}')
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    if not os.path.isdir(od):
        pre.append(f'{out_dir} missing (the launch command creates it with mkdir)')
    else:
        extra = [f for f in sorted(os.listdir(od)) if not LAUNCH_LOG_RX.fullmatch(f)]
        if extra:
            pre.append(f'{out_dir} is not new: it holds {extra} (write-once)')
    clone_full = os.path.join(REPO, CLONE)
    head = _git('rev-parse', 'HEAD', cwd=clone_full)
    if head != CLONE_HEAD:
        pre.append(f'clone HEAD {head} != {CLONE_HEAD}')
    parent = _git('rev-parse', 'HEAD^', cwd=clone_full)
    prev_full = _git('rev-parse', PREV_REV, cwd=clone_full)
    if prev_full != PREV_REV_FULL:
        pre.append(f'{PREV_REV} resolves to {prev_full} != {PREV_REV_FULL}')
    if parent != PREV_REV_FULL:
        pre.append(f'HEAD^ = {parent} is not {PREV_REV_FULL} (HEAD is not one commit after {PREV_REV})')
    clone_log = _git('log', '--format=%H %s', '-3', cwd=clone_full).split('\n')
    porcelain = _git('status', '--porcelain', '--', 'main.tex', LETTER_REL_CLONE, cwd=clone_full)
    if porcelain:
        pre.append(f'clone main.tex / letter not clean: {porcelain!r}')
    main_rel = os.path.join(CLONE, 'main.tex')
    letter_rel = os.path.join(CLONE, LETTER_REL_CLONE)
    with open(os.path.join(REPO, main_rel), 'rb') as h:
        main_bytes = h.read()
    with open(os.path.join(REPO, letter_rel), 'rb') as h:
        letter_bytes = h.read()
    main_sha, letter_sha = _sha_bytes(main_bytes), _sha_bytes(letter_bytes)
    if main_sha != MAIN_SHA:
        pre.append(f'main.tex sha256 {main_sha} != {MAIN_SHA}')
    if letter_sha != LETTER_SHA:
        pre.append(f'letter sha256 {letter_sha} != {LETTER_SHA}')
    head_main_git = subprocess.run(['git', 'show', 'HEAD:main.tex'], capture_output=True, cwd=clone_full).stdout
    head_letter_git = subprocess.run(['git', 'show', f'HEAD:{LETTER_REL_CLONE}'], capture_output=True, cwd=clone_full).stdout
    prev_bytes = subprocess.run(['git', 'show', f'{PREV_REV}:main.tex'], capture_output=True, cwd=clone_full).stdout
    prev_letter = subprocess.run(['git', 'show', f'{PREV_REV}:{LETTER_REL_CLONE}'], capture_output=True, cwd=clone_full).stdout
    prev_sha = _sha_bytes(prev_bytes)
    if prev_sha != PREV_SHA:
        pre.append(f'{PREV_REV}:main.tex sha256 {prev_sha} != {PREV_SHA} (W180a)')
    if _sha_bytes(prev_letter) != PREV_LETTER_SHA:
        pre.append(f'{PREV_REV}:{LETTER_REL_CLONE} sha256 {_sha_bytes(prev_letter)} != {PREV_LETTER_SHA} (letter v7)')
    inputs = {}
    for k, rel in INPUTS.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'capture path missing: {rel}')
            continue
        inputs[k] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': _committed_clean(rel),
                     'last_commit': _git('log', '-1', '--format=%h', '--', rel)}
        if not inputs[k]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    if inputs.get('FROZEN_JSON', {}).get('sha256') != FZ_SHA:
        pre.append('frozen JSON sha256 differs')
    gen_out_rel = os.path.join(out_dir, OUT_NAMES['gen'])
    gen_out_full = gen_out_rel if os.path.isabs(gen_out_rel) else os.path.join(REPO, gen_out_rel)
    if os.path.realpath(gen_out_full) in (os.path.realpath(os.path.join(REPO, APPX_REL)),
                                          os.path.realpath(os.path.join(REPO, 'manuscript_review', 'appE.tex'))):
        pre.append('generator target is a manuscript_review file')
    if not os.path.realpath(gen_out_full).startswith(os.path.realpath(od) + os.sep):
        pre.append('generator target is not inside the output directory')
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] ' + ' | '.join(pre))
        sys.exit(1)
    _log(f'[{tag}] preconditions hold: clone HEAD {head[:8]} (parent {parent[:8]}), main.tex {main_sha[:8]}, letter '
         f'{letter_sha[:8]}, {PREV_REV}:main.tex {prev_sha[:8]} (= W180a), {len(inputs)} inputs committed clean')

    J = {k: _json(v['path']) for k, v in inputs.items() if v['path'].endswith('.json')}
    FZ = J['FROZEN_JSON']
    T = FZ['tables']
    ALL = dict(T['cells'])
    ALL.update(T['cells_appended_w160'])
    for k, c in ALL.items():
        c.setdefault('cell', k)
    COND = cell_conditions(J, ALL)
    lines = main_bytes.decode('utf-8').split('\n')
    plines = prev_bytes.decode('utf-8').split('\n')
    checks = {'main_tex_read_equals_git_show_HEAD': main_bytes == head_main_git,
              'letter_read_equals_git_show_HEAD': letter_bytes == head_letter_git}

    # ---- (a) identity ----------------------------------------------------------------------------------------------
    def rd(rel):
        with open(os.path.join(REPO, rel), 'rb') as h:
            return h.read()
    rc, rcp, l8, l7 = rd(INPUTS['REVIEW_COPY']), rd(INPUTS['REVIEW_COPY_PREV']), rd(INPUTS['LETTER_V8']), rd(INPUTS['LETTER_V7'])
    ident = {'revs': {'head': head, 'prev': prev_full}, 'clone_log_top3': clone_log, 'head_parent': parent,
             'head': {'main.tex': {'sha256': main_sha, 'lines': _nlines(main_bytes),
                                   'equals': {INPUTS['REVIEW_COPY']: main_bytes == rc}},
                      LETTER_REL_CLONE: {'sha256': letter_sha, 'lines': _nlines(letter_bytes),
                                         'equals': {INPUTS['LETTER_V8']: letter_bytes == l8}}},
             'prev': {'main.tex': {'sha256': prev_sha, 'lines': _nlines(prev_bytes),
                                   'equals': {INPUTS['REVIEW_COPY_PREV']: prev_bytes == rcp}},
                      LETTER_REL_CLONE: {'sha256': _sha_bytes(prev_letter), 'lines': _nlines(prev_letter),
                                         'equals': {INPUTS['LETTER_V7']: prev_letter == l7}}}}
    diff_files = [x for x in _git('diff', '--name-only', PREV_REV_FULL, head, cwd=clone_full).split('\n') if x]
    clone_diff_text = subprocess.run(['git', 'diff', PREV_REV_FULL, head], capture_output=True, text=True,
                                     cwd=clone_full).stdout
    clone_diff = parse_udiff(clone_diff_text)
    exp_main = parse_udiff(rd(INPUTS['MAIN_DIFF']).decode('utf-8'), default_file='main.tex')
    exp_letter = parse_udiff(rd(INPUTS['LETTER_DIFF']).decode('utf-8'), default_file=LETTER_REL_CLONE)
    expert = {'main.tex': exp_main.get('main.tex', []), LETTER_REL_CLONE: exp_letter.get(LETTER_REL_CLONE, [])}
    ident['diff_files'] = diff_files
    ident['diff_comparison'] = {f: compare_hunks(clone_diff.get(f, []), expert[f]) for f in EXPECTED_DIFF_FILES}
    ident['clone_diff_sha256'] = _sha_bytes(clone_diff_text.encode('utf-8'))
    ident['parsed_files_clone_diff'] = sorted(clone_diff)
    ident['parsed_files_expert_diffs'] = [sorted(exp_main), sorted(exp_letter)]
    # negative control of the diff comparison (in memory): one character of one '+' line of each expert diff changed
    neg = {}
    for f in EXPECTED_DIFF_FILES:
        mut = json.loads(json.dumps(expert[f]))
        hk = next(k for k, h in enumerate(mut) if any(x[0] == '+' for x in h['lines']))
        j = next(j for j, x in enumerate(mut[hk]['lines']) if x[0] == '+')
        mut[hk]['lines'][j] = ['+', mut[hk]['lines'][j][1] + 'X']
        mut = [dict(h, lines=[tuple(x) for x in h['lines']]) for h in mut]
        cm = compare_hunks(clone_diff.get(f, []), mut)
        neg[f] = {'hunk_mutated': hk + 1, 'caught': (not cm['every_hunk_pm_equal']) and (not cm['concatenated_pm_lines_equal_in_order'])
                  and [p['hunk'] for p in cm['per_hunk'] if not p['pm_lines_equal']] == [hk + 1]}
    ident['diff_comparison_negative_control'] = neg
    checks['diff_comparison_negative_control_caught'] = all(v['caught'] for v in neg.values())
    checks['diff_parse_counts_consistent'] = all(h['counts_consistent'] for d in (clone_diff, exp_main, exp_letter)
                                                 for hs in d.values() for h in hs)
    checks['main_tex_lines_2713'] = _nlines(main_bytes) == MAIN_LINES
    checks['prev_main_tex_lines_2710'] = _nlines(prev_bytes) == PREV_LINES
    _log(f"[{tag}] (a) main.tex {main_sha[:8]} {_nlines(main_bytes)} lines = round7 copy {main_bytes == rc}; letter "
         f"{letter_sha[:8]} {_nlines(letter_bytes)} lines = v8 {letter_bytes == l8}; {PREV_REV}: main {prev_sha[:8]} = round6 "
         f"{prev_bytes == rcp}, letter {_sha_bytes(prev_letter)[:8]} = v7 {prev_letter == l7}")
    _log(f'[{tag}] (a) files in git diff {PREV_REV}..{head[:7]}: {diff_files}')
    for f, c in ident['diff_comparison'].items():
        _log(f"[{tag}] (a) {f}: hunks clone {c['n_hunks_clone']} / expert {c['n_hunks_expert']}; every hunk +/- equal "
             f"{c['every_hunk_pm_equal']}; in order {c['concatenated_pm_lines_equal_in_order']}; ranges "
             f"{c['every_hunk_range_equal']}; context {c['every_hunk_context_equal']}")
        for p in c['per_hunk']:
            _log(f"[{tag}] (a)   hunk {p['hunk']} {p.get('clone_header', '')[:40]!r}: -{p.get('n_minus')} +{p.get('n_plus')} "
                 f"+/- equal {p['pm_lines_equal']}, range {p.get('ranges_equal')}, context {p.get('context_lines_equal')}")

    # ---- (b1) generator and verifier --------------------------------------------------------------------------------
    gen_bytes, gen_stdout = run_generator(gen_out_full)
    _log(f'[{tag}] generator: stdout {gen_stdout.strip()!r}; wrote {gen_out_rel} sha256 {_sha_bytes(gen_bytes)[:8]}')
    ver = run_verifier()
    _log(f"[{tag}] verifier: {ver['n_pass']} PASS, {ver['n_fail']} FAIL, exception {ver['exception']}; full output:")
    for ln in ver['output'].rstrip('\n').split('\n'):
        print(f'    | {ln}', flush=True)
    with open(os.path.join(od, OUT_NAMES['verify']), 'x', encoding='utf-8') as h:
        h.write(ver['output'])
    checks['generator_ran'] = gen_stdout.startswith('ok ')
    checks['verifier_ran_without_exception'] = ver['exception'] is None

    # ---- (b2) bitwise ------------------------------------------------------------------------------------------------
    occ = locate_bytes(main_bytes, gen_bytes)
    appx_bytes = rd(APPX_REL)
    blk = appendix_block(lines)
    pblk = appendix_block(plines)
    bit = {'main_tex_occurrences': occ, 'n_occurrences': len(occ),
           'equals_committed_appendix': gen_bytes == appx_bytes, 'committed_appendix_sha256': _sha_bytes(appx_bytes),
           'equals_verifier_regeneration': ver['tempdir_output_sha256'] == _sha_bytes(gen_bytes),
           'block_head_lines': [blk[0] + 1, blk[1]], 'block_prev_lines': [pblk[0] + 1, pblk[1]],
           'line_before_block': lines[occ[0]['start_line'] - 2] if occ else None,
           'line_after_block': lines[occ[0]['end_line']] if occ else None}
    if occ:
        o0 = occ[0]
        bit['generated_range_within_block'] = blk[0] + 1 <= o0['start_line'] and o0['end_line'] <= blk[1]
        bit['block_lines_outside_generated_range'] = [{'line': i + 1, 'text': lines[i]} for i in range(blk[0], blk[1])
                                                      if not (o0['start_line'] <= i + 1 <= o0['end_line'])]
    bitwise_ok = (len(occ) == 1 and occ[0]['starts_at_line_start'] and occ[0]['ends_at_line_end']
                  and bit['equals_committed_appendix'] and bit['equals_verifier_regeneration'])
    _log(f"[{tag}] bitwise: {len(occ)} occurrence(s) in main.tex {[(o['start_line'], o['end_line']) for o in occ]}; "
         f"= committed appendix {bit['equals_committed_appendix']}; = verifier regeneration {bit['equals_verifier_regeneration']}; "
         f"block {bit['block_head_lines']}; lines of the block outside the generated range "
         f"{[x['line'] for x in bit.get('block_lines_outside_generated_range', [])]}")

    # ---- (b3) entry check at HEAD and at e5322f8; the diffs ---------------------------------------------------------
    R = table_check(lines, ALL, COND, J, T)
    RP = table_check(plines, ALL, COND, J, T)
    checks['e21_61_rows_numbered_1_61'] = [r['no'] for r in R['E21']] == list(range(1, 62))
    checks['e21_map_one_to_one_onto_frozen_cells'] = len(set(R['idmap'].values())) == 61 == len(ALL) and set(R['idmap'].values()) == set(ALL)
    checks['e20_58_rows_numbered_1_58'] = [r['no'] for r in R['E20']] == list(range(1, 59))
    checks['mapping_problems_none'] = not R['mapping_problems']
    checks['e21_map_same_at_prev'] = RP['idmap'] == R['idmap']
    raw_new, raw_old = raw_tables(lines), raw_tables(plines)
    TD = diff_tables(raw_old, raw_new)
    BD = block_diff(plines[pblk[0]:pblk[1]], pblk[0], lines[blk[0]:blk[1]], blk[0], lines)
    cls_old, cls_new = line_classes(raw_old, plines), line_classes(raw_new, lines)
    for h in BD:
        h['class'] = classify_hunk(h, plines, lines, cls_old, cls_new)
    # generator output W180a -> W181a (the generator patch, seen through its output)
    w180a_gen = rd(INPUTS['W180A_GENERATED']).decode('utf-8').split('\n')
    gl = gen_bytes.decode('utf-8').split('\n')
    sm = difflib.SequenceMatcher(a=w180a_gen, b=gl, autojunk=False)
    gh = [{'op': op, 'old': w180a_gen[i1:i2], 'new': gl[j1:j2]} for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal']
    gvw = {'w180a_generated_sha256': inputs['W180A_GENERATED']['sha256'], 'hunks': gh,
           'same_as_block_diff': [(x['old'], x['new']) for x in gh] == [(h['old'], h['new']) for h in BD]}
    n_entries = sum(s['entries'] for s in R['summary'].values())
    n_mm = sum(s['MISMATCH'] for s in R['summary'].values())
    n_dis = sum(s['NOT_FROZEN_DISAGREES'] for s in R['summary'].values())
    for t, s in R['summary'].items():
        _log(f'[{tag}] {head[:7]} {t}: rows {s["rows"]} entries {s["entries"]} MATCH {s["MATCH"]} MISMATCH {s["MISMATCH"]} '
             f'not-a-frozen-field agrees {s["NOT_FROZEN_AGREES"]} disagrees {s["NOT_FROZEN_DISAGREES"]}')
    for b in R['bad']:
        _log(f'[{tag}] {head[:7]} NON-AGREEING {b}')
    for b in RP['bad']:
        _log(f'[{tag}] {PREV_REV} non-agreeing: {b}')
    _log(f"[{tag}] table cells compared {TD['n_cells_compared']}, changed {len(TD['cells'])}")
    for c in TD['cells']:
        _log(f"[{tag}] DIFF cell {c['table']} row {c['row']} col {c['col']} ({c['column']}): {c['old']!r} -> {c['new']!r}")
    for part in ('row_set_changes', 'captions', 'notes', 'group_headers', 'other_lines'):
        for c in TD[part]:
            _log(f"[{tag}] DIFF {part} {c.get('table')} l. {c.get('lines_old')} -> {c.get('lines_new')}: {c.get('token_changes')}")
    for h in BD:
        _log(f"[{tag}] BLOCK HUNK {h['class']} [{h['section'][:60]}] old {h['lines_old']} new {h['lines_new']}: "
             f"{str(h['token_changes'])[:500]}")
    _log(f"[{tag}] generator output W180a -> W181a: {len(gh)} hunk(s); same as the block diff {gvw['same_as_block_diff']}")

    # ---- (c) mutation test (in memory only) --------------------------------------------------------------------------
    base_bad = {(b['table'], str(b['row']), b['key']) for b in R['bad']}
    base_cells_vs_prev = {(c['table'], c['row'], c['col']) for c in TD['cells']}
    plus01 = lambda s: (('$-$' if s.startswith('$-$') else '') + f"{abs(parse_num(s)) + 0.1:.{dec_places(s.replace('$-$', ''))}f}")  # noqa: E731
    first_band_row = next(k for k, r in raw_new['E.21']['rows'].items() if r['cells'][8] != '---' and k not in ('16',))
    cases = [
        ('M1', 'E.20', '3', 3, plus01, 'd_gross_k', 3),
        ('M2', 'E.21', first_band_row, 8, plus01, 'band_k', int(first_band_row)),
        ('M3', 'E.21', '16', 4, lambda s: 'cert., osc.$^{s}$', 'status', 16),
        ('M4', 'E.21', '16', 13, lambda s: 'record', 'rule', 16),
        ('M5', 'E.22', '19', 5, _bump_last_digit, 'phase_b', 19),
        ('M6', 'E.23', 'Passive NRF|cold', 3, lambda s: str(int(s) + 1), 'nrf_n', 'passive/cold'),
        ('M7', 'E.24', '42', 9, _bump_last_digit, 'pf_primal_last', 42),
        ('M8', 'E.21', '40', 13, lambda s: 'run', 'rule', 40),
        ('M9', 'E.20', '34', 9, lambda s: 'w.r.', 'verdict_net', 34),
    ]
    MT = []
    for mid, t, row, col, fn, key, rid in cases:
        L2, old, new = mutate_cell(lines, raw_new, t, row, col, fn)
        R2 = table_check(L2, ALL, COND, J, T)
        bad2 = {(b['table'], str(b['row']), b['key']) for b in R2['bad']}
        caught_f = (bad2 ^ base_bad) == {(t, str(rid), key)}
        D_self = diff_tables(raw_new, raw_tables(L2))
        D_prev = diff_tables(raw_old, raw_tables(L2))
        cells_self = {(c['table'], c['row'], c['col']) for c in D_self['cells']}
        cells_prev = {(c['table'], c['row'], c['col']) for c in D_prev['cells']}
        other_parts = sum(len(D_self[p]) for p in ('row_set_changes', 'captions', 'notes', 'group_headers', 'other_lines'))
        caught_d = cells_self == {(t, row, col)} and other_parts == 0 and (cells_prev ^ base_cells_vs_prev) == {(t, row, col)}
        MT.append({'id': mid, 'edit': f'{t} row {row} col {col} ({COLS[t][col]}): {old!r} -> {new!r}',
                   'caught_frozen_check': caught_f, 'caught_diff': caught_d,
                   'detail': {'frozen_check_new_non_agreeing': sorted(map(list, bad2 - base_bad)),
                              'frozen_check_vanished_non_agreeing': sorted(map(list, base_bad - bad2)),
                              'diff_vs_head_cells': sorted(map(list, cells_self)),
                              'diff_vs_prev_symmetric_change': sorted(map(list, cells_prev ^ base_cells_vs_prev))}})
    NA = 'not applicable (not a frozen entry)'
    text_cases = [('M10', 'E.23', 'caption', 'after the consistency pass.', 'before the consistency pass.', None),
                  ('M11', 'E.21', 'caption', 'within its recorded cycles', 'within its cycles', None),
                  ('M12', 'E.22', 'notes', '51.3~k\\euro{}', '51.4~k\\euro{}', ('E.22', 'note', 'note_superseded')),
                  ('M13', 'E.21', 'notes', 'are re-decided from the records of earlier runs',
                   'continue earlier runs', None)]
    for mid, t, part, needle, repl, fexp in text_cases:
        L2, old, new = mutate_text(lines, raw_new, t, part, needle, repl)
        R2 = table_check(L2, ALL, COND, J, T)
        bad2 = {(b['table'], str(b['row']), b['key']) for b in R2['bad']}
        D_self = diff_tables(raw_new, raw_tables(L2))
        pk_ = 'captions' if part == 'caption' else part
        caught_d = (len(D_self[pk_]) == 1 and not D_self['cells']
                    and sum(len(D_self[p]) for p in ('row_set_changes', 'captions', 'notes', 'group_headers', 'other_lines')) == 1)
        caught_f = ((bad2 ^ base_bad) == {fexp}) if fexp else NA
        if not fexp and bad2 != base_bad:
            caught_f = False   # an edit outside the frozen entries must not move the frozen check
        MT.append({'id': mid, 'edit': f'{t} {part}: {old!r} -> {new!r}', 'caught_frozen_check': caught_f,
                   'caught_diff': caught_d, 'detail': {'diff_vs_head': D_self[pk_], 'frozen_check_change': sorted(map(list, bad2 ^ base_bad))}})
    mt_ok = all(m['caught_diff'] is True and m['caught_frozen_check'] in (True, NA) for m in MT)
    checks['mutation_test_every_edit_caught'] = mt_ok
    checks['mutation_test_at_least_five_single_entry_edits'] = sum(1 for m in MT if m['caught_frozen_check'] is True) >= 5
    for m in MT:
        _log(f"[{tag}] MUTATION {m['id']} {m['edit']}: frozen check {m['caught_frozen_check']}, diff {m['caught_diff']}")

    # ---- predictions ---------------------------------------------------------------------------------------------------
    DC = ident['diff_comparison']
    a1 = (main_bytes == rc and letter_bytes == l8 and main_sha == MAIN_SHA and letter_sha == LETTER_SHA
          and _nlines(main_bytes) == MAIN_LINES and _nlines(letter_bytes) == LETTER_LINES and head == CLONE_HEAD)
    a2 = (diff_files == EXPECTED_DIFF_FILES
          and all(DC[f]['n_hunks_clone'] == DC[f]['n_hunks_expert'] == EXPECTED_HUNKS[f] for f in EXPECTED_DIFF_FILES)
          and all(DC[f]['every_hunk_pm_equal'] and DC[f]['concatenated_pm_lines_equal_in_order'] for f in EXPECTED_DIFF_FILES))
    noncomment = [h for h in BD if h['class'] != ['comment']]
    comment_only = [h for h in BD if h['class'] == ['comment']]
    e4 = [h for h in noncomment if h['class'] == ['prose'] and 'Static Arrangements' in h['section']]   # E.4 subsection
    e21n = [h for h in noncomment if h['class'] == ['E.21 note']]
    only_two = len(noncomment) == 2 and len(e21n) == 1 and len(e4) == 1
    preds = [
        {'prediction': '(a1) main.tex byte-identical to main_e5322f8_round7.tex and the letter to v8 (HEAD cd57efc)',
         'outcome': 'HELD' if a1 else 'FAILED',
         'detail': (f"HEAD {head}; main.tex sha256 {main_sha}, {_nlines(main_bytes)} lines, = round7 copy {main_bytes == rc}; "
                    f"letter sha256 {letter_sha}, {_nlines(letter_bytes)} lines, = v8 {letter_bytes == l8}")},
        {'prediction': '(a2) clone diff e5322f8 -> HEAD = main_e5322f8_round7.diff (6 hunks) + letter_v7_v8.diff (2 hunks), +/- lines in order',
         'outcome': 'HELD' if a2 else 'FAILED',
         'detail': (f"files {diff_files}; " + '; '.join(
             f"{f}: hunks {DC[f]['n_hunks_clone']}/{DC[f]['n_hunks_expert']}, +/- per hunk {DC[f]['every_hunk_pm_equal']}, "
             f"in order {DC[f]['concatenated_pm_lines_equal_in_order']}, ranges {DC[f]['every_hunk_range_equal']}, "
             f"context {DC[f]['every_hunk_context_equal']}" for f in EXPECTED_DIFF_FILES))},
        {'prediction': '(b1) Appendix E regenerated bitwise from the patched gen_appE.py', 'outcome': 'HELD' if bitwise_ok else 'FAILED',
         'detail': (f"{len(occ)} occurrence(s) in main.tex at {head[:7]}, lines {[(o['start_line'], o['end_line']) for o in occ]}, "
                    f"line-aligned {[(o['starts_at_line_start'], o['ends_at_line_end']) for o in occ]}; = committed "
                    f"appendixE_evaluation_record.tex {bit['equals_committed_appendix']}; = verifier regeneration "
                    f"{bit['equals_verifier_regeneration']}")},
        {'prediction': '(b2) verify_appE.py: 33 PASS, 0 FAIL',
         'outcome': 'HELD' if (ver['n_pass'] == 33 and ver['n_fail'] == 0 and ver['exception'] is None) else 'FAILED',
         'detail': f"{ver['n_pass']} PASS, {ver['n_fail']} FAIL, exception {ver['exception']}"},
        {'prediction': '(b3) 0 MISMATCH in 1,623 entries', 'outcome': 'HELD' if (n_entries == 1623 and n_mm == 0 and n_dis == 0 and not R['mapping_problems']) else 'FAILED',
         'detail': (f"{n_entries} entries; {sum(s['MATCH'] for s in R['summary'].values())} MATCH on frozen fields, {n_mm} MISMATCH; "
                    f"{sum(s['NOT_FROZEN_AGREES'] for s in R['summary'].values())} not-a-frozen-field entries agree, {n_dis} disagree; "
                    f"mapping problems {len(R['mapping_problems'])}")},
        {'prediction': '(b4) 0 table cells changed e5322f8 -> HEAD',
         'outcome': 'HELD' if (not TD['cells'] and not TD['row_set_changes']) else 'FAILED',
         'detail': (f"{len(TD['cells'])} of {TD['n_cells_compared']} cells changed; row-set changes {len(TD['row_set_changes'])}; "
                    f"outside the cells: captions {len(TD['captions'])}, notes {len(TD['notes'])}, group headers "
                    f"{len(TD['group_headers'])}, other table lines {len(TD['other_lines'])}")},
        {'prediction': '(b5, task context) only the E.4 paragraph and the E.21 note differ in the appendix block',
         'outcome': ('HELD' if (only_two and not comment_only) else
                     'HELD for every non-comment line; FAILED as worded (the block also differs in a comment)' if only_two
                     else 'FAILED'),
         'detail': (f"{len(BD)} block hunk(s): " + '; '.join(f"{h['class']} in {h['section'][:60]!r} new l. {h['lines_new']}" for h in BD))},
    ]
    guards, pk, guards_ok = guards_state()
    failed = [k for k, v in checks.items() if v is not True]
    code = 0 if (not failed and guards_ok) else 3
    out = {'stage': 'P5.15 Addendum 80 W181a -- identity and Appendix E at Overleaf cd57efc (zero-solve; short check)',
           'script': SCRIPT_REL, 'script_sha256': _sha(SCRIPT_REL), 'git_head': _git('rev-parse', 'HEAD'),
           'utc': datetime.now(timezone.utc).isoformat(), 'interpreter': sys.executable, 'clone_head': head,
           'objective_convention': 'Q = gross_operational_cost, settlement excluded; net = Q - terminal salvage credit; value = Q(0) - Q(x); F = Q + I',
           'rounding_rule': 'MATCH iff |printed - value| <= half a unit of the last printed digit (+1e-9); k EUR = EUR / 1000',
           'entry_check_provenance': ('sections 0-5 and 7-11 copied verbatim from p515_s53_w180a_appendixE_check.py '
                                      '(W179a entry check + W180a row-16 convention, Addendum 79 ruling 1)'),
           'inputs': inputs, 'identity': ident,
           'generator': {'script': GEN_REL, 'output': gen_out_rel, 'sha256': _sha_bytes(gen_bytes), 'n_lines': gen_bytes.count(b'\n'),
                         'stdout': gen_stdout, 'argv_target': gen_out_rel},
           'verifier': {k: v for k, v in ver.items()}, 'bitwise': bit,
           'table_check_head': {'summary': R['summary'], 'bad': R['bad'], 'mapping_problems': R['mapping_problems'],
                                'E20': R['E20'], 'E21': R['E21'], 'E22': R['E22'], 'E22_note': R['E22_note'],
                                'E23': R['E23'], 'E23_note': R['E23_note'], 'E24': R['E24'], 'n_entries': n_entries},
           'table_check_prev': {'summary': RP['summary'], 'bad': RP['bad'], 'mapping_problems': RP['mapping_problems']},
           'e21_map': {str(k): v for k, v in sorted(R['idmap'].items())},
           'table_diff': TD, 'block_diff': BD, 'generator_vs_w180a': gvw,
           'mutation_test': {'cases': MT, 'all_caught': mt_ok}, 'predictions': preds,
           'checks': checks, 'failed': failed, 'guards': guards, 'pickle_guard': pk, 'guards_ok': guards_ok,
           'exit_code': code, 'wall_s': time.time() - t0}

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, set)):
            return [_clean(v) for v in o]
        return o
    out = _clean(out)
    written = {gen_out_rel: _sha_bytes(gen_bytes),
               os.path.join(out_dir, OUT_NAMES['verify']): _sha(os.path.join(out_dir, OUT_NAMES['verify']))}

    def wr(name, data):
        rel = os.path.join(out_dir, OUT_NAMES[name])
        full = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        with open(full, 'xb') as h:
            h.write(data.encode('utf-8'))
        with open(full, 'rb') as h:
            written[rel] = _sha_bytes(h.read())
    wr('json', GRIO.dumps(out, indent=1, sort_keys=True) + '\n')
    wr('md', md_summary(out))
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for v in inputs.values():
        man[v['path']] = v['sha256']
    man[main_rel] = main_sha
    man[letter_rel] = letter_sha
    man[f'git:{CLONE}@{PREV_REV_FULL}:main.tex'] = prev_sha
    man[f'git:{CLONE}@{PREV_REV_FULL}:{LETTER_REL_CLONE}'] = _sha_bytes(prev_letter)
    man[f'git:{CLONE} diff {PREV_REV_FULL} {head}'] = ident['clone_diff_sha256']
    wr('man', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    for p in preds:
        _log(f"[{tag}] PREDICTION {p['prediction']}: {p['outcome']} ({p['detail']})")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    _log(f"[{tag}] wrote {', '.join(f'{k} {v[:8]}' for k, v in written.items())}")
    _log(f"[{tag}] guard counts {guards['counts']} verify(0) {guards['verify_0_failures']}; pickle {pk}; exit {code}; "
         f"wall {time.time() - t0:.1f} s")
    GUARD.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(code)


if __name__ == '__main__':
    main()
