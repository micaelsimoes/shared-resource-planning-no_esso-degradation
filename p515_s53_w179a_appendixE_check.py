"""P5.15 Addendum 78, Planner task W179a -- ITEM (a) OF THE W179 ORDER: APPENDIX E ("Evaluation Record") OF main.tex AT
OVERLEAF 48099ab (main.tex sha256 28008ab0..., byte-identical to manuscript_review/main_e926fb7_appE.tex) AGAINST THE
FROZEN TABLES data/SRP1/Results/P515S53/w160_step6_frozen/frozen_step6_tables_v1_590088fe.json. ZERO SOLVES, NO MODEL
LOADS, READ ONLY.

INDEPENDENT OF manuscript_review/gen_appE.py AND verify_appE.py (neither is imported or read): the LaTeX tables of
Appendix E are parsed from main.tex, and every row is mapped to its frozen cell / claim by this script's own decoding:
  Table E.21 (tab:appE_evaluations): each row's plan text ("node: MVA/MWh", ";"-separated, "each" for several nodes at
    one size), year and condition are decoded and matched against the frozen cells' candidate_canonical and the
    condition each cell ran under, read from the cell's own stage spec (flexibility-price multiplier; ageing model
    variant and SoH floor; for the ageing family also the spec series: production / ext v3 arm / A64 floor cell). Where
    a plan+condition pair matches more than one frozen cell, the row's printed marker decides ($^{s}$ superseded); the
    disambiguation used is recorded per row. The row number -> cell map so built is then used for every
    cross-reference by number (E.20 "Eval.", E.22 "Eval.", E.24 "Eval.", the prose).
  Table E.20 (tab:appE_claims): each row's printed evaluation pair (reference / other) is mapped through the E.21 map to
    a (ref_cell, other_cell) pair, and the claim is the frozen T1 claim (tables.claims) or T10 claim (tables.a64.rows)
    with that pair; the group's sign convention and the description (plan, year, price, calibration) are checked
    against the two cells. The 60 -> 55 + 3 mapping is checked as a set identity.
  Tables E.22-E.24 likewise (tables.phase_b / claims; tables.benchmark; tables.dead_zone).
Every entry is rated MATCH / MISMATCH at the printed rounding (|printed - value| <= half a unit of the last printed
digit; k EUR = EUR / 1000), or NOT_FROZEN_AGREES / NOT_FROZEN_DISAGREES where the entry is not a frozen field (E.21
Condition: the cell's stage spec; E.21 Rule: derived from the frozen certifying_spec and W142 item 1; E.20 Comparison
text: decoded against the two cells), its source named. The E.21 map is checked to be one-to-one onto the 61 frozen
cells and the E.20 map onto 55 distinct T1 claims + 3 distinct T10 claims; a mutation test of the comparisons (scratch,
not part of this script) was run before the real run. Then every number and count of the Appendix E
prose (intro, E.1, E.3, E.4, E.5, E.6) is rated MATCH / MISMATCH / DERIVED (with the arithmetic) against the frozen
JSON or a named record, and the nine interpretations (i)-(ix) of Addendum 78 are confirmed or corrected with evidence.

OBJECTIVE CONVENTION. Q = gross_operational_cost, settlement excluded (the frozen primary); net = Q - terminal salvage
credit; value = Q(0) - Q(x); F = Q + I. EUR in the JSON, k EUR in the manuscript.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import and verified at exactly 0 at the end;
pickle.load / pickle.loads blocked for the whole run (counters verified 0). git reads only (rev-parse, status, ls-files,
log), also in the manuscript clone, which is never pulled or modified. Refuses to run beside another copy of itself.

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir data/SRP1/Results/P515S53/w179a_appendixE && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w179a_appendixE_check.py \\
        > data/SRP1/Results/P515S53/w179a_appendixE/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w179a_appendixE/w179a_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w179a_appendixE/w179a_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w179a_appendixE_check.py --post-run
Write-once: every output is opened 'x' in a directory that holds nothing but its launch log.
Exit: 0 = written, every integrity check holds and the guards are at 0 (MISMATCHES and corrections are FINDINGS and
never change the exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard fault (nothing
written).
"""
import argparse
import hashlib
import json
import os
import pickle
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W179a Appendix E check at Overleaf 48099ab (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W179a: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W179a: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer that refuses string flags)

pickle.load, pickle.loads = _blocked_load, _blocked_loads

SCRIPT_REL = os.path.basename(__file__)
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(S53, 'w179a_appendixE')
OUT_NAMES = {'json': 'w179a_appendixE_check.json', 'md': 'w179a_appendixE_check.md', 'man': 'manifest_sha256.json',
             'log': 'launch.log', 'typing_json': 'w179a_bool_typing_test.json', 'typing_log': 'w179a_bool_typing_test.log',
             'post': 'manifest_post_run_sha256.json'}
LAUNCH_LOG_RX = re.compile(r'launch(_attempt\d+)?\.log')

CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
CLONE_HEAD = '48099abb09db45a1668c1ef8441a3b7c658913a8'
MAIN_SHA = '28008ab0d79b477c4e8d86a0c3a4d1acf9c95ed9c4921f4757f1600cda1c60af'
MAIN_LINES = 2717
FZ_REL = os.path.join(S53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FZ_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'

INPUTS = {
    'FROZEN_JSON': FZ_REL,
    'REVIEW_COPY': os.path.join('manuscript_review', 'main_e926fb7_appE.tex'),
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
    'W153_STATS': os.path.join(S53, 'w153_step5_rows', 'w153_certification_statistics.json'),
    'W163_CHECK': os.path.join(S53, 'w163_paragraphs_v4_check', 'w163_paragraphs_v4_figure_check.json'),
    'W168_X0': os.path.join(S53, 'w168_x0_v6_replay', 'w168_x0_v6_replay_decision.json'),
    'W173_COUNTS': os.path.join(S53, 'w173_search_counts', 'w173_search_counts.json'),
    'CODE_UB': 'uncoordinated_benchmark.py',
    'CODE_W116': 'p515_s53_w116_benchmark_nrf.py',
    'CODE_W93': 'p515_s53_w93_uncoordinated_benchmark.py',
    'CODE_V5_HOOKS': 'p515_s53_w139_resettle_v5_hooks.py',
    'CODE_W153': 'p515_s53_w153_step5_zero_solve_rows.py',
    'STEP6_PACKAGE': 'P5_15_STEP6_PACKAGE.md',
    'A53_REPORT': 'P5_15_ADDENDUM53_SRP1_CONTINUATION_REPORT.md',
    'CODE_SC5': 'settling_criterion_v5.py',
    'BENCH_REPORT_V3': os.path.join(S53, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json'),
}
# code / record needles: (input key, needle) -- located, line number recorded; each must occur
NEEDLES = {
    'nrf_row_one_per_scenario_period': ('CODE_UB', "# rule \"import only\" at the interface. One ROW per (market scenario, operation scenario, period) of every DSO arm"),
    'nrf_violation_entry_per_row': ('CODE_UB', "hard.append({'row': con_data.name, 'kind': 'no_reverse_flow', 'pg_adn_pu': float(pe.value(con_data.body)),"),
    'nrf_row_deactivated_in_reevaluation': ('CODE_UB', '# counted by the structural check. In the consistency re-evaluation the row is deactivated like the voltage and'),
    'w116_violations_counted_at_reevaluation': ('CODE_W116', "nrf_violations = [h for b in reeval['blocks'].values() for h in b['violations']['hard']"),
    'w116_pass_effect_is_phase_C_minus_A': ('CODE_W116', "'effect_on_q_eur': evaluation_c['gross_operational_cost'] - evaluation['gross_operational_cost'],"),
    'w116_phase_C_dso_resolve': ('CODE_W116', "UB.solve_dso_models(planning, arm_out['models']['dso'], phase='sequential_pass:dso', record_callback=sink)"),
    'w116_warm_from_certified_models': ('CODE_W116', 'warm = UB.extract_model_values(planning, certified) if start != UB.START_COLD else None'),
    'w116_certified_is_coordinated': ('CODE_W116', "certified = BENCH._load_pickle_verified(BENCH.COORDINATED['certified_models'])"),
    'w93_coordinated_is_settled': ('CODE_W93', 'COORDINATED = SETTLED'),
    'w93_settled_eval_key': ('CODE_W93', "'eval_key': 'd110bd1a5977df1e811a1b3963afc565de6daa65f76dcc096ead3fdc70b08546',"),
    'acceptable_clean_is_block_exit': ('CODE_V5_HOOKS', "st.cur['acceptable_clean_blocks'] = sorted(k for k, v in entries.items()"),
    'acceptable_clean_reason': ('CODE_V5_HOOKS', "if v['reason'] == 'acceptable_primary_within_factor')"),
    'w153_events_sum_blocks': ('CODE_W153', "'acceptable_clean_block_events_after_N': sum(len(e['blocks']) for e in acc_after),"),
    'step6_package_continued_cell_numbering': ('STEP6_PACKAGE', '| 22 | cell 1 (`b_2a0ba8b2`) re-run bitwise and certifies at 173 (Planner, A59) | **held** |'),
    'a53_three_references_replayed': ('A53_REPORT', '**All three references replayed bitwise to their old certification cycle** (132, 112, 87)'),
    'a53_old_vs_new_cert_x0': ('A53_REPORT', '| x0 | 132 | **181** |'),
    'a53_old_vs_new_cert_unit': ('A53_REPORT', '| unit | 112 | **172** |'),
    'a53_c_star_none': ('A53_REPORT', '| C\\* | 87 | **none (cap 187)** |'),
    'sc5_acceptable_primary_within_factor': ('CODE_SC5', "'acceptable_primary_within_factor': 'clean: \"Solved To Acceptable Level\" on the primary attempt, every metric <= 10x',"),
    'sc5_one_block_final_exit': ('CODE_SC5', '"""The v5 clean classification of ONE block\'s final accepted exit. Pure.'),
    'w116_arm_cost_is_phase_C': ('CODE_W116', "arm_cost_source = 'phase_C_sequential_pass'"),
    'w116_curtailment_rows_phase_A': ('CODE_W116', "'curtailment_rows_are_phase_A': True}"),
    'report_v3_warm_source_is_settled_x0': ('BENCH_REPORT_V3', 'campaign_s53_w101_srp1_cont_x0/evals/d110bd1a5977df1e_x0/certified_models.pkl'),
}

TAU_PRINTED = '4,539.07'
SUSPECT = '\u2014'


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
        exp_status = status_text(c)
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
        lab, why = expected_rule(c, COND[name], J['SPEC_V6'])
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
#  6. prose checks and interpretations
# ======================================================================================================================
def needle_lines(texts):
    out = {}
    for k, (key, needle) in NEEDLES.items():
        lines = texts[key].split('\n')
        hits = [i + 1 for i, ln in enumerate(lines) if needle in ln]
        out[k] = {'file': INPUTS[key], 'needle': needle, 'lines': hits, 'found': bool(hits)}
    return out


def w163_checks(W163):
    found = {}

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get('id'), str) and 'status' in o:
                found.setdefault(o['id'], o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(W163)
    return found


def prose(lines, idmap, ALL, T, J, e20, e21_out, D):
    """Every number / count of the Appendix E prose. D: derived quantities."""
    P = []

    def add(pid, sect, written, status, value, source, line=None):
        P.append({'id': pid, 'section': sect, 'written': written, 'status': status, 'value': value, 'source': source,
                  'line': line})

    def ln_of(frag, start=2380, end=2702):
        hits = [i + 1 for i in range(start - 1, end) if frag in lines[i]]
        return hits[0] if hits else None
    TAU = D['TAU']
    # intro
    add('I1', 'intro', 'tau = 4,539.07 EUR', 'MATCH' if f'{TAU:,.2f}' == TAU_PRINTED else 'MISMATCH', TAU,
        f'{FZ_REL} constants.TAU', ln_of('\\tau = 4{,}539.07'))
    add('I2', 'intro', 'amounts rounded to 0.1 k EUR, ratios to two decimals', 'MATCH' if D['rounding_ok'] else 'MISMATCH',
        D['rounding_detail'], 'decimal places of every printed k EUR amount and every ratio column of E.20-E.24 (parsed here); '
        'E.24 Resid. (residual over its tolerance) is a ratio column', ln_of('Amounts are rounded to 0.1'))
    # E.1
    l1 = ln_of('lists the 58 distinct comparisons')
    add('E1.1', 'E.1', '58 distinct comparisons', 'MATCH' if D['n_e20'] == 58 else 'MISMATCH', D['n_e20'],
        'E.20 rows parsed; each mapped to a distinct frozen claim', l1)
    add('E1.2', 'E.1', 'Fifty-five reported both gross and net', 'MATCH' if D['n_gross_net'] == 55 else 'MISMATCH',
        D['n_gross_net'], 'E.20 rows with a T1 claim (net printed)', l1)
    add('E1.3', 'E.1', 'the remaining three (m = 1.75 unit; two at the 0.50 floor) gross only',
        'MATCH' if D['t10_rows'] == {22: 'H:m1.75:value_minus_I', 52: 'E:soh050:value_minus_I', 58: 'E:soh050:delta_value_vs_070'} else 'MISMATCH',
        D['t10_rows'], 'tables.a64.rows (T10); no d_net field', l1)
    add('E1.4', 'E.1', '1.75 times the base flexibility price; the 0.50 end-of-life floor',
        'MATCH' if D['cond_175'] and D['cond_050'] else 'MISMATCH', {'h_unit_m175 m': D['cond_175'], 'e_soh050 floor 0.5': D['cond_050']},
        f"{INPUTS['SPEC_A64_V1']} cells", l1)
    add('E1.5', 'E.1', 'determinate if |Delta| >= max{3b, 2 tau}, b the larger band', 'DERIVED' if D['thr_formula_ok'] else 'MISMATCH',
        D['thr_formula'], 'max(3 x max(band_width ref, other), 2 TAU) recomputed for every certified-certified claim', l1)
    add('E1.6', 'E.1', 'uncertified: |Delta| > 3 x the largest consensus gap and settling slack involved, gross and gap-corrected',
        'DERIVED' if D['bar_formula_ok'] else 'MISMATCH', D['bar_formula'],
        '3 x max(|gap|, |slack|) over the uncertified cells involved recomputed for every uncertified-form claim; verdict = det. iff gross x > 1 and Q_cc x > 1', l1)
    add('E1.7', 'E.1', 'two certificates on a non-clean turning point: evaluations 4 and 38',
        'MATCH' if D['flagged_evals'] == [4, 38] else 'MISMATCH', D['flagged_evals'],
        'cells with flag_non_clean_turning_point true, numbered by the E.21 map', l1)
    add('E1.8', 'E.1', 'none changes verdict (uncertified form of the comparisons using 4 and 38)',
        'MATCH' if D['flag_verdicts_unchanged'] else 'MISMATCH', D['flag_verdicts'],
        'claims.uncertified_form_beside_A64_ruling4 verdict vs gross_verdict', l1)
    # E.20 note
    ln = D['e20_note_line']
    for pid, w, k, fld in (('N1', 'row 3 bar 70.7 k EUR (2.94x), determinate', 'B:n7_4h_e3', None),
                           ('N2', 'row 26 bar 15.9 k EUR (11.12x), determinate', 'J:e4_value_minus_I', None)):
        u = D['beside'][k]
        printed = re.search(r'([0-9.]+) k EUR \(([0-9.]+)x\)', w)
        okb, vb = at_print(printed.group(1), u['bar'], 1000.0)
        okm, vm = at_print(printed.group(2), u['multiple_Q'])
        row_ok = D['row_of_claim'].get(k) == int(re.search(r'row (\d+)', w).group(1))
        add(pid, 'E.20 note', w, 'MATCH' if okb and okm and u['verdict'] == 'determinate' and row_ok else 'MISMATCH',
            {'bar_k': vb, 'x': vm, 'verdict': u['verdict'], 'row_maps_to_claim': row_ok}, f'claims[{k}].uncertified_form_beside_A64_ruling4', ln)
    add('N3', 'E.20 note', 'rows 30 and 31 already use the uncertified form', 'MATCH' if D['rows_30_31_unc'] else 'MISMATCH',
        D['rows_30_31_rule'], 'claims gross_rule of J:e3_to_e4, J:e4_to_e5', ln)
    add('N4', 'E.20 note', 'evaluation 56 reproduces evaluation 2 bitwise (row 48 repeats row 1 with the sign of value - I)',
        'MATCH' if D['row48_vs_row1'] else 'MISMATCH', D['row48_detail'],
        'at_or_above_0_95_tau.twins.evidence.e_c2_calfade (reproduced through 172) and claims d_gross', ln)
    add('N5', 'E.20 note', 'row 58 against evaluation 61 (bitwise through 172) gives the same figures',
        'MATCH' if D['row58_beside'] else 'MISMATCH', D['row58_beside_detail'],
        'tables.a64.rows E:soh050:delta_value_vs_g070_REPORT_BESIDE vs E:soh050:delta_value_vs_070; twins.evidence.g070_neutrality', ln)
    add('N6', 'E.20 note', '^n: energy capacity at node 7 differs by two lattice steps (row 45)',
        'DERIVED' if D['row45_steps'] == 2 else 'MISMATCH', D['row45_steps'],
        '|3.5 - 2.5| MWh / 0.5 MWh step (ref:5ca4f86c vs l_df1a5525 candidate_canonical)', ln)
    add('N7', 'E.20 note', 'rows 17-20 each recorded once gross and once net', 'MATCH' if D['g_rows_ok'] else 'MISMATCH',
        D['g_rows'], 'claims G:*:gross / G:*:net pairs', ln)
    # E.2
    l2 = ln_of('lists the 61 recourse evaluations')
    add('E2.1', 'E.2', '61 recourse evaluations', 'MATCH' if D['n_e21'] == 61 and D['n_t2'] == 61 else 'MISMATCH',
        [D['n_e21'], D['n_t2']], 'E.21 rows parsed and mapped one-to-one onto the 49 + 12 frozen cells', l2)
    add('E2.2', 'E.21 note', 'evaluations 1, 2, 40 and 41 continue earlier runs; first residual passes not tabulated',
        'MATCH' if D['refs_k0_none'] else 'MISMATCH', D['refs_k0'], 'cells ref:* k0_run None', D['e21_note_line'])
    add('E2.3', 'E.21 note', 'for evaluation 1, k0 = 123 (Fig. res_q_cycle)', 'MATCH' if D['x0_k0'] == 123 else 'MISMATCH',
        D['x0_k0'], f"{INPUTS['W168_X0']} decision_primary.N / k0 (not a frozen field)", D['e21_note_line'])
    add('E2.4', 'E.21 note', 'evaluation 16 superseded: rested on a cycle whose local solve did not end optimal; re-run as 15',
        'MATCH' if D['row16_ok'] else 'MISMATCH', D['row16'], 'cells_appended_w160.pb_y2025_n5 superseded; T5 note; W142 item 1 (v6 not decided, 1 veto)',
        D['e21_note_line'])
    # E.3
    l3 = ln_of('evaluated all fourteen admissible unit neighbours')
    add('E3.1', 'E.3', 'all fourteen admissible unit neighbours evaluated by the search', 'MATCH' if D['w173_14'] else 'MISMATCH',
        D['w173_x0'], f"{INPUTS['W173_COUNTS']} item2.x0 (b_box, a_final_poll, certificate_record) -- not a frozen field", l3)
    add('E3.2', 'E.3', 'eight of them re-evaluated under the certification rule', 'MATCH' if D['n_e22'] == 8 and D['w173_8'] else 'MISMATCH',
        [D['n_e22'], D['w173_resettled']], 'E.22 rows; W173 item2.x0.c_resettled_in_frozen_tables', l3)
    add('E3.3', 'E.3', 'the smallest unit 0.25 MVA / 0.5 MWh, each node in 2025 or 2030, and two pairs of nodes',
        'MATCH' if D['e22_plans_ok'] else 'MISMATCH', D['e22_plans'], 'E.22 cells candidate_canonical', l3)
    add('E3.4', 'E.3', 'every one costs determinately more', 'MATCH' if D['e22_all_det'] else 'MISMATCH', D['e22_verdicts'],
        'phase_b v6_verdict / C claims gross_verdict', l3)
    add('E3.5', 'E.3', 'at its resolution sigma_Q', 'DERIVED' if D['w173_res_note'] else 'MISMATCH', D['w173_res'],
        'W173 item2.x0.rows search_resolution_eur (not a frozen field): the search accepted at max(bar_x + bar_inc, sigma_Q) '
        '(Algorithm, main.tex l. 512), sigma_Q = 18,449.66 EUR (sec. 3, l. 1316) the floor; the threshold equalled sigma_Q '
        'for the count given in the value and exceeded it for the rest', l3)
    # E.4
    l4 = ln_of('was solved from three starts')
    add('E4.1', 'E.4', 'each static arrangement solved from three starts', 'MATCH' if D['n_starts'] == [3, 3] else 'MISMATCH',
        D['n_starts'], 'benchmark.w160_additions.arms_in_full.*.n_starts', l4)
    add('E4.2', 'E.4', 'the best is reported in Table res_benchmark; the spread over the starts is its band',
        'MATCH' if D['band_is_spread'] else 'MISMATCH', D['band_spread'],
        'arms.*.multimodality_band == max - min of Q_by_start', l4)
    # E.5
    l5 = ln_of('lists the evaluations refused by the gap clause')
    add('E5.1', 'E.5', 'refused at m = 2 and evaluation 42: gap -8.8 to -9.4 k EUR', 'MATCH' if D['fp_gap_ok'] else 'MISMATCH',
        D['fp_gap'], 'dead_zone t_sum_at_cap of evals 39, 40, 41, 42, 45, 50, 51', l5)
    add('E5.2', 'E.5', 'split about 55 %, 14 %, 31 % among nodes 5, 7, 9', 'MATCH' if D['fp_share_ok'] else 'MISMATCH',
        D['fp_share'], 'dead_zone share_by_node of the rows that record it (39, 42, 45, 50, 51); evaluations 40 and 41 '
        'print --- (not recorded), so the shared fingerprint is the gap only for them', l5)
    add('E5.3', 'E.5', 'power-flow primal residual 0.68-0.70 of its tolerance, last cycle', 'MATCH' if D['fp_res_ok'] else 'MISMATCH',
        D['fp_res'], 'dead_zone pf_primal_last of the rows that record it (39, 42, 45, 50, 51); 40 and 41 not recorded', l5)
    add('E5.4', 'E.5', 'evaluation 42 reset by two transmission solver recoveries before the gap clause',
        'MATCH' if D['e42_ok'] else 'MISMATCH', D['e42'], 'dead_zone l_0ee93aca cause / lapse_events / entry', l5)
    add('E5.5', 'E.5', 'reference unit at 1.5x: gap -2.5 k EUR, same node split, residual ratio 0.027',
        'MATCH' if D['h15_ok'] else 'MISMATCH', D['h15'], 'dead_zone h_f9eae48f', l5)
    add('E5.6', 'E.5', '1.25 MVA / 5 MWh at base price: growth test after repeated TSO recoveries, small gap of opposite sign',
        'MATCH' if D['d366_ok'] else 'MISMATCH', D['d366'], 'dead_zone d_36686489; cells.d_36686489 non_clean_after_N', l5)
    add('E5.7', 'E.24 caption', 'refused evaluations and two others', 'MATCH' if D['e24_counts_ok'] else 'MISMATCH',
        D['e24_counts'], 'E.24 rows: gap_refused true vs others', ln_of('Evaluations refused by the gap clause, and two others'))
    # E.6 statistics
    l6 = ln_of('The 42 single-scenario evaluations of the certification campaign')
    add('E6.1', 'E.6', '42 = evaluations 3-11, 15, 22, 23, 26-31, 34-39, 42-59', 'MATCH' if D['set42_ok'] else 'MISMATCH',
        D['set42'], f"E.21 map vs {INPUTS['W153_STATS']} result.cells (42) and T2 minus the 3 B cells and 4 references", l6)
    add('E6.2', 'E.6', '32 certified, 24 oscillatory, 8 monotone', 'MATCH' if D['cert_split'] == [32, 24, 8] else 'MISMATCH',
        D['cert_split'], 'frozen cells status / branch over the 42 (= W153 totals)', l6)
    add('E6.3', 'E.6', '10 uncertified: 5 gap clause, 2 lapse reset, 3 growth test', 'MATCH' if D['unc_split'] == [10, 5, 2, 3] else 'MISMATCH',
        D['unc_split'], 'frozen cells cause_uncertified over the 42', l6)
    add('E6.4', 'E.6', 'median certification cycle 174', 'DERIVED' if D['med_kstar'] == 174 else 'MISMATCH', D['med_kstar'],
        'median of k_star over the 32 certified of the 42 (frozen cells)', l6)
    add('E6.5', 'E.6', 'k* - k0 median 65 (range 21-89)', 'DERIVED' if D['kk0'] == [65, 21, 89] else 'MISMATCH', D['kk0'],
        'k_star - k0_run over the 32 (frozen cells); W153 cycles_after_k0_k_star_minus_N beside', l6)
    add('E6.6', 'E.6', 'median range / tau at certification 0.86', 'DERIVED' if D['med_rng'] == 0.86 else 'MISMATCH', D['med_rng_full'],
        'median of range_over_tau over the 32', l6)
    add('E6.7', 'E.6', 'four evaluations added later (32, 33, 60, 61) all certified', 'MATCH' if D['added4_ok'] else 'MISMATCH',
        D['added4'], 'cells_appended_w160 A64 cells', l6)
    add('E6.8', 'E.6', 'Ten of the 46 certificates the tables use', 'DERIVED' if D['n46'] == 46 and D['n10'] == 10 and D['n46_ids_ok'] else 'MISMATCH',
        D['n46_detail'], 'at_or_above_0_95_tau: registry 49 certified - 1 superseded - 2 twins; n_counted 10', l6)
    add('E6.9', 'E.6', 'stop within 5 % of tau: evaluations 2, 3, 5, 10, 11, 12, 14, 17, 20, 23',
        'MATCH' if D['ten_ids'] == [2, 3, 5, 10, 11, 12, 14, 17, 20, 23] else 'MISMATCH', D['ten_ids'],
        'at_or_above_0_95_tau.counted_cells numbered by the E.21 map', l6)
    add('E6.10', 'E.6', 'evaluation 36 at 0.94 on the monotone branch, just below', 'MATCH' if D['e36_ok'] else 'MISMATCH', D['e36'],
        'at_or_above_0_95_tau.beside i_5a6a88b4; cells.i_5a6a88b4', l6)
    add('E6.11', 'E.6', 'the two evaluations continued past their certificates, 14 and 3, moved 0.74 tau and 0.78-0.88 tau',
        D['continued_status'], D['continued'], f"{INPUTS['W142_V6_FROM_RECORDS']} item3_movement_record; {INPUTS['W163_CHECK']} N25", l6)
    add('E6.12', 'E.6', '21 cycles in 10 evaluations not clean after the first residual pass, all in transmission blocks',
        'MATCH' if D['nc_ok'] else 'MISMATCH', D['nc'], 'frozen non_clean_after_N over the 42; W153 totals by family', l6)
    add('E6.13', 'E.6', '119 exits at the acceptable level within 10x, all in distribution blocks, counted clean',
        'MATCH' if D['acc_ok'] else 'MISMATCH', D['acc'], f"{INPUTS['W153_STATS']} totals (not a frozen-table field)", l6)
    # E.6 reproducibility
    l7 = ln_of('\\textbf{Reproducibility.}')
    W = D['w163']
    for pid, wid, w in (('R1', 'N47', 'multi-scenario x = 0 replayed its 72 recorded cycles bitwise'),
                        ('R2', 'N49', 'the three single-scenario reference continuations replayed bitwise to their certification cycles'),
                        ('R3', 'N56', 'Apple M2 Max'), ('R4', 'N57', '32 GiB of memory'), ('R5', 'N55', 'macOS 27.0'),
                        ('R6', 'N45', 'thread caps at one thread'), ('R7', 'N46', 'HSL built without OpenMP; MA97 serial'),
                        ('R8', 'N58', 'two distribution blocks (node 7 2025 Winter, node 5 2035 Winter) end some primary solves at Acceptable'),
                        ('R9', 'N59', 'within ten times the tight-tail tolerances'),
                        ('R10', 'N63', 'tail lowered each of the three references by 1.0-1.2 x 10^-6 relative, same sign'),
                        ('R11', 'N60', 'complementarity tolerance 10^-6 from 10^-4 (DN) and 5 x 10^-4 (TN)'),
                        ('R12', 'N33', '10^-5 p.u. slack on the renewable availability bound'),
                        ('R13', 'N64', 'lowers the objective by about 7.8 k EUR at first order'),
                        ('R14', 'N65', '1.2 x 10^-5 of it')):
        c = W.get(wid, {})
        st = c.get('status')
        status = 'MATCH' if (st == 'match' or (wid == 'N63' and D['tail_range_ok'])) else 'MISMATCH'
        note = c.get('note')
        if wid == 'N63':
            note = (f'{note}; W163 rated the earlier one-value form "1.1" approximate; the range 1.0-1.2 printed in E.6 is '
                    f'the three values at one decimal (checked here: {D["tail_range_ok"]})')
        if wid == 'N49':
            status = 'MISMATCH' if D['r2_wording_mismatch'] else status
            note = D['r2_note']
        add(pid, 'E.6 reproducibility', w, status, {'w163_status': st, 'w163_value': c.get('value'), 'note': note},
            f"{INPUTS['W163_CHECK']} check {wid} ({c.get('counterpart')})", l7)
    add('R15', 'E.6 reproducibility', 'three single-scenario reference continuations (count)',
        'MATCH' if D['three_refs_n'] == 3 else 'MISMATCH', D['three_refs'],
        'W163 N49 / Addendum 53 report: x0 (eval 1), unit (eval 2) and C* (not in Appendix E)', l7)
    # sec. 4.6 salvage sentence (main text, W179b scope) -- rated here at the W179a task's request
    l46 = next((i + 1 for i, ln in enumerate(lines) if 'Across the 55 comparisons of Table~\\ref{tab:appE_claims}' in ln), None)
    S = D['salvage_55']
    add('S1', 'sec. 4.6 (task list)', '55 comparisons reported both gross and net', 'MATCH' if S['n'] == 55 else 'MISMATCH',
        S['n'], 'E.20 rows mapped to T1 claims (net form present)', l46)
    add('S2', 'sec. 4.6 (task list)', 'the salvage credit changes no sign', 'DERIVED' if not S['sign_changes_rows'] else 'MISMATCH',
        S['sign_changes_rows'], 'sign(d_gross) vs sign(d_net) over the 55 (net from the G :net claim for rows 17-20)', l46)
    vc = S['verdict_changes']
    ok3 = (len(vc) == 1 and vc[0]['row'] == 34 and vc[0]['gross'] == [0.95, 'within the uncertified bar']
           and vc[0]['net'] == [1.22, 'determinate'] and S['row34_d_net_positive'])
    add('S3', 'sec. 4.6 (task list)', 'one verdict change: row 34, within the bar gross (0.95x), determinate net (1.22x), '
        "in the returned plan's favour", 'DERIVED' if ok3 else 'MISMATCH', vc,
        'gross_verdict vs net_verdict_W153 over the 55; d_net > 0 = the returned plan costs less (group (c) convention)', l46)
    return P


# ======================================================================================================================
#  7. post-run, guards, other copies, summary
# ======================================================================================================================
def guards_state():
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    g = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    pk = {'counts': dict(PICKLE_COUNTS), 'pickle_load_and_loads_blocked': blocked}
    pk['ok'] = blocked and PICKLE_COUNTS == {'load': 0, 'loads': 0}
    return g, pk, (not g['verify_0_failures']) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('man', 'json', 'md', 'typing_json', 'typing_log')]
    rels += [os.path.join(out_dir, f) for f in sorted(os.listdir(os.path.join(REPO, out_dir))) if LAUNCH_LOG_RX.fullmatch(f)]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W179a post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W179a post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}: " + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
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
    L = [f"# W179a — Appendix E against the frozen tables (Overleaf {o['clone_head'][:7]})", '',
         f"main.tex sha256 `{o['main_sha256']}`; frozen JSON `{FZ_SHA[:8]}`; script `{SCRIPT_REL}`. Zero solves; guards "
         f"{'0/0 verified' if o['guards_ok'] else 'FAILED'}.", '', '## Table entries', '',
         '| table | rows | entries | MATCH | MISMATCH | not a frozen field, agrees | not a frozen field, disagrees |',
         '|---|---|---|---|---|---|---|']
    for t, s in o['table_summary'].items():
        L.append(f"| {t} | {s['rows']} | {s['entries']} | {s['MATCH']} | {s['MISMATCH']} | {s['NOT_FROZEN_AGREES']} | "
                 f"{s['NOT_FROZEN_DISAGREES']} |")
    L += ['', '### Every MISMATCH', '']
    mm = o['mismatches']
    L += [f"- {m}" for m in mm] if mm else ['- none']
    L += ['', '## Prose', '', '| id | section | written | status | value | source |', '|---|---|---|---|---|---|']
    for p in o['prose']:
        L.append(f"| {p['id']} | {p['section']} (l. {p['line']}) | {_cell(p['written'], 90)} | {p['status']} | {_cell(p['value'], 110)} | {_cell(p['source'], 90)} |")
    L += ['', '## Interpretations (i)-(ix)', '', '| item | outcome | evidence |', '|---|---|---|']
    for k, v in o['interpretations'].items():
        L.append(f"| {k} | {v['outcome']} | {_cell(v['evidence'], 400)} |")
    L += ['', '## Predictions', '']
    for p in o['predictions']:
        L.append(f"- {p['prediction']}: **{p['outcome']}** — {p['detail']}")
    L += ['', '## Integrity checks', '']
    L += [f"- {k}: {v}" for k, v in o['checks'].items()]
    return '\n'.join(L) + '\n'


# ======================================================================================================================
#  8. main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--trial-out-dir', default=None, help='trial runs only: an output directory outside the results')
    ap.add_argument('--launch-log', default=OUT_NAMES['log'])
    args = ap.parse_args()
    out_dir = args.trial_out_dir or OUT_DIR
    tag = 'W179a'
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
    head = _git('rev-parse', 'HEAD', cwd=os.path.join(REPO, CLONE))
    if head != CLONE_HEAD:
        pre.append(f'clone HEAD {head} != {CLONE_HEAD}')
    main_rel = os.path.join(CLONE, 'main.tex')
    main_sha = _sha(main_rel)
    if main_sha != MAIN_SHA:
        pre.append(f'main.tex sha256 {main_sha} != {MAIN_SHA}')
    porcelain = _git('status', '--porcelain', '--', 'main.tex', cwd=os.path.join(REPO, CLONE))
    if porcelain:
        pre.append(f'clone main.tex not clean: {porcelain!r}')
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
    if inputs.get('REVIEW_COPY', {}).get('sha256') != MAIN_SHA:
        pre.append('main.tex is not byte-identical to manuscript_review/main_e926fb7_appE.tex')
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] ' + ' | '.join(pre))
        sys.exit(1)
    _log(f'[{tag}] preconditions hold: clone {head[:8]}, main.tex {main_sha[:8]} (= review copy), {len(inputs)} inputs committed clean')

    J = {k: _json(v['path']) for k, v in inputs.items() if v['path'].endswith('.json')}
    texts = {k: _text(v['path']) for k, v in inputs.items()}  # raw text of every input (needles may sit in a JSON record)
    FZ = J['FROZEN_JSON']
    T = FZ['tables']
    TAU = FZ['constants']['TAU']
    ALL = dict(T['cells'])
    ALL.update(T['cells_appended_w160'])
    for k, c in ALL.items():
        c.setdefault('cell', k)
    lines = _text(main_rel).split('\n')
    checks = {'main_tex_lines': len(lines) - (1 if lines[-1] == '' else 0) == MAIN_LINES}
    COND = cell_conditions(J, ALL)

    # ---- E.21 --------------------------------------------------------------------------------------------------------
    e21_rows, e21_note = parse_e21(lines)
    e21_out, idmap, p21 = check_e21(e21_rows, ALL, COND, J)
    checks['e21_61_rows_numbered_1_61'] = [r['no'] for r in e21_out] == list(range(1, 62))
    checks['e21_map_one_to_one_onto_frozen_cells'] = len(set(idmap.values())) == 61 == len(ALL) and set(idmap.values()) == set(ALL)
    checks['e21_mapping_problems_none'] = not p21
    # ---- E.20 --------------------------------------------------------------------------------------------------------
    e20_rows, e20_note, e20_note_line = parse_e20(lines)
    e20_out, used, p20 = check_e20(e20_rows, idmap, ALL, COND, T)
    checks['e20_mapping_problems_none'] = not p20
    t1_ids = {c['claim_id'] for c in T['claims']}
    t10_ids = {c['claim_id'] for c in T['a64']['rows']}
    used_t1 = {i for s, i in used if s == 'T1'}
    used_t10 = {i for s, i in used if s == 'T10'}
    net_dups = {i for i in t1_ids if i.startswith('G:') and i.endswith(':net')}
    row_claims_t1 = {r['claim'] for r in e20_out if r.get('source') == 'T1'}
    unused_t1 = t1_ids - row_claims_t1
    unused_t10 = t10_ids - used_t10
    by_id = {c['claim_id']: c for c in T['claims']}
    a64_by_id = {c['claim_id']: c for c in T['a64']['rows']}
    chk_ = by_id['CHECK:headline_V_minus_I_settled']
    b741 = by_id['B:n7_4h_e1']
    check_equals_row1 = (abs(chk_['d_gross'] + b741['d_gross']) < 1e-6 and chk_['ref_cell'] == b741['ref_cell']
                         and chk_['other_cell'] == b741['other_cell']
                         and abs(chk_['gross_threshold_or_bar'] - b741['gross_threshold_or_bar']) < 1e-6)
    rb = a64_by_id['E:soh050:delta_value_vs_g070_REPORT_BESIDE']
    r58 = a64_by_id['E:soh050:delta_value_vs_070']
    beside_equal = {'d_gross': rb['d_gross'] == r58['d_gross'],
                    'threshold_at_print': round(rb['gross_threshold_or_bar'] / 1000, 1) == round(r58['gross_threshold_or_bar'] / 1000, 1),
                    'multiple_at_print': round(rb['gross_multiple'], 2) == round(r58['gross_multiple'], 2),
                    'verdict': rb['gross_verdict'] == r58['gross_verdict']}
    mapping_i = {'n_T1': len(t1_ids), 'n_T1_rows': len(row_claims_t1), 'unused_T1': sorted(unused_t1),
                 'net_dups': sorted(net_dups), 'n_T10': len(t10_ids), 'n_T10_rows': len(used_t10), 'unused_T10': sorted(unused_t10),
                 'CHECK_equals_row1_with_sign_of_value_minus_I': check_equals_row1, 'REPORT_BESIDE_equals_row58': beside_equal}
    mapping_ok = (len(t1_ids) == 60 and len(row_claims_t1) == 55 and unused_t1 == net_dups | {'CHECK:headline_V_minus_I_settled'}
                  and len(net_dups) == 4 and len(used_t10) == 3 and unused_t10 == {'E:soh050:delta_value_vs_g070_REPORT_BESIDE'}
                  and check_equals_row1 and all(beside_equal.values()) and len(e20_out) == 58
                  and [r['no'] for r in e20_out] == list(range(1, 59)))
    checks['e20_58_rows_numbered_1_58'] = [r['no'] for r in e20_out] == list(range(1, 59))
    # ---- E.22 - E.24 ---------------------------------------------------------------------------------------------------
    e22_out, e22_note = check_e22(lines, idmap, ALL, T)
    e23_out, e23_note = check_e23(lines, T)
    e24_out = check_e24(lines, idmap, ALL, T, e21_out)

    # ---- tallies --------------------------------------------------------------------------------------------------------
    def tally(rows, extra=None):
        s = {'rows': len(rows), 'entries': 0, 'MATCH': 0, 'MISMATCH': 0, 'NOT_FROZEN_AGREES': 0, 'NOT_FROZEN_DISAGREES': 0}
        mm = []
        for r in rows:
            for k, e in r.get('entries', {}).items():
                s['entries'] += 1
                s[e['status']] = s.get(e['status'], 0) + 1
                if e['status'] in ('MISMATCH', 'NOT_FROZEN_DISAGREES'):
                    mm.append((r.get('no') or r.get('eval') or r.get('row'), k, e))
        for k, e in (extra or {}).items():
            s['entries'] += 1
            s[e['status']] = s.get(e['status'], 0) + 1
            if e['status'] in ('MISMATCH', 'NOT_FROZEN_DISAGREES'):
                mm.append(('note', k, e))
        return s, mm
    summ, mismatches = {}, []
    for name, rows, extra in (('E.20', e20_out, None), ('E.21', e21_out, None), ('E.22', e22_out, {'note_superseded': e22_note}),
                              ('E.23', e23_out, e23_note), ('E.24', e24_out, None)):
        s, mm = tally(rows, extra)
        summ[name] = s
        mismatches += [f"{name} {a} {b} [{e['status']}]: printed {e.get('printed')!r} frozen/derived "
                       f"{e.get('frozen', e.get('derived'))!r}" for a, b, e in mm]
    for p in p21 + p20:
        mismatches.append(f'MAPPING {p}')

    # ---- derived quantities for the prose ---------------------------------------------------------------------------
    inv = {v: k for k, v in idmap.items()}
    D = {'TAU': TAU, 'n_e20': len(e20_out), 'n_e21': len(e21_out), 'n_t2': len(ALL), 'e20_note_line': e20_note_line}
    D['n_gross_net'] = sum(1 for r in e20_out if r.get('source') == 'T1')
    D['t10_rows'] = {r['no']: r['claim'] for r in e20_out if r.get('source') == 'T10'}
    D['cond_175'] = COND['h_unit_m175']['m'] == 1.75 and COND['h_x0_m175']['m'] == 1.75
    D['cond_050'] = COND['e_soh050']['ageing'][-1] == 0.5
    thr_bad, bar_bad, verd_bad = [], [], []
    for c in T['claims'] + T['a64']['rows']:
        ra = {'7aa017f0': 'ref:7aa017f0', 'bd504ecf': 'ref:bd504ecf'}.get(c['ref_cell'], c['ref_cell'])
        a, b = ALL[ra], ALL[c['other_cell']]
        if c['gross_rule'].startswith('max'):
            thr = max(3 * max(a['band_width'], b['band_width']), 2 * TAU)
            if abs(thr - c['gross_threshold_or_bar']) > 1e-6:
                thr_bad.append(c['claim_id'])
            if (c['gross_multiple'] >= 1.0) != (c['gross_verdict'] == 'determinate'):
                verd_bad.append(c['claim_id'])
        else:
            us = [x for x in (a, b) if x['status'] == 'uncertified']
            bar = 3 * max(max(abs(x['gap']), abs(x['slack'])) for x in us)
            if abs(bar - c['gross_threshold_or_bar']) > 1e-6:
                bar_bad.append(c['claim_id'])
            det = c['gross_multiple'] > 1 and c['Qcc_multiple_report_only'] > 1
            if det != (c['gross_verdict'] == 'determinate'):
                verd_bad.append(c['claim_id'])
    n_cert = sum(1 for c in T['claims'] + T['a64']['rows'] if c['gross_rule'].startswith('max'))
    D['thr_formula_ok'] = not thr_bad and not verd_bad
    D['thr_formula'] = {'certified_claims': n_cert, 'threshold_reproduced': n_cert - len(thr_bad), 'not_reproduced': thr_bad,
                        'verdict_inconsistent': verd_bad}
    n_unc = sum(1 for c in T['claims'] if not c['gross_rule'].startswith('max'))
    D['bar_formula_ok'] = not bar_bad and not verd_bad
    D['bar_formula'] = {'uncertified_claims': n_unc, 'bar_reproduced': n_unc - len(bar_bad), 'not_reproduced': bar_bad}
    D['flagged_evals'] = sorted(inv[c] for c, x in ALL.items() if x.get('flag_non_clean_turning_point') is True)
    D['beside'] = {k: by_id[k]['uncertified_form_beside_A64_ruling4'] for k in by_id if 'uncertified_form_beside_A64_ruling4' in by_id[k]}
    D['row_of_claim'] = {r['claim']: r['no'] for r in e20_out if r.get('claim')}
    fl = {k: (v['verdict'], by_id[k]['gross_verdict']) for k, v in D['beside'].items()}
    D['flag_verdicts'] = fl
    D['flag_verdicts_unchanged'] = all(a == b for a, b in fl.values())
    D['rows_30_31_rule'] = {D['row_of_claim'].get(k): by_id[k]['gross_rule'] for k in ('J:e3_to_e4', 'J:e4_to_e5')}
    D['rows_30_31_unc'] = D['rows_30_31_rule'] == {30: 'uncertified form', 31: 'uncertified form'}
    tw = T['at_or_above_0_95_tau']['twins']
    cal = by_id['E:n7_4h_e1_C2_calfade:value_minus_I']
    D['row48_vs_row1'] = (tw['evidence']['e_c2_calfade']['reproduced'] is True and tw['evidence']['e_c2_calfade']['through'] == 172
                          and inv['e_c2_calfade'] == 56 and abs(cal['d_gross'] + b741['d_gross']) < 0.01
                          and D['row_of_claim']['E:n7_4h_e1_C2_calfade:value_minus_I'] == 48 and D['row_of_claim']['B:n7_4h_e1'] == 1)
    D['row48_detail'] = {'row48_d_gross': cal['d_gross'], 'row1_d_gross': b741['d_gross'],
                         'twin_evidence': tw['evidence']['e_c2_calfade']}
    D['row58_beside'] = all(beside_equal.values()) and tw['evidence']['g070_neutrality']['through'] == 172 and inv['g070_neutrality'] == 61
    D['row58_beside_detail'] = {'equal_at_print': beside_equal, 'twin_evidence': tw['evidence']['g070_neutrality']}
    D['row45_steps'] = abs(cand_plan(ALL['ref:5ca4f86c'])['7'][1] - cand_plan(ALL['l_df1a5525'])['7'][1]) / 0.5
    g_rows = {r['no']: [r['claim'], r['net_claim']] for r in e20_out if r.get('claim', '') and r['claim'].startswith('G:')}
    D['g_rows'] = g_rows
    D['g_rows_ok'] = sorted(g_rows) == [17, 18, 19, 20] and all(v[1] and v[1].endswith(':net') for v in g_rows.values())
    D['refs_k0'] = {inv[c]: ALL[c].get('k0_run') for c in ('ref:7aa017f0', 'ref:bd504ecf', 'ref:5ca4f86c', 'ref:e28de4ac')}
    D['refs_k0_none'] = sorted(D['refs_k0']) == [1, 2, 40, 41] and all(v is None for v in D['refs_k0'].values())
    D['e21_note_line'] = next(i + 1 for i, ln in enumerate(lines) if ln == e21_note)
    w168 = J['W168_X0']
    D['x0_k0'] = w168['decision_primary']['k0'] if w168['decision_primary']['N'] == w168['decision_primary']['k0'] else None
    D['fig_k0_line'] = next((i + 1 for i, ln in enumerate(lines) if 'from the first residual pass $k_0 = 123$ to the certification '
                             'cycle $k^{*} = 181$' in ln), None)
    D['l47_redecided'] = next((i + 1 for i, ln in enumerate(lines) if 'Every certificate reported in this paper was decided, or '
                               're-decided from its committed record' in ln), None)
    w142 = J['W142_V6_FROM_RECORDS']['item1_v6_on_every_record']
    pb5 = ALL['pb_y2025_n5']
    D['row16'] = {'superseded': pb5.get('superseded'), 'v6_on_record': w142['pb_y2025_n5']['v6'].get('status'),
                  'v6_vetoes': w142['pb_y2025_n5']['v6'].get('n_vetoes'), 'rerun_cell_is_eval_15': inv.get('pb_y2025_n5_v6') == 15,
                  'T5_note': next(r['note'] for r in T['phase_b'] if r['cell'] == 'pb_y2025_n5')}
    D['row16_ok'] = pb5.get('superseded') is True and inv['pb_y2025_n5'] == 16 and inv['pb_y2025_n5_v6'] == 15 and 'non-Optimal' in D['row16']['T5_note']
    w173 = J['W173_COUNTS']['item2']['x0']
    D['w173_x0'] = {'b_box': w173['b_box'], 'certificate_n_feasible': w173['certificate_record']['value']['n_feasible_neighbours'],
                    'all_polled': w173['certificate_record']['value']['all_feasible_neighbours_polled']}
    D['w173_14'] = w173['b_box']['n_feasible'] == 14 and w173['b_box']['n_evaluated'] == 14
    D['w173_resettled'] = w173['c_resettled_in_frozen_tables']['n']
    D['w173_8'] = D['w173_resettled'] == 8
    res = sorted({round(r['search_resolution_eur'], 2) for r in w173['rows']})
    D['w173_res'] = {'search_resolution_eur_distinct': res, 'min': min(res), 'max': max(res)}
    D['w173_res_note'] = min(res) >= 18449.66 - 0.01
    n_at_floor = sum(1 for r in w173['rows'] if abs(r['search_resolution_eur'] - 18449.663947025518) < 0.01)
    D['w173_res'].update({'n_rows': len(w173['rows']), 'n_threshold_equal_to_sigma_Q': n_at_floor,
                          'n_threshold_above_sigma_Q': len(w173['rows']) - n_at_floor})
    D['n_e22'] = len(e22_out)
    D['e22_plans'] = {r['eval']: r['entries']['nodes_year']['frozen'] for r in e22_out}
    D['e22_plans_ok'] = all(r['entries']['nodes_year']['status'] == 'MATCH' for r in e22_out) and len(e22_out) == 8
    D['e22_verdicts'] = {r['eval']: [v['frozen'][3] for k, v in r['entries'].items() if k != 'nodes_year'] for r in e22_out}
    D['e22_all_det'] = all(all(x == 'determinate' for x in v) and v for v in D['e22_verdicts'].values())
    arms = T['benchmark']['w160_additions']['arms_in_full']
    D['n_starts'] = [arms['passive']['n_starts'], arms['price_taker']['n_starts']]
    D['band_spread'] = {a: [arms[a]['multimodality_band'], max(arms[a]['Q_by_start'].values()) - min(arms[a]['Q_by_start'].values())]
                        for a in arms}
    D['band_is_spread'] = all(abs(v[0] - v[1]) < 1e-6 for v in D['band_spread'].values())
    dz = {r['cell']: r for r in T['dead_zone']['cells']}
    fp = [idmap[n] for n in (39, 40, 41, 42, 45, 50, 51)]
    D['fp_gap'] = {inv[c]: round(dz[c]['t_sum_at_cap'] / 1000, 1) for c in fp}
    D['fp_gap_ok'] = min(D['fp_gap'].values()) == -9.4 and max(D['fp_gap'].values()) == -8.8 and all(
        COND[c]['m'] == 2.0 for c in fp)
    sh = {inv[c]: [round(100 * dz[c]['share_by_node'][n], 1) for n in '579'] for c in fp if dz[c].get('share_by_node')}
    D['fp_share'] = sh
    D['fp_share_ok'] = all(round(v[0]) == 55 and round(v[1]) == 14 and round(v[2]) == 31 for v in sh.values())
    rr = {inv[c]: round(dz[c]['pf_primal_last'], 3) for c in fp if dz[c].get('pf_primal_last') is not None}
    D['fp_res'] = rr
    D['fp_res_ok'] = min(rr.values()) >= 0.675 and max(rr.values()) < 0.705
    z42 = dz[idmap[42]]
    D['e42'] = {'cause': z42['cause'], 'lapse_cycles': [e['cycle'] for e in z42['lapse_events']], 'entry': z42['entry']}
    D['e42_ok'] = z42['cause'] == 'lapse reset' and len(z42['lapse_events']) == 2 and 'two TSO recoveries' in z42['entry']
    zh = dz['h_f9eae48f']
    D['h15'] = {'eval': inv['h_f9eae48f'], 'gap_k': round(zh['t_sum_at_cap'] / 1000, 1),
                'share': [round(100 * zh['share_by_node'][n], 1) for n in '579'], 'pf_primal_last': round(zh['pf_primal_last'], 3),
                'm': COND['h_f9eae48f']['m']}
    D['h15_ok'] = (D['h15']['gap_k'] == -2.5 and D['h15']['pf_primal_last'] == 0.027 and D['h15']['m'] == 1.5
                   and [round(x) for x in D['h15']['share']] == [55, 14, 31])
    zd = dz['d_36686489']
    D['d366'] = {'eval': inv['d_36686489'], 'cause': zd['cause'], 'gap_k': round(zd['t_sum_at_cap'] / 1000, 1),
                 'plan': plan_str(cand_plan(ALL['d_36686489'])), 'non_clean_after_N': ALL['d_36686489']['non_clean_after_N']}
    D['d366_ok'] = zd['cause'].startswith('growth test after five TSO recoveries') and zd['t_sum_at_cap'] > 0 and inv['d_36686489'] == 6
    gr = [r['cell'] for r in e24_out if dz.get(r['cell'], {}).get('gap_refused') is True]
    D['e24_counts'] = {'gap_refused_rows': len(gr), 'others': [inv[r['cell']] for r in e24_out if r['cell'] not in gr]}
    D['e24_counts_ok'] = len(gr) == 7 and len(e24_out) == 9
    # E.6
    set42 = set(range(3, 12)) | {15, 22, 23} | set(range(26, 32)) | set(range(34, 40)) | set(range(42, 60))
    cells42 = {idmap[n] for n in set42}
    w153_cells = {c['cell'] for c in J['W153_STATS']['result']['cells']}
    t2_w157 = set(T['cells'])
    b_cells = {c for c in t2_w157 if c.startswith('b_')}
    refs = {c for c in t2_w157 if c.startswith('ref:')}
    D['set42'] = {'n_listed': len(set42), 'equals_W153_42': cells42 == w153_cells,
                  'arith': f'{len(t2_w157)} T2 cells (tables.cells) - {len(b_cells)} B cells - {len(refs)} references = '
                           f'{len(t2_w157) - len(b_cells) - len(refs)}',
                  'B_rows': sorted(inv[c] for c in b_cells), 'reference_rows': sorted(inv[c] for c in refs),
                  'equals_T2_minus_B_minus_refs': cells42 == t2_w157 - b_cells - refs,
                  'certification_stats_included_flag': sorted(c for c in ALL if ALL[c].get('certification_stats_included') is True) == sorted(cells42)}
    D['set42_ok'] = len(set42) == 42 and D['set42']['equals_W153_42'] and D['set42']['equals_T2_minus_B_minus_refs']
    cert42 = [ALL[c] for c in cells42 if ALL[c]['status'] == 'certified']
    unc42 = [ALL[c] for c in cells42 if ALL[c]['status'] != 'certified']
    D['cert_split'] = [len(cert42), sum(1 for c in cert42 if c['branch'] == 'oscillatory'), sum(1 for c in cert42 if c['branch'] == 'monotone')]
    causes = [('growth test' if (c['cause_uncertified'] or '').startswith('growth test') else c['cause_uncertified']) for c in unc42]
    D['unc_split'] = [len(unc42), causes.count('gap clause'), causes.count('lapse reset'), causes.count('growth test')]
    ks = [c['k_star'] for c in cert42]
    D['med_kstar'] = statistics.median(ks)
    kk = [c['k_star'] - c['k0_run'] for c in cert42]
    w153t = J['W153_STATS']['result']['totals']
    D['kk0'] = [statistics.median(kk), min(kk), max(kk)]
    D['kk0_w153'] = {'k_star_minus_N': w153t['cycles_after_k0_k_star_minus_N'], 'at_decision': w153t['cycles_after_k0_k_star_minus_k0_at_decision']}
    rg = [c['range_over_tau'] for c in cert42]
    D['med_rng_full'] = statistics.median(rg)
    D['med_rng'] = round(D['med_rng_full'], 2)
    add4 = {n: (idmap[n], ALL[idmap[n]]['status']) for n in (32, 33, 60, 61)}
    D['added4'] = add4
    D['added4_ok'] = all(v[1] == 'certified' for v in add4.values()) and {v[0] for v in add4.values()} == \
        {'h_x0_m175', 'h_unit_m175', 'e_soh050', 'g070_neutrality'}
    A = T['at_or_above_0_95_tau']
    reg = A['registry']
    n_reg = len(reg)
    n_sup = sum(1 for r in reg if r.get('superseded'))
    n_tw = sum(1 for r in reg if r.get('twin_of'))
    n_cert_all = sum(1 for c in ALL.values() if c['status'] == 'certified')
    D['n46'] = n_reg - n_sup - n_tw
    D['n10'] = A['n_counted']
    reg_cells = {r['cell'] for r in reg}
    sup_ids = sorted(inv[r['cell']] for r in reg if r.get('superseded'))
    tw_ids = sorted(inv[r['cell']] for r in reg if r.get('twin_of'))
    tw_of = sorted({inv[r['twin_of']] for r in reg if r.get('twin_of')})
    D['n46_detail'] = {'registry': n_reg, 'certified_cells_in_T2': n_cert_all, 'superseded': n_sup, 'twins': n_tw,
                       'arith': f'{n_reg} - {n_sup} - {n_tw} = {n_reg - n_sup - n_tw}', 'n_counted': A['n_counted'],
                       'registry_equals_the_certified_rows_of_E21': reg_cells == {c for c, x in ALL.items() if x['status'] == 'certified'},
                       'superseded_E21_rows': sup_ids, 'twin_E21_rows': tw_ids, 'twins_of_E21_row': tw_of}
    D['n46_ids_ok'] = (D['n46_detail']['registry_equals_the_certified_rows_of_E21'] and sup_ids == [16] and tw_ids == [56, 61]
                       and tw_of == [2])
    D['ten_ids'] = sorted(inv[c] for c in A['counted_cells'])
    e36 = ALL['i_5a6a88b4']
    D['e36'] = {'eval': inv['i_5a6a88b4'], 'branch': e36['branch'], 'range_over_tau': e36['range_over_tau'],
                'beside': A['beside'].get('i_5a6a88b4')}
    D['e36_ok'] = inv['i_5a6a88b4'] == 36 and e36['branch'] == 'monotone' and round(e36['range_over_tau'], 2) == 0.94 and e36['range_over_tau'] < 0.95
    mv = J['W142_V6_FROM_RECORDS']['item3_movement_record']['rows']
    n25 = w163_checks(J['W163_CHECK']).get('N25', {})
    D['w163'] = w163_checks(J['W163_CHECK'])
    over = (n25.get('value') or {}).get('over_tau', {})
    D['continued'] = {'W142_item3': {k: round(abs(v['over_tau']), 3) for k, v in mv.items()},
                      'W163_N25_over_tau': {k: round(abs(v), 4) for k, v in over.items()},
                      'E21_numbers': {'b_4649234b': inv['b_4649234b'], 'd_c52e1670': inv['d_c52e1670'], 'b_2a0ba8b2': inv['b_2a0ba8b2']}}
    movements_ok = (round(abs(mv['b_4649234b (cell 3)']['over_tau']), 2) == 0.74
                    and round(abs(mv['d_c52e1670 at its v6 certificate']['over_tau']), 2) == 0.78
                    and round(abs(mv['d_c52e1670 at the last-pair reading\'s 148 (W141 V1)']['over_tau']), 2) == 0.88
                    and inv['b_4649234b'] == 14 and inv['d_c52e1670'] == 3)
    third = any(k.startswith('b_2a0ba8b2') for k in over)
    D['continued_status'] = 'MISMATCH' if (movements_ok and third) else ('MATCH' if movements_ok else 'MISMATCH')
    D['continued']['finding'] = (('the count "two" is wrong: movements 0.74 / 0.78-0.88 tau MATCH for 14 and 3, but W163 N25 records '
                                  'a third evaluation continued past a settling-rule certificate: 12 (b_2a0ba8b2, v3 run to 213 past '
                                  f"its v4 k* 173, {round(abs(over.get('b_2a0ba8b2 (v3 run, Q(213) - Q(173); v4 k* 173)', 0)), 4)} tau)")
                                 if (movements_ok and third) else None)
    nc_cells = {c['cell']: c['non_clean_after_N'] for c in (ALL[x] for x in cells42) if c.get('non_clean_after_N')}
    D['nc'] = {'cycles': sum(len(v) for v in nc_cells.values()), 'evaluations': len(nc_cells),
               'by_family_W153': w153t['non_clean_block_events_after_N_by_family_total'],
               'evals': sorted(inv[c] for c in nc_cells)}
    D['nc_ok'] = D['nc']['cycles'] == 21 and D['nc']['evaluations'] == 10 and w153t['non_clean_block_events_after_N_by_family_total'] == {'TSO': 21, 'DSO': 0, 'ESSO': 0}
    D['acc'] = {'after_N_total': w153t['acceptable_clean_block_events_after_N_total'],
                'by_family': w153t['acceptable_clean_after_N_by_family_total'],
                'cells': w153t['cells_with_acceptable_clean_after_N'],
                'definition': J['W153_STATS']['result']['definitions']['acceptable_clean']}
    D['acc_ok'] = D['acc']['after_N_total'] == 119 and D['acc']['by_family'] == {'TSO': 0, 'DSO': 119, 'ESSO': 0}
    n63 = D['w163'].get('N63', {}).get('value') or {}
    rels = [abs(v) for v in n63.values() if isinstance(v, (int, float))]
    D['tail_range_ok'] = bool(rels) and round(min(rels) * 1e6, 1) == 1.0 and round(max(rels) * 1e6, 1) == 1.2 and all(v < 0 for v in n63.values() if isinstance(v, (int, float)))
    n49v = (D['w163'].get('N49') or {}).get('value') or {}
    D['three_refs'] = {'W163_N49': n49v, 'in_appendix_E': {'x0': inv['ref:7aa017f0'], 'unit': inv['ref:bd504ecf'], 'c_star': None},
                       'E21_k_star': {'x0 (eval 1)': ALL['ref:7aa017f0']['k_star'], 'unit (eval 2)': ALL['ref:bd504ecf']['k_star']},
                       'E21_replay_printed': {'1': ALL['ref:7aa017f0'].get('replay_bitwise_through'),
                                              '2': ALL['ref:bd504ecf'].get('replay_bitwise_through')}}
    D['three_refs_n'] = len(n49v)
    replayed = {k: v.get('through') for k, v in n49v.items()}
    D['r2_wording_mismatch'] = (replayed.get('x0') != ALL['ref:7aa017f0']['k_star']
                                or replayed.get('n7_4h_e1') != ALL['ref:bd504ecf']['k_star'])
    D['r2_note'] = (f"the record (W101 G13/G19 via W163 N49) holds: bitwise through {replayed} -- the cycles at which the EARLIER "
                    f"(residual-based) rule had certified them ('old certification cycle', Addendum 53 report). Appendix E defines "
                    f"the certification cycle as k* (E.21 caption) and prints k* {ALL['ref:7aa017f0']['k_star']} and "
                    f"{ALL['ref:bd504ecf']['k_star']} for evaluations 1 and 2; read with that definition the sentence is wrong for both "
                    f"and C* is not in Appendix E. Rated on the appendix's own definition.")
    # rounding check over the tables
    kdp, rdp = [], []
    for r in e20_out:
        for k in ('d_gross_k', 'threshold_or_bar_k', 'd_net_k'):
            p = r.get('entries', {}).get(k, {}).get('printed')
            if p and p != '---':
                kdp.append(dec_places(p.replace('$-$', '-')))
        for k in ('x_gross', 'x_net'):
            p = r.get('entries', {}).get(k, {}).get('printed')
            if p and p != '---':
                rdp.append(dec_places(p))
    for r in e21_out:
        for k in ('band_k', 'gap_k', 'slack_k'):
            p = r.get('entries', {}).get(k, {}).get('printed')
            if p and p != '---':
                kdp.append(dec_places(p))
        p = r.get('entries', {}).get('range_over_tau', {}).get('printed')
        if p and p != '---':
            rdp.append(dec_places(p))
    for r in e22_out:  # E.22: Delta, threshold (k EUR), x
        for k, e in r['entries'].items():
            if k != 'nodes_year':
                kdp += [dec_places(e['printed'][0].replace('$-$', '-')), dec_places(e['printed'][1])]
                rdp.append(dec_places(e['printed'][2]))
    for r in e23_out:  # E.23: Q and consistency pass (k EUR); counts, p.u. and GWh are neither amounts in k EUR nor ratios
        for k in ('Q_k', 'consistency_pass_k'):
            p = r.get('entries', {}).get(k, {}).get('printed')
            if p and p != '---':
                kdp.append(dec_places(p.replace('$-$', '-').replace('\\,', '')))
    resid_dp = []
    for r in e24_out:  # E.24: gap (k EUR); Resid. = residual / tolerance (a ratio); shares in % (reported beside)
        p = r['entries'].get('gap_k', {}).get('printed')
        if p and p != '---':
            kdp.append(dec_places(p.replace('$-$', '-')))
        p = r['entries'].get('pf_primal_last', {}).get('printed')
        if p and p != '---':
            resid_dp.append(dec_places(p))
    D['rounding_detail'] = {'k_EUR_dp': sorted(set(kdp)), 'ratio_dp_E20_E22': sorted(set(rdp)), 'n_k': len(kdp),
                            'n_ratio': len(rdp), 'E24_resid_ratio_dp': sorted(set(resid_dp)), 'n_E24_resid': len(resid_dp)}
    D['rounding_ok'] = set(kdp) == {1} and set(rdp) == {2} and set(resid_dp) <= {2}

    # ---- sign / verdict changes over the 55 (sec. 4.6 statement) ----------------------------------------------------
    sc, vc = [], []
    for r in e20_out:
        if r.get('source') != 'T1':
            continue
        g = by_id[r['claim']]
        n = by_id[r['net_claim']]
        if (g['d_gross'] > 0) != (n['d_net'] > 0):
            sc.append(r['no'])
        if g['gross_verdict'] != n['net_verdict_W153']:
            vc.append({'row': r['no'], 'claim': r['claim'], 'gross': [round(g['gross_multiple'], 2), g['gross_verdict']],
                       'net': [round(n['net_multiple'], 2), n['net_verdict_W153']]})
    salvage_55 = {'n': D['n_gross_net'], 'sign_changes_rows': sc, 'verdict_changes': vc}
    r34 = next(r for r in e20_out if r['no'] == 34)
    salvage_55['row34_d_net_positive'] = by_id[r34['net_claim']]['d_net'] > 0
    D['salvage_55'] = salvage_55
    P = prose(lines, idmap, ALL, T, J, e20_out, e21_out, D)

    # ---- interpretations ----------------------------------------------------------------------------------------------
    needles = needle_lines(texts)
    checks['every_needle_found'] = all(v['found'] for v in needles.values())
    rule_rows = {r['no']: r['entries']['rule'] for r in e21_out if 'entries' in r}
    rec_cells = [r['cell'] for r in e21_out if r.get('entries', {}).get('rule', {}).get('printed') == 'record']
    rec_ev = {}
    for c in rec_cells:
        k = W142_KEY.get(c)
        v6 = (w142.get(k) or {}).get('v6') if k else None
        rec_ev[inv[c]] = {'cell': c, 'w142_record': k, 'v6_status': None if v6 is None else v6.get('status'),
                          'v6_k_star': None if v6 is None else v6.get('k_star'), 'T2_k_star': ALL[c].get('k_star'),
                          'T2_end': ALL[c].get('end_cycle'), 'v6_k_cap': None if v6 is None else v6.get('k_cap')}
    rec_consistent = {n: (e['v6_status'] == 'certified' and e['v6_k_star'] == e['T2_k_star']) or
                      (e['v6_status'] == 'uncertified' and e['v6_k_cap'] == e['T2_end']) for n, e in rec_ev.items()}
    I = {}
    I['(i) 60 -> 55 + 3 mapping'] = {'outcome': 'confirmed' if mapping_ok else 'corrected', 'evidence': mapping_i}
    plan_rows_ok = all(r['entries']['plan']['status'] == 'MATCH' and r['entries']['year']['status'] == 'MATCH'
                       for r in e21_out if 'entries' in r) and checks['e21_map_one_to_one_onto_frozen_cells']
    special = {'yl_y2030': [plan_str(cand_plan(ALL['yl_y2030'])), ALL['yl_y2030']['candidate_canonical']['investment_year']],
               'yl_y2035': [plan_str(cand_plan(ALL['yl_y2035'])), ALL['yl_y2035']['candidate_canonical']['investment_year']],
               'pb_*': {c: [plan_str(cand_plan(ALL[c])), ALL[c]['candidate_canonical']['investment_year']] for c in ALL if c.startswith('pb_')},
               'ref:5ca4f86c': [plan_str(cand_plan(ALL['ref:5ca4f86c'])), ALL['ref:5ca4f86c']['candidate_canonical']['investment_year']],
               'ref:e28de4ac': [plan_str(cand_plan(ALL['ref:e28de4ac'])), ALL['ref:e28de4ac']['candidate_canonical']['investment_year']]}
    expected_ii = {'yl_y2030': ['7: 0.25/1', 2030], 'yl_y2035': ['7: 0.25/1', 2035],
                   'pb_y2025_n5': ['5: 0.25/0.5', 2025], 'pb_y2025_n7': ['7: 0.25/0.5', 2025], 'pb_y2025_n9': ['9: 0.25/0.5', 2025],
                   'pb_y2030_n5': ['5: 0.25/0.5', 2030], 'pb_y2030_n7': ['7: 0.25/0.5', 2030], 'pb_y2030_n9': ['9: 0.25/0.5', 2030],
                   'ref:5ca4f86c': ['5: 0.25/0.5; 7: 1/3.5', 2030], 'ref:e28de4ac': ['5: 0.25/1; 7: 1/3', 2030]}
    got_ii = {c: [plan_str(cand_plan(ALL[c])), ALL[c]['candidate_canonical']['investment_year']] for c in expected_ii}
    ii_named_ok = got_ii == expected_ii and inv['ref:5ca4f86c'] == 40 and inv['ref:e28de4ac'] == 41
    I['(ii) plan behind every cell of E.21'] = {
        'outcome': 'confirmed' if (plan_rows_ok and ii_named_ok) else 'corrected',
        'evidence': {'all_61_rows_plan_year_match_candidate_canonical': plan_rows_ok, 'named_cells': special,
                     'expert_stated_plans_equal_frozen_candidate_canonical': ii_named_ok, 'expert_stated': expected_ii,
                     'conditions_from_specs': {f"{inv[c]} {c}": COND[c]['source'] for c in ('ref:7aa017f0', 'ref:5ca4f86c', 'yl_y2030', 'e_c4', 'h_x0_m175')}}}
    rule_ok = all(e['status'] == 'NOT_FROZEN_AGREES' for e in rule_rows.values())
    rec_ok_except_16 = all(v for n, v in rec_consistent.items() if n != 16) and rec_consistent.get(16) is False
    I['(iii) Rule column'] = {
        'outcome': 'confirmed, with one qualification (row 16)' if (rule_ok and rec_ok_except_16) else 'corrected',
        'evidence': {'rule_column_matches_certifying_spec': rule_ok,
                     'record_rows_v6_on_record_W142_item1': rec_ev, 'v6_on_record_consistent_with_T2': rec_consistent,
                     'qualification': ('row 16 (pb_y2025_n5, superseded): v6 applied to its W118 record did NOT decide it '
                                       f"(status {w142['pb_y2025_n5']['v6']['status']!r}, {w142['pb_y2025_n5']['v6']['n_vetoes']} veto); "
                                       'the k* 167 / 0.99 / 4.5 printed are the v2 certificate. "record" holds (the rule was applied '
                                       'to the record) but the row is not a certificate under the rule; its $^{s}$ and the note say why. '
                                       'Section 4.7 (main.tex l. ' + str(D['l47_redecided']) + ') "Every certificate reported in this paper was decided, or '
                                       're-decided from its committed record, under the rule ... and certifies at the cycle reported" '
                                       'is therefore not true of row 16 as printed (main text: W179b scope).')}}
    six = [r['label'] for r in w173['rows'] if not r.get('frozen_table_rows')]
    # explicit field paths: cells.<c>.canonical in the v4 / v5 / v6 / W118 v2 specs; the ext v3 and A64 v1 specs carry no
    # canonical, only cells.<c>.candidate_key, resolved through the frozen cells' candidate_key -> candidate_canonical
    key2canon = {}
    for c in ALL.values():
        key2canon.setdefault(c['candidate_key'], set()).add(json.dumps(c['candidate_canonical'], sort_keys=True))
    spec_cands, spec_cand_source = set(), {}
    for key in ('SPEC_V4', 'SPEC_V5', 'SPEC_V6', 'SPEC_EXT_V3', 'SPEC_W118_V2', 'SPEC_A64_V1'):
        n_read = 0
        for nm, sc_ in J[key]['cells'].items():
            if key in ('SPEC_EXT_V3', 'SPEC_A64_V1'):
                canons = key2canon.get(sc_['candidate_key'])
                if not canons or len(canons) != 1:
                    raise RuntimeError(f'{INPUTS[key]} cells.{nm}.candidate_key resolves to {canons} in the frozen cells')
                cc = json.loads(next(iter(canons)))
            else:
                cc = sc_['canonical']
            spec_cands.add(json.dumps({'y': cc['investment_year'], 'n': {k: [float(x) for x in v] for k, v in sorted(cc['nodes'].items())}}, sort_keys=True))
            n_read += 1
        spec_cand_source[INPUTS[key]] = n_read
    frozen_cands = {json.dumps({'y': c['candidate_canonical']['investment_year'], 'n': {k: [float(x) for x in v] for k, v in sorted(c['candidate_canonical']['nodes'].items())}}, sort_keys=True)
                    for c in ALL.values()}

    def lab2cand(lab):
        m = re.match(r'y(\d{4})__(.*)', lab)
        nodes = {'5': [0.0, 0.0], '7': [0.0, 0.0], '9': [0.0, 0.0]}
        for part in m.group(2).split('__'):
            mm = re.fullmatch(r'n(\d)_p([0-9.]+)_e([0-9.]+)', part)
            nodes[mm.group(1)] = [float(mm.group(2)), float(mm.group(3))]
        return json.dumps({'y': int(m.group(1)), 'n': dict(sorted(nodes.items()))}, sort_keys=True)
    six_hits = {lab: {'in_frozen_cells': lab2cand(lab) in frozen_cands, 'in_a_certification_spec': lab2cand(lab) in spec_cands}
                for lab in six}
    expected_six = {'y2025__n5_p0.25_e0.5__n7_p0.25_e0.5', 'y2025__n7_p0.25_e0.5__n9_p0.25_e0.5',
                    'y2030__n5_p0.25_e0.5__n9_p0.25_e0.5', 'y2030__n7_p0.25_e0.5__n9_p0.25_e0.5',
                    'y2025__n5_p0.25_e0.5__n7_p0.25_e0.5__n9_p0.25_e0.5', 'y2030__n5_p0.25_e0.5__n7_p0.25_e0.5__n9_p0.25_e0.5'}
    iv_ok = set(six) == expected_six and not any(v['in_frozen_cells'] or v['in_a_certification_spec'] for v in six_hits.values())
    I['(iv) the six other box neighbours evaluated by the search only'] = {
        'outcome': 'confirmed (scoped)' if iv_ok else 'corrected',
        'evidence': {'six_labels_W173': six, 'equal_to_the_six_named': set(six) == expected_six, 'searched': six_hits,
                     'spec_cells_read': spec_cand_source, 'sister_result': 'P5_15_W179C_CLOSURE_SCOPE.md (3994ac64) item (e): same finding',
                     'scope': ('W173 item2.x0.rows frozen_table_rows empty; candidate absent from the 61 frozen cells and from the '
                               'cells of the six certification stage specs (v4 e11fbc89, v5 ab32ffc9, v6 96c23404, ext v3 84775dc4, '
                               'W118 v2 fc791891, A64 v1 44a2dce8); search-time F - F(0) 101.8-151.8 k EUR per W173 report')}}
    v_unit = needles['nrf_row_one_per_scenario_period']['found'] and needles['nrf_violation_entry_per_row']['found']
    v_where = needles['w116_violations_counted_at_reevaluation']['found'] and needles['nrf_row_deactivated_in_reevaluation']['found']
    v_warm = needles['w116_warm_from_certified_models']['found'] and needles['w93_coordinated_is_settled']['found'] and needles['w93_settled_eval_key']['found']
    v_phase = needles['w116_arm_cost_is_phase_C']['found'] and needles['w116_curtailment_rows_phase_A']['found']
    arms_v = T['benchmark']['w160_additions']['arms_in_full']
    v_sources = {a: {'Q_source_by_start': arms_v[a]['arm_cost_source_by_start'],
                     'trigger_by_start': {k: v['trigger'] for k, v in arms_v[a]['consistency_nrf_by_start'].items()}}
                 for a in arms_v}
    v_all_phase_c = all(set(v['Q_source_by_start'].values()) == {'phase_C_sequential_pass'} for v in v_sources.values())
    v_ok = v_unit and v_where and v_warm and v_phase and v_all_phase_c and needles['report_v3_warm_source_is_settled_x0']['found']
    I['(v) T6: unit of NRF violations, consistency pass, warm'] = {
        'outcome': ('corrected (the reading of the consistency pass); the unit and "warm" confirmed' if v_ok
                    else 'not confirmed (a code needle was not found)'),
        'evidence': {
            'unit': ('one entry per violated row uncoord_no_reverse_flow[s_m, s_o, p] of a DSO block, i.e. per DN interface '
                     'and hour of a year-season block (the single-scenario instance has one market and one operation scenario: '
                     f"{T['benchmark']['w160_additions']['no_reverse_flow_definition']['where']!r}); n = 29 / 70 of the 36 x 24 = 864 "
                     'DN interface-hours. Code: ' + json.dumps({k: needles[k]['lines'] for k in ('nrf_row_one_per_scenario_period', 'nrf_violation_entry_per_row')})),
            'unit_found': v_unit,
            'consistency': ('the exceedances are counted in the consistency RE-EVALUATION (phase B: each DN re-solved at the TN\'s '
                            'actual interface voltage with its voltage, thermal and no-reverse-flow rows deactivated, the NRF row '
                            'evaluated afterwards as a hard DN limit); a violation triggers the declared SEQUENTIAL PASS (phase C: each '
                            'DN re-solved at that voltage WITH the NRF rows, then the TN re-dispatched). The "Consistency pass (k EUR)" '
                            'column is Q(phase C) - Q(phase A) (pass_effect_eur). So "the exceedance columns count the entries of that '
                            'pass" is not right if "that pass" is the consistency pass: they count the entries of the re-evaluation '
                            'that precedes and triggers it. Code lines: ' + json.dumps({k: needles[k]['lines'] for k in (
                                'w116_violations_counted_at_reevaluation', 'nrf_row_deactivated_in_reevaluation',
                                'w116_pass_effect_is_phase_C_minus_A', 'w116_phase_C_dso_resolve')}) +
                            '; frozen: benchmark.w160_additions.no_reverse_flow_definition.consistency_reevaluation'),
            'consistency_found': v_where,
            'warm': ('warm_from_certified = values extracted from BENCH.COORDINATED certified_models = the settled cycle-181 x = 0 '
                     'cell d110bd1a (evaluation 1); confirmed. Code lines: ' + json.dumps({k: needles[k]['lines'] for k in (
                         'w116_warm_from_certified_models', 'w116_certified_is_coordinated', 'w93_coordinated_is_settled', 'w93_settled_eval_key')})),
            'warm_found': v_warm,
            'warm_report_v3': needles['report_v3_warm_source_is_settled_x0'],
            'phases_of_the_other_columns': ('Q (E.23) is the phase-C cost for every start of both arms '
                                            f'(arm_cost_source_by_start; all phase C: {v_all_phase_c}); the curtailment columns are '
                                            'phase A (curtailment_phase_A_by_start; code: curtailment_rows_are_phase_A). The E.23 caption '
                                            'states neither. ' + json.dumps({k: needles[k]['lines'] for k in (
                                                'w116_arm_cost_is_phase_C', 'w116_curtailment_rows_phase_A')})),
            'sources': v_sources,
            'suggested_reading': ('"The consistency check re-solves each distribution network at the transmission network\'s '
                                  'interface voltage with the no-reverse-flow rule relaxed; the exceedance columns count the hours '
                                  'of that re-solve (per network, year and representative day) above the limit and give the largest; '
                                  'any exceedance triggers the consistency pass, in which each distribution network is re-solved '
                                  'with the rule at that voltage and the transmission network re-dispatched; the Consistency column '
                                  'is the change in Q from that pass, and Q is the cost after it." (Worker suggestion; wording is '
                                  'the expert\'s)')}}
    vi_ok = (D['acc_ok'] and needles['acceptable_clean_is_block_exit']['found'] and needles['w153_events_sum_blocks']['found']
             and needles['sc5_acceptable_primary_within_factor']['found'] and needles['sc5_one_block_final_exit']['found'])
    I['(vi) 119 exits at the acceptable level, all in distribution blocks'] = {
        'outcome': 'confirmed' if vi_ok else 'corrected',
        'evidence': {'w153': D['acc'], 'unit': ('an event is one (cycle, block) whose final IPOPT exit was Acceptable on the primary '
                                                'attempt within 10x (reason acceptable_primary_within_factor), summed over the blocks of '
                                                'each cycle after N over the 42 cells -- i.e. one local-solve exit; 109 in DSO|5|2035|Winter '
                                                'and 10 in DSO|7|2025|Winter, in two evaluations (5: d_f759dd48 114; 22: c_156ce2d1 5)'),
                     'code': {k: needles[k]['lines'] for k in ('acceptable_clean_is_block_exit', 'acceptable_clean_reason',
                                                               'w153_events_sum_blocks', 'sc5_acceptable_primary_within_factor',
                                                               'sc5_one_block_final_exit')},
                     'scope': 'the 42 evaluations of the certification campaign, cycles after the first residual pass N',
                     'evals': {inv[c]: n for c, n in D['acc']['cells'].items()}}}
    I['(vii) continued cells and movements'] = {
        'outcome': ('corrected (the set): the movements of 14 (0.74 tau) and 3 (0.78-0.88 tau) are confirmed, but a third evaluation, '
                    '12, was also continued past its certificate (0.004 tau); "the two evaluations" should read three, and the '
                    'sec. 4.7 bound "at most 0.9 tau" still holds') if (third and movements_ok) else
                   ('confirmed' if movements_ok else 'corrected'),
        'evidence': dict(D['continued'], W142_item3_scope=J['W142_V6_FROM_RECORDS']['item3_movement_record']['scope'],
                         note=('W142 item 3 searched only the 18 re-settle records of its floor replay and lists 14 and 3; W162/W163 '
                               'N25 (later, committed) adds the v3 run of 12 (b_2a0ba8b2, 213 cycles, uncertified under v3) past the '
                               'certificate at 173 that v4 - and v6 on the record (W142 item 1 b_2a0ba8b2@v4) - give the same '
                               'trajectory; structurally the same case as 14 (v4 run to 193 past the v5/v6 k* 148)'))}
    n49 = D['three_refs']['W163_N49'] or {}
    viii_found = all(needles[k]['found'] for k in ('a53_three_references_replayed', 'a53_old_vs_new_cert_x0',
                                                   'a53_old_vs_new_cert_unit', 'a53_c_star_none'))
    I['(viii) the three single-scenario reference continuations'] = {
        'outcome': ('corrected (wording): the three are evaluations 1 and 2 and C*, which is not in Appendix E; their "—" in the '
                    'Replay column agrees with T2, but they replayed to the cycles at which the residual-based rule had '
                    'certified them (132, 112, 87), not to the certification cycles k* that Appendix E prints (181, 172)')
                   if (viii_found and D['r2_wording_mismatch'] and D['three_refs_n'] == 3) else 'confirmed',
        'evidence': {'cells': 'x0 (evaluation 1), the unit n7_4h_e1 (evaluation 2) and C* (not in Appendix E; new certification none, cap 187, per the Addendum 53 report)',
                     'replayed_through': {k: v.get('through') for k, v in n49.items()},
                     'T2_replay_field': D['three_refs']['E21_replay_printed'],
                     'meaning': ('they replayed bitwise to the cycles at which the residual-based rule had certified them '
                                 '(x0 132, unit 112, C* 87; "old certification cycle", Addendum 53 report) -- not to k* of the rule '
                                 '(181, 172; C* none); T2 carries no replay field for evaluations 1 and 2 (None), so E.21 prints ---, '
                                 'as the expert stated'),
                     'needles': {k: needles[k]['lines'] for k in ('a53_three_references_replayed', 'a53_old_vs_new_cert_x0',
                                                                    'a53_old_vs_new_cert_unit', 'a53_c_star_none')},
                     'source': f"{INPUTS['W163_CHECK']} N49 (W101 G19/G13); {INPUTS['A53_REPORT']} line {needles['a53_three_references_replayed']['lines']}"}}
    I['(ix) k0 = 123 for x = 0'] = {'outcome': 'confirmed' if (D['x0_k0'] == 123 and D['fig_k0_line'] and ALL['ref:7aa017f0'].get('k0_run') is None) else 'corrected',
                                    'evidence': {'W168_decision_primary_N_k0_kstar': [w168['decision_primary']['N'], w168['decision_primary']['k0'],
                                                                                       w168['decision_primary']['k_star']],
                                                 'T2_k0_run': ALL['ref:7aa017f0'].get('k0_run'),
                                                 'figure_caption_line': D['fig_k0_line'],
                                                 'derived': f"k* - k0 = {ALL['ref:7aa017f0']['k_star']} - 123 = {ALL['ref:7aa017f0']['k_star'] - 123} (sec. 4.7 l. 1694: '58 cycles after')"}}
    n_corr = sum(1 for v in I.values() if v['outcome'].startswith('corrected'))
    n_tab_mm = sum(s['MISMATCH'] for s in summ.values())
    n_tab_nf_dis = sum(s['NOT_FROZEN_DISAGREES'] for s in summ.values())
    preds = [{'prediction': '(a) 0 MISMATCH in table entries', 'outcome': 'HELD' if n_tab_mm == 0 and n_tab_nf_dis == 0 and not (p20 or p21) else 'FAILED',
              'detail': (f'{n_tab_mm} MISMATCH over {sum(s["entries"] for s in summ.values())} entries '
                         f'({sum(s["MATCH"] for s in summ.values())} MATCH on frozen fields; '
                         f'{sum(s["NOT_FROZEN_AGREES"] for s in summ.values())} not-a-frozen-field entries agree with their named source, '
                         f'{n_tab_nf_dis} disagree); mapping problems {len(p20) + len(p21)}')},
             {'prediction': 'at most two of (i)-(ix) corrected, most likely (v) and (vi)',
              'outcome': 'HELD' if n_corr <= 2 else 'FAILED',
              'detail': f"{n_corr} corrected: {[k for k, v in I.items() if v['outcome'].startswith('corrected')]}"}]
    guards, pk, guards_ok = guards_state()
    failed = [k for k, v in checks.items() if v is not True]
    code = 0 if (not failed and guards_ok) else 3
    out = {'stage': 'P5.15 Addendum 78 W179a -- Appendix E vs the frozen tables (zero-solve)', 'script': SCRIPT_REL,
           'script_sha256': _sha(SCRIPT_REL), 'git_head': _git('rev-parse', 'HEAD'),
           'utc': datetime.now(timezone.utc).isoformat(), 'clone_head': head, 'main_sha256': main_sha,
           'objective_convention': 'Q = gross_operational_cost, settlement excluded; net = Q - terminal salvage credit; value = Q(0) - Q(x); F = Q + I',
           'inputs': inputs, 'rounding_rule': 'MATCH iff |printed - value| <= half a unit of the last printed digit (+1e-9); k EUR = EUR / 1000',
           'e21_map': {str(k): v for k, v in sorted(idmap.items())}, 'conditions': {k: dict(v, ageing=list(v['ageing'])) for k, v in COND.items()},
           'table_summary': summ, 'mismatches': mismatches, 'E20': e20_out, 'E21': e21_out, 'E22': e22_out, 'E22_note': e22_note,
           'E23': e23_out, 'E23_note': e23_note, 'E24': e24_out, 'mapping_60_to_55_plus_3': mapping_i,
           'salvage_over_the_55': salvage_55, 'prose': P, 'interpretations': I, 'predictions': preds, 'needles': needles,
           'derived': {k: v for k, v in D.items() if k not in ('w163',)}, 'checks': checks, 'failed': failed,
           'guards': guards, 'pickle_guard': pk, 'guards_ok': guards_ok, 'exit_code': code, 'wall_s': time.time() - t0}

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, set)):
            return [_clean(v) for v in o]
        return o
    out = _clean(out)
    written = {}

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
    wr('man', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    for t, s in summ.items():
        _log(f'[{tag}] {t}: rows {s["rows"]} entries {s["entries"]} MATCH {s["MATCH"]} MISMATCH {s["MISMATCH"]} '
             f'not-a-frozen-field agrees {s["NOT_FROZEN_AGREES"]} disagrees {s["NOT_FROZEN_DISAGREES"]}')
    for m in mismatches:
        _log(f'[{tag}] MISMATCH {m}')
    for p in P:
        if p['status'] != 'MATCH':
            _log(f"[{tag}] PROSE {p['id']} {p['status']}: {p['written']} -> {str(p['value'])[:300]}")
    _log(f'[{tag}] salvage over the 55: {salvage_55}')
    for k, v in I.items():
        _log(f"[{tag}] {k}: {v['outcome']}")
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
