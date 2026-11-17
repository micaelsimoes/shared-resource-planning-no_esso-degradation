"""P5.15 Addendum 66, Planner task W162 -- FIGURE CHECK OF `paragraphs_v3.md` AGAINST THE FROZEN TABLES AND NAMED
COMMITTED RECORDS. ZERO SOLVES, NO MODEL LOADS. A NEW FILE: the W160 and W161 scripts are IMPORTED and their functions
CALLED (`W160.figure_checks`, `W161.v2_definitions`, `W161.evaluate`); neither is edited. Nothing earlier is modified:
every output is a new file opened 'x' in a new directory.

SCOPE. Every number in the v3 prose above "## Unchanged from paragraphs_v2.md": sections (ii)-(iv), the draft and
supplementary notes, and the four accepted sentences. The title line and the Planner's HTML comment block are
inventoried too, as NON-MANUSCRIPT text, and checked separately. The text is NOT edited.

WHAT IT DOES
  1. Integrity. paragraphs_v3.md (committed at cb0a0bc6) is pinned by sha256, so the line numbers below are stable; the
     frozen JSON (590088fe) and every record read are sha-recorded and must be committed clean.
  2. Carries the W161 checks over. The W161 v2 check set (164 checks) is re-derived exactly as W161 built it -- the
     W160 checks (`W160.figure_checks`, re-run now on the v3 text) with the W161 replacements (`W161.v2_definitions`)
     -- and every re-derived counterpart is asserted equal to the one the committed W161 JSON stores (same quantity,
     same value). For each check: if its fragment is in the v3 text verbatim, its status is the one recomputed on v3;
     if v3 rewrote the sentence (CARRY below), the same counterpart is evaluated against the figure v3 writes, by the
     W160/W161 rules (the evaluator is self-tested on the v2 values of every carried check and must reproduce W161's
     status and shown value); otherwise the figure is not in the v3 prose (sources tables, reviewer map, scorecard,
     or a sentence v3 dropped) and is listed as such.
  3. Checks the figures new in v3 (N-ids) against the named records, code constants and specs (each check names its
     counterpart and its kind: frozen table / named record / code or spec constant / record text / environment now).
  4. Definition checks (D-ids, no figure): the settling-slack definition (scorer behaviour + spec text + recomputed
     s on every uncertified cell), every uncertified evaluation gated with an earlier certificate, every repeated
     evaluation bitwise to k0, and two wording observations (the residual-pass clause, clause 5's reads).
  5. Token inventory: every numeric token (digits, superscript powers, versions, dates, number words, the product
     tokens MA97 / MA57 / M2 Max) in the scope is located (line, text, occurrence) and must be covered by a check or by
     an explicit "unchecked" entry with its category; an unassigned token or a stale assignment FAILS the run.

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, with every guard the W161 / W160 imports arm; `pickle.load` / `pickle.loads` blocked for the whole run, every
counter verified at 0. Environment reads (`sysctl`) are plain subprocess calls, not solves.

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w162_paragraphs_v3_check && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w162_paragraphs_v3_check.py \\
        > data/SRP1/Results/P515S53/w162_paragraphs_v3_check/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w162_paragraphs_v3_check.py --post-run
Exit: 0 = written, every integrity check holds and the guards are at 0 (figure MISMATCHES are FINDINGS and never change
the exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard fault.
"""
import argparse
import inspect
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W162 paragraphs v3 figure check (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W162: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W162: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w161_paragraphs_v2 as W161  # noqa: E402 -- arms its guards and imports W160 (neither edited)
import settling_criterion as SC1  # noqa: E402
import settling_criterion_v2 as SC2  # noqa: E402
import settling_criterion_v5 as SC5  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

W160 = W161.W160
L132 = W160.W157.L132          # p515_s53_w132_resettle_v3_campaign (view_from_report, resolve)
GUARDS = W160.W157._dedupe((('w162_paragraphs_v3_check', GUARD),) + tuple(W161.GUARDS))

_log = W160._log
_sha = W160._sha
_sha_bytes = W160._sha_bytes
_dp = W160._dp
_wval = W160._wval

SCRIPT_REL = os.path.basename(__file__)
S53 = W160.S53
EXPORT = os.path.join(S53, 'w160_step6_frozen', 'export')
P3_REL = os.path.join(EXPORT, 'paragraphs_v3.md')
P3_SHA = '62d26e275a3c65a7990c2ec8fd0d30050ad5a8fef32513424415eb43c9682673'
P3_COMMIT = 'cb0a0bc6'
P2_REL = os.path.join(EXPORT, 'paragraphs_v2.md')
P2_SHA = '73b42c49cebc1cd9ae8515ad0803d69b63cf818e6030bfa898f77e0d70506a50'
P2_COMMIT = '2a1d7f92'
FZ_REL = W161.FZ_REL
FZ_SHA = W161.FZ_SHA
W161_JSON = W161.OUT_JSON
W153C_REL = W161.W153C_REL
W153D_REL = W161.W153D_REL
W153_MAN = W161.W153_MAN
V6_SPEC = os.path.join(S53, 'w142_resettle_v6', 'frozen_s53_resettle_spec_v6_96c23404.json')
V6_SHA = '96c234045f15fa4f18ba7149b3b4a39c749f256657d436064c320b04d546e041'
V41_SPEC = os.path.join(S53, 'frozen_s53_spec_v41_fcea4b38.json')
W118_SPEC = os.path.join(S53, 'w118_resettle', 'frozen_s53_resettle_spec_v2_fc791891.json')
W118_SUMMARY = os.path.join(S53, 'w118_resettle', 'w118_resettle_summary.json')
W101_DIR = os.path.join(S53, 'w101_srp1_continuation')
W101_SUMMARY = os.path.join(W101_DIR, 'w101_three_reference_summary.json')
W86_RESULTS = os.path.join(S53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'campaign_results.json')
W98_RESULTS = os.path.join(S53, 'w98_continuation', 'campaign_s53_w98_x0_continuation_r2', 'campaign_results.json')
W98_EVAL = os.path.join(S53, 'w98_continuation', 'campaign_s53_w98_x0_continuation_r2', 'evals', '25b92ae0f1f2c02e_x0_cont',
                        'evaluation_record.json')
W109_JSON = os.path.join(S53, 'w109_tso_curtailment', 'w109_tso_curtailment_look.json')
W141_JSON = os.path.join(S53, 'w141_swing_variants', 'w141_swing_variants.json')
W137_CELL3 = os.path.join(S53, 'w137_resettle_v4', 'campaign_s53_w137_resettle_v4_b_4649234b', 'evals',
                          'edc95b9bc4e5e395_b_4649234b', 'per_cycle_record.jsonl')
W132_CELL1 = os.path.join(S53, 'w132_resettle_v3', 'campaign_s53_w132_resettle_v3_b_2a0ba8b2', 'evals',
                          'cf592cc94ce0d1ef_b_2a0ba8b2', 'per_cycle_record.jsonl')
W139_C52_DIR = os.path.join(S53, 'w139_resettle_v5', 'campaign_s53_w139_resettle_v5_d_c52e1670')
W139_C52_REC = os.path.join(W139_C52_DIR, 'evals', 'af42a163b14f0895_d_c52e1670', 'per_cycle_record.jsonl')
W139_C52_RES = os.path.join(W139_C52_DIR, 'campaign_results.json')
W159_JSON = os.path.join(S53, 'w159_closing_reads', 'w159_closing_reads.json')
W155_SOH050 = os.path.join(S53, 'w155_a64_cells', 'campaign_s53_w155_a64_r2_e_soh050',
                           'campaign_spec_s53_w155_a64_r2_e_soh050_28c9478d.json')
W155_M175 = os.path.join(S53, 'w155_a64_cells', 'campaign_s53_w155_a64_r2_h_unit_m175',
                         'campaign_spec_s53_w155_a64_r2_h_unit_m175_eb85d0f4.json')
SRP1_PARAMS = os.path.join('data', 'SRP1', 'SRP1_params.json')
CASE9_PARAMS = os.path.join('data', 'SRP1', 'case9', 'case9_params.json')
NOTE_5456 = 'P5_15_ADDENDUM54_56_CONSOLIDATED_NOTE.md'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
CODE_FILES = ('definitions.py', 'model_construction_helpers.py', 'shared_resources_planning.py', 'network.py',
              'p515_s44_campaign_harness.py', 'settling_criterion.py', 'settling_criterion_v2.py',
              'settling_criterion_v5.py', 'settling_criterion_v6.py', 'p515_s53_w132_resettle_v3_campaign.py',
              'p515_s53_w160_step6_freeze_export.py', 'p515_s53_w161_paragraphs_v2.py')

OUT_DIR = os.path.join(S53, 'w162_paragraphs_v3_check')
OUT_JSON = os.path.join(OUT_DIR, 'w162_paragraphs_v3_figure_check.json')
OUT_MD = os.path.join(OUT_DIR, 'w162_paragraphs_v3_figure_check.md')
OUT_MAN = os.path.join(OUT_DIR, 'manifest_sha256.json')
OUT_LOG = os.path.join(OUT_DIR, 'launch.log')
OUT_TYPING_JSON = os.path.join(OUT_DIR, 'w162_bool_typing_test.json')
OUT_TYPING_LOG = os.path.join(OUT_DIR, 'w162_bool_typing_test.log')
OUT_POST = os.path.join(OUT_DIR, 'manifest_post_run_sha256.json')

TAU = SC6.TAU


def _jl(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        return json.load(h)


def _rows(rel):
    out = {}
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        for ln in h:
            if ln.strip():
                x = json.loads(ln)
                out[x['cycle']] = x
    return out


def _text(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        return h.read()


# ======================================================================================================================
#  the text: scope, normalisation, token inventory
# ======================================================================================================================
TOK = re.compile(
    r'(?P<date>\d{4}-\d{2}-\d{2})'
    r'|(?P<ver>(?<![\w.])\d+\.\d+\.\d+(?![\w.]))'
    r'|(?P<sci>(?<![\w.])10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)'
    r'|(?P<num>(?<![\w.,])[+−-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![A-Za-z0-9_])'
    r'|(?<![\w.,])[+−-]?\d+(?:\.\d+)?(?![A-Za-z0-9_.]\d|[A-Za-z0-9_]))'
    r'|(?P<word>\b(?:[Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine|[Tt]en|[Ss]ingle|[Tt]hird)\b)'
    r'|(?P<prod>\bM2 Max\b|\bMA97\b|\bMA57\b)')
HEX = re.compile(r'\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,}\b')
SCOPE_END = '## Unchanged from paragraphs_v2.md'
MANUSCRIPT_FROM_LINE = 10          # "## (ii) Certification paragraph"; lines 1-8 are the title and the Planner's comment


def scope_lines(raw):
    lines = raw.split('\n')
    end = next(i for i, ln in enumerate(lines) if ln.startswith(SCOPE_END))
    return lines, end


def inventory(lines, end):
    toks = []
    for i in range(end):
        masked = HEX.sub(lambda m: '#' * len(m.group()), lines[i])
        seen = {}
        for m in TOK.finditer(masked):
            t = m.group()
            occ = seen.get(t, 0)
            seen[t] = occ + 1
            toks.append({'line': i + 1, 'text': t, 'occ': occ, 'col': m.start(), 'type': m.lastgroup,
                         'region': 'non-manuscript (title / Planner comment)' if i + 1 < MANUSCRIPT_FROM_LINE
                         else 'scope'})
    return toks


def normalise(raw_lines, end):
    """Join the scope into one string: blockquote markers and list-continuation indentation removed, whitespace
    collapsed (so a fragment wrapped across lines is still found)."""
    out = []
    for ln in raw_lines[:end]:
        s = ln
        if s.startswith('>'):
            s = s[1:]
        out.append(s.strip())
    return re.sub(r'\s+', ' ', ' '.join(out))


def norm_frag(f):
    f = f.strip()
    if f.startswith('>'):
        f = f[1:]
    return re.sub(r'\s+', ' ', f.strip())


# ======================================================================================================================
#  evaluation helpers
# ======================================================================================================================
def at_prec(written, value):
    """(shown, status): `value` at the written precision, compared numerically."""
    dp = _dp(written)
    shown = f'{value:.{dp}f}'
    return shown, 'match' if float(shown) == _wval(written) else 'MISMATCH'


def eval_kind(kind, written, value, absval=False):
    """The W160 rules: eur/keur/pct/raw/split by `W161.evaluate`; near/set/verdict as W160's figure_checks."""
    if kind in ('eur', 'keur', 'pct', 'raw', 'split'):
        return W161.evaluate(kind, written, value, absval)
    if kind == 'near':
        shown = f'{value[0]:.4f}–{value[1]:.4f}'
        return shown, 'match' if all(abs(x - _wval(written)) <= 0.01 + 1e-12 for x in value) else 'MISMATCH'
    if kind == 'set':
        return '/'.join(value), 'match' if value == [written] else 'MISMATCH'
    if kind == 'verdict':
        return value, 'match' if value == written else 'MISMATCH'
    raise ValueError(kind)


# ======================================================================================================================
#  2. the W161 carry-over
# ======================================================================================================================
# W161 checks whose sentence v3 rewrote: (normalised v3 fragment, the figure as v3 writes it, kind, note)
CARRY = {
    'C4': ('Ten of the 46 certificates the tables use stop within 5 % of τ', '10', 'raw',
           'v3 writes the word "Ten"; the v2 sentence had no N'),
    'C5': ('Of the 42 SRP1 evaluations run under this rule, 32 certified (24 oscillatory, 8 monotone)', '42', 'raw', None),
    'C6': ('Of the 42 SRP1 evaluations run under this rule, 32 certified (24 oscillatory, 8 monotone)', '32', 'raw', None),
    'C7': ('Of the 42 SRP1 evaluations run under this rule, 32 certified (24 oscillatory, 8 monotone)', '24', 'raw', None),
    'C8': ('Of the 42 SRP1 evaluations run under this rule, 32 certified (24 oscillatory, 8 monotone)', '8', 'raw', None),
    'C9': ('divide as follows: 5 refused by the gap clause, 2 reset by residual lapses, 3 failed by the growth test', '5',
           'raw', 'the W160 fragment wraps across a line in v3'),
    'C10': ('divide as follows: 5 refused by the gap clause, 2 reset by residual lapses, 3 failed by the growth test', '2',
            'raw', None),
    'C11': ('divide as follows: 5 refused by the gap clause, 2 reset by residual lapses, 3 failed by the growth test', '3',
            'raw', None),
    'C13': ('certification came a median 65 cycles after the first residual pass (range 21–89)', '65', 'raw',
            'the W161 fragment wraps across a line in v3'),
    'C14': ('certification came a median 65 cycles after the first residual pass (range 21–89)', '21', 'raw', None),
    'C15': ('certification came a median 65 cycles after the first residual pass (range 21–89)', '89', 'raw', None),
    'C17': ('The four further evaluations all certified', '4', 'raw', 'v3 writes the word "four"'),
    'C33': ('number 21 cycles in 10 evaluations, all in transmission blocks', '21', 'raw',
            'v2 had this figure in the sources table only; v3 writes it in the limitations paragraph'),
    'C34': ('number 21 cycles in 10 evaluations, all in transmission blocks', '10', 'raw', None),
    'L6': ('the power-flow primal residual, as a fraction of its tolerance, stuck near 0.69', '0.69', 'near', None),
}

# token assignments: check id -> [(line, token text, occurrence on that line)]
TOKENS = {
    # carried W161 checks
    'C1': [(40, '4,539.07', 0)], 'C2': [(24, '453.91', 0)], 'C3': [(28, '2,269.53', 0)],
    'C4': [(42, 'Ten', 0)], 'C5': [(60, '42', 0)], 'C6': [(60, '32', 0)], 'C7': [(60, '24', 0)],
    'C8': [(60, '8', 0)], 'C9': [(61, '5', 0)], 'C10': [(61, '2', 0)], 'C11': [(61, '3', 0)],
    'C12': [(62, '174', 0)], 'C13': [(62, '65', 0)], 'C14': [(63, '21', 0)], 'C15': [(63, '89', 0)],
    'C16': [(63, '0.86', 0)], 'C17': [(63, 'four', 0)], 'C33': [(84, '21', 0)], 'C34': [(84, '10', 0)],
    'L1': [(75, '−8.8', 0)], 'L2': [(75, '−9.4', 0)], 'L3': [(76, '55', 0), (76, '5', 0)],
    'L4': [(76, '31', 0), (76, '9', 0)], 'L5': [(76, '14', 0), (76, '7', 0)], 'L6': [(77, '0.69', 0)],
    'L7': [(80, '+1.1', 0)], 'L9': [(93, '1.04', 0)], 'L10': [(93, '1.85', 0)], 'L11a': [(94, '0.41', 0)],
    'L11b': [(94, '0.62', 0)], 'L11c': [(94, 'three', 0)],
    'S1a': [(131, '+14.2', 0)], 'S1b': [(131, '+46.3', 0)], 'S1c': [(132, '1.63', 0)], 'S2a': [(133, '+0.18', 0)],
    'S2b': [(134, '+4.9', 0)], 'S2c': [], 'S2d': [(134, '59.5', 0)], 'S2e': [], 'V1a': [(136, '−4.1', 0)],
    'V1b': [], 'V1c': [(137, '31.6', 0)], 'V1d': [(137, '73.7', 0)],
}


def w161_rederived(fz, v3_raw, w153c):
    """The W161 v2 check set, re-derived exactly as W161 built it, with the W160 part re-run on the v3 text."""
    fresh = json.loads(GRIO.dumps(W160.figure_checks(fz, v3_raw, w153c)))
    R, self_test, repl, m2cells, e050 = W161.v2_definitions(fz, fz['paragraph_figure_checks'])
    out = []
    for c in fresh:
        if c['id'] in R:
            for dd in R[c['id']]:
                shown, status = W161.evaluate(dd['kind'], dd['written'], dd['value'], dd['absval'])
                out.append({'id': dd['id'], 'origin_w161': 'W161 replacement' if dd['id'] == dd['replaces']
                            else f"W161 new figure (with the replacement of {dd['replaces']})",
                            'fragment_w161': dd['fragment'], 'written_v2': dd['written'], 'kind': dd['kind'],
                            'absval': dd['absval'], 'table_ref': dd['ref'], 'value': dd['value'],
                            'fragment_found_raw_v3': dd['fragment'] in v3_raw,
                            'status_on_v2_written': status, 'shown_on_v2_written': shown})
        else:
            out.append({'id': c['id'], 'origin_w161': 'W160 check, unchanged', 'fragment_w161': c['fragment_verbatim'],
                        'written_v2': c['written'], 'kind': None, 'absval': False, 'table_ref': c['table_ref'],
                        'value': c['table_value_full_precision'],
                        'fragment_found_raw_v3': c['fragment_found_in_paragraphs_md'],
                        'status_w160_on_v3_text': c['status'],
                        'shown_w160_on_v3_text': c['table_value_at_written_precision']})
    return out, self_test, repl


def carry_over(fz, v3_raw, v3_norm, w153c, w161doc):
    derived, w161_self_test, repl = w161_rederived(fz, v3_raw, w153c)
    stored = {c['id']: c for c in w161doc['figure_check_v2']}
    same, rows, self_test = [], [], {}
    for d in derived:
        s = stored.get(d['id'])
        eq = s is not None and s['written'] == d['written_v2'] and \
            json.dumps(s['table_value_full_precision'], sort_keys=True) == json.dumps(d['value'], sort_keys=True)
        same.append((d['id'], eq))
        row = {'id': d['id'], 'origin_w161': d['origin_w161'], 'table_ref': d['table_ref'],
               'table_value_full_precision': d['value'], 'w161_status': s['status'] if s else None,
               'same_counterpart_as_w161': eq, 'fragment_w161': d['fragment_w161'], 'written_v2': d['written_v2']}
        if d['fragment_found_raw_v3']:
            if d['origin_w161'] == 'W160 check, unchanged':
                status, shown = d['status_w160_on_v3_text'], d['shown_w160_on_v3_text']
            else:
                status, shown = d['status_on_v2_written'], d['shown_on_v2_written']
            row.update({'placement': 'verbatim in v3', 'fragment_v3': d['fragment_w161'], 'written_v3': d['written_v2'],
                        'status': status, 'table_value_at_written_precision': shown,
                        'fragment_found_v3': True})
        elif d['id'] in CARRY:
            frag, w3, kind, note = CARRY[d['id']]
            shown, status = eval_kind(kind, w3, d['value'], d['absval'])
            # self-test: the same evaluator on the v2 written value reproduces W161's stored status and shown value
            s_sh, s_st = eval_kind(kind, d['written_v2'], d['value'], d['absval'])
            self_test[d['id']] = {'kind': kind, 'w161_status': s['status'], 'w161_shown':
                                  s['table_value_at_written_precision'], 'w162_status': s_st, 'w162_shown': s_sh,
                                  'reproduces': s_st == s['status'] and str(s_sh) == str(
                                      s['table_value_at_written_precision'])}
            row.update({'placement': 'v3 rewrote the sentence (carried to the v3 fragment)', 'fragment_v3': frag,
                        'written_v3': w3, 'kind': kind, 'status': status, 'table_value_at_written_precision': shown,
                        'fragment_found_v3': frag in v3_norm, 'note': note})
        else:
            row.update({'placement': 'not in the v3 prose', 'status': 'not in v3 prose', 'fragment_found_v3': False})
        row['tokens'] = [list(t) for t in TOKENS.get(d['id'], [])]
        rows.append(row)
    return rows, {'all_same_counterpart': all(e for _, e in same), 'not_same': [i for i, e in same if not e],
                  'n_w161_checks': len(derived), 'n_w161_stored': len(stored),
                  'w161_evaluator_self_test_on_v1_reproduces': all(v['reproduces'] for v in w161_self_test.values()),
                  'w162_evaluator_self_test_on_v2': self_test,
                  'w162_evaluator_self_test_all_reproduce': all(v['reproduces'] for v in self_test.values()),
                  'replication_of_w160_sets': repl}


# ======================================================================================================================
#  3. the new figures (N) and the non-manuscript identifiers (X)
# ======================================================================================================================
def _chk(cid, tokens, written, counterpart, value, shown, status, kind, fragment, note=None):
    return {'id': cid, 'origin': 'W162 new check (v3)', 'tokens': [list(t) for t in tokens], 'written_v3': written,
            'counterpart': counterpart, 'counterpart_kind': kind, 'value': value,
            'value_at_written_precision': shown, 'status': status, 'fragment_v3': fragment, 'note': note}


def new_checks(fz, v3_norm, rec, envnow):
    t = fz['tables']
    cells = t['cells']
    allc = dict(cells)
    allc.update(t['cells_appended_w160'])
    v6 = rec['v6']
    sr = v6['stop_rule']
    C = []
    add = C.append
    # ---- (ii) residual test --------------------------------------------------------------------------------------
    boyd = rec['srp1_params']['admm']['tol']['boyd']
    srp = rec['code']['shared_resources_planning.py']
    cite = 'Boyd et al. 2011 Sec. 3.3.1' in srp
    add(_chk('N1', [(13, '2011', 0), (13, '3.3', 0)], 'Boyd et al. 2011, §3.3', 'shared_resources_planning.py comment '
             '"P5.15 Step 3.2 (Boyd et al. 2011 Sec. 3.3.1): the stopping test is the Boyd primal/dual residual test"',
             'Boyd et al. 2011 Sec. 3.3.1' if cite else None, '2011, Sec. 3.3.1 (within §3.3)',
             'match' if cite else 'MISMATCH', 'code comment (citation)', '[Boyd et al. 2011, §3.3]',
             'the code cites the subsection 3.3.1 of the same section'))
    add(_chk('N2', [(13, '10⁻⁵', 0)], '10⁻⁵', 'data/SRP1/SRP1_params.json admm.tol.boyd.eps_abs', boyd['eps_abs'],
             f"{boyd['eps_abs']:g}", 'match' if boyd['eps_abs'] == 1e-5 else 'MISMATCH', 'case file constant',
             'ε_abs = 10⁻⁵'))
    add(_chk('N3', [(14, '10⁻⁴', 0)], '10⁻⁴', 'data/SRP1/SRP1_params.json admm.tol.boyd.eps_rel', boyd['eps_rel'],
             f"{boyd['eps_rel']:g}", 'match' if boyd['eps_rel'] == 1e-4 else 'MISMATCH', 'case file constant',
             'ε_rel = 10⁻⁴'))
    # ---- 15-21 kEUR after the residuals first passed -------------------------------------------------------------
    w101 = rec['w101_summary']['reports']
    beside = rec['reference_beside']
    sx, su = w101['x0']['s_signed'], w101['n7_4h_e1']['s_signed']
    sh15, st15 = at_prec('15', sx / 1000.0)
    sh21, st21 = at_prec('21', su / 1000.0)
    note15 = ('"the reference evaluations" = the settled x0 (d110bd1a) and unit (3f084f2f), the two certified W101 '
              'references; C* (uncertified, s = −4,378.88) is outside the range. The figures are s = Q(k*) − Q(N) '
              'measured from the OLD certificate N (x0 132, unit 112), which is k0 + 9 (first residual pass 123 / 103); '
              f"measured from k0 itself the net movement is {beside['x0']['Q_end_minus_Q_k0']:+,.2f} (x0) and "
              f"{beside['n7_4h_e1']['Q_end_minus_Q_k0']:+,.2f} (unit)")
    add(_chk('N4', [(17, '15', 0)], '15', 'W101 three-reference summary (62bdeafe) reports.x0.s_signed (k€)', sx, sh15,
             st15, 'named record', 'moved by a further 15–21 k€ after the residuals first passed', note15))
    add(_chk('N5', [(17, '21', 0)], '21', 'W101 three-reference summary (62bdeafe) reports.n7_4h_e1.s_signed (k€)', su,
             sh21, st21, 'named record', 'moved by a further 15–21 k€ after the residuals first passed', note15))
    # ---- tail ------------------------------------------------------------------------------------------------------
    cit = v6['inputs_in_force_now']['configuration_now']['convergence_depth_tail']['compl_inf_tol']
    add(_chk('N6', [(18, '10⁻⁶', 0)], '10⁻⁶', 'v6 spec inputs_in_force_now.configuration_now.convergence_depth_tail.'
             'compl_inf_tol', cit, f'{cit:g}', 'match' if cit == 1e-6 else 'MISMATCH', 'spec constant',
             'complementarity tolerance 10⁻⁶'))
    # ---- rule clauses ------------------------------------------------------------------------------------------------
    src2 = inspect.getsource(SC2.SettlingRuleV2.evaluate)
    src6 = inspect.getsource(SC6.SettlingRuleV6.evaluate)
    three = "'at_least_3_turning_points': len(T) >= 3" in src2 and 'SC2.SettlingRuleV2.evaluate(self, k)' in src6
    add(_chk('N7', [(22, 'three', 0)], 'three', 'settling_criterion_v2.SettlingRuleV2.evaluate: at_least_3_turning_points '
             '= len(T) >= 3 (called by settling_criterion_v6.SettlingRuleV6.evaluate)', 3 if three else None, '3',
             'match' if three else 'MISMATCH', 'code constant', 'at least three turning points (extrema) since k₀'))
    add(_chk('N8', [(24, '10', 0)], '10', 'settling_criterion_v6.SWING_FLOOR = TAU / 10 (= GROWTH_TEST_FLOOR = '
             'TURNING_POINT_FLOOR); v6 spec stop_rule.swing_floor.F', TAU / SC6.SWING_FLOOR, f'{TAU / SC6.SWING_FLOOR:.0f}',
             'match' if SC6.SWING_FLOOR == TAU / 10.0 and SC6.GROWTH_TEST_FLOOR == SC6.TURNING_POINT_FLOOR ==
             SC6.SWING_FLOOR and sr['swing_floor']['F'] == SC6.SWING_FLOOR else 'MISMATCH', 'code constant',
             'Swings smaller than τ/10 = 453.91 €'))
    add(_chk('N9', [(26, '20', 0)], '20', 'settling_criterion.W_MIN; v6 spec stop_rule.W.oscillatory', SC1.W_MIN,
             str(SC1.W_MIN), 'match' if SC1.W_MIN == 20 and 'max(20, ceil(1.1' in sr['W']['oscillatory'] else 'MISMATCH',
             'code constant', 'W = max(20, ⌈1.1 P̂⌉)'))
    add(_chk('N10', [(26, '1.1', 0)], '1.1', 'settling_criterion.W_FACTOR', SC1.W_FACTOR, f'{SC1.W_FACTOR}',
             'match' if SC1.W_FACTOR == 1.1 else 'MISMATCH', 'code constant', 'W = max(20, ⌈1.1 P̂⌉)'))
    # P_hat: source + a discriminating synthetic replay (period lengthening, so T[-1]-T[-3] != T[2]-T[0])
    phat_src = 'p_hat = T[-1][0] - T[-3][0]' in src2
    q, ph = {}, 0.0
    for k in range(1, 401):
        ph += 2 * math.pi / (14.0 + 0.08 * k)
        q[k] = 1e8 + 30000.0 * math.exp(-k / 70.0) * math.cos(ph)
    _recs, dec, _rule = SC6.replay(q, {k: True for k in q}, {k: 0.0 for k in q}, {k: True for k in q}, p_max=30,
                                   cap=400, cap_ceiling=400)
    T = dec.get('T') or []
    syn = {'status': dec.get('status'), 'k_star': dec.get('k_star'), 'P_hat': dec.get('P_hat'),
           'turning_points': [x[0] for x in T],
           'T_last_minus_T_third_last': (T[-1][0] - T[-3][0]) if len(T) >= 3 else None,
           'T_third_minus_T_first': (T[2][0] - T[0][0]) if len(T) >= 3 else None}
    syn['discriminating'] = syn['T_last_minus_T_third_last'] != syn['T_third_minus_T_first']
    syn['P_hat_is_third_last_to_last'] = syn['P_hat'] == syn['T_last_minus_T_third_last']
    ok = phat_src and syn['discriminating'] and syn['P_hat_is_third_last_to_last'] and \
        "T[-1].t - T[-3].t" in sr['W']['oscillatory']
    add(_chk('N11', [(27, 'three', 0), (27, 'third', 0)], 'the three most recent turning points: from the third-last '
             'to the last', 'settling_criterion_v2.SettlingRuleV2.evaluate: p_hat = T[-1][0] - T[-3][0] (inherited by '
             'v6); v6 spec stop_rule.W.oscillatory; synthetic v6 replay (stdlib, no solve)', syn,
             f"P_hat {syn['P_hat']} = T[-1]-T[-3] {syn['T_last_minus_T_third_last']} (T[2]-T[0] "
             f"{syn['T_third_minus_T_first']})", 'match' if ok else 'MISMATCH', 'code (behaviour)',
             'Here P̂ is the period measured over the three most recent turning points: the number of cycles from the '
             'third-last to the last',
             'Addendum 66 edit (d) reads "between the first and third turning points"; the code (and v3) use the '
             'three MOST RECENT turning points'))
    add(_chk('N12', [(28, '2', 0)], '2', 'settling_criterion_v2.GAP_BOUND = TAU / 2 (v6 GAP_BOUND)', TAU / SC6.GAP_BOUND,
             f'{TAU / SC6.GAP_BOUND:.0f}', 'match' if SC6.GAP_BOUND == TAU / 2.0 else 'MISMATCH', 'code constant',
             '|t_sum| ≤ τ/2 = 2,269.53 €'))
    cr = sr['clean_rule']
    okm = len(SC5.METRICS) == 4 and len(cr['metric_table']) == 4
    add(_chk('N13', [(30, 'four', 0)], 'four', 'settling_criterion_v5.METRICS (v6) and v6 spec stop_rule.clean_rule.'
             'metric_table', list(SC5.METRICS), str(len(SC5.METRICS)), 'match' if okm else 'MISMATCH', 'code constant',
             'all four IPOPT error metrics within 10× the tight-tail tolerances'))
    okf = SC6.CLEAN_FACTOR == 10.0 and cr['factor'] == 10.0 and SC5.PRIMARY_ATTEMPT == 'primary'
    add(_chk('N14', [(30, '10', 0)], '10', 'settling_criterion_v5.CLEAN_FACTOR (v6); v6 spec stop_rule.clean_rule.factor; '
             'PRIMARY_ATTEMPT', SC6.CLEAN_FACTOR, f'{SC6.CLEAN_FACTOR:g}', 'match' if okf else 'MISMATCH',
             'code constant', 'within 10× the tight-tail tolerances'))
    pm = sr['p_max']
    pv = [w101['x0']['P_hat'], w101['n7_4h_e1']['P_hat']]
    okp = pm['value'] == 30 and pm['L'] == 60 == 2 * pm['value'] and max(pv) == 30 and \
        sr['W']['monotone'] == 'L = L_MONO = 2 * P_MAX = 60'
    add(_chk('N15', [(32, '2', 0), (32, '60', 0)], '2 P_max = 60', 'v6 spec stop_rule.p_max.L and stop_rule.W.monotone '
             '("L = L_MONO = 2 * P_MAX = 60")', pm['L'], str(pm['L']), 'match' if okp else 'MISMATCH', 'spec constant',
             'no turning point within 2 P_max = 60 cycles'))
    add(_chk('N16', [(32, '30', 0)], '30', 'v6 spec stop_rule.p_max.value (source "W102 x0 P_hat 29, W103 unit P_hat '
             '30"); W101 summary P_hat x0 / unit', {'p_max': pm['value'], 'W101_P_hat': pv}, str(pm['value']),
             'match' if okp else 'MISMATCH', 'spec constant + named record',
             'where P_max = 30 cycles is the longest period measured on the instance',
             f'P_max = max of the measured P̂ on the two settled SRP1 references ({pv[0]}, {pv[1]}); Addendum 53 Ruling 1 '
             '"2× the longest period seen on the instance". Not the v1 p_max_from_records definition (22 at v39/v41).'))
    okl = 'abs(dQ_k) * L_MONO <= TAU' in json.dumps(sr['constants']) and pm['L'] == 60
    add(_chk('N17', [(36, '60', 0)], '60', 'v6 spec stop_rule.constants.MONOTONE_LAST_STEP_CLAUSE "abs(dQ_k) * L_MONO <= '
             'TAU" with L_MONO 60; settling_criterion_v2 last_step_times_L_le_tau', pm['L'], str(pm['L']),
             'match' if okl else 'MISMATCH', 'spec constant', '|last step| × 60 ≤ τ'))
    # ---- tau ---------------------------------------------------------------------------------------------------------
    okt = SC1.TAU == SC1.DELTA_R * SC1.R_REF / 4.0 and 'DELTA_R * R_REF / 4' in SC1.constants(30)['TAU']['formula']
    add(_chk('N18', [(38, '4', 0)], '4', 'settling_criterion.TAU = DELTA_R * R_REF / 4.0 (constants() formula)', 4.0,
             '4', 'match' if okt else 'MISMATCH', 'code constant', 'τ = δR·V/4'))
    add(_chk('N19', [(38, '0.07', 0)], '0.07', 'settling_criterion.DELTA_R; v6 spec stop_rule.constants.DELTA_R',
             SC1.DELTA_R, f'{SC1.DELTA_R}', 'match' if SC1.DELTA_R == 0.07 and
             sr['constants']['DELTA_R']['value'] == 0.07 else 'MISMATCH', 'code constant', 'δR = 0.07'))
    vold = rec['w101_summary']['predictions_scored']['expert_P2']['V_old']
    shv, stv = at_prec('259,375.33', SC1.R_REF)
    shv2, stv2 = at_prec('259,375.33', vold)
    add(_chk('N20', [(38, '259,375.33', 0)], '259,375.33', 'settling_criterion.R_REF; W101 summary expert_P2.V_old (the '
             'SRP1 value Q(x0, N) − Q(unit, N) at the certificates in force when v39 froze)', {'R_REF': SC1.R_REF,
             'V_old': vold}, f'{shv} / {shv2}', 'match' if stv == stv2 == 'match' else 'MISMATCH',
             'code constant + named record', 'V = 259,375.33 €'))
    ag = t['three_by_three']['rows']
    okr = any('V = Q(0) - Q(unit)' in r['quantity'] for r in ag) and any('V / V_SRP1_settled' in r['label'] for r in ag)
    add(_chk('N21', [(39, 'Four', 0)], 'Four', 'TAU divisor 4 (N18); T11 rows: V = Q(0) − Q(unit) (two evaluations per '
             'value) and R = V / V_SRP1 (two values): 2 × 2 = 4 evaluations', 4, '4', 'match' if okt and okr else
             'MISMATCH', 'code constant + frozen table definition', 'Four evaluations make up a ratio of two values'))
    add(_chk('N22', [(39, 'two', 0)], 'two', 'T11: R = V / V_SRP1_settled (a ratio of two values)', 2, '2',
             'match' if okr else 'MISMATCH', 'frozen table definition', 'a ratio of two values'))
    # ---- what tau bounds -----------------------------------------------------------------------------------------
    reg = t['at_or_above_0_95_tau']['registry']
    status_of = {k: v.get('status') for k, v in allc.items()}
    reg_cert = [r for r in reg if status_of.get(r['cell']) == 'certified']
    n_sup = sum(1 for r in reg_cert if r['superseded'])
    n_twin = sum(1 for r in reg_cert if r['twin_of'])
    n46 = len(reg_cert) - n_sup - n_twin
    unc_in_reg = [r['cell'] for r in reg if status_of.get(r['cell']) != 'certified']
    n_cert_all = sum(1 for v in allc.values() if v.get('status') == 'certified')
    add(_chk('N23', [(42, '46', 0)], '46', 'frozen tables.at_or_above_0_95_tau.registry: certified (status in T2 / '
             'appended cells) − superseded − bitwise twins', {'registry': len(reg), 'certified': len(reg_cert),
             'superseded': n_sup, 'twins': n_twin, 'uncertified_in_registry': unc_in_reg,
             'certified_cells_in_tables': n_cert_all}, str(n46), 'match' if n46 == 46 else 'MISMATCH', 'frozen table',
             'Ten of the 46 certificates the tables use',
             f'{len(reg_cert)} − {n_sup} − {n_twin} = {n46}; the registry holds every certified cell of the tables '
             f'({n_cert_all}); superseded: pb_y2025_n5; twins of ref:bd504ecf: e_c2_calfade, g070_neutrality'))
    fl = t['at_or_above_0_95_tau']['threshold']
    add(_chk('N24', [(42, '5', 0)], '5', 'frozen constants.FLAG_RANGE_OVER_TAU / registry threshold 0.95: 1 − 0.95',
             100 * (1 - fl), f'{100 * (1 - fl):.0f}', 'match' if abs(fl - 0.95) < 1e-12 and
             fz['constants']['FLAG_RANGE_OVER_TAU'] == fl else 'MISMATCH', 'frozen table constant',
             'stop within 5 % of τ', 'counted at range/τ ≥ 0.95'))
    pc = rec['postcert']
    mx = max(abs(v) for v in pc['over_tau'].values())
    add(_chk('N25', [(44, '0.9', 0)], '0.9', 'records: Q(last) − Q(k*) of the runs that continued past a settling-rule '
             'certificate of the same evaluation (cell 1 b_2a0ba8b2 v3 run past its v4 k* 173; cell 3 b_4649234b v4 run '
             'past its v5 k* 148; d_c52e1670 v5 run past its v6 k* 150, and past 148 on the last-pair reading); W141 '
             'd_c52e1670_detail', pc, f'max {mx:.3f} τ', 'match' if mx <= 0.9 else 'MISMATCH',
             'named record (recomputed)', 'Evaluations continued past certification moved by at most 0.9 τ',
             'a bound: every recomputed |movement| / τ is ≤ 0.9 (W142 / Addendum 61: 0.738 / 0.781 / 0.882; cell 1 '
             'adds 0.004). Scope: past SETTLING-RULE certificates. Runs continued past an old residual-only '
             'certificate moved more: the SRP1 references 15-21 k€ (N4/N5), the 3 × 3 x = 0 −4,492.39 (0.99 τ, W98)'))
    # ---- uncertified form, determinacy -------------------------------------------------------------------------------
    rv = L132.resolve(-50000.0, -50000.0, [{'status': 'uncertified', 'gap': 1000.0, 'slack': 2000.0},
                                          {'status': 'certified', 'band': 10.0}])
    ok3 = rv.get('bar') == 6000.0 and rv.get('bar_components') == [1000.0, 2000.0]
    add(_chk('N26', [(49, 'three', 0)], 'three', 'p515_s53_w132_resettle_v3_campaign.resolve (the scorer): bar = 3.0 × '
             'max(gap, slack) over the uncertified cell(s); behaviour on a synthetic view', rv.get('bar'),
             '3 × max(1000, 2000) = 6000', 'match' if ok3 else 'MISMATCH', 'code (behaviour)',
             'its margin exceeds three times the larger of two quantities'))
    add(_chk('N27', [(49, 'two', 0)], 'two', 'resolve: bar_components = [gap, slack] per uncertified cell',
             rv.get('bar_components'), str(len(rv.get('bar_components') or [])), 'match' if ok3 else 'MISMATCH',
             'code (behaviour)', 'the larger of two quantities'))
    thr = SC6.determinacy_threshold(100.0, 200.0)
    thr2 = SC6.determinacy_threshold(5000.0, 10.0)
    okd = SC6.DETERMINACY_BAR_FACTOR == 3.0 and SC6.DETERMINACY_TAU_MULTIPLE == 2.0 and thr == 2 * TAU and \
        thr2 == 15000.0
    add(_chk('N28', [(58, '3', 0)], '3', 'settling_criterion_v6.DETERMINACY_BAR_FACTOR; determinacy_threshold behaviour',
             SC6.DETERMINACY_BAR_FACTOR, '3', 'match' if okd else 'MISMATCH', 'code constant + behaviour',
             'max(3 × the larger band, 2τ)'))
    add(_chk('N29', [(58, '2', 0)], '2', 'settling_criterion_v6.DETERMINACY_TAU_MULTIPLE; determinacy_threshold(100, 200) '
             '= 2 TAU', SC6.DETERMINACY_TAU_MULTIPLE, '2', 'match' if okd else 'MISMATCH', 'code constant + behaviour',
             'max(3 × the larger band, 2τ)'))
    sig = inspect.signature(SC6.determinate_certified)
    okt2 = list(sig.parameters) == ['margin', 'bar_r', 'bar_o']
    add(_chk('N73', [(57, 'two', 0)], 'two', 'settling_criterion_v6.determinate_certified(margin, bar_r, bar_o): the '
             'difference of two certified cells, one bar each', list(sig.parameters), '2 bars',
             'match' if okt2 else 'MISMATCH', 'code (signature)', 'A difference between two certified evaluations'))
    st = [r for r in cells.values() if r.get('certification_stats_included')]
    nunc = sum(1 for r in st if r['status'] != 'certified')
    add(_chk('N30', [(60, '10', 0)], '10', 'T2 cells with certification_stats_included, status uncertified (W153 '
             'totals.n_uncertified)', nunc, str(nunc), 'match' if nunc == 10 ==
             rec['w153c']['result']['totals']['n_uncertified'] else 'MISMATCH', 'frozen table',
             'The 10 uncertified evaluations'))
    # ---- (iii) ---------------------------------------------------------------------------------------------------------
    d3 = cells['d_36686489']
    e = d3['candidate_canonical']['nodes']['7'][1]
    v6c = v6['cells']['d_36686489']
    ok5 = e == 5.0 and v6c['m_flex_price_multiplier'] == 1.0 and d3.get('cause_uncertified_W157') == 'growth test'
    add(_chk('N31', [(79, '5', 0)], '5', 'T2 d_36686489 candidate_canonical.nodes.7 = [1.25 MVA, 5.0 MWh]; v6 spec '
             'm_flex_price_multiplier 1.0 (baseline price); cause growth test', e, f'{e:.0f}', 'match' if ok5 else
             'MISMATCH', 'frozen table', 'the 5 MWh evaluation fails to certify'))
    w153 = rec['w153c']['result']
    blocks = {}
    for c in w153['cells']:
        for d in c['non_clean_after_N_detail']:
            for b in d['blocks']:
                blocks[b] = blocks.get(b, 0) + 1
    top = max(blocks.values())
    tops = sorted(b for b, n in blocks.items() if n == top)
    fam = w153['totals']['non_clean_block_events_after_N_by_family_total']
    okb = tops == ['TSO|2035|Spring'] and fam == {'TSO': 21, 'DSO': 0, 'ESSO': 0}
    add(_chk('N32', [(84, '2035', 0)], '2035 Spring', 'W153 certification statistics (48e76c9f) cells[].'
             'non_clean_after_N_detail blocks, tallied; totals.non_clean_block_events_after_N_by_family_total',
             {'by_block': blocks, 'by_family': fam}, f'{tops} ({top} of {sum(blocks.values())})',
             'match' if okb else 'MISMATCH', 'named record', 'all in transmission blocks; the recurring block is '
             '2035 Spring', '"all in transmission blocks": TSO 21, DSO 0, ESSO 0'))
    defs = rec['code']['definitions.py']
    mch = rec['code']['model_construction_helpers.py']
    okeq = re.search(r'^EQUALITY_TOLERANCE = 1e-5$', defs, re.M) is not None and \
        'return (0.0, gen.pg[s_o][p] + EQUALITY_TOLERANCE)' in mch and rec['w109']['equality_tolerance_pu'] == 1e-5
    for cid, tok in (('N33', (86, '10⁻⁵', 0)), ('N34', (123, '10⁻⁵', 0))):
        add(_chk(cid, [tok], '10⁻⁵', 'definitions.py EQUALITY_TOLERANCE = 1e-5; model_construction_helpers.py curtaillable '
                 'pg upper bound pg_avail + EQUALITY_TOLERANCE; W109 equality_tolerance_pu', 1e-5, '1e-05',
                 'match' if okeq else 'MISMATCH', 'code constant + named record', 'a 10⁻⁵ pu numerical slack'))
    w109 = rec['w109']
    mwh = w109['tso']['neg_part_eur_at_1_block_weighted'] + w109['dso']['neg_part_eur_at_1_block_weighted']
    sh62, st62 = at_prec('62', abs(mwh))
    add(_chk('N35', [(87, '62', 0)], '62', 'W109 (f6e3533f) tso + dso neg_part_eur_at_1_block_weighted (MWh-equivalent '
             'at 1 €/MWh, block-weighted)', mwh, sh62, st62, 'named record', '≈ 62 MWh-equivalent over the horizon'))
    # 3 tau at L = 44
    note = rec['code'][NOTE_5456]
    txt_ok = '≈ 3 τ left whenever the decay' in note and '44-cycle window' in note and "C\\*'s is ≈ 102" in note
    c = 2.0 ** (-1.0 / 102.0)
    bound = (1.0 / (1.0 - c)) / 44.0
    add(_chk('N36', [(90, '3', 0)], '3', f'{NOTE_5456} (ce96d492): "the monotone branch certifies a drifting cell with ≈ 3 '
             'τ left whenever the decay half-life exceeds its 44-cycle window (C*\'s is ≈ 102)"; derived: a geometric '
             'tail with half-life 102 and |last step| × 44 ≤ τ leaves ≤ τ / (44 (1 − 2^(−1/102)))',
             {'record_text_found': txt_ok, 'derived_bound_over_tau': bound}, f'{bound:.2f}',
             'approximate' if txt_ok and round(bound) == 3 else 'MISMATCH', 'record text (not a measurement)',
             '≈ 3 τ was measured on the reference corner plan at the earlier window L = 44',
             'the record states it as an estimate for a half-life above the window, not a measured drift; "measured" '
             'is the prose\'s word'))
    v41 = rec['v41']
    okl44 = v41['report_only_rule']['constants']['L_MONO']['value'] == 44 and \
        v41['cell']['declaration']['settling_rule']['l_mono'] == 44
    add(_chk('N37', [(91, '44', 0)], '44', 'frozen spec v41 (fcea4b38, the C* extension W110) report_only_rule.constants.'
             'L_MONO and cell.declaration.settling_rule.l_mono', 44 if okl44 else None, '44',
             'match' if okl44 else 'MISMATCH', 'spec constant', 'at the earlier window L = 44'))
    soh = v6['inputs_in_force_now']['configuration_now']['ess_ageing_baseline']['minimum_soh']
    soh050 = _jl(W155_SOH050)
    s050 = [soh050['candidates'][0]['settling_resettle']['minimum_soh'], soh050['extra']['minimum_soh']]
    base050 = soh050['configuration']['ess_ageing_baseline']['minimum_soh']
    lab050 = t['ageing']['column_labels']['eps_AE_050_superseded']
    for cid, tok, frag in (('N38', (93, '0.70', 0), 'The 0.70 SoH floor'), ('N40', (133, '0.70', 0), 'from 0.70 to 0.50'),
                           ('N42', (137, '0.70', 0), 'the 0.70 floor')):
        add(_chk(cid, [tok], '0.70', 'v6 spec inputs_in_force_now ess_ageing_baseline.minimum_soh (the baseline label '
                 '"soh_min 0.70")', soh, f'{soh:.2f}', 'match' if soh == 0.7 else 'MISMATCH', 'spec constant', frag))
    add(_chk('N39', [(94, '0.50', 0)], '0.50', 'T8 column label eps_AE_050_superseded ("soh_min 0.50")', lab050, '0.50',
             'match' if 'soh_min 0.50' in lab050 else 'MISMATCH', 'frozen table', 'resolvable arms at 0.50'))
    add(_chk('N41', [(133, '0.50', 0)], '0.50', 'A64 e_soh050 r2 campaign spec (28c9478d) candidates[0].settling_resettle.'
             'minimum_soh and extra.minimum_soh (the override; configuration.ess_ageing_baseline.minimum_soh is the '
             '0.70 baseline it overrides)', {'override': s050, 'baseline_in_spec': base050},
             '/'.join(f'{x:.2f}' for x in s050), 'match' if s050 == [0.5, 0.5] and base050 == 0.7 else 'MISMATCH',
             'spec constant', 'from 0.70 to 0.50'))
    # ---- (iv) --------------------------------------------------------------------------------------------------------
    w159 = rec['w159']
    c3 = w159['c']['c3']
    oneb = c3['ipopt']['matches_v6_pin'] and not c3['ipopt']['modified_at_or_after_first_v6'] and \
        c3['environment_unchanged_since_first_v6_start_by_these_reads'] is True
    add(_chk('N43', [(103, 'one', 1)], 'one (solver build)', 'W159 (62749030) c3: /usr/local/bin/ipopt sha = the v6 pin, '
             'not modified at or after the first v6 start; environment unchanged', oneb, 'pinned binary, unchanged' if oneb else 'not established',
             'match' if oneb else 'MISMATCH', 'named record', 'one solver build'))
    conc = v6['configuration']['concurrency']
    add(_chk('N44', [(103, 'one', 2)], 'one (evaluation at a time)', 'v6 spec configuration.concurrency', conc, str(conc),
             'match' if conc == 1 else 'MISMATCH', 'spec constant', 'one evaluation at a time'))
    c1 = w159['c']['c1']
    hl = c1['git_grep_tracked_python']['harness_lines']
    okth = c1['all_table_evaluations_ran_with_OMP_NUM_THREADS_1'] is True and all(
        any(f"'{v}': '1'" in ln for ln in hl) for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                       'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    add(_chk('N45', [(104, 'single', 0), (104, 'one', 0)], 'single-threaded; one thread', 'W159 c1: harness '
             'THREAD_CAP_ENV lines (OMP / MKL / OPENBLAS / VECLIB / NUMEXPR = 1); all_table_evaluations_ran_with_'
             'OMP_NUM_THREADS_1', okth, 'thread caps 1 on every table evaluation' if okth else 'not established', 'match' if okth else 'MISMATCH', 'named record',
             'to one thread'))
    hsl = w159['c']['c4']['installed_ipopt_hsl']
    okomp = hsl['links_libgomp_or_libomp'] is False and hsl['nm_symbols_openmp'] == [] and \
        hsl['nm_undefined_openmp'] == []
    add(_chk('N46', [(105, 'MA97', 0)], 'MA97 ran serially', 'W159 c4 installed_ipopt_hsl: links_libgomp_or_libomp False, '
             'nm_symbols_openmp [], nm_undefined_openmp []', okomp, 'no OpenMP link or symbol' if okomp else 'OpenMP present or unread', 'match' if okomp else 'MISMATCH',
             'named record', 'The installed HSL library is built without OpenMP, so MA97 ran serially'))
    w98 = rec['w98']
    rg = w98['replay_gate']
    t11txt = [r for r in t['three_by_three']['rows'] if r.get('text') == '72 / 72 cycles']
    ok72 = rg['bitwise_through_N'] is True and rg['N'] == 72 and rg['first_divergence_cycle'] is None and bool(t11txt)
    add(_chk('N47', [(107, '72', 0), (107, '72', 1), (107, '72', 2)], '72; 72/72', 'W98 r2 campaign_results.json (4689475e) '
             'replay_gate: bitwise_through_N, N 72, first divergence none; T11 row "72 / 72 cycles"', rg['N'],
             f"{rg['N']}/{rg['N']}", 'match' if ok72 else 'MISMATCH', 'named record + frozen table',
             'replayed its 72 recorded cycles bitwise (72/72)'))
    cand = w98['gate_detail']['G13']['summary']['declaration']['replay_reference']['per_cycle_record']
    ok3x3 = '3x3' in cand and any('3 x 3' in r['quantity'] for r in t['three_by_three']['rows'])
    x0key = rec['w98_eval_key_x0']
    add(_chk('N48', [(107, '3', 0), (107, '3', 1), (107, '0', 0)], '3 × 3; x = 0', 'W98 r2 replay reference '
             f'{cand} (the 3 × 3 pair, x0); candidate_key {x0key[:8]} = the x = 0 key (8435c718)', cand,
             '3x3 / x0', 'match' if ok3x3 and x0key.startswith('8435c718') and rec['w98_x0_canonical_all_zero'] else 'MISMATCH', 'named record',
             'the 3 × 3 multi-scenario instance\'s x = 0 evaluation'))
    rp = rec['w101_replays']
    ok33 = len(rp) == 3 and all(v['G19'] is True and v['through'] == v['N'] for v in rp.values())
    add(_chk('N49', [(108, 'three', 0), (108, '3', 0), (108, '3', 1)], 'three; 3/3', 'W101 campaign_results (x0, unit, '
             'C*): gates.G19_replay_bitwise_1_N_every_field and G13 replay_bitwise_through_cycle = N (the certification '
             'cycle of the replayed record)', rp, f"{sum(1 for v in rp.values() if v['G19'])}/{len(rp)}",
             'match' if ok33 else 'MISMATCH', 'named record',
             'the three SRP1 reference continuations replayed bitwise to their certification cycles (3/3)'))
    ip = c3['ipopt']['version_output']['stdout']
    add(_chk('N50', [(112, '3.14.18', 0)], '3.14.18', 'W159 c3 ipopt --version; c2 banners (ipopt_versions_in_all_banners)',
             ip.strip(), '3.14.18', 'match' if 'Ipopt 3.14.18 ' in ip and w159['c']['c2'][
                 'tally_d_4a82a64a_all_manifest_logs']['versions'] == ['3.14.18'] else 'MISMATCH', 'named record',
             'IPOPT 3.14.18'))
    tal = w159['c']['c2']['tally_d_4a82a64a_all_manifest_logs']
    add(_chk('N51', [(112, 'MA97', 0)], 'MA97 (network subproblems)', 'W159 c2 banner tally over the 495 hash-matching logs '
             'of d_4a82a64a: network', tal['network'], json.dumps(tal['network']),
             'match' if list(tal['network']) == ['ma97'] else 'MISMATCH', 'named record', 'MA97 (network subproblems)'))
    add(_chk('N52', [(113, 'MA57', 0)], 'MA57 (storage-operator subproblem)', 'W159 c2 banner tally: esso', tal['esso'],
             json.dumps(tal['esso']), 'match' if list(tal['esso']) == ['ma57'] else 'MISMATCH', 'named record',
             'MA57 (storage-operator subproblem)'))
    add(_chk('N53', [(113, '3.11.11', 0)], '3.11.11', 'W159 c3 sys_version', c3['sys_version'], c3['sys_version'][:7],
             'match' if c3['sys_version'].startswith('3.11.11 ') else 'MISMATCH', 'named record', 'Python 3.11.11'))
    add(_chk('N54', [(113, '6.9.5', 0)], '6.9.5', 'W159 c3 pyomo_version', c3['pyomo_version'], c3['pyomo_version'],
             'match' if c3['pyomo_version'] == '6.9.5' else 'MISMATCH', 'named record', 'Pyomo 6.9.5'))
    pv_ = re.search(r'ProductVersion:\s*([0-9.]+)', c3['sw_vers']).group(1)
    add(_chk('N55', [(113, '27.0', 0)], '27.0', 'W159 c3 sw_vers ProductVersion; sw_vers now', {'w159': pv_,
             'now': envnow['sw_vers_product_version']}, pv_, 'match' if pv_ == '27.0' ==
             envnow['sw_vers_product_version'] else 'MISMATCH', 'named record + environment now', 'macOS 27.0'))
    add(_chk('N56', [(113, 'M2 Max', 0)], 'Apple M2 Max', 'sysctl machdep.cpu.brand_string, read now (no committed record '
             'carries it)', envnow['cpu_brand'], envnow['cpu_brand'], 'match' if envnow['cpu_brand'] == 'Apple M2 Max'
             else 'MISMATCH', 'environment now', 'They ran on an Apple M2 Max',
             'read at this run; the campaign records carry no CPU string (W159 uname: arm64, kernel T6020)'))
    mem_now = envnow['hw_memsize']
    mem_v6 = v6['memory_preflight']['measured_at_freeze_non_gating']['hw_memsize_bytes']
    add(_chk('N57', [(114, '32', 0)], '32 GB', 'sysctl hw.memsize now; v6 spec memory_preflight hw_memsize_bytes',
             {'now': mem_now, 'v6_spec': mem_v6}, f'{mem_now / 2 ** 30:g} GiB', 'match' if mem_now == mem_v6 ==
             32 * 2 ** 30 else 'MISMATCH', 'spec record + environment now', 'with 32 GB of memory',
             '34,359,738,368 bytes = 32 GiB (the text writes GB)'))
    acc = {}
    for c in w153['cells']:
        for d in c['acceptable_clean_after_N_detail']:
            for b in d['blocks']:
                acc[b] = acc.get(b, 0) + 1
    oka = sorted(acc) == ['DSO|5|2035|Winter', 'DSO|7|2025|Winter'] and \
        w153['totals']['acceptable_clean_after_N_by_family_total'] == {'TSO': 0, 'DSO': 119, 'ESSO': 0}
    add(_chk('N58', [(116, 'Two', 0), (116, '7', 0), (116, '2025', 0), (116, '5', 0), (116, '2035', 0)],
             'Two distribution blocks (node 7 in 2025 Winter, node 5 in 2035 Winter)', 'W153 cells[].'
             'acceptable_clean_after_N_detail blocks, tallied; totals.acceptable_clean_after_N_by_family_total', acc,
             ', '.join(f'{k} {v}' for k, v in sorted(acc.items())), 'match' if oka else 'MISMATCH', 'named record',
             'Two distribution blocks (node 7 in 2025 Winter, node 5 in 2035 Winter)',
             'scope: the 42 W153 cells, after N'))
    add(_chk('N59', [(117, '10', 0)], '10', 'settling_criterion_v5.CLEAN_FACTOR (W153 acceptable_clean = primary '
             'Acceptable within 10×)', SC6.CLEAN_FACTOR, f'{SC6.CLEAN_FACTOR:g}', 'match' if okf else 'MISMATCH',
             'code constant', 'within 10× the tight-tail tolerances'))
    case9 = rec['case9']
    tso_cit = re.findall(r'"compl_inf_tol":\s*([0-9.eE+-]+)', json.dumps(case9))
    net = rec['code']['network.py']
    dso_default = "IPOPT's default 1e-4" in net
    add(_chk('N60', [(121, '10⁻⁴', 0)], '10⁻⁴', 'production complementarity tolerance before the tail: DSO = IPOPT default '
             '1e-4 (network.py; no DSO case file sets it); TSO = data/SRP1/case9/case9_params.json compl_inf_tol',
             {'dso': 1e-4 if dso_default else None, 'tso': [float(x) for x in tso_cit]},
             f"DSO 1e-04, TSO {', '.join(tso_cit)}", 'MISMATCH' if tso_cit and float(tso_cit[0]) != 1e-4 else 'match',
             'case file + code', 'complementarity 10⁻⁴ → 10⁻⁶',
             'holds for the distribution solves only; the transmission solves were at 5 × 10⁻⁴ before the tail '
             '(REVISION_CONTEXT Addendum 48 entry says the same)'))
    add(_chk('N61', [(121, '10⁻⁶', 0)], '10⁻⁶', 'as N6 (tail compl_inf_tol)', cit, f'{cit:g}',
             'match' if cit == 1e-6 else 'MISMATCH', 'spec constant', 'complementarity 10⁻⁴ → 10⁻⁶'))
    w86 = rec['w86']['per_cell']
    rel = {k: v['comparison']['dQ_relative'] for k, v in w86.items()}
    same_sign = all(x < 0 for x in rel.values())
    mean_rel = sum(rel.values()) / len(rel)
    add(_chk('N62', [(122, 'three', 0)], 'three', 'W86 tail recert campaign_results.json per_cell (x0, unit, C*)', len(rel),
             str(len(rel)), 'match' if len(rel) == 3 else 'MISMATCH', 'named record',
             'each of the three SRP1 references'))
    add(_chk('N63', [(122, '1.1', 0), (122, '10⁻⁶', 0)], '≈ 1.1 × 10⁻⁶ relative, with the same sign', 'W86 per_cell.'
             '*.comparison.dQ_relative', rel, ', '.join(f'{k} {x:.3e}' for k, x in rel.items()) +
             f'; mean {mean_rel:.3e}', 'approximate' if same_sign and f'{abs(mean_rel) * 1e6:.1f}' == '1.1'
             else 'MISMATCH', 'named record', 'by ≈ 1.1 × 10⁻⁶ relative, with the same sign',
             'range 1.04–1.20 × 10⁻⁶ (one decimal: 1.0 / 1.1 / 1.2), mean 1.12; all negative'))
    pr = w109['tso']['priced_neg_part_eur_block_weighted'] + w109['dso']['priced_neg_part_eur_block_weighted']
    sh79, st79 = at_prec('7.9', abs(pr) / 1000.0)
    add(_chk('N64', [(123, '7.9', 0)], '≈ 7.9', 'W109 tso + dso priced_neg_part_eur_block_weighted (k€)', pr, sh79, st79,
             'named record', 'lowers the objective by ≈ 7.9 k€ first-order',
             f'{pr:,.2f} € = {abs(pr) / 1000:.4f} k€, which is 7.8 at one decimal; 7.9 is the rounding of the rounded '
             '7.85 (TASKS W109 line, Addenda 56)'))
    qx0 = cells['ref:7aa017f0']['Q']
    sh12, st12 = at_prec('1.2', abs(pr) / qx0 * 1e5)
    add(_chk('N65', [(123, '1.2', 0), (123, '10⁻⁵', 1)], '≈ 1.2 × 10⁻⁵ of it', 'W109 priced negative parts / Q of the '
             'settled x0 (T2 ref:7aa017f0, the W109 instance d110bd1a cycle 181)', abs(pr) / qx0, sh12 + 'e-5', st12,
             'named record + frozen table', '(≈ 1.2 × 10⁻⁵ of it)'))
    rd = c3['read_at_utc'][:10]
    add(_chk('N66', [(126, '2026-10-06', 0)], '2026-10-06', 'W159 c3 read_at_utc; environment_unchanged_since_first_v6_'
             'start_by_these_reads', rd, rd, 'match' if rd == '2026-10-06' and
             c3['environment_unchanged_since_first_v6_start_by_these_reads'] is True else 'MISMATCH', 'named record',
             'The versions above were read on 2026-10-06'))
    # ---- accepted sentences: the multipliers ----------------------------------------------------------------------
    cl_ids = {c['claim_id'] for c in t['claims']}
    a_ids = {r['claim_id'] for r in t['a64']['rows']}
    m175 = sorted(set(re.findall(r'"flex_price_multiplier": ([0-9.]+)', json.dumps(_jl(W155_M175)))))
    okm175 = 'H:m1.75:value_minus_I' in a_ids and m175 == ['1.75']
    add(_chk('N67', [(131, '1.75', 0), (132, '1.75', 0)], '1.75', 'T10 claim H:m1.75:value_minus_I; A64 h_unit_m175 r2 spec '
             'flex_price_multiplier', m175, '/'.join(m175), 'match' if okm175 else 'MISMATCH', 'frozen table + spec',
             'between 1.5 and 1.75'))
    add(_chk('N68', [(131, '2', 0)], '2', 'T1 claim H:m2:value_minus_I', 'H:m2:value_minus_I' in cl_ids, '2',
             'match' if 'H:m2:value_minus_I' in cl_ids else 'MISMATCH', 'frozen table', '1.75 and 2 times'))
    add(_chk('N69', [(132, '1.5', 0)], '1.5', 'T1 claim H:m1.5:value_minus_I', 'H:m1.5:value_minus_I' in cl_ids, '1.5',
             'match' if 'H:m1.5:value_minus_I' in cl_ids else 'MISMATCH', 'frozen table', 'between 1.5 and 1.75'))
    # S2a: the EFC/day differences against the named record (W161 "beside", now counted as named-record checks)
    efc050 = t['a64']['scored']['B']['efc_per_day']
    py = {str(r['year_block']): r['efc_per_day'] for r in rec['w153d']['result']['settled_unit_ageing']['per_year']}
    for cid, y, w, tok in (('N70', '2025', '+0.18', (133, '+0.18', 0)), ('N71', '2030', '+0.20', (134, '+0.20', 0)),
                           ('N72', '2035', '+0.27', (134, '+0.27', 0))):
        dv = efc050[y] - py[y]
        sh, stt = at_prec(w, dv)
        add(_chk(cid, [tok], w, f'T10 a64.scored.B.efc_per_day[{y}] (0.50) − W153 w153_discount_row.json settled_unit_'
                 f'ageing.per_year[{y}].efc_per_day (0.70)', dv, sh, stt, 'frozen table − named record',
                 'releases cycling in every year (EFC/day +0.18, +0.20, +0.27)', 'W161 S2a beside, counted here'))
    return C


def nonmanuscript_checks(rec):
    brief = rec['code'][BRIEF]
    hdr = re.search(r'^# Addendum 66 — .*\((\d{4}-\d{2}-\d{2})\)$', brief, re.M)
    X = []
    X.append(_chk('X1', [(1, '66', 0), (4, '66', 0), (7, '66', 0), (99, '66', 0)], 'Addendum 66',
                  f'{BRIEF}: "# Addendum 66 — ..." header', hdr.group(0)[:60] if hdr else None, '66',
                  'match' if hdr else 'MISMATCH', 'brief', 'Addendum 66'))
    X.append(_chk('X2', [(3, '2026-10-07', 0)], '2026-10-07', 'Addendum 66 header date; paragraphs_v3.md commit date',
                  {'addendum_66_date': hdr.group(1) if hdr else None, 'commit_date': rec['p3_commit_date']},
                  hdr.group(1) if hdr else None, 'match' if hdr and hdr.group(1) == '2026-10-07' else 'MISMATCH',
                  'brief', 'Planner, 2026-10-07',
                  f"equals the Addendum 66 date; the file was committed {rec['p3_commit_date']}"))
    X.append(_chk('X3', [(4, '2026-09-13', 0)], '2026-09-13', 'the brief file name', BRIEF, BRIEF,
                  'match' if os.path.exists(os.path.join(REPO, BRIEF)) else 'MISMATCH', 'repository', BRIEF))
    ids = {'73b42c49': rec['p2_sha'][:8] == '73b42c49', '2a1d7f92': rec['p2_commit'].startswith(P2_COMMIT),
           '590088fe': FZ_SHA.startswith('590088fe')}
    X.append(_chk('X4', [], 'sha256 73b42c49..., commit 2a1d7f92, frozen_step6_tables_v1_590088fe.json',
                  'sha256 of paragraphs_v2.md; its last commit; the frozen JSON sha256', ids, json.dumps(ids),
                  'match' if all(ids.values()) else 'MISMATCH', 'repository', 'identifiers in the comment block'))
    return X


# token-only classifications (no data counterpart): (line, text, occ) -> (category, reason)
UNCHECKED = {
    (1, '6', 0): ('non-manuscript', 'title "Step 6" (stage label)'),
    (22, '1', 0): ('enumerator', 'list item 1'), (24, '2', 0): ('enumerator', 'list item 2'),
    (26, '3', 0): ('enumerator', 'list item 3'), (28, '4', 0): ('enumerator', 'list item 4'),
    (29, '5', 0): ('enumerator', 'list item 5'),
    (54, 'one', 0): ('not a figure', 'pronoun ("An evaluation without one")'),
    (70, '0', 0): ('notation', 'λ ∈ [0, c_flex]: the lower end of the set-valued interface price; no table carries it'),
    (74, 'one', 0): ('not a figure', '"they share one fingerprint": the three bullets that follow are each checked (L1-L6)'),
    (87, '10⁻⁵', 0): ('NO SOURCE FOUND', '"≈ 10⁻⁵ of renewable energy": Addendum 56 states it; no record carries the '
                      'renewable-energy denominator (searched w109_tso_curtailment_look.json, p515_s53_w109 script, '
                      'TASKS.md, PLANNER_BRIEF Addenda 55-56)'),
    (103, 'one', 0): ('NO SOURCE FOUND', '"one machine": no evaluation record carries a host identifier (launch.json keys: '
                      'command, cwd, thread caps, pids, utc, spec sha, label, key; W159 uname is a read at 2026-10-06)'),
    (120, 'Two', 0): ('structural', '"Two effects": the count of the two bullets that follow (each checked: N60-N65)'),
    (131, '1', 0): ('enumerator', 'sentence 1'), (133, '2', 0): ('enumerator', 'sentence 2'),
    (136, '3', 0): ('enumerator', 'sentence 3'), (138, '4', 0): ('enumerator', 'sentence 4'),
}


# ======================================================================================================================
#  4. definition checks (no figure)
# ======================================================================================================================
def definition_checks(fz, rec):
    t = fz['tables']
    allc = dict(t['cells'])
    allc.update(t['cells_appended_w160'])
    v6 = rec['v6']
    D = []
    # D1: the slack definition (scorer behaviour + spec text)
    vg = L132.view_from_report({'status': 'uncertified', 'gated': True, 's_signed': -1234.5, 'Q_end': 100.0,
                                't_sum_end': -7.0})
    vu = L132.view_from_report({'status': 'uncertified', 'gated': False, 's_signed': None, 'Q_end': 100.0,
                                't_sum_end': -7.0})
    ru = L132.resolve(-50000.0, -50000.0, [vu, {'status': 'certified', 'band': 1.0}])
    spec_s = v6['definitions']['per_cell']['s']
    spec_u = v6['scorer']['uncertified_form']['determinacy_rule']
    d1 = {'gated_view_slack': vg['slack'], 'gated_view_gap': vg['gap'], 'ungated_view_slack': vu['slack'],
          'ungated_resolve_verdict': ru.get('verdict'), 'spec_definitions_per_cell_s': spec_s,
          'spec_uncertified_form_rule': spec_u}
    ok1 = vg['slack'] == 1234.5 and vg['gap'] == 7.0 and vu['slack'] is None and \
        ru.get('verdict') == 'indeterminate (slack undefined)' and 's = Q(end) - Q_N_old' in spec_s and \
        "ORIGINAL record's last cycle" in spec_s and 'INDETERMINATE (slack undefined)' in spec_u
    D.append({'id': 'D1', 'claim': 'settling slack = the objective at the cap minus the objective at the cycle where the '
              'earlier run had certified; no earlier certificate -> no slack -> indeterminate',
              'counterpart': 'L132.view_from_report / L132.resolve (behaviour on synthetic views); v6 spec '
              'definitions.per_cell.s and scorer.uncertified_form.determinacy_rule', 'evidence': d1,
              'verdict': 'consistent' if ok1 else 'INCONSISTENT',
              'note': 'the scorer uses |s| in the bar (the prose\'s "movement" is signed s; the bar takes its absolute '
                      'value); Q_N_old is the ORIGINAL record\'s last cycle, which D2 shows is its certification cycle '
                      'on every uncertified evaluation of the tables'})
    # D2: every uncertified evaluation of the tables is gated, with an earlier certificate at N_old
    unc = sorted(k for k, v in allc.items() if v.get('status') != 'certified')
    per = {}
    for k in unc:
        r = allc[k]
        if k.startswith('ref:'):
            nm = {'ref:5ca4f86c': 'f2_incumbent', 'ref:e28de4ac': 'f2_challenger'}[k]
            sc = rec['w118_spec']['cells'][nm]
            sm = rec['w118_summary']['reports'][nm]
            ed = json.loads(json.dumps(sc['original']))['eval_dir']
            gated, n_old, q_n_old, s_rec, q_end = sc['gated'], sc['N_old'], sc['Q_N_old'], sm['s_signed'], sm['Q_at_cap']
        else:
            sc = v6['cells'][k]
            ed = sc['original']['eval_dir']
            gated, n_old, q_n_old, s_rec, q_end = sc['gated'], sc['N_old'], sc['Q_N_old'], r['s_signed'], r['Q']
        er = _jl(os.path.join(ed, 'evaluation_record.json'))
        rec['inputs_extra'][os.path.join(ed, 'evaluation_record.json')] = _sha(os.path.join(ed, 'evaluation_record.json'))
        per[k] = {'gated_spec': gated, 'gated_table': r.get('gated'), 'N_old': n_old,
                  'original_eval_dir': ed, 'original_status': er['status'],
                  'original_certification_cycle': er['certification_cycle'], 'original_cycles_run': er['cycles_run'],
                  'original_certified_cost_equals_Q_N_old': er['certified_cost'] == q_n_old,
                  's_recorded': s_rec, 's_recomputed_Q_end_minus_Q_N_old': q_end - q_n_old,
                  'table_slack': r.get('slack'),
                  'table_slack_equals_abs_s': r.get('slack') is not None and abs(abs(s_rec) - r['slack']) == 0.0}
        p = per[k]
        p['ok'] = bool(p['gated_spec'] is True and p['original_status'] == 'certified' and
                       p['original_certification_cycle'] == n_old == p['original_cycles_run'] and
                       p['original_certified_cost_equals_Q_N_old'] and p['table_slack_equals_abs_s'] and
                       abs(p['s_recomputed_Q_end_minus_Q_N_old'] - s_rec) == 0.0)
    D.append({'id': 'D2', 'claim': 'Every uncertified evaluation in the tables has such an earlier certificate',
              'counterpart': 'frozen tables cells + cells_appended_w160 (status != certified); v6 spec cells / W118 spec '
              'v2 cells (gated, N_old, Q_N_old, original eval_dir); each original evaluation_record.json',
              'evidence': {'n_uncertified': len(unc), 'cells': per},
              'verdict': 'consistent' if unc and all(p['ok'] for p in per.values()) else 'INCONSISTENT',
              'note': 'scope: the evaluations of the frozen cell registry (T1/T2/T4/T5/T9/T10). T6 benchmark arms and '
                      'the T11 3 × 3 evaluations carry no settling-rule status in the frozen JSON and are not covered'})
    # D3: every repeated evaluation replayed bitwise to k0
    rows = {}
    for k, r in sorted(allc.items()):
        through = r.get('replay_bitwise_through')
        if r.get('gated') is True or through is not None:
            k0 = r.get('k0_run')
            if k == 'd_c52e1670':
                g19 = rec['w139_c52']['gates']['G19_replay_bitwise_1_k0_every_field']
                through = rec['w139_c52']['cell_report']['replay_bitwise_through']
                src = 'v5 run (W139/W140) campaign_results.json (the v6 certificate is from its records)'
            elif k.startswith('ref:'):
                nm = {'ref:5ca4f86c': 'f2_incumbent', 'ref:e28de4ac': 'f2_challenger'}.get(k)
                if nm is None:
                    continue
                k0 = rec['w118_summary']['reports'][nm]['k0_run']
                g19, src = rec['w118_summary']['reports'][nm].get('replay_first_divergence') is None, 'W118 summary'
            else:
                g19, src = r.get('replay_first_divergence') is None, 'frozen table'
            rows[k] = {'k0_run': k0, 'replay_bitwise_through': through, 'first_divergence_none': g19, 'source': src,
                       'ok': bool(g19 and through is not None and k0 is not None and through >= k0)}
    D.append({'id': 'D3', 'claim': 'Every evaluation repeated from an earlier record replayed that record bitwise up to k₀',
              'counterpart': 'frozen tables replay_bitwise_through / replay_first_divergence (gated cells, T5, F2); '
              'd_c52e1670 from its v5 run; W101 references in N49', 'evidence': rows,
              'verdict': 'consistent' if rows and all(v['ok'] for v in rows.values()) else 'INCONSISTENT',
              'note': f'{len(rows)} evaluations; replay through == k0_run on each (the W101 references replay through '
                      'N > k0, N49)'})
    # D4 (observation): the residual-pass clause requires successful local solves, not the clean classification
    srp = rec['code']['shared_resources_planning.py']
    hf = rec['code']['helper_functions.py']
    o4 = 'cycle_convergence = boyd_all_pass and local_solves_ok' in srp and \
        'po.TerminationCondition.optimal' in hf and 'locallyOptimal' in hf
    D.append({'id': 'D4', 'claim': '"Every local NLP must return a clean solution (defined below) in the same cycle" (the '
              'residual-pass condition)', 'counterpart': 'shared_resources_planning.py: cycle_convergence = boyd_all_pass '
              'and local_solves_ok (_admm_local_solves_succeeded -> helper_functions.solver_result_succeeded: solver '
              'status ok and termination optimal / locallyOptimal / globallyOptimal); v6 spec stop_rule.boyd_k',
              'evidence': {'boyd_k_spec': v6['stop_rule']['boyd_k'], 'code_pattern_found': o4},
              'verdict': 'OBSERVATION: the code requires a successful solve, not the 10× clean classification' if o4
              else 'not confirmed',
              'note': 'the clean classification (Optimal on any tier, or Acceptable on the primary attempt within 10×) '
                      'is the certification veto of clause 5 (settling_criterion_v5 all_clean_k), not part of boyd_k; '
                      'v2 read "must solve successfully". Whether IPOPT "Solved To Acceptable Level" maps to a '
                      'successful termination here was not confirmed'})
    # D5 (observation): clause 5 "every cycle the test reads was solved cleanly" vs the veto window
    src5 = inspect.getsource(SC5.SettlingRuleV5._non_clean_in) + inspect.getsource(SC6.SettlingRuleV6.evaluate)
    oow = rec['w153c']['result']['totals']['certificates_with_a_non_clean_cycle_read_out_of_window']
    D.append({'id': 'D5', 'claim': '"every cycle the test reads was solved cleanly" (clause 5)',
              'counterpart': 'settling_criterion_v5._non_clean_in (veto over the certifying window [lo, k] only); v6 spec '
              'stop_rule.sub_test_reads_assertion; W153 totals.certificates_with_a_non_clean_cycle_read_out_of_window',
              'evidence': {'veto_reads_window_only': 'lo <= c <= hi' in src5,
                           'sub_test_reads_assertion': v6['stop_rule']['sub_test_reads_assertion'],
                           'certificates_reading_a_non_clean_cycle_out_of_window': oow},
              'verdict': 'INCONSISTENT with the code: the veto covers the certifying window; '
                         f'{len(oow)} certificates read a non-clean cycle outside it' if oow else 'consistent',
              'note': 'v2 read "every cycle in the window the test reads was solved cleanly", which matches the code; '
                      'Addendum 61 ruling 3 accepted the out-of-window reads as enumerated, not vetoed'})
    # D6: the anchor of "15-21 kEUR after the residuals first passed"
    bes = rec['reference_beside']
    okn = all(b['N'] == b['k0'] + 9 for b in bes.values())
    D.append({'id': 'D6', 'claim': '"on the reference evaluations the objective moved by a further 15–21 k€ after the '
              'residuals first passed"', 'counterpart': 'W101 summary s = Q(k*) − Q(N) (N = the old certificate) and the '
              'W101 per-cycle records (Q at k0, the first residual pass of the continuation)', 'evidence': bes,
              'verdict': 'figures match s (N4, N5); ANCHOR differs: s is measured from the old certificate N = k0 + 9, '
                         'not from the first residual pass k0' if okn else 'not confirmed',
              'note': 'measured from k0 the net movement is ' + ', '.join(
                  f"{nm} {b['Q_end_minus_Q_k0']:+,.2f}" for nm, b in bes.items() if nm != 'c_star') +
                      '; the largest excursion from Q(N) over N..k* is ' + ', '.join(
                  f"{nm} {b['max_abs_Q_k_minus_Q_N']:,.2f}" for nm, b in bes.items() if nm != 'c_star') +
                      '. "The reference evaluations" = x0 (d110bd1a) and the unit (3f084f2f); C* is uncertified '
                      '(s −4,378.88) and outside 15–21'})
    return D


# ======================================================================================================================
#  inputs, guards, post-run
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    base = W160.pickle_state()
    counts = dict(base['counts'], w161=dict(W161.PICKLE_COUNTS), w162=dict(PICKLE_COUNTS))
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run():
    for rel in (OUT_LOG, OUT_MAN, OUT_TYPING_JSON, OUT_TYPING_LOG):
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W162 post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in (OUT_LOG, OUT_MAN, OUT_TYPING_JSON, OUT_TYPING_LOG)}
    with open(os.path.join(REPO, OUT_POST), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f'[W162 post-run] wrote {OUT_POST}: ' + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def env_now():
    def sc(name):
        r = subprocess.run(['sysctl', '-n', name], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    sw = subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True)
    mem = sc('hw.memsize')
    return {'cpu_brand': sc('machdep.cpu.brand_string'), 'hw_memsize': int(mem) if mem else None,
            'sw_vers_product_version': sw.stdout.strip() if sw.returncode == 0 else None,
            'read_utc': datetime.now(timezone.utc).isoformat(), 'how': 'sysctl -n / sw_vers (subprocess; not a solve)'}


def md_summary(rec_out):
    s = rec_out['summary']
    L = ['# W162 -- figure check of paragraphs_v3.md (Addendum 66)', '',
         f"Text: `{P3_REL}` sha256 `{P3_SHA}` (commit `{P3_COMMIT}`). Frozen tables `{os.path.basename(FZ_REL)}` "
         f"(sha256 `{FZ_SHA[:8]}…`). Script `{SCRIPT_REL}`. ZERO SOLVES (guards verified 0), pickle blocked. "
         'The text is not edited.', '',
         '## Counts (manuscript scope: lines 10-140 of paragraphs_v3.md)', '',
         f"- numeric tokens in scope: {s['tokens_scope']} (non-manuscript title/comment: {s['tokens_nonmanuscript']})",
         f"- figure checks touching v3 prose: {s['n_checks_in_v3']} -- match {s['n_match']}, MISMATCH {s['n_mismatch']}, "
         f"approximate {s['n_approximate']}, no table counterpart {s['n_no_table_counterpart']}",
         f"- W161 checks carried: {s['w161_carried_verbatim']} verbatim + {s['w161_carried_rewritten']} on a rewritten "
         f"sentence; {s['w161_not_in_v3']} of 164 not in the v3 prose (sources tables, map, scorecard, dropped)",
         f"- new v3 checks (N): {s['n_new']}; non-manuscript identifier checks (X): {s['n_x']}",
         f"- tokens unchecked: {s['n_unchecked_tokens']} (enumerators {s['unchecked_by_category'].get('enumerator', 0)}, "
         f"NO SOURCE FOUND {s['unchecked_by_category'].get('NO SOURCE FOUND', 0)}, other "
         f"{s['n_unchecked_tokens'] - s['unchecked_by_category'].get('enumerator', 0) - s['unchecked_by_category'].get('NO SOURCE FOUND', 0)})",
         f"- every token assigned: {s['every_token_assigned']}", '',
         '## Mismatches', '']
    for c in rec_out['mismatches']:
        L.append(f"- **{c['id']}** line(s) {sorted({x[0] for x in c['tokens']})}: written `{c.get('written_v3')}`, "
                 f"counterpart {c.get('value_at_written_precision', c.get('table_value_at_written_precision'))} -- "
                 f"{c.get('counterpart', c.get('table_ref'))}. {c.get('note') or ''}")
    L += ['', '## Approximate', '']
    for c in rec_out['approximate']:
        L.append(f"- **{c['id']}**: written `{c['written_v3']}`, counterpart {c['value_at_written_precision']}. "
                 f"{c.get('note') or ''}")
    L += ['', '## Unchecked numbers', '', '| line | token | category | reason |', '|---|---|---|---|']
    for u in rec_out['unchecked_tokens']:
        L.append(f"| {u['line']} | {u['text']} | {u['category']} | {u['reason']} |")
    L += ['', '## Definition checks', '']
    for d in rec_out['definition_checks']:
        L.append(f"- **{d['id']}** {d['claim']}: **{d['verdict']}**. {d['note']}")
    L += ['', '## Every check in v3', '', '| id | lines | written | status | counterpart (at written precision) | source |',
          '|---|---|---|---|---|---|']
    for c in rec_out['checks_in_v3']:
        ln = ','.join(str(x) for x in sorted({x[0] for x in c['tokens']})) or '-'
        val = c.get('value_at_written_precision', c.get('table_value_at_written_precision'))
        src = c.get('counterpart', c.get('table_ref'))
        L.append(f"| {c['id']} | {ln} | {c.get('written_v3')} | {c['status']} | {val} | "
                 f"{str(src).replace('|', '/')[:160]} |")
    L += ['', '## W161 checks not in the v3 prose', '',
          ', '.join(c['id'] for c in rec_out['w161_not_in_v3'])]
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--out-dir', default=None, help='trial runs only: write every W162 output under this directory')
    args = ap.parse_args()
    if args.out_dir:
        g = globals()
        for k in ('OUT_JSON', 'OUT_MD', 'OUT_MAN', 'OUT_LOG', 'OUT_TYPING_JSON', 'OUT_TYPING_LOG', 'OUT_POST'):
            g[k] = os.path.join(args.out_dir, os.path.basename(g[k]))
    if args.post_run:
        post_run()
    t0 = time.time()
    tag = 'W162'
    failed = []
    # ---- preconditions ------------------------------------------------------------------------------------------
    pre = []
    for rel in (OUT_JSON, OUT_MD, OUT_MAN, OUT_POST, OUT_TYPING_JSON):
        if os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} exists (write-once)')
    if not os.path.isdir(os.path.join(REPO, os.path.dirname(OUT_JSON))):
        pre.append(f'{os.path.dirname(OUT_JSON)} missing (create it before the launch; the launch log goes there)')
    w101_eval = {}
    for nm in ('x0', 'n7_4h_e1', 'c_star'):
        ed_root = os.path.join(W101_DIR, f'campaign_s53_w101_srp1_cont_{nm}', 'evals')
        subs = sorted(os.listdir(os.path.join(REPO, ed_root)))
        if len(subs) != 1:
            pre.append(f'{ed_root}: expected one eval dir, found {subs}')
            continue
        w101_eval[nm] = os.path.join(ed_root, subs[0])
    input_rels = {'PARAGRAPHS_V3': P3_REL, 'PARAGRAPHS_V2': P2_REL, 'FROZEN_JSON': FZ_REL, 'W161_JSON': W161_JSON,
                  'W153C': W153C_REL, 'W153D': W153D_REL, 'W153_MANIFEST': W153_MAN, 'V6_SPEC': V6_SPEC,
                  'V41_SPEC': V41_SPEC, 'W118_SPEC': W118_SPEC, 'W118_SUMMARY': W118_SUMMARY,
                  'W101_SUMMARY': W101_SUMMARY, 'W86_RESULTS': W86_RESULTS, 'W98_RESULTS': W98_RESULTS, 'W98_EVAL': W98_EVAL,
                  'W109_JSON': W109_JSON, 'W141_JSON': W141_JSON, 'W137_CELL3_RECORD': W137_CELL3, 'W132_CELL1_RECORD': W132_CELL1,
                  'W139_C52_RECORD': W139_C52_REC, 'W139_C52_RESULTS': W139_C52_RES, 'W159_JSON': W159_JSON,
                  'W155_SOH050_SPEC': W155_SOH050, 'W155_M175_SPEC': W155_M175, 'SRP1_PARAMS': SRP1_PARAMS,
                  'CASE9_PARAMS': CASE9_PARAMS, 'NOTE_5456': NOTE_5456, 'BRIEF': BRIEF, 'HELPER_FUNCTIONS':
                  'helper_functions.py'}
    for nm, ed in w101_eval.items():
        for f in ('per_cycle_record.jsonl', 'settling_decision.json'):
            input_rels[f'W101_{nm}_{f}'] = os.path.join(ed, f)
        input_rels[f'W101_{nm}_campaign_results'] = os.path.join(os.path.dirname(os.path.dirname(ed)),
                                                                 'campaign_results.json')
    for f in CODE_FILES:
        input_rels[f'CODE_{f}'] = f
    inputs = {}
    for key, rel in input_rels.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} missing')
            continue
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': W160.W157.L132._committed_clean(rel),
                       'last_commit': W160._last_commit(rel)}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    for key, want in (('PARAGRAPHS_V3', P3_SHA), ('PARAGRAPHS_V2', P2_SHA), ('FROZEN_JSON', FZ_SHA), ('V6_SPEC', V6_SHA)):
        if inputs[key]['sha256'] != want:
            pre.append(f"{inputs[key]['path']} sha {inputs[key]['sha256']} != {want}")
    w153man = _jl(W153_MAN)
    for key in ('W153C', 'W153D'):
        if w153man.get(inputs[key]['path']) != inputs[key]['sha256']:
            pre.append(f"{inputs[key]['path']} != its W153 manifest entry")
    fz = _jl(FZ_REL)
    # capture-path assertion: every source the checks read exists before anything is evaluated
    for k in ('tables', 'constants', 'paragraph_figure_checks'):
        if k not in fz:
            pre.append(f'frozen JSON lacks {k}')
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    _log(f'[{tag}] script {SCRIPT_REL} sha256 {_sha(SCRIPT_REL)} committed clean {script_clean}; '
         f'{len(inputs)} inputs committed clean; paragraphs_v3 {P3_SHA[:8]}, frozen JSON {FZ_SHA[:8]}, v6 spec '
         f'{V6_SHA[:8]} verified')

    raw = _text(P3_REL)
    lines, end = scope_lines(raw)
    v3_norm = normalise(lines, end)
    toks = inventory(lines, end)
    rec = {'v6': _jl(V6_SPEC), 'v41': _jl(V41_SPEC), 'w118_spec': _jl(W118_SPEC), 'w118_summary': _jl(W118_SUMMARY),
           'w101_summary': _jl(W101_SUMMARY), 'w86': _jl(W86_RESULTS), 'w98': _jl(W98_RESULTS), 'w109': _jl(W109_JSON),
           'w141': _jl(W141_JSON), 'w159': _jl(W159_JSON), 'srp1_params': _jl(SRP1_PARAMS), 'case9': _jl(CASE9_PARAMS),
           'w153c': _jl(W153C_REL), 'w153d': _jl(W153D_REL), 'w139_c52': _jl(W139_C52_RES), 'inputs_extra': {},
           'p2_sha': _sha(P2_REL), 'p2_commit': W160._last_commit(P2_REL),
           'p3_commit_date': W160._git('log', '-1', '--format=%ad', '--date=short', '--', P3_REL),
           'code': {f: _text(f) for f in CODE_FILES + (NOTE_5456, BRIEF, 'helper_functions.py')}}
    rec['w101_rows'] = {nm: _rows(os.path.join(ed, 'per_cycle_record.jsonl')) for nm, ed in w101_eval.items()}
    rec['w101_replays'] = {}
    for nm, ed in w101_eval.items():
        cr = _jl(os.path.join(os.path.dirname(os.path.dirname(ed)), 'campaign_results.json'))
        rec['w101_replays'][nm] = {'G19': cr['gates']['G19_replay_bitwise_1_N_every_field'],
                                   'through': cr['gate_detail']['G13']['summary']['replay_bitwise_through_cycle'],
                                   'N': rec['w101_summary']['reports'][nm]['N']}
    w98e = _jl(W98_EVAL)
    rec['w98_eval_key_x0'] = w98e['candidate_key']
    rec['w98_x0_canonical_all_zero'] = all(v == [0.0, 0.0] for v in w98e['candidate_canonical']['nodes'].values())
    # post-certification movement, recomputed from the records
    q3 = {k: v['gross_operational_cost'] for k, v in _rows(W137_CELL3).items()}
    q5 = {k: v['gross_operational_cost'] for k, v in _rows(W139_C52_REC).items()}
    q1 = {k: v['gross_operational_cost'] for k, v in _rows(W132_CELL1).items()}
    mv = {'b_2a0ba8b2 (v3 run, Q(213) - Q(173); v4 k* 173)': q1[max(q1)] - q1[173],
          'b_4649234b (v4 run, Q(193) - Q(148); v5 k* 148)': q3[max(q3)] - q3[148],
          'd_c52e1670 (v5 run, Q(198) - Q(150); v6 k* 150)': q5[max(q5)] - q5[150],
          'd_c52e1670 (v5 run, Q(198) - Q(148); last-pair reading k* 148)': q5[max(q5)] - q5[148]}
    w141d = rec['w141']['d_c52e1670_detail']['per_variant']
    rec['postcert'] = {'movement_eur': mv, 'over_tau': {k: v / TAU for k, v in mv.items()},
                       'w141_V3_10_over_tau': w141d['V3_10']['Q_last_recorded_minus_Q_k_star_over_tau'],
                       'w141_V1_over_tau': w141d['V1']['Q_last_recorded_minus_Q_k_star_over_tau'],
                       'last_cycles': {'b_2a0ba8b2_v3': max(q1), 'b_4649234b_v4': max(q3), 'd_c52e1670_v5': max(q5)}}
    envnow = env_now()
    w101 = rec['w101_summary']['reports']
    bes = {}
    for nm in ('x0', 'n7_4h_e1', 'c_star'):
        r = w101[nm]
        rows = rec['w101_rows'][nm]
        k0, n, endc = r['k0'], r['N'], (r.get('k_star') or r.get('k_cap'))
        q = {k: rows[k]['gross_operational_cost'] for k in rows}
        bes[nm] = {'status': r['status'], 'k0': k0, 'N': n, 'end': endc, 's_signed': r['s_signed'],
                   'Q_end_minus_Q_N_recomputed': q[endc] - q[n], 'Q_end_minus_Q_k0': q[endc] - q[k0],
                   'max_abs_Q_k_minus_Q_N': max(abs(q[k] - q[n]) for k in range(n, endc + 1)),
                   'max_abs_Q_k_minus_Q_k0': max(abs(q[k] - q[k0]) for k in range(k0, endc + 1)),
                   'boyd_all_pass_k0_minus_1': rows[k0 - 1].get('boyd_all_pass'),
                   'boyd_all_pass_k0': rows[k0].get('boyd_all_pass')}
    rec['reference_beside'] = bes

    # ---- 2. carry-over -------------------------------------------------------------------------------------------
    w161doc = _jl(W161_JSON)
    w153c = rec['w153c']
    carried, carry_meta = carry_over(fz, raw, v3_norm, w153c, w161doc)
    # ---- 3. new and non-manuscript checks ----------------------------------------------------------------------
    newc = new_checks(fz, v3_norm, rec, envnow)
    xch = nonmanuscript_checks(rec)
    # ---- 4. definitions ----------------------------------------------------------------------------------------
    dch = definition_checks(fz, rec)

    # ---- 5. token coverage -------------------------------------------------------------------------------------
    in_v3 = [c for c in carried if c['placement'] != 'not in the v3 prose'] + newc + xch
    key = {(x['line'], x['text'], x['occ']): x for x in toks}
    covered = {}
    stale = []
    for c in in_v3:
        for ln, tx, oc in c['tokens']:
            if (ln, tx, oc) not in key:
                stale.append((c['id'], ln, tx, oc))
            covered.setdefault((ln, tx, oc), []).append(c['id'])
    unchecked = []
    for k, (cat, why) in UNCHECKED.items():
        if k not in key:
            stale.append(('UNCHECKED', *k))
        elif k in covered:
            stale.append(('UNCHECKED-and-checked', *k))
        else:
            unchecked.append({'line': k[0], 'text': k[1], 'occ': k[2], 'category': cat, 'reason': why})
    unassigned = [x for x in toks if (x['line'], x['text'], x['occ']) not in covered and
                  (x['line'], x['text'], x['occ']) not in UNCHECKED]
    for x in toks:
        x['covered_by'] = covered.get((x['line'], x['text'], x['occ']), [])
    # fragment presence for every check touching v3
    frag_missing = []
    for c in carried:
        if c['placement'] == 'v3 rewrote the sentence (carried to the v3 fragment)' and not c['fragment_found_v3']:
            frag_missing.append(c['id'])
    for c in newc:
        if c['fragment_v3'] and norm_frag(c['fragment_v3']) not in v3_norm:
            frag_missing.append(c['id'])
    # every carried check that is in v3 must name its tokens (verdict checks excepted)
    no_tok = [c['id'] for c in carried if c['placement'] != 'not in the v3 prose' and not c['tokens']
              and c['id'] not in ('S2c', 'S2e', 'V1b')]

    checks_v3 = [c for c in in_v3 if c['id'][0] != 'X']
    mism = [c for c in in_v3 if c['status'] == 'MISMATCH']
    appr = [c for c in in_v3 if c['status'] == 'approximate']
    ucat = {}
    for u in unchecked:
        ucat[u['category']] = ucat.get(u['category'], 0) + 1
    summary = {
        'tokens_total': len(toks), 'tokens_scope': sum(1 for x in toks if x['region'] == 'scope'),
        'tokens_nonmanuscript': sum(1 for x in toks if x['region'] != 'scope'),
        'n_checks_in_v3': len(checks_v3), 'n_match': sum(c['status'] == 'match' for c in checks_v3),
        'n_mismatch': sum(c['status'] == 'MISMATCH' for c in checks_v3),
        'n_approximate': sum(c['status'] == 'approximate' for c in checks_v3),
        'n_no_table_counterpart': sum(c['status'] == 'no table counterpart' for c in checks_v3),
        'mismatch_ids': [c['id'] for c in mism], 'approximate_ids': [c['id'] for c in appr],
        'w161_carried_verbatim': sum(c['placement'] == 'verbatim in v3' for c in carried),
        'w161_carried_rewritten': sum(c['placement'].startswith('v3 rewrote') for c in carried),
        'w161_not_in_v3': sum(c['placement'] == 'not in the v3 prose' for c in carried),
        'n_new': len(newc), 'n_x': len(xch), 'x_status': {c['id']: c['status'] for c in xch},
        'n_unchecked_tokens': len(unchecked), 'unchecked_by_category': ucat,
        'every_token_assigned': not unassigned and not stale,
        'definition_verdicts': {d['id']: d['verdict'] for d in dch},
    }
    checks = {
        'paragraphs_v3_sha_pinned': inputs['PARAGRAPHS_V3']['sha256'] == P3_SHA,
        'frozen_json_unchanged': _sha(FZ_REL) == FZ_SHA,
        'w161_counterparts_rederived_equal_to_committed': carry_meta['all_same_counterpart'],
        'w161_count_164': carry_meta['n_w161_checks'] == 164 == carry_meta['n_w161_stored'],
        'w161_evaluator_self_test_on_v1': carry_meta['w161_evaluator_self_test_on_v1_reproduces'],
        'w162_evaluator_self_test_on_v2_values': carry_meta['w162_evaluator_self_test_all_reproduce'],
        'every_token_assigned_no_stale_assignment': not unassigned and not stale,
        'every_v3_fragment_found': not frag_missing,
        'every_carried_check_names_its_tokens': not no_tok,
    }
    failed += [k for k, v in checks.items() if v is not True]
    guards, pk, guards_ok = guards_state()
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    out = {'schema': 'p515_s53_w162_paragraphs_v3_figure_check', 'version': 1,
           'stage': 'P5.15 W162 -- figure check of paragraphs_v3.md (Addendum 66)',
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': W160._git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean},
           'text': {'path': P3_REL, 'sha256': P3_SHA, 'commit': P3_COMMIT, 'scope': f'lines 1-{end} (above '
                    f'"{SCOPE_END}"); manuscript scope lines {MANUSCRIPT_FROM_LINE}-{end}', 'edited': False},
           'frozen_tables': {'path': FZ_REL, 'sha256': FZ_SHA},
           'inputs': inputs, 'inputs_read_in_checks': rec['inputs_extra'], 'environment_now': envnow,
           'summary': summary, 'checks_in_v3': checks_v3, 'nonmanuscript_checks': xch,
           'mismatches': mism, 'approximate': appr, 'unchecked_tokens': unchecked,
           'w161_carry_over': carried, 'w161_carry_meta': carry_meta,
           'w161_not_in_v3': [{'id': c['id'], 'fragment_w161': c['fragment_w161'], 'written_v2': c['written_v2'],
                               'w161_status': c['w161_status']} for c in carried
                              if c['placement'] == 'not in the v3 prose'],
           'definition_checks': dch, 'reference_evaluations_beside': None,
           'post_certification_movement': rec['postcert'],
           'token_inventory': toks, 'unassigned_tokens': unassigned, 'stale_assignments': [list(s) for s in stale],
           'fragments_not_found': frag_missing, 'carried_without_tokens': no_tok,
           'checks': checks, 'failed': sorted(set(failed)), 'guards': guards, 'pickle_guard': pk,
           'exit_code': code, 'wall_s': time.time() - t0}
    out['reference_evaluations_beside'] = rec['reference_beside']

    written = {}

    def wr(rel, data):
        with open(os.path.join(REPO, rel), 'xb') as h:
            h.write(data if isinstance(data, bytes) else data.encode('utf-8'))
        written[rel] = _sha(rel)
    wr(OUT_JSON, GRIO.dumps(out, indent=1, sort_keys=True) + '\n')
    wr(OUT_MD, md_summary(out))
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for v in inputs.values():
        man[v['path']] = v['sha256']
    man.update(rec['inputs_extra'])
    wr(OUT_MAN, GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    # ---- log -----------------------------------------------------------------------------------------------------
    s = summary
    _log(f"[{tag}] tokens: {s['tokens_total']} ({s['tokens_scope']} in scope, {s['tokens_nonmanuscript']} non-manuscript); "
         f"every token assigned {s['every_token_assigned']} (unassigned {len(unassigned)}, stale {len(stale)})")
    _log(f"[{tag}] W161 carry-over: {carry_meta['n_w161_checks']} re-derived, same counterpart as committed "
         f"{carry_meta['all_same_counterpart']}; verbatim {s['w161_carried_verbatim']}, rewritten "
         f"{s['w161_carried_rewritten']}, not in v3 {s['w161_not_in_v3']}")
    _log(f"[{tag}] checks in v3: {s['n_checks_in_v3']} -- match {s['n_match']}, MISMATCH {s['n_mismatch']} "
         f"{s['mismatch_ids']}, approximate {s['n_approximate']} {s['approximate_ids']}, no table counterpart "
         f"{s['n_no_table_counterpart']}; new {s['n_new']}; X {s['x_status']}")
    for c in mism + appr:
        _log(f"[{tag}]   {c['id']} {c['status']}: written {c.get('written_v3')!r}, counterpart "
             f"{c.get('value_at_written_precision', c.get('table_value_at_written_precision'))!r} -- "
             f"{c.get('note') or ''}")
    for u in unchecked:
        _log(f"[{tag}]   UNCHECKED line {u['line']} {u['text']!r} [{u['category']}] {u['reason']}")
    for d in dch:
        _log(f"[{tag}]   {d['id']}: {d['verdict']}")
    for nm, b in bes.items():
        _log(f"[{tag}]   reference {nm}: s {b['s_signed']:+,.2f} (Q(end) − Q(N {b['N']})), Q(end) − Q(k0 {b['k0']}) "
             f"{b['Q_end_minus_Q_k0']:+,.2f}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    if unassigned:
        _log(f'[{tag}] UNASSIGNED tokens: ' + ', '.join(f"{x['line']}:{x['text']}#{x['occ']}" for x in unassigned))
    if stale or frag_missing or no_tok:
        _log(f'[{tag}] stale {stale}; fragments not found {frag_missing}; carried without tokens {no_tok}')
    if failed:
        _log(f'[{tag}] FAILED: {sorted(set(failed))}')
    _log(f"[{tag}] wrote {', '.join(f'{k} {v[:8]}' for k, v in written.items())}")
    _log(f"[{tag}] guards {[(k, v['counts']['permitted_solve'], v['counts']['blocked_solve'], v['verify_0_failures']) for k, v in guards.items()]}; "
         f"pickle ok {pk['ok']}; exit {code}; wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(code)


if __name__ == '__main__':
    main()
