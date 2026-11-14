"""P5.15 Step 6, Planner task W171a -- NUMBER CHECK OF THE MANUSCRIPT .tex FILES AT OVERLEAF 42794d4 (the W164 checker,
DECLARATIONS VERSION 2). ZERO SOLVES, NO MODEL LOADS. A NEW FILE: the W164 script is IMPORTED (it imports W163 -> W162 ->
W161 -> W160, each arming its guards) and its tokenizer, value index, record helpers and version-1 declarations are
USED; W160-W164 are not edited. Every output is a new file opened 'x' in a new directory.

WHAT CHANGES AGAINST W164 (version 1 -> version 2)
  1. The manuscript is the Overleaf clone at 42794d4: section 2 of main.tex is revised (l. 315-898, Addendum 68 and
     STEP6_ROUND1_CORRECTIONS.md section A), the response letter is revised (section B, B.1-B.9), and a draft source
     `section2_expert_draft.tex` (NOT \\input into main.tex) is new. The draft is checked in a separate scope
     "draft, not compiled" and reported apart from the manuscript counts.
  2. Every W164 declaration is RE-LOCATED in the revised texts and reported as carried (fragment found once, evaluated
     unchanged), superseded (fragment found, re-pointed by a version-2 declaration for a stated reason) or removed
     (fragment gone; the version-2 declaration that replaced it is named).
  3. New declarations for every number of the revised section 2 and the revised letter. Parameter values are checked
     against the code / case file / frozen spec they come from (file:line or JSON field, with the file's sha256), not
     against the addendum's text. The values section 2 defers to "Section 3.5" (not yet in main.tex) are checked the same
     way in a parameter table (Addendum 68's list, item by item), so that they are verified before they are written.
  4. Figures the letter quotes from "the submitted version" are checked against the committed submitted source
     `manuscript_submitted/main.tex` (W164 used the Overleaf main.tex of 6191c6c, which already carried first-reply
     edits).
  5. Reviewer quotations: every \\rcomment{...} of the letter is compared with the reviewers' document
     `manuscript_review/Reviewers Comments.docx` (python-docx is not installed: word/document.xml is read with zipfile
     and parsed with xml.etree). Per segment (enumerate item / paragraph): verbatim / differs (the diff is shown) / not
     found. Every numeric token inside a quotation gets "reviewer quotation, verified" (its characters lie in a part
     that matches the document verbatim) or MISMATCH vs reviewers' document.
  6. The main.tex line rules are RE-PINNED to main.tex sha256 7effd898...: lines after section 2 moved by -9 (the
     revision map's submitted-version regions keep that status at their new lines); the revised section 2 is never
     handed to the automatic value index -- every number there is declared or assigned by a stated rule (notation in
     equations / Algorithm 1, structural number words, cross-references), otherwise the run fails.
  The refusal behaviours of W164 stay: declared Overleaf commit and the sha256 of every *.tex must match; the output
  directory must be new (only the launch log); every token assigned; a stale declaration or a token claimed twice
  exits 3.

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, with every guard the W164 chain arms; `pickle.load` / `pickle.loads` blocked for the whole run, every counter
verified at 0. git reads are read-only (`git show` / `git log` / `git status` / `git rev-parse`, also in the clone).

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w171_manuscript_check && \\
    mkdir data/SRP1/Results/P515S53/w171_manuscript_check/overleaf_42794d4 && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w171_manuscript_number_check.py \\
        --overleaf-commit 42794d4 --expect main.tex=<sha256> ... (every *.tex of the clone) \\
        > data/SRP1/Results/P515S53/w171_manuscript_check/overleaf_42794d4/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w171_manuscript_check/overleaf_42794d4/w171_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w171_manuscript_check/overleaf_42794d4/w171_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w171_manuscript_number_check.py \\
        --overleaf-commit 42794d4 --post-run
Exit: 0 = written, every integrity check holds and the guards are at 0 (MISMATCHES are FINDINGS and never change the
exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard fault (nothing written).
"""
import argparse
import collections
import difflib
import glob
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W171a manuscript number check v2 (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W171: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W171: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w164_manuscript_number_check as W164  # noqa: E402 -- arms its guards, imports W163 -> W160 (none edited)

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

W163, W162, W161, W160 = W164.W163, W164.W162, W164.W161, W164.W160
GUARDS = W160.W157._dedupe((('w171_manuscript_number_check', GUARD),) + tuple(W164.GUARDS))

_log, _sha, _sha_bytes, _git, _jl, _text = W160._log, W160._sha, W160._sha_bytes, W160._git, W162._jl, W162._text
R, num, wv, count_eq, unch, ntc, D, w163ref = W164.R, W164.num, W164.wv, W164.count_eq, W164.unch, W164.ntc, W164.D, \
    W164.w163ref
at_prec = W164.at_prec

SCRIPT_REL = os.path.basename(__file__)
W164_SCRIPT, W164_SCRIPT_COMMIT = 'p515_s53_w164_manuscript_number_check.py', '77fcf136'
W164_RESULTS = os.path.join(W160.S53, 'w164_manuscript_check', 'overleaf_6191c6c', 'w164_manuscript_number_check.json')
W164_RESULTS_COMMIT = 'b005ac7f'
OUT_ROOT = os.path.join(W160.S53, 'w171_manuscript_check')
DEFAULT_MANUSCRIPT_DIR = W164.DEFAULT_MANUSCRIPT_DIR
DECLARATIONS_VERSION = 2

LETTER, HIGHLIGHTS, COVER, MAIN = W164.LETTER, W164.HIGHLIGHTS, W164.COVER, W164.MAIN
DRAFT = 'section2_expert_draft.tex'
SUBMITTED_MAIN = os.path.join('manuscript_submitted', 'main.tex')
SUBMITTED_MAIN_SHA = 'ca07d7dbd881e1f9a6889f42e78e38529504602a860906f526eca40b8d624c62'
DOCX = os.path.join('manuscript_review', 'Reviewers Comments.docx')
DOCX_SHA = 'c7afdc8bb3d32fce41116a632397799c82df6c6f11316fb5fa013b0c4ed6e21c'
CORR_MD = 'STEP6_ROUND1_CORRECTIONS.md'
V6_SPEC = W162.V6_SPEC
SPEC_3X3 = os.path.join(W160.S53, 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'campaign_spec_s53_w91_3x3_pair_231558f0.json')
INST_3X3 = os.path.join(W160.S53, 'w89_3x3', 'instance', 'SRP1__s53_3x3.json')
EXT_SPEC_V3 = os.path.join(W160.S53, 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json')
SRP1_PARAMS = W164.SRP1_PARAMS
ESS_PARAMS = W164.ESS_PARAMS
CASE_PARAMS = {n: os.path.join('data', 'SRP1', n, f'{n}_params.json') for n in ('case9', 'case33_1', 'case33_2',
                                                                                  'case33_3')}
CODE = {k: f'{k}.py' for k in ('definitions', 'model_construction_helpers', 'shared_energy_storage',
                               'shared_energy_storage_data', 'network', 'shared_resources_planning', 'admm_parameters',
                               'admm_anderson_acceleration', 'settling_criterion', 'settling_criterion_v2',
                               'settling_criterion_v5', 'settling_criterion_v6', 'p515_s47_phase_b_record',
                               'p515_s51_f2_phase_b', 'p515_s53_w132_resettle_v3_campaign')}

OUT_NAMES = {'json': 'w171_manuscript_number_check.json', 'md': 'w171_manuscript_number_check.md',
             'man': 'manifest_sha256.json', 'log': 'launch.log', 'typing_json': 'w171_bool_typing_test.json',
             'typing_log': 'w171_bool_typing_test.log', 'post': 'manifest_post_run_sha256.json'}

STATUSES = ('match', 'MISMATCH', 'approximate', 'no table counterpart', 'submitted-version figure',
            'reviewer quotation, verified', 'unchecked')
MANUSCRIPT_FILES = (COVER, HIGHLIGHTS, MAIN, LETTER)

# main.tex line rules, re-pinned to this main.tex (Overleaf 42794d4). Section 2 is l. 315-898; every line after the
# revised block (old l. 907 onward) moved by -9 against the W164 pin (main.tex 3cedbb6e).
MAIN_LINE_RULES_SHA = '7effd8983df50bdce94e1bc2de07d95d79098d82ce6e0fa201ce711b515230b7'
SEC2 = (315, 898)
SHIFT = -9
MAIN_REGIONS = tuple(((a + SHIFT, b + SHIFT), key, kind, why) for (a, b), key, kind, why in W164.MAIN_REGIONS)
APP_A = (1409 + SHIFT, 1611 + SHIFT)
NETWORK_TABLES = (974 + SHIFT, 1067 + SHIFT)
APP_BCD = (1612 + SHIFT, 1972 + SHIFT)


# ======================================================================================================================
#  1. records (W164 Records + the submitted source, the reviewers' document, code, specs)
# ======================================================================================================================
class Src:
    """A text file of the repository with line lookup: `cite(rx)` -> (line, text) of the unique match (KeyError on 0 or
    several, so a declaration whose code reference moved goes stale loudly)."""

    def __init__(self, rel):
        self.rel = rel
        self.text = _text(rel)
        self.lines = self.text.split('\n')
        self.sha = _sha(rel)

    def cite(self, rx, expect=1):
        r = re.compile(rx)
        hits = [(i + 1, ln) for i, ln in enumerate(self.lines) if r.search(ln)]
        if expect is not None and len(hits) != expect:
            raise KeyError(f'{self.rel}: /{rx}/ matched {len(hits)} lines (expected {expect})')
        return hits

    def ref(self, rx, expect=1):
        hits = self.cite(rx, expect)
        return f"{self.rel}:{','.join(str(h[0]) for h in hits)} (sha256 {self.sha[:8]})"

    def value(self, rx, expect=1, group=1, cast=float):
        hits = self.cite(rx, expect)
        return cast(re.search(rx, hits[0][1]).group(group))


class RecordsV2(W164.Records):
    def __init__(self, inputs, man_doc, sub_doc):
        super().__init__(inputs, man_doc)
        self.sub = sub_doc
        self.v6 = _jl(V6_SPEC)
        self.s3x3 = _jl(SPEC_3X3)
        self.i3x3 = _jl(INST_3X3)
        self.ext3 = _jl(EXT_SPEC_V3)
        self.cp = {k: _jl(v) for k, v in CASE_PARAMS.items()}
        self.code = {k: Src(v) for k, v in CODE.items()}
        self.srcs = {'SRP1_PARAMS': Src(SRP1_PARAMS), 'ESS_PARAMS': Src(ESS_PARAMS), 'STEP4': Src(W164.STEP4),
                     **{f'CP_{k}': Src(v) for k, v in CASE_PARAMS.items()}}
        self.cells_all = {**self.t['cells'], **self.t['cells_appended_w160']}

    def c(self, name):
        return self.code[name]

    def sub_line(self, frag):
        hits, _n = self.sub.find_fragment(frag)
        if len(hits) != 1:
            raise KeyError(f'submitted main.tex fragment {frag!r}: {len(hits)} hits')
        return self.sub.flat_idx[hits[0]][0]


def jsrc(X, key):
    i = X.inputs[key]
    return f"{i['path']} (sha256 {i['sha256'][:8]})"


def jline(X, key, rx, expect=1):
    """the line of a JSON case file that carries `rx` (for a file:line citation of a JSON field)."""
    return X.srcs[key].ref(rx, expect)


# ======================================================================================================================
#  2. the parameter table: Addendum 68's list of values in force, each checked against its source
# ======================================================================================================================
def _pv(pid, quantity, stated, found, status, sources, note=None):
    return {'id': pid, 'quantity': quantity, 'stated_addendum_68': stated, 'found': found, 'status': status,
            'sources': sources, 'note': note}


def _eq(a, b, rel=1e-12):
    return a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float)) and
                      abs(a - b) <= rel * max(abs(a), abs(b), 1.0))


def median_block_weight(inst):
    """w_b = Y_y D_d / (1 + r)^(y - y0) over the TSO block set and the three DSOs' (the code's block set:
    shared_resources_planning._compute_median_admm_block_weight + _get_admm_block_weight); the four networks share
    years, days and r."""
    years, days, r = inst['Years'], inst['Days'], inst['DiscountFactor']
    y0 = int(sorted(years, key=int)[0])
    w = [float(years[y]) * float(days[d]) / (1.0 + r) ** (int(y) - y0) for y in years for d in days]
    n_networks = 1 + len(inst['DistributionNetworks'])
    allw = sorted(w * n_networks)
    k = len(allw)
    med = allw[k // 2] if k % 2 else 0.5 * (allw[k // 2 - 1] + allw[k // 2])
    return med, len(allw)


def parameter_checks(X):
    P = []
    defs, mch, ses, sesd = X.c('definitions'), X.c('model_construction_helpers'), X.c('shared_energy_storage'), \
        X.c('shared_energy_storage_data')
    net, srp, admm, aa = X.c('network'), X.c('shared_resources_planning'), X.c('admm_parameters'), \
        X.c('admm_anderson_acceleration')
    sp, ep = X.srp1p['admm'], X.ess
    s_sp, s_ep = X.srcs['SRP1_PARAMS'], X.srcs['ESS_PARAMS']

    # efficiencies: the shared-ESS class defaults, never overwritten for a shared ESS in the production modules
    eff_ch = ses.value(r'self\.eff_ch = ([0-9.]+)')
    eff_dch = ses.value(r'self\.eff_dch = ([0-9.]+)')
    assigns = []
    for k in ('shared_resources_planning', 'shared_energy_storage_data', 'network', 'model_construction_helpers'):
        assigns += [(X.c(k).rel, i) for i, ln in X.c(k).cite(r'\.eff_d?ch\s*=[^=]', expect=None)]
    only_local = all(rel == 'network.py' and "energy_storage_data['eff_" in X.c('network').lines[i - 1]
                     for rel, i in assigns)
    P.append(_pv('P01', 'eta_ch (shared ESS, network models and agent)', 0.97, eff_ch,
                 'match' if eff_ch == 0.97 and only_local else 'MISMATCH',
                 [ses.ref(r'self\.eff_ch = '), mch.ref(r'eff_ch = sess\.eff_ch'), sesd.ref(r'eff_ch = shared_energy_storage\.eff_ch')],
                 f'class default; assignments to .eff_ch/.eff_dch in production modules: {assigns} (local-ESS loader '
                 f'only: {only_local})'))
    P.append(_pv('P02', 'eta_dch', 0.96, eff_dch, 'match' if eff_dch == 0.96 and only_local else 'MISMATCH',
                 [ses.ref(r'self\.eff_dch = ')]))
    soc_min = defs.value(r'^ENERGY_STORAGE_MIN_ENERGY_STORED = ([0-9.]+)')
    soc_max = defs.value(r'^ENERGY_STORAGE_MAX_ENERGY_STORED = ([0-9.]+)')
    soc0 = defs.value(r'^ENERGY_STORAGE_RELATIVE_INIT_SOC = ([0-9.]+)')
    P.append(_pv('P03', 'SoC^Min (fraction of E^Av)', 0.10, soc_min, 'match' if soc_min == 0.10 else 'MISMATCH',
                 [defs.ref(r'^ENERGY_STORAGE_MIN_ENERGY_STORED'), mch.ref(r'soc_min = m\.shared_es_e_rated_fixed\[e\] \* ENERGY_STORAGE_MIN')],
                 'shared_es_e_rated_fixed = e_available / s_base (shared_resources_planning, every cycle): the '
                 'fraction applies to the degraded available energy'))
    P.append(_pv('P04', 'SoC^Max', 0.90, soc_max, 'match' if soc_max == 0.90 else 'MISMATCH',
                 [defs.ref(r'^ENERGY_STORAGE_MAX_ENERGY_STORED'), mch.ref(r'soc_max = m\.shared_es_e_rated_fixed\[e\] \* ENERGY_STORAGE_MAX')]))
    P.append(_pv('P05', 'SoC^0 (initial = closure target)', 0.5, soc0, 'match' if soc0 == 0.5 else 'MISMATCH',
                 [defs.ref(r'^ENERGY_STORAGE_RELATIVE_INIT_SOC'),
                  mch.ref(r'soc_prev = m\.shared_es_e_rated_fixed\[e\] \* ENERGY_STORAGE_RELATIVE_INIT_SOC'),
                  mch.ref(r'final_soc = m\.shared_es_e_rated_fixed\[e\] \* ENERGY_STORAGE_RELATIVE_INIT_SOC')]))
    frac = mch.value(r'^ESS_DAY_BALANCE_SLACK_FRACTION = ([0-9.]+)')
    eqt = defs.value(r'^EQUALITY_TOLERANCE = ([0-9.e-]+)')
    dbal = {k: v['slacks']['shared_ess']['day_balance'] for k, v in X.cp.items()}
    P.append(_pv('P06', 'closure slack bound eps^Cl (fraction of E^Av)', 0.05, frac,
                 'match' if frac == 0.05 and all(dbal.values()) else 'MISMATCH',
                 [mch.ref(r'^ESS_DAY_BALANCE_SLACK_FRACTION'),
                  mch.ref(r'slack_ub = 0\.0 if inactive else e_capacity \* ESS_DAY_BALANCE_SLACK_FRACTION \+ EQUALITY_TOLERANCE')]
                 + [jline(X, f'CP_{k}', r'"day_balance": true', expect=None) for k in X.cp],
                 f'slacks.shared_ess.day_balance per case file: {dbal}'))
    P.append(_pv('P07', 'closure slack numerical term', 1e-5, eqt, 'match' if eqt == 1e-5 else 'MISMATCH',
                 [defs.ref(r'^EQUALITY_TOLERANCE = ')],
                 'in model units: per unit at baseMVA 100 (the slack and e_capacity are e_available / s_base), i.e. '
                 '1e-3 MWh -- STEP6_ROUND1_CORRECTIONS A.3(e) writes "10^-5 p.u."'))
    pen = defs.value(r'^PENALTY_SHARED_ESS_BALANCE = ([0-9.e]+)')
    P.append(_pv('P08', 'closure slack penalty c^Cl (EUR/MWh)', 1e3, pen, 'match' if pen == 1e3 else 'MISMATCH',
                 [defs.ref(r'^PENALTY_SHARED_ESS_BALANCE'),
                  mch.ref(r'total \+= base \* PENALTY_SHARED_ESS_BALANCE \* \(model\.slack_shared_es_soc_final_up')],
                 'objective term base * 1e3 * slack_pu with base = network.baseMVA (100): 1e3 per MWh of slack'))
    comp = defs.value(r'^SMALL_TOLERANCE = ([0-9.e-]+)')
    model_kind = {k: v['shared_ess_model'] for k, v in X.cp.items()}
    P.append(_pv('P09', 'network complementarity eps^C (normalised)', 1e-4, comp,
                 'match' if comp == 1e-4 and set(model_kind.values()) == {'BILINEAR_RELAXATION'} else 'MISMATCH',
                 [defs.ref(r'^SMALL_TOLERANCE = '), defs.ref(r'^ESS_COMPLEMENTARITY_TOLERANCE = SMALL_TOLERANCE'),
                  mch.ref(r'\* m\.shared_es_pdch_hat\[e, s_m0, s_o0, p\]\) <= ESS_COMPLEMENTARITY_TOLERANCE')],
                 f'shared_ess_model per case file: {model_kind}; hat = P / S_rated_fixed (hat-link rows)'))
    eps = defs.value(r'^EPS_ESSO_THROUGHPUT = ([0-9.e-]+)')
    P.append(_pv('P10', 'eps^E (ESSO throughput regularisation)', 1e-5, eps, 'match' if eps == 1e-5 else 'MISMATCH',
                 [defs.ref(r'^EPS_ESSO_THROUGHPUT = '), sesd.ref(r'slack_penalty \+= EPS_ESSO_THROUGHPUT \* throughput')]))
    cs = defs.value(r'^PENALTY_ESSO_SLACK = ([0-9.e]+)')
    P.append(_pv('P11', 'c^sigma (ESSO P-net slack penalty)', 1e3, cs, 'match' if cs == 1e3 and ep['slacks'] else 'MISMATCH',
                 [defs.ref(r'^PENALTY_ESSO_SLACK = '), sesd.ref(r'slack_penalty \+= PENALTY_ESSO_SLACK \* \(model\.slack_es_pnet_up'),
                  jline(X, 'ESS_PARAMS', r'"slacks": true')], f"ESS params slacks = {ep['slacks']}"))
    sref = sp['shared_ess_reference_rating_mva']
    n2 = len(srp.cite(r'/ \(2 \* shared_ess_rating\)', expect=None))
    P.append(_pv('P12', 'S_ref (MVA); normalisation 2 S_ref', '2.5 (2 S_ref = 5)', sref,
                 'match' if sref == 2.5 and n2 > 0 else 'MISMATCH',
                 [jline(X, 'SRP1_PARAMS', r'"shared_ess_reference_rating_mva"'),
                  srp.ref(r'return float\(reference_mva\) / s_base'),
                  f"shared_resources_planning.py: {n2} lines divide by (2 * shared_ess_rating)"],
                 f"floor {sp['shared_ess_normalization_floor_mva']} MVA inactive while S_ref is set "
                 '(_shared_ess_admm_normalization_mva returns reference_mva)'))
    rho = sp['rho']
    ok_rho = (set(rho['v'].values()) == {0.0077} and set(rho['pf'].values()) == {0.198} and
              set(rho['ess'].values()) == {0.01})
    P.append(_pv('P13', 'rho initial v / pf / ess', '0.0077 / 0.198 / 0.01', rho, 'match' if ok_rho else 'MISMATCH',
                 [jline(X, 'SRP1_PARAMS', r'"case9": 0\.0077'), jline(X, 'SRP1_PARAMS', r'"case9": 0\.198'),
                  jline(X, 'SRP1_PARAMS', r'"esso": 0\.01')], 'every agent (case9, case33_1..3; ess also the ESSO)'))
    pu = sp['penalty_update']
    ok_pu = (pu['residual_balance_ratio'] == 5.0 and pu['residual_balance_ratio_pf_decrease'] == 3.0 and
             pu['increase_factor'] == 1.5 and pu['decrease_factor'] == 1.5 and pu['min'] == 1e-4 and pu['max'] == 1e4
             and sp['adaptive_penalty'] is True)
    P.append(_pv('P14', 'residual balancing: ratio 5 (pf decrease 3), x/÷1.5, clamp [1e-4, 1e4]',
                 '5 / 3 / 1.5 / 1.5 / 1e-4 / 1e4', {k: pu[k] for k in ('residual_balance_ratio',
                                                                        'residual_balance_ratio_pf_decrease',
                                                                        'increase_factor', 'decrease_factor', 'min',
                                                                        'max')},
                 'match' if ok_pu else 'MISMATCH',
                 [jline(X, 'SRP1_PARAMS', r'"residual_balance_ratio": 5'), jline(X, 'SRP1_PARAMS', r'"residual_balance_ratio_pf_decrease"'),
                  jline(X, 'SRP1_PARAMS', r'"increase_factor"'), jline(X, 'SRP1_PARAMS', r'"decrease_factor"'),
                  jline(X, 'SRP1_PARAMS', r'"min": 1e-4'), jline(X, 'SRP1_PARAMS', r'"max": 1e4'),
                  jline(X, 'SRP1_PARAMS', r'"adaptive_penalty": true')]))
    ex = pu['balancing_exempt_until']['ess']
    ok_fr = pu['freeze_after_unchanged_cycles'] == 10 and pu['freeze_backstop_cycle'] == 200 and \
        ex == {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}
    P.append(_pv('P15', 'per-channel freeze after 10 unchanged cycles; backstop 200; ESS exempt until dual ratio < 1 on '
                        '5 cycles', '10 / 200 / (< 1, 5)',
                 {'freeze_after_unchanged_cycles': pu['freeze_after_unchanged_cycles'],
                  'freeze_backstop_cycle': pu['freeze_backstop_cycle'], 'ess_exempt': ex},
                 'match' if ok_fr else 'MISMATCH',
                 [jline(X, 'SRP1_PARAMS', r'"freeze_after_unchanged_cycles"'), jline(X, 'SRP1_PARAMS', r'"freeze_backstop_cycle"'),
                  jline(X, 'SRP1_PARAMS', r'"ess": \{"dual_ratio_below"')]))
    bo = sp['tol']['boyd']
    P.append(_pv('P16', 'Boyd eps_abs / eps_rel', '1e-5 / 1e-4', bo,
                 'match' if bo == {'eps_abs': 1e-5, 'eps_rel': 1e-4} else 'MISMATCH',
                 [jline(X, 'SRP1_PARAMS', r'"boyd": \{"eps_abs"'), admm.ref(r"boyd_tolerances = params_data\['tol'\]\.get\('boyd'\)")]))
    mc, rc = sp['minimum_consecutive_converged_cycles'], X.s3x3['required_consecutive_cycles']
    P.append(_pv('P17', 'production exit: consecutive passing cycles', 10, {'SRP1': mc, '3x3_spec': rc},
                 'match' if mc == 10 and rc == 10 else 'MISMATCH',
                 [jline(X, 'SRP1_PARAMS', r'"minimum_consecutive_converged_cycles"'),
                  srp.ref(r'convergence = \(consecutive_converged_cycles >= admm_parameters\.minimum_consecutive_converged_cycles\)'),
                  f'{jsrc(X, "SPEC_3X3")} required_consecutive_cycles']))
    gated = {k: v['cap_rule'] for k, v in X.v6['cells'].items() if v.get('gated')}
    ungated = {k: v['cap_rule'] for k, v in X.v6['cells'].items() if not v.get('gated')}
    ok_g = all(c.get('formula') == 'N_old + 100' and c.get('cap') == X.v6['cells'][k]['N_old'] + 100
               for k, c in gated.items())
    over300 = {k: c.get('cap') for k, c in gated.items() if (c.get('cap') or 0) > 300}
    ok_u = all(c.get('after_first_k0') == 109 and c.get('ceiling') == 300 for c in ungated.values())
    k109 = X.c('settling_criterion_v2').value(r'^CAP_AFTER_K0 = (\d+)', cast=int)
    P.append(_pv('P18', 'cycle caps: 3x3 500; SRP1 gated N_old + 100, ungated min(k0 + 109, 300)',
                 '500 / N_old + 100 / min(k0 + 109, 300)',
                 {'3x3_cap': X.s3x3['cap'], 'v6_gated_cells': len(gated), 'v6_ungated_cells': len(ungated),
                  'CAP_AFTER_K0': k109},
                 'match' if X.s3x3['cap'] == 500 and ok_g and ok_u and k109 == 109 and gated and ungated else 'MISMATCH',
                 [f'{jsrc(X, "SPEC_3X3")} cap', f'{jsrc(X, "V6_SPEC")} cells[*].cap_rule (gated: formula, cap = '
                  'N_old + 100, per-cell ceiling; ungated: after_first_k0, ceiling)', X.c('settling_criterion_v2').ref(r'^CAP_AFTER_K0 = ')],
                 f'gated caps are N_old + 100 with a per-cell ceiling (v6 configuration note "gated N_old + 100 (per-cell '
                 f'ceiling), ungated 300"); gated cells above 300: {over300}'))
    sig = sp['objective_scale']
    P.append(_pv('P19', 'sigma (common objective scale)', 93635360, sig, 'match' if sig == 93635360.0 else 'MISMATCH',
                 [jline(X, 'SRP1_PARAMS', r'"objective_scale"')]))
    m1, n1 = median_block_weight(X.srp1)
    m3, n3 = median_block_weight(X.i3x3)
    k1, k3 = sig / m1, sig / m3
    ok_k = round(k1, 3) == 227210.997 and round(k3, 3) == 386258.694 and sp['esso_al_scale'] == \
        'sigma_over_median_block_weight'
    P.append(_pv('P20', 'kappa_ESSO = sigma / median w_b', '227,210.997 (SRP1); 386,258.694 (3x3)',
                 {'SRP1': round(k1, 6), 'SRP1_median_w_b': m1, 'SRP1_n_blocks': n1, '3x3': round(k3, 6),
                  '3x3_median_w_b': m3, '3x3_n_blocks': n3}, 'match' if ok_k else 'MISMATCH',
                 [jline(X, 'SRP1_PARAMS', r'"esso_al_scale"'), srp.ref(r'al_scale_esso = objective_scale_used / median_block_weight'),
                  srp.ref(r'median_weight = float\(np\.median\(weights\)\)'), jsrc(X, 'SRP1_JSON'), jsrc(X, 'INST_3X3')],
                 'recomputed here from the instance files with the code\'s block set (TSO + 3 DSOs) and weight'))
    df = X.srp1['DiscountFactor']
    P.append(_pv('P21', 'w_b = Y_y D_d 1.02^-(y - y0)', 'Y_y D_d 1.02^-(y-y0)',
                 {'SRP1_DiscountFactor': df, '3x3_DiscountFactor': X.i3x3['DiscountFactor']},
                 'match' if df == 0.02 and X.i3x3['DiscountFactor'] == 0.02 else 'MISMATCH',
                 [srp.ref(r'annualization = 1\.0 / \(\(1\.0 \+ network_data\.discount_factor\) \*\* \(int\(year\) - int\(years\[0\]\)\)\)'),
                  srp.ref(r'return float\(network_data\.years\[year\]\) \* float\(network_data\.days\[day\]\) \* annualization'),
                  jsrc(X, 'SRP1_JSON'), jsrc(X, 'INST_3X3')]))
    aac = sp['anderson_acceleration']
    aa_facts = {'type_II': bool(aa.cite(r'type-II Anderson acceleration', expect=None)),
                'ratchet_safeguard': bool(aa.cite(r'ratchet', expect=None)),
                'cleared_on_rho_change': bool(aa.cite(r'def clear_for_rho_change', expect=None)),
                'cleared_on_solve_failure': bool(aa.cite(r"'reset_reason': 'local solve failure this cycle'", expect=None)),
                'off_when_every_channel_passes': bool(aa.cite(r"action='off \(all channels within Boyd tolerance\)'", expect=None))}
    ok_aa = aac == {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'} and \
        all(aa_facts.values())
    P.append(_pv('P22', 'AA: type-II, memory 5, Tikhonov 1e-10, ratchet safeguard, keep_memory, cleared on rho change and '
                        'on a solve failure, off when every channel passes',
                 'II / 5 / 1e-10 / ratchet / keep_memory / clears / off', {**aac, **aa_facts},
                 'match' if ok_aa else 'MISMATCH',
                 [jline(X, 'SRP1_PARAMS', r'"memory": 5'), jline(X, 'SRP1_PARAMS', r'"regularization": 1e-10'),
                  jline(X, 'SRP1_PARAMS', r'"reject_policy": "keep_memory"'), aa.ref(r'type-II Anderson acceleration'),
                  aa.ref(r'def clear_for_rho_change'), aa.ref(r"'reset_reason': 'local solve failure this cycle'"),
                  aa.ref(r"action='off \(all channels within Boyd tolerance\)'")]))
    tail6 = X.v6['inputs_in_force_now']['configuration_now']['convergence_depth_tail']
    tail3 = X.s3x3['configuration']['convergence_depth_tail']
    tso_cit = X.cp['case9']['solver']['options'].get('compl_inf_tol')
    dso_cit = {k: X.cp[k]['solver']['options'].get('compl_inf_tol') for k in ('case33_1', 'case33_2', 'case33_3')}
    ipopt_def = net.value(r'^IPOPT_DEFAULT_COMPL_INF_TOL = ([0-9.e-]+)')
    ok_t = tail6 == {'compl_inf_tol': 1e-06, 'enabled': True} and tail3 == tail6 and tso_cit == 5e-4 and \
        set(dso_cit.values()) == {None} and ipopt_def == 1e-4
    P.append(_pv('P23', 'tail compl_inf_tol 1e-6 (production: TSO 5e-4, DSO 1e-4)', '1e-6 / 5e-4 / 1e-4',
                 {'v6_tail': tail6, '3x3_tail': tail3, 'tso_case_file': tso_cit, 'dso_case_files': dso_cit,
                  'ipopt_default_in_network_py': ipopt_def}, 'match' if ok_t else 'MISMATCH',
                 [f'{jsrc(X, "V6_SPEC")} inputs_in_force_now.configuration_now.convergence_depth_tail',
                  f'{jsrc(X, "SPEC_3X3")} configuration.convergence_depth_tail', jline(X, 'CP_case9', r'"compl_inf_tol"'),
                  net.ref(r'^IPOPT_DEFAULT_COMPL_INF_TOL = ')],
                 'DSO production value = the IPOPT default (no case33 file sets it)'))
    esso_tol = sesd.cite(r"^ESSO_TOL_OVERRIDES = \{'tol': 1e-10, 'acceptable_tol': 1e-9\}")
    ls = ep['solver']['options']['linear_solver']
    P.append(_pv('P24', 'ESSO IPOPT tol / acceptable_tol / linear solver', '1e-10 / 1e-9 / MA57',
                 {'override': '1e-10 / 1e-9', 'file_tol': ep['solver']['options']['tol'],
                  'file_acceptable_tol': ep['solver']['options']['acceptable_tol'], 'linear_solver': ls},
                 'match' if esso_tol and ls == 'ma57' else 'MISMATCH',
                 [sesd.ref(r"^ESSO_TOL_OVERRIDES = "), sesd.ref(r'option_overrides=ESSO_TOL_OVERRIDES,'),
                  sesd.ref(r'if option_overrides:'), jline(X, 'ESS_PARAMS', r'"linear_solver": "ma57"')],
                 'the override replaces the file tol 1e-6 / acceptable 1e-5 for every ESSO solve (applied after the '
                 'file options)'))
    sol = {k: v['solver']['options'] for k, v in X.cp.items()}
    rec = {k: v['solver'].get('recovery_options') for k, v in X.cp.items()}
    ok_net = all(o['tol'] == 1e-5 and o['acceptable_tol'] == 1e-4 and o['linear_solver'] == 'ma97' for o in sol.values())
    mi = net.cite(r"options\['max_iter'\] = 500")
    rec_ok = {k: (r or {}) == {'acceptable_tol': 1e-4, 'acceptable_iter': 1} for k, r in rec.items()}
    P.append(_pv('P25', 'networks IPOPT tol / acceptable / MA97 / max_iter / recovery acceptable_tol & acceptable_iter',
                 '1e-5 / 1e-4 / MA97 / 500 / 1e-4 & 1', {'options': {k: {kk: o[kk] for kk in ('tol', 'acceptable_tol',
                                                                                            'linear_solver')}
                                                                     for k, o in sol.items()},
                                                         'max_iter_line': bool(mi), 'recovery_options': rec},
                 'match' if ok_net and mi and all(rec_ok.values()) else
                 ('partial' if ok_net and mi and rec_ok['case9'] else 'MISMATCH'),
                 [jline(X, f'CP_{k}', r'"tol": 1e-5') for k in X.cp] + [net.ref(r"options\['max_iter'\] = 500"),
                                                                        net.ref(r'recovery_options = \{')],
                 'recovery acceptable_tol 1e-4 / acceptable_iter 1 is set for the TSO only (case9_params.json); '
                 'case33_1 has no recovery_options and case33_2 / case33_3 an empty one: a DSO recovery is the cold '
                 'restart with the primary options (network.py recovery_options built from the case file)'
                 if not all(rec_ok.values()) else None))
    prox = sp['proximal_regularization']
    ok_g = prox['tso']['gamma_policy'] == 'tied_to_rho' and prox['tso']['tau'] == 0.0 and prox['dso']['enabled'] is False
    P.append(_pv('P26', 'proximal gamma (TSO; DSO off)', 0, {'tso': {k: prox['tso'][k] for k in ('enabled', 'gamma_policy',
                                                                                                  'tau')},
                                                             'dso_enabled': prox['dso']['enabled']},
                 'match' if ok_g else 'MISMATCH',
                 [jline(X, 'SRP1_PARAMS', r'"gamma_policy": "tied_to_rho"'), jline(X, 'SRP1_PARAMS', r'"tau": 0\.0'),
                  srp.ref(r"initialize=tso_gamma_tau \* params\.rho\['v'\]")],
                 'gamma = tau * rho with tau = 0 -> gamma = 0 on every channel'))
    alphas = sorted({json.dumps(c.get('interface_deviation_premium')) for c in X.s3x3['candidates']})
    P.append(_pv('P27', 'row 18 alpha (3x3 only)', 0.5, alphas,
                 'match' if alphas == [json.dumps({'alpha': 0.5, 'floor': None})] else 'MISMATCH',
                 [f'{jsrc(X, "SPEC_3X3")} candidates[*].interface_deviation_premium']))
    pin = defs.value(r'^PENALTY_SCENARIO_DEVIATION = ([0-9.e]+)')
    P.append(_pv('P28', 'interface-voltage pin weight (3x3 only; solver-only)', 9e4, pin, 'match' if pin == 9e4 else 'MISMATCH',
                 [defs.ref(r'^PENALTY_SCENARIO_DEVIATION = '),
                  mch.ref(r'model\.scenario_voltage_pin_weight = pe\.Param\(initialize=PENALTY_SCENARIO_DEVIATION'),
                  srp.ref(r'excluded from the reported Q\(x\) by `_get_operational_recourse_components`')]))
    bm = {k: v.get('baseMVA') for k, v in X.cases.items()}
    P.append(_pv('P29', 'baseMVA (TN and DNs)', 100, sorted(set(bm.values())), 'match' if set(bm.values()) == {100.0} else 'MISMATCH',
                 [f'case files data/SRP1/case9/case9_<y>.json and data/SRP1/case33_<n>/case33_<n>_<y>.json at '
                  f'{"/".join(W164.REP_YEARS)}: baseMVA'], f'{len(bm)} files'))
    la = X.lattice()
    ok_la = la['p_step'] == 0.25 and la['e_step'] == 0.5 and la['ep_min'] == 2.0 and la['ep_max'] == 4.0
    P.append(_pv('P30', 'lattice units and duration bounds', '0.25 MVA / 0.5 MWh / 2-4 h', la, 'match' if ok_la else 'MISMATCH',
                 [f'{W164.A1_CAMPAIGN} LATTICE_P_STEP / LATTICE_E_STEP', f'{jsrc(X, "ESS_PARAMS")} min/max_energy_to_power_factor']))
    n_esso = sesd.cite(r'for node_id in self\.active_distribution_network_nodes:', expect=None)
    P.append(_pv('P31', 'storage agents per cycle (one per interface node, all years and days)', 3,
                 X.n_dns(), 'match' if X.n_dns() == 3 and n_esso else 'MISMATCH',
                 [sesd.ref(r"results\[node_id\] = _optimize\("), jsrc(X, 'SRP1_JSON') + ' DistributionNetworks']))
    return P


# ======================================================================================================================
#  3. reviewer quotations (the letter's \rcomment{} against the reviewers' document)
# ======================================================================================================================
W_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
QUOTE_MAP = {'\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'"}


def docx_text(rel):
    with zipfile.ZipFile(os.path.join(REPO, rel)) as z:
        names = z.namelist()
        xml = z.read('word/document.xml')
    root = ET.fromstring(xml)
    paras = []
    for p in root.iter(W_NS + 'p'):
        s = []
        for el in p.iter():
            if el.tag == W_NS + 't':
                s.append(el.text or '')
            elif el.tag == W_NS + 'tab':
                s.append('\t')
            elif el.tag in (W_NS + 'br', W_NS + 'cr'):
                s.append('\n')
        paras.append(''.join(s))
    chars = []
    for i, p in enumerate(paras):
        for ch in p:
            chars.append((QUOTE_MAP.get(ch, ch), i))
        chars.append((' ', i))
    out = []
    for ch, i in chars:
        if ch.isspace():
            if not out or out[-1][0] == ' ':
                continue
            ch = ' '
        out.append((ch, i))
    return {'text': ''.join(c for c, _ in out), 'para_of': [i for _, i in out], 'n_paragraphs': len(paras),
            'parts': names, 'document_xml_sha256': _sha_bytes(xml), 'method': 'zipfile + xml.etree (word/document.xml '
                                                                          'w:p / w:t / w:tab / w:br; python-docx not '
                                                                          'installed)'}


def latex_quote_segments(raw, base):
    """the LaTeX of one \\rcomment argument -> segments (enumerate items / \\\\ paragraphs) of normalised plain text,
    each char carrying its source offset."""
    segs = [[]]
    i, n = 0, len(raw)

    def emit(ch, off):
        segs[-1].append((ch, off))
    while i < n:
        s = raw[i:i + 24]
        if raw.startswith('\\begin{enumerate}', i):
            j = i + len('\\begin{enumerate}')
            if raw.startswith('[', j):
                j = raw.index(']', j) + 1
            segs.append([])
            i = j
            continue
        if raw.startswith('\\end{enumerate}', i):
            segs.append([])
            i += len('\\end{enumerate}')
            continue
        if raw.startswith('\\item', i):
            segs.append([])
            i += 5
            continue
        if s.startswith('\\\\'):
            segs.append([])
            i += 2
            continue
        if s.startswith('``') or s.startswith("''"):
            emit('"', base + i)
            i += 2
            continue
        if s.startswith('---'):
            emit('\u2014', base + i)
            i += 3
            continue
        if s.startswith('--'):
            emit('\u2013', base + i)
            i += 2
            continue
        if s.startswith('\\%') or s.startswith('\\&') or s.startswith('\\$') or s.startswith('\\_'):
            emit(s[1], base + i + 1)
            i += 2
            continue
        if s.startswith('\\times'):
            emit('\u00d7', base + i)
            i += 6
            continue
        m = re.match(r'\\[A-Za-z]+\*?', s)
        if m:
            i += len(m.group())
            continue
        ch = raw[i]
        if ch in '{}$':
            i += 1
            continue
        if ch == '~' or ch.isspace():
            emit(' ', base + i)
        elif ch == '`':
            emit("'", base + i)
        else:
            emit(QUOTE_MAP.get(ch, ch), base + i)
        i += 1
    out = []
    for seg in segs:
        o = []
        for ch, off in seg:
            if ch == ' ' and (not o or o[-1][0] == ' '):
                continue
            o.append((ch, off))
        while o and o[-1][0] == ' ':
            o.pop()
        if len(o) >= 3:
            out.append(o)
    return out


def _anchor(seg, doc):
    for k in range(0, max(1, len(seg) - 30), 10):
        piece = seg[k:k + 30]
        p = doc.find(piece)
        if p >= 0:
            return p - k
    return None


def quotation_check(letter_doc, dx):
    rows = []
    doc = dx['text']
    for qi, (nm, a, b) in enumerate([m for m in letter_doc.margs if m[0] == 'rcomment']):
        raw = letter_doc.raw[a + 1:b - 1]
        for si, seg in enumerate(latex_quote_segments(raw, a + 1)):
            text = ''.join(c for c, _ in seg)
            l0 = letter_doc.raw.count('\n', 0, seg[0][1]) + 1
            l1 = letter_doc.raw.count('\n', 0, seg[-1][1]) + 1
            row = {'quote': qi, 'segment': si, 'lines': f'{l0}-{l1}' if l1 != l0 else str(l0), 'chars': len(text),
                   'start': text[:70], 'offsets': [off for _, off in seg]}
            p = doc.find(text)
            if p >= 0:
                row.update(status='verbatim', docx_paragraph=dx['para_of'][p], equal_spans=[[0, len(text)]], diffs=[],
                           ratio=1.0)
            else:
                anc = _anchor(text, doc)
                if anc is None:
                    row.update(status='not found', docx_paragraph=None, equal_spans=[], diffs=[], ratio=0.0)
                else:
                    lo, hi = max(0, anc - 60), min(len(doc), anc + len(text) + 60)
                    win = doc[lo:hi]
                    sm = difflib.SequenceMatcher(None, text, win, autojunk=False)
                    ops = sm.get_opcodes()
                    eq = [[i1, i2] for tag, i1, i2, j1, j2 in ops if tag == 'equal']
                    # trim the window margins: leading / trailing docx-only parts are context, not differences
                    core = [op for op in ops if not (op[0] == 'insert' and (op[1] == 0 or op[1] == len(text)))]
                    diffs = []
                    for tag, i1, i2, j1, j2 in core:
                        if tag == 'equal':
                            continue
                        diffs.append({'op': tag, 'letter': text[i1:i2], 'document': win[j1:j2],
                                      'context_before': text[max(0, i1 - 30):i1], 'context_after': text[i2:i2 + 30],
                                      'letter_line': letter_doc.raw.count('\n', 0, seg[min(i1, len(seg) - 1)][1]) + 1})
                    ratio = sum(i2 - i1 for i1, i2 in eq) / max(1, len(text))
                    row.update(status='differs' if ratio >= 0.8 else 'not found',
                               docx_paragraph=dx['para_of'][min(len(doc) - 1, lo + (ops[0][3] if ops else 0))],
                               equal_spans=eq if ratio >= 0.8 else [], diffs=diffs, ratio=ratio)
            rows.append(row)
    return rows


def quotation_token_status(qrows, letter_doc, toks):
    """every numeric token inside a \\rcomment -> reviewer quotation, verified / MISMATCH vs reviewers' document."""
    pos = {}
    for r in qrows:
        for k, off in enumerate(r['offsets']):
            pos.setdefault(off, (r, k))
    out = {}
    for t in toks:
        if t['file'] != LETTER or t['macro'] != 'rcomment' or t['excluded'] or t['scope'] != 'body':
            continue
        o0, o1 = letter_doc.off(t['line'], t['col0']), letter_doc.off(t['line'], t['end0'])
        ks = [pos[o] for o in range(o0, o1) if o in pos]
        if not ks:
            out[t['key']] = R('MISMATCH', 'no normalised position for the token', "reviewers' document", None, None,
                              'token not mapped into a quotation segment (harness)')
            continue
        r = ks[0][0]
        idx = [k for rr, k in ks if rr is r]
        inside = r['status'] == 'verbatim' or all(any(a <= k < b for a, b in r['equal_spans']) for k in idx)
        src = (f"{DOCX} (sha256 {DOCX_SHA[:8]}) paragraph {r['docx_paragraph']}; letter quotation {r['quote']} segment "
               f"{r['segment']} ({r['status']})")
        if inside:
            out[t['key']] = R('reviewer quotation, verified', src, "reviewers' document", t['written'], t['written'],
                              'the token lies in a part of the quotation that matches the document verbatim')
        else:
            out[t['key']] = R('MISMATCH', src, "reviewers' document", None, None,
                              'the token lies in a part of the quotation that differs from the document')
            out[t['key']]['mismatch_vs'] = "reviewers' document"
    return out


# ======================================================================================================================
#  4. declarations, version 2
# ======================================================================================================================
def _sub_check(X, frag, written, sub_text, note=None):
    """a letter figure quoting the SUBMITTED version: the fragment is found once in the committed submitted source and
    carries `sub_text` as a token; the letter's value equals it."""
    try:
        ln = X.sub_line(frag)
    except KeyError as e:
        return R('MISMATCH', f'{SUBMITTED_MAIN} (sha256 {SUBMITTED_MAIN_SHA[:8]}): fragment {frag!r} not found',
                 'submitted manuscript (named record)', None, None, f'{e}. {note or ""}'.strip())
    hits, flen = X.sub.find_fragment(frag)
    inside = [t for t in X.sub_tokens if t['text'] == sub_text and X.sub.flat_pos(t['line'], t['col0']) is not None
              and hits[0] <= X.sub.flat_pos(t['line'], t['col0']) < hits[0] + flen]
    ok = bool(inside) and wv(written) == wv(sub_text)
    return R('match' if ok else 'MISMATCH', f'{SUBMITTED_MAIN} (sha256 {SUBMITTED_MAIN_SHA[:8]}) line {ln}: {frag!r}',
             'submitted manuscript (named record)', sub_text, sub_text, note)


def _figures(doc):
    import re as _re
    figs = []
    for nm, a, b in sorted(((nm, a, b) for nm, a, b in doc.envs if nm in ('figure', 'figure*')), key=lambda e: e[1]):
        body = doc.blank[a:b]
        figs.append({'n': len(figs) + 1, 'line': doc.blank.count('\n', 0, a) + 1, 'section': doc.section_at(a),
                     'graphics': _re.findall(r'includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', body),
                     'labels': _re.findall(r'\\label\{([^}]*)\}', body)})
    return figs


def _verdict_change_cell(X):
    vc = X.verdict_changes()
    if len(vc) != 1:
        raise KeyError(f'expected one T1 verdict change, found {len(vc)}')
    cl = X.claims[vc[0]['claim_id']]
    oc = cl['other_cell']
    nodes = X.cells_all[oc]['candidate_canonical']['nodes']
    return cl, oc, nodes, X.v6['cells'][oc]['m_flex_price_multiplier']


def letter_declarations_v2():
    L = LETTER
    out = []
    out.append(D('L03v2', L, '0.50\\,\\% gap was a stopping tolerance', [('0.50', 0)], lambda X, w: _sub_check(
        X, 'to a relative optimality gap of 0.50\\%', w[0], '0.50',
        f"submitted figure quoted; {SRP1_PARAMS} benders.tol_rel = {X.srp1p['benders']['tol_rel']} (= 0.50 %)")))
    out.append(D('L05v2', L, 'The 18.25\\,\\% operating-cost reduction of the submitted version', [('18.25', 0)],
                 lambda X, w: _sub_check(X, 'Relative to uncoordinated operation, coordinated operation with shared '
                                            'ESSs reduces operating costs by up to 18.25\\%', w[0], '18.25',
                                         'submitted abstract')))
    out.append(D('L08v2', L, 'submitted version (five representative years, twenty-five scenarios)',
                 [('five', 0), ('twenty-five', 0)], lambda X, w: [
                     _sub_check(X, 'is represented by five years (2025, 2028, 2031, 2034, and 2037)', w[0], 'five'),
                     _sub_check(X, 'resulting in 25 operating scenarios per representative day', w[1], '25')]))
    out.append(D('L22v2', L, "Section~2.2 (former 2.2.7 ``Benders' cuts'' removed", [('2.2', 0), ('2.2.7', 0)],
                 lambda X, w: [unch('cross-reference', 'revised-manuscript section number (map section B 2.2)'),
                               _sub_former(X, w[1])]))
    out.append(D('L37v2', L, 'R3.2 --- The 0.50\\,\\% gap is not an optimality certificate', [('0.50', 0)],
                 lambda X, w: _sub_check(X, 'to a relative optimality gap of 0.50\\%', w[0], '0.50',
                                         'submitted figure quoted (heading)')))
    out.append(D('L40v2', L, 'On Figure~15 of the submitted version', [('15', 0)], lambda X, w: _fig15(X)))
    out.append(D('L43v2', L, 'announcing 0.5\\,\\% and 2\\,\\% cases', [('0.5', 0), ('2', 0)], lambda X, w: [
        _sub_check(X, 'annual rates of 0.5\\% and 2.0\\% are additionally considered', w[0], '0.5',
                   'the letter says the SUBMITTED version announced these cases; the sentence is in the Overleaf '
                   f'main.tex (line {_main_line_or_none(X, "annual rates of 0.5")}, first-reply text), not in the '
                   'submitted source'),
        _sub_check(X, 'annual rates of 0.5\\% and 2.0\\% are additionally considered', w[1], '2.0',
                   'as the 0.5 token')]))
    out.append(D('L52v2', L, 'single machine, single thread, bitwise replays', [('single', 0), ('single', 1)],
                 lambda X, w: [ntc('no committed record states the machine count; stands on the author\'s attestation '
                                   '(Addendum 67; Addendum 68 Decision 5 "Single machine stands on the author\'s '
                                   'attestation")'),
                               w163ref(X, 'N45', w[1], 1, 'single-threaded; one thread', 'single-threaded')]))
    out.append(D('L21v2', L, 'every ADMM cycle solves one local problem per network block (three representative years '
                             '$\\times$ four representative days $\\times$ four network operators at the single-scenario '
                             'instance, 48 blocks, with the scenarios inside each block) and one per shared-ESS agent '
                             '(three, one per interface node, each spanning all years and days)',
                 [('one', 0), ('three', 0), ('four', 0), ('four', 1), ('single', 0), ('48', 0), ('one', 1),
                  ('three', 1), ('one', 2)], lambda X, w: _blocks_v2(X, w)))
    out.append(D('L32v2', L, 'It changes no sign in the 60 reported comparisons and two verdicts: a two-node plan '
                             'evaluated under the doubled flexibility price becomes determinate in the certificate\'s '
                             'favour, and the investment-year comparison is the other comparison that depends on the '
                             'convention', [('60', 0), ('two', 0), ('two', 1)], lambda X, w: _verdicts_v2(X, w)))
    out.append(D('L45v2', L, 'without ageing $-4.1$~k\\euro{} (within resolution, i.e.\\ break-even); cycling only '
                             '$-31.6$~k\\euro{}; cycling and calendar (the baseline) $-64.4$~k\\euro{}; the '
                             'datasheet-exact calibration $-45.2$~k\\euro{}, the mid-block evaluation point '
                             '$-65.9$~k\\euro{} and the unit-retention reading $-73.7$~k\\euro{}; all determinate except '
                             'the no-ageing case',
                 [('4.1', 0), ('31.6', 0), ('64.4', 0), ('45.2', 0), ('65.9', 0), ('73.7', 0)],
                 lambda X, w: [num(wi, X.ageing[arm]['value_minus_I'], f'T8 tables.ageing.rows[arm={arm}].value_minus_I '
                                   '(k EUR)', scale=1e-3,
                                   note=f"verdict {X.ageing[arm]['verdict']}; k {X.ageing[arm]['k']:.2f}; ext spec v3 "
                                        f"model_variant.arms.{arm} = {X.ext3['model_variant']['arms'][arm]}")
                               for wi, arm in zip(w, ('no_ageing', 'C2', 'C2_calfade', 'C4', 'C3_midblock',
                                                      'C3_unit'))]))
    out.append(D('L54', L, 'on the latter, run on a finer five-block horizon', [('five', 0)], lambda X, w: count_eq(
        w[0], len(X.i3x3['Years']), f"{jsrc(X, 'INST_3X3')} Years {X.i3x3['Years']} (count); {jsrc(X, 'SPEC_3X3')} "
        'configuration.derived_instance.changes_vs_source (Years {2025:5, 2030:5, 2035:5} -> five 3-year blocks)',
        note='"finer": 3-year blocks against SRP1\'s 5-year blocks')))
    out.append(D('L55', L, 'The framework figure (Figure~1 of the revised manuscript)', [('1', 0)],
                 lambda X, w: _fig1_revised(X, w[0])))
    out.append(D('L56', L, 'within each block (one representative year and one representative day) the shared storage '
                           'has a single charging, discharging and state-of-charge schedule, common to all market and '
                           'operation scenarios --- in the network models it is one scenario-free variable per hour '
                           'referenced by every scenario\'s power balance', [('one', 0), ('one', 1), ('single', 0),
                                                                             ('one', 2)],
                 lambda X, w: [unch('method statement', 'a network block = one (representative year, representative '
                                    'day) pair (48 = 3 x 4 x 4, L21v2); structure, no value'),
                               unch('method statement', 'as the first "one"'),
                               unch('method statement', 'one scenario-free schedule (model_construction_helpers.'
                                    'sess_na_scenario / sess_row_is_duplicate); a model statement audited by W171b'),
                               unch('method statement', 'as "a single schedule"')]))
    out.append(D('L57', L, 'and the state-of-health chain is driven by this one schedule', [('one', 0)],
                 lambda X, w: unch('not a figure', 'demonstrative ("this one schedule")')))
    return out


def _main_line_or_none(X, frag):
    try:
        return X.main_line(frag)
    except KeyError:
        return None


def _sub_former(X, written):
    hs = [h for h in X.sub.heads if h['level'] == 'subsubsection' and "Benders' Cuts" in h['title']]
    num_ = hs[0]['number'] if len(hs) == 1 else None
    return R('match' if num_ == written else 'MISMATCH', f'{SUBMITTED_MAIN} section structure (subsubsection "Benders\' '
             f'Cuts" at line {hs[0]["line"] if hs else None}; numbered by its position)', 'submitted manuscript (named record)',
             num_, num_)


def _fig15(X):
    figs = _figures(X.sub)
    f15 = figs[14] if len(figs) >= 15 else None
    s44 = [f['n'] for f in figs if f['section'].startswith('4 Results > 4.4')]
    return unch('cross-reference', f'figure number of the submitted version. In {SUBMITTED_MAIN} the figure '
                f'environments in order give Figure 15 = line {f15 and f15["line"]}, {f15 and f15["section"]}, '
                f'{f15 and f15["graphics"]}; Section 4.4 holds Figures {s44}. The reviewer\'s "Figure 15 in Section 4.4" '
                'does not match the source by environment order; the PDF the reviewer read is not in the repository')


def _fig1_revised(X, written):
    figs = _figures(X.main)
    f1 = figs[0] if figs else None
    ok = f1 is not None and any('framework' in g for g in f1['graphics']) and wv(written) == 1
    return R('match' if ok else 'MISMATCH', f'{MAIN} at the declared Overleaf commit: the first figure environment '
             f'(line {f1 and f1["line"]}, {f1 and f1["graphics"]}, label {f1 and f1["labels"]})',
             'revised manuscript (named record)', 1, '1',
             'numbering by environment order; the \\rchanges line of R1.5 still says "Figure~2 and the graphical abstract" '
             '-- see letter findings')


def _blocks_v2(X, w):
    ny, nd, na = len(X.years()), len(X.srp1['Days']), 1 + X.n_dns()
    sesd = X.c('shared_energy_storage_data')
    esso_ref = sesd.ref(r'results\[node_id\] = _optimize\(')
    src = f'{X.src("SRP1_JSON")}: Years {sorted(X.years())}, Days {sorted(X.srp1["Days"])}, TransmissionNetwork + ' \
          f'{X.n_dns()} DistributionNetworks'
    return [unch('method statement', 'one local problem per network block per cycle (algorithm statement)'),
            count_eq(w[1], ny, src), count_eq(w[2], nd, src),
            count_eq(w[3], na, src, note='"four network operators" = 1 TSO + 3 DSOs'),
            unch('not a figure', 'compound adjective (single-scenario)'),
            count_eq(w[5], ny * nd * na, src + ' (3 x 4 x 4)'),
            unch('method statement', 'one ESSO problem per storage agent per cycle'),
            count_eq(w[7], X.n_dns(), f'{X.src("SRP1_JSON")} DistributionNetworks (count) = the interface nodes; '
                     f'{esso_ref} (one ESSO solve per active distribution-network '
                     'node per cycle; each ESSO model spans every year and day)'),
            unch('method statement', 'one agent per interface node (structure)')]


def _verdicts_v2(X, w):
    vc = X.verdict_changes()
    yl = X.t['year_ladder']
    t4 = yl['gross_v6']['verdict'] != yl['net_v6']['verdict']
    cl, oc, nodes, m = _verdict_change_cell(X)
    n_nodes = sum(1 for v in nodes.values() if v != [0.0, 0.0])
    return [count_eq(w[0], len(X.t['claims']), 'T1 tables.claims (count)', 'frozen-table cell',
                     f'sign changes gross -> net: {len(X.sign_changes())} {X.sign_changes()}'),
            count_eq(w[1], len(vc) + int(t4), 'T1 tables.claims (gross_verdict != net verdict) + T4 tables.year_ladder '
                     '(gross_v6.verdict != net_v6.verdict)', 'frozen-table cells',
                     f"T1: {[c['claim_id'] for c in vc]}; T4 2035 - 2030: {t4}. Note: the T4 comparison is not one of "
                     "T1's 60 claims (the sentence counts the two verdicts across T1 and T4)"),
            R('match' if n_nodes == wv(w[2]) and m == 2.0 else 'MISMATCH',
              f'T1 claim {cl["claim_id"]}: other_cell {oc} candidate_canonical.nodes (non-zero nodes); '
              f'{jsrc(X, "V6_SPEC")} cells.{oc}.m_flex_price_multiplier', 'frozen-table cell + spec field',
              {'nodes': nodes, 'm_flex_price_multiplier': m}, str(n_nodes),
              '"a two-node plan evaluated under the doubled flexibility price": two non-zero nodes and multiplier 2.0')]


def cover_declarations_v2():
    return W164.cover_declarations()


# ---- main.tex section 2 -----------------------------------------------------------------------------------------------
def _settle_third(X, written):
    sc2 = X.c('settling_criterion_v2')
    ref = sc2.ref(r'p_hat = T\[-1\]\[0\] - T\[-3\]\[0\]')
    c = X.w163c.get('N11')
    return R('MISMATCH', f'{ref} (inherited by settling_criterion_v6); W163 N11 written {c and c.get("written_v4")!r}',
             'code (named record)', 'T[-1] - T[-3]', 'third-last to last',
             '"measured between the first and third turning points": the code measures P_hat between the third-last and '
             'the last turning point (the three most recent); the two agree only while exactly three turning points '
             'have occurred since k0')


def _poll_2n(X):
    pb, f2 = X.c('p515_s47_phase_b_record'), X.c('p515_s51_f2_phase_b')
    refs = '; '.join([pb.ref(r"^POLL_DESIGN = 'orthomads_n_plus_1_neg'"),
                      pb.ref(r'neg = \[-sum\(c\[i\] for c in cols\) for i in range\(n\)\]'),
                      f2.ref(r"'n_directions': PB\.N_VARS \+ 1")])
    return R('MISMATCH', refs, 'code (named record)', 'n + 1', 'n+1',
             'Algorithm 1 writes "the 2n OrthoMADS directions"; both planning searches as run (Phase B and the F2 Phase B) '
             'poll n + 1 directions (the n Householder columns and minus their sum; STEP4 section 3 allows "2n ... or '
             'the n+1 minimal positive basis")')


def _poll_completion(X):
    pb = X.c('p515_s47_phase_b_record')
    refs = '; '.join([pb.ref(r'completion = lattice\.completion\(tuple\(inc'), pb.ref(r'^UNIT_POLL_COMPLETION = True'),
                      pb.ref(r'^COMPLETION_CAP = 30')])
    return R('MISMATCH', refs, 'code (named record)', 'Delta^p = 1', 'unit poll',
             'Algorithm 1 adds the completion set "If |P| < n + 1"; the code adds it at every poll of unit size '
             '(UNIT_POLL_COMPLETION and delta == DELTA_MIN), refusing (stop for review) above 30 points')


def _poll_rule(X, which, written):
    pb, s4 = X.c('p515_s47_phase_b_record'), X.srcs['STEP4']
    if which == 'double':
        ok = wv(written) == 2
        src = '; '.join([pb.ref(r"'next_poll_size': delta \* 2"), s4.ref(r'poll size doubles on a successful poll')])
    elif which == 'halve':
        ok = wv(written) == 2
        src = '; '.join([pb.ref(r'delta = max\(DELTA_MIN, delta // 2\)'), s4.ref(r'halves on a failed one')])
    elif which == 'unit':
        ok = wv(written) == pb.value(r'^DELTA_MIN = (\d+)', cast=int) == 1
        src = '; '.join([pb.ref(r'^DELTA_MIN = 1'), s4.ref(r'Termination is \*\*mesh-local optimality on the lattice\*\*')])
    elif which == 'neighbour':
        ok = wv(written) == 1 and 'within l_inf distance 1 of the' in pb.text
        src = pb.ref(r'within l_inf distance 1 of the incumbent in the 7 variables') + ' (COMPLETION_RULE)'
    else:
        raise KeyError(which)
    return R('match' if ok else 'MISMATCH', src, 'code (named record)', written, written)


def _candidates_of(X, cl):
    inst = cl['instance']
    out = []
    if isinstance(inst, dict):
        for k, v in inst.items():
            cc = v.get('candidate_canonical') if isinstance(v, dict) else None
            out.append((k, cc))
    else:
        for k in inst:
            c = X.cells_all.get(k)
            out.append((k, c.get('candidate_canonical') if c else None))
    return out


def _single_cohort(X):
    seen = {}
    for cl in X.claims.values():
        for k, cc in _candidates_of(X, cl):
            seen[k] = cc
    bad = sorted(k for k, cc in seen.items() if not (isinstance(cc, dict) and isinstance(cc.get('investment_year'), int)))
    return R('match' if seen and not bad else 'MISMATCH',
             'T1 tables.claims[*].instance -> candidate_canonical.investment_year (instance dict, or tables.cells / '
             'cells_appended_w160 for a list instance)', 'frozen-table cells',
             {'candidates_checked': len(seen), 'without_a_single_investment_year': bad}, 'single',
             'every candidate of a reported comparison carries one investment year, i.e. one cohort per node')


def _code_const(X, name, rx, written, value):
    s = X.c(name)
    s.cite(rx)
    return R('match' if wv(written) == value else 'MISMATCH', s.ref(rx), 'code (named record)', value, str(value))


def _prod_exit(X, written):
    mc, rc = X.srp1p['admm']['minimum_consecutive_converged_cycles'], X.s3x3['required_consecutive_cycles']
    refs = '; '.join([jline(X, 'SRP1_PARAMS', r'"minimum_consecutive_converged_cycles"') + f' = {mc}',
                      jsrc(X, 'SPEC_3X3') + f' required_consecutive_cycles = {rc}',
                      X.c('shared_resources_planning').ref(r'convergence = \(consecutive_converged_cycles >= ')])
    return R('match' if wv(written) == mc == rc else 'MISMATCH', refs, 'case file + spec field', mc, str(mc))


def sec2_declarations(fname, variant):
    """the numbers of revised section 2 (main.tex) or of the draft (variant 'draft': the older wording, W and V)."""
    M = fname
    dr = variant == 'draft'
    p = f'{"d" if dr else "S"}2'
    win = '$W = \\max\\{20, \\lceil 1.1\\,\\hat{P} \\rceil\\}$' if dr else \
        '$n_{\\mathrm{w}} = \\max\\{20, \\lceil 1.1\\,\\hat{P} \\rceil\\}$'
    tau = '$\\tau = \\delta_R V / 4$ with $\\delta_R = 0.07$' if dr else \
        '$\\tau = \\delta_R \\mathcal{V} / 4$ with $\\delta_R = 0.07$'
    vv = '($V = 259{,}375.33$~\\euro{}, hence $\\tau = 4{,}539.07$~\\euro{}' if dr else \
        '($\\mathcal{V} = 259{,}375.33$~\\euro{}, hence $\\tau = 4{,}539.07$~\\euro{}'
    out = [
        D(f'{p}01', M, 'The values used in the case study are $\\Delta^S = 0.25$~MVA and $\\Delta^E = 0.5$~MWh with '
                       '$\\phi^{\\text{Min}} = 2$~h and $\\phi^{\\text{Max}} = 4$~h', [('0.25', 0), ('0.5', 0), ('2', 0),
                                                                                     ('4', 0)],
          lambda X, w: W164._lattice_checks(X, w)),
        D(f'{p}02', M, '$\\varepsilon_{\\text{abs}} = 10^{-5}$ and $\\varepsilon_{\\text{rel}} = 10^{-4}$',
          [('10^{-5}', 0), ('10^{-4}', 0)], lambda X, w: [w163ref(X, 'N2', w[0], 1e-5, '10⁻⁵', None),
                                                         w163ref(X, 'N3', w[1], 1e-4, '10⁻⁴', None)]),
        D(f'{p}03', M, 'moved by a further 15--21~k\\euro{} after their residual-based certification',
          [('15', 0), ('21', 0)], lambda X, w: [w163ref(X, 'N4', w[0], 15, '15', None),
                                                w163ref(X, 'N5', w[1], 21, '21', None)]),
        D(f'{p}04', M, 'the objective has shown at least three turning points since $k_0$', [('three', 0)],
          lambda X, w: w163ref(X, 'N7', w[0], 3, 'three', None)),
        D(f'{p}05', M, '(measured between the first and third turning points)', [('third', 0)],
          lambda X, w: _settle_third(X, w[0])),
        D(f'{p}06', M, 'swings smaller than $\\tau/10$ are', [('10', 0)],
          lambda X, w: w163ref(X, 'N8', w[0], 10, '10', None)),
        D(f'{p}07', M, win, [('20', 0), ('1.1', 0)], lambda X, w: [w163ref(X, 'N9', w[0], 20, '20', None),
                                                                  w163ref(X, 'N10', w[1], 1.1, '1.1', None)]),
        D(f'{p}08', M, 'is at most $\\tau/2$ in absolute value', [('2', 0)],
          lambda X, w: w163ref(X, 'N12', w[0], 2, '2', None)),
        D(f'{p}09', M, 'with all four termination metrics within ten times the tight-tail tolerances',
          [('four', 0), ('ten', 0)], lambda X, w: [w163ref(X, 'N13', w[0], 4, 'four', None),
                                                   w163ref(X, 'N14', w[1], 10, '10', None)]),
        D(f'{p}10', M, 'over a window of $2P_{\\max}$ cycles with $P_{\\max}$ the longest period measured',
          [('2P_', 0)], lambda X, w: w163ref(X, 'N15', None, None, '2 P_max = 60', None,
                                             'the factor 2 of 2 P_max: v6 spec stop_rule.W.monotone L = 2 * P_MAX')),
        D(f'{p}11', M, '$|\\text{last step}| \\times 2P_{\\max} \\le \\tau$', [('2P_', 0)],
          lambda X, w: w163ref(X, 'N17', None, None, '60', None, 'abs(dQ_k) * L_MONO <= TAU with L_MONO = 2 P_max')),
        D(f'{p}12', M, tau, [('4', 0), ('0.07', 0)], lambda X, w: [w163ref(X, 'N18', w[0], 4, '4', None),
                                                                   w163ref(X, 'N19', w[1], 0.07, '0.07', None)]),
        D(f'{p}13', M, vv, [('259{,}375.33', 0), ('4{,}539.07', 0)],
          lambda X, w: [w163ref(X, 'N20', w[0], 259375.33, '259,375.33', None),
                        w163ref(X, 'C1', w[1], 4539.07, '4,539.07', None)]),
        D(f'{p}14', M, 'so that a ratio of two values --- four evaluations, each in error by at most $\\tau$',
          [('two', 0), ('four', 0)], lambda X, w: [w163ref(X, 'N22', w[0], 2, 'two', None),
                                                   w163ref(X, 'N21', w[1], 4, 'Four', None)]),
        D(f'{p}15', M, 'ten of the certificates in use stop within 5\\,\\% of $\\tau$', [('ten', 0), ('5', 0)],
          lambda X, w: [w163ref(X, 'C4', w[0], 10, '10', None), w163ref(X, 'N24', w[1], 5, '5', None)]),
        D(f'{p}16', M, 'certification moved by at most $0.9\\,\\tau$', [('0.9', 0)],
          lambda X, w: w163ref(X, 'N25', w[0], 0.9, '0.9', None)),
        D(f'{p}17', M, 'A difference between two certified evaluations is called determinate when it exceeds '
                       '$\\max\\{3\\,b, 2\\tau\\}$, where $b$ is the larger of the two certification bands',
          [('two', 0), ('3', 0), ('2', 0), ('two', 1)],
          lambda X, w: [w163ref(X, 'N73', w[0], 2, 'two', None), w163ref(X, 'N28', w[1], 3, '3', None),
                        w163ref(X, 'N29', w[2], 2, '2', None), w163ref(X, 'N73', w[3], 2, 'two', None)]),
        D(f'{p}18', M, 'only if it exceeds three times the larger of that evaluation\'s consensus gap and settling slack',
          [('three', 0)], lambda X, w: w163ref(X, 'N26', w[0], 3, 'three', None)),
        D(f'{p}19', M, 'Generate the $2n$ OrthoMADS directions', [('2n', 0)], lambda X, w: _poll_2n(X)),
        D(f'{p}20', M, 'the admissible lattice points within one unit step of', [('one', 0)],
          lambda X, w: _poll_rule(X, 'neighbour', w[0])),
        D(f'{p}21', M, '\\frac{D_d}{365}', [('365', 0)], lambda X, w: _code_const(
            X, 'shared_energy_storage_data', r'avg_ch_dch \+= \(num_days / 365\.00\)', w[0], 365)),
        D(f'{p}22', M, '\\frac{365 \\, Y_y \\, \\bar{E}^{\\text{Cell}}_{e,y^\\text{Inv},y}}', [('365', 0)],
          lambda X, w: _code_const(X, 'shared_energy_storage_data', r'== 365\.00 \* num_years \* model\.es_avg_ch_dch_per_unit',
                                   w[0], 365)),
        D(f'{p}23', M, '{2 \\, k_e \\, E^{\\text{Rated,Unit}}_{e,y^\\text{Inv},y}}', [('2', 0)], lambda X, w: _code_const(
            X, 'shared_energy_storage_data', r'model\.es_D_per_unit\[y_inv, y\] \* \(2 \* shared_energy_storage\.cl_eff',
            w[0], 2)),
        D(f'{p}24', M, '$2 E^{\\text{Rated,Unit}}$ is the', [('2', 0)], lambda X, w: _code_const(
            X, 'shared_energy_storage_data', r'model\.es_D_per_unit\[y_inv, y\] \* \(2 \* shared_energy_storage\.cl_eff',
            w[0], 2)),
    ]
    if not dr:
        out += [
            D('S225', M, '\\If{$|\\mathcal{P}| < n + 1$}', [('1', 0)], lambda X, w: _poll_completion(X)),
            D('S226', M, '$\\Delta^p \\gets 2\\Delta^p$', [('2', 0)], lambda X, w: _poll_rule(X, 'double', w[0])),
            D('S227', M, '\\lIf{$\\Delta^p = 1$}', [('1', 0)], lambda X, w: _poll_rule(X, 'unit', w[0])),
            D('S228', M, '$\\Delta^p \\gets \\Delta^p / 2$', [('2', 0)], lambda X, w: _poll_rule(X, 'halve', w[0])),
            D('S229', M, 'every plan reported in this paper has a single cohort per node', [('single', 0)],
              lambda X, w: _single_cohort(X)),
            D('S230', M, 'evaluated under the production exit --- ten consecutive cycles passing the residual test',
              [('ten', 0)], lambda X, w: _prod_exit(X, w[0])),
        ]
    else:
        out += [
            D('d225', M, '\\If{$|\\mathcal{P}| < n + 1$}', [('1', 0)], lambda X, w: _poll_completion(X)),
            D('d226', M, '$\\Delta^p \\gets 2\\Delta^p$', [('2', 0)], lambda X, w: _poll_rule(X, 'double', w[0])),
            D('d227', M, '\\lIf{$\\Delta^p = 1$}', [('1', 0)], lambda X, w: _poll_rule(X, 'unit', w[0])),
            D('d228', M, '$\\Delta^p \\gets \\Delta^p / 2$', [('2', 0)], lambda X, w: _poll_rule(X, 'halve', w[0])),
        ]
    return out


def draft_comment_declarations():
    M = DRAFT
    return [
        D('dC01', M, "% section2_expert_draft.tex — expert's draft of the formal pieces of Section 2 (2026-10-07)",
          [('2', 0), ('2026-10-07', 0)], lambda X, w: [
              unch('comment (draft instruction)', 'section number in a title comment'),
              R('match' if '2026-10-07' in X.corr_text.split('\n')[0] else 'MISMATCH', f'{CORR_MD} line 1 header date '
                '(the corrections the draft precedes are of the same date)', 'named record', '2026-10-07', '2026-10-07',
                'comment scope')]),
        D('dC02', M, 'Numbers are the frozen ones (tables v1, 590088fe)', [('v1', 0), ('590088fe', 0)], lambda X, w: [
            unch('identifier', 'version label of the frozen tables'),
            R('match' if W164.FZ_SHA.startswith(w[1]) else 'MISMATCH', f'sha256 of {W164.FZ_REL}', 'identifier',
              W164.FZ_SHA[:8], W164.FZ_SHA[:8], 'comment scope')]),
        D('dC03', M, '% Notation: Delta^S = 0.25 MVA, Delta^E = 0.5 MWh (lattice units)', [('0.25', 0), ('0.5', 0)],
          lambda X, w: W164._lattice_checks(X, [w[0], w[1], '2', '4'])[:2]),
        D('dC04', M, '(double on a determinate success, halve on failure, floor 1) and the completion-set',
          [('1', 0)], lambda X, w: _poll_rule(X, 'unit', w[0])),
        D('dC05', M, '%      cap (30) as run;', [('30', 0)], lambda X, w: _code_const(
            X, 'p515_s47_phase_b_record', r'^COMPLETION_CAP = 30', w[0], 30)),
        D('dC06', M, '% [CONFIRM] (a) the row D*(2*cl_eff*E_rated) == 365*num_years*avg_ch_dch and',
          [('2', 0), ('365', 0)], lambda X, w: [
              _code_const(X, 'shared_energy_storage_data', r'model\.es_D_per_unit\[y_inv, y\] \* \(2 \* shared_energy_storage\.cl_eff', w[0], 2),
              _code_const(X, 'shared_energy_storage_data', r'== 365\.00 \* num_years \* model\.es_avg_ch_dch_per_unit', w[1], 365)]),
        D('dC07', M, '% representative year (equal 5-year blocks); (d) Section 3.4 values: C2 (10,000, 0.80, 0.80) → k = '
                     '35,851;', [('5', 0), ('3.4', 0), ('C2', 0), ('10,000', 0), ('0.80', 0), ('0.80', 1), ('35,851', 0)],
          lambda X, w: [count_eq(w[0], 5 if set(X.years().values()) == {5} else -1, f'{X.src("SRP1_JSON")} Years {X.years()}'),
                        unch('cross-reference', 'section number in a draft instruction'),
                        unch('enumerator', 'ageing calibration name'),
                        num(w[3], X.ess['ageing']['calibration']['cycles_n'], f'{jsrc(X, "ESS_PARAMS")} ageing.calibration.cycles_n', 'named record'),
                        num(w[4], X.ess['ageing']['calibration']['reference_dod_d'], f'{jsrc(X, "ESS_PARAMS")} ageing.calibration.reference_dod_d', 'named record'),
                        num(w[5], X.ess['ageing']['calibration']['eol_retention_r'], f'{jsrc(X, "ESS_PARAMS")} ageing.calibration.eol_retention_r', 'named record'),
                        num(w[6], X.ageing['C2']['k'], 'T8 tables.ageing.rows[arm=C2].k', note='k = 35,851.36')]),
        D('dC08', M, '% C4 (8,000, 1.0, 0.70) → k = 22,430; phi_cal = 0.985; SoH_min = 0.70 (0.50 as a sensitivity row).',
          [('C4', 0), ('8,000', 0), ('1.0', 0), ('0.70', 0), ('22,430', 0), ('0.985', 0), ('0.70', 1), ('0.50', 0)],
          lambda X, w: _c4_comment(X, w)),
        D('dC09', M, '% [CONFIRM] (a) alpha = 0.50 in the 3x3 campaign;', [('0.50', 0), ('3x3', 0)], lambda X, w: [
            num(w[0], X.s3x3['candidates'][0]['interface_deviation_premium']['alpha'],
                f'{jsrc(X, "SPEC_3X3")} candidates[*].interface_deviation_premium.alpha', 'spec field'),
            unch('identifier', 'instance label (3 x 3)')]),
        D('dC10', M, 'the post-solve detector threshold (1e-6 * s_max)', [('1e', 0), ('6', 0)], lambda X, w: [
            _code_const(X, 'shared_energy_storage_data', r'threshold = 1e-6 \* s_max', '1', 1),
            unch('notation', 'exponent of 1e-6 (checked with the 1e token against the code line)')]),
    ]


def _c4_comment(X, w):
    arm = X.ext3['model_variant']['arms']['C4']
    n_, d_ = X.ess['ageing']['calibration']['cycles_n'], X.ess['ageing']['calibration']['reference_dod_d']
    nd = wv(w[1]) * wv(w[2])
    k_ext = X.ext3['model_variant']['k_closed_form_by_arm']['C4']
    return [unch('enumerator', 'ageing calibration name'),
            R('match' if abs(nd - n_ * d_) < 1e-9 else 'MISMATCH', f'{jsrc(X, "EXT_SPEC_V3")} model_variant: C4 runs with '
              f'the file\'s (N, D) = ({n_}, {d_}) and eol_retention_r {arm["eol_retention_r"]}; N x D = {n_ * d_}',
              'spec field', n_ * d_, f'{nd:g}', 'the datasheet pair (8,000 cycles at DoD 1.0) has the run\'s N x D, so '
                                                 'the same k; the run itself used (10,000, 0.80)'),
            R('match' if abs(nd - n_ * d_) < 1e-9 else 'MISMATCH', 'as the 8,000 token', 'spec field', None, None, None),
            num(w[3], arm['eol_retention_r'], f'{jsrc(X, "EXT_SPEC_V3")} model_variant.arms.C4.eol_retention_r', 'spec field'),
            num(w[4], k_ext, f'{jsrc(X, "EXT_SPEC_V3")} model_variant.k_closed_form_by_arm.C4 (= T8 C4 k '
                f'{X.ageing["C4"]["k"]:.3f})', 'spec field', note='8,000 x 1.0 / -ln 0.70 = 22,429.39'),
            num(w[5], X.ess['ageing']['calendar_retention_per_year'], f'{jsrc(X, "ESS_PARAMS")} ageing.calendar_retention_per_year', 'named record'),
            num(w[6], X.ess['ageing']['minimum_soh'], f'{jsrc(X, "ESS_PARAMS")} ageing.minimum_soh', 'named record'),
            w163ref(X, 'N41', w[7], 0.50, '0.50', None, 'the A64 soh_min 0.50 row (T10)')]


def all_declarations_v2():
    return (letter_declarations_v2() + cover_declarations_v2() + W164.main_declarations() +
            sec2_declarations(MAIN, 'main') + sec2_declarations(DRAFT, 'draft') + draft_comment_declarations())


# v1 declarations re-pointed although their fragment is still found (reason recorded)
SUPERSEDED = {
    'L03': ('L03v2', 'a figure quoted from the submitted version: checked against the committed submitted source'),
    'L05': ('L05v2', 'as L03; the fragment of the submitted abstract differs from the Overleaf abstract'),
    'L08': ('L08v2', 'as L03'),
    'L22': ('L22v2', 'the former section number is read from the submitted source'),
    'L37': ('L37v2', 'as L03'),
    'L40': ('L40v2', 'the figure order of the submitted source is now available'),
    'L43': ('L43v2', 'as L03'),
    'L52': ('L52v2', 'Addendum 68 Decision 5: "single machine" stands on the author\'s attestation'),
}
# v1 declarations whose fragment is gone, and the v2 declaration that replaced the sentence
REPLACED_BY = {
    'L21': ('L21v2', 'B.3: "four agents" -> "four network operators" and "and one per shared-ESS agent (three, one per '
                     'interface node, ...)" added'),
    'L32': ('L32v2', 'B.4: "and one verdict (a Phase~B neighbour ...)" -> "and two verdicts: a two-node plan evaluated '
                     'under the doubled flexibility price ..."'),
    'L33': ('L32v2', 'B.4: "The investment-year comparison is the one result that depends on the convention" -> "and the '
                     'investment-year comparison is the other comparison that depends on the convention" (no number '
                     'token left; the count is checked by L32v2)'),
    'L45': ('L45v2', 'B.5: "harsher calibrations -73.7 and -45.2 k EUR; all determinate except the first" -> the five aged '
                     'arms named in order, with -65.9 added'),
}


# ======================================================================================================================
#  5. rules (tokens no declaration claims)
# ======================================================================================================================
XREF_RX = re.compile(r'(?:Subsections?|Sections?|Appendix|Algorithm|Figures?|Fig\.|Tables?|Eqs?\.|Equations?|Reviewers?|'
                     r'§)(?:~|\s)*[\[(]?\d+(?:\.\d+)*[\])]?'
                     r'(?:\s*(?:,|and|--|-|–|or)\s*[\[(]?\d+(?:\.\d+)*[\])]?)*')


def _in_xref(doc, t):
    ln = doc.lines[t['line'] - 1]
    for m in XREF_RX.finditer(ln):
        if m.start() <= t['col0'] < m.end():
            return True
    return False


def _xref_note(X, doc, t):
    if doc.name != MAIN:
        return ''
    ln = doc.lines[t['line'] - 1]
    pre = ln[max(0, t['col0'] - 14):t['col0']]
    if re.search(r'Section~?\s*$', pre) and t['type'] in ('num', 'dotted'):
        titles = {h['number']: h['title'] for h in X.main.heads if h['number']}
        return (f'; main.tex at this commit: section {t["text"]} ' +
                (f'exists ("{titles[t["text"]]}")' if t['text'] in titles else 'DOES NOT EXIST'))
    return ''


def rule_assign_v2(X, doc, role, t, map_xref_numbers, qstat):
    typ, ln, txt = t['type'], t['line'], t['text']
    line = doc.lines[ln - 1]
    in_sec2 = role == 'main' and SEC2[0] <= ln <= SEC2[1]
    # ---- comments --------------------------------------------------------------------------------------------------
    if t['scope'] == 'comment':
        if role == 'main':
            sub_ = 'template boilerplate' if ln <= 60 or line.lstrip().startswith('%%') else \
                '[CONFIRM] / instruction comment in revised section 2' if in_sec2 else 'commented-out text'
            return unch('comment (not typeset)', f'{sub_}; comments are not manuscript text')
        if role == 'draft':
            if typ in ('num', 'dotted') and re.search(r'(?:\bl\.|\blines?)\s*[\d–\-, and]*$', line[:t['col0']]):
                return unch('comment (draft instruction)', 'line number of the Overleaf clone at 6cb4492 the draft refers to')
            if re.search(r'SIAM J\. Optim\.|\b20\d\d;\s*§|Found\. Trends|\d\(\d\),', line):
                return unch('comment (draft instruction)', 'bibliographic data of a reference (volume, issue, pages, year)')
            if typ == 'dotted' or (typ == 'num' and '.' in txt) or (typ == 'num' and re.search(
                    r'(?:Subsection|Section|section|§)\s*$', line[:t['col0']])):
                return unch('comment (draft instruction)', 'section number in a draft instruction (of the manuscript or '
                                                           'of a cited reference)')
            if typ == 'label' or typ == 'hash' or typ == 'alnum':
                return unch('comment (draft instruction)', f'{typ} in a draft instruction')
            if typ == 'word':
                return unch('comment (draft instruction)', 'number word in a draft instruction (structure)')
            return unch('comment (draft instruction)', 'other integer in a draft instruction (algorithm / equation / '
                                                       'addendum number, part counter or formula index)')
        return None
    lines_ok = role == 'main' and X.main_sha == MAIN_LINE_RULES_SHA
    if lines_ok and not in_sec2:
        for (a, b), key, kind, why in MAIN_REGIONS:
            if a <= ln <= b:
                return W164._map_sub(X, key, kind, why)
    # ---- letter: reviewer quotations --------------------------------------------------------------------------------
    if role == 'letter' and t['macro'] == 'rcomment':
        return qstat.get(t['key'])
    # ---- identifiers ------------------------------------------------------------------------------------------------
    if typ == 'doi':
        return unch('reference identifier', 'DOI (reference-list identifier)')
    if typ == 'label':
        if re.fullmatch(r'R\d\.\d+', txt):
            return unch('enumerator', 'reviewer item label')
        if re.fullmatch(r'T\d+', txt):
            return unch('enumerator', 'table identifier of the frozen tables (placeholder [T..] in the letter)')
        if re.fullmatch(r'C\d', txt):
            return unch('enumerator', 'ageing calibration name (T8 arm labels C2 / C2_calfade)')
        return unch('identifier', 'label containing digits')
    if typ in ('hash', 'date', 'alnum'):
        if role == 'main':
            return unch('identifier' if not in_sec2 else 'notation',
                        f'{typ} in the submitted text' if not in_sec2 else f'{typ} in revised section 2 (notation)')
        if role == 'draft':
            return unch('notation', f'{typ} in the draft (notation)')
        return None
    # ---- cross-references written in prose --------------------------------------------------------------------------
    if typ in ('num', 'dotted') and _in_xref(doc, t):
        found = txt in map_xref_numbers
        return unch('cross-reference', 'section / table / figure / equation / algorithm / reviewer number written in '
                    'prose' + (f'; the revision map names section {txt}' if found and role == 'letter' else '') +
                    _xref_note(X, doc, t))
    if role == 'letter' and typ == 'num' and re.search(r'\\section\*\{Reviewer ' + re.escape(txt) + r'\}', line):
        return unch('enumerator', 'reviewer number (section heading)')
    if role == 'letter' and typ == 'num' and re.search(r'revision ' + re.escape(txt) + r'\}', line):
        return unch('enumerator', 'revision number (title block)')
    # ---- number words -----------------------------------------------------------------------------------------------
    if typ == 'word' and W164._compound(doc, t):
        return unch('not a figure', f'compound adjective ({txt}{line[t["end0"]:t["end0"] + 12].split()[0]})')
    if typ == 'word' and txt.lower() == 'single':
        return unch('not a figure', 'article sense ("a single ...")')
    # ---- revised section 2 of main.tex and the draft -------------------------------------------------------------------
    if in_sec2 or role == 'draft':
        where = 'revised section 2' if in_sec2 else 'the draft'
        if typ == 'word':
            return unch('method statement', f'number word in {where} describing structure (no value to check; the '
                        'statement is audited against the code by W171b)')
        if 'algorithm' in t['envs'] or t['in_math']:
            return unch('notation', f'formula constant in {where} (an index offset, exponent, bound or coefficient of the '
                        'printed formula; the equations and Algorithm 1 are audited against the code by W171b)')
        return None
    if role != 'main':
        return None
    # ---- main.tex outside section 2 (the W164 rules, lines re-pinned) ----------------------------------------------
    if typ == 'num' and re.fullmatch(r'(19|20)\d\d', txt):
        if txt in X.years():
            return R('match', f'{X.src("SRP1_JSON")} Years {sorted(X.years())}', 'named record', sorted(X.years()), txt,
                     'a representative year of the revised instance')
        return W164._map_sub(X, 'years', 'input replaced', 'non-representative year of the submitted horizon')
    if typ == 'num' and re.search(r'IEEE\s*$', line[:t['col0']]) and line[t['end0']:t['end0'] + 4] == '-bus':
        return W164._bus(X, 'case9', t['written']) if txt == '9' else \
            W164._bus(X, 'case33', t['written']) if txt == '33' else None
    if typ == 'num' and re.search(r'[Nn]odes?(?:~|\s)*(?:\d+\s*,\s*(?:and\s*)?|\d+\s*,?\s*and\s*)*$', line[:t['col0']]) \
            and 'tabular' not in ' '.join(t['envs']):
        ids = sorted(d['connection_node_id'] for d in X.srp1['DistributionNetworks'])
        return R('match' if int(float(txt)) in ids else 'MISMATCH', f'{X.src("SRP1_JSON")} '
                 f'DistributionNetworks[*].connection_node_id {ids}', 'named record', ids, txt,
                 'TN interface node of a distribution network (an identifier)')
    if 'postcode=' in line:
        return unch('address', 'postcode of an affiliation')
    if not lines_ok:
        return None
    if 'algorithm' in t['envs']:
        return unch('notation', f'constant in an algorithm (audited: {W164.MAP_REL} line {X.map_line(W164.MAP_CITES["appA"])})')
    if t['in_math']:
        extra = f'; passage cited by {W164.MAP_REL} line {X.map_line(W164.MAP_CITES["appA"])}' \
            if APP_A[0] <= ln <= APP_A[1] else ''
        return unch('notation', 'formula constant (equations are audited against the code, W166 / W171b)' + extra)
    if any(e.startswith('tabular') for e in t['envs']):
        if NETWORK_TABLES[0] <= ln <= NETWORK_TABLES[1]:
            return unch('network/data parameter', 'case-data table (RES capacity / generator and node IDs), outside the '
                        f'frozen tables; to be regenerated for 2025/2030/2035 ({W164.MAP_REL} line '
                        f'{X.map_line(W164.MAP_CITES["years"])})')
        if APP_BCD[0] <= ln <= APP_BCD[1]:
            return unch('network/data parameter', 'IEEE 33-bus / market data table, outside the frozen tables '
                        f'({W164.MAP_REL} line {X.map_line(W164.MAP_CITES["appBCD"])}: keep)')
    if typ == 'word':
        if txt.lower() == 'third':
            return unch('enumerator', 'ordinal word')
        return unch('descriptive count', 'number word in unrevised submitted prose (counts of lists, layers, agents, '
                    'reference cases); wording checks are not required this round')
    return None


# ======================================================================================================================
#  6. findings
# ======================================================================================================================
def letter_findings_v2(X, docs, qrows, dx):
    letter = docs[LETTER]
    sub_low = X.sub.raw.lower()
    sub_has_cal = {k: k in sub_low for k in ('calendar ageing', 'calendar aging', 'annual rates of 0.5')}
    stray = [i + 1 for i, ln in enumerate(letter.lines) if '(Section~4.5)".' in ln]
    b6 = '(single-scenario instance; the multi-scenario instance uses five three-year blocks)'
    b6_present = re.sub(r'\s+', ' ', b6) in letter.flat
    placeholder = 'Placeholder figure' in dx['text']
    vc = X.verdict_changes()
    floor = {r['arm']: r['floor_year_070'] for r in X.t['ageing']['rows']}
    figs_sub = _figures(X.sub)
    figs_main = _figures(X.main)
    t9 = {r['cell']: r for r in X.t['dead_zone']['cells']}
    unc = sorted(k for k, r in X.cells_all.items() if r.get('certification_stats_included') and r.get('status') == 'uncertified')
    dz = sorted(k for k in unc if k in t9 and ('gap-refused' in t9[k]['entry'] or 'signature' in t9[k]['entry']))

    def lno(frag):
        hits, _n = letter.find_fragment(frag)
        return letter.flat_idx[hits[0]][0] if len(hits) == 1 else None
    further = letter.raw.split('\\section*{Further changes', 1)[1] if '\\section*{Further changes' in letter.raw else ''
    searched_terms = ['cost file', 'cost-file', 'cost input', 'investment-cost', 'divided', '\\div', '÷', '/5', '4~h']
    cost_terms = [s for s in searched_terms if s in further]
    F = [
        {'id': 'G1', 'lines': str(lno('(within the horizon it never does)')), 'kind': 'claim against a table',
         'finding': f'"the year in which the end-of-life floor binds (within the horizon it never does)": T8 "floor binds '
                    f'(0.70)" (tables.ageing.rows[*].floor_year_070) = {floor}. The floor binds in 2035 for '
                    f'{sorted(a for a, y in floor.items() if y)} -- the baseline C2_calfade among them -- and never for '
                    f'{sorted(a for a, y in floor.items() if not y)}. The sentence (STEP6_ROUND1_CORRECTIONS B.5; '
                    'Addendum 68 Decision 5 "the floor never binds within the horizon") contradicts T8 for three of the '
                    'five aged arms. No number token: reported here only.'},
        {'id': 'G2', 'lines': str(lno('The sentence of the submitted version announcing')), 'kind': 'claim against the submitted source',
         'finding': '"The sentence of the submitted version announcing 0.5 % and 2 % cases that had not been run was '
                    f'removed": the sentence is not in the submitted source ({SUBMITTED_MAIN}, sha256 '
                    f'{SUBMITTED_MAIN_SHA[:8]}; substrings found in it: {sub_has_cal}); it is first-reply text of the '
                    'Overleaf main.tex, still present at this commit, '
                    f'line {_main_line_or_none(X, "annual rates of 0.5")} (Addendum 68 Decision 6: removed in the section '
                    '3.4 rewrite, not yet made). Tokens 0.5 / 2: MISMATCH (L43v2).'},
        {'id': 'G3', 'lines': f"{lno('The framework figure (Figure~1 of the revised manuscript)')}, 100",
         'kind': 'figure numbering',
         'finding': f'R1.5 answers for the framework figure ("Figure~1 of the revised manuscript": main.tex first figure '
                    f'environment, line {figs_main[0]["line"] if figs_main else None}, {figs_main[0]["graphics"] if figs_main else None} '
                    f'-- match). (a) The \\rchanges line still says "Figure~2 and the graphical abstract". (b) In the '
                    f'submitted source the second figure environment -- the reviewer\'s "Figure 2" by environment order -- '
                    f'is {figs_sub[1]["graphics"] if len(figs_sub) > 1 else None} (line {figs_sub[1]["line"] if len(figs_sub) > 1 else None}, '
                    f'{figs_sub[1]["section"] if len(figs_sub) > 1 else None}); the framework figure is Figure 1 '
                    f'({figs_sub[0]["graphics"] if figs_sub else None}). The reviewers\' document carries the author\'s '
                    f'note "Placeholder figure" after the comment: {placeholder}. The PDF numbering is not re-derived '
                    'here.'},
        {'id': 'G4', 'lines': str(lno('On Figure~15 of the submitted version')), 'kind': 'figure numbering',
         'finding': f'"On Figure 15 of the submitted version: the figure and its discussion belonged to the superseded '
                    f'results and were removed": by environment order the submitted Figure 15 is '
                    f'{figs_sub[14]["graphics"] if len(figs_sub) >= 15 else None} ({figs_sub[14]["section"] if len(figs_sub) >= 15 else None}, '
                    f'line {figs_sub[14]["line"] if len(figs_sub) >= 15 else None}), a data figure kept in the revision; '
                    f'Section 4.4 holds Figures {[f["n"] for f in figs_sub if f["section"].startswith("4 Results > 4.4")]}. '
                    'The reviewer\'s "Figure 15 in Section 4.4" matches no figure of the source; the letter\'s claim '
                    'cannot be confirmed from it.'},
        {'id': 'G5', 'lines': str(stray), 'kind': 'typographical',
         'finding': f'R1.2: a stray double quote follows "(Section~4.5)" -- `(Section~4.5)".` found on letter lines {stray} '
                    '-- left by the B.6 paste.'},
        {'id': 'G6', 'lines': f"{lno('the horizon (three representative years standing for five-year blocks)')}, "
                              f"{lno('every ADMM cycle solves one local problem per network block')}",
         'kind': 'correction not applied',
         'finding': 'STEP6_ROUND1_CORRECTIONS B.6 second part: "wherever \'three representative years standing for '
                    'five-year blocks\' is stated as the instance, add \'(single-scenario instance; the multi-scenario '
                    f'instance uses five three-year blocks)\'". Qualifier present anywhere in the letter: {b6_present}. '
                    'R1.2 reads "the horizon (three representative years standing for five-year blocks)" (R2.2 states the '
                    'single-scenario instance explicitly).'},
        {'id': 'G7', 'lines': 'Further changes', 'kind': 'correction not applied',
         'finding': 'Addendum 68 Decision 3: "the letter\'s \'Further changes\' names the correction (÷4 h, not ÷5)". '
                    f'The "Further changes" list names no cost-file correction (searched the section for {searched_terms}: '
                    f'found {cost_terms}); the only mention is the opening paragraph\'s "We also corrected the '
                    'investment-cost input file".'},
        {'id': 'G8', 'lines': str(lno('The three references were added to the literature review')), 'kind': 'internal contradiction (unchanged)',
         'finding': '"The three references were added" against the bracketed status in the same response (two "not yet '
                    'added"); B.8 keeps the status note until the references are in.'},
        {'id': 'K1', 'lines': str(lno('It changes no sign in the 60 reported comparisons and two verdicts')),
         'kind': 'claim checked, consistent',
         'finding': f'"two verdicts": T1 {[c["claim_id"] for c in vc]} (gross {vc[0]["gross"] if vc else None} -> net '
                    f'{vc[0]["net"] if vc else None}) and T4 2035 - 2030 (gross {X.t["year_ladder"]["gross_v6"]["verdict"]}, '
                    f'net {X.t["year_ladder"]["net_v6"]["verdict"]}). The T4 comparison is not among T1\'s 60 claims; the '
                    'sentence reads "no sign in the 60 ... and two verdicts", so the count spans T1 and T4. Sign changes '
                    f'in T1: {len(X.sign_changes())}. "a two-node plan evaluated under the doubled flexibility price": '
                    'checked (L32v2).'},
        {'id': 'K2', 'lines': str(lno('the datasheet-exact calibration')), 'kind': 'claim checked, consistent',
         'finding': 'R3.6: every named value matches T8 (L45v2); arms in order no_ageing, C2, C2_calfade, C4 '
                    '("datasheet-exact"), C3_midblock ("mid-block evaluation point"), C3_unit ("unit-retention reading"); '
                    'all aged arms determinate, no_ageing within resolution. C3_midblock is the C3 calibration (k '
                    f'{X.ageing["C3_midblock"]["k"]:.0f}, eol retention 0.5) evaluated mid-block, not the baseline '
                    'calibration evaluated mid-block.'},
        {'id': 'K3', 'lines': str(lno('every ADMM cycle solves one local problem per network block')), 'kind': 'claim checked, consistent',
         'finding': 'R2.2: 48 network blocks (3 x 4 x 4) plus one ESSO problem per interface node (3) per cycle '
                    '(shared_energy_storage_data.optimize loops over the active distribution-network nodes).'},
        {'id': 'K4', 'lines': str(lno('the limitations section explains the mechanism behind most of them')),
         'kind': 'claim checked, consistent (as W164 K1)',
         'finding': f'{len(dz)} of the {len(unc)} uncertified SRP1 evaluations are T9 dead-zone entries ({dz}).'},
        {'id': 'K5', 'lines': 'rcomment blocks', 'kind': 'quotation check',
         'finding': f"{sum(r['status'] == 'verbatim' for r in qrows)} of {len(qrows)} quotation segments verbatim, "
                    f"{sum(r['status'] == 'differs' for r in qrows)} differ, {sum(r['status'] == 'not found' for r in qrows)} "
                    'not found (table in the MD).'},
    ]
    return F


def main_findings_v2(X):
    heads = {h['number']: h['title'] for h in X.main.heads if h['number']}
    n35 = [i + 1 for i, ln in enumerate(X.main.lines) if 'Section~3.5' in ln]
    n34 = [i + 1 for i, ln in enumerate(X.main.lines) if SEC2[0] <= i + 1 <= SEC2[1] and 'Section~3.4' in ln]
    pb = X.c('p515_s47_phase_b_record')
    pd_ref = pb.ref(r'^POLL_DESIGN = ')
    cp_ref = pb.ref(r'completion = lattice\.completion\(tuple\(inc')
    m4_lines = f"{X.main_line('Generate the $2n$ OrthoMADS directions')}, {X.main_line('If{$|' + chr(92) + 'mathcal{P}| < n + 1$}')}"
    return [
        {'id': 'M1', 'lines': str(n35), 'kind': 'cross-reference to a section not yet written',
         'finding': f'Section 2 defers the values of eta, SoC^Min/Max/0, eps^Cl, c^Cl, eps^C, c^sigma, eps^E and alpha '
                    f'to "Section~3.5"; main.tex at this commit has sections 3.1-3.4 only '
                    f'({[k + " " + v for k, v in heads.items() if k.startswith("3.")]}). The values are checked against the '
                    'code in the parameter table, for when 3.5 is written.'},
        {'id': 'M2', 'lines': str(n34), 'kind': 'cross-reference to unrevised text',
         'finding': f'"The calibrations used, the calendar retention and the end-of-life floor are given in Section~3.4": '
                    f'3.4 is "{heads.get("3.4")}" -- the submitted text, which carries the 1 %/yr calendar sentence and '
                    f'the 60 % / 80 % floors (line {_main_line_or_none(X, "annual rates of 0.5")}).'},
        {'id': 'M3', 'lines': str(X.main_line('(measured between the first and third turning points)')),
         'kind': 'definition against code', 'finding': 'P_hat "measured between the first and third turning points": '
         'the code takes the three most recent (T[-1] - T[-3]); W163 N11 and paragraphs_v5 wrote "from the third-last to '
         'the last". Token "third" MISMATCH (S205; draft d205 likewise).'},
        {'id': 'M4', 'lines': m4_lines,
         'kind': 'algorithm against code',
         'finding': 'Algorithm 1: "the 2n OrthoMADS directions" and the completion "If |P| < n + 1"; the planning searches '
                    f'as run poll n + 1 directions ({pd_ref}) and add the completion at every '
                    f'unit poll ({cp_ref}). Tokens MISMATCH (S219, S225). '
                    'Doubling / halving / unit termination / l_inf-1 completion match. Whether "improves by a determinate '
                    'margin (Subsection certification)" equals the code\'s improvement test (Phase B: resolution = '
                    'max(bar(x) + bar(inc), sigma_Q)) is an algorithm question for W171b, not checked here.'},
        {'id': 'M5', 'lines': str(X.main_line('The rule above, with its holds, produced every single-scenario evaluation')),
         'kind': 'claim with a pending record',
         'finding': '"The rule above, with its holds, produced every single-scenario evaluation reported in this paper": '
                    'the x = 0 reference ref:7aa017f0 was certified under rule v1 (Addendum 68 Decision 4); W168 (commit '
                    '96037a16, not read by this run) reports its v6 replay certifying at the same cycle. Not checked here.'},
    ]


# ======================================================================================================================
#  7. guards, post-run, MD
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    base = W160.pickle_state()
    counts = dict(base['counts'], w161=dict(W161.PICKLE_COUNTS), w162=dict(W162.PICKLE_COUNTS),
                  w163=dict(W163.PICKLE_COUNTS), w164=dict(W164.PICKLE_COUNTS), w171=dict(PICKLE_COUNTS))
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('log', 'man', 'json', 'md', 'typing_json', 'typing_log')]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W171 post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W171 post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}: " +
         ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def _cell(x, n=None):
    s = '' if x is None else str(x)
    s = s.replace('|', '/').replace('\n', ' ')
    return s if n is None else s[:n]


def md_summary(o):
    s = o['summary']
    L = ['# W171a -- number check of the manuscript .tex files at Overleaf 42794d4 (declarations v2)', '',
         f"Overleaf clone `{o['manuscript']['dir']}` at commit `{o['manuscript']['commit']}` (declared "
         f"`{o['manuscript']['declared_commit']}`); files: " +
         ', '.join(f"`{f['name']}` sha256 `{f['sha256'][:8]}`" for f in o['manuscript']['files']) + '.',
         f"Frozen tables `{os.path.basename(W164.FZ_REL)}` (sha256 `{W164.FZ_SHA[:8]}`). Script `{SCRIPT_REL}` (imports "
         f"`{W164_SCRIPT}` @ {W164_SCRIPT_COMMIT}, not edited). Submitted source `{SUBMITTED_MAIN}` (sha256 "
         f"`{SUBMITTED_MAIN_SHA[:8]}`). Reviewers' document `{DOCX}` (sha256 `{DOCX_SHA[:8]}`; {o['docx']['method']}). "
         'ZERO SOLVES (guards verified 0), pickle blocked. Nothing in the clone is edited.', '',
         'Statuses: match (declared / rule against a named record / auto-unique / auto-ambiguous), MISMATCH, approximate, '
         'no table counterpart, submitted-version figure (main.tex, cites the revision-map line), reviewer quotation, '
         'verified (letter quotations), unchecked (with the reason). Excluded LaTeX structure is counted separately.', '',
         '## Counts per file -- manuscript', '']
    hdr = ('| file | tokens | excluded | in scope (body) | comments | match declared | match rule | match auto-unique | '
           'match auto-ambiguous | MISMATCH | approximate | no table counterpart | submitted-version | reviewer quotation, '
           'verified | unchecked | unassigned |')
    sep = '|---|' + '---:|' * 15

    def row(f, c):
        return (f"| {f} | {c['tokens']} | {c['excluded']} | {c['in_scope_body']} | {c['in_scope_comment']} | "
                f"{c['match_declared']} | {c['match_rule']} | {c['match_auto_unique']} | {c['match_auto_ambiguous']} | "
                f"{c['MISMATCH']} | {c['approximate']} | {c['no table counterpart']} | {c['submitted-version figure']} | "
                f"{c['reviewer quotation, verified']} | {c['unchecked']} | {c['unassigned']} |")
    L += [hdr, sep] + [row(f, c) for f, c in s['per_file'].items() if f != DRAFT]
    L += ['', '## Counts -- draft, not compiled (`section2_expert_draft.tex`, not \\input into main.tex; apart from the '
              'manuscript counts)', '', hdr, sep] + [row(f, c) for f, c in s['per_file'].items() if f == DRAFT]
    L += ['', f"Every token assigned: **{s['every_token_assigned']}**; stale declarations: {s['stale_declarations']}; "
              f"tokens claimed twice: {s['claimed_twice']}.", '', '## MISMATCH (all files)', '',
          '| file | line | col | written | counterpart at written precision | source | note |', '|---|---:|---:|---|---|---|---|']
    mm = [t for t in o['tokens'] if t.get('status') == 'MISMATCH']
    for t in mm:
        L.append(f"| {t['file']}{' (draft)' if t['file'] == DRAFT else ''} | {t['line']} | {t['col']} | {t['written']} | "
                 f"{_cell(t.get('value_at_written_precision'))} | {_cell(t.get('counterpart'), 300)} | "
                 f"{_cell(t.get('note'), 400)} |")
    if not mm:
        L.append('| none | | | | | | |')
    L += ['', '## Approximate / no table counterpart', '']
    for t in o['tokens']:
        if t.get('status') in ('approximate', 'no table counterpart'):
            L.append(f"- `{t['file']}` line {t['line']}: `{t['written']}` ({t['status']}) -- {t.get('note') or ''}")
    L += ['', '## W164 declarations re-located (version 1 -> version 2)', '',
          '| v1 id | file | fragment found (times) | outcome | v2 declaration | reason / what replaced it |',
          '|---|---|---:|---|---|---|']
    for r in o['relocation']:
        L.append(f"| {r['id']} | {r['file']} | {r['hits']} | {r['outcome']} | {r.get('v2') or ''} | {_cell(r.get('reason'), 300)} |")
    L += ['', '## Reviewer quotations against the reviewers\' document', '',
          f"Document: `{DOCX}` sha256 `{DOCX_SHA[:8]}`, word/document.xml sha256 `{o['docx']['document_xml_sha256'][:8]}`, "
          f"{o['docx']['n_paragraphs']} paragraphs. Normalisation on both sides: LaTeX quotes and Word curly quotes -> "
          'straight, \\% -> %, ~ and runs of white space -> one space, math delimiters and command names dropped; '
          'enumerate items and \\\\ start a new segment (the document interleaves the authors\' earlier replies).', '',
          '| quote | segment | letter lines | chars | result | match ratio | docx paragraph | start | differences (letter -> document) |',
          '|---:|---:|---|---:|---|---:|---:|---|---|']
    for r in o['quotations']:
        d = '; '.join(f"{x['op']} l.{x['letter_line']}: ...{x['context_before'][-20:]}[{x['letter']!r} -> "
                      f"{x['document']!r}]{x['context_after'][:20]}..." for x in r['diffs'][:6])
        L.append(f"| {r['quote']} | {r['segment']} | {r['lines']} | {r['chars']} | {r['status']} | {r['ratio']:.3f} | "
                 f"{r['docx_paragraph']} | {_cell(r['start'], 50)} | {_cell(d, 600)} |")
    L += ['', f"Quotation tokens: {s['quotation_tokens']}", '',
          '## Revised section 2 of main.tex (l. 315-898): every number with its source', '',
          '| line | written | status | check | counterpart (at written precision) | source / reason |', '|---:|---|---|---|---|---|']
    for t in o['tokens']:
        if t['file'] == MAIN and SEC2[0] <= t['line'] <= SEC2[1] and not t.get('excluded'):
            L.append(f"| {t['line']} | {t['written']} | {t['status_display']} | {t.get('check_id') or t.get('category') or ''} | "
                     f"{_cell(t.get('value_at_written_precision'), 50)} | {_cell(t.get('counterpart') or t.get('note'), 260)} |")
    L += ['', '## Parameter values in force (Addendum 68 list; section 2 defers them to "Section 3.5", not yet in '
              'main.tex) -- each checked against its source', '',
          '| id | quantity | stated (Addendum 68) | found | status | sources | note |', '|---|---|---|---|---|---|---|']
    for p in o['parameter_table']:
        L.append(f"| {p['id']} | {_cell(p['quantity'])} | {_cell(p['stated_addendum_68'])} | {_cell(p['found'], 300)} | "
                 f"**{p['status']}** | {_cell('; '.join(p['sources']), 600)} | {_cell(p['note'], 400)} |")
    L += ['', '## Response letter: every number with its source', '',
          '| line | written | scope | status | check | counterpart (at written precision) | source / reason |',
          '|---:|---|---|---|---|---|---|']
    for t in o['tokens']:
        if t['file'] == LETTER and not t.get('excluded'):
            L.append(f"| {t['line']} | {t['written']} | {t['scope']}{'/' + t['macro'] if t.get('macro') else ''} | "
                     f"{t['status_display']} | {t.get('check_id') or t.get('category') or ''} | "
                     f"{_cell(t.get('value_at_written_precision'), 60)} | {_cell(t.get('counterpart') or t.get('note'), 260)} |")
    L += ['', '## Findings -- letter', '']
    for f in o['letter_findings']:
        L.append(f"- **{f['id']}** (line {f['lines']}, {f['kind']}): {f['finding']}")
    L += ['', '## Findings -- main.tex section 2', '']
    for f in o['main_findings']:
        L.append(f"- **{f['id']}** (line {f['lines']}, {f['kind']}): {f['finding']}")
    for fname in (HIGHLIGHTS, COVER):
        L += ['', f'## {fname}: every number with its source', '']
        rows = [t for t in o['tokens'] if t['file'] == fname and not t.get('excluded')]
        for t in rows:
            L.append(f"- line {t['line']} `{t['written']}`: {t['status_display']} -- {t.get('counterpart') or t.get('note') or ''}")
        if not rows:
            L.append('no in-scope numeric token (structure only: ' + ', '.join(
                f"`{t['text']}` ({t['excluded']})" for t in o['tokens'] if t['file'] == fname) + ')')
    L += ['', '## Draft (not compiled): every number with its source', '',
          '| line | scope | written | status | check | counterpart | source / reason |', '|---:|---|---|---|---|---|---|']
    for t in o['tokens']:
        if t['file'] == DRAFT and not t.get('excluded'):
            L.append(f"| {t['line']} | {t['scope']} | {t['written']} | {t['status_display']} | "
                     f"{t.get('check_id') or t.get('category') or ''} | {_cell(t.get('value_at_written_precision'), 40)} | "
                     f"{_cell(t.get('counterpart') or t.get('note'), 200)} |")
    L += ['', '## main.tex: submitted-version figures by section (line rules re-pinned to 7effd898)', '',
          '| section | n | map lines cited |', '|---|---:|---|']
    for sec, v in s['main_submitted_by_section'].items():
        L.append(f"| {sec} | {v['n']} | {', '.join(v['cites'])} |")
    L += ['', '## main.tex: statuses by section', '', '| section | status | n |', '|---|---|---:|']
    for sec, v in s['main_status_by_section'].items():
        for st, n in v.items():
            L.append(f'| {sec} | {st} | {n} |')
    L += ['', f"Auto index ({s['n_frozen_json_numeric_leaves']} leaves; main.tex outside section 2 only): matches "
              f"{sum(1 for t in o['tokens'] if t.get('match_kind', '').startswith('auto'))}; tokens with >= 1 candidate "
              f"assigned another status by precedence: {s['auto_candidates_overridden']}", '',
          '## Unchecked: by category (all files)', '', '| file | category | n |', '|---|---|---:|']
    for f, cats in s['unchecked_by_category'].items():
        for c, n in cats.items():
            L.append(f'| {f} | {c} | {n} |')
    L += ['', '## Excluded LaTeX structure', '', '| file | category | n | tokens |', '|---|---|---:|---|']
    for f, cats in s['excluded_by_category'].items():
        for c, v in cats.items():
            L.append(f"| {f} | {c} | {v['n']} | {', '.join(v['examples'])} |")
    L += ['', '## Integrity checks', '']
    for k, v in o['checks'].items():
        L.append(f'- {k}: {v}')
    return '\n'.join(L) + '\n'


# ======================================================================================================================
#  8. main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--manuscript-dir', default=DEFAULT_MANUSCRIPT_DIR)
    ap.add_argument('--overleaf-commit', required=True, help='the Overleaf commit the run declares (>= 7 hex)')
    ap.add_argument('--expect', action='append', default=[], help='NAME=SHA256 for every *.tex file (declared)')
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--trial-out-dir', default=None, help='trial runs only: an output directory outside the results')
    args = ap.parse_args()
    out_dir = args.trial_out_dir or os.path.join(OUT_ROOT, f'overleaf_{args.overleaf_commit[:7]}')
    if args.post_run:
        post_run(out_dir)
    t0 = time.time()
    tag = 'W171'
    pre = []
    od = os.path.join(REPO, out_dir) if not os.path.isabs(out_dir) else out_dir
    if not os.path.isdir(od):
        pre.append(f'{out_dir} missing (the launch command creates it with mkdir; the launch log goes there)')
    else:
        extra = sorted(set(os.listdir(od)) - {OUT_NAMES['log']})
        if extra:
            pre.append(f'{out_dir} is not new: it holds {extra} (write-once; never overwrite a previous output)')
    if not re.fullmatch(r'[0-9a-f]{7,40}', args.overleaf_commit):
        pre.append(f'--overleaf-commit {args.overleaf_commit!r} is not a hex commit')
    mdir = args.manuscript_dir
    mabs = os.path.join(REPO, mdir)
    expect = {}
    for e in args.expect:
        if '=' not in e:
            pre.append(f'--expect {e!r}: NAME=SHA256')
            continue
        k, v = e.split('=', 1)
        expect[k] = v
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(mabs, '*.tex')))
    head = subprocess.run(['git', '-C', mabs, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(['git', '-C', mabs, 'status', '--porcelain', '--', '*.tex'], capture_output=True,
                               text=True).stdout.strip()
    mfiles = []
    for f in files:
        with open(os.path.join(mabs, f), 'rb') as h:
            b = h.read()
        blob = subprocess.run(['git', '-C', mabs, 'show', f'{args.overleaf_commit}:{f}'], capture_output=True)
        mfiles.append({'name': f, 'path': os.path.join(mdir, f), 'sha256': _sha_bytes(b), 'bytes': len(b),
                       'lines': b.decode('utf-8').count('\n') + 1,
                       'blob_at_commit_sha256': _sha_bytes(blob.stdout) if blob.returncode == 0 else None,
                       'declared_sha256': expect.get(f), 'scope': 'draft, not compiled' if f == DRAFT else 'manuscript'})
    mchecks = {
        'clone_head_is_declared_commit': bool(head) and head.startswith(args.overleaf_commit),
        'file_set_equals_declared': set(files) == set(expect),
        'every_sha_equals_declared': all(m['sha256'] == m['declared_sha256'] for m in mfiles),
        'every_file_equals_its_blob_at_commit': all(m['sha256'] == m['blob_at_commit_sha256'] for m in mfiles),
        'no_tex_modified_or_untracked_in_clone': porcelain == '',
    }
    for k, v in mchecks.items():
        if not v:
            pre.append(f'manuscript: {k} FAILED (head {head}, files {files}, declared {sorted(expect)}, porcelain '
                       f'{porcelain!r}, shas {[(m["name"], m["sha256"][:8], (m["declared_sha256"] or "")[:8]) for m in mfiles]})')
    input_rels = {'FROZEN_JSON': W164.FZ_REL, 'FROZEN_MD': W164.FZ_MD_REL, 'W163_JSON': W164.W163_JSON,
                  'W163_MANIFEST': W164.W163_MAN, 'W163_SCRIPT': W164.W163_SCRIPT, 'W164_SCRIPT': W164_SCRIPT,
                  'W164_RESULTS': W164_RESULTS, 'P5': W164.P5_REL, 'P2': W164.P2_REL, 'MAP': W164.MAP_REL,
                  'SRP1_JSON': W164.SRP1_JSON, 'SRP1_PARAMS': SRP1_PARAMS, 'ESS_PARAMS': ESS_PARAMS,
                  'STEP4': W164.STEP4, 'A1_CAMPAIGN': W164.A1_CAMPAIGN, 'V41_SPEC': W164.V41_SPEC, 'V6_SPEC': V6_SPEC,
                  'SPEC_3X3': SPEC_3X3, 'INST_3X3': INST_3X3, 'EXT_SPEC_V3': EXT_SPEC_V3,
                  'SUBMITTED_MAIN': SUBMITTED_MAIN, 'DOCX': DOCX, 'CORR_MD': CORR_MD}
    for k, v in W164.CASE_FILES.items():
        input_rels[f'CASE_{k}'] = v
    for k, v in CASE_PARAMS.items():
        input_rels[f'CP_{k}'] = v
    for k, v in CODE.items():
        input_rels[f'CODE_{k}'] = v
    inputs = {}
    for key, rel in input_rels.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} missing')
            continue
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': W160.W157.L132._committed_clean(rel),
                       'last_commit': W160._last_commit(rel)}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    if not pre:
        for key, want in (('FROZEN_JSON', W164.FZ_SHA), ('P5', W164.P5_SHA), ('P2', W164.P2_SHA),
                          ('SUBMITTED_MAIN', SUBMITTED_MAIN_SHA), ('DOCX', DOCX_SHA)):
            if inputs[key]['sha256'] != want:
                pre.append(f"{inputs[key]['path']} sha {inputs[key]['sha256']} != {want}")
        if _jl(W164.W163_MAN).get(W164.W163_JSON) != inputs['W163_JSON']['sha256']:
            pre.append(f'{W164.W163_JSON} != its W163 manifest entry')
        if not (inputs['W164_SCRIPT']['last_commit'] or '').startswith(W164_SCRIPT_COMMIT):
            pre.append(f"{W164_SCRIPT} last commit {inputs['W164_SCRIPT']['last_commit']} != {W164_SCRIPT_COMMIT}")
        if not (inputs['W163_SCRIPT']['last_commit'] or '').startswith(W164.W163_SCRIPT_COMMIT):
            pre.append(f"{W164.W163_SCRIPT} last commit {inputs['W163_SCRIPT']['last_commit']} != {W164.W163_SCRIPT_COMMIT}")
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    _log(f'[{tag}] script {SCRIPT_REL} sha256 {_sha(SCRIPT_REL)} committed clean {script_clean}; {len(inputs)} inputs '
         f'committed clean; frozen JSON {W164.FZ_SHA[:8]}, submitted main.tex {SUBMITTED_MAIN_SHA[:8]}, reviewers\' '
         f'docx {DOCX_SHA[:8]} verified; manuscript {mdir} HEAD {head} (declared {args.overleaf_commit}); files ' +
         ', '.join(f"{m['name']} {m['sha256'][:8]}" for m in mfiles))
    # ---- tokenize --------------------------------------------------------------------------------------------------------
    docs = {m['name']: W164.Doc(m['name'], _text(m['path'])) for m in mfiles}
    roles = {n: ('main' if n == MAIN else 'letter' if n.startswith('response_to_reviewers') else
                 'highlights' if n == HIGHLIGHTS else 'cover' if n == COVER else 'draft' if n == DRAFT else 'other')
             for n in docs}
    toks = []
    for n in files:
        toks += W164.tokenize(docs[n])
    sub_doc = W164.Doc('main.tex', _text(SUBMITTED_MAIN))
    X = RecordsV2(inputs, docs.get(MAIN), sub_doc)
    X.letter_doc = docs.get(LETTER)
    X.letter_raw = docs[LETTER].raw if LETTER in docs else ''
    X.cover_raw = docs[COVER].raw if COVER in docs else ''
    X.main_tokens = [t for t in toks if t['file'] == MAIN and t['scope'] == 'body' and not t['excluded']]
    X.sub_tokens = [t for t in W164.tokenize(sub_doc) if t['scope'] == 'body' and not t['excluded']]
    X.main_sha = next((m['sha256'] for m in mfiles if m['name'] == MAIN), None)
    X.corr_text = _text(CORR_MD)
    leaves = W164.json_leaves(X.fz)
    AX = W164.AutoIndex(leaves)
    map_cite_lines, map_cite_fail = {}, []
    for k, frag in W164.MAP_CITES.items():
        try:
            map_cite_lines[k] = X.map_line(frag)
        except KeyError as e:
            map_cite_fail.append(str(e))
    map_xref_numbers = set(re.findall(r'(?<![\d.])(\d\.\d(?:\.\d)?)(?![\d.])', '\n'.join(X.map)))
    # ---- reviewer quotations -------------------------------------------------------------------------------------------
    dx = docx_text(DOCX)
    qrows = quotation_check(docs[LETTER], dx)
    qstat = quotation_token_status(qrows, docs[LETTER], toks)
    # ---- relocation of the version-1 declarations ----------------------------------------------------------------------
    relocation = []
    v1 = W164.letter_declarations() + W164.cover_declarations() + W164.main_declarations()
    for d in v1:
        doc = docs.get(d['file'])
        hits = len(doc.find_fragment(d['fragment'])[0]) if doc else 0
        if d['id'] in SUPERSEDED:
            outcome, v2, why = 'superseded', SUPERSEDED[d['id']][0], SUPERSEDED[d['id']][1]
        elif hits == 1:
            outcome, v2, why = 'carried', d['id'], 'fragment found once; evaluated unchanged'
        elif d['id'] in REPLACED_BY:
            outcome, v2, why = 'removed', REPLACED_BY[d['id']][0], REPLACED_BY[d['id']][1]
        else:
            outcome, v2, why = 'removed (NO REPLACEMENT DECLARED)', None, 'fragment not found once and no v2 declaration names it'
        relocation.append({'id': d['id'], 'file': d['file'], 'fragment': d['fragment'], 'hits': hits, 'outcome': outcome,
                           'v2': v2, 'reason': why})
    carried_ids = {r['id'] for r in relocation if r['outcome'] == 'carried'}
    unreplaced = [r['id'] for r in relocation if r['outcome'].startswith('removed (NO')]
    decls = [d for d in letter_declarations_v2()] + \
        [d for d in v1 if d['id'] in carried_ids] + \
        [d for d in all_declarations_v2() if d['file'] != LETTER and not re.fullmatch(r'(L|CL|M)\d+', d['id'])]
    seen_ids = collections.Counter(d['id'] for d in decls)
    dup_ids = sorted(k for k, n in seen_ids.items() if n > 1)
    # ---- declared checks -------------------------------------------------------------------------------------------------
    by_key = {t['key']: t for t in toks}
    assigned, claimed_twice, stale = {}, [], []
    decl_out = []
    for d in decls:
        if d['file'] not in docs:
            stale.append({'id': d['id'], 'why': f"{d['file']} not in the manuscript directory"})
            continue
        doc = docs[d['file']]
        hits, flen = doc.find_fragment(d['fragment'])
        if len(hits) != 1:
            stale.append({'id': d['id'], 'why': f'fragment found {len(hits)} times', 'fragment': d['fragment']})
            continue
        a = hits[0]
        inside = [t for t in toks if t['file'] == d['file'] and not t['excluded'] and
                  doc.flat_pos(t['line'], t['col0']) is not None and a <= doc.flat_pos(t['line'], t['col0']) < a + flen]
        picked, ok = [], True
        for txt, occ in d['tokens']:
            c = [t for t in inside if t['text'] == txt]
            if len(c) <= occ:
                ok = False
                stale.append({'id': d['id'], 'why': f'token {txt!r}#{occ} not in the fragment'})
                break
            picked.append(c[occ])
        if not ok:
            continue
        try:
            res = d['fn'](X, [t['written'] for t in picked])
        except Exception as e:  # a declaration that cannot evaluate is stale, loudly
            stale.append({'id': d['id'], 'why': f'evaluation error {type(e).__name__}: {e}'})
            continue
        if isinstance(res, dict):
            res = [res] * len(picked)
        if len(res) != len(picked):
            stale.append({'id': d['id'], 'why': f'{len(res)} results for {len(picked)} tokens'})
            continue
        fl = doc.flat_idx[a]
        decl_out.append({'id': d['id'], 'file': d['file'], 'fragment': d['fragment'], 'line': fl[0],
                         'version': 1 if d['id'] in carried_ids else 2,
                         'tokens': [[t['line'], t['col'], t['written']] for t in picked]})
        for t, r in zip(picked, res):
            if t['key'] in assigned:
                claimed_twice.append([t['key'], assigned[t['key']]['check_id'], d['id']])
                continue
            r = dict(r)
            r['check_id'] = d['id']
            r['rule'] = 'declared'
            assigned[t['key']] = r
    # ---- rules, auto index ---------------------------------------------------------------------------------------------
    for t in toks:
        t['auto_candidates'] = AX.candidates(t) if not t['excluded'] else None
        if t['excluded']:
            t['status'], t['status_display'] = 'excluded', f"excluded ({t['excluded']})"
            continue
        in_sec2 = t['file'] == MAIN and SEC2[0] <= t['line'] <= SEC2[1]
        r = assigned.get(t['key'])
        if r is None:
            r = rule_assign_v2(X, docs[t['file']], roles[t['file']], t, map_xref_numbers, qstat)
            if r is not None:
                r = dict(r)
                r['rule'] = 'rule'
        if r is not None and r['status'] == 'match' and r['rule'] == 'rule':
            r['match_kind'] = 'rule (named record)'
        if r is None and roles[t['file']] == 'main' and not in_sec2 and t['auto_candidates']:
            c = t['auto_candidates']
            r = R('match', c[0]['path'] if len(c) == 1 else f'{len(c)} paths', 'frozen JSON (auto index)', None, None,
                  None)
            r['rule'], r['match_kind'] = 'auto', 'auto-unique' if len(c) == 1 else 'auto-ambiguous'
        if r is None and roles[t['file']] == 'main' and not in_sec2 and X.main_sha == MAIN_LINE_RULES_SHA:
            r = ntc('auto index: no candidate; no declaration or rule this round (submitted text outside section 2)')
            r['rule'] = 'leftover'
        if r is None:
            t['status'], t['status_display'] = None, 'UNASSIGNED'
            continue
        t.update({k: v for k, v in r.items()})
        if t['status'] == 'match' and 'match_kind' not in r:
            t['match_kind'] = 'declared'
        mk = t.get('match_kind')
        t['status_display'] = (f"match (auto, {t['counterpart']})" if mk == 'auto-unique' else
                               f"match (auto, ambiguous: {len(t['auto_candidates'])} paths)" if mk == 'auto-ambiguous'
                               else "MISMATCH vs reviewers' document" if t.get('mismatch_vs') else t['status'])
    unassigned = [t['key'] for t in toks if not t['excluded'] and t['status'] is None]
    # ---- parameter table -------------------------------------------------------------------------------------------------
    try:
        ptable = parameter_checks(X)
        ptable_error = None
    except Exception as e:  # a parameter whose source moved fails the run, loudly
        ptable, ptable_error = [], f'{type(e).__name__}: {e}'
    # ---- summary -------------------------------------------------------------------------------------------------------
    per_file = {}
    for n in files:
        ft = [t for t in toks if t['file'] == n]
        sc = [t for t in ft if not t['excluded']]
        c = {'scope': 'draft, not compiled' if n == DRAFT else 'manuscript', 'tokens': len(ft),
             'excluded': len(ft) - len(sc), 'in_scope_body': sum(t['scope'] == 'body' for t in sc),
             'in_scope_comment': sum(t['scope'] == 'comment' for t in sc),
             'match_declared': sum(t['status'] == 'match' and t.get('match_kind') == 'declared' for t in sc),
             'match_rule': sum(t['status'] == 'match' and t.get('match_kind') == 'rule (named record)' for t in sc),
             'match_auto_unique': sum(t.get('match_kind') == 'auto-unique' for t in sc),
             'match_auto_ambiguous': sum(t.get('match_kind') == 'auto-ambiguous' for t in sc),
             'unassigned': sum(t['status'] is None for t in sc)}
        for st in STATUSES[1:]:
            c[st] = sum(t['status'] == st for t in sc)
        c['MISMATCH_vs_reviewers_document'] = sum(bool(t.get('mismatch_vs')) for t in sc)
        c['match_total'] = c['match_declared'] + c['match_rule'] + c['match_auto_unique'] + c['match_auto_ambiguous']
        c['assigned_total'] = c['match_total'] + sum(c[st] for st in STATUSES[1:])
        per_file[n] = c
    sub_by_sec, stat_by_sec = collections.OrderedDict(), collections.OrderedDict()
    for t in toks:
        if t['file'] != MAIN or t['excluded']:
            continue
        sec = t['section'].split(' > ')[0] if t['scope'] == 'body' else '(comments)'
        stat_by_sec.setdefault(sec, collections.Counter())[
            f"match ({t['match_kind']})" if t['status'] == 'match' else t['status_display']] += 1
        if t['status'] == 'submitted-version figure':
            v = sub_by_sec.setdefault(sec, {'n': 0, 'cites': []})
            v['n'] += 1
            if t.get('map_cite') and t['map_cite'].split(':')[1] not in v['cites']:
                v['cites'].append(t['map_cite'].split(':')[1])
    unc_cat, exc_cat = {}, {}
    for t in toks:
        if t['excluded']:
            e = exc_cat.setdefault(t['file'], {}).setdefault(t['excluded'], {'n': 0, 'examples': []})
            e['n'] += 1
            if len(e['examples']) < 8 and t['text'] not in e['examples']:
                e['examples'].append(t['text'])
        elif t['status'] == 'unchecked':
            unc_cat.setdefault(t['file'], collections.Counter())[t.get('category') or 'other'] += 1
    overridden = collections.Counter(t['status'] or 'UNASSIGNED' for t in toks if not t['excluded'] and
                                     t['auto_candidates'] and not t.get('match_kind', '').startswith('auto'))
    qtok = collections.Counter(t['status_display'] for t in toks if t['file'] == LETTER and t.get('macro') == 'rcomment'
                               and not t['excluded'])
    summary = {'per_file': per_file, 'every_token_assigned': not unassigned, 'unassigned': unassigned,
               'stale_declarations': stale, 'claimed_twice': claimed_twice, 'duplicate_declaration_ids': dup_ids,
               'main_submitted_by_section': sub_by_sec,
               'main_status_by_section': {k: dict(v) for k, v in stat_by_sec.items()},
               'unchecked_by_category': {k: dict(v) for k, v in unc_cat.items()},
               'excluded_by_category': exc_cat, 'auto_candidates_overridden': dict(overridden),
               'n_frozen_json_numeric_leaves': len(leaves), 'n_declarations': len(decls),
               'n_declarations_applied': len(decl_out), 'quotation_tokens': dict(qtok),
               'quotation_segments': dict(collections.Counter(r['status'] for r in qrows)),
               'relocation': dict(collections.Counter(r['outcome'] for r in relocation)),
               'parameter_table': dict(collections.Counter(p['status'] for p in ptable))}
    lf = letter_findings_v2(X, docs, qrows, dx)
    mf = main_findings_v2(X)
    checks = dict(mchecks)
    checks.update({
        'frozen_json_unchanged': _sha(W164.FZ_REL) == W164.FZ_SHA,
        'every_declaration_found_and_evaluated': not stale,
        'no_token_claimed_twice': not claimed_twice,
        'no_duplicate_declaration_id': not dup_ids,
        'every_token_assigned': not unassigned,
        'every_map_citation_found_once': not map_cite_fail,
        'statuses_in_vocabulary': all(t['excluded'] or t['status'] in STATUSES for t in toks if t['status']),
        'main_line_rules_pinned_to_this_main': X.main_sha in (None, MAIN_LINE_RULES_SHA),
        'every_v1_declaration_carried_superseded_or_replaced': not unreplaced,
        'parameter_table_evaluated': ptable_error is None and bool(ptable),
        'every_quotation_segment_located': all(r['status'] != 'not found' for r in qrows),
        'docx_sha_equals_declared': _sha(DOCX) == DOCX_SHA,
    })
    failed = sorted(k for k, v in checks.items() if v is not True)
    guards, pk, guards_ok = guards_state()
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    for t in toks:
        for k in ('col0', 'end0'):
            t.pop(k, None)
        if t.get('auto_candidates') is not None:
            t['auto_candidates_n'] = len(t['auto_candidates'])
            t['auto_candidates'] = t['auto_candidates'][:10]
    for r in qrows:
        r.pop('offsets', None)
    out = {'schema': 'p515_s53_w171_manuscript_number_check', 'version': 1, 'declarations_version': DECLARATIONS_VERSION,
           'stage': 'P5.15 Step 6 W171a -- number check of the manuscript .tex files at Overleaf 42794d4 (W164 logic, '
                    'declarations v2; reviewer quotations against the reviewers\' document)',
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'imports': {W164_SCRIPT: _sha(W164_SCRIPT)}},
           'command_line': sys.argv,
           'manuscript': {'dir': mdir, 'commit': head, 'declared_commit': args.overleaf_commit, 'files': mfiles,
                          'roles': roles, 'clone_porcelain_tex': porcelain, 'edited': False},
           'submitted_source': {'path': SUBMITTED_MAIN, 'sha256': SUBMITTED_MAIN_SHA,
                                'figures': _figures(sub_doc)},
           'docx': {k: v for k, v in dx.items() if k not in ('text', 'para_of')},
           'frozen_tables': {'path': W164.FZ_REL, 'sha256': W164.FZ_SHA, 'numeric_leaves': len(leaves)},
           'inputs': inputs, 'map_citations': map_cite_lines, 'map_citation_failures': map_cite_fail,
           'main_line_rules': {'sha256': MAIN_LINE_RULES_SHA, 'section2': SEC2, 'shift_after_section2': SHIFT,
                               'regions': MAIN_REGIONS, 'appendix_A': APP_A, 'network_tables': NETWORK_TABLES,
                               'appendix_BCD': APP_BCD},
           'conventions': {
               'column': '1-based', 'line': '1-based',
               'written': 'the token as written, sign included, LaTeX thousands separators as ","',
               'precedence': 'excluded structure -> comments (draft: declared value-bearing comments first) -> declared '
                             'checks (v1 carried, v2) -> rules (reviewer quotations; identifiers; cross-references; number '
                             'words; main.tex: re-pinned map regions, years, IEEE n-bus, nodes, algorithms, math, data '
                             'tables; revised section 2 and the draft: notation in formulas / Algorithm 1, structural '
                             'number words) -> auto index (main.tex OUTSIDE section 2) -> leftover (main.tex outside '
                             'section 2). A section-2 token left unassigned fails the run.',
               'quotation': 'segment verbatim = its normalised text is a substring of the normalised document; '
                            'otherwise aligned (difflib, autojunk off) to the document window found by a 30-char anchor; '
                            'a token is verified when all its characters lie in equal blocks',
               'objective_convention': X.fz.get('objective_convention')},
           'summary': summary, 'checks': checks, 'failed': failed, 'relocation': relocation,
           'declarations_applied': decl_out, 'quotations': qrows, 'parameter_table': ptable,
           'parameter_table_error': ptable_error, 'letter_findings': lf, 'main_findings': mf, 'tokens': toks,
           'guards': guards, 'pickle_guard': pk, 'exit_code': code, 'wall_s': time.time() - t0}
    written = {}

    def wr(name, data):
        rel = os.path.join(out_dir, OUT_NAMES[name])
        with open(os.path.join(REPO, rel), 'xb') as h:      # an absolute trial path survives the join
            h.write(data if isinstance(data, bytes) else data.encode('utf-8'))
        written[rel] = _sha(rel)
    wr('json', GRIO.dumps(out, indent=1, sort_keys=True) + '\n')
    wr('md', md_summary(out))
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for v in inputs.values():
        man[v['path']] = v['sha256']
    for m in mfiles:
        man[m['path']] = m['sha256']
    wr('man', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    for n, c in per_file.items():
        _log(f"[{tag}] {n} ({c['scope']}): tokens {c['tokens']} (excluded {c['excluded']}, in scope {c['in_scope_body']} "
             f"body + {c['in_scope_comment']} comment) -- match {c['match_total']} (declared {c['match_declared']}, rule "
             f"{c['match_rule']}, auto unique {c['match_auto_unique']}, auto ambiguous {c['match_auto_ambiguous']}), "
             f"MISMATCH {c['MISMATCH']} (vs reviewers' document {c['MISMATCH_vs_reviewers_document']}), approximate "
             f"{c['approximate']}, no table counterpart {c['no table counterpart']}, submitted-version "
             f"{c['submitted-version figure']}, reviewer quotation verified {c['reviewer quotation, verified']}, "
             f"unchecked {c['unchecked']}, unassigned {c['unassigned']}")
    for t in toks:
        if t['status'] in ('MISMATCH', 'approximate', 'no table counterpart') and \
                (t['file'] != MAIN or t.get('rule') == 'declared'):
            _log(f"[{tag}]   {t['file']}:{t['line']}:{t['col']} {t['status_display']}: written {t['written']!r} -> "
                 f"{t.get('value_at_written_precision')!r} [{t.get('check_id')}] {str(t.get('counterpart'))[:200]}")
    _log(f"[{tag}] relocation of v1 declarations: {summary['relocation']}; "
         + '; '.join(f"{r['id']} {r['outcome']} -> {r['v2']}" for r in relocation if r['outcome'] != 'carried'))
    _log(f"[{tag}] quotation segments: {summary['quotation_segments']}; quotation tokens: {summary['quotation_tokens']}")
    for r in qrows:
        if r['status'] != 'verbatim':
            _log(f"[{tag}]   quotation {r['quote']}.{r['segment']} (l. {r['lines']}) {r['status']} ratio {r['ratio']:.3f}: " +
                 '; '.join(f"{x['op']} {x['letter']!r} -> {x['document']!r}" for x in r['diffs'][:6]))
    _log(f"[{tag}] parameter table: {summary['parameter_table']}" + (f' ERROR {ptable_error}' if ptable_error else ''))
    for p in ptable:
        if p['status'] != 'match':
            _log(f"[{tag}]   {p['id']} {p['quantity']}: {p['status']} -- found {str(p['found'])[:200]}; {p['note']}")
    _log(f"[{tag}] main.tex submitted-version figures by section: {dict((k, v['n']) for k, v in sub_by_sec.items())}")
    _log(f"[{tag}] auto index: {len(leaves)} numeric leaves; overridden by precedence (status: n) {dict(overridden)}")
    for f in lf + mf:
        _log(f"[{tag}]   finding {f['id']} (l. {f['lines']}, {f['kind']}): {f['finding'][:300]}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    if unassigned:
        _log(f'[{tag}] UNASSIGNED tokens ({len(unassigned)}): ' + ', '.join(
            f"{by_key[k]['key']} {by_key[k]['written']!r}" for k in unassigned[:300]))
    if stale or claimed_twice or map_cite_fail or dup_ids or unreplaced:
        _log(f'[{tag}] stale {stale}; claimed twice {claimed_twice}; map citation failures {map_cite_fail}; duplicate '
             f'ids {dup_ids}; unreplaced v1 {unreplaced}')
    if failed:
        _log(f'[{tag}] FAILED: {failed}')
    _log(f"[{tag}] wrote {', '.join(f'{k} {v[:8]}' for k, v in written.items())}")
    _log(f"[{tag}] guards {[(k, v['counts']['permitted_solve'], v['counts']['blocked_solve'], v['verify_0_failures']) for k, v in guards.items()]}; "
         f"pickle ok {pk['ok']}; exit {code}; wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(code)


if __name__ == '__main__':
    main()
