#!/usr/bin/env python3
"""Verify the supplement written by gen_supplement.py (Addendum 89 ruling 8; Planner task W188 (b)). ZERO SOLVES.

It reads the zip (never the generator's code) and checks:
  V0 identity: the record's zip sha256 = the file's; entries sorted, dated 1980-01-01 00:00:00, mode 0644, ZIP_STORED;
     SHA256SUMS.txt inside = every other entry; the record's member inventory = the zip.
  V1 Appendix E against the FROZEN TABLES: every cell of E20-E24 (and the E.22 / E.23 note numbers) is put back into
     its printed LaTeX form by an inverse of the generator's markup removal written HERE, and the W180a entry check
     (p515_s53_w180a_appendixE_check.table_check, imported, not edited; W183h's form: the over-budget mark b of E.20 /
     E.21 left out) is run on it against frozen_step6_tables_v1_590088fe.json: expected 1,623 entries, 0 MISMATCH,
     0 non-frozen disagreements, 0 mapping problems; cell_id / claim_id columns = the check's own mapping. The four
     E.20 note numbers are checked against the frozen claims' uncertified-form-beside fields.
  V2 Appendix E against the manuscript: every reconstructed cell equals, with the marks removed, the printed cell of
     Tables E.20-E.24 in main.tex at Overleaf a87e195 (git show in the clone, read-only; = main_edd7a63_round8.tex),
     and every mark column equals the marks printed there (the over-budget plan fragments of E.20 included). E.24
     prints no budget mark: its over_budget column is a declared difference, listed.
  V3 the over_budget columns against W182d's committed over-budget evaluations (w182d_claims.json A8_1_budget).
  V4 gen_appE.py run in-process into a temporary directory: equal to the committed appendixE_evaluation_record.tex
     and found exactly once, line-aligned, in main.tex at a87e195; verify_appE.py: 33 PASS, 0 FAIL.
  V5 Appendices B-D against the CASE FILES, read here with an independent reader (openpyxl for the xlsx, json for the
     network files): every cell equal (numbers exactly, as floats), every source row present.
  V6 Appendices B-D against the printed tables of Appendices C-D in main.tex at a87e195, at the printed rounding.
  V7 the README placeholder: per file, its column table = the CSV header; the marked placeholders present; tau.
  Negative controls, in memory only: an altered cell of each kind must be caught where it was altered.
Guards: SolveProfileGuard(permitted=()) before any project import, pickle.load / loads blocked; both verified at 0.
Outputs (opened 'x', never overwritten): verify_supplement_v<N>_result.json (gate_result_io) and a summary on stdout.
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u manuscript_review/verify_supplement.py \\
        > manuscript_review/supplement/verify_launch.log 2>&1
Exit: 0 = every check holds; 3 = a check failed (listed); 1 = precondition or guard fault.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='W188b supplement verifier (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('verify_supplement: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('verify_supplement: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads
import openpyxl  # noqa: E402
import gate_result_io as GRIO  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402
import p515_s53_w180a_appendixE_check as A  # noqa: E402  (W180a entry check; arms its own guard; NOT edited)

os.chdir(REPO)
pickle.load, pickle.loads = _blocked_load, _blocked_loads

MR = 'manuscript_review'
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_REV = 'a87e1950fbdbb3667f37e64a43c298ba675e6cff'
MAIN_SHA = 'cec2e2982dfa1509e6e066a208521fb1eea40d5c71f1cdf71a9bf00a12b3520b'
MAIN_REVIEW_COPY = os.path.join(MR, 'main_edd7a63_round8.tex')
W182D = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w182d_claims', 'w182d_claims.json')
EXPECTED = {'entries': 1623, 'MATCH': 1443, 'not_frozen_agrees': 180,
            'per_table': {'E.20': 639, 'E.21': 854, 'E.22': 18, 'E.23': 49, 'E.24': 63}}
E24_DECLARED_OVER_BUDGET = [39, 6]     # W183h finding 1: E.24 prints these over-budget plans without the mark
DATA = os.path.join('data', 'SRP1')
ADNS = (('case33_1', 5), ('case33_2', 7), ('case33_3', 9))
PRINTED_YEARS = (2025, 2028, 2031, 2034, 2037)
TYPE_CODE = {'Wind': 'WIND', 'Solar PV': 'PV'}
BUS_LABEL = {'Ref.': 3, 'PQ': 1}
FAILS = []


def _log(msg):
    print(f'[verify_supplement] {msg}', flush=True)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _fail(msg):
    FAILS.append(msg)
    _log(f'FAIL {msg}')


def _ok(cond, msg):
    if cond:
        _log(f'PASS {msg}')
    else:
        _fail(msg)
    return cond


def read_csv(b):
    rows = list(csv.reader(io.StringIO(b.decode('utf-8'))))
    return rows[0], [dict(zip(rows[0], r)) for r in rows[1:]], [len(r) for r in rows[1:]]


# ======================================================================================================================
#  V0: the zip and its inventories
# ======================================================================================================================
def check_zip(sdir, zbytes_override=None):
    recs = sorted(f for f in os.listdir(sdir) if re.fullmatch(r'supplement_v\d+_record\.json', f))
    if len(recs) != 1:
        raise RuntimeError(f'{sdir}: {len(recs)} record files')
    with open(os.path.join(sdir, recs[0]), encoding='utf-8') as h:
        rec = json.load(h)
    zpath = os.path.join(sdir, rec['zip']['name'])
    with open(zpath, 'rb') as h:
        zb = h.read() if zbytes_override is None else zbytes_override
    out = {'record': recs[0], 'zip': rec['zip']['name'], 'zip_sha256': _sha(zb), 'record_zip_sha256': rec['zip']['sha256']}
    probs = []
    if _sha(zb) != rec['zip']['sha256']:
        probs.append('zip sha256 differs from the record')
    if rec['zip']['sha256'][:8] not in rec['zip']['name']:
        probs.append('zip name does not carry its hash prefix')
    try:
        z = zipfile.ZipFile(io.BytesIO(zb))
        infos = z.infolist()
    except zipfile.BadZipFile as e:
        return out, [f'bad zip: {e}'], {}
    names = [i.filename for i in infos]
    if names != sorted(names):
        probs.append('entries not sorted')
    for i in infos:
        if i.date_time != (1980, 1, 1, 0, 0, 0) or i.compress_type != zipfile.ZIP_STORED or (i.external_attr >> 16) != 0o100644:
            probs.append(f'entry {i.filename}: date {i.date_time}, compression {i.compress_type}, mode {oct(i.external_attr >> 16)}')
    members = {}
    for n in names:
        try:
            members[n] = z.read(n)
        except Exception as e:  # a corrupted entry is a finding, not a crash
            probs.append(f'entry {n} unreadable: {e!r}')
    sums = {}
    for ln in members.get('SHA256SUMS.txt', b'').decode('utf-8').splitlines():
        h, n = ln.split('  ', 1)
        sums[n] = h
    if set(sums) != set(members) - {'SHA256SUMS.txt'}:
        probs.append('SHA256SUMS.txt does not list exactly the other entries')
    for n, h in sums.items():
        if n in members and _sha(members[n]) != h:
            probs.append(f'SHA256SUMS.txt: {n} hash differs')
    inv = {m['name']: m['sha256'] for m in rec['members']}
    if set(inv) != set(members):
        probs.append('record member list differs from the zip')
    for n, h in inv.items():
        if n in members and _sha(members[n]) != h:
            probs.append(f'record: {n} hash differs')
    out.update({'n_entries': len(names), 'entries': names, 'record_version': rec['version'],
                'record_git_head': rec.get('git_head'), 'generator_sha256': rec['generator']['sha256'],
                'generator_committed_clean': rec['generator']['committed_clean']})
    return out, probs, members


# ======================================================================================================================
#  the inverse of the generator's markup removal (written here; the generator is not imported)
# ======================================================================================================================
def tex_text(s):
    if s == '':
        return '---'
    if s == 'x = 0':
        return '$\\boldsymbol{x}=\\boldsymbol{0}$'
    t = s.replace(' -> ', ' $\\to$ ').replace(' - ', ' $-$ ')
    t = re.sub(r'(?<=\d),(?=\d{3}\b)', r'\\,', t)
    t = t.replace(' %', '\\,\\%')
    t = re.sub(r'\b([Rr]et)\. ', r'\1.\\ ', t)
    t = re.sub(r'(^|, )(m=[0-9.]+)', r'\1$\2$', t)
    return t


def tex_num(s, thousands=False):
    if s == '':
        return '---'
    neg = s.startswith('-')
    t = s[1:] if neg else s
    if thousands:
        ip, dot, dp = t.partition('.')
        ip = '{:,}'.format(int(ip)).replace(',', '\\,')
        t = ip + dot + dp
    return ('$-$' if neg else '') + t


def mk(flag, m):
    return f'$^{{{m}}}$' if flag == 'yes' else ''


def recon_E(E):
    """{table: {row_key: [cells in printed LaTeX form, without the budget mark b]}} and the marks expected per row."""
    R = {'E.20': {}, 'E.21': {}, 'E.22': {}, 'E.23': {}, 'E.24': {}}
    M = {'E.20': {}, 'E.21': {}, 'E.23': {}}
    for r in E['E20']:
        desc = tex_text(r['comparison']) + mk(r['bitwise_twin_row'], 't') + mk(r['not_unit_neighbour'], 'n') \
            + mk(r['increment_row'], '\\Delta')
        ev = (f"{r['eval_ref']}{mk(r['eval_ref_uncertified'], 'u')}{mk(r['eval_ref_flagged'], 'f')} / "
              f"{r['eval_other']}{mk(r['eval_other_uncertified'], 'u')}{mk(r['eval_other_flagged'], 'f')}")
        R['E.20'][r['no']] = [r['no'], desc, ev, tex_num(r['delta_gross_kEUR']), tex_num(r['threshold_or_bar_kEUR']),
                              tex_num(r['multiple_gross']), tex_text(r['verdict_gross']), tex_num(r['delta_net_kEUR']),
                              tex_num(r['multiple_net']), tex_text(r['verdict_net'])]
        M['E.20'][r['no']] = {'desc': sorted((['b'] * len([x for x in r['over_budget_plans'].split('; ') if x])
                                              + (['t'] if r['bitwise_twin_row'] == 'yes' else [])
                                              + (['n'] if r['not_unit_neighbour'] == 'yes' else [])
                                              + (['Delta'] if r['increment_row'] == 'yes' else []))),
                              'over_budget': r['over_budget'], 'plans': r['over_budget_plans']}
    for r in E['E21']:
        st = tex_text(r['status']) + mk(r['flagged'], 'f') + mk(r['bitwise_twin'], 't') + mk(r['superseded'], 's')
        R['E.21'][r['no']] = [r['no'], tex_text(r['plan']), tex_num(r['year']), tex_text(r['condition']), st,
                              tex_num(r['k0']), tex_num(r['kstar_or_end']), tex_num(r['range_over_tau']),
                              tex_num(r['band_kEUR']), tex_num(r['gap_kEUR']), tex_num(r['slack_kEUR']),
                              tex_num(r['non_clean_cycles_after_k0']), tex_num(r['replay_bitwise_through']),
                              tex_text(r['rule'])]
        M['E.21'][r['no']] = {'plan': ['b'] if r['over_budget'] == 'yes' else [],
                              'status': sorted([m for m, c in (('f', 'flagged'), ('t', 'bitwise_twin'), ('s', 'superseded'))
                                                if r[c] == 'yes'])}
    for r in E['E22']:
        R['E.22'][r['eval']] = [tex_text(r['nodes']), tex_num(r['year']), r['eval'], tex_num(r['delta_kEUR']),
                                tex_num(r['threshold_kEUR']), tex_num(r['multiple'])]
    for r in E['E23']:
        start = tex_text(r['start']) + mk(r['reported_start'], 'b')
        key = f"{r['arrangement']}|{tex_text(r['start'])}"
        R['E.23'][key] = [r['arrangement'], start, tex_num(r['Q_kEUR'], thousands=True), tex_num(r['nrf_exceedances_n']),
                          tex_num(r['nrf_largest_excess_pu']), tex_num(r['consistency_pass_kEUR']),
                          tex_num(r['curtailment_TN_GWh']), tex_num(r['curtailment_DN_GWh'])]
        M['E.23'][key] = ['b'] if r['reported_start'] == 'yes' else []
    for r in E['E24']:
        R['E.24'][r['eval']] = [r['eval'], tex_text(r['plan']), tex_num(r['year']), tex_text(r['condition']),
                                tex_text(r['cause']), tex_num(r['gap_kEUR']), tex_num(r['share_n5_pct']),
                                tex_num(r['share_n7_pct']), tex_num(r['share_n9_pct']),
                                tex_num(r['pf_primal_residual_ratio']), tex_text(r['lapse_resets'])]
    return R, M


def synth_lines(E, R):
    """A minimal LaTeX document holding the reconstructed tables, in the layout the W180a parsers read."""
    N = {r['item']: r['value'] for r in E['notes']}
    L = []
    for tab, label, ncol, key in (('E.21', 'tab:appE_evaluations', 14, 'E21'), ('E.20', 'tab:appE_claims', 10, 'E20')):
        L.append(f'\\label{{{label}}}\\\\')
        grp = None
        for r in E[key]:
            if r['group'] != grp:
                grp = r['group']
                L.append(f'\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{({grp}) group}}}}\\\\[1pt]')
            L.append(' & '.join(R[tab][r['no']]) + ' \\\\')
        L.append(f'\\multicolumn{{{ncol}}}{{@{{}}p{{\\linewidth}}@{{}}}}{{\\textit{{Note:}} reconstructed}}\\\\')
        L.append('\\end{longtable}}')
    L += ['\\label{tab:appE_neighbours}', '\\midrule'] + [' & '.join(c) + ' \\\\' for c in R['E.22'].values()]
    L += ['\\bottomrule', '\\end{tabular}',
          (f"\\textit{{Note:}} The superseded first certificate of the 2025 node-5 neighbour (evaluation~16) gave "
           f"{N['eval16_superseded_delta_kEUR']}~k\\euro{{}} ({N['eval16_superseded_multiple']}$\\times$).")]
    L += ['\\label{tab:appE_benchmark}', '\\midrule'] + [' & '.join(c) + ' \\\\' for c in R['E.23'].values()]
    L += ['\\bottomrule', '\\end{tabular}',
          (f"\\textit{{Note:}} spread over the starts {N['spread_passive_kEUR']}~k\\euro{{}} (passive) and "
           f"{N['spread_price_taker_kEUR']}~k\\euro{{}} (price-taker); reproducibility band of the coordinated evaluation "
           f"{N['coordinated_reproducibility_band_kEUR']}~k\\euro{{}}. Without any interface rule, from the cold start, the "
           f"TN cannot accept the DN exchange in {N['sweep_passive_blocks_tn_cannot_accept']} of 12 blocks "
           f"({N['sweep_passive_hours']}~h; {N['sweep_passive_failing_blocks']}) under the passive arrangement and in "
           f"{N['sweep_price_taker_blocks_tn_cannot_accept']} of 12 ({N['sweep_price_taker_hours']}~h; "
           f"{N['sweep_price_taker_failing_blocks'].replace('; ', ', ')}) under the price-taker arrangement.")]
    L += ['\\label{tab:appE_gap}', '\\midrule'] + [' & '.join(c) + ' \\\\' for c in R['E.24'].values()]
    L += ['\\bottomrule', '\\end{tabular}']
    return L


def frozen_ctx():
    """W183h's frozen_ctx (p515_s53_w183h_round8.py), restated: the frozen JSON, every W180a input and the conditions."""
    J = {k: A._json(v) for k, v in A.INPUTS.items() if v.endswith('.json')}
    T = J['FROZEN_JSON']['tables']
    ALL = dict(T['cells'])
    ALL.update(T['cells_appended_w160'])
    for k, c in ALL.items():
        c.setdefault('cell', k)
    return J, T, ALL, A.cell_conditions(J, ALL)


def v1_frozen(E, FZ):
    J, T, ALL, COND = FZ
    R, _M = recon_E(E)
    res = A.table_check(synth_lines(E, R), ALL, COND, J, T)
    tot = {'entries': sum(s['entries'] for s in res['summary'].values()),
           'MATCH': sum(s['MATCH'] for s in res['summary'].values()),
           'MISMATCH': sum(s['MISMATCH'] for s in res['summary'].values()),
           'not_frozen_agrees': sum(s['NOT_FROZEN_AGREES'] for s in res['summary'].values()),
           'not_frozen_disagrees': sum(s['NOT_FROZEN_DISAGREES'] for s in res['summary'].values())}
    idmap_ok = {int(r['no']): r['cell_id'] for r in E['E21']} == res['idmap']
    claims_ok = [r['claim_id'] for r in E['E20']] == [x['claim'] for x in res['E20']]
    ids22 = all(r['cell_id'] == res['idmap'].get(int(r['eval'])) for r in E['E22'])
    ids24 = all(r['cell_id'] == res['idmap'].get(int(r['eval'])) for r in E['E24'])
    ids20 = all(r['eval_ref_cell'] == res['idmap'].get(int(r['eval_ref'])) and
                r['eval_other_cell'] == res['idmap'].get(int(r['eval_other'])) for r in E['E20'])
    keys21 = all(ALL[r['cell_id']]['eval_key'].startswith(r['eval_key']) and
                 ALL[r['cell_id']]['candidate_key'].startswith(r['candidate_key']) and r['eval_key'] and r['candidate_key']
                 for r in E['E21'])
    # the four E.20 note numbers (not part of the W180a count)
    notes = {r['item']: r['value'] for r in E['notes']}
    claim = {c['claim_id']: c for c in T['claims']}
    row = {r['no']: r for r in E['E20']}
    n20 = []
    for no, pre in (('3', 'row3'), ('26', 'row26')):
        uf = claim[row[no]['claim_id']]['uncertified_form_beside_A64_ruling4']
        a1, v1 = A.at_print(notes[f'{pre}_uncertified_form_bar_kEUR'], uf['bar'], 1000.0)
        a2, v2 = A.at_print(notes[f'{pre}_uncertified_form_multiple'], uf['multiple_Q'])
        n20.append({'row': no, 'claim': row[no]['claim_id'], 'printed': [notes[f'{pre}_uncertified_form_bar_kEUR'],
                    notes[f'{pre}_uncertified_form_multiple']], 'frozen': [v1, v2], 'verdict': uf['verdict'],
                    'status': 'MATCH' if a1 and a2 and uf['verdict'] == 'determinate' else 'MISMATCH'})
    return {'summary': res['summary'], 'totals': tot, 'bad': res['bad'], 'mapping_problems': res['mapping_problems'],
            'idmap_equals_cell_id_column': idmap_ok, 'claim_ids_equal_check_mapping': claims_ok,
            'e20_eval_cells_equal_mapping': ids20, 'e22_cell_ids_equal_mapping': ids22, 'e24_cell_ids_equal_mapping': ids24,
            'e21_instance_keys_are_frozen_key_prefixes': keys21, 'e20_note_numbers': n20}


def v1_holds(v):
    t = v['totals']
    return (t['entries'] == EXPECTED['entries'] and t['MATCH'] == EXPECTED['MATCH'] and t['MISMATCH'] == 0
            and t['not_frozen_agrees'] == EXPECTED['not_frozen_agrees'] and t['not_frozen_disagrees'] == 0
            and not v['mapping_problems'] and {k: s['entries'] for k, s in v['summary'].items()} == EXPECTED['per_table']
            and v['idmap_equals_cell_id_column'] and v['claim_ids_equal_check_mapping'] and v['e20_eval_cells_equal_mapping']
            and v['e22_cell_ids_equal_mapping'] and v['e24_cell_ids_equal_mapping']
            and v['e21_instance_keys_are_frozen_key_prefixes'] and all(x['status'] == 'MATCH' for x in v['e20_note_numbers']))


# ======================================================================================================================
#  V2 / V3: against the manuscript and W182d
# ======================================================================================================================
def v2_manuscript(E, main_lines):
    R, M = recon_E(E)
    raw = A.raw_tables(main_lines)
    n_cells, bad, mark_bad = 0, [], []
    for t in ('E.20', 'E.21', 'E.22', 'E.23', 'E.24'):
        mrows = raw[t]['rows']
        if list(R[t]) != list(mrows):
            bad.append({'table': t, 'row_keys_supplement': list(R[t]), 'row_keys_main': list(mrows)})
            continue
        for k, cells in R[t].items():
            mc = mrows[k]['cells']
            if len(mc) != len(cells):
                bad.append({'table': t, 'row': k, 'n_cells': [len(cells), len(mc)]})
                continue
            for j, (s, m) in enumerate(zip(cells, mc)):
                if t == 'E.23' and j == 0 and m == '':
                    continue      # the printed arrangement label is not repeated; the row key carries it
                n_cells += 1
                if A.strip_markers(s)[0] != A.strip_markers(m)[0]:
                    bad.append({'table': t, 'row': k, 'col': j, 'supplement': s, 'main': m})
            # marks
            if t == 'E.20':
                got = sorted(A.strip_markers(mc[1])[1])
                frags = re.findall(r'([0-9.]+/[0-9.]+)\$\^\{b\}\$', mc[1])
                exp_plans = [x.split(': ', 1)[1] for x in M[t][k]['plans'].split('; ') if x]
                if got != M[t][k]['desc'] or frags != exp_plans or (M[t][k]['over_budget'] == 'yes') != bool(frags):
                    mark_bad.append({'table': t, 'row': k, 'main_marks': got, 'supplement_marks': M[t][k]['desc'],
                                     'main_b_plans': frags, 'supplement_b_plans': exp_plans})
                ev_m = sorted(A.strip_markers(mc[2].split('/')[0])[1]), sorted(A.strip_markers(mc[2].split('/')[1])[1])
                ev_s = sorted(A.strip_markers(cells[2].split('/')[0])[1]), sorted(A.strip_markers(cells[2].split('/')[1])[1])
                if ev_m != ev_s:
                    mark_bad.append({'table': t, 'row': k, 'eval_marks_main': ev_m, 'eval_marks_supplement': ev_s})
            elif t == 'E.21':
                gp, gs = sorted(A.strip_markers(mc[1])[1]), sorted(A.strip_markers(mc[4])[1])
                if gp != M[t][k]['plan'] or gs != M[t][k]['status']:
                    mark_bad.append({'table': t, 'row': k, 'main': [gp, gs], 'supplement': [M[t][k]['plan'], M[t][k]['status']]})
            elif t == 'E.23':
                if sorted(A.strip_markers(mc[1])[1]) != M[t][k]:
                    mark_bad.append({'table': t, 'row': k, 'main': A.strip_markers(mc[1])[1], 'supplement': M[t][k]})
            elif any(A.strip_markers(c)[1] for c in mc):
                mark_bad.append({'table': t, 'row': k, 'main_has_marks': True})
    # E.23: the printed arrangement labels, filled down, equal the supplement's arrangement column
    arm, labels = '', []
    for k in raw['E.23']['rows']:
        c0 = raw['E.23']['rows'][k]['cells'][0]
        arm = c0 or arm
        labels.append(arm)
    arm_ok = labels == [r['arrangement'] for r in E['E23']]
    declared = [int(r['eval']) for r in E['E24'] if r['over_budget'] == 'yes']
    return {'cells_compared': n_cells, 'cell_differences': bad, 'mark_differences': mark_bad,
            'e23_arrangement_fill_down_equal': arm_ok,
            'e24_over_budget_declared_difference': {'evals_flagged_in_supplement': declared,
                                                    'expected_W183h': E24_DECLARED_OVER_BUDGET,
                                                    'printed_E24_has_no_budget_mark': True,
                                                    'as_expected': declared == E24_DECLARED_OVER_BUDGET}}


def v2_holds(v):
    return (not v['cell_differences'] and not v['mark_differences'] and v['e23_arrangement_fill_down_equal']
            and v['e24_over_budget_declared_difference']['as_expected'])


def v3_budget(E, w182d):
    ob = {e['eval'] for e in w182d['A8_1_budget']['over_budget_evaluations']}
    e21 = sorted(int(r['no']) for r in E['E21'] if r['over_budget'] == 'yes')
    e24 = sorted(int(r['eval']) for r in E['E24'] if r['over_budget'] == 'yes')
    e24_exp = sorted(int(r['eval']) for r in E['E24'] if int(r['eval']) in ob)
    e20 = []
    for r in E['E20']:
        n_ob = len({e for e in (int(r['eval_ref']), int(r['eval_other'])) if e in ob})
        n_pl = len([x for x in r['over_budget_plans'].split('; ') if x])
        if (r['over_budget'] == 'yes') != (n_ob > 0) or n_pl != n_ob:
            e20.append(r['no'])
    return {'w182d_over_budget_evals': sorted(ob), 'e21_flagged': e21, 'e21_equal': e21 == sorted(ob),
            'e24_flagged': e24, 'e24_equal': e24 == e24_exp, 'e20_rows_inconsistent': e20,
            'holds': e21 == sorted(ob) and e24 == e24_exp and not e20}


# ======================================================================================================================
#  V5: Appendices B-D against the case files (independent reader)
# ======================================================================================================================
def xlsx_rows(rel, sheet):
    wb = openpyxl.load_workbook(os.path.join(REPO, rel), read_only=True, data_only=True)
    rows = [r for r in wb[sheet].iter_rows(values_only=True)]
    wb.close()
    hdr = list(rows[0])
    while hdr and hdr[-1] is None:
        hdr.pop()
    body = [list(r[:len(hdr)]) for r in rows[1:] if any(v is not None for v in r)]
    return hdr, body, sum(1 for r in rows[1:] if not any(v is not None for v in r))


def jload(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        return json.load(h)


class Cmp:
    def __init__(self):
        self.n, self.bad, self.rows = 0, [], 0

    def num(self, where, s, v):
        self.n += 1
        try:
            ok = (s == '' and v is None) or (v is not None and not isinstance(v, bool) and float(s) == float(v))
        except ValueError:
            ok = False
        if not ok:
            self.bad.append({'where': where, 'supplement': s, 'source': v})

    def txt(self, where, s, v):
        self.n += 1
        if s != v:
            self.bad.append({'where': where, 'supplement': s, 'source': v})


def v5_case_files(BD):
    C = Cmp()
    srcs = {}

    def profile(fname, rows, rel, sheet, n_keys, lead, filt=None):
        hdr, body, n_blank = xlsx_rows(rel, sheet)
        srcs[f'{rel}:{sheet}'] = {'rows': len(body), 'blank_rows_skipped': n_blank}
        if [int(h) for h in hdr[n_keys:]] != list(range(24)):
            C.bad.append({'where': f'{rel}:{sheet}', 'header': hdr})
            return
        if len(rows) != len(body):
            C.bad.append({'where': f'{fname} vs {rel}:{sheet}', 'rows': [len(rows), len(body)]})
            return
        for i, (r, b) in enumerate(zip(rows, body)):
            C.rows += 1
            for j, key in enumerate(lead):
                v = b[j]
                if isinstance(v, str):
                    C.txt(f'{fname} row {i + 1} {key}', r[key], v)
                else:
                    C.num(f'{fname} row {i + 1} {key}', r[key], v)
            for h in range(24):
                C.num(f'{fname} row {i + 1} h{h:02d}', r[f'h{h:02d}'], b[n_keys + h])

    mk = os.path.join(DATA, 'MarketData', 'SRP1_market_data.xlsx')
    B = BD['B_market_prices.csv']
    for price, sheet in (('energy', 'Energy'), ('flexibility', 'Flexibility')):
        rows = [r for r in B if r['price'] == price]
        profile('B_market_prices.csv', rows, mk, sheet, 2, ['season', 'scenario'])
    if {r['price'] for r in B} != {'energy', 'flexibility'}:
        C.bad.append({'where': 'B_market_prices.csv', 'prices': sorted({r['price'] for r in B})})
    profile('C_tn_res_profiles.csv', BD['C_tn_res_profiles.csv'], os.path.join(DATA, 'case9', 'case9_operational_data.xlsx'),
            'Pg', 3, ['gen_type', 'season', 'scenario'])

    def res_gens(fname, rows, net):
        js = {y: jload(os.path.join(DATA, net, f'{net}_{y}.json')) for y in PRINTED_YEARS}
        src_ids = [g['gen_id'] for g in js[2025]['generators'] if g['type'] in ('WIND', 'PV')]
        if [int(r['generator_id']) for r in rows] != src_ids:
            C.bad.append({'where': f'{fname} {net}', 'ids': [[r['generator_id'] for r in rows], src_ids]})
            return
        for r in rows:
            C.rows += 1
            for y in PRINTED_YEARS:
                g = next(g for g in js[y]['generators'] if g['gen_id'] == int(r['generator_id']))
                C.num(f'{fname} {net} gen {r["generator_id"]} {y}', r[f'capacity_{y}_MW'], g['Pmax'])
                C.num(f'{fname} {net} gen {r["generator_id"]} node {y}', r['node_id'], g['bus'])
                C.txt(f'{fname} {net} gen {r["generator_id"]} type {y}', TYPE_CODE.get(r['type']), g['type'])
    res_gens('C_tn_res_generators.csv', BD['C_tn_res_generators.csv'], 'case9')

    j = {n: jload(os.path.join(DATA, n, f'{n}_2025.json')) for n, _k in ADNS}
    a1 = j['case33_1']
    if [int(r['bus_id']) for r in BD['D_ieee33_buses.csv']] != [b['bus_i'] for b in a1['nodes']]:
        C.bad.append({'where': 'D_ieee33_buses.csv', 'ids': 'differ'})
    else:
        for r, b in zip(BD['D_ieee33_buses.csv'], a1['nodes']):
            C.rows += 1
            for col, key in (('bus_type_code', 'type'), ('g_pu', 'Gs'), ('b_pu', 'Bs'), ('v_base_kV', 'baseKV'),
                             ('v_max_pu', 'Vmax'), ('v_min_pu', 'Vmin')):
                C.num(f'D_ieee33_buses.csv bus {b["bus_i"]} {col}', r[col], b[key])
            C.num(f'D_ieee33_buses.csv bus {b["bus_i"]} bus_type', str(BUS_LABEL.get(r['bus_type'])), b['type'])
            for n, _k in ADNS[1:]:
                if j[n]['nodes'] != a1['nodes']:
                    C.bad.append({'where': f'{n} nodes differ from case33_1'})
    brs = {n: {x['branch_id']: (e, x) for e in ('transformers', 'lines') for x in j[n][e]} for n, _k in ADNS}
    if [int(r['branch_id']) for r in BD['D_ieee33_branches.csv']] != sorted(brs['case33_1']):
        C.bad.append({'where': 'D_ieee33_branches.csv', 'ids': 'differ'})
    else:
        for r in BD['D_ieee33_branches.csv']:
            C.rows += 1
            e, x = brs['case33_1'][int(r['branch_id'])]
            C.txt(f'branch {r["branch_id"]} element', r['element'], {'transformers': 'transformer', 'lines': 'line'}[e])
            for col, key in (('from_bus', 'fbus'), ('to_bus', 'tbus'), ('r_pu', 'r'), ('x_pu', 'x'), ('b_sh_pu', 'b')):
                for n, _k in ADNS:
                    C.num(f'branch {r["branch_id"]} {col} {n}', r[col], brs[n][int(r['branch_id'])][1][key])
            for n, k in ADNS:
                C.num(f'branch {r["branch_id"]} rating {n}', r[f'rating_adn_node{k}_MVA'], brs[n][int(r['branch_id'])][1]['rating'])
    for r, (n, k) in zip(BD['D_ieee33_tn_connection.csv'], ADNS):
        C.rows += 1
        g = [g for g in j[n]['generators'] if g['type'] == 'REF']
        C.txt(f'tn_connection {n}', (r['adn_node'], r['network']), (str(k), n))
        if len(g) != 1:
            C.bad.append({'where': f'{n}: {len(g)} REF generators'})
            continue
        for col, key in (('generator_id', 'gen_id'), ('node_id', 'bus'), ('p_max_MW', 'Pmax'), ('p_min_MW', 'Pmin'),
                         ('q_max_MVAr', 'Qmax'), ('q_min_MVAr', 'Qmin'), ('v_set_pu', 'Vg')):
            C.num(f'tn_connection {n} {col}', r[col], g[0][key])
    if len(BD['D_ieee33_tn_connection.csv']) != 3:
        C.bad.append({'where': 'D_ieee33_tn_connection.csv rows'})
    chars = {}
    for n, k in ADNS:
        hdr, body, _nb = xlsx_rows(os.path.join(DATA, n, f'{n}_operational_data.xlsx'), 'Characterization')
        if hdr[:4] != ['LoadID', 'Pc', 'Qc', 'FL Ratio']:
            C.bad.append({'where': f'{n} Characterization header {hdr}'})
        chars[k] = {int(b[0]): b for b in body}
    loads = {ld['load_id']: ld for ld in a1['loads']}
    if [int(r['load_id']) for r in BD['D_ieee33_loads.csv']] != [ld['load_id'] for ld in a1['loads']] \
            or sorted(loads) != sorted(chars[5]):
        C.bad.append({'where': 'D_ieee33_loads.csv ids'})
    else:
        for r in BD['D_ieee33_loads.csv']:
            C.rows += 1
            lid = int(r['load_id'])
            C.num(f'load {lid} bus', r['bus_id'], loads[lid]['bus'])
            for n, k in ADNS:
                C.num(f'load {lid} fl_ratio {n}', r['fl_ratio'], chars[k][lid][3])
                C.num(f'load {lid} pc {n}', r[f'pc_base_adn_node{k}_MW'], chars[k][lid][1])
                C.num(f'load {lid} qc {n}', r[f'qc_base_adn_node{k}_MVAr'], chars[k][lid][2])
    for n, k in ADNS:
        res_gens('D_adn_res_generators.csv', [r for r in BD['D_adn_res_generators.csv'] if r['adn_node'] == str(k)
                                              and r['network'] == n], n)
        xl = os.path.join(DATA, n, f'{n}_operational_data.xlsx')
        profile('D_adn_res_profiles.csv', [r for r in BD['D_adn_res_profiles.csv'] if r['adn_node'] == str(k) and r['network'] == n],
                xl, 'Pg', 3, ['gen_type', 'season', 'scenario'])
        for q in ('Pc', 'Qc'):
            profile('D_adn_load_profiles.csv', [r for r in BD['D_adn_load_profiles.csv'] if r['adn_node'] == str(k)
                                                and r['network'] == n and r['quantity'] == q],
                    xl, q, 3, ['load_id', 'season', 'scenario'])
        profile('D_adn_flexibility_profiles.csv', [r for r in BD['D_adn_flexibility_profiles.csv'] if r['adn_node'] == str(k)
                                                   and r['network'] == n], xl, 'Flex', 3, ['type', 'season', 'scenario'])
    covered = {'D_adn_res_generators.csv': len(BD['D_adn_res_generators.csv']),
               'D_adn_res_profiles.csv': len(BD['D_adn_res_profiles.csv']),
               'D_adn_load_profiles.csv': len(BD['D_adn_load_profiles.csv']),
               'D_adn_flexibility_profiles.csv': len(BD['D_adn_flexibility_profiles.csv'])}
    exp_cov = {'D_adn_res_generators.csv': 12, 'D_adn_res_profiles.csv': 72, 'D_adn_load_profiles.csv': 2304,
               'D_adn_flexibility_profiles.csv': 108}
    if covered != exp_cov:
        C.bad.append({'where': 'row coverage', 'got': covered, 'expected': exp_cov})
    return {'cells_compared': C.n, 'rows_compared': C.rows, 'differences': C.bad[:50], 'n_differences': len(C.bad),
            'xlsx_sources': srcs}


# ======================================================================================================================
#  V6: Appendices C-D against the printed tables at a87e195
# ======================================================================================================================
def printed_rows(lines, label):
    il = next(i for i, ln in enumerate(lines) if f'\\label{{{label}}}' in ln)
    i0 = max(i for i in range(il) if '\\begin{table}' in lines[i])
    out = []
    for i in range(i0, il):
        t = lines[i].strip()
        if t.endswith('\\\\') and re.match(r'^\d', t):
            out.append((i + 1, A.row_cells(t)))
    return out


def groups(cells, w):
    out, i = [], 0
    while i < len(cells):
        out.append(cells[i:i + w])
        i += w
        if i < len(cells) and cells[i] == '':
            i += 1
    return out


def v6_printed(BD, lines):
    C = Cmp()
    diffs_note = []

    def num(where, p, v):
        C.n += 1
        ok, vv = A.at_print(p.replace('\\%', ''), v)
        if not ok:
            C.bad.append({'where': where, 'printed': p, 'supplement_at_print': vv})

    def txt(where, p, v):
        C.n += 1
        if p != v:
            C.bad.append({'where': where, 'printed': p, 'supplement': v})

    def gens(label, rows):
        pr = printed_rows(lines, label)
        if len(pr) != len(rows):
            C.bad.append({'where': label, 'rows': [len(pr), len(rows)]})
            return
        for (ln, c), r in zip(pr, rows):
            txt(f'{label} l.{ln} id', c[0], r['generator_id'])
            txt(f'{label} l.{ln} node', c[1], r['node_id'])
            txt(f'{label} l.{ln} type', c[2], r['type'])
            for p, y in zip(c[3:], PRINTED_YEARS):
                num(f'{label} l.{ln} {y}', p, float(r[f'capacity_{y}_MW']))
    gens('tab:cs3_tn_res_generators', BD['C_tn_res_generators.csv'])
    for n, k in ADNS:
        gens(f'tab:cs3_adn_node_{k}_res_generators', [r for r in BD['D_adn_res_generators.csv'] if r['adn_node'] == str(k)])
    bus = {r['bus_id']: r for r in BD['D_ieee33_buses.csv']}
    seen = []
    for ln, c in printed_rows(lines, 'tab:cs1_ieee33_buses'):
        for g in groups(c, 7):
            r = bus[g[0]]
            seen.append(g[0])
            txt(f'bus {g[0]} type', g[1], r['bus_type'])
            for p, col in zip(g[2:], ('g_pu', 'b_pu', 'v_base_kV', 'v_max_pu', 'v_min_pu')):
                num(f'bus {g[0]} {col}', p, float(r[col]))
    if sorted(seen, key=int) != sorted(bus, key=int):
        C.bad.append({'where': 'buses printed set'})
    br = {r['branch_id']: r for r in BD['D_ieee33_branches.csv']}
    seen = []
    for ln, c in printed_rows(lines, 'tab:cs1_ieee33_branches'):
        for g in groups(c, 7):
            r = br[g[0]]
            seen.append(g[0])
            txt(f'branch {g[0]} from', g[1], r['from_bus'])
            txt(f'branch {g[0]} to', g[2], r['to_bus'])
            for p, col in zip(g[3:], ('r_pu', 'x_pu', 'b_sh_pu', 'rating_adn_node5_MVA')):
                num(f'branch {g[0]} {col}', p, float(r[col]))
    if sorted(seen, key=int) != sorted(br, key=int):
        C.bad.append({'where': 'branches printed set'})
    pr = printed_rows(lines, 'tab:cs1_ieee33_generators')
    tn = {r['adn_node']: r for r in BD['D_ieee33_tn_connection.csv']}
    per_adn = {}
    for k in ('5', '7', '9'):
        r = tn[k]
        ok = []
        for p, col in zip(pr[0][1][2:], ('p_max_MW', 'p_min_MW', 'q_max_MVAr', 'q_min_MVAr', 'v_set_pu')):
            ok.append(A.at_print(p, float(r[col]))[0])
        per_adn[k] = all(ok) and pr[0][1][0] == r['generator_id'] and pr[0][1][1] == r['node_id']
    for p, col in zip(pr[0][1], ('generator_id', 'node_id', 'p_max_MW', 'p_min_MW', 'q_max_MVAr', 'q_min_MVAr', 'v_set_pu')):
        if col in ('generator_id', 'node_id'):
            txt(f'tn_connection {col}', p, tn['5'][col])
        else:
            num(f'tn_connection {col} (ADN node 5)', p, float(tn['5'][col]))
    if not all(per_adn.values()):
        diffs_note.append({'table': 'tab:cs1_ieee33_generators', 'printed_row_equals_adn': per_adn,
                           'note': 'the caption calls the table common to all ADNs; the case files differ by ADN'})
    loads = {r['load_id']: r for r in BD['D_ieee33_loads.csv']}
    seen = []
    for ln, c in printed_rows(lines, 'tab:cs1_ieee33_loads'):
        for g in groups(c, 3):
            r = loads[g[0]]
            seen.append(g[0])
            txt(f'load {g[0]} bus', g[1], r['bus_id'])
            num(f'load {g[0]} FL ratio %', g[2], 100.0 * float(r['fl_ratio']))
    if sorted(seen, key=int) != sorted(loads, key=int):
        C.bad.append({'where': 'loads printed set'})
    return {'cells_compared': C.n, 'differences': C.bad, 'n_differences': len(C.bad), 'observations': diffs_note}


# ======================================================================================================================
#  V7: README placeholder
# ======================================================================================================================
def v7_readme(members):
    rd = members['README_PLACEHOLDER.md'].decode('utf-8')
    sections = re.split(r'^### `([^`]+)`$', rd, flags=re.M)
    cols = {sections[i]: re.findall(r'^\| `([^`]+)` \|', sections[i + 1], flags=re.M) for i in range(1, len(sections), 2)}
    csvs = {n: read_csv(b)[0] for n, b in members.items() if n.endswith('.csv')}
    probs = [n for n in csvs if cols.get(n) != csvs[n]] + [n for n in cols if n not in csvs]
    marks = ['[EXPERT PROSE PLACEHOLDER', '[PLACEHOLDER: tau]', '[PLACEHOLDER: objective convention]',
             '[PLACEHOLDER: certification rule parameters]',
             'Q: gross operational cost, contracted interface settlement excluded.', repr(SC6.TAU)]
    missing = [m for m in marks if m not in rd]
    return {'files_with_column_table': len(cols), 'csv_files': len(csvs), 'header_mismatches': probs,
            'placeholders_missing': missing, 'holds': not probs and not missing}


# ======================================================================================================================
#  negative controls (in memory)
# ======================================================================================================================
def _bump(s, step):
    dp = len(s.split('.')[1]) if '.' in s else 0
    return f'{float(s) + step:.{dp}f}'


def controls(E, BD, FZ, main_lines, w182d, sdir):
    out = []

    def bad_keys(v):
        return sorted({(b['table'], str(b['row']), b['key']) for b in v['bad']})

    def ctl(cid, desc, fn, expect):
        E2, BD2 = copy.deepcopy(E), copy.deepcopy(BD)
        fn(E2, BD2)
        got = expect[0](E2, BD2)
        caught = got == expect[1]
        out.append({'id': cid, 'edit': desc, 'got': got, 'expected': expect[1], 'caught_exactly': caught})
        _ok(caught, f'control {cid}: {desc} -> {got}')

    def at(rows, key, val):
        return next(r for r in rows if r[key] == val)

    def v1k(E2, _BD2):
        return bad_keys(v1_frozen(E2, FZ))
    ctl('C1', 'E.21 row 1 band_kEUR +0.1', lambda E2, B2: at(E2['E21'], 'no', '1').update(
        band_kEUR=_bump(at(E2['E21'], 'no', '1')['band_kEUR'], 0.1)), (v1k, [('E.21', '1', 'band_k')]))
    ctl('C2', 'E.20 row 3 delta_gross_kEUR +0.1', lambda E2, B2: at(E2['E20'], 'no', '3').update(
        delta_gross_kEUR=_bump(at(E2['E20'], 'no', '3')['delta_gross_kEUR'], 0.1)), (v1k, [('E.20', '3', 'd_gross_k')]))
    ctl('C3', 'E.23 Passive NRF / cold Q_kEUR +0.1', lambda E2, B2: next(
        r for r in E2['E23'] if r['arrangement'] == 'Passive NRF' and r['start'] == 'cold').update(
        Q_kEUR=_bump(next(r for r in E2['E23'] if r['arrangement'] == 'Passive NRF' and r['start'] == 'cold')['Q_kEUR'], 0.1)),
        (v1k, [('E.23', 'passive/cold', 'Q_k')]))
    ctl('C4', 'E.24 eval 42 pf_primal_residual_ratio +0.001', lambda E2, B2: at(E2['E24'], 'eval', '42').update(
        pf_primal_residual_ratio=_bump(at(E2['E24'], 'eval', '42')['pf_primal_residual_ratio'], 0.001)),
        (v1k, [('E.24', '42', 'pf_primal_last')]))
    ctl('C5', 'E.22 eval 19 multiple +0.01', lambda E2, B2: at(E2['E22'], 'eval', '19').update(
        multiple=_bump(at(E2['E22'], 'eval', '19')['multiple'], 0.01)), (v1k, [('E.22', '19', 'phase_b')]))
    ctl('C6', 'E.23 note spread_passive_kEUR +0.1', lambda E2, B2: at(E2['notes'], 'item', 'spread_passive_kEUR').update(
        value=_bump(at(E2['notes'], 'item', 'spread_passive_kEUR')['value'], 0.1)), (v1k, [('E.23', 'note', 'spread_passive')]))
    ctl('C7', 'E.21 row 37 status "growth test" -> "gap clause"', lambda E2, B2: at(E2['E21'], 'no', '37').update(
        status='gap clause'), (v1k, [('E.21', '37', 'status')]))

    def v20(E2, _BD2):
        v = v1_frozen(E2, FZ)
        return [x['row'] for x in v['e20_note_numbers'] if x['status'] != 'MATCH']
    ctl('C8', 'E.20 note row3 bar +0.1', lambda E2, B2: at(E2['notes'], 'item', 'row3_uncertified_form_bar_kEUR').update(
        value=_bump(at(E2['notes'], 'item', 'row3_uncertified_form_bar_kEUR')['value'], 0.1)), (v20, ['3']))

    def vb(E2, _BD2):
        v2 = v2_manuscript(E2, main_lines)
        v3 = v3_budget(E2, w182d)
        return [[(m['table'], m['row']) for m in v2['mark_differences']], v3['e21_flagged']]
    ctl('C9', 'E.21 row 4 over_budget no -> yes', lambda E2, B2: at(E2['E21'], 'no', '4').update(over_budget='yes'),
        (vb, [[('E.21', '4')], [4, 5, 6, 9, 10, 11, 38, 39]]))

    def v2c(E2, _BD2):
        return [(b['table'], b.get('row'), b.get('col')) for b in v2_manuscript(E2, main_lines)['cell_differences']]
    ctl('C10', 'E.21 row 10 k0 +1 (against main.tex)', lambda E2, B2: at(E2['E21'], 'no', '10').update(
        k0=str(int(at(E2['E21'], 'no', '10')['k0']) + 1)), (v2c, [('E.21', '10', 5)]))

    def v5w(_E2, BD2):
        return [d['where'] for d in v5_case_files(BD2)['differences']]
    ctl('C11', 'B-D branch 2 r_pu last digit changed', lambda E2, B2: at(B2['D_ieee33_branches.csv'], 'branch_id', '2').update(
        r_pu=at(B2['D_ieee33_branches.csv'], 'branch_id', '2')['r_pu'][:-1] + '9'),
        (v5w, ['branch 2 r_pu case33_1', 'branch 2 r_pu case33_2', 'branch 2 r_pu case33_3']))
    ctl('C12', 'B-D load profile first row h05 + 1e-9', lambda E2, B2: B2['D_adn_load_profiles.csv'][0].update(
        h05=repr(float(B2['D_adn_load_profiles.csv'][0]['h05']) + 1e-9)), (v5w, ['D_adn_load_profiles.csv row 1 h05']))
    ctl('C13', 'B-D market energy row 1 h00 + 0.01', lambda E2, B2: B2['B_market_prices.csv'][0].update(
        h00=repr(float(B2['B_market_prices.csv'][0]['h00']) + 0.01)), (v5w, ['B_market_prices.csv row 1 h00']))

    def v56(_E2, BD2):
        return [[d['where'] for d in v5_case_files(BD2)['differences']],
                [d['where'] for d in v6_printed(BD2, main_lines)['differences']]]
    ctl('C14', 'B-D bus 2 v_max_pu 1.1 -> 1.2', lambda E2, B2: at(B2['D_ieee33_buses.csv'], 'bus_id', '2').update(v_max_pu='1.2'),
        (v56, [['D_ieee33_buses.csv bus 2 v_max_pu'], ['bus 2 v_max_pu']]))
    # zip: one byte flipped in the middle of the archive
    recs = [f for f in os.listdir(sdir) if re.fullmatch(r'supplement_v\d+_record\.json', f)]
    with open(os.path.join(sdir, recs[0]), encoding='utf-8') as h:
        zname = json.load(h)['zip']['name']
    with open(os.path.join(sdir, zname), 'rb') as h:
        zb = bytearray(h.read())
    zb[len(zb) // 2] ^= 0xFF
    _i, probs, _m = check_zip(sdir, bytes(zb))
    caught = any('zip sha256 differs' in p for p in probs)
    out.append({'id': 'C15', 'edit': 'one byte of the zip flipped', 'got': probs[:5], 'caught_exactly': caught})
    _ok(caught, f'control C15: zip byte flip -> {probs[:3]}')
    return out


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--supplement-dir', default=os.path.join(MR, 'supplement'))
    ap.add_argument('--out-dir', help='where the result JSON goes (default: the supplement directory)')
    args = ap.parse_args()
    sdir = os.path.join(REPO, args.supplement_dir) if not os.path.isabs(args.supplement_dir) else args.supplement_dir
    odir = args.out_dir or sdir
    # V0
    zinfo, zprobs, members = check_zip(sdir)
    version = zinfo['record_version']
    out_json = os.path.join(odir, f'verify_supplement_v{version}_result.json')
    if os.path.exists(out_json):
        _log(f'refused: {out_json} exists (never overwritten)')
        return 1
    _ok(not zprobs, f'V0 zip {zinfo["zip"]} sha256 {zinfo["zip_sha256"]}: {len(members)} entries, inventories equal '
                    f'{zprobs[:3]}')
    E = {}
    for key, name in (('E20', 'E20_comparisons'), ('E21', 'E21_evaluations'), ('E22', 'E22_neighbours'),
                      ('E23', 'E23_benchmark'), ('E24', 'E24_gap_refused'), ('notes', 'E_table_notes')):
        hdr, rows, widths = read_csv(members[f'appendix_E/{name}.csv'])
        if any(w != len(hdr) for w in widths):
            raise RuntimeError(f'{name}: ragged rows')
        E[key] = rows
    BD = {n.split('/', 1)[1]: read_csv(b)[1] for n, b in members.items() if n.startswith('appendices_B_D/')}
    # inputs
    main_b = subprocess.run(['git', '-C', CLONE, 'show', f'{MAIN_REV}:main.tex'], capture_output=True).stdout
    with open(os.path.join(REPO, MAIN_REVIEW_COPY), 'rb') as h:
        review_b = h.read()
    _ok(_sha(main_b) == MAIN_SHA and review_b == main_b,
        f'main.tex at {MAIN_REV[:7]} (clone, git show) sha256 {_sha(main_b)[:8]} = {MAIN_REVIEW_COPY}')
    main_lines = main_b.decode('utf-8').split('\n')
    _ok(A._sha(A.FZ_REL) == A.FZ_SHA, f'frozen tables {A.FZ_REL} sha256 {A.FZ_SHA[:8]}')
    FZ = frozen_ctx()
    w182d = jload(W182D)
    # V1
    v1 = v1_frozen(E, FZ)
    t = v1['totals']
    _ok(v1_holds(v1), f'V1 Appendix E vs frozen tables: {t["entries"]} entries, {t["MATCH"]} MATCH, {t["MISMATCH"]} '
                      f'MISMATCH, {t["not_frozen_agrees"]} not-frozen agree, {t["not_frozen_disagrees"]} disagree, '
                      f'{len(v1["mapping_problems"])} mapping problems; per table '
                      f'{ {k: s["entries"] for k, s in v1["summary"].items()} }; E.20 note numbers '
                      f'{[x["status"] for x in v1["e20_note_numbers"]]}')
    # V2 / V3
    v2 = v2_manuscript(E, main_lines)
    _ok(v2_holds(v2), f'V2 Appendix E vs main.tex a87e195: {v2["cells_compared"]} cells, '
                      f'{len(v2["cell_differences"])} differ, {len(v2["mark_differences"])} mark differences; E.24 '
                      f'over-budget declared difference {v2["e24_over_budget_declared_difference"]["evals_flagged_in_supplement"]}')
    v3 = v3_budget(E, w182d)
    _ok(v3['holds'], f'V3 over_budget vs W182d {v3["w182d_over_budget_evals"]}: E.21 {v3["e21_flagged"]}, E.24 '
                     f'{v3["e24_flagged"]}, E.20 inconsistent rows {v3["e20_rows_inconsistent"]}')
    # V4
    tmpd = tempfile.mkdtemp(prefix='w188b_verify_')
    gen_b, gen_out = A.run_generator(os.path.join(tmpd, 'appE.tex'))
    os.remove(os.path.join(tmpd, 'appE.tex'))
    os.rmdir(tmpd)
    with open(os.path.join(REPO, A.APPX_REL), 'rb') as h:
        committed = h.read()
    occ = A.locate_bytes(main_b, gen_b)
    ver = A.run_verifier()
    v4 = {'gen_stdout': gen_out.strip(), 'generated_sha256': _sha(gen_b), 'equals_committed': gen_b == committed,
          'occurrences_in_main': len(occ), 'line_aligned': bool(occ) and occ[0]['starts_at_line_start'] and occ[0]['ends_at_line_end'],
          'main_lines': [occ[0]['start_line'], occ[0]['end_line']] if occ else None,
          'verify_appE': {'pass': ver['n_pass'], 'fail': ver['n_fail'], 'exception': ver['exception']}}
    _ok(v4['equals_committed'] and len(occ) == 1 and v4['line_aligned'] and ver['n_pass'] == 33 and ver['n_fail'] == 0
        and not ver['exception'], f'V4 gen_appE.py output = committed appendix, once in main.tex l. {v4["main_lines"]}; '
                                  f'verify_appE.py {ver["n_pass"]} PASS {ver["n_fail"]} FAIL')
    # V5 / V6
    v5 = v5_case_files(BD)
    _ok(v5['n_differences'] == 0, f'V5 Appendices B-D vs case files: {v5["cells_compared"]} cells in {v5["rows_compared"]} '
                                  f'rows, {v5["n_differences"]} differ {v5["differences"][:3]}')
    v6 = v6_printed(BD, main_lines)
    _ok(v6['n_differences'] == 0, f'V6 Appendices C-D vs printed tables at a87e195: {v6["cells_compared"]} cells, '
                                  f'{v6["n_differences"]} differ {v6["differences"][:3]}; observations {v6["observations"]}')
    # V7
    v7 = v7_readme(members)
    _ok(v7['holds'], f'V7 README placeholder: {v7["files_with_column_table"]} column tables = {v7["csv_files"]} CSV '
                     f'headers; placeholders missing {v7["placeholders_missing"]}')
    # controls
    ctl = controls(E, BD, FZ, main_lines, w182d, sdir)
    # guards
    g_fail = GUARD.verify(0) + A.GUARD.verify(0)
    _ok(not g_fail and not any(PICKLE_COUNTS.values()) and not any(A.PICKLE_COUNTS.values()),
        f'guards: {GUARD.counts} / W180a {A.GUARD.counts}; pickle {PICKLE_COUNTS} / {A.PICKLE_COUNTS}')
    result = {'stage': 'W188 (b) supplement verifier', 'supplement_dir': os.path.relpath(sdir, REPO),
              'zip': zinfo, 'zip_problems': zprobs, 'V1_frozen': {k: v for k, v in v1.items()},
              'V2_manuscript': v2, 'V3_budget': v3, 'V4_generator': v4, 'V5_case_files': v5, 'V6_printed': v6,
              'V7_readme': v7, 'controls': ctl,
              'guards': {'verifier': dict(GUARD.counts), 'w180a_module': dict(A.GUARD.counts), 'verify_0': g_fail,
                         'pickle': dict(PICKLE_COUNTS), 'pickle_w180a': dict(A.PICKLE_COUNTS)},
              'verifier_sha256': A._sha(os.path.join(MR, 'verify_supplement.py')),
              'git_head': A._git('rev-parse', 'HEAD'), 'failures': list(FAILS), 'verdict_pass': not FAILS}
    os.makedirs(odir, exist_ok=True)
    with open(out_json, 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(result, indent=1, sort_keys=True) + '\n')
    _log(f'wrote {out_json}')
    _log(f'VERDICT: {"PASS" if not FAILS else "FAIL"} ({len(FAILS)} failures)')
    return 0 if not FAILS else 3


if __name__ == '__main__':
    sys.exit(main())
