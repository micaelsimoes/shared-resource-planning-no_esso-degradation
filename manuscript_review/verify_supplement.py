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
     and found exactly once, line-aligned, in main.tex at a87e195; verify_appE.py: 40 PASS, 0 FAIL (33 before 441722c9).
  V5 Appendices B-D against the CASE FILES, read here with an independent reader (openpyxl for the xlsx, json for the
     network files): every cell equal (numbers exactly, as floats), every source row present.
  V6 Appendices B-D against the printed tables of Appendices C-D in main.tex at a87e195, at the printed rounding.
  V7 the README: per file, its column table = the CSV header; tau. Version 1: README_PLACEHOLDER.md with its marked
     placeholders. Version >= 2 (Addendum 90 ruling 7 (iv)): README.md begins with the committed expert prose
     manuscript_review/supplement/supplement_README_prose_v1.md (v2; prose _v2.md for v3, Addendum 91 ruling 3;
     prose _v3.md from v4 on, v5 included, Addendum 91 amendment) byte for byte, its sha256 and path = the record's, no
     README_PLACEHOLDER.md entry and no placeholder mark anywhere.
  V8 (version >= 2; Addendum 90 ruling 7 (iii)) the RES-capacity CSVs reproduce the paper's Table 2
     (tab:res_capacity_summary, main.tex at a87e195 and the round-9b review copy): per network and technology, the sum
     of capacity_2025/2030/2035 equals the printed aggregate; and the sentence beside it (2025 also in 2028, 2030 in
     2031 and 2034, 2035 in 2037) holds generator by generator.
  V9 (version >= 2) every entry against the predecessor zip named in the record (its sha256 checked): the entries that
     differ must be exactly the declared ones (v2: README, the two RES CSVs, SHA256SUMS.txt; v3 and v4: README.md and
     SHA256SUMS.txt only; v5 likewise, V9_DECLARED_BY_VERSION); each RES CSV restricted to
     the predecessor's columns equals the predecessor's bytes; every other entry byte-identical (D_ieee33_tn_connection
     .csv named explicitly).
  V5 checks every capacity column of the RES CSVs (v2: all seven case years) against the case files, and that the
  capacity columns are exactly the expected years.
  V10 (version >= 4; Addendum 91 amendment) the names in the generated half of README.md (after the prose): none of
     the a87e195 names (Table(s) E.2x, Appendix E., Fig. E.19, Appendix B (, Appendix C, Appendix D, Appendices C),
     CSV file-name identifiers (E20_E21_groups.csv, appendices_B_D/, appendix_E/) masked first; and README.md equals
     the predecessor's README with its prose replaced by this version's, its version heading replaced, and the
     amendment's mapping (V10_NAME_MAP, written here from the amendment) applied to its generated half.
  V11 (version >= 4) the names against the round-9c manuscript (main.tex at Overleaf 3c379fa, git show in the clone,
     read-only): a static count of sections, subsections, figures and tables (no TeX on this machine; elsarticle's
     \appendix keeps the figure and table counters and numbers them <letter>.<n>) gives the number each label
     carries; Table B.10 / B.11, Fig. B.9 and Appendix B.2-B.4 must be the numbers of tab:appE_claims,
     tab:appE_neighbours, fig:investment_lattice and app:results_claims / _neighbours / _statistics; every A.n / B.n
     name in README.md must exist with its kind; S1-S6 must each be referenced in main.tex, and README.md names no
     other S-number. Version >= 5: also Table 2 = tab:res_capacity_summary (the new wording names it).
  Version 5 (W189a-v5, Amendment 2 to Addendum 91, ruling 1): V7 reads prose v3 (as v4); V9 declares README.md and
     SHA256SUMS.txt only; V10 rebuilds v5's README from v4's with the heading and the ruling-1 edits
     (V10_EDITS_BY_VERSION[5], written here from the ruling); V12 (new, version >= 5): no '(fig:', '(tab:', '(app:',
     bare label, 'Appendix E' or 'Appendix B-D' in the generated half (CSV identifiers masked), the V10 stale patterns
     0, and no punctuation artefact beyond the predecessor's. Controls C23-C25 (v >= 5) are V12's negative controls.
  Negative controls, in memory only: an altered cell of each kind must be caught where it was altered.
Guards: SolveProfileGuard(permitted=()) before any project import, pickle.load / loads blocked; both verified at 0.
Outputs (opened 'x', never overwritten): verify_supplement_v<N>_result.json (gate_result_io) and a summary on stdout.
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u manuscript_review/verify_supplement.py \\
        > manuscript_review/supplement/verify_launch.log 2>&1
    version 2: add --supplement-dir manuscript_review/supplement_v2 and write supplement_v2/verify_launch.log
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
CAP_YEARS_BY_VERSION = {1: PRINTED_YEARS}          # version >= 2: every case year (Addendum 90 ruling 7 (iii))
CAP_YEARS_V2 = (2025, 2028, 2030, 2031, 2034, 2035, 2037)
TABLE2_LABEL = 'tab:res_capacity_summary'
TABLE2_YEARS = ('2025', '2030', '2035')
TABLE2_SENTENCE = ('the 2025 capacities also apply in 2028, the 2030 capacities apply in 2031 and 2034, and the 2035 '
                   'capacities apply in 2037')
ROUND9B_COPY = os.path.join(MR, 'main_a87e195_round9b.tex')
README_PROSE_REL = os.path.join(MR, 'supplement', 'supplement_README_prose_v1.md')
README_PROSE_V2_REL = os.path.join(MR, 'supplement', 'supplement_README_prose_v2.md')   # Addendum 91 ruling 3
README_PROSE_V3_REL = os.path.join(MR, 'supplement', 'supplement_README_prose_v3.md')   # Addendum 91 amendment
V9_DECLARED = {'changed': ['SHA256SUMS.txt', 'appendices_B_D/C_tn_res_generators.csv',
                           'appendices_B_D/D_adn_res_generators.csv'],
               'added': ['README.md'], 'removed': ['README_PLACEHOLDER.md']}
# V9's declared difference per version (against its predecessor) and the RES columns it adds. v3 (W189a-v3, Addendum
# 91 ruling 3): only the README prose changes, so README.md and SHA256SUMS.txt differ and the RES CSVs are identical.
# v4 (W189a-v4, Addendum 91 amendment): README prose v3 and the round-9c names in the generated descriptions; no CSV
# changes, so again README.md and SHA256SUMS.txt only.
# v5 (W189a-v5, Amendment 2 to Addendum 91, ruling 1): the description wording only, again README.md and SHA256SUMS.txt.
V9_DECLARED_BY_VERSION = {2: V9_DECLARED, 3: {'changed': ['README.md', 'SHA256SUMS.txt'], 'added': [], 'removed': []},
                          4: {'changed': ['README.md', 'SHA256SUMS.txt'], 'added': [], 'removed': []},
                          5: {'changed': ['README.md', 'SHA256SUMS.txt'], 'added': [], 'removed': []}}
V9_RES_COLUMNS_ADDED_BY_VERSION = {2: ['capacity_2030_MW', 'capacity_2035_MW'], 3: [], 4: [], 5: []}
# V10: the Addendum 91 amendment's mapping, written here from the amendment's table (not imported from the generator),
# applied in this order to the predecessor's generated README half; the two plural phrases first.
V10_NAME_MAP = (
    ('Tables E.20, E.22 and E.23', 'Tables B.10, B.11 and S5'), ('Tables E.20 and E.21', 'Tables B.10 and S4'),
    ('Table E.20', 'Table B.10'), ('Appendix E.2', 'Appendix B.2'), ('Table E.22', 'Table B.11'),
    ('Appendix E.4', 'Appendix B.3'), ('Table E.21', 'Table S4'), ('Appendix E.3', 'Table S4'),
    ('Table E.23', 'Table S5'), ('Appendix E.5', 'Table S5'), ('Table E.24', 'Table S6'), ('Appendix E.6', 'Table S6'),
    ('Fig. E.19', 'Fig. B.9'), ('Appendix E.7', 'Appendix B.4'), ('Appendix B (', 'Section S1 ('),
    ('Appendix C (', 'Section S2 ('), ('Appendix D (', 'Section S3 ('), ('Appendices C-D', 'Sections S2-S3'))
V10_STALE = (('Table E.2x', r'Tables? E\.2\d'), ('Appendix E.', r'Appendix E\.'), ('Fig. E.19', r'Fig\. E\.19'),
             ('Appendix B (', r'Appendix B \('), ('Appendix C', r'Appendix C'), ('Appendix D', r'Appendix D'),
             ('Appendices C', r'Appendices C'))
V10_MASK = r'[A-Za-z0-9_./<>*-]*\.csv|appendices_B_D/|appendix_E/'     # CSV file-name identifiers, excluded
V10_RESIDUE = (('Appendix E (bare)', r'Appendix E(?![.0-9])'), ('Appendix B-D', r'Appendix B-D'),
               ('E.2x value', r'(?<![A-Za-z.])E\.2\d'))    # recorded (not failed): names the mapping does not cover
# V10 from v5 on: the edits that rebuild version N's generated half from version N-1's. v4: the amendment's mapping.
# v5 (Amendment 2 to Addendum 91, ruling 1, written here from the ruling): (b) every parenthesised LaTeX label removed
# with the space before it, then (a) and (c) as literal replacements. Nearest wording, as the ruling allows: the
# generated text quotes "---" with double quotes (the ruling: '---') and writes "Sections S2-S3" with an ASCII hyphen
# (the ruling: S2–S3); the replacement strings are the ruling's, en dashes included. Three strings as re-ruled by the
# expert (2026-11-12): "case-data values" lower-case; the capacity-column entry consumes "single-scenario year; " so the
# phrase appears once; reported_start reads "mark b of Table S5: ...".
V10_V5_LABEL_RX = r' \((?:fig|tab|app):[^()]*\)'
V10_V5_WORDING = (
    ('(Appendix E) printed "---"', '(evaluation-record files, Tables B.10, B.11 and S4–S6) printed "---"'),
    ('Appendix E amounts', 'Evaluation-record amounts'),
    ('Appendix B-D values', 'case-data values (Sections S1–S3)'),
    ('single-scenario year; not printed in Sections S2-S3',
     "the single-scenario years of the manuscript's Table 2; the remaining columns are the multi-scenario years"),
    ('printed mark b of Table S5', 'mark b of Table S5'),
    ('Table S6 itself prints no mark', 'Table S6 carries the mark in its over_budget column'))
V10_EDITS_BY_VERSION = {4: tuple(('lit', o, n) for o, n in V10_NAME_MAP),
                        5: (('rx', V10_V5_LABEL_RX, ''),) + tuple(('lit', o, n) for o, n in V10_V5_WORDING)}
# V12 (version >= 5; Amendment 2 ruling 1): in the generated half, CSV identifiers masked (V10_MASK), none of these may
# remain; the V10 stale patterns stay 0; and the punctuation artefacts below (counted on the UNMASKED generated half)
# must not exceed the predecessor's counts (no double space or orphaned punctuation left by the label removal).
V12_ABSENT = (('(fig:', r'\(fig:'), ('(tab:', r'\(tab:'), ('(app:', r'\(app:'),
              ('label without parentheses', r'(?<![A-Za-z0-9_])(?:fig|tab|app):[A-Za-z]'),
              ('Appendix E', r'Appendix E'), ('Appendix B-D', r'Appendix B-D'))
V12_ARTEFACTS = (('double space', r'(?<=[^|\s])  +(?=[^|\s])'), ('space before punctuation', r' [,.;:)]'),
                 ('( followed by space', r'\( '), ('empty ()', r'\(\)'), ('doubled punctuation', r'[,;:]\s*[,;:]|\.\s*\.(?!\.)'),
                 ('line-leading punctuation', r'(?m)^[,;:.)]'), ('trailing space', r'(?m)[^|] +$'))
MAIN9C_REV = '3c379fa44f3755dbdc6fd540df566b55527d921c'      # Overleaf round 9c (= manuscript_review/main_46c9d77_round9c.tex)
MAIN9C_SHA = 'b13b390e31331abd8a45dd677dd453e336c58b9d63a6a7762bda4681bb8508fc'
V11_EXPECTED = {'Table B.10': 'tab:appE_claims', 'Table B.11': 'tab:appE_neighbours', 'Fig. B.9': 'fig:investment_lattice',
                'Appendix B.2': 'app:results_claims', 'Appendix B.3': 'app:results_neighbours',
                'Appendix B.4': 'app:results_statistics'}
V11_EXPECTED_V5 = {'Table 2': 'tab:res_capacity_summary'}   # v5's new wording names "the manuscript's Table 2"
VERIFY_APPE_PASS = 40   # verify_appE.py at 441722c9 (Addendum 91: + 7 reduced-mode checks); 33 at 80778c12
STATE = {'cap_years': PRINTED_YEARS}


def readme_prose_rel(version):
    """The prose the generator prepends: v1 for supplement versions <= 2, v2 for version 3 (Addendum 91 ruling 3), v3
    from version 4 on (Addendum 91 amendment)."""
    if version <= 2:
        return README_PROSE_REL
    return README_PROSE_V2_REL if version == 3 else README_PROSE_V3_REL


def cap_years(version):
    return CAP_YEARS_BY_VERSION.get(version, CAP_YEARS_V2)
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
        years = STATE['cap_years']
        js = {y: jload(os.path.join(DATA, net, f'{net}_{y}.json')) for y in years}
        src_ids = [g['gen_id'] for g in js[2025]['generators'] if g['type'] in ('WIND', 'PV')]
        if [int(r['generator_id']) for r in rows] != src_ids:
            C.bad.append({'where': f'{fname} {net}', 'ids': [[r['generator_id'] for r in rows], src_ids]})
            return
        caps = [k for k in rows[0] if k.startswith('capacity_')]
        if caps != [f'capacity_{y}_MW' for y in years]:
            C.bad.append({'where': f'{fname} {net} capacity columns', 'got': caps, 'expected_years': list(years)})
            return
        for r in rows:
            C.rows += 1
            for y in years:
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
def v7_readme(members, version=1, rec=None):
    name = 'README_PLACEHOLDER.md' if version == 1 else 'README.md'
    if name not in members:
        return {'entry': name, 'holds': False, 'problem': f'{name} absent'}
    rd = members[name].decode('utf-8')
    sections = re.split(r'^### `([^`]+)`$', rd, flags=re.M)
    cols = {sections[i]: re.findall(r'^\| `([^`]+)` \|', sections[i + 1], flags=re.M) for i in range(1, len(sections), 2)}
    csvs = {n: read_csv(b)[0] for n, b in members.items() if n.endswith('.csv')}
    probs = [n for n in csvs if cols.get(n) != csvs[n]] + [n for n in cols if n not in csvs]
    out = {'entry': name, 'files_with_column_table': len(cols), 'csv_files': len(csvs), 'header_mismatches': probs}
    if version == 1:
        marks = ['[EXPERT PROSE PLACEHOLDER', '[PLACEHOLDER: tau]', '[PLACEHOLDER: objective convention]',
                 '[PLACEHOLDER: certification rule parameters]',
                 'Q: gross operational cost, contracted interface settlement excluded.', repr(SC6.TAU)]
        missing = [m for m in marks if m not in rd]
        out.update({'placeholders_missing': missing, 'holds': not probs and not missing})
        return out
    prose_rel = readme_prose_rel(version)
    with open(os.path.join(REPO, prose_rel), 'rb') as h:
        prose = h.read()
    committed = subprocess.run(['git', 'show', f'HEAD:{prose_rel}'], capture_output=True, cwd=REPO).stdout
    rmeta = (rec or {}).get('readme', {})
    required = ['Q: gross operational cost, contracted interface settlement excluded.', repr(SC6.TAU),
                f'## Generated section (supplement version {version})']
    missing = [m for m in required if m not in rd]
    out.update({'prose_file': prose_rel, 'record_prose_file': rmeta.get('prose_prepended'), 'prose_sha256': _sha(prose), 'prose_equals_HEAD': prose == committed,
                'record_prose_sha256': rmeta.get('prose_sha256'),
                'record_prose_sha_equal': rmeta.get('prose_sha256') == _sha(prose),
                'prose_prefix_equal': members[name][:len(prose)] == prose,
                'placeholder_entry_absent': 'README_PLACEHOLDER.md' not in members,
                'placeholder_marks_found': len(re.findall(r'PLACEHOLDER', rd)),
                'generated_required_missing': missing})
    out['holds'] = (not probs and out['prose_equals_HEAD'] and out['record_prose_sha_equal'] and out['prose_prefix_equal']
                    and out['record_prose_file'] == prose_rel and out['placeholder_entry_absent'] and out['placeholder_marks_found'] == 0 and not missing)
    return out


# ======================================================================================================================
#  V8: the RES-capacity CSVs reproduce the paper's Table 2 (version >= 2)
# ======================================================================================================================
def v8_table2(BD, lines_by_source):
    out = {'sources': {}, 'differences': [], 'cells_compared': 0}
    C_rows, D_rows = BD['C_tn_res_generators.csv'], BD['D_adn_res_generators.csv']
    tech = {'Wind': 'Wind', 'Solar PV': 'Solar PV'}

    def total(net, tch, y):
        rows = C_rows if net == 'TN' else [r for r in D_rows if r['adn_node'] == net]
        return sum(float(r[f'capacity_{y}_MW']) for r in rows if r['type'] == tch), len([r for r in rows if r['type'] == tch])
    for src, lines in lines_by_source.items():
        i0, i1 = A.table_body(lines, TABLE2_LABEL, '\\end{tabular}')
        body = lines[i0:i1]
        hdr = [A.row_cells(ln) for ln in body if re.match(r'^\s*& & \d{4}', ln)]
        mid = next(i for i, ln in enumerate(body) if ln.strip() == '\\midrule')
        rows, net = [], None
        for ln in body[mid + 1:]:
            t = ln.strip()
            if not t.endswith('\\\\') or t in ('\\midrule', '\\bottomrule'):
                continue
            c = A.row_cells(t)
            if c[0]:
                m = re.fullmatch(r'ADN at TN node~(\d)', c[0])
                net = 'TN' if c[0] == 'TN' else (m.group(1) if m else None)
            rows.append((net, c[1], c[2:]))
        printed = []
        for net_, tch, vals in rows:
            for y, pv in zip(TABLE2_YEARS, vals):
                out['cells_compared'] += 1
                v, n = total(net_, tech.get(tch, tch), y)
                ok = net_ is not None and n > 0 and A.at_print(pv, v)[0]
                printed.append({'network': net_, 'technology': tch, 'year': y, 'printed': pv, 'supplement_sum': v,
                                'generators': n, 'ok': ok})
                if not ok:
                    out['differences'].append({'source': src, 'network': net_, 'technology': tch, 'year': y,
                                               'printed': pv, 'supplement_sum': v})
        out['sources'][src] = {'header_years': hdr[0][2:] if hdr else None, 'rows': len(rows), 'cells': printed,
                               'header_ok': bool(hdr) and hdr[0][2:] == list(TABLE2_YEARS), 'rows_ok': len(rows) == 8,
                               'sentence_printed': any(TABLE2_SENTENCE in ln for ln in lines)}
    viol = []
    for fname, rows in (('C_tn_res_generators.csv', C_rows), ('D_adn_res_generators.csv', D_rows)):
        for r in rows:
            for a, bs in (('2025', ('2028',)), ('2030', ('2031', '2034')), ('2035', ('2037',))):
                for b in bs:
                    if float(r[f'capacity_{a}_MW']) != float(r[f'capacity_{b}_MW']):
                        viol.append(f'{fname} {r.get("network", "case9")} gen {r["generator_id"]} {a} != {b}')
    out['multi_scenario_sentence_violations'] = viol
    out['holds'] = (not out['differences'] and not viol
                    and all(v['header_ok'] and v['rows_ok'] and v['sentence_printed'] for v in out['sources'].values()))
    return out


# ======================================================================================================================
#  V9: every entry against the predecessor zip (version >= 2)
# ======================================================================================================================
def csv_project(b, header):
    """The CSV bytes restricted to the given columns, written as the generator writes (QUOTE_MINIMAL, '\\n')."""
    rows = list(csv.reader(io.StringIO(b.decode('utf-8'))))
    idx = [rows[0].index(h) for h in header]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
    for r in rows:
        w.writerow([r[i] for i in idx])
    return buf.getvalue().encode('utf-8')


def load_predecessor(rec):
    pred = rec.get('predecessor') or {}
    zpath = os.path.join(REPO, os.path.dirname(pred['record']), pred['zip'])
    with open(zpath, 'rb') as h:
        zb = h.read()
    with open(os.path.join(REPO, pred['record']), encoding='utf-8') as h:
        prec = json.load(h)
    z = zipfile.ZipFile(io.BytesIO(zb))
    return pred, zpath, zb, prec, {n: z.read(n) for n in z.namelist()}


def v9_predecessor(members, PRED, version=2):
    pred, zpath, zb, prec, old = PRED
    declared = V9_DECLARED_BY_VERSION[version]
    res_added = V9_RES_COLUMNS_ADDED_BY_VERSION[version]
    names = sorted(set(members) | set(old))
    same = [n for n in names if n in members and n in old and members[n] == old[n]]
    changed = [n for n in names if n in members and n in old and members[n] != old[n]]
    added = [n for n in names if n in members and n not in old]
    removed = [n for n in names if n not in members and n in old]
    proj = {}
    for n in ('appendices_B_D/C_tn_res_generators.csv', 'appendices_B_D/D_adn_res_generators.csv'):
        if n in members and n in old:
            oh = read_csv(old[n])[0]
            nh = read_csv(members[n])[0]
            try:
                pb = csv_project(members[n], oh)
                eq = pb == old[n]
            except ValueError:
                eq = False
            proj[n] = {'predecessor_columns_retained_bytewise': eq, 'columns_added': [h for h in nh if h not in oh],
                       'columns_removed': [h for h in oh if h not in nh],
                       'column_order_kept': [h for h in nh if h in oh] == oh}
    tn = 'appendices_B_D/D_ieee33_tn_connection.csv'
    out = {'predecessor': pred, 'predecessor_zip_path': os.path.relpath(zpath, REPO), 'predecessor_zip_sha256': _sha(zb),
           'predecessor_sha_equals_record': _sha(zb) == pred['zip_sha256'] == prec['zip']['sha256'],
           'identical': same, 'changed': changed, 'added': added, 'removed': removed,
           'declared': declared, 'res_columns_added_expected': res_added, 'res_projection': proj,
           'entry_sha256': {n: {'predecessor': _sha(old[n]) if n in old else None,
                                'this': _sha(members[n]) if n in members else None} for n in names},
           'tn_connection_identical': tn in same,
           'tn_connection_sha256': _sha(members[tn]) if tn in members else None}
    out['holds'] = (out['predecessor_sha_equals_record'] and changed == declared['changed']
                    and added == declared['added'] and removed == declared['removed']
                    and out['tn_connection_identical']
                    and all(p['predecessor_columns_retained_bytewise'] and p['column_order_kept']
                            and p['columns_added'] == res_added
                            and not p['columns_removed'] for p in proj.values()) and len(proj) == 2)
    return out


# ======================================================================================================================
#  V10 / V11: the README's names (version >= 4, Addendum 91 amendment)
# ======================================================================================================================
def _prose(version):
    with open(os.path.join(REPO, readme_prose_rel(version)), 'rb') as h:
        return h.read()


def v10_names(members, PRED, version):
    rd = members['README.md']
    prose = _prose(version)
    gen = rd[len(prose):].decode('utf-8') if rd.startswith(prose) else rd.decode('utf-8')
    masked, n_masked = re.subn(V10_MASK, '', gen)
    stale = {k: len(re.findall(rx, masked)) for k, rx in V10_STALE}
    residue = {k: len(re.findall(rx, masked)) for k, rx in V10_RESIDUE}
    pred, _zp, _zb, _prec, old = PRED
    pv = pred['version']
    oprose = _prose(pv)
    ord_ = old['README.md']
    ogen = ord_[len(oprose):].decode('utf-8')
    h_old, h_new = f'## Generated section (supplement version {pv})', f'## Generated section (supplement version {version})'
    hits = []
    exp = ogen.replace(h_old, h_new) if ogen.count(h_old) == 1 else None
    if exp is not None:
        for kind, o, n in V10_EDITS_BY_VERSION[version]:
            if kind == 'rx':
                hits.append([o, n, len(re.findall(o, exp))])
                exp = re.sub(o, n, exp)
            else:
                hits.append([o, n, exp.count(o)])
                exp = exp.replace(o, n)
    expected = prose + exp.encode('utf-8') if exp is not None else None
    out = {'prose_file': readme_prose_rel(version), 'generated_half_bytes': len(gen.encode('utf-8')),
           'csv_identifiers_masked': n_masked, 'stale_names': stale, 'stale_total': sum(stale.values()),
           'residual_old_names_not_in_check': residue,
           'predecessor_prose_prefix_equal': ord_.startswith(oprose), 'predecessor_heading_once': exp is not None,
           'map_hits_on_predecessor': hits,
           'readme_equals_predecessor_mapped': expected == rd,
           'first_difference_offset': (None if expected == rd or expected is None else
                                       next((i for i, (a, b) in enumerate(zip(expected, rd)) if a != b),
                                            min(len(expected), len(rd))))}
    out['holds'] = (rd.startswith(prose) and out['stale_total'] == 0 and out['predecessor_prose_prefix_equal']
                    and out['readme_equals_predecessor_mapped'])
    return out


def _generated_half(rd, prose):
    return (rd[len(prose):] if rd.startswith(prose) else rd).decode('utf-8')


def v12_wording(members, PRED, version):
    """Version >= 5 (Amendment 2 to Addendum 91, ruling 1): no LaTeX label, no 'Appendix E' and no 'Appendix B-D' in the
    generated half (CSV identifiers masked), the V10 stale patterns at 0, and no punctuation artefact beyond the
    predecessor's (counted unmasked)."""
    prose = _prose(version)
    gen = _generated_half(members['README.md'], prose)
    masked, n_masked = re.subn(V10_MASK, '', gen)
    absent = {k: len(re.findall(rx, masked)) for k, rx in V12_ABSENT}
    stale = {k: len(re.findall(rx, masked)) for k, rx in V10_STALE}
    pred, _zp, _zb, _prec, old = PRED
    ogen = _generated_half(old['README.md'], _prose(pred['version']))
    art = {k: len(re.findall(rx, gen)) for k, rx in V12_ARTEFACTS}
    oart = {k: len(re.findall(rx, ogen)) for k, rx in V12_ARTEFACTS}
    out = {'csv_identifiers_masked': n_masked, 'absent_counts': absent, 'stale_names': stale,
           'artefacts': art, 'artefacts_predecessor': oart,
           'artefacts_new': {k: art[k] - oart[k] for k in art if art[k] > oart[k]},
           'predecessor_absent_counts': {k: len(re.findall(rx, re.sub(V10_MASK, '', ogen))) for k, rx in V12_ABSENT}}
    out['holds'] = (not any(absent.values()) and not any(stale.values()) and not out['artefacts_new'])
    return out


def tex_numbering(text):
    """label -> (kind, number) by a static walk of main.tex (comments stripped): sections / subsections numbered, figure /
    table counters never reset (elsarticle), \\addtocounter applied, numbers <letter>.<n> after \\appendix."""
    body = '\n'.join(re.sub(r'(?<!\\)%.*', '', ln) for ln in text.split('\n'))
    tok = re.compile(r'\\appendix(?![A-Za-z])|\\(section|subsection)(\*?)\{|\\begin\{(figure\*?|table\*?|longtable)\}'
                     r'|\\end\{(figure\*?|table\*?|longtable)\}|\\addtocounter\{(table|figure)\}\{(-?\d+)\}'
                     r'|\\label\{([^}]*)\}')
    st = {'app': False, 'sec': 0, 'sub': 0, 'figure': 0, 'table': 0, 'cur': None}
    labels = {}

    def sec_letter():
        return chr(64 + st['sec']) if st['app'] else str(st['sec'])
    for m in tok.finditer(body):
        t = m.group(0)
        if t.startswith('\\appendix'):
            st.update(app=True, sec=0, sub=0)
        elif m.group(1):
            if m.group(2) == '*':
                continue
            if m.group(1) == 'section':
                st.update(sec=st['sec'] + 1, sub=0)
            else:
                st['sub'] += 1
        elif m.group(3):
            kind = 'figure' if m.group(3).startswith('figure') else 'table'
            st[kind] += 1
            st['cur'] = kind
        elif m.group(4):
            st['cur'] = None
        elif m.group(5):
            st[m.group(5)] += int(m.group(6))
        elif m.group(7) is not None:
            lab = m.group(7)
            if st['cur'] and lab.startswith(('fig:', 'tab:')):
                n = st[st['cur']]
                labels[lab] = (st['cur'], f'{sec_letter()}.{n}' if st['app'] else str(n))
            elif not st['cur']:
                labels[lab] = ('section', f'{sec_letter()}.{st["sub"]}' if st['sub'] else sec_letter())
    return labels, body


def v11_main9c(members, main9c_b, version=4):
    labels, body = tex_numbering(main9c_b.decode('utf-8'))
    rd = members['README.md'].decode('utf-8')
    by_kind = {}
    for _lab, (kind, num) in labels.items():
        by_kind.setdefault(kind, set()).add(num)
    names = dict(V11_EXPECTED, **(V11_EXPECTED_V5 if version >= 5 else {}))
    expected = {name: {'label': lab, 'computed': labels.get(lab), 'holds': labels.get(lab) == (
        {'Table': 'table', 'Fig.': 'figure', 'Appendix': 'section'}[name.split()[0]], name.split()[1])}
        for name, lab in names.items()}
    kind_of = {'Table': 'table', 'Tables': 'table', 'Fig.': 'figure', 'Appendix': 'section', 'Section': 'section'}
    typed = sorted(set(re.findall(r'(Tables?|Fig\.|Appendix|Section) ([AB]\.\d+)(?![0-9])', rd)))
    typed_bad = [f'{k} {n}' for k, n in typed if n not in by_kind.get(kind_of[k], set())]
    allnum = sorted(set(re.findall(r'(?<![A-Za-z0-9_.])([AB]\.\d+)(?![0-9])', rd)))
    known = set().union(*by_kind.values())
    untyped_bad = [n for n in allnum if n not in known]
    s_main = {f'S{k}': len(re.findall(rf'(?<![A-Za-z0-9_\\$])S{k}(?![0-9])', body)) for k in range(1, 7)}
    s_readme = sorted(set(re.findall(r'(?<![A-Za-z0-9_])S(\d+)(?![0-9])', rd)), key=int)
    out = {'main_rev': MAIN9C_REV, 'main_sha256': _sha(main9c_b), 'main_sha_equal': _sha(main9c_b) == MAIN9C_SHA,
           'expected_names': expected, 'readme_typed_names': [' '.join(x) for x in typed], 'typed_not_in_main': typed_bad,
           'readme_numbers': allnum, 'numbers_not_in_main': untyped_bad, 's_references_in_main': s_main,
           's_numbers_in_readme': [f'S{k}' for k in s_readme],
           'labels_counted': len(labels)}
    out['holds'] = (out['main_sha_equal'] and all(e['holds'] for e in expected.values()) and not typed_bad
                    and not untyped_bad and all(v >= 1 for v in s_main.values())
                    and set(out['s_numbers_in_readme']) <= set(s_main))
    return out


# ======================================================================================================================
#  negative controls (in memory)
# ======================================================================================================================
def _bump(s, step):
    dp = len(s.split('.')[1]) if '.' in s else 0
    return f'{float(s) + step:.{dp}f}'


def controls(E, BD, FZ, main_lines, w182d, sdir, members=None, version=1, rec=None, T2LINES=None, PRED=None):
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
    if version < 2:
        return out
    # version >= 2 (Addendum 90 ruling 7): the new year columns, the README prose, the predecessor diff
    def v58(_E2, BD2):
        return [[d['where'] for d in v5_case_files(BD2)['differences']],
                [(d['source'], d['network'], d['technology'], d['year']) for d in v8_table2(BD2, T2LINES)['differences']]]
    ctl('C16', 'TN gen 4 capacity_2030_MW 30 -> 31', lambda E2, B2: at(B2['C_tn_res_generators.csv'], 'generator_id', '4').update(
        capacity_2030_MW='31'), (v58, [['C_tn_res_generators.csv case9 gen 4 2030'],
                                       [(src, 'TN', 'Wind', '2030') for src in T2LINES]]))
    ctl('C17', 'ADN node 9 gen 5 capacity_2035_MW 40.0 -> 40.5 (V5 only: Table 2 prints 120 for 120.5)',
        lambda E2, B2: next(r for r in B2['D_adn_res_generators.csv'] if r['adn_node'] == '9' and r['generator_id'] == '5').update(
            capacity_2035_MW='40.5'), (v58, [['D_adn_res_generators.csv case33_3 gen 5 2035'], []]))

    def v5cols(_E2, BD2):
        return [d['where'] for d in v5_case_files(BD2)['differences']]

    def drop35(_E2, B2):
        for r in B2['D_adn_res_generators.csv']:
            r.pop('capacity_2035_MW')
    ctl('C18', 'capacity_2035_MW column dropped from D_adn_res_generators.csv', drop35,
        (v5cols, [f'D_adn_res_generators.csv {n} capacity columns' for n, _k in ADNS]))

    def mem_ctl(cid, desc, edit, fn, expect):
        m2 = dict(members)
        edit(m2)
        got = fn(m2)
        caught = got == expect
        out.append({'id': cid, 'edit': desc, 'got': got, 'expected': expect, 'caught_exactly': caught})
        _ok(caught, f'control {cid}: {desc} -> {got}')

    def flip_prose(m2):
        b = bytearray(m2['README.md'])
        i = b.index('Peças'.encode('utf-8'))
        b[i] = ord('p')
        m2['README.md'] = bytes(b)
    mem_ctl('C19', 'README.md: one byte of the prose changed', flip_prose,
            lambda m2: [k for k in ('prose_prefix_equal',) if not v7_readme(m2, version, rec)[k]] + [v7_readme(m2, version, rec)['holds']],
            ['prose_prefix_equal', False])

    def tn250(m2):
        m2['appendices_B_D/D_ieee33_tn_connection.csv'] = m2['appendices_B_D/D_ieee33_tn_connection.csv'].replace(
            b'7,case33_2,1,1,100,-100,100,-100,', b'7,case33_2,1,1,250,-250,250,-250,')
    mem_ctl('C20', 'D_ieee33_tn_connection.csv: node 7 set to the submitted +-250', tn250,
            lambda m2: [v9_predecessor(m2, PRED, version)['changed'], v9_predecessor(m2, PRED, version)['tn_connection_identical']],
            [sorted(V9_DECLARED_BY_VERSION[version]['changed'] + ['appendices_B_D/D_ieee33_tn_connection.csv']), False])

    def oldcol(m2):
        n = 'appendices_B_D/C_tn_res_generators.csv'
        m2[n] = m2[n].replace(b'\n5,6,Wind,40,40,', b'\n5,6,Wind,40,41,')
    mem_ctl('C21', 'C_tn_res_generators.csv: a predecessor column (gen 5 capacity_2028_MW) changed', oldcol,
            lambda m2: {k: v['predecessor_columns_retained_bytewise'] for k, v in v9_predecessor(m2, PRED, version)['res_projection'].items()},
            {'appendices_B_D/C_tn_res_generators.csv': False, 'appendices_B_D/D_adn_res_generators.csv': True})
    if version < 4:
        return out

    def stale20(m2):
        b = m2['README.md']
        anchor = b'Table B.10 (tab:appE_claims)' if version == 4 else b'Table B.10: every pairwise'   # v5: label removed
        i = b.index(anchor, len(_prose(version)))
        m2['README.md'] = b[:i] + b'Table E.20' + b[i + len(b'Table B.10'):]
    mem_ctl('C22', 'README.md generated half: one Table B.10 set back to Table E.20', stale20,
            lambda m2: [v10_names(m2, PRED, version)['stale_names']['Table E.2x'],
                        v10_names(m2, PRED, version)['readme_equals_predecessor_mapped'], v10_names(m2, PRED, version)['holds']],
            [1, False, False])
    if version < 5:
        return out

    def relabel(m2):
        b = m2['README.md']
        i = b.index(b'Table B.10: every pairwise', len(_prose(version))) + len(b'Table B.10')
        m2['README.md'] = b[:i] + b' (tab:appE_claims)' + b[i:]
    mem_ctl('C23', 'README.md generated half: the label (tab:appE_claims) put back after Table B.10', relabel,
            lambda m2: [v12_wording(m2, PRED, version)['absent_counts']['(tab:'],
                        v12_wording(m2, PRED, version)['holds'], v10_names(m2, PRED, version)['readme_equals_predecessor_mapped']],
            [1, False, False])

    def old_amounts(m2):
        b = m2['README.md']
        i = b.index(b'Evaluation-record amounts', len(_prose(version)))
        m2['README.md'] = b[:i] + b'Appendix E amounts' + b[i + len(b'Evaluation-record amounts'):]
    mem_ctl('C24', 'README.md generated half: "Evaluation-record amounts" set back to "Appendix E amounts"', old_amounts,
            lambda m2: [v12_wording(m2, PRED, version)['absent_counts']['Appendix E'],
                        v12_wording(m2, PRED, version)['holds'], v10_names(m2, PRED, version)['readme_equals_predecessor_mapped']],
            [1, False, False])

    def double_space(m2):
        b = m2['README.md']
        i = b.index(b'Section S1: wholesale', len(_prose(version))) + len(b'Section S1:')
        m2['README.md'] = b[:i] + b' ' + b[i:]
    mem_ctl('C25', 'README.md generated half: a double space left after "Section S1:"', double_space,
            lambda m2: [v12_wording(m2, PRED, version)['artefacts_new'], v12_wording(m2, PRED, version)['holds']],
            [{'double space': 1}, False])
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
    STATE['cap_years'] = cap_years(version)
    with open(os.path.join(sdir, zinfo['record']), encoding='utf-8') as h:
        rec = json.load(h)
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
    _ok(v4['equals_committed'] and len(occ) == 1 and v4['line_aligned'] and ver['n_pass'] == VERIFY_APPE_PASS and ver['n_fail'] == 0
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
    v7 = v7_readme(members, version, rec)
    if version == 1:
        _ok(v7['holds'], f'V7 README placeholder: {v7["files_with_column_table"]} column tables = {v7["csv_files"]} CSV '
                         f'headers; placeholders missing {v7["placeholders_missing"]}')
    else:
        _ok(v7['holds'], f'V7 README.md: {v7.get("files_with_column_table")} column tables = {v7.get("csv_files")} CSV '
                         f'headers, mismatches {v7.get("header_mismatches")}; prose {v7.get("prose_file")} sha256 '
                         f'{str(v7.get("prose_sha256"))[:8]} = HEAD {v7.get("prose_equals_HEAD")} = record '
                         f'{v7.get("record_prose_sha_equal")}, prepended byte for byte {v7.get("prose_prefix_equal")}; '
                         f'README_PLACEHOLDER.md absent {v7.get("placeholder_entry_absent")}; PLACEHOLDER marks '
                         f'{v7.get("placeholder_marks_found")}; generated items missing {v7.get("generated_required_missing")}')
    v8 = v9 = T2LINES = PRED = None
    if version >= 2:
        with open(os.path.join(REPO, ROUND9B_COPY), 'rb') as h:
            r9b = h.read()
        T2LINES = {f'main.tex@{MAIN_REV[:7]}': main_lines, ROUND9B_COPY: r9b.decode('utf-8').split('\n')}
        v8 = v8_table2(BD, T2LINES)
        _ok(v8['holds'], f'V8 Table 2 ({TABLE2_LABEL}) from the RES CSVs: {v8["cells_compared"]} printed aggregates over '
                         f'{list(v8["sources"])}, {len(v8["differences"])} differ {v8["differences"][:3]}; multi-scenario '
                         f'sentence violations {v8["multi_scenario_sentence_violations"][:3]}')
        PRED = load_predecessor(rec)
        v9 = v9_predecessor(members, PRED, version)
        _ok(v9['holds'], f'V9 vs predecessor {v9["predecessor"]["zip"]} (sha {v9["predecessor_zip_sha256"][:8]} = record '
                         f'{v9["predecessor_sha_equals_record"]}): {len(v9["identical"])} identical, changed {v9["changed"]}, '
                         f'added {v9["added"]}, removed {v9["removed"]}; RES CSVs keep the predecessor columns bytewise '
                         f'{ {k.split("/")[1]: p["predecessor_columns_retained_bytewise"] for k, p in v9["res_projection"].items()} }; '
                         f'D_ieee33_tn_connection.csv identical {v9["tn_connection_identical"]}')
    v10 = v11 = None
    if version >= 4:
        v10 = v10_names(members, PRED, version)
        _ok(v10['holds'], f'V10 README names: stale a87e195 names in the generated half {v10["stale_names"]} (total '
                          f'{v10["stale_total"]}; {v10["csv_identifiers_masked"]} CSV identifiers masked); README = '
                          f'predecessor README with prose {v10["prose_file"]}, heading and the amendment mapping '
                          f'{v10["readme_equals_predecessor_mapped"]}; residue not in the check '
                          f'{v10["residual_old_names_not_in_check"]}')
        main9c_b = subprocess.run(['git', '-C', CLONE, 'show', f'{MAIN9C_REV}:main.tex'], capture_output=True).stdout
        v11 = v11_main9c(members, main9c_b, version)
        _ok(v11['holds'], f'V11 names vs main.tex at {MAIN9C_REV[:7]} (sha {v11["main_sha256"][:8]} = pinned '
                          f'{v11["main_sha_equal"]}): '
                          f'{ {k: (e["computed"], e["holds"]) for k, e in v11["expected_names"].items()} }; README '
                          f'typed names not in main {v11["typed_not_in_main"]}, numbers not in main '
                          f'{v11["numbers_not_in_main"]}; S references in main {v11["s_references_in_main"]}, README '
                          f'S-numbers {v11["s_numbers_in_readme"]}')
    v12 = None
    if version >= 5:
        v12 = v12_wording(members, PRED, version)
        _ok(v12['holds'], f'V12 README wording (Amendment 2 ruling 1): in the generated half {v12["absent_counts"]} '
                          f'(predecessor {v12["predecessor_absent_counts"]}; {v12["csv_identifiers_masked"]} CSV '
                          f'identifiers masked); stale names {v12["stale_names"]}; punctuation artefacts {v12["artefacts"]} '
                          f'vs predecessor {v12["artefacts_predecessor"]}, new {v12["artefacts_new"]}')
    # controls
    ctl = controls(E, BD, FZ, main_lines, w182d, sdir, members, version, rec, T2LINES, PRED)
    # guards
    g_fail = GUARD.verify(0) + A.GUARD.verify(0)
    _ok(not g_fail and not any(PICKLE_COUNTS.values()) and not any(A.PICKLE_COUNTS.values()),
        f'guards: {GUARD.counts} / W180a {A.GUARD.counts}; pickle {PICKLE_COUNTS} / {A.PICKLE_COUNTS}')
    result = {'stage': 'W188 (b) supplement verifier', 'supplement_dir': os.path.relpath(sdir, REPO),
              'zip': zinfo, 'zip_problems': zprobs, 'V1_frozen': {k: v for k, v in v1.items()},
              'V2_manuscript': v2, 'V3_budget': v3, 'V4_generator': v4, 'V5_case_files': v5, 'V6_printed': v6,
              'V7_readme': v7, 'V8_table2': v8, 'V9_predecessor': v9, 'V10_names': v10, 'V11_main9c': v11, 'V12_wording': v12,
              'capacity_years_checked': list(STATE['cap_years']),
              'controls': ctl, 'controls_all_caught': all(c['caught_exactly'] for c in ctl), 'n_controls': len(ctl),
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
