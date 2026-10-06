"""P5.15 Addendum 65, Planner task W161 -- REGENERATE THE STEP 6 PARAGRAPHS EXPORT FROM THE CORRECTED PACKAGE TEXT AND
RE-RUN THE W160 PARAGRAPH FIGURE CHECK AGAINST THE UNCHANGED FROZEN TABLES. ZERO SOLVES, NO MODEL LOADS. A NEW FILE:
the W160 script (p515_s53_w160_step6_freeze_export.py, 0b2a6a37) is IMPORTED and its functions are CALLED; it is not
edited. Nothing W160 wrote is modified or overwritten (every output here is a new file opened 'x').

WHAT IT DOES
  1. Reproduces W160 first (integrity): `W160.paragraphs_md` on the package blob at e3437284 must equal the committed
     export/paragraphs.md byte for byte, and `W160.figure_checks` on that text against the frozen JSON (590088fe, read
     from disk, sha checked) must equal the 158 checks the frozen JSON stores.
  2. Writes export/paragraphs_v2.md: `W160.paragraphs_md` (same SLICES, same verbatim-copy logic) on the package blob
     at 34ddd323, read from the git object store. W160's paragraphs_md takes no commit argument; it reads the module
     constant PACKAGE_COMMIT only for its header comment, so that constant is set to 34ddd323 for the call and
     restored after it. One W161 comment line is inserted after the header recording the predecessor (paragraphs.md
     sha256 and e3437284); the package blocks are not touched (asserted against the blob's lines).
  3. Re-runs `W160.figure_checks` VERBATIM on paragraphs_v2.md against the unchanged frozen JSON. A check whose
     fragment is no longer in the text (the package was corrected there) is REPLACED in the v2 check set by a W161
     check with the corrected fragment and the written value, against the same table counterpart (taken from the
     W160 check's own table value where it is the same quantity), evaluated by the W160 rules (the evaluator below
     mirrors W160's; it is self-tested on the v1 values of every replaced check and must reproduce W160's status and
     shown value). Figures the correction newly writes get their own W161 checks (C29b-d, L5c, L11a-c). Every W160
     check whose fragment is no longer found must be covered by a replacement, and every replacement fragment must be
     found in paragraphs_v2.md (both asserted).
  4. S2a keeps W160's "no table counterpart" status. Beside it (not counted): the source sentence now in the package
     and the three EFC/day differences derived from T10 (0.50) minus W153 `w153_discount_row.json` settled per-year
     EFC/day (0.70), at the written precision.
  5. Writes export/w161_paragraph_figure_check_v2.json (through gate_result_io), export/README_v2_note.md (one line)
     and export/w161_manifest_sha256.json (the new files, the script, every input). `--post-run` hashes the launch
     log, the typing-test outputs and the manifest into export/w161_manifest_post_run_sha256.json.

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, together with every guard the W160 import arms (W160's and W157's chain); `pickle.load` / `pickle.loads` are
blocked for the whole run and every counter is verified at 0.

MODE (repo root, canonical interpreter; attached, both streams captured):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w161_paragraphs_v2.py \\
        > data/SRP1/Results/P515S53/w160_step6_frozen/export/w161_launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w161_paragraphs_v2.py --post-run
Exit: 0 = written, every integrity check holds and the guards are at 0 (figure-check mismatches are FINDINGS and never
change the exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard fault.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W161 paragraphs v2 (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W161: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W161: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w160_step6_freeze_export as W160  # noqa: E402 -- arms its own guards and the W157 chain; not edited

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the W160 import

GUARDS = W160.W157._dedupe((('w161_paragraphs_v2', GUARD),) + tuple(W160.GUARDS))

SCRIPT_REL = os.path.basename(__file__)
FZ_DIR = os.path.join(W160.S53, 'w160_step6_frozen')
EXPORT = os.path.join(FZ_DIR, 'export')
FZ_REL = os.path.join(FZ_DIR, 'frozen_step6_tables_v1_590088fe.json')
FZ_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
FZ_MAN = os.path.join(FZ_DIR, 'manifest_sha256.json')
P1_REL = os.path.join(EXPORT, 'paragraphs.md')
P1_COMMIT_W160 = '79aca982'
PKG_V1 = 'e3437284'
PKG_V2 = '34ddd323'
PKG_V2_CORRECTIONS = ('27946e6e', '34ddd323')
W160_SCRIPT_COMMIT = '0b2a6a37'
W153C_REL = os.path.join(W160.S53, 'w153_step5_rows', 'w153_certification_statistics.json')
W153D_REL = os.path.join(W160.S53, 'w153_step5_rows', 'w153_discount_row.json')
W153_MAN = os.path.join(W160.S53, 'w153_step5_rows', 'manifest_sha256.json')
W153_COMMIT = '48e76c9f'

OUT_P2 = os.path.join(EXPORT, 'paragraphs_v2.md')
OUT_JSON = os.path.join(EXPORT, 'w161_paragraph_figure_check_v2.json')
OUT_NOTE = os.path.join(EXPORT, 'README_v2_note.md')
OUT_MAN = os.path.join(EXPORT, 'w161_manifest_sha256.json')
OUT_LOG = os.path.join(EXPORT, 'w161_launch.log')
OUT_TYPING_JSON = os.path.join(EXPORT, 'w161_bool_typing_test.json')
OUT_TYPING_LOG = os.path.join(EXPORT, 'w161_bool_typing_test.log')
OUT_POST = os.path.join(EXPORT, 'w161_manifest_post_run_sha256.json')
NOTE_LINE = ('paragraphs_v2.md supersedes paragraphs.md (package text corrected at 34ddd323); the tables are unchanged '
             '(590088fe)')

_log = W160._log
_sha = W160._sha
_sha_bytes = W160._sha_bytes


# ======================================================================================================================
#  the evaluator (mirrors the W160 figure_checks evaluation for the kinds W161 uses; self-tested on the v1 values)
# ======================================================================================================================
def evaluate(kind, written, value, absval=False):
    if kind in ('eur', 'keur', 'pct'):
        dp = W160._dp(written)
        x = value / 1000.0 if kind == 'keur' else 100.0 * value if kind == 'pct' else value
        if absval:
            x = abs(x)
        shown = f'{x:.{dp}f}'
        return shown, 'match' if float(shown) == W160._wval(written) else 'MISMATCH'
    if kind == 'raw':
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            shown = f'{value:.{W160._dp(written)}f}'
            return shown, 'match' if float(shown) == W160._wval(written) else 'MISMATCH'
        shown = str(value)
        return shown, 'match' if shown == written else 'MISMATCH'
    if kind == 'split':
        # W160 'split' = integer % per component; W161 generalises to the precision each component is written at
        parts = written.split('/')
        if len(parts) != len(value):
            return None, 'MISMATCH'
        shown = '/'.join(f'{100 * x:.{W160._dp(p)}f}' for p, x in zip(parts, value))
        return shown, 'match' if all(float(s) == float(p) for s, p in zip(shown.split('/'), parts)) else 'MISMATCH'
    raise ValueError(kind)


# ======================================================================================================================
#  the v2 check definitions (replacements and new figures)
# ======================================================================================================================
def v2_definitions(fz, w160v1):
    """Returns (replacements: {v1 id: [check defs]}, self_tests). Each def: id, fragment, written, kind, value, ref,
    absval, note, replaces. Values are the W160 check's own table value where the quantity is unchanged."""
    t = fz['tables']
    v1 = {c['id']: c for c in w160v1}
    cells = t['cells']
    st = [r for r in cells.values() if r.get('certification_stats_included')]
    cert = [r for r in st if r['status'] == 'certified']
    tse = [r['terminal_step_over_EPS0'] for r in cert]
    tsa = [r['terminal_step_over_EPS0'] for r in st]
    m2cells = [r['cell'] for r in t['dead_zone']['cells']
               if r.get('share_by_node') and (r['cell'].startswith('j_') or r['cell'].startswith('l_'))]
    ag = t['ageing']['rows']
    resolv = [r for r in ag if r.get('eps_AE_resolvable_070') is True]
    e050 = {r['arm']: r['eps_AE_050_superseded'] for r in resolv}
    # replication of W160's sets, checked against W160's own values
    repl = {
        'n_42': len(st), 'n_32': len(cert),
        'C29v1_count_all42_equals_W160': sum(x > 1 for x in tsa) == v1['C29']['table_value_full_precision'],
        'C28_max_tse_equals_W160': max(tse) == v1['C28']['table_value_full_precision'],
        'L5b_m2cells_equals_W160': len(m2cells) == v1['L5b']['table_value_full_precision'],
        'resolvable_band_070_equals_T8_band': [min(r['eps_AE_070'] for r in resolv), max(r['eps_AE_070'] for r in resolv)]
        == t['ageing']['eps_AE_band_over_resolvable_070'],
    }
    srt = sorted(tse)
    med_tse = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2
    repl['C27_median_tse_equals_W160'] = med_tse == v1['C27']['table_value_full_precision']
    n_cert_above1 = sum(x > 1 for x in tse)

    f13 = 'certification came a median 65 cycles after the first residual pass (range 21–89)'
    f27 = 'median 1.72, max 10.42; 22 of the 32 certified cells above 1 (25 of all 42)'
    f5b = 'in each of the five m = 2 cells of T9 that carry a per-node record'
    f5c = '(`j_5f3cccb4` and four L cells)'
    f11 = '> against 0.41–0.62 for the same three resolvable arms at 0.50).'
    f16 = 'The m = 1.5 cell `h_f9eae48f` has about the same split (55 / 14.5 / 31 %) at −2,481.75.'

    def d(cid, frag, written, kind, value, ref, replaces, absval=False, note=None):
        return {'id': cid, 'fragment': frag, 'written': written, 'kind': kind, 'value': value, 'ref': ref,
                'absval': absval, 'note': note, 'replaces': replaces}
    R = {
        'C13': [d('C13', f13, '65', 'raw', v1['C13']['table_value_full_precision'], v1['C13']['table_ref'], 'C13',
                  note='v2 writes 65 = the T2 median of k* − k0_run (k0_run = the first residual pass N); W160 note: '
                       'W153 gives 63.5 for k* − k0 at decision')],
        'C14': [d('C14', f13, '21', 'raw', v1['C14']['table_value_full_precision'], v1['C14']['table_ref'], 'C14')],
        'C15': [d('C15', f13, '89', 'raw', v1['C15']['table_value_full_precision'], v1['C15']['table_ref'], 'C15')],
        'C27': [d('C27', f27, '1.72', 'raw', v1['C27']['table_value_full_precision'], v1['C27']['table_ref'], 'C27')],
        'C28': [d('C28', f27, '10.42', 'raw', v1['C28']['table_value_full_precision'], v1['C28']['table_ref'], 'C28')],
        'C29': [d('C29', f27, '22', 'raw', n_cert_above1, 'T2 terminal step / EPS0 > 1, the 32 certified cells', 'C29',
                  note='W160 C29 counted all 42 cells (25) and noted 22 over the certified cells'),
                d('C29b', f27, '32', 'raw', len(cert), 'T2 certified cells (certification_stats_included)', 'C29'),
                d('C29c', f27, '25', 'raw', v1['C29']['table_value_full_precision'],
                  'T2 terminal step / EPS0 > 1, all 42 cells (the W160 C29 value)', 'C29'),
                d('C29d', f27, '42', 'raw', len(st), 'T2 cells with certification_stats_included', 'C29')],
        'L5b': [d('L5b', f5b, '5', 'raw', v1['L5b']['table_value_full_precision'], v1['L5b']['table_ref'], 'L5b',
                  note=f'T9 carries {", ".join(m2cells)}'),
                d('L5c', f5c, '4', 'raw', sum(c.startswith('l_') for c in m2cells),
                  'T9: L cells among the m = 2 cells with a per-node record', 'L5b',
                  note=f"j_5f3cccb4 in the set: {'j_5f3cccb4' in m2cells}")],
        'L11': [d('L11a', f11, '0.41', 'raw', min(e050.values()),
                  'T8 ε_AE (0.50, superseded), min over the resolvable arms (eps_AE_resolvable_070 true)', 'L11',
                  note='arms ' + ', '.join(f'{a} {v:.4f}' for a, v in e050.items())),
                d('L11b', f11, '0.62', 'raw', max(e050.values()), 'T8 ε_AE (0.50, superseded), max over the resolvable '
                  'arms', 'L11'),
                d('L11c', f11, 'three', 'raw', {3: 'three'}.get(len(e050), str(len(e050))),
                  'T8 count of resolvable arms (the arms of the 0.70 band 1.04–1.85)', 'L11')],
        'L15': [d('L15', f16, '−2,481.75', 'eur', v1['L15']['table_value_full_precision'], v1['L15']['table_ref'], 'L15')],
        'L16': [d('L16', f16, '55/14.5/31', 'split', v1['L16']['table_value_full_precision'],
                  'T9 h_f9eae48f shares n5/n7/n9, % at the written precision (55 / 14.5 / 31)', 'L16',
                  note='"about the same split" as the m = 2 cells (55/14/31)')],
    }
    # self-test: the evaluator reproduces W160's status and shown value on the v1 written values of replaced checks
    v1_kinds = {'C13': 'raw', 'C14': 'raw', 'C15': 'raw', 'C27': 'raw', 'C28': 'raw', 'C29': 'raw', 'L5b': 'raw',
                'L15': 'eur', 'L16': 'split'}
    self_test = {}
    for cid, k in v1_kinds.items():
        c = v1[cid]
        w = '55/14/31' if cid == 'L16' else c['written']
        shown, status = evaluate(k, w, c['table_value_full_precision'])
        self_test[cid] = {'w160_shown': c['table_value_at_written_precision'], 'w160_status': c['status'],
                          'w161_shown': shown, 'w161_status': status,
                          'reproduces': shown == c['table_value_at_written_precision'] and status == c['status']}
    return R, self_test, repl, m2cells, e050


# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    base = W160.pickle_state()
    counts = dict(base['counts'], w161=dict(PICKLE_COUNTS))
    blocked = pickle.load is not _PICKLE_ORIG[0] and pickle.loads is not _PICKLE_ORIG[1]
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run():
    for rel in (OUT_LOG, OUT_MAN, OUT_TYPING_JSON, OUT_TYPING_LOG):
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W161 post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in (OUT_LOG, OUT_MAN, OUT_TYPING_JSON, OUT_TYPING_LOG)}
    with open(os.path.join(REPO, OUT_POST), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f'[W161 post-run] wrote {OUT_POST}: ' + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--out-dir', default=None, help='trial runs only: write every W161 output under this directory')
    args = ap.parse_args()
    if args.out_dir:
        g = globals()
        for k in ('OUT_P2', 'OUT_JSON', 'OUT_NOTE', 'OUT_MAN', 'OUT_LOG', 'OUT_TYPING_JSON', 'OUT_TYPING_LOG', 'OUT_POST'):
            g[k] = os.path.join(args.out_dir, os.path.basename(g[k]))
    if args.post_run:
        post_run()
    t0 = time.time()
    tag = 'W161'
    failed = []
    # ---- preconditions (fail fast, nothing written) ----------------------------------------------------------------
    pre = []
    for rel in (OUT_P2, OUT_JSON, OUT_NOTE, OUT_MAN, OUT_POST, OUT_TYPING_JSON):
        if os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} exists (write-once)')
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    w160_clean = W160.W157.L132._committed_clean(W160.SCRIPT_REL)
    inputs = {}
    for key, rel in (('FROZEN_JSON', FZ_REL), ('W160_MANIFEST', FZ_MAN), ('PARAGRAPHS_V1', P1_REL),
                     ('W153C', W153C_REL), ('W153D', W153D_REL), ('W153_MANIFEST', W153_MAN),
                     ('W160_SCRIPT', W160.SCRIPT_REL)):
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} missing')
            continue
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': W160.W157.L132._committed_clean(rel),
                       'last_commit': W160._last_commit(rel)}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    if inputs['FROZEN_JSON']['sha256'] != FZ_SHA:
        pre.append(f"frozen JSON sha {inputs['FROZEN_JSON']['sha256']} != {FZ_SHA}")
    w160man = json.load(open(os.path.join(REPO, FZ_MAN)))
    for key in ('FROZEN_JSON', 'PARAGRAPHS_V1'):
        inputs[key]['w160_manifest_matches'] = w160man.get(inputs[key]['path']) == inputs[key]['sha256']
        if not inputs[key]['w160_manifest_matches']:
            pre.append(f"{inputs[key]['path']} != its W160 manifest entry")
    w153man = json.load(open(os.path.join(REPO, W153_MAN)))
    for key in ('W153C', 'W153D'):
        inputs[key]['w153_manifest_matches'] = w153man.get(inputs[key]['path']) == inputs[key]['sha256']
        if not inputs[key]['w153_manifest_matches']:
            pre.append(f"{inputs[key]['path']} != its W153 manifest entry")
    blobs = {}
    for c in (PKG_V1, PKG_V2):
        b = W160._git_blob(c, W160.PACKAGE_REL)
        if b is None:
            pre.append(f'git show {c}:{W160.PACKAGE_REL} failed')
        blobs[c] = b
    if W160.PACKAGE_COMMIT != PKG_V1:
        pre.append(f'W160.PACKAGE_COMMIT is {W160.PACKAGE_COMMIT}, expected {PKG_V1}')
    # capture-path assertion: every quantity the task requires has a source before anything runs
    fz = json.load(open(os.path.join(REPO, FZ_REL)))
    for k in ('paragraph_figure_checks', 'paragraph_figure_check_summary', 'paragraphs_md_source_ranges', 'tables'):
        if k not in fz:
            pre.append(f'frozen JSON lacks {k}')
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    w153c = json.load(open(os.path.join(REPO, W153C_REL)))
    w153d = json.load(open(os.path.join(REPO, W153D_REL)))
    _log(f'[{tag}] script {SCRIPT_REL} sha256 {_sha(SCRIPT_REL)} committed clean {script_clean}; W160 script '
         f'committed clean {w160_clean}; frozen JSON {FZ_SHA[:8]} verified; package blobs {PKG_V1} '
         f'{_sha_bytes(blobs[PKG_V1])[:8]}, {PKG_V2} {_sha_bytes(blobs[PKG_V2])[:8]}')

    # ---- 1. reproduce W160 ----------------------------------------------------------------------------------------
    p1_text, p1_ranges, p1_probs = W160.paragraphs_md(blobs[PKG_V1].decode('utf-8'), _sha_bytes(blobs[PKG_V1]))
    p1_disk = open(os.path.join(REPO, P1_REL), 'rb').read()
    rep_paragraphs = (p1_text.encode('utf-8') == p1_disk) and not p1_probs and p1_ranges == fz['paragraphs_md_source_ranges']
    v1_rerun = json.loads(GRIO.dumps(W160.figure_checks(fz, p1_text, w153c)))
    rep_checks = v1_rerun == fz['paragraph_figure_checks']
    _log(f'[{tag}] reproduce W160: paragraphs.md byte-equal {rep_paragraphs}; figure checks equal the 158 stored '
         f'{rep_checks} ({len(v1_rerun)} checks)')

    # ---- 2. paragraphs_v2.md --------------------------------------------------------------------------------------
    pkg2 = blobs[PKG_V2].decode('utf-8')
    pkg2_sha = _sha_bytes(blobs[PKG_V2])
    try:
        W160.PACKAGE_COMMIT = PKG_V2
        p2_raw, p2_ranges, p2_probs = W160.paragraphs_md(pkg2, pkg2_sha)
    finally:
        W160.PACKAGE_COMMIT = PKG_V1
    failed += [f'paragraphs_md v2: {p}' for p in p2_probs]
    p1_sha = inputs['PARAGRAPHS_V1']['sha256']
    w161_line = (f'<!-- W161: regenerated by `{SCRIPT_REL}` with the W160 selection and verbatim-copy logic '
                 f'(`W160.paragraphs_md`, script `{W160_SCRIPT_COMMIT}`) from the package at `{PKG_V2}` (corrections '
                 f'`{PKG_V2_CORRECTIONS[0]}`, `{PKG_V2_CORRECTIONS[1]}`). Supersedes `paragraphs.md` (sha256 `{p1_sha}`, '
                 f'package `{PKG_V1}`, written by W160 at `{P1_COMMIT_W160}`). The frozen tables are unchanged '
                 f'(`frozen_step6_tables_v1_590088fe.json`). This line is also not from the package. -->')
    lines = p2_raw.split('\n')
    hdr_end = 3  # [title, '', W160 header comment, '' ...]
    assert lines[2].startswith('<!-- W160: every block below'), lines[2][:60]
    p2_text = '\n'.join(lines[:hdr_end] + [w161_line] + lines[hdr_end:])
    # assert the package blocks are the blob's lines verbatim
    pkg_lines = pkg2.split('\n')
    blocks_verbatim = all('\n'.join(pkg_lines[r['lines'][0] - 1:r['lines'][1]]) in p2_text for r in p2_ranges)
    non_pkg = [ln for ln in p2_text.split('\n') if ln.startswith('<!-- W16')]
    _log(f'[{tag}] paragraphs_v2.md: {len(p2_ranges)} slices ' +
         ', '.join(f"{r['group']} {r['lines'][0]}-{r['lines'][1]}" for r in p2_ranges) +
         f'; package blocks verbatim {blocks_verbatim}; non-package comment lines {len(non_pkg)}')

    # ---- 3. the figure checks on v2 ------------------------------------------------------------------------------
    v2_w160 = json.loads(GRIO.dumps(W160.figure_checks(fz, p2_text, w153c)))
    not_found = [c['id'] for c in v2_w160 if not c['fragment_found_in_paragraphs_md']]
    R, self_test, repl, m2cells, e050 = v2_definitions(fz, fz['paragraph_figure_checks'])
    uncovered = [i for i in not_found if i not in R]
    unneeded = [i for i in R if i not in not_found]
    v2 = []
    for c in v2_w160:
        if c['id'] in R:
            for dd in R[c['id']]:
                shown, status = evaluate(dd['kind'], dd['written'], dd['value'], dd['absval'])
                v2.append({'id': dd['id'], 'origin': 'W161 replacement' if dd['id'] == dd['replaces']
                           else f"W161 new figure (with the replacement of {dd['replaces']})",
                           'replaces_w160_check': dd['replaces'],
                           'w160_fragment_v1': c['fragment_verbatim'], 'fragment_verbatim': dd['fragment'],
                           'fragment_found_in_paragraphs_md': dd['fragment'] in p2_text, 'written': dd['written'],
                           'table_ref': dd['ref'], 'table_value_full_precision': dd['value'],
                           'table_value_at_written_precision': shown, 'status': status, 'note': dd['note']})
        else:
            v2.append(dict(c, origin='W160 check, unchanged'))
    mism = [c for c in v2 if c['status'] == 'MISMATCH']
    summary = {'n_checks': len(v2), 'n_match': sum(c['status'] == 'match' for c in v2), 'n_mismatch': len(mism),
               'mismatch_ids': [c['id'] for c in mism],
               'n_no_table_counterpart': sum(c['status'] == 'no table counterpart' for c in v2),
               'no_table_counterpart_ids': [c['id'] for c in v2 if c['status'] == 'no table counterpart'],
               'n_approximate': sum(c['status'].startswith('approximate') for c in v2),
               'all_fragments_found': all(c['fragment_found_in_paragraphs_md'] for c in v2),
               'n_w160_unchanged': sum(c['origin'] == 'W160 check, unchanged' for c in v2),
               'n_w161_replacement_or_new': sum(c['origin'] != 'W160 check, unchanged' for c in v2),
               'w160_checks_whose_fragment_is_gone': not_found}
    w160_v2_raw_summary = {'n_checks': len(v2_w160), 'n_match': sum(c['status'] == 'match' for c in v2_w160),
                           'n_mismatch': sum(c['status'] == 'MISMATCH' for c in v2_w160),
                           'fragment_not_found_ids': not_found}
    # S2a beside (not counted)
    efc050 = fz['tables']['a64']['scored']['B']['efc_per_day']
    py = {str(r['year_block']): r['efc_per_day'] for r in w153d['result']['settled_unit_ageing']['per_year']}
    s2a = {'status_in_check_set': next(c['status'] for c in v2 if c['id'] == 'S2a'),
           'source_sentence': 'Source of the EFC/day differences: T10 (0.50) against W153 `48e76c9f` (0.70 per-year EFC).',
           'counted': False, 'derivation': 'T10 a64.scored.B.efc_per_day[year] − W153 w153_discount_row.json '
           'result.settled_unit_ageing.per_year[year].efc_per_day (an input, not a T1–T10 table)', 'rows': []}
    s2a['source_sentence_found_in_paragraphs_v2'] = s2a['source_sentence'] in p2_text
    for y, w, frag in (('2025', '+0.18', 'EFC/day +0.18,'), ('2030', '+0.20', '+0.20, +0.27) but adds only'),
                       ('2035', '+0.27', '+0.20, +0.27) but adds only')):
        dv = efc050[y] - py[y]
        shown, status = evaluate('raw', w, dv)
        s2a['rows'].append({'year': y, 'written': w, 'fragment': frag, 'fragment_found': frag in p2_text,
                            'efc_050_T10': efc050[y], 'efc_070_W153': py[y], 'difference': dv,
                            'difference_at_written_precision': shown, 'status': status})
    # an uncovered figure observed in the corrected slices (no W160 check covers it; reported, not counted)
    obs_line = '| k\\* − k₀ | min 21, median 63.5, max 89 |'
    t2_med = next(c for c in v2 if c['id'] == 'C13')['table_value_full_precision']
    observations = [{'line_verbatim': obs_line, 'found_in_paragraphs_v2': obs_line in p2_text,
                     'covered_by_a_w160_check': any(c['fragment_verbatim'] == obs_line for c in v2),
                     'written_median': 63.5, 't2_median_k_star_minus_k0_run': t2_med,
                     'w153_median_k_star_minus_k0_at_decision':
                         w153c['result']['totals']['cycles_after_k0_k_star_minus_k0_at_decision']['median'],
                     'w153_median_k_star_minus_N':
                         w153c['result']['totals']['cycles_after_k0_k_star_minus_N']['median'],
                     'note': 'sources table of the certification paragraph (W153 statistics); unchanged at 34ddd323 '
                             'while the paragraph now writes 65; not edited (W161 edits no text)'}]

    checks = {
        'w160_paragraphs_md_reproduced_byte_equal': rep_paragraphs,
        'w160_figure_checks_reproduced_equal_to_frozen_json': rep_checks,
        'paragraphs_v2_package_blocks_verbatim': blocks_verbatim,
        'paragraphs_v2_slices_found': not p2_probs and len(p2_ranges) == len(W160.SLICES),
        'every_gone_w160_fragment_covered_by_a_replacement': not uncovered,
        'no_replacement_for_a_w160_check_still_found': not unneeded,
        'every_v2_fragment_found': summary['all_fragments_found'],
        'evaluator_self_test_reproduces_w160_on_v1': all(v['reproduces'] for v in self_test.values()),
        'replication_of_w160_sets': all(v is True for k, v in repl.items() if k not in ('n_42', 'n_32')),
        'w160_package_commit_restored': W160.PACKAGE_COMMIT == PKG_V1,
        's2a_source_sentence_and_fragments_found': s2a['source_sentence_found_in_paragraphs_v2']
        and all(r['fragment_found'] for r in s2a['rows']),
    }
    failed += [k for k, v in checks.items() if v is not True]
    if uncovered:
        failed.append(f'gone W160 fragments without replacement: {uncovered}')
    if unneeded:
        failed.append(f'replacements for W160 checks still found: {unneeded}')

    # ---- 4. write ------------------------------------------------------------------------------------------------
    written = {}

    def wr(rel, data):
        with open(os.path.join(REPO, rel), 'xb') as h:
            h.write(data if isinstance(data, bytes) else data.encode('utf-8'))
        written[rel] = _sha(rel)
    wr(OUT_P2, p2_text)
    wr(OUT_NOTE, NOTE_LINE + '\n')
    guards, pk, guards_ok = guards_state()
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    rec = {'schema': 'p515_s53_w161_paragraph_figure_check_v2', 'version': 1,
           'stage': 'P5.15 W161 -- Step 6 paragraphs v2 and the W160 figure check re-run (Addendum 65)',
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': W160._git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean},
           'w160_script': {'path': W160.SCRIPT_REL, 'sha256': inputs['W160_SCRIPT']['sha256'],
                           'commit': W160_SCRIPT_COMMIT, 'committed_clean': w160_clean, 'edited': False},
           'frozen_tables': {'path': FZ_REL, 'sha256': FZ_SHA, 'unchanged': _sha(FZ_REL) == FZ_SHA},
           'paragraphs_v2': {'path': OUT_P2, 'sha256': written[OUT_P2], 'package': W160.PACKAGE_REL,
                             'package_commit': PKG_V2, 'package_commit_full': W160._git('rev-parse', PKG_V2),
                             'package_blob_sha256': pkg2_sha, 'package_corrections_in': list(PKG_V2_CORRECTIONS),
                             'source_ranges': p2_ranges,
                             'generator': 'W160.paragraphs_md (PACKAGE_COMMIT set to 34ddd323 for the call, restored) '
                                          '+ one W161 comment line after the header'},
           'predecessor': {'path': P1_REL, 'sha256': p1_sha, 'package_commit': PKG_V1,
                           'package_blob_sha256': _sha_bytes(blobs[PKG_V1]), 'written_by': 'W160',
                           'commit': P1_COMMIT_W160, 'unchanged': _sha(P1_REL) == p1_sha},
           'readme_v2_note': {'path': OUT_NOTE, 'sha256': written[OUT_NOTE], 'text': NOTE_LINE},
           'inputs': inputs,
           'figure_check_v2_summary': summary,
           'figure_check_v2': v2,
           'w160_figure_checks_verbatim_on_v2_text': {'summary': w160_v2_raw_summary, 'checks': v2_w160},
           'w160_v1_summary_for_reference': fz['paragraph_figure_check_summary'],
           'evaluator_self_test_on_v1': self_test, 'replication_of_w160_sets': repl,
           'm2_cells_with_per_node_record': m2cells, 'eps_AE_050_resolvable_arms': e050,
           's2a_beside_not_counted': s2a, 'uncovered_figure_observations': observations,
           'checks': checks, 'failed': sorted(set(failed)), 'guards': guards, 'pickle_guard': pk,
           'exit_code': code, 'wall_s': time.time() - t0}
    wr(OUT_JSON, GRIO.dumps(rec, indent=1, sort_keys=True) + '\n')
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for v in inputs.values():
        man[v['path']] = v['sha256']
    wr(OUT_MAN, GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    # ---- log -----------------------------------------------------------------------------------------------------
    s = summary
    _log(f"[{tag}] W160 checks verbatim on v2 text: {w160_v2_raw_summary['n_checks']} checks, fragment not found "
         f"{not_found} (each replaced)")
    _log(f"[{tag}] figure check v2: {s['n_checks']} checks ({s['n_w160_unchanged']} W160 unchanged + "
         f"{s['n_w161_replacement_or_new']} W161 replacement/new), {s['n_match']} match, {s['n_mismatch']} MISMATCH "
         f"{s['mismatch_ids']}, {s['n_no_table_counterpart']} no table counterpart {s['no_table_counterpart_ids']}, "
         f"{s['n_approximate']} approximate; fragments all found {s['all_fragments_found']}")
    for c in v2:
        if c['origin'] != 'W160 check, unchanged' or c['status'] != 'match':
            _log(f"[{tag}]   {c['id']} [{c['origin']}] {c['status']}: written {c['written']!r}, table "
                 f"{c['table_value_at_written_precision']!r} ({c['table_ref']})"
                 f"{' -- ' + c['note'] if c.get('note') else ''}")
    for r in s2a['rows']:
        _log(f"[{tag}]   S2a beside (not counted) {r['year']}: written {r['written']}, T10 {r['efc_050_T10']:.4f} − "
             f"W153 {r['efc_070_W153']:.4f} = {r['difference_at_written_precision']} {r['status']}")
    _log(f"[{tag}]   S2a source sentence found: {s2a['source_sentence_found_in_paragraphs_v2']}")
    for o in observations:
        _log(f"[{tag}]   OBSERVATION (uncovered, not counted): {o['line_verbatim']} found {o['found_in_paragraphs_v2']}; "
             f"T2 median {o['t2_median_k_star_minus_k0_run']}, W153 at decision "
             f"{o['w153_median_k_star_minus_k0_at_decision']}")
    for k, v in self_test.items():
        _log(f"[{tag}]   self-test {k}: W160 {v['w160_status']} {v['w160_shown']!r} / W161 {v['w161_status']} "
             f"{v['w161_shown']!r} -> {v['reproduces']}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
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
