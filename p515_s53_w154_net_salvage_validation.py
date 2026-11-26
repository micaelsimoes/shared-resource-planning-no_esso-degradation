"""P5.15 Planner task W154b -- NET-OF-SALVAGE SCORER VALIDATION (substitute evidence for Addendum 64's "three more
recorded rows from different families"). ZERO SOLVES, NO MODEL LOADS.

Context. W154 found, by a scoped search, that recorded net-of-salvage differences exist only in the year-ladder (G)
family: the 8 v6 G rows, the W118/W129 year ladder (report prose at 0.1 EUR) and the 20 Phase A A1b `G:*:net` rows of
w117_triage_recompute.json (3b2e76de). This script builds the strongest substitute evidence without solving anything.

METHOD. The W153 method as committed (p515_s53_w153_step5_zero_solve_rows.py, 48e76c9f) is IMPORTED AND CALLED, not
re-implemented: `W153._form_d` (the claim form), `W153.DET.resolve_v6` (the committed v6 scorer resolution, called
through W153's own import), `W153.salvage_row` (the W153 salvage add-back section, re-run read-only for its view-to-cell
match and its per-cell identity), and W153's read-side pins (`_load`, `_jsonl`, `_verified`: every JSON / JSONL read is
sha256-verified against the committed manifest that lists it). W117's own verdict rule is `W117.evaluate` (called).
The W118 resolution rule is `L132.resolve` (called; W118's settled-vs-settled rule = sum of the two band widths).

GUARDS. pickle.load / pickle.loads are replaced by blocking counters, and an armed `SolveProfileGuard(permitted=())` is
installed, BEFORE any project import; every imported module's zero-permit guard (W153's, S46's, W132's launcher, W117's)
is collected; all are verified at exactly 0 at the end. W153's import installs its own pickle block on top of this one;
both counters are reported and must be 0.

PARTS (the checks are stated here before the run)
  1. GROSS FORM CHECK, every one of the 60 claims (48 v6 + 11 item E + 1 Phase B): the W153 claim-form code path applied
     to the recorded GROSS views (Q in place of Q_net; Q_cc = Q + t in place of Q_net + t):
       d = W153._form_d(form, I_ref, I_other, ref.Q, other.Q); d_cc = the same on Q + t; res = DET.resolve_v6(d, d_cc, views)
     against the recorded d_Q, d_Qcc and the recorded gross resolution (verdict, rule, threshold / bar, whole dict).
     Families: B (B rows on b_ cells), D (fit-free B rows: the B claims whose other cell is a d_ cell -- item D itself is
     a fit, not a claim), C, E, G, H, I, J, L, CHECK, Phase B (the ext phase_b_claim). Also per claim: the net d of the
     same path minus the gross d equals the salvage-only term (F: sv_ref - sv_other; value: sv_other - sv_ref) within
     1e-6 EUR, i.e. net differs from gross only by the Q -> Q_net substitution.
     Checks: d and d_cc bitwise equal to the record; verdict, rule and the whole resolution dict equal; salvage-only term.
  2. PER-CELL SALVAGE IDENTITY, every cell any of the 60 claims reads (49):
       (a) view level (W153's own check, re-run by calling W153.salvage_row): Q - Q_net - salvage;
       (b) report level: Q_end (= Q_k* when certified, Q at the cap otherwise) - terminal_salvage_value_end - Q_net_end
           (W118 references: Q_at_cap - terminal_salvage_value_last - net_operational_recourse_last, last = cap;
            W101 references: their per-cycle row at k*);
       (c) raw: the cell's per_cycle_record.jsonl row at the end cycle (sha-verified): gross_operational_cost -
           terminal_salvage_value - recourse, and the row equals the view's Q, Q_net and salvage.
     Checks: every residual |.| < 1e-6 EUR; the row's values equal the view's (bitwise).
  3. INDEPENDENT RECORDED NET ROWS outside the 8 v6 G rows.
     (a) The 20 Phase A A1b `G:*:net` rows of w117_triage_recompute.json. Inputs W117 used: S45 A1b campaign_results.json
         points (F_eur, I_x_eur, certified gross, net_operational_recourse, terminal_salvage_value), the x = 0 cell
         7aa017f0 (S45 a0_c7) evaluation record (Q(0), its salvage and net recourse) and the interface detail files
         (t_sum). The W153 net form: d_net = W153._form_d('F', 0, I(x), Q_net(0), Q_net(x)); d_net_cc on Q_net + t.
         Checks: |d_net - recorded| <= 0.01 EUR or <= 1e-6 relative, against W117's `recorded` (F_eur - Q(0) -
         terminal_salvage_value) and W117's d_Q; d_net_cc likewise against W117's d_Qcc; W117's verdicts (R1 primary, R0)
         reproduced by calling W117.evaluate on the same inputs, and no W117 verdict within reach of the d difference.
         The v6 net verdict is put beside where the v6 summary scores the same claim id (a different, re-settled run).
     (b) The W118 year-ladder net (P5_15_ADDENDUM57_BENCHMARK_AND_RESETTLE_REPORT.md lines 49-53, 0.1 EUR prose): the
         W153 form on W118's summary (Q_k*, net_operational_recourse_last, terminal_salvage_value_last, I_j) and the x = 0
         settled reference view; gross M and D reproduced against W118's recorded figures (1e-6); net reported at full
         precision and at the prose's 0.1 EUR; the prose comparison is reported, with the prose's own rounding stated.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w154_net_salvage_validation/):
  w154_net_salvage_validation.json; launch.log beside; manifest_sha256.json by the separate --manifest invocation.
Smoke (nothing written): --dry. Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S53/w154_net_salvage_validation && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w154_net_salvage_validation.py \\
        > data/SRP1/Results/P515S53/w154_net_salvage_validation/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w154_net_salvage_validation.py --manifest \\
        > data/SRP1/Results/P515S53/w154_net_salvage_validation/manifest_launch.log 2>&1
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

# ---- the no-model-load guard (installed before any project import) ------------------------------------------------------
PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W154b: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W154b: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W154b net-of-salvage validation (never solves)').install()

import gate_result_io as GRIO  # noqa: E402 -- the one JSON writer (W100)
import p515_s53_w153_step5_zero_solve_rows as W153  # noqa: E402 -- the committed method (arms its own guards + pickle block)
import p515_s53_w117_triage_recompute as W117  # noqa: E402 -- W117's verdict rule (arms its own zero-permit guard)

DET = W153.DET          # p515_s53_w142_determinacy (the v6 scorer), as W153 imported it
L132 = DET.L132         # p515_s53_w132_resettle_v3_campaign (W132 scorer / references / W118 rule)
GUARDS = W153._dedupe((('w154b_net_salvage_validation', GUARD),) + tuple(W153.GUARDS)
                      + (('w117_triage_recompute', W117._GUARD),))

STAGE = 'P5.15 Planner task W154b -- net-of-salvage scorer validation (gross form check, salvage identity, recorded net rows)'
S53 = W153.S53
OUT_DIR_REL = os.path.join(S53, 'w154_net_salvage_validation')
OUT_JSON = 'w154_net_salvage_validation.json'
OUT_MAN = 'manifest_sha256.json'
LAUNCH_LOG = 'launch.log'

IDENTITY_TOL_EUR = 1e-6           # parts 1 (salvage-only term) and 2
REPRO_ABS_EUR = 0.01              # part 3(a): 0.01 EUR ...
REPRO_REL = 1e-6                  # ... or 1e-6 relative
PROSE_UNIT_EUR = 0.1              # part 3(b): the prose's stated precision

S45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
A1B_ROOT = os.path.join(S45, 'campaign_s45_a1b')
A0C7_ROOT = os.path.join(S45, 'campaign_s45_a0_c7')
X0_OLD_EVAL_DIR = os.path.join(A0C7_ROOT, 'evals', '7aa017f09989b56d_x0')
W117_OUT = {'path': os.path.join(S53, 'w117_triage_recompute', 'w117_triage_recompute.json'),
            'manifest': os.path.join(S53, 'w117_triage_recompute', 'manifest_sha256.json'), 'commit': '3b2e76de'}
W112_OUT = {'path': os.path.join(S53, 'w112_consensus_gap', 'w112_consensus_gap.json'),
            'manifest': os.path.join(S53, 'w112_consensus_gap', 'manifest_sha256.json')}
PROSE = {'path': 'P5_15_ADDENDUM57_BENCHMARK_AND_RESETTLE_REPORT.md', 'lines': (49, 53)}
W118_ROOT = os.path.join(S53, 'w118_resettle')
DETAIL = 'interface_settlement_detail_s31c.json'
PCR = 'per_cycle_record.jsonl'
MANIFEST_NAME = 'campaign_manifest_sha256.json'

UNTRACKED_INPUTS = []


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _within(a, b):
    """part 3(a) reproduction rule: |a - b| <= 0.01 EUR or <= 1e-6 relative."""
    diff = abs(a - b)
    return diff <= REPRO_ABS_EUR or diff <= REPRO_REL * max(abs(a), abs(b))


def _rel(a, b):
    den = max(abs(a), abs(b))
    return abs(a - b) / den if den else 0.0


def _one_eval_dir(root, suffix):
    evs = [d for d in os.listdir(os.path.join(REPO, root, 'evals')) if d.endswith('_' + suffix)]
    if len(evs) != 1:
        raise RuntimeError(f'{root}: expected one eval dir ending _{suffix}, found {evs}')
    return os.path.join(root, 'evals', evs[0])


def _summaries():
    v6 = W153._load(W153.V6_SUMMARY['path'], W153.V6_SUMMARY['manifest'])
    ext = W153._load(W153.EXT_SUMMARY['path'], W153.EXT_SUMMARY['manifest'])
    return v6, ext


def _claims(v6, ext):
    """The 60 claims in W153's order (salvage_row builds the same list)."""
    return ([('v6', 'claims_scored', c) for c in v6['claims_scored']]
            + [('ext', 'item_E', c) for c in ext['item_E']['claims'].values()]
            + [('ext', 'phase_b_claim', c) for c in ext['phase_b_claim']])


def _family(where, c, other_cells):
    if where == 'phase_b_claim':
        return 'Phase B'
    if c['item'] == 'B' and other_cells and all(x.startswith('d_') for x in other_cells):
        return 'D'
    return c['item']


FAMILY_ORDER = ('B', 'C', 'D', 'E', 'G', 'H', 'I', 'J', 'L', 'CHECK', 'Phase B')
FAMILY_NOTE = {'B': 'B rows on the b_ cells', 'D': 'fit-free B rows: B claims whose other cell is a d_ cell (item D '
               'itself is the affine fit, not a claim)', 'Phase B': 'the ext summary phase_b_claim (pb_y2025_n5_v6)'}


# ======================================================================================================================
#  1. gross form check
# ======================================================================================================================
def gross_form_check(w153_rows):
    checks, out = {}, {}
    v6, ext = _summaries()
    claims = _claims(v6, ext)
    checks['g_60_claims'] = len(claims) == 60 == len(w153_rows)
    rows = []
    for (src, where, c), wr in zip(claims, w153_rows):
        if wr['claim_id'] != c['claim_id'] or wr['summary'] != src:
            raise RuntimeError(f'claim order differs from W153.salvage_row at {c["claim_id"]}')
        ref, oth = c['ref'], c['other']
        form, ir, io = c['form'], c['I_ref'], c['I_other']
        d = W153._form_d(form, ir, io, ref['Q'], oth['Q'])
        d_cc = W153._form_d(form, ir, io, ref['Q'] + ref['t'], oth['Q'] + oth['t'])
        res = DET.resolve_v6(d, d_cc, (ref, oth))
        rec = c['gross']
        thr, thr_rec = res.get('threshold', res.get('bar')), rec.get('threshold', rec.get('bar'))
        # net by the same path; net minus gross must be the salvage-only term
        d_n = W153._form_d(form, ir, io, ref['Q_net'], oth['Q_net'])
        sal_term = (ref['salvage'] - oth['salvage']) if form == 'F' else (oth['salvage'] - ref['salvage'])
        gross_primary = c['objective_convention_primary'].startswith('gross')
        rows.append({
            'summary': src, 'list': where, 'claim_id': c['claim_id'], 'item': c['item'],
            'family': _family(where, c, wr['other_cells']), 'form': form,
            'ref_cells': wr['ref_cells'], 'other_cells': wr['other_cells'],
            'ref_status': ref['status'], 'other_status': oth['status'],
            'd_gross_w153_path': d, 'd_gross_recorded': c['d_Q'], 'abs_diff_d': abs(d - c['d_Q']),
            'd_bitwise': d == c['d_Q'],
            'd_cc_w153_path': d_cc, 'd_cc_recorded': c['d_Qcc'], 'abs_diff_d_cc': abs(d_cc - c['d_Qcc']),
            'd_cc_bitwise': d_cc == c['d_Qcc'],
            'rule_w153_path': res.get('rule'), 'rule_recorded': rec.get('rule'),
            'threshold_or_bar_w153_path': thr, 'threshold_or_bar_recorded': thr_rec,
            'abs_diff_threshold': (abs(thr - thr_rec) if (thr is not None and thr_rec is not None) else None),
            'verdict_w153_path': res['verdict'], 'verdict_recorded': rec['verdict'],
            'verdict_match': res['verdict'] == rec['verdict'],
            'resolution_dict_equal': res == rec,
            'primary_convention': c['objective_convention_primary'],
            'primary_verdict_recorded': c['verdict'],
            'primary_verdict_match_where_gross_primary': (res['verdict'] == c['verdict']) if gross_primary else None,
            'sign_positive_w153_path': d > 0,
            'sign_primary_positive_recorded': c.get('sign_primary_positive'),
            'd_net_w153_path': d_n, 'net_minus_gross': d_n - d, 'salvage_only_term': sal_term,
            'abs_net_minus_gross_minus_salvage_term': abs((d_n - d) - sal_term),
            'd_net_equals_w153_salvage_row': d_n == wr['d_net'] or wr['net_source'].startswith('recorded')})
    fam = {}
    for f in FAMILY_ORDER:
        rs = [r for r in rows if r['family'] == f]
        thr_d = [r['abs_diff_threshold'] for r in rs if r['abs_diff_threshold'] is not None]
        fam[f] = {'n': len(rs), 'note': FAMILY_NOTE.get(f),
                  'max_abs_diff_d': max((r['abs_diff_d'] for r in rs), default=None),
                  'max_abs_diff_d_cc': max((r['abs_diff_d_cc'] for r in rs), default=None),
                  'max_abs_diff_threshold': max(thr_d, default=None),
                  'n_d_bitwise': sum(r['d_bitwise'] for r in rs),
                  'n_verdict_match': sum(r['verdict_match'] for r in rs),
                  'n_resolution_dict_equal': sum(r['resolution_dict_equal'] for r in rs),
                  'rules': sorted({r['rule_recorded'] for r in rs}),
                  'verdicts_recorded': {v: sum(r['verdict_recorded'] == v for r in rs)
                                        for v in sorted({r['verdict_recorded'] for r in rs})},
                  'forms': sorted({r['form'] for r in rs}),
                  'max_abs_net_minus_gross_minus_salvage_term': max(
                      (r['abs_net_minus_gross_minus_salvage_term'] for r in rs), default=None),
                  'n_claims_with_nonzero_salvage_term': sum(abs(r['salvage_only_term']) >= W153.ABS_SALVAGE_NEGLIGIBLE_EUR
                                                            for r in rs),
                  'pass': bool(rs) and all(r['d_bitwise'] and r['d_cc_bitwise'] and r['verdict_match']
                                           and r['resolution_dict_equal'] for r in rs)}
    checks['g_every_family_present'] = all(fam[f]['n'] > 0 for f in FAMILY_ORDER)
    checks['g_family_counts_sum_to_60'] = sum(v['n'] for v in fam.values()) == 60
    checks['g_d_bitwise_all'] = all(r['d_bitwise'] for r in rows)
    checks['g_d_cc_bitwise_all'] = all(r['d_cc_bitwise'] for r in rows)
    checks['g_verdict_match_all'] = all(r['verdict_match'] for r in rows)
    checks['g_rule_match_all'] = all(r['rule_w153_path'] == r['rule_recorded'] for r in rows)
    checks['g_resolution_dict_equal_all'] = all(r['resolution_dict_equal'] for r in rows)
    checks['g_primary_verdict_match_where_gross_primary'] = all(
        r['primary_verdict_match_where_gross_primary'] in (True, None) for r in rows)
    checks['g_sign_matches_recorded_primary_where_gross_primary'] = all(
        r['sign_positive_w153_path'] == r['sign_primary_positive_recorded'] for r in rows
        if r['primary_convention'].startswith('gross') and r['sign_primary_positive_recorded'] is not None)
    checks['g_net_minus_gross_is_the_salvage_only_term'] = all(
        r['abs_net_minus_gross_minus_salvage_term'] < IDENTITY_TOL_EUR for r in rows)
    checks['g_net_d_equals_w153_salvage_row'] = all(r['d_net_equals_w153_salvage_row'] for r in rows)
    out['claims'] = rows
    out['per_family'] = fam
    out['definitions'] = {
        'path': ('W153._form_d(form, I_ref, I_other, ref.Q, other.Q) (F: [Q(o) + I(o)] - [Q(r) + I(r)]; value: '
                 '[Q(r) - Q(o)] - [I(o) - I(r)]); d_cc the same on Q + t; resolution DET.resolve_v6(d, d_cc, (ref, other)) '
                 '-- exactly W153 salvage_row with Q in place of Q_net'),
        'compared_with': 'the claim record: d_Q, d_Qcc, gross (the recorded resolution dict: rule, threshold/bar, verdict)',
        'salvage_only_term': 'F: salvage(ref) - salvage(other); value: salvage(other) - salvage(ref)',
        'families': {f: FAMILY_NOTE.get(f, f'claim item {f}') for f in FAMILY_ORDER}}
    return checks, out


# ======================================================================================================================
#  2. per-cell salvage identity
# ======================================================================================================================
def _cell_source(cell, summary, v6, ext):
    """(eval_dir, campaign manifest, end cycle, report-level dict, provenance note) for a cell a claim reads."""
    if cell.startswith('ref:'):
        p8 = cell[4:]
        ref = L132.REFERENCES[p8]
        view = (v6 if summary == 'v6' else ext)['references'][p8]
        if ref['kind'] == 'w101':
            ed = ref['eval_dir']
            root = os.path.dirname(os.path.dirname(ed))
            rows = {r['cycle']: r for r in W153._jsonl(os.path.join(ed, PCR), os.path.join(root, MANIFEST_NAME))}
            k = view['k_star']
            rep = {'level': 'W101 per-cycle row at k*', 'Q': rows[k]['gross_operational_cost'],
                   'salvage': rows[k]['terminal_salvage_value'], 'Q_net': rows[k]['recourse'], 'end': k,
                   'last_cycle_recorded': max(rows)}
            return ed, os.path.join(root, MANIFEST_NAME), k, rep, ref['name']
        w118 = W153._load(L132.W118_SUMMARY, L132.W118_SUMMARY_MANIFEST)
        r = w118['reports'][ref['cell']]
        root = os.path.join(W118_ROOT, f"campaign_s53_w118_resettle_r2_{ref['cell']}")
        ed = _one_eval_dir(root, ref['cell'])
        if not os.path.basename(ed).startswith(r['eval_key'][:16]):
            raise RuntimeError(f'{cell}: {ed} is not the W118 summary eval key {r["eval_key"][:16]}')
        rep = {'level': 'W118 summary report (last = cap)', 'Q': r['Q_at_cap'], 'salvage': r['terminal_salvage_value_last'],
               'Q_net': r['net_operational_recourse_last'], 'end': r['k_cap'], 'cycles_run': r['cycles_run'],
               'last_equals_cap': r['cycles_run'] == r['k_cap']}
        return ed, os.path.join(root, MANIFEST_NAME), r['k_cap'], rep, ref['name']
    doc = v6 if summary == 'v6' else ext
    r = doc['reports'][cell]
    certified = r['status'] == 'certified'
    rep = {'level': 'summary report', 'Q': r['Q_end'], 'salvage': r['terminal_salvage_value_end'], 'Q_net': r['Q_net_end'],
           'end': r['end_cycle'], 'status': r['status'],
           'Q_k_star_equals_Q_end': (r['Q_k_star'] == r['Q_end']) if certified else None,
           'end_is_k_star_or_cap': r['end_cycle'] == (r['k_star'] if certified else r['k_cap'])}
    if cell == W153.D_FROM_RECORDS['cell']:
        cert = W153._load(W153.D_FROM_RECORDS['certificate'], W153.D_FROM_RECORDS['certificate_manifest'])
        ed = cert['source_run']['eval_dir']
        man = os.path.join(W153.D_FROM_RECORDS['campaign_root'], MANIFEST_NAME)
        return ed, man, r['end_cycle'], rep, 'v6 from records (the v5 run 51280961 read through k* = 150)'
    if summary == 'ext':
        root = os.path.join(W153.EXT_ROOT, f'campaign_s53_w142_resettle_ext_v6_{cell}')
        note = 'ext spec v3 84775dc4'
    elif cell in W153.NOT_V6_STAGE_CELLS:
        spec = v6['certifying_spec_per_cell'][cell]['stage_spec']['path']
        sdir = os.path.dirname(spec)
        root = os.path.join(sdir, f'campaign_s53_{os.path.basename(sdir)}_{cell}')
        note = f"certifying spec {os.path.basename(spec)}"
    else:
        root = os.path.join(W153.V6_ROOT, f'campaign_s53_w142_resettle_v6_{cell}')
        note = 'v6 stage spec 96c23404'
    ed = _one_eval_dir(root, cell)
    if r.get('eval_key') and not os.path.basename(ed).startswith(r['eval_key'][:16]):
        raise RuntimeError(f'{cell}: {ed} is not the report eval key {r["eval_key"][:16]}')
    return ed, os.path.join(root, MANIFEST_NAME), r['end_cycle'], rep, note


def salvage_identity(w153_out, v6, ext):
    checks, out = {}, {}
    used = []
    for r in w153_out['claims']:
        for cell in r['ref_cells'] + r['other_cells']:
            if (r['summary'], cell) not in used:
                used.append((r['summary'], cell))
    view_rows = {(c['summary'], c['cell']): c for c in w153_out['cells']}
    rows = []
    for summary, cell in used:
        view = ((v6 if summary == 'v6' else ext)['references'][cell[4:]] if cell.startswith('ref:')
                else (v6 if summary == 'v6' else ext)['reports'][cell]['view'])
        ed, man, end, rep, note = _cell_source(cell, summary, v6, ext)
        pcr = {x['cycle']: x for x in W153._jsonl(os.path.join(ed, PCR), man)}
        row = pcr[end]
        raw_res = row['gross_operational_cost'] - row['terminal_salvage_value'] - row['recourse']
        w = view_rows[(summary, cell)]
        rows.append({
            'summary': summary, 'cell': cell, 'source': note, 'eval_dir': ed, 'status': view['status'], 'end_cycle': end,
            'Q_view': view['Q'], 'Q_net_view': view['Q_net'], 'salvage_view': view['salvage'],
            'a_view_Q_minus_Q_net_minus_salvage_w153': w['gross_minus_net_minus_salvage'],
            'b_report_level': rep['level'],
            'b_report_Q_minus_salvage_minus_Q_net': rep['Q'] - rep['salvage'] - rep['Q_net'],
            'b_report_values_equal_view': (rep['Q'], rep['salvage'], rep['Q_net']) == (view['Q'], view['salvage'],
                                                                                         view['Q_net']),
            'b_report_extra': {k: v for k, v in rep.items() if k not in ('Q', 'salvage', 'Q_net', 'level')},
            'c_raw_row_cycle': row['cycle'],
            'c_raw_gross_minus_salvage_minus_recourse': raw_res,
            'c_raw_row_equals_view': (row['gross_operational_cost'] == view['Q'] and row['recourse'] == view['Q_net']
                                      and row['terminal_salvage_value'] == view['salvage']),
            'salvage_nil': abs(view['salvage']) < W153.ABS_SALVAGE_NEGLIGIBLE_EUR})
    checks['i_49_cells_read_by_the_60_claims'] = len({c for _s, c in used}) == 49
    checks['i_a_view_identity_all'] = all(abs(r['a_view_Q_minus_Q_net_minus_salvage_w153']) < IDENTITY_TOL_EUR for r in rows)
    checks['i_b_report_identity_all'] = all(abs(r['b_report_Q_minus_salvage_minus_Q_net']) < IDENTITY_TOL_EUR for r in rows)
    checks['i_b_report_values_equal_view_all'] = all(r['b_report_values_equal_view'] for r in rows)
    checks['i_b_certified_Q_k_star_equals_Q_end'] = all(r['b_report_extra'].get('Q_k_star_equals_Q_end') in (True, None)
                                                        for r in rows)
    checks['i_b_end_is_k_star_or_cap'] = all(r['b_report_extra'].get('end_is_k_star_or_cap') in (True, None) for r in rows)
    checks['i_b_w118_last_equals_cap'] = all(r['b_report_extra'].get('last_equals_cap') in (True, None) for r in rows)
    checks['i_c_raw_identity_all'] = all(abs(r['c_raw_gross_minus_salvage_minus_recourse']) < IDENTITY_TOL_EUR for r in rows)
    checks['i_c_raw_row_equals_view_all'] = all(r['c_raw_row_equals_view'] for r in rows)
    out['cells'] = rows
    out['counts'] = {
        'summary_cell_pairs': len(rows), 'distinct_cells': len({r['cell'] for r in rows}),
        'cells_with_salvage': sorted({r['cell'] for r in rows if not r['salvage_nil']}),
        'n_cells_with_salvage': len({r['cell'] for r in rows if not r['salvage_nil']}),
        'max_abs_a_view': max(abs(r['a_view_Q_minus_Q_net_minus_salvage_w153']) for r in rows),
        'max_abs_b_report': max(abs(r['b_report_Q_minus_salvage_minus_Q_net']) for r in rows),
        'max_abs_c_raw': max(abs(r['c_raw_gross_minus_salvage_minus_recourse']) for r in rows),
        'n_a_pass': sum(abs(r['a_view_Q_minus_Q_net_minus_salvage_w153']) < IDENTITY_TOL_EUR for r in rows),
        'n_b_pass': sum(abs(r['b_report_Q_minus_salvage_minus_Q_net']) < IDENTITY_TOL_EUR for r in rows),
        'n_c_pass': sum(abs(r['c_raw_gross_minus_salvage_minus_recourse']) < IDENTITY_TOL_EUR for r in rows),
        'n_c_row_equals_view': sum(r['c_raw_row_equals_view'] for r in rows)}
    out['w153_checks_rerun'] = w153_out.get('_checks')
    out['definitions'] = {
        'a': 'view Q - Q_net - salvage (W153.salvage_row, called; its `cells` list)',
        'b': ('report level: Q_end - terminal_salvage_value_end - Q_net_end (Q_end = Q_k* certified, Q at the cap '
              'otherwise); W118 references: Q_at_cap - terminal_salvage_value_last - net_operational_recourse_last '
              '(last = cap checked); W101 references: their per-cycle row at k*'),
        'c': ('the cell\'s per_cycle_record.jsonl row at the end cycle (sha-verified against the campaign manifest): '
              'gross_operational_cost - terminal_salvage_value - recourse; and the row equal to the view (bitwise)'),
        'tolerance_eur': IDENTITY_TOL_EUR}
    return checks, out


# ======================================================================================================================
#  3(a). the 20 A1b G:*:net rows of W117
# ======================================================================================================================
def _old_cell(eval_dir, man):
    """the W117 cell fields of an old (Phase A) certificate, from its sha-verified records."""
    rec = W153._load(os.path.join(eval_dir, 'evaluation_record.json'), man)
    det = W153._load(os.path.join(eval_dir, DETAIL), man)
    q = rec['terminal_gross_operational_cost']
    t = det['t_tso_plus_t_dso_terminal']
    return rec, {'eval_key': rec['eval_key'], 'eval_dir': eval_dir, 'candidate_key': rec['candidate_key'],
                 'label': rec.get('candidate_label'), 'certification_cycle': rec['certification_cycle'], 'Q': q,
                 't_sum': t, 'Q_cc': q + t, 'salvage': rec['recourse_components']['terminal_salvage_value'],
                 'Q_net': rec['terminal_net_operational_recourse']}


def a1b_net_rows(v6):
    checks, out = {}, {}
    w117 = W153._load(W117_OUT['path'], W117_OUT['manifest'])
    w112 = W153._load(W112_OUT['path'], W112_OUT['manifest'])
    S, thr_old = w117['rule']['S_eur'], w117['rule']['old_threshold_eur']
    checks['a_S_equals_w112_s_gross'] = S == w112['task5']['references']['unit']['s_gross']
    checks['a_old_threshold_is_3S'] = abs(3.0 * S - thr_old) < 1e-6
    rec117 = {c['claim_id']: c for c in w117['claims'] if c['claim_id'].startswith('G:') and c['claim_id'].endswith(':net')}
    checks['a_w117_has_20_G_net_rows'] = len(rec117) == 20

    a1b_man = os.path.join(A1B_ROOT, MANIFEST_NAME)
    a1b = W153._load(os.path.join(A1B_ROOT, 'campaign_results.json'), a1b_man)
    pts = W117._points(a1b)
    x0rec, x0 = _old_cell(X0_OLD_EVAL_DIR, os.path.join(A0C7_ROOT, MANIFEST_NAME))
    checks['a_x0_certified_cost_is_terminal_gross'] = x0rec['certified_cost'] == x0['Q']
    checks['a_x0_identity_Q_minus_salvage_minus_Q_net'] = abs(x0['Q'] - x0['salvage'] - x0['Q_net']) < IDENTITY_TOL_EUR
    v6_by_id = {c['claim_id']: c for c in v6['claims_scored']}
    rows = []
    for p in pts:
        cid = f"G:{p['label']}:net"
        r117 = rec117[cid]
        _rec_o, o = _old_cell(p['eval_dir'], a1b_man)
        I = p['I_x_eur']
        # the A1b inputs (campaign_results) and their consistency with the eval record W117 read
        q_o, qn_o, sv_o = p['certified_cost_gross_settlement_excluded'], p['net_operational_recourse'], p['terminal_salvage_value']
        consistency = {'Q_campaign_equals_eval_record': q_o == o['Q'] == p['terminal_gross_operational_cost'],
                       'salvage_campaign_equals_eval_record': sv_o == o['salvage'],
                       'Q_net_campaign_equals_eval_record': qn_o == o['Q_net'],
                       'F_eur_minus_I_minus_Q': p['F_eur'] - I - q_o,
                       'Q_minus_salvage_minus_Q_net': q_o - sv_o - qn_o,
                       'status_certified': p['status'] == 'certified'}
        # the W153 net form on the A1b inputs
        d_net = W153._form_d('F', 0.0, I, x0['Q_net'], qn_o)
        d_net_cc = W153._form_d('F', 0.0, I, x0['Q_net'] + x0['t_sum'], qn_o + o['t_sum'])
        d_gross = W153._form_d('F', 0.0, I, x0['Q'], q_o)
        literal = p['F_eur'] - x0['Q'] - sv_o       # W117's recorded-source formula, on the same inputs
        # W117's own verdict rule, called on the same inputs (its own form inside)
        claim = {'item': 'G', 'claim_id': cid, 'statement': r117['statement'], 'claim_type': 'sign', 'form': 'F',
                 'ref': x0, 'other': o, 'I_ref': 0.0, 'I_other': I, 'I_source': r117['I_source'],
                 'net_of_salvage': True, 'ref_kind': 'old', 'other_kind': 'old', 'recorded': literal,
                 'recorded_source': r117['recorded_source'], 'note': None}
        ev = W117.evaluate(claim, S, thr_old)
        dist_bar = min(abs(ev['margin_Q'] - ev['bar_R1']), abs(ev['margin_Qcc'] - ev['bar_R1']),
                       abs(ev['margin_Q'] - ev['bar_R0']), abs(ev['margin_Qcc'] - ev['bar_R0']))
        v6c = v6_by_id.get(cid)
        v6_beside = None
        if v6c is not None:
            n = v6c['net_of_salvage']
            v6_beside = {'d_net_v6': n['d_net'], 'verdict_net_v6': n['resolution']['verdict'],
                         'threshold_v6': n['resolution'].get('threshold', n['resolution'].get('bar')),
                         'rule_v6': n['resolution'].get('rule'), 'primary_verdict_v6': v6c['verdict'],
                         'other_Q_v6': v6c['other']['Q'], 'other_salvage_v6': v6c['other']['salvage'],
                         'note': ('a DIFFERENT instance: the v6 cell is the re-settled run (C2 ageing baseline, certified '
                                  'under v6) of this year-ladder point; the A1b row is the C3-era Phase A certificate')}
        rows.append({
            'claim_id': cid, 'label': p['label'], 'investment_year': p['investment_year'],
            'candidate_key_other': p['candidate_key'], 'eval_key_other': p['eval_key'],
            'eval_key_ref_x0': x0['eval_key'], 'candidate_key_ref_x0': x0['candidate_key'],
            'inputs': {'F_eur': p['F_eur'], 'I_x_eur': I, 'Q_other_gross': q_o, 'Q_net_other': qn_o,
                       'terminal_salvage_value_other': sv_o, 't_sum_other': o['t_sum'],
                       'Q0_gross': x0['Q'], 'Q_net_0': x0['Q_net'], 'salvage_0': x0['salvage'], 't_sum_0': x0['t_sum']},
            'input_consistency': consistency,
            'd_net_w153_form': d_net, 'd_net_cc_w153_form': d_net_cc, 'd_gross_w153_form': d_gross,
            'w117_recorded': r117['recorded'], 'w117_recorded_source': r117['recorded_source'],
            'w117_d_Q_net': r117['d_Q'], 'w117_d_Qcc_net': r117['d_Qcc'],
            'abs_diff_vs_recorded': abs(d_net - r117['recorded']), 'rel_diff_vs_recorded': _rel(d_net, r117['recorded']),
            'abs_diff_vs_w117_d_Q': abs(d_net - r117['d_Q']), 'rel_diff_vs_w117_d_Q': _rel(d_net, r117['d_Q']),
            'abs_diff_cc_vs_w117_d_Qcc': abs(d_net_cc - r117['d_Qcc']),
            'rel_diff_cc_vs_w117_d_Qcc': _rel(d_net_cc, r117['d_Qcc']),
            'within_vs_recorded': _within(d_net, r117['recorded']), 'within_vs_w117_d_Q': _within(d_net, r117['d_Q']),
            'within_cc_vs_w117_d_Qcc': _within(d_net_cc, r117['d_Qcc']),
            'literal_recorded_formula_on_inputs': literal,
            'literal_equals_w117_recorded': literal == r117['recorded'],
            'salvage_effect_d_net_minus_d_gross': d_net - d_gross,
            'w117_rule': {'verdict_R1_recomputed': ev['verdict_R1'], 'verdict_R1_recorded': r117['verdict_R1'],
                          'verdict_R0_recomputed': ev['verdict_R0'], 'verdict_R0_recorded': r117['verdict_R0'],
                          'bar_R1': ev['bar_R1'], 'bar_R1_recorded': r117['bar_R1'], 'bar_R0': ev['bar_R0'],
                          'margin_Q': ev['margin_Q'], 'margin_Qcc': ev['margin_Qcc'],
                          'margin_Q_over_bar_R1': ev['margin_Q_over_bar_R1'],
                          'evaluate_d_Q_equals_recorded_d_Q': ev['d_Q'] == r117['d_Q'],
                          'abs_evaluate_d_minus_w153_d': abs(ev['d_Q'] - d_net),
                          'min_distance_of_a_margin_to_a_bar': dist_bar,
                          'verdict_insensitive_to_the_form_difference': dist_bar > abs(ev['d_Q'] - d_net)
                          + abs(ev['d_Qcc'] - d_net_cc)},
            'v6_beside': v6_beside})
    checks['a_20_rows'] = len(rows) == 20 and {r['claim_id'] for r in rows} == set(rec117)
    checks['a_inputs_consistent'] = all(
        r['input_consistency']['Q_campaign_equals_eval_record'] and r['input_consistency']['salvage_campaign_equals_eval_record']
        and r['input_consistency']['Q_net_campaign_equals_eval_record'] and r['input_consistency']['status_certified']
        and abs(r['input_consistency']['Q_minus_salvage_minus_Q_net']) < IDENTITY_TOL_EUR
        and abs(r['input_consistency']['F_eur_minus_I_minus_Q']) < IDENTITY_TOL_EUR for r in rows)
    checks['a_reproduces_w117_recorded_within_0.01_or_1e-6_rel'] = all(r['within_vs_recorded'] for r in rows)
    checks['a_reproduces_w117_d_Q_within_0.01_or_1e-6_rel'] = all(r['within_vs_w117_d_Q'] for r in rows)
    checks['a_reproduces_w117_d_Qcc_within_0.01_or_1e-6_rel'] = all(r['within_cc_vs_w117_d_Qcc'] for r in rows)
    checks['a_literal_formula_equals_w117_recorded'] = all(r['literal_equals_w117_recorded'] for r in rows)
    checks['a_w117_verdicts_reproduced_R1_R0'] = all(
        r['w117_rule']['verdict_R1_recomputed'] == r['w117_rule']['verdict_R1_recorded']
        and r['w117_rule']['verdict_R0_recomputed'] == r['w117_rule']['verdict_R0_recorded']
        and r['w117_rule']['bar_R1'] == r['w117_rule']['bar_R1_recorded'] for r in rows)
    checks['a_w117_evaluate_d_equals_recorded_d'] = all(r['w117_rule']['evaluate_d_Q_equals_recorded_d_Q'] for r in rows)
    checks['a_w117_verdict_insensitive_to_form_difference'] = all(
        r['w117_rule']['verdict_insensitive_to_the_form_difference'] for r in rows)
    out['rows'] = rows
    out['summary'] = {
        'n': len(rows),
        'max_abs_diff_vs_recorded': max(r['abs_diff_vs_recorded'] for r in rows),
        'max_rel_diff_vs_recorded': max(r['rel_diff_vs_recorded'] for r in rows),
        'max_abs_diff_vs_w117_d_Q': max(r['abs_diff_vs_w117_d_Q'] for r in rows),
        'max_abs_diff_cc_vs_w117_d_Qcc': max(r['abs_diff_cc_vs_w117_d_Qcc'] for r in rows),
        'n_within_vs_recorded': sum(r['within_vs_recorded'] for r in rows),
        'w117_verdicts_R1': {v: sum(r['w117_rule']['verdict_R1_recomputed'] == v for r in rows)
                             for v in sorted({r['w117_rule']['verdict_R1_recomputed'] for r in rows})},
        'n_with_v6_counterpart': sum(r['v6_beside'] is not None for r in rows),
        'salvage_effect_range': [min(r['salvage_effect_d_net_minus_d_gross'] for r in rows),
                                 max(r['salvage_effect_d_net_minus_d_gross'] for r in rows)]}
    out['definitions'] = {
        'source': (f"{W117_OUT['path']} (commit {W117_OUT['commit']}): G:*:net rows, recorded = S45 A1b campaign_results "
                   'F_eur - Q(0) - terminal_salvage_value; d_Q / d_Qcc = W117 evaluate on the eval records'),
        'w153_net_form': "W153._form_d('F', I_ref=0, I_other=I(x), Q_net(x0), Q_net(x)); d_net_cc the same on Q_net + t_sum",
        'inputs': (f'{A1B_ROOT}/campaign_results.json points[] (F_eur, I_x_eur, certified gross, net_operational_recourse, '
                   f'terminal_salvage_value); x = 0: {X0_OLD_EVAL_DIR}/evaluation_record.json (the W117 x0 cell 7aa017f0); '
                   f't_sum: each eval dir {DETAIL} t_tso_plus_t_dso_terminal'),
        'w117_rule': w117['rule'],
        'reproduction_rule': f'|diff| <= {REPRO_ABS_EUR} EUR or <= {REPRO_REL} relative',
        'v6_beside': 'v6 summary claims_scored entry of the same claim id, where one exists (net d and net verdict)'}
    return checks, out


# ======================================================================================================================
#  3(b). the W118 year-ladder net
# ======================================================================================================================
def _prose_rows():
    path = PROSE['path']
    sha = hashlib.sha256(open(os.path.join(REPO, path), 'rb').read()).hexdigest()
    W153.INPUTS[path] = sha
    lines = open(os.path.join(REPO, path), encoding='utf-8').read().split('\n')
    lo, hi = PROSE['lines']
    block = lines[lo - 1:hi]

    def num(s):
        s = s.replace('*', '').replace('−', '-').replace(',', '').strip()
        return float(s.split()[0])
    cells = {}
    for ln in block:
        parts = [x.strip() for x in ln.strip().strip('|').split('|')]
        if parts and parts[0] in ('2030', '2035'):
            cells[parts[0]] = {'M_gross': num(parts[1]), 'salvage': num(parts[2]), 'M_net': num(parts[3])}
        elif parts and '2035' in parts[0] and '2030' in parts[0]:
            cells['D'] = {'D_gross': num(parts[1]), 'D_net': num(parts[3]), 'text_gross': parts[1], 'text_net': parts[3]}
    tracked_clean = (subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', path], cwd=REPO).returncode == 0)
    return {'path': path, 'sha256': sha, 'lines': [lo, hi], 'text': block, 'parsed': cells,
            'unmodified_against_HEAD': tracked_clean,
            'last_commit': _git(['log', '-1', '--format=%h', '--', path]).strip()}


def w118_year_ladder(v6):
    checks, out = {}, {}
    w118 = W153._load(L132.W118_SUMMARY, L132.W118_SUMMARY_MANIFEST)
    x0 = v6['references']['7aa017f0']
    dif = w118['differences']
    checks['b_x0_view_Q_equals_w118_comparator'] = x0['Q'] == dif['x0_comparator']['Q181']
    checks['b_x0_identity'] = abs(x0['Q'] - x0['Q_net'] - x0['salvage']) < IDENTITY_TOL_EUR
    per = {}
    views = {}
    for y in ('2030', '2035'):
        r, dd = w118['reports'][f'yl_y{y}'], dif['phase_b_and_year_ladder_vs_x0'][f'yl_y{y}']
        q, qn, sv, t, I = r['Q_k_star'], r['net_operational_recourse_last'], r['terminal_salvage_value_last'], \
            r['t_sum_k_star'], dd['I_j']
        views[y] = {'status': r['status'], 'Q': q, 'Q_net': qn, 'salvage': sv, 't': t, 'band': r['band_width'], 'I': I}
        m_g = W153._form_d('F', 0.0, I, x0['Q'], q)
        m_n = W153._form_d('F', 0.0, I, x0['Q_net'], qn)
        per[y] = {'eval_key': r['eval_key'], 'candidate_key': r['candidate_key'], 'k_star': r['k_star'],
                  'cycles_run': r['cycles_run'], 'status': r['status'], 'I_j': I, 'Q_k_star': q, 'Q_net_last': qn,
                  'salvage_last': sv, 'identity_Q_minus_salvage_minus_Q_net': q - sv - qn,
                  'M_gross_w153_form': m_g, 'M_gross_recorded_w118': dd['M'], 'abs_diff_M_gross': abs(m_g - dd['M']),
                  'M_net_w153_form': m_n, 'M_net_equals_M_gross_minus_salvage': abs((m_g - sv) - m_n)}
        checks[f'b_{y}_last_is_k_star'] = r['cycles_run'] == r['k_star']
        checks[f'b_{y}_identity'] = abs(q - sv - qn) < IDENTITY_TOL_EUR
        checks[f'b_{y}_M_gross_reproduced_1e-6'] = abs(m_g - dd['M']) < IDENTITY_TOL_EUR
    a, b = views['2030'], views['2035']
    d_g = W153._form_d('F', a['I'], b['I'], a['Q'], b['Q'])
    d_g_cc = W153._form_d('F', a['I'], b['I'], a['Q'] + a['t'], b['Q'] + b['t'])
    d_n = W153._form_d('F', a['I'], b['I'], a['Q_net'], b['Q_net'])
    d_n_cc = W153._form_d('F', a['I'], b['I'], a['Q_net'] + a['t'], b['Q_net'] + b['t'])
    yl = dif['year_ladder']
    res_g_w118 = L132.resolve(d_g, d_g_cc, (a, b))
    res_n_w118 = L132.resolve(d_n, d_n_cc, (a, b))
    res_n_v6 = DET.difference_certified_pair(d_n, d_n_cc, a, b)
    checks['b_D_gross_reproduced_1e-6'] = abs(d_g - yl['D_2035_minus_2030']) < IDENTITY_TOL_EUR
    checks['b_D_gross_cc_reproduced_1e-6'] = abs(d_g_cc - yl['D_cc']) < IDENTITY_TOL_EUR
    checks['b_w118_rule_resolution_reproduced'] = (abs(res_g_w118['resolution'] - yl['resolution']) < IDENTITY_TOL_EUR
                                                   and res_g_w118['verdict'] == yl['verdict'])
    prose = _prose_rows()
    pp = prose['parsed']
    checks['b_prose_unmodified_against_HEAD'] = prose['unmodified_against_HEAD']
    checks['b_prose_table_parsed'] = set(pp) == {'2030', '2035', 'D'}

    def r1(x):
        return float(f'{x:.1f}')
    cmp = {}
    if checks['b_prose_table_parsed']:
        for y in ('2030', '2035'):
            cmp[y] = {'M_gross': {'full': per[y]['M_gross_w153_form'], 'at_0.1': r1(per[y]['M_gross_w153_form']),
                                  'prose': pp[y]['M_gross']},
                      'salvage': {'full': per[y]['salvage_last'], 'at_0.1': r1(per[y]['salvage_last']),
                                  'prose': pp[y]['salvage']},
                      'M_net': {'full': per[y]['M_net_w153_form'], 'at_0.1': r1(per[y]['M_net_w153_form']),
                                'prose': pp[y]['M_net'],
                                'prose_arithmetic_on_rounded_components': r1(r1(per[y]['M_gross_w153_form'])
                                                                             - r1(per[y]['salvage_last']))}}
        cmp['D'] = {'D_gross': {'full': d_g, 'at_0.1': r1(d_g), 'prose': pp['D']['D_gross']},
                    'D_net': {'full': d_n, 'at_0.1': r1(d_n), 'prose': pp['D']['D_net'],
                              'prose_arithmetic_on_rounded_components': r1(
                                  r1(r1(per['2035']['M_gross_w153_form']) - r1(per['2035']['salvage_last']))
                                  - r1(r1(per['2030']['M_gross_w153_form']) - r1(per['2030']['salvage_last'])))}}
        flat = [v for y in ('2030', '2035') for v in cmp[y].values()] + list(cmp['D'].values())
        for v in flat:
            v['abs_full_minus_prose'] = abs(v['full'] - v['prose'])
            v['correctly_rounded_equals_prose'] = v['at_0.1'] == v['prose']
            if 'prose_arithmetic_on_rounded_components' in v:
                v['prose_equals_arithmetic_on_rounded_components'] = v['prose_arithmetic_on_rounded_components'] == v['prose']
        checks['b_every_prose_figure_within_one_prose_unit_0.1'] = all(v['abs_full_minus_prose'] < PROSE_UNIT_EUR
                                                                       for v in flat)
        checks['b_prose_net_figures_equal_arithmetic_on_rounded_components'] = all(
            v.get('prose_equals_arithmetic_on_rounded_components', True) for v in flat)
    out['per_year'] = per
    out['difference_2035_minus_2030'] = {
        'D_gross_w153_form': d_g, 'D_gross_recorded_w118': yl['D_2035_minus_2030'], 'D_gross_cc_w153_form': d_g_cc,
        'D_net_w153_form': d_n, 'D_net_cc_w153_form': d_n_cc,
        'D_net_minus_D_gross': d_n - d_g, 'salvage_only_term': a['salvage'] - b['salvage'],
        'resolution_w118_rule_gross': res_g_w118, 'resolution_w118_rule_net': res_n_w118,
        'resolution_v6_rule_net_report_only': res_n_v6}
    out['prose'] = prose
    out['prose_comparison'] = cmp
    out['definitions'] = {
        'M': "W153._form_d('F', 0, I_j, Q(x0), Q(yl)) = I_j + Q(yl) - Q181 (net: Q_net in place of Q)",
        'D': "W153._form_d('F', I_2030, I_2035, Q(yl_2030), Q(yl_2035)) = F(2035) - F(2030) (net: Q_net)",
        'x0': "v6 summary references['7aa017f0'] (the W101 d110bd1a Q181 view; salvage 0)",
        'w118_inputs': f'{L132.W118_SUMMARY} reports yl_y2030 / yl_y2035 and differences',
        'w118_rule': 'L132.resolve (settled_vs_settled: resolution = sum of the two band widths, strict >)',
        'prose_precision': ('the report prints 0.1 EUR; "at_0.1" is the full figure correctly rounded; the prose net '
                            'column is compared also with the same arithmetic done on the 0.1-rounded components')}
    return checks, out


# ======================================================================================================================
#  main / manifest
# ======================================================================================================================
def _guards_report():
    return {name: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for name, g in GUARDS}


def _untracked(paths):
    tracked = set(_git(['ls-files', '--'] + sorted(paths)).split('\n'))
    return sorted(p for p in paths if p not in tracked)


def main():
    t0 = time.time()
    dry = '--dry' in sys.argv[1:]
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    if not dry:
        if not os.path.isdir(out_dir):
            raise RuntimeError(f'{OUT_DIR_REL} must exist (created by the launch command, which writes launch.log there)')
        for n in (OUT_JSON, OUT_MAN):
            if os.path.exists(os.path.join(out_dir, n)):
                raise RuntimeError(f'{n} exists in {OUT_DIR_REL}: write-once, refusing')
    head = _git(['rev-parse', 'HEAD']).strip()
    porcelain = _git(['status', '--porcelain', '--untracked-files=no'])
    _log(f'{STAGE}; git HEAD {head}; tracked changes: {porcelain.strip() or "none"}; dry {dry}')
    results, all_checks, errors = {}, {}, {}
    w153_out = None
    try:
        w153_ck, w153_out = W153.salvage_row()       # the committed W153 section, called (read-only)
        w153_out['_checks'] = w153_ck
        all_checks['w153_salvage_row_rerun'] = {f'w153_{k}': v for k, v in w153_ck.items()}
    except Exception:
        errors['w153_salvage_row'] = traceback.format_exc()
        _log(f"ERROR in W153.salvage_row:\n{errors['w153_salvage_row']}")
        all_checks['w153_salvage_row_rerun'] = {'w153_salvage_row_ran': False}
    v6, ext = _summaries()
    plan = (('part1_gross_form_check', lambda: gross_form_check(w153_out['claims'])),
            ('part2_salvage_identity', lambda: salvage_identity(w153_out, v6, ext)),
            ('part3a_a1b_net_rows', lambda: a1b_net_rows(v6)),
            ('part3b_w118_year_ladder', lambda: w118_year_ladder(v6)))
    for key, fn in plan:
        try:
            if w153_out is None:
                raise RuntimeError('W153.salvage_row did not run')
            ck, doc = fn()
        except Exception:
            errors[key] = traceback.format_exc()
            _log(f'ERROR in {key}:\n{errors[key]}')
            ck, doc = {f'{key}_ran': False}, None
        results[key] = doc
        all_checks[key] = ck
    inputs = dict(sorted(W153.INPUTS.items()))
    untracked = _untracked(list(inputs))
    a1b_inputs = [p for p in inputs if p.startswith(A1B_ROOT) or p.startswith(A0C7_ROOT) or p.startswith(W117_OUT['path'])]
    all_checks['inputs'] = {'every_part3a_input_is_git_tracked': not [p for p in a1b_inputs if p in untracked],
                            'every_input_is_git_tracked': not untracked}
    guards = _guards_report()
    failed = {k: [n for n, v in ck.items() if not v] for k, ck in all_checks.items()}
    pickle_ok = PICKLE_COUNTS == {'load': 0, 'loads': 0} and W153.PICKLE_COUNTS == {'load': 0, 'loads': 0}
    ok = (not errors and not any(failed.values()) and pickle_ok
          and all(not v['verify_0_failures'] for v in guards.values()))
    doc = {'stage': STAGE, 'git_HEAD': head, 'git_status_porcelain_tracked': porcelain,
           'script': {'path': os.path.basename(__file__), 'sha256': W153._sha(os.path.basename(__file__))},
           'method_imported': {'w153': {'path': os.path.basename(W153.__file__), 'sha256': W153._sha(os.path.basename(W153.__file__)),
                                        'commit': '48e76c9f'},
                               'det': {'path': os.path.basename(DET.__file__), 'sha256': W153._sha(os.path.basename(DET.__file__))},
                               'w117': {'path': os.path.basename(W117.__file__), 'sha256': W153._sha(os.path.basename(W117.__file__))},
                               'l132': {'path': os.path.basename(L132.__file__), 'sha256': W153._sha(os.path.basename(L132.__file__))}},
           'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 64 (net-of-salvage check before any net figure enters a table)',
                         'Planner task W154b (substitute evidence; W154 scoped search)'],
           'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED (primary); Q_net = net_operational_recourse '
                                    '= Q - terminal salvage credit; Q_cc = Q + t_sum (diagnostic); EUR'),
           'results': results, 'checks': all_checks, 'failed_checks': failed, 'errors': errors,
           'solve_profile_guards': guards, 'pickle_guard': {'w154b': dict(PICKLE_COUNTS), 'w153': dict(W153.PICKLE_COUNTS)},
           'untracked_inputs': untracked, 'all_pass': ok, 'ended_utc': _utc(), 'inputs_sha256': inputs}
    if dry:
        GRIO.dumps(doc)          # the writer's refusal check, nothing written
        _log('dry run: nothing written')
    else:
        with open(os.path.join(out_dir, OUT_JSON), 'x') as handle:
            GRIO.dump(doc, handle, indent=1, sort_keys=False)
    _print(results)
    _log(f"checks failed: {failed}; errors: {list(errors)}; pickle {doc['pickle_guard']}; guards "
         f"{({k: v['verify_0_failures'] for k, v in guards.items()})}; inputs read {len(inputs)}; untracked inputs "
         f"{len(untracked)}; wall {time.time() - t0:.1f} s")
    _log(f'all_pass {ok}')
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(0 if ok else 1)


def _print(results):
    p1 = results.get('part1_gross_form_check')
    if p1:
        for f, v in p1['per_family'].items():
            _log(f"P1 {f}: n {v['n']} max|dd| {v['max_abs_diff_d']!r} max|dd_cc| {v['max_abs_diff_d_cc']!r} max|dthr| "
                 f"{v['max_abs_diff_threshold']!r} bitwise {v['n_d_bitwise']} verdicts {v['n_verdict_match']}/{v['n']} "
                 f"dict-equal {v['n_resolution_dict_equal']} rules {v['rules']} recorded {v['verdicts_recorded']} "
                 f"salvage-term max {v['max_abs_net_minus_gross_minus_salvage_term']!r} pass {v['pass']}")
    p2 = results.get('part2_salvage_identity')
    if p2:
        _log(f"P2 counts {p2['counts']}")
    p3 = results.get('part3a_a1b_net_rows')
    if p3:
        for r in p3['rows']:
            w = r['w117_rule']
            _log(f"P3a {r['claim_id']}: d_net {r['d_net_w153_form']:.6f} recorded {r['w117_recorded']:.6f} (|d| "
                 f"{r['abs_diff_vs_recorded']:.3g}) W117 d_Q {r['w117_d_Q_net']:.6f} (|d| {r['abs_diff_vs_w117_d_Q']:.3g}) "
                 f"cc |d| {r['abs_diff_cc_vs_w117_d_Qcc']:.3g}; W117 R1 {w['verdict_R1_recomputed']} (rec "
                 f"{w['verdict_R1_recorded']}) R0 {w['verdict_R0_recomputed']} bar {w['bar_R1']:.2f} m/bar "
                 f"{w['margin_Q_over_bar_R1']:.3f}; v6 "
                 f"{(r['v6_beside']['d_net_v6'], r['v6_beside']['verdict_net_v6']) if r['v6_beside'] else None}")
        _log(f"P3a summary {p3['summary']}")
    p4 = results.get('part3b_w118_year_ladder')
    if p4:
        for y, v in p4['per_year'].items():
            _log(f"P3b {y}: M gross {v['M_gross_w153_form']:.6f} (W118 {v['M_gross_recorded_w118']:.6f}) salvage "
                 f"{v['salvage_last']:.6f} M net {v['M_net_w153_form']:.6f}")
        d = p4['difference_2035_minus_2030']
        _log(f"P3b D gross {d['D_gross_w153_form']:.6f} (W118 {d['D_gross_recorded_w118']:.6f}); D net "
             f"{d['D_net_w153_form']:.6f}; W118 rule net {d['resolution_w118_rule_net']['verdict']} (res "
             f"{d['resolution_w118_rule_net']['resolution']:.2f}); v6 rule net {d['resolution_v6_rule_net_report_only']['verdict']} "
             f"(thr {d['resolution_v6_rule_net_report_only']['threshold']:.2f})")
        _log(f"P3b prose comparison {json.dumps(p4['prose_comparison'])}")


def write_manifest():
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    if os.path.exists(os.path.join(out_dir, OUT_MAN)):
        raise RuntimeError(f'{OUT_MAN} exists: write-once, refusing')
    with open(os.path.join(out_dir, OUT_JSON)) as handle:
        inputs = json.load(handle)['inputs_sha256']
    files = [os.path.basename(__file__), os.path.join(OUT_DIR_REL, OUT_JSON), os.path.join(OUT_DIR_REL, LAUNCH_LOG)]
    man = {rel: W153._sha(rel) for rel in files}
    man.update(inputs)
    with open(os.path.join(out_dir, OUT_MAN), 'x') as handle:
        GRIO.dump(dict(sorted(man.items())), handle, indent=1)
    gr = _guards_report()
    _log(f'manifest: {len(man)} entries ({len(files)} produced, {len(inputs)} inputs); guards '
         f"{({k: v['verify_0_failures'] for k, v in gr.items()})}; pickle {PICKLE_COUNTS} / {W153.PICKLE_COUNTS}")
    ok = (PICKLE_COUNTS == {'load': 0, 'loads': 0} and W153.PICKLE_COUNTS == {'load': 0, 'loads': 0}
          and all(not v['verify_0_failures'] for v in gr.values()))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    if '--manifest' in sys.argv[1:]:
        write_manifest()
    else:
        main()
