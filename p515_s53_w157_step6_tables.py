"""P5.15 Addendum 64, Planner task W157 item (i) -- THE STEP 6 TABLE BUILDER (draft for the Planner's assembly).
ZERO SOLVES, NO MODEL LOADS. A NEW FILE: no harness, scorer or production module is edited.

WHAT IT DOES. Reads COMMITTED summaries only (each input must be tracked and clean against HEAD; its sha256 is recorded
and, where the input has a committed manifest, checked against that manifest) and writes the Step 6 results tables as
JSON and Markdown:
  T1 claims      -- every claim row (60 = 48 of the v6 summary + 12 of the extension summary): gross primary with net
                    beside, Q_cc report-only, the bar or threshold, the multiple and the verdict, the certification
                    status of each cell and the instance keys; the Addendum 64 rulings applied (below);
  T2 cells       -- the certification status of every cell a claim names (k* or the uncertified cause; range / tau;
                    flags for >= 0.95 tau and for a turning point on a non-clean cycle; the replay gate; instance keys);
  T3 break-even  -- the W145 banded fit as committed, and the conservative fit with d_4a82a64a ALSO as an interval
                    (Addendum 64 ruling 4), computed by W145's own function `banded_breakeven_fit` on W145's committed
                    points (no code change: d_4a82a64a's view is passed as uncertified with gap = |t_sum| and
                    slack = |s|, so `W145.uncertified_bar` -> `L132.resolve` gives its bar); slope b + c/4 (ruling 3);
  T4 year ladder -- W154b's recorded figures (net -2,286.25), gross verdict under the v6 rule beside the recorded one;
  T5 Phase B     -- the W118 Phase B cells and pb_y2025_n5_v6, the v6 rule applied by `DET.resolve_v6`, recorded beside;
  T6 benchmark, T7 discount, T8 ageing arms, T9 dead zone -- the committed figures, re-tabulated;
  T10 A64        -- PLACEHOLDERS, clearly marked PENDING W156, for the m = 1.75 row and the soh_min 0.50 rows. With
                    `--include-a64` they are filled from the W155 summary after e_soh050 (NOT run by W157).

ADDENDUM 64 RULINGS APPLIED (each row carries its note):
  ruling 3  slope = b + c/4 (the 4 h marginal slope); the 5 % prediction scored at the midpoints, the corners reported
            as interval sensitivity;
  ruling 4  certificates on a non-clean turning point (j_a11d7966, d_4a82a64a) kept and flagged, the uncertified form
            beside every claim that names them; the manuscript figures are the conservative ones: break-even margin
            61.3 k EUR/MWh (d_4a82a64a also as an interval) and J 4 MWh at its uncertified bar;
  ruling 5  E:C2_calfade vs C3 reported within resolution (A61 prediction failed on it);
  A58 R3 /  the year-ladder net figure is -2,286.25 (W154b; the prose -2,286.2 was rounded-component arithmetic).
  W154b
NET LABEL. Every net figure is labelled "validated by form + salvage identity (W154b)", except the G rows, whose net
values are recorded in the claim records ("recorded (claim record net_of_salvage)").

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any project import and verified at exactly 0 at the end,
with every imported module's own zero-permit guard (W145's list); `pickle.load` / `pickle.loads` are blocked for the
whole run and every blocking counter verified at 0. The Addendum 64 spec is read from the git object store
(`git show HEAD:<path>`), never from the running campaign's directory.

MODES (repo root, canonical interpreter; attached, both streams captured; outputs opened 'x', never overwritten):
    mkdir -p data/SRP1/Results/P515S53/w157_step6_tables && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w157_step6_tables.py \\
        > data/SRP1/Results/P515S53/w157_step6_tables/launch.log 2>&1
  --include-a64 [--a64-summary REL]  fill T10 from the W155 summary (default
        data/SRP1/Results/P515S53/w155_a64_cells/w155_summary_after_04_e_soh050.json, committed clean) and write to
        data/SRP1/Results/P515S53/w157_step6_tables_a64/ (a NEW directory: the run without A64 is never overwritten).
Exit: 0 = every check holds and the guards are at 0; 3 = written, but a check failed (listed); 1 = harness fault or
precondition (nothing written).
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W157 Step 6 table builder (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W157: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W157: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402
import p515_s53_w145_banded_breakeven_fit as W145  # noqa: E402 -- the banded fit (arms its guards, blocks pickle)
import p515_s53_w142_determinacy as DET  # noqa: E402 -- the v6 scorer (resolve_v6, score_claim_v6)

L132 = W145.L132


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w157_step6_tables', GUARD),) + tuple(W145.GUARDS))

TAU = SC6.TAU
TWO_TAU = SC6.DETERMINACY_TAU_MULTIPLE * TAU
FLAG_RANGE_OVER_TAU = 0.95
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(S53, 'w157_step6_tables')
OUT_DIR_A64 = os.path.join(S53, 'w157_step6_tables_a64')
OUT_JSON = 'w157_step6_tables.json'
OUT_MD = 'w157_step6_tables.md'
OUT_MAN = 'manifest_sha256.json'
SCRIPT_REL = os.path.basename(__file__)
A64_SUMMARY_DEFAULT = os.path.join(S53, 'w155_a64_cells', 'w155_summary_after_04_e_soh050.json')
A64_SPEC_REL = os.path.join(S53, 'w155_a64_cells', 'frozen_s53_a64_cells_spec_v1_44a2dce8.json')
A64_SPEC_SHA = '44a2dce8c2bb12f45bcb907956acda971c574ec6676ce02d8a3caa720524e366'

OBJECTIVE_CONVENTION = ('Q = gross_operational_cost, settlement EXCLUDED (the frozen primary, Addendum 58 Ruling 3); '
                        'Q_net = net_operational_recourse = Q - terminal salvage credit (beside); Q_cc = Q + t_sum '
                        '(first-order consensus-consistent diagnostic, REPORT-ONLY); value = Q(0) - Q(x); F = Q + I; EUR')
NET_LABEL_VALIDATED = 'validated by form + salvage identity (W154b)'
NET_LABEL_RECORDED = 'recorded (claim record net_of_salvage)'
PENDING = 'PENDING W156'
FLAGGED_NCTP = ('j_a11d7966', 'd_4a82a64a')

# ---- inputs: (key, path, manifest or None) --------------------------------------------------------------------------
INPUTS = {
    'S6': (os.path.join(S53, 'w142_resettle_v6', 'w142_summary_after_34_l_195156fa.json'),
           os.path.join(S53, 'w142_resettle_v6', 'w142_summary_after_34_l_195156fa_manifest_sha256.json')),
    'SX': (os.path.join(S53, 'w142_resettle_ext_v6', 'w142_ext_summary_after_07_pb_y2025_n5_v6.json'),
           os.path.join(S53, 'w142_resettle_ext_v6', 'w142_ext_summary_after_07_pb_y2025_n5_v6_manifest_sha256.json')),
    'W145': (os.path.join(S53, 'w145_banded_fit', 'w145_banded_fit.json'),
             os.path.join(S53, 'w145_banded_fit', 'manifest_sha256.json')),
    'W153S': (os.path.join(S53, 'w153_step5_rows', 'w153_salvage_addback.json'),
              os.path.join(S53, 'w153_step5_rows', 'manifest_sha256.json')),
    'W153C': (os.path.join(S53, 'w153_step5_rows', 'w153_certification_statistics.json'),
              os.path.join(S53, 'w153_step5_rows', 'manifest_sha256.json')),
    'W153D': (os.path.join(S53, 'w153_step5_rows', 'w153_discount_row.json'),
              os.path.join(S53, 'w153_step5_rows', 'manifest_sha256.json')),
    'W154': (os.path.join(S53, 'w154_net_salvage_validation', 'w154_net_salvage_validation.json'),
             os.path.join(S53, 'w154_net_salvage_validation', 'manifest_sha256.json')),
    'W149': (os.path.join(S53, 'w149_h_f9eae48f_prices', 'w149_interface_price_read.json'),
             os.path.join(S53, 'w149_h_f9eae48f_prices', 'manifest_sha256.json')),
    'W118': (os.path.join(S53, 'w118_resettle', 'w118_resettle_summary.json'),
             os.path.join(S53, 'w118_resettle', 'w118_resettle_summary_manifest_sha256.json')),
    'BENCH': (os.path.join(S53, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json'),
              os.path.join(S53, 'w116_benchmark_nrf', 'report_v3', 'manifest_sha256.json')),
    'V6CHECKS': (os.path.join(S53, 'w142_resettle_v6', 'zero_solve_checks', 'w142_zero_solve_checks.json'),
                 os.path.join(S53, 'w142_resettle_v6', 'zero_solve_checks', 'w142_zero_solve_checks_manifest_sha256.json')),
    'V6SPEC': (os.path.join(S53, 'w142_resettle_v6', 'frozen_s53_resettle_spec_v6_96c23404.json'), None),
    # d_c52e1670's instance keys (its summary report carries none): the v6-from-records certificate the v6 summary cites
    'C52CERT': (os.path.join(S53, 'w142_resettle_v6', 'v6_from_records', 'd_c52e1670_v6_from_records_certificate.json'),
                os.path.join(S53, 'w142_resettle_v6', 'w142_summary_after_34_l_195156fa_manifest_sha256.json')),
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def load_inputs():
    """Every input committed clean, sha recorded, checked against its manifest where one exists."""
    docs, record, problems = {}, {}, []
    for key, (rel, man) in INPUTS.items():
        clean = L132._committed_clean(rel)
        sha = _sha(rel)
        ent = {'path': rel, 'sha256': sha, 'committed_clean': clean, 'manifest': man, 'manifest_sha256_matches': None}
        if not clean:
            problems.append(f'{rel} is not committed clean')
        if man is not None:
            if not L132._committed_clean(man):
                problems.append(f'{man} is not committed clean')
            m = json.load(open(os.path.join(REPO, man)))
            ent['manifest_sha256_matches'] = (m.get(rel) == sha)
            ent['manifest_file_sha256'] = _sha(man)
            if m.get(rel) != sha:
                problems.append(f'{rel} sha {sha[:8]} != its manifest entry {str(m.get(rel))[:8]}')
        record[key] = ent
        docs[key] = json.load(open(os.path.join(REPO, rel)))
    # the A64 spec from the git object store (never the running campaign's working files)
    blob = subprocess.run(['git', 'show', f'HEAD:{A64_SPEC_REL}'], cwd=REPO, capture_output=True)
    if blob.returncode != 0:
        problems.append(f'git show HEAD:{A64_SPEC_REL} failed')
        docs['A64SPEC'] = None
    else:
        sha = hashlib.sha256(blob.stdout).hexdigest()
        record['A64SPEC'] = {'path': A64_SPEC_REL, 'read_from': 'git object store (git show HEAD:path)', 'sha256': sha,
                             'sha256_matches_frozen_name': sha == A64_SPEC_SHA}
        if sha != A64_SPEC_SHA:
            problems.append(f'{A64_SPEC_REL} at HEAD sha {sha[:8]} != {A64_SPEC_SHA[:8]}')
        docs['A64SPEC'] = json.loads(blob.stdout)
    return docs, record, problems


# ======================================================================================================================
#  formatting
# ======================================================================================================================
def eur(x, nd=2):
    return '—' if x is None else f'{x:,.{nd}f}'


def mult(x):
    return '—' if x is None else f'{x:.2f}×'


def short(k, n=8):
    return '—' if not k else str(k)[:n]


# ======================================================================================================================
#  T2 -- cells
# ======================================================================================================================
def _turning_points(rep):
    return [t[0] for t in (rep.get('T') or [])]


def cells_table(docs):
    s6, sx, c153, d153, w118 = docs['S6'], docs['SX'], docs['W153C'], docs['W153D'], docs['W118']
    stats = {c['cell']: c for c in c153['result']['cells']}
    reports = dict(s6['reports'])
    reports.update(sx['reports'])
    rows = {}
    for cell, rep in reports.items():
        st = stats.get(cell)
        view = rep.get('view') or L132.view_from_report(rep)
        cert = rep.get('status') == 'certified'
        tps = _turning_points(rep)
        nc = rep.get('non_clean_cycles')
        if st is not None:
            nctp = list(st.get('turning_points_at_non_clean_cycle') or [])
            nctp_src = 'W153 certification statistics'
        elif nc is not None:
            nctp = sorted(set(tps) & set(nc))
            nctp_src = 'computed here: decision T cycles intersected with the report non_clean_cycles'
        else:
            nctp = None
            nctp_src = 'not recorded in the summary report (no non_clean_cycles field)'
        r_over_tau = rep.get('range_over_tau') if cert else None
        row = {
            'cell': cell, 'source_summary': 'S6' if cell in s6['reports'] else 'SX', 'item': rep.get('item'),
            'status': rep.get('status'), 'branch': rep.get('branch'),
            'cause_uncertified': (st or {}).get('cause') if not cert else None,
            'contributing': (st or {}).get('contributing') if not cert else None,
            'k0_run': rep.get('k0_run'), 'k_star': rep.get('k_star'),
            'end_cycle': rep.get('end_cycle') or rep.get('k_cap') or rep.get('cycles_run'),
            'cap': (st or {}).get('cap') or rep.get('rule_cap'),
            'band_width': rep.get('band_width'), 'range_over_tau': r_over_tau,
            'flag_range_over_tau_ge_0_95': (r_over_tau is not None and r_over_tau >= FLAG_RANGE_OVER_TAU),
            'turning_points': tps, 'turning_points_at_non_clean_cycle': nctp, 'nctp_source': nctp_src,
            'flag_non_clean_turning_point': bool(cert and rep.get('branch') == 'oscillatory' and nctp),
            'non_clean_after_N': (st or {}).get('non_clean_cycles_after_N'),
            'n_vetoes': (st or {}).get('n_vetoes', rep.get('n_vetoes')),
            'terminal_step_over_EPS0': (st or {}).get('terminal_step_over_EPS0', rep.get('terminal_step_over_EPS0')),
            't_sum_end': rep.get('t_sum_end'), 's_signed': rep.get('s_signed'),
            'gap': view.get('gap'), 'slack': view.get('slack'), 'gap_refused': rep.get('gap_refused'),
            'label': rep.get('label'), 'gated': rep.get('gated'),
            'replay_bitwise_through': rep.get('replay_bitwise_through'),
            'replay_first_divergence': rep.get('replay_first_divergence'),
            'certifying_spec': ((s6.get('certifying_spec_per_cell') or {}).get(cell, rep.get('certifying_spec'))
                                if cell in s6['reports'] else
                                {'series': 'frozen_s53_resettle_ext_spec', 'version': 3,
                                 'criterion_version': rep.get('criterion_version'), 'stage_spec': sx['stage_spec']}),
            'Q': view.get('Q'), 'Q_net': view.get('Q_net'), 'salvage': view.get('salvage'), 't': view.get('t'),
            'Q_cc': view.get('Q_cc'),
            'eval_key': (st or {}).get('eval_key') or rep.get('eval_key'),
            'candidate_key': (st or {}).get('candidate_key') or rep.get('candidate_key'),
            'candidate_canonical': rep.get('candidate_canonical'),
            'certification_stats_included': st is not None}
        if cell == 'd_c52e1670' and not row['eval_key']:
            inst_c = docs['C52CERT']['instance']
            row.update({'eval_key': inst_c['eval_key'], 'candidate_key': inst_c['candidate_key'],
                        'candidate_canonical': inst_c['candidate_canonical'],
                        'key_source': f'{INPUTS["C52CERT"][0]} instance'})
        if cert and cell in FLAGGED_NCTP:
            unc_view = dict(view, status='uncertified (treated as such: Addendum 64 ruling 4)',
                            gap=abs(rep['t_sum_end']), slack=abs(rep['s_signed']))
            row['uncertified_form_beside'] = {'gap': unc_view['gap'], 'slack': unc_view['slack'],
                                              'bar': W145.uncertified_bar(unc_view),
                                              'formula': '3 x max(|t_sum at k*|, |s|) (L132.resolve via W145.uncertified_bar)'}
        rows[cell] = row
    # references
    inst = d153['result']['instance']
    for ref, r in s6['references'].items():
        row = {'cell': f'ref:{ref}', 'source_summary': 'S6 references', 'item': 'reference', 'name': r.get('name'),
               'status': r['status'], 'k_star': r.get('k_star'), 'end_cycle': r.get('k_star') or r.get('k_cap'),
               'band_width': r.get('band'),
               'range_over_tau': (r['band'] / TAU) if r['status'] == 'certified' else None,
               'range_over_tau_source': 'band / TAU (the reference summary carries the band, not the ratio)',
               'gap': r.get('gap'), 'slack': r.get('slack'), 'gap_refused': r.get('gap_refused'),
               'label': r.get('label'), 'Q': r['Q'], 'Q_net': r.get('Q_net'), 'salvage': r.get('salvage'), 't': r['t'],
               'Q_cc': r['Q_cc'], 'reference_id': ref}
        row['flag_range_over_tau_ge_0_95'] = bool(row['range_over_tau'] is not None
                                                  and row['range_over_tau'] >= FLAG_RANGE_OVER_TAU)
        row['flag_non_clean_turning_point'] = None
        row['nctp_source'] = 'not recorded in the reference views'
        if ref == '7aa017f0':
            row.update({'eval_key': inst['x0']['eval_key'], 'candidate_key': inst['x0']['candidate_key'],
                        'candidate_canonical': inst['x0']['candidate_canonical'],
                        'key_source': 'W153 discount row instance (x0 settled W101)'})
        elif ref == 'bd504ecf':
            row.update({'eval_key': inst['unit']['eval_key'], 'candidate_key': inst['unit']['candidate_key'],
                        'candidate_canonical': inst['unit']['candidate_canonical'],
                        'key_source': 'W153 discount row instance (unit settled W101)'})
        else:
            w = w118['reports']['f2_incumbent' if ref == '5ca4f86c' else 'f2_challenger']
            row.update({'eval_key': w.get('eval_key'), 'candidate_key': w.get('candidate_key'),
                        'candidate_canonical': w.get('candidate_canonical'), 'k_cap': w.get('k_cap'),
                        'replay_bitwise_through': w.get('replay_bitwise_through'),
                        'replay_first_divergence': w.get('replay_first_divergence'),
                        'cause_uncertified': 'gap clause', 'key_source': 'W118 summary report'})
        rows[f'ref:{ref}'] = row
    return rows


def cell_status_text(row):
    if row is None:
        return '—'
    if row['status'] == 'certified':
        s = f"cert. {row.get('branch') or ''} k*{row.get('k_star')}".replace('  ', ' ')
        if row.get('range_over_tau') is not None:
            s += f" r/τ {row['range_over_tau']:.3f}"
        if row.get('flag_range_over_tau_ge_0_95'):
            s += ' [≥0.95τ]'
        if row.get('flag_non_clean_turning_point'):
            s += f" [NCTP {','.join(str(c) for c in row['turning_points_at_non_clean_cycle'])}]"
        return s
    cause = row.get('cause_uncertified') or 'uncertified'
    return f"UNCERT. ({cause}) end {row.get('end_cycle')}"


# ======================================================================================================================
#  T1 -- claims
# ======================================================================================================================
def _claim_records(docs):
    recs = {c['claim_id']: c for c in docs['S6']['claims_scored']}
    recs.update(docs['SX']['item_E']['claims'])
    for c in docs['SX']['phase_b_claim']:
        recs[c['claim_id']] = c
    return recs


def _res_numbers(res):
    """(rule, threshold-or-bar, multiple on Q, multiple on Q_cc) of a resolution dict."""
    if res is None:
        return None, None, None, None
    if res.get('rule') == 'uncertified_form':
        return 'uncertified form', res.get('bar'), res.get('margin_over_bar_Q'), res.get('margin_over_bar_Qcc')
    thr = res.get('threshold')
    mcc = (res['margin_Qcc'] / thr) if (thr and res.get('margin_Qcc') is not None) else None
    return 'max(3 x larger bar, 2 tau)', thr, res.get('margin_over_threshold'), mcc


def claims_table(docs, cells):
    recs = _claim_records(docs)
    salv = {c['claim_id']: c for c in docs['W153S']['result']['claims']}
    rows, problems = [], []
    if set(recs) != set(salv):
        problems.append(f'claim sets differ: records-only {sorted(set(recs) - set(salv))}, '
                        f'W153-only {sorted(set(salv) - set(recs))}')
    for cid, s in salv.items():
        rec = recs.get(cid)
        if rec is None:
            continue
        if rec['d_Q'] != s['d_gross']:
            problems.append(f'{cid}: d_Q {rec["d_Q"]} != W153 d_gross {s["d_gross"]}')
        rule, thr, m_q, m_cc = _res_numbers(rec['gross'])
        net_res = DET.resolve_v6(s['d_net'], s['d_net'] + (rec['d_Qcc'] - rec['d_Q']), _views(rec))
        nrule, nthr, nm, _ = _res_numbers(net_res)
        ref_cells, oth_cells = s['ref_cells'], s['other_cells']
        row = {
            'claim_id': cid, 'family': cid.split(':')[0], 'statement': rec['statement'], 'form': rec['form'],
            'claim_type': rec['claim_type'], 'primary_convention_of_the_claim': s['primary_convention'],
            'ref_cell': ref_cells[0], 'other_cell': oth_cells[0],
            'ref_status': cell_status_text(cells.get(ref_cells[0])),
            'other_status': cell_status_text(cells.get(oth_cells[0])),
            'I_ref': rec['I_ref'], 'I_other': rec['I_other'],
            'd_gross': rec['d_Q'], 'gross_rule': rule, 'gross_threshold_or_bar': thr, 'gross_multiple': m_q,
            'gross_verdict': rec['gross']['verdict'],
            'gross_binding_term': rec['gross'].get('binding_term'),
            'd_net': s['d_net'], 'net_threshold_or_bar_W153': s['net_threshold'], 'net_verdict_W153': s['net_verdict'],
            'net_multiple': (abs(s['d_net']) / s['net_threshold']) if s['net_threshold'] else None,
            'net_label': NET_LABEL_RECORDED if s['net_source'].startswith('recorded') else NET_LABEL_VALIDATED,
            'net_source_W153': s['net_source'],
            'net_verdict_recomputed_here': net_res.get('verdict'), 'net_threshold_recomputed_here': nthr,
            'salvage_effect_d_net_minus_d_gross': s['salvage_effect_d_net_minus_d_gross'],
            'd_Qcc_report_only': rec['d_Qcc'], 'Qcc_multiple_report_only': m_cc,
            'Qcc_verdict_report_only': rec['gross'].get(
                'verdict_Qcc_report_only', 'in the verdict (the uncertified form tests gross AND Q_cc)'),
            'verdict_superseded_rule_report_only': rec.get('verdict_superseded_rule_report_only'),
            'verdict_changed_by_v6': rec.get('verdict_changed_by_v6'),
            'instance': {c: {'eval_key': (cells.get(c) or {}).get('eval_key'),
                             'candidate_key': (cells.get(c) or {}).get('candidate_key'),
                             'candidate_canonical': (cells.get(c) or {}).get('candidate_canonical')}
                         for c in ref_cells + oth_cells},
            'notes': []}
        if nthr is not None and abs(nthr - s['net_threshold']) > 1e-6:
            problems.append(f'{cid}: net threshold recomputed {nthr} != W153 {s["net_threshold"]}')
        if net_res.get('verdict') != s['net_verdict']:
            problems.append(f'{cid}: net verdict recomputed {net_res.get("verdict")} != W153 {s["net_verdict"]}')
        # ruling 4: the uncertified form beside every claim naming a flagged certificate
        flagged = [c for c in ref_cells + oth_cells if c in FLAGGED_NCTP]
        if flagged:
            rv, ov = _views(rec)
            if ref_cells[0] in FLAGGED_NCTP:
                rv = _treated_uncertified(rv, cells[ref_cells[0]])
            if oth_cells[0] in FLAGGED_NCTP:
                ov = _treated_uncertified(ov, cells[oth_cells[0]])
            alt = DET.score_claim_v6(_claim_def(rec), rv, ov)
            arule, athr, am_q, am_cc = _res_numbers(alt['gross'])
            row['uncertified_form_beside_A64_ruling4'] = {
                'flagged_cells': flagged, 'rule': arule, 'bar': athr, 'multiple_Q': am_q, 'multiple_Qcc': am_cc,
                'verdict': alt['gross']['verdict'], 'scorer': 'p515_s53_w142_determinacy.score_claim_v6'}
            row['notes'].append(f'A64 ruling 4: certificate(s) {", ".join(flagged)} rest on a non-clean turning point; '
                                f'kept and flagged; uncertified form beside: bar {eur(athr)}, {mult(am_q)} '
                                f'({alt["gross"]["verdict"]})')
            if cid == 'J:e4_value_minus_I':
                row['manuscript_figure_A64'] = {'use': 'the uncertified bar (conservative)', 'bar': athr,
                                                'multiple': am_q, 'verdict': alt['gross']['verdict']}
                row['notes'].append('A64 ruling 4: the MANUSCRIPT uses J 4 MWh at its uncertified bar')
        if cid == 'E:n7_4h_e1_C2_calfade:vs_C3':
            row['notes'].append('A64 ruling 5: reported WITHIN RESOLUTION (the determinacy floor changed this verdict '
                                'from the superseded rule); S1 restated "within resolution of C3 at a 0.70 floor"; '
                                'the A61 prediction "the floor changes no verdict" failed on this claim (recorded, '
                                'not re-litigated)')
        if cid == 'E:n7_4h_e1_no_ageing:value_minus_I':
            row['notes'].append('A64 verbatim sentence: without ageing the unit is at break-even (within resolution)')
        if cid == 'H:m1.5:value_minus_I':
            row['notes'].append('A63: sign unresolved at m = 1.5 (uncertified form, gap-refused unit cell); crossing '
                                'at or below m = 1.75; the m = 1.75 row is PENDING W156')
        if s.get('verdict_differs_between_conventions'):
            row['notes'].append(f'salvage convention changes the verdict: gross {s["gross_verdict"]}, net '
                                f'{s["net_verdict"]}')
        rows.append(row)
    order = {f: i for i, f in enumerate(('B', 'C', 'D', 'G', 'H', 'I', 'J', 'L', 'E', 'CHECK'))}
    rows.sort(key=lambda r: (order.get(r['family'], 99), r['claim_id']))
    return rows, problems


def _views(rec):
    keys = ('status', 'Q', 't', 'Q_cc', 'Q_net', 'salvage', 'band', 'gap', 'slack')
    return {k: rec['ref'].get(k) for k in keys}, {k: rec['other'].get(k) for k in keys}


def _claim_def(rec):
    return {k: rec[k] for k in ('claim_id', 'item', 'statement', 'form', 'claim_type', 'net_of_salvage', 'I_ref',
                                'I_other')}


def _treated_uncertified(view, cell_row):
    return dict(view, status='uncertified (treated as such: Addendum 64 ruling 4)', gap=abs(cell_row['t_sum_end']),
                slack=abs(cell_row['s_signed']))


# ======================================================================================================================
#  T3 -- break-even
# ======================================================================================================================
def break_even(docs, cells):
    w = docs['W145']
    res_c = w['result']
    q0 = dict(w['q0_reference'])
    points = []
    if set(w['points']) != set(W145.LABELS):
        raise RuntimeError(f'W145 points {sorted(w["points"])} != W145.LABELS {W145.LABELS}')
    for lb in W145.LABELS:  # W145's own order (the committed JSON is key-sorted; OLS summation order matters to the ulp)
        v = dict(w['points'][lb])
        v['label'] = lb
        points.append(v)
    p_cost, e_cost = w['constants']['p_cost_eur_per_MVA'], w['constants']['e_cost_eur_per_MWh']
    pc2, ec2 = W145._costs()
    problems = []
    if (pc2, ec2) != (p_cost, e_cost):
        problems.append(f'W145 constants {p_cost}, {e_cost} != W2 table {pc2}, {ec2}')
    # (a) the committed W145 result, reproduced by W145's function on its committed points
    rep = W145.banded_breakeven_fit(points, q0, p_cost, e_cost)
    keys = (('banded_fit', 'breakeven_range'), ('banded_fit', 'margin_to_cost_range'), ('banded_fit', 'slope_range_b'))
    repro = {'/'.join(k): {'committed': res_c[k[0]][k[1]], 'recomputed': rep[k[0]][k[1]],
                           'equal': res_c[k[0]][k[1]] == rep[k[0]][k[1]]} for k in keys}
    for qn in ('breakeven_marginal_4h_energy_cost', 'margin_to_energy_cost_per_MWh', 'b_per_MWh'):
        repro[f'certified_only_fit/{qn}'] = {'committed': res_c['certified_only_fit'][qn],
                                             'recomputed': rep['certified_only_fit'][qn],
                                             'equal': res_c['certified_only_fit'][qn] == rep['certified_only_fit'][qn]}
    if not all(v['equal'] for v in repro.values()):
        problems.append('W145 result not reproduced bitwise from its committed points')
    # (b) the conservative fit: d_4a82a64a (label n7_4h_e3) also as an interval
    d4 = cells['d_4a82a64a']
    lab = [lb for lb, src in w['provenance'].items() if src.get('cell') == 'd_4a82a64a']
    if lab != ['n7_4h_e3']:
        problems.append(f'd_4a82a64a label in W145 provenance is {lab}, expected n7_4h_e3')
    cons_pts = []
    for p in points:
        if p['label'] == 'n7_4h_e3':
            if p['Q'] != d4['Q'] or p['t'] != d4['t_sum_end']:
                problems.append('W145 point n7_4h_e3 differs from the v6 summary view of d_4a82a64a')
            p = dict(p, status='uncertified (treated as such: Addendum 64 ruling 4)', gap=abs(d4['t_sum_end']),
                     slack=abs(d4['s_signed']))
        cons_pts.append(p)
    cons = W145.banded_breakeven_fit(cons_pts, q0, p_cost, e_cost)
    iv = cons['intervals']['n7_4h_e3']
    w_mid = rep['banded_fit']['midpoint_fit']['slack_bound_weights']['n7_4h_e3']
    hand = {'label': 'HAND ARITHMETIC (cross-check only; the table figure is the fit above)',
            'formula': 'e*_max(4 intervals) = e*_max(3 intervals) + |w_n7_4h_e3| x (bar + 2 tau)',
            'w': w_mid, 'h': iv['half_width'],
            'e_star_max': rep['banded_fit']['breakeven_range'][1] + abs(w_mid) * iv['half_width']}
    hand['margin_min'] = e_cost - hand['e_star_max']
    diff = abs(hand['e_star_max'] - cons['banded_fit']['breakeven_range'][1])
    if diff > 1e-6:
        problems.append(f'conservative e*_max {cons["banded_fit"]["breakeven_range"][1]} differs from the linear '
                        f'cross-check by {diff}')

    def summ(r):
        cf, bf = r['certified_only_fit'], r['banded_fit']
        m = bf['midpoint_fit']
        m4 = bf['range_over_the_box']['marginal_4h_slope_b_plus_c_over_4']
        pr = r['prediction_slope_within_5pct']
        return {'certified': r['certified'], 'uncertified_as_intervals': r['uncertified'],
                'intervals': {lb: {k: v for k, v in iv_.items() if k in ('Q_cap', 'gap', 'slack', 'bar', 'half_width')}
                              for lb, iv_ in r['intervals'].items()},
                'certified_only': {k: cf.get(k) for k in ('n', 'b_per_MWh', 'c_per_MVA',
                                                          'marginal_4h_slope_b_plus_c_over_4',
                                                          'breakeven_marginal_4h_energy_cost',
                                                          'margin_to_energy_cost_per_MWh')},
                'banded_mid': {k: m.get(k) for k in ('n', 'b_per_MWh', 'c_per_MVA', 'marginal_4h_slope_b_plus_c_over_4',
                                                     'breakeven_marginal_4h_energy_cost',
                                                     'margin_to_energy_cost_per_MWh')},
                'slope_b_plus_c_over_4_range': [m4['min'], m4['max']],
                'breakeven_range': bf['breakeven_range'], 'margin_to_cost_range': bf['margin_to_cost_range'],
                'slope_b_plus_c_over_4_rel_mid': pr['marginal_4h_slope_report_only']['rel_mid'],
                'slope_b_plus_c_over_4_rel_corners': [pr['marginal_4h_slope_report_only']['rel_at_min'],
                                                      pr['marginal_4h_slope_report_only']['rel_at_max']],
                'slope_b_rel_mid': pr['rel_mid'], 'slope_b_rel_corners': [pr['rel_at_b_min'], pr['rel_at_b_max']],
                'conclusion_holds_both_fits': r['conclusion_breakeven_below_energy_cost']['holds'],
                'min_margin_banded': r['conclusion_breakeven_below_energy_cost']['min_margin_banded']}
    return {'objective_convention': OBJECTIVE_CONVENTION + '; e* = b + c/4 - p_cost/4 (EUR/MWh)',
            'p_cost_eur_per_MVA': p_cost, 'e_cost_eur_per_MWh': e_cost,
            'slope_ruling': 'Addendum 64 ruling 3: the slope is b + c/4 (the 4 h marginal slope); the 5 % prediction is '
                            'scored at the midpoints (held: 0.59 % on b, 0.31 % on b + c/4); the corners are interval '
                            'sensitivity, not the prediction\'s test',
            'committed_W145': summ(rep), 'committed_reproduced': repro,
            'conservative_A64': dict(summ(cons), note=(
                'Addendum 64 ruling 4: d_4a82a64a (n7_4h_e3) ALSO as an interval (its uncertified bar 3 x max(|t_sum|, '
                '|s|) + 2 tau); computed by W145.banded_breakeven_fit on W145\'s committed points with that view '
                'passed as uncertified -- no code change. Its certified-only fit drops d_4a82a64a (n = 6); the '
                'banded midpoint fit is unchanged. THE MANUSCRIPT FIGURE is this fit\'s minimum margin')),
            'manuscript_figure': {'breakeven_max_eur_per_mwh': cons['banded_fit']['breakeven_range'][1],
                                  'margin_min_eur_per_mwh': cons['banded_fit']['margin_to_cost_range'][0],
                                  'source': 'conservative_A64 (W145 function, 4 intervals)'},
            'hand_arithmetic_crosscheck': hand}, problems


# ======================================================================================================================
#  T4..T9
# ======================================================================================================================
def year_ladder(docs):
    p3 = docs['W154']['results']['part3b_w118_year_ladder']
    w118 = docs['W118']['reports']
    va = {'status': 'certified', 'band': w118['yl_y2030']['band_width']}
    vb = {'status': 'certified', 'band': w118['yl_y2035']['band_width']}
    d = p3['difference_2035_minus_2030']
    g6 = DET.resolve_v6(d['D_gross_w153_form'], d['D_gross_cc_w153_form'], (va, vb))
    n6 = DET.resolve_v6(d['D_net_w153_form'], d['D_net_cc_w153_form'], (va, vb))
    rec_n6 = d['resolution_v6_rule_net_report_only']
    problems = []
    if n6['threshold'] != rec_n6['threshold'] or n6['verdict'] != rec_n6['verdict']:
        problems.append('year-ladder net v6 resolution not reproduced from the W118 bands')
    return {'objective_convention': OBJECTIVE_CONVENTION + '; M = I + Q - Q181 (W118 form); net = gross - salvage',
            'per_year': {y: {k: v.get(k) for k in ('eval_key', 'candidate_key', 'k_star', 'status', 'I_j', 'Q_k_star',
                                                   'salvage_last', 'M_gross_w153_form', 'M_net_w153_form')}
                         for y, v in p3['per_year'].items()},
            'bands': {'2030': va['band'], '2035': vb['band']},
            'D_gross': d['D_gross_w153_form'], 'D_gross_cc_report_only': d['D_gross_cc_w153_form'],
            'D_net': d['D_net_w153_form'], 'D_net_cc_report_only': d['D_net_cc_w153_form'],
            'gross_v6': {'threshold': g6['threshold'], 'multiple': g6['margin_over_threshold'], 'verdict': g6['verdict']},
            'gross_recorded_W118_rule': d['resolution_w118_rule_gross'],
            'net_v6': {'threshold': n6['threshold'], 'multiple': n6['margin_over_threshold'], 'verdict': n6['verdict']},
            'net_recorded_W154b_v6': {'threshold': rec_n6['threshold'], 'verdict': rec_n6['verdict']},
            'net_label': NET_LABEL_VALIDATED + '; the table carries -2,286.25 (the prose -2,286.2 was rounded-component '
                                               'arithmetic)',
            'sentence_A58_R3': ('the investment-year comparison in this instance is decided by the salvage convention, '
                                'not by operation')}, problems


def phase_b(docs, claims_rows):
    w = docs['W118']
    x0 = w['differences']['x0_comparator']
    v0 = {'status': 'certified', 'band': x0['band_width']}
    out = []
    for cell, d in w['differences']['phase_b_and_year_ladder_vs_x0'].items():
        if not cell.startswith('pb_'):
            continue
        rep = w['reports'][cell]
        v = {'status': rep['status'], 'band': rep['band_width']}
        r6 = DET.resolve_v6(d['M'], d['M_cc'], (v0, v))
        out.append({'cell': cell, 'eval_key': rep.get('eval_key'), 'candidate_key': rep.get('candidate_key'),
                    'candidate_canonical': rep.get('candidate_canonical'), 'status': rep['status'],
                    'k_star': rep.get('k_star'), 'k0_run': rep.get('k0_run'), 'range_over_tau': rep.get('range_over_tau'),
                    'flag_range_over_tau_ge_0_95': (rep.get('range_over_tau') or 0) >= FLAG_RANGE_OVER_TAU,
                    'replay_bitwise_through': rep.get('replay_bitwise_through'),
                    'M_gross': d['M'], 'M_cc_report_only': d['M_cc'],
                    'v6_threshold': r6['threshold'], 'v6_multiple': r6['margin_over_threshold'], 'v6_verdict': r6['verdict'],
                    'recorded_W118_rule': {'resolution': d['resolution'], 'verdict': d['verdict']},
                    'superseded': (cell == 'pb_y2025_n5'),
                    'note': ('superseded by the v6 re-run pb_y2025_n5_v6 (claim C:y2025__n5_p0.25_e0.5): the W118 '
                             'certificate sat on a non-Optimal cycle (Addendum 59)') if cell == 'pb_y2025_n5'
                    else 'v6 rule applied here by DET.resolve_v6 (zero-solve); net not in the W154b validation set'})
    c = next(r for r in claims_rows if r['claim_id'] == 'C:y2025__n5_p0.25_e0.5')
    out.append({'cell': 'pb_y2025_n5_v6', 'status': 'certified', 'M_gross': c['d_gross'],
                'M_cc_report_only': c['d_Qcc_report_only'], 'v6_threshold': c['gross_threshold_or_bar'],
                'v6_multiple': c['gross_multiple'], 'v6_verdict': c['gross_verdict'],
                'note': 'the extension v6 re-run (claim C:y2025__n5_p0.25_e0.5); T1 carries its full row'})
    return out


def benchmark(docs):
    b = docs['BENCH']
    cl = b['claim']
    return {'objective_convention': b['objective_convention'], 'frozen_spec': b['frozen_benchmark_spec'],
            'coordinated_Q': b['coordinated']['q'], 'coordinated_source': b['coordinated']['source'],
            'coordinated_reproducibility_band': b['coordinated']['reproducibility_band_eur'],
            'arms': {a: {'Q_best': v['q_best'], 'best_start': v['best_start'],
                         'multimodality_band': v['multimodality_band_eur']} for a, v in b['per_arm_nrf'].items()},
            'benefit': cl['benefit_eur'], 'benefit_relative': cl['benefit_relative'],
            'larger_band': cl['larger_band_eur'], 'multiple': cl['benefit_eur'] / cl['larger_band_eur'],
            'determinate': cl['determinate'], 'definition': cl['definition'], 'decomposition': cl['decomposition'],
            'reverse_flow_interface_hours': b['coordinated_reverse_flow_count']['totals']['material']['count'],
            'reverse_flow_energy_mwh_block_weighted':
                b['coordinated_reverse_flow_count']['totals']['material']['energy_mwh_block_weighted'],
            'sweep': {k: {'n_blocks': v['n_blocks_tn_cannot_accept'], 'n_hours': v['n_hours_tn_cannot_accept']}
                      for k, v in b['sweep'].items()},
            'instance': b['instance'], 'nlp_solver_path': b['nlp_solver_path'], 'interpreter': b.get('interpreter')}


def discount(docs):
    t = docs['W153D']['result']['table']
    return [{'rate': r['rate'], 'V': r['V'], 'I': r['I'], 'value_minus_I': r['value_minus_I'],
             'threshold_a61_conservative': r['a61_threshold_conservative'],
             'multiple': abs(r['value_minus_I_over_a61_threshold']), 'verdict': r['value_minus_I_verdict_a61']}
            for r in t]


def ageing(docs):
    e = docs['SX']['item_E']
    st = e['statements']
    out = []
    for arm, r in e['per_arm'].items():
        out.append({'arm': arm, 'cell': r['cell'], 'status': r['status'], 'value': r['value'], 'I': r['I'],
                    'value_minus_I': r['value_minus_I'], 'verdict': (r.get('value_minus_I_claim') or {}).get('verdict'),
                    'floor_year_070': r.get('floor_year'), 'AE': r.get('AE'), 'EFC': r.get('EFC'), 'k': r.get('k'),
                    'eps_AE_070': st['S5_elasticity_to_available_energy']['eps_AE_070'].get(arm),
                    'eps_AE_resolvable_070': st['S5_elasticity_to_available_energy']['resolvable_070'].get(arm),
                    'eps_AE_050_superseded': st['S5_elasticity_to_available_energy']['eps_AE_050'].get(arm)})
    return {'rows': out, 'eps_AE_band_over_resolvable_070':
            st['S5_elasticity_to_available_energy']['band_over_resolvable_070'],
            'expert_prediction_A58_supplement': {a: {k: v.get(k) for k in ('scored', 'floor_year', 'eps_k_050',
                                                                          'eps_k_070', 'verdict', 'holds')}
                                                 for a, v in e['expert_prediction'].items() if isinstance(v, dict)},
            'A64_ruling_6': 'scored "consistent in direction, not resolved at this precision"; S5 restated; the '
                            'late-life-tail reading stays a hypothesis'}


def dead_zone(docs, cells):
    reps = dict(docs['S6']['reports'])
    rows = []
    for cell in ('h_f9eae48f', 'j_5f3cccb4', 'l_45aa25a6', 'l_7c455554', 'l_b2251bc5', 'l_0ee93aca', 'd_36686489'):
        r = reps[cell]
        tbn = r.get('t_by_node_at_cap') or {}
        tot = sum(tbn.values()) if tbn else None
        rows.append({'cell': cell, 'status': r['status'], 'cause': cells[cell].get('cause_uncertified'),
                     'gap_refused': r.get('gap_refused'), 'label': r.get('label'),
                     't_sum_at_cap': r.get('t_sum_at_cap', r.get('t_sum_end')),
                     'share_by_node': {n: v / tot for n, v in tbn.items()} if tot else None,
                     'pf_primal_last': (r.get('pf_primal_ratio_after_k0') or {}).get('last'),
                     'pf_primal_slope_last_50': r.get('pf_primal_slope_last_50'),
                     'lapse_events': r.get('lapse_events'),
                     'entry': ('by its signature, cause stated: two TSO recoveries (lapse resets at 218, 222) reset '
                               'the rule before the gap clause was reached (Addendum 64 ruling 7a)')
                     if cell == 'l_0ee93aca' else
                     ('baseline-price degenerate case with large storage (Addendum 62): uncertified by the growth '
                      'test after repeated TSO recoveries, not by the gap clause') if cell == 'd_36686489' else
                     'gap-refused (the dead-zone label)'})
    for ref in ('5ca4f86c', 'e28de4ac'):
        r = docs['S6']['references'][ref]
        rows.append({'cell': f'ref:{ref}', 'status': r['status'], 'cause': 'gap clause', 'gap_refused': r.get('gap_refused'),
                     'label': r.get('label'), 't_sum_at_cap': r['t'], 'entry': r['name']})
    w149 = [dict(d, column_note=('report_only_tso_cheapest_lever_is_shared_ess: REPORT-ONLY, added POST HOC (Addendum 64 '
                                 'ruling 7b)')) for d in docs['W149']['dead_zone_table']]
    return {'cells': rows, 'w149_rows': w149}


# ======================================================================================================================
#  T10 -- the Addendum 64 rows
# ======================================================================================================================
A64_CLAIM_ORDER = ('H:m1.75:value_minus_I', 'E:soh050:delta_value_vs_070', 'E:soh050:value_minus_I',
                   'E:soh050:delta_value_vs_g070_REPORT_BESIDE')


def a64_rows(docs, include, summary_rel):
    spec = docs['A64SPEC'] or {}
    defs = (spec.get('definitions') or {}).get('claims') or {}
    preds = spec.get('predictions_recorded_before_any_run') or {}
    rows, problems, inp = [], [], {}
    summ = None
    if include:
        if not L132._committed_clean(summary_rel):
            problems.append(f'{summary_rel} is not committed clean')
        else:
            inp[summary_rel] = _sha(summary_rel)
            summ = json.load(open(os.path.join(REPO, summary_rel)))
    for cid in A64_CLAIM_ORDER:
        d = defs.get(cid) or {}
        row = {'claim_id': cid, 'statement': d.get('statement'), 'ref_cell': d.get('ref'), 'other_cell': d.get('other'),
               'form': d.get('form'), 'I_other': d.get('I_other'), 'status': PENDING,
               'source_definition': f'{A64_SPEC_REL} (sha {A64_SPEC_SHA[:8]}) definitions.claims'}
        if summ is not None:
            c = ((summ.get('scored') or {}).get('claims') or {}).get(cid)
            if c is None or 'd_Q' not in c:
                row['status'] = 'NOT SCORED in the W155 summary'
            else:
                rule, thr, m_q, m_cc = _res_numbers(c['gross'])
                reps = summ.get('reports') or {}
                row.update({'status': 'filled from the W155 summary', 'd_gross': c['d_Q'], 'gross_rule': rule,
                            'gross_threshold_or_bar': thr, 'gross_multiple': m_q, 'gross_verdict': c['gross']['verdict'],
                            'd_Qcc_report_only': c['d_Qcc'], 'Qcc_multiple_report_only': m_cc,
                            'cells': {k: {f: (reps.get(k) or {}).get(f) for f in (
                                'status', 'k_star', 'k0_run', 'range_over_tau', 'band_width', 't_sum_end', 's_signed',
                                'eval_key', 'candidate_key', 'candidate_canonical', 'floor_year')}
                                for k in (d.get('ref'), d.get('other')) if k in reps}})
        rows.append(row)
    pred_rows = {'A_m175': preds.get('A_m175'), 'B_soh050': preds.get('B_soh050'), 'gate_g070': preds.get('gate_g070')}
    scored = None
    if summ is not None:
        sc = summ.get('scored') or {}
        scored = {'A': sc.get('A'), 'B': sc.get('B'), 'STOP': sc.get('STOP')}
    return {'rows': rows, 'predictions_recorded': pred_rows, 'scored': scored if include else PENDING,
            'summary': summary_rel if include else None}, problems, inp


# ======================================================================================================================
#  provenance (from committed records only)
# ======================================================================================================================
def provenance(docs):
    chk = docs['V6CHECKS']
    spec = docs['V6SPEC']
    ipopt = chk['clean_rule']['tolerance_sources']['ipopt_defaults']['binary']
    lc = spec['launch_commands']
    one = lc[sorted(lc)[0]]
    return {'ipopt': {'value': ipopt, 'source': f'{INPUTS["V6CHECKS"][0]} clean_rule.tolerance_sources.ipopt_defaults'},
            'launch_command_example': {'value': one, 'source': f'{INPUTS["V6SPEC"][0]} launch_commands'},
            'benchmark_nlp_solver_path': docs['BENCH']['nlp_solver_path'],
            'benchmark_interpreter': docs['BENCH'].get('interpreter'),
            'memory_preflight_hw_memsize_bytes':
                spec['memory_preflight']['measured_at_freeze_non_gating'].get('hw_memsize_bytes'),
            'not_from_this_builder': ('the linear solvers (MA97 networks, MA57 ESSO) are read from the committed '
                                      'params files in the draft text, not by this builder')}


# ======================================================================================================================
#  Markdown
# ======================================================================================================================
def markdown(doc):
    L = []
    a = L.append
    t = doc['tables']
    a('# P5.15 W157 — Step 6 tables (DRAFT for the Planner)')
    a('')
    a(f"Generated {doc['utc']} by `{SCRIPT_REL}` (sha256 `{doc['script']['sha256'][:12]}`), git HEAD "
      f"`{doc['git_head'][:12]}`; zero solves, no model loads. Every input is committed clean; its sha256 is in "
      f"`{OUT_JSON}` → `inputs`. **A64 rows: {'FILLED' if doc['include_a64'] else PENDING}.**")
    a('')
    a(f'**Objective convention (every table):** {OBJECTIVE_CONVENTION}. τ = {TAU:,.2f} €; certified-pair determinacy '
      f'threshold max(3 × larger band, 2τ = {TWO_TAU:,.2f} €) (Addendum 61); uncertified form bar = 3·max(|gap|, |slack|) '
      'in both gross and Q_cc (Addendum 58). Multiple = |d| / threshold-or-bar.')
    a('')
    a(f'**Net label:** every net figure is "{NET_LABEL_VALIDATED}", except G rows: "{NET_LABEL_RECORDED}".')
    a('')
    a('## T1 — claims (gross primary; net beside; Q_cc report-only)')
    a('')
    a('| claim | ref cell [status] | other cell [status] | d gross | rule | threshold / bar | × | verdict (gross) '
      '| d net | net × | net verdict | net label | d Q_cc (r-o) | Q_cc × | instance (eval / cand.) | notes |')
    a('|---|---|---|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---|---|')
    for r in t['claims']:
        inst = '; '.join(f"{c}: {short(v['eval_key'], 8)}/{short(v['candidate_key'], 8)}" for c, v in r['instance'].items())
        a(f"| `{r['claim_id']}` | {r['ref_cell']} [{r['ref_status']}] | {r['other_cell']} [{r['other_status']}] | "
          f"{eur(r['d_gross'])} | {r['gross_rule']} | {eur(r['gross_threshold_or_bar'])} | {mult(r['gross_multiple'])} | "
          f"{r['gross_verdict']} | {eur(r['d_net'])} | {mult(r['net_multiple'])} | {r['net_verdict_W153']} | "
          f"{'recorded' if r['net_label'] == NET_LABEL_RECORDED else 'W154b'} | {eur(r['d_Qcc_report_only'])} | "
          f"{mult(r['Qcc_multiple_report_only'])} | {inst} | {' / '.join(r['notes'])} |")
    a('')
    a('Status legend: `cert. <branch> k*N r/τ x` certified at N with range/τ x; `[≥0.95τ]` range ≥ 0.95 τ; '
      '`[NCTP c]` a turning point on non-clean cycle c (certificate kept, flagged: Addendum 64 ruling 4); '
      '`UNCERT. (cause) end N`.')
    a('')
    a('## T2 — cells (certification status; instance keys)')
    a('')
    a('| cell | item | status | branch | k0 | k* / end | range/τ | ≥0.95τ | NCTP | cause | gap | slack | band '
      '| replay bitwise through | certifying spec | eval key | candidate key |')
    a('|---|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|---|---|')
    for c, r in sorted(t['cells'].items()):
        spec = r.get('certifying_spec')
        if not isinstance(spec, dict):
            spec_s = '—'
        elif spec.get('series') == 'frozen_s53_resettle_ext_spec':
            spec_s = f"ext v{spec.get('version')} (rule v{spec.get('criterion_version')})"
        else:
            spec_s = f"v{spec.get('version')}" + (f" ({spec.get('mode')})" if spec.get('mode') else '')
        nctp = r.get('turning_points_at_non_clean_cycle')
        rot = r.get('range_over_tau')
        rot_s = '—' if rot is None else format(rot, '.3f')
        a(f"| {c} | {r.get('item')} | {r['status']} | {r.get('branch') or '—'} | {r.get('k0_run') or '—'} | "
          f"{r.get('k_star') or r.get('end_cycle') or '—'} | {rot_s} | "
          f"{'yes' if r.get('flag_range_over_tau_ge_0_95') else ''} | "
          f"{'—' if nctp is None else (','.join(str(x) for x in nctp) or '')} | {r.get('cause_uncertified') or ''} | "
          f"{eur(r.get('gap'))} | {eur(r.get('slack'))} | {eur(r.get('band_width'))} | "
          f"{r.get('replay_bitwise_through') or '—'} | {spec_s} | {short(r.get('eval_key'), 16)} | "
          f"{short(r.get('candidate_key'), 8)} |")
    a('')
    for c in FLAGGED_NCTP:
        u = t['cells'][c].get('uncertified_form_beside') or {}
        a(f"- `{c}` uncertified form beside (Addendum 64 ruling 4): gap {eur(u.get('gap'))}, slack {eur(u.get('slack'))}, "
          f"bar {eur(u.get('bar'))}.")
    a('')
    be = t['break_even']
    a('## T3 — break-even fit, node 7 (slope b + c/4; Addendum 64 rulings 3–4)')
    a('')
    a(f"Convention: {be['objective_convention']}. Energy cost e_cost = {eur(be['e_cost_eur_per_MWh'])} €/MWh; "
      f"p_cost = {eur(be['p_cost_eur_per_MVA'])} €/MVA.")
    a('')
    a('| fit | n cert. | intervals | certified-only e* | banded mid e* | e* range | margin range | b + c/4 range |'
      ' b + c/4 rel. mid | corners |')
    a('|---|---:|---|---:|---:|---|---|---|---:|---|')
    for name, f in (('committed W145 (3 intervals)', be['committed_W145']),
                    ('**conservative A64 (4 intervals, + d_4a82a64a)**', be['conservative_A64'])):
        a(f"| {name} | {f['certified_only']['n']} | {', '.join(f['uncertified_as_intervals'])} | "
          f"{eur(f['certified_only']['breakeven_marginal_4h_energy_cost'])} | "
          f"{eur(f['banded_mid']['breakeven_marginal_4h_energy_cost'])} | "
          f"{eur(f['breakeven_range'][0])} – {eur(f['breakeven_range'][1])} | "
          f"{eur(f['margin_to_cost_range'][0])} – {eur(f['margin_to_cost_range'][1])} | "
          f"{eur(f['slope_b_plus_c_over_4_range'][0])} – {eur(f['slope_b_plus_c_over_4_range'][1])} | "
          f"{100 * f['slope_b_plus_c_over_4_rel_mid']:.2f} % | "
          f"{100 * f['slope_b_plus_c_over_4_rel_corners'][0]:.1f} / {100 * f['slope_b_plus_c_over_4_rel_corners'][1]:.1f} % |")
    mf = be['manuscript_figure']
    a('')
    a(f"**Manuscript figure (conservative):** break-even ≤ {eur(mf['breakeven_max_eur_per_mwh'], 0)} €/MWh; margin to "
      f"energy cost ≥ {eur(mf['margin_min_eur_per_mwh'], 0)} €/MWh. Hand cross-check (labelled, not the table "
      f"figure): {eur(be['hand_arithmetic_crosscheck']['margin_min'], 0)} €/MWh. The committed W145 result is "
      f"reproduced bitwise from its points: {all(v['equal'] for v in be['committed_reproduced'].values())}.")
    a('')
    a(f"Interval of d_4a82a64a: {json.dumps({k: round(v, 2) for k, v in be['conservative_A64']['intervals']['n7_4h_e3'].items()})}.")
    a('')
    y = t['year_ladder']
    a('## T4 — year ladder (2035 − 2030), gross primary, net beside')
    a('')
    a(f"Convention: {y['objective_convention']}.")
    a('')
    a('| | M gross | salvage | M net | eval key | k* |')
    a('|---|---:|---:|---:|---|---:|')
    for yr, v in y['per_year'].items():
        a(f"| {yr} | {eur(v['M_gross_w153_form'])} | {eur(v['salvage_last'])} | {eur(v['M_net_w153_form'])} | "
          f"{short(v['eval_key'], 16)} | {v['k_star']} |")
    a(f"| **2035 − 2030** | **{eur(y['D_gross'])}** — {y['gross_v6']['verdict']} {mult(y['gross_v6']['multiple'])} "
      f"(thr {eur(y['gross_v6']['threshold'])}; W118 rule: {y['gross_recorded_W118_rule']['verdict']}) | | "
      f"**{eur(y['D_net'])}** — {y['net_v6']['verdict']} {mult(y['net_v6']['multiple'])} (thr "
      f"{eur(y['net_v6']['threshold'])}) | | |")
    a('')
    a(f"Net label: {y['net_label']}. Sentence (A58 Ruling 3): {y['sentence_A58_R3']}.")
    a('')
    a('## T5 — Phase B certificates against x = 0 (gross; v6 rule)')
    a('')
    a('| cell | M gross | M Q_cc (r-o) | v6 threshold | × | v6 verdict | recorded (W118 rule) | k* | range/τ | note |')
    a('|---|---:|---:|---:|---:|---|---|---:|---:|---|')
    for r in t['phase_b']:
        rec = r.get('recorded_W118_rule') or {}
        a(f"| {r['cell']} | {eur(r['M_gross'])} | {eur(r['M_cc_report_only'])} | {eur(r['v6_threshold'])} | "
          f"{mult(r['v6_multiple'])} | {r['v6_verdict']} | {rec.get('verdict', '—')} | {r.get('k_star') or '—'} | "
          f"{'—' if r.get('range_over_tau') is None else format(r['range_over_tau'], '.3f')} | {r['note']} |")
    a('')
    b = t['benchmark']
    a('## T6 — uncoordinated benchmark (Addendum 57; spec v5 `bca69f97`)')
    a('')
    a(f"Convention: {b['objective_convention'][:160]}…")
    a('')
    a('| arrangement | Q (gross) | band |')
    a('|---|---:|---:|')
    a(f"| coordinated (settled x = 0) | {eur(b['coordinated_Q'])} | {eur(b['coordinated_reproducibility_band'])} |")
    for arm, v in b['arms'].items():
        a(f"| {arm} NRF (best of 3 starts: {v['best_start']}) | {eur(v['Q_best'])} | {eur(v['multimodality_band'], 3)} |")
    a('')
    a(f"Claim: **+{eur(b['benefit'])} € ({100 * b['benefit_relative']:.1f} %)**, {mult(b['multiple'])} the larger band "
      f"{eur(b['larger_band'])} — determinate {b['determinate']}. Reverse-flow interface-hours in the coordinated "
      f"solution: {b['reverse_flow_interface_hours']} ({eur(b['reverse_flow_energy_mwh_block_weighted'])} MWh, "
      f"block-weighted). Sweep (no interface rule): "
      + '; '.join(f"{k}: {v['n_blocks']}/12 blocks, {v['n_hours']} h" for k, v in b['sweep'].items()) + '.')
    a('')
    a('## T7 — discount rate (fixed plan; unit n7 0.25 MVA / 1 MWh)')
    a('')
    a('| rate | V | I | value − I | A61 threshold (cons.) | × | verdict |')
    a('|---:|---:|---:|---:|---:|---:|---|')
    for r in t['discount']:
        a(f"| {100 * r['rate']:.0f} % | {eur(r['V'])} | {eur(r['I'])} | {eur(r['value_minus_I'])} | "
          f"{eur(r['threshold_a61_conservative'])} | {mult(r['multiple'])} | {r['verdict']} |")
    a('')
    g = t['ageing']
    a('## T8 — ageing arms at minimum SoH 0.70 (gross)')
    a('')
    a('| arm | cell | value | value − I | verdict | floor binds | ε_AE (0.70) | resolvable | ε_AE (0.50, superseded) |')
    a('|---|---|---:|---:|---|---|---:|---|---:|')
    for r in g['rows']:
        a(f"| {r['arm']} | {r['cell']} | {eur(r['value'])} | {eur(r['value_minus_I'])} | {r['verdict']} | "
          f"{r['floor_year_070'] or 'never'} | {'—' if r['eps_AE_070'] is None else format(r['eps_AE_070'], '.2f')} | "
          f"{r['eps_AE_resolvable_070']} | {'—' if r['eps_AE_050_superseded'] is None else format(r['eps_AE_050_superseded'], '.2f')} |")
    a('')
    a(f"ε_AE band over resolvable arms: {g['eps_AE_band_over_resolvable_070'][0]:.2f}–{g['eps_AE_band_over_resolvable_070'][1]:.2f}. "
      f"{g['A64_ruling_6']}.")
    a('')
    dz = t['dead_zone']
    a('## T9 — dual dead zone (Addendum 58 Ruling 1; Addendum 62; Addendum 64 ruling 7)')
    a('')
    a('| cell | status | cause | t_sum at cap | share n5 / n7 / n9 | pf_primal last | entry |')
    a('|---|---|---|---:|---|---:|---|')
    for r in dz['cells']:
        sh = r.get('share_by_node')
        sh_s = '—' if not sh else ' / '.join(f"{100 * sh.get(n, 0):.0f} %" for n in ('5', '7', '9'))
        a(f"| {r['cell']} | {r['status']} | {r.get('cause') or ''} | {eur(r.get('t_sum_at_cap'))} | {sh_s} | "
          f"{'—' if r.get('pf_primal_last') is None else format(r['pf_primal_last'], '.4f')} | {r['entry']} |")
    a('')
    a('W149 rows (h cells at m = 1.5 and the F2 pair); the TSO-lever column is **report-only, added post hoc**:')
    a('')
    a('| cell | m | storage | TN at bound | DN flex at kink | TSO cheapest lever = shared ESS (post hoc, r-o) | t_sum | verdict |')
    a('|---|---:|---|---|---|---|---:|---|')
    for r in dz['w149_rows']:
        a(f"| {r['cell']} | {r['m']} | {r['storage']} | {r['tn_at_bound']} | {r['dn_flex_marginal_at_kink']} | "
          f"{r['report_only_tso_cheapest_lever_is_shared_ess']} | {eur(r['t_sum_at_cap_eur'])} | {r['verdict']} |")
    a('')
    a6 = t['a64']
    a(f"## T10 — Addendum 64 rows ({'FILLED from ' + str(a6['summary']) if doc['include_a64'] else '**' + PENDING + '**'})")
    a('')
    a('| claim | ref | other | statement | d gross | threshold / bar | × | verdict | status |')
    a('|---|---|---|---|---:|---:|---:|---|---|')
    for r in a6['rows']:
        a(f"| `{r['claim_id']}` | {r['ref_cell']} | {r['other_cell']} | {r['statement']} | {eur(r.get('d_gross'))} | "
          f"{eur(r.get('gross_threshold_or_bar'))} | {mult(r.get('gross_multiple'))} | {r.get('gross_verdict') or PENDING} | "
          f"**{r['status']}** |")
    a('')
    a('Recorded predictions (frozen spec `44a2dce8`, before any run): A (m = 1.75) value − I positive, point +15 k€, '
      '[+8, +22] k€; B (soh_min 0.50) Δvalue positive, point +12 k€, [+4, +22] k€, more likely within resolution '
      f"(threshold 13,054). **Scored: {'see JSON' if doc['include_a64'] else PENDING}.**")
    a('')
    a('## Checks')
    a('')
    for k, v in doc['checks'].items():
        a(f'- {k}: {v}')
    if doc['failed_checks']:
        a('')
        a('**FAILED:** ' + '; '.join(doc['failed_checks']))
    a('')
    return '\n'.join(L) + '\n'


# ======================================================================================================================
def pickle_state():
    base = W145.pickle_state()
    counts = dict(base['counts'], w157=dict(PICKLE_COUNTS))
    blocked = pickle.load is not _PICKLE_ORIG[0] and pickle.loads is not _PICKLE_ORIG[1]
    ok = blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())
    return {'counts': counts, 'pickle_load_and_loads_blocked': blocked, 'ok': ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--include-a64', action='store_true', help='fill T10 from the W155 summary (after W156 only)')
    ap.add_argument('--a64-summary', default=A64_SUMMARY_DEFAULT)
    args = ap.parse_args()
    t0 = time.time()
    tag = 'W157-A64' if args.include_a64 else 'W157'
    out_dir = OUT_DIR_A64 if args.include_a64 else OUT_DIR
    for name in (OUT_JSON, OUT_MD, OUT_MAN):
        if os.path.exists(os.path.join(REPO, out_dir, name)):
            _log(f'[{tag} PRECONDITION FAILED] {out_dir}/{name} exists (write-once)')
            sys.exit(1)
    script_clean = L132._committed_clean(SCRIPT_REL)
    _log(f'[{tag}] script {SCRIPT_REL} sha256 {_sha(SCRIPT_REL)} committed clean {script_clean}; include_a64 '
         f'{args.include_a64}')
    docs, inputs, problems = load_inputs()
    if problems:
        _log(f'[{tag} PRECONDITION FAILED] inputs: {problems}')
        sys.exit(1)
    _log(f'[{tag}] inputs: {len(inputs)} read, all committed clean; manifests matched '
         f'{sum(1 for v in inputs.values() if v.get("manifest_sha256_matches") is True)}')
    failed = []
    cells = cells_table(docs)
    claims, p = claims_table(docs, cells)
    failed += p
    be, p = break_even(docs, cells)
    failed += p
    yl, p = year_ladder(docs)
    failed += p
    a64, p, a64_inputs = a64_rows(docs, args.include_a64, args.a64_summary)
    failed += p
    for k, v in a64_inputs.items():
        inputs[k] = {'path': k, 'sha256': v, 'committed_clean': True}
    tables = {'claims': claims, 'cells': cells, 'break_even': be, 'year_ladder': yl,
              'phase_b': phase_b(docs, claims), 'benchmark': benchmark(docs), 'discount': discount(docs),
              'ageing': ageing(docs), 'dead_zone': dead_zone(docs, cells), 'a64': a64}
    by_id = {r['claim_id']: r for r in claims}
    j4 = by_id['J:e4_value_minus_I'].get('manuscript_figure_A64') or {}
    nets = [r['net_label'] for r in claims]
    checks = {
        'claims_60': len(claims) == 60,
        'net_labels_8_recorded_G_rows_52_validated': (nets.count(NET_LABEL_RECORDED) == 8
                                                      and all(r['family'] == 'G' for r in claims
                                                              if r['net_label'] == NET_LABEL_RECORDED)
                                                      and nets.count(NET_LABEL_VALIDATED) == 52),
        'cells_49_distinct': len(cells) == 49,
        'every_claim_cell_in_cells_table': all(r['ref_cell'] in cells and r['other_cell'] in cells for r in claims),
        'W145_committed_result_reproduced_bitwise': all(v['equal'] for v in be['committed_reproduced'].values()),
        'conservative_bar_d_4a82a64a_70.7k': abs(be['conservative_A64']['intervals']['n7_4h_e3']['bar'] - 70743.93) < 0.01,
        'conservative_margin_min_61.3k': abs(be['manuscript_figure']['margin_min_eur_per_mwh'] - 61300) < 100,
        'J4_uncertified_bar_15880_and_11.1x': (abs((j4.get('bar') or 0) - 15879.58) < 1.0
                                               and abs((j4.get('multiple') or 0) - 11.12) < 0.01),
        'E_C2_calfade_vs_C3_within_resolution':
            by_id['E:n7_4h_e1_C2_calfade:vs_C3']['gross_verdict'] == 'within resolution',
        'year_ladder_net_-2286.25': round(yl['D_net'], 2) == -2286.25,
        'a64_rows_4': len(a64['rows']) == 4,
        'a64_rows_pending_unless_included': (args.include_a64
                                             or all(r['status'] == PENDING for r in a64['rows'])),
    }
    failed += [k for k, v in checks.items() if v is not True]
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    pk = pickle_state()
    guards_ok = all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']
    doc = {'schema': 'p515_s53_w157_step6_tables_v1', 'stage': 'P5.15 W157 -- Step 6 tables (draft)', 'utc': _utc(),
           'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 64 (Step 6 package item (i); rulings 3-7); Planner task W157',
           'definition': __doc__, 'git_head': _git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'last_commit': _git('log', '-1', '--format=%H', '--', SCRIPT_REL) or None},
           'code_sha256': {rel: _sha(rel) for rel in ('p515_s53_w145_banded_breakeven_fit.py',
                                                      'p515_s53_w142_determinacy.py',
                                                      'p515_s53_w132_resettle_v3_campaign.py',
                                                      'settling_criterion_v6.py', 'gate_result_io.py')},
           'objective_convention': OBJECTIVE_CONVENTION, 'constants': {'TAU': TAU, 'TWO_TAU': TWO_TAU,
                                                                       'FLAG_RANGE_OVER_TAU': FLAG_RANGE_OVER_TAU},
           'net_labels': {'validated': NET_LABEL_VALIDATED, 'recorded': NET_LABEL_RECORDED},
           'include_a64': args.include_a64, 'inputs': inputs, 'provenance': provenance(docs), 'tables': tables,
           'checks': checks, 'failed_checks': failed, 'guards': guards, 'pickle_guard': pk}
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    doc['exit_code'] = code
    doc['wall_s'] = time.time() - t0
    os.makedirs(os.path.join(REPO, out_dir), exist_ok=True)
    jrel = os.path.join(out_dir, OUT_JSON)
    with open(os.path.join(REPO, jrel), 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True)
    mrel = os.path.join(out_dir, OUT_MD)
    with open(os.path.join(REPO, mrel), 'x') as handle:
        handle.write(markdown(doc))
    man = {jrel: _sha(jrel), mrel: _sha(mrel), SCRIPT_REL: _sha(SCRIPT_REL),
           **{v['path']: v['sha256'] for v in inputs.values()}}
    with open(os.path.join(REPO, out_dir, OUT_MAN), 'x') as handle:
        GRIO.dump(man, handle, indent=1, sort_keys=True)
    _log(f'[{tag}] T1 claims {len(claims)}; T2 cells {len(cells)}; break-even margin (conservative) '
         f"{be['manuscript_figure']['margin_min_eur_per_mwh']:.2f}; committed W145 reproduced "
         f"{checks['W145_committed_result_reproduced_bitwise']}; J4 bar {j4.get('bar')} x {j4.get('multiple')}; "
         f"year ladder net {yl['D_net']:.4f}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    if failed:
        _log(f'[{tag}] FAILED: {failed}')
    _log(f"[{tag}] guards {[(k, v['counts']['permitted_solve'], v['counts']['blocked_solve'], v['verify_0_failures']) for k, v in guards.items()]}; "
         f"pickle {pk}; wrote {jrel}, {mrel}, {out_dir}/{OUT_MAN}; exit {code}; wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(code)


if __name__ == '__main__':
    main()
