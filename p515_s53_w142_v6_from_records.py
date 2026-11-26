"""P5.15 Planner task W142 items 1-3 (PLANNER_BRIEF_2026-09-13.md Addendum 61) -- CRITERION v6 FROM RECORDS, NO RE-RUNS:
v6 on every committed re-settle record; d_c52e1670's v6-from-records CERTIFICATE; the determinacy floor applied to every
claim scored so far; the post-certification movement record. ZERO SOLVES, NO MODEL LOADS.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end (every imported module's own zero-permit guard is verified at 0 as well); `pickle.load` / `pickle.loads` are blocked
for the whole run (W141's module, imported, installs the blocking counters at import) and verified at 0. Only JSON /
JSONL records are read.

WHAT IS COMPUTED
  1. v6 (`settling_criterion_v6.replay`, the frozen-to-be rule itself) on the 18 committed re-settle records of the W142
     floor replay (the non-clean cycles as the replay committed them, recomputed there from the logs and cross-checked),
     each against the replay's (A)+(B) arm, field for field: the rule v6 carries (Addendum 61 rule applied in the replay:
     'A_and_B').
  2. d_c52e1670 CERTIFIED UNDER v6 FROM ITS RECORDS (Planner reading of Addendum 61: the order names only the eight
     remaining D cells; the cell is not re-run): its committed v5 run (51280961; stage spec ab32ffc9, uncertified at its
     cap 198) holds every quantity v6 needs -- Q_k (per_cycle_record gross_operational_cost), boyd_k (all_boyd_pass AND
     local_solves_ok), t_sum_k and the in-run all_clean_k (resettle_cycle_record.jsonl) -- every input sha256-verified
     against the run's campaign manifest. The v6 decision over the recorded cycles 1..198 with the cell's v5 cap rule
     (fixed 198, ceiling 300) is written as the cell's certificate record, LABELLED "v6 from records": k*, window, W,
     P_hat, band, band width, range / TAU, t_sum(k*), the out-of-window reads, the turning points and the floor
     rejections, and the scorer's report of the cell at k* (Q, t_sum, Q_cc, Q_net, salvage, band, s = Q(k*) - Q_N_old).
     The in-run all_clean_k is cross-checked against the non-clean cycles the replay recomputed from the logs.
  3. THE DETERMINACY FLOOR ON EVERY CLAIM SCORED SO FAR (Addendum 61 ruling 2; `p515_s53_w142_determinacy`): certified
     cells -- determinate iff |margin| >= max(3 x the larger band width, 2 TAU); uncertified cells -- 3 x max(|gap|,
     |slack|) unchanged. Scored so far, each re-scored from its committed artefact (manifest-verified), with the verdict
     it carried reproduced first:
       B   the W138 claim point #3 summary (w137_summary_after_03_b_4649234b.json, f26e4437): every claim scored there
           (B n5_4h_e1, n7_4h_e1, n9_4h_e1, n9_4h_e3; the CHECK headline; the L F2-pair claim), on the views that summary
           recorded; and B:n9_4h_e3 again on cell 3's v5 certificate (4b3be392), which has since replaced the v4 cell;
       Phase B and the year ladder vs x = 0, and the year ladder 2035 - 2030 (w118_resettle_summary.json 'differences');
       F2: the F2 certificate (w118, both cells uncertified) and the F2 plan vs corner (W130 task 3, 2c1b731a).
     A verdict change is reported per claim (Addendum 61 predicts none).
  4. THE POST-CERTIFICATION MOVEMENT RECORD (Addendum 61 ruling 2): the two cells with records beyond a certifying cycle
     -- cell 3 (b_4649234b: certified at 148 under v5; its v4 run 0734f103 continued to 193, bitwise equal to the v5 run
     through 148, G26) and d_c52e1670 (v6 from records at 150; the last-pair reading's 148, W141; its v5 run continued to
     198): Q(last) - Q(k*) and the largest |Q(c) - Q(k*)| over c > k*, in EUR and in TAU; the statement "cells continued
     past certification moved at most 0.9 tau" checked against them.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w142_resettle_v6/v6_from_records/):
  w142_v6_from_records.json, d_c52e1670_v6_from_records_certificate.json, manifest_sha256.json (launch.log beside)
Launch (attached, alone, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w142_resettle_v6/v6_from_records && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w142_v6_from_records.py \\
        > data/SRP1/Results/P515S53/w142_resettle_v6/v6_from_records/launch.log 2>&1
"""
import contextlib
import hashlib
import io
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W142 v6 from records (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402
import p515_s53_w141_swing_variants as W141  # noqa: E402 -- record inputs (arms its guards; blocks pickle loads)
import p515_s53_w142_determinacy as DET  # noqa: E402 -- the v6 scorer (imports W132's launcher: arms its guards)
import p515_s53_w142_resettle_v6_hooks as V6  # noqa: E402

W139 = W141.W139
L132 = DET.L132


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w142_v6_from_records', GUARD),) + tuple(W141.GUARDS) + tuple(L132.GUARDS))

TAU = SC6.TAU
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR_REL = os.path.join(S53, 'w142_resettle_v6', 'v6_from_records')
OUT_JSON = 'w142_v6_from_records.json'
OUT_CERT = 'd_c52e1670_v6_from_records_certificate.json'
OUT_MAN = 'manifest_sha256.json'
REPLAY = {'path': os.path.join(S53, 'w142_resettle_v6', 'floor_replay', 'w142_floor_replay.json'),
          'manifest': os.path.join(S53, 'w142_resettle_v6', 'floor_replay', 'manifest_sha256.json')}
W138_SUMMARY = {'path': os.path.join(S53, 'w137_resettle_v4', 'w137_summary_after_03_b_4649234b.json'),
                'manifest': os.path.join(S53, 'w137_resettle_v4', 'w137_summary_after_03_b_4649234b_manifest_sha256.json'),
                'commit': 'f26e4437'}
W118_SUMMARY = {'path': os.path.join(S53, 'w118_resettle', 'w118_resettle_summary.json'),
                'manifest': os.path.join(S53, 'w118_resettle', 'w118_resettle_summary_manifest_sha256.json')}
W130 = {'path': os.path.join(S53, 'w130_benchmark_addendum', 'w130_benchmark_addendum.json'),
        'manifest': os.path.join(S53, 'w130_benchmark_addendum', 'manifest_sha256.json'), 'commit': '2c1b731a'}
CELL3_V5 = {'campaign_root': V6.CERTIFIED_UNDER_V5['b_4649234b']['campaign_root'], 'commit': '4b3be392'}
CELL3_V4_RECORD = 'b_4649234b@v4'
D_RECORD = 'd_c52e1670@v5'
D_CELL = 'd_c52e1670'
MOVEMENT_STATEMENT = 'cells continued past certification moved at most 0.9 tau'
MOVEMENT_BOUND_TAU = 0.9


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), 'rb') as handle:
        for b in iter(lambda: handle.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _jsonl(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pinned(src):
    """(doc, sha256) of a committed artefact whose manifest lists it with that sha256."""
    man = json.load(open(os.path.join(REPO, src['manifest'])))
    sha = _sha(src['path'])
    if man.get(src['path']) != sha:
        raise RuntimeError(f"{src['path']}: sha256 {sha} != its manifest {man.get(src['path'])}")
    return json.load(open(os.path.join(REPO, src['path']))), sha


def _core(s):
    return W141._core(s)


# ======================================================================================================================
#  1 + 2
# ======================================================================================================================
def v6_on_records(replay_doc, recs):
    out, mismatch = {}, []
    for rid in replay_doc['records']:
        rec = recs[rid]
        nc = set(replay_doc['reports'][rid]['non_clean_cycles'])
        clean = {k: k not in nc for k in rec['q']}
        _o, d6, r6 = SC6.replay(rec['q'], rec['b'], rec['t'], clean, V6.P_MAX, last=rec['n'], **rec['kw'])
        s6 = W141.summarize(d6, r6, rec['q'], rec['n'])
        ab = replay_doc['reports'][rid]['arms']['AB']
        eq = json.dumps(_core(s6), sort_keys=True, default=str) == json.dumps(
            {k: ab.get(k) for k in _core(s6)}, sort_keys=True, default=str)
        if not eq:
            mismatch.append(rid)
        v0 = replay_doc['reports'][rid]['arms']['V0']
        out[rid] = {'v6': {k: s6.get(k) for k in ('status', 'k_star', 'k_cap', 'window', 'branch', 'W', 'P_hat',
                                                   'band_width', 'range_over_tau', 't_sum_k_star', 'Q_k_star', 'T', 'A',
                                                   'n_vetoes', 'floor_rejections')},
                    'v6_equals_replay_arm_AB': eq,
                    'v5_V0': {k: v0.get(k) for k in ('status', 'k_star', 'window')},
                    'committed_run': rec['run'], 'certifying_spec': rec.get('certifying'),
                    'v6_changes_vs_v5': (s6.get('status'), s6.get('k_star'), s6.get('window')) != (
                        v0.get('status'), v0.get('k_star'), v0.get('window'))}
    return out, mismatch


def d_cell_certificate(recs, replay_doc):
    rec = recs[D_RECORD]
    ev = rec['eval_dir']
    q, b, t, n, kw = rec['q'], rec['b'], rec['t'], rec['n'], rec['kw']
    clean_in_run = rec['in_run_all_clean']
    if not all(isinstance(v, bool) for v in clean_in_run.values()):
        raise RuntimeError('d_c52e1670: in-run all_clean_k is not a bool on every cycle')
    nc_run = sorted(k for k, v in clean_in_run.items() if not v)
    nc_logs = replay_doc['reports'][D_RECORD]['non_clean_cycles']
    out_recs, dec, rule = SC6.replay(q, b, t, clean_in_run, V6.P_MAX, last=n, **kw)
    if dec is None or dec.get('status') != 'certified':
        raise RuntimeError(f'd_c52e1670: v6 from records did not certify within 1..{n}: {dec}')
    k = dec['k_star']
    rows = {r['cycle']: r for r in _jsonl(os.path.join(ev, 'per_cycle_record.jsonl'))}
    lines = {x['cycle']: x for x in _jsonl(os.path.join(ev, V6.CYCLE_FILE))}
    c = V6.V5.CELLS[D_CELL]
    ref_rel = V6.reference_path(D_CELL)
    ref = {r['cycle']: r for r in _jsonl(ref_rel)}
    q_n_old = ref[c['N_old']]['gross_operational_cost']
    v5res = json.load(open(os.path.join(REPO, V6.CERTIFIED_V6_FROM_RECORDS[D_CELL]['campaign_root'],
                                        'campaign_results.json')))
    v5rep = v5res['cell_report']
    evrec = json.load(open(os.path.join(REPO, ev, 'evaluation_record.json')))
    q_k, t_k = q[k], t[k]
    step_k = q[k] - q[k - 1]
    report = {
        'cell': D_CELL, 'item': c['item'], 'claim_group': V6.GROUP_OF_ITEM[c['item']], 'gated': c['gated'],
        'label': 'v6 from records', 'status': 'certified', 'k_star': k, 'end_cycle': k, 'branch': dec['branch'],
        'window': dec['window'], 'W': dec['W'], 'P_hat': dec['P_hat'], 'band': dec['band'],
        'band_width': dec['band_width'], 'range_over_tau': dec['range_over_tau'], 'Q_k_star': q_k,
        't_sum_k_star': t_k, 'Q_cc_k_star': q_k + t_k, 'Q_end': q_k, 't_sum_end': t_k, 'Q_cc_end': q_k + t_k,
        'Q_net_end': rows[k]['recourse'], 'terminal_salvage_value_end': rows[k]['terminal_salvage_value'],
        'N_old': c['N_old'], 'k0_original': c['k0'], 'Q_N_old': q_n_old, 's_signed': q_k - q_n_old,
        's_resolution_band_width': dec['band_width'], 'k0_run': v5rep.get('k0_run'),
        'first_k0_v2': v5rep.get('first_k0_v2'), 'T': dec['T'], 'A': dec['A'],
        'terminal_step_abs_at_k_star': abs(step_k), 'terminal_step_over_EPS0_at_k_star': abs(step_k) / SC6.EPS0,
        'certifying_spec': {'series': 'frozen_s53_resettle_spec', 'version': 6, 'mode': 'v6 from records',
                            'source_run_stage_spec': 'frozen_s53_resettle_spec_v5_ab32ffc9'},
        'objective_convention': 'Q = gross_operational_cost, settlement EXCLUDED; Q_cc = Q + t_sum (diagnostic)'}
    report['view'] = L132.view_from_report(report)
    after = [c_ for c_ in sorted(q) if c_ > k]
    cert = {
        'schema': 'p515_s53_w142_v6_from_records_certificate_v1',
        'label': 'v6 from records',
        'cell': D_CELL, 'orig_label': c['orig_label'], 'item': 'D',
        'authority': ('PLANNER_BRIEF_2026-09-13.md Addendum 61 (swing noise floor; turning-point floor carried by the '
                      'replay rule); Planner task W142 item 2 (Planner reading: the order names only the eight remaining '
                      'D cells -> d_c52e1670 is certified from its committed records under v6, not re-run)'),
        'instance': {'candidate_key': evrec.get('candidate_key'), 'candidate_label': evrec.get('candidate_label'),
                     'candidate_canonical': evrec.get('candidate_canonical'), 'eval_key': evrec.get('eval_key'),
                     'original_eval_key': c['orig_eval_key']},
        'source_run': {'commit': V6.CERTIFIED_V6_FROM_RECORDS[D_CELL]['source_v5_run_commit'], 'eval_dir': ev,
                       'campaign_spec': v5res.get('campaign_spec_path'),
                       'campaign_spec_sha256': v5res.get('campaign_spec_sha256'),
                       'stage_spec': v5res.get('stage_spec'), 'status_under_v5': v5rep.get('status'),
                       'cycles_recorded': n, 'cap_rule_replayed': kw},
        'criterion': {'module': 'settling_criterion_v6.py', 'sha256': _sha('settling_criterion_v6.py'),
                      'version': SC6.VERSION, 'carries': list(SC6.CARRIES), 'swing_floor': SC6.SWING_FLOOR},
        'inputs': {'Q_k': 'per_cycle_record.jsonl gross_operational_cost',
                   'boyd_k': 'per_cycle_record.jsonl boyd_all_pass AND local_solves_ok',
                   't_sum_k': f'{V6.CYCLE_FILE} t_sum', 'all_clean_k': f'{V6.CYCLE_FILE} all_clean_k (in-run capture)',
                   'inputs_sha256_verified_against_the_campaign_manifest': rec['inputs_sha256']},
        'all_clean_crosscheck': {'in_run_non_clean_cycles': nc_run, 'recomputed_from_the_logs_non_clean_cycles': nc_logs,
                                 'equal': nc_run == nc_logs},
        'k_star': k, 'window': dec['window'], 'W': dec['W'], 'P_hat': dec['P_hat'], 'branch': dec['branch'],
        'band': dec['band'], 'band_width': dec['band_width'], 'range_over_tau': dec['range_over_tau'],
        't_sum_k_star': dec['t_sum_k_star'], 'abs_t_sum_over_gap_bound': abs(dec['t_sum_k_star']) / SC6.GAP_BOUND,
        'Q_k_star': dec['Q_k_star'], 'Q_cc_k_star': dec['Q_cc_k_star'], 'k0': dec['k0'], 'N': dec['N'],
        'T': dec['T'], 'A': dec['A'], 'turning_point_floor_rejections': dec.get('turning_point_floor_rejections'),
        'out_of_window_reads': dec.get('out_of_window_reads'), 'vetoes': dec.get('vetoes'),
        'non_clean_cycles': dec.get('non_clean_cycles'), 'gap_refusals': dec.get('gap_refusals'),
        'terminal_step_to_threshold': {'terminal_step_abs': abs(step_k), 'over_EPS0': abs(step_k) / SC6.EPS0,
                                       'range_over_tau': dec['range_over_tau']},
        'recorded_after_k_star': {'cycles': [after[0], after[-1]] if after else None,
                                  'note': ('the v5 run continued to its cap; these cycles are NOT part of the '
                                           'certificate (a v6 run would have stopped at k*); see the movement record')},
        'decision': dec,
        'report': report,
        'per_cycle_rule_records_first_and_last': [out_recs[0], out_recs[-1]],
    }
    return cert, report


# ======================================================================================================================
#  3: the determinacy re-scoring
# ======================================================================================================================
def rescore_w138(w138):
    rows = []
    for s in w138['claims_scored']:
        if str(s.get('verdict', '')).startswith('not scored'):
            continue
        cl = {k: s[k] for k in ('claim_id', 'item', 'statement', 'claim_type', 'form', 'net_of_salvage', 'I_ref',
                                'I_other')}
        cl['I_source'] = None
        rv, ov = s['ref'], s['other']
        old = L132.score_claim(cl, rv, ov)
        new = DET.score_claim_v6(cl, rv, ov)
        rows.append({'claim_id': s['claim_id'], 'source': 'W138 claim point #3 (w137_summary_after_03, f26e4437)',
                     'ref_status': rv.get('status'), 'other_status': ov.get('status'),
                     'bands': [rv.get('band'), ov.get('band')], 'd_Q': new.get('d_Q'), 'd_Qcc': new.get('d_Qcc'),
                     'verdict_committed': s.get('verdict'), 'verdict_reproduced_with_the_old_rule': old.get('verdict'),
                     'reproduced': old.get('verdict') == s.get('verdict') and old.get('d_Q') == s.get('d_Q'),
                     'verdict_v6': new.get('verdict'), 'verdict_changed': new.get('verdict') != s.get('verdict'),
                     'gross_v6': new['gross']})
    return rows


def rescore_b_n9_e3_current(w138, cell3_view):
    s = next(x for x in w138['claims_scored'] if x['claim_id'] == 'B:n9_4h_e3')
    cl = {k: s[k] for k in ('claim_id', 'item', 'statement', 'claim_type', 'form', 'net_of_salvage', 'I_ref', 'I_other')}
    cl['I_source'] = None
    old = L132.score_claim(cl, s['ref'], cell3_view)
    new = DET.score_claim_v6(cl, s['ref'], cell3_view)
    return {'claim_id': 'B:n9_4h_e3', 'source': ('the W138 claim with the other cell replaced by cell 3\'s v5 certificate '
                                                 '(4b3be392, certified at 148)'),
            'ref_status': s['ref'].get('status'), 'other_status': cell3_view.get('status'),
            'bands': [s['ref'].get('band'), cell3_view.get('band')], 'd_Q': new.get('d_Q'), 'd_Qcc': new.get('d_Qcc'),
            'verdict_committed_at_W138_on_the_v4_cell': s.get('verdict'),
            'verdict_old_rule_on_the_v5_certificate': old.get('verdict'), 'verdict_v6': new.get('verdict'),
            'verdict_changed_vs_old_rule_same_inputs': new.get('verdict') != old.get('verdict'), 'gross_v6': new['gross']}


def rescore_w118(w118):
    diff = w118['differences']
    x = diff['x0_comparator']
    x0_view = {'status': 'certified', 'band': x['band_width'], 'Q': x['Q181'], 't': x['t']}
    rows = []
    for cell, r in diff['phase_b_and_year_ladder_vs_x0'].items():
        rep = w118['reports'][cell]
        view = {'status': rep['status'], 'band': rep['band_width'], 'Q': rep['Q_k_star'], 't': rep['t_sum_k_star']}
        m = r['I_j'] + view['Q'] - x0_view['Q']
        m_cc = r['I_j'] + view['Q'] + view['t'] - (x0_view['Q'] + x0_view['t'])
        res = DET.difference_certified_pair(m, m_cc, x0_view, view)
        rows.append({'difference': f'{cell} vs x0 (M = I + Q(k*) - Q181)', 'source': 'w118_resettle_summary differences',
                     'M': m, 'M_cc': m_cc, 'reproduced': abs(m - r['M']) <= 1e-6 and abs(m_cc - r['M_cc']) <= 1e-6,
                     'recomputed_minus_committed': [m - r['M'], m_cc - r['M_cc']],
                     'bands': [x0_view['band'], view['band']], 'verdict_committed': r['verdict'],
                     'verdict_v6': res['verdict'], 'verdict_changed': res['verdict'] != r['verdict'],
                     'note': ('certificate EXCLUDED under v4+ (Addendum 59; the Acceptable recovery at k* 167 lies in its '
                              'window); re-scored as recorded, its re-run pending') if cell == 'pb_y2025_n5' else None,
                     'v6': res})
    yl = diff['year_ladder']
    r30, r35 = w118['reports']['yl_y2030'], w118['reports']['yl_y2035']
    i30 = diff['phase_b_and_year_ladder_vs_x0']['yl_y2030']['I_j']
    i35 = diff['phase_b_and_year_ladder_vs_x0']['yl_y2035']['I_j']
    v30 = {'status': r30['status'], 'band': r30['band_width'], 'Q': r30['Q_k_star'], 't': r30['t_sum_k_star']}
    v35 = {'status': r35['status'], 'band': r35['band_width'], 'Q': r35['Q_k_star'], 't': r35['t_sum_k_star']}
    d = (i35 + v35['Q']) - (i30 + v30['Q'])
    d_cc = d + (v35['t'] - v30['t'])
    res = DET.difference_certified_pair(d, d_cc, v30, v35)
    rows.append({'difference': 'year ladder 2035 - 2030 (D = (I + Q)_2035 - (I + Q)_2030)',
                 'source': 'w118_resettle_summary differences.year_ladder', 'M': d, 'M_cc': d_cc,
                 'reproduced': abs(d - yl['D_2035_minus_2030']) <= 1e-6 and abs(d_cc - yl['D_cc']) <= 1e-6,
                 'recomputed_minus_committed': [d - yl['D_2035_minus_2030'], d_cc - yl['D_cc']],
                 'bands': [v30['band'], v35['band']],
                 'verdict_committed': yl['verdict'], 'verdict_v6': res['verdict'],
                 'verdict_changed': res['verdict'] != yl['verdict'], 'v6': res})
    f2 = diff['f2_certificate']
    rows.append({'difference': 'F2 certificate (challenger vs incumbent)', 'source': 'w118_resettle_summary differences',
                 'M': f2.get('D'), 'verdict_committed': f2['verdict'],
                 'verdict_v6': f2['verdict'], 'verdict_changed': False, 'reproduced': f2.get('D') is None,
                 'bands': [f2['band_widths']['challenger'], f2['band_widths']['incumbent']],
                 'note': ('both cells uncertified: the Addendum 61 floor applies to certified cells only; the '
                          'uncertified-form rule is unchanged, so this verdict is unchanged by construction (its '
                          'uncertified-form scoring is the L F2-pair claim in the W138 summary, re-scored above)')})
    return rows


def rescore_w130(w130):
    t3 = w130['task3']
    inc, cor = t3['incumbent'], t3['corner']
    v_inc = {'status': 'uncertified', 'Q': inc['Q_at_cap'], 't': inc['t_sum_at_cap'],
             'Q_cc': inc['Q_at_cap'] + inc['t_sum_at_cap'], 'band': inc['band_width_last_60'],
             'gap': abs(inc['t_sum_at_cap']), 'slack': abs(inc['s_signed'])}
    v_cor = {'status': 'certified', 'Q': cor['Q'], 't': cor['t_sum'], 'Q_cc': cor['Q'] + cor['t_sum'], 'band': None}
    d_q = (cor['I'] + cor['Q']) - (inc['I'] + inc['Q_at_cap'])
    d_cc = (cor['I'] + cor['Q'] + cor['t_sum']) - (inc['I'] + inc['Q_at_cap'] + inc['t_sum_at_cap'])
    res = DET.resolve_v6(d_q, d_cc, (v_cor, v_inc))
    # W130's own bar reading: gap = max(|t corner|, |t incumbent|), slack = |s| of the incumbent (report beside)
    bar_w130 = 3.0 * max(abs(cor['t_sum']), abs(inc['t_sum_at_cap']), abs(inc['s_signed']))
    v6_verdict = 'stands' if res['verdict'] == 'determinate' else 'pending'
    return [{'difference': 'F2 plan (incumbent, uncertified at its cap) vs corner (old certificate f3aa335e)',
             'source': 'W130 task 3 (w130_benchmark_addendum.json, 2c1b731a)', 'M': d_q, 'M_cc': d_cc,
             'reproduced': abs(d_q - t3['margin_gross']) <= 1e-6 and abs(d_cc - t3['margin_Q_cc']) <= 1e-6,
             'recomputed_minus_committed': [d_q - t3['margin_gross'], d_cc - t3['margin_Q_cc']],
             'bar_committed': t3['bar'], 'bar_v6_uncertified_form': res.get('bar'), 'bar_w130_reading_recomputed': bar_w130,
             'verdict_committed': {'gross': t3['verdict_gross'], 'Q_cc': t3['verdict_Q_cc']},
             'verdict_v6': v6_verdict, 'verdict_changed': v6_verdict != t3['verdict_gross'],
             'note': ('the incumbent is uncertified: the uncertified-form rule applies, unchanged by Addendum 61 (the '
                      'corner\'s own slack is unmeasured, as W130 recorded)'), 'v6': res}]


# ======================================================================================================================
#  4: the post-certification movement record
# ======================================================================================================================
def movement(q, k_star, last):
    after = [c for c in sorted(q) if k_star < c <= last and q.get(c) is not None]
    dev = [(c, q[c] - q[k_star]) for c in after]
    c_max, d_max = max(dev, key=lambda x: abs(x[1])) if dev else (None, None)
    return {'k_star': k_star, 'last_recorded': last, 'Q_k_star': q[k_star], 'Q_last': q[last],
            'Q_last_minus_Q_k_star': q[last] - q[k_star], 'over_tau': (q[last] - q[k_star]) / TAU,
            'max_abs_deviation_after_k_star': {'cycle': c_max, 'Q_c_minus_Q_k_star': d_max,
                                               'over_tau': (abs(d_max) / TAU) if d_max is not None else None},
            'n_cycles_after': len(after)}


def movement_record(recs, cert):
    q3 = recs[CELL3_V4_RECORD]['q']
    n3 = recs[CELL3_V4_RECORD]['n']
    qd = recs[D_RECORD]['q']
    nd = recs[D_RECORD]['n']
    rows = {
        'b_4649234b (cell 3)': dict(movement(q3, 148, n3), source=('the v4 run 0734f103 (b_4649234b@v4), which continued '
                                                                   'to its cap 193; certified at 148 under v5 (run '
                                                                   '4b3be392, bitwise equal to the v4 run through 148: '
                                                                   'G26 held)')),
        'd_c52e1670 at its v6 certificate': dict(movement(qd, cert['k_star'], nd),
                                                 source='the v5 run 51280961, which continued to its cap 198'),
        'd_c52e1670 at the last-pair reading\'s 148 (W141 V1)': dict(movement(qd, 148, nd),
                                                                     source='the v5 run 51280961 (W141 V1 / V2)'),
    }
    endpoint = max(abs(r['over_tau']) for r in rows.values())
    excursion = max(r['max_abs_deviation_after_k_star']['over_tau'] for r in rows.values())
    return {'statement': MOVEMENT_STATEMENT, 'bound_tau': MOVEMENT_BOUND_TAU, 'tau': TAU, 'rows': rows,
            'largest_endpoint_movement_over_tau': endpoint, 'largest_excursion_over_tau': excursion,
            'statement_holds_on_the_endpoint_movement': endpoint <= MOVEMENT_BOUND_TAU,
            'statement_holds_on_the_largest_excursion': excursion <= MOVEMENT_BOUND_TAU,
            'scope': ('the only committed records that continue past a certifying cycle (every other run stops at its '
                      'k*); searched: the 18 re-settle records of the W142 floor replay')}


# ======================================================================================================================
def main():
    t0 = time.time()
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_JSON, OUT_CERT, OUT_MAN):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    replay_doc, replay_sha = _pinned(REPLAY)
    if replay_doc['decision']['v6_carries'] != 'A_and_B' or list(SC6.CARRIES) != ['A_growth_test_floor',
                                                                                  'B_turning_point_floor']:
        raise RuntimeError('settling_criterion_v6 does not carry what the committed replay decided')
    w138, w138_sha = _pinned(W138_SUMMARY)
    w118, w118_sha = _pinned(W118_SUMMARY)
    w130, w130_sha = _pinned(W130)
    with contextlib.redirect_stdout(io.StringIO()):
        recs = W139.record_inputs()
    recs.update(W141.v5_run_inputs())
    v6_all, mismatch = v6_on_records(replay_doc, recs)
    cert, d_report = d_cell_certificate(recs, replay_doc)
    c5_res_rel = os.path.join(CELL3_V5['campaign_root'], 'campaign_results.json')
    c5_man = json.load(open(os.path.join(REPO, CELL3_V5['campaign_root'], 'campaign_manifest_sha256.json')))
    c5_res = json.load(open(os.path.join(REPO, c5_res_rel)))
    cell3_view = c5_res['cell_report']['view']
    b_rows = rescore_w138(w138)
    b_current = rescore_b_n9_e3_current(w138, cell3_view)
    w118_rows = rescore_w118(w118)
    w130_rows = rescore_w130(w130)
    all_rows = b_rows + [b_current] + w118_rows + w130_rows
    changes = [r for r in all_rows if r.get('verdict_changed') or r.get('verdict_changed_vs_old_rule_same_inputs')]
    not_reproduced = [r.get('claim_id') or r.get('difference') for r in all_rows if r.get('reproduced') is False]
    mov = movement_record(recs, cert)
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    git_head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    code = {rel: _sha(rel) for rel in (
        os.path.basename(__file__), 'settling_criterion_v6.py', 'settling_criterion_v5.py', 'settling_criterion_v2.py',
        'settling_criterion.py', 'p515_s53_w142_determinacy.py', 'p515_s53_w142_resettle_v6_hooks.py',
        'p515_s53_w141_swing_variants.py', 'p515_s53_w139_v5_from_records.py', 'p515_s53_w132_resettle_v3_campaign.py')}
    cert_path = os.path.join(out_dir, OUT_CERT)
    cert['code_sha256'] = code
    cert['utc'] = _utc()
    cert['git_head'] = git_head
    with open(cert_path, 'x') as handle:
        GRIO.dump(cert, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    doc = {'schema': 'p515_s53_w142_v6_from_records_v1',
           'task': 'W142 items 1-3 (Addendum 61): v6 from records; d_c52e1670 certificate; determinacy re-score; movement',
           'utc': _utc(), 'git_head': git_head, 'code_sha256': code, 'definition': __doc__,
           'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED; Q_cc = Q + t_sum (diagnostic); '
                                    'margins F-form I + Q (gross) unless a claim states net of salvage'),
           'inputs': {'floor_replay': {**REPLAY, 'sha256': replay_sha},
                      'w138_summary': {**W138_SUMMARY, 'sha256': w138_sha},
                      'w118_summary': {**W118_SUMMARY, 'sha256': w118_sha}, 'w130': {**W130, 'sha256': w130_sha},
                      'cell3_v5_results': {'path': c5_res_rel, 'sha256': _sha(c5_res_rel),
                                           'in_its_campaign_manifest': c5_man.get(c5_res_rel) == _sha(c5_res_rel)}},
           'constants': {'TAU': TAU, 'SWING_FLOOR': SC6.SWING_FLOOR, 'TWO_TAU': 2 * TAU, 'GAP_BOUND': SC6.GAP_BOUND},
           'item1_v6_carries': replay_doc['decision']['v6_carries'],
           'item1_v6_on_every_record': v6_all, 'item1_v6_equals_replay_arm_AB_every_record': not mismatch,
           'item1_mismatches': mismatch,
           'item1_v6_changes_vs_v5': [rid for rid, v in v6_all.items() if v['v6_changes_vs_v5']],
           'item2_d_c52e1670_certificate': {'path': os.path.relpath(cert_path, REPO),
                                            'sha256': _sha(os.path.relpath(cert_path, REPO)),
                                            'k_star': cert['k_star'], 'window': cert['window'], 'W': cert['W'],
                                            'band': cert['band'], 'band_width': cert['band_width'],
                                            't_sum_k_star': cert['t_sum_k_star'], 'Q_k_star': cert['Q_k_star'],
                                            'all_clean_crosscheck': cert['all_clean_crosscheck'],
                                            'report_view': d_report['view']},
           'item3_determinacy_rescore': {'rule': DET.RULE_TEXT, 'rows': all_rows, 'verdict_changes': changes,
                                         'n_rescored': len(all_rows), 'not_reproduced': not_reproduced,
                                         'prediction_addendum61': 'changes no verdict to date',
                                         'prediction_held': not changes},
           'item3_movement_record': mov,
           'guards': guards, 'pickle_guard': dict(W141.PICKLE_COUNTS), 'wall_s': time.time() - t0}
    jp = os.path.join(out_dir, OUT_JSON)
    with open(jp, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    man = {os.path.relpath(jp, REPO): _sha(os.path.relpath(jp, REPO)),
           os.path.relpath(cert_path, REPO): _sha(os.path.relpath(cert_path, REPO)),
           REPLAY['path']: replay_sha, W138_SUMMARY['path']: w138_sha, W118_SUMMARY['path']: w118_sha,
           W130['path']: w130_sha, c5_res_rel: _sha(c5_res_rel)}
    for rid in replay_doc['records']:
        for rel, v in recs[rid]['inputs_sha256'].items():
            if isinstance(v, str):
                man[rel] = v
            elif isinstance(v, dict) and isinstance(v.get('sha256'), str):
                man[rel] = v['sha256']
    with open(os.path.join(out_dir, OUT_MAN), 'x') as handle:
        GRIO.dump(man, handle, indent=1, sort_keys=True)
    _log(f"item 1: v6 carries {doc['item1_v6_carries']}; v6 == replay AB on every record {not mismatch}; v6 changes vs v5: "
         f"{doc['item1_v6_changes_vs_v5']}")
    _log(f"item 2: d_c52e1670 v6 from records: k* {cert['k_star']} window {cert['window']} W {cert['W']} P_hat "
         f"{cert['P_hat']} band {cert['band']} width {cert['band_width']:.2f} range/tau {cert['range_over_tau']:.4f} "
         f"t_sum {cert['t_sum_k_star']:.2f} Q {cert['Q_k_star']!r} s {d_report['s_signed']:.2f} T {cert['T']} A {cert['A']} "
         f"rejections {cert['turning_point_floor_rejections']} all_clean x-check {cert['all_clean_crosscheck']['equal']}")
    oow = cert['out_of_window_reads'] or {}
    _log(f"item 2: out-of-window reads: turning points outside the window {oow.get('turning_points_outside_window')}; "
         f"non-clean read outside {oow.get('non_clean_cycles_read_outside_window')}; any out-of-window read "
         f"{oow.get('any_out_of_window_read')}")
    for r in all_rows:
        g = r.get('gross_v6') or r.get('v6') or {}
        _log(f"item 3: {r.get('claim_id') or r.get('difference')}: d {r.get('d_Q', r.get('M'))} committed "
             f"{r.get('verdict_committed', r.get('verdict_committed_at_W138_on_the_v4_cell'))} -> v6 {r['verdict_v6']} "
             f"(rule {g.get('rule')}, threshold {g.get('threshold', g.get('bar'))}, binds {g.get('binding_term')}) "
             f"reproduced {r.get('reproduced')}")
    _log(f"item 3: verdict changes {[(r.get('claim_id') or r.get('difference')) for r in changes]}; not reproduced "
         f"{not_reproduced}")
    for name, r in mov['rows'].items():
        _log(f"movement {name}: Q(last {r['last_recorded']}) - Q(k* {r['k_star']}) = {r['Q_last_minus_Q_k_star']:.2f} "
             f"({r['over_tau']:.3f} tau); max |dev| {r['max_abs_deviation_after_k_star']}")
    _log(f"movement: largest endpoint {mov['largest_endpoint_movement_over_tau']:.3f} tau, largest excursion "
         f"{mov['largest_excursion_over_tau']:.3f} tau; statement '{MOVEMENT_STATEMENT}' holds on endpoint "
         f"{mov['statement_holds_on_the_endpoint_movement']}, on excursion {mov['statement_holds_on_the_largest_excursion']}")
    _log(f"guards {({k: v['verify_0_failures'] for k, v in guards.items()})}; pickle {W141.PICKLE_COUNTS}; wall "
         f"{time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = W141._PICKLE_ORIG
    ok = (all(not v['verify_0_failures'] for v in guards.values()) and not mismatch and not not_reproduced
          and cert['all_clean_crosscheck']['equal'] and W141.PICKLE_COUNTS == {'load': 0, 'loads': 0})
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
