"""P5.15 Planner task W142 item 1 (PLANNER_BRIEF_2026-09-13.md Addendum 61) -- THE SWING NOISE FLOOR REPLAYED FROM
RECORDS: the growth-test floor (A) alone, and (A) with the turning-point floor (B), on every committed re-settle record,
to decide MECHANICALLY what settling criterion v6 carries. ZERO SOLVES, NO MODEL LOADS.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end (every imported module's own zero-permit guard is verified at 0 as well); `pickle.load` / `pickle.loads` are
blocked for the whole run by W141's module (imported below; it installs the blocking counters at import) and verified at
0. Only JSON / JSONL records and IPOPT text logs are read.

NOTHING COMMITTED IS MODIFIED. The harness approach is W141's (`p515_s53_w141_swing_variants`, d02efe69), imported and
never edited: its record inputs (`W139.record_inputs` + `v5_run_inputs`), its readers (`block_finals` +
`classify_cycle`, i.e. `settling_criterion_v5.classify_block_exit`), its variant rule (`VariantRule`, a subclass of the
committed `settling_criterion_v5.SettlingRuleV5`) and its floor sign state (`FloorSignState`: W141's V3 semantics, "a
turning point registers only if the swing it closes is >= F; a rejected candidate un-registers T[-1] and is CARRIED as
the running extreme of the resumed leg; the rejected reversal is not a sign change for the monotone branch either").

THE ARMS (each differs from v5 ONLY in the named clause; the veto, the gap clause, the monotone branch, the caps are v5's):
  V0      v5 as frozen: `settling_criterion_v5.replay` itself (the committed code path). CONTROL.
  A       (A) THE GROWTH-TEST FLOOR (Addendum 61, adopted): in the "swings not growing" comparison a swing A[i] < F =
          TAU / 10 is EXCLUDED FROM THE COMPARISON; the turning points are still registered and P_hat and W still use
          them. Reading (Worker, recorded for Planner confirmation): the swings >= F, in their order, must be
          non-increasing over every consecutive pair of THAT sequence (an excluded swing does not break the chain: an
          earlier growth trend across it is still compared -- the property the expert gave for rejecting the last-pair
          test). All other clauses v5's.
  AB      (A) + (B) THE TURNING-POINT FLOOR (Addendum 61, conditional): as A, and a sign change whose closing swing is
          below F does not register a turning point (W141's V3 semantics, `FloorSignState`, F = TAU / 10).
  A_pairs SENSITIVITY (Worker-added, NOT a Planner arm, no decision reads it): the other reading of (A) -- compare only
          consecutive pairs (A[i], A[i+1]) of the ORIGINAL sequence both of whose swings are >= F (an excluded swing
          breaks the chain). Reported to show whether the reading matters on the records.
  A_f0    self-test: A at F = 0 (nothing excluded) must equal V0 on every record, field for field.
  AB_vs_W141_V3_10  self-test: AB must equal W141's committed V3_10 result on every record (under (B) every registered
          swing is >= F, so (A) excludes nothing and AB reduces to V3_10 BY CONSTRUCTION -- reported, not assumed).

THE DECISION RULE (Addendum 61, applied mechanically): "if (A)+(B) changes NO committed certificate relative to (A)
alone and V0, v6 carries both; if it changes any, v6 carries (A) only, and the inconsistency is recorded for the
cleanup." A COMMITTED CERTIFICATE = a record whose V0 decision (which reproduces the committed decision, W141) is
certified. "Changes" = a different status, k*, window or branch (so a different certified value Q(k*)). The records
that carry no committed certificate (d_c52e1670 uncertified under v5; pb_y2025_n5 excluded; the F2 pair uncertified) are
reported arm by arm; their changes do not enter the rule (they are not certificates).

THE RECORDS (18): W141's -- the 16 of W139 item 5 and the two committed v5 runs (b_4649234b@v5 4b3be392, d_c52e1670@v5
51280961). all_clean_k recomputed with W139's readers and cross-checked against W141's committed non_clean_cycles.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w142_resettle_v6/floor_replay/):
  w142_floor_replay.json, manifest_sha256.json (launch.log beside, from the shell)
Launch (attached, alone, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w142_resettle_v6/floor_replay && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w142_floor_replay.py \\
        > data/SRP1/Results/P515S53/w142_resettle_v6/floor_replay/launch.log 2>&1
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W142 floor replay (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import settling_criterion_v2 as SC2  # noqa: E402
import settling_criterion_v5 as SC5  # noqa: E402
import p515_s53_w141_swing_variants as W141  # noqa: E402 -- W141's harness (arms its guards; blocks pickle loads)
import p515_s53_w139_resettle_v5_hooks as V5  # noqa: E402

W139 = W141.W139


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w142_floor_replay', GUARD),) + tuple(W141.GUARDS))

TAU = SC5.TAU
F = TAU / 10.0
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR_REL = os.path.join(S53, 'w142_resettle_v6', 'floor_replay')
OUT_JSON = 'w142_floor_replay.json'
OUT_MAN = 'manifest_sha256.json'
W141_JSON = os.path.join(S53, 'w141_swing_variants', 'w141_swing_variants.json')
W141_MAN = os.path.join(S53, 'w141_swing_variants', 'manifest_sha256.json')
D_CELL = W141.D_CELL
ARMS = {
    'A': {'growth_floor': F, 'growth_reading': 'chain', 'tp_floor': None,
          'role': 'Addendum 61 (A): growth-test floor F = TAU / 10 (swings < F excluded from the comparison)'},
    'AB': {'growth_floor': F, 'growth_reading': 'chain', 'tp_floor': F,
           'role': 'Addendum 61 (A) + (B): growth-test floor and turning-point floor, F = TAU / 10'},
    'A_pairs': {'growth_floor': F, 'growth_reading': 'pairs', 'tp_floor': None,
                'role': ('Worker sensitivity (not a Planner arm; no decision reads it): (A) read as "compare only '
                         'original consecutive pairs both >= F"')},
    'A_f0': {'growth_floor': 0.0, 'growth_reading': 'chain', 'tp_floor': None,
             'role': 'self-test: A at F = 0 must equal V0'},
}
CHANGE_FIELDS = ('status', 'k_star', 'window', 'branch')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for b in iter(lambda: handle.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ======================================================================================================================
#  (A): the growth-test floor
# ======================================================================================================================
def growth_test(A, floor, reading):
    """(ok, detail). `reading` 'chain': the swings >= floor, in order, non-increasing over every consecutive pair of that
    subsequence; 'pairs': only consecutive ORIGINAL pairs both >= floor are compared."""
    excluded = [i for i, a in enumerate(A) if a < floor]
    if reading == 'chain':
        kept = [i for i, a in enumerate(A) if a >= floor]
        pairs = [[kept[j], kept[j + 1]] for j in range(len(kept) - 1)]
    elif reading == 'pairs':
        pairs = [[i, i + 1] for i in range(len(A) - 1) if A[i] >= floor and A[i + 1] >= floor]
    else:
        raise ValueError(reading)
    ok = all(A[j2] <= A[j1] for j1, j2 in pairs)
    return ok, {'excluded_swing_indices': excluded, 'pairs_compared': pairs, 'floor': floor, 'reading': reading}


class FloorArmRule(W141.VariantRule):
    """W141's VariantRule (v5 with the sign state optionally W141's FloorSignState = (B)) with the swing test replaced
    by the growth-test floor (A). evaluate() is W141's VariantRule.evaluate with the swing test swapped (version 2's
    clauses called, cert_a / cert_b / reasons recomputed, version 5's veto reproduced)."""

    def __init__(self, p_max, growth_floor, growth_reading, tp_floor, **kw):
        self.growth_floor = growth_floor
        self.growth_reading = growth_reading
        super().__init__(p_max, swing_kind='all_pairs', floor=tp_floor, **kw)

    def evaluate(self, k):
        _ca, _cb, a_parts, b_parts, _r = SC2.SettlingRuleV2.evaluate(self, k)
        ok, det = growth_test(self.sign.A, self.growth_floor, self.growth_reading)
        a_parts.update({'swing_test': f'growth_floor_{self.growth_reading}', 'swings_ok_variant': ok,
                        'swing_pairs_compared': det['pairs_compared'], 'floor': self._floor,
                        'growth_floor': self.growth_floor, 'excluded_swing_indices': det['excluded_swing_indices']})
        cert_a = bool(a_parts['at_least_3_turning_points'] and ok and a_parts['window_inside_run']
                      and a_parts['range_le_tau'])
        b_ok = bool(b_parts['lo_ge_k0_plus_K_EXCL'] and b_parts['no_sign_change_in_window'] and b_parts['range_le_tau']
                    and b_parts['steps_decreasing'] and b_parts['last_step_times_L_le_tau'])
        cert_b = (not cert_a) and b_ok
        reasons = []
        if not cert_a and not cert_b:
            if not a_parts['at_least_3_turning_points']:
                reasons.append('insufficient_turning_points')
            if not ok:
                reasons.append('swings_growing')
            if a_parts['window_inside_run'] is False:
                reasons.append('window_outside_run')
            if a_parts['range_le_tau'] is False or b_parts['range_le_tau'] is False:
                reasons.append('range_above_tau')
            if not (b_parts['lo_ge_k0_plus_K_EXCL'] and b_parts['no_sign_change_in_window']):
                reasons.append('monotone_window_not_reached')
            else:
                if not b_parts['steps_decreasing']:
                    reasons.append('monotone_not_decreasing')
                if not b_parts['last_step_times_L_le_tau']:
                    reasons.append('monotone_last_step_times_L_above_tau')
        vetoed = []
        if cert_a:
            hit = self._non_clean_in(a_parts['window'])
            if hit:
                cert_a = False
                vetoed.append({'branch': 'oscillatory', 'window': list(a_parts['window']), 'W': a_parts['W'],
                               'non_clean_in_window': hit})
                cert_b = b_ok
        if cert_b:
            hit = self._non_clean_in(b_parts['window'])
            if hit:
                cert_b = False
                vetoed.append({'branch': 'monotone', 'window': list(b_parts['window']), 'W': self.l_mono,
                               'non_clean_in_window': hit})
        if vetoed:
            reasons = list(reasons) + [SC5.VETO_REASON]
            self._cycle_veto = {'cycle': k, 'branches': vetoed, 'k0': self.k0}
            self.vetoes.append(dict(self._cycle_veto))
        return cert_a, cert_b, a_parts, b_parts, reasons


def replay_arm(q, b, t, clean, p_max, arm, last, **kw):
    rule = FloorArmRule(p_max, arm['growth_floor'], arm['growth_reading'], arm['tp_floor'], **kw)
    out = []
    k = 0
    while True:
        k += 1
        if last is not None and k > last:
            break
        if k > rule.effective_cap():
            break
        out.append(rule.observe(k, q.get(k), bool(b.get(k, False)), t.get(k), bool(clean.get(k, False))))
        if rule.decision is not None:
            break
    return out, rule.decision, rule


def _key(s):
    return {f: s.get(f) for f in CHANGE_FIELDS}


# ======================================================================================================================
def main():
    t0 = time.time()
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_JSON, OUT_MAN):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    w141_doc = json.load(open(os.path.join(REPO, W141_JSON)))
    w141_man = json.load(open(os.path.join(REPO, W141_MAN)))
    if w141_man.get(W141_JSON) != _sha(os.path.join(REPO, W141_JSON)):
        raise RuntimeError('the committed W141 output does not match its manifest')
    with contextlib.redirect_stdout(io.StringIO()):
        recs = W139.record_inputs()
    recs.update(W141.v5_run_inputs())
    order = list(w141_doc['records'])
    if sorted(order) != sorted(recs) or len(order) != 18:
        raise RuntimeError(f'expected W141\'s 18 records, got {sorted(recs)} vs {order}')
    reports, table = {}, []
    selftest_fail, clean_x = [], {}
    for rid in order:
        rec = recs[rid]
        finals, meta = W139.block_finals(rec['eval_dir'], rec['n'])
        if not meta['coverage_51_every_cycle']:
            raise RuntimeError(f'{rid}: block coverage is not 51 on every cycle')
        per_cycle = {k: W139.classify_cycle(finals[k]) for k in range(1, rec['n'] + 1)}
        clean = {k: all(e['clean'] for e in per_cycle[k].values()) for k in per_cycle}
        non_clean = sorted(k for k, v in clean.items() if not v)
        ref = w141_doc['reports'][rid]['non_clean_cycles']
        clean_x[rid] = {'against': 'W141 committed non_clean_cycles', 'equal': ref == non_clean, 'non_clean_cycles': non_clean}
        q, b, t, n, kw = rec['q'], rec['b'], rec['t'], rec['n'], rec['kw']
        _o, d0, r0 = SC5.replay(q, b, t, clean, V5.P_MAX, last=n, **kw)
        sums = {'V0': W141.summarize(d0, r0, q, n)}
        for name, arm in ARMS.items():
            _o, da, ra = replay_arm(q, b, t, clean, V5.P_MAX, arm, n, **kw)
            s = W141.summarize(da, ra, q, n)
            if da is not None and da.get('status') == 'certified':
                s['growth_test_at_kstar'] = {x: da['certA_parts'].get(x) for x in (
                    'swing_test', 'swings_ok_variant', 'swing_pairs_compared', 'growth_floor', 'excluded_swing_indices',
                    'floor')}
            sums[name] = s
        if W141._core(sums['A_f0']) != W141._core(sums['V0']):
            selftest_fail.append({'record': rid, 'self_test': 'A_f0 == V0', 'v0': W141._core(sums['V0']),
                                  'got': W141._core(sums['A_f0'])})
        v3 = w141_doc['reports'][rid]['variants']['V3_10']
        v3_core = {k: v3.get(k) for k in ('status', 'k_star', 'k_cap', 'branch', 'k0', 'W', 'P_hat', 'window', 'band',
                                          'band_width', 'range', 't_sum_k_star', 'reasons', 'T', 'A', 'n_vetoes',
                                          'n_gap_refusals', 'drift_rate_mean_dQ_last_25')}
        ab_core = W141._core(sums['AB'])
        ab_eq_v3 = json.dumps(v3_core, sort_keys=True, default=str) == json.dumps(ab_core, sort_keys=True, default=str)
        if not ab_eq_v3:
            selftest_fail.append({'record': rid, 'self_test': 'AB == W141 V3_10 (committed)', 'w141_v3_10': v3_core,
                                  'got': ab_core})
        v0_repro = w141_doc['reports'][rid]['v0_reproduces_committed']
        k0v, kav, kabv = _key(sums['V0']), _key(sums['A']), _key(sums['AB'])
        committed_certificate = sums['V0'].get('status') == 'certified'
        row = {'record': rid, 'cell': rec['cell'], 'committed_run': rec['run'],
               'committed_certificate': committed_certificate,
               'certifying_spec': rec.get('certifying'),
               'w141_v0_reproduced_the_committed_decision': v0_repro.get('equal'),
               'AB_changes_vs_A': kabv != kav, 'AB_changes_vs_V0': kabv != k0v, 'A_changes_vs_V0': kav != k0v,
               'A_pairs_changes_vs_A': _key(sums['A_pairs']) != kav}
        for name, s in sums.items():
            row[name] = {'status': s['status'] if s['decided'] else 'undecided', 'k_star': s.get('k_star'),
                         'window': s.get('window'), 'branch': s.get('branch'), 'W': s.get('W'),
                         'band_width': s.get('band_width'), 'range_over_tau': s.get('range_over_tau'),
                         't_sum_k_star': s.get('t_sum_k_star'), 'Q_k_star': s.get('Q_k_star'),
                         'n_floor_rejections': len(s.get('floor_rejections') or []),
                         'excluded_swings_at_kstar': (s.get('growth_test_at_kstar') or {}).get('excluded_swing_indices')}
        table.append(row)
        reports[rid] = {'record': rid, 'cell': rec['cell'], 'eval_dir': rec['eval_dir'],
                        'instance': W141.candidate_key(rec['eval_dir']), 'cycles_recorded': n, 'cap_rule_replayed': kw,
                        'committed_run': rec['run'], 'certifying_spec': rec.get('certifying'),
                        'non_clean_cycles': non_clean, 'all_clean_crosscheck': clean_x[rid],
                        'AB_equals_w141_V3_10': ab_eq_v3, 'arms': sums, 'inputs_sha256': rec['inputs_sha256'],
                        'sources': meta}
        _log(f"{rid}: clean x-check {clean_x[rid]['equal']} | " + ' | '.join(
            f"{nm} {row[nm]['status']}{' ' + str(row[nm]['k_star']) if row[nm]['k_star'] else ''}"
            f"{' ' + str(row[nm]['window']) if row[nm]['window'] else ''}" for nm in ('V0', 'A', 'AB', 'A_pairs', 'A_f0'))
            + f" | AB==W141 V3_10 {ab_eq_v3}")
    certs = [r for r in table if r['committed_certificate']]
    ab_changes_cert = [{'record': r['record'], 'V0': _key(r['V0']), 'A': _key(r['A']), 'AB': _key(r['AB'])}
                       for r in certs if r['AB_changes_vs_A'] or r['AB_changes_vs_V0']]
    a_changes_cert = [{'record': r['record'], 'V0': _key(r['V0']), 'A': _key(r['A'])}
                      for r in certs if r['A_changes_vs_V0']]
    non_cert_changes = [{'record': r['record'], 'V0': _key(r['V0']), 'A': _key(r['A']), 'AB': _key(r['AB'])}
                        for r in table if not r['committed_certificate']
                        and (r['A_changes_vs_V0'] or r['AB_changes_vs_V0'] or r['AB_changes_vs_A'])]
    carries = 'A_and_B' if not ab_changes_cert else 'A_only'
    decision = {
        'rule': ('Addendum 61: if (A)+(B) changes no committed certificate relative to (A) alone and V0, v6 carries both; '
                 'if it changes any, v6 carries (A) only, and the inconsistency is recorded for the cleanup'),
        'committed_certificate_definition': ('a record whose V0 replay (v5 as frozen, which reproduces the committed '
                                             'decision -- W141) is certified'),
        'change_definition': f'a different {", ".join(CHANGE_FIELDS)}',
        'n_committed_certificates': len(certs), 'committed_certificate_records': [r['record'] for r in certs],
        'AB_changes_a_committed_certificate': ab_changes_cert,
        'A_changes_a_committed_certificate_vs_V0_REPORTED': a_changes_cert,
        'changes_on_records_without_a_committed_certificate_REPORTED': non_cert_changes,
        'v6_carries': carries,
        'inconsistency_for_the_cleanup': (None if carries == 'A_and_B' else
                                          'the turning-point floor (B) changes a committed certificate: v6 carries (A) '
                                          'only; the turning-point count keeps registering sub-floor swings'),
    }
    d_row = next(r for r in table if r['record'] == D_CELL)
    d_detail = {name: {k: reports[D_CELL]['arms'][name].get(k) for k in (
        'status', 'k_star', 'branch', 'window', 'W', 'P_hat', 'band', 'band_width', 'range_over_tau', 't_sum_k_star',
        'Q_k_star', 'T', 'A', 'floor_rejections', 'growth_test_at_kstar', 'creep')} for name in ('V0', 'A', 'AB', 'A_pairs')}
    q_d = recs[D_CELL]['q']
    n_d = recs[D_CELL]['n']
    for name, e in d_detail.items():
        if e['status'] == 'certified':
            e['Q_last_recorded_minus_Q_k_star'] = q_d[n_d] - q_d[e['k_star']]
            e['over_tau'] = (q_d[n_d] - q_d[e['k_star']]) / TAU
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    git_head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    code = {rel: _sha(os.path.join(REPO, rel)) for rel in (
        os.path.basename(__file__), 'p515_s53_w141_swing_variants.py', 'settling_criterion.py',
        'settling_criterion_v2.py', 'settling_criterion_v4.py', 'settling_criterion_v5.py',
        'p515_s53_w139_v5_from_records.py', 'p515_s53_w139_resettle_v5_hooks.py',
        'p515_s53_w131_prefreeze_diagnostics.py', 'p515_s53_w137_recurring_acceptable.py')}
    doc = {'schema': 'p515_s53_w142_floor_replay_v1',
           'task': 'W142 item 1 (Addendum 61): (A) vs (A)+(B) replayed on every committed re-settle record',
           'utc': _utc(), 'git_head': git_head, 'code_sha256': code, 'definition': __doc__,
           'objective_convention': 'Q = gross_operational_cost (per_cycle_record.jsonl), EUR, settlement excluded',
           'constants': {'TAU': TAU, 'F_TAU_OVER_10': F, 'EPS0': SC5.EPS0, 'P_MAX': V5.P_MAX,
                         'GAP_BOUND': SC2.GAP_BOUND},
           'arms': ARMS, 'records': order, 'table': table, 'decision': decision,
           'self_tests': {'A_f0_equals_V0_and_AB_equals_w141_V3_10_every_record': not selftest_fail,
                          'failures': selftest_fail},
           'all_clean_crosscheck_all_equal': all(v['equal'] for v in clean_x.values()),
           'd_c52e1670_detail': {'record': D_CELL, 'cycles_recorded': n_d, 'Q_last_recorded': q_d[n_d],
                                 'arms': d_detail,
                                 'where_A_alone_certifies': {k: d_row['A'][k] for k in ('status', 'k_star', 'window')},
                                 'where_A_plus_B_certifies': {k: d_row['AB'][k] for k in ('status', 'k_star', 'window')}},
           'w141_input': {'path': W141_JSON, 'sha256': _sha(os.path.join(REPO, W141_JSON))},
           'reports': reports, 'guards': guards, 'pickle_guard': dict(W141.PICKLE_COUNTS), 'wall_s': time.time() - t0}
    jp = os.path.join(out_dir, OUT_JSON)
    with open(jp, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    man = {os.path.relpath(jp, REPO): _sha(jp), W141_JSON: doc['w141_input']['sha256']}
    for rep in reports.values():
        for rel, v in rep['inputs_sha256'].items():
            if isinstance(v, str):
                man[rel] = v
            elif isinstance(v, dict) and isinstance(v.get('sha256'), str):
                man[rel] = v['sha256']
    with open(os.path.join(out_dir, OUT_MAN), 'x') as handle:
        GRIO.dump(man, handle, indent=1, sort_keys=True)
    _log(f"DECISION: v6 carries {carries}; AB changes a committed certificate: {ab_changes_cert}; A changes a committed "
         f"certificate vs V0: {a_changes_cert}; changes on non-certificate records: {non_cert_changes}")
    for name in ('V0', 'A', 'AB', 'A_pairs'):
        e = d_detail[name]
        _log(f"D {name}: {e['status']} k* {e['k_star']} window {e['window']} W {e['W']} band {e['band_width']} range/tau "
             f"{e['range_over_tau']} t_sum {e['t_sum_k_star']} T {e['T']} A {e['A']} "
             f"Q198-Qk* {e.get('Q_last_recorded_minus_Q_k_star')}")
    _log(f"self-tests pass {not selftest_fail}; all_clean x-check {doc['all_clean_crosscheck_all_equal']}; guards "
         f"{ {k: v['verify_0_failures'] for k, v in guards.items()} }; pickle {W141.PICKLE_COUNTS}; wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = W141._PICKLE_ORIG
    ok = (all(not v['verify_0_failures'] for v in guards.values()) and not selftest_fail
          and doc['all_clean_crosscheck_all_equal'] and W141.PICKLE_COUNTS == {'load': 0, 'loads': 0})
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
