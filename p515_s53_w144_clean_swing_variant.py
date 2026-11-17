"""P5.15 Planner task W144 -- CLEAN-SWING CRITERION VARIANTS OF v6 REPLAYED FROM RECORDS, to inform an expert ruling.
ZERO SOLVES, NO MODEL LOADS.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end (every imported module's own zero-permit guard is verified at 0 as well); `pickle.load` / `pickle.loads` are blocked
for the whole run by W141's module (imported below; it installs the blocking counters at import) and verified at 0.
Only JSON / JSONL records and IPOPT text logs are read.

NOTHING COMMITTED IS MODIFIED. The harness approach is W141's (`p515_s53_w141_swing_variants`, d02efe69): the variants
are a subclass of the committed `settling_criterion_v6.SettlingRuleV6` (`CleanVariantRule`, below); settling_criterion*.py,
W139's per-block clean readers (`block_finals` + `classify_cycle`, i.e. `settling_criterion_v5.classify_block_exit`) and
W141's record inputs are imported, never edited.

CONTEXT (W143, f546dec3): d_36686489 ended uncertified at its cap 191 under v6 with 0 vetoes; non-clean TSO recovery
exits after k0 inject jumps into Q; v6's floored all-pairs growth test (A) reads every swing since k0 while the clean veto
covers only the last-W window.

THE VARIANTS (each a subclass of v6; every clause not named is v6's -- both floors, the window veto, the gap clause, the
monotone branch, N and the caps):
  C0     v6 as frozen: `settling_criterion_v6.replay` itself (the committed code path). CONTROL.
  C0_re  self-test: CleanVariantRule with every switch off must equal C0 on every record, field for field.
  C1     CLEAN-SWING EXCLUSION: a swing A[i] whose span [T[i].t, T[i+1].t] (closed) contains a non-clean cycle (v5/v6
         clean definition: all_clean_k False) is EXCLUDED FROM THE GROWTH COMPARISON, in addition to v6 (A)'s floor
         exclusion; the remaining swings, in order, must be non-increasing over every consecutive pair of that sequence
         (v6 (A)'s chain reading). The turning points are still registered; P_hat and W still use them.
  C2     C1 PLUS CLEAN TURNING POINTS: a sign change whose candidate turning point (v6 (B)'s candidate: the more extreme of
         the carried point and version 1's extremum) lies at a non-clean cycle registers NO turning point -- the
         candidate is rejected exactly as (B) rejects a sub-floor swing: T[-1] (with the swing it closed and the sign
         change that registered it) is un-registered and CARRIED as the running extreme of the resumed leg; the
         rejected reversal is no sign change for the monotone branch (j_change reverts). Worker reading of the edge case
         (recorded per occurrence): when T is empty there is no T[-1] to un-register -- the candidate simply does not
         register, and any carried point is dropped (its kind no longer matches the resumed leg).
  C3     REPORT-ONLY REFERENCE: the growth test over the LAST THREE SWINGS ONLY (A[-3:], with v6 (A)'s floor applied
         within them; chain reading; fewer than three swings: those that exist). The expert already rejected the
         last-pair test as weaker.

THE RECORDS (21): the three committed v6 runs of W143 (d_4a82a64a 354b362a, d_3632b0ae 48625326, d_36686489 f546dec3)
and W141's 18 (the 16 of W139 item 5 -- the three v4 runs, cell 1's v3 run, the ten W118 r2 cells incl. the F2 pair, the
two W101 references x0 / unit_n7_4h_e1 -- and the two W140 v5 runs b_4649234b@v5 4b3be392, d_c52e1670@v5 51280961).
Per record: Q_k = gross_operational_cost (per_cycle_record.jsonl), boyd_k = boyd_all_pass AND local_solves_ok, t_sum_k
from the cell's cycle file, the cap rule of the record's own declaration (for the v6 runs
`p515_s53_w142_resettle_v6_hooks.declaration_for(cell)['cap_rule']`), over the RECORDED cycles 1..n only.
all_clean_k recomputed from the logs with W139's readers and cross-checked against (a) W141's committed non_clean_cycles
for its 18 records and (b) the in-run all_clean_k of the three v6 runs; for the v6 runs the per-block non-clean sets are
also cross-checked against the in-run non_clean_blocks.

C0 CONTROL CHECKS: (a) on the three v6 runs C0 must equal the committed in-run resettle_decision.json (status, k*, k_cap,
window, branch, reasons, T, A, band_width, n_vetoes); (b) on W141's 18 records C0 must equal the committed v6 replay of
W142 (w142_v6_from_records.json item1_v6_on_every_record, every field it recorded) -- which itself reproduced W142's
(A)+(B) arm, which reproduced W141's V3_10, whose V0 reproduced every committed v5 decision; (c) beside, REPORTED: C0
against each record's OWN committed run decision (v1 / v2 / v3 / v4 / v5 / v6; status and k*), the differences being
criterion changes already recorded (W139 / W142).

PER VARIANT AND RECORD: decision (status), k*, branch, window, band, band width, range / TAU, t_sum(k*), Q(k*); versus C0:
same status / k* / window / branch; whether a COMMITTED CERTIFICATE changes (a record whose own committed run is
certified, or the committed d_c52e1670 v6-from-records certificate; pb_y2025_n5's certificate is EXCLUDED, Addendum
58/59, and is reported but flagged); the POST-CERTIFICATE MOVEMENT for every record with cycles beyond the variant's k*:
Q(n) - Q(k*) and max |Q(c) - Q(k*)| over k* < c <= n, in EUR and in TAU, against 0.9 TAU.
D_36686489 DETAIL: per variant where it certifies, the window, range / TAU, t_sum, Q(191) - Q(k*) against 0.9 TAU, the
growth-test detail at k* (swings excluded by the floor / by the clean-span rule, pairs compared), the C2 rejections.
FREQUENCY TABLE: per record, the non-clean cycles strictly after its first residual pass N (= the C0 rule's N), against
the cell's storage (candidate_canonical: per node [P MVA, E MWh]); per block, the cycles after N whose final accepted
attempt is not the primary (the TSO / DSO / ESSO blocks that recover) and how many of those are non-clean.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w144_clean_swing_variant/):
  w144_clean_swing_variant.json, w144_log_inventory_new.json, manifest_sha256.json (launch.log beside, from the shell)
Launch (attached, alone, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w144_clean_swing_variant && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w144_clean_swing_variant.py \\
        > data/SRP1/Results/P515S53/w144_clean_swing_variant/launch.log 2>&1
"""
import collections
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W144 clean-swing variants (never solves)').install()

import gate_result_io as GRIO  # noqa: E402
import settling_criterion as SC1  # noqa: E402
import settling_criterion_v2 as SC2  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402
import p515_s53_w141_swing_variants as W141  # noqa: E402 -- W141's harness (arms its guards; blocks pickle loads)
import p515_s53_w142_resettle_v6_hooks as V6  # noqa: E402

W139 = W141.W139


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w144_clean_swing_variant', GUARD),) + tuple(W141.GUARDS))

TAU = SC6.TAU
MOVEMENT_BOUND_TAU = 0.9
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR_REL = os.path.join(S53, 'w144_clean_swing_variant')
OUT_JSON = 'w144_clean_swing_variant.json'
OUT_INV = 'w144_log_inventory_new.json'
OUT_MAN = 'manifest_sha256.json'
W141_DIR = os.path.join(S53, 'w141_swing_variants')
W141_JSON = os.path.join(W141_DIR, 'w141_swing_variants.json')
W141_MAN = os.path.join(W141_DIR, 'manifest_sha256.json')
W141_INV = os.path.join(W141_DIR, 'w141_log_inventory_new.json')
W142_V6 = {'path': os.path.join(S53, 'w142_resettle_v6', 'v6_from_records', 'w142_v6_from_records.json'),
           'manifest': os.path.join(S53, 'w142_resettle_v6', 'v6_from_records', 'manifest_sha256.json')}
V6_ROOT = os.path.join(S53, 'w142_resettle_v6')
V6_STAGE_SPEC = {'path': os.path.join(V6_ROOT, 'frozen_s53_resettle_spec_v6_96c23404.json'), 'sha8': '96c23404'}
V6_RUNS = {
    'd_4a82a64a@v6': {'cell': 'd_4a82a64a', 'commit': '354b362a',
                      'eval_dir': os.path.join(V6_ROOT, 'campaign_s53_w142_resettle_v6_d_4a82a64a', 'evals',
                                               '14b00a04ffbcfd33_d_4a82a64a')},
    'd_3632b0ae@v6': {'cell': 'd_3632b0ae', 'commit': '48625326',
                      'eval_dir': os.path.join(V6_ROOT, 'campaign_s53_w142_resettle_v6_d_3632b0ae', 'evals',
                                               '1fe57699ce791ef6_d_3632b0ae')},
    'd_36686489@v6': {'cell': 'd_36686489', 'commit': 'f546dec3',
                      'eval_dir': os.path.join(V6_ROOT, 'campaign_s53_w142_resettle_v6_d_36686489', 'evals',
                                               '4de61708589e6e26_d_36686489')},
}
D_FOCUS = 'd_36686489@v6'
D_FROM_RECORDS = {'record': 'd_c52e1670@v5', 'k_star': 150, 'label': 'v6 from records (W142, b64a4233)'}
EXCLUDED_CERTIFICATES = {'pb_y2025_n5': 'Addendum 58 Ruling 2 / Addendum 59: certificate EXCLUDED (Acceptable recovery '
                                        'at k* 167 in its window)'}

VARIANTS = {
    'C0_re': {'clean_swings': False, 'clean_tps': False, 'last_n': None, 'role': 'self-test: must equal C0'},
    'C1': {'clean_swings': True, 'clean_tps': False, 'last_n': None,
           'role': 'clean-swing exclusion: swings whose closed span contains a non-clean cycle leave the growth test'},
    'C2': {'clean_swings': True, 'clean_tps': True, 'last_n': None,
           'role': 'C1 plus clean turning points: a candidate at a non-clean cycle is rejected as (B) rejects'},
    'C3': {'clean_swings': False, 'clean_tps': False, 'last_n': 3,
           'role': 'REPORT-ONLY reference: growth test over the last three swings only (v6 floor applied within them)'},
}
CHANGE_FIELDS = ('status', 'k_star', 'window', 'branch')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(os.path.join(REPO, path), 'rb') as handle:
        for b in iter(lambda: handle.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _jsonl(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pinned(path, manifest):
    man = json.load(open(os.path.join(REPO, manifest)))
    sha = _sha(path)
    if man.get(path) != sha:
        raise RuntimeError(f'{path}: sha256 {sha} != its manifest {man.get(path)}')
    return json.load(open(os.path.join(REPO, path))), sha


# ======================================================================================================================
#  the variant rule (a subclass of the committed v6 rule)
# ======================================================================================================================
class CleanFloorSignState(SC6.FloorSignState):
    """v6 (B)'s sign state; with `is_clean` given (C2), a candidate turning point at a non-clean cycle is rejected
    exactly as (B) rejects a sub-floor swing. `is_clean` None reproduces SC6.FloorSignState (C0_re self-test)."""

    def __init__(self, floor, is_clean=None):
        super().__init__(floor)
        self.is_clean = is_clean

    def update(self, q, k, s):
        if self.is_clean is None:
            return SC6.FloorSignState.update(self, q, k, s)
        if s == 0:
            return False, None
        if self.sign_prev is None:
            self.sign_prev, self.j_prev = s, k
            return False, None
        if s == self.sign_prev:
            self.j_prev = k
            return False, None
        kind = 'max' if self.sign_prev == 1 else 'min'
        t = SC1._extremum(q, self.j_prev, k - 1, kind)
        if self.carry is not None:
            if self.carry[1] != kind:
                raise RuntimeError(f'CleanFloorSignState: carried {self.carry} of the wrong kind at {k} ({kind})')
            ct = self.carry[0]
            if (kind == 'max' and q[ct] >= q[t]) or (kind == 'min' and q[ct] <= q[t]):
                t = ct
        tp = (t, kind, q[t])
        non_clean = not self.is_clean(t)
        swing = abs(q[t] - q[self.T[-1][0]]) if self.T else None
        sub_floor = swing is not None and swing < self.floor
        if non_clean or sub_floor:
            if self.T:
                popped = self.T.pop()
                if self.T:
                    self.A.pop()
                self.j_change = self.jc_stack.pop()
                dropped = None
                self.carry = popped
            else:                                   # the edge case (module docstring): nothing to un-register
                popped = None
                dropped = list(self.carry) if self.carry is not None else None
                self.carry = None
            self.rejections.append({'cycle': k, 'candidate': list(tp), 'closed_swing': swing,
                                    'popped': list(popped) if popped is not None else None,
                                    'cause': ('non_clean_candidate' if non_clean else 'sub_floor_swing')
                                    + ('+sub_floor_swing' if (non_clean and sub_floor) else ''),
                                    'edge_case_T_empty': popped is None, 'dropped_carry': dropped})
            self.sign_prev, self.j_prev = s, k
            return False, None
        self.jc_stack.append(self.j_change)
        self.T.append(tp)
        if len(self.T) >= 2:
            self.A.append(abs(q[t] - q[self.T[-2][0]]))
        self.carry = None
        self.sign_prev, self.j_prev, self.j_change = s, k, k
        return True, tp


class CleanVariantRule(SC6.SettlingRuleV6):
    """v6 with the growth test optionally (C1) excluding swings whose span holds a non-clean cycle, optionally (C3)
    restricted to the last `last_n` swings, and the sign state optionally (C2) rejecting non-clean candidates.
    evaluate() is SettlingRuleV6.evaluate reproduced (version 2's clauses called, the growth test swapped, version 5's
    veto block as v6 reproduces it): SettlingRuleV6.evaluate calls the module-level growth_test and so cannot be
    intercepted. The self-test C0_re proves the reproduction equal to v6 on every record."""

    def __init__(self, p_max, clean_swings=False, clean_tps=False, last_n=None, **kw):
        self.clean_swings = bool(clean_swings)
        self.clean_tps = bool(clean_tps)
        self.last_n = last_n
        super().__init__(p_max, **kw)

    def _is_clean(self, c):
        return bool(getattr(self, 'all_clean', {}).get(c, False))

    @property
    def sign(self):
        return self._sign

    @sign.setter
    def sign(self, value):
        if not (isinstance(value, SC1._SignState) and not value.T and value.sign_prev is None):
            raise RuntimeError('CleanVariantRule: the sign state must only be assigned fresh')
        if isinstance(self._sign, SC6.FloorSignState):
            self.floor_rejections.extend(r for r in self._sign.rejections if r not in self.floor_rejections)
        self._sign = CleanFloorSignState(self.TP_FLOOR, self._is_clean if self.clean_tps else None)

    def growth(self, A, T):
        if T and len(A) != len(T) - 1:
            raise RuntimeError(f'CleanVariantRule: |A| {len(A)} != |T| - 1 {len(T) - 1}')
        idx = list(range(len(A)))
        if self.last_n is not None:
            idx = idx[-self.last_n:]
        excl_floor = [i for i in idx if A[i] < self.GROWTH_FLOOR]
        excl_clean = []
        if self.clean_swings:
            nc = self.non_clean_cycles
            for i in idx:
                lo, hi = T[i][0], T[i + 1][0]
                hit = [c for c in nc if lo <= c <= hi]
                if hit:
                    excl_clean.append(i)
        kept = [i for i in idx if i not in excl_floor and i not in excl_clean]
        pairs = [[kept[j], kept[j + 1]] for j in range(len(kept) - 1)]
        ok = all(A[j2] <= A[j1] for j1, j2 in pairs)
        return ok, {'considered_swing_indices': idx, 'excluded_swing_indices': excl_floor,
                    'excluded_non_clean_span_indices': excl_clean, 'pairs_compared': pairs}

    def evaluate(self, k):
        _ca, _cb, a_parts, b_parts, _r = SC2.SettlingRuleV2.evaluate(self, k)
        ok, det = self.growth(self.sign.A, self.sign.T)
        a_parts.update({'swings_non_increasing_floored': ok, 'growth_floor': self.GROWTH_FLOOR,
                        'turning_point_floor': self.TP_FLOOR,
                        'swings_excluded_below_floor': det['excluded_swing_indices'],
                        'swing_pairs_compared': det['pairs_compared'],
                        'swings_non_increasing_all_pairs_note': ('version 2\'s unfloored all-pairs value, REPORT-ONLY; '
                                                                 'version 6 decides on swings_non_increasing_floored')})
        if self.clean_swings or self.clean_tps or self.last_n is not None:
            a_parts.update({'w144_variant': {'clean_swings': self.clean_swings, 'clean_tps': self.clean_tps,
                                             'last_n': self.last_n},
                            'swings_excluded_non_clean_span': det['excluded_non_clean_span_indices'],
                            'swings_considered': det['considered_swing_indices']})
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
            reasons = list(reasons) + [SC6.VETO_REASON]
            self._cycle_veto = {'cycle': k, 'branches': vetoed, 'k0': self.k0}
            self.vetoes.append(dict(self._cycle_veto))
        return cert_a, cert_b, a_parts, b_parts, reasons


def run_rule(cls_or_none, q, b, t, clean, n, kw, **vkw):
    """settling_criterion_v6.replay's loop (C0: the committed function itself)."""
    if cls_or_none is None:
        _o, dec, rule = SC6.replay(q, b, t, clean, V6.P_MAX, last=n, **kw)
        return dec, rule
    rule = cls_or_none(V6.P_MAX, **vkw, **kw)
    k = 0
    while True:
        k += 1
        if k > n or k > rule.effective_cap():
            break
        rule.observe(k, q.get(k), bool(b.get(k, False)), t.get(k), bool(clean.get(k, False)))
        if rule.decision is not None:
            break
    return rule.decision, rule


# ======================================================================================================================
#  the records
# ======================================================================================================================
def v6_run_inputs():
    out = {}
    for rid, v in V6_RUNS.items():
        ev = v['eval_dir']
        camp = ev.split(os.sep + 'evals' + os.sep)[0]
        man = json.load(open(os.path.join(REPO, camp, 'campaign_manifest_sha256.json')))
        inputs = {}
        for f in ('per_cycle_record.jsonl', V6.CYCLE_FILE, V6.DECISION_FILE, 'network_ipopt_solve_records.jsonl',
                  'evaluation_record.json', 'network_failures_s39_D.jsonl', 'esso_recovery_events_s39_D.jsonl'):
            rel = os.path.join(ev, f)
            now = _sha(rel)
            if man.get(rel) != now:
                raise RuntimeError(f'{rel}: sha256 {now} != campaign manifest {man.get(rel)}')
            inputs[rel] = now
        rows = _jsonl(os.path.join(ev, 'per_cycle_record.jsonl'))
        lines = {x['cycle']: x for x in _jsonl(os.path.join(ev, V6.CYCLE_FILE))}
        dec = json.load(open(os.path.join(REPO, ev, V6.DECISION_FILE)))
        q = {r['cycle']: r['gross_operational_cost'] for r in rows}
        b = {r['cycle']: bool(r['boyd_all_pass'] and r['local_solves_ok']) for r in rows}
        t = {k: (lines.get(k) or {}).get('t_sum') for k in q}
        cr = V6.declaration_for(v['cell'])['cap_rule']
        kw = ({'cap': cr['cap'], 'cap_ceiling': cr['ceiling']} if cr['kind'] == 'fixed' else
              {'cap_after_first_k0': cr['after_first_k0'], 'cap_ceiling': cr['ceiling']})
        run = {'spec': V6_STAGE_SPEC, 'criterion': 'v6 (Addendum 61)', 'commit': v['commit'],
               'status': dec.get('status'), 'k_star': dec.get('k_star'), 'k_cap': dec.get('k_cap')}
        out[rid] = {'cell': v['cell'], 'eval_dir': ev, 'q': q, 'b': b, 't': t, 'n': len(rows), 'kw': kw, 'run': run,
                    'certifying': ({'spec': V6_STAGE_SPEC, 'criterion': 'v6', 'k_star': dec.get('k_star')}
                                   if dec.get('status') == 'certified' else None),
                    'inputs_sha256': inputs, 'in_run_all_clean': {k: lines[k]['all_clean_k'] for k in q},
                    'in_run_non_clean_blocks': {k: sorted(lines[k].get('non_clean_blocks') or []) for k in q},
                    'in_run_decision': dec}
    return out


def instance(eval_dir):
    p = os.path.join(REPO, eval_dir, 'evaluation_record.json')
    if not os.path.exists(p):
        return None
    ev = json.load(open(p))
    cc = ev.get('candidate_canonical')
    nodes = (cc or {}).get('nodes') or {}
    return {'candidate_key': ev.get('candidate_key'), 'candidate_label': ev.get('candidate_label'),
            'candidate_canonical': cc, 'evaluation_record_sha256': _sha(os.path.relpath(p, REPO)),
            'E_MWh_total': sum(v[1] for v in nodes.values()) if nodes else None,
            'P_MVA_total': sum(v[0] for v in nodes.values()) if nodes else None,
            'storage_nodes': {nd: v for nd, v in nodes.items() if v[0] or v[1]}}


# ======================================================================================================================
#  summaries
# ======================================================================================================================
def summarize(dec, rule, q, n):
    s = W141.summarize(dec, rule, q, n)
    s['floor_rejections_all_since_cycle_1'] = rule._all_rejections()
    s['N_first_k0'] = rule.n
    if dec is not None and dec.get('status') == 'certified':
        k = dec['k_star']
        pa = dec['certA_parts'] or {}
        s['growth_test_at_kstar'] = {x: pa.get(x) for x in (
            'swings_non_increasing_floored', 'swings_excluded_below_floor', 'swings_excluded_non_clean_span',
            'swings_considered', 'swing_pairs_compared', 'w144_variant')}
        after = [c for c in sorted(q) if k < c <= n]
        if after:
            mx = max(after, key=lambda c: abs(q[c] - q[k]))
            s['post_certificate_movement'] = {
                'cycles_after': [after[0], after[-1]], 'n_cycles_after': len(after),
                'Q_last_minus_Q_kstar': q[n] - q[k], 'over_tau': (q[n] - q[k]) / TAU,
                'max_abs_Q_c_minus_Q_kstar': abs(q[mx] - q[k]), 'at_cycle': mx,
                'max_abs_over_tau': abs(q[mx] - q[k]) / TAU,
                'last_exceeds_0_9_tau': abs(q[n] - q[k]) > MOVEMENT_BOUND_TAU * TAU,
                'max_exceeds_0_9_tau': abs(q[mx] - q[k]) > MOVEMENT_BOUND_TAU * TAU}
        else:
            s['post_certificate_movement'] = None
    return s


def _key(s):
    return {f: s.get(f) for f in CHANGE_FIELDS}


def solver_frequency(per_cycle, n, N, non_clean):
    """Non-clean cycles strictly after N and the per-block recoveries (final attempt not primary) after N."""
    after = [c for c in range(N + 1, n + 1)] if N else []
    nc_after = [c for c in non_clean if N and c > N]
    rec_by_block = collections.defaultdict(lambda: {'cycles': [], 'non_clean_cycles': []})
    nc_blocks = {}
    for c in after:
        for key, e in per_cycle[c].items():
            if e['attempt'] != 'primary':
                rec_by_block[key]['cycles'].append(c)
                if not e['clean']:
                    rec_by_block[key]['non_clean_cycles'].append(c)
        bad = sorted((key, e['class'], e['attempt'], e['reason']) for key, e in per_cycle[c].items() if not e['clean'])
        if bad:
            nc_blocks[c] = [list(x) for x in bad]
    return {'N_first_k0': N, 'cycles_after_N': len(after), 'non_clean_cycles_after_N': nc_after,
            'n_non_clean_after_N': len(nc_after),
            'non_clean_rate_after_N': (len(nc_after) / len(after)) if after else None,
            'N_itself_non_clean': (N in non_clean) if N else None,
            'non_clean_blocks_by_cycle_after_N': nc_blocks,
            'recovering_blocks_after_N': {k: dict(v) for k, v in sorted(rec_by_block.items())}}


# ======================================================================================================================
def main():
    t0 = time.time()
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_JSON, OUT_INV, OUT_MAN):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    w141_doc, w141_sha = _pinned(W141_JSON, W141_MAN)
    w141_inv_new, w141_inv_sha = _pinned(W141_INV, W141_MAN)
    w139_inv = json.load(open(os.path.join(REPO, W141.W139_INV)))
    w142_doc, w142_sha = _pinned(W142_V6['path'], W142_V6['manifest'])
    with contextlib.redirect_stdout(io.StringIO()):
        recs = W139.record_inputs()
    recs.update(W141.v5_run_inputs())
    if sorted(recs) != sorted(w141_doc['records']):
        raise RuntimeError(f'W141 records mismatch: {sorted(recs)}')
    recs.update(v6_run_inputs())
    order = list(V6_RUNS) + list(w141_doc['records'])
    if len(order) != 21 or len(set(order)) != 21:
        raise RuntimeError(f'expected 21 records, got {order}')
    reports, table, selftest_fail, clean_x, c0_checks = {}, [], [], {}, {}
    for rid in order:
        rec = recs[rid]
        finals, meta = W139.block_finals(rec['eval_dir'], rec['n'])
        if not meta['coverage_51_every_cycle']:
            raise RuntimeError(f'{rid}: block coverage is not 51 on every cycle')
        per_cycle = {k: W139.classify_cycle(finals[k]) for k in range(1, rec['n'] + 1)}
        clean = {k: all(e['clean'] for e in per_cycle[k].values()) for k in per_cycle}
        non_clean = sorted(k for k, v in clean.items() if not v)
        if rid in V6_RUNS:
            ref = sorted(k for k, v in rec['in_run_all_clean'].items() if not v)
            blocks_now = {k: sorted(key for key, e in per_cycle[k].items() if not e['clean']) for k in per_cycle}
            blocks_eq = all(blocks_now[k] == rec['in_run_non_clean_blocks'][k] for k in per_cycle)
            clean_x[rid] = {'against': 'in-run all_clean_k and non_clean_blocks (resettle_cycle_record.jsonl)',
                            'equal': ref == non_clean and blocks_eq, 'cycles_equal': ref == non_clean,
                            'blocks_equal_every_cycle': blocks_eq, 'non_clean_cycles': non_clean}
        else:
            ref = w141_doc['reports'][rid]['non_clean_cycles']
            clean_x[rid] = {'against': 'W141 committed non_clean_cycles', 'equal': ref == non_clean,
                            'non_clean_cycles': non_clean}
        q, b, t, n, kw = rec['q'], rec['b'], rec['t'], rec['n'], rec['kw']
        d0, r0 = run_rule(None, q, b, t, clean, n, kw)
        sums = {'C0': summarize(d0, r0, q, n)}
        for name, v in VARIANTS.items():
            dv, rv = run_rule(CleanVariantRule, q, b, t, clean, n, kw, clean_swings=v['clean_swings'],
                              clean_tps=v['clean_tps'], last_n=v['last_n'])
            sums[name] = summarize(dv, rv, q, n)
        if W141._core(sums['C0_re']) != W141._core(sums['C0']):
            selftest_fail.append({'record': rid, 'self_test': 'C0_re == C0', 'c0': W141._core(sums['C0']),
                                  'got': W141._core(sums['C0_re'])})
        # ---- C0 control checks ----
        if rid in V6_RUNS:
            dec = rec['in_run_decision']
            fields = ('status', 'k_star', 'k_cap', 'window', 'branch', 'reasons', 'T', 'A', 'band_width', 'n_vetoes')
            exp = {k: dec.get(k) for k in fields}
            got = {k: sums['C0'].get(k) for k in fields}
            c0_checks[rid] = {'against': 'the committed in-run resettle_decision.json', 'fields': list(fields),
                              'expected': exp, 'got': got,
                              'equal': json.dumps(exp, sort_keys=True, default=str) == json.dumps(got, sort_keys=True,
                                                                                                 default=str)}
        else:
            w = w142_doc['item1_v6_on_every_record'][rid]['v6']
            exp = dict(w)
            got = {k: sums['C0'].get(k) for k in w}
            c0_checks[rid] = {'against': 'the committed W142 v6 replay (w142_v6_from_records.json item1)',
                              'fields': sorted(w), 'expected': exp, 'got': got,
                              'equal': json.dumps(exp, sort_keys=True, default=str) == json.dumps(got, sort_keys=True,
                                                                                                 default=str)}
        own = rec['run']
        c0_checks[rid]['own_committed_run_REPORTED'] = {
            'criterion': own.get('criterion'), 'commit': own.get('commit'),
            'committed': {'status': own.get('status'), 'k_star': own.get('k_star')},
            'C0': {'status': sums['C0']['status'], 'k_star': sums['C0'].get('k_star')},
            'same_status_and_k_star': (own.get('status') == sums['C0']['status']
                                       and own.get('k_star') == sums['C0'].get('k_star'))}
        # ---- the committed certificate of this record ----
        if rid == D_FROM_RECORDS['record']:
            committed_cert = {'k_star': D_FROM_RECORDS['k_star'], 'source': D_FROM_RECORDS['label'], 'excluded': None}
        elif own.get('status') == 'certified':
            committed_cert = {'k_star': own.get('k_star'), 'source': f"{own.get('criterion')} run "
                              f"{own.get('commit') or own.get('campaign_spec')}",
                              'excluded': EXCLUDED_CERTIFICATES.get(rid)}
        else:
            committed_cert = None
        changes = {}
        for name in ('C1', 'C2', 'C3'):
            s = sums[name]
            ch = {'differs_from_C0': _key(s) != _key(sums['C0']), 'C0': _key(sums['C0']), 'variant': _key(s)}
            if committed_cert is not None:
                ch['committed_certificate_k_star'] = committed_cert['k_star']
                ch['committed_certificate_changes'] = not (s['status'] == 'certified'
                                                           and s.get('k_star') == committed_cert['k_star'])
            changes[name] = ch
        freq = solver_frequency(per_cycle, n, r0.n, non_clean)
        inst = instance(rec['eval_dir'])
        rep = {'record': rid, 'cell': rec['cell'], 'eval_dir': rec['eval_dir'], 'instance': inst,
               'cycles_recorded': n, 'cap_rule_replayed': kw, 'committed_run': own,
               'committed_certificate': committed_cert, 'non_clean_cycles': non_clean,
               'all_clean_crosscheck': clean_x[rid], 'c0_control': c0_checks[rid], 'variants': sums,
               'changes_vs_C0_and_committed': changes, 'solver_frequency': freq,
               'inputs_sha256': rec['inputs_sha256'], 'sources': meta}
        reports[rid] = rep
        row = {'record': rid, 'cell': rec['cell'], 'E_MWh_total': (inst or {}).get('E_MWh_total'),
               'P_MVA_total': (inst or {}).get('P_MVA_total'), 'committed_certificate': committed_cert,
               'c0_control_equal': c0_checks[rid]['equal']}
        for name, s in sums.items():
            pcm = s.get('post_certificate_movement') or {}
            row[name] = {'status': s['status'] if s['decided'] else 'undecided', 'k_star': s.get('k_star'),
                         'window': s.get('window'), 'branch': s.get('branch'), 'band': s.get('band'),
                         'band_width': s.get('band_width'), 'range_over_tau': s.get('range_over_tau'),
                         't_sum_k_star': s.get('t_sum_k_star'), 'Q_k_star': s.get('Q_k_star'),
                         'post_cert_Q_last_minus_Q_kstar': pcm.get('Q_last_minus_Q_kstar'),
                         'post_cert_over_tau': pcm.get('over_tau'), 'post_cert_max_abs_over_tau': pcm.get('max_abs_over_tau'),
                         'post_cert_exceeds_0_9_tau': (pcm.get('last_exceeds_0_9_tau') or pcm.get('max_exceeds_0_9_tau'))
                         if pcm else None}
        table.append(row)
        _log(f"{rid}: clean x-check {clean_x[rid]['equal']} (non-clean after N={r0.n}: {freq['non_clean_cycles_after_N']}); "
             f"C0 control {c0_checks[rid]['equal']} | " + ' | '.join(
                 f"{nm} {row[nm]['status']}{' ' + str(row[nm]['k_star']) if row[nm]['k_star'] else ''}"
                 f"{' ' + str(row[nm]['window']) if row[nm]['window'] else ''}"
                 f"{' post ' + format(row[nm]['post_cert_over_tau'], '.3f') + ' tau' if row[nm]['post_cert_over_tau'] is not None else ''}"
                 for nm in ('C0', 'C0_re', 'C1', 'C2', 'C3')))
    # ---- aggregates ----
    changes_vs_c0 = {}
    committed_cert_changes = {}
    for name in ('C1', 'C2', 'C3'):
        changes_vs_c0[name] = [{'record': rid, 'C0': reports[rid]['changes_vs_C0_and_committed'][name]['C0'],
                                'variant': reports[rid]['changes_vs_C0_and_committed'][name]['variant']}
                               for rid in order if reports[rid]['changes_vs_C0_and_committed'][name]['differs_from_C0']]
        committed_cert_changes[name] = [
            {'record': rid, 'committed_k_star': reports[rid]['committed_certificate']['k_star'],
             'excluded_certificate': reports[rid]['committed_certificate']['excluded'],
             'variant': reports[rid]['changes_vs_C0_and_committed'][name]['variant']}
            for rid in order if reports[rid]['committed_certificate'] is not None
            and reports[rid]['changes_vs_C0_and_committed'][name]['committed_certificate_changes']]
    committed_cert_changes['C0'] = [
        {'record': rid, 'committed_k_star': reports[rid]['committed_certificate']['k_star'],
         'excluded_certificate': reports[rid]['committed_certificate']['excluded'],
         'C0': _key(reports[rid]['variants']['C0'])}
        for rid in order if reports[rid]['committed_certificate'] is not None
        and not (reports[rid]['variants']['C0']['status'] == 'certified'
                 and reports[rid]['variants']['C0'].get('k_star') == reports[rid]['committed_certificate']['k_star'])]
    post_cert = {name: [{'record': r['record'], 'k_star': r[name]['k_star'],
                         'Q_last_minus_Q_kstar': r[name]['post_cert_Q_last_minus_Q_kstar'],
                         'over_tau': r[name]['post_cert_over_tau'], 'max_abs_over_tau': r[name]['post_cert_max_abs_over_tau'],
                         'exceeds_0_9_tau': r[name]['post_cert_exceeds_0_9_tau']}
                        for r in table if r[name]['post_cert_over_tau'] is not None] for name in ('C0', 'C1', 'C2', 'C3')}
    freq_table = [{'record': rid, 'cell': reports[rid]['cell'], 'E_MWh_total': (reports[rid]['instance'] or {}).get('E_MWh_total'),
                   'P_MVA_total': (reports[rid]['instance'] or {}).get('P_MVA_total'),
                   'storage_nodes': (reports[rid]['instance'] or {}).get('storage_nodes'),
                   'N_first_k0': reports[rid]['solver_frequency']['N_first_k0'],
                   'cycles_recorded': reports[rid]['cycles_recorded'],
                   'cycles_after_N': reports[rid]['solver_frequency']['cycles_after_N'],
                   'n_non_clean_after_N': reports[rid]['solver_frequency']['n_non_clean_after_N'],
                   'non_clean_cycles_after_N': reports[rid]['solver_frequency']['non_clean_cycles_after_N'],
                   'non_clean_rate_after_N': reports[rid]['solver_frequency']['non_clean_rate_after_N'],
                   'non_clean_blocks_after_N': sorted({b_[0] for v in reports[rid]['solver_frequency']
                                                       ['non_clean_blocks_by_cycle_after_N'].values() for b_ in v})}
                  for rid in order]
    by_block = collections.defaultdict(lambda: {'records': [], 'recovery_cycles_after_N': 0, 'non_clean_after_N': 0})
    for rid in order:
        for blk, v in reports[rid]['solver_frequency']['recovering_blocks_after_N'].items():
            by_block[blk]['records'].append([rid, v['cycles'], v['non_clean_cycles']])
            by_block[blk]['recovery_cycles_after_N'] += len(v['cycles'])
            by_block[blk]['non_clean_after_N'] += len(v['non_clean_cycles'])
    by_block = {k: dict(v) for k, v in sorted(by_block.items(), key=lambda x: -x[1]['recovery_cycles_after_N'])}
    by_E = collections.defaultdict(lambda: {'records': [], 'n_non_clean_after_N': 0, 'cycles_after_N': 0})
    for r in freq_table:
        e = by_E[str(r['E_MWh_total'])]
        e['records'].append(r['record'])
        e['n_non_clean_after_N'] += r['n_non_clean_after_N']
        e['cycles_after_N'] += r['cycles_after_N']
    for e in by_E.values():
        e['rate'] = (e['n_non_clean_after_N'] / e['cycles_after_N']) if e['cycles_after_N'] else None
    # ---- d_36686489 detail ----
    dq = recs[D_FOCUS]['q']
    dn = recs[D_FOCUS]['n']
    d_detail = {'record': D_FOCUS, 'cycles_recorded': dn, 'Q_at_last_recorded': dq[dn],
                'non_clean_cycles': reports[D_FOCUS]['non_clean_cycles'],
                'solver_frequency': reports[D_FOCUS]['solver_frequency'], 'per_variant': {}}
    for name in ('C0', 'C1', 'C2', 'C3'):
        s = reports[D_FOCUS]['variants'][name]
        e = {k: s.get(k) for k in ('status', 'k_star', 'k_cap', 'branch', 'window', 'W', 'P_hat', 'band', 'band_width',
                                   'range_over_tau', 't_sum_k_star', 'Q_k_star', 'reasons', 'T', 'A', 'n_vetoes',
                                   'floor_rejections', 'floor_rejections_all_since_cycle_1', 'growth_test_at_kstar', 'creep', 'post_certificate_movement')}
        if s['status'] == 'certified':
            k = s['k_star']
            e['Q_cap_minus_Q_kstar'] = dq[dn] - dq[k]
            e['Q_cap_minus_Q_kstar_over_tau'] = (dq[dn] - dq[k]) / TAU
            e['within_0_9_tau'] = abs(dq[dn] - dq[k]) <= MOVEMENT_BOUND_TAU * TAU
        d_detail['per_variant'][name] = e
    # ---- log inventory: entries neither W139 nor W141 inventoried; those they did, verified unchanged ----
    prior = dict(w139_inv)
    prior.update(w141_inv_new)
    new_inv, same, differ = {}, 0, []
    for rel, v in W139.W131.INVENTORY.items():
        old = prior.get(rel)
        if old is None:
            new_inv[rel] = v
        elif old.get('sha256') == v.get('sha256'):
            same += 1
        else:
            differ.append(rel)
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    git_head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    code = {rel: _sha(rel) for rel in (
        os.path.basename(__file__), 'p515_s53_w141_swing_variants.py', 'settling_criterion.py',
        'settling_criterion_v2.py', 'settling_criterion_v5.py', 'settling_criterion_v6.py',
        'p515_s53_w139_v5_from_records.py', 'p515_s53_w139_resettle_v5_hooks.py', 'p515_s53_w142_resettle_v6_hooks.py',
        'p515_s53_w131_prefreeze_diagnostics.py', 'p515_s53_w137_recurring_acceptable.py')}
    all_clean_ok = all(v['equal'] for v in clean_x.values())
    c0_ok = all(v['equal'] for v in c0_checks.values())
    doc = {'schema': 'p515_s53_w144_clean_swing_variant_v1',
           'task': 'W144: zero-solve records-only replay of clean-swing variants of settling criterion v6 (expert ruling input)',
           'utc': _utc(), 'git_head': git_head, 'code_sha256': code, 'definition': __doc__,
           'objective_convention': 'Q = gross_operational_cost (per_cycle_record.jsonl), EUR, settlement excluded',
           'constants': {'TAU': TAU, 'SWING_FLOOR': SC6.SWING_FLOOR, 'EPS0': SC6.EPS0, 'P_MAX': V6.P_MAX,
                         'GAP_BOUND': SC6.GAP_BOUND, 'MOVEMENT_BOUND': MOVEMENT_BOUND_TAU * TAU,
                         'MOVEMENT_BOUND_TAU': MOVEMENT_BOUND_TAU},
           'variants': VARIANTS, 'records': order, 'table': table,
           'self_tests': {'C0_re_equals_C0_every_record': not selftest_fail, 'failures': selftest_fail},
           'all_clean_crosscheck_all_equal': all_clean_ok,
           'c0_reproduces_every_committed_v6_decision': c0_ok,
           'c0_vs_own_committed_run_REPORTED': {rid: c0_checks[rid]['own_committed_run_REPORTED'] for rid in order},
           'changes_vs_C0': changes_vs_c0, 'committed_certificate_changes': committed_cert_changes,
           'post_certificate_movement': post_cert,
           'frequency_non_clean_after_N': freq_table, 'frequency_by_E_MWh': {k: dict(v) for k, v in by_E.items()},
           'recovering_blocks_after_N_all_records': by_block,
           'd_36686489_detail': d_detail,
           'inputs_pinned': {'w141': {'path': W141_JSON, 'sha256': w141_sha},
                             'w141_inventory_new': {'path': W141_INV, 'sha256': w141_inv_sha},
                             'w142_v6_from_records': {'path': W142_V6['path'], 'sha256': w142_sha},
                             'w139_inventory': {'path': W141.W139_INV, 'sha256': _sha(W141.W139_INV)}},
           'log_inventory': {'entries_read_this_run': len(W139.W131.INVENTORY), 'equal_to_prior_inventories': same,
                             'differ_from_prior_inventories': differ, 'new_entries': len(new_inv)},
           'reports': reports, 'guards': guards, 'pickle_guard': dict(W141.PICKLE_COUNTS), 'wall_s': time.time() - t0}
    jp = os.path.join(out_dir, OUT_JSON)
    with open(jp, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    ip = os.path.join(out_dir, OUT_INV)
    with open(ip, 'x') as handle:
        GRIO.dump(new_inv, handle, indent=1, sort_keys=True)
    man = {os.path.relpath(p, REPO): _sha(os.path.relpath(p, REPO)) for p in (jp, ip)}
    for v in doc['inputs_pinned'].values():
        man[v['path']] = v['sha256']
    for rep in reports.values():
        for rel, v in rep['inputs_sha256'].items():
            if isinstance(v, str):
                man[rel] = v
            elif isinstance(v, dict) and isinstance(v.get('sha256'), str):
                man[rel] = v['sha256']
    with open(os.path.join(out_dir, OUT_MAN), 'x') as handle:
        GRIO.dump(man, handle, indent=1, sort_keys=True)
    for name, ch in changes_vs_c0.items():
        _log(f'CHANGES vs C0 {name}: {ch}')
    for name, ch in committed_cert_changes.items():
        _log(f'COMMITTED CERTIFICATE CHANGES {name}: {ch}')
    for name, pc in post_cert.items():
        _log(f'POST-CERT {name}: ' + '; '.join(f"{x['record']} k*{x['k_star']} {x['Q_last_minus_Q_kstar']:.2f} "
                                              f"({x['over_tau']:.3f} tau, max {x['max_abs_over_tau']:.3f})"
                                              f"{' >0.9tau' if x['exceeds_0_9_tau'] else ''}" for x in pc))
    for name, e in d_detail['per_variant'].items():
        _log(f"D36686489 {name}: {e['status']} k* {e['k_star']} {e['branch']} window {e['window']} band {e['band']} "
             f"width {e['band_width']} range/tau {e['range_over_tau']} t_sum {e['t_sum_k_star']} "
             f"Qcap-Qk* {e.get('Q_cap_minus_Q_kstar')} ({e.get('Q_cap_minus_Q_kstar_over_tau')} tau) "
             f"growth {e.get('growth_test_at_kstar')} T {e['T']} A {e['A']} rej {len(e['floor_rejections'] or [])}")
    for r in freq_table:
        _log(f"FREQ {r['record']}: E {r['E_MWh_total']} P {r['P_MVA_total']} N {r['N_first_k0']} n {r['cycles_recorded']} "
             f"non-clean after N {r['n_non_clean_after_N']} {r['non_clean_cycles_after_N']} blocks {r['non_clean_blocks_after_N']}")
    for blk, v in by_block.items():
        _log(f"RECOVERING {blk}: recovery cycles after N {v['recovery_cycles_after_N']}, non-clean {v['non_clean_after_N']}")
    _log(f"self-tests pass {not selftest_fail}; all_clean x-check {all_clean_ok}; C0 control {c0_ok}; logs "
         f"{len(W139.W131.INVENTORY)} (new {len(new_inv)}, differ {len(differ)}); guards "
         f"{ {k: (v['counts'], v['verify_0_failures']) for k, v in guards.items()} }; pickle {W141.PICKLE_COUNTS}; "
         f"wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = W141._PICKLE_ORIG
    ok = (all(not v['verify_0_failures'] for v in guards.values()) and not selftest_fail and all_clean_ok and c0_ok
          and not differ and W141.PICKLE_COUNTS == {'load': 0, 'loads': 0})
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
