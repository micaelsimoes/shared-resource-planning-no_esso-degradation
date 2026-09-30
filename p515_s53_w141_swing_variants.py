"""P5.15 Planner task W141 -- SETTLING-RULE SWING VARIANTS REPLAYED FROM RECORDS, to inform an expert ruling.
ZERO SOLVES, NO MODEL LOADS.

An armed `SolveProfileGuard(permitted=())` is installed BEFORE any other project import and verified at exactly 0 at the
end (every imported module's own zero-permit guard is verified at 0 as well); `pickle.load` / `pickle.loads` are wrapped
by a blocking counter for the whole run (no model or network pickle may be read) and verified at 0. Only JSON / JSONL
records and IPOPT text logs are read.

NOTHING COMMITTED IS MODIFIED. The variants are a subclass of the committed `settling_criterion_v5.SettlingRuleV5`
(`VariantRule`, below); settling_criterion*.py and the W139 readers are imported, never edited.

THE RECORDS (18): the 16 of W139 item 5 (`p515_s53_w139_v5_from_records.record_inputs`: the three v4 runs, cell 1's v3
run, the ten W118 r2 cells -- eight pb/yl cells and the F2 pair -- and the two W101 references x0 / unit_n7_4h_e1) plus
the two committed v5 runs of W140: b_4649234b@v5 (4b3be392) and d_c52e1670@v5 (51280961). The W101 c_star run is NOT a
re-settle record of this series and is not replayed.
Per record: Q_k = gross_operational_cost (per_cycle_record.jsonl), boyd_k = boyd_all_pass AND local_solves_ok, t_sum_k
from the cell's cycle file, the cap rule of the record's own declaration (W139 record_inputs; for the v5 runs
`p515_s53_w139_resettle_v5_hooks.declaration_for(cell)['cap_rule']`), over the RECORDED cycles 1..n only (a run that
stopped at its certificate has no cycles after it: a variant that does not decide by n reads "not decided within the
recorded cycles"). all_clean_k: the v5 classification of every block's final accepted exit, recomputed with W139's
readers (`block_finals` + `classify_cycle`, i.e. `settling_criterion_v5.classify_block_exit`), and cross-checked against
(a) W139's committed non_clean_cycles for its 16 records and (b) the in-run all_clean_k of the two v5 runs.

THE VARIANTS (each differs from v5 ONLY in the named clause; every other clause, constant, the veto, the gap clause, the
monotone branch and the caps are v5's):
  V0     v5 as frozen: `settling_criterion_v5.replay` itself (the committed code path). CONTROL.
  V0_re  the VariantRule re-implementation with the all-pairs swing test and no floor -- a self-test: must equal V0 on
         every record, field for field.
  V0_f0  the VariantRule with the all-pairs test and the floor machinery at F = 0 -- a self-test of the floor sign state:
         must equal V0 on every record.
  V1     last-pair swing test: "swings not growing" := A[-1] <= A[-2] (instead of every consecutive pair since k0).
  V2     pairs within the certifying span, LITERAL: only swings both of whose turning points lie at or after T[-3].t.
         Swing A[i] joins T[i] and T[i+1], so the included swings are A[-2] and A[-1] only: V2 is IDENTICAL TO V1 BY
         CONSTRUCTION (reported, not assumed: both are replayed).
  V2b    (a Worker-added reading sensitivity, NOT a Planner variant, because V2 literal collapses to V1): swings whose
         CLOSING turning point lies at or after T[-3].t, i.e. A[-3], A[-2], A[-1] (two pairs).
  V3_10  swing noise floor F = TAU / 10: a turning point registers only if the swing it closes, |Q_t - Q_T[-1]|, is
         >= F (the first turning point closes no swing and always registers). A REJECTED candidate is treated as noise
         on the leg that ended at T[-1]: T[-1] (and the swing it closed, and the sign change that registered it) is
         un-registered, and it is CARRIED as the running extreme of the resumed leg; at the next sign change the
         candidate is the more extreme of the carried point and the step-5 extremum (ties: the earlier, as
         settling_criterion._extremum). This is the standard zigzag reading of "registers only if the swing it closes
         is >= F"; with F = 0 it is version 1's sign state exactly (V0_f0). The rejected reversal does not count as a
         sign change for the monotone branch either (j_change reverts to its value before T[-1] registered).
  V3_20  as V3_10 with F = TAU / 20.
  V4_10  V1 + V3_10.     V4_20  V1 + V3_20.

PER VARIANT AND RECORD: status (certified / uncertified at the cap / not decided within the recorded cycles), k*,
branch, window, W, P_hat, band and band width, range / TAU, t_sum(k*), Q(k*) (gross_operational_cost), T and A; versus
V0: same status, same k*, Q(k*) - Q_V0(k*) with the terminal step |dQ| at each k* as its resolution. THE CREEP FLAG on
every certification: mean dQ over the certifying window, (Q_k* - Q_lo) / (k* - lo) (the W - 1 steps inside the window),
is > TAU / L_MONO = TAU / 60 in magnitude; the alternative definition with the carry-in step, (Q_k* - Q_(lo-1)) / W, is
reported beside it.
D_C52E1670 DETAIL: per variant where it would certify, the window, the drift over the window, and Q(198) - Q(k*) (the
further descent the recorded run went on to show after that certificate); the per-block movement at cycles 110..114
(recourse_blocks_all.jsonl, the per-block values whose weighted sum is Q) and the solver events at 105..120 (non-Optimal
exits, final attempt tiers, network_failures_s39_D.jsonl recoveries, esso_recovery_events_s39_D.jsonl); the quarter
means of dQ over the steps c = 140..198 and their ratios.

OUTPUT (write-once, new directory data/SRP1/Results/P515S53/w141_swing_variants/):
  w141_swing_variants.json, w141_log_inventory_new.json, manifest_sha256.json (launch.log beside, from the shell)
Launch (attached, alone, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w141_swing_variants && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w141_swing_variants.py \\
        > data/SRP1/Results/P515S53/w141_swing_variants/launch.log 2>&1
"""
import contextlib
import hashlib
import io
import json
import math
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W141 swing variants (never solves)').install()

# ---- the no-model-load guard: pickle.load / pickle.loads raise for the whole run ------------------------------------
PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W141: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W141: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import settling_criterion as SC1  # noqa: E402
import settling_criterion_v2 as SC2  # noqa: E402
import settling_criterion_v5 as SC5  # noqa: E402
import p515_s53_w139_resettle_v5_hooks as V5  # noqa: E402
import p515_s53_w139_v5_from_records as W139  # noqa: E402 -- W139's readers (arms its own guard)


def _dedupe(pairs):
    seen, out = set(), []
    for name, g in pairs:
        if id(g) not in seen:
            seen.add(id(g))
            out.append((name, g))
    return tuple(out)


GUARDS = _dedupe((('w141_swing_variants', GUARD),) + tuple(W139.GUARDS))

TAU = SC5.TAU
L_MONO = 2 * V5.P_MAX
CREEP_BOUND = TAU / L_MONO
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR_REL = os.path.join(S53, 'w141_swing_variants')
OUT_JSON = 'w141_swing_variants.json'
OUT_INV = 'w141_log_inventory_new.json'
OUT_MAN = 'manifest_sha256.json'
W139_JSON = os.path.join(S53, 'w139_resettle_v5', 'v5_from_records', 'w139_v5_from_records.json')
W139_INV = os.path.join(S53, 'w139_resettle_v5', 'v5_from_records', 'w139_v5_from_records_log_inventory.json')
V5_ROOT = os.path.join(S53, 'w139_resettle_v5')
V5_STAGE_SPEC = {'path': os.path.join(V5_ROOT, 'frozen_s53_resettle_spec_v5_ab32ffc9.json'), 'sha8': 'ab32ffc9'}
V5_RUNS = {
    'd_c52e1670@v5': {'cell': 'd_c52e1670', 'commit': '51280961',
                      'eval_dir': os.path.join(V5_ROOT, 'campaign_s53_w139_resettle_v5_d_c52e1670', 'evals',
                                               'af42a163b14f0895_d_c52e1670')},
    'b_4649234b@v5': {'cell': 'b_4649234b', 'commit': '4b3be392',
                      'eval_dir': os.path.join(V5_ROOT, 'campaign_s53_w139_resettle_v5_b_4649234b', 'evals',
                                               '4de68ced93863cec_b_4649234b')},
}
D_CELL = 'd_c52e1670@v5'
BLIP_BLOCK_CYCLES = (110, 111, 112, 113, 114)
EVENT_SPAN = (105, 120)
DESCENT_STEPS = (140, 198)

VARIANTS = {
    'V0_re': {'swing_test': 'all_pairs', 'floor': None, 'role': 'self-test: must equal V0'},
    'V0_f0': {'swing_test': 'all_pairs', 'floor': 0.0, 'role': 'self-test of the floor sign state: must equal V0'},
    'V1': {'swing_test': 'last_pair', 'floor': None, 'role': 'Planner variant: last-pair swing test'},
    'V2': {'swing_test': 'span_T3_both_endpoints', 'floor': None,
           'role': 'Planner variant: swings whose turning points lie at or after T[-3] (literal; == V1 by construction)'},
    'V2b': {'swing_test': 'span_T3_closing_endpoint', 'floor': None,
            'role': 'Worker-added reading sensitivity (not a Planner variant): swings whose closing turning point lies '
                    'at or after T[-3]'},
    'V3_10': {'swing_test': 'all_pairs', 'floor': TAU / 10.0, 'role': 'Planner variant: floor F = TAU / 10'},
    'V3_20': {'swing_test': 'all_pairs', 'floor': TAU / 20.0, 'role': 'Planner variant: floor F = TAU / 20 (reported)'},
    'V4_10': {'swing_test': 'last_pair', 'floor': TAU / 10.0, 'role': 'Planner variant: V1 + V3 (F = TAU / 10)'},
    'V4_20': {'swing_test': 'last_pair', 'floor': TAU / 20.0, 'role': 'Planner variant: V1 + V3 (F = TAU / 20)'},
}


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


def _jsonl(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ======================================================================================================================
#  the variant rule (a subclass of the committed v5 rule)
# ======================================================================================================================
class FloorSignState(SC1._SignState):
    """Version 1's step-5 sign state with a swing noise floor F (see the module docstring, V3). F = 0 reproduces
    version 1 exactly (every swing is >= 0)."""

    def __init__(self, floor):
        super().__init__()
        self.floor = float(floor)
        self.carry = None          # the un-registered T[-1], carried as the running extreme of the resumed leg
        self.jc_stack = []         # j_change before each registered turning point
        self.rejections = []       # [{cycle, candidate, closed_swing, popped}]

    def update(self, q, k, s):
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
                raise RuntimeError(f'FloorSignState: carried {self.carry} of the wrong kind at {k} ({kind})')
            ct = self.carry[0]
            if (kind == 'max' and q[ct] >= q[t]) or (kind == 'min' and q[ct] <= q[t]):
                t = ct
        tp = (t, kind, q[t])
        if self.T:
            swing = abs(q[t] - q[self.T[-1][0]])
            if swing < self.floor:
                popped = self.T.pop()
                if self.T:                       # T had >= 2 entries: A[-1] is the swing popped closed
                    self.A.pop()
                self.j_change = self.jc_stack.pop()
                self.carry = popped
                self.rejections.append({'cycle': k, 'candidate': list(tp), 'closed_swing': swing,
                                        'popped': list(popped)})
                self.sign_prev, self.j_prev = s, k
                return False, None
        self.jc_stack.append(self.j_change)
        self.T.append(tp)
        if len(self.T) >= 2:
            self.A.append(abs(q[t] - q[self.T[-2][0]]))
        self.carry = None
        self.sign_prev, self.j_prev, self.j_change = s, k, k
        return True, tp


def swing_test(kind, A, T):
    """(ok, pairs compared as [[i, i+1], ...] indices into A)."""
    n = len(A)
    if kind == 'all_pairs':
        idx = list(range(n))
    elif kind == 'last_pair':
        idx = list(range(max(0, n - 2), n))
    elif kind == 'span_T3_both_endpoints':
        t3 = T[-3][0] if len(T) >= 3 else None
        idx = [i for i in range(n) if t3 is not None and T[i][0] >= t3 and T[i + 1][0] >= t3]
    elif kind == 'span_T3_closing_endpoint':
        t3 = T[-3][0] if len(T) >= 3 else None
        idx = [i for i in range(n) if t3 is not None and T[i + 1][0] >= t3]
    else:
        raise ValueError(kind)
    pairs = [[idx[j], idx[j + 1]] for j in range(len(idx) - 1)]
    return all(A[j2] <= A[j1] for j1, j2 in pairs), pairs


class VariantRule(SC5.SettlingRuleV5):
    """v5 with the oscillatory swing test replaced (`swing_kind`) and/or a swing noise floor (`floor`). evaluate() is
    version 2's clauses (called, not copied) with cert_a / cert_b / the reasons recomputed from the variant swing test,
    then version 5's veto block (reproduced from settling_criterion_v5.SettlingRuleV5.evaluate, which calls
    SettlingRuleV2.evaluate by class name and so cannot be intercepted). The self-tests V0_re / V0_f0 prove the
    re-implementation equal to v5 on every record."""

    def __init__(self, p_max, swing_kind='all_pairs', floor=None, **kw):
        self._floor = floor
        self.swing_kind = swing_kind
        super().__init__(p_max, **kw)

    @property
    def sign(self):
        return self._sign

    @sign.setter
    def sign(self, value):
        # SettlingRuleV2 assigns a fresh SC1._SignState at init and at every k0 reset; replace it by the floor state.
        if self._floor is None:
            self._sign = value
        else:
            if not (isinstance(value, SC1._SignState) and not value.T and value.sign_prev is None):
                raise RuntimeError('VariantRule: the sign state must only be assigned fresh')
            self._sign = FloorSignState(self._floor)

    def evaluate(self, k):
        _ca, _cb, a_parts, b_parts, _r = SC2.SettlingRuleV2.evaluate(self, k)
        ok, pairs = swing_test(self.swing_kind, self.sign.A, self.sign.T)
        a_parts.update({'swing_test': self.swing_kind, 'swings_ok_variant': ok, 'swing_pairs_compared': pairs,
                        'floor': self._floor})
        cert_a = bool(a_parts['at_least_3_turning_points'] and ok and a_parts['window_inside_run']
                      and a_parts['range_le_tau'])
        b_ok = bool(b_parts['lo_ge_k0_plus_K_EXCL'] and b_parts['no_sign_change_in_window'] and b_parts['range_le_tau']
                    and b_parts['steps_decreasing'] and b_parts['last_step_times_L_le_tau'])
        cert_b = (not cert_a) and b_ok
        reasons = []
        if not cert_a and not cert_b:                     # version 2's reason list, the swing reason from the variant
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
        # ---- version 5's veto (settling_criterion_v5.SettlingRuleV5.evaluate, reproduced) ----
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


def replay_variant(q, b, t, clean, p_max, swing_kind, floor, last, **kw):
    """settling_criterion_v5.replay's loop with VariantRule."""
    rule = VariantRule(p_max, swing_kind=swing_kind, floor=floor, **kw)
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


# ======================================================================================================================
#  the records
# ======================================================================================================================
def v5_run_inputs():
    """The two committed v5 runs, in W139 record_inputs' form."""
    out = {}
    for rid, v in V5_RUNS.items():
        ev = v['eval_dir']
        camp = ev.split(os.sep + 'evals' + os.sep)[0]
        man = json.load(open(os.path.join(REPO, camp, 'campaign_manifest_sha256.json')))
        inputs = {}
        for f in ('per_cycle_record.jsonl', V5.CYCLE_FILE, V5.DECISION_FILE, 'recourse_blocks_all.jsonl',
                  'network_failures_s39_D.jsonl', 'esso_recovery_events_s39_D.jsonl',
                  'network_ipopt_solve_records.jsonl', 'evaluation_record.json'):
            rel = os.path.join(ev, f)
            now = _sha(os.path.join(REPO, rel))
            if man.get(rel) != now:
                raise RuntimeError(f'{rel}: sha256 {now} != campaign manifest {man.get(rel)}')
            inputs[rel] = now
        rows = _jsonl(os.path.join(ev, 'per_cycle_record.jsonl'))
        lines = {x['cycle']: x for x in _jsonl(os.path.join(ev, V5.CYCLE_FILE))}
        dec = json.load(open(os.path.join(REPO, ev, V5.DECISION_FILE)))
        q = {r['cycle']: r['gross_operational_cost'] for r in rows}
        b = {r['cycle']: bool(r['boyd_all_pass'] and r['local_solves_ok']) for r in rows}
        t = {k: (lines.get(k) or {}).get('t_sum') for k in q}
        cr = V5.declaration_for(v['cell'])['cap_rule']
        kw = ({'cap': cr['cap'], 'cap_ceiling': cr['ceiling']} if cr['kind'] == 'fixed' else
              {'cap_after_first_k0': cr['after_first_k0'], 'cap_ceiling': cr['ceiling']})
        run = {'spec': V5_STAGE_SPEC, 'criterion': 'v5 (Addendum 60)', 'commit': v['commit'],
               'status': dec.get('status'), 'k_star': dec.get('k_star'), 'k_cap': dec.get('k_cap')}
        out[rid] = {'cell': v['cell'], 'eval_dir': ev, 'q': q, 'b': b, 't': t, 'n': len(rows), 'kw': kw, 'run': run,
                    'certifying': ({'spec': V5_STAGE_SPEC, 'criterion': 'v5', 'k_star': dec.get('k_star')}
                                   if dec.get('status') == 'certified' else None),
                    'inputs_sha256': inputs, 'in_run_all_clean': {k: lines[k]['all_clean_k'] for k in q},
                    'in_run_decision': dec}
    return out


def candidate_key(eval_dir):
    p = os.path.join(REPO, eval_dir, 'evaluation_record.json')
    if not os.path.exists(p):
        return None
    ev = json.load(open(p))
    return {'candidate_key': ev.get('candidate_key'), 'candidate_label': ev.get('candidate_label'),
            'evaluation_record_sha256': _sha(p)}


# ======================================================================================================================
#  summaries
# ======================================================================================================================
def summarize(dec, rule, q, n):
    if dec is None:
        return {'status': f'not decided within the recorded cycles 1..{n}', 'decided': False, 'k_star': None,
                'k_cap': None, 'k0_at_end': rule.k0, 'T': [list(x) for x in rule.sign.T], 'A': list(rule.sign.A),
                'n_vetoes': len(rule.vetoes), 'n_gap_refusals': len(rule.gap_refusals),
                'gap_refusals': list(rule.gap_refusals), 'vetoes': list(rule.vetoes),
                'floor_rejections': list(getattr(rule.sign, 'rejections', []))}
    out = {k: dec.get(k) for k in ('status', 'k_star', 'k_cap', 'branch', 'k0', 'N', 'W', 'P_hat', 'window', 'band',
                                   'band_width', 'range', 'range_over_tau', 't_sum_k_star', 'Q_k_star', 'reasons',
                                   'T', 'A', 'drift_rate_mean_dQ_last_25', 'band_window')}
    out['decided'] = True
    out['n_vetoes'] = len(rule.vetoes)
    out['vetoes'] = list(rule.vetoes)
    out['n_gap_refusals'] = len(rule.gap_refusals)
    out['gap_refusals'] = list(rule.gap_refusals)
    out['floor_rejections'] = list(getattr(rule.sign, 'rejections', []))
    if dec['status'] == 'certified':
        k, (lo, hi) = dec['k_star'], dec['window']
        assert hi == k
        internal = (q[k] - q[lo]) / (k - lo)
        carry = (q[k] - q[lo - 1]) / (k - lo + 1) if (lo - 1) in q else None
        out['creep'] = {'mean_dQ_window_internal': internal, 'definition': '(Q_k* - Q_lo) / (k* - lo)',
                        'mean_dQ_with_carry_in': carry,
                        'definition_with_carry_in': '(Q_k* - Q_(lo-1)) / W',
                        'drift_over_window_Q_kstar_minus_Q_lo': q[k] - q[lo],
                        'bound_tau_over_L': CREEP_BOUND,
                        'creep_flag': abs(internal) > CREEP_BOUND,
                        'creep_flag_with_carry_in': (abs(carry) > CREEP_BOUND) if carry is not None else None,
                        'last_step_dQ_kstar': q[k] - q[k - 1]}
        out['swing_test_at_kstar'] = {x: dec['certA_parts'].get(x) for x in (
            'swing_test', 'swings_ok_variant', 'swing_pairs_compared', 'swings_non_increasing_all_pairs', 'floor')}
    return out


def compare_to_v0(s, v0, q):
    same_status = s['status'] == v0['status']
    same_k = s.get('k_star') == v0.get('k_star')
    out = {'same_status': same_status, 'same_k_star': same_k if (s['status'] == 'certified'
                                                                 or v0['status'] == 'certified') else None,
           'certifies_where_v0_does_not': s['status'] == 'certified' and v0['status'] != 'certified',
           'v0_certifies_where_variant_does_not': v0['status'] == 'certified' and s['status'] != 'certified'}
    if s['status'] == 'certified' and v0['status'] == 'certified':
        k1, k0 = s['k_star'], v0['k_star']
        out['k_star_minus_v0'] = k1 - k0
        out['Q_kstar_minus_v0_Q_kstar'] = q[k1] - q[k0]
        out['resolution_terminal_steps_abs'] = {'variant': abs(q[k1] - q[k1 - 1]), 'v0': abs(q[k0] - q[k0 - 1])}
        out['certified_value_differs'] = k1 != k0
    return out


def _core(s):
    """The comparable decision fields (for the self-tests)."""
    keys = ('status', 'k_star', 'k_cap', 'branch', 'k0', 'W', 'P_hat', 'window', 'band', 'band_width', 'range',
            't_sum_k_star', 'reasons', 'T', 'A', 'n_vetoes', 'n_gap_refusals', 'drift_rate_mean_dQ_last_25')
    return {k: s.get(k) for k in keys}


# ======================================================================================================================
#  d_c52e1670 detail
# ======================================================================================================================
def d_cell_detail(rec, per_cycle, variant_summaries):
    ev = rec['eval_dir']
    q = rec['q']
    n = rec['n']
    out = {'record': D_CELL, 'eval_dir': ev, 'cycles_recorded': n, 'Q_at_last_recorded': q[n]}
    # where each variant certifies
    per_var = {}
    for name, s in variant_summaries.items():
        e = {'status': s['status'], 'k_star': s.get('k_star'), 'branch': s.get('branch'), 'window': s.get('window'),
             'W': s.get('W'), 'P_hat': s.get('P_hat'), 'band_width': s.get('band_width'),
             'range_over_tau': s.get('range_over_tau'), 't_sum_k_star': s.get('t_sum_k_star'),
             'T': s.get('T'), 'A': s.get('A'), 'n_gap_refusals': s.get('n_gap_refusals'),
             'gap_refusal_cycles': [g['cycle'] for g in s.get('gap_refusals') or []],
             'floor_rejections': s.get('floor_rejections')}
        if s['status'] == 'certified':
            k = s['k_star']
            e.update({'Q_k_star': q[k], 'creep': s['creep'],
                      'Q_last_recorded_minus_Q_k_star': q[n] - q[k],
                      'Q_last_recorded_minus_Q_k_star_over_tau': (q[n] - q[k]) / TAU,
                      'mean_dQ_after_k_star_to_last': (q[n] - q[k]) / (n - k) if n > k else None})
        per_var[name] = e
    out['per_variant'] = per_var
    # the blip: per-block movement
    blocks = {x['cycle']: x for x in _jsonl(os.path.join(ev, 'recourse_blocks_all.jsonl'))}

    def bkey(b):
        return (f"TSO|{b['year']}|{b['day']}" if b['agent'] == 'TSO' else
                f"{b['agent']}|{b.get('node_id')}|{b['year']}|{b['day']}")

    moves = {}
    for c in BLIP_BLOCK_CYCLES:
        prev = {bkey(b): b['value'] for b in blocks[c - 1]['blocks']}
        cur = {bkey(b): b['value'] for b in blocks[c]['blocks']}
        d = {kk: cur[kk] - prev[kk] for kk in cur}
        agg = {}
        for kk, v in d.items():
            a = kk.split('|')[0] + ('|' + kk.split('|')[1] if kk.startswith('DSO') else '')
            agg[a] = agg.get(a, 0.0) + v
        top = sorted(d.items(), key=lambda x: -abs(x[1]))[:6]
        moves[c] = {'dQ_gross': q[c] - q[c - 1],
                    'blocks_gross_field_delta': blocks[c]['gross_operational_cost'] - blocks[c - 1]['gross_operational_cost'],
                    'sum_block_value_deltas_unweighted': sum(d.values()),
                    'n_blocks': len(d), 'by_agent_unweighted': agg, 'top6_blocks_unweighted': top,
                    'note': ('block "value" is the per-block recourse as recorded in recourse_blocks_all.jsonl; the sum '
                             'of the unweighted deltas need not equal dQ (Q weights blocks by year / day factors)')}
    out['blip_block_moves'] = moves
    # solver events 105..120
    fails = _jsonl(os.path.join(ev, 'network_failures_s39_D.jsonl'))
    esso_rec = _jsonl(os.path.join(ev, 'esso_recovery_events_s39_D.jsonl'))
    lo, hi = EVENT_SPAN
    ev_rows = {}
    for c in range(lo, hi + 1):
        pc = per_cycle[c]
        ev_rows[c] = {'dQ': q[c] - q[c - 1],
                      'non_optimal_final_exits': [[kk, e['class'], e['attempt'], e['reason']] for kk, e in pc.items()
                                                  if e['class'] != SC5.OPTIMAL_CLASS],
                      'final_attempt_not_primary': [[kk, e['attempt'], e['class']] for kk, e in pc.items()
                                                    if e['attempt'] != 'primary'],
                      'network_attempts_gt1': [[kk, e['attempts']] for kk, e in pc.items()
                                               if e['family'] == 'network' and isinstance(e.get('attempts'), int)
                                               and e['attempts'] > 1],
                      'network_failures_records': [[f.get('agent'), f.get('node_id'), f.get('year'), f.get('day'),
                                                    f.get('primary_termination')] for f in fails if f['cycle'] == c]}
    out['solver_events_105_120'] = ev_rows
    out['network_failures_all_cycles'] = [[f['cycle'], f.get('agent'), f.get('node_id'), f.get('year'), f.get('day'),
                                           f.get('primary_termination')] for f in fails]
    out['esso_recovery_events_n'] = len(esso_rec)
    # the post-140 descent
    s0, s1 = DESCENT_STEPS
    steps = [(c, q[c] - q[c - 1]) for c in range(s0, s1 + 1)]
    m = len(steps)
    base, extra = divmod(m, 4)
    quarters, i = [], 0
    for j in range(4):
        size = base + (1 if j < extra else 0)
        chunk = steps[i:i + size]
        i += size
        quarters.append({'steps_c': [chunk[0][0], chunk[-1][0]], 'n': len(chunk),
                         'mean_dQ': sum(x[1] for x in chunk) / len(chunk)})
    out['descent_140_198'] = {
        'steps': 'dQ_c = Q_c - Q_(c-1) for c = 140..198', 'n_steps': m,
        'n_negative': sum(1 for _c, d in steps if d < 0), 'first_non_negative': next((c for c, d in steps if d >= 0), None),
        'mean_dQ_all': sum(d for _c, d in steps) / m, 'quarters': quarters,
        'ratio_Q4_over_Q1': quarters[3]['mean_dQ'] / quarters[0]['mean_dQ'],
        'successive_ratios': [quarters[j + 1]['mean_dQ'] / quarters[j]['mean_dQ'] for j in range(3)],
        'min_step': min(d for _c, d in steps), 'max_step': max(d for _c, d in steps),
        'last_10_steps': steps[-10:]}
    return out


# ======================================================================================================================
def main():
    t0 = time.time()
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_JSON, OUT_INV, OUT_MAN):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    w139_doc = json.load(open(os.path.join(REPO, W139_JSON)))
    w139_inv = json.load(open(os.path.join(REPO, W139_INV)))
    with contextlib.redirect_stdout(io.StringIO()):
        recs = W139.record_inputs()
    recs.update(v5_run_inputs())
    order = list(V5_RUNS) + list(W139.V4_RUNS) + [r for r in recs if r not in V5_RUNS and r not in W139.V4_RUNS]
    if len(order) != 18 or len(set(order)) != 18:
        raise RuntimeError(f'expected 18 records, got {order}')
    reports, table, selftest_fail, clean_xcheck = {}, [], [], {}
    d_detail = None
    for rid in order:
        rec = recs[rid]
        finals, meta = W139.block_finals(rec['eval_dir'], rec['n'])
        per_cycle = {k: W139.classify_cycle(finals[k]) for k in range(1, rec['n'] + 1)}
        clean = {k: all(e['clean'] for e in per_cycle[k].values()) for k in per_cycle}
        non_clean = sorted(k for k, v in clean.items() if not v)
        if rid in w139_doc['reports']:
            ref = w139_doc['reports'][rid]['non_clean_cycles']
            clean_xcheck[rid] = {'against': 'W139 committed non_clean_cycles', 'equal': ref == non_clean,
                                 'non_clean_cycles': non_clean}
        else:
            ref = sorted(k for k, v in rec['in_run_all_clean'].items() if not v)
            clean_xcheck[rid] = {'against': 'in-run all_clean_k (resettle_cycle_record.jsonl)', 'equal': ref == non_clean,
                                 'non_clean_cycles': non_clean}
        if not meta['coverage_51_every_cycle']:
            raise RuntimeError(f'{rid}: block coverage is not 51 on every cycle')
        q, b, t, n, kw = rec['q'], rec['b'], rec['t'], rec['n'], rec['kw']
        _o, d0, r0 = SC5.replay(q, b, t, clean, V5.P_MAX, last=n, **kw)
        s0 = summarize(d0, r0, q, n)
        sums = {'V0': s0}
        for name, v in VARIANTS.items():
            _o, dv, rv = replay_variant(q, b, t, clean, V5.P_MAX, v['swing_test'], v['floor'], n, **kw)
            sums[name] = summarize(dv, rv, q, n)
        for st in ('V0_re', 'V0_f0'):
            if _core(sums[st]) != _core(s0):
                selftest_fail.append({'record': rid, 'self_test': st, 'v0': _core(s0), 'got': _core(sums[st])})
        # V0 against the committed decisions
        if rid in w139_doc['reports']:
            w = w139_doc['reports'][rid]['v5_from_records']
            fields = ('status', 'k_star', 'k_cap', 'window', 'branch', 'n_vetoes')
            exp = {k: w.get(k) for k in fields}
            got = {k: s0.get(k) for k in fields}
            # W139 wrote an undecided replay as 'not certified within the recorded cycles 1..n'; this module as 'not
            # decided within the recorded cycles 1..n': both normalised to 'undecided'
            if str(exp['status']).startswith('not certified within'):
                exp['status'] = 'undecided'
            if not s0['decided']:
                got['status'] = 'undecided'
            v0_repro = {'against': 'W139 committed v5_from_records', 'fields': list(fields), 'expected': exp,
                        'got': got, 'equal': exp == got}
        else:
            dec = rec['in_run_decision']
            fields = ('status', 'k_star', 'k_cap', 'window', 'branch', 'reasons', 'T', 'A', 'band_width', 'n_vetoes')
            exp = {k: dec.get(k) for k in fields}
            got = {k: s0.get(k) for k in fields}
            v0_repro = {'against': 'the committed in-run resettle_decision.json', 'fields': list(fields),
                        'expected': exp, 'got': got, 'equal': exp == got}
        rep = {'record': rid, 'cell': rec['cell'], 'eval_dir': rec['eval_dir'], 'instance': candidate_key(rec['eval_dir']),
               'cycles_recorded': n, 'cap_rule_replayed': kw, 'committed_run': rec['run'],
               'certifying_spec': rec['certifying'], 'non_clean_cycles': non_clean,
               'all_clean_crosscheck': clean_xcheck[rid], 'v0_reproduces_committed': v0_repro,
               'variants': {name: dict(s, vs_v0=compare_to_v0(s, s0, q)) for name, s in sums.items()},
               'inputs_sha256': rec['inputs_sha256'], 'sources': meta}
        reports[rid] = rep
        row = {'record': rid, 'committed': {k: rec['run'].get(k) for k in ('criterion', 'commit', 'status', 'k_star',
                                                                            'k_cap')},
               'v0_reproduces_committed': v0_repro['equal']}
        for name, s in sums.items():
            row[name] = {'status': s['status'] if s['decided'] else 'undecided', 'k_star': s.get('k_star'),
                         'branch': s.get('branch'), 'band_width': s.get('band_width'),
                         'range_over_tau': s.get('range_over_tau'), 't_sum_k_star': s.get('t_sum_k_star'),
                         'creep_flag': (s.get('creep') or {}).get('creep_flag'),
                         'mean_dQ_window': (s.get('creep') or {}).get('mean_dQ_window_internal'),
                         'same_k_star_as_v0': rep['variants'][name]['vs_v0']['same_k_star'],
                         'same_status_as_v0': rep['variants'][name]['vs_v0']['same_status']}
        table.append(row)
        _log(f"{rid}: all_clean x-check {clean_xcheck[rid]['equal']} (non-clean {non_clean}); V0 repro "
             f"{v0_repro['equal']} | " + ' | '.join(
                 f"{nm} {(r['status'] if r['status'] != 'certified' else 'cert')}"
                 f"{' ' + str(r['k_star']) if r['k_star'] else ''}"
                 f"{' ' + (r['branch'] or '')[:3] if r['branch'] else ''}"
                 f"{' CREEP' if r['creep_flag'] else ''}" for nm, r in row.items()
                 if nm in ('V0',) + tuple(VARIANTS)))
        if rid == D_CELL:
            d_detail = d_cell_detail(rec, per_cycle, sums)
    changes = {}
    for name in VARIANTS:
        ch = []
        for row in table:
            r, r0 = row[name], row['V0']
            if (r['status'], r['k_star']) != (r0['status'], r0['k_star']):
                ch.append({'record': row['record'], 'v0': [r0['status'], r0['k_star']], 'variant': [r['status'],
                                                                                                    r['k_star']]})
        changes[name] = ch
    creep_certs = {name: [row['record'] for row in table if row[name]['creep_flag']] for name in ('V0',) + tuple(VARIANTS)}
    # the new log inventory: entries W139 did not inventory, and those it did verified unchanged
    new_inv, same, differ = {}, 0, []
    for rel, v in W139.W131.INVENTORY.items():
        old = w139_inv.get(rel)
        if old is None:
            new_inv[rel] = v
        elif old.get('sha256') == v.get('sha256'):
            same += 1
        else:
            differ.append(rel)
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    git_head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    code = {rel: _sha(os.path.join(REPO, rel)) for rel in (
        os.path.basename(__file__), 'settling_criterion.py', 'settling_criterion_v2.py', 'settling_criterion_v4.py',
        'settling_criterion_v5.py', 'p515_s53_w139_v5_from_records.py', 'p515_s53_w139_resettle_v5_hooks.py',
        'p515_s53_w131_prefreeze_diagnostics.py', 'p515_s53_w137_recurring_acceptable.py',
        'p515_s53_w137_resettle_v4_checks.py')}
    doc = {'schema': 'p515_s53_w141_swing_variants_v1',
           'task': 'W141: zero-solve records-only replay of settling-rule swing variants (to inform an expert ruling)',
           'utc': _utc(), 'git_head': git_head, 'code_sha256': code, 'definition': __doc__,
           'objective_convention': 'Q = gross_operational_cost (per_cycle_record.jsonl), EUR',
           'constants': {'TAU': TAU, 'EPS0': SC5.EPS0, 'L_MONO': L_MONO, 'P_MAX': V5.P_MAX, 'GAP_BOUND': SC2.GAP_BOUND,
                         'CREEP_BOUND_TAU_OVER_L': CREEP_BOUND, 'F_TAU_10': TAU / 10.0, 'F_TAU_20': TAU / 20.0},
           'variants': VARIANTS, 'records': order, 'table': table,
           'self_tests': {'V0_re_and_V0_f0_equal_V0_every_record': not selftest_fail, 'failures': selftest_fail},
           'all_clean_crosscheck_all_equal': all(v['equal'] for v in clean_xcheck.values()),
           'v0_reproduces_every_committed_decision': all(r['v0_reproduces_committed']['equal'] for r in reports.values()),
           'changes_vs_v0': changes, 'creep_flagged_certifications': creep_certs,
           'd_c52e1670_detail': d_detail,
           'log_inventory': {'w139_inventory': {'path': W139_INV, 'sha256': _sha(os.path.join(REPO, W139_INV))},
                             'entries_read_this_run': len(W139.W131.INVENTORY), 'equal_to_w139_inventory': same,
                             'differ_from_w139_inventory': differ, 'new_entries': len(new_inv)},
           'reports': reports, 'guards': guards, 'pickle_guard': dict(PICKLE_COUNTS), 'wall_s': time.time() - t0}
    jp = os.path.join(out_dir, OUT_JSON)
    with open(jp, 'x') as handle:
        GRIO.dump(doc, handle, indent=1, sort_keys=True, default=GRIO.json_default)
    ip = os.path.join(out_dir, OUT_INV)
    with open(ip, 'x') as handle:
        GRIO.dump(new_inv, handle, indent=1, sort_keys=True)
    man = {os.path.relpath(p, REPO): _sha(p) for p in (jp, ip)}
    man[W139_JSON] = _sha(os.path.join(REPO, W139_JSON))
    man[W139_INV] = doc['log_inventory']['w139_inventory']['sha256']
    for rep in reports.values():
        for rel, v in rep['inputs_sha256'].items():
            if isinstance(v, str):
                man[rel] = v
            elif isinstance(v, dict) and isinstance(v.get('sha256'), str):
                man[rel] = v['sha256']
    with open(os.path.join(out_dir, OUT_MAN), 'x') as handle:
        GRIO.dump(man, handle, indent=1, sort_keys=True)
    for name, ch in changes.items():
        _log(f'CHANGES {name}: {ch}')
    _log(f'CREEP-flagged certifications: {creep_certs}')
    if d_detail:
        for name, e in d_detail['per_variant'].items():
            cr = e.get('creep') or {}
            _log(f"D {name}: {e['status']} k* {e['k_star']} {e['branch']} window {e['window']} W {e['W']} band "
                 f"{e['band_width']} range/tau {e['range_over_tau']} t_sum {e['t_sum_k_star']} mean dQ window "
                 f"{cr.get('mean_dQ_window_internal')} drift {cr.get('drift_over_window_Q_kstar_minus_Q_lo')} CREEP "
                 f"{cr.get('creep_flag')} Q198-Qk* {e.get('Q_last_recorded_minus_Q_k_star')} gap refusals "
                 f"{e['gap_refusal_cycles']} T {e['T']}")
        dd = d_detail['descent_140_198']
        _log(f"D descent 140..198: n {dd['n_steps']} negative {dd['n_negative']} quarters "
             f"{[(x['steps_c'], round(x['mean_dQ'], 2)) for x in dd['quarters']]} ratio Q4/Q1 {dd['ratio_Q4_over_Q1']:.3f}")
        for c, m in d_detail['blip_block_moves'].items():
            _log(f"D blip c{c}: dQ {m['dQ_gross']:.2f} by agent {({k: round(v, 2) for k, v in m['by_agent_unweighted'].items()})} "
                 f"top {[(k, round(v, 2)) for k, v in m['top6_blocks_unweighted'][:3]]}")
        for c, e in d_detail['solver_events_105_120'].items():
            if e['non_optimal_final_exits'] or e['final_attempt_not_primary'] or e['network_attempts_gt1'] \
                    or e['network_failures_records']:
                _log(f'D events c{c}: {e}')
    _log(f"self-tests pass {not selftest_fail}; all_clean x-check {doc['all_clean_crosscheck_all_equal']}; V0 reproduces "
         f"committed {doc['v0_reproduces_every_committed_decision']}; logs {doc['log_inventory']['entries_read_this_run']} "
         f"(new {len(new_inv)}, differ {len(differ)}); guards {guards}; pickle {PICKLE_COUNTS}; wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    ok = (all(not v['verify_0_failures'] for v in guards.values()) and not selftest_fail
          and doc['all_clean_crosscheck_all_equal'] and doc['v0_reproduces_every_committed_decision']
          and not differ and PICKLE_COUNTS == {'load': 0, 'loads': 0})
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
