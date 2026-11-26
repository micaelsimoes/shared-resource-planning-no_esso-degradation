"""The settling stop rule for multi-scenario certification continuations (P5.15 Addendum 53, Ruling 1; W101).

A PURE function of two per-cycle sequences: Q_k (`gross_operational_cost`, None when any local solve failed) and
boyd_k (`all_boyd_pass` AND `local_solves_ok`, from THIS cycle's `boyd_metrics`). Stdlib only; no I/O; no model code.
The algorithm is the Advisor's operationalisation adopted by the Planner (TASKS.md, Addendum 53 order), implemented
verbatim; the step numbers below are the algorithm's.

CONSTANTS (each with its formula; `constants()` returns them with the formulas for the spec):
  TAU      = DELTA_R * R_REF / 4 = 0.07 * 259,375.33 / 4 = 4,539.068275 EUR   (Addendum 53 Ruling 1; tau = dR V_SRP1 / 4)
  EPS0     = TAU / 100 = 45.39068275 EUR   (a step that cannot accumulate tau within the 100-cycle cap)
  K_EXCL   = 3                               (the first K_EXCL cycles after k0 never enter the sign state)
  W_MIN    = 20;  W_FACTOR = 1.1             (oscillatory window W = max(W_MIN, ceil(W_FACTOR * P_hat)))
  P_MAX    = a SPEC constant: the longest interval between same-kind turning points found by `turning_points` (this
             module; same sign definition, same EPS0 floor) over cycles 1..N of the three SRP1 reference records;
             computed by `p_max_from_records` at freeze and passed in (never a literal here)
  L_MONO   = 2 * P_MAX                       (monotone window length)
  CAP      = N + 100 per cell                (passed in)

THE ALGORITHM, for k = 1..CAP after Q_k and boyd_k are known (`SettlingRule.observe`):
  1. Q_k None or not boyd_k: record a lapse event if k0 was set; k0 = None; reset sign_prev, j_prev, j_change, T, A;
     continue.
  2. k0 None: k0 = k; reset the state.
  3. k < k0 + K_EXCL: continue.
  4. dQ_k = Q_k - Q_(k-1); s_k = 0 if |dQ_k| < EPS0 else sign(dQ_k).
  5. s_k == 0: no update. sign_prev None: sign_prev = s_k, j_prev = k. s_k == sign_prev: j_prev = k. Otherwise a SIGN
     CHANGE: t = argmax Q over [j_prev .. k-1] if sign_prev = +1 else argmin (ties -> the earliest cycle); append
     (t, kind, Q_t) to T; if len(T) >= 2 append |Q_t - Q_(T[-2].t)| to A; sign_prev = s_k; j_prev = k; j_change = k.
  6. k <= N: continue (the replay: state accumulates, no decision is taken).
  7. certA (oscillatory): len(T) >= 3; A[i+1] <= A[i] for ALL consecutive pairs; with P_hat = T[-1].t - T[-3].t and
     W = max(W_MIN, ceil(W_FACTOR P_hat)): k - W + 1 >= k0 and range(Q over [k-W+1 .. k]) <= TAU.
  8. certB (monotone), only if not certA; lo = k - L_MONO + 1: lo >= k0 + K_EXCL; j_change None or j_change < lo;
     range(Q over [lo .. k]) <= TAU; max |dQ| over [lo .. k] < EPS0 (stationary) OR mean |dQ| over the second half of
     [lo .. k] < mean over the first half (strictly decreasing).
  9. certA or certB: certify at k* = k (branch oscillatory / monotone); the caller ends the run at the end of THIS cycle.
 10. k == CAP (not certified): an 'uncertified' record with every failing reason (FAILING_REASONS); band = [min, max] Q
     over the last max(W_MIN, ceil(W_FACTOR P_hat)) cycles, or the last W_MIN cycles if there is no P_hat.

READINGS ADOPTED BY THE PLANNER (READINGS, recorded in the spec): window max(20, ceil(1.1 P_hat)) (option 1(a));
swings compared over ALL consecutive pairs with <=; a Boyd lapse resets k0 strictly; the monotone branch uses
half-window means plus the stationary sub-case; P_MAX is a spec constant computed from the SRP1 records; "successive
swings not growing" needs two half-swings, so certification effectively requires THREE turning points (stricter than
the ruling's "at least two sign changes"). The literal two-sign-change reading is computed REPORT-ONLY
(`literal_two_sign_change_report`) and never decides.

Numbers: dQ is a float difference of two floats (as production's own objective change); ranges are max - min over
the window's Q values; every comparison is exactly as written above (<=, <, >=).
"""

import math

__all__ = ['DELTA_R', 'R_REF', 'TAU', 'EPS0', 'K_EXCL', 'W_MIN', 'W_FACTOR', 'CAP_AFTER_N', 'FAILING_REASONS',
           'READINGS', 'constants', 'sign_of_step', 'turning_points', 'p_max_from_records', 'window_length',
           'SettlingRule', 'replay', 'literal_two_sign_change_report']

DELTA_R = 0.07
R_REF = 259375.33
TAU = DELTA_R * R_REF / 4.0
EPS0 = TAU / 100.0
K_EXCL = 3
W_MIN = 20
W_FACTOR = 1.1
CAP_AFTER_N = 100
FAILING_REASONS = ('insufficient_turning_points', 'swings_growing', 'range_above_tau', 'window_outside_run',
                   'monotone_window_not_reached', 'monotone_not_decreasing')
READINGS = {
    'window': 'W = max(W_MIN, ceil(W_FACTOR * P_hat)) = max(20, ceil(1.1 P_hat)), per option 1(a)',
    'swings': 'successive half-swings compared over ALL consecutive pairs of A since k0, with <= (A[i+1] <= A[i])',
    'boyd_lapse': 'a Boyd lapse (or a failed local solve) resets k0 strictly: T, A and the sign state are cleared',
    'monotone_branch': ('half-window means of |dQ| over [lo .. k] (second half strictly below the first), plus the '
                        'stationary sub-case max |dQ| < EPS0'),
    'p_max': 'P_MAX is a spec constant computed from the SRP1 reference records (turning_points, cycles 1..N)',
    'three_turning_points': ('"successive swings not growing" needs two half-swings, so certification effectively '
                             'requires three turning points -- stricter than the ruling\'s "at least two sign '
                             'changes"; the literal two-sign-change reading is computed report-only and never decides'),
}


def constants(p_max):
    """The constants with their formulas (for the frozen spec). `p_max` is the computed spec constant."""
    return {
        'DELTA_R': {'value': DELTA_R, 'formula': 'Addendum 53 Ruling 1: dR = 0.07'},
        'R_REF': {'value': R_REF, 'formula': 'V_SRP1 = R_ref = 259,375.33 (tight-tail SRP1 value, Addendum 48)'},
        'TAU': {'value': TAU, 'formula': 'TAU = DELTA_R * R_REF / 4 = 0.07 * 259375.33 / 4'},
        'EPS0': {'value': EPS0, 'formula': 'EPS0 = TAU / 100 (a step that cannot accumulate TAU within the cap)'},
        'K_EXCL': {'value': K_EXCL, 'formula': 'K_EXCL = 3'},
        'W_MIN': {'value': W_MIN, 'formula': 'W_MIN = 20'},
        'W_FACTOR': {'value': W_FACTOR, 'formula': 'W_FACTOR = 1.1'},
        'P_MAX': {'value': p_max, 'formula': ('longest interval between same-kind turning points, '
                                              'settling_criterion.turning_points over cycles 1..N of the three SRP1 '
                                              'reference records (settling_criterion.p_max_from_records)')},
        'L_MONO': {'value': 2 * p_max, 'formula': 'L_MONO = 2 * P_MAX'},
        'CAP': {'value': 'N + 100 per cell', 'formula': 'CAP = N + CAP_AFTER_N, CAP_AFTER_N = 100'},
    }


def sign_of_step(dq, eps0=EPS0):
    """Step 4: 0 if |dQ| < EPS0, else the sign of dQ."""
    if abs(dq) < eps0:
        return 0
    return 1 if dq > 0 else -1


def window_length(p_hat):
    """W = max(W_MIN, ceil(W_FACTOR * P_hat)); W_MIN when there is no P_hat."""
    if p_hat is None:
        return W_MIN
    return max(W_MIN, int(math.ceil(W_FACTOR * p_hat)))


def _extremum(q, lo, hi, kind):
    """argmax (kind 'max') / argmin (kind 'min') of q over cycles lo..hi inclusive; ties -> the earliest cycle."""
    best = None
    for c in range(lo, hi + 1):
        v = q[c]
        if best is None or (v > q[best] if kind == 'max' else v < q[best]):
            best = c
    return best


class _SignState:
    """Step 5's sign state over a sequence (shared by the rule and by `turning_points`)."""

    def __init__(self):
        self.sign_prev = None
        self.j_prev = None
        self.j_change = None
        self.T = []     # [(t, kind, Q_t)]
        self.A = []     # half-swings

    def update(self, q, k, s):
        """Returns (sign_change: bool, turning_point: (t, kind, Q_t) or None)."""
        if s == 0:
            return False, None
        if self.sign_prev is None:
            self.sign_prev, self.j_prev = s, k
            return False, None
        if s == self.sign_prev:
            self.j_prev = k
            return False, None
        kind = 'max' if self.sign_prev == 1 else 'min'
        t = _extremum(q, self.j_prev, k - 1, kind)
        tp = (t, kind, q[t])
        self.T.append(tp)
        if len(self.T) >= 2:
            self.A.append(abs(q[t] - q[self.T[-2][0]]))
        self.sign_prev, self.j_prev, self.j_change = s, k, k
        return True, tp


def turning_points(q_by_cycle, first=None, last=None, eps0=EPS0):
    """The turning points of a Q sequence under step 4's sign definition and step 5's update (same EPS0 floor), over
    cycles first..last (default: the sequence's own range). A step is formed only between two consecutive cycles that
    both carry a Q (a None Q -- a failed cycle -- makes the steps into and out of it unavailable: no update).
    Returns {'T': [(t, kind, Q_t)], 'A': [...], 'sign_changes': [k, ...]}."""
    cycles = sorted(c for c, v in q_by_cycle.items() if v is not None)
    if not cycles:
        return {'T': [], 'A': [], 'sign_changes': []}
    first = cycles[0] if first is None else first
    last = cycles[-1] if last is None else last
    st = _SignState()
    changes = []
    for k in range(first + 1, last + 1):
        if q_by_cycle.get(k) is None or q_by_cycle.get(k - 1) is None:
            continue
        changed, _tp = st.update(q_by_cycle, k, sign_of_step(q_by_cycle[k] - q_by_cycle[k - 1], eps0))
        if changed:
            changes.append(k)
    return {'T': list(st.T), 'A': list(st.A), 'sign_changes': changes}


def p_max_from_records(records):
    """P_MAX: the longest interval between same-kind turning points (T[i+2].t - T[i].t: consecutive turning points
    alternate in kind) found by `turning_points` over cycles 1..N of each record. `records` = {name: (q_by_cycle, N)}.
    Returns (p_max, detail)."""
    detail = {}
    best = None
    for name in sorted(records):
        q, n = records[name]
        tp = turning_points(q, first=1, last=n)
        intervals = [(tp['T'][i + 2][0] - tp['T'][i][0], tp['T'][i][0], tp['T'][i + 2][0], tp['T'][i][1])
                     for i in range(len(tp['T']) - 2)]
        longest = max(intervals) if intervals else None
        detail[name] = {'N': n, 'n_turning_points': len(tp['T']),
                        'turning_points': [[t, kind, qt] for t, kind, qt in tp['T']],
                        'same_kind_intervals': [{'interval': a, 'from': b, 'to': c, 'kind': d}
                                                for a, b, c, d in intervals],
                        'longest': (None if longest is None else
                                    {'interval': longest[0], 'from': longest[1], 'to': longest[2], 'kind': longest[3]})}
        if longest is not None and (best is None or longest[0] > best):
            best = longest[0]
    return best, detail


class SettlingRule:
    """Algorithm steps 1-10 over k = 1..cap. `n` = the cell's certification cycle N (no decision for k <= n);
    `cap` = N + 100 for a continuation cell; `p_max` = the spec constant. `observe(k, q, boyd)` returns the cycle's
    full record (the fields the continuation line carries) and, on certification or at the cap, `self.decision`."""

    def __init__(self, n, cap, p_max):
        if not (isinstance(n, int) and isinstance(cap, int) and isinstance(p_max, int)) or p_max < 1 or cap < n:
            raise ValueError(f'SettlingRule(n={n!r}, cap={cap!r}, p_max={p_max!r})')
        self.n, self.cap, self.p_max, self.l_mono = n, cap, p_max, 2 * p_max
        self.q = {}
        self.k0 = None
        self.sign = _SignState()
        self.lapses = []
        self.last_k = None
        self.decision = None

    # ---- window helpers (all over in-run cycles, whose Q is never None) -----------------------------------------
    def _range(self, lo, hi):
        vals = [self.q[c] for c in range(lo, hi + 1)]
        return max(vals) - min(vals), min(vals), max(vals)

    def _dq(self, c):
        return self.q[c] - self.q[c - 1]

    def _reset(self):
        self.sign = _SignState()

    # ---- the clauses (evaluated at any k >= k0 + K_EXCL; the decision uses them only for k > N) -----------------
    def evaluate(self, k):
        s = self.sign
        T, A = s.T, s.A
        a_parts = {'n_turning_points': len(T), 'at_least_3_turning_points': len(T) >= 3,
                   'swings_non_increasing_all_pairs': all(A[i + 1] <= A[i] for i in range(len(A) - 1)),
                   'P_hat': None, 'W': None, 'window': None, 'window_inside_run': None, 'range': None,
                   'range_le_tau': None, 'band': None}
        if len(T) >= 3:
            p_hat = T[-1][0] - T[-3][0]
            w = window_length(p_hat)
            lo = k - w + 1
            a_parts.update({'P_hat': p_hat, 'W': w, 'window': [lo, k], 'window_inside_run': lo >= self.k0})
            if lo >= self.k0:
                rng, mn, mx = self._range(lo, k)
                a_parts.update({'range': rng, 'range_le_tau': rng <= TAU, 'band': [mn, mx]})
        cert_a = bool(a_parts['at_least_3_turning_points'] and a_parts['swings_non_increasing_all_pairs']
                      and a_parts['window_inside_run'] and a_parts['range_le_tau'])
        lo = k - self.l_mono + 1
        b_parts = {'L_MONO': self.l_mono, 'window': [lo, k], 'lo_ge_k0_plus_K_EXCL': lo >= self.k0 + K_EXCL,
                   'j_change': s.j_change, 'no_sign_change_in_window': s.j_change is None or s.j_change < lo,
                   'range': None, 'range_le_tau': None, 'max_abs_dq': None, 'stationary': None,
                   'mean_abs_dq_first_half': None, 'mean_abs_dq_second_half': None, 'strictly_decreasing': None,
                   'band': None}
        if b_parts['lo_ge_k0_plus_K_EXCL']:
            rng, mn, mx = self._range(lo, k)
            steps = [abs(self._dq(c)) for c in range(lo, k + 1)]
            half = len(steps) // 2
            m1 = sum(steps[:half]) / half
            m2 = sum(steps[half:]) / (len(steps) - half)
            b_parts.update({'range': rng, 'range_le_tau': rng <= TAU, 'band': [mn, mx], 'max_abs_dq': max(steps),
                            'stationary': max(steps) < EPS0, 'mean_abs_dq_first_half': m1,
                            'mean_abs_dq_second_half': m2, 'strictly_decreasing': m2 < m1})
        cert_b = (not cert_a) and bool(b_parts['lo_ge_k0_plus_K_EXCL'] and b_parts['no_sign_change_in_window']
                                       and b_parts['range_le_tau']
                                       and (b_parts['stationary'] or b_parts['strictly_decreasing']))
        reasons = []
        if not cert_a and not cert_b:
            if not a_parts['at_least_3_turning_points']:
                reasons.append('insufficient_turning_points')
            if not a_parts['swings_non_increasing_all_pairs']:
                reasons.append('swings_growing')
            if a_parts['window_inside_run'] is False:
                reasons.append('window_outside_run')
            if a_parts['range_le_tau'] is False or b_parts['range_le_tau'] is False:
                reasons.append('range_above_tau')
            if not (b_parts['lo_ge_k0_plus_K_EXCL'] and b_parts['no_sign_change_in_window']):
                reasons.append('monotone_window_not_reached')
            elif not (b_parts['stationary'] or b_parts['strictly_decreasing']):
                reasons.append('monotone_not_decreasing')
        reasons = [r for r in FAILING_REASONS if r in reasons]
        return cert_a, cert_b, a_parts, b_parts, reasons

    # ---- one cycle ------------------------------------------------------------------------------------------------
    def observe(self, k, q, boyd):
        if self.decision is not None and self.decision.get('status') == 'certified':
            raise RuntimeError(f'SettlingRule: cycle {k} observed after certification at {self.decision["k_star"]}')
        if self.last_k is not None and k != self.last_k + 1:
            raise RuntimeError(f'SettlingRule: non-consecutive cycle {k} after {self.last_k}')
        if self.last_k is None and k != 1:
            raise RuntimeError(f'SettlingRule: the first cycle must be 1, got {k}')
        if k > self.cap:
            raise RuntimeError(f'SettlingRule: cycle {k} beyond the cap {self.cap}')
        self.last_k = k
        boyd = bool(boyd)
        rec = {'k': k, 'Q': q, 'boyd_k': boyd, 'phase': 'replay' if k <= self.n else 'decision', 'lapse': False,
               'k0': None, 'eligible': False, 'dQ': None, 's_k': None, 'sign_change': False, 'turning_point': None,
               'len_T': None, 'T': None, 'A': None, 'P_hat': None, 'W': None, 'window': None, 'range': None,
               'range_over_tau': None, 'certA': None, 'certA_parts': None, 'certB': None, 'certB_parts': None,
               'decision': None, 'reasons': None, 'lapse_events': len(self.lapses)}
        if q is not None:
            self.q[k] = q
        # step 1
        if q is None or not boyd:
            if self.k0 is not None:
                self.lapses.append({'cycle': k, 'k0_before': self.k0, 'q_is_none': q is None, 'boyd_k': boyd})
                rec['lapse'] = True
                rec['lapse_events'] = len(self.lapses)
            self.k0 = None
            self._reset()
            return self._finish_cap(k, rec)
        # step 2
        if self.k0 is None:
            self.k0 = k
            self._reset()
        rec['k0'] = self.k0
        # step 3
        if k < self.k0 + K_EXCL:
            return self._finish_cap(k, rec)
        rec['eligible'] = True
        # step 4
        dq = self._dq(k)
        s_k = sign_of_step(dq)
        rec.update({'dQ': dq, 's_k': s_k})
        # step 5
        changed, tp = self.sign.update(self.q, k, s_k)
        rec.update({'sign_change': changed, 'turning_point': list(tp) if tp else None, 'len_T': len(self.sign.T),
                    'T': [list(t) for t in self.sign.T], 'A': list(self.sign.A), 'j_change': self.sign.j_change})
        cert_a, cert_b, a_parts, b_parts, reasons = self.evaluate(k)
        win = a_parts['window'] if a_parts['window'] is not None else None
        rng = a_parts['range']
        rec.update({'P_hat': a_parts['P_hat'], 'W': a_parts['W'], 'window': win, 'range': rng,
                    'range_over_tau': (rng / TAU) if rng is not None else None,
                    'certA': cert_a, 'certA_parts': a_parts, 'certB': cert_b, 'certB_parts': b_parts,
                    'reasons': reasons})
        # step 6
        if k <= self.n:
            rec['decision'] = 'replay_no_decision'
            return self._finish_cap(k, rec)
        # steps 7-9
        if cert_a or cert_b:
            branch = 'oscillatory' if cert_a else 'monotone'
            parts = a_parts if cert_a else b_parts
            band = parts['band']
            self.decision = {'status': 'certified', 'k_star': k, 'branch': branch, 'Q_k_star': q, 'k0': self.k0,
                             'T': [list(t) for t in self.sign.T], 'A': list(self.sign.A), 'P_hat': a_parts['P_hat'],
                             'W': a_parts['W'] if cert_a else self.l_mono, 'window': parts['window'],
                             'band': band, 'band_width': band[1] - band[0], 'range': parts['range'],
                             'range_over_tau': parts['range'] / TAU, 'lapse_events': list(self.lapses),
                             'certA_parts': a_parts, 'certB_parts': b_parts}
            rec['decision'] = f'certified_{branch}'
            return rec
        rec['decision'] = 'continue'
        return self._finish_cap(k, rec)

    def _finish_cap(self, k, rec):
        if k == self.cap and self.decision is None:
            p_hat = None
            if len(self.sign.T) >= 3:
                p_hat = self.sign.T[-1][0] - self.sign.T[-3][0]
            w = window_length(p_hat)
            vals = [self.q[c] for c in range(k - w + 1, k + 1) if self.q.get(c) is not None]
            band = [min(vals), max(vals)] if vals else None
            reasons = rec.get('reasons')
            if reasons is None:
                reasons = ['insufficient_turning_points', 'monotone_window_not_reached']  # no eligible state at cap
            self.decision = {'status': 'uncertified', 'k_cap': k, 'reasons': list(reasons), 'k0': self.k0,
                             'T': [list(t) for t in self.sign.T], 'A': list(self.sign.A), 'P_hat': p_hat,
                             'band_window_length': w, 'band_window': [k - w + 1, k], 'band': band,
                             'band_width': (band[1] - band[0]) if band else None,
                             'lapse_events': list(self.lapses)}
            rec['decision'] = 'uncertified_at_cap'
        return rec


def replay(q_by_cycle, boyd_by_cycle, n, cap, p_max):
    """Runs the rule over cycles 1..cap (the records must hold every cycle 1..cap). Returns (per-cycle records,
    decision or None)."""
    rule = SettlingRule(n, cap, p_max)
    out = []
    for k in range(1, cap + 1):
        out.append(rule.observe(k, q_by_cycle[k], boyd_by_cycle[k]))
        if rule.decision is not None and rule.decision.get('status') == 'certified':
            break
    return out, rule.decision, rule


def literal_two_sign_change_report(rule, k):
    """REPORT-ONLY (never decides): the ruling's literal reading ("at least two sign changes"). With exactly two turning
    points (two sign changes): P_hat = 2 x their spacing; with three or more, P_hat = T[-1].t - T[-3].t as in step 7;
    the other clauses unchanged (swings over all consecutive pairs, vacuous with one half-swing; window inside the
    run; range <= TAU)."""
    T, A = rule.sign.T, rule.sign.A
    if rule.k0 is None or len(T) < 2:
        return {'literal_certifies': False, 'n_sign_changes': len(T), 'reason': 'fewer than two sign changes'}
    p_hat = (T[-1][0] - T[-3][0]) if len(T) >= 3 else 2 * (T[-1][0] - T[-2][0])
    w = window_length(p_hat)
    lo = k - w + 1
    inside = lo >= rule.k0
    swings = all(A[i + 1] <= A[i] for i in range(len(A) - 1))
    rng = None
    if inside:
        vals = [rule.q[c] for c in range(lo, k + 1)]
        rng = max(vals) - min(vals)
    ok = bool(swings and inside and rng is not None and rng <= TAU)
    return {'literal_certifies': ok, 'n_sign_changes': len(T), 'P_hat': p_hat, 'W': w, 'window': [lo, k],
            'window_inside_run': inside, 'swings_non_increasing_all_pairs': swings, 'range': rng,
            'range_le_tau': (rng <= TAU) if rng is not None else None}
