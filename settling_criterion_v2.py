"""The settling stop rule, VERSION 2 (P5.15 Addendum 57 Decisions 2 and 3; Addendum 54 Ruling 2; Planner task W118).

A NEW module: `settling_criterion.py` (version 1, pinned by the frozen stage specs v39 / v41 and by every certificate
issued under them) is NOT edited and stays byte-identical; the certificates issued under version 1 are untouched. This
module imports version 1 for the pieces that are unchanged (the sign definition, the turning-point state, the
oscillatory window) and re-implements only the decision.

A PURE function of three per-cycle sequences: Q_k (`gross_operational_cost`, None when any local solve failed), boyd_k
(`all_boyd_pass` AND `local_solves_ok`, from THIS cycle's `boyd_metrics`) and t_sum_k (the priced interface-consensus
gap of cycle k, EUR; None when unavailable). Stdlib only; no I/O; no model code.

WHAT CHANGED FROM VERSION 1 (and nothing else):
  (1) N := the run's FIRST k0 (its first residual pass). Version 1 took N as an argument (the old certification cycle)
      and took no decision for k <= N; here there is no "no decision before N_old": decisions are possible from
      k0 + K_EXCL (step 3, unchanged). With N = the first k0, version 1's step 6 is vacuous (k >= k0 + K_EXCL > N).
  (2) CAP: either FIXED (a gated re-run cell: N_old + 100) or DYNAMIC (an ungated cell: min(first k0 + CAP_AFTER_K0,
      CAP_CEILING), CAP_AFTER_K0 = 109, CAP_CEILING = production's cycle cap 300), fixed at the first k0.
  (3) THE MONOTONE BRANCH, AMENDED (Addendum 57 Decision 2): window length L = L_MONO = 2 x P_MAX with P_MAX = 30 (an
      instance-measured spec constant, passed in: the post-certification periods on the SRP1 references, 29 and 30);
      the existing clauses kept (no sign change in the window; half-window means of |dQ| strictly decreasing, or the
      stationary sub-case max |dQ| < EPS0) AND the new clause |dQ_k| x L <= TAU (a linear remaining-descent bound, no
      fit). The window-range <= TAU clause of version 1 is KEPT as well (the stricter reading; READINGS).
  (4) THE GAP CLAUSE (Addendum 57 Decision 2): a cycle at which a branch (oscillatory or monotone) would certify is
      certified ONLY IF |t_sum_k| <= GAP_BOUND = TAU / 2; otherwise the refusal is recorded (cycle, branch, t_sum) and
      the rule continues. A missing t_sum (None) cannot satisfy the clause.
  (5) UNCERTIFIED AT CAP: the frozen reporting form -- band [min, max] Q over the last L cycles; drift rate = mean dQ
      over the last DRIFT_WINDOW = 25 cycles; dQ_cc rate = mean (dQ + dt_sum) over the same cycles; t_sum and Q_cc at
      the cap; the failing reasons (including 'gap_clause' when a branch held at the cap but the gap did not).

CONSTANTS (each with its formula; `constants()` returns them with the formulas for the spec):
  TAU        = DELTA_R * R_REF / 4 = 4,539.068275 EUR          (version 1, unchanged)
  EPS0       = TAU / 100                                        (version 1, unchanged)
  K_EXCL     = 3;  W_MIN = 20;  W_FACTOR = 1.1                  (version 1, unchanged)
  GAP_BOUND  = TAU / 2 = 2,269.5341375 EUR                      (Addendum 57 Decision 2)
  P_MAX      = a SPEC constant, passed in (30 for the W118 campaign; instance-measured)
  L_MONO     = 2 * P_MAX (60)                                   (Addendum 57 Decision 2: "L = 60")
  DRIFT_WINDOW = 25                                             (Planner task W118: "mean dQ over the last 25")
  CAP        = fixed, or min(first k0 + 109, 300)               (Planner task W118)

THE ALGORITHM, for k = 1..CAP after Q_k, boyd_k and t_sum_k are known (`SettlingRuleV2.observe`):
  1. Q_k None or not boyd_k: record a lapse event if k0 was set; k0 = None; reset the sign state; continue.
  2. k0 None: k0 = k; reset the state; if this is the run's first k0: N = k0 and (dynamic mode) CAP is fixed now.
  3. k < k0 + K_EXCL: continue.
  4. dQ_k = Q_k - Q_(k-1); s_k = 0 if |dQ_k| < EPS0 else sign(dQ_k).                        (version 1)
  5. the sign-state update (turning points T, half-swings A, j_change).                    (version 1)
  6. certA (oscillatory): len(T) >= 3; A non-increasing over ALL consecutive pairs; W = max(W_MIN, ceil(W_FACTOR
     P_hat)), P_hat = T[-1].t - T[-3].t; k - W + 1 >= k0; range(Q over [k-W+1 .. k]) <= TAU.  (version 1)
  7. certB (monotone), only if not certA; lo = k - L_MONO + 1: lo >= k0 + K_EXCL; j_change None or < lo;
     range(Q over [lo .. k]) <= TAU; (max |dQ| over [lo .. k] < EPS0 OR mean |dQ| second half < first half);
     AND |dQ_k| * L_MONO <= TAU.                                                             (AMENDED)
  8. certA or certB: if |t_sum_k| <= GAP_BOUND certify at k* = k (branch oscillatory / monotone); else record the
     gap refusal and continue.                                                               (NEW)
  9. k == CAP (not certified): the uncertified record (the frozen reporting form above).

READINGS (recorded in the spec): as version 1 (window option 1(a), all-pairs swings, strict k0 reset on a lapse,
half-window means + stationary, three turning points), plus: the monotone window range <= TAU kept beside the new
|dQ_k| L <= TAU clause (stricter reading); the gap clause checked at the certifying cycle only; t_sum None fails it.

Numbers: dQ is a float difference of two floats; ranges are max - min over the window's Q values; every comparison is
exactly as written above (<=, <, >=).
"""

import settling_criterion as SC1

__all__ = ['VERSION', 'TAU', 'EPS0', 'K_EXCL', 'W_MIN', 'W_FACTOR', 'GAP_BOUND', 'DRIFT_WINDOW', 'CAP_AFTER_K0',
           'CAP_CEILING', 'FAILING_REASONS', 'READINGS', 'constants', 'SettlingRuleV2', 'replay']

VERSION = 2
TAU = SC1.TAU
EPS0 = SC1.EPS0
K_EXCL = SC1.K_EXCL
W_MIN = SC1.W_MIN
W_FACTOR = SC1.W_FACTOR
GAP_BOUND = TAU / 2.0
DRIFT_WINDOW = 25
CAP_AFTER_K0 = 109          # ungated cells: N_old + 100 = k0 + 109 on every gated cell (k0 = N_old - 9)
CAP_CEILING = 300           # production's cycle cap (data/SRP1/SRP1_params.json admm num_max_iters)
FAILING_REASONS = ('no_residual_pass', 'insufficient_turning_points', 'swings_growing', 'range_above_tau',
                   'window_outside_run', 'monotone_window_not_reached', 'monotone_not_decreasing',
                   'monotone_last_step_times_L_above_tau', 'gap_clause')
READINGS = dict(SC1.READINGS)
READINGS.update({
    'n_is_first_k0': ('N := the run\'s first residual pass k0; decisions from k0 + K_EXCL; no "no decision before '
                      'N_old" (Addendum 57 / Planner task W118)'),
    'monotone_amended': ('Addendum 57 Decision 2: steps decreasing over the window AND |last step| x L <= TAU, L = '
                         '2 x P_MAX = 60; "steps decreasing" = version 1\'s half-window-means clause or its stationary '
                         'sub-case; no sign change in the window (version 1)'),
    'monotone_range_kept': ('the window range <= TAU clause of version 1 is KEPT beside |dQ_k| L <= TAU: the stricter '
                            'reading (both must hold)'),
    'gap_clause': ('|t_sum_k| <= TAU / 2 checked at the cycle a branch would certify; a failure is recorded and the '
                   'rule continues; t_sum None fails'),
    'uncertified_form': ('band over the last L_MONO cycles; drift = mean dQ over the last 25; dQ_cc rate = mean (dQ + '
                         'dt_sum) over the last 25; t_sum and Q_cc at the cap; reasons'),
})


def constants(p_max):
    """The constants with their formulas (for the frozen spec). `p_max` is the spec constant."""
    out = SC1.constants(p_max)
    out.pop('CAP', None)
    out.update({
        'VERSION': {'value': VERSION, 'formula': 'settling_criterion_v2 (settling_criterion.py = version 1, unchanged)'},
        'GAP_BOUND': {'value': GAP_BOUND, 'formula': 'GAP_BOUND = TAU / 2 (Addendum 57 Decision 2: |t_sum| <= tau/2)'},
        'L_MONO': {'value': 2 * p_max, 'formula': 'L_MONO = 2 * P_MAX (Addendum 57 Decision 2: L = 60)'},
        'MONOTONE_LAST_STEP_CLAUSE': {'value': 'abs(dQ_k) * L_MONO <= TAU',
                                      'formula': 'Addendum 57 Decision 2: |last step| x L <= tau'},
        'DRIFT_WINDOW': {'value': DRIFT_WINDOW, 'formula': 'mean dQ over the last 25 cycles (uncertified form)'},
        'CAP': {'value': 'fixed N_old + 100 (gated) | min(first k0 + 109, 300) (ungated)',
                'formula': 'Planner task W118; CAP_AFTER_K0 = 109, CAP_CEILING = 300'},
        'P_MAX': {'value': p_max, 'formula': ('instance-measured spec constant (W118: the post-certification periods of '
                                              'the settled SRP1 references, P_hat = 29 (x0, W102) and 30 (unit, W103))')},
    })
    return out


class SettlingRuleV2:
    """Steps 1-9 over k = 1..CAP. Exactly one of `cap` (fixed) or `cap_after_first_k0` (dynamic, with `cap_ceiling`)
    is given; `p_max` = the spec constant. `observe(k, q, boyd, t_sum)` returns the cycle's full record and, on
    certification or at the cap, sets `self.decision`."""

    def __init__(self, p_max, cap=None, cap_after_first_k0=None, cap_ceiling=CAP_CEILING):
        if not (isinstance(p_max, int) and not isinstance(p_max, bool) and p_max >= 1):
            raise ValueError(f'SettlingRuleV2: p_max {p_max!r}')
        if (cap is None) == (cap_after_first_k0 is None):
            raise ValueError('SettlingRuleV2: give exactly one of cap (fixed) and cap_after_first_k0 (dynamic)')
        for v in (cap, cap_after_first_k0, cap_ceiling):
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
                raise ValueError(f'SettlingRuleV2: cap arguments must be positive ints; got {v!r}')
        if cap is not None and cap > cap_ceiling:
            raise ValueError(f'SettlingRuleV2: cap {cap} above the ceiling {cap_ceiling}')
        self.p_max, self.l_mono = p_max, 2 * p_max
        self.cap_mode = 'fixed' if cap is not None else 'dynamic'
        self.cap_fixed, self.cap_after, self.cap_ceiling = cap, cap_after_first_k0, cap_ceiling
        self.cap = cap                       # dynamic: set at the first k0
        self.n = None                        # := the first k0
        self.q = {}
        self.t = {}
        self.k0 = None
        self.sign = SC1._SignState()
        self.lapses = []
        self.gap_refusals = []
        self.last_k = None
        self.decision = None

    # ---- helpers ------------------------------------------------------------------------------------------------------
    def effective_cap(self):
        """The cap in force: fixed; dynamic after the first k0; the ceiling before it."""
        return self.cap if self.cap is not None else self.cap_ceiling

    def _range(self, lo, hi):
        vals = [self.q[c] for c in range(lo, hi + 1)]
        return max(vals) - min(vals), min(vals), max(vals)

    def _dq(self, c):
        return self.q[c] - self.q[c - 1]

    def evaluate(self, k):
        """The branch clauses at k (k >= k0 + K_EXCL). Returns (cert_a, cert_b, a_parts, b_parts, branch_reasons)."""
        s = self.sign
        T, A = s.T, s.A
        a_parts = {'n_turning_points': len(T), 'at_least_3_turning_points': len(T) >= 3,
                   'swings_non_increasing_all_pairs': all(A[i + 1] <= A[i] for i in range(len(A) - 1)),
                   'P_hat': None, 'W': None, 'window': None, 'window_inside_run': None, 'range': None,
                   'range_le_tau': None, 'band': None}
        if len(T) >= 3:
            p_hat = T[-1][0] - T[-3][0]
            w = SC1.window_length(p_hat)
            lo = k - w + 1
            a_parts.update({'P_hat': p_hat, 'W': w, 'window': [lo, k], 'window_inside_run': lo >= self.k0})
            if lo >= self.k0:
                rng, mn, mx = self._range(lo, k)
                a_parts.update({'range': rng, 'range_le_tau': rng <= TAU, 'band': [mn, mx]})
        cert_a = bool(a_parts['at_least_3_turning_points'] and a_parts['swings_non_increasing_all_pairs']
                      and a_parts['window_inside_run'] and a_parts['range_le_tau'])
        lo = k - self.l_mono + 1
        last = abs(self._dq(k))
        b_parts = {'L_MONO': self.l_mono, 'window': [lo, k], 'lo_ge_k0_plus_K_EXCL': lo >= self.k0 + K_EXCL,
                   'j_change': s.j_change, 'no_sign_change_in_window': s.j_change is None or s.j_change < lo,
                   'range': None, 'range_le_tau': None, 'max_abs_dq': None, 'stationary': None,
                   'mean_abs_dq_first_half': None, 'mean_abs_dq_second_half': None, 'strictly_decreasing': None,
                   'steps_decreasing': None, 'last_step_abs': last, 'last_step_times_L': last * self.l_mono,
                   'last_step_times_L_le_tau': last * self.l_mono <= TAU, 'band': None}
        if b_parts['lo_ge_k0_plus_K_EXCL']:
            rng, mn, mx = self._range(lo, k)
            steps = [abs(self._dq(c)) for c in range(lo, k + 1)]
            half = len(steps) // 2
            m1 = sum(steps[:half]) / half
            m2 = sum(steps[half:]) / (len(steps) - half)
            b_parts.update({'range': rng, 'range_le_tau': rng <= TAU, 'band': [mn, mx], 'max_abs_dq': max(steps),
                            'stationary': max(steps) < EPS0, 'mean_abs_dq_first_half': m1,
                            'mean_abs_dq_second_half': m2, 'strictly_decreasing': m2 < m1,
                            'steps_decreasing': bool(max(steps) < EPS0 or m2 < m1)})
        cert_b = (not cert_a) and bool(b_parts['lo_ge_k0_plus_K_EXCL'] and b_parts['no_sign_change_in_window']
                                       and b_parts['range_le_tau'] and b_parts['steps_decreasing']
                                       and b_parts['last_step_times_L_le_tau'])
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
            else:
                if not b_parts['steps_decreasing']:
                    reasons.append('monotone_not_decreasing')
                if not b_parts['last_step_times_L_le_tau']:
                    reasons.append('monotone_last_step_times_L_above_tau')
        return cert_a, cert_b, a_parts, b_parts, reasons

    # ---- one cycle ----------------------------------------------------------------------------------------------------
    def observe(self, k, q, boyd, t_sum):
        if self.decision is not None:
            raise RuntimeError(f'SettlingRuleV2: cycle {k} observed after the decision ({self.decision.get("status")})')
        if self.last_k is not None and k != self.last_k + 1:
            raise RuntimeError(f'SettlingRuleV2: non-consecutive cycle {k} after {self.last_k}')
        if self.last_k is None and k != 1:
            raise RuntimeError(f'SettlingRuleV2: the first cycle must be 1, got {k}')
        if k > self.effective_cap():
            raise RuntimeError(f'SettlingRuleV2: cycle {k} beyond the cap {self.effective_cap()}')
        self.last_k = k
        boyd = bool(boyd)
        if q is not None:
            self.q[k] = q
        if t_sum is not None:
            self.t[k] = t_sum
        rec = {'k': k, 'Q': q, 'boyd_k': boyd, 't_sum': t_sum, 'Q_cc': (q + t_sum) if (q is not None and t_sum is not None)
               else None, 'lapse': False, 'k0': None, 'N': self.n, 'cap': self.cap, 'eligible': False, 'dQ': None,
               's_k': None, 'sign_change': False, 'turning_point': None, 'len_T': None, 'T': None, 'A': None,
               'P_hat': None, 'W': None, 'window': None, 'range': None, 'range_over_tau': None, 'certA': None,
               'certA_parts': None, 'certB': None, 'certB_parts': None, 'branch_would_certify': None,
               'gap_abs': abs(t_sum) if t_sum is not None else None, 'gap_ok': None, 'decision': None,
               'reasons': None, 'lapse_events': len(self.lapses), 'gap_refusals': len(self.gap_refusals)}
        # step 1
        if q is None or not boyd:
            if self.k0 is not None:
                self.lapses.append({'cycle': k, 'k0_before': self.k0, 'q_is_none': q is None, 'boyd_k': boyd})
                rec['lapse'] = True
                rec['lapse_events'] = len(self.lapses)
            self.k0 = None
            self.sign = SC1._SignState()
            return self._finish_cap(k, rec, None)
        # step 2
        if self.k0 is None:
            self.k0 = k
            self.sign = SC1._SignState()
            if self.n is None:
                self.n = k
                if self.cap_mode == 'dynamic':
                    self.cap = min(k + self.cap_after, self.cap_ceiling)
                rec.update({'N': self.n, 'cap': self.cap, 'first_residual_pass': True})
        rec['k0'] = self.k0
        # step 3
        if k < self.k0 + K_EXCL:
            return self._finish_cap(k, rec, None)
        rec['eligible'] = True
        # steps 4-5
        dq = self._dq(k)
        s_k = SC1.sign_of_step(dq)
        rec.update({'dQ': dq, 's_k': s_k})
        changed, tp = self.sign.update(self.q, k, s_k)
        rec.update({'sign_change': changed, 'turning_point': list(tp) if tp else None, 'len_T': len(self.sign.T),
                    'T': [list(x) for x in self.sign.T], 'A': list(self.sign.A), 'j_change': self.sign.j_change})
        # steps 6-7
        cert_a, cert_b, a_parts, b_parts, reasons = self.evaluate(k)
        rng = a_parts['range']
        rec.update({'P_hat': a_parts['P_hat'], 'W': a_parts['W'], 'window': a_parts['window'], 'range': rng,
                    'range_over_tau': (rng / TAU) if rng is not None else None, 'certA': cert_a,
                    'certA_parts': a_parts, 'certB': cert_b, 'certB_parts': b_parts, 'reasons': reasons})
        # step 8
        if cert_a or cert_b:
            branch = 'oscillatory' if cert_a else 'monotone'
            gap_ok = t_sum is not None and abs(t_sum) <= GAP_BOUND
            rec.update({'branch_would_certify': branch, 'gap_ok': gap_ok})
            if gap_ok:
                parts = a_parts if cert_a else b_parts
                band = parts['band']
                self.decision = {'status': 'certified', 'version': VERSION, 'k_star': k, 'branch': branch,
                                 'Q_k_star': q, 't_sum_k_star': t_sum, 'Q_cc_k_star': q + t_sum, 'k0': self.k0,
                                 'N': self.n, 'cap': self.cap, 'cap_mode': self.cap_mode,
                                 'T': [list(x) for x in self.sign.T], 'A': list(self.sign.A),
                                 'P_hat': a_parts['P_hat'], 'W': a_parts['W'] if cert_a else self.l_mono,
                                 'window': parts['window'], 'band': band, 'band_width': band[1] - band[0],
                                 'range': parts['range'], 'range_over_tau': parts['range'] / TAU,
                                 'gap_bound': GAP_BOUND, 'gap_refusals': list(self.gap_refusals),
                                 'lapse_events': list(self.lapses), 'certA_parts': a_parts, 'certB_parts': b_parts}
                rec['decision'] = f'certified_{branch}'
                return rec
            self.gap_refusals.append({'cycle': k, 'branch': branch, 't_sum': t_sum,
                                      'abs_t_sum': abs(t_sum) if t_sum is not None else None, 'gap_bound': GAP_BOUND})
            rec['gap_refusals'] = len(self.gap_refusals)
            rec['reasons'] = ['gap_clause']
            rec['decision'] = 'gap_clause_refused'
            return self._finish_cap(k, rec, branch)
        rec['decision'] = 'continue'
        return self._finish_cap(k, rec, None)

    def _mean_step(self, k, cc):
        vals = []
        for c in range(k - DRIFT_WINDOW + 1, k + 1):
            if c in self.q and (c - 1) in self.q:
                d = self.q[c] - self.q[c - 1]
                if cc:
                    if c in self.t and (c - 1) in self.t:
                        vals.append(d + (self.t[c] - self.t[c - 1]))
                else:
                    vals.append(d)
        return (sum(vals) / len(vals)) if vals else None, len(vals)

    def _finish_cap(self, k, rec, gap_refused_branch):
        if self.decision is not None or k != self.effective_cap():
            return rec
        lo = k - self.l_mono + 1
        vals = [self.q[c] for c in range(lo, k + 1) if self.q.get(c) is not None]
        band = [min(vals), max(vals)] if vals else None
        drift, n_drift = self._mean_step(k, cc=False)
        drift_cc, n_drift_cc = self._mean_step(k, cc=True)
        reasons = list(rec.get('reasons') or [])
        if self.n is None:
            reasons = ['no_residual_pass']
        elif rec.get('reasons') is None:
            reasons = ['insufficient_turning_points', 'monotone_window_not_reached']   # no eligible state at the cap
        if gap_refused_branch is not None and 'gap_clause' not in reasons:
            reasons.append('gap_clause')
        reasons = [r for r in FAILING_REASONS if r in reasons]
        t_cap = self.t.get(k)
        q_cap = self.q.get(k)
        self.decision = {'status': 'uncertified', 'version': VERSION, 'k_cap': k, 'reasons': reasons, 'k0': self.k0,
                         'N': self.n, 'cap': k, 'cap_mode': self.cap_mode,
                         'T': [list(x) for x in self.sign.T], 'A': list(self.sign.A),
                         'band_window_length': self.l_mono, 'band_window': [lo, k], 'band': band,
                         'band_width': (band[1] - band[0]) if band else None,
                         'drift_rate_mean_dQ_last_25': drift, 'drift_n_steps': n_drift,
                         'dQ_cc_rate_mean_last_25': drift_cc, 'dQ_cc_n_steps': n_drift_cc,
                         'Q_at_cap': q_cap, 't_sum_at_cap': t_cap,
                         'Q_cc_at_cap': (q_cap + t_cap) if (q_cap is not None and t_cap is not None) else None,
                         'gap_bound': GAP_BOUND, 'gap_refusals': list(self.gap_refusals),
                         'gap_clause_refused_at_cap': gap_refused_branch is not None,
                         'lapse_events': list(self.lapses)}
        rec['decision'] = 'uncertified_at_cap'
        return rec


def replay(q_by_cycle, boyd_by_cycle, t_by_cycle, p_max, cap=None, cap_after_first_k0=None, cap_ceiling=CAP_CEILING,
           last=None):
    """Runs the rule over cycles 1..last (default: the cap in force) from the records ({cycle: value}; a missing cycle
    reads as Q None / boyd False / t None). Stops at the decision. Returns (per-cycle records, decision, rule)."""
    rule = SettlingRuleV2(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)
    out = []
    k = 0
    while True:
        k += 1
        if last is not None and k > last:
            break
        if k > rule.effective_cap():
            break
        out.append(rule.observe(k, q_by_cycle.get(k), bool(boyd_by_cycle.get(k, False)), t_by_cycle.get(k)))
        if rule.decision is not None:
            break
    return out, rule.decision, rule
