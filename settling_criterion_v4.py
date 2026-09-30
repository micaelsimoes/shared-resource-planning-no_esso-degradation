"""The settling stop rule, VERSION 4 (P5.15 Addendum 59 and its Supplement: reading (gamma), the certifying window =
reading (a); Planner task W137).

A NEW module: `settling_criterion.py` (version 1), `settling_criterion_v2.py` (version 2) and `settling_criterion_v3.py`
(version 3) are NOT edited and stay byte-identical; every certificate issued under them is untouched. This module
imports version 2 (and through it version 1) and re-implements only what version 4 changes.

WHAT CHANGED FROM VERSION 2 (and nothing else):
  (1) NO RESET ON A NON-OPTIMAL CYCLE (Addendum 59: "No reset of the count"). The rule is version 2 fed the version-2
      boyd_k (`all_boyd_pass` AND `local_solves_ok`): only a Boyd lapse (or a failed cycle, Q None) resets k0 and the
      sign state. A non-Optimal accepted solve (all_optimal_k False: the final accepted attempt of some block of the 12
      TSO, 36 DSO and 3 ESSO solves did not exit "Optimal Solution Found"; the classification is the caller's) is NOT a
      lapse.
  (2) THE CERTIFICATION VETO (Addendum 59 Supplement, reading (a)). At a cycle k where a branch would certify, the
      CERTIFYING WINDOW is the last W cycles the certification test reads: the oscillatory branch [k - W + 1, k]
      (W = max(W_MIN, ceil(W_FACTOR * P_hat))), the monotone branch [k - L + 1, k] (L = 2 P_MAX). If any cycle of the
      window is non-Optimal the branch verdict is VETOED at that cycle only (reason 'certification_vetoed_non_optimal';
      the veto is recorded with the window and its non-Optimal cycles) and the rule state (k0, the sign state, the
      turning points) is UNCHANGED -- exactly as version 2 would continue. When the oscillatory verdict is vetoed the
      monotone conjunction is re-evaluated from version 2's own b_parts (version 2 evaluates it only when the
      oscillatory branch fails; the construction W131 validated and version 3's gamma shadow used), and is itself
      subject to the veto on its own window.
  (3) The gap clause (|t_sum_k| <= TAU / 2 at the certifying cycle), the residuals and the holds are UNCHANGED: the gap
      clause is evaluated only at a cycle whose verdict survives the veto.
  (4) THE SUB-TEST READS ARE ENUMERATED, NOT ASSERTED (Addendum 59 Supplement: "v4 states W and asserts no sub-test reads
      outside it"; Planner task W137: record the discrepancy instead). As implemented (version 2's clauses, unchanged),
      the oscillatory branch reads OUTSIDE [k - W + 1, k]: the all-pairs "swings not growing" test reads the half-swing
      amplitudes A of EVERY turning point since k0, P_hat reads the positions T[-1] and T[-3], and every turning point
      is located by the sign state over the steps since k0 + K_EXCL. The monotone branch reads Q_(lo - 1) (its first
      step) and the sign of the last non-zero step before lo (its "no sign change" test). `SUB_TEST_READS` states, per
      sub-test, which cycles it reads; every certification records its concrete out-of-window reads
      (`out_of_window_reads`: which turning points, which cycles, and whether any of those cycles is non-Optimal). The
      algorithm is NOT changed to avoid them (Planner task W137: the discrepancy goes to the expert).
  (5) N AND THE DYNAMIC CAP: version 2's (the first residual pass; with no non-Optimal reset this is the version-2 first
      pass, as version 3 keyed them). THE CAP CEILING IS PER CELL (a required argument, as version 3).

CONSTANTS: those of version 2 unchanged (TAU, EPS0, K_EXCL, W_MIN, W_FACTOR, GAP_BOUND, DRIFT_WINDOW, CAP_AFTER_K0);
CAP_CEILING per cell (spec); P_MAX a spec constant (passed in).

THE ALGORITHM, for k = 1..CAP after Q_k, boyd_k (version-2 definition), t_sum_k and all_optimal_k are known
(`SettlingRuleV4.observe`):
  1-7. version 2's steps 1-7 exactly (a lapse = Q_k None or not boyd_k; all_optimal_k plays no part in them).
  V.   a branch that would certify at k is vetoed if its window contains a non-Optimal cycle (oscillatory vetoed -> the
       monotone conjunction re-evaluated, then vetoed on its own window); the veto is recorded; the state is unchanged.
  8-9. version 2's steps 8-9 on the surviving verdict (the gap clause; the uncertified record at the cap).
Every per-cycle record is version 2's plus: all_optimal_k, vetoed (the veto of the cycle, or None) and, at a
certification, out_of_window_reads. The decision is version 2's with version 4, the reading, the window rule, N, the
cap ceiling, the non-Optimal cycles, every veto, and (certified) window_all_optimal and out_of_window_reads; an
uncertified decision whose cap cycle was vetoed carries 'certification_vetoed_non_optimal' among its reasons.

Stdlib only; no I/O; no model code. Numbers and comparisons exactly as version 2.
"""

import settling_criterion_v2 as SC2

__all__ = ['VERSION', 'SCHEMA', 'TAU', 'EPS0', 'K_EXCL', 'W_MIN', 'W_FACTOR', 'GAP_BOUND', 'DRIFT_WINDOW',
           'CAP_AFTER_K0', 'FAILING_REASONS', 'READINGS', 'OPTIMAL_CLASS', 'VETO_REASON', 'SUB_TEST_READS',
           'constants', 'SettlingRuleV4', 'replay']

VERSION = 4
SCHEMA = 'settling_criterion_v4'
TAU = SC2.TAU
EPS0 = SC2.EPS0
K_EXCL = SC2.K_EXCL
W_MIN = SC2.W_MIN
W_FACTOR = SC2.W_FACTOR
GAP_BOUND = SC2.GAP_BOUND
DRIFT_WINDOW = SC2.DRIFT_WINDOW
CAP_AFTER_K0 = SC2.CAP_AFTER_K0
OPTIMAL_CLASS = 'optimal'   # p515_s44_campaign_harness.ipopt_exit_class("... Optimal Solution Found ...")
VETO_REASON = 'certification_vetoed_non_optimal'
FAILING_REASONS = tuple(SC2.FAILING_REASONS) + (VETO_REASON,)
READING = 'gamma_window_a'
READINGS = dict(SC2.READINGS)
READINGS.update({
    'gamma_decision': ('Addendum 59: reading (gamma) -- no reset of the count on a non-Optimal cycle; only a Boyd lapse '
                       '(or a failed cycle) resets k0 and the sign state'),
    'window_reading_a': ('Addendum 59 Supplement: the certifying window is the last W cycles the certification test '
                         'reads at k -- oscillatory [k - W + 1, k], monotone [k - L + 1, k]; every cycle of it must be '
                         'all-Optimal on all 51 blocks, otherwise certification is refused at that cycle only '
                         '(certification_vetoed_non_optimal) and the state is unchanged'),
    'all_optimal_definition': ('all_optimal_k: the final accepted attempt of every block of cycle k (12 TSO, 36 DSO, '
                               '3 ESSO) exited "Optimal Solution Found" (ipopt_exit_class == "optimal"); a block with no '
                               'result or no exit message is not Optimal'),
    'oscillatory_veto_then_monotone': ('an oscillatory verdict vetoed -> the monotone conjunction re-evaluated from '
                                       'version 2\'s own b_parts (W131\'s construction; version 3\'s gamma shadow), '
                                       'then vetoed on its own window if it holds'),
    'gap_clause_after_veto': 'the gap clause is evaluated only on a verdict that survives the veto (unchanged clause)',
    'sub_test_reads_enumerated_not_asserted': ('Planner task W137: as implemented, the oscillatory branch reads outside '
                                               'the last W cycles (all-pairs swings since k0; P_hat from T[-1] and '
                                               'T[-3]; the turning points located over the steps since k0 + K_EXCL); '
                                               'the monotone branch reads Q_(lo-1) and the sign of the last non-zero '
                                               'step before lo. Enumerated in SUB_TEST_READS and recorded per '
                                               'certification (out_of_window_reads); the algorithm is not changed'),
    'no_retry_tier': 'no retry tier (the solve path is unchanged; Addendum 59)',
    'n_and_cap': ('N and the dynamic cap: version 2\'s first residual pass (Q_k not None and boyd_k), which with no '
                  'non-Optimal reset is the version-2 first pass (as version 3 keyed them)'),
    'cap_ceiling_per_cell': 'the cap ceiling is a per-cell spec value (as version 3), not a code constant',
})

# ---- the sub-test read enumeration (Addendum 59 Supplement; Planner task W137) -------------------------------------
# Symbols: k the candidate cycle; k0 the rule's current k0 (first residual pass after the last Boyd lapse); T the
# turning points since k0 (t, kind, Q_t), A the half-swings |Q_T[i+1] - Q_T[i]|; lo_osc = k - W + 1; lo_mono = k - L + 1.
SUB_TEST_READS = {
    'window': {
        'oscillatory': '[k - W + 1, k], W = max(W_MIN, ceil(W_FACTOR * P_hat)) (W_MIN 20, W_FACTOR 1.1)',
        'monotone': '[k - L + 1, k], L = L_MONO = 2 * P_MAX (60)',
        'veto': 'every cycle of the window of the branch that would certify must be all-Optimal on all 51 blocks',
    },
    'oscillatory': [
        {'sub_test': 'at_least_3_turning_points', 'reads': 'len(T): the turning points found since k0',
         'cycles': ('Q over [k0 + K_EXCL - 1, k]: every step dQ_c, c in [k0 + K_EXCL, k], enters the sign state; each '
                    'turning point t is the extremum of Q over [j_prev, j - 1] at the sign change j'),
         'inside_the_window_as_implemented': False},
        {'sub_test': 'swings_non_increasing_all_pairs', 'reads': 'A[i] for EVERY pair of consecutive turning points since k0',
         'cycles': 'Q_t at every turning point t in T (all of them since k0), plus their location (above)',
         'inside_the_window_as_implemented': False},
        {'sub_test': 'P_hat_and_W', 'reads': 'P_hat = T[-1].t - T[-3].t; W = max(W_MIN, ceil(W_FACTOR * P_hat))',
         'cycles': 'the positions T[-1].t and T[-3].t (T[-3].t may precede k - W + 1)',
         'inside_the_window_as_implemented': False},
        {'sub_test': 'window_inside_run', 'reads': 'k - W + 1 >= k0',
         'cycles': 'k0: the absence of a Boyd lapse (Q None or not boyd_k) over [k0, k]',
         'inside_the_window_as_implemented': False},
        {'sub_test': 'range_le_tau', 'reads': 'max - min of Q over the window', 'cycles': '[k - W + 1, k]',
         'inside_the_window_as_implemented': True},
        {'sub_test': 'gap_clause', 'reads': '|t_sum_k| <= GAP_BOUND', 'cycles': '[k, k]',
         'inside_the_window_as_implemented': True},
        {'sub_test': 'veto', 'reads': 'all_optimal_c', 'cycles': '[k - W + 1, k]', 'inside_the_window_as_implemented': True},
    ],
    'monotone': [
        {'sub_test': 'lo_ge_k0_plus_K_EXCL', 'reads': 'k - L + 1 >= k0 + K_EXCL',
         'cycles': 'k0: the absence of a Boyd lapse over [k0, k]', 'inside_the_window_as_implemented': False},
        {'sub_test': 'no_sign_change_in_window', 'reads': 'j_change (the last sign change) < lo',
         'cycles': ('the steps over [lo, k] AND the sign of the last non-zero step before lo (cycles j - 1, j, j the '
                    'last c in [k0 + K_EXCL, lo - 1] with |dQ_c| >= EPS0): a sign change at the first non-zero step '
                    'of the window is judged against it'),
         'inside_the_window_as_implemented': False},
        {'sub_test': 'range_le_tau', 'reads': 'max - min of Q over the window', 'cycles': '[lo, k]',
         'inside_the_window_as_implemented': True},
        {'sub_test': 'steps_decreasing', 'reads': ('the half-window means of |dQ_c|, c in [lo, k] (or the stationary '
                                                   'sub-case max |dQ_c| < EPS0)'),
         'cycles': '[lo - 1, k] (dQ_lo = Q_lo - Q_(lo-1))', 'inside_the_window_as_implemented': False},
        {'sub_test': 'last_step_times_L_le_tau', 'reads': '|dQ_k| * L <= TAU', 'cycles': '[k - 1, k]',
         'inside_the_window_as_implemented': True},
        {'sub_test': 'gap_clause', 'reads': '|t_sum_k| <= GAP_BOUND', 'cycles': '[k, k]',
         'inside_the_window_as_implemented': True},
        {'sub_test': 'veto', 'reads': 'all_optimal_c', 'cycles': '[k - L + 1, k]', 'inside_the_window_as_implemented': True},
    ],
    'status': ('ENUMERATED, NOT ASSERTED: the Addendum 59 Supplement states the test reads only the last W cycles; as '
               'implemented the entries marked inside_the_window_as_implemented False read further back. Per '
               'certification the concrete out-of-window reads are recorded (decision out_of_window_reads). Flagged by '
               'the Planner for the expert (TASKS.md, Addendum 59 Supplement entry); the algorithm is not changed'),
}


def constants(p_max):
    """The constants with their formulas (for the frozen spec). `p_max` is the spec constant."""
    out = SC2.constants(p_max)
    out.update({
        'VERSION': {'value': VERSION, 'formula': ('settling_criterion_v4 (settling_criterion.py = version 1, '
                                                  'settling_criterion_v2.py = version 2, settling_criterion_v3.py = '
                                                  'version 3, all unchanged)')},
        'BOYD_K': {'value': 'boyd_k (version 2: all_boyd_pass AND local_solves_ok); no non-Optimal reset',
                   'formula': 'Addendum 59 reading (gamma)'},
        'VETO_WINDOW': {'value': 'oscillatory [k - W + 1, k]; monotone [k - L_MONO + 1, k]',
                        'formula': 'Addendum 59 Supplement, reading (a): the last W cycles the certification test reads'},
        'VETO_REASON': {'value': VETO_REASON, 'formula': 'certification refused at that cycle only; state unchanged'},
        'OPTIMAL_CLASS': {'value': OPTIMAL_CLASS,
                          'formula': 'p515_s44_campaign_harness.ipopt_exit_class(message) == "optimal"'},
        'CAP': {'value': 'fixed N_old + 100 (gated) | min(first k0 + 109, CAP_CEILING) (ungated)',
                'formula': 'as version 3; CAP_AFTER_K0 = 109; CAP_CEILING per cell (spec)'},
        'CAP_CEILING': {'value': 'per cell (the spec cell table)', 'formula': 'as version 3'},
    })
    return out


def _span(lo, hi):
    return [lo, hi] if (lo is not None and hi is not None and lo <= hi) else None


class SettlingRuleV4(SC2.SettlingRuleV2):
    """Version 2 fed the version-2 boyd_k (no non-Optimal reset) with the certification veto on the certifying window.
    Exactly one of `cap` (fixed) or `cap_after_first_k0` (dynamic) is given; `cap_ceiling` (per cell) is REQUIRED.
    `observe(k, q, boyd, t_sum, all_optimal)` returns the cycle's full record and, on certification or at the cap, sets
    `self.decision`."""

    def __init__(self, p_max, cap=None, cap_after_first_k0=None, cap_ceiling=None):
        if cap_ceiling is None:
            raise ValueError('SettlingRuleV4: cap_ceiling is a per-cell spec value and must be given')
        super().__init__(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)
        self.all_optimal = {}
        self.non_optimal_cycles = []
        self.vetoes = []
        self._cycle_veto = None

    # ---- the veto (steps 6-7 of version 2, then V) -------------------------------------------------------------------
    def _non_optimal_in(self, window):
        lo, hi = window
        return [c for c in self.non_optimal_cycles if lo <= c <= hi]

    def evaluate(self, k):
        cert_a, cert_b, a_parts, b_parts, reasons = SC2.SettlingRuleV2.evaluate(self, k)
        vetoed = []
        if cert_a:
            hit = self._non_optimal_in(a_parts['window'])
            if hit:
                cert_a = False
                vetoed.append({'branch': 'oscillatory', 'window': list(a_parts['window']), 'W': a_parts['W'],
                               'non_optimal_in_window': hit})
                cert_b = bool(b_parts['lo_ge_k0_plus_K_EXCL'] and b_parts['no_sign_change_in_window']
                              and b_parts['range_le_tau'] and b_parts['steps_decreasing']
                              and b_parts['last_step_times_L_le_tau'])
        if cert_b:
            hit = self._non_optimal_in(b_parts['window'])
            if hit:
                cert_b = False
                vetoed.append({'branch': 'monotone', 'window': list(b_parts['window']), 'W': self.l_mono,
                               'non_optimal_in_window': hit})
        if vetoed:
            reasons = list(reasons) + [VETO_REASON]
            self._cycle_veto = {'cycle': k, 'branches': vetoed, 'k0': self.k0}
            self.vetoes.append(dict(self._cycle_veto))
        return cert_a, cert_b, a_parts, b_parts, reasons

    # ---- the out-of-window reads of a certification (enumerated, never asserted away) --------------------------------
    def out_of_window_reads(self, k, branch, a_parts, b_parts):
        """The concrete reads of the certification test at k OUTSIDE the branch's window (SUB_TEST_READS applied to
        this state): which turning points, which cycles, and whether any of those cycles is non-Optimal. Pure."""
        nonopt = set(self.non_optimal_cycles)
        window = a_parts['window'] if branch == 'oscillatory' else b_parts['window']
        lo = window[0]

        def span_entry(first, what):
            s = _span(first, k)
            out_s = _span(first, lo - 1)
            bad = sorted(c for c in nonopt if out_s and out_s[0] <= c <= out_s[1])
            return {'reads': what, 'cycles': s, 'outside_window': out_s, 'non_optimal_cycles_outside_window': bad,
                    'any_non_optimal_outside_window': bool(bad)}

        def tp_entry(tp):
            t = tp[0]
            return {'t': t, 'kind': tp[1], 'Q_t': tp[2], 'outside_window': t < lo, 'non_optimal': t in nonopt}

        reads = {
            'k0_no_boyd_lapse_span': span_entry(self.k0, 'boyd_k and Q not None (no lapse since k0)'),
            'sign_state_Q_span': span_entry(self.k0 + K_EXCL - 1, ('Q (every step dQ_c, c >= k0 + K_EXCL, and the '
                                                                  'extremum search locating each turning point)')),
        }
        T = list(self.sign.T)
        if branch == 'oscillatory':
            outside_tp = [tp_entry(tp) for tp in T if tp[0] < lo]
            reads['turning_point_count'] = {'reads': 'len(T) >= 3', 'n_turning_points': len(T),
                                            'turning_points_outside_window': outside_tp}
            reads['swings_non_increasing_all_pairs'] = {
                'reads': 'A[i] = |Q_T[i+1] - Q_T[i]| for every consecutive pair since k0',
                'turning_points_read': [tp_entry(tp) for tp in T],
                'turning_points_outside_window': outside_tp,
                'pairs_with_an_endpoint_outside_window': [[T[i][0], T[i + 1][0]] for i in range(len(T) - 1)
                                                          if T[i][0] < lo]}
            reads['P_hat'] = {'reads': 'T[-1].t - T[-3].t', 'P_hat': a_parts['P_hat'],
                              'positions': [tp_entry(T[-3]), tp_entry(T[-1])],
                              'positions_outside_window': [tp_entry(tp) for tp in (T[-3], T[-1]) if tp[0] < lo]}
            tps_out = outside_tp
        else:
            j_carry = None
            for c in range(lo - 1, self.k0 + K_EXCL - 1, -1):
                if c in self.q and (c - 1) in self.q and abs(self.q[c] - self.q[c - 1]) >= EPS0:
                    j_carry = c
                    break
            carry = [j_carry - 1, j_carry] if j_carry is not None else None
            reads['steps_decreasing_first_step'] = {
                'reads': 'dQ_lo = Q_lo - Q_(lo-1)', 'cycles_outside_window': [lo - 1],
                'non_optimal': (lo - 1) in nonopt}
            reads['no_sign_change_carry_in_step'] = {
                'reads': 'the sign of the last non-zero step before lo (judges the first non-zero step of the window)',
                'j_change': self.sign.j_change, 'carry_in_step_cycles': carry,
                'non_optimal': bool(carry) and any(c in nonopt for c in carry)}
            tps_out = []
        cycles_out = set()
        for key in ('k0_no_boyd_lapse_span', 'sign_state_Q_span'):
            s = reads[key]['outside_window']
            if s:
                cycles_out.update(range(s[0], s[1] + 1))
        if branch == 'monotone':
            cycles_out.add(lo - 1)
        bad_tp = [x['t'] for x in tps_out if x['non_optimal']]
        return {'branch': branch, 'k': k, 'k0': self.k0, 'window': list(window),
                'W': a_parts['W'] if branch == 'oscillatory' else self.l_mono,
                'window_all_optimal': not self._non_optimal_in(window),
                'reads': reads,
                'any_out_of_window_read': bool(cycles_out),
                'turning_points_outside_window': [x['t'] for x in tps_out],
                'non_optimal_turning_points_outside_window': bad_tp,
                'non_optimal_cycles_read_outside_window': sorted(c for c in cycles_out if c in nonopt),
                'any_non_optimal_read_outside_window': any(c in nonopt for c in cycles_out),
                'enumeration': 'settling_criterion_v4.SUB_TEST_READS'}

    # ---- one cycle -----------------------------------------------------------------------------------------------------
    def observe(self, k, q, boyd, t_sum, all_optimal):
        if not isinstance(all_optimal, bool):
            raise TypeError(f'SettlingRuleV4: all_optimal must be a bool at cycle {k}; got {all_optimal!r}')
        if self.decision is not None:
            raise RuntimeError(f'SettlingRuleV4: cycle {k} observed after the decision ({self.decision.get("status")})')
        self.all_optimal[k] = all_optimal
        if not all_optimal:
            self.non_optimal_cycles.append(k)
        self._cycle_veto = None
        rec = super().observe(k, q, bool(boyd), t_sum)
        veto = self._cycle_veto
        rec.update({'all_optimal_k': all_optimal, 'vetoed': veto})
        if veto is not None and rec.get('decision') == 'continue':
            rec['decision'] = VETO_REASON
        dec = self.decision
        if dec is not None and dec.get('version') != VERSION:
            dec.update({'version': VERSION, 'reading': READING, 'N': self.n,
                        'N_definition': 'the first residual pass (version 2; no non-Optimal reset)',
                        'window_rule': READINGS['window_reading_a'],
                        'non_optimal_cycles': list(self.non_optimal_cycles), 'vetoes': [dict(v) for v in self.vetoes],
                        'n_vetoes': len(self.vetoes), 'cap_ceiling': self.cap_ceiling})
            if dec['status'] == 'certified':
                oow = self.out_of_window_reads(k, dec['branch'], dec['certA_parts'], dec['certB_parts'])
                dec['window_all_optimal'] = oow['window_all_optimal']
                dec['out_of_window_reads'] = oow
                rec['out_of_window_reads'] = oow
            else:
                if veto is not None and VETO_REASON not in dec['reasons']:
                    dec['reasons'] = [r for r in FAILING_REASONS if r in set(dec['reasons']) | {VETO_REASON}]
                dec['certification_vetoed_at_cap'] = veto is not None
                dec['window_all_optimal'] = None
                dec['out_of_window_reads'] = None
        return rec


def replay(q_by_cycle, boyd_by_cycle, t_by_cycle, all_optimal_by_cycle, p_max, cap=None, cap_after_first_k0=None,
           cap_ceiling=None, last=None):
    """Runs the rule over cycles 1..last (default: the cap in force) from the records ({cycle: value}; a missing cycle
    reads as Q None / boyd False / t None / all_optimal False). Stops at the decision. Returns (records, decision,
    rule)."""
    rule = SettlingRuleV4(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)
    out = []
    k = 0
    while True:
        k += 1
        if last is not None and k > last:
            break
        if k > rule.effective_cap():
            break
        out.append(rule.observe(k, q_by_cycle.get(k), bool(boyd_by_cycle.get(k, False)), t_by_cycle.get(k),
                                bool(all_optimal_by_cycle.get(k, False))))
        if rule.decision is not None:
            break
    return out, rule.decision, rule
