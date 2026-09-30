"""The settling stop rule, VERSION 6 (P5.15 Addendum 61: the swing noise floor; Planner task W142).

A NEW module: `settling_criterion.py` (version 1) and `settling_criterion_v2.py` .. `settling_criterion_v5.py` (versions
2-5) are NOT edited and stay byte-identical; every certificate issued under them is untouched. This module imports
versions 1, 2 and 5 and re-implements only what version 6 changes.

WHAT CHANGED FROM VERSION 5 (and nothing else):
  (A) THE GROWTH-TEST FLOOR (Addendum 61 ruling 1, adopted). In the oscillatory branch's "swings not growing" test a
      swing A[i] < F = SWING_FLOOR = TAU / 10 is EXCLUDED FROM THE COMPARISON: the swings >= F, in their order, must be
      non-increasing over every consecutive pair of that sequence (an excluded swing does not break the chain, so an
      earlier growth trend across it is still compared -- the property for which the expert rejected the last-pair
      test). The turning points are still registered, and P_hat and W still use them.
  (B) THE TURNING-POINT FLOOR (Addendum 61 ruling 1, conditional; CARRIED -- W142 item 1: replayed on the 18 committed
      re-settle records, (A)+(B) changes no committed certificate relative to (A) alone and v5, so by the Addendum 61
      rule v6 carries both; data/SRP1/Results/P515S53/w142_resettle_v6/floor_replay/w142_floor_replay.json). A sign
      change whose closing swing |Q_t - Q_T[-1]| is below F does NOT register a turning point: the candidate is
      rejected, T[-1] (with the swing it closed and the sign change that registered it) is un-registered and CARRIED as
      the running extreme of the resumed leg; at the next sign change the candidate is the more extreme of the carried
      point and version 1's extremum (ties: the carried, i.e. the earlier, point). The rejected reversal is not a sign
      change for the monotone branch either (j_change reverts to its value before T[-1] registered). The first turning
      point closes no swing and always registers. This is W141's V3 semantics (`p515_s53_w141_swing_variants.
      FloorSignState`, d02efe69), restated here stdlib-only; with F = 0 it is version 1's sign state exactly.
      Under (B) every registered swing is >= F, so (A) excludes nothing whenever (B) is in force; (A) is carried as
      ruled and is the operative clause only if (B) is ever switched off (the checks test both).
  Everything else is version 5 EXACTLY: the clean veto (all_clean_k; the 10x rule), reading (gamma), window (a), the
  gap clause, the monotone branch, N and the caps, the sub-test read enumeration (extended by the two floor reads).

THE REPORT-STAGE DETERMINACY FLOOR (Addendum 61 ruling 2; a SCORER rule, not a stop-rule clause -- the rule above never
reads it): a difference between two CERTIFIED cells is DETERMINATE iff |margin| >= max(3 x the larger of the two cells'
bars, 2 TAU), a cell's bar being its band width (`determinacy_threshold`, `determinate_certified`). The 2 TAU floor is
"the honest consequence of cells moving up to TAU each" (cells continued past certification moved at most 0.9 TAU). The
uncertified-form rule (3 x max(gap, slack)) for differences involving an uncertified cell is unchanged (not here).

CONSTANTS: version 5's unchanged; SWING_FLOOR = TAU / 10 (Addendum 61: "Swing noise floor tau/10 (~454 EUR)");
GROWTH_TEST_FLOOR = TURNING_POINT_FLOOR = SWING_FLOOR; DETERMINACY_BAR_FACTOR = 3; DETERMINACY_TAU_MULTIPLE = 2.

Stdlib only; no I/O; no model code. Numbers and comparisons exactly as version 5 (and version 2) elsewhere; the floors
compare with `<` (a swing EQUAL to F is kept / registers).
"""

import copy

import settling_criterion as SC1
import settling_criterion_v2 as SC2
import settling_criterion_v5 as SC5

__all__ = ['VERSION', 'SCHEMA', 'TAU', 'EPS0', 'K_EXCL', 'W_MIN', 'W_FACTOR', 'GAP_BOUND', 'DRIFT_WINDOW',
           'CAP_AFTER_K0', 'CLEAN_FACTOR', 'METRICS', 'METRIC_TABLE', 'TOLERANCES', 'TOLERANCE_SOURCES',
           'ATTEMPT_TIERS', 'PRIMARY_ATTEMPT', 'OPTIMAL_CLASS', 'ACCEPTABLE_CLASS', 'CLEAN_REASONS', 'FAILING_REASONS',
           'READINGS', 'READING', 'VETO_REASON', 'SUB_TEST_READS', 'SWING_FLOOR', 'GROWTH_TEST_FLOOR',
           'TURNING_POINT_FLOOR', 'CARRIES', 'DETERMINACY_BAR_FACTOR', 'DETERMINACY_TAU_MULTIPLE', 'constants',
           'classify_block_exit', 'growth_test', 'FloorSignState', 'SettlingRuleV6', 'replay',
           'determinacy_threshold', 'determinate_certified']

VERSION = 6
SCHEMA = 'settling_criterion_v6'
TAU = SC5.TAU
EPS0 = SC5.EPS0
K_EXCL = SC5.K_EXCL
W_MIN = SC5.W_MIN
W_FACTOR = SC5.W_FACTOR
GAP_BOUND = SC5.GAP_BOUND
DRIFT_WINDOW = SC5.DRIFT_WINDOW
CAP_AFTER_K0 = SC5.CAP_AFTER_K0
CLEAN_FACTOR = SC5.CLEAN_FACTOR
METRICS = SC5.METRICS
METRIC_TABLE = SC5.METRIC_TABLE
TOLERANCES = SC5.TOLERANCES
TOLERANCE_SOURCES = SC5.TOLERANCE_SOURCES
ATTEMPT_TIERS = SC5.ATTEMPT_TIERS
PRIMARY_ATTEMPT = SC5.PRIMARY_ATTEMPT
OPTIMAL_CLASS = SC5.OPTIMAL_CLASS
ACCEPTABLE_CLASS = SC5.ACCEPTABLE_CLASS
CLEAN_REASONS = SC5.CLEAN_REASONS
VETO_REASON = SC5.VETO_REASON
FAILING_REASONS = SC5.FAILING_REASONS
classify_block_exit = SC5.classify_block_exit

# ---- the swing noise floor (Addendum 61 ruling 1) ---------------------------------------------------------------------
SWING_FLOOR = TAU / 10.0                 # "Swing noise floor tau/10 (~454 EUR), adopted"
GROWTH_TEST_FLOOR = SWING_FLOOR          # (A): swings below it are excluded from the "not growing" comparison
TURNING_POINT_FLOOR = SWING_FLOOR        # (B): a sign change closing a swing below it registers no turning point
CARRIES = ('A_growth_test_floor', 'B_turning_point_floor')
CARRY_EVIDENCE = {'replay': 'data/SRP1/Results/P515S53/w142_resettle_v6/floor_replay/w142_floor_replay.json',
                  'script': 'p515_s53_w142_floor_replay.py',
                  'outcome': ('(A)+(B) changes no committed certificate relative to (A) alone and v5 on the 18 committed '
                              're-settle records (14 of them carry a committed certificate; on each, status, k*, window '
                              'and branch are equal under v5, (A) and (A)+(B)) -> v6 carries both (Addendum 61 rule, '
                              'applied mechanically); the one record that changes, d_c52e1670 (uncertified under v5), '
                              'carries no committed certificate: (A) alone certifies it at 139, (A)+(B) at 150')}

# ---- the report-stage determinacy floor (Addendum 61 ruling 2) --------------------------------------------------------
DETERMINACY_BAR_FACTOR = 3.0
DETERMINACY_TAU_MULTIPLE = 2.0

READING = 'gamma_window_a_clean_10x_swing_floor_tau_over_10'
READINGS = dict(SC5.READINGS)
READINGS.update({
    'growth_test_floor_A': ('Addendum 61 ruling 1 (A): in the "swings not growing" comparison a swing A[i] < TAU / 10 '
                            'is excluded; the swings >= TAU / 10, in order, must be non-increasing over every '
                            'consecutive pair of that sequence (the chain reading: an excluded swing does not break '
                            'the comparison across it; Worker reading, the other reading -- only original consecutive '
                            'pairs both >= F -- gives the same decisions on every committed record, W142 A_pairs); '
                            'turning points still registered, P_hat and W still use them'),
    'turning_point_floor_B': ('Addendum 61 ruling 1 (B), carried by the Addendum 61 rule on the W142 replay: a sign '
                              'change whose closing swing is < TAU / 10 registers no turning point -- W141 V3 semantics '
                              '(un-register T[-1] and carry it as the running extreme of the resumed leg; the rejected '
                              'reversal is no sign change for the monotone branch)'),
    'floor_comparison': 'a swing equal to the floor is kept (growth test) and registers (turning points): strict <',
    'determinacy_floor_report_stage': ('Addendum 61 ruling 2: a difference between CERTIFIED cells is determinate iff '
                                       '|margin| >= max(3 x the larger band width, 2 TAU); the uncertified form '
                                       'unchanged; a scorer rule, never read by the stop rule'),
})

# ---- the sub-test read enumeration: version 5's plus the two floor reads ------------------------------------------------
SUB_TEST_READS = copy.deepcopy(SC5.SUB_TEST_READS)
for _e in SUB_TEST_READS['oscillatory']:
    if _e['sub_test'] == 'swings_non_increasing_all_pairs':
        _e['sub_test'] = 'swings_non_increasing_floored'
        _e['reads'] = ('A[i] for every pair of consecutive turning points since k0; the swings >= GROWTH_TEST_FLOOR, in '
                       'order, compared pairwise (version 6 (A))')
SUB_TEST_READS['oscillatory'].insert(1, {
    'sub_test': 'turning_point_floor',
    'reads': ('at every sign change since k0: the candidate extremum, the carried point (if any) and |Q_t - Q_T[-1]| '
              'against TURNING_POINT_FLOOR (version 6 (B)); a rejected candidate un-registers T[-1]'),
    'cycles': 'the same Q span as the turning-point detection, plus Q at T[-1] (which may precede the window)',
    'inside_the_window_as_implemented': False})
SUB_TEST_READS['status'] = SC5.SUB_TEST_READS['status'].replace('(unchanged in v5)', '(unchanged in v5 and v6)')


def constants(p_max):
    """The constants with their formulas (for the frozen spec). `p_max` is the spec constant."""
    out = SC5.constants(p_max)
    out.update({
        'VERSION': {'value': VERSION, 'formula': ('settling_criterion_v6 (versions 1-5 in settling_criterion.py, _v2.py '
                                                  '.. _v5.py, all unchanged)')},
        'SWING_FLOOR': {'value': SWING_FLOOR, 'formula': 'F = TAU / 10 (Addendum 61 ruling 1: swing noise floor tau/10)'},
        'GROWTH_TEST_FLOOR': {'value': GROWTH_TEST_FLOOR,
                              'formula': '(A) = F: swings < F excluded from the "not growing" comparison'},
        'TURNING_POINT_FLOOR': {'value': TURNING_POINT_FLOOR,
                                'formula': ('(B) = F: a sign change closing a swing < F registers no turning point '
                                            '(carried: W142 replay, Addendum 61 rule)')},
        'CARRIES': {'value': list(CARRIES), 'formula': CARRY_EVIDENCE},
        'DETERMINACY_FLOOR': {'value': {'bar_factor': DETERMINACY_BAR_FACTOR, 'tau_multiple': DETERMINACY_TAU_MULTIPLE,
                                        'two_tau': DETERMINACY_TAU_MULTIPLE * TAU},
                              'formula': ('report stage (scorer): certified-cell difference determinate iff |margin| >= '
                                          'max(3 x max(band_r, band_o), 2 TAU) (Addendum 61 ruling 2)')},
    })
    return out


# ======================================================================================================================
#  (A) and (B)
# ======================================================================================================================
def growth_test(A, floor):
    """(A): (ok, detail) -- the swings >= floor, in order, non-increasing over every consecutive pair of that sequence.
    Pure."""
    kept = [i for i, a in enumerate(A) if a >= floor]
    pairs = [[kept[j], kept[j + 1]] for j in range(len(kept) - 1)]
    ok = all(A[j2] <= A[j1] for j1, j2 in pairs)
    return ok, {'excluded_swing_indices': [i for i, a in enumerate(A) if a < floor], 'pairs_compared': pairs,
                'floor': floor}


class FloorSignState(SC1._SignState):
    """(B): version 1's step-5 sign state with a turning-point floor (W141 V3 semantics; see the module docstring).
    floor 0 reproduces version 1 exactly (every swing is >= 0)."""

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


# ======================================================================================================================
#  the rule
# ======================================================================================================================
class SettlingRuleV6(SC5.SettlingRuleV5):
    """Version 5's algorithm with (A) and (B) (see the module docstring). Exactly one of `cap` (fixed) or
    `cap_after_first_k0` (dynamic) is given; `cap_ceiling` (per cell) is REQUIRED. `observe(k, q, boyd, t_sum,
    all_clean)` returns the cycle's full record and, on certification or at the cap, sets `self.decision`.
    The floors are class attributes (GROWTH_FLOOR, TP_FLOOR); the checks subclass with 0 to prove v6 == v5 at F = 0."""

    GROWTH_FLOOR = GROWTH_TEST_FLOOR
    TP_FLOOR = TURNING_POINT_FLOOR

    def __init__(self, p_max, cap=None, cap_after_first_k0=None, cap_ceiling=None):
        self.floor_rejections = []      # every (B) rejection since cycle 1, across k0 resets
        self._sign = None
        super().__init__(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)

    # SettlingRuleV2 assigns a fresh SC1._SignState at init and at every k0 reset: it is replaced by the floor state.
    @property
    def sign(self):
        return self._sign

    @sign.setter
    def sign(self, value):
        if not (isinstance(value, SC1._SignState) and not value.T and value.sign_prev is None):
            raise RuntimeError('SettlingRuleV6: the sign state must only be assigned fresh')
        if isinstance(self._sign, FloorSignState):
            self.floor_rejections.extend(r for r in self._sign.rejections if r not in self.floor_rejections)
        self._sign = FloorSignState(self.TP_FLOOR)

    def _all_rejections(self):
        out = list(self.floor_rejections)
        out.extend(r for r in self._sign.rejections if r not in out)
        return out

    def evaluate(self, k):
        _ca, _cb, a_parts, b_parts, _r = SC2.SettlingRuleV2.evaluate(self, k)
        ok, det = growth_test(self.sign.A, self.GROWTH_FLOOR)
        a_parts.update({'swings_non_increasing_floored': ok, 'growth_floor': self.GROWTH_FLOOR,
                        'turning_point_floor': self.TP_FLOOR, 'swings_excluded_below_floor': det['excluded_swing_indices'],
                        'swing_pairs_compared': det['pairs_compared'],
                        'swings_non_increasing_all_pairs_note': ('version 2\'s unfloored all-pairs value, REPORT-ONLY; '
                                                                 'version 6 decides on swings_non_increasing_floored')})
        cert_a = bool(a_parts['at_least_3_turning_points'] and ok and a_parts['window_inside_run']
                      and a_parts['range_le_tau'])
        b_ok = bool(b_parts['lo_ge_k0_plus_K_EXCL'] and b_parts['no_sign_change_in_window'] and b_parts['range_le_tau']
                    and b_parts['steps_decreasing'] and b_parts['last_step_times_L_le_tau'])
        cert_b = (not cert_a) and b_ok
        reasons = []
        if not cert_a and not cert_b:                     # version 2's reason list, the swing reason from (A)
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
        # ---- version 5's veto (settling_criterion_v5.SettlingRuleV5.evaluate, reproduced: it calls
        #      SettlingRuleV2.evaluate by class name and so cannot be reused around a replaced swing test) ----
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
            reasons = list(reasons) + [VETO_REASON]
            self._cycle_veto = {'cycle': k, 'branches': vetoed, 'k0': self.k0}
            self.vetoes.append(dict(self._cycle_veto))
        return cert_a, cert_b, a_parts, b_parts, reasons

    def out_of_window_reads(self, k, branch, a_parts, b_parts):
        """Version 5's enumeration plus the two floor reads (the (B) rejections since k0 and the swings (A) excluded)."""
        out = SC5.SettlingRuleV5.out_of_window_reads(self, k, branch, a_parts, b_parts)
        lo = out['window'][0]
        rej = list(self._sign.rejections)
        out['reads']['turning_point_floor'] = {
            'reads': 'the (B) rejections since k0: the candidate, the swing it closed, the point un-registered',
            'rejections_since_k0': rej,
            'rejections_outside_window': [r for r in rej if r['cycle'] < lo]}
        if branch == 'oscillatory':
            out['reads']['swings_excluded_below_growth_floor'] = {
                'reads': '(A): the swings < GROWTH_TEST_FLOOR excluded from the comparison',
                'excluded_swing_indices': a_parts.get('swings_excluded_below_floor'),
                'pairs_compared': a_parts.get('swing_pairs_compared')}
        out['enumeration'] = 'settling_criterion_v6.SUB_TEST_READS'
        return out

    def observe(self, k, q, boyd, t_sum, all_clean):
        n_rej = len(self._all_rejections())
        rec = super().observe(k, q, boyd, t_sum, all_clean)
        rej = self._all_rejections()
        rec['turning_point_floor_rejection'] = rej[-1] if len(rej) > n_rej else None
        dec = self.decision
        if dec is not None and dec.get('version') != VERSION:
            dec.update({'version': VERSION, 'reading': READING,
                        'swing_floor': {'F': SWING_FLOOR, 'growth_test_floor': self.GROWTH_FLOOR,
                                        'turning_point_floor': self.TP_FLOOR, 'carries': list(CARRIES)},
                        'turning_point_floor_rejections': rej})
            if dec.get('out_of_window_reads'):
                dec['out_of_window_reads']['enumeration'] = 'settling_criterion_v6.SUB_TEST_READS'
                rec['out_of_window_reads'] = dec['out_of_window_reads']
        return rec


def replay(q_by_cycle, boyd_by_cycle, t_by_cycle, all_clean_by_cycle, p_max, cap=None, cap_after_first_k0=None,
           cap_ceiling=None, last=None, rule_class=None):
    """Runs the rule over cycles 1..last (default: the cap in force) from the records ({cycle: value}; a missing cycle
    reads as Q None / boyd False / t None / all_clean False). Stops at the decision. Returns (records, decision, rule).
    `rule_class` (checks only): a subclass of SettlingRuleV6."""
    cls = rule_class or SettlingRuleV6
    rule = cls(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)
    out = []
    k = 0
    while True:
        k += 1
        if last is not None and k > last:
            break
        if k > rule.effective_cap():
            break
        out.append(rule.observe(k, q_by_cycle.get(k), bool(boyd_by_cycle.get(k, False)), t_by_cycle.get(k),
                                bool(all_clean_by_cycle.get(k, False))))
        if rule.decision is not None:
            break
    return out, rule.decision, rule


# ======================================================================================================================
#  the report-stage determinacy floor (Addendum 61 ruling 2) -- pure; the scorer calls it, the rule never does
# ======================================================================================================================
def determinacy_threshold(bar_r, bar_o):
    """max(3 x max(bar_r, bar_o), 2 TAU); a bar is a certified cell's band width."""
    return max(DETERMINACY_BAR_FACTOR * max(bar_r, bar_o), DETERMINACY_TAU_MULTIPLE * TAU)


def determinate_certified(margin, bar_r, bar_o):
    """(determinate: bool, threshold, which term binds). Determinate iff |margin| >= the threshold."""
    three = DETERMINACY_BAR_FACTOR * max(bar_r, bar_o)
    two_tau = DETERMINACY_TAU_MULTIPLE * TAU
    thr = max(three, two_tau)
    return abs(margin) >= thr, thr, ('3 x larger bar' if three >= two_tau else '2 TAU')
