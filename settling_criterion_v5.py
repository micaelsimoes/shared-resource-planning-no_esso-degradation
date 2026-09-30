"""The settling stop rule, VERSION 5 (P5.15 Addendum 60: the veto bounded -- an Acceptable exit within 10x the tail
tolerances is clean; Planner task W139).

A NEW module: `settling_criterion.py` (version 1), `settling_criterion_v2.py` (version 2), `settling_criterion_v3.py`
(version 3) and `settling_criterion_v4.py` (version 4) are NOT edited and stay byte-identical; every certificate issued
under them is untouched. This module imports versions 2 and 4 and re-implements only what version 5 changes.

WHAT CHANGED FROM VERSION 4 (and nothing else):
  (1) THE PER-CYCLE FLAG THE VETO READS IS all_clean_k, NOT all_optimal_k. A cycle is CLEAN iff the final accepted exit
      of every block of the cycle (12 TSO, 36 DSO, 3 ESSO) is clean (`classify_block_exit`):
        * it is "Optimal Solution Found" (exit class 'optimal', on ANY attempt tier -- as version 4, where an Optimal
          recovery counted as Optimal: version 5 is strictly more permissive on the veto and identical otherwise,
          Addendum 60 "a v4 certificate is a v5 certificate"); or
        * it is "Solved To Acceptable Level" (exit class 'acceptable') on the PRIMARY attempt (not 'recovery', not
          'recovery_tier2') AND each of the four IPOPT metrics of that final iterate is <= CLEAN_FACTOR (10) x its
          tolerance (METRIC_TABLE: which log line, which column -- scaled or unscaled -- and which IPOPT option;
          TOLERANCES: the values in force under the tail, read from source).
      Anything else is NON-CLEAN: an Acceptable exit on a recovery tier (of any size), an Acceptable exit with any
      metric beyond 10x (or a metric that could not be read), any other exit, a block with no exit.
  (2) Everything else is version 4 EXACTLY: reading (gamma) -- no reset on a non-clean cycle, only a Boyd lapse (or a
      failed cycle) resets k0 and the sign state; the certification veto on the certifying window = reading (a), the
      last W cycles the test reads (oscillatory [k - W + 1, k], monotone [k - L + 1, k]), refusing certification at
      that cycle only (reason 'certification_vetoed_non_clean'), state unchanged; an oscillatory verdict vetoed -> the
      monotone conjunction re-evaluated from version 2's own b_parts, then vetoed on its own window; the gap clause,
      the residuals, the holds, N and the dynamic cap, the per-cell cap ceiling; the sub-test reads enumerated (not
      asserted) and the out-of-window reads recorded per certification. The records and the decision carry version
      4's fields under the v5 names (all_clean_k, non_clean_cycles, window_all_clean, non_clean_* in the out-of-window
      reads). `V4_FIELD_RENAMES` maps them; the checks prove version 5 == version 4 under that map for every flag
      series (V_eq_v4).

CONSTANTS: those of version 4 (and so of version 2) unchanged; CLEAN_FACTOR = 10.0 (Addendum 60: "one order of
magnitude, not a value tuned to 2.51"); METRIC_TABLE and TOLERANCES below (spec constants, asserted against source by
the W139 checks before any run and by the capture before the first solve).

Stdlib only; no I/O; no model code. Numbers and comparisons exactly as version 4 (and version 2).
"""

import copy
import math

import settling_criterion_v2 as SC2
import settling_criterion_v4 as SC4

__all__ = ['VERSION', 'SCHEMA', 'TAU', 'EPS0', 'K_EXCL', 'W_MIN', 'W_FACTOR', 'GAP_BOUND', 'DRIFT_WINDOW',
           'CAP_AFTER_K0', 'CLEAN_FACTOR', 'METRICS', 'METRIC_TABLE', 'TOLERANCES', 'TOLERANCE_SOURCES',
           'ATTEMPT_TIERS', 'PRIMARY_ATTEMPT', 'OPTIMAL_CLASS', 'ACCEPTABLE_CLASS', 'CLEAN_REASONS',
           'FAILING_REASONS', 'READINGS', 'VETO_REASON', 'SUB_TEST_READS', 'V4_FIELD_RENAMES', 'constants',
           'classify_block_exit', 'SettlingRuleV5', 'replay']

VERSION = 5
SCHEMA = 'settling_criterion_v5'
TAU = SC4.TAU
EPS0 = SC4.EPS0
K_EXCL = SC4.K_EXCL
W_MIN = SC4.W_MIN
W_FACTOR = SC4.W_FACTOR
GAP_BOUND = SC4.GAP_BOUND
DRIFT_WINDOW = SC4.DRIFT_WINDOW
CAP_AFTER_K0 = SC4.CAP_AFTER_K0

# ---- the clean-exit rule (Addendum 60) ------------------------------------------------------------------------------
CLEAN_FACTOR = 10.0                      # "within 10x the tight-tail tolerances"; one order of magnitude (Addendum 60)
OPTIMAL_CLASS = SC4.OPTIMAL_CLASS        # 'optimal'   = p515_s44_campaign_harness.ipopt_exit_class("... Optimal ...")
ACCEPTABLE_CLASS = 'acceptable'          # 'acceptable' = ipopt_exit_class("... Solved To Acceptable Level ...")
PRIMARY_ATTEMPT = 'primary'
ATTEMPT_TIERS = ('primary', 'recovery', 'recovery_tier2')   # network.py / shared_energy_storage_data.py log_suffix
METRICS = ('overall_nlp_error', 'dual_infeasibility', 'constraint_violation', 'complementarity')
# Which number of IPOPT's final summary ("Number of Iterations ... (scaled) (unscaled)") is compared with which option.
# IPOPT 3.14 (IpOptErrorConv::CheckConvergence, the options documentation): successful termination requires the
# (scaled) overall NLP error <= tol, the max-norm of the UNSCALED dual infeasibility <= dual_inf_tol, of the UNSCALED
# constraint violation <= constr_viol_tol and of the UNSCALED complementarity <= compl_inf_tol. Version 5 compares the
# same four numbers against the same four options, each multiplied by CLEAN_FACTOR.
METRIC_TABLE = {
    'overall_nlp_error': {'log_line': 'Overall NLP error', 'column': 'scaled', 'option': 'tol',
                          'ipopt_meaning': 'the (scaled) NLP error; successful termination requires it <= tol'},
    'dual_infeasibility': {'log_line': 'Dual infeasibility', 'column': 'unscaled', 'option': 'dual_inf_tol',
                           'ipopt_meaning': 'max-norm of the unscaled dual infeasibility; termination requires <= '
                                            'dual_inf_tol'},
    'constraint_violation': {'log_line': 'Constraint violation', 'column': 'unscaled', 'option': 'constr_viol_tol',
                             'ipopt_meaning': 'max-norm of the unscaled constraint violation; termination requires <= '
                                              'constr_viol_tol'},
    'complementarity': {'log_line': 'Complementarity', 'column': 'unscaled', 'option': 'compl_inf_tol',
                        'ipopt_meaning': 'max-norm of the unscaled complementarity; termination requires <= '
                                         'compl_inf_tol'},
}
# The tolerances IN FORCE UNDER THE TAIL, per block family (the options actually passed, read from source; an option
# not passed is IPOPT 3.14's documented default). Asserted against source by the W139 checks and the capture.
TOLERANCES = {
    'network': {'tol': 1e-5, 'dual_inf_tol': 1.0, 'constr_viol_tol': 1e-4, 'compl_inf_tol': 1e-6},
    'esso': {'tol': 1e-10, 'dual_inf_tol': 1.0, 'constr_viol_tol': 1e-4, 'compl_inf_tol': 1e-4},
}
TOLERANCE_SOURCES = {
    'network': {
        'tol': ('passed: every TSO / DSO case file solver.options tol = 1e-05 (data/SRP1/case9/case9_params.json, '
                'case33_1/2/3_params.json); network._create_smopf_solver merges solver_params.options, then the retry '
                'option_overrides (which do not carry tol)'),
        'dual_inf_tol': ('NOT passed (absent from every network case file options and recovery_options, from the '
                         'retry overrides and from the tail): IPOPT 3.14 default 1'),
        'constr_viol_tol': ('NOT passed (absent likewise): IPOPT 3.14 default 1e-4'),
        'compl_inf_tol': ('passed by the convergence-depth tail: 1e-6 on every TSO / DSO solve while the tail is on '
                          '(shared_resources_planning._apply_convergence_depth_tail; the spec configuration '
                          'convergence_depth_tail {enabled: True, compl_inf_tol: 1e-6}; the holds keep it on after the '
                          'run\'s first residual pass). The comparison uses the TAIL value on every cycle (a cycle run '
                          'at the production value -- 1e-4 default for the DSOs, 5e-4 in case9 -- is compared against '
                          'the tighter tail value, the conservative side)'),
    },
    'esso': {
        'tol': ('passed: shared_energy_storage_data.ESSO_TOL_OVERRIDES tol = 1e-10 (applied to every ESSO subproblem '
                'solve, primary and recovery, after the case file\'s tol 1e-06)'),
        'dual_inf_tol': 'NOT passed (ESS params options / recovery_options / ESSO_TOL_OVERRIDES): IPOPT 3.14 default 1',
        'constr_viol_tol': 'NOT passed: IPOPT 3.14 default 1e-4',
        'compl_inf_tol': ('NOT passed: IPOPT 3.14 default 1e-4 (the convergence-depth tail touches only the TSO / DSO '
                          'holders, never the ESSO: "The ESSO is not a network solve and is never touched")'),
    },
    'ipopt_defaults': {'source': 'IPOPT 3.14 options documentation (coin-or.github.io/Ipopt/OPTIONS.html): tol 1e-8, '
                                 'dual_inf_tol 1, constr_viol_tol 1e-4, compl_inf_tol 1e-4',
                       'binary': '/usr/local/bin/ipopt (Ipopt 3.14.18, as every log banner states)'},
}
CLEAN_REASONS = {
    'optimal': 'clean: "Optimal Solution Found" (any attempt tier)',
    'acceptable_primary_within_factor': 'clean: "Solved To Acceptable Level" on the primary attempt, every metric <= 10x',
    'acceptable_not_primary': 'NON-CLEAN: "Solved To Acceptable Level" on a recovery tier',
    'acceptable_beyond_factor': 'NON-CLEAN: "Solved To Acceptable Level" with a metric beyond 10x its tolerance',
    'acceptable_metrics_unavailable': 'NON-CLEAN: "Solved To Acceptable Level" whose metrics or tier could not be read',
    'other_exit': 'NON-CLEAN: an exit that is neither Optimal nor Acceptable',
    'no_exit': 'NON-CLEAN: no result / no exit message',
}

VETO_REASON = 'certification_vetoed_non_clean'
FAILING_REASONS = tuple(SC2.FAILING_REASONS) + (VETO_REASON,)
READING = 'gamma_window_a_clean_10x'
READINGS = dict(SC4.READINGS)
READINGS.pop('all_optimal_definition', None)
READINGS.update({
    'window_reading_a': ('Addendum 59 Supplement (unchanged in v5): the certifying window is the last W cycles the '
                         'certification test reads at k -- oscillatory [k - W + 1, k], monotone [k - L + 1, k]; every '
                         'cycle of it must be all-CLEAN on all 51 blocks, otherwise certification is refused at that '
                         'cycle only (certification_vetoed_non_clean) and the state is unchanged'),
    'all_clean_definition': ('all_clean_k: the final accepted attempt of every block of cycle k (12 TSO, 36 DSO, 3 ESSO) '
                             'is clean (classify_block_exit): Optimal on any tier, or Acceptable on the primary attempt '
                             'with all four IPOPT metrics <= 10x their tolerances (METRIC_TABLE, TOLERANCES); a block '
                             'with no result or no exit message is not clean'),
    'clean_optimal_any_tier': ('an Optimal final exit is clean on ANY attempt tier (a recovery that ends Optimal was '
                               'Optimal under v4; v5 is strictly more permissive on the veto and identical otherwise, '
                               'Addendum 60)'),
    'clean_factor': 'CLEAN_FACTOR = 10 (Addendum 60: one order of magnitude, not tuned to 2.51)',
    'metrics_of_the_final_iterate': ('the four numbers IPOPT prints in the final summary of the final accepted attempt '
                                     '(the scaled Overall NLP error; the unscaled Dual infeasibility, Constraint '
                                     'violation, Complementarity); Variable bound violation is recorded, not compared'),
    'tail_tolerance_on_every_cycle': ('the comparison uses the tail tolerances (network compl_inf_tol 1e-6) on every '
                                      'cycle, including a cycle run at the production tolerance (conservative side)'),
})

# ---- the sub-test read enumeration: version 4's, the veto reading all_clean_c --------------------------------------
SUB_TEST_READS = copy.deepcopy(SC4.SUB_TEST_READS)
for _branch in ('oscillatory', 'monotone'):
    for _e in SUB_TEST_READS[_branch]:
        if _e['sub_test'] == 'veto':
            _e['reads'] = 'all_clean_c'
SUB_TEST_READS['window']['veto'] = ('every cycle of the window of the branch that would certify must be all-CLEAN on '
                                    'all 51 blocks')
SUB_TEST_READS['status'] = SC4.SUB_TEST_READS['status'] + ' (unchanged in v5)'

# version 4 field name -> version 5 field name (records, decisions, vetoes, out-of-window reads)
V4_FIELD_RENAMES = {
    'all_optimal_k': 'all_clean_k',
    'non_optimal_cycles': 'non_clean_cycles',
    'window_all_optimal': 'window_all_clean',
    'non_optimal_in_window': 'non_clean_in_window',
    'non_optimal_cycles_outside_window': 'non_clean_cycles_outside_window',
    'any_non_optimal_outside_window': 'any_non_clean_outside_window',
    'non_optimal': 'non_clean',
    'non_optimal_turning_points_outside_window': 'non_clean_turning_points_outside_window',
    'non_optimal_cycles_read_outside_window': 'non_clean_cycles_read_outside_window',
    'any_non_optimal_read_outside_window': 'any_non_clean_read_outside_window',
    SC4.VETO_REASON: VETO_REASON,
    'settling_criterion_v4.SUB_TEST_READS': 'settling_criterion_v5.SUB_TEST_READS',
}


def constants(p_max):
    """The constants with their formulas (for the frozen spec). `p_max` is the spec constant."""
    out = SC4.constants(p_max)
    out.pop('OPTIMAL_CLASS', None)
    out.update({
        'VERSION': {'value': VERSION, 'formula': ('settling_criterion_v5 (versions 1-4 in settling_criterion.py, '
                                                  '_v2.py, _v3.py, _v4.py, all unchanged)')},
        'BOYD_K': {'value': 'boyd_k (version 2: all_boyd_pass AND local_solves_ok); no non-clean reset',
                   'formula': 'Addendum 59 reading (gamma), unchanged in v5'},
        'VETO_WINDOW': {'value': 'oscillatory [k - W + 1, k]; monotone [k - L_MONO + 1, k]',
                        'formula': 'Addendum 59 Supplement, reading (a) (unchanged in v5)'},
        'VETO_REASON': {'value': VETO_REASON, 'formula': 'certification refused at that cycle only; state unchanged'},
        'CLEAN_FACTOR': {'value': CLEAN_FACTOR, 'formula': 'Addendum 60: within 10x the tight-tail tolerances'},
        'CLEAN_RULE': {'value': ('clean iff exit class optimal (any tier), or exit class acceptable AND attempt primary '
                                 'AND every metric m: value_m <= CLEAN_FACTOR * tolerance_m'),
                       'formula': 'Addendum 60 option 2'},
        'METRIC_TABLE': {'value': METRIC_TABLE, 'formula': 'IPOPT 3.14 termination test (scaled NLP error; unscaled '
                                                           'dual inf., constraint violation, complementarity)'},
        'TOLERANCES': {'value': TOLERANCES, 'formula': TOLERANCE_SOURCES},
    })
    return out


def _finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def classify_block_exit(exit_class, attempt, metrics, family):
    """The v5 clean classification of ONE block's final accepted exit. Pure.
    `exit_class`: 'optimal' | 'acceptable' | 'other' | None (p515_s44_campaign_harness.ipopt_exit_class);
    `attempt`: the final attempt tier ('primary' | 'recovery' | 'recovery_tier2' | None);
    `metrics`: {metric: value} for METRICS (the METRIC_TABLE column of the final summary), or None;
    `family`: 'network' | 'esso' (selects TOLERANCES).
    Returns {'clean': bool, 'reason': CLEAN_REASONS key, 'ratios': {metric: value / tolerance} or None,
    'max_ratio', 'max_ratio_metric', 'tolerances', 'factor'}."""
    if family not in TOLERANCES:
        raise ValueError(f'classify_block_exit: family must be one of {sorted(TOLERANCES)}; got {family!r}')
    tols = TOLERANCES[family]
    ratios = None
    max_ratio, max_metric = None, None
    if isinstance(metrics, dict) and all(_finite_number(metrics.get(m)) for m in METRICS):
        ratios = {m: float(metrics[m]) / tols[METRIC_TABLE[m]['option']] for m in METRICS}
        max_metric = max(METRICS, key=lambda m: ratios[m])
        max_ratio = ratios[max_metric]
    if exit_class == OPTIMAL_CLASS:
        reason = 'optimal'
    elif exit_class == ACCEPTABLE_CLASS:
        if attempt not in ATTEMPT_TIERS or ratios is None:
            reason = 'acceptable_metrics_unavailable'
        elif attempt != PRIMARY_ATTEMPT:
            reason = 'acceptable_not_primary'
        elif all(float(metrics[m]) <= CLEAN_FACTOR * tols[METRIC_TABLE[m]['option']] for m in METRICS):
            reason = 'acceptable_primary_within_factor'
        else:
            reason = 'acceptable_beyond_factor'
    elif exit_class is None:
        reason = 'no_exit'
    else:
        reason = 'other_exit'
    return {'clean': reason in ('optimal', 'acceptable_primary_within_factor'), 'reason': reason, 'ratios': ratios,
            'max_ratio': max_ratio, 'max_ratio_metric': max_metric, 'tolerances': dict(tols), 'factor': CLEAN_FACTOR,
            'family': family, 'exit_class': exit_class, 'attempt': attempt}


def _span(lo, hi):
    return [lo, hi] if (lo is not None and hi is not None and lo <= hi) else None


class SettlingRuleV5(SC2.SettlingRuleV2):
    """Version 4's algorithm with the veto reading all_clean_k (see the module docstring). Exactly one of `cap` (fixed)
    or `cap_after_first_k0` (dynamic) is given; `cap_ceiling` (per cell) is REQUIRED. `observe(k, q, boyd, t_sum,
    all_clean)` returns the cycle's full record and, on certification or at the cap, sets `self.decision`."""

    def __init__(self, p_max, cap=None, cap_after_first_k0=None, cap_ceiling=None):
        if cap_ceiling is None:
            raise ValueError('SettlingRuleV5: cap_ceiling is a per-cell spec value and must be given')
        super().__init__(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)
        self.all_clean = {}
        self.non_clean_cycles = []
        self.vetoes = []
        self._cycle_veto = None

    # ---- the veto (steps 6-7 of version 2, then V) -- version 4's, on the clean flag ------------------------------------
    def _non_clean_in(self, window):
        lo, hi = window
        return [c for c in self.non_clean_cycles if lo <= c <= hi]

    def evaluate(self, k):
        cert_a, cert_b, a_parts, b_parts, reasons = SC2.SettlingRuleV2.evaluate(self, k)
        vetoed = []
        if cert_a:
            hit = self._non_clean_in(a_parts['window'])
            if hit:
                cert_a = False
                vetoed.append({'branch': 'oscillatory', 'window': list(a_parts['window']), 'W': a_parts['W'],
                               'non_clean_in_window': hit})
                cert_b = bool(b_parts['lo_ge_k0_plus_K_EXCL'] and b_parts['no_sign_change_in_window']
                              and b_parts['range_le_tau'] and b_parts['steps_decreasing']
                              and b_parts['last_step_times_L_le_tau'])
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

    # ---- the out-of-window reads of a certification (version 4's enumeration; enumerated, never asserted away) ---------
    def out_of_window_reads(self, k, branch, a_parts, b_parts):
        """The concrete reads of the certification test at k OUTSIDE the branch's window (SUB_TEST_READS applied to
        this state): which turning points, which cycles, and whether any of those cycles is non-clean. Pure."""
        nonclean = set(self.non_clean_cycles)
        window = a_parts['window'] if branch == 'oscillatory' else b_parts['window']
        lo = window[0]

        def span_entry(first, what):
            s = _span(first, k)
            out_s = _span(first, lo - 1)
            bad = sorted(c for c in nonclean if out_s and out_s[0] <= c <= out_s[1])
            return {'reads': what, 'cycles': s, 'outside_window': out_s, 'non_clean_cycles_outside_window': bad,
                    'any_non_clean_outside_window': bool(bad)}

        def tp_entry(tp):
            t = tp[0]
            return {'t': t, 'kind': tp[1], 'Q_t': tp[2], 'outside_window': t < lo, 'non_clean': t in nonclean}

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
                'non_clean': (lo - 1) in nonclean}
            reads['no_sign_change_carry_in_step'] = {
                'reads': 'the sign of the last non-zero step before lo (judges the first non-zero step of the window)',
                'j_change': self.sign.j_change, 'carry_in_step_cycles': carry,
                'non_clean': bool(carry) and any(c in nonclean for c in carry)}
            tps_out = []
        cycles_out = set()
        for key in ('k0_no_boyd_lapse_span', 'sign_state_Q_span'):
            s = reads[key]['outside_window']
            if s:
                cycles_out.update(range(s[0], s[1] + 1))
        if branch == 'monotone':
            cycles_out.add(lo - 1)
        bad_tp = [x['t'] for x in tps_out if x['non_clean']]
        return {'branch': branch, 'k': k, 'k0': self.k0, 'window': list(window),
                'W': a_parts['W'] if branch == 'oscillatory' else self.l_mono,
                'window_all_clean': not self._non_clean_in(window),
                'reads': reads,
                'any_out_of_window_read': bool(cycles_out),
                'turning_points_outside_window': [x['t'] for x in tps_out],
                'non_clean_turning_points_outside_window': bad_tp,
                'non_clean_cycles_read_outside_window': sorted(c for c in cycles_out if c in nonclean),
                'any_non_clean_read_outside_window': any(c in nonclean for c in cycles_out),
                'enumeration': 'settling_criterion_v5.SUB_TEST_READS'}

    # ---- one cycle -----------------------------------------------------------------------------------------------------
    def observe(self, k, q, boyd, t_sum, all_clean):
        if not isinstance(all_clean, bool):
            raise TypeError(f'SettlingRuleV5: all_clean must be a bool at cycle {k}; got {all_clean!r}')
        if self.decision is not None:
            raise RuntimeError(f'SettlingRuleV5: cycle {k} observed after the decision ({self.decision.get("status")})')
        self.all_clean[k] = all_clean
        if not all_clean:
            self.non_clean_cycles.append(k)
        self._cycle_veto = None
        rec = super().observe(k, q, bool(boyd), t_sum)
        veto = self._cycle_veto
        rec.update({'all_clean_k': all_clean, 'vetoed': veto})
        if veto is not None and rec.get('decision') == 'continue':
            rec['decision'] = VETO_REASON
        dec = self.decision
        if dec is not None and dec.get('version') != VERSION:
            dec.update({'version': VERSION, 'reading': READING, 'N': self.n,
                        'N_definition': 'the first residual pass (version 2; no non-clean reset)',
                        'window_rule': READINGS['window_reading_a'],
                        'clean_rule': {'factor': CLEAN_FACTOR, 'metric_table': METRIC_TABLE, 'tolerances': TOLERANCES},
                        'non_clean_cycles': list(self.non_clean_cycles), 'vetoes': [dict(v) for v in self.vetoes],
                        'n_vetoes': len(self.vetoes), 'cap_ceiling': self.cap_ceiling})
            if dec['status'] == 'certified':
                oow = self.out_of_window_reads(k, dec['branch'], dec['certA_parts'], dec['certB_parts'])
                dec['window_all_clean'] = oow['window_all_clean']
                dec['out_of_window_reads'] = oow
                rec['out_of_window_reads'] = oow
            else:
                if veto is not None and VETO_REASON not in dec['reasons']:
                    dec['reasons'] = [r for r in FAILING_REASONS if r in set(dec['reasons']) | {VETO_REASON}]
                dec['certification_vetoed_at_cap'] = veto is not None
                dec['window_all_clean'] = None
                dec['out_of_window_reads'] = None
        return rec


def replay(q_by_cycle, boyd_by_cycle, t_by_cycle, all_clean_by_cycle, p_max, cap=None, cap_after_first_k0=None,
           cap_ceiling=None, last=None):
    """Runs the rule over cycles 1..last (default: the cap in force) from the records ({cycle: value}; a missing cycle
    reads as Q None / boyd False / t None / all_clean False). Stops at the decision. Returns (records, decision, rule)."""
    rule = SettlingRuleV5(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)
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
