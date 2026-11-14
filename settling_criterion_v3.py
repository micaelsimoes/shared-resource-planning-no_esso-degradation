"""The settling stop rule, VERSION 3 (P5.15 Addendum 58 Ruling 2; TASKS.md Addendum 58 Planner rulings (ii); Planner
task W132).

A NEW module: `settling_criterion.py` (version 1) and `settling_criterion_v2.py` (version 2) are NOT edited and stay
byte-identical; every certificate issued under them is untouched. This module imports version 2 (and through it version
1) and re-implements only what version 3 changes.

WHAT CHANGED FROM VERSION 2 (and nothing else):
  (1) READING ALPHA (the decision). Inside the rule, boyd_k is replaced by
          boyd_k_v3 = boyd_k AND all_optimal_k,
      all_optimal_k = the FINAL ACCEPTED attempt of EVERY block of cycle k -- the 12 TSO blocks, the 36 DSO blocks and
      the 3 ESSO solves -- exited IPOPT's "Optimal Solution Found" (classified by
      `p515_s44_campaign_harness.ipopt_exit_class(message) == 'optimal'`; the classification is the caller's). A
      non-Optimal cycle is therefore a LAPSE exactly as a Boyd lapse is in version 2: k0 := None and the sign state
      resets ("the count restarts", Addendum 58 Ruling 2). NO RETRY TIER (a solve-path change, Planner ruling (ii)).
  (2) N AND THE DYNAMIC CAP stay keyed on the run's first residual pass under the VERSION 2 definition (the first k
      with Q_k not None and boyd_k), not on the first reading-alpha pass: the holds (AA off, tail on, rho frozen) and
      the dynamic cap are the version-2 regime, unchanged. Both first passes are recorded: N (= the v2 first pass) and
      first_k0_alpha (the first k0 of the alpha rule).
  (3) READING GAMMA, REPORT-ONLY (never a decision): version 2 fed the ORIGINAL boyd_k (no reset on a non-Optimal
      cycle), with a branch verdict VETOED at any cycle whose certifying window -- the oscillatory window [k-W+1, k] or
      the monotone window [k-L+1, k] -- contains a non-Optimal cycle ("certification is blocked while a non-Optimal
      cycle lies in the window; there is no reset"). The oscillatory veto re-evaluates the monotone conjunction from
      version 2's own b_parts (the construction W131 validated). Its decision is recorded beside the alpha decision.
  (4) THE CAP CEILING IS PER CELL: `cap_ceiling` is a required argument (the spec's per-cell value), not the version-2
      code constant 300; a fixed cap above the given ceiling is refused.

CONSTANTS: those of version 2 unchanged (TAU, EPS0, K_EXCL, W_MIN, W_FACTOR, GAP_BOUND, DRIFT_WINDOW, CAP_AFTER_K0);
CAP_CEILING per cell (spec); P_MAX a spec constant (passed in).

THE ALGORITHM, for k = 1..CAP after Q_k, boyd_k (v2 definition), t_sum_k and all_optimal_k are known
(`SettlingRuleV3.observe`):
  0. if N is None and Q_k is not None and boyd_k: N := k; dynamic mode: CAP := min(k + CAP_AFTER_K0, CAP_CEILING).
  1-9. version 2's steps 1-9 with boyd_k_v3 = boyd_k AND all_optimal_k in place of boyd_k (so a non-Optimal cycle is a
     lapse: k0 := None, sign state reset); version 2's step 2 no longer sets N (step 0 did).
  G. the reading-gamma shadow observes (Q_k, boyd_k, t_sum_k, all_optimal_k) until its own decision (report-only).
Every per-cycle record is version 2's plus: boyd_k_v2, all_optimal_k, boyd_k_v3, first_residual_pass_v2,
first_k0_alpha, lapse_cause (on a lapse) and gamma (the shadow's cycle summary). The decision is version 2's with
version 3, the reading, N, first_k0_alpha, the non-Optimal cycles, the annotated lapse events and the gamma decision.

Stdlib only; no I/O; no model code. Numbers and comparisons exactly as version 2.
"""

import settling_criterion_v2 as SC2

__all__ = ['VERSION', 'SCHEMA', 'TAU', 'EPS0', 'K_EXCL', 'W_MIN', 'W_FACTOR', 'GAP_BOUND', 'DRIFT_WINDOW',
           'CAP_AFTER_K0', 'FAILING_REASONS', 'READINGS', 'OPTIMAL_CLASS', 'constants', 'SettlingRuleV3',
           'GammaShadow', 'replay']

VERSION = 3
SCHEMA = 'settling_criterion_v3'
TAU = SC2.TAU
EPS0 = SC2.EPS0
K_EXCL = SC2.K_EXCL
W_MIN = SC2.W_MIN
W_FACTOR = SC2.W_FACTOR
GAP_BOUND = SC2.GAP_BOUND
DRIFT_WINDOW = SC2.DRIFT_WINDOW
CAP_AFTER_K0 = SC2.CAP_AFTER_K0
FAILING_REASONS = SC2.FAILING_REASONS
OPTIMAL_CLASS = 'optimal'   # p515_s44_campaign_harness.ipopt_exit_class("... Optimal Solution Found ...")
GAMMA_VETO_REASON = 'gamma_non_optimal_cycle_in_window'
READINGS = dict(SC2.READINGS)
READINGS.update({
    'alpha_decision': ('Addendum 58 Ruling 2 / Planner ruling (ii): boyd_k_v3 = boyd_k AND all_optimal_k inside the '
                       'rule; a non-Optimal cycle is a lapse (k0 := None, sign state reset) -- "the count restarts"'),
    'all_optimal_definition': ('all_optimal_k: the final accepted attempt of every block of cycle k (12 TSO, 36 DSO, '
                               '3 ESSO) exited "Optimal Solution Found" (ipopt_exit_class == "optimal"); a block with no '
                               'result or no exit message is not Optimal'),
    'no_retry_tier': 'no retry tier (a solve-path change; Planner ruling (ii))',
    'n_and_cap_keyed_on_v2_first_pass': ('N and the dynamic cap are keyed on the run\'s first residual pass under the '
                                         'version-2 definition (Q_k not None and boyd_k); both first passes recorded'),
    'gamma_report_only': ('reading gamma, report-only: version 2 fed the version-2 boyd_k (no reset), the branch verdict '
                          'vetoed at any cycle whose certifying window contains a non-Optimal cycle; the oscillatory '
                          'veto re-evaluates the monotone conjunction from version 2\'s b_parts (W131\'s construction)'),
    'cap_ceiling_per_cell': 'the cap ceiling is a per-cell spec value (Planner ruling (iii)), not a code constant',
})


def constants(p_max):
    """The constants with their formulas (for the frozen spec). `p_max` is the spec constant."""
    out = SC2.constants(p_max)
    out.update({
        'VERSION': {'value': VERSION, 'formula': ('settling_criterion_v3 (settling_criterion.py = version 1, '
                                                  'settling_criterion_v2.py = version 2, both unchanged)')},
        'BOYD_K_V3': {'value': 'boyd_k AND all_optimal_k', 'formula': 'Addendum 58 Ruling 2; Planner ruling (ii)'},
        'OPTIMAL_CLASS': {'value': OPTIMAL_CLASS,
                          'formula': 'p515_s44_campaign_harness.ipopt_exit_class(message) == "optimal"'},
        'CAP': {'value': 'fixed N_old + 100 (gated) | min(first v2 k0 + 109, CAP_CEILING) (ungated)',
                'formula': 'Planner task W132; CAP_AFTER_K0 = 109; CAP_CEILING per cell (spec)'},
        'CAP_CEILING': {'value': 'per cell (the spec cell table)', 'formula': 'Planner ruling (iii) / task W132'},
    })
    return out


def _veto_evaluate(rule, k, nonopt):
    """Reading gamma's veto on a version-2 `evaluate(k)` (W131's construction, re-stated): returns the version-2 tuple
    with the vetoed verdicts and the list of vetoed branches."""
    cert_a, cert_b, a_parts, b_parts, reasons = SC2.SettlingRuleV2.evaluate(rule, k)
    vetoed = []
    if cert_a and any(a_parts['window'][0] <= c <= a_parts['window'][1] for c in nonopt):
        cert_a = False
        vetoed.append('oscillatory')
        cert_b = bool(b_parts['lo_ge_k0_plus_K_EXCL'] and b_parts['no_sign_change_in_window']
                      and b_parts['range_le_tau'] and b_parts['steps_decreasing']
                      and b_parts['last_step_times_L_le_tau'])
    if cert_b and any(b_parts['window'][0] <= c <= b_parts['window'][1] for c in nonopt):
        cert_b = False
        vetoed.append('monotone')
    if vetoed:
        reasons = list(reasons) + [GAMMA_VETO_REASON]
    return cert_a, cert_b, a_parts, b_parts, reasons, vetoed


class GammaShadow(SC2.SettlingRuleV2):
    """Reading gamma (REPORT-ONLY): version 2 fed the version-2 boyd_k, the branch verdict vetoed while a non-Optimal
    cycle lies in the certifying window. `observe(k, q, boyd, t_sum, all_optimal)`."""

    def __init__(self, p_max, cap=None, cap_after_first_k0=None, cap_ceiling=None):
        if cap_ceiling is None:
            raise ValueError('GammaShadow: cap_ceiling is a per-cell spec value and must be given')
        super().__init__(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)
        self.nonopt = set()
        self.vetoes = []
        self._last_vetoed = None

    def evaluate(self, k):
        cert_a, cert_b, a_parts, b_parts, reasons, vetoed = _veto_evaluate(self, k, self.nonopt)
        self._last_vetoed = vetoed or None
        if vetoed:
            self.vetoes.append({'cycle': k, 'branches': list(vetoed), 'window_a': a_parts['window'],
                                'window_b': b_parts['window']})
        return cert_a, cert_b, a_parts, b_parts, reasons

    def observe(self, k, q, boyd, t_sum, all_optimal):
        if not isinstance(all_optimal, bool):
            raise TypeError(f'GammaShadow: all_optimal must be a bool at cycle {k}; got {all_optimal!r}')
        if not all_optimal:
            self.nonopt.add(k)
        self._last_vetoed = None
        rec = super().observe(k, q, boyd, t_sum)
        rec['gamma_vetoed'] = self._last_vetoed
        return rec


class SettlingRuleV3(SC2.SettlingRuleV2):
    """Version 2 with reading alpha (the decision) and reading gamma (a report-only shadow). Exactly one of `cap`
    (fixed) or `cap_after_first_k0` (dynamic) is given; `cap_ceiling` (per cell) is REQUIRED. `observe(k, q, boyd,
    t_sum, all_optimal)` returns the cycle's full record and, on certification or at the cap, sets `self.decision`."""

    def __init__(self, p_max, cap=None, cap_after_first_k0=None, cap_ceiling=None):
        if cap_ceiling is None:
            raise ValueError('SettlingRuleV3: cap_ceiling is a per-cell spec value and must be given')
        super().__init__(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)
        self.first_k0_alpha = None
        self.all_optimal = {}
        self.non_optimal_cycles = []
        self.gamma = GammaShadow(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)
        self.gamma_decision = None
        self.gamma_decided_at = None

    def _gamma_summary(self):
        g = self.gamma
        if self.gamma_decision is not None:
            d = self.gamma_decision
            return {'status': d.get('status'), 'k_star': d.get('k_star'), 'k_cap': d.get('k_cap'),
                    'branch': d.get('branch'), 'k0': d.get('k0'), 'window': d.get('window'),
                    'decided_at_cycle': self.gamma_decided_at, 'n_vetoes': len(g.vetoes), 'vetoes': list(g.vetoes)}
        return {'status': 'undecided', 'last_cycle_observed': g.last_k, 'k0': g.k0, 'n_vetoes': len(g.vetoes),
                'vetoes': list(g.vetoes)}

    def observe(self, k, q, boyd, t_sum, all_optimal):
        if not isinstance(all_optimal, bool):
            raise TypeError(f'SettlingRuleV3: all_optimal must be a bool at cycle {k}; got {all_optimal!r}')
        if self.decision is not None:
            raise RuntimeError(f'SettlingRuleV3: cycle {k} observed after the decision ({self.decision.get("status")})')
        boyd_v2 = bool(boyd)
        # step 0: N and the dynamic cap keyed on the version-2 first residual pass
        first_v2 = False
        if self.n is None and q is not None and boyd_v2:
            if k > self.effective_cap():
                raise RuntimeError(f'SettlingRuleV3: cycle {k} beyond the cap {self.effective_cap()}')
            self.n = k
            if self.cap_mode == 'dynamic':
                self.cap = min(k + self.cap_after, self.cap_ceiling)
            first_v2 = True
        self.all_optimal[k] = all_optimal
        if not all_optimal:
            self.non_optimal_cycles.append(k)
        boyd_v3 = boyd_v2 and all_optimal
        n_lapses_before = len(self.lapses)
        k0_before = self.k0
        rec = super().observe(k, q, boyd_v3, t_sum)
        rec.update({'boyd_k_v2': boyd_v2, 'all_optimal_k': all_optimal, 'boyd_k_v3': boyd_v3,
                    'first_residual_pass_v2': first_v2, 'N': self.n, 'cap': self.cap})
        if self.k0 == k and k0_before is None and self.first_k0_alpha is None:
            self.first_k0_alpha = k
        rec['first_k0_alpha'] = self.first_k0_alpha
        if len(self.lapses) > n_lapses_before:
            cause = [c for c, hit in (('q_none', q is None), ('boyd', not boyd_v2), ('non_optimal', not all_optimal))
                     if hit]
            self.lapses[-1].update({'boyd_k_v2': boyd_v2, 'all_optimal_k': all_optimal, 'cause': cause})
            rec['lapse_cause'] = cause
        # G: the reading-gamma shadow (report-only) until its own decision
        if self.gamma_decision is None and k <= self.gamma.effective_cap():
            grec = self.gamma.observe(k, q, boyd_v2, t_sum, all_optimal)
            if self.gamma.decision is not None:
                self.gamma_decision = dict(self.gamma.decision)
                self.gamma_decided_at = k
            rec['gamma'] = {'k0': grec.get('k0'), 'decision': grec.get('decision'), 'vetoed': grec.get('gamma_vetoed'),
                            'certA': grec.get('certA'), 'certB': grec.get('certB'), 'reasons': grec.get('reasons')}
        else:
            rec['gamma'] = {'decided': True, 'status': (self.gamma_decision or {}).get('status'),
                            'decided_at_cycle': self.gamma_decided_at}
        if self.decision is not None and self.decision.get('version') != VERSION:
            self.decision.update({'version': VERSION, 'reading': 'alpha', 'N': self.n,
                                  'N_definition': 'the first residual pass under the version-2 definition',
                                  'first_k0_alpha': self.first_k0_alpha,
                                  'non_optimal_cycles': list(self.non_optimal_cycles),
                                  'lapse_events': [dict(x) for x in self.lapses],
                                  'cap_ceiling': self.cap_ceiling, 'gamma_report_only': self._gamma_summary()})
        return rec


def replay(q_by_cycle, boyd_by_cycle, t_by_cycle, all_optimal_by_cycle, p_max, cap=None, cap_after_first_k0=None,
           cap_ceiling=None, last=None):
    """Runs the rule over cycles 1..last (default: the cap in force) from the records ({cycle: value}; a missing cycle
    reads as Q None / boyd False / t None / all_optimal False). Stops at the decision. Returns (records, decision,
    rule)."""
    rule = SettlingRuleV3(p_max, cap=cap, cap_after_first_k0=cap_after_first_k0, cap_ceiling=cap_ceiling)
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
