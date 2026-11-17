"""
P5.15 Addendum 39 follow-up (task W46) -- the COORDINATED interface-dispersion decomposition as a
COMMITTED, ZERO-SOLVE script (CLAUDE.md: preserve the formula, not only the inputs). It replaces
the W44 scratch computation (never committed) that split the aggregate dispersion into a
market-scenario part and an operation-scenario part and measured the cross-scenario
redistribution volumes; it adds the covariance (settlement deviation) and the per-hour
market-arbitrage threshold alpha_arb,t and its test.

It reads ONLY committed artifacts of `p515_s51_2x2_limit_gate.py` runs (every input file is
verified against its run's committed manifest_sha256.json before use) and never builds or
solves a model. Run as a script, a SolveProfileGuard with NO permitted call site is installed
for the whole run and verified at exactly 0 solves. Imported as a module (the gate imports the
pure formula functions below so the gate's monotonicity metric and this script's are ONE
definition), it installs nothing.

NOTATION, per DSO (year, day) block, block-local, probability-weighted, UNWEIGHTED by
year/day/discount; one period = one hour, so a sum over hours of MW is MWh and of MW^2 is MW^2 h:
  s = (s_m, s_o), omega_s = prob_market[s_m] * prob_operation[s_o]   (the block's own vector)
  d_{s,t} = p_int_{s,t} - pbar_t (MW), production's `_get_local_interface_dispersion`
            per_scenario d_p_mw (pbar_t = E_s p_int_{s,t} by the coupled definition, so
            E_s d_{s,t} = 0 up to the solver tolerance)
  omega_m = sum_{s_o} omega_{(m, s_o)};  dbar_{m,t} = sum_{s_o} omega_{(m,s_o)} d_{(m,s_o),t} / omega_m
  pi_{m,t}  market price of scenario m at hour t;  pibar_t = sum_m omega_m pi_{m,t}

FORMULAS (per block; summed over hours; `aggregate` sums blocks):
  sum_omega_d2       = sum_t sum_s omega_s d_{s,t}^2                     (MW^2 h)  -- Sigma omega d^2
  market_part        = sum_t sum_m omega_m dbar_{m,t}^2                  (MW^2 h)
  operation_part     = sum_t sum_s omega_s (d_{s,t} - dbar_{m(s),t})^2   (MW^2 h), computed
                       directly; the identity sum_omega_d2 = market_part + operation_part is
                       checked, not assumed
  E_abs_d            = sum_t sum_s omega_s |d_{s,t}|                     (MWh)
  E_abs_d_market     = sum_t sum_m omega_m |dbar_{m,t}|                  (MWh)
  covariance         = sum_t sum_m omega_m (pi_{m,t} - pibar_t) dbar_{m,t}   (currency)
                     = sum_t Cov_s(pi_t, p_int_t) EXACTLY (sum_m omega_m (pi_m - pibar) = 0 by the
                       definition of pibar, so pbar_t drops out) -- the settlement DEVIATION
                       part of `interface_energy_settlement` at weight 1 (Addendum 38 (C)); it is
                       cross-checked against the captured `settlement.deviation` of each block.
                       Negative = the DSO imports less where the price is high (it EARNS it).
  pooled_rms_mw      = sqrt(sum_omega_d2 / (n_blocks * n_hours))  (aggregate only)
Redistribution volume of a quantity x (flex DOWN leg, flex UP leg, RES curtailment), from the
W44 mechanism / coordination capture (absent for arms run before W44):
  redis(x)           = sum_t sum_s omega_s |x_{s,t} - sum_s' omega_s' x_{s',t}|      (MWh)

THE MARKET-ARBITRAGE THRESHOLD (per block, per hour). At fixed pbar_t, move v MWh of
probability-weighted import from market scenario hi to lo (dbar_hi = -v/omega_hi,
dbar_lo = +v/omega_lo, so E d_t is unchanged and the AL term on pbar is untouched). The
settlement changes by -v (pi_hi - pi_lo) -- a gain of v (pi_hi - pi_lo) -- and the row 18
P charge rises by alpha * pibar_t * (omega_hi |dbar_hi| + omega_lo |dbar_lo|) = 2 alpha pibar_t v.
With a FREE physical lever the shift pays iff alpha < alpha_arb,t, where
    alpha_arb,t = (max_m pi_{m,t} - min_m pi_{m,t}) / (2 * pibar_t)          (pibar_t > 0)
(with two equiprobable market scenarios this is |pi_{m,t} - pibar_t| / pibar_t). It is a
NECESSARY-condition threshold: the lever's own cost (flexibility, curtailment, losses), the Q
leg of row 18, the operation-scenario part and intra-day coupling (the fl_reg day balance moves
deviation between hours) are all ignored, so alpha >= alpha_arb,t predicts "holds" in that hour
for the market part, while alpha < alpha_arb,t predicts only that arbitrage is not unprofitable.
The W44 report's own alpha_arb test was never written to a file; this definition is the one now
committed, derived as above, and is the one every number here uses.
PER-HOUR TEST, every arm, every (block, hour) with pibar_t > 0: predicted "deviates" iff
alpha < alpha_arb,t; observed "deviates (market)" iff omega-weighted |dbar| at that hour,
sum_m omega_m |dbar_{m,t}|, exceeds HOUR_DEV_TOL_MWH; observed "deviates (total)" iff
sum_s omega_s |d_{s,t}| does. The 2 x 2 confusion counts for both are reported, together with the
market deviation volume in predicted-hold hours (what the threshold cannot explain), and the
same on the subset of hours whose market part deviates in the alpha = 0 arm.

AGGREGATE alpha* (Planner ruling, W46): the monotonicity and the threshold are stated on
sum_omega_d2 summed over every DSO block. AGG_TOL = n_blocks * n_hours * DISPERSION_ZERO_TOL_MW^2
-- the aggregate value at which every block's RMS equals the per-block zero tolerance, i.e.
pooled_rms_mw <= DISPERSION_ZERO_TOL_MW. alpha*_agg = smallest tested alpha with
sum_omega_d2 <= AGG_TOL; bracket = (largest tested alpha below it that is above AGG_TOL, alpha*].
Non-monotone iff sum_omega_d2(alpha_k) > sum_omega_d2(alpha_{k-1}) + AGG_TOL for consecutive
tested alphas. The per-DSO max-over-blocks RMS is reported beside it for continuity only.

INPUTS. `--sweeps` names sweep-mode runs of the gate (W44 or later: mechanism_<arm>.json exists);
`--limit-arms LABEL:ARM` names arms of pre-W44 runs (dispersion detail only: no redistribution
volumes, no captured prices). For those, the scenario probabilities and prices are taken from the
sweep arms of the SAME derived instance (derived-case sha256 and scenario checksum equal, checked)
-- they are instance data, and are checked identical across every sweep arm first -- and the
borrowed probabilities are checked against production's own per-block RMS
(sum_omega_d2 = rms_mw^2 * n_hours) before use.

EXACT COMMAND (repo root; attached; both streams captured):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s51_coordinated_decomposition.py --label <write-once-label> \\
        --sweeps alpha_sweep_r1 <later sweep labels> --limit-arms limit_r1:large \\
        > data/SRP1/Results/P515S51/coordinated_decomposition_<label>.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S51/coordinated_decomposition/<label>/analysis.json
Exit 0 on success, 1 on error (including a non-monotone aggregate: the result is still written),
2 on a precondition refusal.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))

STAGE = ('P5.15 Addendum 39 follow-up W46 -- coordinated dispersion decomposition '
         '(market / operation split, covariance, redistribution volumes, alpha_arb test, aggregate alpha*)')
SCHEMA = 'p515_s51_coordinated_decomposition_v1'
GATE_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S51', '2x2_limit_gate')
OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S51', 'coordinated_decomposition')

# Declared constants (operational definitions, not fixture literals). Each input run's own
# declared value is checked equal to these before use.
DISPERSION_ZERO_TOL_MW = 1.0e-2     # = p515_s51_2x2_limit_gate.DISPERSION_ZERO_TOL_MW
HOUR_DEV_TOL_MWH = 1.0e-2           # = p515_s51_2x2_limit_gate.HOUR_DEV_TOL_MWH
IDENTITY_REL_TOL = 1.0e-9           # relative tolerance of the recorded identity checks
PROBABILITY_SUM_TOL = 1.0e-12
COVARIANCE_ABS_TOL = 1.0e-6         # currency per block: covariance vs captured settlement deviation.
                                    # The identity is EXACT (sum_m omega_m (pi_m - pibar) = 0 by the
                                    # definition of pibar, so pbar drops out), so only rounding remains.
MECHANISM_D_ABS_TOL_MW = 1.0e-9     # W44 mechanism d_p_mw vs production's dispersion d_p_mw: the same
                                    # quantity read off the same model, differing only in rounding.

FORMULAS = {
    'omega_s': 'prob_market[s_m] * prob_operation[s_o] (the block own vector)',
    'd_st': 'p_int[s,t] - pbar_t (MW), production _get_local_interface_dispersion per_scenario d_p_mw',
    'dbar_mt': 'sum_{s_o} omega_(m,s_o) d_(m,s_o),t / omega_m',
    'sum_omega_d2': 'sum_t sum_s omega_s d_st^2 (MW^2 h)',
    'market_part': 'sum_t sum_m omega_m dbar_mt^2 (MW^2 h)',
    'operation_part': 'sum_t sum_s omega_s (d_st - dbar_m(s)t)^2 (MW^2 h), computed directly',
    'E_abs_d': 'sum_t sum_s omega_s |d_st| (MWh)',
    'E_abs_d_market': 'sum_t sum_m omega_m |dbar_mt| (MWh)',
    'covariance': 'sum_t sum_m omega_m (pi_mt - pibar_t) dbar_mt (currency) = settlement deviation part at weight 1',
    'pooled_rms_mw': 'sqrt(sum_omega_d2 / (n_blocks * n_hours))',
    'redis_x': 'sum_t sum_s omega_s |x_st - sum_s2 omega_s2 x_s2t| (MWh); x = flex_p_down_mw, flex_p_up_mw, curt_res_mw',
    'alpha_arb_t': '(max_m pi_mt - min_m pi_mt) / (2 pibar_t), pibar_t > 0',
    'hour_test': ('predicted deviates iff alpha < alpha_arb_t; observed market deviates iff '
                  'sum_m omega_m |dbar_mt| > HOUR_DEV_TOL_MWH; observed total deviates iff '
                  'sum_s omega_s |d_st| > HOUR_DEV_TOL_MWH'),
    'agg_tol': 'n_blocks * n_hours * DISPERSION_ZERO_TOL_MW^2 (MW^2 h)',
    'alpha_star_agg': ('smallest tested alpha with aggregate sum_omega_d2 <= agg_tol; bracket = (largest '
                       'tested alpha below it with sum_omega_d2 > agg_tol, alpha*]'),
    'non_monotone': 'sum_omega_d2(alpha_k) > sum_omega_d2(alpha_(k-1)) + agg_tol, consecutive tested alphas',
}


# ------------------------------------------------------------------------------------------
# pure formula functions (json / math only) -- imported by p515_s51_2x2_limit_gate.py
# ------------------------------------------------------------------------------------------

def _market_of(scenario_key):
    return scenario_key.split('_')[0]


def block_decomposition(d_by_scenario, prob_by_scenario, pi_by_market=None, pibar=None):
    """The per-hour and total decomposition of ONE block (module docstring formulas).
    `d_by_scenario`: {'<s_m>_<s_o>': [d_t (MW)]}; `prob_by_scenario`: {'<s_m>_<s_o>': omega_s};
    `pi_by_market`: optional per-hour list of [pi_{m,t} for m = 0, 1, ...]; `pibar`: per-hour list."""
    keys = sorted(d_by_scenario)
    if sorted(prob_by_scenario) != keys:
        raise ValueError(f'scenario keys differ: {keys} vs {sorted(prob_by_scenario)}')
    n = len(d_by_scenario[keys[0]])
    omega_m = {}
    for k in keys:
        omega_m[_market_of(k)] = omega_m.get(_market_of(k), 0.0) + prob_by_scenario[k]
    hours = {name: [0.0] * n for name in ('sum_omega_d2', 'market_part', 'operation_part',
                                           'E_abs_d', 'E_abs_d_market', 'E_d_signed')}
    hours['covariance'] = [0.0] * n if pi_by_market is not None else None
    for t in range(n):
        dbar = {m: 0.0 for m in omega_m}
        for k in keys:
            dbar[_market_of(k)] += prob_by_scenario[k] * d_by_scenario[k][t]
        for m in dbar:
            dbar[m] /= omega_m[m]
        for k in keys:
            w, d = prob_by_scenario[k], d_by_scenario[k][t]
            hours['sum_omega_d2'][t] += w * d * d
            hours['operation_part'][t] += w * (d - dbar[_market_of(k)]) ** 2
            hours['E_abs_d'][t] += w * abs(d)
            hours['E_d_signed'][t] += w * d
        for m, w in omega_m.items():
            hours['market_part'][t] += w * dbar[m] ** 2
            hours['E_abs_d_market'][t] += w * abs(dbar[m])
            if pi_by_market is not None:
                hours['covariance'][t] += w * (pi_by_market[t][int(m)] - pibar[t]) * dbar[m]
    totals = {name: (sum(v) if v is not None else None) for name, v in hours.items()}
    return {'n_hours': n, 'omega_market': omega_m, 'hours': hours, 'totals': totals,
            'identity_residual_market_plus_operation': (totals['sum_omega_d2'] - totals['market_part']
                                                        - totals['operation_part'])}


def aggregate(blocks, zero_tol_mw=DISPERSION_ZERO_TOL_MW):
    """Sum block decompositions ({'DSO:<node>:<year>:<day>': block_decomposition(...)}) over all
    blocks and per DSO; the aggregate tolerance and pooled RMS (module docstring)."""
    names = ('sum_omega_d2', 'market_part', 'operation_part', 'E_abs_d', 'E_abs_d_market', 'covariance')
    total = {k: 0.0 for k in names}
    per_dso = {}
    n_hours = None
    for key, blk in blocks.items():
        node = key.split(':')[1]
        entry = per_dso.setdefault(node, {k: 0.0 for k in names})
        for k in names:
            value = blk['totals'].get(k)
            if value is None:
                total[k] = entry[k] = None
                continue
            if total[k] is not None:
                total[k] += value
            if entry[k] is not None:
                entry[k] += value
        if n_hours is None:
            n_hours = blk['n_hours']
        elif n_hours != blk['n_hours']:
            raise ValueError('blocks with different horizons')
    n_blocks = len(blocks)
    agg_tol = n_blocks * (n_hours or 0) * zero_tol_mw ** 2
    for entry in [total] + list(per_dso.values()):
        entry['market_share_of_sum_omega_d2'] = (entry['market_part'] / entry['sum_omega_d2']
                                                 if entry['sum_omega_d2'] else None)
    return {'all_dso': total, 'per_dso': per_dso, 'n_blocks': n_blocks, 'n_hours': n_hours,
            'agg_tol_mw2h': agg_tol,
            'pooled_rms_mw': math.sqrt(total['sum_omega_d2'] / (n_blocks * n_hours)) if n_blocks else None}


def alpha_star_aggregate(value_by_alpha, agg_tol):
    """alpha*_agg and its bracket, and the non-monotone steps (module docstring)."""
    alphas = sorted(value_by_alpha)
    at_or_below = [a for a in alphas if value_by_alpha[a] <= agg_tol]
    above = [a for a in alphas if value_by_alpha[a] > agg_tol]
    star = at_or_below[0] if at_or_below else None
    below = [a for a in above if star is None or a < star]
    steps = [{'from_alpha': a0, 'to_alpha': a1, 'from': value_by_alpha[a0], 'to': value_by_alpha[a1],
              'non_monotone': value_by_alpha[a1] > value_by_alpha[a0] + agg_tol}
             for a0, a1 in zip(alphas, alphas[1:])]
    return {'sum_omega_d2_by_alpha': {str(a): value_by_alpha[a] for a in alphas}, 'agg_tol_mw2h': agg_tol,
            'alpha_star': star, 'bracket': [below[-1] if below else None, star],
            'above_tol_after_alpha_star': [a for a in above if star is not None and a > star],
            'steps': steps, 'monotone_non_increasing': not any(s['non_monotone'] for s in steps)}


def redistribution_volumes(mechanism_block, coordination_block):
    """redis(x) for the flex DOWN leg, the flex UP leg and RES curtailment of one block (MWh)."""
    per_scenario = mechanism_block['per_scenario']
    series = {'flex_down': {k: e['series']['flex_p_down_mw'] for k, e in per_scenario.items()},
              'flex_up': {k: e['series']['flex_p_up_mw'] for k, e in per_scenario.items()},
              'curt_res': {k: coordination_block['curtailment'][k]['curt_res_mw'] for k in per_scenario}}
    prob = {k: e['probability'] for k, e in per_scenario.items()}
    n = len(next(iter(per_scenario.values()))['series']['d_p_mw'])
    out = {}
    for name, x in series.items():
        total = 0.0
        for t in range(n):
            mean = sum(prob[k] * x[k][t] for k in prob)
            total += sum(prob[k] * abs(x[k][t] - mean) for k in prob)
        out[name] = total
    return out


def alpha_arb_by_hour(pi_by_market, pibar):
    """alpha_arb,t per hour (None where pibar_t <= 0)."""
    return [((max(pi) - min(pi)) / (2.0 * pb) if pb > 0.0 else None) for pi, pb in zip(pi_by_market, pibar)]


# ------------------------------------------------------------------------------------------
# script
# ------------------------------------------------------------------------------------------

def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    try:
        return subprocess.run(['git'] + args, capture_output=True, text=True,
                              check=True, cwd=REPO).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f'<git failed: {error}>'


class _Run:
    """One committed gate run; every file read is verified against the run's manifest."""

    def __init__(self, label, inventory):
        self.label = label
        self.dir = os.path.join(GATE_ROOT, label)
        self.inventory = inventory
        with open(os.path.join(self.dir, 'manifest_sha256.json')) as handle:
            self.manifest = json.load(handle)
        inventory[os.path.relpath(os.path.join(self.dir, 'manifest_sha256.json'), REPO)] = _sha256_file(
            os.path.join(self.dir, 'manifest_sha256.json'))
        self.gate = self.load('gate.json')
        self.launch = self.load('launch.json')

    def load(self, rel_in_run):
        path = os.path.join(self.dir, rel_in_run)
        rel = os.path.relpath(path, REPO)
        actual = _sha256_file(path)
        if self.manifest.get(rel) != actual:
            raise RuntimeError(f'{rel}: sha256 {actual} does not match the manifest ({self.manifest.get(rel)})')
        self.inventory[rel] = actual
        with open(path) as handle:
            return json.load(handle)

    def identity(self):
        return {'derived_case_sha256': self.launch['derived_case']['sha256'],
                'scenario_checksum': self.launch['scenario_checksum'],
                'git_head_of_run': self.launch['git_head'], 'script_sha256_of_run': self.launch['script_sha256'],
                'cycles': self.launch['cycles'],
                'thresholds_declared_before_the_run': self.launch['thresholds_declared_before_the_run']}


def _d_by_scenario(detail_block):
    return {k: v['d_p_mw'] for k, v in detail_block['per_scenario'].items()}


def _arm_entry(run, arm, record, prob_by_block, prices_by_block, mechanism, coordination):
    """Everything this script reports for one arm."""
    detail = run.load(os.path.relpath(os.path.join(REPO, record['dispersion_detail_path']), run.dir))
    blocks, checks = {}, {'rms_identity_max_rel': 0.0, 'mechanism_d_max_abs_diff_mw': None,
                          'covariance_vs_captured_settlement_max_abs': None, 'E_d_signed_max_abs_mwh': 0.0,
                          'market_plus_operation_identity_max_abs': 0.0}
    redis = {} if mechanism is not None else None
    for key, det in detail.items():
        if not key.startswith('DSO:') or det is None:
            continue
        d = _d_by_scenario(det)
        prices = prices_by_block[key]
        blk = block_decomposition(d, prob_by_block[key], prices['pi_by_market'], prices['pibar'])
        blocks[key] = blk
        n = blk['n_hours']
        production = det['p']['rms_mw'] ** 2 * n
        rel = abs(blk['totals']['sum_omega_d2'] - production) / max(production, 1e-300) if production else abs(
            blk['totals']['sum_omega_d2'])
        checks['rms_identity_max_rel'] = max(checks['rms_identity_max_rel'], rel)
        checks['E_d_signed_max_abs_mwh'] = max(checks['E_d_signed_max_abs_mwh'],
                                               max(abs(v) for v in blk['hours']['E_d_signed']))
        checks['market_plus_operation_identity_max_abs'] = max(
            checks['market_plus_operation_identity_max_abs'], abs(blk['identity_residual_market_plus_operation']))
        if mechanism is not None:
            mech = mechanism[key]
            diff = max(abs(mech['per_scenario'][k]['series']['d_p_mw'][t] - d[k][t]) for k in d for t in range(n))
            checks['mechanism_d_max_abs_diff_mw'] = max(checks['mechanism_d_max_abs_diff_mw'] or 0.0, diff)
            captured = coordination[key]['settlement']['deviation']
            cdiff = abs(blk['totals']['covariance'] - captured)
            checks['covariance_vs_captured_settlement_max_abs'] = max(
                checks['covariance_vs_captured_settlement_max_abs'] or 0.0, cdiff)
            redis[key] = redistribution_volumes(mech, coordination[key])
    agg = aggregate(blocks)
    redis_total = None
    if redis is not None:
        redis_total = {'all_dso': {}, 'per_dso': {}}
        for key, r in redis.items():
            node = key.split(':')[1]
            for name, v in r.items():
                redis_total['all_dso'][name] = redis_total['all_dso'].get(name, 0.0) + v
                redis_total['per_dso'].setdefault(node, {})
                redis_total['per_dso'][node][name] = redis_total['per_dso'][node].get(name, 0.0) + v
    per_dso_rms = {}
    for key, det in detail.items():
        if key.startswith('DSO:') and det is not None:
            node = key.split(':')[1]
            per_dso_rms[node] = max(per_dso_rms.get(node, 0.0), det['p']['rms_mw'])
    return {
        'run': run.label, 'arm': arm, 'alpha': record['alpha'],
        'has_mechanism_capture': mechanism is not None,
        'cycles_run': record.get('cycles_run'), 'solves_in_arm': record.get('solves_in_arm'),
        'recourse_net_operational': record.get('recourse'),
        'gross_operational_cost': record.get('gross_operational_cost'),
        'rule_ten_terminal_step_over_threshold': record.get('rule_ten_terminal_step_over_threshold'),
        'terminal_objective_change_abs': record.get('terminal_objective_change_abs'),
        'terminal_objective_tolerance': record.get('terminal_objective_tolerance'),
        'aggregate': agg,
        'per_dso_rms_mw_max_over_blocks': per_dso_rms,
        'row18_total_charge_all_dso': sum(det['row18_charge'] for key, det in detail.items()
                                          if key.startswith('DSO:') and det is not None),
        'redistribution_volumes_mwh': redis_total,
        'checks': checks,
        '_blocks': blocks,
    }


def _hour_test(entry, alpha_arb, reference_blocks):
    """The per-hour alpha_arb test of one arm (module docstring)."""
    alpha = entry['alpha']

    def empty():
        return {'pred_dev_obs_dev': 0, 'pred_dev_obs_hold': 0, 'pred_hold_obs_dev': 0, 'pred_hold_obs_hold': 0,
                'n_tested': 0, 'market_E_abs_d_in_pred_hold_hours_mwh': 0.0,
                'market_E_abs_d_in_pred_dev_hours_mwh': 0.0}

    out = {'market': empty(), 'total': empty(), 'market_on_alpha0_market_deviating_hours': empty(),
           'pred_hold_obs_market_dev_hours': []}
    for key, blk in entry['_blocks'].items():
        for t, a_t in enumerate(alpha_arb[key]):
            if a_t is None:
                continue
            predicted = alpha < a_t
            m_obs = blk['hours']['E_abs_d_market'][t]
            obs = {'market': m_obs > HOUR_DEV_TOL_MWH, 'total': blk['hours']['E_abs_d'][t] > HOUR_DEV_TOL_MWH}
            targets = [('market', obs['market']), ('total', obs['total'])]
            if reference_blocks is not None and reference_blocks[key]['hours']['E_abs_d_market'][t] > HOUR_DEV_TOL_MWH:
                targets.append(('market_on_alpha0_market_deviating_hours', obs['market']))
            for name, observed in targets:
                c = out[name]
                c['n_tested'] += 1
                c[f"pred_{'dev' if predicted else 'hold'}_obs_{'dev' if observed else 'hold'}"] += 1
                if name != 'total':
                    c[f"market_E_abs_d_in_pred_{'dev' if predicted else 'hold'}_hours_mwh"] += m_obs
            if not predicted and obs['market']:
                out['pred_hold_obs_market_dev_hours'].append(
                    {'block': key, 'hour': t, 'alpha_arb_t': a_t, 'E_abs_d_market_mwh': m_obs,
                     'E_abs_d_mwh': blk['hours']['E_abs_d'][t],
                     'alpha_over_alpha_arb_t': alpha / a_t})
    for name in ('market', 'total', 'market_on_alpha0_market_deviating_hours'):
        c = out[name]
        c['agreement'] = (c['pred_dev_obs_dev'] + c['pred_hold_obs_hold']) / c['n_tested'] if c['n_tested'] else None
    out['pred_hold_obs_market_dev_hours'].sort(key=lambda r: -r['E_abs_d_market_mwh'])
    return out


def main():
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument('--label', required=True, help='write-once output label')
    parser.add_argument('--sweeps', nargs='+', required=True,
                        help='committed sweep-mode gate run labels (with mechanism_<arm>.json)')
    parser.add_argument('--limit-arms', nargs='*', default=[],
                        help='LABEL:ARM of pre-W44 gate runs (dispersion detail only)')
    args = parser.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.label)
    if os.path.exists(out_dir):
        print(f'REFUSED: output directory already exists (write-once): {out_dir}', file=sys.stderr)
        return 2

    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    os.chdir(REPO)
    from p513_solve_profile_guard import SolveProfileGuard
    guard = SolveProfileGuard((), label='P5.15 W46 coordinated decomposition -- zero solves').install()
    try:
        inventory = {}
        runs = {label: _Run(label, inventory) for label in args.sweeps}
        for spec in args.limit_arms:
            label = spec.split(':')[0]
            if label not in runs:
                runs[label] = _Run(label, inventory)
        identities = {label: run.identity() for label, run in runs.items()}
        instance = {(i['derived_case_sha256'], i['scenario_checksum']) for i in identities.values()}
        if len(instance) != 1:
            raise RuntimeError(f'input runs are not one instance: {identities}')
        for label, i in identities.items():
            th = i['thresholds_declared_before_the_run']
            if th.get('dispersion_zero_tol_mw') != DISPERSION_ZERO_TOL_MW or (
                    'hour_dev_tol_mwh' in th and th['hour_dev_tol_mwh'] != HOUR_DEV_TOL_MWH):
                raise RuntimeError(f'{label}: declared tolerances differ from this script: {th}')

        # ---- instance data from the sweep arms: probabilities and prices, checked identical
        mech_by_arm, prob_by_block, prices_by_block = {}, {}, {}
        for label in args.sweeps:
            run = runs[label]
            for arm, record in run.gate['arms'].items():
                mpath = os.path.join(REPO, record['mechanism_path'])
                cap = run.load(os.path.relpath(mpath, run.dir))
                mech_by_arm[(label, arm)] = cap
                for key, mech in cap['mechanism'].items():
                    prob = {k: e['probability'] for k, e in mech['per_scenario'].items()}
                    hours = cap['coordination'][key]['hours']
                    prices = {'pi_by_market': [h['pi_by_market_scenario'] for h in hours],
                              'pibar': [h['pibar'] for h in hours]}
                    if abs(sum(prob.values()) - 1.0) > PROBABILITY_SUM_TOL:
                        raise RuntimeError(f'{label} {arm} {key}: probabilities sum to {sum(prob.values())}')
                    if key in prob_by_block and (prob_by_block[key] != prob or prices_by_block[key] != prices):
                        raise RuntimeError(f'{label} {arm} {key}: probabilities or prices differ across arms')
                    prob_by_block[key], prices_by_block[key] = prob, prices

        # ---- per arm
        entries = []
        for label in args.sweeps:
            run = runs[label]
            for arm, record in run.gate['arms'].items():
                cap = mech_by_arm[(label, arm)]
                entries.append(_arm_entry(run, arm, record, prob_by_block, prices_by_block,
                                          cap['mechanism'], cap['coordination']))
        for spec in args.limit_arms:
            label, arm = spec.split(':')
            run = runs[label]
            entries.append(_arm_entry(run, arm, run.gate['arms'][arm], prob_by_block, prices_by_block, None, None))

        # ---- duplicate alphas across runs: bitwise agreement of production's per-block d
        entries.sort(key=lambda e: e['alpha'])
        by_alpha, duplicates = {}, []
        for e in entries:
            if e['alpha'] in by_alpha:
                first = by_alpha[e['alpha']]
                duplicates.append({'alpha': e['alpha'], 'first': f"{first['run']}:{first['arm']}",
                                   'second': f"{e['run']}:{e['arm']}",
                                   'sum_omega_d2_bitwise_equal': (first['aggregate']['all_dso']['sum_omega_d2']
                                                                  == e['aggregate']['all_dso']['sum_omega_d2'])})
            else:
                by_alpha[e['alpha']] = e
        unique = [by_alpha[a] for a in sorted(by_alpha)]

        # ---- checks gate the script (exit 1 if any fails; the result is still written)
        failures = []
        for e in entries:
            c = e['checks']
            if c['rms_identity_max_rel'] > IDENTITY_REL_TOL:
                failures.append(f"{e['run']}:{e['arm']} sum_omega_d2 vs production rms: rel {c['rms_identity_max_rel']}")
            if c['mechanism_d_max_abs_diff_mw'] is not None and c['mechanism_d_max_abs_diff_mw'] > MECHANISM_D_ABS_TOL_MW:
                failures.append(f"{e['run']}:{e['arm']} mechanism d differs from production d by "
                                f"{c['mechanism_d_max_abs_diff_mw']} MW")
            if (c['covariance_vs_captured_settlement_max_abs'] is not None
                    and c['covariance_vs_captured_settlement_max_abs'] > COVARIANCE_ABS_TOL):
                failures.append(f"{e['run']}:{e['arm']} covariance vs captured settlement deviation: "
                                f"{c['covariance_vs_captured_settlement_max_abs']}")
        for dup in duplicates:
            if not dup['sum_omega_d2_bitwise_equal']:
                failures.append(f'duplicate alpha not bitwise equal: {dup}')

        # ---- aggregate alpha*, monotonicity, alpha_arb test
        agg_tols = {e['aggregate']['agg_tol_mw2h'] for e in unique}
        if len(agg_tols) != 1:
            raise RuntimeError(f'arms differ in block count / horizon: {agg_tols}')
        agg_tol = agg_tols.pop()
        star = alpha_star_aggregate({e['alpha']: e['aggregate']['all_dso']['sum_omega_d2'] for e in unique}, agg_tol)
        alpha_arb = {key: alpha_arb_by_hour(p['pi_by_market'], p['pibar']) for key, p in prices_by_block.items()}
        arb_values = [a for v in alpha_arb.values() for a in v if a is not None]
        reference = by_alpha.get(0.0)
        hour_tests = {str(e['alpha']): _hour_test(e, alpha_arb, reference['_blocks'] if reference else None)
                      for e in unique}

        table = []
        for e in unique:
            a = e['aggregate']
            table.append({
                'alpha': e['alpha'], 'source': f"{e['run']}:{e['arm']}",
                'per_dso_rms_mw_max_over_blocks': e['per_dso_rms_mw_max_over_blocks'],
                'E_abs_d_mwh': a['all_dso']['E_abs_d'], 'E_abs_d_market_mwh': a['all_dso']['E_abs_d_market'],
                'sum_omega_d2_mw2h': a['all_dso']['sum_omega_d2'], 'market_part_mw2h': a['all_dso']['market_part'],
                'operation_part_mw2h': a['all_dso']['operation_part'],
                'market_share': a['all_dso']['market_share_of_sum_omega_d2'],
                'pooled_rms_mw': a['pooled_rms_mw'], 'covariance': a['all_dso']['covariance'],
                'row18_total_charge_all_dso': e['row18_total_charge_all_dso'],
                'redistribution_volumes_mwh': (e['redistribution_volumes_mwh'] or {}).get('all_dso'),
                'recourse_net_operational': e['recourse_net_operational'],
                'rule_ten_terminal_step_over_threshold': e['rule_ten_terminal_step_over_threshold'],
            })
        guard_failures = guard.verify(0)
        if guard_failures:
            failures.append(f'solve guard: {guard_failures}')
        payload = {
            'schema': SCHEMA, 'stage': STAGE, 'label': args.label, 'created_utc': _utc(),
            'argv': sys.argv, 'interpreter': sys.executable,
            'script': os.path.basename(__file__), 'script_sha256': _sha256_file(os.path.abspath(__file__)),
            'git_head': _git(['rev-parse', 'HEAD']),
            'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
            'instance': {'derived_case_sha256': next(iter(instance))[0], 'scenario_checksum': next(iter(instance))[1],
                         'candidate': 'x = 0 (no shared-ESS investment) at every active node (per the gate)'},
            'inputs': {'runs': identities, 'files_sha256': inventory},
            'declared_constants': {'DISPERSION_ZERO_TOL_MW': DISPERSION_ZERO_TOL_MW,
                                   'HOUR_DEV_TOL_MWH': HOUR_DEV_TOL_MWH, 'IDENTITY_REL_TOL': IDENTITY_REL_TOL,
                                   'COVARIANCE_ABS_TOL': COVARIANCE_ABS_TOL,
                                   'MECHANISM_D_ABS_TOL_MW': MECHANISM_D_ABS_TOL_MW,
                                   'PROBABILITY_SUM_TOL': PROBABILITY_SUM_TOL},
            'formulas': FORMULAS,
            'objective_convention': ('block-local, probability-weighted, UNWEIGHTED by year/day/discount, '
                                     'summed over DSO (year, day) blocks; covariance and row 18 charge in '
                                     'block-local currency; recourse = net_operational_recourse as the gate records it'),
            'not_a_result': ('two-cycle ADMM arms, NOT converged: every quantity is the state after cycle 2 of '
                             'an unsettled run (see rule_ten_terminal_step_over_threshold per arm)'),
            'table': table,
            'alpha_star_aggregate': star,
            'alpha_arb_distribution': {'min': min(arb_values), 'max': max(arb_values), 'n': len(arb_values),
                                       'sorted_quantiles': {q: sorted(arb_values)[int(q * (len(arb_values) - 1))]
                                                            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)}},
            'alpha_arb_by_block_hour': alpha_arb,
            'hour_tests': hour_tests,
            'duplicate_alpha_checks': duplicates,
            'arms': [{k: v for k, v in e.items() if k != '_blocks'} | {
                'block_totals': {key: blk['totals'] for key, blk in e['_blocks'].items()},
                'block_hours_E_abs_d_market': {key: blk['hours']['E_abs_d_market'] for key, blk in e['_blocks'].items()}}
                for e in entries],
            'check_failures': failures,
            'solve_profile': {'declared': 0, 'counts': dict(guard.counts), 'verify_failures': guard_failures},
        }
        os.makedirs(out_dir)
        with open(os.path.join(out_dir, 'analysis.json'), 'w') as handle:
            json.dump(payload, handle, indent=1, default=str)

        print(f'{"alpha":>7} {"rms n5/n7/n9 (MW)":>26} {"E|d|":>9} {"E|d|mkt":>9} {"Sum wd2":>10} '
              f'{"market":>10} {"oper":>9} {"mkt%":>6} {"cov":>10} {"charge":>9} {"rule10":>7}  source')
        for r in table:
            rms = '/'.join(f"{r['per_dso_rms_mw_max_over_blocks'].get(n, float('nan')):.4g}" for n in ('5', '7', '9'))
            print(f"{r['alpha']:>7g} {rms:>26} {r['E_abs_d_mwh']:>9.3f} {r['E_abs_d_market_mwh']:>9.3f} "
                  f"{r['sum_omega_d2_mw2h']:>10.3f} {r['market_part_mw2h']:>10.3f} {r['operation_part_mw2h']:>9.3f} "
                  f"{(r['market_share'] or 0) * 100:>6.1f} {r['covariance']:>10.2f} "
                  f"{r['row18_total_charge_all_dso']:>9.2f} {r['rule_ten_terminal_step_over_threshold'] or float('nan'):>7.1f}  "
                  f"{r['source']}")
            if r['redistribution_volumes_mwh']:
                print(f"{'':>7} redistribution (MWh): " +
                      ', '.join(f'{k} {v:.2f}' for k, v in r['redistribution_volumes_mwh'].items()))
        print(f"alpha*_agg: {star['alpha_star']} bracket {star['bracket']} (agg_tol {agg_tol:.4g} MW^2 h); "
              f"monotone non-increasing: {star['monotone_non_increasing']}")
        for s in star['steps']:
            print(f"   {s['from_alpha']:g} -> {s['to_alpha']:g}: {s['from']:.4f} -> {s['to']:.4f} "
                  f"{'NON-MONOTONE' if s['non_monotone'] else ''}")
        print(f"alpha_arb,t over all DSO (block, hour): min {min(arb_values):.4f} max {max(arb_values):.4f}")
        for a, h in hour_tests.items():
            m, t0 = h['market'], h['market_on_alpha0_market_deviating_hours']
            print(f"hour test alpha={a}: market dd/dh/hd/hh = {m['pred_dev_obs_dev']}/{m['pred_dev_obs_hold']}/"
                  f"{m['pred_hold_obs_dev']}/{m['pred_hold_obs_hold']} agree {m['agreement']:.3f}; market |d| in "
                  f"pred-hold hours {m['market_E_abs_d_in_pred_hold_hours_mwh']:.3f} MWh, in pred-dev "
                  f"{m['market_E_abs_d_in_pred_dev_hours_mwh']:.3f}; alpha0-subset n={t0['n_tested']} "
                  f"agree {t0['agreement']}")
        for e in entries:
            print(f"checks {e['run']}:{e['arm']}: {e['checks']}")
        print(f'duplicates: {duplicates}')
        print(f'check failures: {failures}')
        print(f'guard: {guard.counts} verify {guard_failures}')
        if failures:
            return 1
        return 0 if star['monotone_non_increasing'] else 1
    finally:
        guard.uninstall()


if __name__ == '__main__':
    sys.exit(main())
