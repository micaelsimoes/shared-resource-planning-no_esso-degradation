"""
P5.15 Addendum 38 (task W39) -- GATE 2 of 3: the 2 x 2 TWO-CYCLE alpha -> LARGE LIMIT
CHECK. As the imbalance premium grows without bound, row 18 must reproduce the PINNED
behaviour: the DSO's per-scenario interface deviation from its own committed schedule
goes to zero.

Authority: frozen spec v21 `data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json`,
`gates[1]` -- "2x2 two-cycle limit check: alpha -> large reproduces the pinned behaviour
(dispersion -> 0)"; PLANNER_BRIEF_2026-09-13.md Addendum 37 ("Gates") and Addendum 38.

WHAT IS UNDER TEST, and what is NOT. Row 18 replaces a hard pin by a PRICE. The claim is
that the price recovers the pin in the limit -- i.e. that the mechanism is a relaxation of
the pin and not something else. This gate measures the dispersion at the pilot's alpha and
at a large alpha on the SAME instance, the SAME candidate and the SAME two cycles, and
requires the large-alpha dispersion to collapse. It is NOT a convergence study, NOT a
result about the pilot's dispersion (two cycles is not a converged run), and NOT
comparable with any SRP1 (1 x 1) figure, where row 18 does not exist at all.

DISPERSION METRIC (Addendum 37, verbatim): per DSO, the RMS and the max of
d_{s,t} = p_int_{s,t} - pbar_t over scenarios and hours, in MW and as a share of the mean
interface flow, plus the total charge. Computed by production's own
`shared_resources_planning._get_operational_interface_dispersion`, so the gate cannot
drift from what the objective prices.

ARMS (one instance, one candidate, two cycles each):
  * `pilot` -- alpha = 0.50, the author's pilot value (Addendum 36);
  * `large` -- alpha = ALPHA_LARGE below, the limit arm.
GATE ITEMS ARE SCOPED PER ARM (CLAUDE.md stage template): the "dispersion collapses" items
apply to the `large` arm only; the `pilot` arm supplies the reference and is gated only on
running, solving and reconciling its solves.

DECLARED BEFORE THE RUN: the two alphas, the cycle cap, the candidate, the derived case
file and its sha256, the per-arm solve count, and the two dispersion thresholds.

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s51_2x2_limit_gate.py --label <write-once-label> \\
        > data/SRP1/Results/P515S51/2x2_limit_gate_launch_<label>.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S51/2x2_limit_gate/<label>/

OPTIONAL `--alphas a1 a2 ...` (P5.15 Addendum 39 follow-up, task W44 "coordinated alpha
sweep"): replaces the arm list. Omitted, the arms are exactly the committed `pilot` / `large`
pair, with their committed names and gate items. In SWEEP mode the arms are named
`alpha_<a>` and run in ascending alpha on the same instance, candidate and cycle cap; gate
items are scoped per arm: row 18 wired on every DSO block of every alpha > 0 arm and absent
from every alpha = 0 arm; the `large` items only if ALPHA_LARGE is an arm (the collapse ratio
only if ALPHA_PILOT is one too); items whose arm is absent are recorded as None and skipped.
The sweep STOPS (remaining arms not run, exit 1) after the first arm that does not run its
declared cycles, does not reconcile its solves, has an unrecovered / not-attempted /
indeterminate network failure, has a capture defect, or whose per-DSO dispersion exceeds the
previous (smaller) alpha's by more than DISPERSION_ZERO_TOL_MW (non-monotone). Every sweep
records `alpha_threshold` (alpha* = smallest tested alpha with dispersion <=
DISPERSION_ZERO_TOL_MW, overall and per DSO, with its bracket) and `attribution`.

MECHANISM AND PRICE CAPTURE (W44, ZERO extra solves), every DSO block of every arm, both
modes, read with pe.value off the TERMINAL DSO model the arm already solved (never re-solved):
  * `mechanism` -- `p515_s51_single_block_ab.mechanism_record` (W42), imported, not
    re-implemented: per scenario and hour d, the row 18 legs, the fl_reg DOWN / UP legs,
    c_flex * down, ESS, non-reference generation, residual, day-balance slacks, objective
    components;
  * `coordination` -- per hour pibar_t, pi_t[s_m], c_flex_t[s_m], the row 18 premium, the
    DSO interface duals dual_pf_p_req / dual_pf_q_req AS SET ON THE MODEL FOR ITS LAST SOLVE
    (the duals it responded to), the TSO request, the committed pbar_t, rho_pf, the interface
    rating, the block's effective objective scale, and the AL marginal price of the committed
    schedule  lambda_AL,t = scale * (dual + rho * (pbar - req) / rating) / (rating * baseMVA)
    (currency / MWh, the block objective's own units; p58_rescale multiplies the whole AL
    objective by that scale); per scenario and hour the RES curtailment
    sum_{g curtaillable} (pg_avail - pg), cross-checked against production's
    `gen_curtailment_definitional_value` at weight 1; the in-force penalty_gen_curtailment and
    interface_settlement_weight; the settlement total / contracted / deviation parts.
ATTRIBUTION FORMULA (per arm, per DSO block, per hour deviating in the alpha = 0 arm, E|d_t| >
HOUR_DEV_TOL_MWH): the curtail-and-reimport threshold
    alpha_t = (kappa + pibar_t + lambda_AL,t) / (2 * omega_lo,t * pibar_t)
-- curtailing c MWh in the higher-availability operation scenario costs
omega_hi * c * (kappa + pibar_t + lambda_AL,t) (the curtailment penalty in force, the
replacement import settled at the scenario prices pi_t[s_m], whose expectation is pibar_t, and
the AL on the committed schedule, which rises by omega_hi * c) and saves
alpha * pibar_t * 2 * omega_hi * omega_lo * c of row 18 P charge (losses, the Q leg and
flexibility ignored); predicted "deviates" iff alpha < alpha_t, own-arm duals. The OBSERVED
class of each such hour: `deviate-and-pay` if E|d_t| > HOUR_DEV_TOL_MWH, else
`curtail-and-reimport` if the curtailment increase over the alpha = 0 arm is the larger of the
two increases and exceeds VOLUME_TOL_MWH, else `priced-down-leg` if the fl_reg DOWN-leg increase
does, else `other`. Increases are against the alpha = 0 arm, whose ADMM path (TSO requests,
duals) differs -- the attribution is of two-cycle arms, not of a converged response.
Exit 0 on PASS, 1 on FAIL, 2 on a precondition refusal.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import psutil  # noqa: E402

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402

PERMITTED = tuple(tuple(p) for p in N.PERMITTED)
GUARD = SolveProfileGuard(PERMITTED, label='P5.15 W39 gate 2 -- 2x2 alpha limit check').install()

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import model_construction_helpers as MCH  # noqa: E402
import pyomo.environ as pe  # noqa: E402
# W44: W42's `mechanism_record`, imported (never re-implemented). The module installs its own
# SolveProfileGuard at import, on top of GUARD; uninstall it at once so GUARD -- installed
# first and verified exactly -- is again the only armed guard of this process.
import p515_s51_single_block_ab as SB  # noqa: E402
SB.GUARD.uninstall()
if any(SB.GUARD.counts.values()):
    raise RuntimeError(f'p515_s51_single_block_ab guard counted at import: {SB.GUARD.counts}')

STAGE = ('P5.15 Addendum 38 W39 gate 2 -- 2x2 two-cycle alpha -> large limit check '
         '(row 18 recovers the pin: dispersion -> 0)')
SCHEMA = 'p515_s51_2x2_limit_gate_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addenda 37, 38',
             'data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json gates[1]']

OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S51', '2x2_limit_gate')
OWN_LOCK_PATH = os.path.join(REPO, '.p515_s51_2x2_limit_gate.lock')

# ---- instance (identical to the W37 2x2 smoke's, so the two are comparable) ----
OVERRIDE_YEARS = {'2025': 5}
OVERRIDE_MARKET_SCENARIOS = 2
OVERRIDE_OPERATION_SCENARIOS = 2
INSTANCE_LABEL = 's51_2x2_limit'
ACTIVE_NODES = (5, 7, 9)
CYCLES = 2
REQUIRED_CONSECUTIVE_CYCLES = 10

# ---- the two arms, declared ----
ALPHA_PILOT = 0.50
ALPHA_LARGE = 1000.0
ARMS = ('pilot', 'large')
ALPHA_BY_ARM = {'pilot': ALPHA_PILOT, 'large': ALPHA_LARGE}

# ---- thresholds, declared BEFORE the run ----
DISPERSION_ZERO_TOL_MW = 1.0e-2        # "dispersion -> 0" on the `large` arm
DISPERSION_COLLAPSE_RATIO = 0.10       # large-arm RMS must be <= 10% of the pilot arm's
# W44 sweep / attribution constants, declared BEFORE the run (operational definitions)
HOUR_DEV_TOL_MWH = 1.0e-2     # an hour "deviates" iff E|d_t| exceeds this (= W43's)
VOLUME_TOL_MWH = 1.0e-2       # a leg "is used" iff its increase over the alpha = 0 arm exceeds this
AVAIL_EQUAL_TOL_MW = 1.0e-9   # operation scenarios with equal availability define no omega_lo (= W43's)
SWEEP_MODE = False            # set by --alphas

GIB = 1 << 30
RSS_LIMIT_GIB = 12.0
EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
THREAD_CAP_ENV = dict(S.THREAD_CAP_ENV)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(message):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W39-gate2] {message}', flush=True)


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


PRODUCTION_FILES_TO_CHECK_CLEAN = (
    'model_construction_helpers.py', 'shared_resources_planning.py', 'network.py',
    'admm_parameters.py', 'p515_g_g1_g4_admm_gates.py', 'p515_s44_campaign_harness.py',
    'p515_s44_scale_measurement.py', 'p56a_oracle.py', 'p515_s51_single_block_ab.py',
    os.path.basename(__file__))


def check_preconditions(out_dir):
    failures = []
    for path in (OWN_LOCK_PATH, G.CAMPAIGN_LOCK_PATH, os.path.join(REPO, '.p515_g_gate.lock'),
                 os.path.join(REPO, '.p515_s44_scale_measurement.lock')):
        if os.path.exists(path):
            failures.append(f'lock file exists: {path}')
    if os.path.exists(out_dir):
        failures.append(f'output directory already exists (write-once): {out_dir}')
    me = {os.getpid(), os.getppid()}
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if proc.info['pid'] in me or 'python' not in cmd:
            continue
        if S.HARNESS_PATTERN.search(cmd):
            failures.append(f'another p51x/p514 harness is alive: pid={proc.info["pid"]} {cmd[:200]}')
    status = _git(['status', '--porcelain', '--'] + list(PRODUCTION_FILES_TO_CHECK_CLEAN))
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')
    bad_env = {k: os.environ.get(k) for k, v in THREAD_CAP_ENV.items() if os.environ.get(k) != v}
    if bad_env:
        failures.append(f'thread caps not in force (export them before launching): {bad_env}')
    return failures


def capture_path_checklist():
    """CLAUDE.md rule eleven: assert BEFORE the run that a capture path exists for every
    quantity this gate's specification requires."""
    return {
        'dispersion_metric_callable': callable(getattr(srp, '_get_operational_interface_dispersion', None)),
        'per_block_dispersion_callable': callable(getattr(srp, '_get_local_interface_dispersion', None)),
        'row18_wiring_callable': callable(getattr(MCH, 'add_scenario_commitment_terms', None)),
        'recourse_components_callable': callable(getattr(srp, '_get_operational_recourse_components', None)),
        'voltage_mismatch_callable': callable(getattr(srp, '_get_local_scenario_voltage_mismatch', None)),
        'alpha_is_threaded_by_run_operational_planning': (
            'interface_deviation_premium' in __import__('inspect').getsource(srp._run_operational_planning)),
        'dispersion_dict_has_the_required_fields': True,   # verified live on the first block below
        # W44 mechanism / price capture (zero-solve, read off the terminal DSO models)
        'mechanism_record_callable': callable(getattr(SB, 'mechanism_record_or_error', None)),
        'gen_curtailment_definitional_value_callable': callable(
            getattr(MCH, 'gen_curtailment_definitional_value', None)),
        'expected_market_price_callable': callable(getattr(MCH, 'expected_market_price', None)),
        'settlement_split_callable': callable(getattr(srp, '_get_local_interface_settlement', None)),
        'dso_duals_set_on_the_model_before_each_dso_solve': (
            'dual_pf_p_req[p].set_value' in __import__('inspect').getsource(
                srp.update_distribution_coordination_models_and_solve_sequential)),
        'dso_al_objective_carries_dual_rho_rating_scale': all(
            token in __import__('inspect').getsource(srp.update_distribution_models_to_admm)
            for token in ('dual_pf_p_req', 'rho_pf', 'admm_objective_scale', 'get_interface_branch_rating')),
    }


def coordination_record(model, network, params):
    """W44 (ZERO SOLVES): the prices and duals the DSO block faced in its LAST solve, and its
    RES curtailment, read with `pe.value` off the already-solved terminal model. MW x baseMVA;
    one period = one hour. See the module docstring for lambda_AL,t."""
    s_base = network.baseMVA
    rating_pu = network.get_interface_branch_rating() / s_base
    has_al = hasattr(model, 'dual_pf_p_req')
    rho = float(pe.value(model.rho_pf)) if has_al else None
    scale = float(pe.value(model.admm_objective_scale)) if hasattr(model, 'admm_objective_scale') else None
    n_market = len(network.prob_market_scenarios)
    hours = []
    for p in model.periods:
        pbar_pu = float(pe.value(model.expected_interface_pf_p[p]))
        entry = {'period': p, 'pibar': float(MCH.expected_market_price(network, p)),
                 'pi_by_market_scenario': [float(network.cost_energy_p[s_m][p]) for s_m in range(n_market)],
                 'c_flex_by_market_scenario': [float(network.cost_flex[s_m][p]) for s_m in range(n_market)],
                 'row18_premium': (float(pe.value(model.row18_premium[p]))
                                   if hasattr(model, 'row18_premium') else None),
                 'pbar_mw': pbar_pu * s_base}
        if has_al:
            dual = float(pe.value(model.dual_pf_p_req[p]))
            req_pu = float(pe.value(model.p_pf_req[p]))
            entry.update({
                'dual_pf_p_req': dual, 'dual_pf_q_req': float(pe.value(model.dual_pf_q_req[p])),
                'p_pf_req_mw': req_pu * s_base, 'q_pf_req_mvar': float(pe.value(model.q_pf_req[p])) * s_base,
                'lambda_al_per_mwh': (scale * (dual + rho * (pbar_pu - req_pu) / rating_pu) / (rating_pu * s_base)
                                      if scale is not None else None)})
        hours.append(entry)
    curtaillable = [g for g in model.generators if network.generators[g].is_curtaillable()]
    ref_gen = network.get_reference_gen_idx()
    curtailment = {}
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            series, avail = [], []
            for p in model.periods:
                if params.rg_curt:
                    series.append(sum(float(pe.value(model.pg_avail[g, s_o, p])) - float(pe.value(model.pg[g, s_m, s_o, p]))
                                      for g in curtaillable) * s_base)
                else:
                    series.append(0.0)
                avail.append(sum(float(pe.value(model.pg_avail[g, s_o, p])) for g in curtaillable) * s_base
                             if hasattr(model, 'pg_avail') else None)
            production_mwh = float(pe.value(MCH.gen_curtailment_definitional_value(model, network, s_m, s_o, params, 1.0)))
            curtailment[f'{s_m}_{s_o}'] = {'curt_res_mw': series, 'avail_curtaillable_mw': avail,
                                           'production_definitional_mwh': production_mwh,
                                           'check_residual_mwh': sum(series) - production_mwh}
    return {
        'units': 'MW per hour (x baseMVA); lambda_al_per_mwh and prices in currency / MWh',
        'interface_rating_pu': rating_pu, 'rho_pf': rho, 'admm_objective_scale': scale,
        'penalty_gen_curtailment': (float(pe.value(model.penalty_gen_curtailment))
                                    if hasattr(model, 'penalty_gen_curtailment') else None),
        'interface_settlement_weight': float(pe.value(model.interface_settlement_weight)),
        'settlement': {part: float(srp._get_local_interface_settlement(model, part=part))
                       for part in ('total', 'contracted', 'deviation')},
        'row18_charge': (float(pe.value(model.row18_deviation_charge))
                         if hasattr(model, 'row18_deviation_charge') else 0.0),
        'rg_curt': bool(params.rg_curt), 'n_curtaillable_generators': len(curtaillable),
        'n_noncurtaillable_nonref_generators': sum(1 for g in model.generators
                                                   if g != ref_gen and g not in curtaillable),
        'hours': hours, 'curtailment': curtailment,
    }


def coordination_record_or_error(model, network, params, where):
    """A capture defect must not destroy the arm's solved result: recorded and logged."""
    try:
        return coordination_record(model, network, params)
    except Exception as error:  # noqa: BLE001
        _log(f'COORDINATION CAPTURE FAILED at {where}: {error!r}')
        return {'capture_error': repr(error)}


def block_summary(mech, coord, alpha):
    """W44 per-block derived quantities (formulas in the module docstring). Probability-
    weighted, block-local, UNWEIGHTED by year/day/discount; MWh/day."""
    if 'capture_error' in mech or 'capture_error' in coord:
        return {'capture_error': {'mechanism': mech.get('capture_error'), 'coordination': coord.get('capture_error')}}
    per_scenario = mech['per_scenario']
    n = len(coord['hours'])

    def expect(fn):
        return [sum(entry['probability'] * fn(key, entry['series'], t) for key, entry in per_scenario.items())
                for t in range(n)]

    hours = {
        'E_abs_d_mwh': expect(lambda k, s, t: abs(s['d_p_mw'][t])),
        'E_curt_mwh': expect(lambda k, s, t: coord['curtailment'][k]['curt_res_mw'][t]),
        'E_flex_down_mwh': expect(lambda k, s, t: s['flex_p_down_mw'][t]),
        'E_flex_up_mwh': expect(lambda k, s, t: s['flex_p_up_mw'][t]),
        'E_flex_down_cost': expect(lambda k, s, t: s['flex_p_down_cost'][t]),
        'E_interface_p_mwh': expect(lambda k, s, t: s['interface_p_mw'][t]),
    }
    omega = {}
    avail_by_so = {}
    for key, entry in per_scenario.items():
        s_o = key.split('_')[1]
        omega[s_o] = omega.get(s_o, 0.0) + entry['probability']
        avail_by_so.setdefault(s_o, coord['curtailment'][key]['avail_curtaillable_mw'])
    omega_lo = [None] * n
    if len(omega) == 2 and all(v is not None for v in avail_by_so.values()):
        (so_a, av_a), (so_b, av_b) = sorted(avail_by_so.items())
        for t in range(n):
            if abs(av_a[t] - av_b[t]) > AVAIL_EQUAL_TOL_MW:
                omega_lo[t] = omega[so_a] if av_a[t] < av_b[t] else omega[so_b]
    kappa = coord['penalty_gen_curtailment'] or 0.0
    alpha_t = []
    for t, h in enumerate(coord['hours']):
        lam = h.get('lambda_al_per_mwh') or 0.0
        alpha_t.append((kappa + h['pibar'] + lam) / (2.0 * omega_lo[t] * h['pibar'])
                       if (omega_lo[t] is not None and h['pibar'] > 0.0) else None)
    return {
        'alpha': alpha, 'hours': hours, 'totals': {k: sum(v) for k, v in hours.items()},
        'omega_operation': omega, 'omega_lo_by_hour': omega_lo,
        'kappa_in_force': kappa, 'alpha_t_curtail_by_hour': alpha_t,
        'pibar_by_hour': [h['pibar'] for h in coord['hours']],
        'lambda_al_by_hour': [h.get('lambda_al_per_mwh') for h in coord['hours']],
        'settlement': coord['settlement'], 'interface_settlement_weight': coord['interface_settlement_weight'],
        'row18_charge': coord['row18_charge'],
        'curtailment_check_max_abs_residual_mwh': max(abs(c['check_residual_mwh'])
                                                      for c in coord['curtailment'].values()),
        'n_noncurtaillable_nonref_generators': coord['n_noncurtaillable_nonref_generators'],
    }


def _dispersion_summary(dispersion_detail):
    """Collapse production's per-block dispersion detail into the Addendum 37 metric, per
    DSO node and overall. Returns (per_node, overall)."""
    per_node = {}
    for (kind, node_id, year, day), detail in dispersion_detail.items():
        if detail is None:
            continue
        entry = per_node.setdefault(str(node_id), {'blocks': {}, 'rms_mw_max_over_blocks': 0.0,
                                                   'max_abs_mw': 0.0, 'total_charge': 0.0,
                                                   'rms_share_max': None})
        block_key = f'{year}:{day}'
        entry['blocks'][block_key] = {
            'p_rms_mw': detail['p']['rms_mw'],
            'p_max_abs_mw': detail['p']['max_abs_mw'],
            'p_rms_share_of_mean_flow': detail['p']['rms_share_of_mean_flow'],
            'p_max_abs_share_of_mean_flow': detail['p']['max_abs_share_of_mean_flow'],
            'q_rms_mvar': detail['q']['rms_mvar'],
            'q_max_abs_mvar': detail['q']['max_abs_mvar'],
            'row18_charge': detail['row18_charge'],
            'row18_alpha': detail['row18_alpha'],
            'row18_wired': detail['row18_wired'],
            'mean_abs_committed_flow_mw': detail['p']['mean_abs_committed_flow_mw'],
        }
        entry['rms_mw_max_over_blocks'] = max(entry['rms_mw_max_over_blocks'], detail['p']['rms_mw'])
        entry['max_abs_mw'] = max(entry['max_abs_mw'], detail['p']['max_abs_mw'])
        entry['total_charge'] += detail['row18_charge']
        share = detail['p']['rms_share_of_mean_flow']
        if share is not None:
            entry['rms_share_max'] = share if entry['rms_share_max'] is None else max(entry['rms_share_max'], share)
    overall = {
        'rms_mw_max_over_all_dso_blocks': max([v['rms_mw_max_over_blocks'] for v in per_node.values()] or [0.0]),
        'max_abs_mw_over_all_dso_blocks': max([v['max_abs_mw'] for v in per_node.values()] or [0.0]),
        'total_charge_all_dso': sum(v['total_charge'] for v in per_node.values()),
        'n_dso_nodes': len(per_node),
    }
    return per_node, overall


def _make_post_run_hook(arm, record):
    def hook(planning=None, sed=None, models=None, rows=None, report=None, out_dir=None, label=None):
        detail = srp._get_operational_interface_dispersion(planning, models)
        per_node, overall = _dispersion_summary(detail)
        record['dispersion_per_node'] = per_node
        record['dispersion_overall'] = overall
        record['recourse_components'] = srp._get_operational_recourse_components(planning, models)
        record['voltage_mismatch'] = {
            f'{k[0]}:{k[1]}:{k[2]}:{k[3]}': v
            for k, v in srp._get_operational_scenario_voltage_mismatch(planning, models).items()}
        raw_path = os.path.join(out_dir, f'dispersion_detail_{arm}.json')
        G._refuse_overwrite(raw_path)
        with open(raw_path, 'w') as handle:
            json.dump({f'{k[0]}:{k[1]}:{k[2]}:{k[3]}': v for k, v in detail.items()},
                      handle, indent=1, default=str)
        record['dispersion_detail_path'] = os.path.relpath(raw_path, REPO)
        # W44 (zero solves): mechanism + prices/duals per DSO block, off the terminal models
        mechanism, coordination, summary = {}, {}, {}
        for node_id, network_data in planning.distribution_networks.items():
            for year in network_data.years:
                for day in network_data.days:
                    key = f'DSO:{node_id}:{year}:{day}'
                    model = models['dso'][node_id][year][day]
                    network = network_data.network[year][day]
                    mechanism[key] = SB.mechanism_record_or_error(model, network, network_data.params, f'{arm} {key}')
                    coordination[key] = coordination_record_or_error(model, network, network_data.params, f'{arm} {key}')
                    try:
                        summary[key] = block_summary(mechanism[key], coordination[key], record['alpha'])
                    except Exception as error:  # noqa: BLE001
                        _log(f'BLOCK SUMMARY FAILED at {arm} {key}: {error!r}')
                        summary[key] = {'capture_error': repr(error)}
        record['block_summary'] = summary
        record['capture_errors'] = sorted(k for k, v in summary.items() if 'capture_error' in v)
        record['settlement_deviation_weighted_dso'] = sum(
            v for k, v in srp._get_operational_interface_settlement_blocks(planning, models, part='deviation').items()
            if k[0] == 'DSO')
        mech_path = os.path.join(out_dir, f'mechanism_{arm}.json')
        G._refuse_overwrite(mech_path)
        with open(mech_path, 'w') as handle:
            json.dump({'mechanism': mechanism, 'coordination': coordination}, handle, default=str)
        record['mechanism_path'] = os.path.relpath(mech_path, REPO)
    return hook


def arm_sweep_row(record):
    """W44: one row of the alpha table -- per DSO and overall. Volumes are block-local,
    probability-weighted, summed over the DSO's (year, day) blocks UNWEIGHTED (MWh/day summed
    over representative days); the row 18 charge and the settlement deviation part likewise
    (block-local currency), plus production's year/day/discount-weighted settlement deviation."""
    per_dso = {}
    for key, s in record.get('block_summary', {}).items():
        node = key.split(':')[1]
        entry = per_dso.setdefault(node, {'E_abs_d_mwh': 0.0, 'E_curt_mwh': 0.0, 'E_flex_down_mwh': 0.0,
                                          'E_flex_up_mwh': 0.0, 'E_flex_down_cost': 0.0,
                                          'settlement_deviation': 0.0, 'row18_charge_from_capture': 0.0})
        if 'capture_error' in s:
            entry['capture_error'] = True
            continue
        for name in ('E_abs_d_mwh', 'E_curt_mwh', 'E_flex_down_mwh', 'E_flex_up_mwh', 'E_flex_down_cost'):
            entry[name] += s['totals'][name]
        entry['settlement_deviation'] += s['settlement']['deviation']
        entry['row18_charge_from_capture'] += s['row18_charge']
    for node, disp in record['dispersion_per_node'].items():
        entry = per_dso.setdefault(node, {})
        entry['rms_mw_max_over_blocks'] = disp['rms_mw_max_over_blocks']
        entry['max_abs_mw'] = disp['max_abs_mw']
        entry['rms_share_of_mean_flow_max_over_blocks'] = disp['rms_share_max']
        shares = [b['p_max_abs_share_of_mean_flow'] for b in disp['blocks'].values()
                  if b['p_max_abs_share_of_mean_flow'] is not None]
        entry['max_abs_share_of_mean_flow_max_over_blocks'] = max(shares) if shares else None
        entry['total_row18_charge'] = disp['total_charge']
    totals = {name: sum(v.get(name, 0.0) for v in per_dso.values())
              for name in ('E_abs_d_mwh', 'E_curt_mwh', 'E_flex_down_mwh', 'E_flex_up_mwh',
                           'E_flex_down_cost', 'settlement_deviation', 'total_row18_charge')}
    return {'alpha': record['alpha'], 'per_dso': per_dso, 'all_dso': totals,
            'rms_mw_max_over_all_dso_blocks': record['dispersion_overall']['rms_mw_max_over_all_dso_blocks'],
            'max_abs_mw_over_all_dso_blocks': record['dispersion_overall']['max_abs_mw_over_all_dso_blocks'],
            'settlement_deviation_weighted_dso': record.get('settlement_deviation_weighted_dso'),
            'rule_ten_terminal_step_over_threshold': record.get('rule_ten_terminal_step_over_threshold')}


def alpha_threshold(holder):
    """W44: alpha* = the smallest tested alpha whose dispersion (max over blocks of the per-block
    RMS interface-P dispersion, MW) is <= DISPERSION_ZERO_TOL_MW; bracket = (largest tested alpha
    above the tolerance and below alpha*, alpha*]. Overall (max over every DSO block) and per DSO."""
    records = sorted(holder.values(), key=lambda r: r['alpha'])

    def locate(rms):
        alphas = sorted(rms)
        at_or_below = [a for a in alphas if rms[a] <= DISPERSION_ZERO_TOL_MW]
        above = [a for a in alphas if rms[a] > DISPERSION_ZERO_TOL_MW]
        star = at_or_below[0] if at_or_below else None
        below = [a for a in above if star is None or a < star]
        return {'rms_mw_by_alpha': {str(a): rms[a] for a in alphas}, 'alpha_star': star,
                'bracket': [below[-1] if below else None, star],
                'above_tol_after_alpha_star': [a for a in above if star is not None and a > star]}

    overall = locate({r['alpha']: r['dispersion_overall']['rms_mw_max_over_all_dso_blocks'] for r in records})
    nodes = sorted({n for r in records for n in r['dispersion_per_node']})
    per_dso = {n: locate({r['alpha']: r['dispersion_per_node'][n]['rms_mw_max_over_blocks']
                          for r in records if n in r['dispersion_per_node']}) for n in nodes}
    return {'definition': ('alpha* = smallest tested alpha with dispersion (max over blocks of the '
                           'per-block RMS interface-P dispersion, MW) <= DISPERSION_ZERO_TOL_MW; '
                           'bracket = (largest tested alpha above it and below alpha*, alpha*]'),
            'dispersion_zero_tol_mw': DISPERSION_ZERO_TOL_MW, 'overall': overall, 'per_dso': per_dso}


def attribution(holder):
    """W44: per arm, per DSO block, per hour deviating in the alpha = 0 arm -- the predicted
    curtail-and-reimport threshold against the observed class of the hour (module docstring)."""
    reference = next((r for r in holder.values() if r['alpha'] == 0.0), None)
    if reference is None:
        return {'available': False, 'reason': 'no alpha = 0 arm in this run'}
    ref_blocks = reference['block_summary']
    out = {'available': True, 'reference_arm': reference['arm'], 'per_arm': {}}
    for record in sorted(holder.values(), key=lambda r: r['alpha']):
        alpha = record['alpha']
        arm_out = {'alpha': alpha, 'class_hours': {}, 'class_hours_per_dso': {}, 'blocks': {},
                   'prediction_agree': 0, 'prediction_tested': 0, 'prediction_disagreements': [],
                   'E_abs_d_ref_mwh': 0.0, 'E_abs_d_mwh': 0.0, 'dE_curt_mwh': 0.0, 'dE_flex_down_mwh': 0.0}
        for key, s in record['block_summary'].items():
            ref = ref_blocks.get(key)
            if ref is None or 'capture_error' in s or 'capture_error' in ref:
                continue
            node = key.split(':')[1]
            deviating = [t for t, v in enumerate(ref['hours']['E_abs_d_mwh']) if v > HOUR_DEV_TOL_MWH]
            block_hours = []
            for t in deviating:
                dev = s['hours']['E_abs_d_mwh'][t]
                d_curt = s['hours']['E_curt_mwh'][t] - ref['hours']['E_curt_mwh'][t]
                d_down = s['hours']['E_flex_down_mwh'][t] - ref['hours']['E_flex_down_mwh'][t]
                if dev > HOUR_DEV_TOL_MWH:
                    cls = 'deviate-and-pay'
                elif d_curt > VOLUME_TOL_MWH and d_curt >= d_down:
                    cls = 'curtail-and-reimport'
                elif d_down > VOLUME_TOL_MWH:
                    cls = 'priced-down-leg'
                else:
                    cls = 'other'
                arm_out['class_hours'][cls] = arm_out['class_hours'].get(cls, 0) + 1
                per_node = arm_out['class_hours_per_dso'].setdefault(node, {})
                per_node[cls] = per_node.get(cls, 0) + 1
                a_t = s['alpha_t_curtail_by_hour'][t]
                if a_t is not None:
                    arm_out['prediction_tested'] += 1
                    if (alpha < a_t) == (dev > HOUR_DEV_TOL_MWH):
                        arm_out['prediction_agree'] += 1
                    else:
                        arm_out['prediction_disagreements'].append(
                            {'block': key, 'hour': t, 'alpha_t': a_t, 'observed_E_abs_d_mwh': dev,
                             'margin_alpha_minus_alpha_t_over_alpha_t': (alpha - a_t) / a_t})
                arm_out['E_abs_d_ref_mwh'] += ref['hours']['E_abs_d_mwh'][t]
                arm_out['E_abs_d_mwh'] += dev
                arm_out['dE_curt_mwh'] += d_curt
                arm_out['dE_flex_down_mwh'] += d_down
                block_hours.append({'hour': t, 'class': cls, 'E_abs_d_mwh': dev,
                                    'E_abs_d_ref_mwh': ref['hours']['E_abs_d_mwh'][t],
                                    'dE_curt_mwh': d_curt, 'dE_flex_down_mwh': d_down,
                                    'E_flex_up_mwh': s['hours']['E_flex_up_mwh'][t],
                                    'alpha_t_curtail': a_t, 'pibar': s['pibar_by_hour'][t],
                                    'lambda_al_per_mwh': s['lambda_al_by_hour'][t],
                                    'omega_lo': s['omega_lo_by_hour'][t]})
            arm_out['blocks'][key] = block_hours
        removed = arm_out['E_abs_d_ref_mwh'] - arm_out['E_abs_d_mwh']
        arm_out['E_abs_d_removed_mwh'] = removed
        arm_out['dE_curt_over_E_abs_d_removed'] = arm_out['dE_curt_mwh'] / removed if removed > 1.0 else None
        arm_out['dE_flex_down_over_E_abs_d_removed'] = arm_out['dE_flex_down_mwh'] / removed if removed > 1.0 else None
        alpha_ts = [a for key in record['block_summary'] if 'capture_error' not in record['block_summary'][key]
                    for a in record['block_summary'][key]['alpha_t_curtail_by_hour'] if a is not None]
        arm_out['alpha_t_curtail_min_max_all_hours'] = [min(alpha_ts), max(alpha_ts)] if alpha_ts else None
        out['per_arm'][record['arm']] = arm_out
    return out


def sweep_stop_reasons(record, previous):
    """W44 sweep mode: why the sweep must STOP after this arm (empty list = continue)."""
    reasons = []
    if record['cycles_run'] != CYCLES:
        reasons.append(f"cycles_run {record['cycles_run']} != {CYCLES}")
    expected = (record['event_level_reconciliation'] or {}).get('expected')
    if expected is None or record['solves_in_arm'] != expected:
        reasons.append(f"solves {record['solves_in_arm']} do not reconcile (expected {expected})")
    classes = (record.get('network_failures_summary') or {}).get('classes') or {}
    bad = {c: classes.get(c, 0) for c in ('unrecovered', 'not_attempted', 'indeterminate')}
    if any(bad.values()):
        reasons.append(f'network failures {bad}')
    if record.get('capture_errors'):
        reasons.append(f"capture defects on {record['capture_errors']}")
    if previous is not None:
        for node, disp in record['dispersion_per_node'].items():
            before = previous['dispersion_per_node'].get(node, {}).get('rms_mw_max_over_blocks')
            if before is not None and disp['rms_mw_max_over_blocks'] > before + DISPERSION_ZERO_TOL_MW:
                reasons.append(f"non-monotone: DSO {node} rms {disp['rms_mw_max_over_blocks']} at alpha "
                               f"{record['alpha']} > {before} at alpha {previous['alpha']} + tol")
    return reasons


def _make_pre_solve_hook(arm, alpha, record, cfg_holder):
    spec_like = {'configuration': {'overrides': {},
                                   'case_file_anderson_acceleration': dict(S.CASE_FILE_AA)},
                 'cap': CYCLES, 'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES}
    inner = H._config_hook_factory(spec_like, cfg_holder, overrides={})
    inner = S.snapshot_hook_wrapper(inner, 'off', record)

    def hook(planning=None, sed=None, candidate=None, report=None):
        inner(planning=planning, sed=sed, candidate=candidate, report=report)
        # THE ONE ARM-DEFINING DIFFERENCE: alpha. Set on this arm's own deep-copied
        # planning object, immediately before the first solve; nothing else differs.
        planning.params.admm.interface_deviation_premium = {
            'alpha': float(alpha), 'floor': None, 'source': f'W39 gate 2 arm {arm!r}'}
        record['alpha_applied'] = dict(planning.params.admm.interface_deviation_premium)
    return hook


def run_arm(arm, out_root, planning0, declared_base, holder):
    alpha = ALPHA_BY_ARM[arm]
    record = holder.setdefault(arm, {'arm': arm, 'alpha': alpha})
    arm_dir = os.path.join(out_root, f'arm_{arm}')
    os.makedirs(arm_dir)
    label = f's51limit_{arm}'
    eval_id = f'p515s51_limit_{os.path.basename(out_root)}_{arm}'
    record['eval_id'] = eval_id
    investment_map = {node: (0.0, 0.0) for node in ACTIVE_NODES}
    record['investment_map'] = {str(k): list(v) for k, v in investment_map.items()}
    record['candidate_label'] = 'x = 0 (no shared-ESS investment) at every active node'

    cfg_holder = {}
    pre_hook = _make_pre_solve_hook(arm, alpha, record, cfg_holder)
    t0 = time.time()
    before = GUARD.counts['permitted_solve']
    report, report_path = G.run_admm_arm(
        label, arm_dir, k_override=None, investment_map=investment_map,
        num_max_iters_override=CYCLES, eval_id=eval_id, apply_rho=False,
        full_diagnostics_in_rows=True, pre_solve_hook=pre_hook,
        post_run_hook=_make_post_run_hook(arm, record))
    record['wall_s'] = time.time() - t0
    record['solves_in_arm'] = GUARD.counts['permitted_solve'] - before
    record['report_path'] = os.path.relpath(report_path, REPO)
    record['cycles_run'] = report.get('cycles_run')
    record['converged_at_cycle'] = report.get('converged_at_cycle')
    record['recourse'] = report.get('recourse')
    record['gross_operational_cost'] = report.get('gross_operational_cost')
    record['objective_convention'] = (
        'recourse = net_operational_recourse (contracted-settlement-excluded, '
        'voltage-pin-excluded, salvage-netted); gross_operational_cost is the same '
        'without the salvage credit -- P5.15 Addendum 38 (C)/(D) conventions, as '
        'shared_resources_planning._get_operational_recourse_components defines them')
    # CLAUDE.md: the terminal-step-to-threshold ratio, for every cell of every evaluation
    record['rule_ten_terminal_step_over_threshold'] = report.get(
        'rule_ten_terminal_step_over_threshold')
    record['terminal_objective_change_abs'] = report.get('terminal_objective_change_abs')
    record['terminal_objective_tolerance'] = report.get('terminal_objective_tolerance')
    record['cycle_trajectory'] = report.get('cycle_trajectory')       # per-cycle state by default
    record['network_failures_summary'] = report.get('network_failures_summary')
    record['arm_solve_profile'] = report.get('solve_profile')
    record['event_level_reconciliation'] = S.event_level_solve_reconciliation(report, declared_base)
    record['anderson_acceleration_effective'] = cfg_holder.get('anderson_acceleration_effective')
    record['configuration_checks'] = cfg_holder.get('configuration_checks')
    return record


def main():
    global CYCLES
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument('--label', required=True, help='write-once output label')
    parser.add_argument('--cycles', type=int, default=CYCLES)
    parser.add_argument('--alphas', type=float, nargs='+', default=None,
                        help='W44 sweep arm list; omitted = the committed pilot/large pair')
    args = parser.parse_args()
    CYCLES = args.cycles
    global ARMS, ALPHA_BY_ARM, SWEEP_MODE
    alpha_list_source = 'default ARMS (pilot / large, as committed)'
    if args.alphas is not None:
        if len(set(args.alphas)) != len(args.alphas) or any(a < 0.0 for a in args.alphas):
            print(f'REFUSED: --alphas must be distinct and non-negative: {args.alphas}', file=sys.stderr)
            return EXIT_REFUSED
        alphas = sorted(float(a) for a in args.alphas)
        ARMS = tuple(f'alpha_{a}' for a in alphas)
        ALPHA_BY_ARM = {f'alpha_{a}': a for a in alphas}
        SWEEP_MODE = True
        alpha_list_source = ('command line --alphas (P5.15 Addendum 39 follow-up, W44 coordinated '
                             'alpha sweep; ascending)')

    out_root = os.path.join(OUT_ROOT, args.label)
    failures = check_preconditions(out_root)
    if failures:
        for failure in failures:
            print(f'[W39-gate2 PRECONDITION FAILED] {failure}', file=sys.stderr)
        return EXIT_REFUSED
    try:
        fd = os.open(OWN_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f'REFUSED: lock held {OWN_LOCK_PATH}', file=sys.stderr)
        return EXIT_REFUSED
    with os.fdopen(fd, 'w') as handle:
        json.dump({'pid': os.getpid(), 'label': args.label, 'started_utc': _utc()}, handle)

    started = time.time()
    holder = {}
    os.environ.update(THREAD_CAP_ENV)
    try:
        os.makedirs(out_root)
        case_dir = os.path.join(out_root, 'case')
        os.makedirs(case_dir)
        case, spec, changes = S.derive_case('srp1', {
            'years': OVERRIDE_YEARS,
            'num_market_scenarios': OVERRIDE_MARKET_SCENARIOS,
            'num_operation_scenarios': OVERRIDE_OPERATION_SCENARIOS})
        case_path = os.path.join(case_dir, 'SRP1__s51_2x2.json')
        with open(case_path, 'w') as handle:
            json.dump(case, handle, indent='\t')

        launch = {
            'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY,
            'label': args.label, 'instance': INSTANCE_LABEL, 'instance_definition': spec,
            'derived_case': {'path': os.path.relpath(case_path, REPO),
                             'sha256': _sha256_file(case_path),
                             'source': os.path.relpath(S.SOURCE_CASE, REPO),
                             'source_sha256': _sha256_file(S.SOURCE_CASE),
                             'changes_vs_source': changes},
            'argv': sys.argv, 'interpreter': sys.executable,
            'script': os.path.basename(__file__),
            'script_sha256': _sha256_file(os.path.abspath(__file__)),
            'git_head': _git(['rev-parse', 'HEAD']),
            'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
            'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH'),
            'cycles': CYCLES, 'arms': list(ARMS), 'alpha_by_arm': ALPHA_BY_ARM,
            'alpha_list_source': alpha_list_source, 'sweep_mode': SWEEP_MODE,
            'thresholds_declared_before_the_run': {
                'dispersion_zero_tol_mw': DISPERSION_ZERO_TOL_MW,
                'dispersion_collapse_ratio': DISPERSION_COLLAPSE_RATIO,
                'hour_dev_tol_mwh': HOUR_DEV_TOL_MWH, 'volume_tol_mwh': VOLUME_TOL_MWH,
                'avail_equal_tol_mw': AVAIL_EQUAL_TOL_MW},
            'guard_permitted': [list(p) for p in PERMITTED],
            'snapshots': 'off',
            'started_utc': _utc(), 'pid': os.getpid(),
        }

        stages = S.StageLog(os.path.join(out_root, 'stages.jsonl'),
                            S.Watchdog(out_root, 's51limit', limit_bytes=int(RSS_LIMIT_GIB * GIB)))
        stages.wd.start()
        planning0 = S.read_planning_from_derived_case(launch, out_root, stages)
        launch['scenario_checksum'] = S.inject_oracle_baseline(O, planning0, launch)
        launch['planning_dimensions'] = S.planning_dimensions(planning0)
        launch['expected_block_counts'] = S.expected_block_counts(planning0)
        provenance = S.provenance_record(planning0, INSTANCE_LABEL, launch['scenario_checksum'])
        launch['provenance'] = provenance
        non_checksum = [f for f in provenance['gate_failures'] if f['identity'] != 'scenario checksum']
        if non_checksum:
            raise RuntimeError(f'provenance: non-canonical identity: {non_checksum}')

        declared = S.declared_solve_profile(planning0, CYCLES)
        declared_base = declared['declared_base_solves']
        launch['declared_solve_profile'] = {**declared, 'n_arms': len(ARMS),
                                            'declared_total_strict': len(ARMS) * declared_base}

        checklist = capture_path_checklist()
        launch['capture_path_checklist_asserted_before_run'] = checklist
        if not all(checklist.values()):
            raise RuntimeError(f'capture-path checklist failed: '
                               f'{[k for k, v in checklist.items() if not v]}')

        launch_path = os.path.join(out_root, 'launch.json')
        G._refuse_overwrite(launch_path)
        with open(launch_path, 'w') as handle:
            json.dump(launch, handle, indent=1, default=str)
        _log(f'declared {declared_base} base solves per arm; {len(ARMS)} arms; '
             f'alphas {ALPHA_BY_ARM}')

        # The legacy run lock (`.p515_g_gate.lock`). It is released by
        # `_acquire_exclusive_run_lock`'s own atexit handler -- the module exposes no
        # explicit release entry point, and the committed gates do not release it either.
        G._acquire_exclusive_run_lock()
        stopped = None
        previous = None
        for index, arm in enumerate(ARMS):
            _log(f'arm {arm}: alpha = {ALPHA_BY_ARM[arm]}')
            run_arm(arm, out_root, planning0, declared_base, holder)
            _log(f"arm {arm}: cycles={holder[arm]['cycles_run']} "
                 f"rms_mw={holder[arm]['dispersion_overall']['rms_mw_max_over_all_dso_blocks']} "
                 f"charge={holder[arm]['dispersion_overall']['total_charge_all_dso']}")
            row = arm_sweep_row(holder[arm])
            _log(f"arm {arm}: solves={holder[arm]['solves_in_arm']} "
                 f"per-DSO rms={ {n: v.get('rms_mw_max_over_blocks') for n, v in row['per_dso'].items()} } "
                 f"E|d|={row['all_dso']['E_abs_d_mwh']:.4f} Ecurt={row['all_dso']['E_curt_mwh']:.4f} "
                 f"Edown={row['all_dso']['E_flex_down_mwh']:.4f} "
                 f"settle_dev={row['all_dso']['settlement_deviation']:.4f} "
                 f"capture_errors={holder[arm].get('capture_errors')}")
            if SWEEP_MODE:
                reasons = sweep_stop_reasons(holder[arm], previous)
                if reasons:
                    stopped = {'after_arm': arm, 'reasons': reasons, 'arms_not_run': list(ARMS[index + 1:])}
                    _log(f'SWEEP STOPPED after {arm}: {reasons}; not run: {stopped["arms_not_run"]}')
                    break
            previous = holder[arm]

        def _arm_at(alpha):
            return next((r for r in holder.values() if math.isclose(r['alpha'], alpha)), None)
        pilot_record, large_record = _arm_at(ALPHA_PILOT), _arm_at(ALPHA_LARGE)
        pilot = pilot_record['dispersion_overall'] if pilot_record else None
        large = large_record['dispersion_overall'] if large_record else None
        ratio = (large['rms_mw_max_over_all_dso_blocks'] / pilot['rms_mw_max_over_all_dso_blocks']
                 if (pilot and large and pilot['rms_mw_max_over_all_dso_blocks'] > 0) else None)

        reconciled_total = sum((r['event_level_reconciliation'] or {}).get('expected') or 0
                               for r in holder.values())
        supported = all((r['event_level_reconciliation'] or {}).get('expected') is not None
                        for r in holder.values())
        guard_failures = GUARD.verify(reconciled_total) if supported else [
            'event-level reconciliation unsupported on at least one arm']

        if not SWEEP_MODE:
            gate_items = {
                # both arms
                'both_arms_ran_the_declared_cycles': all(r['cycles_run'] == CYCLES for r in holder.values()),
                'both_arms_reconcile_their_solves': supported and all(
                    r['solves_in_arm'] == (r['event_level_reconciliation'] or {}).get('expected')
                    for r in holder.values()),
                'process_guard_verified_exactly': not guard_failures,
                'no_blocked_solver_calls': GUARD.counts['blocked_solve'] == 0 and GUARD.counts['blocked_exec'] == 0,
                'row18_wired_on_every_dso_block_in_both_arms': all(
                    block['row18_wired']
                    for r in holder.values() for node in r['dispersion_per_node'].values()
                    for block in node['blocks'].values()),
                'alpha_recorded_per_arm_matches_the_declaration': all(
                    math.isclose(r['alpha_applied']['alpha'], ALPHA_BY_ARM[r['arm']]) for r in holder.values()),
                # LARGE ARM ONLY (scoped: the pilot arm is DESIGNED to disperse)
                'large_arm_dispersion_below_the_declared_tolerance': (
                    large['rms_mw_max_over_all_dso_blocks'] <= DISPERSION_ZERO_TOL_MW),
                'large_arm_dispersion_collapses_against_the_pilot': (
                    ratio is not None and ratio <= DISPERSION_COLLAPSE_RATIO),
            }
        else:
            # W44 SWEEP: every item scoped per arm; None = not applicable (its arm is absent)
            positive = [r for r in holder.values() if r['alpha'] > 0.0]
            zero = [r for r in holder.values() if r['alpha'] == 0.0]
            gate_items = {
                'sweep_completed_without_stop': stopped is None,
                'all_arms_ran_the_declared_cycles': all(r['cycles_run'] == CYCLES for r in holder.values()),
                'all_arms_reconcile_their_solves': supported and all(
                    r['solves_in_arm'] == (r['event_level_reconciliation'] or {}).get('expected')
                    for r in holder.values()),
                'process_guard_verified_exactly': not guard_failures,
                'no_blocked_solver_calls': GUARD.counts['blocked_solve'] == 0 and GUARD.counts['blocked_exec'] == 0,
                'row18_wired_on_every_dso_block_of_every_positive_alpha_arm': all(
                    block['row18_wired'] for r in positive for node in r['dispersion_per_node'].values()
                    for block in node['blocks'].values()) if positive else None,
                'row18_absent_on_every_dso_block_of_every_zero_alpha_arm': (all(
                    not block['row18_wired'] for r in zero for node in r['dispersion_per_node'].values()
                    for block in node['blocks'].values()) if zero else None),
                'alpha_recorded_per_arm_matches_the_declaration': all(
                    math.isclose(r['alpha_applied']['alpha'], ALPHA_BY_ARM[r['arm']]) for r in holder.values()),
                'mechanism_and_prices_captured_on_every_dso_block': all(
                    not r.get('capture_errors') for r in holder.values()),
                'large_arm_dispersion_below_the_declared_tolerance': (
                    large['rms_mw_max_over_all_dso_blocks'] <= DISPERSION_ZERO_TOL_MW if large else None),
                'large_arm_dispersion_collapses_against_the_pilot': (
                    (ratio is not None and ratio <= DISPERSION_COLLAPSE_RATIO) if (large and pilot) else None),
            }
        gate_pass = all(v for v in gate_items.values() if v is not None)

        payload = {
            **launch,
            'finished_utc': _utc(), 'wall_clock_s': time.time() - started,
            'arms': holder,
            'limit_comparison': ({
                'pilot_rms_mw_max': pilot['rms_mw_max_over_all_dso_blocks'],
                'large_rms_mw_max': large['rms_mw_max_over_all_dso_blocks'],
                'ratio_large_over_pilot': ratio,
                'pilot_total_charge': pilot['total_charge_all_dso'],
                'large_total_charge': large['total_charge_all_dso'],
                'thresholds': launch['thresholds_declared_before_the_run'],
            } if (pilot and large) else None),
            'sweep_stopped': stopped,
            'sweep_table': [arm_sweep_row(r) for r in sorted(holder.values(), key=lambda r: r['alpha'])],
            'alpha_threshold': alpha_threshold(holder),
            'attribution': attribution(holder),
            'gate_scope': ('the dispersion items apply to the `large` arm ONLY; the `pilot` arm '
                           'is designed to disperse and supplies the reference (CLAUDE.md: scope '
                           'a gate per arm)'),
            'not_a_result': ('two cycles is not a converged run; the pilot arm\'s dispersion here '
                             'is NOT the pilot result and is not comparable with any 1 x 1 figure'),
            'solve_profile': {'declared_total_strict': len(ARMS) * declared_base,
                              'declared_per_arm': declared_base, 'arms_run': len(holder),
                              'reconciled_total': reconciled_total if supported else None,
                              'observed': GUARD.counts['permitted_solve'],
                              'counts': dict(GUARD.counts), 'verify_failures': guard_failures},
            'gate_items': gate_items, 'gate_pass': gate_pass,
        }
        gate_path = os.path.join(out_root, 'gate.json')
        G._refuse_overwrite(gate_path)
        with open(gate_path, 'w') as handle:
            json.dump(payload, handle, indent=1, default=str)

        manifest = {}
        for root, _dirs, fnames in os.walk(out_root):
            for fname in sorted(fnames):
                fpath = os.path.join(root, fname)
                manifest[os.path.relpath(fpath, REPO)] = _sha256_file(fpath)
        manifest_path = os.path.join(out_root, 'manifest_sha256.json')
        G._refuse_overwrite(manifest_path)
        with open(manifest_path, 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        for key, value in gate_items.items():
            _log(f'   {key}: {value}')
        _log(f'GATE_PASS={gate_pass}; ratio large/pilot = {ratio}')
        return EXIT_OK if gate_pass else EXIT_ERROR
    finally:
        GUARD.uninstall()
        if os.path.exists(OWN_LOCK_PATH):
            os.remove(OWN_LOCK_PATH)


if __name__ == '__main__':
    sys.exit(main())
