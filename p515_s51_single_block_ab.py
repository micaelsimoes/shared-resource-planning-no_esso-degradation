"""
P5.15 Addendum 38 (task W39) -- GATE 3 of 3: the SINGLE-BLOCK 2 x 2 A/B at
alpha in {0, 0.5, large}, BEFORE the pilot, reporting alpha * pibar_t against each DSO's
INTERNAL FLEXIBILITY PRICE per hour.

Authority: frozen spec v21 `data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json`,
`gates[2]` -- "single-block 2x2 A/B at alpha in {0, 0.5, large} BEFORE the pilot; report
alpha*pibar_t against each DSO's internal flexibility price per hour";
PLANNER_BRIEF_2026-09-13.md Addendum 38 ("dispersion is a threshold outcome").

WHY THIS GATE EXISTS. Dispersion under row 18 is a THRESHOLD outcome: a DSO deviates from
its commitment when deviating is cheaper than the alternative it has, and its alternative
is its own internal flexibility, priced at `network.cost_flex[s_m][t]` (row 2). So the
decisive comparison is the PRICE RATIO alpha * pibar_t / c_flex_t, hour by hour and DSO by
DSO -- if alpha * pibar_t sits far above c_flex_t at every hour, the pilot will show no
dispersion whatever alpha is chosen, and the pilot would measure nothing. This gate
reports that ratio (zero-solve, for EVERY DSO and every hour) and, on ONE DSO block,
measures the realized dispersion at the three alphas, so the ratio's prediction can be
checked against behaviour before the 7-10 h pilot is spent.

WHAT IS SOLVED, and what is not. Each arm builds the SELECTED DSO's blocks through
production's own builder (`shared_resources_planning.create_distribution_networks_models`
with the arm's alpha) and lets that builder solve them -- the DSO's standalone
(uncoordinated) SMOPF, which is what the ADMM initialization solves. That is ONE solve per
(year, day) block per arm, declared before the run and guard-verified EXACTLY. NO ADMM
cycle runs here; no TSO or ESSO block is built; the dispersion measured is the DSO's own
response to the premium at its uncoordinated point, NOT a converged coordinated result.

ARMS: alpha = 0 (the free-deviation reference: row 18 is not constructed at all),
alpha = 0.50 (the pilot value), alpha = ALPHA_LARGE (the pinned limit).

GATE ITEMS ARE SCOPED PER ARM (CLAUDE.md stage template):
  * every arm: its blocks solved; row 18 present iff alpha > 0; solves reconciled.
  * alpha = 0 arm: row 18 structurally absent (no charge component at all).
  * alpha = large arm: dispersion below the declared tolerance.
  * across arms: dispersion non-increasing in alpha.
The PRICE TABLE is reported for every DSO and hour and is not gated -- it is the
measurement this gate exists to produce.

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s51_single_block_ab.py --label <write-once-label> \\
        > data/SRP1/Results/P515S51/single_block_ab_launch_<label>.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S51/single_block_ab/<label>/

OPTIONAL `--alphas a1 a2 ...` (P5.15 Addendum 39 ruling 2, task W41 "locate alpha*";
frozen spec v22 `data/SRP1/Results/P515S52/frozen_s52_spec_v22_5d8df1e8.json`
`ruling2_alpha`): replaces the arm list. Omitted, the arms are exactly the committed
default ALPHAS, so the committed runs stay reproducible. Gate items are scoped per arm: the
alpha = 0 item applies only if 0 is an arm, the alpha = large item only if ALPHA_LARGE is an
arm; items whose arm is absent are recorded as not applicable and skipped. Every run also
records `alpha_threshold`: alpha* = the smallest tested alpha whose dispersion (max over
blocks of the RMS, MW) is at or below DISPERSION_ZERO_TOL_MW, its bracket, and the price
ratio alpha* * pibar_t / c_flex_t from the zero-solve price table.
MECHANISM CAPTURE (P5.15 Addendum 39 ruling 2 follow-up, task W42, ZERO extra solves):
every block of every arm also records `mechanism` -- read with `pe.value` off the model the
arm has ALREADY solved, never re-solved -- so the question "how does the DSO remove its
deviation d as alpha rises?" can be answered from the artifact: per scenario and hour, the
interface P, the committed P, d, the row 18 legs d+/d-, the load, the priced DOWN and
unpriced UP flexibility legs (fl_reg loads, the loads `flexibility_cost` prices), the
priced down-leg cost c_flex * down, load curtailment, ordinary / shared ESS, the non-
reference generation, the losses-and-slacks residual sum(pg_node) - sum(pc_node), the
P day-balance slacks, and the per-scenario objective components.
Exit 0 on PASS, 1 on FAIL, 2 on a precondition refusal.
"""

import argparse
import hashlib
import json
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
GUARD = SolveProfileGuard(PERMITTED, label='P5.15 W39 gate 3 -- single-block 2x2 A/B').install()

import pyomo.environ as pe  # noqa: E402
import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import p56a_oracle as O  # noqa: E402
import model_construction_helpers as MCH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from helper_functions import solver_result_succeeded  # noqa: E402

STAGE = ('P5.15 Addendum 38 W39 gate 3 -- single-block 2x2 A/B at alpha in {0, 0.5, large}; '
         'alpha*pibar_t against each DSO\'s internal flexibility price per hour')
SCHEMA = 'p515_s51_single_block_ab_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 38',
             'data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json gates[2]']

OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S51', 'single_block_ab')
OWN_LOCK_PATH = os.path.join(REPO, '.p515_s51_single_block_ab.lock')

OVERRIDE_YEARS = {'2025': 5}
OVERRIDE_MARKET_SCENARIOS = 2
OVERRIDE_OPERATION_SCENARIOS = 2
INSTANCE_LABEL = 's51_2x2_single_block'
SELECTED_NODE = 7                 # the node the F2 demonstration and the ladder use
ALPHA_LARGE = 1000.0
ALPHAS = (0.0, 0.50, ALPHA_LARGE)
DISPERSION_ZERO_TOL_MW = 1.0e-2   # declared BEFORE the run; the alpha = large arm only

EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
THREAD_CAP_ENV = dict(S.THREAD_CAP_ENV)
PRODUCTION_FILES_TO_CHECK_CLEAN = (
    'model_construction_helpers.py', 'shared_resources_planning.py', 'network.py',
    'admm_parameters.py', 'p56a_oracle.py', 'p515_s44_scale_measurement.py',
    os.path.basename(__file__))


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(message):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W39-gate3] {message}', flush=True)


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


def check_preconditions(out_dir):
    failures = []
    for path in (OWN_LOCK_PATH, G.CAMPAIGN_LOCK_PATH, os.path.join(REPO, '.p515_g_gate.lock'),
                 os.path.join(REPO, '.p515_s44_scale_measurement.lock'),
                 os.path.join(REPO, '.p515_s51_2x2_limit_gate.lock')):
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


def capture_path_checklist(planning):
    """CLAUDE.md rule eleven: every quantity this gate's specification requires must have a
    capture path, verified BEFORE the run."""
    node = next(iter(planning.distribution_networks))
    network_data = planning.distribution_networks[node]
    year = next(iter(network_data.years))
    day = next(iter(network_data.days))
    network = network_data.network[year][day]
    return {
        'dispersion_metric_callable': callable(getattr(srp, '_get_local_interface_dispersion', None)),
        'row18_wiring_callable': callable(getattr(MCH, 'add_scenario_commitment_terms', None)),
        'expected_market_price_callable': callable(getattr(MCH, 'expected_market_price', None)),
        'network_carries_cost_flex': hasattr(network, 'cost_flex') and len(network.cost_flex) > 0,
        'network_carries_cost_energy_p': hasattr(network, 'cost_energy_p') and len(network.cost_energy_p) > 0,
        'builder_accepts_premium_alpha': 'premium_alpha' in __import__('inspect').signature(
            srp.create_distribution_networks_models).parameters,
        'selected_node_present': SELECTED_NODE in planning.distribution_networks,
        # W42 mechanism capture (zero-solve, read off the solved model)
        'flexibility_cost_callable': callable(getattr(MCH, 'flexibility_cost', None)),
        'flexibility_p_day_balance_slack_penalty_callable': callable(
            getattr(MCH, 'flexibility_p_day_balance_slack_penalty', None)),
        'interface_pf_p_distribution_def_callable': callable(
            getattr(MCH, 'interface_pf_p_distribution_def', None)),
    }


def price_table(planning):
    """ZERO-SOLVE, for EVERY DSO and every hour: alpha * pibar_t against the DSO's own
    internal flexibility price c_flex_t (row 2's price -- the alternative to deviating).
    Reported per market scenario AND probability-weighted, since both prices are
    scenario-dependent above one market scenario."""
    table = {}
    for node_id, network_data in planning.distribution_networks.items():
        for year in network_data.years:
            for day in network_data.days:
                network = network_data.network[year][day]
                n_periods = len(network.cost_energy_p[0])
                pibar = [MCH.expected_market_price(network, p) for p in range(n_periods)]
                flex_expected = []
                for p in range(n_periods):
                    total = 0.0
                    for s_m in range(len(network.prob_market_scenarios)):
                        total += network.prob_market_scenarios[s_m] * network.cost_flex[s_m][p]
                    flex_expected.append(total)
                rows = []
                for p in range(n_periods):
                    entry = {'period': p, 'pibar': pibar[p], 'c_flex_expected': flex_expected[p],
                             'c_flex_by_market_scenario': [network.cost_flex[s_m][p] for s_m in
                                                           range(len(network.prob_market_scenarios))],
                             'pi_by_market_scenario': [network.cost_energy_p[s_m][p] for s_m in
                                                       range(len(network.prob_market_scenarios))]}
                    for alpha in ALPHAS:
                        entry[f'alpha_{alpha}_premium'] = alpha * pibar[p]
                        entry[f'alpha_{alpha}_over_c_flex'] = (
                            (alpha * pibar[p]) / flex_expected[p] if flex_expected[p] else None)
                    rows.append(entry)
                table[f'DSO:{node_id}:{network.name}:{year}:{day}'] = {
                    'n_periods': n_periods,
                    'pibar_min': min(pibar), 'pibar_max': max(pibar),
                    'c_flex_expected_min': min(flex_expected), 'c_flex_expected_max': max(flex_expected),
                    'n_hours_premium_below_c_flex': {
                        str(alpha): sum(1 for p in range(n_periods)
                                        if alpha * pibar[p] < flex_expected[p])
                        for alpha in ALPHAS},
                    'rows': rows,
                }
    return table


def mechanism_record(model, network, params):
    """W42 (ZERO SOLVES): the per-scenario, per-hour composition of the DSO's interface P,
    read with `pe.value` off the ALREADY-SOLVED block model. MW (x baseMVA); one period is
    one hour, so a sum over periods of MW is MWh. Identity recorded, per (s, t):

        interface_p = sum_i pc_node - sum_{g != ref} pg + residual - shared_ess_at_ref
        sum_i pc_node = load + flex_up - flex_down - curt_down + curt_up + es_pnet + shared_es_pnet
        residual      = sum_i pg_node - sum_i pc_node   (losses, shunts, node-balance slacks)

    `flex_*` are summed over the fl_reg loads -- the loads `flexibility_cost` prices, whose
    DOWN legs cost c_flex[s_m][t] * baseMVA and whose UP legs are unpriced, subject only to
    the per-scenario P day balance sum_t up = sum_t down (+ bounded slacks)."""
    s_base = network.baseMVA
    ref_gen = network.get_reference_gen_idx()
    ref_node = network.get_reference_node_id()
    fl_loads = [c for c in model.loads if network.loads[c].fl_reg] if hasattr(model, 'flex_p_up') else []

    def _v(component, *index):
        return float(pe.value(component[index]))

    per_scenario = {}
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            prob = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
            c_flex = network.cost_flex[s_m]
            series = {k: [] for k in (
                'interface_p_mw', 'committed_p_mw', 'd_p_mw', 'row18_dev_p_up_mw', 'row18_dev_p_down_mw',
                'load_p_mw', 'flex_p_up_mw', 'flex_p_down_mw', 'flex_q_up_mvar', 'flex_q_down_mvar',
                'flex_p_down_cost', 'curt_p_down_mw', 'curt_p_up_mw', 'es_pnet_mw', 'shared_es_pnet_mw',
                'gen_nonref_p_mw', 'gen_curtaillable_avail_mw', 'residual_losses_slacks_mw',
                'shared_ess_at_ref_mw', 'c_flex')}
            for p in model.periods:
                interface = float(pe.value(model.pg_adn[s_m, s_o, p])) * s_base
                committed = float(pe.value(model.expected_interface_pf_p[p])) * s_base
                series['interface_p_mw'].append(interface)
                series['committed_p_mw'].append(committed)
                series['d_p_mw'].append(interface - committed)
                series['row18_dev_p_up_mw'].append(
                    _v(model.row18_dev_p_up, s_m, s_o, p) * s_base if hasattr(model, 'row18_dev_p_up') else None)
                series['row18_dev_p_down_mw'].append(
                    _v(model.row18_dev_p_down, s_m, s_o, p) * s_base if hasattr(model, 'row18_dev_p_down') else None)
                series['load_p_mw'].append(sum(_v(model.pc, c, s_m, s_o, p) for c in model.loads) * s_base)
                down = sum(_v(model.flex_p_down, c, s_m, s_o, p) for c in fl_loads) * s_base
                series['flex_p_up_mw'].append(sum(_v(model.flex_p_up, c, s_m, s_o, p) for c in fl_loads) * s_base)
                series['flex_p_down_mw'].append(down)
                series['flex_q_up_mvar'].append(sum(_v(model.flex_q_up, c, s_m, s_o, p) for c in fl_loads) * s_base)
                series['flex_q_down_mvar'].append(sum(_v(model.flex_q_down, c, s_m, s_o, p) for c in fl_loads) * s_base)
                series['flex_p_down_cost'].append(float(c_flex[p]) * down)
                series['c_flex'].append(float(c_flex[p]))
                if hasattr(model, 'pc_curt_down'):
                    series['curt_p_down_mw'].append(sum(_v(model.pc_curt_down, c, s_m, s_o, p) for c in model.loads) * s_base)
                    series['curt_p_up_mw'].append(sum(_v(model.pc_curt_up, c, s_m, s_o, p) for c in model.loads) * s_base)
                else:
                    series['curt_p_down_mw'].append(0.0)
                    series['curt_p_up_mw'].append(0.0)
                series['es_pnet_mw'].append(
                    sum(_v(model.es_pnet, e, s_m, s_o, p) for e in model.energy_storages) * s_base
                    if hasattr(model, 'es_pnet') else 0.0)
                # the ONE scenario-free shared-ESS variable every scenario's node balance uses
                s_m0, s_o0 = MCH.sess_na_scenario(model)
                series['shared_es_pnet_mw'].append(
                    sum(_v(model.shared_es_pnet, e, s_m0, s_o0, p) for e in model.shared_energy_storages) * s_base)
                series['shared_ess_at_ref_mw'].append(
                    (float(pe.value(model.pg[ref_gen, s_m, s_o, p])) * s_base) - interface)
                series['gen_nonref_p_mw'].append(
                    sum(_v(model.pg, g, s_m, s_o, p) for g in model.generators if g != ref_gen) * s_base)
                series['gen_curtaillable_avail_mw'].append(
                    sum(_v(model.pg_avail, g, s_o, p) for g in model.generators
                        if network.generators[g].is_curtaillable()) * s_base
                    if hasattr(model, 'pg_avail') else None)
                series['residual_losses_slacks_mw'].append(
                    (sum(float(pe.value(model.pg_node[i, s_m, s_o, p])) for i in model.nodes)
                     - sum(float(pe.value(model.pc_node[i, s_m, s_o, p])) for i in model.nodes)) * s_base)
            day_balance_slack = None
            if hasattr(model, 'slack_flex_p_balance_up'):
                day_balance_slack = {
                    'up_mwh': sum(_v(model.slack_flex_p_balance_up, c, s_m, s_o) for c in fl_loads) * s_base,
                    'down_mwh': sum(_v(model.slack_flex_p_balance_down, c, s_m, s_o) for c in fl_loads) * s_base}
            objective_components = {}
            for name in ('gen_cost_scenario', 'flex_cost_scenario', 'load_curt_cost_scenario',
                         'gen_curt_penalty_scenario', 'ess_utilization_cost_penalty_scenario',
                         'slack_penalties_scenario', 'ess_complementarity_penalty_scenario'):
                if hasattr(model, name):
                    objective_components[name] = float(pe.value(getattr(model, name)[s_m, s_o]))
            objective_components['flexibility_p_day_balance_slack_penalty'] = float(pe.value(
                MCH.flexibility_p_day_balance_slack_penalty(model, network, s_m, s_o, params)))
            per_scenario[f'{s_m}_{s_o}'] = {
                'probability': prob, 'series': series,
                'flex_p_day_balance_slack': day_balance_slack,
                'objective_components': objective_components,
            }
    totals = {}
    for name in ('total_gen_cost', 'total_flex_cost', 'total_load_curt_cost', 'total_gen_curt_penalty',
                 'total_ess_utilization_cost_penalty', 'total_slack_penalties',
                 'total_ess_complementarity_penalties', 'row18_deviation_charge', 'interface_settlement'):
        if hasattr(model, name):
            totals[name] = float(pe.value(getattr(model, name)))
    totals['interface_settlement_weight'] = float(pe.value(model.interface_settlement_weight))
    totals['voltage_pin_weighted'] = (
        float(pe.value(model.scenario_voltage_pin_weight) * pe.value(model.scenario_voltage_pin))
        if hasattr(model, 'scenario_voltage_pin') else 0.0)
    return {
        'units': 'MW / MVAr per hour (x baseMVA); currency for *_cost and objective components',
        'base_mva': s_base, 'reference_gen_idx': ref_gen, 'reference_node_id': ref_node,
        'n_fl_reg_loads': len(fl_loads), 'n_loads': len(list(model.loads)),
        'n_energy_storages': len(list(model.energy_storages)),
        'n_shared_energy_storages': len(list(model.shared_energy_storages)),
        'per_scenario': per_scenario, 'objective_totals': totals,
    }


def mechanism_record_or_error(model, network, params, where):
    """The capture runs AFTER the arm's solves; a capture defect must not destroy the arm's
    solved result, so it is recorded (and logged) in place of the record, never swallowed."""
    try:
        return mechanism_record(model, network, params)
    except Exception as error:  # noqa: BLE001
        _log(f'MECHANISM CAPTURE FAILED at {where}: {error!r}')
        return {'capture_error': repr(error)}


def run_arm(alpha, out_root, holder):
    arm = f'alpha_{alpha}'
    record = holder.setdefault(arm, {'arm': arm, 'alpha': alpha})
    eval_id = f'p515s51_ab_{os.path.basename(out_root)}_{arm}'
    record['eval_id'] = eval_id
    before = GUARD.counts['permitted_solve']
    t0 = time.time()

    planning = O.fresh_planning(eval_id)
    planning.params.admm.interface_deviation_premium = {
        'alpha': float(alpha), 'floor': None, 'source': f'W39 gate 3 arm {arm!r}'}
    record['alpha_applied'] = dict(planning.params.admm.interface_deviation_premium)

    consensus_vars, _dual = srp.create_admm_variables(planning)
    candidate = planning.get_initial_candidate_solution()
    record['candidate_label'] = 'x = 0 (no shared-ESS investment) -- the initial candidate'

    network_data = planning.distribution_networks[SELECTED_NODE]
    models, results = srp.create_distribution_networks_models(
        {SELECTED_NODE: network_data}, consensus_vars, candidate['total_capacity'],
        parallel_execution=False,
        premium_alpha=planning.params.admm.interface_deviation_premium['alpha'],
        premium_floor=planning.params.admm.interface_deviation_premium['floor'])

    blocks = {}
    for year in network_data.years:
        for day in network_data.days:
            model = models[SELECTED_NODE][year][day]
            network = network_data.network[year][day]
            result = results[SELECTED_NODE][year][day]
            dispersion = srp._get_local_interface_dispersion(model, network)
            blocks[f'{year}:{day}'] = {
                'solved': bool(solver_result_succeeded(result)),
                'termination': str(getattr(getattr(result, 'solver', None), 'termination_condition', None)),
                'row18_wired': hasattr(model, 'row18_deviation_charge'),
                'row18_alpha_on_model': (float(pe.value(model.row18_alpha))
                                         if hasattr(model, 'row18_alpha') else None),
                'voltage_pin_wired': hasattr(model, 'scenario_voltage_pin'),
                'objective_function_rule_value': float(
                    pe.value(MCH.objective_function_rule(model, network_data.params))),
                'model_objective_value': float(pe.value(model.objective.expr)),
                'interface_settlement': float(pe.value(model.interface_settlement)),
                'interface_settlement_contracted': float(pe.value(model.interface_settlement_contracted)),
                'interface_settlement_deviation': float(pe.value(model.interface_settlement_deviation)),
                'voltage_mismatch': srp._get_local_scenario_voltage_mismatch(model, network),
                'dispersion': dispersion,
                'mechanism': mechanism_record_or_error(model, network, network_data.params,
                                                       f'{arm} {year}:{day}'),
            }
    record['blocks'] = blocks
    record['n_blocks'] = len(blocks)
    record['all_solved'] = all(b['solved'] for b in blocks.values())
    record['rms_mw_max_over_blocks'] = max(b['dispersion']['p']['rms_mw'] for b in blocks.values())
    record['max_abs_mw_over_blocks'] = max(b['dispersion']['p']['max_abs_mw'] for b in blocks.values())
    record['total_charge'] = sum(b['dispersion']['row18_charge'] for b in blocks.values())
    record['solves'] = GUARD.counts['permitted_solve'] - before
    record['wall_s'] = time.time() - t0
    record['objective_convention'] = (
        'block-local, UNWEIGHTED by year/day/discount; `objective_function_rule` is the '
        'quantity Q(x) is built from (it carries row 18 and the voltage pin, and the '
        'voltage pin is subtracted back out at the recourse level, Addendum 38 (D))')
    return record


def alpha_threshold(holder, table):
    """alpha* = the smallest tested alpha whose dispersion (max over the selected DSO's blocks
    of the RMS, MW) is at or below DISPERSION_ZERO_TOL_MW; bracket = (the largest tested alpha
    with dispersion above the tolerance, alpha*). Also the price ratio alpha * pibar_t /
    c_flex_expected_t at alpha* (min / max over hours, per DSO block) and the hours with the
    premium below c_flex, per arm, for the selected DSO."""
    alphas = sorted(ALPHAS)
    rms = {a: holder[f'alpha_{a}']['rms_mw_max_over_blocks'] for a in alphas}
    at_or_below = [a for a in alphas if rms[a] <= DISPERSION_ZERO_TOL_MW]
    above = [a for a in alphas if rms[a] > DISPERSION_ZERO_TOL_MW]
    alpha_star = at_or_below[0] if at_or_below else None
    below_star = [a for a in above if alpha_star is None or a < alpha_star]
    ratio_at_star = {}
    if alpha_star is not None:
        for key, block in table.items():
            ratios = [r[f'alpha_{alpha_star}_over_c_flex'] for r in block['rows']
                      if r[f'alpha_{alpha_star}_over_c_flex'] is not None]
            ratio_at_star[key] = {'min': min(ratios) if ratios else None,
                                  'max': max(ratios) if ratios else None,
                                  'n_hours_ratio_defined': len(ratios),
                                  'n_hours_ratio_below_1': sum(1 for x in ratios if x < 1.0)}
    selected = {k: v for k, v in table.items() if k.startswith(f'DSO:{SELECTED_NODE}:')}
    return {
        'definition': ('alpha* = smallest tested alpha with dispersion (max over blocks of the '
                       'per-block RMS interface-P dispersion, MW) <= the declared tolerance; '
                       'bracket = [largest tested alpha with dispersion above the tolerance, '
                       'alpha*]'),
        'dispersion_zero_tol_mw': DISPERSION_ZERO_TOL_MW,
        'dispersion_rms_mw_by_alpha': rms,
        'dispersion_max_abs_mw_by_alpha': {a: holder[f'alpha_{a}']['max_abs_mw_over_blocks'] for a in alphas},
        'charge_by_alpha': {a: holder[f'alpha_{a}']['total_charge'] for a in alphas},
        'alpha_star': alpha_star,
        'bracket': [below_star[-1] if below_star else None, alpha_star],
        'any_alpha_above_tol_after_alpha_star': [a for a in above if alpha_star is not None and a > alpha_star],
        'alpha_star_is_smallest_positive_tested': (
            alpha_star is not None and alpha_star == min([a for a in alphas if a > 0.0], default=None)),
        'ratio_alpha_star_pibar_over_c_flex_expected_by_dso_block': ratio_at_star,
        'n_hours_premium_below_c_flex_selected_dso_by_block': {
            k: v['n_hours_premium_below_c_flex'] for k, v in selected.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument('--label', required=True, help='write-once output label')
    parser.add_argument('--alphas', type=float, nargs='+', default=None,
                        help='arm list (W41); omitted = the committed default ALPHAS')
    args = parser.parse_args()
    global ALPHAS
    alpha_list_source = 'default ALPHAS (as committed)'
    if args.alphas is not None:
        if len(set(args.alphas)) != len(args.alphas) or any(a < 0.0 for a in args.alphas):
            print(f'REFUSED: --alphas must be distinct and non-negative: {args.alphas}', file=sys.stderr)
            return EXIT_REFUSED
        ALPHAS = tuple(sorted(float(a) for a in args.alphas))
        alpha_list_source = ('command line --alphas (P5.15 Addendum 39 ruling 2, W41; '
                             'frozen_s52_spec_v22_5d8df1e8.json ruling2_alpha)')

    out_root = os.path.join(OUT_ROOT, args.label)
    failures = check_preconditions(out_root)
    if failures:
        for failure in failures:
            print(f'[W39-gate3 PRECONDITION FAILED] {failure}', file=sys.stderr)
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
        case_path = os.path.join(case_dir, 'SRP1__s51_2x2_ab.json')
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
            'alphas': list(ALPHAS), 'alpha_list_source': alpha_list_source,
            'selected_node': SELECTED_NODE,
            'dispersion_zero_tol_mw_declared_before_the_run': DISPERSION_ZERO_TOL_MW,
            'guard_permitted': [list(p) for p in PERMITTED],
            'started_utc': _utc(), 'pid': os.getpid(),
        }

        stages = S.StageLog(os.path.join(out_root, 'stages.jsonl'),
                            S.Watchdog(out_root, 's51ab', limit_bytes=int(12 * (1 << 30))))
        stages.wd.start()
        planning0 = S.read_planning_from_derived_case(launch, out_root, stages)
        launch['scenario_checksum'] = S.inject_oracle_baseline(O, planning0, launch)
        launch['planning_dimensions'] = S.planning_dimensions(planning0)
        provenance = S.provenance_record(planning0, INSTANCE_LABEL, launch['scenario_checksum'])
        launch['provenance'] = provenance
        non_checksum = [f for f in provenance['gate_failures'] if f['identity'] != 'scenario checksum']
        if non_checksum:
            raise RuntimeError(f'provenance: non-canonical identity: {non_checksum}')

        checklist = capture_path_checklist(planning0)
        launch['capture_path_checklist_asserted_before_run'] = checklist
        if not all(checklist.values()):
            raise RuntimeError(f'capture-path checklist failed: '
                               f'{[k for k, v in checklist.items() if not v]}')

        # declared BEFORE the run: one solve per (year, day) block of the selected DSO, per arm
        network_data = planning0.distribution_networks[SELECTED_NODE]
        n_blocks = len(list(network_data.years)) * len(list(network_data.days))
        declared_total = len(ALPHAS) * n_blocks
        launch['declared_solve_profile'] = {
            'n_blocks_per_arm': n_blocks, 'n_arms': len(ALPHAS),
            'declared_total_strict': declared_total,
            'derivation': 'one standalone DSO SMOPF solve per (year, day) block per arm',
        }

        # the price table is ZERO-SOLVE and is written BEFORE any arm runs
        launch['price_table'] = price_table(planning0)
        launch_path = os.path.join(out_root, 'launch.json')
        G._refuse_overwrite(launch_path)
        with open(launch_path, 'w') as handle:
            json.dump(launch, handle, indent=1, default=str)
        _log(f'declared {declared_total} solves ({len(ALPHAS)} arms x {n_blocks} blocks); '
             f'price table written for {len(launch["price_table"])} DSO blocks')

        # The legacy run lock (`.p515_g_gate.lock`). It is released by
        # `_acquire_exclusive_run_lock`'s own atexit handler -- the module exposes no
        # explicit release entry point, and the committed gates do not release it either.
        G._acquire_exclusive_run_lock()
        for alpha in ALPHAS:
            _log(f'arm alpha = {alpha}')
            record = run_arm(alpha, out_root, holder)
            _log(f"arm alpha={alpha}: solved={record['all_solved']} "
                 f"rms_mw={record['rms_mw_max_over_blocks']} charge={record['total_charge']}")

        guard_failures = GUARD.verify(declared_total)
        rms = {a: holder[f'alpha_{a}']['rms_mw_max_over_blocks'] for a in ALPHAS}
        monotone = all(rms[ALPHAS[i]] >= rms[ALPHAS[i + 1]] - DISPERSION_ZERO_TOL_MW
                       for i in range(len(ALPHAS) - 1))

        gate_items = {
            'every_arm_solved_every_block': all(r['all_solved'] for r in holder.values()),
            'solve_count_verified_exactly': not guard_failures,
            'no_blocked_solver_calls': GUARD.counts['blocked_solve'] == 0 and GUARD.counts['blocked_exec'] == 0,
            'row18_present_iff_alpha_positive': all(
                all(b['row18_wired'] == (r['alpha'] > 0.0) for b in r['blocks'].values())
                for r in holder.values()),
            'alpha_zero_arm_has_no_row18_component': (all(
                b['row18_wired'] is False and b['dispersion']['row18_charge'] == 0.0
                for b in holder['alpha_0.0']['blocks'].values()) if 0.0 in ALPHAS else None),
            'large_arm_dispersion_below_the_declared_tolerance': (
                rms[ALPHA_LARGE] <= DISPERSION_ZERO_TOL_MW if ALPHA_LARGE in ALPHAS else None),
            'dispersion_non_increasing_in_alpha': monotone,
            'objective_equals_objective_function_rule_on_every_block': all(
                abs(b['model_objective_value'] - b['objective_function_rule_value'])
                <= 1e-9 * max(1.0, abs(b['objective_function_rule_value']))
                for r in holder.values() for b in r['blocks'].values()),
        }
        gate_items_not_applicable = [k for k, v in gate_items.items() if v is None]
        gate_items = {k: v for k, v in gate_items.items() if v is not None}
        gate_pass = all(gate_items.values())
        threshold = alpha_threshold(holder, launch['price_table'])

        payload = {
            **launch,
            'finished_utc': _utc(), 'wall_clock_s': time.time() - started,
            'arms': holder,
            'dispersion_by_alpha_mw': rms,
            'charge_by_alpha': {a: holder[f'alpha_{a}']['total_charge'] for a in ALPHAS},
            'gate_scope': ('the "no row 18 component" item applies to the alpha = 0 arm ONLY and '
                           'the "dispersion below tolerance" item to the alpha = large arm ONLY; '
                           'the alpha = 0.50 arm is the measurement, not a gated reference '
                           '(CLAUDE.md: scope a gate per arm)'),
            'not_a_result': ('these are UNCOORDINATED standalone DSO solves at a single candidate, '
                             'not ADMM results; they establish the threshold behaviour before the '
                             'pilot, and are not comparable with any coordinated figure'),
            'solve_profile': {'declared_total_strict': declared_total,
                              'observed': GUARD.counts['permitted_solve'],
                              'counts': dict(GUARD.counts), 'verify_failures': guard_failures},
            'gate_items': gate_items, 'gate_pass': gate_pass,
            'gate_items_not_applicable_arm_absent': gate_items_not_applicable,
            'alpha_threshold': threshold,
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
        _log(f'GATE_PASS={gate_pass}; dispersion by alpha (MW) = {rms}')
        _log(f"alpha* = {threshold['alpha_star']}; bracket = {threshold['bracket']}")
        return EXIT_OK if gate_pass else EXIT_ERROR
    finally:
        GUARD.uninstall()
        if os.path.exists(OWN_LOCK_PATH):
            os.remove(OWN_LOCK_PATH)


if __name__ == '__main__':
    sys.exit(main())
