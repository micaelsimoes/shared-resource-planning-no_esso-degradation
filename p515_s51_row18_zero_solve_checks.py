"""P5.15 Addendum 38 (task W39) -- ZERO-SOLVE checks for row 18 as signed.

Authority: PLANNER_BRIEF_2026-09-13.md Addenda 10 (the signed row 18), 37 (i)-(iv) and
38 (A)-(D); frozen spec v21 `data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json`,
keys `row18_design` and `gates` (the fourth gate is "zero-solve structural checks at 2x2").

Nothing here solves. The three SOLVE-BEARING gates are separate scripts, run by the
Planner in the main checkout AFTER this branch is merged:
  * `p515_s51_srp1_bitwise_gate.py`       -- SRP1 two-cycle bitwise identity
  * `p515_s51_2x2_limit_gate.py`          -- 2x2 two-cycle alpha -> large limit check
  * `p515_s51_single_block_ab.py`         -- single-block 2x2 A/B at alpha in {0, 0.5, large}

CHECKS (each records its own numbers; `pass` per check, `all_checks_pass` overall)

 A  pibar SIGN CHECK, RUN FIRST. Is any hour's pibar_t = sum_sm prob_market[s_m] *
    cost_energy_p[s_m][t] non-positive, on the SRP1, the 2x2 and the PAPER price data?
    The premium floor is applied ONLY if it is (Addendum 38); this check is what decides,
    and it reports the minimum either way.
 B  ROW 18 VACUOUS AT 1 x 1, twice over: `ADMMParameters.interface_deviation_premium`
    defaults to alpha = 0 (row 18 inactive -- every committed result untouched), AND at
    one scenario `add_scenario_commitment_terms` constructs nothing whatever alpha says
    (structural absence, not a zero weight), leaving the objective expression identical.
 C  ROW AND VARIABLE COUNTS AT 2 x 2, per block, at alpha = 0 and alpha = 0.5.
 D  (A) TSO PER-SCENARIO INTERFACE DEVIATIONS FIXED AT ZERO: only the first scenario
    pair's `interface_delta_p/q` is freed (with the +/- rating bounds); every other copy
    stays fixed at 0 and unwired; and the free interface variable inside `pc_adn[dn,s,p]`
    is ONE AND THE SAME VarData for every scenario, so the TSO's interface flow -- hence
    its deviation from its own commitment -- cannot differ across scenarios.
 E  (B) ONE SCENARIO-FREE STORAGE VARIABLE, REFERENCED BY EVERY SCENARIO'S BALANCE ROW;
    the per-scenario copies are PRESENT but appear in no active row and no objective.
    Checked separately for charge and discharge (non-anticipativity covers both).
 F  PRESERVED FIXTURES STILL UNPICKLE (every tracked .pkl in the worktree, by sha256).
 G  FINITE DIFFERENCE: a unit deviation changes the DSO objective expression by exactly
    omega_s * alpha * pibar_t * baseMVA, for P and for Q.
 H  (C) THE SETTLEMENT SPLIT RECONCILES TO THE COVARIANCE IDENTITY on a built 2x2 block:
    settlement - contracted == sum_t baseMVA * Cov_s(pi_t, p_int) recomputed independently;
    and the TSO's deviation part is identically zero (it is pinned).
 I  MODEL.OBJECTIVE == OBJECTIVE_FUNCTION_RULE AT 2 x 2 on every built block -- the
    equality W37 found does not hold above 1 x 1 today, and which restores
    "polish Delta <= 0 by construction".
 J  The retired scenario-deviation quadratic is never called on any production path
    (ARMED tripwire around both functions for the whole run, not an assertion).

ZERO SOLVES: `SolveProfileGuard(permitted=())` is installed before any production import
and `verify(0)`-ed. `NetworkData.optimize` is additionally stubbed for the two production
model builders that would otherwise solve, so production's own construction code runs --
no re-implementation -- while nothing is handed to a solver.

EXACT COMMAND (repo root, canonical interpreter, attached, BOTH streams captured):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s51_row18_zero_solve_checks.py \
      > data/SRP1/Results/P515S51/row18_zero_solve_checks_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S51/row18_zero_solve_checks/row18_zero_solve_checks.json
Exit 0 when every check passes, 1 otherwise.
"""

import hashlib
import json
import os
import pickle
import subprocess
import sys
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W39 row 18 zero-solve checks (never solves)').install()

import pyomo.environ as pe  # noqa: E402
import model_construction_helpers as MCH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402
from network_data import NetworkData  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

STAGE = 'P5.15 Addendum 38 (W39) -- row 18 as signed: zero-solve structural checks'
SCHEMA = 'p515_s51_row18_zero_solve_checks_v1'
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addenda 10, 37, 38',
    'data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json (row18_design, gates)',
]
SPEC_PATH = os.path.join('data', 'SRP1', 'Results', 'P515S51', 'frozen_s51_spec_v21_13cb828c.json')
SPEC_SHA256 = '13cb828c988dfaa5613675a4d34607ca0f74bcc885fc5fb2397904baf19e0530'

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S51', 'row18_zero_solve_checks')
OUT_PATH = os.path.join(OUT_DIR, 'row18_zero_solve_checks.json')

ALPHA_PILOT = 0.50          # Addendum 36: the pilot's alpha
TOL_EXACT_REL = 1e-12
TOL_IDENTITY_ABS = 1e-9


# ======================================================================================
#  scaffolding
# ======================================================================================
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


class _NoSolve:
    """Stubs `NetworkData.optimize` so production's OWN model builders run to completion
    without handing anything to a solver. The stub returns `None` per (year, day), which
    `helper_functions.solver_result_succeeded` reports as a failure, so every post-solve
    extraction in the builders is skipped exactly as it is after a failed local solve."""

    def __init__(self):
        self.calls = 0
        self._original = None

    def __enter__(self):
        self._original = NetworkData.optimize
        stub = self

        def optimize(self_nd, model, *args, **kwargs):
            stub.calls += 1
            return {year: {day: None for day in self_nd.days} for year in self_nd.years}

        NetworkData.optimize = optimize
        return self

    def __exit__(self, *exc):
        NetworkData.optimize = self._original
        return False


class _RetiredQuadraticTripwire:
    """CLAUDE.md rule six: the claim "the retired quadratic is not called on any
    production path" is ARMED for the whole run, never asserted. Every call is recorded
    with its stack."""

    def __init__(self):
        self.calls = []
        self._originals = {}

    def install(self):
        for name in ('_add_tso_scenario_deviation_penalty', '_add_dso_scenario_deviation_penalty'):
            original = getattr(srp, name)
            self._originals[name] = original

            def wrapped(*args, __name=name, __original=original, **kwargs):
                self.calls.append({'function': __name, 'stack': ''.join(traceback.format_stack(limit=25))})
                return __original(*args, **kwargs)

            setattr(srp, name, wrapped)
        return self

    def uninstall(self):
        for name, original in self._originals.items():
            setattr(srp, name, original)


TRIPWIRE = _RetiredQuadraticTripwire()


def _read_planning(label, overrides, work_dir):
    """Production's own reader on a derived case file (`S.derive_case` is the committed
    derivation used by the W37 smoke and the scale measurement). Diagram/result/log dirs
    are redirected into this stage's work directory before reading, so nothing is written
    to a shared location."""
    case, spec, changes = S.derive_case('srp1', overrides)
    case_dir = os.path.join(work_dir, 'case')
    os.makedirs(case_dir, exist_ok=True)
    case_path = os.path.join(case_dir, f'SRP1__{label}.json')
    with open(case_path, 'w') as handle:
        json.dump(case, handle, indent='\t')
    planning = SharedResourcesPlanning(S.DATA_DIR, os.path.relpath(case_path, S.DATA_DIR))
    planning.name = 'SRP1'
    planning.results_dir = os.path.join(work_dir, label, 'Results')
    planning.diagrams_dir = os.path.join(work_dir, label, 'Diagrams')
    planning.logs_dir = os.path.join(planning.results_dir, 'Logs')
    planning.read_planning_problem()
    return planning, {'instance_spec': spec, 'changes_vs_source': changes,
                      'case_path': os.path.relpath(case_path, REPO),
                      'case_sha256': _sha256_file(case_path)}


def _build_blocks(planning, premium_alpha, premium_floor=None):
    """One full production build of every DSO block and the TSO block, with the solve
    stubbed. Returns (dso_models, tso_model, n_stubbed_solve_calls)."""
    consensus_vars, _dual = srp.create_admm_variables(planning)
    candidate = planning.get_initial_candidate_solution()
    with _NoSolve() as stub:
        dso_models, _ = srp.create_distribution_networks_models(
            planning.distribution_networks, consensus_vars, candidate['total_capacity'],
            parallel_execution=False, premium_alpha=premium_alpha, premium_floor=premium_floor)
        tso_model, _ = srp.create_transmission_network_model(
            planning, consensus_vars, candidate['total_capacity'])
    return dso_models, tso_model, stub.calls


def _first_block(planning, models, node_id=None):
    if node_id is None:
        network_data = planning.transmission_network
    else:
        network_data = planning.distribution_networks[node_id]
    year = next(iter(network_data.years))
    day = next(iter(network_data.days))
    model = models[year][day] if node_id is None else models[node_id][year][day]
    return model, network_data.network[year][day], network_data.params, year, day


def _randomize_point(model, seed):
    """Put the block on a DETERMINISTIC, non-trivial point: every unfixed variable gets a
    value derived from `seed` and its own name. Freshly built models sit at 0/1 initial
    values, where most objective terms vanish and an objective identity checked there
    would be vacuous (the first run of this harness, discarded before any commit, read
    0.0 == 0.0 on all 16 blocks). Nothing here is solved or feasible -- the point exists
    only to give the two expressions something to disagree about. Values respect each
    variable's own bounds where it has them, so Pyomo raises no out-of-bounds warning and
    the point is at least box-feasible (it is NOT power-flow feasible, and is not meant
    to be)."""
    n_set = 0
    for component in model.component_objects(pe.Var, active=None):
        for data in component.values():
            if data.fixed:
                continue
            digest = hashlib.sha256(f'{seed}|{data.name}'.encode()).digest()
            unit = int.from_bytes(digest[:8], 'big') / float(1 << 64)   # in [0, 1)
            lower, upper = data.lb, data.ub
            if lower is not None and upper is not None:
                value = lower + unit * (upper - lower)
            elif lower is not None:
                value = lower + 0.1 + 0.9 * unit
            elif upper is not None:
                value = upper - (0.1 + 0.9 * unit)
            else:
                value = 0.1 + 0.9 * unit
            data.set_value(value)
            n_set += 1
    return n_set


def _variables_in(expression, fixed=None):
    """The VarData objects appearing in a Pyomo expression, optionally filtered by
    `fixed`. Uses Pyomo's own identification walker."""
    from pyomo.core.expr.visitor import identify_variables
    out = []
    for var in identify_variables(expression, include_fixed=True):
        if fixed is None or var.fixed == fixed:
            out.append(var)
    return out


# ======================================================================================
#  A -- pibar sign check (run first)
# ======================================================================================
def check_pibar_sign(planning_by_label, results):
    per_instance = {}
    any_non_positive_anywhere = False
    for label, planning in planning_by_label.items():
        entries = []
        minimum = None
        n_non_positive = 0
        n_hours = 0
        network_datas = [('TSO', planning.transmission_network)] + [
            (f'DSO:{node_id}', dn) for node_id, dn in planning.distribution_networks.items()]
        for tag, network_data in network_datas:
            for year in network_data.years:
                for day in network_data.days:
                    network = network_data.network[year][day]
                    n_periods = len(network.cost_energy_p[0])
                    values = [MCH.expected_market_price(network, p) for p in range(n_periods)]
                    n_hours += len(values)
                    block_min = min(values)
                    n_non_positive += sum(1 for v in values if v <= 0.0)
                    minimum = block_min if minimum is None else min(minimum, block_min)
                    entries.append({
                        'block': f'{tag}:{network.name}:{year}:{day}',
                        'n_market_scenarios': len(network.prob_market_scenarios),
                        'prob_market_scenarios': list(network.prob_market_scenarios),
                        'pibar_min': block_min,
                        'pibar_max': max(values),
                        'n_periods': len(values),
                    })
        per_instance[label] = {
            'n_blocks': len(entries),
            'n_hours_checked': n_hours,
            'pibar_min_over_all_blocks': minimum,
            'n_hours_with_pibar_non_positive': n_non_positive,
            'any_non_positive': n_non_positive > 0,
            'blocks': entries,
        }
        any_non_positive_anywhere = any_non_positive_anywhere or n_non_positive > 0

    results['A_pibar_sign'] = {
        'definition': ('pibar_t = sum_sm prob_market_scenarios[s_m] * cost_energy_p[s_m][t] '
                       '(model_construction_helpers.expected_market_price)'),
        'ruling': ('PLANNER_BRIEF Addendum 38: "a premium floor only if any hour\'s mean price '
                   'is non-positive". This check decides it; it is not assumed either way.'),
        'per_instance': per_instance,
        'any_hour_non_positive_anywhere': any_non_positive_anywhere,
        'premium_floor_required': any_non_positive_anywhere,
        'premium_floor_configured_default': ADMMParameters().interface_deviation_premium['floor'],
        'pass': True,   # this check REPORTS; it cannot fail, it decides
    }


# ======================================================================================
#  B -- row 18 vacuous at 1 x 1
# ======================================================================================
def check_vacuous_at_1x1(planning_1x1, results):
    defaults = ADMMParameters().interface_deviation_premium
    dso_models, tso_model, _calls = _build_blocks(planning_1x1, premium_alpha=0.0)
    node_id = next(iter(planning_1x1.distribution_networks))
    model, network, params, year, day = _first_block(planning_1x1, dso_models, node_id)

    n_randomized = _randomize_point(model, seed='B_1x1')
    before = {
        'n_variables_randomized': n_randomized,
        'objective_value': float(pe.value(model.objective.expr)),
        'objective_function_rule_value': float(pe.value(MCH.objective_function_rule(model, params))),
        'n_components': len(list(model.component_objects())),
        'has_row18': hasattr(model, 'row18_deviation_charge'),
        'has_voltage_pin': hasattr(model, 'scenario_voltage_pin'),
    }
    # Call the wiring function AGAIN, this time with the pilot alpha: at one scenario it
    # must construct nothing at all.
    wired = MCH.add_scenario_commitment_terms(model, network, params, premium_alpha=ALPHA_PILOT)
    after = {
        'n_variables_randomized': n_randomized,
        'objective_value': float(pe.value(model.objective.expr)),
        'objective_function_rule_value': float(pe.value(MCH.objective_function_rule(model, params))),
        'n_components': len(list(model.component_objects())),
        'has_row18': hasattr(model, 'row18_deviation_charge'),
        'has_voltage_pin': hasattr(model, 'scenario_voltage_pin'),
    }

    results['B_vacuous_at_1x1'] = {
        'admm_default_alpha': defaults['alpha'],
        'admm_default_floor': defaults['floor'],
        'admm_default_source': defaults['source'],
        'n_scenarios': wired['n_scenarios'],
        'wired_report_at_alpha_pilot': wired,
        'block': f'DSO:{node_id}:{network.name}:{year}:{day}',
        'before': before,
        'after': after,
        'alpha_probe': ALPHA_PILOT,
        'pass': (defaults['alpha'] == 0.0 and defaults['floor'] is None
                 and defaults['source'] == 'default'
                 and wired['n_scenarios'] == 1
                 and not wired['row18_wired'] and not wired['voltage_pin_wired']
                 and not wired['objective_rebuilt']
                 and before == after
                 and before['objective_value'] != 0.0
                 and before['objective_value'] == before['objective_function_rule_value']
                 and not after['has_row18'] and not after['has_voltage_pin']),
    }


# ======================================================================================
#  C -- counts at 2 x 2
# ======================================================================================
def _component_counts(model):
    counts = {}
    for ctype, key in ((pe.Var, 'var'), (pe.Constraint, 'constraint'), (pe.Expression, 'expression')):
        total = 0
        active = 0
        for component in model.component_objects(ctype, active=None):
            total += len(component)
            if ctype is pe.Constraint:
                active += sum(1 for data in component.values() if data.active)
        counts[f'n_{key}_data'] = total
        if ctype is pe.Constraint:
            counts['n_constraint_data_active'] = active
    counts['n_var_data_unfixed'] = sum(
        1 for component in model.component_objects(pe.Var, active=None)
        for data in component.values() if not data.fixed)
    return counts


def check_counts_2x2(built, results):
    detail = {}
    for arm, (dso_models, tso_model, planning) in built.items():
        arm_detail = {}
        model, network, _params, year, day = _first_block(planning, tso_model)
        arm_detail[f'TSO:{network.name}:{year}:{day}'] = _component_counts(model)
        for node_id in planning.distribution_networks:
            model, network, _params, year, day = _first_block(planning, dso_models, node_id)
            block = _component_counts(model)
            block['row18_components'] = {
                name: len(getattr(model, name)) if hasattr(model, name) else 0
                for name in ('row18_dev_p_up', 'row18_dev_p_down', 'row18_dev_q_up',
                             'row18_dev_q_down', 'row18_dev_p_def', 'row18_dev_q_def')}
            block['has_row18_charge'] = hasattr(model, 'row18_deviation_charge')
            block['has_voltage_pin'] = hasattr(model, 'scenario_voltage_pin')
            block['row18_alpha'] = float(pe.value(model.row18_alpha)) if hasattr(model, 'row18_alpha') else None
            arm_detail[f'DSO:{node_id}:{network.name}:{year}:{day}'] = block
        detail[arm] = arm_detail

    # Expected: at alpha > 0 each DSO block gains 4 * n_scenarios * n_periods deviation
    # variables and 2 * n_scenarios * n_periods rows; at alpha = 0 it gains none.
    planning = built['alpha_0.5'][2]
    node_id = next(iter(planning.distribution_networks))
    model, network, _params, year, day = _first_block(planning, built['alpha_0.5'][0], node_id)
    n_scen = len(model.scenarios_market) * len(model.scenarios_operation)
    n_per = len(model.periods)
    key = f'DSO:{node_id}:{network.name}:{year}:{day}'
    expected_vars = 4 * n_scen * n_per
    expected_rows = 2 * n_scen * n_per
    observed_vars = sum(detail['alpha_0.5'][key]['row18_components'][n]
                        for n in ('row18_dev_p_up', 'row18_dev_p_down', 'row18_dev_q_up', 'row18_dev_q_down'))
    observed_rows = sum(detail['alpha_0.5'][key]['row18_components'][n]
                        for n in ('row18_dev_p_def', 'row18_dev_q_def'))
    var_delta = detail['alpha_0.5'][key]['n_var_data'] - detail['alpha_0'][key]['n_var_data']
    row_delta = (detail['alpha_0.5'][key]['n_constraint_data_active']
                 - detail['alpha_0'][key]['n_constraint_data_active'])

    results['C_counts_2x2'] = {
        'per_arm': detail,
        'reference_block': key,
        'n_scenarios': n_scen,
        'n_periods': n_per,
        'expected_row18_variables': expected_vars,
        'observed_row18_variables': observed_vars,
        'expected_row18_rows': expected_rows,
        'observed_row18_rows': observed_rows,
        'var_data_delta_alpha05_minus_alpha0': var_delta,
        'active_row_delta_alpha05_minus_alpha0': row_delta,
        'pass': (observed_vars == expected_vars and observed_rows == expected_rows
                 and var_delta == expected_vars and row_delta == expected_rows
                 and detail['alpha_0'][key]['has_row18_charge'] is False
                 and detail['alpha_0.5'][key]['has_row18_charge'] is True
                 and detail['alpha_0'][key]['has_voltage_pin'] is True
                 and detail['alpha_0.5'][key]['has_voltage_pin'] is True),
    }


# ======================================================================================
#  D -- TSO per-scenario interface deviations fixed at zero
# ======================================================================================
def check_tso_deviation_zero(planning, tso_models, results):
    model, network, _params, year, day = _first_block(planning, tso_models)
    s_m0, s_o0 = MCH.sess_na_scenario(model)

    free_first = []
    non_first_fixed_at_zero = []
    non_first_free = []
    for dn in model.adn_nodes:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    for component in (model.interface_delta_p, model.interface_delta_q):
                        data = component[dn, s_m, s_o, p]
                        if (s_m, s_o) == (s_m0, s_o0):
                            free_first.append({'fixed': data.fixed, 'lb': data.lb, 'ub': data.ub})
                        elif data.fixed and abs(pe.value(data)) == 0.0:
                            non_first_fixed_at_zero.append(1)
                        else:
                            non_first_free.append(str(data))

    # the free interface variable inside pc_adn must be ONE AND THE SAME VarData for every
    # scenario -- that is what makes the TSO's per-scenario deviation identically zero.
    same_object = True
    per_scenario_free_ids = {}
    dn0 = next(iter(model.adn_nodes))
    p0 = next(iter(model.periods))
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            free_vars = _variables_in(model.pc_adn[dn0, s_m, s_o, p0], fixed=False)
            per_scenario_free_ids[f'{s_m}_{s_o}'] = sorted(str(v) for v in free_vars)
    reference = per_scenario_free_ids[f'{s_m0}_{s_o0}']
    for key, names in per_scenario_free_ids.items():
        if names != reference:
            same_object = False

    results['D_tso_deviation_fixed_at_zero'] = {
        'block': f'TSO:{network.name}:{year}:{day}',
        'first_scenario_pair': [s_m0, s_o0],
        'n_first_pair_entries': len(free_first),
        'n_first_pair_free': sum(1 for e in free_first if not e['fixed']),
        'first_pair_bounds_sample': free_first[0] if free_first else None,
        'n_non_first_fixed_at_zero': len(non_first_fixed_at_zero),
        'non_first_free_offenders': non_first_free[:20],
        'free_variables_in_pc_adn_per_scenario': per_scenario_free_ids,
        'pc_adn_free_variables_identical_across_scenarios': same_object,
        'pass': (len(non_first_free) == 0
                 and len(free_first) > 0
                 and sum(1 for e in free_first if not e['fixed']) == len(free_first)
                 and same_object),
    }


# ======================================================================================
#  E -- one scenario-free storage variable, referenced by every scenario's balance row
# ======================================================================================
def _active_row_variable_names(model):
    names = set()
    for component in model.component_objects(pe.Constraint, active=True):
        for data in component.values():
            if not data.active:
                continue
            names.update(str(v) for v in _variables_in(data.body))
    for component in model.component_objects(pe.Objective, active=None):
        for data in component.values():
            names.update(str(v) for v in _variables_in(data.expr))
    return names


def check_storage_scenario_free(planning, dso_models, tso_models, results):
    detail = {}
    overall_pass = True
    targets = [('TSO', None, tso_models)] + [('DSO', node_id, dso_models)
                                             for node_id in planning.distribution_networks]
    for kind, node_id, models in targets:
        model, network, _params, year, day = _first_block(planning, models, node_id)
        s_m0, s_o0 = MCH.sess_na_scenario(model)
        e0 = next(iter(model.shared_energy_storages))
        bus = network.shared_energy_storages[e0].bus
        node_idx = network.get_node_idx(bus)
        p0 = next(iter(model.periods))

        referenced = {}
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                body = model.node_balance_p[node_idx, s_m, s_o, p0].body
                names = sorted(str(v) for v in _variables_in(body)
                               if str(v).startswith('shared_es_pnet'))
                referenced[f'{s_m}_{s_o}'] = names
        expected = [f'shared_es_pnet[{e0},{s_m0},{s_o0},{p0}]']
        balance_ok = all(names == expected for names in referenced.values())

        # the per-scenario copies must be PRESENT but appear in no active row and no objective
        wired_names = _active_row_variable_names(model)
        present = {}
        unwired = {}
        for family in ('shared_es_pch', 'shared_es_pdch', 'shared_es_pnet', 'shared_es_qnet',
                       'shared_es_soc', 'shared_es_pch_hat', 'shared_es_pdch_hat'):
            component = getattr(model, family)
            copies = [component[e0, s_m, s_o, p]
                      for s_m in model.scenarios_market
                      for s_o in model.scenarios_operation
                      for p in model.periods
                      if (s_m, s_o) != (s_m0, s_o0)]
            present[family] = len(copies)
            unwired[family] = sum(1 for c in copies if str(c) not in wired_names)
        free_copies_wired = {k: present[k] - unwired[k] for k in present}
        unwired_ok = all(present[k] == unwired[k] for k in present)

        block_pass = balance_ok and unwired_ok and all(v > 0 for v in present.values())
        overall_pass = overall_pass and block_pass
        detail[f'{kind}:{node_id}:{network.name}:{year}:{day}'] = {
            'first_scenario_pair': [s_m0, s_o0],
            'shared_ess_index': e0,
            'shared_ess_bus': bus,
            'shared_es_pnet_referenced_by_each_scenario_balance_row': referenced,
            'expected_single_variable': expected,
            'balance_rows_reference_one_scenario_free_variable': balance_ok,
            'per_scenario_copies_present': present,
            'per_scenario_copies_unwired': unwired,
            'per_scenario_copies_still_wired': free_copies_wired,
            'pass': block_pass,
        }
    results['E_storage_scenario_free'] = {'per_block': detail, 'pass': overall_pass}


# ======================================================================================
#  F -- preserved fixtures still unpickle
# ======================================================================================
def check_fixtures(results):
    tracked = _git(['ls-files', '--', '*.pkl']).splitlines()
    tracked = [p for p in tracked if p and os.path.exists(os.path.join(REPO, p))]
    loaded, failed = [], []
    for rel in tracked:
        path = os.path.join(REPO, rel)
        try:
            with open(path, 'rb') as handle:
                payload = pickle.load(handle)
            loaded.append({'path': rel, 'sha256': _sha256_file(path), 'type': type(payload).__name__})
        except Exception as error:  # noqa: BLE001
            failed.append({'path': rel, 'sha256': _sha256_file(path),
                           'error': f'{type(error).__name__}: {error}'})
    results['F_fixtures_unpickle'] = {
        'scope': ('every .pkl TRACKED IN GIT and present in this worktree. Untracked fixtures '
                  'held only in the main checkout (e.g. data/SRP1/Results/P512R/cycle21_pre_setup/'
                  'snapshot.pkl) are NOT in this worktree and were not loaded here; that part of '
                  'the claim is scoped to the tracked set.'),
        'n_tracked': len(tracked),
        'n_loaded': len(loaded),
        'n_failed': len(failed),
        'failed': failed,
        'loaded': loaded,
        'pass': len(tracked) > 0 and not failed,
    }


# ======================================================================================
#  G -- finite difference on the row 18 charge
# ======================================================================================
def check_finite_difference(planning, dso_models, results):
    node_id = next(iter(planning.distribution_networks))
    model, network, params, year, day = _first_block(planning, dso_models, node_id)
    if not hasattr(model, 'row18_deviation_charge'):
        results['G_finite_difference'] = {'pass': False, 'error': 'row 18 not wired on this block'}
        return
    alpha = float(pe.value(model.row18_alpha))
    s_base = network.baseMVA
    probes = []
    ok = True
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            omega = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
            for p in (0, len(model.periods) // 2, len(model.periods) - 1):
                pibar = float(pe.value(model.row18_premium[p]))
                for family, var in (('p_up', model.row18_dev_p_up), ('p_down', model.row18_dev_p_down),
                                    ('q_up', model.row18_dev_q_up), ('q_down', model.row18_dev_q_down)):
                    base = float(pe.value(MCH.objective_function_rule(model, params)))
                    var[s_m, s_o, p].set_value(1.0)
                    perturbed = float(pe.value(MCH.objective_function_rule(model, params)))
                    var[s_m, s_o, p].set_value(0.0)
                    observed = perturbed - base
                    expected = omega * alpha * pibar * s_base
                    rel = abs(observed - expected) / abs(expected) if expected else abs(observed)
                    probe_ok = rel <= TOL_EXACT_REL
                    ok = ok and probe_ok
                    probes.append({'scenario': f'{s_m}_{s_o}', 'period': p, 'variable': family,
                                   'omega': omega, 'alpha': alpha, 'pibar': pibar, 'baseMVA': s_base,
                                   'expected': expected, 'observed': observed, 'rel_error': rel,
                                   'pass': probe_ok})
    results['G_finite_difference'] = {
        'block': f'DSO:{node_id}:{network.name}:{year}:{day}',
        'identity': 'd(objective_function_rule)/d(unit deviation variable) == omega_s * alpha * pibar_t * baseMVA',
        'tolerance_rel': TOL_EXACT_REL,
        'n_probes': len(probes),
        'probes': probes,
        'pass': ok and bool(probes),
    }


# ======================================================================================
#  H -- the settlement split reconciles to the covariance identity
# ======================================================================================
def check_settlement_split(planning, dso_models, tso_models, results):
    node_id = next(iter(planning.distribution_networks))
    model, network, _params, year, day = _first_block(planning, dso_models, node_id)

    # Give the block a point with genuine scenario spread: the reference generator's active
    # power (`pg_adn = pg[ref] - shared_ess_pnet`) is set to a different value per scenario.
    ref_gen_idx = network.get_reference_gen_idx()
    for index, (s_m, s_o) in enumerate((s_m, s_o) for s_m in model.scenarios_market
                                       for s_o in model.scenarios_operation):
        for p in model.periods:
            model.pg[ref_gen_idx, s_m, s_o, p].set_value(0.10 + 0.05 * index + 0.01 * p)

    settlement = float(pe.value(model.interface_settlement))
    contracted = float(pe.value(model.interface_settlement_contracted))
    deviation = float(pe.value(model.interface_settlement_deviation))

    # independent recomputation of sum_t baseMVA * Cov_s(pi_t, p_int_{s,t})
    covariance = 0.0
    for p in model.periods:
        pibar = MCH.expected_market_price(network, p)
        e_pi_p = 0.0
        e_p = 0.0
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                omega = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
                p_int = float(pe.value(model.pg_adn[s_m, s_o, p]))
                e_pi_p += omega * network.cost_energy_p[s_m][p] * p_int
                e_p += omega * p_int
        covariance += network.baseMVA * (e_pi_p - pibar * e_p)

    dso_ok = (abs(deviation - covariance) <= TOL_IDENTITY_ABS * max(1.0, abs(covariance))
              and abs((settlement - contracted) - deviation) <= TOL_IDENTITY_ABS * max(1.0, abs(deviation)))

    # the TSO is pinned: its deviation part must be identically zero
    tso_model, tso_network, _p, tyear, tday = _first_block(planning, tso_models)
    tso_deviation = float(pe.value(tso_model.interface_settlement_deviation))
    tso_ok = abs(tso_deviation) <= TOL_IDENTITY_ABS

    results['H_settlement_split'] = {
        'dso_block': f'DSO:{node_id}:{network.name}:{year}:{day}',
        'tso_block': f'TSO:{tso_network.name}:{tyear}:{tday}',
        'n_market_scenarios': len(model.scenarios_market),
        'market_prices_differ_across_scenarios': (
            len({tuple(network.cost_energy_p[s_m]) for s_m in model.scenarios_market}) > 1),
        'settlement': settlement,
        'contracted': contracted,
        'deviation_reported': deviation,
        'covariance_recomputed': covariance,
        'abs_difference': abs(deviation - covariance),
        'tso_deviation_part': tso_deviation,
        'tolerance_abs': TOL_IDENTITY_ABS,
        'identity': 'interface_settlement - interface_settlement_contracted == sum_t baseMVA * Cov_s(pi_t, p_int)',
        'pass': dso_ok and tso_ok,
    }


# ======================================================================================
#  I -- model.objective == objective_function_rule at 2 x 2
# ======================================================================================
def check_objective_equals_rule(arms, results):
    detail = {}
    ok = True
    n_trivial = 0
    for arm, (dso_models, tso_models, planning) in arms.items():
        targets = [('TSO', None, tso_models)] + [('DSO', node_id, dso_models)
                                                 for node_id in planning.distribution_networks]
        for kind, node_id, models in targets:
            network_data = (planning.transmission_network if node_id is None
                            else planning.distribution_networks[node_id])
            for year in network_data.years:
                for day in network_data.days:
                    model = models[year][day] if node_id is None else models[node_id][year][day]
                    params = network_data.params
                    key = f'{arm}|{kind}:{node_id}:{network_data.network[year][day].name}:{year}:{day}'
                    n_randomized = _randomize_point(model, seed=key)
                    objective_value = float(pe.value(model.objective.expr))
                    rule_value = float(pe.value(MCH.objective_function_rule(model, params)))
                    difference = objective_value - rule_value
                    block_ok = abs(difference) <= TOL_IDENTITY_ABS * max(1.0, abs(rule_value))
                    if objective_value == 0.0:
                        n_trivial += 1
                    ok = ok and block_ok
                    detail[key] = {
                        'n_variables_randomized': n_randomized,
                        'model_objective': objective_value,
                        'objective_function_rule': rule_value,
                        'difference': difference,
                        'has_retired_quadratic': hasattr(model, 'scenario_deviation_penalty'),
                        'has_row18': hasattr(model, 'row18_deviation_charge'),
                        'has_voltage_pin': hasattr(model, 'scenario_voltage_pin'),
                        'pass': block_ok,
                    }
    results['I_objective_equals_rule_2x2'] = {
        'why': ('W37 found that above 1 x 1 the hull polish minimises `model.objective` (which '
                'carried the retired quadratic) while Delta is measured on '
                '`objective_function_rule` (which did not), so "Delta <= 0 by construction" did '
                'not hold there. With the quadratic unwired and row 18 + the voltage pin inside '
                'the rule, the two coincide again.'),
        'evaluated_at': ('a deterministic pseudo-random point (`_randomize_point`), NOT the '
                         'built model\'s 0/1 initial point, where both sides evaluate to 0.0 '
                         'and the identity would be vacuous'),
        'per_block': detail,
        'n_blocks_evaluating_to_zero': n_trivial,
        'tolerance_abs': TOL_IDENTITY_ABS,
        'pass': ok and bool(detail) and n_trivial == 0,
    }


# ======================================================================================
#  main
# ======================================================================================
def main():
    if os.path.exists(OUT_PATH):
        print(f'REFUSED: output exists (write-once): {OUT_PATH}', file=sys.stderr)
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    work_dir = os.path.join(OUT_DIR, 'work')
    os.makedirs(work_dir, exist_ok=True)

    results = {}
    instances = {}
    TRIPWIRE.install()
    try:
        # ---- instances -------------------------------------------------------------
        planning_1x1, meta_1x1 = _read_planning(
            '1x1', {'years': {'2025': 5}, 'num_market_scenarios': None,
                    'num_operation_scenarios': None}, work_dir)
        instances['1x1'] = meta_1x1
        planning_2x2_a0, meta_2x2 = _read_planning(
            '2x2_alpha0', {'years': {'2025': 5}, 'num_market_scenarios': 2,
                           'num_operation_scenarios': 2}, work_dir)
        instances['2x2'] = meta_2x2
        planning_2x2_a5, _ = _read_planning(
            '2x2_alpha05', {'years': {'2025': 5}, 'num_market_scenarios': 2,
                            'num_operation_scenarios': 2}, work_dir)
        planning_paper, meta_paper = _read_planning(
            'paper', {'years': S.PAPER_YEARS, 'num_market_scenarios': 5,
                      'num_operation_scenarios': 5}, work_dir)
        instances['paper'] = meta_paper

        # ---- A: the sign check, first ----------------------------------------------
        check_pibar_sign({'srp1_1x1': planning_1x1, 'derived_2x2': planning_2x2_a0,
                          'paper_5x5': planning_paper}, results)
        del planning_paper

        # ---- B: vacuity at 1 x 1 ----------------------------------------------------
        check_vacuous_at_1x1(planning_1x1, results)
        del planning_1x1

        # ---- builds at 2 x 2 --------------------------------------------------------
        dso_a0, tso_a0, calls_a0 = _build_blocks(planning_2x2_a0, premium_alpha=0.0)
        dso_a5, tso_a5, calls_a5 = _build_blocks(planning_2x2_a5, premium_alpha=ALPHA_PILOT)
        results['build_record'] = {
            'stubbed_optimize_calls_alpha_0': calls_a0,
            'stubbed_optimize_calls_alpha_0.5': calls_a5,
            'note': ('production\'s own builders ran; every local solve was replaced by a stub '
                     'that reports failure, so the post-solve extractions were skipped'),
        }

        check_counts_2x2({'alpha_0': (dso_a0, tso_a0, planning_2x2_a0),
                          'alpha_0.5': (dso_a5, tso_a5, planning_2x2_a5)}, results)
        check_tso_deviation_zero(planning_2x2_a5, tso_a5, results)
        check_storage_scenario_free(planning_2x2_a5, dso_a5, tso_a5, results)
        check_finite_difference(planning_2x2_a5, dso_a5, results)
        # I randomizes every block's point, so it runs after the structural checks; H then
        # sets its own point on the alpha = 0 build, so it runs after I.
        check_objective_equals_rule({'alpha_0': (dso_a0, tso_a0, planning_2x2_a0),
                                     'alpha_0.5': (dso_a5, tso_a5, planning_2x2_a5)}, results)
        check_settlement_split(planning_2x2_a0, dso_a0, tso_a0, results)

        # ---- F: fixtures -------------------------------------------------------------
        check_fixtures(results)
    finally:
        TRIPWIRE.uninstall()
        GUARD.uninstall()

    results['J_retired_quadratic_never_called'] = {
        'mechanism': ('ARMED tripwire around `_add_tso_scenario_deviation_penalty` and '
                      '`_add_dso_scenario_deviation_penalty` for the whole run (CLAUDE.md rule six: '
                      'armed, never asserted)'),
        'n_calls': len(TRIPWIRE.calls),
        'calls': TRIPWIRE.calls,
        'pass': len(TRIPWIRE.calls) == 0,
    }

    verify_failures = GUARD.verify(expected_solves=0)
    all_pass = all(v.get('pass', True) for v in results.values() if isinstance(v, dict)) and not verify_failures

    payload = {
        'schema': SCHEMA,
        'stage': STAGE,
        'authority': AUTHORITY,
        'frozen_spec': {'path': SPEC_PATH, 'sha256_declared': SPEC_SHA256,
                        'sha256_observed': _sha256_file(os.path.join(REPO, SPEC_PATH))
                        if os.path.exists(os.path.join(REPO, SPEC_PATH)) else None},
        'timestamp_utc': _utc(),
        'interpreter': sys.executable,
        'script': os.path.basename(__file__),
        'script_sha256': _sha256_file(os.path.abspath(__file__)),
        'git_head': _git(['rev-parse', 'HEAD']),
        'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
        'instances': instances,
        'alpha_pilot': ALPHA_PILOT,
        'solve_profile_guard': {'permitted': [], 'counts': dict(GUARD.counts),
                                'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': bool(all_pass),
    }
    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[W39 zero-solve checks] wrote {OUT_PATH}')
    for name, value in results.items():
        if isinstance(value, dict) and 'pass' in value:
            print(f'  {name}: {"PASS" if value["pass"] else "FAIL"}')
    print(f'[W39 zero-solve checks] all_checks_pass = {all_pass}; '
          f'solve guard counts = {dict(GUARD.counts)}')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
