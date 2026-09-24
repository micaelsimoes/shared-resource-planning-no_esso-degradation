"""
P5.15 Addendum 40 ruling 2 (task W51) -- ZERO-SOLVE checks for "row 18 INACTIVE at the initialisation solve,
activated with the settlement weight".

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 2; frozen spec v23
`data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json`, key `ruling2_init_fix`.

THE CHANGE UNDER TEST (shared_resources_planning.py):
  * `_set_row18_inactive_for_initialisation(block)` -- called by both DSO initialisation builders
    (`create_distribution_networks_models_sequential`, `create_distribution_network_model`) right after
    `add_scenario_commitment_terms` and before the initialisation solve: records the run's alpha as the
    Param `row18_alpha_admm` and sets the existing MUTABLE `row18_alpha` to 0;
  * `_activate_row18_with_settlement(block)` -- called in `_prepare_distribution_objectives_for_admm` right
    after `interface_settlement_weight` is set to 1: restores the recorded alpha (a Param update, no rebuild).

Nothing here solves. `SolveProfileGuard(permitted=())` is installed before any production import and verified
at exactly 0 (the imported W39 checks module installs its own zero guard too; both are verified). Every
`.optimize` of the planning instance under test is replaced ON THE INSTANCE by a recording stub that, at the
moment the solve would happen, records the state IPOPT would have been handed, and returns "no result" (None per
block), which production reads as a failed local solve.

CHECKS (each records its own numbers; `pass` per check; `all_checks_pass` overall)
 A  INIT-PHASE ALPHA IS 0 THROUGH PRODUCTION'S OWN PATH (2 x 2, run alpha = 0.5): `_run_operational_planning`
    itself is called (fresh initialisation, `parallel_execution` False, asserted) with
    `ADMMParameters.interface_deviation_premium = {alpha 0.5, floor None}` in force. At every DSO block's would-be
    initialisation solve: row 18 is WIRED (deviation Vars + rows + charge present), `row18_alpha` == 0.0,
    `row18_alpha_admm` == 0.5, `interface_settlement_weight` == 0.0, and a unit step on each of the four
    deviation variables changes the ACTIVE objective (`model.objective.expr`) and the charge by exactly 0.0.
    The init returns `initialization_failed` (every solve was stubbed), which is what makes its models
    available here before `_prepare_*`.
 A2 THE PARALLEL BUILDER (`create_distribution_network_model`, the ProcessPool worker function) called
    in-process on one node with alpha = 0.5: the same init-phase state at its would-be solve.
 B  ACTIVATION WITH THE SETTLEMENT WEIGHT: production's `_prepare_distribution_objectives_for_admm` on the
    models A returned: on EVERY DSO block `row18_alpha` == 0.5 (the run's alpha, exactly), the settlement weight
    == 1.0; `row18_premium[t]` unchanged from the init phase and equal to `expected_market_price(network, t)`
    (no floor in force); `row18_deviation_charge` is the SAME component object with an identical expression
    string; unit steps now change the objective rule and the charge by exactly omega_s * 0.5 * pibar_t * baseMVA.
    Then `update_distribution_models_to_admm` (placeholder objective scale, recorded -- nothing is solved) builds
    the ADMM objective, which references `row18_alpha` BY IDENTITY (so every ADMM cycle sees the Param's value,
    0.5), and a scan of every tracked root-level .py finds `row18_alpha.set_value(` ONLY in the two new
    functions -- nothing in the cycle loop writes it.
 C  STRUCTURE UNCHANGED ACROSS THE PHASES, AND VS alpha = 0: per DSO block the component-name set and the
    Var / Constraint / Expression counts at the init solve equal those after `_prepare_*`; vs the same path at
    alpha = 0 (row 18 not wired -- the fix is a no-op there, so every alpha = 0 result is untouched) the block
    differs by exactly 4 n T deviation Vars, 2 n T rows, and the components {row18_* , row18_alpha_admm}.
 D  1 x 1 (THE SRP1 CASE FILE, UNCHANGED): the same production path at alpha = 0.5 and at alpha = 0: in neither
    phase does any block carry a row 18 component, `row18_alpha_admm` or the voltage pin, and the component-name
    sets and counts are identical between the two alphas in both phases -- nothing is wired at one scenario.
 E  THE UNCOORDINATED BENCHMARK IS UNCHANGED: (1) the source of `_run_operational_planning_without_coordination`
    in the live module is byte-identical to that function at the parent commit (e1f98be8) and does not read
    `interface_deviation_premium`; (2) running it (stubbed) on the 2 x 2 instance with alpha = 0.5 IN FORCE in
    ADMMParameters, no DSO block at its solve carries row 18 or `row18_alpha_admm`.
 F  PRESERVED FIXTURES STILL UNPICKLE: every .pkl tracked in git and present in this worktree (W39's
    `check_fixtures`, by import), PLUS, read-only from the main checkout, the untracked anchors CLAUDE.md names:
    `data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl` and every `data/SRP1/Results/FrozenSMOPF/**/*.pkl`.

INSTANCES: 2 x 2 = `p515_s44_scale_measurement.derive_case('srp1', {years {'2025': 5}, 2 market, 2 operation})`,
the W39 zero-solve checks' 2 x 2 (one year: 4 days x 3 DSOs + TSO), chosen over the 5-year pilot instance to keep
memory low while a campaign runs in the main checkout; 1 x 1 = `derive_case('srp1', {})`, the SRP1 case file
unchanged. Derived case files are written to <out>/cases/ and hash-recorded; the reader's Diagrams/Results dirs
go to --scratch.

EXACT COMMAND (worktree/repo root, canonical interpreter, attached, BOTH streams captured):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_init_fix_zero_solve_checks.py \\
      --label r1 --scratch <dir outside the repo> \\
      > data/SRP1/Results/P515S53/init_fix_zero_solve_checks_r1_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/init_fix_zero_solve_checks/<label>/
    {init_fix_zero_solve_checks.json, manifest_sha256.json, cases/}
Exit 0 when every check passes, 1 otherwise.
"""

import argparse
import ast
import glob
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W51 init-fix zero-solve checks (never solves)').install()

import pyomo.environ as pe  # noqa: E402
from pyomo.core.expr.visitor import identify_mutable_parameters  # noqa: E402
import model_construction_helpers as MCH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import p515_s51_row18_zero_solve_checks as W39C  # noqa: E402 -- installs its own zero guard (verified below)
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

STAGE = 'P5.15 Addendum 40 ruling 2 (W51) -- row 18 inactive at the initialisation solve: zero-solve checks'
SCHEMA = 'p515_s53_init_fix_zero_solve_checks_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 2',
             'data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json (ruling2_init_fix)']
SPEC_PATH = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'frozen_s53_spec_v23_39a07fd8.json')
SPEC_SHA256 = '39a07fd8'   # prefix, as named in the file name; the full hash is recorded on output
PARENT_COMMIT = 'e1f98be8'
OUT_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'init_fix_zero_solve_checks')

ALPHA_RUN = 0.50
PREMIUM_RUN = {'alpha': ALPHA_RUN, 'floor': None, 'source': 'W51 zero-solve check (run alpha)'}
PREMIUM_ZERO = {'alpha': 0.0, 'floor': None, 'source': 'W51 zero-solve check (alpha 0 reference)'}
OVERRIDES_2X2 = {'years': {'2025': 5}, 'num_market_scenarios': 2, 'num_operation_scenarios': 2}
OVERRIDES_1X1 = {}
PLACEHOLDER_OBJECTIVE_SCALE = 93635360.0   # SRP1's fixed sigma; only builds the ADMM objective, nothing solved
ROW18_COMPONENTS = ('row18_alpha', 'row18_premium', 'row18_dev_p_up', 'row18_dev_p_down', 'row18_dev_q_up',
                    'row18_dev_q_down', 'row18_dev_p_def', 'row18_dev_q_def', 'row18_deviation_charge')
NEW_COMPONENT = 'row18_alpha_admm'
DEV_FAMILIES = ('row18_dev_p_up', 'row18_dev_p_down', 'row18_dev_q_up', 'row18_dev_q_down')
# Declared before the run. The charge is a short linear sum: exact to 1e-12. The objective is a large-magnitude
# sum, so a unit-step difference carries cancellation error of order eps * |objective|: 1e-9 relative.
TOL_REL_CHARGE = 1e-12
TOL_REL_OBJECTIVE = 1e-9
NAMED_CTYPES = (pe.Var, pe.Param, pe.Constraint, pe.Expression, pe.Objective, pe.Block)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W51-checks] {msg}', flush=True)


def _sha256_file(path):
    return W39C._sha256_file(path)


def _git(args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


# ======================================================================================================================
#  instances (production reader on a derived case file; the committed derivation `S.derive_case`)
# ======================================================================================================================
def read_planning(label, overrides, case_dir, scratch):
    case, spec, changes = S.derive_case('srp1', overrides)
    os.makedirs(case_dir, exist_ok=True)
    case_path = os.path.join(case_dir, f'SRP1__{label}.json')
    if os.path.exists(case_path):
        raise SystemExit(f'REFUSED: derived case exists (write-once): {case_path}')
    with open(case_path, 'w') as handle:
        json.dump(case, handle, indent='\t')
    planning = SharedResourcesPlanning(S.DATA_DIR, os.path.relpath(case_path, S.DATA_DIR))
    planning.name = 'SRP1'
    planning.results_dir = os.path.join(scratch, label, 'Results')
    planning.diagrams_dir = os.path.join(scratch, label, 'Diagrams')
    planning.logs_dir = os.path.join(planning.results_dir, 'Logs')
    planning.read_planning_problem()
    with open(S.SOURCE_CASE) as handle:
        source = json.load(handle)
    return planning, {'label': label, 'derive_base': 'srp1', 'overrides': overrides, 'instance_spec': spec,
                      'changes_vs_source': changes, 'case_path': os.path.relpath(case_path, REPO),
                      'case_sha256': _sha256_file(case_path),
                      'source_case': os.path.relpath(S.SOURCE_CASE, REPO),
                      'source_case_sha256': _sha256_file(S.SOURCE_CASE),
                      'semantically_equal_to_source_case': case == source}


# ======================================================================================================================
#  block-state capture
# ======================================================================================================================
def _component_names(block):
    """Named modelling components (Pyomo's implicit index Sets excluded, they follow the components)."""
    return sorted(c.local_name for c in block.component_objects(ctype=NAMED_CTYPES, descend_into=False))


def _pval(block, name):
    return float(pe.value(getattr(block, name))) if hasattr(block, name) else None


def _unit_step(block, network, params, var_name, s_m, s_o, p):
    """A unit step on one deviation variable, evaluated on the ACTIVE objective (or the rule when the block's
    objective has been deactivated) and on the charge expression; the variable is restored exactly."""
    var = getattr(block, var_name)[s_m, s_o, p]
    active = block.objective.active

    def objective_value():
        if active:
            return float(pe.value(block.objective.expr))
        return float(pe.value(MCH.objective_function_rule(block, params)))

    original = var.value
    base_obj, base_charge = objective_value(), float(pe.value(block.row18_deviation_charge))
    var.set_value((original or 0.0) + 1.0)
    step_obj, step_charge = objective_value(), float(pe.value(block.row18_deviation_charge))
    var.set_value(original)
    omega = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
    pibar = float(pe.value(block.row18_premium[p]))
    return {'variable': f'{var_name}[{s_m},{s_o},{p}]', 'objective_evaluated': 'objective.expr' if active
            else 'objective_function_rule (objective deactivated)', 'omega': omega, 'pibar': pibar,
            'baseMVA': network.baseMVA, 'alpha_on_model': float(pe.value(block.row18_alpha)),
            'delta_objective': step_obj - base_obj, 'delta_charge': step_charge - base_charge,
            'unit_rate_at_alpha_run': omega * ALPHA_RUN * pibar * network.baseMVA}


def _probes(block, network, params):
    s_m = next(iter(block.scenarios_market))
    s_o = next(iter(block.scenarios_operation))
    periods = list(block.periods)
    out = []
    for p in (periods[0], periods[len(periods) // 2]):
        for name in DEV_FAMILIES:
            out.append(_unit_step(block, network, params, name, s_m, s_o, p))
    return out


def block_state(block, network, params, with_probes=False):
    state = {
        'row18_components_present': {n: hasattr(block, n) for n in ROW18_COMPONENTS + (NEW_COMPONENT,)},
        'row18_wired': hasattr(block, 'row18_deviation_charge'),
        'voltage_pin_wired': hasattr(block, 'scenario_voltage_pin'),
        'row18_alpha': _pval(block, 'row18_alpha'),
        'row18_alpha_admm': _pval(block, NEW_COMPONENT),
        'interface_settlement_weight': float(pe.value(block.interface_settlement_weight)),
        'counts': W39C._component_counts(block),
        'component_names': _component_names(block),
        'n_scenarios': len(block.scenarios_market) * len(block.scenarios_operation),
        'n_periods': len(block.periods),
    }
    if state['row18_wired']:
        state['row18_premium'] = [float(pe.value(block.row18_premium[p])) for p in block.periods]
        state['expected_market_price'] = [float(MCH.expected_market_price(network, p)) for p in block.periods]
        state['row18_charge_expression_id'] = id(block.row18_deviation_charge)
        state['row18_charge_expression_str_sha256'] = hashlib.sha256(
            str(block.row18_deviation_charge.expr).encode()).hexdigest()
        state['row18_n_deviation_vars'] = sum(len(getattr(block, n)) for n in DEV_FAMILIES)
        state['row18_n_rows'] = len(block.row18_dev_p_def) + len(block.row18_dev_q_def)
        if with_probes:
            state['unit_steps'] = _probes(block, network, params)
    return state


class InitProbe:
    """Replaces `.optimize` on ONE planning instance's agents (instance attributes; deleted on exit). At each
    would-be solve it records the block state IPOPT would have been handed, then returns "no result"."""

    def __init__(self, planning, probes=True):
        self.planning = planning
        self.probes = probes
        self.calls = []
        self.dso_state = {}
        self.tso_state = {}

    def _dso_stub(self, node_id, dn):
        def stub(model, *args, **kwargs):
            self.calls.append(f'dso:{node_id}')
            premium = dict(self.planning.params.admm.interface_deviation_premium)
            for y in dn.years:
                for d in dn.days:
                    first = self.probes and y == next(iter(dn.years)) and d == next(iter(dn.days))
                    st = block_state(model[y][d], dn.network[y][d], dn.params, with_probes=first)
                    st['premium_in_force_at_the_solve'] = premium
                    self.dso_state.setdefault(node_id, {})[f'{y}:{d}'] = st
            return {y: {d: None for d in dn.days} for y in dn.years}
        return stub

    def _tso_stub(self, tn):
        def stub(model, *args, **kwargs):
            self.calls.append('tso')
            for y in tn.years:
                for d in tn.days:
                    self.tso_state[f'{y}:{d}'] = {
                        'row18_components_present': any(hasattr(model[y][d], n) for n in ROW18_COMPONENTS),
                        'row18_alpha_admm_present': hasattr(model[y][d], NEW_COMPONENT)}
            return {y: {d: None for d in tn.days} for y in tn.years}
        return stub

    def __enter__(self):
        tn = self.planning.transmission_network
        tn.optimize = self._tso_stub(tn)
        for node_id, dn in self.planning.distribution_networks.items():
            dn.optimize = self._dso_stub(node_id, dn)
        self.esso = S.Interceptor()
        self.planning.shared_ess_data.optimize = self.esso.esso(self.planning.shared_ess_data)
        return self

    def __exit__(self, *exc):
        del self.planning.transmission_network.optimize
        for dn in self.planning.distribution_networks.values():
            del dn.optimize
        del self.planning.shared_ess_data.optimize
        return False

    def counts(self):
        out = {}
        for c in self.calls + list(self.esso.calls):
            out[c] = out.get(c, 0) + 1
        return out


def run_init(planning, premium):
    """Production's `_run_operational_planning`, fresh initialisation, every solve stubbed."""
    planning.parallel_execution = False
    planning.params.admm.interface_deviation_premium = dict(premium)
    candidate = planning.get_initial_candidate_solution()
    with InitProbe(planning) as probe:
        out = srp._run_operational_planning(planning, candidate)
    state = out[6]
    return probe, state['models'], {'returned_tuple_len': len(out), 'converged_flag': out[0],
                                    'initialization_failed': state.get('initialization_failed'),
                                    'stub_calls': probe.counts(),
                                    'parallel_execution': planning.parallel_execution,
                                    'premium_in_force': dict(planning.params.admm.interface_deviation_premium)}


def post_prepare_states(planning, dso_models, probes=True):
    srp._prepare_distribution_objectives_for_admm(planning.distribution_networks, dso_models)
    out = {}
    for node_id, dn in planning.distribution_networks.items():
        for y in dn.years:
            for d in dn.days:
                first = probes and y == next(iter(dn.years)) and d == next(iter(dn.days))
                out.setdefault(node_id, {})[f'{y}:{d}'] = block_state(dso_models[node_id][y][d], dn.network[y][d],
                                                                      dn.params, with_probes=first)
    return out


def _strip(state):
    return {k: v for k, v in state.items() if k not in ('component_names',)}


# ======================================================================================================================
#  A / B / C at 2 x 2
# ======================================================================================================================
def _steps_ok(steps, expect_zero):
    ok = bool(steps)
    for s in steps:
        if expect_zero:
            s['pass'] = s['delta_objective'] == 0.0 and s['delta_charge'] == 0.0
        else:
            rate = s['unit_rate_at_alpha_run']
            s['rel_err_objective'] = abs(s['delta_objective'] - rate) / abs(rate) if rate else abs(s['delta_objective'])
            s['rel_err_charge'] = abs(s['delta_charge'] - rate) / abs(rate) if rate else abs(s['delta_charge'])
            s['pass'] = (s['rel_err_objective'] <= TOL_REL_OBJECTIVE and s['rel_err_charge'] <= TOL_REL_CHARGE
                         and rate != 0.0)
        ok = ok and s['pass']
    return ok


def check_2x2(planning, results):
    _log('A: _run_operational_planning (stubbed) at alpha = 0.5 on the 2 x 2 instance')
    probe, models, run_record = run_init(planning, PREMIUM_RUN)
    dso_models = models['dso']
    init = probe.dso_state
    n_blocks = sum(len(v) for v in init.values())
    a_ok = (run_record['initialization_failed'] is True and run_record['parallel_execution'] is False
            and n_blocks == sum(len(dn.years) * len(dn.days) for dn in planning.distribution_networks.values()))
    steps_init = []
    per_block_a = {}
    for node_id, blocks in init.items():
        for key, st in blocks.items():
            ok = (st['row18_wired'] and all(st['row18_components_present'].values())
                  and st['row18_alpha'] == 0.0 and st['row18_alpha_admm'] == ALPHA_RUN
                  and st['interface_settlement_weight'] == 0.0
                  and st['premium_in_force_at_the_solve']['alpha'] == ALPHA_RUN)
            if 'unit_steps' in st:
                ok = _steps_ok(st['unit_steps'], expect_zero=True) and ok
                steps_init.extend(st['unit_steps'])
            per_block_a[f'DSO:{node_id}:{key}'] = {**_strip(st), 'pass': ok}
            a_ok = a_ok and ok
    tso_ok = all(not v['row18_components_present'] and not v['row18_alpha_admm_present']
                 for v in probe.tso_state.values())
    results['A_init_alpha_zero_through_production'] = {
        'call': 'shared_resources_planning._run_operational_planning(planning, get_initial_candidate_solution())',
        'run_alpha': ALPHA_RUN, 'run_record': run_record,
        'n_dso_blocks_at_init_solve': n_blocks, 'tso_blocks_carry_no_row18': tso_ok,
        'per_block': per_block_a, 'n_unit_steps': len(steps_init),
        'pass': a_ok and tso_ok and len(steps_init) > 0,
    }

    _log('B: _prepare_distribution_objectives_for_admm on the returned models')
    post = post_prepare_states(planning, dso_models)
    b_ok = True
    c_ok = True
    per_block_b, per_block_c = {}, {}
    n_steps_b = 0
    for node_id, blocks in post.items():
        for key, st in blocks.items():
            st0 = init[node_id][key]
            ok = (st['row18_alpha'] == ALPHA_RUN and st['row18_alpha_admm'] == ALPHA_RUN
                  and st['interface_settlement_weight'] == 1.0
                  and st['row18_premium'] == st0['row18_premium']
                  and st['row18_premium'] == st['expected_market_price']
                  and st['row18_charge_expression_id'] == st0['row18_charge_expression_id']
                  and st['row18_charge_expression_str_sha256'] == st0['row18_charge_expression_str_sha256'])
            if 'unit_steps' in st:
                ok = _steps_ok(st['unit_steps'], expect_zero=False) and ok
                n_steps_b += len(st['unit_steps'])
            per_block_b[f'DSO:{node_id}:{key}'] = {**_strip(st), 'pass': ok}
            b_ok = b_ok and ok
            same = (st['component_names'] == st0['component_names'] and st['counts'] == st0['counts'])
            per_block_c[f'DSO:{node_id}:{key}'] = {'component_names_identical': st['component_names'] == st0['component_names'],
                                                   'counts_init': st0['counts'], 'counts_after_prepare': st['counts'],
                                                   'pass': same}
            c_ok = c_ok and same

    _log('B: update_distribution_models_to_admm (placeholder scale) -> ADMM objective references row18_alpha')
    admm_refs = {}
    try:
        srp.update_distribution_models_to_admm(planning, dso_models, planning.params.admm,
                                               PLACEHOLDER_OBJECTIVE_SCALE)
        for node_id, dn in planning.distribution_networks.items():
            for y in dn.years:
                for d in dn.days:
                    blk = dso_models[node_id][y][d]
                    params_in = list(identify_mutable_parameters(blk.admm_objective.expr))
                    admm_refs[f'DSO:{node_id}:{y}:{d}'] = {
                        'admm_objective_active': bool(blk.admm_objective.active),
                        'original_objective_deactivated': not blk.objective.active,
                        'references_row18_alpha_by_identity': any(q is blk.row18_alpha for q in params_in),
                        'row18_alpha_value': float(pe.value(blk.row18_alpha))}
        admm_error = None
    except Exception as error:  # noqa: BLE001
        admm_error = f'{type(error).__name__}: {error}'
    admm_ok = admm_error is None and bool(admm_refs) and all(
        v['admm_objective_active'] and v['original_objective_deactivated']
        and v['references_row18_alpha_by_identity'] and v['row18_alpha_value'] == ALPHA_RUN
        for v in admm_refs.values())

    writers = {}
    for rel in _git(['ls-files', '--', '*.py']).splitlines():
        if '/' in rel:
            continue
        with open(os.path.join(REPO, rel)) as handle:
            text = handle.read()
        if 'row18_alpha.set_value(' not in text:
            continue
        found = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and 'row18_alpha.set_value(' in (ast.get_source_segment(text, node) or ''):
                nested = [n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n is not node
                          and 'row18_alpha.set_value(' in (ast.get_source_segment(text, n) or '')]
                if not nested:
                    found.append(node.name)
        writers[rel] = sorted(found)
    writers_ok = writers == {'shared_resources_planning.py': sorted(
        ['_set_row18_inactive_for_initialisation', '_activate_row18_with_settlement'])}

    results['B_activation_with_settlement_weight'] = {
        'call': 'shared_resources_planning._prepare_distribution_objectives_for_admm(distribution_networks, models)',
        'mechanism': 'Param update (row18_alpha.set_value(row18_alpha_admm)); no rebuild',
        'tolerances_declared': {'charge_rel': TOL_REL_CHARGE, 'objective_rel': TOL_REL_OBJECTIVE},
        'per_block': per_block_b, 'n_unit_steps': n_steps_b,
        'admm_objective': {'placeholder_objective_scale': PLACEHOLDER_OBJECTIVE_SCALE, 'error': admm_error,
                           'per_block': admm_refs, 'pass': admm_ok},
        'row18_alpha_writers_in_tracked_root_py': writers, 'only_the_two_new_functions_write_row18_alpha': writers_ok,
        'pass': b_ok and admm_ok and writers_ok and n_steps_b > 0,
    }
    results['C_structure_unchanged_across_phases'] = {'per_block': per_block_c, 'pass': c_ok}
    return init


def check_2x2_vs_alpha0(planning0, init_run, results):
    _log('C: the same path at alpha = 0 (row 18 not wired: the fix is a no-op there)')
    probe0, models0, run0 = run_init(planning0, PREMIUM_ZERO)
    post0 = post_prepare_states(planning0, models0['dso'], probes=False)
    per_block = {}
    ok_all = True
    for node_id, blocks in init_run.items():
        for key, st in blocks.items():
            z_init = probe0.dso_state[node_id][key]
            z_post = post0[node_id][key]
            n_s, n_t = st['n_scenarios'], st['n_periods']
            extra = sorted(set(st['component_names']) - set(z_init['component_names']))
            missing = sorted(set(z_init['component_names']) - set(st['component_names']))
            dv = st['counts']['n_var_data'] - z_init['counts']['n_var_data']
            dr = st['counts']['n_constraint_data_active'] - z_init['counts']['n_constraint_data_active']
            ok = (not z_init['row18_wired'] and z_init['row18_alpha'] is None and z_init['row18_alpha_admm'] is None
                  and not z_post['row18_wired'] and z_post['row18_alpha_admm'] is None
                  and z_init['component_names'] == z_post['component_names']
                  and z_init['voltage_pin_wired'] and st['voltage_pin_wired']
                  and extra == sorted(ROW18_COMPONENTS + (NEW_COMPONENT,)) and not missing
                  and dv == 4 * n_s * n_t and dr == 2 * n_s * n_t)
            per_block[f'DSO:{node_id}:{key}'] = {
                'n_scenarios': n_s, 'n_periods': n_t, 'components_extra_at_alpha_run': extra,
                'components_missing_at_alpha_run': missing, 'var_delta': dv, 'active_row_delta': dr,
                'expected_var_delta': 4 * n_s * n_t, 'expected_row_delta': 2 * n_s * n_t,
                'alpha0_row18_wired_init': z_init['row18_wired'], 'alpha0_row18_wired_after_prepare': z_post['row18_wired'],
                'pass': ok}
            ok_all = ok_all and ok
    results['C2_vs_alpha0_same_path'] = {'run_record_alpha0': run0, 'per_block': per_block,
                                         'pass': ok_all and run0['initialization_failed'] is True}


def check_parallel_builder(planning, results):
    _log('A2: create_distribution_network_model (the parallel worker function) in-process')
    node_id = next(iter(planning.distribution_networks))
    dn = planning.distribution_networks[node_id]
    candidate = planning.get_initial_candidate_solution()
    planning.params.admm.interface_deviation_premium = dict(PREMIUM_RUN)
    with InitProbe(planning) as probe:
        returned_node, _res, model = srp.create_distribution_network_model(
            node_id, dn, candidate['total_capacity'], ALPHA_RUN, None)
    per_block = {}
    ok = returned_node == node_id and node_id in probe.dso_state
    for key, st in probe.dso_state.get(node_id, {}).items():
        b = (st['row18_wired'] and st['row18_alpha'] == 0.0 and st['row18_alpha_admm'] == ALPHA_RUN
             and st['interface_settlement_weight'] == 0.0)
        if 'unit_steps' in st:
            b = _steps_ok(st['unit_steps'], expect_zero=True) and b
        per_block[f'DSO:{node_id}:{key}'] = {**_strip(st), 'pass': b}
        ok = ok and b
    y, d = next(iter(dn.years)), next(iter(dn.days))
    srp._prepare_distribution_objectives_for_admm({node_id: dn}, {node_id: model})
    after = float(pe.value(model[y][d].row18_alpha))
    results['A2_parallel_builder_in_process'] = {
        'call': f'create_distribution_network_model({node_id}, ..., premium_alpha={ALPHA_RUN}, premium_floor=None)',
        'per_block': per_block, 'row18_alpha_after_prepare_first_block': after,
        'stub_calls': probe.counts(), 'pass': ok and after == ALPHA_RUN}


# ======================================================================================================================
#  D -- 1 x 1
# ======================================================================================================================
def check_1x1(planning_a, planning_0, results):
    _log('D: the SRP1 case file (1 x 1) at alpha = 0.5 and at alpha = 0')
    per = {}
    ok = True
    runs = {}
    for label, planning, premium in (('alpha_0.5', planning_a, PREMIUM_RUN), ('alpha_0', planning_0, PREMIUM_ZERO)):
        probe, models, rec = run_init(planning, premium)
        post = post_prepare_states(planning, models['dso'], probes=False)
        runs[label] = (probe.dso_state, post, rec)
    init_a, post_a, rec_a = runs['alpha_0.5']
    init_0, post_0, rec_0 = runs['alpha_0']
    for node_id, blocks in init_a.items():
        for key, st in blocks.items():
            checks = {
                'n_scenarios_is_1': st['n_scenarios'] == 1,
                'nothing_wired_init_alpha05': (not any(st['row18_components_present'].values())
                                               and not st['voltage_pin_wired']),
                'nothing_wired_after_prepare_alpha05': (not any(post_a[node_id][key]['row18_components_present'].values())
                                                        and not post_a[node_id][key]['voltage_pin_wired']),
                'names_equal_alpha05_vs_alpha0_init': st['component_names'] == init_0[node_id][key]['component_names'],
                'names_equal_alpha05_vs_alpha0_after_prepare': (post_a[node_id][key]['component_names']
                                                                == post_0[node_id][key]['component_names']),
                'counts_equal_alpha05_vs_alpha0_init': st['counts'] == init_0[node_id][key]['counts'],
                'counts_equal_alpha05_vs_alpha0_after_prepare': (post_a[node_id][key]['counts']
                                                                 == post_0[node_id][key]['counts']),
                'names_equal_across_phases': st['component_names'] == post_a[node_id][key]['component_names'],
            }
            per[f'DSO:{node_id}:{key}'] = {**checks, 'counts': st['counts'], 'pass': all(checks.values())}
            ok = ok and all(checks.values())
    results['D_nothing_wired_at_1x1'] = {'run_record_alpha05': rec_a, 'run_record_alpha0': rec_0,
                                         'n_blocks': len(per), 'per_block': per,
                                         'pass': ok and len(per) > 0 and rec_a['initialization_failed'] is True}


# ======================================================================================================================
#  E -- the uncoordinated benchmark
# ======================================================================================================================
def _function_source_at(commit, path, name):
    text = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    return None


def check_benchmark(planning, results):
    import inspect
    _log('E: _run_operational_planning_without_coordination (stubbed) with alpha = 0.5 in force')
    name = '_run_operational_planning_without_coordination'
    live = inspect.getsource(getattr(srp, name)).rstrip('\n')
    parent = (_function_source_at(PARENT_COMMIT, 'shared_resources_planning.py', name) or '').rstrip('\n')
    planning.params.admm.interface_deviation_premium = dict(PREMIUM_RUN)
    planning.parallel_execution = False
    with InitProbe(planning, probes=False) as probe:
        error = None
        try:
            srp._run_operational_planning_without_coordination(planning)
        except Exception as exc:  # noqa: BLE001
            error = f'{type(exc).__name__}: {exc}'
    per = {}
    ok = True
    for node_id, blocks in probe.dso_state.items():
        for key, st in blocks.items():
            b = (not st['row18_wired'] and st['row18_alpha'] is None and st['row18_alpha_admm'] is None
                 and st['voltage_pin_wired'])
            per[f'DSO:{node_id}:{key}'] = {'row18_wired': st['row18_wired'], 'row18_alpha': st['row18_alpha'],
                                           'row18_alpha_admm': st['row18_alpha_admm'],
                                           'voltage_pin_wired': st['voltage_pin_wired'],
                                           'n_scenarios': st['n_scenarios'], 'pass': b}
            ok = ok and b
    n_expected = sum(len(dn.years) * len(dn.days) for dn in planning.distribution_networks.values())
    results['E_uncoordinated_benchmark_unchanged'] = {
        'source_identical_to_parent_commit': live == parent, 'parent_commit': PARENT_COMMIT,
        'source_sha256_live': hashlib.sha256(live.encode()).hexdigest(),
        'source_sha256_parent': hashlib.sha256(parent.encode()).hexdigest(),
        'source_reads_interface_deviation_premium': 'interface_deviation_premium' in live,
        'source_passes_premium_alpha_zero': 'premium_alpha=0.0' in live,
        'run_error': error, 'stub_calls': probe.counts(), 'n_dso_blocks_seen': len(per),
        'per_block': per,
        'pass': (live == parent and 'interface_deviation_premium' not in live and 'premium_alpha=0.0' in live
                 and error is None and ok and len(per) == n_expected),
    }


# ======================================================================================================================
#  F -- fixtures
# ======================================================================================================================
def check_fixtures(results):
    _log('F: preserved fixtures unpickle')
    W39C.check_fixtures(results)          # tracked set, by import -> results['F_fixtures_unpickle']
    tracked = results.pop('F_fixtures_unpickle')
    common = _git(['rev-parse', '--path-format=absolute', '--git-common-dir'])
    main_root = os.path.dirname(common)
    anchors = [os.path.join(main_root, 'data', 'SRP1', 'Results', 'P512R', 'cycle21_pre_setup', 'snapshot.pkl')]
    anchors += sorted(glob.glob(os.path.join(main_root, 'data', 'SRP1', 'Results', 'FrozenSMOPF', '**', '*.pkl'),
                                recursive=True))
    loaded, failed, absent = [], [], []
    for path in anchors:
        if not os.path.isfile(path):
            absent.append(path)
            continue
        try:
            with open(path, 'rb') as handle:
                payload = pickle.load(handle)
            loaded.append({'path': path, 'sha256': _sha256_file(path), 'type': type(payload).__name__})
            del payload
        except Exception as error:  # noqa: BLE001
            failed.append({'path': path, 'sha256': _sha256_file(path), 'error': f'{type(error).__name__}: {error}'})
    results['F_fixtures_unpickle'] = {
        'tracked': tracked,
        'untracked_anchors_main_checkout_read_only': {
            'main_checkout_root': main_root,
            'modules_resolved_from': REPO,
            'n_listed': len(anchors), 'n_loaded': len(loaded), 'n_failed': len(failed), 'absent': absent,
            'loaded': loaded, 'failed': failed},
        'pass': tracked['pass'] and not failed and not absent and len(loaded) >= 2,
    }


# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--scratch', required=True)
    args = ap.parse_args()
    scratch = os.path.abspath(args.scratch)
    if scratch.startswith(REPO + os.sep):
        raise SystemExit('--scratch must be outside the repository')
    out_dir = os.path.join(REPO, OUT_ROOT_REL, args.label)
    out_path = os.path.join(out_dir, 'init_fix_zero_solve_checks.json')
    if os.path.exists(out_dir):
        print(f'REFUSED: output directory exists (write-once): {out_dir}', file=sys.stderr)
        return 1
    os.makedirs(out_dir)
    case_dir = os.path.join(out_dir, 'cases')
    started = time.time()
    results, instances = {}, {}
    _log(STAGE)

    p2a, instances['2x2_alpha05'] = read_planning('2x2_alpha05', OVERRIDES_2X2, case_dir, scratch)
    init_run = check_2x2(p2a, results)
    del p2a
    p20, instances['2x2_alpha0'] = read_planning('2x2_alpha0', OVERRIDES_2X2, case_dir, scratch)
    check_2x2_vs_alpha0(p20, init_run, results)
    del p20
    p2p, instances['2x2_parallel_builder'] = read_planning('2x2_parallel_builder', OVERRIDES_2X2, case_dir, scratch)
    check_parallel_builder(p2p, results)
    del p2p
    p2b, instances['2x2_benchmark'] = read_planning('2x2_benchmark', OVERRIDES_2X2, case_dir, scratch)
    check_benchmark(p2b, results)
    del p2b
    p1a, instances['1x1_alpha05'] = read_planning('1x1_alpha05', OVERRIDES_1X1, case_dir, scratch)
    p10, instances['1x1_alpha0'] = read_planning('1x1_alpha0', OVERRIDES_1X1, case_dir, scratch)
    check_1x1(p1a, p10, results)
    del p1a, p10
    check_fixtures(results)

    verify_mine = GUARD.verify(expected_solves=0)
    verify_w39 = W39C.GUARD.verify(expected_solves=0)
    all_pass = all(v.get('pass') for v in results.values()) and not verify_mine and not verify_w39
    payload = {
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY,
        'frozen_spec': {'path': SPEC_PATH, 'sha256_prefix_declared': SPEC_SHA256,
                        'sha256_observed': _sha256_file(os.path.join(REPO, SPEC_PATH))},
        'timestamp_utc': _utc(), 'interpreter': sys.executable, 'argv': sys.argv,
        'script': os.path.basename(__file__), 'script_sha256': _sha256_file(os.path.abspath(__file__)),
        'production_sha256': {n: _sha256_file(os.path.join(REPO, n)) for n in (
            'shared_resources_planning.py', 'model_construction_helpers.py', 'admm_parameters.py')},
        'imported_sha256': {n: _sha256_file(os.path.join(REPO, n)) for n in (
            'p515_s51_row18_zero_solve_checks.py', 'p515_s44_scale_measurement.py', 'p513_solve_profile_guard.py')},
        'git_head': _git(['rev-parse', 'HEAD']),
        'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
        'instances': instances, 'run_alpha': ALPHA_RUN, 'scratch': scratch,
        'solve_profile_guard': {
            'this_script': {'permitted': [], 'counts': dict(GUARD.counts), 'verify_failures': verify_mine},
            'w39_checks_module_guard': {'permitted': [], 'counts': dict(W39C.GUARD.counts),
                                        'verify_failures': verify_w39}},
        'checks': results,
        'failing_checks': [k for k, v in results.items() if not v.get('pass')],
        'all_checks_pass': bool(all_pass), 'wall_s': time.time() - started,
    }
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_dir):
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = _sha256_file(fpath)
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    for name, value in results.items():
        _log(f'  {name}: {"PASS" if value.get("pass") else "FAIL"}')
    _log(f'guards: mine {dict(GUARD.counts)} {verify_mine}; w39 {dict(W39C.GUARD.counts)} {verify_w39}')
    _log(f'all_checks_pass = {all_pass}; wrote {os.path.relpath(out_path, REPO)}')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
