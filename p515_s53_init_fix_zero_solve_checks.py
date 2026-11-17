"""
P5.15 Addendum 40 ruling 2 (tasks W51 -> W54 -> W56 -> W57) -- ZERO-SOLVE checks for "row 18 INACTIVE at the
initialisation solve, activated with the settlement weight", in its STRUCTURAL form.

W57 UPDATE (schema v3; frozen spec v3 `frozen_s53_row18_structural_spec_v3_1064db50.json`, key `zero_solve_checks_r6`,
predecessor v2 5123e67b; Planner rulings on W56's Q1/Q2):
  * B and A2 run on an EMULATED LOADED SOLUTION, NOT A SOLVE. At r5 every stubbed solve loaded nothing, so
    e = value(flow) - value(expectation) was 0.0 at all 2304 indices and the minimal-split assertion tested nothing.
    Now, before `_prepare_distribution_objectives_for_admm`, `emulate_loaded_solution` sets per block:
    expected_interface_pf_{p,q}[t] to distinct non-zero values, and the reference generator's pg/qg[ref, s, t] so that
    e hits a declared target per index -- a distinct magnitude per (block, family, index), both signs (negative iff
    (k + f + b) % 3 == 0, i.e. 32 of 96 per family per block). The emulated point is NOT feasible and nothing is
    solved; it exists only to make the split non-trivial.
  * The defining-row BODY, evaluated by Pyomo from the model itself (not from the recomputed e), must be 0 within
    TOL_BODY_ABS. That is the check that cross-validates the family table against the actual rows (the per-index
    split assertion uses this script's own table, which equals production's by the checklist, so on its own it
    cannot detect a table that is wrong in both places).
  * NON-VACUITY (fails the check if not met): in every (block, family) >= 25 % of indices e > 0, >= 25 % e < 0, none
    e == 0; all |e| of a block (both families) pairwise distinct with gap >= MIN_MAG_GAP; all |e| of the run
    distinct; the emulation hit every target within TOL_EMULATION_ABS.
  * Production (W57 item 2) now evaluates each defining row's body after activation and raises above
    `_ROW18_ACTIVATION_BODY_TOL`. A2 ends with a NEGATIVE CONTROL: on one block, reset to the initialisation state by
    production's own `_set_row18_inactive_for_initialisation`, `_ROW18_DEVIATION_FAMILIES` swapped between families
    for one call -> production MUST raise that RuntimeError (restored in `finally`; A2's recorded assertions precede it).
  * C2: n_expression_data +1 in BOTH phases (`row18_deviation_charge` is a scalar Expression) -- v2 predicted +0;
    that was a spec error (r5 C2 FAIL stays on the record).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 2; frozen spec v23
`data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json`, key `ruling2_init_fix` (the ruling), as amended in
its MECHANISM by `data/SRP1/Results/P515S53/row18_structural/frozen_s53_row18_structural_spec_v2_5123e67b.json`
(v2; predecessor v1 05934ab0), key `zero_solve_checks` -- the assertions and structure predictions below were
recorded there before this script's first run.

THE CHANGE UNDER TEST (shared_resources_planning.py, W54 d573e145; W51's Param-only form is gone):
  * `_set_row18_inactive_for_initialisation(block)` -- called by both DSO initialisation builders
    (`create_distribution_networks_models_sequential`, `create_distribution_network_model`) right after
    `add_scenario_commitment_terms` and before the initialisation solve: per index, DEACTIVATE
    `row18_dev_{p,q}_def`, then FIX `row18_dev_{p,q}_{up,down}` at 0. `row18_alpha` keeps the run's alpha;
    `row18_alpha_admm` no longer exists;
  * `_activate_row18_with_settlement(block)` -- called in `_prepare_distribution_objectives_for_admm` right
    after `interface_settlement_weight` is set to 1: per index, e = value(flow) - value(expectation), minimal split
    up = max(e, 0), down = max(-e, 0), UNFIX the pair, then ACTIVATE the row.

Nothing here solves. `SolveProfileGuard(permitted=())` is installed before any production import and verified
at exactly 0 (the imported W39 checks module installs its own zero guard too; both are verified). Every
`.optimize` of the planning instance under test is replaced ON THE INSTANCE by a recording stub that, at the
moment the solve would happen, records the state IPOPT would have been handed, and returns "no result" (None per
block), which production reads as a failed local solve.

PER-INDEX STRUCTURAL ASSERTIONS (`row18_structural`, every index of every wired DSO block, both phases):
  initialisation   row inactive; pair FIXED at exactly 0.0; no index with a fixed pair and an active row;
                   value(row18_deviation_charge) == 0.0 exactly.
  after activation row active; pair unfixed; up == max(e, 0), down == max(-e, 0) exactly, with e recomputed HERE
                   from the block's own flow (pg_adn / qg_adn) and expectation (expected_interface_pf_p / q) values
                   using this script's own family table (compared with production's `_ROW18_DEVIATION_FAMILIES`);
                   defining-row residual value(body) - upper == 0.0 exactly; no index with a fixed pair and an
                   active row.
  NON-VACUITY: the minimal-split assertion counts as tested only if some index has e > 0 and some has e < 0 across
  the checked blocks (spec v2); the sign counts are recorded and B fails if either is zero.
  W51's unit steps on the deviation Vars at INITIALISATION (checks A, A2) are REMOVED: the Vars are fixed there, so
  a unit step is not an admissible move and tested nothing. B's unit steps AFTER activation are kept (free Vars,
  alpha = 0.5: they test the charge rate).

CHECKS (each records its own numbers; `pass` per check; `all_checks_pass` overall)
 A  INITIALISATION THROUGH PRODUCTION'S OWN PATH (2 x 2, run alpha = 0.5): `_run_operational_planning` itself is
    called (fresh initialisation, `parallel_execution` False, asserted) with
    `ADMMParameters.interface_deviation_premium = {alpha 0.5, floor None}` in force. At every DSO block's would-be
    initialisation solve: row 18 WIRED (all its components present), `row18_alpha` == 0.5, `row18_alpha_admm`
    absent, `interface_settlement_weight` == 0.0, and the per-index initialisation assertions. The init returns
    `initialization_failed` (every solve was stubbed), which is what makes its models available here.
 A2 THE PARALLEL BUILDER (`create_distribution_network_model`, the ProcessPool worker function) called in-process on
    one node with alpha = 0.5: the per-index initialisation assertions at its would-be solve, then
    `_prepare_distribution_objectives_for_admm` on that node and the per-index activation assertions.
 B  ACTIVATION WITH THE SETTLEMENT WEIGHT: production's `_prepare_distribution_objectives_for_admm` on the models A
    returned: on EVERY DSO block `row18_alpha` == 0.5 exactly, `row18_alpha_admm` absent, the settlement weight
    == 1.0, the per-index activation assertions (+ non-vacuity); `row18_premium[t]` unchanged from the init phase and
    equal to `expected_market_price(network, t)` (no floor in force); `row18_deviation_charge` the SAME component
    object with an identical expression string; unit steps (first block) change the objective rule and the charge by
    exactly omega_s * 0.5 * pibar_t * baseMVA. Then `update_distribution_models_to_admm` (placeholder objective
    scale, recorded -- nothing is solved) builds the ADMM objective, which references `row18_alpha` BY IDENTITY; and
    an AST scan of every tracked root-level .py finds NO call `<expr>.row18_alpha.set_value(...)` (nothing writes
    row18_alpha any more).
 C  PHASE STRUCTURE, SAME BLOCK: component-name sets, n_var_data, n_constraint_data, n_expression_data identical
    between the init solve and after `_prepare_*`; n_constraint_data_active +2nT and n_var_data_unfixed +4nT.
 C2 vs alpha = 0 ON THE SAME PATH (row 18 not wired -- the fix is a no-op there): extra components exactly the nine
    row 18 components (no `row18_alpha_admm`), none missing; at init n_var_data +4nT, n_constraint_data +2nT,
    n_constraint_data_active +0, n_var_data_unfixed +0; after activation n_constraint_data_active +2nT,
    n_var_data_unfixed +4nT.
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

CAPTURE-PATH CHECKLIST (asserted before anything is built; refuses on failure): the v2 spec's sha256 matches the pin;
production's two functions and `_ROW18_DEVIATION_FAMILIES` exist, the family table equals this script's own, and
neither function writes `row18_alpha` or names `row18_alpha_admm`.

INSTANCES: 2 x 2 = `p515_s44_scale_measurement.derive_case('srp1', {years {'2025': 5}, 2 market, 2 operation})`,
the W39 zero-solve checks' 2 x 2 (one year: 4 days x 3 DSOs + TSO), chosen over the 5-year pilot instance to keep
memory low while a campaign runs in the main checkout; 1 x 1 = `derive_case('srp1', {})`, the SRP1 case file
unchanged. Derived case files are written to <out>/cases/ and hash-recorded; the reader's Diagrams/Results dirs
go to --scratch.

EXACT COMMAND (worktree/repo root, canonical interpreter, attached, BOTH streams captured; label r6 -- r4 is W51's
and r5 W56's committed evidence, r1-r3 are W51's abandoned labels; none is reused):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt LP_SOLVER_PATH=<from the main checkout .env> \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_init_fix_zero_solve_checks.py \\
      --label r6 --scratch <dir outside the repo> \\
      > data/SRP1/Results/P515S53/init_fix_zero_solve_checks_r6_launch.log 2>&1
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W57 init-fix zero-solve checks (never solves)').install()

import pyomo.environ as pe  # noqa: E402
from pyomo.core.expr.visitor import identify_mutable_parameters  # noqa: E402
import model_construction_helpers as MCH  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import p515_s51_row18_zero_solve_checks as W39C  # noqa: E402 -- installs its own zero guard (verified below)
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402
from network import Network  # noqa: E402

STAGE = ('P5.15 Addendum 40 ruling 2 (W57) -- row 18 STRUCTURALLY inactive at the initialisation solve, activated '
         'with the settlement weight: zero-solve checks on an EMULATED loaded solution (not a solve)')
SCHEMA = 'p515_s53_init_fix_zero_solve_checks_v3'
AUTHORITY = ['Planner task W57 (2026-09-24): rulings on W56 Q1 (re-freeze v3, C2 +1) and Q2 (option (a): emulate a '
             'loaded solution)',
             'Planner task W56 (2026-09-24)',
             'PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 2',
             'data/SRP1/Results/P515S53/frozen_s53_spec_v23_39a07fd8.json (ruling2_init_fix)',
             'data/SRP1/Results/P515S53/row18_structural/frozen_s53_row18_structural_spec_v2_5123e67b.json '
             '(zero_solve_checks)',
             'data/SRP1/Results/P515S53/row18_structural/frozen_s53_row18_structural_spec_v3_1064db50.json '
             '(zero_solve_checks_r6)']
SPEC_PATH = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'frozen_s53_spec_v23_39a07fd8.json')
SPEC_SHA256 = '39a07fd8'   # prefix, as named in the file name; the full hash is recorded on output
SPEC_V2_PATH = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'row18_structural',
                            'frozen_s53_row18_structural_spec_v2_5123e67b.json')
SPEC_V2_SHA256 = '5123e67bd9643ff2037977def18ce3e12aaa1cd7de6d7ead4a2bae4f9ba7675f'
SPEC_V3_PATH = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'row18_structural',
                            'frozen_s53_row18_structural_spec_v3_1064db50.json')
SPEC_V3_SHA256 = '1064db50f8d339ac8bbc9ffdcdb2485e9e06b20707dbb73748dd840fe3878ff7'
SPEC_V3_KEY = 'zero_solve_checks_r6'
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
NEW_COMPONENT = 'row18_alpha_admm'   # W51's recorded-alpha Param: RETIRED by W54, asserted ABSENT everywhere
DEV_FAMILIES = ('row18_dev_p_up', 'row18_dev_p_down', 'row18_dev_q_up', 'row18_dev_q_down')
# This script's OWN table of row 18's defining identity  flow[s,t] - expectation[t] == up[s,t] - down[s,t]
# (from model_construction_helpers.row18_interface_deviation_{p,q}_rule), used to recompute e independently;
# compared with production's `_ROW18_DEVIATION_FAMILIES` in the capture-path checklist.
FAMILIES = (
    ('row18_dev_p_def', 'row18_dev_p_up', 'row18_dev_p_down', 'pg_adn', 'expected_interface_pf_p'),
    ('row18_dev_q_def', 'row18_dev_q_up', 'row18_dev_q_down', 'qg_adn', 'expected_interface_pf_q'),
)
# Declared before the run (B's post-activation unit steps). The charge is a short linear sum: exact to 1e-12. The
# objective is a large-magnitude sum, so a unit-step difference carries cancellation error of order
# eps * |objective|: 1e-9 relative.
TOL_REL_CHARGE = 1e-12
TOL_REL_OBJECTIVE = 1e-9
# W57, declared in spec v3 before the r6 run.
# The EMULATED loaded solution (NOT a solve): block ordinal b (the block's position in the node/year/day order of the
# run), family f (0 = P, 1 = Q), index k (the defining row's own iteration order, 0 .. N-1, N = n T):
#   target e = sign * (EMU_MAG_BASE + EMU_MAG_STEP * ((2 b + f) N + k)),  sign = -1 iff (k + f + b) % EMU_NEG_EVERY == 0
#   expected_interface_pf_{p,q}[t] = a_f + c_f * t_ordinal + d_f * b          (EMU_EXPECTATION[f] = (a_f, c_f, d_f))
# and the reference generator's pg (P) / qg (Q) [ref, s_m, s_o, t] is shifted so that value(flow) - value(expectation)
# hits the target.
EMU_MAG_BASE = 0.01       # per unit
EMU_MAG_STEP = 1e-4       # per unit, per global ordinal: every |e| of a run distinct by construction
EMU_NEG_EVERY = 3
EMU_EXPECTATION = ((0.20, 0.003, 0.0005), (-0.07, -0.002, -0.0003))
TOL_EMULATION_ABS = 1e-12  # |achieved e - target| per index
TOL_BODY_ABS = 1e-12       # |value(body) - value(upper)| per defining row after activation (Pyomo, from the model)
MIN_SIGN_SHARE = 0.25      # per (block, family): share of e > 0 and share of e < 0, each at least this
MIN_MAG_GAP = 5e-5         # per unit: minimum gap between sorted distinct |e| (block, both families; and whole run)
PRODUCTION_BODY_TOL_DECLARED = 1e-9   # production's `_ROW18_ACTIVATION_BODY_TOL` (W57 item 2), checked
SOLUTION_SOURCE = ('EMULATED loaded solution, NOT a solve: expected_interface_pf_{p,q} and the reference generator '
                   'pg/qg set by emulate_loaded_solution (declared formula, spec v3); the point is not feasible and '
                   'nothing is solved')
NAMED_CTYPES = (pe.Var, pe.Param, pe.Constraint, pe.Expression, pe.Objective, pe.Block)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W57-checks] {msg}', flush=True)


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


def emulate_loaded_solution(block, network, ordinal):
    """EMULATED loaded solution -- NOT a solve (W57, Planner ruling on W56 Q2, option (a)). Sets the values a loaded
    initialisation solution would have set, so that e = value(flow) - value(expectation) is non-zero with both signs
    and a distinct magnitude per (block, family, index); formula and constants declared in spec v3 (see the EMU_*
    constants). The flow is moved through the reference generator's pg (P) / qg (Q) Var, which enters pg_adn / qg_adn
    with coefficient +1 (model_construction_helpers.interface_pf_{p,q}_distribution_def); the achieved e is read back
    from the model per index and compared with its target, so a wrong Var or coefficient fails here. Bounds are NOT
    respected and nothing is feasible: the point only has to make the minimal split non-trivial."""
    if not hasattr(block, 'row18_alpha'):
        return {'wired': False}
    ref_gen = network.get_reference_gen_idx()
    t_ord = {p: i for i, p in enumerate(block.periods)}
    gen_vars = (block.pg, block.qg)       # this emulation's own mapping; verified through the achieved e below
    out = {'wired': True, 'ordinal': ordinal, 'reference_gen_idx': ref_gen, 'families': {}}
    for f, (row_name, _up, _down, flow_name, expectation_name) in enumerate(FAMILIES):
        row, flow, expectation = getattr(block, row_name), getattr(block, flow_name), getattr(block, expectation_name)
        a, c, d = EMU_EXPECTATION[f]
        for p in block.periods:
            expectation[p].set_value(a + c * t_ord[p] + d * ordinal)
        n_idx = len(row)
        max_err, n_pos, n_neg, mags = 0.0, 0, 0, []
        for k, index in enumerate(row):
            s_m, s_o, p = index
            sign = -1.0 if (k + f + ordinal) % EMU_NEG_EVERY == 0 else 1.0
            target = sign * (EMU_MAG_BASE + EMU_MAG_STEP * ((2 * ordinal + f) * n_idx + k))
            var = gen_vars[f][ref_gen, s_m, s_o, p]
            shift = (float(pe.value(expectation[p])) + target) - float(pe.value(flow[s_m, s_o, p]))
            var.set_value(float(var.value) + shift)
            achieved = float(pe.value(flow[s_m, s_o, p])) - float(pe.value(expectation[p]))
            max_err = max(max_err, abs(achieved - target))
            n_pos += target > 0.0
            n_neg += target < 0.0
            mags.append(abs(target))
        out['families'][row_name] = {
            'generator_var': gen_vars[f].local_name, 'n_indices': n_idx, 'n_target_pos': n_pos,
            'n_target_neg': n_neg, 'target_abs_min': min(mags), 'target_abs_max': max(mags),
            'expectation_values': [float(pe.value(expectation[p])) for p in block.periods],
            'max_abs_achieved_minus_target': max_err, 'within_tolerance': max_err <= TOL_EMULATION_ABS}
    out['pass'] = all(v['within_tolerance'] for v in out['families'].values())
    return out


def emulate_all(planning, dso_models, node_ids=None):
    """`emulate_loaded_solution` on every wired DSO block, ordinal = position in node/year/day order."""
    records, ordinal = {}, 0
    for node_id, dn in planning.distribution_networks.items():
        if node_ids is not None and node_id not in node_ids:
            continue
        for y in dn.years:
            for d in dn.days:
                records[f'DSO:{node_id}:{y}:{d}'] = emulate_loaded_solution(dso_models[node_id][y][d],
                                                                            dn.network[y][d], ordinal)
                ordinal += 1
    return records


def _min_gap(values):
    s = sorted(values)
    return min((b - a for a, b in zip(s, s[1:])), default=float('inf'))


def non_vacuity(states, emulation):
    """W57 NON-VACUITY (declared in spec v3; the check FAILS if any requirement is not met): per (block, family)
    >= MIN_SIGN_SHARE of indices e > 0 and e < 0 and none e == 0; per block all |e| (both families) pairwise
    distinct with gap >= MIN_MAG_GAP; over the run all |e| distinct with gap >= MIN_MAG_GAP; every emulation target
    hit within TOL_EMULATION_ABS. `states` maps block key -> the active-phase `row18_structural` output (with its
    private `_abs_e_by_family`, popped here)."""
    per_block, all_mags, ok = {}, [], bool(states)
    for key, st in states.items():
        abs_e = st.pop('_abs_e_by_family')
        fam = {}
        for row_name, counts in st['e_sign_counts_by_family'].items():
            n = sum(counts.values())
            share_pos, share_neg = counts['e_pos'] / n, counts['e_neg'] / n
            fam[row_name] = {**counts, 'share_pos': share_pos, 'share_neg': share_neg,
                             'abs_e_min': min(abs_e[row_name]), 'abs_e_max': max(abs_e[row_name]),
                             'pass': share_pos >= MIN_SIGN_SHARE and share_neg >= MIN_SIGN_SHARE
                             and counts['e_zero'] == 0}
        block_mags = [m for v in abs_e.values() for m in v]
        gap = _min_gap(block_mags)
        distinct = len(set(block_mags)) == len(block_mags) and gap >= MIN_MAG_GAP
        emu_ok = bool(emulation.get(key, {}).get('pass'))
        b_ok = all(v['pass'] for v in fam.values()) and distinct and emu_ok
        per_block[key] = {'families': fam, 'n_abs_e': len(block_mags), 'n_distinct_abs_e': len(set(block_mags)),
                          'min_gap_abs_e_both_families': gap, 'magnitudes_distinct': distinct,
                          'emulation_within_tolerance': emu_ok, 'pass': b_ok}
        all_mags += block_mags
        ok = ok and b_ok
    run_gap = _min_gap(all_mags)
    run_distinct = len(set(all_mags)) == len(all_mags) and run_gap >= MIN_MAG_GAP
    return {'requirements': {'min_sign_share_each_sign_per_block_family': MIN_SIGN_SHARE, 'e_zero_allowed': 0,
                             'min_mag_gap': MIN_MAG_GAP, 'tol_emulation_abs': TOL_EMULATION_ABS},
            'per_block': per_block, 'n_abs_e_run': len(all_mags), 'n_distinct_abs_e_run': len(set(all_mags)),
            'min_gap_abs_e_run': run_gap, 'magnitudes_distinct_run': run_distinct,
            'pass': bool(ok and run_distinct)}


def row18_structural(block, phase):
    """PER-INDEX structural assertions on a wired block (spec v2 `zero_solve_checks`). `phase` is 'init' (at the
    would-be initialisation solve) or 'active' (after `_prepare_distribution_objectives_for_admm`). Every index of
    both defining-row families is checked; violations are listed (first 20 per family) and counted. In the active
    phase e = value(flow) - value(expectation) is recomputed HERE from the block's own components, via this script's
    FAMILIES table."""
    assert phase in ('init', 'active')
    n_expected = len(block.scenarios_market) * len(block.scenarios_operation) * len(block.periods)
    families = {}
    fixed_pair_active_row = []
    signs = {'e_pos': 0, 'e_neg': 0, 'e_zero': 0}
    signs_by_family = {}
    abs_e_by_family = {}
    max_abs_e = 0.0
    ok = True
    for row_name, up_name, down_name, flow_name, expectation_name in FAMILIES:
        fam_signs = signs_by_family.setdefault(row_name, {'e_pos': 0, 'e_neg': 0, 'e_zero': 0})
        fam_abs_e = abs_e_by_family.setdefault(row_name, [])
        max_abs_residual, n_residual_exactly_zero = 0.0, 0
        row, up, down = getattr(block, row_name), getattr(block, up_name), getattr(block, down_name)
        flow, expectation = getattr(block, flow_name), getattr(block, expectation_name)
        index_sets_equal = set(row.keys()) == set(up.keys()) == set(down.keys())
        violations = []
        n_idx = 0
        for index in row:
            n_idx += 1
            r, u, d = row[index], up[index], down[index]
            if r.active and (u.fixed or d.fixed):
                fixed_pair_active_row.append(f'{row_name}[{index}]')
            bad = []
            if phase == 'init':
                if r.active:
                    bad.append('row_active')
                if not u.fixed:
                    bad.append('up_not_fixed')
                if not d.fixed:
                    bad.append('down_not_fixed')
                if u.value != 0.0:
                    bad.append(f'up_value={u.value!r}')
                if d.value != 0.0:
                    bad.append(f'down_value={d.value!r}')
            else:
                s_m, s_o, p = index
                e = float(pe.value(flow[s_m, s_o, p])) - float(pe.value(expectation[p]))
                sign_key = 'e_pos' if e > 0.0 else ('e_neg' if e < 0.0 else 'e_zero')
                signs[sign_key] += 1
                fam_signs[sign_key] += 1
                fam_abs_e.append(abs(e))
                max_abs_e = max(max_abs_e, abs(e))
                expected_up, expected_down = max(e, 0.0), max(-e, 0.0)
                # The BODY of the defining row, evaluated by Pyomo from the model itself -- NOT from the e recomputed
                # above. The split assertion below uses this script's FAMILIES table, which equals production's (the
                # checklist), so it cannot detect a table wrong in both places; the body can: it is zero only if the
                # split production wrote satisfies the actual row. This is the check that cross-validates the family
                # table against the rows (W57).
                residual = float(pe.value(r.body)) - float(pe.value(r.upper))
                max_abs_residual = max(max_abs_residual, abs(residual))
                n_residual_exactly_zero += residual == 0.0
                if not r.active:
                    bad.append('row_inactive')
                if u.fixed:
                    bad.append('up_fixed')
                if d.fixed:
                    bad.append('down_fixed')
                if u.value != expected_up:
                    bad.append(f'up={u.value!r}!=max(e,0)={expected_up!r}')
                if d.value != expected_down:
                    bad.append(f'down={d.value!r}!=max(-e,0)={expected_down!r}')
                if not r.equality or float(pe.value(r.lower)) != float(pe.value(r.upper)):
                    bad.append('row_not_an_equality')
                if not abs(residual) <= TOL_BODY_ABS:
                    bad.append(f'body_residual={residual!r}')
            if bad:
                violations.append({'index': str(index), 'failures': bad})
        fam_ok = index_sets_equal and n_idx == n_expected and not violations
        families[row_name] = {'n_indices': n_idx, 'n_expected': n_expected, 'index_sets_equal': index_sets_equal,
                              'n_violations': len(violations), 'violations_first_20': violations[:20],
                              'pass': fam_ok}
        if phase == 'active':
            families[row_name]['body_residual_max_abs'] = max_abs_residual
            families[row_name]['body_residual_n_exactly_zero'] = n_residual_exactly_zero
            families[row_name]['body_residual_tolerance'] = TOL_BODY_ABS
        ok = ok and fam_ok
    out = {'phase': phase, 'families': families,
           'n_indices_fixed_pair_with_active_row': len(fixed_pair_active_row),
           'fixed_pair_with_active_row_first_20': fixed_pair_active_row[:20],
           'charge_value': float(pe.value(block.row18_deviation_charge))}
    ok = ok and not fixed_pair_active_row
    if phase == 'init':
        out['charge_exactly_zero'] = out['charge_value'] == 0.0
        ok = ok and out['charge_exactly_zero']
    else:
        out['e_sign_counts'] = signs
        out['e_sign_counts_by_family'] = signs_by_family
        out['max_abs_e'] = max_abs_e
        out['_abs_e_by_family'] = abs_e_by_family    # private: consumed (popped) by non_vacuity
    out['pass'] = bool(ok)
    return out


def block_state(block, network, params, phase, with_probes=False):
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
        state['structural'] = row18_structural(block, phase)
        if with_probes:
            assert phase == 'active', 'unit steps on FIXED deviation Vars test nothing (W56): active phase only'
            state['unit_steps'] = _probes(block, network, params)
    return state


class InitProbe:
    """Replaces `.optimize` on ONE planning instance's agents (instance attributes; deleted on exit). At each
    would-be solve it records the block state IPOPT would have been handed, then returns "no result"."""

    def __init__(self, planning):
        self.planning = planning
        self.calls = []
        self.dso_state = {}
        self.tso_state = {}

    def _dso_stub(self, node_id, dn):
        def stub(model, *args, **kwargs):
            self.calls.append(f'dso:{node_id}')
            premium = dict(self.planning.params.admm.interface_deviation_premium)
            for y in dn.years:
                for d in dn.days:
                    st = block_state(model[y][d], dn.network[y][d], dn.params, 'init')
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
                                                                      dn.params, 'active', with_probes=first)
    return out


def _strip(state):
    return {k: v for k, v in state.items() if k not in ('component_names',)}


# ======================================================================================================================
#  A / B / C at 2 x 2
# ======================================================================================================================
def _init_block_ok(st):
    """Initialisation phase, one DSO block: wired, the W51 Param absent, alpha = the run's, settlement weight 0, and
    every per-index initialisation assertion."""
    return bool(st['row18_wired'] and all(st['row18_components_present'][n] for n in ROW18_COMPONENTS)
                and not st['row18_components_present'][NEW_COMPONENT] and st['row18_alpha_admm'] is None
                and st['row18_alpha'] == ALPHA_RUN and st['interface_settlement_weight'] == 0.0
                and st['structural']['phase'] == 'init' and st['structural']['pass'])


def _active_block_ok(st):
    """After activation, one DSO block: alpha = the run's, the W51 Param absent, settlement weight 1, and every
    per-index activation assertion."""
    return bool(st['row18_wired'] and not st['row18_components_present'][NEW_COMPONENT]
                and st['row18_alpha_admm'] is None and st['row18_alpha'] == ALPHA_RUN
                and st['interface_settlement_weight'] == 1.0
                and st['structural']['phase'] == 'active' and st['structural']['pass'])


def _row18_alpha_writers():
    """AST scan of every tracked root-level .py for a CALL `<expr>.row18_alpha.set_value(...)` (a call node, so a
    string literal naming it -- e.g. in this scan's own docs -- is not a writer)."""
    calls = []
    files = [rel for rel in _git(['ls-files', '--', '*.py']).splitlines() if '/' not in rel]
    for rel in files:
        with open(os.path.join(REPO, rel)) as handle:
            text = handle.read()
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'set_value'
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'row18_alpha'):
                calls.append({'file': rel, 'line': node.lineno})
    return {'n_files_scanned': len(files), 'calls': calls}


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
    n_indices_init = 0
    per_block_a = {}
    for node_id, blocks in init.items():
        for key, st in blocks.items():
            ok = _init_block_ok(st) and st['premium_in_force_at_the_solve']['alpha'] == ALPHA_RUN
            if st['row18_wired']:
                n_indices_init += sum(f['n_indices'] for f in st['structural']['families'].values())
            per_block_a[f'DSO:{node_id}:{key}'] = {**_strip(st), 'pass': ok}
            a_ok = a_ok and ok
    tso_ok = all(not v['row18_components_present'] and not v['row18_alpha_admm_present']
                 for v in probe.tso_state.values())
    results['A_init_structurally_inactive_through_production'] = {
        'call': 'shared_resources_planning._run_operational_planning(planning, get_initial_candidate_solution())',
        'run_alpha': ALPHA_RUN, 'run_record': run_record,
        'n_dso_blocks_at_init_solve': n_blocks, 'tso_blocks_carry_no_row18': tso_ok,
        'per_block': per_block_a, 'n_row_indices_checked': n_indices_init,
        'pass': a_ok and tso_ok and n_indices_init > 0,
    }

    _log('B: EMULATED loaded solution (NOT a solve) on every DSO block, then _prepare_distribution_objectives_for_admm')
    emulation_b = emulate_all(planning, dso_models)
    post = post_prepare_states(planning, dso_models)
    b_ok = True
    c_ok = True
    per_block_b, per_block_c = {}, {}
    n_steps_b = 0
    signs_b = {'e_pos': 0, 'e_neg': 0, 'e_zero': 0}
    structural_b = {}
    for node_id, blocks in post.items():
        for key, st in blocks.items():
            st0 = init[node_id][key]
            ok = (_active_block_ok(st)
                  and st['row18_premium'] == st0['row18_premium']
                  and st['row18_premium'] == st['expected_market_price']
                  and st['row18_charge_expression_id'] == st0['row18_charge_expression_id']
                  and st['row18_charge_expression_str_sha256'] == st0['row18_charge_expression_str_sha256'])
            if 'unit_steps' in st:
                ok = _steps_ok(st['unit_steps'], expect_zero=False) and ok
                n_steps_b += len(st['unit_steps'])
            for k in signs_b:
                signs_b[k] += st['structural']['e_sign_counts'][k]
            per_block_b[f'DSO:{node_id}:{key}'] = {**_strip(st), 'pass': ok}
            structural_b[f'DSO:{node_id}:{key}'] = st['structural']
            b_ok = b_ok and ok
            n_s, n_t = st['n_scenarios'], st['n_periods']
            delta = {k: st['counts'][k] - st0['counts'][k] for k in st['counts']}
            expected_delta = {'n_var_data': 0, 'n_constraint_data': 0, 'n_constraint_data_active': 2 * n_s * n_t,
                              'n_expression_data': 0, 'n_var_data_unfixed': 4 * n_s * n_t}
            same = (st['component_names'] == st0['component_names'] and delta == expected_delta)
            per_block_c[f'DSO:{node_id}:{key}'] = {'component_names_identical': st['component_names'] == st0['component_names'],
                                                   'counts_init': st0['counts'], 'counts_after_prepare': st['counts'],
                                                   'delta_after_minus_init': delta,
                                                   'expected_delta': expected_delta, 'pass': same}
            c_ok = c_ok and same
    non_vacuity_b = non_vacuity(structural_b, emulation_b)
    non_vacuous_b = non_vacuity_b['pass'] and signs_b['e_pos'] > 0 and signs_b['e_neg'] > 0

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

    writers = _row18_alpha_writers()
    writers_ok = writers['calls'] == [] and writers['n_files_scanned'] > 0

    results['B_activation_with_settlement_weight'] = {
        'call': 'shared_resources_planning._prepare_distribution_objectives_for_admm(distribution_networks, models)',
        'mechanism': ('structural (W54): per index minimal split, unfix, then activate; row18_alpha untouched; '
                      'no rebuild'),
        'solution_source': SOLUTION_SOURCE, 'emulation_per_block': emulation_b,
        'e_sign_counts_all_blocks': signs_b, 'non_vacuity': non_vacuity_b,
        'minimal_split_assertion_non_vacuous': non_vacuous_b,
        'tolerances_declared': {'charge_rel': TOL_REL_CHARGE, 'objective_rel': TOL_REL_OBJECTIVE,
                                'body_abs': TOL_BODY_ABS, 'emulation_abs': TOL_EMULATION_ABS},
        'per_block': per_block_b, 'n_unit_steps': n_steps_b,
        'admm_objective': {'placeholder_objective_scale': PLACEHOLDER_OBJECTIVE_SCALE, 'error': admm_error,
                           'per_block': admm_refs, 'pass': admm_ok},
        'row18_alpha_set_value_calls_in_tracked_root_py': writers, 'nothing_writes_row18_alpha': writers_ok,
        'pass': b_ok and admm_ok and writers_ok and n_steps_b > 0 and non_vacuous_b,
    }
    results['C_structure_across_phases'] = {'per_block': per_block_c, 'pass': c_ok}
    return init, {node_id: {key: {'counts': st['counts']} for key, st in blocks.items()}
                  for node_id, blocks in post.items()}


def check_2x2_vs_alpha0(planning0, init_run, post_run, results):
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
            a_post = post_run[node_id][key]
            delta_init = {k: st['counts'][k] - z_init['counts'][k] for k in st['counts']}
            delta_post = {k: a_post['counts'][k] - z_post['counts'][k] for k in a_post['counts']}
            # W57 (spec v3): n_expression_data +1 in both phases -- `row18_deviation_charge` is a scalar Expression.
            # v2 predicted +0 (a spec error; r5's C2 FAIL stays on the record).
            expected_init = {'n_var_data': 4 * n_s * n_t, 'n_constraint_data': 2 * n_s * n_t,
                             'n_constraint_data_active': 0, 'n_expression_data': 1, 'n_var_data_unfixed': 0}
            expected_post = {'n_var_data': 4 * n_s * n_t, 'n_constraint_data': 2 * n_s * n_t,
                             'n_constraint_data_active': 2 * n_s * n_t, 'n_expression_data': 1,
                             'n_var_data_unfixed': 4 * n_s * n_t}
            ok = (not z_init['row18_wired'] and z_init['row18_alpha'] is None and z_init['row18_alpha_admm'] is None
                  and not z_post['row18_wired'] and z_post['row18_alpha_admm'] is None
                  and z_init['component_names'] == z_post['component_names']
                  and z_init['voltage_pin_wired'] and st['voltage_pin_wired']
                  and extra == sorted(ROW18_COMPONENTS) and not missing
                  and delta_init == expected_init and delta_post == expected_post)
            per_block[f'DSO:{node_id}:{key}'] = {
                'n_scenarios': n_s, 'n_periods': n_t, 'components_extra_at_alpha_run': extra,
                'components_missing_at_alpha_run': missing,
                'delta_vs_alpha0_at_init': delta_init, 'expected_delta_at_init': expected_init,
                'delta_vs_alpha0_after_activation': delta_post, 'expected_delta_after_activation': expected_post,
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
        b = _init_block_ok(st)
        per_block[f'DSO:{node_id}:{key}'] = {**_strip(st), 'pass': b}
        ok = ok and b
    n_init_blocks = len(per_block)
    _log('A2: EMULATED loaded solution (NOT a solve) on the node, then _prepare_distribution_objectives_for_admm')
    emulation = emulate_all(planning, {node_id: model}, node_ids=(node_id,))
    srp._prepare_distribution_objectives_for_admm({node_id: dn}, {node_id: model})
    per_block_active = {}
    structural = {}
    ok_active = True
    signs = {'e_pos': 0, 'e_neg': 0, 'e_zero': 0}
    for y in dn.years:
        for d in dn.days:
            st = block_state(model[y][d], dn.network[y][d], dn.params, 'active')
            b = _active_block_ok(st)
            for k in signs:
                signs[k] += st['structural']['e_sign_counts'][k]
            per_block_active[f'DSO:{node_id}:{y}:{d}'] = {**_strip(st), 'pass': b}
            structural[f'DSO:{node_id}:{y}:{d}'] = st['structural']
            ok_active = ok_active and b
    nv = non_vacuity(structural, emulation)
    n_expected = len(dn.years) * len(dn.days)
    negative = negative_control_swapped_family_table(
        model[next(iter(dn.years))][next(iter(dn.days))], f'DSO:{node_id}:{next(iter(dn.years))}:{next(iter(dn.days))}')
    results['A2_parallel_builder_in_process'] = {
        'call': f'create_distribution_network_model({node_id}, ..., premium_alpha={ALPHA_RUN}, premium_floor=None)',
        'solution_source': SOLUTION_SOURCE, 'emulation_per_block': emulation,
        'per_block_init': per_block, 'per_block_after_activation': per_block_active,
        'e_sign_counts_after_activation': signs, 'non_vacuity': nv, 'stub_calls': probe.counts(),
        'negative_control_production_body_check': negative,
        'pass': (ok and ok_active and nv['pass'] and negative['pass'] and n_init_blocks == n_expected
                 and len(per_block_active) == n_expected)}


def negative_control_swapped_family_table(block, key):
    """W57 NEGATIVE CONTROL for production's body check (item 2), run AFTER A2's activation assertions are recorded:
    the block is returned to the initialisation state by production's own `_set_row18_inactive_for_initialisation`
    (the emulated flows and expectations are untouched), `_ROW18_DEVIATION_FAMILIES` is replaced for ONE call by a
    table whose flow/expectation are swapped between the P and Q families, and `_activate_row18_with_settlement`
    MUST raise its body-check RuntimeError. The table is restored in `finally` and asserted restored."""
    original = srp._ROW18_DEVIATION_FAMILIES
    swapped = ((original[0][0], original[0][1], original[0][2], original[1][3], original[1][4]),
               (original[1][0], original[1][1], original[1][2], original[0][3], original[0][4]))
    srp._set_row18_inactive_for_initialisation(block)
    error = None
    srp._ROW18_DEVIATION_FAMILIES = swapped
    try:
        srp._activate_row18_with_settlement(block)
    except RuntimeError as exc:
        error = str(exc)
    finally:
        srp._ROW18_DEVIATION_FAMILIES = original
    restored = srp._ROW18_DEVIATION_FAMILIES is original
    raised_by_body_check = error is not None and 'defining-row body' in error
    return {'block': key, 'swapped_table': [list(r) for r in swapped], 'raised': error is not None,
            'error': error, 'raised_by_body_check': raised_by_body_check, 'table_restored': restored,
            'note': 'the block is left in the swapped-activation state; nothing after this reads it',
            'pass': bool(raised_by_body_check and restored)}


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
    with InitProbe(planning) as probe:
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
def capture_path_checklist():
    """Before anything is built (CLAUDE.md rule eleven): the capture paths this script's assertions rely on exist,
    and the live production functions are the structural (W54) form."""
    import inspect
    inactive = getattr(srp, '_set_row18_inactive_for_initialisation', None)
    activate = getattr(srp, '_activate_row18_with_settlement', None)
    sources = [inspect.getsource(f) for f in (inactive, activate) if callable(f)]
    activate_src = inspect.getsource(activate) if callable(activate) else ''
    with open(os.path.join(REPO, SPEC_V3_PATH)) as handle:
        spec_v3 = json.load(handle)
    declared = spec_v3.get(SPEC_V3_KEY, {}).get('declared_constants', {})
    return {
        'frozen_spec_v2_sha256_matches': _sha256_file(os.path.join(REPO, SPEC_V2_PATH)) == SPEC_V2_SHA256,
        'frozen_spec_v3_sha256_matches': _sha256_file(os.path.join(REPO, SPEC_V3_PATH)) == SPEC_V3_SHA256,
        'frozen_spec_v3_constants_equal_this_scripts': declared == DECLARED_CONSTANTS,
        'production_body_check_present_with_declared_tolerance': (
            getattr(srp, '_ROW18_ACTIVATION_BODY_TOL', None) == PRODUCTION_BODY_TOL_DECLARED
            and 'pe.value(row[index].body) - pe.value(row[index].upper)' in activate_src
            and 'if not abs(body_residual) <= _ROW18_ACTIVATION_BODY_TOL:' in activate_src),
        'emulation_capture_path_reference_gen_idx': callable(getattr(Network, 'get_reference_gen_idx', None)),
        'both_production_functions_present': callable(inactive) and callable(activate),
        'production_family_table_equals_this_scripts': getattr(srp, '_ROW18_DEVIATION_FAMILIES', None) == FAMILIES,
        'production_functions_never_write_row18_alpha': bool(sources) and all(
            'row18_alpha.set_value' not in s and 'row18_alpha_admm' not in s for s in sources),
        'W39_counts_helper_reports_active_and_unfixed': {'n_constraint_data_active', 'n_var_data_unfixed'} <= set(
            W39C._component_counts(pe.ConcreteModel()).keys()),
    }


DECLARED_CONSTANTS = {
    'EMU_MAG_BASE': EMU_MAG_BASE, 'EMU_MAG_STEP': EMU_MAG_STEP, 'EMU_NEG_EVERY': EMU_NEG_EVERY,
    'EMU_EXPECTATION': [list(x) for x in EMU_EXPECTATION], 'TOL_EMULATION_ABS': TOL_EMULATION_ABS,
    'TOL_BODY_ABS': TOL_BODY_ABS, 'MIN_SIGN_SHARE': MIN_SIGN_SHARE, 'MIN_MAG_GAP': MIN_MAG_GAP,
    'PRODUCTION_BODY_TOL_DECLARED': PRODUCTION_BODY_TOL_DECLARED}


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
    checklist = capture_path_checklist()
    if not all(checklist.values()):
        print(f'REFUSED: capture-path checklist failed: {checklist}', file=sys.stderr)
        return 1
    os.makedirs(out_dir)
    case_dir = os.path.join(out_dir, 'cases')
    started = time.time()
    results, instances = {}, {}
    _log(STAGE)
    _log(f'capture-path checklist {checklist}')

    p2a, instances['2x2_alpha05'] = read_planning('2x2_alpha05', OVERRIDES_2X2, case_dir, scratch)
    init_run, post_run = check_2x2(p2a, results)
    del p2a
    p20, instances['2x2_alpha0'] = read_planning('2x2_alpha0', OVERRIDES_2X2, case_dir, scratch)
    check_2x2_vs_alpha0(p20, init_run, post_run, results)
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
        'frozen_spec_v2': {'path': SPEC_V2_PATH, 'sha256_pinned': SPEC_V2_SHA256,
                           'sha256_observed': _sha256_file(os.path.join(REPO, SPEC_V2_PATH))},
        'frozen_spec_v3': {'path': SPEC_V3_PATH, 'key': SPEC_V3_KEY, 'sha256_pinned': SPEC_V3_SHA256,
                           'sha256_observed': _sha256_file(os.path.join(REPO, SPEC_V3_PATH))},
        'solution_source': SOLUTION_SOURCE, 'declared_constants': DECLARED_CONSTANTS,
        'capture_path_checklist': checklist,
        'timestamp_utc': _utc(), 'interpreter': sys.executable, 'argv': sys.argv,
        'script': os.path.basename(__file__), 'script_sha256': _sha256_file(os.path.abspath(__file__)),
        'production_sha256': {n: _sha256_file(os.path.join(REPO, n)) for n in (
            'shared_resources_planning.py', 'model_construction_helpers.py', 'admm_parameters.py')},
        'imported_sha256': {n: _sha256_file(os.path.join(REPO, n)) for n in (
            'p515_s51_row18_zero_solve_checks.py', 'p515_s44_scale_measurement.py', 'p513_solve_profile_guard.py')},
        'pyomo_version': __import__('pyomo').version.version,
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
