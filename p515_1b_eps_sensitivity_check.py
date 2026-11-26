"""
P5.15-1b -- EPS_ESSO_THROUGHPUT sensitivity check (Addendum 2 dispatch,
PLANNER_BRIEF_2026-09-13.md, "eps check (before the gates)").

Two direct ESSO solves of the SAME reformulated instance, differing ONLY in
`EPS_ESSO_THROUGHPUT` (1e-3 vs 1e-7). This precedes gates G1-G5 so that G1
measures the reformulation, not the regularization weight.

INSTANCE. Reuses VERBATIM the construction path of the Step 1 reformulation
smoke test (`p515_1_esso_reform_smoke.py`, commit `b03c9b14`,
`P5_15_PLANNER_STEP1_PROGRESS.md` section 3): a fresh production planning
problem (`p56a_oracle.fresh_planning`), a zero-investment candidate for every
active distribution-network node except nodes 7 and 9
(`SMOKE_NODES_WITH_INVESTMENT`, S=1.00 MVA / E=2.00 MVAh in the first
representative year), and the same non-trivial +/-10% duty-cycle `p_req`
profile (`_set_nonzero_charge_discharge_request`, amplitude = 0.10 * S). The
functions are IMPORTED from that module, not reimplemented -- this is exactly
the smoke test's fixture, not a new one. Node 7 is the instance this task is
about; node 9 comes along because the smoke test's construction gives it an
investment too (not invented here); node 5 (zero investment) comes along
because `create_shared_energy_storage_model` always solves every ACTIVE node.

EPS OVERRIDE. `EPS_ESSO_THROUGHPUT` enters `shared_energy_storage_data.py`'s
namespace via `from helper_functions import *` -> `from definitions import
*`, so inside that module it is a plain module-level global that
`_build_subproblem` looks up by name at call time. This harness monkeypatches
the module attribute `shared_energy_storage_data.EPS_ESSO_THROUGHPUT` before
each arm's build and restores it immediately afterward (verified: an
assertion re-reads `definitions.EPS_ESSO_THROUGHPUT` after every arm and
fails loudly if it is not the untouched case-default 1e-3).
`definitions.py` is never edited on disk.

NO ADMM RUN. The production entry point used is `create_shared_energy_storage_model`
(shared_resources_planning.py) exactly as the smoke test used it. That
function FIXES `es_pnet`/`es_qnet` to the supplied p_req/q_req profile
(`fix_or_set`) and solves `model.objective == model.feasibility_penalty`
directly; it never builds `model.admm_objective` (that expression is only
created by `update_shared_energy_storage_model_to_admm`, which this harness
never calls). One direct consequence, reported rather than hidden: because
`es_pnet` is a HARD FIX to an input that does not depend on eps, its value is
identical between arms by construction, not by a converged numerical
coincidence -- see the report for the reading of what this means for the
"pnet displacement" quantity as literally specified.

GUARD. A `SolveProfileGuard` is armed for the whole run, permitting solves
only from `shared_energy_storage_data.py:_run_solver_attempt`. Declared count
in advance: 6 (2 arms x 3 active nodes -- 5, 7, 9 -- with the Step-1 smoke
test as prior evidence that nodes 7 and 9 converge to `optimal` with no
recovery retry at eps=1e-3, and node 5, zero investment, is a trivial solve).
If a recovery retry fires on either arm, the observed count will exceed 6;
that is reported as an extra, explicitly declared solve per node (see
"Results" in the written report), not hidden.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_1b_eps_sensitivity_check.py
"""

import hashlib
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402

import definitions  # noqa: E402
import p56a_oracle as oracle  # noqa: E402
import shared_energy_storage_data as SED  # noqa: E402
from shared_energy_storage_data import _esso_cohort_pair_is_within_lifetime  # noqa: E402
from shared_resources_planning import (  # noqa: E402
    create_admm_variables,
    create_shared_energy_storage_model,
)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_1_esso_reform_smoke import (  # noqa: E402
    SMOKE_NODES_WITH_INVESTMENT,
    S_CANDIDATE_MVA,
    E_CANDIDATE_MVAH,
    _zero_candidate,
    _set_nonzero_charge_discharge_request,
)

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P5151')
LOG_ROOT = os.path.join(OUT_DIR, 'eps_check_logs')
os.makedirs(OUT_DIR, exist_ok=True)

PERMITTED = [('shared_energy_storage_data.py', '_run_solver_attempt')]
EXPECTED_SOLVES = 6  # 2 arms x 3 active nodes, declared in advance (see docstring)

ARMS = (('eps1e-3', 1e-3), ('eps1e-7', 1e-7))

CASE_DEFAULT_EPS = 1e-3  # definitions.py committed value -- must be unchanged after every arm

IPOPT_ITER_RE = re.compile(r'Number of Iterations\.+:\s*(\d+)')

REPORT = {
    'stage': 'P5.15-1b',
    'objective': (
        'EPS_ESSO_THROUGHPUT sensitivity check (Addendum 2, "eps check before the '
        'gates"): two direct ESSO solves of the same reformulated node-7 instance, '
        'differing only in EPS_ESSO_THROUGHPUT (1e-3 vs 1e-7).'
    ),
    'authority': 'PLANNER_BRIEF_2026-09-13.md, Addendum 2, bullet "eps check (before the gates)"',
    'construction_commit': 'b03c9b14',
    'construction_source': 'p515_1_esso_reform_smoke.py (imported, not reimplemented)',
    # Declared BEFORE running, per CLAUDE.md ("if a quantity has no capture path, say
    # so before running rather than after"): `create_shared_energy_storage_model` never
    # builds the TSO/DSO models this task is barred from solving, so production's
    # `gross_operational_cost` convention (TSO objective + sum of DSO objectives,
    # shared_resources_planning.py:727-729) has NO capture path in this experiment.
    # Only the ESSO's own `model.objective` (== `model.feasibility_penalty`, no AL
    # terms -- see docstring) is reported.
    'capture_path_notes': {
        'gross_operational_cost': (
            'NOT CAPTURED. Production defines gross_operational_cost as the TSO '
            "objective plus the sum of DSO objectives (shared_resources_planning.py "
            "get_primal_value calls at lines ~727-729); it structurally EXCLUDES the "
            "ESSO objective and requires solving the TSO/DSO SMOPFs, which this task's "
            "'no ADMM run, direct ESSO solves only' restriction forbids. There is no "
            'quantity in this experiment that convention applies to.'
        ),
    },
}


def parse_ipopt_log(log_path):
    if not os.path.exists(log_path):
        return {'error': 'log file not found', 'path': log_path}
    text = open(log_path, 'r', errors='replace').read()
    exit_lines = [ln.strip() for ln in text.splitlines() if ln.startswith('EXIT:')]
    iters = IPOPT_ITER_RE.findall(text)
    return {
        'log_path': log_path,
        'exit_line': exit_lines[-1] if exit_lines else None,
        'n_iterations': int(iters[-1]) if iters else None,
    }


def sha256_of(payload_obj):
    blob = json.dumps(payload_obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _stringify(mapping):
    out = {}
    for key, value in mapping.items():
        str_key = '|'.join(str(part) for part in key) if isinstance(key, tuple) else str(key)
        out[str_key] = value
    return out


def run_arm(tag, eps_value):
    arm_dir = os.path.join(LOG_ROOT, tag)
    os.makedirs(arm_dir, exist_ok=True)

    original_eps_in_module = SED.EPS_ESSO_THROUGHPUT

    with redirect_stdout(io.StringIO()):
        planning = oracle.fresh_planning(f'p5151b_eps_{tag}')
    shared_ess_data = planning.shared_ess_data

    consensus_vars, dual_vars = create_admm_variables(planning)
    candidate_solution = _zero_candidate(planning)
    first_year = list(planning.years)[0]
    for node_id in SMOKE_NODES_WITH_INVESTMENT:
        candidate_solution[node_id][first_year] = {'s': S_CANDIDATE_MVA, 'e': E_CANDIDATE_MVAH}
        _set_nonzero_charge_discharge_request(
            planning, consensus_vars, node_id, amplitude_mw=0.10 * S_CANDIDATE_MVA
        )

    # Harness override, NOT a production-code edit: reassigns the module-level
    # global that `_build_subproblem` looks up at call time. Restored in `finally`.
    SED.EPS_ESSO_THROUGHPUT = eps_value

    cwd_before = os.getcwd()
    started = time.time()
    try:
        os.chdir(arm_dir)
        with redirect_stdout(io.StringIO()):
            esso_models, esso_op_results = create_shared_energy_storage_model(
                shared_ess_data, consensus_vars, candidate_solution
            )
    finally:
        os.chdir(cwd_before)
        SED.EPS_ESSO_THROUGHPUT = original_eps_in_module
    wall = time.time() - started

    # Belt-and-braces: definitions.py itself must be untouched by this harness.
    assert definitions.EPS_ESSO_THROUGHPUT == CASE_DEFAULT_EPS, (
        f'definitions.EPS_ESSO_THROUGHPUT changed on disk to '
        f'{definitions.EPS_ESSO_THROUGHPUT}; harness must not edit the constant.'
    )
    assert SED.EPS_ESSO_THROUGHPUT == CASE_DEFAULT_EPS, (
        'shared_energy_storage_data.EPS_ESSO_THROUGHPUT not restored after arm '
        f'{tag}: {SED.EPS_ESSO_THROUGHPUT}'
    )

    arm = {
        'tag': tag,
        'eps_value_in_force': eps_value,
        'wall_seconds': wall,
        'nodes': {},
    }

    for node_id in shared_ess_data.active_distribution_network_nodes:
        model = esso_models[node_id]
        result = esso_op_results.get(node_id)
        termination = str(result.solver.termination_condition) if result is not None else None
        status = str(result.solver.status) if result is not None else None

        primary_log = os.path.join(arm_dir, f'optim_log_node_{node_id}.txt')
        recovery_log = os.path.join(arm_dir, f'optim_log_node_{node_id}_recovery.txt')
        log_tail = parse_ipopt_log(primary_log)
        recovery_log_tail = parse_ipopt_log(recovery_log) if os.path.exists(recovery_log) else None

        try:
            objective = pe.value(model.objective)
        except Exception as exc:  # noqa: BLE001 -- report, do not hide
            objective = f'ERROR: {exc}'
        try:
            feasibility_penalty = pe.value(model.feasibility_penalty)
        except Exception as exc:  # noqa: BLE001
            feasibility_penalty = f'ERROR: {exc}'

        pnet = {}
        for y in model.years:
            for d in model.days:
                for p in model.periods:
                    pnet[(y, d, p)] = pe.value(model.es_pnet[y, d, p])

        pch_pdch = {}
        s_max_by_cohort = {}
        for y_inv in model.years:
            for y in model.years:
                if not _esso_cohort_pair_is_within_lifetime(model, y_inv, y):
                    continue
                if model._esso_cohort_inactive.get(y_inv, False):
                    continue
                s_max = pe.value(model.es_s_rated_per_unit[y_inv, y])
                s_max_by_cohort[(y_inv, y)] = s_max
                for d in model.days:
                    for p in model.periods:
                        pch = pe.value(model.es_pch_per_unit[y_inv, y, d, p])
                        pdch = pe.value(model.es_pdch_per_unit[y_inv, y, d, p])
                        pch_pdch[(y_inv, y, d, p)] = (pch, pdch)

        arm['nodes'][node_id] = {
            'termination_condition': termination,
            'status': status,
            'objective': objective,
            'feasibility_penalty': feasibility_penalty,
            'log_tail': log_tail,
            'recovery_log_tail': recovery_log_tail,
            'pnet': pnet,
            'pch_pdch': pch_pdch,
            's_max_by_cohort': s_max_by_cohort,
        }

    arm['complementarity_detector_absolute_max_violation'] = (
        shared_ess_data.get_complementarity_violation(esso_models)
    )
    arm['recovery_diagnostics'] = list(shared_ess_data.solver_recovery_diagnostics)
    arm['total_objective_all_active_nodes'] = shared_ess_data.get_primal_value(esso_models)
    arm['total_feasibility_penalty_all_active_nodes'] = shared_ess_data.get_feasibility_penalty(esso_models)

    return arm


def ratio_complementarity_detector(arm):
    """max over active cohort-periods of min(pch, pdch) / s_max, skipping s_max == 0."""
    max_ratio = 0.0
    argmax = None
    for node_id, node_data in arm['nodes'].items():
        s_max_by_cohort = node_data['s_max_by_cohort']
        for (y_inv, y, d, p), (pch, pdch) in node_data['pch_pdch'].items():
            s_max = s_max_by_cohort.get((y_inv, y))
            if not s_max:  # skips None and exactly 0.0
                continue
            ratio = min(pch, pdch) / s_max
            if ratio > max_ratio:
                max_ratio = ratio
                argmax = {
                    'node': node_id, 'y_inv': y_inv, 'y': y, 'd': d, 'p': p,
                    'pch': pch, 'pdch': pdch, 's_max': s_max,
                }
    return max_ratio, argmax


def pnet_displacement(arm_a, arm_b):
    max_abs = 0.0
    argmax = None
    per_node = {}
    for node_id in arm_a['nodes']:
        node_max = 0.0
        node_argmax = None
        pnet_a = arm_a['nodes'][node_id]['pnet']
        pnet_b = arm_b['nodes'][node_id]['pnet']
        for key in pnet_a:
            diff = abs(pnet_a[key] - pnet_b[key])
            if diff > node_max:
                node_max = diff
                node_argmax = key
            if diff > max_abs:
                max_abs = diff
                argmax = {'node': node_id, 'y_d_p': key, 'a': pnet_a[key], 'b': pnet_b[key]}
        per_node[node_id] = {'max_abs_diff': node_max, 'argmax_y_d_p': node_argmax}
    return max_abs, argmax, per_node


def dispatch_displacement(arm_a, arm_b):
    """Supplementary (not the literal ask): max abs diff in pch and in pdch,
    the actual free decision variables in this fixed-es_pnet construction."""
    max_abs = 0.0
    argmax = None
    per_node = {}
    for node_id in arm_a['nodes']:
        node_max = 0.0
        node_argmax = None
        pp_a = arm_a['nodes'][node_id]['pch_pdch']
        pp_b = arm_b['nodes'][node_id]['pch_pdch']
        for key in pp_a:
            pch_a, pdch_a = pp_a[key]
            pch_b, pdch_b = pp_b.get(key, (None, None))
            if pch_b is None:
                continue
            diff = max(abs(pch_a - pch_b), abs(pdch_a - pdch_b))
            if diff > node_max:
                node_max = diff
                node_argmax = key
            if diff > max_abs:
                max_abs = diff
                argmax = {'node': node_id, 'y_inv_y_d_p': key,
                          'a': (pch_a, pdch_a), 'b': (pch_b, pdch_b)}
        per_node[node_id] = {'max_abs_diff': node_max, 'argmax_y_inv_y_d_p': node_argmax}
    return max_abs, argmax, per_node


def _serializable(arm):
    out = dict(arm)
    out['nodes'] = {}
    for node_id, node_data in arm['nodes'].items():
        nd = dict(node_data)
        nd['pnet'] = _stringify(node_data['pnet'])
        nd['pch_pdch'] = _stringify(node_data['pch_pdch'])
        nd['s_max_by_cohort'] = _stringify(node_data['s_max_by_cohort'])
        out['nodes'][str(node_id)] = nd
    return out


def main():
    os.makedirs(LOG_ROOT, exist_ok=True)

    baseline = oracle.load_baseline()
    instance_descriptor = {
        'construction_commit': 'b03c9b14',
        'construction_module': 'p515_1_esso_reform_smoke.py',
        'baseline_checksum': baseline['checksum'],
        'smoke_nodes_with_investment': list(SMOKE_NODES_WITH_INVESTMENT),
        's_candidate_mva': S_CANDIDATE_MVA,
        'e_candidate_mvah': E_CANDIDATE_MVAH,
        'amplitude_mw_formula': '0.10 * S_CANDIDATE_MVA',
        'primary_node_for_this_task': 7,
    }
    instance_hash = sha256_of(instance_descriptor)
    REPORT['instance'] = instance_descriptor
    REPORT['instance_hash_sha256'] = instance_hash

    guard = SolveProfileGuard(PERMITTED, label='P5.15-1b eps check').install()
    try:
        arm_reports = {}
        for tag, eps_value in ARMS:
            print(f'[P5.15-1b] running arm {tag} (EPS_ESSO_THROUGHPUT={eps_value}) ...', flush=True)
            arm_reports[tag] = run_arm(tag, eps_value)
            for node_id, node_data in arm_reports[tag]['nodes'].items():
                print(
                    f'  node {node_id}: {node_data["termination_condition"]} '
                    f'iters={node_data["log_tail"].get("n_iterations")} '
                    f'objective={node_data["objective"]}', flush=True
                )
    finally:
        guard.uninstall()

    guard_failures = guard.verify(EXPECTED_SOLVES)
    REPORT['guard'] = {
        'permitted_call_sites': PERMITTED,
        'expected_solves_declared_in_advance': EXPECTED_SOLVES,
        'observed_counts': guard.counts,
        'permitted_sites_hit': guard.permitted_sites,
        'verify_failures': guard_failures,
    }

    arm_a = arm_reports['eps1e-3']
    arm_b = arm_reports['eps1e-7']

    pnet_max_abs, pnet_argmax, pnet_per_node = pnet_displacement(arm_a, arm_b)
    dispatch_max_abs, dispatch_argmax, dispatch_per_node = dispatch_displacement(arm_a, arm_b)

    ratio_a, ratio_a_argmax = ratio_complementarity_detector(arm_a)
    ratio_b, ratio_b_argmax = ratio_complementarity_detector(arm_b)

    decision_rule_pass_1e3 = ratio_a <= 1e-6

    REPORT['results'] = {
        'pnet_displacement': {
            'convention': (
                'max abs diff in es_pnet[y,d,p] between the two arms, over all '
                '(year, day, period), for every active node produced by the '
                'production entry point. es_pnet is HARD-FIXED to an eps-independent '
                'input in this construction (create_shared_energy_storage_model uses '
                'fix_or_set, not an AL penalty) -- see docstring for the reading.'
            ),
            'predeclared_prediction': '< 1e-6 p.u.',
            'max_abs_diff_all_nodes': pnet_max_abs,
            'max_abs_diff_node_7': pnet_per_node.get(7, {}),
            'per_node': {str(k): v for k, v in pnet_per_node.items()},
            'argmax': pnet_argmax,
        },
        'dispatch_displacement_supplementary': {
            'convention': (
                'max(|pch_a-pch_b|, |pdch_a-pdch_b|) over all active cohort-periods -- '
                'the actual free decision in this construction, reported because pnet '
                'is fixed and therefore trivially eps-invariant by construction.'
            ),
            'max_abs_diff_all_nodes': dispatch_max_abs,
            'max_abs_diff_node_7': dispatch_per_node.get(7, {}),
            'per_node': {str(k): v for k, v in dispatch_per_node.items()},
            'argmax': dispatch_argmax,
        },
        'complementarity_detector_ratio_form': {
            'convention': 'max over active cohort-periods of min(pch, pdch) / s_max, skipping s_max == 0',
            'predeclared_prediction': 'ratio <= 1e-6 at eps=1e-3; drift upward at eps=1e-7 argues for keeping 1e-3',
            'eps1e-3': {'max_ratio': ratio_a, 'argmax': ratio_a_argmax},
            'eps1e-7': {'max_ratio': ratio_b, 'argmax': ratio_b_argmax},
        },
        'complementarity_detector_absolute_form': {
            'convention': 'SharedEnergyStorageData.get_complementarity_violation(models): '
                          'max(0, min(pch, pdch) - 1e-6 * s_max) over active cohort-periods',
            'eps1e-3': arm_a['complementarity_detector_absolute_max_violation'],
            'eps1e-7': arm_b['complementarity_detector_absolute_max_violation'],
        },
        'iterations_and_termination': {
            tag: {
                str(node_id): {
                    'termination_condition': nd['termination_condition'],
                    'status': nd['status'],
                    'n_iterations_primary': nd['log_tail'].get('n_iterations'),
                    'exit_line_primary': nd['log_tail'].get('exit_line'),
                    'recovery_fired': nd['recovery_log_tail'] is not None,
                    'n_iterations_recovery': (nd['recovery_log_tail'] or {}).get('n_iterations'),
                    'exit_line_recovery': (nd['recovery_log_tail'] or {}).get('exit_line'),
                }
                for node_id, nd in arm_reports[tag]['nodes'].items()
            }
            for tag in arm_reports
        },
        'recovery_diagnostics': {
            tag: arm_reports[tag]['recovery_diagnostics'] for tag in arm_reports
        },
        'objective_and_cost': {
            'convention': (
                'ESSO model.objective == model.feasibility_penalty (no AL terms; see '
                'docstring on why gross_operational_cost has no capture path here). '
                'total_* sums over the 3 active nodes (5, 7, 9).'
            ),
            **{
                tag: {
                    'total_objective_all_active_nodes': arm_reports[tag]['total_objective_all_active_nodes'],
                    'total_feasibility_penalty_all_active_nodes': arm_reports[tag]['total_feasibility_penalty_all_active_nodes'],
                    'per_node_objective': {
                        str(node_id): nd['objective'] for node_id, nd in arm_reports[tag]['nodes'].items()
                    },
                }
                for tag in arm_reports
            },
        },
    }

    REPORT['decision_rule'] = {
        'rule': (
            'G1-G5 run at eps=1e-3 regardless of outcome UNLESS the 1e-3 arm shows a '
            'complementarity detector ratio above 1e-6, in which case STOP and report; '
            'do not recommend a value.'
        ),
        'eps1e-3_ratio_detector_max': ratio_a,
        'threshold': 1e-6,
        'eps1e-3_passes_threshold': decision_rule_pass_1e3,
        'verdict': (
            'PASS -- gates may proceed at eps=1e-3 per the predeclared rule (Planner decides).'
            if decision_rule_pass_1e3 else
            'STOP -- 1e-3 arm exceeds the 1e-6 ratio threshold; no value is recommended.'
        ),
    }

    full_path = os.path.join(OUT_DIR, 'eps_sensitivity_check_full.json')
    with open(full_path, 'w') as fh:
        json.dump({
            'meta': {k: v for k, v in REPORT.items() if k not in ('results',)},
            'arms_full': {tag: _serializable(arm_reports[tag]) for tag in arm_reports},
        }, fh, indent=2, default=str)

    summary_path = os.path.join(OUT_DIR, 'eps_sensitivity_check_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(REPORT, fh, indent=2, default=str)

    print(json.dumps(REPORT, indent=2, default=str))
    print(f'\n[P5.15-1b] full per-period data written to {full_path}')
    print(f'[P5.15-1b] summary written to {summary_path}')
    print(f'\n[P5.15-1b] guard verify failures: {guard_failures}')
    print(f'[P5.15-1b] decision-rule verdict: {REPORT["decision_rule"]["verdict"]}')


if __name__ == '__main__':
    main()
