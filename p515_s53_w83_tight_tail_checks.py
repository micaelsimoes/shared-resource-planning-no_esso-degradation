"""
P5.15 Addendum 46 ruling 7, Planner task W83 -- the convergence-depth "TIGHT TAIL": ZERO-SOLVE checks, calibration and
the frozen spec v28. Nothing here builds a model's solve, runs a solve or re-runs a harness; every mode arms
`SolveProfileGuard(permitted=())` at import (before `network` / `shared_resources_planning` are imported) and
verifies it at EXACTLY 0 on every exit path.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 46 ("(7) Convergence depth: option (c) in its minimal form --
compl_inf_tol 1e-4 -> 1e-6 only in the certifying cycles, switched by the same all-channels-inside-tolerance
condition that turns AA off; everything before the tail bitwise unchanged; per-solve floor status reported for
every cell from now on"); Planner task W83 (ruling on the 1e-4 / 5e-4 wording: implement the intent, 1e-6 on ALL
network solves in the certifying cycles). Predecessor spec v27 654046ec (not edited).

WHAT IS UNDER TEST (production, commit "W83 step 1"):
  network.py                 `parse_ipopt_attempt_segment`, `_append_ipopt_solve_record` (per-attempt floor status,
                             read from the attempt's own byte range of its IPOPT output file), called from
                             `_run_smopf_solver_attempt` after `solver.solve`
  admm_parameters.py         `convergence_depth_tail = {'enabled': False, 'compl_inf_tol': 1e-6}` (default OFF)
  shared_resources_planning  `_capture_convergence_depth_tail_baseline`, `_apply_convergence_depth_tail`,
                             `_convergence_depth_tail_next_state`, `_drain_network_ipopt_solve_records`, and their
                             wiring in `_run_operational_planning`

MODES (repo root, canonical interpreter, attached, both streams captured, noclobber; one at a time):
  --calibrate      reads the committed C* re-certification run's IPOPT logs (s47_recert 070f833e, cycles 78-87 =
                   the certifying streak, the tail would have been ON in 79-87) with the PRODUCTION parser; per solve:
                   exit, iterations, floor status under the production compl_inf_tol, mu_final / tight floor, and the
                   trailing run of IPOPT-"acceptable" iterates at exit. Writes tight_tail_w83/calibration/.
  --freeze-spec    writes data/SRP1/Results/P515S53/frozen_s53_spec_v28_<sha8>.json (write-once, named by its sha256;
                   v27 must be tracked, clean and hash as pinned; the calibration must exist and match its manifest).
  --run            the positive control (a) the switch reads the AA-off predicate, (b) false -> options untouched,
                   (c) true -> compl_inf_tol = 1e-6 on every network holder (and every retry tier), ESSO untouched,
                   the predicate is not permanently false; the production capture path; and the PARSER VALIDATION:
                   the production per-attempt parser reproduces W82's committed per-block terminal figures (2x2
                   x0/unit, SRP1 Phase A x0/unit/unit_dup, s47_recert unit: 352 terminal solves, 15 above floor).
                   Writes tight_tail_w83/checks/.

    set -o noclobber
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w83_tight_tail_checks.py --calibrate \\
        > data/SRP1/Results/P515S53/tight_tail_w83/calibrate_launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w83_tight_tail_checks.py --freeze-spec \\
        > data/SRP1/Results/P515S53/tight_tail_w83/freeze_spec_launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w83_tight_tail_checks.py --run \\
        > data/SRP1/Results/P515S53/tight_tail_w83/checks_launch.log 2>&1
Exit 0 when every check holds, 1 otherwise.
"""

import argparse
import ast
import copy
import hashlib
import inspect
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W83 tight-tail zero-solve checks').install()

import network as NET  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import admm_anderson_acceleration as AA  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402

STAGE = ('P5.15 Addendum 46 ruling 7, W83 -- convergence-depth tight tail: compl_inf_tol 1e-6 on every TSO/DSO '
         'network solve in the certifying cycles, switched by the AA-off predicate; per-solve floor status')
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT = os.path.join(S53, 'tight_tail_w83')
CAL_DIR = os.path.join(OUT, 'calibration')
CAL_JSON = os.path.join(CAL_DIR, 'calibration_w83.json')
CAL_MANIFEST = os.path.join(CAL_DIR, 'calibration_w83_manifest_sha256.json')
CHECKS_DIR = os.path.join(OUT, 'checks')
CHECKS_JSON = os.path.join(CHECKS_DIR, 'checks_w83.json')
CHECKS_MANIFEST = os.path.join(CHECKS_DIR, 'checks_w83_manifest_sha256.json')
SPEC_V27 = {'path': os.path.join(S53, 'frozen_s53_spec_v27_654046ec.json'),
            'sha256': '654046ecc97c485616abffcbd258d4a7c85b7d18531dc0cbd83f424f4e3e846b'}
TAIL_VALUE = 1e-6
ACCEPTABLE_DEFAULTS = {'acceptable_compl_inf_tol': 1e-2, 'acceptable_constr_viol_tol': 1e-2,
                       'acceptable_dual_inf_tol': 1e10,
                       'source': 'IPOPT 3.14.18 option defaults (none of these is set in any SRP1 case file)'}

# The committed C* reference (W32.BASELINE_REFERENCE, restated here so this module does not import the gate chain,
# whose own guard would be installed): s47_recert c_star.
CSTAR = {
    'report': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert', 'evals',
                           '070f833e1e318f85_c_star', 'g_s39_D.json'),
    'sha256': 'ced90e47873925b5bb87360984d348558d11ea11fd004e93f3a2668fdc389971',
    'logs_dir': os.path.join('data', 'SRP1', 'Results', 'P56A', 'evals', 'p515s44_s47_recert_070f833e1e318f85_run',
                             'logs'),
    'candidate_key': '578636daa6d6360d6701764c73ddf795e53c2c37e511e21be1400024f8f6350c',
    'cycles_run': 87, 'certified_cycles': list(range(78, 88)),
}
W82 = {'json': os.path.join(S53, 'gap_closeout_w82', 'gap_closeout_w82.json'),
       'manifest': os.path.join(S53, 'gap_closeout_w82', 'gap_closeout_w82_manifest_sha256.json')}
# (the W82 manifest is committed at 590c298f; the json's entry in it is checked before use)
CASE_PARAMS = {'TSO': os.path.join('data', 'SRP1', 'case9', 'case9_params.json'),
               'DSO5': os.path.join('data', 'SRP1', 'case33_1', 'case33_1_params.json'),
               'DSO7': os.path.join('data', 'SRP1', 'case33_2', 'case33_2_params.json'),
               'DSO9': os.path.join('data', 'SRP1', 'case33_3', 'case33_3_params.json'),
               'ESSO': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')}
PRODUCTION_FILES = ('network.py', 'shared_resources_planning.py', 'admm_parameters.py',
                    'admm_anderson_acceleration.py', 'solver_parameters.py')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W83-checks] {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    h = hashlib.sha256()
    with open(_abs(rel), 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True).stdout.strip()


def _refuse_overwrite(rel):
    if os.path.exists(_abs(rel)):
        raise SystemExit(f'refusing to overwrite existing artifact: {rel}')


def _write_json(rel, payload):
    _refuse_overwrite(rel)
    os.makedirs(os.path.dirname(_abs(rel)), exist_ok=True)
    with open(_abs(rel), 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)


def _write_manifest(rel, paths):
    _write_json(rel, {p: _sha(p) for p in sorted(paths)})


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


# ======================================================================================================================
#  attempt segments of a committed appended log -- each attempt's output begins with IPOPT's options list, which is
#  exactly the byte range production's capture reads ([size before the solve, size after])
# ======================================================================================================================
def attempt_segments(rel):
    with open(_abs(rel), errors='replace') as handle:
        text = handle.read()
    starts = [m.start() for m in re.finditer(re.escape(NET._IPOPT_OPTIONS_HEADER), text)]
    return [text[s:(starts[i + 1] if i + 1 < len(starts) else len(text))] for i, s in enumerate(starts)]


def logged_passed_options(segment):
    """The options IPOPT printed for this attempt (merge-aware, production's own `_ipopt_logged_option`), as the
    'passed' dict production would have had: only the keys the floor formula and its blockers read."""
    block = segment[:segment.find(NET._IPOPT_BANNER)] if NET._IPOPT_BANNER in segment else segment
    out = {}
    for key in ('tol', 'compl_inf_tol', 'acceptable_tol') + NET.IPOPT_MU_FLOOR_FORMULA_BLOCKERS:
        value = NET._ipopt_logged_option(block, key)
        if value is not None:
            out[key] = value if key == 'mu_strategy' else float(value)
    return out


_NLP_VALUES_SPLIT = '***Current NLP Values for Iteration '
_ROW = {name: re.compile(rf'{pat}\.*:\s+(\S+)\s+(\S+)') for name, pat in (
    ('dual', 'Dual infeasibility'), ('constr', 'Constraint violation'), ('compl', 'Complementarity'),
    ('overall', 'Overall NLP error'))}


def iterate_values(segment):
    """Per-iteration (scaled, unscaled) NLP values printed at print level >= 6, in order."""
    out = []
    for chunk in segment.split(_NLP_VALUES_SPLIT)[1:]:
        head = chunk[:chunk.find('\n\n\n')] if '\n\n\n' in chunk else chunk[:2000]
        row = {'iter': int(re.match(r'(\d+)', chunk).group(1))}
        for name, rx in _ROW.items():
            m = rx.search(head)
            row[name] = (float(m.group(1)), float(m.group(2))) if m else None
        out.append(row)
    return out


def acceptable_run_at_exit(values, acceptable_tol):
    """Number of consecutive iterates, ending at the LAST printed one, that meet IPOPT's acceptable criteria
    (scaled overall error <= acceptable_tol; unscaled dual <= 1e10, constraint violation <= 1e-2, complementarity
    <= 1e-2 -- the defaults; the objective-change criterion is at its default 1e20 and never binds)."""
    run = 0
    for row in reversed(values):
        if None in (row['overall'], row['dual'], row['constr'], row['compl']):
            break
        ok = (row['overall'][0] <= acceptable_tol and row['dual'][1] <= ACCEPTABLE_DEFAULTS['acceptable_dual_inf_tol']
              and row['constr'][1] <= ACCEPTABLE_DEFAULTS['acceptable_constr_viol_tol']
              and row['compl'][1] <= ACCEPTABLE_DEFAULTS['acceptable_compl_inf_tol'])
        if not ok:
            break
        run += 1
    return run


# ======================================================================================================================
#  --calibrate
# ======================================================================================================================
def _block_logs(logs_dir):
    case = _load(os.path.join('data', 'SRP1', 'SRP1.json'))
    out = []
    names = [('TSO', case['TransmissionNetwork']['name'])] + [
        (f"DSO{dn['connection_node_id']}", dn['name']) for dn in case['DistributionNetworks']]
    for agent, name in names:
        for year in sorted(case['Years'], key=int):
            for day in case['Days']:
                out.append((agent, name, int(year), day, os.path.join(logs_dir, f'optim_log_{name}_{year}_{day}.log')))
    return out


def calibrate():
    cstar = _load(CSTAR['report'])
    if _sha(CSTAR['report']) != CSTAR['sha256']:
        raise SystemExit('C* reference sha256 != pin')
    rows = cstar['cycle_trajectory']
    conv = {r['cycle']: bool(r.get('cycle_convergence')) for r in rows}
    schedule = [c + 1 for c in sorted(conv) if conv[c] and c + 1 in conv]
    per_solve, logs_sha = [], {}
    for agent, name, year, day, rel in _block_logs(CSTAR['logs_dir']):
        logs_sha[rel] = _sha(rel)
        segs = attempt_segments(rel)
        if len(segs) != CSTAR['cycles_run'] + 1:
            raise SystemExit(f'{rel}: {len(segs)} attempt segments, expected {CSTAR["cycles_run"] + 1} '
                             '(initialisation + one per cycle; recoveries are in separate files)')
        for cycle in CSTAR['certified_cycles']:
            seg = segs[cycle]
            passed = logged_passed_options(seg)
            rec = NET.parse_ipopt_attempt_segment(seg, passed)
            vals = iterate_values(seg)
            s = rec['obj_scaling_factor']
            tight_floor = (min(rec['tol_in_force'], TAIL_VALUE * s) / (NET.IPOPT_DEFAULT_BARRIER_TOL_FACTOR + 1.0)
                           if s else None)
            last = vals[-1] if vals else None
            per_solve.append({
                'agent': agent, 'network': name, 'year': year, 'day': day, 'cycle': cycle,
                'tail_would_be_on': cycle in schedule, 'log': rel, 'segment_index_0based': cycle,
                **{k: rec[k] for k in ('compl_inf_tol_in_force', 'tol_in_force', 'obj_scaling_factor', 'mu_final',
                                       'mu_floor', 'mu_over_floor', 'floor_status', 'iterations', 'exit',
                                       'parse_reason')},
                'n_iterate_value_blocks': len(vals),
                'final_compl_unscaled': last['compl'][1] if last and last['compl'] else None,
                'final_overall_scaled': last['overall'][0] if last and last['overall'] else None,
                'final_compl_meets_tight_cit': (last['compl'][1] <= TAIL_VALUE) if last and last['compl'] else None,
                'tight_floor': tight_floor,
                'mu_final_over_tight_floor': (rec['mu_final'] / tight_floor) if tight_floor and rec['mu_final'] else None,
                'acceptable_run_at_exit': acceptable_run_at_exit(vals, passed.get('acceptable_tol', 1e-6)),
            })
    tail = [p for p in per_solve if p['tail_would_be_on']]

    def summary(items):
        runs = [p['acceptable_run_at_exit'] for p in items]
        hist = {str(k): runs.count(k) for k in sorted(set(runs))}
        return {
            'n': len(items),
            'exits': {e: sum(p['exit'] == e for p in items) for e in sorted({p['exit'] for p in items})},
            'floor_status_production': {s: sum(p['floor_status'] == s for p in items)
                                        for s in sorted({str(p['floor_status']) for p in items})},
            'iterations_median': statistics.median(p['iterations'] for p in items),
            'iterations_max': max(p['iterations'] for p in items),
            'final_compl_meets_tight_cit': sum(bool(p['final_compl_meets_tight_cit']) for p in items),
            'mu_final_over_tight_floor_median': statistics.median(p['mu_final_over_tight_floor'] for p in items),
            'mu_final_over_tight_floor_min': min(p['mu_final_over_tight_floor'] for p in items),
            'acceptable_run_at_exit_histogram': hist,
            'acceptable_run_at_exit_ge_4': sum(r >= 4 for r in runs),
            'acceptable_run_at_exit_ge_5': sum(r >= 5 for r in runs),
            'parse_reasons': sorted({str(p['parse_reason']) for p in items}),
        }
    families = {'TSO': [p for p in tail if p['agent'] == 'TSO'], 'DSO': [p for p in tail if p['agent'] != 'TSO']}
    payload = {
        'stage': STAGE, 'mode': '--calibrate', 'utc': datetime.now(timezone.utc).isoformat(),
        'git_head': _git(['rev-parse', 'HEAD']), 'script_sha256': _sha(os.path.basename(__file__)),
        'network_py_sha256': _sha('network.py'),
        'reference': {**CSTAR, 'certified_at_cycle': cstar.get('cycles_run'),
                      'gross_operational_cost': cstar.get('gross_operational_cost')},
        'derived_tail_schedule_on_the_committed_run': {
            'rule': 'tail ON for cycle k+1 iff cycle_convergence(k) (= the AA-off predicate) -- no predicate precedes cycle 1',
            'cycles_with_predicate_true': sorted(c for c in conv if conv[c]),
            'cycles_tail_would_be_on': schedule,
            'aa_action_on_predicate_cycles': sorted({r.get('aa_action') for r in rows if r.get('cycle_convergence')})},
        'what_is_measured': (
            'the committed production-tolerance solves of the certifying streak (cycles 78-87; the tail would have '
            'been ON in 79-87): exit, iterations, floor status under the production compl_inf_tol (TSO 5e-4 passed, '
            'DSO 1e-4 IPOPT default), mu_final / the tight floor min(tol, 1e-6 s) / 11, whether the final '
            'unscaled complementarity already meets 1e-6, and the trailing run of IPOPT-acceptable iterates at the '
            'exit (acceptable_tol 1e-4 from the options list, other acceptable_* at IPOPT defaults, acceptable_iter '
            '5 in every SRP1 network case file). A run of r at an optimal exit means IPOPT\'s acceptable counter '
            'stood at r - 1 there; under a tighter compl_inf_tol a solve needing m more iterations while staying '
            'acceptable exits "Solved To Acceptable Level" once the counter reaches 5.'),
        'acceptable_defaults_used': ACCEPTABLE_DEFAULTS,
        'summary_tail_cycles': {k: summary(v) for k, v in families.items()},
        'summary_all_certified_cycles': summary(per_solve),
        'per_solve': per_solve,
        'logs_sha256_at_read': logs_sha,
        'guard': dict(GUARD.counts),
    }
    return payload


# ======================================================================================================================
#  --run: the positive control
# ======================================================================================================================
def _load_planning():
    stream = io.StringIO()
    with redirect_stdout(stream):
        planning = srp.SharedResourcesPlanning('data/SRP1', 'SRP1.json')
        planning.read_planning_problem()
    return planning


def _holder_snapshot(planning):
    snap = {}
    for label, nd in srp._convergence_depth_tail_holders(planning):
        opts = nd.params.solver_params.options
        snap[label] = {'id': id(opts), 'items': copy.deepcopy(list(opts.items())) if opts is not None else None}
    esso_opts = planning.shared_ess_data.params.solver_params.options
    snap['ESSO'] = {'id': id(esso_opts), 'items': copy.deepcopy(list(esso_opts.items())) if esso_opts else None}
    return snap


def _retry_override_sets(solver_params):
    """The three attempt tiers' option_overrides EXACTLY as `network._run_smopf` builds them (restated; the source
    check `run_smopf_builds_retry_overrides_as_restated` asserts the construction is the one restated here)."""
    recovery = {k: v for k, v in (solver_params.recovery_options or {}).items() if k != 'hessian_approximation'}
    recovery['warm_start_init_point'] = 'no'
    tier2 = dict(recovery)
    tier2['mu_strategy'] = 'adaptive'
    return {'primary': None, 'recovery': recovery, 'recovery_tier2': tier2}


def _solver_options_for_every_tier(planning):
    out = {}
    for label, nd in srp._convergence_depth_tail_holders(planning):
        network = nd.network[next(iter(nd.years))][next(iter(nd.days))]
        per = {}
        for tier, overrides in _retry_override_sets(nd.params.solver_params).items():
            solver, _log_path, _ctx = NET._create_smopf_solver(network, None, nd.params, from_warm_start=False,
                                                               option_overrides=overrides,
                                                               log_suffix=None if tier == 'primary' else tier)
            per[tier] = dict(solver.options).get('compl_inf_tol')
        out[label] = per
    return out


def _run_source():
    return inspect.getsource(srp._run_operational_planning)


def _assign_values(func_src, name):
    tree = ast.parse(func_src)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    found.append(ast.unparse(node.value))
    return found


TAIL_FUNCTIONS = ('_capture_convergence_depth_tail_baseline', '_apply_convergence_depth_tail',
                  '_convergence_depth_tail_next_state')


def _tail_calls_guarded(run_src):
    """AST: every call of a tail function in `_run_operational_planning` sits inside an `if convergence_depth_tail_on:`
    block (and there is at least one call of each)."""
    tree = ast.parse(run_src)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    seen = {name: 0 for name in TAIL_FUNCTIONS}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in seen:
            seen[node.func.id] += 1
            up, guarded = node, False
            while up in parents:
                up = parents[up]
                if isinstance(up, ast.If) and isinstance(up.test, ast.Name) and up.test.id == 'convergence_depth_tail_on':
                    guarded = True
                    break
            if not guarded:
                return False
    return all(n >= 1 for n in seen.values())


def check_a_predicate():
    """(a) the switch reads the AA-off predicate."""
    run_src = _run_source()
    step_src = inspect.getsource(srp._anderson_acceleration_cycle_step)
    tree_assigns = _assign_values(run_src, 'convergence_depth_tail_active_next')
    pos = lambda s: run_src.find(s)  # noqa: E731
    static = {
        'boyd_all_pass_is_read_from_boyd_metrics': "boyd_all_pass = boyd_metrics['all_boyd_pass']" in run_src,
        'cycle_convergence_is_boyd_all_pass_and_local_solves_ok':
            'cycle_convergence = boyd_all_pass and local_solves_ok' in run_src,
        'aa_step_is_handed_the_same_boyd_metrics': (
            '_anderson_acceleration_cycle_step(' in run_src
            and "boyd_all_pass = boyd_metrics['all_boyd_pass']" in step_src
            and 'aa_state.step(iter, w_before, g, combined_residual, boyd_all_pass)' in step_src),
        'aa_step_called_only_when_local_solves_ok': _ordered(
            run_src, 'if aa_enabled:\n            if local_solves_ok:', '_anderson_acceleration_cycle_step(',
            'aa_record = aa_state.skip_on_failure(iter)'),
        'tail_next_state_assignments_are_exactly_init_false_and_the_predicate': (
            sorted(tree_assigns) == sorted(['False',
                                            '_convergence_depth_tail_next_state(cycle_convergence, aa_enabled, aa_record)'])),
        'tail_next_state_read_after_cycle_convergence_and_after_the_aa_step': (
            0 <= pos('_anderson_acceleration_cycle_step(') < pos('cycle_convergence = boyd_all_pass and local_solves_ok')
            < pos('convergence_depth_tail_active_next = _convergence_depth_tail_next_state(')),
        'apply_is_called_with_the_previous_cycles_predicate': (
            '_apply_convergence_depth_tail(\n                planning_problem, admm_parameters, '
            'convergence_depth_tail_active_next,' in run_src),
        'apply_precedes_the_dso_solve_in_the_cycle': (
            pos('tail_cycle_record = _apply_convergence_depth_tail(') < pos("results['dso'] = update_distribution_coordination_models_and_solve(")
            and pos("for iter in range(1, admm_parameters.num_max_iters + 1):") < pos('tail_cycle_record = _apply_convergence_depth_tail(')),
        'tail_starts_off_each_call': 'convergence_depth_tail_active_next = False' in run_src,
        'every_tail_call_is_behind_the_flag': _tail_calls_guarded(run_src),
        'aa_module_off_literal_equals_production_constant': (
            f"action='{srp.CONVERGENCE_DEPTH_TAIL_AA_OFF_ACTION}'," in inspect.getsource(AA.AndersonAccelerationState.step)),
    }
    # dynamic, with the REAL AA state machine (keep_memory, the case-file policy)
    import numpy as np
    dyn = {}
    st = AA.AndersonAccelerationState(memory=5, regularization=1e-10, reject_policy='keep_memory')
    w, g = np.zeros(3), np.ones(3) * 1e-3
    _w, rec_true = st.step(1, w, g, 1.0, True)
    _w, rec_false = st.step(2, w + g, g * 0.5, 0.5, False)
    rec_skip = st.skip_on_failure(3)
    dyn['aa_step_true_action'] = rec_true['action']
    dyn['aa_step_false_action'] = rec_false['action']
    dyn['aa_step_true_is_the_off_constant'] = rec_true['action'] == srp.CONVERGENCE_DEPTH_TAIL_AA_OFF_ACTION
    dyn['aa_step_false_is_not_off'] = rec_false['action'] != srp.CONVERGENCE_DEPTH_TAIL_AA_OFF_ACTION
    dyn['next_state_true_with_aa_off'] = srp._convergence_depth_tail_next_state(True, True, rec_true) is True
    dyn['next_state_false_with_aa_not_off'] = srp._convergence_depth_tail_next_state(False, True, rec_false) is False
    dyn['next_state_false_after_local_failure_skip'] = srp._convergence_depth_tail_next_state(False, True, rec_skip) is False
    dyn['next_state_follows_predicate_with_aa_disabled'] = (
        srp._convergence_depth_tail_next_state(True, False, None) is True
        and srp._convergence_depth_tail_next_state(False, False, None) is False)
    mismatch = {}
    for label, args in (('true_vs_aa_not_off', (True, True, rec_false)), ('false_vs_aa_off', (False, True, rec_true))):
        try:
            srp._convergence_depth_tail_next_state(*args)
            mismatch[label] = 'did not raise'
        except RuntimeError as error:
            mismatch[label] = f'raised: {error}'
    dyn['disagreement_raises'] = all(v.startswith('raised') for v in mismatch.values())
    dyn['disagreement_messages'] = mismatch
    # the predicate is not permanently false: the committed C* run
    if _sha(CSTAR['report']) != CSTAR['sha256']:
        raise SystemExit('C* reference sha256 != pin')
    rows = _load(CSTAR['report'])['cycle_trajectory']
    true_cycles = [r['cycle'] for r in rows if r.get('cycle_convergence')]
    consistent = all((r.get('aa_action') == srp.CONVERGENCE_DEPTH_TAIL_AA_OFF_ACTION)
                     == bool(r.get('boyd_all_pass') and r.get('local_solves_ok')) == bool(r.get('cycle_convergence'))
                     for r in rows)
    not_permanently_false = {
        'reference': CSTAR['report'], 'reference_sha256': CSTAR['sha256'],
        'cycles_with_predicate_true': true_cycles,
        'n_true': len(true_cycles),
        'tail_would_be_on_in_cycles': [c + 1 for c in true_cycles if c + 1 <= len(rows)],
        'aa_off_iff_boyd_all_pass_and_local_ok_iff_cycle_convergence_on_every_row': consistent,
        'holds': len(true_cycles) >= 1 and consistent,
    }
    ok = all(static.values()) and all(v for k, v in dyn.items() if isinstance(v, bool)) and not_permanently_false['holds']
    return {'static': static, 'dynamic_with_the_real_aa_state_machine': dyn,
            'predicate_not_permanently_false': not_permanently_false, 'holds': ok}


def _ordered(text, *needles):
    positions = [text.find(n) for n in needles]
    return all(p >= 0 for p in positions) and positions == sorted(positions)


def check_b_c_holders(planning):
    """(b) false -> untouched; (c) true -> 1e-6 on every network holder and every retry tier; ESSO untouched;
    restore exact; baseline refusal on a recovery_options override; persistent-worker refusal."""
    admm = copy.deepcopy(planning.params.admm)
    admm.convergence_depth_tail = {'enabled': True, 'compl_inf_tol': TAIL_VALUE}
    labels = [label for label, _nd in srp._convergence_depth_tail_holders(planning)]
    case_files = {k: _load(v)['solver'] for k, v in CASE_PARAMS.items()}
    pre = _holder_snapshot(planning)
    baseline = srp._capture_convergence_depth_tail_baseline(planning, admm)
    tiers_before = _solver_options_for_every_tier(planning)
    out = {'holders': labels, 'baseline': baseline, 'solver_options_compl_inf_tol_before_by_tier': tiers_before}

    # (b)
    rec_false = srp._apply_convergence_depth_tail(planning, admm, False, baseline, 1)
    after_false = _holder_snapshot(planning)
    out['b'] = {
        'record': rec_false,
        'acted_false': rec_false['acted'] is False and not any(h['wrote'] for h in rec_false['holders'].values()),
        'options_identical_objects_items_and_order': after_false == pre,
        'baseline_equals_case_files': all(
            baseline[k]['has_key'] == ('compl_inf_tol' in case_files[k]['options'])
            and baseline[k]['value'] == case_files[k]['options'].get('compl_inf_tol') for k in labels),
        'baseline_tso_5e-4_dso_absent': (baseline['TSO']['value'] == 5e-4 and baseline['TSO']['has_key']
                                         and all(not baseline[k]['has_key'] for k in labels if k != 'TSO')),
    }
    out['b']['holds'] = all(v for k, v in out['b'].items() if isinstance(v, bool))

    # (c)
    rec_true = srp._apply_convergence_depth_tail(planning, admm, True, baseline, 2)
    after_true = _holder_snapshot(planning)
    tiers_true = _solver_options_for_every_tier(planning)
    out['c'] = {
        'record': rec_true,
        'every_network_holder_at_1e-6': all(
            dict(after_true[k]['items']).get('compl_inf_tol') == TAIL_VALUE for k in labels),
        'every_holder_wrote': all(rec_true['holders'][k]['wrote'] for k in labels),
        'every_tier_passes_1e-6_to_ipopt': all(v == TAIL_VALUE for per in tiers_true.values() for v in per.values()),
        'tiers': tiers_true,
        'esso_untouched': after_true['ESSO'] == pre['ESSO'],
        'only_compl_inf_tol_changed': all(
            {k2: v2 for k2, v2 in after_true[k]['items'] if k2 != 'compl_inf_tol'}
            == {k2: v2 for k2, v2 in pre[k]['items'] if k2 != 'compl_inf_tol'} for k in labels),
        'idempotent_second_true_writes_nothing': (
            srp._apply_convergence_depth_tail(planning, admm, True, baseline, 3)['acted'] is False),
    }
    out['c']['holds'] = all(v for k, v in out['c'].items() if isinstance(v, bool))

    # restore
    rec_restore = srp._apply_convergence_depth_tail(planning, admm, False, baseline, None)
    after_restore = _holder_snapshot(planning)
    tiers_restored = _solver_options_for_every_tier(planning)
    out['restore'] = {
        'record': rec_restore,
        'every_holder_wrote_once': all(rec_restore['holders'][k]['wrote'] for k in labels),
        'options_identical_objects_items_and_order': after_restore == pre,
        'tiers_back_to_production': tiers_restored == tiers_before,
        'second_restore_writes_nothing': (
            srp._apply_convergence_depth_tail(planning, admm, False, baseline, None)['acted'] is False),
    }
    out['restore']['holds'] = all(v for k, v in out['restore'].items() if isinstance(v, bool))

    # retry inheritance, by source and by the refusal
    smopf_src = inspect.getsource(NET._run_smopf)
    create_src = inspect.getsource(NET._create_smopf_solver)
    tso_sp = planning.transmission_network.params.solver_params
    saved = tso_sp.recovery_options
    try:
        tso_sp.recovery_options = dict(saved or {}, compl_inf_tol=1e-4)
        try:
            srp._capture_convergence_depth_tail_baseline(planning, admm)
            refusal = 'did not raise'
        except ValueError as error:
            refusal = f'raised: {error}'
    finally:
        tso_sp.recovery_options = saved
    out['retry_inheritance'] = {
        'create_smopf_solver_merges_holder_options_before_overrides': _ordered(
            create_src, 'options.update(solver_params.options)', 'options.update(option_overrides)'),
        'run_smopf_builds_retry_overrides_as_restated': (
            "if key != 'hessian_approximation'" in smopf_src
            and "recovery_options['warm_start_init_point'] = 'no'" in smopf_src
            and 'tier2_options = dict(recovery_options)' in smopf_src
            and "tier2_options['mu_strategy'] = 'adaptive'" in smopf_src),
        'no_case_file_recovery_options_sets_compl_inf_tol': all(
            'compl_inf_tol' not in (case_files[k].get('recovery_options') or {}) for k in labels),
        'baseline_refuses_a_recovery_override_of_compl_inf_tol': refusal.startswith('raised'),
        'refusal_message': refusal,
        'every_tier_passes_the_holder_value_production_and_tail': (
            all(len(set(per.values())) == 1 for per in tiers_before.values())
            and all(len(set(per.values())) == 1 for per in tiers_true.values())),
        'tiers_production': tiers_before,
    }
    out['retry_inheritance']['holds'] = all(v for k, v in out['retry_inheritance'].items() if isinstance(v, bool))

    # ESSO: not a network solve, never a holder
    out['esso'] = {
        'not_among_the_holders': 'ESSO' not in labels and all(
            nd is not planning.shared_ess_data for _l, nd in srp._convergence_depth_tail_holders(planning)),
        'esso_options_unchanged_through_b_c_restore': after_restore['ESSO'] == pre['ESSO'] == after_true['ESSO'],
        'esso_case_file_options': case_files['ESSO']['options'],
    }
    out['esso']['holds'] = all(v for k, v in out['esso'].items() if isinstance(v, bool))
    out['persistent_workers_refused'] = (
        "raise ValueError('convergence_depth_tail is not supported together with persistent_workers.')"
        in _run_source())
    out['holds'] = all(out[k]['holds'] for k in ('b', 'c', 'restore', 'retry_inheritance', 'esso')) and \
        out['persistent_workers_refused']
    return out


def check_capture_path():
    attempt_src = inspect.getsource(NET._run_smopf_solver_attempt)
    run_src = _run_source()
    net_init = inspect.getsource(NET.Network.__init__)
    checks = {
        'solve_stays_inside_the_guarded_call_site': 'result = solver.solve(model' in attempt_src,
        'offset_taken_before_the_solve_record_appended_after': _ordered(
            attempt_src, 'log_offset = _ipopt_log_size(solver_log_path)', 'result = solver.solve(model',
            '_append_ipopt_solve_record(network, params, solver, solver_log_path, log_offset, log_suffix, from_warm_start)'),
        'network_init_creates_a_bounded_deque': 'self.ipopt_solve_records = deque(maxlen=IPOPT_SOLVE_RECORDS_MAXLEN)' in net_init,
        'initialisation_round_drained_as_round_0': (
            "network_ipopt_solve_records.extend(_drain_network_ipopt_solve_records(planning_problem, 0))" in run_src),
        'every_cycle_drained_unconditionally': (
            "network_ipopt_solve_records.extend(_drain_network_ipopt_solve_records(planning_problem, iter))" in run_src),
        'state_carries_records_and_tail_state': (
            run_src.count("'network_ipopt_solve_records': network_ipopt_solve_records,") == 2
            and run_src.count("'convergence_depth_tail': convergence_depth_tail_state,") == 2),
        'default_admm_parameters_tail_off': ADMMParameters().convergence_depth_tail == {'enabled': False,
                                                                                      'compl_inf_tol': 1e-6},
        'enabled_helper_false_on_default_and_on_a_legacy_object': (
            srp.convergence_depth_tail_enabled(ADMMParameters()) is False
            and srp.convergence_depth_tail_enabled(object()) is False),
        'aa_module_unchanged_in_git': _git(['status', '--porcelain', '--', 'admm_anderson_acceleration.py']) == '',
    }
    return {'checks': checks, 'holds': all(checks.values())}


# ======================================================================================================================
#  --run: parser validation against W82's committed per-block figures
# ======================================================================================================================
def _w82_block_sets(w82):
    two = w82['item1_item2_2x2']['cells']
    pa = w82['srp1_phase_a_rederived']['cells']
    return {'2x2:x0_a0p50': two['x0']['blocks'], '2x2:n7_4h_e1_a0p50': two['unit']['blocks'],
            'phaseA:x0': pa['x0']['blocks'], 'phaseA:unit': pa['unit']['blocks'],
            'phaseA:unit_dup': pa['unit_dup']['blocks'],
            's47_recert:n7_4h_e1': w82['item3_srp1_c2_baseline']['unit']['blocks']}


def validate_parser_against_w82():
    man = _load(W82['manifest'])
    if man.get(W82['json']) != _sha(W82['json']):
        raise SystemExit('W82 json does not hash to its committed manifest')
    w82 = _load(W82['json'])
    sets = _w82_block_sets(w82)
    per_set, mismatches, n_total = {}, [], 0
    sha_cache = {}
    for set_label, blocks in sets.items():
        counts = {'n': 0, 'at': 0, 'above': 0, 'below': 0, 'fields_equal': 0, 'log_sha_equal': 0}
        above_tso = 0
        for key, b in blocks.items():
            n_total += 1
            counts['n'] += 1
            rel = b['log']
            sha = sha_cache.get(rel) or _sha(rel)
            sha_cache[rel] = sha
            counts['log_sha_equal'] += int(sha == b['sha256_at_read'])
            segs = attempt_segments(rel)
            t = b['terminal_admm_solve']
            k = b['terminal_index_1based']
            if len(segs) != b['n_solves']:
                mismatches.append({'set': set_label, 'block': key, 'why': f'{len(segs)} segments vs {b["n_solves"]}'})
                continue
            seg = segs[k - 1]
            rec = NET.parse_ipopt_attempt_segment(seg, logged_passed_options(seg))
            pairs = {'obj_scaling_factor': (rec['obj_scaling_factor'], t['obj_scale']),
                     'mu_final': (rec['mu_final'], t['mu_last']),
                     'iterations': (rec['iterations'], t['iterations']),
                     'exit': (rec['exit'], t['exit']),
                     'compl_inf_tol_in_force': (rec['compl_inf_tol_in_force'], b['compl_inf_tol']),
                     'tol_in_force': (rec['tol_in_force'], b['tol']),
                     'mu_floor': (rec['mu_floor'], b['mu_floor']),
                     'mu_over_floor': (rec['mu_over_floor'], b['mu_over_floor']),
                     'floor_status': (rec['floor_status'], b['floor_status'])}
            bad = {f: v for f, v in pairs.items() if v[0] != v[1]}
            if bad or rec['parse_reason'] is not None:
                mismatches.append({'set': set_label, 'block': key, 'bad': bad, 'parse_reason': rec['parse_reason']})
            else:
                counts['fields_equal'] += 1
            counts[str(rec['floor_status'])] = counts.get(str(rec['floor_status']), 0) + 1
            if key.startswith('TSO') and rec['floor_status'] == 'above':
                above_tso += 1
        counts['tso_above'] = above_tso
        per_set[set_label] = counts
    n_at = sum(v['at'] for v in per_set.values())
    known = {
        'x0_a0p50_tso_above_is_15_of_20': per_set['2x2:x0_a0p50']['tso_above'] == 15,
        'every_other_terminal_solve_at_floor_337_of_352': n_total == 352 and n_at == 337,
        '2x2_145_of_160_at_floor': per_set['2x2:x0_a0p50']['at'] + per_set['2x2:n7_4h_e1_a0p50']['at'] == 145,
        'phase_a_144_of_144': sum(per_set[s]['at'] for s in ('phaseA:x0', 'phaseA:unit', 'phaseA:unit_dup')) == 144,
        's47_recert_48_of_48': per_set['s47_recert:n7_4h_e1']['at'] == 48,
    }
    return {'source': W82['json'], 'source_sha256': _sha(W82['json']),
            'rule': ('each committed log is split at IPOPT\'s "List of options:" header (each attempt\'s output begins '
                     'with it -- the byte range production reads); the terminal ADMM segment (W82 '
                     'terminal_index_1based) is parsed by network.parse_ipopt_attempt_segment with the options printed '
                     'in that segment; every field is compared for EXACT equality with W82\'s committed block record'),
            'n_blocks': n_total, 'per_set': per_set, 'n_mismatches': len(mismatches), 'mismatches': mismatches[:50],
            'known_figures_reproduced': known,
            'holds': not mismatches and all(known.values())}


# ======================================================================================================================
#  --freeze-spec
# ======================================================================================================================
def build_spec(cal_rel, cal_sha, cal_manifest_sha):
    cal = _load(cal_rel)
    return {
        'schema': 'p515_frozen_spec_v28', 'version': 28,
        'stage': STAGE,
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 46 (ruling 7: convergence depth, minimal (c))',
            'Planner task W83 (implement, gate, STOP; ruling on the 1e-4 / 5e-4 wording)'],
        'predecessor': SPEC_V27,
        'predecessor_not_edited': 'v27 stays as frozen; v28 records the W83 tight tail and adds nothing to v27\'s tests',
        'ruling_verbatim': (
            '"(7) Convergence depth: option (c) in its minimal form -- compl_inf_tol 1e-4 -> 1e-6 only in the '
            'certifying cycles, switched by the same all-channels-inside-tolerance condition that turns AA off; '
            'everything before the tail bitwise unchanged; per-solve floor status reported for every cell from now on."'),
        'brief_addendum_46_text': (
            'Convergence depth (item 7): minimal (c). compl_inf_tol 1e-4 -> 1e-6 only in the certifying cycles, '
            'switched by the all-channels-inside-tolerance condition that turns AA off; all cycles before the tail '
            'bitwise unchanged; per-solve floor status reported for every cell from now on.'),
        'wording_discrepancy_and_planner_ruling': {
            'discrepancy': ('The ruling names 1e-4, which is IPOPT\'s DEFAULT: the DSO blocks inherit it because '
                            'compl_inf_tol is absent from their options (case33_1/2/3_params.json). The TSO sets it '
                            'explicitly to 5e-4 (case9_params.json), and all fifteen 2x2 early-stopping solves (W82) '
                            'were TSO solves. A change applied only where the value is literally 1e-4 would tighten '
                            'DSO blocks that already reach their floor 120/120 and leave the actual problem untouched.'),
            'planner_ruling': ('Implement the intent: compl_inf_tol = 1e-6 on ALL network solves (TSO and every DSO) '
                               'in the certifying cycles -- TSO from 5e-4, DSO from the 1e-4 default. Recorded here '
                               'so the author can overrule it.'),
            'as_implemented': 'ADMMParameters.convergence_depth_tail = {enabled, compl_inf_tol: 1e-6}; every TSO/DSO holder',
        },
        'author_prediction_verbatim': (
            'Prediction: SRP1 references (C*, x = 0, smallest unit) unchanged or within 1e-6 relative (SRP1 already '
            'at the floor 192/192); at 2 x 2 the fifteen early stops would reach the floor.'),
        'implementation': {
            'switch': {
                'predicate': ("the AA-off predicate: `cycle_convergence = boyd_all_pass and local_solves_ok` with "
                              "`boyd_all_pass = boyd_metrics['all_boyd_pass']` (shared_resources_planning."
                              "_run_operational_planning); the SAME boyd_metrics['all_boyd_pass'] is what "
                              "_anderson_acceleration_cycle_step hands to AndersonAccelerationState.step (called only "
                              "when local_solves_ok), whose action is 'off (all channels within Boyd tolerance)' "
                              "exactly then; it is also the certificate counter's condition"),
                'read_not_rederived': ('_convergence_depth_tail_next_state(cycle_convergence, aa_enabled, aa_record) '
                                       'returns cycle_convergence itself and raises if AA\'s own record for the cycle '
                                       'disagrees (AA enabled)'),
                'timing': ('non-latching, mirroring AA: tail ON for cycle k+1 iff the predicate held at the end of '
                           'cycle k; OFF at cycle 1 of every call; holders restored at loop exit. The first cycle of a '
                           'converged streak runs at production tolerance, cycles 2..10 tight (the certified '
                           'terminal cycle is tight whenever the streak is >= 2)'),
                'default': 'OFF; programmatic only (no case-file key); enabled per run by the harness',
            },
            'what_changes_when_on': ('only the compl_inf_tol key of solver_params.options on the TSO holder and each '
                                     'DSO holder (NetworkData.params.solver_params); nothing else'),
            'retry_tiers': ('inherit: network._create_smopf_solver merges solver_params.options BEFORE the retry '
                            'option_overrides; no SRP1 recovery_options sets compl_inf_tol; the baseline capture '
                            'REFUSES to run if one does (asserted by --run, functionally on all three tiers)'),
            'esso': ('NOT touched: the ESSO is not a network solve (the ruling and the task name network solves), its '
                     'objective carries the oracle\'s own D5 scaling (sigma fixed, S_ref 2.5 MVA) and is outside Q; '
                     'its options (tol 1e-6, compl_inf_tol at the IPOPT default) are recorded before/after and must '
                     'be unchanged -- the scaling-pin test\'s treatment (spec v27 configuration.esso)'),
            'persistent_workers': 'refused with the tail enabled (workers hold their own holder copies)',
            'parallel_execution': ('the holder change reaches pickled DSO tasks, but per-attempt records from worker '
                                   'processes are not returned; every campaign runs parallel_execution off'),
        },
        'floor_status_capture': {
            'where': ('production: network._run_smopf_solver_attempt -> _append_ipopt_solve_record (every TSO/DSO '
                      'IPOPT attempt: primary, recovery, recovery_tier2), drained per round by '
                      'shared_resources_planning._drain_network_ipopt_solve_records into '
                      "state['network_ipopt_solve_records'] (round 0 = initialisation, k = cycle k); per-cycle tail "
                      "state in state['convergence_depth_tail']"),
            'fields': ['network', 'year', 'day', 'agent', 'round', 'attempt', 'warm_start', 'log_path', 'log_bytes',
                       'compl_inf_tol_passed', 'compl_inf_tol_in_force', 'compl_inf_tol_logged', 'options_list_agrees',
                       'tol_in_force', 'mu_strategy_passed', 'obj_scaling_factor', 'mu_final', 'mu_floor',
                       'mu_over_floor', 'floor_status', 'iterations', 'exit', 'parse_reason'],
            'formulas': {
                'segment': "the attempt's own byte range [size before solve, size after] of its appended output_file",
                'compl_inf_tol_in_force': 'value passed to IPOPT (solver.options) else the IPOPT default 1e-4',
                'tol_in_force': 'value passed else the IPOPT default 1e-8',
                'obj_scaling_factor': "FIRST 'objective scaling factor = S' line of the segment",
                'mu_final': "LAST 'Current barrier parameter mu = X' line of the segment",
                'mu_floor': ('min(tol, compl_inf_tol x S) / (barrier_tol_factor + 1), barrier_tol_factor 10; None '
                             '(reason recorded) when mu_strategy / barrier_tol_factor / mu_target / mu_min is passed'),
                'floor_status': "'at' iff |mu_final/mu_floor - 1| <= 1e-3; 'above' iff > 1 + 1e-3; else 'below'",
            },
            'validation_rule': ('Addendum 46 standing rule (a log-derived quantity is validated by reproducing a known '
                                'figure before it drives a decision): --run must reproduce W82\'s committed terminal '
                                'figures exactly (352 blocks; 15 of 20 x0_a0p50 TSO above floor; 337 of 352 at floor)'),
            'persistence_note': ('production keeps the records in the returned state; writing them into every eval '
                                 'directory is a harness step (the gate below writes its own sidecar; the campaign '
                                 'harness does not yet -- a question for the Planner)'),
        },
        'gate': {
            'script': 'p515_s53_w83_tight_tail_gate.py',
            'kind': ('SRP1 two-cycle bitwise gate, built on the committed r2 gate (p515_s53_srp1_bitwise_gate.py, '
                     'd5a00bd9) by import, arm with the tail ENABLED'),
            'arm': 's53w83tailgate (distinct: the eval dir under P56A/evals is arm-named)',
            'output_root': os.path.join(OUT, 'srp1_bitwise_gate'),
            'launch_log': os.path.join(OUT, 'srp1_bitwise_gate_launch.log'),
            'instance': 'C* 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, 2025; candidate_key ' + CSTAR['candidate_key'],
            'configuration': 'the r2 configuration (baseline, cap 2, snapshots on, apply_rho False) + tail enabled',
            'solve_profile': ('declared before the run: 51 per cycle x (cap 2 + 1) = 153 base; observed == 153 + every '
                              'attempted retry (per event); W10.GUARD.verify(that) EXACTLY'),
            'items': [
                'every committed r2 gate item (W35: 2 cycles, 0 diffs vs committed C* rows[:2] + derived fields, 0 '
                'field mismatches, event-level solve identity, GUARD.verify exact, no blocked call, FrozenSMOPF '
                'untouched, ESS-ageing declaration; W39 tripwire 0 calls; W51 counter 36/36 each, 0 acting)',
                '0 genuine diffs vs the committed r2 arm (P515S53/srp1_bitwise_gate/arm)',
                'tail enabled and read back in the arm; tail apply counter EXACTLY cap + 1 = 3 calls (cycles 1, 2, '
                'exit restore), every one active False and acted False; next-state counter EXACTLY 2 calls, both '
                'returning False',
                'returned state: per_cycle 2 records, aa_off_predicate_end_of_cycle False on both, restore_at_exit '
                'acted False; holders after the arm identical to before (items and order); ESSO options unchanged',
                'floor capture complete: records == observed solves - 9 ESSO solves, every parse_reason None, '
                'compl_inf_tol_in_force == production on every record (TSO 5e-4 passed and logged; DSO 1e-4 '
                'default, absent from the list), options_list_agrees on every record',
                'no "Convergence-depth tail ON" line in the arm stdout',
                'W83 code present in the modules that ran (before and after the arm)'],
            'applies_to': 'the single tail-enabled arm; there is no other arm',
        },
        'positive_control_zero_solve': {
            'script': os.path.basename(__file__) + ' --run',
            'a': 'the switch reads the AA-off predicate (source/AST + the real AA state machine; disagreement raises)',
            'b': 'predicate false -> no holder written, options identical (object, items, order), ESSO unchanged',
            'c': 'predicate true -> compl_inf_tol 1e-6 on every TSO/DSO holder and on all three retry tiers; ESSO unchanged',
            'not_permanently_false': 'committed C* rows: predicate true on cycles 78-87 (AA off there), tail would be ON 79-87',
        },
        'calibration_zero_solve': {'path': cal_rel, 'sha256': cal_sha, 'manifest_sha256': cal_manifest_sha,
                                   'summary_tail_cycles': cal['summary_tail_cycles'],
                                   'derived_tail_schedule': cal['derived_tail_schedule_on_the_committed_run']},
        'fallback_rule_operational': {
            'cells': {
                'C*': {'reference': 'P515S47/campaign_s47_recert c_star (eval 070f833e), certified cycle 87',
                       'Q_ref_gross': 650912327.1956586, 'bar_ref': 25146.48616361618},
                'x0': {'reference': 'P515S45/campaign_s45_a0_c7 x0 (eval 7aa017f0), certified cycle 132 (the pinned Q(0))',
                       'Q_ref_gross': 653859461.2279255, 'bar_ref': 9629.97833788395},
                'unit n7_4h_e1': {'reference': 'P515S47/campaign_s47_recert n7_4h_e1 (eval bd504ecf), certified cycle 112',
                                  'Q_ref_gross': 653600033.4601703, 'bar_ref': 25104.654562950134},
            },
            'objective_convention': 'gross_operational_cost (settlement-excluded gross; equal to net recourse at these cells)',
            'fails_to_certify': ('TEST: the tail-enabled run of the cell ends with status != "certified" under the '
                                 'UNCHANGED certification criterion (all three Boyd channels inside tolerance and every '
                                 'local solve successful for 10 consecutive cycles) within the cap of that cell\'s '
                                 'reference campaign; a harness error counts as a failure to certify'),
            'moves_materially': ('TEST: |Q_tail - Q_ref| > bar_ref(cell), the reference\'s own bar (max |step| of the '
                                 'gross cost over its last 10 cycles). REPORTED beside it, never replacing it: '
                                 '|dQ| / (bar_ref + bar_tail) (the two-run resolution; below 1 the difference is '
                                 'indeterminate, CLAUDE.md), |dQ| / (1e-6 |Q_ref|) (the author\'s band, ~650 EUR, '
                                 '~39x below the smallest bar), and the tail run\'s rule-ten terminal step / threshold'),
            'consequence': ('if ANY of the three cells fails to certify or moves materially: the fallback -- the '
                            'unchanged configuration (tail disabled) with the depth caveat recorded (the Planner\'s '
                            '8 reading); otherwise the tail-enabled references replace them for what follows'),
        },
        'predictions_recorded_before_any_run': {
            'author': 'see author_prediction_verbatim',
            'worker': {
                'G1_gate': ('the tail-enabled two-cycle arm is bit-identical to the committed C* rows (0 diffs) and '
                            'has 0 genuine diffs vs the r2 arm; 153/153 solves; tail counter 3/3 inactive, 0 acted'),
                'G2_floor_capture': ('144/144 network records parse; compl_inf_tol_in_force TSO 5e-4 / DSO 1e-4 on '
                                     'every record; floor status NOT predicted as all-at at cycles 0-2 (early, cold '
                                     'ADMM rounds) -- reported, not gated'),
                'P1_certifies': 'each of the three SRP1 cells certifies with the tail (probability 0.85)',
                'P2_cert_cycle': 'certification cycle within +3 of the reference for each cell (0.6)',
                'P3_systematic_shift': ('the at-floor barrier-gap estimate (W82: SRP1 cells G ~ 3,387-3,466 EUR, TSO '
                                        '~950-1,020 at cit 5e-4, DSO ~2,435-2,450 at 1e-4) falls by factors 500 (TSO) '
                                        'and 100 (DSO), i.e. a systematic dQ ~ -3.4e3 EUR per cell (~5e-6 relative) on '
                                        'top of path/stopping differences of the order of the bar'),
                'P4_author_band': ('|dQ| > 1e-6 |Q_ref| (~650 EUR) on each of the three cells -- the author\'s band '
                                   'is REFUTED (0.8)'),
                'P5_material': '|dQ| < bar_ref on each cell -- NOT material, no fallback (0.75)',
                'P6_R': 'R = Q(0) - Q(unit) changes by less than the two-cell bar-sum 34,734.63 (0.9)',
                'P7_acceptable_exits': ('some tight-cycle solves could exit "Solved To Acceptable Level" above the '
                                        'tight floor, because acceptable_iter 5 / acceptable_tol 1e-4 / '
                                        'acceptable_compl_inf_tol 1e-2 are unchanged by the ruling. The calibration '
                                        'shows every one of the 432 reference solves of cycles 79-87 exiting with a '
                                        'trailing acceptable run of only 0-2 iterates (counter 0-1 at the optimal '
                                        'exit), and the tail needs ONE further barrier update (mu^1.5 undershoots the '
                                        'tight floor: TSO 4.5e-8 -> 9.1e-11, DSO 9.1e-9 -> 9.1e-11) plus a few Newton '
                                        'steps, so an acceptable exit needs >= 4 more consecutive acceptable iterates. '
                                        'Probability that >= 1 tight solve in a cell exits acceptable: 0.35'),
                'P8_iterations': ('median iterations of tight-cycle solves +1 to +6 over the reference\'s same cycles '
                                  '(reference medians: TSO 30, DSO 44) (0.6)'),
                'P10_retries': ('network retries in the tight cycles at most twice the reference\'s in the same '
                                'window (the reference already has 2 DSO max_iter primaries in cycles 79-87, recovered) '
                                '(0.7); every retry attempt in a tight cycle passes 1e-6 (certain by construction; '
                                'the per-attempt record shows it)'),
                'P9_2x2': ('at 2x2 (NOT run in this task) the fifteen x0 TSO early stops, which ended "Optimal" with '
                           'unscaled complementarity ~2.6e-4 against 5e-4, can no longer stop there; each either '
                           'reaches the tight floor or exits acceptable -- the author\'s "would reach the floor" is '
                           'CONFIRMED only for the former; probability all fifteen reach the floor: 0.5'),
            },
            'seen_before_these_predictions': [
                'calibration_w83.json and calibrate_launch.log (this spec\'s calibration_zero_solve, run at '
                'a51ad9ba; its summary is embedded above) -- seen BEFORE P7, P8 and P10 were written; P1-P6 and P9 '
                'were written before it',
                'W81/W82 committed outputs (SRP1 G, 2x2 early stops and their exits/complementarity)',
                's47_recert campaign_results.json (Q and bars of C* and the unit), W82 x0 pin (Q(0), bar)',
                'the committed r2 gate outputs and the committed C* trajectory (predicate on cycles 78-87)'],
        },
        'not_permitted': ['the SRP1 re-certification (C*, x = 0, smallest unit) is NOT run in W83',
                          'no change to the ADMM formulation, the certification criterion, the AA predicate or '
                          'anything outside the solver-option tail',
                          'no committed artifact modified or re-run onto; distinct arm name and fresh launch logs'],
        'production_sha256_at_freeze': {f: _sha(f) for f in PRODUCTION_FILES},
        'harness_sha256_at_freeze': {os.path.basename(__file__): _sha(os.path.basename(__file__)),
                                     'p515_s53_w83_tight_tail_gate.py': _sha('p515_s53_w83_tight_tail_gate.py')},
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_freeze': _git(['rev-parse', 'HEAD']),
    }


def freeze_spec():
    for rel in (SPEC_V27['path'], CAL_JSON, CAL_MANIFEST):
        if _git(['ls-files', '--error-unmatch', rel]) == '' or _git(['status', '--porcelain', '--', rel]):
            raise SystemExit(f'{rel} must be tracked and clean before the spec is frozen')
    if _sha(SPEC_V27['path']) != SPEC_V27['sha256']:
        raise SystemExit('spec v27 sha256 != pin')
    man = _load(CAL_MANIFEST)
    if man.get(CAL_JSON) != _sha(CAL_JSON):
        raise SystemExit('calibration json does not hash to its manifest')
    dirty = _git(['status', '--porcelain', '--'] + list(PRODUCTION_FILES)
                 + [os.path.basename(__file__), 'p515_s53_w83_tight_tail_gate.py'])
    if dirty:
        raise SystemExit(f'production / harness files not clean in git:\n{dirty}')
    spec = build_spec(CAL_JSON, _sha(CAL_JSON), _sha(CAL_MANIFEST))
    text = json.dumps(spec, indent=1, default=str) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(S53, f'frozen_s53_spec_v28_{sha[:8]}.json')
    _refuse_overwrite(rel)
    with open(_abs(rel), 'w') as handle:
        handle.write(text)
    if _sha(rel) != sha:
        raise SystemExit('written spec does not hash to its name')
    _log(f'wrote {rel} (sha256 {sha}); predecessor v27 {SPEC_V27["sha256"]}')
    return rel


def run_checks():
    started = time.time()
    _log('(a) the switch reads the AA-off predicate')
    a = check_a_predicate()
    _log(f"    holds {a['holds']}; static {sum(a['static'].values())}/{len(a['static'])}; predicate true on "
         f"{a['predicate_not_permanently_false']['n_true']} committed C* cycles")
    _log('(b)/(c) holders, retry tiers, ESSO, restore -- on the SRP1 planning object (read only, no build, no solve)')
    planning = _load_planning()
    bc = check_b_c_holders(planning)
    _log(f"    b {bc['b']['holds']}  c {bc['c']['holds']}  restore {bc['restore']['holds']}  retry "
         f"{bc['retry_inheritance']['holds']}  esso {bc['esso']['holds']}  pw refused {bc['persistent_workers_refused']}")
    cap = check_capture_path()
    _log(f"production capture path: {cap['holds']} ({sum(cap['checks'].values())}/{len(cap['checks'])})")
    _log('parser validation vs W82 (reads ~4 GB of committed logs)')
    val = validate_parser_against_w82()
    _log(f"    {val['n_blocks']} blocks, mismatches {val['n_mismatches']}, known figures {val['known_figures_reproduced']}")
    items = {'a_switch_reads_the_aa_off_predicate': a['holds'],
             'b_false_leaves_options_untouched': bc['b']['holds'],
             'c_true_sets_1e-6_on_every_network_holder': bc['c']['holds'],
             'restore_is_exact': bc['restore']['holds'],
             'retry_tiers_inherit_the_tail': bc['retry_inheritance']['holds'],
             'esso_untouched': bc['esso']['holds'],
             'persistent_workers_refused': bc['persistent_workers_refused'],
             'predicate_not_permanently_false': a['predicate_not_permanently_false']['holds'],
             'production_capture_path_present': cap['holds'],
             'parser_reproduces_w82_exactly': val['holds']}
    return {'stage': STAGE, 'mode': '--run', 'utc': datetime.now(timezone.utc).isoformat(),
            'git_head': _git(['rev-parse', 'HEAD']), 'script_sha256': _sha(os.path.basename(__file__)),
            'production_sha256': {f: _sha(f) for f in PRODUCTION_FILES},
            'a_predicate': a, 'b_c_holders': bc, 'capture_path': cap, 'parser_validation_vs_w82': val,
            'items': items, 'all_hold': all(items.values()), 'wall_s': time.time() - started}


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--calibrate', action='store_true')
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    status = 1
    try:
        _log(STAGE)
        _log(f'git HEAD {_git(["rev-parse", "HEAD"])}; guard armed permitted=() (zero solves)')
        if args.calibrate:
            _refuse_overwrite(CAL_JSON)
            payload = calibrate()
            _write_json(CAL_JSON, payload)
            _write_manifest(CAL_MANIFEST, [CAL_JSON, os.path.basename(__file__), 'network.py']
                            + sorted(payload['logs_sha256_at_read']) + [CSTAR['report']])
            for fam, s in payload['summary_tail_cycles'].items():
                _log(f'{fam}: {json.dumps({k: v for k, v in s.items() if k != "parse_reasons"})}')
            _log(f'tail schedule on the committed run: {payload["derived_tail_schedule_on_the_committed_run"]}')
            status = 0
        elif args.freeze_spec:
            freeze_spec()
            status = 0
        else:
            _refuse_overwrite(CHECKS_JSON)
            payload = run_checks()
            _write_json(CHECKS_JSON, payload)
            _write_manifest(CHECKS_MANIFEST, [CHECKS_JSON, os.path.basename(__file__)] + list(PRODUCTION_FILES)
                            + [W82['json'], CSTAR['report']])
            for k, v in payload['items'].items():
                _log(f'   {k}: {v}')
            _log(f"ALL_HOLD={payload['all_hold']}")
            status = 0 if payload['all_hold'] else 1
    except SystemExit as error:
        _log(f'STOP: {error}')
        status = 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        status = 1
    finally:
        failures = GUARD.verify(0)
        _log(f'GUARD {dict(GUARD.counts)}; verify(0) -> {failures}')
        GUARD.uninstall()
        if failures:
            status = 1
    return status


if __name__ == '__main__':
    sys.exit(main())
