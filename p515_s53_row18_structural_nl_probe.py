"""P5.15 Addendum 40 ruling 2 (task W54) -- .NL PROBE for row 18's STRUCTURAL inactivation at the ADMM
initialisation solve. ZERO SOLVES.

Authority: frozen spec `data/SRP1/Results/P515S53/row18_structural/frozen_s53_row18_structural_spec_v1_05934ab0.json`
(amends v23 `ruling2_init_fix`'s mechanism; its predictions P0-P3 were recorded before this run);
PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 2. Template: `p515_s51_nl_identity_probe.py`.

THE CLAIM UNDER TEST. W54 holds row 18 inactive at initialisation by deactivating its defining rows and fixing the
deviation pairs at 0 (`shared_resources_planning._set_row18_inactive_for_initialisation`). The approach is licensed
only if the .nl file production hands IPOPT for such a block is BYTE-IDENTICAL to the alpha = 0 block's -- i.e.
Pyomo's nl_v2 writer folds the fixed Vars to constants and skips the inactive rows. That is expected from the writer
source but was not verified before this probe.

ARMS (one 2 x 2 DSO block: the first DSO node's first (year, day) block, captured at the moment production would
hand it to IPOPT -- the node's `.optimize` is replaced ON THE INSTANCE by a stub that writes the .nl files and
returns no result):
  A         alpha = 0 build (row 18 not constructed)
  A_repeat  declared determinism control: a second, independent alpha = 0 build
  B         alpha = 0.5, W54 structural fix (rows inactive, deviation Vars fixed at 0)
  C         alpha = 0.5, W51's Param-only state (W51's function extracted verbatim from git at 7a98b25e and patched
            into the builder for this arm only)
  D         NEGATIVE CONTROL: alpha = 0.5, deviation Vars fixed at 0, rows left ACTIVE (probe-local, never production)
Each block is written twice: symbolic_solver_labels off (the production form) and on (names; .row/.col).

PREDICTIONS (recorded in the frozen spec before the run; evaluated here, never tuned):
  P0  sha256(A) == sha256(A_repeat)          (control)
  P1  sha256(A) == sha256(B) exactly          (STOP if false -- no workaround)
  P2  C - A = +4nT columns, +2nT rows, and the extra names are exactly row 18's
  P3  D has fewer columns than A              (STOP if false -- the hazard reading would be wrong)

Writing an .nl is not a solve: `SolveProfileGuard(permitted=())` is armed for the whole run and verified at exactly 0.

EXACT COMMAND (worktree root, canonical interpreter, attached, BOTH streams captured):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt LP_SOLVER_PATH=<from the main checkout .env> \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_row18_structural_nl_probe.py \\
      --label r1 --scratch <dir outside the repo> \\
      > data/SRP1/Results/P515S53/row18_structural/nl_probe_r1_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/row18_structural/nl_probe/<label>/{nl_probe.json, manifest_sha256.json,
cases/}; the .nl/.row/.col files go to <scratch>/nl and are hash-recorded in nl_probe.json.
Exit 0 when P0-P3 all hold, 1 otherwise.
"""

import argparse
import ast
import hashlib
import inspect
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

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W54 row 18 structural .nl probe (never solves)').install()

import pyomo  # noqa: E402
import pyomo.environ as pe  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

STAGE = 'P5.15 Addendum 40 ruling 2 (W54) -- .nl probe: row 18 structurally inactive at initialisation'
SCHEMA = 'p515_s53_row18_structural_nl_probe_v1'
SPEC_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'row18_structural',
                        'frozen_s53_row18_structural_spec_v1_05934ab0.json')
SPEC_SHA256 = '05934ab0803a6744a5f22106865baafa27eb46f76d4240adbbc53244aac7be04'
OUT_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'row18_structural', 'nl_probe')
W51_COMMIT = '7a98b25e'
FUNCTION = '_set_row18_inactive_for_initialisation'
OVERRIDES_2X2 = {'years': {'2025': 5}, 'num_market_scenarios': 2, 'num_operation_scenarios': 2}
ALPHA_RUN = 0.50
DEV_ROWS = ('row18_dev_p_def', 'row18_dev_q_def')
DEV_VARS = ('row18_dev_p_up', 'row18_dev_p_down', 'row18_dev_q_up', 'row18_dev_q_down')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W54-nl-probe] {msg}', flush=True)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


# ======================================================================================================================
#  the arm-specific initialisation functions
# ======================================================================================================================
def w51_function():
    """W51's `_set_row18_inactive_for_initialisation`, verbatim from git at W51_COMMIT (arm C only)."""
    text = subprocess.run(['git', 'show', f'{W51_COMMIT}:shared_resources_planning.py'], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == FUNCTION:
            source = ast.get_source_segment(text, node)
            namespace = {'pe': pe}
            exec(compile(source, f'<{W51_COMMIT}:shared_resources_planning.py:{FUNCTION}>', 'exec'), namespace)
            return namespace[FUNCTION], source
    raise SystemExit(f'{FUNCTION} not found at {W51_COMMIT}')


def negative_control_fix_rows_active(model):
    """ARM D ONLY -- the hazard the production code excludes: the deviation pairs FIXED at 0 with their rows left
    ACTIVE. Never used by production."""
    if not hasattr(model, 'row18_alpha'):
        return
    for name in DEV_VARS:
        for index in getattr(model, name):
            getattr(model, name)[index].fix(0.0)


# ======================================================================================================================
#  capture
# ======================================================================================================================
def row18_state(block):
    if not hasattr(block, 'row18_alpha'):
        return {'row18_wired': False, 'row18_alpha_admm_present': hasattr(block, 'row18_alpha_admm')}
    rows = [(name, index, getattr(block, name)[index]) for name in DEV_ROWS for index in getattr(block, name)]
    dvars = [getattr(block, name)[index] for name in DEV_VARS for index in getattr(block, name)]
    fixed_pair_active_row = 0
    for row_name, up_name, down_name in (('row18_dev_p_def', 'row18_dev_p_up', 'row18_dev_p_down'),
                                         ('row18_dev_q_def', 'row18_dev_q_up', 'row18_dev_q_down')):
        for index in getattr(block, row_name):
            if getattr(block, row_name)[index].active and (getattr(block, up_name)[index].fixed
                                                           or getattr(block, down_name)[index].fixed):
                fixed_pair_active_row += 1
    return {
        'row18_wired': True,
        'row18_alpha': float(pe.value(block.row18_alpha)),
        'row18_alpha_admm_present': hasattr(block, 'row18_alpha_admm'),
        'n_rows': len(rows), 'n_rows_active': sum(1 for _, _, r in rows if r.active),
        'n_dev_vars': len(dvars), 'n_dev_vars_fixed': sum(1 for v in dvars if v.fixed),
        'fixed_dev_values': sorted({v.value for v in dvars if v.fixed}),
        'n_indices_fixed_pair_with_active_row': fixed_pair_active_row,
        'charge_value': float(pe.value(block.row18_deviation_charge)),
        'n_scenarios': len(block.scenarios_market) * len(block.scenarios_operation),
        'n_periods': len(block.periods),
    }


def _nl_header(path):
    with open(path) as handle:
        line1 = handle.readline().rstrip('\n')
        line2 = handle.readline().rstrip('\n')
    nums = [int(x) for x in line2.split('#')[0].split()]
    return {'line1': line1, 'line2': line2, 'n_vars_columns': nums[0], 'n_cons_rows': nums[1],
            'n_objs': nums[2], 'n_ranges': nums[3], 'n_eqns': nums[4]}


def write_nl(block, arm, nl_dir):
    out = {}
    for flag, tag in ((False, 'labels_off'), (True, 'labels_on')):
        path = os.path.join(nl_dir, f'{arm}_{tag}.nl')
        if os.path.exists(path):
            raise SystemExit(f'REFUSED: exists: {path}')
        block.write(path, format='nl', io_options={'symbolic_solver_labels': flag})
        entry = {'path': path, 'sha256': _sha256_file(path), 'bytes': os.path.getsize(path), **_nl_header(path)}
        for suffix in ('row', 'col'):
            side = path[:-3] + f'.{suffix}'
            if os.path.exists(side):
                entry[f'{suffix}_path'] = side
                entry[f'{suffix}_sha256'] = _sha256_file(side)
        out[tag] = entry
    return out


def build_arm(planning, node_id, arm, alpha, init_fn, nl_dir):
    """Production's sequential DSO builder on one node, `.optimize` stubbed: writes the first block's .nl at the
    moment it would be handed to IPOPT. `init_fn` (arms C/D) replaces the module-level initialisation function the
    builder calls, for this build only."""
    dn = planning.distribution_networks[node_id]
    year, day = next(iter(dn.years)), next(iter(dn.days))
    captured = {'calls': 0}

    def stub(model, *args, **kwargs):
        captured['calls'] += 1
        block = model[year][day]
        captured['state'] = row18_state(block)
        captured['nl'] = write_nl(block, arm, nl_dir)
        return {y: {d: None for d in dn.days} for y in dn.years}

    original = srp._set_row18_inactive_for_initialisation
    dn.optimize = stub
    if init_fn is not None:
        srp._set_row18_inactive_for_initialisation = init_fn
    try:
        consensus_vars, _dual = srp.create_admm_variables(planning)
        candidate = planning.get_initial_candidate_solution()
        srp.create_distribution_networks_models_sequential({node_id: dn}, consensus_vars, candidate['total_capacity'],
                                                           premium_alpha=alpha, premium_floor=None)
    finally:
        srp._set_row18_inactive_for_initialisation = original
        del dn.optimize
    assert srp._set_row18_inactive_for_initialisation is original
    return {'arm': arm, 'alpha': alpha, 'block': f'DSO:{node_id}:{year}:{day}',
            'init_function': 'production' if init_fn is None else init_fn.__name__ + (
                f' ({W51_COMMIT})' if init_fn.__name__ == FUNCTION else ' (probe-local negative control)'),
            'stub_calls': captured['calls'], 'row18_state_at_would_be_solve': captured.get('state'),
            'nl': captured.get('nl')}


def _names(path):
    with open(path) as handle:
        return [line.rstrip('\n') for line in handle]


def _family(name):
    return name.split('[')[0]


def _family_counts(names):
    out = {}
    for n in names:
        out[_family(n)] = out.get(_family(n), 0) + 1
    return dict(sorted(out.items()))


def name_diff(arm_x, arm_y):
    """Names in y and not x / in x and not y, from the labels-on .col / .row files."""
    out = {}
    for suffix in ('col', 'row'):
        nx_ = _names(arm_x['nl']['labels_on'][f'{suffix}_path'])
        ny_ = _names(arm_y['nl']['labels_on'][f'{suffix}_path'])
        sx, sy = set(nx_), set(ny_)
        out[suffix] = {'n_x': len(nx_), 'n_y': len(ny_),
                       'only_in_y_by_family': _family_counts(sorted(sy - sx)),
                       'only_in_x_by_family': _family_counts(sorted(sx - sy)),
                       'only_in_x_first_20': sorted(sx - sy)[:20]}
    return out


def _same(a, b):
    return {tag: {'nl': a['nl'][tag]['sha256'] == b['nl'][tag]['sha256'],
                  **({'row': a['nl'][tag]['row_sha256'] == b['nl'][tag]['row_sha256'],
                      'col': a['nl'][tag]['col_sha256'] == b['nl'][tag]['col_sha256']}
                     if tag == 'labels_on' else {})}
            for tag in ('labels_off', 'labels_on')}


def _all_true(d):
    return all(_all_true(v) if isinstance(v, dict) else bool(v) for v in d.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--scratch', required=True)
    args = ap.parse_args()
    scratch = os.path.abspath(args.scratch)
    if scratch.startswith(REPO + os.sep):
        raise SystemExit('--scratch must be outside the repository')
    out_dir = os.path.join(REPO, OUT_ROOT_REL, args.label)
    if os.path.exists(out_dir):
        print(f'REFUSED: output directory exists (write-once): {out_dir}', file=sys.stderr)
        return 1
    nl_dir = os.path.join(scratch, args.label, 'nl')
    if os.path.exists(nl_dir):
        print(f'REFUSED: scratch .nl directory exists: {nl_dir}', file=sys.stderr)
        return 1

    # ---- capture-path checklist, BEFORE anything is built (fails fast) ----
    spec_sha = _sha256_file(os.path.join(REPO, SPEC_REL))
    w51_fn, w51_src = w51_function()
    prod_init_src = inspect.getsource(srp._set_row18_inactive_for_initialisation)
    checklist = {
        'frozen_spec_sha256_matches': spec_sha == SPEC_SHA256,
        'production_init_is_structural': ('.deactivate()' in prod_init_src and '.fix(0.0)' in prod_init_src
                                          and 'row18_alpha.set_value' not in prod_init_src
                                          and 'row18_alpha_admm' not in prod_init_src),
        'production_families_constant_present': hasattr(srp, '_ROW18_DEVIATION_FAMILIES'),
        'w51_function_is_param_only': ('model.row18_alpha.set_value(0.0)' in w51_src
                                       and '.fix(' not in w51_src and '.deactivate(' not in w51_src),
    }
    if not all(checklist.values()):
        print(f'REFUSED: capture-path checklist failed: {checklist}', file=sys.stderr)
        return 1
    os.makedirs(out_dir)
    os.makedirs(nl_dir)
    started = time.time()
    _log(STAGE)
    _log(f'checklist {checklist}')

    case, spec, changes = S.derive_case('srp1', OVERRIDES_2X2)
    case_dir = os.path.join(out_dir, 'cases')
    os.makedirs(case_dir)
    case_path = os.path.join(case_dir, 'SRP1__2x2.json')
    with open(case_path, 'w') as handle:
        json.dump(case, handle, indent='\t')
    planning = SharedResourcesPlanning(S.DATA_DIR, os.path.relpath(case_path, S.DATA_DIR))
    planning.name = 'SRP1'
    planning.results_dir = os.path.join(scratch, args.label, 'Results')
    planning.diagrams_dir = os.path.join(scratch, args.label, 'Diagrams')
    planning.logs_dir = os.path.join(planning.results_dir, 'Logs')
    planning.read_planning_problem()
    planning.parallel_execution = False
    node_id = next(iter(planning.distribution_networks))

    arms = {}
    plan = (('A', 0.0, None), ('A_repeat', 0.0, None), ('B', ALPHA_RUN, None),
            ('C', ALPHA_RUN, w51_fn), ('D', ALPHA_RUN, negative_control_fix_rows_active))
    for arm, alpha, fn in plan:
        _log(f'arm {arm}: alpha {alpha}, init function {"production" if fn is None else fn.__name__}')
        arms[arm] = build_arm(planning, node_id, arm, alpha, fn, nl_dir)
        for tag, e in arms[arm]['nl'].items():
            _log(f"   {tag}: columns {e['n_vars_columns']} rows {e['n_cons_rows']} sha256 {e['sha256'][:16]}")
        _log(f"   row 18 state: {arms[arm]['row18_state_at_would_be_solve']}")
    labels_on_ok = all('row_sha256' in a['nl']['labels_on'] and 'col_sha256' in a['nl']['labels_on']
                       for a in arms.values())

    A, B, C, D = arms['A'], arms['B'], arms['C'], arms['D']
    n = B['row18_state_at_would_be_solve']['n_scenarios']
    T = B['row18_state_at_would_be_solve']['n_periods']
    cols = {k: v['nl']['labels_off']['n_vars_columns'] for k, v in arms.items()}
    rows = {k: v['nl']['labels_off']['n_cons_rows'] for k, v in arms.items()}
    header_counts_label_invariant = all(
        v['nl']['labels_off']['line2'] == v['nl']['labels_on']['line2'] for v in arms.values())

    # ---- arm-state assertions (per index, summarised) ----
    sa, sb, sc, sd = (A['row18_state_at_would_be_solve'], B['row18_state_at_would_be_solve'],
                      C['row18_state_at_would_be_solve'], D['row18_state_at_would_be_solve'])
    arm_states = {
        'A_not_wired': sa['row18_wired'] is False and not sa['row18_alpha_admm_present'],
        'B_structural_init_state': (sb['row18_wired'] and sb['row18_alpha'] == ALPHA_RUN
                                    and not sb['row18_alpha_admm_present']
                                    and sb['n_rows'] == 2 * n * T and sb['n_rows_active'] == 0
                                    and sb['n_dev_vars'] == 4 * n * T and sb['n_dev_vars_fixed'] == 4 * n * T
                                    and sb['fixed_dev_values'] == [0.0]
                                    and sb['n_indices_fixed_pair_with_active_row'] == 0
                                    and sb['charge_value'] == 0.0),
        'C_w51_param_only_state': (sc['row18_wired'] and sc['row18_alpha'] == 0.0 and sc['row18_alpha_admm_present']
                                   and sc['n_rows_active'] == 2 * n * T and sc['n_dev_vars_fixed'] == 0),
        'D_negative_control_state': (sd['row18_wired'] and sd['row18_alpha'] == ALPHA_RUN
                                     and sd['n_rows_active'] == 2 * n * T and sd['n_dev_vars_fixed'] == 4 * n * T
                                     and sd['n_indices_fixed_pair_with_active_row'] == 2 * n * T),
        'one_stub_call_per_arm': all(a['stub_calls'] == 1 for a in arms.values()),
    }

    p0 = _same(A, arms['A_repeat'])
    p1 = _same(A, B)
    diff_ac = name_diff(A, C)
    diff_ad = name_diff(A, D)
    expected_c_cols = {'row18_dev_p_down': n * T, 'row18_dev_p_up': n * T, 'row18_dev_q_down': n * T,
                       'row18_dev_q_up': n * T}
    expected_c_rows = {'row18_dev_p_def': n * T, 'row18_dev_q_def': n * T}
    predictions = {
        'P0_control_A_equals_A_repeat': {'outcome': p0, 'holds': _all_true(p0)},
        'P1_A_equals_B': {'outcome': p1, 'holds': _all_true(p1),
                          'on_false': 'STOP and report the difference; no workaround'},
        'P2_C_minus_A': {'n': n, 'T': T, 'column_delta': cols['C'] - cols['A'], 'row_delta': rows['C'] - rows['A'],
                         'expected_column_delta': 4 * n * T, 'expected_row_delta': 2 * n * T,
                         'extra_names': diff_ac,
                         'holds': (cols['C'] - cols['A'] == 4 * n * T and rows['C'] - rows['A'] == 2 * n * T
                                   and diff_ac['col']['only_in_y_by_family'] == expected_c_cols
                                   and diff_ac['row']['only_in_y_by_family'] == expected_c_rows
                                   and not diff_ac['col']['only_in_x_by_family']
                                   and not diff_ac['row']['only_in_x_by_family'])},
        'P3_D_fewer_columns_than_A': {'column_delta': cols['D'] - cols['A'], 'row_delta': rows['D'] - rows['A'],
                                      'names_vs_A': diff_ad, 'holds': cols['D'] < cols['A'],
                                      'on_false': 'STOP and report: the stated hazard reading would be wrong'},
    }

    GUARD.uninstall()
    verify_failures = GUARD.verify(expected_solves=0)
    all_pass = (all(p['holds'] for p in predictions.values()) and all(arm_states.values()) and labels_on_ok
                and header_counts_label_invariant and not verify_failures)
    payload = {
        'schema': SCHEMA, 'stage': STAGE,
        'frozen_spec': {'path': SPEC_REL, 'sha256_pinned': SPEC_SHA256, 'sha256_observed': spec_sha},
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'interpreter': sys.executable, 'argv': sys.argv,
        'pyomo_version': pyomo.version.version,
        'script': os.path.basename(__file__), 'script_sha256': _sha256_file(os.path.abspath(__file__)),
        'production_sha256': {f: _sha256_file(os.path.join(REPO, f)) for f in (
            'shared_resources_planning.py', 'model_construction_helpers.py', 'admm_parameters.py')},
        'git_head': _git(['rev-parse', 'HEAD']),
        'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
        'capture_path_checklist': checklist,
        'w51_function': {'commit': W51_COMMIT, 'source_sha256': hashlib.sha256(w51_src.encode()).hexdigest()},
        'production_init_function_source_sha256': hashlib.sha256(prod_init_src.encode()).hexdigest(),
        'instance': {'overrides': OVERRIDES_2X2, 'definition': spec, 'changes_vs_source': changes,
                     'case_path': os.path.relpath(case_path, REPO), 'case_sha256': _sha256_file(case_path),
                     'dso_node': node_id, 'n_scenarios': n, 'n_periods': T},
        'scratch': scratch,
        'arms': arms, 'columns_by_arm': cols, 'rows_by_arm': rows,
        'header_counts_identical_between_label_settings': header_counts_label_invariant,
        'labels_on_row_col_written_for_every_arm': labels_on_ok,
        'arm_state_assertions': arm_states,
        'predictions': predictions,
        'solve_profile_guard': {'permitted': [], 'counts': dict(GUARD.counts), 'verify_failures': verify_failures},
        'all_checks_pass': bool(all_pass), 'wall_s': time.time() - started,
    }
    out_path = os.path.join(out_dir, 'nl_probe.json')
    with open(out_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_dir):
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = _sha256_file(fpath)
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    _log(f'columns {cols}')
    _log(f'rows    {rows}')
    _log(f'arm states {arm_states}')
    for key, value in predictions.items():
        _log(f"  {key}: {'HOLDS' if value['holds'] else 'FAILS'}")
    _log(f'guard {dict(GUARD.counts)} verify_failures {verify_failures}')
    _log(f'all_checks_pass = {all_pass}; wrote {os.path.relpath(out_path, REPO)}')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
