"""P5.15 Addendum 40 ruling 2 (tasks W54 -> W57) -- .NL PROBE for row 18's STRUCTURAL inactivation at the ADMM
initialisation solve. ZERO SOLVES.

Authority: frozen spec `data/SRP1/Results/P515S53/row18_structural/frozen_s53_row18_structural_spec_v3_1064db50.json`
(key `nl_probe_r2`; predecessor v2 5123e67b, which succeeded v1 05934ab0 -- v1 governed r1 of this probe);
PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 2. Template: `p515_s51_nl_identity_probe.py`.

W57 (Planner item 3): r1 showed P1 (A == B) on ONE block only (node 5, 2025 Spring). This version (schema v2)
probes EVERY DSO block of the 2 x 2 instance (3 nodes x 1 year x 4 days = 12 blocks; the declared list is asserted
against the instance before any build), all arms, and evaluates each prediction PER BLOCK. The r1 script
(sha256 a15cfaef..., committed at d573e145) is in git; r1's evidence is cited in a Planner report and is never
re-run onto: the output directory is write-once and label r1 exists. P3 is v2's restated P3 (same columns,
+2nT rows), recorded in v3 BEFORE this run -- at r2 it is a genuine prediction, not an explanation. P4 (new,
v3): the r1 block's A and B files reproduce r1's hashes exactly.

THE CLAIM UNDER TEST. W54 holds row 18 inactive at initialisation by deactivating its defining rows and fixing the
deviation pairs at 0 (`shared_resources_planning._set_row18_inactive_for_initialisation`). The approach is licensed
only if the .nl file production hands IPOPT for such a block is BYTE-IDENTICAL to the alpha = 0 block's -- i.e.
Pyomo's nl_v2 writer folds the fixed Vars to constants and skips the inactive rows. That is expected from the writer
source but was not verified before this probe.

ARMS (every 2 x 2 DSO block, each captured at the moment production would hand it to IPOPT -- the node's
`.optimize` is replaced ON THE INSTANCE by a stub that writes the .nl files of every block of the node and returns
no result; one build per (arm, node)):
  A         alpha = 0 build (row 18 not constructed)
  A_repeat  declared determinism control: a second, independent alpha = 0 build
  B         alpha = 0.5, W54 structural fix (rows inactive, deviation Vars fixed at 0)
  C         alpha = 0.5, W51's Param-only state (W51's function extracted verbatim from git at 7a98b25e and patched
            into the builder for this arm only)
  D         NEGATIVE CONTROL: alpha = 0.5, deviation Vars fixed at 0, rows left ACTIVE (probe-local, never production)
Each block is written twice: symbolic_solver_labels off (the production form) and on (names; .row/.col).

PREDICTIONS (recorded in frozen spec v3 before the run; evaluated PER BLOCK here, never tuned):
  P0  sha256(A) == sha256(A_repeat), both label settings, .row/.col identical      (control)
  P1  sha256(A) == sha256(B) EXACTLY, both label settings, .row/.col identical     (STOP if false on ANY block)
  P2  C - A = +4nT columns, +2nT rows, and the extra names are exactly row 18's
  P3  D has the SAME columns as A (.col identical) and +2nT rows, the extra rows exactly row18_dev_{p,q}_def
  P4  (block DSO:5:2025:Spring only) A's and B's files reproduce r1's sha256 exactly (both label settings, .row/.col)

Writing an .nl is not a solve: `SolveProfileGuard(permitted=())` is armed for the whole run and verified at exactly 0.

EXACT COMMAND (worktree root, canonical interpreter, attached, BOTH streams captured):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt LP_SOLVER_PATH=<from the main checkout .env> \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_row18_structural_nl_probe.py \\
      --label r2 --scratch <dir outside the repo> \\
      > data/SRP1/Results/P515S53/row18_structural/nl_probe_r2_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/row18_structural/nl_probe/<label>/{nl_probe.json, manifest_sha256.json,
cases/}; the .nl/.row/.col files go to <scratch>/nl and are hash-recorded in nl_probe.json.
Exit 0 when P0-P4 all hold on every block, 1 otherwise.
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

STAGE = ('P5.15 Addendum 40 ruling 2 (W57) -- .nl probe r2: row 18 structurally inactive at initialisation, '
         'every 2 x 2 DSO block')
SCHEMA = 'p515_s53_row18_structural_nl_probe_v2'
SPEC_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'row18_structural',
                        'frozen_s53_row18_structural_spec_v3_1064db50.json')
SPEC_SHA256 = '1064db50f8d339ac8bbc9ffdcdb2485e9e06b20707dbb73748dd840fe3878ff7'
SPEC_KEY = 'nl_probe_r2'
# Declared in spec v3 before the run: every DSO block of the 2 x 2 instance (asserted against the instance read).
DECLARED_BLOCKS = tuple(f'DSO:{n}:2025:{d}' for n in (5, 7, 9) for d in ('Spring', 'Summer', 'Autumn', 'Winter'))
# r1 (v1, committed at bf36d414): block DSO:5:2025:Spring, arms A and B (identical); P4 compares against these.
R1_BLOCK = 'DSO:5:2025:Spring'
R1_HASHES = {'labels_off': {'nl': 'a65a93ff3c49f0e1362dea6b100b5b85ef315c6aa59e7aa68cf5fb6556e9d87d'},
             'labels_on': {'nl': '8379c362fb5e261da8e0c1fe6605803a671820d18e05b21e396c2a506ccfe968',
                           'row': '59af856a5034545b99343c68d767abcf794f873663a74964bd81c8430b69b389',
                           'col': 'a12d53c2cd24bcb67c35047c0324c2b1e9bcfab6bf9b7ea0878ce692b5d05144'}}
OUT_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'row18_structural', 'nl_probe')
W51_COMMIT = '7a98b25e'
FUNCTION = '_set_row18_inactive_for_initialisation'
OVERRIDES_2X2 = {'years': {'2025': 5}, 'num_market_scenarios': 2, 'num_operation_scenarios': 2}
ALPHA_RUN = 0.50
DEV_ROWS = ('row18_dev_p_def', 'row18_dev_q_def')
DEV_VARS = ('row18_dev_p_up', 'row18_dev_p_down', 'row18_dev_q_up', 'row18_dev_q_down')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W57-nl-probe] {msg}', flush=True)


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


def write_nl(block, stem, nl_dir):
    out = {}
    for flag, tag in ((False, 'labels_off'), (True, 'labels_on')):
        path = os.path.join(nl_dir, f'{stem}_{tag}.nl')
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
    """Production's sequential DSO builder on one node, `.optimize` stubbed: writes EVERY block's .nl at the moment
    it would be handed to IPOPT. `init_fn` (arms C/D) replaces the module-level initialisation function the builder
    calls, for this build only. Returns one record per block, keyed 'DSO:<node>:<year>:<day>'."""
    dn = planning.distribution_networks[node_id]
    captured = {'calls': 0, 'blocks': {}}

    def stub(model, *args, **kwargs):
        captured['calls'] += 1
        for y in dn.years:
            for d in dn.days:
                block = model[y][d]
                captured['blocks'][f'DSO:{node_id}:{y}:{d}'] = {
                    'state': row18_state(block), 'nl': write_nl(block, f'{arm}_{node_id}_{y}_{d}', nl_dir)}
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
    init_label = 'production' if init_fn is None else init_fn.__name__ + (
        f' ({W51_COMMIT})' if init_fn.__name__ == FUNCTION else ' (probe-local negative control)')
    return {key: {'arm': arm, 'alpha': alpha, 'block': key, 'init_function': init_label,
                  'stub_calls_for_node': captured['calls'], 'row18_state_at_would_be_solve': rec['state'],
                  'nl': rec['nl']}
            for key, rec in captured['blocks'].items()}


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


def _block_predictions(A, A_repeat, B, C, D):
    """P0-P3 on one block (P4 is evaluated separately, on the r1 block only)."""
    sb = B['row18_state_at_would_be_solve']
    n, T = sb['n_scenarios'], sb['n_periods']
    cols = {k: v['nl']['labels_off']['n_vars_columns'] for k, v in (('A', A), ('C', C), ('D', D))}
    rows = {k: v['nl']['labels_off']['n_cons_rows'] for k, v in (('A', A), ('C', C), ('D', D))}
    p0 = _same(A, A_repeat)
    p1 = _same(A, B)
    diff_ac = name_diff(A, C)
    diff_ad = name_diff(A, D)
    expected_c_cols = {'row18_dev_p_down': n * T, 'row18_dev_p_up': n * T, 'row18_dev_q_down': n * T,
                       'row18_dev_q_up': n * T}
    expected_rows = {'row18_dev_p_def': n * T, 'row18_dev_q_def': n * T}
    d_col_identical = A['nl']['labels_on']['col_sha256'] == D['nl']['labels_on']['col_sha256']
    return n, T, {
        'P0_control_A_equals_A_repeat': {'outcome': p0, 'holds': _all_true(p0)},
        'P1_A_equals_B': {'outcome': p1, 'holds': _all_true(p1),
                          'on_false': 'STOP and report the difference; no workaround'},
        'P2_C_minus_A': {'n': n, 'T': T, 'column_delta': cols['C'] - cols['A'], 'row_delta': rows['C'] - rows['A'],
                         'expected_column_delta': 4 * n * T, 'expected_row_delta': 2 * n * T,
                         'extra_names': diff_ac,
                         'holds': (cols['C'] - cols['A'] == 4 * n * T and rows['C'] - rows['A'] == 2 * n * T
                                   and diff_ac['col']['only_in_y_by_family'] == expected_c_cols
                                   and diff_ac['row']['only_in_y_by_family'] == expected_rows
                                   and not diff_ac['col']['only_in_x_by_family']
                                   and not diff_ac['row']['only_in_x_by_family'])},
        'P3_D_same_columns_plus_2nT_rows': {
            'column_delta': cols['D'] - cols['A'], 'row_delta': rows['D'] - rows['A'],
            'expected_column_delta': 0, 'expected_row_delta': 2 * n * T,
            'col_file_identical_to_A': d_col_identical, 'names_vs_A': diff_ad,
            'holds': (cols['D'] == cols['A'] and rows['D'] - rows['A'] == 2 * n * T and d_col_identical
                      and not diff_ad['col']['only_in_y_by_family'] and not diff_ad['col']['only_in_x_by_family']
                      and diff_ad['row']['only_in_y_by_family'] == expected_rows
                      and not diff_ad['row']['only_in_x_by_family'])},
    }


def _arm_state_assertions(A, B, C, D, n, T):
    sa, sb, sc, sd = (A['row18_state_at_would_be_solve'], B['row18_state_at_would_be_solve'],
                      C['row18_state_at_would_be_solve'], D['row18_state_at_would_be_solve'])
    return {
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
    }


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
    with open(os.path.join(REPO, SPEC_REL)) as handle:
        spec_v3 = json.load(handle)
    w51_fn, w51_src = w51_function()
    prod_init_src = inspect.getsource(srp._set_row18_inactive_for_initialisation)
    checklist = {
        'frozen_spec_sha256_matches': spec_sha == SPEC_SHA256,
        'frozen_spec_declares_these_blocks_and_predictions': (
            tuple(spec_v3.get(SPEC_KEY, {}).get('blocks_declared', ())) == DECLARED_BLOCKS
            and sorted(spec_v3.get(SPEC_KEY, {}).get('predictions_per_block', {})) == ['P0', 'P1', 'P2', 'P3', 'P4']),
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
    instance_blocks = tuple(f'DSO:{node_id}:{y}:{d}' for node_id, dn in planning.distribution_networks.items()
                            for y in dn.years for d in dn.days)
    if instance_blocks != DECLARED_BLOCKS:
        print(f'REFUSED: instance blocks {instance_blocks} != declared {DECLARED_BLOCKS}', file=sys.stderr)
        return 1

    arms = {}           # arms[arm][block_key] -> record
    stub_calls = {}
    plan = (('A', 0.0, None), ('A_repeat', 0.0, None), ('B', ALPHA_RUN, None),
            ('C', ALPHA_RUN, w51_fn), ('D', ALPHA_RUN, negative_control_fix_rows_active))
    for arm, alpha, fn in plan:
        arms[arm] = {}
        for node_id in planning.distribution_networks:
            _log(f'arm {arm} node {node_id}: alpha {alpha}, init function '
                 f'{"production" if fn is None else fn.__name__}')
            recs = build_arm(planning, node_id, arm, alpha, fn, nl_dir)
            stub_calls[f'{arm}:{node_id}'] = next(iter(recs.values()))['stub_calls_for_node'] if recs else 0
            arms[arm].update(recs)
            for key, rec in recs.items():
                e = rec['nl']['labels_off']
                _log(f"   {key}: columns {e['n_vars_columns']} rows {e['n_cons_rows']} sha256 {e['sha256'][:16]}")

    labels_on_ok = all('row_sha256' in r['nl']['labels_on'] and 'col_sha256' in r['nl']['labels_on']
                       for recs in arms.values() for r in recs.values())
    header_counts_label_invariant = all(r['nl']['labels_off']['line2'] == r['nl']['labels_on']['line2']
                                        for recs in arms.values() for r in recs.values())
    every_arm_every_block = all(tuple(arms[a].keys()) == DECLARED_BLOCKS for a in arms)
    one_stub_call_per_arm_node = bool(stub_calls) and all(v == 1 for v in stub_calls.values())

    per_block = {}
    for key in DECLARED_BLOCKS:
        A, A_rep, B, C, D = (arms[a][key] for a in ('A', 'A_repeat', 'B', 'C', 'D'))
        n, T, preds = _block_predictions(A, A_rep, B, C, D)
        states = _arm_state_assertions(A, B, C, D, n, T)
        per_block[key] = {
            'n_scenarios': n, 'n_periods': T,
            'columns_by_arm': {a: arms[a][key]['nl']['labels_off']['n_vars_columns'] for a in arms},
            'rows_by_arm': {a: arms[a][key]['nl']['labels_off']['n_cons_rows'] for a in arms},
            'sha256_labels_off_by_arm': {a: arms[a][key]['nl']['labels_off']['sha256'] for a in arms},
            'sha256_labels_on_by_arm': {a: arms[a][key]['nl']['labels_on']['sha256'] for a in arms},
            'arm_state_assertions': states, 'predictions': preds}

    rA, rB = arms['A'][R1_BLOCK], arms['B'][R1_BLOCK]
    p4_outcome = {arm: {tag: {kind: (rec['nl'][tag]['sha256'] if kind == 'nl' else rec['nl'][tag][f'{kind}_sha256'])
                              == expected for kind, expected in R1_HASHES[tag].items()}
                        for tag in R1_HASHES}
                  for arm, rec in (('A', rA), ('B', rB))}
    per_block[R1_BLOCK]['predictions']['P4_reproduces_r1_hashes'] = {
        'r1_hashes': R1_HASHES, 'outcome': p4_outcome, 'holds': _all_true(p4_outcome)}

    def _holds(pid):
        return {key: b['predictions'][pid]['holds'] for key, b in per_block.items() if pid in b['predictions']}

    summary = {pid: _holds(pid) for pid in ('P0_control_A_equals_A_repeat', 'P1_A_equals_B', 'P2_C_minus_A',
                                            'P3_D_same_columns_plus_2nT_rows', 'P4_reproduces_r1_hashes')}
    p1_all = all(summary['P1_A_equals_B'].values()) and len(summary['P1_A_equals_B']) == len(DECLARED_BLOCKS)
    arm_states_all = all(all(b['arm_state_assertions'].values()) for b in per_block.values())

    GUARD.uninstall()
    verify_failures = GUARD.verify(expected_solves=0)
    all_pass = (all(all(v.values()) and v for v in summary.values()) and arm_states_all and labels_on_ok
                and header_counts_label_invariant and every_arm_every_block and one_stub_call_per_arm_node
                and not verify_failures)
    payload = {
        'schema': SCHEMA, 'stage': STAGE,
        'frozen_spec': {'path': SPEC_REL, 'key': SPEC_KEY, 'sha256_pinned': SPEC_SHA256, 'sha256_observed': spec_sha},
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
                     'blocks': list(DECLARED_BLOCKS)},
        'scratch': scratch,
        'arms': arms, 'stub_calls_per_arm_node': stub_calls,
        'header_counts_identical_between_label_settings': header_counts_label_invariant,
        'labels_on_row_col_written_for_every_arm_and_block': labels_on_ok,
        'every_arm_wrote_every_declared_block': every_arm_every_block,
        'one_stub_call_per_arm_node': one_stub_call_per_arm_node,
        'per_block': per_block,
        'predictions_holds_by_block': summary,
        'P1_holds_on_every_block': p1_all,
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
    for key, b in per_block.items():
        _log(f"{key}: columns {b['columns_by_arm']} rows {b['rows_by_arm']}")
        _log(f"   {({pid: ('HOLDS' if p['holds'] else 'FAILS') for pid, p in b['predictions'].items()})}; "
             f"arm states {all(b['arm_state_assertions'].values())}")
    for pid, v in summary.items():
        _log(f'  {pid}: {sum(v.values())}/{len(v)} blocks hold')
    if not p1_all:
        _log('P1 FAILS on at least one block -> STOP (Planner W57)')
    _log(f'guard {dict(GUARD.counts)} verify_failures {verify_failures}')
    _log(f'all_checks_pass = {all_pass}; wrote {os.path.relpath(out_path, REPO)}')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
