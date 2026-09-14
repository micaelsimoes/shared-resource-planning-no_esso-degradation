"""
WORKER T3 -- re-run the exact two arms of p515_h_tol_remedy_check.py BY IMPORT
(never re-running the module's main(), which would write onto the committed
data/SRP1/Results/P5151/tol_remedy_check_summary.json / tol_check_logs/ that a
committed report cites -- CLAUDE.md evidence rule). Output is redirected to a
NEW location under the authorized data/SRP1/Results/P515F/ directory by
monkeypatching the imported module's LOG_ROOT global (the same technique the
module itself uses for SED.ESSO_TOL_OVERRIDES), restored afterward.

Compares mu_final, s_obj, detector ratio (complementarity_ratio_max) and
spurious_throughput_measured against the COMMITTED
data/SRP1/Results/P5151/tol_remedy_check_summary.json, per node per arm.
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout

REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_h_tol_remedy_check as H  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515F')
os.makedirs(OUT, exist_ok=True)
COMMITTED_SUMMARY = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P5151', 'tol_remedy_check_summary.json')

# Redirect this re-run's output to a NEW path -- never touch the committed
# data/SRP1/Results/P5151 tree.
NEW_LOG_ROOT = os.path.join(OUT, 't3_tol_check_logs')
original_log_root = H.LOG_ROOT
H.LOG_ROOT = NEW_LOG_ROOT
os.makedirs(NEW_LOG_ROOT, exist_ok=True)

PERMITTED = [('shared_energy_storage_data.py', '_run_solver_attempt')]
EXPECTED_SOLVES = 6  # same declared count as the original harness: 2 arms x 3 active nodes


def main():
    guard = SolveProfileGuard(PERMITTED, label='P5.15-F T3 tol remedy re-check').install()
    try:
        arm_reports = {}
        for tag, tol_overrides in H.ARMS:
            with redirect_stdout(io.StringIO()):
                arm_reports[tag] = H.run_arm(tag, tol_overrides)
    finally:
        guard.uninstall()
        H.LOG_ROOT = original_log_root

    guard_failures = guard.verify(EXPECTED_SOLVES)

    with open(COMMITTED_SUMMARY) as handle:
        committed = json.load(handle)

    fields = ('mu_final', 's_obj', 'complementarity_ratio_max', 'spurious_throughput_measured',
              'spurious_throughput_bound')
    comparison = {}
    max_abs_rel_diff = 0.0
    max_abs_rel_diff_where = None
    harness_own_reconstruction_broken = []

    for arm_key, committed_table_key in (
        ('control', 'control_tol_1e-6_acceptable_1e-5'),
        ('remedy_h', 'remedy_h_tol_1e-8_acceptable_1e-7'),
    ):
        committed_table = committed['results'][committed_table_key]
        rerun_arm = arm_reports[arm_key]

        # Production's OWN sink (shared_ess_data.esso_complementarity_diagnostics,
        # populated inside _optimize() with the CORRECT, currently-in-force log
        # path) -- keyed by node_id. This is the authoritative source of the
        # single-solve detector values this fix's item 1/2 changed the FILE
        # LOCATION of, not the computation of.
        sink_by_node = {entry['node_id']: entry for entry in rerun_arm['esso_complementarity_diagnostics_sink']}

        for node_id_str, committed_node in committed_table.items():
            node_id = int(node_id_str)

            # The harness's OWN manual reconstruction (p515_h_tol_remedy_check.py
            # lines 182-190: primary_log = os.path.join(arm_dir, f'optim_log_node_{node_id}.txt')).
            # That path matched what production wrote under the OLD naming
            # convention (bare relative filename resolved against a chdir'd cwd
            # equal to arm_dir); under the NEW convention (this fix, items 1-2)
            # production writes to shared_ess_data.logs_dir with a different
            # filename stem, so the harness's guess no longer resolves to the
            # file production actually wrote. Recorded, not silently worked
            # around.
            harness_diag = rerun_arm['nodes'][node_id]['diagnostics']
            if harness_diag.get('parse_reason'):
                harness_own_reconstruction_broken.append(
                    {'arm': arm_key, 'node_id': node_id, 'parse_reason': harness_diag['parse_reason'],
                     'path_it_guessed': None}
                )

            rerun_diag = sink_by_node.get(node_id, {})
            row = {}
            for field in fields:
                committed_value = committed_node.get(field)
                rerun_value = rerun_diag.get(field)
                if isinstance(committed_value, (int, float)) and isinstance(rerun_value, (int, float)):
                    abs_diff = abs(rerun_value - committed_value)
                    denom = abs(committed_value) if committed_value != 0 else 1.0
                    rel_diff = abs_diff / denom
                    if rel_diff > max_abs_rel_diff:
                        max_abs_rel_diff = rel_diff
                        max_abs_rel_diff_where = f'{arm_key}/node{node_id}/{field}'
                else:
                    rel_diff = None
                row[field] = {'committed': committed_value,
                               'rerun_via_production_sink': rerun_value,
                               'rerun_via_harness_own_path_guess': harness_diag.get(field),
                               'rel_diff_vs_production_sink': rel_diff}
            comparison[f'{arm_key}/node{node_id}'] = row

    report = {
        'stage': 'P5.15-F T3',
        'committed_summary_path': COMMITTED_SUMMARY,
        'rerun_log_root': NEW_LOG_ROOT,
        'guard_counts': guard.counts,
        'guard_verify_failures': guard_failures,
        'comparison': comparison,
        'max_relative_difference': max_abs_rel_diff,
        'max_relative_difference_where': max_abs_rel_diff_where,
        'pass_le_1e-12': max_abs_rel_diff <= 1e-12,
        'harness_own_path_reconstruction_broken_by_this_fix': harness_own_reconstruction_broken,
    }
    with open(os.path.join(OUT, 't3_tol_remedy_recheck.json'), 'w') as handle:
        json.dump(report, handle, indent=2, default=str)

    print('GUARD_COUNTS', guard.counts)
    print('GUARD_VERIFY_FAILURES', guard_failures)
    for key, row in comparison.items():
        print(key)
        for field, vals in row.items():
            print('   ', field, vals)
    print('MAX_RELATIVE_DIFFERENCE', max_abs_rel_diff, 'at', max_abs_rel_diff_where)
    print('PASS_LE_1E-12', max_abs_rel_diff <= 1e-12)
    print('HARNESS_OWN_PATH_RECONSTRUCTION_BROKEN', harness_own_reconstruction_broken)


if __name__ == '__main__':
    main()
