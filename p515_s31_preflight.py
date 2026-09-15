"""
P5.15 Step 3.1 (S31 worker task) -- Part 4: pre-flight.

Runs `run_admm_arm` BY IMPORT (never through this module's own `__main__` --
`p515_g_g1_g4_admm_gates.py` is a separate module and its `__main__` block is never
executed here) for ONE cycle under the new (Step 3.1) production code, into a fresh
smoke root `data/SRP1/Results/P515S31/preflight/` and a fresh eval id, then runs the
Part 3 level writer on its final models.

    python p515_s31_preflight.py
"""

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31', 'preflight')


def main():
    G._require_fresh_output_root(OUT_DIR)

    precheck_eval_id = 'p515s31_preflight_precheck'
    precheck_eval_dir = os.path.join(G.O.WORK_DIR, precheck_eval_id)
    if os.path.exists(precheck_eval_dir):
        raise RuntimeError(f'refusing to start: precheck eval dir already exists: {precheck_eval_dir}')
    precheck_planning = G.O.fresh_planning(precheck_eval_id)
    checklist = G.assert_s31_capture_paths(precheck_planning)
    del precheck_planning
    print(f'[S31 Part 4] capture-path pre-flight passed: {checklist}')

    def hook(planning, sed, models, rows, report, out_dir, label):
        G.write_component_levels_terminal(planning, sed, models, rows, report, out_dir, label)

    report, report_path = G.run_admm_arm(
        'preflight', OUT_DIR, k_override=None, eval_id='p515s31_preflight',
        num_max_iters_override=1, post_run_hook=hook)

    print(f'[S31 Part 4] arm report: {report_path}')
    print(f"[S31 Part 4] cycles_run={report['cycles_run']} "
          f"local_solve_failures={report['local_solve_failures']} "
          f"solve_profile={report['solve_profile']} "
          f"network_failures={report['network_failures_summary']}")


if __name__ == '__main__':
    main()
