"""
P5.15 Step 3.1-C (S31C worker task) -- Part 4: pre-flight.

Runs `run_admm_arm` BY IMPORT (never through `p515_g_g1_g4_admm_gates.py`'s own
`__main__` -- the Planner launches the gate, not this script) for ONE cycle under the
Addendum 12 production code, into a fresh smoke root
`data/SRP1/Results/P515S31C/preflight/` and a fresh eval id, then runs the S31C level
writer (which itself reuses the S31 level writer, zero extra solves) on the final
models.

    python p515_s31c_preflight.py
"""

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31C', 'preflight')


def main():
    G._require_fresh_output_root(OUT_DIR)

    precheck_eval_id = 'p515s31c_preflight_precheck'
    precheck_eval_dir = os.path.join(G.O.WORK_DIR, precheck_eval_id)
    if os.path.exists(precheck_eval_dir):
        raise RuntimeError(f'refusing to start: precheck eval dir already exists: {precheck_eval_dir}')
    precheck_planning = G.O.fresh_planning(precheck_eval_id)
    checklist = G.assert_s31c_capture_paths(precheck_planning)
    del precheck_planning
    print(f'[S31C Part 4] capture-path pre-flight passed: {checklist}')

    def hook(planning, sed, models, rows, report, out_dir, label):
        G.write_interface_settlement_detail_s31c(planning, sed, models, rows, report, out_dir, label)

    report, report_path = G.run_admm_arm(
        'preflight', OUT_DIR, k_override=None, eval_id='p515s31c_preflight',
        num_max_iters_override=1, post_run_hook=hook)

    print(f'[S31C Part 4] arm report: {report_path}')
    print(f"[S31C Part 4] cycles_run={report['cycles_run']} "
          f"local_solve_failures={report['local_solve_failures']} "
          f"solve_profile={report['solve_profile']} "
          f"network_failures={report['network_failures_summary']}")

    # Print T_TSO / T_DSO / their sum / the priced consensus residual, per the
    # Planner task's Part 4 reporting requirement.
    import json
    detail_path = os.path.join(OUT_DIR, 'interface_settlement_detail_s31c.json')
    with open(detail_path) as handle:
        detail = json.load(handle)
    print(f"[S31C Part 4] t_tso_total={detail['t_tso_total']:.6f} "
          f"t_dso_by_node={detail['t_dso_by_node']} "
          f"t_tso_plus_t_dso_terminal={detail['t_tso_plus_t_dso_terminal']:.6f}")
    total_priced_residual_weighted = sum(
        v['sum_pi_baseMVA_residual_weighted'] for v in detail['interface_consensus_residual_per_dso'].values()
    )
    total_priced_residual_unweighted = sum(
        v['sum_pi_baseMVA_residual_unweighted'] for v in detail['interface_consensus_residual_per_dso'].values()
    )
    print(f"[S31C Part 4] total_priced_consensus_residual_weighted={total_priced_residual_weighted:.6f} "
          f"(comparable to t_tso_plus_t_dso_terminal) "
          f"total_priced_consensus_residual_unweighted={total_priced_residual_unweighted:.6f}")


if __name__ == '__main__':
    main()
