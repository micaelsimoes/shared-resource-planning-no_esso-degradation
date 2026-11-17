"""
WORKER T4 -- capacity-ladder initialization at 1.00 MVA / 4.00 MWh (node 7),
via p514_l_capacity_ladder.main, OUT redirected into data/SRP1/Results/P515F/
(never touching the committed data/SRP1/Results/P514L/ladder_s1.json this
task must not overwrite -- CLAUDE.md evidence rule).

Zero-solve check (PLANNER_BRIEF_2026-09-13.md Addendum 5): from the new
per-solve init logs this fix's item 1/2 produces, compute
mu_final/(2*s_obj*eps) per node and compare against the measured detector
value already on record (P5_15_EXPERT_HANDOFF_3.md / WORKER_REPORT_G1_G4.md:
2.4031e-05 absolute, global, at 1.00 MVA). Also reads the tol actually in
force from each log's own header.
"""

import io
import json
import os
import re
import sys
from contextlib import redirect_stdout

REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p514_l_capacity_ladder as L  # noqa: E402
import shared_energy_storage_data as SED  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515F')
os.makedirs(OUT, exist_ok=True)

# Redirect the ladder harness's own report-JSON output away from the committed
# data/SRP1/Results/P514L tree (which a committed report cites) -- never
# re-run a harness onto cited evidence.
original_out = L.OUT
L.OUT = os.path.join(OUT, 't4_ladder')
os.makedirs(L.OUT, exist_ok=True)

MEASURED_GLOBAL_ABSOLUTE_REFERENCE = 2.4031e-05  # WORKER_REPORT_G1_G4.md / P5_15_EXPERT_HANDOFF_3.md, 1.00 MVA rung
MEASURED_RATIO_REFERENCE_APPROX = 2.5e-05  # Addendum 5's own approximation of the above


def main():
    try:
        rc = L.main(1.00)
    finally:
        L.OUT = original_out

    # The ESSO init logs land under the oracle eval's own logs dir for this
    # ladder rung (fresh_planning('ladder_s1') -> logs_dir), NOT under L.OUT
    # (which only ever held the report JSON, never the ESSO logs, even before
    # this fix -- see finding below).
    esso_logs_dir = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P56A', 'evals', 'ladder_s1', 'logs')
    init_logs = sorted(
        f for f in os.listdir(esso_logs_dir)
        if f.startswith('optim_log_esso_node') and f.endswith('_init.txt')
    )

    eps_in_force = SED.EPS_ESSO_THROUGHPUT
    per_node = {}
    for filename in init_logs:
        match = re.match(r'optim_log_esso_node(\d+)_init\.txt', filename)
        node_id = int(match.group(1))
        log_path = os.path.join(esso_logs_dir, filename)
        mu_final, s_obj, parse_reason = SED._parse_ipopt_barrier_terms(log_path)

        text = open(log_path, 'r', errors='replace').read()
        # IPOPT's option-value echo line has the form "tol                             = 1e-08 ..."
        # (present when print_level default reports the option summary block).
        tol_match = re.search(r'^\s*tol\s*=\s*(\S+)', text, re.MULTILINE)
        acceptable_tol_match = re.search(r'^\s*acceptable_tol\s*=\s*(\S+)', text, re.MULTILINE)
        n_iterations_match = re.findall(r'^\s*(\d+)\s+[\deE.+\-]+\s+[\deE.+\-]+', text, re.MULTILINE)

        predicted = None
        if mu_final is not None and s_obj not in (None, 0.0) and eps_in_force:
            predicted = mu_final / (2.0 * s_obj * eps_in_force)

        per_node[node_id] = {
            'log_path': log_path,
            'tol_in_force_from_log_header': tol_match.group(1) if tol_match else None,
            'acceptable_tol_in_force_from_log_header': (
                acceptable_tol_match.group(1) if acceptable_tol_match else None),
            'mu_final': mu_final,
            's_obj': s_obj,
            'parse_reason': parse_reason,
            'eps_esso_throughput_in_force': eps_in_force,
            'predicted_mu_over_2_s_obj_eps': predicted,
            'measured_reference_absolute_global_2_4031e-05': MEASURED_GLOBAL_ABSOLUTE_REFERENCE,
            'measured_reference_ratio_approx_2_5e-05': MEASURED_RATIO_REFERENCE_APPROX,
            'predicted_over_measured_ratio_approx': (
                predicted / MEASURED_RATIO_REFERENCE_APPROX if predicted is not None else None),
        }

    report = {
        'stage': 'P5.15-F T4',
        'ladder_main_return_code': rc,
        'esso_logs_dir': esso_logs_dir,
        'init_log_files_found': init_logs,
        'eps_esso_throughput_in_force': eps_in_force,
        'per_node': per_node,
        'note_pre_fix_capture_gap': (
            'Under the pre-fix code, this same eval directory '
            '(data/SRP1/Results/P56A/evals/ladder_s1/logs/) contains ZERO '
            'optim_log_node_* or optim_log_esso_* files from any prior run -- '
            'confirmed by directory listing before this run. The ESSO never '
            'had logs_dir awareness (this fix item 1), so its bare relative '
            'output_file resolved against whatever the process cwd happened '
            'to be at solve time, never into the oracle eval logs directory. '
            'This is the first run under which per-node ESSO init logs are '
            'captured in the structured per-eval location at all.'
        ),
    }
    with open(os.path.join(OUT, 't4_ladder_zero_solve_check.json'), 'w') as handle:
        json.dump(report, handle, indent=2, default=str)

    print('LADDER_MAIN_RC', rc)
    print('ESSO_LOGS_DIR', esso_logs_dir)
    print('INIT_LOG_FILES', init_logs)
    print('EPS_IN_FORCE', eps_in_force)
    for node_id, row in sorted(per_node.items()):
        print(f'NODE {node_id}:', json.dumps(row, indent=2, default=str))


if __name__ == '__main__':
    main()
