"""
Stage P5.10-F -- full oracle replay, CURRENT production against the stabilized
rescaled oracle.

The CURRENT arm is NOT re-run here. P5.9-D already ran production's own oracle
for these three candidates over eight warm-start generations from frozen T0, and
that arm reproduced the accepted P5.6-D base chain to the digit at every
generation. Re-running it would consume an hour to reproduce a number that has
now been reproduced four times. It is read from `p59_d_replay.json`, which is
production as it actually is -- inherited history included, which is the point of
the comparison.

The RESCALED arm runs the P5.10 stabilized configuration: RESCALED objective,
adaptive penalty disabled, rho fixed, and the history channels neutralised AT
EVERY GENERATION, so that the declared stopping rule stays in force down the
whole chain instead of drifting into the inherited one after generation 1.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p510_f_replay.py <candidate>
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56b_candidates as BC  # noqa: E402
import p510_oracle as OR  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

GENERATIONS = 8
RHO_PF = 1000.0
CANDIDATES = ['base', 'se|node5|2025|-10%', 'se|node9|2025|-10%']


def neutralize(state):
    """Re-apply the declared history policy to a chain state."""
    for key in OR.HISTORY_KEYS:
        state[key] = 0 if key == 'consecutive_converged_cycles' else None
    state['candidate_solution'] = None
    state['initialization_failed'] = False
    return state


def main():
    os.makedirs(OR.OUT_DIR, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tag = (only or 'all').replace('|', '_').replace('%', 'pct').replace(' ', '_')
    out_path = os.path.join(OR.OUT_DIR, f'p510_f_replay_{tag}.json')

    try:
        provenance, planning_gate = gate('P5.10-F oracle replay', OR.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.10] ABORTED\n{error}')
        sys.exit(1)

    config = OR.OracleConfig(
        scaling_mode=OR.SCALING_RESCALED, rho_v=1.5, rho_pf=RHO_PF,
        rho_ess=1.0, adaptive_penalty=False, neutralize_history=True,
        notes='P5.10 stabilized oracle')

    population = dict(BC.population(planning_gate))
    report = {'stage': 'P5.10-F', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'config': config.as_dict(), 'config_hash': config.config_hash,
              'generations': GENERATIONS,
              'current_arm_source': ('data/SRP1/Results/P59/p59_d_replay.json '
                                     '-- production as it is, not re-run'),
              'chains': {}}

    def persist():
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    for name in CANDIDATES:
        if only is not None and name != only:
            continue
        safe = name.replace('|', '_').replace('%', 'pct').replace(' ', '_')
        print(f'\n[P5.10-F] {name} -- stabilized rescaled oracle', flush=True)
        state, template_info = OR.prepare_template(config)
        chain, previous = [], None
        for j in range(1, GENERATIONS + 1):
            case_id = f'f_{safe}_j{j}'
            record, new_state = OR.evaluate(
                population[name], config, case_id=case_id, template_state=state)
            row = OR.row(record)
            row['generation'] = j
            objective = row.get('total_objective')
            row['step_delta'] = (objective - previous) if (
                objective is not None and previous is not None) else None
            chain.append(row)
            print(f"      gen {j}: {row['status']:16s} cycles={row['admm_cycles']} "
                  f"pre-polish={row['admm_net_recourse_before_polish']} "
                  f"Q={objective} fail={row['n_failed_polish_blocks']} "
                  f"step={row['step_delta']}", flush=True)
            if objective is not None:
                previous = objective
            if new_state is None:
                row['chain_stopped_here'] = True
                break
            state = neutralize(new_state)
        report['chains'][name] = {'template_info': template_info,
                                  'generations': chain}
        persist()

    print(f'\n[P5.10-F] report -> {out_path}')


if __name__ == '__main__':
    main()
