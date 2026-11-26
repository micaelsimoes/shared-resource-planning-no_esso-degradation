"""
P5.14-X22 — the two NEW warm cells of the 2x2, under C3, on the recovered AB1 candidate.

Frozen spec: data/SRP1/Results/P514X22/frozen_2x2_spec_v1_55b10d16.json

    python p514_h_warm_cells.py <cell>     # cell in {warm_fixed, warm_adaptive}

The cold cells already exist (AB1). Cap 50 for every cell, matching the cold ones, set by
patching num_max_iters before p510_oracle applies its own rho/adaptive overrides.

The warm fixed cell is NOT gated on P5.10-B's 6 cycles: P5.10-B is pre-C3 and a mismatch
is expected.
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p510_oracle as OR  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p56a_candidates as CAND  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P514X22')
SPEC = 'data/SRP1/Results/P514X22/frozen_2x2_spec_v1_55b10d16.json'
CAP = 50
SOLVES_PER_CYCLE = 51
HARD_LIMIT = 2000
PERMITTED = [('network.py', '_run_smopf_solver_attempt'),
             ('shared_energy_storage_data.py', '_run_solver_attempt')]
CELLS = {'warm_fixed': False, 'warm_adaptive': True}


def patch_cap():
    """Set the cap on every planning problem built during this run."""
    original = O.fresh_planning

    def patched(eval_id, *args, **kwargs):
        planning = original(eval_id, *args, **kwargs)
        planning.params.admm.num_max_iters = CAP
        return planning

    O.fresh_planning = patched
    return original


def main(cell):
    if cell not in CELLS:
        raise SystemExit(f'cell must be one of {sorted(CELLS)}')
    adaptive = CELLS[cell]
    os.makedirs(OUT, exist_ok=True)

    guard = SolveProfileGuard(PERMITTED, label=f'X22 {cell}').install()
    original_fresh = patch_cap()
    started = time.time()
    try:
        with redirect_stdout(io.StringIO()):
            planning = O.fresh_planning(f'x22_{cell}_candidate')
            x = CAND.base_vector(planning)          # same constructor as AB1's candidate
            config = OR.OracleConfig(
                scaling_mode=OR.SCALING_RESCALED, rho_v=1.5, rho_pf=300.0, rho_ess=1.0,
                adaptive_penalty=adaptive, neutralize_history=True,
                notes='P5.14-X22 warm cell under C3')
            template_state, template_info = OR.prepare_template(config)
            template_solves = guard.counts['permitted_solve']
            record, _state = OR.evaluate(x, config, case_id=f'x22_{cell}',
                                         template_state=template_state)
    finally:
        O.fresh_planning = original_fresh
        guard.uninstall()
    wall = time.time() - started

    cycles = record.get('cycle_detail') or []
    admm_solves = guard.counts['permitted_solve'] - template_solves
    identity_core = SOLVES_PER_CYCLE * len(cycles)
    report = {
        'stage': 'P5.14-X22', 'cell': cell, 'spec': SPEC,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'instance': {'candidate_vector': {str(k): v for k, v in x.items()},
                     'constructor': 'p56a_candidates.base_vector -> '
                                    'srp._build_positive_bootstrap_candidate',
                     'formulation': 'C3 ACTIVE, k = 11541.560327111707'},
        'config': config.as_dict(), 'config_hash': config.config_hash,
        'template_info': template_info, 'cap': CAP,
        'adaptive_penalty': adaptive,
        'admm_cycles': record.get('admm_cycles'),
        'admm_converged': record.get('admm_converged'),
        'terminating_criterion': record.get('terminating_criterion'),
        'admm_net_recourse_before_polish': record.get('admm_net_recourse_before_polish'),
        'polished_net_operational_recourse': record.get('polished_net_operational_recourse'),
        'cost_families': record.get('cost_families'),
        'rho_observed_final': record.get('rho_observed_final'),
        'n_adaptive_updates': record.get('n_adaptive_updates'),
        'adaptive_actions': record.get('adaptive_actions'),
        'polish_all_solved': record.get('polish_all_solved'),
        'failed_blocks': record.get('failed_blocks'),
        'cycle_detail': cycles,
        'wall_clock_s': wall,
        'solve_profile': {
            'observed': dict(guard.counts),
            'template_prep_solves': template_solves,
            'admm_phase_solves': admm_solves,
            'identity_51_x_cycles': identity_core,
            'residual_after_identity': admm_solves - identity_core,
            'residual_explained_as': ('one polish pass over 48 network blocks'
                                      if admm_solves - identity_core == 48 else
                                      'no polish pass' if admm_solves - identity_core == 0
                                      else 'UNEXPLAINED — must be accounted for in the report'),
            'hard_failures': ([f'blocked solves: {guard.counts["blocked_solve"]}']
                              if guard.counts['blocked_solve'] else [])
                             + ([f'total {guard.counts["permitted_solve"]} exceeds {HARD_LIMIT}']
                                if guard.counts['permitted_solve'] > HARD_LIMIT else []),
        },
    }
    out = os.path.join(OUT, f'x22_{cell}.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)
    print(f'[X22 {cell}] cycles={report["admm_cycles"]} converged={report["admm_converged"]} '
          f'recourse_prepolish={report["admm_net_recourse_before_polish"]} '
          f'rho_final={report["rho_observed_final"]} solves={guard.counts["permitted_solve"]} '
          f'(template {template_solves} + admm {admm_solves}) wall={wall:.0f}s -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else 'warm_fixed'))
