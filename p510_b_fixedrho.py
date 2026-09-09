"""
Stage P5.10-B -- fixed-rho rescaled ADMM validation.

RESCALED objective only. Adaptive rho DISABLED, because P5.9-B established that
production's rule is self-cancelling after rescaling: the dual residual is
`rho * |z_current - z_prev| / base` (`shared_resources_planning.py:4897, 4938`),
linear in rho, so raising rho trips the rule's own decrease branch and every
requested penalty decays as `requested / 1.5^n`. The redesign is audited in
stage D and deliberately not attempted here.

`rho_v = 1.5` and `rho_ess = 1.0` are held fixed and not swept: P5.9-A measured
`rho_v` inert over a 667x range (voltage consensus never within 32x of its
tolerance) and ESS consensus never within 175x of its tolerance on any of 96
cycles.

Every evaluation runs under a fully explicit `OracleConfig` with history
neutralised, so the base candidate and the perturbed candidates are evaluated
under the SAME stopping rule -- which stage A shows is not true of production.

The selection criterion is stability, NOT the lowest objective. It is declared
here before any result exists:

    1. the direct evaluation must succeed with zero failed polish blocks for
       ALL THREE candidates (no continuation requirement);
    2. among configurations meeting 1, prefer the smallest spread of
       `Delta(x) = Q(x) - Q(base)` across candidates relative to its own mean,
       i.e. the configuration whose candidate differences are best resolved;
    3. ties broken by lower total runtime.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p510_b_fixedrho.py <rho_pf>
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

RHO_PF = [300.0, 500.0, 1000.0]
CANDIDATES = ['base', 'se|node5|2025|-10%', 'se|node9|2025|-10%']


def main():
    os.makedirs(OR.OUT_DIR, exist_ok=True)
    only = float(sys.argv[1]) if len(sys.argv) > 1 else None
    tag = f'pf{only:g}' if only else 'all'
    out_path = os.path.join(OR.OUT_DIR, f'p510_b_fixedrho_{tag}.json')

    try:
        provenance, planning_gate = gate('P5.10-B fixed rho', OR.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.10] ABORTED\n{error}')
        sys.exit(1)

    population = dict(BC.population(planning_gate))
    report = {'stage': 'P5.10-B', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'scaling_mode': 'RESCALED', 'adaptive_penalty': False,
              'rho_v_fixed': 1.5, 'rho_ess_fixed': 1.0,
              'rho_pf_values': RHO_PF,
              'history_neutralised': True,
              'selection_criterion': (
                  '1) zero failed polish blocks for all three candidates; '
                  '2) smallest relative spread of Delta(x); 3) lower runtime'),
              'rows': []}

    def persist():
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    for rho_pf in RHO_PF:
        if only is not None and rho_pf != only:
            continue
        config = OR.OracleConfig(
            scaling_mode=OR.SCALING_RESCALED, rho_v=1.5, rho_pf=rho_pf,
            rho_ess=1.0, adaptive_penalty=False, neutralize_history=True,
            notes='P5.10-B fixed-rho validation')
        for name in CANDIDATES:
            safe = name.replace('|', '_').replace('%', 'pct').replace(' ', '_')
            case_id = f'b_pf{rho_pf:g}_{safe}'
            print(f'[P5.10-B] rho_pf={rho_pf:g}  {name} ...', flush=True)
            record, _ = OR.evaluate(population[name], config, case_id=case_id)
            row = OR.row(record)
            row.update({'candidate': name, 'rho_pf': rho_pf})
            cycles = record.get('cycle_detail') or []
            last = cycles[-1] if cycles else {}
            row.update({
                'primal_v': last.get('primal_v'),
                'primal_pf': last.get('primal_pf'),
                'primal_ess': last.get('primal_ess'),
                'dual_v_mean': last.get('dual_v_mean'),
                'dual_pf_mean': last.get('dual_pf_mean'),
                'dual_ess_mean': last.get('dual_ess_mean'),
                'worst_pf_primal_difference': last.get('worst_pf_primal_difference'),
                'base_objective_prepolish': record.get(
                    'admm_base_objective_before_polish'),
                'continuation_required': record.get('status') != 'VALID',
            })
            row['cycle_detail'] = cycles
            report['rows'].append(row)
            persist()
            print(f"          cycles={row['admm_cycles']} "
                  f"pre-polish={row['admm_net_recourse_before_polish']} "
                  f"polished={row['total_objective']} "
                  f"fail={row['n_failed_polish_blocks']} "
                  f"primal_pf={row['primal_pf']} "
                  f"runtime={row['wall_clock_s']:.0f}s", flush=True)

    print(f'\n[P5.10-B] report -> {out_path}')


if __name__ == '__main__':
    main()
