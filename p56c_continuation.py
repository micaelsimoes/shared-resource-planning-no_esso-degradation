"""
Stage P5.6-C4 -- deterministic continuation fallback.

P5.6-B3 showed that a candidate the T0 start cannot polish can sometimes be
solved from a cold start.  A solver failure is therefore NOT evidence that the
candidate is physically infeasible, and treating it as a hidden constraint
silently removes reachable parts of the feasible polyhedron from the search.

The first-stage feasible set is a polyhedron, so for a master-feasible target x
and the canonical base x0 the whole segment

    x(lambda) = (1 - lambda) * x0 + lambda * x

is first-stage feasible.  This walks a FIXED schedule along it,

    lambda = 0.25, 0.50, 0.75, 1.00

starting from the frozen T0 base state and using each VALID intermediate state to
initialize the next.  Every point uses the original nonlinear ESSO, the production
ADMM, the exact midpoint polish and the complete feasibility audit.  There is no
anchor switching and no adaptive schedule, so the path is identical every time the
same target is evaluated.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p56c_continuation.py
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

import p56a_oracle as O  # noqa: E402
import p56b_candidates as BC  # noqa: E402
import p56b_policy as P  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P56C')
LAMBDAS = (0.25, 0.50, 0.75, 1.00)
POLICY = P.POLICY_MIDPOINT_ONLY

# the three T0/midpoint POLISH_FAILURE candidates and the SOLVER_CRASH candidate
# from P5.6-B2, plus two T0-direct VALID controls
TARGETS = [
    ('se|node5|2025|-10%', 'B2 POLISH_FAILURE'),
    ('se|node9|2025|-10%', 'B2 POLISH_FAILURE'),
    ('se|ALL|-10%', 'B2 POLISH_FAILURE'),
    ('se|ALL|x19 (budget boundary)', 'B2 SOLVER_CRASH'),
    ('e|node5|2025|+25%', 'control, T0-direct VALID'),
    ('se|node9|2025|+10%', 'control, T0-direct VALID'),
]


def blend(x0, x, lam):
    return {key: {'s': (1.0 - lam) * x0[key]['s'] + lam * x[key]['s'],
                  'e': (1.0 - lam) * x0[key]['e'] + lam * x[key]['e']}
            for key in x0}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        provenance, planning_gate = gate('P5.6-C4 continuation', OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[C4] ABORTED\n{error}')
        sys.exit(1)

    population = dict(BC.population(planning_gate))
    x0 = population['base']

    report = {'stage': 'P5.6-C4', 'provenance': provenance,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'lambda_schedule': list(LAMBDAS), 'policy': POLICY,
              'targets': {}}
    out_path = os.path.join(OUT_DIR, 'p56c_continuation.json')

    def persist():
        with open(out_path, 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    print('[C4] building the frozen T0 base state ...', flush=True)
    started = time.time()
    t0 = O.build_fixed_template(verbose=False)
    print(f'[C4] T0 built in {time.time() - started:.1f}s\n', flush=True)

    for label, origin in TARGETS:
        if label not in population:
            print(f'[C4] {label}: not in the population, skipped')
            continue
        target = population[label]
        print(f'[C4] === {label}   ({origin})', flush=True)
        entry = {'origin': origin, 'steps': [], 'total_runtime_s': 0.0}
        state = t0
        final = None
        for lam in LAMBDAS:
            x_lam = blend(x0, target, lam)
            eid = f'c4_{label[:16].replace("|", "_").replace(" ", "")}_{lam:.2f}'
            r = P.evaluate(x_lam, template_state=state, anchor_policy=POLICY,
                           eval_id=eid, keep_state=True)
            step = {
                'lambda': lam, 'status': r['status'],
                'total_objective': r.get('total_objective'),
                'wall_clock_s': r['wall_clock_s'],
                'admm_cycles': (r.get('admm') or {}).get('cycles'),
                'admm_converged': (r.get('admm') or {}).get('converged'),
                'failed_blocks': r.get('failed_blocks'),
                'master_feasible': r.get('master_feasible'),
            }
            if r['status'] == O.STATUS_VALID:
                step['coordination_residuals'] = r['coordination_residuals']
                step['esso_max_violation'] = r['esso_audit']['max_violation']
                step['esso_production_violation'] = \
                    r['esso_audit']['production_feasibility_violation']
                step['network_max_violation'] = r['network_audit']['max_violation']
                step['network_h1_violation'] = \
                    r['network_audit']['max_h1_complementarity_violation']
                step['physical_salvage'] = r['physical_salvage']
                step['investment_cost'] = r['investment_cost']
                state = r['_state']
                final = r
            entry['steps'].append(step)
            entry['total_runtime_s'] += r['wall_clock_s']
            print(f"      lambda={lam:.2f}  {r['status']:16s} "
                  f"Q={r.get('total_objective')} "
                  f"cycles={(r.get('admm') or {}).get('cycles')} "
                  f"t={r['wall_clock_s']:.0f}s", flush=True)
            persist()
            if r['status'] != O.STATUS_VALID:
                entry['stopped_at_lambda'] = lam
                break
        entry['rescued'] = (final is not None
                            and entry['steps'][-1]['lambda'] == 1.0
                            and entry['steps'][-1]['status'] == O.STATUS_VALID)
        entry['final_total_objective'] = (final['total_objective']
                                          if entry['rescued'] else None)
        report['targets'][label] = entry
        persist()
        print(f"      -> rescued={entry['rescued']} "
              f"Q={entry['final_total_objective']} "
              f"total {entry['total_runtime_s']:.0f}s\n", flush=True)

    # ---- compare against direct T0 evaluation on the controls ---------------
    b2 = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P56B',
                      'p56b_b2b3_policy.json')
    if os.path.exists(b2):
        with open(b2) as handle:
            direct = json.load(handle)
        lookup = {r['label']: r.get('anchors', {}).get('midpoint', {})
                  for r in direct['B2_rows']}
        comparison = {}
        for label, entry in report['targets'].items():
            d = lookup.get(label, {})
            if d.get('status') == O.STATUS_VALID and entry.get('rescued'):
                comparison[label] = {
                    'direct_T0': d['total_objective'],
                    'continuation': entry['final_total_objective'],
                    'continuation_minus_direct': (entry['final_total_objective']
                                                  - d['total_objective']),
                }
        report['control_comparison'] = comparison
        print('[C4] continuation versus direct T0 on already-valid controls')
        for label, v in comparison.items():
            print(f"      {label:32s} direct={v['direct_T0']:.4f} "
                  f"cont={v['continuation']:.4f} "
                  f"diff={v['continuation_minus_direct']:+.4f}")
    persist()

    rescued = [l for l, e in report['targets'].items()
               if e.get('rescued') and 'FAIL' in e['origin'] or
               (e.get('rescued') and 'CRASH' in e['origin'])]
    report['rescued_previously_failing'] = rescued
    persist()
    print(f'\n[C4] previously failing candidates rescued: {rescued}')
    print(f'[C4] report -> {out_path}')


if __name__ == '__main__':
    main()
