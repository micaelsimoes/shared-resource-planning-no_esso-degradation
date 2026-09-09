"""
Stage P5.9-D -- full rescaled oracle replay, CURRENT versus RESCALED+stabilized.

For each candidate both chains start from the SAME frozen T0 primal state and
run the same warm-start refinement the accepted oracle uses, so generation j of
one chain is directly comparable to generation j of the other.  The only
differences are the network subproblem objective scaling and rho.

NOTE ON COMPARABILITY WITH P5.6-D.  P5.6-D's `H_K` is UNIFORM REFINEMENT: it
interpolates capacity from `x0` toward the candidate in K steps.  At `x0` that
is by construction pure self-refinement and the two coincide, which is why the
base chain here reproduces P5.6-D exactly.  Away from `x0` they do not, so
non-base H_2/H_4/H_8 values from P5.6-D are quoted as context, never as a
like-for-like control.  The like-for-like control is the CURRENT chain run here.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p59_d_replay.py
"""

import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import p56b_candidates as BC  # noqa: E402
import p56a_oracle as O  # noqa: E402
import p59_eval as EV  # noqa: E402
from p54r_provenance import ProvenanceError, gate  # noqa: E402

OUT_PATH = os.path.join(EV.OUT_DIR, 'p59_d_replay.json')

# One candidate per process.  P5.6-B7 measured four upper-level workers as the
# safe maximum (eight risked the IPOPT process crashes already observed under
# heavy concurrency), so three is inside the documented ceiling.  Concurrency
# affects wall time only: every solve is deterministic given its inputs, and
# each process loads its OWN copy of T0 and its own deep copy of the planning
# problem, so no state is shared between them.
A_PATH = os.path.join(EV.OUT_DIR, 'p59_a_sweep.json')
GENERATIONS = 8
PLANNING_SIGNAL = 33031.0
TAU_NUMERICAL = 10.0

CANDIDATES = [
    ('base', 'canonical base x0'),
    ('se|node5|2025|-10%', 'improving candidate; P5.6-D best known incumbent'),
    ('se|node9|2025|-10%', 'previously problematic; direct-T0 POLISH_FAILURE '
                           'under CURRENT, rescued only by continuation (P5.7-D)'),
]

# accepted CURRENT base chain -- reproduced by P5.7 (4.8e-07) and P5.8-A0 (exact)
CURRENT_BASE_CHAIN = {1: 828021090.360850, 2: 827415318.563944,
                      3: 826824028.845478, 4: 826405022.193437,
                      5: 825961521.882321, 6: 825531306.746985,
                      7: 825108709.695181, 8: 824795363.718628}


def run_chain(x, label, rescale, rho, generations=GENERATIONS):
    """One warm-start refinement chain from a private copy of T0."""
    state, info = EV.load_template(rescale=rescale, rho=rho)
    chain, previous = [], None
    for j in range(1, generations + 1):
        case_id = f'{label}_j{j}'
        record, new_state = EV.evaluate(
            x, template_state=state, case_id=case_id, rho=rho,
            archive_label=case_id)
        row = EV.row(record)
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
        state = new_state
    return {'template_info': info, 'generations': chain}


def summarize_chain(chain):
    rows = chain['generations']
    objectives = [r['total_objective'] for r in rows
                  if r.get('total_objective') is not None]
    deltas = [r['step_delta'] for r in rows if r.get('step_delta') is not None]
    return {
        'n_generations_run': len(rows),
        'n_valid': sum(1 for r in rows if r['status'] == O.STATUS_VALID),
        'n_polish_failures': sum(1 for r in rows
                                 if (r.get('n_failed_polish_blocks') or 0) > 0),
        'generation_1_polish_clean': (rows[0].get('n_failed_polish_blocks') == 0
                                      if rows else None),
        'H2': objectives[1] if len(objectives) > 1 else None,
        'H4': objectives[3] if len(objectives) > 3 else None,
        'H8': objectives[7] if len(objectives) > 7 else None,
        'first_objective': objectives[0] if objectives else None,
        'last_objective': objectives[-1] if objectives else None,
        'total_drift': (objectives[-1] - objectives[0]) if len(objectives) > 1 else None,
        'step_deltas': deltas,
        'final_step_delta': deltas[-1] if deltas else None,
        'reaches_planning_signal_at_generation': next(
            (i + 2 for i, d in enumerate(deltas) if abs(d) <= PLANNING_SIGNAL), None),
        'reaches_tau_numerical_at_generation': next(
            (i + 2 for i, d in enumerate(deltas) if abs(d) <= TAU_NUMERICAL), None),
        'admm_to_polish_gap': [r.get('admm_to_polish_improvement') for r in rows],
    }


def main():
    os.makedirs(EV.OUT_DIR, exist_ok=True)
    os.makedirs(EV.ARCHIVE_DIR, exist_ok=True)
    only = None
    if len(sys.argv) > 1:
        only = sys.argv[1]
        globals()['OUT_PATH'] = os.path.join(
            EV.OUT_DIR,
            'p59_d_replay_%s.json' % only.replace('|', '_')
                                        .replace('%', 'pct').replace(' ', '_'))
    try:
        provenance, planning_gate = gate('P5.9-D oracle replay', EV.OUT_DIR)
    except ProvenanceError as error:
        print(f'\n[P5.9] ABORTED\n{error}')
        sys.exit(1)

    with open(A_PATH) as handle:
        a_report = json.load(handle)
    best_row = a_report.get('best_case_row')
    if not best_row:
        print('[P5.9-D] stage A has no selected configuration; not continuing.')
        sys.exit(1)
    best_rho = best_row['rho_requested']
    print(f"[P5.9-D] stage A selected {a_report['best_case']} -> rho={best_rho}\n",
          flush=True)

    population = dict(BC.population(planning_gate))
    report = {
        'stage': 'P5.9-D', 'provenance': provenance,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'best_configuration_from_A': a_report['best_case'],
        'best_rho': best_rho,
        'generations': GENERATIONS,
        'planning_signal': PLANNING_SIGNAL,
        'current_base_chain_reference': CURRENT_BASE_CHAIN,
        'candidates': {}}

    def persist():
        with open(globals()['OUT_PATH'], 'w') as handle:
            json.dump(report, handle, indent=1, default=str)

    persist()
    for name, note in CANDIDATES:
        if only is not None and name != only:
            continue
        x = population[name]
        entry = {'note': note}
        report['candidates'][name] = entry
        safe = name.replace('|', '_').replace('%', 'pct').replace(' ', '_')

        print(f'\n[P5.9-D] {name} -- CURRENT formulation', flush=True)
        entry['CURRENT'] = run_chain(x, f'd_current_{safe}', rescale=False,
                                     rho=None)
        entry['CURRENT_summary'] = summarize_chain(entry['CURRENT'])
        persist()

        print(f'\n[P5.9-D] {name} -- RESCALED + rho={best_rho}', flush=True)
        entry['RESCALED'] = run_chain(x, f'd_rescaled_{safe}', rescale=True,
                                      rho=best_rho)
        entry['RESCALED_summary'] = summarize_chain(entry['RESCALED'])
        persist()

    print('\n[P5.9-D] summary')
    for name, entry in report['candidates'].items():
        c, r = entry['CURRENT_summary'], entry['RESCALED_summary']
        print(f'\n   {name}')
        print(f"      {'':22} {'CURRENT':>20} {'RESCALED':>20}")
        for key in ('generation_1_polish_clean', 'n_polish_failures', 'H2',
                    'H4', 'H8', 'total_drift', 'final_step_delta',
                    'reaches_planning_signal_at_generation'):
            print(f'      {key:22} {str(c.get(key)):>20} {str(r.get(key)):>20}')
    print(f"\n[P5.9-D] report -> {globals()['OUT_PATH']}")


if __name__ == '__main__':
    main()
