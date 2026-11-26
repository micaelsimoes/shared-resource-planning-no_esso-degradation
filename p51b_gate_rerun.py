"""
P5.1-B full-initialization gate re-run with a corrected result enumerator.

The first gate pass computed the authoritative production verdict correctly
(`_admm_local_solves_succeeded`), but its result walker mis-descended into
Pyomo `SolverResults` (a dict-like MapContainer), so the per-solve identities
were lost. This script re-runs ONLY the gate, with the same in-memory cap, and
enumerates every DSO/TSO/ESSO initialization solve properly.

Diagnostic only: the cap is applied by temporarily wrapping
`model_construction_helpers.shared_ess_snet_def_scale` inside this process and
restoring it afterwards. No production file is modified. ADMM and the outer
planning loop are not entered.

    python p51b_gate_rerun.py [--cap 1000]
"""

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P51B')


def enumerate_results(container, prefix, out):
    """Pyomo SolverResults is dict-like, so test for a results object FIRST."""
    if hasattr(container, 'solver'):
        out[prefix] = {
            'status': str(getattr(container.solver, 'status', None)),
            'termination': str(getattr(container.solver, 'termination_condition', None)),
            'succeeded': bool(srp._solver_result_succeeded(container)),
        }
        return out
    try:
        items = container.items()
    except AttributeError:
        out[prefix] = {'unparsed': type(container).__name__}
        return out
    for key, value in items:
        enumerate_results(value, f'{prefix}/{key}', out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap', type=float, default=1000.0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    original_scale = mch.shared_ess_snet_def_scale

    def capped_scale(s_capacity):
        return min(original_scale(s_capacity), args.cap)

    quiet = io.StringIO()
    report = {'stage': 'P5.1-B gate re-run', 'cap': args.cap,
              'cap_applied_in_memory_only': True,
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    try:
        mch.shared_ess_snet_def_scale = capped_scale
        with redirect_stdout(quiet):
            planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
            planning.read_planning_problem()
            candidate = srp._build_positive_bootstrap_candidate(
                planning, planning.params.benders.positive_bootstrap)
            consensus_vars, _dual = srp.create_admm_variables(planning)
            res = {'tso': dict(), 'dso': dict(), 'esso': dict()}
            _dso_models, res['dso'] = srp.create_distribution_networks_models(
                planning.distribution_networks, consensus_vars,
                candidate['total_capacity'],
                parallel_execution=planning.parallel_execution)
            _tso_model, res['tso'] = srp.create_transmission_network_model(
                planning, consensus_vars, candidate['total_capacity'])
            _esso_model, res['esso'] = srp.create_shared_energy_storage_model(
                planning.shared_ess_data, consensus_vars, candidate['investment'])
            all_ok = srp._admm_local_solves_succeeded(planning, res)
    finally:
        mch.shared_ess_snet_def_scale = original_scale

    solves = {}
    for agent in ('dso', 'tso', 'esso'):
        enumerate_results(res[agent], agent, solves)
    failures = {k: v for k, v in solves.items() if not v.get('succeeded', False)}

    by_agent = {}
    for key, value in solves.items():
        agent = key.split('/')[0]
        entry = by_agent.setdefault(agent, {'total': 0, 'failures': 0})
        entry['total'] += 1
        if not value.get('succeeded', False):
            entry['failures'] += 1

    report.update({
        'total_local_solves': len(solves),
        'n_failures': len(failures),
        'failures': failures,
        'per_agent': by_agent,
        'all_solves': solves,
        'admm_local_solves_succeeded': bool(all_ok),
        'would_enter_admm': bool(all_ok),
    })

    out_path = os.path.join(OUT_DIR, f'p51b_gate_cap{int(args.cap)}_report.json')
    with open(out_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f'[gate] cap={args.cap} -> {out_path}')
    print(f"[gate] total local solves = {report['total_local_solves']} | per agent = {by_agent}")
    print(f"[gate] failures = {report['n_failures']}")
    for key, value in failures.items():
        print(f"    FAIL {key}: {value['status']}/{value['termination']}")
    print(f"[gate] _admm_local_solves_succeeded = {report['admm_local_solves_succeeded']}")
    print(f"[gate] would enter ADMM = {report['would_enter_admm']}")


if __name__ == '__main__':
    main()
