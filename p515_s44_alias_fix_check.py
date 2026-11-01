"""P5.15 Addendum 25 item 1 (Task 1) -- zero-solve, synthetic-input evidence that the
alias tie-break fix in `p515_g_g1_g4_admm_gates.s34_capture_hooks` makes the ordering
of `objective_component_block_deltas` deterministic across `PYTHONHASHSEED` values.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 25 ("Alias tie-break non-determinism:
fix (sort by name, not over a set)"); frozen spec v14 `data/SRP1/Results/P515S44/
frozen_s44_selection_spec_v14_e4500e27.json` item1_harness.alias_fix.

Background (WORKER_REPORT_S43_AA_PREP.md, Unexpected Finding 1): the recourse-jump
capture hook built `objective_component_block_deltas` by iterating
`set(flat_current) | set(flat_previous)` (tuples of `(block_key, component_name)`)
then a STABLE sort on `abs_delta` alone. `'economic_market_cost'` and
`'generation_cost'` are two always-numerically-identical aliases of the same block
objective component (`shared_resources_planning.py:5158`), so their `abs_delta` ties
EXACTLY and the tie was broken by `set` iteration order, which depends on Python's
per-process string-hash randomization (`PYTHONHASHSEED`, unset in production). The fix
(committed alongside this check) sorts by `(-abs_delta, block_key, component)` -- a
total order by name, independent of set iteration order.

This script exercises the REAL, unmodified `s34_capture_hooks` wrapper (never a
reimplementation): only the two block-decomposition production functions it calls
(`srp._get_operational_recourse_block_components`,
`srp._get_operational_objective_component_blocks`) and the function it wraps
(`srp.get_admm_boyd_residual_metrics`) are monkeypatched with tiny synthetic
stand-ins for the duration of the check, so the wrapper's OWN sort/tie-break code runs
unchanged on controlled synthetic data. Zero Pyomo models are built; zero solves occur
(nothing here touches `OptSolver` or `SystemCallSolver`, so no `SolveProfileGuard` is
even needed to prove it).

======================================================================
EXACT LAUNCH COMMAND
======================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python \\
        p515_s44_alias_fix_check.py \\
        > data/SRP1/Results/P515S44/alias_fix_check_launch.log 2>&1

Runs itself as a child subprocess once per PYTHONHASHSEED value in `HASH_SEEDS` (hash
randomization is fixed at interpreter start, so this cannot be done in one process),
each child exercising the wrapper on a synthetic two-cycle input where the alias pair's
delta is deliberately the LARGEST of three synthetic components (so both entries land
in the reported top-10 and their relative order is the fact under test).
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PYTHON = sys.executable
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44')
RESULTS_PATH = os.path.join(OUT_DIR, 'alias_fix_determinism_check.json')

# Arbitrary, fixed in advance -- not tuned to any observed outcome.
HASH_SEEDS = ['0', '1', '2', '42', '100', '999', '12345', '777', '55']

_CHILD_MARKER = '--child-run'


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _run_child_check():
    """Runs INSIDE a child process with a fixed PYTHONHASHSEED (set by the parent
    before Python starts). Exercises the real `s34_capture_hooks` wrapper on
    synthetic input and prints one JSON line with the resulting order."""
    import p515_g_g1_g4_admm_gates as G  # noqa: E402
    import shared_resources_planning as srp  # noqa: E402

    tmpdir = tempfile.mkdtemp(prefix='p515s44_alias_fix_')
    recourse_jump_path = os.path.join(tmpdir, 'recourse_jump.jsonl')
    ess_stride_path = os.path.join(tmpdir, 'ess_stride.jsonl')

    # Two cycles of synthetic block decomposition. The alias pair
    # ('economic_market_cost', 'generation_cost') is ALWAYS numerically identical
    # (by construction here, mirroring the real invariant) and its delta (40.0) is
    # the LARGEST of the three synthetic components -- both entries tie exactly on
    # abs_delta and must appear first, in a name-determined order.
    cycle_recourse_blocks = [{}, {}]  # empty: isolates the fact under test
    cycle_obj_blocks = [
        {'blockA': {'economic_market_cost': 10.0, 'generation_cost': 10.0, 'other_x': 3.0}},
        {'blockA': {'economic_market_cost': 50.0, 'generation_cost': 50.0, 'other_x': 3.5}},
    ]
    call_state = {'i': 0}

    def fake_recourse_blocks(planning_problem, operational_models):
        return cycle_recourse_blocks[call_state['i']]

    def fake_obj_blocks(planning_problem, operational_models):
        return cycle_obj_blocks[call_state['i']]

    def fake_real_fn(*args, **kwargs):
        return 'STUB_BOYD_RESULT'

    class _FakePlanning:
        active_distribution_network_nodes = []

    orig_recourse_fn = srp._get_operational_recourse_block_components
    orig_obj_fn = srp._get_operational_objective_component_blocks
    orig_wrapped = srp.get_admm_boyd_residual_metrics
    srp._get_operational_recourse_block_components = fake_recourse_blocks
    srp._get_operational_objective_component_blocks = fake_obj_blocks
    srp.get_admm_boyd_residual_metrics = fake_real_fn
    try:
        with G.s34_capture_hooks(recourse_jump_path, ess_stride_path, stride=10 ** 9):
            for i in range(2):
                call_state['i'] = i
                srp.get_admm_boyd_residual_metrics(
                    planning_problem=_FakePlanning(), tso_model=None, dso_models=None,
                    esso_model={}, consensus_vars={'ess': {'z': {'current': {}}}},
                    dual_vars=None, admm_parameters=None)
    finally:
        srp._get_operational_recourse_block_components = orig_recourse_fn
        srp._get_operational_objective_component_blocks = orig_obj_fn
        srp.get_admm_boyd_residual_metrics = orig_wrapped

    with open(recourse_jump_path) as handle:
        rows = [json.loads(line) for line in handle]
    deltas = rows[1]['objective_component_block_deltas']
    order = [d['component'] for d in deltas]
    print(json.dumps({
        'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
        'full_order': order,
        'tied_pair_order': [c for c in order if c in ('economic_market_cost', 'generation_cost')],
        'abs_deltas': [[d['component'], d['abs_delta']] for d in deltas],
    }))


def main():
    _refuse_overwrite(RESULTS_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    per_seed = []
    for seed in HASH_SEEDS:
        env = dict(os.environ)
        env['PYTHONHASHSEED'] = seed
        proc = subprocess.run([PYTHON, __file__, _CHILD_MARKER], env=env,
                               capture_output=True, text=True, cwd=REPO, check=True)
        line = proc.stdout.strip().splitlines()[-1]
        per_seed.append(json.loads(line))
        print(f'[ALIAS-FIX-CHECK] seed={seed}: {line}')

    orders = {tuple(r['tied_pair_order']) for r in per_seed}
    deterministic = (len(orders) == 1) and (len(per_seed) == len(HASH_SEEDS))
    expected_order = ['economic_market_cost', 'generation_cost']
    matches_name_order = deterministic and list(next(iter(orders))) == expected_order

    payload = {
        'stage': 'P5.15 Addendum 25 item 1 (Task 1) -- alias tie-break fix determinism check',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 25',
            'data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json',
            'WORKER_REPORT_S43_AA_PREP.md (Unexpected Finding 1)',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'fix': ('p515_g_g1_g4_admm_gates.py s34_capture_hooks: '
                "obj_deltas.sort(key=lambda e: (-e['abs_delta'], e['block_key'], e['component']))"),
        'hash_seeds_tested': HASH_SEEDS,
        'per_seed_results': per_seed,
        'distinct_tied_pair_orders_observed': [list(o) for o in orders],
        'deterministic_across_seeds': deterministic,
        'order_matches_alphabetical_component_name': matches_name_order,
    }
    with open(RESULTS_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[ALIAS-FIX-CHECK] wrote {RESULTS_PATH}')
    print(f'[ALIAS-FIX-CHECK] deterministic_across_seeds={deterministic} '
          f'order_matches_alphabetical_component_name={matches_name_order}')
    if not deterministic:
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == _CHILD_MARKER:
        _run_child_check()
    else:
        main()
