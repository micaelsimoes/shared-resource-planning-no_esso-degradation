"""P5.15 Addendum 38 (task W39) -- NL-FILE IDENTITY PROBE at ONE scenario. ZERO SOLVES.

Authority: frozen spec v21 `data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json`,
`gates[0]` (the SRP1 two-cycle bitwise identity), which this probe DE-RISKS but does not
replace.

THE RISK THIS PROBE ADDRESSES. Row 18 is structurally absent at one scenario, and every
alias it introduces resolves to the identity there -- but `build_objective` now adds TWO
named Expressions to EVERY model it builds, at one scenario as well:

    model.interface_settlement_contracted
    model.interface_settlement_deviation

They are REPORTING quantities: `objective_function_rule` does not reference them and no
constraint does. The bitwise gate's claim therefore rests on an implementation detail of
Pyomo's NL writer -- that a named Expression reached by no active constraint and no active
objective is not emitted. If that were false, the SRP1 two-cycle bitwise gate would fail
for a reason that has nothing to do with the formulation, and the remedy would be to build
the two Expressions only above one scenario.

WHAT IS COMPARED. One freshly built 1 x 1 DSO block and one 1 x 1 TSO block, each written
to an `.nl` file twice:
  A  as production builds it (both Expressions present);
  B  after `del_component`-ing exactly those two Expressions and nothing else.
The two files must be BYTE-IDENTICAL (and their `.row`/`.col` symbol files too). That is
the whole probe. Writing an `.nl` is not a solve: `SolveProfileGuard(permitted=())` is
armed for the whole run and verified at exactly 0.

WHAT IT DOES NOT ESTABLISH, stated rather than implied: this probe compares the CURRENT
code against itself with two components removed. It does NOT compare against the
pre-Addendum-38 code, and it says nothing about the ADMM path's dual/warm-start state.
Only the SRP1 two-cycle bitwise gate settles that.

EXACT COMMAND (repo root, canonical interpreter, attached, BOTH streams captured):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s51_nl_identity_probe.py \
      > data/SRP1/Results/P515S51/nl_identity_probe_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S51/nl_identity_probe/nl_identity_probe.json
Exit 0 when the files are byte-identical, 1 otherwise.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W39 NL identity probe (never solves)').install()

import p515_s44_scale_measurement as S  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

STAGE = 'P5.15 Addendum 38 (W39) -- NL-file identity probe at one scenario'
SCHEMA = 'p515_s51_nl_identity_probe_v1'
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S51', 'nl_identity_probe')
OUT_PATH = os.path.join(OUT_DIR, 'nl_identity_probe.json')
REPORTING_ONLY_EXPRESSIONS = ('interface_settlement_contracted', 'interface_settlement_deviation')


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    try:
        return subprocess.run(['git'] + args, capture_output=True, text=True,
                              check=True, cwd=REPO).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f'<git failed: {error}>'


def _write_nl(model, path):
    """Write the model exactly as the solver interface would, symbolic labels ON so the
    comparison is over names as well as structure. Writing is not solving."""
    model.write(path, io_options={'symbolic_solver_labels': True})
    return path


def probe_block(tag, model, work_dir):
    before_path = _write_nl(model, os.path.join(work_dir, f'{tag}_A_as_built.nl'))
    present = {name: hasattr(model, name) for name in REPORTING_ONLY_EXPRESSIONS}
    for name in REPORTING_ONLY_EXPRESSIONS:
        if hasattr(model, name):
            model.del_component(getattr(model, name))
    after_path = _write_nl(model, os.path.join(work_dir, f'{tag}_B_without_reporting_expressions.nl'))

    entry = {
        'reporting_expressions_present_as_built': present,
        'nl_A_sha256': _sha256_file(before_path),
        'nl_B_sha256': _sha256_file(after_path),
        'nl_A_bytes': os.path.getsize(before_path),
        'nl_B_bytes': os.path.getsize(after_path),
    }
    entry['nl_identical'] = entry['nl_A_sha256'] == entry['nl_B_sha256']
    for suffix in ('row', 'col'):
        a = before_path[:-3] + f'.{suffix}'
        b = after_path[:-3] + f'.{suffix}'
        if os.path.exists(a) and os.path.exists(b):
            entry[f'{suffix}_A_sha256'] = _sha256_file(a)
            entry[f'{suffix}_B_sha256'] = _sha256_file(b)
            entry[f'{suffix}_identical'] = entry[f'{suffix}_A_sha256'] == entry[f'{suffix}_B_sha256']
    entry['pass'] = all(v for k, v in entry.items() if k.endswith('_identical'))
    return entry


def main():
    if os.path.exists(OUT_PATH):
        print(f'REFUSED: output exists (write-once): {OUT_PATH}', file=sys.stderr)
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    work_dir = os.path.join(OUT_DIR, 'work')
    os.makedirs(work_dir, exist_ok=True)

    case, spec, changes = S.derive_case('srp1', {'years': {'2025': 5},
                                                 'num_market_scenarios': None,
                                                 'num_operation_scenarios': None})
    case_path = os.path.join(work_dir, 'SRP1__1x1.json')
    with open(case_path, 'w') as handle:
        json.dump(case, handle, indent='\t')

    planning = SharedResourcesPlanning(S.DATA_DIR, os.path.relpath(case_path, S.DATA_DIR))
    planning.name = 'SRP1'
    planning.results_dir = os.path.join(work_dir, 'Results')
    planning.diagrams_dir = os.path.join(work_dir, 'Diagrams')
    planning.logs_dir = os.path.join(planning.results_dir, 'Logs')
    planning.read_planning_problem()

    blocks = {}
    node_id = next(iter(planning.distribution_networks))
    dso = planning.distribution_networks[node_id]
    year = next(iter(dso.years))
    day = next(iter(dso.days))
    dso_models = dso.build_model()
    blocks[f'DSO:{node_id}:{dso.network[year][day].name}:{year}:{day}'] = probe_block(
        'dso', dso_models[year][day], work_dir)

    tso = planning.transmission_network
    tyear = next(iter(tso.years))
    tday = next(iter(tso.days))
    tso_models = tso.build_model()
    blocks[f'TSO:{tso.network[tyear][tday].name}:{tyear}:{tday}'] = probe_block(
        'tso', tso_models[tyear][tday], work_dir)

    GUARD.uninstall()
    verify_failures = GUARD.verify(expected_solves=0)
    all_pass = all(entry['pass'] for entry in blocks.values()) and not verify_failures

    payload = {
        'schema': SCHEMA, 'stage': STAGE,
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 38',
                      'data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json gates[0]'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'interpreter': sys.executable,
        'script': os.path.basename(__file__),
        'script_sha256': _sha256_file(os.path.abspath(__file__)),
        'git_head': _git(['rev-parse', 'HEAD']),
        'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
        'instance': {'definition': spec, 'changes_vs_source': changes,
                     'case_sha256': _sha256_file(case_path),
                     'note': 'ONE market x ONE operation scenario, one representative year'},
        'reporting_only_expressions_removed_in_arm_B': list(REPORTING_ONLY_EXPRESSIONS),
        'blocks': blocks,
        'does_not_establish': ('this compares the CURRENT code against itself with two '
                               'components removed; it does NOT compare against the '
                               'pre-Addendum-38 code, which only the SRP1 two-cycle bitwise '
                               'gate settles'),
        'solve_profile_guard': {'permitted': [], 'counts': dict(GUARD.counts),
                                'verify_failures': verify_failures},
        'all_checks_pass': bool(all_pass),
    }
    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    for key, entry in blocks.items():
        print(f'  {key}: nl_identical={entry["nl_identical"]} '
              f'A={entry["nl_A_sha256"][:12]} B={entry["nl_B_sha256"][:12]}')
    print(f'[W39 NL probe] all_checks_pass = {all_pass}; guard = {dict(GUARD.counts)}')
    print(f'[W39 NL probe] wrote {OUT_PATH}')
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
