"""
P5.15 S56 step 1 COMPLETION -- G-a / G-b zero-solve NL builds re-run into NEW `_r3` subpaths (launcher only).

Why: the step-1 completion (Planner task 2026-11-23: harness-side sweep fixes, `p58_rescale` re-snapshot after the
per-scenario wiring, and author decision 6's Q convention in production) must leave every option-off model, and the
one-scenario option-on model, byte-identical. This launcher re-runs the frozen gate script's OWN zero-solve build
child (`p515_s56_coupling_gates.py --child-build`, unchanged, sha256 recorded in the spec
`frozen_coupling_gates_spec_v1_10c98258.json`) for a declared subset of its `BUILD_CONFIGS`, writing each child
record into `builds_<label>_r3/` (the committed `builds_<label>/` records are never touched), and compares every NL
hash (plain, labelled, .row, .col; initialisation and ADMM cycle 1) with the committed record of the same config and
label using the frozen script's own comparator (`_nl_equal`), plus its build checks and zero-solve guard.

Usage (repo root, canonical interpreter, attached, alone, both streams captured):
  NLP_SOLVER_PATH=/usr/local/bin/ipopt <python> -u p515_s56_coupling_gates_builds_r3.py --spec-sha256 <S> \
      --scratch <scratchpad dir> > <log> 2>&1
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s56_coupling_gates as G  # noqa: E402  (installs its zero-solve guard on this process)

# (config, label): the G-a off configurations on both instances and the G-b one-scenario option-on configuration,
# plus the G-f configuration (3 x 3, option on; its NL must also be unchanged -- the edits touch no model build).
R3_BUILDS = (('srp1_x0', 'post'), ('srp1grid_x0_a05', 'post'), ('srp1_x0_on', 'on'),
             ('srp1grid_x0_a05_on_v1_noq', 'on'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec-sha256', required=True)
    ap.add_argument('--scratch', required=True)
    args = ap.parse_args()
    if os.environ.get('NLP_SOLVER_PATH') != G.SOLVER_PATH:
        G._log(f'REFUSING: NLP_SOLVER_PATH must be {G.SOLVER_PATH}')
        return 1
    spec = G._find_spec(args.spec_sha256)
    scratch_root = os.path.abspath(args.scratch)
    results = {}
    for config, label in R3_BUILDS:
        if label not in G.BUILD_CONFIGS[config]['labels']:
            raise RuntimeError(f'{config} is not declared for label {label}')
        out_dir = os.path.join(G._gate_dir(), f'builds_{label}_r3')
        os.makedirs(out_dir, exist_ok=True)
        child_out = os.path.join(out_dir, f'{config}.json')
        if os.path.exists(child_out):
            raise RuntimeError(f'refusing to overwrite {child_out}')
        log_path = os.path.join(out_dir, f'{config}.log')
        cmd = [sys.executable, '-u', G.SCRIPT_NAME, '--child-build', '--config', config, '--label', label,
               '--child-out', child_out, '--scratch', os.path.join(scratch_root, f'{label}_r3_{config}')]
        G._log(f'build r3 {label}/{config}: {" ".join(cmd[2:])}')
        t0 = time.time()
        with open(log_path, 'x') as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        new = G._load(child_out) if os.path.isfile(child_out) else None
        ref = G._child(label, config)
        cmp = G._nl_equal(ref, new) if (ref and new) else {'identical': False, 'error': 'missing build'}
        ok_new, ok_ref = G._child_ok(new), G._child_ok(ref)
        failed = G._checks_all(new)
        results[f'{label}/{config}'] = {
            'exit_code': rc, 'wall_s': time.time() - t0, 'json': G._rel(child_out), 'log': G._rel(log_path),
            'reference': G._rel(os.path.join(G._gate_dir(), f'builds_{label}', f'{config}.json')),
            'reference_sha256': G._sha_file(os.path.join(G._gate_dir(), f'builds_{label}', f'{config}.json')),
            'nl_vs_reference': cmp, 'build': ok_new, 'reference_build': ok_ref,
            'structural_checks_failed': failed[:10],
            'coupling_active_in_child': (new or {}).get('coupling_active'),
            'holds': (rc == 0 and cmp.get('identical') is True and ok_new.get('build_checks_all') is True
                      and ok_new.get('guard_verified_zero') is True and not failed)}
        G._log(f"build r3 {label}/{config}: exit {rc}, NL identical {cmp.get('identical')} "
               f"({cmp.get('n_nl_compared')} compared), holds {results[f'{label}/{config}']['holds']}")
    res = {'schema': 'p515_s56_coupling_builds_r3_v1', 'spec': G._rel(spec['_path']), 'spec_sha256': spec['_sha256'],
           'utc': G._utc(), 'purpose': ('step 1 completion: option-off (and one-scenario option-on) NL identity '
                                        'after the sweep fixes and the Q-convention change'),
           'per_build': results, 'holds': all(r['holds'] for r in results.values()),
           'provenance': G.provenance(), 'parent_guard': G.guard_report()}
    path = os.path.join(G._gate_dir(), f'verdict_builds_r3_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")}.json')
    G._write_once_json(path, res)
    G._log(f"builds r3 holds {res['holds']} -> {G._rel(path)}")
    return 0 if res['holds'] else 2


if __name__ == '__main__':
    sys.exit(main())
