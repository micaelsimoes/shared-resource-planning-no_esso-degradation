"""P5.12-R fresh R0 baseline generator (R0_v2), written by the Planner 2026-09-11.

DEFINITION (explicit; no equivalence to the missing original R0 generator is
claimed):
  tracked_hashes  = {path: SHA-256 of the working-tree bytes of path}
                    for every path printed by `git ls-files` (sorted).
  approved_diff   = `git diff` (unstaged tracked modifications) at creation.
  staged_diff     = `git diff --cached` at creation (required empty).
  head / branch   = `git rev-parse HEAD` / `git branch --show-current`.
  upstream        = `git rev-parse --abbrev-ref @{u}`; divergence =
                    `git rev-list --left-right --count HEAD...@{u}` (no fetch).
  harness_sha256  = SHA-256 of p512_r_presolve_recapture.py (must equal the
                    authorized value below).
The procedure is deterministic given the repository and runtime state: it only
reads git/working-tree state, runs p54r_provenance.gate(), hashes files and
writes new files in a new directory with exclusive-create mode. It aborts on
any precondition failure and never overwrites anything.

Run from the repository root with the canonical interpreter and -B.
"""
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation')
PY = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
R0_DIR = ROOT / 'data/SRP1/Results/P512R/R0_v2_ba202e2b'
HARNESS = 'p512_r_presolve_recapture.py'
HARNESS_SHA = 'f0f120c26ec2c50b774ff42051c233b87fba0341e3d959faafe70283301d86f0'
EXPECTED_HEAD = 'ba202e2b0e937306f3c163de2173951c1d0c24f0'
EXPECTED_BRANCH = 'feature/derivative-free-planning'
APPROVED_MODIFIED = {'REVISION_CONTEXT.md', 'LOCAL_NLP_STABILITY_PLAN.md'}
OLD_R0 = ROOT / 'data/SRP1/Results/P512R'
OLD_R0_FILES = {
    'data/SRP1/Results/P512R/provenance.json': '9d654a0012f3dd66b08b26e73dbadd8e39b244afcd68e5ef44addd372b977e5a',
    'data/SRP1/Results/P512R/initial_repository.json': '9c96d7de348d63e2411c9019cb7f3c21de440e3e8f601d6e3798556277b4a17b',
    'data/SRP1/Results/P512R/runtime_identity.json': '1e715f1dd565e603d231036d9cc6249cee63c531d7a4f281d3bb4f3f4316b791',
    'data/SRP1/Results/P512R/accepted_hashes.json': 'dce5c7e6532565c6843345198923aa590c258d345023a3e36886e738f0b0046c',
}
P512C = ('P5_12_C_BOUND_MULTIPLIER_AB_REPORT.md',
         '36812879af01aa3cd549b62cdb424af33a8dd080866041398a930c942e9a24b5')
REHEARSAL_SUMMARY = 'data/SRP1/Results/P512R_REHEARSAL/20260911T160420Z/rehearsal_summary.json'
# Commits between the old R0 HEAD and the current HEAD, and the only files they may touch.
RECONCILED_COMMITS = {
    'd20220dd': 'governing documents (bytes already hashed as approved edits by old R0) + CLAUDE.md',
    '8b83b139': 'adds .claude/agents/*.md and .claude/settings.json (agent configuration)',
    'ba202e2b': 'CLAUDE.md only',
}
ALLOWED_COMMIT_FILES = {'CLAUDE.md', 'REVISION_CONTEXT.md', 'LOCAL_NLP_STABILITY_PLAN.md',
                        '.claude/agents/advisor.md', '.claude/agents/planner.md',
                        '.claude/agents/worker.md', '.claude/settings.json'}
EXPECTED_ADDED = {'.claude/agents/advisor.md', '.claude/agents/planner.md',
                  '.claude/agents/worker.md', '.claude/settings.json'}
RUNTIME = {'python': '3.11.11', 'numpy': '2.4.2', 'pandas': '3.0.0', 'scipy': '1.17.0',
           'pyomo': '6.9.5', 'copulas': '0.14.0', 'machine': 'arm64'}


def fail(msg):
    print('[R0_v2] ABORT:', msg, flush=True)
    sys.exit(3)


def req(cond, msg):
    if not cond:
        fail(msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=str(ROOT), capture_output=True, text=True,
                          check=True).stdout


def write_json(name, value):
    with (R0_DIR / name).open('x') as f:
        json.dump(value, f, indent=1, default=str)


def main():
    req(Path.cwd() == ROOT, 'cwd must be the repository root')
    req(sys.executable == PY, 'wrong interpreter: ' + sys.executable)
    sys.path.insert(0, str(ROOT))
    req(not R0_DIR.exists(), 'R0_v2 directory already exists; refusing to overwrite')

    # --- preconditions: harness, historical R0, git state -------------------
    req(sha(ROOT / HARNESS) == HARNESS_SHA, 'harness SHA-256 is not the authorized value')
    for p, h in OLD_R0_FILES.items():
        req(sha(ROOT / p) == h, 'historical R0 file changed: ' + p)
    req(sorted(x.name for x in OLD_R0.iterdir()) ==
        sorted(Path(p).name for p in OLD_R0_FILES), 'P512R root contains unexpected entries')
    head = git('rev-parse', 'HEAD').strip()
    branch = git('branch', '--show-current').strip()
    req(head == EXPECTED_HEAD, 'HEAD differs: ' + head)
    req(branch == EXPECTED_BRANCH, 'branch differs: ' + branch)
    staged = git('diff', '--cached')
    req(staged == '', 'staged changes present')
    modified = {l[3:] for l in git('status', '--porcelain', '--untracked-files=no').splitlines() if l}
    req(modified == APPROVED_MODIFIED, 'unexpected tracked modifications: ' + str(sorted(modified)))
    commit_files = set(git('diff', '--name-only', 'dd0001675e4e38cbf4ec0282df312469bcb435e5', 'HEAD').split())
    req(commit_files <= ALLOWED_COMMIT_FILES, 'intervening commits touch unexpected files: ' +
        str(sorted(commit_files - ALLOWED_COMMIT_FILES)))
    commits = git('rev-list', '--abbrev-commit', 'dd0001675e4e38cbf4ec0282df312469bcb435e5..HEAD').split()
    req(set(c[:8] for c in commits) == set(RECONCILED_COMMITS), 'intervening commit set differs: ' + str(commits))
    try:
        upstream = git('rev-parse', '--abbrev-ref', '@{u}').strip()
        divergence = git('rev-list', '--left-right', '--count', 'HEAD...@{u}').strip()
    except subprocess.CalledProcessError:
        upstream, divergence = None, None

    files = sorted(f for f in git('ls-files').splitlines() if f)
    tracked_hashes = {f: sha(ROOT / f) for f in files}
    req(tracked_hashes[HARNESS] if HARNESS in tracked_hashes else True, 'unreachable')
    req(HARNESS not in tracked_hashes, 'harness unexpectedly tracked')

    # --- reconciliation with the old R0 -------------------------------------
    old = json.loads((OLD_R0 / 'initial_repository.json').read_text())['tracked_hashes']
    added = sorted(set(tracked_hashes) - set(old))
    removed = sorted(set(old) - set(tracked_hashes))
    changed = sorted(k for k in tracked_hashes if k in old and tracked_hashes[k] != old[k])
    req(set(added) == EXPECTED_ADDED, 'added tracked files differ: ' + str(added))
    req(not removed, 'tracked files removed: ' + str(removed))
    req(set(changed) <= {'CLAUDE.md', 'REVISION_CONTEXT.md', 'LOCAL_NLP_STABILITY_PLAN.md'},
        'unexpected changed tracked files vs old R0: ' + str(changed))

    # --- accepted evidence ---------------------------------------------------
    old_accepted = json.loads((OLD_R0 / 'accepted_hashes.json').read_text())
    accepted = {}
    for p, item in old_accepted.items():
        accepted[p] = {'expected': item['expected'], 'actual': sha(ROOT / p), 'source': 'old R0 accepted list'}
    accepted[P512C[0]] = {'expected': P512C[1], 'actual': sha(ROOT / P512C[0]),
                          'source': 'REVISION_CONTEXT.md 2026-09-11'}
    for p, h in OLD_R0_FILES.items():
        accepted[p] = {'expected': h, 'actual': sha(ROOT / p), 'source': 'historical R0 (REVISION_CONTEXT.md 2026-09-11)'}
    rh = sha(ROOT / REHEARSAL_SUMMARY)
    accepted[REHEARSAL_SUMMARY] = {'expected': rh, 'actual': rh,
                                   'source': 'accepted final rehearsal; hash fixed at R0_v2 creation'}
    for p, item in accepted.items():
        req(item['expected'] == item['actual'], 'accepted artifact hash mismatch: ' + p)
    rehearsal = json.loads((ROOT / REHEARSAL_SUMMARY).read_text())
    req(rehearsal['items']['harness_sha256_start'] == HARNESS_SHA and
        rehearsal['items']['harness_sha256_end'] == HARNESS_SHA and rehearsal['status'] == 'COMPLETED',
        'accepted rehearsal does not correspond to the authorized harness bytes')

    # --- runtime identity (asserted) ----------------------------------------
    import importlib
    versions = {}
    for m in ('numpy', 'pandas', 'scipy', 'pyomo', 'copulas'):
        versions[m] = importlib.import_module(m).__version__
    ident = {'python': platform.python_version(), 'machine': platform.machine(), **versions}
    for k, v in RUNTIME.items():
        req(ident.get(k) == v, f'runtime identity mismatch {k}: {ident.get(k)} != {v}')

    # --- write the baseline (exclusive-create, new directory) ---------------
    R0_DIR.mkdir(parents=False, exist_ok=False)
    from p54r_provenance import gate, ProvenanceError
    try:
        provenance, _ = gate('P5.12-R R0_v2 baseline', str(R0_DIR), verbose=True)
    except ProvenanceError as e:
        fail('provenance gate failed: ' + str(e))
    req(provenance.get('gate_passes') is True, 'provenance gate did not pass')
    req(provenance.get('git_head') == head, 'gate HEAD differs')
    req(provenance.get('checksum_matches_canonical') is True, 'scenario checksum not canonical')

    write_json('runtime_identity.json', {'versions': versions, 'python': sys.executable,
                                         'python_version': ident['python'],
                                         'python_sha256': sha(sys.executable),
                                         'architecture': ident['machine']})
    write_json('accepted_hashes.json', accepted)
    generator_copy = R0_DIR / 'r0_v2_generator.py'
    shutil.copyfile(__file__, generator_copy)
    initial = {
        'baseline_id': 'R0_v2_ba202e2b',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'root': str(ROOT),
        'head': head, 'branch': branch, 'upstream': upstream, 'divergence': divergence,
        'status': git('status', '--porcelain', '--branch', '--untracked-files=no'),
        'recent': git('log', '--oneline', '-12'),
        'tracked_path_count': len(files),
        'tracked_hashes': tracked_hashes,
        'approved_diff': git('diff'),
        'approved_modified_files': sorted(modified),
        'staged_diff': staged,
        'harness_path': HARNESS, 'harness_sha256': HARNESS_SHA,
        'definition': __doc__,
        'generator_sha256': sha(generator_copy),
        'reconciliation': {
            'previous_r0': 'data/SRP1/Results/P512R/initial_repository.json (HEAD dd000167)',
            'intervening_commits': RECONCILED_COMMITS,
            'files_touched_by_intervening_commits': sorted(commit_files),
            'tracked_added_vs_previous_r0': added,
            'tracked_removed_vs_previous_r0': removed,
            'tracked_changed_vs_previous_r0': changed,
            'production_py_or_parameter_changes': [f for f in commit_files if f.endswith('.py') or f.startswith('data/')],
        },
    }
    write_json('initial_repository.json', initial)
    sums = {p.name: sha(p) for p in sorted(R0_DIR.iterdir()) if p.is_file()}
    write_json('SHA256SUMS.json', sums)
    print('[R0_v2] CREATED', R0_DIR, flush=True)
    print(json.dumps(sums, indent=1), flush=True)


if __name__ == '__main__':
    main()
