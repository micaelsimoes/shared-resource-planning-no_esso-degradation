"""P5.15 S56 -- clean export for the D_b re-solve and the T8 binding / T9 solves (zero solves; no project import).

Writes a `git archive` export of EXPORT_COMMIT into a NEW directory (it must not exist), restricted to:
  every tracked path outside data/SRP1/Results, plus the declared data/SRP1/Results inputs (--results-path, repeatable);
then overlays the stage files (--overlay, repeatable) taken byte for byte from OVERLAY_COMMIT (`git show`), refusing
any overlay path that is tracked at EXPORT_COMMIT or already exists in the export (overlays are additions only), and
writes EXPORT_PROVENANCE.json at the export root: export commit, overlay commit, the sha256 of every exported root-level
file, of every overlay file and of every exported results input. The stage scripts check themselves, their spec and
every imported root module against this record before running.

Usage (from the main checkout):
  <canonical python> p515_s56_db_t8_export.py --commit 4c586e5e --overlay-commit <C> --dest <scratch dir> \\
      --overlay p515_s56_db_tn_resolve.py --overlay <spec> ... --results-path <path> ...
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))


def _git(*args, binary=False):
    out = subprocess.run(['git', *args], cwd=REPO, capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--commit', required=True)
    ap.add_argument('--overlay-commit', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--overlay', action='append', default=[])
    ap.add_argument('--results-path', action='append', default=[])
    a = ap.parse_args()
    if os.path.exists(a.dest):
        print(f'REFUSE: {a.dest} exists', flush=True)
        return 1
    commit = _git('rev-parse', a.commit).strip()
    overlay_commit = _git('rev-parse', a.overlay_commit).strip()
    # two archives: an exclude pathspec wins over an include in one call, so the results inputs come separately
    pathspecs = [['.', ':(exclude)data/SRP1/Results'], list(a.results_path)]
    blobs = [_git('archive', '--format=tar', commit, '--', *ps, binary=True) for ps in pathspecs if ps]
    for rel in a.overlay:
        if subprocess.run(['git', 'cat-file', '-e', f'{commit}:{rel}'], cwd=REPO, capture_output=True).returncode == 0:
            print(f'REFUSE: overlay {rel} is tracked at {commit} (overlays are additions only)', flush=True)
            return 2
    os.makedirs(a.dest)
    for blob in blobs:
        with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
            tf.extractall(a.dest)
    overlay = {}
    for rel in a.overlay:
        dst = os.path.join(a.dest, rel)
        if os.path.exists(dst):
            print(f'REFUSE: overlay {rel} exists in the export', flush=True)
            return 2
        data = _git('show', f'{overlay_commit}:{rel}', binary=True)
        os.makedirs(os.path.dirname(dst) or a.dest, exist_ok=True)
        with open(dst, 'xb') as h:
            h.write(data)
        overlay[rel] = {'sha256': _sha(dst), 'from_commit': overlay_commit}
    root_files = {f: _sha(os.path.join(a.dest, f)) for f in sorted(os.listdir(a.dest))
                  if os.path.isfile(os.path.join(a.dest, f))}
    results = {}
    for rp in a.results_path:
        base = os.path.join(a.dest, rp)
        if os.path.isfile(base):
            results[rp] = _sha(base)
        else:
            for dp, _d, fs in os.walk(base):
                for f in fs:
                    p = os.path.join(dp, f)
                    results[os.path.relpath(p, a.dest)] = _sha(p)
    prov = {'export_commit': commit, 'overlay_commit': overlay_commit, 'utc': datetime.now(timezone.utc).isoformat(),
            'pathspecs': pathspecs, 'exported_root_files_sha256': root_files, 'overlay': overlay,
            'results_inputs_sha256': results, 'n_results_inputs': len(results),
            'export_script': os.path.basename(__file__), 'export_script_sha256': _sha(os.path.abspath(__file__))}
    with open(os.path.join(a.dest, 'EXPORT_PROVENANCE.json'), 'x', encoding='utf-8') as h:
        json.dump(prov, h, indent=1, sort_keys=True)
    print(f'export {commit} -> {a.dest}; overlay {len(overlay)} files from {overlay_commit}; '
          f'{len(root_files)} root files; {len(results)} results inputs', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
