"""S56 2x phase -- the author's / Planner's check of ONE phase launch record (spec v2, first-block hold; Planner
decision R2/P2, TASKS.md step 5a, 2026-11-24). Read-only, standard library only, zero solves: it imports nothing from
the repository, so its command line and its process carry no harness module name.

Reads <phase root>/launches/launch_<NNN>.json (default NNN = 001) and prints, per evaluation: label, depth, status,
read_cycle, the guards, the child wall, and PASS/FAIL against
  certified                (summary status == 'certified')
  read_cycle < 300         (an exit run that certified before the cap)
  all guards true          (every value of summary.guards is True, and there is at least one)
  wall 0.7-1.6 h           (summary.wall_s / 3600, the child process wall)
It also prints the launch wall and whether the launch manifest holds the record's sha256.

Usage:  python s56_phase_check.py <phase root> [--launch N] [--data-root DIR]
Exit:   0 every evaluation PASS; 1 at least one FAIL; 2 the record (or its manifest) is missing or unreadable.
"""
import argparse
import hashlib
import json
import os
import sys

READ_CYCLE_LT = 300
WALL_H_MIN, WALL_H_MAX = 0.7, 1.6


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def label_of(rec):
    return os.path.basename(rec.get('eval_dir') or '')[17:] or str(rec.get('z'))


def check_record(rec):
    """One launch-record entry -> (row dict, list of failed criteria)."""
    s = rec.get('summary') or {}
    guards = s.get('guards')
    wall_s = s.get('wall_s')
    wall_h = wall_s / 3600.0 if isinstance(wall_s, (int, float)) and not isinstance(wall_s, bool) else None
    rc = s.get('read_cycle')
    failed = []
    if s.get('status') != 'certified':
        failed.append('certified')
    if not (isinstance(rc, int) and not isinstance(rc, bool) and rc < READ_CYCLE_LT):
        failed.append(f'read_cycle < {READ_CYCLE_LT}')
    if not (isinstance(guards, dict) and guards and all(v is True for v in guards.values())):
        failed.append('all guards true')
    if not (wall_h is not None and WALL_H_MIN <= wall_h <= WALL_H_MAX):
        failed.append(f'wall {WALL_H_MIN}-{WALL_H_MAX} h')
    row = {'label': label_of(rec), 'depth': rec.get('depth'), 'status': s.get('status'), 'read_cycle': rc,
           'guards': guards, 'wall_h': wall_h, 'F': s.get('F'), 'bar': s.get('bar'),
           'verdict': 'PASS' if not failed else 'FAIL', 'failed': failed}
    return row, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('root', help='the phase root (absolute, or relative to --data-root)')
    ap.add_argument('--launch', type=int, default=1)
    ap.add_argument('--data-root', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args(argv)
    root = args.root if os.path.isabs(args.root) else os.path.join(args.data_root, args.root)
    name = f'launch_{args.launch:03d}'
    rec_p = os.path.join(root, 'launches', name + '.json')
    man_p = os.path.join(root, 'launches', name + '_manifest_sha256.json')
    try:
        with open(rec_p) as handle:
            lr = json.load(handle)
    except (OSError, ValueError) as error:
        print(f'MISSING / UNREADABLE launch record {rec_p}: {error}')
        return 2
    rel = os.path.relpath(rec_p, root)
    try:
        with open(man_p) as handle:
            man = json.load(handle)
        manifest_ok = man.get(rel) == sha256_file(rec_p)
    except (OSError, ValueError) as error:
        print(f'MISSING / UNREADABLE launch manifest {man_p}: {error}')
        return 2
    act = lr.get('action') or {}
    wall = lr.get('wall_s')
    lwall = f'{wall / 3600.0:.3f} h' if isinstance(wall, (int, float)) and not isinstance(wall, bool) else 'None'
    print(f"phase {lr.get('phase_id')} launch {lr.get('launch_index')}: stage {act.get('stage')} purpose "
          f"{act.get('purpose')} depth {act.get('depth')} x{len(act.get('zs') or [])}; launch wall {lwall}; "
          f"started {lr.get('started_utc')} finished {lr.get('finished_utc')}")
    print(f"record sha256 {sha256_file(rec_p)}; manifest holds it: {manifest_ok}")
    print(f'criteria: certified; read_cycle < {READ_CYCLE_LT}; all guards true; wall {WALL_H_MIN}-{WALL_H_MAX} h')
    n_fail = 0 if manifest_ok else 1
    recs = lr.get('records') or []
    if not recs:
        print('FAIL: the launch record holds no evaluation')
        n_fail += 1
    for rec in recs:
        row, failed = check_record(rec)
        n_fail += bool(failed)
        g = row['guards']
        gtxt = (', '.join(f'{k}={v}' for k, v in sorted(g.items())) if isinstance(g, dict) else repr(g))
        wtxt = f"{row['wall_h']:.3f} h" if row['wall_h'] is not None else 'None'
        print(f"{row['verdict']}  {row['label']}  depth {row['depth']}  status {row['status']}  read_cycle "
              f"{row['read_cycle']}  wall {wtxt}  F {row['F']}  bar {row['bar']}")
        print(f'      guards: {gtxt}')
        if failed:
            print(f'      failed: {failed}')
    print(f"overall: {'PASS' if n_fail == 0 else 'FAIL'} ({len(recs)} evaluation(s); manifest ok {manifest_ok})")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
