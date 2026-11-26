"""P5.15 S56 -- ZERO-SOLVE TESTS of the extended solve-profile guard (oracle mapping v2; TASKS.md step 5b, Planner
decision 2026-11-25: "extend the reconciliation exactly (ESSO recovery solves counted; every failure recovered) --
harness accounting, not formulation").

The 2x phase v2 stopped at launch 10 (exit 5, stop_for_review_not_settled). The failing check on every not_settled /
uncertified record was guards['solve_profile_identity_holds'], because the harness declares its identity unsupported
whenever an ESSO recovery event exists. `p515_s56_search_driver.solve_profile_reconciliation` (section comment there)
now counts 1 + [tier2_attempted] solves per ESSO event, requires every network failure recovered and every ESSO event
recorded as succeeded, and leaves the harness-identity path unchanged.

Guards (installed before any repository import): SolveProfileGuard(permitted=()), verify(0) at the end; pickle.load /
loads blocked and counted (0). Every record is READ ONLY; nothing is written outside --out.

TESTS
  E1  capture paths: `assert_mapping_capture_paths` holds on the harness as committed, including the three inputs the
      new guard reads (failure summary in the record, ESSO events file written and counted, production's event fields).
  E2  the 4 real records (v2 launches 3, 7, 9, 10): the pre-fix driver (git PRE_FIX_REV, asserted = the v2 spec's frozen
      driver hash) gives guard False and status not_certified / not_settled; the new guard holds via the extended
      identity, observed == base + credited + ESSO solves exactly, and the status is certified / settled.
  E3  synthetic failures, each built from a copy of a real record's files (the launch 7 record, 3 ESSO events): an
      unrecovered ESSO event; recovery_succeeded absent; observed + 1 and - 1 (off by one); a tier-2 attempt not paid
      for (and the positive control: paid for -> holds); the events file absent; one event dropped; an unrecovered /
      not-attempted / indeterminate network block; credited != tier1 + 2 tier2; reconciliation declared supported; an
      inconsistent tier record; identity False with no ESSO event (the harness path). Each fails, and the mapping then
      gives not_certified / not_settled.
  E4  the scan: every real record of the pilot root and the v2 phase root, at every depth it is read at, mapped by the
      pre-fix driver and by the new one. Every record that passed before still passes with an identical summary (only
      `mapping_schema` and the new `solve_profile_reconciliation` field differ); the records whose summary changes are
      exactly the 4 of E2. Also: every evaluation_record.json with a solve_profile under the S56 results tree, guard
      value only (old pass -> new pass).

Launch (attached, both streams captured):
  python -u p515_s56_esso_guard_tests.py --out <new dir> > <dir>_stdout.log 2> <dir>_stderr.log
Exit: 0 every check holds; 3 a check failed; 2 refused.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 ESSO guard tests (never solve)').install()
import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('ESSO guard tests: pickle.load is blocked')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('ESSO guard tests: pickle.loads is blocked')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import copy  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s56_search_driver as D  # noqa: E402
import p515_s56_search_policy as P  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
CHECKS = []
# HEAD when the fix was built: its driver is the driver the v2 phase spec froze (asserted against the spec's hash).
PRE_FIX_REV = '2d46d3c8'
RESULTS_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56')
V2_ROOT_REL = os.path.join(RESULTS_REL, 'phase2x_ev_v2')
PILOT_ROOT_REL = os.path.join(RESULTS_REL, 'pilot_ev_v1')
# The 4 records of the Planner's diagnosis (v2 launch, eval dir prefix, depth, labels)
AFFECTED = (
    ('launch_003', '4193f7f81fbaa8de', 'exit', 'n9_y2025_p0.25_e0.5'),
    ('launch_007', '643d2614a24412b1', 'settled', 'n5_y2030_p1_e3.5 + n5_y2035_p0.25_e0.5'),
    ('launch_009', '862c0cdc1636f348', 'settled', 'n7_y2025_p0.25_e0.5'),
    ('launch_010', 'a3337e25e72fe9ce', 'settled', 'n5_y2025_p0.25_e0.5 + n7_y2025_p0.25_e1'),
)
CHANGED_ONLY = ('mapping_schema', 'solve_profile_reconciliation')


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def ck(test, name, ok, detail=None):
    CHECKS.append({'test': test, 'check': name, 'ok': bool(ok), 'detail': detail})
    log(f'{"OK  " if ok else "FAIL"} [{test}] {name}' + (f' -- {str(detail)[:600]}' if detail is not None and not ok
                                                          else ''))
    return bool(ok)


def load_pre_fix_driver(out_dir):
    src = subprocess.run(['git', 'show', f'{PRE_FIX_REV}:p515_s56_search_driver.py'], cwd=CODE_ROOT,
                         capture_output=True, text=True, check=True)
    path = os.path.join(out_dir, f'pre_{PRE_FIX_REV}_p515_s56_search_driver.py')
    with open(path, 'x') as h:
        h.write(src.stdout)
    spec = importlib.util.spec_from_file_location('pre_fix_driver', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, D.sha256_file(path)


def affected_dir(data_root, launch, prefix):
    hits = glob.glob(os.path.join(data_root, V2_ROOT_REL, 'launches', launch, 'evals', prefix + '_*'))
    if len(hits) != 1:
        raise RuntimeError(f'{launch}/{prefix}: {len(hits)} eval dirs')
    return hits[0]


def depths_of(eval_dir, root_rel):
    base = os.path.basename(eval_dir)
    if root_rel == PILOT_ROOT_REL or '__settled__' in base:
        return (P.DEPTH_SETTLED, P.DEPTH_EXIT)
    if '__exit__' in base:
        return (P.DEPTH_EXIT,)
    raise RuntimeError(f'depth of {eval_dir} unknown')


def strip(summary):
    return {k: v for k, v in summary.items() if k not in CHANGED_ONLY}


def real_dirs(data_root, root_rel):
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(data_root, root_rel, '**', 'evals', '*',
                                                                     'evaluation_record.json'), recursive=True))


# ======================================================================================================================
def e1():
    import p515_s44_campaign_harness as H
    cl = D.assert_mapping_capture_paths(H)
    new = ('network_failures_summary_in_record', 'esso_recovery_events_written', 'esso_recovery_event_fields')
    ck('E1', 'the mapping capture-path checklist holds, including the three inputs of the extended guard',
       all(cl.values()) and all(cl.get(k) is True for k in new), cl)
    return cl


def e2(data_root, OLD):
    out = []
    for launch, prefix, depth, label in AFFECTED:
        ed = affected_dir(data_root, launch, prefix)
        f = D.read_eval_files(ed)
        rec = f['record']
        sp = rec['solve_profile']
        nfs = rec['network_failures_summary']
        old_g = OLD.guards_exact(rec, f)['solve_profile_identity_holds']
        old_s = OLD.map_record(f, depth)
        rc = D.solve_profile_reconciliation(rec, f)
        new_s = D.map_record(f, depth)
        obs = sp['observed']['permitted_solve']
        diff = obs - sp['base_solves'] - sp['retry_solves_credited']
        want_old, want_new = ('not_certified', 'certified') if depth == P.DEPTH_EXIT else ('not_settled', 'settled')
        ck('E2', f'{launch} {label}: pre-fix guard False and status {want_old} (non-vacuous)',
           old_g is False and old_s['status'] == want_old, [old_g, old_s['status']])
        ck('E2', f'{launch} {label}: observed - base - credited = n_esso_recovery_events = {nfs["n_esso_recovery_events"]}'
                 f', every event tier 1 and succeeded',
           diff == nfs['n_esso_recovery_events'] == len(f['esso_recovery_events'])
           and all(e['tier2_attempted'] is False and e['recovery_succeeded'] is True
                   for e in f['esso_recovery_events']), [diff, nfs['n_esso_recovery_events']])
        ck('E2', f'{launch} {label}: the new guard holds via the extended identity, observed {obs} == expected',
           rc['holds'] is True and rc['via'] == 'extended_esso_recovery' and rc['expected_solves'] == obs
           and rc['failed'] == [], rc)
        ck('E2', f'{launch} {label}: new status {want_new}', new_s['status'] == want_new
           and new_s['guards']['solve_profile_identity_holds'] is True
           and (depth == P.DEPTH_EXIT or new_s['exit_view']['status'] == 'certified'),
           [new_s['status'], new_s.get('not_settled_reasons'), new_s.get('exit_view')])
        out.append({'launch': launch, 'eval_dir': os.path.relpath(ed, data_root), 'depth': depth, 'label': label,
                    'observed': obs, 'base': sp['base_solves'], 'credited': sp['retry_solves_credited'],
                    'n_esso': nfs['n_esso_recovery_events'], 'classes': nfs['classes'],
                    'old_status': old_s['status'], 'new_status': new_s['status'], 'reconciliation': rc,
                    'record_sha256': D.sha256_file(os.path.join(ed, 'evaluation_record.json')),
                    'esso_events_file': f['esso_recovery_events_file'],
                    'esso_events_sha256': D.sha256_file(os.path.join(ed, f['esso_recovery_events_file']))})
    return out


def e3(data_root):
    launch, prefix, depth, _label = AFFECTED[1]
    real = D.read_eval_files(affected_dir(data_root, launch, prefix))
    ck('E3', 'fixture: the launch 7 record holds (3 ESSO events)', D.solve_profile_reconciliation(real['record'],
                                                                                                    real)['holds']
       and len(real['esso_recovery_events']) == 3)

    def ev(f, i, **kw):
        f['esso_recovery_events'][i].update(kw)

    def cls(f, **kw):
        f['record']['network_failures_summary']['classes'].update(kw)

    def obs(f, d):
        f['record']['solve_profile']['observed']['permitted_solve'] += d

    cases = [
        ('an ESSO event recorded as not succeeded', lambda f: ev(f, 1, recovery_succeeded=False)),
        ('an ESSO event without recovery_succeeded', lambda f: f['esso_recovery_events'][0].pop('recovery_succeeded')),
        ('observed + 1 (off by one)', lambda f: obs(f, +1)),
        ('observed - 1 (off by one)', lambda f: obs(f, -1)),
        ('a tier-2 attempt recorded but not paid for in the count', lambda f: ev(f, 2, tier2_attempted=True,
                                                                                 tier='tier2')),
        ('the events file absent', lambda f: f.update(esso_recovery_events=None)),
        ('one event dropped from the file', lambda f: f['esso_recovery_events'].pop()),
        ('a network block unrecovered', lambda f: (cls(f, unrecovered=1), f['record']['network_failures_summary']
                                                   .update(n_blocks=f['record']['network_failures_summary']
                                                           ['n_blocks'] + 1))),
        ('a network block not attempted', lambda f: (cls(f, not_attempted=1), f['record']['network_failures_summary']
                                                     .update(n_blocks=f['record']['network_failures_summary']
                                                             ['n_blocks'] + 1))),
        ('a network block indeterminate', lambda f: (cls(f, indeterminate=1), f['record']['network_failures_summary']
                                                     .update(n_blocks=f['record']['network_failures_summary']
                                                             ['n_blocks'] + 1))),
        ('credited != tier1 + 2 tier2 (credited + 1, observed + 1)',
         lambda f: (f['record']['solve_profile'].update(retry_solves_credited=f['record']['solve_profile']
                                                        ['retry_solves_credited'] + 1), obs(f, +1))),
        ('reconciliation declared supported', lambda f: f['record']['solve_profile'].update(
            reconciliation_supported=True)),
        ('tier record inconsistent (tier2 False, tier "tier2")', lambda f: ev(f, 0, tier='tier2')),
        ('identity False with no ESSO event (the harness path)', lambda f: (
            f['record']['network_failures_summary'].update(n_esso_recovery_events=0),
            f.update(esso_recovery_events=[]))),
        ('an event of another record type', lambda f: ev(f, 0, record_type='network_block')),
    ]
    out = {}
    for name, mut in cases:
        f = copy.deepcopy(real)
        mut(f)
        rc = D.solve_profile_reconciliation(f['record'], f)
        m = D.map_record(f, depth)
        me = D.map_record(f, P.DEPTH_EXIT)
        ck('E3', f'{name} -> the guard fails; settled not_settled, exit not_certified',
           rc['holds'] is False and m['guards']['solve_profile_identity_holds'] is False
           and m['status'] == 'not_settled' and me['status'] == 'not_certified', [rc['failed'], m['status'],
                                                                                  me['status']])
        out[name] = rc['failed']
    f = copy.deepcopy(real)
    ev(f, 2, tier2_attempted=True, tier='tier2')
    obs(f, +1)
    rc = D.solve_profile_reconciliation(f['record'], f)
    ck('E3', 'positive control: the tier-2 attempt paid for (observed + 1) -> holds, ESSO solves 4 for 3 events',
       rc['holds'] is True and rc['esso_recovery_solves'] == 4, rc['failed'])
    return out


def e4(data_root, OLD):
    rows, changed, errors = [], [], []
    for root_rel in (PILOT_ROOT_REL, V2_ROOT_REL):
        for ed in real_dirs(data_root, root_rel):
            fo = OLD.read_eval_files(ed)
            fn = D.read_eval_files(ed)
            for depth in depths_of(ed, root_rel):
                so, sn = OLD.map_record(fo, depth), D.map_record(fn, depth)
                old_pass = OLD.guards_exact(fo['record'], fo)['solve_profile_identity_holds']
                new_pass = D.guards_exact(fn['record'], fn)['solve_profile_identity_holds']
                same = P.canonical_json(strip(so)) == P.canonical_json(strip(sn))
                rel = os.path.relpath(ed, data_root)
                rows.append({'eval_dir': rel, 'depth': depth, 'old_guard': old_pass, 'new_guard': new_pass,
                             'old_status': so['status'], 'new_status': sn['status'], 'summary_equal_but_v2_fields': same,
                             'old_schema': so.get('mapping_schema'), 'new_schema': sn.get('mapping_schema')})
                if not same:
                    changed.append((rel, depth))
                if old_pass and not (new_pass and same):
                    errors.append((rel, depth))
    n_dirs = len({r['eval_dir'] for r in rows})
    ck('E4', f'pilot + v2 phase: {n_dirs} real eval dirs, {len(rows)} (record, depth) mappings; every record that '
             f'passed before passes now with an identical summary (but {list(CHANGED_ONLY)})', not errors and rows,
       errors)
    want = set()
    for launch, prefix, depth, _l in AFFECTED:
        rel = os.path.relpath(affected_dir(data_root, launch, prefix), data_root)
        want |= {(rel, d) for d in ((P.DEPTH_EXIT,) if depth == P.DEPTH_EXIT else (P.DEPTH_SETTLED, P.DEPTH_EXIT))}
    ck('E4', 'the (record, depth) mappings whose summary changes are exactly the 4 affected records (settled ones at '
             'both depths)', set(changed) == want, sorted(set(changed) ^ want))
    ck('E4', 'old summaries carry mapping v1, new ones v2', all(r['old_schema'] == 'p515_s56_oracle_mapping_v1'
                                                               and r['new_schema'] == 'p515_s56_oracle_mapping_v2'
                                                               for r in rows))
    # broader: every evaluation_record.json with a solve_profile under the S56 results tree, guard value only
    broad, bad = 0, []
    for rp in sorted(glob.glob(os.path.join(data_root, RESULTS_REL, '**', 'evaluation_record.json'), recursive=True)):
        ed = os.path.dirname(rp)
        try:
            rec = D.load_json(rp)
        except ValueError:
            continue
        if not isinstance(rec.get('solve_profile'), dict):
            continue
        broad += 1
        fo, fn = OLD.read_eval_files(ed), D.read_eval_files(ed)
        if OLD.guards_exact(fo['record'], fo)['solve_profile_identity_holds'] and not \
                D.guards_exact(fn['record'], fn)['solve_profile_identity_holds']:
            bad.append(os.path.relpath(ed, data_root))
    ck('E4', f'every evaluation record with a solve_profile under {RESULTS_REL} ({broad}): old guard pass -> new pass',
       broad > 0 and not bad, bad)
    return {'rows': rows, 'changed': sorted(changed), 'n_dirs': n_dirs, 'broad_records_scanned': broad}


# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root', default=CODE_ROOT)
    args = ap.parse_args()
    data_root = os.path.abspath(args.data_root)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(data_root, args.out)
    if os.path.exists(out_dir) and [n for n in os.listdir(out_dir) if not n.endswith('.log')]:
        log(f'REFUSE: {out_dir} exists and holds non-log files')
        return 2
    os.makedirs(out_dir, exist_ok=True)
    started, t0 = datetime.now(timezone.utc).isoformat(), time.time()
    OLD, old_sha = load_pre_fix_driver(out_dir)
    _p, v2_sha, v2_spec = D.load_phase_spec(os.path.join(data_root, V2_ROOT_REL))
    ck('pre', f'the pre-fix driver ({PRE_FIX_REV}) is the driver the v2 phase spec froze',
       old_sha == v2_spec['code_sha256']['p515_s56_search_driver.py'], [old_sha[:16],
                                                                         v2_spec['code_sha256']['p515_s56_search_driver.py'][:16]])
    ck('pre', 'the comparison is not vacuous: the working-tree driver differs from the pre-fix driver',
       old_sha != D.sha256_file(os.path.join(CODE_ROOT, 'p515_s56_search_driver.py')))
    res = {'E1': e1(), 'E2': e2(data_root, OLD), 'E3': e3(data_root), 'E4': e4(data_root, OLD)}
    gf = GUARD.verify(0)
    ck('guards', 'SolveProfileGuard(permitted=()) verify(0) == []', not gf, gf)
    ck('guards', 'pickle load/loads 0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    code = {n: D.sha256_file(os.path.join(CODE_ROOT, n)) for n in (
        SCRIPT, 'p515_s56_search_driver.py', 'p515_s56_search_policy.py', 'p515_s44_campaign_harness.py',
        'p515_g_g1_g4_admm_gates.py', 'shared_energy_storage_data.py', 'gate_result_io.py', 'p513_solve_profile_guard.py')}
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=data_root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--', *code], cwd=CODE_ROOT, capture_output=True,
                           text=True).stdout
    payload = {'schema': 'p515_s56_esso_guard_tests_v1', 'script': SCRIPT, 'started_utc': started,
               'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0,
               'interpreter': sys.executable, 'python': platform.python_version(), 'git_head': head,
               'code_files_dirty_vs_head': dirty, 'pre_fix_rev': PRE_FIX_REV, 'pre_fix_driver_sha256': old_sha,
               'v2_phase_spec_sha256': v2_sha,
               'rule': D.SOLVE_PROFILE_RECONCILIATION_RULE,
               'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) installed before any '
                              'repository import, verify(0)); pickle blocked (0); records read only',
               'results': res, 'guards': {'counts': dict(GUARD.counts), 'verify_0': gf, 'pickle': dict(PICKLE_COUNTS)},
               'code_files': code, 'checks': CHECKS,
               'failed_checks': [f"[{c['test']}] {c['check']}" for c in CHECKS if not c['ok']],
               'all_checks_ok': all(c['ok'] for c in CHECKS)}
    sha = D.write_once_json(os.path.join(out_dir, 'esso_guard_tests.json'), payload)
    files = [f for f in D.files_under(out_dir) if os.path.basename(f) != 'esso_guard_tests.json'
             and not f.endswith('.log')]
    D.write_once_json(os.path.join(out_dir, 'manifest_sha256.json'),
                      {'esso_guard_tests.json': sha,
                       'files': {os.path.relpath(f, out_dir): D.sha256_file(f) for f in files}, 'code_files': code})
    log(f'checks {sum(c["ok"] for c in CHECKS)}/{len(CHECKS)} ok; wall {time.time() - t0:.0f} s; failed '
        f'{payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] else 3


if __name__ == '__main__':
    sys.exit(main())
