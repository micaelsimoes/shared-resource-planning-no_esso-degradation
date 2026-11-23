"""P5.15 S56 EV search -- the ZERO-SOLVE DETERMINISM GATE of the search driver (S56_SEARCH_DRIVER_DESIGN.md, "Determinism
gate (zero solves; guard armed)", items 1-9). Engineering build: a script with pass/fail checks (TASKS.md "Working mode
and EV priority": no prediction spec).

Guards (installed before any repository import): `SolveProfileGuard(permitted=())`, verified at exactly 0 at the end;
pickle.load / pickle.loads blocked and counted (0). Every driver subprocess the gate launches arms its own guard
(`p515_s56_search_driver.main`) and is checked for its exit code. No model is built; no evaluation is real: the
oracles are the synthetic landscape (in process) and, for items 4-5, the harness stub children ('stub' records, no
model import), with SYNTHETIC values attached by the driver.

ITEMS
  1  N(x) == s56_neighbourhood on the named points (|N| 18 / 22 / 13); max |N| <= 80 over all 37,042 plans; the
     domain re-enumerated by the lattice rule == the committed diagnostic-1 table (hash-checked); driver |N| == table
     |N| on every plan; cap 80 owned by the policy (an 81-point neighbourhood is refused; plans with |N| > 60 order);
     every diagonal in N(x) same-sign.
  2  the order is byte-identical across fresh processes (PYTHONHASHSEED 0 / 4242 / random) with shuffled input
     points, for every plan, with no last move, with a deterministic last move, and with the reversed class rank.
  3  the order is unchanged when every cached value is perturbed, and identical at 1x and 2x.
  4  stub-children replay: identical incumbent history, evaluated set and budget count, serial and 5-way, randomised
     per-child delays (harness `evaluate`, stub mode). Control (re-scoped by integration ruling 6, 601775c0): the 2x
     EV configuration FREEZES -- the smoke alias is absent (the cap rule removed it): no collision, and no key of the
     committed ev_harness_v1 smoke spec among the configuration's 74,084 domain keys.
  5  a kill at every transition, then a resume: the history byte-equal to the uninterrupted run, killed evaluations
     uncounted (synthetic oracle at every transition; stub children during evaluate).
  6  keys: 37,042 distinct lattice keys; keys pairwise distinct across the configuration product; staged key ==
     GeneralLattice.key; the collision scan excludes the campaign's own root.
  7  classification boundaries (dF = R + e exactly, -R + e exactly, the e floor, a missing bar, uncertified, strict
     status), confirmation boundaries, mixed depth refused, re-classification; a not_settled confirmation ->
     unresolved after confirmation, and the review trigger's boundaries (integration ruling 5, 601775c0).
  8  terminal arithmetic (partition == |N|, roles == budget, the count identity) and budget exhaustion mid-poll;
     not_settled confirmations in a search: one -> the search continues; 3 consecutive -> stop for review; 2 of 4 ->
     stop for review (ruling 5); the driver maps that stop to exit 5.
  9  published histories (s47 Phase B 1x, s51 F2 Phase B 2x, s53 F2 certificate 2x) replayed where the old cache
     covers N(x), with coverage reported.

Launch (attached, both streams captured):
  python -u p515_s56_search_gate.py --out <new dir> [--items 1,2,3,6,7,8,9] [--data-root <repo>]
      > <out>_launch_stdout.log 2> <out>_launch_stderr.log
Exit: 0 every check holds; 3 a check failed; 2 refused.
"""
import os
import sys

CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 search driver gate (never solves)').install()
import pickle  # noqa: E402

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('search gate: pickle.load is blocked')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('search gate: pickle.loads is blocked')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from itertools import product  # noqa: E402

import p515_s56_search_driver as D  # noqa: E402
import p515_s56_search_policy as P  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
DIAG1_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'diag_neighbourhood_residual_v2')
DIAG1_TABLE = 'neighbourhood_sizes_table.csv'
DIAG1_JSON = 'neighbourhood_sizes.json'
DIAG1_MANIFEST = 'manifest_neighbourhood_sha256.json'
NAMED_DIAG1 = {'x0': 'x0', 'reference_unit': 'reference_unit_n7_0.25_1_y2025',
               'plan_2030': 'plan_2030_n5_0.25_0.5__n7_1_3.5'}
NAMED_SIZES = {'x0': 18, 'reference_unit': 22, 'plan_2030': 13}
PUBLISHED = (
    ('s47_phase_b_1x', os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b',
                                    'campaign_results.json')),
    ('s51_f2_phase_b_2x', os.path.join('data', 'SRP1', 'Results', 'P515S51', 'campaign_s51_f2_phase_b',
                                       'campaign_results.json')),
    ('s53_f2_certificate_2x', os.path.join('data', 'SRP1', 'Results', 'P515S53', 'campaign_s53_f2_certificate_r1',
                                           'campaign_results.json')),
)
CHECKS = []


def log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def ck(item, name, ok, detail=None):
    CHECKS.append({'item': item, 'check': name, 'ok': bool(ok), 'detail': detail})
    log(f'{"OK  " if ok else "FAIL"} [{item}] {name}' + (f' -- {str(detail)[:500]}' if detail is not None and not ok
                                                          else ''))
    return bool(ok)


def sha(path):
    return D.sha256_file(path)


def git(argv, cwd):
    r = subprocess.run(['git', *argv], cwd=cwd, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def tracked_clean(path, data_root):
    rel = os.path.relpath(path, data_root)
    t = git(['ls-files', '--', rel], data_root)
    c = git(['status', '--porcelain', '--', rel], data_root)
    return bool(t and t.strip()) and c is not None and not c.strip()


# ======================================================================================================================
#  the domain (lattice rule) and the committed diagnostic-1 table
# ======================================================================================================================
def enumerate_domain(lat):
    """The staged domain (the driver's `enumerate_domain`, lattice rule)."""
    return D.enumerate_domain(lat)


def load_diag1(data_root):
    d = os.path.join(data_root, DIAG1_DIR_REL)
    man = D.load_json(os.path.join(d, DIAG1_MANIFEST))
    ev = {}
    for name in (DIAG1_TABLE, DIAG1_JSON):
        p = os.path.join(d, name)
        ev[name] = {'sha256': sha(p), 'manifest_sha256': man['outputs'][name], 'tracked_clean': tracked_clean(p, data_root)}
    rows = list(csv.DictReader(open(os.path.join(d, DIAG1_TABLE))))
    table = {tuple(int(v) for v in r['z'].split()): r for r in rows}
    return table, D.load_json(os.path.join(d, DIAG1_JSON)), ev


# ======================================================================================================================
#  item 1
# ======================================================================================================================
def item1(lat, data_root, ctx):
    t0 = time.time()
    table, diag, ev = load_diag1(data_root)
    ck(1, 'diagnostic-1 table and json hash to their committed manifest, tracked and clean',
       all(v['sha256'] == v['manifest_sha256'] and v['tracked_clean'] for v in ev.values()), ev)
    domain = enumerate_domain(lat)
    ctx['domain'] = domain
    ck(1, 'domain re-enumerated by the lattice rule: 37,042 plans == DOMAIN_SIZE == the committed table',
       len(domain) == D.DOMAIN_SIZE == len(table) and set(domain) == set(table), [len(domain), len(table)])
    env = D.LatticeEnv(lat, memoize=False)
    sizes, mismatches, over60, not_same_sign, orders_failed = {}, [], 0, [], []
    for z in domain:
        pts = env.neighbourhood_points(z)
        try:
            o = P.order(lat, z, None, P.CLASS_RANK, pts)
        except P.PolicyError as error:
            orders_failed.append((list(z), str(error)))
            continue
        sizes[z] = len(o)
        if len(o) != int(table[z]['N_size']):
            mismatches.append((list(z), len(o), table[z]['N_size']))
        if len(o) > 60:
            over60 += 1
        for r in o:
            if r['move'].startswith('diag') and r['move'].split(':')[1] not in P.SAME_SIGN_DIAGONALS:
                not_same_sign.append(r['move'])
    vals = sorted(sizes.values())
    ctx['sizes'] = sizes
    ck(1, 'P.order accepts N(x) of every plan (no refusal)', not orders_failed, orders_failed[:5])
    ck(1, 'driver |N(x)| == committed table N_size on every plan', not mismatches, mismatches[:5])
    ck(1, 'max |N(x)| over the 37,042 plans <= 80 (CAP)', vals and vals[-1] <= P.CAP == 80,
       {'min': vals[0], 'median': vals[len(vals) // 2], 'max': vals[-1]})
    ck(1, 'CAP 60 of p516_lattice ignored: plans with |N| > 60 are ordered', over60 > 0, {'n_plans_N_gt_60': over60})
    ck(1, 'every diagonal move in N(x) is same-sign (no e_opp point)', not not_same_sign, not_same_sign[:5])
    try:
        P.order(lat, domain[0], None, P.CLASS_RANK, [{'class': 'r3', 'part': 'b', 'move': 'activate:x', 'z': (0,) * 18}]
                * (P.CAP + 1))
        refused = False
    except P.CapExceeded:
        refused = True
    ck(1, 'an 81-point neighbourhood is refused (CapExceeded)', refused)
    import p515_s56_neighbourhood as NB
    named = {}
    for name, size in NAMED_SIZES.items():
        z = D.named_start(lat, name)
        nb = NB.s56_neighbourhood(lat, z, keep_points=True)
        nb_set = {tuple(p['z']) for p in nb['points'] if p['class'] in NB.CLASS_ORDER}
        o = P.order(lat, z, None, P.CLASS_RANK, D.LatticeEnv(lat).neighbourhood_points(z))
        o_set = {tuple(r['z']) for r in o}
        committed = diag['named_points'][NAMED_DIAG1[name]]
        # the committed `points` also list the informational class e_opp (not in N(x)); compare the N(x) classes
        committed_keys = sorted(p['key'] for p in committed['points'] if p['class'] in NB.CLASS_ORDER)
        ok = (len(o) == size == nb['size'] == committed['size'] and o_set == nb_set
              and sorted(lat.key(z2) for z2 in o_set) == committed_keys)
        named[name] = {'N': len(o), 'expected': size, 'nb_size': nb['size'], 'committed': committed['size'],
                       'committed_e_opp_listed': sum(1 for p in committed['points'] if p['class'] == 'e_opp')}
        ck(1, f'{name}: driver N(x) == s56_neighbourhood == committed diagnostic-1 keys, |N| = {size}', ok, named[name])
    return {'domain_size': len(domain), 'N_min_median_max': [vals[0], vals[len(vals) // 2], vals[-1]],
            'n_plans_N_gt_60': over60, 'named': named, 'diag1_evidence': ev, 'wall_s': time.time() - t0}


# ======================================================================================================================
#  item 2 (worker subprocesses)
# ======================================================================================================================
def order_digest(lat, domain, shuffle_seed):
    """Every plan: the order with no last move, with a deterministic last move (the displacement to the neighbour at
    position (index mod |N|) of the lattice-order N), and with the reversed class rank; shuffled input points."""
    env = D.LatticeEnv(lat, memoize=False)
    rng = random.Random(shuffle_seed)
    h = hashlib.sha256()
    named = {}
    names = {D.named_start(lat, n): n for n in NAMED_SIZES}
    for i, z in enumerate(domain):
        pts = env.neighbourhood_points(z)
        last = [a - b for a, b in zip(pts[i % len(pts)]['z'], z)] if pts else None
        rng.shuffle(pts)
        rows = []
        for lm, rank in ((None, P.CLASS_RANK), (last, P.CLASS_RANK), (None, P.CLASS_RANK_REVERSED)):
            o = P.order(lat, z, lm, rank, pts)
            rows.append([r['label'] for r in o])
        line = P.canonical_json([list(z), rows])
        h.update(line.encode())
        h.update(b'\n')
        if z in names:
            named[names[z]] = rows
    return h.hexdigest(), named


def item2_worker(args):
    lat, _inst = D.build_lattice(args.data_root)
    domain = enumerate_domain(lat)
    digest, named = order_digest(lat, domain, args.shuffle_seed)
    with open(args.worker_out, 'x') as handle:
        json.dump({'digest': digest, 'named': named, 'n_plans': len(domain),
                   'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'), 'shuffle_seed': args.shuffle_seed,
                   'pid': os.getpid(), 'guard_verify_0': GUARD.verify(0), 'pickle_counts': PICKLE_COUNTS}, handle)
    return 0


def item2(lat, data_root, out_dir, ctx):
    t0 = time.time()
    wdir = os.path.join(out_dir, 'item2_workers')
    os.makedirs(wdir)
    procs = []
    for tag, seed_env, shuffle in (('a', '0', 'shuffle-1'), ('b', '4242', 'shuffle-2'), ('c', 'random', 'shuffle-3')):
        env = dict(os.environ, PYTHONHASHSEED=seed_env)
        outp = os.path.join(wdir, f'worker_{tag}.json')
        so, se = open(os.path.join(wdir, f'worker_{tag}_stdout.log'), 'w'), open(
            os.path.join(wdir, f'worker_{tag}_stderr.log'), 'w')
        p = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--item2-worker', '--data-root',
                              data_root, '--worker-out', outp, '--shuffle-seed', shuffle], env=env, stdout=so,
                             stderr=se, cwd=CODE_ROOT)
        procs.append((tag, p, outp, so, se))
    res = {}
    for tag, p, outp, so, se in procs:
        rc = p.wait()
        so.close()
        se.close()
        res[tag] = D.load_json(outp) if rc == 0 and os.path.exists(outp) else {'exit': rc}
    digests = {t: r.get('digest') for t, r in res.items()}
    ck(2, 'three fresh processes (PYTHONHASHSEED 0 / 4242 / random; shuffled input) exit 0 with guard 0',
       all('digest' in r and not r['guard_verify_0'] for r in res.values()), {t: r.get('exit') for t, r in res.items()})
    ck(2, 'order digest over all 37,042 plans x 3 orders byte-identical across the three processes',
       len(set(digests.values())) == 1 and None not in digests.values(), digests)
    ck(2, 'named-point orders identical across the processes',
       all(r.get('named') == res['a'].get('named') for r in res.values()))
    in_proc, _ = order_digest(lat, ctx['domain'], 'shuffle-in-process')
    ck(2, 'in-process digest (another shuffle) == the workers', in_proc == digests.get('a'), [in_proc, digests.get('a')])
    return {'digests': digests, 'in_process': in_proc, 'workers': {t: {k: v for k, v in r.items() if k != 'named'}
                                                                    for t, r in res.items()}, 'wall_s': time.time() - t0}


# ======================================================================================================================
#  item 3
# ======================================================================================================================
def item3(lat, ctx):
    env = D.LatticeEnv(lat)
    target = D.named_start(lat, 'plan_2030')
    rng = random.Random(3)
    sample = [D.named_start(lat, n) for n in NAMED_SIZES] + rng.sample(ctx['domain'], 40)
    diffs = []
    for z in sample:
        orders = []
        for arm, seed in (('ev_2x', 'base'), ('ev_2x', 'perturbed-1'), ('ev_2x', 'perturbed-2'), ('ev_1x', 'base')):
            land = D.SyntheticLandscape(seed, target, z, alpha=30000.0 if seed == 'base' else 7000.0)
            recs = {(tuple(z), 'exit'): dict(land.summary(z, 'exit'), status='certified')}
            for p in env.neighbourhood_points(z):
                recs[(tuple(p['z']), 'exit')] = land.summary(p['z'], 'exit')
            st = P.initial_state({'start': list(z), 'n_max': 1000, 'arm': arm})
            st, _act = P.step(st, recs, env)
            poll = st['polls'][0] if st['polls'] else st['poll']
            orders.append(P.canonical_json([r['z'] for r in poll['order']]))
        if len(set(orders)) != 1:
            diffs.append(list(z))
    ck(3, f'first-poll order unchanged under perturbed cached values and at 1x vs 2x ({len(sample)} plans, every '
          f'neighbour value cached and perturbed)', not diffs, diffs[:5])
    # through the driver's launch machinery: 1x and 2x specs issue the same first block
    return {'n_plans': len(sample)}


# ======================================================================================================================
#  item 7
# ======================================================================================================================
def _ex(F, bar, status='certified'):
    return {'depth': 'exit', 'status': status, 'F': F, 'bar': bar}


def _se(F, S=100.0, PR=0.0, G=0.0, A=0.0, gates=True, status='settled'):
    return {'depth': 'settled', 'status': status, 'F': F, 'V': {'S': S, 'PR': PR, 'G': G, 'A': A}, 'gates_pass': gates}


def item7():
    R = P.R_EUR
    inc = _ex(1.0e8, 4000.0)
    e = max(P.E_FLOOR_EUR, 4000.0 + 5000.0)
    cases = {
        'dF = R + e exactly -> worse': (P.classify_exit(_ex(1.0e8 + (R + e), 5000.0), inc)['class'] == 'worse'
                                        if (1.0e8 + (R + e)) - 1.0e8 == R + e else None),
        'dF = -R + e exactly -> confirm_candidate': P.classify_exit(_ex(1.0e8 + (-R + e), 5000.0), inc)['class']
        == 'confirm_candidate',
        'dF just below R + e -> unresolved': P.classify_exit(_ex(math.nextafter(1.0e8 + (R + e), 0), 5000.0), inc)[
            'class'] == 'unresolved',
        'dF just above -R + e -> unresolved': P.classify_exit(_ex(math.nextafter(1.0e8 + (-R + e), math.inf), 5000.0),
                                                              inc)['class'] == 'unresolved',
        'e floor 15,000 applies (bars 4,000 + 5,000)': P.classify_exit(_ex(1.0e8, 5000.0), inc)['e'] == 15000.0,
        'e = bar sum above the floor (bars 4,000 + 20,000)': P.classify_exit(_ex(1.0e8, 20000.0), inc)['e'] == 24000.0,
        'missing bar (None) -> harness_error': P.classify_exit(_ex(1.0e8, None), inc)['class'] == 'harness_error',
        'missing bar (absent) -> harness_error': P.classify_exit({'depth': 'exit', 'status': 'certified', 'F': 1.0e8},
                                                                 inc)['class'] == 'harness_error',
        'non-finite bar (nan) -> harness_error': P.classify_exit(_ex(1.0e8, float('nan')), inc)['class']
        == 'harness_error',
        'uncertified with a large improvement -> uncertified': P.classify_exit(_ex(0.0, 1.0, 'not_certified'), inc)[
            'class'] == 'uncertified',
        'uncertified with a large loss -> uncertified (never worse)': P.classify_exit(
            _ex(1e12, 1.0, 'not_certified'), inc)['class'] == 'uncertified',
        'strict status: "Certified" is not certified -> harness_error (unknown status)': P.classify_exit(
            _ex(1.0e8, 1.0, 'Certified'), inc)['class'] == 'harness_error',
        'status harness_error -> harness_error': P.classify_exit(_ex(1.0e8, 1.0, 'harness_error'), inc)['class']
        == 'harness_error',
    }
    for name, ok in cases.items():
        ck(7, name, ok is True)
    refusals = {
        'mixed depth (candidate settled vs incumbent exit) refused': lambda: P.classify_exit(_se(1.0e8), inc),
        'mixed depth (incumbent settled) refused': lambda: P.classify_exit(_ex(1.0e8, 1.0), _se(1.0e8)),
        'confirm with an exit record as settled refused': lambda: P.confirm(_ex(1.0e8, 1.0), _se(1.0e8),
                                                                            _ex(1.0e8, 1.0), inc, 15000.0),
        'incumbent without a bar refused': lambda: P.classify_exit(_ex(1.0e8, 1.0), _ex(1.0e8, None)),
    }
    expected_exc = {'incumbent without a bar refused': P.PolicyError}
    for name, fn in refusals.items():
        want = expected_exc.get(name, P.MixedDepthError)
        try:
            fn()
            got = None
        except P.PolicyError as error:  # MixedDepthError is a PolicyError
            got = type(error)
        ck(7, name, got is want, got.__name__ if got else 'not refused')
    ix, ii = _ex(1.0e8 - 20000.0, 100.0), _ex(1.0e8, 100.0)
    cf = {
        'improvement exactly R, V_pair exactly R/4, A exactly R/2 -> accepted':
            P.confirm(_se(1.0e8 - R - 20000.0, S=1250.0, A=2500.0), _se(1.0e8 - 20000.0, S=1250.0, A=2500.0), ix, ii,
                      15000.0)['result'] == 'accepted',
        'improvement just below R -> unresolved_after_confirmation':
            P.confirm(_se(math.nextafter(1.0e8 - R - 20000.0, math.inf)), _se(1.0e8 - 20000.0), ix, ii, 15000.0)[
                'result'] == 'unresolved_after_confirmation',
        'V_pair just above R/4 -> unresolved_after_confirmation':
            P.confirm(_se(1.0e8 - R - 20000.0, S=1250.0), _se(1.0e8 - 20000.0, S=1250.0, PR=0.001), ix, ii, 15000.0)[
                'result'] == 'unresolved_after_confirmation',
        'A sum above R/2 -> unresolved_after_confirmation':
            P.confirm(_se(1.0e8 - R - 20000.0, A=2500.0), _se(1.0e8 - 20000.0, A=2500.01), ix, ii, 15000.0)['result']
            == 'unresolved_after_confirmation',
        'a gate failed -> unresolved_after_confirmation':
            P.confirm(_se(1.0e8 - R - 20000.0, gates=False), _se(1.0e8 - 20000.0), ix, ii, 15000.0)['result']
            == 'unresolved_after_confirmation',
        'held-out offset exactly e -> not a stop':
            P.confirm(_se(1.0e8 - 20000.0 - 15000.0), _se(1.0e8), ix, ii, 15000.0)['result'] != 'stop_for_review',
        'held-out offset above e -> stop_for_review':
            P.confirm(_se(1.0e8 - 20000.0 - 15000.01), _se(1.0e8), ix, ii, 15000.0)['result'] == 'stop_for_review',
        # integration ruling 5 (601775c0): a not_settled confirmation is "unresolved after confirmation"
        'candidate settled not_settled -> unresolved_after_confirmation, flagged, the candidate named':
            (lambda r: r['result'] == 'unresolved_after_confirmation' and r['not_settled'] is True
             and r['not_settled_sides'] == ['candidate'])(
                P.confirm(_se(1.0e8 - R - 20000.0, status='not_settled'), _se(1.0e8 - 20000.0), ix, ii, 15000.0)),
        'incumbent settled not_settled -> unresolved_after_confirmation, the incumbent named':
            (lambda r: r['result'] == 'unresolved_after_confirmation' and r['not_settled_sides'] == ['incumbent'])(
                P.confirm(_se(1.0e8 - R - 20000.0), _se(1.0e8 - 20000.0, status='not_settled'), ix, ii, 15000.0)),
        'both not_settled -> unresolved_after_confirmation, both named':
            P.confirm(_se(1.0e8, status='not_settled'), _se(1.0e8, status='not_settled'), ix, ii, 15000.0)[
                'not_settled_sides'] == ['candidate', 'incumbent'],
        'not_settled with an offset far above e -> no held-out stop (no test on a not-settled value)':
            P.confirm(_se(1.0e8 - 1.0e6, status='not_settled'), _se(1.0e8), ix, ii, 15000.0)['result']
            == 'unresolved_after_confirmation',
        'not_settled without V components -> still unresolved_after_confirmation (never accepted, never a stop)':
            P.confirm({'depth': 'settled', 'status': 'not_settled', 'F': None}, _se(1.0e8), ix, ii, 15000.0)[
                'result'] == 'unresolved_after_confirmation',
        'not_settled candidate vs incumbent harness_error -> harness_error (stops)':
            P.confirm(_se(1.0e8, status='not_settled'), _se(1.0e8, status='harness_error'), ix, ii, 15000.0)[
                'result'] == 'harness_error',
        'unknown settled status "Settled" -> harness_error (strict)':
            P.confirm(_se(1.0e8, status='Settled'), _se(1.0e8), ix, ii, 15000.0)['result'] == 'harness_error',
        'settled without V components -> harness_error':
            P.confirm({'depth': 'settled', 'status': 'settled', 'F': 1.0e8, 'gates_pass': True}, _se(1.0e8), ix, ii,
                      15000.0)['result'] == 'harness_error',
        'gates_pass not a bool ("True") -> harness_error':
            P.confirm(dict(_se(1.0e8), gates_pass='yes'), _se(1.0e8), ix, ii, 15000.0)['result'] == 'harness_error',
    }
    for name, ok in cf.items():
        ck(7, name, ok)
    # re-classification at zero solves
    lat_free = {'polls': [{'index': 0, 'z_inc': [0], 'classified': [{'pos': 0, 'z': [1], 'exit_class': 'unresolved'},
                                                                     {'pos': 1, 'z': [2], 'exit_class': 'worse'}]}],
                'poll': None, 'config': {'R': R}}
    recs = {((0,), 'exit'): _ex(1.0e8, 1000.0), ((1,), 'exit'): _ex(1.0e8 + 8000.0, 1000.0),
            ((2,), 'exit'): _ex(1.0e8 + 40000.0, 1000.0)}
    ch = P.reclassify(lat_free, recs, 20000.0)
    ck(7, 're-classification at a measured e = 20,000: dF 8,000 unresolved -> confirm_candidate; dF 40,000 stays worse',
       [(c['pos'], c['old'], c['new']) for c in ch] == [(0, 'unresolved', 'confirm_candidate')], ch)
    trig = review_trigger_cases()
    return {'n_cases': len(cases) + len(refusals) + len(cf) + 1 + len(trig), 'review_trigger': trig}


def _hist_state(pattern, split=None):
    """A state carrying confirmations in order: pattern of 'n' (not_settled), 'a' (accepted), 'u' (unresolved after
    confirmation, settled), 'x' (a confirmation not run for budget: not a confirmation), '-' (no confirmation, an
    exit class). `split`: the index at which a second (open) poll starts."""
    res = {'n': {'result': 'unresolved_after_confirmation', 'not_settled': True},
           'a': {'result': 'accepted', 'not_settled': False},
           'u': {'result': 'unresolved_after_confirmation', 'not_settled': False},
           'x': {'result': 'not_run_budget'}, '-': None}
    entries = [{'pos': i, 'confirmation': res[c]} for i, c in enumerate(pattern)]
    if split is None:
        return {'polls': [], 'poll': {'index': 0, 'classified': entries}}
    first = entries[:split]
    second = [dict(e, pos=i) for i, e in enumerate(entries[split:])]
    return {'polls': [{'index': 0, 'classified': list(reversed(first))}], 'poll': {'index': 1, 'classified': second}}


def review_trigger_cases():
    """Integration ruling 5's review trigger (`P.not_settled_review`) at its boundaries."""
    cases = [  # (pattern, split, stop, consecutive rule, share rule)
        ('nn', None, False, False, False),
        ('nnn', None, True, True, False),
        ('n-u-n', None, False, False, False),        # 2 of 3: share rule needs >= 4 confirmations
        ('uunu', None, False, False, False),         # 1 of 4 = 25 %: not MORE than 25 %
        ('unun', None, True, False, True),           # 2 of 4 = 50 %
        ('uuuunuuun', None, False, False, False),     # 2 of 9 = 22 %
        ('uuunun', None, True, False, True),         # 2 of 6 = 33 %
        ('aannn', None, True, True, True),           # 3 consecutive and 3 of 5
        ('nnxn', None, True, True, False),           # a budget-unrun confirmation is not a confirmation: n n n
        ('uunn', 2, True, False, True),              # across polls: 2 of 4 (closed poll order by position)
        ('nnn', 2, True, True, False),               # consecutive across a poll boundary
    ]
    out = []
    for pat, split, stop, rc, rs in cases:
        r = P.not_settled_review(_hist_state(pat, split))
        ok = r['stop'] is stop and r['rule_consecutive'] is rc and r['rule_share'] is rs
        ck(7, f'review trigger: confirmations {pat!r}{" (2 polls)" if split else ""} -> stop {stop} '
              f'(consecutive {rc}, share {rs})', ok, r)
        out.append({'pattern': pat, 'split': split, 'expected_stop': stop, 'got': r})
    return out


# ======================================================================================================================
#  item 8
# ======================================================================================================================
def _search(lat, env, start, n_max, seed='item8', overrides=None, actions=None):
    land = D.SyntheticLandscape(seed, D.named_start(lat, 'plan_2030'), start)
    st = P.initial_state({'start': list(start), 'n_max': n_max, 'arm': 'ev_2x', 'domain_size': D.DOMAIN_SIZE})

    def ev(action):
        if actions is not None:
            actions.append(action)
        out = {}
        for z in action['zs']:
            s = land.summary(z, action['depth'])
            if overrides and (tuple(z), action['depth']) in overrides:
                s = overrides[(tuple(z), action['depth'])](s)
            out[(tuple(z), action['depth'])] = s
        return out
    return P.run_to_end(st, {}, env, ev)


def _arith_ok(st, recs):
    rep = P.terminal_report(st, recs)
    roles = rep['evaluations']['by_role']
    fin = rep['final_poll']
    ok = (sum(roles.values()) == st['budget_used'] == len(st['evaluated']) <= st['config']['n_max']
          and (fin is None or sum(rep['class_partition_final_poll'].values()) == fin['N'])
          and len({(tuple(e['z']), e['depth']) for e in st['evaluated']}) == len(st['evaluated']))
    return ok, rep


def item8(lat):
    env = D.LatticeEnv(lat)
    out = {}
    for name in ('x0', 'reference_unit', 'plan_2030'):
        z0 = D.named_start(lat, name)
        full_actions = []
        st, recs = _search(lat, env, z0, 10000, actions=full_actions)
        ok, rep = _arith_ok(st, recs)
        ck(8, f'{name}: partition == |N_final|, roles == budget == evaluated set, no evaluation twice', ok,
           rep['class_partition_final_poll'])
        ck(8, f'{name}: count identity N_search = S_0 + sum(E_j + c_j) + E_f + c_f holds', rep['count_identity']['holds'],
           rep['count_identity'])
        full = st['budget_used']
        out[name] = {'full_budget': full, 'termination': st['termination']['reason'],
                     'n_moves': rep['n_moves'], 'count_identity': rep['count_identity']}
        kinds = {'mid_poll': 0, 'block_truncated': 0, 'exhausted_before_block': 0, 'confirmation_does_not_fit': 0}
        exact, prefix = True, True
        for n_max in range(0, full + 1):
            acts = []
            st2, recs2 = _search(lat, env, z0, n_max, actions=acts)
            ok2, rep2 = _arith_ok(st2, recs2)
            exact &= ok2 and rep2['count_identity']['holds']
            if n_max < full:
                exact &= st2['termination']['reason'] == 'budget_exhausted' and st2['budget_used'] <= n_max
                det = st2['termination']['detail'] or {}
                if isinstance(det, dict) and 'confirmation_needs' in det:
                    kinds['confirmation_does_not_fit'] += 1
                elif isinstance(det, dict) and 'pos' in det:
                    kinds['exhausted_before_block'] += 1
                fp = rep2['final_poll']
                if fp and 0 < fp['k_classified'] < fp['N']:
                    kinds['mid_poll'] += 1
                # the budget-limited run is the unlimited run's prefix; its last action is truncated, never altered
                k = len(acts)
                prefix &= all(P.canonical_json(a) == P.canonical_json(b) for a, b in zip(acts[:-1], full_actions))
                if k:
                    a, b = acts[-1], full_actions[k - 1]
                    same_head = a['zs'] == b['zs'][:len(a['zs'])] and a['depth'] == b['depth']
                    prefix &= same_head
                    if len(a['zs']) < len(b['zs']):
                        kinds['block_truncated'] += 1
            else:  # at N_max = the unlimited run's count the run is identical (only config.n_max differs)
                exact &= P.canonical_json(dict(st2, config=dict(st2['config'], n_max=None))) == \
                    P.canonical_json(dict(st, config=dict(st['config'], n_max=None)))
        ck(8, f'{name}: every N_max in 0..{full}: arithmetic holds, budget never exceeded, budget_exhausted below '
              f'{full}, identical run at {full}', exact)
        ck(8, f'{name}: every budget-limited run is a prefix of the unlimited run (last block truncated, never '
              f're-ordered)', prefix)
        ck(8, f'{name}: budget exhaustion mid-poll exercised (mid-poll, truncated block, exhaustion before a block'
              f'{", confirmation not fitting" if name != "plan_2030" else ""})',
           kinds['mid_poll'] > 0 and kinds['block_truncated'] > 0 and kinds['exhausted_before_block'] > 0
           and (name == 'plan_2030' or kinds['confirmation_does_not_fit'] > 0), kinds)
        out[name]['exhaustion_kinds'] = kinds
    # stops: harness_error, stop_for_review, start not certified
    z0 = D.named_start(lat, 'x0')
    first = P.order(lat, z0, None, P.CLASS_RANK, env.neighbourhood_points(z0))
    bad = tuple(first[0]['z'])
    st, recs = _search(lat, env, z0, 10000, overrides={(bad, 'exit'): lambda s: dict(s, status='harness_error')})
    ok, rep = _arith_ok(st, recs)
    ck(8, 'harness_error at position 0 stops the search; positions 1-4 of the block become speculative_post_termination',
       st['termination']['reason'] == 'harness_error' and rep['evaluations']['by_role'][
           'speculative_post_termination'] == 4 and rep['class_partition_final_poll']['harness_error'] == 1
       and rep['class_partition_final_poll']['not_reached'] == len(first) - 1 and ok, rep['evaluations'])
    st, recs = _search(lat, env, z0, 10000, overrides={(z0, 'exit'): lambda s: dict(s, status='not_certified')})
    ck(8, 'an uncertified start incumbent ends the search (start_incumbent_not_certified)',
       st['termination']['reason'] == 'start_incumbent_not_certified' and st['budget_used'] == 1)
    base, _r = _search(lat, env, z0, 10000)
    conf = [e for e in base['evaluated'] if e['depth'] == 'settled']
    zc = tuple(conf[0]['z'])
    st, recs = _search(lat, env, z0, 10000, overrides={(zc, 'settled'): lambda s: dict(s, F=s['F'] + 1.0e6)})
    ck(8, 'a held-out offset above e at a confirmation stops the search (stop_for_review_offset)',
       st['termination']['reason'] == 'stop_for_review_offset')
    out['not_settled'] = item8_not_settled(lat, env, z0, base)
    return out


def _confirmed_candidates(st):
    """The candidates of the confirmations that produced a result, in confirmation order."""
    hist = P.confirmation_history(st)
    by = {(p['index'], c['pos']): c for p in st['polls'] + ([st['poll']] if st['poll'] else []) for c in p['classified']}
    return [(tuple(by[(h['poll'], h['pos'])]['z']), h['not_settled']) for h in hist]


def item8_not_settled(lat, env, z0, base):
    """Integration ruling 5 in a search (synthetic oracle): the candidates' settled summaries are set to
    not_settled one confirmation at a time, so each pattern is reached on the path it itself produces."""
    ns = lambda s: dict(s, status='not_settled')   # noqa: E731
    res = {}
    hist0 = _confirmed_candidates(base)
    # one not_settled confirmation: the search continues and ends normally
    ov1 = {(hist0[0][0], 'settled'): ns}
    st1, recs1 = _search(lat, env, z0, 10000, overrides=ov1)
    ok1, rep1 = _arith_ok(st1, recs1)
    h1 = _confirmed_candidates(st1)
    ck(8, 'one not_settled confirmation: unresolved_after_confirmation, the search continues past it and terminates '
          'normally (arithmetic holds)',
       ok1 and h1[0] == (hist0[0][0], True) and len(h1) > 1 and st1['termination']['reason'] not in
       ('stop_for_review_not_settled', 'harness_error'), {'termination': st1['termination']['reason'], 'n_conf': len(h1)})
    res['one'] = {'termination': st1['termination'], 'n_confirmations': len(h1)}

    def build(targets):
        """Mark, in turn, the confirmation at each index of `targets` (on the path the earlier marks produce)."""
        ov, st = {}, base
        for idx in targets:
            h = _confirmed_candidates(st)
            if len(h) <= idx:
                return None, None, f'only {len(h)} confirmations on the path'
            ov[(h[idx][0], 'settled')] = ns
            st, _r = _search(lat, env, z0, 10000, overrides=dict(ov))
        return st, _r, None
    st3, r3, why3 = build([0, 1, 2])
    ok3 = why3 is None and st3['termination']['reason'] == 'stop_for_review_not_settled'
    det3 = (st3 or {}).get('termination') if why3 is None else why3
    ck(8, '3 consecutive not_settled confirmations stop the search for review (stop_for_review_not_settled, the '
          'consecutive rule)', ok3 and det3['detail']['rule_consecutive'] and det3['detail']['n_confirmations'] == 3,
       det3)
    if ok3:
        ok3b, _rep = _arith_ok(st3, r3)
        ck(8, 'the not_settled stop: terminal arithmetic holds (partition == |N|, roles == budget == evaluated set)',
           ok3b, _rep['evaluations'])
    st4, r4, why4 = build([1, 3])
    det4 = (st4 or {}).get('termination') if why4 is None else why4
    ck(8, 'not_settled on 2 of 4 confirmations (> 25 %, >= 4) stops the search for review (the share rule)',
       why4 is None and st4['termination']['reason'] == 'stop_for_review_not_settled'
       and det4['detail']['rule_share'] and not det4['detail']['rule_consecutive']
       and det4['detail']['n_confirmations'] == 4 and det4['detail']['n_not_settled'] == 2, det4)
    st5, r5, why5 = build([1])
    h5 = _confirmed_candidates(st5) if st5 else []
    ck(8, 'not_settled on 1 confirmation only: never a review stop, however many confirmations follow (1 of >= 4 is '
          '<= 25 %)', why5 is None and st5['termination']['reason'] != 'stop_for_review_not_settled' and len(h5) >= 4,
       {'termination': (st5 or {}).get('termination', {}).get('reason'), 'n_conf': len(h5)})
    ck(8, 'the driver maps stop_for_review_not_settled to exit 5 (review), and it is a policy termination',
       'stop_for_review_not_settled' in D.REVIEW_TERMINATIONS and 'stop_for_review_not_settled' in P.TERMINATIONS)
    if st3 is not None:
        rep3 = P.terminal_report(st3, r3)
        n_flag = sum(1 for u in rep3['unresolved_final_poll'] if u['not_settled'])
        n_ns_final = sum(1 for c in st3['polls'][-1]['classified']
                         if (c.get('confirmation') or {}).get('not_settled') is True)
        ck(8, 'terminal report: the final poll\'s not_settled confirmations are listed unresolved_after_confirmation '
              'with the flag, and the review record is reported',
           rep3['not_settled_review']['stop'] and n_flag == n_ns_final >= 1
           and rep3['class_partition_final_poll']['unresolved_after_confirmation'] >= n_flag,
           {'review': rep3['not_settled_review'], 'flagged': n_flag, 'final_poll_not_settled': n_ns_final})
    res.update({'three_consecutive': det3, 'two_of_four': det4,
                'one_of_many': {'termination': (st5 or {}).get('termination'), 'n_confirmations': len(h5)}})
    return res


# ======================================================================================================================
#  item 6 (lattice side + collision scan)
# ======================================================================================================================
def item6_lattice(lat, data_root, out_dir, ctx):
    domain = ctx['domain']
    keys = [lat.key(z) for z in domain]
    ck(6, '37,042 distinct GeneralLattice.key over the domain', len(set(keys)) == len(domain) == D.DOMAIN_SIZE,
       len(set(keys)))
    ck(6, 'labels distinct; canonical v2 round trip z -> canonical -> z exact on every plan',
       len({lat.label(z) for z in domain}) == len(domain)
       and all(lat.z_of_canonical(lat.canonical_of_z(z)) == z for z in domain))
    specs = {}
    for arm in ('ev_1x', 'ev_2x'):
        specs[arm] = {'arm': arm, 'oracle': {'kind': 'synthetic', 'seed': 's1'}}
    syn = [D.synthetic_eval_key(lat, specs[a], z, d) for a in specs for d in P.DEPTHS for z in domain]
    ck(6, 'synthetic eval keys distinct over {1x, 2x} x {exit, settled} x domain (148,168)',
       len(set(syn)) == 4 * D.DOMAIN_SIZE, len(set(syn)))
    # the collision scan excludes the search's own root (and the declared cache roots), and names what it found
    t = os.path.join(out_dir, 'item6_collision_tree')
    own, other, cache_root = (os.path.join(t, 'own_root'), os.path.join(t, 'other_campaign'),
                              os.path.join(t, 'cache_root'))
    for d in (own, other, cache_root):
        os.makedirs(d)
    mine = keys[:3]
    D.write_once_json(os.path.join(own, 'campaign_spec_own.json'), {'candidates': [{'eval_key': k} for k in mine]})
    D.write_once_json(os.path.join(cache_root, 'campaign_spec_cache.json'), {'candidates': [{'eval_key': mine[0]}]})
    D.write_once_json(os.path.join(other, 'campaign_spec_other.json'), {'candidates': [{'eval_key': 'f' * 64}]})
    files = sorted(D.files_under(t))
    a = D.collision_scan(files, mine, own, allowed_roots=[cache_root])
    ck(6, 'collision scan: own root and declared cache root excluded -> no collision, no refusal',
       not a['collisions'] and not a['refusals'] and len(a['excluded_own_root']) == 1
       and len(a['excluded_cache_roots']) == 1, a)
    D.write_once_json(os.path.join(other, 'campaign_spec_other2.json'), {'x': [{'eval_key': mine[1]}]})
    b = D.collision_scan(sorted(D.files_under(t)), mine, own, allowed_roots=[cache_root])
    ck(6, 'collision scan: a foreign artefact carrying a search key is reported, and the refusal names it',
       len(b['collisions']) == 1 and b['refusals'] and 'campaign_spec_other2.json' in b['refusals'][0], b)
    c = D.collision_scan(sorted(D.files_under(t)), mine, t)
    ck(6, 'collision scan with the wrong own root (the parent) excludes everything: guard against a vacuous scan '
          'is the caller\'s (recorded: n_files_scanned 0)', c['n_files_scanned'] == 0, c)
    committed = D.committed_spec_files(data_root)
    s = D.collision_scan(committed, set(keys), os.path.join(data_root, D.RESULTS_PARENT_REL, 'search_driver_v1'))
    ck(6, 'committed campaign/search specs and launch records under data/SRP1/Results carry no GeneralLattice key '
          'as an eval key', not s['collisions'], {'n_files_scanned': s['n_files_scanned'],
                                                   'collisions': s['collisions'][:3]})
    return {'n_keys': len(set(keys)), 'n_committed_files_scanned': s['n_files_scanned']}


# ======================================================================================================================
#  item 9
# ======================================================================================================================
def _old_records(d):
    """{z-canonical-old: summary} of one published history's cache (cache table at start + points + incumbents)."""
    out = {}

    def add(canon, status, q, bar, label):
        if canon is None:
            return
        key = P.canonical_json(canon)
        st = 'certified' if status == 'certified' else ('not_certified' if status == 'not_certified'
                                                        else 'harness_error')
        out[key] = {'canonical': canon, 'status': st, 'Q': q, 'bar': bar, 'label': label}
    for _k, c in (d.get('cache_table_at_start') or {}).items():
        add(c.get('canonical'), c.get('status'), c.get('Q'), c.get('bar'), c.get('label'))
    for lab, p in (d.get('points') or {}).items():
        q = p.get('Q', p.get('certified_cost_gross_settlement_excluded'))
        bar = p.get('bar')
        if isinstance(bar, dict):
            bar = bar.get('value')
        if isinstance(q, dict):
            q = q.get('value')
        add(p.get('candidate_canonical'), p.get('status'), q, bar, lab)
    return out


def _z_of_old(lat, canon):
    nodes = {int(n): {int(canon['investment_year']): (v[0], v[1])} for n, v in canon['nodes'].items()}
    try:
        return lat.z_of_canonical(lat.canonicalize(nodes))
    except ValueError:
        return None


def item9(lat, data_root):
    env = D.LatticeEnv(lat)
    out = {}
    for name, rel in PUBLISHED:
        p = os.path.join(data_root, rel)
        d = D.load_json(p)
        old = _old_records(d)
        by_z, off_lattice = {}, []
        for rec in old.values():
            z = _z_of_old(lat, rec['canonical'])
            if z is None:
                off_lattice.append(rec['label'])
                continue
            by_z[z] = rec
        incs, seen = [], []
        for ph in d['poll_history']:
            z = tuple(ph['incumbent']['z'])  # old 7-variable z; use the incumbent's label / canonical via eval key
            lab = ph['incumbent']['label']
            if lab not in seen:
                seen.append(lab)
                incs.append(ph['incumbent'])
        hist = []
        prev = None
        for inc in incs:
            # the old 7-variable z (zP5, zE5, zP7, zE7, zP9, zE9, zY; p515_s47_phase_b_record.Lattice), zY = year index
            oz = inc['z']
            if len(oz) != 7:
                raise RuntimeError(f'{name}: incumbent {inc["label"]} z {oz} is not the 7-variable old lattice')
            canon = {'investment_year': lat.years[oz[6]],
                     'nodes': {'5': [0.25 * oz[0], 0.5 * oz[1]], '7': [0.25 * oz[2], 0.5 * oz[3]],
                               '9': [0.25 * oz[4], 0.5 * oz[5]]}}
            zi = _z_of_old(lat, canon)
            if inc['label'] != 'x0' and str(canon['investment_year']) not in inc['label']:
                raise RuntimeError(f'{name}: incumbent {inc["label"]} year mapping disagrees with its label')
            i_new = lat.investment_cost(zi)
            entry = {'incumbent': inc['label'], 'z': list(zi), 'F_published': inc['F'], 'I_published': inc['I'],
                     'I_lattice': i_new, 'I_abs_diff': abs(i_new - inc['I'])}
            inc_sum = {'depth': 'exit', 'status': 'certified', 'F': inc['F'], 'bar': inc['bar']}
            last = None if prev is None else [a - b for a, b in zip(zi, prev)]
            o = P.order(lat, zi, last, P.CLASS_RANK, env.neighbourhood_points(zi))
            covered, classes, first_cc = 0, {}, None
            for r in o:
                rec2 = by_z.get(tuple(r['z']))
                if rec2 is None:
                    continue
                covered += 1
                fx = (lat.investment_cost(tuple(r['z'])) + rec2['Q']) if rec2['Q'] is not None else None
                cls = P.classify_exit({'depth': 'exit', 'status': rec2['status'], 'F': fx, 'bar': rec2['bar']},
                                      inc_sum)
                classes[cls['class']] = classes.get(cls['class'], 0) + 1
                if cls['class'] == 'confirm_candidate' and first_cc is None:
                    first_cc = {'pos': r['pos'], 'label': r['label'], 'dF_exit': cls['dF'], 'e': cls['e']}
            entry.update({'N': len(o), 'covered': covered, 'coverage': covered / len(o) if o else None,
                          'covered_classes': classes, 'first_confirm_candidate_among_covered': first_cc,
                          'full_coverage': covered == len(o)})
            hist.append(entry)
            prev = zi
        out[name] = {'path': rel, 'sha256': sha(p), 'tracked_clean': tracked_clean(p, data_root),
                     'n_cache_records': len(old), 'n_off_lattice': len(off_lattice), 'incumbents': hist}
        ck(9, f'{name}: published history replayed ({len(hist)} incumbent(s)); coverage of N(x) reported',
           all(h['N'] > 0 for h in hist) and out[name]['tracked_clean'],
           [(h['incumbent'], f"{h['covered']}/{h['N']}") for h in hist])
        ck(9, f'{name}: I(x) of every published incumbent recomputed on the 3x3 lattice within 0.01 EUR',
           all(h['I_abs_diff'] <= 0.01 for h in hist), [h['I_abs_diff'] for h in hist])
    return out


# ======================================================================================================================
#  driver subprocess helpers (items 4, 5)
# ======================================================================================================================
def drv(argv, data_root, log_prefix, env_extra=None, new_session=True):
    env = dict(os.environ)
    env.pop(D.KILL_ENV, None)
    env.update(env_extra or {})
    with open(log_prefix + '_stdout.log', 'w') as so, open(log_prefix + '_stderr.log', 'w') as se:
        p = subprocess.run([sys.executable, '-u', os.path.join(CODE_ROOT, 'p515_s56_search_driver.py'), *argv,
                            '--data-root', data_root], stdout=so, stderr=se, env=env, cwd=CODE_ROOT,
                           start_new_session=new_session)
    return p.returncode


def run_search(root, data_root, lock, logs, kill=None, max_launches=400, env_extra=None):
    """Re-issue --run until the driver stops returning 4 (the author's foreground loop). `kill` = (transition, n):
    the launch carrying it gets the kill env; the following launches resume with --take-over-stale-lock."""
    codes, killed = [], False
    for i in range(max_launches):
        extra = dict(env_extra or {})
        if kill and not killed:
            extra[D.KILL_ENV] = f'{kill[0]}:{kill[1]}:{kill[2] if len(kill) > 2 else 0.3}'
        argv = ['--run', '--root', root, '--lock-path', lock]
        if killed:
            argv.append('--take-over-stale-lock')
        rc = drv(argv, data_root, os.path.join(logs, f'run_{i:03d}'), extra)
        codes.append(rc)
        if rc == -9:
            killed = True
            continue
        if rc != D.EXIT_SUSPENDED:
            return rc, codes
    return None, codes


def final_history(root):
    if not os.path.exists(os.path.join(root, D.STATE_VIEW)):
        return {'state': None, 'history': None, 'evaluated': None, 'budget': None, 'results_state': None,
                'n_launches': None}
    view = D.load_json(os.path.join(root, D.STATE_VIEW))
    res = D.load_json(os.path.join(root, D.RESULTS_FILE)) if os.path.exists(os.path.join(root, D.RESULTS_FILE)) \
        else None
    st = view['state']
    return {'state': P.canonical_json(st), 'history': P.canonical_json(st['incumbent_history']),
            'evaluated': P.canonical_json(sorted([e['z'], e['depth']] for e in st['evaluated'])),
            'budget': st['budget_used'], 'results_state': P.canonical_json(res['state']) if res else None,
            'n_launches': view['n_launches']}


def item5_synthetic(lat, data_root, out_dir):
    base = os.path.join(out_dir, 'item5')
    os.makedirs(base)
    lock = os.path.join(base, 'scratch_campaign.lock')

    def fresh(tag):
        root = os.path.join(base, tag)
        logs = os.path.join(base, f'{tag}_logs')
        os.makedirs(logs)
        rc = drv(['--freeze', '--root', root, '--search-id', 'gate5', '--arm', 'ev_2x', '--start', 'reference_unit',
                  '--n-max', '60', '--k-ev', '400', '--oracle', 'synthetic', '--synthetic-seed', 'g5', '--test-only'],
                 data_root, os.path.join(logs, 'freeze'))
        return root, logs, rc
    root0, logs0, rc = fresh('uninterrupted')
    rc, codes = run_search(root0, data_root, lock, logs0)
    ref = final_history(root0)
    ck(5, 'uninterrupted synthetic search terminates (exit 0) through repeated launches',
       rc == 0 and set(codes[:-1]) == {4}, {'codes': codes[-3:], 'n': len(codes)})
    # the in-process run_to_end reaches the same final state as the launch machinery
    spec = D.load_spec(root0)[2]
    land = D.synthetic_landscape_of(spec, lat)
    st, _r = P.run_to_end(P.initial_state(D.policy_config(spec)), {}, D.LatticeEnv(lat),
                          lambda a: {(tuple(z), a['depth']): land.summary(z, a['depth']) for z in a['zs']})
    ck(5, 'launch machinery final state == in-process run_to_end (byte-equal)', P.canonical_json(st) == ref['state'])
    lr = [D.load_json(D.launch_paths(root0, n)[1]) for n in range(1, ref['n_launches'] + 1)]
    conf = next(i for i, x in enumerate(lr, 1) if x['action']['purpose'] == 'confirmation')
    blk = next(i for i, x in enumerate(lr, 1) if x['action']['purpose'] == 'block' and i > 1)
    last = len(lr)
    targets = [(t, n) for t in D.KILL_TRANSITIONS if t not in ('after_results', 'during_evaluate')
               for n in (1, blk, conf, last)] + [('during_evaluate', n, 0.0) for n in (blk, conf)] + \
        [('after_results', last)]
    rows = []
    for k, tgt in enumerate(targets):
        root, logs, _rc = fresh(f'kill_{k:02d}_{tgt[0]}_l{tgt[1]}')
        rc, codes = run_search(root, data_root, lock, logs, kill=tgt)
        h = final_history(root)
        idir = os.path.join(root, 'interrupted')
        irecs = [D.load_json(os.path.join(idir, n)) for n in sorted(os.listdir(idir))
                 if n.endswith('.json')] if os.path.isdir(idir) else []
        killed_evals = sum(len(r['killed_evaluations']) for r in irecs)
        stale = os.path.isdir(os.path.join(root, 'stale_locks'))
        same = (h['state'] == ref['state'] and h['budget'] == ref['budget'] and h['results_state'] == ref['results_state']
                and h['n_launches'] == ref['n_launches'])
        rows.append({'target': list(tgt), 'exit': rc, 'codes_has_kill': -9 in codes, 'same_history': same,
                     'interrupted_records': len(irecs), 'killed_evaluations_uncounted': killed_evals,
                     'stale_lock_taken_over': stale})
        ck(5, f'kill at {tgt[0]} (launch {tgt[1]}) + resume: history byte-equal, budget equal, killed evaluations '
              f'uncounted ({killed_evals}), stale lock taken over', rc == 0 and -9 in codes and same and stale
           and all(not e['counted'] for r in irecs for e in r['killed_evaluations']), rows[-1])
    return {'reference': {k: v for k, v in ref.items() if k in ('budget', 'n_launches')}, 'kills': rows,
            'launch_kinds': {'first_block': 1, 'block': blk, 'confirmation': conf, 'last': last}}


def item5_refusals(lat, data_root, out_dir):
    """Tamper checks: the replay refuses (exit 2, nothing written) when a record or the view is altered; a live lock
    is never taken over; a lock naming another search is never taken over."""
    base = os.path.join(out_dir, 'item5_refusals')
    os.makedirs(base)
    lock = os.path.join(base, 'scratch_campaign.lock')
    root = os.path.join(base, 'root')
    logs = os.path.join(base, 'logs')
    os.makedirs(logs)
    drv(['--freeze', '--root', root, '--search-id', 'gate5r', '--arm', 'ev_2x', '--start', 'x0', '--n-max', '12',
         '--k-ev', '400', '--oracle', 'synthetic', '--synthetic-seed', 'g5r', '--test-only'], data_root,
        os.path.join(logs, 'freeze'))
    for i in range(3):
        drv(['--run', '--root', root, '--lock-path', lock], data_root, os.path.join(logs, f'run_{i}'))
    # 1. a live lock (this gate's pid) naming the search: refused even with the takeover flag
    with open(lock, 'w') as h:
        json.dump({'pid': os.getpid(), 'campaign_id': 'gate5r_l004'}, h)
    rc1 = drv(['--run', '--root', root, '--lock-path', lock, '--take-over-stale-lock'], data_root,
              os.path.join(logs, 'live_lock'))
    os.remove(lock)
    # 2. a dead pid lock naming ANOTHER search: refused
    with open(lock, 'w') as h:
        json.dump({'pid': 999999, 'campaign_id': 'other_search_l001'}, h)
    rc2 = drv(['--run', '--root', root, '--lock-path', lock, '--take-over-stale-lock'], data_root,
              os.path.join(logs, 'other_lock'))
    # 3. a dead pid lock naming this search, WITHOUT the explicit flag: refused
    with open(lock, 'w') as h:
        json.dump({'pid': 999999, 'campaign_id': 'gate5r_l004'}, h)
    rc3 = drv(['--run', '--root', root, '--lock-path', lock], data_root, os.path.join(logs, 'no_flag'))
    os.remove(lock)
    ck(5, 'a live lock, a lock naming another search, and a stale lock without --take-over-stale-lock are refused',
       (rc1, rc2, rc3) == (2, 2, 2), [rc1, rc2, rc3])
    # 4. tampered evaluation record: refused, nothing written
    rec = D.load_json(D.launch_paths(root, 2)[1])['records'][0]
    rf = os.path.join(root, rec['record_file'])
    before = sorted(D.files_under(root))
    orig = open(rf).read()
    os.chmod(rf, 0o644)
    with open(rf, 'w') as h:
        h.write(orig.replace('"F": ', '"F": 1', 1))
    rc4 = drv(['--run', '--root', root, '--lock-path', lock], data_root, os.path.join(logs, 'tampered_record'))
    after = sorted(D.files_under(root))
    with open(rf, 'w') as h:
        h.write(orig)
    ck(5, 'a tampered evaluation record: the replay refuses (exit 2) and writes nothing', rc4 == 2 and before == after,
       rc4)
    # 5. tampered state view: refused
    vp = os.path.join(root, D.STATE_VIEW)
    view = D.load_json(vp)
    view_orig = open(vp).read()
    view['state']['budget_used'] += 1
    with open(vp, 'w') as h:
        json.dump(view, h)
    rc5 = drv(['--run', '--root', root, '--lock-path', lock], data_root, os.path.join(logs, 'tampered_view'))
    with open(vp, 'w') as h:
        h.write(view_orig)
    ck(5, 'a state view that does not equal the replayed state: refused (exit 2)', rc5 == 2, rc5)
    rc6 = drv(['--run', '--root', root, '--lock-path', lock], data_root, os.path.join(logs, 'restored'))
    ck(5, 'restored files: the launch proceeds (exit 0 or 4)', rc6 in (0, 4), rc6)
    # 6. the kill env on a non-test spec is refused (no such spec can be frozen with a synthetic oracle: checked)
    rc7 = drv(['--freeze', '--root', os.path.join(base, 'not_test'), '--search-id', 'x', '--n-max', '5', '--k-ev', '5',
               '--oracle', 'synthetic'], data_root, os.path.join(logs, 'freeze_not_test'))
    ck(5, 'a synthetic-oracle spec without --test-only is refused at freeze', rc7 == 2, rc7)
    return {'codes': [rc1, rc2, rc3, rc4, rc5, rc6, rc7]}


# ======================================================================================================================
#  items 4, 5 (stub children) and 6 (harness keys)
# ======================================================================================================================
K_EV_TEST = 400       # a placeholder cap for the settled depth in these zero-solve tests (K_EV is set after the pilot)
EXIT_CAP_TEST = 500   # a placeholder exit cap (stub children never run a cycle)
COHORT_MODES = ('legacy_h3', 'independent_units', 'independent_units_all_node_years')


def _harness():
    import p515_s44_campaign_harness as H
    return H


def settled_stop_key(H):
    """The harness entry option that disables the certification stop on a settled EV evaluation AND enters the eval
    key (the design's harness requirement)."""
    key = getattr(H, 'S56_SETTLED_STOP_DISABLED_KEY', None)
    if key is None:
        raise RuntimeError('the harness has no keyed settled stop-disabled option for EV entries')
    return key


def harness_config(H, *, flex, concurrency, ev=True, cohort='independent_units_all_node_years', tv=True):
    """The per-depth harness configuration of the tests: the harness Worker's smoke configuration (by import:
    `p515_s56_ev_harness_tests._spec_like` = the step 4 spec-like declarations -- case-file AA, ESS ageing baseline,
    convergence-depth tail -- and `srp1grid_ev_declaration` = the SRP1-grid EV twin) with the S56 EV-plan keys; the
    settled depth adds the keyed stop-disabled option and the cap K_EV_TEST. `ev` False = the draw-0 instance (no
    derived instance, no EV profiles)."""
    import p515_s56_ev_harness_tests as EVT
    derived, evp = EVT.srp1grid_ev_declaration()
    cfg, required = EVT._spec_like(derived if ev else None, cap=EXIT_CAP_TEST)
    opts = {H.S56_COHORT_FORMULATION_KEY: cohort, H.S56_TV_IN_OBJECTIVE_KEY: tv, H.S56_ESSO_SLACK_PENALTY_KEY: 1e4,
            'flex_price_multiplier': flex}
    if ev:
        opts[H.S56_EV_PROFILES_KEY] = dict(evp)
    settled = dict(opts, **{settled_stop_key(H): True})
    return {'by_depth': {'exit': {'configuration': cfg, 'cap': EXIT_CAP_TEST, 'entry_options': opts},
                         'settled': {'configuration': cfg, 'cap': K_EV_TEST, 'entry_options': settled}},
            'concurrency': concurrency, 'required_consecutive_cycles': required}


def _oracle_for(lat, hc, arm='k'):
    spec = {'oracle': {'kind': 'stub_children', 'seed': 'keys', 'target': list(lat.x0())}, 'start': {'z': list(lat.x0())},
            'harness': hc, 'arm': arm, 'test_only': True, 'search_id': 'keys'}
    return D.HarnessOracle(spec, lat, lock_path=None)


def _max_overlap(root):
    """Max number of stub children alive at once within one launch (from the stub records' started_t / ended_t)."""
    best = 0
    ld = os.path.join(root, 'launches')
    for name in sorted(os.listdir(ld)):
        ev = os.path.join(ld, name, 'evals')
        if not os.path.isdir(ev):
            continue
        spans = []
        for e in os.listdir(ev):
            rp = os.path.join(ev, e, 'evaluation_record.json')
            if os.path.isfile(rp):
                r = D.load_json(rp)
                spans.append((r['started_t'], r['ended_t']))
        for t, _e in spans:
            best = max(best, sum(1 for a, b in spans if a <= t < b))
    return best


STUB_FLEX = 1.0   # the stub searches run the 1x configuration: no committed evaluation carries its keys (the 2x
                 # configuration is the harness smoke's, whose committed spec the collision scan names -- item 4 control)


def _stub_search(tag, lat, data_root, base, conc, n_max, seed, kill=None, flex=STUB_FLEX):
    H = _harness()
    root, logs = os.path.join(base, tag), os.path.join(base, f'{tag}_logs')
    os.makedirs(logs)
    hc_path = os.path.join(base, f'{tag}_harness_config.json')
    D.write_once_json(hc_path, harness_config(H, flex=flex, concurrency=conc))
    rc0 = drv(['--freeze', '--root', root, '--search-id', 'g4stub', '--arm', f'ev_{flex:g}x', '--start', 'x0',
               '--n-max',
               str(n_max), '--k-ev', str(K_EV_TEST), '--oracle', 'stub_children', '--synthetic-seed', seed,
               '--harness-config', hc_path, '--test-only'], data_root, os.path.join(logs, 'freeze'))
    if rc0 != 0:
        return root, rc0, None, []
    rc, codes = run_search(root, data_root, os.path.join(base, f'{tag}_scratch_campaign.lock'), logs, kill=kill,
                           env_extra={D.STUB_DELAY_ENV: f'{"1.5:2.5" if kill else "0.5:1.5"}:{tag}'})
    return root, rc0, rc, codes


def item4(lat, data_root, out_dir):
    base = os.path.join(out_dir, 'item4')
    os.makedirs(base)
    n_max, seed = 40, 'g4'
    # control, re-scoped by integration ruling 6 (601775c0): the smoke alias is ABSENT. The 2x EV configuration is the
    # harness smoke's (ev_harness_v1/smoke, cap 2); before the cap rule its committed spec carried the x0 / unit exit
    # keys and this freeze was refused. Now the freeze passes its collision scan, and no key of the smoke spec is a key
    # of the configuration's domain (74,084 keys, both depths).
    root_c = os.path.join(base, 'control_2x_smoke_alias')
    hc_c = os.path.join(base, 'control_2x_smoke_alias_harness_config.json')
    os.makedirs(os.path.join(base, 'control_2x_smoke_alias_logs'))
    D.write_once_json(hc_c, harness_config(_harness(), flex=2.0, concurrency=5))
    rc_c = drv(['--freeze', '--root', root_c, '--search-id', 'g4ctl', '--arm', 'ev_2x', '--start', 'x0', '--n-max',
                str(n_max), '--k-ev', str(K_EV_TEST), '--oracle', 'stub_children', '--synthetic-seed', seed,
                '--harness-config', hc_c, '--test-only'], data_root,
               os.path.join(base, 'control_2x_smoke_alias_logs', 'freeze'))
    flog = open(os.path.join(base, 'control_2x_smoke_alias_logs', 'freeze_stdout.log')).read()
    smoke_rel = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'ev_harness_v1', 'smoke',
                             'campaign_s56evh_v1_smoke', 'campaign_spec_s56evh_v1_smoke_39882ca6.json')
    smoke_keys = {e['eval_key'] for e in D.load_json(os.path.join(data_root, smoke_rel))['candidates']}
    tracked = os.path.join(data_root, smoke_rel) in D.committed_spec_files(data_root)
    detail = {'freeze_exit': rc_c, 'log_tail': flog[-300:], 'smoke_spec_tracked_and_scanned': tracked}
    ok = False
    if rc_c == 0:
        sp = D.load_spec(root_c)[2]
        orc = D.HarnessOracle(sp, lat, lock_path=None)
        keys = {orc.eval_key(z, d) for z in D.enumerate_domain(lat) for d in P.DEPTHS}
        cs = sp.get('collision_scan') or {}
        detail.update({'n_domain_keys': len(keys), 'smoke_keys': len(smoke_keys), 'hits': len(keys & smoke_keys),
                       'collision_scan': cs})
        ok = (len(keys) == 2 * D.DOMAIN_SIZE and not (keys & smoke_keys) and tracked and cs.get('collisions') == []
              and cs.get('n_keys') == 2 * D.DOMAIN_SIZE and cs.get('n_files_scanned', 0) > 0)
    ck(4, 'smoke alias ABSENT (control, re-scoped by ruling 6): the 2x EV stub configuration FREEZES, its collision '
          'scan (committed smoke spec in scope) finds nothing, and none of the smoke spec\'s eval keys is among the '
          'configuration\'s 74,084 domain keys', ok, detail)
    runs = {}
    for conc in (1, 5):
        root, rc0, rc, codes = _stub_search(f'conc{conc}', lat, data_root, base, conc, n_max, seed)
        runs[conc] = {'root': root, 'freeze': rc0, 'exit': rc, 'n_launches': len(codes), 'h': final_history(root),
                      'max_overlap': _max_overlap(root)}
    spec = D.load_spec(runs[1]['root'])[2]
    land = D.synthetic_landscape_of(spec, lat)
    st, _r = P.run_to_end(P.initial_state(D.policy_config(spec)), {}, D.LatticeEnv(lat),
                          lambda a: {(tuple(z), a['depth']): land.summary(z, a['depth']) for z in a['zs']})
    ref = {'state': P.canonical_json(st), 'history': P.canonical_json(st['incumbent_history']),
           'evaluated': P.canonical_json(sorted([e['z'], e['depth']] for e in st['evaluated'])), 'budget': st['budget_used']}
    ck(4, 'stub-children searches (concurrency 1 and 5) freeze and terminate through repeated launches',
       all(r['freeze'] == 0 and r['exit'] in (0, 5) for r in runs.values()),
       {c: [r['freeze'], r['exit'], r['n_launches']] for c, r in runs.items()})
    for key in ('history', 'evaluated', 'budget', 'state'):
        ck(4, f'{key}: serial stub == 5-way stub == in-process synthetic replay',
           runs[1]['h'][key] == runs[5]['h'][key] == ref[key], {c: str(r['h'][key])[:80] for c, r in runs.items()})
    ck(4, 'concurrency observed from the stub records: 1 in the serial run, 5 in the 5-way run',
       runs[1]['max_overlap'] == 1 and runs[5]['max_overlap'] == 5, {c: r['max_overlap'] for c, r in runs.items()})
    durs = []
    for name in D.files_under(os.path.join(runs[5]['root'], 'launches')):
        if name.endswith('evaluation_record.json'):
            r = D.load_json(name)
            durs.append(round(r['ended_t'] - r['started_t'], 2))
    ck(4, 'stub delays randomised per child (distinct sleeps)', len(set(durs)) > len(durs) // 2,
       {'n': len(durs), 'distinct': len(set(durs))})
    return {c: {k: v for k, v in r.items() if k != 'h'} | {'budget': r['h']['budget']} for c, r in runs.items()}


def item5_stub(lat, data_root, out_dir, _unused=None):
    base = os.path.join(out_dir, 'item5_stub')
    os.makedirs(base)
    ref_root = os.path.join(out_dir, 'item4', 'conc5')
    ref = final_history(ref_root)
    lr = [D.load_json(D.launch_paths(ref_root, n)[1]) for n in range(1, ref['n_launches'] + 1)]
    blk = next(i for i, x in enumerate(lr, 1) if x['action']['purpose'] == 'block' and len(x['action']['zs']) == 5)
    conf = next(i for i, x in enumerate(lr, 1) if x['action']['purpose'] == 'confirmation')
    rows = []
    for n in (blk, conf):
        tag = f'kill_during_evaluate_l{n}'
        root, _rc0, rc, codes = _stub_search(tag, lat, data_root, base, 5, 40, 'g4', kill=('during_evaluate', n, 0.4))
        h = final_history(root)
        idir = os.path.join(root, 'interrupted')
        irecs = [D.load_json(os.path.join(idir, f)) for f in sorted(os.listdir(idir))
                 if f.endswith('.json')] if os.path.isdir(idir) else []
        killed = [e for r in irecs for e in r['killed_evaluations']]
        same = all(h[k] == ref[k] for k in ('state', 'history', 'evaluated', 'budget', 'n_launches'))
        rows.append({'launch': n, 'exit': rc, 'killed': len(killed), 'with_record': sum(e['record_present'] for e in
                                                                                        killed), 'same': same})
        ck(5, f'stub children: kill of the driver and its {len(lr[n - 1]["action"]["zs"])} running children during '
              f'evaluate (launch {n}) + resume: history byte-equal to the uninterrupted 5-way run, killed evaluations '
              f'uncounted', rc in (0, 5) and -9 in codes and same and killed and not any(e['counted'] for e in killed),
           rows[-1])
    return rows


def item6_harness(lat, data_root, out_dir, ctx):
    H = _harness()
    domain = ctx['domain']
    t0 = time.time()
    out = {}
    import p515_s56_ev_harness_tests as EVT
    derived, _evp = EVT.srp1grid_ev_declaration()
    for tag, d in (('ev_twin', derived), ('draw0_srp1', None)):
        case_rel = H.s56_case_rel(d)
        hl = H.s56_lattice(case_rel)
        bad = [list(z) for z in domain
               if H.candidate_key(H.s56_staged_canonical(D.staged_plan_of(lat, z), case_rel, lattice=hl)[0])
               != lat.key(z)]
        ck(6, f'staged key (harness s56_staged_canonical + candidate_key, case {tag}) == GeneralLattice.key(z) on all '
              f'37,042 plans', not bad, bad[:3])
    configs = list(product((1.0, 2.0), P.DEPTHS, (True, False), COHORT_MODES, (False, True)))
    allkeys, per_cfg_ok = set(), True
    per_cfg = {}
    for flex, depth, ev, cohort, tv in configs:
        orc = _oracle_for(lat, harness_config(H, flex=flex, concurrency=5, ev=ev, cohort=cohort, tv=tv))
        ks = {bytes.fromhex(orc.eval_key(z, depth)) for z in domain}
        name = f'{flex:g}x|{depth}|{"EV" if ev else "draw0"}|{cohort}|TV{tv}'
        per_cfg[name] = len(ks)
        per_cfg_ok &= len(ks) == D.DOMAIN_SIZE
        allkeys |= ks
    ck(6, f'37,042 distinct eval keys in each of the {len(configs)} configurations', per_cfg_ok,
       {k: v for k, v in per_cfg.items() if v != D.DOMAIN_SIZE})
    ck(6, 'eval keys pairwise distinct across {1x, 2x} x {exit, settled} x {EV, draw-0} x cohort formulation (3) x TV '
          f'(2): {len(configs)} x 37,042 = {len(configs) * D.DOMAIN_SIZE:,}', len(allkeys) == len(configs) * D.DOMAIN_SIZE,
       len(allkeys))
    # the composition == what freeze_campaign_spec freezes (named points x every configuration)
    fz = os.path.join(out_dir, 'item6_freeze')
    os.makedirs(fz)
    mism = []
    names = {n: D.named_start(lat, n) for n in NAMED_SIZES}
    for i, (flex, depth, ev, cohort, tv) in enumerate(configs):
        hc = harness_config(H, flex=flex, concurrency=5, ev=ev, cohort=cohort, tv=tv)
        orc = _oracle_for(lat, hc)
        bd = hc['by_depth'][depth]
        cands = [(f'{n}_{depth}', None, orc.entry_options(z, depth)) for n, z in names.items()]
        _p, _s, hspec = H.freeze_campaign_spec(os.path.join(fz, f'cfg_{i:02d}'), f'keys_{i:02d}', cands,
                                               dict(bd['configuration'], name=f'keys {i}'), bd['cap'], 5, AUTH_TEST,
                                               required_consecutive_cycles=hc['required_consecutive_cycles'],
                                               extra={'test_only_stub': True})
        for e in hspec['candidates']:
            z = names[e['label'].rsplit('_', 1)[0]]
            if e['eval_key'] != orc.eval_key(z, depth) or e['key'] != lat.key(z):
                mism.append([i, e['label']])
    ck(6, f'the driver key composition == the eval key freeze_campaign_spec freezes (3 named points x {len(configs)} '
          'configurations)', not mism, mism[:5])
    out.update({'n_configurations': len(configs), 'n_keys': len(allkeys), 'wall_s': time.time() - t0})
    return out


AUTH_TEST = ['p515_s56_search_gate.py item 6 (zero-solve key checks; scratch specs)']
PRE_SETTLED_KEY_REV = 'b0391b80'   # the harness before the settled-run key (the harness Worker's last commit)
VOLATILE_SPEC_FIELDS = ('frozen_utc', 'harness', 'git_head')


def _spec_body(path):
    s = D.load_json(path)
    return P.canonical_json({k: v for k, v in s.items() if k not in VOLATILE_SPEC_FIELDS})


def item6_settled_key(lat, data_root, out_dir):
    """The settled-run key added to the harness by this task (design: "Harness requirement"): refusals, keying, and
    the OPTION-OFF IDENTITY against the harness at PRE_SETTLED_KEY_REV (module diff = insertions only; frozen specs of
    entries without the key byte-identical but for the volatile fields; record fields empty when absent)."""
    H = _harness()
    import p515_s56_ev_harness_tests as EVT
    base = os.path.join(out_dir, 'item6_settled_key')
    os.makedirs(base)
    sk = H.S56_SETTLED_STOP_DISABLED_KEY
    diff = git(['diff', '-U0', PRE_SETTLED_KEY_REV, '--', 'p515_s44_campaign_harness.py'], CODE_ROOT) or ''
    deleted = [ln for ln in diff.splitlines() if ln.startswith('-') and not ln.startswith('---')]
    added = [ln for ln in diff.splitlines() if ln.startswith('+') and not ln.startswith('+++')]
    ck(6, f'harness vs {PRE_SETTLED_KEY_REV}: insertions only ({len(added)} lines added, 0 deleted)',
       not deleted and added, deleted[:5])
    pre, pre_path = EVT._load_module_at(PRE_SETTLED_KEY_REV, 'p515_s44_campaign_harness.py', base, 'harness_pre')
    ck(6, 'the pre-change harness has no settled key (the change is the only source of it)',
       not hasattr(pre, 'S56_SETTLED_STOP_DISABLED_KEY') and sk in H.EVALUATION_OPTION_KEYS
       and sk not in pre.EVALUATION_OPTION_KEYS)
    derived, evp = EVT.srp1grid_ev_declaration()
    cfg_ev, required = EVT._spec_like(derived, cap=EXIT_CAP_TEST)
    cfg_d0, _r = EVT._spec_like(None, cap=EXIT_CAP_TEST)
    cfg_notail = {k: v for k, v in cfg_ev.items() if k != 'convergence_depth_tail'}
    s56 = {H.S56_COHORT_FORMULATION_KEY: 'independent_units_all_node_years', H.S56_TV_IN_OBJECTIVE_KEY: True,
           H.S56_ESSO_SLACK_PENALTY_KEY: 1e4, 'flex_price_multiplier': 2.0}
    ev_opts = dict(s56, **{H.S56_EV_PROFILES_KEY: dict(evp)})
    coupling = dict(EVT.COUPLING_V1_NOQ)
    plans = {n: D.staged_plan_of(lat, D.named_start(lat, n)) for n in NAMED_SIZES}
    k = [0]

    def freeze(mod, cands, cfg, tag, cid=None):
        k[0] += 1
        root = os.path.join(base, f'{tag}_{k[0]:03d}')
        cid = cid or f'g6s_{k[0]:03d}'
        return mod.freeze_campaign_spec(root, cid, cands, dict(cfg, name=f'g6s {cid}'), EXIT_CAP_TEST, 5,
                                        AUTH_TEST, required_consecutive_cycles=required, extra={'test_only_stub': True})

    def refused(cands, cfg):
        try:
            freeze(H, cands, cfg, 'refusal')
            return 'not refused'
        except ValueError as error:
            return str(error)
    r = {
        'no S56 EV-plan key (single-year SRP1 candidate)': refused(
            [('u', {5: (0, 0), 7: (0.25, 1.0), 9: (0, 0)}, {'flex_price_multiplier': 2.0, sk: True})], cfg_d0),
        'per-scenario coupling declared': refused(
            [('u', None, dict(ev_opts, **{H.S56_STAGED_INVESTMENT_KEY: plans['reference_unit'], sk: True,
                                          'interface_coupling': coupling}))], cfg_ev),
        'together with the calibration key': refused(
            [('u', None, dict(ev_opts, **{H.S56_STAGED_INVESTMENT_KEY: plans['reference_unit'], sk: True,
                                          H.S56_CERTIFICATION_STOP_DISABLED_KEY: False}))], cfg_ev),
        'no enabled convergence-depth tail': refused(
            [('u', None, dict(ev_opts, **{H.S56_STAGED_INVESTMENT_KEY: plans['reference_unit'], sk: True}))],
            cfg_notail),
        'non-bool value': refused(
            [('u', None, dict(ev_opts, **{H.S56_STAGED_INVESTMENT_KEY: plans['reference_unit'], sk: 1}))], cfg_ev),
    }
    for name, msg in r.items():
        ck(6, f'settled key True refused: {name}', msg != 'not refused' and sk in msg, msg[:200])
    # allowed and keyed on single-scenario EV and draw-0 S56 entries; False recorded, not keyed
    for tag, cfg, opts in (('ev', cfg_ev, ev_opts), ('draw0', cfg_d0, s56)):
        cands = []
        for n, plan in plans.items():
            o = dict(opts, **{H.S56_STAGED_INVESTMENT_KEY: plan})
            cands += [(f'{n}_exit', None, o), (f'{n}_settled', None, dict(o, **{sk: True}))]
        try:
            _p, _s, sp = freeze(H, cands, cfg, f'allowed_{tag}')
            by = {e['label']: e for e in sp['candidates']}
            ok = all(by[f'{n}_settled']['eval_key'] == H.s56_settled_stop_evaluation_key(by[f'{n}_exit']['eval_key'], True)
                     != by[f'{n}_exit']['eval_key'] and by[f'{n}_settled'][sk] is True
                     and by[f'{n}_settled']['key'] == by[f'{n}_exit']['key'] == lat.key(D.named_start(lat, n))
                     for n in plans) and sk not in by['x0_exit']
            # False is the default: recorded, not keyed (so it cannot sit in the same spec as the exit entry)
            _p2, _s2, spf = freeze(H, [('x0_false', None, dict(opts, **{H.S56_STAGED_INVESTMENT_KEY: plans['x0'],
                                                                        sk: False}))], cfg, f'false_{tag}')
            ef = spf['candidates'][0]
            ok_false = ef[sk] is False and ef['eval_key'] == by['x0_exit']['eval_key']
            detail = None
        except ValueError as error:
            ok, ok_false, detail = False, False, str(error)
        ck(6, f'{tag}: settled key True allowed; eval key = sha256{{base, key}} != the exit key; candidate key '
              f'unchanged; absent -> no entry field', ok, detail)
        ck(6, f'{tag}: settled key False recorded in the entry, not keyed (the W33 default rule)', ok_false, detail)
    # option-off identity: entries without the key freeze identically under the pre- and post-change harness
    sets = {
        'srp1_single_year': ([('x0', {5: (0, 0), 7: (0, 0), 9: (0, 0)}, {'flex_price_multiplier': 2.0}),
                              ('unit', {5: (0, 0), 7: (0.25, 1.0), 9: (0, 0)}, {'flex_price_multiplier': 2.0})], cfg_d0),
        's56_ev_staged': ([(n, None, dict(ev_opts, **{H.S56_STAGED_INVESTMENT_KEY: p})) for n, p in plans.items()],
                          cfg_ev),
        's56_draw0_staged': ([(n, None, dict(s56, **{H.S56_STAGED_INVESTMENT_KEY: p})) for n, p in plans.items()],
                             cfg_d0),
        'calibration_key_coupled': ([(n, None, dict(ev_opts, **{H.S56_STAGED_INVESTMENT_KEY: p, 'interface_coupling': coupling,
                                                            H.S56_CERTIFICATION_STOP_DISABLED_KEY: True}))
                                     for n, p in plans.items()], cfg_ev),
    }
    for name, (cands, cfg) in sets.items():
        try:
            p_pre, _s1, s_pre = freeze(pre, cands, cfg, f'pre_{name}', cid=f'g6s_identity_{name}')
            p_post, _s2, s_post = freeze(H, cands, cfg, f'post_{name}', cid=f'g6s_identity_{name}')
            same = _spec_body(p_pre) == _spec_body(p_post)
            keys = [e['eval_key'] for e in s_pre['candidates']] == [e['eval_key'] for e in s_post['candidates']]
            detail = None
        except ValueError as error:
            same, keys, detail = False, False, str(error)
        ck(6, f'option-off identity ({name}): frozen spec byte-identical pre vs post (volatile fields '
              f'{list(VOLATILE_SPEC_FIELDS)} excluded), eval keys equal', same and keys, detail)
    ck(6, 'option-off identity: record fields empty when the key is absent',
       H.s56_settled_stop_record_fields({'label': 'x'}, {}) == {})
    return {'pre_rev': PRE_SETTLED_KEY_REV, 'lines_added': len(added), 'lines_deleted': len(deleted)}


def item3_driver(lat, data_root, out_dir):
    """Item 3 through the launch machinery: 1x and 2x specs (and a perturbed oracle seed) issue the same first two
    blocks."""
    base = os.path.join(out_dir, 'item3_driver')
    os.makedirs(base)
    lock = os.path.join(base, 'scratch_campaign.lock')
    acts = {}
    for arm, seed in (('ev_1x', 'a'), ('ev_2x', 'a'), ('ev_2x', 'perturbed')):
        root = os.path.join(base, f'{arm}_{seed}')
        logs = os.path.join(base, f'{arm}_{seed}_logs')
        os.makedirs(logs)
        drv(['--freeze', '--root', root, '--search-id', f'g3{arm}{seed}', '--arm', arm, '--start', 'x0', '--n-max',
             '10', '--k-ev', '400', '--oracle', 'synthetic', '--synthetic-seed', seed, '--test-only'], data_root,
            os.path.join(logs, 'freeze'))
        for i in range(2):
            drv(['--run', '--root', root, '--lock-path', lock], data_root, os.path.join(logs, f'run_{i}'))
        acts[f'{arm}_{seed}'] = [D.load_json(D.launch_paths(root, n)[1])['action'] for n in (1, 2)]
    ok = len({P.canonical_json(v) for v in acts.values()}) == 1
    ck(3, 'driver launches: the first two actions are identical at 1x, 2x and under a perturbed oracle', ok,
       {k: [a['zs'] for a in v] for k, v in acts.items()})
    return {'actions': acts}


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    ap.add_argument('--data-root', default=CODE_ROOT)
    ap.add_argument('--items', default='1,2,3,4,5,6,7,8,9')
    ap.add_argument('--item2-worker', action='store_true')
    ap.add_argument('--worker-out')
    ap.add_argument('--shuffle-seed')
    ap.add_argument('--no-harness', action='store_true',
                    help='skip the harness-dependent parts (item 4, item 5 stub children, item 6 harness keys); they '
                         'are listed as pending and the gate exits 3')
    args = ap.parse_args()
    args.data_root = os.path.abspath(args.data_root)
    if args.item2_worker:
        return item2_worker(args)
    items = [s.strip() for s in args.items.split(',') if s.strip()]
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(args.data_root, args.out)
    if os.path.exists(out_dir) and [n for n in os.listdir(out_dir) if not n.endswith('.log')]:
        log(f'REFUSE: output directory {out_dir} exists and holds non-log files')
        return 2
    os.makedirs(out_dir, exist_ok=True)
    started, t0 = datetime.now(timezone.utc).isoformat(), time.time()
    lat, inst = D.build_lattice(args.data_root)
    ctx, results = {}, {}
    pending = []
    if any(i in items for i in ('1', '2', '3', '6')):
        results['1'] = item1(lat, args.data_root, ctx) if '1' in items else None
        if 'domain' not in ctx:
            ctx['domain'] = enumerate_domain(lat)
    if '2' in items:
        results['2'] = item2(lat, args.data_root, out_dir, ctx)
    if '3' in items:
        results['3'] = {'policy': item3(lat, ctx), 'driver': item3_driver(lat, args.data_root, out_dir)}
    if '4' in items:
        if args.no_harness:
            pending.append('4')
        else:
            results['4'] = item4(lat, args.data_root, out_dir)
    if '5' in items:
        results['5'] = {'synthetic': item5_synthetic(lat, args.data_root, out_dir),
                        'refusals': item5_refusals(lat, args.data_root, out_dir)}
        if args.no_harness or '4' not in items:
            pending.append('5 (stub children)')
        else:
            results['5']['stub_children'] = item5_stub(lat, args.data_root, out_dir, None)
    if '6' in items:
        results['6'] = {'lattice': item6_lattice(lat, args.data_root, out_dir, ctx)}
        if args.no_harness:
            pending.append('6 (harness keys)')
        else:
            results['6']['settled_key'] = item6_settled_key(lat, args.data_root, out_dir)
            results['6']['harness'] = item6_harness(lat, args.data_root, out_dir, ctx)
    if '7' in items:
        results['7'] = item7()
    if '8' in items:
        results['8'] = item8(lat)
    if '9' in items:
        results['9'] = item9(lat, args.data_root)
    guard_fail = GUARD.verify(0)
    ck('guards', 'SolveProfileGuard(permitted=()) verify(0) == []', not guard_fail, guard_fail)
    ck('guards', 'pickle load/loads 0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, PICKLE_COUNTS)
    code_files = {n: sha(os.path.join(CODE_ROOT, n)) for n in (
        SCRIPT, 'p515_s56_search_driver.py', 'p515_s56_search_policy.py', 'p515_s56_neighbourhood.py',
        'p516_lattice.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py', 'p513_solve_profile_guard.py')}
    export = os.path.join(CODE_ROOT, 'EXPORT_RECORD.json')
    payload = {'schema': 'p515_s56_search_gate_v1', 'script': SCRIPT, 'started_utc': started,
               'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - t0,
               'interpreter': sys.executable, 'python': platform.python_version(), 'code_root': CODE_ROOT,
               'data_root': args.data_root, 'git_head_data_root': (git(['rev-parse', 'HEAD'], args.data_root) or
                                                                   '').strip(),
               'export_record': D.load_json(export) if os.path.isfile(export) else None,
               'items_requested': items, 'items_pending_harness': pending, 'instance': inst,
               'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) installed before any '
                              'repository import, verify(0)); pickle.load/loads blocked (0); driver subprocesses arm '
                              'their own guard and exit non-zero on a failed verify',
               'values_note': 'every objective value in this gate is SYNTHETIC (landscape or stub), except item 9, '
                              'which reads the published histories\' committed records',
               'results': results, 'guards': {'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                                              'verify_0': guard_fail, 'pickle_counts': dict(PICKLE_COUNTS)},
               'code_files': code_files, 'checks': CHECKS,
               'failed_checks': [f"[{c['item']}] {c['check']}" for c in CHECKS if not c['ok']],
               'all_checks_ok': all(c['ok'] for c in CHECKS)}
    gp = os.path.join(out_dir, 'search_gate.json')
    gsha = D.write_once_json(gp, payload)
    files = [f for f in D.files_under(out_dir) if not f.endswith('_launch_stdout.log') and not
             f.endswith('_launch_stderr.log')]
    D.write_once_json(os.path.join(out_dir, 'manifest_sha256.json'),
                      {'search_gate.json': gsha, 'files': {os.path.relpath(f, out_dir): sha(f) for f in files
                                                           if os.path.basename(f) != 'search_gate.json'},
                       'code_files': code_files})
    log(f'items {items}; pending harness: {pending}; checks {sum(c["ok"] for c in CHECKS)}/{len(CHECKS)} ok; '
        f'wall {time.time() - t0:.0f} s')
    log(f'failed: {payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] and not pending else 3


if __name__ == '__main__':
    sys.exit(main())
