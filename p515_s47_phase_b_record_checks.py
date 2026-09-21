"""
P5.15 Addendum 30 (tasks W23, W24) -- checks of the Phase B record launcher `p515_s47_phase_b_record.py` with a FAKED
evaluate() and a SYNTHETIC cache. ZERO SOLVES: the launcher's module-level SolveProfileGuard(permitted=()) is
installed on import and verified at exactly 0 at the end; nothing here builds a model, takes the campaign lock,
freezes a spec or launches a harness child.

Synthetic cache files (fake campaign_results.json + campaign specs) are written under --scratch (outside the
repository). Read-only inputs from the repository (zero solves): the pinned W2 unit-cost table (I(x)), the
Phase A tables (sigma_Q), the A0 x0 record (Q(0)), the committed S2 results and the frozen S3 spec (eval keys
only) -- used for the expected first poll from x = 0.

Tests:
  (a)  all storage worse -> the incumbent is x = 0 and the run terminates at x = 0 after the poll rules
  (b)  a neighbour better by more than the resolution -> the incumbent moves (Delta doubles)
  (c)  better but within the resolution -> INDETERMINATE, not accepted (Delta halves; unresolved at the end)
  (d)  an over-budget neighbour is never evaluated
  (e)  a non-certified evaluation is an extreme-barrier point (and the barrier stop rule fires)
  plus: cache hits never re-evaluated; batches <= 5; evaluation-budget stop; direction properties; lattice /
  eval-key identity vs the harness; cache loader acceptance / C3 exclusion / duplicate rule on synthetic files;
  the expected poll sequence from x = 0 on the real inputs.
W24 (Planner ruling A2, the unit-poll completion):
  (f)  from x = 0 the completion is exactly the 14 points 0.25/0.5 at every non-empty node subset x {2025, 2030},
       all evaluated in batches 5 + 5 + 4, none cached; all worse -> terminates at x = 0 with the certificate
  (g)  one completion point better by more than the resolution -> accepted, Delta doubles, the run continues
  (h)  a completion of > 30 feasible points at a synthetic interior incumbent -> refused, STOP FOR REVIEW, nothing
       evaluated, the completion not truncated
  (i)  an over-budget (I(x) > B) completion point is never evaluated (synthetic B = 400,000 EUR so that the x = 0
       completion contains over-budget points)
  (c3) a <= 30-point completion, every point indeterminate -> unit-poll failure, all listed unresolved,
       certificate with the indeterminate count
W23 checks whose expectation changes under ruling A2 (the unit poll of an interior incumbent now meets the cap):
  (a) renamed (poll + completion); (a2) and (c) now end with the completion-cap refusal; the evaluation-budget
  check now binds on the x = 0 unit poll (14 new > a synthetic budget of 10).

Usage (repo root, canonical interpreter, attached):
  python -u p515_s47_phase_b_record_checks.py --scratch <dir outside the repo> --out <results.json> > <log> 2>&1
"""

import argparse
import json
import os
import sys
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s47_phase_b_record as B  # noqa: E402  (installs the parent guard; imports the stdlib-only harness)

H = B.H
Q0 = 653_859_461.2279255  # a synthetic Q(0); tests with the real Q(0) read it from the pinned A0 record
RESULTS = {}


def check(name, cond, evidence=None):
    RESULTS[name] = {'pass': bool(cond), 'evidence': evidence}
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f' :: {json.dumps(evidence, default=str)[:600]}'
                                                      if evidence is not None else ''), flush=True)


def real_lattice():
    w2 = B._load(B.INVESTMENT_COST_RESULTS['path'])
    return B.Lattice(tuple(sorted(B.unit_costs_from_w2(w2))), B.unit_costs_from_w2(w2)), w2


def x0_cache_entry(lattice, key_of, q0=Q0, bar=9_629.98):
    z = lattice.x0()
    return key_of(z), {'label': 'x0', 'status': 'certified', 'Q': q0, 'bar': bar,
                       'canonical': B.canonical_of(lattice, z), 'source': 'synthetic x0'}


def inc_of(lattice, key_of, cache, z):
    e = cache[key_of(z)]
    i_x = lattice.investment_cost(z)
    return {'eval_key': key_of(z), 'z': tuple(z), 'label': lattice.label(z), 'I': i_x, 'Q': e['Q'],
            'F': i_x + e['Q'], 'bar': e['bar'], 'source': e['source']}


class FakeOracle:
    """evaluate(list[z]) -> records; F(x) = I(x) + Q(x) with Q from `q_of(z)`; logs every call."""

    def __init__(self, lattice, key_of, q_of, bar=8_000.0, status_of=None, cache_before=None):
        self.lattice, self.key_of, self.q_of, self.bar = lattice, key_of, q_of, bar
        self.status_of = status_of or (lambda z: 'certified')
        self.calls, self.cache_before = [], set(cache_before or ())

    def __call__(self, batch):
        self.calls.append([tuple(z) for z in batch])
        out = []
        for z in batch:
            key = self.key_of(z)
            if key in self.cache_before:
                raise AssertionError(f'cache hit re-evaluated: {self.lattice.label(z)}')
            if self.lattice.reasons(z):
                raise AssertionError(f'infeasible point evaluated: {self.lattice.label(z)} {self.lattice.reasons(z)}')
            status = self.status_of(z)
            out.append({'label': self.lattice.label(z), 'status': status, 'eval_key': key,
                        'Q': self.q_of(z) if status == 'certified' else None,
                        'bar': self.bar if status == 'certified' else None,
                        'canonical': B.canonical_of(self.lattice, z),
                        'barrier_cause': None if status == 'certified' else 'synthetic non-certification',
                        'source': {'kind': 'fake'}})
        return out

    def evaluated(self):
        return [z for batch in self.calls for z in batch]


def find_incumbent(lattice, predicate, polls=((0, 4), (1, 2), (2, 1))):
    """First domain point (domain order) whose polls (k, Delta) satisfy `predicate(list of per-poll raw points)`."""
    for z in lattice.domain():
        if not lattice.has_storage(z):
            continue
        per_poll = []
        for k, d in polls:
            _t, _u, dirs = B.poll_directions(k, d)
            per_poll.append([tuple(a + b for a, b in zip(z, dd)) for dd in dirs])
        if predicate(z, per_poll):
            return z
    return None


class injected_first_poll:
    """TEST-ONLY: replace the poll directions of poll k = 0 by `dirs` (the literal OrthoMADS directions have at most
    one feasible point at Delta = 4 on this domain, and at most 5 at any poll size, so the batching and the
    multi-candidate decision logic cannot be reached through them). Every other poll keeps the real directions;
    the real function is restored on exit."""

    def __init__(self, dirs):
        self.dirs, self.real = [tuple(d) for d in dirs], B.poll_directions

    def __enter__(self):
        real = self.real

        def patched(k, delta, *a, **kw):
            if k == 0:
                return B.HALTON_T0, None, list(self.dirs)
            return real(k, delta, *a, **kw)
        B.poll_directions = patched
        return self

    def __exit__(self, *exc):
        B.poll_directions = self.real
        return False


# incumbent y2035 n5 0.5/1.5 + n7 0.5/1.5 (I ~ 634k) and 8 injected directions, >= 6 of them feasible
INJ_Z = (2, 3, 2, 3, 0, 0, 2)
INJ_DIRS = [(0, 1, 0, 0, 0, 0, 0), (0, -1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0), (0, 0, 0, -1, 0, 0, 0),
            (0, 0, 0, 0, 1, 1, 0), (0, 0, 0, 0, 1, 2, 0), (0, 0, 0, 0, 0, 0, -1), (0, 0, 0, 0, 1, 0, 0)]


def feasible_new(lattice, z_inc, pts):
    return [p for p in pts if not lattice.reasons(p) and lattice.canonical_z(p) != tuple(z_inc)]


# ======================================================================================================================
def test_directions():
    ok, bad = True, []
    for k in range(12):
        for d in (1, 2, 4, 8):
            t, _u, dirs = B.poll_directions(k, d)
            if len(dirs) != B.N_VARS + 1 or t != B.HALTON_T0 + k:
                ok = False
            for dd in dirs:
                if max(abs(a) for a in dd) != d:
                    ok, bad = False, bad + [(k, d, dd)]
            if B.poll_directions(k, d) != (t, _u, dirs):
                ok = False
    check('directions: n+1 = 8 per poll, ||d||_inf = Delta, deterministic, t = t0 + k', ok, {'violations': bad[:5]})
    u, cols = B.householder_columns(B.HALTON_T0)
    ortho = max(abs(sum(a * b for a, b in zip(cols[i], cols[j])) - (1.0 if i == j else 0.0))
                for i in range(B.N_VARS) for j in range(B.N_VARS))
    check('directions: H = I - 2vv^T is orthogonal (max |H^T H - I|)', ortho < 1e-12, {'max_dev': ortho, 'u_t0': u})


def test_lattice_identity(lattice, w2):
    xc = B.w2_cross_check(lattice, w2)
    check('I(x) closed form == W2 production values for all 105 W2 candidates (<= 1e-6 EUR)', xc['ok'], xc)
    domain = lattice.domain()
    key_of = B.make_key_of(lattice)
    by_node_ok = all(not lattice.reasons(z) for z in domain) and len(set(domain)) == len(domain)
    check('domain: every point feasible, no duplicates, x = 0 first', by_node_ok and domain[0] == lattice.x0(),
          {'n_points': len(domain)})
    s3 = B._load(os.path.join(B._P47, 'campaign_s47_a1a_baseline', 'campaign_spec_s47_a1a_baseline_add4ebd8.json'))
    s2 = B._load(os.path.join(B._P47, 'campaign_s47_recert', 'campaign_spec_s47_recert_902f93aa.json'))
    mism, in_domain = [], 0
    for e in s3['candidates'] + s2['candidates']:
        z = lattice.z_of_canonical(e['canonical'])
        if z is None:
            continue
        if B.eval_key_of(B.canonical_of(lattice, z)) != e['eval_key']:
            mism.append(e['label'])
        if not lattice.reasons(z):
            in_domain += 1
    check('eval keys here == the committed S2 / frozen S3 eval keys for the same candidate (same declaration)',
          not mism, {'mismatches': mism, 'n_S2_S3_entries_in_budget_domain': in_domain})
    x0_undeclared = H.evaluation_key(H.candidate_key(B.canonical_of(lattice, lattice.x0())), {},
                                     case_file_aa=B.CASE_FILE_AA)
    check('x = 0 baseline key differs from the undeclared (C3-era) key', key_of(lattice.x0()) != x0_undeclared)
    return key_of, domain, s3, s2


def test_a(lattice, key_of, sigma_q):
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache = {x0_key: x0e}
    worse = []
    for z in [(1, 2, 0, 0, 0, 0, 0), (0, 0, 1, 2, 0, 0, 0), (0, 0, 0, 0, 1, 2, 0), (2, 2, 0, 0, 0, 0, 0)]:
        k = key_of(z)
        cache[k] = {'label': lattice.label(z), 'status': 'certified', 'Q': Q0 - 0.8 * lattice.investment_cost(z),
                    'bar': 8_000.0, 'canonical': B.canonical_of(lattice, z), 'source': 'synthetic'}
        worse.append(lattice.label(z))
    inc, rec = B.initial_incumbent(lattice, cache, x0_key, sigma_q)
    check('(a) initial incumbent: every storage point F > F(0) -> x = 0',
          rec['is_x0'] and rec['every_storage_point_F_gt_F0'], {'incumbent': rec['incumbent']['label'],
                                                                'ranked': rec['eligible_ranked'][:3]})
    fake = FakeOracle(lattice, key_of, lambda z: Q0 - 0.8 * lattice.investment_cost(z), cache_before=cache)
    out = B.run_mads(lattice, dict(cache), key_of, inc, fake, sigma_q, log=lambda m: None)
    deltas = [p['poll_size_delta'] for p in out['history']]
    check('(a) from x = 0 (STEP4 5 poll + unit-poll completion, ruling A2): terminates at x = 0 by the unit-poll '
          'failure, certificate holds',
          out['termination']['reason'] == 'mesh_local_optimum_unit_poll_failed' and out['incumbent']['label'] == 'x0'
          and deltas == [4, 2, 1] and out['termination_certificate']['holds'],
          {'deltas': deltas, 'n_new_evaluations': out['n_new_evaluations'],
           'feasible_direction_points_per_poll': [sum(c['feasible'] for c in p['candidates']
                                                      if c['poll_part'] == 'direction') for p in out['history']],
           'feasible_points_polled_per_poll': [sum(c['feasible'] for c in p['candidates']) for p in out['history']],
           'rejection_reasons_poll0': [c['infeasibility_reasons'] for c in out['history'][0]['candidates']],
           'lattice_neighbourhood_of_x0_polled': [n for n in out['lattice_neighbourhood_of_incumbent']
                                                  if n['polled_in_final_poll']],
           'lattice_neighbourhood_of_x0_size': len(out['lattice_neighbourhood_of_incumbent'])})
    # (a2) from an interior storage incumbent whose polls do hit feasible points, all worse -> stays, Delta 4,2,1
    z_inc = find_incumbent(lattice, lambda z, pp: all(feasible_new(lattice, z, p) for p in pp))
    cache2 = dict(cache)
    cache2[key_of(z_inc)] = {'label': lattice.label(z_inc), 'status': 'certified', 'Q': Q0 - 5e6, 'bar': 8_000.0,
                             'canonical': B.canonical_of(lattice, z_inc), 'source': 'synthetic'}
    inc2 = inc_of(lattice, key_of, cache2, z_inc)
    fake2 = FakeOracle(lattice, key_of, lambda z: inc2['F'] - lattice.investment_cost(z) + 1e6, cache_before=cache2)
    out2 = B.run_mads(lattice, dict(cache2), key_of, inc2, fake2, sigma_q, log=lambda m: None)
    check('(a2) interior incumbent, every polled neighbour worse -> stays; Delta 4 -> 2 -> 1; the unit poll is '
          'refused by the completion cap (ruling A2: > 30 feasible neighbours) -> STOP FOR REVIEW',
          out2['incumbent']['label'] == inc2['label'] and [p['poll_size_delta'] for p in out2['history']] == [4, 2, 1]
          and [p['decision'] for p in out2['history']] == ['failure', 'failure', 'stopped_for_review_completion_cap']
          and out2['termination']['reason'] == 'STOP_FOR_REVIEW_completion_cap'
          and out2['history'][-1]['completion']['n_feasible'] > B.COMPLETION_CAP
          and out2['history'][-1]['batches'] == []
          and all(c['outcome'] in ('no_improvement', 'barrier_infeasible_not_evaluated', 'dropped',
                                   'not_evaluated_completion_cap', None)
                  for p in out2['history'] for c in p['candidates']),
          {'incumbent': inc2['label'], 'n_new_evaluations': out2['n_new_evaluations'],
           'evaluated': [lattice.label(z) for z in fake2.evaluated()], 'termination': out2['termination']})
    return z_inc


def test_b(lattice, key_of, sigma_q, z_inc):
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache = {x0_key: x0e}
    cache[key_of(z_inc)] = {'label': lattice.label(z_inc), 'status': 'certified', 'Q': Q0 - 5e6, 'bar': 8_000.0,
                            'canonical': B.canonical_of(lattice, z_inc), 'source': 'synthetic'}
    inc = inc_of(lattice, key_of, cache, z_inc)
    # first poll: one feasible point is better by 5 x the resolution; every other point is worse
    first = feasible_new(lattice, z_inc, [tuple(a + b for a, b in zip(z_inc, d)) for d in B.poll_directions(0, 4)[2]])
    target = lattice.canonical_z(first[0])
    margin = 5 * max(8_000.0 + inc['bar'], sigma_q)

    def q_of(z):
        if lattice.canonical_z(z) == target:
            return inc['F'] - margin - lattice.investment_cost(z)
        return inc['F'] + 1e6 - lattice.investment_cost(z)
    fake = FakeOracle(lattice, key_of, q_of, cache_before=cache)
    out = B.run_mads(lattice, dict(cache), key_of, inc, fake, sigma_q, log=lambda m: None)
    p0 = out['history'][0]
    hit = [c for c in p0['candidates'] if c['label'] == lattice.label(target)]
    check('(b) a neighbour better by more than the resolution -> accepted; incumbent moves; Delta doubles',
          p0['decision'] == 'success' and p0['next_incumbent'] == lattice.label(target) and p0['next_poll_size'] == 8
          and hit and hit[0]['outcome'] == 'improvement' and hit[0]['F_inc_minus_F_eur'] > hit[0]['resolution_eur']
          and out['incumbent']['label'] == lattice.label(target),
          {'from': inc['label'], 'to': lattice.label(target), 'F_inc_minus_F': hit[0]['F_inc_minus_F_eur'] if hit else None,
           'resolution': hit[0]['resolution_eur'] if hit else None, 'final': out['incumbent']['label'],
           'termination': out['termination'], 'deltas': [p['poll_size_delta'] for p in out['history']]})


def test_c(lattice, key_of, sigma_q, z_inc):
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache = {x0_key: x0e}
    cache[key_of(z_inc)] = {'label': lattice.label(z_inc), 'status': 'certified', 'Q': Q0 - 5e6, 'bar': 8_000.0,
                            'canonical': B.canonical_of(lattice, z_inc), 'source': 'synthetic'}
    inc = inc_of(lattice, key_of, cache, z_inc)
    res = max(8_000.0 + inc['bar'], sigma_q)
    fake = FakeOracle(lattice, key_of, lambda z: inc['F'] - 0.5 * res - lattice.investment_cost(z), cache_before=cache)
    out = B.run_mads(lattice, dict(cache), key_of, inc, fake, sigma_q, log=lambda m: None)
    outcomes = sorted({c['outcome'] for p in out['history'][:-1] for c in p['candidates']
                       if c['disposition'] == 'new_evaluation'})
    refused = sorted({c['outcome'] for c in out['history'][-1]['candidates'] if c['disposition'] == 'new_evaluation'})
    check('(c) better but within the resolution -> INDETERMINATE, not accepted; Delta halves; the unit poll of this '
          'interior incumbent is refused by the completion cap (unresolved listing: see (c3))',
          out['incumbent']['label'] == inc['label'] and outcomes == ['indeterminate']
          and refused in ([], ['not_evaluated_completion_cap'])
          and [p['decision'] for p in out['history']] == ['failure', 'failure', 'stopped_for_review_completion_cap'],
          {'incumbent': inc['label'], 'improvement_by': 0.5 * res, 'resolution': res, 'outcomes': outcomes,
           'refused_unit_poll_outcomes': refused, 'decisions': [p['decision'] for p in out['history']],
           'termination': out['termination']})
    # boundary: exactly equal to the resolution is NOT an improvement (strict >)
    check('(c2) classify: diff == resolution -> indeterminate; diff > resolution -> improvement; <= 0 -> none',
          B.classify(100.0, 100.0 - res, res) == 'indeterminate' and B.classify(100.0, 100.0 - res - 1e-6, res) ==
          'improvement' and B.classify(100.0, 100.0, res) == 'no_improvement' and B.classify(1.0, None, res) == 'barrier'
          and B.resolution(None, 1.0, sigma_q) == float('inf') and B.resolution(1.0, 2.0, sigma_q) == sigma_q
          and B.resolution(2e4, 2e4, sigma_q) == 4e4)


def test_d(lattice, key_of, sigma_q):
    z_inc = find_incumbent(lattice, lambda z, pp: any(any(r.startswith('budget') for r in lattice.reasons(p))
                                                      for poll in pp for p in poll)
                           and any(feasible_new(lattice, z, poll) for poll in pp))
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache = {x0_key: x0e, key_of(z_inc): {'label': lattice.label(z_inc), 'status': 'certified', 'Q': Q0 - 5e6,
                                         'bar': 8_000.0, 'canonical': B.canonical_of(lattice, z_inc),
                                         'source': 'synthetic'}}
    inc = inc_of(lattice, key_of, cache, z_inc)
    fake = FakeOracle(lattice, key_of, lambda z: inc['F'] + 1e6 - lattice.investment_cost(z), cache_before=cache)
    out = B.run_mads(lattice, dict(cache), key_of, inc, fake, sigma_q, log=lambda m: None)
    budget_rej = [c for p in out['history'] for c in p['candidates']
                  if any(r.startswith('budget') for r in c['infeasibility_reasons'])]
    over = [lattice.label(z) for z in fake.evaluated() if lattice.investment_cost(z) > B.BUDGET_EUR]
    check('(d) over-budget poll points are rejected before evaluation and never evaluated',
          budget_rej and not over and all(c['disposition'] == 'rejected_infeasible' and c['F_eur'] is None
                                          for c in budget_rej),
          {'incumbent': inc['label'], 'I_inc': inc['I'], 'n_budget_rejections': len(budget_rej),
           'example': budget_rej[0]['infeasibility_reasons'] if budget_rej else None,
           'evaluated': [lattice.label(z) for z in fake.evaluated()], 'evaluated_over_budget': over})


def test_e(lattice, key_of, sigma_q, z_inc):
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache = {x0_key: x0e, key_of(z_inc): {'label': lattice.label(z_inc), 'status': 'certified', 'Q': Q0 - 5e6,
                                         'bar': 8_000.0, 'canonical': B.canonical_of(lattice, z_inc),
                                         'source': 'synthetic'}}
    inc = inc_of(lattice, key_of, cache, z_inc)
    first = feasible_new(lattice, z_inc, [tuple(a + b for a, b in zip(z_inc, d)) for d in B.poll_directions(0, 4)[2]])
    target = lattice.canonical_z(first[0])
    fake = FakeOracle(lattice, key_of, lambda z: inc['F'] - 1e6 - lattice.investment_cost(z),  # would be far better
                      status_of=lambda z: 'not_certified' if lattice.canonical_z(z) == target else 'certified',
                      cache_before=cache)

    fake_all_worse = FakeOracle(lattice, key_of, lambda z: inc['F'] + 1e6 - lattice.investment_cost(z),
                                status_of=lambda z: 'not_certified' if lattice.canonical_z(z) == target else 'certified',
                                cache_before=cache)
    out = B.run_mads(lattice, dict(cache), key_of, inc, fake_all_worse, sigma_q, log=lambda m: None)
    c = [c for c in out['history'][0]['candidates'] if c['label'] == lattice.label(target)][0]
    check('(e) a non-certified evaluation is an extreme-barrier point: F = +inf (None), cause kept, counted, '
          'never the incumbent',
          c['outcome'] == 'barrier' and c['F_eur'] is None and c['status'] == 'not_certified'
          and c['barrier_cause'] == 'synthetic non-certification' and out['incumbent']['label'] == inc['label']
          and out['n_barrier_new_evaluations'] == 1 and target in [lattice.canonical_z(z) for z in fake_all_worse.evaluated()],
          {'target': lattice.label(target), 'candidate': {k: c[k] for k in ('outcome', 'status', 'F_eur', 'barrier_cause')},
           'n_new': out['n_new_evaluations'], 'termination': out['termination']})
    # (e2)/(e3) on the injected first poll (>= 6 feasible points; see injected_first_poll)
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache_i = {x0_key: x0e, key_of(INJ_Z): {'label': lattice.label(INJ_Z), 'status': 'certified', 'Q': Q0 - 5e6,
                                           'bar': 8_000.0, 'canonical': B.canonical_of(lattice, INJ_Z),
                                           'source': 'synthetic'}}
    inc_i = inc_of(lattice, key_of, cache_i, INJ_Z)
    feas = feasible_new(lattice, INJ_Z, [tuple(a + b for a, b in zip(INJ_Z, d)) for d in INJ_DIRS])
    best_t, second = lattice.canonical_z(feas[0]), lattice.canonical_z(feas[1])

    def q_e2(z):
        z = lattice.canonical_z(z)
        gain = 1e6 if z == best_t else (5e5 if z == second else -1e6)
        return inc_i['F'] - gain - lattice.investment_cost(z)
    fake_e2 = FakeOracle(lattice, key_of, q_e2, status_of=lambda z: 'not_certified'
                         if lattice.canonical_z(z) == best_t else 'certified', cache_before=cache_i)
    with injected_first_poll(INJ_DIRS):
        out_b = B.run_mads(lattice, dict(cache_i), key_of, inc_i, fake_e2, sigma_q, log=lambda m: None)
    check('(e2) the barrier point would be the best, yet the certified determinate improver is chosen',
          out_b['history'][0]['decision'] == 'success' and out_b['history'][0]['next_incumbent'] == lattice.label(second),
          {'barrier_point': lattice.label(best_t), 'next': out_b['history'][0]['next_incumbent']})
    fake_bad = FakeOracle(lattice, key_of, lambda z: 0.0, status_of=lambda z: 'not_certified', cache_before=cache_i)
    with injected_first_poll(INJ_DIRS):
        out_s = B.run_mads(lattice, dict(cache_i), key_of, inc_i, fake_bad, sigma_q, log=lambda m: None)
    p0 = out_s['history'][0]
    check('(e3) barrier stop rule: >= 2 new barrier evaluations in one poll -> STOP_FOR_REVIEW after that batch; '
          'the next batch is not launched and its points are recorded not_evaluated_stop_rule',
          out_s['termination']['reason'] == 'STOP_FOR_REVIEW_barrier_rule'
          and out_s['termination']['barrier_new_this_poll'] >= 2 and p0['decision'] == 'stopped_for_review'
          and len(p0['batches']) == 1 and any(c['outcome'] == 'not_evaluated_stop_rule' for c in p0['candidates']),
          {'termination': out_s['termination'], 'batches': p0['batches'],
           'n_new_needed': p0['n_new_evaluations'], 'n_evaluated': out_s['n_new_evaluations']})


# ======================================================================================================================
#  W24: the unit-poll completion (Planner ruling A2)
# ======================================================================================================================
def x0_completion_expected(lattice):
    """The Planner's statement, built independently of Lattice.neighbourhood: 0.25 MVA / 0.5 MWh at every non-empty
    subset of the nodes x {2025, 2030}."""
    out = set()
    for mask in range(1, 2 ** len(B.ACTIVE_NODES)):
        for yi in (lattice.years.index(2025), lattice.years.index(2030)):
            z = [0] * B.N_VARS
            for i in range(len(B.ACTIVE_NODES)):
                if mask >> i & 1:
                    z[2 * i], z[2 * i + 1] = 1, 1
            z[-1] = yi
            out.add(tuple(z))
    return out


def test_f(lattice, key_of, sigma_q):
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache = {x0_key: x0e}
    inc = inc_of(lattice, key_of, cache, lattice.x0())
    expected = x0_completion_expected(lattice)
    fake = FakeOracle(lattice, key_of, lambda z: Q0 + 1e5 - 0.5 * lattice.investment_cost(z), cache_before=cache)
    out = B.run_mads(lattice, dict(cache), key_of, inc, fake, sigma_q, log=lambda m: None)
    unit = out['history'][-1]
    comp = [c for c in unit['candidates'] if c['poll_part'] == 'completion']
    evaluated = [lattice.canonical_z(z) for z in fake.evaluated()]
    cert = out['termination_certificate']
    check('(f) x = 0: the completion is exactly the 14 points 0.25/0.5 at every non-empty node subset x {2025, 2030}',
          len(expected) == B.X0_COMPLETION_SIZE == 14 and unit['completion']['n_feasible'] == 14
          and {tuple(c['z']) for c in comp} == expected and len(comp) == 14,
          {'completion': unit['completion']['points'], 'rejected_raw_offsets_by_class':
           unit['completion']['rejected_raw_offsets_by_class'], 'budget_rejected': unit['completion']['budget_rejected']})
    check('(f) x = 0: every rounded direction infeasible at every poll; all 14 completion points evaluated, none '
          'cached, batches 5 + 5 + 4, nothing else evaluated',
          all(not c['feasible'] for p in out['history'] for c in p['candidates'] if c['poll_part'] == 'direction')
          and all(c['disposition'] == 'new_evaluation' for c in comp) and unit['n_cache_hits'] == 0
          and sorted(evaluated) == sorted(expected) and [len(b) for b in fake.calls] == [5, 5, 4]
          and [len(b) for b in unit['batches']] == [5, 5, 4] and out['n_new_evaluations'] == 14,
          {'batches': unit['batches'], 'n_new_evaluations': out['n_new_evaluations'],
           'polls': [(p['poll_index'], p['poll_size_delta'], p['n_new_evaluations'], p['decision'])
                     for p in out['history']]})
    check('(f) x = 0, all 14 worse -> terminates at x = 0 by the unit-poll failure with the certificate stated',
          out['termination']['reason'] == 'mesh_local_optimum_unit_poll_failed' and out['incumbent']['label'] == 'x0'
          and cert is not None and cert['holds'] and cert['statement'] == B.TERMINATION_CERTIFICATE
          and cert['n_feasible_neighbours'] == 14 and cert['all_feasible_neighbours_polled']
          and cert['n_no_improvement'] == 14 and all(c['outcome'] == 'no_improvement' for c in comp),
          {'termination': out['termination'], 'certificate': cert})


def test_g(lattice, key_of, sigma_q):
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache = {x0_key: x0e}
    inc = inc_of(lattice, key_of, cache, lattice.x0())
    target = (0, 0, 1, 1, 0, 0, 0)  # y2025__n7_p0.25_e0.5, a completion point of x = 0
    fake_bar = 8_000.0
    res = max(fake_bar + inc['bar'], sigma_q)
    margin = 5 * res

    def q_of(z):
        if lattice.canonical_z(z) == target:
            return inc['F'] - margin - lattice.investment_cost(z)
        return inc['F'] + 1e6 - lattice.investment_cost(z)
    fake = FakeOracle(lattice, key_of, q_of, bar=fake_bar, cache_before=cache)
    out = B.run_mads(lattice, dict(cache), key_of, inc, fake, sigma_q, log=lambda m: None)
    h = out['history']
    hit = [c for c in h[2]['candidates'] if c['label'] == lattice.label(target)] if len(h) > 2 else []
    check('(g) one completion point better by more than the resolution -> accepted; Delta doubles; the run continues',
          len(h) > 3 and h[2]['unit_poll'] and h[2]['decision'] == 'success'
          and h[2]['next_incumbent'] == lattice.label(target) and h[2]['next_poll_size'] == 2
          and hit and hit[0]['poll_part'] == 'completion' and hit[0]['outcome'] == 'improvement'
          and hit[0]['F_inc_minus_F_eur'] > hit[0]['resolution_eur']
          and h[3]['poll_size_delta'] == 2 and h[3]['incumbent']['label'] == lattice.label(target)
          and out['incumbent']['label'] == lattice.label(target),
          {'F_inc_minus_F': hit[0]['F_inc_minus_F_eur'] if hit else None,
           'resolution': hit[0]['resolution_eur'] if hit else None,
           'polls': [(p['poll_index'], p['poll_size_delta'], p['incumbent']['label'], p['n_new_evaluations'],
                      (p['completion'] or {}).get('n_feasible'), p['decision']) for p in h],
           'termination': out['termination'], 'n_new_evaluations': out['n_new_evaluations']})
    return {'polls': [(p['poll_index'], p['poll_size_delta'], p['incumbent']['label'], p['n_new_evaluations'],
                       (p['completion'] or {}).get('n_feasible'), p['decision']) for p in h],
            'termination': out['termination'], 'n_new_evaluations': out['n_new_evaluations']}


def test_h(lattice, key_of, sigma_q, z_inc):
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache = {x0_key: x0e, key_of(z_inc): {'label': lattice.label(z_inc), 'status': 'certified', 'Q': Q0 - 5e6,
                                         'bar': 8_000.0, 'canonical': B.canonical_of(lattice, z_inc),
                                         'source': 'synthetic'}}
    inc = inc_of(lattice, key_of, cache, z_inc)
    fake = FakeOracle(lattice, key_of, lambda z: inc['F'] - 1e6 - lattice.investment_cost(z), cache_before=cache)
    out = B.run_mads(lattice, dict(cache), key_of, inc, fake, sigma_q, delta0=B.DELTA_MIN, log=lambda m: None)
    p0 = out['history'][0]
    n_nb = len(lattice.neighbourhood(z_inc))
    check('(h) completion > 30 feasible points at a synthetic interior incumbent -> refused: STOP FOR REVIEW, nothing '
          'of that poll evaluated (not even the directions), the completion recorded in full (not truncated)',
          n_nb > B.COMPLETION_CAP and out['termination']['reason'] == 'STOP_FOR_REVIEW_completion_cap'
          and out['termination']['reason'].startswith('STOP_FOR_REVIEW') and p0['decision'] ==
          'stopped_for_review_completion_cap' and not fake.calls and out['n_new_evaluations'] == 0
          and p0['completion']['n_feasible'] == n_nb and len(p0['completion']['points']) == n_nb
          and not any(c['poll_part'] == 'completion' for c in p0['candidates'])
          and out['incumbent']['label'] == inc['label'] and out['termination_certificate'] is None
          and len(out['history']) == 1,
          {'incumbent': inc['label'], 'completion_n_feasible': p0['completion']['n_feasible'],
           'termination': out['termination']})


def test_i(sigma_q, w2):
    costs = B.unit_costs_from_w2(w2)
    small = B.Lattice(tuple(sorted(costs)), costs, budget=400_000.0)  # synthetic B (test fixture only)
    key_small = B.make_key_of(small)
    x0_key, x0e = x0_cache_entry(small, key_small)
    cache = {x0_key: x0e}
    inc = inc_of(small, key_small, cache, small.x0())
    fake = FakeOracle(small, key_small, lambda z: Q0 + 1e5, cache_before=cache)
    out = B.run_mads(small, dict(cache), key_small, inc, fake, sigma_q, delta0=B.DELTA_MIN, log=lambda m: None)
    p0 = out['history'][0]
    over = [z for z in x0_completion_expected(small) if small.investment_cost(z) > small.budget]
    evaluated = [small.canonical_z(z) for z in fake.evaluated()]
    check('(i) budget: an over-budget completion point (I(x) > B) is never evaluated -- synthetic B = 400,000 EUR; '
          'x = 0 completion 14 -> 12 feasible, the 2 three-node points rejected by the budget',
          len(over) == 2 and p0['completion']['n_feasible'] == 12
          and sorted(e['label'] for e in p0['completion']['budget_rejected']) == sorted(small.label(z) for z in over)
          and not any(z in evaluated for z in over) and not any(small.investment_cost(z) > small.budget
                                                                  for z in evaluated)
          and len(evaluated) == 12 and out['termination_certificate']['holds']
          and out['termination_certificate']['n_feasible_neighbours'] == 12,
          {'budget_rejected': p0['completion']['budget_rejected'], 'n_evaluated': len(evaluated),
           'batches': [len(b) for b in fake.calls], 'termination': out['termination']})


def test_c3(lattice, key_of, sigma_q):
    z_inc = (0, 0, 1, 1, 0, 0, 0)  # y2025__n7_p0.25_e0.5: completion of exactly 30 (= the cap, admitted)
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache = {x0_key: x0e, key_of(z_inc): {'label': lattice.label(z_inc), 'status': 'certified', 'Q': Q0 - 5e6,
                                         'bar': 8_000.0, 'canonical': B.canonical_of(lattice, z_inc),
                                         'source': 'synthetic'}}
    inc = inc_of(lattice, key_of, cache, z_inc)
    res = max(8_000.0 + inc['bar'], sigma_q)
    fake = FakeOracle(lattice, key_of, lambda z: inc['F'] - 0.5 * res - lattice.investment_cost(z), cache_before=cache)
    out = B.run_mads(lattice, dict(cache), key_of, inc, fake, sigma_q, delta0=B.DELTA_MIN, max_new_evaluations=40,
                     log=lambda m: None)
    p0 = out['history'][0]
    cert = out['termination_certificate']
    check('(c3) completion of exactly 30 (= cap, admitted); every new completion point within the resolution -> '
          'INDETERMINATE, not accepted; unit-poll failure; all listed unresolved; certificate counts them '
          '(synthetic evaluation budget 40 > the 29 new points)',
          p0['completion']['n_feasible'] == 30 and p0['decision'] == 'failure_at_unit_poll_size'
          and out['incumbent']['label'] == inc['label'] and len(out['final_poll_unresolved_indeterminate']) == 29
          and cert['holds'] and cert['n_indeterminate_unresolved'] == 29 and cert['n_no_improvement'] == 1
          and p0['n_cache_hits'] == 1 and all(len(b) <= 5 for b in fake.calls),
          {'certificate': cert, 'batches': [len(b) for b in fake.calls], 'n_new': out['n_new_evaluations']})


def test_cache_hits_batches_budget(lattice, key_of, sigma_q, z_inc):
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    cache = {x0_key: x0e, key_of(z_inc): {'label': lattice.label(z_inc), 'status': 'certified', 'Q': Q0 - 5e6,
                                         'bar': 8_000.0, 'canonical': B.canonical_of(lattice, z_inc),
                                         'source': 'synthetic'}}
    inc = inc_of(lattice, key_of, cache, z_inc)
    first = feasible_new(lattice, z_inc, [tuple(a + b for a, b in zip(z_inc, d)) for d in B.poll_directions(0, 4)[2]])
    pre = lattice.canonical_z(first[0])
    cache[key_of(pre)] = {'label': lattice.label(pre), 'status': 'certified',
                          'Q': inc['F'] + 1e6 - lattice.investment_cost(pre), 'bar': 8_000.0,
                          'canonical': B.canonical_of(lattice, pre), 'source': 'synthetic pre-cached'}
    fake = FakeOracle(lattice, key_of, lambda z: inc['F'] + 1e6 - lattice.investment_cost(z), cache_before=cache)
    out = B.run_mads(lattice, dict(cache), key_of, inc, fake, sigma_q, log=lambda m: None)
    hit = [c for c in out['history'][0]['candidates'] if c['label'] == lattice.label(pre)]
    check('cache hit is never re-evaluated and is recorded as cache_hit',
          hit and hit[0]['disposition'] == 'cache_hit' and pre not in fake.evaluated(),
          {'pre_cached': lattice.label(pre)})
    check('every evaluate() call carries <= 5 points (literal directions)', all(len(b) <= 5 for b in fake.calls),
          {'batch_sizes': [len(b) for b in fake.calls]})
    x0_key2, x0e2 = x0_cache_entry(lattice, key_of)
    cache_i = {x0_key2: x0e2, key_of(INJ_Z): {'label': lattice.label(INJ_Z), 'status': 'certified', 'Q': Q0 - 5e6,
                                             'bar': 8_000.0, 'canonical': B.canonical_of(lattice, INJ_Z),
                                             'source': 'synthetic'}}
    inc_i = inc_of(lattice, key_of, cache_i, INJ_Z)
    fake_i = FakeOracle(lattice, key_of, lambda z: inc_i['F'] + 1e6 - lattice.investment_cost(z), cache_before=cache_i)
    with injected_first_poll(INJ_DIRS):
        out_i = B.run_mads(lattice, dict(cache_i), key_of, inc_i, fake_i, sigma_q, log=lambda m: None)
    p0 = out_i['history'][0]
    check('full poll in batches of <= 5: a poll with > 5 new points is split (5 + rest); incumbent updated only '
          'after the whole poll', p0['n_new_evaluations'] > 5 and [len(b) for b in p0['batches']] ==
          [5, p0['n_new_evaluations'] - 5] and len(fake_i.calls[0]) == 5 and p0['decision'] == 'failure',
          {'n_new': p0['n_new_evaluations'], 'batches': p0['batches'],
           'rejected': [(c['direction'], c['infeasibility_reasons']) for c in p0['candidates'] if not c['feasible']]})
    # W24: from x = 0 the only evaluating poll is the unit poll (14 completion points); a synthetic budget of 10
    x0_key3, x0e3 = x0_cache_entry(lattice, key_of)
    cache3 = {x0_key3: x0e3}
    inc3 = inc_of(lattice, key_of, cache3, lattice.x0())
    fake2 = FakeOracle(lattice, key_of, lambda z: inc3['F'] + 1e6 - lattice.investment_cost(z), cache_before=cache3)
    out2 = B.run_mads(lattice, dict(cache3), key_of, inc3, fake2, sigma_q, max_new_evaluations=10, log=lambda m: None)
    check('evaluation budget: a poll that would exceed it is not launched (x = 0 unit poll with its 14-point '
          'completion vs a synthetic budget of 10); incumbent reported with the poll size',
          out2['termination']['reason'] == 'evaluation_budget_exhausted' and out2['n_new_evaluations'] == 0
          and not fake2.calls and out2['history'][-1]['decision'] == 'not_launched_evaluation_budget'
          and out2['history'][-1]['unit_poll'] and out2['history'][-1]['n_new_evaluations'] == 14,
          {'termination': out2['termination'], 'n_new': out2['n_new_evaluations']})


def test_initial_incumbent_ties(lattice, key_of, sigma_q):
    x0_key, x0e = x0_cache_entry(lattice, key_of)
    a, b = (1, 1, 0, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0, 0)  # same I (same unit, 2025)
    cache = {x0_key: x0e}
    for z in (a, b):
        cache[key_of(z)] = {'label': lattice.label(z), 'status': 'certified', 'Q': Q0 - 5e5 - lattice.investment_cost(z),
                            'bar': 8_000.0, 'canonical': B.canonical_of(lattice, z), 'source': 'synthetic'}
    z_over = (10, 10, 0, 0, 0, 0, 0)  # 2.5 MVA / 5 MWh 2025: over budget, very good F -> excluded
    cache[key_of(z_over)] = {'label': lattice.label(z_over), 'status': 'certified', 'Q': Q0 - 1e8, 'bar': 1.0,
                             'canonical': B.canonical_of(lattice, z_over), 'source': 'synthetic'}
    z_nc = (2, 2, 0, 0, 0, 0, 0)
    cache[key_of(z_nc)] = {'label': lattice.label(z_nc), 'status': 'not_certified', 'Q': None, 'bar': None,
                           'canonical': B.canonical_of(lattice, z_nc), 'source': 'synthetic'}
    inc, rec = B.initial_incumbent(lattice, cache, x0_key, sigma_q)
    check('initial incumbent: argmin F (tie -> label), over-budget and non-certified excluded, margin recorded',
          inc['label'] == min(lattice.label(a), lattice.label(b)) and not rec['is_x0']
          and any('budget' in e['reason'] for e in rec['excluded']) and any('barrier' in e['reason'] for e in rec['excluded'])
          and rec['margin_vs_x0']['determinate'] and rec['margin_vs_runner_up']['determinate'] is False,
          {'incumbent': inc['label'], 'margin_vs_x0': rec['margin_vs_x0'], 'runner_up': rec['margin_vs_runner_up'],
           'excluded': rec['excluded']})


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(obj, handle, indent=1)


def test_cache_loader(lattice, key_of, scratch):
    case_sha, ess_sha = 'c' * 64, B.ESS_PARAMS_FILE['sha256']
    cfg = {'ess_ageing_baseline': B.ESS_AGEING_BASELINE, 'case_file_anderson_acceleration': B.CASE_FILE_AA,
           'ess_params_file': {'sha256': ess_sha}, 'case_file_sha256': case_sha, 'overrides': {}, 'arm_label': 's39_D'}

    def camp(name, config, points):
        root = os.path.join(scratch, 'P515S47', f'campaign_{name}')
        spec = {'campaign_id': name, 'cap': 500, 'required_consecutive_cycles': 10, 'configuration': config,
                'candidates': [{'label': l, 'overrides': {}} for l in points]}
        spec_rel = os.path.relpath(os.path.join(root, 'spec.json'), scratch)
        _write(os.path.join(scratch, spec_rel), spec)
        res = {'campaign_spec_path': spec_rel, 'campaign_spec_sha256': H.sha256_file(os.path.join(scratch, spec_rel)),
               'points': points}
        _write(os.path.join(root, 'campaign_results.json'), res)
        return os.path.relpath(os.path.join(root, 'campaign_results.json'), scratch)

    def pt(z, q, status='certified', bar=8000.0, key=None):
        can = B.canonical_of(lattice, z)
        return {'status': status, 'eval_key': key or key_of(z), 'candidate_canonical': can,
                'certified_cost_gross_settlement_excluded': q, 'bar': {'value': bar}}
    u1, u2 = (0, 0, 1, 2, 0, 0, 0), (1, 2, 0, 0, 0, 0, 0)
    base_a = camp('base_a', cfg, {'n7_4h_e1': pt(u1, 1.0e8), 'n5_4h_e1': pt(u2, 1.1e8)})
    base_b = camp('base_b', cfg, {'n7_dup': pt(u1, 1.0e8), 'nl': {'status': 'not_launched_stop_rule'}})
    base_c = camp('base_c', cfg, {'n7_dup_diff': pt(u1, 1.0e8 + 1.0)})
    c3 = camp('c3_like', dict(cfg, ess_ageing_baseline=None), {'n7': pt(u1, 9e7)})
    wrong_key = camp('wrong_key', cfg, {'n7': pt(u1, 1e8, key='0' * 64)})
    ok_a, info_a, ent_a = B.load_cache_source(base_a, case_sha, ess_sha, repo=scratch, require_git=False)
    ok_b, info_b, ent_b = B.load_cache_source(base_b, case_sha, ess_sha, repo=scratch, require_git=False)
    ok_c, _i, ent_c = B.load_cache_source(base_c, case_sha, ess_sha, repo=scratch, require_git=False)
    ok_3, info_3, _e = B.load_cache_source(c3, case_sha, ess_sha, repo=scratch, require_git=False)
    check('cache loader: a baseline campaign is accepted, a C3-like (undeclared) spec is rejected, '
          'not_launched points skipped', ok_a and ok_b and not ok_3 and len(ent_a) == 2 and len(ent_b) == 1,
          {'c3_like_reason': info_3.get('reason'), 'c3_checks': info_3.get('spec_checks'),
           'skipped': info_b.get('skipped_not_evaluations')})
    try:
        B.load_cache_source(wrong_key, case_sha, ess_sha, repo=scratch, require_git=False)
        wk = False
    except AssertionError as error:
        wk = 'recomputed' in str(error)
    check('cache loader: a record whose eval key does not recompute under the declaration is refused', wk)
    cache, dups = B.merge_cache([ent_a, ent_b])
    check('duplicate eval key with bitwise-identical status/Q/bar merges (S2/S3 n7_4h_e1 case)',
          len(cache) == 2 and dups and dups[0]['status_Q_bar_bitwise_identical'], dups)
    try:
        B.merge_cache([ent_a, ent_c])
        dd = False
    except AssertionError as error:
        dd = 'determinism' in str(error)
    check('duplicate eval key with a different Q is refused (determinism failure -> STOP)', dd)
    # C3 exclusion over synthetic C3-era globs
    c3_file = os.path.join(scratch, 'C3', 'campaign_x', 'campaign_results.json')
    _write(c3_file, {'points': {'n7': {'eval_key': 'f' * 64}}})
    ex_ok = B.c3_exclusion({key_of(u1)}, {key_of(u1)}, repo=scratch, globs=('C3/**/campaign_results.json',))
    _write(os.path.join(scratch, 'C3b', 'campaign_y', 'campaign_results.json'), {'points': [{'eval_key': key_of(u1)}]})
    ex_bad = B.c3_exclusion({key_of(u1)}, set(), repo=scratch, globs=('C3b/**/campaign_results.json',))
    check('C3 exclusion: passes when no C3 key collides; fails on a collision', ex_ok['ok'] and not ex_bad['ok'],
          {'ok_case': ex_ok['n_files'], 'bad_case': ex_bad['key_collisions']})


def test_real_c3_exclusion_and_s2(lattice, key_of, domain):
    """Read-only on the repository: the committed S2 results as cache, the real C3-era scan."""
    case_sha = H.sha256_file(H.CASE_FILE)
    rel = os.path.join(B._P47, 'campaign_s47_recert', 'campaign_results.json')
    ok, info, ent = B.load_cache_source(rel, case_sha, B.ESS_PARAMS_FILE['sha256'])
    check('real: committed S2 s47_recert results accepted as cache (git-tracked, clean, same declaration, '
          'manifest record hashes match)', ok and len(ent) == 2 and not info['manifest']['mismatched'],
          {k: info.get(k) for k in ('sha256', 'n_entries', 'manifest', 'reason')})
    ex = B.c3_exclusion(set(ent), {key_of(z) for z in domain} - {key_of(lattice.x0())})
    check('real: C3-era results (P515S44/S45/S46) share no eval key with the cache / domain; none declares a baseline',
          ex['ok'], {'n_files': ex['n_files'], 'collisions': ex['key_collisions'],
                     'declaring': ex['specs_declaring_a_baseline']})
    return ent


def expected_poll_from_x0(lattice, key_of, s2_entries, s3_spec, sigma_q):
    x0_key, x0 = B.x0_entry(lattice, B._load(B.A0_RESULTS['path']))
    cache, _d = B.merge_cache([s2_entries, {x0_key: x0}])
    inc, rec = B.initial_incumbent(lattice, cache, x0_key, sigma_q)
    dry = B._dry_summary(lattice, cache, key_of, inc, sigma_q)
    s3_keys = {e['eval_key']: e['label'] for e in s3_spec['candidates']}
    s2_keys = {k: v['label'] for k, v in s2_entries.items()}
    nb = []
    for z in lattice.neighbourhood(lattice.x0()):
        nb.append({'label': lattice.label(z), 'I_x_eur': lattice.investment_cost(z), 'in_S2': s2_keys.get(key_of(z)),
                   'in_S3_spec': s3_keys.get(key_of(z))})
    planner = []
    for n_i, n in enumerate(B.ACTIVE_NODES):
        for ze in (1, 2):
            for yi in range(len(lattice.years)):
                z = [0] * B.N_VARS
                z[2 * n_i], z[2 * n_i + 1], z[-1] = 1, ze, yi
                z = tuple(z)
                planner.append({'label': lattice.label(z), 'I_x_eur': lattice.investment_cost(z),
                                'budget_feasible': not lattice.reasons(z), 'in_S2': s2_keys.get(key_of(z)),
                                'in_S3_spec': s3_keys.get(key_of(z))})
    fp = dry.get('first_evaluating_poll') or {}
    check('real: expected run from x = 0 (zero solves, S2 + pinned x0 as cache; ruling A2): polls Delta 4, 2 need no '
          'evaluation; the unit poll (k = 2) needs the 14 completion points, batches 5 + 5 + 4; none of them in S2 '
          'or in the frozen S3 spec',
          rec['is_x0'] and dry.get('complete_without_new_evaluations') is False and fp.get('poll_index') == 2
          and fp.get('Delta') == 1 and fp.get('n_new_evaluations') == 14 and fp.get('batch_sizes') == [5, 5, 4]
          and fp.get('completion_n_feasible') == 14 and all(n['in_S2'] is None and n['in_S3_spec'] is None for n in nb),
          {'initial_incumbent_with_S2_only': rec['incumbent']['label'], 'first_evaluating_poll': fp})
    sizes = {lattice.label(z): len(lattice.neighbourhood(z)) for z in lattice.neighbourhood(lattice.x0())}
    return {'initial_incumbent_record_S2_only': rec, 'dry_run': dry,
            'lattice_neighbourhood_of_x0_inf_norm_1': nb, 'planner_example_units_single_node': planner,
            'completion_size_at_each_x0_neighbour': sizes,
            'x0_neighbours_whose_completion_exceeds_the_cap': sorted(k for k, v in sizes.items()
                                                                    if v > B.COMPLETION_CAP)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    scratch = os.path.abspath(args.scratch)
    if scratch.startswith(REPO + os.sep):
        raise SystemExit('--scratch must be outside the repository')
    os.makedirs(scratch, exist_ok=True)
    os.chdir(REPO)
    extra = {}
    try:
        lattice, w2 = real_lattice()
        sq = B.sigma_q_from_tables(B._load(B.PHASE_A_TABLES['path']))
        sigma_q = sq['sigma_Q_eur']
        extra['sigma_Q'] = sq
        extra['degradation_clause'] = dict(B.degradation_clause(lattice, sigma_q), **B.min_year_step_cost(lattice))
        check('sigma_Q pinned = T3 residual_max_abs_eur; 5.2 degradation clause not triggered',
              abs(sigma_q - 18449.663947025518) < 1e-9 and not extra['degradation_clause']['triggered'],
              extra['degradation_clause'])
        test_directions()
        key_of, domain, s3_spec, _s2_spec = test_lattice_identity(lattice, w2)
        extra['domain'] = {'n_points': len(domain), 'per_year': {y: sum(1 for z in domain if lattice.has_storage(z)
                                                                        and lattice.years[z[-1]] == y)
                                                                 for y in lattice.years}}
        z_inc = test_a(lattice, key_of, sigma_q)
        extra['interior_test_incumbent'] = {'z': z_inc, 'label': lattice.label(z_inc),
                                            'rule': 'first domain point whose polls (k, Delta) = (0, 4), (1, 2), '
                                                    '(2, 1) each contain a feasible new point'}
        test_b(lattice, key_of, sigma_q, z_inc)
        test_c(lattice, key_of, sigma_q, z_inc)
        test_d(lattice, key_of, sigma_q)
        test_e(lattice, key_of, sigma_q, z_inc)
        test_cache_hits_batches_budget(lattice, key_of, sigma_q, z_inc)
        test_f(lattice, key_of, sigma_q)
        extra['g_run'] = test_g(lattice, key_of, sigma_q)
        test_h(lattice, key_of, sigma_q, z_inc)
        test_i(sigma_q, w2)
        test_c3(lattice, key_of, sigma_q)
        test_initial_incumbent_ties(lattice, key_of, sigma_q)
        test_cache_loader(lattice, key_of, scratch)
        s2_entries = test_real_c3_exclusion_and_s2(lattice, key_of, domain)
        extra['expected_from_x0'] = expected_poll_from_x0(lattice, key_of, s2_entries, s3_spec, sigma_q)
        rule11 = B.rule_eleven(lattice, sigma_q)
        check('rule eleven (Phase B record capture paths) asserts clean', all(rule11['checks'].values()),
              rule11['checks'])
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        check('no exception', False, tb)
    guard = B.PARENT_GUARD.verify(0)
    check('solve-profile guard: exactly 0 solves / execs', not guard, {'counts': B.PARENT_GUARD.counts,
                                                                        'verify_0_failures': guard})
    out = {'script': os.path.basename(__file__), 'script_sha256': H.sha256_file(os.path.abspath(__file__)),
           'launcher': 'p515_s47_phase_b_record.py',
           'launcher_sha256': H.sha256_file(os.path.join(REPO, 'p515_s47_phase_b_record.py')),
           'git_head': H._git(['rev-parse', 'HEAD']), 'all_pass': all(r['pass'] for r in RESULTS.values()),
           'n_checks': len(RESULTS), 'checks': RESULTS, 'extra': extra,
           'guard': {'counts': dict(B.PARENT_GUARD.counts), 'verify_0_failures': guard}}
    _write(os.path.abspath(args.out), out)
    print(f"ALL PASS: {out['all_pass']} ({sum(r['pass'] for r in RESULTS.values())}/{len(RESULTS)})", flush=True)
    sys.exit(0 if out['all_pass'] else 1)


if __name__ == '__main__':
    main()
