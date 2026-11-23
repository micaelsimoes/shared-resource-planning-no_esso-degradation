"""P5.15 S56 EV search -- the SEARCH POLICY (pure; standard library only; no repository import, no I/O).

Authority: S56_SEARCH_DRIVER_DESIGN.md (commit 2c15db47; Advisor review 2026-11-23 and the Planner decisions logged
there); TASKS.md "Working mode and EV priority" and "Search policy review (O1-O15)"; S56_PLAN_v4_1.md sections 3-4.
Every decision of the search lives here; the driver (`p515_s56_search_driver.py`) only supplies the neighbourhood,
evaluates what `step` asks for and stores records. Gate: `p515_s56_search_gate.py` (zero solves, guard armed).

THE ORDER (O1-O15; `order`). The neighbourhood N(x) is enumerated in full first (the driver passes the points of
`p515_s56_neighbourhood.s56_neighbourhood`) and sorted by the strictly total key
    ( -c(d), class_rank(move), cohorts(d), label(z) )
  d = z - z_inc, the integer displacement in lattice granules (zP in 0.25 MVA, zE in 0.5 MWh);
  c(d) = sign(d . m) (d . m)^2 / |d|^2, m = the LAST ACCEPTED MOVE (z_inc - z_previous_incumbent): c is a monotone
         transform of cos(d, m) (the common factor 1/|m|^2 is dropped), held as an exact `fractions.Fraction` of
         integers, so cosines are compared in exact integer arithmetic. Larger cosine first.
         NO LAST MOVE (the first poll from a start): c(d) = 0 for every point, so the order falls to the next keys.
  class_rank: the move's class position in `class_rank` (default CLASS_RANK = resize, shift, redistribute, activate,
         remove; the EV control uses CLASS_RANK_REVERSED). Move class from the neighbourhood point
         (`move_class`): R3 (a) coordinate steps and (e) diagonal resizes -> resize; R3 (c) -> shift;
         (f) -> redistribute; R3 (b) activations -> activate; R3 (b) deactivations and (d) removals -> remove.
  cohorts(d): the tuple of changed cohorts (node, year), in lattice cohort order (node-major, then year): "node, year".
  label(z): `GeneralLattice.label(z)` -- unique per point, so the key is strictly total (asserted).
  No evaluated value and nothing from another arm enters the order.
  CAP = 80 (Expert ruling 5) is owned here: a neighbourhood larger than CAP is refused (`CapExceeded`); p516_lattice's
  COMPLETION_CAP = 60 and `over_cap` are ignored. Diagonal moves are same-sign only (the neighbourhood module's rule;
  an opposite-sign diagonal is refused by `move_class`).

EXIT CLASSIFICATION (`classify_exit`; Planner decision, logged in the design). Exit values are compared with exit values
only; any other depth is refused (`MixedDepthError`). On the candidate's exit summary:
  harness_error   status 'harness_error', or a certified summary whose F or bar is missing / non-finite -> stops;
  uncertified     status != 'certified' (strict string test) -- never accepted, never "worse", listed;
  otherwise, with e = max(E_FLOOR_EUR, bar_x + bar_inc) and dF = F_exit(x) - F_exit(inc):
  worse              dF >= R + e
  confirm_candidate  dF <= -R + e
  unresolved         otherwise.
  The classes partition N(x) exactly (`terminal_report` asserts it).

CONFIRMATION (`confirm`; plan v4.1 section 3, unchanged). Settled summaries of x and of the incumbent (tail on,
certification stop disabled, to the pre-declared cap K_EV). Accepted iff
  F_set(inc) - F_set(x) >= R,  V_pair = S(x) + S(inc) + |PR(x) - PR(inc)| + G(x) + G(inc) <= R/4,
  A(x) + A(inc) <= R/2 (the cancellation guard, 5,000 EUR), and every gate of both evaluations passes.
  HELD-OUT CHECK (design, pilot protocol): offset = |dF_exit - dF_set| > e (the e of the pair's exit classification)
  stops the search for review. Otherwise the candidate is "unresolved after confirmation".

EXECUTION (`step`). Fixed blocks of BLOCK = 5: the next 5 uncached candidates in order are requested in one action;
they are then classified in order, so the incumbent path equals the serial path (no rolling refill). Candidates in
the confirm-candidate class are confirmed in order (one confirmation action: the candidate's settled run plus the
incumbent's if not yet stored); the first accepted becomes the incumbent and a new poll starts. A poll that ends with
no acceptance is COMPLETE and ends the search ("complete poll on failure").
BUDGET (Planner decision): every evaluation `step` requested and received counts 1 against N_max -- block
evaluations ordered after an accepted candidate included (role 'speculative_post_acceptance'; they are cache hits,
counting 0, when a later poll reaches them), settled confirmations included. A record that is present without having
been requested by this search is a cache hit (0). An evaluation that was requested but never delivered (killed by an
interruption) is not counted: the same action is re-issued. The budget is checked before each block: an exit block is
truncated to the remaining budget; a confirmation that does not fit, or a block when nothing remains, ends the search
('budget_exhausted', coverage reported).
RE-CLASSIFICATION (`reclassify`): classification is a pure function of stored values; when step 5 measures e, the
history is re-classified at zero solves with the measured floor and the changed classes are listed.

Interfaces (duck-typed; nothing imported):
  env.lattice                 -- a `p516_lattice.GeneralLattice` (uses .cohorts, .years, .nodes, .label, .pair, .dim)
  env.neighbourhood_points(z) -- the points of `s56_neighbourhood(lattice, z)` ({'class', 'part', 'move', 'z'})
  records                     -- mapping (tuple(z), depth) -> summary dict:
     exit:    {'depth': 'exit', 'status': 'certified' | 'not_certified' | 'harness_error', 'F', 'bar', ...}
     settled: {'depth': 'settled', 'status': 'settled' | 'not_settled' | 'harness_error', 'F',
               'V': {'S', 'PR', 'G', 'A'}, 'gates_pass': bool, ...}
"""

import copy
import json
import math
from fractions import Fraction

SCHEMA_STATE = 'p515_s56_search_state_v1'
R_EUR = 10000.0                 # S56_PLAN_v4_1.md section 3
E_FLOOR_EUR = 15000.0           # interim exit error floor (Planner decision, logged in the design)
V_PAIR_FRACTION = 0.25          # V_pair <= R/4
A_GUARD_FRACTION = 0.5          # A(x) + A(x') <= 5,000 EUR = R/2
BLOCK = 5                       # fixed blocks of 5
CAP = 80                        # Expert ruling 5; owned by the driver/policy (p516_lattice's 60 is ignored)
CLASS_RANK = ('resize', 'shift', 'redistribute', 'activate', 'remove')
CLASS_RANK_REVERSED = tuple(reversed(CLASS_RANK))
DEPTH_EXIT = 'exit'
DEPTH_SETTLED = 'settled'
DEPTHS = (DEPTH_EXIT, DEPTH_SETTLED)
EXIT_STATUSES = ('certified', 'not_certified', 'harness_error')
SETTLED_STATUSES = ('settled', 'not_settled', 'harness_error')
EXIT_CLASSES = ('confirm_candidate', 'worse', 'unresolved', 'uncertified', 'harness_error')
# the terminal partition of the final poll: confirm_candidate splits by its confirmation outcome
PARTITION_CLASSES = ('accepted', 'unresolved_after_confirmation', 'confirm_candidate_unconfirmed', 'worse',
                     'unresolved', 'uncertified', 'harness_error', 'not_reached')
ROLES = ('incumbent_exit', 'decision_relevant', 'speculative_post_acceptance', 'speculative_post_termination',
         'confirmation', 'unclassified')
TERMINATIONS = ('complete_poll_no_accepted_candidate', 'budget_exhausted', 'harness_error', 'stop_for_review_offset',
                'start_incumbent_not_certified')
SAME_SIGN_DIAGONALS = ('+1+1', '-1-1', '+1+2', '-1-2')


class PolicyError(ValueError):
    """A refused input (never a classification)."""


class MixedDepthError(PolicyError):
    pass


class CapExceeded(PolicyError):
    pass


# ======================================================================================================================
#  the order
# ======================================================================================================================
def move_class(point):
    """The O1-O15 class of one neighbourhood point (module docstring). Refuses anything it does not recognise."""
    cls, part, move = point.get('class'), point.get('part'), str(point.get('move'))
    if cls == 'r3':
        if part == 'a' and move.startswith('coord:'):
            return 'resize'
        if part == 'b' and move.startswith('activate:'):
            return 'activate'
        if part == 'b' and move.startswith('deactivate:'):
            return 'remove'
        if part == 'c' and move.startswith('shift:'):
            return 'shift'
    elif cls == 'd' and move.startswith('remove:'):
        return 'remove'
    elif cls == 'e' and move.startswith('diag:'):
        if move.split(':')[1] not in SAME_SIGN_DIAGONALS:
            raise PolicyError(f'opposite-sign diagonal in N(x) (same-sign only): {move}')
        return 'resize'
    elif cls == 'f' and move.startswith('redist:'):
        return 'redistribute'
    raise PolicyError(f'unrecognised neighbourhood point: class={cls!r} part={part!r} move={move!r}')


def _ints(z, dim=None, name='z'):
    z = tuple(z)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in z) or (dim is not None and len(z) != dim):
        raise PolicyError(f'{name} must be {dim} ints, got {z!r}')
    return z


def cosine_key(d, last_move):
    """Exact monotone transform of cos(d, m): sign(d.m) (d.m)^2 / |d|^2 as a Fraction (0 when there is no last move)."""
    if last_move is None:
        return Fraction(0)
    s = sum(a * b for a, b in zip(d, last_move))
    q = sum(a * a for a in d)
    if q == 0:
        raise PolicyError('zero displacement (the incumbent is never a neighbour)')
    return Fraction((1 if s > 0 else -1 if s < 0 else 0) * s * s, q)


def order(lattice, z_inc, last_move, class_rank, points):
    """The poll order of N(z_inc) (module docstring). `points`: the s56_neighbourhood points (any input order).
    Returns a list of dicts {'pos', 'z', 'label', 'class', 'move', 'cos_key', 'rank', 'cohorts'}."""
    z_inc = _ints(z_inc, lattice.dim, 'z_inc')
    if last_move is not None:
        last_move = _ints(last_move, lattice.dim, 'last_move')
        if not any(last_move):
            raise PolicyError('last_move is the zero vector')
    class_rank = tuple(class_rank)
    if sorted(class_rank) != sorted(CLASS_RANK):
        raise PolicyError(f'class_rank must be a permutation of {CLASS_RANK}, got {class_rank}')
    if len(points) > CAP:
        raise CapExceeded(f'|N(x)| = {len(points)} > CAP = {CAP} at {list(z_inc)}')
    rows, seen = [], set()
    for p in points:
        z = _ints(p['z'], lattice.dim)
        if z == z_inc or z in seen:
            raise PolicyError(f'neighbourhood point repeated or equal to the incumbent: {list(z)}')
        seen.add(z)
        d = tuple(a - b for a, b in zip(z, z_inc))
        cls = move_class(p)
        cohorts = tuple((n, y) for n, y in lattice.cohorts if lattice.pair(z, n, y) != lattice.pair(z_inc, n, y))
        ck = cosine_key(d, last_move)
        label = lattice.label(z)
        rows.append({'sort': (-ck, class_rank.index(cls), cohorts, label), 'z': list(z), 'label': label,
                     'class': cls, 'move': p['move'], 'cos_key': [ck.numerator, ck.denominator],
                     'rank': class_rank.index(cls), 'cohorts': [list(c) for c in cohorts]})
    keys = [r['sort'] for r in rows]
    if len(set(keys)) != len(keys):
        raise PolicyError('the sort key is not strictly total on this neighbourhood')
    rows.sort(key=lambda r: r['sort'])
    out = []
    for pos, r in enumerate(rows):
        r = {k: v for k, v in r.items() if k != 'sort'}
        r['pos'] = pos
        out.append(r)
    return out


# ======================================================================================================================
#  classification and confirmation (pure functions of stored values)
# ======================================================================================================================
def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _need_depth(summary, depth, name):
    if not isinstance(summary, dict) or summary.get('depth') != depth:
        got = summary.get('depth') if isinstance(summary, dict) else type(summary).__name__
        raise MixedDepthError(f'{name}: depth {got!r} where {depth!r} is required (mixed-depth comparison refused)')


def exit_error(bar_x, bar_inc, e_floor=E_FLOOR_EUR):
    return max(float(e_floor), float(bar_x) + float(bar_inc))


def classify_exit(x, inc, R=R_EUR, e_floor=E_FLOOR_EUR):
    """Exit class of candidate summary `x` against incumbent summary `inc` (module docstring)."""
    _need_depth(x, DEPTH_EXIT, 'candidate')
    _need_depth(inc, DEPTH_EXIT, 'incumbent')
    if inc.get('status') != 'certified' or not _finite(inc.get('F')) or not _finite(inc.get('bar')):
        raise PolicyError(f'the incumbent exit summary must be certified with a finite F and bar: {inc}')
    out = {'class': None, 'dF': None, 'e': None, 'R': R, 'e_floor': e_floor, 'criterion': None, 'reason': None}
    status = x.get('status')
    if status == 'harness_error' or status not in EXIT_STATUSES:
        return dict(out, **{'class': 'harness_error', 'reason': f'status {status!r}'})
    if status != 'certified':
        return dict(out, **{'class': 'uncertified', 'reason': f'status {status!r}'})
    if not _finite(x.get('F')) or not _finite(x.get('bar')):
        return dict(out, **{'class': 'harness_error',
                            'reason': f"certified summary with F={x.get('F')!r} bar={x.get('bar')!r} (missing bar)"})
    e = exit_error(x['bar'], inc['bar'], e_floor)
    dF = float(x['F']) - float(inc['F'])
    if dF >= R + e:
        cls, crit = 'worse', 'dF_exit >= R + e'
    elif dF <= -R + e:
        cls, crit = 'confirm_candidate', 'dF_exit <= -R + e'
    else:
        cls, crit = 'unresolved', '-R + e < dF_exit < R + e'
    return dict(out, **{'class': cls, 'dF': dF, 'e': e, 'criterion': crit})


def v_pair(vx, vi):
    return vx['S'] + vi['S'] + abs(vx['PR'] - vi['PR']) + vx['G'] + vi['G']


def confirm(x_set, inc_set, x_exit, inc_exit, e, R=R_EUR):
    """Settled confirmation of a confirm-candidate (module docstring). `e`: the e of the pair's exit classification.
    -> {'result': 'accepted' | 'unresolved_after_confirmation' | 'stop_for_review' | 'harness_error', ...}."""
    _need_depth(x_set, DEPTH_SETTLED, 'candidate settled')
    _need_depth(inc_set, DEPTH_SETTLED, 'incumbent settled')
    _need_depth(x_exit, DEPTH_EXIT, 'candidate exit')
    _need_depth(inc_exit, DEPTH_EXIT, 'incumbent exit')
    out = {'result': None, 'improvement_set': None, 'dF_set': None, 'dF_exit': None, 'offset': None, 'e': e,
           'V_pair': None, 'A_sum': None, 'gates_pass': None, 'failed': [], 'R': R}
    for name, s in (('candidate', x_set), ('incumbent', inc_set)):
        st = s.get('status')
        if st != 'settled':
            why = (f'{name} settled status {st!r}' if st in SETTLED_STATUSES and st != 'harness_error'
                   else f'{name} settled status {st!r} (harness error)')
            return dict(out, result='harness_error', failed=[why])
        v = s.get('V')
        if not _finite(s.get('F')) or not isinstance(v, dict) or not all(_finite(v.get(k)) for k in 'S PR G A'.split()):
            return dict(out, result='harness_error', failed=[f'{name} settled summary lacks F or V components'])
        if not isinstance(s.get('gates_pass'), bool):
            return dict(out, result='harness_error', failed=[f'{name} settled gates_pass is not a bool'])
    dF_set = float(x_set['F']) - float(inc_set['F'])
    dF_exit = float(x_exit['F']) - float(inc_exit['F'])
    offset = abs(dF_exit - dF_set)
    out.update(dF_set=dF_set, improvement_set=-dF_set, dF_exit=dF_exit, offset=offset)
    if offset > e:
        return dict(out, result='stop_for_review', failed=[f'held-out check: offset {offset} > e {e}'])
    vp = v_pair(x_set['V'], inc_set['V'])
    a_sum = x_set['V']['A'] + inc_set['V']['A']
    gates = x_set['gates_pass'] is True and inc_set['gates_pass'] is True
    out.update(V_pair=vp, A_sum=a_sum, gates_pass=gates)
    failed = []
    if not -dF_set >= R:
        failed.append('F_set(inc) - F_set(x) < R')
    if not vp <= V_PAIR_FRACTION * R:
        failed.append('V_pair > R/4')
    if not a_sum <= A_GUARD_FRACTION * R:
        failed.append('A(x) + A(inc) > R/2')
    if not gates:
        failed.append('a gate failed')
    return dict(out, result='accepted' if not failed else 'unresolved_after_confirmation', failed=failed)


# ======================================================================================================================
#  the state machine
# ======================================================================================================================
def initial_state(config):
    """`config` keys: start (z list), n_max (int), arm (str); optional R, e_floor, block, class_rank, domain_size,
    k_ev and any provenance (copied verbatim)."""
    cfg = dict(config)
    cfg.setdefault('R', R_EUR)
    cfg.setdefault('e_floor', E_FLOOR_EUR)
    cfg.setdefault('block', BLOCK)
    cfg.setdefault('cap', CAP)
    cfg.setdefault('class_rank', list(CLASS_RANK))
    if cfg['cap'] != CAP or cfg['block'] != BLOCK:
        raise PolicyError(f'cap and block are fixed by the design ({CAP}, {BLOCK})')
    if not isinstance(cfg.get('n_max'), int) or isinstance(cfg.get('n_max'), bool) or cfg['n_max'] < 0:
        raise PolicyError('n_max must be a non-negative int')
    cfg['start'] = list(_ints(cfg['start'], None, 'start'))
    cfg['class_rank'] = list(cfg['class_rank'])
    return {'schema': SCHEMA_STATE, 'config': cfg, 'incumbent': list(cfg['start']), 'last_move': None,
            'incumbent_history': [{'z': list(cfg['start']), 'poll': None, 'how': 'start'}],
            'poll': None, 'polls': [], 'evaluated': [], 'budget_used': 0, 'pending': None,
            'terminated': False, 'termination': None, 'n_actions': 0}


def _rk(z, depth):
    return (tuple(z), depth)


def _budget_left(state):
    return state['config']['n_max'] - state['budget_used']


def _ev_index(state, z, depth):
    for i, e in enumerate(state['evaluated']):
        if e['z'] == list(z) and e['depth'] == depth:
            return i
    return None


def _request(state, depth, zs, purpose):
    action = {'kind': 'evaluate', 'depth': depth, 'zs': [list(z) for z in zs], 'purpose': purpose,
              'poll': state['poll']['index'] if state['poll'] else None, 'action_index': state['n_actions']}
    state['pending'] = action
    state['n_actions'] += 1
    return state, copy.deepcopy(action)


def _terminate(state, reason, detail=None):
    if reason not in TERMINATIONS:
        raise PolicyError(f'unknown termination {reason}')
    for e in state['evaluated']:
        if e['role'] == 'unclassified':
            e['role'] = 'speculative_post_termination'
    if state['poll'] is not None:
        state['poll']['closed'] = reason
        state['polls'].append(state['poll'])
        state['poll'] = None
    state['terminated'] = True
    state['termination'] = {'reason': reason, 'detail': detail, 'budget_used': state['budget_used']}
    return state, {'kind': 'terminated', 'termination': copy.deepcopy(state['termination'])}


def _start_poll(state, env):
    z_inc = tuple(state['incumbent'])
    ordered = order(env.lattice, z_inc, state['last_move'], state['config']['class_rank'],
                    env.neighbourhood_points(z_inc))
    state['poll'] = {'index': len(state['polls']), 'z_inc': list(z_inc), 'last_move': state['last_move'],
                     'order': ordered, 'n': len(ordered), 'pos': 0, 'classified': [], 'closed': None,
                     'requested_in_poll': [], 'cache_hits_at_start': None}


def step(state, records, env):
    """One decision step (module docstring). Pure: returns a NEW state and the action
    {'kind': 'evaluate', 'depth', 'zs', 'purpose', ...} or {'kind': 'terminated', 'termination'}."""
    state = copy.deepcopy(state)
    if state['schema'] != SCHEMA_STATE:
        raise PolicyError(f"state schema {state['schema']!r}")
    if state['terminated']:
        return state, {'kind': 'terminated', 'termination': copy.deepcopy(state['termination'])}
    cfg = state['config']
    pend = state['pending']
    if pend is not None:
        missing = [z for z in pend['zs'] if _rk(z, pend['depth']) not in records]
        if missing:  # not delivered (killed): re-issue the same action, nothing counted
            return state, copy.deepcopy(pend)
        for z in pend['zs']:
            if _ev_index(state, z, pend['depth']) is not None:
                raise PolicyError(f'evaluated twice: {z} {pend["depth"]}')
            role = {'block': 'unclassified', 'confirmation': 'confirmation',
                    'incumbent_exit': 'incumbent_exit'}[pend['purpose']]
            state['evaluated'].append({'z': list(z), 'depth': pend['depth'], 'role': role, 'poll': pend['poll'],
                                       'action_index': pend['action_index']})
            state['budget_used'] += 1
            if state['poll'] is not None and pend['depth'] == DEPTH_EXIT:
                state['poll']['requested_in_poll'].append(list(z))
        state['pending'] = None
    while True:
        z_inc = tuple(state['incumbent'])
        inc_exit = records.get(_rk(z_inc, DEPTH_EXIT))
        if inc_exit is None:
            if _budget_left(state) < 1:
                return _terminate(state, 'budget_exhausted', 'no budget for the start incumbent exit')
            return _request(state, DEPTH_EXIT, [z_inc], 'incumbent_exit')
        if state['poll'] is None:
            if inc_exit.get('status') != 'certified' or not _finite(inc_exit.get('F')) \
                    or not _finite(inc_exit.get('bar')):
                return _terminate(state, 'start_incumbent_not_certified', {'status': inc_exit.get('status')})
            _start_poll(state, env)
            state['poll']['cache_hits_at_start'] = [r['pos'] for r in state['poll']['order']
                                                    if _rk(r['z'], DEPTH_EXIT) in records]
        poll = state['poll']
        if poll['pos'] >= poll['n']:
            return _terminate(state, 'complete_poll_no_accepted_candidate')
        cand = poll['order'][poll['pos']]
        cz = tuple(cand['z'])
        x_exit = records.get(_rk(cz, DEPTH_EXIT))
        if x_exit is None:
            left = _budget_left(state)
            if left < 1:
                return _terminate(state, 'budget_exhausted', {'poll': poll['index'], 'pos': poll['pos']})
            block = [r['z'] for r in poll['order'][poll['pos']:] if _rk(r['z'], DEPTH_EXIT) not in records]
            block = block[:min(cfg['block'], left)]
            return _request(state, DEPTH_EXIT, block, 'block')
        cls = classify_exit(x_exit, inc_exit, cfg['R'], cfg['e_floor'])
        i = _ev_index(state, cz, DEPTH_EXIT)
        if i is not None and state['evaluated'][i]['role'] == 'unclassified':
            state['evaluated'][i]['role'] = 'decision_relevant'
        entry = {'pos': poll['pos'], 'z': list(cz), 'label': cand['label'], 'exit_class': cls['class'],
                 'dF_exit': cls['dF'], 'e': cls['e'], 'criterion': cls['criterion'], 'reason': cls['reason'],
                 'cache_hit': poll['pos'] in poll['cache_hits_at_start'], 'confirmation': None}
        if cls['class'] == 'harness_error':
            poll['classified'].append(entry)
            poll['pos'] += 1
            return _terminate(state, 'harness_error', {'pos': entry['pos'], 'z': entry['z'], 'reason': cls['reason']})
        if cls['class'] != 'confirm_candidate':
            poll['classified'].append(entry)
            poll['pos'] += 1
            continue
        need = [z for z in (cz, z_inc) if _rk(z, DEPTH_SETTLED) not in records]
        if need:
            if _budget_left(state) < len(need):
                entry['confirmation'] = {'result': 'not_run_budget'}
                poll['classified'].append(entry)
                poll['pos'] += 1
                return _terminate(state, 'budget_exhausted',
                                  {'poll': poll['index'], 'pos': entry['pos'], 'confirmation_needs': len(need)})
            return _request(state, DEPTH_SETTLED, need, 'confirmation')
        res = confirm(records[_rk(cz, DEPTH_SETTLED)], records[_rk(z_inc, DEPTH_SETTLED)], x_exit, inc_exit,
                      cls['e'], cfg['R'])
        entry['confirmation'] = res
        poll['classified'].append(entry)
        poll['pos'] += 1
        if res['result'] in ('stop_for_review', 'harness_error'):
            return _terminate(state, 'stop_for_review_offset' if res['result'] == 'stop_for_review'
                              else 'harness_error', {'pos': entry['pos'], 'z': entry['z'], 'failed': res['failed']})
        if res['result'] != 'accepted':
            continue
        for e in state['evaluated']:
            if e['role'] == 'unclassified':
                e['role'] = 'speculative_post_acceptance'
        state['last_move'] = [a - b for a, b in zip(cz, z_inc)]
        state['incumbent'] = list(cz)
        state['incumbent_history'].append({'z': list(cz), 'poll': poll['index'], 'how': 'accepted',
                                           'pos': entry['pos'], 'label': cand['label'],
                                           'improvement_set': res['improvement_set'], 'V_pair': res['V_pair']})
        poll['closed'] = 'accepted'
        state['polls'].append(poll)
        state['poll'] = None


def run_to_end(state, records, env, evaluate_fn, max_actions=100000):
    """Drive `step` with an in-process `evaluate_fn(action) -> {(tuple(z), depth): summary}` (dry runs, gate)."""
    for _ in range(max_actions):
        state, action = step(state, records, env)
        if action['kind'] == 'terminated':
            return state, records
        new = evaluate_fn(action)
        for k, v in new.items():
            if k in records:
                raise PolicyError(f'record {k} delivered twice (write-once)')
            records[k] = v
    raise PolicyError('max_actions reached')


# ======================================================================================================================
#  reporting
# ======================================================================================================================
def terminal_report(state, records=None):
    """The v3 section 4 terminal report from the state alone (plus wall times from `records` when given)."""
    if not state['terminated']:
        raise PolicyError('terminal_report on a search that has not terminated')
    final = state['polls'][-1] if state['polls'] else None
    partition = {c: 0 for c in PARTITION_CLASSES}
    unresolved = []
    if final is not None:
        for c in final['classified']:
            k = c['exit_class']
            if k == 'confirm_candidate':
                r = (c['confirmation'] or {}).get('result')
                k = {'accepted': 'accepted', 'unresolved_after_confirmation': 'unresolved_after_confirmation'}.get(
                    r, 'confirm_candidate_unconfirmed')
            partition[k] += 1
            if k in ('unresolved', 'unresolved_after_confirmation', 'confirm_candidate_unconfirmed'):
                unresolved.append({'pos': c['pos'], 'label': c['label'], 'class': k, 'dF_exit': c['dF_exit'],
                                   'e': c['e'], 'criterion': c['criterion'],
                                   'confirmation_failed': (c['confirmation'] or {}).get('failed')})
        partition['not_reached'] = final['n'] - len(final['classified'])
        if sum(partition.values()) != final['n']:
            raise PolicyError('the class partition does not sum to |N|')
    roles = {r: 0 for r in ROLES}
    for e in state['evaluated']:
        roles[e['role']] += 1
    if roles['unclassified']:
        raise PolicyError('unclassified evaluations after termination')
    cache_hits = sum(1 for p in state['polls'] for c in p['classified'] if c['cache_hit'])
    hours = None
    if records is not None:
        walls = [(records.get(_rk(e['z'], e['depth'])) or {}).get('wall_s') for e in state['evaluated']]
        hours = sum(w for w in walls if _finite(w)) / 3600.0 if any(_finite(w) for w in walls) else None
    distinct = len({tuple(e['z']) for e in state['evaluated']})
    dom = state['config'].get('domain_size')
    return {
        'termination': state['termination'],
        'final_incumbent': state['incumbent'],
        'n_moves': len(state['incumbent_history']) - 1,
        'final_poll': None if final is None else {'index': final['index'], 'N': final['n'],
                                                  'k_classified': len(final['classified']),
                                                  'k_of_N': f"{len(final['classified'])} of {final['n']}"},
        'class_partition_final_poll': partition,
        'unresolved_final_poll': unresolved,
        'evaluations': {'budget_used': state['budget_used'], 'n_max': state['config']['n_max'], 'by_role': roles,
                        'cache_hits_classified': cache_hits},
        'hours_from_records': hours,
        'distinct_plans_evaluated': distinct,
        'share_of_domain_evaluated': (distinct / dom) if dom else None,
        'count_identity': count_identity(state),
    }


def count_identity(state):
    """The design's count: N_search = S_0 + sum_j (E_j + c_j) + E_f + c_f with E_j = min(n_j, B ceil(p_j / B)) for an
    accepted poll (p_j = the number of uncached candidates up to and including the accepted one) and E_f = the
    uncached candidates the final poll requested. Checked against the budget actually used."""
    B = state['config']['block']
    s0 = sum(1 for e in state['evaluated'] if e['role'] == 'incumbent_exit')
    rows, total = [], s0
    for p in state['polls']:
        cached = set(p['cache_hits_at_start'] or [])
        n_j = p['n'] - len(cached)
        c_j = sum(1 for e in state['evaluated'] if e['depth'] == DEPTH_SETTLED and e['poll'] == p['index'])
        if p['closed'] == 'accepted':
            acc = max(c['pos'] for c in p['classified'])
            p_j = sum(1 for r in p['order'][:acc + 1] if r['pos'] not in cached)
            e_j = min(n_j, B * -(-p_j // B))
        else:
            p_j = None
            e_j = sum(1 for e in state['evaluated'] if e['depth'] == DEPTH_EXIT and e['poll'] == p['index']
                      and e['role'] != 'incumbent_exit')
        requested = sum(1 for e in state['evaluated'] if e['depth'] == DEPTH_EXIT and e['poll'] == p['index']
                        and e['role'] != 'incumbent_exit')
        rows.append({'poll': p['index'], 'closed': p['closed'], 'n_j': n_j, 'p_j': p_j, 'E_j': e_j, 'c_j': c_j,
                     'exit_requested': requested, 'formula_matches': e_j == requested})
        total += requested + c_j
    # a block truncated by the budget (only possible when the search then ends 'budget_exhausted') requests fewer than
    # the formula's E_j; the identity is then stated on the evaluations actually requested
    budget_end = (state['termination'] or {}).get('reason') == 'budget_exhausted'
    return {'S_0': s0, 'per_poll': rows, 'N_search_counted': total, 'budget_used': state['budget_used'],
            'truncated_by_budget': [r['poll'] for r in rows if not r['formula_matches']],
            'holds': total == state['budget_used'] and (budget_end or all(r['formula_matches'] for r in rows))}


def reclassify(state, records, e_floor):
    """Zero-solve re-classification of every classified exit comparison at a new e floor (design: when step 5
    measures e). -> list of {'poll', 'pos', 'z', 'old', 'new'} for the comparisons whose class changes."""
    cfg = state['config']
    changes = []
    for p in state['polls'] + ([state['poll']] if state['poll'] else []):
        inc = records[_rk(p['z_inc'], DEPTH_EXIT)]
        for c in p['classified']:
            new = classify_exit(records[_rk(c['z'], DEPTH_EXIT)], inc, cfg['R'], e_floor)
            if new['class'] != c['exit_class']:
                changes.append({'poll': p['index'], 'pos': c['pos'], 'z': c['z'], 'old': c['exit_class'],
                                'new': new['class'], 'dF': new['dF'], 'e_new': new['e']})
    return changes


def canonical_json(obj):
    """The byte form used for every state comparison and hash."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)
