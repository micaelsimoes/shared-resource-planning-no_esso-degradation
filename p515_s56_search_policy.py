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
  NOT SETTLED (integration ruling 5, S56_SEARCH_DRIVER_DESIGN.md "Integration rulings", 601775c0; option b): a
  confirmation in which the candidate's or the incumbent's settled summary has status 'not_settled' is "unresolved
  after confirmation" (flagged `not_settled`, the not-settled side(s) named; no acceptance test and no held-out check
  is applied to a not-settled value) and the search continues. A 'harness_error' or unknown status on either side
  still stops the search (harness_error). REVIEW TRIGGER: the search stops for review
  ('stop_for_review_not_settled') when the last NOT_SETTLED_CONSECUTIVE_STOP = 3 confirmations were all not_settled,
  or when, with at least NOT_SETTLED_MIN_CONFIRMATIONS = 4 confirmations so far, more than NOT_SETTLED_SHARE_STOP =
  25 % of them were not_settled. "Confirmations so far" = every confirm-candidate whose confirmation produced a result
  ('accepted' or 'unresolved_after_confirmation'), in confirmation order (poll, then position), over the whole search;
  derived from the state (`not_settled_review`), so no state field is added.

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

POLL MODES (2x phase build, TASKS.md "Author decisions on report (i)" 2026-11-24, step 5a; both read from the config
with `.get`, so a config without them -- every search frozen before the phase -- runs exactly as before):
  poll_depth 'settled' (the EFFICIENCY RULE's settled-direct branch): every poll evaluation runs directly to K_EV
     (settled depth) and is classified on settled values (`classify_settled`) under the plan section 3 rule:
       improver    F_set(x) - F_set(inc) <= -R, V_pair <= R/4, A(x) + A(inc) <= R/2, every gate of both passes
       worse       F_set(x) - F_set(inc) >= R with the same V_pair / A / gate conditions (determinate)
       unresolved  otherwise (the failed conditions listed)
       not_settled the candidate's settled summary is 'not_settled' (no test applied; integration ruling 5)
       harness_error  stops the search.
     The first improver is accepted (no confirmation step: the value IS settled; no held-out check: there is no exit
     comparison). Every settled poll classification counts as a confirmation for the not_settled review trigger.
     The incumbent's settled summary must be 'settled' with gates passing (else 'start_incumbent_not_certified').
  poll_only True: a COMPLETE POLL of N(start) with no acceptance -- confirm-candidates (exit) are listed unconfirmed,
     improvers (settled) are listed unaccepted; the poll ends 'complete_poll_only'. Used for the screen candidates'
     polls of the 2x phase.
THE 2x PHASE (`phase_step`; section "the 2x phase" below): screen -> ladder -> marginal-value table + efficiency
rule -> candidate confirmations -> candidate polls -> start pool -> one search; every decision is a pure function of
the phase state and the stored records.

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
NOT_SETTLED_CONSECUTIVE_STOP = 3   # integration ruling 5 (601775c0): 3 consecutive not_settled confirmations -> review
NOT_SETTLED_SHARE_STOP = 0.25      # ... or not_settled on MORE than 25 % of the confirmations so far ...
NOT_SETTLED_MIN_CONFIRMATIONS = 4  # ... once there are at least 4 confirmations
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
                'start_incumbent_not_certified', 'stop_for_review_not_settled', 'complete_poll_only')
# settled-direct polls (poll_depth 'settled'): classes and the terminal partition
SETTLED_CLASSES = ('improver', 'worse', 'unresolved', 'not_settled', 'harness_error')
SETTLED_PARTITION_CLASSES = ('accepted', 'improver_unaccepted', 'worse', 'unresolved', 'not_settled', 'harness_error',
                             'not_reached')
ROLES_SETTLED_EXTRA = ('incumbent_settled',)
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
           'V_pair': None, 'A_sum': None, 'gates_pass': None, 'failed': [], 'R': R, 'not_settled': False}
    sides = (('candidate', x_set), ('incumbent', inc_set))
    for name, s in sides:   # a harness error or an unknown status on either side stops the search
        st = s.get('status')
        if st not in SETTLED_STATUSES or st == 'harness_error':
            return dict(out, result='harness_error', failed=[f'{name} settled status {st!r} (harness error)'])
    ns = [name for name, s in sides if s.get('status') == 'not_settled']
    if ns:   # integration ruling 5 (option b): unresolved after confirmation; the search continues
        if _finite(x_set.get('F')) and _finite(inc_set.get('F')) and _finite(x_exit.get('F')) \
                and _finite(inc_exit.get('F')):   # reported only; no test is applied to a not-settled value
            dF_set = float(x_set['F']) - float(inc_set['F'])
            dF_exit = float(x_exit['F']) - float(inc_exit['F'])
            out.update(dF_set=dF_set, improvement_set=-dF_set, dF_exit=dF_exit, offset=abs(dF_exit - dF_set))
        return dict(out, result='unresolved_after_confirmation', not_settled=True, not_settled_sides=ns,
                    failed=[f'{name} settled status not_settled' for name in ns])
    for name, s in sides:
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


def _settled_complete(s):
    v = s.get('V')
    return _finite(s.get('F')) and isinstance(v, dict) and all(_finite(v.get(k)) for k in ('S', 'PR', 'G', 'A')) \
        and isinstance(s.get('gates_pass'), bool)


def settled_incumbent_ok(s):
    """A settled summary usable as an incumbent / start: status 'settled', complete, every gate passing."""
    return isinstance(s, dict) and s.get('depth') == DEPTH_SETTLED and s.get('status') == 'settled' \
        and _settled_complete(s) and s.get('gates_pass') is True


def classify_settled(x, inc, R=R_EUR):
    """Settled-direct class of candidate settled summary `x` against the incumbent's (module docstring, POLL MODES).
    The plan section 3 determinacy rule applied to settled values; no exit value is read."""
    _need_depth(x, DEPTH_SETTLED, 'candidate settled')
    _need_depth(inc, DEPTH_SETTLED, 'incumbent settled')
    if not settled_incumbent_ok(inc):
        raise PolicyError(f'the incumbent settled summary must be settled, complete and gate-passing: {inc}')
    out = {'class': None, 'dF': None, 'V_pair': None, 'A_sum': None, 'gates_pass': None, 'failed': [], 'R': R,
           'criterion': None, 'reason': None}
    st = x.get('status')
    if st not in SETTLED_STATUSES or st == 'harness_error':
        return dict(out, **{'class': 'harness_error', 'reason': f'status {st!r}'})
    if st == 'not_settled':
        if _finite(x.get('F')):
            out['dF'] = float(x['F']) - float(inc['F'])
        return dict(out, **{'class': 'not_settled', 'reason': 'candidate settled status not_settled'})
    if not _settled_complete(x):
        return dict(out, **{'class': 'harness_error', 'reason': 'settled summary lacks F, V components or gates_pass'})
    dF = float(x['F']) - float(inc['F'])
    vp = v_pair(x['V'], inc['V'])
    a_sum = x['V']['A'] + inc['V']['A']
    gates = x['gates_pass'] is True and inc['gates_pass'] is True
    failed = []
    if not abs(dF) >= R:
        failed.append('|F_set(x) - F_set(inc)| < R')
    if not vp <= V_PAIR_FRACTION * R:
        failed.append('V_pair > R/4')
    if not a_sum <= A_GUARD_FRACTION * R:
        failed.append('A(x) + A(inc) > R/2')
    if not gates:
        failed.append('a gate failed')
    if failed:
        cls, crit = 'unresolved', 'not determinate (plan v4.1 section 3)'
    elif dF <= -R:
        cls, crit = 'improver', 'F_set(inc) - F_set(x) >= R, determinate'
    else:
        cls, crit = 'worse', 'F_set(x) - F_set(inc) >= R, determinate'
    return dict(out, **{'class': cls, 'dF': dF, 'V_pair': vp, 'A_sum': a_sum, 'gates_pass': gates,
                        'failed': failed, 'criterion': crit})


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
    poll_depth = cfg.get('poll_depth', DEPTH_EXIT)
    if poll_depth not in DEPTHS:
        raise PolicyError(f'poll_depth {poll_depth!r}')
    poll_only = cfg.get('poll_only') is True
    pend = state['pending']
    if pend is not None:
        missing = [z for z in pend['zs'] if _rk(z, pend['depth']) not in records]
        if missing:  # not delivered (killed): re-issue the same action, nothing counted
            return state, copy.deepcopy(pend)
        for z in pend['zs']:
            if _ev_index(state, z, pend['depth']) is not None:
                raise PolicyError(f'evaluated twice: {z} {pend["depth"]}')
            role = {'block': 'unclassified', 'confirmation': 'confirmation',
                    'incumbent_exit': 'incumbent_exit', 'incumbent_settled': 'incumbent_settled'}[pend['purpose']]
            state['evaluated'].append({'z': list(z), 'depth': pend['depth'], 'role': role, 'poll': pend['poll'],
                                       'action_index': pend['action_index']})
            state['budget_used'] += 1
            if state['poll'] is not None and pend['depth'] == poll_depth:
                state['poll']['requested_in_poll'].append(list(z))
        state['pending'] = None
    if poll_depth == DEPTH_SETTLED:
        return _step_settled(state, records, env, poll_only)
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
            return _terminate(state, 'complete_poll_only' if poll_only else 'complete_poll_no_accepted_candidate')
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
        if cls['class'] != 'confirm_candidate' or poll_only:   # poll_only: listed, never confirmed
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
        if res.get('not_settled'):
            review = not_settled_review(state)
            if review['stop']:
                return _terminate(state, 'stop_for_review_not_settled',
                                  dict(review, pos=entry['pos'], z=entry['z']))
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


def _step_settled(state, records, env, poll_only):
    """The settled-direct loop of `step` (module docstring, POLL MODES). `state` is already a private copy with the
    pending action registered."""
    cfg = state['config']
    while True:
        z_inc = tuple(state['incumbent'])
        inc_set = records.get(_rk(z_inc, DEPTH_SETTLED))
        if inc_set is None:
            if _budget_left(state) < 1:
                return _terminate(state, 'budget_exhausted', 'no budget for the start incumbent settled run')
            return _request(state, DEPTH_SETTLED, [z_inc], 'incumbent_settled')
        if state['poll'] is None:
            if not settled_incumbent_ok(inc_set):
                return _terminate(state, 'start_incumbent_not_certified',
                                  {'status': inc_set.get('status'), 'depth': DEPTH_SETTLED,
                                   'gates_pass': inc_set.get('gates_pass')})
            _start_poll(state, env)
            state['poll']['cache_hits_at_start'] = [r['pos'] for r in state['poll']['order']
                                                    if _rk(r['z'], DEPTH_SETTLED) in records]
        poll = state['poll']
        if poll['pos'] >= poll['n']:
            return _terminate(state, 'complete_poll_only' if poll_only else 'complete_poll_no_accepted_candidate')
        cand = poll['order'][poll['pos']]
        cz = tuple(cand['z'])
        x_set = records.get(_rk(cz, DEPTH_SETTLED))
        if x_set is None:
            left = _budget_left(state)
            if left < 1:
                return _terminate(state, 'budget_exhausted', {'poll': poll['index'], 'pos': poll['pos']})
            block = [r['z'] for r in poll['order'][poll['pos']:] if _rk(r['z'], DEPTH_SETTLED) not in records]
            block = block[:min(cfg['block'], left)]
            return _request(state, DEPTH_SETTLED, block, 'block')
        cls = classify_settled(x_set, inc_set, cfg['R'])
        i = _ev_index(state, cz, DEPTH_SETTLED)
        if i is not None and state['evaluated'][i]['role'] == 'unclassified':
            state['evaluated'][i]['role'] = 'decision_relevant'
        entry = {'pos': poll['pos'], 'z': list(cz), 'label': cand['label'], 'exit_class': None,
                 'settled_class': cls['class'], 'dF_set': cls['dF'], 'V_pair': cls['V_pair'], 'A_sum': cls['A_sum'],
                 'gates_pass': cls['gates_pass'], 'failed': cls['failed'], 'criterion': cls['criterion'],
                 'reason': cls['reason'], 'cache_hit': poll['pos'] in poll['cache_hits_at_start'],
                 'confirmation': None}
        poll['classified'].append(entry)
        poll['pos'] += 1
        if cls['class'] == 'harness_error':
            return _terminate(state, 'harness_error', {'pos': entry['pos'], 'z': entry['z'], 'reason': cls['reason']})
        if cls['class'] == 'not_settled':
            review = not_settled_review(state)
            if review['stop']:
                return _terminate(state, 'stop_for_review_not_settled', dict(review, pos=entry['pos'], z=entry['z']))
        if cls['class'] != 'improver' or poll_only:
            continue
        for e in state['evaluated']:
            if e['role'] == 'unclassified':
                e['role'] = 'speculative_post_acceptance'
        state['last_move'] = [a - b for a, b in zip(cz, z_inc)]
        state['incumbent'] = list(cz)
        state['incumbent_history'].append({'z': list(cz), 'poll': poll['index'], 'how': 'accepted',
                                           'pos': entry['pos'], 'label': cand['label'],
                                           'improvement_set': -cls['dF'], 'V_pair': cls['V_pair']})
        poll['closed'] = 'accepted'
        state['polls'].append(poll)
        state['poll'] = None


def confirmation_history(state):
    """Every confirmation that produced a result ('accepted' / 'unresolved_after_confirmation'), in confirmation order
    (closed polls in order, then the open poll; within a poll by position) -> [{'poll', 'pos', 'result',
    'not_settled'}]. Settled-direct polls: every settled classification but harness_error counts (result
    'settled_direct_<class>')."""
    out = []
    for p in state['polls'] + ([state['poll']] if state['poll'] else []):
        for c in sorted(p['classified'], key=lambda c: c['pos']):
            sc = c.get('settled_class')
            if sc is not None:
                if sc in ('improver', 'worse', 'unresolved', 'not_settled'):
                    out.append({'poll': p['index'], 'pos': c['pos'], 'result': f'settled_direct_{sc}',
                                'not_settled': sc == 'not_settled'})
                continue
            r = (c.get('confirmation') or {}).get('result')
            if r in ('accepted', 'unresolved_after_confirmation'):
                out.append({'poll': p['index'], 'pos': c['pos'], 'result': r,
                            'not_settled': (c['confirmation'] or {}).get('not_settled') is True})
    return out


def not_settled_review(state):
    """The review trigger of integration ruling 5 (module docstring), from the state alone."""
    hist = confirmation_history(state)
    n, n_ns = len(hist), sum(1 for h in hist if h['not_settled'])
    tail = 0
    for h in reversed(hist):
        if not h['not_settled']:
            break
        tail += 1
    consecutive = tail >= NOT_SETTLED_CONSECUTIVE_STOP
    share = n >= NOT_SETTLED_MIN_CONFIRMATIONS and n_ns > NOT_SETTLED_SHARE_STOP * n
    return {'stop': consecutive or share, 'n_confirmations': n, 'n_not_settled': n_ns, 'consecutive_not_settled': tail,
            'rule_consecutive': consecutive, 'rule_share': share,
            'rule': (f'{NOT_SETTLED_CONSECUTIVE_STOP} consecutive not_settled confirmations, or not_settled on more '
                     f'than {NOT_SETTLED_SHARE_STOP:.0%} of >= {NOT_SETTLED_MIN_CONFIRMATIONS} confirmations '
                     f'(integration ruling 5)')}


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
    direct = state['config'].get('poll_depth', DEPTH_EXIT) == DEPTH_SETTLED
    poll_only = state['config'].get('poll_only') is True
    partition = {c: 0 for c in (SETTLED_PARTITION_CLASSES if direct else PARTITION_CLASSES)}
    unresolved = []
    if final is not None and direct:
        for c in final['classified']:
            k = c['settled_class']
            if k == 'improver':
                k = 'improver_unaccepted' if poll_only else 'accepted'
            partition[k] += 1
            if k in ('unresolved', 'not_settled'):
                unresolved.append({'pos': c['pos'], 'label': c['label'], 'class': k, 'dF_set': c['dF_set'],
                                   'V_pair': c['V_pair'], 'failed': c['failed'], 'not_settled': k == 'not_settled'})
        partition['not_reached'] = final['n'] - len(final['classified'])
        if sum(partition.values()) != final['n']:
            raise PolicyError('the class partition does not sum to |N|')
    elif final is not None:
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
                                   'confirmation_failed': (c['confirmation'] or {}).get('failed'),
                                   'not_settled': (c['confirmation'] or {}).get('not_settled') is True})
        partition['not_reached'] = final['n'] - len(final['classified'])
        if sum(partition.values()) != final['n']:
            raise PolicyError('the class partition does not sum to |N|')
    roles = {r: 0 for r in ROLES + (ROLES_SETTLED_EXTRA if direct else ())}
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
        'not_settled_review': not_settled_review(state),
    }


def count_identity(state):
    """The design's count: N_search = S_0 + sum_j (E_j + c_j) + E_f + c_f with E_j = min(n_j, B ceil(p_j / B)) for an
    accepted poll (p_j = the number of uncached candidates up to and including the accepted one) and E_f = the
    uncached candidates the final poll requested. Checked against the budget actually used."""
    B = state['config']['block']
    bd = state['config'].get('poll_depth', DEPTH_EXIT)   # the depth of the poll blocks (settled-direct: settled)
    inc_roles = ('incumbent_exit', 'incumbent_settled')
    s0 = sum(1 for e in state['evaluated'] if e['role'] in inc_roles)
    rows, total = [], s0
    for p in state['polls']:
        cached = set(p['cache_hits_at_start'] or [])
        n_j = p['n'] - len(cached)
        if bd == DEPTH_EXIT:
            c_j = sum(1 for e in state['evaluated'] if e['depth'] == DEPTH_SETTLED and e['poll'] == p['index'])
        else:   # settled-direct: no confirmation step (c_j = 0 by construction)
            c_j = sum(1 for e in state['evaluated'] if e['role'] == 'confirmation' and e['poll'] == p['index'])
        if p['closed'] == 'accepted':
            acc = max(c['pos'] for c in p['classified'])
            p_j = sum(1 for r in p['order'][:acc + 1] if r['pos'] not in cached)
            e_j = min(n_j, B * -(-p_j // B))
        else:
            p_j = None
            e_j = sum(1 for e in state['evaluated'] if e['depth'] == bd and e['poll'] == p['index']
                      and e['role'] not in inc_roles)
        requested = sum(1 for e in state['evaluated'] if e['depth'] == bd and e['poll'] == p['index']
                        and e['role'] not in inc_roles)
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
        if any(c.get('settled_class') is not None for c in p['classified']):
            continue   # settled-direct polls carry no exit classification
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


# ======================================================================================================================
#  THE 2x PHASE (TASKS.md "Author decisions on report (i)", 2026-11-24, and step 5a; S56_PLAN_v4_1.md section 4 (g)).
#  Pure: `phase_step(state, records, env)` -> (new state, action), like `step`. The driver (`--phase-run`) evaluates
#  what it asks for, one block of <= BLOCK evaluations per launch, and applies the 120 h ceiling (measured hours are
#  not a decision input of this function: the ceiling suspends the phase, it never changes what runs next).
# ======================================================================================================================
#  STAGES (in order; every rule below is frozen in the phase spec):
#   screen    the screen plans (3 nodes x 3 years x 2 durations: 0.25 MVA with 1 MWh, 0.25 MVA with 0.5 MWh = 18
#             single-unit plans), at exit depth, in the spec's order; x0's exit summary (the start incumbent of the
#             marginal-value table) must be certified (else 'start_incumbent_not_certified').
#   ladder    the BEST DURATION is the duration of the certified screen plan with the lowest F_exit (ties by label);
#             the TWO BEST NODE-YEARS are the node-years with the lowest F_exit among the certified screen plans AT
#             that duration (ties by label); the ladder is P in ladder_zp (0.5, 1.0 MVA) at that duration at those
#             node-years (node-year rank, then size). A ladder point the lattice refuses (reasons(z) non-empty; e.g.
#             1.0 MVA x 4 h in 2025 exceeds the budget) is SKIPPED and logged; nothing replaces it.
#   table     the marginal-value table, every screen and ladder plan: F_exit, bar, dF_exit = F_exit(x) - F_exit(x0)
#             and the class against x0 = `classify_exit(x, x0, R, e_floor = e_EV)` (worse dF >= R + e,
#             confirm_candidate dF <= -R + e, else unresolved; e = max(e_EV, bar_x + bar_x0) -- the design rule with
#             e_EV as its floor), uncertified / harness_error listed.
#   EFFICIENCY RULE (author-approved, applied here automatically; MEASURE re-defined by the Planner, decision P1,
#             2026-11-24, phase spec v2): fraction = #(table rows whose class against x0 is NOT 'worse', i.e.
#             confirm_candidate or unresolved: the rows the exit value cannot resolve) / #(table rows) (denominator:
#             every screen and ladder row, uncertified ones included -- they are neither); fraction > 1/2 (exact
#             rational comparison) -> branch 'settled_direct': every LATER POLL evaluation (candidate polls and the
#             search) runs directly to K_EV and is classified on settled values (`classify_settled`); otherwise
#             'exit_then_confirm'. The per-class counts are recorded; the v1 measure (certified rows with
#             |F_exit(x) - F_exit(x0)| <= e_EV) is recorded beside it as INFORMATION ONLY (no decision reads it).
#   candidate_confirmation   CANDIDATES: the rows classed confirm_candidate, by (F_exit, label); greedily the best,
#             then the next whose node AND year both differ from every pick, up to n_candidates (2); fewer -> logged.
#             Each gets a settled run (K_EV, stop-disabled, pricing capture; x0's settled run beside) and the plan
#             section 3 pair test against x0 (`confirm`, e = the e of its table classification): harness_error ->
#             stop; offset > e -> stop_for_review_offset (the held-out check); not_settled -> not confirmed. CONFIRMED
#             = settled status 'settled' with every gate passing (`settled_incumbent_ok`); the pair result is recorded
#             beside, it does not gate pool entry.
#             EXIT-VIEW CHECK (Planner decision R4, 2026-11-24, phase spec v2; `exit_view_check`): for each candidate,
#             its exit run's (read_cycle, F, bar) must equal, BITWISE, the `exit_view` of its settled summary (the
#             same three read at the would-be certification cycle of the stop-disabled run) -- the check that an exit
#             view (x0's, the unit's and the staged plan's exit values in the phase) reads what an exit run reads.
#             Config 'exit_view_check' = 'stop' (default; the phase serves exit views): a mismatch on any candidate
#             whose confirmation is not a harness error stops the phase, 'exit_view_mismatch' (a first-in-order
#             harness_error stop precedes it; it precedes the offset stop). 'record' (a continuation that serves no
#             exit view, the R4 fallback): the check is recorded, nothing stops on it.
#   candidate_polls   each confirmed candidate in order: a COMPLETE POLL of N(candidate) (`step` with poll_only, at
#             the branch's depth). An unconfirmed candidate is not polled (logged).
#   pool      members: x0, the reference unit, the pilot's staged plan, the cache-index members (recomputed
#             historical plans found in the declared cache roots, if any), the confirmed candidates; BEST START = the
#             lowest settled F among members whose settled summary passes `settled_incumbent_ok` (ties by label).
#   search    one opportunistic search from the best start (`step`, the branch's poll depth), ending with a complete
#             poll at the final incumbent.
#  NOT-SETTLED REVIEW (integration ruling 5, phase-wide): the rule of `not_settled_review` over every confirmation of
#  the phase in order -- candidate confirmations, then each candidate poll's and the search's confirmation history.
#  TERMINATIONS: 'phase_complete' (the search ended complete_poll_no_accepted_candidate or budget_exhausted) or a
#  review stop: harness_error, stop_for_review_offset, stop_for_review_not_settled, start_incumbent_not_certified,
#  exit_view_mismatch (phase spec v2).
SCHEMA_PHASE_STATE = 'p515_s56_phase_state_v2'   # v2: efficiency measure P1, exit-view check R4 (phase spec v2)
PHASE_STAGES = ('screen', 'ladder', 'candidate_confirmation', 'candidate_polls', 'pool', 'search')
PHASE_TERMINATIONS = ('phase_complete', 'harness_error', 'stop_for_review_offset', 'stop_for_review_not_settled',
                      'start_incumbent_not_certified', 'exit_view_mismatch')
PHASE_REVIEW_TERMINATIONS = PHASE_TERMINATIONS[1:]
BRANCH_EXIT_THEN_CONFIRM, BRANCH_SETTLED_DIRECT = 'exit_then_confirm', 'settled_direct'
EFFICIENCY_THRESHOLD = Fraction(1, 2)
DURATION_RATIO = {'4h': 2, '2h': 1}   # zE = ratio x zP: 0.25 MVA x 1 MWh = 4 h; 0.25 MVA x 0.5 MWh = 2 h
PHASE_CONFIG_KEYS = ('phase_id', 'arm', 'x0', 'screen', 'ladder_zp', 'ladder_n_node_years', 'n_candidates',
                     'pool_fixed', 'pool_cache_members', 'R', 'e_ev', 'k_ev', 'block', 'cap', 'class_rank',
                     'n_max_sub', 'domain_size')
EFFICIENCY_NOT_WORSE_CLASSES = ('confirm_candidate', 'unresolved')   # Planner decision P1: not resolvable at exit
EXIT_VIEW_CHECK_MODES = ('stop', 'record')   # config 'exit_view_check' (optional; default 'stop')
EXIT_VIEW_FIELDS = ('read_cycle', 'F', 'bar')


def phase_initial_state(config):
    cfg = copy.deepcopy(dict(config))
    missing = [k for k in PHASE_CONFIG_KEYS if k not in cfg]
    if missing:
        raise PolicyError(f'phase config lacks {missing}')
    if cfg['cap'] != CAP or cfg['block'] != BLOCK:
        raise PolicyError(f'cap and block are fixed by the design ({CAP}, {BLOCK})')
    if not (_finite(cfg['e_ev']) and cfg['e_ev'] > 0 and _finite(cfg['R']) and cfg['R'] > 0):
        raise PolicyError('R and e_EV must be positive numbers')
    if cfg.get('exit_view_check', 'stop') not in EXIT_VIEW_CHECK_MODES:
        raise PolicyError(f"exit_view_check {cfg.get('exit_view_check')!r} not in {EXIT_VIEW_CHECK_MODES}")
    return {'schema': SCHEMA_PHASE_STATE, 'config': cfg, 'stage': 'screen', 'pending': None, 'n_actions': 0,
            'evaluated': [], 'x0_exit_check': None, 'ladder': None, 'table': None, 'efficiency': None,
            'candidates': None, 'candidate_selection': None, 'cand_pos': 0, 'pool': None, 'sub': None,
            'search': None, 'log': [], 'terminated': False, 'termination': None}


def _phase_request(state, depth, zs, purpose, sub_action=None):
    action = {'kind': 'evaluate', 'depth': depth, 'zs': [list(z) for z in zs], 'purpose': purpose,
              'stage': state['stage'], 'action_index': state['n_actions'],
              'sub_action': copy.deepcopy(sub_action)}
    state['pending'] = action
    state['n_actions'] += 1
    return state, copy.deepcopy(action)


def _phase_terminate(state, reason, detail=None):
    if reason not in PHASE_TERMINATIONS:
        raise PolicyError(f'unknown phase termination {reason}')
    state['terminated'] = True
    state['termination'] = {'reason': reason, 'detail': detail, 'stage': state['stage']}
    return state, {'kind': 'terminated', 'termination': copy.deepcopy(state['termination'])}


def _single_cohort(lattice, z):
    act = lattice.active_cohorts(tuple(z))
    if len(act) != 1:
        return None
    n, y = act[0]
    zp, ze = lattice.pair(tuple(z), n, y)
    return {'node': n, 'year': y, 'zp': zp, 'ze': ze,
            'duration': next((k for k, r in DURATION_RATIO.items() if ze == r * zp), None)}


def _table_row(lattice, z, source, summ, x0_exit, R, e_ev):
    sc = _single_cohort(lattice, z) or {}
    cls = classify_exit(summ, x0_exit, R, e_ev)
    f = summ.get('F') if summ.get('status') == 'certified' and _finite(summ.get('F')) else None
    in_band = f is not None and abs(float(f) - float(x0_exit['F'])) <= e_ev
    return {'label': lattice.label(tuple(z)), 'z': list(z), 'source': source, 'node': sc.get('node'),
            'year': sc.get('year'), 'zp': sc.get('zp'), 'ze': sc.get('ze'), 'duration': sc.get('duration'),
            'status': summ.get('status'), 'F_exit': summ.get('F'), 'bar': summ.get('bar'), 'dF_exit': cls['dF'],
            'e': cls['e'], 'class': cls['class'], 'criterion': cls['criterion'], 'reason': cls['reason'],
            'in_efficiency_band': in_band, 'served_as': summ.get('served_as')}


def ladder_plan(lattice, cfg, screen_rows):
    """The ladder rule (section comment) from the screen rows -> {'best_duration', 'node_years', 'plans', 'skipped',
    'log'}. Pure."""
    cert = [r for r in screen_rows if r['status'] == 'certified' and _finite(r['F_exit']) and r['duration']]
    out = {'rule': ('best duration = duration of the certified screen plan with the lowest F_exit (ties by label); '
                    'two best node-years = lowest F_exit among certified screen plans at that duration (ties by '
                    'label); P in ladder_zp at that duration; a ladder point the lattice refuses is skipped'),
           'best_duration': None, 'node_years': [], 'plans': [], 'skipped': [], 'log': []}
    if not cert:
        out['log'].append('no certified screen plan: no ladder')
        return out
    best = min(cert, key=lambda r: (float(r['F_exit']), r['label']))
    out['best_duration'] = best['duration']
    at = sorted((r for r in cert if r['duration'] == best['duration']), key=lambda r: (float(r['F_exit']), r['label']))
    ny = [(r['node'], r['year']) for r in at[:cfg['ladder_n_node_years']]]
    out['node_years'] = [list(v) for v in ny]
    if len(ny) < cfg['ladder_n_node_years']:
        out['log'].append(f'only {len(ny)} certified node-year(s) at the best duration: fewer ladder points')
    ratio = DURATION_RATIO[best['duration']]
    for n, y in ny:
        for zp in cfg['ladder_zp']:
            z = lattice.with_pair(lattice.x0(), n, y, zp, ratio * zp)
            why = lattice.reasons(z)
            row = {'node': n, 'year': y, 'zp': zp, 'ze': ratio * zp, 'z': list(z), 'label': lattice.label(z)}
            if why:
                out['skipped'].append(dict(row, reasons=why))
                out['log'].append(f'ladder point {row["label"]} skipped: {why}')
            else:
                out['plans'].append(row)
    return out


def efficiency_rule(rows, e_ev, threshold=EFFICIENCY_THRESHOLD):
    """The efficiency rule (section comment; measure of Planner decision P1): the share of table rows NOT classed
    'worse' against x0 (confirm_candidate + unresolved) decides; the v1 band measure is recorded as information."""
    n = len(rows)
    counts = {c: 0 for c in EXIT_CLASSES}
    for r in rows:
        if r['class'] not in counts:
            raise PolicyError(f"efficiency rule: unknown row class {r['class']!r}")
        counts[r['class']] += 1
    k = sum(counts[c] for c in EFFICIENCY_NOT_WORSE_CLASSES)
    frac = Fraction(k, n) if n else Fraction(0)
    branch = BRANCH_SETTLED_DIRECT if frac > threshold else BRANCH_EXIT_THEN_CONFIRM
    kb = sum(1 for r in rows if r['in_efficiency_band'])
    fb = Fraction(kb, n) if n else Fraction(0)
    return {'measure': 'not_worse_against_x0', 'n_rows': n, 'n_not_worse': k, 'class_counts': counts,
            'not_worse_classes': list(EFFICIENCY_NOT_WORSE_CLASSES),
            'fraction': [frac.numerator, frac.denominator], 'fraction_float': float(frac),
            'threshold': [threshold.numerator, threshold.denominator], 'branch': branch,
            'rule': ('fraction of the screen and ladder rows whose class against x0 at exit is NOT worse '
                     '(confirm_candidate + unresolved; Planner decision P1) (denominator: every screen and ladder '
                     'row, uncertified ones included); > 1/2 -> every later poll evaluation runs directly to K_EV '
                     '(settled_direct), else exit_then_confirm'),
            'band_measure_information_only': {
                'n_in_band': kb, 'fraction': [fb.numerator, fb.denominator], 'fraction_float': float(fb),
                'band_eur': e_ev,
                'branch_the_v1_measure_would_give': BRANCH_SETTLED_DIRECT if fb > threshold
                else BRANCH_EXIT_THEN_CONFIRM,
                'rule': ('v1 measure: certified rows with |F_exit(x) - F_exit(x0)| <= e_EV over every row; recorded '
                         'beside the deciding measure, read by no decision (Planner decision P1)')}}


def _bitwise(v):
    """The exact identity of a JSON scalar for the exit-view check: floats by float.hex, ints by value, with the type."""
    if isinstance(v, bool) or v is None:
        return [type(v).__name__, v]
    if isinstance(v, float):
        return ['float', v.hex()]
    if isinstance(v, int):
        return ['int', v]
    return [type(v).__name__, repr(v)]


def exit_view_check(x_exit, x_set):
    """R4 (section comment): the exit run's (read_cycle, F, bar) against the settled summary's `exit_view`, bitwise.
    -> {'equal', 'differing', 'exit_run', 'exit_view', 'reason'}. A missing exit_view is a mismatch."""
    a = {f: (x_exit or {}).get(f) for f in EXIT_VIEW_FIELDS}
    ev = (x_set or {}).get('exit_view') if isinstance(x_set, dict) else None
    if not isinstance(ev, dict):
        return {'equal': False, 'differing': list(EXIT_VIEW_FIELDS), 'exit_run': a, 'exit_view': None,
                'reason': 'the settled summary carries no exit_view', 'fields': list(EXIT_VIEW_FIELDS)}
    b = {f: ev.get(f) for f in EXIT_VIEW_FIELDS}
    diff = [f for f in EXIT_VIEW_FIELDS if _bitwise(a[f]) != _bitwise(b[f])]
    return {'equal': not diff, 'differing': diff, 'exit_run': a, 'exit_view': b,
            'exit_run_bits': {f: _bitwise(a[f]) for f in EXIT_VIEW_FIELDS},
            'exit_view_bits': {f: _bitwise(b[f]) for f in EXIT_VIEW_FIELDS},
            'reason': None if not diff else f'differs in {diff}', 'fields': list(EXIT_VIEW_FIELDS)}


def select_candidates(rows, n_candidates):
    qual = sorted((r for r in rows if r['class'] == 'confirm_candidate'), key=lambda r: (float(r['F_exit']), r['label']))
    picked, passed = [], []
    for r in qual:
        if len(picked) == n_candidates:
            break
        if all(r['node'] != p['node'] and r['year'] != p['year'] for p in picked):
            picked.append(r)
        else:
            passed.append({'label': r['label'], 'reason': 'shares a node or a year with a pick'})
    log = []
    if len(picked) < n_candidates:
        log.append(f'{len(picked)} of {n_candidates} candidates: {len(qual)} row(s) classed confirm_candidate')
    return {'rule': ('rows classed confirm_candidate against x0, by (F_exit, label); greedily the best, then the next '
                     'whose node AND year both differ from every pick'),
            'n_qualified': len(qual), 'qualified': [r['label'] for r in qual], 'passed_over': passed,
            'picked': [r['label'] for r in picked], 'log': log}, picked


def _sub_config(cfg, start, poll_only, branch, tag):
    c = {'start': list(start), 'n_max': cfg['n_max_sub'], 'arm': cfg['arm'], 'R': cfg['R'], 'e_floor': cfg['e_ev'],
         'block': cfg['block'], 'cap': cfg['cap'], 'class_rank': list(cfg['class_rank']),
         'domain_size': cfg['domain_size'], 'k_ev': cfg['k_ev'], 'search_id': f"{cfg['phase_id']}_{tag}"}
    if branch == BRANCH_SETTLED_DIRECT:
        c['poll_depth'] = DEPTH_SETTLED
    if poll_only:
        c['poll_only'] = True
    return c


def phase_confirmation_history(state):
    out = []
    for c in state['candidates'] or []:
        conf = c.get('confirmation') or {}
        if conf.get('result') in ('accepted', 'unresolved_after_confirmation'):
            out.append({'where': f"candidate {c['label']}", 'result': conf['result'],
                        'not_settled': conf.get('not_settled') is True})
    for c in state['candidates'] or []:
        if c.get('poll_state') is not None:
            out += [dict(h, where=f"poll {c['label']}") for h in confirmation_history(c['poll_state'])]
    if state['sub'] is not None:
        out += [dict(h, where=state['sub']['config']['search_id']) for h in confirmation_history(state['sub'])]
    if state['search'] is not None:
        out += [dict(h, where='search') for h in confirmation_history(state['search']['state'])]
    return out


def phase_not_settled_review(state):
    hist = phase_confirmation_history(state)
    n, n_ns = len(hist), sum(1 for h in hist if h['not_settled'])
    tail = 0
    for h in reversed(hist):
        if not h['not_settled']:
            break
        tail += 1
    consecutive = tail >= NOT_SETTLED_CONSECUTIVE_STOP
    share = n >= NOT_SETTLED_MIN_CONFIRMATIONS and n_ns > NOT_SETTLED_SHARE_STOP * n
    return {'stop': consecutive or share, 'n_confirmations': n, 'n_not_settled': n_ns, 'consecutive_not_settled': tail,
            'rule_consecutive': consecutive, 'rule_share': share, 'scope': 'phase-wide'}


def _sub_review_reason(reason):
    return reason if reason in PHASE_REVIEW_TERMINATIONS else None


def phase_step(state, records, env):
    """One phase decision step (section comment). Pure: returns a NEW state and the action
    {'kind': 'evaluate', 'depth', 'zs', 'purpose', 'stage', 'action_index', 'sub_action'} or
    {'kind': 'terminated', 'termination'}."""
    state = copy.deepcopy(state)
    if state['schema'] != SCHEMA_PHASE_STATE:
        raise PolicyError(f"phase state schema {state['schema']!r}")
    if state['terminated']:
        return state, {'kind': 'terminated', 'termination': copy.deepcopy(state['termination'])}
    cfg = state['config']
    lat = env.lattice
    x0 = tuple(cfg['x0'])
    pend = state['pending']
    if pend is not None:
        if [z for z in pend['zs'] if _rk(z, pend['depth']) not in records]:
            return state, copy.deepcopy(pend)   # not delivered (killed): re-issue, nothing counted
        for z in pend['zs']:
            if any(e['z'] == list(z) and e['depth'] == pend['depth'] for e in state['evaluated']):
                raise PolicyError(f'phase: evaluated twice: {z} {pend["depth"]}')
            state['evaluated'].append({'z': list(z), 'depth': pend['depth'], 'stage': pend['stage'],
                                       'purpose': pend['purpose'], 'action_index': pend['action_index']})
        state['pending'] = None
    B = cfg['block']
    while True:
        stage = state['stage']
        x0_exit = records.get(_rk(x0, DEPTH_EXIT))
        if stage in ('screen', 'ladder'):
            if x0_exit is None:
                return _phase_request(state, DEPTH_EXIT, [x0], 'x0_exit')
            if state['x0_exit_check'] is None:
                ok = x0_exit.get('status') == 'certified' and _finite(x0_exit.get('F')) and _finite(x0_exit.get('bar'))
                state['x0_exit_check'] = {'status': x0_exit.get('status'), 'F': x0_exit.get('F'),
                                          'bar': x0_exit.get('bar'), 'ok': ok, 'served_as': x0_exit.get('served_as')}
                if not ok:
                    return _phase_terminate(state, 'start_incumbent_not_certified',
                                            {'z': list(x0), 'status': x0_exit.get('status')})
        if stage == 'screen':
            need = [z for z in cfg['screen'] if _rk(z, DEPTH_EXIT) not in records]
            if need:
                return _phase_request(state, DEPTH_EXIT, need[:B], 'screen')
            bad = [lat.label(tuple(z)) for z in cfg['screen']
                   if classify_exit(records[_rk(z, DEPTH_EXIT)], x0_exit, cfg['R'], cfg['e_ev'])['class']
                   == 'harness_error']
            if bad:
                return _phase_terminate(state, 'harness_error', {'stage': 'screen', 'labels': bad})
            rows = [_table_row(lat, z, 'screen', records[_rk(z, DEPTH_EXIT)], x0_exit, cfg['R'], cfg['e_ev'])
                    for z in cfg['screen']]
            state['ladder'] = ladder_plan(lat, cfg, rows)
            state['log'] += state['ladder']['log']
            state['stage'] = 'ladder'
            continue
        if stage == 'ladder':
            lz = [p['z'] for p in state['ladder']['plans']]
            need = [z for z in lz if _rk(z, DEPTH_EXIT) not in records]
            if need:
                return _phase_request(state, DEPTH_EXIT, need[:B], 'ladder')
            rows = [_table_row(lat, z, 'screen', records[_rk(z, DEPTH_EXIT)], x0_exit, cfg['R'], cfg['e_ev'])
                    for z in cfg['screen']]
            rows += [_table_row(lat, z, 'ladder', records[_rk(z, DEPTH_EXIT)], x0_exit, cfg['R'], cfg['e_ev'])
                     for z in lz]
            bad = [r['label'] for r in rows if r['class'] == 'harness_error']
            if bad:
                return _phase_terminate(state, 'harness_error', {'stage': 'ladder', 'labels': bad})
            state['table'] = {'x0': {'F_exit': x0_exit['F'], 'bar': x0_exit['bar'],
                                     'served_as': x0_exit.get('served_as')},
                              'R': cfg['R'], 'e_ev': cfg['e_ev'], 'rows': rows,
                              'classification': 'classify_exit(x, x0, R, e_floor = e_EV); e = max(e_EV, bar_x + bar_x0)',
                              'objective_convention': 'F = I(x) + gross_operational_cost - TV credit, EUR of total '
                                                      'discounted planning cost; dF_exit = F_exit(x) - F_exit(x0)'}
            state['efficiency'] = efficiency_rule(rows, cfg['e_ev'])
            state['log'].append(f"efficiency rule: {state['efficiency']['n_not_worse']} of "
                                f"{state['efficiency']['n_rows']} not worse against x0 "
                                f"{state['efficiency']['class_counts']} -> {state['efficiency']['branch']} (v1 band "
                                f"measure, information only: "
                                f"{state['efficiency']['band_measure_information_only']['n_in_band']} in band)")
            sel, picked = select_candidates(rows, cfg['n_candidates'])
            state['candidate_selection'] = sel
            state['log'] += sel['log']
            state['candidates'] = [{'label': r['label'], 'z': r['z'], 'node': r['node'], 'year': r['year'],
                                    'F_exit': r['F_exit'], 'e': r['e'], 'confirmation': None, 'confirmed': None,
                                    'poll_state': None, 'poll_report': None, 'poll_skipped': None} for r in picked]
            state['stage'] = 'candidate_confirmation'
            continue
        if stage == 'candidate_confirmation':
            cz = [tuple(c['z']) for c in state['candidates']]
            need = [z for z in ([x0] + cz if cz else []) if _rk(z, DEPTH_SETTLED) not in records]
            if need:
                return _phase_request(state, DEPTH_SETTLED, need[:B], 'candidate_confirmation')
            stop = None
            mode = cfg.get('exit_view_check', 'stop')
            for c in state['candidates']:
                z = tuple(c['z'])
                res = confirm(records[_rk(z, DEPTH_SETTLED)], records[_rk(x0, DEPTH_SETTLED)],
                              records[_rk(z, DEPTH_EXIT)], x0_exit, c['e'], cfg['R'])
                c['confirmation'] = res
                c['confirmed'] = res['result'] in ('accepted', 'unresolved_after_confirmation') \
                    and not res.get('not_settled') and settled_incumbent_ok(records[_rk(z, DEPTH_SETTLED)])
                c['exit_view_check'] = dict(exit_view_check(records[_rk(z, DEPTH_EXIT)], records[_rk(z, DEPTH_SETTLED)]),
                                            mode=mode)
                if stop is None and res['result'] in ('harness_error', 'stop_for_review'):
                    stop = ('harness_error' if res['result'] == 'harness_error' else 'stop_for_review_offset',
                            {'candidate': c['label'], 'failed': res['failed']})
            if stop and stop[0] == 'harness_error':
                return _phase_terminate(state, *stop)
            mism = [{'candidate': c['label'], 'differing': c['exit_view_check']['differing'],
                     'exit_run': c['exit_view_check']['exit_run'], 'exit_view': c['exit_view_check']['exit_view']}
                    for c in state['candidates'] if c['confirmation']['result'] != 'harness_error'
                    and not c['exit_view_check']['equal']]
            if mism and mode == 'stop':
                return _phase_terminate(state, 'exit_view_mismatch', {'mismatches': mism, 'rule': 'R4 (phase spec v2)'})
            if mism:
                state['log'].append(f'exit-view check (mode record): {len(mism)} mismatch(es) recorded, not stopping')
            if stop:
                return _phase_terminate(state, *stop)
            rv = phase_not_settled_review(state)
            if rv['stop']:
                return _phase_terminate(state, 'stop_for_review_not_settled', rv)
            state['stage'] = 'candidate_polls'
            continue
        if stage == 'candidate_polls':
            if state['cand_pos'] >= len(state['candidates']):
                state['stage'] = 'pool'
                continue
            c = state['candidates'][state['cand_pos']]
            if not c['confirmed']:
                c['poll_skipped'] = 'candidate not confirmed (settled status / gates)'
                state['log'].append(f"candidate {c['label']} not confirmed: poll skipped")
                state['cand_pos'] += 1
                continue
            if state['sub'] is None:
                state['sub'] = initial_state(_sub_config(cfg, c['z'], True, state['efficiency']['branch'],
                                                         f"cand{state['cand_pos'] + 1}_poll"))
            sub, act = step(state['sub'], records, env)
            state['sub'] = sub
            rv = phase_not_settled_review(state)
            if rv['stop'] and not sub['terminated']:
                c['poll_state'], state['sub'] = sub, None
                return _phase_terminate(state, 'stop_for_review_not_settled', rv)
            if act['kind'] == 'evaluate':
                return _phase_request(state, act['depth'], act['zs'], 'candidate_poll', act)
            c['poll_state'], c['poll_report'], state['sub'] = sub, terminal_report(sub), None
            why = _sub_review_reason(act['termination']['reason'])
            if why:
                return _phase_terminate(state, why, {'candidate': c['label'], 'sub': act['termination']})
            if rv['stop']:
                return _phase_terminate(state, 'stop_for_review_not_settled', rv)
            if act['termination']['reason'] != 'complete_poll_only':
                state['log'].append(f"candidate {c['label']} poll ended {act['termination']['reason']}")
            state['cand_pos'] += 1
            continue
        if stage == 'pool':
            members = [dict(m, kind='fixed') for m in cfg['pool_fixed']]
            members += [dict(m, kind='cache_member') for m in cfg['pool_cache_members']]
            members += [{'name': f'candidate_{i + 1}', 'z': c['z'], 'kind': 'confirmed_candidate'}
                        for i, c in enumerate(state['candidates']) if c['confirmed']]
            seen, uniq, aliases = {}, [], []
            for m in members:
                k = tuple(m['z'])
                if k in seen:
                    aliases.append({'name': m['name'], 'same_as': seen[k]})
                    continue
                seen[k] = m['name']
                uniq.append(m)
            need = [tuple(m['z']) for m in uniq if _rk(m['z'], DEPTH_SETTLED) not in records]
            if need:
                return _phase_request(state, DEPTH_SETTLED, need[:B], 'pool_settled')
            rows = []
            for m in uniq:
                s = records[_rk(m['z'], DEPTH_SETTLED)]
                rows.append({'name': m['name'], 'kind': m['kind'], 'z': list(m['z']), 'label': lat.label(tuple(m['z'])),
                             'status': s.get('status'), 'F_set': s.get('F'), 'gates_pass': s.get('gates_pass'),
                             'eligible': settled_incumbent_ok(s), 'served_as': s.get('served_as')})
            elig = sorted((r for r in rows if r['eligible']), key=lambda r: (float(r['F_set']), r['label']))
            state['pool'] = {'rule': 'argmin settled F over the members whose settled summary is settled, complete and '
                                     'gate-passing (ties by label)', 'members': rows, 'aliases': aliases,
                             'chosen': elig[0]['name'] if elig else None,
                             'chosen_label': elig[0]['label'] if elig else None,
                             'runner_up': ({'name': elig[1]['name'], 'label': elig[1]['label'],
                                            'margin_eur': float(elig[1]['F_set']) - float(elig[0]['F_set'])}
                                           if len(elig) > 1 else None)}
            if not elig:
                return _phase_terminate(state, 'start_incumbent_not_certified', {'stage': 'pool'})
            state['sub'] = initial_state(_sub_config(cfg, elig[0]['z'], False, state['efficiency']['branch'],
                                                     'search'))
            state['stage'] = 'search'
            continue
        if stage == 'search':
            sub, act = step(state['sub'], records, env)
            state['sub'] = sub
            rv = phase_not_settled_review(state)
            if act['kind'] == 'evaluate':
                if rv['stop']:
                    state['search'], state['sub'] = {'state': sub, 'report': None}, None
                    return _phase_terminate(state, 'stop_for_review_not_settled', rv)
                return _phase_request(state, act['depth'], act['zs'], 'search', act)
            state['search'], state['sub'] = {'state': sub, 'report': terminal_report(sub)}, None
            why = _sub_review_reason(act['termination']['reason'])
            if why:
                return _phase_terminate(state, why, {'search': act['termination']})
            if rv['stop']:
                return _phase_terminate(state, 'stop_for_review_not_settled', rv)
            return _phase_terminate(state, 'phase_complete', {'search': act['termination']})
        raise PolicyError(f'unknown phase stage {stage}')


def phase_terminal_report(state):
    """The phase record from the state alone (the driver adds hours and per-record facts)."""
    if not state['terminated']:
        raise PolicyError('phase_terminal_report on a phase that has not terminated')
    by_stage = {}
    for e in state['evaluated']:
        k = f"{e['stage']}|{e['depth']}"
        by_stage[k] = by_stage.get(k, 0) + 1
    srch = state['search']
    return {'termination': state['termination'], 'x0_exit_check': state['x0_exit_check'],
            'ladder': state['ladder'], 'marginal_value_table': state['table'], 'efficiency': state['efficiency'],
            'candidate_selection': state['candidate_selection'],
            'candidates': [{k: v for k, v in c.items() if k != 'poll_state'} for c in (state['candidates'] or [])],
            'pool': state['pool'], 'search_report': srch['report'] if srch else None,
            'search_incumbent_history': srch['state']['incumbent_history'] if srch else None,
            'evaluations_requested': len(state['evaluated']), 'evaluations_by_stage_depth': by_stage,
            'phase_not_settled_review': phase_not_settled_review(state), 'log': state['log']}
