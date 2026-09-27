"""
P5.15 Addendum 51, Planner task W98 -- the CERTIFICATION CONTINUATION hooks (stage 1, x = 0), frozen stage spec v37.

WHAT THIS MODULE IS. A child-side, harness-installed set of pass-through wrappers on SIX production functions of
`shared_resources_planning` (module globals, resolved by `_run_operational_planning` at call time -- the technique every
capture hook of `p515_s44_campaign_harness` / `p515_g_g1_g4_admm_gates` already uses). No production module is edited.
It is installed ONLY for a campaign-spec entry that declares `certification_continuation`
(`p515_s44_campaign_harness`, W98); the declaration enters the entry's eval key, so a continuation evaluation never
shares a key -- hence a directory, a working dir or a cache hit -- with the certified evaluation it replays.

THE RUN IT SERVES (PLANNER_BRIEF_2026-09-13.md Addendum 51, route A, stage 1). The certified 3 x 3 x = 0 cell (campaign
s53_w91_3x3_pair, spec 231558f0, eval key f6e9cd53...) certified at cycle N = 72. The continuation re-runs it from
scratch under the IDENTICAL configuration (the replay, gated bitwise by the launcher against the 72 committed
per-cycle records), then runs up to `continuation_cycles` (30) further cycles with
  * the certification rule DISABLED (the loop may not exit on the certificate), and
  * the certifying regime HELD for every cycle > N: Anderson acceleration OFF, the tight tail ON, rho FROZEN,
and stops early when |dQ| < `abs_gross_step_below_eur` (500 EUR) for `consecutive_cycles` (3) consecutive
post-certification cycles.

THE WRAPPERS (every one calls the production function; for cycle <= N each passes its arguments through as the SAME
objects and returns the production return value UNCHANGED -- the zero-solve checks prove this by object identity):
  1. `_capture_convergence_depth_tail_baseline` (called once, before cycle 1): records production's
     `minimum_consecutive_converged_cycles` (the certificate length, 10) and raises it to
     `CERTIFICATION_DISABLED_THRESHOLD`. That parameter is read in exactly three places (`CERTIFICATE_LENGTH_READS`,
     asserted before any solve): the loop's exit test
     `convergence = (consecutive_converged_cycles >= admm_parameters.minimum_consecutive_converged_cycles)`, the
     per-cycle diagnostics row's `required_consecutive_cycles` field (a record of the value in force) and the cycle's
     INFO print. The per-cycle consecutive count, the Boyd test and every computed quantity are untouched. Its only
     effect through cycle N is that the loop does not exit at N (and the raw row records 10**9 instead of 10).
     Raised before cycle 1 (not at N) so that a replay that diverges and certifies early still runs to N + 30 ("do
     not stop on divergence").
  2. `_apply_convergence_depth_tail` (top of every cycle; `cycle=None` once at loop exit): the CYCLE TRACKER (refuses
     a non-consecutive cycle). TAIL HOLD, part (a): for cycle > N the tail is applied ON (`active=True`) whatever the
     previous cycle's predicate said. At the loop exit it restores the production certificate length.
  3. `_anderson_acceleration_cycle_step` (after the plain cycle, when every local solve succeeded): AA HOLD -- for
     cycle > N production's own AA step is called with `boyd_metrics['all_boyd_pass']` forced True on a shallow COPY
     (the caller's dict is not touched), i.e. production's own "off (all channels within Boyd tolerance)" branch: no
     extrapolation, no write-back. The action is asserted.
  4. `_convergence_depth_tail_next_state` (end of every cycle): TAIL HOLD, part (b) -- for cycle > N it returns True
     (the tail stays on). Production's consistency check (AA-off == predicate) is bypassed ONLY there, because the
     AA hold makes AA 'off' by construction; the natural predicate is recorded.
  5. `_update_admm_penalties` (end of every cycle): RHO HOLD -- for cycle > N production's function is called with
     `allow_update=False` (its scaling loop, the only place rho / gamma are written, does not run); rho and gamma
     before == after is asserted exactly on every channel.
  6. `_get_operational_recourse_components` (in-cycle, right after every successful cycle's solves; also called
     outside the loop by terminal captures -- those calls pass straight through): captures the cycle's gross Q, and
     -- ALL-BLOCK CAPTURE -- every block of production's own `_get_operational_recourse_block_components` (80 network
     blocks + SALVAGE) and `_get_operational_objective_component_blocks`, every cycle (the harness's recourse-jump
     sidecar keeps only the top 10 by |delta|); and, for cycle > N, the EARLY-STOP rule: when the last
     `consecutive_cycles` post-certification steps dQ_k = Q_k - Q_(k-1) all satisfy |dQ_k| < the threshold, the
     certificate length is set to `EARLY_STOP_THRESHOLD` (0), so production's own loop test exits at the end of THIS
     cycle (the only loop exit production has besides the cap).
  Nothing else is wrapped. `_get_operational_recourse_block_components` and
  `_get_operational_objective_component_blocks` are CALLED (pure read functions, the ones the s34 recourse-jump
  sidecar already calls every cycle), not wrapped.

W97 found that AA-off and tail-on coincide by construction (tail on at AA-off + 1); this run holds both and CANNOT
separate them. It discriminates by the SHAPE of the post-certification steps.

Zero solves: nothing here solves or builds a model. Stdlib-only at import (the harness parent imports it for key
computation); `shared_resources_planning` is imported inside the context manager only.
"""
import json
import math
import os
import time
from contextlib import contextmanager

SCHEMA = 'p515_s53_w98_certification_continuation_v1'
LABEL = ('CONTINUATION RUN (W98, spec v37) -- NOT a certification: certification rule disabled; regime held after '
         'cycle N (AA off, tight tail on, rho frozen)')
DECLARATION_KEYS = frozenset({'label', 'hold_after_cycle', 'continuation_cycles', 'early_stop', 'record_all_blocks',
                              'replay_reference'})
EARLY_STOP_KEYS = frozenset({'abs_gross_step_below_eur', 'consecutive_cycles'})
REPLAY_REFERENCE_KEYS = frozenset({'per_cycle_record', 'sha256', 'n_cycles'})
CERTIFICATION_DISABLED_THRESHOLD = 10 ** 9   # > any cap: the certificate can never complete
EARLY_STOP_THRESHOLD = 0                     # consecutive_converged_cycles >= 0 always: the loop exits this cycle
AA_OFF_ACTION = 'off (all channels within Boyd tolerance)'   # asserted against production's literal by the checks
# The ONLY three reads of the certificate length in `_run_operational_planning` (asserted before any solve): the loop's
# exit test, the per-cycle diagnostics row's `required_consecutive_cycles` field (a RECORD of the value in force -- it
# reads CERTIFICATION_DISABLED_THRESHOLD in this run, and EARLY_STOP_THRESHOLD on an early-stop cycle), and the cycle's
# INFO print. None of them feeds a computed quantity.
CERTIFICATE_LENGTH_READS = (
    'convergence = (consecutive_converged_cycles >= admm_parameters.minimum_consecutive_converged_cycles)',
    "'required_consecutive_cycles': admm_parameters.minimum_consecutive_converged_cycles,",
    "f'{admm_parameters.minimum_consecutive_converged_cycles} | '",
)
CYCLE_FILE = 'continuation_cycle_record.jsonl'
BLOCKS_FILE = 'recourse_blocks_all.jsonl'
SUMMARY_KEY = 'certification_continuation'
CHANNELS = ('v', 'pf', 'ess')
WRAPPED = ('_capture_convergence_depth_tail_baseline', '_apply_convergence_depth_tail',
           '_anderson_acceleration_cycle_step', '_convergence_depth_tail_next_state', '_update_admm_penalties',
           '_get_operational_recourse_components')
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- stage 1 (x = 0) of Addendum 51: the declaration the W98 launcher freezes (one source for launcher and checks) ----
STAGE1_N = 72                      # the recorded certification cycle of the certified x = 0 cell
STAGE1_CONTINUATION_CYCLES = 30    # Addendum 51: "continue 30 further cycles" -> cap N + 30 = 102
STAGE1_EARLY_STOP = {'abs_gross_step_below_eur': 500.0, 'consecutive_cycles': 3}
STAGE1_REPLAY_REFERENCE = {
    'per_cycle_record': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair',
                                     'evals', 'f6e9cd53fdbb8ee8_x0', 'per_cycle_record.jsonl'),
    'sha256': '3f0908014805cbd2e0af558cdca2d1cc07842efc55202db727539c0f9812fe80',
    'n_cycles': STAGE1_N,
}


def stage1_declaration():
    return validate_certification_continuation({
        'label': LABEL, 'hold_after_cycle': STAGE1_N, 'continuation_cycles': STAGE1_CONTINUATION_CYCLES,
        'early_stop': dict(STAGE1_EARLY_STOP), 'record_all_blocks': True,
        'replay_reference': dict(STAGE1_REPLAY_REFERENCE)})


def _is_pos_int(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 1


def validate_certification_continuation(value):
    """None = not declared (nothing is installed, nothing changes). Otherwise EXACTLY `DECLARATION_KEYS`:
    label == LABEL; hold_after_cycle N >= 1 (int); continuation_cycles >= 1 (int); early_stop =
    {abs_gross_step_below_eur: positive finite float, consecutive_cycles: int >= 1}; record_all_blocks: bool;
    replay_reference = {per_cycle_record: repo-relative path, sha256: 64 hex, n_cycles: int == N}. Returns a new dict
    (canonical: floats stay floats). Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != DECLARATION_KEYS:
        raise ValueError(f'certification_continuation must have exactly {sorted(DECLARATION_KEYS)}; got {value!r}')
    if value['label'] != LABEL:
        raise ValueError(f'certification_continuation.label must be {LABEL!r}')
    n, c = value['hold_after_cycle'], value['continuation_cycles']
    if not (_is_pos_int(n) and _is_pos_int(c)):
        raise ValueError(f'hold_after_cycle / continuation_cycles must be ints >= 1; got {n!r} / {c!r}')
    es = value['early_stop']
    if not isinstance(es, dict) or set(es) != EARLY_STOP_KEYS:
        raise ValueError(f'early_stop must have exactly {sorted(EARLY_STOP_KEYS)}; got {es!r}')
    thr, k = es['abs_gross_step_below_eur'], es['consecutive_cycles']
    if not isinstance(thr, float) or not math.isfinite(thr) or not thr > 0.0 or not _is_pos_int(k):
        raise ValueError(f'early_stop values invalid: {es!r}')
    if not isinstance(value['record_all_blocks'], bool):
        raise ValueError('record_all_blocks must be a bool')
    ref = value['replay_reference']
    if not isinstance(ref, dict) or set(ref) != REPLAY_REFERENCE_KEYS:
        raise ValueError(f'replay_reference must have exactly {sorted(REPLAY_REFERENCE_KEYS)}; got {ref!r}')
    sha = ref['sha256']
    if not (isinstance(sha, str) and len(sha) == 64 and all(ch in '0123456789abcdef' for ch in sha)):
        raise ValueError('replay_reference.sha256 must be 64 lowercase hex')
    if not isinstance(ref['per_cycle_record'], str) or os.path.isabs(ref['per_cycle_record']):
        raise ValueError('replay_reference.per_cycle_record must be a repo-relative path')
    if ref['n_cycles'] != n:
        raise ValueError('replay_reference.n_cycles must equal hold_after_cycle')
    return {'label': LABEL, 'hold_after_cycle': n, 'continuation_cycles': c,
            'early_stop': {'abs_gross_step_below_eur': thr, 'consecutive_cycles': k},
            'record_all_blocks': value['record_all_blocks'],
            'replay_reference': {'per_cycle_record': ref['per_cycle_record'], 'sha256': sha, 'n_cycles': ref['n_cycles']}}


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_replay_reference(declaration):
    """{cycle: gross hex} from the certified cell's committed per_cycle_record.jsonl; refuses unless it hashes to the
    declaration and holds exactly cycles 1..N."""
    ref = declaration['replay_reference']
    path = os.path.join(REPO, ref['per_cycle_record'])
    got = _sha256(path)
    if got != ref['sha256']:
        raise RuntimeError(f'replay reference {ref["per_cycle_record"]} sha256 {got} != declared {ref["sha256"]}')
    out = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                g = r.get('gross_operational_cost')
                out[int(r['cycle'])] = float.hex(g) if isinstance(g, float) else None
    if sorted(out) != list(range(1, ref['n_cycles'] + 1)):
        raise RuntimeError(f'replay reference cycles {sorted(out)[:3]}..{sorted(out)[-3:]} != 1..{ref["n_cycles"]}')
    return out


def assert_continuation_preconditions(declaration, spec, tail_checklist, aa_on):
    """Rule eleven for the continuation, BEFORE any solve (child): the cap equals N + continuation cycles; the tail is
    enabled for this run (the cycle tracker and the tail hold ride on its per-cycle calls); AA is enabled (the AA hold
    acts on its step); every wrapped name exists in production with the expected signature; the AA 'off' literal is
    production's; the replay reference hashes to its declaration. Raises on any failure; returns the checklist."""
    import inspect
    import shared_resources_planning as srp
    import admm_anderson_acceleration as aam
    cap = int(spec['cap'])
    params = {
        '_capture_convergence_depth_tail_baseline': ['planning_problem', 'admm_parameters'],
        '_apply_convergence_depth_tail': ['planning_problem', 'admm_parameters', 'active', 'baseline', 'cycle'],
        '_anderson_acceleration_cycle_step': ['aa_state', 'aa_layout', 'consensus_vars', 'dual_vars', 'w_before',
                                              'rho_channel', 'boyd_metrics', 'iter'],
        '_convergence_depth_tail_next_state': ['cycle_convergence', 'aa_enabled', 'aa_record'],
        '_update_admm_penalties': ['tso_model', 'dso_models', 'esso_model', 'residual_metrics', 'boyd_metrics',
                                   'params', 'iter', 'allow_update', 'freeze_state'],
        '_get_operational_recourse_components': ['planning_problem', 'models'],
    }
    loop_src = inspect.getsource(srp._run_operational_planning)
    step_src = inspect.getsource(aam.AndersonAccelerationState.step)
    checks = {
        'cap_equals_N_plus_continuation': cap == declaration['hold_after_cycle'] + declaration['continuation_cycles'],
        'tail_enabled_for_this_run': bool((tail_checklist or {}).get('tail_enabled_for_this_run')),
        'anderson_acceleration_on': bool(aa_on),
        'aa_off_literal_is_production': (srp.CONVERGENCE_DEPTH_TAIL_AA_OFF_ACTION == AA_OFF_ACTION
                                         and repr(AA_OFF_ACTION)[1:-1] in step_src),
        'certificate_length_read_only_by_the_loop_test_the_record_and_the_print': (
            sorted(line.strip() for line in loop_src.splitlines() if 'minimum_consecutive_converged_cycles' in line)
            == sorted(CERTIFICATE_LENGTH_READS)),
        'loop_exit_is_the_convergence_break': 'if convergence:\n            print(f"[INFO] \\t - ADMM converged' in loop_src,
        'block_functions_callable': all(callable(getattr(srp, n, None)) for n in (
            '_get_operational_recourse_block_components', '_get_operational_objective_component_blocks')),
    }
    for name, expected in params.items():
        fn = getattr(srp, name, None)
        checks[f'signature:{name}'] = (callable(fn)
                                       and list(inspect.signature(fn).parameters) == expected)
    try:
        load_replay_reference(declaration)
        checks['replay_reference_hashes_to_declaration'] = True
    except Exception:  # noqa: BLE001 -- recorded as a failing check, raised below
        checks['replay_reference_hashes_to_declaration'] = False
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W98 continuation preconditions fail (before any solve): {failing}')
    return checks


class ContinuationState:
    """All the per-run bookkeeping. `n` = hold_after_cycle."""

    def __init__(self, declaration, eval_dir, cap, reference=None, sink=None):
        self.decl = declaration
        self.n = declaration['hold_after_cycle']
        self.cap = cap
        self.thr = declaration['early_stop']['abs_gross_step_below_eur']
        self.k_required = declaration['early_stop']['consecutive_cycles']
        self.eval_dir = eval_dir
        self.reference = reference or {}
        self.sink = sink                 # checks: a list collecting lines instead of files
        self.phase = 'before'            # before -> in_cycle -> ended
        self.cycle = None
        self.params = None
        self.threshold_original = None
        self.threshold_restored = None
        self.gross = {}
        self.prev_blocks = None
        self.prev_blocks_cycle = None
        self.streak = 0
        self.early_stop_cycle = None
        self.cur = None
        self.lines = 0
        self.first_live_divergence = None
        self.errors = []
        self.events = []
        self.t0 = time.time()

    # ---- output ------------------------------------------------------------------------------------------------
    def _write(self, fname, obj):
        if self.sink is not None:
            self.sink.append((fname, obj))
            return
        with open(os.path.join(self.eval_dir, fname), 'a') as handle:
            handle.write(json.dumps(obj, default=str) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def summary(self):
        cont_cycles = sorted(c for c in self.gross if c > self.n)
        return {
            'schema': SCHEMA, 'declaration': self.decl, 'N': self.n, 'cap': self.cap, 'phase': self.phase,
            'certificate_length_original': self.threshold_original,
            'certificate_length_raised_to': CERTIFICATION_DISABLED_THRESHOLD,
            'certificate_length_restored_at_exit': self.threshold_restored,
            'last_cycle': self.cycle, 'cycle_lines_written': self.lines,
            'early_stop_cycle': self.early_stop_cycle,
            'stopped_by': ('early_stop' if self.early_stop_cycle is not None else
                           'cap' if self.cycle == self.cap else 'other'),
            'post_certification_cycles_captured': cont_cycles,
            'first_live_gross_divergence_vs_reference': self.first_live_divergence,
            'errors': list(self.errors),
            'ok': (not self.errors and self.phase == 'ended' and self.threshold_restored == self.threshold_original
                   and self.lines == (self.cycle or 0)),
            'files': {'cycle_record': CYCLE_FILE, 'blocks_all': BLOCKS_FILE if self.decl['record_all_blocks'] else None},
        }


def _block_rows(blocks):
    rows = []
    for (agent, node_id, year, day), value in blocks.items():
        rows.append({'agent': agent, 'node_id': node_id, 'year': None if year is None else str(year),
                     'day': None if day is None else str(day), 'value': value})
    rows.sort(key=lambda r: (str(r['agent']), str(r['node_id']), str(r['year']), str(r['day'])))
    return rows


def make_wrappers(st, originals, srp_module=None):
    """The six wrappers over `originals` (name -> callable). Separated from the context manager so the zero-solve
    checks can drive them with stand-in originals and sentinel arguments."""

    def raise_(msg):
        st.errors.append(msg)
        raise RuntimeError(f'W98 continuation hook: {msg}')

    def w_baseline(planning_problem, admm_parameters):
        out = originals['_capture_convergence_depth_tail_baseline'](planning_problem, admm_parameters)
        if st.params is not None:
            raise_('tail baseline captured twice (a second ADMM call inside one continuation run)')
        st.params = admm_parameters
        st.threshold_original = admm_parameters.minimum_consecutive_converged_cycles
        admm_parameters.minimum_consecutive_converged_cycles = CERTIFICATION_DISABLED_THRESHOLD
        st.events.append({'event': 'certificate_disabled', 'from': st.threshold_original,
                          'to': CERTIFICATION_DISABLED_THRESHOLD})
        return out

    def w_apply(planning_problem, admm_parameters, active, baseline, cycle):
        if cycle is None:   # the loop's exit restore
            out = originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, active, baseline, cycle)
            if st.phase == 'in_cycle':
                if st.params is not None:
                    st.params.minimum_consecutive_converged_cycles = st.threshold_original
                    st.threshold_restored = st.params.minimum_consecutive_converged_cycles
                st.phase = 'ended'
                st.events.append({'event': 'loop_exit', 'last_cycle': st.cycle, 'restored_to': st.threshold_restored})
            return out
        if st.phase == 'ended':
            raise_(f'a cycle ({cycle}) after the loop exit')
        expected = (st.cycle or 0) + 1
        if cycle != expected:
            raise_(f'cycle tracker: got cycle {cycle!r}, expected {expected}')
        st.cycle = cycle
        st.phase = 'in_cycle'
        st.cur = {'cycle': cycle, 'phase': 'replay' if cycle <= st.n else 'continuation', 't_start_s': time.time() - st.t0}
        if cycle <= st.n:
            st.cur['tail_apply'] = {'hold': False, 'active_passed': bool(active)}
            return originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, active, baseline, cycle)
        st.cur['tail_apply'] = {'hold': True, 'natural_active': bool(active), 'active_passed': True,
                                'hold_changed_value': not bool(active)}
        return originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, True, baseline, cycle)

    def w_aa(aa_state, aa_layout, consensus_vars, dual_vars, w_before, rho_channel, boyd_metrics, iter):
        if iter != st.cycle:
            raise_(f'AA step at iter {iter!r} but the tracker is at cycle {st.cycle!r}')
        if iter <= st.n:
            rec = originals['_anderson_acceleration_cycle_step'](aa_state, aa_layout, consensus_vars, dual_vars,
                                                                 w_before, rho_channel, boyd_metrics, iter)
            st.cur['aa'] = {'hold': False, 'action': (rec or {}).get('action')}
            return rec
        forced = dict(boyd_metrics)
        forced['all_boyd_pass'] = True
        rec = originals['_anderson_acceleration_cycle_step'](aa_state, aa_layout, consensus_vars, dual_vars,
                                                             w_before, rho_channel, forced, iter)
        st.cur['aa'] = {'hold': True, 'natural_all_boyd_pass': bool(boyd_metrics.get('all_boyd_pass')),
                        'action': (rec or {}).get('action'),
                        'hold_changed_value': not bool(boyd_metrics.get('all_boyd_pass'))}
        if (rec or {}).get('action') != AA_OFF_ACTION:
            raise_(f"AA hold at cycle {iter}: production's step returned {(rec or {}).get('action')!r}, not off")
        return rec

    def w_next(cycle_convergence, aa_enabled, aa_record):
        c = st.cycle
        if aa_record is not None and aa_record.get('cycle') is not None and aa_record.get('cycle') != c:
            raise_(f"tail next-state: AA record cycle {aa_record.get('cycle')!r} != tracker {c!r}")
        if c is None or c <= st.n:
            out = originals['_convergence_depth_tail_next_state'](cycle_convergence, aa_enabled, aa_record)
            if st.cur is not None:
                st.cur['tail_next'] = {'hold': False, 'value': bool(out), 'cycle_convergence': bool(cycle_convergence)}
            return out
        st.cur['tail_next'] = {'hold': True, 'natural_cycle_convergence': bool(cycle_convergence),
                               'aa_action': (aa_record or {}).get('action'), 'returned': True,
                               'hold_changed_value': not bool(cycle_convergence)}
        return True

    def w_penalties(tso_model, dso_models, esso_model, residual_metrics, boyd_metrics, params, iter=None,
                    allow_update=True, freeze_state=None):
        if iter != st.cycle:
            raise_(f'penalty update at iter {iter!r} but the tracker is at cycle {st.cycle!r}')
        hold = iter > st.n
        result = originals['_update_admm_penalties'](tso_model, dso_models, esso_model, residual_metrics, boyd_metrics,
                                                     params, iter=iter,
                                                     allow_update=(False if hold else allow_update),
                                                     freeze_state=freeze_state)
        actions, before, after, bg, ag, rfa, fs = result
        rec = {'hold': hold, 'allow_update_natural': bool(allow_update),
               'allow_update_passed': False if hold else bool(allow_update),
               'rho_before': dict(before), 'rho_after': dict(after), 'gamma_before': dict(bg), 'gamma_after': dict(ag),
               'actions': dict(actions), 'rho_freeze_active': bool(rfa),
               'frozen': {g: bool((fs or {}).get(g, {}).get('frozen')) for g in CHANNELS}}
        if hold:
            changed = [g for g in CHANNELS if before[g] != after[g] or bg[g] != ag[g]]
            rec['changed_channels'] = changed
            if changed:
                st.cur['rho'] = rec
                raise_(f'rho hold at cycle {iter}: rho / gamma changed on {changed}')
        st.cur['rho'] = rec
        _finalize_cycle()
        return result

    def w_recourse(planning_problem, models):
        rc = originals['_get_operational_recourse_components'](planning_problem, models)
        if st.phase != 'in_cycle' or st.cur is None or 'gross' in st.cur:
            return rc          # outside the loop (initialisation identity, terminal captures), or a second call
        c = st.cycle
        gross = rc['gross_operational_cost']
        st.cur['gross'] = gross
        st.cur['gross_hex'] = float.hex(gross) if isinstance(gross, float) else None
        st.cur['net_operational_recourse'] = rc.get('net_operational_recourse')
        st.cur['terminal_salvage_value'] = rc.get('terminal_salvage_value')
        prev = st.gross.get(c - 1)
        step = (gross - prev) if (prev is not None and gross is not None) else None
        st.cur['step'] = step
        st.gross[c] = gross
        if st.decl['record_all_blocks'] and srp_module is not None:
            _capture_blocks(planning_problem, models, c, rc)
        if c > st.n:
            qualifies = step is not None and abs(step) < st.thr
            st.streak = (st.streak + 1) if qualifies else 0
            st.cur['early_stop'] = {'step_qualifies': qualifies, 'streak': st.streak, 'required': st.k_required,
                                    'threshold_eur': st.thr}
            if st.streak >= st.k_required and st.early_stop_cycle is None:
                st.early_stop_cycle = c
                st.params.minimum_consecutive_converged_cycles = EARLY_STOP_THRESHOLD
                st.cur['early_stop']['fired'] = True
                st.events.append({'event': 'early_stop', 'cycle': c, 'streak': st.streak})
        return rc

    def _capture_blocks(planning_problem, models, c, rc):
        t = time.time()
        blocks = srp_module._get_operational_recourse_block_components(planning_problem, models)
        obj = srp_module._get_operational_objective_component_blocks(planning_problem, models)
        net = rc.get('net_operational_recourse')
        total = sum(blocks.values())
        tol = max(1e-4, 1e-10 * max(abs(net), 1.0)) if net is not None else None
        line = {'cycle': c, 'phase': 'replay' if c <= st.n else 'continuation',
                'gross_operational_cost': rc.get('gross_operational_cost'), 'net_operational_recourse': net,
                'terminal_salvage_value': rc.get('terminal_salvage_value'), 'n_blocks': len(blocks),
                'blocks': _block_rows(blocks), 'block_sum': total,
                'block_sum_minus_net': (total - net) if net is not None else None,
                'reconciles_to_net': (abs(total - net) <= tol) if tol is not None else None,
                'objective_component_blocks': [{'agent': a, 'node_id': nid, 'year': str(y), 'day': str(d), **comp}
                                               for (a, nid, y, d), comp in sorted(
                                                   obj.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]}
        if st.prev_blocks is not None and st.prev_blocks_cycle == c - 1:
            deltas = []
            for r in line['blocks']:
                key = (r['agent'], r['node_id'], r['year'], r['day'])
                previous = st.prev_blocks.get(key)
                deltas.append({'agent': r['agent'], 'node_id': r['node_id'], 'year': r['year'], 'day': r['day'],
                               'previous': previous, 'current': r['value'],
                               'delta': (r['value'] - previous) if previous is not None else None})
            line['deltas_vs_previous_cycle'] = deltas
        st.prev_blocks = {(r['agent'], r['node_id'], r['year'], r['day']): r['value'] for r in line['blocks']}
        st.prev_blocks_cycle = c
        line['capture_s'] = time.time() - t
        st.cur['blocks_captured'] = len(blocks)
        st._write(BLOCKS_FILE, line)

    def _finalize_cycle():
        cur = st.cur
        c = cur['cycle']
        if 'gross' not in cur:          # a failed cycle: no recourse computed; the streak resets
            cur['gross'] = None
            cur['step'] = None
            if c > st.n:
                st.streak = 0
                cur['early_stop'] = {'step_qualifies': False, 'streak': 0, 'required': st.k_required,
                                     'threshold_eur': st.thr, 'note': 'no gross this cycle (a local solve failed)'}
        ref_hex = st.reference.get(c)
        if ref_hex is not None:
            equal = cur.get('gross_hex') == ref_hex
            cur['replay_gross_equals_reference_bitwise'] = equal
            if not equal and st.first_live_divergence is None:
                st.first_live_divergence = {'cycle': c, 'gross_hex': cur.get('gross_hex'), 'reference_hex': ref_hex,
                                            'difference': (cur['gross'] - float.fromhex(ref_hex))
                                            if cur.get('gross') is not None else None}
        cur['certificate_length_in_force_at_cycle_end'] = (st.params.minimum_consecutive_converged_cycles
                                                          if st.params is not None else None)
        cur['t_end_s'] = time.time() - st.t0
        st._write(CYCLE_FILE, cur)
        st.lines += 1
        step = cur.get('step')
        print(f"[W98-CONTINUATION] cycle {c} ({cur['phase']}) gross={cur.get('gross')!r} step="
              f"{step if step is None else round(step, 3)} "
              f"replay_equal={cur.get('replay_gross_equals_reference_bitwise')} "
              f"holds aa={(cur.get('aa') or {}).get('hold')} tail={(cur.get('tail_apply') or {}).get('hold')} "
              f"rho={(cur.get('rho') or {}).get('hold')} streak={(cur.get('early_stop') or {}).get('streak')} "
              f"early_stop_fired={(cur.get('early_stop') or {}).get('fired', False)}", flush=True)

    return {'_capture_convergence_depth_tail_baseline': w_baseline, '_apply_convergence_depth_tail': w_apply,
            '_anderson_acceleration_cycle_step': w_aa, '_convergence_depth_tail_next_state': w_next,
            '_update_admm_penalties': w_penalties, '_get_operational_recourse_components': w_recourse}


@contextmanager
def continuation_hooks(eval_dir, declaration, holder, cap):
    """Install the wrappers for the run (the harness enters this FIRST, so these wrap production directly and every
    harness capture hook -- the tail appender, the s39 penalty sidecar -- wraps them and records the held values).
    Restores every production function on exit, even on error; `holder[SUMMARY_KEY]` gets the summary."""
    import shared_resources_planning as srp
    declaration = validate_certification_continuation(declaration)
    for fname in (CYCLE_FILE, BLOCKS_FILE):
        if os.path.exists(os.path.join(eval_dir, fname)):
            raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ContinuationState(declaration, eval_dir, int(cap), reference=load_replay_reference(declaration))
    originals = {name: getattr(srp, name) for name in WRAPPED}
    wrappers = make_wrappers(st, originals, srp_module=srp)
    for name, fn in wrappers.items():
        setattr(srp, name, fn)
    try:
        yield st
    finally:
        for name, fn in originals.items():
            setattr(srp, name, fn)
        holder[SUMMARY_KEY] = st.summary()
