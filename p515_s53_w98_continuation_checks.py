"""
P5.15 Addendum 51, Planner task W98 -- ZERO-SOLVE checks for the certification continuation (spec v37, stage 1, x = 0).

Every check here runs without a solve and without building a network model (tiny stand-in Pyomo Params for the rho test
only); a `SolveProfileGuard(permitted=())` is armed for the whole run and verified at exactly 0.

WHAT IS PROVED (each hold in BOTH directions -- a hold that fires early breaks the replay; one that never fires lets the
regime drift):
  C1  AA hold, stand-in: for cycle <= N the wrapper calls production with the SAME argument objects (identity) and
      returns production's record object; for cycle > N production receives a COPY of boyd_metrics with
      all_boyd_pass True (the caller's dict unchanged).
  C2  AA hold, REAL production `_anderson_acceleration_cycle_step` + `AndersonAccelerationState` (memory 5,
      keep_memory; collect_w / write_back_w pointed at a numpy store for the test only): wrapped and unwrapped runs on
      identical inputs are BITWISE identical (records and iterates) for cycles 1..N; for cycle > N the unwrapped run
      extrapolates ('accepted') and the wrapped one returns production's 'off' action with no write-back.
  C3  tail hold, REAL production `_capture_convergence_depth_tail_baseline` / `_apply_convergence_depth_tail` /
      `_convergence_depth_tail_next_state` on stand-in solver-option holders: for cycle <= N the option state and the
      returned records / values equal the unwrapped calls (including production's raise on an AA / predicate
      disagreement); for cycle > N the tail is applied ON although the predicate is False, and next-state returns True.
  C4  rho hold, REAL production `_update_admm_penalties` on stand-in Pyomo Params (TSO / DSO / ESSO rho and TSO gamma)
      with the case file's own ADMM parameters: wrapped and unwrapped runs equal bitwise (rho, gamma, actions,
      freeze state) for cycles 1..N; for cycle > N the unwrapped run changes rho and the wrapped one does not.
  C5  certification rule + early stop, stand-ins: the certificate length is 10**9 from the tail baseline on; no
      early-stop action for cycle <= N however small the steps; for cycle > N it fires on the 3rd consecutive
      |dQ| < 500 (a large step and a failed cycle reset the streak); the length is restored to 10 at the loop exit.
  C6  layering, REAL install: `continuation_hooks` entered first and the harness's `convergence_depth_append_hooks`
      on top (the order `_child_real` uses) -> the appender records the HELD tail value; every production function is
      restored on exit; `_child_real` enters `continuation_cm` first (source).
  C7  keys: for EVERY entry of every committed campaign spec the modified harness's key (no declaration) equals the
      pre-W98 harness's (252996a1, loaded from git) on the same arguments; the certified pair's keys reproduce
      byte-identically; the continuation key follows its declared formula, differs from the certified cell's key and
      appears in no committed campaign spec.
  C8  preconditions (`assert_continuation_preconditions`) hold for the stage-1 spec shape and refuse three negative
      controls; the declaration validator refuses malformed declarations.

Run (repo root, canonical interpreter, attached, both streams captured):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w98_continuation_checks.py \\
      > data/SRP1/Results/P515S53/w98_continuation/zero_solve_checks_launch.log 2>&1
"""
import contextlib
import copy
import hashlib
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W98 continuation zero-solve checks (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402
import p515_s53_w98_continuation_hooks as C  # noqa: E402

OUT_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w98_continuation', 'zero_solve_checks')
OUT_FILE = 'w98_zero_solve_checks.json'
OUT_MANIFEST = 'w98_zero_solve_checks_manifest_sha256.json'
PAIR_KEYS = {'x0': 'f6e9cd53fdbb8ee8c80388a13d723c49de5b18173bcfa5c7351cbe0d9e5000d4',
             'n7_4h_e1': 'c82522f470b35b58399bfe47ebd45612a95641ae308128de5d1c3c08ad5408a1'}
PAIR_SPEC_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair',
                             'campaign_spec_s53_w91_3x3_pair_231558f0.json')


def _utc():
    return datetime.now(timezone.utc).isoformat()


class _Sentinel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<sentinel {self.name}>'

    def __bool__(self):   # the wrappers record bool(); a sentinel standing for a True flag
        return True


def _fresh_state(n=C.STAGE1_N, cap=None, reference=None):
    decl = C.stage1_declaration()
    if n != decl['hold_after_cycle']:
        decl = dict(decl)
        decl['hold_after_cycle'] = n
        decl['replay_reference'] = dict(decl['replay_reference'], n_cycles=n)
    sink = []
    st = C.ContinuationState(decl, eval_dir=None, cap=cap or (n + decl['continuation_cycles']),
                             reference=reference or {}, sink=sink)
    return st, sink


# ======================================================================================================================
#  stand-in originals (record every call)
# ======================================================================================================================
class Recorder:
    def __init__(self):
        self.calls = {}

    def log(self, name, *args, **kwargs):
        self.calls.setdefault(name, []).append((args, kwargs))


def standin_originals(rec, aa_action_natural='accepted', penalty_change=True):
    ret = {}

    def baseline(pp, admm):
        rec.log('baseline', pp, admm)
        ret['baseline'] = _Sentinel('baseline-return')
        return ret['baseline']

    def apply(pp, admm, active, baseline_state, cycle):
        rec.log('apply', pp, admm, active, baseline_state, cycle)
        out = _Sentinel(f'apply-return-{cycle}')
        ret.setdefault('apply', []).append(out)
        return out

    def aa(aa_state, aa_layout, cv, dv, w_before, rho, boyd, it):
        rec.log('aa', aa_state, aa_layout, cv, dv, w_before, rho, boyd, it)
        out = {'cycle': it, 'action': C.AA_OFF_ACTION if boyd['all_boyd_pass'] else aa_action_natural}
        ret.setdefault('aa', []).append(out)
        return out

    def nxt(cycle_convergence, aa_enabled, aa_record):
        rec.log('next', cycle_convergence, aa_enabled, aa_record)
        out = _Sentinel('next-return')
        ret.setdefault('next', []).append(out)
        return out

    def pen(tso, dso, esso, rm, bm, params, iter=None, allow_update=True, freeze_state=None):
        rec.log('pen', tso, dso, esso, rm, bm, params, iter=iter, allow_update=allow_update, freeze_state=freeze_state)
        before = {'v': 1.0, 'pf': 2.0, 'ess': 3.0}
        after = ({'v': 1.5, 'pf': 2.0, 'ess': 3.0} if (penalty_change and bool(allow_update)) else dict(before))
        out = ({'v': 'x', 'pf': 'x', 'ess': 'x'}, before, after, {'v': 0.0, 'pf': 0.0, 'ess': 0.0},
               {'v': 0.0, 'pf': 0.0, 'ess': 0.0}, False, {g: {'frozen': False} for g in C.CHANNELS})
        ret.setdefault('pen', []).append(out)
        return out

    def recourse(pp, models):
        rec.log('recourse', pp, models)
        out = dict(ret['gross_script'].pop(0))
        ret.setdefault('recourse', []).append(out)
        return out

    ret['gross_script'] = []
    return {'_capture_convergence_depth_tail_baseline': baseline, '_apply_convergence_depth_tail': apply,
            '_anderson_acceleration_cycle_step': aa, '_convergence_depth_tail_next_state': nxt,
            '_update_admm_penalties': pen, '_get_operational_recourse_components': recourse}, ret


def _drive_cycle(w, st, c, gross, *, active, boyd_pass, cycle_convergence, allow_update, failed=False, sentinels=None):
    """One emulated production cycle through the wrappers, in production's call order."""
    s = sentinels if sentinels is not None else {}
    pp, admm = s.get('pp', object()), s.get('admm')
    w['_apply_convergence_depth_tail'](pp, admm, active, s.get('baseline_state', object()), c)
    aa_rec = None
    if not failed:
        boyd = {'all_boyd_pass': boyd_pass, 'v': {'r': 1.0, 's': 1.0}}
        aa_rec = w['_anderson_acceleration_cycle_step'](s.get('aa_state', object()), None, {}, {}, None, {}, boyd, c)
        w['_get_operational_recourse_components'](pp, {'models': c})
    else:
        aa_rec = {'cycle': c, 'action': 'skipped (local solve failure this cycle)'}
    w['_convergence_depth_tail_next_state'](cycle_convergence, True, aa_rec)
    w['_update_admm_penalties']({}, {}, {}, {}, {}, s.get('params', object()), iter=c, allow_update=allow_update,
                                freeze_state={})


# ======================================================================================================================
#  C1 -- AA hold with a stand-in (object identity)
# ======================================================================================================================
def check_c1_aa_standin():
    n = C.STAGE1_N
    st, _sink = _fresh_state()
    rec = Recorder()
    orig, ret = standin_originals(rec)
    w = C.make_wrappers(st, orig)
    admm = SimpleNamespace(minimum_consecutive_converged_cycles=10)
    w['_capture_convergence_depth_tail_baseline'](object(), admm)
    results = {}
    for c in range(1, n + 3):
        st.cycle, st.phase, st.cur = c, 'in_cycle', {'cycle': c}
        aa_state, cv, dv, wb, rho = object(), {}, {}, object(), {}
        boyd = {'all_boyd_pass': False, 'v': {'r': 1.0, 's': 1.0}}
        boyd_copy = copy.deepcopy(boyd)
        out = w['_anderson_acceleration_cycle_step'](aa_state, 'layout', cv, dv, wb, rho, boyd, c)
        args, _kw = rec.calls['aa'][-1]
        same_objects = (args[0] is aa_state and args[1] == 'layout' and args[2] is cv and args[3] is dv
                        and args[4] is wb and args[5] is rho and args[7] == c)
        if c <= n:
            results[c] = {'inert': same_objects and args[6] is boyd and out is ret['aa'][-1],
                          'action': out['action']}
        else:
            results[c] = {'acts': (same_objects and args[6] is not boyd and args[6]['all_boyd_pass'] is True
                                   and boyd == boyd_copy and out['action'] == C.AA_OFF_ACTION
                                   and st.cur['aa']['natural_all_boyd_pass'] is False),
                          'action': out['action']}
    inert = all(results[c]['inert'] for c in range(1, n + 1))
    acts = all(results[c]['acts'] for c in range(n + 1, n + 3))
    return {'id': 'C1_aa_hold_standin', 'inert_through_N': inert, 'acts_after_N': acts,
            'boundary': {str(n): results[n], str(n + 1): results[n + 1]},
            'natural_action_at_boundary_plus_1': 'accepted (stand-in, all_boyd_pass False)',
            'holds': inert and acts}


# ======================================================================================================================
#  C2 -- AA hold with REAL production functions
# ======================================================================================================================
def check_c2_aa_real():
    import numpy as np
    import admm_anderson_acceleration as aam
    import shared_resources_planning as srp
    n = C.STAGE1_N
    dim = 6
    rng = np.random.default_rng(20260927)
    A = 0.9 * np.eye(dim) + 0.02 * rng.standard_normal((dim, dim))
    b = rng.standard_normal(dim)
    real_collect, real_write = aam.collect_w, aam.write_back_w

    def collect(layout, cv, dv, rho, check_antisymmetry=True):
        return np.array(cv['w'], dtype=float)

    def write_back(layout, w, cv, dv, rho):
        cv['w'] = np.array(w, dtype=float)

    def run(wrapped):
        st, _sink = _fresh_state()
        orig = {'_anderson_acceleration_cycle_step': srp._anderson_acceleration_cycle_step}
        w = C.make_wrappers(st, dict(orig, **{k: None for k in C.WRAPPED if k not in orig}))
        step = w['_anderson_acceleration_cycle_step'] if wrapped else srp._anderson_acceleration_cycle_step
        state = aam.AndersonAccelerationState(memory=5, regularization=1e-10, reject_policy='keep_memory')
        cv = {'w': np.zeros(dim)}
        trace = []
        for c in range(1, n + 5):
            st.cycle, st.phase, st.cur = c, 'in_cycle', {'cycle': c}
            w_before = np.array(cv['w'])
            cv['w'] = A @ cv['w'] + b * (1.0 + 0.1 * np.sin(c))         # the "plain cycle" (non-stationary)
            r = 1.0 / c                                                  # strictly decreasing -> AA accepts
            boyd = {'all_boyd_pass': False, 'v': {'r': r, 's': r}, 'pf': {'r': r, 's': r}, 'ess': {'r': r, 's': r}}
            rec = step(state, None, cv, {}, w_before, {'v': 1.0, 'pf': 1.0, 'ess': 1.0}, boyd, c)
            trace.append({'cycle': c, 'action': rec['action'], 'w_hex': [float.hex(float(x)) for x in cv['w']],
                          'boyd_all_pass_seen_by_caller': boyd['all_boyd_pass']})
        return trace

    aam.collect_w, aam.write_back_w = collect, write_back
    try:
        ref = run(False)
        wrp = run(True)
    finally:
        aam.collect_w, aam.write_back_w = real_collect, real_write
    restored = aam.collect_w is real_collect and aam.write_back_w is real_write
    inert = all(ref[i] == wrp[i] for i in range(n))
    after = [(ref[i]['action'], wrp[i]['action'], ref[i]['w_hex'] != wrp[i]['w_hex']) for i in range(n, n + 4)]
    acts = (all(a_ref == 'accepted' and a_w == C.AA_OFF_ACTION for a_ref, a_w, _d in after)
            and all(d for _a, _b, d in after) and all(not t['boyd_all_pass_seen_by_caller'] for t in wrp))
    n_accepted_through_n = sum(1 for t in ref[:n] if t['action'] == 'accepted')
    return {'id': 'C2_aa_hold_real_production', 'inert_through_N_bitwise': inert, 'acts_after_N': acts,
            'n_accepted_extrapolations_through_N_in_both_runs': n_accepted_through_n,
            'after_N_ref_vs_wrapped': [{'cycle': n + 1 + i, 'unwrapped_action': a, 'wrapped_action': b_,
                                        'iterates_differ': d} for i, (a, b_, d) in enumerate(after)],
            'aam_functions_restored': restored, 'holds': inert and acts and restored and n_accepted_through_n > 0}


# ======================================================================================================================
#  C3 -- tail hold with REAL production functions
# ======================================================================================================================
def _fake_holders():
    def mk(name):
        return SimpleNamespace(name=name, params=SimpleNamespace(solver_params=SimpleNamespace(
            options={'compl_inf_tol': 1e-4, 'tol': 1e-5}, recovery_options={})))
    return SimpleNamespace(transmission_network=mk('case9'),
                           distribution_networks={5: mk('case33_1'), 7: mk('case33_2'), 9: mk('case33_3')})


def _holder_options(pp):
    out = {'TSO': dict(pp.transmission_network.params.solver_params.options)}
    for k, d in pp.distribution_networks.items():
        out[f'DSO{k}'] = dict(d.params.solver_params.options)
    return out


def check_c3_tail_real():
    import shared_resources_planning as srp
    n = C.STAGE1_N
    names = ('_capture_convergence_depth_tail_baseline', '_apply_convergence_depth_tail',
             '_convergence_depth_tail_next_state')
    real = {k: getattr(srp, k) for k in names}

    def run(wrapped, predicate):
        st, _sink = _fresh_state()
        w = C.make_wrappers(st, dict(real, **{k: None for k in C.WRAPPED if k not in real}))
        fn = w if wrapped else real
        pp = _fake_holders()
        admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                               minimum_consecutive_converged_cycles=10)
        base = fn['_capture_convergence_depth_tail_baseline'](pp, admm)
        active_next = False
        trace = []
        for c in range(1, n + 5):
            # wrapped: the wrapper's own tracker advances on this call (it refuses a non-consecutive cycle)
            rec = fn['_apply_convergence_depth_tail'](pp, admm, active_next, base, c)
            options_after_apply = _holder_options(pp)
            conv = predicate(c)
            aa_rec = {'cycle': c, 'action': C.AA_OFF_ACTION if conv else 'accepted'}
            active_next = fn['_convergence_depth_tail_next_state'](conv, True, aa_rec)
            trace.append({'cycle': c, 'apply_record': rec, 'options_after_apply': options_after_apply,
                          'next': bool(active_next), 'predicate': conv})
        return trace, admm

    # the recorded x = 0 pattern: predicate True from cycle 63; then False after N (the case the hold must cover)
    def predicate(c):
        return 63 <= c <= n

    ref, _admm_ref = run(False, predicate)
    wrp, admm_w = run(True, predicate)
    inert = all(ref[i] == wrp[i] for i in range(n))
    after = wrp[n:]
    acts = (all(t['apply_record']['active'] is True for t in after)
            and all(set(v['compl_inf_tol'] for v in t['options_after_apply'].values()) == {1e-6} for t in after)
            and all(t['next'] is True for t in after)
            and any(r['apply_record']['active'] is False for r in ref[n + 1:]))
    # production's own raise on a disagreement is preserved for cycle <= N
    st, _sink = _fresh_state()
    w = C.make_wrappers(st, dict(real, **{k: None for k in C.WRAPPED if k not in real}))
    st.cycle, st.phase, st.cur = n, 'in_cycle', {'cycle': n}
    raised_wrapped = raised_real = None
    try:
        w['_convergence_depth_tail_next_state'](False, True, {'cycle': n, 'action': C.AA_OFF_ACTION})
    except RuntimeError as error:
        raised_wrapped = str(error)
    try:
        real['_convergence_depth_tail_next_state'](False, True, {'cycle': n, 'action': C.AA_OFF_ACTION})
    except RuntimeError as error:
        raised_real = str(error)
    st.cycle, st.cur = n + 1, {'cycle': n + 1}
    held = w['_convergence_depth_tail_next_state'](False, True, {'cycle': n + 1, 'action': C.AA_OFF_ACTION})
    raise_preserved = raised_wrapped is not None and raised_wrapped == raised_real
    return {'id': 'C3_tail_hold_real_production', 'inert_through_N': inert, 'acts_after_N': acts,
            'unwrapped_tail_after_N': [r['apply_record']['active'] for r in ref[n:]],
            'wrapped_tail_after_N': [t['apply_record']['active'] for t in after],
            'production_raise_on_disagreement_preserved_at_N': raise_preserved,
            'held_next_state_at_N_plus_1_despite_disagreement': held is True,
            'certificate_length_after_baseline': admm_w.minimum_consecutive_converged_cycles,
            'holds': inert and acts and raise_preserved and held is True
            and admm_w.minimum_consecutive_converged_cycles == C.CERTIFICATION_DISABLED_THRESHOLD}


# ======================================================================================================================
#  C4 -- rho hold with REAL production `_update_admm_penalties`
# ======================================================================================================================
def _rho_models():
    import pyomo.environ as pe

    def blk(kind):
        m = pe.ConcreteModel()
        if kind == 'esso':
            m.rho = pe.Param(initialize=0.01, mutable=True)
            return m
        m.rho_v = pe.Param(initialize=0.0077, mutable=True)
        m.rho_pf = pe.Param(initialize=0.198, mutable=True)
        m.rho_ess = pe.Param(initialize=0.01, mutable=True)
        if kind == 'tso':
            m.prox_gamma_v = pe.Param(initialize=0.0, mutable=True)
            m.prox_gamma_pf = pe.Param(initialize=0.0, mutable=True)
            m.prox_gamma_ess = pe.Param(initialize=0.0, mutable=True)
        return m
    tso = {2025: {'Winter': blk('tso'), 'Summer': blk('tso')}}
    dso = {5: {2025: {'Winter': blk('dso'), 'Summer': blk('dso')}}, 7: {2025: {'Winter': blk('dso')}}}
    esso = {5: blk('esso'), 7: blk('esso')}
    return tso, dso, esso


def _rho_read(tso, dso, esso):
    import pyomo.environ as pe
    vals = []
    for y in tso.values():
        for m in y.values():
            vals += [pe.value(m.rho_v), pe.value(m.rho_pf), pe.value(m.rho_ess), pe.value(m.prox_gamma_v),
                     pe.value(m.prox_gamma_pf), pe.value(m.prox_gamma_ess)]
    for nd in dso.values():
        for y in nd.values():
            for m in y.values():
                vals += [pe.value(m.rho_v), pe.value(m.rho_pf), pe.value(m.rho_ess)]
    for m in esso.values():
        vals.append(pe.value(m.rho))
    return [float.hex(float(v)) for v in vals]


def check_c4_rho_real():
    import shared_resources_planning as srp
    from planning_parameters import PlanningParameters
    n = C.STAGE1_N
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    admm = params.admm
    real = srp._update_admm_penalties

    def metrics(c):
        # alternate the balancing direction on v / pf so rho moves every cycle and no channel ever freezes; ess dual
        # ratio below 1 so its conditional exemption lifts after 5 cycles and it then balances as well
        hi = (c % 2 == 1)
        rm = {'primal': {g: 1.0 for g in ('v', 'pf', 'ess')} | {f'{g}_mean': 1.0 for g in ('v', 'pf', 'ess')},
              'dual': {f'{g}_mean': 1.0 for g in ('v', 'pf', 'ess')}}
        bm = {g: {'primal_ratio': 50.0 if hi else 0.5, 'dual_ratio_balance': 0.5 if hi else 50.0,
                  'dual_ratio': 0.5, 'r': 1.0, 's': 1.0, 'eps_pri': 1.0, 'eps_dual': 1.0} for g in ('v', 'pf', 'ess')}
        return rm, bm

    def run(wrapped):
        st, _sink = _fresh_state()
        w = C.make_wrappers(st, dict({'_update_admm_penalties': real},
                                     **{k: None for k in C.WRAPPED if k != '_update_admm_penalties'}))
        st.params = SimpleNamespace(minimum_consecutive_converged_cycles=C.CERTIFICATION_DISABLED_THRESHOLD)
        tso, dso, esso = _rho_models()
        fs = srp._init_admm_freeze_state()
        trace = []
        with contextlib.redirect_stdout(io.StringIO()):
            for c in range(1, n + 5):
                st.cycle, st.phase, st.cur = c, 'in_cycle', {'cycle': c, 'phase': 'x'}
                rm, bm = metrics(c)
                fn = w['_update_admm_penalties'] if wrapped else real
                actions, before, after, bg, ag, rfa, fs = fn(tso, dso, esso, rm, bm, admm, iter=c, allow_update=True,
                                                             freeze_state=fs)
                trace.append({'cycle': c, 'actions': dict(actions), 'rho_hex': _rho_read(tso, dso, esso),
                              'freeze_state': copy.deepcopy(fs), 'changed': before != after})
        return trace, st

    ref, _st_ref = run(False)
    wrp, st_w = run(True)
    inert = all(ref[i] == wrp[i] for i in range(n))
    moved_through_n = sum(1 for t in ref[:n] if t['changed'])
    after_ref = [t['changed'] for t in ref[n:]]
    after_wrp = [t['changed'] for t in wrp[n:]]
    rho_frozen_after = all(wrp[i]['rho_hex'] == wrp[n - 1]['rho_hex'] for i in range(n, n + 4))
    acts = all(after_ref) and not any(after_wrp) and rho_frozen_after
    return {'id': 'C4_rho_hold_real_production', 'inert_through_N_bitwise': inert, 'acts_after_N': acts,
            'cycles_with_rho_change_through_N_both_runs': moved_through_n,
            'unwrapped_rho_changed_after_N': after_ref, 'wrapped_rho_changed_after_N': after_wrp,
            'wrapped_rho_after_N_equals_rho_at_N_bitwise': rho_frozen_after,
            'wrapped_actions_after_N': [t['actions'] for t in wrp[n:]],
            'case_file_penalty_update': admm.penalty_update, 'cycle_lines_emitted_by_wrapper': st_w.lines,
            'holds': inert and acts and moved_through_n > 0}


# ======================================================================================================================
#  C5 -- certification rule disabled + early stop (stand-ins)
# ======================================================================================================================
def check_c5_certificate_and_early_stop():
    n = C.STAGE1_N
    scenarios = {}
    for name, steps_after, fail_at in (
            ('fires_on_third_small_step', [800.0, -400.0, 300.0, -200.0, 100.0], None),
            ('failed_cycle_resets_streak', [-400.0, 300.0, None, -200.0, 100.0, 50.0, 10.0], 3),
            ('never_fires', [-4000.0, 3000.0, -600.0, 499.0, 501.0], None)):
        st, sink = _fresh_state()
        rec = Recorder()
        orig, ret = standin_originals(rec, penalty_change=False)
        w = C.make_wrappers(st, orig)
        admm = SimpleNamespace(minimum_consecutive_converged_cycles=10)
        sentinels = {'admm': admm}
        w['_capture_convergence_depth_tail_baseline'](object(), admm)
        after_baseline = admm.minimum_consecutive_converged_cycles
        q = 842832534.7623764
        in_force = []
        fired = None
        # through N: tiny steps (|dQ| = 1 EUR) -- the rule must NOT act
        for c in range(1, n + 1):
            q -= 1.0
            ret['gross_script'].append({'gross_operational_cost': q, 'net_operational_recourse': q,
                                        'terminal_salvage_value': 0.0})
            _drive_cycle(w, st, c, q, active=False, boyd_pass=True, cycle_convergence=True, allow_update=True,
                         sentinels=sentinels)
            in_force.append(admm.minimum_consecutive_converged_cycles)
        c = n
        for i, s in enumerate(steps_after):
            c += 1
            failed = (fail_at is not None and i == fail_at - 1) or s is None
            if not failed:
                q += s
                ret['gross_script'].append({'gross_operational_cost': q, 'net_operational_recourse': q,
                                            'terminal_salvage_value': 0.0})
            _drive_cycle(w, st, c, q, active=True, boyd_pass=True, cycle_convergence=True, allow_update=True,
                         failed=failed, sentinels=sentinels)
            in_force.append(admm.minimum_consecutive_converged_cycles)
            if st.early_stop_cycle is not None:
                fired = st.early_stop_cycle
                break
        w['_apply_convergence_depth_tail'](object(), admm, False, object(), None)
        scenarios[name] = {
            'certificate_length_after_baseline': after_baseline,
            'certificate_length_in_force_through_N': sorted(set(in_force[:n])),
            'early_stop_cycle': fired, 'certificate_length_at_early_stop_cycle': in_force[-1] if fired else None,
            'certificate_length_restored_at_exit': admm.minimum_consecutive_converged_cycles,
            'summary_ok': st.summary()['ok'], 'stopped_by': st.summary()['stopped_by'],
            'streaks_after_N': [line.get('early_stop', {}).get('streak') for f, line in sink
                                if f == C.CYCLE_FILE and line['cycle'] > n]}
    s1, s2, s3 = (scenarios['fires_on_third_small_step'], scenarios['failed_cycle_resets_streak'],
                  scenarios['never_fires'])
    holds = (all(s['certificate_length_after_baseline'] == C.CERTIFICATION_DISABLED_THRESHOLD
                 and s['certificate_length_in_force_through_N'] == [C.CERTIFICATION_DISABLED_THRESHOLD]
                 and s['certificate_length_restored_at_exit'] == 10 and s['summary_ok'] for s in scenarios.values())
             and s1['early_stop_cycle'] == n + 4 and s1['certificate_length_at_early_stop_cycle'] == 0
             and s1['streaks_after_N'] == [0, 1, 2, 3] and s1['stopped_by'] == 'early_stop'
             and s2['early_stop_cycle'] == n + 7 and s2['streaks_after_N'] == [1, 2, 0, 0, 1, 2, 3]
             and s3['early_stop_cycle'] is None and s3['stopped_by'] == 'other')
    return {'id': 'C5_certificate_disabled_and_early_stop', 'scenarios': scenarios,
            'rule': ('from the tail baseline (before cycle 1) the certificate length is 10**9; for cycle > N the '
                     'early stop fires on the 3rd consecutive post-certification step with |dQ| < 500 EUR (a step '
                     '>= 500 or a failed cycle resets the streak, and the step after a failed cycle is unavailable); '
                     'firing sets the length to 0 so production exits at the end of that cycle; the loop exit '
                     'restores 10'),
            'holds': holds}


# ======================================================================================================================
#  C6 -- layering with the REAL install (continuation first, the harness appender on top)
# ======================================================================================================================
class _AppenderStub:
    def __init__(self):
        self.events = []
        self.original_drain = None

    def on_drain(self, *a, **k):
        self.events.append(('drain', a, k))

    def on_tail_event(self, kind, **k):
        self.events.append((kind, k))


def check_c6_layering():
    import shared_resources_planning as srp
    n = C.STAGE1_N
    before = {name: getattr(srp, name) for name in C.WRAPPED + ('_drain_network_ipopt_solve_records',)}
    stub = _AppenderStub()
    holder = {}
    scratch = tempfile.mkdtemp(prefix='w98_layering_')
    pp = _fake_holders()
    admm = SimpleNamespace(convergence_depth_tail={'enabled': True, 'compl_inf_tol': 1e-6},
                           minimum_consecutive_converged_cycles=10)
    seen = []
    with C.continuation_hooks(scratch, C.stage1_declaration(), holder, cap=n + 30) as st:
        with H.convergence_depth_append_hooks(stub):
            base = srp._capture_convergence_depth_tail_baseline(pp, admm)
            active = False
            for c in range(1, n + 3):
                srp._apply_convergence_depth_tail(pp, admm, active, base, c)
                conv = (c <= n)
                active = srp._convergence_depth_tail_next_state(
                    conv if c <= n else False, True,
                    {'cycle': c, 'action': C.AA_OFF_ACTION if (conv or c > n) else 'accepted'})
                seen.append((c, active))
                st.cur = {'cycle': c}
            srp._apply_convergence_depth_tail(pp, admm, active, base, None)
    after = {name: getattr(srp, name) for name in before}
    restored = all(after[k] is before[k] for k in before)
    next_events = [e[1]['value'] for e in stub.events if e[0] == 'next_state']
    apply_events = [e[1] for e in stub.events if e[0] == 'apply']
    appender_saw_held = (next_events[n] is True and apply_events[n + 1]['record']['active'] is True
                         and next_events[n - 1] is True)
    child_src = inspect.getsource(H._child_real)
    with_idx = child_src.find('with continuation_cm, \\')
    first_in_with = (with_idx > 0 and child_src.find('G.s38_pf_capture_hooks(', with_idx) > with_idx
                     and child_src.find('convergence_depth_append_hooks(appender)', with_idx) > with_idx)
    shutil.rmtree(scratch)
    return {'id': 'C6_layering_real_install', 'appender_recorded_held_tail_value_after_N': appender_saw_held,
            'next_state_values_seen_by_appender_last4': next_events[-4:],
            'production_functions_restored_on_exit': restored,
            'child_real_enters_continuation_first': first_in_with,
            'certificate_length_restored_at_exit': admm.minimum_consecutive_converged_cycles,
            'summary': {k: holder[C.SUMMARY_KEY][k] for k in ('phase', 'certificate_length_original',
                                                              'certificate_length_restored_at_exit', 'errors')},
            'holds': (appender_saw_held and restored and first_in_with
                      and admm.minimum_consecutive_converged_cycles == 10
                      and holder[C.SUMMARY_KEY]['phase'] == 'ended' and not holder[C.SUMMARY_KEY]['errors'])}


# ======================================================================================================================
#  C7 -- keys
# ======================================================================================================================
def _entry_key_args(spec, e):
    cfg = spec.get('configuration') or {}
    overrides = e.get('overrides') if 'overrides' in e else (cfg.get('overrides') or {})
    return (e['key'], overrides), dict(
        case_file_aa=cfg.get('case_file_anderson_acceleration'), model_variant=e.get('model_variant'),
        ess_ageing_baseline=cfg.get('ess_ageing_baseline'), flex_price_multiplier=e.get('flex_price_multiplier'),
        derived_instance=cfg.get('derived_instance'), interface_deviation_premium=e.get('interface_deviation_premium'),
        convergence_depth_tail=cfg.get('convergence_depth_tail'))


def _recompute_entry_key(spec, e, module=None):
    args, kwargs = _entry_key_args(spec, e)
    if module is None:
        kwargs['certification_continuation'] = e.get('certification_continuation')
        return H.evaluation_key(*args, **kwargs)
    return module.evaluation_key(*args, **kwargs)


PRE_W98_HARNESS = {'commit': 'b736e4e5', 'sha256': '252996a170c30bba69e9d09778a89186683dfbaf715652415c8493abe5e1bc92'}


def _harness_pre_w98():
    """The PRE-W98 harness (the one the certified pair ran, sha256 252996a1 pinned by spec 231558f0), read from git at
    commit b736e4e5 and loaded as a separate module -- the key function every committed key was computed with."""
    import importlib.util
    import subprocess
    src = subprocess.run(['git', 'show', f"{PRE_W98_HARNESS['commit']}:p515_s44_campaign_harness.py"], cwd=REPO,
                         capture_output=True, check=True).stdout
    sha = hashlib.sha256(src).hexdigest()
    if sha != PRE_W98_HARNESS['sha256']:
        raise RuntimeError(f'pre-W98 harness sha256 {sha} != pinned {PRE_W98_HARNESS["sha256"]}')
    tmp = tempfile.mkdtemp(prefix='w98_pre_harness_')
    path = os.path.join(tmp, '_w98_pre_harness.py')
    with open(path, 'wb') as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location('_w98_pre_harness', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.rmtree(tmp)
    return mod, sha


def check_c7_keys():
    """(a) DEFAULT-OFF REGRESSION (gated): for every entry of every committed campaign spec, the modified harness's
    `evaluation_key` (no continuation declared) equals the PRE-W98 harness's (sha256 252996a1, the one the certified
    pair ran) on the same arguments; (b) the certified pair's keys reproduce byte-identically; (c) the continuation key
    follows its declared formula, differs from the certified x = 0 key and is absent from every committed spec;
    (d) REPORTED: how many frozen keys the current harness rebuilds from the spec alone (a mismatch there is a key an
    older harness rule produced, and is not a W98 effect when (a) holds)."""
    pre, pre_sha = _harness_pre_w98()
    pair = json.load(open(os.path.join(REPO, PAIR_SPEC_REL)))
    pair_now = {e['label']: _recompute_entry_key(pair, e) for e in pair['candidates']}
    pair_ok = pair_now == PAIR_KEYS
    x0 = next(e for e in pair['candidates'] if e['label'] == 'x0')
    cfg = pair['configuration']
    decl = C.stage1_declaration()
    cont_key = H.evaluation_key(x0['key'], {}, case_file_aa=cfg['case_file_anderson_acceleration'],
                                ess_ageing_baseline=cfg['ess_ageing_baseline'], derived_instance=cfg['derived_instance'],
                                interface_deviation_premium=x0['interface_deviation_premium'],
                                convergence_depth_tail=cfg['convergence_depth_tail'], certification_continuation=decl)
    expected_formula = hashlib.sha256(json.dumps({'base_evaluation_key': PAIR_KEYS['x0'],
                                                  'certification_continuation': decl},
                                                 sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    committed = {}
    n_specs = n_entries = n_old_new_equal = n_with_eval_key = n_frozen_equal = 0
    old_new_mismatch, frozen_mismatch, errors = [], [], []
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        spec = json.load(open(os.path.join(REPO, rel)))
        n_specs += 1
        for e in spec.get('candidates') or []:
            n_entries += 1
            committed.setdefault(H._entry_eval_key(e), []).append(rel)
            try:
                new, old = _recompute_entry_key(spec, e), _recompute_entry_key(spec, e, module=pre)
            except Exception as error:  # noqa: BLE001 -- recorded; both harnesses must agree on raising too
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if new == old:
                n_old_new_equal += 1
            else:
                old_new_mismatch.append({'spec': rel, 'label': e.get('label'), 'new': new[:16], 'old': old[:16]})
            if 'eval_key' in e:
                n_with_eval_key += 1
                if new == e['eval_key']:
                    n_frozen_equal += 1
                else:
                    frozen_mismatch.append({'spec': rel, 'label': e.get('label'), 'frozen': e['eval_key'][:16],
                                            'recomputed': new[:16]})
    return {'id': 'C7_keys', 'pre_w98_harness': {**PRE_W98_HARNESS, 'sha256_loaded': pre_sha},
            'default_off_regression': {'committed_specs_scanned': n_specs, 'committed_entries_scanned': n_entries,
                                       'new_equals_pre_w98': n_old_new_equal, 'mismatches': old_new_mismatch,
                                       'errors': errors},
            'pair_keys_recomputed_byte_identical': pair_ok, 'pair_keys_now': pair_now,
            'continuation_key_x0': cont_key, 'continuation_key_equals_declared_formula': cont_key == expected_formula,
            'continuation_key_differs_from_certified_key': cont_key != PAIR_KEYS['x0'],
            'continuation_key_absent_from_committed_specs': cont_key not in committed,
            'frozen_key_rebuild_reported': {'entries_with_eval_key': n_with_eval_key, 'rebuilt_equal': n_frozen_equal,
                                            'not_rebuilt_from_spec_alone': frozen_mismatch},
            'holds': (pair_ok and cont_key == expected_formula and cont_key != PAIR_KEYS['x0']
                      and cont_key not in committed and not old_new_mismatch and not errors
                      and n_old_new_equal == n_entries)}


# ======================================================================================================================
#  C8 -- preconditions and validator
# ======================================================================================================================
def check_c8_preconditions():
    decl = C.stage1_declaration()
    good = C.assert_continuation_preconditions(decl, {'cap': 102}, {'tail_enabled_for_this_run': True}, True)
    negatives = {}
    for name, (spec, tail, aa) in {'cap_500': ({'cap': 500}, {'tail_enabled_for_this_run': True}, True),
                                   'tail_off': ({'cap': 102}, {'tail_enabled_for_this_run': False}, True),
                                   'aa_off': ({'cap': 102}, {'tail_enabled_for_this_run': True}, False)}.items():
        try:
            C.assert_continuation_preconditions(decl, spec, tail, aa)
            negatives[name] = 'NOT refused'
        except RuntimeError as error:
            negatives[name] = f'refused: {error}'
    bad_decls = {'wrong_label': dict(decl, label='x'), 'extra_key': dict(decl, extra=1),
                 'int_threshold': dict(decl, early_stop={'abs_gross_step_below_eur': 500, 'consecutive_cycles': 3}),
                 'n_mismatch': dict(decl, replay_reference=dict(decl['replay_reference'], n_cycles=71))}
    refused = {}
    for name, d in bad_decls.items():
        try:
            C.validate_certification_continuation(d)
            refused[name] = False
        except ValueError:
            refused[name] = True
    return {'id': 'C8_preconditions_and_validator', 'preconditions_hold_for_stage1': good,
            'negative_controls': negatives, 'validator_refuses': refused,
            'holds': (all(good.values()) and all(v.startswith('refused') for v in negatives.values())
                      and all(refused.values()))}


CHECKS = (check_c1_aa_standin, check_c2_aa_real, check_c3_tail_real, check_c4_rho_real,
          check_c5_certificate_and_early_stop, check_c6_layering, check_c7_keys, check_c8_preconditions)


def run_all_checks():
    out, ok = {}, True
    for fn in CHECKS:
        try:
            r = fn()
        except Exception as error:  # noqa: BLE001 -- recorded as a failing check
            r = {'id': fn.__name__, 'holds': False, 'error': f'{type(error).__name__}: {error}',
                 'traceback': traceback.format_exc()}
        out[r['id']] = r
        ok = ok and r['holds'] is True
    inertness = {
        'AA': {'inert_through_N': out['C1_aa_hold_standin']['inert_through_N']
               and out['C2_aa_hold_real_production'].get('inert_through_N_bitwise') is True,
               'acts_after_N': out['C1_aa_hold_standin']['acts_after_N']
               and out['C2_aa_hold_real_production'].get('acts_after_N') is True},
        'tail': {'inert_through_N': out['C3_tail_hold_real_production'].get('inert_through_N') is True,
                 'acts_after_N': out['C3_tail_hold_real_production'].get('acts_after_N') is True},
        'rho': {'inert_through_N': out['C4_rho_hold_real_production'].get('inert_through_N_bitwise') is True,
                'acts_after_N': out['C4_rho_hold_real_production'].get('acts_after_N') is True},
    }
    return {'all_hold': ok, 'inertness_proof': inertness, 'checks': out}


def main():
    started = _utc()
    out_dir = os.path.join(REPO, OUT_DIR_REL)
    os.makedirs(out_dir, exist_ok=True)
    for f in (OUT_FILE, OUT_MANIFEST):
        if os.path.exists(os.path.join(out_dir, f)):
            raise SystemExit(f'refusing to overwrite existing artifact: {os.path.join(OUT_DIR_REL, f)}')
    res = run_all_checks()
    code_pins = {rel: H.sha256_file(os.path.join(REPO, rel)) for rel in (
        os.path.basename(__file__), 'p515_s53_w98_continuation_hooks.py', 'p515_s44_campaign_harness.py',
        'shared_resources_planning.py', 'admm_anderson_acceleration.py')}
    failures = GUARD.verify(0)
    doc = {'schema': 'p515_s53_w98_zero_solve_checks_v1', 'task': 'W98 (PLANNER_BRIEF_2026-09-13.md Addendum 51)',
           'started_utc': started, 'finished_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
           'N': C.STAGE1_N, 'declaration': C.stage1_declaration(), 'code_sha256': code_pins,
           'guard': {'counts': dict(GUARD.counts), 'verify_0_failures': failures},
           **res}
    path = os.path.join(out_dir, OUT_FILE)
    with open(path, 'x') as handle:
        json.dump(doc, handle, indent=1, sort_keys=True, default=str)
    manifest = {os.path.relpath(path, REPO): H.sha256_file(path)}
    with open(os.path.join(out_dir, OUT_MANIFEST), 'x') as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    for cid, r in res['checks'].items():
        print(f"[W98-CHECKS] {cid}: holds={r['holds']}" + (f" error={r.get('error')}" if r.get('error') else ''))
    print(f"[W98-CHECKS] inertness proof: {json.dumps(res['inertness_proof'])}")
    print(f"[W98-CHECKS] all_hold={res['all_hold']} guard verify(0) failures={failures}")
    print(f"[W98-CHECKS] wrote {os.path.relpath(path, REPO)} sha256={manifest[os.path.relpath(path, REPO)]}")
    GUARD.uninstall()
    sys.exit(0 if (res['all_hold'] and not failures) else 1)


if __name__ == '__main__':
    main()
