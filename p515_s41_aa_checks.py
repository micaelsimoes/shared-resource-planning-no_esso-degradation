"""
P5.15 Step 3.7 (Anderson acceleration) -- ZERO-SOLVE checks for
`admm_anderson_acceleration.py` and its wiring into
`shared_resources_planning.py:_run_operational_planning`.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 22/23 ("Step 3.7 --
Anderson acceleration"); frozen spec `data/SRP1/Results/P515S41/
frozen_s41_hull_aa_spec_v12_6e5a546f.json`, `item4_step_3_7_anderson`.

A `SolveProfileGuard(permitted=())` is armed for the WHOLE script (zero
solves anywhere); `verify(expected_solves=0, expected_execs=0)` is checked
at the end. `p515_s32_zero_solve_checks._build_admm_ready_state` (a REAL,
production-sequence, zero-solve SRP1 ADMM-ready state -- `.optimize`
monkeypatched to never reach a solver, same technique used throughout the
P5.15 zero-solve checks) is reused, not reimplemented, for Check D.

Checks (each described in full at its own function's docstring below):

  A. Flag off -> no new code path (static source check + the real gating
     predicate).
  B. Type-II AA reproduces an independently hand-computed result on a 1-D
     linear fixed-point map, and accelerates convergence on a slower,
     higher-dimensional contractive linear map (textbook property).
  C. Memory clears on a simulated rho change; the safeguard rejects a step
     when the residual rises (and clears memory); AA switches off while
     all channels are within tolerance and resumes when one leaves.
  D. The stack/scale -> unscale/write-back round trip
     (`build_iterate_layout` + `collect_w` + `write_back_w`) reproduces the
     REAL production consensus/dual stores BITWISE, on the real,
     zero-solve-built SRP1 initial ADMM state, for the identity case
     (raw == scale, or raw == 0) and for a power-of-two perturbation
     (both provably exact in IEEE-754 double precision -- see the check's
     own docstring for why).

Usage:
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s41_aa_checks.py

Writes (both NEW files; refuses to overwrite):
    data/SRP1/Results/P515S41/aa_checks/aa_checks.json
    data/SRP1/Results/P515S41/aa_checks/aa_checks_manifest_sha256.json
"""

import ast
import hashlib
import inspect
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import admm_anderson_acceleration as aa  # noqa: E402
import admm_parameters  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p515_s32_zero_solve_checks import _build_admm_ready_state  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S41', 'aa_checks')
OUT_PATH = os.path.join(OUT_DIR, 'aa_checks.json')
MANIFEST_PATH = os.path.join(OUT_DIR, 'aa_checks_manifest_sha256.json')


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


# ======================================================================================================================
#  Check A -- flag off -> no new code path
# ======================================================================================================================
def check_a_flag_off():
    """
    (1) The real gating predicate: a FRESH `ADMMParameters()` (built
    through the real production class, not a stub) has
    `anderson_acceleration == {'enabled': False, 'memory': 5,
    'regularization': 1e-10}`, and `admm_anderson_acceleration.
    anderson_acceleration_enabled(...)` reads False from it; toggling
    `enabled` to True flips the predicate (proves the function actually
    reads the field, not a constant).

    (2) Static source check, via `ast`, of the ONE production caller
    (`shared_resources_planning._run_operational_planning`): every
    reference inside that function's body to a name whose value is a
    `admm_anderson_acceleration` module attribute (`aa_layout =
    admm_anderson_acceleration.build_iterate_layout(...)`,
    `admm_anderson_acceleration.AndersonAccelerationState(...)`, and the
    two orchestration helpers `_anderson_acceleration_cycle_step` /
    `_get_admm_rho_channel_scalars_for_aa`, which are the ONLY other call
    sites into AA logic) is a direct child (at any depth) of an `ast.If`
    node whose test contains the name `aa_enabled`. This is the source-level
    enforcement that with the flag off, `aa_enabled` is False and NONE of
    these statements execute -- equivalent, for a straight-line CPython
    function with no dynamic code generation, to a runtime guard, and is
    the same technique (`_in source_` checks) used throughout the P5.15
    zero-solve checks (e.g. `p515_g_g1_g4_admm_gates.py`'s s39 checklist)
    for claims about what a production function does NOT call.
    """
    fresh = admm_parameters.ADMMParameters()
    default_settings = dict(fresh.anderson_acceleration)
    default_enabled = aa.anderson_acceleration_enabled(fresh)
    fresh.anderson_acceleration['enabled'] = True
    toggled_enabled = aa.anderson_acceleration_enabled(fresh)

    source = inspect.getsource(srp._run_operational_planning)
    tree = ast.parse(source)
    func_node = tree.body[0]
    assert isinstance(func_node, ast.FunctionDef) and func_node.name == '_run_operational_planning'

    # Build parent links so we can walk UP from any node to find its
    # enclosing `ast.If`.
    parent_of = {}
    for node in ast.walk(func_node):
        for child in ast.iter_child_nodes(node):
            parent_of[child] = node

    def enclosing_if_guards_aa(node):
        current = parent_of.get(node)
        while current is not None and current is not func_node:
            if isinstance(current, ast.If):
                test_src = ast.dump(current.test)
                if 'aa_enabled' in test_src:
                    return True
            current = parent_of.get(current)
        return False

    aa_module_call_names = {'admm_anderson_acceleration'}
    orchestration_helper_names = {'_anderson_acceleration_cycle_step', '_get_admm_rho_channel_scalars_for_aa'}
    # The ONE call that legitimately sits OUTSIDE any `if aa_enabled:` block:
    # it is what DEFINES `aa_enabled` in the first place, so it cannot be
    # guarded by it. Every OTHER reference into the module (including
    # constructing `aa_layout`/`aa_state`) must be guarded.
    unguarded_by_design = {'admm_anderson_acceleration.anderson_acceleration_enabled'}

    unguarded = []
    checked_sites = []
    for node in ast.walk(func_node):
        flagged_name = None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aa_module_call_names:
            flagged_name = f'admm_anderson_acceleration.{node.attr}'
        elif isinstance(node, ast.Name) and node.id in orchestration_helper_names and isinstance(parent_of.get(node), ast.Call):
            flagged_name = node.id
        if flagged_name is None:
            continue
        guarded = enclosing_if_guards_aa(node)
        checked_sites.append({'name': flagged_name, 'lineno': getattr(node, 'lineno', None), 'guarded': guarded})
        if not guarded and flagged_name not in unguarded_by_design:
            unguarded.append({'name': flagged_name, 'lineno': getattr(node, 'lineno', None)})

    return {
        'default_settings': default_settings,
        'default_enabled': default_enabled,
        'toggled_enabled_after_setting_true': toggled_enabled,
        'default_settings_as_expected': default_settings == {'enabled': False, 'memory': 5, 'regularization': 1e-10},
        'default_enabled_is_false': default_enabled is False,
        'toggled_enabled_is_true': toggled_enabled is True,
        'n_call_sites_checked': len(checked_sites),
        'call_sites': checked_sites,
        'n_unguarded_call_sites': len(unguarded),
        'unguarded_call_sites': unguarded,
        'passed': (
            default_settings == {'enabled': False, 'memory': 5, 'regularization': 1e-10}
            and default_enabled is False and toggled_enabled is True
            and len(checked_sites) >= 6  # sanity: the AA wiring has at least 6 distinct call sites
            and len(unguarded) == 0
        ),
    }


# ======================================================================================================================
#  Check B -- textbook AA on a synthetic linear fixed-point map
# ======================================================================================================================
def check_b_linear_map():
    """
    (1) Hand-computable 1-D case. F(w) = 0.5*w + 1.0 (fixed point w* = 2.0),
    w_0 = 0.0. Cycle 1 (m_k=0, insufficient memory): w_1 = F(w_0) = 1.0,
    exactly as the plain iterate (AA is a numerical no-op with empty
    memory -- "active from cycle 1" without a special case). Cycle 2
    (m_k=1): g_0=1.0, g_1=F(1.0)-1.0=0.5, DeltaW=[1.0], DeltaG=[-0.5].
    Regularized normal equations: gamma = (DeltaG^T g_1) /
    (DeltaG^T DeltaG + 1e-10) = (-0.5*0.5)/(0.25+1e-10) =~ -0.999999999601.
    w_hat = w_1 + g_1 - (DeltaW+DeltaG)*gamma = 1.5 - 0.5*gamma =~
    1.999999999800..., i.e. the fixed point to ~2e-10 -- the classic
    "Anderson acceleration / vector (Aitken) extrapolation reaches a
    LINEAR map's fixed point in at most (dimension+1) steps" property,
    here dimension 1 -> 2 steps. This script recomputes gamma/w_hat with
    RAW numpy arithmetic, independently of `AndersonAccelerationState`,
    and compares the two to machine precision.

    (2) Acceleration on a slower map. F(w) = A w + b, A = 0.9 * I (2x2),
    a MUCH slower contraction than case (1) (a plain fixed-point iteration
    needs ~200 cycles to reach 1e-10 from a unit-norm start, since
    0.9^200 =~ 7e-10). AA (memory 5, same regularization, safeguard using
    `combined_residual = ||g_k||_2`, `boyd_all_pass` forced False every
    cycle so it always attempts) is run from the SAME start; the number of
    cycles each needs to reach ||w-w*|| < 1e-10 is reported. AA is
    expected to need far fewer cycles (finite termination on an EXACT
    linear map once memory >= problem dimension is a textbook property of
    Anderson acceleration / Krylov-subspace-equivalent methods -- Walker &
    Ni 2011).
    """
    # ---- (1) hand-computed 1-D case -----------------------------------
    def f_1d(w):
        return 0.5 * w + 1.0

    state = aa.AndersonAccelerationState(memory=5, regularization=1e-10)
    w0 = np.array([0.0])
    g0 = f_1d(w0) - w0
    w1, rec1 = state.step(cycle=1, w_k=w0, g_k=g0, combined_residual=float(np.linalg.norm(g0)), boyd_all_pass=False)

    w1_expected_plain = w0 + g0
    g1 = f_1d(w1) - w1
    w2, rec2 = state.step(cycle=2, w_k=w1, g_k=g1, combined_residual=float(np.linalg.norm(g1)), boyd_all_pass=False)

    # Independent, raw-numpy recomputation of cycle 2's gamma/w_hat (NOT
    # calling AndersonAccelerationState internals).
    delta_w = (w1 - w0).reshape(1, 1)
    delta_g = (g1 - g0).reshape(1, 1)
    gram = delta_g.T @ delta_g + 1e-10 * np.eye(1)
    rhs = delta_g.T @ g1
    gamma_hand = np.linalg.solve(gram, rhs)
    w_hat_hand = w1 + g1 - (delta_w + delta_g) @ gamma_hand

    fixed_point = 2.0
    hand_case = {
        'w1_action': rec1['action'],
        'w1_equals_plain': bool(np.allclose(w1, w1_expected_plain, atol=0.0, rtol=0.0)),
        'w2_action': rec2['action'],
        'w2_accepted': rec2['accepted'],
        'module_w2': w2.tolist(),
        'hand_w2': w_hat_hand.tolist(),
        'module_vs_hand_max_abs_diff': float(np.max(np.abs(w2 - w_hat_hand))),
        'module_w2_vs_fixed_point_abs_diff': float(np.max(np.abs(w2 - fixed_point))),
        'passed': (
            rec1['action'] == 'insufficient memory (m_k=0)'
            and bool(np.allclose(w1, w1_expected_plain, atol=0.0, rtol=0.0))
            and rec2['action'] == 'accepted'
            and float(np.max(np.abs(w2 - w_hat_hand))) < 1e-12
            and float(np.max(np.abs(w2 - fixed_point))) < 1e-8
        ),
    }

    # ---- (2) acceleration on a slower 2-D map --------------------------
    rng = np.random.default_rng(20260918)
    a_mat = 0.9 * np.eye(2)
    b_vec = rng.normal(size=2)
    w_star = np.linalg.solve(np.eye(2) - a_mat, b_vec)

    def f_2d(w):
        return a_mat @ w + b_vec

    def run_plain(max_cycles=5000, tol=1e-10):
        w = np.zeros(2)
        for k in range(1, max_cycles + 1):
            w = f_2d(w)
            if np.linalg.norm(w - w_star) < tol:
                return k
        return None

    def run_aa(max_cycles=5000, tol=1e-10):
        st = aa.AndersonAccelerationState(memory=5, regularization=1e-10)
        w = np.zeros(2)
        for k in range(1, max_cycles + 1):
            g = f_2d(w) - w
            residual = float(np.linalg.norm(g))
            w, _rec = st.step(cycle=k, w_k=w, g_k=g, combined_residual=residual, boyd_all_pass=False)
            if np.linalg.norm(w - w_star) < tol:
                return k
        return None

    plain_cycles = run_plain()
    aa_cycles = run_aa()
    accel_case = {
        'contraction_factor': 0.9,
        'w_star': w_star.tolist(),
        'plain_cycles_to_1e-10': plain_cycles,
        'aa_cycles_to_1e-10': aa_cycles,
        'passed': (plain_cycles is not None and aa_cycles is not None and aa_cycles < plain_cycles),
    }

    return {
        'hand_computed_1d_case': hand_case,
        'acceleration_2d_case': accel_case,
        'passed': hand_case['passed'] and accel_case['passed'],
    }


# ======================================================================================================================
#  Check C -- memory clear on rho change; safeguard rejection; on/off with tolerance
# ======================================================================================================================
def check_c_state_machine():
    """
    Drives the REAL `AndersonAccelerationState` (not reimplemented) through
    a synthetic cycle sequence on a 1-D map that exercises all three rules
    in the frozen spec's `method` block:

    (i)   memory clears on a simulated rho change --
          `clear_for_rho_change` directly, after building up a nonzero
          memory, and confirm `memory_size() == 0` afterwards and the next
          `step()` call reports 'insufficient memory'.
    (ii)  the safeguard rejects a step when the residual RISES past the
          last accepted mark -- construct combined_residual sequences where
          a later cycle's residual exceeds `last_accepted_residual`, and
          confirm `action == 'rejected (safeguard)'`, memory cleared, and
          `last_accepted_residual` UNCHANGED (the recorded ratchet choice,
          module docstring).
    (iii) AA is off while `boyd_all_pass=True` (action records 'off ...',
          `accepted=False`, no write-back needed since nothing beyond the
          plain iterate is returned) and resumes (attempts an
          extrapolation) the cycle `boyd_all_pass` goes back to False,
          using the memory accumulated WHILE off.
    """
    def f(w):
        return 0.5 * w + 1.0

    st = aa.AndersonAccelerationState(memory=5, regularization=1e-10)
    w = np.array([0.0])
    trace = []

    # Cycles 1-2: ordinary attempts (builds memory to size 1, cycle 2
    # accepted -- reproduces the hand-computed case B result).
    for cyc in (1, 2):
        g = f(w) - w
        w_next, rec = st.step(cyc, w, g, combined_residual=float(np.linalg.norm(g)), boyd_all_pass=False)
        trace.append(rec)
        w = w_next
    memory_after_cycle_2 = st.memory_size()
    baseline_after_cycle_2 = st.last_accepted_residual

    # (i) simulated rho change -- memory must clear.
    rho_change_rec = st.clear_for_rho_change(cycle=3, channels_changed=['pf'])
    memory_after_rho_change = st.memory_size()
    trace.append(rho_change_rec)

    # Cycle 4: insufficient memory again (freshly cleared).
    g4 = f(w) - w
    w4, rec4 = st.step(4, w, g4, combined_residual=float(np.linalg.norm(g4)), boyd_all_pass=False)
    trace.append(rec4)
    w = w4

    # Rebuild memory to size >=1 with cycles 5-6 (5 pushes the first pair,
    # 6 has m_k=1 and is evaluated against last_accepted_residual, which
    # was left UNCHANGED by the rho-change clear above (design choice) --
    # use a deliberately LARGE combined_residual for cycle 6 so it exceeds
    # that stale baseline and is REJECTED (ii), even though a real
    # accepted step would ordinarily follow an insufficient-memory cycle
    # (as in case B) -- this is exactly the scenario the ratchet is meant
    # to catch.
    g5 = f(w) - w
    w5, rec5 = st.step(5, w, g5, combined_residual=float(np.linalg.norm(g5)), boyd_all_pass=False)
    trace.append(rec5)
    w = w5

    g6 = f(w) - w
    inflated_residual = baseline_after_cycle_2 * 10.0  # deliberately worse than the stale mark
    memory_before_cycle_6 = st.memory_size()
    baseline_before_cycle_6 = st.last_accepted_residual
    w6, rec6 = st.step(6, w, g6, combined_residual=inflated_residual, boyd_all_pass=False)
    trace.append(rec6)
    memory_after_cycle_6 = st.memory_size()
    baseline_after_cycle_6 = st.last_accepted_residual
    # On rejection the caller keeps the PLAIN iterate (production never
    # writes back on a reject); this check keeps `w` = plain too, matching
    # what `_anderson_acceleration_cycle_step` would leave in the stores.
    w = w6

    # (iii) certificate independence: force boyd_all_pass True for a few
    # cycles (in tolerance), then False again (leaves tolerance) and
    # confirm AA resumes using the memory built while "off".
    off_records = []
    for cyc in (7, 8, 9):
        g_off = f(w) - w
        w_off, rec_off = st.step(cyc, w, g_off, combined_residual=float(np.linalg.norm(g_off)), boyd_all_pass=True)
        off_records.append(rec_off)
        w = w_off  # 'off' returns the plain iterate unconditionally
    memory_while_off = [r['memory_size_after'] for r in off_records]

    g_resume = f(w) - w
    memory_before_resume = st.memory_size()
    w_resume, rec_resume = st.step(10, w, g_resume, combined_residual=float(np.linalg.norm(g_resume)), boyd_all_pass=False)

    return {
        'memory_after_cycle_2': memory_after_cycle_2,
        'rho_change_record': rho_change_rec,
        'memory_after_rho_change': memory_after_rho_change,
        'cycle_4_after_clear': rec4,
        'cycle_6_reject': {
            'memory_before': memory_before_cycle_6, 'baseline_before': baseline_before_cycle_6,
            'record': rec6, 'memory_after': memory_after_cycle_6, 'baseline_after': baseline_after_cycle_6,
        },
        'off_tolerance_records': off_records,
        'memory_while_off': memory_while_off,
        'resume_record': {'memory_before_resume': memory_before_resume, 'record': rec_resume},
        'passed': (
            memory_after_cycle_2 == 1
            and rho_change_rec['action'] == 'memory reset (rho change)'
            and rho_change_rec['reset'] is True
            and memory_after_rho_change == 0
            and rec4['action'] == 'insufficient memory (m_k=0)'
            and rec6['action'] == 'rejected (safeguard)'
            and rec6['reset'] is True
            and memory_after_cycle_6 == 0
            and baseline_after_cycle_6 == baseline_before_cycle_6  # ratchet unchanged on reject
            and all(r['action'].startswith('off (all channels') for r in off_records)
            and all(r['accepted'] is False for r in off_records)
            # Cycle 6 rejected -> cleared to 0; each "off" cycle still PUSHES
            # its (w, g) pair (module docstring's recorded choice), so
            # memory_size() (len(history)-1) grows 0 -> 1 -> 2 over cycles
            # 7, 8, 9 -- one push short of a new column on cycle 7 itself.
            and memory_while_off == [0, 1, 2]
            and rec_resume['action'] in ('accepted', 'rejected (safeguard)')  # a genuine attempt was made
            and memory_before_resume == 2
        ),
    }


# ======================================================================================================================
#  Check D -- collect_w / write_back_w round trip on the REAL SRP1 state
# ======================================================================================================================
def _deepcopy_plain(x):
    if isinstance(x, dict):
        return {k: _deepcopy_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_deepcopy_plain(v) for v in x]
    return x


def check_d_round_trip():
    """
    Builds a REAL, zero-solve SRP1 ADMM-ready state
    (`p515_s32_zero_solve_checks._build_admm_ready_state`, production
    sequence, `.optimize` monkeypatched, never reaches a solver) and runs
    `build_iterate_layout` + `collect_w` + `write_back_w` as a round trip
    in three parts:

      (identity, BITWISE) On the untouched initial state: every V/PF/ESS z
      entry is either exactly its own scale (`consensus_vars['vmag'][...]`
      is initialized to `v_base` itself, `create_admm_variables`) or
      exactly 0.0 (every other initial value, including every dual). `x/x
      == 1.0` exactly for any nonzero finite double (IEEE-754
      round-to-nearest), so `collect_w` -> 1.0 or 0.0 exactly, and
      `write_back_w` -> `1.0*scale == scale` or `0.0*scale == 0.0` exactly
      -- the round trip reproduces EVERY entry (z AND u) bit-for-bit. This
      is also the scenario that matters most in production: the m=0
      "insufficient memory" cycle-1 case is a fresh, never-yet-perturbed
      state exactly like this one.

      (perturbed z, BITWISE) Every z entry (single scale factor) is set to
      a POWER-OF-TWO multiple of its own scale: multiplying a finite
      double by an exact power of two only shifts the exponent (no
      mantissa rounding), and the mathematical quotient of an
      exactly-computed `k*s` by `s` is exactly `k` (a representable
      double, so a correctly-rounded division returns it exactly) --
      z's round trip is ALSO bitwise exact here.

      (perturbed u, TOLERANCE ONLY -- NOT bitwise, and this is stated
      honestly rather than claimed as bitwise). Each scaled dual entry
      `w[i]` is written back through TWO chained non-power-of-two factors
      (`value * rho_channel[group] * scale`, neither rho nor scale is a
      power of two), so even a power-of-two `value` does not, in general,
      survive `y = value*rho*scale` then `w2 = (y/scale)/rho` bit-for-bit
      (each `*`/`/` by a non-power-of-two operand is its own independent
      rounding, and the two directions do not cancel exactly -- verified
      empirically: the FIRST version of this check used power-of-two
      multipliers here too and found `dual_bitwise_match: false`,
      `w_entries_all_exact_multipliers: false`; the z entries, which have
      only ONE scale factor, passed bitwise in that same run). The
      corrected check below verifies the u round trip to a tight RELATIVE
      tolerance (1e-9, several orders above the ~1e-16 machine epsilon
      compounded over 4 chained operations) instead, and reports this
      honestly as a tolerance check, not a bitwise one.

    `check_antisymmetry` in `collect_w` is exercised on every sub-case
    (every dual entry's TSO/DSO mirror pair is set consistently with the
    production invariant before each collect call).

    Additionally (static, `ast`-based, the SAME technique as check A): the
    PRODUCTION m=0/rejected "no-op" claim does not actually rest on any
    round-trip arithmetic at all -- `_anderson_acceleration_cycle_step`
    only calls `write_back_w` inside the `if record['action'] ==
    'accepted':` branch, so on m=0 ('insufficient memory') or a rejected
    step, NOTHING touches `consensus_vars`/`dual_vars`: the stores are left
    bit-for-bit as the plain cycle wrote them, trivially (by non-execution,
    not by an exact-arithmetic coincidence). This is checked directly from
    the source of `_anderson_acceleration_cycle_step`.
    """
    orchestrator_source = inspect.getsource(srp._anderson_acceleration_cycle_step)
    orchestrator_tree = ast.parse(orchestrator_source)
    orchestrator_func = orchestrator_tree.body[0]
    write_back_call_guarded = False
    write_back_call_count = 0
    for node in ast.walk(orchestrator_func):
        if (isinstance(node, ast.Attribute) and node.attr == 'write_back_w'
                and isinstance(node.value, ast.Name) and node.value.id == 'admm_anderson_acceleration'):
            write_back_call_count += 1
            # Find the nearest enclosing `ast.If` and check its test source
            # mentions the accepted-action comparison.
            parent_map = {}
            for n2 in ast.walk(orchestrator_func):
                for c2 in ast.iter_child_nodes(n2):
                    parent_map[c2] = n2
            current = parent_map.get(node)
            while current is not None and current is not orchestrator_func:
                if isinstance(current, ast.If) and "'accepted'" in ast.dump(current.test):
                    write_back_call_guarded = True
                    break
                current = parent_map.get(current)
    write_back_guard_static_check = {
        'write_back_w_call_count_in_orchestrator': write_back_call_count,
        'write_back_w_call_guarded_by_accepted_check': write_back_call_guarded,
        'passed': (write_back_call_count == 1 and write_back_call_guarded),
    }

    eval_id = 'p515s41_aa_zero_solve_check'
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = _build_admm_ready_state(eval_id)
    admm_parameters_obj = planning.params.admm

    assert admm_parameters_obj.shared_ess_reference_rating_mva is not None, (
        'Check D requires shared_ess_reference_rating_mva set by the case file (SRP1_params.json); '
        'got None -- build_iterate_layout would raise, which is the correct behaviour, but this '
        'check specifically exercises the round trip and needs it set.'
    )

    layout = aa.build_iterate_layout(planning, admm_parameters_obj)
    rho_channel = srp._get_admm_rho_channel_scalars_for_aa(tso_model, dso_models, esso_model, admm_parameters_obj)

    # ---- identity case --------------------------------------------------
    original_consensus = _deepcopy_plain(consensus_vars)
    original_dual = _deepcopy_plain(dual_vars)

    w_identity = aa.collect_w(layout, consensus_vars, dual_vars, rho_channel, check_antisymmetry=True)
    aa.write_back_w(layout, w_identity, consensus_vars, dual_vars, rho_channel)

    identity_consensus_matches = (consensus_vars == original_consensus)
    identity_dual_matches = (dual_vars == original_dual)
    identity_w_is_0_or_1 = bool(np.all((w_identity == 0.0) | (w_identity == 1.0)))

    # ---- perturbed z (BITWISE, power-of-two multipliers) ----------------
    # Write a distinct, exact power-of-two multiple of each z entry's scale
    # directly into the stores (bypassing collect/write_back; duals are
    # left at their identity-case value, i.e. 0.0, for this sub-case), then
    # round trip and compare against THIS perturbed state.
    z_multipliers = [2.0, -0.5, 4.0, -1.0, 0.25]
    z_entry_positions = [i for i, e in enumerate(layout['entries']) if e['kind'] == 'z']
    for j, i in enumerate(z_entry_positions):
        e = layout['entries'][i]
        node_id, year, day, p, pt = e['node_id'], e['year'], e['day'], e['p'], e['power_type']
        channel, scale = e['channel'], e['scale']
        mult = z_multipliers[j % len(z_multipliers)]
        if channel == 'v':
            consensus_vars['vmag']['tso']['current'][node_id][year][day][p] = mult * scale
        elif channel == 'pf':
            consensus_vars['pf']['tso']['current'][node_id][year][day][pt][p] = mult * scale
        else:
            consensus_vars['ess']['z']['current'][node_id][year][day][pt][p] = mult * scale

    perturbed_z_consensus = _deepcopy_plain(consensus_vars)
    w_perturbed_z = aa.collect_w(layout, consensus_vars, dual_vars, rho_channel, check_antisymmetry=True)
    aa.write_back_w(layout, w_perturbed_z, consensus_vars, dual_vars, rho_channel)
    perturbed_z_consensus_matches = (consensus_vars == perturbed_z_consensus)
    z_w_values = w_perturbed_z[np.array(z_entry_positions, dtype=int)]
    perturbed_z_w_matches_multipliers = bool(np.array_equal(
        z_w_values, np.array([z_multipliers[j % len(z_multipliers)] for j in range(len(z_entry_positions))])
    ))

    # ---- perturbed u (TOLERANCE ONLY, documented above) ------------------
    # Choose w DIRECTLY in scaled space (arbitrary, not restricted to
    # powers of two) for every dual entry, write it back (produces the
    # physical y = w*rho*scale, both TSO/DSO mirrors), collect again, and
    # compare to a tight RELATIVE tolerance -- not `==`.
    rng = np.random.default_rng(20260918)
    u_entry_positions = [i for i, e in enumerate(layout['entries']) if e['kind'] != 'z']
    w_target = aa.collect_w(layout, consensus_vars, dual_vars, rho_channel, check_antisymmetry=True)
    u_values = rng.uniform(-5.0, 5.0, size=len(u_entry_positions))
    w_target[np.array(u_entry_positions, dtype=int)] = u_values
    aa.write_back_w(layout, w_target, consensus_vars, dual_vars, rho_channel)
    w_roundtrip = aa.collect_w(layout, consensus_vars, dual_vars, rho_channel, check_antisymmetry=True)

    u_target = w_target[np.array(u_entry_positions, dtype=int)]
    u_roundtrip = w_roundtrip[np.array(u_entry_positions, dtype=int)]
    u_max_abs_diff = float(np.max(np.abs(u_roundtrip - u_target)))
    u_max_rel_diff = float(np.max(np.abs(u_roundtrip - u_target) / np.maximum(np.abs(u_target), 1e-300)))
    u_bitwise_match = bool(np.array_equal(u_roundtrip, u_target))  # reported, NOT required to pass

    return {
        'eval_id': eval_id,
        'layout_n_entries': layout['n'],
        'n_z_entries': len(z_entry_positions),
        'n_u_entries': len(u_entry_positions),
        'rho_channel': rho_channel,
        'shared_ess_reference_rating_mva': admm_parameters_obj.shared_ess_reference_rating_mva,
        'identity_case_bitwise': {
            'consensus_bitwise_match': identity_consensus_matches,
            'dual_bitwise_match': identity_dual_matches,
            'w_entries_all_0_or_1': identity_w_is_0_or_1,
        },
        'perturbed_z_case_bitwise': {
            'multipliers_used': z_multipliers,
            'consensus_bitwise_match': perturbed_z_consensus_matches,
            'w_entries_match_multipliers': perturbed_z_w_matches_multipliers,
        },
        'perturbed_u_case_tolerance_only': {
            'note': 'NOT bitwise by construction (two chained non-power-of-two factors, rho and scale); see docstring.',
            'max_abs_diff': u_max_abs_diff,
            'max_rel_diff': u_max_rel_diff,
            'happens_to_be_bitwise_exact': u_bitwise_match,
            'tolerance_used_rel': 1e-9,
            'within_tolerance': u_max_rel_diff < 1e-9,
        },
        'write_back_guard_static_check': write_back_guard_static_check,
        'passed': (
            identity_consensus_matches and identity_dual_matches and identity_w_is_0_or_1
            and perturbed_z_consensus_matches and perturbed_z_w_matches_multipliers
            and u_max_rel_diff < 1e-9
            and write_back_guard_static_check['passed']
        ),
    }


# ======================================================================================================================
#  Main
# ======================================================================================================================
def main():
    _refuse_overwrite(OUT_PATH)
    _refuse_overwrite(MANIFEST_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    guard = SolveProfileGuard(permitted=(), label='P5.15 s41 Anderson acceleration checks').install()
    try:
        result_a = check_a_flag_off()
        result_b = check_b_linear_map()
        result_c = check_c_state_machine()
        result_d = check_d_round_trip()
    finally:
        guard.uninstall()
    guard_failures = guard.verify(expected_solves=0, expected_execs=0)
    if guard_failures:
        raise RuntimeError(guard_failures)

    all_passed = all([result_a['passed'], result_b['passed'], result_c['passed'], result_d['passed']])

    out = {
        'stage': 'P5.15 Step 3.7 Anderson acceleration -- zero-solve checks',
        'authority': (
            'PLANNER_BRIEF_2026-09-13.md Addendum 22/23; frozen spec '
            'data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json, '
            'item4_step_3_7_anderson'
        ),
        'check_A_flag_off': result_a,
        'check_B_linear_map': result_b,
        'check_C_state_machine': result_c,
        'check_D_round_trip': result_d,
        'solve_profile_guard': {'counts': dict(guard.counts), 'verify_failures': guard_failures},
        'all_checks_passed': all_passed,
    }
    with open(OUT_PATH, 'w') as handle:
        json.dump(out, handle, indent=1, default=str)

    manifest = {}
    for path in (OUT_PATH,):
        with open(path, 'rb') as handle:
            manifest[os.path.relpath(path, REPO)] = hashlib.sha256(handle.read()).hexdigest()
    with open(MANIFEST_PATH, 'w') as handle:
        json.dump({'sha256': manifest}, handle, indent=1)

    print(json.dumps({
        'check_A_passed': result_a['passed'],
        'check_B_passed': result_b['passed'],
        'check_C_passed': result_c['passed'],
        'check_D_passed': result_d['passed'],
        'all_checks_passed': all_passed,
        'solve_profile_guard_counts': dict(guard.counts),
    }, indent=1))
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
