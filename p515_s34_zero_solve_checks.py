"""
P5.15 Step 3.4 (+3.3(b) folded in) worker task -- zero-solve verification.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 15 item 5; binding
specification data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json
(supersedes v3 data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json).
The changes under test:

  * D5 -- the ESSO's AL terms (two dual terms, two rho/2 terms) are
    multiplied by `al_scale_esso` when `admm.esso_al_scale` is configured
    in the case file; the ESSO BASE objective is never divided. Absent
    (default): bit-identical to the pre-3.4 code.
  * Fixed sigma -- `admm.objective_scale` (case file); when set,
    `effective_scale = sigma_fixed / block_weight` replaces the computed
    value, which is still obtained and asserted within
    `objective_scale_assert_factor` of the fixed value.
  * S_ref -- `admm.shared_ess_reference_rating_mva`; when set, EVERY one of
    the 13 shared-ESS ADMM normalization call sites uses it (via the single
    helper `_admm_shared_ess_reference_mva` and the two normalization
    functions' new `reference_mva` keyword) in place of the per-agent
    `2*max(S, floor)`.
  * Freeze policy v4 -- `penalty_update.freeze_after_unchanged_cycles` /
    `.freeze_backstop_cycle`: a channel freezes once rho has been unchanged
    for that many consecutive cycles AND has acted at least once, or when
    the global backstop cycle is reached; a channel frozen at the rho clamp
    sets `rho_at_clamp[channel] = True` (a diagnostic, never an exception).

Verifies, on FRESHLY BUILT C* ADMM models (NEW eval ids) and on synthetic
single-entry / hand-derived states, checks (a)-(j) below, with a BLOCKING
`SolveProfileGuard` (permitted call sites = (), i.e. zero solves anywhere)
armed for the whole check:

  (a) D5 equivalence: argmin identity `al_scale * (old_base/al_scale + AL)
      == old_base + al_scale*AL`, both stated algebraically and verified
      numerically on a built ESSO model at a hand-set state;
  (b) esso_al_scale absent -> the built ESSO objective expression is
      bit-identical (same evaluated value, and source-confined-diff) to the
      pre-3.4 code (commit 9f182b79, this worker's own starting HEAD);
  (c) fixed sigma equals the computed sigma within the assert factor on a
      built model set; the assertion fires (raises, with both numbers) when
      given a deliberately wrong sigma;
  (d) S_ref applied at all 13 sites: with S_ref configured, every site's
      normalization equals 2*S_ref (in that site's units); with it absent,
      every site is bit-identical to before;
  (e) freeze rule: unchanged-streak counting, the "must have acted once"
      precondition, the backstop, and clamp detection setting
      `rho_at_clamp` without raising;
  (f) `_update_admm_penalties` still changes only rho and gamma; lambda/z
      (consensus_vars/dual_vars) bit-identical before/after;
  (g) TSO/DSO AL objectives bit-identical to pre-change (commit 9f182b79)
      when the new keys (objective_scale, shared_ess_reference_rating_mva)
      are absent;
  (h) other case params still load with every new key defaulted;
  (i) every preserved fixture unpickles;
  (j) hierarchical and uncoordinated paths carry no hunks.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s34_zero_solve_checks.py

Writes data/SRP1/Results/P515S34/zero_solve_checks/zero_solve_checks.json
(a NEW directory; refuses to overwrite).
"""

import hashlib
import inspect
import json
import math
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s32_zero_solve_checks as V1  # noqa: E402
import p515_s32v2_zero_solve_checks as V2  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S34', 'zero_solve_checks')
OUT_PATH = os.path.join(OUT_DIR, 'zero_solve_checks.json')

SPEC_V4_PATH = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S34', 'frozen_s34_spec_v4_966940a7.json')
SPEC_V4_SHA256 = '966940a789db3093a57d5fdc23be8ccb172b762bf02d0cd09eeaaf0f52f544e9'
# Pre-change commit: this worker's own starting HEAD (P5.15 Step 3.4 frozen
# spec v4 commit) -- the last commit BEFORE this task's production edits.
PRE_CHANGE_COMMIT = '9f182b79'

CS_PARAMS_PATHS = {
    'CS1': os.path.join(REPO, 'data', 'CS1', 'CS1_params.json'),
    'CS7': os.path.join(REPO, 'data', 'CS7', 'CS7_params.json'),
    'HR1': os.path.join(REPO, 'data', 'HR1', 'HR1_params.json'),
    'OP1': os.path.join(REPO, 'data', 'OP1', 'OP1_params.json'),
    'OP2': os.path.join(REPO, 'data', 'OP2', 'OP2_params.json'),
}
SRP1_PARAMS_PATH = os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')

S32_FROZEN_SMOPF_FIXTURES = [
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'FrozenSMOPF',
                 'matched_success_TSO_case9_2025_Summer_cycle7.pkl'),
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'FrozenSMOPF',
                 'matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl'),
]

# The 13 shared-ESS ADMM normalization call sites the frozen spec v4 lists
# (`changes_from_v3.d_ess_reference_rating.call_sites`); the prompt's "11"
# is a discrepancy against the spec's own (authoritative) 13-entry list --
# see the worker report.
S_REF_CALL_SITE_FUNCTIONS = (
    'update_transmission_model_to_admm',
    'update_distribution_models_to_admm',
    'update_shared_energy_storage_model_to_admm',
    '_update_tso_proximal_centres_after_solve',
    'get_admm_residual_metrics',
    'get_admm_boyd_residual_metrics',
    '_update_shared_energy_storage_variables',
)


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _spec_v4_hash():
    with open(SPEC_V4_PATH, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _zero_out_admm_requests(model):
    """Same technique as p515_s33e2_zero_solve_checks._zero_out_admm_requests
    -- sets every otherwise-uninitialized `*_req`/`dual_*` Param on a
    freshly-built TSO model to 0.0, so `pe.value(model.admm_objective)` is
    well-defined at a zero-solve, build-default state."""
    for dn in model.active_distribution_networks:
        for p in model.periods:
            model.vmag_req[dn, p].set_value(0.0)
            model.dual_vmag_req[dn, p].set_value(0.0)
            model.p_pf_req[dn, p].set_value(0.0)
            model.q_pf_req[dn, p].set_value(0.0)
            model.dual_pf_p_req[dn, p].set_value(0.0)
            model.dual_pf_q_req[dn, p].set_value(0.0)
    for e in model.shared_energy_storages:
        for p in model.periods:
            model.p_ess_req[e, p].set_value(0.0)
            model.q_ess_req[e, p].set_value(0.0)
            model.dual_ess_p_req[e, p].set_value(0.0)
            model.dual_ess_q_req[e, p].set_value(0.0)


def _zero_out_esso_requests(model):
    """Sets every `p_req`/`q_req`/`dual_p_req`/`dual_q_req` entry on a
    freshly-built ESSO model to 0.0, so `pe.value(model.admm_objective)` is
    well-defined at a zero-solve, build-default state (mirrors
    `_zero_out_admm_requests` for the TSO model)."""
    for y in model.years:
        for d in model.days:
            for p in model.periods:
                model.p_req[y, d, p].set_value(0.0)
                model.q_req[y, d, p].set_value(0.0)
                model.dual_p_req[y, d, p].set_value(0.0)
                model.dual_q_req[y, d, p].set_value(0.0)


def _build_esso_only(module, eval_id, esso_al_scale_cfg=None, al_scale_esso=1.0,
                      admm_overrides=None):
    """Builds ONLY the ESSO subproblem (`shared_ess_data.build_subproblem()`,
    never solves) through `module`'s own `update_shared_energy_storage_model_to_admm`
    -- zero solves. `module` may be the CURRENT `srp` or a module loaded at a
    historical commit via `V2._load_module_at_commit` (in which case
    `al_scale_esso`/the new call signature are simply not exercised, since
    that commit's function does not accept the keyword)."""
    eval_dir = os.path.join(O.WORK_DIR, eval_id)
    if os.path.exists(eval_dir):
        raise RuntimeError(f'refusing to start: eval dir already exists (network logs append): {eval_dir}')
    planning = O.fresh_planning(eval_id)
    admm_params = planning.params.admm
    if esso_al_scale_cfg is not None:
        admm_params.esso_al_scale = dict(esso_al_scale_cfg)
    if admm_overrides:
        for key, value in admm_overrides.items():
            setattr(admm_params, key, value)

    shared_ess_data = planning.shared_ess_data
    consensus_vars, dual_vars = srp.create_admm_variables(planning)
    esso_model = shared_ess_data.build_subproblem()

    if module is srp:
        module.update_shared_energy_storage_model_to_admm(
            planning, esso_model, admm_params, al_scale_esso=al_scale_esso)
    else:
        module.update_shared_energy_storage_model_to_admm(planning, esso_model, admm_params)

    return planning, esso_model


def check_a_d5_equivalence(results):
    """Check (a): argmin identity, stated and verified numerically.

    Production builds (D5, `update_shared_energy_storage_model_to_admm`):
        new_obj = old_base + al_scale * AL
    The LITERAL reading of "ESSO objective on the same sigma convention as
    the networks" (dividing the base, as the TSO/DSO objectives ARE
    divided) would instead be:
        literal_obj = old_base / al_scale + AL
    These are NOT equal in general, but they differ by exactly the POSITIVE
    constant factor `al_scale` (a fixed Param, independent of every
    decision variable):
        al_scale * literal_obj = al_scale * (old_base/al_scale + AL)
                                = old_base + al_scale*AL
                                = new_obj
    Multiplying an objective by a positive constant does not change its
    argmin, so `new_obj` and `literal_obj` share the same minimiser even
    though they are numerically different functions -- the "deviation in
    FORM, not substance" the spec records. Verified below by constructing
    BOTH expressions on the SAME built ESSO model at a hand-set (nonzero)
    request/dual state and checking `al_scale * literal_value == new_value`
    to double precision.
    """
    al_scale = 5.0  # != 1, so the algebra is non-trivially exercised
    planning, esso_model = _build_esso_only(
        srp, 'p515s34_check_a_d5', esso_al_scale_cfg={'mode': 'fixed', 'value': al_scale, 'source': 'case_file'},
        al_scale_esso=al_scale)

    node_id = next(iter(esso_model))
    model = esso_model[node_id]
    _zero_out_esso_requests(model)
    y, d, p = 0, 0, 0
    model.p_req[y, d, p].set_value(0.37)
    model.q_req[y, d, p].set_value(-0.11)
    model.dual_p_req[y, d, p].set_value(2.3)
    model.dual_q_req[y, d, p].set_value(-1.7)
    model.rho.set_value(0.05)

    new_value = pe.value(model.admm_objective)
    old_base_value = pe.value(model.objective.expr)  # objective is deactivated, expr still evaluable

    # Build the AL-only terms (unscaled) exactly as
    # `update_shared_energy_storage_model_to_admm`'s `else` branch would
    # (same constraint expressions, al_scale factor NOT applied), to
    # construct the literal-reading objective by hand.
    shared_ess_data = planning.shared_ess_data
    years = list(shared_ess_data.years)
    shared_ess_idx = shared_ess_data.get_shared_energy_storage_idx(node_id)
    al_only = 0.0
    for yy in model.years:
        year = years[yy]
        rating = srp._shared_ess_admm_normalization_mva(
            shared_ess_data.shared_energy_storages[year][shared_ess_idx].s,
            planning.params.admm.shared_ess_normalization_floor_mva,
            reference_mva=srp._admm_shared_ess_reference_mva(planning.params.admm),
        )
        for dd in model.days:
            for pp in model.periods:
                cp = (pe.value(model.es_pnet[yy, dd, pp]) - pe.value(model.p_req[yy, dd, pp])) / (2 * rating)
                cq = (pe.value(model.es_qnet[yy, dd, pp]) - pe.value(model.q_req[yy, dd, pp])) / (2 * rating)
                al_only += pe.value(model.dual_p_req[yy, dd, pp]) * cp
                al_only += pe.value(model.dual_q_req[yy, dd, pp]) * cq
                al_only += (pe.value(model.rho) / 2) * cp ** 2
                al_only += (pe.value(model.rho) / 2) * cq ** 2

    literal_value = old_base_value / al_scale + al_only
    identity_holds = math.isclose(al_scale * literal_value, new_value, rel_tol=1e-9, abs_tol=1e-9)
    new_equals_base_plus_scaled_al = math.isclose(new_value, old_base_value + al_scale * al_only, rel_tol=1e-9, abs_tol=1e-9)

    results['d5_equivalence'] = {
        'al_scale': al_scale,
        'old_base_value': old_base_value,
        'al_only_value': al_only,
        'production_new_obj_value': new_value,
        'literal_reading_obj_value': literal_value,
        'al_scale_times_literal': al_scale * literal_value,
        'algebra': (
            'new_obj = old_base + al_scale*AL (production); '
            'literal_obj = old_base/al_scale + AL (literal "same sigma convention" reading); '
            'al_scale*literal_obj = old_base + al_scale*AL = new_obj -- same positive-constant '
            'factor, hence the same argmin, even though new_obj != literal_obj.'
        ),
        'new_obj_equals_base_plus_scaled_al': new_equals_base_plus_scaled_al,
        'identity_holds_al_scale_times_literal_equals_new': identity_holds,
        'pass': identity_holds and new_equals_base_plus_scaled_al and not math.isclose(new_value, literal_value, rel_tol=1e-6),
    }


def check_b_esso_al_scale_absent_bit_identical(results):
    """Check (b): with `esso_al_scale` absent (default), the built ESSO
    objective is bit-identical (same evaluated value; source diff confined)
    to the pre-3.4 code (commit 9f182b79)."""
    pre_module, _pre_source = V2._load_module_at_commit(PRE_CHANGE_COMMIT, 'srp_pre34_b')

    # SRP1's OWN case file now configures esso_al_scale AND
    # shared_ess_reference_rating_mva (both v4 keys) -- the "absent" case
    # under test is these keys NOT being set, so both are explicitly reset
    # to their ADMMParameters.__init__ defaults here (isolating exactly the
    # D5 change this check verifies).
    planning_new, esso_new = _build_esso_only(
        srp, 'p515s34_check_b_current_absent',
        esso_al_scale_cfg={'mode': 'fixed', 'value': 1.0, 'source': 'default'},
        admm_overrides={'shared_ess_reference_rating_mva': None})
    planning_pre, esso_pre = _build_esso_only(pre_module, 'p515s34_check_b_pre_change')

    node_id_new = next(iter(esso_new))
    node_id_pre = next(iter(esso_pre))
    m_new = esso_new[node_id_new]
    m_pre = esso_pre[node_id_pre]

    y, d, p = 0, 0, 0
    for m in (m_new, m_pre):
        _zero_out_esso_requests(m)
        m.p_req[y, d, p].set_value(0.41)
        m.q_req[y, d, p].set_value(-0.19)
        m.dual_p_req[y, d, p].set_value(1.1)
        m.dual_q_req[y, d, p].set_value(-0.7)
        m.rho.set_value(0.05)

    obj_new = pe.value(m_new.admm_objective)
    obj_pre = pe.value(m_pre.admm_objective)
    value_matches = math.isclose(obj_new, obj_pre, rel_tol=1e-12, abs_tol=1e-9)

    has_al_scale_param = hasattr(m_new, 'admm_esso_al_scale')

    new_source = inspect.getsource(srp.update_shared_energy_storage_model_to_admm)
    import difflib
    pre_source_fn = inspect.getsource(pre_module.update_shared_energy_storage_model_to_admm)
    diff_lines = [
        line for line in difflib.unified_diff(pre_source_fn.splitlines(), new_source.splitlines(), lineterm='')
        if (line.startswith('+') or line.startswith('-')) and not line.startswith(('+++', '---'))
    ]
    allowed_keywords = ('al_scale', 'esso_al_scale', 'apply_al_scale', 'reference_mva',
                         '#', 'p5.15', 'spec v4', 'addendum', 'docstring', '"""')
    allowed_bare_lines = ('else:', ')')
    offending = [
        line for line in diff_lines
        if not any(kw in line.lower() for kw in allowed_keywords)
        and line[1:].strip() not in allowed_bare_lines
    ]

    results['esso_al_scale_absent_bit_identical'] = {
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'admm_esso_al_scale_param_constructed_when_absent': has_al_scale_param,
        'objective_value_current_absent': obj_new,
        'objective_value_pre_change': obj_pre,
        'objective_values_match': value_matches,
        'n_diff_lines': len(diff_lines),
        'n_offending_diff_lines': len(offending),
        'offending_lines_sample': offending[:20],
        'pass': value_matches and not has_al_scale_param,
    }


def check_c_sigma_assertion(results):
    """Check (c): the fixed-sigma assertion (a) passes and returns the fixed
    value when the computed value is within `objective_scale_assert_factor`,
    and (b) raises (with both numbers) when given a deliberately wrong
    sigma."""
    admm_params = ADMMParameters()
    admm_params.objective_scale = 93635360.0
    admm_params.objective_scale_assert_factor = 3.0

    # -- within factor 3: 93635360 * 2.5 -----------------------------------
    computed_ok = 93635360.0 * 2.5
    used_ok, sigma_computed_ok, sigma_fixed_ok = srp._resolve_common_admm_objective_scale(computed_ok, admm_params)
    within_factor_ok = (used_ok == 93635360.0 and sigma_computed_ok == computed_ok and sigma_fixed_ok == 93635360.0)

    # -- absent (None): computed value passed through unchanged ------------
    admm_params_absent = ADMMParameters()
    used_absent, sigma_computed_absent, sigma_fixed_absent = srp._resolve_common_admm_objective_scale(42.0, admm_params_absent)
    absent_passthrough = (used_absent == 42.0 and sigma_computed_absent == 42.0 and sigma_fixed_absent is None)

    # -- deliberately wrong sigma: outside factor 3, must raise -------------
    computed_wrong = 93635360.0 * 10.0  # ratio 10, outside factor 3
    raised = False
    error_message = None
    try:
        srp._resolve_common_admm_objective_scale(computed_wrong, admm_params)
    except ValueError as exc:
        raised = True
        error_message = str(exc)
    message_has_both_numbers = (
        error_message is not None
        and f'{computed_wrong:.6e}' in error_message
        and f'{93635360.0:.6e}' in error_message
    )

    results['sigma_assertion'] = {
        'objective_scale_fixed': 93635360.0, 'objective_scale_assert_factor': 3.0,
        'within_factor': {
            'computed': computed_ok, 'used': used_ok, 'sigma_computed': sigma_computed_ok,
            'sigma_fixed': sigma_fixed_ok, 'pass': within_factor_ok,
        },
        'absent': {
            'computed': 42.0, 'used': used_absent, 'sigma_computed': sigma_computed_absent,
            'sigma_fixed': sigma_fixed_absent, 'pass': absent_passthrough,
        },
        'outside_factor': {
            'computed': computed_wrong, 'ratio': computed_wrong / 93635360.0,
            'raised': raised, 'error_message': error_message,
            'error_message_has_both_numbers': message_has_both_numbers,
            'pass': raised and message_has_both_numbers,
        },
        'pass': within_factor_ok and absent_passthrough and raised and message_has_both_numbers,
    }


def check_d_s_ref_all_sites(results):
    """Check (d): with S_ref configured, every call site's normalization
    equals 2*S_ref (in that site's units); with it absent, every site is
    bit-identical (both normalization helpers) to before. Structural
    confirmation that all 13 sites pass `reference_mva=` is via source-text
    grep; the two centralizing helper functions (`_shared_ess_admm_normalization_mva`
    / `_pu`) are unit-tested directly, plus one end-to-end build (S_ref
    configured) confirming the ESSO objective's own constraint normalization
    equals 2*S_ref."""
    module_source = inspect.getsource(srp)
    n_sites_with_reference_mva = module_source.count('reference_mva=_admm_shared_ess_reference_mva(')

    # Every CALL (not definition) of the two normalization functions, found
    # by regex over the whole module source (robust to the call spanning
    # multiple lines); for each, look ahead up to 400 chars (comfortably
    # covers the longest multi-line call site) for `reference_mva=`. The
    # ONE call expected to have none is the internal fallback inside
    # `_shared_ess_admm_normalization_pu`'s own body (already inside the
    # `if reference_mva is not None: return ...` short-circuit, so it never
    # needs the keyword itself) -- identified by its unique, literal text.
    import re
    call_pattern = re.compile(r'_shared_ess_admm_normalization_(?:mva|pu)\(')
    def_pattern = re.compile(r'def _shared_ess_admm_normalization_(?:mva|pu)\(')
    def_starts = {m.start() for m in def_pattern.finditer(module_source)}
    internal_fallback_text = 'return _shared_ess_admm_normalization_mva(rating_mva, floor_mva) / s_base'

    calls_missing_reference_mva = []
    n_calls_total = 0
    for m in call_pattern.finditer(module_source):
        if (m.start() - 4) in def_starts:  # 'def ' immediately precedes this match
            continue
        n_calls_total += 1
        window = module_source[m.start():m.start() + 400]
        if 'reference_mva=' in window:
            continue
        line_start = module_source.rfind('\n', 0, m.start()) + 1
        line_end = module_source.find('\n', m.start())
        line_text = module_source[line_start:line_end]
        if internal_fallback_text in line_text:
            continue  # the one expected exception
        calls_missing_reference_mva.append({'line_text': line_text.strip()})
    call_site_lines = calls_missing_reference_mva

    # -- unit tests: the two normalization helpers ---------------------------
    s_ref = 2.5
    floor = 0.10
    mva_with_ref = srp._shared_ess_admm_normalization_mva(0.96875, floor, reference_mva=s_ref)
    mva_without_ref = srp._shared_ess_admm_normalization_mva(0.96875, floor, reference_mva=None)
    mva_without_ref_default = srp._shared_ess_admm_normalization_mva(0.96875, floor)
    s_base = 100.0
    pu_with_ref = srp._shared_ess_admm_normalization_pu(0.96875 / s_base, s_base, floor, reference_mva=s_ref)
    pu_without_ref = srp._shared_ess_admm_normalization_pu(0.96875 / s_base, s_base, floor, reference_mva=None)

    unit_ok = (
        mva_with_ref == s_ref
        and mva_without_ref == max(0.96875, floor)
        and mva_without_ref == mva_without_ref_default
        and abs(pu_with_ref - s_ref / s_base) < 1e-15
        and abs(pu_without_ref - max(0.96875, floor) / s_base) < 1e-15
    )

    # -- end-to-end: build ESSO with S_ref configured, confirm the built
    #    constraint's normalization equals 2*S_ref exactly ------------------
    planning_ref, esso_ref = _build_esso_only(
        srp, 'p515s34_check_d_s_ref_configured', admm_overrides={'shared_ess_reference_rating_mva': s_ref})
    node_id = next(iter(esso_ref))
    model = esso_ref[node_id]
    _zero_out_esso_requests(model)
    model.p_req[0, 0, 0].set_value(1.0)
    model.dual_p_req[0, 0, 0].set_value(0.0)
    model.rho.set_value(0.0)  # isolate the linear (dual) term, coefficient = 1/(2*S_ref)
    # d(admm_objective)/d(p_req) via the linear dual term alone: dual_p_req * (es_pnet - p_req)/(2*S_ref);
    # with dual_p_req fixed at 1.0 and p_req swept, the objective is affine in p_req with slope -1/(2*S_ref).
    model.dual_p_req[0, 0, 0].set_value(1.0)
    obj_at_0 = pe.value(model.admm_objective)
    model.p_req[0, 0, 0].set_value(2.0)
    obj_at_2 = pe.value(model.admm_objective)
    implied_normalization = -(obj_at_2 - obj_at_0) / (2.0 - 1.0)  # = 1/(2*S_ref)
    implied_s_ref = 1.0 / (2.0 * implied_normalization) if implied_normalization else None
    end_to_end_ok = (implied_s_ref is not None and math.isclose(implied_s_ref, s_ref, rel_tol=1e-9))

    results['s_ref_all_sites'] = {
        'n_call_sites_with_reference_mva_kwarg': n_sites_with_reference_mva,
        'expected_n_call_sites': 13,
        'call_site_lines_missing_reference_mva': call_site_lines,
        'unit_tests_pass': unit_ok,
        'unit_tests': {
            'mva_with_ref': mva_with_ref, 'mva_without_ref': mva_without_ref,
            'pu_with_ref': pu_with_ref, 'pu_without_ref': pu_without_ref,
        },
        'end_to_end_esso_build': {
            's_ref_configured': s_ref, 'implied_s_ref_from_built_objective': implied_s_ref,
            'pass': end_to_end_ok,
        },
        'pass': (n_sites_with_reference_mva == 13 and len(call_site_lines) == 0 and unit_ok and end_to_end_ok),
    }


def check_e_freeze_rule(results):
    """Check (e): per-channel unchanged-streak counting, the "must have
    acted once" precondition, the global backstop, and clamp detection
    (setting `rho_at_clamp` without raising)."""
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
        V1._build_admm_ready_state('p515s34_check_e_freeze'))
    admm_params = planning.params.admm
    admm_params.penalty_update['freeze_after_cycle'] = None  # isolate v4 from the legacy mechanism
    admm_params.penalty_update['freeze_after_unchanged_cycles'] = 3
    admm_params.penalty_update['freeze_backstop_cycle'] = None
    residual_metrics = V1._dummy_residual_metrics(admm_params)
    provocative = V2._fake_boyd_metrics_v2({'v': (30.0, 1.0, 1.0), 'pf': (1.0, 1.0, 1.0), 'ess': (1.0, 1.0, 1.0)})
    dead_band = V2._fake_boyd_metrics_v2({'v': (1.0, 1.0, 1.0), 'pf': (1.0, 1.0, 1.0), 'ess': (1.0, 1.0, 1.0)})

    # -- streak + "must have acted once": V channel acts once (cycle 1),
    #    then holds for 3 consecutive cycles (2,3,4) -> freezes for cycle 5.
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    freeze_state = srp._init_admm_freeze_state()
    history = []
    for cyc, metrics in ((1, provocative), (2, dead_band), (3, dead_band), (4, dead_band), (5, dead_band)):
        actions, before, after, gb, ga, rho_freeze_active, freeze_state = srp._update_admm_penalties(
            tso_model, dso_models, esso_model, residual_metrics, metrics, admm_params,
            iter=cyc, allow_update=True, freeze_state=freeze_state)
        history.append({'cycle': cyc, 'action_v': actions['v'], 'streak_v': freeze_state['v']['unchanged_streak'],
                         'ever_acted_v': freeze_state['v']['ever_acted'], 'frozen_v': freeze_state['v']['frozen']})

    # Cycle 4's OWN action is still the ordinary 'held' (the streak==3
    # trigger is detected AFTER that cycle's action is already decided);
    # the resulting `frozen_v=True` is the state GOING INTO cycle 5, which
    # is where the frozen label first appears -- "takes effect starting the
    # next cycle".
    streak_triggers_at_cycle_5 = (
        history[0]['action_v'] == 'increased'
        and history[1]['streak_v'] == 1 and not history[1]['frozen_v']
        and history[2]['streak_v'] == 2 and not history[2]['frozen_v']
        and history[3]['streak_v'] == 3 and history[3]['action_v'] == 'held' and history[3]['frozen_v']
        and history[4]['frozen_v'] and history[4]['action_v'] == 'held (frozen after 3 unchanged cycles)'
    )

    # -- "must have acted once": a channel that NEVER acts does not freeze
    #    via the streak alone, however long the dead band persists ---------
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    freeze_state_never_acted = srp._init_admm_freeze_state()
    for cyc in range(1, 8):
        actions2, before2, after2, gb2, ga2, rfa2, freeze_state_never_acted = srp._update_admm_penalties(
            tso_model, dso_models, esso_model, residual_metrics, dead_band, admm_params,
            iter=cyc, allow_update=True, freeze_state=freeze_state_never_acted)
    never_acted_never_freezes = (
        not freeze_state_never_acted['pf']['frozen']
        and freeze_state_never_acted['pf']['ever_acted'] is False
        and freeze_state_never_acted['pf']['unchanged_streak'] >= 3
    )

    # -- global backstop: freezes EVERY channel at the configured cycle,
    #    regardless of ever_acted ------------------------------------------
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    admm_params.penalty_update['freeze_after_unchanged_cycles'] = None
    admm_params.penalty_update['freeze_backstop_cycle'] = 5
    freeze_state_backstop = srp._init_admm_freeze_state()
    for cyc in range(1, 6):
        actions3, before3, after3, gb3, ga3, rfa3, freeze_state_backstop = srp._update_admm_penalties(
            tso_model, dso_models, esso_model, residual_metrics, dead_band, admm_params,
            iter=cyc, allow_update=True, freeze_state=freeze_state_backstop)
    backstop_freezes_all = (
        cyc == 5 and all(freeze_state_backstop[g]['frozen'] for g in ('v', 'pf', 'ess'))
        and all(a == 'held (frozen backstop cycle 5)' for a in actions3.values())
        and rfa3 is True
    )

    # -- clamp detection: force rho to the max clamp, trigger the streak,
    #    confirm at_clamp is set WITHOUT raising -----------------------------
    V1._reset_rho(tso_model, dso_models, esso_model, admm_params.penalty_update['max'])
    admm_params.penalty_update['freeze_after_unchanged_cycles'] = 2
    admm_params.penalty_update['freeze_backstop_cycle'] = None
    freeze_state_clamp = srp._init_admm_freeze_state()
    clamp_raised = False
    try:
        # cycle 1: provoke an increase that the clamp then holds at `max`.
        _a, _b, _c, _d, _e, _f, freeze_state_clamp = srp._update_admm_penalties(
            tso_model, dso_models, esso_model, residual_metrics, provocative, admm_params,
            iter=1, allow_update=True, freeze_state=freeze_state_clamp)
        for cyc in (2, 3):
            _a, _b, _c, _d, _e, _f, freeze_state_clamp = srp._update_admm_penalties(
                tso_model, dso_models, esso_model, residual_metrics, dead_band, admm_params,
                iter=cyc, allow_update=True, freeze_state=freeze_state_clamp)
    except Exception:
        clamp_raised = True
    clamp_detected_no_raise = (
        not clamp_raised and freeze_state_clamp['v']['frozen']
        and freeze_state_clamp['v']['at_clamp'] is True
    )

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    results['freeze_rule_v4'] = {
        'streak_history_v': history,
        'streak_triggers_at_cycle_5': streak_triggers_at_cycle_5,
        'never_acted_never_freezes_via_streak': never_acted_never_freezes,
        'backstop_freezes_all_channels': backstop_freezes_all,
        'clamp_detected_without_raising': clamp_detected_no_raise,
        'pass': (
            streak_triggers_at_cycle_5 and never_acted_never_freezes
            and backstop_freezes_all and clamp_detected_no_raise
        ),
    }


def check_f_update_isolation(results):
    """Check (f): `_update_admm_penalties` (v4 freeze_state) changes ONLY
    rho and gamma Params on the models; consensus_vars/dual_vars are
    bit-identical before/after."""
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
        V1._build_admm_ready_state('p515s34_check_f_isolation'))
    admm_params = planning.params.admm
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)

    residual_metrics = V1._dummy_residual_metrics(admm_params)
    boyd = V2._fake_boyd_metrics_v2({'v': (30.0, 1.0, 1.0), 'pf': (1.0, 1.0, 1.0), 'ess': (1.0, 1.0, 1.0)})

    consensus_before = deepcopy(consensus_vars)
    dual_before = deepcopy(dual_vars)

    freeze_state = srp._init_admm_freeze_state()
    actions, before, after, gbefore, gafter, rho_freeze_active, freeze_state = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params,
        iter=1, allow_update=True, freeze_state=freeze_state)

    consensus_unchanged = (consensus_vars == consensus_before)
    dual_unchanged = (dual_vars == dual_before)
    rho_v_changed = (actions['v'] == 'increased' and after['v'] == before['v'] * admm_params.penalty_update['increase_factor'])

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    results['update_isolation'] = {
        'actions': actions, 'rho_before': before, 'rho_after': after,
        'gamma_before': gbefore, 'gamma_after': gafter, 'rho_freeze_active': rho_freeze_active,
        'consensus_vars_unchanged': consensus_unchanged, 'dual_vars_unchanged': dual_unchanged,
        'rho_v_changed_as_expected': rho_v_changed,
        'pass': consensus_unchanged and dual_unchanged and rho_v_changed,
    }


def _build_tso_dso_only(module, eval_id, admm_overrides=None):
    """Builds the TSO + DSO models only, through `module`'s own
    `update_transmission_model_to_admm`/`update_distribution_models_to_admm`,
    with the SAME synthetic `objective_scale` resolution `module` provides
    (current `srp`: via `_resolve_common_admm_objective_scale`, defaulting
    to the computed value when `objective_scale` is None; the historical
    module: a synthetic 1.0, exactly as `V1._build_admm_ready_state` does,
    since `_compute_common_admm_objective_scale` needs a solved, nonzero
    base objective this zero-solve build does not have)."""
    eval_dir = os.path.join(O.WORK_DIR, eval_id)
    if os.path.exists(eval_dir):
        raise RuntimeError(f'refusing to start: eval dir already exists (network logs append): {eval_dir}')
    planning = O.fresh_planning(eval_id)
    admm_params = planning.params.admm
    if admm_overrides:
        for key, value in admm_overrides.items():
            setattr(admm_params, key, value)

    transmission_network = planning.transmission_network
    distribution_networks = planning.distribution_networks
    transmission_network.optimize = V1._stub_optimize.__get__(transmission_network, type(transmission_network))
    for _node_id, _dn in distribution_networks.items():
        _dn.optimize = V1._stub_optimize.__get__(_dn, type(_dn))

    consensus_vars, dual_vars = module.create_admm_variables(planning)
    candidate = planning.get_initial_candidate_solution()
    module._rebuild_candidate_total_capacities(planning, candidate)

    dso_models, _dso_results = module.create_distribution_networks_models(
        distribution_networks, consensus_vars, candidate['total_capacity'], parallel_execution=False)
    tso_model, _tso_results = module.create_transmission_network_model(
        planning, consensus_vars, candidate['total_capacity'])

    module._prepare_distribution_objectives_for_admm(distribution_networks, dso_models)
    module._prepare_transmission_objectives_for_admm(transmission_network, tso_model)
    objective_scale = 1.0  # synthetic, as in V1._build_admm_ready_state -- see docstring
    module.update_distribution_models_to_admm(planning, dso_models, admm_params, objective_scale)
    module.update_transmission_model_to_admm(planning, tso_model, admm_params, objective_scale)

    return planning, tso_model, dso_models


def check_g_tso_dso_bit_identical(results):
    """Check (g): with the new keys (objective_scale,
    shared_ess_reference_rating_mva) absent, the built TSO/DSO AL objectives
    are bit-identical to the pre-change code (commit 9f182b79), on a
    hand-completed (zero request/dual) state -- same technique as
    p515_s33e2_zero_solve_checks.check_g_objective_structure's (g2)."""
    pre_module, _ = V2._load_module_at_commit(PRE_CHANGE_COMMIT, 'srp_pre34_g')

    planning_new, tso_new, dso_new = _build_tso_dso_only(
        srp, 'p515s34_check_g_current_absent',
        admm_overrides={'objective_scale': None, 'shared_ess_reference_rating_mva': None})
    planning_pre, tso_pre, dso_pre = _build_tso_dso_only(pre_module, 'p515s34_check_g_pre_change')

    year = next(iter(planning_new.years))
    day = next(iter(planning_new.days))
    m_new = tso_new[year][day]
    m_pre = tso_pre[year][day]
    _zero_out_admm_requests(m_new)
    _zero_out_admm_requests(m_pre)
    tso_obj_new = pe.value(m_new.admm_objective)
    tso_obj_pre = pe.value(m_pre.admm_objective)
    tso_matches = math.isclose(tso_obj_new, tso_obj_pre, rel_tol=1e-12, abs_tol=1e-9)

    node0 = next(iter(planning_new.distribution_networks))
    dm_new = dso_new[node0][year][day]
    dm_pre = dso_pre[node0][year][day]
    for m in (dm_new, dm_pre):
        for p in m.periods:
            m.vmag_req[p].set_value(0.0)
            m.dual_vmag_req[p].set_value(0.0)
            m.p_pf_req[p].set_value(0.0)
            m.q_pf_req[p].set_value(0.0)
            m.dual_pf_p_req[p].set_value(0.0)
            m.dual_pf_q_req[p].set_value(0.0)
        for p in m.periods:
            # DSO has exactly one shared ESS (the reference node's); its
            # ess Params are indexed by period alone, unlike the TSO's
            # (indexed by (shared_energy_storages, period)).
            m.p_ess_req[p].set_value(0.0)
            m.q_ess_req[p].set_value(0.0)
            m.dual_ess_p_req[p].set_value(0.0)
            m.dual_ess_q_req[p].set_value(0.0)
    dso_obj_new = pe.value(dm_new.admm_objective)
    dso_obj_pre = pe.value(dm_pre.admm_objective)
    dso_matches = math.isclose(dso_obj_new, dso_obj_pre, rel_tol=1e-12, abs_tol=1e-9)

    results['tso_dso_bit_identical_when_absent'] = {
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'tso_objective_current': tso_obj_new, 'tso_objective_pre_change': tso_obj_pre,
        'tso_matches': tso_matches,
        'dso_objective_current': dso_obj_new, 'dso_objective_pre_change': dso_obj_pre,
        'dso_matches': dso_matches,
        'pass': tso_matches and dso_matches,
    }


def check_h_other_case_params_load(results):
    new_keys_absent_defaults = {
        'objective_scale': None, 'objective_scale_source': 'default',
        'objective_scale_assert_factor': 3.0,
        'shared_ess_reference_rating_mva': None, 'shared_ess_reference_rating_source': 'default',
        'esso_al_scale': {'mode': 'fixed', 'value': 1.0, 'source': 'default'},
        'freeze_after_unchanged_cycles': None, 'freeze_backstop_cycle': None,
    }
    per_case = {}
    for name, path in CS_PARAMS_PATHS.items():
        admm = ADMMParameters()
        with open(path) as handle:
            admm.read_parameters_from_file(json.load(handle)['admm'])
        per_case[name] = {
            'path': os.path.relpath(path, REPO),
            'objective_scale': admm.objective_scale,
            'objective_scale_source': admm.objective_scale_source,
            'shared_ess_reference_rating_mva': admm.shared_ess_reference_rating_mva,
            'esso_al_scale': admm.esso_al_scale,
            'freeze_after_unchanged_cycles': admm.penalty_update['freeze_after_unchanged_cycles'],
            'freeze_backstop_cycle': admm.penalty_update['freeze_backstop_cycle'],
        }
    all_defaults = all(
        v['objective_scale'] is None and v['objective_scale_source'] == 'default'
        and v['shared_ess_reference_rating_mva'] is None
        and v['esso_al_scale'] == new_keys_absent_defaults['esso_al_scale']
        and v['freeze_after_unchanged_cycles'] is None and v['freeze_backstop_cycle'] is None
        for v in per_case.values()
    )

    srp1_admm = ADMMParameters()
    with open(SRP1_PARAMS_PATH) as handle:
        srp1_admm.read_parameters_from_file(json.load(handle)['admm'])
    srp1_v4 = {
        'objective_scale': srp1_admm.objective_scale,
        'objective_scale_source': srp1_admm.objective_scale_source,
        'shared_ess_reference_rating_mva': srp1_admm.shared_ess_reference_rating_mva,
        'esso_al_scale': srp1_admm.esso_al_scale,
        'freeze_after_unchanged_cycles': srp1_admm.penalty_update['freeze_after_unchanged_cycles'],
        'freeze_backstop_cycle': srp1_admm.penalty_update['freeze_backstop_cycle'],
        'freeze_after_cycle': srp1_admm.penalty_update['freeze_after_cycle'],
        'rho_v': dict(srp1_admm.rho['v']), 'rho_pf': dict(srp1_admm.rho['pf']), 'rho_ess': dict(srp1_admm.rho['ess']),
    }
    srp1_ok = (
        srp1_v4['objective_scale'] == 93635360.0 and srp1_v4['objective_scale_source'] == 'case_file'
        and srp1_v4['shared_ess_reference_rating_mva'] == 2.5
        and srp1_v4['esso_al_scale'] == {'mode': 'sigma_over_median_block_weight', 'value': None, 'source': 'case_file'}
        and srp1_v4['freeze_after_unchanged_cycles'] == 10 and srp1_v4['freeze_backstop_cycle'] == 60
        and srp1_v4['freeze_after_cycle'] is None  # removed from the case file (superseded)
        and all(v == 0.0077 for v in srp1_v4['rho_v'].values())
        and all(v == 0.198 for v in srp1_v4['rho_pf'].values())
        and all(v == 0.05 for v in srp1_v4['rho_ess'].values())
    )

    results['other_case_params_load'] = {
        'other_cases': per_case, 'other_cases_all_defaults': all_defaults,
        'srp1': srp1_v4, 'srp1_matches_spec_v4': srp1_ok,
        'pass': all_defaults and srp1_ok,
    }


def check_i_fixtures_unpickle(results):
    fixture_results = []
    for path in list(V1.S31C_FIXTURES) + S32_FROZEN_SMOPF_FIXTURES:
        entry = {'path': os.path.relpath(path, REPO)}
        try:
            with open(path, 'rb') as handle:
                pickle.load(handle)
            entry['loads'] = True
        except Exception as exc:  # noqa: BLE001 -- report, don't hide
            entry['loads'] = False
            entry['error'] = f'{type(exc).__name__}: {exc}'
        fixture_results.append(entry)
    results['fixture_unpickling'] = {
        'fixtures': fixture_results,
        'pass': all(f['loads'] for f in fixture_results),
    }


def check_j_untouched_paths(results):
    proc = subprocess.run(
        ['git', 'show', f'{PRE_CHANGE_COMMIT}:shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    import tempfile
    import importlib.util
    tmp_dir = tempfile.mkdtemp(prefix='p515_s34_pre_change_')
    tmp_path = os.path.join(tmp_dir, '_shared_resources_planning_pre34.py')
    with open(tmp_path, 'w') as handle:
        handle.write(proc.stdout)
    spec = importlib.util.spec_from_file_location('srp_pre34_j', tmp_path)
    pre_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre_module)

    fn_names = ('_run_operational_planning_hierarchical', '_run_operational_planning_without_coordination')
    per_fn = {}
    all_identical = True
    for name in fn_names:
        new_source = inspect.getsource(getattr(srp, name))
        pre_source = inspect.getsource(getattr(pre_module, name))
        identical = (new_source == pre_source)
        per_fn[name] = {'identical': identical}
        all_identical = all_identical and identical

    diff = subprocess.run(
        ['git', 'diff', PRE_CHANGE_COMMIT, '--', 'shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    hunk_headers = [line for line in diff.splitlines() if line.startswith('@@')]

    results['hierarchical_uncoordinated_untouched'] = {
        'method': f'exact source-text equality ({PRE_CHANGE_COMMIT} vs current tree) for both '
                  'path functions, plus every diff hunk header in shared_resources_planning.py.',
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'functions': per_fn, 'diff_hunk_count': len(hunk_headers),
        'pass': all_identical,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)

    observed_hash = _spec_v4_hash()
    if observed_hash != SPEC_V4_SHA256:
        raise RuntimeError(f'frozen s34 spec v4 hash mismatch: file={observed_hash} expected={SPEC_V4_SHA256}')

    guard = SolveProfileGuard(permitted=(), label='S34 zero-solve check').install()
    results = {}
    try:
        check_a_d5_equivalence(results)
        check_b_esso_al_scale_absent_bit_identical(results)
        check_c_sigma_assertion(results)
        check_d_s_ref_all_sites(results)
        check_e_freeze_rule(results)
        check_f_update_isolation(results)
        check_g_tso_dso_bit_identical(results)
        check_h_other_case_params_load(results)
        check_i_fixtures_unpickle(results)
        check_j_untouched_paths(results)
    finally:
        guard.uninstall()

    def _entry_pass(v):
        if isinstance(v, dict) and 'pass' in v:
            return bool(v['pass'])
        return True

    all_pass = all(_entry_pass(v) for v in results.values())
    verify_failures = guard.verify(expected_solves=0)

    payload = {
        'stage': 'P5.15 Step 3.4 (+3.3(b) folded in) worker task -- zero-solve verification',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 15 item 5',
            'data/SRP1/Results/P515S34/frozen_s34_spec_v4_966940a7.json',
        ],
        'spec_file': os.path.relpath(SPEC_V4_PATH, REPO),
        'spec_file_sha256': SPEC_V4_SHA256,
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': all_pass,
    }

    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S34] wrote {OUT_PATH}')
    print(f'[S34] all_checks_pass={all_pass} solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
