"""
P5.15 Addendum 59 and its Supplement, Planner task W137 -- the v4 RE-SETTLING CAMPAIGN hooks (claim groups 1, 2 and 4;
the same 38 cells as W132; frozen stage spec `frozen_s53_resettle_spec_v4_<sha8>.json`, predecessor v3 139d1e62).

WHAT THIS MODULE IS. The W132 v3 re-settling machinery, REUSED BY IMPORT (`p515_s53_w132_resettle_v3_hooks`, pinned by
139d1e62, NOT edited): its cell table (the 38 cells, their order, caps and ceilings), its nine pass-through wrappers
(`make_wrappers`: W118's eight + the IPOPT-exit wrapper on `_admm_local_solves_succeeded`), its replay-reference loader,
its exit-capture source facts and its state (`ResettleStateV3`, subclassed here). What is NEW here is only:
  * the stop rule VERSION 4 (`settling_criterion_v4`: reading (gamma) -- no reset on a non-Optimal cycle, only a Boyd
    lapse resets; the certification veto on the last W cycles the test reads, reading (a) of the Addendum 59
    Supplement; the sub-test reads enumerated, the out-of-window reads recorded per certification);
  * the declaration schema that routes a cell to this module (the harness dispatches on it: the v4 router branch);
  * the preconditions checklist restated for the v4 declaration.
The run is exactly W132's (gated replay through k0, the holds after the run's first residual pass keyed on the
version-2 definition, the captures, the exit capture of all 51 blocks per cycle); only the rule the recourse wrapper
consults differs. N and the dynamic cap are version 2's first residual pass (with no non-Optimal reset, identical to
W132's v2-keyed N), so the holds act at the same cycle as in W132.

Zero solves: nothing here solves or builds a model. Stdlib (+ the W132 / W118 / W105 / W101 hooks modules, themselves
stdlib-only at import) at import: the harness parent imports it for keys.
"""
import copy
import inspect
import json
import os
from contextlib import contextmanager

import settling_criterion_v4 as SC4
import p515_s53_w101_settling_continuation_hooks as C101
import p515_s53_w105_settling_extension_hooks as E105
import p515_s53_w118_resettle_hooks as R
import p515_s53_w132_resettle_v3_hooks as V

SCHEMA = 'p515_s53_w137_settling_resettle_v4'
DECLARATION_SCHEMA = SCHEMA         # the declaration's 'schema' key: the harness dispatches on it
OPTION_NAME = V.OPTION_NAME         # 'settling_resettle'
LABEL = ('SRP1 RE-SETTLING RUN v4 (W137, frozen_s53_resettle_spec_v4) -- current production configuration (C2, tight '
         'tail); a gated cell replayed bitwise against its original record through its first residual pass k0 (abort '
         'on divergence), an ungated cell recorded in full; the certifying regime held after the run\'s first residual '
         'pass (AA off, tight tail on, rho frozen); settling rule v4 (reading gamma: no reset on a non-Optimal accepted '
         'solve; certification vetoed while a non-Optimal cycle lies in the last W cycles the test reads) until it '
         'certifies or the cap; W105 captures, t_sum, and the IPOPT exit of every block of every cycle')
P_MAX = V.P_MAX
L_MONO = V.L_MONO
CAP_AFTER_N_OLD = V.CAP_AFTER_N_OLD     # 100
CAP_AFTER_K0 = SC4.CAP_AFTER_K0         # 109
UNGATED_CAP_CEILING = V.UNGATED_CAP_CEILING
N_TSO_BLOCKS, N_DSO_BLOCKS, N_ESSO = V.N_TSO_BLOCKS, V.N_DSO_BLOCKS, V.N_ESSO
OPTIMAL = SC4.OPTIMAL_CLASS
WRAPPED = V.WRAPPED
CYCLE_FILE = V.CYCLE_FILE
BLOCKS_FILE = V.BLOCKS_FILE
CREEP_FILE = V.CREEP_FILE
ESS_SCHEDULE_FILE = V.ESS_SCHEDULE_FILE
DECISION_FILE = V.DECISION_FILE
SUMMARY_KEY = V.SUMMARY_KEY
FORBIDDEN_DECLARATION_KEYS = V.FORBIDDEN_DECLARATION_KEYS
CERTIFICATION_DISABLED_THRESHOLD = V.CERTIFICATION_DISABLED_THRESHOLD
SETTLING_END_THRESHOLD = V.SETTLING_END_THRESHOLD
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- the 38 cells: W132's table, reused unchanged (same cells, order, caps, ceilings, original records) --------------
CELLS = V.CELLS
CELL_ORDER = V.CELL_ORDER
GROUP_OF_ITEM = V.GROUP_OF_ITEM
GATED_CELLS = V.GATED_CELLS
UNGATED_CELLS = V.UNGATED_CELLS
DEAD_ZONE_CANDIDATES = V.DEAD_ZONE_CANDIDATES
DEAD_ZONE_BORDERLINE = V.DEAD_ZONE_BORDERLINE
original_eval_dir = V.original_eval_dir
reference_path = V.reference_path
cap_rule = V.cap_rule
spec_cap = V.spec_cap
load_replay_reference = V.load_replay_reference
exit_capture_checklist = V.exit_capture_checklist
exit_by_block = V.exit_by_block
exit_counts = V.exit_counts
block_keys = V.block_keys
make_exit_wrapper = V.make_exit_wrapper
make_wrappers = V.make_wrappers
_key_text = V._key_text


def settling_rule_declaration():
    return {'module': 'settling_criterion_v4', 'class': 'settling_criterion_v4.SettlingRuleV4', 'version': SC4.VERSION,
            'reading': 'gamma', 'window': 'a', 'reset_on_non_optimal': False,
            'veto': ('a branch verdict is vetoed at k while a non-Optimal cycle lies in its certifying window: '
                     'oscillatory [k - W + 1, k], monotone [k - L + 1, k]; state unchanged'),
            'veto_reason': SC4.VETO_REASON, 'retry_tier': None, 'optimal_class': OPTIMAL,
            'n_and_dynamic_cap_keyed_on': 'the first residual pass under the version-2 definition',
            'sub_test_reads': 'enumerated in settling_criterion_v4.SUB_TEST_READS; out-of-window reads recorded',
            'tau': SC4.TAU, 'eps0': SC4.EPS0, 'k_excl': SC4.K_EXCL, 'w_min': SC4.W_MIN, 'w_factor': SC4.W_FACTOR,
            'p_max': P_MAX, 'l_mono': L_MONO, 'gap_bound': SC4.GAP_BOUND, 'drift_window': SC4.DRIFT_WINDOW}


def declaration_for(cell):
    """The one valid declaration of a cell: W132's declaration with this schema, label and the v4 rule."""
    out = V.declaration_for(cell)
    out.update({'schema': DECLARATION_SCHEMA, 'label': LABEL, 'settling_rule': settling_rule_declaration()})
    return out


def is_v4_declaration(value):
    return isinstance(value, dict) and value.get('schema') == DECLARATION_SCHEMA


def validate_settling_resettle(value):
    """None = not declared. Otherwise the value must equal `declaration_for(value['cell'])` EXACTLY (an `early_stop`
    key is refused by name). Returns a new dict. Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} (v4) must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} (v4) must NOT carry {forbidden} (the settling rule is the only stop before the '
                         f'cap)')
    if value.get('schema') != DECLARATION_SCHEMA:
        raise ValueError(f'{OPTION_NAME} (v4): schema must be {DECLARATION_SCHEMA!r}; got {value.get("schema")!r}')
    cell = value.get('cell')
    if cell not in CELLS:
        raise ValueError(f'{OPTION_NAME} (v4).cell must be one of {sorted(CELLS)}; got {cell!r}')
    want = declaration_for(cell)
    if set(value) != set(want):
        raise ValueError(f'{OPTION_NAME} (v4) must have exactly {sorted(want)}; got {sorted(value)}')
    bad = sorted(k for k in want if json.dumps(value[k], sort_keys=True) != json.dumps(want[k], sort_keys=True))
    if bad:
        raise ValueError(f'{OPTION_NAME} (v4) differs from the W137 declaration of {cell} on {bad}')
    return copy.deepcopy(want)


# ======================================================================================================================
#  capture-path assertion (rule eleven, BEFORE any solve) -- W132's items restated for the v4 declaration
# ======================================================================================================================
PRODUCTION_SIGNATURES = {
    '_capture_convergence_depth_tail_baseline': ['planning_problem', 'admm_parameters'],
    '_apply_convergence_depth_tail': ['planning_problem', 'admm_parameters', 'active', 'baseline', 'cycle'],
    'get_admm_boyd_residual_metrics': ['planning_problem', 'tso_model', 'dso_models', 'esso_model',
                                       'consensus_vars', 'dual_vars', 'admm_parameters'],
    '_anderson_acceleration_cycle_step': ['aa_state', 'aa_layout', 'consensus_vars', 'dual_vars', 'w_before',
                                          'rho_channel', 'boyd_metrics', 'iter'],
    '_convergence_depth_tail_next_state': ['cycle_convergence', 'aa_enabled', 'aa_record'],
    '_update_admm_penalties': ['tso_model', 'dso_models', 'esso_model', 'residual_metrics', 'boyd_metrics',
                               'params', 'iter', 'allow_update', 'freeze_state'],
    '_get_operational_recourse_components': ['planning_problem', 'models'],
    '_get_admm_efc_per_day_max': ['esso_model'],
    '_admm_local_solves_succeeded': ['planning_problem', 'results'],
}


def rule_declaration_checks(rule):
    """The declared rule is this module's v4 declaration and its constants are settling_criterion_v4's. Pure."""
    return {
        'f_rule_is_the_v4_declaration': rule == settling_rule_declaration(),
        'f_rule_constants_equal_settling_criterion_v4': (rule['tau'] == SC4.TAU and rule['eps0'] == SC4.TAU / 100.0
                                                         and rule['gap_bound'] == SC4.TAU / 2.0
                                                         and rule['p_max'] == 30 and rule['l_mono'] == 60
                                                         and rule['version'] == 4 and rule['retry_tier'] is None
                                                         and rule['reset_on_non_optimal'] is False
                                                         and rule['veto_reason'] == SC4.VETO_REASON),
        'f_rule_class_callable': callable(getattr(SC4, 'SettlingRuleV4', None)),
    }


def assert_resettle_preconditions(decl, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child; and the parent's copy): W132's items restated for the v4
    declaration (the spec cap is the cell's and within its per-cell ceiling; the rule is the v4 declaration with
    settling_criterion_v4's constants), the signatures of the nine wrapped functions, W105's capture paths, the t_sum
    source facts, the exit-capture facts (W132's own function) and the W118 state attributes carried by the v4 state.
    Raises on any failure; returns the checklist."""
    import shared_resources_planning as srp
    import admm_anderson_acceleration as aam
    import interface_dual_capture as IDC
    decl = validate_settling_resettle(decl)
    cell = decl['cell']
    cap = int(spec['cap'])
    loop_src = inspect.getsource(srp._run_operational_planning)
    step_src = inspect.getsource(aam.AndersonAccelerationState.step)
    boyd_src = inspect.getsource(srp.get_admm_boyd_residual_metrics)
    pos = [loop_src.find(s) for s in C101.SOURCE_ORDER]
    cr = decl['cap_rule']
    checks = {
        'a_boyd_all_pass_key_in_production': "'all_boyd_pass'" in boyd_src,
        'a_aa_step_called_with_boyd_metrics': ('aa_record = _anderson_acceleration_cycle_step(\n'
                                               '                    aa_state, aa_layout, consensus_vars, dual_vars,\n'
                                               '                    aa_w_before, aa_rho_before, boyd_metrics, iter,')
        in loop_src,
        'a_anderson_acceleration_on': bool(aa_on),
        'b_source_order_aa_step_lt_recourse_lt_convergence_test': all(p >= 0 for p in pos) and pos[0] < pos[1] < pos[2],
        'c_no_early_stop_in_declaration': not any(k in decl for k in FORBIDDEN_DECLARATION_KEYS),
        'c_certificate_length_written_only_by_disable_restore_rule_end': (
            V._certificate_length_writes_in_source() == sorted(R.CERTIFICATE_LENGTH_WRITES)),
        'c_certificate_length_read_only_by_the_loop_test_the_record_and_the_print': (
            sorted(line.strip() for line in loop_src.splitlines() if 'minimum_consecutive_converged_cycles' in line)
            == sorted(C101.CERTIFICATE_LENGTH_READS)),
        'd_spec_cap_equals_the_cell_cap_within_its_ceiling': (cap == spec_cap(cell) and cap <= cr['ceiling']
                                                              and cr['ceiling'] == CELLS[cell]['cap_ceiling']),
        **rule_declaration_checks(decl['settling_rule']),
        'tail_enabled_for_this_run': bool((tail_checklist or {}).get('tail_enabled_for_this_run')),
        'aa_off_literal_is_production': (srp.CONVERGENCE_DEPTH_TAIL_AA_OFF_ACTION == R.AA_OFF_ACTION
                                         and repr(R.AA_OFF_ACTION)[1:-1] in step_src),
        'loop_exit_is_the_convergence_break': 'if convergence:\n            print(f"[INFO] \\t - ADMM converged' in loop_src,
        'block_functions_callable': all(callable(getattr(srp, nm, None)) for nm in (
            '_get_operational_recourse_block_components', '_get_operational_objective_component_blocks')),
        'efc_read_once_per_cycle_after_penalties_before_the_row': (
            loop_src.count('_get_admm_efc_per_day_max(esso_model)') == 1
            and 0 <= loop_src.find('= _update_admm_penalties(') < loop_src.find('_get_admm_efc_per_day_max(esso_model)')
            < loop_src.find('admm_diagnostics.append({')),
        'state_carries_every_w118_state_attribute': _state_attribute_superset(),
        'state_class_is_the_v4_state_on_the_w132_state': issubclass(ResettleStateV4, V.ResettleStateV3),
        'rule_class_is_the_v4_adapter': issubclass(HookedRuleV4, SC4.SettlingRuleV4),
    }
    for name, expected in PRODUCTION_SIGNATURES.items():
        fn = getattr(srp, name, None)
        checks[f'signature:{name}'] = callable(fn) and list(inspect.signature(fn).parameters) == expected
    if decl['replay_reference'] is not None:
        try:
            load_replay_reference(decl)
            checks['e_original_record_hashes_holds_1_N_old_first_pass_k0_lapses_as_declared'] = True
        except Exception:  # noqa: BLE001 -- recorded as a failing check, raised below
            checks['e_original_record_hashes_holds_1_N_old_first_pass_k0_lapses_as_declared'] = False
    try:
        IDC.assert_capture_path()
        checks['h_lambda_t_capture_path'] = True
    except Exception:  # noqa: BLE001
        checks['h_lambda_t_capture_path'] = False
    for k, v in E105.capture_path_checklist().items():
        checks[f'capture:{k}'] = bool(v)
    for k, v in R.t_sum_capture_checklist().items():
        checks[f't_sum:{k}'] = bool(v)
    for k, v in exit_capture_checklist().items():
        checks[f'exit:{k}'] = bool(v)
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W137 settling-resettle v4 preconditions fail (before any solve): {failing}')
    return checks


# ======================================================================================================================
#  the v4 rule adapter and the state (W132's, subclassed)
# ======================================================================================================================
_MISSING = object()


class HookedRuleV4(SC4.SettlingRuleV4):
    """The v4 rule as W118's wrappers call it: `observe(k, q, boyd, t_sum)` reads all_optimal_k from the state (the
    exit wrapper's capture of cycle k); a cycle whose exits were not captured RAISES (fail loudly)."""

    def bind(self, state):
        self._state = state
        return self

    def observe(self, k, q, boyd, t_sum, all_optimal=_MISSING):
        if all_optimal is _MISSING:
            got = self._state.all_optimal.get(k, _MISSING)
            if got is _MISSING:
                self._state.errors.append(f'cycle {k}: all_optimal_k not captured before the rule')
                raise RuntimeError(f'W137 settling-resettle v4 hook: cycle {k}: all_optimal_k not captured (the local-'
                                   f'solve wrapper did not run this cycle) -- the rule cannot decide')
            all_optimal = got
        return super().observe(k, q, boyd, t_sum, all_optimal)


def _v4_rule(decl):
    cr = decl['cap_rule']
    p_max = decl['settling_rule']['p_max']
    if cr['kind'] == 'fixed':
        return HookedRuleV4(p_max, cap=cr['cap'], cap_ceiling=cr['ceiling'])
    return HookedRuleV4(p_max, cap_after_first_k0=cr['after_first_k0'], cap_ceiling=cr['ceiling'])


class ResettleStateV4(V.ResettleStateV3):
    """W132's v3 state (constructor, exit capture, W118's attributes) with the v4 rule: the constructor is W132's (it
    mirrors W118's attribute by attribute; `_state_attribute_superset` asserts it before any solve), then the rule is
    replaced by the v4 adapter bound to this state BEFORE any cycle is observed. The summary is W118's plus the exit
    capture (as W132) and the v4 fields."""

    def __init__(self, decl, eval_dir, cap, reference=None, sink=None):
        super().__init__(decl, eval_dir, cap, reference=reference, sink=sink)
        self.rule = _v4_rule(decl).bind(self)

    def summary(self):
        s = R.ResettleState.summary(self)
        rule = self.rule
        n_cycles = self.cycle or 0
        exit_ok = (sorted(self.all_optimal) == list(range(1, n_cycles + 1)) and self.init_exit is not None
                   and self.local_check_calls == n_cycles + 1)
        s.update({
            'schema': SCHEMA, 'criterion_version': SC4.VERSION, 'reading': SC4.READING,
            'first_k0_v2': rule.n, 'non_optimal_cycles': list(rule.non_optimal_cycles),
            'vetoes': [dict(v) for v in rule.vetoes], 'n_vetoes': len(rule.vetoes),
            'n_cycles_exit_captured': len(self.all_optimal), 'local_check_calls': self.local_check_calls,
            'exit_capture_init_round_0': self.init_exit, 'exit_classifier': self.exit_classifier,
            'exit_capture_complete': exit_ok, 'cap_ceiling': self.decl['cap_rule']['ceiling'],
            'state_class': ('p515_s53_w137_resettle_v4_hooks.ResettleStateV4 (on p515_s53_w132_resettle_v3_hooks.'
                            'ResettleStateV3, reused by import)'),
        })
        s['ok'] = bool(s['ok'] and exit_ok)
        return s


def _state_attribute_superset():
    """Every attribute a fresh W118 state carries is carried by a fresh v4 state, and the v4 state's rule is the v4
    adapter bound to it (drift guard for the reused constructor)."""
    w118 = R.ResettleState(R.declaration_for(R.CELL_ORDER[2]), None, R.spec_cap(R.CELL_ORDER[2]), reference={}, sink=[])
    cell = CELL_ORDER[0]
    v4 = ResettleStateV4(declaration_for(cell), None, spec_cap(cell), reference={}, sink=[])
    return (set(vars(w118)) <= set(vars(v4)) and isinstance(v4.rule, HookedRuleV4)
            and getattr(v4.rule, '_state', None) is v4)


@contextmanager
def settling_resettle_hooks(eval_dir, decl, holder, cap):
    """Install W132's nine wrappers for the run on the v4 state (the harness enters this FIRST, so these wrap production
    directly and every harness capture hook wraps them). Restores every production function on exit, even on error;
    `holder[SUMMARY_KEY]` gets the summary. Mirrors `V.settling_resettle_hooks` with this module's declaration."""
    import shared_resources_planning as srp
    import p515_s44_campaign_harness as HAR
    decl = validate_settling_resettle(decl)
    if int(cap) != spec_cap(decl['cell']):
        raise RuntimeError(f'settling resettle v4: spec cap {cap} != the cell cap {spec_cap(decl["cell"])}')
    for fname in (CYCLE_FILE, BLOCKS_FILE, CREEP_FILE, ESS_SCHEDULE_FILE, DECISION_FILE):
        if os.path.exists(os.path.join(eval_dir, fname)):
            raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ResettleStateV4(decl, eval_dir, int(cap), reference=load_replay_reference(decl))
    originals = {name: getattr(srp, name) for name in WRAPPED}
    wrappers = make_wrappers(st, originals, HAR.ipopt_exit_class, srp_module=srp,
                             classifier_label=f'p515_s44_campaign_harness.ipopt_exit_class ({HAR.__file__})')
    for name, fn in wrappers.items():
        setattr(srp, name, fn)
    try:
        yield st
    finally:
        for name, fn in originals.items():
            setattr(srp, name, fn)
        holder[SUMMARY_KEY] = st.summary()


# ======================================================================================================================
#  G8 on production's certificate (Addendum 59: "gate G8 persistence follows production's certificate, not the settling
#  label"; the W133 label-interaction defect). Pure (stdlib); used by the campaign gate and the zero-solve checks.
# ======================================================================================================================
def _sha256_file(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            b = handle.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def persistence_check_production_certificate(rec, eval_dir, verify_pkl_sha256=True):
    """G8 for a re-settling record: post-certification persistence runs iff PRODUCTION's residual certificate holds on
    the last row (`status_production_trajectory.status == 'certified'`, the view `_apply_settling_resettle_status` keeps
    verbatim), whatever the settling label. Holds iff (production certified: certified_models.pkl present and the
    post-certification status is neither None, 'skipped' nor 'error') or (production not certified: no pkl and the
    post-certification status is 'skipped'). A record without `status_production_trajectory` FAILS (every re-settling
    record carries it). The pkl sha256 is compared with the recorded persisted_models.sha256 (report-only detail)."""
    prod = rec.get('status_production_trajectory')
    pc = rec.get('post_certification') or {}
    pkl = os.path.join(eval_dir, 'certified_models.pkl')
    has_pkl = os.path.exists(pkl)
    prod_cert = isinstance(prod, dict) and prod.get('status') == 'certified'
    if not isinstance(prod, dict):
        ok = False
    elif prod_cert:
        ok = has_pkl and pc.get('status') not in (None, 'skipped', 'error')
    else:
        ok = (not has_pkl) and pc.get('status') == 'skipped'
    recorded = (pc.get('persisted_models') or {}).get('sha256')
    sha_now = _sha256_file(pkl) if (has_pkl and verify_pkl_sha256) else None
    return bool(ok), {'production_trajectory_status': (prod or {}).get('status') if isinstance(prod, dict) else None,
                      'record_status_settling_label': rec.get('status'), 'post_certification_status': pc.get('status'),
                      'certified_models_pkl': has_pkl, 'pkl_sha256_now': sha_now, 'pkl_sha256_recorded': recorded,
                      'pkl_sha256_matches_recorded_report_only': (sha_now == recorded) if sha_now else None,
                      'rule': ('persist_certified_models runs iff production\'s residual certificate holds on the last '
                               'row (status_production_trajectory), whatever the settling verdict (Addendum 59)')}
