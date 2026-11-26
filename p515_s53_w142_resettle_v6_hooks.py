"""
P5.15 Addendum 61, Planner task W142 -- the v6 RE-SETTLING CAMPAIGN hooks (the 34 cells not yet decided: the eight
remaining D cells first, then claim groups 2 and 4 in the v5 order; frozen stage spec
`frozen_s53_resettle_spec_v6_<sha8>.json`, predecessor v5 ab32ffc9).

WHAT THIS MODULE IS. The W139 v5 re-settling machinery, REUSED BY IMPORT (`p515_s53_w139_resettle_v5_hooks`, pinned by
ab32ffc9; not edited): the cell table (W139's, less the two cells decided since -- b_4649234b certified under v5 by its
run 4b3be392, d_c52e1670 certified under v6 FROM ITS RECORDS, W142 item 2), the nine pass-through wrappers
(`p515_s53_w139_resettle_v5_hooks.make_wrappers`: W118's eight + the v5 exit wrapper with the clean capture, UNCHANGED),
the replay-reference loader, the capture checklists, G8 on production's certificate, and the state
(`ResettleStateV5`, subclassed here). What is NEW here is only:
  * the stop rule VERSION 6 (`settling_criterion_v6`: version 5 with the swing noise floor TAU / 10 in the growth test
    (A) and on the turning-point count (B));
  * the declaration schema that routes a cell to this module (the harness dispatches on it: the v6 router branch, which
    also routes the W142 extension's declarations -- `hooks_module`);
  * the preconditions checklist restated for the v6 declaration and state.
The run is exactly W139's (gated replay through k0, the holds after the run's first residual pass keyed on the
version-2 definition, the captures, W132's exit capture and W139's clean capture); only the rule the recourse wrapper
consults differs.

Zero solves: nothing here solves or builds a model. Stdlib (+ the W139 / W137 / W132 / W118 / W105 / W101 hooks modules,
themselves stdlib-only at import) at import: the harness parent imports it for keys.
"""
import copy
import inspect
import json
import os
from contextlib import contextmanager

import settling_criterion_v6 as SC6
import p515_s53_w101_settling_continuation_hooks as C101
import p515_s53_w105_settling_extension_hooks as E105
import p515_s53_w118_resettle_hooks as R
import p515_s53_w132_resettle_v3_hooks as V
import p515_s53_w139_resettle_v5_hooks as V5

SCHEMA = 'p515_s53_w142_settling_resettle_v6'
DECLARATION_SCHEMA = SCHEMA         # the declaration's 'schema' key: the harness dispatches on it
EXT_DECLARATION_SCHEMA = 'p515_s53_w142_settling_resettle_ext_v6'   # the W142 extension (its own module)
EXT_HOOKS_MODULE = 'p515_s53_w142_resettle_ext_v6_hooks'
OPTION_NAME = V.OPTION_NAME         # 'settling_resettle'
LABEL = ('SRP1 RE-SETTLING RUN v6 (W142, frozen_s53_resettle_spec_v6) -- current production configuration (C2, tight '
         'tail); a gated cell replayed bitwise against its original record through its first residual pass k0 (abort '
         'on divergence), an ungated cell recorded in full; the certifying regime held after the run\'s first residual '
         'pass (AA off, tight tail on, rho frozen); settling rule v6 (v5 -- reading gamma, window (a), the clean veto '
         'within 10x the tail tolerances -- with the swing noise floor tau/10: swings below it excluded from the '
         '"not growing" comparison and a sign change closing a swing below it registering no turning point) until it '
         'certifies or the cap; W105 captures, t_sum, the IPOPT exit, attempt tier and final metrics of every block')
P_MAX = V5.P_MAX
L_MONO = V5.L_MONO
CAP_AFTER_N_OLD = V5.CAP_AFTER_N_OLD     # 100
CAP_AFTER_K0 = SC6.CAP_AFTER_K0         # 109
UNGATED_CAP_CEILING = V5.UNGATED_CAP_CEILING
N_TSO_BLOCKS, N_DSO_BLOCKS, N_ESSO = V5.N_TSO_BLOCKS, V5.N_DSO_BLOCKS, V5.N_ESSO
OPTIMAL = SC6.OPTIMAL_CLASS
WRAPPED = V5.WRAPPED
CYCLE_FILE = V5.CYCLE_FILE
BLOCKS_FILE = V5.BLOCKS_FILE
CREEP_FILE = V5.CREEP_FILE
ESS_SCHEDULE_FILE = V5.ESS_SCHEDULE_FILE
DECISION_FILE = V5.DECISION_FILE
SUMMARY_KEY = V5.SUMMARY_KEY
FORBIDDEN_DECLARATION_KEYS = V5.FORBIDDEN_DECLARATION_KEYS
CERTIFICATION_DISABLED_THRESHOLD = V5.CERTIFICATION_DISABLED_THRESHOLD
SETTLING_END_THRESHOLD = V5.SETTLING_END_THRESHOLD
REPO = os.path.dirname(os.path.abspath(__file__))
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')

# ---- the cells: W139's table less the two decided since v5 froze; order = the v5 order (the 8 D cells first, then
#      groups 2 and 4 as before) ------------------------------------------------------------------------------------------
KEPT_UNDER_V4 = V5.KEPT_UNDER_V4
V4_CELL3 = V5.V4_CELL3
CERTIFIED_UNDER_V5 = {
    'b_4649234b': {'v5_run_commit': '4b3be392', 'k_star': 148, 'status': 'certified',
                   'campaign_root': os.path.join(_P53, 'w139_resettle_v5', 'campaign_s53_w139_resettle_v5_b_4649234b'),
                   'eval_dir': '4de68ced93863cec_b_4649234b', 'eval_key_prefix': '4de68ced'},
}
CERTIFIED_V6_FROM_RECORDS = {
    'd_c52e1670': {'source_v5_run_commit': '51280961', 'status_under_v5': 'uncertified at the cap 198',
                   'campaign_root': os.path.join(_P53, 'w139_resettle_v5', 'campaign_s53_w139_resettle_v5_d_c52e1670'),
                   'eval_dir': 'af42a163b14f0895_d_c52e1670', 'eval_key_prefix': 'af42a163',
                   'certificate_record': os.path.join(_P53, 'w142_resettle_v6', 'v6_from_records',
                                                      'd_c52e1670_v6_from_records_certificate.json'),
                   'label': 'v6 from records'},
}
DECIDED_SINCE_V5 = tuple(CERTIFIED_UNDER_V5) + tuple(CERTIFIED_V6_FROM_RECORDS)
CELL_ORDER = tuple(c for c in V5.CELL_ORDER if c not in DECIDED_SINCE_V5)
CELLS = {c: V5.CELLS[c] for c in CELL_ORDER}
GROUP_OF_ITEM = V5.GROUP_OF_ITEM
GATED_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['gated'])
UNGATED_CELLS = tuple(c for c in CELL_ORDER if not CELLS[c]['gated'])
D_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['item'] == 'D')
DEAD_ZONE_CANDIDATES = V5.DEAD_ZONE_CANDIDATES
DEAD_ZONE_BORDERLINE = V5.DEAD_ZONE_BORDERLINE
LAST_CELL = CELL_ORDER[-1]
original_eval_dir = V5.original_eval_dir
reference_path = V5.reference_path
cap_rule = V5.cap_rule
spec_cap = V5.spec_cap
load_replay_reference = V5.load_replay_reference
exit_capture_checklist = V5.exit_capture_checklist
clean_capture_checklist = V5.clean_capture_checklist
exit_by_block = V5.exit_by_block
exit_counts = V5.exit_counts
block_keys = V5.block_keys
_key_text = V5._key_text
persistence_check_production_certificate = V5.persistence_check_production_certificate
make_wrappers = V5.make_wrappers
make_exit_wrapper = V5.make_exit_wrapper


def settling_rule_declaration():
    """The v5 declaration with the v6 module, class, version and the swing noise floor declared."""
    out = V5.settling_rule_declaration()
    out.update({'module': 'settling_criterion_v6', 'class': 'settling_criterion_v6.SettlingRuleV6',
                'version': SC6.VERSION,
                'swing_floor': {'F': SC6.SWING_FLOOR, 'formula': 'TAU / 10 (Addendum 61 ruling 1)',
                                'growth_test_floor_A': SC6.GROWTH_TEST_FLOOR,
                                'turning_point_floor_B': SC6.TURNING_POINT_FLOOR, 'carries': list(SC6.CARRIES),
                                'carried_by': SC6.CARRY_EVIDENCE['replay']},
                'sub_test_reads': 'enumerated in settling_criterion_v6.SUB_TEST_READS; out-of-window reads recorded'})
    return out


def declaration_for(cell):
    """The one valid declaration of a cell: W139's declaration with this schema, label and the v6 rule."""
    if cell not in CELLS:
        raise KeyError(f'{cell} is not a v6 cell (the cells decided earlier keep their certificates)')
    out = V5.declaration_for(cell)
    out.update({'schema': DECLARATION_SCHEMA, 'label': LABEL, 'settling_rule': settling_rule_declaration()})
    return out


def is_v6_declaration(value):
    """A declaration the v6 router branch takes: this module's schema or the W142 extension's."""
    return isinstance(value, dict) and value.get('schema') in (DECLARATION_SCHEMA, EXT_DECLARATION_SCHEMA)


def hooks_module(value):
    """The module that implements a v6-family declaration (the router branch returns it): this module for the main
    schema, the W142 extension module for the extension schema. Stdlib-only (lazy import)."""
    if isinstance(value, dict) and value.get('schema') == EXT_DECLARATION_SCHEMA:
        import importlib
        return importlib.import_module(EXT_HOOKS_MODULE)
    import sys
    return sys.modules[__name__]


def validate_settling_resettle(value):
    """None = not declared. Otherwise the value must equal `declaration_for(value['cell'])` EXACTLY (an `early_stop`
    key is refused by name). Returns a new dict. Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} (v6) must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} (v6) must NOT carry {forbidden} (the settling rule is the only stop before the '
                         f'cap)')
    if value.get('schema') != DECLARATION_SCHEMA:
        raise ValueError(f'{OPTION_NAME} (v6): schema must be {DECLARATION_SCHEMA!r}; got {value.get("schema")!r}')
    cell = value.get('cell')
    if cell not in CELLS:
        raise ValueError(f'{OPTION_NAME} (v6).cell must be one of {sorted(CELLS)}; got {cell!r}')
    want = declaration_for(cell)
    if set(value) != set(want):
        raise ValueError(f'{OPTION_NAME} (v6) must have exactly {sorted(want)}; got {sorted(value)}')
    bad = sorted(k for k in want if json.dumps(value[k], sort_keys=True) != json.dumps(want[k], sort_keys=True))
    if bad:
        raise ValueError(f'{OPTION_NAME} (v6) differs from the W142 declaration of {cell} on {bad}')
    return copy.deepcopy(want)


# ======================================================================================================================
#  capture-path assertion (rule eleven, BEFORE any solve) -- W139's items restated for the v6 declaration and state
# ======================================================================================================================
PRODUCTION_SIGNATURES = V5.PRODUCTION_SIGNATURES


def rule_declaration_checks(rule):
    """The declared rule is this module's v6 declaration and its constants are settling_criterion_v6's. Pure."""
    return {
        'f_rule_is_the_v6_declaration': rule == settling_rule_declaration(),
        'f_rule_constants_equal_settling_criterion_v6': (rule['tau'] == SC6.TAU and rule['eps0'] == SC6.TAU / 100.0
                                                         and rule['gap_bound'] == SC6.TAU / 2.0
                                                         and rule['p_max'] == 30 and rule['l_mono'] == 60
                                                         and rule['version'] == 6 and rule['retry_tier'] is None
                                                         and rule['reset_on_non_clean'] is False
                                                         and rule['veto_reason'] == SC6.VETO_REASON
                                                         and rule['clean']['factor'] == 10.0
                                                         and rule['swing_floor']['F'] == SC6.TAU / 10.0
                                                         == SC6.SettlingRuleV6.GROWTH_FLOOR
                                                         == SC6.SettlingRuleV6.TP_FLOOR),
        'f_rule_class_callable': callable(getattr(SC6, 'SettlingRuleV6', None)),
    }


def assert_resettle_preconditions(decl, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child; and the parent's copy): W139's items restated for the v6
    declaration and state (the clean-capture facts and the tolerance table unchanged). Raises on any failure; returns
    the checklist."""
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
    tail_cfg = ((spec or {}).get('configuration') or {}).get('convergence_depth_tail') or {}
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
        'state_class_is_the_v6_state_on_the_w139_v5_state': issubclass(ResettleStateV6, V5.ResettleStateV5),
        'rule_class_is_the_v6_adapter': issubclass(HookedRuleV6, SC6.SettlingRuleV6),
        'declaration_captures_exit_clean_by_block': decl['captures'].get('exit_clean_by_block') is True,
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
    for k, v in clean_capture_checklist(tail_cfg.get('compl_inf_tol') if tail_cfg else None).items():
        checks[f'clean:{k}'] = bool(v)
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W142 settling-resettle v6 preconditions fail (before any solve): {failing}')
    return checks


# ======================================================================================================================
#  the v6 rule adapter and the state (W139's, subclassed)
# ======================================================================================================================
_MISSING = object()


class HookedRuleV6(SC6.SettlingRuleV6):
    """The v6 rule as W118's wrappers call it: `observe(k, q, boyd, t_sum)` reads all_clean_k from the state (the v5 exit
    wrapper's clean capture of cycle k); a cycle whose exits were not captured RAISES (fail loudly)."""

    def bind(self, state):
        self._state = state
        return self

    def observe(self, k, q, boyd, t_sum, all_clean=_MISSING):
        if all_clean is _MISSING:
            got = self._state.all_clean.get(k, _MISSING)
            if got is _MISSING:
                self._state.errors.append(f'cycle {k}: all_clean_k not captured before the rule')
                raise RuntimeError(f'W142 settling-resettle v6 hook: cycle {k}: all_clean_k not captured (the local-'
                                   f'solve wrapper did not run this cycle) -- the rule cannot decide')
            all_clean = got
        return super().observe(k, q, boyd, t_sum, all_clean)


def _v6_rule(decl):
    cr = decl['cap_rule']
    p_max = decl['settling_rule']['p_max']
    if cr['kind'] == 'fixed':
        return HookedRuleV6(p_max, cap=cr['cap'], cap_ceiling=cr['ceiling'])
    return HookedRuleV6(p_max, cap_after_first_k0=cr['after_first_k0'], cap_ceiling=cr['ceiling'])


class ResettleStateV6(V5.ResettleStateV5):
    """W139's v5 state (constructor, exit capture, clean capture, summary) with the v6 rule: the constructor is W139's,
    then the rule is replaced by the v6 adapter bound to this state BEFORE any cycle is observed. The summary is W139's
    with this schema, the criterion version and the floor rejections."""

    def __init__(self, decl, eval_dir, cap, reference=None, sink=None):
        super().__init__(decl, eval_dir, cap, reference=reference, sink=sink)
        self.rule = _v6_rule(decl).bind(self)

    def summary(self):
        s = super().summary()
        rule = self.rule
        s.update({'schema': SCHEMA, 'criterion_version': SC6.VERSION, 'reading': SC6.READING,
                  'swing_floor': {'F': SC6.SWING_FLOOR, 'carries': list(SC6.CARRIES)},
                  'turning_point_floor_rejections': rule._all_rejections(),
                  'state_class': ('p515_s53_w142_resettle_v6_hooks.ResettleStateV6 (on p515_s53_w139_resettle_v5_hooks.'
                                  'ResettleStateV5, reused by import)')})
        return s


def _state_attribute_superset():
    """Every attribute a fresh W118 state carries is carried by a fresh v6 state, and the v6 state's rule is the v6
    adapter bound to it (drift guard for the reused constructor)."""
    w118 = R.ResettleState(R.declaration_for(R.CELL_ORDER[2]), None, R.spec_cap(R.CELL_ORDER[2]), reference={}, sink=[])
    cell = CELL_ORDER[0]
    v6 = ResettleStateV6(declaration_for(cell), None, spec_cap(cell), reference={}, sink=[])
    return (set(vars(w118)) <= set(vars(v6)) and isinstance(v6.rule, HookedRuleV6)
            and getattr(v6.rule, '_state', None) is v6)


@contextmanager
def settling_resettle_hooks(eval_dir, decl, holder, cap):
    """Install the nine wrappers for the run on the v6 state (the harness enters this FIRST, so these wrap production
    directly and every harness capture hook wraps them). Restores every production function on exit, even on error;
    `holder[SUMMARY_KEY]` gets the summary."""
    import shared_resources_planning as srp
    import p515_s44_campaign_harness as HAR
    decl = validate_settling_resettle(decl)
    if int(cap) != spec_cap(decl['cell']):
        raise RuntimeError(f'settling resettle v6: spec cap {cap} != the cell cap {spec_cap(decl["cell"])}')
    for fname in (CYCLE_FILE, BLOCKS_FILE, CREEP_FILE, ESS_SCHEDULE_FILE, DECISION_FILE):
        if os.path.exists(os.path.join(eval_dir, fname)):
            raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ResettleStateV6(decl, eval_dir, int(cap), reference=load_replay_reference(decl))
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
