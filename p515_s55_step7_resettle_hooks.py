"""
P5.15 Addendum 86 ruling 1, Planner task W183i (a) -- the STEP-7 `settling_resettle` hooks: the fixed cell table of the
Step-7 certified evaluations E3 (1 cell) and E2 stage 1 (8 cells), under the settling rule v6 exactly as the W142
extension (`p515_s53_w142_resettle_ext_v6_hooks`, spec 10f4aa7a) and W155 (`p515_s53_w155_a64_hooks`) use it.

NEW FILE (the W155 pattern, Addendum 86 ruling 1 option (a)). Nothing pinned by an earlier stage is edited: this module
REUSES BY IMPORT the v6 machinery (`p515_s53_w142_resettle_v6_hooks`: the v6 rule declaration
`settling_rule_declaration()` / `settling_criterion_v6`, the v6 state `ResettleStateV6`, the nine wrappers, the capture
checklists) and the W142 extension module (the instance constants, the production signatures). What is NEW here:
  * the declaration schema `p515_s55_step7_settling_resettle_v1`, which the harness router routes here (the one-branch
    router patch in `p515_s44_campaign_harness.resettle_hooks_module`, W183i (a), gated);
  * the cell table below (E3 1, E2 8; the cross-evaluations are NOT in it -- they are added after W186, Addendum 86
    ruling 1), each cell's identity stated truthfully in its declaration (and hence in its eval key);
  * the expected evaluation options carried INSIDE the declaration (`nrf_option_expected`, `rho0_scale_expected`,
    `flex_price_multiplier_expected`) and `spec_entry_checks`: the entry carries EXACTLY those options, each with its
    label at spec and entry level, and nothing else (W155's `flex_price_multiplier_expected` precedent).

THE DECLARATION (the ext-v6 shape; the values the v6 cells, the ext-v6 cells and W155 declare):
  cap rule dynamic `min(k0_run + 109, 300)` (after_first_k0 = V6.CAP_AFTER_K0 = 109, ceiling = W.CAP_CEILING = 300, so
  the spec cap is 300); holds after the run's first residual pass (AA off, tight tail on, rho frozen); the v6 rule
  (`V6.settling_rule_declaration()`); every block recorded; the captures of `e_no_ageing` (10f4aa7a) exactly;
  expected blocks TSO 12 / DSO 36 / ESSO 3; floor-row lower bound 0.7 (the file's minimum_soh, not overridden);
  `replay_reference` None, `abort_on_replay_divergence` False, `first_residual_pass_expected` / `N_old` /
  `original_lapses_after_k0` None (ungated first evaluations).
  Cell identity: `item` 'E3' / 'E2' (the Step-7 study), `claim_group` 'step7_E3' / 'step7_E2', `arm` 'A' (the Step-7
  arm A: the baseline ageing law of the file in force, declared BY ABSENCE of `model_variant` -- Addendum 85 ruling 4
  (ii); it is NOT a key of the ext-v6 ARMS table, and `model_variant` is None), `gate` 'none_first_evaluation' (W155's
  convention for an ungated first evaluation that carries no model-variant arm; ext v6's
  'none_first_evaluation_model_variant_arm' would state a model-variant arm these cells do not have).

THE CELLS (launch order CELL_ORDER; E2 in the pair order of Addendum 86 ruling 3 (i)):
  e3_cnrf_x0_1x       E3: C_NRF -- x = 0, baseline price (1x), nrf_option True (Addendum 85 ruling 4 (viii))
  e2_{x0,unit}_{1x,2x}_rho{0p5,2p0}
                      E2 stage 1: {x = 0, the reference unit node 7 0.25 MVA / 1 MWh 2025} x {1x, 2x} x
                      {rho0_scale 0.5, 2.0} (Addendum 85 ruling 4 (ix))
  1x = no `flex_price_multiplier` in the entry (the baseline price by absence); 2x = `flex_price_multiplier` 2.0 with
  its label. E3 carries no `rho0_scale`; E2 carries no `nrf_option`.

Zero solves: nothing here solves; nothing here builds a model at import. Stdlib (+ the hooks modules it imports,
themselves stdlib-only at import) at import: the harness parent imports it for keys.
"""
import copy
import json
import os
from contextlib import contextmanager

import p515_s53_w101_settling_continuation_hooks as C101
import p515_s53_w105_settling_extension_hooks as E105
import p515_s53_w118_resettle_hooks as R
import p515_s53_w132_resettle_v3_hooks as V
import p515_s53_w142_resettle_v6_hooks as V6
import p515_s53_w142_resettle_ext_v6_hooks as W

SCHEMA = 'p515_s55_step7_settling_resettle_v1'
DECLARATION_SCHEMA = SCHEMA
OPTION_NAME = V.OPTION_NAME
LABEL = ('SRP1 STEP-7 CERTIFIED EVALUATIONS (W183i, Addenda 85-86) -- current production configuration (C2 ageing file '
         'with minimum SoH 0.70, tight tail; no model variant: Step-7 arm A by absence); E3 (C_NRF: x = 0 at the '
         'baseline price with the no-reverse-flow option) and E2 stage 1 ({x = 0, the node-7 reference unit} x {1x, 2x} '
         'x {rho0 x 0.5, x 2.0}), each option keyed and labelled; ungated first evaluations; the certifying regime held '
         'after the run\'s first residual pass (AA off, tight tail on, rho frozen); settling rule v6 until it certifies '
         'or the cap; W105 captures, t_sum, the IPOPT exit, attempt tier and final metrics of every block')
P_MAX = V6.P_MAX
L_MONO = V6.L_MONO
CAP_AFTER_K0 = V6.CAP_AFTER_K0
CAP_CEILING = W.CAP_CEILING
N_TSO_BLOCKS, N_DSO_BLOCKS, N_ESSO = V6.N_TSO_BLOCKS, V6.N_DSO_BLOCKS, V6.N_ESSO
OPTIMAL = V6.OPTIMAL
WRAPPED = V6.WRAPPED
CYCLE_FILE = V6.CYCLE_FILE
BLOCKS_FILE = V6.BLOCKS_FILE
CREEP_FILE = V6.CREEP_FILE
ESS_SCHEDULE_FILE = V6.ESS_SCHEDULE_FILE
DECISION_FILE = V6.DECISION_FILE
SUMMARY_KEY = V6.SUMMARY_KEY
FORBIDDEN_DECLARATION_KEYS = V6.FORBIDDEN_DECLARATION_KEYS
CERTIFICATION_DISABLED_THRESHOLD = V6.CERTIFICATION_DISABLED_THRESHOLD
SETTLING_END_THRESHOLD = V6.SETTLING_END_THRESHOLD
FLOOR_SIDECAR_FILE = W.FLOOR_SIDECAR_FILE
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- the instance and the file in force (the W135 / W142 extension's) -------------------------------------------------
ESS_PARAMS_REL = W.ESS_PARAMS_REL
ESS_PARAMS_SHA256 = W.ESS_PARAMS_SHA256
FLOOR_ROW_LOWER_EXPECTED = W.FLOOR_ROW_LOWER_EXPECTED   # 0.70: the file's minimum_soh (not overridden)
UNIT_CANDIDATE_KEY = W.UNIT_CANDIDATE_KEY                # node 7 0.25 MVA / 1 MWh, 2025
UNIT_NODES = W.UNIT_NODES
X0_CANDIDATE_KEY = '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57'   # = W155's (checked by the gate)
X0_NODES = {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}
INVESTMENT_YEAR = W.UNIT_YEAR                            # 2025
ARM = 'A'
GATE = 'none_first_evaluation'
# The harness's option labels (literal here so this module stays stdlib-only; equality with
# p515_s44_campaign_harness.{NRF_OPTION_LABEL, RHO0_SCALE_LABEL, FLEX_PRICE_LABEL} is checked by the W183i (a) gates).
NRF_OPTION_LABEL = 'EVALUATION OPTION — no reverse flow at the interface (TSO pc_adn >= 0, DSO pg_adn >= 0)'
RHO0_SCALE_LABEL = 'EVALUATION OPTION — rho0 × s (S39 rho0 scaled on every channel and the ESSO)'
FLEX_PRICE_LABEL = 'MODEL VARIANT — flexibility price × m'
# The keys a freeze_campaign_spec entry carries without any option (case-file AA declared, as in 10f4aa7a).
BASE_ENTRY_KEYS = frozenset({'label', 'canonical', 'key', 'eval_key', 'overrides', 'post_certification', 'eval_dir',
                             'working_dir_ids', 'effective_anderson_acceleration', 'settling_resettle'})

_RES = os.path.join('data', 'SRP1', 'Results')
E3_ROOT_REL = os.path.join(_RES, 'P515S55', 'e3_cnrf')      # Addendum 85 ruling 4 (viii)
E2_ROOT_REL = os.path.join(_RES, 'P515S55', 'e2_rho')       # Addendum 85 ruling 4 (ix)
GROUP_OF_ITEM = {'E3': 'step7_E3', 'E2': 'step7_E2'}
STUDY_OF_ITEM = {'E3': 'Step 7 E3: coordination under a matched no-reverse-flow restriction (C_NRF)',
                 'E2': 'Step 7 E2 stage 1: penalty pilot (rho0 x s on every channel and the ESSO)'}


def _cell(item, plan, flex, rho, nrf, role):
    nodes, key = (X0_NODES, X0_CANDIDATE_KEY) if plan == 'x0' else (UNIT_NODES, UNIT_CANDIDATE_KEY)
    return {'item': item, 'plan': plan, 'arm': ARM, 'nodes': nodes, 'investment_year': INVESTMENT_YEAR,
            'candidate_key': key, 'flex_price_multiplier': flex, 'rho0_scale': rho, 'nrf_option': nrf,
            'gated': False, 'N_old': None, 'k0': None, 'original_lapses_after_k0': None, 'cap_ceiling': CAP_CEILING,
            'phase_root': E3_ROOT_REL if item == 'E3' else E2_ROOT_REL, 'role': role}


def _e2_name(plan, flex, rho):
    return f'e2_{plan}_{"1x" if flex is None else "2x"}_rho{str(rho).replace(".", "p")}'


CELLS = {'e3_cnrf_x0_1x': _cell('E3', 'x0', None, None, True,
                                'C_NRF: Q(C_NRF) against Q(C_OPEN) (row 1) and Q(P_NRF) (plan section 3)')}
E2_ORDER = []
for _flex, _rho in ((None, 0.5), (None, 2.0), (2.0, 0.5), (2.0, 2.0)):     # Addendum 86 ruling 3 (i) pair order
    for _plan in ('x0', 'unit'):
        _name = _e2_name(_plan, _flex, _rho)
        CELLS[_name] = _cell('E2', _plan, _flex, _rho, None,
                             f'E2 stage 1: {"x = 0" if _plan == "x0" else "the node-7 reference unit"} at '
                             f'{"1x" if _flex is None else "2x"} under rho0 x {_rho} (its pair: the '
                             f'{"unit" if _plan == "x0" else "x = 0"} cell of the same price and scale)')
        E2_ORDER.append(_name)
E2_CELLS = tuple(E2_ORDER)
E3_CELLS = ('e3_cnrf_x0_1x',)
CELL_ORDER = E3_CELLS + E2_CELLS
GATED_CELLS = ()
UNGATED_CELLS = CELL_ORDER
del _flex, _rho, _plan, _name


def original_eval_dir(cell):
    """Not used (no cell is gated); kept for the reused drivers' signature."""
    raise KeyError(f'{cell}: no replay reference (every Step-7 cell is an ungated first evaluation)')


def reference_path(cell):
    raise KeyError(f'{cell}: no replay reference (every Step-7 cell is an ungated first evaluation)')


def cap_rule(cell):
    return {'kind': 'dynamic', 'after_first_k0': CAP_AFTER_K0, 'ceiling': CELLS[cell]['cap_ceiling'],
            'formula': 'min(k0_run + 109, ceiling) (k0_run: the first residual pass under the version-2 definition)'}


def spec_cap(cell):
    return cap_rule(cell)['ceiling']


def declaration_for(cell):
    """The one valid declaration of a cell: the W142 extension's shape with this schema and label, the v6 rule, the
    e_no_ageing captures, and the three expected options (None = the option is NOT declared in the entry)."""
    if cell not in CELLS:
        raise KeyError(f'{cell} is not a Step-7 cell')
    c = CELLS[cell]
    return {
        'schema': DECLARATION_SCHEMA, 'label': LABEL, 'cell': cell, 'item': c['item'],
        'claim_group': GROUP_OF_ITEM[c['item']],
        'arm': c['arm'], 'model_variant': None,
        'floor_row_lower_expected': FLOOR_ROW_LOWER_EXPECTED,
        'nrf_option_expected': c['nrf_option'],
        'rho0_scale_expected': c['rho0_scale'],
        'flex_price_multiplier_expected': c['flex_price_multiplier'],
        'gate': GATE,
        'first_residual_pass_expected': None, 'N_old': None, 'original_lapses_after_k0': None,
        'holds_after': ('the run first residual pass k0_run (version-2 definition): AA off, tight tail on, rho frozen '
                        'for every later cycle'),
        'cap_rule': cap_rule(cell),
        'settling_rule': V6.settling_rule_declaration(),
        'record_all_blocks': True,
        'captures': {'q_decomposition': True, 'ess_schedule_movement': True, 'boyd_full': True, 't_sum': True,
                     'ipopt_exit_by_block': True, 'soh_floor_sidecar': True,
                     'ageing_trajectory_terminal': True,
                     'exit_clean_by_block': True, 'attempt_tier_and_final_metrics': True},
        'expected_blocks': {'tso': N_TSO_BLOCKS, 'dso': N_DSO_BLOCKS, 'esso': N_ESSO},
        'replay_reference': None,
        'abort_on_replay_divergence': False,
    }


def is_step7_declaration(value):
    return isinstance(value, dict) and value.get('schema') == DECLARATION_SCHEMA


def validate_settling_resettle(value):
    """None = not declared. Otherwise the value must equal `declaration_for(value['cell'])` EXACTLY (types included,
    through the canonical JSON text). Named refusal first: an `early_stop` key. A cell not in the table is refused.
    Returns a new dict. Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} (Step 7) must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} (Step 7) must NOT carry {forbidden} (the settling rule is the only stop before '
                         f'the cap)')
    if value.get('schema') != DECLARATION_SCHEMA:
        raise ValueError(f'{OPTION_NAME} (Step 7): schema must be {DECLARATION_SCHEMA!r}; got {value.get("schema")!r}')
    cell = value.get('cell')
    if cell not in CELLS:
        raise ValueError(f'{OPTION_NAME} (Step 7).cell must be one of {sorted(CELLS)}; got {cell!r}')
    want = declaration_for(cell)
    if set(value) != set(want):
        raise ValueError(f'{OPTION_NAME} (Step 7) must have exactly {sorted(want)}; got {sorted(value)}')
    bad = sorted(k for k in want if json.dumps(value[k], sort_keys=True) != json.dumps(want[k], sort_keys=True))
    if bad:
        raise ValueError(f'{OPTION_NAME} (Step 7) differs from the Step-7 declaration of {cell} on {bad}')
    return copy.deepcopy(want)


def load_replay_reference(decl):
    """W132's loader (None for every Step-7 cell: ungated)."""
    return V.load_replay_reference(decl)


# ======================================================================================================================
#  the entry checks (the child's spec; pure)
# ======================================================================================================================
PRODUCTION_SIGNATURES = W.PRODUCTION_SIGNATURES


def _option_checks(prefix, entry, spec, option, label_key, label, expected):
    """An option declared (expected not None): the entry carries it with exactly the expected value and type and the
    label at entry AND spec level. Not declared (expected None): neither the option nor its label anywhere."""
    if expected is None:
        return {f'{prefix}_absent_from_the_entry': option not in entry,
                f'{prefix}_label_absent_at_entry_and_spec_level': label_key not in entry and label_key not in spec}
    return {f'{prefix}_is_the_declared_value': (option in entry and type(entry[option]) is type(expected)
                                                and entry[option] == expected),
            f'{prefix}_label_at_entry_and_spec_level': entry.get(label_key) == label and spec.get(label_key) == label}


def spec_entry_checks(decl, spec):
    """The campaign spec the child runs holds exactly this cell's entry, and the entry carries EXACTLY the cell's options
    with their labels: `nrf_option` True + `nrf_option_label` (E3) or neither; `rho0_scale` s + `rho0_scale_label`
    (E2) or neither; `flex_price_multiplier` 2.0 + `flex_price_label` (2x) or neither (1x); no `model_variant` /
    `model_variant_label` (arm A by absence); no premium, derived instance, AA override or other wrapper; the
    declared ESS ageing baseline keeps minimum_soh 0.70; the ESS file pinned; the candidate is the cell's. Pure."""
    spec = spec or {}
    cands = spec.get('candidates') or []
    entry = cands[0] if len(cands) == 1 else {}
    cfg = spec.get('configuration') or {}
    cell = decl['cell']
    c = CELLS[cell]
    ess = cfg.get('ess_ageing_baseline') or {}
    options = {'nrf_option': ('nrf_option_label', NRF_OPTION_LABEL, decl['nrf_option_expected']),
               'rho0_scale': ('rho0_scale_label', RHO0_SCALE_LABEL, decl['rho0_scale_expected']),
               'flex_price_multiplier': ('flex_price_label', FLEX_PRICE_LABEL, decl['flex_price_multiplier_expected'])}
    allowed = set(BASE_ENTRY_KEYS)
    for option, (label_key, _label, expected) in options.items():
        if expected is not None:
            allowed |= {option, label_key}
    out = {
        'step7_spec_holds_one_entry': len(cands) == 1,
        'step7_entry_label_is_the_cell': entry.get('label') == cell,
        'step7_entry_settling_resettle_is_the_declaration': (
            json.dumps(entry.get('settling_resettle'), sort_keys=True) == json.dumps(decl, sort_keys=True)),
        'step7_entry_keys_are_exactly_the_base_and_the_declared_options': set(entry) <= allowed and all(
            o in entry for o, (_lk, _l, e) in options.items() if e is not None),
        'step7_no_model_variant_arm_a_by_absence': (decl['model_variant'] is None and 'model_variant' not in entry
                                                    and 'model_variant_label' not in entry
                                                    and 'model_variant_label' not in spec),
        'step7_entry_overrides_empty': entry.get('overrides') == {},
        'step7_ess_ageing_baseline_declared': bool(ess),
        'step7_floor_not_overridden_minimum_soh_is_the_declared_floor': (
            ess.get('minimum_soh') == decl['floor_row_lower_expected'] == FLOOR_ROW_LOWER_EXPECTED),
        'step7_ess_params_file_pinned_as_expected': (cfg.get('ess_params_file') or {}).get('sha256') == ESS_PARAMS_SHA256,
        'step7_no_premium_or_derived_instance': ('interface_deviation_premium' not in entry
                                                 and not cfg.get('derived_instance')),
        'step7_candidate_key_is_the_cell_candidate': entry.get('key') == c['candidate_key'],
        'step7_candidate_investment_year_is_the_cell_year': (
            (entry.get('canonical') or {}).get('investment_year') == c['investment_year']),
        'step7_declared_options_are_the_cell_options': (decl['nrf_option_expected'] == c['nrf_option']
                                                        and decl['rho0_scale_expected'] == c['rho0_scale']
                                                        and decl['flex_price_multiplier_expected']
                                                        == c['flex_price_multiplier']),
    }
    for option, (label_key, label, expected) in options.items():
        out.update(_option_checks(f'step7_{option}', entry, spec, option, label_key, label, expected))
    return out


# ======================================================================================================================
#  capture-path assertion (rule eleven, BEFORE any solve)
# ======================================================================================================================
def assert_resettle_preconditions(decl, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child; and the parent's copy): the W142 extension's / W155's items
    restated for this declaration (the v6 rule, the v6 state) and the Step-7 entry checks (`spec_entry_checks`).
    Raises on any failure, naming every failing check; returns the checklist."""
    import inspect

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
        **V6.rule_declaration_checks(decl['settling_rule']),
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
        'state_carries_every_w118_state_attribute': V6._state_attribute_superset(),
        'state_class_is_the_step7_state_on_the_v6_state': issubclass(ResettleStateStep7, V6.ResettleStateV6),
        'declaration_captures_exit_clean_by_block': decl['captures'].get('exit_clean_by_block') is True,
        'e_no_replay_reference_every_step7_cell_ungated': decl['replay_reference'] is None,
    }
    for name, expected in PRODUCTION_SIGNATURES.items():
        fn = getattr(srp, name, None)
        checks[f'signature:{name}'] = callable(fn) and list(inspect.signature(fn).parameters) == expected
    try:
        IDC.assert_capture_path()
        checks['h_lambda_t_capture_path'] = True
    except Exception:  # noqa: BLE001
        checks['h_lambda_t_capture_path'] = False
    for k, v in E105.capture_path_checklist().items():
        checks[f'capture:{k}'] = bool(v)
    for k, v in R.t_sum_capture_checklist().items():
        checks[f't_sum:{k}'] = bool(v)
    for k, v in V.exit_capture_checklist().items():
        checks[f'exit:{k}'] = bool(v)
    for k, v in V6.clean_capture_checklist(tail_cfg.get('compl_inf_tol') if tail_cfg else None).items():
        checks[f'clean:{k}'] = bool(v)
    for k, v in spec_entry_checks(decl, spec).items():
        checks[k] = bool(v)
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'Step-7 settling-resettle preconditions fail (before any solve): {failing}')
    return checks


# ======================================================================================================================
#  the state and the install
# ======================================================================================================================
class ResettleStateStep7(V6.ResettleStateV6):
    """The v6 state, reused as is; the summary names this schema, the cell's study and arm and its declared options."""

    def summary(self):
        s = super().summary()
        s['schema'] = SCHEMA
        s['state_class'] = ('p515_s53_w142_resettle_v6_hooks.ResettleStateV6 (reused by import; the summary schema '
                            'stamped by p515_s55_step7_resettle_hooks.ResettleStateStep7)')
        s['item'] = self.decl.get('item')
        s['arm'] = self.decl.get('arm')
        s['model_variant_declared'] = self.decl.get('model_variant')
        s['nrf_option_expected'] = self.decl.get('nrf_option_expected')
        s['rho0_scale_expected'] = self.decl.get('rho0_scale_expected')
        s['flex_price_multiplier_expected'] = self.decl.get('flex_price_multiplier_expected')
        return s


@contextmanager
def settling_resettle_hooks(eval_dir, decl, holder, cap):
    """Install the nine v6 wrappers (W139's, unchanged) on the Step-7 state for the run (the harness enters this FIRST).
    Restores every production function on exit, even on error; `holder[SUMMARY_KEY]` gets the summary."""
    import shared_resources_planning as srp
    import p515_s44_campaign_harness as HAR
    decl = validate_settling_resettle(decl)
    if int(cap) != spec_cap(decl['cell']):
        raise RuntimeError(f'settling resettle Step 7: spec cap {cap} != the cell cap {spec_cap(decl["cell"])}')
    if eval_dir is not None:
        for fname in (CYCLE_FILE, BLOCKS_FILE, CREEP_FILE, ESS_SCHEDULE_FILE, DECISION_FILE):
            if os.path.exists(os.path.join(eval_dir, fname)):
                raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ResettleStateStep7(decl, eval_dir, int(cap), reference=load_replay_reference(decl))
    originals = {name: getattr(srp, name) for name in WRAPPED}
    wrappers = V6.make_wrappers(st, originals, HAR.ipopt_exit_class, srp_module=srp,
                                classifier_label=f'p515_s44_campaign_harness.ipopt_exit_class ({HAR.__file__})')
    for name, fn in wrappers.items():
        setattr(srp, name, fn)
    try:
        yield st
    finally:
        for name, fn in originals.items():
            setattr(srp, name, fn)
        holder[SUMMARY_KEY] = st.summary()
