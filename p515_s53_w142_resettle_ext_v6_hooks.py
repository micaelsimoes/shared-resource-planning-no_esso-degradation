"""
P5.15 Addendum 61, Planner task W142 item 6 -- the RE-SETTLING EXTENSION hooks RE-FROZEN AGAINST v6: claim group 3
(ageing E, six model-variant arms on the current ESS parameters file, minimum SoH 0.70 NOT overridden) and the
pb_y2025_n5 re-run (7 cells), under the settling rule v6, exactly as the 34 v6 cells use it.

NEW FILE. The W135 extension (`p515_s53_w135_resettle_ext_hooks.py`, frozen under a9d6d1a7) and the W139 extension
re-targeted to v5 (`p515_s53_w139_resettle_ext_v5_hooks.py`, frozen under `frozen_s53_resettle_ext_spec_v2_6e5f60f7`)
are NOT edited: their committed campaign specs stay valid under the harness exactly as frozen. This module is the W139
extension module re-targeted to v6 (a copy with only these changes): it REUSES the W135 extension by import -- its arm
definitions, the instance constants, the spec-entry checks, the pure post-run readers (the G9 replacement, the
floor-year reading, the trajectory equality) -- and restates only what v6 changes:
  * the declaration: schema `p515_s53_w142_settling_resettle_ext_v6` (routed by the v6 router branch through
    `p515_s53_w142_resettle_v6_hooks.hooks_module`), the v6 rule declaration, the clean capture declared;
  * the Phase B cell renamed pb_y2025_n5_v6 (its identifier names its rule, as W137 and W139 renamed it);
  * the state: the v6 state (`p515_s53_w142_resettle_v6_hooks.ResettleStateV6`: the v6 rule adapter on W139's v5
    state, the v5 exit wrapper with the clean capture), subclassed here only to stamp this schema and the arm;
  * the launch gate: the v6 campaign's LAST cell (l_195156fa, v6 #34) results and manifest committed.
The ext cells, their order, caps, configuration and captures are W135's (the 6 E arms ungated, dynamic cap
min(k0_run + 109, 300); pb gated through k0 110 against 4a852725, fixed cap 219).

Zero solves: nothing here solves or builds a model. Stdlib (+ the hooks modules it imports, themselves stdlib-only at
import) at import.
"""
import copy
import hashlib
import inspect
import json
import os
from contextlib import contextmanager

import p515_s53_w101_settling_continuation_hooks as C101
import p515_s53_w105_settling_extension_hooks as E105
import p515_s53_w118_resettle_hooks as R
import p515_s53_w132_resettle_v3_hooks as V
import p515_s53_w135_resettle_ext_hooks as W
import p515_s53_w142_resettle_v6_hooks as V6

SCHEMA = V6.EXT_DECLARATION_SCHEMA
DECLARATION_SCHEMA = SCHEMA
OPTION_NAME = V.OPTION_NAME
LABEL = ('SRP1 RE-SETTLING EXTENSION v6 (W142, frozen_s53_resettle_ext_spec_v3) -- current production configuration (C2 '
         'ageing file with minimum SoH 0.70, tight tail); ageing E arms as MODEL VARIANTS of the ageing law (ungated '
         'first evaluations); pb_y2025_n5 replayed bitwise against its original record through its first residual pass '
         'k0 (abort on divergence); the certifying regime held after the run\'s first residual pass (AA off, tight tail '
         'on, rho frozen); settling rule v6 (v5 -- certification vetoed while a NON-CLEAN cycle lies in the last W cycles '
         'the test reads; clean = Optimal, or Acceptable on the primary attempt within 10x the tail tolerances -- with the '
         'swing noise floor tau/10 in the growth test and on the turning-point count) until it certifies or the cap; '
         'W105 captures, t_sum, the IPOPT exit, attempt tier and final metrics of every block')
P_MAX = W.P_MAX
L_MONO = W.L_MONO
CAP_AFTER_N_OLD = W.CAP_AFTER_N_OLD
CAP_AFTER_K0 = W.CAP_AFTER_K0
CAP_CEILING = W.CAP_CEILING
N_TSO_BLOCKS, N_DSO_BLOCKS, N_ESSO = W.N_TSO_BLOCKS, W.N_DSO_BLOCKS, W.N_ESSO
OPTIMAL = V6.OPTIMAL
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
FLOOR_SIDECAR_FILE = W.FLOOR_SIDECAR_FILE
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- the instance and the file in force: W135's (W134; Addendum 58 Supplement) ---------------------------------------
ESS_PARAMS_REL = W.ESS_PARAMS_REL
ESS_PARAMS_SHA256 = W.ESS_PARAMS_SHA256
FLOOR_ROW_LOWER_EXPECTED = W.FLOOR_ROW_LOWER_EXPECTED
READBACK_RTOL = W.READBACK_RTOL
MODEL_VARIANT_LABEL = W.MODEL_VARIANT_LABEL
UNIT_CANDIDATE_KEY = W.UNIT_CANDIDATE_KEY
UNIT_NODES = W.UNIT_NODES
UNIT_NODE = W.UNIT_NODE
UNIT_YEAR = W.UNIT_YEAR
UNIT_COHORT_INDEX = W.UNIT_COHORT_INDEX
INSTANCE_YEARS = W.INSTANCE_YEARS
ARMS = W.ARMS
BASELINE_EQUIVALENT_ARM = W.BASELINE_EQUIVALENT_ARM

# ---- the cells: W135's, the Phase B cell renamed for its rule ----------------------------------------------------------
RENAMED = {'pb_y2025_n5_v4': 'pb_y2025_n5_v6'}
CELLS = {RENAMED.get(c, c): copy.deepcopy(W.CELLS[c]) for c in W.CELL_ORDER}
CELLS['pb_y2025_n5_v6']['w135_cell'] = 'pb_y2025_n5_v4 (frozen_s53_resettle_ext_spec_v1_a9d6d1a7; never run)'
CELL_ORDER = tuple(RENAMED.get(c, c) for c in W.CELL_ORDER)
E_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['item'] == 'E')
GATED_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['gated'])
UNGATED_CELLS = tuple(c for c in CELL_ORDER if not CELLS[c]['gated'])
GROUP_OF_ITEM = W.GROUP_OF_ITEM
CELL_OF_ARM = {CELLS[c]['arm']: c for c in E_CELLS}

# ---- the v6 campaign's last cell: the cells start only after its results are committed -------------------------------
# Found by the v6 stage spec's series prefix (its name carries its sha256 prefix, checked): this module is on the v6
# router's dispatch path, so it cannot pin the v6 spec's hash; the extension's own frozen spec records the v6 stage
# spec's path and sha256.
V6_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w142_resettle_v6')
V6_STAGE_SPEC_PREFIX = 'frozen_s53_resettle_spec_v6_'
V6_LAST_CELL = V6.LAST_CELL
V6_CAMPAIGN_ID_PREFIX = 's53_w142_resettle_v6_'
V6_LAST_CELL_INDEX_1_BASED = len(V6.CELL_ORDER)


def v6_stage_spec_rel():
    """(rel, sha256) of the ONE frozen v6 stage spec in the v6 root whose file name carries its own sha256 prefix;
    (None, None) when there is none, more than one, or the name does not carry the hash."""
    root = os.path.join(REPO, V6_ROOT_REL)
    hits = sorted(f for f in os.listdir(root) if f.startswith(V6_STAGE_SPEC_PREFIX) and f.endswith('.json')) \
        if os.path.isdir(root) else []
    if len(hits) != 1:
        return None, None
    rel = os.path.join(V6_ROOT_REL, hits[0])
    with open(os.path.join(REPO, rel), 'rb') as handle:
        sha = hashlib.sha256(handle.read()).hexdigest()
    return (rel, sha) if hits[0] == f'{V6_STAGE_SPEC_PREFIX}{sha[:8]}.json' else (None, None)


def v6_last_cell_files_rel():
    root = os.path.join(V6_ROOT_REL, f'campaign_{V6_CAMPAIGN_ID_PREFIX}{V6_LAST_CELL}')
    return (os.path.join(root, 'campaign_results.json'), os.path.join(root, 'campaign_manifest_sha256.json'))


def original_eval_dir(cell):
    return os.path.join(CELLS[cell]['orig_root'], 'evals', CELLS[cell]['orig_eval_dir'])


def reference_path(cell):
    return os.path.join(original_eval_dir(cell), 'per_cycle_record.jsonl')


def arm_variant(cell):
    arm = CELLS[cell]['arm']
    return copy.deepcopy(ARMS[arm]) if arm is not None else None


def cap_rule(cell):
    c = CELLS[cell]
    if c['gated']:
        return {'kind': 'fixed', 'cap': c['N_old'] + CAP_AFTER_N_OLD, 'ceiling': c['cap_ceiling'],
                'formula': 'N_old + 100'}
    return {'kind': 'dynamic', 'after_first_k0': CAP_AFTER_K0, 'ceiling': c['cap_ceiling'],
            'formula': 'min(k0_run + 109, ceiling) (k0_run: the first residual pass under the version-2 definition)'}


def spec_cap(cell):
    r = cap_rule(cell)
    return r['cap'] if r['kind'] == 'fixed' else r['ceiling']


def declaration_for(cell):
    """The one valid declaration of a cell: W135's shape with this schema, label, the v6 rule and the clean capture."""
    c = CELLS[cell]
    gated = c['gated']
    return {
        'schema': DECLARATION_SCHEMA, 'label': LABEL, 'cell': cell, 'item': c['item'],
        'claim_group': GROUP_OF_ITEM[c['item']],
        'arm': c['arm'], 'model_variant': arm_variant(cell),
        'floor_row_lower_expected': FLOOR_ROW_LOWER_EXPECTED,
        'gate': 'bitwise_through_first_residual_pass' if gated else 'none_first_evaluation_model_variant_arm',
        'first_residual_pass_expected': c['k0'] if gated else None,
        'N_old': c['N_old'] if gated else None,
        'original_lapses_after_k0': list(c['original_lapses_after_k0']) if gated else None,
        'holds_after': ('the run first residual pass k0_run (version-2 definition): AA off, tight tail on, rho frozen '
                        'for every later cycle'),
        'cap_rule': cap_rule(cell),
        'settling_rule': V6.settling_rule_declaration(),
        'record_all_blocks': True,
        'captures': {'q_decomposition': True, 'ess_schedule_movement': True, 'boyd_full': True, 't_sum': True,
                     'ipopt_exit_by_block': True, 'soh_floor_sidecar': True,
                     'ageing_trajectory_terminal': c['arm'] is not None,
                     'exit_clean_by_block': True, 'attempt_tier_and_final_metrics': True},
        'expected_blocks': {'tso': N_TSO_BLOCKS, 'dso': N_DSO_BLOCKS, 'esso': N_ESSO},
        'replay_reference': ({'per_cycle_record': reference_path(cell), 'sha256': c['per_cycle_record_sha256'],
                              'n_cycles': c['N_old'], 'gated_through_cycle': c['k0'],
                              'original_eval_key': c['orig_eval_key']} if gated else None),
        'abort_on_replay_divergence': bool(gated),
    }


def is_ext_v6_declaration(value):
    return isinstance(value, dict) and value.get('schema') == DECLARATION_SCHEMA


def validate_settling_resettle(value):
    """None = not declared. Otherwise the value must equal `declaration_for(value['cell'])` EXACTLY (an `early_stop`
    key is refused by name). Returns a new dict. Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} (W142 ext v6) must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} (W142 ext v6) must NOT carry {forbidden} (the settling rule is the only stop '
                         f'before the cap)')
    if value.get('schema') != DECLARATION_SCHEMA:
        raise ValueError(f'{OPTION_NAME} (W142 ext v6): schema must be {DECLARATION_SCHEMA!r}; got '
                         f'{value.get("schema")!r}')
    cell = value.get('cell')
    if cell not in CELLS:
        raise ValueError(f'{OPTION_NAME} (W142 ext v6).cell must be one of {sorted(CELLS)}; got {cell!r}')
    want = declaration_for(cell)
    if set(value) != set(want):
        raise ValueError(f'{OPTION_NAME} (W142 ext v6) must have exactly {sorted(want)}; got {sorted(value)}')
    bad = sorted(k for k in want if json.dumps(value[k], sort_keys=True) != json.dumps(want[k], sort_keys=True))
    if bad:
        raise ValueError(f'{OPTION_NAME} (W142 ext v6) differs from the W142 declaration of {cell} on {bad}')
    return copy.deepcopy(want)


def load_replay_reference(decl):
    """W132's loader (generic in the declaration; reused unchanged)."""
    return V.load_replay_reference(decl)


PRODUCTION_SIGNATURES = W.PRODUCTION_SIGNATURES
spec_entry_checks = W.spec_entry_checks
closed_form_expected = W.closed_form_expected
variant_readback_gate = W.variant_readback_gate
floor_year_reading = W.floor_year_reading
trajectory_equality = W.trajectory_equality


def assert_resettle_preconditions(decl, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child; and the parent's copy): W135's items restated for v6 (the rule
    is the v6 declaration with settling_criterion_v6's constants; the state is the v6 state on W139's v5 state),
    W135's entry checks (`spec_entry_checks`, reused) and the v5 clean-capture facts with the tolerance table. Raises
    on any failure; returns the checklist."""
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
    rule = decl['settling_rule']
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
        **V6.rule_declaration_checks(rule),
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
        'state_class_is_the_w142_v6_state': issubclass(ResettleStateExtV6, V6.ResettleStateV6),
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
    for k, v in V.exit_capture_checklist().items():
        checks[f'exit:{k}'] = bool(v)
    for k, v in V6.clean_capture_checklist(tail_cfg.get('compl_inf_tol') if tail_cfg else None).items():
        checks[f'clean:{k}'] = bool(v)
    for k, v in spec_entry_checks(decl, spec).items():
        checks[k] = bool(v)
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W142 settling-resettle extension v6 preconditions fail (before any solve): {failing}')
    return checks


class ResettleStateExtV6(V6.ResettleStateV6):
    """The v6 state, reused as is (W139's constructor, exit and clean capture, the v6 rule adapter, the v6 summary);
    only the summary's schema names this extension and carries the arm."""

    def summary(self):
        s = super().summary()
        s['schema'] = SCHEMA
        s['state_class'] = ('p515_s53_w142_resettle_v6_hooks.ResettleStateV6 (reused by import; the summary schema '
                            'stamped by p515_s53_w142_resettle_ext_v6_hooks.ResettleStateExtV6)')
        s['arm'] = self.decl.get('arm')
        s['model_variant_declared'] = self.decl.get('model_variant')
        return s


@contextmanager
def settling_resettle_hooks(eval_dir, decl, holder, cap):
    """Install the nine wrappers (W139's, unchanged) on the extension v6 state for the run (the harness enters this
    FIRST). Restores every production function on exit, even on error; `holder[SUMMARY_KEY]` gets the summary."""
    import shared_resources_planning as srp
    import p515_s44_campaign_harness as HAR
    decl = validate_settling_resettle(decl)
    if int(cap) != spec_cap(decl['cell']):
        raise RuntimeError(f'settling resettle W142 ext v6: spec cap {cap} != the cell cap {spec_cap(decl["cell"])}')
    for fname in (CYCLE_FILE, BLOCKS_FILE, CREEP_FILE, ESS_SCHEDULE_FILE, DECISION_FILE):
        if os.path.exists(os.path.join(eval_dir, fname)):
            raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ResettleStateExtV6(decl, eval_dir, int(cap), reference=load_replay_reference(decl))
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
