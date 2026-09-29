"""
P5.15 Addendum 58 Supplement, Planner task W135 -- the RE-SETTLING EXTENSION hooks: claim group 3 (ageing E, six
model-variant arms on the current ESS parameters file, minimum SoH 0.70 NOT overridden) and the pb_y2025_n5 re-run
under the settling rule v3 (7 cells). BUILT IN NEW FILES ONLY: no file pinned by the W132 stage spec
`frozen_s53_resettle_spec_v3_139d1e62.json` is edited.

WHAT THIS MODULE IS. The W132 v3 re-settling machinery, REUSED BY IMPORT (`p515_s53_w132_resettle_v3_hooks`, pinned by
139d1e62, not edited): its state (`ResettleStateV3`, subclassed here only to stamp this schema on the summary), its v3
rule adapter (`HookedRuleV3`), its nine pass-through wrappers (`make_wrappers`: W118's eight + the IPOPT-exit wrapper),
its replay-reference loader and its exit-capture source facts. What is NEW here is only the cell table (7 cells), the
declaration schema that routes those cells to this module, the preconditions checklist restated for them, and three
pure post-run readers (the G9 replacement for a model-variant entry, the floor-year reading, the trajectory equality
used by the C2_calfade consistency check).

ROUTING. The harness dispatches a `settling_resettle` declaration on its `schema` (`resettle_hooks_module`). A
declaration carrying THIS schema reaches this module ONLY after the prepared one-branch router edit
(`p515_s53_w135_harness_router.patch`) is applied -- a separate task, after W132 cell #38 has its results committed.
Until then the harness (pinned bf0a757c by 139d1e62) refuses this declaration (its W118 validator does not know the
schema), so nothing here can run by accident.

THE CELLS (launch order = CELL_ORDER; after W132's #38):
  E (ungated, first evaluations under the current configuration + the arm's `model_variant`; dynamic cap
     min(k0_run + 109, 300); criterion v3 reading alpha), candidate db77e154... (node 7, 0.25 MVA / 1.0 MWh, 2025):
       e_c3_unit     {eol 0.50, phi 1.0,   'end', on}      the C3 unit (the reference arm of every x C3 ratio)
       e_c2          {eol 0.80, phi 1.0,   'end', on}
       e_c4          {eol 0.70, phi 1.0,   'end', on}
       e_c2_calfade  {eol 0.80, phi 0.985, 'end', on}      == the current file's own ageing law (the baseline)
       e_c3_midblock {eol 0.50, phi 1.0,   'mid', on}
       e_no_ageing   {eol 0.50, phi 1.0,   'end', off}
  pb_y2025_n5_v3 (gated bitwise against its ORIGINAL S47 record 4a852725 through k0 = 110; fixed cap N_old + 100 = 219;
     criterion v3). A NEW eval key and campaign root; the W118 v2 run (ca29c5e8, Addendum 58 Ruling 2: accepted,
     flagged) stays as it is.
The key formula is unchanged: the base key already carries `model_variant` (harness `evaluation_key`), the resettle key
wraps the base key with this declaration.

Zero solves: nothing here solves or builds a model. Stdlib (+ the W132 / W118 / W105 / W101 hooks modules, themselves
stdlib-only at import) at import: the harness parent imports it for keys.
"""
import copy
import inspect
import json
import math
import os
from contextlib import contextmanager

import settling_criterion_v3 as SC3
import p515_s53_w101_settling_continuation_hooks as C101
import p515_s53_w105_settling_extension_hooks as E105
import p515_s53_w118_resettle_hooks as R
import p515_s53_w132_resettle_v3_hooks as V

SCHEMA = 'p515_s53_w135_settling_resettle_ext_v1'
DECLARATION_SCHEMA = SCHEMA         # the declaration's 'schema' key: the (patched) harness dispatches on it
OPTION_NAME = V.OPTION_NAME         # 'settling_resettle'
LABEL = ('SRP1 RE-SETTLING EXTENSION v3 (W135, frozen_s53_resettle_ext_spec) -- current production configuration (C2 '
         'ageing file with minimum SoH 0.70, tight tail); ageing E arms as MODEL VARIANTS of the ageing law (ungated '
         'first evaluations); pb_y2025_n5 replayed bitwise against its original record through its first residual pass '
         'k0 (abort on divergence); the certifying regime held after the run\'s first residual pass (AA off, tight tail '
         'on, rho frozen); settling rule v3 (reading alpha; reading gamma report-only) until it certifies or the cap; '
         'W105 captures, t_sum, and the IPOPT exit of every block of every cycle')
P_MAX = V.P_MAX
L_MONO = V.L_MONO
CAP_AFTER_N_OLD = V.CAP_AFTER_N_OLD     # 100
CAP_AFTER_K0 = V.CAP_AFTER_K0           # 109
CAP_CEILING = V.UNGATED_CAP_CEILING     # 300 (production's case-file num_max_iters)
N_TSO_BLOCKS, N_DSO_BLOCKS, N_ESSO = V.N_TSO_BLOCKS, V.N_DSO_BLOCKS, V.N_ESSO
OPTIMAL = V.OPTIMAL
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
FLOOR_SIDECAR_FILE = 'soh_floor_sidecar_baseline.jsonl'   # the harness's per-cycle SoH-floor capture (s35ref hooks)
REPO = os.path.dirname(os.path.abspath(__file__))

# ---- the instance and the file in force (W134; Addendum 58 Supplement: option (b), minimum SoH 0.70 NOT overridden) ---
ESS_PARAMS_REL = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
ESS_PARAMS_SHA256 = '39106f934bf3edbf18f01a5ef1fadfefc2f7a518706e6c8fa6d962617a312706'
FLOOR_ROW_LOWER_EXPECTED = 0.70         # the file's minimum_soh; a model_variant has no key that could change it
READBACK_RTOL = 1e-12                   # = p515_s44_campaign_harness.MODEL_VARIANT_READBACK_RTOL (checked by the checks)
MODEL_VARIANT_LABEL = 'MODEL VARIANT — not the baseline'   # = the harness's MODEL_VARIANT_LABEL (checked)
UNIT_CANDIDATE_KEY = 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a'
UNIT_NODES = {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}
UNIT_NODE = 7
UNIT_YEAR = 2025
UNIT_COHORT_INDEX = 0                   # the 2025 cohort is y_inv 0 in the instance years (2025, 2030, 2035)
INSTANCE_YEARS = (2025, 2030, 2035)     # cross-checked against the record's ageing_trajectory_terminal block years

# ---- the arms (W134's arm definitions table, as the Planner transcribed it in task W135) ------------------------------
ARMS = {
    'C3_unit': {'eol_retention_r': 0.5, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end',
                'ageing_enabled': True},
    'C2': {'eol_retention_r': 0.8, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end',
           'ageing_enabled': True},
    'C4': {'eol_retention_r': 0.7, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end',
           'ageing_enabled': True},
    'C2_calfade': {'eol_retention_r': 0.8, 'calendar_retention_per_year': 0.985, 'available_energy_soh_point': 'end',
                   'ageing_enabled': True},
    'C3_midblock': {'eol_retention_r': 0.5, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'mid',
                    'ageing_enabled': True},
    'no_ageing': {'eol_retention_r': 0.5, 'calendar_retention_per_year': 1.0, 'available_energy_soh_point': 'end',
                  'ageing_enabled': False},
}
# The arm whose variant equals the current file's own ageing law (C2 calibration 0.80, phi_cal 0.985, 'end', on).
BASELINE_EQUIVALENT_ARM = 'C2_calfade'

_RES = os.path.join('data', 'SRP1', 'Results')
_S45A1A = os.path.join(_RES, 'P515S45', 'campaign_s45_a1a')
_S46 = os.path.join(_RES, 'P515S46', 'campaign_s46_ageing')
_S47PB = os.path.join(_RES, 'P515S47', 'campaign_s47_phase_b')
_S46_SPEC = 'campaign_spec_s46_ageing_6544c17e.json'
# The original (0.50-era) records are PROVENANCE only for the E cells (ungated first evaluations: nothing is replayed
# against them); for pb_y2025_n5_v3 the original is the replay reference.
CELLS = {
    'e_c3_unit': {'item': 'E', 'arm': 'C3_unit', 'gated': False, 'orig_root': _S45A1A,
        'orig_spec': 'campaign_spec_s45_a1a_c71f52ee.json', 'orig_campaign_id': 's45_a1a',
        'orig_eval_key': '7eb1ce62c2509f54201191ed7220a7e25f06ca8830cb73e5afe2a14101a76b20',
        'orig_eval_dir': '7eb1ce62c2509f54_n7_4h_e1', 'orig_label': 'n7_4h_e1', 'orig_model_variant': None,
        'orig_cycles': 125, 'N_old': None, 'k0': None, 'original_lapses_after_k0': None, 'cap_ceiling': CAP_CEILING,
        'per_cycle_record_sha256': 'ace6ba2f586759957d7494cdd8286f2d12807b13416940190d0df86cb36bc8bb'},
    'e_c2': {'item': 'E', 'arm': 'C2', 'gated': False, 'orig_root': _S46, 'orig_spec': _S46_SPEC,
        'orig_campaign_id': 's46_ageing',
        'orig_eval_key': 'c6b53015fcf65e248bcf936d7e4fd9f61f1e09ebd8f78b7e4dda55d6736e7b78',
        'orig_eval_dir': 'c6b53015fcf65e24_n7_4h_e1_c2', 'orig_label': 'n7_4h_e1_C2', 'orig_model_variant': 'C2',
        'orig_cycles': 119, 'N_old': None, 'k0': None, 'original_lapses_after_k0': None, 'cap_ceiling': CAP_CEILING,
        'per_cycle_record_sha256': '2f3e2f428098537153c8e866e6cf5706aa65e3bd85aa7df36bb18d8752b0e046'},
    'e_c4': {'item': 'E', 'arm': 'C4', 'gated': False, 'orig_root': _S46, 'orig_spec': _S46_SPEC,
        'orig_campaign_id': 's46_ageing',
        'orig_eval_key': '65a5da775d1ff5b270518f9a90734bbf6e3894c98510f0af16dc81f3346f6850',
        'orig_eval_dir': '65a5da775d1ff5b2_n7_4h_e1_c4', 'orig_label': 'n7_4h_e1_C4', 'orig_model_variant': 'C4',
        'orig_cycles': 119, 'N_old': None, 'k0': None, 'original_lapses_after_k0': None, 'cap_ceiling': CAP_CEILING,
        'per_cycle_record_sha256': 'fb93587fa04d86be15511ecb28b31479fc7336a565958a1991de71b11d9bc0a1'},
    'e_c2_calfade': {'item': 'E', 'arm': 'C2_calfade', 'gated': False, 'orig_root': _S46, 'orig_spec': _S46_SPEC,
        'orig_campaign_id': 's46_ageing',
        'orig_eval_key': '98e2857016a16d1c09aea36acb06929baed718ffc7c01ca53de1e24d4f3e9cfd',
        'orig_eval_dir': '98e2857016a16d1c_n7_4h_e1_c2_calfade', 'orig_label': 'n7_4h_e1_C2_calfade',
        'orig_model_variant': 'C2_calfade', 'orig_cycles': 121, 'N_old': None, 'k0': None,
        'original_lapses_after_k0': None, 'cap_ceiling': CAP_CEILING,
        'per_cycle_record_sha256': '8392100874ec5ace1d8dacdcc640293fc06bb5bd710a4726426f5128ef0b3b5b'},
    'e_c3_midblock': {'item': 'E', 'arm': 'C3_midblock', 'gated': False, 'orig_root': _S46, 'orig_spec': _S46_SPEC,
        'orig_campaign_id': 's46_ageing',
        'orig_eval_key': 'ed4a1acc7059784d03b2847e7639bfae00e80ee0a8487520a9e532fbdbf3967a',
        'orig_eval_dir': 'ed4a1acc7059784d_n7_4h_e1_c3_midblock', 'orig_label': 'n7_4h_e1_C3_midblock',
        'orig_model_variant': 'C3_midblock', 'orig_cycles': 123, 'N_old': None, 'k0': None,
        'original_lapses_after_k0': None, 'cap_ceiling': CAP_CEILING,
        'per_cycle_record_sha256': '18ce54dcff5023aa4476326b27e7bd58842c34f6be75a3d1fe00775e8db25f63'},
    'e_no_ageing': {'item': 'E', 'arm': 'no_ageing', 'gated': False, 'orig_root': _S46, 'orig_spec': _S46_SPEC,
        'orig_campaign_id': 's46_ageing',
        'orig_eval_key': '06f092d164f1381936520acd48d0b8ac4ec8c739adbd32557e7158a46c1276e1',
        'orig_eval_dir': '06f092d164f13819_n7_4h_e1_no_ageing', 'orig_label': 'n7_4h_e1_no_ageing',
        'orig_model_variant': 'no_ageing', 'orig_cycles': 118, 'N_old': None, 'k0': None,
        'original_lapses_after_k0': None, 'cap_ceiling': CAP_CEILING,
        'per_cycle_record_sha256': '19de84ca2bb4fc308cbe13344805246c0b66ab858801737b8519cc407385c0ce'},
    'pb_y2025_n5_v3': {'item': 'C', 'arm': None, 'gated': True, 'orig_root': _S47PB,
        'orig_spec': 'campaign_spec_s47_phase_b_8cfa264e.json', 'orig_campaign_id': 's47_phase_b',
        'orig_eval_key': '4a8527254a85b9cca48e1eb872b35ed9f3d5209f84002e2c255516b8af5b304a',
        'orig_eval_dir': '4a8527254a85b9cc_y2025__n5_p0_25_e0_5', 'orig_label': 'y2025__n5_p0.25_e0.5',
        'orig_model_variant': None, 'orig_cycles': 119, 'N_old': 119, 'k0': 110, 'original_lapses_after_k0': [],
        'cap_ceiling': CAP_CEILING,
        'per_cycle_record_sha256': 'c84067449b6c5908b8c1187936d0516ba953c89bacc0255fb746c04437e2d9e7',
        'v2_run_stays': {'campaign_root': os.path.join(_RES, 'P515S53', 'w118_resettle',
                                                       'campaign_s53_w118_resettle_r2_pb_y2025_n5'),
                         'eval_dir': 'ca29c5e818366a6e_pb_y2025_n5',
                         'status': ('the W118 r2 run (settling rule v2, certified at 167 with a non-Optimal accepted '
                                    'solve at 167) stays as it is: Addendum 58 Ruling 2, accepted, flagged')}},
}
# Launch order (Planner task W135): C3 unit first (every x C3 ratio needs it), then C2, C4, C2_calfade, C3_midblock,
# no_ageing, then pb_y2025_n5_v3 -- all AFTER W132's cell #38.
CELL_ORDER = ('e_c3_unit', 'e_c2', 'e_c4', 'e_c2_calfade', 'e_c3_midblock', 'e_no_ageing', 'pb_y2025_n5_v3')
E_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['item'] == 'E')
GATED_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['gated'])
UNGATED_CELLS = tuple(c for c in CELL_ORDER if not CELLS[c]['gated'])
GROUP_OF_ITEM = {'E': 3, 'C': 4}
CELL_OF_ARM = {CELLS[c]['arm']: c for c in E_CELLS}

# ---- W132's last cell: this extension starts only after its results are committed (Planner task W135) ----------------
W132_ROOT_REL = os.path.join(_RES, 'P515S53', 'w132_resettle_v3')
W132_STAGE_SPEC_REL = os.path.join(W132_ROOT_REL, 'frozen_s53_resettle_spec_v3_139d1e62.json')
W132_STAGE_SPEC_SHA256 = '139d1e62339248df16b6fbf8019de9fb8569285f658b7fa7b1cd1ed71d060136'
W132_LAST_CELL = 'l_195156fa'
W132_CAMPAIGN_ID_PREFIX = 's53_w132_resettle_v3_'


def w132_last_cell_files_rel():
    root = os.path.join(W132_ROOT_REL, f'campaign_{W132_CAMPAIGN_ID_PREFIX}{W132_LAST_CELL}')
    return (os.path.join(root, 'campaign_results.json'), os.path.join(root, 'campaign_manifest_sha256.json'))


def original_eval_dir(cell):
    return os.path.join(CELLS[cell]['orig_root'], 'evals', CELLS[cell]['orig_eval_dir'])


def reference_path(cell):
    return os.path.join(original_eval_dir(cell), 'per_cycle_record.jsonl')


def arm_variant(cell):
    """The cell's validated-form model_variant (a new dict), or None (pb)."""
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
    """The harness cap (production's num_max_iters): gated N_old + 100; ungated its ceiling (the rule stops earlier)."""
    r = cap_rule(cell)
    return r['cap'] if r['kind'] == 'fixed' else r['ceiling']


def declaration_for(cell):
    """The one valid declaration of a cell (W132's declaration shape + schema, arm, model_variant, floor)."""
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
        'settling_rule': V.settling_rule_declaration(),
        'record_all_blocks': True,
        'captures': {'q_decomposition': True, 'ess_schedule_movement': True, 'boyd_full': True, 't_sum': True,
                     'ipopt_exit_by_block': True, 'soh_floor_sidecar': True,
                     'ageing_trajectory_terminal': c['arm'] is not None},
        'expected_blocks': {'tso': N_TSO_BLOCKS, 'dso': N_DSO_BLOCKS, 'esso': N_ESSO},
        'replay_reference': ({'per_cycle_record': reference_path(cell), 'sha256': c['per_cycle_record_sha256'],
                              'n_cycles': c['N_old'], 'gated_through_cycle': c['k0'],
                              'original_eval_key': c['orig_eval_key']} if gated else None),
        'abort_on_replay_divergence': bool(gated),
    }


def is_w135_declaration(value):
    return isinstance(value, dict) and value.get('schema') == DECLARATION_SCHEMA


def validate_settling_resettle(value):
    """None = not declared. Otherwise the value must equal `declaration_for(value['cell'])` EXACTLY (an `early_stop`
    key is refused by name). Returns a new dict. Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} (W135) must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} (W135) must NOT carry {forbidden} (the settling rule is the only stop before '
                         f'the cap)')
    if value.get('schema') != DECLARATION_SCHEMA:
        raise ValueError(f'{OPTION_NAME} (W135): schema must be {DECLARATION_SCHEMA!r}; got {value.get("schema")!r}')
    cell = value.get('cell')
    if cell not in CELLS:
        raise ValueError(f'{OPTION_NAME} (W135).cell must be one of {sorted(CELLS)}; got {cell!r}')
    want = declaration_for(cell)
    if set(value) != set(want):
        raise ValueError(f'{OPTION_NAME} (W135) must have exactly {sorted(want)}; got {sorted(value)}')
    bad = sorted(k for k in want if json.dumps(value[k], sort_keys=True) != json.dumps(want[k], sort_keys=True))
    if bad:
        raise ValueError(f'{OPTION_NAME} (W135) differs from the W135 declaration of {cell} on {bad}')
    return copy.deepcopy(want)


def load_replay_reference(decl):
    """W132's loader (generic in the declaration; reused unchanged)."""
    return V.load_replay_reference(decl)


# ======================================================================================================================
#  capture-path assertion (rule eleven, BEFORE any solve) -- restated for the W135 cells from W132's items
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


def spec_entry_checks(decl, spec):
    """The campaign spec the child runs holds exactly this cell's entry: its settling_resettle IS the declaration, its
    model_variant IS the declared arm variant (E) or absent (pb), the variant label sits at spec and entry level (E),
    and the spec's declared ESS ageing baseline keeps minimum_soh = the declared floor (NOT overridden). Pure."""
    cands = (spec or {}).get('candidates') or []
    entry = cands[0] if len(cands) == 1 else {}
    cfg = (spec or {}).get('configuration') or {}
    mv = decl['model_variant']
    ess = cfg.get('ess_ageing_baseline') or {}
    out = {
        'w135_spec_holds_one_entry': len(cands) == 1,
        'w135_entry_label_is_the_cell': entry.get('label') == decl['cell'],
        'w135_entry_settling_resettle_is_the_declaration': (
            json.dumps(entry.get('settling_resettle'), sort_keys=True) == json.dumps(decl, sort_keys=True)),
        'w135_entry_model_variant_is_the_declared_arm': (
            json.dumps(entry.get('model_variant'), sort_keys=True) == json.dumps(mv, sort_keys=True)),
        'w135_variant_label_at_spec_and_entry_level_iff_variant': (
            (mv is None and 'model_variant_label' not in entry and 'model_variant_label' not in (spec or {}))
            or (mv is not None and entry.get('model_variant_label') == MODEL_VARIANT_LABEL
                and (spec or {}).get('model_variant_label') == MODEL_VARIANT_LABEL)),
        'w135_ess_ageing_baseline_declared': bool(ess),
        'w135_floor_not_overridden_minimum_soh_is_the_declared_floor': (
            ess.get('minimum_soh') == decl['floor_row_lower_expected'] == FLOOR_ROW_LOWER_EXPECTED),
        'w135_ess_params_file_pinned_as_expected': ((cfg.get('ess_params_file') or {}).get('sha256')
                                                    == ESS_PARAMS_SHA256),
        'w135_no_flex_multiplier_premium_or_derived_instance': not any(
            k in entry for k in ('flex_price_multiplier', 'interface_deviation_premium')) and not cfg.get(
            'derived_instance'),
        'w135_candidate_is_the_unit_for_e_cells': (decl['item'] != 'E' or entry.get('key') == UNIT_CANDIDATE_KEY),
    }
    return out


def assert_resettle_preconditions(decl, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child; and the parent's copy): W132's items restated for a W135 cell
    (the spec cap is the cell's and within its ceiling; the rule constants are settling_criterion_v3's), the signatures
    of the nine wrapped functions, W105's capture paths, the t_sum source facts, the exit-capture facts (W132's own
    function), the W118 state attributes carried by the reused v3 state, and the W135 entry checks
    (`spec_entry_checks`). Raises on any failure; returns the checklist."""
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
        'f_rule_is_the_w132_v3_declaration': rule == V.settling_rule_declaration(),
        'f_rule_constants_equal_settling_criterion_v3': (rule['tau'] == SC3.TAU and rule['eps0'] == SC3.TAU / 100.0
                                                         and rule['gap_bound'] == SC3.TAU / 2.0
                                                         and rule['p_max'] == 30 and rule['l_mono'] == 60
                                                         and rule['version'] == 3 and rule['retry_tier'] is None),
        'f_rule_class_callable': callable(getattr(SC3, 'SettlingRuleV3', None)),
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
        'state_carries_every_w118_state_attribute': V._state_attribute_superset(),
        'state_class_is_the_w132_v3_state': issubclass(ResettleStateW135, V.ResettleStateV3),
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
    for k, v in spec_entry_checks(decl, spec).items():
        checks[k] = bool(v)
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W135 settling-resettle extension preconditions fail (before any solve): {failing}')
    return checks


# ======================================================================================================================
#  the state (W132's, reused) and the install
# ======================================================================================================================
class ResettleStateW135(V.ResettleStateV3):
    """W132's v3 state, reused as is (constructor, rule adapter, exit capture, summary); only the summary's schema
    names this extension, so a record says which declaration schema produced it."""

    def summary(self):
        s = super().summary()
        s['schema'] = SCHEMA
        s['state_class'] = ('p515_s53_w132_resettle_v3_hooks.ResettleStateV3 (reused by import; the summary schema '
                            'stamped by p515_s53_w135_resettle_ext_hooks.ResettleStateW135)')
        s['arm'] = self.decl.get('arm')
        s['model_variant_declared'] = self.decl.get('model_variant')
        return s


@contextmanager
def settling_resettle_hooks(eval_dir, decl, holder, cap):
    """Install W132's nine wrappers for the run (the harness enters this FIRST, so these wrap production directly and
    every harness capture hook wraps them). Restores every production function on exit, even on error;
    `holder[SUMMARY_KEY]` gets the summary. Mirrors `V.settling_resettle_hooks` with this module's declaration."""
    import shared_resources_planning as srp
    import p515_s44_campaign_harness as HAR
    decl = validate_settling_resettle(decl)
    if int(cap) != spec_cap(decl['cell']):
        raise RuntimeError(f'settling resettle W135: spec cap {cap} != the cell cap {spec_cap(decl["cell"])}')
    for fname in (CYCLE_FILE, BLOCKS_FILE, CREEP_FILE, ESS_SCHEDULE_FILE, DECISION_FILE):
        if os.path.exists(os.path.join(eval_dir, fname)):
            raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ResettleStateW135(decl, eval_dir, int(cap), reference=load_replay_reference(decl))
    originals = {name: getattr(srp, name) for name in WRAPPED}
    wrappers = V.make_wrappers(st, originals, HAR.ipopt_exit_class, srp_module=srp,
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
#  pure post-run readers (stdlib; used by the campaign's gates and scorer and by the zero-solve checks)
# ======================================================================================================================
def closed_form_expected(model_variant, cycles_n, reference_dod_d):
    """The constants a model variant MUST produce in the built ESSO model (the harness's `model_variant_expected`,
    restated from the declaration and the file's (N, D) alone): k = N * D / (-ln R) (None when ageing is off), phi
    (1.0 when off), the SoH point, the D-row form."""
    enabled = model_variant['ageing_enabled']
    k = cycles_n * reference_dod_d / (-math.log(model_variant['eol_retention_r']))
    return {'k': k if enabled else None, 'k_if_enabled': k,
            'phi_cal_in_model': model_variant['calendar_retention_per_year'] if enabled else 1.0,
            'available_energy_soh_point': model_variant['available_energy_soh_point'], 'ageing_enabled': enabled,
            'd_row_form': 'D * 2kE == 365 n avg' if enabled else 'D == 0',
            'formula': f'k = N * D / (-ln R) = {cycles_n} * {reference_dod_d} / (-ln {model_variant["eol_retention_r"]})'}


def _close(a, b, rtol=READBACK_RTOL):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rtol, abs_tol=0.0)


def variant_readback_gate(rec, decl, nodes=('5', '7', '9')):
    """THE G9 REPLACEMENT for a model-variant entry (the W86 G9 reads `ess_ageing_readback_*`, which the harness does
    not compute for a variant entry). Holds iff:
      (a) the record's variant and label are the declaration's;
      (b) PRE-RUN (probe models) and TERMINAL (clones of the run's own ESSO models) variant readbacks: all_match, every
          node present, every node's per-check dict all True, and every node's read-back k / phi / SoH point / D-row
          form equal the CLOSED FORMS of the declared variant with the file's own (N, D) (k, phi to rtol 1e-12);
      (c) every node's read-back floor-row lower bound == the declared floor 0.70 (rtol 1e-12; exact equality recorded
          beside), pre-run and terminal;
      (d) the file-vs-declaration checks: the child's `verify_ess_ageing_in_child` checks all True, the file hashed in
          the child == the pin, the loaded minimum_soh == the declared one == 0.70;
      (e) the variant took effect (`apply_model_variant` checks all True) and left every other ageing constant --
          soh_min included -- unchanged (its own check), and every ESS carries soh_min 0.70 after the apply.
    Returns (ok, detail). Pure."""
    mv = decl['model_variant']
    floor = decl['floor_row_lower_expected']
    ev = rec.get('ess_ageing_verified_pre_run') or {}
    loaded = ev.get('loaded') or {}
    cal = loaded.get('calibration') or {}
    parts, detail = {}, {}
    parts['a_record_variant_is_the_declaration'] = (json.dumps(rec.get('model_variant'), sort_keys=True)
                                                    == json.dumps(mv, sort_keys=True))
    parts['a_record_label'] = rec.get('model_variant_label') == MODEL_VARIANT_LABEL
    cf = None
    if cal.get('cycles_n') is not None and cal.get('reference_dod_d') is not None and mv is not None:
        cf = closed_form_expected(mv, cal['cycles_n'], cal['reference_dod_d'])
    detail['closed_form'] = cf
    for phase in ('pre_run', 'terminal'):
        rb = rec.get(f'model_variant_readback_{phase}') or {}
        per = rb.get('per_node') or {}
        node_ok, floors = {}, {}
        for n in nodes:
            e = per.get(n) or {}
            r = e.get('readback') or {}
            c = e.get('checks') or {}
            fl = r.get('floor_row_lower')
            floors[n] = {'floor_row_lower': fl, 'exact_equal': fl == floor}
            node_ok[n] = bool(c) and all(v is True for v in c.values()) and cf is not None and (
                _close(r.get('k'), cf['k']) and _close(r.get('phi_cal_in_model'), cf['phi_cal_in_model'])
                and r.get('available_energy_soh_point') == cf['available_energy_soh_point']
                and r.get('d_row_form') == cf['d_row_form'])
            parts[f'c_{phase}_floor_row_lower_is_{floor}_node_{n}'] = fl is not None and _close(fl, floor)
        exp = rb.get('expected') or {}
        parts[f'b_{phase}_all_match'] = rb.get('all_match') is True
        parts[f'b_{phase}_every_node_present'] = sorted(per) == sorted(nodes)
        parts[f'b_{phase}_every_node_checks_and_closed_forms'] = bool(node_ok) and all(node_ok.values())
        parts[f'b_{phase}_harness_expected_equals_closed_form'] = cf is not None and (
            _close(exp.get('k'), cf['k']) and _close(exp.get('phi_cal_in_model'), cf['phi_cal_in_model'])
            and exp.get('available_energy_soh_point') == cf['available_energy_soh_point']
            and exp.get('d_row_form') == cf['d_row_form'] and exp.get('ageing_enabled') == cf['ageing_enabled'])
        detail[phase] = {'node_ok': node_ok, 'floor_row_lower': floors}
    ch = ev.get('checks') or {}
    parts['d_file_vs_declaration_checks_all_true'] = bool(ch) and all(v is True for v in ch.values())
    parts['d_file_hashed_in_child_is_the_pin'] = rec.get('ess_params_sha256_in_child') == ESS_PARAMS_SHA256
    parts['d_loaded_minimum_soh_is_the_floor'] = (loaded.get('minimum_soh') == floor
                                                  == (rec.get('ess_ageing_baseline') or {}).get('minimum_soh'))
    ap = rec.get('model_variant_applied_in_child') or {}
    ac = ap.get('checks') or {}
    parts['e_variant_applied_checks_all_true'] = bool(ac) and all(v is True for v in ac.values())
    after = ((ap.get('after') or {}).get('per_ess')) or []
    parts['e_every_ess_soh_min_is_the_floor_after_the_apply'] = bool(after) and all(
        e.get('soh_min') == floor for e in after)
    detail['file_vs_declaration_checks'] = ch
    detail['applied_checks'] = ac
    return all(v is True for v in parts.values()), {'parts': parts, **detail}


def floor_year_reading(sidecar_lines, cycle, node=UNIT_NODE, cohort_index=UNIT_COHORT_INDEX, years=INSTANCE_YEARS):
    """THE FLOOR YEAR of an arm, read at `cycle` (k*, or the end cycle when uncertified) from the harness's per-cycle
    SoH-floor sidecar (`soh_floor_sidecar_baseline.jsonl`; `active` = |SoH - soh_min| <= 1e-6, set by the s35ref capture
    hooks): the first block year with an ACTIVE floor row of `node` for the cohort `cohort_index` (2025). None when no
    block's floor row is active. Returns (reading, capture_ok). Pure."""
    line = next((x for x in sidecar_lines if x.get('cycle') == cycle), None)
    if line is None:
        return {'cycle': cycle, 'error': 'no sidecar line for the cycle'}, False
    ents = sorted((e for e in line.get('entries') or [] if e.get('node_id') == node
                   and str(e.get('y_inv')) == str(cohort_index)), key=lambda e: int(e['y']))
    ys = [int(e['y']) for e in ents]
    ok = ys == list(range(cohort_index, len(years))) and all(isinstance(e.get('active'), bool) for e in ents)
    active = [int(e['y']) for e in ents if e.get('active') is True]
    first = min(active) if active else None
    return {'cycle': cycle, 'node': node, 'cohort_year': years[cohort_index],
            'floor_year': years[first] if first is not None else None,
            'active_block_years': [years[y] for y in active],
            'per_block': [{'block_year': years[int(e['y'])], 'soh_end': e.get('es_soh_per_unit_cumul'),
                           'soh_min': e.get('soh_min'), 'active': e.get('active'), 'dual': e.get('dual'),
                           'efc_per_day': e.get('efc_per_day')} for e in ents],
            'rule': ('first block year whose node-7 floor row (cohort 2025) is active at the cycle; active = |SoH - '
                     'soh_min| <= 1e-6 (the s35ref capture)')}, ok


def trajectory_equality(run_rows, ref_rows, through, field='gross_operational_cost'):
    """The C2_calfade consistency criterion: `field` (Q) at every cycle 1..through equal EXACTLY (floats compared as
    JSON text, so a one-ulp difference counts); every other shared field compared the same way, report-only. Returns
    {'reproduced', 'through', 'first_difference', ...}. Pure."""
    run = {r['cycle']: r for r in run_rows}
    ref = {r['cycle']: r for r in ref_rows}
    first, first_any = None, None
    for k in range(1, through + 1):
        a, b = run.get(k), ref.get(k)
        if a is None or b is None:
            first = first or {'cycle': k, 'missing_in': 'run' if a is None else 'reference'}
            break
        if first is None and json.dumps(a.get(field)) != json.dumps(b.get(field)):
            first = {'cycle': k, 'run': a.get(field), 'reference': b.get(field),
                     'difference_run_minus_reference': (a.get(field) - b.get(field)) if (
                         a.get(field) is not None and b.get(field) is not None) else None}
        if first_any is None:
            diff = sorted(f for f in set(a) & set(b) if json.dumps(a.get(f), sort_keys=True)
                          != json.dumps(b.get(f), sort_keys=True))
            if diff:
                first_any = {'cycle': k, 'fields_differing': diff}
        if first is not None and first_any is not None:
            break
    return {'reproduced': first is None, 'field': field, 'through': through, 'first_difference': first,
            'first_difference_any_shared_field_report_only': first_any}
