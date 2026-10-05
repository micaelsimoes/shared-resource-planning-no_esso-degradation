"""
P5.15 Addendum 64, Planner task W155 -- the hooks of the FOUR Addendum 64 cells: the m = 1.75 flexibility-ladder pair
(x = 0 and the node-7 unit) and the minimum-SoH 0.50 row (with its identity-value neutrality cell at 0.70), under the
settling rule v6 exactly as the v6 cells and the W142 extension use it.

NEW FILE. Nothing pinned by the v6 stage spec (96c23404) or the extension-v6 stage spec (84775dc4) is edited: this module
REUSES BY IMPORT the v6 machinery (`p515_s53_w142_resettle_v6_hooks`: the v6 rule declaration, the v6 state on W139's v5
state, the nine W139 wrappers, the capture checklists) and the W142 extension module
(`p515_s53_w142_resettle_ext_v6_hooks`: the arm table, the unit instance constants, the floor-year reader, the trajectory
equality). What is NEW here:
  * the cell table (4 cells) and the declaration schema `p515_s53_w155_a64_settling_resettle_v1`, which the harness
    router routes here (the one-branch router patch, W155, its own commit);
  * the keyed `minimum_soh` of the two SoH cells, carried INSIDE the settling_resettle declaration (the harness's
    `validate_model_variant` refuses extra keys, so the model_variant stays the standard 4-key C2_calfade variant);
  * THE SoH-FLOOR PLUMBING (Advisor H1-H3), installed by `settling_resettle_hooks` for a SoH cell only:
      H1  the harness's `apply_model_variant` (in every loaded copy of the harness module -- the child runs it as
          `__main__`) is wrapped: the original runs first, then the loaded ageing parameters' `soh_min` is set to the
          declared value and production's own `EnergyStorageAgeingParameters.apply_to` is re-run on every
          SharedEnergyStorage (`ess.soh_min` is NEVER written directly), verified, and recorded under
          `minimum_soh_applied` in the returned record;
      H2  `p515_g_g1_g4_admm_gates.s38_pf_capture_hooks` is wrapped to CAPTURE the reference of the shared
          `floor_rows_by_node` dict when the harness enters it; after the apply its `soh_min` entries are set to the
          declared value IN PLACE, so the harness's floor-row identity check (`floor_rows_variant == expected_floor_rows`,
          harness ~3003) passes BY EQUALITY on probes built after the apply (0.5 on both sides), never by bypass;
      H3  the floor sidecar (gates ~3602-3610) reads `soh_min` / `active` from that same dict, so its lines carry the
          declared value;
      and an ARMED fail-fast: `p515_g_g1_g4_admm_gates.esso_capture_hooks` (entered by `run_admm_arm` after the
      configuration hook and BEFORE the first solve) is wrapped to RAISE unless the apply ran exactly once, the dict
      was captured and updated, and every ESS of the run's planning object carries the declared soh_min.
    At the identity value 0.70 (g070_neutrality) every step runs and changes no value: that cell is the NEUTRALITY GATE
    of the router patch and of these wrappers (bitwise against 3f084f2f through 172). It CANNOT prove that 0.50 takes
    effect -- the 0.50 cell's own gates (the readback floor 0.5, the sidecar at 0.5, the trajectory diverging from
    3f084f2f by cycle 5) do.

THE CELLS (launch order CELL_ORDER; all UNGATED first evaluations in the E-cell shape: no replay, dynamic cap
min(k0_run + 109, 300), holds after the run's first residual pass, criterion v6, gap clause tau/2):
  #1 g070_neutrality  the C2_calfade unit (y2025 n7 0.25 MVA / 1 MWh; model_variant R 0.8 / phi 0.985 / end / on) through
                      the new SoH path at minimum_soh 0.70; post-run gate bitwise vs 3f084f2f through 172 (STOP on failure)
  #2 h_x0_m175        x = 0 at flex_price_multiplier 1.75 (keyed through the base key); no model_variant
  #3 h_unit_m175      the node-7 unit (y2025, 0.25 MVA / 1 MWh, I = 317,957.01) at m 1.75; no model_variant
  #4 e_soh050         the C2_calfade unit at minimum_soh 0.50

Zero solves: nothing here solves; nothing here builds a model at import. Stdlib (+ the hooks modules it imports,
themselves stdlib-only at import) at import: the harness parent imports it for keys.
"""
import copy
import functools
import inspect
import json
import math
import os
import sys
from contextlib import contextmanager

import p515_s53_w101_settling_continuation_hooks as C101
import p515_s53_w105_settling_extension_hooks as E105
import p515_s53_w118_resettle_hooks as R
import p515_s53_w132_resettle_v3_hooks as V
import p515_s53_w142_resettle_v6_hooks as V6
import p515_s53_w142_resettle_ext_v6_hooks as W

SCHEMA = 'p515_s53_w155_a64_settling_resettle_v1'
DECLARATION_SCHEMA = SCHEMA
OPTION_NAME = V.OPTION_NAME
LABEL = ('SRP1 ADDENDUM 64 CELLS (W155, frozen_s53_a64_cells_spec) -- current production configuration (C2 ageing file '
         'with minimum SoH 0.70, tight tail); the m = 1.75 flexibility-ladder pair (keyed multiplier) and the minimum-SoH '
         'row (keyed minimum_soh in this declaration, applied after the standard C2_calfade model variant through '
         'production\'s apply_to; the shared floor-row dict carried to the declared value); ungated first evaluations; '
         'the certifying regime held after the run\'s first residual pass (AA off, tight tail on, rho frozen); settling '
         'rule v6 until it certifies or the cap; W105 captures, t_sum, the IPOPT exit, attempt tier and final metrics of '
         'every block, the SoH-floor sidecar')
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
HARNESS_FILE = os.path.join(REPO, 'p515_s44_campaign_harness.py')

# ---- the instance and the file in force (the W135 / W142 extension's) -------------------------------------------------
ESS_PARAMS_REL = W.ESS_PARAMS_REL
ESS_PARAMS_SHA256 = W.ESS_PARAMS_SHA256
FILE_MINIMUM_SOH = W.FLOOR_ROW_LOWER_EXPECTED          # 0.70: the file's minimum_soh (NOT edited; the spec declares it)
READBACK_RTOL = W.READBACK_RTOL
MODEL_VARIANT_LABEL = W.MODEL_VARIANT_LABEL
UNIT_CANDIDATE_KEY = W.UNIT_CANDIDATE_KEY
X0_CANDIDATE_KEY = '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57'   # S49 x0_m1p5 key (checks O)
UNIT_NODES = W.UNIT_NODES
X0_NODES = {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}
UNIT_NODE = W.UNIT_NODE
UNIT_YEAR = W.UNIT_YEAR
UNIT_COHORT_INDEX = W.UNIT_COHORT_INDEX
INSTANCE_YEARS = W.INSTANCE_YEARS
SOH_ARM = W.BASELINE_EQUIVALENT_ARM                   # 'C2_calfade' == the file's own ageing law
FLEX_M = 1.75
UNIT_I = 317957.0085035586                            # W117 I_other of the unit (I = 317,957.01)
FLOOR_ROWS_PER_NODE = 6                               # cohorts 2025 (3 blocks), 2030 (2), 2035 (1)

_RES = os.path.join('data', 'SRP1', 'Results')
_S49 = os.path.join(_RES, 'P515S49', 'campaign_s49_flex_ladder')
_S49_SPEC = 'campaign_spec_s49_flex_ladder_2203f6c1.json'
_P53 = os.path.join(_RES, 'P515S53')
# The settled unit 3f084f2f (W101; certified at 172): the 0.70 reference of the SoH row and of the neutrality gate.
UNIT_REF_EVAL_DIR = os.path.join(_P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_n7_4h_e1', 'evals',
                                 '3f084f2ffaeef2b7_n7_4h_e1')
UNIT_REF = {'eval_key_prefix': '3f084f2f', 'reference_id': 'bd504ecf', 'k_star': 172,
            'per_cycle_record_sha256': 'fd8aad17f8a8a40e1ca73a2af2d5edeeee83e2d3265f2bdad8c38c68fb41b324'}
X0_REF = {'eval_key_prefix': 'd110bd1a', 'reference_id': '7aa017f0', 'k_star': 181}

# ---- the cells -----------------------------------------------------------------------------------------------------
# `source`: where the candidate (canonical + key) is taken from -- PROVENANCE ONLY (every cell is a first evaluation;
# nothing is replayed against these records).
_E_SOURCE = {'root': W.CELLS['e_c2_calfade']['orig_root'], 'spec': W.CELLS['e_c2_calfade']['orig_spec'],
             'eval_key': W.CELLS['e_c2_calfade']['orig_eval_key'], 'label': W.CELLS['e_c2_calfade']['orig_label']}
_H_X0_SOURCE = {'root': _S49, 'spec': _S49_SPEC,
                'eval_key': 'aa8a76d71ae49f134831ae5bf40579285249b228f21ec66556eb43b04f4dd1a1', 'label': 'x0_m1p5'}
_H_UNIT_SOURCE = {'root': _S49, 'spec': _S49_SPEC,
                  'eval_key': 'f9eae48ff6133f6ca988cb14f511e1578fecd8f8c264675f705d3fdda7defbda',
                  'label': 'n7_4h_e1_m1p5'}


def _cell(item, kind, arm, minimum_soh, flex, source, nodes, key, role):
    return {'item': item, 'kind': kind, 'arm': arm, 'minimum_soh': minimum_soh, 'flex_price_multiplier': flex,
            'gated': False, 'N_old': None, 'k0': None, 'original_lapses_after_k0': None, 'cap_ceiling': CAP_CEILING,
            'source': source, 'nodes': nodes, 'investment_year': UNIT_YEAR, 'candidate_key': key, 'role': role}


CELLS = {
    'g070_neutrality': _cell('E', 'soh', SOH_ARM, 0.70, None, _E_SOURCE, UNIT_NODES, UNIT_CANDIDATE_KEY,
                             'NEUTRALITY GATE of the router patch and the SoH wrappers: the C2_calfade unit through the '
                             'new SoH path at the identity value 0.70; must reproduce 3f084f2f bitwise through 172. At '
                             'the identity value it CANNOT prove that 0.50 takes effect.'),
    'h_x0_m175': _cell('H', 'flex', None, None, FLEX_M, _H_X0_SOURCE, X0_NODES, X0_CANDIDATE_KEY,
                       'Q(0) at m = 1.75 (Q(0) depends on m: apply_flex_price_multiplier, harness ~2676-2728)'),
    'h_unit_m175': _cell('H', 'flex', None, None, FLEX_M, _H_UNIT_SOURCE, UNIT_NODES, UNIT_CANDIDATE_KEY,
                         'the node-7 unit at m = 1.75 (value - I against h_x0_m175)'),
    'e_soh050': _cell('E', 'soh', SOH_ARM, 0.50, None, _E_SOURCE, UNIT_NODES, UNIT_CANDIDATE_KEY,
                      'the C2_calfade unit at minimum_soh 0.50 (value against the x0 Q181 d110bd1a; Delta value '
                      'against the settled 0.70 unit 3f084f2f)'),
}
CELL_ORDER = ('g070_neutrality', 'h_x0_m175', 'h_unit_m175', 'e_soh050')
SOH_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['kind'] == 'soh')
FLEX_CELLS = tuple(c for c in CELL_ORDER if CELLS[c]['kind'] == 'flex')
GATED_CELLS = ()
UNGATED_CELLS = CELL_ORDER
NEUTRALITY_CELL = 'g070_neutrality'
SOH050_CELL = 'e_soh050'
GROUP_OF_ITEM = {'E': W.GROUP_OF_ITEM['E'], 'H': V6.GROUP_OF_ITEM['H']}


def original_eval_dir(cell):
    """Not used (no cell is gated); kept for the reused drivers' signature."""
    raise KeyError(f'{cell}: no replay reference (every W155 cell is an ungated first evaluation)')


def reference_path(cell):
    raise KeyError(f'{cell}: no replay reference (every W155 cell is an ungated first evaluation)')


def arm_variant(cell):
    arm = CELLS[cell]['arm']
    return copy.deepcopy(W.ARMS[arm]) if arm is not None else None


def cap_rule(cell):
    return {'kind': 'dynamic', 'after_first_k0': CAP_AFTER_K0, 'ceiling': CELLS[cell]['cap_ceiling'],
            'formula': 'min(k0_run + 109, ceiling) (k0_run: the first residual pass under the version-2 definition)'}


def spec_cap(cell):
    return cap_rule(cell)['ceiling']


def floor_expected(cell):
    """The floor-row lower bound the cell's ESSO models must carry: the declared minimum_soh (SoH cells), else the
    file's 0.70."""
    c = CELLS[cell]
    return c['minimum_soh'] if c['kind'] == 'soh' else FILE_MINIMUM_SOH


def declaration_for(cell):
    """The one valid declaration of a cell: the W142 extension's shape with this schema and label, the v6 rule, and
    `minimum_soh` (SoH cells ONLY; absent on the H cells) / `flex_price_multiplier_expected` (H cells; None on the SoH
    cells)."""
    if cell not in CELLS:
        raise KeyError(f'{cell} is not a W155 cell')
    c = CELLS[cell]
    soh = c['kind'] == 'soh'
    out = {
        'schema': DECLARATION_SCHEMA, 'label': LABEL, 'cell': cell, 'item': c['item'],
        'claim_group': GROUP_OF_ITEM[c['item']], 'kind': c['kind'],
        'arm': c['arm'], 'model_variant': arm_variant(cell),
        'floor_row_lower_expected': floor_expected(cell),
        'flex_price_multiplier_expected': c['flex_price_multiplier'],
        'gate': ('post_run_bitwise_vs_3f084f2f_through_172' if cell == NEUTRALITY_CELL
                 else 'none_first_evaluation'),
        'first_residual_pass_expected': None, 'N_old': None, 'original_lapses_after_k0': None,
        'holds_after': ('the run first residual pass k0_run (version-2 definition): AA off, tight tail on, rho frozen '
                        'for every later cycle'),
        'cap_rule': cap_rule(cell),
        'settling_rule': V6.settling_rule_declaration(),
        'record_all_blocks': True,
        'captures': {'q_decomposition': True, 'ess_schedule_movement': True, 'boyd_full': True, 't_sum': True,
                     'ipopt_exit_by_block': True, 'soh_floor_sidecar': True,
                     'ageing_trajectory_terminal': True,
                     'exit_clean_by_block': True, 'attempt_tier_and_final_metrics': True,
                     'soh_floor_plumbing': soh},
        'expected_blocks': {'tso': N_TSO_BLOCKS, 'dso': N_DSO_BLOCKS, 'esso': N_ESSO},
        'replay_reference': None,
        'abort_on_replay_divergence': False,
    }
    if soh:
        out['minimum_soh'] = c['minimum_soh']
    return out


def is_w155_declaration(value):
    return isinstance(value, dict) and value.get('schema') == DECLARATION_SCHEMA


def validate_settling_resettle(value):
    """None = not declared. Otherwise the value must equal `declaration_for(value['cell'])` EXACTLY. Named refusals
    first: an `early_stop` key; a `minimum_soh` key on an H cell; a flexibility multiplier on a SoH cell. Returns a new
    dict. Parent-safe (no model import)."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{OPTION_NAME} (W155) must be a dict; got {value!r}')
    forbidden = [k for k in FORBIDDEN_DECLARATION_KEYS if k in value]
    if forbidden:
        raise ValueError(f'{OPTION_NAME} (W155) must NOT carry {forbidden} (the settling rule is the only stop before '
                         f'the cap)')
    if value.get('schema') != DECLARATION_SCHEMA:
        raise ValueError(f'{OPTION_NAME} (W155): schema must be {DECLARATION_SCHEMA!r}; got {value.get("schema")!r}')
    cell = value.get('cell')
    if cell not in CELLS:
        raise ValueError(f'{OPTION_NAME} (W155).cell must be one of {sorted(CELLS)}; got {cell!r}')
    if CELLS[cell]['kind'] == 'flex' and 'minimum_soh' in value:
        raise ValueError(f'{OPTION_NAME} (W155): minimum_soh is FORBIDDEN on the H cell {cell}')
    if CELLS[cell]['kind'] == 'soh' and value.get('flex_price_multiplier_expected') is not None:
        raise ValueError(f'{OPTION_NAME} (W155): a flexibility multiplier is FORBIDDEN on the SoH cell {cell}')
    want = declaration_for(cell)
    if set(value) != set(want):
        raise ValueError(f'{OPTION_NAME} (W155) must have exactly {sorted(want)}; got {sorted(value)}')
    bad = sorted(k for k in want if json.dumps(value[k], sort_keys=True) != json.dumps(want[k], sort_keys=True))
    if bad:
        raise ValueError(f'{OPTION_NAME} (W155) differs from the W155 declaration of {cell} on {bad}')
    return copy.deepcopy(want)


def load_replay_reference(decl):
    """W132's loader (None for every W155 cell: ungated)."""
    return V.load_replay_reference(decl)


# ======================================================================================================================
#  the entry checks (the child's spec; pure)
# ======================================================================================================================
PRODUCTION_SIGNATURES = W.PRODUCTION_SIGNATURES
closed_form_expected = W.closed_form_expected
floor_year_reading = W.floor_year_reading
trajectory_equality = W.trajectory_equality


def _has_key(obj, key):
    """True iff a dict anywhere in `obj` carries `key` as a KEY (text that merely mentions it does not count)."""
    if isinstance(obj, dict):
        return key in obj or any(_has_key(v, key) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_key(v, key) for v in obj)
    return False


def spec_entry_checks(decl, spec):
    """The campaign spec the child runs holds exactly this cell's entry. A flexibility multiplier is ALLOWED (and
    required, = 1.75, with its label at spec and entry level) on the H cells and FORBIDDEN on the SoH cells;
    minimum_soh is FORBIDDEN on the H cells (anywhere in the entry); the model variant is the C2_calfade arm on the
    SoH cells and absent on the H cells; the spec's declared ESS ageing baseline keeps the FILE's minimum_soh 0.70 (the
    file is not edited: the 0.50 value enters through the keyed declaration only). Pure."""
    cands = (spec or {}).get('candidates') or []
    entry = cands[0] if len(cands) == 1 else {}
    cfg = (spec or {}).get('configuration') or {}
    cell = decl['cell']
    c = CELLS[cell]
    mv = decl['model_variant']
    ess = cfg.get('ess_ageing_baseline') or {}
    soh = c['kind'] == 'soh'
    flex_label_spec = (spec or {}).get('flex_price_label')
    out = {
        'w155_spec_holds_one_entry': len(cands) == 1,
        'w155_entry_label_is_the_cell': entry.get('label') == cell,
        'w155_entry_settling_resettle_is_the_declaration': (
            json.dumps(entry.get('settling_resettle'), sort_keys=True) == json.dumps(decl, sort_keys=True)),
        'w155_entry_model_variant_is_the_declared_arm': (
            json.dumps(entry.get('model_variant'), sort_keys=True) == json.dumps(mv, sort_keys=True)),
        'w155_variant_label_at_spec_and_entry_level_iff_variant': (
            (mv is None and 'model_variant_label' not in entry and 'model_variant_label' not in (spec or {}))
            or (mv is not None and entry.get('model_variant_label') == MODEL_VARIANT_LABEL
                and (spec or {}).get('model_variant_label') == MODEL_VARIANT_LABEL)),
        'w155_ess_ageing_baseline_declared': bool(ess),
        'w155_spec_minimum_soh_is_the_file_070_not_overridden': ess.get('minimum_soh') == FILE_MINIMUM_SOH,
        'w155_ess_params_file_pinned_as_expected': ((cfg.get('ess_params_file') or {}).get('sha256')
                                                    == ESS_PARAMS_SHA256),
        'w155_no_premium_or_derived_instance': ('interface_deviation_premium' not in entry
                                                and not cfg.get('derived_instance')),
        'w155_candidate_key_is_the_cell_candidate': entry.get('key') == c['candidate_key'],
        'w155_floor_expected_is_the_cell_floor': decl['floor_row_lower_expected'] == floor_expected(cell),
    }
    if soh:
        out.update({
            'w155_soh_cell_carries_minimum_soh_in_the_declaration': decl.get('minimum_soh') == c['minimum_soh'],
            'w155_soh_cell_minimum_soh_is_a_float_in_0_1': (isinstance(decl.get('minimum_soh'), float)
                                                            and 0.0 <= decl['minimum_soh'] < 1.0),
            'w155_flex_multiplier_forbidden_on_a_soh_cell': ('flex_price_multiplier' not in entry
                                                             and 'flex_price_label' not in entry
                                                             and flex_label_spec is None
                                                             and decl.get('flex_price_multiplier_expected') is None),
            'w155_soh_cell_variant_is_c2_calfade': (json.dumps(mv, sort_keys=True)
                                                    == json.dumps(W.ARMS[SOH_ARM], sort_keys=True)),
        })
    else:
        out.update({
            'w155_flex_multiplier_on_the_h_cell_is_1_75': entry.get('flex_price_multiplier') == FLEX_M
            == decl.get('flex_price_multiplier_expected'),
            'w155_flex_label_at_spec_and_entry_level': (entry.get('flex_price_label') is not None
                                                        and entry.get('flex_price_label') == flex_label_spec),
            'w155_minimum_soh_forbidden_on_an_h_cell': 'minimum_soh' not in decl and not _has_key(entry, 'minimum_soh'),
            'w155_h_cell_has_no_model_variant': mv is None and 'model_variant' not in entry,
        })
    return out


# ======================================================================================================================
#  the SoH-floor plumbing (H1-H3; child side: model code imported lazily)
# ======================================================================================================================
UNCHANGED_ESS_KEYS = ('bus', 't_cal', 'cl_nom', 'dod_nom', 'cl_eff', 'phi_cal')


def _ess_state(sed):
    return [{'year': str(y), 'bus': e.bus, 't_cal': e.t_cal, 'cl_nom': e.cl_nom, 'dod_nom': e.dod_nom,
             'soh_min': e.soh_min, 'cl_eff': e.cl_eff, 'phi_cal': e.phi_cal,
             'repr': {k: repr(getattr(e, k)) for k in UNCHANGED_ESS_KEYS + ('soh_min',)}}
            for y in sed.years for e in sed.shared_energy_storages[y]]


def apply_minimum_soh(sed, minimum_soh):
    """Set the LOADED ageing parameters' soh_min (`sed.params.ageing.soh_min`) to `minimum_soh` and re-run production's
    `EnergyStorageAgeingParameters.apply_to` on every SharedEnergyStorage -- never `ess.soh_min` directly. Verified:
    every ESS carries the declared soh_min, every other ageing constant (cl_eff = k and phi_cal included) is
    repr-identical, and the two ageing-model switches are unchanged. Raises on any failure. Returns the record."""
    import shared_energy_storage_data as SED
    if isinstance(minimum_soh, bool) or not isinstance(minimum_soh, float) or not 0.0 <= minimum_soh < 1.0:
        raise ValueError(f'W155 minimum_soh must be a float in [0, 1); got {minimum_soh!r}')
    ageing = sed.params.ageing
    settings_before = SED._esso_ageing_model_settings(sed)
    before = _ess_state(sed)
    ageing_before = ageing.soh_min
    ageing.soh_min = minimum_soh
    for year in sed.years:
        for ess in sed.shared_energy_storages[year]:
            ageing.apply_to(ess)
    after = _ess_state(sed)
    settings_after = SED._esso_ageing_model_settings(sed)
    checks = {
        'loaded_ageing_soh_min_is_declared': ageing.soh_min == minimum_soh and type(ageing.soh_min) is float,
        'every_ess_soh_min_is_declared': bool(after) and all(e['soh_min'] == minimum_soh
                                                             and type(e['soh_min']) is float for e in after),
        'every_other_ess_constant_repr_identical': ([{k: e['repr'][k] for k in UNCHANGED_ESS_KEYS} for e in before]
                                                    == [{k: e['repr'][k] for k in UNCHANGED_ESS_KEYS} for e in after]),
        'ess_count_unchanged': len(before) == len(after),
        'ageing_model_switches_unchanged': settings_before == settings_after,
        'applied_by_production_apply_to': 'shared_energy_storage.soh_min = self.soh_min' in inspect.getsource(
            type(ageing).apply_to),
    }
    failed = sorted(k for k, v in checks.items() if not v)
    if failed:
        raise RuntimeError(f'W155 minimum_soh {minimum_soh} did not take effect as specified: {failed}')
    return {'declared_minimum_soh': minimum_soh, 'loaded_ageing_soh_min_before': ageing_before,
            'per_ess_soh_min_before': [e['soh_min'] for e in before], 'per_ess_soh_min_after': [e['soh_min'] for e in after],
            'after': {'per_ess': [{k: v for k, v in e.items() if k != 'repr'} for e in after]},
            'ageing_model_settings': list(settings_after), 'checks': checks,
            'method': ('sed.params.ageing.soh_min set, then production EnergyStorageAgeingParameters.apply_to re-run on '
                       'every SharedEnergyStorage (shared_energy_storage_parameters.py)')}


def update_floor_rows(floor_rows_by_node, minimum_soh):
    """Set every row's `soh_min` of the shared floor-row dict IN PLACE (the reference the harness passed to both the
    configuration hook's identity check and the floor sidecar). Returns {node: [[y_inv, y, before, after]]}."""
    changed = {}
    for node_id, rows in floor_rows_by_node.items():
        for (y_inv, y), info in sorted(rows.items()):
            before = info['soh_min']
            info['soh_min'] = minimum_soh
            changed.setdefault(str(node_id), []).append([y_inv, y, before, info['soh_min']])
    return changed


def floor_rows_summary(floor_rows_by_node):
    return {str(n): {'n_rows': len(rows), 'soh_min_values': sorted({r['soh_min'] for r in rows.values()}),
                     'constraint_idx': sorted(r['constraint_idx'] for r in rows.values())}
            for n, rows in floor_rows_by_node.items()}


def _harness_modules():
    """Every LOADED copy of the campaign harness (the child runs it as `__main__`; a parent or a check imports it as
    `p515_s44_campaign_harness`): the modules whose file is the harness and which define `_config_hook_factory`."""
    out, seen = [], set()
    for name in ('__main__', 'p515_s44_campaign_harness'):
        mod = sys.modules.get(name)
        if mod is None or id(mod) in seen:
            continue
        f = getattr(mod, '__file__', None)
        if f and os.path.abspath(f) == HARNESS_FILE and callable(getattr(mod, '_config_hook_factory', None)) \
                and callable(getattr(mod, 'apply_model_variant', None)):
            out.append((name, mod))
            seen.add(id(mod))
    return out


class SohFloorPlumbing:
    """The installed H1-H3 wrappers' shared state for one run (or one zero-solve probe)."""

    def __init__(self, minimum_soh):
        self.minimum_soh = minimum_soh
        self.floor_dicts = []
        self.applies = []
        self.s38_entries = 0
        self.fail_fast_checks = []
        self.errors = []
        self.patched = []

    def summary(self):
        return {'declared_minimum_soh': self.minimum_soh, 'n_apply_calls': len(self.applies),
                'applies': copy.deepcopy(self.applies), 's38_entries': self.s38_entries,
                'n_floor_dicts_captured': len(self.floor_dicts),
                'floor_rows_now': [floor_rows_summary(d) for d in self.floor_dicts],
                'fail_fast_checks': copy.deepcopy(self.fail_fast_checks), 'errors': list(self.errors),
                'patched': list(self.patched)}

    def assert_in_force(self, planning=None):
        """THE ARMED FAIL-FAST (raises): the apply ran exactly once, the floor dict was captured (exactly one) and every
        row carries the declared value, every ESS of `planning` (when given) and its loaded ageing parameters carry
        it."""
        parts = {'apply_ran_exactly_once': len(self.applies) == 1,
                 'one_floor_dict_captured': len(self.floor_dicts) == 1,
                 'every_floor_row_carries_the_declared_value': bool(self.floor_dicts) and all(
                     r['soh_min'] == self.minimum_soh for d in self.floor_dicts for rows in d.values()
                     for r in rows.values()),
                 'every_node_has_six_floor_rows': bool(self.floor_dicts) and all(
                     len(rows) == FLOOR_ROWS_PER_NODE for d in self.floor_dicts for rows in d.values())}
        if planning is not None:
            sed = planning.shared_ess_data
            parts['planning_ageing_soh_min_is_declared'] = sed.params.ageing.soh_min == self.minimum_soh
            parts['planning_every_ess_soh_min_is_declared'] = all(
                e.soh_min == self.minimum_soh for y in sed.years for e in sed.shared_energy_storages[y])
        self.fail_fast_checks.append(parts)
        failed = sorted(k for k, v in parts.items() if not v)
        if failed:
            self.errors.append(f'fail-fast: {failed}')
            raise RuntimeError(f'W155 SoH plumbing NOT in force before the first solve (the key would not reach the '
                               f'model): {failed}')
        return parts


@contextmanager
def soh_floor_plumbing(minimum_soh):
    """Install H1 (apply wrapper on every loaded harness copy), H2 (floor-dict capture on G.s38_pf_capture_hooks) and the
    fail-fast (G.esso_capture_hooks) for the duration; restore every attribute on exit, even on error. Yields the
    SohFloorPlumbing state."""
    import p515_g_g1_g4_admm_gates as G
    pl = SohFloorPlumbing(minimum_soh)
    harness_mods = _harness_modules()
    if not harness_mods:
        raise RuntimeError('W155: no loaded campaign-harness module defines _config_hook_factory/apply_model_variant')
    originals = [(mod, 'apply_model_variant', mod.apply_model_variant) for _n, mod in harness_mods]
    originals += [(G, 's38_pf_capture_hooks', G.s38_pf_capture_hooks), (G, 'esso_capture_hooks', G.esso_capture_hooks)]

    def make_apply(original):
        @functools.wraps(original)
        def apply_model_variant(sed, model_variant):
            rec = original(sed, model_variant)
            if pl.applies:
                pl.errors.append('apply_model_variant called more than once')
                raise RuntimeError('W155: apply_model_variant called more than once in one evaluation')
            if len(pl.floor_dicts) != 1:
                pl.errors.append(f'apply before exactly one floor dict was captured ({len(pl.floor_dicts)})')
                raise RuntimeError(f'W155: the shared floor-row dict was not captured before the apply '
                                   f'({len(pl.floor_dicts)} captured) -- the identity check could not pass by equality')
            soh_rec = apply_minimum_soh(sed, pl.minimum_soh)
            soh_rec['floor_rows_updated'] = update_floor_rows(pl.floor_dicts[0], pl.minimum_soh)
            soh_rec['floor_rows_after_update'] = floor_rows_summary(pl.floor_dicts[0])
            pl.applies.append({'checks': dict(soh_rec['checks']),
                               'per_ess_soh_min_after': list(soh_rec['per_ess_soh_min_after']),
                               'n_floor_rows_updated': sum(len(v) for v in soh_rec['floor_rows_updated'].values())})
            rec['minimum_soh_applied'] = soh_rec
            return rec
        return apply_model_variant

    s38_orig = G.s38_pf_capture_hooks

    @functools.wraps(s38_orig)
    def s38_pf_capture_hooks(recourse_jump_path, ess_stride_path, floor_sidecar_path, pf_stride_path,
                             floor_rows_by_node, stride=1):
        pl.s38_entries += 1
        pl.floor_dicts.append(floor_rows_by_node)
        return s38_orig(recourse_jump_path, ess_stride_path, floor_sidecar_path, pf_stride_path, floor_rows_by_node,
                        stride=stride)

    esso_orig = G.esso_capture_hooks

    @functools.wraps(esso_orig)
    def esso_capture_hooks(planning, hook_state):
        pl.assert_in_force(planning)
        return esso_orig(planning, hook_state)

    try:
        for name, mod in harness_mods:
            mod.apply_model_variant = make_apply(mod.apply_model_variant)
            pl.patched.append(f'{name}.apply_model_variant')
        G.s38_pf_capture_hooks = s38_pf_capture_hooks
        G.esso_capture_hooks = esso_capture_hooks
        pl.patched += ['p515_g_g1_g4_admm_gates.s38_pf_capture_hooks', 'p515_g_g1_g4_admm_gates.esso_capture_hooks']
        yield pl
    finally:
        for mod, attr, fn in originals:
            setattr(mod, attr, fn)


# ======================================================================================================================
#  capture-path assertion (rule eleven, BEFORE any solve)
# ======================================================================================================================
def soh_plumbing_capture_checklist():
    """The production facts the SoH plumbing relies on, read from source BEFORE any solve: the harness's configuration
    hook calls `apply_model_variant(sed, model_variant)` by module-global name, then builds probes and compares the
    floor rows with `expected_floor_rows`; the child passes ONE dict to both s38_pf_capture_hooks and the hook; the
    floor sidecar reads soh_min from that dict; run_admm_arm enters esso_capture_hooks after the configuration hook and
    before run_operational_planning; production's apply_to writes soh_min from the ageing parameters; the floor row is
    built from `shared_energy_storage.soh_min`."""
    import p515_g_g1_g4_admm_gates as G
    import shared_energy_storage_data as SED
    import shared_energy_storage_parameters as SEP
    hmods = _harness_modules()
    hsrc = open(HARNESS_FILE).read()
    hook_src = hsrc[hsrc.find('def _config_hook_factory('):hsrc.find('# ==', hsrc.find('def _config_hook_factory('))]
    child_src = hsrc[hsrc.find('def _child_real('):hsrc.find('def main_child(')]
    arm_src = inspect.getsource(G.run_admm_arm)
    s35_src = inspect.getsource(G.s35ref_capture_hooks)
    i_apply = hook_src.find('applied_mv = apply_model_variant(sed, model_variant)')
    i_ident = hook_src.find('floor_rows_variant, _floor_counts = G._identify_soh_floor_rows(probes)')
    i_cmp = hook_src.find('floor_rows_ok = expected_floor_rows is None or floor_rows_variant == expected_floor_rows')
    i_hook = arm_src.find('pre_solve_hook(planning=planning, sed=sed, candidate=candidate, report=report)')
    i_esso = arm_src.find('esso_capture_hooks(planning, hook_state)')
    i_run = arm_src.find('_c, _results, models, _s, _p, state = planning.run_operational_planning(')
    return {
        'harness_module_loaded_and_defines_the_hook': bool(hmods),
        'hook_calls_apply_model_variant_by_global_name_then_identifies_then_compares': 0 <= i_apply < i_ident < i_cmp,
        'child_builds_one_floor_dict_and_passes_it_to_s38_and_the_hook': (
            '_cc, floor_rows_by_node, _fc = _build_floor_rows(ids[\'precheck\'])' in child_src
            and 'paths[\'pf_stride\'], floor_rows_by_node, stride=1)' in child_src
            and 'expected_floor_rows=floor_rows_by_node,' in child_src),
        'child_enters_resettle_hooks_before_s38': (0 <= child_src.find('resettle_cm, \\')
                                                   < child_src.find('G.s38_pf_capture_hooks(')),
        's38_signature_as_wrapped': list(inspect.signature(G.s38_pf_capture_hooks).parameters) == [
            'recourse_jump_path', 'ess_stride_path', 'floor_sidecar_path', 'pf_stride_path', 'floor_rows_by_node',
            'stride'],
        'sidecar_reads_soh_min_from_the_dict': "soh_min = row_info['soh_min']" in s35_src,
        'run_admm_arm_hook_then_esso_capture_then_run': 0 <= i_hook < i_esso < i_run,
        'esso_capture_hooks_signature': list(inspect.signature(G.esso_capture_hooks).parameters) == [
            'planning', 'hook_state'],
        'production_apply_to_writes_soh_min_from_the_parameters': (
            'shared_energy_storage.soh_min = self.soh_min' in inspect.getsource(SEP.EnergyStorageAgeingParameters.apply_to)),
        'production_floor_row_reads_ess_soh_min': (
            'model.es_soh_per_unit_cumul[y_inv, y] >= shared_energy_storage.soh_min' in inspect.getsource(SED)),
        'production_salvage_reads_ess_soh_min': ('min_soh = shared_energy_storage.soh_min' in inspect.getsource(
            SED._build_terminal_salvage_value_expression)),
    }


def assert_resettle_preconditions(decl, spec, tail_checklist, aa_on):
    """The capture checklist, BEFORE any solve (child; and the parent's copy): the W142 extension's items restated for
    this declaration (the v6 rule, the v6 state), the W155 entry checks (`spec_entry_checks`) and, for a SoH cell, the
    SoH-plumbing source facts (`soh_plumbing_capture_checklist`). Raises on any failure, naming every failing check;
    returns the checklist."""
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
        'state_class_is_the_w155_state_on_the_v6_state': issubclass(ResettleStateW155, V6.ResettleStateV6),
        'declaration_captures_exit_clean_by_block': decl['captures'].get('exit_clean_by_block') is True,
        'declaration_soh_plumbing_iff_soh_cell': decl['captures'].get('soh_floor_plumbing') is (
            CELLS[cell]['kind'] == 'soh'),
    }
    for name, expected in PRODUCTION_SIGNATURES.items():
        fn = getattr(srp, name, None)
        checks[f'signature:{name}'] = callable(fn) and list(inspect.signature(fn).parameters) == expected
    checks['e_no_replay_reference_every_w155_cell_ungated'] = decl['replay_reference'] is None
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
    if CELLS[cell]['kind'] == 'soh':
        for k, v in soh_plumbing_capture_checklist().items():
            checks[f'soh_plumbing:{k}'] = bool(v)
    failing = sorted(k for k, v in checks.items() if not v)
    if failing:
        raise RuntimeError(f'W155 settling-resettle preconditions fail (before any solve): {failing}')
    return checks


# ======================================================================================================================
#  the state and the install
# ======================================================================================================================
class ResettleStateW155(V6.ResettleStateV6):
    """The v6 state, reused as is; the summary names this schema, the cell kind and, for a SoH cell, the plumbing's
    record (the apply, the dict captured and updated, the fail-fast check)."""

    plumbing = None

    def summary(self):
        s = super().summary()
        s['schema'] = SCHEMA
        s['state_class'] = ('p515_s53_w142_resettle_v6_hooks.ResettleStateV6 (reused by import; the summary schema '
                            'stamped by p515_s53_w155_a64_hooks.ResettleStateW155)')
        s['arm'] = self.decl.get('arm')
        s['kind'] = self.decl.get('kind')
        s['model_variant_declared'] = self.decl.get('model_variant')
        s['minimum_soh_declared'] = self.decl.get('minimum_soh')
        s['flex_price_multiplier_expected'] = self.decl.get('flex_price_multiplier_expected')
        if self.plumbing is not None:
            p = self.plumbing.summary()
            s['soh_floor_plumbing'] = p
            s['soh_floor_plumbing_ok'] = bool(p['n_apply_calls'] == 1 and p['n_floor_dicts_captured'] == 1
                                              and p['s38_entries'] == 1 and len(p['fail_fast_checks']) == 1
                                              and not p['errors'])
        return s


@contextmanager
def settling_resettle_hooks(eval_dir, decl, holder, cap):
    """Install the nine v6 wrappers (W139's, unchanged) on the W155 state and, for a SoH cell, the SoH-floor plumbing
    (`soh_floor_plumbing`), for the run (the harness enters this FIRST, so the plumbing's G.s38_pf_capture_hooks
    wrapper is in place when the harness evaluates and enters G.s38_pf_capture_hooks next). Restores every production
    and harness attribute on exit, even on error; `holder[SUMMARY_KEY]` gets the summary."""
    import shared_resources_planning as srp
    import p515_s44_campaign_harness as HAR
    decl = validate_settling_resettle(decl)
    if int(cap) != spec_cap(decl['cell']):
        raise RuntimeError(f'settling resettle W155: spec cap {cap} != the cell cap {spec_cap(decl["cell"])}')
    if eval_dir is not None:
        for fname in (CYCLE_FILE, BLOCKS_FILE, CREEP_FILE, ESS_SCHEDULE_FILE, DECISION_FILE):
            if os.path.exists(os.path.join(eval_dir, fname)):
                raise RuntimeError(f'refusing to overwrite existing artifact: {os.path.join(eval_dir, fname)}')
    st = ResettleStateW155(decl, eval_dir, int(cap), reference=load_replay_reference(decl))
    originals = {name: getattr(srp, name) for name in WRAPPED}
    wrappers = V6.make_wrappers(st, originals, HAR.ipopt_exit_class, srp_module=srp,
                                classifier_label=f'p515_s44_campaign_harness.ipopt_exit_class ({HAR.__file__})')
    soh = CELLS[decl['cell']]['kind'] == 'soh'
    plumbing_cm = soh_floor_plumbing(decl['minimum_soh']) if soh else None
    for name, fn in wrappers.items():
        setattr(srp, name, fn)
    try:
        if plumbing_cm is not None:
            with plumbing_cm as pl:
                st.plumbing = pl
                yield st
        else:
            yield st
    finally:
        for name, fn in originals.items():
            setattr(srp, name, fn)
        holder[SUMMARY_KEY] = st.summary()


# ======================================================================================================================
#  pure post-run readers (stdlib; the campaign's gates and the zero-solve checks)
# ======================================================================================================================
def _close(a, b, rtol=READBACK_RTOL):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rtol, abs_tol=0.0)


def soh_readback_gate(rec, decl, nodes=('5', '7', '9')):
    """THE READBACK GATE of a SoH cell (the G9 replacement at the declared floor). Holds iff:
      (a) the record's variant and label are the declaration's (the C2_calfade arm);
      (b) pre-run (probe models) and terminal (clones of the run's own ESSO models) variant readbacks: all_match,
          every node present, every node's per-check dict all True, k / phi / SoH point / D-row form equal the closed
          forms of the declared variant with the file's own (N, D) (k, phi to rtol 1e-12);
      (c) every node's read-back floor-row lower bound == the DECLARED minimum_soh EXACTLY, pre-run and terminal (a 0.70
          floor on a 0.50 declaration -- or the reverse -- is refused);
      (d) the FILE is not edited: the child's file-vs-declaration checks all True, the file hashed in the child == the
          pin, the loaded minimum_soh == the spec's declared 0.70;
      (e) the variant took effect (`apply_model_variant` checks all True) and the minimum_soh apply took effect
          (`minimum_soh_applied`: declared == the declaration's, its checks all True, every ESS soh_min after == the
          declared value).
    Returns (ok, detail). Pure."""
    mv = decl['model_variant']
    floor = decl['minimum_soh']
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
            parts[f'c_{phase}_floor_row_lower_is_exactly_{floor}_node_{n}'] = fl is not None and fl == floor
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
    parts['d_loaded_minimum_soh_is_the_file_070'] = (loaded.get('minimum_soh') == FILE_MINIMUM_SOH
                                                     == (rec.get('ess_ageing_baseline') or {}).get('minimum_soh'))
    ap = rec.get('model_variant_applied_in_child') or {}
    ac = ap.get('checks') or {}
    parts['e_variant_applied_checks_all_true'] = bool(ac) and all(v is True for v in ac.values())
    sa = ap.get('minimum_soh_applied') or {}
    sc = sa.get('checks') or {}
    parts['e_minimum_soh_applied_present'] = bool(sa)
    parts['e_minimum_soh_applied_declared_is_the_declaration'] = sa.get('declared_minimum_soh') == floor
    parts['e_minimum_soh_applied_checks_all_true'] = bool(sc) and all(v is True for v in sc.values())
    after = ((sa.get('after') or {}).get('per_ess')) or []
    parts['e_every_ess_soh_min_is_the_declared_value_after_the_apply'] = bool(after) and all(
        e.get('soh_min') == floor for e in after)
    detail['file_vs_declaration_checks'] = ch
    detail['applied_checks'] = ac
    detail['minimum_soh_applied_checks'] = sc
    return all(v is True for v in parts.values()), {'parts': parts, **detail}


def soh_sidecar_gate(sidecar_lines, g_report, decl, cycles_run, summary=None):
    """THE POST-RUN SoH GATE of a SoH cell: every sidecar line (one per cycle, 1..cycles_run) has every entry's soh_min
    == the declared value exactly, 6 rows per node; the configuration hook recorded
    `floor_rows_identical_to_baseline_probe` True (the identity check passed BY EQUALITY on the updated dict); the
    plumbing summary (when given) records exactly one apply, one dict captured, one fail-fast check, no error. Returns
    (ok, detail). Pure."""
    floor = decl['minimum_soh']
    w20 = ((g_report or {}).get('rule_eleven_checklist') or {}).get('w20_model_variant') or {}
    bad_lines, per_node_counts = [], set()
    for x in sidecar_lines:
        ents = x.get('entries') or []
        counts = {}
        for e in ents:
            counts[e.get('node_id')] = counts.get(e.get('node_id'), 0) + 1
        per_node_counts.add(tuple(sorted(counts.values())))
        if not ents or any(e.get('soh_min') != floor for e in ents):
            bad_lines.append(x.get('cycle'))
    parts = {
        'one_line_per_cycle': [x.get('cycle') for x in sidecar_lines] == list(range(1, (cycles_run or 0) + 1)),
        'every_line_every_entry_soh_min_is_the_declared_value': bool(sidecar_lines) and not bad_lines,
        'six_rows_per_node_every_line': per_node_counts == {(FLOOR_ROWS_PER_NODE,) * 3},
        'floor_rows_identical_to_baseline_probe_is_true': w20.get('floor_rows_identical_to_baseline_probe') is True,
    }
    if summary is not None:
        parts['plumbing_summary_ok'] = (summary.get('soh_floor_plumbing_ok') is True
                                        and summary.get('minimum_soh_declared') == floor)
    return all(parts.values()), {'parts': parts, 'bad_lines_first': bad_lines[:10], 'n_bad_lines': len(bad_lines),
                                 'n_lines': len(sidecar_lines), 'w20_model_variant': w20}
