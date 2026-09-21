"""
P5.15 Addenda 28-29 (task W20, items 1, 2 and 4(a)) -- ZERO-SOLVE checks for the
two ageing-model switches in `shared_energy_storage_data.py` and the campaign
harness's `model_variant`. Armed `SolveProfileGuard(permitted=())` for the whole
script (installed before any model import), `verify(0)` at the end.

Authority: PLANNER_BRIEF_2026-09-13.md Addenda 28-29; frozen spec v16
`data/SRP1/Results/P515S46/frozen_s46_ageing_spec_v16_f4295086.json`
(`ageing_batch`); Planner task W20.

CHECKS (production functions only; nothing re-implemented except the CLOSED FORMS
the model is compared against, which are stated here and in the output):
  K  Key identity. Every committed campaign spec under P515S44 / P515S45
     (`git ls-files`): canonical form, candidate key, eval key, eval dir and
     working-dir ids recomputed by the CURRENT harness equal the recorded ones;
     every committed evaluation record's candidate key recomputes. A model
     variant changes the eval key (never the bare candidate key); the five W20
     variants give five distinct keys, none equal to the baseline key of the
     same evaluation. `validate_model_variant` refusals. A fixture spec frozen by
     `freeze_campaign_spec` into a TEMPORARY directory (not an artifact) carries
     the variant and `MODEL_VARIANT_LABEL` at spec and entry level, and an entry
     without a variant keeps the pre-W20 entry format.
  D  Defaults unchanged. For x = 0, C* and the unit (n7 0.25 MVA / 1.0 MWh,
     2025), every node's ESSO model built with the DEFAULT settings by the
     current module and by the pre-change module (`shared_energy_storage_data.py`
     at PRE_CHANGE_COMMIT, loaded from git into a private module) -- each with its
     own `_build_subproblem` + `_update_model_with_candidate_solution` on the SAME
     shared-ESS data -- is identical: every Constraint (expression string, lower,
     upper, active), Var (value, lb, ub, fixed), Param, Expression, Objective and
     the cohort bookkeeping. The defaults are also READ BACK as C3 (k =
     11541.56..., phi 1, 'end', ageing on).
  V  Variant read-back (item 4(a)). For each of the five variants, a planning
     object is built by `_construct_arm_planning` (the code `run_admm_arm` calls)
     at the unit, the variant is applied by the campaign child's OWN
     configuration hook (`_config_hook_factory(..., model_variant=...)`, which
     itself reads back from probe models and refuses on mismatch), the ESSO
     models are built as production builds them at initialization
     (`update_data_with_candidate_solution`, `build_subproblem`,
     `update_model_with_candidate_solution`), and on a CLONE of node 7's model:
       - k, phi and the SoH-point mode are read back numerically from the rows
         (`model_variant_readback`);
       - a FIXED SYNTHETIC EFC/day trajectory (SYNTHETIC_EFC) is pushed through the
         model's own rows (each row solved for its one unknown: D from the linear
         D row, SoH from the SoH row, E_available from the available row) and the
         results are compared with the closed forms, relative tolerance 1e-12:
             D_y      = 365 n EFC_y / k                     (0 when ageing is off)
             SoH_y    = SoH_{y-1} exp(-D_y) phi**n           (SoH before the cohort = 1)
             E_av_y   = E SoH_y                              ('end')
                      = E SoH_{y-1} exp(-D_y/2) phi**(n/2)   ('mid')
       - the terminal salvage (a 2030 cohort, which has remaining calendar life at
         the horizon) evaluated from the model's `salvage_value` Expression is
         compared with its closed form on the END-of-block SoH, for every variant
         (so 'mid' is shown to keep the end-of-block SoH in the salvage):
             salvage = disc * f_rec * c_E * frac_life * (f_floor E + (1 - f_floor)
                       (E SoH_end - soh_min E) / (1 - soh_min))
         with disc, c_E, frac_life, f_rec, f_floor read from production helpers
         (data, not the SoH law).

Output (write-once): data/SRP1/Results/P515S46/variant_checks/
    variant_checks.json, manifest_sha256.json (scratch in a temporary directory outside the repository)

Launch (attached, alone, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s46_variant_checks.py \\
        > data/SRP1/Results/P515S46/variant_checks_launch.log 2>&1
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W20 variant checks (zero solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

STAGE = 'P5.15 Addenda 28-29 W20 -- ageing model switches and model_variant: zero-solve checks'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addenda 28-29',
             'data/SRP1/Results/P515S46/frozen_s46_ageing_spec_v16_f4295086.json (ageing_batch)',
             'Planner task W20 items 1, 2, 4(a)']
PRE_CHANGE_COMMIT = 'dd3afa6e'
SPEC_DIRS = ('data/SRP1/Results/P515S44', 'data/SRP1/Results/P515S45')
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S46', 'variant_checks')
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
RTOL = 1e-12

UNIT_NODE, UNIT_S, UNIT_E, UNIT_YEAR = 7, 0.25, 1.0, 2025
SALVAGE_YEAR = 2030
A1A_UNIT_SPEC_REL = 'data/SRP1/Results/P515S45/campaign_s45_a1a'
A1A_UNIT_EVAL_KEY = '7eb1ce62c2509f54201191ed7220a7e25f06ca8830cb73e5afe2a14101a76b20'
SYNTHETIC_EFC = {0: 0.9, 1: 0.6, 2: 0.3}   # EFC/day by POSITION in the cohort's window (fixed, synthetic)

VARIANTS = {
    'C2': {'eol_retention_r': 0.80, 'calendar_retention_per_year': 1.0,
           'available_energy_soh_point': 'end', 'ageing_enabled': True},
    'C4': {'eol_retention_r': 0.70, 'calendar_retention_per_year': 1.0,
           'available_energy_soh_point': 'end', 'ageing_enabled': True},
    'C2_calfade': {'eol_retention_r': 0.80, 'calendar_retention_per_year': 0.985,
                   'available_energy_soh_point': 'end', 'ageing_enabled': True},
    'C3_midblock': {'eol_retention_r': 0.50, 'calendar_retention_per_year': 1.0,
                    'available_energy_soh_point': 'mid', 'ageing_enabled': True},
    'no_ageing': {'eol_retention_r': 0.50, 'calendar_retention_per_year': 1.0,
                  'available_energy_soh_point': 'end', 'ageing_enabled': False},
}
BASELINE_AS_VARIANT = {'eol_retention_r': 0.50, 'calendar_retention_per_year': 1.0,
                       'available_energy_soh_point': 'end', 'ageing_enabled': True}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W20-checks] {msg}', flush=True)


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _load_json(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _expect_raise(fn, exc=ValueError):
    try:
        fn()
    except exc as error:
        return {'raised': True, 'type': type(error).__name__, 'message': str(error)}
    except Exception as error:  # noqa: BLE001
        return {'raised': False, 'unexpected': f'{type(error).__name__}: {error}'}
    return {'raised': False}


def _rel_close(a, b):
    if a == b:
        return True
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=0.0)


# ======================================================================================================================
#  K -- key identity, validation, fixture freeze
# ======================================================================================================================
def check_keys():
    checks, detail = {}, {}
    tracked = _git(['ls-files', '--', *SPEC_DIRS]).splitlines()
    spec_rels = sorted(r for r in tracked if os.path.basename(r).startswith('campaign_spec_') and r.endswith('.json'))
    record_rels = sorted(r for r in tracked if os.path.basename(r) == 'evaluation_record.json')
    per_spec, all_ok, n_entries = {}, True, 0
    for rel in spec_rels:
        spec = _load_json(rel)
        cfg = spec.get('configuration') or {}
        decl = H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration'))
        rows = []
        for entry in spec['candidates']:
            canon = entry['canonical']
            recorded_ekey = entry.get('eval_key', entry['key'])
            overrides = H.validate_overrides(entry['overrides'] if 'overrides' in entry
                                             else (cfg.get('overrides') or {}))
            new_canon = H.canonical_candidate({int(n): tuple(v) for n, v in canon['nodes'].items()},
                                              investment_year=canon['investment_year'])
            new_key = H.candidate_key(new_canon)
            new_ekey = H.evaluation_key(new_key, overrides, case_file_aa=decl,
                                        model_variant=entry.get('model_variant'))
            ok = {'no_model_variant_in_committed_entry': 'model_variant' not in entry,
                  'canonical_identical': json.dumps(new_canon, sort_keys=True) == json.dumps(canon, sort_keys=True),
                  'candidate_key': new_key == entry['key'],
                  'eval_key': new_ekey == recorded_ekey,
                  'eval_key_without_the_new_argument': H.evaluation_key(new_key, overrides,
                                                                        case_file_aa=decl) == recorded_ekey}
            if 'eval_dir' in entry:
                ok['eval_dir'] = H.eval_dir_name(new_ekey, entry['label']) == entry['eval_dir']
            if 'working_dir_ids' in entry:
                ok['working_dir_ids'] = H.eval_ids(spec['campaign_id'], new_ekey) == entry['working_dir_ids']
            all_ok = all_ok and all(ok.values())
            n_entries += 1
            rows.append({'label': entry['label'], 'recorded_eval_key16': recorded_ekey[:16],
                         'recomputed_eval_key16': new_ekey[:16], 'ok': ok})
        per_spec[rel] = {'campaign_id': spec.get('campaign_id'), 'schema': spec.get('schema'),
                         'sha256': H.sha256_file(os.path.join(REPO, rel)),
                         'has_model_variant_label': 'model_variant_label' in spec,
                         'n_entries': len(rows), 'all_ok': all(all(r['ok'].values()) for r in rows),
                         'entries': rows}
    per_record, records_ok = {}, True
    for rel in record_rels:
        record = _load_json(rel)
        canon = record.get('candidate_canonical')
        if not canon:
            per_record[rel] = {'skipped': 'no candidate_canonical'}
            continue
        new_key = H.candidate_key(H.canonical_candidate({int(n): tuple(v) for n, v in canon['nodes'].items()},
                                                        investment_year=canon['investment_year']))
        ok = new_key == record.get('candidate_key') and 'model_variant' not in record
        records_ok = records_ok and ok
        per_record[rel] = {'label': record.get('candidate_label'), 'ok': ok}
    checks['K1_committed_specs_found'] = len(spec_rels) > 0
    checks['K1_committed_spec_keys_eval_keys_dirs_ids_identical'] = all_ok and bool(spec_rels)
    checks['K1_no_committed_spec_carries_a_model_variant'] = not any(v['has_model_variant_label']
                                                                      for v in per_spec.values())
    checks['K1_committed_record_candidate_keys_identical'] = records_ok and bool(record_rels)
    detail['K1_scope'] = {'searched': f'git ls-files -- {" ".join(SPEC_DIRS)}',
                          'included': ['campaign_spec_*.json', 'evaluation_record.json'],
                          'excluded': ['frozen_s4x_*_spec_*.json (Planner specs, no candidate keys)',
                                       'uncommitted campaign roots'],
                          'n_campaign_specs': len(spec_rels), 'n_spec_entries': n_entries,
                          'n_evaluation_records': len(record_rels)}
    detail['K1_per_spec'] = per_spec
    detail['K1_per_record'] = per_record

    # the unit's baseline eval key (A1a) and the five variant keys
    unit_nodes = {5: (0.0, 0.0), 7: (UNIT_S, UNIT_E), 9: (0.0, 0.0)}
    unit_key = H.candidate_key(H.canonical_candidate(unit_nodes, investment_year=UNIT_YEAR))
    base_ekey = H.evaluation_key(unit_key, {}, case_file_aa=CASE_FILE_AA)
    variant_keys = {name: H.evaluation_key(unit_key, {}, case_file_aa=CASE_FILE_AA, model_variant=mv)
                    for name, mv in VARIANTS.items()}
    checks['K2_unit_baseline_eval_key_equals_committed_a1a'] = base_ekey == A1A_UNIT_EVAL_KEY
    checks['K2_five_variant_keys_distinct'] = len(set(variant_keys.values())) == len(VARIANTS)
    checks['K2_no_variant_key_equals_baseline_or_candidate_key'] = all(
        k not in (base_ekey, unit_key) for k in variant_keys.values())
    checks['K2_variant_key_without_aa_declaration_is_not_the_candidate_key'] = (
        H.evaluation_key(unit_key, {}, model_variant=VARIANTS['C2']) != unit_key)
    checks['K2_variant_key_is_order_independent'] = (
        H.evaluation_key(unit_key, {}, case_file_aa=CASE_FILE_AA,
                         model_variant=dict(reversed(list(VARIANTS['C2'].items())))) == variant_keys['C2'])
    detail['K2_keys'] = {'unit_candidate_key': unit_key, 'unit_baseline_eval_key': base_ekey,
                         'variant_eval_keys': variant_keys}

    bad = {
        'missing_key': {k: v for k, v in VARIANTS['C2'].items() if k != 'ageing_enabled'},
        'extra_key': dict(VARIANTS['C2'], soh_min=0.5),
        'r_zero': dict(VARIANTS['C2'], eol_retention_r=0.0),
        'r_one': dict(VARIANTS['C2'], eol_retention_r=1.0),
        'r_bool': dict(VARIANTS['C2'], eol_retention_r=True),
        'r_string': dict(VARIANTS['C2'], eol_retention_r='0.8'),
        'phi_zero': dict(VARIANTS['C2'], calendar_retention_per_year=0.0),
        'phi_above_one': dict(VARIANTS['C2'], calendar_retention_per_year=1.01),
        'phi_nan': dict(VARIANTS['C2'], calendar_retention_per_year=float('nan')),
        'soh_point_unknown': dict(VARIANTS['C2'], available_energy_soh_point='start'),
        'ageing_enabled_int': dict(VARIANTS['C2'], ageing_enabled=1),
        'not_a_dict': [('eol_retention_r', 0.8)],
    }
    refusals = {name: _expect_raise(lambda mv=mv: H.validate_model_variant(mv)) for name, mv in bad.items()}
    checks['K3_validation_refuses_every_bad_variant'] = all(v['raised'] for v in refusals.values())
    checks['K3_validation_accepts_the_five'] = all(H.validate_model_variant(mv) == mv for mv in VARIANTS.values())
    checks['K3_none_is_no_variant'] = H.validate_model_variant(None) is None
    detail['K3_refusals'] = refusals

    # fixture freeze into a TEMPORARY directory (not an artifact)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, 'campaign_fixture')
        items = [('baseline_unit', unit_nodes, {'investment_year': UNIT_YEAR})]
        items += [(f'n7_4h_e1_{name}', unit_nodes, {'investment_year': UNIT_YEAR, 'model_variant': mv})
                  for name, mv in VARIANTS.items()]
        _path, _sha, spec = H.freeze_campaign_spec(
            root, 'w20_fixture', items,
            configuration={'name': 'fixture', 'arm_label': 's39_D', 'overrides': {},
                           'case_file_anderson_acceleration': dict(CASE_FILE_AA)},
            cap=CAP, concurrency=5, authority=['fixture'], extra={'test_only_stub': True})
        by_label = {e['label']: e for e in spec['candidates']}
        base_entry = by_label['baseline_unit']
        checks['K4_spec_top_level_label'] = spec.get('model_variant_label') == H.MODEL_VARIANT_LABEL
        checks['K4_variant_entries_carry_variant_and_label'] = all(
            by_label[f'n7_4h_e1_{n}'].get('model_variant') == mv
            and by_label[f'n7_4h_e1_{n}'].get('model_variant_label') == H.MODEL_VARIANT_LABEL
            and by_label[f'n7_4h_e1_{n}']['eval_key'] == variant_keys[n] for n, mv in VARIANTS.items())
        checks['K4_baseline_entry_keeps_pre_w20_format'] = (
            'model_variant' not in base_entry and 'model_variant_label' not in base_entry
            and base_entry['eval_key'] == A1A_UNIT_EVAL_KEY)
        detail['K4_fixture_entries'] = [{k: e.get(k) for k in ('label', 'eval_key', 'eval_dir', 'model_variant',
                                                                 'model_variant_label')}
                                        for e in spec['candidates']]
    with tempfile.TemporaryDirectory() as tmp:
        refused = _expect_raise(lambda: H.freeze_campaign_spec(
            os.path.join(tmp, 'r'), 'w20_fixture_bad',
            [('x', unit_nodes, {'model_variant': dict(VARIANTS['C2'], extra=1)})],
            configuration={'name': 'fixture', 'overrides': {}}, cap=CAP, concurrency=5, authority=['fixture']))
        checks['K4_freeze_refuses_an_invalid_variant'] = refused['raised']
        checks['K4_refused_freeze_left_no_directory'] = not os.path.exists(os.path.join(tmp, 'r'))
    return checks, detail


# ======================================================================================================================
#  model fingerprints (D)
# ======================================================================================================================
def _fingerprint(model):
    import pyomo.environ as pe
    out = {}
    for comp in model.component_objects(pe.Constraint, descend_into=True):
        for idx, con in comp.items():
            out[f'C:{comp.name}[{idx}]'] = (str(con.expr), repr(None if con.lower is None else pe.value(con.lower)),
                                             repr(None if con.upper is None else pe.value(con.upper)),
                                             con.active)
    for comp in model.component_objects(pe.Var, descend_into=True):
        for idx, var in comp.items():
            out[f'V:{comp.name}[{idx}]'] = (repr(var.value), repr(var.lb), repr(var.ub), var.fixed,
                                             str(var.domain))
    for comp in model.component_objects(pe.Param, descend_into=True):
        for idx in comp:
            out[f'P:{comp.name}[{idx}]'] = repr(pe.value(comp[idx]))
    for comp in model.component_objects(pe.Expression, descend_into=True):
        for idx, e in comp.items():
            out[f'E:{comp.name}[{idx}]'] = str(e.expr)
    for comp in model.component_objects(pe.Objective, descend_into=True):
        out[f'O:{comp.name}'] = (str(comp.expr), comp.active, comp.sense)
    out['_esso_cohort_constraints'] = repr(model._esso_cohort_constraints)
    out['_esso_cohort_inactive'] = repr(model._esso_cohort_inactive)
    return out


def _fp_digest(fp):
    return hashlib.sha256(json.dumps(fp, sort_keys=True, default=str).encode()).hexdigest()


def _pre_change_module(scratch):
    source = _git(['show', f'{PRE_CHANGE_COMMIT}:shared_energy_storage_data.py'])
    path = os.path.join(scratch, f'shared_energy_storage_data_at_{PRE_CHANGE_COMMIT}.py')
    with open(path, 'w') as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location(f'sed_pre_w20_{PRE_CHANGE_COMMIT}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {'commit': PRE_CHANGE_COMMIT, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                    'has_esso_ageing_model_settings': hasattr(module, '_esso_ageing_model_settings')}


def _construct(G, name, run_tag, investment_map, investment_year, scratch):
    report = {}
    eval_id = f'p515s46_w20_checks_{run_tag}_{name}'
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(scratch, name), report, investment_map=investment_map, eval_id=eval_id,
        num_max_iters_override=CAP, apply_rho=False, investment_year=investment_year)
    return planning, sed, candidate


def _build_esso(module, sed, candidate):
    """ESSO models exactly as `create_shared_energy_storage_model` builds them, minus the solve."""
    sed.update_data_with_candidate_solution(candidate['investment'])
    models = {node_id: module._build_subproblem(sed, node_id) for node_id in sed.active_distribution_network_nodes}
    module._update_model_with_candidate_solution(sed, models, candidate['investment'])
    return models


def check_defaults(G, SED, old, run_tag, scratch):
    checks, detail = {}, {}
    import p514_n_instrumented_cstar as N
    instances = {
        'x0': ({n: (0.0, 0.0) for n in H.ACTIVE_NODES}, UNIT_YEAR),
        'c_star': ({n: (N.S_INV, N.E_INV) for n in H.ACTIVE_NODES}, UNIT_YEAR),
        'unit': ({5: (0.0, 0.0), 7: (UNIT_S, UNIT_E), 9: (0.0, 0.0)}, UNIT_YEAR),
        'unit_2030': ({5: (0.0, 0.0), 7: (UNIT_S, UNIT_E), 9: (0.0, 0.0)}, SALVAGE_YEAR),
    }
    per_instance, all_equal = {}, True
    for name, (inv_map, year) in instances.items():
        _planning, sed, candidate = _construct(G, f'defaults_{name}', run_tag, inv_map, year, scratch)
        settings = SED._esso_ageing_model_settings(sed)
        new_models = _build_esso(SED, sed, candidate)
        sed_old = copy.deepcopy(sed)
        old_models = _build_esso(old, sed_old, candidate)
        per_node = {}
        for node_id in sed.active_distribution_network_nodes:
            fp_new, fp_old = _fingerprint(new_models[node_id]), _fingerprint(old_models[node_id])
            diff_keys = sorted(k for k in set(fp_new) | set(fp_old) if fp_new.get(k) != fp_old.get(k))
            per_node[str(node_id)] = {'n_items': len(fp_new), 'digest_current': _fp_digest(fp_new),
                                      'digest_pre_change': _fp_digest(fp_old), 'n_differences': len(diff_keys),
                                      'first_differences': diff_keys[:10]}
            all_equal = all_equal and not diff_keys
        readback = H.model_variant_readback_models(new_models, sed, BASELINE_AS_VARIANT, year, clone=True)
        per_instance[name] = {'investment_year': year,
                              'investment_map': {str(n): list(v) for n, v in inv_map.items()},
                              'candidate_key': H.candidate_key(H.canonical_candidate(inv_map, investment_year=year)),
                              'settings_read_by_production': list(settings),
                              'per_node': per_node,
                              'defaults_read_back_as_C3': readback['all_match'],
                              'readback_node7': readback['per_node']['7']['readback']}
        del new_models, old_models, sed_old
    checks['D1_default_settings_are_end_and_ageing_on'] = all(
        v['settings_read_by_production'] == ['end', True] for v in per_instance.values())
    checks['D2_default_esso_models_identical_to_pre_change_module'] = all_equal
    checks['D3_defaults_read_back_as_C3'] = all(v['defaults_read_back_as_C3'] for v in per_instance.values())
    detail['D_per_instance'] = per_instance
    return checks, detail


# ======================================================================================================================
#  V -- the five variants: read-back, synthetic EFC trajectory, salvage
# ======================================================================================================================
def _solve_linear_row(con, var):
    """The value of `var` that zeroes an equality row's residual when the row is AFFINE in `var`
    (checked by a third point)."""
    var.set_value(0.0)
    r0 = H._eq_residual(con)
    var.set_value(1.0)
    r1 = H._eq_residual(con)
    var.set_value(0.5)
    r_half = H._eq_residual(con)
    if not math.isclose(r_half - r0, 0.5 * (r1 - r0), rel_tol=RTOL, abs_tol=1e-300):
        raise RuntimeError(f'row {con.name} is not affine in {var.name}')
    value = -r0 / (r1 - r0) + 0.0   # + 0.0 folds -0.0 into 0.0
    var.set_value(value)
    return value


def synthetic_trajectory(model, sed, y_inv, expected, mv):
    """Push SYNTHETIC_EFC through the model's own rows and compare with the closed forms."""
    import pyomo.environ as pe
    years = list(sed.years)
    n = sed.years[years[y_inv]]
    e_inv = pe.value(model.es_e_investment_fixed[y_inv])
    triples = H._degradation_triples(model, y_inv)
    window = sorted(triples)
    k = expected['k']
    phi = expected['phi_cal_in_model']
    rows, ok = [], True
    soh_prev_cf = 1.0
    for j, y in enumerate(window):
        efc = SYNTHETIC_EFC[j]
        d_row, soh_row, _floor = triples[y]
        model.es_e_rated_per_unit[y_inv, y].set_value(e_inv)
        model.es_avg_ch_dch_per_unit[y_inv, y].set_value(efc * 2.0 * e_inv)
        d_model = _solve_linear_row(d_row, model.es_D_per_unit[y_inv, y])
        soh_model = _solve_linear_row(soh_row, model.es_soh_per_unit_cumul[y_inv, y])
        e_av_model = _solve_linear_row(H._available_energy_row(model, y_inv, y),
                                       model.es_e_available_per_unit[y_inv, y])
        d_cf = 365.0 * n * efc / k if mv['ageing_enabled'] else 0.0
        soh_cf = soh_prev_cf * math.exp(-d_cf) * phi ** n
        soh_mid_cf = soh_prev_cf * math.exp(-d_cf / 2.0) * phi ** (n / 2.0)
        e_av_cf = e_inv * (soh_cf if mv['available_energy_soh_point'] == 'end' else soh_mid_cf)
        cell_ok = {'D': _rel_close(d_model, d_cf) if d_cf else abs(d_model) == 0.0,
                   'SoH_end': _rel_close(soh_model, soh_cf),
                   'E_available': _rel_close(e_av_model, e_av_cf)}
        ok = ok and all(cell_ok.values())
        rows.append({'block_index': y, 'block_year': str(years[y]), 'efc_per_day': efc, 'n_years': n,
                     'D_model': d_model, 'D_closed_form': d_cf,
                     'SoH_end_model': soh_model, 'SoH_end_closed_form': soh_cf,
                     'SoH_mid_closed_form': soh_mid_cf,
                     'E_available_model': e_av_model, 'E_available_closed_form': e_av_cf,
                     'SoH_used_for_available_energy_model': e_av_model / e_inv,
                     'rel_err': {'D': (abs(d_model - d_cf) / abs(d_cf)) if d_cf else abs(d_model),
                                 'SoH_end': abs(soh_model - soh_cf) / abs(soh_cf),
                                 'E_available': abs(e_av_model - e_av_cf) / abs(e_av_cf)},
                     'ok': cell_ok})
        soh_prev_cf = soh_cf
    return {'y_inv': y_inv, 'E': e_inv, 'window': window, 'rows': rows, 'all_ok': ok}


def salvage_check(SED, model, sed, y_inv, traj):
    """`model.salvage_value` (production Expression) after the synthetic trajectory vs the closed form on
    the END-of-block terminal SoH; the mid-block value is reported beside it."""
    import pyomo.environ as pe
    years = list(sed.years)
    idx = sed.get_shared_energy_storage_idx(UNIT_NODE)
    ess = sed.shared_energy_storages[years[y_inv]][idx]
    params = sed.params.salvage_value
    disc = SED._get_terminal_discount_factor(sed)
    cost = SED._get_expected_energy_investment_cost(sed, years[y_inv])
    _age, _rem, frac = SED._get_remaining_calendar_life(sed, y_inv, ess)
    e_inv = traj['E']
    last = traj['rows'][-1]
    soh_min = ess.soh_min

    def closed(soh):
        residual = (params.recycling_floor_fraction * e_inv
                    + (1.0 - params.recycling_floor_fraction) * (e_inv * soh - soh_min * e_inv) / (1.0 - soh_min))
        return disc * params.energy_recovery_fraction * cost * frac * residual

    model_value = pe.value(model.salvage_value)
    end_cf = closed(last['SoH_end_closed_form'])
    mid_cf = closed(last['SoH_mid_closed_form'])
    return {'terminal_block': last['block_year'], 'remaining_life_fraction': frac, 'terminal_discount': disc,
            'expected_energy_cost': cost, 'salvage_model': model_value, 'salvage_closed_form_end_soh': end_cf,
            'salvage_closed_form_if_mid_soh': mid_cf,
            'equals_end_soh_closed_form': _rel_close(model_value, end_cf),
            'differs_from_mid_soh_value': not _rel_close(model_value, mid_cf)}


def check_variants(G, SED, run_tag, scratch, floor_rows_expected):
    checks, detail = {}, {}
    table = {}
    for name, mv in [('C3_baseline_control', None)] + list(VARIANTS.items()):
        entry = {'model_variant': mv}
        for year, purpose in ((UNIT_YEAR, 'unit'), (SALVAGE_YEAR, 'salvage_2030')):
            inv_map = {5: (0.0, 0.0), UNIT_NODE: (UNIT_S, UNIT_E), 9: (0.0, 0.0)}
            planning, sed, candidate = _construct(G, f'{name}_{purpose}', run_tag, inv_map, year, scratch)
            holder, report = {}, {}
            spec_like = {'configuration': {'overrides': {}, 'arm_label': 's39_D',
                                           'case_file_anderson_acceleration': dict(CASE_FILE_AA)},
                         'cap': CAP, 'required_consecutive_cycles': 10}
            hook = H._config_hook_factory(spec_like, holder, overrides={}, model_variant=mv, investment_year=year,
                                          expected_floor_rows=floor_rows_expected)
            hook(planning=planning, sed=sed, candidate=candidate, report=report)
            models = _build_esso(SED, sed, candidate)
            as_variant = mv if mv is not None else BASELINE_AS_VARIANT
            expected = H.model_variant_expected(as_variant, sed)
            y_inv = [int(y) for y in sed.years].index(year)
            readback = H.model_variant_readback(models[UNIT_NODE].clone(), sed, y_inv)
            readback_checks = H.compare_readback(readback, expected)
            clone = models[UNIT_NODE].clone()
            traj = synthetic_trajectory(clone, sed, y_inv, expected, as_variant)
            salvage = salvage_check(SED, clone, sed, y_inv, traj)
            entry[purpose] = {
                'investment_year': year,
                'candidate_key': H.candidate_key(H.canonical_candidate(inv_map, investment_year=year)),
                'hook_pre_run_readback_all_match': (holder.get('model_variant_readback_pre_run') or {}).get(
                    'all_match') if mv is not None else None,
                'hook_apply_checks': (holder.get('model_variant_applied') or {}).get('checks'),
                'rule_eleven_w20': report.get('rule_eleven_checklist', {}).get('w20_model_variant'),
                'expected': expected, 'readback': readback, 'readback_checks': readback_checks,
                'synthetic_trajectory': traj, 'salvage': salvage}
            del models, clone, planning
        u = entry['unit']
        table[name] = {'k_read_back': u['readback']['k'], 'k_expected': u['expected']['k'],
                       'phi_read_back': u['readback']['phi_cal_in_model'],
                       'phi_expected': u['expected']['phi_cal_in_model'],
                       'soh_point_read_back': u['readback']['available_energy_soh_point'],
                       'd_row_form_read_back': u['readback']['d_row_form'],
                       'readback_all_match': all(u['readback_checks'].values()),
                       'synthetic_trajectory_all_within_1e-12': u['synthetic_trajectory']['all_ok'],
                       'salvage_2030_equals_end_soh_closed_form':
                           entry['salvage_2030']['salvage']['equals_end_soh_closed_form'],
                       'SoH_end_per_block': [r['SoH_end_model'] for r in u['synthetic_trajectory']['rows']],
                       'SoH_available_per_block': [r['SoH_used_for_available_energy_model']
                                                   for r in u['synthetic_trajectory']['rows']]}
        detail[name] = entry
    checks['V1_every_variant_reads_back_as_specified'] = all(v['readback_all_match'] for v in table.values())
    checks['V2_hook_pre_run_readback_matched_for_every_variant'] = all(
        detail[n][p]['hook_pre_run_readback_all_match'] is True for n in VARIANTS for p in ('unit', 'salvage_2030'))
    checks['V3_synthetic_trajectory_within_1e-12_everywhere'] = all(
        detail[n][p]['synthetic_trajectory']['all_ok'] for n in detail for p in ('unit', 'salvage_2030'))
    checks['V4_salvage_uses_end_of_block_soh_in_every_variant'] = all(
        detail[n]['salvage_2030']['salvage']['equals_end_soh_closed_form'] for n in detail)
    checks['V5_midblock_salvage_differs_from_a_mid_soh_salvage'] = (
        detail['C3_midblock']['salvage_2030']['salvage']['differs_from_mid_soh_value'])
    checks['V6_salvage_2030_nonzero'] = all(detail[n]['salvage_2030']['salvage']['salvage_model'] > 0 for n in detail)
    checks['V7_no_ageing_soh_identically_one'] = all(
        r['SoH_end_model'] == 1.0 and r['SoH_used_for_available_energy_model'] == 1.0
        for p in ('unit', 'salvage_2030') for r in detail['no_ageing'][p]['synthetic_trajectory']['rows'])
    return checks, detail, table


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=OUT_REL, help='output directory (write-once)')
    args = parser.parse_args()
    out_root = os.path.join(REPO, args.out)
    if os.path.exists(out_root):
        raise SystemExit(f'output directory exists (write-once): {out_root}')
    os.makedirs(out_root)
    # scratch (the pre-change module copy loaded from git, results-dir redirects of the unsolved planning
    # objects) lives OUTSIDE the repository and is removed at the end: it is not evidence (the source's
    # sha256 and commit are recorded).
    scratch = tempfile.mkdtemp(prefix='p515s46_w20_checks_')
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    started = time.time()
    _log(STAGE)
    head = _git(['rev-parse', 'HEAD']).strip()
    _log(f'git HEAD {head}; output {os.path.relpath(out_root, REPO)}; run tag {run_tag}')

    import p515_g_g1_g4_admm_gates as G
    import shared_energy_storage_data as SED
    import p515_s40_polish_gap as PG

    checks, detail = {}, {}
    c, d = check_keys()
    checks.update(c)
    detail['K'] = d
    _log(f'K: {c}')

    old, old_info = _pre_change_module(scratch)
    detail['pre_change_module'] = old_info
    c, d = check_defaults(G, SED, old, run_tag, scratch)
    checks.update(c)
    detail['D'] = d
    _log(f'D: {c}')

    _cc, floor_rows, _fc = PG._build_floor_rows(f'p515s46_w20_checks_{run_tag}_precheck')
    c, d, table = check_variants(G, SED, run_tag, scratch, floor_rows)
    checks.update(c)
    detail['V'] = d
    _log(f'V: {c}')
    for name, row in table.items():
        _log(f"  {name}: k={row['k_read_back']} (expected {row['k_expected']}) phi={row['phi_read_back']} "
             f"(expected {row['phi_expected']}) mode={row['soh_point_read_back']} D-row={row['d_row_form_read_back']} "
             f"match={row['readback_all_match']} traj1e-12={row['synthetic_trajectory_all_within_1e-12']} "
             f"salvage_end={row['salvage_2030_equals_end_soh_closed_form']}")

    guard_failures = GUARD.verify(0)
    checks['guard_zero_solves_verified'] = not guard_failures
    all_ok = all(checks.values())
    payload = {'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': _utc(), 'git_head_at_run': head,
               'script': os.path.basename(__file__),
               'script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'harness_sha256': H.sha256_file(H.HARNESS_PATH),
               'shared_energy_storage_data_sha256': H.sha256_file(os.path.join(REPO, 'shared_energy_storage_data.py')),
               'tolerance_relative': RTOL, 'synthetic_efc_per_day_by_block_index': SYNTHETIC_EFC,
               'variants': VARIANTS, 'model_variant_label': H.MODEL_VARIANT_LABEL,
               'closed_forms': {
                   'k': 'k = cycles_n * reference_dod_d / (-ln eol_retention_r)',
                   'D': 'D_y = 365 * n * EFC_y / k (0 when ageing is off)',
                   'SoH_end': 'SoH_y = SoH_{y-1} * exp(-D_y) * phi**n',
                   'SoH_mid': 'SoH_mid_y = SoH_{y-1} * exp(-D_y/2) * phi**(n/2)',
                   'E_available': "E * SoH_y ('end') or E * SoH_mid_y ('mid')",
                   'salvage': ('disc * f_rec * c_E * frac_life * (f_floor E + (1 - f_floor) (E SoH_end - soh_min E) '
                               '/ (1 - soh_min)), END-of-block terminal SoH')},
               'readback_table': table, 'checks': checks, 'all_ok': all_ok, 'detail': detail,
               'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
               'wall_clock_s': time.time() - started}
    with open(os.path.join(out_root, 'variant_checks.json'), 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, files in os.walk(out_root):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    with open(os.path.join(out_root, 'manifest_sha256.json'), 'w') as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    GUARD.uninstall()
    shutil.rmtree(scratch, ignore_errors=True)
    for k, v in checks.items():
        _log(f'  {"OK  " if v else "FAIL"} {k}')
    _log(f'ALL_OK={all_ok} guard={dict(GUARD.counts)} wall={time.time() - started:.1f}s')
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
