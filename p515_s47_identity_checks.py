"""
P5.15 Addendum 30 (task W21, item 2) -- ZERO-SOLVE checks for the ESS-ageing evaluation
identity added to the campaign harness (`p515_s44_campaign_harness.py`,
`configuration.ess_ageing_baseline`). Armed `SolveProfileGuard(permitted=())` for the whole
script (installed before any model import), `verify(0)` at the end.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 30; frozen spec v17
`data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json`
(`baseline.identity_requirement`, step S1); Planner task W21 item 2.

WHY. Before W21 no campaign spec pinned the ESS parameters file, so an evaluation of a
candidate under the new ageing baseline (C2 + phi_cal 0.985 + soh_min 0.70) would carry the
SAME evaluation key as its C3 evaluation -- an identifier that does not denote its content.

CHECKS (production / harness functions only):
  K1  Committed keys unchanged. Every committed campaign spec under P515S44 / P515S45 /
      P515S46 (`git ls-files`): canonical form, candidate key, eval key, eval dir and
      working-dir ids recomputed by the CURRENT harness equal the recorded ones, both with
      the new argument omitted and passed explicitly as None; no committed spec carries the
      declaration. Every committed evaluation_record.json: candidate key and (when recorded)
      eval key recompute from the record's own canonical form, configuration, effective
      overrides and model variant.
  K2  Distinct keys. With the Addendum-30 baseline declared (`BASELINE_C2`, the dict the
      edited file will load to), the eval key of n7_4h_e1 differs from (a) the committed A1a
      C3 key and (b) the committed s46 C2_calfade variant key (soh_min 0.50 -- a different
      model), and from the bare candidate key; C*'s differs from every committed C* eval key.
      Declaring the CURRENT (C3) dict also gives a key different from the undeclared one.
      The key is insensitive to dict order and sensitive to the declaration's content.
  K3  `validate_ess_ageing_baseline` refuses malformed declarations.
  K4  Fixture freezes into TEMPORARY directories (not artifacts): a declaration equal to
      what the file loads to freezes, pins the file (path, sha256 = the file's, last commit)
      and every entry's eval key recomputes with it; a declaration differing in one value,
      or only in a number's TYPE (10000 vs 10000.0), is refused; a label without a
      declaration and a declaration without a label are refused; a refused freeze leaves no
      directory; an undeclared fixture spec carries none of the new keys.
  K5  The child's configuration hook (`_config_hook_factory`) on a planning object built by
      `_construct_arm_planning` (the code `run_admm_arm` calls) at the unit n7 0.25/1.0 2025:
      the matching declaration passes and k / phi / floor bound are read back from probe
      ESSO models equal to the declaration's closed forms; a mismatching declaration and a
      wrong sha256 pin each raise before any solve.

Output (write-once): data/SRP1/Results/P515S47/identity_checks/
    identity_checks.json, manifest_sha256.json
Launch (attached, alone, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_identity_checks.py \\
        > data/SRP1/Results/P515S47/identity_checks_launch.log 2>&1
"""

import argparse
import copy
import json
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W21 identity checks (zero solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

STAGE = 'P5.15 Addendum 30 W21 item 2 -- ESS ageing parameters in the evaluation identity: zero-solve checks'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 30',
             'data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json (baseline.identity_requirement, S1)',
             'Planner task W21 item 2']
SPEC_DIRS = ('data/SRP1/Results/P515S44', 'data/SRP1/Results/P515S45', 'data/SRP1/Results/P515S46')
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'identity_checks')
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
BASELINE_LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
# The dict the Addendum-30 ESS parameters file loads to (types as the loader keeps them: JSON ints stay int).
BASELINE_C2 = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
               'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
               'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.8}}
UNIT_NODES = {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}
UNIT_YEAR = 2025
A1A_SPEC_REL = 'data/SRP1/Results/P515S45/campaign_s45_a1a/campaign_spec_s45_a1a_c71f52ee.json'
S46_SPEC_REL = 'data/SRP1/Results/P515S46/campaign_s46_ageing/campaign_spec_s46_ageing_6544c17e.json'


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W21-identity] {msg}', flush=True)


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _load_json(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _expect_raise(fn, exc=(ValueError, RuntimeError)):
    try:
        fn()
    except exc as error:
        return {'raised': True, 'type': type(error).__name__, 'message': str(error)[:600]}
    except Exception as error:  # noqa: BLE001
        return {'raised': False, 'unexpected': f'{type(error).__name__}: {error}'}
    return {'raised': False}


def _canon(nodes, year):
    return H.canonical_candidate(nodes, investment_year=year)


# ======================================================================================================================
#  K1 -- committed keys unchanged
# ======================================================================================================================
def check_committed():
    tracked = _git(['ls-files', '--', *SPEC_DIRS]).splitlines()
    spec_rels = sorted(r for r in tracked if os.path.basename(r).startswith('campaign_spec_') and r.endswith('.json'))
    record_rels = sorted(r for r in tracked if os.path.basename(r) == 'evaluation_record.json')
    per_spec, n_entries, all_ok = {}, 0, True
    committed_eval_keys = {}  # candidate key -> [(spec, label, eval_key)]
    for rel in spec_rels:
        spec = _load_json(rel)
        cfg = spec.get('configuration') or {}
        decl = H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration'))
        rows = []
        for entry in spec['candidates']:
            canon = entry['canonical']
            recorded = entry.get('eval_key', entry['key'])
            overrides = H.validate_overrides(entry['overrides'] if 'overrides' in entry
                                             else (cfg.get('overrides') or {}))
            new_canon = _canon({int(n): tuple(v) for n, v in canon['nodes'].items()}, canon['investment_year'])
            new_key = H.candidate_key(new_canon)
            mv = entry.get('model_variant')
            ek_omitted = H.evaluation_key(new_key, overrides, case_file_aa=decl, model_variant=mv)
            ek_none = H.evaluation_key(new_key, overrides, case_file_aa=decl, model_variant=mv, ess_ageing_baseline=None)
            ok = {'canonical_identical': json.dumps(new_canon, sort_keys=True) == json.dumps(canon, sort_keys=True),
                  'candidate_key': new_key == entry['key'],
                  'eval_key_new_argument_omitted': ek_omitted == recorded,
                  'eval_key_new_argument_none': ek_none == recorded}
            if 'eval_dir' in entry:
                ok['eval_dir'] = H.eval_dir_name(ek_none, entry['label']) == entry['eval_dir']
            if 'working_dir_ids' in entry:
                ok['working_dir_ids'] = H.eval_ids(spec['campaign_id'], ek_none) == entry['working_dir_ids']
            all_ok = all_ok and all(ok.values())
            n_entries += 1
            committed_eval_keys.setdefault(entry['key'], []).append((rel, entry['label'], recorded))
            rows.append({'label': entry['label'], 'recorded_eval_key16': recorded[:16],
                         'recomputed_eval_key16': ek_none[:16], 'ok': ok})
        per_spec[rel] = {'campaign_id': spec.get('campaign_id'), 'sha256': H.sha256_file(os.path.join(REPO, rel)),
                         'declares_ess_ageing_baseline': 'ess_ageing_baseline' in cfg or 'ess_params_file' in cfg,
                         'n_entries': len(rows), 'all_ok': all(all(r['ok'].values()) for r in rows), 'entries': rows}
    per_record, n_eval_key_checked, records_ok = {}, 0, True
    for rel in record_rels:
        rec = _load_json(rel)
        canon = rec.get('candidate_canonical')
        if not canon:
            per_record[rel] = {'skipped': 'no candidate_canonical'}
            continue
        new_key = H.candidate_key(_canon({int(n): tuple(v) for n, v in canon['nodes'].items()},
                                         canon['investment_year']))
        ok = {'candidate_key': new_key == rec.get('candidate_key'),
              'no_ess_ageing_declaration': 'ess_ageing_baseline' not in rec
                                          and 'ess_ageing_baseline' not in (rec.get('configuration') or {})}
        if rec.get('eval_key') is not None:
            cfg = rec.get('configuration') or {}
            overrides = H.validate_overrides(rec['evaluation_overrides_effective']
                                             if 'evaluation_overrides_effective' in rec
                                             else (cfg.get('overrides') or {}))
            decl = H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration'))
            ok['eval_key'] = H.evaluation_key(new_key, overrides, case_file_aa=decl,
                                              model_variant=rec.get('model_variant'),
                                              ess_ageing_baseline=None) == rec['eval_key']
            n_eval_key_checked += 1
        records_ok = records_ok and all(ok.values())
        per_record[rel] = {'label': rec.get('candidate_label'), 'status': rec.get('status'), 'ok': ok}
    checks = {
        'K1_committed_specs_found': len(spec_rels) > 0,
        'K1_committed_spec_keys_eval_keys_dirs_ids_identical': all_ok and bool(spec_rels),
        'K1_no_committed_spec_declares_ess_ageing': not any(v['declares_ess_ageing_baseline']
                                                            for v in per_spec.values()),
        'K1_committed_records_keys_identical': records_ok and bool(record_rels),
    }
    scope = {'searched': f'git ls-files -- {" ".join(SPEC_DIRS)}',
             'included': ['campaign_spec_*.json', 'evaluation_record.json'],
             'excluded': ['frozen_s4x_*_spec_*.json (Planner specs: no candidate / eval keys)',
                          'uncommitted campaign roots', 'campaign specs outside P515S44-S46 (none tracked: '
                                                        'git ls-files data/SRP1/Results | grep campaign_spec_)'],
             'n_campaign_specs': len(spec_rels), 'n_spec_entries': n_entries,
             'n_evaluation_records': len(record_rels), 'n_record_eval_keys_recomputed': n_eval_key_checked}
    return checks, {'scope': scope, 'per_spec': per_spec, 'per_record': per_record}, committed_eval_keys


# ======================================================================================================================
#  K2 -- distinct keys
# ======================================================================================================================
def check_distinct(committed_eval_keys, loaded_now):
    unit_key = H.candidate_key(_canon(UNIT_NODES, UNIT_YEAR))
    c_star_nodes = {n: (0.96875, 3.875) for n in H.ACTIVE_NODES}
    c_star_key = H.candidate_key(_canon(c_star_nodes, UNIT_YEAR))
    a1a = {e['label']: e for e in _load_json(A1A_SPEC_REL)['candidates']}
    s46 = {e['label']: e for e in _load_json(S46_SPEC_REL)['candidates']}
    a1a_unit = a1a['n7_4h_e1']
    s46_calfade = s46['n7_4h_e1_C2_calfade']
    unit_c2 = H.evaluation_key(unit_key, {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=BASELINE_C2)
    unit_c3_undeclared = H.evaluation_key(unit_key, {}, case_file_aa=CASE_FILE_AA)
    unit_now_declared = H.evaluation_key(unit_key, {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=loaded_now)
    c_star_c2 = H.evaluation_key(c_star_key, {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=BASELINE_C2)
    reordered = dict(reversed(list(BASELINE_C2.items())))
    perturbed = copy.deepcopy(BASELINE_C2)
    perturbed['minimum_soh'] = 0.5
    committed_c_star = committed_eval_keys.get(c_star_key, [])
    all_committed = {ek for items in committed_eval_keys.values() for _s, _l, ek in items}
    checks = {
        'K2_a1a_unit_is_the_committed_c3_key': (a1a_unit['key'] == unit_key
                                                and a1a_unit['eval_key'] == unit_c3_undeclared),
        'K2_s46_calfade_variant_is_c2_calfade_soh_min_0p50': (
            s46_calfade['key'] == unit_key and s46_calfade['model_variant'] == {
                'eol_retention_r': 0.8, 'calendar_retention_per_year': 0.985, 'available_energy_soh_point': 'end',
                'ageing_enabled': True}),
        'K2_baseline_unit_key_differs_from_a1a_c3_key': unit_c2 != a1a_unit['eval_key'],
        'K2_baseline_unit_key_differs_from_s46_c2_calfade_key': unit_c2 != s46_calfade['eval_key'],
        'K2_baseline_unit_key_is_not_the_candidate_key': unit_c2 != unit_key,
        'K2_baseline_unit_key_differs_from_every_committed_eval_key': unit_c2 not in all_committed,
        'K2_committed_c_star_evaluations_found': len(committed_c_star) > 0,
        'K2_baseline_c_star_key_differs_from_every_committed_c_star_key': all(
            c_star_c2 != ek for _s, _l, ek in committed_c_star),
        'K2_baseline_c_star_key_differs_from_every_committed_eval_key': c_star_c2 not in all_committed,
        'K2_declaring_the_current_file_dict_differs_from_undeclared': unit_now_declared != unit_c3_undeclared,
        'K2_key_is_dict_order_independent': H.evaluation_key(
            unit_key, {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=reordered) == unit_c2,
        'K2_key_is_content_sensitive': H.evaluation_key(
            unit_key, {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=perturbed) != unit_c2,
        'K2_declared_key_without_aa_declaration_is_not_the_candidate_key': H.evaluation_key(
            unit_key, {}, ess_ageing_baseline=BASELINE_C2) != unit_key,
    }
    detail = {'unit_candidate_key': unit_key, 'c_star_candidate_key': c_star_key,
              'baseline_c2_declaration': BASELINE_C2, 'dict_the_file_loads_to_now': loaded_now,
              'unit_eval_key_baseline_c2': unit_c2, 'unit_eval_key_committed_a1a_c3': a1a_unit['eval_key'],
              'unit_eval_key_committed_s46_c2_calfade': s46_calfade['eval_key'],
              'unit_eval_key_current_file_declared': unit_now_declared,
              'c_star_eval_key_baseline_c2': c_star_c2,
              'committed_c_star_eval_keys': [{'spec': s, 'label': l, 'eval_key': ek} for s, l, ek in committed_c_star],
              'n_committed_eval_keys_compared': len(all_committed)}
    return checks, detail


# ======================================================================================================================
#  K3 -- validation
# ======================================================================================================================
def check_validation():
    def mut(**kw):
        d = copy.deepcopy(BASELINE_C2)
        for k, v in kw.items():
            if k.startswith('cal_'):
                d['calibration'][k[4:]] = v
            else:
                d[k] = v
        return d
    missing = copy.deepcopy(BASELINE_C2)
    del missing['minimum_soh']
    cal_missing = copy.deepcopy(BASELINE_C2)
    del cal_missing['calibration']['eol_retention_r']
    bad = {'not_a_dict': [('minimum_soh', 0.7)], 'missing_key': missing, 'extra_key': mut(soh_max=1.0),
           'calibration_missing_key': cal_missing, 'calibration_extra_key': mut(cal_k=35851.0),
           'soh_min_bool': mut(minimum_soh=True), 'soh_min_string': mut(minimum_soh='0.7'),
           'soh_min_one': mut(minimum_soh=1.0), 'phi_zero': mut(calendar_retention_per_year=0.0),
           'phi_above_one': mut(calendar_retention_per_year=1.01), 'phi_nan': mut(calendar_retention_per_year=float('nan')),
           'r_one': mut(cal_eol_retention_r=1.0), 'r_none_while_active': mut(cal_eol_retention_r=None),
           'status_unknown': mut(cal_status='ON'), 'calibration_not_dict': mut(calibration=[1])}
    refusals = {name: _expect_raise(lambda d=d: H.validate_ess_ageing_baseline(d), exc=ValueError)
                for name, d in bad.items()}
    checks = {'K3_validation_refuses_every_malformed_declaration': all(v['raised'] for v in refusals.values()),
              'K3_validation_accepts_the_baseline_unchanged': H.validate_ess_ageing_baseline(BASELINE_C2) == BASELINE_C2,
              'K3_none_is_no_declaration': H.validate_ess_ageing_baseline(None) is None}
    return checks, refusals


# ======================================================================================================================
#  K4 -- fixture freezes (temporary directories)
# ======================================================================================================================
def check_fixture_freeze(loaded_now):
    ess_path = os.path.join(REPO, H.ESS_PARAMS_FILE_REL)
    unit_key = H.candidate_key(_canon(UNIT_NODES, UNIT_YEAR))
    items = [('unit', UNIT_NODES, {'investment_year': UNIT_YEAR})]
    base_cfg = {'name': 'fixture', 'arm_label': 's39_D', 'overrides': {},
                'case_file_anderson_acceleration': dict(CASE_FILE_AA)}
    checks, detail = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dict(base_cfg, ess_ageing_baseline=loaded_now, ess_ageing_baseline_label='FIXTURE')
        _p, _s, spec = H.freeze_campaign_spec(os.path.join(tmp, 'ok'), 'w21_fixture', items, configuration=cfg,
                                              cap=CAP, concurrency=5, authority=['fixture'],
                                              extra={'test_only_stub': True})
        c = spec['configuration']
        entry = spec['candidates'][0]
        checks['K4_declared_freeze_records_declaration_and_label'] = (
            c.get('ess_ageing_baseline') == loaded_now and c.get('ess_ageing_baseline_label') == 'FIXTURE')
        checks['K4_declared_freeze_pins_the_file'] = (
            (c.get('ess_params_file') or {}).get('path') == H.ESS_PARAMS_FILE_REL
            and (c.get('ess_params_file') or {}).get('sha256') == H.sha256_file(ess_path)
            and bool((c.get('ess_params_file') or {}).get('last_commit')))
        checks['K4_declared_entry_eval_key_recomputes'] = entry['eval_key'] == H.evaluation_key(
            unit_key, {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=loaded_now)
        checks['K4_declared_entry_keeps_entry_format'] = sorted(entry) == sorted(
            ['label', 'canonical', 'key', 'eval_key', 'overrides', 'post_certification', 'eval_dir',
             'working_dir_ids', 'effective_anderson_acceleration'])
        detail['declared_fixture'] = {'configuration': c, 'entry': entry}
        _p, _s, spec_u = H.freeze_campaign_spec(os.path.join(tmp, 'undeclared'), 'w21_fixture_u', items,
                                                configuration=dict(base_cfg), cap=CAP, concurrency=5,
                                                authority=['fixture'], extra={'test_only_stub': True})
        checks['K4_undeclared_fixture_has_none_of_the_new_keys'] = not any(
            k in spec_u['configuration'] for k in ('ess_ageing_baseline', 'ess_ageing_baseline_label',
                                                   'ess_params_file'))
        checks['K4_undeclared_fixture_eval_key_is_the_pre_w21_key'] = (
            spec_u['candidates'][0]['eval_key'] == H.evaluation_key(unit_key, {}, case_file_aa=CASE_FILE_AA))
    value_mismatch = copy.deepcopy(loaded_now)
    value_mismatch['minimum_soh'] = round(loaded_now['minimum_soh'] + 0.05, 10)
    type_mismatch = copy.deepcopy(loaded_now)
    type_mismatch['cycle_life_nominal'] = float(loaded_now['cycle_life_nominal'])
    refused = {}
    for name, cfg in (('value_mismatch', dict(base_cfg, ess_ageing_baseline=value_mismatch,
                                              ess_ageing_baseline_label='X')),
                      ('type_mismatch_int_vs_float', dict(base_cfg, ess_ageing_baseline=type_mismatch,
                                                          ess_ageing_baseline_label='X')),
                      ('label_without_declaration', dict(base_cfg, ess_ageing_baseline_label='X')),
                      ('declaration_without_label', dict(base_cfg, ess_ageing_baseline=loaded_now))):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, 'r')
            r = _expect_raise(lambda cfg=cfg, root=root: H.freeze_campaign_spec(
                root, 'w21_fixture_bad', items, configuration=cfg, cap=CAP, concurrency=5, authority=['fixture']),
                exc=ValueError)
            r['left_no_directory'] = not os.path.exists(root)
            refused[name] = r
    checks['K4_freeze_refuses_every_bad_configuration'] = all(v['raised'] for v in refused.values())
    checks['K4_refused_freezes_left_no_directory'] = all(v['left_no_directory'] for v in refused.values())
    detail['refusals'] = refused
    return checks, detail


# ======================================================================================================================
#  K5 -- the child's configuration hook (zero solves)
# ======================================================================================================================
def check_child_hook(G, PG, loaded_now, run_tag, scratch):
    ess_path = os.path.join(REPO, H.ESS_PARAMS_FILE_REL)
    good_pin = {'path': H.ESS_PARAMS_FILE_REL, 'sha256': H.sha256_file(ess_path)}
    _cc, floor_rows, _fc = PG._build_floor_rows(f'p515s47_w21_identity_{run_tag}_precheck')

    def run_hook(name, declared, pin):
        report = {}
        eval_id = f'p515s47_w21_identity_{run_tag}_{name}'
        planning, sed, candidate = G._construct_arm_planning(
            's39_D', os.path.join(scratch, name), report, investment_map=UNIT_NODES, eval_id=eval_id,
            num_max_iters_override=CAP, apply_rho=False, investment_year=UNIT_YEAR)
        holder = {}
        spec_like = {'configuration': {'overrides': {}, 'arm_label': 's39_D',
                                       'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                                       'ess_ageing_baseline': declared, 'ess_params_file': pin,
                                       'ess_ageing_baseline_label': 'FIXTURE'},
                     'cap': CAP, 'required_consecutive_cycles': 10}
        hook = H._config_hook_factory(spec_like, holder, overrides={}, investment_year=UNIT_YEAR,
                                      expected_floor_rows=floor_rows)
        hook(planning=planning, sed=sed, candidate=candidate, report=report)
        return holder, report

    holder, report = run_hook('match', loaded_now, good_pin)
    verified = holder.get('ess_ageing_verified_pre_run') or {}
    rb = verified.get('readback_pre_run') or {}
    node7 = ((rb.get('per_node') or {}).get('7') or {}).get('readback') or {}
    expected = H.ess_ageing_baseline_expected(loaded_now)
    mismatch = copy.deepcopy(loaded_now)
    mismatch['calendar_retention_per_year'] = 0.97 if loaded_now['calendar_retention_per_year'] != 0.97 else 0.96
    r_mismatch = _expect_raise(lambda: run_hook('mismatch', mismatch, good_pin), exc=RuntimeError)
    r_pin = _expect_raise(lambda: run_hook('badpin', loaded_now, dict(good_pin, sha256='0' * 64)), exc=RuntimeError)
    checks = {
        'K5_matching_declaration_passes_the_hook': bool(verified) and all(verified['checks'].values()),
        'K5_readback_from_probe_models_matches_the_declaration': rb.get('all_match') is True,
        'K5_probe_floor_rows_identical_to_precheck': verified.get('floor_rows_identical_to_precheck') is True,
        'K5_rule_eleven_entry_written': 'w21_ess_ageing_baseline' in report.get('rule_eleven_checklist', {}),
        'K5_mismatching_declaration_raises_before_any_solve': r_mismatch['raised'],
        'K5_wrong_sha256_pin_raises_before_any_solve': r_pin['raised'],
    }
    detail = {'expected_closed_forms': expected,
              'node7_readback': {k: node7.get(k) for k in ('k', 'phi_cal_in_model', 'floor_row_lower', 'd_row_form',
                                                           'available_energy_soh_point', 'y_inv', 'y0')},
              'per_node_checks': {n: v.get('checks') for n, v in (rb.get('per_node') or {}).items()},
              'verified_checks': verified.get('checks'), 'k_production': verified.get('k_production'),
              'mismatch_refusal': r_mismatch, 'bad_pin_refusal': r_pin}
    return checks, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=OUT_REL, help='output directory (write-once)')
    args = parser.parse_args()
    out_root = os.path.join(REPO, args.out)
    if os.path.exists(out_root):
        raise SystemExit(f'output directory exists (write-once): {out_root}')
    os.makedirs(out_root)
    scratch = tempfile.mkdtemp(prefix='p515s47_w21_identity_')
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    started = time.time()
    head = _git(['rev-parse', 'HEAD']).strip()
    ess_path = os.path.join(REPO, H.ESS_PARAMS_FILE_REL)
    loaded_now = H.load_ess_ageing_parameters(ess_path)
    _log(STAGE)
    _log(f'git HEAD {head}; {H.ESS_PARAMS_FILE_REL} sha256 {H.sha256_file(ess_path)} loads to {loaded_now}')

    import p515_g_g1_g4_admm_gates as G
    import p515_s40_polish_gap as PG

    checks, detail = {}, {}
    c, d, committed = check_committed()
    checks.update(c)
    detail['K1'] = d
    _log(f"K1: {c} scope {d['scope']}")
    c, d = check_distinct(committed, loaded_now)
    checks.update(c)
    detail['K2'] = d
    _log(f'K2: {c}')
    _log(f"   unit C2-baseline {d['unit_eval_key_baseline_c2'][:16]} vs A1a C3 {d['unit_eval_key_committed_a1a_c3'][:16]}"
         f" vs s46 C2_calfade {d['unit_eval_key_committed_s46_c2_calfade'][:16]}; C* baseline "
         f"{d['c_star_eval_key_baseline_c2'][:16]} vs {len(d['committed_c_star_eval_keys'])} committed C* keys")
    c, d = check_validation()
    checks.update(c)
    detail['K3'] = d
    _log(f'K3: {c}')
    c, d = check_fixture_freeze(loaded_now)
    checks.update(c)
    detail['K4'] = d
    _log(f'K4: {c}')
    c, d = check_child_hook(G, PG, loaded_now, run_tag, scratch)
    checks.update(c)
    detail['K5'] = d
    _log(f"K5: {c}; node-7 read-back {d['node7_readback']} expected {d['expected_closed_forms']}")

    guard_failures = GUARD.verify(0)
    checks['guard_zero_solves_verified'] = not guard_failures
    all_ok = all(checks.values())
    payload = {'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': _utc(), 'git_head_at_run': head,
               'script': os.path.basename(__file__), 'script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'harness_sha256': H.sha256_file(H.HARNESS_PATH),
               'ess_params_file': {'path': H.ESS_PARAMS_FILE_REL, 'sha256': H.sha256_file(ess_path),
                                   'loads_to': loaded_now},
               'baseline_label': BASELINE_LABEL, 'baseline_c2_declaration': BASELINE_C2,
               'key_formula': ('declared: sha256 of compact sorted JSON {candidate_key, overrides, '
                               'ess_ageing_baseline[, effective_anderson_acceleration][, model_variant]}; '
                               'undeclared: unchanged (p515_s44_campaign_harness.evaluation_key)'),
               'checks': checks, 'all_ok': all_ok, 'detail': detail,
               'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
               'wall_clock_s': time.time() - started}
    with open(os.path.join(out_root, 'identity_checks.json'), 'w') as handle:
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
