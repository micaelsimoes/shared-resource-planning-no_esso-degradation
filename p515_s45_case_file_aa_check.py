"""
P5.15 Addendum 27 item 1 (task W1, "AA-on into the case file") -- ZERO-SOLVE
checks for the case-file Anderson-acceleration key, its loader, and the
campaign harness's case-file-AA declaration. Armed
`SolveProfileGuard(permitted=())` for the whole script (installed before any
model import), `verify(0)` at the end.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27 item 1; frozen spec v15
`data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json`
(`configuration.oracle`: AA-on keep_memory adopted and written into
`data/SRP1/SRP1_params.json`).

Checks (production functions only, never re-implemented):
  (i)   SRP1 through the production loader (`PlanningParameters.
        read_parameters_from_file`): anderson_acceleration == {enabled: True,
        memory: 5, regularization: 1e-10, reject_policy: 'keep_memory'};
        every OTHER attribute of the ADMM settings object is pickle-identical
        to the pre-change loader (`admm_parameters.py` at PRE_CHANGE_COMMIT)
        reading the pre-change case file.
  (ii)  Every other tracked top-level case file with an `admm` section
        (OTHER_CASE_FILES): AA dict is exactly the old default with no
        `reject_policy` key, and the whole settings object (`vars`) is
        pickle-identical to the pre-change loader's.  Also: the pre-change
        SRP1 case file under the new loader is pickle-identical to the old.
  (iii) Loader refusals (ValueError): unknown sub-key, bad reject_policy,
        wrong types (enabled, memory, regularization), non-dict object.
  (iv)  Harness identity: a one-candidate spec at C* with
        `configuration.case_file_anderson_acceleration` declared, no
        overrides, cap 500, 10 consecutive cycles, no post-certification,
        frozen by the harness's own `freeze_campaign_spec` into a TEMPORARY
        directory (a fixture, not an artifact; its entry is copied into
        check.json): eval key != the D key (578636daa6d6360d...) and != the
        committed AA-override key (837fc982565dbba3...). Declaration
        refusals. Every committed S44 campaign spec: candidate key, eval key,
        eval dir and working-dir ids recomputed by the harness equal the ones
        recorded in the spec.
  (v)   The harness's configuration hook (`_config_hook_factory`) on a REAL,
        unsolved C* planning object built by `_construct_arm_planning` (the
        code `run_admm_arm` calls; cap 500, apply_rho False): PASSES with the
        declaration (no override applied, effective AA dict recorded); FAILS
        without it (committed s44_gate / s44_aa_variant / s44_selection_aa
        specs, every entry) and with a wrong declaration.

Output (new, write-once): data/SRP1/Results/P515S45/case_file_aa/
  check.json, manifest_sha256.json, scratch/ (results-dir redirect of the
  unsolved planning object).

Launch (attached, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s45_case_file_aa_check.py \\
        > data/SRP1/Results/P515S45/case_file_aa_launch.log 2>&1
"""

import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import traceback
import types
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15-S45 case-file AA check (zero solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402
from planning_parameters import PlanningParameters  # noqa: E402
import admm_parameters  # noqa: E402

OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S45', 'case_file_aa')
PRE_CHANGE_COMMIT = '0a188005'  # HEAD before Addendum 27 item 1 (frozen spec v15 commit)
SRP1_CASE_FILE = os.path.join('data', 'SRP1', 'SRP1_params.json')
OTHER_CASE_FILES = (os.path.join('data', 'CS1', 'CS1_params.json'), os.path.join('data', 'CS7', 'CS7_params.json'),
                    os.path.join('data', 'HR1', 'HR1_params.json'), os.path.join('data', 'OP1', 'OP1_params.json'),
                    os.path.join('data', 'OP2', 'OP2_params.json'))
EXPECTED_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
OLD_DEFAULT_AA = {'enabled': False, 'memory': 5, 'regularization': 1e-10}
C_STAR = {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)}
D_C_STAR_KEY = '578636daa6d6360d6701764c73ddf795e53c2c37e511e21be1400024f8f6350c'
AA_OVERRIDE_C_STAR_EVAL_KEY = '837fc982565dbba32241ed21a46da14984395006e3dbb4b03d5f72b9b72c50a7'
CANONICAL_SOURCE_SPEC = os.path.join('data', 'SRP1', 'Results', 'P515S44', 'campaign_s44_aa_variant',
                                     'campaign_spec_s44_aa_variant_f90618e8.json')
S44_SPEC_PATHSPEC = 'data/SRP1/Results/P515S44/*campaign_spec_*.json'
HOOK_SPECS_WITHOUT_DECLARATION = (
    os.path.join('data', 'SRP1', 'Results', 'P515S44', 'campaign_s44_gate', 'campaign_spec_s44_gate_4047b4e3.json'),
    CANONICAL_SOURCE_SPEC,
    os.path.join('data', 'SRP1', 'Results', 'P515S44', 'campaign_s44_selection_aa',
                 'campaign_spec_s44_selection_aa_4135c8d4.json'),
)
EVAL_ID = 'p515s45_case_file_aa_check_cstar'


def _utc():
    return datetime.now(timezone.utc).isoformat()


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
    except Exception as error:  # noqa: BLE001 -- wrong exception type is a failure, recorded
        return {'raised': False, 'wrong_type': type(error).__name__, 'message': str(error)}
    return {'raised': False}


def _pre_change_admm_module():
    """`admm_parameters.py` exactly as at PRE_CHANGE_COMMIT, as a separate module object."""
    source = _git(['show', f'{PRE_CHANGE_COMMIT}:admm_parameters.py'])
    module = types.ModuleType('admm_parameters_pre_addendum27')
    exec(compile(source, f'{PRE_CHANGE_COMMIT}:admm_parameters.py', 'exec'), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def _admm_vars(module, params_admm):
    obj = module.ADMMParameters()
    obj.read_parameters_from_file(copy.deepcopy(params_admm))
    return vars(obj)


def _pickled(d):
    return pickle.dumps(d, protocol=4)


# ======================================================================================================================
#  (i) / (ii) / (iii) -- the loader
# ======================================================================================================================
def check_loader(old_module):
    checks, detail = {}, {}
    # (i) SRP1 through the production loader
    prod = PlanningParameters()
    prod.read_parameters_from_file(os.path.join(REPO, SRP1_CASE_FILE))
    srp1_aa = prod.admm.anderson_acceleration
    checks['i_srp1_aa_equals_expected'] = srp1_aa == EXPECTED_AA
    checks['i_srp1_aa_types_exact'] = (type(srp1_aa['enabled']) is bool and type(srp1_aa['memory']) is int
                                       and type(srp1_aa['regularization']) is float
                                       and type(srp1_aa['reject_policy']) is str)
    detail['i_srp1_aa_loaded'] = srp1_aa
    new_srp1 = json.loads(open(os.path.join(REPO, SRP1_CASE_FILE)).read())
    old_srp1 = json.loads(_git(['show', f'{PRE_CHANGE_COMMIT}:{SRP1_CASE_FILE}']))
    stripped = copy.deepcopy(new_srp1)
    stripped['admm'].pop('anderson_acceleration')
    checks['i_srp1_case_file_only_change_is_the_aa_key'] = stripped == old_srp1
    new_vars = dict(_admm_vars(admm_parameters, new_srp1['admm']))
    old_vars = dict(_admm_vars(old_module, old_srp1['admm']))
    checks['i_srp1_attribute_names_and_order_identical'] = list(new_vars) == list(old_vars)
    new_aa = new_vars.pop('anderson_acceleration')
    old_aa = old_vars.pop('anderson_acceleration')
    checks['i_srp1_every_other_attribute_pickle_identical_to_pre_change'] = _pickled(new_vars) == _pickled(old_vars)
    checks['i_srp1_pre_change_aa_was_old_default'] = old_aa == OLD_DEFAULT_AA
    detail['i_srp1_new_vs_pre_change_aa'] = {'new': new_aa, 'pre_change': old_aa}
    # the pre-change SRP1 case file (no AA key) under the NEW loader: byte-identical settings
    checks['ii_pre_change_srp1_under_new_loader_pickle_identical'] = (
        _pickled(_admm_vars(admm_parameters, old_srp1['admm'])) == _pickled(_admm_vars(old_module, old_srp1['admm'])))

    # (ii) the other case files
    detail['ii_other_case_files'] = {}
    for rel in OTHER_CASE_FILES:
        pp = PlanningParameters()
        pp.read_parameters_from_file(os.path.join(REPO, rel))
        data = json.loads(open(os.path.join(REPO, rel)).read())
        new_v = _admm_vars(admm_parameters, data['admm'])
        old_v = _admm_vars(old_module, data['admm'])
        per = {
            'has_aa_key_in_file': 'anderson_acceleration' in data['admm'],
            'aa_loaded_via_production_loader': pp.admm.anderson_acceleration,
            'aa_is_old_default_no_reject_policy': (pp.admm.anderson_acceleration == OLD_DEFAULT_AA
                                                   and 'reject_policy' not in pp.admm.anderson_acceleration),
            'settings_pickle_identical_to_pre_change_loader': _pickled(new_v) == _pickled(old_v),
            'production_loader_settings_pickle_identical': _pickled(vars(pp.admm)) == _pickled(new_v),
            'file_sha256': H.sha256_file(os.path.join(REPO, rel)),
        }
        detail['ii_other_case_files'][rel] = per
        checks[f'ii_{rel}_old_default_and_identical'] = (
            not per['has_aa_key_in_file'] and per['aa_is_old_default_no_reject_policy']
            and per['settings_pickle_identical_to_pre_change_loader']
            and per['production_loader_settings_pickle_identical'])

    # (iii) refusals
    base = copy.deepcopy(new_srp1['admm'])

    def _load_with(aa_value):
        data = copy.deepcopy(base)
        data['anderson_acceleration'] = aa_value
        admm_parameters.ADMMParameters().read_parameters_from_file(data)

    cases = {
        'unknown_sub_key': dict(EXPECTED_AA, foo=1),
        'bad_reject_policy': dict(EXPECTED_AA, reject_policy='bogus'),
        'enabled_not_bool': dict(EXPECTED_AA, enabled='true'),
        'enabled_int': dict(EXPECTED_AA, enabled=1),
        'memory_float': dict(EXPECTED_AA, memory=5.0),
        'memory_bool': dict(EXPECTED_AA, memory=True),
        'memory_zero': dict(EXPECTED_AA, memory=0),
        'regularization_str': dict(EXPECTED_AA, regularization='1e-10'),
        'regularization_negative': dict(EXPECTED_AA, regularization=-1e-10),
        'not_a_dict_list': [True, 5],
        'not_a_dict_null': None,
    }
    detail['iii_refusals'] = {}
    for name, value in cases.items():
        res = _expect_raise(lambda v=value: _load_with(v))
        detail['iii_refusals'][name] = {'value': value, **res}
        checks[f'iii_refuses_{name}'] = res['raised']
    # a partial object merges onto the defaults (sanity of the merge, not required by the task)
    obj = admm_parameters.ADMMParameters()
    data = copy.deepcopy(base)
    data['anderson_acceleration'] = {'enabled': True}
    obj.read_parameters_from_file(data)
    checks['iii_partial_object_merges_onto_defaults'] = obj.anderson_acceleration == {
        'enabled': True, 'memory': 5, 'regularization': 1e-10}
    return checks, detail


# ======================================================================================================================
#  (iv) -- harness identity
# ======================================================================================================================
def check_harness_identity():
    checks, detail = {}, {}
    source = _load_json(CANONICAL_SOURCE_SPEC)
    first = source['candidates'][0]
    canon = H.canonical_candidate(C_STAR)
    checks['iv_c_star_canonical_equals_committed_first_candidate'] = (
        canon == first['canonical'] and H.candidate_key(canon) == first['key'] == D_C_STAR_KEY)

    with tempfile.TemporaryDirectory(prefix='p515s45_case_file_aa_fixture_') as tmp:
        root = os.path.join(tmp, 'fixture_campaign')
        spec_path, spec_sha, spec = H.freeze_campaign_spec(
            root, 's45_case_file_aa_fixture', [('c_star', C_STAR)],
            configuration={'name': 'fixture: case-file AA declared, no overrides', 'arm_label': 's39_D',
                           'overrides': {}, 'case_file_anderson_acceleration': dict(EXPECTED_AA),
                           'note': 'zero-solve check fixture; never launched'},
            cap=500, concurrency=1, authority=['p515_s45_case_file_aa_check.py fixture (never launched)'],
            required_consecutive_cycles=10, extra={'mode': 'zero_solve_check_fixture', 'not_for_launch': True})
        entry = spec['candidates'][0]
        detail['iv_fixture_spec'] = {'sha256_in_tempdir': spec_sha, 'file_name': os.path.basename(spec_path),
                                     'configuration': spec['configuration'], 'entry': entry,
                                     'cap': spec['cap'], 'required_consecutive_cycles':
                                         spec['required_consecutive_cycles'], 'concurrency': spec['concurrency']}
        # the same fixture's hook check is done in (v) on this exact spec dict
        fixture_spec = copy.deepcopy(spec)
    ekey = entry['eval_key']
    checks['iv_fixture_declaration_recorded_in_spec'] = (
        spec['configuration'].get('case_file_anderson_acceleration') == EXPECTED_AA)
    checks['iv_fixture_no_overrides_no_post_certification'] = (
        entry['overrides'] == {} and entry['post_certification'] is None and spec['configuration']['overrides'] == {})
    checks['iv_fixture_effective_aa_recorded'] = entry.get('effective_anderson_acceleration') == EXPECTED_AA
    checks['iv_eval_key_differs_from_D_key'] = ekey != D_C_STAR_KEY and not ekey.startswith(D_C_STAR_KEY[:16])
    checks['iv_eval_key_differs_from_aa_override_key'] = (ekey != AA_OVERRIDE_C_STAR_EVAL_KEY
                                                          and not ekey.startswith(AA_OVERRIDE_C_STAR_EVAL_KEY[:16]))
    checks['iv_eval_dir_and_ids_from_eval_key'] = (
        entry['eval_dir'] == H.eval_dir_name(ekey, 'c_star')
        and entry['working_dir_ids'] == H.eval_ids('s45_case_file_aa_fixture', ekey))
    checks['iv_eval_key_recomputes'] = ekey == H.evaluation_key(D_C_STAR_KEY, {}, case_file_aa=EXPECTED_AA)
    # the same candidate under the declared case file with AA switched OFF by override: yet another key
    off_key = H.evaluation_key(D_C_STAR_KEY, H.validate_overrides({'anderson_acceleration': {'enabled': False}}),
                               case_file_aa=EXPECTED_AA)
    checks['iv_declared_with_off_override_distinct'] = len({off_key, ekey, D_C_STAR_KEY,
                                                            AA_OVERRIDE_C_STAR_EVAL_KEY}) == 4
    detail['iv_keys'] = {'fixture_eval_key': ekey, 'D_key': D_C_STAR_KEY,
                         'aa_override_eval_key': AA_OVERRIDE_C_STAR_EVAL_KEY,
                         'declared_plus_enabled_false_override_key': off_key}
    # declaration refusals at freeze time
    decl_cases = {
        'unknown_key': dict(EXPECTED_AA, foo=1),
        'memory_not_frozen': dict(EXPECTED_AA, memory=4),
        'regularization_not_frozen': dict(EXPECTED_AA, regularization=1e-8),
        'bad_reject_policy': dict(EXPECTED_AA, reject_policy='bogus'),
        'enabled_not_bool': dict(EXPECTED_AA, enabled=1),
        'missing_memory': {'enabled': True, 'regularization': 1e-10},
        'not_a_dict': 'on',
    }
    detail['iv_declaration_refusals'] = {}
    for name, value in decl_cases.items():
        res = _expect_raise(lambda v=value: H.validate_case_file_anderson_acceleration(v))
        detail['iv_declaration_refusals'][name] = {'value': value, **res}
        checks[f'iv_declaration_refuses_{name}'] = res['raised']

    # every committed S44 campaign spec: identity recomputed == recorded
    committed = sorted(p for p in _git(['ls-files', '--', S44_SPEC_PATHSPEC]).split('\n') if p)
    per_spec = {}
    all_ok = True
    for rel in committed:
        s = _load_json(rel)
        cfg = s.get('configuration') or {}
        decl = H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration'))
        rows = []
        for e in s['candidates']:
            ov = H.validate_overrides(e['overrides'] if 'overrides' in e else (cfg.get('overrides') or {}))
            recorded_ekey = e.get('eval_key', e['key'])
            new_key = H.candidate_key(e['canonical'])
            new_ekey = H.evaluation_key(new_key, ov, case_file_aa=decl)
            ok = {'candidate_key': new_key == e['key'], 'eval_key': new_ekey == recorded_ekey}
            if 'eval_dir' in e:
                ok['eval_dir'] = H.eval_dir_name(new_ekey, e['label']) == e['eval_dir']
            if 'working_dir_ids' in e:
                ok['working_dir_ids'] = H.eval_ids(s['campaign_id'], new_ekey) == e['working_dir_ids']
            rows.append({'label': e['label'], 'recorded_eval_key16': recorded_ekey[:16],
                         'recomputed_eval_key16': new_ekey[:16], 'ok': ok})
            all_ok = all_ok and all(ok.values())
        per_spec[rel] = {'schema': s.get('schema'), 'campaign_id': s.get('campaign_id'),
                         'declares_case_file_aa': decl is not None, 'n_candidates': len(rows), 'entries': rows}
    checks['iv_committed_s44_specs_found'] = len(committed) == 22
    checks['iv_committed_s44_specs_keys_unchanged'] = all_ok and bool(committed)
    checks['iv_no_committed_s44_spec_declares_case_file_aa'] = not any(v['declares_case_file_aa']
                                                                       for v in per_spec.values())
    detail['iv_committed_s44_specs'] = {'pathspec': S44_SPEC_PATHSPEC, 'n_specs': len(committed),
                                        'n_entries': sum(v['n_candidates'] for v in per_spec.values()),
                                        'per_spec': per_spec}
    return checks, detail, fixture_spec


# ======================================================================================================================
#  (v) -- the configuration hook on a real, unsolved C* planning object
# ======================================================================================================================
def check_hook(fixture_spec):
    import p515_g_g1_g4_admm_gates as G
    checks, detail = {}, {}
    if os.path.exists(os.path.join(G.O.WORK_DIR, EVAL_ID)):
        raise RuntimeError(f'working dir id already used (never reusable): {EVAL_ID}')
    report = {}
    planning, sed, cand = G._construct_arm_planning(
        's39_D', os.path.join(OUT_ROOT, 'scratch'), report,
        investment_map=H.investment_map_from_canonical(H.canonical_candidate(C_STAR)), eval_id=EVAL_ID,
        num_max_iters_override=500, apply_rho=False)
    loaded = copy.deepcopy(planning.params.admm.anderson_acceleration)
    checks['v_planning_object_carries_case_file_aa'] = loaded == EXPECTED_AA
    detail['v_planning_object_aa'] = loaded
    detail['v_instance'] = report.get('instance')

    def _run_hook(spec, overrides):
        holder, rep = {}, {}
        H._config_hook_factory(spec, holder, overrides=overrides)(planning=planning, sed=sed, candidate=cand,
                                                                   report=rep)
        return holder, rep

    # without the declaration: every entry of the committed specs must FAIL on the AA-off check
    detail['v_without_declaration'] = {}
    all_fail = True
    for rel in HOOK_SPECS_WITHOUT_DECLARATION:
        s = _load_json(rel)
        for e in s['candidates']:
            ov = e['overrides'] if 'overrides' in e else (s['configuration'].get('overrides') or {})
            res = _expect_raise(lambda: _run_hook(s, ov), exc=RuntimeError)
            names_aa_off = 'anderson_acceleration_off_before_overrides' in res.get('message', '')
            detail['v_without_declaration'][f"{rel}::{e['label']}"] = {'overrides': ov, **res,
                                                                       'names_aa_off_check': names_aa_off}
            all_fail = all_fail and res['raised'] and names_aa_off
    checks['v_committed_specs_without_declaration_fail_on_aa_off_check'] = all_fail
    checks['v_aa_dict_untouched_by_failed_hooks'] = planning.params.admm.anderson_acceleration == EXPECTED_AA

    # a minimal undeclared spec (same cap/bar as the fixture): fails, and ONLY on the AA check
    bare = {'cap': 500, 'required_consecutive_cycles': 10, 'configuration': {'overrides': {}}}
    res = _expect_raise(lambda: _run_hook(bare, {}), exc=RuntimeError)
    detail['v_bare_undeclared_spec'] = res
    checks['v_bare_undeclared_spec_fails_only_on_aa_off'] = (
        res['raised'] and res['message'].endswith("['anderson_acceleration_off_before_overrides']"))

    # a wrong declaration fails
    wrong = copy.deepcopy(fixture_spec)
    wrong['configuration']['case_file_anderson_acceleration'] = dict(EXPECTED_AA, reject_policy='clear_memory')
    res = _expect_raise(lambda: _run_hook(wrong, {}), exc=RuntimeError)
    detail['v_wrong_declaration'] = res
    checks['v_wrong_declaration_fails_on_match_check'] = (
        res['raised'] and 'anderson_acceleration_case_file_matches_declaration' in res['message'])

    # with the declaration (the fixture spec from (iv), no overrides): passes
    holder, rep = _run_hook(fixture_spec, fixture_spec['candidates'][0]['overrides'])
    checks['v_with_declaration_all_checks_pass'] = bool(holder['configuration_checks']) and all(
        holder['configuration_checks'].values())
    checks['v_with_declaration_has_match_checks_not_off_check'] = (
        'anderson_acceleration_case_file_matches_declaration' in holder['configuration_checks']
        and 'anderson_acceleration_case_file_memory_regularization_frozen' in holder['configuration_checks']
        and 'anderson_acceleration_off_before_overrides' not in holder['configuration_checks'])
    checks['v_with_declaration_no_override_applied'] = holder['overrides_applied'] == {}
    checks['v_with_declaration_effective_aa_recorded'] = holder.get('anderson_acceleration_effective') == EXPECTED_AA
    checks['v_aa_dict_unchanged_after_passing_hook'] = planning.params.admm.anderson_acceleration == EXPECTED_AA
    detail['v_with_declaration'] = {'configuration_checks': holder['configuration_checks'],
                                    'overrides_applied': holder['overrides_applied'],
                                    'anderson_acceleration_effective': holder.get('anderson_acceleration_effective'),
                                    'rule_eleven_checklist': rep.get('rule_eleven_checklist')}

    # the consumer: `_run_operational_planning` reads exactly these keys from params.admm.anderson_acceleration
    import inspect
    import shared_resources_planning as srp
    src = inspect.getsource(srp._run_operational_planning)
    needles = ('admm_parameters = planning_problem.params.admm',
               'aa_enabled = admm_anderson_acceleration.anderson_acceleration_enabled(admm_parameters)',
               'aa_settings = admm_parameters.anderson_acceleration',
               "memory=aa_settings.get('memory', 5)",
               "regularization=aa_settings.get('regularization', 1e-10)",
               "reject_policy=aa_settings.get('reject_policy', admm_anderson_acceleration.DEFAULT_REJECT_POLICY)")
    detail['v_consumer_source_needles'] = {n: n in src for n in needles}
    checks['v_consumer_reads_params_admm_anderson_acceleration'] = all(detail['v_consumer_source_needles'].values())
    return checks, detail


def main():
    started = _utc()
    if os.path.exists(OUT_ROOT):
        raise SystemExit(f'refusing: output root exists (write-once): {OUT_ROOT}')
    os.makedirs(OUT_ROOT)
    results = {'stage': 'P5.15 Addendum 27 item 1 (W1) -- case-file AA: loader, case file, harness (zero solves)',
               'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 27 item 1',
                             'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json'],
               'started_utc': started, 'git_head': _git(['rev-parse', 'HEAD']).strip(),
               'git_status_porcelain_relevant': _git(['status', '--porcelain', '--', 'admm_parameters.py',
                                                      SRP1_CASE_FILE, 'p515_s44_campaign_harness.py',
                                                      os.path.basename(__file__)]).splitlines(),
               'file_sha256': {rel: H.sha256_file(os.path.join(REPO, rel)) for rel in (
                   'admm_parameters.py', SRP1_CASE_FILE, 'p515_s44_campaign_harness.py',
                   os.path.basename(__file__), 'admm_anderson_acceleration.py', 'shared_resources_planning.py')},
               'pre_change_commit': PRE_CHANGE_COMMIT}
    checks, sections, errors = {}, {}, {}
    old_module, old_sha = _pre_change_admm_module()
    results['pre_change_admm_parameters_sha256'] = old_sha
    fixture_spec = None
    for name, fn in (('loader', lambda: check_loader(old_module)), ('harness_identity', check_harness_identity)):
        try:
            out = fn()
            checks.update(out[0])
            sections[name] = out[1]
            if name == 'harness_identity':
                fixture_spec = out[2]
        except Exception:  # noqa: BLE001 -- recorded; the check then fails
            errors[name] = traceback.format_exc()
            print(errors[name], file=sys.stderr, flush=True)
    if fixture_spec is not None:
        try:
            c, d = check_hook(fixture_spec)
            checks.update(c)
            sections['hook'] = d
        except Exception:  # noqa: BLE001
            errors['hook'] = traceback.format_exc()
            print(errors['hook'], file=sys.stderr, flush=True)
    else:
        errors['hook'] = 'not run: the harness-identity section failed before producing the fixture spec'
    guard_failures = GUARD.verify(0)
    results.update({
        'checks': checks, 'failed_checks': sorted(k for k, v in checks.items() if not v), 'errors': errors,
        'sections': sections,
        'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
        'ended_utc': _utc(),
    })
    results['all_pass'] = bool(checks) and not results['failed_checks'] and not errors and not guard_failures
    H._write_once_json(os.path.join(OUT_ROOT, 'check.json'), results)
    manifest = {}
    for root, _dirs, files in os.walk(OUT_ROOT):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(OUT_ROOT, 'manifest_sha256.json'), manifest)
    GUARD.uninstall()
    print(f"[S45-CFAA] checks={len(checks)} failed={results['failed_checks']} errors={sorted(errors)} "
          f"guard={GUARD.counts} verify0_failures={guard_failures} all_pass={results['all_pass']}", flush=True)
    if not results['all_pass']:
        sys.exit(1)


if __name__ == '__main__':
    main()
