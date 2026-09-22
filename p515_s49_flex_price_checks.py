"""
P5.15 Addendum 34 (task W33, items 1-2) -- ZERO-SOLVE checks for the flexibility-price multiplier added to the
campaign harness (`p515_s44_campaign_harness.py`, evaluation option `flex_price_multiplier`). Armed
`SolveProfileGuard(permitted=())` for the whole script (installed before any model import), `verify(0)` at the
end; the W21 module imported for its digest function installs its own zero-permitted guard, verified too.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 34 ("flexibility price as a break-even axis ... via harness
override (no workbook edit)"); frozen spec v19 `data/SRP1/Results/P515S49/frozen_s49_spec_v19_f8adcc97.json`
`flex_price_ladder`; Planner task W33 items 1-2.

THE OVERRIDE (harness): `cost_flex` is an hourly profile per (year, day), growth 1.02^(y - 2025) included
(shared_resources_planning.py `_read_market_data`), bound to every DSO block (`network[year][day].cost_flex`) and
BAKED INTO the DSO objective as constants when the block is BUILT (model_construction_helpers.flexibility_cost:
c_flex[p] * baseMVA * (flex_p_down + flex_q_down), inside the `flex_cost_scenario` Expression -- no mutable
Param). So the multiplier is applied BEFORE the DSO models are built (the child's configuration hook = run_admm_arm's
pre_solve_hook), by replacing each DSO block's array with a NEW array m * cost_flex; rescaling after the build
would mean rewriting expressions. The same array object is bound to the TSO and every DSO, hence never in place.

CHECKS
  K1  Committed keys unchanged. Every committed campaign spec under P515S44..P515S49 (`git ls-files`): candidate
      key, eval key, eval dir and working-dir ids recomputed by the CURRENT harness equal the recorded ones with
      the new argument omitted, passed as None and passed as 1.0 (each spec's own AA / ESS-ageing declarations,
      overrides and model variant applied); no committed spec entry carries the new option. Every committed
      evaluation_record.json: candidate key and (when recorded) eval key recompute likewise.
  K2  `validate_flex_price_multiplier` refuses bool, 0, negatives, NaN, +/-inf, strings; accepts ints (as float).
  K3  Distinct keys: under the BASELINE declaration (the committed s47_recert / s48_x0_capture configuration) the
      eval keys of x = 0 and of the unit n7 0.25/1.0 2025 at m in {1.5, 2, 3} are pairwise distinct, distinct from
      every committed eval key, and m absent == m = 1.0 == the committed baseline keys (s48 x0 d2c96b14...,
      s47 unit bd504ecf...). Integer 2 and float 2.0 give one key; 1.5 and nextafter(1.5) give two.
  K4  Fixture freezes into TEMPORARY directories (not artifacts): entries with m in {1.5, 2, 3} freeze with the
      label at spec and entry level, their eval keys recompute; an m = 1.0 entry keeps the baseline key and no
      label; x0 at m = 1.0 beside x0 without m is refused as a duplicate evaluation; an invalid m is refused and
      leaves no directory.
  K5  READ-BACK at m in {1.5, 2, 3} (and 1.0 explicit) at x = 0 and at the unit: a planning object built by
      `_construct_arm_planning` (the code run_admm_arm calls), passed through the campaign child's configuration
      hook with the BASELINE declarations and the multiplier (`_config_hook_factory(..., flex_price_multiplier=m)`,
      which applies it and reads it back from probe DSO blocks, refusing on mismatch); then the production
      builders run_operational_planning calls at initialization (create_admm_variables,
      create_distribution_networks_models, create_transmission_network_model, create_shared_energy_storage_model)
      with every holder's `optimize` replaced by a stub that DIGESTS the blocks handed to it (nothing is solved;
      the digest is W21's `block_state`, BY IMPORT). Against the build WITHOUT the multiplier: (a) m = 1.0: every
      block digest identical; (b) m != 1: every TSO and ESSO block digest identical, DSO blocks differ ONLY in the
      components holding the flexibility-cost coefficients, and in each DSO block's objective (standard repn) the
      flex_p_down / flex_q_down coefficients are m x the m = 1 ones (<= 1e-15 relative; bitwise count reported)
      with every other term identical; (c) the post-run read-back function (`flex_price_readback_run_models`) on
      these built DSO blocks matches; (d) the hook's own pre-run read-back matched.
  K6  Refusals: a sabotaged scaling (one block's array scaled by m * (1 + 1e-7), via a monkeypatch of the
      harness's `_scaled_cost_flex` in THIS process only) makes the configuration hook raise before any solve;
      a child whose entry has m != 1 without the label refuses (`_child_real`, before any model is built).

Output (write-once): data/SRP1/Results/P515S49/flex_price_checks/{flex_price_checks.json, manifest_sha256.json}
Launch (attached, alone, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_flex_price_checks.py \\
        > data/SRP1/Results/P515S49/flex_price_checks_launch.log 2>&1
"""

import argparse
import copy
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W33 flexibility-price checks (zero solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

STAGE = 'P5.15 Addendum 34 W33 items 1-2 -- flexibility-price multiplier override: zero-solve checks'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 34',
             'data/SRP1/Results/P515S49/frozen_s49_spec_v19_f8adcc97.json flex_price_ladder',
             'Planner task W33 items 1-2']
SPEC_DIRS = tuple(f'data/SRP1/Results/P515S4{i}' for i in range(4, 10))
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S49', 'flex_price_checks')
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
MULTIPLIERS = (1.5, 2.0, 3.0)
YEAR = 2025
INSTANCES = {'x0': {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)},
             'unit_n7_4h_e1': {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}}
RECERT_SPEC_REL = 'data/SRP1/Results/P515S47/campaign_s47_recert/campaign_spec_s47_recert_902f93aa.json'
X0_CAPTURE_SPEC_REL = 'data/SRP1/Results/P515S48/x0_capture/campaign_spec_s48_x0_capture_4a50c0e2.json'
COMMITTED_BASELINE_KEYS = {'x0': 'd2c96b1480402a3b61aca4abc188e41c6009eb582d8e6ccdd380e51651f996c7',
                           'unit_n7_4h_e1': 'bd504ecf5a288d447e53ef6f7e8090b017d15549b257aa095ff4f95890a06e42'}
# The components of a DSO block that hold the flexibility-cost coefficients (by construction:
# model_construction_helpers.build_objective / objective_function_rule) -- the ONLY ones allowed to differ.
FLEX_COMPONENTS = ('Expression:flex_cost_scenario', 'Objective:objective')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W33-checks] {msg}', flush=True)


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


def _baseline_configuration():
    """The committed BASELINE configuration (s47_recert), and s48_x0_capture's, which must be the same."""
    recert = _load_json(RECERT_SPEC_REL)['configuration']
    x0cap = _load_json(X0_CAPTURE_SPEC_REL)['configuration']
    keys = ('case_file_anderson_acceleration', 'ess_ageing_baseline', 'ess_ageing_baseline_label', 'ess_params_file')
    same = {k: recert.get(k) == x0cap.get(k) for k in keys}
    return {k: recert[k] for k in keys}, same


def _key(nodes, conf, m=None, omit=False):
    ck = H.candidate_key(_canon(nodes, YEAR))
    if omit:
        return H.evaluation_key(ck, {}, case_file_aa=conf['case_file_anderson_acceleration'],
                                ess_ageing_baseline=conf['ess_ageing_baseline'])
    return H.evaluation_key(ck, {}, case_file_aa=conf['case_file_anderson_acceleration'],
                            ess_ageing_baseline=conf['ess_ageing_baseline'], flex_price_multiplier=m)


# ======================================================================================================================
#  K1 -- committed keys unchanged
# ======================================================================================================================
def check_committed():
    tracked = _git(['ls-files', '--', *SPEC_DIRS]).splitlines()
    spec_rels = sorted(r for r in tracked if os.path.basename(r).startswith('campaign_spec_') and r.endswith('.json'))
    record_rels = sorted(r for r in tracked if os.path.basename(r) == 'evaluation_record.json')
    per_spec, n_entries, all_ok = {}, 0, True
    committed_eval_keys = set()
    for rel in spec_rels:
        spec = _load_json(rel)
        cfg = spec.get('configuration') or {}
        decl = H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration'))
        ess = cfg.get('ess_ageing_baseline')
        rows = []
        for entry in spec['candidates']:
            canon = entry['canonical']
            recorded = entry.get('eval_key', entry['key'])
            overrides = H.validate_overrides(entry['overrides'] if 'overrides' in entry
                                             else (cfg.get('overrides') or {}))
            new_canon = _canon({int(n): tuple(v) for n, v in canon['nodes'].items()}, canon['investment_year'])
            new_key = H.candidate_key(new_canon)
            mv = entry.get('model_variant')
            kw = dict(case_file_aa=decl, model_variant=mv, ess_ageing_baseline=ess)
            ek_omitted = H.evaluation_key(new_key, overrides, **kw)
            ek_none = H.evaluation_key(new_key, overrides, flex_price_multiplier=None, **kw)
            ek_one = H.evaluation_key(new_key, overrides, flex_price_multiplier=1.0, **kw)
            ek_one_int = H.evaluation_key(new_key, overrides, flex_price_multiplier=1, **kw)
            ok = {'candidate_key': new_key == entry['key'],
                  'eval_key_new_argument_omitted': ek_omitted == recorded,
                  'eval_key_new_argument_none': ek_none == recorded,
                  'eval_key_new_argument_1_0': ek_one == recorded,
                  'eval_key_new_argument_int_1': ek_one_int == recorded,
                  'entry_has_no_flex_price_option': ('flex_price_multiplier' not in entry
                                                     and 'flex_price_label' not in entry)}
            if 'eval_dir' in entry:
                ok['eval_dir'] = H.eval_dir_name(ek_none, entry['label']) == entry['eval_dir']
            if 'working_dir_ids' in entry:
                ok['working_dir_ids'] = H.eval_ids(spec['campaign_id'], ek_none) == entry['working_dir_ids']
            all_ok = all_ok and all(ok.values())
            n_entries += 1
            committed_eval_keys.add(recorded)
            rows.append({'label': entry['label'], 'recorded_eval_key16': recorded[:16],
                         'recomputed_eval_key16': ek_none[:16], 'ok': ok})
        per_spec[rel] = {'campaign_id': spec.get('campaign_id'), 'sha256': H.sha256_file(os.path.join(REPO, rel)),
                         'declares_ess_ageing_baseline': ess is not None,
                         'declares_case_file_aa': decl is not None,
                         'spec_has_flex_price_label': 'flex_price_label' in spec,
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
              'record_has_no_flex_price_field': 'flex_price_multiplier' not in rec}
        if rec.get('eval_key') is not None:
            cfg = rec.get('configuration') or {}
            overrides = H.validate_overrides(rec['evaluation_overrides_effective']
                                             if 'evaluation_overrides_effective' in rec
                                             else (cfg.get('overrides') or {}))
            decl = H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration'))
            kw = dict(case_file_aa=decl, model_variant=rec.get('model_variant'),
                      ess_ageing_baseline=cfg.get('ess_ageing_baseline'))
            ok['eval_key_omitted'] = H.evaluation_key(new_key, overrides, **kw) == rec['eval_key']
            ok['eval_key_1_0'] = H.evaluation_key(new_key, overrides, flex_price_multiplier=1.0, **kw) == rec['eval_key']
            n_eval_key_checked += 1
            committed_eval_keys.add(rec['eval_key'])
        records_ok = records_ok and all(ok.values())
        per_record[rel] = {'label': rec.get('candidate_label'), 'status': rec.get('status'), 'ok': ok}
    per_dir = {d: sum(1 for r in spec_rels if r.startswith(d + '/')) for d in SPEC_DIRS}
    checks = {
        'K1_committed_specs_found': len(spec_rels) > 0,
        'K1_committed_spec_keys_eval_keys_dirs_ids_identical': all_ok and bool(spec_rels),
        'K1_committed_records_keys_identical': records_ok and bool(record_rels),
        'K1_declared_specs_covered': any(v['declares_ess_ageing_baseline'] for v in per_spec.values()),
    }
    scope = {'searched': f'git ls-files -- {" ".join(SPEC_DIRS)}',
             'included': ['campaign_spec_*.json', 'evaluation_record.json'],
             'excluded': ['frozen_s4x_*_spec_*.json (Planner specs: no candidate / eval keys)',
                          'uncommitted campaign roots'],
             'committed_campaign_specs_per_dir': per_dir,
             'note_p515s49': 'P515S49 holds no committed campaign spec or evaluation record at the time of this run',
             'n_campaign_specs': len(spec_rels), 'n_spec_entries': n_entries,
             'n_evaluation_records': len(record_rels), 'n_record_eval_keys_recomputed': n_eval_key_checked}
    return checks, {'scope': scope, 'per_spec': per_spec, 'per_record': per_record}, committed_eval_keys


# ======================================================================================================================
#  K2 -- validation
# ======================================================================================================================
def check_validation():
    bad = {'True': True, 'False': False, 'zero': 0, 'zero_float': 0.0, 'negative': -1.5, 'nan': float('nan'),
           'inf': float('inf'), 'minus_inf': float('-inf'), 'string': '2', 'list': [2.0]}
    refusals = {name: _expect_raise(lambda v=v: H.validate_flex_price_multiplier(v), exc=ValueError)
                for name, v in bad.items()}
    accepted = {'None': H.validate_flex_price_multiplier(None), 'int_2': H.validate_flex_price_multiplier(2),
                'float_1_5': H.validate_flex_price_multiplier(1.5), 'int_1': H.validate_flex_price_multiplier(1)}
    in_key = {'None': H.flex_price_multiplier_in_key(None), '1.0': H.flex_price_multiplier_in_key(1.0),
              '1': H.flex_price_multiplier_in_key(1), '2': H.flex_price_multiplier_in_key(2)}
    checks = {
        'K2_invalid_values_refused': all(r['raised'] for r in refusals.values()),
        'K2_valid_values_normalized_to_float': (accepted['None'] is None and accepted['int_2'] == 2.0
                                                and type(accepted['int_2']) is float and accepted['float_1_5'] == 1.5
                                                and type(accepted['int_1']) is float),
        'K2_key_form_absent_and_1_are_none': in_key['None'] is None and in_key['1.0'] is None and in_key['1'] is None,
        'K2_key_form_2_is_float_2': in_key['2'] == 2.0 and type(in_key['2']) is float,
    }
    return checks, {'refusals': refusals, 'accepted': {k: repr(v) for k, v in accepted.items()},
                    'in_key': {k: repr(v) for k, v in in_key.items()}}


# ======================================================================================================================
#  K3 -- distinct keys
# ======================================================================================================================
def check_distinct(conf, committed_eval_keys):
    keys = {}
    for name, nodes in INSTANCES.items():
        keys[name] = {'omitted': _key(nodes, conf, omit=True), 'None': _key(nodes, conf, None),
                      '1.0': _key(nodes, conf, 1.0)}
        for m in MULTIPLIERS:
            keys[name][repr(m)] = _key(nodes, conf, m)
    ladder = [keys[n][repr(m)] for n in INSTANCES for m in MULTIPLIERS]
    int2 = _key(INSTANCES['x0'], conf, 2)
    near = _key(INSTANCES['x0'], conf, math.nextafter(1.5, 2.0))
    checks = {
        'K3_absent_none_1_0_equal_committed_baseline_keys': all(
            keys[n]['omitted'] == keys[n]['None'] == keys[n]['1.0'] == COMMITTED_BASELINE_KEYS[n] for n in INSTANCES),
        'K3_six_ladder_keys_pairwise_distinct': len(set(ladder)) == len(ladder) == 6,
        'K3_ladder_keys_distinct_from_baseline_keys': not (set(ladder) & set(COMMITTED_BASELINE_KEYS.values())),
        'K3_ladder_keys_distinct_from_every_committed_eval_key': not (set(ladder) & committed_eval_keys),
        'K3_ladder_keys_distinct_from_bare_candidate_keys': not (set(ladder) & {
            H.candidate_key(_canon(nodes, YEAR)) for nodes in INSTANCES.values()}),
        'K3_int_2_equals_float_2': int2 == keys['x0']['2.0'],
        'K3_key_sensitive_to_one_ulp_of_m': near != keys['x0']['1.5'],
    }
    return checks, {'keys': keys, 'n_committed_eval_keys_compared': len(committed_eval_keys),
                    'baseline_configuration': {k: conf[k] for k in ('case_file_anderson_acceleration',
                                                                    'ess_ageing_baseline')}}


# ======================================================================================================================
#  K4 -- fixture freezes (temporary directories)
# ======================================================================================================================
def check_fixture_freeze(conf):
    tmp = tempfile.mkdtemp(prefix='p515s49_w33_fixture_')
    configuration = {'name': 'W33 fixture (temporary, not an artifact)', 'arm_label': 's39_D', 'overrides': {},
                     'case_file_anderson_acceleration': dict(conf['case_file_anderson_acceleration']),
                     'ess_ageing_baseline': copy.deepcopy(conf['ess_ageing_baseline']),
                     'ess_ageing_baseline_label': conf['ess_ageing_baseline_label']}
    detail, checks = {}, {}
    try:
        cands = [(f'{n}_m{repr(m).replace(".", "p")}', nodes, {'investment_year': YEAR, 'flex_price_multiplier': m})
                 for n, nodes in INSTANCES.items() for m in MULTIPLIERS]
        cands.append(('x0_m1_explicit', INSTANCES['x0'], {'investment_year': YEAR, 'flex_price_multiplier': 1.0}))
        root = os.path.join(tmp, 'ok')
        path, sha, spec = H.freeze_campaign_spec(root, 'w33_fixture', cands, configuration, cap=CAP, concurrency=5,
                                                 authority=['W33 fixture'], extra={'fixture': True})
        by_label = {e['label']: e for e in spec['candidates']}
        ladder_ok = []
        for label, nodes, opts in cands:
            e = by_label[label]
            m = opts['flex_price_multiplier']
            expect = _key(nodes, conf, m)
            if m == 1.0:
                ladder_ok.append(e['eval_key'] == COMMITTED_BASELINE_KEYS['x0'] and 'flex_price_label' not in e
                                 and e['flex_price_multiplier'] == 1.0)
            else:
                ladder_ok.append(e['eval_key'] == expect and e['flex_price_label'] == H.FLEX_PRICE_LABEL
                                 and e['flex_price_multiplier'] == m
                                 and e['eval_dir'] == H.eval_dir_name(expect, label))
        checks['K4_fixture_freezes_entries_keys_labels_recompute'] = all(ladder_ok)
        checks['K4_spec_carries_the_label'] = spec.get('flex_price_label') == H.FLEX_PRICE_LABEL
        checks['K4_label_text'] = H.FLEX_PRICE_LABEL == 'MODEL VARIANT — flexibility price × m'
        detail['fixture_spec'] = {'sha256': sha, 'entries': {e['label']: {'eval_key16': e['eval_key'][:16],
                                                                          'flex_price_multiplier': e.get('flex_price_multiplier'),
                                                                          'flex_price_label': e.get('flex_price_label')}
                                                             for e in spec['candidates']}}
        dup_root = os.path.join(tmp, 'dup')
        r_dup = _expect_raise(lambda: H.freeze_campaign_spec(
            dup_root, 'w33_dup', [('x0', INSTANCES['x0'], {'investment_year': YEAR}),
                                  ('x0_m1', INSTANCES['x0'], {'investment_year': YEAR, 'flex_price_multiplier': 1.0})],
            configuration, cap=CAP, concurrency=5, authority=['W33 fixture']), exc=ValueError)
        checks['K4_m1_beside_absent_refused_as_duplicate'] = r_dup['raised'] and not os.path.exists(dup_root)
        bad_root = os.path.join(tmp, 'bad')
        r_bad = _expect_raise(lambda: H.freeze_campaign_spec(
            bad_root, 'w33_bad', [('x0', INSTANCES['x0'], {'investment_year': YEAR, 'flex_price_multiplier': -2.0})],
            configuration, cap=CAP, concurrency=5, authority=['W33 fixture']), exc=ValueError)
        checks['K4_invalid_multiplier_refused_no_directory'] = r_bad['raised'] and not os.path.exists(bad_root)
        plain_root = os.path.join(tmp, 'plain')
        _p, _s, plain = H.freeze_campaign_spec(plain_root, 'w33_plain', [('x0', INSTANCES['x0'],
                                                                          {'investment_year': YEAR})],
                                               configuration, cap=CAP, concurrency=5, authority=['W33 fixture'])
        checks['K4_spec_without_option_has_no_new_fields'] = (
            'flex_price_label' not in plain and all('flex_price_multiplier' not in e and 'flex_price_label' not in e
                                                    for e in plain['candidates'])
            and plain['candidates'][0]['eval_key'] == COMMITTED_BASELINE_KEYS['x0'])
        detail['refusals'] = {'duplicate': r_dup, 'invalid': r_bad}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return checks, detail


# ======================================================================================================================
#  K5 -- read-back and digest comparison on the oracle-path build
# ======================================================================================================================
def _spec_like(conf):
    return {'configuration': {'overrides': {}, 'arm_label': 's39_D',
                              'case_file_anderson_acceleration': dict(conf['case_file_anderson_acceleration']),
                              'ess_ageing_baseline': copy.deepcopy(conf['ess_ageing_baseline']),
                              'ess_params_file': dict(conf['ess_params_file']),
                              'ess_ageing_baseline_label': conf['ess_ageing_baseline_label']},
            'cap': CAP, 'required_consecutive_cycles': 10}


def oracle_path_build(G, srp, W21, conf, name, nodes, m, run_tag, scratch, floor_rows):
    """W21's oracle-path build (p515_s47_case_file_baseline_check.oracle_path_build) with the multiplier passed to
    the child's configuration hook; DSO blocks additionally keep their objective repn summary, and the DSO models
    handed to `optimize` are kept for the post-run read-back function."""
    report = {}
    tag = 'none' if m is None else repr(m).replace('.', 'p')
    eval_id = f'p515s49_w33_checks_{run_tag}_{name}_{tag}'
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(scratch, f'{name}_{tag}'), report, investment_map=nodes, eval_id=eval_id,
        num_max_iters_override=CAP, apply_rho=False, investment_year=YEAR)
    holder = {}
    blocks, summaries, dso_models = {}, {}, {}

    def network_stub(block_tag, node_id=None):
        def stub(model, *args, **kwargs):
            results = {}
            if node_id is not None:
                dso_models[node_id] = model
            for year in model:
                results[year] = {}
                for day in model[year]:
                    b, comps, _items = W21.block_state(model[year][day], keep_items=False)
                    blocks[f'{block_tag}|{year}|{day}'] = {'sha256': b, 'components': comps}
                    if node_id is not None:
                        summaries[f'{block_tag}|{year}|{day}'] = H._flex_price_objective_summary(model[year][day])
                    results[year][day] = None
            return results
        return stub

    def esso_stub(models, *args, **kwargs):
        for node_id, model in models.items():
            b, comps, _items = W21.block_state(model, keep_items=False)
            blocks[f'ESSO|node{node_id}'] = {'sha256': b, 'components': comps}
        return {node_id: None for node_id in models}

    t0 = time.time()
    H._config_hook_factory(_spec_like(conf), holder, overrides={}, investment_year=YEAR,
                           expected_floor_rows=floor_rows, flex_price_multiplier=m)(
        planning=planning, sed=sed, candidate=candidate, report=report)
    hook_s = time.time() - t0
    planning.transmission_network.optimize = network_stub('TSO')
    for node_id, dn in planning.distribution_networks.items():
        dn.optimize = network_stub(f'DSO|{node_id}', node_id)
    sed.optimize = esso_stub
    consensus_vars, _dual_vars = srp.create_admm_variables(planning)
    srp.create_distribution_networks_models(planning.distribution_networks, consensus_vars,
                                            candidate['total_capacity'],
                                            parallel_execution=planning.parallel_execution)
    srp.create_transmission_network_model(planning, consensus_vars, candidate['total_capacity'])
    srp.create_shared_energy_storage_model(sed, consensus_vars, candidate['investment'])
    post = None
    if m is not None:
        post = H.flex_price_readback_run_models(dso_models, planning, holder['_flex_price_original_arrays'], m)
        post = {k: v for k, v in post.items() if k != 'per_block'} | {
            'n_blocks_failing': sum(1 for v in post['per_block'].values()
                                    if not (v['equals_applied_closed_form'] and v['equals_m_times_original_within_tol']))}
    applied = holder.get('flex_price_applied') or {}
    return {'name': name, 'm': m, 'eval_id': eval_id, 'hook_s': hook_s,
            'candidate_key': H.candidate_key(_canon(nodes, YEAR)),
            'hook_ess_ageing_readback_all_match': ((holder.get('ess_ageing_verified_pre_run') or {}).get(
                'readback_pre_run') or {}).get('all_match'),
            'hook_flex_checks': applied.get('checks'),
            'hook_flex_readback': {k: (applied.get('readback_pre_run') or {}).get(k)
                                   for k in ('n_blocks', 'all_match', 'n_flex_coefficients', 'n_bitwise_exact',
                                             'max_rel_dev', 'rel_tol')} if applied else None,
            'hook_profile_identical_across_dso': applied.get('profile_identical_across_dso'),
            'hook_profile_applied_2025': ({k: v for k, v in applied.get('profile_applied_per_year_day', {}).items()
                                           if k.startswith('2025|')} if applied else None),
            'rule_eleven_entry': (report.get('rule_eleven_checklist') or {}).get('w33_flex_price_multiplier'),
            'post_run_readback_on_built_dso_blocks': post,
            'blocks': blocks, 'summaries': summaries}


def compare_builds(ref, other, m):
    """Build at m (other) vs the build without the multiplier (ref)."""
    out = {'n_blocks_ref': len(ref['blocks']), 'n_blocks_other': len(other['blocks']),
           'same_block_set': set(ref['blocks']) == set(other['blocks'])}
    identical, differing = [], {}
    for key in sorted(ref['blocks']):
        a, b = ref['blocks'][key], other['blocks'].get(key)
        if b is None:
            continue
        if a['sha256'] == b['sha256']:
            identical.append(key)
        else:
            differing[key] = sorted(c for c in set(a['components']) | set(b['components'])
                                    if a['components'].get(c) != b['components'].get(c))
    out['n_identical'] = len(identical)
    out['differing_blocks'] = differing
    tso_esso_differ = [k for k in differing if not k.startswith('DSO|')]
    dso_keys = [k for k in ref['blocks'] if k.startswith('DSO|')]
    other_components = sorted({c for comps in differing.values() for c in comps} - set(FLEX_COMPONENTS))
    repn = {}
    for key in dso_keys:
        checks, figures = H.compare_flex_price_summaries(ref['summaries'][key], other['summaries'][key], 1.0 if m is None else m)
        repn[key] = {'checks': checks, **figures}
    out['tso_or_esso_blocks_differing'] = tso_esso_differ
    out['components_differing_outside_flex_components'] = other_components
    out['dso_objective_repn'] = {
        'n_blocks': len(repn), 'all_match': all(all(v['checks'].values()) for v in repn.values()),
        'n_flex_coefficients': sum(v['n_flex_coefficients'] for v in repn.values()),
        'n_bitwise_exact': sum(v['n_bitwise_exact'] for v in repn.values()),
        'max_rel_dev': max((v['max_rel_dev'] or 0.0) for v in repn.values()) if repn else None,
        'failing_blocks': [k for k, v in repn.items() if not all(v['checks'].values())]}
    return out


def check_readback(G, srp, W21, PG, conf, run_tag, scratch):
    _cc, floor_rows, _fc = PG._build_floor_rows(f'p515s49_w33_checks_{run_tag}_precheck')
    per_instance, checks = {}, {}
    for name, nodes in INSTANCES.items():
        _log(f'K5 {name}: build without the multiplier')
        ref = oracle_path_build(G, srp, W21, conf, name, nodes, None, run_tag, scratch, floor_rows)
        rows = {'none': {k: v for k, v in ref.items() if k not in ('blocks', 'summaries')}}
        for m in (1.0,) + MULTIPLIERS:
            _log(f'K5 {name}: build at m = {m}')
            other = oracle_path_build(G, srp, W21, conf, name, nodes, m, run_tag, scratch, floor_rows)
            cmp = compare_builds(ref, other, m)
            rows[repr(m)] = {**{k: v for k, v in other.items() if k not in ('blocks', 'summaries')},
                             'comparison_vs_no_multiplier': cmp}
            pre = other['hook_flex_readback'] or {}
            post = other['post_run_readback_on_built_dso_blocks'] or {}
            pfx = f'K5_{name}_m{repr(m)}'
            checks[f'{pfx}_hook_applied_and_read_back'] = (bool(other['hook_flex_checks'])
                                                          and all(other['hook_flex_checks'].values())
                                                          and pre.get('all_match') is True
                                                          and pre.get('n_blocks') == 36)
            checks[f'{pfx}_ess_ageing_readback_still_matches'] = other['hook_ess_ageing_readback_all_match'] is True
            checks[f'{pfx}_same_block_set'] = cmp['same_block_set'] and cmp['n_blocks_ref'] == 51
            checks[f'{pfx}_post_run_readback_function_matches'] = post.get('all_match') is True
            checks[f'{pfx}_dso_objective_flex_coefficients_scale_rest_identical'] = (
                cmp['dso_objective_repn']['all_match'] and cmp['dso_objective_repn']['n_blocks'] == 36)
            if m == 1.0:
                checks[f'{pfx}_every_block_digest_identical'] = not cmp['differing_blocks']
                checks[f'{pfx}_bitwise_coefficients'] = (cmp['dso_objective_repn']['n_bitwise_exact']
                                                         == cmp['dso_objective_repn']['n_flex_coefficients'] > 0)
            else:
                checks[f'{pfx}_tso_esso_blocks_identical'] = not cmp['tso_or_esso_blocks_differing']
                checks[f'{pfx}_only_flex_components_differ'] = not cmp['components_differing_outside_flex_components']
                checks[f'{pfx}_every_dso_block_differs'] = sorted(cmp['differing_blocks']) == sorted(
                    k for k in ref['blocks'] if k.startswith('DSO|'))
                if m == 2.0:
                    checks[f'{pfx}_power_of_two_bitwise'] = (cmp['dso_objective_repn']['n_bitwise_exact']
                                                             == cmp['dso_objective_repn']['n_flex_coefficients'] > 0)
            _log(f"K5 {name} m={m}: differing blocks {len(cmp['differing_blocks'])}, components outside flex "
                 f"{cmp['components_differing_outside_flex_components']}, repn {cmp['dso_objective_repn']['all_match']} "
                 f"({cmp['dso_objective_repn']['n_bitwise_exact']}/{cmp['dso_objective_repn']['n_flex_coefficients']} "
                 f"bitwise, max rel dev {cmp['dso_objective_repn']['max_rel_dev']}); hook {pre.get('all_match')}; "
                 f"post {post.get('all_match')}; hook {other['hook_s']:.1f}s")
        per_instance[name] = rows
    return checks, per_instance


# ======================================================================================================================
#  K6 -- refusals
# ======================================================================================================================
def check_refusals(G, PG, conf, run_tag, scratch):
    _cc, floor_rows, _fc = PG._build_floor_rows(f'p515s49_w33_checks_{run_tag}_refusal_precheck')
    report = {}
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(scratch, 'sabotage'), report, investment_map=INSTANCES['unit_n7_4h_e1'],
        eval_id=f'p515s49_w33_checks_{run_tag}_sabotage', num_max_iters_override=CAP, apply_rho=False,
        investment_year=YEAR)
    original_fn = H._scaled_cost_flex
    calls = {'n': 0}

    def sabotaged(array, m):
        calls['n'] += 1
        return original_fn(array, m * (1.0 + 1e-7) if calls['n'] == 7 else m)

    H._scaled_cost_flex = sabotaged
    try:
        r_sab = _expect_raise(lambda: H._config_hook_factory(
            _spec_like(conf), {}, overrides={}, investment_year=YEAR, expected_floor_rows=floor_rows,
            flex_price_multiplier=2.0)(planning=planning, sed=sed, candidate=candidate, report=report),
            exc=RuntimeError)
    finally:
        H._scaled_cost_flex = original_fn
    restored = H._scaled_cost_flex is original_fn
    tmp = tempfile.mkdtemp(prefix='p515s49_w33_child_')
    try:
        spec = _spec_like(conf)
        spec['configuration']['case_file_anderson_acceleration'] = dict(conf['case_file_anderson_acceleration'])
        canon = _canon(INSTANCES['x0'], YEAR)
        entry = {'label': 'x0_m2_unlabelled', 'canonical': canon, 'key': H.candidate_key(canon), 'overrides': {},
                 'flex_price_multiplier': 2.0, 'post_certification': None,
                 'working_dir_ids': {'run': f'p515s49_w33_never_{run_tag}_run',
                                     'precheck': f'p515s49_w33_never_{run_tag}_precheck'}}
        r_label = _expect_raise(lambda: H._child_real(None, spec, None, entry, tmp, None, None, time.time()),
                                exc=RuntimeError)
        child_wrote_nothing = os.listdir(tmp) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    checks = {'K6_sabotaged_scaling_makes_the_hook_raise': (r_sab['raised']
                                                           and 'did not take effect' in r_sab.get('message', '')),
              'K6_monkeypatch_restored': restored,
              'K6_unlabelled_child_refuses_before_any_build': (r_label['raised']
                                                              and 'without the label' in r_label.get('message', '')
                                                              and child_wrote_nothing)}
    return checks, {'sabotage': r_sab, 'unlabelled_child': r_label, 'sabotaged_call_index': 7}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=OUT_REL, help='output directory (write-once)')
    args = parser.parse_args()
    out_root = os.path.join(REPO, args.out)
    if os.path.exists(out_root):
        raise SystemExit(f'output directory exists (write-once): {out_root}')
    os.makedirs(out_root)
    scratch = tempfile.mkdtemp(prefix='p515s49_w33_checks_')
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    started = time.time()
    head = _git(['rev-parse', 'HEAD']).strip()
    dirty = _git(['status', '--porcelain', '--', 'p515_s44_campaign_harness.py', os.path.basename(__file__)]).strip()
    _log(STAGE)
    _log(f'git HEAD {head}; harness sha256 {H.sha256_file(H.HARNESS_PATH)}; uncommitted: {dirty or "none"}')

    import p515_g_g1_g4_admm_gates as G
    import p515_s40_polish_gap as PG
    import shared_resources_planning as srp
    import p515_s47_case_file_baseline_check as W21  # its block_state digest, BY IMPORT (installs its own 0-guard)

    conf, same = _baseline_configuration()
    checks, detail = {'K0_recert_and_x0_capture_configurations_identical': all(same.values())}, {'K0': same}
    c, d, committed = check_committed()
    checks.update(c)
    detail['K1'] = d
    _log(f"K1: {c} scope {d['scope']}")
    c, d = check_validation()
    checks.update(c)
    detail['K2'] = d
    _log(f'K2: {c}')
    c, d = check_distinct(conf, committed)
    checks.update(c)
    detail['K3'] = d
    _log(f'K3: {c}')
    for n, ks in d['keys'].items():
        _log(f'   {n}: ' + ', '.join(f'{m}={k[:16]}' for m, k in ks.items()))
    c, d = check_fixture_freeze(conf)
    checks.update(c)
    detail['K4'] = d
    _log(f'K4: {c}')
    c, d = check_readback(G, srp, W21, PG, conf, run_tag, scratch)
    checks.update(c)
    detail['K5'] = d
    c, d = check_refusals(G, PG, conf, run_tag, scratch)
    checks.update(c)
    detail['K6'] = d
    _log(f'K6: {c}')

    guard_failures = GUARD.verify(0)
    w21_guard_failures = W21.GUARD.verify(0)
    checks['guard_zero_solves_verified'] = not guard_failures
    checks['w21_module_guard_zero_solves_verified'] = not w21_guard_failures
    all_ok = all(checks.values())
    payload = {'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': _utc(), 'git_head_at_run': head,
               'uncommitted_at_run': dirty or None,
               'script': os.path.basename(__file__), 'script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'harness_sha256': H.sha256_file(H.HARNESS_PATH),
               'design': {
                   'where_applied': ('the campaign child configuration hook (= run_admm_arm pre_solve_hook), LAST, before '
                                     'run_operational_planning builds any DSO model'),
                   'why_before_build': ('model_construction_helpers.flexibility_cost multiplies the numpy value '
                                        'cost_flex[s_m][p] * baseMVA into the flex_cost_scenario Expression at build '
                                        'time (a constant, no mutable Param); applying it to the data before the build '
                                        'is how production itself would build the objective at another price'),
                   'never_in_place': ('the same planning.cost_flex[year][day] array object is bound to the TSO and to '
                                      'every DSO block; the DSO arrays are replaced by new arrays m * cost_flex'),
                   'key_rule': ('eval key payload + flex_price_multiplier only when present and != 1.0 '
                                '(p515_s44_campaign_harness.evaluation_key)'),
                   'label': H.FLEX_PRICE_LABEL, 'readback_rel_tol': H.FLEX_PRICE_READBACK_REL_TOL,
                   'flex_components_allowed_to_differ': list(FLEX_COMPONENTS)},
               'baseline_configuration': conf, 'checks': checks, 'all_ok': all_ok, 'detail': detail,
               'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures,
                                       'w21_module_guard_counts': dict(W21.GUARD.counts),
                                       'w21_module_guard_verify_0_failures': w21_guard_failures},
               'wall_clock_s': time.time() - started}
    with open(os.path.join(out_root, 'flex_price_checks.json'), 'w') as handle:
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
    _log(f'ALL_OK={all_ok} guard={dict(GUARD.counts)} w21_guard={dict(W21.GUARD.counts)} '
         f'wall={time.time() - started:.1f}s')
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
