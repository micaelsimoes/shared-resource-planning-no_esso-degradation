"""
P5.15 Addendum 27 (task W14, "harness support for investment years other than
2025") -- ZERO-SOLVE gate. Armed `SolveProfileGuard(permitted=())` installed
before any model import and `verify(0)` at the end.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27 (W14); frozen spec v15
`data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json` A1
`year_ladder` ("investment years 2030 and 2035 at the best node").

WHAT W14 CHANGED (the three sites this gate covers)
  1. `p515_s44_campaign_harness.freeze_campaign_spec` -- an evaluation option
     `investment_year` (default `INVESTMENT_YEAR` = 2025) reaching
     `canonical_candidate`; `_spec_candidate` accepts the canonical shape
     `{'investment_year': y, 'nodes': {...}}` for batch resolution.
  2. `p515_s44_campaign_harness._child_real` -- the 2025-only refusal replaced
     by validation against THIS instance's investment years
     (`instance_investment_years`, read from the shared-ESS data), forwarded to
     `run_admm_arm(..., investment_year=...)`.
  3. `p515_g_g1_g4_admm_gates._construct_arm_planning` -- writes the investment
     at the candidate's own cohort year (`investment_year`, default
     `N.INVEST_YEAR`), after validating it against `sed.years`.
  Multi-cohort (staging) candidates are NOT in scope and are not supported.

CHECKS (production / harness functions only, never re-implemented)
  (i)   Every COMMITTED campaign spec under `data/SRP1/Results/P515S44/` and
        `data/SRP1/Results/P515S45/` (found via `git ls-files`), every entry:
        the canonical form recomputed from its own recorded nodes + year, the
        candidate key, the eval key, the eval dir and the working-dir ids all
        equal what the spec records, byte for byte. Same for the candidate
        canonical/key of every committed `evaluation_record.json`.
        SCOPE of the claim: committed files only, those two directories only,
        campaign specs (schema `p515_s44_campaign_spec_v*`) and evaluation
        records only -- the planner specs v14/v15 carry no candidate keys.
  (ii)  The 2030/2035 fixture: every candidate of
        `data/SRP1/Results/P515S45/investment_cost/investment_cost_results.json`
        (W2, commit 9e623dd3 -- which computed its canonical forms and keys by
        calling `canonical_candidate(..., investment_year=year)` directly),
        deduplicated by key, re-frozen THROUGH `freeze_campaign_spec` with the
        `investment_year` evaluation option, into a TEMPORARY directory (a
        fixture, not an artifact). Entry canonical and key must equal the
        fixture's, for all three years.
  (iii) A real, unsolved planning object for a 2030 single-node candidate built
        by `_construct_arm_planning` (the code `run_admm_arm` calls): the
        investment sits at 2030 and NOT at 2025, `_rebuild_candidate_total_
        capacities` gives total capacity 0 in 2025 and the invested value in
        2030 and 2035, and after production's own `_restore_candidate_data`
        the shared-ESS cohort objects and the TSO network data carry the same.
        The ESSO subproblem for the node, built and loaded with the candidate
        by production (`_build_subproblem`, `_update_model_with_candidate_
        solution` -> `_configure_esso_cohort_state`), reports cohort 2025
        INACTIVE and cohort 2030 ACTIVE. No solve.
  (iv)  The same construction at 2025, in the same process: `report['instance']`
        and the whole candidate structure equal a build with `investment_year`
        left at its default -- today's structure exactly.
  (v)   A year outside the instance's investment years raises: in
        `_construct_arm_planning` (ValueError) and on the CHILD path
        (`_child_real`, RuntimeError, raised before `_build_floor_rows` and
        before any solve -- the guard proves it).
  (vi)  `_config_hook_factory` on the AA-on case file, with a
        `case_file_anderson_acceleration` declaration, run against the 2030
        planning object of (iii): every configuration check passes.

Output (write-once files, directory created by the launcher):
  data/SRP1/Results/P515S45/year_support_checks/
      year_support_checks_<tag>.json, launch_<tag>.log, manifest_sha256.json,
      scratch_<tag>/ (results-dir redirect of the unsolved planning objects)
  `<tag>` is the optional first CLI argument (default `r1`) and also suffixes the
  `p56a_oracle.WORK_DIR` eval ids, so a re-run neither overwrites nor is blocked
  by a previous run's artifacts.

Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S45/year_support_checks
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s45_year_support_checks.py r1 \\
        > data/SRP1/Results/P515S45/year_support_checks/launch_r1.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s45_year_support_checks.py --manifest
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W14 year-support checks (zero solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

STAGE = 'P5.15 Addendum 27 W14 -- campaign support for investment years other than 2025 (zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 27 (W14)',
             'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json (A1 year_ladder)']
OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S45', 'year_support_checks')
MANIFEST_NAME = 'manifest_sha256.json'
# Every run is tagged (default r1, overridable as the first CLI argument): the results file,
# the scratch sub-directory and the `p56a_oracle.WORK_DIR` eval ids all carry the tag, so a
# re-run can never write onto -- or be blocked by -- a previous run's artifacts.
RUN_TAG = 'r1'

SPEC_DIRS = ('data/SRP1/Results/P515S44', 'data/SRP1/Results/P515S45')
W2_FIXTURE_REL = 'data/SRP1/Results/P515S45/investment_cost/investment_cost_results.json'
W2_FIXTURE_COMMIT = '9e623dd3'

# the (iii)/(iv) instance: one A1 ladder point, node 7 at 2 h, E = 3 MWh
PROBE_NODE = 7
PROBE_S_MVA, PROBE_E_MWH = 1.5, 3.0
PROBE_YEAR, CONTROL_YEAR = 2030, 2025
BAD_YEAR = 2027
CAP, REQUIRED_CYCLES = 500, 10
EXPECTED_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}


def _eval_id(name):
    return f'p515s45_year_support_probe_{name}_{RUN_TAG}'


def _results_name():
    return f'year_support_checks_{RUN_TAG}.json'


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True,
                          check=True).stdout


def _load_json(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _expect_raise(fn, exc):
    try:
        fn()
    except exc as error:
        return {'raised': True, 'type': type(error).__name__, 'message': str(error)}
    except Exception as error:  # noqa: BLE001 -- wrong type is a failure, recorded
        return {'raised': False, 'wrong_type': type(error).__name__, 'message': str(error),
                'traceback': traceback.format_exc()}
    return {'raised': False}


def _node_map_from_canonical(canonical):
    return {int(n): (float(v[0]), float(v[1])) for n, v in canonical['nodes'].items()}


def _probe_map(node=PROBE_NODE, s=PROBE_S_MVA, e=PROBE_E_MWH):
    return {n: ((s, e) if n == node else (0.0, 0.0)) for n in H.ACTIVE_NODES}


# ======================================================================================================================
#  (i) -- every committed campaign spec and evaluation record recomputes identical
# ======================================================================================================================
def check_committed_identity():
    checks, detail = {}, {}
    tracked = _git(['ls-files', '--', *SPEC_DIRS]).splitlines()
    spec_rels = sorted(r for r in tracked
                       if os.path.basename(r).startswith('campaign_spec_') and r.endswith('.json'))
    record_rels = sorted(r for r in tracked if os.path.basename(r) == 'evaluation_record.json')

    per_spec, all_ok, n_entries = {}, True, 0
    for rel in spec_rels:
        spec = _load_json(rel)
        cfg = spec.get('configuration') or {}
        decl = H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration'))
        rows = []
        for entry in spec['candidates']:
            recorded_canon = entry['canonical']
            recorded_ekey = entry.get('eval_key', entry['key'])
            overrides = H.validate_overrides(entry['overrides'] if 'overrides' in entry
                                             else (cfg.get('overrides') or {}))
            # recompute the CANONICAL FORM itself from its own nodes + year, then everything keyed off it
            new_canon = H.canonical_candidate(_node_map_from_canonical(recorded_canon),
                                              investment_year=recorded_canon['investment_year'])
            new_key = H.candidate_key(new_canon)
            new_ekey = H.evaluation_key(new_key, overrides, case_file_aa=decl)
            ok = {
                'canonical_json_identical': (json.dumps(new_canon, sort_keys=True, separators=(',', ':'))
                                             == json.dumps(recorded_canon, sort_keys=True,
                                                           separators=(',', ':'))),
                'candidate_key': new_key == entry['key'],
                'eval_key': new_ekey == recorded_ekey,
            }
            if 'eval_dir' in entry:
                ok['eval_dir'] = H.eval_dir_name(new_ekey, entry['label']) == entry['eval_dir']
            if 'working_dir_ids' in entry:
                ok['working_dir_ids'] = H.eval_ids(spec['campaign_id'], new_ekey) == entry['working_dir_ids']
            all_ok = all_ok and all(ok.values())
            n_entries += 1
            rows.append({'label': entry['label'],
                         'investment_year': recorded_canon['investment_year'],
                         'recorded_key16': entry['key'][:16], 'recomputed_key16': new_key[:16],
                         'recorded_eval_key16': recorded_ekey[:16], 'recomputed_eval_key16': new_ekey[:16],
                         'ok': ok})
        per_spec[rel] = {'schema': spec.get('schema'), 'campaign_id': spec.get('campaign_id'),
                         'sha256': H.sha256_file(os.path.join(REPO, rel)),
                         'n_candidates': len(rows), 'entries': rows}

    per_record, records_ok = {}, True
    for rel in record_rels:
        record = _load_json(rel)
        canon = record.get('candidate_canonical')
        if not canon:
            per_record[rel] = {'skipped': 'record carries no candidate_canonical'}
            continue
        new_canon = H.canonical_candidate(_node_map_from_canonical(canon),
                                          investment_year=canon['investment_year'])
        new_key = H.candidate_key(new_canon)
        ok = {'canonical_json_identical': (json.dumps(new_canon, sort_keys=True, separators=(',', ':'))
                                           == json.dumps(canon, sort_keys=True, separators=(',', ':'))),
              'candidate_key': new_key == record.get('candidate_key')}
        records_ok = records_ok and all(ok.values())
        per_record[rel] = {'label': record.get('candidate_label'),
                           'investment_year': canon['investment_year'],
                           'recorded_key16': str(record.get('candidate_key'))[:16],
                           'recomputed_key16': new_key[:16], 'ok': ok}

    checks['i_committed_campaign_specs_found'] = len(spec_rels) > 0
    checks['i_committed_campaign_spec_keys_byte_identical'] = all_ok and bool(spec_rels)
    checks['i_committed_evaluation_record_keys_byte_identical'] = records_ok and bool(record_rels)
    checks['i_all_committed_specs_are_2025'] = all(
        e['investment_year'] == 2025 for v in per_spec.values() for e in v['entries'])
    detail['i_scope'] = {
        'searched': f'git ls-files -- {" ".join(SPEC_DIRS)}',
        'included': ['campaign_spec_*.json (campaign specs)', 'evaluation_record.json (evaluation records)'],
        'excluded': ['frozen_s44_selection_spec_v14_e4500e27.json and '
                     'frozen_s45_phaseA_spec_v15_5feefd7b.json: PLANNER specs, no candidate keys',
                     'uncommitted campaign roots'],
        'n_campaign_specs': len(spec_rels), 'n_spec_entries': n_entries,
        'n_evaluation_records': len(record_rels),
    }
    detail['i_per_spec'] = per_spec
    detail['i_per_evaluation_record'] = per_record
    return checks, detail


# ======================================================================================================================
#  (ii) -- 2030/2035 candidates through freeze_campaign_spec against the W2 fixture
# ======================================================================================================================
def check_year_fixture():
    checks, detail = {}, {}
    fixture = _load_json(W2_FIXTURE_REL)
    entries = fixture['candidates']

    # dedupe by candidate key: the fixture holds the same canonical candidate under several labels
    # (e.g. lattice_plan / lattice_plan_n7_p1.5_e3.0 / n7_2h_e3), and a spec's keys must be unique.
    by_key, collisions = {}, {}
    for label, value in entries.items():
        key = value['candidate_key']
        if key in by_key:
            collisions.setdefault(key, [by_key[key]]).append(label)
            continue
        by_key[key] = label
    detail['ii_key_collisions_in_fixture'] = {k[:16]: v for k, v in collisions.items()}

    candidates = []
    for key, label in by_key.items():
        canon = entries[label]['candidate_canonical']
        candidates.append((label, _node_map_from_canonical(canon),
                           {'investment_year': canon['investment_year']}))
    candidates.sort(key=lambda item: item[0])

    tmp_root = tempfile.mkdtemp(prefix='p515s45_w14_fixture_spec_')
    try:
        campaign_root = os.path.join(tmp_root, 'campaign')
        spec_path, spec_sha, spec = H.freeze_campaign_spec(
            campaign_root=campaign_root, campaign_id='w14_year_fixture', candidates=candidates,
            configuration={'name': 'D (case file) -- fixture only, never run',
                           'case_file_anderson_acceleration': EXPECTED_AA},
            cap=CAP, concurrency=1, authority=AUTHORITY, required_consecutive_cycles=REQUIRED_CYCLES)
        frozen_entries = {e['label']: e for e in spec['candidates']}
        detail['ii_fixture_spec'] = {'temporary_path': spec_path, 'sha256': spec_sha,
                                     'n_entries': len(spec['candidates']),
                                     'note': 'temporary fixture, deleted after the check'}
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    rows, ok_by_year = [], {}
    for label, value in entries.items():
        key = value['candidate_key']
        if by_key[key] != label:
            continue                                  # a collision alias, already checked under its first label
        entry = frozen_entries[label]
        year = value['candidate_canonical']['investment_year']
        ok = {
            'canonical_json_identical': (json.dumps(entry['canonical'], sort_keys=True, separators=(',', ':'))
                                         == json.dumps(value['candidate_canonical'], sort_keys=True,
                                                       separators=(',', ':'))),
            'candidate_key': entry['key'] == key,
            'canonical_year_is_the_requested_year': entry['canonical']['investment_year'] == year,
        }
        acc = ok_by_year.setdefault(year, {'n': 0, 'all_ok': True})
        acc['n'] += 1
        acc['all_ok'] = acc['all_ok'] and all(ok.values())
        rows.append({'label': label, 'investment_year': year, 'fixture_key16': key[:16],
                     'frozen_key16': entry['key'][:16], 'ok': ok})

    # the (iii)/(iv) probe IS a fixture point: n7_2h_e3_y2030 / n7_2h_e3_y2035 / n7_2h_e3
    probe_links = {}
    for label, year in (('n7_2h_e3_y2030', 2030), ('n7_2h_e3_y2035', 2035), ('n7_2h_e3', 2025)):
        recomputed = H.candidate_key(H.canonical_candidate(_probe_map(), investment_year=year))
        probe_links[label] = {'year': year, 'fixture_key': entries[label]['candidate_key'],
                              'recomputed_key': recomputed,
                              'match': entries[label]['candidate_key'] == recomputed}
    detail['ii_probe_candidate_links'] = probe_links
    checks['ii_probe_candidate_is_a_fixture_point'] = all(v['match'] for v in probe_links.values())

    checks['ii_fixture_read'] = len(entries) > 0
    for year in (2025, 2030, 2035):
        acc = ok_by_year.get(year, {'n': 0, 'all_ok': False})
        checks[f'ii_year_{year}_keys_match_w2_fixture'] = acc['all_ok'] and acc['n'] > 0
    detail['ii_scope'] = {'fixture': W2_FIXTURE_REL, 'fixture_commit': W2_FIXTURE_COMMIT,
                          'fixture_sha256': H.sha256_file(os.path.join(REPO, W2_FIXTURE_REL)),
                          'n_fixture_candidates': len(entries), 'n_unique_keys': len(by_key),
                          'per_year': {str(k): v for k, v in sorted(ok_by_year.items())}}
    detail['ii_entries'] = rows
    return checks, detail


# ======================================================================================================================
#  (iii) / (iv) / (vi) -- real, unsolved planning objects at 2030 and at 2025
# ======================================================================================================================
def _capacity_tables(planning, sed, candidate, node):
    """The ACTUAL data structures, after `_construct_arm_planning` (which calls
    `srp._rebuild_candidate_total_capacities`)."""
    years = list(sed.years)
    return {
        'years': years,
        'candidate_investment': {str(y): dict(candidate['investment'][node][y]) for y in years},
        'candidate_total_capacity': {str(y): dict(candidate['total_capacity'][node][y]) for y in years},
    }


def _applied_tables(planning, sed, node):
    """After production's own `srp._restore_candidate_data`: the shared-ESS cohort
    objects and the TSO network data (per-unit -> MVA / MVAh via baseMVA)."""
    years = list(sed.years)
    idx = sed.get_shared_energy_storage_idx(node)
    cohorts = {str(y): {'s': sed.shared_energy_storages[y][idx].s,
                        'e': sed.shared_energy_storages[y][idx].e} for y in years}
    tso = planning.transmission_network
    tso_rows = {}
    for y in years:
        day = list(tso.network[y])[0]
        net = tso.network[y][day]
        ess = net.shared_energy_storages[net.get_shared_energy_storage_idx(node)]
        tso_rows[str(y)] = {'day': str(day), 'baseMVA': net.baseMVA,
                            's_mva': ess.s * net.baseMVA, 'e_mwh': ess.e * net.baseMVA}
    return {'shared_ess_cohort_investment': cohorts, 'tso_network_total_capacity': tso_rows}


def _esso_cohort_state(sed, candidate, node):
    """Production's own ESSO subproblem build + candidate load: which cohort is
    active. Zero solves (the guard proves it)."""
    import shared_energy_storage_data as SED
    model = SED._build_subproblem(sed, node)
    SED._update_model_with_candidate_solution(sed, {node: model}, candidate['investment'])
    years = list(sed.years)
    import pyomo.environ as pe
    return {str(years[y_inv]): {
        'cohort_index': y_inv,
        'inactive': bool(model._esso_cohort_inactive[y_inv]),
        'es_s_investment_fixed': pe.value(model.es_s_investment_fixed[y_inv]),
        'es_e_investment_fixed': pe.value(model.es_e_investment_fixed[y_inv]),
    } for y_inv in model.years}


def _build(name, investment_year, G, investment_map=None):
    report = {}
    eval_id = _eval_id(name)
    if os.path.exists(os.path.join(G.O.WORK_DIR, eval_id)):
        raise RuntimeError(f'working dir id already used (never reusable): {eval_id}')
    kwargs = {} if investment_year is None else {'investment_year': investment_year}
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(OUT_ROOT, f'scratch_{RUN_TAG}', name), report,
        investment_map=investment_map if investment_map is not None else _probe_map(),
        eval_id=eval_id, num_max_iters_override=CAP, apply_rho=False, **kwargs)
    return planning, sed, candidate, report


def check_construction():
    import shared_resources_planning as srp
    import p515_g_g1_g4_admm_gates as G
    checks, detail = {}, {}

    instance_years = H.instance_investment_years()
    detail['instance_investment_years'] = instance_years
    detail['instance_investment_years_source'] = ('p56a_oracle.load_baseline()["planning"].'
                                                  'shared_ess_data.years (NOT a literal)')
    checks['years_include_2030_and_2035'] = 2030 in instance_years and 2035 in instance_years
    checks['bad_year_not_in_instance_years'] = BAD_YEAR not in instance_years

    # ---- (iii) 2030 ----
    p30, sed30, cand30, rep30 = _build('y2030', PROBE_YEAR, G)
    tables30 = _capacity_tables(p30, sed30, cand30, PROBE_NODE)
    inv30 = tables30['candidate_investment']
    tot30 = tables30['candidate_total_capacity']
    checks['iii_investment_at_2030'] = (inv30['2030']['s'] == PROBE_S_MVA and inv30['2030']['e'] == PROBE_E_MWH)
    checks['iii_no_investment_at_2025'] = (inv30['2025']['s'] == 0.0 and inv30['2025']['e'] == 0.0)
    checks['iii_no_investment_at_2035'] = (inv30['2035']['s'] == 0.0 and inv30['2035']['e'] == 0.0)
    checks['iii_total_capacity_zero_in_2025'] = (tot30['2025']['s'] == 0.0 and tot30['2025']['e'] == 0.0)
    checks['iii_total_capacity_invested_in_2030'] = (tot30['2030']['s'] == PROBE_S_MVA
                                                     and tot30['2030']['e'] == PROBE_E_MWH)
    checks['iii_total_capacity_invested_in_2035'] = (tot30['2035']['s'] == PROBE_S_MVA
                                                     and tot30['2035']['e'] == PROBE_E_MWH)
    checks['iii_report_instance_year_is_2030'] = rep30['instance']['year'] == PROBE_YEAR
    detail['iii_2030'] = {'investment_map': _probe_map(), 'report_instance': rep30['instance'], **tables30}

    srp._restore_candidate_data(p30, cand30)
    applied30 = _applied_tables(p30, sed30, PROBE_NODE)
    detail['iii_2030_after_restore_candidate_data'] = applied30
    checks['iii_shared_ess_cohort_investment_at_2030_only'] = (
        applied30['shared_ess_cohort_investment']['2025'] == {'s': 0.0, 'e': 0.0}
        and applied30['shared_ess_cohort_investment']['2030'] == {'s': PROBE_S_MVA, 'e': PROBE_E_MWH}
        and applied30['shared_ess_cohort_investment']['2035'] == {'s': 0.0, 'e': 0.0})
    checks['iii_tso_total_capacity_zero_2025_invested_2030_2035'] = (
        abs(applied30['tso_network_total_capacity']['2025']['s_mva']) < 1e-12
        and abs(applied30['tso_network_total_capacity']['2030']['s_mva'] - PROBE_S_MVA) < 1e-9
        and abs(applied30['tso_network_total_capacity']['2035']['s_mva'] - PROBE_S_MVA) < 1e-9
        and abs(applied30['tso_network_total_capacity']['2030']['e_mwh'] - PROBE_E_MWH) < 1e-9)

    cohort30 = _esso_cohort_state(sed30, cand30, PROBE_NODE)
    detail['iii_2030_esso_cohort_state'] = cohort30
    checks['iii_esso_cohort_2025_inactive_2030_active'] = (
        cohort30['2025']['inactive'] is True and cohort30['2030']['inactive'] is False
        and cohort30['2035']['inactive'] is True
        and cohort30['2030']['es_e_investment_fixed'] == PROBE_E_MWH)

    # ---- (vi) the configuration hook on the AA-on case file, 2030 object ----
    tmp_root = tempfile.mkdtemp(prefix='p515s45_w14_hook_spec_')
    try:
        _p, _s, hook_spec = H.freeze_campaign_spec(
            campaign_root=os.path.join(tmp_root, 'campaign'), campaign_id='w14_hook_2030',
            candidates=[('probe_2030', _probe_map(), {'investment_year': PROBE_YEAR})],
            configuration={'name': 'D (case file, AA on) -- fixture only, never run',
                           'case_file_anderson_acceleration': EXPECTED_AA},
            cap=CAP, concurrency=1, authority=AUTHORITY, required_consecutive_cycles=REQUIRED_CYCLES)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    holder, hook_report = {}, {}
    H._config_hook_factory(hook_spec, holder, overrides=hook_spec['candidates'][0]['overrides'])(
        planning=p30, sed=sed30, candidate=cand30, report=hook_report)
    checks['vi_hook_passes_on_2030_candidate_with_aa_on_case_file'] = (
        bool(holder['configuration_checks']) and all(holder['configuration_checks'].values()))
    checks['vi_hook_effective_aa_is_the_case_file_dict'] = (
        holder.get('anderson_acceleration_effective') == EXPECTED_AA)
    checks['vi_hook_entry_year_is_2030'] = hook_spec['candidates'][0]['canonical']['investment_year'] == PROBE_YEAR
    detail['vi_hook'] = {'configuration_checks': holder['configuration_checks'],
                         'overrides_applied': holder['overrides_applied'],
                         'anderson_acceleration_effective': holder.get('anderson_acceleration_effective'),
                         'entry_canonical': hook_spec['candidates'][0]['canonical'],
                         'case_file_sha256': H.sha256_file(H.CASE_FILE)}

    # ---- (iv) 2025, explicit, versus 2025 by default, in the same process ----
    _p25, sed25, cand25, rep25 = _build('y2025', CONTROL_YEAR, G)
    _pd, sedd, candd, repd = _build('y2025_default', None, G)
    tables25 = _capacity_tables(_p25, sed25, cand25, PROBE_NODE)
    tablesd = _capacity_tables(_pd, sedd, candd, PROBE_NODE)
    checks['iv_explicit_2025_equals_default_report_instance'] = rep25['instance'] == repd['instance']
    checks['iv_explicit_2025_equals_default_candidate'] = cand25 == candd
    checks['iv_default_report_instance_year_is_2025'] = repd['instance']['year'] == CONTROL_YEAR
    checks['iv_2025_investment_at_2025'] = (tables25['candidate_investment']['2025']['s'] == PROBE_S_MVA
                                            and tables25['candidate_investment']['2025']['e'] == PROBE_E_MWH)
    checks['iv_2025_total_capacity_invested_in_all_three_years'] = all(
        tables25['candidate_total_capacity'][str(y)] == {'s': PROBE_S_MVA, 'e': PROBE_E_MWH}
        for y in (2025, 2030, 2035))
    checks['iv_2030_and_2025_candidates_differ'] = cand30 != cand25
    detail['iv_2025_explicit'] = {'report_instance': rep25['instance'], **tables25}
    detail['iv_2025_default'] = {'report_instance': repd['instance'], **tablesd}

    # ---- (v) a year outside the instance's years raises, in the constructor ----
    res = _expect_raise(lambda: _build('ybad', BAD_YEAR, G), ValueError)
    detail['v_construct_arm_planning_bad_year'] = res
    checks['v_construct_arm_planning_raises_on_bad_year'] = (
        res['raised'] and str(BAD_YEAR) in res['message'] and 'instance investment years' in res['message'])
    return checks, detail


# ======================================================================================================================
#  (v) -- the CHILD path refuses a year outside the instance's investment years
# ======================================================================================================================
def check_child_refusal():
    import inspect
    checks, detail = {}, {}
    fake_entry = {
        'label': 'w14_bad_year', 'overrides': {},
        'canonical': H.canonical_candidate(_probe_map(), investment_year=BAD_YEAR),
        'working_dir_ids': {'run': _eval_id('w14_badyear_run'), 'precheck': _eval_id('w14_badyear_precheck')},
    }
    fake_entry['key'] = H.candidate_key(fake_entry['canonical'])
    fake_spec = {'campaign_id': 'w14_child_refusal', 'cap': CAP,
                 'required_consecutive_cycles': REQUIRED_CYCLES,
                 'configuration': {'name': 'D', 'arm_label': 's39_D', 'overrides': {},
                                   'case_file_anderson_acceleration': EXPECTED_AA}}
    tmp_dir = tempfile.mkdtemp(prefix='p515s45_w14_child_')
    try:
        res = _expect_raise(lambda: H._child_real(
            args=types.SimpleNamespace(spec_sha256='0' * 64), spec=fake_spec, spec_path=tmp_dir,
            entry=fake_entry, eval_dir=tmp_dir, lock_content={}, env_caps={}, started=time.time()),
            RuntimeError)
        leftovers = sorted(os.listdir(tmp_dir))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    detail['v_child_real_bad_year'] = res
    detail['v_child_real_eval_dir_after'] = leftovers
    checks['v_child_refuses_year_outside_instance_years'] = (
        res['raised'] and str(BAD_YEAR) in res['message']
        and 'instance investment years' in res['message'])
    checks['v_child_wrote_nothing_before_refusing'] = leftovers == []

    src = '\n'.join(line for line in inspect.getsource(H._child_real).splitlines()
                     if not line.strip().startswith('#'))   # comments are prose, not wiring
    needles = {
        'reads_the_year_from_the_canonical': 'investment_year_from_canonical(entry[\'canonical\'])' in src,
        'validates_against_instance_years': 'instance_investment_years()' in src,
        'no_2025_literal_comparison': 'G.N.INVEST_YEAR' not in src,
        'forwards_to_run_admm_arm': 'investment_year=investment_year' in src,
    }
    detail['v_child_source_needles'] = needles
    checks['v_child_wiring_as_designed'] = all(needles.values())
    return checks, detail


# ======================================================================================================================
def _manifest():
    files = {}
    for root, _dirs, names in os.walk(OUT_ROOT):
        for name in sorted(names):
            if name == MANIFEST_NAME:
                continue
            path = os.path.join(root, name)
            files[os.path.relpath(path, REPO)] = {'sha256': H.sha256_file(path),
                                                  'bytes': os.path.getsize(path)}
    manifest = {'stage': STAGE, 'generated_utc': _utc(),
                'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
                'files': dict(sorted(files.items()))}
    path = os.path.join(OUT_ROOT, MANIFEST_NAME)
    if os.path.exists(path):
        raise SystemExit(f'refusing to overwrite the manifest: {path}')
    with open(path, 'w') as handle:
        json.dump(manifest, handle, indent=1)
    print(f'[W14] manifest: {len(files)} files -> {os.path.relpath(path, REPO)}', flush=True)


def main():
    started = _utc()
    os.makedirs(OUT_ROOT, exist_ok=True)
    results_path = os.path.join(OUT_ROOT, _results_name())
    if os.path.exists(results_path):
        raise SystemExit(f'refusing to overwrite (write-once): {results_path}')
    tracked = ('p515_s44_campaign_harness.py', 'p515_g_g1_g4_admm_gates.py',
               os.path.basename(__file__), 'shared_resources_planning.py',
               'shared_energy_storage_data.py', 'p56a_oracle.py',
               os.path.join('data', 'SRP1', 'SRP1_params.json'))
    results = {
        'stage': STAGE, 'authority': AUTHORITY, 'started_utc': started,
        'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
        'git_status_porcelain_relevant': _git(['status', '--porcelain', '--', *tracked]).splitlines(),
        'file_sha256': {rel: H.sha256_file(os.path.join(REPO, rel)) for rel in tracked},
        'instance_note': (f'probe candidate: node {PROBE_NODE} at {PROBE_S_MVA} MVA / {PROBE_E_MWH} MWh, '
                          f'other active nodes zero; an A1 2025-ladder point (n7_2h_e3) placed at '
                          f'{PROBE_YEAR} and at {CONTROL_YEAR}'),
        'probe_candidate_key': H.candidate_key(H.canonical_candidate(_probe_map(),
                                                                     investment_year=PROBE_YEAR)),
        'probe_candidate_key_2025': H.candidate_key(H.canonical_candidate(_probe_map())),
    }
    checks, sections, errors = {}, {}, {}
    for name, fn in (('i_committed_identity', check_committed_identity),
                     ('ii_year_fixture', check_year_fixture),
                     ('iii_iv_vi_construction', check_construction),
                     ('v_child_refusal', check_child_refusal)):
        try:
            out = fn()
            checks.update(out[0])
            sections[name] = out[1]
        except Exception:  # noqa: BLE001 -- recorded; the gate then fails
            errors[name] = traceback.format_exc()
            print(errors[name], file=sys.stderr, flush=True)
    guard_failures = GUARD.verify(0)
    results.update({
        'checks': checks, 'failed_checks': sorted(k for k, v in checks.items() if not v),
        'errors': errors, 'sections': sections,
        'solve_profile_guard': {'permitted': [], 'counts': dict(GUARD.counts),
                                'verify_0_failures': guard_failures},
        'ended_utc': _utc(),
    })
    results['all_pass'] = (bool(checks) and not results['failed_checks'] and not errors
                           and not guard_failures)
    H._write_once_json(results_path, results)
    GUARD.uninstall()
    print(f"[W14] checks={len(checks)} failed={results['failed_checks']} errors={sorted(errors)} "
          f"guard={dict(GUARD.counts)} verify0_failures={guard_failures} "
          f"all_pass={results['all_pass']}", flush=True)
    if not results['all_pass']:
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--manifest':
        GUARD.uninstall()
        _manifest()
        sys.exit(0)
    if len(sys.argv) > 1:
        RUN_TAG = sys.argv[1]          # noqa: F811 -- the run tag, see RUN_TAG above
    main()
