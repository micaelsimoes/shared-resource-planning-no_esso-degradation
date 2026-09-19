"""
P5.15 Addendum 25 item 1 -- zero-solve checks for the campaign harness
(`p515_s44_campaign_harness.py`), run and committed BEFORE the gate.

Armed `SolveProfileGuard(permitted=())` in this (parent) process for the
whole script, verified at exactly 0. The stub children import no model code
and never solve (they run the harness's `--stub-mode`, refused by the child
unless the frozen spec is a test-only stub spec).

Checks (all write under the write-once root
`data/SRP1/Results/P515S44/harness_checks/`; every lock used here is a
TEMPORARY lock path inside that root -- the real `.p515_s44_campaign.lock`
and `.p515_g_gate.lock` are never touched):
  C1 canonical candidate form and key (stability, zero handling, validation).
  C2 frozen campaign spec: file name carries its own sha256 prefix, load by
     hash, write-once refusal, unsupported override / duplicate refused.
  C3 campaign lock: exclusive acquire, legacy-lock refusal, release by the
     wrong pid refused.
  C4 process spawn with a stub evaluation: 4 candidates, concurrency 2 --
     fresh interpreters, thread caps in the child env, the child verifies the
     lock names its parent, slot limit respected with real overlap, per-
     evaluation launch.json / both streams / exit_code.txt / wait4 rusage,
     the spec sha256 in every record, write-once re-evaluation refusal; a
     failing child -> parent-synthesized barrier record with the stderr tail;
     a child under a lock naming another pid refuses; a stub request under a
     non-test spec refuses.
  C5 the STEP4 2.5 record schema, built by `build_evaluation_record` from the
     D reference's COMMITTED artifacts (read only), cross-checked against the
     committed `s39_evaluation.json`; plus a truncated, uncertified trajectory
     (barrier path).
  C6 rule eleven: `assert_record_capture_paths()` passes.
  C7 the child's configuration hook on a real, unsolved planning object built
     by `_construct_arm_planning` with the C* map and cap 500: every D-oracle
     check passes, no override applied for D; the AA override path applies
     `anderson_acceleration.enabled` on a separate object.

Launch (attached, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s44_campaign_harness_checks.py \\
        > data/SRP1/Results/P515S44/harness_checks_launch.log 2>&1
"""

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15-S44 harness zero-solve checks').install()

import p515_s44_campaign_harness as H  # noqa: E402

_SUFFIX = next(('_' + a.strip('_') for a in sys.argv[1:] if not a.startswith('--')), '')
OUT = os.path.join(H.RESULTS_ROOT, f'harness_checks{_SUFFIX}')  # optional argv suffix: a re-run never overwrites
D_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39_D_run')
C_STAR = {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)}
PAPER = {5: (0.0, 0.0), 7: (1.62, 3.24), 9: (0.0, 0.0)}
N7E = {5: (0.96875, 3.875), 7: (0.0, 0.0), 9: (0.96875, 3.875)}
EVAL_ID_PREFIX = f'p515s44_hchk{_SUFFIX}'


def _expect_raise(fn, exc_types):
    try:
        fn()
    except exc_types as error:  # noqa: PERF203
        return {'raised': True, 'type': type(error).__name__, 'message': str(error)[:300]}
    except BaseException as error:  # noqa: BLE001
        return {'raised': True, 'type': type(error).__name__, 'message': str(error)[:300],
                'unexpected_type': True}
    return {'raised': False}


def c1_canonical():
    k1 = H.candidate_key(H.canonical_candidate(C_STAR))
    k2 = H.candidate_key(H.canonical_candidate({'9': ('0.96875', 3.875), '5': (0.96875, '3.875'),
                                                7: (0.96875, 3.875)}))
    z1 = H.candidate_key(H.canonical_candidate({5: (0, 0), 7: (1.62, 3.24), 9: (0.0, -0.0)}))
    z2 = H.candidate_key(H.canonical_candidate(PAPER))
    keys = {lab: H.candidate_key(H.canonical_candidate(c)) for lab, c in
            (('c_star', C_STAR), ('paper_plan', PAPER), ('node7_empty', N7E))}
    errors = {
        's_zero_e_positive': _expect_raise(lambda: H.canonical_candidate({5: (0.0, 1.0), 7: (1, 2), 9: (1, 2)}), ValueError),
        'missing_node': _expect_raise(lambda: H.canonical_candidate({5: (1, 2), 7: (1, 2)}), ValueError),
        'extra_node': _expect_raise(lambda: H.canonical_candidate({5: (1, 2), 7: (1, 2), 9: (1, 2), 11: (1, 2)}), ValueError),
        'negative': _expect_raise(lambda: H.canonical_candidate({5: (-1, 2), 7: (1, 2), 9: (1, 2)}), ValueError),
        'nan': _expect_raise(lambda: H.canonical_candidate({5: (float('nan'), 2), 7: (1, 2), 9: (1, 2)}), ValueError),
    }
    checks = {
        'key_independent_of_order_and_numeric_spelling': k1 == k2,
        'zero_spellings_same_key': z1 == z2,
        'gate_candidates_distinct_keys': len(set(keys.values())) == 3,
        'all_invalid_candidates_rejected': all(v['raised'] and not v.get('unexpected_type') for v in errors.values()),
        'canonical_c_star': H.canonical_candidate(C_STAR) == {
            'investment_year': 2025, 'nodes': {'5': [0.96875, 3.875], '7': [0.96875, 3.875], '9': [0.96875, 3.875]}},
    }
    return {'checks': checks, 'keys': keys, 'error_cases': errors,
            'eval_ids_c_star_s44_gate': H.eval_ids('s44_gate', keys['c_star'])}


def _stub_spec(root, campaign_id, candidates, concurrency):
    return H.freeze_campaign_spec(
        root, campaign_id, candidates,
        configuration={'name': 'STUB (checks only)', 'arm_label': 's39_D', 'overrides': {}},
        cap=500, concurrency=concurrency, authority=['checks only'],
        extra={'test_only_stub': True})


def c2_spec():
    root = os.path.join(OUT, 'c2_spec_root')
    path, sha, spec = _stub_spec(root, 'c2', [('a', C_STAR), ('b', PAPER)], 2)
    loaded_path, loaded = H.load_frozen_spec(root, sha)
    checks = {
        'file_sha256_equals_returned': H.sha256_file(path) == sha,
        'file_name_carries_sha8': os.path.basename(path) == f'campaign_spec_c2_{sha[:8]}.json',
        'load_by_hash_finds_same_file': loaded_path == path and loaded == spec,
        'spec_records_case_file_sha256': spec['configuration']['case_file_sha256'] == H.sha256_file(H.CASE_FILE),
        'spec_records_harness_sha256': spec['harness']['sha256'] == H.sha256_file(H.HARNESS_PATH),
        'spec_records_thread_caps': spec['thread_caps'] == H.THREAD_CAP_ENV,
    }
    refusals = {
        'second_freeze_into_nonempty_root': _expect_raise(lambda: _stub_spec(root, 'c2', [('a', C_STAR)], 1), RuntimeError),
        'unsupported_override': _expect_raise(lambda: H.freeze_campaign_spec(
            os.path.join(OUT, 'c2_bad_override'), 'x', [('a', C_STAR)],
            configuration={'name': 'x', 'overrides': {'rho': 1.0}}, cap=1, concurrency=1, authority=[]), ValueError),
        'duplicate_candidate': _expect_raise(lambda: H.freeze_campaign_spec(
            os.path.join(OUT, 'c2_dup'), 'x', [('a', C_STAR), ('b', dict(C_STAR))],
            configuration={'name': 'x', 'overrides': {}}, cap=1, concurrency=1, authority=[]), ValueError),
    }
    checks['all_refusals_raised'] = all(v['raised'] and not v.get('unexpected_type') for v in refusals.values())
    return {'checks': checks, 'spec_path': os.path.relpath(path, REPO), 'spec_sha256': sha, 'refusals': refusals}


def c3_lock():
    lock = os.path.join(OUT, 'c3_campaign.lock')
    legacy = os.path.join(OUT, 'c3_legacy.lock')
    content = H.acquire_campaign_lock('c3', 'abc', lock_path=lock, legacy_lock_path=legacy)
    second = _expect_raise(lambda: H.acquire_campaign_lock('c3', 'abc', lock_path=lock, legacy_lock_path=legacy),
                           SystemExit)
    wrong_pid = _expect_raise(lambda: H.release_campaign_lock(lock_path=lock, expected_pid=os.getpid() + 999999),
                              RuntimeError)
    released = H.release_campaign_lock(lock_path=lock, expected_pid=os.getpid())
    with open(legacy, 'w') as handle:
        handle.write('legacy holder')
    legacy_refusal = _expect_raise(lambda: H.acquire_campaign_lock('c3', 'abc', lock_path=lock,
                                                                   legacy_lock_path=legacy), SystemExit)
    os.remove(legacy)
    checks = {
        'acquire_writes_pid_and_spec': content['pid'] == os.getpid() and content['campaign_spec_sha256'] == 'abc',
        'second_acquire_refused': second['raised'] and second['type'] == 'SystemExit',
        'release_by_wrong_pid_refused': wrong_pid['raised'] and wrong_pid['type'] == 'RuntimeError',
        'release_ok_and_removed': released and not os.path.exists(lock),
        'legacy_lock_blocks_campaign': legacy_refusal['raised'] and legacy_refusal['type'] == 'SystemExit',
        'no_leftover_lock_files': not os.path.exists(lock) and not os.path.exists(legacy),
    }
    return {'checks': checks, 'second': second, 'wrong_pid': wrong_pid, 'legacy_refusal': legacy_refusal}


def _ctx(root, spec_path, sha, spec, lock, extra):
    return H.CampaignContext(root, spec_path, sha, spec, log=lambda m: print(m, flush=True),
                             child_extra_args=['--lock-path', lock] + list(extra))


def c4_spawn():
    cands = [('s1', C_STAR), ('s2', PAPER), ('s3', N7E),
             ('s4', {5: (0.25, 0.5), 7: (0.25, 0.5), 9: (0.25, 0.5)})]
    root = os.path.join(OUT, 'c4_stub_campaign')
    lock = os.path.join(OUT, 'c4_campaign.lock')
    legacy = os.path.join(OUT, 'c4_legacy_absent.lock')
    path, sha, spec = _stub_spec(root, 'c4', cands, 2)
    H.acquire_campaign_lock('c4', sha, lock_path=lock, legacy_lock_path=legacy)
    try:
        ctx = _ctx(root, path, sha, spec, lock, ['--stub-mode', 'ok', '--stub-sleep-s', '4', '--stub-alloc-mb', '96'])
        t0 = time.time()
        records = H.evaluate([c for _l, c in cands], ctx)
        wall = time.time() - t0
        info = dict(getattr(H.evaluate, 'last_batch_info', {}))
        rerun = _expect_raise(lambda: H.evaluate([C_STAR], ctx), RuntimeError)
    finally:
        H.release_campaign_lock(lock_path=lock, expected_pid=os.getpid())
    # concurrency profile from the parent's own start/end timeline
    events = sorted(info.get('timeline', []), key=lambda e: (e['t'], 0 if e['event'] == 'end' else 1))
    live, peak = 0, 0
    for e in events:
        live += 1 if e['event'] == 'start' else -1
        peak = max(peak, live)
    per_eval = []
    for entry, rec in zip(spec['candidates'], records):
        d = os.path.join(root, 'evals', entry['eval_dir'])
        with open(os.path.join(d, 'exit_code.txt')) as handle:
            code = handle.read().strip()
        with open(os.path.join(d, 'wait4_rusage.json')) as handle:
            ru = json.load(handle)
        per_eval.append({
            'label': entry['label'], 'dir_files': sorted(os.listdir(d)), 'exit_code': code,
            'record_status': rec.get('status'), 'record_spec_sha256_ok': rec.get('campaign_spec_sha256') == sha,
            'child_pid': rec.get('child_pid'), 'child_ppid': rec.get('child_ppid'),
            'thread_caps_seen': rec.get('thread_caps_seen'), 'wait4_ru_maxrss': ru['rusage']['ru_maxrss'],
            'child_self_ru_maxrss': (rec.get('peak_rss') or {}).get('child_self_ru_maxrss'),
            'lock_pid_seen': (rec.get('lock_content') or {}).get('pid'),
            'PYTHONHASHSEED_in_child': rec.get('PYTHONHASHSEED'),
        })
    required_files = {'launch.json', 'child_stdout.log', 'child_stderr.log', 'exit_code.txt',
                      'wait4_rusage.json', 'evaluation_record.json'}
    checks = {
        'four_records_in_batch_order': [r.get('candidate_label') for r in records] == ['s1', 's2', 's3', 's4'],
        'all_stub_ok_exit_0': all(p['exit_code'] == '0' and p['record_status'] == 'stub' for p in per_eval),
        'spec_sha256_in_every_record': all(p['record_spec_sha256_ok'] for p in per_eval),
        'distinct_child_processes': len({p['child_pid'] for p in per_eval}) == 4,
        'children_are_direct_children_of_parent': all(p['child_ppid'] == os.getpid() for p in per_eval),
        'child_saw_lock_naming_parent': all(p['lock_pid_seen'] == os.getpid() for p in per_eval),
        'thread_caps_in_every_child': all(p['thread_caps_seen'] == H.THREAD_CAP_ENV for p in per_eval),
        'slot_limit_respected_and_used': info.get('max_concurrent_observed') == 2 and peak == 2,
        'wall_consistent_with_2_slots': wall < 4 * 4.0,  # 4 x 4 s sequential would be >= 16 s
        'per_eval_files_present': all(required_files <= set(p['dir_files']) for p in per_eval),
        'wait4_rusage_captures_child_memory': all(p['wait4_ru_maxrss'] >= 96 * (1 << 20) for p in per_eval),
        'reevaluation_in_same_campaign_refused': rerun['raised'] and rerun['type'] == 'RuntimeError',
    }

    # failing child -> parent-synthesized barrier record
    root_f = os.path.join(OUT, 'c4_fail_campaign')
    lock_f = os.path.join(OUT, 'c4f_campaign.lock')
    path_f, sha_f, spec_f = _stub_spec(root_f, 'c4f', [('fail1', C_STAR)], 1)
    H.acquire_campaign_lock('c4f', sha_f, lock_path=lock_f, legacy_lock_path=legacy)
    try:
        rec_f = H.evaluate([C_STAR], _ctx(root_f, path_f, sha_f, spec_f, lock_f, ['--stub-mode', 'fail']))[0]
    finally:
        H.release_campaign_lock(lock_path=lock_f, expected_pid=os.getpid())
    checks['failing_child_gives_parent_barrier_record'] = (
        rec_f.get('status') == 'harness_error' and rec_f.get('barrier') is True
        and rec_f.get('parent_view', {}).get('exit_code') == 3
        and any('failing on purpose' in line for line in rec_f.get('stderr_tail', [])))

    # lock naming another pid -> child refuses
    root_l = os.path.join(OUT, 'c4_foreign_lock_campaign')
    lock_l = os.path.join(OUT, 'c4l_campaign.lock')
    path_l, sha_l, spec_l = _stub_spec(root_l, 'c4l', [('foreign', C_STAR)], 1)
    with open(lock_l, 'w') as handle:
        json.dump({'pid': 1, 'campaign_id': 'c4l', 'campaign_spec_sha256': sha_l}, handle)
    try:
        rec_l = H.evaluate([C_STAR], _ctx(root_l, path_l, sha_l, spec_l, lock_l, ['--stub-mode', 'ok']))[0]
    finally:
        os.remove(lock_l)
    checks['child_refuses_foreign_lock'] = (
        rec_l.get('barrier') is True and rec_l.get('parent_view', {}).get('exit_code') != 0
        and any('CHILD REFUSES: campaign lock pid' in line for line in rec_l.get('stderr_tail', [])))

    # stub requested under a NON-test spec -> child refuses
    root_n = os.path.join(OUT, 'c4_nonstub_spec_campaign')
    lock_n = os.path.join(OUT, 'c4n_campaign.lock')
    path_n, sha_n, spec_n = H.freeze_campaign_spec(
        root_n, 'c4n', [('nonstub', C_STAR)], configuration={'name': 'real-looking', 'overrides': {}},
        cap=500, concurrency=1, authority=['checks only'])
    H.acquire_campaign_lock('c4n', sha_n, lock_path=lock_n, legacy_lock_path=legacy)
    try:
        rec_n = H.evaluate([C_STAR], _ctx(root_n, path_n, sha_n, spec_n, lock_n, ['--stub-mode', 'ok']))[0]
    finally:
        H.release_campaign_lock(lock_path=lock_n, expected_pid=os.getpid())
    checks['stub_refused_under_non_test_spec'] = (
        rec_n.get('barrier') is True
        and any('stub mode requested but the frozen spec is not a test-only stub spec' in line
                for line in rec_n.get('stderr_tail', [])))
    return {'checks': checks, 'per_eval': per_eval, 'batch_info': info, 'wall_s': wall,
            'peak_concurrency_from_timeline': peak, 'rerun_refusal': rerun,
            'fail_record': rec_f, 'foreign_lock_record': rec_l, 'nonstub_record': rec_n}


def c5_record_schema():
    import p515_s40_clone_capture_preflight as CP
    report = CP._load_json(os.path.join(D_DIR, 'g_s39_D.json'))
    cl = CP._load_json(os.path.join(D_DIR, 'component_levels_terminal.json'))
    bt = CP._load_json(os.path.join(D_DIR, 'boyd_terminal.json'))
    s39_eval = CP._load_json(os.path.join(D_DIR, 's39_evaluation.json'))
    canon = H.canonical_candidate(C_STAR)
    key = H.candidate_key(canon)
    spec = {'campaign_id': 'c5_schema', 'cap': 500, 'required_consecutive_cycles': 10,
            'configuration': {'name': 'D', 'arm_label': 's39_D'}}
    entry = {'label': 'c_star', 'canonical': canon, 'key': key,
             'working_dir_ids': H.eval_ids('c5_schema', key)}
    rec = H.build_evaluation_record(
        spec=spec, spec_path=os.path.join(OUT, 'no_spec.json'), spec_sha256='0' * 64, entry=entry,
        report=report, component_levels=cl,
        floor_terminal=bt.get('soh_floor_multiplier_and_efc_per_cohort_year_terminal'),
        published_caps=None, peak_rss={'note': 'schema check'}, wall={'note': 'schema check'},
        eval_dir=D_DIR)
    rows = report['cycle_trajectory']
    expected_bar = max(r['objective_change_abs'] for r in rows[-10:])
    required_keys = ('schema', 'campaign_spec_sha256', 'candidate_key', 'candidate_canonical', 'status',
                     'barrier', 'barrier_cause', 'objective_convention', 'certified_cost', 'bar',
                     'certification_cycle', 'first_pass_cycle_per_channel', 'terminal_ratios_per_channel',
                     'rule_ten', 'component_decomposition_totals_weighted', 'recourse_components',
                     'settlement_remainder', 'storage_per_node', 'wall_time_s', 'peak_rss')
    lead = s39_eval.get('LEAD', {})
    checks = {
        'all_2_5_keys_present': all(k in rec for k in required_keys),
        'certified': rec['status'] == 'certified' and rec['barrier'] is False,
        'certification_cycle_139': rec['certification_cycle'] == 139,
        'certified_cost_equals_D': rec['certified_cost'] == 650966975.2943751,
        'bar_is_max_step_last_10': rec['bar']['value'] == expected_bar and rec['bar']['n_cycles_in_window'] == 10,
        'pf_first_pass_matches_committed_s39_evaluation': (
            rec['first_pass_cycle_per_channel']['pf'] == lead.get('pf_first_pass_cycle')),
        'ess_terminal_ratio_matches_committed_s39_evaluation': (
            rec['terminal_ratios_per_channel']['ess']['max'] == lead.get('storage_terminal_ratio')),
        'settlement_remainder_equals_component_levels': (
            rec['settlement_remainder']['value'] == cl['recourse_components']['interface_settlement_total']),
        'storage_per_node_three_nodes_with_efc': (
            sorted(rec['storage_per_node']) == ['5', '7', '9']
            and all(v['efc_per_day_max'] is not None and v['terminal_soh_min_over_active_cohort_years'] is not None
                    for v in rec['storage_per_node'].values())),
        'record_json_serializable': bool(json.dumps(rec, default=str)),
    }
    # barrier path: D's first 50 rows under a cap of 50 (never certified by cycle 50)
    report_trunc = dict(report)
    report_trunc['cycle_trajectory'] = rows[:50]
    spec_trunc = dict(spec)
    spec_trunc['cap'] = 50
    rec_b = H.build_evaluation_record(
        spec=spec_trunc, spec_path=os.path.join(OUT, 'no_spec.json'), spec_sha256='0' * 64, entry=entry,
        report=report_trunc, component_levels=cl, floor_terminal=None, published_caps=None,
        peak_rss={}, wall={}, eval_dir=D_DIR)
    checks['truncated_uncertified_is_barrier'] = (
        rec_b['status'] == 'not_certified' and rec_b['barrier'] is True and rec_b['certified_cost'] is None
        and rec_b['barrier_cause'] == 'cap reached without certification')
    # zero-node storage block (schema only, D's capture reused for positive nodes)
    zero_block = H._storage_per_node(H.canonical_candidate(PAPER), report.get('esso_capture'), None, None)
    checks['zero_node_marked_no_storage'] = (zero_block['5']['has_storage'] is False
                                             and zero_block['9']['has_storage'] is False
                                             and zero_block['7']['has_storage'] is True)
    out_path = os.path.join(OUT, 'c5_record_from_D_reference.json')
    H._write_once_json(out_path, rec)
    return {'checks': checks, 'record_path': os.path.relpath(out_path, REPO),
            'record_summary': {k: rec[k] for k in ('status', 'certification_cycle', 'certified_cost',
                                                   'first_pass_cycle_per_channel', 'settlement_remainder')},
            'bar': rec['bar']['value']}


def c6_capture_paths():
    checks = H.assert_record_capture_paths()
    return {'checks': {'assert_record_capture_paths_passes': all(checks.values())}, 'detail': checks}


def c7_config_hook():
    import p515_g_g1_g4_admm_gates as G
    for suffix in ('d', 'aa'):
        if os.path.exists(os.path.join(G.O.WORK_DIR, f'{EVAL_ID_PREFIX}_{suffix}')):
            raise RuntimeError(f'eval dir already exists: {EVAL_ID_PREFIX}_{suffix}')
    spec_d = {'cap': 500, 'required_consecutive_cycles': 10, 'configuration': {'overrides': {}}}
    report, holder = {}, {}
    planning, sed, cand = G._construct_arm_planning(
        's39_D', os.path.join(OUT, 'c7_scratch_d'), report, investment_map=H.investment_map_from_canonical(
            H.canonical_candidate(C_STAR)), eval_id=f'{EVAL_ID_PREFIX}_d', num_max_iters_override=500,
        apply_rho=False)
    H._config_hook_factory(spec_d, holder)(planning=planning, sed=sed, candidate=cand, report=report)
    d_ok = all(holder['configuration_checks'].values()) and holder['overrides_applied'] == {}
    aa_enabled_after_d = planning.params.admm.anderson_acceleration.get('enabled')
    spec_aa = {'cap': 500, 'required_consecutive_cycles': 10,
               'configuration': {'overrides': {'anderson_acceleration': {'enabled': True}}}}
    report2, holder2 = {}, {}
    planning2, sed2, cand2 = G._construct_arm_planning(
        's39_D', os.path.join(OUT, 'c7_scratch_aa'), report2, investment_map=H.investment_map_from_canonical(
            H.canonical_candidate(C_STAR)), eval_id=f'{EVAL_ID_PREFIX}_aa', num_max_iters_override=500,
        apply_rho=False)
    H._config_hook_factory(spec_aa, holder2)(planning=planning2, sed=sed2, candidate=cand2, report=report2)
    checks = {
        'd_configuration_checks_all_pass_no_override': d_ok,
        'd_leaves_aa_off': aa_enabled_after_d is False,
        'aa_override_applied': planning2.params.admm.anderson_acceleration.get('enabled') is True,
        'hook_records_provenance_in_rule_eleven_checklist': (
            's44_campaign_configuration_checks' in report.get('rule_eleven_checklist', {})),
        'instance_is_per_node_map': report.get('instance', {}).get('assignment') == 'per-node',
    }
    return {'checks': checks, 'configuration_checks': holder['configuration_checks'],
            'aa_settings_after_override': holder2['overrides_applied']}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'output root already exists (write-once): {OUT}')
    os.makedirs(OUT)
    started = time.time()
    results = {'stage': 'P5.15 Addendum 25 item 1 -- campaign harness zero-solve checks',
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'harness_sha256': H.sha256_file(H.HARNESS_PATH),
               'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                                          text=True).stdout.strip()}
    all_pass = True
    for name, fn in (('C1_canonical_candidate', c1_canonical), ('C2_frozen_spec', c2_spec),
                     ('C3_campaign_lock', c3_lock), ('C4_spawn_stub_evaluations', c4_spawn),
                     ('C5_record_schema', c5_record_schema), ('C6_capture_paths', c6_capture_paths),
                     ('C7_configuration_hook', c7_config_hook)):
        print(f'[S44-HCHK] ===== {name} =====', flush=True)
        try:
            out = fn()
            ok = all(out['checks'].values())
        except BaseException as error:  # noqa: BLE001
            out = {'exception': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            ok = False
        out['pass'] = ok
        all_pass = all_pass and ok
        results[name] = out
        print(f'[S44-HCHK] {name}: pass={ok} checks={out.get("checks", out.get("exception"))}', flush=True)
    guard_failures = GUARD.verify(0)
    results['solve_profile_guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures}
    results['all_pass'] = all_pass and not guard_failures
    results['wall_clock_s'] = time.time() - started
    H._write_once_json(os.path.join(OUT, 'harness_checks.json'), results)
    manifest = {}
    for root, _dirs, files in os.walk(OUT):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(OUT, 'manifest_sha256.json'), manifest)
    GUARD.uninstall()
    print(f'[S44-HCHK] ALL PASS={results["all_pass"]} guard={GUARD.counts} failures={guard_failures}')
    if not results['all_pass']:
        sys.exit(1)


if __name__ == '__main__':
    main()
