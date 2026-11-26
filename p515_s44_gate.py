"""
P5.15 Addendum 25 item 1 -- the campaign harness's bitwise GATE (campaign `s44_gate`).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 25; frozen spec v14
`data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`,
`item1_harness.gate` / `gate_companions`, `item3_selection_run.stop_rule`.

Three concurrent evaluations through `p515_s44_campaign_harness.evaluate`,
D configuration (the case file, `fb3de341`, no overrides), cap 500,
concurrency 3:
  1. c_star      : 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9
  2. paper_plan  : node 7 only, 1.62 MVA / 3.24 MWh (nodes 5, 9 zero)
  3. node7_empty : C* at nodes 5 and 9, node 7 zero

GATE (evaluation 1 only -- the companions are designed to differ from D and
are NOT compared to it; "scope a gate per arm"): the c_star evaluation
reproduces arm D bitwise on every numeric trajectory field and cost
(certified at cycle 139, gross_operational_cost 650,966,975.2943751) against
`data/SRP1/Results/P515S39_D_run/`, with the comparator conventions of
`p515_s40_clone_capture_preflight.py` (8f5cff48; `_diff`, exclusion sets,
`ARTIFACT_FILES`, `SIDECAR_JSONL_FILES`, all BY IMPORT) and the diff
classification of `p515_s43_aa_flagoff_gate._classify_diffs` (BY IMPORT):
  - provenance (rule_eleven_checklist subtree): reported, non-gating;
  - aa_new_field (the twelve aa_* trajectory fields D predates, at their
    documented flag-off values): reported, non-gating;
  - known tie-break (the economic_market_cost / generation_cost alias pair's
    `.component` label in the recourse-jump top-k; alphabetical since
    8682cfdd, hash-seed order in D's committed sidecar): reported, non-gating;
  - candidate-specification provenance (NEW here, the one addition): the
    report's `instance` subtree differs because D was built by the uniform
    path (`investment_map=None`: {s_mva, e_mwh, year, assignment: uniform})
    and the harness builds every candidate by the per-node map path. It is
    classified non-gating ONLY IF the map assigns D's exact (s_mva, e_mwh)
    to every active node at D's year (checked here), which with Task A's
    `map_path_candidate_equals_uniform_path_candidate_exactly = True`
    (committed d8df67eb) means the two paths build the identical candidate;
  - everything else is GENUINE and gates.
Scalar checks (gating): c_star certified; cycles_run == 139;
gross_operational_cost == 650966975.2943751 (exact).
Supplementary (reported, NOT part of the gate's comparator conventions):
leak_classification / network_failures / esso_recovery_events /
frozen_snapshots JSONL vs D.

Companions: outcome per spec v14 item 3 (certified?, cycle, cost, bar); a
companion at which D does not certify within cap 500 triggers the spec v14
stop_rule flag (printed prominently, recorded; the Planner decides).

The parent never solves: `SolveProfileGuard(permitted=())` is installed at
the top of this script (before any model module is imported) and verified at
exactly 0 at the end.

EXACT LAUNCH COMMAND (attached; both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_gate.py \\
        > data/SRP1/Results/P515S44/campaign_s44_gate_launch.log 2>&1
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S44 campaign parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- comparator conventions (8f5cff48), BY IMPORT
import p515_s43_aa_flagoff_gate as FG  # noqa: E402 -- diff classification, BY IMPORT

CAMPAIGN_ID = 's44_gate'
CAMPAIGN_ROOT = os.path.join(H.RESULTS_ROOT, 'campaign_s44_gate')
CAP = 500
CONCURRENCY = 3
CANDIDATES = [
    ('c_star', {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)}),
    ('paper_plan', {5: (0.0, 0.0), 7: (1.62, 3.24), 9: (0.0, 0.0)}),
    ('node7_empty', {5: (0.96875, 3.875), 7: (0.0, 0.0), 9: (0.96875, 3.875)}),
]
D_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39_D_run')
D_CERTIFICATION_CYCLE = 139
D_CERTIFIED_COST = 650966975.2943751
SUPPLEMENTARY_JSONL = ('leak_classification_s39_D.jsonl', 'network_failures_s39_D.jsonl',
                       'esso_recovery_events_s39_D.jsonl', 'frozen_snapshots_s39_D.jsonl')
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 25 (campaign harness, gate)',
    'PLANNER_BRIEF_2026-09-13.md Addendum 26',
    'STEP4_DFO_METHOD.md section 2',
    'data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json item1_harness',
]


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _instance_equivalent(d_instance, my_instance, active_nodes):
    """D built by the uniform path, the harness by the per-node map path: the
    instance subtree differs by construction. Equivalent iff the map gives D's
    exact (s_mva, e_mwh) to every active node at D's year."""
    try:
        s, e, y = d_instance['s_mva'], d_instance['e_mwh'], d_instance['year']
        m = my_instance['investment_map']
        return (my_instance.get('year') == y and my_instance.get('assignment') == 'per-node'
                and set(m) == {str(n) for n in active_nodes}
                and all(list(v) == [s, e] for v in m.values()))
    except (KeyError, TypeError):
        return False


def _classify(diffs, instance_ok, prefix):
    prov, aa_new, tie, genuine = FG._classify_diffs(diffs)
    inst, rest = [], []
    for d in genuine:
        f = str(d.get('field', ''))
        if instance_ok and (f == f'{prefix}.instance' or f.startswith(f'{prefix}.instance.')):
            inst.append(d)
        else:
            rest.append(d)
    return {'provenance_diffs': prov, 'aa_new_field_diffs': aa_new, 'known_tie_break_diffs': tie,
            'candidate_specification_provenance_diffs': inst, 'genuine_diffs': rest,
            'n': {'provenance': len(prov), 'aa_new_field': len(aa_new), 'known_tie_break': len(tie),
                  'candidate_specification_provenance': len(inst), 'genuine': len(rest)}}


def compare_c_star_to_d(eval_dir):
    d_report = CP._load_json(os.path.join(D_DIR, 'g_s39_D.json'))
    my_report = CP._load_json(os.path.join(eval_dir, 'g_s39_D.json'))
    active_nodes = (my_report or {}).get('active_distribution_network_nodes') or list(H.ACTIVE_NODES)
    instance_ok = _instance_equivalent((d_report or {}).get('instance', {}),
                                       (my_report or {}).get('instance', {}), active_nodes)
    artifacts = {}
    for fname in CP.ARTIFACT_FILES:
        a = CP._load_json(os.path.join(D_DIR, fname))
        b = CP._load_json(os.path.join(eval_dir, fname))
        if a is None or b is None:
            artifacts[fname] = {'error': f'missing: D={a is not None} mine={b is not None}'}
            continue
        artifacts[fname] = _classify(CP._diff(a, b, fname), instance_ok, fname)
    sidecars = {}
    for fname in CP.SIDECAR_JSONL_FILES:
        a = CP._load_jsonl(os.path.join(D_DIR, fname))
        b = CP._load_jsonl(os.path.join(eval_dir, fname))
        if a is None or b is None:
            sidecars[fname] = {'error': f'missing: D={a is not None} mine={b is not None}'}
            continue
        c = _classify(CP._diff(a, b, fname), instance_ok, fname)
        c['n_rows'] = {'D': len(a), 'mine': len(b)}
        sidecars[fname] = c
    supplementary = {}
    for fname in SUPPLEMENTARY_JSONL:
        a = CP._load_jsonl(os.path.join(D_DIR, fname))
        b = CP._load_jsonl(os.path.join(eval_dir, fname))
        if a is None or b is None:
            supplementary[fname] = {'error': f'missing: D={a is not None} mine={b is not None}'}
            continue
        raw = CP._diff(a, b, fname)
        supplementary[fname] = {'n_rows': {'D': len(a), 'mine': len(b)}, 'n_raw_diffs': len(raw),
                                'first_raw_diffs': raw[:20]}
    all_present = all('error' not in v for v in list(artifacts.values()) + list(sidecars.values()))
    n_genuine = sum(v['n']['genuine'] for v in list(artifacts.values()) + list(sidecars.values())
                    if 'n' in v)
    totals = {}
    for v in list(artifacts.values()) + list(sidecars.values()):
        for k, n in (v.get('n') or {}).items():
            totals[k] = totals.get(k, 0) + n
    scalar = {
        'cycles_run': {'mine': (my_report or {}).get('cycles_run'), 'D': D_CERTIFICATION_CYCLE},
        'gross_operational_cost': {'mine': (my_report or {}).get('gross_operational_cost'), 'D': D_CERTIFIED_COST},
    }
    for v in scalar.values():
        v['match'] = (v['mine'] == v['D'])
    return {
        'd_dir': os.path.relpath(D_DIR, REPO),
        'eval_dir': os.path.relpath(eval_dir, REPO),
        'comparator': 'p515_s40_clone_capture_preflight._diff (8f5cff48) + p515_s43_aa_flagoff_gate._classify_diffs',
        'excluded_field_names': sorted(CP.EXCLUDE_KEY_NAMES),
        'excluded_dotted_suffixes': sorted(CP.EXCLUDE_DOTTED_SUFFIXES),
        'intentional_difference_suffixes': sorted(CP.INTENTIONAL_DIFF_SUFFIXES),
        'known_tie_break_alias_pairs': [sorted(p) for p in FG.KNOWN_TIE_BREAK_ALIAS_PAIRS],
        'instance_equivalence': {'D_instance': (d_report or {}).get('instance'),
                                 'mine_instance': (my_report or {}).get('instance'),
                                 'equivalent': instance_ok},
        'artifacts': artifacts, 'sidecars': sidecars,
        'supplementary_not_gating': supplementary,
        'totals_by_class': totals,
        'all_artifacts_present': all_present,
        'n_genuine_diffs': n_genuine,
        'scalar_checks': scalar,
    }


def _summary(record):
    if record is None:
        return None
    return {
        'label': record.get('candidate_label'), 'key': (record.get('candidate_key') or '')[:16],
        'status': record.get('status'), 'barrier': record.get('barrier'),
        'barrier_cause': record.get('barrier_cause'),
        'certification_cycle': record.get('certification_cycle'), 'cycles_run': record.get('cycles_run'),
        'certified_cost_gross': record.get('certified_cost'),
        'terminal_gross_operational_cost': record.get('terminal_gross_operational_cost'),
        'bar': (record.get('bar') or {}).get('value'),
        'first_pass_cycle_per_channel': record.get('first_pass_cycle_per_channel'),
        'terminal_ratios_per_channel': record.get('terminal_ratios_per_channel'),
        'rule_ten_terminal_step_over_threshold': (record.get('rule_ten') or {}).get('terminal_step_over_threshold'),
        'settlement_remainder': (record.get('settlement_remainder') or {}).get('value'),
        'local_solve_failures': record.get('local_solve_failures'),
        'network_failures_classes': ((record.get('network_failures_summary') or {}).get('classes')),
        'peak_rss': record.get('peak_rss'), 'wall_time_s': record.get('wall_time_s'),
        'parent_view': record.get('parent_view'),
        'campaign_spec_sha256': record.get('campaign_spec_sha256'),
    }


def main():
    started = time.time()
    failures = H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=('p515_s44_gate.py',))
    if not os.path.isfile(os.path.join(D_DIR, 'g_s39_D.json')):
        failures.append(f"D's committed reference missing: {D_DIR}")
    if failures:
        for f in failures:
            _log(f'[S44-GATE PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    _log('[S44-GATE] preconditions passed (no legacy/campaign lock, no forbidden process, fresh root, '
         'production + harness + case file clean in git, D reference present)')
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        CAMPAIGN_ROOT, CAMPAIGN_ID, CANDIDATES,
        configuration={'name': 'D (case file oracle, fb3de341)', 'arm_label': 's39_D', 'overrides': {},
                       'note': 'no Python-side overrides; num_max_iters := cap (run_admm_arm always sets it)'},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY, required_consecutive_cycles=10,
        extra={'gate_script': 'p515_s44_gate.py',
               'gate_script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'gate': 'evaluation 1 (c_star) bitwise vs data/SRP1/Results/P515S39_D_run',
               'companions': 'paper_plan, node7_empty: D-configuration selection-run evaluations (spec v14 item 3)'})
    _log(f'[S44-GATE] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    lock = H.acquire_campaign_lock(CAMPAIGN_ID, spec_sha)
    _log(f'[S44-GATE] campaign lock acquired: {lock}')
    records = None
    try:
        ctx = H.CampaignContext(CAMPAIGN_ROOT, spec_path, spec_sha, spec, log=_log)
        records = H.evaluate([c for _label, c in CANDIDATES], ctx)
        batch_info = getattr(H.evaluate, 'last_batch_info', {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log('[S44-GATE] campaign lock released')

    by_label = {r.get('candidate_label'): r for r in records}
    c_rec = by_label.get('c_star')
    c_dir = os.path.join(CAMPAIGN_ROOT, 'evals', spec['candidates'][0]['eval_dir'])
    comparison = compare_c_star_to_d(c_dir)
    gate_pass = bool(
        c_rec is not None and c_rec.get('status') == 'certified'
        and comparison['all_artifacts_present'] and comparison['n_genuine_diffs'] == 0
        and all(v['match'] for v in comparison['scalar_checks'].values()))
    companions = {lab: _summary(by_label.get(lab)) for lab in ('paper_plan', 'node7_empty')}
    stop_rule = {lab: (s is None or s.get('status') != 'certified') for lab, s in companions.items()}
    spec_hash_in_every_record = all(r.get('campaign_spec_sha256') == spec_sha for r in records)
    guard_failures = PARENT_GUARD.verify(0)
    results = {
        'stage': 'P5.15 Addendum 25 item 1 -- campaign harness bitwise gate (campaign s44_gate)',
        'authority': AUTHORITY,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha,
        'gate_scope': 'evaluation 1 (c_star) only; companions are not compared to D',
        'gate_pass': gate_pass,
        'c_star': _summary(c_rec),
        'c_star_vs_D': comparison,
        'companions': companions,
        'stop_rule_spec_v14_item3_triggered': stop_rule,
        'spec_sha256_in_every_record': spec_hash_in_every_record,
        'batch_info': batch_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started,
    }
    out = os.path.join(CAMPAIGN_ROOT, 'gate_results.json')
    H._write_once_json(out, results)
    manifest = {}
    for root, _dirs, files in os.walk(CAMPAIGN_ROOT):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_manifest_sha256.json'), manifest)
    PARENT_GUARD.uninstall()

    _log(f"[S44-GATE] c_star: {results['c_star']}")
    _log(f"[S44-GATE] c_star vs D: totals_by_class={comparison['totals_by_class']} "
         f"all_artifacts_present={comparison['all_artifacts_present']} scalar={comparison['scalar_checks']} "
         f"instance_equivalent={comparison['instance_equivalence']['equivalent']}")
    for lab, s in companions.items():
        _log(f'[S44-GATE] companion {lab}: {s}')
    for lab, trig in stop_rule.items():
        if trig:
            _log(f'[S44-GATE] ****** STOP RULE (spec v14 item3.stop_rule): D did NOT certify within cap {CAP} '
                 f'at {lab} -- the Planner decides ******')
    _log(f'[S44-GATE] spec sha256 in every record: {spec_hash_in_every_record}; parent guard '
         f'{PARENT_GUARD.counts} verify0_failures={guard_failures}')
    _log(f'[S44-GATE] GATE {"PASS" if gate_pass else "FAIL"}')
    if not gate_pass or guard_failures:
        sys.exit(1)


if __name__ == '__main__':
    main()
