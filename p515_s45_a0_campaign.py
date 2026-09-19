"""
P5.15 Addendum 27, Phase A step A0 -- the eight A0 evaluations through the
campaign harness (`p515_s44_campaign_harness.py`), configuration = the case
file alone (AA-on keep_memory adopted in data/SRP1/SRP1_params.json), NO
overrides, no post-certification.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27; frozen spec v15
`data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json`
(`A0`, `execution`, `configuration`); P5_15_S45_REVERIFY_RULING.md (the oracle
re-verified; consequence 2: the bar is the GROSS cost step, fixed in the
harness by task W5 before this campaign).

Campaign id `s45_a0`, root `data/SRP1/Results/P515S45/campaign_s45_a0/`: the 8
points of spec v15 `A0.points`, investment year 2025, nodes not named are zero:
  x0                         : 0 at nodes 5, 7, 9
  n5_p0.25_e0.5 / _e1.0      : node 5 at 0.25 MVA / 0.5 MWh, 0.25 MVA / 1.0 MWh
  n7_p0.25_e0.5 / _e1.0      : node 7, same
  n9_p0.25_e0.5 / _e1.0      : node 9, same
  lattice_plan_n7_p1.5_e3.0  : node 7 at 1.5 MVA / 3.0 MWh
cap 500, 10 consecutive all-pass cycles, concurrency 8 (one batch: every point
is launched at once, so nothing is ever pending when a result arrives).
Each point's I(x) (corrected cost file e17bd588...) is taken from W2's committed
`data/SRP1/Results/P515S45/investment_cost/investment_cost_results.json`
(9e623dd3) by candidate key and frozen into the spec; F = I + Q with Q the
certified `gross_operational_cost` (settlement-excluded; terminal salvage
reported, excluded from F).

STOP RULE (spec v15 A0.stop_rule / execution.barrier): any A0 point not
certified -> campaign_results.json carries `STOP_FOR_REVIEW: true` (and the
launcher prints it prominently and exits 3). The evaluations already running
finish; nothing further is launched (there is nothing further: one batch).

MEMORY PREFLIGHT: 8 concurrent children x ~2.2-2.5 GiB peak each (S44/S45 RSS
evidence). --run refuses unless free + inactive memory (vm_stat) >= 8 x 3 GiB;
the reading is recorded (at --freeze too, non-gating there).

Two modes, both attached, both streams captured, never detached:
  --freeze          ZERO SOLVES. Campaign preconditions (locks, live processes,
                    clean git for the production files AND this script, root
                    absent); the production loader reads the case file's AA
                    dict == the declaration; spec v15, the cost file and W2's
                    I(x) file hash-verified; A0 points == spec v15 A0.points;
                    every point found in W2's file; rule eleven (capture paths
                    for every campaign_results field); the spec frozen by the
                    harness's `freeze_campaign_spec` into the campaign root and
                    validated. No lock, no evaluation.
  --run --spec-sha256 <sha>
                    loads THAT frozen spec (the root must hold only it),
                    re-checks everything above against the files on disk plus
                    the harness / case-file / script sha256 recorded in the
                    spec, the memory preflight (refusing), rule eleven; takes
                    the campaign lock; evaluates the 8 points; writes
                    campaign_results.json and campaign_manifest_sha256.json.
Exit codes (--run): 0 all 8 certified; 3 STOP_FOR_REVIEW (a point not
certified, harness clean); 1 harness / guard / precondition failure.
The parent never solves: SolveProfileGuard(permitted=()) is installed before
any model import and verified at exactly 0 (both modes).

EXACT COMMANDS (repo root):
  freeze (zero solves):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s45_a0_campaign.py --freeze \\
        > data/SRP1/Results/P515S45/campaign_s45_a0_freeze_launch.log 2>&1
  run (Planner):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s45_a0_campaign.py --run \\
        --spec-sha256 <sha256 printed by --freeze> \\
        > data/SRP1/Results/P515S45/campaign_s45_a0_launch.log 2>&1
"""

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S45 A0 campaign parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

CAMPAIGN_ID = 's45_a0'
_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
CAMPAIGN_ROOT = os.path.join(REPO, _P45, 'campaign_s45_a0')
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
CONCURRENCY = 8
REQUIRED_CONSECUTIVE_CYCLES = 10
INVESTMENT_YEAR = 2025
SPEC_V15 = {'path': os.path.join(_P45, 'frozen_s45_phaseA_spec_v15_5feefd7b.json'),
            'sha256': '5feefd7b642fc3d480156ad5e52ed6e1cf9d6698cfd40dbb33389bab6e6229fe'}
COST_FILE = {'path': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
             'sha256': 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'}
INVESTMENT_COST_RESULTS = {'path': os.path.join(_P45, 'investment_cost', 'investment_cost_results.json'),
                           'sha256': '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b',
                           'commit': '9e623dd304837f2632964f47130d0860c1f016ca',
                           'field': 'candidates.<label>.I_new_eur (corrected cost file), matched by candidate_key'}
# spec v15 A0.points, in spec order; nodes not named are zero (spec v15: "other nodes zero")
A0_POINTS = (
    ('x0', {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}),
    ('n5_p0.25_e0.5', {5: (0.25, 0.5), 7: (0.0, 0.0), 9: (0.0, 0.0)}),
    ('n5_p0.25_e1.0', {5: (0.25, 1.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}),
    ('n7_p0.25_e0.5', {5: (0.0, 0.0), 7: (0.25, 0.5), 9: (0.0, 0.0)}),
    ('n7_p0.25_e1.0', {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}),
    ('n9_p0.25_e0.5', {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.25, 0.5)}),
    ('n9_p0.25_e1.0', {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.25, 1.0)}),
    ('lattice_plan_n7_p1.5_e3.0', {5: (0.0, 0.0), 7: (1.5, 3.0), 9: (0.0, 0.0)}),
)
GIB = 1 << 30
MEMORY_PER_CHILD_BUDGET_BYTES = 3 * GIB
MEMORY_REQUIRED_BYTES = CONCURRENCY * MEMORY_PER_CHILD_BUDGET_BYTES
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 27 (Phase A, A0)',
    'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json A0, execution, configuration',
    'P5_15_S45_REVERIFY_RULING.md (oracle re-verified; consequence 2: bar on the gross cost step, fixed by W5)',
]
STOP_RULE = ('spec v15 A0.stop_rule: "a non-certified point stops the campaign for review"; execution.barrier: '
             '"non-certified evaluations recorded with cause; in A0 a non-certified point stops the campaign". '
             'Implementation: one batch of 8 at concurrency 8 (nothing pending); any point not certified -> '
             'STOP_FOR_REVIEW true in campaign_results.json, exit 3.')
OBJECTIVE_CONVENTION = ('Q(x) = certified_cost = gross_operational_cost (settlement-excluded); F(x) = I(x) + Q(x); '
                        'terminal_salvage_value reported, excluded from F (Addendum 27 item 3)')
EXTRA_CLEAN_FILES = (os.path.basename(__file__), 'p515_s45_harness_phasea_check.py')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _banner(lines):
    bar = '!' * 100
    _log(bar)
    for line in lines:
        _log(f'!!! {line}')
    _log(bar)


# ======================================================================================================================
#  checks shared by --freeze and --run (zero solves)
# ======================================================================================================================
def _check_case_file_loads_to_declaration():
    from planning_parameters import PlanningParameters
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    loaded = params.admm.anderson_acceleration
    return {'loaded': loaded, 'equals_declaration': loaded == CASE_FILE_AA}


def _check_pinned_files():
    out = {}
    for name, pin in (('spec_v15', SPEC_V15), ('cost_file', COST_FILE), ('investment_cost_results',
                                                                          INVESTMENT_COST_RESULTS)):
        path = os.path.join(REPO, pin['path'])
        got = H.sha256_file(path) if os.path.isfile(path) else None
        out[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256_on_disk': got,
                     'match': got == pin['sha256']}
    return out


def _points_vs_spec_v15():
    with open(os.path.join(REPO, SPEC_V15['path'])) as handle:
        a0 = json.load(handle)['A0']
    spec_points = []
    for p in a0['points']:
        nodes = {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}
        for n, (s_val, e_val) in p['nodes'].items():
            nodes[int(n)] = (float(s_val), float(e_val))
        spec_points.append((p['label'], nodes))
    mine = [(label, {n: (float(v[0]), float(v[1])) for n, v in pts.items()}) for label, pts in A0_POINTS]
    return {'equal_to_spec_v15_A0_points_in_order': mine == spec_points,
            'spec_n': a0.get('n'), 'n_points': len(A0_POINTS), 'n_matches_spec_n': a0.get('n') == len(A0_POINTS),
            'spec_investment_year': a0.get('investment_year'),
            'investment_year_matches': a0.get('investment_year') == INVESTMENT_YEAR == H.INVESTMENT_YEAR,
            'spec_post_certification': a0.get('post_certification'), 'spec_stop_rule': a0.get('stop_rule'),
            'spec_batch': a0.get('batch')}


def _investment_costs():
    """I(x) per point from W2's committed file, by candidate key; every point must be found, and every
    entry carrying that key must agree on I."""
    with open(os.path.join(REPO, INVESTMENT_COST_RESULTS['path'])) as handle:
        cands = json.load(handle)['candidates']
    per_point, problems = {}, []
    for label, pts in A0_POINTS:
        key = H.candidate_key(H.canonical_candidate(pts))
        hits = {name: c for name, c in cands.items() if c.get('candidate_key') == key}
        values = sorted({c.get('I_new_eur') for c in hits.values()}, key=repr)
        if not hits:
            problems.append(f'{label}: candidate key {key[:16]} not found in {INVESTMENT_COST_RESULTS["path"]}')
            per_point[label] = {'candidate_key': key, 'found': False}
            continue
        if len(values) != 1 or values[0] is None:
            problems.append(f'{label}: entries with key {key[:16]} disagree on / lack I_new_eur: {values}')
        first = hits[sorted(hits)[0]]
        per_point[label] = {
            'candidate_key': key, 'found': True, 'matched_entries': sorted(hits),
            'I_x_eur': values[0] if len(values) == 1 else None,
            'I_power_eur': first.get('I_new_power_eur'), 'I_energy_eur': first.get('I_new_energy_eur'),
            'budget_feasible_corrected_file': first.get('budget_feasible_new'),
            'first_stage_feasible_production_check': first.get('first_stage_feasible_new_production_check'),
            'note': 'budget not applied in Phase A (spec v15 master_constraints.budget); recorded only'}
    return per_point, problems


def memory_preflight():
    """free + inactive (vm_stat pages x page size) against CONCURRENCY x 3 GiB."""
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    first = text.splitlines()[0]
    page = int(first.split('page size of')[1].split('bytes')[0].strip())
    pages = {}
    for line in text.splitlines()[1:]:
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().rstrip('.')
            if v.isdigit():
                pages[k.strip()] = int(v)
    free_b = pages.get('Pages free', 0) * page
    inactive_b = pages.get('Pages inactive', 0) * page
    total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, check=True).stdout)
    avail = free_b + inactive_b
    return {'utc': datetime.now(timezone.utc).isoformat(), 'page_size_bytes': page,
            'pages_free': pages.get('Pages free'), 'pages_inactive': pages.get('Pages inactive'),
            'pages_speculative': pages.get('Pages speculative'), 'hw_memsize_bytes': total,
            'free_plus_inactive_bytes': avail, 'free_plus_inactive_gib': avail / GIB,
            'required_bytes': MEMORY_REQUIRED_BYTES, 'required_gib': MEMORY_REQUIRED_BYTES / GIB,
            'rule': f'free + inactive >= {CONCURRENCY} x {MEMORY_PER_CHILD_BUDGET_BYTES // GIB} GiB',
            'pass': avail >= MEMORY_REQUIRED_BYTES}


CAMPAIGN_RESULT_FIELDS = (
    'status', 'certification_cycle', 'certified_cost', 'bar', 'rule_ten', 'terminal_salvage_value', 'I_x_eur',
    'F_eur', 'wall_time', 'peak_rss', 'aa_action_counts', 'per_cycle_trajectory')


def rule_eleven():
    """Before any evaluation: a capture path exists for every per-point campaign_results field."""
    harness_checks = H.assert_record_capture_paths()  # the harness's own rule-eleven assertion (no solve)
    build_src = inspect.getsource(H.build_evaluation_record)
    child_src = inspect.getsource(H._child_real)
    import shared_resources_planning as srp
    rc_src = inspect.getsource(srp._get_operational_recourse_components)
    checks = {
        'status': "'status': status" in build_src,
        'certification_cycle': "'certification_cycle':" in build_src,
        'certified_cost_gross': "'certified_cost': report.get('gross_operational_cost')" in build_src,
        'bar_gross_step': "'bar': bar" in build_src and "gross_step_abs" in inspect.getsource(H._max_step_last_n),
        'bar_net_reported': "'bar_net_recourse_step_reported':" in build_src,
        'rule_ten': "'rule_ten':" in build_src and "'terminal_step_over_threshold':" in build_src,
        'terminal_salvage_value_in_recourse_components': ("'recourse_components': rc" in build_src
                                                          and "'terminal_salvage_value':" in rc_src),
        'I_x_frozen_in_spec': True,  # asserted by _validate_spec on the frozen spec itself
        'wall_time': "'wall_time_s': wall" in build_src,
        'peak_rss': "'peak_rss': peak_rss" in build_src,
        'aa_action_counts': ("'aa_per_cycle': holder.get('aa_sidecar')" in child_src
                             and "'action_counts'" in inspect.getsource(H.aa_sidecar_summary)),
        'per_cycle_trajectory_written': "'per_cycle_record.jsonl'" in child_src,
        'per_cycle_fields': all(f in H.PER_CYCLE_RECORD_FIELDS for f in (
            'cycle', 'gross_operational_cost', 'terminal_salvage_value', 'objective_change_abs',
            'objective_tolerance', 'consecutive_converged_cycles', 'boyd_all_pass', 'local_solves_ok')),
        'error_records_same_schema': ("'anderson_acceleration_effective_in_child':" in inspect.getsource(H.main_child)
                                      and "'anderson_acceleration_effective_in_child':"
                                      in inspect.getsource(H._barrier_record_for_missing)),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (A0 campaign_results): capture paths missing: {missing}')
    return {'campaign_results_fields': list(CAMPAIGN_RESULT_FIELDS), 'checks': checks,
            'harness_record_capture_checklist': harness_checks}


def _validate_spec(spec, i_x):
    entries = spec['candidates']
    by_label = {e['label']: e for e in entries}
    checks = {
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_ID,
        'eight_entries_in_order': [e['label'] for e in entries] == [lab for lab, _p in A0_POINTS],
        'declaration': spec['configuration'].get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'no_campaign_overrides': spec['configuration'].get('overrides') == {},
        'cap': spec.get('cap') == CAP,
        'concurrency_equals_n_points': spec.get('concurrency') == CONCURRENCY == len(A0_POINTS),
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label_s39_D': spec['configuration'].get('arm_label') == 's39_D',
        'not_a_stub_spec': not spec.get('extra', {}).get('test_only_stub'),
        'spec_v15_recorded': spec['extra'].get('spec_v15') == SPEC_V15,
        'cost_file_recorded': (spec['extra'].get('cost_file') or {}).get('sha256') == COST_FILE['sha256'],
        'investment_cost_results_recorded': spec['extra'].get('investment_cost_results') == INVESTMENT_COST_RESULTS,
    }
    for label, pts in A0_POINTS:
        e = by_label.get(label) or {}
        canon = H.canonical_candidate(pts)
        key = H.candidate_key(canon)
        rec_i = (spec['extra'].get('points') or {}).get(label) or {}
        checks[f'{label}:canonical_key'] = e.get('canonical') == canon and e.get('key') == key
        checks[f'{label}:no_overrides_no_post_certification'] = (e.get('overrides') == {}
                                                                 and e.get('post_certification') is None)
        checks[f'{label}:effective_aa_is_declaration'] = e.get('effective_anderson_acceleration') == CASE_FILE_AA
        checks[f'{label}:eval_key_recomputes'] = e.get('eval_key') == H.evaluation_key(key, {},
                                                                                      case_file_aa=CASE_FILE_AA)
        checks[f'{label}:I_x_frozen'] = (rec_i.get('candidate_key') == key and rec_i.get('I_x_eur') is not None
                                         and rec_i.get('I_x_eur') == (i_x.get(label) or {}).get('I_x_eur'))
    return checks


def _common_checks():
    """Everything checked identically at --freeze and --run; returns (failures, evidence)."""
    failures, evidence = [], {}
    case_file = _check_case_file_loads_to_declaration()
    evidence['case_file_aa'] = case_file
    if not case_file['equals_declaration']:
        failures.append(f"case file AA {case_file['loaded']} != declaration {CASE_FILE_AA}")
    pinned = _check_pinned_files()
    evidence['pinned_files'] = pinned
    failures += [f'{k}: sha256 on disk {v["sha256_on_disk"]} != pinned {v["sha256_pinned"]}'
                 for k, v in pinned.items() if not v['match']]
    pts = _points_vs_spec_v15()
    evidence['points_vs_spec_v15'] = pts
    if not (pts['equal_to_spec_v15_A0_points_in_order'] and pts['n_matches_spec_n'] and pts['investment_year_matches']):
        failures.append(f'A0 points differ from spec v15 A0.points: {pts}')
    i_x, problems = _investment_costs()
    evidence['investment_cost_per_point'] = i_x
    failures += problems
    try:
        evidence['rule_eleven'] = rule_eleven()
    except AssertionError as error:
        failures.append(str(error))
    return failures, evidence, i_x


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def freeze(started):
    failures = H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
    more, evidence, i_x = _common_checks()
    failures += more
    memory = memory_preflight()
    if failures:
        for f in failures:
            _log(f'[S45-A0 FREEZE PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    points = {label: dict(i_x[label]) for label, _p in A0_POINTS}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        CAMPAIGN_ROOT, CAMPAIGN_ID, [(label, pts) for label, pts in A0_POINTS],
        configuration={'name': 'case file alone: AA keep_memory adopted in data/SRP1/SRP1_params.json (Addendum 27)',
                       'arm_label': 's39_D', 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'note': ('no overrides; the case file carries AA (declared, checked by the child hook); '
                                'num_max_iters := cap (run_admm_arm always sets it)')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY,
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra={'campaign_script': os.path.basename(__file__),
               'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'mode': 'full', 'phase': 'A0',
               'spec_v15': dict(SPEC_V15),
               'cost_file': dict(COST_FILE, sha256_on_disk_at_freeze=evidence['pinned_files']['cost_file']['sha256_on_disk']),
               'investment_cost_results': dict(INVESTMENT_COST_RESULTS),
               'points': points,
               'post_certification': 'none (spec v15 A0.post_certification: "none (hull polish on incumbents only)")',
               'stop_rule': STOP_RULE,
               'objective_convention': OBJECTIVE_CONVENTION,
               'bar_definition': ('record.bar = max over the last 10 cycles of |gross_operational_cost[k] - '
                                  'gross_operational_cost[k-1]| (harness W5); record.bar_net_recourse_step_reported '
                                  '= the net-recourse objective_change_abs max (reported)'),
               'memory_preflight_rule': (f'--run refuses unless free + inactive >= {CONCURRENCY} x '
                                         f'{MEMORY_PER_CHILD_BUDGET_BYTES // GIB} GiB'),
               'memory_at_freeze_non_gating': memory,
               'rule_eleven_fields': list(CAMPAIGN_RESULT_FIELDS)})
    checks = _validate_spec(spec, i_x)
    guard_failures = PARENT_GUARD.verify(0)
    _log(f'[S45-A0] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    for e in spec['candidates']:
        _log(f"[S45-A0]   {e['label']}: key={e['key'][:16]} eval_key={e['eval_key'][:16]} eval_dir={e['eval_dir']} "
             f"I(x)={points[e['label']]['I_x_eur']}")
    _log(f'[S45-A0] spec checks: all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f"[S45-A0] case file AA (production loader): {evidence['case_file_aa']}")
    _log(f"[S45-A0] pinned files: {evidence['pinned_files']}")
    _log(f"[S45-A0] points vs spec v15: {evidence['points_vs_spec_v15']}")
    _log(f"[S45-A0] rule eleven: {evidence['rule_eleven']['checks']}")
    _log(f"[S45-A0] memory at freeze (non-gating): free+inactive={memory['free_plus_inactive_gib']:.2f} GiB, "
         f"required at --run {memory['required_gib']:.0f} GiB -> would {'PASS' if memory['pass'] else 'REFUSE'} now")
    _log(f'[S45-A0] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures} '
         f'wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not guard_failures
    _log(f'[S45-A0] freeze {"OK" if ok else "NOT OK"}')
    PARENT_GUARD.uninstall()
    if not ok:
        sys.exit(1)


# ======================================================================================================================
#  --run
# ======================================================================================================================
def _per_cycle_trajectory(rec):
    path = rec.get('per_cycle_record_path')
    if not path or not os.path.isfile(os.path.join(REPO, path)):
        return {'path': path, 'present': False}, []
    with open(os.path.join(REPO, path)) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {'path': path, 'present': True, 'sha256': H.sha256_file(os.path.join(REPO, path)),
            'n_rows': len(rows), 'n_rows_equals_cycles_run': len(rows) == rec.get('cycles_run')}, rows


def _point_result(label, rec, spec_point):
    rec = rec or {}
    traj, rows = _per_cycle_trajectory(rec)
    certified = rec.get('status') == 'certified'
    q = rec.get('certified_cost') if certified else None
    i_x = spec_point.get('I_x_eur')
    last = rows[-1] if rows else {}
    prev = rows[-2] if len(rows) >= 2 else {}
    gross_step = (abs(last['gross_operational_cost'] - prev['gross_operational_cost'])
                  if last.get('gross_operational_cost') is not None and prev.get('gross_operational_cost') is not None
                  and prev.get('cycle') == (last.get('cycle') or 0) - 1 else None)
    tol = last.get('objective_tolerance')
    rt = rec.get('rule_ten') or {}
    return {
        'label': label, 'candidate_key': rec.get('candidate_key'), 'candidate_canonical': rec.get('candidate_canonical'),
        'eval_key': rec.get('eval_key'), 'eval_dir': rec.get('eval_dir'),
        'status': rec.get('status'), 'barrier': rec.get('barrier'), 'barrier_cause': rec.get('barrier_cause'),
        'certification_cycle': rec.get('certification_cycle'), 'cycles_run': rec.get('cycles_run'),
        'certified_cost_gross_settlement_excluded': q,
        'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost'),
        'bar': {'value': (rec.get('bar') or {}).get('value'), 'definition': (rec.get('bar') or {}).get('definition'),
                'n_steps_available': (rec.get('bar') or {}).get('n_steps_available')},
        'bar_net_recourse_step_reported': (rec.get('bar_net_recourse_step_reported') or {}).get('value'),
        'rule_ten': {
            'terminal_step_over_threshold_production': rt.get('terminal_step_over_threshold'),
            'terminal_objective_change_abs_production_net': rt.get('terminal_objective_change_abs'),
            'terminal_objective_tolerance': rt.get('terminal_objective_tolerance'),
            'terminal_gross_step_abs': gross_step,
            'terminal_gross_step_over_threshold': (gross_step / tol) if (gross_step is not None and tol) else None,
            'boyd_terminal_ratio_max_per_channel': rt.get('boyd_terminal_ratio_max_per_channel'),
            'note': ('production ratio = the net-recourse step the stopping test used / its tolerance; the gross '
                     'version is computed here from the per-cycle record (terminal row vs its predecessor)')},
        'terminal_salvage_value': (rec.get('recourse_components') or {}).get('terminal_salvage_value'),
        'I_x_eur': i_x, 'F_eur': (i_x + q) if (q is not None and i_x is not None) else None,
        'wall_time': {'record': rec.get('wall_time_s'), 'parent_view_s': (rec.get('parent_view') or {}).get('wall_s')},
        'peak_rss': {'record': rec.get('peak_rss'),
                     'parent_wait4_ru_maxrss_bytes': (rec.get('parent_view') or {}).get('wait4_ru_maxrss')},
        'aa_action_counts': (rec.get('aa_per_cycle') or {}).get('action_counts'),
        'aa_per_cycle': rec.get('aa_per_cycle'),
        'first_pass_cycle_per_channel': rec.get('first_pass_cycle_per_channel'),
        'terminal_ratios_per_channel': rec.get('terminal_ratios_per_channel'),
        'local_solve_failures': rec.get('local_solve_failures'),
        'anderson_acceleration_effective_in_child': rec.get('anderson_acceleration_effective_in_child'),
        'case_file_sha256_in_child': rec.get('case_file_sha256_in_child'),
        'exit_code': (rec.get('parent_view') or {}).get('exit_code'),
        'per_cycle_trajectory': traj,
    }


def run(started, spec_sha256):
    spec_path, spec = H.load_frozen_spec(CAMPAIGN_ROOT, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {CAMPAIGN_ROOT}']
    root_contents = sorted(os.listdir(CAMPAIGN_ROOT))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    more, evidence, i_x = _common_checks()
    failures += more
    checks = _validate_spec(spec, i_x)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    if spec['harness']['sha256'] != H.sha256_file(H.HARNESS_PATH):
        failures.append('harness sha256 differs from the frozen spec')
    if spec['configuration']['case_file_sha256'] != H.sha256_file(H.CASE_FILE):
        failures.append('case file sha256 differs from the frozen spec')
    if spec['extra'].get('campaign_script_sha256') != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script sha256 differs from the frozen spec')
    memory = memory_preflight()
    _log(f"[S45-A0] memory preflight: free+inactive={memory['free_plus_inactive_gib']:.2f} GiB, required "
         f"{memory['required_gib']:.0f} GiB -> {'PASS' if memory['pass'] else 'REFUSE'}")
    if not memory['pass']:
        failures.append(f'memory preflight: free+inactive {memory["free_plus_inactive_gib"]:.2f} GiB < '
                        f'{memory["required_gib"]:.0f} GiB ({memory["rule"]})')
    if failures:
        for f in failures:
            _log(f'[S45-A0 PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    head = H._git(['rev-parse', 'HEAD'])
    _log(f'[S45-A0] preconditions passed; spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; '
         f'git HEAD {head} (spec frozen at {spec["git_head"]})')
    lock = H.acquire_campaign_lock(CAMPAIGN_ID, spec_sha256)
    _log(f'[S45-A0] campaign lock acquired: {lock}')
    try:
        ctx = H.CampaignContext(CAMPAIGN_ROOT, spec_path, spec_sha256, spec, log=_log)
        records = H.evaluate([label for label, _p in A0_POINTS], ctx)
        batch_info = getattr(H.evaluate, 'last_batch_info', {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log('[S45-A0] campaign lock released')
    by_label = {(r or {}).get('candidate_label'): r for r in records}
    points = {label: _point_result(label, by_label.get(label), spec['extra']['points'][label])
              for label, _p in A0_POINTS}
    non_certified = [lab for lab, p in points.items() if p['status'] != 'certified']
    harness_errors = [lab for lab, p in points.items()
                      if p['status'] not in ('certified', 'not_certified') or p['exit_code'] != 0]
    stop = bool(non_certified)
    guard_failures = PARENT_GUARD.verify(0)
    results = {
        'STOP_FOR_REVIEW': stop,
        'non_certified_points': non_certified,
        'stop_rule': STOP_RULE,
        'stage': 'P5.15 Addendum 27 Phase A, A0 -- 8 points, case-file AA-on keep_memory, campaign harness',
        'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': head, 'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256, 'objective_convention': OBJECTIVE_CONVENTION,
        'points': points,
        'harness_errors': harness_errors,
        'memory_preflight_at_run': memory,
        'pre_run_evidence': {k: evidence[k] for k in ('case_file_aa', 'pinned_files', 'points_vs_spec_v15')},
        'rule_eleven_asserted_before_run': evidence['rule_eleven']['checks'],
        'batch_info': batch_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_results.json'), results)
    manifest = {}
    for r_, _dirs, files in os.walk(CAMPAIGN_ROOT):
        for fname in sorted(files):
            fpath = os.path.join(r_, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_manifest_sha256.json'), manifest)
    PARENT_GUARD.uninstall()
    for lab, p in points.items():
        _log(f"[S45-A0] {lab}: status={p['status']} cycles={p['cycles_run']} cert={p['certification_cycle']} "
             f"Q={p['certified_cost_gross_settlement_excluded']} I={p['I_x_eur']} F={p['F_eur']} bar={p['bar']['value']} "
             f"rule10={p['rule_ten']['terminal_step_over_threshold_production']} aa={p['aa_action_counts']}")
    _log(f'[S45-A0] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}')
    if harness_errors:
        _log(f'[S45-A0] evaluations with an error status or non-zero exit: {harness_errors}')
    if stop:
        _banner([f'STOP_FOR_REVIEW: A0 point(s) not certified: {non_certified}',
                 'spec v15 A0 stop rule: a non-certified point stops the campaign for review.',
                 'Nothing further is launched. See campaign_results.json.'])
    if guard_failures or harness_errors:
        _log('[S45-A0] NOT OK')
        sys.exit(1)
    if stop:
        sys.exit(3)
    _log('[S45-A0] OK: all 8 A0 points certified')


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true', help='zero solves: freeze + validate the spec only')
    mode.add_argument('--run', action='store_true', help='evaluate the frozen spec named by --spec-sha256')
    parser.add_argument('--spec-sha256', default=None)
    args = parser.parse_args()
    started = time.time()
    if args.freeze:
        if args.spec_sha256:
            parser.error('--spec-sha256 is for --run only')
        freeze(started)
    else:
        if not args.spec_sha256:
            parser.error('--run requires --spec-sha256')
        run(started, args.spec_sha256)


if __name__ == '__main__':
    main()
