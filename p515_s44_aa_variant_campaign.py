"""
P5.15 Addendum 25 item 2 -- the AA memory-retaining VARIANT arm at C*, plus
2 x C* under D, through the campaign harness (two concurrent evaluations).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 25; frozen spec v14
`data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`,
`item2_aa_variant` and `item3_selection_run`; `P5_15_S44_GATE_RULING.md`
(8ce0b872) "Deviation" (post-certification step) and "Follow-ups".

FULL campaign (Planner launches; cap 500, concurrency 2), campaign id
`s44_aa_variant`, root `data/SRP1/Results/P515S44/campaign_s44_aa_variant/`:
  1. c_star_aa_keep_memory : C* (0.96875 MVA / 3.875 MWh at nodes 5, 7, 9) with
     anderson_acceleration {enabled: True, reject_policy: 'keep_memory'};
     post-certification: gates (b)/(c) against the committed D evaluation of
     C* (`campaign_s44_gate/evals/578636daa6d6360d_c_star`, certified 139,
     650,966,975.2943751), persisted certified models, hull polish (gate (d)).
  2. two_c_star_d          : 2 x C* (1.9375 MVA / 7.75 MWh at nodes 5, 7, 9)
     under D (case file, no override); post-certification: persisted
     certified models and hull polish (no reference; D's own hull value).
SMOKE (`--smoke`; the Worker runs it; cap 3, concurrency 2), campaign id
`s44_aa_variant_smoke`:
  1. c_star_aa_keep_memory : as above, same post-certification request (at 3
     cycles nothing certifies, so the step must SKIP with a recorded reason);
  2. c_star_d              : C* under D, flag off, no post-certification.

Reported (not gates of this script -- the Planner rules): per evaluation the
STEP4 2.5 record summary, the post-certification outcome ((b) |Q-Q_ref| <=
1.5e-4 Q_ref, (c) residual <= 1.0 with other priced components identically 0,
(d) hull polish < 0.1 % with every block solved), the AA action summary and the
first cycle at which the AA-on trajectory departs from the committed Step 3.7
AA run (`data/SRP1/Results/P515S43/aa_run`, clear_memory policy; the two
policies are identical until the first safeguard rejection, cycle 14 there);
for the full campaign the spec v14 item-2 choice-rule inputs (variant vs Step
3.7 arm: 109 cycles) and the item-3 stop-rule flag for 2 x C* under D. In the
smoke additionally: each evaluation's first 3 rows vs its committed reference
(D: `P515S39_D_run` via `p515_s40_polish_gap._reproduction_check`; AA: the
committed Step 3.7 AA run's first 3 rows), reported only.

The parent never solves: SolveProfileGuard(permitted=()) is installed before
any model import and verified at exactly 0.

EXACT LAUNCH COMMANDS (repo root; attached; both streams captured; never detached):
  smoke:
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_aa_variant_campaign.py --smoke \\
        > data/SRP1/Results/P515S44/campaign_s44_aa_variant_smoke_launch.log 2>&1
  full (Planner):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_aa_variant_campaign.py \\
        > data/SRP1/Results/P515S44/campaign_s44_aa_variant_launch.log 2>&1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S44 AA-variant campaign parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- comparator (8f5cff48), BY IMPORT
import p515_s43_aa_flagoff_gate as FG  # noqa: E402 -- diff classification, BY IMPORT
from p515_s40_polish_gap import _reproduction_check  # noqa: E402 -- truncated D comparison, BY IMPORT

C_STAR = {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)}
TWO_C_STAR = {5: (1.9375, 7.75), 7: (1.9375, 7.75), 9: (1.9375, 7.75)}
AA_KEEP = {'anderson_acceleration': {'enabled': True, 'reject_policy': 'keep_memory'}}
D_CSTAR_EVAL_REL = os.path.join('data', 'SRP1', 'Results', 'P515S44', 'campaign_s44_gate', 'evals',
                                '578636daa6d6360d_c_star')
STEP37_AA_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S43', 'aa_run')
STEP37_CERTIFICATION_CYCLE = 109  # committed Step 3.7 AA arm (76095561; aa_run_results.json)
D_CSTAR_CERTIFICATION_CYCLE = 139
POST_AA = {'persist_certified_models': True, 'hull_polish': True, 'reference': {'eval_dir': D_CSTAR_EVAL_REL}}
POST_D = {'persist_certified_models': True, 'hull_polish': True}
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 25 (AA variant arm; selection run)',
    'data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json item2_aa_variant, item3_selection_run',
    'P5_15_S44_GATE_RULING.md (8ce0b872) Deviation / Follow-ups',
]
PREDICTIONS = {'source': 'frozen spec v14 predictions_recorded_in_advance.item2_variant',
               'expert': '80-95 cycles', 'planner': 'central 88, range 78-100; (b)-(d) pass'}


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _plan(smoke, suffix):
    if smoke:
        cid = f's44_aa_variant_smoke{suffix}'
        cands = [('c_star_aa_keep_memory', C_STAR, {'overrides': AA_KEEP, 'post_certification': POST_AA}),
                 ('c_star_d', C_STAR)]
        cap = 3
    else:
        cid = f's44_aa_variant{suffix}'
        cands = [('c_star_aa_keep_memory', C_STAR, {'overrides': AA_KEEP, 'post_certification': POST_AA}),
                 ('two_c_star_d', TWO_C_STAR, {'post_certification': POST_D})]
        cap = 500
    return cid, os.path.join(H.RESULTS_ROOT, f'campaign_{cid}'), cands, cap


def _summary(record):
    if record is None:
        return None
    return {
        'label': record.get('candidate_label'), 'candidate_key': (record.get('candidate_key') or '')[:16],
        'eval_key': (record.get('eval_key') or '')[:16], 'overrides': record.get('evaluation_overrides_effective'),
        'status': record.get('status'), 'barrier': record.get('barrier'), 'barrier_cause': record.get('barrier_cause'),
        'certification_cycle': record.get('certification_cycle'), 'cycles_run': record.get('cycles_run'),
        'certified_cost_gross': record.get('certified_cost'),
        'terminal_gross_operational_cost': record.get('terminal_gross_operational_cost'),
        'objective_convention': 'gross_operational_cost, settlement-excluded',
        'bar': (record.get('bar') or {}).get('value'),
        'first_pass_cycle_per_channel': record.get('first_pass_cycle_per_channel'),
        'terminal_ratios_per_channel': record.get('terminal_ratios_per_channel'),
        'rule_ten_terminal_step_over_threshold': (record.get('rule_ten') or {}).get('terminal_step_over_threshold'),
        'settlement_remainder': (record.get('settlement_remainder') or {}).get('value'),
        'local_solve_failures': record.get('local_solve_failures'),
        'network_failures_classes': (record.get('network_failures_summary') or {}).get('classes'),
        'post_certification': record.get('post_certification'),
        'aa_per_cycle': record.get('aa_per_cycle'),
        'peak_rss': record.get('peak_rss'), 'wall_time_s': record.get('wall_time_s'),
        'parent_view': record.get('parent_view'), 'campaign_spec_sha256': record.get('campaign_spec_sha256'),
    }


def _first_divergence(ref_rows, rows):
    """First cycle whose trajectory row differs (CP._diff conventions) from the reference row."""
    n = min(len(ref_rows), len(rows))
    for i in range(n):
        d = CP._diff(ref_rows[i], rows[i], f'cycle_trajectory[{i}]')
        if d:
            return {'first_differing_cycle': rows[i].get('cycle'), 'n_rows_compared': n,
                    'n_diffs_in_that_row': len(d), 'first_fields': [x['field'] for x in d[:12]]}
    return {'first_differing_cycle': None, 'n_rows_compared': n, 'identical_over_compared_rows': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    suffix = ('_' + args.suffix.strip('_')) if args.suffix.strip('_') else ''
    cid, root, cands, cap = _plan(args.smoke, suffix)
    started = time.time()

    failures = H.check_campaign_preconditions(
        root, extra_clean_files=('p515_s44_aa_variant_campaign.py', 'p515_s44_campaign_harness_checks.py'))
    for p in (os.path.join(REPO, D_CSTAR_EVAL_REL, 'evaluation_record.json'),
              os.path.join(STEP37_AA_DIR, 'g_s39_D.json')):
        if not os.path.isfile(p):
            failures.append(f'reference missing: {p}')
    if failures:
        for f in failures:
            _log(f'[S44-AAV-CAMPAIGN PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    _log(f'[S44-AAV-CAMPAIGN] preconditions passed ({"SMOKE" if args.smoke else "FULL"}; campaign {cid}; cap {cap})')

    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, cid, cands,
        configuration={'name': 'per-evaluation: AA keep_memory (override) and D (case file, fb3de341)',
                       'arm_label': 's39_D', 'overrides': {},
                       'note': ('campaign-level overrides empty; each entry carries its own (AA entry only); '
                                'num_max_iters := cap (run_admm_arm always sets it)')},
        cap=cap, concurrency=2, authority=AUTHORITY, required_consecutive_cycles=10,
        extra={'campaign_script': os.path.basename(__file__),
               'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'mode': 'smoke' if args.smoke else 'full',
               'predictions_recorded_in_advance': PREDICTIONS,
               'step37_aa_reference': {'dir': os.path.relpath(STEP37_AA_DIR, REPO),
                                       'certification_cycle': STEP37_CERTIFICATION_CYCLE,
                                       'g_s39_D_sha256': H.sha256_file(os.path.join(STEP37_AA_DIR, 'g_s39_D.json'))},
               'choice_rule': 'spec v14 item2: fewer certification cycles among arms passing (b)-(d); tie -> Step 3.7 arm',
               'stop_rule': 'spec v14 item3: a candidate at which D fails to certify within cap 500 stops the run'})
    _log(f'[S44-AAV-CAMPAIGN] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    lock = H.acquire_campaign_lock(cid, spec_sha)
    _log(f'[S44-AAV-CAMPAIGN] campaign lock acquired: {lock}')
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha, spec, log=_log)
        records = H.evaluate([c[0] for c in cands], ctx)
        batch_info = getattr(H.evaluate, 'last_batch_info', {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log('[S44-AAV-CAMPAIGN] campaign lock released')

    by_label = {r.get('candidate_label'): r for r in records}
    entries = {e['label']: e for e in spec['candidates']}
    eval_dirs = {lab: os.path.join(root, 'evals', e['eval_dir']) for lab, e in entries.items()}
    summaries = {lab: _summary(by_label.get(lab)) for lab in entries}

    # the AA-on trajectory vs the committed Step 3.7 AA run (reported)
    step37_rows = CP._load_json(os.path.join(STEP37_AA_DIR, 'g_s39_D.json'))['cycle_trajectory']
    aa_report = CP._load_json(os.path.join(eval_dirs['c_star_aa_keep_memory'], 'g_s39_D.json')) or {}
    aa_rows = aa_report.get('cycle_trajectory') or []
    divergence_vs_step37 = _first_divergence(step37_rows, aa_rows)

    extra_checks = {}
    if args.smoke:
        d_report = CP._load_json(os.path.join(eval_dirs['c_star_d'], 'g_s39_D.json')) or {}
        d_repro = _reproduction_check(d_report) if d_report else {'diffs': [], 'error': 'no report'}
        _p, d_aa_new, d_tie, d_genuine = FG._classify_diffs(d_repro.get('diffs', []))
        aa_rec = by_label.get('c_star_aa_keep_memory') or {}
        d_rec = by_label.get('c_star_d') or {}
        aa_pc = aa_rec.get('post_certification') or {}
        aa_dir = eval_dirs['c_star_aa_keep_memory']
        sidecar = FG._load_jsonl(os.path.join(aa_dir, H.AA_SIDECAR_FILE)) or []
        extra_checks = {
            'both_records_present_not_certified_at_cap_3': all(
                (by_label.get(l) or {}).get('status') == 'not_certified' for l in entries),
            'aa_post_certification_skipped_with_reason': aa_pc.get('status') == 'skipped'
            and bool(aa_pc.get('skip_reason')) and 'not certified' in aa_pc.get('skip_reason', ''),
            'aa_post_certification_file_written': os.path.isfile(os.path.join(aa_dir, H.POST_CERTIFICATION_FILE)),
            'aa_no_persisted_models_no_hull_detail': not os.path.exists(os.path.join(aa_dir, 'certified_models.pkl'))
            and not os.path.exists(os.path.join(aa_dir, H.HULL_BOUND_DETAIL_FILE)),
            'aa_settings_in_force_keep_memory': (aa_rec.get('overrides_applied_in_child') or {}).get(
                'anderson_acceleration') == {'enabled': True, 'memory': 5, 'regularization': 1e-10,
                                             'reject_policy': 'keep_memory'},
            'aa_sidecar_3_rows_aa_enabled': len(sidecar) == 3 and all(r.get('aa_enabled') is True for r in sidecar),
            'd_no_aa_sidecar_no_post_cert': not os.path.exists(os.path.join(eval_dirs['c_star_d'], H.AA_SIDECAR_FILE))
            and d_rec.get('post_certification') is None and d_rec.get('evaluation_overrides_effective') == {},
            'd_first_3_rows_vs_D_committed_zero_genuine': not d_genuine and d_repro.get('n_cycles_compared') == 3,
            'aa_first_3_rows_identical_to_step37_aa_run': divergence_vs_step37.get('first_differing_cycle') is None
            and divergence_vs_step37.get('n_rows_compared') == 3,
            'spec_sha256_in_every_record': all(r.get('campaign_spec_sha256') == spec_sha for r in records),
        }
        extra_checks_detail = {
            'aa_sidecar_actions': [(r.get('cycle'), r.get('aa_action'), r.get('aa_memory_size_before'),
                                    r.get('aa_memory_size_after'), r.get('aa_rho_changed_channels')) for r in sidecar],
            'aa_accept_or_reject_present': any(isinstance(r.get('aa_action'), str) and
                                               (r['aa_action'] == 'accepted' or r['aa_action'].startswith('rejected'))
                                               for r in sidecar),
            'd_vs_D_committed': {'mode': d_repro.get('mode'), 'n_aa_new_field_diffs': len(d_aa_new),
                                 'n_known_tie_break_diffs': len(d_tie), 'genuine_diffs': d_genuine},
        }
    else:
        extra_checks_detail = {}

    aa_s = summaries.get('c_star_aa_keep_memory') or {}
    aa_pc = aa_s.get('post_certification') or {}
    gates_bcd = {'b': (aa_pc.get('gate_b') or {}).get('pass'), 'c': aa_pc.get('gate_c_pass'),
                 'd': (aa_pc.get('gate_d') or {}).get('pass')}
    choice_rule_inputs = {
        'variant_certified': aa_s.get('status') == 'certified',
        'variant_certification_cycle': aa_s.get('certification_cycle'),
        'step37_arm_certification_cycle': STEP37_CERTIFICATION_CYCLE,
        'D_certification_cycle_at_c_star': D_CSTAR_CERTIFICATION_CYCLE,
        'variant_gates_b_c_d': gates_bcd,
        'mechanical_reading': (None if args.smoke else (
            'variant goes forward' if (aa_s.get('status') == 'certified' and all(v is True for v in gates_bcd.values())
                                       and (aa_s.get('certification_cycle') or 10 ** 9) < STEP37_CERTIFICATION_CYCLE)
            else 'Step 3.7 arm goes forward (variant not strictly fewer cycles with (b)-(d) passing)')),
        'note': 'reported for the Planner; the ruling is the Planner\'s',
    }
    stop_rule = None
    if not args.smoke:
        two = summaries.get('two_c_star_d') or {}
        stop_rule = {'two_c_star_d_certified': two.get('status') == 'certified',
                     'spec_v14_item3_stop_rule_triggered': two.get('status') != 'certified'}

    guard_failures = PARENT_GUARD.verify(0)
    results = {
        'stage': 'P5.15 Addendum 25 item 2 -- AA keep_memory variant arm at C* + 2xC* under D (campaign harness)',
        'mode': 'smoke' if args.smoke else 'full', 'authority': AUTHORITY,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha,
        'evaluations': summaries,
        'aa_trajectory_vs_committed_step37_aa_run': divergence_vs_step37,
        'choice_rule_inputs_spec_v14_item2': choice_rule_inputs,
        'stop_rule_spec_v14_item3': stop_rule,
        'predictions_recorded_in_advance': PREDICTIONS,
        'smoke_checks': extra_checks, 'smoke_detail': extra_checks_detail,
        'batch_info': batch_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(root, 'campaign_results.json'), results)
    manifest = {}
    for r_, _dirs, files in os.walk(root):
        for fname in sorted(files):
            fpath = os.path.join(r_, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(root, 'campaign_manifest_sha256.json'), manifest)
    PARENT_GUARD.uninstall()

    for lab, s in summaries.items():
        _log(f'[S44-AAV-CAMPAIGN] {lab}: status={s and s["status"]} cycles={s and s["cycles_run"]} '
             f'cert_cycle={s and s["certification_cycle"]} cost={s and s["certified_cost_gross"]} '
             f'post_cert={s and (s["post_certification"] or {}).get("status")} '
             f'reason={s and (s["post_certification"] or {}).get("skip_reason")}')
        if s and s.get('aa_per_cycle'):
            _log(f"[S44-AAV-CAMPAIGN]   aa actions: {s['aa_per_cycle'].get('action_counts')} "
                 f"retained-on-reject: {len(s['aa_per_cycle'].get('rejections_with_memory_retained') or [])}")
    _log(f'[S44-AAV-CAMPAIGN] AA trajectory vs committed Step 3.7 AA run: {divergence_vs_step37}')
    _log(f'[S44-AAV-CAMPAIGN] choice-rule inputs: {choice_rule_inputs}')
    if stop_rule and stop_rule['spec_v14_item3_stop_rule_triggered']:
        _log('[S44-AAV-CAMPAIGN] ****** STOP RULE (spec v14 item3): D did NOT certify at 2 x C* within cap 500 ******')
    if args.smoke:
        _log(f'[S44-AAV-CAMPAIGN] smoke checks: {extra_checks}')
        _log(f"[S44-AAV-CAMPAIGN] smoke AA sidecar: {extra_checks_detail.get('aa_sidecar_actions')}")
    _log(f'[S44-AAV-CAMPAIGN] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}')
    harness_errors = [r.get('candidate_label') for r in records
                      if r.get('status') not in ('certified', 'not_certified')
                      or (r.get('parent_view') or {}).get('exit_code') != 0]
    if harness_errors:
        _log(f'[S44-AAV-CAMPAIGN] evaluations with an error status or non-zero exit: {harness_errors}')
    ok = (not guard_failures and not harness_errors and (all(extra_checks.values()) if args.smoke else True))
    _log(f'[S44-AAV-CAMPAIGN] {"OK" if ok else "NOT OK"}')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
