"""
P5.15 Addendum 25 item 3 -- configuration-selection run, AA arm at the three
non-C* candidates, through the campaign harness (three concurrent evaluations).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 25; frozen spec v14
`data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`,
`item3_selection_run` (adoption rule, stop rule, per-candidate report).

The chosen AA arm (spec v14 item-2 choice rule; `campaign_s44_aa_variant`,
d178c5c6): anderson_acceleration {enabled: True, reject_policy: 'keep_memory'}
(memory 5, Tikhonov 1e-10 -- frozen by the harness). It certified at C* in 107
cycles (D: 139), passing (b)-(d).

FULL campaign (Planner launches; cap 500, concurrency 3), campaign id
`s44_selection_aa`, root `data/SRP1/Results/P515S44/campaign_s44_selection_aa/`:
  1. paper_plan_aa_keep_memory  : node 7 only, 1.62 MVA / 3.24 MWh; nodes 5, 9 zero
  2. node7_empty_aa_keep_memory : 0.96875 MVA / 3.875 MWh at nodes 5 and 9; node 7 zero
  3. two_c_star_aa_keep_memory  : 1.9375 MVA / 7.75 MWh at nodes 5, 7, 9
Each requests the post-certification step against the committed D evaluation of
its OWN candidate: (b) |Q - Q_ref| <= 1.5e-4 Q_ref; (c) decomposition vs the
reference (residual <= 1.0, other priced components identically 0); persisted
certified models; (d) hull polish (all 48 blocks solve; < 0.1 %). The harness
adds the AA per-cycle sidecar and peak RSS to every record.

D references (certified, case-file configuration, hash-pinned below):
  paper_plan  : campaign_s44_gate/evals/d1a02e67107d0bac_paper_plan   (139; 653,029,766.99)
  node7_empty : campaign_s44_gate/evals/e30704e6e4dd3765_node7_empty  (136; 651,900,014.16)
  two_c_star  : campaign_s44_aa_variant/evals/4e53fa5560bbfa10_two_c_star_d (187; 648,138,276.77)
C* (not re-run; committed records, hash-pinned):
  D  : campaign_s44_gate/evals/578636daa6d6360d_c_star (139)
  AA : campaign_s44_aa_variant/evals/837fc982565dbba3_c_star_aa_keep_memory (107)

`campaign_results.json` carries the spec v14 item-3 ADOPTION INPUTS across all
four candidates (per candidate: D's and the AA arm's certification cycle, AA < D,
(b)-(d) for the AA arm, certified cost of both arms, per-node EFC/day and
terminal SoH for both arms) and the mechanical adoption reading: AA adopted ONLY
IF it certifies in strictly fewer cycles than D at all four candidates AND
passes (b)-(d) at each; otherwise D. The ruling is the Planner's.

DRY RUN (`--dry-run`; zero solves; the Worker runs it): the campaign
preconditions are checked against the REAL campaign root (which must not
exist); the four D references and the AA C* record are verified against their
pinned sha256 and their canonical candidate / key against this script's
candidates; the harness's own `validate_overrides` / `resolve_post_certification`
are exercised through `freeze_campaign_spec`, which freezes the spec into the
separate evidence directory `data/SRP1/Results/P515S44/selection_aa_dry_run/`
(extra.mode = 'dry_run', so its sha256 differs from the real spec's). No
campaign lock is taken, `evaluate` is never called, and the real campaign root
and lock are asserted absent at the end.

The parent never solves: SolveProfileGuard(permitted=()) is installed before
any model import and verified at exactly 0 (both modes).

EXACT LAUNCH COMMANDS (repo root; attached; both streams captured; never detached):
  dry run (Worker):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_selection_aa_campaign.py --dry-run \\
        > data/SRP1/Results/P515S44/selection_aa_dry_run_launch.log 2>&1
  full (Planner):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_selection_aa_campaign.py \\
        > data/SRP1/Results/P515S44/campaign_s44_selection_aa_launch.log 2>&1
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

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S44 selection AA campaign parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

CAMPAIGN_ID = 's44_selection_aa'
CAMPAIGN_ROOT = os.path.join(H.RESULTS_ROOT, f'campaign_{CAMPAIGN_ID}')
DRY_RUN_ROOT = os.path.join(H.RESULTS_ROOT, 'selection_aa_dry_run')
CAP = 500
CONCURRENCY = 3
AA_KEEP = {'anderson_acceleration': {'enabled': True, 'reject_policy': 'keep_memory'}}
_P = os.path.join('data', 'SRP1', 'Results', 'P515S44')

# Candidates (spec v14 item3_selection_run.candidates); C* is not re-run here.
CANDIDATES = {
    'C_star': {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)},
    'paper_plan': {5: (0.0, 0.0), 7: (1.62, 3.24), 9: (0.0, 0.0)},
    'node7_empty': {5: (0.96875, 3.875), 7: (0.0, 0.0), 9: (0.96875, 3.875)},
    'two_c_star': {5: (1.9375, 7.75), 7: (1.9375, 7.75), 9: (1.9375, 7.75)},
}
CANDIDATE_ORDER = ('C_star', 'paper_plan', 'node7_empty', 'two_c_star')
NEW_EVALUATIONS = (('paper_plan', 'paper_plan_aa_keep_memory'),
                   ('node7_empty', 'node7_empty_aa_keep_memory'),
                   ('two_c_star', 'two_c_star_aa_keep_memory'))

# Committed D evaluations (case-file configuration), one per candidate. sha256 pinned from the
# committed campaign manifests (campaign_s44_gate, campaign_s44_aa_variant); cycles/costs as committed.
D_REFERENCES = {
    'C_star': {'eval_dir': os.path.join(_P, 'campaign_s44_gate', 'evals', '578636daa6d6360d_c_star'),
               'evaluation_record_sha256': 'f7722fc44a2785f1d085dbc2f72703bc0962d06cf9a3796dbd4bce89d9ed7870',
               'component_levels_terminal_sha256': 'ff9050bb1759cff14f9d4e94d7e095b7a9b7fc3fa7ca8439f2a9ed5c0aa0444d',
               'certification_cycle': 139, 'certified_cost': 650966975.2943751},
    'paper_plan': {'eval_dir': os.path.join(_P, 'campaign_s44_gate', 'evals', 'd1a02e67107d0bac_paper_plan'),
                   'evaluation_record_sha256': '36d9bf6bda46de65450060d15262362b4372bbabcaf1f13c80fc135de31e5da5',
                   'component_levels_terminal_sha256':
                       '6e5d2fccfaf751a22feded918e176ad2f960702c2d705ce94838bda341573319',
                   'certification_cycle': 139, 'certified_cost': 653029766.9858261},
    'node7_empty': {'eval_dir': os.path.join(_P, 'campaign_s44_gate', 'evals', 'e30704e6e4dd3765_node7_empty'),
                    'evaluation_record_sha256': '02f5a1b2ff7818afaeee2dc200e6d21d784a7959c3563bca2cf02d526f7ed7a4',
                    'component_levels_terminal_sha256':
                        'f4e74af099b4940928fd2c024800804ad9fc0cba57da40aab76d2d7dd53c28d4',
                    'certification_cycle': 136, 'certified_cost': 651900014.159667},
    'two_c_star': {'eval_dir': os.path.join(_P, 'campaign_s44_aa_variant', 'evals', '4e53fa5560bbfa10_two_c_star_d'),
                   'evaluation_record_sha256': '44c9538537a25076894d728b08a8052267d92290cbdfc579e9a40bac2263da2e',
                   'component_levels_terminal_sha256':
                       '53817e7703aa9d4c15d6b4973f1a9e2b5bb531c7e9da171665000fd5c776d8de',
                   'certification_cycle': 187, 'certified_cost': 648138276.7714801},
}
# The committed AA keep_memory evaluation at C* (campaign_s44_aa_variant, d178c5c6).
AA_C_STAR = {'eval_dir': os.path.join(_P, 'campaign_s44_aa_variant', 'evals', '837fc982565dbba3_c_star_aa_keep_memory'),
             'evaluation_record_sha256': '6e90d4424df3405b0cc066bba1f57c28666a1524e5cdcc62890c6e6a74e69a4f',
             'component_levels_terminal_sha256': '8452f6261b1bf1fe84d4dfd8a177fa8f4145ade4c042a4904a29d4483a745782',
             'certification_cycle': 107, 'certified_cost': 650982939.9389359}

AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 25 (selection run)',
    'data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json item3_selection_run',
    'campaign_s44_aa_variant (d178c5c6): item-2 choice -> AA keep_memory variant',
]
ADOPTION_RULE = ('spec v14 item3: AA-on adopted ONLY IF it certifies in fewer cycles than D at all four '
                 'candidates AND passes (b)-(d) at each, band 1.5e-4 relative to D\'s cost at the SAME '
                 'candidate; otherwise D')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _load(rel):
    with open(os.path.join(REPO, rel)) as handle:
        return json.load(handle)


def _plan():
    return [(label, CANDIDATES[cand],
             {'overrides': AA_KEEP,
              'post_certification': {'persist_certified_models': True, 'hull_polish': True,
                                     'reference': {'eval_dir': D_REFERENCES[cand]['eval_dir']}}})
            for cand, label in NEW_EVALUATIONS]


def _verify_committed_record(name, pinned, cand, expect_overrides):
    """Pinned hashes, same candidate (canonical form AND key), certified, cycle/cost as committed,
    configuration as expected. Returns (check dict, problems list)."""
    problems = []
    rec_rel = os.path.join(pinned['eval_dir'], 'evaluation_record.json')
    cl_rel = os.path.join(pinned['eval_dir'], 'component_levels_terminal.json')
    got_rec = H.sha256_file(os.path.join(REPO, rec_rel)) if os.path.isfile(os.path.join(REPO, rec_rel)) else None
    got_cl = H.sha256_file(os.path.join(REPO, cl_rel)) if os.path.isfile(os.path.join(REPO, cl_rel)) else None
    if got_rec != pinned['evaluation_record_sha256']:
        problems.append(f'{name}: evaluation_record sha256 {got_rec} != pinned {pinned["evaluation_record_sha256"]}')
    if got_cl != pinned['component_levels_terminal_sha256']:
        problems.append(f'{name}: component_levels sha256 {got_cl} != pinned '
                        f'{pinned["component_levels_terminal_sha256"]}')
    if problems:
        return {'name': name, 'eval_dir': pinned['eval_dir'], 'ok': False}, problems
    rec = _load(rec_rel)
    cl = _load(cl_rel)
    canon = H.canonical_candidate(CANDIDATES[cand])
    key = H.candidate_key(canon)
    overrides = (rec.get('evaluation_overrides_effective') if 'evaluation_overrides_effective' in rec
                 else (rec.get('configuration') or {}).get('overrides')) or {}
    checks = {
        'canonical_equal': rec.get('candidate_canonical') == canon,
        'candidate_key_equal': rec.get('candidate_key') == key,
        'record_key_recomputed_from_its_canonical': H.candidate_key(rec.get('candidate_canonical')) == rec.get(
            'candidate_key'),
        'certified': rec.get('status') == 'certified',
        'certification_cycle_as_committed': rec.get('certification_cycle') == pinned['certification_cycle'],
        'certified_cost_as_committed': rec.get('certified_cost') == pinned['certified_cost'],
        'component_levels_gross_equals_certified_cost':
            (cl.get('recourse_components') or {}).get('gross_operational_cost') == rec.get('certified_cost'),
        'configuration_as_expected': overrides == expect_overrides,
    }
    for k, v in checks.items():
        if not v:
            problems.append(f'{name}: check {k} failed')
    return {'name': name, 'eval_dir': pinned['eval_dir'], 'candidate': cand, 'candidate_key': key,
            'record_candidate_key': rec.get('candidate_key'), 'record_schema': rec.get('schema'),
            'certification_cycle': rec.get('certification_cycle'), 'certified_cost_gross': rec.get('certified_cost'),
            'overrides': overrides, 'evaluation_record_sha256': got_rec, 'component_levels_terminal_sha256': got_cl,
            'checks': checks, 'ok': not problems}, problems


def verify_references():
    out, problems = {}, []
    for cand in CANDIDATE_ORDER:
        res, p = _verify_committed_record(f'D@{cand}', D_REFERENCES[cand], cand, {})
        out[f'D@{cand}'] = res
        problems += p
    res, p = _verify_committed_record('AA@C_star', AA_C_STAR, 'C_star', AA_KEEP)
    out['AA@C_star'] = res
    problems += p
    return out, problems


def _storage(record):
    out = {}
    for node, s in ((record or {}).get('storage_per_node') or {}).items():
        out[node] = {k: s.get(k) for k in ('s_mva', 'e_mwh', 'has_storage', 'efc_per_day_max',
                                           'efc_per_day_per_cohort_year', 'terminal_soh_per_active_cohort_year',
                                           'terminal_soh_min_over_active_cohort_years')}
    return out


def _gates_bcd(record):
    pc = (record or {}).get('post_certification') or {}
    return {'post_certification_status': pc.get('status'),
            'b': (pc.get('gate_b') or {}).get('pass'), 'c': pc.get('gate_c_pass'),
            'd': (pc.get('gate_d') or {}).get('pass'),
            'gate_b': pc.get('gate_b'), 'gate_c': pc.get('gate_c'), 'gate_d': pc.get('gate_d')}


def adoption_inputs(d_records, aa_records):
    """Spec v14 item3 adoption inputs across ALL FOUR candidates, and the mechanical reading."""
    per = {}
    for cand in CANDIDATE_ORDER:
        d, aa = d_records.get(cand) or {}, aa_records.get(cand) or {}
        g = _gates_bcd(aa)
        d_cert, aa_cert = d.get('status') == 'certified', aa.get('status') == 'certified'
        d_cyc, aa_cyc = d.get('certification_cycle'), aa.get('certification_cycle')
        bcd = bool(g['post_certification_status'] == 'evaluated' and g['b'] is True and g['c'] is True
                   and g['d'] is True)
        per[cand] = {
            'candidate_key': d.get('candidate_key'), 'candidate_canonical': d.get('candidate_canonical'),
            'D': {'eval_dir': d.get('eval_dir'), 'status': d.get('status'), 'certification_cycle': d_cyc,
                  'certified_cost_gross': d.get('certified_cost'), 'storage_per_node': _storage(d),
                  'peak_rss': d.get('peak_rss'), 'wall_time_s': d.get('wall_time_s')},
            'AA': {'eval_dir': aa.get('eval_dir'), 'eval_key': aa.get('eval_key'), 'status': aa.get('status'),
                   'certification_cycle': aa_cyc, 'certified_cost_gross': aa.get('certified_cost'),
                   'overrides': aa.get('evaluation_overrides_effective'), 'storage_per_node': _storage(aa),
                   'gates_b_c_d': g, 'aa_per_cycle_action_counts': (aa.get('aa_per_cycle') or {}).get('action_counts'),
                   'peak_rss': aa.get('peak_rss'), 'wall_time_s': aa.get('wall_time_s')},
            'aa_fewer_cycles_than_d': bool(d_cert and aa_cert and aa_cyc is not None and d_cyc is not None
                                           and aa_cyc < d_cyc),
            'aa_b_c_d_pass': bcd,
            'objective_convention': 'gross_operational_cost, settlement-excluded',
        }
    failing = [c for c in CANDIDATE_ORDER if not (per[c]['aa_fewer_cycles_than_d'] and per[c]['aa_b_c_d_pass'])]
    return {
        'rule': ADOPTION_RULE,
        'per_candidate': per,
        'candidates_failing_the_rule': failing,
        'mechanical_reading': 'AA adopted' if not failing else 'D (AA not adopted)',
        'note': "reported for the Planner; the ruling is the Planner's",
    }


def _write_manifest(root):
    manifest = {}
    for r_, _dirs, files in os.walk(root):
        for fname in sorted(files):
            fpath = os.path.join(r_, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    return manifest


def _freeze(root, mode, ref_check):
    return H.freeze_campaign_spec(
        root, CAMPAIGN_ID, _plan(),
        configuration={'name': 'per-evaluation: AA keep_memory (override) at the three non-C* candidates',
                       'arm_label': 's39_D', 'overrides': {},
                       'note': ('campaign-level overrides empty; each entry carries AA keep_memory; '
                                'num_max_iters := cap (run_admm_arm always sets it)')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY, required_consecutive_cycles=10,
        extra={'campaign_script': os.path.basename(__file__),
               'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'mode': mode,
               'd_references_pinned': D_REFERENCES, 'aa_c_star_pinned': AA_C_STAR,
               'reference_verification': ref_check,
               'adoption_rule': ADOPTION_RULE,
               'stop_rule': ('spec v14 item3: a candidate at which D fails to certify within cap 500 stops the run; '
                             'all four D evaluations are committed and certified (D is not re-run here)')})


def _spec_entry_checks(spec):
    expect_ov = H.validate_overrides(AA_KEEP)
    out = {}
    for cand, label in NEW_EVALUATIONS:
        e = next((x for x in spec['candidates'] if x['label'] == label), None)
        ref = (e or {}).get('post_certification') or {}
        r = ref.get('reference') or {}
        pinned = D_REFERENCES[cand]
        canon = H.canonical_candidate(CANDIDATES[cand])
        key = H.candidate_key(canon)
        ekey = H.evaluation_key(key, expect_ov)
        out[label] = {
            'entry_present': e is not None,
            'canonical_equal': (e or {}).get('canonical') == canon,
            'candidate_key_equal_reference': (e or {}).get('key') == key == r.get('candidate_key'),
            'overrides_pass_allow_list_and_equal': (e or {}).get('overrides') == expect_ov == AA_KEEP,
            'eval_key_is_override_key': (e or {}).get('eval_key') == ekey and ekey != key,
            'eval_dir': (e or {}).get('eval_dir'),
            'persist_and_polish_requested': ref.get('persist_certified_models') is True
            and ref.get('hull_polish') is True,
            'reference_eval_dir': r.get('eval_dir') == pinned['eval_dir'],
            'reference_hashes_equal_pinned': (r.get('evaluation_record_sha256') == pinned['evaluation_record_sha256']
                                              and r.get('component_levels_terminal_sha256')
                                              == pinned['component_levels_terminal_sha256']),
            'reference_certified_cost_equal': r.get('certified_cost') == pinned['certified_cost'],
            'reference_certification_cycle_equal': r.get('certification_cycle') == pinned['certification_cycle'],
            'reference_configuration_overrides_empty': r.get('configuration_overrides') == {},
            'reference_unchanged_child_side_check': H.verify_reference_unchanged(r) == {
                'evaluation_record_sha256': pinned['evaluation_record_sha256'],
                'component_levels_terminal_sha256': pinned['component_levels_terminal_sha256']} if r else False,
        }
    return out


def dry_run(started):
    real_root_before = os.path.exists(CAMPAIGN_ROOT)
    failures = H.check_campaign_preconditions(
        CAMPAIGN_ROOT, extra_clean_files=('p515_s44_selection_aa_campaign.py',))
    _log(f'[S44-SEL-AA DRY-RUN] campaign preconditions vs the REAL root: {failures or "all pass"}')
    if os.path.exists(DRY_RUN_ROOT):
        _log(f'[S44-SEL-AA DRY-RUN] refusing: dry-run evidence dir exists (write-once): {DRY_RUN_ROOT}')
        raise SystemExit(1)
    ref_check, ref_problems = verify_references()
    _log(f'[S44-SEL-AA DRY-RUN] reference verification problems: {ref_problems or "none"}')
    for name, r in ref_check.items():
        _log(f'[S44-SEL-AA DRY-RUN]   {name}: ok={r["ok"]} key={str(r.get("candidate_key"))[:16]} '
             f'cycle={r.get("certification_cycle")} cost={r.get("certified_cost_gross")} overrides={r.get("overrides")}')
    spec_path, spec_sha, spec = _freeze(DRY_RUN_ROOT, 'dry_run', ref_check)
    _log(f'[S44-SEL-AA DRY-RUN] frozen (dry-run) spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    entry_checks = _spec_entry_checks(spec)
    for label, c in entry_checks.items():
        _log(f'[S44-SEL-AA DRY-RUN]   {label}: eval_dir={c["eval_dir"]} all_checks='
             f'{all(v for k, v in c.items() if k != "eval_dir")}')
    # adoption computation exercised on the committed C* pair only (the three AA records do not exist yet)
    d_records = {c: _load(os.path.join(D_REFERENCES[c]['eval_dir'], 'evaluation_record.json'))
                 for c in CANDIDATE_ORDER}
    adoption_preview = adoption_inputs(d_records, {'C_star': _load(os.path.join(AA_C_STAR['eval_dir'],
                                                                                 'evaluation_record.json'))})
    guard_failures = PARENT_GUARD.verify(0)
    absent = {'real_campaign_root_absent': not os.path.exists(CAMPAIGN_ROOT) and not real_root_before,
              'campaign_lock_absent': not os.path.exists(H.CAMPAIGN_LOCK_PATH),
              'legacy_lock_absent': not os.path.exists(H.LEGACY_RUN_LOCK_PATH)}
    checks = {
        'campaign_preconditions_pass': not failures,
        'references_verified': not ref_problems,
        'spec_entries_verified': all(all(v for k, v in c.items() if k != 'eval_dir') for c in entry_checks.values()),
        'three_entries_concurrency_3_cap_500': (len(spec['candidates']) == 3 and spec['concurrency'] == CONCURRENCY
                                                and spec['cap'] == CAP),
        'parent_guard_zero_solves': not guard_failures,
        **absent,
    }
    results = {
        'stage': 'P5.15 Addendum 25 item 3 -- selection run AA arm: ZERO-SOLVE DRY RUN of the campaign script',
        'mode': 'dry_run', 'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'real_campaign_root': os.path.relpath(CAMPAIGN_ROOT, REPO),
        'dry_run_spec_path': os.path.relpath(spec_path, REPO), 'dry_run_spec_sha256': spec_sha,
        'dry_run_spec_note': ('same campaign id, candidates, overrides, post-certification requests, cap and '
                              'concurrency as the real launch; differs in frozen_utc, git_head and extra.mode, '
                              'so the real spec will have a different sha256'),
        'campaign_preconditions_failures': failures,
        'reference_verification': ref_check, 'reference_problems': ref_problems,
        'spec_entry_checks': entry_checks,
        'adoption_computation_preview_c_star_only': adoption_preview,
        'checks': checks,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(DRY_RUN_ROOT, 'dry_run_results.json'), results)
    H._write_once_json(os.path.join(DRY_RUN_ROOT, 'manifest_sha256.json'), _write_manifest(DRY_RUN_ROOT))
    PARENT_GUARD.uninstall()
    _log(f'[S44-SEL-AA DRY-RUN] checks: {checks}')
    ok = all(checks.values())
    _log(f'[S44-SEL-AA DRY-RUN] {"OK" if ok else "NOT OK"}')
    if not ok:
        sys.exit(1)


def full(started):
    failures = H.check_campaign_preconditions(
        CAMPAIGN_ROOT, extra_clean_files=('p515_s44_selection_aa_campaign.py',))
    ref_check, ref_problems = verify_references()
    failures += [f'reference: {p}' for p in ref_problems]
    if failures:
        for f in failures:
            _log(f'[S44-SEL-AA PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    _log(f'[S44-SEL-AA] preconditions and references passed (campaign {CAMPAIGN_ID}; cap {CAP}; '
         f'concurrency {CONCURRENCY})')
    spec_path, spec_sha, spec = _freeze(CAMPAIGN_ROOT, 'full', ref_check)
    _log(f'[S44-SEL-AA] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    entry_checks = _spec_entry_checks(spec)
    bad = [lab for lab, c in entry_checks.items() if not all(v for k, v in c.items() if k != 'eval_dir')]
    if bad:
        _log(f'[S44-SEL-AA PRECONDITION FAILED] spec entry checks failed for {bad}: {entry_checks}')
        raise SystemExit(1)
    lock = H.acquire_campaign_lock(CAMPAIGN_ID, spec_sha)
    _log(f'[S44-SEL-AA] campaign lock acquired: {lock}')
    try:
        ctx = H.CampaignContext(CAMPAIGN_ROOT, spec_path, spec_sha, spec, log=_log)
        records = H.evaluate([label for _c, label in NEW_EVALUATIONS], ctx)
        batch_info = getattr(H.evaluate, 'last_batch_info', {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log('[S44-SEL-AA] campaign lock released')

    by_label = {r.get('candidate_label'): r for r in records}
    # references re-verified after the run (they must not have changed while it ran)
    ref_after, ref_after_problems = verify_references()
    d_records = {c: _load(os.path.join(D_REFERENCES[c]['eval_dir'], 'evaluation_record.json'))
                 for c in CANDIDATE_ORDER}
    aa_records = {'C_star': _load(os.path.join(AA_C_STAR['eval_dir'], 'evaluation_record.json'))}
    for cand, label in NEW_EVALUATIONS:
        aa_records[cand] = by_label.get(label)
    adoption = adoption_inputs(d_records, aa_records)

    guard_failures = PARENT_GUARD.verify(0)
    results = {
        'stage': 'P5.15 Addendum 25 item 3 -- selection run: AA keep_memory arm at paper plan, node-7-empty, 2xC*',
        'mode': 'full', 'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha,
        'evaluations': {lab: by_label.get(lab) and {
            k: by_label[lab].get(k) for k in (
                'candidate_label', 'candidate_key', 'eval_key', 'eval_dir', 'evaluation_overrides_effective',
                'overrides_applied_in_child', 'status', 'barrier', 'barrier_cause', 'certification_cycle',
                'cycles_run', 'certified_cost', 'terminal_gross_operational_cost', 'bar',
                'first_pass_cycle_per_channel', 'terminal_ratios_per_channel', 'rule_ten', 'settlement_remainder',
                'recourse_components', 'storage_per_node', 'local_solve_failures', 'network_failures_summary',
                'post_certification', 'aa_per_cycle', 'peak_rss', 'wall_time_s', 'parent_view',
                'campaign_spec_sha256')} for _c, lab in NEW_EVALUATIONS},
        'objective_convention': 'gross_operational_cost (settlement-excluded) throughout',
        'adoption_inputs_spec_v14_item3': adoption,
        'stop_rule_spec_v14_item3': {'all_four_D_references_certified': all(
            (d_records[c] or {}).get('status') == 'certified' for c in CANDIDATE_ORDER), 'triggered': not all(
            (d_records[c] or {}).get('status') == 'certified' for c in CANDIDATE_ORDER)},
        'references_before_run': ref_check, 'references_after_run_problems': ref_after_problems,
        'references_after_run': ref_after,
        'batch_info': batch_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_results.json'), results)
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_manifest_sha256.json'), _write_manifest(CAMPAIGN_ROOT))
    PARENT_GUARD.uninstall()

    for cand in CANDIDATE_ORDER:
        p = adoption['per_candidate'][cand]
        g = p['AA']['gates_b_c_d']
        _log(f'[S44-SEL-AA] {cand}: D cycle={p["D"]["certification_cycle"]} AA status={p["AA"]["status"]} '
             f'cycle={p["AA"]["certification_cycle"]} AA<D={p["aa_fewer_cycles_than_d"]} '
             f'(b)={g["b"]} (c)={g["c"]} (d)={g["d"]} pc={g["post_certification_status"]} '
             f'cost D={p["D"]["certified_cost_gross"]} AA={p["AA"]["certified_cost_gross"]}')
    _log(f'[S44-SEL-AA] adoption (mechanical reading): {adoption["mechanical_reading"]}; '
         f'failing: {adoption["candidates_failing_the_rule"]}')
    _log(f'[S44-SEL-AA] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}')
    harness_errors = [r.get('candidate_label') for r in records
                      if r.get('status') not in ('certified', 'not_certified')
                      or (r.get('parent_view') or {}).get('exit_code') != 0]
    if harness_errors:
        _log(f'[S44-SEL-AA] evaluations with an error status or non-zero exit: {harness_errors}')
    if ref_after_problems:
        _log(f'[S44-SEL-AA] references changed during the run: {ref_after_problems}')
    ok = not guard_failures and not harness_errors and not ref_after_problems
    _log(f'[S44-SEL-AA] {"OK" if ok else "NOT OK"}')
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='zero-solve: preconditions, reference verification, spec frozen into '
                             'selection_aa_dry_run/; no lock, no evaluation')
    args = parser.parse_args()
    started = time.time()
    if args.dry_run:
        dry_run(started)
    else:
        full(started)


if __name__ == '__main__':
    main()
