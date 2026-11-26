"""
P5.15 Addendum 46 ruling 7, Planner task W86 -- the tight-tail SRP1 RE-CERTIFICATION (C*, the smallest node-7 unit,
x = 0) through the campaign harness, and its one-cell CHILD SMOKE. Frozen stage spec v29 (predecessor v28 c8b5a715).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 46 (ruling 7: convergence depth, minimal (c); "tight-tail SRP1
re-certification -> s47 identity look -> 3 x 3 pair"); frozen spec v28 (the tail, its fallback rule, the predictions);
Planner task W86 (rulings Q1 / Q2 on W85; steps 2 and 3).

THE PATH. Every evaluation runs through the production campaign harness (`p515_s44_campaign_harness`, BY IMPORT):
`H.evaluate` spawns one fresh interpreter per evaluation (`--child`, `main_child` -> `_child_real` ->
`p515_g_g1_g4_admm_gates.run_admm_arm`), so the smoke exercises exactly what the re-certification will run -- the
child's rule-eleven checklists, the tail declaration applied and read back, the per-round append (W85), the end-of-run
capture, `reconcile`, `convergence_depth_tail_state_check`, the post-certification request, the record. THIS process
never solves: `SolveProfileGuard(permitted=())` is armed at import and verified at exactly 0 in every mode.

THE SOLVE CLAIM IS RECONCILED PER EVENT, NOT GUARD-VERIFIED. The child runs in its own process; this parent cannot arm
or verify a guard there. Inside the child, `run_admm_arm` arms its own `SolveProfileGuard(N.PERMITTED)` around the
ADMM run (an undeclared call site raises), and the record's `solve_profile` reconciles that guard's count per event:
observed == (1 + n_dso) x n_years x n_days + n_esso per round x (cycles_run + 1) + every retry attempted
(`identity_holds`, `reconciliation_supported`). Nothing verifies an exact count declared in advance against a guard
the parent armed, and the child's pre-run and post-run phases (floor-row precheck, post-run hook, post-certification
persist) lie outside that guard's window. The claim is therefore stated as "reconciled per event", never as
"guard-verified". Independent cross-check from the capture: network floor records == observed - n_esso x rounds.

STAGES / MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --stage smoke --freeze                      ZERO SOLVES. Freezes the smoke campaign spec (campaign id
                                              `s53_w86_tail_smoke`, fresh root): ONE cell (c_star), cap 2,
                                              concurrency 1, the re-certification configuration exactly (AA + ESS
                                              ageing declarations, tail declared {True, 1e-6}, persist_certified_models).
  --stage smoke --run --spec-sha256 S         THE CHILD SMOKE: one evaluation through `H.evaluate`, then the smoke
                                              checks S1-S12 below -> <smoke root>/smoke_gate.json + manifest.
  --freeze-spec                               ZERO SOLVES. Writes frozen spec v29 (write-once, named by its sha256;
                                              predecessor v28, NOT edited): cells, configuration, the three expected
                                              eval keys, references, declared solve profile, per-entry gates,
                                              comparison formulas, fallback test, predictions, estimates. Requires the
                                              smoke gate committed and PASS (pinned in v29).
  --stage recert --freeze                     ZERO SOLVES. Freezes the re-certification campaign spec (campaign id
                                              `s53_w86_tail_recert`, fresh root), pinning v29, and runs the pre-launch
                                              assertion (the three eval keys EXACTLY v29's).
  --stage recert --run --spec-sha256 S        THE RE-CERTIFICATION (NOT RUN IN W86): pre-launch assertions again,
                                              memory preflight (refusing), campaign lock, the three cells at
                                              concurrency 3 (one batch), per-entry gates, comparison, fallback test ->
                                              <recert root>/campaign_results.json + manifest.

SMOKE GATE (declared here and in the smoke spec BEFORE the smoke runs; all must hold):
  S1  child exit 0; evaluation_record.json written by the child (no parent barrier record); status not_certified
      (cap 2); cycles_run 2;
  S2  the per-round append file is written and reconciles: the record's `convergence_depth_per_round_append.ok` True
      (appended file byte-identical to the end-of-run file, tail state rebuilt from the events == returned state), AND,
      recomputed here, the two files byte-identical, their sha256 == the record's end-of-run sha256, rounds appended
      [0, 1, 2], n appended == n end-of-run;
  S3  the tail checklist is line 1 of the events file (event 'checklist', == the record's checklist, tail enabled,
      declared {True, 1e-6}, pid == the child's) and was written BEFORE ANY SOLVE: its utc precedes the creation time
      (st_birthtime) of every IPOPT output file in the run working dir (>= 1 file) and the first 'drained' event;
  S4  `convergence_depth_tail_state_check.match` True (expected enabled True, in state True, tail value 1e-6);
  S5  solve profile reconciled per event: reconciliation_supported, base_solves == 51 x 3 = 153 (DECLARED: 51 per round
      x (cap 2 + 1)), identity_holds (observed == 153 + retries attempted), 0 blocked; network floor records ==
      observed - 3 ESSO x 3 rounds;
  S6  the tail is correctly INACTIVE at cap 2: 2 per-cycle tail records, none active, none acted, the AA-off predicate
      False at the end of both cycles, restore at exit not acted; events: 'apply' x3 (cycles 1, 2, exit) all
      inactive / not acted, 'next_state' x2 both False; no trajectory row with cycle_convergence True; no
      "Convergence-depth tail ON" line in the child's stdout;
  S7  the floor records are captured at PRODUCTION compl_inf_tol: every record's compl_inf_tol_in_force == its
      network's production value from the tail baseline (TSO case9 5e-4 passed; DSO key absent -> IPOPT default
      1e-4), parse_reason None, options_list_agrees True;
  S8  the append was sealed after the reconciliation (last event 'sealed'), no append write error;
  S9  post-certification: persist requested, trajectory uncertified -> status 'skipped', no certified_models.pkl;
  S10 the entry's eval key is EXACTLY the expected tail-ON C* key 96c5aa50...; fresh campaign id / root / working dirs;
  S11 ESS ageing read-back all_match before and after the run;
  S12 this process solved nothing: PARENT_GUARD.verify(0) == [].
  Reported, not gated: floor-status tally, records per round, peak RSS, wall.

EXACT COMMANDS (repo root):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
      p515_s53_w86_tail_recert_campaign.py --stage smoke --freeze \\
      > data/SRP1/Results/P515S53/tight_tail_w86/smoke_freeze_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
      p515_s53_w86_tail_recert_campaign.py --stage smoke --run --spec-sha256 <sha> \\
      > data/SRP1/Results/P515S53/tight_tail_w86/smoke_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
      p515_s53_w86_tail_recert_campaign.py --freeze-spec \\
      > data/SRP1/Results/P515S53/tight_tail_w86/freeze_spec_v29_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
      p515_s53_w86_tail_recert_campaign.py --stage recert --freeze \\
      > data/SRP1/Results/P515S53/tight_tail_w86/recert_freeze_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
      p515_s53_w86_tail_recert_campaign.py --stage recert --run --spec-sha256 <sha> \\
      > data/SRP1/Results/P515S53/tight_tail_w86/recert_launch.log 2>&1
Exit codes (--run): smoke 0 PASS / 1 FAIL; recert 0 every per-entry gate holds and no fallback, 2 every gate holds and
the fallback test fires (harness clean), 1 a gate / harness / guard / precondition failure.
"""

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W86 tail re-certification launcher (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402 -- stdlib-only at import (no model code)

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRING = 'p515_s53_w86_tail_recert_campaign'
STAGE_TEXT = ('P5.15 Addendum 46 ruling 7, W86 -- tight-tail SRP1 re-certification (C*, n7_4h_e1, x = 0) through the '
              'campaign harness, tail declared {enabled True, compl_inf_tol 1e-6}')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W86_ROOT_REL = os.path.join(_P53, 'tight_tail_w86')
SPEC_V28 = {'path': os.path.join(_P53, 'frozen_s53_spec_v28_c8b5a715.json'),
            'sha256': 'c8b5a715083c47cdaa1d5af0f24259115682542d159a206365ccf7f76c50edde'}
SPEC_V29_PREFIX = 'frozen_s53_spec_v29_'
LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
YEAR = 2025
ACTIVE_NODES = (5, 7, 9)
ARM_LABEL = 's39_D'
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
ESS_AGEING_BASELINE = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
                       'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
                       'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8,
                                       'eol_retention_r': 0.8}}
ESS_PARAMS_SHA256 = '39106f934bf3edbf18f01a5ef1fadfefc2f7a518706e6c8fa6d962617a312706'
TAIL = {'enabled': True, 'compl_inf_tol': 1e-6}
POST_CERTIFICATION = {'persist_certified_models': True, 'hull_polish': False}
REQUIRED_CONSECUTIVE_CYCLES = 10
IPOPT_DEFAULT_COMPL_INF_TOL = 1e-4
N_ESSO = 3
SOLVES_PER_ROUND = 51   # (1 + 3 DSO) x 3 years x 4 days + 3 ESSO; the child's record derives it from the instance

CELLS = {
    'c_star': {n: (0.96875, 3.875) for n in ACTIVE_NODES},
    'n7_4h_e1': {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)},
    'x0': {n: (0.0, 0.0) for n in ACTIVE_NODES},
}
# The pre-launch assertion (Planner, W86 step 3): the three frozen eval keys are EXACTLY these (W85 table; W86 step 1
# K4 re-derived them after the declared-off fix). x0's is the C2-baseline x = 0 key WITH the tail -- a new evaluation,
# not the pinned A0 / s48 record.
EXPECTED_EVAL_KEYS = {
    'c_star': '96c5aa50cc229cc14f8bfd34de306b7cc79caf77ac28a09e0046ad3978c49772',
    'n7_4h_e1': 'ca8927e75d628bd188cf74f9d81436cbb3e0be53f64e02d9876ebb271a21951d',
    'x0': '5cfe69a615ae370835a5896eaa6899f1d9973b7adf649437c2ddd94d1d2836ee',
}
PRE_TAIL_EVAL_KEYS = {
    'c_star': '070f833e1e318f8500f51a85c05993425e11081db6d17065e1e60463db26e6cb',
    'n7_4h_e1': 'bd504ecf5a288d447e53ef6f7e8090b017d15549b257aa095ff4f95890a06e42',
    'x0': 'd2c96b1480402a3b61aca4abc188e41c6009eb582d8e6ccdd380e51651f996c7',
}
W86_KEY_CHECKS = os.path.join(W86_ROOT_REL, 'key_fix_checks', 'checks_w86_key_r2.json')

# The committed references (pre-tail, same declarations). x0: the C2-declared s48 x0_capture record (d2c96b14), which
# reproduced the pinned A0 x0 record (a0_c7 7aa017f0) bitwise -- both pinned, the equality re-checked at freeze.
REFERENCES = {
    'c_star': {'eval_dir': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert', 'evals',
                                        '070f833e1e318f85_c_star'),
               'campaign': 's47_recert (spec 902f93aa)'},
    'n7_4h_e1': {'eval_dir': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert', 'evals',
                                          'bd504ecf5a288d44_n7_4h_e1'),
                 'campaign': 's47_recert (spec 902f93aa)'},
    'x0': {'eval_dir': os.path.join('data', 'SRP1', 'Results', 'P515S48', 'x0_capture', 'evals', 'd2c96b1480402a3b_x0'),
           'campaign': 's48_x0_capture (spec 4a50c0e2), C2-baseline declaration',
           'cross_check_eval_dir': os.path.join('data', 'SRP1', 'Results', 'P515S45', 'campaign_s45_a0_c7', 'evals',
                                                '7aa017f09989b56d_x0')},
}
REFERENCE_FILES = ('evaluation_record.json', 'per_cycle_record.jsonl')

STAGES = {
    'smoke': {'campaign_id': 's53_w86_tail_smoke', 'labels': ('c_star',), 'cap': 2, 'concurrency': 1},
    'recert': {'campaign_id': 's53_w86_tail_recert', 'labels': ('c_star', 'n7_4h_e1', 'x0'), 'cap': 500,
               'concurrency': 3},
}
SMOKE_GATE_FILE = 'smoke_gate.json'
SMOKE_MANIFEST_FILE = 'smoke_manifest_sha256.json'

# Memory: measured, not assumed. The committed x0_capture child (persist_certified_models True) peaked at
# 3,470,491,648 B (3.23 GiB) against production's own peak of 2,395,537,408 B before the persist step (+1.00 GiB, the
# pickling); the persist-free C* / unit children peaked at 2.40 GiB. The Planner's "~2.6 GB per child" is that
# persist-free figure (2.58e9 B); with persist the per-child budget is 3.5 GiB.
GIB = 1 << 30
MEMORY_PER_CHILD_BUDGET_BYTES = 7 * GIB // 2   # 3.5 GiB
MEMORY_RULE_TEMPLATE = ('hw.memsize - (wired + anonymous + compressor-occupied) x page size >= {concurrency} x '
                        '{per_child:g} GiB')
MEMORY_RULE_RATIONALE = ('non-reclaimable load = wired + anonymous + compressor-occupied pages; file-backed pages '
                         '(active or inactive) are cache the kernel reclaims on demand, so free + inactive '
                         'under-counts what new processes can obtain (free + inactive recorded alongside)')
MEMORY_BUDGET_DERIVATION = {
    'x0_capture_child_peak_with_persist_bytes': 3470491648,
    'x0_capture_production_peak_before_persist_bytes': 2395537408,
    's47_recert_c_star_child_peak_no_persist_bytes': 2577645568,
    's47_recert_unit_child_peak_no_persist_bytes': 2574581760,
    'source': ('data/SRP1/Results/P515S48/x0_capture/evals/d2c96b1480402a3b_x0/{evaluation_record,post_certification}'
               '.json and data/SRP1/Results/P515S47/campaign_s47_recert/evals/*/evaluation_record.json peak_rss'),
    'budget': '3.5 GiB per child = the measured persist peak (3.23 GiB) + 0.27 GiB margin',
}

EXTRA_CLEAN_FILES = (SCRIPT_NAME, H.ESS_PARAMS_FILE_REL, 'shared_energy_storage_parameters.py',
                     'shared_energy_storage.py', 'p515_s53_w86_key_fix_checks.py')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git_state(rel):
    tracked = bool(H._git(['ls-files', '--', rel]).strip())
    dirty = bool(H._git(['status', '--porcelain', '--', rel]).strip())
    return {'git_tracked': tracked, 'git_clean': not dirty}


def campaign_root(stage):
    return _abs(os.path.join(W86_ROOT_REL, f"campaign_{STAGES[stage]['campaign_id']}"))


def _nodes(label):
    return {n: tuple(map(float, v)) for n, v in CELLS[label].items()}


def _key_of(label):
    return H.candidate_key(H.canonical_candidate(_nodes(label), investment_year=YEAR))


def _eval_key(label, tail=TAIL):
    return H.evaluation_key(_key_of(label), {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=ESS_AGEING_BASELINE,
                            convergence_depth_tail=tail)


def own_process_alive():
    """Other live processes running THIS script (never this process or its ancestors): the launcher refuses to run
    concurrently with another copy of itself. The scan reads `ps` output; no pattern is passed on a command line."""
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and OWN_PROCESS_SUBSTRING in parts[1]:
            hits.append(line.strip())
    return hits


# ======================================================================================================================
#  references
# ======================================================================================================================
def reference_quantities(eval_dir_rel):
    rec = _load(os.path.join(eval_dir_rel, 'evaluation_record.json'))
    rows = _read_jsonl(_abs(os.path.join(eval_dir_rel, 'per_cycle_record.jsonl')))
    g = [r['gross_operational_cost'] for r in rows]
    step = abs(g[-1] - g[-2]) if len(g) >= 2 else None
    tol = rows[-1].get('objective_tolerance') if rows else None
    return {
        'eval_dir': eval_dir_rel, 'status': rec.get('status'), 'candidate_key': rec.get('candidate_key'),
        'eval_key': rec.get('eval_key') or rec.get('candidate_key'),
        'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
        'Q_gross': rec.get('certified_cost'), 'bar': (rec.get('bar') or {}).get('value'),
        'terminal_gross_step_abs': step, 'terminal_objective_tolerance': tol,
        'terminal_gross_step_over_threshold': (step / tol) if (step is not None and tol) else None,
        'rule_ten_terminal_step_over_threshold_production': (rec.get('rule_ten') or {}).get(
            'terminal_step_over_threshold'),
        'terminal_salvage_value': (rec.get('recourse_components') or {}).get('terminal_salvage_value'),
        'solve_profile_observed': ((rec.get('solve_profile') or {}).get('observed') or {}).get('permitted_solve'),
        'wall_child_process_s': (rec.get('wall_time_s') or {}).get('child_process_s'),
        'peak_rss_child_bytes': (rec.get('peak_rss') or {}).get('child_python_process_ru_maxrss'),
        'files_sha256': {f: _sha(os.path.join(eval_dir_rel, f)) for f in REFERENCE_FILES},
        'files_git': {f: _git_state(os.path.join(eval_dir_rel, f)) for f in REFERENCE_FILES},
    }


def references():
    out, failures = {}, []
    for label, ref in REFERENCES.items():
        q = reference_quantities(ref['eval_dir'])
        q['campaign'] = ref['campaign']
        ok = (q['status'] == 'certified' and q['candidate_key'] == _key_of(label)
              and q['eval_key'] == PRE_TAIL_EVAL_KEYS[label]
              and all(v['git_tracked'] and v['git_clean'] for v in q['files_git'].values()))
        if 'cross_check_eval_dir' in ref:
            x = reference_quantities(ref['cross_check_eval_dir'])
            same = {k: x[k] == q[k] for k in ('Q_gross', 'bar', 'cycles_run', 'certification_cycle',
                                               'terminal_gross_step_abs', 'candidate_key')}
            q['cross_check'] = {'eval_dir': ref['cross_check_eval_dir'], 'eval_key': x['eval_key'], 'equal': same,
                                'files_sha256': x['files_sha256'], 'note': (
                                    'the pinned A0 x0 record (C3-era spec, no ageing declaration) -- the mixed-era '
                                    'artefact this re-run retires; equal to the C2-declared record on every field')}
            ok = ok and all(same.values())
        q['ok'] = ok
        if not ok:
            failures.append(f'reference {label} failed its checks: {q}')
        out[label] = q
    return out, failures


# ======================================================================================================================
#  memory preflight (the vm_stat rule of the committed launchers, per-child budget from measurement)
# ======================================================================================================================
def memory_rule(concurrency):
    return MEMORY_RULE_TEMPLATE.format(concurrency=concurrency, per_child=MEMORY_PER_CHILD_BUDGET_BYTES / GIB)


def memory_preflight(concurrency):
    required = concurrency * MEMORY_PER_CHILD_BUDGET_BYTES
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    first = text.splitlines()[0]
    page = int(first.split('page size of')[1].split('bytes')[0].strip())
    raw = {}
    for line in text.splitlines()[1:]:
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().rstrip('.')
            if v.isdigit():
                raw[k.strip().strip('"')] = int(v)
    need = {'pages_free': 'Pages free', 'pages_inactive': 'Pages inactive', 'pages_wired_down': 'Pages wired down',
            'anonymous_pages': 'Anonymous pages', 'pages_occupied_by_compressor': 'Pages occupied by compressor'}
    pages = {name: raw.get(label) for name, label in need.items()}
    missing = [n for n, v in pages.items() if v is None]
    total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, check=True).stdout)
    out = {'utc': _utc(), 'vm_stat_header': first, 'page_size_bytes': page, 'vm_stat_pages': pages,
           'hw_memsize_bytes': total, 'concurrency': concurrency, 'required_bytes': required,
           'required_gib': required / GIB, 'per_child_budget_gib': MEMORY_PER_CHILD_BUDGET_BYTES / GIB,
           'rule': memory_rule(concurrency), 'rule_rationale': MEMORY_RULE_RATIONALE,
           'budget_derivation': MEMORY_BUDGET_DERIVATION, 'missing_vm_stat_figures': missing}
    if missing:
        out.update({'available_bytes': None, 'available_gib': None, 'pass': False})
        return out
    non_reclaimable = (pages['pages_wired_down'] + pages['anonymous_pages'] + pages['pages_occupied_by_compressor']) * page
    avail = total - non_reclaimable
    out.update({'non_reclaimable_gib': non_reclaimable / GIB, 'available_bytes': avail, 'available_gib': avail / GIB,
                'free_plus_inactive_gib': (pages['pages_free'] + pages['pages_inactive']) * page / GIB,
                'pass': avail >= required})
    return out


def _memory_line(m):
    if m.get('available_gib') is None:
        return f"vm_stat figures missing {m['missing_vm_stat_figures']} (rule {m['rule']})"
    return (f"available (memsize - wired - anonymous - compressor) = {m['available_gib']:.2f} GiB; free+inactive = "
            f"{m['free_plus_inactive_gib']:.2f} GiB (recorded only); required {m['required_gib']:.2f} GiB ({m['rule']})")


# ======================================================================================================================
#  rule eleven: a capture path for every quantity the gates / v29 require, asserted BEFORE any run
# ======================================================================================================================
def launcher_checklist():
    child_src = inspect.getsource(H._child_real)
    build_src = inspect.getsource(H.build_evaluation_record)
    main_child_src = inspect.getsource(H.main_child)
    post_src = inspect.getsource(H.run_post_certification)
    import p515_g_g1_g4_admm_gates as G
    arm_src = inspect.getsource(G.run_admm_arm)
    checks = {
        'status_cycles_cert_cycle': all(s in build_src for s in ("'status': status", "'cycles_run': len(rows)",
                                                                 "'certification_cycle':")),
        'Q_gross': "'certified_cost': report.get('gross_operational_cost')" in build_src,
        'bar': "'bar': bar" in build_src,
        'rule_ten_ratio': "'terminal_step_over_threshold':" in build_src,
        'per_cycle_trajectory_with_gross_and_tolerance': (
            "'per_cycle_record.jsonl'" in child_src
            and all(f in H.PER_CYCLE_TRAJECTORY_FIELDS for f in ('cycle', 'gross_operational_cost',
                                                                 'objective_tolerance', 'cycle_convergence',
                                                                 'consecutive_converged_cycles', 'boyd_all_pass',
                                                                 'local_solves_ok'))),
        'solve_profile_per_event_in_record': ("'solve_profile': report.get('solve_profile')" in build_src
                                              and "'identity_holds':" in arm_src
                                              and "'reconciliation_supported':" in arm_src
                                              and "'retry_solves_credited':" in arm_src),
        'tail_checklist_before_run': ('tail_checklist = assert_convergence_depth_tail_capture(spec)' in child_src
                                      and child_src.find('assert_convergence_depth_tail_capture(spec)')
                                      < child_src.find('G.run_admm_arm(')),
        'appender_created_before_run': (0 <= child_src.find('appender = ConvergenceDepthAppender(eval_dir, '
                                                            'tail_checklist)') < child_src.find('G.run_admm_arm(')),
        'append_hooks_around_run': 'convergence_depth_append_hooks(appender)' in child_src,
        'end_of_run_capture': 'persist_convergence_depth_capture(state, eval_dir)' in child_src,
        'tail_state_check': 'convergence_depth_tail_state_check(tail_checklist, state)' in child_src,
        'append_reconcile_then_seal': (0 <= child_src.find("appender.reconcile(holder['convergence_depth_capture'], "
                                                           "state)") < child_src.find('appender.seal()')),
        'record_carries_tail_fields': all(f"'{k}':" in child_src for k in (
            'convergence_depth_tail_checklist_asserted_before_run', 'convergence_depth_tail_state_check',
            'convergence_depth_capture', 'convergence_depth_per_round_append')),
        'child_exits_2_on_capture_error': ("outcome.get('convergence_depth_capture_error')" in main_child_src),
        'failure_path_drains_the_round_in_flight': 'drain_on_failure(' in main_child_src,
        'post_certification_persist': "out['persisted_models'] = persist_fn(models, eval_dir)" in post_src,
        'ess_ageing_readback': ("'ess_ageing_verified_pre_run':" in child_src
                                and "'ess_ageing_readback_terminal':" in child_src),
        'files_named_by_harness': (H.NETWORK_IPOPT_SOLVE_RECORDS_FILE == 'network_ipopt_solve_records.jsonl'
                                   and H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE
                                   == 'network_ipopt_solve_records_append.jsonl'
                                   and H.CONVERGENCE_DEPTH_APPEND_EVENTS_FILE == 'convergence_depth_append_events.jsonl'
                                   and H.CONVERGENCE_DEPTH_TAIL_STATE_FILE == 'convergence_depth_tail_state.json'),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (W86 launcher): capture paths missing: {missing}')
    return {'checks': checks, 'harness_record_capture_checklist': H.assert_record_capture_paths()}


# ======================================================================================================================
#  the pre-launch assertion (Planner, W86 step 3)
# ======================================================================================================================
def committed_eval_keys(exclude_roots=()):
    """eval key -> [spec paths] over every git-tracked campaign spec, except those under `exclude_roots`."""
    out = {}
    for rel in sorted(p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()):
        if any(rel.startswith(r + os.sep) for r in exclude_roots):
            continue
        for e in _load(rel).get('candidates') or []:
            out.setdefault(H._entry_eval_key(e), []).append(rel)
    return out


def pre_launch_assertion(spec=None):
    """The three eval keys, recomputed now with the declarations, are EXACTLY the expected ones (and, for a frozen
    spec, its entries carry exactly them); none is a pre-tail key; none appears in any committed campaign spec other
    than the W86 smoke (the smoke's c_star shares the key by design: same candidate x configuration, cap 2, never
    certified). x0's key differs from the pinned x0 keys (A0 7aa017f0, s48 d2c96b14) -- it is a re-run, not a pin."""
    smoke_rel = os.path.relpath(campaign_root('smoke'), REPO)
    recert_rel = os.path.relpath(campaign_root('recert'), REPO)
    committed = committed_eval_keys(exclude_roots=(smoke_rel, recert_rel))
    per = {}
    for label in STAGES['recert']['labels']:
        now = _eval_key(label)
        undeclared = _eval_key(label, tail=None)
        declared_off = _eval_key(label, tail={'enabled': False, 'compl_inf_tol': 1e-6})
        entry = next((e for e in (spec or {}).get('candidates') or [] if e['label'] == label), None)
        per[label] = {
            'expected': EXPECTED_EVAL_KEYS[label], 'recomputed_now': now,
            'recomputed_equals_expected': now == EXPECTED_EVAL_KEYS[label],
            'frozen_entry_eval_key': (entry or {}).get('eval_key'),
            'frozen_entry_equals_expected': (entry is None if spec is None
                                             else (entry or {}).get('eval_key') == EXPECTED_EVAL_KEYS[label]),
            'pre_tail_key': PRE_TAIL_EVAL_KEYS[label], 'undeclared_key_equals_pre_tail': undeclared
            == PRE_TAIL_EVAL_KEYS[label], 'declared_off_key_equals_pre_tail': declared_off == PRE_TAIL_EVAL_KEYS[label],
            'differs_from_pre_tail': now != PRE_TAIL_EVAL_KEYS[label],
            'absent_from_every_committed_spec_outside_w86': now not in committed,
            'committed_specs_holding_it_outside_w86': committed.get(now, []),
        }
    x0_pins = {'a0_c7_x0': '7aa017f09989b56d17a1307d7ae2d9d86582e9666295ee9e25ef560ee32f975b',
               's48_x0_capture': PRE_TAIL_EVAL_KEYS['x0']}
    per['x0']['differs_from_every_pinned_x0_key'] = all(per['x0']['recomputed_now'] != v for v in x0_pins.values())
    per['x0']['pinned_x0_keys'] = x0_pins
    holds = all(v['recomputed_equals_expected'] and v['frozen_entry_equals_expected'] and v['differs_from_pre_tail']
                and v['undeclared_key_equals_pre_tail'] and v['declared_off_key_equals_pre_tail']
                and v['absent_from_every_committed_spec_outside_w86'] for v in per.values()) \
        and per['x0']['differs_from_every_pinned_x0_key']
    return {'per_cell': per, 'holds': holds}


# ======================================================================================================================
#  freeze (both stages)
# ======================================================================================================================
def _configuration():
    return {'name': (f'{LABEL}: the case file alone (AA keep_memory declared) with the ESS ageing parameters declared, '
                     'the convergence-depth tight tail DECLARED ENABLED (compl_inf_tol 1e-6, certifying cycles only), '
                     'post-certification: persist the certified TSO/DSO models only'),
            'arm_label': ARM_LABEL, 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
            'ess_ageing_baseline': copy.deepcopy(ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': LABEL,
            'convergence_depth_tail': dict(TAIL),
            'note': ('no overrides; no model variant; post-certification persist_certified_models only (no hull '
                     'polish, no reference); num_max_iters := cap; tail per spec v28 (AA-off predicate, '
                     'non-latching, ESSO untouched)')}


def _find_v29():
    hits = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_V29_PREFIX) and f.endswith('.json'))
    if len(hits) != 1:
        return None, None
    rel = os.path.join(_P53, hits[0])
    sha = _sha(rel)
    if not hits[0] == f'{SPEC_V29_PREFIX}{sha[:8]}.json':
        raise RuntimeError(f'v29 file name does not carry its sha256 prefix: {rel} {sha}')
    return rel, sha


def _common_checks(stage):
    failures, evidence = [], {}
    v28_sha = _sha(SPEC_V28['path'])
    evidence['spec_v28'] = {'path': SPEC_V28['path'], 'sha256': v28_sha, **_git_state(SPEC_V28['path'])}
    if v28_sha != SPEC_V28['sha256'] or not evidence['spec_v28']['git_clean']:
        failures.append(f"spec v28 not as pinned: {evidence['spec_v28']}")
    ess_sha = _sha(H.ESS_PARAMS_FILE_REL)
    evidence['ess_params_sha256'] = ess_sha
    if ess_sha != ESS_PARAMS_SHA256:
        failures.append(f'ESS params file sha256 {ess_sha} != {ESS_PARAMS_SHA256}')
    loaded = H.ess_ageing_canonical_text(H.load_ess_ageing_parameters(_abs(H.ESS_PARAMS_FILE_REL)))
    if loaded != H.ess_ageing_canonical_text(ESS_AGEING_BASELINE):
        failures.append('the ESS params file does not load to the declaration')
    from planning_parameters import PlanningParameters
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    evidence['case_file_aa_loaded'] = params.admm.anderson_acceleration
    if params.admm.anderson_acceleration != CASE_FILE_AA:
        failures.append(f'case file AA {params.admm.anderson_acceleration} != declaration {CASE_FILE_AA}')
    evidence['case_file_tail_default'] = dict(params.admm.convergence_depth_tail)
    if params.admm.convergence_depth_tail.get('enabled') is not False:
        failures.append('production default tail is not OFF')
    kc = _load(W86_KEY_CHECKS)
    evidence['w86_key_checks'] = {'path': W86_KEY_CHECKS, 'sha256': _sha(W86_KEY_CHECKS), 'all_hold': kc.get('all_hold'),
                                  **_git_state(W86_KEY_CHECKS)}
    if not (kc.get('all_hold') and evidence['w86_key_checks']['git_clean'] and evidence['w86_key_checks']['git_tracked']):
        failures.append(f"W86 key checks not committed / not ALL HOLD: {evidence['w86_key_checks']}")
    refs, more = references()
    evidence['references'] = refs
    failures += more
    try:
        evidence['rule_eleven'] = launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
    others = own_process_alive()
    evidence['own_process_alive'] = others
    if others:
        failures.append(f'another copy of this launcher is alive: {others}')
    return failures, evidence


def _validate_spec(stage, spec, v29=None):
    cfg = spec['configuration']
    entries = spec['candidates']
    extra = spec.get('extra') or {}
    st = STAGES[stage]
    checks = {
        'campaign_id': spec.get('campaign_id') == st['campaign_id'],
        'entries_in_order': [e['label'] for e in entries] == list(st['labels']),
        'cap': spec.get('cap') == st['cap'], 'concurrency': spec.get('concurrency') == st['concurrency'],
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'ess_params_file_pinned': (cfg.get('ess_params_file') or {}).get('sha256') == ESS_PARAMS_SHA256,
        'tail_declared_enabled_1e-6': cfg.get('convergence_depth_tail') == TAIL,
        'no_overrides': cfg.get('overrides') == {} and all(e.get('overrides') == {} for e in entries),
        'no_model_variant': not any('model_variant' in e for e in entries),
        'arm_label': cfg.get('arm_label') == ARM_LABEL,
        'not_a_stub_spec': not extra.get('test_only_stub'),
        'script_recorded': extra.get('campaign_script') == SCRIPT_NAME,
        'post_certification_persist_only': all(
            (e.get('post_certification') or {}).get('persist_certified_models') is True
            and (e.get('post_certification') or {}).get('hull_polish') is False
            and (e.get('post_certification') or {}).get('reference') is None for e in entries),
    }
    for e in entries:
        label = e['label']
        canon = H.canonical_candidate(_nodes(label), investment_year=YEAR)
        checks[f'{label}:canonical_and_key'] = e.get('canonical') == canon and e.get('key') == H.candidate_key(canon)
        checks[f'{label}:eval_key_expected'] = e.get('eval_key') == EXPECTED_EVAL_KEYS[label]
    if stage == 'recert':
        checks['spec_v29_pinned'] = v29 is not None and extra.get('spec_v29') == {'path': v29[0], 'sha256': v29[1]}
    return checks


def freeze(stage, started):
    tag = f'W86-{stage.upper()}-FREEZE'
    root = campaign_root(stage)
    failures = H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
    more, evidence = _common_checks(stage)
    failures += more
    v29 = None
    if stage == 'recert':
        v29 = _find_v29()
        if v29[0] is None:
            failures.append('frozen spec v29 not found (run --freeze-spec first)')
        else:
            failures += [f'v29: {k}' for k, v in _git_state(v29[0]).items() if not v]
            if not _load(v29[0]).get('smoke_gate', {}).get('pass'):
                failures.append('v29 does not record a PASS smoke gate')
    pre = pre_launch_assertion()
    evidence['pre_launch_assertion_before_freeze'] = pre
    if not pre['holds']:
        failures.append(f'pre-launch assertion (recomputed keys) fails: {pre}')
    memory = memory_preflight(STAGES[stage]['concurrency'])
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    st = STAGES[stage]
    extra = {
        'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
        'stage': stage, 'stage_text': STAGE_TEXT, 'label': LABEL,
        'spec_v28': evidence['spec_v28'], 'w86_key_checks': evidence['w86_key_checks'],
        'expected_eval_keys': {label: EXPECTED_EVAL_KEYS[label] for label in st['labels']},
        'pre_launch_assertion_at_freeze': pre,
        'references': {label: evidence['references'][label] for label in st['labels']},
        'objective_convention': ('Q = certified_cost = gross_operational_cost (settlement excluded); salvage and net '
                                 'recourse reported beside it (equal to gross at the x0 cell)'),
        'solve_claim': ('RECONCILED PER EVENT, NOT GUARD-VERIFIED: the child record solve_profile (the child\'s own '
                        'in-window SolveProfileGuard(N.PERMITTED) count == 51 x (cycles_run + 1) + retries attempted); '
                        'this parent\'s SolveProfileGuard(permitted=()) verify(0)'),
        'memory_preflight_rule': memory_rule(st['concurrency']), 'memory_preflight_rule_rationale': MEMORY_RULE_RATIONALE,
        'memory_budget_derivation': MEMORY_BUDGET_DERIVATION, 'memory_at_freeze_non_gating': memory,
        'rule_eleven': evidence['rule_eleven']['checks'],
    }
    if stage == 'smoke':
        extra['smoke_gate_declared_before_run'] = smoke_gate_text()
        extra['declared_solves'] = {'base': SOLVES_PER_ROUND * (st['cap'] + 1),
                                    'rule': '51 per round x (cap 2 + 1) + every retry attempted, per event'}
    else:
        extra['spec_v29'] = {'path': v29[0], 'sha256': v29[1]}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, st['campaign_id'],
        [(label, _nodes(label), {'investment_year': YEAR, 'post_certification': dict(POST_CERTIFICATION)})
         for label in st['labels']],
        configuration=_configuration(), cap=st['cap'], concurrency=st['concurrency'],
        authority=['PLANNER_BRIEF_2026-09-13.md Addendum 46 ruling 7', 'Planner task W86',
                   SPEC_V28['path']] + ([v29[0]] if v29 else []),
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES, extra=extra)
    checks = _validate_spec(stage, spec, v29)
    pre_frozen = pre_launch_assertion(spec) if stage == 'recert' else None
    guard_failures = PARENT_GUARD.verify(0)
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    for e in spec['candidates']:
        _log(f"[{tag}]   {e['label']}: key={e['key'][:16]} eval_key={e['eval_key']} eval_dir={e['eval_dir']} "
             f"post_certification={e['post_certification']}")
    _log(f'[{tag}] spec checks: all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    if pre_frozen is not None:
        _log(f"[{tag}] pre-launch assertion on the FROZEN spec: holds={pre_frozen['holds']}")
        for label, v in pre_frozen['per_cell'].items():
            _log(f"[{tag}]   {label}: frozen {v['frozen_entry_eval_key']} == expected {v['expected'][:16]}: "
                 f"{v['frozen_entry_equals_expected']}; pre-tail {v['pre_tail_key'][:16]} differs "
                 f"{v['differs_from_pre_tail']}; absent from committed specs {v['absent_from_every_committed_spec_outside_w86']}")
    _log(f"[{tag}] memory at freeze (non-gating): {_memory_line(memory)} -> would {'PASS' if memory['pass'] else 'REFUSE'}")
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures} wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not guard_failures and (pre_frozen is None or pre_frozen['holds'])
    _log(f'[{tag}] freeze {"OK" if ok else "NOT OK"}; run with: --stage {stage} --run --spec-sha256 {spec_sha}')
    PARENT_GUARD.uninstall()
    if not ok:
        sys.exit(1)


# ======================================================================================================================
#  the smoke
# ======================================================================================================================
def smoke_gate_text():
    doc = __doc__
    return doc[doc.index('SMOKE GATE'):doc.index('EXACT COMMANDS')].strip()


def _production_compl_inf_tol(baseline):
    """network name -> the production compl_inf_tol in force (the tail baseline: the value passed, else IPOPT's
    default 1e-4)."""
    out = {}
    for _label, b in (baseline or {}).items():
        out[b['network']] = b['value'] if b.get('has_key') else IPOPT_DEFAULT_COMPL_INF_TOL
    return out


def evaluation_checks(eval_dir, rec, run_working_dir_id, expect_certified=None):
    """The capture checks shared by the smoke and every re-certification entry (zero solves; files only)."""
    d = {}
    rounds_expected = (rec.get('cycles_run') or 0) + 1
    rec_path = os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE)
    app_path = os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE)
    ev_path = os.path.join(eval_dir, H.CONVERGENCE_DEPTH_APPEND_EVENTS_FILE)
    tail_path = os.path.join(eval_dir, H.CONVERGENCE_DEPTH_TAIL_STATE_FILE)
    present = {p: os.path.isfile(os.path.join(eval_dir, p)) for p in (
        H.NETWORK_IPOPT_SOLVE_RECORDS_FILE, H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE,
        H.CONVERGENCE_DEPTH_APPEND_EVENTS_FILE, H.CONVERGENCE_DEPTH_TAIL_STATE_FILE)}
    d['files_present'] = present
    if not all(present.values()):
        return {'all_files_present': False}, d
    end_bytes, app_bytes = open(rec_path, 'rb').read(), open(app_path, 'rb').read()
    records = _read_jsonl(rec_path)
    events = _read_jsonl(ev_path)
    tail_state = json.load(open(tail_path))
    app = rec.get('convergence_depth_per_round_append') or {}
    cap_summary = rec.get('convergence_depth_capture') or {}
    drained = [e for e in events if e.get('event') == 'drained']
    d['append'] = {'record_field': app, 'end_of_run_sha256': hashlib.sha256(end_bytes).hexdigest(),
                   'append_sha256': hashlib.sha256(app_bytes).hexdigest(), 'n_end_of_run': len(records),
                   'rounds_drained': [e['round'] for e in drained], 'n_bytes': len(end_bytes)}
    c = {}
    c['append_reconciles_byte_identical'] = (
        app.get('ok') is True and app.get('records_append_byte_identical_to_end_of_run_file') is True
        and app.get('tail_state_rebuilt_from_events_equals_returned_state') is True and not app.get('write_errors')
        and end_bytes == app_bytes and d['append']['end_of_run_sha256'] == cap_summary.get(
            'network_ipopt_solve_records_sha256')
        and app.get('n_records_appended') == len(records) and app.get('rounds_appended') == list(range(rounds_expected))
        and d['append']['rounds_drained'] == list(range(rounds_expected)))
    # checklist = line 1, before any solve
    first = events[0] if events else {}
    work_logs = os.path.join(_work_dir(), run_working_dir_id, 'logs')
    births = []
    if os.path.isdir(work_logs):
        for f in os.listdir(work_logs):
            births.append(os.stat(os.path.join(work_logs, f)).st_birthtime)
    t_check = datetime.fromisoformat(first['utc']).timestamp() if first.get('utc') else None
    t_first_drain = datetime.fromisoformat(drained[0]['utc']).timestamp() if drained else None
    d['checklist'] = {'first_event': first.get('event'), 'utc': first.get('utc'), 'pid': first.get('pid'),
                      'record_child_pid': rec.get('child_pid'), 'work_logs_dir': os.path.relpath(work_logs, REPO),
                      'n_log_files': len(births),
                      'earliest_log_birth_utc': (datetime.fromtimestamp(min(births), timezone.utc).isoformat()
                                                 if births else None),
                      'first_drain_utc': drained[0]['utc'] if drained else None,
                      'seconds_before_first_log': (min(births) - t_check) if (births and t_check) else None}
    checklist = first.get('checklist') or {}
    c['checklist_line1_before_any_solve'] = (
        first.get('event') == 'checklist'
        and checklist == rec.get('convergence_depth_tail_checklist_asserted_before_run')
        and checklist.get('tail_enabled_for_this_run') is True and checklist.get('declared') == TAIL
        and first.get('pid') == rec.get('child_pid') and bool(births) and t_check is not None
        and t_check < min(births) and t_first_drain is not None and t_check < t_first_drain)
    sc = rec.get('convergence_depth_tail_state_check') or {}
    c['tail_state_check_matches'] = (sc.get('match') is True and sc.get('expected_enabled') is True
                                     and sc.get('enabled_in_state') is True
                                     and sc.get('compl_inf_tol_tail_in_state') == TAIL['compl_inf_tol'])
    sp = rec.get('solve_profile') or {}
    obs = (sp.get('observed') or {})
    d['solve_profile'] = sp
    n_net = len(records)
    c['solve_profile_reconciled_per_event'] = (
        sp.get('reconciliation_supported') is True and sp.get('identity_holds') is True
        and sp.get('solves_per_cycle') == SOLVES_PER_ROUND and sp.get('rounds') == rounds_expected
        and sp.get('base_solves') == SOLVES_PER_ROUND * rounds_expected
        and obs.get('blocked_solve') == 0 and obs.get('blocked_exec') == 0
        and obs.get('permitted_solve') == sp.get('expected_solves')
        and n_net == obs.get('permitted_solve') - N_ESSO * rounds_expected)
    # floor records at the compl_inf_tol in force: production value, or the tail value on tail-active cycles
    prod = _production_compl_inf_tol(tail_state.get('baseline'))
    active_cycles = {p['cycle'] for p in (tail_state.get('per_cycle') or []) if p.get('active')}
    bad = []
    for r in records:
        want = TAIL['compl_inf_tol'] if r.get('round') in active_cycles else prod.get(r.get('network'))
        if not (r.get('compl_inf_tol_in_force') == want and r.get('parse_reason') is None
                and r.get('options_list_agrees') is True):
            bad.append({k: r.get(k) for k in ('network', 'year', 'day', 'round', 'attempt', 'compl_inf_tol_in_force',
                                              'parse_reason', 'options_list_agrees')})
    from collections import Counter
    d['floor'] = {'production_compl_inf_tol': prod, 'tail_active_cycles': sorted(active_cycles),
                  'n_records': n_net, 'n_bad': len(bad), 'bad_first': bad[:10],
                  'floor_status_tally': cap_summary.get('floor_status_tally'),
                  'floor_status_tally_tail_cycles': dict(sorted(Counter(
                      f"{'TSO' if r.get('agent') == 'TSO' else 'DSO'}|{r.get('floor_status')}" for r in records
                      if r.get('round') in active_cycles).items())),
                  'exit_tally': cap_summary.get('exit_tally'), 'records_per_round': cap_summary.get('records_per_round')}
    c['floor_records_at_compl_inf_tol_in_force'] = bool(records) and not bad
    c['append_sealed_after_reconcile'] = (bool(events) and events[-1].get('event') == 'sealed'
                                          and not app.get('write_errors'))
    pc = rec.get('post_certification') or {}
    pkl = os.path.join(eval_dir, 'certified_models.pkl')
    if expect_certified is None:
        expect_certified = rec.get('status') == 'certified'
    if expect_certified:
        c['post_certification_persisted'] = (pc.get('status') == 'evaluated' and os.path.isfile(pkl)
                                             and (pc.get('persisted_models') or {}).get('sha256') == H.sha256_file(pkl))
    else:
        c['post_certification_skipped_uncertified'] = pc.get('status') == 'skipped' and not os.path.exists(pkl)
    d['post_certification'] = pc
    d['certified_models_pkl'] = ({'path': os.path.relpath(pkl, REPO), 'bytes': os.path.getsize(pkl),
                                  'sha256': H.sha256_file(pkl)} if os.path.isfile(pkl) else None)
    ea_pre = ((rec.get('ess_ageing_verified_pre_run') or {}).get('readback_pre_run') or {}).get('all_match')
    ea_post = (rec.get('ess_ageing_readback_terminal') or {}).get('all_match')
    c['ess_ageing_readback_all_match'] = ea_pre is True and ea_post is True
    d['tail_state_check'] = sc
    d['tail_state_summary'] = {'n_per_cycle': len(tail_state.get('per_cycle') or []),
                               'cycles_active': sorted(active_cycles),
                               'n_acted': sum(1 for p in tail_state.get('per_cycle') or [] if p.get('acted')),
                               'restore_at_exit_acted': (tail_state.get('restore_at_exit') or {}).get('acted')}
    c['all_files_present'] = True
    return c, d


def _work_dir():
    import p56a_oracle as O   # its WORK_DIR constant (the child's G.O.WORK_DIR); the import builds / solves nothing
    return O.WORK_DIR


def smoke_checks(entry, eval_dir):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip()) \
        if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None
    checks, detail = {}, {'exit_code': exit_code}
    checks['S1_child_exit0_record_uncertified_2_cycles'] = (
        exit_code == 0 and bool(rec) and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json'))
        and rec.get('status') == 'not_certified' and rec.get('cycles_run') == 2)
    if not rec:
        return checks, detail, rec
    c, d = evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'], expect_certified=False)
    detail.update(d)
    checks['S2_append_written_and_byte_identical_to_end_of_run'] = c.get('append_reconciles_byte_identical', False)
    checks['S3_checklist_line1_before_any_solve'] = c.get('checklist_line1_before_any_solve', False)
    checks['S4_tail_state_check_matches'] = c.get('tail_state_check_matches', False)
    checks['S5_solve_profile_reconciled_per_event_153'] = (c.get('solve_profile_reconciled_per_event', False)
                                                          and (rec.get('solve_profile') or {}).get('base_solves') == 153)
    # S6: tail inactive at cap 2
    events = _read_jsonl(os.path.join(eval_dir, H.CONVERGENCE_DEPTH_APPEND_EVENTS_FILE))
    tail_state = json.load(open(os.path.join(eval_dir, H.CONVERGENCE_DEPTH_TAIL_STATE_FILE)))
    applies = [e for e in events if e.get('event') == 'apply']
    nexts = [e for e in events if e.get('event') == 'next_state']
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    stdout_text = open(os.path.join(eval_dir, 'child_stdout.log')).read()
    per_cycle = tail_state.get('per_cycle') or []
    detail['S6'] = {'apply_cycles': [e.get('cycle') for e in applies],
                    'apply_active': [e['record'].get('active') for e in applies],
                    'apply_acted': [e['record'].get('acted') for e in applies],
                    'next_state_values': [e.get('value') for e in nexts],
                    'row_cycle_convergence': [r.get('cycle_convergence') for r in rows],
                    'tail_on_lines_in_child_stdout': stdout_text.count('Convergence-depth tail ON')}
    checks['S6_tail_inactive_at_cap2'] = (
        len(per_cycle) == 2 and not any(p.get('active') or p.get('acted') for p in per_cycle)
        and [p.get('aa_off_predicate_end_of_cycle') for p in per_cycle] == [False, False]
        and (tail_state.get('restore_at_exit') or {}).get('acted') is False
        and detail['S6']['apply_cycles'] == [1, 2, None] and not any(detail['S6']['apply_active'])
        and not any(detail['S6']['apply_acted']) and detail['S6']['next_state_values'] == [False, False]
        and not any(detail['S6']['row_cycle_convergence']) and detail['S6']['tail_on_lines_in_child_stdout'] == 0)
    checks['S7_floor_records_at_production_compl_inf_tol'] = (c.get('floor_records_at_compl_inf_tol_in_force', False)
                                                             and d['floor']['tail_active_cycles'] == [])
    checks['S8_append_sealed_no_write_error'] = c.get('append_sealed_after_reconcile', False)
    checks['S9_post_certification_skipped_uncertified'] = c.get('post_certification_skipped_uncertified', False)
    checks['S10_eval_key_expected_fresh_names'] = (rec.get('eval_key') == EXPECTED_EVAL_KEYS['c_star']
                                                   and entry['eval_key'] == EXPECTED_EVAL_KEYS['c_star'])
    checks['S11_ess_ageing_readback_all_match'] = c.get('ess_ageing_readback_all_match', False)
    return checks, detail, rec


def _manifest(root):
    out = {}
    for directory, _dirs, files in os.walk(root):
        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            out[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    return out


def _run_preconditions(stage, spec_sha256, tag):
    root = campaign_root(stage)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    more, evidence = _common_checks(stage)
    failures += more
    v29 = _find_v29() if stage == 'recert' else None
    checks = _validate_spec(stage, spec, v29)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    if spec['harness']['sha256'] != H.sha256_file(H.HARNESS_PATH):
        failures.append('harness sha256 differs from the frozen spec')
    if spec['extra'].get('campaign_script_sha256') != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script sha256 differs from the frozen spec')
    if spec['configuration']['case_file_sha256'] != H.sha256_file(H.CASE_FILE):
        failures.append('case file sha256 differs from the frozen spec')
    for e in spec['candidates']:
        for eid in e['working_dir_ids'].values():
            if os.path.exists(os.path.join(_work_dir(), eid)):
                failures.append(f'working dir already exists (never reusable): {eid}')
    memory = memory_preflight(STAGES[stage]['concurrency'])
    _log(f"[{tag}] memory preflight: {_memory_line(memory)} -> {'PASS' if memory['pass'] else 'REFUSE'}")
    if not memory['pass']:
        failures.append(f'memory preflight REFUSED: {_memory_line(memory)}')
    return root, spec_path, spec, failures, evidence, memory, v29


def run_smoke(started, spec_sha256):
    tag = 'W86-SMOKE'
    root, spec_path, spec, failures, evidence, memory, _v29 = _run_preconditions('smoke', spec_sha256, tag)
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    entry = spec['candidates'][0]
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f"[{tag}] ONE cell {entry['label']} eval_key {entry['eval_key']} cap {spec['cap']} through H.evaluate "
         f"(real child: main_child -> _child_real); DECLARED: {spec['extra']['declared_solves']}")
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f'[{tag}] campaign lock acquired: {lock}')
    batch_info = None
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        H.evaluate([entry['label']], ctx)
        batch_info = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    try:
        checks, detail, rec = smoke_checks(entry, eval_dir)
    except Exception as error:  # noqa: BLE001 -- the gate FAILS, recorded
        checks, detail, rec = ({'smoke_checks_ran': False},
                               {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}, {})
        print(detail['traceback'], file=sys.stderr, flush=True)
    guard_failures = PARENT_GUARD.verify(0)
    checks['S12_parent_guard_zero'] = not guard_failures
    gate = {
        'stage': STAGE_TEXT, 'gate': 'W86 step 2: the one-cell child smoke (c_star, cap 2, tail enabled)',
        'utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']), 'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256, 'eval_dir': os.path.relpath(eval_dir, REPO),
        'checks': checks, 'pass': all(checks.values()), 'failing': sorted(k for k, v in checks.items() if not v),
        'solve_claim': ('RECONCILED PER EVENT, NOT GUARD-VERIFIED: the child\'s in-window guard count '
                        f"{((rec.get('solve_profile') or {}).get('observed') or {})} reconciled in its record "
                        f"(expected {((rec.get('solve_profile') or {}).get('expected_solves'))}); the parent cannot "
                        'verify a guard in the child process'),
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'detail': detail,
        'reported_not_gated': {'peak_rss': rec.get('peak_rss'), 'wall_time_s': rec.get('wall_time_s'),
                               'launcher_wall_s': time.time() - started,
                               'certified_cost_terminal_gross': rec.get('terminal_gross_operational_cost'),
                               'cycles_run': rec.get('cycles_run')},
        'memory_preflight': memory, 'batch_info': batch_info, 'smoke_gate_text': smoke_gate_text(),
        'pre_run_evidence': {k: evidence[k] for k in ('spec_v28', 'w86_key_checks', 'ess_params_sha256',
                                                      'case_file_aa_loaded', 'case_file_tail_default')},
    }
    H._write_once_json(os.path.join(root, SMOKE_GATE_FILE), gate)
    H._write_once_json(os.path.join(root, SMOKE_MANIFEST_FILE), _manifest(root))
    PARENT_GUARD.uninstall()
    for k, v in checks.items():
        _log(f'[{tag}]   {k}: {"PASS" if v else "FAIL"}')
    _log(f"[{tag}] solve claim: {gate['solve_claim']}")
    _log(f"[{tag}] GATE {'PASS' if gate['pass'] else 'FAIL'} failing={gate['failing']} wall={time.time() - started:.0f}s")
    sys.exit(0 if gate['pass'] else 1)


# ======================================================================================================================
#  frozen spec v29
# ======================================================================================================================
PREDICTIONS = {
    'recorded': 'BEFORE the re-certification runs (the W86 smoke, cap 2, was seen first; it tests capture, not values)',
    'author_verbatim': ('Prediction: SRP1 references (C*, x = 0, smallest unit) unchanged or within 1e-6 relative (SRP1 '
                        'already at the floor 192/192); at 2 x 2 the fifteen early stops would reach the floor.'),
    'worker': {
        'W1_systematic_shift': ('each cell: dQ = Q_tail - Q_ref ~ -3.4e3 EUR (~ -5e-6 relative) -- the W82 at-floor '
                                'barrier gap per SRP1 cell (G ~ 3,387-3,466 EUR: TSO ~950-1,020 at compl_inf_tol 5e-4, '
                                'DSO ~2,435-2,450 at the 1e-4 default) falling by ~500x (TSO) and ~100x (DSO) in the '
                                'tight certifying cycles (v28 P3); on top of path / stopping differences of the order '
                                'of the terminal step'),
        'W2_inside_own_bar': ('|dQ| < bar_ref on every cell -- NOT material, no fallback (v28 P5, 0.75); bars: C* '
                              '25,146.49, unit 25,104.65, x0 9,629.98'),
        'W3_author_band_refuted': ('|dQ| > 1e-6 |Q_ref| (~650 EUR) on each cell -- the author\'s band is refuted '
                                   '(v28 P4, 0.8)'),
        'W4_determinacy': ('the -3.4e3 shift is resolvable against the terminal-step error bar (step_ref + step_tail) '
                           'only where that bar is < 3.4e3: likely at x0 (ref step 219.87) and C* (1,753.61), not '
                           'guaranteed at the unit (3,702.76); where |dQ| < the bar the sign is indeterminate '
                           '(CLAUDE.md), not a confirmation'),
        'W5_certifies': 'each cell certifies within cap 500 with the tail (v28 P1, 0.85)',
        'W6_cert_cycle': 'certification cycle within +3 of the reference for each cell (v28 P2, 0.6): C* 87, unit 112, x0 132',
        'W7_R': ('R = Q(0) - Q(unit) changes by less than the two-cell bar-sum 34,734.63 (v28 P6, 0.9); committed '
                 'R = 259,427.77'),
        'W8_solver_side': ('v28 P7 (>= 1 tight solve exits acceptable in a cell: 0.35), P8 (tight-cycle median '
                           'iterations +1 to +6), P10 (retries in the tight cycles at most twice the reference\'s) carry '
                           'over unchanged'),
    },
    'conflict': ('the author predicts |dQ| <= 1e-6 |Q_ref| (~650 EUR); the Worker predicts ~3.4e3 EUR (~5e-6). Both '
                 'predict no fallback (|dQ| < bar_ref); the run discriminates on the 1e-6 band, and only where the '
                 'terminal-step error bar resolves it'),
}

FALLBACK_TEST = {
    'source': 'spec v28 fallback_rule_operational (restated, not changed); Addendum 46 ruling 7',
    'fails_to_certify': ('TEST, per cell: the tail-enabled run ends with status != "certified" under the UNCHANGED '
                         'certification criterion (all three Boyd channels inside tolerance and every local solve '
                         'successful for 10 consecutive cycles) within cap 500; a harness error (child exit != 0, a '
                         'failed per-entry gate) counts as a failure to certify'),
    'moves_materially': 'TEST, per cell: |Q_tail - Q_ref| > bar_ref(cell), the reference\'s own bar',
    'reported_beside_never_replacing': {
        'author_band_ratio': '|dQ| / (1e-6 |Q_ref|)  (> 1: outside the author\'s 1e-6 relative band)',
        'own_bar_ratio': '|dQ| / bar_ref',
        'two_run_bar_resolution': '|dQ| / (bar_ref + bar_tail)  (< 1: indeterminate at the bar)',
        'terminal_step_error_bar': 'e_term = step_ref + step_tail (the gross-cost step at each run\'s last cycle)',
        'terminal_step_resolution': '|dQ| / e_term  (< 1: indeterminate -- not explained beyond stopping slack)',
        'terminal_step_to_threshold': 'step / objective_tolerance at the last cycle, for BOTH runs (rule ten)',
    },
    'consequence': ('if ANY of the three cells fails to certify or moves materially: the fallback -- the unchanged '
                    'configuration (tail disabled) with the depth caveat recorded (the Planner\'s section 8 reading); '
                    'otherwise the tail-enabled references replace them for what follows'),
}

PER_ENTRY_GATES = {
    'applies_to': 'each of the three re-certification cells (c_star, n7_4h_e1, x0); there is no other arm',
    'G1_harness_clean': 'child exit 0, evaluation_record.json written by the child, status certified or not_certified',
    'G2_eval_key': 'record eval_key == the v29 expected key',
    'G3_append_reconcile': ('record convergence_depth_per_round_append.ok True, AND recomputed: appended file '
                            'byte-identical to the end-of-run file, sha256 == the record\'s, rounds 0..cycles_run'),
    'G4_tail_state_check': 'record convergence_depth_tail_state_check.match True (enabled, 1e-6)',
    'G5_solve_profile_reconciled_per_event': ('reconciliation_supported, identity_holds, observed == 51 x (cycles_run '
                                              '+ 1) + retries attempted, 0 blocked; network floor records == observed '
                                              '- 3 x (cycles_run + 1)'),
    'G6_floor_records': ('every record parse_reason None, options_list_agrees, compl_inf_tol_in_force == 1e-6 on '
                         'tail-active cycles and == production (TSO 5e-4, DSO 1e-4 default) elsewhere'),
    'G7_append_sealed': 'last event sealed, no append write error',
    'G8_post_certification': 'certified -> status evaluated and certified_models.pkl written (sha256 recorded); else skipped',
    'G9_ess_ageing_readback': 'all_match before and after the run',
    'comparison': 'FALLBACK_TEST quantities vs the committed reference, reported per cell',
}


def v29_content(smoke_pin, refs):
    cells = {label: {'nodes': {str(n): list(v) for n, v in CELLS[label].items()}, 'investment_year': YEAR,
                     'candidate_key': _key_of(label), 'expected_eval_key': EXPECTED_EVAL_KEYS[label],
                     'pre_tail_eval_key': PRE_TAIL_EVAL_KEYS[label]} for label in STAGES['recert']['labels']}
    solve_profile = {label: {
        'rule': ('51 x (cycles_run + 1) + every retry attempted (per event); cycles_run is the run\'s own (<= 500); '
                 'the post-certification persist step solves nothing (outside the child guard window)'),
        'at_the_reference_certification_cycle': SOLVES_PER_ROUND * (refs[label]['cycles_run'] + 1),
        'reference_observed': refs[label]['solve_profile_observed'],
        'verification': 'RECONCILED PER EVENT in the child record; NOT guard-verified'}
        for label in STAGES['recert']['labels']}
    return {
        'schema': 'p515_frozen_spec_v29', 'version': 29,
        'stage': STAGE_TEXT,
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 46 ruling 7', 'Planner task W86 (steps 1-3, rulings Q1/Q2)'],
        'predecessor': {'path': SPEC_V28['path'], 'sha256': _sha(SPEC_V28['path'])},
        'predecessor_not_edited': 'v28 stays as frozen; v29 records the re-certification it called for',
        'planner_decisions_w86': [
            'three entries: c_star, n7_4h_e1, x0 (nodes 5/7/9 at (0, 0), 2025) -- x = 0 RE-RUN under the C2 baseline '
            'declaration, not pinned; the mixed-era artefact is retired',
            "declare configuration.convergence_depth_tail = {'enabled': True, 'compl_inf_tol': 1e-6}",
            'pre-launch assertion: the three frozen eval keys are exactly 96c5aa50..., ca8927e7..., 5cfe69a6...',
            'cap 500, 10 consecutive cycles; concurrency 3 (one per cell) subject to a memory preflight',
            'persist_certified_models: True',
            'predictions recorded before the run (author and Worker)',
            'fallback test stated operationally (v28), reported against the 1e-6 band and each cell\'s own bar',
            'per-entry gates: reconcile ok, tail state check, solve profile reconciled per event, comparison with the '
            'committed references (terminal-step error bar, terminal-step-to-threshold ratio)',
            'Q1 on W85: a declared-OFF tail shares the undeclared key (implemented and checked in W86 step 1)',
            'Q2 on W85: the child smoke is required before the run; solve claim reconciled per event if no guard can '
            'be verified',
        ],
        'configuration': {**_configuration(), 'cap': 500, 'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES,
                          'concurrency': 3, 'post_certification': dict(POST_CERTIFICATION),
                          'ess_params_file_sha256': ESS_PARAMS_SHA256, 'case_file_sha256': H.sha256_file(H.CASE_FILE)},
        'cells': cells,
        'pre_launch_assertion': pre_launch_assertion(),
        'references': refs,
        'reference_R': {'R_ref': refs['x0']['Q_gross'] - refs['n7_4h_e1']['Q_gross'],
                        'bar_sum': refs['x0']['bar'] + refs['n7_4h_e1']['bar']},
        'declared_solve_profile': solve_profile,
        'solve_claim': ('RECONCILED PER EVENT, NOT GUARD-VERIFIED: the parent (this launcher) cannot arm or verify a '
                        'guard in a child process; the child\'s run_admm_arm guard (armed, raising on an undeclared '
                        'call site) is reconciled per event in its record; the parent\'s own guard permitted=() '
                        'verify(0)'),
        'per_entry_gates': PER_ENTRY_GATES,
        'fallback_test_operational': FALLBACK_TEST,
        'predictions_recorded_before_the_run': PREDICTIONS,
        'smoke_gate': smoke_pin,
        'estimates': {
            'wall': ('~55-65 min expected, <= ~1.5 h: the three cells run concurrently; references: C* 2,858 s (87 '
                     'cycles), unit 3,146 s (112), x0 3,193 s (132, alone) child wall at concurrency 2 / 1; 12 cores, '
                     'one single-threaded child per cell; the tight cycles (~9 per cell) add a few IPOPT iterations '
                     'per solve; the persist step ~10 s'),
            'memory': ('per child peak ~2.40 GiB during the run, ~3.23 GiB after the persist step (measured on '
                       'x0_capture); preflight 3 x 3.5 GiB = 10.5 GiB of 32 GiB'),
            'disk': ('per cell: eval dir ~140-265 MB (references 180 / 137 / 263 MB) + certified_models.pkl ~165 MB + '
                     'tail capture ~10 MB; oracle working dir (IPOPT logs) ~0.8 GB per cell; total ~3.5 GB'),
        },
        'not_permitted': ['no change to the ADMM formulation, the certification criterion, the AA predicate or any '
                          'solver option beyond the tail\'s compl_inf_tol',
                          'no committed artifact modified or re-run onto; fresh campaign id / root / working dirs',
                          'no screen / nohup / &; attached, alone, both streams captured'],
        'harness_sha256': H.sha256_file(H.HARNESS_PATH), 'launcher_sha256': H.sha256_file(os.path.abspath(__file__)),
        'git_head_at_freeze': H._git(['rev-parse', 'HEAD']), 'frozen_utc': _utc(),
    }


def freeze_spec(started):
    tag = 'W86-V29'
    failures = []
    existing = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_V29_PREFIX))
    if existing:
        failures.append(f'v29 already exists (write-once): {existing}')
    smoke_root_rel = os.path.relpath(campaign_root('smoke'), REPO)
    gate_rel = os.path.join(smoke_root_rel, SMOKE_GATE_FILE)
    man_rel = os.path.join(smoke_root_rel, SMOKE_MANIFEST_FILE)
    smoke_pin = None
    if not os.path.isfile(_abs(gate_rel)):
        failures.append(f'smoke gate not found: {gate_rel}')
    else:
        gate = _load(gate_rel)
        states = {r: _git_state(r) for r in (gate_rel, man_rel)}
        smoke_pin = {'path': gate_rel, 'sha256': _sha(gate_rel), 'manifest': man_rel, 'manifest_sha256': _sha(man_rel),
                     'pass': gate.get('pass'), 'failing': gate.get('failing'), 'campaign_spec_sha256':
                     gate.get('campaign_spec_sha256'), 'solve_claim': gate.get('solve_claim'), 'git': states}
        if not gate.get('pass') or not all(s['git_tracked'] and s['git_clean'] for s in states.values()):
            failures.append(f'smoke gate not PASS / not committed: {smoke_pin}')
    more, evidence = _common_checks('recert')
    failures += more
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    content = v29_content(smoke_pin, evidence['references'])
    text = json.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_V29_PREFIX}{sha[:8]}.json')
    with open(_abs(rel), 'x') as handle:
        handle.write(text)
    if _sha(rel) != sha:
        raise RuntimeError('v29 written bytes do not hash to the name')
    guard_failures = PARENT_GUARD.verify(0)
    _log(f'[{tag}] wrote {rel} sha256={sha} (predecessor {content["predecessor"]})')
    _log(f"[{tag}] pre-launch assertion holds={content['pre_launch_assertion']['holds']}")
    for label, v in content['pre_launch_assertion']['per_cell'].items():
        _log(f"[{tag}]   {label}: {v['recomputed_now']} == expected: {v['recomputed_equals_expected']}")
    for label, r in content['references'].items():
        _log(f"[{tag}]   reference {label}: Q {r['Q_gross']} bar {r['bar']} cert {r['certification_cycle']} "
             f"terminal step {r['terminal_gross_step_abs']} ratio {r['terminal_gross_step_over_threshold']}")
    _log(f'[{tag}] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures} wall={time.time() - started:.1f}s')
    PARENT_GUARD.uninstall()
    sys.exit(0 if (not guard_failures and content['pre_launch_assertion']['holds']) else 1)


# ======================================================================================================================
#  the re-certification run (NOT RUN IN W86)
# ======================================================================================================================
def compare(label, rec, ref, eval_dir):
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl')) \
        if os.path.isfile(os.path.join(eval_dir, 'per_cycle_record.jsonl')) else []
    g = [r['gross_operational_cost'] for r in rows]
    step = abs(g[-1] - g[-2]) if len(g) >= 2 else None
    tol = rows[-1].get('objective_tolerance') if rows else None
    certified = rec.get('status') == 'certified'
    q = rec.get('certified_cost') if certified else None
    bar = (rec.get('bar') or {}).get('value')
    dq = (q - ref['Q_gross']) if q is not None else None
    e_term = (step + ref['terminal_gross_step_abs']) if step is not None else None
    out = {
        'objective_convention': 'gross_operational_cost (settlement excluded)',
        'Q_tail': q, 'Q_ref': ref['Q_gross'], 'dQ': dq, 'dQ_relative': (dq / ref['Q_gross']) if dq is not None else None,
        'bar_tail': bar, 'bar_ref': ref['bar'],
        'cert_cycle_tail': rec.get('certification_cycle'), 'cert_cycle_ref': ref['certification_cycle'],
        'terminal_step_tail': step, 'terminal_step_ref': ref['terminal_gross_step_abs'],
        'terminal_step_over_threshold_tail': (step / tol) if (step is not None and tol) else None,
        'terminal_step_over_threshold_ref': ref['terminal_gross_step_over_threshold'],
        'terminal_step_error_bar': e_term,
        'terminal_step_resolution': (abs(dq) / e_term) if (dq is not None and e_term) else None,
        'two_run_bar_resolution': (abs(dq) / (ref['bar'] + bar)) if (dq is not None and bar is not None) else None,
        'own_bar_ratio': (abs(dq) / ref['bar']) if dq is not None else None,
        'author_band_ratio': (abs(dq) / (1e-6 * abs(ref['Q_gross']))) if dq is not None else None,
    }
    out['determinate_beyond_stopping_slack'] = (out['terminal_step_resolution'] is not None
                                                and out['terminal_step_resolution'] > 1)
    return out


def run_recert(started, spec_sha256):
    tag = 'W86-RECERT'
    root, spec_path, spec, failures, evidence, memory, v29 = _run_preconditions('recert', spec_sha256, tag)
    pre = pre_launch_assertion(spec)
    _log(f"[{tag}] PRE-LAUNCH ASSERTION: holds={pre['holds']} "
         + '; '.join(f"{k} {v['frozen_entry_eval_key']}" for k, v in pre['per_cell'].items()))
    if not pre['holds']:
        failures.append(f'PRE-LAUNCH ASSERTION FAILED: {pre}')
    if v29 is None or v29[0] is None or spec['extra'].get('spec_v29') != {'path': v29[0], 'sha256': v29[1]}:
        failures.append(f'v29 pin mismatch: {v29} vs {spec["extra"].get("spec_v29")}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    refs = _load(v29[0])['references']
    labels = list(STAGES['recert']['labels'])
    _log(f'[{tag}] {STAGE_TEXT}; spec {os.path.relpath(spec_path, REPO)} {spec_sha256}; v29 {v29[1]}')
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f'[{tag}] campaign lock acquired: {lock}; one batch {labels} at concurrency {spec["concurrency"]}')
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate(labels, ctx)
        batch_info = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log(f'[{tag}] campaign lock released')
    by_label = {(r or {}).get('candidate_label'): r for r in records}
    per_cell, gates_all, fallback_cells = {}, True, []
    for e in spec['candidates']:
        label = e['label']
        rec = dict(by_label.get(label) or {})
        eval_dir = os.path.join(root, 'evals', e['eval_dir'])
        exit_code = (rec.get('parent_view') or {}).get('exit_code')
        g = {'G1_harness_clean': (exit_code == 0 and os.path.isfile(os.path.join(eval_dir, 'evaluation_record.json'))
                                  and rec.get('status') in ('certified', 'not_certified')),
             'G2_eval_key': rec.get('eval_key') == EXPECTED_EVAL_KEYS[label]}
        detail = {}
        try:
            c, detail = evaluation_checks(eval_dir, rec, e['working_dir_ids']['run'])
            g.update({'G3_append_reconcile': c.get('append_reconciles_byte_identical', False),
                      'G4_tail_state_check': c.get('tail_state_check_matches', False),
                      'G5_solve_profile_reconciled_per_event': c.get('solve_profile_reconciled_per_event', False),
                      'G6_floor_records': c.get('floor_records_at_compl_inf_tol_in_force', False),
                      'G7_append_sealed': c.get('append_sealed_after_reconcile', False),
                      'G8_post_certification': c.get('post_certification_persisted',
                                                     c.get('post_certification_skipped_uncertified', False)),
                      'G9_ess_ageing_readback': c.get('ess_ageing_readback_all_match', False)})
        except Exception as error:  # noqa: BLE001 -- recorded; the gates fail
            detail = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            g['checks_ran'] = False
        cmp = compare(label, rec, refs[label], eval_dir)
        fails_to_certify = rec.get('status') != 'certified' or not all(g.values())
        moves = cmp['dQ'] is not None and abs(cmp['dQ']) > refs[label]['bar']
        if fails_to_certify or moves:
            fallback_cells.append(label)
        gates_all = gates_all and all(g.values())
        per_cell[label] = {'gates': g, 'gates_pass': all(g.values()), 'detail': detail, 'comparison': cmp,
                           'fallback_test': {'fails_to_certify': fails_to_certify, 'moves_materially': moves},
                           'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
                           'certification_cycle': rec.get('certification_cycle'), 'exit_code': exit_code,
                           'eval_dir': os.path.relpath(eval_dir, REPO), 'wall_time_s': rec.get('wall_time_s'),
                           'peak_rss': rec.get('peak_rss'), 'solve_profile': rec.get('solve_profile')}
    q = {k: per_cell[k]['comparison']['Q_tail'] for k in per_cell}
    r_tail = (q['x0'] - q['n7_4h_e1']) if (q.get('x0') is not None and q.get('n7_4h_e1') is not None) else None
    guard_failures = PARENT_GUARD.verify(0)
    results = {
        'stage': STAGE_TEXT, 'utc': _utc(), 'git_head_at_run': H._git(['rev-parse', 'HEAD']),
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
        'spec_v29': {'path': v29[0], 'sha256': v29[1]}, 'pre_launch_assertion': pre,
        'objective_convention': 'Q gross (settlement excluded) on every table',
        'per_cell': per_cell, 'all_gates_pass': gates_all, 'fallback_triggered': bool(fallback_cells),
        'fallback_cells': fallback_cells,
        'R': {'R_tail': r_tail, 'R_ref': refs['x0']['Q_gross'] - refs['n7_4h_e1']['Q_gross'],
              'dR': (r_tail - (refs['x0']['Q_gross'] - refs['n7_4h_e1']['Q_gross'])) if r_tail is not None else None},
        'solve_claim': 'RECONCILED PER EVENT, NOT GUARD-VERIFIED (see v29 solve_claim)',
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'memory_preflight_at_run': memory, 'batch_info': batch_info, 'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(root, 'campaign_results.json'), results)
    H._write_once_json(os.path.join(root, 'campaign_manifest_sha256.json'), _manifest(root))
    PARENT_GUARD.uninstall()
    for label, v in per_cell.items():
        _log(f"[{tag}] {label}: status {v['status']} cert {v['certification_cycle']} gates {v['gates_pass']} "
             f"{v['comparison']} fallback {v['fallback_test']}")
    _log(f"[{tag}] all gates {gates_all}; fallback {results['fallback_triggered']} {fallback_cells}; R {results['R']}")
    if guard_failures or not gates_all:
        sys.exit(1)
    sys.exit(2 if fallback_cells else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=sorted(STAGES))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--freeze-spec', action='store_true')
    parser.add_argument('--spec-sha256', default=None)
    args = parser.parse_args()
    started = time.time()
    if args.freeze_spec:
        if args.stage or args.spec_sha256:
            parser.error('--freeze-spec takes no --stage / --spec-sha256')
        freeze_spec(started)
    elif args.freeze:
        if not args.stage or args.spec_sha256:
            parser.error('--freeze requires --stage and no --spec-sha256')
        freeze(args.stage, started)
    else:
        if not args.stage or not args.spec_sha256:
            parser.error('--run requires --stage and --spec-sha256')
        if args.stage == 'smoke':
            run_smoke(started, args.spec_sha256)
        else:
            run_recert(started, args.spec_sha256)


if __name__ == '__main__':
    main()
