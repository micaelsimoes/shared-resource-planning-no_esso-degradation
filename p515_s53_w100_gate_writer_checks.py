"""P5.15 Addendum 52, Planner task W100 -- gate-result hygiene: ZERO-SOLVE build of the hash-pinned historical list and
verification of the shared writer (`gate_result_io`), the migrated writers and the repository-wide boolean-typing test
(`p515_gate_result_bool_typing_test.py`). A `SolveProfileGuard(permitted=())` is armed BEFORE any project import and
verified at exactly 0 at the end of each mode. Nothing committed is modified, deleted or re-run onto.

MODE --build-registry (run once, before the checks): the census of every *.json / *.jsonl under data/ with >= 1 string
flag (the test's own scanner and flag definition), each file recorded with path, sha256, tracked, the commit that
produced it (the last commit touching the path; its blob is verified equal to the working file) and the commit that
first added it, the per-field counts, and a family / note from the declared FAMILY_RULES (an unclassified file
REFUSES the build: every entry is judged, none is exempted by directory). Written once to
gate_result_bool_typing_historical.json (refuses to overwrite).

MODE (default) -- checks, results written once under OUT:
  C0 no campaign running: no campaign / gate / benchmark lock file; no p515_s4* / p515_g_ / p515_s53_ / ipopt process
     other than this one.
  C1 writer behaviour: (a) GRIO.json_default == H._json_default == G._json_default, GRIO.json_default_item ==
     BENCH._json_default on a type battery (return value and json text); (b) the refusal matrix -- spellings refused
     and not refused, at every position (top, dict value, list / tuple element, deep, via the default hook), keys never
     refused; (c) a refused payload writes nothing (StringIO and a real file both empty); (d) numpy bools are written
     as JSON booleans (strict `is True` after load) under both hooks; (e) the defect itself: json.dumps(numpy bool,
     default=str) writes the string, GRIO.dumps with the SAME default=str refuses it.
  C2 byte identity, old vs new serialisation: (a) a synthetic battery of non-boolean values (numpy float16/32/64,
     int8/32/64, uint8, ndarray, set, frozenset, tuple, Path, datetime, Decimal, complex, bytes, None, nested) through
     every migrated call form (indent=1; compact separators; jsonl dumps; indent=1 sort_keys) -- old hook vs new
     writer byte-identical; (b) committed artifacts of the migrated writers (declared REPRESENTATIVE, the W98 stage-1
     eval dir and the 3 x 3 pair x0 eval dir): the declared call form reproduces the committed bytes from the parsed
     content, and the new writer gives the same bytes; (c) the W98 hooks' committed defect lines: numpy bools put back
     at the string-flag positions, the OLD hook (default=str) reproduces each committed line exactly and the new writer
     differs only at those tokens (JSON booleans).
  C3 confined change: for each migrated file, the pre-W100 blob (git HEAD, sha256-pinned) with the migration applied
     in the AST (json.dump(s) -> GRIO.dump(s), declared default hook -> GRIO hook, skipping calls nested in json.loads)
     equals the new file's AST with the one added import removed; site counts equal the declared counts; every
     remaining json.dump(s) call is in the declared NOT_MIGRATED table (file, enclosing function, reason), exactly.
  C4 evaluation_key: the top-level names whose source changed are exactly the declared set (`evaluation_key`,
     `candidate_key`, `canonical_candidate`, `ess_ageing_canonical_text`, `freeze_campaign_spec` and `_json_default` not
     among them); for every entry of every committed campaign spec, the new harness's `evaluation_key` equals the
     pre-W100 harness's (loaded from git) and the frozen `eval_key` (reported).
  C5 hash pins: (a) every committed file naming the pre-W100 sha256 of a modified file (enumeration) and the pre-W100
     blob resolving from git to that sha256 (how such pins are re-verified); (b) every `inspect.getsource(<module>.<fn>)`
     in a *.py whose function changed; (c) the hash-feeding serialisations re-derive committed hashes byte-identically
     with the current code: frozen stage spec v38 (8bc0ffa6, launcher form with H._json_default), campaign specs
     c2b02e21 / 231558f0 (harness freeze form, default=str).
  C6 the repository-wide test: PASS on the current tree; FAIL (exactly F1 on the planted file) with a planted new
     artifact carrying string flags written by plain json (the recurrence), the planted file then removed; FAIL
     (exactly F2 on the perturbed entry) with a copy of the list whose one pinned sha256 is altered (the listed artifact
     itself is never touched); PASS again after the planted file is removed.
  C7 the three defect families are on the list and detected: objective_convergence, determinate_at_gt_error_bar,
     pass (post_certification.json), reconciles_to_net (W98 hooks).

COMMANDS (repo root, canonical interpreter, attached, both streams, noclobber):
  set -o noclobber
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w100_gate_writer_checks.py \
      --build-registry > data/SRP1/Results/P515S53/w100_registry_build_launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w100_gate_writer_checks.py \
      > data/SRP1/Results/P515S53/w100_gate_writer_checks_r2_launch.log 2>&1
  (attempt 1 wrote data/SRP1/Results/P515S53/w100_gate_writer/ and w100_gate_writer_checks_launch.log; kept as run)
"""

import ast
import datetime as _dt
import decimal
import hashlib
import importlib.util
import inspect
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W100 gate-writer checks (never solves)').install()

import numpy as np  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_gate_result_bool_typing_test as T  # noqa: E402

STAGE = 'P5.15 Addendum 52 W100 -- gate-result hygiene: one shared writer, string flags refused, repository-wide test'
SCRIPT_NAME = os.path.basename(__file__)
# r2: attempt 1 (root w100_gate_writer, log w100_gate_writer_checks_launch.log) is kept as run: its final record was
# REFUSED by gate_result_io (a bare repr(True) in C1) and C2 / C4 carried two defects of this script; nothing re-run onto it.
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w100_gate_writer_r2')
OUT = os.path.join(REPO, OUT_REL)
REGISTRY = T.DEFAULT_REGISTRY
PRE_COMMIT = '58ff8d88'   # HEAD before W100 (the four migrated files unchanged since their last commits)
PRE_SHA256 = {
    'p515_s44_campaign_harness.py': 'f7bcfac6677457d46ba588df86d57ac22327be4038a94d0d4943bbb048b995f2',
    'p515_g_g1_g4_admm_gates.py': 'e147a34f1be39119196845059eaa77b581148f6f191cc3197cd9275c4b0ad5aa',
    'p515_s53_w98_continuation_hooks.py': '9920f7d6d569fc78c2e4c638c619ccf3f2a09ccbac137833802de61b99717680',
    'p515_s53_w93_uncoordinated_benchmark.py': 'e65dc3ec82462d0ab639944f7b1dd1118bc44b37fcab420842c151922cbb3698',
}
# (old default expression, new default expression, declared number of migrated sites)
MIGRATION = {
    'p515_s44_campaign_harness.py': ('_json_default', 'GRIO.json_default', 8),
    'p515_g_g1_g4_admm_gates.py': ('_json_default', 'GRIO.json_default', 21),
    'p515_s53_w98_continuation_hooks.py': ('str', 'GRIO.json_default', 1),
    'p515_s53_w93_uncoordinated_benchmark.py': ('_json_default', 'GRIO.json_default_item', 2),
}
ADDED_IMPORT = 'import gate_result_io as GRIO'
# Every json.dump(s) call LEFT in a migrated file: (file, enclosing function) -> (count, reason). Checked exactly.
NOT_MIGRATED = {
    ('p515_s44_campaign_harness.py', 'candidate_key'): (1, 'hash: candidate key (sha256 of canonical JSON)'),
    ('p515_s44_campaign_harness.py', 'ess_ageing_canonical_text'): (1, 'hash input: canonical text compared / keyed'),
    ('p515_s44_campaign_harness.py', 'evaluation_key'): (8, 'hash: evaluation key payloads'),
    ('p515_s44_campaign_harness.py', 'freeze_campaign_spec'): (1, 'hash: frozen campaign spec text, sha8 in its name'),
    ('p515_s44_campaign_harness.py', '_flex_price_objective_summary'): (2, 'hash: sha256 of objective term lists'),
    ('p515_s44_campaign_harness.py', 'encode_curtailment_entries'): (1, 'in-memory grouping signature (not written)'),
    ('p515_s44_campaign_harness.py', 'validate_derived_instance'): (1, 'in-memory deep copy (json round trip)'),
    ('p515_s44_campaign_harness.py', 'write_response_terminal'): (3, 'in-memory: the round-trip VERIFICATION of the '
                                                                     'written file (expected text and comparison)'),
    ('p515_s44_campaign_harness.py', 'ConvergenceDepthAppender.reconcile'): (1, 'in-memory round trip compared with '
                                                                                'the rebuilt tail state'),
    ('p515_s44_campaign_harness.py', 'register_initialisation_identity'): (1, 'identity record, no default= hook (a '
                                                                              'numpy value raises, cannot stringify); '
                                                                              'compared across sibling evaluations'),
    ('p515_s44_campaign_harness.py', 'acquire_campaign_lock'): (1, 'lock file content'),
    ('p515_s44_campaign_harness.py', 'acquire_terminal_phase_lock'): (1, 'lock file content'),
    ('p515_g_g1_g4_admm_gates.py', '_hash_consensus_ess_z'): (1, 'hash: consensus ESS z digest'),
    ('p515_s53_w93_uncoordinated_benchmark.py', 'acquire_lock'): (1, 'lock file content'),
}
DECLARED_CHANGED_TOP_LEVEL = {
    'p515_s44_campaign_harness.py': {'_atomic_write_json', '_write_once_json', '_write_once_json_compact',
                                     'alpha_row_run_hooks', 'persist_convergence_depth_capture',
                                     'ConvergenceDepthAppender', '_child_real'},
    'p515_g_g1_g4_admm_gates.py': {'_atomic_write_json', '_capture_esso_solve', '_scan_and_write_network_failures',
                                   'esso_capture_hooks', 'run_admm_arm', 's34_capture_hooks', 's35ref_capture_hooks',
                                   's35ref_replay_cycle0_lmp_hooks', 's38_pf_capture_hooks',
                                   's39_exempt_until_capture_hooks', 'write_boyd_terminal_s32',
                                   'write_boyd_terminal_s33e2', 'write_boyd_terminal_s34', 'write_boyd_terminal_s35pt',
                                   'write_boyd_terminal_s35ref', 'write_component_levels_terminal',
                                   'write_interface_settlement_detail_s31c', 'write_interface_voltage_terminal',
                                   'write_terminal_storage_duals_s35ref_replay'},
    'p515_s53_w98_continuation_hooks.py': {'ContinuationState'},
    'p515_s53_w93_uncoordinated_benchmark.py': {'_write_json_once', '_append_jsonl'},
}
KEY_FUNCTIONS = ('evaluation_key', 'candidate_key', 'canonical_candidate', 'ess_ageing_canonical_text',
                 'freeze_campaign_spec', '_json_default', '_entry_eval_key', 'eval_dir_name', 'eval_ids')
# C2 (b): committed artifacts of the migrated writers, with the call form each writer uses (declared; reproduction of
# the committed bytes from the parsed content is the evidence the form is right).
W98_EVAL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w98_continuation', 'campaign_s53_w98_x0_continuation_r2',
                        'evals', '25b92ae0f1f2c02e_x0_cont')
PAIR_EVAL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals',
                         'f6e9cd53fdbb8ee8_x0')
FORM_INDENT1 = ('json', {'indent': 1})
FORM_COMPACT = ('json', {'separators': (',', ':')})
FORM_LINES = ('jsonl', {})
REPRESENTATIVE = []
for _d in (W98_EVAL, PAIR_EVAL):
    REPRESENTATIVE += [
        (os.path.join(_d, 'evaluation_record.json'), 'harness._write_once_json', FORM_INDENT1),
        (os.path.join(_d, 'convergence_depth_tail_state.json'), 'harness._write_once_json', FORM_INDENT1),
        (os.path.join(_d, 'response_terminal.json'), 'harness._write_once_json_compact', FORM_COMPACT),
        (os.path.join(_d, 'per_cycle_record.jsonl'), 'harness._child_real', FORM_LINES),
        (os.path.join(_d, 'per_cycle_response.jsonl'), 'harness.alpha_row_run_hooks (response sidecar)', FORM_LINES),
        (os.path.join(_d, 'network_ipopt_solve_records.jsonl'), 'harness.persist_convergence_depth_capture', FORM_LINES),
        (os.path.join(_d, 'network_ipopt_solve_records_append.jsonl'), 'harness.ConvergenceDepthAppender', FORM_LINES),
        (os.path.join(_d, 'convergence_depth_append_events.jsonl'), 'harness.ConvergenceDepthAppender', FORM_LINES),
        (os.path.join(_d, 'g_s39_D.json'), 'gates (g_<label>.json)', FORM_INDENT1),
        (os.path.join(_d, 'boyd_terminal.json'), 'gates (boyd_terminal.json)', FORM_INDENT1),
        (os.path.join(_d, 'component_levels_terminal.json'), 'gates (component_levels_terminal.json)', FORM_INDENT1),
        (os.path.join(_d, 'interface_settlement_detail_s31c.json'), 'gates (s31c settlement)', FORM_INDENT1),
        (os.path.join(_d, 'interface_voltage_terminal.json'), 'gates (s33e2 voltage)', FORM_INDENT1),
        (os.path.join(_d, 'recourse_jump_sidecar_baseline.jsonl'), 'gates (recourse jump sidecar)', FORM_LINES),
        (os.path.join(_d, 'ess_entry_stride_baseline.jsonl'), 'gates (ESS stride)', FORM_LINES),
        (os.path.join(_d, 'soh_floor_sidecar_baseline.jsonl'), 'gates (SOH floor sidecar)', FORM_LINES),
        (os.path.join(_d, 'pf_entry_stride_s39_D.jsonl'), 'gates (PF stride)', FORM_LINES),
        (os.path.join(_d, 'ess_exempt_until_state_s39_D.jsonl'), 'gates (exempt-until state)', FORM_LINES),
        (os.path.join(_d, 'leak_classification_s39_D.jsonl'), 'gates (leak classification)', FORM_LINES),
    ]
REPRESENTATIVE += [
    (os.path.join(W98_EVAL, 'continuation_cycle_record.jsonl'), 'hooks.ContinuationState._write', FORM_LINES),
    (os.path.join(W98_EVAL, 'recourse_blocks_all.jsonl'), 'hooks.ContinuationState._write', FORM_LINES),
    (os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w98_continuation', 'campaign_s53_w98_x0_continuation_r2',
                  'campaign_results.json'), 'harness._write_once_json (W98 launcher)', FORM_INDENT1),
]
HOOKS_FILES = {'continuation_cycle_record.jsonl', 'recourse_blocks_all.jsonl'}
SPEC_V38 = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'frozen_s53_spec_v38_8bc0ffa6.json')
CAMPAIGN_SPECS = [
    os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w98_continuation', 'campaign_s53_w98_x0_continuation_r2',
                 'campaign_spec_s53_w98_x0_continuation_r2_c2b02e21.json'),
    os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair',
                 'campaign_spec_s53_w91_3x3_pair_231558f0.json'),
]
LOCK_FILES = ('.p515_s44_campaign.lock', '.p515_g_gate.lock', '.p515_s53_w93_benchmark.lock')
# Every historical entry is classified by exactly one rule (first match); an unmatched file refuses the build.
FAMILY_RULES = [
    (r'^data/SRP1/Results/P513[CD]/state_[^/]+\.json$', 'deliberate_repr',
     'Pyomo v.fixed flags serialised by repr() (p513_c_param_move_gate.py / p513_e_gated_capture.py; W75)'),
    (r'^data/SRP1/Results/(FrozenSMOPF/P45/p45_report|P5/p5_report)\.json$', 'log_text_parse',
     'warm_start regex group parsed from log text (p45_seed2026_smoke_test.py; W75)'),
    (r'^data/SRP1/Results/P515S31C/zero_solve_checks\.json$', 'defect_default_str',
     'numpy comparisons written with default=str (p515_s31c_zero_solve_checks.py; W75)'),
    (r'^data/SRP1/Results/P515S36/A17_partA_dual_reconstruction/parta_dual_reconstruction_results\.json$',
     'defect_default_str', 'numpy comparisons written with default=str (p515_s36_a17_parta_dual_reconstruction.py; W75)'),
    (r'^data/SRP1/Results/P515S45/case_file_aa/check\.json$', 'deliberate_fixture',
     "negative-control INPUT: a string-typed 'enabled' the AA loader must refuse (Addendum 27 W1)"),
    (r'^data/SRP1/Results/P515S47/case_file_baseline/(before|after)/case_file_baseline_(before|after)\.json$',
     'deliberate_repr', 'repr() state serialisation (p515_s47_case_file_baseline_check.py; W75)'),
    (r'^data/SRP1/Results/(P515S47/tso_marginal_cost/tso_marginal_cost|P515S49/flex_pq_split/flex_pq_split)\.json$',
     'defect_default_str', 'from_warm_start through a str() default hook (p515_s47_tso_marginal_cost.py / '
                           'p515_s49_flex_pq_split.py; W75)'),
    (r'^data/SRP1/Results/(P515S51/2x2_limit_gate|P515S52/campaign_s52_pilot[a-z_]*|P515S53/alpha_row/'
     r'campaign_s53_alpha_row[a-z0-9_]*)/.*(g_s51limit_[a-z0-9_.]+|gate|g_s39_D|boyd_terminal)\.json$',
     'defect_default_str', 'numpy bool through p515_g_g1_g4_admm_gates.py default=str before W76 '
                           '(objective_convergence / determinate_at_gt_error_bar; W75)'),
    (r'^data/SRP1/Results/(P515S52/campaign_s52_pilot[a-z_]*|P515S53/alpha_row/campaign_s53_alpha_row[a-z0-9_]*)/'
     r'evals/[^/]+/post_certification\.json$', 'defect_default_str',
     'hull_polish_full.gate.pass numpy bool through p515_s44_campaign_harness.py default=str before W74 (W74/W75)'),
    (r'^data/SRP1/Results/P515S53/alpha_row/recompute_w72/alpha_row_recompute\.json$', 'pass_through',
     'full_gate_pass copied from the post_certification.json strings, type recorded alongside '
     '(p515_s53_alpha_row_recompute.py; W75)'),
    (r'^data/SRP1/Results/P515S53/gates_json_default_w76/[^/]+\.json$', 'deliberate_fixture',
     "W76 negative controls: the pre-W76 hook's output and its record (p515_s53_w76_gates_json_default_checks.py)"),
    (r'^data/SRP1/Results/P515S53/harness_fix_w74/harness_fix_w74_checks\.json$', 'deliberate_fixture',
     'W74 negative-control fixture of the pre-W74 hook (p515_s53_w74_harness_fix_checks.py)'),
    (r'^data/SRP1/Results/P515S53/polish_feasibility_w74/polish_feasibility_w74\.json$', 'deliberate_repr',
     'repr() of the raw committed gate value (p515_s53_w74_polish_feasibility.py)'),
    (r'^data/SRP1/Results/P515S53/w98_continuation/campaign_s53_w98_x0_continuation_r2/evals/[^/]+/'
     r'(continuation_cycle_record|recourse_blocks_all)\.jsonl$', 'defect_default_str',
     'numpy bool through p515_s53_w98_continuation_hooks.py default=str: reconciles_to_net (the G14 failure) and '
     'early_stop.step_qualifies (W98 / W99)'),
    (r'^data/SRP1/Results/P515S53/w99_stage1_posthoc/w99_stage1_posthoc_analysis\.json$', 'pass_through',
     'W99 records the stringified G14 flag values it analysed (p515_s53_w99_stage1_posthoc_analysis.py)'),
]
T0 = time.time()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W100 +{time.time() - T0:7.1f}s] {msg}', flush=True)


def _git(args, binary=False):
    out = subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, check=True).stdout
    return out if binary else out.decode().strip()


def _sha(rel):
    return T.sha256_file(os.path.join(REPO, rel))


def _write_once(path, obj):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    with open(path, 'x') as handle:
        GRIO.dump(obj, handle, indent=1)
    return path


# ======================================================================================================================
#  --build-registry
# ======================================================================================================================
def build_registry():
    if os.path.exists(REGISTRY):
        raise RuntimeError(f'refusing to overwrite {REGISTRY}')
    files = T.list_files()
    tracked = T.tracked_files()
    entries, unclassified = [], []
    for i, rel in enumerate(files):
        if i % 20000 == 0:
            _log(f'census {i}/{len(files)}')
        res = T.scan_file(rel)
        if not res['flags']:
            continue
        fam = [(f, n) for pat, f, n in FAMILY_RULES if re.search(pat, rel)]
        if not fam:
            unclassified.append(rel)
            continue
        family, note = fam[0]
        is_tracked = rel in tracked
        commit = first = None
        if is_tracked:
            if _git(['status', '--porcelain', '--', rel]):
                raise RuntimeError(f'tracked historical artifact is not clean: {rel}')
            commit = _git(['log', '-1', '--format=%H', '--', rel])
            first = _git(['log', '--diff-filter=A', '--format=%H', '--', rel]).splitlines()[-1]
            if _git(['rev-parse', f'{commit}:{rel}']) != _git(['hash-object', rel]):
                raise RuntimeError(f'blob at {commit} differs from the working file: {rel}')
        entries.append({'path': rel, 'sha256': res['sha256'], 'tracked': is_tracked, 'commit': commit,
                        'first_added_commit': first, 'fields': T.field_counts(res['flags']),
                        'n_string_flags': len(res['flags']), 'parse': res['parse'], 'family': family, 'note': note})
    if unclassified:
        raise RuntimeError(f'unclassified files with string flags (judge each; no directory exemption): {unclassified}')
    registry = {
        'schema': T.REGISTRY_SCHEMA,
        'purpose': ('Known HISTORICAL artifacts carrying string flags. Committed artifacts are never modified, so the '
                    'boolean-typing test passes on these files ONLY while each one hashes to its pinned sha256. A file '
                    'not listed here, or a listed file whose sha256 changed, is judged fresh (the test fails). Add an '
                    'entry only by an explicit, reviewed decision -- never by directory.'),
        'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 52; Planner task W100',
        'flag_definition': ('a JSON value (object member value or array element, any depth; keys excluded) that is a '
                            'string whose strip().lower() is "true" or "false" (gate_result_io.is_string_flag)'),
        'field_convention': ('innermost object key above the value (array elements inherit it); Pyomo index-repr keys '
                             '(starting with "(", a digit or "-") are counted as "<index-key>"; fields maps field -> '
                             '{flag text: count}'),
        'scan_scope': ('every *.json / *.jsonl under data/ in the working tree, tracked and untracked; untracked '
                       'entries are machine-local (commit null) and are reported, not failed, when absent'),
        'commit_convention': ('commit = the last commit touching the path (its blob verified equal to the working file '
                              'at build time); first_added_commit = the commit that added the path'),
        'built_by': {'script': SCRIPT_NAME, 'script_sha256': _sha(SCRIPT_NAME),
                     'test_module_sha256': _sha('p515_gate_result_bool_typing_test.py'),
                     'writer_module_sha256': _sha('gate_result_io.py'), 'git_head': _git(['rev-parse', 'HEAD']),
                     'utc': _utc(), 'files_scanned': len(files)},
        'counts': {'entries': len(entries), 'tracked': sum(e['tracked'] for e in entries),
                   'untracked': sum(not e['tracked'] for e in entries),
                   'by_family': {f: sum(e['family'] == f for e in entries) for f in sorted({e['family'] for e in entries})},
                   'string_flags': sum(e['n_string_flags'] for e in entries)},
        'entries': entries,
    }
    _write_once(REGISTRY, registry)
    _log(f"registry written: {len(entries)} entries ({registry['counts']['tracked']} tracked); "
         f"sha256 {T.sha256_file(REGISTRY)}")
    return registry


# ======================================================================================================================
#  C0 -- no campaign running
# ======================================================================================================================
def check_c0():
    locks = {p: os.path.exists(os.path.join(REPO, p)) for p in LOCK_FILES}
    ps = subprocess.run(['ps', 'axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me = {str(os.getpid()), str(os.getppid())}
    live = [line.strip() for line in ps.splitlines()
            if any(s in line for s in ('p515_s4', 'p515_g_', 'p515_s53_', 'ipopt'))
            and line.split()[0] not in me and SCRIPT_NAME not in line]
    return {'id': 'C0_no_campaign', 'lock_files_present': locks, 'live_processes': live,
            'holds': not any(locks.values()) and not live}


# ======================================================================================================================
#  C1 -- writer behaviour
# ======================================================================================================================
class _Obj:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


def _battery():
    return [np.bool_(True), np.bool_(False), np.float16(1.5), np.float32(0.1), np.float64(0.1), np.int8(-3),
            np.int32(7), np.int64(2 ** 40), np.uint8(200), np.array([1.0, 2.0]), np.array(3.5), np.array([True]),
            {1, 2}, frozenset({'a'}), pathlib.Path('a/b'), _dt.datetime(2026, 9, 27, 12, 0),
            _dt.date(2026, 9, 27), decimal.Decimal('1.10'), complex(1, 2), b'xy', _Obj('custom'), None, 1, 1.5,
            'text', True, [np.int64(1), (np.float32(2.0),)], {'k': np.int64(5)}]


def check_c1():
    import p515_s44_campaign_harness as H
    import p515_g_g1_g4_admm_gates as G
    import p515_s53_w93_uncoordinated_benchmark as BENCH
    out = {'id': 'C1_writer_behaviour'}
    # (a) hook equivalence
    eq = []
    for x in _battery():
        row = {'value': f'{type(x).__name__}:{x!r}'[:60]}   # type-prefixed: a bare repr(True) would be a string flag
        native = isinstance(x, (str, int, float, type(None), list, dict))
        if not native:
            row['json_default_eq_H'] = repr(GRIO.json_default(x)) == repr(H._json_default(x))
            row['json_default_eq_G'] = repr(GRIO.json_default(x)) == repr(G._json_default(x))
            row['json_default_item_eq_BENCH'] = repr(GRIO.json_default_item(x)) == repr(BENCH._json_default(x))
        row['text_eq_H'] = (json.dumps([x], default=H._json_default) == json.dumps([x], default=GRIO.json_default))
        row['text_eq_BENCH'] = (json.dumps([x], default=BENCH._json_default)
                                == json.dumps([x], default=GRIO.json_default_item))
        eq.append(row)
    out['a_hook_equivalence'] = eq
    a_ok = all(all(v for k, v in r.items() if k != 'value') for r in eq)
    # (b) refusal matrix
    refused_spellings = ['True', 'False', 'true', 'false', 'TRUE', 'FaLsE', ' True', 'False\n', '\tfalse ',
                         'True ']
    kept_spellings = ['Truex', 'untrue', "'True'", 'yes', '1', 'T', '', 'True False', 'none', 'None']
    matrix = []

    def _refuses(payload, default=GRIO.json_default):
        try:
            GRIO.dumps(payload, default=default)
            return False
        except GRIO.StringFlagError:
            return True
    for s in refused_spellings + kept_spellings:
        positions = {'top': s, 'dict_value': {'flag': s}, 'list_element': [1, s], 'tuple_element': (s,),
                     'deep': {'a': [{'b': ({'c': [s]},)}]}, 'via_default_hook': _Obj(s)}
        res = {pos: _refuses(p) for pos, p in positions.items()}
        res['via_item_hook_set'] = _refuses({s}, default=GRIO.json_default_item)
        res['as_key_only'] = _refuses({s: 1})
        matrix.append({'spelling': repr(s), 'expected_refused': s in refused_spellings, 'refused': res})
    b_ok = all(all(v == m['expected_refused'] for k, v in m['refused'].items() if k != 'as_key_only')
               and m['refused']['as_key_only'] is False for m in matrix)
    b_ok = b_ok and not _refuses([True, False, np.bool_(True), None, 0, 1.0])
    b_ok = b_ok and _refuses(np.array(True))            # 0-d numpy bool array: str() -> "True" -> refused
    out['b_refusal_matrix'] = matrix
    # (c) nothing written on refusal
    sio = io.StringIO()
    try:
        GRIO.dump({'ok': 1.0, 'reconciles_to_net': 'True'}, sio, indent=1)
        c_raised = False
    except GRIO.StringFlagError as error:
        c_raised = True
        c_flags = [[p, f, repr(v)] for p, f, v in error.flags]
    tmpdir = tempfile.mkdtemp(prefix='w100_c1_')
    fpath = os.path.join(tmpdir, 'refused.json')
    try:
        with open(fpath, 'x') as handle:
            GRIO.dump({'rows': [{'objective_convergence': 'False'}]}, handle, indent=1)
    except GRIO.StringFlagError:
        pass
    file_bytes = os.path.getsize(fpath)
    shutil.rmtree(tmpdir)
    out['c_nothing_written'] = {'raised': c_raised, 'flags_reported': c_flags if c_raised else None,
                                'stringio_after': repr(sio.getvalue()), 'file_bytes_after': file_bytes}
    c_ok = c_raised and sio.getvalue() == '' and file_bytes == 0
    # (d) numpy bools as JSON booleans, strict reads
    payload = {'objective_convergence': np.float64(1.0) < np.float64(2.0),
               'determinate_at_gt_error_bar': np.float64(3.0) < np.float64(2.0),
               'hull_polish_full': {'gate': {'pass': np.bool_(True)}}, 'reconciles_to_net': np.bool_(True)}
    d = {}
    for name, hook in (('json_default', GRIO.json_default), ('json_default_item', GRIO.json_default_item)):
        back = json.loads(GRIO.dumps(payload, default=hook))
        d[name] = {'objective_convergence_is_True': back['objective_convergence'] is True,
                   'determinate_is_False': back['determinate_at_gt_error_bar'] is False,
                   'pass_is_True': back['hull_polish_full']['gate']['pass'] is True,
                   'reconciles_is_True': back['reconciles_to_net'] is True}
    out['d_numpy_bools_written_as_booleans'] = d
    d_ok = all(all(v.values()) for v in d.values())
    # (e) the defect: default=str
    old = json.dumps({'reconciles_to_net': np.bool_(True)}, default=str)
    try:
        GRIO.dumps({'reconciles_to_net': np.bool_(True)}, default=str)
        e_refused = False
    except GRIO.StringFlagError:
        e_refused = True
    out['e_defect_default_str'] = {'old_json_dumps_text': old, 'old_writes_string': old == '{"reconciles_to_net": "True"}',
                                   'grio_with_default_str_refuses': e_refused}
    e_ok = old == '{"reconciles_to_net": "True"}' and e_refused
    out['holds_parts'] = {'a': a_ok, 'b': b_ok, 'c': c_ok, 'd': d_ok, 'e': e_ok}
    out['holds'] = all(out['holds_parts'].values())
    return out


# ======================================================================================================================
#  C2 -- byte identity
# ======================================================================================================================
def _ser(obj, form, default, new):
    kind, kw = form
    if kind == 'jsonl':
        return (GRIO.dumps(obj, default=default, **kw) if new else json.dumps(obj, default=default, **kw)) + '\n'
    sio = io.StringIO()
    (GRIO.dump if new else json.dump)(obj, sio, default=default, **kw)
    return sio.getvalue()


def _synthetic_payload():
    return {'floats': [np.float16(1.5), np.float32(0.1), np.float64(0.1), float('nan'), float('inf'), -0.0, 1e-300],
            'ints': [np.int8(-3), np.int32(7), np.int64(2 ** 40), np.uint8(200), 10 ** 20],
            'arrays': [np.array([1.0, 2.0]), np.array(3.5)], 'sets': [{1, 2}, frozenset({'a'})], 'tuple': (1, 'x'),
            'misc': [pathlib.Path('a/b'), _dt.datetime(2026, 9, 27, 12, 0), decimal.Decimal('1.10'), complex(1, 2),
                     b'xy', _Obj('custom'), None, 'text', '', 'café'],
            'nested': {'a': [{'b': np.int64(1)}], 'c': {'d': (np.float32(2.0),)}}, 'order': {'z': 1, 'a': 2}}


def check_c2():
    import p515_s44_campaign_harness as H
    import p515_s53_w93_uncoordinated_benchmark as BENCH
    out = {'id': 'C2_byte_identity'}
    # (a) synthetic battery through every call form
    forms = {'indent1': ('json', {'indent': 1}), 'compact': ('json', {'separators': (',', ':')}),
             'jsonl': ('jsonl', {}), 'indent1_sorted': ('json', {'indent': 1, 'sort_keys': True}),
             'jsonl_sorted': ('jsonl', {'sort_keys': True})}
    syn = {}
    for name, form in forms.items():
        for label, old_hook, new_hook in (('harness_gates', H._json_default, GRIO.json_default),
                                          ('hooks_old_str', str, GRIO.json_default),
                                          ('benchmark', BENCH._json_default, GRIO.json_default_item)):
            if label == 'benchmark' and 'sort_keys' not in form[1]:
                continue
            if label != 'benchmark' and 'sort_keys' in form[1]:
                continue
            a = _ser(_synthetic_payload(), form, old_hook, new=False)
            b = _ser(_synthetic_payload(), form, new_hook, new=True)
            syn[f'{label}/{name}'] = {'identical': a == b, 'bytes': len(a.encode()),
                                      'sha256': hashlib.sha256(a.encode()).hexdigest()}
    out['a_synthetic'] = syn
    a_ok = all(v['identical'] for v in syn.values()) and len(syn) == 8
    # (b) committed artifacts
    tracked = T.tracked_files()
    arts = []
    for rel, writer, form in REPRESENTATIVE:
        rec = {'path': rel, 'writer': writer, 'form': form[0] + json.dumps(form[1], sort_keys=True),
               'tracked': rel in tracked, 'sha256': _sha(rel)}
        raw = open(os.path.join(REPO, rel), 'rb').read()
        text = raw.decode('utf-8')
        is_hooks = os.path.basename(rel) in HOOKS_FILES
        old_hook = str if is_hooks else H._json_default
        if form[0] == 'jsonl':
            lines = text.splitlines(keepends=True)
            n_repro = n_same = n_flag_lines = n_flag_ok = 0
            for line in lines:
                obj = json.loads(line)
                flags = GRIO.find_string_flags(obj)
                if flags:
                    n_flag_lines += 1
                    n_flag_ok += int(_flag_line_ok(obj, line, old_hook))
                    continue
                old = _ser(obj, form, old_hook, new=False)
                new = _ser(obj, form, GRIO.json_default, new=True)
                n_repro += int(old == line)
                n_same += int(old == new)
            rec.update({'lines': len(lines), 'flag_free_lines': len(lines) - n_flag_lines,
                        'old_reproduces_committed_line': n_repro, 'new_equals_old': n_same,
                        'string_flag_lines': n_flag_lines, 'string_flag_lines_old_exact_new_booleans': n_flag_ok})
            rec['ok'] = (n_repro == n_same == len(lines) - n_flag_lines and n_flag_ok == n_flag_lines
                         and (n_flag_lines == 0 or is_hooks))
        else:
            obj = json.loads(text)
            if GRIO.find_string_flags(obj):
                rec.update({'ok': False, 'error': 'representative artifact carries string flags'})
            else:
                old = _ser(obj, form, old_hook, new=False)
                new = _ser(obj, form, GRIO.json_default, new=True)
                rec.update({'old_reproduces_committed_bytes': old.encode() == raw, 'new_equals_old': old == new,
                            'bytes': len(raw)})
                rec['ok'] = old.encode() == raw and old == new
        arts.append(rec)
        _log(f"C2 {rel.split('/')[-1]}: ok={rec['ok']}")
    out['b_committed_artifacts'] = arts
    b_ok = all(r['ok'] for r in arts)   # tracked or not, each artifact's sha256 is recorded here
    hooks = [r for r in arts if os.path.basename(r['path']) in HOOKS_FILES]
    out['c_hooks_defect_lines'] = [{k: r.get(k) for k in ('path', 'lines', 'string_flag_lines',
                                                           'string_flag_lines_old_exact_new_booleans')} for r in hooks]
    c_ok = all(r['string_flag_lines'] > 0 and r['string_flag_lines_old_exact_new_booleans'] == r['string_flag_lines']
               for r in hooks) and len(hooks) == 2
    out['holds_parts'] = {'a': a_ok, 'b': b_ok, 'c': c_ok}
    out['holds'] = all(out['holds_parts'].values())
    return out


def _put_numpy_bools(obj):
    if isinstance(obj, dict):
        return {k: _put_numpy_bools(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_put_numpy_bools(v) for v in obj]
    if GRIO.is_string_flag(obj):
        if obj not in ('True', 'False'):
            raise ValueError(f'unexpected flag spelling {obj!r}')
        return np.bool_(obj == 'True')
    return obj


def _flag_line_ok(obj, line, old_hook):
    """A committed defect line: numpy bools at the string-flag positions; the OLD hook reproduces the line exactly and
    the new writer's line differs from it only at those tokens (JSON booleans), strict-reading as bools."""
    restored = _put_numpy_bools(obj)
    old = json.dumps(restored, default=old_hook) + '\n'
    new = GRIO.dumps(restored, default=GRIO.json_default) + '\n'
    as_bools = json.loads(new)
    expect = old.replace('"True"', 'true').replace('"False"', 'false')
    return old == line and new == expect and not GRIO.find_string_flags(as_bools)


# ======================================================================================================================
#  C3 -- confined change (AST)
# ======================================================================================================================
def _pre_source(rel):
    raw = _git(['show', f'{PRE_COMMIT}:{rel}'], binary=True)
    got = hashlib.sha256(raw).hexdigest()
    if got != PRE_SHA256[rel]:
        raise RuntimeError(f'{rel}: pre-W100 blob sha256 {got} != pinned {PRE_SHA256[rel]}')
    return raw.decode('utf-8')


def _migrate_ast(tree, old_default, new_default):
    parents = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parents[ch] = node
    n = 0
    for node in list(ast.walk(tree)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'json'
                and node.func.attr in ('dump', 'dumps')):
            continue
        kw = [k for k in node.keywords if k.arg == 'default']
        if not kw or ast.unparse(kw[0].value) != old_default:
            continue
        p = parents.get(node)
        if isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute) and p.func.attr == 'loads':
            continue
        node.func.value = ast.Name(id='GRIO', ctx=ast.Load())
        kw[0].value = ast.parse(new_default, mode='eval').body
        n += 1
    return n


def _strip_added_import(tree):
    before = len(tree.body)
    tree.body = [s for s in tree.body if not (isinstance(s, ast.Import) and ast.unparse(s) == ADDED_IMPORT)]
    return before - len(tree.body)


def _json_calls(tree):
    enc = {}

    def visit(n, stack):
        for ch in ast.iter_child_nodes(n):
            s = stack + [ch.name] if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else stack
            enc[ch] = '.'.join(s) or '<module>'
            visit(ch, s)
    visit(tree, [])
    calls = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and n.func.value.id in ('json', 'GRIO') and n.func.attr in ('dump', 'dumps')):
            kw = [ast.unparse(k.value) for k in n.keywords if k.arg == 'default']
            calls.append({'module': n.func.value.id, 'fn': n.func.attr, 'line': n.lineno, 'enclosing': enc[n],
                          'default': kw[0] if kw else None})
    return calls


def _top_level_segments(src):
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = ast.dump(node)
    return out


def check_c3():
    out = {'id': 'C3_confined_change', 'files': {}}
    ok = True
    for rel, (old_d, new_d, n_declared) in MIGRATION.items():
        pre_src, new_src = _pre_source(rel), open(os.path.join(REPO, rel)).read()
        pre_tree, new_tree = ast.parse(pre_src), ast.parse(new_src)
        n_migrated = _migrate_ast(pre_tree, old_d, new_d)
        n_import = _strip_added_import(new_tree)
        identical = ast.dump(pre_tree) == ast.dump(new_tree)
        calls = _json_calls(ast.parse(new_src))
        grio = [c for c in calls if c['module'] == 'GRIO']
        left = [c for c in calls if c['module'] == 'json']
        grouped = {}
        for c in left:
            grouped.setdefault(c['enclosing'], []).append(c['line'])
        declared = {fn: cnt for (f, fn), (cnt, _r) in NOT_MIGRATED.items() if f == rel}
        left_ok = {fn: len(v) for fn, v in grouped.items()} == declared
        pre_segments, new_segments = _top_level_segments(pre_src), _top_level_segments(new_src)
        changed = sorted(n for n in set(pre_segments) & set(new_segments) if pre_segments[n] != new_segments[n])
        added = sorted(set(new_segments) - set(pre_segments))
        removed = sorted(set(pre_segments) - set(new_segments))
        decl_changed = DECLARED_CHANGED_TOP_LEVEL[rel]
        rec = {'pre_commit': PRE_COMMIT, 'pre_sha256': PRE_SHA256[rel], 'new_sha256': _sha(rel),
               'sites_migrated_in_ast': n_migrated, 'sites_declared': n_declared, 'added_import_removed': n_import,
               'ast_identical_after_migration': identical,
               'grio_calls_in_new_file': [[c['line'], c['enclosing'], c['fn'], c['default']] for c in grio],
               'json_calls_left': {fn: lines for fn, lines in sorted(grouped.items())},
               'json_calls_left_match_declared_table': left_ok,
               'not_migrated_reasons': {fn: NOT_MIGRATED[(rel, fn)][1] for fn in sorted(declared)},
               'changed_top_level': changed, 'added_top_level': added, 'removed_top_level': removed,
               'key_functions_unchanged': not (set(KEY_FUNCTIONS) & set(changed)),
               'retained_default_hook_unchanged': (pre_segments.get('_json_default') == new_segments.get('_json_default')
                                                   if '_json_default' in pre_segments else None)}
        rec['holds'] = (identical and n_migrated == n_declared == len(grio) and n_import == 1 and left_ok
                        and not added and not removed and rec['key_functions_unchanged']
                        and (decl_changed is None or set(changed) == decl_changed)
                        and rec['retained_default_hook_unchanged'] in (True, None))
        ok = ok and rec['holds']
        out['files'][rel] = rec
    out['holds'] = ok
    return out


# ======================================================================================================================
#  C4 -- evaluation_key
# ======================================================================================================================
def _entry_key_args(spec, e):
    """Arguments of `evaluation_key` for one frozen spec entry (the argument extraction of
    p515_s53_w98_continuation_checks._entry_key_args, verbatim, plus the continuation declaration)."""
    cfg = spec.get('configuration') or {}
    overrides = e.get('overrides') if 'overrides' in e else (cfg.get('overrides') or {})
    return (e['key'], overrides), dict(
        case_file_aa=cfg.get('case_file_anderson_acceleration'), model_variant=e.get('model_variant'),
        ess_ageing_baseline=cfg.get('ess_ageing_baseline'), flex_price_multiplier=e.get('flex_price_multiplier'),
        derived_instance=cfg.get('derived_instance'), interface_deviation_premium=e.get('interface_deviation_premium'),
        convergence_depth_tail=cfg.get('convergence_depth_tail'),
        certification_continuation=e.get('certification_continuation'))


def _load_pre_harness():
    src = _pre_source('p515_s44_campaign_harness.py').encode()
    tmp = tempfile.mkdtemp(prefix='w100_pre_harness_')
    path = os.path.join(tmp, '_w100_pre_harness.py')
    with open(path, 'wb') as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location('_w100_pre_harness', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shutil.rmtree(tmp)
    return mod


def check_c4():
    import p515_s44_campaign_harness as H
    pre = _load_pre_harness()
    n = n_eq_pre = n_frozen = n_frozen_eq = 0
    mism, frozen_mism, errors = [], [], []
    specs = sorted(p for p in _git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip())
    for rel in specs:
        spec = json.load(open(os.path.join(REPO, rel)))
        for e in spec.get('candidates') or []:
            n += 1
            args, kw = _entry_key_args(spec, e)
            try:
                new = H.evaluation_key(*args, **kw)
                old = pre.evaluation_key(*args, **kw)
            except Exception as error:  # noqa: BLE001
                errors.append({'spec': rel, 'label': e.get('label'), 'error': f'{type(error).__name__}: {error}'})
                continue
            if new == old:
                n_eq_pre += 1
            else:
                mism.append({'spec': rel, 'label': e.get('label')})
            if 'eval_key' in e:
                n_frozen += 1
                if new == e['eval_key']:
                    n_frozen_eq += 1
                else:
                    frozen_mism.append({'spec': rel, 'label': e.get('label'), 'frozen': e['eval_key'][:16],
                                        'recomputed': new[:16]})
    pre_src = _pre_source('p515_s44_campaign_harness.py')
    pre_lines = pre_src.splitlines()
    pre_seg = {n.name: '\n'.join(pre_lines[n.lineno - 1:n.end_lineno]) for n in ast.parse(pre_src).body
               if isinstance(n, ast.FunctionDef)}   # whole lines (a trailing comment is part of the source)
    src_eq = {fn: inspect.getsource(getattr(H, fn)).rstrip('\n') == pre_seg.get(fn) for fn in KEY_FUNCTIONS
              if hasattr(pre, fn)}
    # the certified cells named in the reports
    pair = json.load(open(os.path.join(REPO, CAMPAIGN_SPECS[1])))
    cont = json.load(open(os.path.join(REPO, CAMPAIGN_SPECS[0])))
    named = {}
    for spec, rel in ((pair, CAMPAIGN_SPECS[1]), (cont, CAMPAIGN_SPECS[0])):
        for e in spec['candidates']:
            args, kw = _entry_key_args(spec, e)
            named[f"{os.path.basename(rel)}:{e['label']}"] = {'frozen': e['eval_key'],
                                                              'recomputed': H.evaluation_key(*args, **kw)}
    named_ok = all(v['frozen'] == v['recomputed'] for v in named.values())
    return {'id': 'C4_evaluation_key', 'committed_specs': len(specs), 'entries': n, 'new_equals_pre_w100': n_eq_pre,
            'mismatches_vs_pre_w100': mism, 'errors': errors, 'entries_with_frozen_eval_key': n_frozen,
            'recomputed_equals_frozen': n_frozen_eq, 'not_rebuilt_from_spec_alone_reported': frozen_mism,
            'key_function_source_identical': src_eq, 'named_cells': named,
            'holds': (not mism and not errors and n_eq_pre == n and all(src_eq.values()) and named_ok)}


# ======================================================================================================================
#  C5 -- hash pins
# ======================================================================================================================
def check_c5():
    import p515_s44_campaign_harness as H
    out = {'id': 'C5_hash_pins'}
    pins = {}
    for rel, sha in PRE_SHA256.items():
        proc = subprocess.run(['git', 'grep', '-l', sha, 'HEAD', '--', '.'], cwd=REPO, capture_output=True, text=True)
        hits = proc.stdout.split() if proc.returncode == 0 else []
        pins[rel] = {'pre_sha256': sha, 'pre_blob_resolves_from_git': hashlib.sha256(
            _git(['show', f'{PRE_COMMIT}:{rel}'], binary=True)).hexdigest() == sha,
            'committed_files_naming_pre_sha256': [h.split(':', 1)[1] for h in hits],
            'current_sha256': _sha(rel)}
    out['a_pins_of_modified_files'] = pins
    a_ok = all(v['pre_blob_resolves_from_git'] for v in pins.values())
    # (b) getsource references to changed functions
    changed = {}
    for rel in MIGRATION:
        pre_s, new_s = _top_level_segments(_pre_source(rel)), _top_level_segments(open(os.path.join(REPO, rel)).read())
        changed[rel] = sorted(n for n in pre_s if n in new_s and pre_s[n] != new_s[n])
    alias = {'p515_s44_campaign_harness.py': 'p515_s44_campaign_harness', 'p515_g_g1_g4_admm_gates.py':
             'p515_g_g1_g4_admm_gates', 'p515_s53_w98_continuation_hooks.py': 'p515_s53_w98_continuation_hooks',
             'p515_s53_w93_uncoordinated_benchmark.py': 'p515_s53_w93_uncoordinated_benchmark'}
    refs = []
    rx = re.compile(r'getsource\(\s*(\w+)\.(\w+)')
    for py in sorted(p for p in _git(['ls-files', '*.py']).splitlines() if '/' not in p):
        text = open(os.path.join(REPO, py), errors='replace').read()
        imports = dict(re.findall(r'^import (\w+) as (\w+)', text, flags=re.M))
        for m in rx.finditer(text):
            modname = next((k for k, v in imports.items() if v == m.group(1)), None)
            for rel, mod in alias.items():
                if modname == mod and m.group(2) in changed[rel]:
                    line = text[:m.start()].count('\n') + 1
                    refs.append({'file': py, 'line': line, 'target': f'{rel}:{m.group(2)}',
                                 'text': text.splitlines()[line - 1].strip()[:160]})
    out['b_getsource_refs_to_changed_functions'] = refs
    out['b_changed_top_level'] = changed
    # (c) hash-feeding serialisations re-derive committed hashes
    rederive = {}
    content = json.load(open(os.path.join(REPO, SPEC_V38)))
    text = json.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    raw = open(os.path.join(REPO, SPEC_V38), 'rb').read()
    rederive[SPEC_V38] = {'bytes_identical': text.encode() == raw,
                          'sha8': hashlib.sha256(text.encode()).hexdigest()[:8], 'name_sha8': '8bc0ffa6'}
    for rel in CAMPAIGN_SPECS:
        spec = json.load(open(os.path.join(REPO, rel)))
        text = json.dumps(spec, indent=1, sort_keys=True, default=str)
        raw = open(os.path.join(REPO, rel), 'rb').read()
        rederive[rel] = {'bytes_identical': text.encode() == raw, 'sha8': hashlib.sha256(text.encode()).hexdigest()[:8],
                         'name_sha8': rel.rsplit('_', 1)[1][:8]}
    out['c_hash_feeding_rederived'] = rederive
    c_ok = all(v['bytes_identical'] and v['sha8'] == v['name_sha8'] for v in rederive.values())
    out['holds_parts'] = {'a_pre_blobs_resolve': a_ok, 'c_rederived': c_ok}
    out['holds'] = a_ok and c_ok
    return out


# ======================================================================================================================
#  C6 / C7 -- the repository-wide test and its negative controls
# ======================================================================================================================
def _run_test(tag, extra_args=()):
    path = os.path.join(OUT, f'test_{tag}.json')
    code = T.main(['--quiet', '--out', path] + list(extra_args))
    res = json.load(open(path))
    _log(f"test {tag}: exit {code}, verdict {res['verdict']}, failures {res['failure_counts']}, "
         f"wall {res['wall_s']:.1f} s")
    return code, res, os.path.relpath(path, REPO)


def check_c6():
    out = {'id': 'C6_repository_test'}
    code, res, p = _run_test('pass_before')
    out['pass_before'] = {'exit': code, 'verdict': res['verdict'], 'result': p, 'scope': res['scope'],
                          'known_historical_matched': res['known_historical_matched'],
                          'listed_untracked_absent': res['listed_untracked_absent']}
    ok_pass = code == 0 and res['verdict'] == 'PASS'
    # NC1: a planted NEW artifact written by plain json (the recurrence)
    planted_dir = os.path.join(OUT, 'negative_control_planted')
    planted = os.path.join(planted_dir, 'planted_string_flag.json')
    os.makedirs(planted_dir)
    payload = {'gate': 'G14', 'reconciles_to_net': np.bool_(True), 'objective_convergence': np.bool_(False),
               'nested': {'pass': np.bool_(True)}}
    with open(planted, 'x') as handle:
        json.dump(payload, handle, default=str)          # deliberately NOT through gate_result_io
    planted_sha = T.sha256_file(planted)
    planted_rel = os.path.relpath(planted, REPO)
    try:
        code1, res1, p1 = _run_test('negative_control_planted')
    finally:
        os.remove(planted)
        os.rmdir(planted_dir)
    f1 = res1['failures']['F1_new_string_flag']
    others1 = {k: v for k, v in res1['failure_counts'].items() if k != 'F1_new_string_flag'}
    out['negative_control_planted'] = {
        'planted': planted_rel, 'planted_sha256': planted_sha, 'written_with': 'json.dump(payload, default=str)',
        'exit': code1, 'verdict': res1['verdict'], 'result': p1, 'F1': f1, 'other_failure_counts': others1,
        'planted_removed_after': not os.path.exists(planted_dir)}
    ok_nc1 = (code1 == 1 and res1['verdict'] == 'FAIL' and len(f1) == 1 and f1[0]['path'] == planted_rel
              and not any(others1.values()) and not os.path.exists(planted_dir))
    # NC2: a copy of the list with one pinned sha256 altered (the artifact itself is never touched)
    reg = json.load(open(REGISTRY))
    target = next(e for e in reg['entries'] if e['path'].endswith('recourse_blocks_all.jsonl'))
    tmpdir = tempfile.mkdtemp(prefix='w100_nc2_')
    perturbed = os.path.join(tmpdir, 'registry_perturbed.json')
    old_sha = target['sha256']
    target['sha256'] = old_sha[:-1] + ('0' if old_sha[-1] != '0' else '1')
    with open(perturbed, 'x') as handle:
        GRIO.dump(reg, handle, indent=1)
    try:
        code2, res2, p2 = _run_test('negative_control_hash_changed', ['--registry', perturbed])
    finally:
        shutil.rmtree(tmpdir)
    f2 = res2['failures']['F2_hash_changed']
    others2 = {k: v for k, v in res2['failure_counts'].items() if k != 'F2_hash_changed'}
    out['negative_control_hash_changed'] = {
        'entry': target['path'], 'pinned_sha256': old_sha, 'perturbed_sha256_in_copy': target['sha256'],
        'artifact_sha256_now': _sha(target['path']), 'exit': code2, 'verdict': res2['verdict'], 'result': p2,
        'F2': f2, 'other_failure_counts': others2}
    ok_nc2 = (code2 == 1 and res2['verdict'] == 'FAIL' and len(f2) == 1 and f2[0]['path'] == target['path']
              and f2[0]['sha256'] == old_sha and not any(others2.values()) and _sha(target['path']) == old_sha)
    code3, res3, p3 = _run_test('pass_after')
    out['pass_after'] = {'exit': code3, 'verdict': res3['verdict'], 'result': p3}
    ok_pass2 = code3 == 0 and res3['verdict'] == 'PASS'
    out['holds_parts'] = {'pass_before': ok_pass, 'nc1_planted_fails': ok_nc1, 'nc2_hash_changed_fails': ok_nc2,
                          'pass_after': ok_pass2}
    out['holds'] = all(out['holds_parts'].values())
    return out


def check_c7():
    reg = json.load(open(REGISTRY))
    fields = {}
    for e in reg['entries']:
        for f, c in e['fields'].items():
            fields.setdefault(f, {'files': 0, 'values': 0})
            fields[f]['files'] += 1
            fields[f]['values'] += sum(c.values())
    fam = {'objective_convergence': 'g_<label>.json cycle_trajectory (gates, pre-W76)',
           'determinate_at_gt_error_bar': 'boyd_terminal.json (gates, pre-W76)',
           'pass': 'post_certification.json hull_polish_full.gate.pass (harness, pre-W74)',
           'reconciles_to_net': 'recourse_blocks_all.jsonl (W98 hooks, G14)'}
    post_cert = [e['path'] for e in reg['entries'] if e['path'].endswith('post_certification.json')]
    return {'id': 'C7_families_detected', 'fields_on_list': fields, 'families': fam,
            'post_certification_entries': post_cert,
            'holds': all(f in fields for f in fam) and len(post_cert) > 0}


# ======================================================================================================================
def main():
    mode_build = '--build-registry' in sys.argv[1:]
    status = 1
    try:
        if mode_build:
            build_registry()
            status = 0
        else:
            if os.path.exists(OUT):
                raise RuntimeError(f'refusing to reuse output root (write-once): {OUT}')
            if not os.path.exists(REGISTRY):
                raise RuntimeError('registry missing: run --build-registry first')
            os.makedirs(OUT)
            results = {'stage': STAGE, 'utc': _utc(), 'git_head': _git(['rev-parse', 'HEAD']),
                       'script': SCRIPT_NAME, 'script_sha256': _sha(SCRIPT_NAME), 'interpreter': sys.executable,
                       'module_sha256': {r: _sha(r) for r in ['gate_result_io.py', 'p515_gate_result_bool_typing_test.py',
                                                              os.path.basename(REGISTRY)] + list(MIGRATION)},
                       'checks': {}}
            for name, fn in (('C0', check_c0), ('C1', check_c1), ('C2', check_c2), ('C3', check_c3),
                             ('C4', check_c4), ('C5', check_c5), ('C6', check_c6), ('C7', check_c7)):
                _log(f'{name} ...')
                try:
                    results['checks'][name] = fn()
                except Exception as error:  # noqa: BLE001
                    results['checks'][name] = {'holds': False, 'error': f'{type(error).__name__}: {error}',
                                               'traceback': traceback.format_exc()}
                _log(f"{name} holds={results['checks'][name].get('holds')}")
            results['summary'] = {k: v.get('holds') for k, v in results['checks'].items()}
            status = 0 if all(results['summary'].values()) else 1
            results['guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
            results['wall_s'] = time.time() - T0
            _write_once(os.path.join(OUT, 'w100_gate_writer_checks.json'), results)
            manifest = {os.path.relpath(os.path.join(dp, f), REPO): T.sha256_file(os.path.join(dp, f))
                        for dp, _dn, fn in os.walk(OUT) for f in sorted(fn)}
            manifest[os.path.relpath(REGISTRY, REPO)] = T.sha256_file(REGISTRY)
            _write_once(os.path.join(OUT, 'w100_gate_writer_manifest_sha256.json'), manifest)
            _log(f"summary {results['summary']}")
    finally:
        failures = GUARD.verify(0)
        _log(f'guard counts {GUARD.counts}; verify(0) failures {failures}')
        if failures:
            status = 1
    _log(f'exit {status}')
    return status


if __name__ == '__main__':
    sys.exit(main())
