"""
P5.15 Addendum 46 ruling 7, Planner task W84 items 2-3 -- ZERO-SOLVE verification of the campaign-harness changes
(`p515_s44_campaign_harness.py`): per-evaluation persistence of the per-solve IPOPT floor-status records and the
convergence-depth tail state (Planner ruling Q2), and the rule-eleven checklist assertion that records whether the tail
is enabled for a run (Planner ruling Q3). Nothing here solves: `SolveProfileGuard(permitted=())` is armed at import,
before any production module is imported, and verified at EXACTLY 0 on every exit path.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 46 ruling 7; Planner task W84 (rulings Q1-Q4 on W83's questions:
"Q2 -- persist them ... Keep the identity of evaluation_key untouched, and verify that"; "Q3 -- default stays off in
admm_parameters.py; each launcher enables it explicitly. But add a capture-checklist assertion that records whether
the tail was enabled for that run"; item 3: "evaluation_key unchanged (no diff hunk, and recomputed keys for known
cells equal the committed ones); the persisted records round-trip; the checklist assertion fires correctly both when
the tail is enabled and when it is not").

CHECKS (all must hold):
  K  evaluation_key UNCHANGED against BASE_COMMIT (the harness before W84):
     K1 no `git diff BASE_COMMIT` hunk touches the old-side line range of `evaluation_key` (nor of any function it
        calls or that names an evaluation: candidate_key, canonical_candidate, eval_ids, eval_dir_name,
        _entry_eval_key, effective_anderson_acceleration and the six validators / identity helpers it uses);
     K2 those functions' live source is byte-identical to BASE_COMMIT's (AST-extracted);
     K3 for EVERY entry of EVERY git-tracked campaign spec (`data/**/campaign_spec_*.json`): key =
        candidate_key(canonical), eval_key = evaluation_key(...) with the spec's own declarations, eval_dir and
        working_dir_ids -- recomputed by the LIVE harness -- equal the committed values;
     K4 freeze reproduction: `freeze_campaign_spec` (LIVE) on the committed s47_recert spec's own candidates and
        configuration, into a scratch root -- (a) undeclared: every candidate entry identical to the committed one and
        the configuration identical except `case_file_last_commit` / `ess_params_file.last_commit` (reported), no
        `convergence_depth_tail` key; (b) declared enabled: identical candidate entries (the declaration does NOT
        enter the key -- recorded as the consequence), configuration = (a) + exactly `convergence_depth_tail`.
  R  persistence round-trip (`persist_convergence_depth_capture` -> `load_*`):
     R1 records produced by the PRODUCTION capture functions (`network._append_ipopt_solve_record` on the first
        IPOPT segment of a hash-verified W83 gate log, drained by `shared_resources_planning.
        _drain_network_ipopt_solve_records`) and a tail state built by the production tail functions: loaded back
        equal (==) to what was persisted; the reproduced record equals the committed W83 record of the same solve
        (every field but log_path / log_bytes / warm_start);
     R2 the committed W83 gate's 144 records and tail state (hash-verified against its committed manifest):
        loaded back equal, and the records file BYTE-IDENTICAL (sha256) to the committed W83 sidecar;
     R3 tail-off state ({'enabled': False}, no records): both files written, n_records 0, loads back equal;
     R4 write-once: a second persist into the same eval dir raises;
     R5 the summary (n_records, per round, floor tally, parse problems) equals a recount of the loaded file.
  C  the checklist assertion in BOTH states:
     C1 `assert_convergence_depth_tail_capture`: undeclared -> tail_enabled_for_this_run False; declared enabled ->
        True with compl_inf_tol 1e-6; declared disabled -> False; invalid declarations refused; NEGATIVE CONTROL:
        with production's `_run_operational_planning` replaced by a function lacking the capture path, it RAISES;
     C2 the configuration hook (`_config_hook_factory`, on a planning object built by
        `p515_g_g1_g4_admm_gates._construct_arm_planning` -- the gate's / child's own construction, no solve -- with
        the W10 configuration): undeclared -> tail in force False, dict unchanged, checklist recorded in the report;
        declared enabled -> dict in force == declaration, enabled; declared disabled -> dict == declaration, off;
        NEGATIVE CONTROL: tail switched on in the ADMM parameters without a declaration -> the hook RAISES;
     C3 `convergence_depth_tail_state_check`: expected/returned agree -> match; enabled-but-returned-off (a
        forgotten enable after the fact), off-but-returned-on, wrong tail value, no tail state -> match False;
     C4 wiring: `_child_real` asserts the checklist before `run_admm_arm`, persists in the post-run hook before the
        terminal phase, records all four fields, and `main_child` exits 2 on `convergence_depth_capture_error`; the
        error record carries the checklist.

EXACT LAUNCH COMMAND (repo root; attached, both streams captured):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_w84_harness_capture_checks.py --scratch-dir <an EMPTY directory outside the repository> \\
        > data/SRP1/Results/P515S53/tight_tail_w84/harness_checks_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/tight_tail_w84/harness_checks/{checks_w84.json,
checks_w84_manifest_sha256.json}. Exit 0 when every check holds, 1 otherwise.
"""

import argparse
import ast
import copy
import hashlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W84 harness capture zero-solve checks').install()

import p515_s44_campaign_harness as H  # noqa: E402
import network as NET  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402

STAGE = ('P5.15 Addendum 46 ruling 7, W84 items 2-3 -- campaign harness: per-evaluation floor-status / tail-state '
         'persistence (Q2) and the tail-enabled checklist assertion (Q3); evaluation_key unchanged')
BASE_COMMIT = 'eb8cf81f'   # HEAD when W84 started: the harness before the W84 change
HARNESS = 'p515_s44_campaign_harness.py'
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT = os.path.join(S53, 'tight_tail_w84', 'harness_checks')
CHECKS_JSON = os.path.join(OUT, 'checks_w84.json')
CHECKS_MANIFEST = os.path.join(OUT, 'checks_w84_manifest_sha256.json')
PRODUCTION_FILES = ('network.py', 'shared_resources_planning.py', 'admm_parameters.py',
                    'admm_anderson_acceleration.py', 'solver_parameters.py')
KEY_FUNCTIONS = ('evaluation_key', 'candidate_key', 'canonical_candidate', 'eval_ids', 'eval_dir_name',
                 '_entry_eval_key', 'effective_anderson_acceleration', 'validate_model_variant',
                 'validate_ess_ageing_baseline', 'flex_price_multiplier_in_key', 'validate_derived_instance',
                 'derived_instance_identity', 'validate_interface_deviation_premium', 'validate_overrides',
                 'validate_flex_price_multiplier', 'validate_case_file_anderson_acceleration', '_sanitize_id')
S47_SPEC = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert',
                        'campaign_spec_s47_recert_902f93aa.json')
W83_GATE = os.path.join(S53, 'tight_tail_w83', 'srp1_bitwise_gate')
W83_RECORDS = os.path.join(W83_GATE, 'w83_network_ipopt_solve_records.jsonl')
W83_TAIL = os.path.join(W83_GATE, 'w83_tail_state.json')
W83_MANIFEST = os.path.join(W83_GATE, 'w83_manifest_sha256.json')
W83_LOG_HASHES = os.path.join(S53, 'tight_tail_w83', 'srp1_bitwise_gate_arm_logs_sha256.txt')
DECLARED_ON = {'enabled': True, 'compl_inf_tol': 1e-6}
DECLARED_OFF = {'enabled': False, 'compl_inf_tol': 1e-6}


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W84-checks] {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _refuse_overwrite(rel):
    if os.path.exists(_abs(rel)):
        raise SystemExit(f'refusing to overwrite existing artifact: {rel}')


def _raises(fn, exc=Exception):
    try:
        fn()
    except exc as error:  # noqa: BLE001
        return f'{type(error).__name__}: {str(error)[:300]}'
    return None


# ======================================================================================================================
#  K -- evaluation_key unchanged
# ======================================================================================================================
def _function_spans(text):
    tree = ast.parse(text)
    return {node.name: (node.lineno, node.end_lineno) for node in tree.body if isinstance(node, ast.FunctionDef)}


def _function_source(text, name):
    start, end = _function_spans(text)[name]
    return '\n'.join(text.splitlines()[start - 1:end])


def check_k1_k2():
    base_text = _git(['show', f'{BASE_COMMIT}:{HARNESS}'])
    live_text = open(_abs(HARNESS)).read()
    base_spans = _function_spans(base_text)
    diff = _git(['diff', '-U0', BASE_COMMIT, '--', HARNESS])
    hunks = []
    for line in diff.splitlines():
        m = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        if m:
            old_start, old_len = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            # a pure insertion (old_len 0) sits AFTER old line old_start; it touches a function only if strictly inside
            hunks.append({'old_start': old_start, 'old_len': old_len, 'header': line})
    touched = {}
    for name in KEY_FUNCTIONS:
        lo, hi = base_spans[name]
        hit = []
        for h in hunks:
            if h['old_len'] == 0:
                inside = lo <= h['old_start'] < hi
            else:
                inside = not (h['old_start'] + h['old_len'] - 1 < lo or h['old_start'] > hi)
            if inside:
                hit.append(h['header'])
        touched[name] = hit
    identical = {name: _function_source(base_text, name) == _function_source(live_text, name) for name in KEY_FUNCTIONS}
    live_matches_module = {name: inspect.getsource(getattr(H, name)).rstrip('\n') == _function_source(live_text, name)
                           for name in KEY_FUNCTIONS}
    return {
        'base_commit': BASE_COMMIT, 'n_hunks': len(hunks), 'hunk_headers': [h['header'] for h in hunks],
        'base_line_spans': {n: base_spans[n] for n in KEY_FUNCTIONS},
        'hunks_touching_key_functions': touched,
        'k1_no_hunk_touches_a_key_function': all(not v for v in touched.values()),
        'source_identical_to_base': identical,
        'live_module_is_the_file': live_matches_module,
        'k2_sources_identical': all(identical.values()) and all(live_matches_module.values()),
    }


def check_k3():
    paths = sorted(p for p in _git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip())
    per_spec, mismatches, n_entries = {}, [], 0
    for rel in paths:
        spec = json.load(open(_abs(rel)))
        cfg = spec.get('configuration') or {}
        case_file_aa = H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration'))
        ess = H.validate_ess_ageing_baseline(cfg.get('ess_ageing_baseline'))
        derived = H.validate_derived_instance(cfg.get('derived_instance'))
        n_ok = 0
        for entry in spec.get('candidates') or []:
            n_entries += 1
            got = {}
            got['key'] = H.candidate_key(entry['canonical'])
            if 'eval_key' in entry:
                got['eval_key'] = H.evaluation_key(
                    got['key'], entry.get('overrides') or {}, case_file_aa=case_file_aa,
                    model_variant=entry.get('model_variant'), ess_ageing_baseline=ess,
                    flex_price_multiplier=entry.get('flex_price_multiplier'), derived_instance=derived,
                    interface_deviation_premium=entry.get('interface_deviation_premium'))
                got['entry_eval_key'] = H._entry_eval_key(entry)
            if 'eval_dir' in entry:
                got['eval_dir'] = H.eval_dir_name(got.get('eval_key', got['key']), entry['label'])
            if 'working_dir_ids' in entry:
                got['working_dir_ids'] = H.eval_ids(spec['campaign_id'], got.get('eval_key', got['key']))
            want = {'key': entry.get('key'), 'eval_key': entry.get('eval_key'),
                    'entry_eval_key': entry.get('eval_key'),
                    'eval_dir': entry.get('eval_dir'), 'working_dir_ids': entry.get('working_dir_ids')}
            bad = {k: {'committed': want[k], 'recomputed': v} for k, v in got.items() if v != want[k]}
            if bad:
                mismatches.append({'spec': rel, 'label': entry.get('label'), 'fields': bad})
            else:
                n_ok += 1
        per_spec[rel] = {'sha256': _sha(rel), 'n_entries': len(spec.get('candidates') or []), 'n_ok': n_ok,
                         'declares': sorted(k for k in ('case_file_anderson_acceleration', 'ess_ageing_baseline',
                                                        'derived_instance') if cfg.get(k) is not None)}
    return {'n_specs': len(paths), 'n_entries': n_entries, 'n_mismatches': len(mismatches),
            'mismatches': mismatches[:20], 'per_spec': per_spec,
            'k3_every_committed_key_recomputed_equal': len(paths) > 0 and n_entries > 0 and not mismatches}


def _s47_candidates(spec):
    out = []
    for entry in spec['candidates']:
        canon = entry['canonical']
        nodes = {int(n): tuple(v) for n, v in canon['nodes'].items()}
        out.append((entry['label'], nodes, {'investment_year': canon['investment_year']}))
    return out


def check_k4(scratch):
    spec = json.load(open(_abs(S47_SPEC)))
    cfg = spec['configuration']
    configuration = {'name': cfg['name'], 'arm_label': cfg['arm_label'], 'overrides': cfg['overrides'],
                     'case_file_anderson_acceleration': cfg['case_file_anderson_acceleration'],
                     'ess_ageing_baseline': cfg['ess_ageing_baseline'],
                     'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'], 'note': cfg['note']}
    results = {}
    for tag, extra_cfg in (('undeclared', {}), ('declared_on', {'convergence_depth_tail': dict(DECLARED_ON)})):
        root = os.path.join(scratch, f'k4_{tag}', 's47_recert')
        _path, _sha256, frozen = H.freeze_campaign_spec(
            root, spec['campaign_id'], _s47_candidates(spec), {**configuration, **extra_cfg}, spec['cap'],
            spec['concurrency'], ['W84 zero-solve freeze reproduction (scratch, not a campaign)'],
            required_consecutive_cycles=spec['required_consecutive_cycles'])
        entries_equal = frozen['candidates'] == spec['candidates']
        fc = frozen['configuration']
        differing = sorted(k for k in set(fc) | set(cfg) if fc.get(k) != cfg.get(k))
        results[tag] = {'candidates_identical_to_committed': entries_equal,
                        'eval_keys': {e['label']: e['eval_key'] for e in frozen['candidates']},
                        'configuration_keys_differing_from_committed': differing,
                        'differing_values': {k: {'committed': cfg.get(k), 'frozen_now': fc.get(k)} for k in differing},
                        'has_convergence_depth_tail': 'convergence_depth_tail' in fc,
                        'convergence_depth_tail': fc.get('convergence_depth_tail')}
    allowed_drift = {'case_file_last_commit', 'ess_params_file'}
    und, dec = results['undeclared'], results['declared_on']
    und_ok = (und['candidates_identical_to_committed'] and not und['has_convergence_depth_tail']
              and set(und['configuration_keys_differing_from_committed']) <= allowed_drift
              and all(k != 'ess_params_file' or (und['differing_values'][k]['committed'] or {}).get('sha256')
                      == (und['differing_values'][k]['frozen_now'] or {}).get('sha256')
                      for k in und['configuration_keys_differing_from_committed']))
    dec_ok = (dec['candidates_identical_to_committed'] and dec['convergence_depth_tail'] == DECLARED_ON
              and set(dec['configuration_keys_differing_from_committed'])
              == set(und['configuration_keys_differing_from_committed']) | {'convergence_depth_tail'})
    return {'results': results, 'allowed_provenance_drift': sorted(allowed_drift),
            'k4a_undeclared_reproduces_the_committed_spec': und_ok,
            'k4b_declared_adds_only_the_declaration_and_keeps_the_keys': dec_ok,
            'consequence_recorded': ('the tail declaration is NOT in evaluation_key: a tail-enabled C* evaluation '
                                     'under the s47_recert declarations has eval_key '
                                     f"{dec['eval_keys'].get('c_star')} -- the committed s47_recert C* eval_key")}


# ======================================================================================================================
#  planning object (the gate's / child's construction, no solve)
# ======================================================================================================================
def _build_planning(scratch):
    import p56a_oracle as O
    import p515_g_g1_g4_admm_gates as G
    work = os.path.join(scratch, 'P56A_evals')
    os.makedirs(work, exist_ok=True)
    original = O.WORK_DIR
    O.WORK_DIR = work    # fresh_planning's logs dir -> the scratch dir, never the repository
    try:
        report = {}
        canonical = H.canonical_candidate({5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)})
        stream = io.StringIO()
        with redirect_stdout(stream):
            planning, sed, candidate = G._construct_arm_planning(
                'w84_hookcheck', os.path.join(scratch, 'arm_out'), report, k_override=None,
                investment_map=H.investment_map_from_canonical(canonical), eval_id='w84_hookcheck_run',
                num_max_iters_override=2, apply_rho=False)
    finally:
        O.WORK_DIR = original
    return planning, sed, candidate


# ======================================================================================================================
#  R -- persistence round-trip
# ======================================================================================================================
class _FakeSolver:
    """Only `.options` is read by `network._append_ipopt_solve_record`; no solve method exists on it."""

    def __init__(self, options):
        self.options = dict(options)


def _verify_w83_inputs():
    manifest = json.load(open(_abs(W83_MANIFEST)))
    ok = {rel: manifest.get(rel) == _sha(rel) for rel in (W83_RECORDS, W83_TAIL)}
    log_hashes = {}
    for line in open(_abs(W83_LOG_HASHES)):
        parts = line.split()
        if len(parts) >= 2:
            log_hashes[parts[-1]] = parts[0]
    return ok, log_hashes


def check_r(scratch, planning):
    results = {}
    inputs_ok, log_hashes = _verify_w83_inputs()
    results['w83_inputs_match_committed_manifest'] = inputs_ok
    w83_records = H.load_network_ipopt_solve_records(_abs(W83_RECORDS))
    w83_tail = json.load(open(_abs(W83_TAIL)))

    # R1: production-made records and tail state
    first = next(r for r in w83_records if r['agent'] == 'TSO' and r['round'] == 0 and r['day'] == 'Spring')
    log_rel = os.path.relpath(first['log_path'], REPO)
    log_sha_committed = next((v for k, v in log_hashes.items() if k.endswith(log_rel) or log_rel.endswith(k)), None)
    log_sha_now = H.sha256_file(first['log_path']) if os.path.isfile(first['log_path']) else None
    segment_path = os.path.join(scratch, 'r1_segment_optim_log_case9_2025_Spring.log')
    with open(first['log_path'], 'rb') as src, open(segment_path, 'wb') as dst:
        dst.write(src.read()[first['log_bytes'][0]:first['log_bytes'][1]])
    tso = planning.transmission_network
    net = tso.network[first['year']][first['day']]
    _stale = srp._drain_network_ipopt_solve_records(planning, None)
    # the options production passes: `network._create_smopf_solver` builds them (it creates the solver object and
    # never calls it); only `.options` is handed on, with the IPOPT output file replaced by the scratch segment.
    solver, _prod_log_path, _ctx = NET._create_smopf_solver(net, None, tso.params)
    passed = dict(solver.options)
    NET._append_ipopt_solve_record(net, tso.params, _FakeSolver(passed), segment_path, 0, None, False)
    drained = srp._drain_network_ipopt_solve_records(planning, 0)
    baseline = srp._capture_convergence_depth_tail_baseline(planning, _admm_with(DECLARED_ON))
    per_cycle = srp._apply_convergence_depth_tail(planning, _admm_with(DECLARED_ON), False, baseline, 1)
    restore = srp._apply_convergence_depth_tail(planning, _admm_with(DECLARED_ON), False, baseline, None)
    per_cycle['aa_off_predicate_end_of_cycle'] = False
    tail_state = {'enabled': True, 'compl_inf_tol_tail': 1e-6, 'option': srp.CONVERGENCE_DEPTH_TAIL_OPTION,
                  'baseline': baseline, 'per_cycle': [per_cycle], 'restore_at_exit': restore}
    state = {'network_ipopt_solve_records': drained, 'convergence_depth_tail': tail_state,
             'stale_network_ipopt_solve_records_discarded': len(_stale)}
    d1 = os.path.join(scratch, 'r1_eval')
    os.makedirs(d1)
    s1 = H.persist_convergence_depth_capture(state, d1)
    back1 = H.load_network_ipopt_solve_records(os.path.join(d1, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    tail1 = H.load_convergence_depth_tail_state(os.path.join(d1, H.CONVERGENCE_DEPTH_TAIL_STATE_FILE))
    compare_keys = sorted(set(first) - {'log_path', 'log_bytes', 'warm_start'})
    reproduced = back1[0] if back1 else {}
    results['r1'] = {
        'log': log_rel, 'log_sha256_now': log_sha_now, 'log_sha256_committed_record': log_sha_committed,
        'segment_bytes': first['log_bytes'], 'n_drained': len(drained), 'options_passed': passed,
        'records_roundtrip_equal': back1 == drained, 'tail_state_roundtrip_equal': tail1 == tail_state,
        'reproduces_committed_w83_record': {k: (reproduced.get(k), first.get(k)) for k in compare_keys
                                            if reproduced.get(k) != first.get(k)},
        'summary': s1,
    }
    results['r1']['holds'] = (log_sha_now is not None and log_sha_now == log_sha_committed and len(drained) == 1
                              and results['r1']['records_roundtrip_equal']
                              and results['r1']['tail_state_roundtrip_equal']
                              and not results['r1']['reproduces_committed_w83_record'])

    # R2: the committed W83 run's own 144 records and tail state
    d2 = os.path.join(scratch, 'r2_eval')
    os.makedirs(d2)
    s2 = H.persist_convergence_depth_capture({'network_ipopt_solve_records': copy.deepcopy(w83_records),
                                              'convergence_depth_tail': copy.deepcopy(w83_tail)}, d2)
    back2 = H.load_network_ipopt_solve_records(os.path.join(d2, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    tail2 = H.load_convergence_depth_tail_state(os.path.join(d2, H.CONVERGENCE_DEPTH_TAIL_STATE_FILE))
    results['r2'] = {'n_records': len(back2), 'records_roundtrip_equal': back2 == w83_records,
                     'tail_state_roundtrip_equal': tail2 == w83_tail,
                     'records_file_sha256': s2['network_ipopt_solve_records_sha256'],
                     'committed_w83_records_sha256': _sha(W83_RECORDS),
                     'records_file_byte_identical_to_committed': s2['network_ipopt_solve_records_sha256']
                     == _sha(W83_RECORDS), 'summary': s2}
    results['r2']['holds'] = (all(inputs_ok.values()) and len(back2) == 144 and results['r2']['records_roundtrip_equal']
                              and results['r2']['tail_state_roundtrip_equal']
                              and results['r2']['records_file_byte_identical_to_committed'])

    # R3: tail off, no records
    d3 = os.path.join(scratch, 'r3_eval')
    os.makedirs(d3)
    off_state = {'network_ipopt_solve_records': [], 'convergence_depth_tail': {'enabled': False},
                 'stale_network_ipopt_solve_records_discarded': 0}
    s3 = H.persist_convergence_depth_capture(off_state, d3)
    results['r3'] = {'summary': s3,
                     'records_loaded': H.load_network_ipopt_solve_records(os.path.join(d3, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE)),
                     'tail_loaded': H.load_convergence_depth_tail_state(os.path.join(d3, H.CONVERGENCE_DEPTH_TAIL_STATE_FILE))}
    results['r3']['holds'] = (s3['status'] == 'written' and s3['n_records'] == 0 and results['r3']['records_loaded'] == []
                              and results['r3']['tail_loaded'] == {'enabled': False})

    # R4: write-once
    results['r4'] = {'second_persist_raises': _raises(lambda: H.persist_convergence_depth_capture(off_state, d3))}
    results['r4']['holds'] = results['r4']['second_persist_raises'] is not None

    # R5: the summary is a recount of the file
    recount_rounds, recount_tally = {}, {}
    for r in back2:
        recount_rounds[str(r['round'])] = recount_rounds.get(str(r['round']), 0) + 1
        k = f"{'TSO' if r['agent'] == 'TSO' else 'DSO'}|{r['floor_status']}"
        recount_tally[k] = recount_tally.get(k, 0) + 1
    results['r5'] = {'recount_rounds': recount_rounds, 'recount_tally': recount_tally}
    results['r5']['holds'] = (s2['n_records'] == len(back2) and s2['records_per_round'] == recount_rounds
                              and s2['floor_status_tally'] == recount_tally
                              and s2['n_parse_problems'] == sum(1 for r in back2 if r['parse_reason'] is not None))
    return results


def _admm_with(tail):
    admm = ADMMParameters()
    admm.convergence_depth_tail = dict(tail)
    return admm


# ======================================================================================================================
#  C -- the checklist assertion in both states
# ======================================================================================================================
def _spec_like(declared=None, drop=False):
    configuration = {'overrides': {}, 'arm_label': 's39_D',
                     'case_file_anderson_acceleration': {'enabled': True, 'memory': 5, 'regularization': 1e-10,
                                                         'reject_policy': 'keep_memory'}}
    if not drop:
        configuration['convergence_depth_tail'] = declared
    return {'configuration': configuration, 'cap': 2, 'required_consecutive_cycles': 10}


def check_c1():
    out = {}
    for tag, spec in (('undeclared', _spec_like(drop=True)), ('declared_none', _spec_like(None)),
                      ('declared_on', _spec_like(dict(DECLARED_ON))), ('declared_off', _spec_like(dict(DECLARED_OFF)))):
        c = H.assert_convergence_depth_tail_capture(spec)
        out[tag] = {'tail_enabled_for_this_run': c['tail_enabled_for_this_run'],
                    'compl_inf_tol_tail': c['compl_inf_tol_tail'], 'source': c['source'],
                    'checks_all': all(c['checks'].values()), 'n_checks': len(c['checks'])}
    invalid = {
        'int_tol': {'enabled': True, 'compl_inf_tol': 1},
        'bool_tol': {'enabled': True, 'compl_inf_tol': True},
        'negative_tol': {'enabled': True, 'compl_inf_tol': -1e-6},
        'nan_tol': {'enabled': True, 'compl_inf_tol': float('nan')},
        'string_enabled': {'enabled': 'yes', 'compl_inf_tol': 1e-6},
        'extra_key': {'enabled': True, 'compl_inf_tol': 1e-6, 'cycles': 9},
        'missing_key': {'enabled': True},
        'not_a_dict': True,
    }
    out['invalid_refused'] = {k: _raises(lambda v=v: H.assert_convergence_depth_tail_capture(_spec_like(v)))
                              for k, v in invalid.items()}
    original = srp._run_operational_planning

    def _run_operational_planning(planning_problem, candidate_solution, initial_state=None, debug_flag=False):
        return {'admm_diagnostics': []}   # negative control: no capture path at all

    srp._run_operational_planning = _run_operational_planning
    try:
        out['negative_control_missing_capture_path_raises'] = _raises(
            lambda: H.assert_convergence_depth_tail_capture(_spec_like(dict(DECLARED_ON))), AssertionError)
    finally:
        srp._run_operational_planning = original
    out['holds'] = (out['undeclared']['tail_enabled_for_this_run'] is False
                    and out['declared_none']['tail_enabled_for_this_run'] is False
                    and out['declared_on']['tail_enabled_for_this_run'] is True
                    and out['declared_on']['compl_inf_tol_tail'] == 1e-6
                    and out['declared_off']['tail_enabled_for_this_run'] is False
                    and all(out[t]['checks_all'] for t in ('undeclared', 'declared_none', 'declared_on',
                                                           'declared_off'))
                    and all(v is not None for v in out['invalid_refused'].values())
                    and out['negative_control_missing_capture_path_raises'] is not None
                    and srp._run_operational_planning is original)
    return out


def check_c2(planning, sed, candidate):
    a = planning.params.admm
    default = copy.deepcopy(ADMMParameters().convergence_depth_tail)
    out = {'default_before': copy.deepcopy(a.convergence_depth_tail)}

    def run(spec):
        a.convergence_depth_tail = copy.deepcopy(default)
        holder, report = {}, {}
        hook = H._config_hook_factory(spec, holder, overrides={})
        hook(planning=planning, sed=sed, candidate=candidate, report=report)
        return {'applied': holder.get('convergence_depth_tail_applied'),
                'report_checklist': (report.get('rule_eleven_checklist') or {}).get('w84_convergence_depth_tail'),
                'dict_in_force_after': copy.deepcopy(a.convergence_depth_tail),
                'enabled_helper_after': srp.convergence_depth_tail_enabled(a)}

    out['undeclared'] = run(_spec_like(drop=True))
    out['declared_on'] = run(_spec_like(dict(DECLARED_ON)))
    out['declared_off'] = run(_spec_like(dict(DECLARED_OFF)))

    # negative control: the tail switched on in the ADMM parameters, the spec declaring nothing -> must raise
    holder, report = {}, {}
    hook = H._config_hook_factory(_spec_like(drop=True), holder, overrides={})
    a.convergence_depth_tail = dict(DECLARED_ON)
    out['negative_control_on_without_declaration_raises'] = _raises(
        lambda: hook(planning=planning, sed=sed, candidate=candidate, report=report), RuntimeError)
    a.convergence_depth_tail = copy.deepcopy(default)
    out['invalid_declaration_refused_at_factory'] = _raises(
        lambda: H._config_hook_factory(_spec_like({'enabled': True, 'compl_inf_tol': 1}), {}, overrides={}),
        ValueError)
    u, on, off = out['undeclared'], out['declared_on'], out['declared_off']
    out['holds'] = (
        out['default_before'] == default
        and u['applied']['ok'] and u['applied']['enabled_in_force'] is False and u['dict_in_force_after'] == default
        and u['report_checklist'] == u['applied'] and u['applied']['declared'] is None
        and on['applied']['ok'] and on['applied']['enabled_in_force'] is True
        and on['dict_in_force_after'] == DECLARED_ON and on['enabled_helper_after'] is True
        and on['report_checklist'] == on['applied']
        and off['applied']['ok'] and off['applied']['enabled_in_force'] is False
        and off['dict_in_force_after'] == DECLARED_OFF and off['report_checklist'] == off['applied']
        and out['negative_control_on_without_declaration_raises'] is not None
        and out['invalid_declaration_refused_at_factory'] is not None
        and a.convergence_depth_tail == default)
    return out


def check_c3():
    on = H.assert_convergence_depth_tail_capture(_spec_like(dict(DECLARED_ON)))
    off = H.assert_convergence_depth_tail_capture(_spec_like(drop=True))
    tail_on = {'enabled': True, 'compl_inf_tol_tail': 1e-6, 'per_cycle': [
        {'cycle': 1, 'active': False, 'acted': False}, {'cycle': 2, 'active': True, 'acted': True}],
        'restore_at_exit': {'acted': True}}
    cases = {
        'expected_on_returned_on': (on, {'convergence_depth_tail': tail_on}, True),
        'expected_off_returned_off': (off, {'convergence_depth_tail': {'enabled': False}}, True),
        'expected_on_returned_off_forgotten_enable': (on, {'convergence_depth_tail': {'enabled': False}}, False),
        'expected_off_returned_on': (off, {'convergence_depth_tail': tail_on}, False),
        'expected_on_returned_wrong_value': (on, {'convergence_depth_tail': {**tail_on, 'compl_inf_tol_tail': 1e-5}},
                                             False),
        'expected_off_no_tail_state': (off, {}, False),
        'expected_on_no_state': (on, None, False),
    }
    out = {}
    for tag, (checklist, state, want) in cases.items():
        got = H.convergence_depth_tail_state_check(checklist, state)
        out[tag] = {'check': got, 'expected_match': want, 'as_expected': got['match'] is want}
    out['on_case_cycles_active'] = out['expected_on_returned_on']['check']['cycles_tail_active']
    out['holds'] = all(v['as_expected'] for k, v in out.items() if isinstance(v, dict) and 'as_expected' in v) \
        and out['on_case_cycles_active'] == [2]
    return out


def _ordered(text, *needles):
    pos = -1
    for needle in needles:
        nxt = text.find(needle, pos + 1)
        if nxt < 0:
            return False
        pos = nxt
    return True


def check_c4():
    child = inspect.getsource(H._child_real)
    main_child = inspect.getsource(H.main_child)
    hook_src = inspect.getsource(H._config_hook_factory)
    checks = {
        'checklist_asserted_before_the_run': _ordered(
            child, 'tail_checklist = assert_convergence_depth_tail_capture(spec)', 'G.run_admm_arm('),
        'sidecars_refused_if_present_before_the_run': _ordered(
            child, 'NETWORK_IPOPT_SOLVE_RECORDS_FILE, CONVERGENCE_DEPTH_TAIL_STATE_FILE)]:', 'G.run_admm_arm('),
        'persisted_first_in_the_post_run_hook_before_the_terminal_phase': _ordered(
            child, 'def post_run_hook(', 'persist_convergence_depth_capture(state, eval_dir)',
            'convergence_depth_tail_state_check(tail_checklist, state)', 'G.write_boyd_terminal_s35ref(',
            '_terminal_phase(planning, models, rows, report, state, st, optimization_results, primal_evolution)'),
        'record_carries_the_four_fields': all(
            f"'{k}':" in child for k in ('convergence_depth_tail_checklist_asserted_before_run',
                                         'convergence_depth_tail_applied_in_child',
                                         'convergence_depth_tail_state_check', 'convergence_depth_capture')),
        'outcome_flags_the_error': "'convergence_depth_capture_error': convergence_depth_error" in child,
        'main_child_exits_2_on_it': _ordered(main_child, "outcome.get('convergence_depth_capture_error')",
                                             'sys.exit(2)'),
        'error_record_carries_the_checklist': (
            "'convergence_depth_tail_checklist_asserted_before_run': progress.get(" in main_child),
        'hook_applies_after_the_d_checks_and_overrides': _ordered(
            hook_src, "raise RuntimeError(f'S44 campaign child: configuration not as frozen",
            "holder['anderson_acceleration_effective'] = dict(a.anderson_acceleration)",
            "holder['convergence_depth_tail_applied'] = apply_convergence_depth_tail_declaration(a, tail_declared)"),
        'w47_presence_strings_kept': ('if premium is not None:  # W47' in hook_src
                                      and 'if derived is not None:  # W47' in hook_src),
        'admm_default_unchanged_off': ADMMParameters().convergence_depth_tail == DECLARED_OFF,
        'production_files_unchanged_vs_base': _git(['diff', '--name-only', BASE_COMMIT, '--']
                                                   + list(PRODUCTION_FILES)).strip() == '',
    }
    return {'checks': checks, 'holds': all(checks.values())}


def run(scratch):
    started = time.time()
    _log('K1/K2 evaluation_key source and diff hunks vs ' + BASE_COMMIT)
    k12 = check_k1_k2()
    _log(f"    hunks {k12['n_hunks']}; K1 {k12['k1_no_hunk_touches_a_key_function']}; K2 {k12['k2_sources_identical']}")
    _log('K3 every committed campaign spec entry, keys recomputed by the live harness')
    k3 = check_k3()
    _log(f"    specs {k3['n_specs']}, entries {k3['n_entries']}, mismatches {k3['n_mismatches']}")
    _log('K4 freeze reproduction of s47_recert (scratch)')
    k4 = check_k4(scratch)
    _log(f"    undeclared {k4['k4a_undeclared_reproduces_the_committed_spec']}; declared "
         f"{k4['k4b_declared_adds_only_the_declaration_and_keeps_the_keys']}; drift "
         f"{k4['results']['undeclared']['configuration_keys_differing_from_committed']}")
    _log('building the planning object (G._construct_arm_planning, no solve)')
    planning, sed, candidate = _build_planning(scratch)
    _log('R persistence round-trip')
    r = check_r(scratch, planning)
    _log(f"    r1 {r['r1']['holds']} r2 {r['r2']['holds']} r3 {r['r3']['holds']} r4 {r['r4']['holds']} "
         f"r5 {r['r5']['holds']}")
    _log('C checklist assertion, both states')
    c1 = check_c1()
    c2 = check_c2(planning, sed, candidate)
    c3 = check_c3()
    c4 = check_c4()
    _log(f"    c1 {c1['holds']} c2 {c2['holds']} c3 {c3['holds']} c4 {c4['holds']}")
    items = {
        'K1_no_diff_hunk_in_evaluation_key_or_its_helpers': k12['k1_no_hunk_touches_a_key_function'],
        'K2_key_function_sources_identical_to_base': k12['k2_sources_identical'],
        'K3_every_committed_eval_key_recomputed_equal': k3['k3_every_committed_key_recomputed_equal'],
        'K4a_undeclared_freeze_reproduces_s47_recert': k4['k4a_undeclared_reproduces_the_committed_spec'],
        'K4b_declared_freeze_adds_only_the_declaration': k4['k4b_declared_adds_only_the_declaration_and_keeps_the_keys'],
        'R1_production_made_records_roundtrip': r['r1']['holds'],
        'R2_committed_w83_records_roundtrip_byte_identical': r['r2']['holds'],
        'R3_tail_off_state_roundtrip': r['r3']['holds'],
        'R4_write_once': r['r4']['holds'],
        'R5_summary_is_a_recount': r['r5']['holds'],
        'C1_checklist_records_enabled_and_disabled_and_fires': c1['holds'],
        'C2_hook_applies_reads_back_and_refuses': c2['holds'],
        'C3_post_run_state_check_both_states': c3['holds'],
        'C4_wiring': c4['holds'],
    }
    return {'stage': STAGE, 'utc': datetime.now(timezone.utc).isoformat(),
            'git_head': _git(['rev-parse', 'HEAD']).strip(), 'base_commit': BASE_COMMIT,
            'script_sha256': _sha(os.path.basename(__file__)), 'harness_sha256': _sha(HARNESS),
            'production_sha256': {f: _sha(f) for f in PRODUCTION_FILES},
            'scratch_dir_note': 'scratch outputs (frozen scratch specs, persisted sidecars, planning logs dir) are '
                                'outside the repository and not preserved; every value checked is recorded here',
            'K1_K2': k12, 'K3': k3, 'K4': k4, 'R': r, 'C1': c1, 'C2': c2, 'C3': c3, 'C4': c4,
            'items': items, 'all_hold': all(items.values()), 'wall_s': time.time() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch-dir', required=True)
    args = parser.parse_args()
    status = 1
    try:
        _log(STAGE)
        _log(f"git HEAD {_git(['rev-parse', 'HEAD']).strip()}; guard armed permitted=() (zero solves)")
        scratch = os.path.abspath(args.scratch_dir)
        if scratch == REPO or scratch.startswith(REPO + os.sep):
            raise SystemExit(f'--scratch-dir must be outside the repository: {scratch}')
        if os.path.exists(scratch) and os.listdir(scratch):
            raise SystemExit(f'--scratch-dir must be empty or absent: {scratch}')
        os.makedirs(scratch, exist_ok=True)
        _refuse_overwrite(CHECKS_JSON)
        _refuse_overwrite(CHECKS_MANIFEST)
        payload = run(scratch)
        os.makedirs(_abs(OUT), exist_ok=True)
        with open(_abs(CHECKS_JSON), 'w') as handle:
            json.dump(payload, handle, indent=1, default=H._json_default)
        inputs = [CHECKS_JSON, os.path.basename(__file__), HARNESS, S47_SPEC, W83_RECORDS, W83_TAIL, W83_MANIFEST,
                  W83_LOG_HASHES] + list(PRODUCTION_FILES)
        with open(_abs(CHECKS_MANIFEST), 'w') as handle:
            json.dump({rel: _sha(rel) for rel in inputs}, handle, indent=2, sort_keys=True)
        for k, v in payload['items'].items():
            _log(f'   {k}: {v}')
        _log(f"K4 consequence: {payload['K4']['consequence_recorded']}")
        _log(f"ALL_HOLD={payload['all_hold']}  (wall {payload['wall_s']:.1f} s)")
        status = 0 if payload['all_hold'] else 1
    except SystemExit as error:
        _log(f'STOP: {error}')
        status = 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        status = 1
    finally:
        failures = GUARD.verify(0)
        _log(f'GUARD {dict(GUARD.counts)}; verify(0) -> {failures}')
        GUARD.uninstall()
        if failures:
            status = 1
    return status


if __name__ == '__main__':
    sys.exit(main())
