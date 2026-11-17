"""
P5.15 Addendum 46 ruling 7, Planner task W85 -- ZERO-SOLVE verification of two campaign-harness corrections
(`p515_s44_campaign_harness.py`) before the SRP1 re-certification. Nothing here solves: `SolveProfileGuard(permitted=())`
is armed at import, before any production module is imported, and verified at EXACTLY 0 on every exit path (the A4
kill-test subprocess arms its own and reports its counts before it is killed).

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 46 ruling 7; Planner task W85:
  Q1 -- "the eval-key collision is NOT acceptable; fix it. ... The tail must enter `evaluation_key`, but ONLY WHEN
        DECLARED, following the existing W21/W33/W47 pattern so that undeclared specs produce byte-identical keys."
        Verification: the 7,016-entry check over all 50 committed campaign specs with 0 mismatches; a declared tail
        gives a key DIFFERENT from the undeclared one for C*, x = 0 and the unit (the three pairs); re-freezing
        s47_recert without a tail declaration reproduces the committed spec exactly.
  Q3 -- "Append `network_ipopt_solve_records` per cycle so a crashed or killed child retains everything up to the
        failure. Keep the existing end-of-run write and its sha256 summary consistent with the appended file";
        and W84 finding 2 (a child raising mid-run leaves no records; a parent-synthesised record carries no tail
        checklist).
  "Confirm: the tail still defaults off; the checklist assertion still behaves in both states; the persisted files
   still round-trip; and the C* reference comparison is unaffected."

CHECKS (all must hold):
  K  evaluation_key
     K1 vs BASE_COMMIT (HEAD when W85 started): of the key functions, ONLY `evaluation_key` differs in code; its diff
        removes exactly two lines (the signature's last line, the docstring's last line) and every other line is
        added; `validate_convergence_depth_tail` differs in its docstring only (AST without docstrings equal);
     K2 EVERY entry of EVERY git-tracked campaign spec (`data/**/campaign_spec_*.json`), recomputed by the live
        harness with the spec's own declarations (convergence_depth_tail included -- None in all of them): key,
        eval_key, eval_dir, working_dir_ids equal the committed values -- 0 mismatches;
     K3 freeze reproduction of the committed s47_recert spec (scratch root): (a) undeclared -> candidates identical,
        configuration identical up to the provenance drift W84 allowed; (b) declared enabled / (c) declared disabled
        -> every entry differs from the committed one in EXACTLY {eval_key, eval_dir, working_dir_ids}, each eval_key
        differs from the committed one and equals `evaluation_key(..., convergence_depth_tail=<declaration>)`, the
        configuration = (a) + exactly `convergence_depth_tail`; the enabled and disabled keys differ;
     K4 the three pairs (undeclared vs declared enabled {True, 1e-6}): C* and the unit under the s47_recert
        declarations, x = 0 under the s48 x0_capture declarations (the C2 baseline); undeclared recomputes to the
        committed key; declared differs; no declared key equals any eval key of any committed spec;
     K5 `convergence_depth_tail=None` == omitted (the three cells); key sensitive to compl_inf_tol; invalid
        declarations refused by `evaluation_key`.
  A  per-round append (`ConvergenceDepthAppender`, `convergence_depth_append_hooks`)
     A1 completed-run replay on a planning object built by `p515_g_g1_g4_admm_gates._construct_arm_planning` (no
        solve): the committed W83 gate's 144 records (hash-verified) are put back into the TSO/DSO `Network` deques
        round by round and drained by PRODUCTION's `_drain_network_ipopt_solve_records` through the hooks, and the
        tail by production's baseline / apply / next-state functions (tail declared on, never active, 2 cycles);
        the state assembled as `_run_operational_planning` assembles it is persisted by the unchanged end-of-run
        write; `reconcile` -> ok; the appended records file is BYTE-IDENTICAL to the end-of-run file AND to the
        committed W83 sidecar; the tail state rebuilt from the events equals the assembled state and the committed
        W83 tail state; hooks restored; NEGATIVE CONTROLS: a records file missing its last line, a returned tail
        state that differs, a failed end-of-run write and a recorded append-write error each make `reconcile`
        return ok False;
     A1b an append write that raises (unwritable path) does NOT propagate into production's drain: the 48 records
        are returned to production, the error is recorded, `reconcile` returns ok False;
     A2 tail-off run: no tail events; rebuilt {'enabled': False}; reconcile ok;
     A3 Python exception in the round in flight (cycle 2): rounds 0-1 appended; `drain_on_failure` drains round 2
        from the Network objects; `recover_convergence_depth_append` -> 144 records, rounds_drained [0, 1],
        round_drained_at_failure 2, the child's checklist, per_cycle rebuilt up to cycle 2 (no predicate for 2);
        records file byte-identical to the W83 sidecar;
     A3b the REAL `main_child` exception path (in-process, scratch campaign root / lock, `_child_real` replaced by a
        stand-in that creates the appender exactly as `_child_real` does and raises in cycle 2): the error record
        carries `convergence_depth_failure_drain` (round 2, 48 records) and
        `convergence_depth_per_round_append_recovered` (144 records, the checklist); exit code 1;
     A4 KILLED child (SIGKILL, subprocess of this script, production's drain on a duck-typed planning): the rounds
        drained before the kill survive (96 records, rounds [0, 1]); round 2's records (in memory at the kill) are
        NOT recoverable -- recorded; the parent-synthesised barrier record (`_barrier_record_for_missing`) carries the
        spec's declaration, the child's checklist and the recovered summary; a child that died before creating the
        appender gives None (field present);
     A5 seal: after `seal()` drains write nothing (counted) and `drain_on_failure` skips; the files are unchanged;
     A6 wiring: appender created after the checklist and before `run_admm_arm`; hooks installed around it; in the
        post-run hook reconcile after the end-of-run write and before the terminal phase, then seal; the error flag
        includes it; `main_child` drains on failure before writing the record; record / barrier fields present;
        `_run_operational_planning` resolves the four wrapped names as module globals; append files refused if
        present before the run.
  C  retained behaviour (the committed W84 check functions, imported with their guard disarmed, run on the live
     harness): R1-R5 round-trip, C1-C3 checklist in both states, C4 wiring; tail default OFF; the C* reference
     comparison: `resolve_post_certification` never reads an eval key, and the committed W84 gate's comparator
     module (`p515_s53_w84_tight_tail_gate_r2`, `H = T83.H`) references only harness names whose module-level
     definitions are unchanged since BASE_COMMIT.

EXACT LAUNCH COMMAND (repo root; attached, both streams captured):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_w85_harness_checks.py --scratch-dir <an EMPTY directory outside the repository> \\
        > data/SRP1/Results/P515S53/tight_tail_w85/harness_checks_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/tight_tail_w85/harness_checks/{checks_w85.json,
checks_w85_manifest_sha256.json}. Exit 0 when every check holds, 1 otherwise.
"""

import argparse
import ast
import copy
import difflib
import inspect
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
from contextlib import redirect_stdout
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W85 harness zero-solve checks').install()

import p515_s44_campaign_harness as H  # noqa: E402

STAGE = ('P5.15 Addendum 46 ruling 7, W85 -- campaign harness: the declared convergence-depth tail enters '
         'evaluation_key (Q1); the floor-status records and tail events appended per round (Q3, W84 finding 2)')
BASE_COMMIT = '4726fc32'   # HEAD when W85 started
HARNESS = 'p515_s44_campaign_harness.py'
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT = os.path.join(S53, 'tight_tail_w85', 'harness_checks')
CHECKS_JSON = os.path.join(OUT, 'checks_w85.json')
CHECKS_MANIFEST = os.path.join(OUT, 'checks_w85_manifest_sha256.json')
PRODUCTION_FILES = ('network.py', 'shared_resources_planning.py', 'admm_parameters.py',
                    'admm_anderson_acceleration.py', 'solver_parameters.py')
KEY_FUNCTIONS = ('evaluation_key', 'candidate_key', 'canonical_candidate', 'eval_ids', 'eval_dir_name',
                 '_entry_eval_key', 'effective_anderson_acceleration', 'validate_model_variant',
                 'validate_ess_ageing_baseline', 'flex_price_multiplier_in_key', 'validate_derived_instance',
                 'derived_instance_identity', 'validate_interface_deviation_premium', 'validate_overrides',
                 'validate_flex_price_multiplier', 'validate_case_file_anderson_acceleration', '_sanitize_id',
                 'validate_convergence_depth_tail')
S47_SPEC = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert',
                        'campaign_spec_s47_recert_902f93aa.json')
X0_SPEC = os.path.join('data', 'SRP1', 'Results', 'P515S48', 'x0_capture', 'campaign_spec_s48_x0_capture_4a50c0e2.json')
W83_GATE = os.path.join(S53, 'tight_tail_w83', 'srp1_bitwise_gate')
W83_RECORDS = os.path.join(W83_GATE, 'w83_network_ipopt_solve_records.jsonl')
W83_TAIL = os.path.join(W83_GATE, 'w83_tail_state.json')
W83_MANIFEST = os.path.join(W83_GATE, 'w83_manifest_sha256.json')
W84_CHECKS_MODULE = 'p515_s53_w84_harness_capture_checks'
W84_GATE_MODULE_FILE = 'p515_s53_w84_tight_tail_gate_r2.py'
DECLARED_ON = {'enabled': True, 'compl_inf_tol': 1e-6}
DECLARED_OFF = {'enabled': False, 'compl_inf_tol': 1e-6}
KILL_READY = 'W85-A4-READY'


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W85-checks] {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _raises(fn, exc=Exception):
    try:
        fn()
    except exc as error:  # noqa: BLE001
        return f'{type(error).__name__}: {str(error)[:300]}'
    return None


# ======================================================================================================================
#  K -- evaluation_key
# ======================================================================================================================
def _functions(text):
    tree = ast.parse(text)
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _source(text, node):
    return '\n'.join(text.splitlines()[node.lineno - 1:node.end_lineno])


def _code_dump(node):
    node = copy.deepcopy(node)
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], 'value', None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        node.body = body[1:]
    return ast.dump(node, include_attributes=False)


def check_k1():
    base_text = _git(['show', f'{BASE_COMMIT}:{HARNESS}'])
    live_text = open(_abs(HARNESS)).read()
    base_fns, live_fns = _functions(base_text), _functions(live_text)
    per_fn = {}
    for name in KEY_FUNCTIONS:
        b, l = _source(base_text, base_fns[name]), _source(live_text, live_fns[name])
        per_fn[name] = {'source_identical': b == l,
                        'code_identical_without_docstring': _code_dump(base_fns[name]) == _code_dump(live_fns[name])}
    b = _source(base_text, base_fns['evaluation_key']).splitlines()
    l = _source(live_text, live_fns['evaluation_key']).splitlines()
    diff = [d for d in difflib.unified_diff(b, l, lineterm='', n=0) if not d.startswith(('---', '+++', '@@'))]
    removed = [d[1:] for d in diff if d.startswith('-')]
    added = [d[1:] for d in diff if d.startswith('+')]
    expected_removed = [
        '                   flex_price_multiplier=None, derived_instance=None, interface_deviation_premium=None):',
        '    premiums, never shares a key. Both None returns exactly what the formulas above return."""']
    others_identical = all(v['source_identical'] for k, v in per_fn.items()
                           if k not in ('evaluation_key', 'validate_convergence_depth_tail'))
    return {
        'base_commit': BASE_COMMIT, 'per_function': per_fn,
        'evaluation_key_diff': diff, 'evaluation_key_removed_lines': removed, 'n_added_lines': len(added),
        'k1_only_evaluation_key_code_changed': (
            others_identical and not per_fn['evaluation_key']['code_identical_without_docstring']
            and per_fn['validate_convergence_depth_tail']['code_identical_without_docstring']),
        'k1_evaluation_key_diff_removes_only_signature_and_docstring_tail': removed == expected_removed,
    }


def _module_level_sources(text):
    """name -> source of its module-level definition (function, class or assignment)."""
    out = {}
    lines = text.splitlines()
    for node in ast.parse(text).body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for name in names:
            out[name] = '\n'.join(lines[node.lineno - 1:node.end_lineno])
    return out


def _spec_paths():
    return sorted(p for p in _git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip())


def _entry_key(spec, entry, tail='from_spec'):
    cfg = spec.get('configuration') or {}
    return H.evaluation_key(
        H.candidate_key(entry['canonical']), entry.get('overrides') or {},
        case_file_aa=H.validate_case_file_anderson_acceleration(cfg.get('case_file_anderson_acceleration')),
        model_variant=entry.get('model_variant'),
        ess_ageing_baseline=H.validate_ess_ageing_baseline(cfg.get('ess_ageing_baseline')),
        flex_price_multiplier=entry.get('flex_price_multiplier'),
        derived_instance=H.validate_derived_instance(cfg.get('derived_instance')),
        interface_deviation_premium=entry.get('interface_deviation_premium'),
        convergence_depth_tail=cfg.get('convergence_depth_tail') if tail == 'from_spec' else tail)


def check_k2():
    paths = _spec_paths()
    per_spec, mismatches, n_entries, all_keys = {}, [], 0, set()
    for rel in paths:
        spec = json.load(open(_abs(rel)))
        cfg = spec.get('configuration') or {}
        n_ok = 0
        for entry in spec.get('candidates') or []:
            n_entries += 1
            all_keys.add(H._entry_eval_key(entry))
            got = {'key': H.candidate_key(entry['canonical'])}
            if 'eval_key' in entry:
                got['eval_key'] = _entry_key(spec, entry)
                got['entry_eval_key'] = H._entry_eval_key(entry)
            if 'eval_dir' in entry:
                got['eval_dir'] = H.eval_dir_name(got.get('eval_key', got['key']), entry['label'])
            if 'working_dir_ids' in entry:
                got['working_dir_ids'] = H.eval_ids(spec['campaign_id'], got.get('eval_key', got['key']))
            want = {'key': entry.get('key'), 'eval_key': entry.get('eval_key'), 'entry_eval_key': entry.get('eval_key'),
                    'eval_dir': entry.get('eval_dir'), 'working_dir_ids': entry.get('working_dir_ids')}
            bad = {k: {'committed': want[k], 'recomputed': v} for k, v in got.items() if v != want[k]}
            if bad:
                mismatches.append({'spec': rel, 'label': entry.get('label'), 'fields': bad})
            else:
                n_ok += 1
        per_spec[rel] = {'sha256': _sha(rel), 'n_entries': len(spec.get('candidates') or []), 'n_ok': n_ok,
                         'declares_convergence_depth_tail': cfg.get('convergence_depth_tail') is not None}
    return {'n_specs': len(paths), 'n_entries': n_entries, 'n_mismatches': len(mismatches),
            'mismatches': mismatches[:20], 'per_spec': per_spec,
            'n_specs_declaring_a_tail': sum(1 for v in per_spec.values() if v['declares_convergence_depth_tail']),
            'k2_every_committed_key_recomputed_equal': (len(paths) == 50 and n_entries == 7016 and not mismatches)}, \
        all_keys


def _s47_candidates(spec):
    out = []
    for entry in spec['candidates']:
        canon = entry['canonical']
        nodes = {int(n): tuple(v) for n, v in canon['nodes'].items()}
        out.append((entry['label'], nodes, {'investment_year': canon['investment_year']}))
    return out


def check_k3(scratch):
    spec = json.load(open(_abs(S47_SPEC)))
    cfg = spec['configuration']
    configuration = {'name': cfg['name'], 'arm_label': cfg['arm_label'], 'overrides': cfg['overrides'],
                     'case_file_anderson_acceleration': cfg['case_file_anderson_acceleration'],
                     'ess_ageing_baseline': cfg['ess_ageing_baseline'],
                     'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'], 'note': cfg['note']}
    results, frozen_specs = {}, {}
    for tag, extra_cfg in (('undeclared', {}), ('declared_on', {'convergence_depth_tail': dict(DECLARED_ON)}),
                           ('declared_off', {'convergence_depth_tail': dict(DECLARED_OFF)})):
        root = os.path.join(scratch, f'k3_{tag}', 's47_recert')
        path, digest, frozen = H.freeze_campaign_spec(
            root, spec['campaign_id'], _s47_candidates(spec), {**configuration, **extra_cfg}, spec['cap'],
            spec['concurrency'], ['W85 zero-solve freeze reproduction (scratch, not a campaign)'],
            required_consecutive_cycles=spec['required_consecutive_cycles'])
        frozen_specs[tag] = (root, path, digest, frozen)
        fc = frozen['configuration']
        differing = sorted(k for k in set(fc) | set(cfg) if fc.get(k) != cfg.get(k))
        per_entry = {}
        for committed, new in zip(spec['candidates'], frozen['candidates']):
            fields = sorted(k for k in set(committed) | set(new) if committed.get(k) != new.get(k))
            per_entry[committed['label']] = {
                'label_matches': committed['label'] == new['label'], 'fields_differing': fields,
                'eval_key_committed': committed['eval_key'], 'eval_key_frozen_now': new['eval_key'],
                'eval_key_recomputes': new['eval_key'] == _entry_key(frozen, new)}
        results[tag] = {'n_entries': len(frozen['candidates']), 'per_entry': per_entry,
                        'candidates_identical_to_committed': frozen['candidates'] == spec['candidates'],
                        'configuration_keys_differing_from_committed': differing,
                        'differing_values': {k: {'committed': cfg.get(k), 'frozen_now': fc.get(k)} for k in differing},
                        'convergence_depth_tail': fc.get('convergence_depth_tail')}
    allowed_drift = {'case_file_last_commit', 'ess_params_file'}
    und = results['undeclared']
    und_ok = (und['candidates_identical_to_committed'] and und['convergence_depth_tail'] is None
              and set(und['configuration_keys_differing_from_committed']) <= allowed_drift
              and all(k != 'ess_params_file' or (und['differing_values'][k]['committed'] or {}).get('sha256')
                      == (und['differing_values'][k]['frozen_now'] or {}).get('sha256')
                      for k in und['configuration_keys_differing_from_committed']))
    changed = {'eval_key', 'eval_dir', 'working_dir_ids'}

    def declared_ok(tag, decl):
        r = results[tag]
        return (r['n_entries'] == len(spec['candidates']) and r['convergence_depth_tail'] == decl
                and set(r['configuration_keys_differing_from_committed'])
                == set(und['configuration_keys_differing_from_committed']) | {'convergence_depth_tail'}
                and all(e['label_matches'] and set(e['fields_differing']) == changed
                        and e['eval_key_frozen_now'] != e['eval_key_committed'] and e['eval_key_recomputes']
                        for e in r['per_entry'].values()))
    on_off_differ = all(results['declared_on']['per_entry'][lab]['eval_key_frozen_now']
                        != results['declared_off']['per_entry'][lab]['eval_key_frozen_now']
                        for lab in results['declared_on']['per_entry'])
    return {'results': results, 'allowed_provenance_drift': sorted(allowed_drift),
            'k3a_undeclared_reproduces_the_committed_spec': und_ok,
            'k3b_declared_on_changes_exactly_the_key_fields': declared_ok('declared_on', DECLARED_ON),
            'k3c_declared_off_changes_exactly_the_key_fields': declared_ok('declared_off', DECLARED_OFF),
            'k3d_on_and_off_keys_differ': on_off_differ}, frozen_specs


def check_k4_k5(all_committed_keys):
    s47 = json.load(open(_abs(S47_SPEC)))
    x0s = json.load(open(_abs(X0_SPEC)))
    cells = {'c_star': (s47, next(e for e in s47['candidates'] if e['label'] == 'c_star')),
             'unit': (s47, next(e for e in s47['candidates'] if e['label'] == 'n7_4h_e1')),
             'x0': (x0s, next(e for e in x0s['candidates'] if e['label'] == 'x0'))}
    pairs = {}
    for name, (spec, entry) in cells.items():
        und = _entry_key(spec, entry, tail=None)
        omitted = H.evaluation_key(H.candidate_key(entry['canonical']), entry.get('overrides') or {},
                                   case_file_aa=spec['configuration'].get('case_file_anderson_acceleration'),
                                   ess_ageing_baseline=spec['configuration'].get('ess_ageing_baseline'))
        on = _entry_key(spec, entry, tail=dict(DECLARED_ON))
        off = _entry_key(spec, entry, tail=dict(DECLARED_OFF))
        tighter = _entry_key(spec, entry, tail={'enabled': True, 'compl_inf_tol': 1e-7})
        pairs[name] = {'spec': os.path.relpath(_abs(X0_SPEC if name == 'x0' else S47_SPEC), REPO),
                       'label': entry['label'], 'canonical': entry['canonical'],
                       'committed_eval_key': entry['eval_key'], 'undeclared_eval_key': und,
                       'declared_on_eval_key': on, 'declared_off_eval_key': off,
                       'undeclared_equals_committed': und == entry['eval_key'],
                       'none_equals_omitted': und == omitted,
                       'declared_on_differs_from_undeclared': on != und,
                       'declared_off_differs_from_undeclared': off != und,
                       'declared_on_differs_from_off': on != off,
                       'sensitive_to_compl_inf_tol': tighter != on,
                       'declared_keys_absent_from_every_committed_spec': (on not in all_committed_keys
                                                                         and off not in all_committed_keys)}
    invalid = {'int_tol': {'enabled': True, 'compl_inf_tol': 1}, 'extra_key': {**DECLARED_ON, 'x': 1},
               'not_a_dict': True, 'string_enabled': {'enabled': 'yes', 'compl_inf_tol': 1e-6}}
    refused = {k: _raises(lambda v=v: H.evaluation_key('0' * 64, {}, convergence_depth_tail=v))
               for k, v in invalid.items()}
    k4 = all(p['undeclared_equals_committed'] and p['declared_on_differs_from_undeclared']
             and p['declared_keys_absent_from_every_committed_spec'] for p in pairs.values())
    k5 = (all(p['none_equals_omitted'] and p['sensitive_to_compl_inf_tol'] and p['declared_off_differs_from_undeclared']
              and p['declared_on_differs_from_off'] for p in pairs.values())
          and all(v is not None for v in refused.values()))
    return {'pairs': pairs, 'invalid_declarations_refused': refused,
            'k4_three_pairs_differ_and_undeclared_reproduce': k4, 'k5_properties': k5}


# ======================================================================================================================
#  A -- the per-round append
# ======================================================================================================================
def _w83_inputs():
    manifest = json.load(open(_abs(W83_MANIFEST)))
    ok = {rel: manifest.get(rel) == _sha(rel) for rel in (W83_RECORDS, W83_TAIL)}
    return ok, H.load_network_ipopt_solve_records(_abs(W83_RECORDS)), json.load(open(_abs(W83_TAIL)))


def _holders(planning):
    import shared_resources_planning as srp
    return dict(srp._convergence_depth_tail_holders(planning))


def _inject_round(planning, records, round_index):
    """Put the W83 records of one round back into the Network deques production drains, without the two keys the
    drain adds (they are the last two keys of every record, so the drain re-creates the exact key order)."""
    holders = _holders(planning)
    n = 0
    for record in records:
        if record['round'] != round_index:
            continue
        assert list(record)[-2:] == ['agent', 'round'], list(record)[-2:]
        bare = {k: v for k, v in record.items() if k not in ('agent', 'round')}
        holders[record['agent']].network[record['year']][record['day']].ipopt_solve_records.append(copy.deepcopy(bare))
        n += 1
    return n


def _admm_with(tail):
    from admm_parameters import ADMMParameters
    admm = ADMMParameters()
    admm.convergence_depth_tail = dict(tail)
    return admm


def _checklist(declared):
    spec = {'configuration': {'convergence_depth_tail': declared} if declared is not None else {}}
    return H.assert_convergence_depth_tail_capture(spec)


def _replay(planning, eval_dir, records, tail_on, fail_in_round=None):
    """Drive production's drain / tail functions through the hooks the way `_run_operational_planning` calls them
    for an initialisation + 2-cycle run (tail declared on or off, never active), assembling the state as production
    does. With `fail_in_round`, that round's records are put in the Network objects and an exception is raised
    before it is drained. Returns (appender, state, hooks_restored, exception)."""
    import shared_resources_planning as srp
    admm = _admm_with(DECLARED_ON if tail_on else DECLARED_OFF)
    appender = H.ConvergenceDepthAppender(eval_dir, _checklist(DECLARED_ON if tail_on else None))
    originals = {n: getattr(srp, n) for n in ('_drain_network_ipopt_solve_records',
                                              '_capture_convergence_depth_tail_baseline',
                                              '_apply_convergence_depth_tail', '_convergence_depth_tail_next_state')}
    exception = None
    state = None
    try:
        with H.convergence_depth_append_hooks(appender):
            wrapped = {n: getattr(srp, n) is not fn for n, fn in originals.items()}
            stale = len(srp._drain_network_ipopt_solve_records(planning, None))
            network_ipopt_solve_records = []
            tail_state = {'enabled': srp.convergence_depth_tail_enabled(admm)}
            _inject_round(planning, records, 0)
            network_ipopt_solve_records.extend(srp._drain_network_ipopt_solve_records(planning, 0))
            if tail_state['enabled']:
                tail_state.update(compl_inf_tol_tail=admm.convergence_depth_tail['compl_inf_tol'],
                                  option=srp.CONVERGENCE_DEPTH_TAIL_OPTION,
                                  baseline=srp._capture_convergence_depth_tail_baseline(planning, admm),
                                  per_cycle=[], restore_at_exit=None)
            active_next = False
            for cycle in (1, 2):
                if tail_state['enabled']:
                    tail_state['per_cycle'].append(srp._apply_convergence_depth_tail(
                        planning, admm, active_next, tail_state['baseline'], cycle))
                _inject_round(planning, records, cycle)
                if fail_in_round == cycle:
                    raise RuntimeError(f'W85 A3: simulated failure in cycle {cycle}, before its drain')
                if tail_state['enabled']:
                    active_next = srp._convergence_depth_tail_next_state(False, False, None)
                    tail_state['per_cycle'][-1]['aa_off_predicate_end_of_cycle'] = active_next
                network_ipopt_solve_records.extend(srp._drain_network_ipopt_solve_records(planning, cycle))
            if tail_state['enabled']:
                tail_state['restore_at_exit'] = srp._apply_convergence_depth_tail(
                    planning, admm, False, tail_state['baseline'], None)
            state = {'network_ipopt_solve_records': network_ipopt_solve_records,
                     'stale_network_ipopt_solve_records_discarded': stale, 'convergence_depth_tail': tail_state,
                     '_wrapped_inside_hooks': wrapped}
    except RuntimeError as error:
        exception = f'{type(error).__name__}: {error}'
    restored = all(getattr(srp, n) is fn for n, fn in originals.items())
    return appender, state, restored, exception


def check_a1_a2_a3_a5(scratch, planning):
    inputs_ok, w83_records, w83_tail = _w83_inputs()
    out = {'w83_inputs_match_committed_manifest': inputs_ok}
    w83_sha = _sha(W83_RECORDS)

    # A1: completed run, tail on
    d1 = os.path.join(scratch, 'a1_eval')
    os.makedirs(d1)
    app1, st1, restored1, exc1 = _replay(planning, d1, w83_records, tail_on=True)
    wrapped1 = st1.pop('_wrapped_inside_hooks')
    s1 = H.persist_convergence_depth_capture(st1, d1)
    rec1 = app1.reconcile(s1, st1)
    rebuilt1 = H.reconstruct_convergence_depth_tail_state(H.load_convergence_depth_append_events(app1.events_path)[0])
    out['a1'] = {'exception': exc1, 'wrapped_inside_hooks': wrapped1, 'hooks_restored': restored1,
                 'end_of_run_summary': s1, 'reconcile': rec1,
                 'append_sha256': rec1['records_append_sha256'], 'end_of_run_sha256': s1['network_ipopt_solve_records_sha256'],
                 'committed_w83_sha256': w83_sha,
                 'append_equals_committed_w83_sidecar': rec1['records_append_sha256'] == w83_sha,
                 'assembled_tail_state_equals_committed_w83': json.loads(json.dumps(st1['convergence_depth_tail'])) == w83_tail,
                 'rebuilt_tail_state_equals_committed_w83': rebuilt1 == w83_tail,
                 'rounds_appended': rec1['rounds_appended']}
    # negative controls: a records file with one line missing, and a returned tail state that differs, must fail
    tampered = copy.copy(app1)
    tampered.records_path = os.path.join(d1, 'tampered_records_append.jsonl')
    with open(app1.records_path) as src, open(tampered.records_path, 'w') as dst:
        dst.writelines(src.readlines()[:-1])
    st1_bad = copy.deepcopy(st1)
    st1_bad['convergence_depth_tail']['per_cycle'][-1]['aa_off_predicate_end_of_cycle'] = True
    neg = {'missing_last_record_line': tampered.reconcile(s1, st1),
           'tail_state_differs': app1.reconcile(s1, st1_bad),
           'end_of_run_write_failed': app1.reconcile({**s1, 'status': 'error'}, st1)}
    app1.write_errors.append({'where': 'W85 A1 negative control', 'error': 'injected'})
    neg['append_write_error_during_run'] = app1.reconcile(s1, st1)
    app1.write_errors.pop()
    out['a1']['negative_controls'] = {k: {kk: v.get(kk) for kk in (
        'ok', 'records_append_byte_identical_to_end_of_run_file',
        'tail_state_rebuilt_from_events_equals_returned_state')} for k, v in neg.items()}
    out['a1']['negative_controls_all_fail'] = (
        neg['missing_last_record_line']['ok'] is False
        and neg['missing_last_record_line']['records_append_byte_identical_to_end_of_run_file'] is False
        and neg['tail_state_differs']['ok'] is False
        and neg['tail_state_differs']['tail_state_rebuilt_from_events_equals_returned_state'] is False
        and neg['end_of_run_write_failed']['ok'] is False
        and neg['append_write_error_during_run']['ok'] is False)
    out['a1']['holds'] = (out['a1']['negative_controls_all_fail'] and exc1 is None and all(wrapped1.values()) and restored1 and rec1['ok']
                          and rec1['records_append_byte_identical_to_end_of_run_file']
                          and rec1['tail_state_rebuilt_from_events_equals_returned_state']
                          and out['a1']['append_equals_committed_w83_sidecar']
                          and out['a1']['assembled_tail_state_equals_committed_w83']
                          and out['a1']['rebuilt_tail_state_equals_committed_w83']
                          and rec1['rounds_appended'] == [0, 1, 2] and s1['n_records'] == 144)

    # A1b: a failing append write must not propagate into production's run (recorded; reconcile then fails)
    d1b = os.path.join(scratch, 'a1b_eval')
    os.makedirs(d1b)
    app1b = H.ConvergenceDepthAppender(d1b, _checklist(dict(DECLARED_ON)))
    app1b.records_path = os.path.join(d1b, 'no_such_dir', 'records.jsonl')   # open(..., 'a') raises
    import shared_resources_planning as srp
    with H.convergence_depth_append_hooks(app1b):
        _inject_round(planning, w83_records, 0)
        returned = srp._drain_network_ipopt_solve_records(planning, 0)
    out['a1b'] = {'records_returned_to_production': len(returned), 'write_errors': app1b.write_errors,
                  'reconcile_ok': app1b.reconcile({'status': 'written'}, {'convergence_depth_tail': {'enabled': False}})['ok']}
    out['a1b']['holds'] = (len(returned) == 48 and len(app1b.write_errors) == 1 and out['a1b']['reconcile_ok'] is False)

    # A5: seal -- after it, drains write nothing and the failure drain skips
    import shared_resources_planning as srp
    app1.seal()
    sizes_before = (os.path.getsize(app1.records_path), os.path.getsize(app1.events_path))
    with H.convergence_depth_append_hooks(app1):
        _inject_round(planning, w83_records, 1)
        n_after_seal = len(srp._drain_network_ipopt_solve_records(planning, 7))
    fd = app1.drain_on_failure('W85 A5 after seal')
    sizes_after = (os.path.getsize(app1.records_path), os.path.getsize(app1.events_path))
    out['a5'] = {'records_drained_after_seal_by_production': n_after_seal, 'drains_after_seal': app1.drains_after_seal,
                 'failure_drain_after_seal': fd, 'sizes_before': sizes_before, 'sizes_after': sizes_after}
    out['a5']['holds'] = (n_after_seal == 48 and app1.drains_after_seal == 1 and fd['status'] == 'skipped'
                          and sizes_before == sizes_after)

    # A2: completed run, tail off
    d2 = os.path.join(scratch, 'a2_eval')
    os.makedirs(d2)
    app2, st2, restored2, exc2 = _replay(planning, d2, w83_records, tail_on=False)
    st2.pop('_wrapped_inside_hooks')
    s2 = H.persist_convergence_depth_capture(st2, d2)
    rec2 = app2.reconcile(s2, st2)
    events2 = H.load_convergence_depth_append_events(app2.events_path)[0]
    out['a2'] = {'exception': exc2, 'hooks_restored': restored2, 'reconcile': rec2,
                 'event_kinds': sorted({e['event'] for e in events2}),
                 'rebuilt': H.reconstruct_convergence_depth_tail_state(events2),
                 'checklist_line_says_disabled': events2[0]['checklist']['tail_enabled_for_this_run'] is False}
    out['a2']['holds'] = (exc2 is None and restored2 and rec2['ok'] and out['a2']['rebuilt'] == {'enabled': False}
                          and out['a2']['event_kinds'] == ['checklist', 'drained']
                          and out['a2']['checklist_line_says_disabled']
                          and rec2['records_append_sha256'] == w83_sha)

    # A3: exception in cycle 2 before its drain
    d3 = os.path.join(scratch, 'a3_eval')
    os.makedirs(d3)
    app3, st3, restored3, exc3 = _replay(planning, d3, w83_records, tail_on=True, fail_in_round=2)
    before_recovery = H.recover_convergence_depth_append(d3)
    fd3 = app3.drain_on_failure(exc3)
    rec3 = H.recover_convergence_depth_append(d3)
    left_in_networks = len(srp._drain_network_ipopt_solve_records(planning, None))
    rebuilt3 = rec3['tail_state_rebuilt']
    out['a3'] = {'exception': exc3, 'hooks_restored': restored3, 'state_returned': st3 is not None,
                 'recovered_before_failure_drain': {k: before_recovery[k] for k in ('n_records', 'rounds_drained')},
                 'failure_drain': fd3, 'recovered': rec3, 'records_left_in_networks_after': left_in_networks}
    out['a3']['holds'] = (exc3 is not None and restored3 and st3 is None
                          and before_recovery['n_records'] == 96 and before_recovery['rounds_drained'] == [0, 1]
                          and fd3['status'] == 'drained' and fd3['round_in_flight'] == 2 and fd3['n_records'] == 48
                          and rec3['n_records'] == 144 and rec3['rounds_drained'] == [0, 1]
                          and rec3['round_drained_at_failure'] == 2
                          and rec3['records_append_sha256'] == w83_sha and rec3['n_unparsable_record_lines'] == 0
                          and rec3['tail_checklist_from_child']['tail_enabled_for_this_run'] is True
                          and [p['cycle'] for p in rebuilt3['per_cycle']] == [1, 2]
                          and 'aa_off_predicate_end_of_cycle' in rebuilt3['per_cycle'][0]
                          and 'aa_off_predicate_end_of_cycle' not in rebuilt3['per_cycle'][1]
                          and rebuilt3['restore_at_exit'] is None and left_in_networks == 0)
    return out, w83_records


def check_a3b(scratch, frozen_on, planning, w83_records):
    """The REAL `main_child` exception path, in-process: scratch campaign root (the K3 declared-on freeze), a scratch
    lock naming this process's parent, the thread caps in the environment, and `_child_real` replaced by a stand-in
    that sets up the appender exactly as `_child_real` does, then fails in cycle 2."""
    root, _path, digest, frozen = frozen_on
    entry = next(e for e in frozen['candidates'] if e['label'] == 'c_star')
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    os.makedirs(eval_dir)
    lock_path = os.path.join(scratch, 'a3b_campaign.lock')
    with open(lock_path, 'w') as handle:
        json.dump({'pid': os.getppid(), 'campaign_id': frozen['campaign_id'], 'campaign_spec_sha256': digest}, handle)
    import shared_resources_planning as srp

    def stand_in(args, spec, spec_path, entry_, eval_dir_, lock_content, env_caps, started, progress=None):
        tail_checklist = H.assert_convergence_depth_tail_capture(spec)
        progress['convergence_depth_tail_checklist'] = tail_checklist
        appender = H.ConvergenceDepthAppender(eval_dir_, tail_checklist)
        progress['convergence_depth_appender'] = appender
        with H.convergence_depth_append_hooks(appender):
            srp._drain_network_ipopt_solve_records(planning, None)
            for r in (0, 1):
                _inject_round(planning, w83_records, r)
                srp._drain_network_ipopt_solve_records(planning, r)
            _inject_round(planning, w83_records, 2)
            raise RuntimeError('W85 A3b: simulated failure in cycle 2 (stand-in for _child_real)')

    original_child_real = H._child_real
    saved_env = {k: os.environ.get(k) for k in H.THREAD_CAP_ENV}
    os.environ.update(H.THREAD_CAP_ENV)
    H._child_real = stand_in
    exit_code, stderr = None, io.StringIO()
    try:
        from contextlib import redirect_stderr
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            H.main_child(['--child', '--campaign-root', root, '--spec-sha256', digest, '--eval-key', entry['eval_key'],
                          '--lock-path', lock_path])
    except SystemExit as error:
        exit_code = error.code
    finally:
        H._child_real = original_child_real
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    record = json.load(open(os.path.join(eval_dir, 'evaluation_record.json')))
    rec = record.get('convergence_depth_per_round_append_recovered') or {}
    out = {'exit_code': exit_code, 'child_real_restored': H._child_real is original_child_real,
           'record_status': record.get('status'), 'barrier_cause': record.get('barrier_cause'),
           'failure_drain': record.get('convergence_depth_failure_drain'),
           'recovered': {k: rec.get(k) for k in ('n_records', 'rounds_drained', 'round_drained_at_failure',
                                                 'records_append_sha256', 'records_per_round', 'sealed')},
           'checklist_in_record': (record.get('convergence_depth_tail_checklist_asserted_before_run') or {}).get(
               'tail_enabled_for_this_run'),
           'checklist_in_append': (rec.get('tail_checklist_from_child') or {}).get('tail_enabled_for_this_run'),
           'stderr_tail': stderr.getvalue().splitlines()[-3:]}
    fd = out['failure_drain'] or {}
    out['holds'] = (exit_code == 1 and out['child_real_restored'] and record.get('status') == 'error'
                    and fd.get('status') == 'drained' and fd.get('round_in_flight') == 2 and fd.get('n_records') == 48
                    and rec.get('n_records') == 144 and rec.get('rounds_drained') == [0, 1]
                    and rec.get('round_drained_at_failure') == 2
                    and rec.get('records_append_sha256') == _sha(W83_RECORDS)
                    and out['checklist_in_record'] is True and out['checklist_in_append'] is True)
    return out


# ---- A4: the killed child ------------------------------------------------------------------------------------------
class _Net:
    def __init__(self):
        self.ipopt_solve_records = deque()


class _NetworkData:
    def __init__(self, name):
        self.name = name
        self.years = [2025, 2030, 2035]
        self.days = ['Spring', 'Summer', 'Autumn', 'Winter']
        self.network = {y: {d: _Net() for d in self.days} for y in self.years}


def _duck_planning():
    return SimpleNamespace(transmission_network=_NetworkData('case9'),
                           distribution_networks={5: _NetworkData('case33_1'), 7: _NetworkData('case33_2'),
                                                  9: _NetworkData('case33_3')})


def a4_child_main(eval_dir):
    """Runs in the kill-test SUBPROCESS: drains rounds 0-1 through the hooks with production's drain on a duck-typed
    planning object, puts round 2's records in memory, reports READY with its guard counts, then waits to be killed."""
    import shared_resources_planning as srp
    _ok, records, _tail = _w83_inputs()
    planning = _duck_planning()
    appender = H.ConvergenceDepthAppender(eval_dir, _checklist(dict(DECLARED_ON)))
    with H.convergence_depth_append_hooks(appender):
        srp._drain_network_ipopt_solve_records(planning, None)
        for r in (0, 1):
            _inject_round(planning, records, r)
            srp._drain_network_ipopt_solve_records(planning, r)
        _inject_round(planning, records, 2)
        print(f'{KILL_READY} {json.dumps(dict(GUARD.counts))}', flush=True)
        time.sleep(600)


def check_a4(scratch, frozen_on):
    root, _path, digest, frozen = frozen_on
    entry = next(e for e in frozen['candidates'] if e['label'] == 'n7_4h_e1')
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    os.makedirs(eval_dir)
    cmd = [sys.executable, '-u', os.path.abspath(__file__), '--a4-child', eval_dir]
    err_path = os.path.join(scratch, 'a4_child_stderr.log')
    with open(err_path, 'w') as err:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=err, text=True, stdin=subprocess.DEVNULL)
        ready, t0 = None, time.time()
        while time.time() - t0 < 300:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith(KILL_READY):
                ready = line.strip()
                break
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=60)
    exit_code = proc.returncode
    child_guard = json.loads(ready.split(' ', 1)[1]) if ready else None
    ctx = SimpleNamespace(spec=frozen, spec_path=_path, spec_sha256=digest)
    barrier = H._barrier_record_for_missing(ctx, entry, eval_dir, exit_code)
    rec = barrier.get('convergence_depth_per_round_append_recovered') or {}
    out = {'ready_line': ready, 'child_guard_counts_at_ready': child_guard, 'exit_code': exit_code,
           'barrier_declared_in_spec': barrier.get('convergence_depth_tail_declared_in_spec'),
           'recovered': {k: rec.get(k) for k in ('n_records', 'rounds_drained', 'round_drained_at_failure',
                                                 'records_per_round', 'n_unparsable_record_lines',
                                                 'n_unparsable_event_lines', 'sealed')},
           'checklist_from_child': (rec.get('tail_checklist_from_child') or {}).get('tail_enabled_for_this_run'),
           'unrecoverable': 'round 2 records were in the killed process memory (Network deques) -- not on disk',
           'child_stderr_tail': open(err_path).read().splitlines()[-5:]}
    empty_dir = os.path.join(scratch, 'a4_child_died_before_the_appender')
    os.makedirs(empty_dir)
    out['no_files_recovers_none'] = H.recover_convergence_depth_append(empty_dir) is None
    out['no_files_barrier_field'] = H._barrier_record_for_missing(ctx, entry, empty_dir, 1).get(
        'convergence_depth_per_round_append_recovered', 'absent')
    out['holds'] = (out['no_files_recovers_none'] and out['no_files_barrier_field'] is None and ready is not None and child_guard is not None and not any(child_guard.values())
                    and exit_code == -signal.SIGKILL
                    and out['barrier_declared_in_spec'] == DECLARED_ON
                    and rec.get('n_records') == 96 and rec.get('rounds_drained') == [0, 1]
                    and rec.get('round_drained_at_failure') is None and rec.get('n_unparsable_record_lines') == 0
                    and rec.get('sealed') is False and out['checklist_from_child'] is True)
    return out


def _ordered(text, *needles):
    pos = -1
    for needle in needles:
        nxt = text.find(needle, pos + 1)
        if nxt < 0:
            return False
        pos = nxt
    return True


def check_a6():
    import shared_resources_planning as srp
    child = inspect.getsource(H._child_real)
    main_child = inspect.getsource(H.main_child)
    barrier = inspect.getsource(H._barrier_record_for_missing)
    names = srp._run_operational_planning.__code__.co_names
    wrapped = ('_drain_network_ipopt_solve_records', '_capture_convergence_depth_tail_baseline',
               '_apply_convergence_depth_tail', '_convergence_depth_tail_next_state')
    local_names = srp._run_operational_planning.__code__.co_varnames
    checks = {
        'appender_after_checklist_before_run': _ordered(
            child, 'tail_checklist = assert_convergence_depth_tail_capture(spec)',
            'appender = ConvergenceDepthAppender(eval_dir, tail_checklist)',
            "progress['convergence_depth_appender'] = appender", 'G.run_admm_arm('),
        'append_files_refused_before_the_run': _ordered(
            child, 'NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE, CONVERGENCE_DEPTH_APPEND_EVENTS_FILE,',
            'NETWORK_IPOPT_SOLVE_RECORDS_FILE, CONVERGENCE_DEPTH_TAIL_STATE_FILE)]:',
            'appender = ConvergenceDepthAppender(', 'G.run_admm_arm('),
        'hooks_installed_around_the_run': _ordered(child, 'convergence_depth_append_hooks(appender)',
                                                   'G.run_admm_arm('),
        'reconcile_after_end_of_run_write_then_seal_before_terminal_phase': _ordered(
            child, 'def post_run_hook(', 'persist_convergence_depth_capture(state, eval_dir)',
            "appender.reconcile(holder['convergence_depth_capture'], state)", 'appender.seal()',
            'G.write_boyd_terminal_s35ref(',
            '_terminal_phase(planning, models, rows, report, state, st, optimization_results, primal_evolution)'),
        'record_field_present': "'convergence_depth_per_round_append':" in child,
        'error_flag_includes_the_reconcile': _ordered(child, 'convergence_depth_error = (',
                                                      "(holder.get('convergence_depth_append') or {}).get('ok')"),
        'main_child_drains_on_failure_before_the_record': _ordered(
            main_child, 'except BaseException as error:', "appender = progress.get('convergence_depth_appender')",
            'appender.drain_on_failure(', "_write_once_json(record_path, {",
            "'convergence_depth_failure_drain': failure_drain",
            "'convergence_depth_per_round_append_recovered': recover_convergence_depth_append(eval_dir)"),
        'barrier_carries_declaration_and_recovery': (
            "'convergence_depth_tail_declared_in_spec':" in barrier
            and "'convergence_depth_per_round_append_recovered': recover_convergence_depth_append(eval_dir)" in barrier),
        'production_resolves_wrapped_names_as_globals': (all(n in names for n in wrapped)
                                                         and not any(n in local_names for n in wrapped)),
        'no_production_file_changed_vs_base': _git(['diff', '--name-only', BASE_COMMIT, '--']
                                                   + list(PRODUCTION_FILES)).strip() == '',
    }
    return {'checks': checks, 'holds': all(checks.values())}


# ======================================================================================================================
#  C -- retained behaviour (the committed W84 checks, imported with their guard disarmed)
# ======================================================================================================================
def check_c(scratch, planning, sed, candidate):
    W84 = H.import_disarmed_diagnostic(W84_CHECKS_MODULE)
    from admm_parameters import ADMMParameters
    r_scratch = os.path.join(scratch, 'w84_r')
    os.makedirs(r_scratch)
    r = W84.check_r(r_scratch, planning)
    c1 = W84.check_c1()
    c2 = W84.check_c2(planning, sed, candidate)
    c3 = W84.check_c3()
    c4 = W84.check_c4()
    gate_src = open(_abs(W84_GATE_MODULE_FILE)).read()
    harness_refs = sorted(set(re.findall(r'\bH\.([A-Za-z_]\w*)', gate_src)))
    base_text = _git(['show', f'{BASE_COMMIT}:{HARNESS}'])
    live_text = open(_abs(HARNESS)).read()
    base_defs, live_defs = _module_level_sources(base_text), _module_level_sources(live_text)
    harness_ref_unchanged = {name: (name in base_defs and base_defs.get(name) == live_defs.get(name))
                             for name in harness_refs}
    ref_src = inspect.getsource(H.resolve_post_certification)
    out = {
        'r': {k: v.get('holds') for k, v in r.items() if isinstance(v, dict) and 'holds' in v},
        'r_detail': r, 'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4,
        'tail_default_off': ADMMParameters().convergence_depth_tail == DECLARED_OFF,
        'reference_resolution_reads_no_eval_key': 'eval_key' not in ref_src and 'evaluation_key' not in ref_src,
        'w84_gate_binds_H_through_the_w83_gate': 'H = T83.H' in gate_src,
        'w84_gate_comparator_harness_references': harness_refs,
        'w84_gate_harness_references_unchanged_since_base': harness_ref_unchanged,
    }
    out['w84_gate_uses_harness_only_for_fields_and_sha'] = (
        out['w84_gate_binds_H_through_the_w83_gate'] and bool(harness_refs) and all(harness_ref_unchanged.values()))
    out['holds'] = (all(out['r'].values()) and len(out['r']) == 5 and c1['holds'] and c2['holds'] and c3['holds']
                    and c4['holds'] and out['tail_default_off'] and out['reference_resolution_reads_no_eval_key']
                    and out['w84_gate_uses_harness_only_for_fields_and_sha'])
    return out


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
        with redirect_stdout(io.StringIO()):
            planning, sed, candidate = G._construct_arm_planning(
                'w85_hookcheck', os.path.join(scratch, 'arm_out'), report, k_override=None,
                investment_map=H.investment_map_from_canonical(canonical), eval_id='w85_hookcheck_run',
                num_max_iters_override=2, apply_rho=False)
    finally:
        O.WORK_DIR = original
    return planning, sed, candidate


def run(scratch):
    started = time.time()
    _log('K1 evaluation_key diff vs ' + BASE_COMMIT)
    k1 = check_k1()
    _log(f"    only evaluation_key code changed {k1['k1_only_evaluation_key_code_changed']}; removed lines "
         f"{k1['k1_evaluation_key_diff_removes_only_signature_and_docstring_tail']}; added {k1['n_added_lines']}")
    _log('K2 every committed campaign spec entry, keys recomputed by the live harness')
    k2, all_keys = check_k2()
    _log(f"    specs {k2['n_specs']}, entries {k2['n_entries']}, mismatches {k2['n_mismatches']}, "
         f"specs declaring a tail {k2['n_specs_declaring_a_tail']}")
    _log('K3 freeze reproduction of s47_recert (scratch): undeclared / declared on / declared off')
    k3, frozen_specs = check_k3(scratch)
    _log(f"    a {k3['k3a_undeclared_reproduces_the_committed_spec']} b "
         f"{k3['k3b_declared_on_changes_exactly_the_key_fields']} c "
         f"{k3['k3c_declared_off_changes_exactly_the_key_fields']} d {k3['k3d_on_and_off_keys_differ']}")
    _log('K4/K5 the three pairs')
    k45 = check_k4_k5(all_keys)
    for name, p in k45['pairs'].items():
        _log(f"    {name}: undeclared {p['undeclared_eval_key']} (committed {p['committed_eval_key'][:16]}) -> "
             f"declared on {p['declared_on_eval_key']}")
    _log('building the planning object (G._construct_arm_planning, no solve)')
    planning, sed, candidate = _build_planning(scratch)
    _log('A1/A2/A3/A5 per-round append on the real planning object')
    a, w83_records = check_a1_a2_a3_a5(scratch, planning)
    _log(f"    a1 {a['a1']['holds']} a1b {a['a1b']['holds']} a2 {a['a2']['holds']} a3 {a['a3']['holds']} a5 {a['a5']['holds']}")
    _log('A3b the real main_child exception path (in-process, scratch root and lock)')
    a3b = check_a3b(scratch, frozen_specs['declared_on'], planning, w83_records)
    _log(f"    a3b {a3b['holds']} (exit {a3b['exit_code']})")
    _log('A4 killed child (SIGKILL) and the parent barrier record')
    a4 = check_a4(scratch, frozen_specs['declared_on'])
    _log(f"    a4 {a4['holds']} (exit {a4['exit_code']}, recovered {a4['recovered']['n_records']})")
    a6 = check_a6()
    _log(f"    a6 {a6['holds']}")
    _log('C retained behaviour (W84 checks on the live harness)')
    c = check_c(scratch, planning, sed, candidate)
    _log(f"    c {c['holds']} (r {c['r']}; c1 {c['c1']['holds']} c2 {c['c2']['holds']} c3 {c['c3']['holds']} "
         f"c4 {c['c4']['holds']})")
    items = {
        'K1_only_evaluation_key_code_changed': k1['k1_only_evaluation_key_code_changed'],
        'K1_diff_removes_only_signature_and_docstring_tail': k1['k1_evaluation_key_diff_removes_only_signature_and_docstring_tail'],
        'K2_every_committed_eval_key_recomputed_equal_7016': k2['k2_every_committed_key_recomputed_equal'],
        'K3a_undeclared_freeze_reproduces_s47_recert': k3['k3a_undeclared_reproduces_the_committed_spec'],
        'K3b_declared_on_changes_exactly_the_key_fields': k3['k3b_declared_on_changes_exactly_the_key_fields'],
        'K3c_declared_off_changes_exactly_the_key_fields': k3['k3c_declared_off_changes_exactly_the_key_fields'],
        'K3d_on_and_off_keys_differ': k3['k3d_on_and_off_keys_differ'],
        'K4_three_pairs': k45['k4_three_pairs_differ_and_undeclared_reproduce'],
        'K5_key_properties': k45['k5_properties'],
        'A1_completed_run_append_byte_identical': a['a1']['holds'],
        'A1b_append_write_error_does_not_reach_the_run': a['a1b']['holds'],
        'A2_tail_off_run': a['a2']['holds'],
        'A3_exception_in_flight_recovered': a['a3']['holds'],
        'A3b_main_child_exception_path': a3b['holds'],
        'A4_killed_child_and_parent_barrier': a4['holds'],
        'A5_seal': a['a5']['holds'],
        'A6_wiring': a6['holds'],
        'C_retained_roundtrip_checklist_default_reference': c['holds'],
    }
    return {'stage': STAGE, 'utc': datetime.now(timezone.utc).isoformat(),
            'git_head': _git(['rev-parse', 'HEAD']).strip(), 'base_commit': BASE_COMMIT,
            'script_sha256': _sha(os.path.basename(__file__)), 'harness_sha256': _sha(HARNESS),
            'production_sha256': {f: _sha(f) for f in PRODUCTION_FILES},
            'scratch_dir_note': 'scratch outputs (frozen scratch specs, eval dirs, append files, planning logs dir) are '
                                'outside the repository and not preserved; every value checked is recorded here',
            'K1': k1, 'K2': k2, 'K3': k3, 'K4_K5': k45, 'A': a, 'A3b': a3b, 'A4': a4, 'A6': a6, 'C': c,
            'items': items, 'all_hold': all(items.values()), 'wall_s': time.time() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch-dir')
    parser.add_argument('--a4-child')
    args = parser.parse_args()
    if args.a4_child:
        a4_child_main(args.a4_child)
        return 0
    status = 1
    try:
        _log(STAGE)
        _log(f"git HEAD {_git(['rev-parse', 'HEAD']).strip()}; guard armed permitted=() (zero solves)")
        if not args.scratch_dir:
            raise SystemExit('--scratch-dir is required')
        scratch = os.path.abspath(args.scratch_dir)
        if scratch == REPO or scratch.startswith(REPO + os.sep):
            raise SystemExit(f'--scratch-dir must be outside the repository: {scratch}')
        if os.path.exists(scratch) and os.listdir(scratch):
            raise SystemExit(f'--scratch-dir must be empty or absent: {scratch}')
        os.makedirs(scratch, exist_ok=True)
        for rel in (CHECKS_JSON, CHECKS_MANIFEST):
            if os.path.exists(_abs(rel)):
                raise SystemExit(f'refusing to overwrite existing artifact: {rel}')
        payload = run(scratch)
        os.makedirs(_abs(OUT), exist_ok=True)
        with open(_abs(CHECKS_JSON), 'w') as handle:
            json.dump(payload, handle, indent=1, default=H._json_default)
        inputs = [CHECKS_JSON, os.path.basename(__file__), HARNESS, S47_SPEC, X0_SPEC, W83_RECORDS, W83_TAIL,
                  W83_MANIFEST, W84_CHECKS_MODULE + '.py', W84_GATE_MODULE_FILE] + list(PRODUCTION_FILES)
        with open(_abs(CHECKS_MANIFEST), 'w') as handle:
            json.dump({rel: _sha(rel) for rel in inputs}, handle, indent=2, sort_keys=True)
        for k, v in payload['items'].items():
            _log(f'   {k}: {v}')
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
