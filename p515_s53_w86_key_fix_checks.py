"""
P5.15 Addendum 46 ruling 7, Planner task W86 step 1 -- ZERO-SOLVE verification of the evaluation-key fix in the campaign
harness (`p515_s44_campaign_harness.py`). Nothing here solves: `SolveProfileGuard(permitted=())` is armed at import,
before any production module is imported, and verified at EXACTLY 0 on every exit path.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 46 ruling 7; Planner task W86, ruling Q1 on W85:
  "a declared-off tail must share the UNDECLARED key. Follow the W33 flex-multiplier precedent exactly (m != 1.0 enters
   the key, m == 1.0 does not). Nothing in production changes when the tail is declared off ... Re-run the 7,016-entry
   check after the change, and confirm the declared-OFF key now equals the undeclared key for all three cells while
   the declared-ON keys are unchanged from your table (96c5aa50..., ca8927e7..., 5cfe69a6...)."

THE FIX (harness only): `convergence_depth_tail_in_key(value)` -- validated declaration when enabled True, else None
(absent or enabled False), the `flex_price_multiplier_in_key` pattern -- and `evaluation_key` branches on it instead of
on `validate_convergence_depth_tail`. Production: with `enabled` False none of the tail functions is called and
`compl_inf_tol` is never read (`shared_resources_planning._run_operational_planning`, `convergence_depth_tail_enabled`),
so a declared-off evaluation IS the undeclared computation -- checked below from production's source.

CHECKS (all must hold):
  K1 vs BASE_COMMIT (HEAD when W86 started, 4f8a1dea): every module-level definition of the harness is source-identical
     except exactly {evaluation_key, validate_convergence_depth_tail, freeze_campaign_spec} and the one ADDED function
     `convergence_depth_tail_in_key`; validate_convergence_depth_tail and freeze_campaign_spec differ in their docstrings
     only (AST without docstring equal); evaluation_key's AST without docstring equals the base one once the single
     call `convergence_depth_tail_in_key` is renamed back to `validate_convergence_depth_tail` (the only code change);
  K2 the committed 7,016-entry check (W85's `check_k2`, BY IMPORT, unchanged): every entry of every git-tracked
     campaign spec (50) recomputed by the live harness with the spec's own declarations -- 0 mismatches;
  K3 freeze reproduction of the committed s47_recert spec in a scratch root (W85's `check_k3`, BY IMPORT): undeclared
     reproduces the committed spec; declared ON changes exactly {eval_key, eval_dir, working_dir_ids} of each entry and
     gives the W85 table's ON keys; declared OFF now leaves EVERY candidate entry identical to the committed one (eval
     key, eval dir, working-dir ids), the configuration differing from the undeclared freeze only by
     `convergence_depth_tail` (recorded, not keyed);
  K4 the three cells (C* and the unit under the s47_recert declarations, x = 0 under the s48 x0_capture declarations --
     the C2 baseline): declared OFF key == undeclared key == committed key; declared ON key == the W85 table's value
     (read from the committed checks_w85.json, whose sha256 is verified against its committed manifest) and absent
     from every committed spec; OFF with another tolerance (1e-7) also == undeclared; ON sensitive to compl_inf_tol;
     invalid declarations (ON or OFF) still refused by `evaluation_key`;
  K5 production: with enabled False no tail function is reached -- `_run_operational_planning` guards every call of
     `_capture_convergence_depth_tail_baseline`, `_apply_convergence_depth_tail`, `_convergence_depth_tail_next_state`
     behind `convergence_depth_tail_on` (source check), and `convergence_depth_tail_enabled` returns False for a
     declared-off dict (called).

EXACT LAUNCH COMMAND (repo root; attached, both streams captured):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_w86_key_fix_checks.py --scratch-dir <an EMPTY directory outside the repository> \\
        > data/SRP1/Results/P515S53/tight_tail_w86/key_fix_checks_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/tight_tail_w86/key_fix_checks/{checks_w86_key.json,
checks_w86_key_manifest_sha256.json}. Exit 0 when every check holds, 1 otherwise.
"""

import argparse
import ast
import copy
import inspect
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
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W86 key-fix zero-solve checks').install()

import p515_s44_campaign_harness as H  # noqa: E402

W85 = H.import_disarmed_diagnostic('p515_s53_w85_harness_checks')   # its check_k2 / check_k3 / _entry_key, unchanged

STAGE = ('P5.15 Addendum 46 ruling 7, W86 step 1 -- evaluation_key: a declared-OFF convergence-depth tail shares the '
         'undeclared key (Planner ruling Q1 on W85, the W33 precedent); declared-ON keys unchanged')
BASE_COMMIT = '4f8a1dea'   # HEAD when W86 started
HARNESS = 'p515_s44_campaign_harness.py'
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT = os.path.join(S53, 'tight_tail_w86', 'key_fix_checks')
CHECKS_JSON = os.path.join(OUT, 'checks_w86_key.json')
CHECKS_MANIFEST = os.path.join(OUT, 'checks_w86_key_manifest_sha256.json')
W85_CHECKS = os.path.join(S53, 'tight_tail_w85', 'harness_checks', 'checks_w85.json')
W85_MANIFEST = os.path.join(S53, 'tight_tail_w85', 'harness_checks', 'checks_w85_manifest_sha256.json')
PRODUCTION_FILES = W85.PRODUCTION_FILES
DECLARED_ON = {'enabled': True, 'compl_inf_tol': 1e-6}
DECLARED_OFF = {'enabled': False, 'compl_inf_tol': 1e-6}
EXPECTED_ON_PREFIX = {'c_star': '96c5aa50', 'unit': 'ca8927e7', 'x0': '5cfe69a6'}   # the Planner's task text
CHANGED = {'evaluation_key', 'validate_convergence_depth_tail', 'freeze_campaign_spec'}
ADDED = {'convergence_depth_tail_in_key'}


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W86-key] {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True, check=True).stdout


def _raises(fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        return f'{type(error).__name__}: {str(error)[:300]}'
    return None


class _Rename(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id == 'convergence_depth_tail_in_key':
            return ast.copy_location(ast.Name(id='validate_convergence_depth_tail', ctx=node.ctx), node)
        return node


def check_k1():
    base_text = _git(['show', f'{BASE_COMMIT}:{HARNESS}'])
    live_text = open(_abs(HARNESS)).read()
    base_src, live_src = W85._module_level_sources(base_text), W85._module_level_sources(live_text)
    changed = sorted(n for n in set(base_src) & set(live_src) if base_src[n] != live_src[n])
    added, removed = sorted(set(live_src) - set(base_src)), sorted(set(base_src) - set(live_src))
    base_fns, live_fns = W85._functions(base_text), W85._functions(live_text)
    docstring_only = {n: W85._code_dump(base_fns[n]) == W85._code_dump(live_fns[n])
                      for n in ('validate_convergence_depth_tail', 'freeze_campaign_spec')}
    live_ek = _Rename().visit(copy.deepcopy(live_fns['evaluation_key']))
    ek_code_identical_up_to_the_call = W85._code_dump(base_fns['evaluation_key']) == W85._code_dump(live_ek)
    ek_code_differs = W85._code_dump(base_fns['evaluation_key']) != W85._code_dump(live_fns['evaluation_key'])
    base_doc = ast.get_docstring(ast.parse(base_text))
    live_doc = ast.get_docstring(ast.parse(live_text))
    return {'base_commit': BASE_COMMIT, 'changed_definitions': changed, 'added_definitions': added,
            'removed_definitions': removed, 'docstring_only': docstring_only,
            'evaluation_key_code_differs': ek_code_differs,
            'evaluation_key_code_identical_once_the_new_call_is_renamed_back': ek_code_identical_up_to_the_call,
            'module_docstring_changed': base_doc != live_doc,
            'new_function_source': live_src.get('convergence_depth_tail_in_key'),
            'k1_holds': (set(changed) == CHANGED and set(added) == ADDED and not removed
                         and all(docstring_only.values()) and ek_code_differs and ek_code_identical_up_to_the_call)}


def check_k3(scratch, w85_on_keys):
    k3, _frozen = W85.check_k3(scratch)
    r = k3['results']
    und, on, off = r['undeclared'], r['declared_on'], r['declared_off']
    changed = {'eval_key', 'eval_dir', 'working_dir_ids'}
    on_ok = (set(on['configuration_keys_differing_from_committed'])
             == set(und['configuration_keys_differing_from_committed']) | {'convergence_depth_tail'}
             and on['convergence_depth_tail'] == DECLARED_ON
             and all(e['label_matches'] and set(e['fields_differing']) == changed and e['eval_key_recomputes']
                     and e['eval_key_frozen_now'] == w85_on_keys[lab] for lab, e in on['per_entry'].items()))
    off_ok = (off['candidates_identical_to_committed']
              and set(off['configuration_keys_differing_from_committed'])
              == set(und['configuration_keys_differing_from_committed']) | {'convergence_depth_tail'}
              and off['convergence_depth_tail'] == DECLARED_OFF
              and all(e['label_matches'] and e['fields_differing'] == [] and e['eval_key_recomputes']
                      and e['eval_key_frozen_now'] == e['eval_key_committed'] for e in off['per_entry'].values()))
    return {'w85_check_k3_result': k3,
            'k3a_undeclared_reproduces_the_committed_spec': k3['k3a_undeclared_reproduces_the_committed_spec'],
            'k3b_declared_on_changes_exactly_the_key_fields_and_gives_the_w85_keys': on_ok,
            'k3c_declared_off_candidates_identical_to_the_committed_spec': off_ok}


def check_k4(all_committed_keys, w85_pairs):
    s47 = json.load(open(_abs(W85.S47_SPEC)))
    x0s = json.load(open(_abs(W85.X0_SPEC)))
    cells = {'c_star': (s47, next(e for e in s47['candidates'] if e['label'] == 'c_star')),
             'unit': (s47, next(e for e in s47['candidates'] if e['label'] == 'n7_4h_e1')),
             'x0': (x0s, next(e for e in x0s['candidates'] if e['label'] == 'x0'))}
    pairs = {}
    for name, (spec, entry) in cells.items():
        und = W85._entry_key(spec, entry, tail=None)
        on = W85._entry_key(spec, entry, tail=dict(DECLARED_ON))
        off = W85._entry_key(spec, entry, tail=dict(DECLARED_OFF))
        off_other_tol = W85._entry_key(spec, entry, tail={'enabled': False, 'compl_inf_tol': 1e-7})
        on_tighter = W85._entry_key(spec, entry, tail={'enabled': True, 'compl_inf_tol': 1e-7})
        w85_on = w85_pairs[name]['declared_on_eval_key']
        pairs[name] = {
            'label': entry['label'], 'canonical': entry['canonical'],
            'spec': W85.X0_SPEC if name == 'x0' else W85.S47_SPEC,
            'committed_eval_key': entry['eval_key'], 'undeclared_eval_key': und, 'declared_off_eval_key': off,
            'declared_off_1e-7_eval_key': off_other_tol, 'declared_on_eval_key': on,
            'w85_declared_on_eval_key': w85_on, 'w85_declared_off_eval_key': w85_pairs[name]['declared_off_eval_key'],
            'undeclared_equals_committed': und == entry['eval_key'],
            'declared_off_equals_undeclared': off == und,
            'declared_off_any_tolerance_equals_undeclared': off_other_tol == und,
            'declared_on_unchanged_from_w85': on == w85_on,
            'declared_on_matches_planner_prefix': on.startswith(EXPECTED_ON_PREFIX[name]),
            'declared_on_differs_from_undeclared': on != und,
            'declared_on_sensitive_to_compl_inf_tol': on_tighter != on,
            'declared_on_absent_from_every_committed_spec': on not in all_committed_keys,
        }
    invalid = {'on_int_tol': {'enabled': True, 'compl_inf_tol': 1}, 'off_int_tol': {'enabled': False, 'compl_inf_tol': 1},
               'off_negative_tol': {'enabled': False, 'compl_inf_tol': -1e-6},
               'extra_key': {**DECLARED_OFF, 'x': 1}, 'not_a_dict': False,
               'string_enabled': {'enabled': 'no', 'compl_inf_tol': 1e-6}}
    refused = {k: _raises(lambda v=v: H.evaluation_key('0' * 64, {}, convergence_depth_tail=v))
               for k, v in invalid.items()}
    holds = (all(all(v for k, v in p.items() if isinstance(v, bool)) for p in pairs.values())
             and all(v is not None for v in refused.values()))
    return {'pairs': pairs, 'invalid_declarations_refused': refused, 'k4_holds': holds}


def check_k5():
    import shared_resources_planning as srp
    src = inspect.getsource(srp._run_operational_planning)
    lines = src.splitlines()
    guarded = {}
    for fn in ('_capture_convergence_depth_tail_baseline(', '_apply_convergence_depth_tail(',
               '_convergence_depth_tail_next_state('):
        idx = [i for i, line in enumerate(lines) if fn in line and not line.strip().startswith('#')]
        per_call = []
        for i in idx:
            indent = len(lines[i]) - len(lines[i].lstrip())
            j = i - 1
            guard_line = None
            while j >= 0:   # the nearest enclosing `if` at a smaller indent
                lj = lines[j]
                ind = len(lj) - len(lj.lstrip())
                if lj.strip() and ind < indent and lj.strip().startswith(('if ', 'for ', 'while ', 'with ', 'try',
                                                                          'def ', 'else', 'elif ')):
                    guard_line = lj.strip()
                    break
                j -= 1
            per_call.append({'line': lines[i].strip(), 'enclosing': guard_line,
                             'guarded_by_tail_on': guard_line == 'if convergence_depth_tail_on:'})
        guarded[fn.rstrip('(')] = per_call
    on_flag = 'convergence_depth_tail_on = convergence_depth_tail_state[\'enabled\']' in src
    state_line = ("convergence_depth_tail_state = {'enabled': convergence_depth_tail_enabled(admm_parameters)}" in src)
    enabled_off = srp.convergence_depth_tail_enabled(type('A', (), {'convergence_depth_tail': dict(DECLARED_OFF)})())
    enabled_on = srp.convergence_depth_tail_enabled(type('A', (), {'convergence_depth_tail': dict(DECLARED_ON)})())
    holds = (on_flag and state_line and enabled_off is False and enabled_on is True
             and all(c and all(x['guarded_by_tail_on'] for x in c) for c in guarded.values()))
    return {'calls': guarded, 'tail_on_flag_from_state': on_flag, 'state_from_convergence_depth_tail_enabled': state_line,
            'convergence_depth_tail_enabled_declared_off': enabled_off,
            'convergence_depth_tail_enabled_declared_on': enabled_on, 'k5_holds': holds}


def run(scratch):
    started = time.time()
    w85_manifest = json.load(open(_abs(W85_MANIFEST)))
    w85_ok = w85_manifest.get(W85_CHECKS) == _sha(W85_CHECKS)
    w85 = json.load(open(_abs(W85_CHECKS)))
    w85_pairs = w85['K4_K5']['pairs']
    w85_on_keys = {e: v['eval_key_frozen_now'] for e, v in w85['K3']['results']['declared_on']['per_entry'].items()}
    _log(f'committed W85 checks {W85_CHECKS}: sha256 matches its manifest {w85_ok}')
    _log('K1 harness diff vs ' + BASE_COMMIT)
    k1 = check_k1()
    _log(f"    changed {k1['changed_definitions']} added {k1['added_definitions']} removed {k1['removed_definitions']} "
         f"docstring-only {k1['docstring_only']} ek-code-up-to-the-call "
         f"{k1['evaluation_key_code_identical_once_the_new_call_is_renamed_back']} -> {k1['k1_holds']}")
    _log('K2 every committed campaign spec entry, keys recomputed by the live harness (W85 check_k2 by import)')
    k2, all_keys = W85.check_k2()
    _log(f"    specs {k2['n_specs']}, entries {k2['n_entries']}, mismatches {k2['n_mismatches']}, "
         f"specs declaring a tail {k2['n_specs_declaring_a_tail']} -> {k2['k2_every_committed_key_recomputed_equal']}")
    _log('K3 freeze reproduction of s47_recert (scratch): undeclared / declared on / declared off')
    k3 = check_k3(scratch, w85_on_keys)
    _log(f"    a {k3['k3a_undeclared_reproduces_the_committed_spec']} b "
         f"{k3['k3b_declared_on_changes_exactly_the_key_fields_and_gives_the_w85_keys']} c "
         f"{k3['k3c_declared_off_candidates_identical_to_the_committed_spec']}")
    _log('K4 the three cells')
    k4 = check_k4(all_keys, w85_pairs)
    for name, p in k4['pairs'].items():
        _log(f"    {name}: undeclared {p['undeclared_eval_key']} (committed {p['committed_eval_key'][:16]}) | "
             f"declared OFF {p['declared_off_eval_key']} (W85: {p['w85_declared_off_eval_key'][:16]}) | declared ON "
             f"{p['declared_on_eval_key']} (W85: {p['w85_declared_on_eval_key'][:16]})")
    _log(f"    refused: {k4['invalid_declarations_refused']} -> {k4['k4_holds']}")
    _log('K5 production: declared-off reaches no tail function')
    k5 = check_k5()
    _log(f"    {k5['k5_holds']}")
    items = {
        'W85_checks_artifact_hash_verified': w85_ok,
        'K1_only_the_key_branch_changed': k1['k1_holds'],
        'K2_every_committed_eval_key_recomputed_equal_7016': k2['k2_every_committed_key_recomputed_equal'],
        'K3a_undeclared_freeze_reproduces_s47_recert': k3['k3a_undeclared_reproduces_the_committed_spec'],
        'K3b_declared_on_keys_unchanged_from_w85': k3['k3b_declared_on_changes_exactly_the_key_fields_and_gives_the_w85_keys'],
        'K3c_declared_off_freeze_candidates_identical_to_committed': k3['k3c_declared_off_candidates_identical_to_the_committed_spec'],
        'K4_three_cells_off_equals_undeclared_on_unchanged': k4['k4_holds'],
        'K5_production_declared_off_is_the_undeclared_computation': k5['k5_holds'],
    }
    return {'stage': STAGE, 'utc': datetime.now(timezone.utc).isoformat(),
            'git_head': _git(['rev-parse', 'HEAD']).strip(), 'base_commit': BASE_COMMIT,
            'script_sha256': _sha(os.path.basename(__file__)), 'harness_sha256': _sha(HARNESS),
            'imported_w85_checks_script_sha256': _sha('p515_s53_w85_harness_checks.py'),
            'production_sha256': {f: _sha(f) for f in PRODUCTION_FILES},
            'scratch_dir_note': 'the three scratch freezes of K3 are outside the repository and not preserved; every value '
                                'checked is recorded here',
            'K1': k1, 'K2': k2, 'K3': k3, 'K4': k4, 'K5': k5,
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
        for rel in (CHECKS_JSON, CHECKS_MANIFEST):
            if os.path.exists(_abs(rel)):
                raise SystemExit(f'refusing to overwrite existing artifact: {rel}')
        payload = run(scratch)
        os.makedirs(_abs(OUT), exist_ok=True)
        with open(_abs(CHECKS_JSON), 'x') as handle:
            json.dump(payload, handle, indent=1, default=H._json_default)
        inputs = [CHECKS_JSON, os.path.basename(__file__), HARNESS, 'p515_s53_w85_harness_checks.py', W85.S47_SPEC,
                  W85.X0_SPEC, W85_CHECKS, W85_MANIFEST] + list(PRODUCTION_FILES)
        with open(_abs(CHECKS_MANIFEST), 'x') as handle:
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
