"""
P5.15 Addendum 25 item 2 -- Task 2: ZERO-SOLVE checks of the Anderson-
acceleration reject-policy sub-option (`anderson_acceleration['reject_policy']`,
'clear_memory' default = the Step 3.7 behaviour, or 'keep_memory' = the
Addendum 25 variant).

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 25; frozen spec v14
`data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`,
`item2_aa_variant`.

Checks:
  V0 The committed Step 3.7 zero-solve checks A, B, C of `p515_s41_aa_checks.py`
     (BY IMPORT, unchanged) still pass on the modified module; B's and C's full
     outputs equal the COMMITTED outputs
     (`data/SRP1/Results/P515S41/aa_checks/aa_checks.json`) exactly.
  V1 Default path unchanged, BITWISE: the Step 3.7 module as committed at
     `9a965494` (read with `git show`, executed as a separate module object)
     and the modified module -- default constructor AND explicit
     `reject_policy='clear_memory'` -- are driven through identical randomized
     sequences (a nonlinear contractive map, random certificate-independence
     cycles, random rho-change clears and failure skips, residuals inflated at
     random to force rejections); every returned iterate (`np.array_equal`) and
     every record (dict equality) must be identical. Non-vacuity: the sequences
     must contain accepts, rejections, rho clears, failure skips and 'off'
     cycles.
  V2 `keep_memory` semantics on synthetic sequences (the REAL class): a
     rejection takes the plain iterate, leaves the mark unchanged, leaves the
     memory at its post-append size (the rejection itself removes nothing) and
     has the cycle's (w, g) pair appended as the newest pair; at full memory the
     oldest pair is dropped by the deque exactly as on an accept; the next
     attempt uses the retained memory; a rho change still clears; a failure
     cycle still clears; against 'clear_memory' on the same inputs the two
     policies are identical up to and including the first rejection's iterate,
     differing only in that record's action/reset/memory fields; an unknown
     policy is refused.
  V3 Wiring (static, ast): `_run_operational_planning` passes
     `reject_policy=aa_settings.get('reject_policy', DEFAULT_REJECT_POLICY)`
     to `AndersonAccelerationState`, inside an `if aa_enabled:` block (Check A
     of V0 covers the guard for every AA reference); the ADMMParameters
     default dict is unchanged (no `reject_policy` key).

Armed `SolveProfileGuard(permitted=())`, verified 0 solves and 0 execs.
Write-once output root (argv suffix redirects it):
    data/SRP1/Results/P515S44/aa_variant_checks/

Launch (attached, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_aa_variant_checks.py \\
        > data/SRP1/Results/P515S44/aa_variant_checks_launch.log 2>&1
"""

import ast
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
import types
from datetime import datetime, timezone

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15-S44 AA variant zero-solve checks').install()

import admm_anderson_acceleration as AA  # noqa: E402
import admm_parameters  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import p515_s41_aa_checks as S41  # noqa: E402 -- checks A, B, C, BY IMPORT, unchanged

_SUFFIX = next(('_' + a.strip('_') for a in sys.argv[1:] if not a.startswith('--')), '')
OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', f'aa_variant_checks{_SUFFIX}')
COMMITTED_S41 = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S41', 'aa_checks', 'aa_checks.json')
PRE_VARIANT_COMMIT = '9a965494'  # the Step 3.7 AA module (run 76095561 used it; last commit touching it before this task)
KEEP = 'rejected (safeguard; memory retained)'


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _json_roundtrip(x):
    return json.loads(json.dumps(x, default=str))


# ======================================================================================================================
#  V0 -- committed Step 3.7 checks A, B, C still pass; B and C reproduce the committed output exactly
# ======================================================================================================================
def v0_committed_checks():
    with open(COMMITTED_S41) as handle:
        committed = json.load(handle)
    a = S41.check_a_flag_off()
    b = S41.check_b_linear_map()
    c = S41.check_c_state_machine()
    checks = {
        'check_A_passes': a['passed'],
        'check_A_no_unguarded_aa_reference': a['n_unguarded_call_sites'] == 0,
        'check_A_default_settings_unchanged': a['default_settings_as_expected'],
        'check_B_passes': b['passed'],
        'check_C_passes': c['passed'],
        'check_B_output_equals_committed': _json_roundtrip(b) == committed['check_B_linear_map'],
        'check_C_output_equals_committed': _json_roundtrip(c) == committed['check_C_state_machine'],
    }
    return {'checks': checks, 'check_A': a, 'committed_path': os.path.relpath(COMMITTED_S41, REPO),
            'committed_sha256': _sha_bytes(open(COMMITTED_S41, 'rb').read())}


# ======================================================================================================================
#  V1 -- default path bitwise identical to the committed Step 3.7 module
# ======================================================================================================================
def _load_committed_module():
    src = subprocess.run(['git', 'show', f'{PRE_VARIANT_COMMIT}:admm_anderson_acceleration.py'], cwd=REPO,
                         capture_output=True, check=True).stdout
    mod = types.ModuleType('admm_anderson_acceleration_committed_9a965494')
    exec(compile(src, f'<git {PRE_VARIANT_COMMIT}:admm_anderson_acceleration.py>', 'exec'), mod.__dict__)
    blob = subprocess.run(['git', 'rev-parse', f'{PRE_VARIANT_COMMIT}:admm_anderson_acceleration.py'], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout.strip()
    return mod, {'commit': PRE_VARIANT_COMMIT, 'git_blob': blob, 'source_sha256': _sha_bytes(src)}


def _drive(states, seed, n=6, cycles=200, memory=5):
    """Drive every state in `states` through the SAME randomized sequence; return
    (per-state list of (w, record)), event counts. The iterate fed to each state
    is its own previous output, so any divergence propagates and is caught."""
    rng = np.random.default_rng(seed)
    a_mat = rng.normal(size=(n, n))
    a_mat *= 0.95 / max(abs(np.linalg.eigvals(a_mat)))
    b_vec = rng.normal(size=n)

    def f(w):
        return a_mat @ w + 0.05 * np.tanh(w) + b_vec

    plan = []
    for k in range(1, cycles + 1):
        u = rng.random()
        plan.append({
            'k': k,
            'failure': u < 0.03,
            'rho_change': 0.03 <= u < 0.08,
            'all_pass': 0.08 <= u < 0.14,
            'inflate': rng.random() < 0.25,
            'inflate_factor': float(1.0 + 5.0 * rng.random()),
        })
    traces = [[] for _ in states]
    ws = [np.zeros(n) for _ in states]
    events = {'accepted': 0, 'rejected': 0, 'rho': 0, 'failure': 0, 'off': 0, 'insufficient': 0}
    for step in plan:
        for i, st in enumerate(states):
            w = ws[i]
            if step['failure']:
                rec = st.skip_on_failure(step['k'])
                traces[i].append((w.copy(), dict(rec)))
                ws[i] = f(w)  # the plain cycle still moved the stores
                continue
            g = f(w) - w
            res = float(np.linalg.norm(g)) * (step['inflate_factor'] if step['inflate'] else 1.0)
            w_next, rec = st.step(step['k'], w, g, res, step['all_pass'])
            rec = dict(rec)
            if step['rho_change']:
                rec2 = st.clear_for_rho_change(step['k'], ['pf'])
                rec['rho_changed_channels'] = ['pf']
                rec['memory_size_after'] = rec2['memory_size_after']
            traces[i].append((np.array(w_next, copy=True), rec))
            ws[i] = w_next
        # count on the first state only
        rec0 = traces[0][-1][1]
        act = rec0.get('action', '')
        if step['failure']:
            events['failure'] += 1
        elif act == 'accepted':
            events['accepted'] += 1
        elif act.startswith('rejected'):
            events['rejected'] += 1
        elif act.startswith('off'):
            events['off'] += 1
        elif act.startswith('insufficient'):
            events['insufficient'] += 1
        if step['rho_change'] and not step['failure']:
            events['rho'] += 1
    return traces, events


def _records_equal(r1, r2):
    if set(r1) != set(r2):
        return False
    for k in r1:
        x, y = r1[k], r2[k]
        if isinstance(x, float) and isinstance(y, float):
            if not (x == y or (x != x and y != y)):
                return False
        elif x != y:
            return False
    return True


def v1_default_path_bitwise():
    old, provenance = _load_committed_module()
    totals = {'accepted': 0, 'rejected': 0, 'rho': 0, 'failure': 0, 'off': 0, 'insufficient': 0}
    mismatches = []
    seeds = list(range(20260919, 20260919 + 24))
    for seed in seeds:
        states = [old.AndersonAccelerationState(memory=5, regularization=1e-10),
                  AA.AndersonAccelerationState(memory=5, regularization=1e-10),
                  AA.AndersonAccelerationState(memory=5, regularization=1e-10, reject_policy='clear_memory')]
        traces, events = _drive(states, seed)
        for k in totals:
            totals[k] += events[k]
        for j in (1, 2):
            for c, ((w0, r0), (wj, rj)) in enumerate(zip(traces[0], traces[j])):
                if not (np.array_equal(w0, wj) and _records_equal(r0, rj)):
                    mismatches.append({'seed': seed, 'variant': j, 'cycle': c + 1, 'old_record': r0, 'new_record': rj})
                    break
    # Sensitivity (the same comparison must SEE a policy change): committed module vs keep_memory.
    sensitivity_seeds_differing = 0
    for seed in seeds:
        states = [old.AndersonAccelerationState(memory=5, regularization=1e-10),
                  AA.AndersonAccelerationState(memory=5, regularization=1e-10, reject_policy='keep_memory')]
        traces, _events = _drive(states, seed)
        if any(not (np.array_equal(w0, w1) and _records_equal(r0, r1))
               for (w0, r0), (w1, r1) in zip(traces[0], traces[1])):
            sensitivity_seeds_differing += 1
    checks = {
        'no_mismatch_any_seed_default_or_explicit_clear_memory': not mismatches,
        'sensitivity_keep_memory_detected_as_different_in_every_seed': sensitivity_seeds_differing == len(seeds),
        'sequences_contain_accepts': totals['accepted'] > 0,
        'sequences_contain_rejections': totals['rejected'] > 0,
        'sequences_contain_rho_clears': totals['rho'] > 0,
        'sequences_contain_failure_skips': totals['failure'] > 0,
        'sequences_contain_off_cycles': totals['off'] > 0,
        'default_constructor_policy_is_clear_memory': AA.AndersonAccelerationState().reject_policy == 'clear_memory',
    }
    return {'checks': checks, 'pre_variant_module': provenance, 'seeds': [seeds[0], seeds[-1]],
            'n_sequences': len(seeds), 'cycles_per_sequence': 200, 'event_totals': totals,
            'sensitivity_seeds_differing_keep_memory_vs_committed': sensitivity_seeds_differing,
            'mismatches': mismatches[:5]}


# ======================================================================================================================
#  V2 -- keep_memory semantics
# ======================================================================================================================
def v2_keep_memory():
    # An 8-D affine contraction (spectral radius 0.9): AA does not reach the fixed point within the
    # memory, so iterates stay distinct and "which step was taken" is observable in w itself.
    rng = np.random.default_rng(20260919)
    n = 8
    a_mat = rng.normal(size=(n, n))
    a_mat *= 0.9 / max(abs(np.linalg.eigvals(a_mat)))
    b_vec = rng.normal(size=n)

    def f(w):
        return a_mat @ w + b_vec

    out = {}
    # (1) build memory to 3 columns with accepted steps (residual strictly falling), then force a rejection
    st = AA.AndersonAccelerationState(memory=5, regularization=1e-10, reject_policy='keep_memory')
    w = np.zeros(n)
    resid = 100.0
    recs = []
    for cyc in (1, 2, 3, 4):
        g = f(w) - w
        resid *= 0.5
        w, rec = st.step(cyc, w, g, resid, False)
        recs.append(rec)
    mem_before = st.memory_size()
    mark_before = st.last_accepted_residual
    hist_len_before = len(st._history)
    g5 = f(w) - w
    w5_in = w
    w5, rec5 = st.step(5, w5_in, g5, mark_before * 10.0, False)
    newest = st._history[-1]
    out['reject_below_capacity'] = {
        'actions_before': [r['action'] for r in recs], 'memory_before': mem_before, 'record': rec5,
        'memory_after': st.memory_size(), 'history_len_before': hist_len_before, 'history_len_after': len(st._history),
        'mark_before': mark_before, 'mark_after': st.last_accepted_residual}
    ok1 = (
        [r['action'] for r in recs] == ['insufficient memory (m_k=0)'] + ['accepted'] * 3
        and rec5['action'] == KEEP and rec5['accepted'] is False and rec5['reset'] is False
        and rec5['reset_reason'] is None
        and rec5['memory_size_before'] == mem_before == 3 and rec5['memory_size_after'] == mem_before + 1
        and st.memory_size() == mem_before + 1 and len(st._history) == hist_len_before + 1
        and newest[0] is w5_in and newest[1] is g5          # this cycle's pair, appended and retained
        and np.array_equal(w5, w5_in + g5)                  # plain iterate taken
        and st.last_accepted_residual == mark_before        # mark unchanged
        and rec5['gamma_columns'] == mem_before + 1
    )

    # (2) the next attempt extrapolates from the retained memory
    g6 = f(w5) - w5
    w6, rec6 = st.step(6, w5, g6, mark_before * 0.5, False)
    out['next_attempt_after_reject'] = rec6
    ok2 = (rec6['action'] == 'accepted' and rec6['memory_size_before'] == mem_before + 1
           and rec6['gamma_columns'] == min(mem_before + 2, 5) and not np.array_equal(w6, w5 + g6))

    # (3) at full memory a rejection keeps 5 columns; the deque drops the oldest pair exactly as on an accept
    st_full = AA.AndersonAccelerationState(memory=5, regularization=1e-10, reject_policy='keep_memory')
    w = np.zeros(n)
    resid = 100.0
    for cyc in range(1, 8):
        g = f(w) - w
        resid *= 0.5
        w, _r = st_full.step(cyc, w, g, resid, False)
    full_before = st_full.memory_size()
    oldest_pair_before = st_full._history[0]
    second_pair_before = st_full._history[1]
    g_r = f(w) - w
    w_r_in = w
    w_r, rec_r = st_full.step(8, w_r_in, g_r, st_full.last_accepted_residual * 3.0, False)
    out['reject_at_full_memory'] = {'memory_before': full_before, 'record': rec_r, 'memory_after': st_full.memory_size()}
    ok3 = (full_before == 5 and rec_r['action'] == KEEP and rec_r['memory_size_after'] == 5
           and st_full.memory_size() == 5 and st_full._history[0] is second_pair_before
           and all(p is not oldest_pair_before for p in st_full._history)
           and st_full._history[-1][0] is w_r_in and st_full._history[-1][1] is g_r
           and np.array_equal(w_r, w_r_in + g_r))

    # (4) a rho change still clears; (5) a failure cycle still clears
    rho_rec = st_full.clear_for_rho_change(9, ['v'])
    after_rho = st_full.memory_size()
    g10 = f(w_r) - w_r
    w10, rec10 = st_full.step(10, w_r, g10, 1e-9, False)
    for cyc in (11, 12, 13):
        g = f(w10) - w10
        w10, _r = st_full.step(cyc, w10, g, 1e-12 / cyc, False)
    before_fail = st_full.memory_size()
    fail_rec = st_full.skip_on_failure(14)
    after_fail = st_full.memory_size()
    out['rho_change_clear'] = {'record': rho_rec, 'memory_after': after_rho, 'next_record': rec10}
    out['failure_clear'] = {'memory_before': before_fail, 'record': fail_rec, 'memory_after': after_fail}
    ok4 = (rho_rec['action'] == 'memory reset (rho change)' and after_rho == 0
           and rec10['action'] == 'insufficient memory (m_k=0)')
    ok5 = (before_fail == 3 and fail_rec['reset'] is True and after_fail == 0)

    # (6) keep vs clear on the same inputs: identical through the first rejection (cycle 6), differing from
    #     cycle 7 (keep extrapolates from its retained memory; clear has one pair only -> plain)
    schedule = [100.0 * 0.5 ** c for c in range(1, 6)] + [1e6] + [100.0 * 0.5 ** c for c in range(6, 9)]
    seqs = {}
    for pol in ('clear_memory', 'keep_memory'):
        s = AA.AndersonAccelerationState(memory=5, regularization=1e-10, reject_policy=pol)
        w = np.zeros(n)
        trace = []
        for cyc, res in enumerate(schedule, start=1):
            g = f(w) - w
            w, rec = s.step(cyc, w, g, res, False)
            trace.append((w.copy(), dict(rec)))
        seqs[pol] = trace
    first_diff_cycle = None
    for c, ((w_c, _r_c), (w_k, _r_k)) in enumerate(zip(seqs['clear_memory'], seqs['keep_memory'])):
        if not np.array_equal(w_c, w_k):
            first_diff_cycle = c + 1
            break
    rej_c, rej_k = seqs['clear_memory'][5][1], seqs['keep_memory'][5][1]
    differing_fields_at_reject = sorted(k for k in set(rej_c) | set(rej_k) if rej_c.get(k) != rej_k.get(k))
    same_before = all(_records_equal(seqs['clear_memory'][c][1], seqs['keep_memory'][c][1]) for c in range(5))
    c7 = (seqs['clear_memory'][6][1]['action'], seqs['keep_memory'][6][1]['action'])
    out['keep_vs_clear'] = {'residual_schedule': schedule, 'first_iterate_difference_cycle': first_diff_cycle,
                            'reject_record_clear': rej_c, 'reject_record_keep': rej_k,
                            'fields_differing_at_reject': differing_fields_at_reject,
                            'records_identical_cycles_1_to_5': same_before, 'cycle_7_actions_clear_keep': c7}
    ok6 = (same_before and rej_c['action'] == 'rejected (safeguard)' and rej_k['action'] == KEEP
           and differing_fields_at_reject == ['action', 'memory_size_after', 'reset', 'reset_reason']
           and np.array_equal(seqs['clear_memory'][5][0], seqs['keep_memory'][5][0])
           and first_diff_cycle == 7 and c7 == ('insufficient memory (m_k=0)', 'accepted'))

    # (7) unknown policy refused
    try:
        AA.AndersonAccelerationState(reject_policy='keep')
        refused = False
    except ValueError:
        refused = True

    checks = {
        'reject_takes_plain_iterate_keeps_mark_appends_pair_no_removal': ok1,
        'next_attempt_uses_retained_memory': ok2,
        'reject_at_full_memory_keeps_5_and_drops_oldest_like_an_accept': ok3,
        'rho_change_still_clears': ok4,
        'failure_cycle_still_clears': ok5,
        'keep_equals_clear_through_first_rejection_then_differs_at_7': ok6,
        'unknown_policy_refused': refused,
        'policies_constant': AA.REJECT_POLICIES == ('clear_memory', 'keep_memory')
        and AA.DEFAULT_REJECT_POLICY == 'clear_memory',
    }
    return {'checks': checks, **_json_roundtrip(out)}


# ======================================================================================================================
#  V3 -- wiring
# ======================================================================================================================
def v3_wiring():
    source = inspect.getsource(srp._run_operational_planning)
    tree = ast.parse(source)
    func = tree.body[0]
    parent_of = {}
    for node in ast.walk(func):
        for child in ast.iter_child_nodes(node):
            parent_of[child] = node
    sites = []
    for node in ast.walk(func):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'AndersonAccelerationState'):
            kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
            guarded = False
            cur = parent_of.get(node)
            while cur is not None and cur is not func:
                if isinstance(cur, ast.If) and 'aa_enabled' in ast.dump(cur.test):
                    guarded = True
                    break
                cur = parent_of.get(cur)
            sites.append({'keywords': kw, 'guarded_by_aa_enabled': guarded, 'lineno': node.lineno})
    expected_kw = "aa_settings.get('reject_policy', admm_anderson_acceleration.DEFAULT_REJECT_POLICY)"
    fresh = admm_parameters.ADMMParameters()
    checks = {
        'exactly_one_construction_site': len(sites) == 1,
        'reject_policy_passed_from_settings_with_default': bool(sites) and sites[0]['keywords'].get(
            'reject_policy') == expected_kw,
        'memory_and_regularization_unchanged': bool(sites) and sites[0]['keywords'].get('memory') == "aa_settings.get('memory', 5)"
        and sites[0]['keywords'].get('regularization') == "aa_settings.get('regularization', 1e-10)",
        'construction_guarded_by_aa_enabled': bool(sites) and sites[0]['guarded_by_aa_enabled'],
        'admm_parameters_default_dict_has_no_reject_policy_key': 'reject_policy' not in fresh.anderson_acceleration,
    }
    return {'checks': checks, 'construction_sites': sites}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'output root already exists (write-once): {OUT}')
    os.makedirs(OUT)
    started = time.time()
    files = ('admm_anderson_acceleration.py', 'shared_resources_planning.py', 'admm_parameters.py',
             os.path.basename(__file__))
    results = {'stage': 'P5.15 Addendum 25 item 2 -- Task 2: AA reject-policy sub-option, zero-solve checks',
               'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 25',
                             'data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json item2_aa_variant'],
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                                          text=True).stdout.strip(),
               'file_sha256_at_run': {f: _sha_bytes(open(os.path.join(REPO, f), 'rb').read()) for f in files}}
    all_pass = True
    for name, fn in (('V0_committed_step37_checks', v0_committed_checks),
                     ('V1_default_path_bitwise_vs_9a965494', v1_default_path_bitwise),
                     ('V2_keep_memory_semantics', v2_keep_memory), ('V3_wiring', v3_wiring)):
        print(f'[S44-AAV] ===== {name} =====', flush=True)
        try:
            out = fn()
            ok = all(out['checks'].values())
        except BaseException as error:  # noqa: BLE001
            out = {'exception': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            ok = False
        out['pass'] = ok
        all_pass = all_pass and ok
        results[name] = out
        print(f'[S44-AAV] {name}: pass={ok} checks={out.get("checks", out.get("exception"))}', flush=True)
    GUARD.uninstall()
    guard_failures = GUARD.verify(expected_solves=0, expected_execs=0)
    results['solve_profile_guard'] = {'counts': dict(GUARD.counts), 'verify_failures': guard_failures}
    results['all_pass'] = all_pass and not guard_failures
    results['wall_clock_s'] = time.time() - started
    out_path = os.path.join(OUT, 'aa_variant_checks.json')
    with open(out_path, 'w') as handle:
        json.dump(results, handle, indent=1, default=str)
    with open(os.path.join(OUT, 'manifest_sha256.json'), 'w') as handle:
        json.dump({os.path.relpath(out_path, REPO): _sha_bytes(open(out_path, 'rb').read())}, handle, indent=2)
    print(f'[S44-AAV] ALL PASS={results["all_pass"]} guard={GUARD.counts} failures={guard_failures}')
    if not results['all_pass']:
        sys.exit(1)


if __name__ == '__main__':
    main()
