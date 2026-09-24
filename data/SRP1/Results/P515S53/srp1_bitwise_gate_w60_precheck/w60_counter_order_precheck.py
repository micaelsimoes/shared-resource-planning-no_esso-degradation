"""W60 zero-solve pre-check of the re-ordered W51 counter in p515_s53_srp1_bitwise_gate.py.

Runs the gate's main() in-process through the WHOLE precondition/presence stage with:
  * a blocking SolveProfileGuard(permitted=()) installed OUTERMOST (any solve raises) and verified at exactly 0;
  * a sys.setprofile/threading.setprofile call counter keyed on the CODE OBJECTS of the two live functions
    (nothing in srp is replaced, so inspect.getsource still reads the live functions);
  * the run-lock function stubbed (no lock file) and, from inside that stub, W10.run_arm stubbed to raise
    before any arm work -- so the run stops exactly where the arm would start.
Checks: (a) the gate's presence dict (recorded, not substituted) is 21/21
(positive control first: the counter registers exactly one call of each when each is called once); w51 items true and all items true;
(b) calls to either function during the precondition stage == 0; (c) at the arm boundary COUNTER is installed
(srp attributes are the wrappers) with 0 calls; (d) after main() the live functions are restored; (e) no output
root created; (f) guards at 0.
"""
import json
import os
import sys
import threading

REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'
sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s53_srp1_bitwise_gate as S53  # noqa: E402  (installs W10.GUARD, bounded, at import)

BLOCK = SolveProfileGuard(permitted=(), label='W60 precheck (zero solves)').install()

srp, W10, W35G = S53.srp, S53.W10, S53.W35G
G = W10.G
LIVE = {'_set_row18_inactive_for_initialisation': srp._set_row18_inactive_for_initialisation,
        '_activate_row18_with_settlement': srp._activate_row18_with_settlement}
CODES = {fn.__code__: name for name, fn in LIVE.items()}
calls = {name: 0 for name in LIVE}
out = {'live_w51_presence_before': None, 'gate_presence_results': [], 'calls_during_preconditions': None,
       'at_arm_boundary': None}


def prof(frame, event, arg):
    if event == 'call':
        name = CODES.get(frame.f_code)
        if name is not None:
            calls[name] += 1


class _StopBeforeArm(Exception):
    pass


real_lock = G._acquire_exclusive_run_lock
real_run_arm = W10.run_arm
real_combined = S53.combined_code_presence


def recording_combined():          # records what the gate's own presence check returns; does not substitute it
    result = real_combined()
    out['gate_presence_results'].append({
        'n_true': sum(bool(v) for v in result.values()), 'n': len(result),
        'w51_true': sum(bool(v) for k, v in result.items() if k.startswith('w51:')),
        'w51_n': sum(1 for k in result if k.startswith('w51:')),
        'false_items': sorted(k for k, v in result.items() if not v),
        'srp_functions_are_live_at_check': {n: getattr(srp, n) is fn for n, fn in LIVE.items()}})
    return result


def stub_run_arm(*args, **kwargs):
    out['at_arm_boundary'] = {
        'counter_installed_srp_attrs_are_wrappers': {n: getattr(srp, n) is not fn for n, fn in LIVE.items()},
        'counter_originals_are_the_live_functions': {n: S53.COUNTER._originals.get(n) is fn for n, fn in LIVE.items()},
        'counter_calls': dict(S53.COUNTER.calls)}
    raise _StopBeforeArm()


def stub_lock(*args, **kwargs):
    sys.setprofile(None)
    threading.setprofile(None)
    out['calls_during_preconditions'] = dict(calls)
    out['srp_functions_live_at_lock'] = {n: getattr(srp, n) is fn for n, fn in LIVE.items()}
    W10.run_arm = stub_run_arm      # after the capture-path checklist has read W10.run_arm's source
    return None


# POSITIVE CONTROL: the profile counter must register a call when one happens (else '0 calls' is vacuous).
import pyomo.environ as pe  # noqa: E402
sys.setprofile(prof)
for fn in LIVE.values():
    fn(pe.ConcreteModel())       # no `row18_alpha` -> each returns at once
sys.setprofile(None)
out['positive_control_calls'] = dict(calls)
for name in calls:
    calls[name] = 0

out['live_w51_presence_before'] = S53.w51_code_presence()
S53.combined_code_presence = recording_combined
G._acquire_exclusive_run_lock = stub_lock
outcome = None
threading.setprofile(prof)
sys.setprofile(prof)
try:
    S53.main()
    outcome = 'main returned (UNEXPECTED: the arm stub should have raised)'
except _StopBeforeArm:
    outcome = 'stopped at the arm boundary (expected)'
except SystemExit as exc:
    outcome = f'SystemExit({exc.code!r}) -- precondition refusal'
finally:
    sys.setprofile(None)
    threading.setprofile(None)
    W10.run_arm = real_run_arm
    S53.combined_code_presence = real_combined
    out['lock_attr_after_main_is_stub_restored_by_gate'] = G._acquire_exclusive_run_lock is stub_lock
    G._acquire_exclusive_run_lock = real_lock
    BLOCK.uninstall()

out['outcome'] = outcome
out['srp_functions_live_after_main'] = {n: getattr(srp, n) is fn for n, fn in LIVE.items()}
out['output_root_exists_after'] = os.path.exists(os.path.join(REPO, S53.OUT_REL))
out['lock_files_exist_after'] = [p for p in ('.p515_g_gate.lock', '.p515_s44_campaign.lock')
                                 if os.path.exists(os.path.join(REPO, p))]
out['blocking_guard'] = {'counts': dict(BLOCK.counts), 'verify_0_failures': BLOCK.verify(0)}
out['w10_bounded_guard_counts'] = dict(W10.GUARD.counts)
live = out['live_w51_presence_before']
checks = {
    'positive_control_counter_detects_one_call_each': out['positive_control_calls'] == {n: 1 for n in LIVE},
    'live_w51_presence_21_of_21': sum(live.values()) == 21 and len(live) == 21,
    'gate_presence_evaluated_at_least_once': bool(out['gate_presence_results']),
    'gate_presence_w51_21_of_21_every_evaluation': all(
        r['w51_true'] == 21 and r['w51_n'] == 21 for r in out['gate_presence_results']),
    'gate_presence_all_items_true_every_evaluation': all(
        r['n_true'] == r['n'] for r in out['gate_presence_results']),
    'gate_presence_read_live_functions': all(
        all(r['srp_functions_are_live_at_check'].values()) for r in out['gate_presence_results']),
    'reached_the_arm_boundary': outcome == 'stopped at the arm boundary (expected)',
    'zero_calls_during_preconditions': out['calls_during_preconditions'] == {n: 0 for n in LIVE},
    'counter_installed_at_arm_boundary': bool(out['at_arm_boundary']) and all(
        out['at_arm_boundary']['counter_installed_srp_attrs_are_wrappers'].values())
        and all(out['at_arm_boundary']['counter_originals_are_the_live_functions'].values()),
    'counter_zero_at_arm_boundary': bool(out['at_arm_boundary'])
        and out['at_arm_boundary']['counter_calls'] == {n: 0 for n in LIVE},
    'live_functions_restored_after_main': all(out['srp_functions_live_after_main'].values()),
    'gate_restored_lock_function': out['lock_attr_after_main_is_stub_restored_by_gate'],
    'no_output_root_created': not out['output_root_exists_after'],
    'no_lock_file': not out['lock_files_exist_after'],
    'blocking_guard_verified_exactly_0': out['blocking_guard']['verify_0_failures'] == [],
    'w10_guard_all_zero': all(v == 0 for v in out['w10_bounded_guard_counts'].values()),
}
out['checks'] = checks
out['all_pass'] = all(checks.values())
print('W60_PRECHECK_RESULT ' + json.dumps(out, indent=1, default=str))
sys.exit(0 if out['all_pass'] else 1)
