"""
P5.15 Addendum 86 W183i order item (d) (with Addendum 85 W183g (d), Addendum 86 rulings 2 and 3), Planner task
W183i (d) -- the STEP-7 CAMPAIGN SPEC v1 FREEZE on the patched harness. ZERO SOLVES, no model build, pickle blocked.

What this script does (one mode per process; repo root; canonical interpreter; attached; both streams captured):

  --freeze-spec   Writes ONCE `data/SRP1/Results/P515S55/frozen_step7_campaign_spec_v1_<sha8>.json` (the name carries
                  the first 8 hex of the file's own sha256; written through `gate_result_io`; no predecessor) and the
                  sha256 manifest `data/SRP1/Results/P515S55/w183i_d_spec_v1/w183i_d_spec_v1_manifest_sha256.json`.
                  BEFORE writing anything it asserts (fail fast; a refusal writes nothing and exits 1):
                    A. the accepted state, re-hashed from the working tree and clean against git: the production head
                       and every production module's sha256 (38), the harness d9a944f2, the gates module 6f4631e1, the
                       Step-7 hooks 9318ade3, the Step-7 launcher 3b990d19, the E1 launcher v3 532cf6e7, p56a_oracle,
                       the settling modules v1-v6, p515_s53_f2_certificate, the 81 SRP1 inputs, the W2 and Phase A
                       tables, the G0 artifact ca195bd3 and record 8cef45d6, the ruled two-cycle gate record 06b9cdc3
                       (GATE_PASS_RULED True), `ParallelExecution: false` (data/SRP1/SRP1.json), the solver sha256;
                    B. the keys of every phase recomputed by the launchers' OWN dry-run code (Step-7: build_lattice /
                       cell_table / key_checks / screen_alias; E1: build_lattice / seed_plan / seed_key_assertions /
                       key_table / g0_check / make_arm_key_of) and asserted equal to the committed dry-run outputs
                       (W183i (b) w183i_b_step7_dryrun.json f4bc728c; W183i (c) gate JSON 201771e6 and the launcher
                       dry run it cites, e1_launcher_v3_dryrun.json 1256f63b);
                    C. the capture checklist of every phase (rule eleven) asserted present in the code: the launchers'
                       own pre-run checklists (Step-7 parent_capture_checklist on each frozen E3/E2 cell spec and on
                       the screen cells; E1 rule_eleven) plus the spec-required quantities of Addendum 83 (campaign
                       spec v1) mapped to their writers by source inspection;
                    D. the real E3 / E2 freezes made by the Step-7 launcher (`--freeze --phase e3|e2`): one stage spec
                       per phase, self-hashing, pinning each cell spec by sha256; every cell spec re-validated by the
                       launcher's own validate_campaign_spec and pre_launch_assertion; no screen / E1 / batch root
                       exists;
                    E. the rho read-back rule validated on the committed W183f arm (the launcher's own function).
                  The declarations (B+ rule, the five confirmed choices, A7, the rho rule, ruling 3, the counts of
                  Addendum 85 ruling 9, the hypotheses, the stops, the convention) are EXTRACTED VERBATIM from the
                  brief / the plan by line range, each range's first and last line checked against a marker, and the
                  source files' sha256 recorded.
  --post-manifest Writes ONCE `.../w183i_d_spec_v1/w183i_d_post_preconditions_manifest_sha256.json`: the sha256 of
                  every artifact of the W183i (d) evidence base, including the --preconditions-only log; zero solves.

  --smoke --out-root D   (pre-commit only) as --freeze-spec, but into a NEW directory D outside the repository, with
                  the script's own committed-and-clean check and the stage-spec requirement relaxed.

Objective convention: no objective value is computed here. Q = gross_operational_cost, contracted interface
settlement excluded (Addendum 83 ruling 2), is the convention the spec declares.
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183i (d) spec v1 freeze (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W183i (d): pickle.load called -- this script never unpickles')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W183i (d): pickle.loads called -- this script never unpickles')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import pyomo.environ as pe  # noqa: E402  (already loaded by the solve guard; the model-build tripwire only)

MODEL_BUILD_COUNTS = {'ConcreteModel.__init__': 0}


def _blocked_concrete_init(self, *_a, **_k):
    MODEL_BUILD_COUNTS['ConcreteModel.__init__'] += 1
    raise RuntimeError('W183i (d): a pyomo ConcreteModel was constructed -- no model build')


pe.ConcreteModel.__init__ = _blocked_concrete_init

import gate_result_io as GRIO  # noqa: E402
import p515_s55_step7_launcher as L7  # noqa: E402  (installs its own parent guard, pickle block, build tripwire)
import p515_s55_e1_launcher as E1  # noqa: E402  (installs its own parent guard, pickle block, tripwires)

H = L7.H
S7 = L7.S7
PB = L7.PB

SCRIPT_NAME = os.path.basename(__file__)
STAGE = 'P5.15 W183i (d) -- Step-7 campaign spec v1 freeze on the patched harness (Addendum 86; Addendum 85 W183g (d))'
P55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
OUT_DIR_REL = os.path.join(P55, 'w183i_d_spec_v1')
SPEC_DOC_FMT = 'frozen_step7_campaign_spec_v1_{}.json'
SPEC_DOC_RX = re.compile(r'^frozen_step7_campaign_spec_v1_([0-9a-f]{8})\.json$')
MANIFEST_FILE = 'w183i_d_spec_v1_manifest_sha256.json'
POST_MANIFEST_FILE = 'w183i_d_post_preconditions_manifest_sha256.json'
SCHEMA = 'p515_s55_step7_campaign_spec_v1'
OBJECTIVE_CONVENTION = ('Q = gross_operational_cost, contracted interface settlement EXCLUDED (Addendum 83 ruling 2); '
                        'salvage reported separately; net = gross - terminal salvage')
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 86 W183i order item (d); rulings 2 and 3',
    'PLANNER_BRIEF_2026-09-13.md Addendum 85 W183g order item (d); rulings 2, 3, 4, 9',
    'PLANNER_BRIEF_2026-09-13.md Addendum 83 campaign spec v1 paragraph; ruling 2 (convention); ruling 3 (vi) (pinning)',
    'STEP7_SIMULATION_PLAN_2026-10-25.md sections 3-5, 9, 11-13',
    'P5_15_W183I_A_ROUTER.md, P5_15_W183I_B_STEP7_LAUNCHER.md, P5_15_W183I_C_E1_V3.md, P5_15_W183G_B_E1_V2.md, '
    'P5_15_W183D_G0.md',
]

# ---- the accepted state (Planner task W183i (d)); each re-hashed from the working tree and checked clean ----
PRODUCTION_HEAD = '6f43479ed8ece7a4a5878942ca38952ce1cae1bc'
N_PRODUCTION_MODULES = 38
N_SRP1_INPUTS = 81
PINNED_FILES = {   # path -> expected sha256 (full where the task gives a prefix that the reports state in full)
    'p515_s44_campaign_harness.py': 'd9a944f220e791b82147f08d0e606126ad454b8b6b2994024f94e106798fb191',
    'p515_g_g1_g4_admm_gates.py': '6f4631e10a2f8a7786390c0f96e45332706779ae9cdf6a87ddd716d9ad9ad346',
    'p515_s55_step7_resettle_hooks.py': '9318ade3e6d0d5fec25d0c276c4d82fba5bb163c7da61ec10ee6f49aaf2fa1d2',
    'p515_s55_step7_launcher.py': '3b990d19bbde8e9fb6bf75d2e5447c55116d4a3d4d21c174e53776dffc561757',
    'p515_s55_e1_launcher.py': '532cf6e7bf03b5cd794843614ae5a4c19ae06485d750305a90a5cd49aea76fa5',
    os.path.join(P55, 'w183_nl_post_w183c_off', 'w183_nl_digest.json.gz'):
        'ca195bd38fe2384493012dcf7127a36a8923d8a5029c2c821b5bb0f47f57ab6a',
    os.path.join(P55, 'w183d_g0', 'g0.json'): '8cef45d684bc7e5dc8b719fd504b8e7705933597a9ba07f44349650046b2911c',
    os.path.join(P55, 'w183i_srp1_bitwise_gate', 'w183i_wrapper_record.json'):
        '06b9cdc3bf7fed9256d2ecf4cd86b978356363c5c0599c58350befc1b2b95350',
}
RECORDED_FILES = ('p56a_oracle.py', 'p515_s53_f2_certificate.py', 'p515_s47_phase_b_record.py', 'gate_result_io.py',
                  'settling_criterion.py', 'settling_criterion_v2.py', 'settling_criterion_v3.py',
                  'settling_criterion_v4.py', 'settling_criterion_v5.py', 'settling_criterion_v6.py',
                  'p515_s55_w183i_c_e1_v3_resume_gate.py', 'PLANNER_BRIEF_2026-09-13.md',
                  'STEP7_SIMULATION_PLAN_2026-10-25.md')
SRP1_CASE_JSON = os.path.join('data', 'SRP1', 'SRP1.json')
SOLVER = '/usr/local/bin/ipopt'
COMMITTED_DRY_RUNS = {
    'w183i_b_step7_dryrun': {'path': os.path.join(P55, 'w183i_b_step7_dryrun', 'w183i_b_step7_dryrun.json'),
                             'sha256': 'f4bc728c439edb7b2268834b29d38dd506aa8d3889c15295d981f1621595078c'},
    'w183i_c_gate': {'path': os.path.join(P55, 'w183i_c_e1_v3_resume', 'w183i_c_e1_v3_resume_gate.json'),
                     'sha256': '201771e6ffe14e034e5cf94d1f2396824989ca8144dffb2abf0791e7ac9c2ff5'},
    'w183i_c_e1_launcher_dry_run': {'path': os.path.join(P55, 'w183i_c_e1_v3_resume', 'launcher_dry_run',
                                                         'e1_launcher_v3_dryrun.json'),
                                    'sha256': '1256f63b04f5aec01e9b062c13a5ae7c89ba778a2268db1ec530d2504f40448b'},
}
NOT_FROZEN_ROOTS = (os.path.join(P55, 'e1_screen_1x'), os.path.join(P55, 'e1_2x'), os.path.join(P55, 'e1_2x_batch4'))
E2_PAIR_ORDER = [('1x', 0.5), ('1x', 2.0), ('2x', 0.5), ('2x', 2.0)]   # Addendum 86 ruling 3 (i)
E2_EXPECTED_ORDER = ['e2_x0_1x_rho0p5', 'e2_unit_1x_rho0p5', 'e2_x0_1x_rho2p0', 'e2_unit_1x_rho2p0',
                     'e2_x0_2x_rho0p5', 'e2_unit_2x_rho0p5', 'e2_x0_2x_rho2p0', 'e2_unit_2x_rho2p0']

# ---- the verbatim declarations: (file, first line, last line, marker in the first line, marker in the last line) ----
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
PLAN = 'STEP7_SIMULATION_PLAN_2026-10-25.md'
TEXT_SEGMENTS = {
    'addendum83_ruling2_objective_convention': (BRIEF, 3785, 3789, '2. **Objective convention.**',
                                                'corrected accordingly from here on.'),
    'addendum85_ruling2_e1_poll_rule_b_plus': (BRIEF, 3964, 3998, '2. **E1 poll rule — B+',
                                               'every other decision path are unchanged.'),
    'addendum85_ruling3_a7': (BRIEF, 4000, 4001, '3. **Barrier stop rule: A7 confirmed**',
                              'every barrier point is reported with its cause.'),
    'addendum85_ruling4_spec_v1_declarations': (BRIEF, 4003, 4025, '4. **Spec v1 declarations',
                                                'for (vi)–(ix).'),
    'addendum85_ruling9_counts_and_limits': (BRIEF, 4053, 4055, '9. **Counts and time',
                                             'every extension needs a ruling.'),
    'addendum85_w184_line': (BRIEF, 4075, 4086, '**W184 (E3) — authorized conditionally',
                             'are active at terminal.'),
    'addendum85_w185_line': (BRIEF, 4088, 4093, '**W185 (E2 stage 1) — authorized conditionally.**',
                             'wait for Addendum 86 after W185.'),
    'addendum86_ruling2_rho_read_back': (BRIEF, 4136, 4143, '2. **Ruling 7 restated.**',
                                         'so the recursion is replicable).'),
    'addendum86_ruling3_declarations': (BRIEF, 4145, 4166, '3. **W185 and campaign declarations.**',
                                        'ruled on if it occurs, never patched mid-run.'),
    'addendum86_w184_w185_reauthorized': (BRIEF, 4189, 4195, '**W184 (E3) and W185 (E2) — re-authorized',
                                          'Addendum 87 after W185.'),
    'plan_s3_hypotheses_h3a_h3b': (PLAN, 87, 91, '**Hypotheses (recorded).** H3a:', 'which `edd7a63` already does.'),
    'plan_s4_hypothesis_h2': (PLAN, 119, 121, '**Hypothesis (recorded).** H2:',
                              'reports the margin as a range over configurations.'),
    'plan_s4_feasibility_checks_per_cell': (PLAN, 106, 108, '**Feasibility checks per cell**',
                                            'Unsuccessful or unaccepted runs are reported, not dropped.'),
    'plan_s5_2_hypotheses_h1a_h1c': (PLAN, 165, 167, '**Hypotheses (recorded; none is required).** H1a:',
                                     'Outcome → claim mapping in §8.'),
    'plan_s5_3_screen_and_h1d': (PLAN, 171, 177, 'Not a search and not a poll of Algorithm 1',
                                 '**Hypothesis H1d:** no neighbour improves on x = 0 beyond the criterion.'),
    'plan_s12_e2_e3_runtime_checks_h3c': (PLAN, 310, 314, '- **E2 (§4) adds a per-cell runtime check:**',
                                          'hours of C_NRF.'),
    'plan_s13_amendments_after_w183g': (PLAN, 327, 339, '## 13. Amendments after W183g',
                                        'exhaustion → stop for review as frozen.'),
}
HYPOTHESES_INDEX = {
    'H1a': 'plan_s5_2_hypotheses_h1a_h1c', 'H1b': 'plan_s5_2_hypotheses_h1a_h1c', 'H1c': 'plan_s5_2_hypotheses_h1a_h1c',
    'H1d': 'plan_s5_3_screen_and_h1d', 'H2': 'plan_s4_hypothesis_h2 (+ addendum85_w185_line; per pair: '
                                             'addendum86_ruling3_declarations (i), plan_s13)',
    'H3a': 'plan_s3_hypotheses_h3a_h3b (+ addendum85_w184_line)', 'H3b': 'plan_s3_hypotheses_h3a_h3b '
                                                                         '(+ addendum85_w184_line)',
    'H3c': 'addendum85_w184_line (+ plan_s12_e2_e3_runtime_checks_h3c)'}


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W183i-d] {msg}', flush=True)


def _abs(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def _rel(path):
    return os.path.relpath(path, REPO)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _sha_or_none(rel):
    return _sha(rel) if os.path.isfile(_abs(rel)) else None


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _tracked_clean(rel):
    tracked = bool(_git(['ls-files', '--', rel]).strip())
    status = _git(['status', '--porcelain', '--', rel]).strip()
    return tracked and not status


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str)


def guards_report():
    """Every SolveProfileGuard reachable from the repo modules (this script's, both launchers', the PB module's ...)
    verified at 0; the pickle and model-build counters of this script and of both launchers; the E1 launcher's
    tripwires on the committed launchers' functions."""
    seen, guards = set(), {}
    for name, mod in sorted(sys.modules.items()):
        if not getattr(mod, '__file__', None) or not str(mod.__file__).startswith(REPO + os.sep):
            continue
        for attr, value in sorted(vars(mod).items()):
            if isinstance(value, SolveProfileGuard) and id(value) not in seen:
                seen.add(id(value))
                guards[f'{name}.{attr}'] = {'counts': dict(value.counts), 'verify_0_failures': value.verify(0)}
    if id(GUARD) not in seen:
        guards['__main__.GUARD'] = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    pick = {'w183i_d': dict(PICKLE_COUNTS), 'step7_launcher': dict(L7.PICKLE_COUNTS), 'e1_launcher': dict(E1.PICKLE_COUNTS)}
    build = {'w183i_d': dict(MODEL_BUILD_COUNTS), 'step7_launcher': dict(L7.MODEL_BUILD_COUNTS),
             'e1_launcher': dict(E1.MODEL_BUILD_COUNTS)}
    blockers = {_blocked_load, L7._blocked_load, E1._blocked_load}
    blockers_s = {_blocked_loads, L7._blocked_loads, E1._blocked_loads}
    pickle_blocked = pickle.load in blockers and pickle.loads in blockers_s
    trip = dict(sorted(E1.TRIPWIRE_COUNTS.items()))
    all_zero = (all(not g['verify_0_failures'] for g in guards.values())
                and not any(v for d in pick.values() for v in d.values())
                and not any(v for d in build.values() for v in d.values())
                and not any(trip.values()) and pickle_blocked)
    return {'solve_profile_guards': guards, 'n_guards_verified': len(guards), 'pickle': pick,
            'pickle_blocked_at_end': pickle_blocked, 'model_build_tripwires': build,
            'e1_launcher_committed_function_tripwires': trip, 'all_zero': all_zero}


class Refusal(Exception):
    pass


def _need(cond, what, refusals):
    if not cond:
        refusals.append(what)
    return bool(cond)


# ======================================================================================================================
#  A. the accepted state
# ======================================================================================================================
def accepted_state(refusals, smoke=False):
    out = {}
    prod = L7.production_now()
    out['production'] = {'definition': prod['definition'], 'head': prod['head'], 'n_modules': len(prod['modules_sha256']),
                         'modules_sha256': prod['modules_sha256'],
                         'modules_tracked_and_clean': {m: _tracked_clean(m) for m in prod['modules_sha256']}}
    _need(prod['head'] == PRODUCTION_HEAD, f"production head {prod['head']} != {PRODUCTION_HEAD}", refusals)
    _need(len(prod['modules_sha256']) == N_PRODUCTION_MODULES,
          f"production modules {len(prod['modules_sha256'])} != {N_PRODUCTION_MODULES}", refusals)
    _need(all(out['production']['modules_tracked_and_clean'].values()), 'a production module is not committed/clean',
          refusals)
    files = {}
    for rel, expected in PINNED_FILES.items():
        got = _sha_or_none(rel)
        files[rel] = {'sha256': got, 'expected': expected, 'equals_expected': got == expected,
                      'tracked_and_clean': _tracked_clean(rel)}
        _need(got == expected, f'{rel}: sha256 {got} != {expected}', refusals)
        _need(files[rel]['tracked_and_clean'], f'{rel}: not committed and clean', refusals)
    for rel in RECORDED_FILES:
        files[rel] = {'sha256': _sha_or_none(rel), 'tracked_and_clean': _tracked_clean(rel)}
        _need(files[rel]['sha256'] is not None and files[rel]['tracked_and_clean'], f'{rel}: missing or not clean',
              refusals)
    out['files'] = files
    # both launchers' run files (each launcher's own list)
    out['step7_launcher_run_files_sha256'] = L7.code_sha256()
    out['e1_launcher_run_files_sha256'] = {f: _sha_or_none(f) for f in E1.RUN_FILES}
    for grp in ('step7_launcher_run_files_sha256', 'e1_launcher_run_files_sha256'):
        for f, s in out[grp].items():
            _need(s is not None and _tracked_clean(f), f'{grp} {f}: missing or not clean', refusals)
    # the 81 SRP1 inputs; the named inputs
    inputs = {f: _sha(f) for f in L7.tracked_inputs()}
    out['srp1_inputs'] = {'definition': 'every tracked file under data/SRP1 outside Results/ and Diagrams/',
                          'n': len(inputs), 'sha256': inputs,
                          'all_tracked_and_clean': not _git(['status', '--porcelain', '--'] + sorted(inputs)).strip()}
    _need(len(inputs) == N_SRP1_INPUTS, f'SRP1 inputs {len(inputs)} != {N_SRP1_INPUTS}', refusals)
    _need(out['srp1_inputs']['all_tracked_and_clean'], 'an SRP1 input is not clean', refusals)
    named = {}
    for k, v in L7.NAMED_INPUTS.items():
        named[k] = ({'path': v, 'sha256': inputs.get(v)} if isinstance(v, str)
                    else [{'path': p, 'sha256': inputs.get(p)} for p in v])
    out['named_inputs'] = named
    _need(named['ess_params_file']['sha256'] == PB.ESS_PARAMS_FILE['sha256'], 'ESS params file != PB pin', refusals)
    _need(named['cost_file']['sha256'] == PB.COST_FILE['sha256'], 'cost file != PB pin', refusals)
    # the W2 and Phase A tables
    tables = {}
    for name, pin in L7.SEARCH_TABLES.items():
        got = _sha_or_none(pin['path'])
        tables[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256': got,
                        'equals_pin': got == pin['sha256'], 'tracked_and_clean': _tracked_clean(pin['path'])}
        _need(tables[name]['equals_pin'] and tables[name]['tracked_and_clean'], f'table {name} not as pinned', refusals)
    out['search_tables'] = tables
    # the ruled two-cycle gate record
    wr = _load(os.path.join(P55, 'w183i_srp1_bitwise_gate', 'w183i_wrapper_record.json'))
    out['two_cycle_gate_ruled_record'] = {'path': os.path.join(P55, 'w183i_srp1_bitwise_gate',
                                                               'w183i_wrapper_record.json'),
                                          'GATE_PASS_as_frozen': wr.get('GATE_PASS'),
                                          'GATE_PASS_RULED': wr.get('GATE_PASS_RULED'), 'schema': wr.get('schema')}
    _need(wr.get('GATE_PASS_RULED') is True, 'two-cycle gate record: GATE_PASS_RULED is not True', refusals)
    # G0 record
    g0 = _load(os.path.join(P55, 'w183d_g0', 'g0.json'))
    out['g0_record_G0_PASS'] = g0.get('G0_PASS')
    _need(g0.get('G0_PASS') is True, 'g0.json: G0_PASS is not True', refusals)
    # ParallelExecution
    case = _load(SRP1_CASE_JSON)
    out['parallel_execution'] = {'path': SRP1_CASE_JSON, 'sha256': _sha(SRP1_CASE_JSON),
                                 'ParallelExecution': case.get('ParallelExecution'),
                                 'scope_note': ('Addendum 85 ruling 1: the solve-bearing two-cycle gate covers the '
                                                'sequential DSO builder, the only one this configuration runs; a run '
                                                'that turned ParallelExecution on would need its own gate')}
    _need(case.get('ParallelExecution') is False, 'SRP1.json ParallelExecution is not false', refusals)
    # solver
    solver = L7.solver_check()
    out['solver'] = {'path': SOLVER, 'resolved': solver['resolved'], 'sha256': solver['sha256'], 'ok': solver['ok']}
    _need(solver['ok'], f'solver: {solver}', refusals)
    # the committed dry-run outputs
    dr = {}
    for name, pin in COMMITTED_DRY_RUNS.items():
        got = _sha_or_none(pin['path'])
        dr[name] = {'path': pin['path'], 'sha256': got, 'equals_pin': got == pin['sha256'],
                    'tracked_and_clean': _tracked_clean(pin['path'])}
        _need(dr[name]['equals_pin'] and dr[name]['tracked_and_clean'], f'dry-run output {name} not as pinned', refusals)
    out['committed_dry_run_outputs'] = dr
    # the launchers' complete pin sets as each launcher computes them (recorded; the imported-module part is THIS
    # process's, which imports both launchers -- each phase freeze records its own)
    out['step7_launcher_pins_now'] = L7.pins_now()
    out['e1_launcher_pins_now_refreshed_at_spec_v1'] = E1.pins_now()
    for nm in ('step7_launcher_pins_now', 'e1_launcher_pins_now_refreshed_at_spec_v1'):
        p = out[nm]
        unc = [ln for ln in p['uncommitted_changes'] if not (smoke and ln.endswith(SCRIPT_NAME))]
        unt = [f for f in p['untracked'] if not (smoke and f == SCRIPT_NAME)]
        _need(not unc and not unt,
              f"{nm}: uncommitted {p['uncommitted_changes']} untracked {p['untracked']}", refusals)
    out['step7_pins_recorded_check'] = L7.pins_recorded_check(out['step7_launcher_pins_now'])
    _need(all(out['step7_pins_recorded_check'].values()), 'step-7 pins_recorded_check fails', refusals)
    out['e1_pins_recorded_check'] = E1.pins_recorded_check(out['e1_launcher_pins_now_refreshed_at_spec_v1'])
    _need(all(out['e1_pins_recorded_check'].values()), f"E1 pins_recorded_check fails: "
          f"{[k for k, v in out['e1_pins_recorded_check'].items() if not v]}", refusals)
    out['git_head'] = _git(['rev-parse', 'HEAD']).strip()
    return out


# ======================================================================================================================
#  B. keys of every phase (the launchers' own dry-run code) vs the committed dry runs
# ======================================================================================================================
KEY_ROW_FIELDS = ('cell', 'model', 'candidate_key', 'eval_key', 'base_key', 'key_without', 'checks', 'nrf_option',
                  'rho0_scale', 'flex_price_multiplier')


def step7_keys(refusals):
    dr = _load(COMMITTED_DRY_RUNS['w183i_b_step7_dryrun']['path'])
    lattice, lev, lfail = L7.build_lattice()
    _need(not lfail, f'step-7 lattice: {lfail}', refusals)
    if lattice is None:
        raise Refusal('the Step-7 lattice could not be built')
    out = {'source_dry_run': COMMITTED_DRY_RUNS['w183i_b_step7_dryrun'], 'phases': {}}
    holders, n_specs, excluded = L7.committed_eval_keys()
    for phase in ('e3', 'e2', 'screen'):
        rows, cmp = [], []
        committed = dr['key_tables'][phase]
        for i, cdef in enumerate(L7.cell_table(phase, lattice)):
            kc = L7.key_checks(cdef)
            row = {'cell': cdef['cell'], 'model': cdef['model'], 'candidate_key': cdef['candidate_key'],
                   'eval_key': kc['eval_key'], 'base_key': kc['base_key'],
                   'key_without': {k[len('key_without_'):]: v for k, v in kc.items() if k.startswith('key_without_')},
                   'checks': kc['checks'], 'nrf_option': cdef['nrf_option'], 'rho0_scale': cdef['rho0_scale'],
                   'flex_price_multiplier': cdef['flex_price_multiplier'],
                   'canonical': cdef['canonical'], 'investment_year': cdef['investment_year'],
                   'committed_holders_outside_the_stage_roots': holders.get(kc['eval_key'], [])}
            c = committed[i] if i < len(committed) else {}
            diff = [f for f in KEY_ROW_FIELDS if _canon(row.get(f)) != _canon(c.get(f))]
            cmp.append({'cell': cdef['cell'], 'equal_to_committed_dry_run': not diff, 'differing_fields': diff})
            _need(not diff, f'{phase} {cdef["cell"]}: recomputed key row differs from the committed dry run: {diff}',
                  refusals)
            _need(all(v is True for v in kc['checks'].values()), f'{phase} {cdef["cell"]}: key checks fail', refusals)
            rows.append(row)
        _need(len(rows) == len(committed), f'{phase}: {len(rows)} cells != committed {len(committed)}', refusals)
        out['phases'][phase] = {'n_cells': len(rows), 'n_distinct_eval_keys': len({r['eval_key'] for r in rows}),
                                'cells': rows, 'comparison_with_committed_dry_run': cmp}
    counts = {p: out['phases'][p]['n_distinct_eval_keys'] for p in out['phases']}
    _need(counts == {'e3': 1, 'e2': 8, 'screen': 15}, f'step-7 key counts {counts} != 1/8/15', refusals)
    for p in ('e3', 'e2'):
        for r in out['phases'][p]['cells']:
            _need(not r['committed_holders_outside_the_stage_roots'],
                  f"{p} {r['cell']}: committed holder {r['committed_holders_outside_the_stage_roots']}", refusals)
    _need([r['cell'] for r in out['phases']['e2']['cells']] == E2_EXPECTED_ORDER == list(S7.E2_CELLS),
          'E2 order is not the pair order of Addendum 86 ruling 3 (i)', refusals)
    g0_ok, alias = L7.screen_alias(lattice, require_current_production=True)
    out['screen_alias'] = {'alias_label': alias['alias_label'], 'alias_eval_key': alias['alias_eval_key'],
                           'reuse_of_label': alias['reuse_of_label'], 'reuse_of': alias['reuse_of'],
                           'g0_accepted_with_current_production': g0_ok, 'g0_checks': alias['g0'].get('checks'),
                           'committed_dry_run_alias_eval_key': dr['screen_alias']['alias_eval_key'],
                           'equal_to_committed_dry_run': (alias['alias_eval_key'] == dr['screen_alias']['alias_eval_key']
                                                          and alias['reuse_of'] == dr['screen_alias']['reuse_of'])}
    _need(g0_ok, 'screen G0 alias not accepted', refusals)
    _need(out['screen_alias']['equal_to_committed_dry_run'], 'screen alias differs from the committed dry run', refusals)
    out['committed_spec_scan'] = {'n_specs_scanned': n_specs, 'n_excluded_specs': len(excluded),
                                  'excluded_roots': list(L7.STAGE_ROOTS),
                                  'scope': 'git-tracked data/*campaign_spec_*.json outside the four Step-7 stage roots'}
    out['cross'] = {'root': L7.PHASE_ROOTS['cross'], 'n_cells': len(L7.cell_table('cross', lattice)),
                    'note': 'declared only; entries added after W186 (Addendum 86 ruling 1); --freeze / --run refuse'}
    _need(out['cross']['n_cells'] == 0, 'cross has cells', refusals)
    return out, lattice, lev


def e1_keys(refusals):
    gate = _load(COMMITTED_DRY_RUNS['w183i_c_gate']['path'])
    drj = _load(COMMITTED_DRY_RUNS['w183i_c_e1_launcher_dry_run']['path'])
    _need(gate['item5'].get('launcher_dry_run_sha256') == COMMITTED_DRY_RUNS['w183i_c_e1_launcher_dry_run']['sha256'],
          'the W183i (c) gate does not cite the pinned launcher dry run', refusals)
    lattice, lev, lfail = E1.build_lattice()
    _need(not lfail, f'E1 lattice: {lfail}', refusals)
    if lattice is None:
        raise Refusal('the E1 lattice could not be built')
    plan = E1.seed_plan(lattice)
    s_checks, summary = E1.seed_key_assertions(plan)
    _need(all(v is True for v in s_checks.values()), f'E1 seed key assertions fail: {s_checks}', refusals)
    table = E1.key_table(plan)
    committed_table = {r['label']: r for r in drj['key_table']}
    cmp = []
    for r in table:
        c = committed_table.get(r['label']) or {}
        diff = [f for f in ('seed', 'model', 'lattice_label', 'candidate_key', 'eval_key', 'disposition',
                            'alias_of_eval_key') if _canon(r.get(f)) != _canon(c.get(f))]
        cmp.append({'label': r['label'], 'equal_to_committed_dry_run': not diff, 'differing_fields': diff})
        _need(not diff, f"E1 seed {r['label']}: differs from the committed launcher dry run: {diff}", refusals)
    _need(len(table) == len(drj['key_table']) == 8, 'E1 key table size', refusals)
    gsum = gate['item5']['seed_key_summary']
    gate_eq = {'keys_to_evaluate_equal': summary['keys_to_evaluate'] == gsum['keys_to_evaluate'],
               'n_distinct_7': summary['n_distinct_keys_to_evaluate'] == gsum['n_distinct_keys_to_evaluate'] == 7,
               'one_alias': summary['n_aliases'] == gsum['n_aliases'] == 1,
               'alias_eval_key_equal': summary['alias']['eval_key'] == gsum['alias']['eval_key'],
               'alias_target_equal': summary['alias']['alias_of']['eval_key'] == gsum['alias']['alias_of']['eval_key']}
    _need(all(gate_eq.values()), f'E1 seed keys differ from the W183i (c) gate JSON: {gate_eq}', refusals)
    x0_keys = {m: next(r['eval_key'] for r in plan if r['seed'] == 'x0' and r['model'] == m) for m in E1.MODELS}
    g0_ok, g0_info = E1.g0_check(L7.G0_ARTIFACT_REL, L7.G0_ARTIFACT_SHA256, x0_keys, require_current_production=True)
    _need(g0_ok, f"E1 G0 alias (2x) not accepted: {g0_info.get('checks')}", refusals)
    by_label = {r['label']: r for r in table}
    batch = [{'label': lb, 'eval_key': by_label[lb]['eval_key'], 'candidate_key': by_label[lb]['candidate_key'],
              'model': by_label[lb]['model']} for lb in E1.BATCH_SEEDS]
    _need([b['label'] for b in batch] == ['unit_A_2x', 'unit_N_2x', 'n7_05_2_A_2x', 'n7_05_2_N_2x'],
          'batch seeds are not the four declared', refusals)
    _need(all(b['eval_key'] in summary['keys_to_evaluate'] for b in batch), 'a batch key is not a seed key', refusals)
    dom = lattice.domain()
    dkeys = {arm: [E1.make_arm_key_of(lattice, arm)(z) for z in dom] for arm in E1.MODELS}
    dom_out = {'n_points': len(dom),
               **{f'n_distinct_{a}': len(set(dkeys[a])) for a in E1.MODELS},
               'A_N_disjoint': not (set(dkeys['A']) & set(dkeys['N'])),
               **{f'sha256_of_sorted_keys_{a}': hashlib.sha256('\n'.join(sorted(dkeys[a])).encode()).hexdigest()
                  for a in E1.MODELS},
               'sha256_definition': "sha256 of '\\n'.join(sorted(eval keys)) per arm"}
    cd = drj['domain_keys']
    dom_out['equal_to_committed_dry_run'] = all(dom_out[k] == cd[k] for k in ('n_points', 'n_distinct_A',
                                                                             'n_distinct_N', 'A_N_disjoint'))
    _need(dom_out['equal_to_committed_dry_run'] and dom_out['n_points'] == 1705, 'E1 domain keys differ', refusals)
    holders, n_specs = E1.committed_eval_key_holders([r['eval_key'] for r in table])
    roots = {
        E1.E1_ROOT_REL: {'phases': {ph: f'campaign_{E1.campaign_id(E1.E1_ROOT_REL, ph)}' for ph in E1.PHASES},
                         'cross': L7.PHASE_ROOTS['cross'], 'declared': E1.DECLARED_ROOTS[E1.E1_ROOT_REL]},
        E1.E1_BATCH_ROOT_REL: {'phases': {'seeds': f"campaign_{E1.campaign_id(E1.E1_BATCH_ROOT_REL, 'seeds')}"},
                               'declared': E1.DECLARED_ROOTS[E1.E1_BATCH_ROOT_REL]}}
    return {'source_gate_json': COMMITTED_DRY_RUNS['w183i_c_gate'],
            'source_launcher_dry_run': COMMITTED_DRY_RUNS['w183i_c_e1_launcher_dry_run'],
            'seed_key_assertions': s_checks, 'seed_key_summary': summary, 'seed_key_table': table,
            'comparison_with_committed_launcher_dry_run': cmp, 'comparison_with_committed_gate_json': gate_eq,
            'g0_alias_2x': {'accepted_with_current_production': g0_ok, 'checks': g0_info.get('checks'),
                            'artifact_sha256': g0_info.get('sha256_on_disk')},
            'batch4': {'root': E1.E1_BATCH_ROOT_REL, 'concurrency': E1.BATCH_CONCURRENCY, 'seeds': batch,
                       'note': ('the same eval keys as the serial seeds (concurrency and campaign id do not enter '
                                'the key); root-scoped cache, never seeds an arm')},
            'search_domain_keys': dom_out,
            'committed_holders_of_seed_keys': {k: v for k, v in holders.items()}, 'n_specs_scanned': n_specs,
            'roots': roots}, lattice, lev


# ======================================================================================================================
#  C. the capture checklists (rule eleven), fail fast
# ======================================================================================================================
def spec_required_capture_map(refusals):
    """Addendum 83's campaign-spec-v1 capture list, each item mapped to its writer by source inspection."""
    import inspect
    G = __import__('p515_g_g1_g4_admm_gates')
    build = inspect.getsource(H.build_evaluation_record)
    hsrc = inspect.getsource(H)
    run_arm = inspect.getsource(G.run_admm_arm)
    boyd = inspect.getsource(G.write_boyd_terminal_s35ref)
    sett = inspect.getsource(G.write_interface_settlement_detail_s31c)
    levels = inspect.getsource(G.write_component_levels_terminal)
    blocks = inspect.getsource(G._s31_block_components)
    post = inspect.getsource(H.run_post_certification)
    items = {
        'Q_gross_operational_cost': (('all',), "'certified_cost': report.get('gross_operational_cost')" in build,
                                     'evaluation_record.certified_cost (= gross_operational_cost)'),
        'status': (('all',), "'status': status" in build, 'evaluation_record.status (+ E3/E2: resettle_decision.json)'),
        'end_cycle_and_certification_cycle': (('all',), "'cycles_run': len(rows)" in build
                                              and "'certification_cycle':" in build,
                                              'evaluation_record.cycles_run / certification_cycle'),
        'k_star_band_gap_slack_E3_E2': (('e3', 'e2'), bool(S7.DECISION_FILE) and S7.DECISION_FILE == 'resettle_decision.json'
                                        and "open(os.path.join(self.eval_dir, DECISION_FILE), 'x')"
                                        in inspect.getsource(__import__('p515_s53_w118_resettle_hooks')),
                                        'resettle_decision.json (k_star / k_cap, band, band_width, gap_refusals; the '
                                        'v6 state, written once by the W118 state the v6 state inherits)'),
        'harness_bar': (('all',), "'bar': bar" in build, 'evaluation_record.bar'),
        'stopping_ratio_rule_ten': (('all',), "'terminal_step_over_threshold':" in build,
                                    'evaluation_record.rule_ten.terminal_step_over_threshold'),
        'candidate_hash': (('all',), "'candidate_key': entry['key']" in build, 'evaluation_record.candidate_key'),
        'spec_hash': (('all',), "'campaign_spec_sha256': spec_sha256" in build, 'evaluation_record.campaign_spec_sha256'),
        'code_and_harness_hashes': (('all',), "'harness':" in inspect.getsource(H.freeze_campaign_spec)
                                    and callable(L7.pins_recorded_check) and callable(E1.pins_recorded_check),
                                    'campaign spec harness sha256 + extra.pins (both launchers pins_recorded_check)'),
        'interface_residual_ratios': (('all',), "'boyd_terminal_ratio_max_per_channel':" in build,
                                      'evaluation_record.boyd_terminal_ratio_max_per_channel; boyd_terminal.json'),
        'interface_settlement_and_gap': (('all',), 'G.write_boyd_terminal_s35ref(' in hsrc
                                         and 'write_interface_settlement_detail_s31c(' in boyd,
                                         'interface_settlement_detail_s31c.json (written by write_boyd_terminal_s35ref)'),
        'closure_slack_terminal_per_block': (('all',), "'shared_ess_day_balance_slack'" in blocks
                                             and '_s31_block_components(' in levels
                                             and 'write_component_levels_terminal(' in sett,
                                             'component_levels_terminal.json blocks[*] (shared/local ESS day-balance '
                                             'slack levels, _s31_block_components)'),
        'esso_terminal_models_persisted': (('all',), "pickle.dump(models['esso'], handle)" in run_arm,
                                           'esso_models_<label>.pkl (run_admm_arm): the terminal ESSO models -- the '
                                           'source of the converter-circle residual, max |P^E| / S^Av, SOC within '
                                           'limits and any ESSO slack (a guarded read; no recorded scalar)'),
        'esso_per_cycle_capture': (('all',), "report['esso_capture'] = N.capture_esso(models['esso'], sed)" in run_arm,
                                   'g_<label>.json esso_capture + esso_capture/<label>/node*_cycle*.jsonl'),
        'certified_models_persisted_E3_E2': (('e3', 'e2'), 'persist_fn(' in post
                                             and "'certified_models.pkl'" in hsrc,
                                             'certified_models.pkl (post_certification persist_certified_models ON): '
                                             'every terminal TSO/DSO model'),
        'binding_nrf_hours_E3': (('e3',), callable(getattr(H, 'capture_nrf_terminal', None))
                                 and "holder['nrf_terminal'] = capture_nrf_terminal(planning, models)" in hsrc,
                                 'evaluation_record nrf_terminal (H.capture_nrf_terminal on the terminal models)'),
        'e_av_and_terminal_soh_per_node_E1': (('seeds', 'search_N', 'search_A', 'batch4', 'screen'),
                                              "'storage_per_node': _storage_per_node(" in build
                                              and "'published_available_capacity_terminal'"
                                              in inspect.getsource(H._storage_per_node)
                                              and "'terminal_soh_per_active_cohort_year'"
                                              in inspect.getsource(H._storage_per_node),
                                              'evaluation_record.storage_per_node (E^Av terminal, terminal SoH)'),
        'rho_trajectory_E2': (('e2',), all(f in H.PER_CYCLE_TRAJECTORY_FIELDS for g in L7.RHO_CHANNELS
                                           for f in (f'rho_{g}_after', f'rho_{g}_action')),
                              'per_cycle_record.jsonl rho_{v,pf,ess}_after / _action'),
        'per_cycle_trajectory_all': (('all',), "'per_cycle_record.jsonl'" in inspect.getsource(H._child_real),
                                     'per_cycle_record.jsonl (record per-cycle state by default)'),
    }
    out = {}
    for name, (phases, ok, where) in items.items():
        out[name] = {'phases': list(phases), 'capture_path_present': bool(ok), 'where': where}
        _need(ok, f'spec-required capture path missing: {name} ({where})', refusals)
    return out


def capture_checklists(refusals, s7_lattice, e1_lattice, e1_lev, frozen):
    out = {'step7': {}, 'e1': None}
    for phase in ('e3', 'e2'):
        out['step7'][phase] = {}
        for cdef in L7.cell_table(phase, s7_lattice):
            spec = (frozen.get(phase) or {}).get('specs', {}).get(cdef['cell'])
            if spec is None:   # smoke only: the frozen spec is not there yet -> the would-be entry checks are skipped
                out['step7'][phase][cdef['cell']] = {'ok': None, 'note': 'no frozen spec (smoke)'}
                continue
            ok, cap = L7.parent_capture_checklist(cdef, spec)
            n = {k: (len(v.get('checks', v)) if isinstance(v, dict) and 'error' not in v else None)
                 for k, v in cap.items()}
            out['step7'][phase][cdef['cell']] = {'ok': ok, 'n_items_by_part': n, 'checklist': cap}
            _need(ok, f'{phase} {cdef["cell"]}: parent capture checklist fails', refusals)
        try:
            pc = H.assert_post_certification_capture_paths()
            out['step7'][phase]['__post_certification_capture_paths'] = {'ok': True, 'checks': pc}
        except Exception as error:  # noqa: BLE001
            out['step7'][phase]['__post_certification_capture_paths'] = {'ok': False, 'error': f'{type(error).__name__}: '
                                                                                               f'{error}'}
            refusals.append(f'{phase}: post-certification capture paths: {error}')
    out['step7']['screen'] = {}
    for cdef in L7.cell_table('screen', s7_lattice)[:2]:   # the checklist is phase-level: x = 0 (A) and one N cell
        ok, cap = L7.parent_capture_checklist(cdef, None)
        out['step7']['screen'][cdef['cell']] = {'ok': ok, 'checklist': cap}
        _need(ok, f'screen {cdef["cell"]}: parent capture checklist fails', refusals)
    try:
        r11 = E1.rule_eleven(e1_lattice, e1_lev['sigma_Q']['sigma_Q_eur'])
        out['e1'] = {'ok': True, 'phases': ['seeds', 'search_N', 'search_A', 'batch4 (seeds)'],
                     'n_checks': len(r11['checks']), 'checks': r11['checks'],
                     'n_harness_record_capture_items': len(r11['harness_record_capture_checklist']),
                     'point_fields': r11['point_fields']}
    except AssertionError as error:
        out['e1'] = {'ok': False, 'error': str(error)}
        refusals.append(f'E1 rule eleven: {error}')
    return out


# ======================================================================================================================
#  D. the real E3 / E2 freezes
# ======================================================================================================================
def frozen_roots(refusals, s7_lattice, smoke):
    out = {}
    for phase in ('e3', 'e2'):
        base_rel = L7.PHASE_ROOTS[phase]
        base_abs = _abs(base_rel)
        st = L7.stage_spec(base_abs, phase, require_git=False)
        if st is None:
            if smoke:
                out[phase] = {'stage_spec': None, 'note': 'smoke: no stage spec yet', 'specs': {}}
                continue
            refusals.append(f'{phase}: the stage spec is missing / not unique / not self-hashing')
            out[phase] = {'specs': {}}
            continue
        name, sha, content = st
        cells, specs, files = {}, {}, []
        for directory, _dirs, fnames in os.walk(base_abs):
            for f in sorted(fnames):
                files.append(_rel(os.path.join(directory, f)))
        expected_files = {os.path.join(base_rel, name)}
        _need(content.get('dry_run_scratch_freeze') is False, f'{phase}: stage spec is a scratch freeze', refusals)
        _need(content.get('persist_certified_models') is True, f'{phase}: persist_certified_models is not True', refusals)
        _need(content.get('pins', {}).get('harness_sha256') == PINNED_FILES['p515_s44_campaign_harness.py'],
              f'{phase}: stage spec harness pin', refusals)
        _need(content.get('pins', {}).get('hooks_sha256') == PINNED_FILES['p515_s55_step7_resettle_hooks.py'],
              f'{phase}: stage spec hooks pin', refusals)
        _need(content.get('cell_order') == [c['cell'] for c in L7.cell_table(phase, s7_lattice)],
              f'{phase}: stage spec cell order', refusals)
        for cdef in L7.cell_table(phase, s7_lattice):
            c = (content.get('cells') or {}).get(cdef['cell']) or {}
            root = L7.cell_root(base_abs, cdef['cell'])
            spec_path = os.path.join(base_abs, c.get('campaign_spec', '__missing__'))
            exists = os.path.isfile(spec_path)
            spec_sha = H.sha256_file(spec_path) if exists else None
            spec = json.load(open(spec_path)) if exists else None
            expected_files.add(_rel(spec_path))
            checks, _vs6 = L7.validate_campaign_spec(cdef, spec) if spec else ({'spec_missing': False}, None)
            pre = L7.pre_launch_assertion(cdef, base_abs, spec=spec) if spec else {'holds': False, 'parts': {}}
            root_only_spec = os.path.isdir(root) and sorted(os.listdir(root)) == [os.path.basename(spec_path)]
            cells[cdef['cell']] = {
                'campaign_root': _rel(root), 'campaign_spec': _rel(spec_path), 'campaign_spec_sha256': spec_sha,
                'stage_spec_pins_sha256': c.get('sha256'), 'stage_spec_pins_this_spec': c.get('sha256') == spec_sha,
                'name_carries_its_sha8': exists and os.path.basename(spec_path).endswith(f'_{spec_sha[:8]}.json'),
                'eval_key': (spec or {}).get('candidates', [{}])[0].get('eval_key'),
                'campaign_id': (spec or {}).get('campaign_id'), 'cap': (spec or {}).get('cap'),
                'post_certification': (spec or {}).get('candidates', [{}])[0].get('post_certification'),
                'validate_campaign_spec_failing': [k for k, v in checks.items() if not v],
                'n_validate_campaign_spec_checks': len(checks),
                'pre_launch_assertion_holds': pre['holds'],
                'pre_launch_assertion_parts': pre['parts'],
                'campaign_root_holds_only_its_spec': root_only_spec,
                'launch_command': c.get('launch_command'),
                'preconditions_only_command': c.get('preconditions_only_command')}
            cc = cells[cdef['cell']]
            _need(exists and cc['stage_spec_pins_this_spec'] and cc['name_carries_its_sha8'],
                  f"{phase} {cdef['cell']}: cell spec missing / not pinned by the stage spec", refusals)
            _need(not cc['validate_campaign_spec_failing'], f"{phase} {cdef['cell']}: validate_campaign_spec fails "
                  f"{cc['validate_campaign_spec_failing']}", refusals)
            _need(pre['holds'], f"{phase} {cdef['cell']}: pre-launch assertion fails", refusals)
            _need(root_only_spec, f"{phase} {cdef['cell']}: the campaign root does not hold only its spec", refusals)
            _need(cc['post_certification'] == L7.PERSIST, f"{phase} {cdef['cell']}: persist not on", refusals)
            specs[cdef['cell']] = spec
        logs = [f for f in files if f not in expected_files]
        _need(all(L7.LOG_RX.match(os.path.basename(f)) for f in logs), f'{phase}: unexpected files {logs}', refusals)
        out[phase] = {'phase_root': base_rel, 'stage_spec': os.path.join(base_rel, name), 'stage_spec_sha256': sha,
                      'stage_spec_name_carries_its_sha8': name.endswith(f'_{sha[:8]}.json'),
                      'stage_spec_git_head_at_freeze': content.get('git_head'),
                      'stage_spec_frozen_utc': content.get('frozen_utc'), 'cells': cells,
                      'report_command': content.get('report_command'), 'stops': content.get('stops'),
                      'files_in_phase_root': files, 'logs_in_phase_root': logs, 'specs': specs}
    for rel in NOT_FROZEN_ROOTS:
        _need(not os.path.exists(_abs(rel)), f'{rel} exists: the screen / E1 / batch roots wait for Addendum 87',
              refusals)
    out['not_frozen_roots'] = {rel: {'exists': os.path.exists(_abs(rel)),
                                     'reason': 'waits for Addendum 87 (W186 / W187); keys recorded in this spec only'}
                               for rel in NOT_FROZEN_ROOTS}
    return out


# ======================================================================================================================
#  E. verbatim text
# ======================================================================================================================
def text_segments(refusals):
    out, lines_of = {}, {}
    for name, (path, first, last, m_first, m_last) in TEXT_SEGMENTS.items():
        if path not in lines_of:
            with open(_abs(path), encoding='utf-8') as handle:
                lines_of[path] = handle.read().split('\n')
        lines = lines_of[path][first - 1:last]
        ok = bool(lines) and m_first in lines[0] and m_last in lines[-1]
        _need(ok, f'text segment {name}: markers not found at {path}:{first}-{last}', refusals)
        out[name] = {'source': path, 'source_sha256': _sha(path), 'lines': [first, last], 'text': '\n'.join(lines)}
    return out


# ======================================================================================================================
#  the declarations recorded with code assertions
# ======================================================================================================================
def declarations(refusals, s7_lattice, e1_lattice):
    import inspect
    d = {}
    d['b_plus'] = {'rule_text_in_the_e1_launcher': E1.B_PLUS_RULE, 'advance_window': E1.ADVANCE_WINDOW,
                   'halton_k0': E1.HALTON_K0, 'n_max_new_evaluations_per_arm': E1.N_MAX_NEW_EVALUATIONS,
                   'max_polls': E1.MAX_POLLS, 'completion_cap': E1.COMPLETION_CAP,
                   'min_feasible_poll_points': E1.MIN_FEASIBLE_POLL_POINTS, 'sigma_q_eur': E1.SIGMA_Q_EXPECTED_EUR,
                   'verbatim_ruling': 'text.addendum85_ruling2_e1_poll_rule_b_plus',
                   'five_confirmed_interpretation_choices_verbatim': 'text.addendum86_ruling3_declarations (iv)',
                   'five_choices_as_implemented': {
                       '(a) the window includes the first refused index': (
                           'ADVANCE_WINDOW = 10 consecutive Halton indices examined per launched poll (the refused one '
                           'included)'),
                       '(b) on exhaustion the terminating record is the first examined index, byte-equal to the frozen '
                       'loop, with the nine further indices in halton_indices_skipped': 'B_PLUS_RULE (iii)',
                       '(c) advance_rule.polled_index null on a cap refusal': 'ADVANCE_RULE_FIELDS includes polled_index',
                       '(d) --freeze/--run accept only P515S55/e1_2x and P515S55/e1_2x_batch4, cross under e1_2x/cross':
                           sorted(E1.DECLARED_ROOTS),
                       '(e) the renamed dry-run output': E1.DRY_RUN_FILE}}
    _need(E1.ADVANCE_WINDOW == 10 and E1.HALTON_K0 == 0 and E1.N_MAX_NEW_EVALUATIONS == 30 and E1.MAX_POLLS == 10
          and E1.COMPLETION_CAP == 30 and E1.MIN_FEASIBLE_POLL_POINTS == 8, 'B+ constants', refusals)
    _need(sorted(E1.DECLARED_ROOTS) == sorted([E1.E1_ROOT_REL, E1.E1_BATCH_ROOT_REL]), 'E1 declared roots', refusals)
    _need(E1.DRY_RUN_FILE == 'e1_launcher_v3_dryrun.json', 'E1 dry-run file name', refusals)
    _need('polled_index' in E1.ADVANCE_RULE_FIELDS and 'halton_indices_skipped' in E1.POLL_RECORD_FIELDS_E1,
          'B+ record fields', refusals)
    d['a7'] = {'barrier_stop_per_poll': PB.BARRIER_STOP_PER_POLL, 'barrier_stop_overall': PB.BARRIER_STOP_OVERALL,
               'verbatim': 'text.addendum85_ruling3_a7'}
    _need(PB.BARRIER_STOP_PER_POLL == 2 and PB.BARRIER_STOP_OVERALL == 3, 'A7 constants', refusals)
    d['rho_read_back'] = {'verbatim_ruling': 'text.addendum86_ruling2_rho_read_back',
                          'replication_formula_as_implemented': L7.RHO_RULE_TEXT,
                          'increase_factor': L7.RHO_INCREASE_FACTOR, 'decrease_divisor': L7.RHO_DECREASE_FACTOR,
                          'clamp': list(L7.RHO_CLAMP), 'channels': list(L7.RHO_CHANNELS),
                          'factor_one_actions_regex': L7.RHO_FACTOR_ONE_RX.pattern,
                          'validation_target_w183f_cycle1': {k: list(v) for k, v in L7.W183F_EXPECTED_CYCLE1.items()},
                          'a_mismatch_is': 'a harness-fault stop (exit 3) of the E2 cell'}
    rv = L7.rho_validation_w183f()
    d['rho_read_back']['validation_on_the_committed_w183f_arm_at_spec_v1'] = {
        'holds': rv['holds'], 'checks': rv['checks'], 'model_counts': rv['model_counts'],
        'n_cycles': rv['read_back']['n_cycles'], 'n_checks': rv['read_back']['n_checks'],
        'mismatches': rv['read_back']['mismatches'], 'cycle_1': rv['read_back']['cycle_1'],
        'record': rv['record'], 'solve_profile_record': rv['solve_profile_record']}
    _need(rv['holds'], f"rho read-back validation fails: {rv['checks']}", refusals)
    e2_cells = [c['cell'] for c in L7.cell_table('e2', s7_lattice)]
    s7_screen_entries = [L7.entry_options(c) for c in L7.cell_table('screen', s7_lattice)]
    e1_seed_entries, _rows = E1.seeds_entries(e1_lattice, E1.seed_plan(e1_lattice), None)
    e1_dom_any = E1.domain_entries(e1_lattice, 'N')[:3] + E1.domain_entries(e1_lattice, 'A')[:3]
    d['addendum86_ruling3'] = {
        'verbatim': 'text.addendum86_ruling3_declarations',
        '(i)_e2_completes_all_eight': {'n_cells': len(e2_cells), 'cells_in_launch_order': e2_cells,
                                       'h2_pairs_in_order': [{'price': p, 'rho0_scale': s,
                                                              'cells': [c for c in e2_cells
                                                                        if f'_{p}_rho{str(s).replace(".", "p")}' in c]}
                                                             for p, s in E2_PAIR_ORDER],
                                       'an_uncertified_e2_cell_is_not_a_stop': True,
                                       'a_failed_h2_pair': ('recorded as a failed prediction; E2 continues; the chain '
                                                            'stops before E1 (W186 waits for Addendum 87 in any case)')},
        '(ii)_persist_certified_models': {
            'e3': L7.PERSIST, 'e2': L7.PERSIST,
            'screen': 'off (no post_certification; production exit cap 500)',
            'e1_seeds_and_search': 'off (no post_certification; production exit cap 500)',
            'pickles': 'untracked under the roots; the manifests hash them'},
        '(iii)_e1_resumable_launches': {'launch_units': list(E1.LAUNCH_UNITS), 'default_unit': 'batch',
                                        'exit_suspended': E1.EXIT_SUSPENDED, 'resume_rule': E1.RESUME_RULE,
                                        'commit_between_launches': ('required: a resume needs the earlier launch '
                                                                    'records and manifests git-tracked and clean (the '
                                                                    'cache loader\'s rule; W183i (c) decision 2)')},
        '(iv)_b_plus_choices': 'see b_plus.five_choices_as_implemented', '(v)_window': E1.ADVANCE_WINDOW}
    _need(len(e2_cells) == 8 and all(len(p['cells']) == 2 for p in d['addendum86_ruling3']['(i)_e2_completes_all_eight']
                                     ['h2_pairs_in_order']), 'E2 pairs', refusals)
    _need(not any('post_certification' in o for o in s7_screen_entries), 'screen entries carry post_certification',
          refusals)
    _need(not any('post_certification' in e[2] for e in e1_seed_entries + e1_dom_any),
          'E1 entries carry post_certification', refusals)
    _need(L7.SCREEN_CAP == 500 and E1.CAP == 500, 'production-exit cap 500', refusals)
    _need('git-tracked and clean at --run' in E1.RESUME_RULE, 'E1 resume rule names the git condition', refusals)
    _need(E1.LAUNCH_UNITS == ('batch', 'poll'), 'E1 launch units', refusals)
    _need("args.launch_unit = 'batch'" in inspect.getsource(E1.main), 'E1 default launch unit is batch', refusals)
    d['stops'] = {
        'e3': {'as_implemented': L7.STOPS['e3'], 'exit_code': 3,
               'verbatim': ['text.addendum85_w184_line', 'text.addendum86_w184_w185_reauthorized']},
        'e2': {'as_implemented': L7.STOPS['e2'], 'exit_code': 3,
               'h2_failure': 'recorded as a failed prediction; E2 continues (Addendum 86 ruling 3 (i)); stop before E1',
               'uncertified_cell': 'not a stop (E2 completes all eight)',
               'verbatim': ['text.addendum85_w185_line', 'text.addendum86_ruling3_declarations',
                            'text.addendum86_w184_w185_reauthorized']},
        'screen': {'as_implemented': L7.STOPS['screen'],
                   'note': 'no declared stop; a barrier or other child exit is a fault (exit 1)'},
        'e1': {'search': ['STOP_FOR_REVIEW_barrier_rule (A7: 2 new barrier evaluations in one poll or 3 overall)',
                          'STOP_FOR_REVIEW_completion_cap (B+ window of 10 exhausted; a reportable outcome, ruled on if '
                          'it occurs, never patched mid-run)',
                          'evaluation_budget_exhausted / max_polls_reached (exit 2, reported as such)',
                          'a harness fault (harness_error point or a read-back mismatch) stops the arm at that launch '
                          '(exit 1; the next resume refuses; W183i (c) decision 3)'],
               'exit_codes': '0 done; 2 budget / max polls; 3 stop for review; 1 fault; 4 launch done, phase not '
                             'finished'},
        'cross': 'declared after W186'}
    d['counts_and_limits'] = {'verbatim': 'text.addendum85_ruling9_counts_and_limits',
                              'e3': 1, 'e2': 8, 'e1_max': 75, 'e1_seeds': 7, 'e1_batch': 4,
                              'e1_search_max_per_arm': 30, 'e1_cross_max': 4, 'screen': 15, 'e6': 0,
                              'serial_worst_case_h': 81, 'with_batch_gate_h': 49,
                              'over_4h': 'D1 covers runs over 4 h within these limits; every extension needs a ruling'}
    d['objective_convention'] = {'convention': OBJECTIVE_CONVENTION,
                                 'verbatim': 'text.addendum83_ruling2_objective_convention',
                                 'step7_launcher_constant': L7.OBJECTIVE_CONVENTION,
                                 'e1_launcher_constant': E1.OBJECTIVE_CONVENTION}
    d['hypotheses'] = {'index': HYPOTHESES_INDEX,
                       'note': 'verbatim in text.*; recorded hypotheses, none required (plan section 1)'}
    return d


# ======================================================================================================================
#  modes
# ======================================================================================================================
def freeze_spec(args):
    started = time.time()
    smoke = bool(args.smoke)
    if smoke:
        out_root = os.path.abspath(args.out_root)
        if out_root.startswith(REPO + os.sep) or os.path.exists(out_root):
            raise SystemExit(f'REFUSED: --out-root must be a NEW directory outside the repository: {out_root}')
        doc_dir, man_dir = out_root, out_root
    else:
        doc_dir, man_dir = _abs(P55), _abs(OUT_DIR_REL)
    refusals = []
    if not smoke:
        _need(_tracked_clean(SCRIPT_NAME), f'{SCRIPT_NAME} is not committed and clean', refusals)
        existing = [f for f in os.listdir(doc_dir) if SPEC_DOC_RX.match(f)]
        _need(not existing, f'spec v1 already exists (write-once): {existing}', refusals)
        _need(os.path.isdir(man_dir) and all(L7.LOG_RX.match(f) for f in os.listdir(man_dir)),
              f'{OUT_DIR_REL} must exist and hold only launch logs', refusals)
    _log(f'mode {"SMOKE" if smoke else "FREEZE-SPEC"}; HEAD {_git(["rev-parse", "HEAD"]).strip()}')
    content = {'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'version': 1,
               'predecessor': None,
               'predecessor_statement': (
                   'v1: no predecessor. Searched: git ls-files for frozen_*campaign_spec* and frozen_step7* (none), '
                   'data/SRP1/Results/P515S55/ (no spec v1 file), the reports P5_15_W183_REPORT.md ("Spec v1 is not '
                   'frozen"), P5_15_W183F_BITWISE_GATE.md (gate part only; spec v1 freeze held) and '
                   'P5_15_W183G_REPORT.md (W183g (d) held). The campaign spec v1 of Addendum 83 / Addendum 85 W183g (d) '
                   'is frozen here for the first time, on the patched harness (Addendum 86).'),
               'objective_convention': OBJECTIVE_CONVENTION, 'smoke': smoke, 'started_utc': _utc(),
               'interpreter': sys.executable, 'argv': sys.argv, 'script': SCRIPT_NAME,
               'script_sha256': _sha(SCRIPT_NAME)}
    _log('A. accepted state')
    content['pins'] = accepted_state(refusals, smoke)
    _log(f'   refusals so far: {len(refusals)}')
    _log('B. keys (Step-7 launcher dry-run code)')
    s7, s7_lattice, _s7_lev = step7_keys(refusals)
    _log('B. keys (E1 launcher dry-run code)')
    e1, e1_lattice, e1_lev = e1_keys(refusals)
    content['keys'] = {'step7': s7, 'e1': e1,
                       'phase_key_counts': {'e3': s7['phases']['e3']['n_distinct_eval_keys'],
                                            'e2': s7['phases']['e2']['n_distinct_eval_keys'],
                                            'screen': s7['phases']['screen']['n_distinct_eval_keys'],
                                            'screen_alias': 1,
                                            'e1_seeds': e1['seed_key_summary']['n_distinct_keys_to_evaluate'],
                                            'e1_seeds_alias': e1['seed_key_summary']['n_aliases'],
                                            'e1_batch4': len(e1['batch4']['seeds']),
                                            'e1_search_domain_per_arm': e1['search_domain_keys']['n_points'],
                                            'cross': 0}}
    _log(f"   key counts {content['keys']['phase_key_counts']}; refusals so far: {len(refusals)}")
    _log('D. the frozen E3 / E2 roots')
    fr = frozen_roots(refusals, s7_lattice, smoke)
    _log('C. capture checklists (fail fast)')
    content['capture_checklists'] = {'launchers': capture_checklists(refusals, s7_lattice, e1_lattice, e1_lev, fr),
                                     'spec_required_quantities_addendum83': spec_required_capture_map(refusals)}
    content['frozen_roots'] = {p: {k: v for k, v in fr[p].items() if k != 'specs'} for p in ('e3', 'e2')}
    content['frozen_roots']['not_frozen_roots'] = fr['not_frozen_roots']
    content['launch_commands'] = {
        p: {c: {'launch': v['launch_command'], 'preconditions_only': v['preconditions_only_command']}
            for c, v in (fr[p].get('cells') or {}).items()} for p in ('e3', 'e2')}
    content['launch_commands']['report'] = {p: (fr[p] or {}).get('report_command') for p in ('e3', 'e2')}
    content['roots'] = {
        'e3': L7.PHASE_ROOTS['e3'], 'e2': L7.PHASE_ROOTS['e2'], 'screen': L7.PHASE_ROOTS['screen'],
        'e1_2x': {'root': E1.E1_ROOT_REL, 'phases': ['seeds', 'search_A', 'search_N', 'cross'],
                  'campaign_dirs': e1['roots'][E1.E1_ROOT_REL]['phases'], 'cross_root': L7.PHASE_ROOTS['cross']},
        'e1_2x_batch4': {'root': E1.E1_BATCH_ROOT_REL, 'phases': ['seeds'],
                         'campaign_dirs': e1['roots'][E1.E1_BATCH_ROOT_REL]['phases']},
        'frozen_now': ['e3', 'e2'], 'frozen_after_addendum_87': ['e1_2x', 'e1_2x_batch4', 'screen'],
        'cross': 'declared only (entries after W186)'}
    _log('E. declarations and verbatim text')
    content['declarations'] = declarations(refusals, s7_lattice, e1_lattice)
    content['text'] = text_segments(refusals)
    g = guards_report()
    content['guards_before_write'] = g
    _need(g['all_zero'], 'guards not all zero before the write', refusals)
    if refusals:
        for r in refusals:
            _log(f'[REFUSED] {r}')
        _log(f'REFUSED: {len(refusals)} refusal(s); nothing written')
        return 1
    content['finished_utc'] = _utc()
    content['wall_clock_s_before_write'] = time.time() - started
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    os.makedirs(doc_dir, exist_ok=True)
    doc_path = os.path.join(doc_dir, SPEC_DOC_FMT.format(sha[:8]))
    with open(doc_path, 'x') as handle:
        handle.write(text)
    if H.sha256_file(doc_path) != sha:
        raise RuntimeError('the spec document does not hash to its name')
    _log(f'wrote {_rel(doc_path) if not smoke else doc_path} sha256 {sha}')
    manifest = {'schema': 'p515_s55_w183i_d_manifest_v1', 'spec_v1': {'path': _rel(doc_path) if not smoke else doc_path,
                                                                     'sha256': sha},
                'script': {'path': SCRIPT_NAME, 'sha256': _sha(SCRIPT_NAME)}, 'frozen_files': {}}
    for p in ('e3', 'e2'):
        for f in fr[p].get('files_in_phase_root') or []:
            manifest['frozen_files'][f] = _sha(f)
    g2 = guards_report()
    manifest['guards_at_end'] = {'all_zero': g2['all_zero'], 'n_guards_verified': g2['n_guards_verified']}
    os.makedirs(man_dir, exist_ok=True)
    man_text = GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n'
    man_path = os.path.join(man_dir, MANIFEST_FILE)
    with open(man_path, 'x') as handle:
        handle.write(man_text)
    _log(f"wrote manifest {man_path} sha256 {H.sha256_file(man_path)}; guards all_zero {g2['all_zero']} "
         f"({g2['n_guards_verified']} guards)")
    for p in ('e3', 'e2'):
        for c, v in (fr[p].get('cells') or {}).items():
            _log(f"  {p} {c}: spec {v['campaign_spec_sha256']} eval {v['eval_key'][:16]}")
        if fr[p].get('stage_spec'):
            _log(f"  {p} stage spec {fr[p]['stage_spec']} {fr[p]['stage_spec_sha256']}")
    _log(f'wall {time.time() - started:.1f} s')
    return 0 if g2['all_zero'] else 1


def post_manifest(_args):
    man_dir = _abs(OUT_DIR_REL)
    path = os.path.join(man_dir, POST_MANIFEST_FILE)
    if os.path.exists(path):
        raise SystemExit(f'REFUSED: {path} exists (write-once)')
    docs = [f for f in os.listdir(_abs(P55)) if SPEC_DOC_RX.match(f)]
    if len(docs) != 1:
        raise SystemExit(f'REFUSED: expected exactly one spec v1 document, found {docs}')
    files = [os.path.join(P55, docs[0])]
    for p in ('e3', 'e2'):
        for directory, _dirs, fnames in os.walk(_abs(L7.PHASE_ROOTS[p])):
            files += [_rel(os.path.join(directory, f)) for f in sorted(fnames)]
    for f in sorted(os.listdir(man_dir)):
        files.append(os.path.join(OUT_DIR_REL, f))
    files.append(SCRIPT_NAME)
    man = {'schema': 'p515_s55_w183i_d_post_preconditions_manifest_v1', 'utc': _utc(),
           'git_head': _git(['rev-parse', 'HEAD']).strip(),
           'sha256': {f: _sha(f) for f in files},
           'git_tracked_and_clean': {f: _tracked_clean(f) for f in files}}
    log = os.path.join(L7.PHASE_ROOTS['e3'], 'run_e3_cnrf_x0_1x_preconditions_only.log')
    if os.path.isfile(_abs(log)):
        with open(_abs(log)) as handle:
            man['preconditions_only_log_lines'] = [ln.rstrip('\n') for ln in handle
                                                   if '[RUN' in ln or 'final guards' in ln]
    g = guards_report()
    man['guards'] = {'all_zero': g['all_zero'], 'n_guards_verified': g['n_guards_verified']}
    text = GRIO.dumps(man, indent=1, sort_keys=True) + '\n'
    with open(path, 'x') as handle:
        handle.write(text)
    _log(f'wrote {_rel(path)} sha256 {H.sha256_file(path)} ({len(files)} files); guards all_zero {g["all_zero"]}')
    return 0 if g['all_zero'] else 1


def main():
    parser = argparse.ArgumentParser(description='W183i (d) Step-7 campaign spec v1 freeze (zero solves)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--post-manifest', action='store_true')
    parser.add_argument('--out-root', help='--smoke only: a NEW directory outside the repository')
    args = parser.parse_args()
    if args.smoke and not args.out_root:
        parser.error('--smoke needs --out-root')
    if not args.smoke and args.out_root:
        parser.error('--out-root is for --smoke only')
    try:
        code = post_manifest(args) if args.post_manifest else freeze_spec(args)
    except Refusal as error:
        _log(f'REFUSED: {error}')
        code = 1
    g = guards_report()
    _log(f"final guards: all_zero={g['all_zero']} n={g['n_guards_verified']} pickle={g['pickle']} "
         f"build={g['model_build_tripwires']} e1_tripwires_nonzero="
         f"{ {k: v for k, v in g['e1_launcher_committed_function_tripwires'].items() if v} }")
    sys.exit(code if g['all_zero'] else 1)


if __name__ == '__main__':
    main()
