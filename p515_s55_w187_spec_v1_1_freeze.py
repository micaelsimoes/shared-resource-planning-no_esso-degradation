"""
P5.15 Addendum 90 ruling 4, Planner task W187 preparation (resumed) -- the STEP-7 CAMPAIGN SPEC v1.1 FREEZE: spec v1
(`frozen_step7_campaign_spec_v1_96345241.json`, sha256 96345241..., production 6f43479e) re-pinned on HEAD
(production f6c23977, Addendum 89 / 90). ZERO SOLVES, no model build, pickle blocked.

Spec v1.1 = spec v1's content (a deep copy of the committed file, sha256-checked) with ONLY these changes, each listed
in `changes_from_v1` with its old and new value:
  * schema / version / stage / authority (v1's kept as `authority_v1`) / script / predecessor (spec v1 path + sha256)
    / timestamps / argv / guards;
  * the production head 6f43479e -> f6c23977 and the 38 module hashes (exactly the three f6c23977 modules differ:
    shared_energy_storage_data.py, shared_energy_storage_parameters.py, shared_resources_planning.py);
  * the Step-7 launcher's new sha256 (the G0 re-pin: exactly the two constants G0_ARTIFACT_REL / G0_ARTIFACT_SHA256;
    the unified diff against the v1 blob 3b990d19 is asserted to be those two lines), its run-file and pins_now sets
    recomputed by the launcher's own code;
  * the G0 re-pin: artifact `P515S55/w183_nl_post_w183c_off/w183_nl_digest.json.gz` ca195bd3 -> the gate (i) pin-check
    artifact `P515S55/pincheck_nl_head_f6c23977/w183_nl_digest.json.gz` 315ca94a (built at f6c23977, commit 09b3e3cf);
    the W183d G0 record pin `w183d_g0/g0.json` 8cef45d6 unchanged; `L7.g0_check(require_current_production=True)`
    passes every item;
  * the SRP1 inputs 81 -> 83: the 81 of v1 unchanged; the two P5.16 inputs (`data/SRP1/SRP1_5x3y_1x1.json`,
    `data/SRP1/SharedESS/SRP1_5x3y_ESS_Params.json`) recorded as "declared outside every SRP1 run's read set (static
    search: ...), pinned by the launcher" (Planner ruling within Addendum 90 ruling 4; no exclusion-list code);
  * gate (i): the pin check 09b3e3cf cited (696/696 blocks identical, Q aggregation AST-equal);
  * gate (ii): the 153-solve two-cycle gate on HEAD (`p515_s55_w187_bitwise_gate.py`) -- record path, sha256,
    verdicts, solve counts (`--freeze-spec` refuses unless it is committed and GATE_PASS_RULED is True);
  * the Step-7 keys (E3 1, E2 8, screen 15 + alias) recomputed by the launcher's own dry-run code
    (`p515_s55_w183i_d_spec_v1_freeze.step7_keys`, imported) and asserted equal to the committed W183i (b) dry run
    (f4bc728c) and to spec v1's key tables;
  * the screen's phase-level parent capture checklist re-asserted at v1.1.
Carried from v1 unchanged, and said so in `carried_from_v1_unchanged`: the E1 launcher sections (E1 is outside W187;
its pins are v1's and an E1 launch needs its own re-pin), the frozen E3 / E2 roots (run under spec v1 on 2026-10-26,
Addendum 90 correction 1; not re-run), the declarations and verbatim text (line ranges of the brief at 8b83aeb5),
the Addendum 83 capture map.

MODES (one per process; repo root; canonical interpreter; attached; both streams captured)
  --check         ZERO SOLVES, WRITES NOTHING: every assertion of --freeze-spec except gate (ii) (recorded as pending)
                  and except that the launcher, this script and the gate (ii) wrapper may be uncommitted (the step-1
                  tests before the commit). Exit 0 iff no refusal.
  --freeze-spec   Writes ONCE `data/SRP1/Results/P515S55/frozen_step7_campaign_spec_v1_1_<sha8>.json` (through
                  `gate_result_io`; the name carries its own sha8) and `.../w187_spec_v1_1/w187_spec_v1_1_manifest_sha256.json`.
                  Refuses (nothing written, exit 1) on any failed assertion.
  --post-manifest Writes ONCE `.../w187_spec_v1_1/w187_post_preconditions_manifest_sha256.json` over spec v1.1, the
                  frozen screen root, the gate (ii) root manifests and this directory (incl. the preconditions-only log).

Objective convention (declared, no objective computed here): Q = gross_operational_cost, contracted interface
settlement EXCLUDED (Addendum 83 ruling 2); salvage reported separately; net = gross - terminal salvage.
"""

import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W187 spec v1.1 freeze (never solves)').install()

# the spec v1 freeze module: installs its own guard, the pickle block and the model-build tripwire BEFORE it imports
# both launchers; its step7_keys / guards_report are used unchanged (the launcher's own dry-run code)
import p515_s55_w183i_d_spec_v1_freeze as V1F  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

L7 = V1F.L7
E1 = V1F.E1
H = L7.H
PB = L7.PB

SCRIPT_NAME = os.path.basename(__file__)
STAGE = ('P5.15 Addendum 90 ruling 4 -- Step-7 campaign spec v1.1: spec v1 re-pinned on HEAD (production f6c23977), '
         'G0 re-pinned to the gate (i) pin-check artifact, gates (i) and (ii) cited (Planner task W187 preparation)')
SCHEMA = 'p515_s55_step7_campaign_spec_v1_1'
VERSION = '1.1'
P55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
OUT_DIR_REL = os.path.join(P55, 'w187_spec_v1_1')
SPEC_DOC_FMT = 'frozen_step7_campaign_spec_v1_1_{}.json'
SPEC_DOC_RX = re.compile(r'^frozen_step7_campaign_spec_v1_1_([0-9a-f]{8})\.json$')
MANIFEST_FILE = 'w187_spec_v1_1_manifest_sha256.json'
POST_MANIFEST_FILE = 'w187_post_preconditions_manifest_sha256.json'
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 90 ruling 4 (W187 under spec v1.1; gates (i) and (ii)); corrections 1-2',
    'PLANNER_BRIEF_2026-09-13.md Addendum 89 (pin check; W187 order)',
    'Planner task W187 preparation (resumed): G0 re-pin option (b) (the two launcher constants); the two P5.16 inputs '
    'stay pinned by the launcher, recorded as outside every SRP1 run\'s read set',
    'P5_15_PINCHECK_F6C23977.md (gate (i), commit 09b3e3cf)',
]

SPEC_V1 = {'path': os.path.join(P55, 'frozen_step7_campaign_spec_v1_96345241.json'),
           'sha256': '963452416fd4882bc14b0f5a28d5cef7bccf93d0ff57860292e9d6a91220f3c0'}
PRODUCTION_HEAD_V1 = '6f43479ed8ece7a4a5878942ca38952ce1cae1bc'
PRODUCTION_HEAD = 'f6c23977413c3dff9ee42281a4fcadab8ecfd994'
N_PRODUCTION_MODULES = 38
F6C23977_MODULES = {   # module -> (spec v1 pin, the f6c23977 sha256)
    'shared_energy_storage_data.py': ('9acd095fb5e50e9b64c253600ae05ebb8e6463fd7a999d3a90f96ee82f037796',
                                      'e8bf4f923e28db83480e786340c32a291eca56288fa17f15ecac7fb72fc3cb85'),
    'shared_energy_storage_parameters.py': ('5fe0c4fb876a9c2f1e506e982f2a120c320a3fd4198b870f877d9a310cd390bb',
                                            'f31a2caa83a7c07f8f2375bf2016dd2180ff470ee33c1df0a04ece038d46aff6'),
    'shared_resources_planning.py': ('ade9c843bc7c5b6514ab91ecc54f327330a5249709766699ff2f50c8288473d6',
                                     'c9336a61a713ef6890fbffcda845395d757e26c41ae2422e82dd88bb881c33e8'),
}
LAUNCHER_REL = L7.SCRIPT_NAME
LAUNCHER_V1_SHA256 = '3b990d19bbde8e9fb6bf75d2e5447c55116d4a3d4d21c174e53776dffc561757'
LAUNCHER_DIFF_EXPECTED = {
    'removed': ["G0_ARTIFACT_REL = os.path.join(P55, 'w183_nl_post_w183c_off', 'w183_nl_digest.json.gz')\n",
                "G0_ARTIFACT_SHA256 = 'ca195bd38fe2384493012dcf7127a36a8923d8a5029c2c821b5bb0f47f57ab6a'\n"],
    'added': ["G0_ARTIFACT_REL = os.path.join(P55, 'pincheck_nl_head_f6c23977', 'w183_nl_digest.json.gz')\n",
              "G0_ARTIFACT_SHA256 = '315ca94a8d2ccd59d5c9ee045694fcc998030995ad322fc99fa2fe49bd0da5b4'\n"]}
G0_OLD = {'path': os.path.join(P55, 'w183_nl_post_w183c_off', 'w183_nl_digest.json.gz'),
          'sha256': 'ca195bd38fe2384493012dcf7127a36a8923d8a5029c2c821b5bb0f47f57ab6a',
          'built_at_production': PRODUCTION_HEAD_V1}
G0_NEW = {'path': os.path.join(P55, 'pincheck_nl_head_f6c23977', 'w183_nl_digest.json.gz'),
          'sha256': '315ca94a8d2ccd59d5c9ee045694fcc998030995ad322fc99fa2fe49bd0da5b4',
          'built_at_production': PRODUCTION_HEAD, 'commit': '09b3e3cfd210bf30e8e6d55091c4e326dc886fb0'}
G0_RECORD_PIN = {'path': os.path.join(P55, 'w183d_g0', 'g0.json'),
                 'sha256': '8cef45d684bc7e5dc8b719fd504b8e7705933597a9ba07f44349650046b2911c'}
N_SRP1_INPUTS_V1 = 81
P516_INPUTS = {
    os.path.join('data', 'SRP1', 'SRP1_5x3y_1x1.json'):
        'f916e0c6feff4ce1bdf62d843efef386b791e7649442176bacfa16cec3d55b79',
    os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_5x3y_ESS_Params.json'):
        '08ade261858aeadd852c9f823c9f273b7490ba426c52a50f229fdbca80a50517',
}
P516_STATUS_TEXT = ("declared outside every SRP1 run's read set (static search: see p516_inputs.static_search), "
                    'pinned by the launcher (pins_now inputs; a change refuses every --run)')
GATE_I = {'commit': '09b3e3cfd210bf30e8e6d55091c4e326dc886fb0', 'report': 'P5_15_PINCHECK_F6C23977.md',
          'compare_json': {'path': os.path.join(P55, 'pincheck_nl_head_f6c23977', 'pincheck', 'pincheck_compare.json'),
                           'sha256': '961b8350592c94e0fb23df4bb42eca5deda2dd55a53289335d3f2fc4646a578d'},
          'ast_json': {'path': os.path.join(P55, 'pincheck_nl_head_f6c23977', 'pincheck', 'pincheck_ast.json'),
                       'sha256': '6d490e2e4a3f3a8e15af2299a812b6e962f4b1c3762d095cfab2e52eebed01c0'},
          'digest': {'path': G0_NEW['path'], 'sha256': G0_NEW['sha256']}}
Q_AGGREGATION_FUNCTION = 'shared_resources_planning.py:_get_operational_recourse_components'
GATE_II_WRAPPER = 'p515_s55_w187_bitwise_gate.py'
GATE_II_ROOT = os.path.join(P55, 'w187_v1_1_srp1_bitwise_gate')
GATE_II = {'wrapper': GATE_II_WRAPPER, 'root': GATE_II_ROOT,
           'record': os.path.join(GATE_II_ROOT, 'w187_wrapper_record.json'),
           'manifests': [os.path.join(GATE_II_ROOT, f) for f in ('manifest_sha256.json', 'w51_manifest_sha256.json',
                                                                 'w187_manifest_sha256.json')],
           'launch_log': os.path.join(P55, 'w187_v1_1_srp1_bitwise_gate_launch.log'),
           'declared_solves': '51 solves/cycle x (cap 2 + 1) = 153, plus every attempted retry',
           'recorded_prediction': '0 genuine, GATE_PASS_RULED True, provenance <= 4 (expert, Addendum 90 ruling 4)'}
RULING4_TEXT = ('PLANNER_BRIEF_2026-09-13.md', 4855, 4862, '4. **W187 — spec v1.1 (option (a)) authorized.**',
                'blind prediction (recorded 11-10, before the run).')
UNCOMMITTED_ALLOWED_IN_CHECK = (LAUNCHER_REL, SCRIPT_NAME, GATE_II_WRAPPER)
FS_LISTING_NAMES = ('listdir', 'scandir', 'walk', 'glob', 'iglob', 'rglob', 'iterdir')
INSPECTED_LISTING_CALLS = {   # inspected line by line when this script was written; the freeze refuses on any change
    'n': 47, 'sha256': 'd2fdc8f2c8b984df5167581bd5b6a8d38719e263c6d687f0a447dad9459fde78',
    'verdict': ('every call lists a results / campaign / eval / output / scratch / frozen-snapshot directory under '
                'data/SRP1/Results or the scratch, or is ast.walk; none lists data/SRP1 outside Results/ and Diagrams/ '
                '(the launcher reads the tracked input set through git ls-files to HASH it: that is the pin)')}


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W187-v1.1] {msg}', flush=True)


_abs, _rel, _sha, _sha_or_none, _git, _tracked_clean, _load, _need = (
    V1F._abs, V1F._rel, V1F._sha, V1F._sha_or_none, V1F._git, V1F._tracked_clean, V1F._load, V1F._need)


def _git_bytes(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, check=True).stdout


def _is_ancestor(commit):
    return subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=REPO).returncode == 0


def _clean_or_allowed(rel, check):
    return _tracked_clean(rel) or (check and rel in UNCOMMITTED_ALLOWED_IN_CHECK)


# ======================================================================================================================
#  the launcher diff (exactly the two G0 constants)
# ======================================================================================================================
def launcher_diff(refusals):
    blob, blob_commit = None, None
    for commit in _git(['log', '--format=%H', '--', LAUNCHER_REL]).split():
        data = _git_bytes(['show', f'{commit}:{LAUNCHER_REL}'])
        if hashlib.sha256(data).hexdigest() == LAUNCHER_V1_SHA256:
            blob, blob_commit = data.decode(), commit
            break
    _need(blob is not None, f'no committed blob of {LAUNCHER_REL} hashes to the spec v1 pin {LAUNCHER_V1_SHA256}',
          refusals)
    with open(_abs(LAUNCHER_REL)) as handle:
        now = handle.read()
    if blob is None:
        return {'v1_blob_found': False}
    diff = list(difflib.unified_diff(blob.splitlines(keepends=True), now.splitlines(keepends=True),
                                     f'a/{LAUNCHER_REL} (v1 blob {LAUNCHER_V1_SHA256[:8]})',
                                     f'b/{LAUNCHER_REL} (now {_sha(LAUNCHER_REL)[:8]})', n=0))
    hunks = [ln for ln in diff if ln.startswith('@@')]
    removed = [ln[1:] for ln in diff if ln.startswith('-') and not ln.startswith('---')]
    added = [ln[1:] for ln in diff if ln.startswith('+') and not ln.startswith('+++')]
    out = {'v1_blob_found': True, 'v1_blob_commit': blob_commit, 'v1_sha256': LAUNCHER_V1_SHA256,
           'now_sha256': _sha(LAUNCHER_REL), 'n_hunks': len(hunks), 'n_removed': len(removed), 'n_added': len(added),
           'unified_diff_n0': ''.join(diff),
           'exactly_the_two_g0_constants': (len(hunks) == 1 and removed == LAUNCHER_DIFF_EXPECTED['removed']
                                            and added == LAUNCHER_DIFF_EXPECTED['added'])}
    _need(out['exactly_the_two_g0_constants'], f'launcher diff vs v1 is not exactly the two G0 constants: {diff}',
          refusals)
    return out


# ======================================================================================================================
#  the accepted state at v1.1 (each item against spec v1's pins)
# ======================================================================================================================
def accepted_state(refusals, v1, check):
    out = {}
    prod = L7.production_now()
    clean = {m: _tracked_clean(m) for m in prod['modules_sha256']}
    out['production'] = {'definition': prod['definition'], 'head': prod['head'], 'n_modules': len(prod['modules_sha256']),
                         'modules_sha256': prod['modules_sha256'], 'modules_tracked_and_clean': clean}
    _need(prod['head'] == PRODUCTION_HEAD, f"production head {prod['head']} != {PRODUCTION_HEAD}", refusals)
    _need(len(prod['modules_sha256']) == N_PRODUCTION_MODULES, f"{len(prod['modules_sha256'])} production modules",
          refusals)
    _need(all(clean.values()), f'production modules not clean: {[m for m, c in clean.items() if not c]}', refusals)
    a, b = v1['pins']['production']['modules_sha256'], prod['modules_sha256']
    differing = sorted(m for m in set(a) | set(b) if a.get(m) != b.get(m))
    out['production_modules_differing_from_v1'] = {m: {'v1': a.get(m), 'v1_1': b.get(m)} for m in differing}
    _need(differing == sorted(F6C23977_MODULES), f'modules differing from v1 {differing} != the three f6c23977 '
          'modules', refusals)
    for m, (old, new) in F6C23977_MODULES.items():
        _need(a.get(m) == old and b.get(m) == new, f'{m}: v1 {a.get(m)} / now {b.get(m)} != {old} / {new}', refusals)
    _need([c for c in _git(['log', '--format=%H', f'{PRODUCTION_HEAD_V1}..HEAD', '--'] + sorted(b)).split()]
          == [PRODUCTION_HEAD], f'production commits since {PRODUCTION_HEAD_V1} are not exactly {PRODUCTION_HEAD}',
          refusals)
    # the Step-7 launcher's run files (its own list): only the launcher differs from v1
    run_now = L7.code_sha256()
    run_v1 = v1['pins']['step7_launcher_run_files_sha256']
    rdiff = sorted(f for f in set(run_v1) | set(run_now) if run_v1.get(f) != run_now.get(f))
    out['step7_launcher_run_files_sha256'] = run_now
    out['step7_launcher_run_files_differing_from_v1'] = {f: {'v1': run_v1.get(f), 'v1_1': run_now.get(f)}
                                                         for f in rdiff}
    _need(rdiff == [LAUNCHER_REL], f'Step-7 run files differing from v1: {rdiff} != [{LAUNCHER_REL}]', refusals)
    for f in run_now:
        _need(_clean_or_allowed(f, check), f'run file {f} not committed and clean', refusals)
    # the v1 pinned / recorded files: unchanged except the launcher and the G0 artifact (re-pinned)
    files = {}
    for rel, entry in v1['pins']['files'].items():
        if rel == G0_OLD['path']:
            continue
        got = _sha_or_none(rel)
        files[rel] = {'sha256': got, 'tracked_and_clean': _clean_or_allowed(rel, check)}
        if 'expected' in entry:
            exp = _sha(LAUNCHER_REL) if rel == LAUNCHER_REL else entry['expected']
            files[rel].update({'expected': exp, 'equals_expected': got == exp})
            _need(got == exp, f'{rel}: sha256 {got} != {exp}', refusals)
        elif rel != 'PLANNER_BRIEF_2026-09-13.md':   # the brief has grown (Addenda 87-90); v1's text cites 8b83aeb5
            _need(got == entry.get('sha256'), f'{rel}: {got} != v1 {entry.get("sha256")}', refusals)
        _need(files[rel]['tracked_and_clean'] or rel == 'PLANNER_BRIEF_2026-09-13.md', f'{rel}: not clean', refusals)
    g0_now = _sha_or_none(G0_NEW['path'])
    files[G0_NEW['path']] = {'sha256': g0_now, 'expected': G0_NEW['sha256'], 'equals_expected': g0_now == G0_NEW['sha256'],
                             'tracked_and_clean': _tracked_clean(G0_NEW['path'])}
    _need(files[G0_NEW['path']]['equals_expected'] and files[G0_NEW['path']]['tracked_and_clean'],
          'the new G0 artifact is not as pinned / not committed', refusals)
    files[GATE_II_WRAPPER] = {'sha256': _sha_or_none(GATE_II_WRAPPER),
                              'tracked_and_clean': _clean_or_allowed(GATE_II_WRAPPER, check)}
    files[SCRIPT_NAME] = {'sha256': _sha(SCRIPT_NAME), 'tracked_and_clean': _clean_or_allowed(SCRIPT_NAME, check)}
    for rel in (GATE_II_WRAPPER, SCRIPT_NAME):
        _need(files[rel]['sha256'] and files[rel]['tracked_and_clean'], f'{rel}: missing or not clean', refusals)
    out['files'] = files
    # the SRP1 inputs: 81 of v1 unchanged + the two P5.16 inputs
    inputs = {f: _sha(f) for f in L7.tracked_inputs()}
    in_v1 = v1['pins']['srp1_inputs']['sha256']
    shared_equal = all(inputs.get(f) == s for f, s in in_v1.items())
    new = sorted(set(inputs) - set(in_v1))
    out['srp1_inputs'] = {'definition': v1['pins']['srp1_inputs']['definition'], 'n': len(inputs), 'sha256': inputs,
                          'n_v1': len(in_v1), 'v1_inputs_unchanged': shared_equal, 'new_since_v1': new,
                          'removed_since_v1': sorted(set(in_v1) - set(inputs)),
                          'all_tracked_and_clean': not _git(['status', '--porcelain', '--'] + sorted(inputs)).strip()}
    _need(len(in_v1) == N_SRP1_INPUTS_V1 and shared_equal and not out['srp1_inputs']['removed_since_v1'],
          'the 81 SRP1 inputs of v1 are not unchanged', refusals)
    _need(new == sorted(P516_INPUTS) and all(inputs[f] == s for f, s in P516_INPUTS.items()),
          f'new SRP1 inputs {new} are not exactly the two P5.16 inputs at their pins', refusals)
    _need(out['srp1_inputs']['all_tracked_and_clean'], 'an SRP1 input is not clean', refusals)
    named = {}
    for k, v in L7.NAMED_INPUTS.items():
        named[k] = ({'path': v, 'sha256': inputs.get(v)} if isinstance(v, str)
                    else [{'path': p, 'sha256': inputs.get(p)} for p in v])
    out['named_inputs'] = named
    _need(named == v1['pins']['named_inputs'], 'named inputs differ from v1', refusals)
    # tables, committed records, ParallelExecution, solver, dry-run outputs: as v1
    tables = {}
    for name, pin in L7.SEARCH_TABLES.items():
        got = _sha_or_none(pin['path'])
        tables[name] = {'path': pin['path'], 'sha256_pinned': pin['sha256'], 'sha256': got,
                        'equals_pin': got == pin['sha256'], 'tracked_and_clean': _tracked_clean(pin['path'])}
        _need(tables[name]['equals_pin'] and tables[name]['tracked_and_clean'], f'table {name} not as pinned', refusals)
    out['search_tables'] = tables
    _need(tables == v1['pins']['search_tables'], 'search tables differ from v1', refusals)
    recs = L7.record_pins_check()
    out['committed_records'] = recs
    _need(all(v['holds'] for v in recs.values()), f'committed records not as pinned: {recs}', refusals)
    case = _load(V1F.SRP1_CASE_JSON)
    out['parallel_execution'] = dict(v1['pins']['parallel_execution'], sha256=_sha(V1F.SRP1_CASE_JSON),
                                     ParallelExecution=case.get('ParallelExecution'))
    _need(out['parallel_execution'] == v1['pins']['parallel_execution'], 'SRP1.json / ParallelExecution differ from v1',
          refusals)
    solver = L7.solver_check()
    out['solver'] = {'path': V1F.SOLVER, 'resolved': solver['resolved'], 'sha256': solver['sha256'], 'ok': solver['ok']}
    _need(solver['ok'] and out['solver'] == v1['pins']['solver'], f'solver differs from v1: {solver}', refusals)
    dr = {}
    for name, pin in V1F.COMMITTED_DRY_RUNS.items():
        got = _sha_or_none(pin['path'])
        dr[name] = {'path': pin['path'], 'sha256': got, 'equals_pin': got == pin['sha256'],
                    'tracked_and_clean': _tracked_clean(pin['path'])}
        _need(dr[name]['equals_pin'] and dr[name]['tracked_and_clean'], f'dry-run output {name} not as pinned', refusals)
    out['committed_dry_run_outputs'] = dr
    # the launcher's complete pin set, as the launcher computes it now
    pn = L7.pins_now()
    out['step7_launcher_pins_now'] = pn
    unc = [ln for ln in pn['uncommitted_changes'] if not (check and ln.split()[-1] in UNCOMMITTED_ALLOWED_IN_CHECK)]
    unt = [f for f in pn['untracked'] if not (check and f in UNCOMMITTED_ALLOWED_IN_CHECK)]
    _need(not unc and not unt, f"pins_now: uncommitted {pn['uncommitted_changes']} untracked "
          f"{pn['untracked']}", refusals)
    _need(pn['g0']['artifact'] == {'path': G0_NEW['path'], 'sha256': G0_NEW['sha256']}, 'pins_now G0 artifact', refusals)
    _need(pn['g0']['record'] == G0_RECORD_PIN and L7.G0_RECORD == G0_RECORD_PIN, 'the W183d G0 record pin changed',
          refusals)
    out['step7_pins_recorded_check'] = L7.pins_recorded_check(pn)
    _need(all(out['step7_pins_recorded_check'].values()), 'step-7 pins_recorded_check fails', refusals)
    out['e1_launcher_sha256_now'] = _sha('p515_s55_e1_launcher.py')
    out['git_head'] = _git(['rev-parse', 'HEAD']).strip()
    return out


# ======================================================================================================================
#  G0 (explicit, all items)
# ======================================================================================================================
def g0_now(refusals, lattice):
    ok, alias = L7.screen_alias(lattice, require_current_production=True)
    checks = alias['g0'].get('checks') or {}
    out = {'accepted': ok, 'n_items': len(checks), 'n_true': sum(v is True for v in checks.values()),
           'checks': checks, 'artifact_production': alias['g0'].get('artifact_production'),
           'require_current_production': True, 'artifact': {'path': L7.G0_ARTIFACT_REL, 'sha256': L7.G0_ARTIFACT_SHA256},
           'record': L7.G0_RECORD, 'alias_eval_key': alias['alias_eval_key'], 'reuse_of': alias['reuse_of']}
    _need(ok and checks and all(v is True for v in checks.values()), f'G0 not accepted: {checks}', refusals)
    _need({'path': L7.G0_ARTIFACT_REL, 'sha256': L7.G0_ARTIFACT_SHA256} == {'path': G0_NEW['path'],
                                                                           'sha256': G0_NEW['sha256']},
          'the launcher G0 constants are not the gate (i) artifact', refusals)
    return out


# ======================================================================================================================
#  the two P5.16 inputs: static search of every SRP1 run's read set
# ======================================================================================================================
def _fs_listing_calls(rel):
    with open(_abs(rel)) as handle:
        src = handle.read()
    lines = src.split('\n')
    hits = []
    for node in ast.walk(ast.parse(src, filename=rel)):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name in FS_LISTING_NAMES:
                hits.append({'line': node.lineno, 'call': name, 'text': lines[node.lineno - 1].strip()})
    return sorted(hits, key=lambda h: (h['line'], h['call']))


def _listing_set_sha(hits_by_file):
    rows = sorted(f'{f}:{h["line"]}:{h["call"]}:{h["text"]}' for f, hs in hits_by_file.items() for h in hs)
    return hashlib.sha256('\n'.join(rows).encode()).hexdigest(), len(rows)


def p516_inputs(refusals, gate_clean_files):
    prod = sorted(L7.production_now()['modules_sha256'])
    run_files = list(L7.RUN_FILES)
    case_files = [V1F.SRP1_CASE_JSON, H.CASE_FILE_REL, H.ESS_PARAMS_FILE_REL, PB.COST_FILE['path']] + list(
        L7.NAMED_INPUTS['network_params_files'])
    read_set_code = sorted(set(prod) | set(run_files) | {f for f in gate_clean_files if f.endswith('.py')})
    read_set_data = sorted(set(case_files) | {f for f in gate_clean_files if not f.endswith('.py')})
    out = {'scope': ('every SRP1 run in W187 -- the screen cells (Step-7 launcher -> harness -> production) and gate '
                     '(ii) (the committed gate chain -> production): (a) the 38 production modules, (b) the Step-7 '
                     "launcher's RUN_FILES, (c) gate (ii)'s EXTRA_CLEAN_FILES (.py), (d) the SRP1 case file SRP1.json, "
                     'the planning params SRP1_params.json, the ESS params and cost file and the four network params files (and gate (ii)\'s data clean files); '
                     'searched: (1) a fixed-string search of every git-tracked file for each basename, (2) every file '
                     'of (a)-(c) for a call of ' + '/'.join(FS_LISTING_NAMES) + ' (AST), (3) the SRP1 case file\'s '
                     'file references'),
           'read_set_code': read_set_code, 'read_set_data': read_set_data, 'per_input': {}}
    fs = {f: _fs_listing_calls(f) for f in read_set_code if os.path.isfile(_abs(f))}
    out['production_modules_with_fs_listing_calls'] = {f: h for f, h in fs.items() if f in prod and h}
    out['non_production_read_set_files_with_fs_listing_calls'] = {f: h for f, h in fs.items() if f not in prod and h}
    sha_l, n_l = _listing_set_sha(out['non_production_read_set_files_with_fs_listing_calls'])
    out['non_production_listing_calls'] = {'n': n_l, 'sha256_of_sorted_file_line_call_text': sha_l,
                                           'inspected_set_sha256': INSPECTED_LISTING_CALLS['sha256'],
                                           'inspected_n': INSPECTED_LISTING_CALLS['n'],
                                           'verdict_of_the_inspection': INSPECTED_LISTING_CALLS['verdict']}
    _need(sha_l == INSPECTED_LISTING_CALLS['sha256'] and n_l == INSPECTED_LISTING_CALLS['n'],
          f'the non-production directory-listing calls ({n_l}, {sha_l}) are not the inspected set', refusals)
    _need(not out['production_modules_with_fs_listing_calls'],
          f"production lists directories: {out['production_modules_with_fs_listing_calls']}", refusals)
    for rel, sha in P516_INPUTS.items():
        base = os.path.basename(rel)
        hits = sorted(p for p in _git(['grep', '-l', '-I', '-F', base]).splitlines() if p.strip()) \
            if subprocess.run(['git', 'grep', '-q', '-I', '-F', base], cwd=REPO).returncode == 0 else []
        in_code = sorted(set(hits) & set(read_set_code))
        in_data = sorted(set(hits) & set(read_set_data))
        out['per_input'][rel] = {'sha256': sha, 'sha256_now': _sha(rel), 'status': P516_STATUS_TEXT,
                                 'tracked_files_naming_it': hits, 'read_set_code_files_naming_it': in_code,
                                 'read_set_data_files_naming_it': in_data,
                                 'outside_the_read_set': not in_code and not in_data}
        _need(not in_code and not in_data, f'{rel} is named in the SRP1 read set: {in_code + in_data}', refusals)
    case = _load(V1F.SRP1_CASE_JSON)
    out['srp1_case_file'] = V1F.SRP1_CASE_JSON
    out['srp1_case_file_ess_params_file'] = (case.get('SharedEnergyStorage') or {}).get('params_file')
    with open(_abs(V1F.SRP1_CASE_JSON)) as handle:
        out['srp1_case_file_text_names'] = sorted(set(re.findall(r'"([A-Za-z0-9_.\-]+\.(?:json|xlsx))"',
                                                                 handle.read())))
    _need(not any(os.path.basename(r) in out['srp1_case_file_text_names'] for r in P516_INPUTS),
          'the SRP1 case file names a P5.16 input', refusals)
    return out


# ======================================================================================================================
#  gates (i) and (ii)
# ======================================================================================================================
def gate_i(refusals):
    out = {'commit': GATE_I['commit'], 'report': GATE_I['report'], 'commit_is_ancestor_of_HEAD':
           _is_ancestor(GATE_I['commit'])}
    _need(out['commit_is_ancestor_of_HEAD'], 'gate (i) commit is not an ancestor of HEAD', refusals)
    for k in ('compare_json', 'ast_json', 'digest'):
        got = _sha_or_none(GATE_I[k]['path'])
        out[k] = {'path': GATE_I[k]['path'], 'sha256': got, 'equals_pin': got == GATE_I[k]['sha256'],
                  'tracked_and_clean': _tracked_clean(GATE_I[k]['path'])}
        _need(out[k]['equals_pin'] and out[k]['tracked_and_clean'], f'gate (i) {k} not as pinned', refusals)
    cmp_ = _load(GATE_I['compare_json']['path'])
    astj = _load(GATE_I['ast_json']['path'])
    qa = astj.get(Q_AGGREGATION_FUNCTION) or {}
    out.update({'n_blocks_compared': cmp_.get('n_blocks_compared'), 'n_blocks_identical': cmp_.get('n_blocks_identical'),
                'expected_blocks': cmp_.get('expected_blocks'), 'verdict_all_identical': cmp_.get('verdict_all_identical'),
                'production_modules_differing': cmp_.get('production_modules_differing'),
                'inputs_differing': cmp_.get('inputs_differing'),
                'reference_expected_production_commit': cmp_.get('reference_expected_production_commit'),
                'post_expected_production_commit': cmp_.get('post_expected_production_commit'),
                'q_aggregation': {'function': Q_AGGREGATION_FUNCTION, **qa}})
    _need(out['n_blocks_compared'] == out['n_blocks_identical'] == out['expected_blocks'] == 696
          and out['verdict_all_identical'] is True, 'gate (i): not 696/696 identical', refusals)
    _need(qa.get('off_path_ast_equal_incl_signature') is True, 'gate (i): Q aggregation not AST-equal', refusals)
    _need(out['post_expected_production_commit'] == PRODUCTION_HEAD
          and out['reference_expected_production_commit'] == PRODUCTION_HEAD_V1, 'gate (i): production commits',
          refusals)
    _need(sorted(out['production_modules_differing'] or []) == sorted(F6C23977_MODULES)
          and sorted(out['inputs_differing'] or []) == sorted(P516_INPUTS), 'gate (i): differing sets', refusals)
    return out


def gate_ii(refusals, required):
    rec_path = GATE_II['record']
    out = dict(GATE_II)
    if not os.path.isfile(_abs(rec_path)):
        out['status'] = 'pending (not run)'
        _need(not required, f'gate (ii) record missing: {rec_path}', refusals)
        return out
    rec = _load(rec_path)
    out['status'] = 'ran'
    out['record_sha256'] = _sha(rec_path)
    out['record_tracked_and_clean'] = _tracked_clean(rec_path)
    out['manifests_sha256'] = {m: _sha_or_none(m) for m in GATE_II['manifests']}
    out['manifests_tracked_and_clean'] = {m: _tracked_clean(m) for m in GATE_II['manifests']}
    out['launch_log_sha256'] = _sha_or_none(GATE_II['launch_log'])
    out['launch_log_tracked_and_clean'] = _tracked_clean(GATE_II['launch_log'])
    oc = rec.get('outcome_vs_prediction') or {}
    out.update({'schema': rec.get('schema'), 'git_head_at_run': rec.get('git_head_at_run'),
                'GATE_PASS_as_frozen': rec.get('GATE_PASS'), 'GATE_PASS_RULED': rec.get('GATE_PASS_RULED'),
                'committed_gate_exit_status': rec.get('committed_gate_exit_status'),
                'guard_counts_at_end': rec.get('guard_counts_at_end'), 'outcome_vs_prediction': oc,
                'code_unchanged_during_run': rec.get('code_unchanged_during_run'),
                'drift_keys_vs_W48': (rec.get('vs_W48') or {}).get('drift_keys'),
                'per_file_gate_vs_W48': (rec.get('vs_W48') or {}).get('per_file_gate'),
                'per_file_gate_vs_W83': (rec.get('vs_W83') or {}).get('per_file_gate'),
                'provenance_fields_vs_W48': sorted({d.get('field') for d in (rec.get('vs_W48') or {}).get('every_diff', [])
                                                    if d.get('class') == 'provenance'}),
                'provenance_fields_vs_W83': sorted({d.get('field') for d in (rec.get('vs_W83') or {}).get('every_diff', [])
                                                    if d.get('class') == 'provenance'})})
    if required:
        _need(out['record_tracked_and_clean'] and all(out['manifests_tracked_and_clean'].values())
              and out['launch_log_tracked_and_clean'], 'gate (ii) record / manifests / log not committed', refusals)
    _need(rec.get('GATE_PASS_RULED') is True, 'gate (ii): GATE_PASS_RULED is not True', refusals)
    _need(rec.get('code_unchanged_during_run') is True, 'gate (ii): code changed during the run', refusals)
    _need(bool(rec.get('git_head_at_run')) and _is_ancestor(rec['git_head_at_run']), 'gate (ii) head', refusals)
    before = rec.get('code_sha256_before_run') or {}
    stale = sorted(f for f, s in before.items() if _sha_or_none(f) != s)
    out['gated_code_changed_since_the_run'] = stale
    _need(not stale, f'gate (ii): gated code changed since the run: {stale}', refusals)
    g = rec.get('guard_counts_at_end') or {}
    _need(g.get('blocked_solve') == 0 and g.get('blocked_exec') == 0 and (g.get('permitted_solve') or 0) >= 153,
          f'gate (ii) guard counts {g}', refusals)
    return out


def text_ruling4(refusals):
    path, first, last, m1, m2 = RULING4_TEXT
    with open(_abs(path), encoding='utf-8') as handle:
        lines = handle.read().split('\n')[first - 1:last]
    ok = bool(lines) and m1 in lines[0] and m2 in lines[-1]
    _need(ok, f'Addendum 90 ruling 4 markers not found at {path}:{first}-{last}', refusals)
    return {'source': path, 'source_sha256': _sha(path), 'lines': [first, last], 'text': '\n'.join(lines)}


# ======================================================================================================================
#  modes
# ======================================================================================================================
def build(check):
    started = time.time()
    refusals = []
    v1_sha = _sha_or_none(SPEC_V1['path'])
    _need(v1_sha == SPEC_V1['sha256'] and _tracked_clean(SPEC_V1['path']), f'spec v1 not as pinned: {v1_sha}', refusals)
    if refusals:
        return refusals, None, None
    v1 = _load(SPEC_V1['path'])
    _log(f'mode {"CHECK" if check else "FREEZE-SPEC"}; HEAD {_git(["rev-parse", "HEAD"]).strip()}')
    _log('launcher diff vs the spec v1 blob')
    ldiff = launcher_diff(refusals)
    _log(f"   hunks {ldiff.get('n_hunks')} removed {ldiff.get('n_removed')} added {ldiff.get('n_added')}; exactly the "
         f"two G0 constants {ldiff.get('exactly_the_two_g0_constants')}")
    _log('accepted state')
    st = accepted_state(refusals, v1, check)
    _log(f'   refusals so far {len(refusals)}')
    _log('Step-7 keys (the launcher\'s own dry-run code via spec v1\'s step7_keys) vs the committed dry run')
    s7, lattice, _lev = V1F.step7_keys(refusals)
    v1_keys = v1['keys']['step7']['phases']
    key_eq = {p: [r['eval_key'] for r in s7['phases'][p]['cells']] == [r['eval_key'] for r in v1_keys[p]['cells']]
              for p in ('e3', 'e2', 'screen')}
    s7['equal_to_spec_v1_key_tables'] = key_eq
    _need(all(key_eq.values()), f'step-7 eval keys differ from spec v1: {key_eq}', refusals)
    counts = {p: (s7['phases'][p]['n_distinct_eval_keys'],
                  sum(c['equal_to_committed_dry_run'] for c in s7['phases'][p]['comparison_with_committed_dry_run']))
              for p in ('e3', 'e2', 'screen')}
    _log(f"   (distinct keys, rows equal to the dry run) {counts}; alias {s7['screen_alias']['alias_eval_key'][:8]} "
         f"reuse_of {s7['screen_alias']['reuse_of'][:8]} equal {s7['screen_alias']['equal_to_committed_dry_run']}")
    _log('G0 (require_current_production=True)')
    g0 = g0_now(refusals, lattice)
    _log(f"   accepted {g0['accepted']}: {g0['n_true']}/{g0['n_items']} items")
    _log('screen parent capture checklist (phase-level)')
    scr = {}
    for cdef in L7.cell_table('screen', lattice)[:2]:
        ok, cap = L7.parent_capture_checklist(cdef, None)
        scr[cdef['cell']] = {'ok': ok, 'checklist': cap}
        _need(ok, f'screen {cdef["cell"]}: parent capture checklist fails', refusals)
    _log(f"   {[(c, v['ok']) for c, v in scr.items()]}")
    _log('gate (i) citation')
    gi = gate_i(refusals)
    _log(f"   {gi['n_blocks_identical']}/{gi['n_blocks_compared']} identical; Q aggregation AST-equal "
         f"{gi['q_aggregation'].get('off_path_ast_equal_incl_signature')}")
    _log('gate (ii) record')
    gii = gate_ii(refusals, required=not check)
    _log(f"   status {gii['status']} GATE_PASS_RULED {gii.get('GATE_PASS_RULED')}")
    gate_clean = (_load(GATE_II['record'])['declared_substitutions']['2_extra_clean_files']
                  if gii['status'] == 'ran' else
                  _load(os.path.join(P55, 'w183i_srp1_bitwise_gate', 'w183i_wrapper_record.json'))[
                      'declared_substitutions']['2_extra_clean_files'] + sorted(F6C23977_MODULES)
                  + [LAUNCHER_REL, GATE_II_WRAPPER])
    _log('the two P5.16 inputs: static search')
    p516 = p516_inputs(refusals, gate_clean)
    _log(f"   outside the read set: { {k: v['outside_the_read_set'] for k, v in p516['per_input'].items()} }")
    r4 = text_ruling4(refusals)

    content = copy.deepcopy(v1)
    changes = {}

    def put(path, new):
        node = content
        for k in path[:-1]:
            node = node[k]
        old = node.get(path[-1], '<absent in v1>')
        node[path[-1]] = new
        changes['.'.join(path)] = {'v1': old, 'v1_1': new}

    put(('schema',), SCHEMA)
    put(('version',), VERSION)
    put(('stage',), STAGE)
    content['authority_v1'] = v1['authority']
    put(('authority',), AUTHORITY)
    put(('script',), SCRIPT_NAME)
    put(('script_sha256',), _sha(SCRIPT_NAME))
    put(('predecessor',), {'path': SPEC_V1['path'], 'sha256': SPEC_V1['sha256'], 'version': 1})
    put(('predecessor_statement',), (
        'v1.1 succeeds spec v1 (frozen_step7_campaign_spec_v1_96345241.json, sha256 ' + SPEC_V1['sha256'] + ', '
        'production 6f43479e), under which W183i, W184 (E3) and W185 (E2) ran on 2026-10-26 (Addendum 90 correction '
        '1). v1 is not replaced; v1.1 re-pins it on production f6c23977 for W187 (the screen) only.'))
    put(('interpreter',), sys.executable)
    put(('argv',), sys.argv)
    put(('smoke',), False)
    put(('started_utc',), _utc())
    put(('pins', 'production'), st['production'])
    put(('pins', 'files'), st['files'])
    put(('pins', 'step7_launcher_run_files_sha256'), st['step7_launcher_run_files_sha256'])
    put(('pins', 'step7_launcher_pins_now'), st['step7_launcher_pins_now'])
    put(('pins', 'step7_pins_recorded_check'), st['step7_pins_recorded_check'])
    put(('pins', 'srp1_inputs'), dict(st['srp1_inputs'], p516_inputs={
        r: {'sha256': p516['per_input'][r]['sha256'], 'status': P516_STATUS_TEXT} for r in P516_INPUTS}))
    put(('pins', 'committed_dry_run_outputs'), st['committed_dry_run_outputs'])
    put(('pins', 'git_head'), st['git_head'])
    put(('pins', 'two_cycle_gate_ruled_record'), {
        'path': GATE_II['record'], 'sha256': gii.get('record_sha256'), 'schema': gii.get('schema'),
        'GATE_PASS_as_frozen': gii.get('GATE_PASS_as_frozen'), 'GATE_PASS_RULED': gii.get('GATE_PASS_RULED'),
        'status': gii['status']})
    put(('keys', 'step7'), s7)
    put(('capture_checklists', 'launchers', 'step7', 'screen'), scr)
    content['spec_v1_1'] = {
        'production_head_repin': {'v1': PRODUCTION_HEAD_V1, 'v1_1': PRODUCTION_HEAD,
                                  'modules_differing': st['production_modules_differing_from_v1']},
        'launcher': {'path': LAUNCHER_REL, 'v1_sha256': LAUNCHER_V1_SHA256, 'v1_1_sha256': _sha(LAUNCHER_REL),
                     'diff_vs_v1': ldiff, 'run_files_differing_from_v1': st['step7_launcher_run_files_differing_from_v1']},
        'g0_repin': {'old': G0_OLD, 'new': G0_NEW, 'w183d_record_pin_unchanged': G0_RECORD_PIN,
                     'check_logic_unchanged': 'only the two constants changed (diff_vs_v1)', 'g0_check_now': g0,
                     'ruling': 'Planner option (b), within Addendum 90 ruling 4 re-pin authority'},
        'p516_inputs': dict(p516, ruling=('Planner (within Addendum 90 ruling 4): the two P5.16 inputs stay PINNED by '
                                          "the launcher's pins_now (stricter than excluded); no exclusion-list code")),
        'gate_i_pin_check': gi, 'gate_ii_two_cycle': gii, 'text_addendum90_ruling4': r4,
        'e1_launcher_sha256_now': st['e1_launcher_sha256_now'],
        'committed_records_check': st['committed_records'],
        'launch_scope': ('W187 = the 15-key 1x screen (phase screen, root P515S55/e1_screen_1x), frozen after this spec '
                         'by the Step-7 launcher; launch by 2026-11-20, 21 h serial, attached, one task (Addendum 90 '
                         'ruling 4)'),
    }
    content['carried_from_v1_unchanged'] = {
        'e1_launcher_sections': ('pins.e1_launcher_pins_now_refreshed_at_spec_v1, pins.e1_pins_recorded_check, '
                                 'pins.e1_launcher_run_files_sha256, keys.e1, capture_checklists.launchers.e1: v1\'s '
                                 '(production 6f43479e). E1 is outside W187; an E1 launch needs its own re-pin.'),
        'frozen_roots_e3_e2': 'run under spec v1 on 2026-10-26 (W184, W185); not re-run (Addendum 90 correction 1)',
        'declarations_text_hypotheses': 'v1\'s, verbatim line ranges of the brief at 8b83aeb5 and the plan at 18593428',
        'spec_required_quantities_addendum83': 'v1\'s capture map',
        'launch_commands': 'v1\'s E3 / E2 commands (already run); the screen commands are in its stage spec'}
    g = V1F.guards_report()
    put(('guards_before_write',), g)
    _need(g['all_zero'], 'guards not all zero before the write', refusals)
    content['changes_from_v1_paths'] = sorted(changes)
    content['changes_from_v1'] = {k: v for k, v in changes.items()
                                  if k not in ('guards_before_write', 'keys.step7', 'pins.step7_launcher_pins_now',
                                               'pins.production', 'pins.files', 'pins.srp1_inputs',
                                               'capture_checklists.launchers.step7.screen')}
    content['finished_utc'] = _utc()
    content['wall_clock_s_before_write'] = time.time() - started
    summary = {'launcher_diff': ldiff, 'keys': counts, 'g0': {'accepted': g0['accepted'], 'n_true': g0['n_true'],
                                                                'n_items': g0['n_items']},
               'gate_i': {k: gi[k] for k in ('n_blocks_identical', 'n_blocks_compared')},
               'gate_ii': gii['status'], 'p516': {k: v['outside_the_read_set'] for k, v in p516['per_input'].items()}}
    return refusals, content, summary


def freeze_spec(_args):
    refusals = []
    _need(_tracked_clean(SCRIPT_NAME), f'{SCRIPT_NAME} is not committed and clean', refusals)
    existing = [f for f in os.listdir(_abs(P55)) if SPEC_DOC_RX.match(f)]
    _need(not existing, f'spec v1.1 already exists (write-once): {existing}', refusals)
    man_dir = _abs(OUT_DIR_REL)
    _need(os.path.isdir(man_dir) and all(L7.LOG_RX.match(f) for f in os.listdir(man_dir)),
          f'{OUT_DIR_REL} must exist and hold only launch logs', refusals)
    if refusals:
        for r in refusals:
            _log(f'[REFUSED] {r}')
        return 1
    refusals, content, summary = build(check=False)
    if refusals:
        for r in refusals:
            _log(f'[REFUSED] {r}')
        _log(f'REFUSED: {len(refusals)} refusal(s); nothing written')
        return 1
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    doc_path = os.path.join(_abs(P55), SPEC_DOC_FMT.format(sha[:8]))
    with open(doc_path, 'x') as handle:
        handle.write(text)
    if H.sha256_file(doc_path) != sha:
        raise RuntimeError('the spec document does not hash to its name')
    _log(f'wrote {_rel(doc_path)} sha256 {sha}')
    files = [GATE_II['record'], GATE_II['launch_log']] + GATE_II['manifests'] + [
        LAUNCHER_REL, GATE_II_WRAPPER, SCRIPT_NAME, G0_NEW['path'], SPEC_V1['path']]
    manifest = {'schema': 'p515_s55_w187_spec_v1_1_manifest_v1', 'spec_v1_1': {'path': _rel(doc_path), 'sha256': sha},
                'sha256': {f: _sha(f) for f in files}}
    g2 = V1F.guards_report()
    manifest['guards_at_end'] = {'all_zero': g2['all_zero'], 'n_guards_verified': g2['n_guards_verified']}
    man_path = os.path.join(man_dir, MANIFEST_FILE)
    with open(man_path, 'x') as handle:
        handle.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')
    _log(f"wrote {_rel(man_path)} sha256 {H.sha256_file(man_path)}; guards all_zero {g2['all_zero']}")
    _log(f'summary {json.dumps(summary, default=str)[:2000]}')
    return 0 if g2['all_zero'] else 1


def check(_args):
    refusals, content, summary = build(check=True)
    if content is not None:   # the freeze's serialisation, exercised (nothing written)
        try:
            text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
            _log(f'serialisation through gate_result_io: OK, {len(text)} bytes (not written); fs-listing calls in '
                 f"non-production read-set files (informative): "
                 f"{content['spec_v1_1']['p516_inputs']['non_production_read_set_files_with_fs_listing_calls']}")
        except Exception as error:  # noqa: BLE001
            refusals.append(f'serialisation through gate_result_io fails: {type(error).__name__}: {error}')
    for r in refusals:
        _log(f'[REFUSED] {r}')
    if summary:
        _log(f"launcher unified diff (n=0):\n{summary['launcher_diff'].get('unified_diff_n0')}")
        _log(f'summary {json.dumps({k: v for k, v in summary.items() if k != "launcher_diff"}, default=str)}')
    _log(f'CHECK {"PASS" if not refusals else "FAIL"}: {len(refusals)} refusal(s); nothing written')
    return 0 if not refusals else 1


def post_manifest(_args):
    man_dir = _abs(OUT_DIR_REL)
    path = os.path.join(man_dir, POST_MANIFEST_FILE)
    if os.path.exists(path):
        raise SystemExit(f'REFUSED: {path} exists (write-once)')
    docs = [f for f in os.listdir(_abs(P55)) if SPEC_DOC_RX.match(f)]
    if len(docs) != 1:
        raise SystemExit(f'REFUSED: expected exactly one spec v1.1 document, found {docs}')
    files = [os.path.join(P55, docs[0])]
    for directory, _dirs, fnames in os.walk(_abs(L7.PHASE_ROOTS['screen'])):
        files += [_rel(os.path.join(directory, f)) for f in sorted(fnames)]
    files += [GATE_II['record'], GATE_II['launch_log']] + GATE_II['manifests']
    files += [os.path.join(OUT_DIR_REL, f) for f in sorted(os.listdir(man_dir))]
    files += [LAUNCHER_REL, GATE_II_WRAPPER, SCRIPT_NAME]
    man = {'schema': 'p515_s55_w187_post_preconditions_manifest_v1', 'utc': _utc(),
           'git_head': _git(['rev-parse', 'HEAD']).strip(), 'sha256': {f: _sha(f) for f in files},
           'git_tracked_and_clean': {f: _tracked_clean(f) for f in files}}
    g = V1F.guards_report()
    man['guards'] = {'all_zero': g['all_zero'], 'n_guards_verified': g['n_guards_verified']}
    with open(path, 'x') as handle:
        handle.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f'wrote {_rel(path)} sha256 {H.sha256_file(path)} ({len(files)} files); guards all_zero {g["all_zero"]}')
    return 0 if g['all_zero'] else 1


def main():
    parser = argparse.ArgumentParser(description='W187 Step-7 campaign spec v1.1 freeze (zero solves)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--post-manifest', action='store_true')
    args = parser.parse_args()
    try:
        code = (check(args) if args.check else post_manifest(args) if args.post_manifest else freeze_spec(args))
    except V1F.Refusal as error:
        _log(f'REFUSED: {error}')
        code = 1
    g = V1F.guards_report()
    _log(f"final guards: all_zero={g['all_zero']} n={g['n_guards_verified']} pickle={g['pickle']} "
         f"build={g['model_build_tripwires']} e1_tripwires_nonzero="
         f"{ {k: v for k, v in g['e1_launcher_committed_function_tripwires'].items() if v} }")
    sys.exit(code if g['all_zero'] else 1)


if __name__ == '__main__':
    main()
