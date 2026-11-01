"""
P5.15 Addendum 27 item 5(a), task W10 -- the SRP1 TWO-CYCLE BITWISE GATE for the
switchable pristine snapshot clones: an `'off'` run against an `'on'` run.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27 item 5(a) ("pristine snapshot
clones switchable, off for campaign and verification runs"); Planner task W10
("THIS TASK SOLVES"); W9 (commit b1272593) which added the `'off'` mode, the
validation helper `shared_resources_planning._validate_snapshot_capture_modes`,
`_build_pristine_snapshot_bases`, the two `'off'` dispatch branches and
`p515_s44_scale_measurement.{apply_snapshot_setting,snapshot_hook_wrapper}`.

WHAT W9 ESTABLISHED, AND WHAT IT DID NOT
----------------------------------------
W9's zero-solve checks showed that with mode `'off'` the pre-solve block state is
identical to `'lightweight'` (48/48 block digests) and that the dispatch reaches
`BlockData.clone` / `capture_block_mutable_state` zero times -- on ONE stubbed
dispatch, with a canned `run_smopf`. It said NOTHING about a real trajectory. This
gate runs the real thing.

THE TWO ARMS (both in THIS process, sequentially, never concurrently)
---------------------------------------------------------------------
Candidate C* -- 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025
(built from `p514_n_instrumented_cstar.S_INV/E_INV/INVEST_YEAR`, canonicalized by
`p515_s44_campaign_harness.canonical_candidate`; its candidate key is asserted
against the programme's pinned C* key, so the problem instance is recorded, not
implied).

Both arms call `p515_g_g1_g4_admm_gates.run_admm_arm` BY IMPORT with cap 2, arm
label `s39_D`, `apply_rho=False`, `full_diagnostics_in_rows=True` -- the campaign
child's own call (`p515_s44_campaign_harness._child_real`) -- with the SAME
`pre_solve_hook`: `_config_hook_factory(spec_like, holder, overrides={})` where
`spec_like.configuration.case_file_anderson_acceleration` is the case-file AA
declaration {enabled: True, memory: 5, regularization: 1e-10, reject_policy:
'keep_memory'}, exactly as `p515_s45_a0_campaign.py` freezes it. The configuration
is therefore VERIFIED by the campaign harness's own hook (rho/tau/gamma policy,
exempt-until, freeze, AA == the declaration with the frozen memory/regularization,
persistent workers off, parallel execution off, num_max_iters == cap), not asserted
here.

The ONLY difference between the arms is the argument handed to
`p515_s44_scale_measurement.snapshot_hook_wrapper`, which wraps that same hook:
  ARM ON   `'on'`  -- `apply_snapshot_setting` performs NO assignment at all (read
                      its source: the 'off' branch is the only one that assigns);
                      the modes stay at the `ADMMParameters` defaults
                      ('lightweight'/'lightweight'). It still RECORDS the modes in
                      force, so "untouched" is evidenced rather than assumed.
  ARM OFF  `'off'` -- both `tso_snapshot_capture_mode` and
                      `dso_snapshot_capture_mode` are set to 'off' and the setting
                      is VERIFIED by production-side read-back (the helper raises
                      if it did not take effect); the applied record is kept.

INSTRUMENTATION (pass-through counters; never stubs, never behaviour changes)
-----------------------------------------------------------------------------
`pyomo.core.base.block.BlockData.clone` and
`shared_resources_planning.capture_block_mutable_state` (the module attribute both
production dispatch sites resolve at call time) are wrapped for the WHOLE run with
counters that call straight through and additionally record the immediate caller
site (file:line). The counts are bucketed by phase, so nothing escapes attribution:
'preflight', 'arm_on', 'arm_off', 'post'.

THE GATE (every item must hold)
-------------------------------
 1. Both arms complete with cycles_run == 2.
 2. Clone/capture: arm ON has clones > 0 AND captures > 0; arm OFF has EXACTLY 0
    and EXACTLY 0.
 3. The snapshot setting took effect and is recorded for both arms; OFF reads
    ('off', 'off'), ON reads ('lightweight', 'lightweight').
 4. Zero GENUINE diffs between the arms over every compared artifact -- the ADMM
    report `g_s39_D.json`, `CP.ARTIFACT_FILES`, `CP.SIDECAR_JSONL_FILES`,
    `aa_per_cycle.jsonl` and `per_cycle_record.jsonl` -- where a diff is NOT
    genuine only if it falls in one of these THREE declared-in-advance buckets:
      (a) the `rule_eleven_checklist` subtree (`FG._classify_diffs`): how the run
          was configured/verified, incl. the recorded snapshot modes;
      (b) an arm-path difference: both values are strings that become EQUAL after
          substituting each arm's own output directory and working-dir ids (the
          substitution must actually change something);
      (c) the dotted suffixes in `PERMITTED_PROVENANCE_SUFFIXES` -- today exactly
          `network_failures_summary.n_frozen_snapshots`, which the Planner's task
          text names as permitted, plus the `frozen_snapshots_*.jsonl` inventory
          which is reported separately and is not in the gating set.
    Anything else -- including `FG`'s `aa_new_field` and alias buckets, which are
    NOT trusted here (both arms run the same code with AA on) -- is genuine and
    FAILS. Diffs inside the recourse-jump sidecar's two top-k lists are passed
    through `p515_s44_tie_classifier.reclassify_sidecar_diffs` before they are
    called genuine.
 5. Every field of `p515_s44_campaign_harness.RECORD_TRAJECTORY_FIELDS` is
    bitwise equal, cycle by cycle, between the arms (an explicit table on top of
    the whole-report diff, which already covers every field).
 6. Arm OFF wrote ZERO `.pkl` under its own `results/FrozenSMOPF/`.
 7. Solve profile (below).
 8. The armed `SolveProfileGuard` reports no blocked call and verifies EXACTLY.

SOLVE PROFILE -- DECLARED BEFORE THE RUN, GATED ON THE RECONCILIATION IDENTITY
------------------------------------------------------------------------------
Derived from the instance BEFORE anything runs, from `data/SRP1/SRP1.json`:
3 years x 4 days = 12 (year, day) blocks; 1 TSO + 3 DSOs = 4 networks -> 48 network
solves per cycle; 3 active distribution nodes -> 3 ESSO solves per cycle; 51 solves
per cycle. `run_admm_arm`'s own identity counts initialization as one more such
round, so the BASE for cap 2 is 51 * (2 + 1) = 153 per arm, 306 for the two arms.
The same number is RE-DERIVED inside the pre-solve hook from the planning object
(`p515_s44_scale_measurement.expected_block_counts`) and must agree.

A local-solve recovery adds solves: +1 per tier-1 recovered block, +2 per tier-2.
An exact number therefore cannot be declared in advance for a run whose failures
are not known in advance, so -- as the task permits -- the BASE is declared and the
gate is on the RECONCILIATION IDENTITY:

    observed_arm == 153 + 1 * recovered_tier1 + 2 * recovered_tier2   (per arm)

with the recovery counts read from that arm's OWN `network_failures_summary`. The
process-wide `SolveProfileGuard(p514_n_instrumented_cstar.PERMITTED)` is armed for
the whole run (any solve from an undeclared call site raises immediately) and is
then `verify()`-ed EXACTLY against the reconciled total -- too few fails as loudly
as too many.

NON-GATING, REPORTED: ARM ON vs THE COMMITTED AA-ON C* TRAJECTORY
------------------------------------------------------------------
`data/SRP1/Results/P515S45/reverify_aa_c_star/evals/3e741dac72c9e1bc_c_star_aa_case_file/g_s39_D.json`
`cycle_trajectory[:2]` plus the rows-derived top-level fields
(`p515_s40_polish_gap._derive_top_level_from_rows`, BY IMPORT). That run used cap
500; this one uses cap 2, and cap-independence with AA on is UNVERIFIED, so this
comparison is REPORTED ONLY. If it differs, the first differing field is stated
plainly.

PRECONDITIONS (checked before anything is written; refuses loudly otherwise)
----------------------------------------------------------------------------
  1. Neither `.p515_g_gate.lock` nor `.p515_s44_campaign.lock` exists.
  2. No other live process matches the forbidden substrings (this process and its
     ancestors excluded, `CP._ancestor_pids`).
  3. The output root does not exist (write-once; `--suffix` redirects it).
  4. The production files this task depends on are clean in git.
  5. The production loader reads the case file's AA dict == the declaration.
  6. The committed AA-on reference report exists and matches its pinned sha256.
  7. A capture-path checklist (rule eleven) asserts, BEFORE the run, that every
     quantity this gate reports has a capture path and every function it calls
     resolves -- including `H.assert_record_capture_paths()`.
Then this harness takes the legacy run lock via `G._acquire_exclusive_run_lock()`.

EXACT LAUNCH COMMAND (repo root; attached, alone, both streams captured)
------------------------------------------------------------------------
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s45_snapshot_off_two_cycle_gate.py \\
        > data/SRP1/Results/P515S45/snapshot_off_two_cycle_gate_launch.log 2>&1

OUTPUT (all NEW; refuses to overwrite)
    data/SRP1/Results/P515S45/snapshot_off_two_cycle_gate/gate.json
    data/SRP1/Results/P515S45/snapshot_off_two_cycle_gate/manifest_sha256.json
    data/SRP1/Results/P515S45/snapshot_off_two_cycle_gate/{on,off}/   (arm artifacts)
Exit 0 on PASS, 1 on FAIL or a precondition refusal.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

# `p514_n_instrumented_cstar` is imported first ONLY for its `PERMITTED` call-site
# list and the C* constants; importing it (like every other import below) defines
# names and cannot solve. The guard is installed immediately afterwards, before any
# harness call, and stays armed for the whole run.
import p514_n_instrumented_cstar as N  # noqa: E402

GUARD = SolveProfileGuard(
    N.PERMITTED, label='P5.15 Addendum 27 W10 snapshot off/on two-cycle gate').install()

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- comparator conventions, BY IMPORT
import p515_s40_polish_gap as PG  # noqa: E402 -- ARM_LABEL, _build_floor_rows, _derive_top_level_from_rows
import p515_s43_aa_flagoff_gate as FG  # noqa: E402 -- diff classification + returned-state capture
import p515_s43_aa_run as S43  # noqa: E402 -- the AA per-cycle sidecar builder
import p515_s44_campaign_harness as H  # noqa: E402 -- _config_hook_factory, trajectory field list
import p515_s44_scale_measurement as S44  # noqa: E402 -- the W9 snapshot switch under test
import p515_s44_tie_classifier as TC  # noqa: E402 -- top-k tie order, BY IMPORT
import shared_resources_planning as srp  # noqa: E402
from pyomo.core.base.block import BlockData  # noqa: E402

STAGE = ('P5.15 Addendum 27 item 5(a), W10 -- SRP1 two-cycle bitwise gate: pristine '
         'snapshot clones off vs on')
SCHEMA = 'p515_s45_snapshot_off_two_cycle_gate_v1'
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 27 item 5(a)',
    'Planner task W10 (two-cycle bitwise gate, snapshots off vs on; THIS TASK SOLVES)',
    'commit b1272593 (W9: switchable pristine snapshot clones, default unchanged)',
    'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json (configuration.oracle)',
]

ARM_LABEL = PG.ARM_LABEL            # 's39_D' -- BY IMPORT, so CP's file names apply
CAP = 2
REQUIRED_CONSECUTIVE_CYCLES = 10    # case-file value; the campaign's own spec field
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10,
                'reject_policy': 'keep_memory'}
# C* from the programme's own constants (p514_n_instrumented_cstar), not re-typed.
C_STAR = {node: (N.S_INV, N.E_INV) for node in H.ACTIVE_NODES}
C_STAR_KEY_PIN = '578636daa6d6360d6701764c73ddf795e53c2c37e511e21be1400024f8f6350c'

_P45 = os.path.join('data', 'SRP1', 'Results', 'P515S45')
AA_REFERENCE = {
    'report': os.path.join(_P45, 'reverify_aa_c_star', 'evals',
                           '3e741dac72c9e1bc_c_star_aa_case_file', 'g_s39_D.json'),
    'sha256': '7959be12a16a8c8be46fa6209ab581036ec177c9c21a2a47f4e7d9e67ba558be',
    'campaign': 's45_reverify_aa_c_star (spec 4a214b99), cap 500, certified at cycle 107, '
                'gross_operational_cost 650982939.9389359',
}

CAMPAIGN_ID = 's45_w10_snapoff'

# Instance-derived solve profile, computed BEFORE the run from the case file itself
# (see the module docstring); re-derived from the planning object in the pre-solve hook.
CASE_JSON = os.path.join(REPO, 'data', 'SRP1', 'SRP1.json')

FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = tuple(CP.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS) + (
    'p515_s40_', 'p515_s41_', 'p515_s42_', 'p515_s43_', 'p515_s44_', 'p515_s45_')
PRODUCTION_FILES_TO_CHECK_CLEAN = tuple(CP.PRODUCTION_FILES_TO_CHECK_CLEAN) + (
    'admm_anderson_acceleration.py', 'admm_persistent_workers.py',
    'p515_s44_campaign_harness.py', 'p515_s44_scale_measurement.py')

# Files compared BITWISE between the arms (gating).
GATING_JSON = tuple(CP.ARTIFACT_FILES)
GATING_JSONL = tuple(CP.SIDECAR_JSONL_FILES) + ('aa_per_cycle.jsonl', 'per_cycle_record.jsonl')
RECOURSE_JUMP = 'recourse_jump_sidecar_baseline.jsonl'
# Reported, NOT gating: these inventories carry absolute snapshot/log paths and mtimes,
# and the frozen-snapshot one is exactly what the mode under test changes.
SUPPLEMENTARY_JSONL = (f'leak_classification_{ARM_LABEL}.jsonl',
                       f'network_failures_{ARM_LABEL}.jsonl',
                       f'esso_recovery_events_{ARM_LABEL}.jsonl',
                       f'frozen_snapshots_{ARM_LABEL}.jsonl')
SUPPLEMENTARY_PATH_LIKE_KEYS = ('log_path', 'primary_log', 'recovery_log', 'tier2_log',
                                'path', 'mtime_utc')
NOT_COMPARED = {
    f'esso_models_{ARM_LABEL}.pkl': 'binary pickle; its size is compared via the report field '
                                    'esso_models_pickle.bytes; sha256 recorded (informational)',
    'esso_capture/, results/, logs': 'per-arm working files and logs (absolute paths by construction); '
                                     'hash-recorded in manifest_sha256.json',
}
# Declared IN ADVANCE: dotted suffixes whose difference is provenance, with the reason.
PERMITTED_PROVENANCE_SUFFIXES = {
    'network_failures_summary.n_frozen_snapshots':
        'the count of FrozenSMOPF snapshots written; switching capture off is exactly what this '
        'counts (Planner task W10 names it as permitted provenance)',
}
MAX_LISTED_GENUINE_PER_FILE = 200


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W10] {msg}', flush=True)


def _rel(path):
    return os.path.relpath(path, REPO)


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True).stdout.strip()


# ===========================================================================
#  instance-derived solve profile (BEFORE the run)
# ===========================================================================
def derive_solves_from_case_file():
    with open(CASE_JSON) as handle:
        case = json.load(handle)
    n_years = len(case['Years'])
    n_days = len(case['Days'])
    n_dso = len(case['DistributionNetworks'])
    active_nodes = sorted(dn['connection_node_id'] for dn in case['DistributionNetworks'])
    n_yd = n_years * n_days
    per_cycle = (1 + n_dso) * n_yd + len(active_nodes)
    return {
        'case_file': _rel(CASE_JSON), 'years': sorted(case['Years']), 'days': sorted(case['Days']),
        'n_year_day_blocks': n_yd, 'n_networks': 1 + n_dso, 'network_solves_per_cycle': (1 + n_dso) * n_yd,
        'active_distribution_network_nodes': active_nodes, 'esso_solves_per_cycle': len(active_nodes),
        'solves_per_cycle': per_cycle, 'cap': CAP,
        'declared_base_solves_per_arm': per_cycle * (CAP + 1),
        'declared_base_solves_both_arms': 2 * per_cycle * (CAP + 1),
        'identity': ('observed_arm == base + 1 * recovered_tier1 + 2 * recovered_tier2 '
                     '(recovery counts from the arm own network_failures_summary)'),
    }


# ===========================================================================
#  pass-through counters: BlockData.clone and capture_block_mutable_state
# ===========================================================================
class CloneCaptureCounters:
    """Counts the REAL `BlockData.clone` and production's own
    `shared_resources_planning.capture_block_mutable_state`, bucketed by phase, and
    records each call's immediate caller site. Every call is passed straight
    through -- never a stub, never a behaviour change (the same technique
    `p515_s45_snapshot_switch_checks.py` used, extended with site attribution and
    phase buckets so no call anywhere in the process escapes attribution)."""

    def __init__(self):
        self.phase = 'preflight'
        self.buckets = {}
        self._orig_clone = None
        self._orig_capture = None

    def _bucket(self):
        return self.buckets.setdefault(
            self.phase, {'clone': 0, 'capture': 0, 'clone_sites': {}, 'capture_sites': {}})

    @staticmethod
    def _site():
        frame = sys._getframe(2)
        return f'{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno} in {frame.f_code.co_name}'

    def _count(self, kind):
        bucket = self._bucket()
        bucket[kind] += 1
        site = self._site()
        sites = bucket[f'{kind}_sites']
        sites[site] = sites.get(site, 0) + 1

    @contextmanager
    def installed(self):
        self._orig_clone = BlockData.clone
        self._orig_capture = srp.capture_block_mutable_state
        counters = self

        def counting_clone(block_self, *args, **kwargs):
            counters._count('clone')
            return counters._orig_clone(block_self, *args, **kwargs)

        def counting_capture(model):
            counters._count('capture')
            return counters._orig_capture(model)

        BlockData.clone = counting_clone
        srp.capture_block_mutable_state = counting_capture
        try:
            yield self
        finally:
            BlockData.clone = self._orig_clone
            srp.capture_block_mutable_state = self._orig_capture

    @contextmanager
    def phase_of(self, phase):
        previous = self.phase
        self.phase = phase
        self._bucket()
        try:
            yield
        finally:
            self.phase = previous


# ===========================================================================
#  preconditions and the capture-path checklist (rule eleven)
# ===========================================================================
def check_preconditions(out_dir):
    failures = []
    for lock_path in (os.path.join(REPO, '.p515_g_gate.lock'),
                      os.path.join(REPO, '.p515_s44_campaign.lock')):
        if os.path.exists(lock_path):
            try:
                holder = open(lock_path).read().strip()
            except OSError:
                holder = 'unknown'
            failures.append(f'lock file already exists: {lock_path} ({holder})')

    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan the process table: {error}')
        ps_output = ''
    excluded_pids = {str(p) for p in CP._ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        pid = fields[1] if len(fields) > 1 else None
        if pid in excluded_pids:
            continue
        if any(s in line for s in FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')

    if os.path.exists(out_dir):
        failures.append(f'output directory already exists (write-once): {out_dir}')

    status = subprocess.run(['git', 'status', '--porcelain', '--'] + list(PRODUCTION_FILES_TO_CHECK_CLEAN),
                            cwd=REPO, capture_output=True, text=True).stdout
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')

    reference = os.path.join(REPO, AA_REFERENCE['report'])
    if not os.path.isfile(reference):
        failures.append(f'committed AA-on reference report missing: {AA_REFERENCE["report"]}')
    elif CP._sha256_file(reference) != AA_REFERENCE['sha256']:
        failures.append(f'committed AA-on reference report sha256 != pin: {AA_REFERENCE["report"]}')

    from planning_parameters import PlanningParameters
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    loaded_aa = params.admm.anderson_acceleration
    if loaded_aa != CASE_FILE_AA:
        failures.append(f'case file AA {loaded_aa} != declaration {CASE_FILE_AA}')

    canonical = H.canonical_candidate(C_STAR)
    key = H.candidate_key(canonical)
    if key != C_STAR_KEY_PIN:
        failures.append(f'candidate key {key} != pinned C* key {C_STAR_KEY_PIN}')

    return failures, {'case_file_anderson_acceleration_loaded': loaded_aa,
                      'candidate_canonical': canonical, 'candidate_key': key}


def capture_path_checklist():
    """Rule eleven: fails fast, BEFORE the run, if a quantity this gate reports has
    no capture path or a function it calls does not resolve with the signature used."""
    import inspect
    checks = {'record_capture_paths (campaign harness)': bool(H.assert_record_capture_paths())}
    for module, name in ((G, 'run_admm_arm'), (G, 'write_boyd_terminal_s35ref'),
                         (G, 's38_pf_capture_hooks'), (G, 's39_exempt_until_capture_hooks'),
                         (G, '_acquire_exclusive_run_lock'),
                         (H, '_config_hook_factory'), (H, 'canonical_candidate'),
                         (H, 'investment_map_from_canonical'), (H, 'eval_ids'),
                         (S44, 'apply_snapshot_setting'), (S44, 'snapshot_hook_wrapper'),
                         (S44, 'expected_block_counts'),
                         (S43, '_build_aa_per_cycle_sidecar'),
                         (PG, '_build_floor_rows'), (PG, '_derive_top_level_from_rows'),
                         (CP, '_diff'), (CP, '_load_json'), (CP, '_load_jsonl'), (CP, '_sha256_file'),
                         (FG, '_classify_diffs'), (FG, '_capture_returned_state'),
                         (TC, 'reclassify_sidecar_diffs'),
                         (srp, '_validate_snapshot_capture_modes'),
                         (srp, '_build_pristine_snapshot_bases')):
        checks[f'callable_{module.__name__}.{name}'] = callable(getattr(module, name, None))
    checks['production_SNAPSHOT_CAPTURE_MODES_has_off'] = 'off' in getattr(srp, 'SNAPSHOT_CAPTURE_MODES', ())
    checks['production_capture_block_mutable_state_is_module_attr'] = callable(
        getattr(srp, 'capture_block_mutable_state', None))
    checks['BlockData_clone_is_the_method_ConcreteModel_uses'] = True  # verified below
    import pyomo.environ as pe
    checks['BlockData_clone_is_the_method_ConcreteModel_uses'] = (
        type(pe.ConcreteModel()).clone is BlockData.clone)
    for field in ('EXCLUDE_KEY_NAMES', 'EXCLUDE_DOTTED_SUFFIXES', 'INTENTIONAL_DIFF_SUFFIXES',
                  'ARTIFACT_FILES', 'SIDECAR_JSONL_FILES'):
        checks[f'comparator_convention_{field}'] = bool(getattr(CP, field, None))
    checks['trajectory_field_list_RECORD_TRAJECTORY_FIELDS'] = bool(H.RECORD_TRAJECTORY_FIELDS)
    sig = inspect.signature(G.run_admm_arm).parameters
    for p in ('investment_map', 'num_max_iters_override', 'eval_id', 'apply_rho',
              'full_diagnostics_in_rows', 'post_run_hook', 'pre_solve_hook'):
        checks[f'run_admm_arm_parameter_{p}'] = p in sig
    checks['report_field_network_failures_summary_classes'] = (
        "'recovered_tier1'" in inspect.getsource(G.run_admm_arm))
    checks['report_field_solve_profile_observed'] = "'solve_profile'" in inspect.getsource(G.run_admm_arm)
    checks['production_state_peak_rss'] = "'peak_rss_ru_maxrss':" in inspect.getsource(srp)
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN: capture paths missing for this gate: {missing}')
    return checks


# ===========================================================================
#  one arm
# ===========================================================================
def run_arm(arm, snapshots, out_dir, counters, expected_block_counts_holder):
    """One `run_admm_arm` call, wired exactly as the campaign child wires it
    (`p515_s44_campaign_harness._child_real`), with the snapshot setting applied by
    `p515_s44_scale_measurement.snapshot_hook_wrapper`."""
    os.makedirs(out_dir, exist_ok=True)
    canonical = H.canonical_candidate(C_STAR)
    investment_map = H.investment_map_from_canonical(canonical)
    ids = H.eval_ids(f'{CAMPAIGN_ID}_{arm}', H.candidate_key(canonical))

    holder = {}
    child_record = {}
    spec_like = {
        'configuration': {'overrides': {}, 'arm_label': ARM_LABEL,
                          'case_file_anderson_acceleration': dict(CASE_FILE_AA)},
        'cap': CAP, 'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES,
    }
    inner_hook = H._config_hook_factory(spec_like, holder, overrides={})

    def configure_hook(planning, sed, candidate, report):
        inner_hook(planning=planning, sed=sed, candidate=candidate, report=report)
        # The solve profile, RE-DERIVED from the instance before any solve.
        counts = S44.expected_block_counts(planning)
        expected_block_counts_holder[arm] = counts
        report.setdefault('rule_eleven_checklist', {})['w10_expected_block_counts'] = counts
        report['rule_eleven_checklist']['w10_arm'] = arm

    hook = S44.snapshot_hook_wrapper(configure_hook, snapshots, child_record)

    paths = {
        'recourse_jump': os.path.join(out_dir, 'recourse_jump_sidecar_baseline.jsonl'),
        'ess_stride': os.path.join(out_dir, 'ess_entry_stride_baseline.jsonl'),
        'floor': os.path.join(out_dir, 'soh_floor_sidecar_baseline.jsonl'),
        'pf_stride': os.path.join(out_dir, f'pf_entry_stride_{ARM_LABEL}.jsonl'),
        'exempt': os.path.join(out_dir, f'ess_exempt_until_state_{ARM_LABEL}.jsonl'),
        'aa': os.path.join(out_dir, 'aa_per_cycle.jsonl'),
        'per_cycle': os.path.join(out_dir, 'per_cycle_record.jsonl'),
    }
    for path in paths.values():
        _refuse_overwrite(path)

    with counters.phase_of(f'{arm}_precheck'):
        _cc, floor_rows_by_node, _fc = PG._build_floor_rows(ids['precheck'])

    def post_run_hook(planning, sed, models, rows, report, out_dir, label, state=None):
        report['s34_recourse_jump_sidecar_path'] = _rel(paths['recourse_jump'])
        report['s34_ess_entry_stride_sidecar_path'] = _rel(paths['ess_stride'])
        report['s35ref_soh_floor_sidecar_path'] = _rel(paths['floor'])
        report['s38_pf_entry_stride_sidecar_path'] = _rel(paths['pf_stride'])
        report['s39_ess_exempt_until_state_sidecar_path'] = _rel(paths['exempt'])
        G.write_boyd_terminal_s35ref(planning, sed, models, rows, report, out_dir, label,
                                     floor_rows_by_node=floor_rows_by_node,
                                     floor_sidecar_path=paths['floor'])
        caps = sed.get_updated_capacities(models['esso'])
        holder['published_caps'] = {str(n): {str(y): v for y, v in per_year.items()}
                                    for n, per_year in caps.items()}
        holder['peak_rss_ru_maxrss_production'] = (state or {}).get('peak_rss_ru_maxrss')
        holder['peak_rss_platform_units'] = (state or {}).get('peak_rss_platform_units')
        S43._build_aa_per_cycle_sidecar(rows, paths['aa'])

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.time()
    with counters.phase_of(f'arm_{arm}'), \
            G.s38_pf_capture_hooks(paths['recourse_jump'], paths['ess_stride'], paths['floor'],
                                   paths['pf_stride'], floor_rows_by_node, stride=1), \
            G.s39_exempt_until_capture_hooks(paths['exempt']), \
            FG._capture_returned_state() as state_holder:
        report, report_path = G.run_admm_arm(
            ARM_LABEL, out_dir, k_override=None, investment_map=investment_map,
            num_max_iters_override=CAP, eval_id=ids['run'], apply_rho=False,
            full_diagnostics_in_rows=True, post_run_hook=post_run_hook, pre_solve_hook=hook)
    wall = time.time() - started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    rows = report.get('cycle_trajectory') or []
    with open(paths['per_cycle'], 'w') as handle:
        for row in rows:
            handle.write(json.dumps({k: row.get(k) for k in H.PER_CYCLE_RECORD_FIELDS},
                                    default=str) + '\n')

    frozen_dir = os.path.join(out_dir, 'results', 'FrozenSMOPF')
    frozen_inventory = sorted(os.listdir(frozen_dir)) if os.path.isdir(frozen_dir) else []

    summary = {
        'arm': arm, 'snapshots_requested': snapshots, 'out_dir': _rel(out_dir),
        'working_dir_ids': ids, 'report_path': _rel(report_path),
        'candidate_canonical': canonical, 'candidate_key': H.candidate_key(canonical),
        'cycles_run': report.get('cycles_run'),
        'recourse': report.get('recourse'),
        'gross_operational_cost': report.get('gross_operational_cost'),
        'terminal_objective_change_abs': report.get('terminal_objective_change_abs'),
        'terminal_objective_tolerance': report.get('terminal_objective_tolerance'),
        'rule_ten_terminal_step_over_threshold': report.get('rule_ten_terminal_step_over_threshold'),
        'local_solve_failures': report.get('local_solve_failures'),
        'network_failures_summary': report.get('network_failures_summary'),
        'solve_profile': report.get('solve_profile'),
        'snapshot_setting': child_record.get('snapshot_setting'),
        'configuration_checks': holder.get('configuration_checks'),
        'anderson_acceleration_effective': holder.get('anderson_acceleration_effective'),
        'expected_block_counts_from_planning': expected_block_counts_holder.get(arm),
        'frozen_smopf_dir': _rel(frozen_dir),
        'frozen_smopf_inventory': frozen_inventory,
        'frozen_smopf_n_pkl': sum(1 for f in frozen_inventory if f.endswith('.pkl')),
        'frozen_smopf_n_skipped_markers': sum(1 for f in frozen_inventory
                                              if f.startswith('snapshot_skipped_')),
        'wall_clock_s': {'measured_here': wall, 'reported_by_run_admm_arm': report.get('wall_clock_s')},
        'peak_rss': {
            'units': 'bytes on macOS/BSD (ru_maxrss)',
            'process_ru_maxrss_before_arm': rss_before,
            'process_ru_maxrss_after_arm': rss_after,
            'production_state_peak_rss_ru_maxrss': (state_holder.get('state') or {}).get('peak_rss_ru_maxrss'),
            'production_state_peak_rss_platform_units': (state_holder.get('state') or {}).get(
                'peak_rss_platform_units'),
            'solver_subprocesses_ru_maxrss_before': children_before,
            'solver_subprocesses_ru_maxrss_after': children_after,
            'semantics': ('ru_maxrss is a process high-water mark, so "after arm 2" is the max over both '
                          'arms; the production_state figure is taken by production at the end of '
                          'run_operational_planning and is per-arm'),
        },
        'published_caps': holder.get('published_caps'),
    }
    return report, summary


# ===========================================================================
#  diff classification
# ===========================================================================
def _arm_tokens(out_dirs, ids_by_arm):
    tokens = []
    for arm in ('on', 'off'):
        arm_tokens = [os.path.abspath(out_dirs[arm]), _rel(out_dirs[arm])]
        arm_tokens += list(ids_by_arm[arm].values())
        tokens.append(arm_tokens)
    return tokens


def _normalize(value, token_groups):
    if not isinstance(value, str):
        return value, False
    changed = False
    for group in token_groups:
        for index, token in enumerate(group):
            if token and token in value:
                value = value.replace(token, f'<ARM_TOKEN_{index}>')
                changed = True
    return value, changed


def _is_arm_path_diff(diff, token_groups):
    left, right = diff.get('legacy'), diff.get('lightweight')
    if not (isinstance(left, str) and isinstance(right, str)):
        return False
    norm_left, changed_left = _normalize(left, token_groups)
    norm_right, changed_right = _normalize(right, token_groups)
    return (changed_left or changed_right) and norm_left == norm_right


def _permitted_suffix(field):
    for suffix, reason in PERMITTED_PROVENANCE_SUFFIXES.items():
        if field.endswith(suffix):
            return reason
    return None


def classify(fname, a, b, token_groups):
    """`a` = arm ON (CP._diff names it 'legacy'), `b` = arm OFF ('lightweight')."""
    raw = CP._diff(a, b, fname)
    fg_prov, fg_aa_new, fg_alias, fg_genuine = FG._classify_diffs(raw)
    provenance = [dict(d, reason='rule_eleven_checklist subtree (how the run was configured/verified, '
                                 'including the recorded snapshot modes)') for d in fg_prov]
    # FG's aa_new_field and alias buckets are NOT trusted here: both arms run the same
    # code with AA on, so neither category can legitimately explain a difference.
    candidates = list(fg_aa_new) + list(fg_alias) + list(fg_genuine)
    genuine = []
    for d in candidates:
        field = str(d.get('field', ''))
        reason = _permitted_suffix(field)
        if reason is not None:
            provenance.append(dict(d, reason=reason))
            continue
        if _is_arm_path_diff(d, token_groups):
            provenance.append(dict(d, reason='arm path: the two values are equal after substituting each '
                                             "arm's own output directory / working-dir ids"))
            continue
        genuine.append(d)
    tie_order, row_classes = [], {}
    if fname == RECOURSE_JUMP and genuine:
        tie_order, genuine, row_classes = TC.reclassify_sidecar_diffs(genuine, a, b, fname)
    return {
        'n_raw_diffs': len(raw),
        'fg_buckets_raw_counts': {'provenance': len(fg_prov), 'aa_new_field': len(fg_aa_new),
                                  'known_tie_break_alias': len(fg_alias), 'genuine': len(fg_genuine)},
        'n': {'provenance': len(provenance), 'tie_order': len(tie_order), 'genuine': len(genuine)},
        'provenance_diffs': provenance,
        'tie_order_diffs': tie_order,
        'tie_order_row_classes': row_classes,
        'genuine_diffs_first': genuine[:MAX_LISTED_GENUINE_PER_FILE],
        'genuine_diffs_listed_truncated': len(genuine) > MAX_LISTED_GENUINE_PER_FILE,
    }


def _strip_keys(obj, keys):
    if isinstance(obj, dict):
        return {k: _strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip_keys(v, keys) for v in obj]
    return obj


def trajectory_field_table(rows_on, rows_off):
    """Explicit per-field, per-cycle bitwise check over the campaign harness's own
    trajectory field list (`H.RECORD_TRAJECTORY_FIELDS`, BY IMPORT), on top of the
    whole-report diff which already covers every field."""
    table = {}
    n = min(len(rows_on), len(rows_off))
    for field in H.RECORD_TRAJECTORY_FIELDS:
        mismatches = []
        for i in range(n):
            left, right = rows_on[i].get(field), rows_off[i].get(field)
            if not (type(left) is type(right) and left == right):
                mismatches.append({'index': i, 'cycle': rows_on[i].get('cycle'),
                                   'on': left, 'off': right})
        table[field] = {'n_compared': n, 'n_mismatch': len(mismatches), 'mismatches': mismatches}
    return {
        'field_list_source': 'p515_s44_campaign_harness.RECORD_TRAJECTORY_FIELDS (BY IMPORT)',
        'n_fields': len(H.RECORD_TRAJECTORY_FIELDS),
        'n_rows_on': len(rows_on), 'n_rows_off': len(rows_off),
        'rows_length_equal': len(rows_on) == len(rows_off),
        'total_mismatches': sum(v['n_mismatch'] for v in table.values()),
        'fields_with_mismatch': sorted(k for k, v in table.items() if v['n_mismatch']),
        'per_field': table,
    }


def compare_on_against_committed_reference(report_on):
    """NON-GATING. Truncated comparison of arm ON's first two cycles against the
    committed AA-on C* trajectory, mirroring `p515_s40_polish_gap._reproduction_check`'s
    truncated branch (its `_derive_top_level_from_rows` is used BY IMPORT). The
    reference ran at cap 500; this arm runs at cap 2 and cap-independence with AA on
    is UNVERIFIED -- so a difference here is reported, never rationalized, and never
    gates."""
    path = os.path.join(REPO, AA_REFERENCE['report'])
    with open(path) as handle:
        reference = json.load(handle)
    n = len(report_on.get('cycle_trajectory') or [])
    ref_rows = (reference.get('cycle_trajectory') or [])[:n]
    row_diffs = CP._diff(ref_rows, report_on.get('cycle_trajectory') or [], 'cycle_trajectory')
    derived_ref = PG._derive_top_level_from_rows(ref_rows)
    derived_mine = {k: report_on.get(k) for k in derived_ref}
    scalar_diffs = CP._diff(derived_ref, derived_mine, 'derived_top_level')
    diffs = row_diffs + scalar_diffs
    provenance = [d for d in diffs if '.rule_eleven_checklist' in str(d.get('field', ''))
                  or str(d.get('field', '')).startswith('rule_eleven_checklist')]
    diffs = [d for d in diffs if d not in provenance]
    return {
        'gating': False,
        'reference_report': AA_REFERENCE['report'],
        'reference_sha256': CP._sha256_file(path),
        'reference_campaign': AA_REFERENCE['campaign'],
        'reference_cycles_run': reference.get('cycles_run'),
        'reference_gross_operational_cost': reference.get('gross_operational_cost'),
        'mode': (f'truncated: cycle_trajectory[:{n}] + the rows-derived top-level fields, vs the '
                 f"reference's first {n} committed rows (cycle-count-dependent artifacts -- "
                 'esso_capture, solve_profile, network_failures_summary, '
                 'esso_complementarity_diagnostics_by_round, esso_models_pickle -- out of scope)'),
        'n_cycles_compared': n,
        'diff_value_naming': "CP._diff names the reference value 'legacy' and arm ON's value 'lightweight'",
        'n_provenance_diffs': len(provenance),
        'n_diffs': len(diffs),
        'reproduces': len(diffs) == 0,
        'first_differing_field': (diffs[0] if diffs else None),
        'first_diffs': diffs[:20],
        'caveat': ('the reference ran at cap 500 and this arm at cap 2; cap-independence with AA on '
                   'is UNVERIFIED, so this comparison is reported and not gated'),
    }


# ===========================================================================
#  main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='', help='redirects the output root and the working-dir ids')
    args = parser.parse_args()
    suffix = ('_' + args.suffix.strip('_')) if args.suffix else ''
    out_root = os.path.join(REPO, _P45, f'snapshot_off_two_cycle_gate{suffix}')
    started = time.time()

    _log(f'{STAGE}')
    _log(f'git HEAD {_git(["rev-parse", "HEAD"])}; output root {_rel(out_root)}')

    env_before = {k: os.environ.get(k) for k in H.THREAD_CAP_ENV}
    os.environ.update(H.THREAD_CAP_ENV)
    env_after = {k: os.environ.get(k) for k in H.THREAD_CAP_ENV}
    _log(f'thread caps: before={env_before} after={env_after} (set before any solve, so both arms and '
         'the IPOPT subprocesses see the campaign child\'s own caps)')

    failures, instance = check_preconditions(out_root)
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    checklist = capture_path_checklist()
    _log(f'preconditions passed; capture-path checklist: {len(checklist)} items, all true')
    _log(f"instance: candidate_key={instance['candidate_key']} canonical={instance['candidate_canonical']}")

    declared = derive_solves_from_case_file()
    _log(f"DECLARED BEFORE THE RUN: {declared['solves_per_cycle']} solves/cycle "
         f"({declared['network_solves_per_cycle']} network + {declared['esso_solves_per_cycle']} ESSO); "
         f"base = {declared['declared_base_solves_per_arm']} per arm "
         f"(= solves/cycle x (cap {CAP} + initialization)), "
         f"{declared['declared_base_solves_both_arms']} for both arms; gate on the identity "
         f"{declared['identity']}")

    G._acquire_exclusive_run_lock()
    _log('legacy run lock acquired (.p515_g_gate.lock)')

    out_dirs = {'on': os.path.join(out_root, 'on'), 'off': os.path.join(out_root, 'off')}
    counters = CloneCaptureCounters()
    expected_block_counts_holder = {}
    reports, summaries = {}, {}
    with counters.installed():
        for arm, snapshots in (('on', 'on'), ('off', 'off')):
            _log(f'--- ARM {arm.upper()}: run_admm_arm cap {CAP}, snapshots={snapshots!r} ---')
            report, summary = run_arm(arm, snapshots, out_dirs[arm], counters,
                                      expected_block_counts_holder)
            reports[arm], summaries[arm] = report, summary
            bucket = counters.buckets.get(f'arm_{arm}', {})
            _log(f"arm {arm}: cycles={summary['cycles_run']} recourse={summary['recourse']} "
                 f"gross={summary['gross_operational_cost']} "
                 f"clones={bucket.get('clone')} captures={bucket.get('capture')} "
                 f"solves={(summary['solve_profile'] or {}).get('observed', {}).get('permitted_solve')} "
                 f"failures={summary['network_failures_summary']} "
                 f"snapshots_dir={summary['frozen_smopf_inventory']} "
                 f"wall={summary['wall_clock_s']['measured_here']:.1f}s")
        counters.phase = 'post'

    # ---------------- clone / capture gate ----------------
    on_bucket = counters.buckets.get('arm_on', {'clone': 0, 'capture': 0})
    off_bucket = counters.buckets.get('arm_off', {'clone': 0, 'capture': 0})
    clone_capture = {
        'buckets': counters.buckets,
        'arm_on': {'clone': on_bucket.get('clone', 0), 'capture': on_bucket.get('capture', 0)},
        'arm_off': {'clone': off_bucket.get('clone', 0), 'capture': off_bucket.get('capture', 0)},
        'gate_on_positive': on_bucket.get('clone', 0) > 0 and on_bucket.get('capture', 0) > 0,
        'gate_off_zero': off_bucket.get('clone', 0) == 0 and off_bucket.get('capture', 0) == 0,
    }
    n_yd = declared['n_year_day_blocks']
    n_snapshots_on = summaries['on']['frozen_smopf_n_pkl']
    clone_capture['reconciliation_arm_on'] = {
        'expected_pristine_base_clones': 2 * n_yd,
        'note': (f'{n_yd} TSO blocks + {n_yd} node-7 DSO blocks cloned once per '
                 '_run_operational_planning call (_build_pristine_snapshot_bases), plus one '
                 'on-demand rebuild clone per snapshot actually written'),
        'snapshots_written': n_snapshots_on,
        'expected_total_clones': 2 * n_yd + n_snapshots_on,
        'observed_clones': on_bucket.get('clone', 0),
        'identity_holds': on_bucket.get('clone', 0) == 2 * n_yd + n_snapshots_on,
        'expected_captures': 2 * n_yd * (summaries['on']['cycles_run'] or 0),
        'observed_captures': on_bucket.get('capture', 0),
        'capture_identity_holds': on_bucket.get('capture', 0) == 2 * n_yd * (summaries['on']['cycles_run'] or 0),
        'gating': False,
    }

    # ---------------- solve profile ----------------
    solve = {'declared_before_the_run': declared, 'per_arm': {}, 'method':
             ('the base count was declared in advance and the gate is on the reconciliation '
              'identity (an exact number cannot be declared for failures not known in advance); '
              'the process-wide guard is additionally verify()-ed EXACTLY against the reconciled total')}
    reconciled_total = 0
    for arm in ('on', 'off'):
        observed = ((summaries[arm]['solve_profile'] or {}).get('observed') or {}).get('permitted_solve')
        classes = (summaries[arm]['network_failures_summary'] or {}).get('classes') or {}
        tier1, tier2 = classes.get('recovered_tier1', 0), classes.get('recovered_tier2', 0)
        base = declared['declared_base_solves_per_arm']
        expected = base + tier1 + 2 * tier2
        reconciled_total += expected
        solve['per_arm'][arm] = {
            'base_declared': base, 'recovered_tier1': tier1, 'recovered_tier2': tier2,
            'expected_after_reconciliation': expected, 'observed': observed,
            'identity_holds': observed == expected,
            'expected_block_counts_from_planning': expected_block_counts_holder.get(arm),
            'planning_derivation_agrees_with_case_file': (
                (expected_block_counts_holder.get(arm) or {}).get('solves_per_cycle')
                == declared['solves_per_cycle']),
            'arm_inner_guard_counts': (summaries[arm]['solve_profile'] or {}).get('observed'),
        }
    guard_failures = GUARD.verify(reconciled_total)
    solve['process_guard'] = {
        'permitted_call_sites': [list(p) for p in N.PERMITTED],
        'counts': dict(GUARD.counts),
        'reconciled_total_expected': reconciled_total,
        'verify_failures': guard_failures,
        'sum_of_arm_counts': sum(((summaries[a]['solve_profile'] or {}).get('observed') or {}).get(
            'permitted_solve', 0) for a in ('on', 'off')),
    }
    solve['process_guard']['no_solves_outside_the_arms'] = (
        GUARD.counts['permitted_solve'] == solve['process_guard']['sum_of_arm_counts'])
    solve['identity_holds_both_arms'] = all(v['identity_holds'] for v in solve['per_arm'].values())

    # ---------------- bitwise comparison ----------------
    ids_by_arm = {arm: summaries[arm]['working_dir_ids'] for arm in ('on', 'off')}
    token_groups = _arm_tokens(out_dirs, ids_by_arm)
    _log('comparing the two arms bitwise ...')
    files = {}
    for fname in GATING_JSON:
        a = CP._load_json(os.path.join(out_dirs['on'], fname))
        b = CP._load_json(os.path.join(out_dirs['off'], fname))
        if a is None or b is None:
            files[fname] = {'error': f'missing: on={a is not None} off={b is not None}'}
            _log(f'  {fname}: {files[fname]["error"]}')
            continue
        files[fname] = classify(fname, a, b, token_groups)
        _log(f'  {fname}: {files[fname]["n"]}')
    for fname in GATING_JSONL:
        a = CP._load_jsonl(os.path.join(out_dirs['on'], fname))
        b = CP._load_jsonl(os.path.join(out_dirs['off'], fname))
        if a is None or b is None:
            files[fname] = {'error': f'missing: on={a is not None} off={b is not None}'}
            _log(f'  {fname}: {files[fname]["error"]}')
            continue
        entry = classify(fname, a, b, token_groups)
        entry['n_rows'] = {'on': len(a), 'off': len(b)}
        files[fname] = entry
        _log(f'  {fname}: rows={entry["n_rows"]} {entry["n"]}')
        del a, b

    supplementary = {}
    for fname in SUPPLEMENTARY_JSONL:
        a = CP._load_jsonl(os.path.join(out_dirs['on'], fname))
        b = CP._load_jsonl(os.path.join(out_dirs['off'], fname))
        if a is None or b is None:
            supplementary[fname] = {'error': f'missing: on={a is not None} off={b is not None}'}
            continue
        raw = CP._diff(a, b, fname)
        stripped = CP._diff(_strip_keys(a, SUPPLEMENTARY_PATH_LIKE_KEYS),
                            _strip_keys(b, SUPPLEMENTARY_PATH_LIKE_KEYS), fname)
        supplementary[fname] = {'n_rows': {'on': len(a), 'off': len(b)}, 'n_raw_diffs': len(raw),
                                'n_diffs_ignoring_path_like_keys': len(stripped),
                                'path_like_keys_ignored_in_that_count': list(SUPPLEMENTARY_PATH_LIKE_KEYS),
                                'first_diffs_ignoring_path_like_keys': stripped[:20]}

    pickles = {}
    for arm in ('on', 'off'):
        p = os.path.join(out_dirs[arm], f'esso_models_{ARM_LABEL}.pkl')
        pickles[arm] = CP._sha256_file(p) if os.path.isfile(p) else None
    pickles['sha256_equal'] = pickles['on'] is not None and pickles['on'] == pickles['off']

    totals = {'provenance': 0, 'tie_order': 0, 'genuine': 0}
    for v in files.values():
        for k, n in (v.get('n') or {}).items():
            totals[k] += n
    all_files_present = all('error' not in v for v in files.values())

    trajectory = trajectory_field_table(reports['on'].get('cycle_trajectory') or [],
                                        reports['off'].get('cycle_trajectory') or [])
    reference_comparison = compare_on_against_committed_reference(reports['on'])

    # ---------------- the gate ----------------
    snapshot_setting_ok = (
        (summaries['on']['snapshot_setting'] or {}).get('took_effect') is True
        and (summaries['on']['snapshot_setting'] or {}).get('tso_mode_after') == 'lightweight'
        and (summaries['on']['snapshot_setting'] or {}).get('dso_mode_after') == 'lightweight'
        and (summaries['off']['snapshot_setting'] or {}).get('took_effect') is True
        and (summaries['off']['snapshot_setting'] or {}).get('tso_mode_after') == 'off'
        and (summaries['off']['snapshot_setting'] or {}).get('dso_mode_after') == 'off')
    gate_items = {
        'both_arms_ran_two_cycles': (summaries['on']['cycles_run'] == CAP
                                     and summaries['off']['cycles_run'] == CAP),
        'snapshot_setting_verified_both_arms': snapshot_setting_ok,
        'arm_on_clones_and_captures_positive': clone_capture['gate_on_positive'],
        'arm_off_clones_and_captures_exactly_zero': clone_capture['gate_off_zero'],
        'arm_off_wrote_no_pkl_snapshot': summaries['off']['frozen_smopf_n_pkl'] == 0,
        'all_compared_files_present': all_files_present,
        'zero_genuine_diffs': totals['genuine'] == 0,
        'trajectory_fields_identical': (trajectory['total_mismatches'] == 0
                                        and trajectory['rows_length_equal']),
        'solve_reconciliation_identity_holds': solve['identity_holds_both_arms'],
        'process_guard_verified_exactly': not guard_failures,
        'no_solves_outside_the_arms': solve['process_guard']['no_solves_outside_the_arms'],
        'no_blocked_solver_calls': (GUARD.counts['blocked_solve'] == 0
                                    and GUARD.counts['blocked_exec'] == 0),
        'shared_frozen_smopf_untouched': not any(
            reports[a].get('shared_frozen_smopf_modified') or reports[a].get('shared_frozen_smopf_new_files')
            for a in ('on', 'off')),
    }
    gate_pass = all(gate_items.values())

    payload = {
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': _utc(),
        'git_head_at_run': _git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__),
        'script_sha256': CP._sha256_file(os.path.abspath(__file__)),
        'harness_sha256': CP._sha256_file(H.HARNESS_PATH),
        'case_file': {'path': H.CASE_FILE_REL, 'sha256': CP._sha256_file(H.CASE_FILE),
                      'anderson_acceleration_loaded': instance['case_file_anderson_acceleration_loaded'],
                      'declaration': CASE_FILE_AA},
        'instance': {'candidate_label': 'C* (0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025)',
                     'candidate_canonical': instance['candidate_canonical'],
                     'candidate_key': instance['candidate_key'],
                     'candidate_key_pin_matches': instance['candidate_key'] == C_STAR_KEY_PIN},
        'configuration': {
            'arm_label': ARM_LABEL, 'cap': CAP,
            'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES,
            'apply_rho': False, 'full_diagnostics_in_rows': True,
            'configuration_verified_by': ('p515_s44_campaign_harness._config_hook_factory with the '
                                          'case-file AA declaration (the campaign child\'s own hook)'),
            'the_only_difference_between_arms': ("p515_s44_scale_measurement.snapshot_hook_wrapper's "
                                                 "`snapshots` argument: 'on' (no assignment at all, modes "
                                                 "stay at the ADMMParameters defaults) vs 'off' (both modes "
                                                 'set to off and read back)'),
            'thread_caps': {'before': env_before, 'after': env_after, 'source': 'p515_s44_campaign_harness.THREAD_CAP_ENV'},
            'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
            'nlp_solver_path': H._resolve_solver_path_from_dotenv(),
            'both_arms_in_one_process': True,
            'note_on_one_process': ('both arms run sequentially in THIS process, so the per-process string-hash '
                                    'seed -- the cause of the sort-tie non-determinism documented in '
                                    'p515_s43_aa_flagoff_gate -- is identical for the two arms and cannot '
                                    'explain any difference between them'),
        },
        'objective_convention': ('gross_operational_cost is the settlement-excluded gross cost; '
                                 'net_operational_recourse differs by the terminal salvage credit'),
        'capture_path_checklist_asserted_before_run': checklist,
        'preconditions': {'forbidden_live_process_substrings': list(FORBIDDEN_LIVE_PROCESS_SUBSTRINGS),
                          'production_files_checked_clean': list(PRODUCTION_FILES_TO_CHECK_CLEAN)},
        'arms': summaries,
        'clone_capture_counters': clone_capture,
        'solve_profile': solve,
        'comparison_on_vs_off': {
            'comparator': ('p515_s40_clone_capture_preflight._diff (+ EXCLUDE_KEY_NAMES, '
                           'EXCLUDE_DOTTED_SUFFIXES, INTENTIONAL_DIFF_SUFFIXES, ARTIFACT_FILES, '
                           'SIDECAR_JSONL_FILES) + p515_s43_aa_flagoff_gate._classify_diffs + '
                           'p515_s44_tie_classifier.reclassify_sidecar_diffs, all BY IMPORT'),
            'diff_value_naming': "CP._diff names arm ON's value 'legacy' and arm OFF's value 'lightweight'",
            'excluded_field_names': sorted(CP.EXCLUDE_KEY_NAMES),
            'excluded_dotted_suffixes': sorted(CP.EXCLUDE_DOTTED_SUFFIXES),
            'intentional_difference_suffixes_excluded': sorted(CP.INTENTIONAL_DIFF_SUFFIXES),
            'permitted_provenance_suffixes': PERMITTED_PROVENANCE_SUFFIXES,
            'arm_path_normalization_tokens': token_groups,
            'gating_files': list(GATING_JSON + GATING_JSONL),
            'files': files,
            'totals_by_class': totals,
            'all_files_present': all_files_present,
            'supplementary_not_gating': supplementary,
            'not_compared': NOT_COMPARED,
            'esso_models_pickle_sha256_informational': pickles,
        },
        'trajectory_field_table': trajectory,
        'arm_on_vs_committed_aa_reference_NON_GATING': reference_comparison,
        'gate_items': gate_items,
        'gate_pass': gate_pass,
        'wall_clock_s': time.time() - started,
    }

    os.makedirs(out_root, exist_ok=True)
    gate_path = os.path.join(out_root, 'gate.json')
    _refuse_overwrite(gate_path)
    with open(gate_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    _log(f'wrote {_rel(gate_path)}')

    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            manifest[_rel(fpath)] = CP._sha256_file(fpath)
    manifest_path = os.path.join(out_root, 'manifest_sha256.json')
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    _log(f'wrote {_rel(manifest_path)} ({len(manifest)} files)')

    _log(f'GATE_PASS={gate_pass}')
    for name, value in gate_items.items():
        _log(f'  {"OK " if value else "FAIL"} {name}')
    _log(f"diff classification totals: {totals}; trajectory mismatches: {trajectory['total_mismatches']}")
    _log(f"clone/capture: on={clone_capture['arm_on']} off={clone_capture['arm_off']}")
    _log(f"solves: {solve['per_arm']}")
    _log(f"NON-GATING vs the committed AA-on reference: reproduces={reference_comparison['reproduces']} "
         f"n_diffs={reference_comparison['n_diffs']} "
         f"first={reference_comparison['first_differing_field']}")
    if not gate_pass:
        _log('*** GATE FAILED *** first genuine diffs, reported as-is:')
        for fname, entry in files.items():
            for d in (entry.get('genuine_diffs_first') or [])[:10]:
                _log(f'  [GENUINE] {fname}: {d}')
        sys.exit(1)


if __name__ == '__main__':
    main()
