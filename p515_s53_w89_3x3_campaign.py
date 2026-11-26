"""
P5.15 Addendum 46 (3 x 3 pair confirmed under production + the tight tail), Planner task W89 step 2 -- THE 3 x 3 PAIR:
instance, memory probe, frozen stage spec v34 (predecessor v33 f0f7a4a4 <- v32 69449731), launcher, child smoke. BUILT AND FROZEN, NOT
RUN IN W89.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 40 ruling 3 (3 x 3 = the paper's multi-scenario instance), Addendum 44
("3 x 3 confirmed with production's prefix draw [1,2,3] x [1,2,3] and R = 0.9331 recorded"; the 3 x 3 prediction band
[0.93, 1.09] with the resolution caveat), Addendum 45 (persist certified models if memory allows), Addendum 46 (the pair
under production + tight tail, R restated against the SRP1 reference re-run under it); Planner task W89 step 2.

THE INSTANCE (`s53_3x3`). The paper's structure at 3 market x 3 operation scenarios: 5 representative years (2025 /
2028 / 2031 / 2034 / 2037, each a 3-year block), 4 days, NumMarketScenarios 3 and num_operation_scenarios 3 on the TSO
and every DSO -- derived by the scale harness's own machinery (`p515_s44_scale_measurement.derive_case('paper',
{num_market_scenarios: 3, num_operation_scenarios: 3})`, which edits ONLY Years / NumMarketScenarios /
num_operation_scenarios of data/SRP1/SRP1.json; the writer of the 2 x 2 pilot, json.dumps(indent='\\t')) and written
ONCE to INSTANCE_CASE_REL. The PREFIX DRAW is verified on the REALIZED data: the 3 x 3 case and the committed paper
case (5 x 5, d726307c) are both read with production's reader, and for every network block (TSO + 3 DSOs x 5 years x
4 days) every realized scenario array -- market energy and flexibility prices, every load's pd / qd / flexibility
up / down, every generator's pg / qg -- of scenario s in the 3 x 3 case must equal, BIT FOR BIT, scenario s of the
5 x 5 case for s = 1, 2, 3 (W52 checked only the index-level `sample(n=3)` prefix of `sample(n=5)`).
R = 0.9331 is W52's R_r2 of the market subset [1, 2, 3] (selection_3x3.json, rank 4 of 10), recorded, not recomputed.

THE CELLS (both certified; cap 500; 10 consecutive all-pass cycles): x0 (nodes 5 / 7 / 9 at (0, 0), 2025) and
n7_4h_e1 (node 7 at 0.25 MVA / 1.0 MWh, 2025). CONFIGURATION = the multi-scenario production baseline of the 2 x 2
alpha row's pair 1 (C2 ageing baseline declared, case-file AA declared, row 18 premium alpha = 0.5 with no floor, the
derived instance) PLUS the convergence-depth tail DECLARED {'enabled': True, 'compl_inf_tol': 1e-6}; post-certification:
persist REQUESTED ({'persist_certified_models': True, 'hull_polish': False}; Addendum 46 voided the polished
convention), DECIDED at --freeze-spec by the declared memory rule (MEMORY_RULE_TEXT) over the committed measurements --
"persist unless the memory preflight forbids it" (Planner W89); concurrency decided by the same rule. Per-cycle response capture and the floor-status capture are ON: the harness turns the
alpha-row capture on for every derived-instance evaluation and the floor-status capture on for every evaluation; both
are asserted before any run (rule eleven, `launcher_checklist`).

THE PATH. `H.evaluate` spawns one fresh interpreter per evaluation (`--child` -> `main_child` -> `_child_real` ->
`p515_g_g1_g4_admm_gates.run_admm_arm`). THIS process never solves: SolveProfileGuard(permitted=()) is armed at import
before any project import, with every imported launcher's own permitted=() guard; all verified at exactly 0 on every
exit path. THE SOLVE CLAIM IS RECONCILED PER EVENT IN THE CHILD RECORD, NOT GUARD-VERIFIED (the W86 wording): observed
== 83 x (cycles_run + 1) + every retry attempted, 83 = (1 + 3 DSO) x 5 years x 4 days + 3 ESSO.

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-instance --scratch D     ZERO SOLVES. Derives the 3 x 3 case (write-once), reads it and the committed 5 x 5
                                    case with production's reader (plots redirected into D), verifies the prefix on the
                                    realized arrays, records the instance facts -> <root>/instance/instance_record.json.
  --memory-probe W --scratch D      ZERO SOLVES. W in {srp1, s52_pilot_2x2, s53_3x3}: production's ADMM model build of
                                    the unit candidate (80 or 48 network blocks + 3 ESSO; `.optimize` intercepted), RSS
                                    after each stage, then the persist function's pickle into D (size, RSS transient;
                                    the file is deleted after measuring) -> <root>/memory_probe/memory_probe_<W>.json.
  --freeze-spec                     ZERO SOLVES. Frozen stage spec v34 (write-once, named by its sha256; predecessor
                                    v32): instance, cells, configuration, eval keys, pre-launch assertion, declared
                                    solve profile, gates, smoke gate, R reference, predictions, memory rule, estimates.
  --stage {smoke,pair} --freeze     ZERO SOLVES. The campaign spec (fresh root) pinning v34; the pre-launch assertion
                                    on the FROZEN entries.
  --stage smoke --run --spec-sha256 S   NOT RUN IN W89. One x0 evaluation, cap 2, through H.evaluate; the smoke gate.
  --stage pair --run --spec-sha256 S    NOT RUN IN W89. Requires the smoke gate committed and PASS; memory preflight
                                    (refusing); the two cells at the stage-spec concurrency (1: sequential); per-cell gates; value / R.

PER-CELL GATES (pair; both cells -- there is no other arm): G1 harness clean; G2 eval key == the stage spec's; G3 per-round append
reconciles; G4 tail state check; G5 solve profile reconciled per event (83 per round); G6 floor records under v32
(B = 80, the final accepted attempt per block of the terminal round); G7 append sealed; G8 certified -> models
persisted iff the stage spec decided persist (sha256 recorded), absent otherwise; G9 ESS ageing read-back; G10 the alpha-row capture (multi-scenario terminal, response
terminal and workbook written; multi-scenario checks pass; activation read-back all ok; per-cycle response on every
cycle; derived-instance checksum and premium read back in the child; initialisation identity recorded); G11 the
acceptance rule of v32 reproduces production's failure-event classification on every round.

SMOKE GATE (x0, cap 2; declared in the stage spec before it runs): S1 exit 0, record written by the child, not_certified,
cycles_run 2; S2 append byte-identical to the end-of-run file; S3 tail checklist line 1 before any solve; S4 tail state
check; S5 solve profile == 83 x 3 = 249 + retries, reconciled per event; S6 tail INACTIVE at cap 2; S7 every record
passes v32's per-record predicate at production compl_inf_tol (no tail round); S8 append sealed; S9 post-certification
skipped (uncertified), no pickle; S10 eval key == the pair's x0 key; S11 ESS ageing read-back; S12 the alpha-row capture
(G10); S13 sigma calibration inside production's band; S14 parent guards verify(0) == [].

EXACT COMMANDS (repo root):
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w89_3x3_campaign.py \\
      --freeze-instance --scratch <dir outside the repo> > data/SRP1/Results/P515S53/w89_3x3/freeze_instance_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w89_3x3_campaign.py \\
      --memory-probe <W> --scratch <dir> > data/SRP1/Results/P515S53/w89_3x3/memory_probe_<W>_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w89_3x3_campaign.py \\
      --freeze-spec > data/SRP1/Results/P515S53/w89_3x3/freeze_spec_v34_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w89_3x3_campaign.py \\
      --stage <smoke|pair> --freeze > data/SRP1/Results/P515S53/w89_3x3/<smoke|pair>_r2_freeze_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w89_3x3_campaign.py \\
      --stage smoke --run --spec-sha256 <sha> > data/SRP1/Results/P515S53/w89_3x3/smoke_r2_launch.log 2>&1
  set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w89_3x3_campaign.py \\
      --stage pair --run --spec-sha256 <sha> > data/SRP1/Results/P515S53/w89_3x3/pair_r2_launch.log 2>&1
Exit codes (--run): smoke 0 PASS / 1 FAIL; pair 0 every gate holds and both cells certified, 2 a cell not certified
(harness clean), 1 a gate / harness / guard / precondition failure.
"""

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W89 3x3 launcher (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402 -- stdlib-only at import
# W89 step 1: the v32 G6 evaluator (arms its own guard and, through W88 / W87, the W86 launcher's -- all permitted=()).
import p515_s53_w89_g6_final_attempt_reeval as X  # noqa: E402
L = X.L    # the W86 launcher: evaluation_checks, memory_preflight, committed_eval_keys, _production_compl_inf_tol
# The alpha-row launcher: its rule-eleven checklist and cell formulas (arms its own permitted=() guard; it installs and
# removes the s52 pilot launcher's at import).
import p515_s53_alpha_row_campaign as A  # noqa: E402

GUARDS_LIFO = (A.PARENT_GUARD, L.PARENT_GUARD, X.W.GUARD, X.M88.GUARD, X.GUARD, PARENT_GUARD)

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRING = 'p515_s53_w89_3x3_campaign'
STAGE_TEXT = ('P5.15 Addendum 46, W89 step 2 -- the 3 x 3 pair (x = 0 and the smallest node-7 unit) on the prefix-draw '
              '3 x 3 instance under production + the tight tail {True, 1e-6}; certified-model persistence and '
              'concurrency decided by the declared memory rule')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = os.path.join(_P53, 'w89_3x3')
SPEC_V32 = {'path': os.path.join(_P53, 'frozen_s53_spec_v32_69449731.json'),
            'sha256': '69449731af8b8c620ecac45627d7d9d10126a87dc7e0b05af125907bce764a28'}
SPEC_V33 = {'path': os.path.join(_P53, 'frozen_s53_spec_v33_f0f7a4a4.json'),
            'sha256': 'f0f7a4a43314d2634d0211ca3bdd1e9ae4f2ae232fecf5c210337404e977b9d9'}
# v34 (predecessor ss, NOT edited): the stage spec this launcher freezes. ss pinned launcher revision fb73b9b0, whose
# campaign freezes FAILED their own validation (two launcher bugs, nothing run): a post-certification request with
# nothing enabled is resolved by the harness to None (the entry check, G8 and S9 expected a dict), and the pre-launch
# assertion on a frozen spec demanded entries for labels the smoke spec does not hold. Fixed here; v34 re-derives
# every v33 decision from the same committed inputs.
SPEC_PREFIX = 'frozen_s53_spec_v34_'
SPEC_VERSION = 34
REEVAL_W89 = {'path': os.path.join(X.OUT_ROOT, X.OUT_FILE),
              'sha256': '722272014523583ed7b60cfe8f339e1bb050ddcab41ba9c510c2eadfd5496224'}

# ---- the instance ------------------------------------------------------------------------------------------------
INSTANCE_LABEL = 's53_3x3'
INSTANCE_DIR_REL = os.path.join(ROOT_REL, 'instance')
INSTANCE_CASE_REL = os.path.join(INSTANCE_DIR_REL, 'SRP1__s53_3x3.json')
INSTANCE_RECORD_REL = os.path.join(INSTANCE_DIR_REL, 'instance_record.json')
DERIVE_BASE = 'paper'
DERIVE_OVERRIDES = {'num_market_scenarios': 3, 'num_operation_scenarios': 3}
N_SCEN = 3
EXPECTED_YEARS = {'2025': 3, '2028': 3, '2031': 3, '2034': 3, '2037': 3}
SOURCE_CASE_REL = os.path.join('data', 'SRP1', 'SRP1.json')
PAPER_CASE = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S44', 'scale_measurement', 'paper_build', 'case',
                                   'SRP1__paper.json'),
              'sha256': 'd726307c1949d91b34b00241fbc713412c689b113222929133eda38d9756ca14',
              'scenario_checksum': '1e8bdd3e5233442a87fbe44b78281c8ae61b09c09700388fed20ab9684d18aef'}
SELECTION = {'path': os.path.join(_P53, 'selection_3x3', 'selection_3x3.json'),
             'sha256': '291be9c798d7914a8d5962871aa266e3e88d9a6f01ef8cbaa9a69101605cdd12'}
PREFIX_SUBSET = [1, 2, 3]
R_PREFIX_RECORDED = 0.9331   # Addendum 44: R_r2 of [1, 2, 3], rounded as the Planner recorded it
PILOT_2X2 = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S52', 'pilot_instance', 'SRP1__s52_pilot_2x2.json'),
             'sha256': '7ecff44a874892187d1dd2e3d5ed0664a2f90a4bfa4cdb820a4abcbbd828f949'}
BLOCKS_PER_ROUND = 80          # B = (1 + 3 DSO) x 5 years x 4 days; asserted against the instance at every freeze
N_ESSO = 3
SOLVES_PER_ROUND = BLOCKS_PER_ROUND + N_ESSO   # 83; asserted against S.declared_solve_profile
EXPECTED_DSO_BLOCKS = 60
EXPECTED_TSO_BLOCKS = 20

# ---- configuration (the alpha row's pair-1 baseline + the tail + persist) -----------------------------------------
LABEL = L.LABEL
ARM_LABEL = L.ARM_LABEL
YEAR = L.YEAR
CASE_FILE_AA = dict(L.CASE_FILE_AA)
ESS_AGEING_BASELINE = copy.deepcopy(L.ESS_AGEING_BASELINE)
ESS_PARAMS_SHA256 = L.ESS_PARAMS_SHA256
TAIL = {'enabled': True, 'compl_inf_tol': 1e-6}
PREMIUM = {'alpha': 0.5, 'floor': None}
# Post-certification: persist REQUESTED (Addenda 45 / 46; Planner W89), decided at --freeze-spec by the declared
# memory rule (MEMORY_RULE_TEXT); the decided setting is the stage spec's configuration.post_certification and every entry carries it.
POST_CERTIFICATION_REQUESTED = {'persist_certified_models': True, 'hull_polish': False}
POST_CERTIFICATION_IF_PERSIST_FORBIDDEN = {'persist_certified_models': False, 'hull_polish': False}
REQUIRED_CONSECUTIVE_CYCLES = 10
CELLS = {
    'x0': {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)},
    'n7_4h_e1': {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)},
}
UNIT_E_MWH = 1.0
# The 2 x 2 alpha-row keys of the SAME candidates at alpha = 0.5 (pair 1, campaign_s53_alpha_row_v25) -- the pre-3x3
# evaluations the 3 x 3 keys must differ from; recomputed and checked at every freeze.
ALPHA_ROW_2X2_KEYS = {'x0': '7d53b6f21b686a44', 'n7_4h_e1': '711fce9aa74d6878'}
ALPHA_ROW_PAIR1 = {'path': os.path.join(_P53, 'alpha_row', 'campaign_s53_alpha_row_v25', 'pair_1_results.json'),
                   'sha256': '300c2de33632afeb70e88da99e18749df47993c15ed857f8f1af302143b3aa4e'}
STAGES = {
    'smoke': {'campaign_id': 's53_w89_3x3_smoke_r2', 'labels': ('x0',), 'cap': 2},
    'pair': {'campaign_id': 's53_w89_3x3_pair_r2', 'labels': ('x0', 'n7_4h_e1'), 'cap': 500},
}   # concurrency: smoke 1; pair from the stage spec (the memory rule, decided at --freeze-spec from the probes)
# The r1 campaign freezes under v33 (launcher fb73b9b0): FAILED their own validation, never run; kept as evidence.
SUPERSEDED_CAMPAIGN_FREEZES = {
    'smoke': {'path': os.path.join(ROOT_REL, 'campaign_s53_w89_3x3_smoke',
                                   'campaign_spec_s53_w89_3x3_smoke_4d2bf2f5.json'),
              'sha256': '4d2bf2f52cdfd35345adcab171533a7ee15ce4349fed8882c39b8927783598dd',
              'failing': ['spec check post_certification_as_v33', 'pre-launch assertion on the frozen spec']},
    'pair': {'path': os.path.join(ROOT_REL, 'campaign_s53_w89_3x3_pair', 'campaign_spec_s53_w89_3x3_pair_82379368.json'),
             'sha256': '82379368dde1d46ff0eda864a46720d008c3d0295bc7dbb8d14e0b70cf3b7fac',
             'failing': ['spec check post_certification_as_v33']},
    'reason': ('launcher bugs, caught by the launcher\'s own validation at freeze (exit 1): the harness resolves a '
               'post-certification request with nothing enabled to None; the frozen-entry key check iterated the pair '
               'labels on the one-label smoke spec'),
}
SMOKE_GATE_FILE = 'smoke_gate.json'
SMOKE_MANIFEST_FILE = 'smoke_manifest_sha256.json'
PAIR_RESULTS_FILE = 'campaign_results.json'
PAIR_MANIFEST_FILE = 'campaign_manifest_sha256.json'
EXTRA_CLEAN_FILES = (SCRIPT_NAME, H.ESS_PARAMS_FILE_REL, 'shared_energy_storage_parameters.py',
                     'shared_energy_storage.py', 'p515_s44_scale_measurement.py', 'p515_s53_w89_g6_final_attempt_reeval.py',
                     'p515_s53_alpha_row_campaign.py', 'p515_s52_pilot_campaign.py', 'p515_s53_w86_tail_recert_campaign.py',
                     'p515_s53_w87_g6_rescope_reeval.py', 'p515_s53_w88_g6_floor_reeval.py', SOURCE_CASE_REL)
GIB = 1 << 30
MEMORY_PROBES = ('srp1', 's52_pilot_2x2', 's53_3x3')
MEMORY_PROBE_DIR_REL = os.path.join(ROOT_REL, 'memory_probe')
# Measured runtime peaks the probe is calibrated against (committed records, same code path as the pair's children):
RUNTIME_PEAKS = {
    's52_pilot_2x2': {'source': os.path.join(_P53, 'alpha_row', 'campaign_s53_alpha_row_v25', 'pair_1_results.json'),
                      'what': 'max child wait4 ru_maxrss of the alpha row pair 1 (x0 and unit at alpha 0.5; no persist, '
                              'hull polish on)'},
    'srp1': {'source': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'tight_tail_w86',
                                    'campaign_s53_w86_tail_recert', 'campaign_results.json'),
             'what': 'the W86 tail re-certification children (persist on): production-state peak (before the persist '
                     'step) and child peak (after it)'},
    's52_pilot_2x2_smoke': {'source': os.path.join(_P53, 'alpha_row', 'campaign_s53_alpha_row_smoke_r2', 'evals',
                                                   '7d53b6f21b686a44_x0_a0p50', 'evaluation_record.json'),
                            'what': ('the 2 x 2 alpha-row smoke r2 (x0, alpha 0.5, cap 2, ALONE: concurrency 1): child '
                                     'peak, production-state peak, and the RSS at the start of the terminal capture')},
}
# Recorded memory availability (the vm_stat measure) at every committed multi-child launch on this machine.
AVAILABILITY_OBSERVED = (
    (os.path.join(_P53, 'alpha_row', 'campaign_s53_alpha_row_v25', 'pair_1_results.json'), ('memory_preflight',)),
    (os.path.join(_P53, 'alpha_row', 'campaign_s53_alpha_row_v25', 'pair_2_results.json'), ('memory_preflight',)),
    (os.path.join(_P53, 'alpha_row', 'campaign_s53_alpha_row_v25', 'pair_3_results.json'), ('memory_preflight',)),
    (os.path.join(_P53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'campaign_results.json'),
     ('memory_preflight_at_run',)),
)


# ======================================================================================================================
#  utilities
# ======================================================================================================================
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
    return L._git_state(rel)


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def guards_verify():
    names = ('alpha_row_launcher', 'w86_launcher', 'w87', 'w88', 'w89_step1', 'w89_3x3_parent')
    return {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in zip(names, GUARDS_LIFO)}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    """EVERY exit path: verify every guard at exactly 0, uninstall them LIFO, exit (1 if a guard fails)."""
    g = guards_verify()
    _log(f'[W89-3x3] guards {g} {extra_msg}')
    for guard in GUARDS_LIFO:
        guard.uninstall()
    sys.exit(code if _guards_ok(g) else 1)


def _own_process_alive():
    """Other live processes running THIS script (never this process or its ancestors). Reads `ps` output; no pattern
    is passed on a command line."""
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and OWN_PROCESS_SUBSTRING in parts[1]:
            hits.append(line.strip())
    return hits


def _write_once_text(rel, text):
    with open(_abs(rel), 'x') as handle:
        handle.write(text)


# ======================================================================================================================
#  the instance (zero solves)
# ======================================================================================================================
def derive_instance_text():
    import p515_s44_scale_measurement as S
    case, spec, changes = S.derive_case(DERIVE_BASE, dict(DERIVE_OVERRIDES))
    return json.dumps(case, indent='\t'), case, spec, changes


def _read(case_rel, scratch, tag):
    """Production's reader (the scale harness's `read_planning_from_derived_case`, as the child's
    `install_derived_instance` uses it) with the read's plots / results / logs redirected under `scratch`."""
    import p515_s44_scale_measurement as S
    out_dir = tempfile.mkdtemp(prefix=f'w89_{tag}_', dir=scratch)
    t0 = time.time()
    planning = S.read_planning_from_derived_case({'derived_case': {'path': case_rel}}, out_dir, H._NoStageLog())
    return planning, out_dir, time.time() - t0


def _f64(value):
    import numpy as np
    return np.asarray(value, dtype=np.float64)


def _prefix_equal(a3, a5, n):
    """a3 (n rows) equals the first n rows of a5, bit for bit (float64 bytes)."""
    x, y = _f64(a3), _f64(a5)
    if x.shape[0] != n or y.shape[0] < n or x.shape[1:] != y.shape[1:]:
        return False, {'shape_3x3': list(x.shape), 'shape_5x5': list(y.shape)}
    return x.tobytes() == y[:n].tobytes(), {'shape_3x3': list(x.shape), 'shape_5x5': list(y.shape)}


def prefix_verification(p3, p5):
    """The realized-data prefix check (module docstring)."""
    holders = [('TSO', p3.transmission_network, p5.transmission_network)] + [
        (f'DSO{n}', p3.distribution_networks[n], p5.distribution_networks[n]) for n in sorted(p3.distribution_networks)]
    n_arrays, n_blocks, mismatches, families = 0, 0, [], Counter()
    probs = {}
    same_keys = sorted(p3.distribution_networks) == sorted(p5.distribution_networks)
    for tag, h3, h5 in holders:
        same_keys = same_keys and list(h3.years) == list(h5.years) and list(h3.days) == list(h5.days)
        for y in h3.years:
            for d in h3.days:
                a, b = h3.network[y][d], h5.network[y][d]
                n_blocks += 1
                probs[f'{tag}|{y}|{d}'] = {'market': [float(v) for v in a.prob_market_scenarios],
                                           'operation': [float(v) for v in a.prob_operation_scenarios]}
                items = [('cost_energy_p', None, a.cost_energy_p, b.cost_energy_p),
                         ('cost_flex', None, a.cost_flex, b.cost_flex)]
                loads5 = {ld.load_id: ld for ld in b.loads}
                for ld in a.loads:
                    lb = loads5.get(ld.load_id)
                    if lb is None:
                        mismatches.append({'block': f'{tag}|{y}|{d}', 'what': f'load {ld.load_id} missing in 5x5'})
                        continue
                    items += [('load.pd', ld.load_id, ld.pd, lb.pd), ('load.qd', ld.load_id, ld.qd, lb.qd),
                              ('load.flex_p_up', ld.load_id, ld.flexibility.active_power.upward,
                               lb.flexibility.active_power.upward),
                              ('load.flex_p_down', ld.load_id, ld.flexibility.active_power.downward,
                               lb.flexibility.active_power.downward)]
                gens5 = {g.gen_id: g for g in b.generators}
                for g in a.generators:
                    gb = gens5.get(g.gen_id)
                    if gb is None:
                        mismatches.append({'block': f'{tag}|{y}|{d}', 'what': f'generator {g.gen_id} missing in 5x5'})
                        continue
                    items += [('gen.pg', g.gen_id, g.pg, gb.pg), ('gen.qg', g.gen_id, g.qg, gb.qg)]
                for fam, ident, v3, v5 in items:
                    ok, shapes = _prefix_equal(v3, v5, N_SCEN)
                    n_arrays += 1
                    families[fam] += 1
                    if not ok:
                        mismatches.append({'block': f'{tag}|{y}|{d}', 'family': fam, 'id': ident, **shapes})
    one_third = [1.0 / N_SCEN] * N_SCEN
    probs_ok = all(v['market'] == one_third and v['operation'] == one_third for v in probs.values())
    return {'definition': ('for every network block of the 3 x 3 read: every realized scenario array (market energy and '
                           'flexibility prices; per load pd, qd, flexibility active-power up / down; per generator pg, '
                           'qg), as float64 bytes, equals the first 3 scenario rows of the same array in the 5 x 5 read'),
            'n_blocks': n_blocks, 'n_arrays_compared': n_arrays, 'arrays_per_family': dict(sorted(families.items())),
            'n_mismatches': len(mismatches), 'mismatches_first': mismatches[:20],
            'same_networks_years_days': same_keys,
            'probabilities_3x3_all_one_third': probs_ok,
            'probabilities_sample': dict(list(probs.items())[:2]),
            'prefix_holds': same_keys and not mismatches and n_arrays > 0,
            'realized_subset_market': PREFIX_SUBSET if (same_keys and not mismatches) else None,
            'realized_subset_operation': PREFIX_SUBSET if (same_keys and not mismatches) else None}


def instance_facts(planning):
    """Zero solves: checksum, dimensions, B, solves per cycle, block weights, min hourly mean price, I(x) per cell."""
    import pyomo.environ as pe
    import p515_s44_scale_measurement as S
    import p56a_oracle as O
    import shared_resources_planning as srp
    import model_construction_helpers as MCH
    tn = planning.transmission_network
    facts = {'scenario_checksum': planning.scenario_metadata['combined_scenario_checksum'],
             'planning_dimensions': S.planning_dimensions(planning),
             'expected_block_counts': S.expected_block_counts(planning),
             'declared_solve_profile_per_cycle': S.declared_solve_profile(planning, 1)['solves_per_cycle'],
             'n_tso_blocks': len(tn.years) * len(tn.days),
             'n_dso_blocks': sum(len(dn.years) * len(dn.days) for dn in planning.distribution_networks.values()),
             'num_market_scenarios': planning.num_market_scenarios,
             'num_operation_scenarios_per_network': {'TSO': tn.num_oper_scenarios, **{
                 f'DSO{n}': dn.num_oper_scenarios for n, dn in planning.distribution_networks.items()}}}
    facts['B_blocks_per_round'] = facts['n_tso_blocks'] + facts['n_dso_blocks']
    facts['block_weights_tso'] = {f'{y}|{d}': srp._get_admm_block_weight(tn, y, d) for y in tn.years for d in tn.days}
    facts['median_block_weight_production'] = srp._compute_median_admm_block_weight(planning)
    pib = [float(MCH.expected_market_price(tn.network[y][d], p)) for y in tn.years for d in tn.days
           for p in range(planning.num_instants)]
    facts['min_hourly_mean_price'] = min(pib)
    facts['premium_floor_needed'] = bool(min(pib) <= 0.0)
    sed = planning.shared_ess_data
    master = sed.build_master_problem()
    i_x = {}
    for label, nodes in CELLS.items():
        x = {(n, y): {'s': 0.0, 'e': 0.0} for n in sed.active_distribution_network_nodes for y in sed.years}
        for n, (s_val, e_val) in nodes.items():
            if s_val or e_val:
                ykey = next(y for y in sed.years if int(y) == YEAR)
                x[(n, ykey)] = {'s': s_val, 'e': e_val}
        cand = O.vector_to_candidate(planning, x)
        sed.load_candidate_solution_into_master_model(master, cand)
        i_x[label] = {'candidate_key': _key_of(label), 'I_x_eur_master_expression': float(pe.value(master.investment_cost)),
                      'I_x_eur_p56a_transcription': O.investment_cost(planning, cand),
                      'case_file_budget_eur': sed.params.budget}
    facts['investment_cost'] = i_x
    facts['investment_cost_note'] = ('production master expression on THIS instance (no solve); scenario-free, so it '
                                     'equals the 2 x 2 instance\'s (same years)')
    return facts


def freeze_instance(started, scratch):
    tag = 'W89-INSTANCE'
    failures = []
    for rel in (INSTANCE_CASE_REL, INSTANCE_RECORD_REL):
        if os.path.exists(_abs(rel)):
            failures.append(f'{rel} exists (write-once)')
    for name, pin in (('paper_case', PAPER_CASE), ('selection_3x3', SELECTION)):
        st = _git_state(pin['path'])
        if _sha(pin['path']) != pin['sha256'] or not (st['git_tracked'] and st['git_clean']):
            failures.append(f'pin {name} {pin} not as committed: {st}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    text, case, spec, changes = derive_instance_text()
    os.makedirs(_abs(INSTANCE_DIR_REL), exist_ok=True)
    _write_once_text(INSTANCE_CASE_REL, text)
    case_sha = _sha(INSTANCE_CASE_REL)
    # the committed 5 x 5 case against the same derivation (checked): same writer and source -> same bytes
    import p515_s44_scale_measurement as S
    paper_case, _ps, paper_changes = S.derive_case('paper', {})
    paper_text = json.dumps(paper_case, indent='\t')
    with open(_abs(PAPER_CASE['path'])) as handle:
        paper_on_disk = handle.read()
    p3, dir3, wall3 = _read(INSTANCE_CASE_REL, scratch, '3x3')
    p5, dir5, wall5 = _read(PAPER_CASE['path'], scratch, '5x5')
    prefix = prefix_verification(p3, p5)
    checksum5 = p5.scenario_metadata['combined_scenario_checksum']
    del p5
    facts = instance_facts(p3)
    sel = _load(SELECTION['path'])
    rank = next(i + 1 for i, r in enumerate(sel['market']['ranking']) if list(r['subset']) == PREFIX_SUBSET)
    row = sel['market']['ranking'][rank - 1]
    derived = H.validate_derived_instance({
        'instance_label': INSTANCE_LABEL, 'case_path': INSTANCE_CASE_REL, 'case_sha256': case_sha,
        'scenario_checksum': facts['scenario_checksum'], 'source_case_path': SOURCE_CASE_REL,
        'source_case_sha256': _sha(SOURCE_CASE_REL), 'changes_vs_source': changes})
    checks = {
        'years_as_paper': case['Years'] == EXPECTED_YEARS,
        'changes_only_scenario_counts_and_years': sorted({c['key'].split('[')[0] for c in changes}) == [
            'DistributionNetworks', 'NumMarketScenarios', 'TransmissionNetwork', 'Years'],
        'paper_case_rederives_byte_identical': paper_text == paper_on_disk,
        'paper_read_checksum_equals_committed': checksum5 == PAPER_CASE['scenario_checksum'],
        'prefix_holds_on_realized_arrays': prefix['prefix_holds'],
        'probabilities_one_third': prefix['probabilities_3x3_all_one_third'],
        'B_equals_80': facts['B_blocks_per_round'] == BLOCKS_PER_ROUND,
        'dso_tso_blocks': (facts['n_dso_blocks'], facts['n_tso_blocks']) == (EXPECTED_DSO_BLOCKS, EXPECTED_TSO_BLOCKS),
        'solves_per_cycle_83': facts['declared_solve_profile_per_cycle'] == SOLVES_PER_ROUND,
        'no_premium_floor_needed': not facts['premium_floor_needed'],
        'scenario_checksum_differs_from_5x5': facts['scenario_checksum'] != PAPER_CASE['scenario_checksum'],
    }
    record = {
        'schema': 'p515_s53_w89_3x3_instance_record_v1', 'stage': STAGE_TEXT, 'utc': _utc(),
        'git_head': H._git(['rev-parse', 'HEAD']), 'script': SCRIPT_NAME,
        'script_sha256_at_instance_freeze': H.sha256_file(os.path.abspath(__file__)),
        'instance_label': INSTANCE_LABEL, 'case_path': INSTANCE_CASE_REL, 'case_sha256': case_sha,
        'derive_base': DERIVE_BASE, 'derive_overrides': DERIVE_OVERRIDES, 'derive_description': spec.get('description'),
        'changes_vs_source': changes, 'source_case': {'path': SOURCE_CASE_REL, 'sha256': _sha(SOURCE_CASE_REL)},
        'paper_case': {**PAPER_CASE, 'rederived_byte_identical': checks['paper_case_rederives_byte_identical'],
                       'paper_changes_vs_source': paper_changes, 'read_checksum': checksum5},
        'derived_instance_declaration': derived,
        'prefix_verification': prefix,
        'prefix_draw': {'market': PREFIX_SUBSET, 'operation': PREFIX_SUBSET,
                        'R_recorded': R_PREFIX_RECORDED, 'selection_source': SELECTION,
                        'selection_row_for_subset': {k: row[k] for k in ('subset', 'spread_u', 'spread_r2', 'R_u', 'R_r2',
                                                                         'R_gn', 'abs_diff_u_to_target')},
                        'selection_rank_of_subset': rank, 'selection_selected': sel['market']['selected'],
                        'R_r2_rounds_to_recorded': round(row['R_r2'], 4) == R_PREFIX_RECORDED,
                        'w52_index_level_check': sel['prefix_check']},
        'facts': facts, 'checks': checks, 'all_checks_pass': all(checks.values()),
        'reads': {'3x3': {'redirected_to': dir3, 'wall_s': wall3}, '5x5': {'redirected_to': dir5, 'wall_s': wall5}},
        'solve_claim': 'ZERO SOLVES, guard-verified (every armed permitted=() guard verify(0))',
        'guards': guards_verify(), 'wall_s': time.time() - started,
    }
    H._write_once_json(_abs(INSTANCE_RECORD_REL), record)
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] wrote {INSTANCE_CASE_REL} sha256 {case_sha}; scenario checksum {facts["scenario_checksum"]}')
    _log(f"[{tag}] prefix on realized arrays: holds {prefix['prefix_holds']} ({prefix['n_arrays_compared']} arrays over "
         f"{prefix['n_blocks']} blocks, {prefix['n_mismatches']} mismatches; families {prefix['arrays_per_family']}); "
         f"probabilities 1/3 {prefix['probabilities_3x3_all_one_third']}")
    _log(f"[{tag}] R recorded {R_PREFIX_RECORDED} (W52 R_r2 {row['R_r2']}, rank {rank} of 10; W52 selected "
         f"{sel['market']['selected']})")
    _log(f"[{tag}] facts: B {facts['B_blocks_per_round']} solves/cycle {facts['declared_solve_profile_per_cycle']} min "
         f"pibar {facts['min_hourly_mean_price']:.4f}")
    _log(f"[{tag}] I(x): { {k: v['I_x_eur_master_expression'] for k, v in facts['investment_cost'].items()} }")
    _log(f'[{tag}] checks {checks}')
    _log(f'[{tag}] wrote {INSTANCE_RECORD_REL} sha256 {_sha(INSTANCE_RECORD_REL)}')
    _finish(0 if record['all_checks_pass'] else 1, f'wall={time.time() - started:.1f}s')


def load_instance_record():
    rec = _load(INSTANCE_RECORD_REL)
    derived = H.validate_derived_instance(rec['derived_instance_declaration'])
    return rec, derived


# ======================================================================================================================
#  the memory probe (zero solves)
# ======================================================================================================================
def _probe_case(which, scratch):
    import p515_s44_scale_measurement as S
    if which == 'srp1':
        case, _spec, _changes = S.derive_case('srp1', {})
        path = os.path.join(tempfile.mkdtemp(prefix='w89_case_srp1_', dir=scratch), 'SRP1__srp1.json')
        with open(path, 'w') as handle:
            json.dump(case, handle, indent='\t')
        return os.path.relpath(path, REPO), H.sha256_file(path), False
    if which == 's52_pilot_2x2':
        if _sha(PILOT_2X2['path']) != PILOT_2X2['sha256']:
            raise RuntimeError('the 2 x 2 pilot case does not hash to its pin')
        return PILOT_2X2['path'], PILOT_2X2['sha256'], True
    rec, derived = load_instance_record()
    if _sha(INSTANCE_CASE_REL) != derived['case_sha256']:
        raise RuntimeError('the 3 x 3 case does not hash to its instance record')
    return INSTANCE_CASE_REL, derived['case_sha256'], True


def memory_probe(which, started, scratch):
    """Production's ADMM model build of the UNIT candidate (the larger cell) on one instance, zero solves, RSS per stage;
    then the persist function's pickle (the exact callable the child's post-certification uses) into scratch."""
    import psutil
    import p515_s44_scale_measurement as S
    import p56a_oracle as O
    import shared_resources_planning as srp
    import p515_s42_exact_fix_rerun as EF
    tag = f'W89-PROBE-{which}'
    out_rel = os.path.join(MEMORY_PROBE_DIR_REL, f'memory_probe_{which}.json')
    if os.path.exists(_abs(out_rel)):
        _log(f'[{tag} PRECONDITION FAILED] {out_rel} exists (write-once)')
        _finish(1)
    others = _own_process_alive()
    if others:
        _log(f'[{tag} PRECONDITION FAILED] another copy of this launcher is alive: {others}')
        _finish(1)
    proc = psutil.Process()
    t0 = time.time()
    marks = []

    def mark(stage):
        marks.append({'stage': stage, 't_s': round(time.time() - t0, 3), 'rss_bytes': proc.memory_info().rss,
                      'ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        _log(f"[{tag}] {stage}: rss {marks[-1]['rss_bytes'] / GIB:.3f} GiB, ru_maxrss "
             f"{marks[-1]['ru_maxrss_bytes'] / GIB:.3f} GiB")

    mark('start (launcher, harness and production modules imported)')
    case_rel, case_sha, derived = _probe_case(which, scratch)
    planning, read_dir, _wall = _read(case_rel, scratch, f'probe_{which}')
    mark('planning read (production reader)')
    sed = planning.shared_ess_data
    x = {(n, y): {'s': 0.0, 'e': 0.0} for n in sed.active_distribution_network_nodes for y in sed.years}
    ykey = next(y for y in sed.years if int(y) == YEAR)
    x[(7, ykey)] = {'s': 0.25, 'e': 1.0}
    candidate = O.vector_to_candidate(planning, x)
    if derived:   # the configuration hook's premium write (H._config_hook_factory), for the multi-scenario instances
        planning.params.admm.interface_deviation_premium = dict(PREMIUM)
    premium = planning.params.admm.interface_deviation_premium
    interceptor = S.Interceptor()
    tn = planning.transmission_network
    tn.optimize = interceptor.network(tn, 'tso')
    for dn in planning.distribution_networks.values():
        dn.optimize = interceptor.network(dn, 'dso')
    sed.optimize = interceptor.esso(sed)
    t_build = time.time()
    try:
        cv, _dv = srp.create_admm_variables(planning)
        dso_models, _r1 = srp.create_distribution_networks_models(
            planning.distribution_networks, cv, candidate['total_capacity'], parallel_execution=False,
            premium_alpha=premium['alpha'], premium_floor=premium['floor'])
        tso_model, _r2 = srp.create_transmission_network_model(planning, cv, candidate['total_capacity'])
        esso_model, _r3 = srp.create_shared_energy_storage_model(sed, cv, candidate['investment'])
        srp._prepare_distribution_objectives_for_admm(planning.distribution_networks, dso_models)
        srp._prepare_transmission_objectives_for_admm(tn, tso_model)
    finally:
        del tn.optimize
        for dn in planning.distribution_networks.values():
            del dn.optimize
        del sed.optimize
    build_wall = time.time() - t_build
    models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}
    mark('ADMM models built (network blocks + 3 ESSO; no pristine clones, no solver state)')
    persist_dir = tempfile.mkdtemp(prefix=f'w89_persist_{which}_', dir=scratch)
    t_p = time.time()
    persisted = EF._persist_certified_models(models, persist_dir)
    persist_wall = time.time() - t_p
    mark('certified-model pickle written (EF._persist_certified_models, the child post-certification callable)')
    pkl = _abs(persisted['path'])
    size = os.path.getsize(pkl)
    os.remove(pkl)
    n_blocks = sum(len(v) for v in tso_model.values()) + sum(len(v2) for v in dso_models.values() for v2 in v.values())
    by = {m['stage'].split(' (')[0]: m for m in marks}
    rss_before_pickle = marks[-2]['rss_bytes']
    out = {
        'schema': 'p515_s53_w89_memory_probe_v1', 'stage': STAGE_TEXT, 'which': which, 'utc': _utc(),
        'git_head': H._git(['rev-parse', 'HEAD']), 'script_sha256': H.sha256_file(os.path.abspath(__file__)),
        'case_path': case_rel, 'case_sha256': case_sha,
        'candidate': 'n7_4h_e1 (node 7: 0.25 MVA / 1.0 MWh, 2025) -- the larger of the two cells',
        'premium_in_force': dict(premium), 'interceptor_calls': interceptor.counts(),
        'n_network_blocks_built': n_blocks,
        'scenario_combinations': planning.num_market_scenarios * planning.transmission_network.num_oper_scenarios,
        'marks': marks, 'build_wall_s': build_wall,
        'models_built_rss_bytes': marks[-2]['rss_bytes'],
        'models_increment_over_read_bytes': marks[-2]['rss_bytes'] - by['planning read']['rss_bytes'],
        'pickle': {'size_bytes': size, 'wall_s': persist_wall, 'rss_before_bytes': rss_before_pickle,
                   'ru_maxrss_after_bytes': marks[-1]['ru_maxrss_bytes'],
                   'transient_over_rss_before_bytes': marks[-1]['ru_maxrss_bytes'] - rss_before_pickle,
                   'retained_after_bytes': marks[-1]['rss_bytes'] - rss_before_pickle,
                   'file': 'written to scratch, measured, DELETED (never in the repository)'},
        'not_measured': ('solver state: pristine snapshot clones, multiplier suffixes, SolverResults, AA memory, the '
                         'terminal capture / workbook transients -- they need solves; the memory rule calibrates them '
                         'against measured runtime peaks (RUNTIME_PEAKS)'),
        'read_redirected_to': read_dir,
        'solve_claim': 'ZERO SOLVES, guard-verified (every armed permitted=() guard verify(0))',
        'guards': guards_verify(), 'wall_s': time.time() - started,
    }
    os.makedirs(_abs(MEMORY_PROBE_DIR_REL), exist_ok=True)
    H._write_once_json(_abs(out_rel), out)
    _log(f"[{tag}] {n_blocks} network blocks; models built rss {out['models_built_rss_bytes'] / GIB:.3f} GiB "
         f"(+{out['models_increment_over_read_bytes'] / GIB:.3f} over the read); pickle {size / 1e9:.3f} GB, transient "
         f"+{out['pickle']['transient_over_rss_before_bytes'] / GIB:.3f} GiB, retained "
         f"+{out['pickle']['retained_after_bytes'] / GIB:.3f} GiB; wrote {out_rel} sha256 {_sha(out_rel)}")
    _finish(0, f'wall={time.time() - started:.1f}s')


# ======================================================================================================================
#  the memory rule (from the committed probes and measured runtime peaks)
# ======================================================================================================================
def runtime_peaks():
    p1 = _load(RUNTIME_PEAKS['s52_pilot_2x2']['source'])
    peaks_2x2 = {k: v['peak_rss_bytes'] for k, v in p1['points'].items()}
    w86 = _load(RUNTIME_PEAKS['srp1']['source'])
    srp1 = {}
    for label, cell in w86['per_cell'].items():
        rec = _load(os.path.join(cell['eval_dir'], 'evaluation_record.json'))
        pr = rec.get('peak_rss') or {}
        pkl = os.path.join(cell['eval_dir'], 'certified_models.pkl')
        srp1[label] = {'production_state_before_persist': pr.get('production_state_peak_rss_ru_maxrss'),
                       'child_after_persist': pr.get('child_python_process_ru_maxrss'),
                       'pickle_bytes': os.path.getsize(_abs(pkl)) if os.path.isfile(_abs(pkl)) else None}
    sm = _load(RUNTIME_PEAKS['s52_pilot_2x2_smoke']['source'])
    cost = (sm.get('response_terminal') or {}).get('capture_cost') or {}
    smoke = {'child_peak': (sm.get('peak_rss') or {}).get('child_python_process_ru_maxrss'),
             'production_state_peak': (sm.get('peak_rss') or {}).get('production_state_peak_rss_ru_maxrss'),
             'rss_at_terminal_capture_start': cost.get('rss_before_bytes')}
    avail = []
    for rel, keys in AVAILABILITY_OBSERVED:
        m = _load(rel)
        for k in keys:
            m = m.get(k) or {}
        avail.append({'source': rel, 'utc': m.get('utc'), 'available_bytes': m.get('available_bytes'),
                      'available_gib': m.get('available_gib'), 'concurrency_running': m.get('concurrency')})
    return {'s52_pilot_2x2_alpha_row_pair1': peaks_2x2, 'srp1_w86': srp1, 's52_pilot_2x2_smoke_r2': smoke,
            'availability_observed': avail}


MEMORY_RULE_TEXT = (
    'Measured inputs: the models-built RSS of the UNIT candidate on SRP1, the 2 x 2 and the 3 x 3 instance (the three '
    'zero-solve probes, one code path); the committed runtime peaks (RUNTIME_PEAKS); the recorded availability at every '
    'committed multi-child launch (AVAILABILITY_OBSERVED). Derived: SUSTAINED (per-child peak, no persist) = build_3x3 x '
    'k_run, k_run = max 2 x 2 alpha-row pair-1 child peak / build_2x2; SMOKE = build_3x3 x k_smoke, k_smoke = the 2 x 2 '
    'smoke r2 child peak (alone, cap 2) / build_2x2; PERSIST_PEAK (per-child peak with persist) = max(SUSTAINED, f_term x '
    'SUSTAINED + k_tr x T_3x3), f_term = the 2 x 2 smoke\'s RSS at the start of its terminal capture / its production-state '
    'peak, T = a probe\'s pickle transient (ru_maxrss after the pickle - RSS before it), k_tr = max over the three W86 SRP1 '
    'cells of (child peak after persist - f_term x production-state peak) / T_srp1 (the real models carry solution '
    'suffixes the zero-solve pickle lacks). Decisions, taken at --freeze-spec and recorded in the stage spec: concurrency 2 iff 2 x '
    'SUSTAINED <= hw.memsize (else 1); persist iff PERSIST_PEAK (x c) <= the BEST recorded availability (else '
    'persist_certified_models False -- the Planner\'s "unless the memory preflight forbids it"). Gate at --run (refusing): '
    'available (hw.memsize - (wired + anonymous + compressor-occupied) x page size, L.memory_preflight) >= smoke: SMOKE; '
    'pair: c x (PERSIST_PEAK if persist else SUSTAINED). RSS is pressure-dependent on macOS (compressed pages leave '
    'RSS): the 2 x 2 pair ran two children at concurrency 2 under pressure; the smoke r2 ran alone -- both enter.')


def memory_model():
    probes = {}
    for w in MEMORY_PROBES:
        rel = os.path.join(MEMORY_PROBE_DIR_REL, f'memory_probe_{w}.json')
        if not os.path.isfile(_abs(rel)):
            return None, f'memory probe {w} missing: {rel}'
        st = _git_state(rel)
        if not (st['git_tracked'] and st['git_clean']):
            return None, f'memory probe {w} not committed / clean'
        probes[w] = {'path': rel, 'sha256': _sha(rel), **_load(rel)}
    peaks = runtime_peaks()
    build = {w: probes[w]['models_built_rss_bytes'] for w in probes}
    trans = {w: probes[w]['pickle']['transient_over_rss_before_bytes'] for w in probes}
    hw = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, check=True).stdout)
    k_run = max(peaks['s52_pilot_2x2_alpha_row_pair1'].values()) / build['s52_pilot_2x2']
    k_run_srp1 = max(v['production_state_before_persist'] for v in peaks['srp1_w86'].values()) / build['srp1']
    sm = peaks['s52_pilot_2x2_smoke_r2']
    k_smoke = sm['child_peak'] / build['s52_pilot_2x2']
    f_term = sm['rss_at_terminal_capture_start'] / sm['production_state_peak']
    k_tr = max((v['child_after_persist'] - f_term * v['production_state_before_persist']) / trans['srp1']
               for v in peaks['srp1_w86'].values())
    sustained = build['s53_3x3'] * k_run
    smoke = build['s53_3x3'] * k_smoke
    persist_peak = max(sustained, f_term * sustained + k_tr * trans['s53_3x3'])
    best = max(a['available_bytes'] for a in peaks['availability_observed'] if a['available_bytes'])
    concurrency = 2 if 2 * sustained <= hw else 1
    persist = concurrency * persist_peak <= best
    model = {
        'rule': MEMORY_RULE_TEXT,
        'probes': {w: {k: probes[w][k] for k in ('path', 'sha256', 'models_built_rss_bytes',
                                                  'models_increment_over_read_bytes', 'n_network_blocks_built',
                                                  'scenario_combinations', 'pickle', 'build_wall_s')} for w in probes},
        'runtime_peaks_measured': peaks, 'hw_memsize_bytes': hw,
        'k_run': k_run, 'k_run_srp1_low': k_run_srp1, 'k_smoke': k_smoke, 'f_term': f_term, 'k_tr': k_tr,
        'sustained_bytes': sustained, 'sustained_gib': sustained / GIB,
        'sustained_range_gib': [build['s53_3x3'] * k_run_srp1 / GIB, sustained / GIB],
        'smoke_bytes': smoke, 'smoke_gib': smoke / GIB,
        'persist_peak_bytes': persist_peak, 'persist_peak_gib': persist_peak / GIB,
        'pickle_3x3_zero_solve_bytes': probes['s53_3x3']['pickle']['size_bytes'],
        'pickle_3x3_estimate_bytes': probes['s53_3x3']['pickle']['size_bytes'] * max(
            v['pickle_bytes'] for v in peaks['srp1_w86'].values()) / probes['srp1']['pickle']['size_bytes'],
        'best_available_observed_bytes': best, 'best_available_observed_gib': best / GIB,
        'decision': {'concurrency': concurrency,
                     'concurrency_reason': (f'2 x SUSTAINED = {2 * sustained / GIB:.2f} GiB '
                                            f"{'<=' if concurrency == 2 else '>'} hw.memsize {hw / GIB:.2f} GiB"),
                     'persist_certified_models': persist,
                     'persist_reason': (f'{concurrency} x PERSIST_PEAK = {concurrency * persist_peak / GIB:.2f} GiB '
                                        f"{'<=' if persist else '>'} best recorded availability {best / GIB:.2f} GiB"),
                     'requested_by_planner': POST_CERTIFICATION_REQUESTED},
        'required_at_run_gib': {'smoke': smoke / GIB,
                                'pair': concurrency * (persist_peak if persist else sustained) / GIB},
    }
    return model, None


def post_certification_decided(ss):
    return dict(ss['configuration']['post_certification'])


def post_certification_resolved(ss, candidate_key_hex):
    """The decided request as the harness resolves it into a spec entry (None when nothing is requested)."""
    return H.resolve_post_certification(post_certification_decided(ss), candidate_key_hex)


def memory_preflight(stage, model, ss=None):
    m = L.memory_preflight(1)
    if stage == 'smoke':
        required = model['smoke_bytes']
        concurrency = 1
    else:
        concurrency = int(ss['configuration']['concurrency'])
        persist = post_certification_decided(ss)['persist_certified_models']
        required = concurrency * (model['persist_peak_bytes'] if persist else model['sustained_bytes'])
    m = {**m, 'stage': stage, 'concurrency': concurrency, 'required_bytes': required, 'required_gib': required / GIB,
         'rule': MEMORY_RULE_TEXT, 'w86_rule_fields_superseded': ['required_bytes', 'required_gib', 'rule',
                                                                   'per_child_budget_gib', 'concurrency']}
    m['pass'] = m.get('available_bytes') is not None and m['available_bytes'] >= required
    return m


def _memory_line(m):
    if m.get('available_gib') is None:
        return f"vm_stat figures missing {m.get('missing_vm_stat_figures')}"
    return (f"available {m['available_gib']:.2f} GiB (free+inactive {m['free_plus_inactive_gib']:.2f}, recorded only); "
            f"required {m['required_gib']:.2f} GiB at concurrency {m['concurrency']}")


# ======================================================================================================================
#  keys and the pre-launch assertion
# ======================================================================================================================
def _nodes(label):
    return {n: tuple(map(float, v)) for n, v in CELLS[label].items()}


def _key_of(label):
    return H.candidate_key(H.canonical_candidate(_nodes(label), investment_year=YEAR))


_TAIL_DEFAULT = object()


def _eval_key(label, derived, tail=_TAIL_DEFAULT):
    return H.evaluation_key(_key_of(label), {}, case_file_aa=CASE_FILE_AA, ess_ageing_baseline=ESS_AGEING_BASELINE,
                            derived_instance=derived,
                            interface_deviation_premium=PREMIUM,
                            convergence_depth_tail=TAIL if tail is _TAIL_DEFAULT else tail)


def campaign_root(stage):
    return _abs(os.path.join(ROOT_REL, f"campaign_{STAGES[stage]['campaign_id']}"))


def pre_launch_assertion(derived, spec=None):
    """Every cell's eval key, recomputed now with every declaration: (a) differs from the same cell WITHOUT the tail
    (a pre-tail 3 x 3 evaluation) and the declared-OFF key equals that pre-tail key (W86 K-check form); (b) differs from
    the 2 x 2 alpha-row key of the same candidate (the pre-3x3 evaluation); (c) is absent from every committed campaign
    spec outside this launcher's two roots; and, for a frozen spec, its entry carries exactly the recomputed key. The
    smoke's x0 shares the pair's x0 key by design (same candidate x configuration; cap 2, never certified)."""
    own = (ROOT_REL,)   # every campaign root of this launcher (incl. the superseded r1 freezes, same keys by design)
    committed = L.committed_eval_keys(exclude_roots=own)
    spec_labels = {e['label'] for e in (spec or {}).get('candidates') or []}
    alpha_pair1 = _load(ALPHA_ROW_PAIR1['path'])
    per = {}
    for label in STAGES['pair']['labels']:
        now = _eval_key(label, derived)
        pre_tail = _eval_key(label, derived, tail=None)
        declared_off = _eval_key(label, derived, tail={'enabled': False, 'compl_inf_tol': 1e-6})
        a_label = {'x0': 'x0_a0p50', 'n7_4h_e1': 'n7_4h_e1_a0p50'}[label]
        a_dir = (alpha_pair1['points'].get(a_label) or {}).get('eval_dir') or ''
        a_key = _load(os.path.join(a_dir, 'evaluation_record.json')).get('eval_key') if a_dir else None
        entries = [e for e in (spec or {}).get('candidates') or [] if e['label'] == label]
        per[label] = {
            'eval_key': now, 'candidate_key': _key_of(label),
            'pre_tail_3x3_key': pre_tail, 'differs_from_pre_tail_3x3_key': now != pre_tail,
            'declared_off_key_equals_pre_tail_key': declared_off == pre_tail,
            'pre_tail_3x3_key_in_committed_specs': committed.get(pre_tail, []),
            'alpha_row_2x2_key_same_candidate': a_key,
            'alpha_row_2x2_key_prefix_as_pinned': (a_key or '').startswith(ALPHA_ROW_2X2_KEYS[label]),
            'differs_from_2x2_alpha_row_key': a_key is not None and now != a_key,
            'absent_from_every_committed_spec_outside_w89': now not in committed,
            'committed_specs_holding_it_outside_w89': committed.get(now, []),
            'frozen_entry_eval_key': entries[0].get('eval_key') if entries else None,
            'frozen_entry_equals_recomputed': (not entries) if (spec is None or label not in spec_labels) else (
                len(entries) == 1 and entries[0].get('eval_key') == now),
            'label_in_frozen_spec': (label in spec_labels) if spec is not None else None,
        }
    holds = all(v['differs_from_pre_tail_3x3_key'] and v['declared_off_key_equals_pre_tail_key']
                and v['alpha_row_2x2_key_prefix_as_pinned'] and v['differs_from_2x2_alpha_row_key']
                and v['absent_from_every_committed_spec_outside_w89'] and v['frozen_entry_equals_recomputed']
                for v in per.values())
    return {'per_cell': per, 'holds': holds, 'n_committed_keys_scanned': len(committed),
            'excluded_roots': list(own)}


# ======================================================================================================================
#  rule eleven: a capture path for every quantity the stage spec requires, asserted BEFORE any run
# ======================================================================================================================
def launcher_checklist():
    child_src = inspect.getsource(H._child_real)
    w86 = L.launcher_checklist()             # the tail / floor-record / append / persist capture (W86 rule eleven)
    alpha = A.launcher_checklist()           # the alpha-row capture and formulas (W64 / W65 rule eleven)
    x_src = inspect.getsource(X.g6_v32_evaluate_records)
    checks = {
        'alpha_row_capture_on_for_derived_instance': 'capture_multiscenario = derived is not None or premium is not None'
                                                     in child_src,
        'per_cycle_response_merged_into_per_cycle_record': ('per_cycle_fields = PER_CYCLE_RECORD_FIELDS if '
                                                             'capture_multiscenario' in child_src),
        'floor_status_capture_every_evaluation': 'persist_convergence_depth_capture(state, eval_dir)' in child_src,
        'derived_instance_installed_before_anything': (0 <= child_src.find('install_derived_instance(derived, eval_dir)')
                                                       < child_src.find('instance_investment_years()')),
        'tail_declaration_read_from_spec': 'assert_convergence_depth_tail_capture(spec)' in child_src,
        'post_certification_persist_in_terminal_phase_under_lock': (
            child_src.find('acquire_terminal_phase_lock(terminal_lock_path) if capture_multiscenario')
            < child_src.find('run_post_certification(')),
        'g6_v32_block_rules': all(t in x_src for t in ('judge_block(', 'non_vacuous_v32(', 'per_record_failures(')),
        'acceptance_cross_check_defined': 'expected_event_class(' in inspect.getsource(X.acceptance_cross_check),
        'cell_formulas_alpha_row': 'def cell_quantities' in inspect.getsource(A),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (W89 3x3 launcher): capture paths missing: {missing}')
    return {'launcher': checks, 'w86_launcher': w86['checks'], 'alpha_row_launcher': alpha['launcher'],
            'harness_alpha_row': alpha['harness_alpha_row'], 'harness_record': w86['harness_record_capture_checklist']}


# ======================================================================================================================
#  the per-cell gates (pair) and the smoke gate
# ======================================================================================================================
def _solve_profile_check(rec, n_records):
    sp = rec.get('solve_profile') or {}
    obs = sp.get('observed') or {}
    rounds = (rec.get('cycles_run') or 0) + 1
    ok = (sp.get('reconciliation_supported') is True and sp.get('identity_holds') is True
          and sp.get('solves_per_cycle') == SOLVES_PER_ROUND and sp.get('rounds') == rounds
          and sp.get('base_solves') == SOLVES_PER_ROUND * rounds and obs.get('blocked_solve') == 0
          and obs.get('blocked_exec') == 0 and obs.get('permitted_solve') == sp.get('expected_solves')
          and n_records == obs.get('permitted_solve') - N_ESSO * rounds)
    return ok, {'solve_profile': sp, 'n_network_records': n_records, 'rounds': rounds}


def alpha_row_capture_check(rec, eval_dir, require_compared=False):
    written = {k: (rec.get(k) or {}).get('status') for k in ('multiscenario_terminal', 'operational_workbook',
                                                             'response_terminal')}
    pcr = rec.get('per_cycle_response') or {}
    di = rec.get('derived_instance_installed_in_child') or {}
    prem = rec.get('interface_deviation_premium_applied_in_child') or {}
    ii = rec.get('initialisation_identity') or {}
    ar = rec.get('activation_readback') or {}
    checklist = rec.get('alpha_row_capture_checklist_asserted_before_run') or {}
    parts = {
        'written_all_three': all(v == 'written' for v in written.values()),
        'multiscenario_checks_pass': (rec.get('multiscenario_terminal') or {}).get('all_checks_pass') is True,
        'activation_readback_all_ok': ar.get('all_ok') is True,
        'per_cycle_response_every_cycle': (pcr.get('cycles_match') is True
                                           and pcr.get('n_lines') == pcr.get('n_trajectory_rows') == pcr.get('n_captured')
                                           == rec.get('cycles_run')),
        'derived_checksum_in_child': (di.get('scenario_checksum_in_child') is not None and di.get(
            'scenario_checksum_in_child') == (rec.get('derived_instance') or {}).get('scenario_checksum')),
        'premium_applied_alpha_0p5': (prem.get('took_effect') is True and (prem.get('after') or {}).get('alpha') == 0.5
                                      and prem.get('declared') == PREMIUM),
        'initialisation_identity_recorded_and_equal': (bool(ii.get('gross_operational_cost_hex'))
                                                       and ii.get('all_equal') is True
                                                       and (not require_compared or (ii.get('n_compared') or 0) >= 1)),
        'capture_checklist_asserted_before_run': bool(checklist) and all(
            v for v in checklist.values() if isinstance(v, bool)),
        'response_terminal_file_present': os.path.isfile(os.path.join(eval_dir, H.RESPONSE_TERMINAL_FILE)),
        'multiscenario_terminal_file_present': os.path.isfile(os.path.join(eval_dir, H.MULTISCENARIO_TERMINAL_FILE)),
    }
    return all(parts.values()), {'parts': parts, 'written': written, 'per_cycle_response': pcr,
                                 'premium_applied': prem, 'activation_readback_all_ok': ar.get('all_ok'),
                                 'initialisation_identity': ii}


def sigma_check(rec):
    s = rec.get('sigma_calibration') or {}
    comp, fixed, band = s.get('sigma_computed'), s.get('sigma_fixed'), s.get('band')
    ratio = (comp / fixed) if (_finite(comp) and _finite(fixed) and fixed) else None
    ok = ratio is not None and bool(band) and band[0] <= ratio <= band[1]
    return ok, {'sigma_computed': comp, 'sigma_fixed': fixed, 'ratio': ratio, 'band': band}


def post_certification_check(rec, eval_dir, ss):
    """G8 under the stage-spec decision: certified -> status evaluated, and the pickle written with the record's sha256 iff
    persist was decided, absent otherwise; not certified -> skipped, no pickle."""
    pc = rec.get('post_certification') or {}
    pkl = os.path.join(eval_dir, 'certified_models.pkl')
    persist = post_certification_decided(ss)['persist_certified_models']
    requested = post_certification_resolved(ss, rec.get('candidate_key'))
    if requested is None:   # nothing requested: no post-certification step at all, no pickle
        ok = rec.get('post_certification') is None and not os.path.exists(pkl)
    elif rec.get('status') == 'certified':
        if persist:
            ok = (pc.get('status') == 'evaluated' and os.path.isfile(pkl)
                  and (pc.get('persisted_models') or {}).get('sha256') == H.sha256_file(pkl))
        else:
            ok = pc.get('status') == 'evaluated' and not os.path.exists(pkl) and not pc.get('persisted_models')
    else:
        ok = pc.get('status') == 'skipped' and not os.path.exists(pkl)
    return ok, {'post_certification': pc, 'persist_decided': persist,
                'pickle': ({'bytes': os.path.getsize(pkl), 'sha256': H.sha256_file(pkl)} if os.path.isfile(pkl) else None)}


def cell_gates(entry, eval_dir, ss):
    """The pair's per-cell gates G1-G11 (module docstring), files only."""
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    gates, detail = {}, {'exit_code': exit_code}
    gates['G1_harness_clean'] = (exit_code == 0 and bool(rec) and rec.get('status') in ('certified', 'not_certified')
                                 and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json')))
    if not rec:
        return gates, detail, rec
    gates['G2_eval_key'] = rec.get('eval_key') == ss['cells'][entry['label']]['eval_key'] == entry['eval_key']
    c, d = L.evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'])
    detail['w86_evaluation_checks'] = {'checks': c, 'detail': d,
                                       'note': ('W86 solve-profile and floor-record checks are SRP1-specific / v29 and '
                                                'are superseded by G5 and G6 below (reported only)')}
    gates['G3_append_reconcile'] = c.get('append_reconciles_byte_identical', False)
    gates['G4_tail_state_check'] = c.get('tail_state_check_matches', False)
    records = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    gates['G5_solve_profile_reconciled_per_event'], detail['G5'] = _solve_profile_check(rec, len(records))
    g6 = X.g6_v32_evaluate(eval_dir, rec, BLOCKS_PER_ROUND)
    gates['G6_floor_records_v32'] = g6['gate_pass']
    detail['G6'] = g6
    gates['G7_append_sealed'] = c.get('append_sealed_after_reconcile', False)
    gates['G8_post_certification'], detail['G8'] = post_certification_check(rec, eval_dir, ss)
    gates['G9_ess_ageing_readback'] = c.get('ess_ageing_readback_all_match', False)
    gates['G10_alpha_row_capture'], detail['G10'] = alpha_row_capture_check(rec, eval_dir,
                                                                            require_compared=entry['label'] == 'x0')
    xc = X.acceptance_cross_check(eval_dir)
    gates['G11_acceptance_cross_check'] = xc['agrees']
    detail['G11'] = xc
    return gates, detail, rec


def smoke_gate_checks(entry, eval_dir, pair_x0_key, ss):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    checks, detail = {}, {'exit_code': exit_code}
    checks['S1_child_exit0_record_uncertified_2_cycles'] = (
        exit_code == 0 and bool(rec) and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json'))
        and rec.get('status') == 'not_certified' and rec.get('cycles_run') == 2)
    if not rec:
        return checks, detail, rec
    c, d = L.evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'], expect_certified=False)
    detail['w86_evaluation_checks'] = {'checks': c, 'detail': d}
    checks['S2_append_byte_identical'] = c.get('append_reconciles_byte_identical', False)
    checks['S3_checklist_line1_before_any_solve'] = c.get('checklist_line1_before_any_solve', False)
    checks['S4_tail_state_check'] = c.get('tail_state_check_matches', False)
    records = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    ok5, detail['S5'] = _solve_profile_check(rec, len(records))
    checks['S5_solve_profile_249_plus_retries'] = ok5 and (rec.get('solve_profile') or {}).get('base_solves') == 249
    events = _read_jsonl(os.path.join(eval_dir, H.CONVERGENCE_DEPTH_APPEND_EVENTS_FILE))
    ts = json.load(open(os.path.join(eval_dir, H.CONVERGENCE_DEPTH_TAIL_STATE_FILE)))
    applies = [e for e in events if e.get('event') == 'apply']
    nexts = [e for e in events if e.get('event') == 'next_state']
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    stdout_text = open(os.path.join(eval_dir, 'child_stdout.log')).read()
    per_cycle = ts.get('per_cycle') or []
    detail['S6'] = {'apply_cycles': [e.get('cycle') for e in applies],
                    'apply_active': [e['record'].get('active') for e in applies],
                    'next_state_values': [e.get('value') for e in nexts],
                    'row_cycle_convergence': [r.get('cycle_convergence') for r in rows],
                    'tail_on_lines_in_child_stdout': stdout_text.count('Convergence-depth tail ON')}
    checks['S6_tail_inactive_at_cap2'] = (
        len(per_cycle) == 2 and not any(p.get('active') or p.get('acted') for p in per_cycle)
        and (ts.get('restore_at_exit') or {}).get('acted') is False and detail['S6']['apply_cycles'] == [1, 2, None]
        and not any(detail['S6']['apply_active']) and detail['S6']['next_state_values'] == [False, False]
        and not any(detail['S6']['row_cycle_convergence']) and detail['S6']['tail_on_lines_in_child_stdout'] == 0)
    prod = L._production_compl_inf_tol(ts.get('baseline'))
    bad = [{**X._short(r), 'failures': X.per_record_failures(r, set(), prod)} for r in records
           if X.per_record_failures(r, set(), prod)]
    detail['S7'] = {'n_records': len(records), 'n_bad': len(bad), 'bad_first': bad[:10], 'production': prod,
                    'records_per_round': dict(sorted(Counter(r.get('round') for r in records).items())),
                    'floor_status_tally': dict(sorted(Counter(f"{r.get('agent')}|{r.get('floor_status')}"
                                                              for r in records).items())),
                    'attempts': dict(sorted(Counter(r.get('attempt') for r in records).items()))}
    checks['S7_records_pass_v32_per_record_predicate_at_production_tol'] = bool(records) and not bad
    checks['S8_append_sealed'] = c.get('append_sealed_after_reconcile', False)
    checks['S9_post_certification_as_stage_spec'], detail['S9'] = post_certification_check(rec, eval_dir, ss)
    checks['S10_eval_key_equals_pair_x0'] = rec.get('eval_key') == entry['eval_key'] == pair_x0_key
    checks['S11_ess_ageing_readback'] = c.get('ess_ageing_readback_all_match', False)
    checks['S12_alpha_row_capture'], detail['S12'] = alpha_row_capture_check(rec, eval_dir)
    checks['S13_sigma_inside_band'], detail['S13'] = sigma_check(rec)
    return checks, detail, rec


# ======================================================================================================================
#  the stage spec (v34)
# ======================================================================================================================
PER_ENTRY_GATES = {
    'applies_to': 'both pair cells (x0, n7_4h_e1); there is no other arm',
    'G1_harness_clean': 'child exit 0, evaluation_record.json written by the child (no parent barrier record), status '
                        'certified or not_certified',
    'G2_eval_key': 'record eval_key == the stage-spec key == the frozen entry key',
    'G3_append_reconcile': 'W86 G3 verbatim (L.evaluation_checks append_reconciles_byte_identical)',
    'G4_tail_state_check': 'W86 G4 verbatim (convergence_depth_tail_state_check.match, enabled, 1e-6)',
    'G5_solve_profile_reconciled_per_event': ('reconciliation_supported, identity_holds, solves_per_cycle == 83, rounds '
                                              '== cycles_run + 1, base == 83 x rounds, observed == expected (base + '
                                              'retries attempted), 0 blocked; network floor records == observed - 3 x '
                                              'rounds'),
    'G6_floor_records_v32': ('frozen spec v32 G6_floor_records_v32 verbatim with B = 80 '
                             '(p515_s53_w89_g6_final_attempt_reeval.g6_v32_evaluate, by import)'),
    'G7_append_sealed': 'W86 G7 verbatim',
    'G8_post_certification': ('as the stage spec decided, resolved by H.resolve_post_certification: nothing requested '
                              '(persist forbidden, no polish) -> the record carries no post-certification and no '
                              'certified_models.pkl exists; persist requested -> certified: status evaluated and the '
                              'pickle written with sha256 == the record\'s; not certified: skipped, no pickle'),
    'G9_ess_ageing_readback': 'W86 G9 verbatim (all_match before and after the run)',
    'G10_alpha_row_capture': ('multiscenario_terminal / operational_workbook / response_terminal status "written"; '
                              'multiscenario all_checks_pass; activation_readback all_ok; per_cycle_response n_lines == '
                              'n_trajectory_rows == n_captured == cycles_run and cycles_match; the child\'s scenario '
                              'checksum == the declared one; premium took effect at alpha 0.5; initialisation identity '
                              'recorded and bitwise equal to every record of the same candidate in the root (x0: '
                              'compared with >= 1 -- the smoke\'s x0 initialisation record, placed as reference before '
                              'the run, the alpha-row precedent); the alpha-row capture checklist asserted before the '
                              'run; the terminal files present'),
    'G11_acceptance_cross_check': 'v32 ACCEPTANCE_CROSS_CHECK on this cell: 0 mismatches, every round',
    'fallback': ('NOT APPLICABLE: there is no prior 3 x 3 evaluation to move; a cell that fails to certify or fails a gate '
                 'is reported as such (exit 2 / 1) and the Planner decides'),
}
SMOKE_GATE = {k: v for k, v in {
    'S1': 'child exit 0; record written by the child; status not_certified; cycles_run 2',
    'S2': 'W86 S2: append file byte-identical to the end-of-run file, sha256 == the record\'s, rounds 0..2',
    'S3': 'W86 S3: the tail checklist is line 1 of the events file, written before any IPOPT output file and drain',
    'S4': 'W86 S4: tail state check match (enabled, 1e-6)',
    'S5': 'solve profile reconciled per event: base 83 x 3 = 249, observed == 249 + retries attempted, 0 blocked, '
          'network records == observed - 9',
    'S6': 'W86 S6: the tail INACTIVE at cap 2 (no active / acted cycle, apply x3 inactive, next_state [False, False], no '
          'cycle_convergence row, no "Convergence-depth tail ON" line)',
    'S7': 'every record passes v32\'s per-record predicate with W empty (production compl_inf_tol: TSO case9 5e-4, DSO '
          'IPOPT default 1e-4; options list agrees; parse_reason None, or a ruling-2 tier-2 declaration)',
    'S8': 'W86 S8: last event sealed, no append write error',
    'S9': ('post-certification as the stage spec decided: nothing requested -> the record carries none and no '
           'certified_models.pkl; persist requested -> uncertified at cap 2 -> skipped, no pickle'),
    'S10': 'record eval_key == the smoke entry key == the pair x0 key',
    'S11': 'ESS ageing read-back all_match before and after',
    'S12': 'G10 (the alpha-row capture) at cap 2',
    'S13': 'sigma_calibration: sigma_computed / sigma_fixed inside production\'s band [1/3, 3]',
    'S14': 'every parent guard verify(0) == []',
}.items()}
SMOKE_REPORTED = ('peak RSS (child ru_maxrss, production state), wall per cycle (child), records per round, floor-status '
                  'tally, exits, retries -- the first measured 3 x 3 cycle time and RSS, fed back into the pair estimate')

OBJECTIVE_CONVENTION = A.OBJECTIVE_CONVENTION


def predictions(model, inst, refs):
    return {
        'recorded': ('BEFORE any 3 x 3 evaluation exists; NOT a blind test of the instance facts (the Worker ran the '
                     'instance freeze and the memory probes, whose results are recorded in this spec, before writing '
                     'the predictions) but blind to every 3 x 3 ADMM quantity: no 3 x 3 solve has ever run'),
        'author_expert_verbatim': {
            'addendum_44': ('3 x 3 confirmed with production\'s prefix draw [1,2,3] x [1,2,3] and R = 0.9331 recorded '
                            '(reproducibility over a 0.6 % refinement); 3 x 3 prediction band [0.93, 1.09] recorded with '
                            'the resolution caveat'),
            'report_section_10': ('R(3x3) in [0.93, 1.09], centred on the 2 x 2 result of 1.031, on the mean-profile '
                                  'argument; value - I remains negative at 3 x 3 under the baseline flexibility price, '
                                  'determinate at >= 2 x the four-cell bar; cycle time ~ 9 min, ~16 h per evaluation'),
        },
        'R_definition': ('ratio = value_3x3 / R_SRP1_tail, value_3x3 = Q(x0) - Q(n7_4h_e1) on this instance (Q = '
                         'certified_cost, gross, settlement excluded), R_SRP1_tail = the SRP1 x0 - unit value re-run '
                         'under the tight tail (W86; reeval_w89.json R_tail)'),
        'R_reference': refs,
        'P1_ratio_band': ('ratio in [0.93, 1.09] (Addendum 44 band, restated against 259,375.33; the reference moved by '
                          '-52.44 = -2.0e-4 relative, 1/760 of the 2 x 2 resolution, so the band is not re-derived)'),
        'P1_resolution_caveat': ('at 2 x 2 the ratio\'s resolution was 0.152; a 3 x 3 point inside the band confirms '
                                 'nothing sharper than "not distinguishable". The 3 x 3 resolution is reported as '
                                 'hypot((bar_x0 + bar_unit) / R_SRP1_tail, value x bar_sum_SRP1 / R_SRP1_tail^2), the '
                                 'alpha-row formula'),
        'P1_mean_profile_R': ('R = 0.9331 (W52 R_r2 of [1, 2, 3]) is the first-order price-spread prediction for this '
                              'draw; recorded beside the band, not a second test'),
        'W1_worker_point': ('ratio ~ 0.97 (68 % interval [0.90, 1.05]): the 2 x 2 value 267,548.82 (ratio 1.0315 against '
                            '259,375.33) carried the 15,066 incomplete-convergence excess the Addendum 46 decomposition '
                            'attributed to fifteen x0 TSO solves stopping above their mu floor; with the tight tail that '
                            'excess should vanish (~252,480, ratio ~0.973), and the mean-profile argument puts the '
                            'price part at 0.933 x SRP1'),
        'W2_value_minus_I_negative': ('value - I < 0 and determinate (|value - I| > bar_x0 + bar_unit) at 3 x 3, I = '
                                      f"{inst['facts']['investment_cost']['n7_4h_e1']['I_x_eur_master_expression']:.2f}"),
        'W3_certifies': ('both cells certify within cap 500 (0.85); certification cycle 60-95 (2 x 2 alpha row 55-74, '
                         'SRP1 87-132; the tail adds ~0-3 cycles)'),
        'W4_gates': ('every per-cell gate holds on both cells; G6 v32 terminal round 80 blocks, all judged attempts '
                     'primary and "at"; the tail active on the last ~9 cycles'),
        'W5_retries': 'retry solves per cell 0-10 (2 x 2 unit 4, x0 0; SRP1 C* 40)',
        'W6_rule_ten': 'terminal gross step / objective tolerance < 0.25 on both cells (2 x 2 alpha row <= 0.066)',
        'W7_sigma': ('sigma_computed / sigma_fixed in [0.49, 0.61] (2 x 2 0.498, paper 5 x 5 0.597), inside production\'s '
                     '[1/3, 3] band -- the smoke measures it first'),
        'W8_wall_memory_disk': 'as the estimates below',
    }


def estimates(model, concurrency, persist):
    per_cycle = {'srp1': {'units': 48, 's': 32.0}, '2x2': {'units': 320, 's': 195.0}, 'paper': {'units': 2000, 's': 2538.0}}
    units = BLOCKS_PER_ROUND * N_SCEN * N_SCEN
    lin = per_cycle['2x2']['s'] + (units - 320) * (per_cycle['paper']['s'] - per_cycle['2x2']['s']) / (2000 - 320)
    expo = math.log(per_cycle['paper']['s'] / per_cycle['2x2']['s']) / math.log(2000 / 320)
    powr = per_cycle['2x2']['s'] * (units / 320) ** expo
    lo_s, hi_s = min(lin, powr), max(lin, powr)
    return {
        'cycle_time': {'inputs_measured': {'SRP1': '32 s/cycle at 48 x 1', '2x2': ('~190 s/cycle at 80 x 4 (alpha row '
                                                                                  'pair 1: 184.5 / 214.9 s, concurrency '
                                                                                  '2)'),
                                           'paper': '42.3 min/cycle at 80 x 25 (one cycle, concurrency 1)'},
                       'units_3x3': units, 'power_law_exponent_2x2_to_paper': expo,
                       'power_law_s': powr, 'linear_s': lin, 'range_min': [lo_s / 60, hi_s / 60],
                       'note': 'the smoke measures the 3 x 3 cycle time before the pair runs'},
        'wall_per_evaluation_h': {'cycles_assumed': [60, 95], 'central_cycles': 75,
                                  'central_h': [75 * lo_s / 3600, 75 * hi_s / 3600],
                                  'range_h': [60 * lo_s / 3600, 95 * hi_s / 3600],
                                  'plus_terminal_phase_min': '~15-30 (capture, workbook, pickle; serialized by the lock)'},
        'wall_pair_h': ({'concurrency': concurrency,
                         'central_h': [75 * lo_s / 3600 * (1 if concurrency == 2 else 2),
                                       75 * hi_s / 3600 * (1 if concurrency == 2 else 2)]}),
        'smoke_wall_min': [(3 * lo_s + 300) / 60, (3 * hi_s + 600) / 60],
        'memory': {k: model[k] for k in ('sustained_gib', 'sustained_range_gib', 'persist_peak_gib', 'smoke_gib',
                                         'best_available_observed_gib', 'required_at_run_gib', 'k_run',
                                         'k_run_srp1_low', 'k_smoke', 'f_term', 'k_tr', 'decision')},
        'disk': {'per_cell_eval_dir_gb': ('~0.5-0.65 (2 x 2 alpha row 0.22-0.29 GB, capture files scale ~x2.25 with '
                                          'the scenario count)'),
                 'per_cell_certified_models_pkl_gb': (model['pickle_3x3_estimate_bytes'] / 1e9 if persist else 0.0),
                 'pickle_if_persisted_gb': model['pickle_3x3_estimate_bytes'] / 1e9,
                 'per_cell_oracle_working_dir_gb': '~1.5-2.5 (2 x 2 0.78-0.99 GB of IPOPT logs; x ~2 per solve)',
                 'total_pair_gb': ('~2 x (0.6 + pickle + 2.5) -- see per-cell figures; the machine has ~425 GB free; '
                                   'only the eval dirs minus the pickles are committed')},
    }


def _find_stage_spec():
    hits = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_PREFIX) and f.endswith('.json'))
    if len(hits) != 1:
        return None, None
    rel = os.path.join(_P53, hits[0])
    sha = _sha(rel)
    if hits[0] != f'{SPEC_PREFIX}{sha[:8]}.json':
        raise RuntimeError(f'v{SPEC_VERSION} file name does not carry its sha256 prefix: {rel} {sha}')
    return rel, sha


def _reference_R():
    r = _load(REEVAL_W89['path'])
    cells = r['per_cell']
    bar_x0, bar_unit = cells['x0']['comparison']['bar_tail'], cells['n7_4h_e1']['comparison']['bar_tail']
    return {'source': REEVAL_W89, 'R_SRP1_tail': r['R_from_w87_reproduced']['R_tail'],
            'R_SRP1_pre_tail_superseded': r['R_from_w87_reproduced']['R_ref'],
            'bar_x0_tail': bar_x0, 'bar_unit_tail': bar_unit, 'bar_sum_SRP1_tail': bar_x0 + bar_unit,
            'bar_caveat': ('bar_tail == bar_ref bitwise on both SRP1 cells (W88 caveat: the two-run bar is not an '
                           'independent measure); used here as the SRP1 side of the ratio resolution only'),
            'Q_x0_tail': cells['x0']['comparison']['Q_tail'], 'Q_unit_tail': cells['n7_4h_e1']['comparison']['Q_tail'],
            'objective_convention': 'gross_operational_cost (settlement excluded)'}


def _alpha_row_2x2_context():
    p1 = _load(ALPHA_ROW_PAIR1['path'])
    q = {k: _load(os.path.join(v['eval_dir'], 'evaluation_record.json')) for k, v in p1['points'].items()}
    value = q['x0_a0p50']['certified_cost'] - q['n7_4h_e1_a0p50']['certified_cost']
    return {'source': ALPHA_ROW_PAIR1, 'value_2x2': value,
            'bar_sum_2x2': q['x0_a0p50']['bar']['value'] + q['n7_4h_e1_a0p50']['bar']['value'],
            'cycles': {k: v.get('cycles_run') for k, v in q.items()}}


def stage_spec_content(inst, derived, model, concurrency, post_certification, pre, rule11):
    refs = _reference_R()
    cells = {label: {'nodes': {str(n): list(v) for n, v in CELLS[label].items()}, 'investment_year': YEAR,
                     'candidate_key': _key_of(label), 'eval_key': _eval_key(label, derived),
                     'pre_tail_3x3_key': pre['per_cell'][label]['pre_tail_3x3_key'],
                     'alpha_row_2x2_key_same_candidate': pre['per_cell'][label]['alpha_row_2x2_key_same_candidate']}
             for label in STAGES['pair']['labels']}
    return {
        'schema': f'p515_frozen_spec_v{SPEC_VERSION}', 'version': SPEC_VERSION, 'stage': STAGE_TEXT,
        'authority': ['Planner task W89 step 2 (build the 3 x 3 launcher and spec; DO NOT RUN)',
                      'PLANNER_BRIEF_2026-09-13.md Addendum 46 (3 x 3 pair under production + tight tail; R restated '
                      'against the SRP1 reference re-run under it; certified models persisted if memory allows)',
                      'Addendum 44 (prefix draw [1,2,3] x [1,2,3], R = 0.9331 recorded; band [0.93, 1.09])',
                      'Addendum 45 (persist certified models if memory allows)', 'Addendum 40 ruling 3'],
        'predecessor': {'path': SPEC_V33['path'], 'sha256': _sha(SPEC_V33['path'])},
        'predecessor_not_edited': ('v33 stays as frozen (its campaign freezes failed their own validation, launcher bugs, '
                                   'nothing run); v34 re-derives every v33 decision from the same committed inputs with '
                                   'the fixed launcher'),
        'predecessor_chain': {'v33': SPEC_V33, 'v32': SPEC_V32, 'g6_gate_from': 'v32'},
        'superseded_campaign_freezes_under_v33': SUPERSEDED_CAMPAIGN_FREEZES,
        'changes_from_v33': ['post-certification resolved through H.resolve_post_certification (nothing requested -> '
                             'None) in the entry check, G8 and S9',
                             'the frozen-entry key check scoped to the labels a campaign spec holds; the committed-key '
                             'scan excludes every root under ' + ROOT_REL,
                             'campaign ids s53_w89_3x3_smoke_r2 / s53_w89_3x3_pair_r2',
                             'launcher sha256 (v33 pinned fb73b9b0\'s)'],
        'instance': {'record': {'path': INSTANCE_RECORD_REL, 'sha256': _sha(INSTANCE_RECORD_REL)},
                     'case': {'path': INSTANCE_CASE_REL, 'sha256': derived['case_sha256']},
                     'derived_instance_declaration': derived, 'prefix_verification': inst['prefix_verification'],
                     'prefix_draw': inst['prefix_draw'], 'facts': inst['facts'], 'checks': inst['checks']},
        'configuration': {
            'name': (f'{LABEL}; the 3 x 3 prefix-draw instance {INSTANCE_LABEL}; row 18 premium alpha 0.5 (no floor); '
                     'case-file AA declared; the convergence-depth tail DECLARED {True, 1e-6}; post-certification: '
                     'persist the certified TSO/DSO models'),
            'arm_label': ARM_LABEL, 'overrides': {}, 'case_file_anderson_acceleration': CASE_FILE_AA,
            'ess_ageing_baseline': ESS_AGEING_BASELINE, 'ess_ageing_baseline_label': LABEL,
            'ess_params_file_sha256': ESS_PARAMS_SHA256, 'derived_instance': derived,
            'convergence_depth_tail': TAIL, 'interface_deviation_premium': PREMIUM,
            'post_certification': post_certification,
            'post_certification_requested': POST_CERTIFICATION_REQUESTED,
            'post_certification_decision': model['decision'],
            'post_certification_rationale': ('persist REQUESTED per Addenda 45 / 46 (the alpha row\'s missing terminal '
                                             'models made Addendum 45 ruling 5 unexecutable) and decided by the declared '
                                             'memory rule ("unless the memory preflight forbids it"); no hull polish: '
                                             'Addendum 46 voided the polished-convention proposal'),
            'cap': STAGES['pair']['cap'], 'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES,
            'concurrency': concurrency, 'case_file_sha256': H.sha256_file(H.CASE_FILE),
            'per_cycle_response_capture': 'ON (harness: every derived-instance evaluation; asserted, rule eleven)',
            'floor_status_capture': 'ON (harness: every evaluation; asserted, rule eleven)'},
        'cells': cells,
        'pre_launch_assertion': pre,
        'declared_solve_profile': {
            'pair_per_cell': {'rule': ('83 x (cycles_run + 1) + every retry attempted (per event), 83 = (1 + 3) x 5 x 4 + '
                                       '3 ESSO; cycles_run <= 500; the post-certification persist solves nothing'),
                              'at_75_cycles': SOLVES_PER_ROUND * 76,
                              'verification': 'RECONCILED PER EVENT in the child record; NOT guard-verified'},
            'smoke': {'rule': '83 x (2 + 1) = 249 + every retry attempted', 'base': 249},
            'parent': 'this launcher: SolveProfileGuard(permitted=()) and every imported launcher guard verify(0)'},
        'per_entry_gates': PER_ENTRY_GATES,
        'g6_v32_verbatim': _load(SPEC_V32['path'])['per_entry_gates']['G6_floor_records_v32'],
        'smoke_gate_declared_before_run': SMOKE_GATE, 'smoke_reported_not_gated': SMOKE_REPORTED,
        'smoke_required_before_pair': ('the pair --run REFUSES unless the smoke gate is committed, clean and PASS: the '
                                       'derived instance + premium + tail + persist combination has never run in one '
                                       'child (the alpha row predates the tail; W86 was SRP1)'),
        'objective_convention': OBJECTIVE_CONVENTION,
        'value_definition': ('value = Q(x0) - Q(n7_4h_e1) on this instance; resolution = bar_x0 + bar_unit (each the '
                             'max gross step over the last 10 cycles); |value| or |value - I| <= resolution is '
                             'INDETERMINATE (the bar bounds stopping slack only); ratio = value / R_SRP1_tail; rule ten '
                             '(terminal step / threshold) reported for both cells'),
        'reference_R': refs, 'alpha_row_2x2_context': _alpha_row_2x2_context(),
        'predictions_recorded_before_any_3x3_run': predictions(model, inst, refs),
        'memory_model': model, 'memory_rule': MEMORY_RULE_TEXT,
        'estimates': estimates(model, concurrency, post_certification['persist_certified_models']),
        'rule_eleven': rule11,
        'not_permitted': ['no change to the ADMM formulation, the certification criterion, the AA predicate, the tail or '
                          'any solver option', 'no committed artifact modified or re-run onto; fresh campaign ids / '
                          'roots / working dirs', 'no screen / nohup / &; attached, alone, both streams captured'],
        'harness_sha256': H.sha256_file(H.HARNESS_PATH), 'launcher_sha256': H.sha256_file(os.path.abspath(__file__)),
        'w89_step1_sha256': H.sha256_file(os.path.abspath(X.__file__)),
        'git_head_at_freeze': H._git(['rev-parse', 'HEAD']), 'frozen_utc': _utc(),
    }


def _common_checks():
    failures, ev = [], {}
    for name, pin in (('spec_v32', SPEC_V32), ('reeval_w89', REEVAL_W89), ('alpha_row_pair1', ALPHA_ROW_PAIR1),
                      ('paper_case', PAPER_CASE), ('selection_3x3', SELECTION)):
        st = _git_state(pin['path'])
        ev[name] = {'path': pin['path'], 'sha256': _sha(pin['path']), **st}
        if ev[name]['sha256'] != pin['sha256'] or not (st['git_tracked'] and st['git_clean']):
            failures.append(f'pin {name} not as committed: {ev[name]}')
    if _sha(H.ESS_PARAMS_FILE_REL) != ESS_PARAMS_SHA256:
        failures.append('ESS params file sha256 differs from the W86 pin')
    for rel in (INSTANCE_RECORD_REL, INSTANCE_CASE_REL):
        st = _git_state(rel)
        if not (st['git_tracked'] and st['git_clean']):
            failures.append(f'{rel} not committed / clean')
    from planning_parameters import PlanningParameters
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    if params.admm.anderson_acceleration != CASE_FILE_AA:
        failures.append('case file AA differs from the declaration')
    if params.admm.convergence_depth_tail.get('enabled') is not False:
        failures.append('production default tail is not OFF')
    others = _own_process_alive()
    if others:
        failures.append(f'another copy of this launcher is alive: {others}')
    return failures, ev


def freeze_spec(started):
    tag = f'W89-V{SPEC_VERSION}'
    failures, ev = _common_checks()
    existing = sorted(f for f in os.listdir(_abs(_P53)) if f.startswith(SPEC_PREFIX))
    if existing:
        failures.append(f'v{SPEC_VERSION} already exists (write-once): {existing}')
    inst, derived = load_instance_record()
    if not inst.get('all_checks_pass'):
        failures.append(f"instance record checks do not all pass: {inst.get('checks')}")
    if _sha(INSTANCE_CASE_REL) != derived['case_sha256']:
        failures.append('instance case sha256 != its declaration')
    text, _c, _s, _ch = derive_instance_text()
    with open(_abs(INSTANCE_CASE_REL)) as handle:
        if handle.read() != text:
            failures.append('the instance case no longer re-derives byte-identical')
    model, err = memory_model()
    if err:
        failures.append(err)
    try:
        rule11 = launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
        rule11 = None
    pre = pre_launch_assertion(derived)
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails: {pre}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    m_now = L.memory_preflight(1)
    avail = m_now.get('available_bytes') or 0
    dec = model['decision']
    concurrency = dec['concurrency']
    post_cert = dict(POST_CERTIFICATION_REQUESTED if dec['persist_certified_models']
                     else POST_CERTIFICATION_IF_PERSIST_FORBIDDEN)
    if model['sustained_bytes'] > model['hw_memsize_bytes']:
        _log(f"[{tag} PRECONDITION FAILED] one 3 x 3 child exceeds physical memory: {model['sustained_gib']:.2f} GiB")
        _finish(1)
    content = stage_spec_content(inst, derived, model, concurrency, post_cert, pre, rule11)
    content['memory_at_freeze_non_gating'] = {
        **m_now, 'note': ('the momentary measure at freeze, recorded; the decisions use the BEST recorded availability; '
                          'the run-time preflight gates'),
        'would_pass_now': {'smoke': avail >= model['smoke_bytes'],
                           'pair': avail >= model['required_at_run_gib']['pair'] * GIB}}
    text = json.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(_P53, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError(f'v{SPEC_VERSION} written bytes do not hash to the name')
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] wrote {rel} sha256={sha} (predecessor {content["predecessor"]})')
    for label, c in content['cells'].items():
        _log(f"[{tag}]   {label}: eval_key {c['eval_key']} (pre-tail {c['pre_tail_3x3_key'][:16]}, 2x2 "
             f"{(c['alpha_row_2x2_key_same_candidate'] or '')[:16]})")
    _log(f"[{tag}] memory: sustained {model['sustained_gib']:.2f} GiB/child (range {model['sustained_range_gib']}), "
         f"with persist {model['persist_peak_gib']:.2f} GiB, smoke {model['smoke_gib']:.2f} GiB; best recorded available "
         f"{model['best_available_observed_gib']:.2f} GiB, now {avail / GIB:.2f} GiB")
    _log(f"[{tag}] decision: concurrency {concurrency} ({dec['concurrency_reason']}); persist "
         f"{dec['persist_certified_models']} ({dec['persist_reason']}); post_certification {post_cert}; required at run "
         f"{model['required_at_run_gib']}")
    _log(f"[{tag}] estimates: {content['estimates']['cycle_time']['range_min']} min/cycle; wall/eval "
         f"{content['estimates']['wall_per_evaluation_h']}")
    _finish(0, f'-- next: --stage smoke --freeze, --stage pair --freeze')


def load_stage_spec():
    rel, sha = _find_stage_spec()
    if rel is None:
        raise RuntimeError(f'frozen stage spec v{SPEC_VERSION} not found')
    return rel, sha, _load(rel)


# ======================================================================================================================
#  campaign freeze / run
# ======================================================================================================================
def _stage_concurrency(stage, ss):
    return 1 if stage == 'smoke' else int(ss['configuration']['concurrency'])


def _entries(stage, ss):
    return [(label, _nodes(label), {'investment_year': YEAR, 'interface_deviation_premium': dict(PREMIUM),
                                    'post_certification': post_certification_decided(ss)})
            for label in STAGES[stage]['labels']]


def validate_spec(stage, spec, derived, ss_pin, ss):
    cfg = spec['configuration']
    entries = spec['candidates']
    extra = spec.get('extra') or {}
    st = STAGES[stage]
    checks = {
        'campaign_id': spec.get('campaign_id') == st['campaign_id'],
        'entries_in_order': [e['label'] for e in entries] == list(st['labels']),
        'cap': spec.get('cap') == st['cap'], 'concurrency': spec.get('concurrency') == _stage_concurrency(stage, ss),
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label': cfg.get('arm_label') == ARM_LABEL, 'no_overrides': cfg.get('overrides') == {},
        'aa_declaration': cfg.get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'ess_ageing_declaration': cfg.get('ess_ageing_baseline') == ESS_AGEING_BASELINE,
        'ess_ageing_label': cfg.get('ess_ageing_baseline_label') == LABEL,
        'ess_params_file_pinned': (cfg.get('ess_params_file') or {}).get('sha256') == ESS_PARAMS_SHA256,
        'derived_instance_declared': cfg.get('derived_instance') == derived,
        'tail_declared_enabled_1e-6': cfg.get('convergence_depth_tail') == TAIL,
        'no_model_variant': 'model_variant_label' not in spec and not any('model_variant' in e for e in entries),
        'no_flex_price_variant': 'flex_price_label' not in spec and not any('flex_price_multiplier' in e for e in entries),
        'post_certification_as_stage_spec': all(e.get('post_certification') == post_certification_resolved(ss, e['key'])
                                                for e in entries),
        'stage_spec_pinned': extra.get('stage_spec') == ss_pin,
        'script_recorded': extra.get('campaign_script') == SCRIPT_NAME,
    }
    for e in entries:
        label = e['label']
        checks[f'{label}:canonical_key'] = e.get('key') == _key_of(label)
        checks[f'{label}:eval_key_stage_spec'] = e.get('eval_key') == ss['cells'][label]['eval_key'] == _eval_key(label, derived)
        checks[f'{label}:premium'] = e.get('interface_deviation_premium') == PREMIUM
        checks[f'{label}:no_overrides'] = e.get('overrides') == {}
    return checks


def freeze(stage, started):
    tag = f'W89-{stage.upper()}-FREEZE'
    root = campaign_root(stage)
    failures = H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
    more, ev = _common_checks()
    failures += more
    try:
        ss_rel, ss_sha, ss = load_stage_spec()
        failures += [f'stage spec {k} False' for k, v in _git_state(ss_rel).items() if not v]
    except RuntimeError as error:
        failures.append(str(error))
        ss = None
    inst, derived = load_instance_record()
    if ss is not None and ss['configuration']['derived_instance'] != derived:
        failures.append('the instance record declaration differs from the stage spec')
    if ss is not None and ss['launcher_sha256'] != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this launcher changed since the stage spec froze')
    try:
        rule11 = launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
        rule11 = None
    pre = pre_launch_assertion(derived)
    if not pre['holds']:
        failures.append(f'pre-launch assertion (recomputed) fails: {pre}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    st = STAGES[stage]
    ss_pin = {'path': ss_rel, 'sha256': ss_sha}
    extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
             'stage': stage, 'stage_text': STAGE_TEXT, 'label': LABEL, 'stage_spec': ss_pin,
             'expected_eval_keys': {label: ss['cells'][label]['eval_key'] for label in st['labels']},
             'pre_launch_assertion_at_freeze': pre, 'objective_convention': OBJECTIVE_CONVENTION,
             'solve_claim': ('RECONCILED PER EVENT, NOT GUARD-VERIFIED: the child record solve_profile (83 x (cycles_run '
                             '+ 1) + retries attempted); this parent\'s permitted=() guards verify(0)'),
             'memory_rule': MEMORY_RULE_TEXT, 'rule_eleven': rule11}
    if stage == 'smoke':
        extra['smoke_gate_declared_before_run'] = SMOKE_GATE
        extra['declared_solves'] = {'base': 249, 'rule': '83 per round x (cap 2 + 1) + every retry attempted'}
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        root, st['campaign_id'], _entries(stage, ss),
        configuration={'name': ss['configuration']['name'], 'arm_label': ARM_LABEL, 'overrides': {},
                       'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'ess_ageing_baseline': copy.deepcopy(ESS_AGEING_BASELINE), 'ess_ageing_baseline_label': LABEL,
                       'derived_instance': derived, 'convergence_depth_tail': dict(TAIL),
                       'note': ('no overrides, no model variant, no flexibility-price variant; row 18 alpha 0.5 per '
                                'entry; post-certification as the stage spec decided; tail declared')},
        cap=st['cap'], concurrency=_stage_concurrency(stage, ss),
        authority=['PLANNER_BRIEF_2026-09-13.md Addendum 46', 'Planner task W89 step 2', ss_rel],
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES, extra=extra)
    checks = validate_spec(stage, spec, derived, ss_pin, ss)
    pre_frozen = pre_launch_assertion(derived, spec)
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    for e in spec['candidates']:
        _log(f"[{tag}]   {e['label']}: key={e['key'][:16]} eval_key={e['eval_key']} eval_dir={e['eval_dir']} "
             f"premium={e.get('interface_deviation_premium')} post_certification={e['post_certification']}")
    _log(f'[{tag}] spec checks all={all(checks.values())} failing={[k for k, v in checks.items() if not v]}')
    _log(f"[{tag}] pre-launch assertion on the FROZEN spec: holds={pre_frozen['holds']} "
         f"(committed keys scanned {pre_frozen['n_committed_keys_scanned']})")
    model, _err = memory_model()
    m = memory_preflight(stage, model, ss)
    _log(f"[{tag}] memory at freeze (non-gating): {_memory_line(m)} -> would {'PASS' if m['pass'] else 'REFUSE'}")
    ok = all(checks.values()) and pre_frozen['holds']
    _finish(0 if ok else 1, f'freeze {"OK" if ok else "NOT OK"}; run with --stage {stage} --run --spec-sha256 {spec_sha}')


def _manifest_of(paths):
    out = {}
    for p in paths:
        if os.path.isdir(p):
            for directory, _dirs, files in os.walk(p):
                for fname in sorted(files):
                    fpath = os.path.join(directory, fname)
                    out[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
        elif os.path.isfile(p):
            out[os.path.relpath(p, REPO)] = H.sha256_file(p)
    return out


def _run_preconditions(stage, spec_sha256, tag):
    root = campaign_root(stage)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    more, _ev = _common_checks()
    failures += more
    ss_rel, ss_sha, ss = load_stage_spec()
    inst, derived = load_instance_record()
    checks = validate_spec(stage, spec, derived, {'path': ss_rel, 'sha256': ss_sha}, ss)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    for what, pinned, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('script', spec['extra'].get('campaign_script_sha256'),
                               H.sha256_file(os.path.abspath(__file__))),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE))):
        if pinned != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    pre = pre_launch_assertion(derived, spec)
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails on the frozen spec: {pre}')
    for e in spec['candidates']:
        for eid in e['working_dir_ids'].values():
            if os.path.exists(os.path.join(L._work_dir(), eid)):
                failures.append(f'working dir already exists (never reusable): {eid}')
    try:
        launcher_checklist()
    except AssertionError as error:
        failures.append(str(error))
    model, err = memory_model()
    if err:
        failures.append(err)
        m = {'pass': False}
    else:
        m = memory_preflight(stage, model, ss)
        _log(f"[{tag}] memory preflight: {_memory_line(m)} -> {'PASS' if m['pass'] else 'REFUSE'}")
        if not m['pass']:
            failures.append(f'memory preflight REFUSED: {_memory_line(m)}')
    return root, spec_path, spec, ss_rel, ss_sha, ss, derived, pre, m, failures


def run_smoke(started, spec_sha256):
    tag = 'W89-SMOKE'
    root, spec_path, spec, ss_rel, ss_sha, ss, derived, pre, mem, failures = _run_preconditions(
        'smoke', spec_sha256, tag)
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    entry = spec['candidates'][0]
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f"[{tag}] {STAGE_TEXT}; ONE cell {entry['label']} cap {spec['cap']}; lock {lock}")
    batch = None
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        H.evaluate([entry['label']], ctx)
        batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
    try:
        checks, detail, rec = smoke_gate_checks(entry, eval_dir, ss['cells']['x0']['eval_key'], ss)
    except Exception as error:  # noqa: BLE001 -- the gate FAILS, recorded
        checks, detail, rec = ({'smoke_checks_ran': False},
                               {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}, {})
        print(detail['traceback'], file=sys.stderr, flush=True)
    g = guards_verify()
    checks['S14_parent_guards_zero'] = _guards_ok(g)
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl')) if os.path.isfile(
        os.path.join(eval_dir, 'per_cycle_record.jsonl')) else []
    gate = {'stage': STAGE_TEXT, 'gate': 'W89 3 x 3 child smoke (x0, cap 2)', 'utc': _utc(),
            'git_head': H._git(['rev-parse', 'HEAD']), 'campaign_spec_path': os.path.relpath(spec_path, REPO),
            'campaign_spec_sha256': spec_sha256, 'stage_spec': {'path': ss_rel, 'sha256': ss_sha},
            'eval_dir': os.path.relpath(eval_dir, REPO), 'checks': checks, 'pass': all(checks.values()),
            'failing': sorted(k for k, v in checks.items() if not v), 'detail': detail,
            'reported_not_gated': {'peak_rss': rec.get('peak_rss'), 'wall_time_s': rec.get('wall_time_s'),
                                   'cycle_wall_s': [r.get('cycle_wall_s') for r in rows],
                                   'launcher_wall_s': time.time() - started},
            'pre_launch_assertion': pre, 'memory_preflight': mem, 'batch_info': batch, 'guards': g}
    H._write_once_json(os.path.join(root, SMOKE_GATE_FILE), gate)
    H._write_once_json(os.path.join(root, SMOKE_MANIFEST_FILE), _manifest_of([root]))
    for k, v in checks.items():
        _log(f'[{tag}]   {k}: {"PASS" if v else "FAIL"}')
    _finish(0 if gate['pass'] else 1, f"GATE {'PASS' if gate['pass'] else 'FAIL'} failing={gate['failing']}")


def value_block(cells, refs, inst):
    x0, un = cells.get('x0') or {}, cells.get('n7_4h_e1') or {}
    if not (x0.get('Q') is not None and un.get('Q') is not None):
        return None
    value = x0['Q'] - un['Q']
    res = (x0.get('bar') or 0.0) + (un.get('bar') or 0.0)
    r_srp1, res_srp1 = refs['R_SRP1_tail'], refs['bar_sum_SRP1_tail']
    ratio = value / r_srp1
    i_x = inst['facts']['investment_cost']['n7_4h_e1']['I_x_eur_master_expression']
    return {'value_eur': value, 'resolution': res, 'value_determinate': abs(value) > res,
            'value_per_mwh': value / UNIT_E_MWH, 'I_eur': i_x, 'value_minus_I': value - i_x,
            'value_minus_I_determinate': abs(value - i_x) > res,
            'ratio_to_srp1_tail': ratio,
            'ratio_resolution': math.hypot(res / r_srp1, value * res_srp1 / r_srp1 ** 2),
            'band': [0.93, 1.09], 'inside_band': 0.93 <= ratio <= 1.09, 'R_prefix_recorded': R_PREFIX_RECORDED,
            'both_certified': bool(x0.get('certified') and un.get('certified'))}


def run_pair(started, spec_sha256):
    tag = 'W89-PAIR'
    root, spec_path, spec, ss_rel, ss_sha, ss, derived, pre, mem, failures = _run_preconditions(
        'pair', spec_sha256, tag)
    smoke_rel = os.path.relpath(os.path.join(campaign_root('smoke'), SMOKE_GATE_FILE), REPO)
    st = _git_state(smoke_rel)
    smoke = _load(smoke_rel) if os.path.isfile(_abs(smoke_rel)) else {}
    if not (st['git_tracked'] and st['git_clean'] and smoke.get('pass') is True):
        failures.append(f"the smoke gate must be committed, clean and PASS: {st} pass={smoke.get('pass')}")
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    labels = list(STAGES['pair']['labels'])
    # the smoke's x0 initialisation record becomes the reference of the pair's x0 identity group (alpha-row precedent)
    src = os.path.join(campaign_root('smoke'), H.INIT_IDENTITY_DIR_NAME, f"{os.path.basename(smoke['eval_dir'])}.json")
    dst_dir = os.path.join(root, H.INIT_IDENTITY_DIR_NAME)
    os.makedirs(dst_dir)
    shutil.copyfile(src, os.path.join(dst_dir, f'smoke_reference__{os.path.basename(src)}'))
    _log(f'[{tag}] placed the smoke x0 initialisation record as the x0 identity reference ({H.sha256_file(src)})')
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f"[{tag}] {STAGE_TEXT}; cells {labels} at concurrency {spec['concurrency']}; lock {lock}")
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate(labels, ctx)
        batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
    by_label = {(r or {}).get('candidate_label'): r for r in records}
    entries = {e['label']: e for e in spec['candidates']}
    per_cell, cells_q = {}, {}
    for label in labels:
        eval_dir = os.path.join(root, 'evals', entries[label]['eval_dir'])
        try:
            gates, detail, rec = cell_gates(entries[label], eval_dir, ss)
        except Exception as error:  # noqa: BLE001 -- recorded; the gates FAIL
            gates, detail, rec = {'cell_gates_ran': False}, {'error': f'{type(error).__name__}: {error}',
                                                             'traceback': traceback.format_exc()}, {}
        cell = None
        if rec.get('status') in ('certified', 'not_certified'):
            try:
                cell = A.cell_quantities(os.path.relpath(eval_dir, REPO))
            except Exception as error:  # noqa: BLE001
                cell = {'error': f'{type(error).__name__}: {error}'}
        cells_q[label] = cell or {}
        per_cell[label] = {'status': rec.get('status'), 'cycles_run': rec.get('cycles_run'),
                           'certification_cycle': rec.get('certification_cycle'), 'Q': rec.get('certified_cost'),
                           'bar': (rec.get('bar') or {}).get('value'), 'gates': gates, 'gates_pass': all(gates.values()),
                           'gate_detail': detail, 'cell_quantities': cell,
                           'parent_view': (by_label.get(label) or {}).get('parent_view')}
    refs = ss['reference_R']
    inst, _d = load_instance_record()
    value = value_block(cells_q, refs, inst)
    g = guards_verify()
    results = {'stage': STAGE_TEXT, 'utc': _utc(), 'git_head_at_run': H._git(['rev-parse', 'HEAD']),
               'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
               'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'pre_launch_assertion': pre,
               'objective_convention': OBJECTIVE_CONVENTION, 'per_cell': per_cell,
               'all_gates_pass': all(per_cell[k]['gates_pass'] for k in labels), 'value_and_R': value,
               'reference_R': refs, 'smoke_gate': {'path': smoke_rel, 'sha256': _sha(smoke_rel)},
               'solve_claim': 'RECONCILED PER EVENT in each child record (G5); parent guards verify(0)',
               'guards': g, 'memory_preflight_at_run': mem, 'batch_info': batch, 'wall_clock_s': time.time() - started}
    H._write_once_json(os.path.join(root, PAIR_RESULTS_FILE), results)
    H._write_once_json(os.path.join(root, PAIR_MANIFEST_FILE), _manifest_of([root]))
    for label in labels:
        p = per_cell[label]
        _log(f"[{tag}] {label}: status {p['status']} cycles {p['cycles_run']} Q {p['Q']} bar {p['bar']} gates {p['gates']}")
    _log(f'[{tag}] value / R: {value}')
    code = 0
    if not results['all_gates_pass'] or not _guards_ok(g):
        code = 1
    elif any(per_cell[k]['status'] != 'certified' for k in labels):
        code = 2
    _finish(code, f'wall={time.time() - started:.0f}s')


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-instance', action='store_true')
    mode.add_argument('--memory-probe', choices=MEMORY_PROBES)
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--stage', choices=tuple(STAGES))
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--scratch', default=None)
    args = parser.parse_args()
    started = time.time()
    try:
        if args.freeze_instance or args.memory_probe:
            if not args.scratch or os.path.abspath(args.scratch).startswith(REPO + os.sep):
                parser.error('--scratch <dir outside the repository> is required')
            os.makedirs(args.scratch, exist_ok=True)
            if args.freeze_instance:
                freeze_instance(started, args.scratch)
            else:
                memory_probe(args.memory_probe, started, args.scratch)
        elif args.freeze_spec:
            freeze_spec(started)
        else:
            if args.freeze == args.run:
                parser.error('--stage needs exactly one of --freeze / --run')
            if args.freeze:
                freeze(args.stage, started)
            else:
                if not args.spec_sha256:
                    parser.error('--run requires --spec-sha256')
                (run_smoke if args.stage == 'smoke' else run_pair)(started, args.spec_sha256)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        _finish(1, 'EXCEPTION')


if __name__ == '__main__':
    main()
