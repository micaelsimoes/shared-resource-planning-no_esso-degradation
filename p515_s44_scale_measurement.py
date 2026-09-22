"""
P5.15 Addendum 25 -- paper-scale BUILD measurement (Step 4/5 design input; "measure, do not
assume").

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 25 ("Scale"); STEP4_DFO_METHOD.md section 7;
frozen spec v14 `data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`,
key `item4_scale_measurement`: "build only (zero solves, SolveProfileGuard armed): block
count, variable/constraint counts, peak RSS; a memory WATCHDOG aborts the build above 24 GB
(32 GB machine), recording RSS reached and build stage; only if it fits, time ONE cycle, the
single declared solve set. Runs alone."

================================================================================
WHAT "PAPER SCALE" IS IN THIS CODE BASE (established from code and repository history)
================================================================================
The planning instance is configured entirely by the case file `data/SRP1/SRP1.json`:
`Years` ({year: number of calendar years the representative year stands for}), `Days`
({season: days}), `NumMarketScenarios`, and per network `num_operation_scenarios`.
Scenarios are NOT data files: `_read_market_data_from_file`
(shared_resources_planning.py) and `network_data._read_network_data` fit a Gaussian copula
with KDE marginals to the season base profiles of `SRP1_market_data.xlsx` /
`<network>_operational_data.xlsx`, draw 100 synthetic days, and SAMPLE
`NumMarketScenarios` / `num_operation_scenarios` of them per (year, day) from seeds derived
from `RandomSeed`. Per-year network data are `data/SRP1/<network>/<network>_<year>.json`
(present for 2025-2040 and 2045 for case9 and case33_1..3); ESS unit costs
(`SRP1_ESS.xlsx`) cover 2020-2054. So a larger instance needs NO new data file, only a
different case file.

The paper instance ("5 years x 4 days x 25 scenarios"):
  * EXPERT_REVIEW.md:81 -- "The paper specifies five representative years and 25
    market/operational combinations per day"; :147 -- "five three-year blocks beginning in
    2025 cover through 2039" (label 2037 is the last representative year); :140-145 --
    sum_y N_y = 15.
  * git 0198407e (2025-12-21, "Update. Case studies"), `data/SRP1/SRP1.json`: Years
    {2025: 3, 2028: 3, 2031: 3, 2034: 3, 2037: 3}, the same four days, NumMarketScenarios 5,
    num_operation_scenarios 5 for case9 and all three DSOs -- the only 5-year, 25-combination
    SRP1 configuration in the file's history (all 170 commits of data/SRP1/SRP1.json read).
  * data/SRP1/Results/20251221_3 years/ (untracked, preserved): "Main Info" reports years
    2025/2028/2031/2034/2037 x 4 days with 5 market and 5 operation scenarios.
The derived case file therefore changes EXACTLY these keys of the current SRP1.json and
nothing else (RandomSeed 2026, DiscountFactor, days, networks, params files unchanged):
    Years -> {2025: 3, 2028: 3, 2031: 3, 2034: 3, 2037: 3}
    NumMarketScenarios -> 5
    TransmissionNetwork.num_operation_scenarios and every DistributionNetworks[*]
    .num_operation_scenarios -> 5
It is written into the (write-once) measurement directory, never into data/SRP1. The scenario
realization it produces is NEW (different years and counts -> different derived seeds); its
combined checksum is recorded, and no canonical value exists to compare it with.

================================================================================
SCALING MECHANISM (code trace)
================================================================================
A network BLOCK is one Pyomo ConcreteModel per (network, year, day): `NetworkData.build_model`
loops years x days and calls `Network.build_model` once each. Scenarios are INDEXED INSIDE
each block: `network._build_model` declares `model.scenarios_market` and
`model.scenarios_operation` and indexes every operational Var/Constraint by
[..., s_m, s_o, p] (e.g. `model.e[node, s_m, s_o, p]`). So:
    network blocks = (1 TSO + 3 DSOs) x |years| x |days|  -> 48 (SRP1), 80 (paper)
    scenario combinations per block = NumMarketScenarios x num_operation_scenarios -> 1, 25
    ESSO models = one per active node (3), each indexed by (year, day, period), NO scenario
    index -> grows with |years| only.
The expert's "2,000 blocks" is 80 blocks x 25 scenario combinations; the Pyomo block count is
80, each block ~25x larger in its scenario-indexed part. The per-cycle solve count is
80 + 3 = 83 (51 at SRP1); each network NLP is ~25x larger.

================================================================================
HOW THE BUILD IS MEASURED WITH ZERO SOLVES
================================================================================
Production's initialization interleaves build and solve (each `create_*_model` builds its
blocks and then calls `.optimize`), so "build everything, solve nothing" cannot be obtained
by stopping production at its first solve (that would build one DSO only). This script
follows the zero-solve precedent of `p515_s44_addendum26_confirmations.py` item 3 (spec v14,
key `addendum26_confirmations_zero_solve`): the oracle's construction path
(`p515_g_g1_g4_admm_gates._construct_arm_planning`, D arm, `apply_rho=False`, C* at nodes 5,
7, 9, investment year 2025) and then production's own initialization functions, called in
the order of `_run_operational_planning`'s `initial_state is None` branch, with each agent's
`.optimize` replaced ON THAT PLANNING INSTANCE ONLY by an interceptor that records the call
and returns "no result" (never a solver, never a fabricated solution). Declared intercepts:
one per DSO network, one TSO, one ESSO initialization -- checked exactly.
`SolveProfileGuard(permitted=())` is installed before any production import and verified
at exactly 0 solves / 0 launches / 0 blocked.
  Steps (production order): create_admm_variables; create_distribution_networks_models
  (sequential); create_transmission_network_model; create_shared_energy_storage_model;
  _prepare_distribution/transmission_objectives_for_admm;
  _compute_common_admm_objective_scale (attempted and recorded: it reads SOLVED objective
  values, so on unsolved blocks it may fail -- trace artifact); the case-file fixed sigma is
  then used (DECLARED SUBSTITUTION, as in the precedent; the sigma calibration assertion needs
  the initialization solves and is not exercised); _resolve_esso_al_scale;
  update_distribution/transmission_models_to_admm under p58_rescale.patched_admm_objectives
  (as run_admm_arm does); update_shared_energy_storage_model_to_admm;
  _initialize_shared_ess_consensus; update_interface_power_flow_variables (skips unsolved
  blocks by production's own test); get_updated_capacities; and the two pristine clones the
  ADMM loop keeps (`tso_pristine_base`: every TSO block; `dso_pristine_base`: every node-7 DSO
  block), built by CALLING production's own `shared_resources_planning.
  _build_pristine_snapshot_bases` (which holds both the mode validation and the clone
  expressions `_run_operational_planning` uses), declared. P5.15 Addendum 27 item 5(a):
  `--snapshots off` sets BOTH capture modes to 'off' on this child's planning object, so that
  helper builds NO pristine base and this stage costs nothing; the modes applied are verified
  and recorded (`build_record.json` -> `snapshot_setting`, `cycle_record.json` likewise).
NOT in the build figure (they exist only after real solves): the IPOPT SolverResults objects
kept per block, the multiplier suffix contents loaded by `model.solutions.load_from`, the NL
writer's transient peak, and the IPOPT processes' own memory. The oracle's zero-solve SoH
floor-row precheck (`p515_s40_polish_gap._build_floor_rows`, a transient planning copy plus
3 ESSO builds, deleted) and the harness's s38/s39 capture wrappers are also not included.
`--time-one-cycle` measures all of these directly.

================================================================================
MEMORY WATCHDOG
================================================================================
A daemon thread in each child samples every 0.5 s: RSS of the process and of all its
descendants (psutil), the process phys_footprint (libproc proc_pid_rusage -- counts
compressed/swapped pages RSS does not), system available memory and swap. EVERY one of these
is recorded on every sample regardless of which one gates.

GATING MEASURE (P5.15 Addendum 27 W13; `--watchdog-measure`, default `rss_tree`). The memory
limit is compared against `rss_tree` = RSS of the process plus its descendants. Before W13 it
was compared against `measure` = max(rss_tree, phys_footprint + descendants' RSS); that
quantity is still computed, peak-tracked and reported, and `--watchdog-measure footprint`
restores it as the gate, so the change is visible and reversible. The reason for the default:
label `paper_cycle_snapoff_r1` aborted at t=933 s with footprint_self 25.78 GB above the 24
GiB limit while rss_tree was 15.93 GB, system available 12.14 GB and swap used 0.25 MB -- the
abort fired on compressed pages, not on residency, and the machine was not short of memory.
The gating measure's NAME is written into every sample, every peak record, launch.json,
summary.json and `watchdog_abort_<mode>.json` (`gating_measure`), so the two quantities can
never be confused by a later reader.

Triggers (all recorded in launch.json -> thresholds, and each names itself in the abort record):
  1. gating measure > the RSS limit (`--rss-limit-gib`, default 24 = the spec's "24 GB" on a
     machine psutil reports as 32.0 GiB);
  2. system available memory < 1.0 GiB (declared secondary trigger, machine protection);
  3. THRASHING (W13): swap used > 2.0 GiB, or swap used grown by more than 1.0 GiB within any
     60 s window (the swap series is in every sample, as it already was).
On trigger the watchdog writes `watchdog_abort_<mode>.json` (cause, the triggered guard, the
gating measure and its value, RSS, footprint, swap, the stage in force, elapsed time, the last
samples), kills the process's descendants and exits with the distinct code 97 (`os._exit`, so
the memory is released at once). Every sample is appended live to `rss_samples_<mode>.jsonl`
and every stage transition to `stages_<mode>.jsonl`, so an aborted run keeps its trajectory.
The parent (which imports no model code) polls the child tree as a backstop on the SAME gating
measure and kills it above limit + 1 GiB (exit record `parent_backstop_kill_<mode>.json`). The
limit and the gating measure are recorded in launch.json and read by the children from there;
`--rss-limit-gib` is recorded with a `rss_limit_is_spec_default` flag.

================================================================================
PROCESS MODEL, OUTPUT, LOCKS
================================================================================
Parent (this file, no `--child`): refuses if the label directory exists (write-once), if
this script's own lock `.p515_s44_scale_measurement.lock` exists, and -- for any instance
other than `srp1` -- if the campaign lock or the legacy run lock exists or another p51x/p514
harness process is running ("runs alone"). Writes the derived case file, launch.json and
parent_run.log, then runs the BUILD child (fresh interpreter, attached, thread caps of the
campaign harness, stdout/stderr to build_child_stdout.log / build_child_stderr.log), then --
only with `--time-one-cycle` AND a complete build under the watchdog -- the CYCLE child (a
production evaluation, `run_admm_arm`, cap 1: initialization + exactly one ADMM cycle under
SolveProfileGuard(p514_n PERMITTED); see SOLVE DECLARATION below). Finally summary.json --
whose `scale_measurement` block holds the whole calibration in one place (per-cycle wall time,
the initialization time separately, peak RSS of each child, sigma / ESSO AL scale and the sigma
calibration outcome, snapshot mode, effective AA, solves, failures) -- and manifest_sha256.json
(every file in the label directory, plus the P56A working-dir files of the run's eval ids).

================================================================================
CASE-FILE ANDERSON ACCELERATION (P5.15 Addendum 27 item 1; W12)
================================================================================
Since b5629311 `data/SRP1/SRP1_params.json` carries `admm.anderson_acceleration` ON. Both
`spec_like` dicts this file hands to `p515_s44_campaign_harness._config_hook_factory` therefore
DECLARE `configuration.case_file_anderson_acceleration` = the exact dict the case file loads to
(`CASE_FILE_AA`, the literal `p515_s45_a0_campaign.CASE_FILE_AA` carries; the two are checked to
agree before the run, by parsing that file's source -- importing it would install a
`permitted=()` guard). Without the declaration the hook keeps its pre-Addendum-27 check and
refuses BOTH children with `configuration not as frozen:
['anderson_acceleration_off_before_overrides']`. With it, the hook compares the LOADED AA dict
against the declaration and RAISES on any difference -- so an instance that ever loaded a case
file WITHOUT AA (the `admm_parameters` default is `{'enabled': False, 'memory': 5,
'regularization': 1e-10}`, which has no `reject_policy` and `enabled` False) fails
`anderson_acceleration_case_file_matches_declaration` and is refused, never run silently.

================================================================================
SOLVE DECLARATION FOR THE TIMED CYCLE (W12)
================================================================================
Not a strict constant: production retries a failed local NLP (tier 1) and may then retry from a
frozen snapshot (tier 2). `declared_solve_profile` derives, from the planning object and BEFORE
the run,
    solves_per_cycle = (1 + n_dso) x n_years x n_days + n_esso_nodes   (51 SRP1; 83 paper)
    base             = solves_per_cycle x (1 initialization + CYCLE_CAP cycles)  (102; 166)
and the run is gated on the identity
    observed == base + 1 x recovered_tier1 + 2 x recovered_tier2
with the tier counts read from the run's OWN `network_failures_summary`. The bounded
`SolveProfileGuard(p514_n.PERMITTED)` is armed for the whole cycle child and `verify()`-ed
EXACTLY against that reconciled total (too few fails as loudly as too many). This follows
`p515_s45_snapshot_off_failure_gate.py` section (5), which does the same for its two arms.
Output: `data/SRP1/Results/P515S44/scale_measurement/<label>/`.
Exit codes: 0 complete; 97 watchdog abort; 98 parent backstop kill; 1 error; 2 refused.

================================================================================
COMMANDS (repo root; attached; both streams captured; never detached)
================================================================================
SRP1 calibration (Worker; may run alongside the selection campaign):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_scale_measurement.py \\
      --instance srp1 --label srp1_calibration_r2 \\
      > data/SRP1/Results/P515S44/scale_measurement/srp1_calibration_r2_launch.log 2>&1
Paper scale (Planner; alone, after the selection run):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_scale_measurement.py \\
      --instance paper --label paper_build \\
      > data/SRP1/Results/P515S44/scale_measurement/paper_build_launch.log 2>&1
  (add `--time-one-cycle` to time one cycle if, and only if, the build completes under the
  watchdog; the flag is off by default.)
Paper scale without the pristine snapshot clones (P5.15 Addendum 27 item 5(a); the clone stage
is what crossed the 24 GiB watchdog in label `paper_build`):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_scale_measurement.py \\
      --instance paper --label paper_build_nosnap --snapshots off \\
      > data/SRP1/Results/P515S44/scale_measurement/paper_build_nosnap_launch.log 2>&1
SRP1 per-cycle calibration under the AA-on case file, snapshots off (P5.15 Addendum 27 W12;
this is the run the paper-scale estimate is calibrated on):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_scale_measurement.py \\
      --instance srp1 --label srp1_cycle_snapoff_r1 --snapshots off --time-one-cycle \\
      > data/SRP1/Results/P515S44/scale_measurement/srp1_cycle_snapoff_r1_launch.log 2>&1
Paper scale, snapshots off, one timed cycle (Planner; alone):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_scale_measurement.py \\
      --instance paper --label paper_cycle_snapoff_r1 --snapshots off --time-one-cycle \\
      > data/SRP1/Results/P515S44/scale_measurement/paper_cycle_snapoff_r1_launch.log 2>&1
Paper scale, snapshots off, one timed cycle, W13 watchdog (gating measure `rss_tree`, limit 28
GiB on the 32 GiB machine, system-available floor and thrashing guards active) -- the re-run of
`paper_cycle_snapoff_r1`, which aborted on the footprint measure at t=933 s while rss_tree was
15.93 GB (Planner; alone):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_scale_measurement.py \\
      --instance paper --label paper_cycle_snapoff_r2 --snapshots off --time-one-cycle \\
      --rss-limit-gib 28 \\
      > data/SRP1/Results/P515S44/scale_measurement/paper_cycle_snapoff_r2_launch.log 2>&1
Paper scale, snapshots off, one timed cycle, W13 watchdog, solution bookkeeping released (P5.15
Addendum 29 W32; Planner; alone):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_scale_measurement.py \\
      --instance paper --label paper_cycle_snapoff_memfix_r1 --snapshots off --time-one-cycle \\
      --rss-limit-gib 28 --release-solution-bookkeeping \\
      > data/SRP1/Results/P515S44/scale_measurement/paper_cycle_snapoff_memfix_r1_launch.log 2>&1
Watchdog abort-path test (Worker; SRP1 scale, limit lowered to 0.5 GiB, recorded):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_scale_measurement.py \\
      --instance srp1 --label srp1_watchdog_abort_test_r2 --rss-limit-gib 0.5 \\
      > data/SRP1/Results/P515S44/scale_measurement/srp1_watchdog_abort_test_r2_launch.log 2>&1
"""

import argparse
import ast
import ctypes
import hashlib
import json
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

import psutil

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SCRIPT_PATH = os.path.abspath(__file__)
PYTHON = sys.executable
STAGE = 'P5.15 Addendum 25 paper-scale build measurement (frozen spec v14 item4_scale_measurement)'
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 25 (Scale)',
    'STEP4_DFO_METHOD.md section 7',
    'data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json item4_scale_measurement',
]
SCHEMA = 'p515_s44_scale_measurement_v1'

DATA_DIR = os.path.join(REPO, 'data', 'SRP1')
SOURCE_CASE_REL = os.path.join('data', 'SRP1', 'SRP1.json')
SOURCE_CASE = os.path.join(REPO, SOURCE_CASE_REL)
OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', 'scale_measurement')
OWN_LOCK_PATH = os.path.join(REPO, '.p515_s44_scale_measurement.lock')
CAMPAIGN_LOCK_PATH = os.path.join(REPO, '.p515_s44_campaign.lock')
LEGACY_RUN_LOCK_PATH = os.path.join(REPO, '.p515_g_gate.lock')

GIB = 1 << 30
RSS_LIMIT_GIB_DEFAULT = 24           # spec: "aborts the build above 24 GB (32 GB machine)"
RSS_LIMIT_BYTES = RSS_LIMIT_GIB_DEFAULT * GIB
BACKSTOP_MARGIN_BYTES = 1 * GIB      # parent kills the child tree above limit + 1 GiB if the child's watchdog failed
MIN_AVAILABLE_BYTES = 1 * GIB        # declared secondary trigger (machine protection)
SAMPLE_INTERVAL_S = 0.5
PARENT_POLL_S = 1.0

# P5.15 Addendum 27 W13. WHICH measured quantity the memory limit gates on. Both are
# computed and recorded on every sample, whichever gates:
#   'rss_tree'  -- resident memory of the process and its descendants (the DEFAULT since W13:
#                  it is what actually occupies physical RAM);
#   'footprint' -- the pre-W13 quantity `measure` = max(rss_tree, phys_footprint + children
#                  RSS). macOS `phys_footprint` counts COMPRESSED pages, so it can exceed true
#                  residency by many GiB on a machine under no memory pressure.
# Evidence for the change: label `paper_cycle_snapoff_r1` aborted at t=933 s with
# footprint_self 25.78 GB above the 24 GiB limit while rss_tree was 15.93 GB, system
# available memory 12.14 GB and swap used 0.25 MB -- the machine was not short of memory.
# Nothing is removed: `measure` and `footprint_self` are still sampled, still peak-tracked and
# still reported, so the r1 artifacts stay comparable field for field.
GATING_MEASURES = ('rss_tree', 'footprint')
GATING_MEASURE_DEFAULT = 'rss_tree'
GATING_MEASURE_FIELD = {'rss_tree': 'rss_tree', 'footprint': 'measure'}
# Thrashing guard (W13 item 3): swap is what actually threatens the machine once compression
# stops being free. Absolute ceiling, and a growth rate over a sliding window.
SWAP_USED_LIMIT_BYTES = 2 * GIB
SWAP_GROWTH_LIMIT_BYTES = 1 * GIB
SWAP_GROWTH_WINDOW_S = 60.0


def gating_value(sample, measure=GATING_MEASURE_DEFAULT):
    """The value the memory limit is compared against, for the named gating measure."""
    return (sample or {}).get(GATING_MEASURE_FIELD[measure]) or 0

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
EXIT_WATCHDOG = 97
EXIT_BACKSTOP = 98

# C* of the selection run (spec v14 item3_selection_run.candidates.C_star; the harness's
# INVESTMENT_YEAR 2025 and nodes 5, 7, 9).
C_STAR = {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)}
C_STAR_LABEL = '0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025 (spec v14 C_star)'
BUILD_CAP = 500   # spec v14 item3: the campaign cap (num_max_iters; irrelevant to the build)
CYCLE_CAP = 1     # --time-one-cycle: exactly one ADMM cycle
REQUIRED_CONSECUTIVE_CYCLES = 10  # campaign spec field (campaign_spec_s44_gate_4047b4e3.json); case-file value

# P5.15 Addendum 27 item 1 (W12). Since b5629311 the SRP1 case file carries Anderson
# acceleration ON (`data/SRP1/SRP1_params.json` -> `admm.anderson_acceleration`). The campaign
# harness's `_config_hook_factory` only accepts an AA-on case file when the spec it is handed
# DECLARES the exact dict the case file loads to; without the declaration it keeps its
# pre-Addendum-27 check and refuses the run with `configuration not as frozen:
# ['anderson_acceleration_off_before_overrides']`. Both `spec_like` dicts in this file
# therefore declare it. The literal is the one `p515_s45_a0_campaign.CASE_FILE_AA` carries;
# `check_case_file_aa_declaration()` verifies the two agree WITHOUT importing that module
# (importing it installs a `permitted=()` SolveProfileGuard at module level, which would block
# this stage's solves).
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CASE_FILE_AA_SOURCE = 'p515_s45_a0_campaign.py'
CASE_FILE_AA_SOURCE_NAME = 'CASE_FILE_AA'

PAPER_YEARS = {'2025': 3, '2028': 3, '2031': 3, '2034': 3, '2037': 3}
INSTANCES = {
    'srp1': {
        'description': 'the current SRP1 case file unchanged (3 years x 4 days x 1 x 1 scenario)',
        'years': None, 'num_market_scenarios': None, 'num_operation_scenarios': None,
    },
    'paper': {
        'description': ('the paper instance: 5 representative years (2025, 2028, 2031, 2034, 2037; '
                        'each a 3-year block, sum 15) x 4 days x 5 market x 5 operation scenarios '
                        '(25 combinations per block)'),
        'years': PAPER_YEARS, 'num_market_scenarios': 5, 'num_operation_scenarios': 5,
        'sources': [
            'EXPERT_REVIEW.md:81 (five representative years, 25 market/operational combinations per day)',
            'EXPERT_REVIEW.md:147 (five three-year blocks beginning in 2025; 2037 last label)',
            'git 0198407e:data/SRP1/SRP1.json (Years 2025/2028/2031/2034/2037 x 3; NumMarketScenarios 5; '
            'num_operation_scenarios 5 for case9 and case33_1..3)',
            'data/SRP1/Results/20251221_3 years/ Main Info (5 years, 5 market / 5 operation scenarios)',
        ],
    },
}

THREAD_CAP_ENV = {  # = p515_s44_campaign_harness.THREAD_CAP_ENV (checked at child start)
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
}
HARNESS_PATTERN = re.compile(r'p51\d\w*\.py|p514_\w*\.py')


# ======================================================================================
#  small utilities
# ======================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once_json(path, obj):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(obj, handle, indent=1, default=str)
    os.replace(tmp, path)


def _append_jsonl(path, obj):
    with open(path, 'a') as handle:
        handle.write(json.dumps(obj, default=str) + '\n')
        handle.flush()


def _git(args):
    try:
        return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as error:  # noqa: BLE001 -- recorded
        return f'<git error: {error}>'


def label_dir(label):
    return os.path.join(OUT_ROOT, label)


def eval_id(label, mode):
    return f'p515s44_scale_{label}_{mode}'


def check_case_file_aa_declaration():
    """Verify this file's `CASE_FILE_AA` is the literal `p515_s45_a0_campaign.py` declares.

    Parsed with `ast` from that file's SOURCE, never imported: importing it installs a
    module-level `SolveProfileGuard(permitted=())`, which would block every solve of the
    cycle child. Raises on disagreement (the declaration is what the configuration hook
    checks the loaded case file against, so the two must not drift); records both literals."""
    path = os.path.join(REPO, CASE_FILE_AA_SOURCE)
    record = {'declared_here': dict(CASE_FILE_AA), 'source_file': CASE_FILE_AA_SOURCE,
              'source_name': CASE_FILE_AA_SOURCE_NAME, 'source_present': os.path.exists(path),
              'method': 'ast.literal_eval of the assignment in the source (never imported)'}
    if not record['source_present']:
        record['agree'] = None
        record['note'] = 'source file absent; the declaration here stands on its own'
        return record
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    found = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == CASE_FILE_AA_SOURCE_NAME for t in node.targets):
            found = ast.literal_eval(node.value)
    record['source_literal'] = found
    record['agree'] = found == CASE_FILE_AA
    if not record['agree']:
        raise RuntimeError(f'case-file AA declaration disagrees with {CASE_FILE_AA_SOURCE}: '
                           f'here {CASE_FILE_AA} vs there {found}')
    return record


# ======================================================================================
#  memory sampling (psutil + macOS phys_footprint)
# ======================================================================================
class _RUsageInfoV0(ctypes.Structure):
    _fields_ = [('ri_uuid', ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            'ri_user_time', 'ri_system_time', 'ri_pkg_idle_wkups', 'ri_interrupt_wkups',
            'ri_pageins', 'ri_wired_size', 'ri_resident_size', 'ri_phys_footprint',
            'ri_proc_start_abstime', 'ri_proc_exit_abstime')]


try:
    _LIBPROC = ctypes.CDLL('/usr/lib/libproc.dylib')
    _LIBPROC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
except OSError:  # non-macOS: footprint unavailable, recorded as None
    _LIBPROC = None


def phys_footprint(pid):
    if _LIBPROC is None:
        return None
    info = _RUsageInfoV0()
    if _LIBPROC.proc_pid_rusage(int(pid), 0, ctypes.byref(info)) != 0:
        return None
    return int(info.ri_phys_footprint)


def tree_memory(pid):
    """RSS of `pid` and of all its live descendants, plus pid's phys_footprint."""
    try:
        proc = psutil.Process(pid)
        rss_self = proc.memory_info().rss
        kids = proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return None
    rss_children = 0
    for kid in kids:
        try:
            rss_children += kid.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    footprint = phys_footprint(pid)
    return {'rss_self': rss_self, 'rss_children': rss_children, 'n_children': len(kids),
            'rss_tree': rss_self + rss_children, 'footprint_self': footprint,
            'measure': max(rss_self + rss_children, (footprint or 0) + rss_children)}


def _kill_descendants(pid):
    try:
        kids = psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return []
    killed = []
    for kid in kids:
        try:
            kid.kill()
            killed.append(kid.pid)
        except psutil.NoSuchProcess:
            pass
    return killed


# ======================================================================================
#  CHILD: stage tracker + watchdog
# ======================================================================================
STAGE_REF = {'name': 'child start', 'since': time.time()}


class Watchdog(threading.Thread):
    def __init__(self, out_dir, mode, limit_bytes=RSS_LIMIT_BYTES, interval=SAMPLE_INTERVAL_S,
                 min_available=MIN_AVAILABLE_BYTES, gating_measure=GATING_MEASURE_DEFAULT,
                 swap_used_limit=SWAP_USED_LIMIT_BYTES, swap_growth_limit=SWAP_GROWTH_LIMIT_BYTES,
                 swap_growth_window_s=SWAP_GROWTH_WINDOW_S):
        super().__init__(name='rss-watchdog', daemon=True)
        if gating_measure not in GATING_MEASURES:
            raise ValueError(f'unknown gating measure {gating_measure!r} (expected one of {GATING_MEASURES})')
        self.out_dir, self.mode = out_dir, mode
        self.limit, self.interval, self.min_available = limit_bytes, interval, min_available
        self.gating_measure = gating_measure
        self.gating_field = GATING_MEASURE_FIELD[gating_measure]
        self.swap_used_limit = swap_used_limit
        self.swap_growth_limit = swap_growth_limit
        self.swap_growth_window_s = swap_growth_window_s
        self.samples_path = os.path.join(out_dir, f'rss_samples_{mode}.jsonl')
        self.abort_path = os.path.join(out_dir, f'watchdog_abort_{mode}.json')
        self.t0 = time.time()
        self.pid = os.getpid()
        self.peak = {'measure': 0, 'rss_tree': 0, 'rss_self': 0, 'footprint_self': 0,
                     'stage_at_peak_measure': None, 't_at_peak_measure': None,
                     # W13: the GATED quantity, named, alongside every pre-W13 field (unchanged)
                     'gating_measure': gating_measure, 'gating_value': 0,
                     'stage_at_peak_gating_value': None, 't_at_peak_gating_value': None,
                     'swap_used': 0, 'stage_at_peak_swap_used': None}
        self.tail = []
        self.swap_window = []      # (t, swap_used) pairs within the growth window
        self.n_samples = 0
        self._halt = threading.Event()

    def thresholds(self):
        """Every guard threshold in force, by name (W13: recorded, never implicit)."""
        return {'gating_measure': self.gating_measure, 'gating_measure_sample_field': self.gating_field,
                'gating_measures_available': list(GATING_MEASURES),
                'rss_limit_bytes': self.limit, 'rss_limit_gib': self.limit / GIB,
                'min_available_bytes': self.min_available,
                'swap_used_limit_bytes': self.swap_used_limit,
                'swap_growth_limit_bytes': self.swap_growth_limit,
                'swap_growth_window_s': self.swap_growth_window_s,
                'sample_interval_s': self.interval}

    def sample(self):
        mem = tree_memory(self.pid)
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        s = {'t': round(time.time() - self.t0, 3), 'stage': STAGE_REF['name'],
             **(mem or {}), 'sys_available': vm.available, 'sys_used_pct': vm.percent,
             'swap_used': sw.used}
        # W13: the gated quantity, by name, on every sample. `measure` (footprint-based) and
        # `footprint_self` remain exactly as before, whichever of them gates.
        s['gating_measure'] = self.gating_measure
        s['gating_value'] = gating_value(s, self.gating_measure)
        return s

    def _update_peak(self, s):
        for key in ('rss_tree', 'rss_self', 'footprint_self'):
            if (s.get(key) or 0) > self.peak[key]:
                self.peak[key] = s[key]
        if (s.get('measure') or 0) > self.peak['measure']:
            self.peak['measure'] = s['measure']
            self.peak['stage_at_peak_measure'] = s['stage']
            self.peak['t_at_peak_measure'] = s['t']
        if (s.get('gating_value') or 0) > self.peak['gating_value']:
            self.peak['gating_value'] = s['gating_value']
            self.peak['stage_at_peak_gating_value'] = s['stage']
            self.peak['t_at_peak_gating_value'] = s['t']
        if (s.get('swap_used') or 0) > self.peak['swap_used']:
            self.peak['swap_used'] = s['swap_used']
            self.peak['stage_at_peak_swap_used'] = s['stage']

    def _swap_growth(self, s):
        """Swap growth within the trailing window: current `swap_used` minus the smallest
        `swap_used` seen in the last `swap_growth_window_s` seconds. Returns
        (growth_bytes, window_baseline, window_span_s)."""
        t = s['t']
        self.swap_window.append((t, s.get('swap_used') or 0))
        cutoff = t - self.swap_growth_window_s
        while len(self.swap_window) > 1 and self.swap_window[0][0] < cutoff:
            self.swap_window.pop(0)
        baseline = min(v for _, v in self.swap_window)
        return (s.get('swap_used') or 0) - baseline, baseline, round(t - self.swap_window[0][0], 3)

    def _abort(self, s, cause, guard, detail=None):
        record = {
            'schema': SCHEMA, 'mode': self.mode, 'status': 'watchdog_abort', 'cause': cause,
            # W13: WHICH guard fired, and WHICH quantity the memory limit gates on
            'guard_triggered': guard, 'guard_detail': detail or {},
            'gating_measure': self.gating_measure,
            'gating_value_at_abort': s.get('gating_value'),
            'thresholds': self.thresholds(),
            'limit_bytes': self.limit, 'limit_gib': self.limit / GIB,
            'min_available_bytes': self.min_available,
            'rss_reached': s, 'stage_at_abort': s.get('stage'),
            'stage_since_s': round(time.time() - STAGE_REF['since'], 3),
            'elapsed_s': round(time.time() - self.t0, 3), 'peak_so_far': dict(self.peak),
            'ru_maxrss_self': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'last_samples': list(self.tail), 'utc': _utc(),
            'exit_code': EXIT_WATCHDOG,
        }
        try:
            with open(self.abort_path, 'w') as handle:
                json.dump(record, handle, indent=1, default=str)
            record['killed_descendants'] = _kill_descendants(self.pid)
            print(f'[WATCHDOG] ABORT ({guard}: {cause}): gating_measure={self.gating_measure} '
                  f'gating_value={s.get("gating_value")} rss_tree={s.get("rss_tree")} '
                  f'footprint_self={s.get("footprint_self")} measure={s.get("measure")} '
                  f'swap_used={s.get("swap_used")} sys_available={s.get("sys_available")} '
                  f'at stage "{s.get("stage")}" -- exiting {EXIT_WATCHDOG}',
                  file=sys.stderr, flush=True)
            sys.stdout.flush()
        finally:
            os._exit(EXIT_WATCHDOG)

    def run(self):
        with open(self.samples_path, 'a') as handle:
            while not self._halt.is_set():
                s = self.sample()
                self.n_samples += 1
                self._update_peak(s)
                self.tail.append(s)
                del self.tail[:-20]
                handle.write(json.dumps(s) + '\n')
                handle.flush()
                growth, baseline, span = self._swap_growth(s)
                if (s.get('gating_value') or 0) > self.limit:
                    self._abort(s, f'gating measure {self.gating_measure} above '
                                   f'{self.limit / GIB:.2f} GiB ({self.limit} bytes)',
                                'rss_limit',
                                {'gating_measure': self.gating_measure, 'gating_value': s.get('gating_value'),
                                 'limit_bytes': self.limit, 'rss_tree': s.get('rss_tree'),
                                 'measure_footprint_based': s.get('measure')})
                if self.min_available and s['sys_available'] < self.min_available:
                    self._abort(s, f'system available memory below {self.min_available / GIB:.1f} GiB '
                                   '(declared secondary trigger)', 'sys_available_floor',
                                {'sys_available': s.get('sys_available'),
                                 'min_available_bytes': self.min_available})
                # W13 item 3: thrashing guard -- absolute swap ceiling and swap growth rate
                if self.swap_used_limit and (s.get('swap_used') or 0) > self.swap_used_limit:
                    self._abort(s, f'swap used above {self.swap_used_limit / GIB:.2f} GiB (thrashing guard)',
                                'swap_used_limit',
                                {'swap_used': s.get('swap_used'),
                                 'swap_used_limit_bytes': self.swap_used_limit})
                if self.swap_growth_limit and growth > self.swap_growth_limit:
                    self._abort(s, f'swap used grew by {growth} bytes (above '
                                   f'{self.swap_growth_limit / GIB:.2f} GiB) within '
                                   f'{self.swap_growth_window_s:.0f} s (thrashing guard)',
                                'swap_growth_rate',
                                {'swap_used': s.get('swap_used'), 'window_baseline_swap_used': baseline,
                                 'growth_bytes': growth, 'window_span_s': span,
                                 'swap_growth_limit_bytes': self.swap_growth_limit,
                                 'swap_growth_window_s': self.swap_growth_window_s})
                self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
        final = self.sample()
        self._update_peak(final)
        return final


class StageLog:
    def __init__(self, path, watchdog):
        self.path, self.wd = path, watchdog
        self.entries = []

    @contextmanager
    def stage(self, name):
        previous = STAGE_REF['name']
        STAGE_REF['name'], STAGE_REF['since'] = name, time.time()
        t0 = time.time()
        before = self.wd.sample()
        _append_jsonl(self.path, {'event': 'start', 'stage': name, 't': round(t0 - self.wd.t0, 3),
                                  'rss_tree': before.get('rss_tree'),
                                  'footprint_self': before.get('footprint_self')})
        ok, err = True, None
        try:
            yield
        except BaseException as error:
            ok, err = False, f'{type(error).__name__}: {error}'
            raise
        finally:
            after = self.wd.sample()
            entry = {'event': 'end', 'stage': name, 'ok': ok, 'error': err,
                     't_start': round(t0 - self.wd.t0, 3), 'wall_s': round(time.time() - t0, 3),
                     'rss_tree_before': before.get('rss_tree'), 'rss_tree_after': after.get('rss_tree'),
                     'rss_self_after': after.get('rss_self'),
                     'footprint_self_after': after.get('footprint_self'),
                     'peak_measure_so_far': self.wd.peak['measure'],
                     'ru_maxrss_self_after': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            _append_jsonl(self.path, entry)
            self.entries.append(entry)
            STAGE_REF['name'], STAGE_REF['since'] = previous, time.time()


# ======================================================================================
#  CHILD: model counting
# ======================================================================================
def len_counts(block, pe):
    """Cheap: sum of len() over component objects (constructed data objects), per type."""
    out = {}
    for key, ctype in (('var', pe.Var), ('constraint', pe.Constraint), ('expression', pe.Expression),
                       ('param', pe.Param), ('objective', pe.Objective), ('suffix', pe.Suffix)):
        n = 0
        for comp in block.component_objects(ctype, active=None, descend_into=True):
            try:
                n += len(comp)
            except TypeError:
                n += 1
        out[key] = n
    return out


def deep_counts(block, pe):
    """Full iteration: fixed variables, active / equality constraints, active objectives."""
    nv = nfixed = 0
    for v in block.component_data_objects(pe.Var, descend_into=True):
        nv += 1
        if v.fixed:
            nfixed += 1
    nc = nact = neq = 0
    for c in block.component_data_objects(pe.Constraint, active=None, descend_into=True):
        nc += 1
        if c.active:
            nact += 1
            if c.equality:
                neq += 1
    nobj = sum(1 for _ in block.component_data_objects(pe.Objective, active=True, descend_into=True))
    return {'var_data': nv, 'var_fixed': nfixed, 'var_free': nv - nfixed, 'constraint_data': nc,
            'constraint_active': nact, 'constraint_active_equality': neq,
            'constraint_active_inequality': nact - neq, 'objective_active': nobj}


def _scenario_combinations(block):
    try:
        return len(block.scenarios_market) * len(block.scenarios_operation)
    except AttributeError:
        return None


def count_all_models(tso, dso, esso, pe, deep=True):
    blocks = []
    for year in tso:
        for day in tso[year]:
            m = tso[year][day]
            blocks.append({'agent': 'TSO', 'node': None, 'year': year, 'day': day, 'name': m.name,
                           'scenario_combinations': _scenario_combinations(m),
                           'len_counts': len_counts(m, pe), **({'deep': deep_counts(m, pe)} if deep else {})})
    for node in sorted(dso):
        for year in dso[node]:
            for day in dso[node][year]:
                m = dso[node][year][day]
                blocks.append({'agent': 'DSO', 'node': node, 'year': year, 'day': day, 'name': m.name,
                               'scenario_combinations': _scenario_combinations(m),
                               'len_counts': len_counts(m, pe),
                               **({'deep': deep_counts(m, pe)} if deep else {})})
    for node in sorted(esso):
        m = esso[node]
        blocks.append({'agent': 'ESSO', 'node': node, 'year': None, 'day': None, 'name': m.name,
                       'scenario_combinations': None, 'len_counts': len_counts(m, pe),
                       **({'deep': deep_counts(m, pe)} if deep else {})})
    agg = {}
    for b in blocks:
        a = agg.setdefault(b['agent'], {'models': 0, 'len_counts': {}, 'deep': {}})
        a['models'] += 1
        for k, v in b['len_counts'].items():
            a['len_counts'][k] = a['len_counts'].get(k, 0) + v
        for k, v in (b.get('deep') or {}).items():
            a['deep'][k] = a['deep'].get(k, 0) + v
    total = {'models': 0, 'len_counts': {}, 'deep': {}}
    for a in agg.values():
        total['models'] += a['models']
        for part in ('len_counts', 'deep'):
            for k, v in a[part].items():
                total[part][k] = total[part].get(k, 0) + v
    agg['ALL'] = total
    return blocks, agg


# ======================================================================================
#  CHILD: shared setup (planning from the derived case file, oracle baseline injection)
# ======================================================================================
def _child_env_check():
    bad = {k: os.environ.get(k) for k, v in THREAD_CAP_ENV.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f'CHILD REFUSES: thread caps not in force: {bad}')


def _launch_limit(out_dir):
    return int(_load_launch(out_dir)['thresholds']['rss_limit_bytes'])


def _child_watchdog(out_dir, mode):
    """The child's watchdog, configured from launch.json: the limit AND (W13) the gating
    measure and the thrashing-guard thresholds, so every guard in force is the one the parent
    recorded before the run."""
    th = _load_launch(out_dir).get('thresholds') or {}
    return Watchdog(
        out_dir, mode,
        limit_bytes=int(th['rss_limit_bytes']),
        min_available=int(th.get('min_available_bytes', MIN_AVAILABLE_BYTES)),
        gating_measure=th.get('gating_measure', GATING_MEASURE_DEFAULT),
        swap_used_limit=int(th.get('swap_used_limit_bytes', SWAP_USED_LIMIT_BYTES)),
        swap_growth_limit=int(th.get('swap_growth_limit_bytes', SWAP_GROWTH_LIMIT_BYTES)),
        swap_growth_window_s=float(th.get('swap_growth_window_s', SWAP_GROWTH_WINDOW_S)))


def _load_launch(out_dir):
    with open(os.path.join(out_dir, 'launch.json')) as handle:
        return json.load(handle)


def read_planning_from_derived_case(launch, out_dir, stages):
    """Production's own reader on the derived case file. data_dir is data/SRP1 (networks and
    market data are resolved from it); the derived file lives in the measurement directory
    and is passed as a path relative to data_dir. Diagram/result/log dirs are redirected into
    the measurement directory BEFORE reading (the reader plots scenarios; production would
    otherwise overwrite data/SRP1/Diagrams). planning.name is set to 'SRP1' (the case name)."""
    from shared_resources_planning import SharedResourcesPlanning
    case_path = os.path.join(REPO, launch['derived_case']['path'])
    rel = os.path.relpath(case_path, DATA_DIR)
    planning = SharedResourcesPlanning(DATA_DIR, rel)
    planning.name = 'SRP1'
    planning.results_dir = os.path.join(out_dir, 'planning_read', 'Results')
    planning.diagrams_dir = os.path.join(out_dir, 'planning_read', 'Diagrams')
    planning.logs_dir = os.path.join(planning.results_dir, 'Logs')
    with stages.stage('read_planning_problem (data, copula/KDE scenario generation, networks, ESS)'):
        planning.read_planning_problem()
    return planning


def inject_oracle_baseline(O, planning, launch):
    checksum = planning.scenario_metadata['combined_scenario_checksum']
    if launch['instance'] == 'srp1' and checksum != O.CANONICAL_CHECKSUM:
        raise RuntimeError(f'srp1 instance: scenario checksum {checksum} != canonical {O.CANONICAL_CHECKSUM}')
    if O._BASELINE is not None:
        raise RuntimeError('oracle baseline already loaded in this process; refusing to inject')
    O._BASELINE = {'planning': planning, 'checksum': checksum}
    return checksum


# ======================================================================================
#  P5.15 Addendum 27 item 5(a): --snapshots on|off
# ======================================================================================
def apply_snapshot_setting(planning, snapshots, record):
    """Applies `--snapshots` to ONE planning object and records what happened.

    'on' (default) changes nothing at all -- not even an assignment -- so the run is the
    committed behaviour. 'off' sets BOTH `admm_parameters.tso_snapshot_capture_mode` and
    `dso_snapshot_capture_mode` to 'off' (production validates the value and builds no
    pristine base, `shared_resources_planning._build_pristine_snapshot_bases`), then
    VERIFIES that the setting took effect and raises if it did not -- the setting must be
    demonstrable, not assumed. The record goes into the child's record under
    'snapshot_setting'."""
    params = planning.params.admm
    applied = {
        'requested': snapshots,
        'tso_mode_before': getattr(params, 'tso_snapshot_capture_mode', None),
        'dso_mode_before': getattr(params, 'dso_snapshot_capture_mode', None),
    }
    if snapshots == 'off':
        params.tso_snapshot_capture_mode = 'off'
        params.dso_snapshot_capture_mode = 'off'
    applied['tso_mode_after'] = getattr(params, 'tso_snapshot_capture_mode', None)
    applied['dso_mode_after'] = getattr(params, 'dso_snapshot_capture_mode', None)
    expected = ('off', 'off') if snapshots == 'off' else (applied['tso_mode_before'],
                                                          applied['dso_mode_before'])
    applied['expected'] = list(expected)
    applied['took_effect'] = (applied['tso_mode_after'], applied['dso_mode_after']) == expected
    applied['persistent_workers_enabled'] = bool(
        (getattr(params, 'persistent_workers', None) or {}).get('enabled', False))
    if record is not None:
        record['snapshot_setting'] = applied
    if not applied['took_effect']:
        raise RuntimeError(f'--snapshots {snapshots}: capture modes did not take effect: {applied}')
    return applied


def snapshot_hook_wrapper(inner_hook, snapshots, record):
    """Wraps a `run_admm_arm` `pre_solve_hook` so `--snapshots` is applied to the planning
    object the arm actually runs on, AFTER the inner hook has done its own configuration --
    the same wrap-the-hook pattern as `p515_s40_clone_capture_preflight._capture_mode_hook_
    override`, never an edit to the harness. The applied record is written both into the
    child's record (via `apply_snapshot_setting`) and into the arm report's
    `rule_eleven_checklist`."""
    def hook(planning, sed, candidate, report):
        inner_hook(planning=planning, sed=sed, candidate=candidate, report=report)
        applied = apply_snapshot_setting(planning, snapshots, record)
        report.setdefault('rule_eleven_checklist', {})['s44_snapshot_setting'] = applied
    return hook


def set_release_solution_bookkeeping(planning, value):
    """P5.15 Addendum 29 (W32): sets `SolverParameters.release_solution_bookkeeping` on the TSO's
    and every DSO's network parameters -- the `params` `NetworkData.optimize` hands to
    `network._run_smopf`, which consults the switch -- READS IT BACK, and raises if it did not
    take effect. The shared-ESS solver parameters are not touched (the ESSO solve path does not
    consult the switch). Returns the applied record."""
    value = bool(value)
    holders = [('tso', planning.transmission_network)] + [
        (f'dso_{n}', planning.distribution_networks[n]) for n in sorted(planning.distribution_networks)]
    before, read_back = {}, {}
    for name, holder in holders:
        solver_params = holder.params.solver_params
        before[name] = getattr(solver_params, 'release_solution_bookkeeping', None)
        solver_params.release_solution_bookkeeping = value
        read_back[name] = solver_params.release_solution_bookkeeping
    applied = {'requested': value, 'before': before, 'read_back': read_back,
               'solver_params_objects_distinct': len({id(h.params.solver_params) for _, h in holders}) == len(holders),
               'took_effect': len(holders) > 1 and all(v is value for v in read_back.values())}
    if not applied['took_effect']:
        raise RuntimeError(f'release_solution_bookkeeping={value} did not take effect: {applied}')
    return applied


def release_bookkeeping_hook_wrapper(inner_hook, record):
    """P5.15 Addendum 29 (W32): wraps a `run_admm_arm` `pre_solve_hook` (same pattern as
    `snapshot_hook_wrapper`) so `--release-solution-bookkeeping` is applied to the planning object
    the arm runs on, after the inner hook; recorded in the child record and in the arm report's
    `rule_eleven_checklist`."""
    def hook(planning, sed, candidate, report):
        inner_hook(planning=planning, sed=sed, candidate=candidate, report=report)
        applied = set_release_solution_bookkeeping(planning, True)
        if record is not None:
            record['release_solution_bookkeeping'] = applied
        report.setdefault('rule_eleven_checklist', {})['w32_release_solution_bookkeeping'] = applied
    return hook


def provenance_record(planning, instance, checksum):
    import p54r_provenance as P
    prov, _ = P.collect(planning)
    prov['scenario_checksum'] = checksum
    prov['checksum_matches_canonical'] = checksum == P.CANONICAL_CHECKSUM
    failures = P.check(prov)
    non_checksum = [f for f in failures if f['identity'] != 'scenario checksum']
    prov['gate_failures'] = failures
    prov['checksum_expected_to_differ'] = instance != 'srp1'
    prov['gate_passes_for_this_instance'] = (not non_checksum) and (
        instance != 'srp1' or checksum == P.CANONICAL_CHECKSUM)
    return prov


def planning_dimensions(planning):
    return {
        'years': {str(y): w for y, w in planning.years.items()},
        'days': dict(planning.days),
        'num_instants': planning.num_instants,
        'random_seed': planning.random_seed,
        'num_market_scenarios': planning.num_market_scenarios,
        'num_operation_scenarios': {
            planning.transmission_network.name: planning.transmission_network.num_oper_scenarios,
            **{planning.distribution_networks[n].name: planning.distribution_networks[n].num_oper_scenarios
               for n in sorted(planning.distribution_networks)}},
        'active_distribution_network_nodes': list(planning.active_distribution_network_nodes),
        'scenario_metadata': planning.scenario_metadata,
    }


def expected_block_counts(planning):
    n_yd = len(planning.years) * len(planning.days)
    n_dso = len(planning.distribution_networks)
    return {'tso_blocks': n_yd, 'dso_blocks': n_dso * n_yd, 'network_blocks': (1 + n_dso) * n_yd,
            'esso_models': len(planning.active_distribution_network_nodes),
            'solves_per_cycle': (1 + n_dso) * n_yd + len(planning.active_distribution_network_nodes)}


def declared_solve_profile(planning, cap):
    """The solve declaration for the timed cycle, DERIVED FROM THE INSTANCE (never hard-coded).

    solves_per_cycle = (1 + n_dso) x n_years x n_days + n_esso_nodes
      SRP1: (1 + 3) x 3 x 4 + 3 = 51; the paper instance: (1 + 3) x 5 x 4 + 3 = 83.
    base = solves_per_cycle x (cap + 1) -- one initialization round plus `cap` ADMM cycles;
    with `CYCLE_CAP` = 1 that is solves_per_cycle x 2 (102 at SRP1, 166 at paper scale).

    The base is NOT the strict count: production retries a failed local NLP (tier 1) and may
    then retry from a frozen snapshot (tier 2), so the identity gated on is
        observed == base + 1 * recovered_tier1 + 2 * recovered_tier2
    with the recovery counts taken from the run's OWN `network_failures_summary`
    (p515_s45_snapshot_off_failure_gate.py section (5), which does exactly this)."""
    n_years, n_days = len(planning.years), len(planning.days)
    n_dso = len(planning.distribution_networks)
    n_esso = len(planning.active_distribution_network_nodes)
    n_yd = n_years * n_days
    per_cycle = (1 + n_dso) * n_yd + n_esso
    return {
        'derivation': '(1 + n_dso) * n_years * n_days + n_esso_nodes, per cycle, from the planning object',
        'n_years': n_years, 'n_days': n_days, 'n_year_day_blocks': n_yd, 'n_dso': n_dso,
        'n_networks': 1 + n_dso, 'network_solves_per_cycle': (1 + n_dso) * n_yd,
        'n_esso_nodes': n_esso, 'esso_solves_per_cycle': n_esso,
        'solves_per_cycle': per_cycle, 'cap': cap, 'rounds': cap + 1,
        'rounds_note': 'one initialization round + cap ADMM cycles',
        'declared_base_solves': per_cycle * (cap + 1),
        'identity': ('observed == base + 1 * recovered_tier1 + 2 * recovered_tier2 (recovery counts '
                     'from the run\'s own network_failures_summary)'),
    }


def d_configuration_check(H, planning, sed, candidate, report, cap):
    """The campaign child's own D-configuration verification (no overrides), reused.

    Addendum 27 item 1 (W12): the spec DECLARES the case file's AA dict (`CASE_FILE_AA`), so
    the hook verifies the LOADED `anderson_acceleration` against that declaration instead of
    requiring AA off. A case file whose AA differs from the declaration in any key -- including
    one with no AA block at all, which loads the `admm_parameters` default
    {'enabled': False, 'memory': 5, 'regularization': 1e-10} -- fails
    `anderson_acceleration_case_file_matches_declaration` and the hook RAISES; it never
    proceeds silently. Returns the checks together with the declaration and the effective AA."""
    spec_like = {'configuration': {'overrides': {},
                                   'case_file_anderson_acceleration': dict(CASE_FILE_AA)},
                 'cap': cap, 'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES}
    holder = {}
    H._config_hook_factory(spec_like, holder, overrides={})(planning=planning, sed=sed,
                                                            candidate=candidate, report=report)
    return {'configuration_checks': holder.get('configuration_checks'),
            'case_file_anderson_acceleration_declared': dict(CASE_FILE_AA),
            'anderson_acceleration_effective': holder.get('anderson_acceleration_effective'),
            'overrides_applied': holder.get('overrides_applied'),
            'declaration_note': ('spec_like declares configuration.case_file_anderson_acceleration; '
                                 'the harness hook checks the loaded AA equals it exactly and raises '
                                 'otherwise (p515_s44_campaign_harness._config_hook_factory)')}


# ======================================================================================
#  CHILD: build mode (zero solves)
# ======================================================================================
class Interceptor:
    """Same contract as p515_s44_addendum26_confirmations._Interceptor: replaces `.optimize`
    on ONE planning instance's agents, records the call, returns "no result" (None per
    block/node). Never calls a solver; never fabricates a solution."""

    def __init__(self):
        self.calls = []

    def network(self, holder, kind):
        def _stub(model, *args, **kwargs):
            self.calls.append(kind)
            return {year: {day: None for day in holder.days} for year in holder.years}
        return _stub

    def esso(self, sed):
        def _stub(models, *args, **kwargs):
            self.calls.append('esso_coordination' if kwargs.get('cycle') is not None else 'esso_init')
            return {node_id: None for node_id in sed.active_distribution_network_nodes}
        return _stub

    def counts(self):
        out = {}
        for c in self.calls:
            out[c] = out.get(c, 0) + 1
        return out


@contextmanager
def per_block_build_recorder(network_module, pe, blocks_path, wd):
    """Class-level wrapper of `Network.build_model` (this process only, restored on exit):
    per block, the stage label for the watchdog, wall time, RSS after, cheap counts."""
    original = network_module.Network.build_model
    records = []

    def wrapped(self, params):
        agent = 'TSO' if self.is_transmission else 'DSO'
        node = None if self.is_transmission else getattr(self, 'tn_connection_nodeid', None)
        previous = STAGE_REF['name']
        STAGE_REF['name'] = f'{previous} :: build_model {agent} {self.name} {self.year} {self.day}'
        t0 = time.time()
        try:
            model = original(self, params)
        finally:
            STAGE_REF['name'] = previous
        s = wd.sample()
        rec = {'agent': agent, 'node': node, 'network': self.name, 'year': self.year, 'day': self.day,
               'wall_s': round(time.time() - t0, 3), 'rss_tree_after': s.get('rss_tree'),
               'footprint_self_after': s.get('footprint_self'),
               'scenario_combinations': _scenario_combinations(model),
               'len_counts_at_build_model': len_counts(model, pe)}
        _append_jsonl(blocks_path, rec)
        records.append(rec)
        return model

    network_module.Network.build_model = wrapped
    try:
        yield records
    finally:
        network_module.Network.build_model = original


def child_build(args):
    out_dir = label_dir(args.label)
    started = time.time()
    _child_env_check()
    wd = _child_watchdog(out_dir, 'build')
    wd.start()
    stages = StageLog(os.path.join(out_dir, 'stages_build.jsonl'), wd)
    launch = _load_launch(out_dir)

    from p513_solve_profile_guard import SolveProfileGuard
    guard = SolveProfileGuard(permitted=(), label='P5.15 S44 scale measurement BUILD (zero solves)').install()

    steps_not_exercised = [
        'every solve: the initialization solves (intercepted, "no result") and the ADMM loop',
        '_compute_common_admm_objective_scale on solved blocks -> the fixed-sigma calibration '
        'assertion (needs the initialization solves); case-file fixed sigma used (declared substitution)',
        'IPOPT SolverResults objects, multiplier suffix contents loaded by load_from, NL-writer transient, '
        'IPOPT process memory (exist only after real solves; --time-one-cycle measures them)',
        'p515_s40_polish_gap._build_floor_rows (the campaign child\'s zero-solve floor-row precheck: a '
        'transient planning copy + 3 ESSO builds, deleted before the run)',
        'the harness capture wrappers s38_pf_capture_hooks / s39_exempt_until_capture_hooks and '
        'esso_capture_hooks (per-solve sidecar writers)',
    ]
    record = {'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'mode': 'build',
              'label': args.label, 'instance': launch['instance'],
              'instance_definition': launch['instance_definition'], 'derived_case': launch['derived_case'],
              'candidate': C_STAR_LABEL, 'status': 'running', 'pid': os.getpid(),
              'thread_caps_seen': {k: os.environ.get(k) for k in THREAD_CAP_ENV},
              'steps_not_exercised': steps_not_exercised,
              'declared_substitutions': [
                  'initialization .optimize intercepted on the planning instance (returns no result)',
                  'objective scale: case-file fixed sigma (admm.objective_scale) used without the '
                  'calibration assertion against a computed sigma',
                  'tso_pristine_base / dso_pristine_base: production\'s own '
                  'shared_resources_planning._build_pristine_snapshot_bases is CALLED (since P5.15 '
                  'Addendum 27 item 5(a)); before that its clone expressions were replicated here']}
    try:
        with stages.stage('import production modules'):
            import pyomo.environ as pe
            import network as network_module
            import shared_resources_planning as srp
            import p515_g_g1_g4_admm_gates as G
            import p515_s44_campaign_harness as H
            O, R = G.O, G.R
        planning0 = read_planning_from_derived_case(launch, out_dir, stages)
        checksum = inject_oracle_baseline(O, planning0, launch)
        record['scenario_checksum'] = checksum
        record['checksum_matches_canonical'] = checksum == O.CANONICAL_CHECKSUM
        record['planning_dimensions'] = planning_dimensions(planning0)
        record['expected_block_counts'] = expected_block_counts(planning0)
        with stages.stage('provenance (IPOPT identity probe: `ipopt --version`, not a solve)'):
            record['provenance'] = provenance_record(planning0, launch['instance'], checksum)

        construct_report = {}
        with stages.stage('oracle construction: _construct_arm_planning (fresh_planning deepcopy, results '
                          'redirect, budget, candidate C*)'):
            planning, sed, candidate = G._construct_arm_planning(
                's44_scale_build', os.path.join(out_dir, 'arm_build'), construct_report,
                investment_map=C_STAR, eval_id=eval_id(args.label, 'build'),
                num_max_iters_override=BUILD_CAP, apply_rho=False)
        record['construct_report'] = construct_report
        record['d_configuration_checks'] = d_configuration_check(H, planning, sed, candidate,
                                                                 construct_report, BUILD_CAP)
        params = planning.params.admm
        # P5.15 Addendum 27 item 5(a): --snapshots, applied to THIS child's planning object
        # (the build child constructs its own arm planning, so there is no pre_solve_hook to
        # wrap here) and verified before anything is built.
        apply_snapshot_setting(planning, launch.get('snapshots', 'on'), record)
        # P5.15 Addendum 29 (W32): recorded here for completeness (the build child makes no solves).
        if launch.get('release_solution_bookkeeping'):
            record['release_solution_bookkeeping'] = set_release_solution_bookkeeping(planning, True)
        if planning.parallel_execution:
            raise RuntimeError('ParallelExecution is on; the build trace follows the sequential path')

        interceptor = Interceptor()
        tn = planning.transmission_network
        tn.optimize = interceptor.network(tn, 'tso')
        for dn in planning.distribution_networks.values():
            dn.optimize = interceptor.network(dn, 'dso')
        sed.optimize = interceptor.esso(sed)
        trace = {}
        blocks_path = os.path.join(out_dir, 'blocks_build.jsonl')
        with per_block_build_recorder(network_module, pe, blocks_path, wd) as block_records:
            with stages.stage('create_admm_variables'):
                cv, dv = srp.create_admm_variables(planning)
            with stages.stage('create_distribution_networks_models (sequential; optimize intercepted)'):
                dso_models, res_dso = srp.create_distribution_networks_models(
                    planning.distribution_networks, cv, candidate['total_capacity'],
                    parallel_execution=planning.parallel_execution)
            with stages.stage('create_transmission_network_model (optimize intercepted)'):
                tso_model, res_tso = srp.create_transmission_network_model(planning, cv, candidate['total_capacity'])
            with stages.stage('create_shared_energy_storage_model (optimize intercepted)'):
                esso_model, res_esso = srp.create_shared_energy_storage_model(sed, cv, candidate['investment'])
        results = {'tso': res_tso, 'dso': res_dso, 'esso': res_esso}
        trace['admm_local_solves_succeeded_on_intercepted_results (expected False)'] = \
            srp._admm_local_solves_succeeded(planning, results)
        with stages.stage('_prepare_distribution/transmission_objectives_for_admm'):
            srp._prepare_distribution_objectives_for_admm(planning.distribution_networks, dso_models)
            srp._prepare_transmission_objectives_for_admm(tn, tso_model)
        with stages.stage('_compute_common_admm_objective_scale (attempted on unsolved blocks)'):
            try:
                trace['sigma_computed_on_unsolved_blocks'] = srp._compute_common_admm_objective_scale(
                    planning, tso_model, dso_models)
            except Exception as error:  # noqa: BLE001 -- recorded trace artifact
                trace['sigma_computed_on_unsolved_blocks'] = f'{type(error).__name__}: {error}'
        if params.objective_scale is None:
            raise RuntimeError('case file has no fixed objective_scale; the zero-solve trace cannot continue')
        scale = params.objective_scale
        trace['objective_scale_used'] = scale
        with stages.stage('ADMM preparation: ESSO AL scale; update_*_to_admm (TSO/DSO under '
                          'patched_admm_objectives); ESSO AL objective; consensus initialization'):
            al_scale = srp._resolve_esso_al_scale(planning, params, scale)[0]
            with R.patched_admm_objectives():
                srp.update_distribution_models_to_admm(planning, dso_models, params, scale)
                srp.update_transmission_model_to_admm(planning, tso_model, params, scale)
            srp.update_shared_energy_storage_model_to_admm(planning, esso_model, params, al_scale_esso=al_scale)
            srp._initialize_shared_ess_consensus(planning, cv)
        trace['al_scale_esso'] = al_scale
        trace['shared_ess_initialization'] = params.shared_ess_initialization
        with stages.stage('update_interface_power_flow_variables (skips unsolved blocks)'):
            planning.update_interface_power_flow_variables(tso_model, dso_models, cv, dv, results, params,
                                                           update_tn=True, update_dns=True)
        with stages.stage('get_updated_capacities'):
            sed.get_updated_capacities(esso_model)
        pristine = {}
        with stages.stage('pristine clones kept by the ADMM loop (tso_pristine_base, dso_pristine_base)'):
            # P5.15 Addendum 27 item 5(a): production's own helper is called here (it holds
            # both the validation and the clone expressions), instead of the replica this
            # script carried before -- so the measured stage is the production path, and
            # --snapshots off reaches it exactly as a production run would.
            tso_pristine_base, dso_pristine_base = srp._build_pristine_snapshot_bases(
                params, tn, tso_model, planning.distribution_networks, dso_models)
            if tso_pristine_base is not None:
                pristine['tso'] = tso_pristine_base
            if dso_pristine_base is not None:
                pristine['dso7'] = dso_pristine_base
        trace['tso_snapshot_capture_mode'] = params.tso_snapshot_capture_mode
        trace['dso_snapshot_capture_mode'] = params.dso_snapshot_capture_mode
        trace['pristine_clone_blocks'] = {k: sum(len(v) for v in d.values()) for k, d in pristine.items()}

        build_complete = wd.sample()
        record['memory_at_build_complete'] = build_complete
        record['ru_maxrss_self_at_build_complete'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record['build_wall_s'] = round(time.time() - started, 3)
        STAGE_REF['name'] = 'counting (after build complete)'

        del planning.transmission_network.optimize
        for dn in planning.distribution_networks.values():
            del dn.optimize
        del sed.optimize

        with stages.stage('model counting (len per component type, all models; deep iteration, all models)'):
            blocks, agg = count_all_models(tso_model, dso_models, esso_model, pe, deep=not args.no_deep_counts)
        record['counts_by_agent'] = agg
        record['counts_per_model'] = blocks
        record['counts_note'] = (
            'len_counts = sum of len() over component objects (constructed data objects) per type, '
            'descend_into=True; deep = full iteration (fixed vars, active/equality constraints). '
            'Counted on the final models after ADMM preparation. The pristine clones (see '
            'trace.pristine_clone_blocks) are structural copies of the TSO blocks and the node-7 DSO '
            'blocks at loop start and are not re-counted.')
        record['blocks_at_build_model'] = block_records
        record['block_counts'] = {
            'tso_blocks': sum(len(v) for v in tso_model.values()),
            'dso_blocks': sum(len(v) for d in dso_models.values() for v in d.values()),
            'dso_blocks_per_node': {str(n): sum(len(v) for v in d.values()) for n, d in dso_models.items()},
            'esso_models': len(esso_model),
            'network_build_model_calls': len(block_records),
            'pristine_clone_blocks': trace['pristine_clone_blocks'],
        }
        record['block_counts']['network_blocks'] = (record['block_counts']['tso_blocks']
                                                    + record['block_counts']['dso_blocks'])
        record['block_counts_match_expected'] = (
            record['block_counts']['network_blocks'] == record['expected_block_counts']['network_blocks']
            and record['block_counts']['esso_models'] == record['expected_block_counts']['esso_models'])
        declared = {'dso': len(planning.distribution_networks), 'tso': 1, 'esso_init': 1}
        observed = interceptor.counts()
        record['interceptor_check'] = {'declared': declared, 'observed': observed,
                                       'exact_match': observed == declared}
        record['trace'] = trace
    except BaseException as error:  # noqa: BLE001 -- recorded, then re-raised as exit code
        guard.uninstall()
        final = wd.stop()
        record.update({'status': 'error', 'error': f'{type(error).__name__}: {error}',
                       'traceback': traceback.format_exc(), 'stage_at_error': STAGE_REF['name'],
                       'guard_counts': dict(guard.counts), 'watchdog_peak': wd.peak,
                       'memory_final': final, 'stages': stages.entries,
                       'wall_s': round(time.time() - started, 3)})
        _write_once_json(os.path.join(out_dir, 'build_error.json'), record)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return EXIT_ERROR

    guard.uninstall()
    guard_failures = guard.verify(0)
    final = wd.stop()
    record.update({
        'status': 'complete' if not guard_failures and record['interceptor_check']['exact_match'] else 'check_failed',
        'guard': {'permitted': [], 'counts': dict(guard.counts), 'declared_solves': 0,
                  'verify_failures': guard_failures},
        'watchdog': {'limit_bytes': wd.limit, 'sample_interval_s': SAMPLE_INTERVAL_S,
                     'n_samples': wd.n_samples, 'peak': wd.peak,
                     'min_available_bytes': MIN_AVAILABLE_BYTES,
                     # W13: the gating measure and every guard threshold in force
                     'gating_measure': wd.gating_measure, 'thresholds': wd.thresholds()},
        # W13: the gate that decides whether the cycle child runs uses the GATING measure;
        # the footprint-based comparison is kept alongside it, recorded, not gating.
        'peak_under_limit': wd.peak['gating_value'] < wd.limit,
        'peak_under_limit_gating_measure': wd.gating_measure,
        'peak_footprint_measure_under_limit': wd.peak['measure'] < wd.limit,
        'memory_final': final,
        'ru_maxrss_self_final': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'ru_maxrss_units': 'bytes on macOS',
        'stages': stages.entries,
        'wall_s': round(time.time() - started, 3),
        'utc_end': _utc(),
    })
    _write_once_json(os.path.join(out_dir, 'build_record.json'), record)
    print(f"[SCALE-BUILD] status={record['status']} network_blocks={record['block_counts']['network_blocks']} "
          f"esso={record['block_counts']['esso_models']} peak_tree_rss={wd.peak['rss_tree']} "
          f"at_build_complete={build_complete.get('rss_tree')} solves={guard.counts['permitted_solve']} "
          f"wall={record['wall_s']:.1f}s", flush=True)
    return EXIT_OK if record['status'] == 'complete' else EXIT_ERROR


# ======================================================================================
#  CHILD: one-cycle timing mode (a production evaluation, cap 1)
# ======================================================================================
@contextmanager
def srp_stage_wrappers(srp, network_module, stamps):
    """Stage labels + first-call timestamps (this process only, restored on exit)."""
    names = {
        'create_admm_variables': 'init: create_admm_variables',
        'create_distribution_networks_models': 'init: DSO build + solve',
        'create_transmission_network_model': 'init: TSO build + solve',
        'create_shared_energy_storage_model': 'init: ESSO build + solve',
        'update_distribution_coordination_models_and_solve': 'cycle: DSO solves',
        'update_transmission_coordination_model_and_solve': 'cycle: TSO solves',
        'update_shared_energy_storages_coordination_model_and_solve': 'cycle: ESSO solves',
    }
    originals = {name: getattr(srp, name) for name in names}

    def make(name, label, original):
        def wrapped(*a, **k):
            previous = STAGE_REF['name']
            STAGE_REF['name'], STAGE_REF['since'] = label, time.time()
            t0 = time.time()
            stamps.setdefault(name, []).append({'t_start': t0})
            try:
                return original(*a, **k)
            finally:
                stamps[name][-1]['t_end'] = time.time()
                STAGE_REF['name'] = previous
        return wrapped

    original_smopf = network_module.Network.run_smopf

    def smopf(self, *a, **k):
        previous = STAGE_REF['name']
        STAGE_REF['name'] = f'{previous} :: solve {self.name} {self.year} {self.day}'
        try:
            return original_smopf(self, *a, **k)
        finally:
            STAGE_REF['name'] = previous

    for name, label in names.items():
        setattr(srp, name, make(name, label, originals[name]))
    network_module.Network.run_smopf = smopf
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(srp, name, original)
        network_module.Network.run_smopf = original_smopf


def objective_scale_record(rows, stdout_text, assert_factor):
    """sigma (fixed and computed), the ESSO AL scale, and whether the sigma CALIBRATION check
    passed, taken from the run itself.

    Primary evidence is production's own print in the arm's stdout --
    `[ADMM OF SCALE] Fixed sigma in force sigma_fixed=... | sigma_computed=... | ratio=... |
    assert_factor=...` and `[ADMM ESSO AL SCALE] ... al_scale_esso=...`
    (shared_resources_planning._resolve_common_admm_objective_scale / _resolve_esso_al_scale).
    The calibration check is an ASSERTION INSIDE PRODUCTION: it raises outside
    [1/factor, factor], so reaching this point at all means it did not raise; the ratio is
    re-derived here and re-checked so the outcome is a recorded number, not an inference.
    The per-cycle row fields (sigma_fixed/sigma_computed/al_scale_esso, run-level constants
    carried on every row) are recorded alongside and must agree."""
    row = rows[0] if rows else {}
    out = {'sigma_fixed': row.get('sigma_fixed'), 'sigma_computed': row.get('sigma_computed'),
           'al_scale_esso': row.get('al_scale_esso'),
           'assert_factor_from_case_file': assert_factor,
           'source': 'cycle_trajectory row 1 (run-level constants) + the arm stdout prints'}
    m = re.search(r'\[ADMM OF SCALE\] Fixed sigma in force sigma_fixed=([0-9.eE+-]+) \| '
                  r'sigma_computed=([0-9.eE+-]+) \| ratio=([0-9.eE+-]+) \| '
                  r'assert_factor=([0-9.eE+-]+)', stdout_text or '')
    out['printed_of_scale_line'] = m.group(0) if m else None
    if m:
        out['printed'] = {'sigma_fixed': float(m.group(1)), 'sigma_computed': float(m.group(2)),
                          'ratio': float(m.group(3)), 'assert_factor': float(m.group(4))}
    m2 = re.search(r'\[ADMM ESSO AL SCALE\][^\n]*al_scale_esso=([0-9.eE+-]+)', stdout_text or '')
    out['printed_esso_al_scale_line'] = m2.group(0) if m2 else None
    if m2:
        out['printed_al_scale_esso'] = float(m2.group(1))
    sf, sc = out['sigma_fixed'], out['sigma_computed']
    factor = (out.get('printed') or {}).get('assert_factor', assert_factor)
    if sf and sc and factor:
        ratio = sc / sf
        out['ratio_sigma_computed_over_fixed'] = ratio
        out['calibration_range'] = [1.0 / factor, factor]
        out['calibration_check_passed'] = bool((1.0 / factor) <= ratio <= factor)
    else:
        out['calibration_check_passed'] = None
    out['calibration_note'] = (
        'production raises ValueError in _resolve_common_admm_objective_scale outside the range; '
        'a completed run necessarily passed it -- the value here makes the margin explicit')
    def _close(a, b):
        # the prints carry 7 significant figures (`:.6e`), so agreement is relative, not exact
        return a is not None and b is not None and abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1.0)

    out['row_agrees_with_print'] = (
        None if not out.get('printed') else
        (_close(out['printed']['sigma_fixed'], sf) and _close(out['printed']['sigma_computed'], sc)
         and _close(out.get('printed_al_scale_esso'), out['al_scale_esso'])))
    out['row_print_agreement_tolerance'] = 'relative 1e-6 (the prints are :.6e)'
    return out


def child_cycle(args):
    out_dir = label_dir(args.label)
    started = time.time()
    _child_env_check()
    wd = _child_watchdog(out_dir, 'cycle')
    wd.start()
    stages = StageLog(os.path.join(out_dir, 'stages_cycle.jsonl'), wd)
    launch = _load_launch(out_dir)
    record = {'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'mode': 'cycle',
              'label': args.label, 'instance': launch['instance'], 'derived_case': launch['derived_case'],
              'candidate': C_STAR_LABEL, 'status': 'running', 'pid': os.getpid(),
              'cap': CYCLE_CAP,
              'path': ('p515_g_g1_g4_admm_gates.run_admm_arm (the campaign child\'s evaluation call), D '
                       'configuration verified by p515_s44_campaign_harness._config_hook_factory, cap 1; '
                       'includes run_admm_arm\'s own esso_capture_hooks; without the s38/s39 capture '
                       'wrappers and the floor-row precheck')}
    from p513_solve_profile_guard import SolveProfileGuard
    import p514_n_instrumented_cstar as N
    guard = SolveProfileGuard(N.PERMITTED, label='P5.15 S44 scale measurement ONE CYCLE').install()
    stamps = {}
    try:
        with stages.stage('import production modules'):
            import network as network_module
            import shared_resources_planning as srp
            import p515_g_g1_g4_admm_gates as G
            import p515_s44_campaign_harness as H
            O = G.O
        planning0 = read_planning_from_derived_case(launch, out_dir, stages)
        checksum = inject_oracle_baseline(O, planning0, launch)
        record['scenario_checksum'] = checksum
        record['planning_dimensions'] = planning_dimensions(planning0)
        expected = expected_block_counts(planning0)
        # W12: the solve DECLARATION, derived from the instance and stated BEFORE the run.
        # `declared_base_solves` = solves_per_cycle x (1 initialization + CYCLE_CAP cycles);
        # the count gated on is base + 1*tier1 + 2*tier2, reconciled after the run against the
        # run's own failure summary (see `declared_solve_profile`).
        declared = declared_solve_profile(planning0, CYCLE_CAP)
        declared_solves = declared['declared_base_solves']
        if declared['solves_per_cycle'] != expected['solves_per_cycle']:
            raise RuntimeError(f'solve declaration disagrees with expected_block_counts: '
                               f'{declared} vs {expected}')
        record['expected_block_counts'] = expected
        record['declared_solve_profile'] = declared
        record['declared_solves'] = declared_solves
        record['case_file_anderson_acceleration_declared'] = dict(CASE_FILE_AA)
        record['case_file_aa_declaration_agreement'] = check_case_file_aa_declaration()
        record['objective_scale_assert_factor_from_case_file'] = getattr(
            planning0.params.admm, 'objective_scale_assert_factor', None)
        prov = provenance_record(planning0, launch['instance'], checksum)
        record['provenance'] = prov
        if [f for f in prov['gate_failures'] if f['identity'] != 'scenario checksum']:
            raise RuntimeError(f'provenance: non-canonical solver identity: {prov["gate_failures"]}')
        holder = {}
        # Addendum 27 item 1 (W12): declare the case file's AA dict, as d_configuration_check
        # does for the build child -- without it the hook refuses the AA-on case file.
        spec_like = {'configuration': {'overrides': {},
                                       'case_file_anderson_acceleration': dict(CASE_FILE_AA)},
                     'cap': CYCLE_CAP,
                     'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES}
        cycle_dir = os.path.join(out_dir, 'one_cycle')
        # P5.15 Addendum 27 item 5(a): --snapshots is applied to the arm's planning object by
        # wrapping the harness's own pre_solve_hook (never editing it); with 'on' the wrapper
        # changes nothing but still records the modes in force.
        cycle_hook = snapshot_hook_wrapper(H._config_hook_factory(spec_like, holder, overrides={}),
                                           launch.get('snapshots', 'on'), record)
        # P5.15 Addendum 29 (W32): --release-solution-bookkeeping, applied after the hooks above.
        if launch.get('release_solution_bookkeeping'):
            cycle_hook = release_bookkeeping_hook_wrapper(cycle_hook, record)
        with srp_stage_wrappers(srp, network_module, stamps), \
                stages.stage('run_admm_arm, cap 1 (initialization + one ADMM cycle)'):
            report, report_path = G.run_admm_arm(
                's44_scale_cycle', cycle_dir, k_override=None, investment_map=C_STAR,
                num_max_iters_override=CYCLE_CAP, eval_id=eval_id(args.label, 'cycle'), apply_rho=False,
                full_diagnostics_in_rows=True,
                pre_solve_hook=cycle_hook)
        t_end = time.time()
    except BaseException as error:  # noqa: BLE001
        guard.uninstall()
        final = wd.stop()
        record.update({'status': 'error', 'error': f'{type(error).__name__}: {error}',
                       'traceback': traceback.format_exc(), 'stage_at_error': STAGE_REF['name'],
                       'guard_counts': dict(guard.counts), 'watchdog_peak': wd.peak, 'memory_final': final,
                       'stamps': stamps, 'stages': stages.entries, 'wall_s': round(time.time() - started, 3)})
        _write_once_json(os.path.join(out_dir, 'cycle_error.json'), record)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return EXIT_ERROR
    guard.uninstall()
    # W12 solve reconciliation: the base was declared before the run; the count actually gated
    # on adds the run's OWN recovery attempts (1 extra solve per tier-1 recovery, 2 per tier-2),
    # and the process-wide guard -- armed for the whole child -- is verify()-ed EXACTLY against
    # it (too few fails as loudly as too many).
    _classes = ((report.get('network_failures_summary') or {}).get('classes')) or {}
    _tier1 = _classes.get('recovered_tier1', 0)
    _tier2 = _classes.get('recovered_tier2', 0)
    reconciled_solves = declared_solves + _tier1 + 2 * _tier2
    guard_failures = guard.verify(reconciled_solves)
    final = wd.stop()

    def first(name, key):
        return (stamps.get(name) or [{}])[0].get(key)

    init_start = first('create_admm_variables', 't_start')
    cycle_start = first('update_distribution_coordination_models_and_solve', 't_start')
    esso_cycle_end = first('update_shared_energy_storages_coordination_model_and_solve', 't_end')
    printed = None
    stdout_text = ''
    try:
        with open(os.path.join(REPO, report['stdout_path'])) as handle:
            stdout_text = handle.read()
        m = re.search(r'Iteration 1: ([0-9.]+) s', stdout_text)
        printed = float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        printed = None
    rows = report.get('cycle_trajectory') or []
    record['objective_scale'] = objective_scale_record(
        rows, stdout_text, record.get('objective_scale_assert_factor_from_case_file'))
    record.update({
        'status': 'complete' if not guard_failures else 'solve_count_mismatch',
        'guard': {'permitted': [list(p) for p in N.PERMITTED], 'counts': dict(guard.counts),
                  'declared_base_solves': declared_solves,
                  'recovered_tier1': _tier1, 'recovered_tier2': _tier2,
                  'reconciled_expected_solves': reconciled_solves,
                  'observed_solves': guard.counts['permitted_solve'],
                  'identity': record['declared_solve_profile']['identity'],
                  'identity_holds': guard.counts['permitted_solve'] == reconciled_solves,
                  'declared_solves': reconciled_solves, 'verify_failures': guard_failures},
        'run_admm_arm_identity_holds_note': ('run_admm_arm\'s own solve_profile.identity_holds uses the SRP1 '
                                             'constant 51 per cycle and no recovery term; superseded here by '
                                             'the declared base reconciled with the run\'s own tier counts'),
        'anderson_acceleration_effective_in_child': holder.get('anderson_acceleration_effective'),
        'configuration_checks': holder.get('configuration_checks'),
        'timing_s': {
            'initialization (create_admm_variables -> first cycle DSO dispatch)':
                (cycle_start - init_start) if init_start and cycle_start else None,
            'cycle_1 (first DSO dispatch -> ESSO coordination return)':
                (esso_cycle_end - cycle_start) if cycle_start and esso_cycle_end else None,
            'cycle_1_production_printed (Iteration 1: X s)': printed,
            'run_admm_arm_wall_clock_s': report.get('wall_clock_s'),
            'child_total_s': t_end - started,
        },
        'stage_stamps': stamps,
        'cycle_row': {k: (rows[0].get(k) if rows else None) for k in (
            'cycle', 'local_solves_ok', 'recourse', 'gross_operational_cost', 'objective_change_abs')},
        'network_failures_summary': report.get('network_failures_summary'),
        'arm_solve_profile': report.get('solve_profile'),
        'g_report_path': os.path.relpath(report_path, REPO),
        'watchdog': {'limit_bytes': wd.limit, 'n_samples': wd.n_samples, 'peak': wd.peak,
                     # W13: the gating measure and every guard threshold in force
                     'gating_measure': wd.gating_measure, 'thresholds': wd.thresholds()},
        'memory_final': final,
        'ru_maxrss_self': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'ru_maxrss_children_ipopt': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'stages': stages.entries,
        'utc_end': _utc(),
    })
    _write_once_json(os.path.join(out_dir, 'cycle_record.json'), record)
    print(f"[SCALE-CYCLE] status={record['status']} timing={record['timing_s']} "
          f"solves={guard.counts['permitted_solve']}/{reconciled_solves} "
          f"(base {declared_solves} + {_tier1} tier1 + 2 x {_tier2} tier2) "
          f"aa={record['anderson_acceleration_effective_in_child']} "
          f"sigma={record['objective_scale'].get('sigma_fixed')} "
          f"al_scale_esso={record['objective_scale'].get('al_scale_esso')} "
          f"peak_tree_rss={wd.peak['rss_tree']}", flush=True)
    return EXIT_OK if record['status'] == 'complete' else EXIT_ERROR


# ======================================================================================
#  PARENT
# ======================================================================================
class ParentLog:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        line = f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        with open(self.path, 'a') as handle:
            handle.write(line + '\n')


def derive_case(instance, overrides_cli):
    with open(SOURCE_CASE) as handle:
        case = json.load(handle)
    spec = dict(INSTANCES[instance])
    for key in ('years', 'num_market_scenarios', 'num_operation_scenarios'):
        if overrides_cli.get(key) is not None:
            spec[key] = overrides_cli[key]
    changes = []
    if spec['years'] is not None:
        changes.append({'key': 'Years', 'from': case['Years'], 'to': spec['years']})
        case['Years'] = spec['years']
    if spec['num_market_scenarios'] is not None:
        changes.append({'key': 'NumMarketScenarios', 'from': case['NumMarketScenarios'],
                        'to': spec['num_market_scenarios']})
        case['NumMarketScenarios'] = spec['num_market_scenarios']
    if spec['num_operation_scenarios'] is not None:
        tn = case['TransmissionNetwork']
        changes.append({'key': f'TransmissionNetwork[{tn["name"]}].num_operation_scenarios',
                        'from': tn['num_operation_scenarios'], 'to': spec['num_operation_scenarios']})
        tn['num_operation_scenarios'] = spec['num_operation_scenarios']
        for dn in case['DistributionNetworks']:
            changes.append({'key': f'DistributionNetworks[{dn["name"]}].num_operation_scenarios',
                            'from': dn['num_operation_scenarios'], 'to': spec['num_operation_scenarios']})
            dn['num_operation_scenarios'] = spec['num_operation_scenarios']
    return case, spec, changes


def _other_harness_processes():
    me = {os.getpid(), os.getppid()}
    found = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if proc.info['pid'] in me or 'python' not in cmd:
            continue
        if HARNESS_PATTERN.search(cmd):
            found.append({'pid': proc.info['pid'], 'cmdline': cmd[:300]})
    return found


def run_child(mode, args, out_dir, log):
    backstop_bytes = int(args.rss_limit_gib * GIB) + BACKSTOP_MARGIN_BYTES
    cmd = [PYTHON, '-u', SCRIPT_PATH, '--child', mode, '--label', args.label]
    if mode == 'build' and args.no_deep_counts:
        cmd.append('--no-deep-counts')
    env = dict(os.environ)
    env.update(THREAD_CAP_ENV)
    out_path = os.path.join(out_dir, f'{mode}_child_stdout.log')
    err_path = os.path.join(out_dir, f'{mode}_child_stderr.log')
    for p in (out_path, err_path):
        if os.path.exists(p):
            raise RuntimeError(f'refusing to overwrite {p}')
    t0 = time.time()
    backstop = None
    # W13: the parent backstop polls the SAME gating measure the child's watchdog gates on
    # (otherwise the parent would keep killing on footprint after the child stopped doing so);
    # both quantities are still peak-tracked here.
    measure_name = args.watchdog_measure
    peak = {'measure': 0, 'rss_tree': 0, 'gating_measure': measure_name, 'gating_value': 0}
    with open(out_path, 'w') as out_h, open(err_path, 'w') as err_h:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=out_h, stderr=err_h)
        log(f'{mode} child started pid={proc.pid}: {" ".join(cmd)}')
        last_note = t0
        while True:
            pid, status, ru = os.wait4(proc.pid, os.WNOHANG)
            if pid != 0:
                break
            mem = tree_memory(proc.pid)
            if mem:
                gate = gating_value(mem, measure_name)
                peak['measure'] = max(peak['measure'], mem['measure'])
                peak['rss_tree'] = max(peak['rss_tree'], mem['rss_tree'])
                peak['gating_value'] = max(peak['gating_value'], gate)
                if gate > backstop_bytes and backstop is None:
                    backstop = {'mode': mode, 'status': 'parent_backstop_kill', 'memory': mem,
                                'gating_measure': measure_name, 'gating_value': gate,
                                'limit_bytes': backstop_bytes, 'utc': _utc(),
                                'elapsed_s': time.time() - t0}
                    killed = _kill_descendants(proc.pid)
                    os.kill(proc.pid, signal.SIGKILL)
                    backstop['killed'] = [proc.pid] + killed
                    log(f'PARENT BACKSTOP: child tree {measure_name} above '
                        f'{backstop_bytes / GIB:.1f} GiB -- killed')
            if time.time() - last_note > 60:
                log(f'{mode} child alive: tree_rss={mem and mem["rss_tree"]} '
                    f'footprint_measure={mem and mem["measure"]} swap_used={psutil.swap_memory().used} '
                    f'peak_{measure_name}={peak["gating_value"]} peak_footprint_measure={peak["measure"]}')
                last_note = time.time()
            time.sleep(PARENT_POLL_S)
        proc.returncode = os.waitstatus_to_exitcode(status)
    exit_code = proc.returncode if backstop is None else EXIT_BACKSTOP
    info = {'mode': mode, 'command': cmd, 'exit_code': exit_code, 'raw_returncode': proc.returncode,
            'wall_s': time.time() - t0, 'parent_observed_peak': peak,
            'parent_backstop_gating_measure': measure_name, 'parent_backstop_bytes': backstop_bytes,
            'wait4_rusage': {k: getattr(ru, k) for k in ('ru_utime', 'ru_stime', 'ru_maxrss', 'ru_minflt',
                                                          'ru_majflt', 'ru_nvcsw', 'ru_nivcsw')},
            'thread_caps_in_child_env': THREAD_CAP_ENV}
    if backstop is not None:
        _write_once_json(os.path.join(out_dir, f'parent_backstop_kill_{mode}.json'), backstop)
    with open(os.path.join(out_dir, f'{mode}_exit_code.txt'), 'w') as handle:
        handle.write(f'{exit_code}\n')
    log(f'{mode} child exited {exit_code} after {info["wall_s"]:.1f}s')
    return info


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def write_manifest(out_dir, label):
    files = {}
    for root, _dirs, names in os.walk(out_dir):
        for name in sorted(names):
            path = os.path.join(root, name)
            if name == 'manifest_sha256.json':
                continue
            files[os.path.relpath(path, REPO)] = {'sha256': sha256_file(path), 'bytes': os.path.getsize(path)}
    work = {}
    work_root = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P56A', 'evals')
    for mode in ('build', 'cycle'):
        d = os.path.join(work_root, eval_id(label, mode))
        for root, _dirs, names in os.walk(d):
            for name in sorted(names):
                path = os.path.join(root, name)
                work[os.path.relpath(path, REPO)] = {'sha256': sha256_file(path), 'bytes': os.path.getsize(path)}
    manifest = {'stage': STAGE, 'label': label, 'generated_utc': _utc(), 'files': files,
                'p56a_working_dir_files': work}
    _write_once_json(os.path.join(out_dir, 'manifest_sha256.json'), manifest)
    return manifest


def scale_measurement_block(args, launch, build_record, build_info, cycle_info, cycle_record):
    """P5.15 Addendum 27 W12 item 3: every quantity the calibration is read from, in ONE place.

    Per-cycle wall time; the initialization time SEPARATELY; peak RSS of the build child and of
    the cycle child separately; the computed sigma and the ESSO AL scale; whether the sigma
    calibration check passed; the snapshot mode; the effective AA; solves observed vs declared
    base vs reconciled; and the failure / recovery counts. Every value is copied from the
    children's own records -- nothing is recomputed here."""
    build_record = build_record or {}
    cycle_record = cycle_record or {}
    timing = cycle_record.get('timing_s') or {}
    failures = cycle_record.get('network_failures_summary') or {}
    guard = cycle_record.get('guard') or {}
    scale = cycle_record.get('objective_scale') or {}
    init_key = 'initialization (create_admm_variables -> first cycle DSO dispatch)'
    cycle_key = 'cycle_1 (first DSO dispatch -> ESSO coordination return)'
    return {
        'instance': args.instance, 'label': args.label,
        'snapshots_requested': launch.get('snapshots'),
        'snapshot_setting_build': build_record.get('snapshot_setting'),
        'snapshot_setting_cycle': cycle_record.get('snapshot_setting'),
        'release_solution_bookkeeping_requested': launch.get('release_solution_bookkeeping', False),
        'release_solution_bookkeeping_cycle': cycle_record.get('release_solution_bookkeeping'),
        'case_file_anderson_acceleration_declared': launch.get('case_file_anderson_acceleration_declared'),
        'anderson_acceleration_effective_cycle': cycle_record.get('anderson_acceleration_effective_in_child'),
        'anderson_acceleration_effective_build': (
            (build_record.get('d_configuration_checks') or {}).get('anderson_acceleration_effective')
            if isinstance(build_record.get('d_configuration_checks'), dict) else None),
        'timing_s': {
            'build_child_wall_s': build_info.get('wall_s') if build_info else None,
            'build_stage_wall_s': build_record.get('build_wall_s'),
            'initialization_s': timing.get(init_key),
            'one_cycle_s': timing.get(cycle_key),
            'one_cycle_production_printed_s': timing.get('cycle_1_production_printed (Iteration 1: X s)'),
            'run_admm_arm_wall_clock_s': timing.get('run_admm_arm_wall_clock_s'),
            'cycle_child_wall_s': cycle_info.get('wall_s') if cycle_info else None,
        },
        'peak_rss_bytes': {
            'build_child_watchdog_peak': (build_record.get('watchdog') or {}).get('peak'),
            'build_child_parent_observed_peak': (build_info or {}).get('parent_observed_peak'),
            'cycle_child_watchdog_peak': (cycle_record.get('watchdog') or {}).get('peak'),
            'cycle_child_parent_observed_peak': (cycle_info or {}).get('parent_observed_peak'),
            'cycle_child_ru_maxrss_self': cycle_record.get('ru_maxrss_self'),
            'cycle_child_ru_maxrss_children_ipopt': cycle_record.get('ru_maxrss_children_ipopt'),
            'watchdog_limit_bytes': (launch.get('thresholds') or {}).get('rss_limit_bytes'),
            # W13: which quantity the limit gated on, and every guard threshold in force
            'watchdog_gating_measure': (launch.get('thresholds') or {}).get('gating_measure'),
            'watchdog_thresholds': launch.get('thresholds'),
            'watchdog_thresholds_note': launch.get('thresholds_note'),
        },
        'objective_scale': {
            'sigma_fixed': scale.get('sigma_fixed'), 'sigma_computed': scale.get('sigma_computed'),
            'ratio_sigma_computed_over_fixed': scale.get('ratio_sigma_computed_over_fixed'),
            'assert_factor': (scale.get('printed') or {}).get(
                'assert_factor', scale.get('assert_factor_from_case_file')),
            'calibration_check_passed': scale.get('calibration_check_passed'),
            'al_scale_esso': scale.get('al_scale_esso'),
        },
        'solves': {
            'declared_base': guard.get('declared_base_solves'),
            'derivation': (cycle_record.get('declared_solve_profile') or {}).get('derivation'),
            'solves_per_cycle': (cycle_record.get('declared_solve_profile') or {}).get('solves_per_cycle'),
            'recovered_tier1': guard.get('recovered_tier1'), 'recovered_tier2': guard.get('recovered_tier2'),
            'reconciled_expected': guard.get('reconciled_expected_solves'),
            'observed': guard.get('observed_solves'),
            'identity': guard.get('identity'), 'identity_holds': guard.get('identity_holds'),
            'guard_verify_failures': guard.get('verify_failures'),
            'build_child_zero_solve_guard': build_record.get('guard'),
        },
        'failures': {
            'n_blocks_with_failures': failures.get('n_blocks'),
            'classes': failures.get('classes'),
            'n_frozen_snapshots': failures.get('n_frozen_snapshots'),
            'n_esso_recovery_events': failures.get('n_esso_recovery_events'),
            'local_solves_ok_cycle_1': (cycle_record.get('cycle_row') or {}).get('local_solves_ok'),
        },
    }


def main_parent(args):
    from p513_solve_profile_guard import SolveProfileGuard  # pyomo only; no model code
    parent_guard = SolveProfileGuard(permitted=(), label='P5.15 S44 scale measurement parent (never solves)').install()
    out_dir = label_dir(args.label)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.label):
        print(f'REFUSED: label must match [A-Za-z0-9_.-]+: {args.label!r}', file=sys.stderr)
        return EXIT_REFUSED
    if os.path.exists(out_dir):
        print(f'REFUSED: output directory exists (write-once): {out_dir}', file=sys.stderr)
        return EXIT_REFUSED
    for mode in ('build', 'cycle'):
        wd_path = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P56A', 'evals', eval_id(args.label, mode))
        if os.path.exists(wd_path):
            print(f'REFUSED: working-dir id already used (never reusable): {wd_path}', file=sys.stderr)
            return EXIT_REFUSED
    runs_alone = args.instance != 'srp1'
    if runs_alone:
        held = [p for p in (CAMPAIGN_LOCK_PATH, LEGACY_RUN_LOCK_PATH) if os.path.exists(p)]
        others = _other_harness_processes()
        if held or others:
            print(f'REFUSED: instance {args.instance!r} runs alone; locks held={held} '
                  f'harness processes={others}', file=sys.stderr)
            return EXIT_REFUSED
    try:
        fd = os.open(OWN_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f'REFUSED: another scale measurement holds {OWN_LOCK_PATH}', file=sys.stderr)
        return EXIT_REFUSED
    with os.fdopen(fd, 'w') as handle:
        json.dump({'pid': os.getpid(), 'label': args.label, 'started_utc': _utc()}, handle)
    try:
        os.makedirs(out_dir)
        log = ParentLog(os.path.join(out_dir, 'parent_run.log'))
        log(f'{STAGE}: label={args.label} instance={args.instance} time_one_cycle={args.time_one_cycle} '
            f'rss_limit_gib={args.rss_limit_gib} watchdog_measure={args.watchdog_measure} '
            f'snapshots={args.snapshots} min_available_gib={MIN_AVAILABLE_BYTES / GIB:.1f} '
            f'swap_used_limit_gib={SWAP_USED_LIMIT_BYTES / GIB:.1f} '
            f'swap_growth_limit_gib={SWAP_GROWTH_LIMIT_BYTES / GIB:.1f}/{SWAP_GROWTH_WINDOW_S:.0f}s')
        case_dir = os.path.join(out_dir, 'case')
        os.makedirs(case_dir)
        overrides_cli = {'years': (json.loads(args.override_years) if args.override_years else None),
                         'num_market_scenarios': args.override_market_scenarios,
                         'num_operation_scenarios': args.override_operation_scenarios}
        # W12: the case-file AA declaration must agree with p515_s45_a0_campaign.CASE_FILE_AA
        # BEFORE anything is launched (it raises on disagreement).
        aa_declaration = check_case_file_aa_declaration()
        log(f'case-file AA declaration {CASE_FILE_AA} (agrees with {CASE_FILE_AA_SOURCE}: '
            f'{aa_declaration["agree"]})')
        case, spec, changes = derive_case(args.instance, overrides_cli)
        case_path = os.path.join(case_dir, f'SRP1__{args.instance}.json')
        with open(case_path, 'w') as handle:
            json.dump(case, handle, indent='\t')
        derived = {'path': os.path.relpath(case_path, REPO), 'sha256': sha256_file(case_path),
                   'source': SOURCE_CASE_REL, 'source_sha256': sha256_file(SOURCE_CASE),
                   'source_last_commit': _git(['log', '-1', '--format=%H', '--', SOURCE_CASE_REL]),
                   'changes_vs_source': changes,
                   'cli_overrides': {k: v for k, v in overrides_cli.items() if v is not None}}
        tracked_dirty = _git(['status', '--porcelain', '--untracked-files=no'])
        launch = {
            'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'label': args.label,
            'instance': args.instance, 'instance_definition': spec, 'derived_case': derived,
            'candidate': C_STAR_LABEL, 'argv': sys.argv, 'interpreter': PYTHON, 'script': os.path.relpath(SCRIPT_PATH, REPO),
            'script_sha256': sha256_file(SCRIPT_PATH), 'git_head': _git(['rev-parse', 'HEAD']),
            'git_tracked_changes': tracked_dirty.splitlines(),
            'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH'),
            # W13: EVERY guard threshold, and the NAME of the quantity the memory limit gates
            # on, stated here before the run and read back by both children.
            'thresholds': {'rss_limit_bytes': int(args.rss_limit_gib * GIB), 'rss_limit_gib': args.rss_limit_gib,
                           'rss_limit_is_spec_default': args.rss_limit_gib == RSS_LIMIT_GIB_DEFAULT,
                           'gating_measure': args.watchdog_measure,
                           'gating_measure_sample_field': GATING_MEASURE_FIELD[args.watchdog_measure],
                           'gating_measure_is_default': args.watchdog_measure == GATING_MEASURE_DEFAULT,
                           'gating_measures_available': list(GATING_MEASURES),
                           'parent_backstop_bytes': int(args.rss_limit_gib * GIB) + BACKSTOP_MARGIN_BYTES,
                           'parent_backstop_gating_measure': args.watchdog_measure,
                           'min_available_bytes': MIN_AVAILABLE_BYTES,
                           'swap_used_limit_bytes': SWAP_USED_LIMIT_BYTES,
                           'swap_growth_limit_bytes': SWAP_GROWTH_LIMIT_BYTES,
                           'swap_growth_window_s': SWAP_GROWTH_WINDOW_S,
                           'sample_interval_s': SAMPLE_INTERVAL_S},
            'thresholds_note': (
                'P5.15 Addendum 27 W13. The memory limit gates on `gating_measure`: "rss_tree" '
                '(default) = RSS of the child and its descendants; "footprint" = the pre-W13 '
                'quantity `measure` = max(rss_tree, macOS phys_footprint + descendants\' RSS), '
                'which counts COMPRESSED pages. Both are computed and recorded on every sample '
                'whichever gates. Guards in force: (1) gating measure > rss_limit_bytes; '
                '(2) system available memory < min_available_bytes; (3) thrashing -- swap used '
                '> swap_used_limit_bytes, or swap used grown by more than swap_growth_limit_bytes '
                'within any swap_growth_window_s window.'),
            'runs_alone_enforced': runs_alone, 'time_one_cycle_requested': args.time_one_cycle,
            # Addendum 27 item 1 (W12): the AA dict both spec_like declarations hand the harness hook
            'case_file_anderson_acceleration_declared': dict(CASE_FILE_AA),
            'case_file_aa_declaration_agreement': aa_declaration,
            'case_file_aa_note': ('the harness hook (p515_s44_campaign_harness._config_hook_factory) '
                                  'checks the LOADED admm.anderson_acceleration equals this declaration '
                                  'exactly and RAISES otherwise; a case file without AA loads the '
                                  "admm_parameters default (enabled False, no reject_policy), which does "
                                  'not equal the declaration, so the hook refuses rather than proceeding'),
            'release_solution_bookkeeping': bool(args.release_solution_bookkeeping),
            'release_solution_bookkeeping_note': (
                'P5.15 Addendum 29 (W32): when true, SolverParameters.release_solution_bookkeeping is set '
                'on the TSO and every DSO network params of the child planning object (read back, '
                'recorded under "release_solution_bookkeeping"); network._run_smopf then clears '
                'model.solutions and result.solution after each successful load. Default false = '
                'the pre-W32 behaviour.'),
            'snapshots': args.snapshots,
            'snapshots_note': ('P5.15 Addendum 27 item 5(a): "on" = committed behaviour (capture modes '
                               'untouched); "off" = both admm_parameters.*_snapshot_capture_mode set to '
                               '\'off\' on the planning object, verified per child and recorded in '
                               'build_record.json / cycle_record.json under "snapshot_setting"'),
            'machine': {'total_memory_bytes': psutil.virtual_memory().total, 'cpu_count': os.cpu_count(),
                        'available_at_launch': psutil.virtual_memory().available},
            'started_utc': _utc(), 'parent_pid': os.getpid(),
        }
        _write_once_json(os.path.join(out_dir, 'launch.json'), launch)
        log(f'derived case {derived["path"]} sha256={derived["sha256"][:16]} changes={len(changes)}')

        summary = {'schema': SCHEMA, 'label': args.label, 'instance': args.instance, 'launch': 'launch.json'}
        build_info = run_child('build', args, out_dir, log)
        summary['build_child'] = build_info
        build_record = _read_json(os.path.join(out_dir, 'build_record.json'))
        abort = _read_json(os.path.join(out_dir, 'watchdog_abort_build.json'))
        summary['build_status'] = (build_record or {}).get('status') or (abort and 'watchdog_abort') or \
            ('parent_backstop_kill' if build_info['exit_code'] == EXIT_BACKSTOP else 'error')
        if build_record:
            summary['build'] = {
                'd_configuration_checks': build_record.get('d_configuration_checks'),
                'snapshot_setting': build_record.get('snapshot_setting'),
                'block_counts': build_record.get('block_counts'),
                'counts_by_agent': build_record.get('counts_by_agent'),
                'memory_at_build_complete': build_record.get('memory_at_build_complete'),
                'watchdog_peak': (build_record.get('watchdog') or {}).get('peak'),
                'ru_maxrss_self_final': build_record.get('ru_maxrss_self_final'),
                'guard': build_record.get('guard'), 'interceptor_check': build_record.get('interceptor_check'),
                'scenario_checksum': build_record.get('scenario_checksum'),
                'stage_wall_s': [(e['stage'], e['wall_s'], e['rss_tree_after']) for e in build_record.get('stages', [])],
            }
        if abort:
            summary['build_abort'] = {k: abort.get(k) for k in (
                'cause', 'guard_triggered', 'guard_detail', 'gating_measure', 'gating_value_at_abort',
                'thresholds', 'stage_at_abort', 'rss_reached', 'elapsed_s', 'peak_so_far')}
        exit_code = build_info['exit_code']
        if args.time_one_cycle:
            fits = (build_info['exit_code'] == EXIT_OK and build_record is not None
                    and build_record.get('status') == 'complete' and build_record.get('peak_under_limit'))
            summary['time_one_cycle'] = {'requested': True, 'build_fits_under_watchdog': bool(fits)}
            if fits:
                cycle_info = run_child('cycle', args, out_dir, log)
                summary['cycle_child'] = cycle_info
                cycle_record = _read_json(os.path.join(out_dir, 'cycle_record.json'))
                cycle_abort = _read_json(os.path.join(out_dir, 'watchdog_abort_cycle.json'))
                summary['cycle_status'] = (cycle_record or {}).get('status') or (cycle_abort and 'watchdog_abort') \
                    or ('parent_backstop_kill' if cycle_info['exit_code'] == EXIT_BACKSTOP else 'error')
                if cycle_record:
                    summary['cycle'] = {k: cycle_record.get(k) for k in (
                        'timing_s', 'guard', 'watchdog', 'ru_maxrss_self', 'ru_maxrss_children_ipopt',
                        'cycle_row', 'network_failures_summary', 'declared_solve_profile',
                        'objective_scale', 'snapshot_setting', 'anderson_acceleration_effective_in_child',
                        'configuration_checks', 'arm_solve_profile')}
                if cycle_abort:
                    summary['cycle_abort'] = {k: cycle_abort.get(k) for k in (
                        'cause', 'guard_triggered', 'guard_detail', 'gating_measure',
                        'gating_value_at_abort', 'thresholds', 'stage_at_abort', 'rss_reached',
                        'elapsed_s', 'peak_so_far')}
                exit_code = exit_code or cycle_info['exit_code']
            else:
                summary['time_one_cycle']['skipped_reason'] = 'build did not complete under the watchdog'
                log('time-one-cycle SKIPPED: build did not complete under the watchdog')
        parent_guard.uninstall()
        summary['parent_guard'] = {'counts': dict(parent_guard.counts), 'verify_failures': parent_guard.verify(0)}
        # W12 item 3: the single block the calibration is read from, every quantity in one place.
        summary['scale_measurement'] = scale_measurement_block(
            args, launch, build_record, build_info, summary.get('cycle_child'),
            _read_json(os.path.join(out_dir, 'cycle_record.json')))
        summary['exit_code'] = exit_code
        summary['ended_utc'] = _utc()
        _write_once_json(os.path.join(out_dir, 'summary.json'), summary)
        log(f'summary: build_status={summary["build_status"]} exit={exit_code}')
        write_manifest(out_dir, args.label)
        log(f'manifest written; done ({out_dir})')
        return exit_code
    finally:
        try:
            os.remove(OWN_LOCK_PATH)
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument('--instance', choices=sorted(INSTANCES), help='instance preset')
    parser.add_argument('--label', required=True, help='write-once output label')
    parser.add_argument('--time-one-cycle', action='store_true',
                        help='after a complete build under the watchdog, time ONE ADMM cycle (real solves)')
    parser.add_argument('--snapshots', choices=('on', 'off'), default='on',
                        help=('P5.15 Addendum 27 item 5(a): FrozenSMOPF snapshot capture. '
                              '"on" (default) = the committed behaviour, both capture modes left '
                              'exactly as the case file/defaults set them (lightweight: pristine '
                              'TSO and node-7 DSO clones are built). "off" sets BOTH '
                              'admm_parameters.tso_snapshot_capture_mode and '
                              'dso_snapshot_capture_mode to \'off\' on the planning object of the '
                              'build child and (through the pre_solve_hook) of the cycle child, so '
                              'no pristine base is cloned and no per-cycle capture is taken.'))
    parser.add_argument('--release-solution-bookkeeping', action='store_true',
                        help=('P5.15 Addendum 29 (W32): switch SolverParameters.release_solution_bookkeeping '
                              'ON for the TSO and every DSO (network._run_smopf clears model.solutions and '
                              'result.solution after each successful load). Default off = pre-W32 behaviour.'))
    parser.add_argument('--rss-limit-gib', type=float, default=RSS_LIMIT_GIB_DEFAULT,
                        help='watchdog limit in GiB (default 24 = the spec; lower ONLY to test the abort path)')
    parser.add_argument('--watchdog-measure', choices=GATING_MEASURES, default=GATING_MEASURE_DEFAULT,
                        help=('P5.15 Addendum 27 W13: WHICH measured quantity the memory limit '
                              'gates on. "rss_tree" (default) = RSS of the child and its '
                              'descendants -- what actually occupies physical RAM. "footprint" = '
                              'the pre-W13 quantity max(rss_tree, macOS phys_footprint + '
                              'descendants\' RSS), which counts COMPRESSED pages and aborted '
                              'label paper_cycle_snapoff_r1 at 25.78 GB footprint while rss_tree '
                              'was 15.93 GB and the machine had 12.14 GB available. Both are '
                              'recorded on every sample whichever gates; the name in force is '
                              'written into launch.json, every record and watchdog_abort_*.json.'))
    parser.add_argument('--no-deep-counts', action='store_true',
                        help='skip the full-iteration counts (fixed vars, active constraints)')
    parser.add_argument('--override-years', default=None,
                        help='JSON {year: weight} replacing the preset years (recorded)')
    parser.add_argument('--override-market-scenarios', type=int, default=None)
    parser.add_argument('--override-operation-scenarios', type=int, default=None)
    parser.add_argument('--child', choices=('build', 'cycle'), default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child == 'build':
        return child_build(args)
    if args.child == 'cycle':
        return child_cycle(args)
    if not args.instance:
        parser.error('--instance is required')
    return main_parent(args)


if __name__ == '__main__':
    sys.exit(main())
