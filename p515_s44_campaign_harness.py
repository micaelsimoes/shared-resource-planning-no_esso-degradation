"""
P5.15 Addendum 25 item 1 -- the Step 4 CAMPAIGN HARNESS.

Authority: PLANNER_BRIEF_2026-09-13.md Addenda 25 and 26; STEP4_DFO_METHOD.md
section 2 (oracle contract; 2.5 record, 2.7 interface); frozen spec v14
`data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`,
`item1_harness`.

================================================================================
PROCESS MODEL
================================================================================
`evaluate(batch, ctx) -> records` (STEP4_DFO_METHOD.md 2.7). Every candidate
of the batch is evaluated in its OWN OS process -- a fresh interpreter
(`<python> -u p515_s44_campaign_harness.py --child ...`), never a fork of the
parent -- with the thread caps `THREAD_CAP_ENV` (OMP / MKL / OPENBLAS /
VECLIB / NUMEXPR = 1) placed in the child's environment before it starts (the
IPOPT executables it launches inherit them). At most `ctx.concurrency`
children run at once; the parent polls them (`os.wait4`, non-blocking, so
each child's own rusage is captured), fills free slots in batch order, and
returns one record per candidate, in batch order.

THE PARENT NEVER SOLVES: `main_parent`-level callers arm
`SolveProfileGuard(permitted=())` for the whole campaign and verify 0 at the
end (the gate script `p515_s44_gate.py` does). The parent imports no model
code on the evaluation path; it only spawns, watches and reads records.

EACH CHILD (one evaluation) runs the production oracle exactly as the D
oracle's case-file runs do (`p515_s43_aa_run.py` / `p515_s40_polish_gap.py`
pattern): `p515_g_g1_g4_admm_gates.run_admm_arm(label, eval_dir,
investment_map=<candidate>, num_max_iters_override=<spec cap>,
eval_id=<derived>, apply_rho=False, full_diagnostics_in_rows=True, ...)`
inside the SAME capture wrappers (`s38_pf_capture_hooks`,
`s39_exempt_until_capture_hooks`) and with the same terminal writer
(`write_boyd_terminal_s35ref`), so every artifact the D reference carries is
produced under the same names. Configuration = the case file alone; the
ONLY Python-side configuration action is `configuration.overrides` of the
frozen campaign spec, restricted to `SUPPORTED_OVERRIDE_KEYS` (empty for the
D configuration; the AA arm will name `anderson_acceleration`).

================================================================================
LOCKS
================================================================================
ONE campaign-level lock, `CAMPAIGN_LOCK_PATH` (`.p515_s44_campaign.lock`,
repo root), created O_EXCL by the parent for the whole campaign; content =
JSON {pid, campaign_id, campaign_spec_sha256, started_utc}; removed on exit.
The children NEVER take the legacy one-run lock (`.p515_g_gate.lock`,
`p515_g_g1_g4_admm_gates._acquire_exclusive_run_lock`) -- `run_admm_arm`
does not take it either (only the legacy harnesses' own `main`s do), so N
children run concurrently. Instead each child REFUSES TO START unless the
campaign lock exists, names ITS parent's pid (`os.getppid()`) and the same
campaign spec sha256 -- a child can run only under the live campaign that
spawned it. The parent refuses to start if EITHER lock exists. The legacy
lock is not taken by the parent (spec v14: the campaign lock replaces it for
campaign use). Since Addendum 25 item 2 (gate-ruling follow-up) the legacy
lock function `_acquire_exclusive_run_lock` refuses while the campaign lock
exists; both acquirers re-check the other lock AFTER creating their own, so a
simultaneous start cannot let both proceed.

Why concurrency is safe (checked, not assumed; see the worker report): each
evaluation has its own `eval_id` working dir (`p56a_oracle.fresh_planning`
isolates every holder's `logs_dir`, hence every IPOPT `output_file`, TSO,
DSO and ESSO), its own `results_dir` (`_set_results_dir_for_arm`, hence its
own FrozenSMOPF snapshots), its own output files, and pyomo's temporary NL/
SOL files are uniquely named; `ParallelExecution` is false and persistent
workers are off, so no process pool is shared.

================================================================================
WHAT EACH EVALUATION WRITES (write-once dir `<campaign_root>/evals/<key16>_<label>/`)
================================================================================
Parent: `launch.json` (command, env caps, start time, pid), `child_stdout.log`,
`child_stderr.log` (both streams), `exit_code.txt`, `wait4_rusage.json`.
Child: every `run_admm_arm` artifact (g_<label>.json with the full per-cycle
trajectory, heartbeat_<label>.json -- updated every ESSO solve --, stdout,
esso_capture/, leak/network-failure/recovery sidecars, esso_models pickle,
results/FrozenSMOPF), the capture sidecars (recourse-jump, ESS/PF entry
strides, SoH floor, ESS exempt-until state -- all appended EVERY cycle, so
they survive a crash), the terminal artifacts (boyd_terminal.json,
component_levels_terminal.json, interface_settlement_detail_s31c.json,
interface_voltage_terminal.json), `per_cycle_record.jsonl` (standard
per-cycle subset, derived from the trajectory), `evaluation_record.json`
(the STEP4 2.5 record, schema `RECORD_SCHEMA`), `child_manifest_sha256.json`.

Working-dir ids (`p56a_oracle.WORK_DIR/<id>`): `p515s44_<campaign_id>_<key16>_run`
and `..._precheck` -- derived from the campaign id and the canonical
candidate key; the child refuses if either exists (never reusable).

================================================================================
PER-EVALUATION CONFIGURATION AND THE POST-CERTIFICATION STEP (Addendum 25 item 2)
================================================================================
A spec entry is one EVALUATION (candidate x configuration). An entry may carry
its own `overrides` (replacing the campaign-level ones; ONLY
`anderson_acceleration.{enabled, reject_policy}` -- `validate_overrides`) and a
`post_certification` request (`resolve_post_certification`):
`persist_certified_models`, `hull_polish`, and `reference` = the D evaluation
of the SAME candidate (certified, no overrides; its record and component levels
hash-recorded in the spec at freeze time and re-verified by the child before the
run and before use). `eval_key` = the candidate key for the case-file
configuration, else sha256{candidate_key, overrides} (`evaluation_key`); it names
the eval dir and working-dir ids, so one campaign can hold C* under D and under AA.
Addendum 27 item 1: a spec may declare `configuration.case_file_anderson_acceleration`
(the exact AA dict the case file loads to, `validate_case_file_anderson_acceleration`);
the configuration hook then checks the loaded dict equals it instead of requiring
case-file AA off, `eval_key` becomes sha256{candidate_key, effective AA dict,
overrides} (never the bare candidate key), and each entry records its effective AA
dict. Specs without the declaration keep their exact meaning and keys.
Addendum 27 (W5, pre-A1 fixes): the record's `bar` is the max GROSS cost step
over the last 10 cycles (`_max_step_last_n`; the net-recourse step production
records as `objective_change_abs` is kept as `bar_net_recourse_step_reported`);
a post-certification reference is D iff its EFFECTIVE AA is off (declaration +
overrides; undeclared records: no overrides); error / parent-synthesized
records carry `anderson_acceleration_effective_in_child` and
`case_file_sha256_in_child` (None when unknowable).
Addendum 27 (W14, the A1 year ladder): an entry may carry `investment_year` --
the SINGLE cohort year its candidate is placed at (`canonical_candidate`, hence
the candidate key); omitted => 2025, so every pre-W14 spec key, eval key, eval
dir and working-dir id is byte-identical. The child validates the year against
THIS instance's investment years (`instance_investment_years`, read from the
shared-ESS data, not a literal) and forwards it to `run_admm_arm`, which writes
`candidate['investment'][node][year]` there. Multi-cohort (staging) candidates
are NOT supported.
Addenda 28-29 (W20, the ageing batch): an entry may carry `model_variant` -- a
MODEL VARIANT of the shared-ESS ageing law, a dict of EXACTLY
`MODEL_VARIANT_KEYS` {eol_retention_r, calendar_retention_per_year,
available_energy_soh_point, ageing_enabled} (`validate_model_variant`). It is
applied in the child, in the configuration hook (before any ESSO model is
built), to the evaluation's OWN deep-copied shared-ESS parameters
(`apply_model_variant`; the committed case files are never edited), then READ
BACK from ESSO models built by production (`model_variant_readback`: k, phi and
the SoH-point mode recovered numerically from the built rows) -- once on
pre-run probes (refusing on any mismatch) and once, post-run, on clones of the
run's own ESSO models. It enters the eval key (`evaluation_key`); an entry
without it keeps its exact key. A spec holding one carries
`model_variant_label` == `MODEL_VARIANT_LABEL` at the top level and on each such
entry, and every record of such an entry carries the variant and the label.
Addendum 30 (W21, evaluation identity of the ESS ageing baseline): a spec may declare
`configuration.ess_ageing_baseline` -- the EXACT ageing dict the ESS parameters file
`ESS_PARAMS_FILE_REL` loads to (`validate_ess_ageing_baseline`,
`ess_ageing_parameters_as_loaded`; types included) -- with a non-empty
`ess_ageing_baseline_label`. `freeze_campaign_spec` refuses unless the file loads to the
declaration, and pins the file (`configuration.ess_params_file`: path, sha256, last
commit). The declaration enters the eval key (`evaluation_key`), so the same candidate under
two ageing baselines never shares a key. The child refuses unless the file hashes to the
pin, the LOADED parameters equal the declaration and every ESS carries its soh_min / phi / k
(`verify_ess_ageing_in_child`), and -- without a model variant -- the declaration's k, phi
and floor bound are read back from probe ESSO models (`ess_ageing_readback_models`); post-run
the read-back is repeated on clones and the ageing trajectory is captured. Undeclared specs
keep their exact format, keys and behaviour.
Addendum 34 (W33, the flexibility-price ladder): an entry may carry `flex_price_multiplier` -- a positive
float m (`validate_flex_price_multiplier`; absent or 1.0 = today's price). In the child, LAST in the
configuration hook (before any DSO model is built), every DSO block's `network[year][day].cost_flex` -- the
hourly profile per (year, day), growth included, that `model_construction_helpers.flexibility_cost` bakes into
the DSO objective as constants at BUILD time -- is replaced by a NEW array m * cost_flex (never in place: the
same array object is bound to the TSO and every DSO), uniformly over years / days / hours; the TSO's arrays and
`planning.cost_flex` are untouched and no file is edited (`apply_flex_price_multiplier`). READ-BACK: probe DSO
blocks built by production before and after, objective standard repn compared -- flex coefficients = m x the
m = 1 coefficients, everything else identical -- refusing on any mismatch; post-run the coefficients of the
run's own DSO models are read back (`flex_price_readback_run_models`). m enters the eval key ONLY when != 1.0
(`evaluation_key`), with `flex_price_label` == FLEX_PRICE_LABEL on the spec and the entry; every record of such
an entry carries the multiplier (and the label when != 1.0). Entries without it keep their exact format and keys.
P5.15 Addendum 39 (W47, the multi-scenario pilot): a spec may declare `configuration.derived_instance` -- a
DERIVED case file written by `p515_s44_scale_measurement.derive_case` (it edits only Years / NumMarketScenarios /
num_operation_scenarios of data/SRP1/SRP1.json), declared by EXACTLY `DERIVED_INSTANCE_KEYS`
(`validate_derived_instance`): its repo-relative path and sha256, the scenario checksum production's reader
computes for it, the instance label (never 'srp1'), and its source and changes (provenance). `freeze_campaign_spec`
refuses unless the file hashes to the declaration. In the child, FIRST (before anything reads the oracle baseline),
`install_derived_instance` hashes the file again, reads it with the scale harness's own reader
(`p515_s44_scale_measurement.read_planning_from_derived_case`, plots redirected into the eval dir), refuses unless
the combined scenario checksum equals the declaration, and installs it with `p56a_oracle.install_baseline`
(W35's installable-baseline route), so every later `fresh_planning` -- the precheck and the run -- is the derived
instance. An entry may carry `interface_deviation_premium` = {'alpha': a >= 0, 'floor': None | float}
(`validate_interface_deviation_premium`): row 18's premium, applied in the configuration hook to
`planning.params.admm.interface_deviation_premium` before any model is built and read back from the run's own
DSO models afterwards. Both enter the eval key (`evaluation_key`) only when declared; every other spec keeps its
exact format, keys and behaviour. With either declared, the post-run hook ALSO writes, zero solves and BEFORE any
post-certification step (so on the terminal, unpolished models): `MULTISCENARIO_TERMINAL_FILE`
(`multiscenario_terminal_capture`: per-block interface dispersion with per-DSO max-over-blocks RMS, E|d| and
sum omega d^2; the row 18 charge and its read-back; the settlement split with the covariance recomputed
independently; per-scenario costs reconciled to the block recourse; per-scenario interface profiles; the
scenario-free shared-ESS schedule; the interface-voltage mismatch; the sigma calibration inputs), and production's
own operational-planning workbook (`SharedResourcesPlanning.write_operational_planning_results_to_excel`, with the
run's own SolverResults and primal evolution). A capture error is recorded in the record and the child exits 2
after writing it (the evaluation itself stands), as a post-certification error does.
P5.15 Addendum 40 ruling 1 (W64, the alpha row): for the same derived-instance / premium evaluations, and ONLY for
them, the child also (i) asserts `assert_alpha_row_capture_paths` before any solve; (ii) wraps six production
functions pass-through for the run (`alpha_row_run_hooks`): at ACTIVATION (after the initialisation solve, before any
ADMM-cycle solve) it writes `ACTIVATION_READBACK_FILE` (row 18 alpha / rows / pair state on every DSO block,
penalty_gen_curtailment 0 and settlement weight 1 on every block) and the initialisation-identity record (cycle-0
gross cost, float.hex, registered in `<campaign root>/init_identity/` and compared bitwise with every record of the
same candidate), RAISING on any failure; and every cycle appends `per_cycle_response_record` to
`PER_CYCLE_RESPONSE_FILE`, merged by cycle into `per_cycle_record.jsonl` (`PER_CYCLE_RESPONSE_FIELDS`); (iii) writes
`RESPONSE_TERMINAL_FILE` on the terminal models right after the multi-scenario capture and before the workbook and
any post-certification step (`response_terminal_capture`: the dual-based curtailment entries with the W53 audit's
helpers, the W44 coordination record per DSO block, flexibility legs, the row 18 charge by leg, the P |d| form).
Zero solves throughout. `evaluation_key` is unchanged; every other evaluation runs exactly as before.
P5.15 Addendum 46 ruling 7 (W84, Planner rulings Q2 / Q3 on W83): EVERY evaluation now persists, in its eval dir, the
per-attempt IPOPT floor-status records production returns in `state['network_ipopt_solve_records']`
(`NETWORK_IPOPT_SOLVE_RECORDS_FILE`: one JSON line per TSO / DSO IPOPT attempt, `round` 0 = initialisation, k = cycle
k) and the convergence-depth tail state `state['convergence_depth_tail']` (`CONVERGENCE_DEPTH_TAIL_STATE_FILE`),
written FIRST in the post-run hook (before any terminal step) by `persist_convergence_depth_capture` and summarised
in the record. The tail stays OFF by default (`admm_parameters.ADMMParameters.convergence_depth_tail`); a launcher
enables it ONLY by declaring `configuration.convergence_depth_tail` = {'enabled': bool, 'compl_inf_tol': float} in
its frozen spec (`validate_convergence_depth_tail`; written into the spec only when declared, so undeclared specs
keep their exact format). Rule eleven for it: the child asserts BEFORE the run that the capture path exists and
records whether the tail is enabled for THIS run (`assert_convergence_depth_tail_capture`, in every record, error
records included); the configuration hook applies a declaration, reads it back and REFUSES before any solve when the
tail in force differs from the declaration -- in particular when it is on without one
(`apply_convergence_depth_tail_declaration`); after the run the tail state production returns must agree with the
declaration (`convergence_depth_tail_state_check`), else the record says so and the child exits 2. Zero solves.
W84 kept the declaration out of `evaluation_key`; W85 (Planner ruling Q1 on W84) puts it IN, only when declared
(enabled True or False): a tail-declared evaluation never shares a key -- hence never a cache hit -- with a pre-tail
evaluation of the same candidate, and every undeclared key is byte-identical to its pre-W85 value.
W86 (Planner ruling Q1 on W85, the W33 flex-multiplier precedent): only a declaration that CHANGES THE COMPUTATION
enters the key -- enabled True (`convergence_depth_tail_in_key`). A declared-OFF tail runs exactly what production's
default runs (none of the tail functions is called, its `compl_inf_tol` is never read), so it shares the UNDECLARED
key and may reuse a cached undeclared evaluation; every declared-ON key is unchanged from W85.
W85 (Planner ruling Q3 on W84): the same capture is also APPENDED PER ROUND while the run is in progress
(`ConvergenceDepthAppender`, `convergence_depth_append_hooks`: `NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE` and
`CONVERGENCE_DEPTH_APPEND_EVENTS_FILE`, fsync'd after every write, the tail checklist as the events file's first line
before any solve), so a child that raises keeps every completed round plus the round in flight (drained on the
exception path), and a killed child keeps every completed round; both failure records -- the child's and the
parent-synthesised one -- carry `recover_convergence_depth_append`. On a completed run the appended records file must
be byte-identical to the end-of-run file and the tail state rebuilt from the events must equal the returned state
(`reconcile`), else the child exits 2 after writing its record.
P5.15 Addendum 51 (W98, the post-certification continuation): an entry may carry `certification_continuation`
(`validate_certification_continuation`, implemented by `p515_s53_w98_continuation_hooks`): the child asserts its
preconditions before any solve and enters `continuation_hooks` FIRST (innermost wrappers): the certification rule is
disabled, the certifying regime (AA off, tight tail on, rho frozen) is held for every cycle after the declared N, an
early-stop rule ends the loop through production's own exit test, and every block's recourse is recorded every cycle.
It enters the eval key; entries without it keep their exact keys, format and behaviour.
AFTER the run, in the child, inside `run_admm_arm`'s post_run_hook (same live
models/state), `run_post_certification` does, only if the trajectory is
certified under the spec's bar (else it records `status: skipped` + reason):
(b) |Q - Q_ref| <= 1.5e-4 Q_ref and (c) the decomposition vs the reference
(`p515_s43_aa_run._cost_decomposition_vs_d(..., reference_dir=...)`: residual
<= 1.0, other priced components identically 0); then persists the certified
TSO/DSO models (`p515_s42_exact_fix_rerun._persist_certified_models`, BEFORE the
polish mutates them); then the interval-hull polish
(`p515_s41_hull_polish._polish_all_blocks_hull`: gate on the sum of per-block
changes, plus the settlement-excluded change, the settlement remainder and the
non-degenerate active-bound counts). Output: `post_certification.json`,
`hull_bound_detail.json`, `certified_models.pkl`, and a summary in the record.
An evaluation with AA on also gets `aa_per_cycle.jsonl`
(`p515_s43_aa_run._build_aa_per_cycle_sidecar`) and an action summary. A
post-certification exception is recorded (status 'error', traceback) and the
child exits 2 after writing its record; the evaluation itself stands.

================================================================================
FROZEN CAMPAIGN SPEC
================================================================================
`freeze_campaign_spec(...)` writes `<campaign_root>/campaign_spec_<id>_<sha8>.json`
(write-once; `<sha8>` = first 8 hex of the file's own sha256): campaign id,
candidates (label, canonical form, key), configuration (case file path,
sha256 and last commit; arm label; overrides), cap, concurrency, required
consecutive cycles, thread caps, interpreter, solver path, git HEAD, harness
sha256. Its sha256 is passed to every child, verified by the child against
the file, and written into every evaluation record.
"""

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

HARNESS_PATH = os.path.abspath(__file__)
PYTHON = sys.executable
CAMPAIGN_LOCK_PATH = os.path.join(REPO, '.p515_s44_campaign.lock')
LEGACY_RUN_LOCK_PATH = os.path.join(REPO, '.p515_g_gate.lock')
RESULTS_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44')
CASE_FILE_REL = os.path.join('data', 'SRP1', 'SRP1_params.json')
CASE_FILE = os.path.join(REPO, CASE_FILE_REL)
RECORD_SCHEMA = 'p515_s44_evaluation_record_v2'  # v2 (Addendum 25 item 2): per-evaluation config + post-certification
SPEC_SCHEMA = 'p515_s44_campaign_spec_v2'

THREAD_CAP_ENV = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}
SUPPORTED_OVERRIDE_KEYS = frozenset({'anderson_acceleration'})
# Addendum 25 item 2: the ONLY configuration a campaign spec may override is the AA flag and its
# reject-policy sub-option (memory / regularization stay at the frozen 5 / 1e-10).
SUPPORTED_AA_OVERRIDE_SUBKEYS = frozenset({'enabled', 'reject_policy'})
FROZEN_AA_MEMORY = 5
FROZEN_AA_REGULARIZATION = 1e-10
POST_CERTIFICATION_KEYS = frozenset({'persist_certified_models', 'hull_polish', 'reference'})
# Addendum 27 (W14): 'investment_year' is the SINGLE cohort year this evaluation's candidate is
# placed at; omitted => INVESTMENT_YEAR (2025), so every spec frozen before W14 is unchanged.
EVALUATION_OPTION_KEYS = frozenset({'overrides', 'post_certification', 'investment_year', 'model_variant',
                                    'flex_price_multiplier', 'interface_deviation_premium',
                                    'release_solution_bookkeeping', 'certification_continuation'})
# P5.15 Addendum 39 (W47): a DERIVED INSTANCE (see the module docstring) and row 18's premium. The identity keys
# of a derived instance enter the eval key; the other keys are provenance. Absent -> nothing changes.
DERIVED_INSTANCE_KEYS = frozenset({'instance_label', 'case_path', 'case_sha256', 'scenario_checksum',
                                   'source_case_path', 'source_case_sha256', 'changes_vs_source'})
DERIVED_INSTANCE_IDENTITY_KEYS = ('instance_label', 'case_sha256', 'scenario_checksum')
DERIVED_INSTANCE_DATA_DIR_REL = os.path.join('data', 'SRP1')  # the scale harness's DATA_DIR (checked in the child)
INTERFACE_DEVIATION_PREMIUM_KEYS = frozenset({'alpha', 'floor'})
MULTISCENARIO_TERMINAL_FILE = 'multiscenario_terminal.json'
MULTISCENARIO_IDENTITY_REL_TOL = 1e-9   # declared before any run: relative tolerance of the zero-solve identities
# P5.15 Addendum 46 ruling 7 (W84): the convergence-depth tail declaration and the per-evaluation floor-status capture
# (see the module docstring). Absent declaration -> production's default (tail OFF); the files are written for EVERY
# evaluation.
CONVERGENCE_DEPTH_TAIL_KEYS = frozenset({'enabled', 'compl_inf_tol'})
NETWORK_IPOPT_SOLVE_RECORDS_FILE = 'network_ipopt_solve_records.jsonl'
CONVERGENCE_DEPTH_TAIL_STATE_FILE = 'convergence_depth_tail_state.json'
# W85 (Planner ruling Q3 on W84): the same capture appended per round, fsync'd, so a failed / killed child keeps it
NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE = 'network_ipopt_solve_records_append.jsonl'
CONVERGENCE_DEPTH_APPEND_EVENTS_FILE = 'convergence_depth_append_events.jsonl'
# P5.15 Addendum 34 (W33): a uniform multiplier m on the DSO flexibility-price profile `cost_flex` (see
# `validate_flex_price_multiplier` / `apply_flex_price_multiplier`). Absent or 1.0 = today's price; it enters the
# eval key ONLY when present and != 1.0, so every key frozen before W33 is byte-identical.
FLEX_PRICE_LABEL = 'MODEL VARIANT — flexibility price × m'
FLEX_PRICE_VARS = ('flex_p_down', 'flex_q_down')  # the variables `mch.flexibility_cost` prices at cost_flex
# Read-back tolerance: coef(m) vs m * coef(1) differ only by the rounding order of (m * c) * baseMVA against
# m * (c * baseMVA) -- at most a few ulp; 1e-15 relative (~4.5 ulp) is stated before any run.
FLEX_PRICE_READBACK_REL_TOL = 1e-15
# Addenda 28-29 (W20): a MODEL VARIANT of the shared-ESS ageing law (see `validate_model_variant`). Exactly these
# four keys; `available_energy_soh_point` values mirror shared_energy_storage_data.AVAILABLE_ENERGY_SOH_POINTS
# (re-checked against production in the child, so the parent stays free of model imports).
MODEL_VARIANT_KEYS = frozenset({'eol_retention_r', 'calendar_retention_per_year', 'available_energy_soh_point',
                                'ageing_enabled'})
MODEL_VARIANT_SOH_POINTS = ('end', 'mid')
MODEL_VARIANT_LABEL = 'MODEL VARIANT \u2014 not the baseline'
MODEL_VARIANT_READBACK_RTOL = 1e-12
# P5.15 Addendum 30 (W21): a spec may DECLARE the shared-ESS ageing parameters its evaluations run with
# (`configuration.ess_ageing_baseline`, `validate_ess_ageing_baseline`): the exact dict production's loader yields
# from the ESS parameters file (`ess_ageing_parameters_as_loaded`), keyed as in the file. Declared -> the child
# refuses unless the loaded parameters equal it (types included) and the file hashes to the pinned sha256, and the
# declaration enters the eval key; undeclared -> every key and format is exactly as before.
ESS_PARAMS_FILE_REL = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
ESS_AGEING_KEYS = frozenset({'calendar_life_years', 'cycle_life_nominal', 'depth_of_discharge_nominal',
                             'minimum_soh', 'calendar_retention_per_year', 'calibration'})
ESS_AGEING_CALIBRATION_KEYS = frozenset({'status', 'cycles_n', 'reference_dod_d', 'eol_retention_r'})
ESS_AGEING_CALIBRATION_STATUSES = ('ACTIVE', 'DECLARED_NOT_CONSUMED')
ACTIVE_NODES = (5, 7, 9)
INVESTMENT_YEAR = 2025
BAR_WINDOW = 10  # STEP4 2.5: "its bar (max objective step over the last 10 cycles)"
CHANNELS = ('v', 'pf', 'ess')

POLL_S = 5.0
HEARTBEAT_EVERY_S = 60.0

# The per-cycle fields read off production's trajectory rows (`run_admm_arm` cycle_trajectory). This is the tuple
# that was `PER_CYCLE_RECORD_FIELDS` before W64, unchanged; `RECORD_TRAJECTORY_FIELDS` (below) is built from it, so
# the rule-eleven trajectory checklist (every field must appear in production's source) is exactly as before.
PER_CYCLE_TRAJECTORY_FIELDS = (
    'cycle', 'local_solves_ok', 'recourse', 'gross_operational_cost', 'terminal_salvage_value',
    'objective_change_abs', 'objective_tolerance', 'objective_change_ratio',
    'cycle_convergence', 'consecutive_converged_cycles', 'boyd_all_pass', 'boyd_stop',
    'boyd_v_primal_ratio', 'boyd_v_dual_ratio', 'boyd_v_channel_pass',
    'boyd_pf_primal_ratio', 'boyd_pf_dual_ratio', 'boyd_pf_channel_pass',
    'boyd_ess_primal_ratio', 'boyd_ess_dual_ratio', 'boyd_ess_channel_pass',
    'rho_v_after', 'rho_pf_after', 'rho_ess_after', 'rho_v_action', 'rho_pf_action', 'rho_ess_action',
    'rho_freeze_active', 'efc_per_day_max',
)
# P5.15 Addendum 40 ruling 1 (W64): the per-cycle RESPONSE fields, captured by the harness itself every cycle
# (`per_cycle_response_record`, zero solves, read off the cycle's own models and SolverResults) for derived-instance
# / premium evaluations only, appended to `PER_CYCLE_RESPONSE_FILE` as each cycle completes (so a run that fails at
# hour 9 of 12 leaves every completed cycle on disk) and merged into `per_cycle_record.jsonl` by cycle. Every other
# evaluation writes `PER_CYCLE_TRAJECTORY_FIELDS` only, i.e. exactly its pre-W64 per-cycle record.
PER_CYCLE_RESPONSE_FIELDS = (
    'response_cycle', 'response_captured', 'response_capture_error',
    'E_abs_d_p_mwh_weighted', 'E_abs_d_p_mwh_unweighted',
    'sum_omega_d2_p_mw2h_weighted', 'sum_omega_d2_p_mw2h_unweighted',
    'market_part_mw2h_weighted', 'operation_part_mw2h_weighted',
    'market_part_mw2h_unweighted', 'operation_part_mw2h_unweighted',
    'row18_charge_weighted', 'covariance_dso_weighted',
    'curtailed_res_dso_mwh_weighted', 'curtailed_res_tso_mwh_weighted',
    'max_abs_d_p_mw', 'n_non_optimal_block_terminations', 'non_optimal_blocks',
    'cycle_wall_s', 'rss_bytes', 'ru_maxrss_bytes', 'response_capture_s',
)
PER_CYCLE_RECORD_FIELDS = PER_CYCLE_TRAJECTORY_FIELDS + PER_CYCLE_RESPONSE_FIELDS


# ==============================================================================
#  small utilities
# ==============================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _json_default(obj):
    """The `default=` hook of every artifact writer in this harness (P5.15 Addendum 44, W74). A numpy boolean is
    written as a JSON boolean; everything else exactly as before (`str`). Before W74 the hook was `str` alone, so a
    numpy boolean -- which is NOT a subclass of `bool`, unlike numpy.float64 of `float` -- was written as the STRING
    "True"/"False": `hull_polish_full.gate.pass` on five of the six alpha-row cells (alpha > 0 makes the gate's
    `relative` a numpy float, so its comparison is a numpy bool), which a strict `is True` reader misreads as a failure
    and a truthiness reader misreads "False" as a pass. Detected by type identity so the parent needs no numpy import.
    Not used by any hashing / key / spec-freeze serialization (those keep `default=str` or none, unchanged)."""
    if type(obj).__module__ == 'numpy' and type(obj).__name__ in ('bool', 'bool_'):
        return bool(obj)
    return str(obj)


def _atomic_write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(obj, handle, indent=1, default=_json_default)
    os.replace(tmp, path)


def _write_once_json(path, obj):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
    with open(path, 'w') as handle:
        json.dump(obj, handle, indent=1, default=_json_default)


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def _sanitize_id(text):
    out = ''.join(c if (c.isalnum() or c == '_') else '_' for c in str(text).lower())
    if not out:
        raise ValueError(f'empty id after sanitization: {text!r}')
    return out


# ==============================================================================
#  candidates: canonical form and key
# ==============================================================================
def canonical_candidate(candidate, active_nodes=ACTIVE_NODES, investment_year=INVESTMENT_YEAR):
    """Canonical per-node (s_mva, e_mwh) investment map at one investment year.

    `candidate` is {node_id: (s_mva, e_mwh)} (node ids int or str). Every active
    node must be present (no implicit zeros -- `run_admm_arm`'s own docstring:
    "pass a full dict covering every active node to avoid ambiguity"); values
    are floats >= 0; s == 0 <=> e == 0 (production raises otherwise,
    `_configure_esso_cohort_state`). Canonical form: {'investment_year': y,
    'nodes': {'5': [s, e], ...}} with float values and string node keys in
    ascending node order -- two candidates with the same canonical form are
    the same evaluation (STEP4 5.4)."""
    if not isinstance(candidate, dict):
        raise TypeError(f'candidate must be a dict {{node: (s, e)}}, got {type(candidate).__name__}')
    by_node = {}
    for key, value in candidate.items():
        node = int(key)
        if node in by_node:
            raise ValueError(f'node {node} given twice')
        s_val, e_val = value
        s_val, e_val = float(s_val), float(e_val)
        for name, v in (('s', s_val), ('e', e_val)):
            if not (v == v) or v in (float('inf'), float('-inf')):
                raise ValueError(f'node {node}: non-finite {name}={v}')
            if v < 0.0:
                raise ValueError(f'node {node}: negative {name}={v}')
        if (s_val == 0.0) != (e_val == 0.0):
            raise ValueError(f'node {node}: s == 0 <=> e == 0 violated (s={s_val}, e={e_val})')
        by_node[node] = (s_val + 0.0, e_val + 0.0)  # +0.0 folds -0.0 into 0.0
    missing = sorted(set(active_nodes) - set(by_node))
    extra = sorted(set(by_node) - set(active_nodes))
    if missing or extra:
        raise ValueError(f'candidate must cover exactly the active nodes {list(active_nodes)}: '
                         f'missing={missing} extra={extra}')
    return {'investment_year': int(investment_year),
            'nodes': {str(n): [by_node[n][0], by_node[n][1]] for n in sorted(by_node)}}


def candidate_key(canonical):
    """sha256 (hex) of the canonical form's compact, key-sorted JSON (floats by repr)."""
    text = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def investment_map_from_canonical(canonical):
    return {int(n): (v[0], v[1]) for n, v in canonical['nodes'].items()}


def investment_year_from_canonical(canonical):
    """The SINGLE cohort year of a canonical candidate (Addendum 27, W14)."""
    return int(canonical['investment_year'])


def instance_investment_years():
    """THIS instance's investment years, read from the shared-ESS data of the
    baseline planning problem (`p56a_oracle.load_baseline`, cached, zero solves)
    -- never a literal. CHILD-SIDE ONLY: the parent imports no model code on the
    evaluation path, so this import is local to the function."""
    import p56a_oracle as O  # local: model code, child side only
    return [int(y) for y in O.load_baseline()['planning'].shared_ess_data.years]


def eval_ids(campaign_id, key):
    stub = f'p515s44_{_sanitize_id(campaign_id)}_{key[:16]}'
    return {'run': f'{stub}_run', 'precheck': f'{stub}_precheck'}


# ==============================================================================
#  per-evaluation configuration (Addendum 25 item 2)
# ==============================================================================
def validate_overrides(overrides):
    """The spec's configuration overrides: ONLY `anderson_acceleration` with
    sub-keys `enabled` (bool) and `reject_policy` (one of
    `admm_anderson_acceleration.REJECT_POLICIES`). Returns a normalized copy."""
    overrides = dict(overrides or {})
    unsupported = sorted(set(overrides) - SUPPORTED_OVERRIDE_KEYS)
    if unsupported:
        raise ValueError(f'unsupported configuration overrides {unsupported}; supported: '
                         f'{sorted(SUPPORTED_OVERRIDE_KEYS)}')
    out = {}
    if 'anderson_acceleration' in overrides:
        import admm_anderson_acceleration as AA  # numpy only; no model code
        aa = overrides['anderson_acceleration']
        if not isinstance(aa, dict):
            raise ValueError('anderson_acceleration override must be a dict')
        bad = sorted(set(aa) - SUPPORTED_AA_OVERRIDE_SUBKEYS)
        if bad:
            raise ValueError(f'unsupported anderson_acceleration override sub-keys {bad}; supported: '
                             f'{sorted(SUPPORTED_AA_OVERRIDE_SUBKEYS)}')
        if 'enabled' in aa and not isinstance(aa['enabled'], bool):
            raise ValueError('anderson_acceleration.enabled must be a bool')
        if 'reject_policy' in aa and aa['reject_policy'] not in AA.REJECT_POLICIES:
            raise ValueError(f"anderson_acceleration.reject_policy must be one of {AA.REJECT_POLICIES}")
        out['anderson_acceleration'] = dict(aa)
    return out


CASE_FILE_AA_KEYS = frozenset({'enabled', 'memory', 'regularization', 'reject_policy'})


def validate_case_file_anderson_acceleration(declared):
    """P5.15 Addendum 27 item 1: a campaign spec may declare
    `configuration.case_file_anderson_acceleration` -- the EXACT
    `anderson_acceleration` dict the case file must load to (checked by the
    child's configuration hook). None = not declared (the pre-Addendum-27
    meaning: the case file must carry AA off). Returns a normalized copy."""
    if declared is None:
        return None
    import admm_anderson_acceleration as AA  # numpy only; no model code
    if not isinstance(declared, dict):
        raise ValueError('case_file_anderson_acceleration must be a dict')
    bad = sorted(set(declared) - CASE_FILE_AA_KEYS)
    missing = sorted({'enabled', 'memory', 'regularization'} - set(declared))
    if bad or missing:
        raise ValueError(f'case_file_anderson_acceleration: unsupported keys {bad} / missing keys {missing}; '
                         f'keys: {sorted(CASE_FILE_AA_KEYS)} (reject_policy optional)')
    if not isinstance(declared['enabled'], bool):
        raise ValueError('case_file_anderson_acceleration.enabled must be a bool')
    if declared['memory'] != FROZEN_AA_MEMORY or declared['regularization'] != FROZEN_AA_REGULARIZATION:
        raise ValueError(f'case_file_anderson_acceleration memory/regularization must be the frozen '
                         f'{FROZEN_AA_MEMORY}/{FROZEN_AA_REGULARIZATION}: {declared}')
    if 'reject_policy' in declared and declared['reject_policy'] not in AA.REJECT_POLICIES:
        raise ValueError(f'case_file_anderson_acceleration.reject_policy must be one of {AA.REJECT_POLICIES}')
    return dict(declared)


def effective_anderson_acceleration(case_file_aa, overrides):
    """The AA settings an evaluation runs with when the spec declares the case
    file's AA dict: the declaration with the evaluation's AA override merged on
    top (as `_config_hook_factory` applies it). None when not declared."""
    if case_file_aa is None:
        return None
    merged = dict(case_file_aa)
    merged.update((overrides or {}).get('anderson_acceleration') or {})
    return merged


def validate_model_variant(model_variant):
    """Addenda 28-29 (W20): a MODEL VARIANT of the shared-ESS ageing law. None = no variant
    (the baseline model). Otherwise a dict of EXACTLY `MODEL_VARIANT_KEYS`:
      eol_retention_r              float in (0, 1): the calibration's end-of-life retention R
                                   (k = N * D / (-ln R); N, D stay the case file's);
      calendar_retention_per_year  float in (0, 1]: phi_cal;
      available_energy_soh_point   'end' | 'mid' (shared_energy_storage_data._esso_ageing_model_settings);
      ageing_enabled               bool (False: SoH == 1 everywhere).
    Returns a normalized copy (floats as float, JSON-stable). No model import (parent side)."""
    if model_variant is None:
        return None
    if not isinstance(model_variant, dict):
        raise ValueError('model_variant must be a dict')
    missing = sorted(MODEL_VARIANT_KEYS - set(model_variant))
    extra = sorted(set(model_variant) - MODEL_VARIANT_KEYS)
    if missing or extra:
        raise ValueError(f'model_variant must carry exactly {sorted(MODEL_VARIANT_KEYS)}: missing={missing} '
                         f'extra={extra}')
    out = {}
    for name in ('eol_retention_r', 'calendar_retention_per_year'):
        value = model_variant[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            raise ValueError(f'model_variant.{name} must be a finite number, got {value!r}')
        out[name] = float(value)
    if not 0.0 < out['eol_retention_r'] < 1.0:
        raise ValueError(f"model_variant.eol_retention_r must lie in (0, 1), got {out['eol_retention_r']}")
    if not 0.0 < out['calendar_retention_per_year'] <= 1.0:
        raise ValueError(f"model_variant.calendar_retention_per_year must lie in (0, 1], got "
                         f"{out['calendar_retention_per_year']}")
    if model_variant['available_energy_soh_point'] not in MODEL_VARIANT_SOH_POINTS:
        raise ValueError(f'model_variant.available_energy_soh_point must be one of {MODEL_VARIANT_SOH_POINTS}, '
                         f"got {model_variant['available_energy_soh_point']!r}")
    out['available_energy_soh_point'] = model_variant['available_energy_soh_point']
    if not isinstance(model_variant['ageing_enabled'], bool):
        raise ValueError(f"model_variant.ageing_enabled must be a bool, got {model_variant['ageing_enabled']!r}")
    out['ageing_enabled'] = model_variant['ageing_enabled']
    return out


def validate_flex_price_multiplier(value):
    """P5.15 Addendum 34 (W33): the uniform multiplier m on the `cost_flex` profile bound to every DSO block.
    None = not given (today's price). Otherwise a finite number > 0 (bool refused), returned as float.
    No model import (parent side)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'flex_price_multiplier must be a positive finite number, got {value!r}')
    out = float(value)
    if out != out or out in (float('inf'), float('-inf')) or out <= 0.0:
        raise ValueError(f'flex_price_multiplier must be a positive finite number, got {value!r}')
    return out


def flex_price_multiplier_in_key(value):
    """The multiplier as it enters the eval key: None when absent OR exactly 1.0 (today's price -- the same
    evaluation as the baseline), else the validated float."""
    out = validate_flex_price_multiplier(value)
    return None if (out is None or out == 1.0) else out


_HEX64 = frozenset('0123456789abcdef')


def _is_hex64(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def validate_derived_instance(declared):
    """P5.15 Addendum 39 (W47): a spec's declaration of a DERIVED instance (module docstring). None = not declared
    (the canonical SRP1 instance, every pre-W47 meaning and key unchanged). Otherwise a dict of EXACTLY
    `DERIVED_INSTANCE_KEYS`: instance_label (non-empty str, never 'srp1' -- the SRP1 contract is the canonical
    checksum, which `p56a_oracle.install_baseline` enforces for that label), case_path / source_case_path
    (repo-relative str), case_sha256 / source_case_sha256 / scenario_checksum (64 lowercase hex), changes_vs_source
    (non-empty list). Returns a copy. No model import (parent side)."""
    if declared is None:
        return None
    if not isinstance(declared, dict):
        raise ValueError('derived_instance must be a dict')
    missing, extra = sorted(DERIVED_INSTANCE_KEYS - set(declared)), sorted(set(declared) - DERIVED_INSTANCE_KEYS)
    if missing or extra:
        raise ValueError(f'derived_instance must carry exactly {sorted(DERIVED_INSTANCE_KEYS)}: missing={missing} '
                         f'extra={extra}')
    label = declared['instance_label']
    if not isinstance(label, str) or not label.strip() or label == 'srp1':
        raise ValueError(f"derived_instance.instance_label must be a non-empty str other than 'srp1', got {label!r}")
    for name in ('case_path', 'source_case_path'):
        value = declared[name]
        if not isinstance(value, str) or not value or os.path.isabs(value):
            raise ValueError(f'derived_instance.{name} must be a repo-relative path, got {value!r}')
    for name in ('case_sha256', 'source_case_sha256', 'scenario_checksum'):
        if not _is_hex64(declared[name]):
            raise ValueError(f'derived_instance.{name} must be 64 lowercase hex characters, got {declared[name]!r}')
    if not isinstance(declared['changes_vs_source'], list) or not declared['changes_vs_source']:
        raise ValueError('derived_instance.changes_vs_source must be a non-empty list')
    return json.loads(json.dumps(declared))


def derived_instance_identity(derived_instance):
    """The part of a validated derived-instance declaration that enters the eval key."""
    return {k: derived_instance[k] for k in DERIVED_INSTANCE_IDENTITY_KEYS}


def validate_interface_deviation_premium(value):
    """P5.15 Addendum 39 (W47): row 18's premium for one evaluation, {'alpha': a, 'floor': f}. None = not given
    (the case file's own setting -- SRP1: absent, i.e. alpha 0, row 18 inactive). alpha: finite number >= 0 (bool
    refused); floor: None or a finite number (Addendum 38: a floor only if some hour's mean price is non-positive).
    Returns {'alpha': float, 'floor': None | float}. No model import (parent side)."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != INTERFACE_DEVIATION_PREMIUM_KEYS:
        raise ValueError(f'interface_deviation_premium must be a dict with exactly '
                         f'{sorted(INTERFACE_DEVIATION_PREMIUM_KEYS)}, got {value!r}')
    alpha, floor = value['alpha'], value['floor']
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha != alpha \
            or alpha in (float('inf'), float('-inf')) or alpha < 0.0:
        raise ValueError(f'interface_deviation_premium.alpha must be a finite number >= 0, got {alpha!r}')
    if floor is not None and (isinstance(floor, bool) or not isinstance(floor, (int, float)) or floor != floor
                              or floor in (float('inf'), float('-inf'))):
        raise ValueError(f'interface_deviation_premium.floor must be None or a finite number, got {floor!r}')
    return {'alpha': float(alpha), 'floor': None if floor is None else float(floor)}


def validate_convergence_depth_tail(value):
    """P5.15 Addendum 46 ruling 7 (W84): the convergence-depth tail declaration of a campaign spec's
    `configuration`. None = not declared (production's default, `ADMMParameters.convergence_depth_tail`: OFF).
    Otherwise EXACTLY {'enabled': bool, 'compl_inf_tol': positive finite float} -- a bool or an int is refused as the
    tolerance, as production's `_capture_convergence_depth_tail_baseline` refuses a non-float. Returns a new dict.
    No model import (parent side). W86: only an ENABLED declaration enters `evaluation_key`
    (`convergence_depth_tail_in_key`); None and enabled False do not."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != CONVERGENCE_DEPTH_TAIL_KEYS:
        raise ValueError(f'convergence_depth_tail must be a dict with exactly {sorted(CONVERGENCE_DEPTH_TAIL_KEYS)}, '
                         f'got {value!r}')
    enabled, tol = value['enabled'], value['compl_inf_tol']
    if not isinstance(enabled, bool):
        raise ValueError(f'convergence_depth_tail.enabled must be a bool, got {enabled!r}')
    if not isinstance(tol, float) or tol != tol or tol in (float('inf'), float('-inf')) or not tol > 0.0:
        raise ValueError(f'convergence_depth_tail.compl_inf_tol must be a positive finite float, got {tol!r}')
    return {'enabled': enabled, 'compl_inf_tol': tol}


def convergence_depth_tail_in_key(value):
    """W86 (Planner ruling Q1 on W85; the W33 precedent `flex_price_multiplier_in_key`): the tail declaration as it
    enters the eval key -- None when absent OR declared with enabled False (production's default: no tail function
    is called and `compl_inf_tol` is never read, so the computation is the undeclared one), else the validated
    declaration. Validation is applied in both cases, so an invalid declaration is still refused."""
    out = validate_convergence_depth_tail(value)
    return None if (out is None or not out['enabled']) else out


def validate_release_solution_bookkeeping(value):
    """P5.15 Addendum 48 (W90): an entry's `release_solution_bookkeeping` option -- option (b), production's
    `SolverParameters.release_solution_bookkeeping` (network.py `_release_solution_bookkeeping`, P5.15 Addendum 29 W32).
    None = not declared: nothing is applied and the entry / record keep their exact pre-W90 format (production's
    default, False, stays in force). Otherwise a bool, applied in the child by the config hook through
    `p515_s44_scale_measurement.set_release_solution_bookkeeping` (the setter the committed SRP1 (b) bitwise gate,
    P515S49/memory_fix_gate, used; it reads the switch back and raises if it did not take effect).
    It NEVER enters `evaluation_key`: the switch drops only Pyomo's solution copies after the values and suffixes are
    loaded (a memory-only change; the SRP1 gate reproduced the committed C* trajectory bitwise with it on), so an
    evaluation with it on or off is the same candidate x configuration -- Addendum 48 carries the frozen 3 x 3 keys
    over. A campaign that compares the two settings must therefore use two campaign ids (the working-dir ids differ by
    campaign id, `eval_ids`)."""
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f'release_solution_bookkeeping must be a bool, got {value!r}')
    return value


class _ReleaseBookkeepingCallCounter:
    """P5.15 Addendum 48 (W90): counts production's `network._release_solution_bookkeeping` calls during a run -- a
    pass-through wrapper installed on the module attribute (which `network._run_smopf` looks up at call time); restored
    on exit; the count goes to `holder['release_solution_bookkeeping_calls']`. Installed only for an entry that declares
    the option."""

    def __init__(self, holder):
        self.holder = holder
        self.calls = 0
        self._module = None
        self._original = None

    def __enter__(self):
        import network as network_module
        self._module = network_module
        self._original = network_module._release_solution_bookkeeping
        original = self._original

        def counted(model, result):
            self.calls += 1
            return original(model, result)

        network_module._release_solution_bookkeeping = counted
        return self

    def __exit__(self, *exc):
        self._module._release_solution_bookkeeping = self._original
        self.holder['release_solution_bookkeeping_calls'] = self.calls
        return False


def _ess_number(value, name, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value \
            or value in (float('inf'), float('-inf')):
        raise ValueError(f'ess_ageing_baseline.{name} must be a finite number, got {value!r}')
    return value


def validate_ess_ageing_baseline(declared):
    """P5.15 Addendum 30 (W21): a spec's declaration of the shared-ESS ageing parameters, i.e. the EXACT dict
    `ess_ageing_parameters_as_loaded` returns for the ESS parameters file the evaluations run with. None = not
    declared (every pre-W21 meaning and key unchanged). Keys as in the file's `ageing` block: EXACTLY
    `ESS_AGEING_KEYS`, `calibration` EXACTLY `ESS_AGEING_CALIBRATION_KEYS`. Numbers keep their JSON type (the
    loader preserves int vs float, and the eval key is computed on this dict), so the declaration must carry
    them as the loader yields them; the child compares canonical JSON, types included. Returns a copy.
    No model import (parent side)."""
    if declared is None:
        return None
    if not isinstance(declared, dict):
        raise ValueError('ess_ageing_baseline must be a dict')
    missing, extra = sorted(ESS_AGEING_KEYS - set(declared)), sorted(set(declared) - ESS_AGEING_KEYS)
    if missing or extra:
        raise ValueError(f'ess_ageing_baseline must carry exactly {sorted(ESS_AGEING_KEYS)}: missing={missing} '
                         f'extra={extra}')
    cal = declared['calibration']
    if not isinstance(cal, dict):
        raise ValueError('ess_ageing_baseline.calibration must be a dict')
    missing, extra = (sorted(ESS_AGEING_CALIBRATION_KEYS - set(cal)), sorted(set(cal) - ESS_AGEING_CALIBRATION_KEYS))
    if missing or extra:
        raise ValueError(f'ess_ageing_baseline.calibration must carry exactly {sorted(ESS_AGEING_CALIBRATION_KEYS)}: '
                         f'missing={missing} extra={extra}')
    if cal['status'] not in ESS_AGEING_CALIBRATION_STATUSES:
        raise ValueError(f"ess_ageing_baseline.calibration.status must be one of {ESS_AGEING_CALIBRATION_STATUSES}, "
                         f"got {cal['status']!r}")
    out = {name: _ess_number(declared[name], name) for name in sorted(ESS_AGEING_KEYS - {'calibration'})}
    active = cal['status'] == 'ACTIVE'
    out['calibration'] = {'status': cal['status']}
    for name in ('cycles_n', 'reference_dod_d', 'eol_retention_r'):
        out['calibration'][name] = _ess_number(cal[name], f'calibration.{name}', allow_none=not active)
    if not 0.0 <= out['minimum_soh'] < 1.0:
        raise ValueError(f"ess_ageing_baseline.minimum_soh must lie in [0, 1), got {out['minimum_soh']}")
    if not 0.0 < out['calendar_retention_per_year'] <= 1.0:
        raise ValueError(f"ess_ageing_baseline.calendar_retention_per_year must lie in (0, 1], got "
                         f"{out['calendar_retention_per_year']}")
    r = out['calibration']['eol_retention_r']
    if r is not None and not 0.0 < r < 1.0:
        raise ValueError(f'ess_ageing_baseline.calibration.eol_retention_r must lie in (0, 1), got {r}')
    return out


def ess_ageing_canonical_text(ageing_dict):
    """Canonical JSON of an ageing dict (types preserved: 10000 and 10000.0 differ) -- the comparison form."""
    return json.dumps(ageing_dict, sort_keys=True, separators=(',', ':'))


def ess_ageing_parameters_as_loaded(ageing):
    """The dict of a LOADED `shared_energy_storage_parameters.EnergyStorageAgeingParameters` object, keyed as in
    the ESS parameters file's `ageing` block (values exactly as the production loader stored them)."""
    cal = ageing.calibration
    return {'calendar_life_years': ageing.t_cal, 'cycle_life_nominal': ageing.cl_nom,
            'depth_of_discharge_nominal': ageing.dod_nom, 'minimum_soh': ageing.soh_min,
            'calendar_retention_per_year': ageing.calendar_retention_per_year,
            'calibration': {'status': cal.status, 'cycles_n': cal.cycles_n, 'reference_dod_d': cal.reference_dod_d,
                            'eol_retention_r': cal.eol_retention_r}}


def load_ess_ageing_parameters(path):
    """Load an ESS parameters file with PRODUCTION's loader (`SharedEnergyStorageParameters.
    read_parameters_from_file`; parameters only, no model is built) and return its ageing dict."""
    from shared_energy_storage_parameters import SharedEnergyStorageParameters  # local: loader only
    params = SharedEnergyStorageParameters()
    params.read_parameters_from_file(path)
    return ess_ageing_parameters_as_loaded(params.ageing)


def validate_certification_continuation(value):
    """P5.15 Addendum 51 (W98): an entry's `certification_continuation` option -- the post-certification continuation
    (certification rule disabled, regime held after cycle N, early stop), validated by the hooks module that
    implements it (`p515_s53_w98_continuation_hooks.validate_certification_continuation`; stdlib-only at import, so the
    parent stays free of model code). None = not declared: nothing is installed and every key / entry / record keeps its
    exact pre-W98 format."""
    if value is None:
        return None
    import p515_s53_w98_continuation_hooks as W98C
    return W98C.validate_certification_continuation(value)


def evaluation_key(candidate_key_hex, overrides, case_file_aa=None, model_variant=None, ess_ageing_baseline=None,
                   flex_price_multiplier=None, derived_instance=None, interface_deviation_premium=None,
                   convergence_depth_tail=None, certification_continuation=None):
    """Identity of one EVALUATION (candidate x configuration). The case-file
    configuration (no overrides) keeps the candidate key itself, so a D
    evaluation's directory name is `<candidate key16>_<label>` as in s44_gate;
    any override gives sha256 of {candidate_key, overrides}.
    Addendum 27 item 1: when the spec declares `case_file_anderson_acceleration`
    (`case_file_aa`), the key is sha256 of {candidate_key,
    effective_anderson_acceleration, overrides}, so a case-file-AA evaluation
    never collides with the D (or an AA-override) evaluation of the same
    candidate. Specs without the declaration keep the formula above exactly.
    Addenda 28-29 (W20): with a `model_variant` (validated), the key is sha256 of
    the SAME payload plus 'model_variant' -- never the bare candidate key -- so a
    variant evaluation can never collide with the baseline evaluation of the same
    candidate. `model_variant=None` returns exactly what the formulas above return.
    P5.15 Addendum 30 (W21): with a declared `ess_ageing_baseline` (validated), the key is sha256 of the SAME
    payload the rules above would hash (with 'effective_anderson_acceleration' when `case_file_aa` is declared,
    'model_variant' when given) plus 'ess_ageing_baseline' -- never the bare candidate key -- so evaluations of
    one candidate under two ageing baselines never share a key. `ess_ageing_baseline=None` returns exactly what
    the formulas above return.
    P5.15 Addendum 34 (W33): with a `flex_price_multiplier` m that is present AND != 1.0
    (`flex_price_multiplier_in_key`), the key is sha256 of the SAME payload the rules above would hash plus
    'flex_price_multiplier': m -- never the bare candidate key (a case-file D evaluation under m != 1 hashes
    {candidate_key, overrides, flex_price_multiplier}). Absent or 1.0 returns exactly what the formulas above
    return, so every key frozen before W33 is byte-identical.
    P5.15 Addendum 39 (W47): with a declared `derived_instance` and/or an `interface_deviation_premium` (validated),
    the key is sha256 of {candidate_key, overrides} plus every declared item the rules above would hash
    ('effective_anderson_acceleration', 'model_variant', 'ess_ageing_baseline', 'flex_price_multiplier' -- the last
    only when != 1.0) plus 'derived_instance' (its identity keys: label, case sha256, scenario checksum) and/or
    'interface_deviation_premium' -- never the bare candidate key, so one candidate on two instances, or under two
    premiums, never shares a key. Both None returns exactly what the formulas above return.
    P5.15 Addendum 46 ruling 7 (W85, Planner ruling Q1 on W84; W86, Planner ruling Q1 on W85): with a DECLARED AND
    ENABLED `convergence_depth_tail` (`convergence_depth_tail_in_key`; a declared-OFF tail changes nothing in
    production and returns exactly what an undeclared one returns -- the W33 m == 1.0 rule), the key is sha256 of
    {candidate_key, overrides} plus every declared item the rules above would hash ('effective_anderson_acceleration',
    'model_variant', 'ess_ageing_baseline', 'flex_price_multiplier' -- only when != 1.0 --, 'derived_instance' identity,
    'interface_deviation_premium') plus 'convergence_depth_tail' (the validated declaration) -- never the bare candidate
    key, so a tail-enabled evaluation can never share a key (hence a cache hit) with a pre-tail one.
    `convergence_depth_tail=None` or enabled False returns exactly what the formulas above return.
    P5.15 Addendum 51 (W98): with a declared `certification_continuation` (validated), the key is sha256 of
    {'base_evaluation_key': <the key every rule above gives for the same arguments>, 'certification_continuation': <the
    validated declaration>} -- never the base key, so a continuation evaluation can never share a key (hence a cache
    hit, an eval dir or a working dir) with the certified evaluation it replays. The cap and the certificate length are
    spec-level and enter no key; the continuation declaration is what separates the two. `certification_continuation=
    None` returns exactly what the formulas above return, so every pre-W98 key is byte-identical."""
    if certification_continuation is not None:
        base = evaluation_key(candidate_key_hex, overrides, case_file_aa=case_file_aa, model_variant=model_variant,
                              ess_ageing_baseline=ess_ageing_baseline, flex_price_multiplier=flex_price_multiplier,
                              derived_instance=derived_instance, interface_deviation_premium=interface_deviation_premium,
                              convergence_depth_tail=convergence_depth_tail)
        payload = {'base_evaluation_key': base,
                   'certification_continuation': validate_certification_continuation(certification_continuation)}
        text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()
    model_variant = validate_model_variant(model_variant)
    ess_ageing_baseline = validate_ess_ageing_baseline(ess_ageing_baseline)
    flex_m = flex_price_multiplier_in_key(flex_price_multiplier)
    derived_instance = validate_derived_instance(derived_instance)
    premium = validate_interface_deviation_premium(interface_deviation_premium)
    tail = convergence_depth_tail_in_key(convergence_depth_tail)
    if tail is not None:  # W86: only when declared ENABLED, so every undeclared / declared-off key is byte-identical
        payload = {'candidate_key': candidate_key_hex, 'overrides': overrides or {}}
        if case_file_aa is not None:
            payload['effective_anderson_acceleration'] = effective_anderson_acceleration(case_file_aa, overrides)
        if model_variant is not None:
            payload['model_variant'] = model_variant
        if ess_ageing_baseline is not None:
            payload['ess_ageing_baseline'] = ess_ageing_baseline
        if flex_m is not None:
            payload['flex_price_multiplier'] = flex_m
        if derived_instance is not None:
            payload['derived_instance'] = derived_instance_identity(derived_instance)
        if premium is not None:
            payload['interface_deviation_premium'] = premium
        payload['convergence_depth_tail'] = tail
        text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()
    if derived_instance is not None or premium is not None:
        payload = {'candidate_key': candidate_key_hex, 'overrides': overrides or {}}
        if case_file_aa is not None:
            payload['effective_anderson_acceleration'] = effective_anderson_acceleration(case_file_aa, overrides)
        if model_variant is not None:
            payload['model_variant'] = model_variant
        if ess_ageing_baseline is not None:
            payload['ess_ageing_baseline'] = ess_ageing_baseline
        if flex_m is not None:
            payload['flex_price_multiplier'] = flex_m
        if derived_instance is not None:
            payload['derived_instance'] = derived_instance_identity(derived_instance)
        if premium is not None:
            payload['interface_deviation_premium'] = premium
        text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()
    if ess_ageing_baseline is not None:
        payload = {'candidate_key': candidate_key_hex, 'overrides': overrides or {},
                   'ess_ageing_baseline': ess_ageing_baseline}
        if case_file_aa is not None:
            payload['effective_anderson_acceleration'] = effective_anderson_acceleration(case_file_aa, overrides)
        if model_variant is not None:
            payload['model_variant'] = model_variant
        if flex_m is not None:
            payload['flex_price_multiplier'] = flex_m
        text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()
    if case_file_aa is not None:
        payload = {'candidate_key': candidate_key_hex,
                   'effective_anderson_acceleration': effective_anderson_acceleration(case_file_aa, overrides),
                   'overrides': overrides or {}}
        if model_variant is not None:
            payload['model_variant'] = model_variant
        if flex_m is not None:
            payload['flex_price_multiplier'] = flex_m
        text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()
    if model_variant is not None:
        payload = {'candidate_key': candidate_key_hex, 'overrides': overrides or {}, 'model_variant': model_variant}
        if flex_m is not None:
            payload['flex_price_multiplier'] = flex_m
        text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()
    if flex_m is not None:
        text = json.dumps({'candidate_key': candidate_key_hex, 'overrides': overrides or {},
                           'flex_price_multiplier': flex_m}, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(text.encode()).hexdigest()
    if not overrides:
        return candidate_key_hex
    text = json.dumps({'candidate_key': candidate_key_hex, 'overrides': overrides}, sort_keys=True,
                      separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def resolve_post_certification(request, candidate_key_hex):
    """Validate a per-evaluation post-certification request and resolve its
    reference evaluation (hash-recorded) at spec-freeze time. Returns None if
    nothing is requested. The reference must be a CERTIFIED evaluation of the
    SAME candidate under the D configuration: for a reference record without a
    `case_file_anderson_acceleration` declaration, the case-file configuration
    with no overrides (unchanged); for a declared one (Addendum 27), an
    effective AA dict (declaration + overrides) with AA off -- so a case-file-AA
    evaluation never passes as a D reference. `reference` is optional: the
    persist / hull-polish items run without one (gates (b)/(c) are then None)."""
    if not request:
        return None
    if not isinstance(request, dict):
        raise ValueError('post_certification must be a dict')
    bad = sorted(set(request) - POST_CERTIFICATION_KEYS)
    if bad:
        raise ValueError(f'unsupported post_certification keys {bad}; supported: {sorted(POST_CERTIFICATION_KEYS)}')
    persist = bool(request.get('persist_certified_models', False))
    polish = bool(request.get('hull_polish', False))
    for name in ('persist_certified_models', 'hull_polish'):
        if name in request and not isinstance(request[name], bool):
            raise ValueError(f'post_certification.{name} must be a bool')
    ref = request.get('reference')
    resolved_ref = None
    if ref is not None:
        if not isinstance(ref, dict) or set(ref) != {'eval_dir'}:
            raise ValueError("post_certification.reference must be {'eval_dir': <repo-relative eval dir>}")
        ref_dir = os.path.join(REPO, ref['eval_dir'])
        rec_path = os.path.join(ref_dir, 'evaluation_record.json')
        cl_path = os.path.join(ref_dir, 'component_levels_terminal.json')
        for p in (rec_path, cl_path):
            if not os.path.isfile(p):
                raise ValueError(f'reference evaluation file missing: {p}')
        with open(rec_path) as handle:
            ref_rec = json.load(handle)
        with open(cl_path) as handle:
            ref_cl = json.load(handle)
        ref_overrides = (ref_rec.get('evaluation_overrides_effective')
                         if 'evaluation_overrides_effective' in ref_rec
                         else (ref_rec.get('configuration') or {}).get('overrides'))
        # Addendum 27 (W5): D-ness is read from the reference's EFFECTIVE AA configuration. A reference
        # whose spec declared `case_file_anderson_acceleration` ran with (declaration + overrides); it is D
        # only if that effective AA is off. Records without the declaration keep the pre-Addendum-27
        # reading exactly (no overrides <=> D), since their case file had to carry AA off.
        ref_case_file_aa = (ref_rec.get('configuration') or {}).get('case_file_anderson_acceleration')
        ref_effective_aa = (effective_anderson_acceleration(ref_case_file_aa, ref_overrides)
                            if ref_case_file_aa is not None else None)
        problems = []
        if ref_rec.get('status') != 'certified' or ref_rec.get('certified_cost') is None:
            problems.append(f"reference not certified (status={ref_rec.get('status')})")
        if ref_rec.get('candidate_key') != candidate_key_hex:
            problems.append(f"reference candidate_key {str(ref_rec.get('candidate_key'))[:16]} != "
                            f'{candidate_key_hex[:16]} (must be the SAME candidate)')
        if ref_case_file_aa is None:
            if ref_overrides:
                problems.append(f'reference is not the case-file (D) configuration: overrides={ref_overrides}')
        elif ref_effective_aa.get('enabled'):
            problems.append(f'reference is not the D configuration: effective anderson_acceleration '
                            f'{ref_effective_aa} (case-file declaration {ref_case_file_aa} + overrides '
                            f'{ref_overrides or {}})')
        ref_gross = (ref_cl.get('recourse_components') or {}).get('gross_operational_cost')
        if ref_gross != ref_rec.get('certified_cost'):
            problems.append(f"reference component_levels gross {ref_gross} != record certified_cost "
                            f"{ref_rec.get('certified_cost')}")
        if problems:
            raise ValueError(f'invalid post_certification.reference {ref["eval_dir"]}: {problems}')
        resolved_ref = {
            'eval_dir': ref['eval_dir'],
            'evaluation_record_sha256': sha256_file(rec_path),
            'component_levels_terminal_sha256': sha256_file(cl_path),
            'certified_cost': ref_rec.get('certified_cost'),
            'certification_cycle': ref_rec.get('certification_cycle'),
            'candidate_key': ref_rec.get('candidate_key'),
            'campaign_spec_sha256': ref_rec.get('campaign_spec_sha256'),
            'configuration_overrides': ref_overrides or {},
        }
        if ref_case_file_aa is not None:  # only when the reference declared it, so old resolutions are unchanged
            resolved_ref['case_file_anderson_acceleration'] = ref_case_file_aa
            resolved_ref['effective_anderson_acceleration'] = ref_effective_aa
    if not (persist or polish or resolved_ref):
        return None
    return {'persist_certified_models': persist, 'hull_polish': polish, 'reference': resolved_ref}


def verify_reference_unchanged(resolved_ref):
    """Child side (before the run, and again before use): the reference files
    still hash to what the frozen spec recorded."""
    ref_dir = os.path.join(REPO, resolved_ref['eval_dir'])
    got = {'evaluation_record_sha256': sha256_file(os.path.join(ref_dir, 'evaluation_record.json')),
           'component_levels_terminal_sha256': sha256_file(os.path.join(ref_dir, 'component_levels_terminal.json'))}
    bad = {k: (resolved_ref[k], v) for k, v in got.items() if resolved_ref[k] != v}
    if bad:
        raise RuntimeError(f'post-certification reference changed since the spec was frozen: {bad}')
    return got


def eval_dir_name(key, label):
    return f'{key[:16]}_{_sanitize_id(label)}'


# ==============================================================================
#  frozen campaign spec
# ==============================================================================
def freeze_campaign_spec(campaign_root, campaign_id, candidates, configuration, cap, concurrency,
                         authority, required_consecutive_cycles=10, extra=None):
    """Write the campaign's frozen spec (write-once) and return (path, sha256, spec).

    `candidates`: list of (label, {node: (s, e)}) or (label, {node: (s, e)}, options).
    `options` (Addendum 25 item 2) may carry
      - 'overrides': this evaluation's configuration overrides, REPLACING the
        campaign-level `configuration['overrides']` for it (validated by
        `validate_overrides`: the AA flag and its reject-policy only);
      - 'post_certification': {'persist_certified_models': bool, 'hull_polish':
        bool, 'reference': {'eval_dir': ...} | None} (`resolve_post_certification`).
      - 'investment_year' (Addendum 27, W14): the SINGLE cohort year this
        candidate is placed at; omitted => `INVESTMENT_YEAR` (2025). It enters
        the canonical form (`canonical_candidate`), hence the candidate key --
        the canonical SHAPE is unchanged, so every key of every spec frozen
        before W14 is byte-identical. Multi-cohort candidates are not supported.
      - 'model_variant' (Addenda 28-29, W20): a MODEL VARIANT of the ageing law
        (`validate_model_variant`); enters the eval key; the entry and the spec
        then carry `model_variant_label` == MODEL_VARIANT_LABEL. Entries without
        it (and specs with no such entry) keep their exact format and keys.
      - 'flex_price_multiplier' (Addendum 34, W33): the uniform multiplier m on the
        DSO flexibility-price profile (`validate_flex_price_multiplier`). The entry
        records it; it enters the eval key only when != 1.0, and then the entry and
        the spec carry `flex_price_label` == FLEX_PRICE_LABEL. Entries without it
        keep their exact format and keys.
    `configuration` may carry (Addendum 30, W21) `ess_ageing_baseline` (the exact loaded ageing dict, see
    `validate_ess_ageing_baseline`) with `ess_ageing_baseline_label`; then the ESS parameters file must load to
    it (refused otherwise), its sha256 is pinned as `configuration.ess_params_file`, and it enters every entry's
    eval key. Without it the spec keeps its exact format and keys.
    P5.15 Addendum 39 (W47): `configuration` may carry `derived_instance` (`validate_derived_instance`; the case
    file must hash to its `case_sha256`, refused otherwise) and an entry's options `interface_deviation_premium`
    (`validate_interface_deviation_premium`); both enter the eval key and are recorded (spec
    `configuration.derived_instance`, entry `interface_deviation_premium`) only when given.
    P5.15 Addendum 46 ruling 7 (W84): `configuration` may carry `convergence_depth_tail`
    (`validate_convergence_depth_tail`) -- how a launcher enables the tail; recorded in the spec's configuration only
    when declared. W85 (Planner ruling Q1 on W84): a declaration enters every entry's eval key (`evaluation_key`);
    without one every key is byte-identical to the pre-W85 key. W86 (Planner ruling Q1 on W85): only an ENABLED
    declaration enters it; a declared-OFF tail gives the undeclared key (it is still recorded in the configuration).
    P5.15 Addendum 48 (W90): an entry's options may carry `release_solution_bookkeeping` (a bool,
    `validate_release_solution_bookkeeping`); recorded in the entry only when given; it NEVER enters the eval key.
    P5.15 Addendum 51 (W98): an entry's options may carry `certification_continuation`
    (`validate_certification_continuation`); recorded in the entry only when given; it ENTERS the eval key
    (`evaluation_key`), so a continuation never shares a key with the evaluation it continues.
    One entry = one EVALUATION: its `eval_key` (`evaluation_key`) identifies
    candidate x configuration; labels and eval keys must be unique (the same
    candidate may appear under two configurations)."""
    if os.path.exists(campaign_root) and os.listdir(campaign_root):
        raise RuntimeError(f'campaign root exists and is not empty (write-once): {campaign_root}')
    overrides = validate_overrides(configuration.get('overrides'))
    # Addendum 27 item 1: optional declaration of the case file's AA dict (None = not declared).
    case_file_aa = validate_case_file_anderson_acceleration(configuration.get('case_file_anderson_acceleration'))
    # Addendum 30 (W21): optional declaration of the ESS ageing parameters (None = not declared). When declared,
    # the ESS parameters file must load (production loader) to EXACTLY the declaration, types included, and its
    # sha256 is pinned in the spec; a label may only accompany a declaration.
    ess_ageing = validate_ess_ageing_baseline(configuration.get('ess_ageing_baseline'))
    ess_label = configuration.get('ess_ageing_baseline_label')
    ess_params_pin = None
    if ess_ageing is None and ess_label is not None:
        raise ValueError('ess_ageing_baseline_label given without an ess_ageing_baseline declaration')
    if ess_ageing is not None:
        if not isinstance(ess_label, str) or not ess_label.strip():
            raise ValueError('an ess_ageing_baseline declaration needs a non-empty ess_ageing_baseline_label')
        ess_path = os.path.join(REPO, ESS_PARAMS_FILE_REL)
        loaded = load_ess_ageing_parameters(ess_path)
        if ess_ageing_canonical_text(loaded) != ess_ageing_canonical_text(ess_ageing):
            raise ValueError(f'ess_ageing_baseline declaration does not equal what {ESS_PARAMS_FILE_REL} loads to '
                             f'(types included): declared {ess_ageing}, loaded {loaded}')
        ess_params_pin = {'path': ESS_PARAMS_FILE_REL, 'sha256': sha256_file(ess_path)}
    # Addendum 39 (W47): optional derived instance; the file must hash to the declaration at freeze time.
    derived = validate_derived_instance(configuration.get('derived_instance'))
    if derived is not None:
        case_abs = os.path.join(REPO, derived['case_path'])
        got = sha256_file(case_abs) if os.path.isfile(case_abs) else None
        if got != derived['case_sha256']:
            raise ValueError(f"derived_instance: {derived['case_path']} sha256 {got} != declared "
                             f"{derived['case_sha256']}")
    # W84 (Addendum 46 ruling 7): optional convergence-depth tail declaration (None = not declared = production OFF).
    tail = validate_convergence_depth_tail(configuration.get('convergence_depth_tail'))
    cand_entries, seen_labels, seen_keys = [], set(), set()
    any_model_variant = False
    any_flex_price = False
    for item in candidates:
        if len(item) == 2:
            (label, cand), options = item, {}
        else:
            label, cand, options = item
            options = dict(options or {})
        bad = sorted(set(options) - EVALUATION_OPTION_KEYS)
        if bad:
            raise ValueError(f'unsupported evaluation options {bad} for {label}; supported: '
                             f'{sorted(EVALUATION_OPTION_KEYS)}')
        canon = canonical_candidate(cand, investment_year=options.get('investment_year',
                                                                      INVESTMENT_YEAR))
        key = candidate_key(canon)
        eff_overrides = validate_overrides(options['overrides']) if 'overrides' in options else dict(overrides)
        model_variant = validate_model_variant(options.get('model_variant'))
        flex_m = validate_flex_price_multiplier(options.get('flex_price_multiplier'))
        premium = validate_interface_deviation_premium(options.get('interface_deviation_premium'))
        release_bk = validate_release_solution_bookkeeping(options.get('release_solution_bookkeeping'))  # W90: not keyed
        continuation = validate_certification_continuation(options.get('certification_continuation'))  # W98: keyed
        ekey = evaluation_key(key, eff_overrides, case_file_aa=case_file_aa, model_variant=model_variant,
                              ess_ageing_baseline=ess_ageing, flex_price_multiplier=flex_m,
                              derived_instance=derived, interface_deviation_premium=premium,
                              convergence_depth_tail=tail, certification_continuation=continuation)
        if label in seen_labels or ekey in seen_keys:
            raise ValueError(f'duplicate evaluation label or key (candidate x configuration): {label} / {ekey[:16]}')
        seen_labels.add(label)
        seen_keys.add(ekey)
        post_cert = resolve_post_certification(options.get('post_certification'), key)
        cand_entry = {'label': label, 'canonical': canon, 'key': key, 'eval_key': ekey,
                      'overrides': eff_overrides, 'post_certification': post_cert,
                      'eval_dir': eval_dir_name(ekey, label),
                      'working_dir_ids': eval_ids(campaign_id, ekey)}
        if case_file_aa is not None:  # only when declared, so undeclared specs keep their exact format
            cand_entry['effective_anderson_acceleration'] = effective_anderson_acceleration(case_file_aa, eff_overrides)
        if model_variant is not None:  # only when given, so every other entry keeps its exact format (W20)
            cand_entry['model_variant'] = model_variant
            cand_entry['model_variant_label'] = MODEL_VARIANT_LABEL
            any_model_variant = True
        if flex_m is not None:  # W33: only when given, so every other entry keeps its exact format
            cand_entry['flex_price_multiplier'] = flex_m
            if flex_price_multiplier_in_key(flex_m) is not None:
                cand_entry['flex_price_label'] = FLEX_PRICE_LABEL
                any_flex_price = True
        if premium is not None:  # W47: only when given, so every other entry keeps its exact format
            cand_entry['interface_deviation_premium'] = premium
        if release_bk is not None:  # W90: only when given, so every other entry keeps its exact format
            cand_entry['release_solution_bookkeeping'] = release_bk
        if continuation is not None:  # W98: only when given, so every other entry keeps its exact format
            cand_entry['certification_continuation'] = continuation
        cand_entries.append(cand_entry)
    os.makedirs(campaign_root, exist_ok=True)  # only after every validation above has passed
    try:
        head = _git(['rev-parse', 'HEAD'])
        case_file_commit = _git(['log', '-1', '--format=%H', '--', CASE_FILE_REL])
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f'git provenance unavailable: {error}') from error
    spec = {
        'schema': SPEC_SCHEMA,
        'campaign_id': campaign_id,
        'frozen_utc': _utc(),
        'authority': list(authority),
        'candidates': cand_entries,
        'configuration': {
            'name': configuration['name'],
            'arm_label': configuration.get('arm_label', 's39_D'),
            'case_file': CASE_FILE_REL,
            'case_file_sha256': sha256_file(CASE_FILE),
            'case_file_last_commit': case_file_commit,
            'overrides': overrides,
            'apply_rho': False,
            'full_diagnostics_in_rows': True,
            'note': configuration.get('note'),
        },
        'cap': int(cap),
        'concurrency': int(concurrency),
        'required_consecutive_cycles': int(required_consecutive_cycles),
        'bar_window_cycles': BAR_WINDOW,
        'thread_caps': dict(THREAD_CAP_ENV),
        'interpreter': PYTHON,
        'nlp_solver_path_env': _resolve_solver_path_from_dotenv(),
        'git_head': head,
        'harness': {'path': os.path.relpath(HARNESS_PATH, REPO), 'sha256': sha256_file(HARNESS_PATH)},
        'extra': extra or {},
    }
    if case_file_aa is not None:  # only when declared, so undeclared specs keep their exact format
        spec['configuration']['case_file_anderson_acceleration'] = case_file_aa
    if ess_ageing is not None:  # W21: only when declared, so undeclared specs keep their exact format
        try:
            ess_params_pin['last_commit'] = _git(['log', '-1', '--format=%H', '--', ESS_PARAMS_FILE_REL])
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f'git provenance unavailable: {error}') from error
        spec['configuration']['ess_ageing_baseline'] = ess_ageing
        spec['configuration']['ess_ageing_baseline_label'] = ess_label
        spec['configuration']['ess_params_file'] = ess_params_pin
    if derived is not None:  # W47: only when declared, so undeclared specs keep their exact format
        spec['configuration']['derived_instance'] = derived
    if tail is not None:  # W84: only when declared, so undeclared specs keep their exact format
        spec['configuration']['convergence_depth_tail'] = tail
    if any_model_variant:  # W20: a campaign holding a model variant says so at the top level
        spec['model_variant_label'] = MODEL_VARIANT_LABEL
    if any_flex_price:  # W33: a campaign holding a flexibility-price variant says so at the top level
        spec['flex_price_label'] = FLEX_PRICE_LABEL
    text = json.dumps(spec, indent=1, sort_keys=True, default=str)
    digest = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(campaign_root, f'campaign_spec_{_sanitize_id(campaign_id)}_{digest[:8]}.json')
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite frozen campaign spec: {path}')
    with open(path, 'w') as handle:
        handle.write(text)
    if sha256_file(path) != digest:
        raise RuntimeError('frozen spec hash mismatch after write')
    return path, digest, spec


def _resolve_solver_path_from_dotenv():
    """What the children will resolve for NLP_SOLVER_PATH (solver_parameters.py loads .env)."""
    value = os.environ.get('NLP_SOLVER_PATH')
    if value:
        return {'NLP_SOLVER_PATH': value, 'source': 'environment'}
    env_path = os.path.join(REPO, '.env')
    if os.path.exists(env_path):
        with open(env_path) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith('NLP_SOLVER_PATH='):
                    return {'NLP_SOLVER_PATH': line.split('=', 1)[1].strip().strip('"\''), 'source': '.env'}
    return {'NLP_SOLVER_PATH': None, 'source': 'unresolved'}


def load_frozen_spec(campaign_root, expected_sha256):
    hits = [f for f in os.listdir(campaign_root) if f.startswith('campaign_spec_') and f.endswith('.json')]
    matches = [f for f in hits if sha256_file(os.path.join(campaign_root, f)) == expected_sha256]
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one frozen spec with sha256 {expected_sha256} in '
                           f'{campaign_root}; found {matches} among {hits}')
    path = os.path.join(campaign_root, matches[0])
    with open(path) as handle:
        return path, json.load(handle)


# ==============================================================================
#  the campaign lock
# ==============================================================================
def acquire_campaign_lock(campaign_id, spec_sha256, lock_path=CAMPAIGN_LOCK_PATH,
                          legacy_lock_path=LEGACY_RUN_LOCK_PATH):
    if os.path.exists(legacy_lock_path):
        raise SystemExit(f'REFUSING TO RUN: the legacy one-run lock exists: {legacy_lock_path}')
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = open(lock_path).read().strip()
        except OSError:
            holder = 'unknown'
        raise SystemExit(f'REFUSING TO RUN: campaign lock held: {lock_path} ({holder})')
    content = {'pid': os.getpid(), 'campaign_id': campaign_id, 'campaign_spec_sha256': spec_sha256,
               'started_utc': _utc()}
    os.write(fd, json.dumps(content).encode())
    os.close(fd)
    if os.path.exists(legacy_lock_path):
        # Mirror of `p515_g_g1_g4_admm_gates._acquire_exclusive_run_lock`'s
        # post-create re-check (Addendum 25 item 2 follow-up): a legacy one-run
        # lock appeared between the first check and our O_EXCL create -- back
        # out (remove OUR lock) and refuse, so the two can never both proceed.
        os.remove(lock_path)
        raise SystemExit(f'REFUSING TO RUN: the legacy one-run lock appeared: {legacy_lock_path}')
    return content


def release_campaign_lock(lock_path=CAMPAIGN_LOCK_PATH, expected_pid=None):
    if not os.path.exists(lock_path):
        return False
    try:
        with open(lock_path) as handle:
            content = json.load(handle)
    except (OSError, ValueError):
        content = {}
    if expected_pid is not None and content.get('pid') != expected_pid:
        raise RuntimeError(f'campaign lock {lock_path} is not held by pid {expected_pid}: {content}')
    os.remove(lock_path)
    return True


def verify_child_lock(spec_sha256, lock_path=CAMPAIGN_LOCK_PATH):
    """Child side: the campaign lock must exist, name THIS child's parent and the same spec."""
    if not os.path.exists(lock_path):
        raise SystemExit(f'CHILD REFUSES: no campaign lock at {lock_path}')
    with open(lock_path) as handle:
        content = json.load(handle)
    if content.get('pid') != os.getppid():
        raise SystemExit(f'CHILD REFUSES: campaign lock pid {content.get("pid")} != parent pid {os.getppid()}')
    if content.get('campaign_spec_sha256') != spec_sha256:
        raise SystemExit('CHILD REFUSES: campaign lock spec sha256 differs from the spec this child was given')
    return content


# ==============================================================================
#  preconditions (parent)
# ==============================================================================
PRODUCTION_FILES_TO_CHECK_CLEAN = (
    'shared_resources_planning.py', 'network.py', 'network_data.py', 'shared_energy_storage_data.py',
    'admm_parameters.py', 'admm_anderson_acceleration.py', 'admm_persistent_workers.py',
    'model_construction_helpers.py', 'p515_g_g1_g4_admm_gates.py', 'p56a_oracle.py',
    'p514_n_instrumented_cstar.py', 'p515_s39_evaluate.py', 'p515_s40_polish_gap.py',
    'p515_s40_clone_capture_preflight.py', 'p515_s43_aa_flagoff_gate.py',
    'p515_s44_campaign_harness.py', CASE_FILE_REL,
    # Addendum 25 item 2: imported by the post-certification step / AA sidecar
    'p515_s41_hull_polish.py', 'p515_s42_exact_fix_rerun.py', 'p515_s43_aa_run.py',
    'p515_s40_cost_decomposition.py',
)
FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = ('p515_g_g1_g4_admm_gates.py', 'p515_s4')


def _ancestor_pids(max_depth=15):
    """This process and its ancestors (same reasoning as
    `p515_s40_clone_capture_preflight._ancestor_pids`, which is not imported
    here so the parent stays free of model imports)."""
    pids = {os.getpid()}
    current = os.getpid()
    for _ in range(max_depth):
        try:
            ppid_text = subprocess.run(['ps', '-o', 'ppid=', '-p', str(current)], capture_output=True,
                                       text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            break
        if not ppid_text:
            break
        ppid = int(ppid_text)
        if ppid <= 1 or ppid in pids:
            break
        pids.add(ppid)
        current = ppid
    return pids


def check_campaign_preconditions(campaign_root, extra_clean_files=(), lock_path=CAMPAIGN_LOCK_PATH,
                                 legacy_lock_path=LEGACY_RUN_LOCK_PATH):
    failures = []
    if os.path.exists(legacy_lock_path):
        failures.append(f'legacy one-run lock exists: {legacy_lock_path}')
    if os.path.exists(lock_path):
        failures.append(f'campaign lock exists: {lock_path}')
    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan process table: {error}')
        ps_output = ''
    excluded = {str(p) for p in _ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        pid = fields[1] if len(fields) > 1 else None
        if pid in excluded:
            continue
        if any(s in line for s in FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')
    if os.path.exists(campaign_root):
        failures.append(f'campaign root already exists (write-once): {campaign_root}')
    try:
        status = _git(['status', '--porcelain', '--'] + list(PRODUCTION_FILES_TO_CHECK_CLEAN)
                      + list(extra_clean_files))
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not run git status: {error}')
        status = ''
    if status.strip():
        failures.append(f'production/harness files not clean in git:\n{status}')
    return failures


# ==============================================================================
#  evaluate(batch) -- the parent side
# ==============================================================================
class CampaignContext:
    def __init__(self, campaign_root, spec_path, spec_sha256, spec, log=print, child_extra_args=()):
        self.campaign_root = campaign_root
        self.spec_path = spec_path
        self.spec_sha256 = spec_sha256
        self.spec = spec
        self.concurrency = int(spec['concurrency'])
        self.log = log
        self.child_extra_args = tuple(child_extra_args)  # checks-only (stub mode); never set by a campaign
        self.evals_root = os.path.join(campaign_root, 'evals')


def _entry_eval_key(entry):
    return entry.get('eval_key', entry['key'])  # v1 specs (s44_gate) carry no eval_key: eval key == candidate key


def _spec_candidate(ctx, candidate):
    """Resolve one batch item to its spec entry: a str is an evaluation LABEL;
    a dict is a candidate, which must then match exactly one entry (a candidate
    listed under two configurations must be named by label).

    Addendum 27 (W14): a candidate dict carrying the key `investment_year` is read
    as the CANONICAL shape `{'investment_year': y, 'nodes': {node: (s, e)}}`; a
    plain `{node: (s, e)}` map keeps its pre-W14 meaning exactly (year 2025)."""
    if isinstance(candidate, str):
        hits = [e for e in ctx.spec['candidates'] if e['label'] == candidate]
        if len(hits) != 1:
            raise ValueError(f'evaluation label {candidate!r} is not (uniquely) in the frozen campaign spec')
        return hits[0]
    if 'investment_year' in candidate:
        if set(candidate) != {'investment_year', 'nodes'}:
            raise ValueError("a candidate naming 'investment_year' must be "
                             "{'investment_year': y, 'nodes': {node: (s, e)}}; got keys "
                             f'{sorted(candidate)}')
        canon = canonical_candidate(candidate['nodes'],
                                    investment_year=candidate['investment_year'])
    else:
        canon = canonical_candidate(candidate)
    key = candidate_key(canon)
    hits = [e for e in ctx.spec['candidates'] if e['key'] == key]
    if len(hits) > 1:
        raise ValueError(f'candidate {key[:16]} appears under {len(hits)} configurations; name it by label')
    if hits:
        return hits[0]
    raise ValueError(f'candidate {canon} (key {key[:16]}) is not in the frozen campaign spec')


def _child_command(ctx, entry):
    return [PYTHON, '-u', HARNESS_PATH, '--child', '--campaign-root', ctx.campaign_root,
            '--spec-sha256', ctx.spec_sha256, '--eval-key', _entry_eval_key(entry)] + list(ctx.child_extra_args)


def _rusage_dict(ru):
    return {k: getattr(ru, k) for k in ('ru_utime', 'ru_stime', 'ru_maxrss', 'ru_minflt', 'ru_majflt',
                                         'ru_nvcsw', 'ru_nivcsw')}


def _tail(path, n=40):
    try:
        with open(path, errors='replace') as handle:
            return handle.readlines()[-n:]
    except OSError:
        return []


def _barrier_record_for_missing(ctx, entry, eval_dir, exit_code):
    return {
        'schema': RECORD_SCHEMA,
        'campaign_id': ctx.spec['campaign_id'],
        'campaign_spec_path': os.path.relpath(ctx.spec_path, REPO),
        'campaign_spec_sha256': ctx.spec_sha256,
        'candidate_label': entry['label'], 'candidate_canonical': entry['canonical'],
        'candidate_key': entry['key'], 'eval_key': _entry_eval_key(entry),
        'status': 'harness_error', 'barrier': True,
        'barrier_cause': f'child exited with code {exit_code} and wrote no evaluation_record.json',
        'stderr_tail': _tail(os.path.join(eval_dir, 'child_stderr.log')),
        'stdout_tail': _tail(os.path.join(eval_dir, 'child_stdout.log')),
        'synthesized_by_parent': True,
        # Addendum 27 (W5): same schema as a child record; the parent cannot know the child's values.
        'anderson_acceleration_effective_in_child': None,
        'case_file_sha256_in_child': None,
        # W85: the tail as the spec declares it (None = undeclared = production OFF), and what the child's per-round
        # append preserved up to its death -- its tail checklist included, when it got that far (else None).
        'convergence_depth_tail_declared_in_spec': ctx.spec['configuration'].get('convergence_depth_tail'),
        'convergence_depth_per_round_append_recovered': recover_convergence_depth_append(eval_dir),
        # W20: a model-variant entry's record carries the variant and its label on every path.
        **({'model_variant': entry['model_variant'], 'model_variant_label': MODEL_VARIANT_LABEL}
           if entry.get('model_variant') is not None else {}),
        # W21: a declared-ESS-ageing spec's record carries the declaration and its label on every path.
        **({'ess_ageing_baseline': ctx.spec['configuration']['ess_ageing_baseline'],
            'ess_ageing_baseline_label': ctx.spec['configuration'].get('ess_ageing_baseline_label'),
            'ess_params_sha256_in_child': None}
           if ctx.spec['configuration'].get('ess_ageing_baseline') is not None else {}),
        # W33: a flexibility-price entry's record carries the multiplier (and its label when != 1.0) on every path.
        **(_flex_price_record_fields(entry) if 'flex_price_multiplier' in entry else {}),
        # W47: a derived-instance spec / premium entry carries its declaration on every path.
        **_derived_record_fields(ctx.spec, entry),
        # W90: an entry declaring option (b) carries it on every path.
        **({'release_solution_bookkeeping': entry['release_solution_bookkeeping']}
           if 'release_solution_bookkeeping' in entry else {}),
    }


def _derived_record_fields(spec, entry):
    """W47: the declaration fields every record of a derived-instance spec / premium entry carries (empty
    otherwise, so every other record keeps its exact format)."""
    out = {}
    if spec['configuration'].get('derived_instance') is not None:
        out['derived_instance'] = spec['configuration']['derived_instance']
    if 'interface_deviation_premium' in entry:
        out['interface_deviation_premium'] = entry['interface_deviation_premium']
    return out


def _flex_price_record_fields(entry):
    """W33: the multiplier fields every record of a flexibility-price entry carries (label only when != 1.0)."""
    out = {'flex_price_multiplier': entry.get('flex_price_multiplier')}
    if flex_price_multiplier_in_key(entry.get('flex_price_multiplier')) is not None:
        out['flex_price_label'] = FLEX_PRICE_LABEL
    return out


HEARTBEAT_FILE = 'campaign_heartbeat.json'   # the untagged (pre-W74) campaign heartbeat, one shared file per root


def heartbeat_file_name(heartbeat_tag=None):
    """P5.15 Addendum 44 (W74): the campaign-heartbeat file name of one `evaluate()` call. Untagged -> the pre-W74
    shared `campaign_heartbeat.json` (unchanged: a campaign run as ONE call keeps its exact file). Tagged ->
    `campaign_heartbeat_<sanitized tag>.json`, one file per pair / wave, so a later launch in the same campaign root
    can never overwrite an earlier launch's end state (in the alpha row, pair 3's launch overwrote pair 2's)."""
    if heartbeat_tag is None:
        return HEARTBEAT_FILE
    return f'campaign_heartbeat_{_sanitize_id(heartbeat_tag)}.json'


def evaluate(batch, ctx, heartbeat_tag=None):
    """STEP4_DFO_METHOD.md 2.7: `evaluate(batch: list[x]) -> list[record]`.

    `batch`: list of candidates ({node: (s, e)}) or evaluation labels (str),
    each present in the frozen campaign spec, no duplicates (a candidate listed
    under two configurations must be given by label). Returns the evaluation records in batch
    order (a parent-synthesized barrier record when a child left none).

    `heartbeat_tag` (P5.15 Addendum 44, W74): None (default) writes the campaign heartbeat to the shared
    `campaign_heartbeat.json` exactly as before. A campaign that calls `evaluate()` more than once on one root (pairs,
    waves) passes a distinct tag per call, e.g. 'pair_2': the heartbeat then goes to `heartbeat_file_name(tag)`,
    carries the tag, and is write-once per call -- the call refuses, before launching anything, if that file exists."""
    entries = [_spec_candidate(ctx, x) for x in batch]
    keys = [_entry_eval_key(e) for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate evaluations in one batch')
    heartbeat_path = os.path.join(ctx.campaign_root, heartbeat_file_name(heartbeat_tag))
    heartbeat_extra = {} if heartbeat_tag is None else {'heartbeat_tag': heartbeat_tag}
    if heartbeat_tag is not None and os.path.exists(heartbeat_path):
        raise RuntimeError(f'per-wave campaign heartbeat already exists (write-once per tag, never reusable): '
                           f'{heartbeat_path}')
    os.makedirs(ctx.evals_root, exist_ok=True)
    for entry in entries:
        eval_dir = os.path.join(ctx.evals_root, entry['eval_dir'])
        if os.path.exists(eval_dir):
            raise RuntimeError(f'evaluation dir already exists (write-once, never reusable): {eval_dir}')
    pending = list(range(len(entries)))
    running = {}  # pid -> (index, proc, started, handles)
    results = [None] * len(entries)
    child_env = dict(os.environ)
    child_env.update(THREAD_CAP_ENV)
    max_concurrent_observed = 0
    last_heartbeat = 0.0
    timeline = []

    def _launch(i):
        entry = entries[i]
        eval_dir = os.path.join(ctx.evals_root, entry['eval_dir'])
        os.makedirs(eval_dir)
        out_h = open(os.path.join(eval_dir, 'child_stdout.log'), 'w')
        err_h = open(os.path.join(eval_dir, 'child_stderr.log'), 'w')
        cmd = _child_command(ctx, entry)
        proc = subprocess.Popen(cmd, cwd=REPO, env=child_env, stdout=out_h, stderr=err_h,
                                stdin=subprocess.DEVNULL)
        started = time.time()
        _write_once_json(os.path.join(eval_dir, 'launch.json'), {
            'command': cmd, 'cwd': REPO, 'thread_caps_in_child_env': dict(THREAD_CAP_ENV),
            'pid': proc.pid, 'parent_pid': os.getpid(), 'started_utc': _utc(),
            'campaign_spec_sha256': ctx.spec_sha256, 'label': entry['label'], 'key': entry['key'],
            'eval_key': _entry_eval_key(entry),
        })
        running[proc.pid] = (i, proc, started, (out_h, err_h))
        timeline.append({'event': 'start', 'label': entry['label'], 'pid': proc.pid, 't': started})
        ctx.log(f'[S44-HARNESS] launched {entry["label"]} (key {entry["key"][:16]}) pid={proc.pid}')

    def _reap(pid, status, ru):
        i, proc, started, handles = running.pop(pid)
        for h in handles:
            h.close()
        exit_code = os.waitstatus_to_exitcode(status)
        proc.returncode = exit_code  # reaped via os.wait4; keep Popen consistent
        ended = time.time()
        entry = entries[i]
        eval_dir = os.path.join(ctx.evals_root, entry['eval_dir'])
        with open(os.path.join(eval_dir, 'exit_code.txt'), 'w') as handle:
            handle.write(f'{exit_code}\n')
        _write_once_json(os.path.join(eval_dir, 'wait4_rusage.json'), {
            'rusage': _rusage_dict(ru), 'ru_maxrss_units': 'bytes on macOS/BSD, kilobytes on Linux',
            'semantics': 'os.wait4 rusage of the child process as reaped by the parent',
            'wall_s_parent_view': ended - started, 'exit_code': exit_code,
        })
        record_path = os.path.join(eval_dir, 'evaluation_record.json')
        if os.path.exists(record_path):
            with open(record_path) as handle:
                record = json.load(handle)
        else:
            record = _barrier_record_for_missing(ctx, entry, eval_dir, exit_code)
            _write_once_json(os.path.join(eval_dir, 'parent_barrier_record.json'), record)
        record = dict(record)
        record['parent_view'] = {'exit_code': exit_code, 'wall_s': ended - started,
                                 'wait4_ru_maxrss': ru.ru_maxrss}
        results[i] = record
        timeline.append({'event': 'end', 'label': entry['label'], 'pid': pid, 't': ended,
                         'exit_code': exit_code})
        ctx.log(f'[S44-HARNESS] finished {entry["label"]} pid={pid} exit={exit_code} '
                f'status={record.get("status")} wall={ended - started:.0f}s')

    def _heartbeat():
        status = []
        for pid, (i, _p, started, _h) in running.items():
            entry = entries[i]
            eval_dir = os.path.join(ctx.evals_root, entry['eval_dir'])
            hb_path = os.path.join(eval_dir, f"heartbeat_{ctx.spec['configuration']['arm_label']}.json")
            hb = None
            if os.path.exists(hb_path):
                try:
                    with open(hb_path) as handle:
                        hb = json.load(handle)
                except (OSError, ValueError):
                    hb = 'unreadable (being written)'
            status.append({'label': entry['label'], 'pid': pid, 'elapsed_s': time.time() - started,
                           'production_heartbeat': hb})
        _atomic_write_json(heartbeat_path, {
            'utc': _utc(), 'parent_pid': os.getpid(), 'running': status,
            'pending': [entries[i]['label'] for i in pending],
            'done': [entries[i]['label'] for i, r in enumerate(results) if r is not None], **heartbeat_extra})
        ctx.log('[S44-HARNESS] heartbeat ' + '; '.join(
            f"{s['label']}: {s['elapsed_s']:.0f}s cycle="
            f"{s['production_heartbeat'].get('cycle') if isinstance(s['production_heartbeat'], dict) else None}"
            for s in status))

    try:
        while pending or running:
            while pending and len(running) < ctx.concurrency:
                _launch(pending.pop(0))
            max_concurrent_observed = max(max_concurrent_observed, len(running))
            for pid in list(running):
                reaped_pid, status, ru = os.wait4(pid, os.WNOHANG)
                if reaped_pid == pid:
                    _reap(pid, status, ru)
            now = time.time()
            if running and now - last_heartbeat >= HEARTBEAT_EVERY_S:
                _heartbeat()
                last_heartbeat = now
            if running or pending:
                time.sleep(POLL_S)
    except BaseException:
        for pid, (_i, proc, _s, _h) in list(running.items()):
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        for pid in list(running):
            try:
                _rp, status, ru = os.wait4(pid, 0)
                _reap(pid, status, ru)
            except ChildProcessError:
                running.pop(pid, None)
        raise
    _atomic_write_json(heartbeat_path, {
        'utc': _utc(), 'parent_pid': os.getpid(), 'running': [], 'pending': [],
        'done': [e['label'] for e in entries], **heartbeat_extra})
    evaluate.last_batch_info = {'max_concurrent_observed': max_concurrent_observed, 'timeline': timeline}
    if heartbeat_tag is not None:
        evaluate.last_batch_info['heartbeat_file'] = os.path.relpath(heartbeat_path, REPO)
    return results


# ==============================================================================
#  the STEP4 2.5 record (pure function of an evaluation's own artifacts)
# ==============================================================================
def _max_step_last_n(rows, n=BAR_WINDOW):
    """The bar (STEP4 2.5): max over the last `n` cycles of the GROSS cost step
    |gross_operational_cost[k] - gross_operational_cost[k-1]| (P5.15 Addendum 27,
    P5_15_S45_REVERIFY_RULING.md consequence 2). The step at cycle k uses the row of
    cycle k-1 looked up in the FULL trajectory `rows` (not only the window), so the
    first window row's step is exact whenever cycle k-1 exists; it is None (not
    available) when cycle k-1 is absent (cycle 1: no predecessor) or either gross
    value is None (a failed cycle) -- the same availability rule production applies
    to `objective_change_abs` (`previous_recourse = recourse` every cycle,
    shared_resources_planning.py:3192)."""
    tail = rows[-n:] if len(rows) >= n else rows
    gross_by_cycle = {r.get('cycle'): r.get('gross_operational_cost') for r in rows}
    cycles = [r.get('cycle') for r in rows]
    window = []
    for r in tail:
        c, g = r.get('cycle'), r.get('gross_operational_cost')
        prev_c = (c - 1) if isinstance(c, int) and (c - 1) in gross_by_cycle else None
        g_prev = gross_by_cycle.get(prev_c) if prev_c is not None else None
        step = abs(g - g_prev) if (g is not None and g_prev is not None) else None
        window.append({'cycle': c, 'gross_operational_cost': g, 'predecessor_cycle': prev_c,
                       'predecessor_gross_operational_cost': g_prev, 'gross_step_abs': step})
    vals = [w['gross_step_abs'] for w in window if w['gross_step_abs'] is not None]
    return {'definition': (f'max over the last {n} cycles of |gross_operational_cost[k] - '
                           f'gross_operational_cost[k-1]| (settlement-excluded gross cost; the predecessor of the '
                           f'first window row is taken from the full trajectory; a step is unavailable when cycle '
                           f'k-1 is absent or either gross value is None)'),
            'value': max(vals) if vals else None, 'n_cycles_in_window': len(tail),
            'n_steps_available': len(vals),
            'trajectory_cycles_contiguous_from_1': cycles == list(range(1, len(rows) + 1)),
            'window': window}


def _max_net_recourse_step_last_n(rows, n=BAR_WINDOW):
    """REPORTED, not the bar: max `objective_change_abs` over the last `n` cycles.
    Production computes `objective_change_abs` on the NET recourse
    (shared_resources_planning.py:2849, `abs(recourse - previous_recourse)` with
    `recourse = net_operational_recourse`, :2814), i.e. gross minus the terminal
    salvage credit. Until Addendum 27 this was the record's `bar` (mislabelled
    "|gross cost step|"); kept as `bar_net_recourse_step_reported`."""
    tail = rows[-n:] if len(rows) >= n else rows
    steps = [(r.get('cycle'), r.get('objective_change_abs')) for r in tail]
    vals = [s for _c, s in steps if s is not None]
    return {'definition': (f'max objective_change_abs over the last {n} cycles (production: |net_operational_recourse'
                           f'[k] - net_operational_recourse[k-1]|, shared_resources_planning.py:2849/2814; NET of the '
                           f'terminal salvage credit; reported, not the bar)'),
            'value': max(vals) if vals else None, 'n_cycles_in_window': len(tail),
            'n_steps_available': len(vals), 'window': [{'cycle': c, 'objective_change_abs': s} for c, s in steps]}


def _storage_per_node(canonical, esso_capture, floor_terminal, published_caps):
    out = {}
    per_node_floor = {}
    if floor_terminal and floor_terminal.get('available'):
        for e in floor_terminal.get('per_node_per_cohort_year') or []:
            per_node_floor.setdefault(str(e.get('node_id')), []).append(e)
    for node, (s_val, e_val) in canonical['nodes'].items():
        cap = esso_capture.get(node) if esso_capture else None
        has_storage = (s_val > 0.0)
        soh_active, efc_cells = {}, {}
        if cap is not None:
            rated = cap.get('es_e_rated_per_unit') or {}
            soh = cap.get('es_soh_per_unit_cumul') or {}
            for key, r in rated.items():
                if r:
                    soh_active[key] = soh.get(key)
            for key, v in (cap.get('efc_per_day_per_cohort_year') or {}).items():
                if v:
                    efc_cells[key] = v.get('efc_per_day')
        floor_entries = per_node_floor.get(node, [])
        out[node] = {
            's_mva': s_val, 'e_mwh': e_val, 'has_storage': has_storage,
            'efc_per_day_max': cap.get('efc_per_day_max') if cap else None,
            'efc_per_day_per_cohort_year': efc_cells,
            'terminal_soh_per_active_cohort_year': soh_active,
            'terminal_soh_min_over_active_cohort_years': (
                min(v for v in soh_active.values() if v is not None)
                if any(v is not None for v in soh_active.values()) else None),
            'soh_floor_rows_active_at_terminal': sorted(
                f"({e.get('y_inv')}, {e.get('y')})" for e in floor_entries if e.get('active')),
            'published_available_capacity_terminal': (published_caps or {}).get(node),
            'note_zero_node': (None if has_storage else
                               'zero-capacity node: no active cohort; EFC undefined (None), SoH variables '
                               'fixed at 1.0 by production and not reported as a result'),
        }
    return out


def build_evaluation_record(*, spec, spec_path, spec_sha256, entry, report, component_levels,
                            floor_terminal, published_caps, peak_rss, wall, eval_dir, extra=None):
    """The STEP4_DFO_METHOD.md 2.5 record, built ONLY from the evaluation's own
    report/artifacts. Pure function (no I/O) -- exercised by the zero-solve
    checks on the D reference's committed artifacts."""
    import p515_s39_evaluate as E39  # generic helpers, BY IMPORT
    import p515_g_g1_g4_admm_gates as G

    rows = report.get('cycle_trajectory') or []
    cap = int(spec['cap'])
    required = int(spec['required_consecutive_cycles'])
    cert = E39._certification_from_trajectory(rows, cap, required) if rows else {'certified': False}
    stopped = G._derive_stopped_by_from_trajectory(rows, cap=cap, required_consecutive=required) if rows else None
    certified = bool(cert.get('certified'))
    last = rows[-1] if rows else {}
    bar = _max_step_last_n(rows)
    terminal_ratios = {c: {'primal': last.get(f'boyd_{c}_primal_ratio'),
                           'dual': last.get(f'boyd_{c}_dual_ratio'),
                           'max': E39._terminal_ratios(last)[c] if last else None} for c in CHANNELS}
    rc = (component_levels or {}).get('recourse_components') or {}
    status = 'certified' if certified else ('not_certified' if rows else 'no_trajectory')
    cause = None
    if not certified:
        cause = ('cap reached without certification' if rows and len(rows) >= cap else
                 'run ended before the cap without certification' if rows else 'no trajectory')
    record = {
        'schema': RECORD_SCHEMA,
        'campaign_id': spec['campaign_id'],
        'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256,
        'candidate_label': entry['label'],
        'candidate_canonical': entry['canonical'],
        'candidate_key': entry['key'],
        'instance_note': 'candidate_key = sha256 of the canonical candidate; identifies the problem instance',
        'working_dir_ids': entry['working_dir_ids'],
        'eval_dir': os.path.relpath(eval_dir, REPO),
        'configuration': spec['configuration'],
        'cap': cap,
        'required_consecutive_cycles': required,
        'status': status,
        'barrier': not certified,
        'barrier_cause': cause,
        'objective_convention': ('gross_operational_cost: settlement-EXCLUDED gross operational cost '
                                 '(the oracle cost convention); net_operational_recourse = gross minus '
                                 'terminal salvage credit'),
        'certified_cost': report.get('gross_operational_cost') if certified else None,
        'terminal_gross_operational_cost': report.get('gross_operational_cost'),
        'terminal_net_operational_recourse': rc.get('net_operational_recourse'),
        'bar': bar,
        'bar_net_recourse_step_reported': _max_net_recourse_step_last_n(rows),
        'certification': cert,
        'certification_cycle': cert.get('certification_cycle'),
        'cycles_run': len(rows),
        'stopped_by_trajectory': stopped,
        'first_pass_cycle_per_channel': {c: E39._first_pass_cycle(rows, c) for c in CHANNELS},
        'terminal_ratios_per_channel': terminal_ratios,
        'rule_ten': {
            'reported_not_gated': True,
            'terminal_objective_change_abs': report.get('terminal_objective_change_abs'),
            'terminal_objective_tolerance': report.get('terminal_objective_tolerance'),
            'terminal_step_over_threshold': report.get('rule_ten_terminal_step_over_threshold'),
            'boyd_terminal_ratio_max_per_channel': {c: terminal_ratios[c]['max'] for c in CHANNELS},
        },
        'component_decomposition_totals_weighted': (component_levels or {}).get('totals_weighted'),
        'recourse_components': rc,
        'settlement_remainder': {
            'definition': ('recourse_components.interface_settlement_total: T_TSO + sum(T_DSO), the '
                           'non-cancelling interface settlement at the terminal point, excluded from the cost'),
            'value': rc.get('interface_settlement_total'),
        },
        'storage_per_node': _storage_per_node(entry['canonical'], report.get('esso_capture') or {},
                                              floor_terminal, published_caps),
        'local_solve_failures': report.get('local_solve_failures'),
        'network_failures_summary': report.get('network_failures_summary'),
        'solve_profile': report.get('solve_profile'),
        'wall_time_s': wall,
        'peak_rss': peak_rss,
    }
    if extra:
        record.update(extra)
    return record


# ==============================================================================
#  rule eleven for the record: capture paths asserted BEFORE the run
# ==============================================================================
RECORD_TRAJECTORY_FIELDS = tuple(sorted(set(PER_CYCLE_TRAJECTORY_FIELDS) | {
    'boyd_v_primal_ratio', 'boyd_pf_primal_ratio', 'boyd_ess_primal_ratio', 'boyd_v_dual_ratio',
    'boyd_pf_dual_ratio', 'boyd_ess_dual_ratio', 'boyd_v_channel_pass', 'boyd_pf_channel_pass',
    'boyd_ess_channel_pass', 'cycle_convergence', 'consecutive_converged_cycles',
    'objective_change_abs', 'objective_tolerance', 'gross_operational_cost', 'local_solves_ok'}))


def assert_record_capture_paths():
    """Fails fast (before any solve) if a quantity the 2.5 record needs has no capture path."""
    import inspect
    import shared_resources_planning as srp
    import p515_g_g1_g4_admm_gates as G
    import p514_n_instrumented_cstar as N
    import p515_s39_evaluate as E39
    src = inspect.getsource(srp)
    checks = {}
    derived_by_cycle_row = {'cycle', 'recourse', 'objective_change_ratio'}
    for field in RECORD_TRAJECTORY_FIELDS:
        checks[f'trajectory_field_{field}'] = (f"'{field}':" in src) or (field in derived_by_cycle_row)
    for name in ('es_avg_ch_dch_per_unit', 'es_soh_per_unit_cumul', 'es_e_rated_per_unit'):
        checks[f'esso_capture_attr_{name}'] = name in N.REQUIRED_ESSO_ATTRS
    for fn in ('run_admm_arm', 'write_boyd_terminal_s35ref', 's38_pf_capture_hooks',
               's39_exempt_until_capture_hooks', '_s35ref_terminal_floor_and_efc',
               '_derive_stopped_by_from_trajectory', '_identify_soh_floor_rows'):
        checks[f'harness_fn_{fn}'] = callable(getattr(G, fn, None))
    for fn in ('_certification_from_trajectory', '_first_pass_cycle', '_terminal_ratios'):
        checks[f'evaluator_fn_{fn}'] = callable(getattr(E39, fn, None))
    # settlement remainder: produced by production `_get_operational_recourse_components`
    # (key 'interface_settlement_total'), which `write_component_levels_terminal` calls.
    checks['component_levels_recourse_key_interface_settlement_total'] = (
        "'interface_settlement_total':" in inspect.getsource(srp._get_operational_recourse_components)
        and '_get_operational_recourse_components' in inspect.getsource(G.write_component_levels_terminal))
    checks['run_admm_arm_accepts_investment_map'] = (
        'investment_map' in inspect.signature(G.run_admm_arm).parameters)
    checks['run_admm_arm_post_run_hook_gets_state'] = ("'state' in inspect.signature(post_run_hook)"
                                                        in inspect.getsource(G.run_admm_arm))
    checks['production_state_has_peak_rss'] = "'peak_rss_ru_maxrss':" in src
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN: capture paths missing for the evaluation record: {missing}')
    return checks


def assert_convergence_depth_tail_capture(spec):
    """P5.15 Addendum 46 ruling 7 (W84, Planner ruling Q3): rule eleven for the per-solve floor status and the tail,
    asserted in the child BEFORE the run for EVERY evaluation. Fails fast (AssertionError) if production has no
    capture path for the records / tail state, if production's default is not OFF, or if `run_admm_arm` does not
    hand the returned state to the post-run hook. Returns the checklist, which RECORDS whether the tail is enabled for
    this run (`tail_enabled_for_this_run`: True iff the spec declares it with enabled True) -- so a run is never
    ambiguous about its own configuration, and a forgotten enable shows as False instead of passing silently."""
    import inspect
    import network as NET
    import shared_resources_planning as srp
    import p515_g_g1_g4_admm_gates as G
    from admm_parameters import ADMMParameters
    declared = validate_convergence_depth_tail(spec['configuration'].get('convergence_depth_tail'))
    production_default = dict(ADMMParameters().convergence_depth_tail)
    run_src = inspect.getsource(srp._run_operational_planning)
    attempt_src = inspect.getsource(NET._run_smopf_solver_attempt)
    checks = {
        'production_default_tail_off': production_default.get('enabled') is False,
        'production_tail_helper_present': callable(getattr(srp, 'convergence_depth_tail_enabled', None)),
        'production_both_returns_carry_the_records': (
            run_src.count("'network_ipopt_solve_records': network_ipopt_solve_records,") == 2),
        'production_both_returns_carry_the_tail_state': (
            run_src.count("'convergence_depth_tail': convergence_depth_tail_state,") == 2),
        'production_records_drained_at_initialisation': (
            'network_ipopt_solve_records.extend(_drain_network_ipopt_solve_records(planning_problem, 0))' in run_src),
        'production_records_drained_every_cycle': (
            'network_ipopt_solve_records.extend(_drain_network_ipopt_solve_records(planning_problem, iter))'
            in run_src),
        'network_attempt_appends_a_record_after_the_solve': (
            0 <= attempt_src.find('result = solver.solve(model') < attempt_src.find('_append_ipopt_solve_record(')),
        'run_admm_arm_post_run_hook_gets_state': ("'state' in inspect.signature(post_run_hook)"
                                                  in inspect.getsource(G.run_admm_arm)),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (W84): convergence-depth capture paths missing: {missing}')
    return {
        'checks': checks,
        'declared': declared,
        'tail_enabled_for_this_run': bool(declared is not None and declared['enabled']),
        'compl_inf_tol_tail': declared['compl_inf_tol'] if declared is not None and declared['enabled'] else None,
        'source': ('campaign spec configuration.convergence_depth_tail' if declared is not None
                   else 'not declared -> production default (ADMMParameters.convergence_depth_tail, OFF)'),
        'production_default': production_default,
        'persisted_files': [NETWORK_IPOPT_SOLVE_RECORDS_FILE, CONVERGENCE_DEPTH_TAIL_STATE_FILE],
    }


def apply_convergence_depth_tail_declaration(admm, declared):
    """W84: called by the configuration hook (before any model is built or solved). A declaration is written to
    `admm.convergence_depth_tail` (a new dict) and read back; without one nothing is written. Either way the tail in
    force (`shared_resources_planning.convergence_depth_tail_enabled`) must equal the declaration's `enabled`
    (undeclared -> False) and, when declared, the dict in force must equal the declaration; otherwise RuntimeError."""
    import copy
    import shared_resources_planning as srp
    declared = validate_convergence_depth_tail(declared)
    before = copy.deepcopy(getattr(admm, 'convergence_depth_tail', None))
    if declared is not None:
        admm.convergence_depth_tail = dict(declared)
    after = copy.deepcopy(getattr(admm, 'convergence_depth_tail', None))
    enabled_in_force = srp.convergence_depth_tail_enabled(admm)
    expected_enabled = bool(declared is not None and declared['enabled'])
    applied = {
        'declared': declared,
        'source': ('campaign spec configuration.convergence_depth_tail' if declared is not None
                   else 'not declared -> production default (ADMMParameters.convergence_depth_tail)'),
        'before': before, 'after': after,
        'enabled_in_force': enabled_in_force, 'expected_enabled': expected_enabled,
        'dict_in_force_equals_declaration': (after == declared) if declared is not None else None,
    }
    applied['ok'] = enabled_in_force is expected_enabled and (declared is None or after == declared)
    if not applied['ok']:
        raise RuntimeError(f'convergence-depth tail in force does not match the declaration (W84): {applied}')
    return applied


# ==============================================================================
#  the child (one evaluation)
# ==============================================================================
def _child_verify_env():
    bad = {k: os.environ.get(k) for k, v in THREAD_CAP_ENV.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f'CHILD REFUSES: thread caps not in force in the child environment: {bad}')
    return {k: os.environ.get(k) for k in THREAD_CAP_ENV}


def _child_stub(args, spec, spec_path, entry, eval_dir, lock_content, env_caps, started):
    """CHECKS ONLY (`p515_s44_campaign_harness_checks.py`): no model import, no
    solve -- sleeps, allocates, and writes a stub record through the same
    parent/child plumbing. Refused unless the frozen spec says test_only_stub."""
    if not spec.get('extra', {}).get('test_only_stub'):
        raise SystemExit('CHILD REFUSES: stub mode requested but the frozen spec is not a test-only stub spec')
    mode = args.stub_mode
    if mode == 'fail':
        print('stub child: failing on purpose', file=sys.stderr)
        raise SystemExit(3)
    blob = bytearray(int(args.stub_alloc_mb) * (1 << 20))
    for i in range(0, len(blob), 4096):
        blob[i] = 1
    time.sleep(float(args.stub_sleep_s))
    record = {
        'schema': RECORD_SCHEMA, 'stub': True, 'campaign_id': spec['campaign_id'],
        'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': args.spec_sha256,
        'candidate_label': entry['label'], 'candidate_canonical': entry['canonical'],
        'candidate_key': entry['key'], 'status': 'stub', 'barrier': False,
        'child_pid': os.getpid(), 'child_ppid': os.getppid(), 'lock_content': lock_content,
        'thread_caps_seen': env_caps, 'started_t': started, 'ended_t': time.time(),
        'peak_rss': {'child_self_ru_maxrss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
    }
    _write_once_json(os.path.join(eval_dir, 'evaluation_record.json'), record)


# ==============================================================================
#  model variants (Addenda 28-29, W20) -- CHILD SIDE (model code imported locally)
# ==============================================================================
def model_variant_expected(model_variant, sed):
    """The constants a validated `model_variant` MUST produce in the built ESSO model, from closed
    forms and the case file's own (N, D): k = N * D / (-ln R) (None when ageing is off: the D row is
    D == 0), phi as consumed by the SoH row (1.0 when ageing is off), the SoH-point mode."""
    from math import log
    model_variant = validate_model_variant(model_variant)
    cal = sed.params.ageing.calibration
    k = cal.cycles_n * cal.reference_dod_d / (-log(model_variant['eol_retention_r']))
    enabled = model_variant['ageing_enabled']
    return {'k': k if enabled else None,
            'k_formula': f"cycles_n * reference_dod_d / (-ln eol_retention_r) = {cal.cycles_n} * "
                         f"{cal.reference_dod_d} / (-ln {model_variant['eol_retention_r']})",
            'phi_cal_in_model': model_variant['calendar_retention_per_year'] if enabled else 1.0,
            'available_energy_soh_point': model_variant['available_energy_soh_point'],
            'ageing_enabled': enabled,
            'd_row_form': 'D * 2kE == 365 n avg' if enabled else 'D == 0'}


def _ageing_state(sed):
    ageing = sed.params.ageing
    per_ess = []
    for year in sed.years:
        for ess in sed.shared_energy_storages[year]:
            per_ess.append({'year': str(year), 'bus': ess.bus, 't_cal': ess.t_cal, 'cl_nom': ess.cl_nom,
                            'dod_nom': ess.dod_nom, 'soh_min': ess.soh_min, 'cl_eff': ess.cl_eff,
                            'phi_cal': ess.phi_cal})
    return {'calibration': {'status': ageing.calibration.status, 'cycles_n': ageing.calibration.cycles_n,
                            'reference_dod_d': ageing.calibration.reference_dod_d,
                            'eol_retention_r': ageing.calibration.eol_retention_r},
            'calendar_retention_per_year': ageing.calendar_retention_per_year,
            'available_energy_soh_point': getattr(sed, 'available_energy_soh_point', None),
            'ageing_enabled': getattr(sed, 'ageing_enabled', None),
            'per_ess': per_ess}


def apply_model_variant(sed, model_variant):
    """Apply a validated `model_variant` to THIS evaluation's (deep-copied) shared-ESS data, BEFORE any
    ESSO model is built: the calibration's eol_retention_r and calendar_retention_per_year on the loaded
    ageing parameters, re-applied to every SharedEnergyStorage by production's own
    `EnergyStorageAgeingParameters.apply_to` (so cl_eff = k is recomputed by production); the two
    ageing-model switches on the shared-ESS data object. Verified to have taken effect: every ESS carries
    the expected k and phi, and every other ageing constant is unchanged. Returns the applied record."""
    import shared_energy_storage_data as SED
    model_variant = validate_model_variant(model_variant)
    if tuple(SED.AVAILABLE_ENERGY_SOH_POINTS) != MODEL_VARIANT_SOH_POINTS:
        raise RuntimeError(f'MODEL_VARIANT_SOH_POINTS {MODEL_VARIANT_SOH_POINTS} != production '
                           f'{SED.AVAILABLE_ENERGY_SOH_POINTS}')
    ageing = sed.params.ageing
    if not ageing.calibration.is_active():
        raise RuntimeError('model_variant.eol_retention_r needs an ACTIVE degradation calibration (otherwise '
                           'production consumes cl_nom and the retention would silently not apply)')
    before = _ageing_state(sed)
    ageing.calibration.eol_retention_r = model_variant['eol_retention_r']
    ageing.calendar_retention_per_year = model_variant['calendar_retention_per_year']
    for year in sed.years:
        for ess in sed.shared_energy_storages[year]:
            ageing.apply_to(ess)
    sed.available_energy_soh_point = model_variant['available_energy_soh_point']
    sed.ageing_enabled = model_variant['ageing_enabled']
    settings = SED._esso_ageing_model_settings(sed)  # production's own validation
    after = _ageing_state(sed)
    expected = model_variant_expected(model_variant, sed)
    k_expected = model_variant_expected(dict(model_variant, ageing_enabled=True), sed)['k']
    unchanged = ('bus', 't_cal', 'cl_nom', 'dod_nom', 'soh_min')
    checks = {
        'settings_read_by_production': settings == (model_variant['available_energy_soh_point'],
                                                    model_variant['ageing_enabled']),
        'every_ess_cl_eff_is_k': all(e['cl_eff'] == k_expected for e in after['per_ess']),
        'every_ess_phi_cal_is_variant': all(e['phi_cal'] == model_variant['calendar_retention_per_year']
                                            for e in after['per_ess']),
        'other_ageing_constants_unchanged': ([{k: e[k] for k in unchanged} for e in before['per_ess']]
                                             == [{k: e[k] for k in unchanged} for e in after['per_ess']]),
        'calibration_n_d_status_unchanged': all(before['calibration'][k] == after['calibration'][k]
                                                for k in ('status', 'cycles_n', 'reference_dod_d')),
    }
    failed = sorted(k for k, v in checks.items() if not v)
    if failed:
        raise RuntimeError(f'model_variant did not take effect as specified: {failed}')
    return {'model_variant': model_variant, 'label': MODEL_VARIANT_LABEL, 'before': before, 'after': after,
            'expected_in_model': expected, 'k_of_calibration_as_applied': k_expected, 'checks': checks}


def _eq_residual(con):
    import pyomo.environ as pe
    lhs, rhs = con.expr.args
    return pe.value(lhs) - pe.value(rhs)


def _degradation_triples(model, y_inv):
    """{y: (D row, SoH row, floor row)} for cohort y_inv, from production's own construction-order
    bookkeeping `model._esso_cohort_constraints` (the same grouping `_identify_soh_floor_rows` uses)."""
    rows = [(idx, y) for name, idx, y in model._esso_cohort_constraints[y_inv]
            if name == 'energy_storage_capacity_degradation']
    if len(rows) % 3:
        raise RuntimeError(f'energy_storage_capacity_degradation rows for cohort {y_inv} are not triples')
    triples = {}
    for i in range(0, len(rows), 3):
        (i_d, y_d), (i_s, y_s), (i_f, y_f) = rows[i:i + 3]
        if not y_d == y_s == y_f:
            raise RuntimeError(f'degradation triple years disagree: {rows[i:i + 3]}')
        family = model.energy_storage_capacity_degradation
        triples[y_d] = (family[i_d], family[i_s], family[i_f])
    return triples


def _available_energy_row(model, y_inv, y):
    from pyomo.core.expr.visitor import identify_variables
    target = model.es_e_available_per_unit[y_inv, y]
    hits = [con for con in model.available_e_capacity_unit.values()
            if any(v is target for v in identify_variables(con.body, include_fixed=True))]
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one available-energy row for ({y_inv}, {y}), found {len(hits)}')
    return hits[0]


def model_variant_readback(model, sed, y_inv):
    """READ BACK k, phi and the SoH-point mode from a BUILT ESSO model, numerically, from its own rows
    (never from the settings): MUTATES Var/Param values of `model` -- call it on a probe or a clone only.
      D row  (cohort y_inv, first block y0): residual r(D, avg) with E = 1: slope in D = 2kE -> k;
             slope in avg = -365 n; a row with slope 1 in D, none in avg and zero intercept is D == 0.
      SoH row (y0): with SoH = 0, D = 0 the residual is -phi**n -> phi (n from the data); with D = 0.3 the
             ratio must be exp(-0.3) (the exponential form).
      available row (y1 = y0 + 1 when in the window): with E_av = 0, E_rated = 1, SoH_end = 0.9,
             SoH_prev = 0.95, D = 0.2 the residual is -X; X == SoH_end -> 'end',
             X == SoH_prev * exp(-0.1) * phi**(n/2) -> 'mid'."""
    from math import exp, isclose
    import pyomo.environ as pe
    years = list(sed.years)
    n_data = sed.years[years[y_inv]]
    triples = _degradation_triples(model, y_inv)
    y0 = min(triples)
    d_row, soh_row, floor_row = triples[y0]
    e_inv = 1.0
    model.es_e_investment_fixed[y_inv].set_value(e_inv)
    d_var = model.es_D_per_unit[y_inv, y0]
    a_var = model.es_avg_ch_dch_per_unit[y_inv, y0]
    a_var.set_value(1.0)
    d_var.set_value(0.0)
    r00 = _eq_residual(d_row)
    d_var.set_value(1.0)
    r10 = _eq_residual(d_row)
    a_var.set_value(2.0)
    d_var.set_value(0.0)
    r02 = _eq_residual(d_row)
    slope_d, slope_avg = r10 - r00, r02 - r00
    if slope_avg == 0.0 and slope_d == 1.0 and r00 == 0.0:
        d_form, k, n_from_d = 'D == 0', None, None
    else:
        d_form, k, n_from_d = 'D * 2kE == 365 n avg', slope_d / (2.0 * e_inv), -slope_avg / 365.0
    s_var = model.es_soh_per_unit_cumul[y_inv, y0]
    s_var.set_value(0.0)
    d_var.set_value(0.0)
    phi_pow_n = -_eq_residual(soh_row)
    phi = phi_pow_n ** (1.0 / n_data)
    d_var.set_value(0.3)
    exp_ratio = (-_eq_residual(soh_row)) / phi_pow_n
    y1 = y0 + 1 if (y0 + 1) in triples else y0
    row = _available_energy_row(model, y_inv, y1)
    soh_end, soh_prev, d_val = 0.9, 0.95, 0.2
    model.es_e_available_per_unit[y_inv, y1].set_value(0.0)
    model.es_e_rated_per_unit[y_inv, y1].set_value(1.0)
    model.es_soh_per_unit_cumul[y_inv, y1].set_value(soh_end)
    if y1 > y0:
        model.es_soh_per_unit_cumul[y_inv, y1 - 1].set_value(soh_prev)
    else:
        soh_prev = 1.0
    model.es_D_per_unit[y_inv, y1].set_value(d_val)
    x_val = -_eq_residual(row)
    x_mid = soh_prev * exp(-d_val / 2.0) * phi ** (n_data / 2.0)
    rtol = MODEL_VARIANT_READBACK_RTOL
    mode = ('end' if isclose(x_val, soh_end, rel_tol=rtol, abs_tol=0.0) else
            'mid' if isclose(x_val, x_mid, rel_tol=rtol, abs_tol=0.0) else 'unrecognized')
    return {'y_inv': y_inv, 'y0': y0, 'y1_available_row': y1, 'n_years_data': n_data,
            'd_row_form': d_form, 'k': k, 'n_years_from_d_row': n_from_d,
            'phi_cal_in_model': phi, 'soh_row_exp_form_ok': isclose(exp_ratio, exp(-0.3), rel_tol=rtol),
            'available_energy_soh_point': mode,
            'available_row_probe': {'soh_end': soh_end, 'soh_prev': soh_prev, 'D': d_val, 'X': x_val,
                                    'X_end': soh_end, 'X_mid_closed_form': x_mid},
            'floor_row_lower': None if floor_row.lower is None else float(pe.value(floor_row.lower))}


def compare_readback(readback, expected):
    """Readback vs `model_variant_expected`: k and phi to MODEL_VARIANT_READBACK_RTOL (relative), the
    D-row form, the SoH-point mode and the exponential form exactly. Returns {check: bool}."""
    from math import isclose
    rtol = MODEL_VARIANT_READBACK_RTOL
    k_ok = ((readback['k'] is None and expected['k'] is None)
            or (readback['k'] is not None and expected['k'] is not None
                and isclose(readback['k'], expected['k'], rel_tol=rtol, abs_tol=0.0)))
    n_ok = (readback['n_years_from_d_row'] is None
            or isclose(readback['n_years_from_d_row'], readback['n_years_data'], rel_tol=rtol, abs_tol=0.0))
    return {'k': k_ok,
            'n_years_from_d_row_equals_data': n_ok,
            'phi_cal_in_model': isclose(readback['phi_cal_in_model'], expected['phi_cal_in_model'],
                                        rel_tol=rtol, abs_tol=0.0),
            'd_row_form': readback['d_row_form'] == expected['d_row_form'],
            'available_energy_soh_point': readback['available_energy_soh_point']
            == expected['available_energy_soh_point'],
            'soh_row_exp_form': readback['soh_row_exp_form_ok']}


def model_variant_readback_models(models, sed, model_variant, investment_year, clone=True):
    """`model_variant_readback` for every node's ESSO model at the cohort of `investment_year`, against
    `model_variant_expected`. `clone=True` reads back from CLONES (the given models are left untouched)."""
    expected = model_variant_expected(model_variant, sed)
    y_inv = [int(y) for y in sed.years].index(int(investment_year))
    per_node, all_ok = {}, True
    for node_id, model in models.items():
        target = model.clone() if clone else model
        readback = model_variant_readback(target, sed, y_inv)
        checks = compare_readback(readback, expected)
        all_ok = all_ok and all(checks.values())
        per_node[str(node_id)] = {'readback': readback, 'checks': checks}
        if clone:
            del target
    return {'expected': expected, 'per_node': per_node, 'all_match': all_ok,
            'method': ('numerical read-back from the built rows (model_variant_readback); '
                       + ('on clones of the given models' if clone else 'on probe models'))}


def ess_ageing_baseline_expected(declared):
    """The constants a validated `ess_ageing_baseline` declaration MUST produce in a built ESSO model, from closed
    forms of the DECLARATION alone: k = cycles_n * reference_dod_d / (-ln eol_retention_r) when the calibration
    is ACTIVE, else cycle_life_nominal; phi = calendar_retention_per_year; the floor row's lower bound =
    minimum_soh; the default SoH point ('end') and ageing on (no model variant)."""
    from math import log
    declared = validate_ess_ageing_baseline(declared)
    cal = declared['calibration']
    if cal['status'] == 'ACTIVE':
        k = cal['cycles_n'] * cal['reference_dod_d'] / (-log(cal['eol_retention_r']))
        k_formula = (f"cycles_n * reference_dod_d / (-ln eol_retention_r) = {cal['cycles_n']} * "
                     f"{cal['reference_dod_d']} / (-ln {cal['eol_retention_r']})")
    else:
        k, k_formula = declared['cycle_life_nominal'], 'cycle_life_nominal (calibration not ACTIVE)'
    return {'k': k, 'k_formula': k_formula, 'phi_cal_in_model': declared['calendar_retention_per_year'],
            'floor_row_lower': declared['minimum_soh'], 'available_energy_soh_point': 'end',
            'ageing_enabled': True, 'd_row_form': 'D * 2kE == 365 n avg'}


def ess_ageing_readback_models(models, sed, declared, investment_year, clone=True):
    """`model_variant_readback` (k, phi, SoH-point mode, floor-row lower bound, read NUMERICALLY from the built
    rows) for every node's ESSO model at the cohort of `investment_year`, against `ess_ageing_baseline_expected`
    (the declaration's closed forms). `clone=True` reads back from CLONES (the given models are untouched)."""
    from math import isclose
    expected = ess_ageing_baseline_expected(declared)
    y_inv = [int(y) for y in sed.years].index(int(investment_year))
    per_node, all_ok = {}, True
    for node_id, model in models.items():
        target = model.clone() if clone else model
        readback = model_variant_readback(target, sed, y_inv)
        checks = compare_readback(readback, expected)
        checks['floor_row_lower'] = (readback['floor_row_lower'] is not None and isclose(
            readback['floor_row_lower'], expected['floor_row_lower'], rel_tol=MODEL_VARIANT_READBACK_RTOL, abs_tol=0.0))
        all_ok = all_ok and all(checks.values())
        per_node[str(node_id)] = {'readback': readback, 'checks': checks}
        if clone:
            del target
    return {'expected': expected, 'per_node': per_node, 'all_match': all_ok,
            'method': ('numerical read-back from the built rows (model_variant_readback) against the declaration\'s '
                       'closed forms; ' + ('on clones of the given models' if clone else 'on probe models'))}


def verify_ess_ageing_in_child(sed, declared, pin):
    """Child side (configuration hook, before any ESSO model of the run is built): the shared-ESS data of THIS
    evaluation was read from the pinned file, loaded to EXACTLY the declaration (canonical JSON, types
    included), and every SharedEnergyStorage carries the declaration's soh_min / phi and production's k
    (`effective_cycle_constant`). Returns the evidence; raises on any mismatch."""
    declared = validate_ess_ageing_baseline(declared)
    pin = pin or {}
    file_used = os.path.join(sed.data_dir, 'SharedESS', sed.params_file)
    pinned_path = os.path.join(REPO, pin.get('path') or '')
    loaded = ess_ageing_parameters_as_loaded(sed.params.ageing)
    k_prod = sed.params.ageing.effective_cycle_constant()
    per_ess = [{'year': str(y), 'bus': e.bus, 'soh_min': e.soh_min, 'phi_cal': e.phi_cal, 'cl_eff': e.cl_eff}
               for y in sed.years for e in sed.shared_energy_storages[y]]
    sha_now = sha256_file(pinned_path) if os.path.isfile(pinned_path) else None
    checks = {
        'pin_path_is_ess_params_file': pin.get('path') == ESS_PARAMS_FILE_REL,
        'file_used_by_production_is_pinned_file': (os.path.isfile(file_used) and os.path.isfile(pinned_path)
                                                   and os.path.samefile(file_used, pinned_path)),
        'file_sha256_equals_pin': sha_now is not None and sha_now == pin.get('sha256'),
        'loaded_equals_declaration_types_included': (ess_ageing_canonical_text(loaded)
                                                     == ess_ageing_canonical_text(declared)),
        'every_ess_soh_min_is_declared': all(e['soh_min'] == declared['minimum_soh'] for e in per_ess),
        'every_ess_phi_cal_is_declared': all(e['phi_cal'] == declared['calendar_retention_per_year'] for e in per_ess),
        'every_ess_cl_eff_is_production_k': all(e['cl_eff'] == k_prod for e in per_ess),
    }
    out = {'declared': declared, 'loaded': loaded, 'file_used_by_production': os.path.relpath(file_used, REPO),
           'file_sha256': sha_now, 'pin': pin, 'k_production': k_prod, 'per_ess': per_ess, 'checks': checks}
    failed = sorted(k for k, v in checks.items() if not v)
    if failed:
        raise RuntimeError(f'ess_ageing_baseline: the loaded shared-ESS ageing parameters are not the declared '
                           f'ones: {failed}; declared {declared}, loaded {loaded}, file {out["file_used_by_production"]} '
                           f'sha256 {sha_now} vs pin {pin.get("sha256")}')
    return out


def ageing_trajectory_terminal(models, sed):
    """READ-ONLY capture from the run's own ESSO models (no mutation): per node, per ACTIVE (y_inv, y)
    (e_rated not fixed): E_rated, throughput, EFC/day = avg / (2 E_rated), D, the END-of-block SoH, the
    SoH used for available energy (E_available / E_rated), the previous block's end SoH, and the mid-block
    closed form SoH_prev * exp(-D/2) * phi**(n/2) beside it; plus each node's terminal salvage value."""
    import pyomo.environ as pe
    import shared_energy_storage_data as SED
    from math import exp
    soh_point, enabled = SED._esso_ageing_model_settings(sed)
    years = list(sed.years)
    out = {'available_energy_soh_point': soh_point, 'ageing_enabled': enabled, 'nodes': {}}
    for node_id, model in models.items():
        idx = sed.get_shared_energy_storage_idx(node_id)
        cells = []
        for y_inv in model.years:
            ess = sed.shared_energy_storages[years[y_inv]][idx]
            n = sed.years[years[y_inv]]
            phi = ess.phi_cal if enabled else 1.0
            for y in model.years:
                if model.es_e_rated_per_unit[y_inv, y].fixed:
                    continue
                rated = pe.value(model.es_e_rated_per_unit[y_inv, y])
                if not rated:
                    continue
                avg = pe.value(model.es_avg_ch_dch_per_unit[y_inv, y])
                d_val = pe.value(model.es_D_per_unit[y_inv, y])
                soh_end = pe.value(model.es_soh_per_unit_cumul[y_inv, y])
                soh_prev = pe.value(model.es_soh_per_unit_cumul[y_inv, y - 1]) if y > y_inv else 1.0
                e_av = pe.value(model.es_e_available_per_unit[y_inv, y])
                cells.append({'y_inv': y_inv, 'investment_year': str(years[y_inv]), 'y': y,
                              'block_year': str(years[y]), 'n_years': n, 'e_rated': rated,
                              'avg_ch_dch': avg, 'efc_per_day': avg / (2.0 * rated), 'D': d_val,
                              'soh_prev_end': soh_prev, 'soh_end': soh_end,
                              'soh_used_for_available_energy': e_av / rated, 'e_available': e_av,
                              'soh_mid_closed_form': soh_prev * exp(-d_val / 2.0) * phi ** (n / 2.0),
                              'phi_cal_in_model': phi, 'cl_eff': ess.cl_eff})
        out['nodes'][str(node_id)] = {'cells': cells, 'salvage_value': pe.value(model.salvage_value)}
    return out


# ==============================================================================
#  the flexibility-price multiplier (P5.15 Addendum 34, W33) -- CHILD SIDE
# ==============================================================================
def _scaled_cost_flex(array, m):
    """The ONLY arithmetic of the override: a NEW float64 array m * array. Never in place -- production binds the
    SAME `planning.cost_flex[year][day]` array object to the TSO and to every DSO block
    (shared_resources_planning.py, the `cost_flex` bindings of `_read_planning_problem`)."""
    import numpy as np
    return np.asarray(array, dtype=np.float64) * m


def _flex_price_objective_summary(model):
    """Standard repn of a BUILT block's `objective` (values computed): the coefficient of every FLEX_PRICE_VARS
    variable by name, and digests of everything else (other linear terms, quadratic terms, constant, nonlinear
    part). Read-only."""
    from pyomo.repn import generate_standard_repn
    repn = generate_standard_repn(model.objective.expr, compute_values=True, quadratic=True)
    flex, other = {}, []
    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        if var.parent_component().local_name in FLEX_PRICE_VARS:
            if var.name in flex:
                raise RuntimeError(f'flex variable {var.name} appears twice in the linear repn')
            flex[var.name] = float(coef)
        else:
            other.append([var.name, repr(float(coef))])
    quad = sorted([v1.name, v2.name, repr(float(c))] for (v1, v2), c in zip(repn.quadratic_vars, repn.quadratic_coefs))
    flex_in_quadratic = any(v.parent_component().local_name in FLEX_PRICE_VARS
                            for pair in repn.quadratic_vars for v in pair)
    nonlinear = None if repn.nonlinear_expr is None else str(repn.nonlinear_expr)
    return {'flex': flex,
            'other_linear_sha256': hashlib.sha256(json.dumps(sorted(other)).encode()).hexdigest(),
            'n_other_linear': len(other),
            'quadratic_sha256': hashlib.sha256(json.dumps(quad).encode()).hexdigest(), 'n_quadratic': len(quad),
            'flex_in_quadratic': flex_in_quadratic,
            'constant': repr(float(repn.constant)) if repn.constant is not None else None,
            'nonlinear_sha256': None if nonlinear is None else hashlib.sha256(nonlinear.encode()).hexdigest(),
            'flex_in_nonlinear': bool(nonlinear) and any(n in nonlinear for n in FLEX_PRICE_VARS)}


def compare_flex_price_summaries(before, after, m):
    """`_flex_price_objective_summary` at m (after) against m = 1 (before), for the SAME block: the flex
    coefficients scale by m (relative deviation <= FLEX_PRICE_READBACK_REL_TOL; bitwise-exact count reported) and
    every other part of the objective is identical. Returns {check: bool} plus the deviation figures."""
    keys_equal = set(before['flex']) == set(after['flex'])
    devs, exact = [], 0
    for name, c1 in before['flex'].items():
        cm = after['flex'].get(name)
        if cm is None:
            continue
        target = m * c1
        devs.append(abs(cm - target) / abs(target) if target else abs(cm))
        exact += int(cm == target)
    max_dev = max(devs) if devs else None
    checks = {
        'flex_variables_identical_and_present': keys_equal and bool(before['flex']),
        'flex_coefficients_scale_by_m': bool(devs) and max_dev <= FLEX_PRICE_READBACK_REL_TOL,
        'other_linear_terms_identical': (before['other_linear_sha256'] == after['other_linear_sha256']
                                         and before['n_other_linear'] == after['n_other_linear']),
        'quadratic_terms_identical': before['quadratic_sha256'] == after['quadratic_sha256'],
        'constant_identical': before['constant'] == after['constant'],
        'nonlinear_part_identical': before['nonlinear_sha256'] == after['nonlinear_sha256'],
        'flex_variables_only_linear': not (before['flex_in_quadratic'] or after['flex_in_quadratic']
                                           or before['flex_in_nonlinear'] or after['flex_in_nonlinear']),
    }
    return checks, {'n_flex_coefficients': len(devs), 'n_bitwise_exact': exact, 'max_rel_dev': max_dev}


def _flex_price_block_iter(planning):
    for node_id in sorted(planning.distribution_networks):
        dn = planning.distribution_networks[node_id]
        for year in dn.years:
            for day in dn.days:
                yield node_id, dn, year, day


def apply_flex_price_multiplier(planning, m):
    """Apply a validated flexibility-price multiplier m to THIS evaluation's planning object BEFORE any DSO model
    is built (`run_admm_arm`'s pre_solve_hook): every DSO block's `network[year][day].cost_flex` -- the array
    `model_construction_helpers.flexibility_cost` reads when the block's objective is BUILT (it is baked into the
    `flex_cost_scenario` Expression as constants, not a mutable Param, so rescaling after the build would need
    rewriting expressions) -- is REPLACED by a new array m * cost_flex (`_scaled_cost_flex`), for every year / day
    / hour uniformly; each DSO holder's `cost_flex` dict is rebound to those arrays. The TSO's arrays (its flexibility
    charge excludes the ADN-interface loads, its only loads on SRP1) and `planning.cost_flex` are left untouched;
    no file is edited.

    READ-BACK before returning (zero solves): for EVERY DSO block a probe is built by production
    (`network.build_model(params)`) before and after the replacement, and the objective's standard repn is
    compared (`compare_flex_price_summaries`): flex coefficients = m x the m = 1 coefficients, everything else
    identical. Any mismatch raises. Returns a JSON-able record; the original arrays are returned under the
    private key '_original_arrays' (popped by the caller, used for the post-run read-back)."""
    import numpy as np
    from definitions import OBJ_MIN_COST
    m = validate_flex_price_multiplier(m)
    if m is None:
        raise ValueError('apply_flex_price_multiplier needs a multiplier')
    tso = planning.transmission_network
    tso_arrays = {(y, d): tso.network[y][d].cost_flex for y in tso.years for d in tso.days}
    tso_values = {k: np.array(v, copy=True) for k, v in tso_arrays.items()}
    planning_arrays = {(y, d): planning.cost_flex[y][d] for y in planning.cost_flex for d in planning.cost_flex[y]}
    energy_arrays = {(n, y, d): dn.network[y][d].cost_energy_p for n, dn, y, d in _flex_price_block_iter(planning)}
    precondition = {}
    for node_id in sorted(planning.distribution_networks):
        params = planning.distribution_networks[node_id].params
        precondition[str(node_id)] = {'obj_type_is_min_cost': params.obj_type == OBJ_MIN_COST,
                                      'fl_reg': bool(params.fl_reg)}
    if not all(all(v.values()) for v in precondition.values()):
        raise RuntimeError(f'flex_price_multiplier: a DSO does not price flexibility at cost_flex '
                           f'(obj_type OBJ_MIN_COST and fl_reg required): {precondition}')
    before = {}
    for node_id, dn, year, day in _flex_price_block_iter(planning):
        before[(node_id, year, day)] = _flex_price_objective_summary(dn.network[year][day].build_model(dn.params))
    original = {}
    for node_id in sorted(planning.distribution_networks):
        dn = planning.distribution_networks[node_id]
        rebound = {}
        for year in dn.years:
            rebound[year] = {}
            for day in dn.days:
                net = dn.network[year][day]
                orig = net.cost_flex
                if not isinstance(orig, np.ndarray) or orig.dtype != np.float64:
                    raise RuntimeError(f'flex_price_multiplier: DSO {node_id} {year} {day} cost_flex is not a float64 '
                                       f'ndarray ({type(orig).__name__})')
                scaled = _scaled_cost_flex(orig, m)
                net.cost_flex = scaled
                rebound[year][day] = scaled
                original[(node_id, year, day)] = orig
        dn.cost_flex = rebound
    per_block, all_match, agg = {}, True, {'n_flex_coefficients': 0, 'n_bitwise_exact': 0, 'max_rel_dev': 0.0}
    for node_id, dn, year, day in _flex_price_block_iter(planning):
        after = _flex_price_objective_summary(dn.network[year][day].build_model(dn.params))
        checks, figures = compare_flex_price_summaries(before[(node_id, year, day)], after, m)
        all_match = all_match and all(checks.values())
        per_block[f'DSO|{node_id}|{year}|{day}'] = {'checks': checks, **figures}
        agg['n_flex_coefficients'] += figures['n_flex_coefficients']
        agg['n_bitwise_exact'] += figures['n_bitwise_exact']
        agg['max_rel_dev'] = max(agg['max_rel_dev'], figures['max_rel_dev'] or 0.0)
    arrays = [(k, dn.network[k[1]][k[2]].cost_flex) for k, dn in
              ((k, planning.distribution_networks[k[0]]) for k in original)]
    checks = {
        'every_dso_array_is_m_times_original_bitwise': all(np.array_equal(a, _scaled_cost_flex(original[k], m))
                                                           for k, a in arrays),
        'every_dso_array_is_a_new_object': all(a is not original[k] for k, a in arrays),
        'no_dso_array_shared_with_the_tso': not any(a is t for _k, a in arrays for t in tso_arrays.values()),
        'dso_holder_cost_flex_rebound': all(planning.distribution_networks[n].cost_flex[y][d]
                                            is planning.distribution_networks[n].network[y][d].cost_flex
                                            for (n, y, d) in original),
        'tso_arrays_unchanged': all(tso.network[y][d].cost_flex is tso_arrays[(y, d)]
                                    and np.array_equal(tso_arrays[(y, d)], tso_values[(y, d)])
                                    for (y, d) in tso_arrays),
        'planning_cost_flex_unchanged': all(planning.cost_flex[y][d] is a for (y, d), a in planning_arrays.items()),
        'dso_energy_prices_unchanged': all(planning.distribution_networks[n].network[y][d].cost_energy_p is a
                                           for (n, y, d), a in energy_arrays.items()),
        'readback_every_dso_block_objective': all_match and len(per_block) == len(original),
    }
    profile = {}
    for (node_id, year, day), orig in sorted(original.items(), key=lambda kv: (kv[0][0], kv[0][1], str(kv[0][2]))):
        if node_id != min(planning.distribution_networks):
            continue  # the same profile object is bound to every DSO (checked below); record it once
        profile[f'{year}|{day}'] = {'base': [float(v) for v in orig.ravel()],
                                    'applied': [float(v) for v in _scaled_cost_flex(orig, m).ravel()]}
    shared_profile = all(np.array_equal(original[(n, y, d)], original[(min(planning.distribution_networks), y, d)])
                         for (n, y, d) in original)
    out = {'flex_price_multiplier': m, 'label': FLEX_PRICE_LABEL if m != 1.0 else None,
           'where_applied': ('pre_solve_hook (after _construct_arm_planning, before run_operational_planning): '
                             'DSO network[year][day].cost_flex replaced by m * cost_flex (new arrays); DSO '
                             'holder cost_flex rebound; TSO and planning.cost_flex untouched; no file edited'),
           'precondition': precondition, 'checks': checks,
           'readback_pre_run': {'method': ('probe DSO blocks built by production network.build_model(params) before '
                                           'and after the replacement; objective standard repn compared '
                                           '(compare_flex_price_summaries)'),
                                'rel_tol': FLEX_PRICE_READBACK_REL_TOL, 'n_blocks': len(per_block),
                                'all_match': all_match, **agg, 'per_block': per_block},
           'profile_applied_per_year_day': profile,
           'profile_identical_across_dso': shared_profile,
           '_original_arrays': original}
    failed = sorted(k for k, v in checks.items() if not v)
    if failed:
        raise RuntimeError(f'flex_price_multiplier {m} did not take effect as specified: {failed}; '
                           f'failing blocks: {[k for k, v in per_block.items() if not all(v["checks"].values())][:6]}')
    return out


def flex_price_readback_run_models(dso_models, planning, original, m):
    """READ-ONLY read-back from the run's OWN DSO models (post-run): for every block and every
    `flex_cost_scenario[s_m, s_o]` Expression (the flexibility term of the objective), the coefficient of each
    FLEX_PRICE_VARS variable must equal the closed form production computes, `cost_flex[s_m][p] * baseMVA` with the
    APPLIED array (bitwise), and m x the same with the ORIGINAL array (FLEX_PRICE_READBACK_REL_TOL). Also re-checks
    that the TSO arrays are still the planning's."""
    from pyomo.repn import generate_standard_repn
    m = validate_flex_price_multiplier(m)
    per_block, all_match = {}, True
    n_coef = n_exact_applied = n_exact_ratio = 0
    max_dev = 0.0
    for node_id, dn, year, day in _flex_price_block_iter(planning):
        net = dn.network[year][day]
        model = dso_models[node_id][year][day]
        applied, orig, base = net.cost_flex, original[(node_id, year, day)], net.baseMVA
        ok_applied = ok_ratio = True
        count = 0
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                repn = generate_standard_repn(model.flex_cost_scenario[s_m, s_o].expr, compute_values=True)
                if repn.nonlinear_expr is not None or repn.quadratic_vars:
                    ok_applied = ok_ratio = False
                for var, coef in zip(repn.linear_vars, repn.linear_coefs):
                    if var.parent_component().local_name not in FLEX_PRICE_VARS:
                        ok_applied = ok_ratio = False
                        continue
                    p = var.index()[-1]
                    expected_applied = float(applied[s_m][p] * base)
                    target = m * float(orig[s_m][p] * base)
                    dev = abs(float(coef) - target) / abs(target) if target else abs(float(coef))
                    max_dev = max(max_dev, dev)
                    ok_applied = ok_applied and float(coef) == expected_applied
                    ok_ratio = ok_ratio and dev <= FLEX_PRICE_READBACK_REL_TOL
                    n_exact_applied += int(float(coef) == expected_applied)
                    n_exact_ratio += int(float(coef) == target)
                    count += 1
        n_coef += count
        ok = ok_applied and ok_ratio and count > 0
        all_match = all_match and ok
        per_block[f'DSO|{node_id}|{year}|{day}'] = {'n_coefficients': count, 'equals_applied_closed_form': ok_applied,
                                                    'equals_m_times_original_within_tol': ok_ratio}
    tso = planning.transmission_network
    tso_ok = all(tso.network[y][d].cost_flex is planning.cost_flex[y][d] for y in tso.years for d in tso.days)
    return {'method': ('generate_standard_repn of every DSO block flex_cost_scenario Expression of the run\'s own '
                       'models (read-only)'),
            'flex_price_multiplier': m, 'rel_tol': FLEX_PRICE_READBACK_REL_TOL, 'n_blocks': len(per_block),
            'n_coefficients': n_coef, 'n_bitwise_equal_applied_closed_form': n_exact_applied,
            'n_bitwise_equal_m_times_original': n_exact_ratio, 'max_rel_dev_vs_m_times_original': max_dev,
            'tso_arrays_still_the_planning_arrays': tso_ok,
            'all_match': all_match and tso_ok and bool(per_block), 'per_block': per_block}


def _config_hook_factory(spec, holder, overrides=None, model_variant=None, investment_year=INVESTMENT_YEAR,
                         expected_floor_rows=None, flex_price_multiplier=None, interface_deviation_premium=None,
                         release_solution_bookkeeping=None):
    """pre_solve_hook: verify the case file carries the D oracle configuration
    (same checks as `p515_s43_aa_run._aa_on_pre_solve_hook`), then apply the
    evaluation's overrides (`overrides`; default = the campaign-level
    `spec['configuration']['overrides']`; none for D). Only the AA flag and its
    reject-policy may be overridden (`validate_overrides`); after an AA override
    the frozen memory (5) and regularization (1e-10) are verified unchanged.
    Records into the report's rule_eleven_checklist (provenance).
    Addendum 27 item 1: if the spec declares
    `configuration.case_file_anderson_acceleration`, the loaded AA dict must
    EQUAL that declaration exactly (and carry the frozen memory/regularization)
    in place of the "AA off before overrides" check; without the declaration
    the AA-off check stays, so a spec frozen before Addendum 27 can never run
    AA from the case file.
    Addenda 28-29 (W20): with `model_variant` (validated), AFTER the checks above
    the variant is applied to the evaluation's own shared-ESS data
    (`apply_model_variant`) and READ BACK from probe ESSO models built by
    production's `_build_subproblem` (`model_variant_readback_models`, cohort of
    `investment_year`); any mismatch -- or a change of the SoH-floor row
    identification against `expected_floor_rows` (the baseline probe's, used by
    the floor sidecar) -- raises before any solve. Without it nothing changes.
    Addendum 34 (W33): with `flex_price_multiplier` (validated; 1.0 included), LAST
    of all, the multiplier is applied to the DSO flexibility-price arrays and read
    back from probe DSO blocks (`apply_flex_price_multiplier`); any mismatch raises
    before any solve. Without it (None) nothing changes.
    Addendum 39 (W47): with `interface_deviation_premium` (validated), after the D checks and overrides,
    `planning.params.admm.interface_deviation_premium` is set to {alpha, floor, source} -- the ONLY place row 18's
    premium enters (`_run_operational_planning` threads it to the DSO builders) -- and verified to have taken
    effect; the run's own DSO models are read back post-run (`multiscenario_terminal_capture`). With a declared
    `derived_instance` the planning object's combined scenario checksum is checked against the declaration.
    Without them (None) nothing changes.
    P5.15 Addendum 48 (W90): with `release_solution_bookkeeping` (a bool), after everything above, the switch is set on
    the TSO's and every DSO's solver parameters by `p515_s44_scale_measurement.set_release_solution_bookkeeping` (reads
    it back; raises if it did not take effect) -- the order of the SRP1 (b) gate (applied after the inner hook). It
    changes no model. Without it (None) nothing changes."""
    import p515_g_g1_g4_admm_gates as G
    model_variant = validate_model_variant(model_variant)
    flex_price_multiplier = validate_flex_price_multiplier(flex_price_multiplier)
    premium = validate_interface_deviation_premium(interface_deviation_premium)
    release_bk = validate_release_solution_bookkeeping(release_solution_bookkeeping)
    derived = validate_derived_instance(spec['configuration'].get('derived_instance'))
    if overrides is None:
        overrides = spec['configuration'].get('overrides') or {}
    overrides = validate_overrides(overrides)
    case_file_aa = validate_case_file_anderson_acceleration(
        spec['configuration'].get('case_file_anderson_acceleration'))
    # Addendum 30 (W21): a declared ESS ageing baseline is verified against the loaded parameters (and, without a
    # model variant, read back from probe ESSO models) before any solve; undeclared specs: nothing changes.
    ess_ageing = validate_ess_ageing_baseline(spec['configuration'].get('ess_ageing_baseline'))
    ess_params_pin = spec['configuration'].get('ess_params_file')
    # W84 (Addendum 46 ruling 7): the convergence-depth tail declaration (None = not declared = production OFF).
    tail_declared = validate_convergence_depth_tail(spec['configuration'].get('convergence_depth_tail'))

    def hook(planning, sed, candidate, report):
        a = planning.params.admm
        checks = {
            'rho_v_matches_D': all(float(v) == G.S39_RHO_V for v in a.rho['v'].values()),
            'rho_pf_matches_D': all(float(v) == G.S39_RHO_PF for v in a.rho['pf'].values()),
            'rho_ess_matches_D': all(float(v) == G.S39_RHO_ESS for v in a.rho['ess'].values()),
            'tau_is_0': a.proximal_regularization['tso'].get('tau') == float(G.S39_TAU),
            'gamma_policy_tied_to_rho': a.proximal_regularization['tso'].get('gamma_policy') == 'tied_to_rho',
            'balancing_exempt_until_matches_D': (a.penalty_update.get('balancing_exempt_until')
                                                 == {'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}}),
            'balancing_exempt_channels_empty': not a.penalty_update.get('balancing_exempt_channels'),
            'freeze_after_unchanged_cycles_is_10': (a.penalty_update.get('freeze_after_unchanged_cycles')
                                                    == G.S39_FREEZE_AFTER_UNCHANGED_CYCLES),
            'freeze_backstop_cycle_is_200': a.penalty_update.get('freeze_backstop_cycle') == G.S39_FREEZE_BACKSTOP_CYCLE,
            'minimum_consecutive_converged_cycles_matches_spec': (
                a.minimum_consecutive_converged_cycles == int(spec['required_consecutive_cycles'])),
            'shared_ess_initialization_is_standalone': a.shared_ess_initialization == 'standalone',
            'num_max_iters_is_spec_cap': a.num_max_iters == int(spec['cap']),
        }
        if case_file_aa is None:
            checks['anderson_acceleration_off_before_overrides'] = not a.anderson_acceleration.get('enabled')
        else:
            checks['anderson_acceleration_case_file_matches_declaration'] = (
                a.anderson_acceleration == case_file_aa)
            checks['anderson_acceleration_case_file_memory_regularization_frozen'] = (
                a.anderson_acceleration.get('memory') == FROZEN_AA_MEMORY
                and a.anderson_acceleration.get('regularization') == FROZEN_AA_REGULARIZATION)
        checks['persistent_workers_off'] = not a.persistent_workers.get('enabled')
        checks['parallel_execution_off'] = not planning.parallel_execution
        missing = sorted(k for k, v in checks.items() if not v)
        if missing:
            raise RuntimeError(f'S44 campaign child: configuration not as frozen (case file D + cap): {missing}')
        applied = {}
        for key, value in overrides.items():
            if key not in SUPPORTED_OVERRIDE_KEYS:
                raise RuntimeError(f'unsupported override {key}')
            if key == 'anderson_acceleration':
                merged = dict(a.anderson_acceleration)
                merged.update(value)
                a.anderson_acceleration = merged
                if a.anderson_acceleration != merged:
                    raise RuntimeError('anderson_acceleration override did not take effect')
                if (a.anderson_acceleration.get('memory') != FROZEN_AA_MEMORY
                        or a.anderson_acceleration.get('regularization') != FROZEN_AA_REGULARIZATION):
                    raise RuntimeError(f'anderson_acceleration memory/regularization not at the frozen '
                                       f'{FROZEN_AA_MEMORY}/{FROZEN_AA_REGULARIZATION}: {a.anderson_acceleration}')
                applied[key] = dict(a.anderson_acceleration)
        report.setdefault('rule_eleven_checklist', {})['s44_campaign_configuration_checks'] = checks
        report['rule_eleven_checklist']['s44_campaign_overrides_applied'] = applied
        holder['configuration_checks'] = checks
        holder['overrides_applied'] = applied
        holder['anderson_acceleration_effective'] = dict(a.anderson_acceleration)
        # W84: the tail is enabled ONLY by a declaration; declared or not, what is in force is recorded, and a tail in
        # force that differs from the declaration (e.g. on without one) raises here, before any solve.
        holder['convergence_depth_tail_applied'] = apply_convergence_depth_tail_declaration(a, tail_declared)
        report['rule_eleven_checklist']['w84_convergence_depth_tail'] = holder['convergence_depth_tail_applied']
        if derived is not None:  # W47: the planning object IS the declared derived instance
            got = (getattr(planning, 'scenario_metadata', None) or {}).get('combined_scenario_checksum')
            derived_checks = {'planning_scenario_checksum_equals_declaration': got == derived['scenario_checksum'],
                              'planning_has_more_than_one_scenario_combination': (
                                  planning.num_market_scenarios * planning.transmission_network.num_oper_scenarios
                                  > 1)}
            holder['derived_instance_checks'] = derived_checks
            report['rule_eleven_checklist']['w47_derived_instance'] = {'declared': derived, 'checks': derived_checks,
                                                                        'planning_scenario_checksum': got}
            if not all(derived_checks.values()):
                raise RuntimeError(f'derived_instance: the planning object is not the declared instance: '
                                   f'{derived_checks} (checksum {got})')
        if premium is not None:  # W47: row 18's premium, set before any model is built, verified
            before = dict(a.interface_deviation_premium)
            a.interface_deviation_premium = {'alpha': premium['alpha'], 'floor': premium['floor'],
                                             'source': 'campaign spec entry interface_deviation_premium '
                                                       '(p515_s44_campaign_harness, W47)'}
            after = dict(a.interface_deviation_premium)
            took = after.get('alpha') == premium['alpha'] and after.get('floor') == premium['floor']
            applied_premium = {'declared': premium, 'before': before, 'after': after, 'took_effect': took}
            holder['interface_deviation_premium_applied'] = applied_premium
            report['rule_eleven_checklist']['w47_interface_deviation_premium'] = applied_premium
            if not took:
                raise RuntimeError(f'interface_deviation_premium did not take effect: {applied_premium}')
        if ess_ageing is not None:  # W21: before the model variant (if any) touches the ageing parameters
            import shared_energy_storage_data as SED
            verified = verify_ess_ageing_in_child(sed, ess_ageing, ess_params_pin)
            if model_variant is None:
                probes = {node_id: SED._build_subproblem(sed, node_id)
                          for node_id in sed.active_distribution_network_nodes}
                floor_rows_probe, _floor_counts = G._identify_soh_floor_rows(probes)
                verified['readback_pre_run'] = ess_ageing_readback_models(probes, sed, ess_ageing, investment_year,
                                                                          clone=False)
                del probes
                verified['floor_rows_identical_to_precheck'] = (expected_floor_rows is None
                                                                or floor_rows_probe == expected_floor_rows)
            holder['ess_ageing_verified_pre_run'] = verified
            report['rule_eleven_checklist']['w21_ess_ageing_baseline'] = {
                'declared': ess_ageing, 'label': spec['configuration'].get('ess_ageing_baseline_label'),
                'checks': verified['checks'],
                'readback_all_match': (verified.get('readback_pre_run') or {}).get('all_match'),
                'floor_rows_identical_to_precheck': verified.get('floor_rows_identical_to_precheck')}
            if model_variant is None and not verified['readback_pre_run']['all_match']:
                raise RuntimeError(f'ess_ageing_baseline read-back from the built ESSO model does not match the '
                                   f'declaration: '
                                   f"{ {n: v['checks'] for n, v in verified['readback_pre_run']['per_node'].items()} }")
            if model_variant is None and not verified['floor_rows_identical_to_precheck']:
                raise RuntimeError('ess_ageing_baseline: the probe SoH-floor rows differ from the precheck floor rows')
        if model_variant is not None:
            import shared_energy_storage_data as SED
            applied_mv = apply_model_variant(sed, model_variant)
            probes = {node_id: SED._build_subproblem(sed, node_id)
                      for node_id in sed.active_distribution_network_nodes}
            floor_rows_variant, _floor_counts = G._identify_soh_floor_rows(probes)
            readback = model_variant_readback_models(probes, sed, model_variant, investment_year, clone=False)
            del probes
            floor_rows_ok = expected_floor_rows is None or floor_rows_variant == expected_floor_rows
            holder['model_variant_applied'] = applied_mv
            holder['model_variant_readback_pre_run'] = readback
            report['rule_eleven_checklist']['w20_model_variant'] = {
                'model_variant': model_variant, 'label': MODEL_VARIANT_LABEL,
                'apply_checks': applied_mv['checks'], 'readback_all_match': readback['all_match'],
                'floor_rows_identical_to_baseline_probe': floor_rows_ok}
            if not readback['all_match']:
                raise RuntimeError(f'model_variant read-back from the built ESSO model does not match: '
                                   f"{ {n: v['checks'] for n, v in readback['per_node'].items()} }")
            if not floor_rows_ok:
                raise RuntimeError('model_variant changed the SoH-floor row identification of the ESSO model')
        if flex_price_multiplier is not None:  # W33: last, so every check above ran on the unmodified planning
            applied_fp = apply_flex_price_multiplier(planning, flex_price_multiplier)  # raises on any mismatch
            holder['_flex_price_original_arrays'] = applied_fp.pop('_original_arrays')
            holder['flex_price_applied'] = applied_fp
            report['rule_eleven_checklist']['w33_flex_price_multiplier'] = {
                'flex_price_multiplier': flex_price_multiplier, 'label': applied_fp['label'],
                'checks': applied_fp['checks'], 'readback_all_match': applied_fp['readback_pre_run']['all_match'],
                'n_flex_coefficients': applied_fp['readback_pre_run']['n_flex_coefficients'],
                'max_rel_dev': applied_fp['readback_pre_run']['max_rel_dev']}
        if release_bk is not None:  # W90: after everything above (it changes no model); read back, raises if not in force
            import p515_s44_scale_measurement as S44
            applied_bk = S44.set_release_solution_bookkeeping(planning, release_bk)
            holder['release_solution_bookkeeping_applied'] = applied_bk
            report['rule_eleven_checklist']['w90_release_solution_bookkeeping'] = applied_bk
    return hook


# ==============================================================================
#  the optional POST-CERTIFICATION step (Addendum 25 item 2; P5_15_S44_GATE_RULING.md "Deviation")
# ==============================================================================
POST_CERTIFICATION_FILE = 'post_certification.json'
HULL_BOUND_DETAIL_FILE = 'hull_bound_detail.json'
AA_SIDECAR_FILE = 'aa_per_cycle.jsonl'


def assert_post_certification_capture_paths():
    """Rule eleven for the post-certification step: every function it reuses
    exists with the signature it is called with -- asserted in the child
    BEFORE the run whenever the evaluation requests the step."""
    import inspect
    import p515_s39_evaluate as E39
    import p515_g_g1_g4_admm_gates as G
    import p515_s41_hull_polish as HP
    import p515_s42_exact_fix_rerun as EF
    import p515_s43_aa_run as S43
    checks = {
        'certification_fn': callable(getattr(E39, '_certification_from_trajectory', None)),
        'stopped_by_fn': callable(getattr(G, '_derive_stopped_by_from_trajectory', None)),
        'hull_polish_fn': list(inspect.signature(HP._polish_all_blocks_hull).parameters)
        == ['planning', 'models', 'consensus_vars'],
        'persist_fn': list(inspect.signature(EF._persist_certified_models).parameters) == ['models', 'out_dir'],
        'decomposition_fn_accepts_reference_dir': 'reference_dir' in inspect.signature(
            S43._cost_decomposition_vs_d).parameters,
        'cost_band_constant_1_5e_4': S43.COST_RELATIVE_TOLERANCE == 1.5e-4,
        'reconciliation_tol_1_0': S43.RECONCILIATION_RESIDUAL_ABS_TOL == 1.0,
        'hull_gate_threshold_0_1_pct': HP.GATE_THRESHOLD_PCT == 0.1,
        'aa_sidecar_fn': callable(getattr(S43, '_build_aa_per_cycle_sidecar', None)),
        'run_admm_arm_passes_state_to_hook': ("'state' in inspect.signature(post_run_hook)"
                                              in inspect.getsource(G.run_admm_arm)),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN: post-certification capture paths missing: {missing}')
    return checks


def non_degenerate_hull_counts(hull_bound_detail):
    """Per channel: hull descriptors and ACTIVE descriptors EXCLUDING degenerate
    intervals (Addendum 24 convention, as `p515_s42_hull_counts.py` applied to
    the committed Step 3.5 evidence; `_hull_bounds_active` counts a degenerate
    interval as active by definition)."""
    out = {}
    for d in hull_bound_detail or []:
        c = out.setdefault(d['channel'], {'total': 0, 'degenerate': 0, 'non_degenerate': 0,
                                          'active_non_degenerate': 0})
        c['total'] += 1
        if d['degenerate']:
            c['degenerate'] += 1
        else:
            c['non_degenerate'] += 1
            if d['active']:
                c['active_non_degenerate'] += 1
    return out


def aa_sidecar_summary(rows):
    """Counts over the AA per-cycle fields already in the trajectory."""
    actions = {}
    retained = []
    for r in rows:
        act = r.get('aa_action')
        actions[act] = actions.get(act, 0) + 1
        if isinstance(act, str) and act.startswith('rejected') and (r.get('aa_memory_size_after') or 0) > 0:
            retained.append({'cycle': r.get('cycle'), 'memory_size_before': r.get('aa_memory_size_before'),
                             'memory_size_after': r.get('aa_memory_size_after'),
                             'rho_changed_channels': r.get('aa_rho_changed_channels')})
    return {'n_rows': len(rows), 'action_counts': actions,
            'n_accepted': sum(1 for r in rows if r.get('aa_accepted') is True),
            'n_rejected': sum(v for k, v in actions.items() if isinstance(k, str) and k.startswith('rejected')),
            'rejections_with_memory_retained': retained,
            'first_accept_cycle': next((r.get('cycle') for r in rows if r.get('aa_accepted') is True), None),
            'first_reject_cycle': next((r.get('cycle') for r in rows if isinstance(r.get('aa_action'), str)
                                        and r['aa_action'].startswith('rejected')), None)}


def run_post_certification(*, planning, models, rows, report, state, spec, entry, eval_dir,
                           polish_fn=None, persist_fn=None, decomposition_fn=None):
    """The optional post-certification step, IN the child, inside `run_admm_arm`'s
    post_run_hook (same live models / state, after `write_boyd_terminal_s35ref`
    wrote component_levels_terminal.json). Order as `p515_s43_aa_run.py`:
    certification test -> (b) cost vs reference -> (c) decomposition vs
    reference -> persist certified models (BEFORE the polish mutates them) ->
    (d) interval-hull polish. Skipped cleanly, with the reason recorded, when
    the trajectory is not certified under the spec's own bar. The *_fn
    parameters exist only so the zero-solve checks can substitute fakes for the
    two solving/pickling calls; production callers pass nothing."""
    import p515_s39_evaluate as E39
    import p515_g_g1_g4_admm_gates as G
    import p515_s41_hull_polish as HP
    import p515_s42_exact_fix_rerun as EF
    import p515_s43_aa_run as S43
    polish_fn = polish_fn or HP._polish_all_blocks_hull
    persist_fn = persist_fn or EF._persist_certified_models
    decomposition_fn = decomposition_fn or S43._cost_decomposition_vs_d

    request = entry.get('post_certification')
    cap, required = int(spec['cap']), int(spec['required_consecutive_cycles'])
    out = {'requested': request, 'status': None}
    if not request:
        out.update(status='not_requested', evaluated=False)
        return out, None
    if not rows:
        out.update(status='skipped', evaluated=False, skip_reason='no trajectory')
        return out, None
    cert = E39._certification_from_trajectory(rows, cap, required)
    stopped = G._derive_stopped_by_from_trajectory(rows, cap=cap, required_consecutive=required)
    out['certification'] = cert
    out['stopped_by'] = stopped
    if not cert.get('certified'):
        out.update(status='skipped', evaluated=False, skip_reason=(
            f"trajectory not certified under the spec's bar (cycles_run={cert.get('cycles_run')}, cap={cap}, "
            f"required_consecutive={required}, terminal_consecutive_converged_cycles="
            f"{cert.get('terminal_consecutive_converged_cycles')}, stopped_by={stopped.get('stopped_by')!r}); "
            'post-certification items are never evaluated at an uncertified point'))
        return out, None
    out['evaluated'] = True
    q = report.get('gross_operational_cost')
    out['certified_cost'] = q
    out['objective_convention'] = 'gross_operational_cost (settlement-excluded), as the evaluation record'

    ref = request.get('reference')
    if ref:
        verify_reference_unchanged(ref)
        q_ref = ref['certified_cost']
        abs_tol = S43.COST_RELATIVE_TOLERANCE * q_ref
        abs_diff = abs(q - q_ref) if q is not None else None
        out['gate_b_cost_vs_reference'] = {
            'definition': '|Q - Q_ref| <= 1.5e-4 * Q_ref (p515_s43_aa_run gate (b), reference = the D evaluation '
                          'of the same candidate)',
            'certified_cost': q, 'reference_certified_cost': q_ref, 'reference_eval_dir': ref['eval_dir'],
            'abs_diff': abs_diff, 'relative_tolerance': S43.COST_RELATIVE_TOLERANCE, 'abs_tolerance': abs_tol,
            'pass': bool(abs_diff is not None and abs_diff <= abs_tol)}
        with open(os.path.join(eval_dir, 'component_levels_terminal.json')) as handle:
            my_cl = json.load(handle)
        decomposition = decomposition_fn(my_cl, q, reference_dir=os.path.join(REPO, ref['eval_dir']))
        decomposition['labels_note'] = ("keys named 'D'/'AA' by p515_s43_aa_run: 'D' = the reference evaluation, "
                                        "'AA' = this evaluation")
        out['gate_c_cost_decomposition_vs_reference'] = decomposition
        out['gate_c_pass'] = bool(decomposition.get('reconciles'))
    else:
        out['gate_b_cost_vs_reference'] = None
        out['gate_c_cost_decomposition_vs_reference'] = None
        out['gate_c_pass'] = None

    if request.get('persist_certified_models'):
        out['persisted_models'] = persist_fn(models, eval_dir)
    else:
        out['persisted_models'] = None

    hull_bound_detail = None
    if request.get('hull_polish'):
        if state is None or 'consensus_vars' not in state:
            raise RuntimeError('post-certification hull polish: state/consensus_vars not available')
        t0 = time.time()
        polish, hull_bound_detail = polish_fn(planning, models, state['consensus_vars'])
        polish['runtime_s'] = time.time() - t0
        gate = polish.get('gate')
        n_solved = sum(1 for b in polish.get('per_block') or [] if b.get('solved'))
        out['gate_d_hull_polish'] = {
            'definition': ('p515_s41_hull_polish gate: |sum over blocks of [f_i(polished) - f_i(certified)]| / '
                           'certified cost < 0.1 %, evaluated only when every block solves'),
            'blocks_solved': n_solved, 'n_blocks': polish.get('n_blocks'), 'all_solved': polish.get('all_solved'),
            'failed_blocks': polish.get('failed_blocks'),
            'delta_sum_blocks': gate.get('delta_sum_blocks') if gate else None,
            'relative_pct': gate.get('relative_pct') if gate else None,
            'threshold_pct': HP.GATE_THRESHOLD_PCT,
            'settlement_excluded_change': (gate['reported_not_gated']['gross_operational_cost_change_settlement_excluded']
                                           if gate else None),
            'settlement_remainder_before': (gate['reported_not_gated']['interface_settlement_total_before']
                                            if gate else None),
            'settlement_remainder_after': (gate['reported_not_gated']['interface_settlement_total_after']
                                           if gate else None),
            'hull_bounds_active_by_channel_incl_degenerate': polish.get('hull_bounds_active_by_channel'),
            'hull_bounds_non_degenerate_by_channel': non_degenerate_hull_counts(hull_bound_detail),
            'flagged_blocks': polish.get('flagged_blocks'), 'flag_abs_threshold': polish.get('flag_abs_threshold'),
            'solve_profile': polish.get('solve_profile'), 'runtime_s': polish['runtime_s'],
            'pass': bool(gate is not None and polish.get('all_solved') and gate.get('pass')),
        }
        out['hull_polish_full'] = polish
        path = os.path.join(eval_dir, HULL_BOUND_DETAIL_FILE)
        _write_once_json(path, hull_bound_detail)
        out['hull_bound_detail_path'] = os.path.relpath(path, REPO)
    else:
        out['gate_d_hull_polish'] = None
    out['status'] = 'evaluated'
    return out, hull_bound_detail


def post_certification_summary(pc):
    """The compact block copied into evaluation_record.json (full detail stays in post_certification.json)."""
    if pc is None:
        return None
    s = {k: pc.get(k) for k in ('status', 'evaluated', 'skip_reason', 'error', 'certified_cost',
                                 'gate_c_pass', 'persisted_models', 'hull_bound_detail_path')}
    s['requested'] = pc.get('requested')
    b = pc.get('gate_b_cost_vs_reference')
    s['gate_b'] = ({k: b.get(k) for k in ('abs_diff', 'abs_tolerance', 'reference_certified_cost', 'pass')}
                   if b else None)
    c = pc.get('gate_c_cost_decomposition_vs_reference')
    s['gate_c'] = ({k: c.get(k) for k in ('headline_diff_AA_minus_D', 'dominant_two_diff', 'unaccounted_residual',
                                          'other_priced_components_nonzero', 'reconciles')} if c else None)
    d = pc.get('gate_d_hull_polish')
    s['gate_d'] = ({k: d.get(k) for k in ('blocks_solved', 'n_blocks', 'relative_pct', 'threshold_pct',
                                          'settlement_excluded_change', 'settlement_remainder_before',
                                          'settlement_remainder_after', 'hull_bounds_non_degenerate_by_channel',
                                          'pass')} if d else None)
    return s


# ==============================================================================
#  P5.15 Addendum 39 (W47): the derived instance and the multi-scenario terminal capture -- CHILD SIDE
# ==============================================================================
class _NoStageLog:
    """`p515_s44_scale_measurement.read_planning_from_derived_case` wraps the read in `stages.stage(label)`;
    the campaign child keeps no stage log / watchdog (the parent records wait4 rusage), so this is a no-op
    context manager with the same interface."""

    def stage(self, label):
        from contextlib import nullcontext
        return nullcontext(label)


def install_derived_instance(derived, eval_dir):
    """CHILD SIDE, FIRST (before anything reads the oracle baseline): hash the declared case file, read it with
    the scale harness's own reader (`p515_s44_scale_measurement.read_planning_from_derived_case`: production's
    `SharedResourcesPlanning` on data/SRP1 with the derived file; plots / results / logs of the READ redirected into
    `<eval_dir>/planning_read`), refuse unless the combined scenario checksum equals the declaration, and install
    it as the oracle baseline (`p56a_oracle.install_baseline`, instance label = the declaration's), so every later
    `fresh_planning` of this process -- the floor-row precheck and the run -- deep-copies the derived instance.
    Zero solves. Returns the evidence; raises on any mismatch."""
    import p56a_oracle as O
    import p515_s44_scale_measurement as S
    derived = validate_derived_instance(derived)
    case_abs = os.path.join(REPO, derived['case_path'])
    got = sha256_file(case_abs) if os.path.isfile(case_abs) else None
    if got != derived['case_sha256']:
        raise RuntimeError(f"derived_instance: {derived['case_path']} sha256 {got} != declared {derived['case_sha256']}")
    if os.path.abspath(S.DATA_DIR) != os.path.abspath(os.path.join(REPO, DERIVED_INSTANCE_DATA_DIR_REL)):
        raise RuntimeError(f'derived_instance: the scale harness data dir {S.DATA_DIR} is not '
                           f'{DERIVED_INSTANCE_DATA_DIR_REL}')
    if O._BASELINE is not None:
        raise RuntimeError('derived_instance: an oracle baseline is already installed in this process')
    read_dir = os.path.join(eval_dir, 'planning_read')
    if os.path.exists(read_dir):
        raise RuntimeError(f'refusing to overwrite existing artifact: {read_dir}')
    t0 = time.time()
    planning = S.read_planning_from_derived_case({'derived_case': {'path': derived['case_path']}}, eval_dir,
                                                 _NoStageLog())
    checksum = (planning.scenario_metadata or {}).get('combined_scenario_checksum')
    if checksum != derived['scenario_checksum']:
        raise RuntimeError(f"derived_instance: combined scenario checksum {checksum} != declared "
                           f"{derived['scenario_checksum']}")
    installed = O.install_baseline(planning, checksum, instance_label=derived['instance_label'])
    return {'case_path': derived['case_path'], 'case_sha256_in_child': got, 'scenario_checksum_in_child': checksum,
            'instance_label': installed['instance_label'],
            'checksum_matches_srp1_canonical': installed['checksum_matches_srp1_canonical'],
            'planning_dimensions': S.planning_dimensions(planning),
            'expected_block_counts': S.expected_block_counts(planning),
            'planning_read_dir': os.path.relpath(read_dir, REPO), 'read_wall_s': time.time() - t0,
            'reader': 'p515_s44_scale_measurement.read_planning_from_derived_case (production SharedResourcesPlanning)'}


def _scenario_probability(network, s_m, s_o):
    return network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]


def _rel_diff(a, b, scale=None):
    """|a - b| relative to `scale` (default max(|a|, |b|, 1)); a python float (never a numpy scalar, so every
    comparison built on it is a python bool and serializes as JSON true/false)."""
    if a is None or b is None:
        return None
    denom = max(abs(a), abs(b), 1.0) if scale is None else max(abs(scale), abs(a), abs(b), 1.0)
    return float(abs(a - b) / denom)


def _dispersion_block_metrics(detail, network, model):
    """Addendum 37's metric (production `_get_local_interface_dispersion`) plus the two aggregates of the Planner's
    alpha-sweep ruling, per DSO block, probability-weighted, block-local (MWh per representative day; one period is
    one hour, so a sum over periods of MW is MWh):
        E|d|_p     = sum_s omega_s sum_t |d_{s,t}|             (MWh)
        sum w d2_p = sum_s omega_s sum_t d_{s,t}^2             (MW^2 h)  -- identically rms_mw^2 * n_periods
    and the reactive counterparts. d_{s,t} = p_int_{s,t} - pbar_t, read from production's own per-scenario
    profiles (`per_scenario[...]['d_p_mw' / 'd_q_mvar']`)."""
    probs = {f'{s_m}_{s_o}': _scenario_probability(network, s_m, s_o)
             for s_m in model.scenarios_market for s_o in model.scenarios_operation}
    per = detail['per_scenario']
    e_abs_p = sum(probs[k] * sum(abs(x) for x in v['d_p_mw']) for k, v in per.items())
    e_abs_q = sum(probs[k] * sum(abs(x) for x in v['d_q_mvar']) for k, v in per.items())
    s2_p = sum(probs[k] * sum(x * x for x in v['d_p_mw']) for k, v in per.items())
    s2_q = sum(probs[k] * sum(x * x for x in v['d_q_mvar']) for k, v in per.items())
    n = len(model.periods)
    return {'probabilities': probs, 'E_abs_d_p_mwh': e_abs_p, 'E_abs_d_q_mvarh': e_abs_q,
            'sum_omega_d2_p_mw2h': s2_p, 'sum_omega_d2_q_mvar2h': s2_q,
            'identity_sum_omega_d2_p_equals_rms_sq_times_n_rel_diff': _rel_diff(
                s2_p, detail['p']['rms_mw'] ** 2 * n),
            'identity_sum_omega_d2_q_equals_rms_sq_times_n_rel_diff': _rel_diff(
                s2_q, detail['q']['rms_mvar'] ** 2 * n)}


def _covariance_recomputed(model, network):
    """sum_t baseMVA * Cov_s(pi_t[s_m], p_int[s,t]) for a DSO block (+) or minus the same over the ADN interfaces
    for the TSO block -- recomputed from the model's own Var values and the network's price and probability
    vectors, INDEPENDENTLY of `interface_settlement_deviation` (the form of `p515_s51_row18_zero_solve_checks`
    check H). pbar_t = `model_construction_helpers.expected_market_price`."""
    import pyomo.environ as pe
    import model_construction_helpers as MCH
    scen = [(s_m, s_o) for s_m in model.scenarios_market for s_o in model.scenarios_operation]
    total = 0.0
    if network.is_transmission:
        for dn in model.adn_nodes:
            for p in model.periods:
                pibar = MCH.expected_market_price(network, p)
                e_pi_p = sum(_scenario_probability(network, s_m, s_o) * network.cost_energy_p[s_m][p]
                             * float(pe.value(model.pc_adn[dn, s_m, s_o, p])) for s_m, s_o in scen)
                e_p = sum(_scenario_probability(network, s_m, s_o) * float(pe.value(model.pc_adn[dn, s_m, s_o, p]))
                          for s_m, s_o in scen)
                total -= network.baseMVA * (e_pi_p - pibar * e_p)
    else:
        for p in model.periods:
            pibar = MCH.expected_market_price(network, p)
            e_pi_p = sum(_scenario_probability(network, s_m, s_o) * network.cost_energy_p[s_m][p]
                         * float(pe.value(model.pg_adn[s_m, s_o, p])) for s_m, s_o in scen)
            e_p = sum(_scenario_probability(network, s_m, s_o) * float(pe.value(model.pg_adn[s_m, s_o, p]))
                      for s_m, s_o in scen)
            total += network.baseMVA * (e_pi_p - pibar * e_p)
    return total


def _per_scenario_costs(model, network, params):
    """Per scenario: production's per-scenario base objective and its components
    (`Network.process_results_summary_detail`), plus the two per-scenario terms the base objective does not
    carry -- the interface settlement at the SCENARIO price (the form of
    `model_construction_helpers.interface_energy_settlement`, un-weighted by the settlement weight) and, on a DSO
    block with row 18 wired, the scenario's row 18 charge (the form of `add_scenario_commitment_terms`) -- each
    reconciled to production's block totals (`interface_settlement`, `row18_deviation_charge`)."""
    import pyomo.environ as pe
    detail = network.process_results_summary_detail(model, params)
    base_mva = network.baseMVA
    wired = hasattr(model, 'row18_deviation_charge')
    out, sum_settle, sum_row18, sum_obj = {}, 0.0, 0.0, 0.0
    for s_m in model.scenarios_market:
        c_p = network.cost_energy_p[s_m]
        for s_o in model.scenarios_operation:
            prob = _scenario_probability(network, s_m, s_o)
            if network.is_transmission:
                settle = -sum(c_p[p] * base_mva * float(pe.value(model.pc_adn[dn, s_m, s_o, p]))
                              for dn in model.adn_nodes for p in model.periods)
            else:
                settle = sum(c_p[p] * base_mva * float(pe.value(model.pg_adn[s_m, s_o, p])) for p in model.periods)
            row18 = None
            if wired:
                row18 = sum(float(pe.value(model.row18_alpha)) * float(pe.value(model.row18_premium[p])) * base_mva
                            * float(pe.value(model.row18_dev_p_up[s_m, s_o, p] + model.row18_dev_p_down[s_m, s_o, p]
                                             + model.row18_dev_q_up[s_m, s_o, p] + model.row18_dev_q_down[s_m, s_o, p]))
                            for p in model.periods)
                sum_row18 += prob * row18
            entry = dict(detail['scenarios'][s_m][s_o])
            entry.update({'settlement_at_scenario_price': settle, 'row18_charge_scenario': row18})
            out[f'{s_m}_{s_o}'] = entry
            sum_settle += prob * settle
            sum_obj += prob * entry['obj']
    settlement_model = float(pe.value(model.interface_settlement))
    row18_model = float(pe.value(model.row18_deviation_charge)) if wired else None
    return out, {'expected_base_objective': sum_obj,
                 'expected_settlement_recomputed': sum_settle, 'interface_settlement_model': settlement_model,
                 'settlement_rel_diff': _rel_diff(sum_settle, settlement_model),
                 'expected_row18_charge_recomputed': sum_row18 if wired else None,
                 'row18_deviation_charge_model': row18_model,
                 'row18_rel_diff': _rel_diff(sum_row18, row18_model) if wired else None}


def _shared_ess_schedule(model, network, kind):
    """The scenario-free shared-ESS schedule of a network block (Addendum 38 (B)): read at
    `model_construction_helpers.sess_na_scenario` -- the ONE copy every scenario's balance rows reference; the
    other per-scenario copies are unwired and are NOT read -- beside the coupled expectation Var."""
    import pyomo.environ as pe
    import model_construction_helpers as MCH
    s_m0, s_o0 = MCH.sess_na_scenario(model)
    base_mva = network.baseMVA
    out = {}
    for e in model.shared_energy_storages:
        node_id = network.shared_energy_storages[e].bus
        pnet = [float(pe.value(model.shared_es_pnet[e, s_m0, s_o0, p])) * base_mva for p in model.periods]
        qnet = [float(pe.value(model.shared_es_qnet[e, s_m0, s_o0, p])) * base_mva for p in model.periods]
        soc = [float(pe.value(model.shared_es_soc[e, s_m0, s_o0, p])) * base_mva for p in model.periods]
        if kind == 'TSO':
            exp_p = [float(pe.value(model.expected_shared_ess_p[e, p])) * base_mva for p in model.periods]
            exp_q = [float(pe.value(model.expected_shared_ess_q[e, p])) * base_mva for p in model.periods]
        else:
            exp_p = [float(pe.value(model.expected_shared_ess_p[p])) * base_mva for p in model.periods]
            exp_q = [float(pe.value(model.expected_shared_ess_q[p])) * base_mva for p in model.periods]
        out[str(node_id)] = {'na_scenario': [s_m0, s_o0], 'pnet_mw': pnet, 'qnet_mvar': qnet, 'soc_mwh': soc,
                             'expected_p_mw': exp_p, 'expected_q_mvar': exp_q,
                             'max_abs_na_minus_expected_p_mw': max(abs(a - b) for a, b in zip(pnet, exp_p)),
                             'max_abs_na_minus_expected_q_mvar': max(abs(a - b) for a, b in zip(qnet, exp_q))}
    return out


def _interface_profiles(model, network, kind):
    """Per-scenario interface V / P / Q (production `Network.process_results_interface`) and the block's own
    committed (expected) interface schedule beside it."""
    import pyomo.environ as pe
    base_mva = network.baseMVA
    per = network.process_results_interface(model)
    if kind == 'TSO':
        committed = {}
        for dn in model.adn_nodes:
            node_id = network.active_distribution_network_nodes[dn]
            committed[str(node_id)] = {
                'p_mw': [float(pe.value(model.expected_interface_pf_p[dn, p])) * base_mva for p in model.periods],
                'q_mvar': [float(pe.value(model.expected_interface_pf_q[dn, p])) * base_mva for p in model.periods],
                'v_pu': [float(pe.value(model.expected_interface_vmag[dn, p])) for p in model.periods]}
        per = {str(n): {f'{s_m}_{s_o}': v for s_m, by_o in by_m.items() for s_o, v in by_o.items()}
               for n, by_m in per.items()}
    else:
        committed = {'p_mw': [float(pe.value(model.expected_interface_pf_p[p])) * base_mva for p in model.periods],
                     'q_mvar': [float(pe.value(model.expected_interface_pf_q[p])) * base_mva for p in model.periods],
                     'v_pu': [float(pe.value(model.expected_interface_vmag[p])) for p in model.periods]}
        per = {f'{s_m}_{s_o}': v for s_m, by_o in per.items() for s_o, v in by_o.items()}
    return {'per_scenario': per, 'committed': committed}


def _row18_readback(model, network, alpha_in_force, floor_in_force):
    """Row 18 as BUILT on a DSO block of the run: wired iff alpha > 0 and the block has more than one scenario
    (`add_scenario_commitment_terms`); alpha on the model == the one in force; premium_t == pibar_t
    (`model_construction_helpers.expected_market_price`, the floor applied only when given and binding) --
    exact equality."""
    import pyomo.environ as pe
    import model_construction_helpers as MCH
    n_scen = len(model.scenarios_market) * len(model.scenarios_operation)
    wired = hasattr(model, 'row18_deviation_charge')
    expected_wired = alpha_in_force > 0.0 and n_scen > 1
    out = {'wired': wired, 'expected_wired': expected_wired, 'wired_as_expected': wired == expected_wired}
    if wired:
        expected_premium = []
        for p in model.periods:
            pibar = MCH.expected_market_price(network, p)
            if floor_in_force is not None and pibar < floor_in_force:
                pibar = floor_in_force
            expected_premium.append(pibar)
        premium = [float(pe.value(model.row18_premium[p])) for p in model.periods]
        out.update({'alpha_on_model': float(pe.value(model.row18_alpha)),
                    'alpha_equals_in_force': float(pe.value(model.row18_alpha)) == alpha_in_force,
                    'premium_equals_pibar_exactly': premium == expected_premium,
                    'min_premium': min(premium), 'max_premium': max(premium)})
    out['all_match'] = out['wired_as_expected'] and (not wired or (out['alpha_equals_in_force']
                                                                  and out['premium_equals_pibar_exactly']))
    return out


def sigma_calibration_record(planning, state):
    """The fixed-sigma calibration assertion's inputs and outcome, as production resolved them
    (`shared_resources_planning._resolve_common_admm_objective_scale`: raises unless
    1/F <= sigma_computed / sigma_fixed <= F). A run that reaches the post-run hook passed it."""
    st = state or {}
    fixed, computed = st.get('sigma_fixed'), st.get('sigma_computed')
    factor = planning.params.admm.objective_scale_assert_factor
    ratio = (computed / fixed) if (fixed and computed is not None) else None
    return {'sigma_fixed': fixed, 'sigma_computed': computed, 'ratio_computed_over_fixed': ratio,
            'assert_factor': factor, 'band': [1.0 / factor, factor] if factor else None,
            'ratio_over_lower_edge': (ratio * factor) if (ratio is not None and factor) else None,
            'upper_edge_over_ratio': (factor / ratio) if (ratio and factor) else None,
            'within_band': (ratio is not None and factor is not None and 1.0 / factor <= ratio <= factor),
            'objective_scale_source': planning.params.admm.objective_scale_source,
            'definition': ('sigma_computed = max over TSO/DSO blocks of |w_b * f_b| at the initialization solve '
                           '(shared_resources_planning._compute_common_admm_objective_scale); the resolver raises '
                           'unless 1/F <= sigma_computed/sigma_fixed <= F')}


def multiscenario_terminal_capture(planning, models, state=None, premium=None):
    """W47, ZERO SOLVES: the manuscript's multi-scenario quantities at the TERMINAL point of the run (read with
    `pe.value` off the run's own final models, before any post-certification step mutates them). Every reported
    total is also reconciled against production's own function for it; the identities, their residuals and the
    declared relative tolerance `MULTISCENARIO_IDENTITY_REL_TOL` are recorded. Returns (payload, summary)."""
    import pyomo.environ as pe
    import shared_resources_planning as srp
    admm = planning.params.admm
    alpha_in_force = float(admm.interface_deviation_premium.get('alpha') or 0.0)
    floor_in_force = admm.interface_deviation_premium.get('floor')
    tol = MULTISCENARIO_IDENTITY_REL_TOL
    tn = planning.transmission_network
    blocks = [('TSO', None, tn, year, day) for year in tn.years for day in tn.days]
    for node_id, dn in planning.distribution_networks.items():
        blocks += [('DSO', node_id, dn, year, day) for year in dn.years for day in dn.days]
    dispersion = srp._get_operational_interface_dispersion(planning, models)
    block_components = srp._get_operational_recourse_block_components(planning, models)
    per_block, per_dso = {}, {}
    worst = {'settlement_split_rel': 0.0, 'covariance_vs_deviation_rel': 0.0, 'per_scenario_settlement_rel': 0.0,
             'per_scenario_row18_rel': 0.0, 'per_scenario_q_reconciliation_rel': 0.0,
             'dispersion_identity_rel': 0.0}
    row18_ok, n_row18_wired, weighted = True, 0, {'covariance_recomputed': 0.0}
    for kind, node_id, holder, year, day in blocks:
        model = models['tso'][year][day] if kind == 'TSO' else models['dso'][node_id][year][day]
        network = holder.network[year][day]
        weight = srp._get_admm_block_weight(holder, year, day)
        key = f'{kind}|{node_id}|{year}|{day}' if kind == 'DSO' else f'TSO|{year}|{day}'
        total = srp._get_local_interface_settlement(model, part='total')
        contracted = srp._get_local_interface_settlement(model, part='contracted')
        deviation = srp._get_local_interface_settlement(model, part='deviation')
        s_weight = float(pe.value(model.interface_settlement_weight))
        covariance = s_weight * _covariance_recomputed(model, network)
        costs, cost_checks = _per_scenario_costs(model, network, holder.params)
        # per-block Q reconciliation: production's block recourse (weighted) / weight, against
        #   sum_s omega_s obj_s + (settlement - contracted) + row 18 charge
        local_q = block_components[(kind, node_id, year, day)] / weight
        rebuilt_q = (cost_checks['expected_base_objective'] + s_weight * cost_checks['interface_settlement_model']
                     - contracted + (cost_checks['row18_deviation_charge_model'] or 0.0))
        entry = {
            'kind': kind, 'node_id': node_id, 'year': str(year), 'day': str(day), 'admm_block_weight': weight,
            'n_scenarios': len(model.scenarios_market) * len(model.scenarios_operation),
            # the deviation part is a DIFFERENCE of two settlement-sized sums, so its rounding error scales with
            # the settlement magnitude: the covariance identity is measured relative to that scale
            'settlement': {'total': total, 'contracted': contracted, 'deviation': deviation,
                           'covariance_recomputed': covariance, 'settlement_weight': s_weight,
                           'split_rel_diff': _rel_diff(total - contracted, deviation, scale=max(abs(total),
                                                                                              abs(contracted))),
                           'deviation_vs_covariance_rel_diff': _rel_diff(deviation, covariance,
                                                                         scale=max(abs(total), abs(contracted))),
                           'relative_to': 'max(|total|, |contracted|, |a|, |b|, 1)'},
            'per_scenario_costs': costs, 'per_scenario_cost_checks': cost_checks,
            'block_recourse_production_local': local_q, 'block_recourse_rebuilt_from_scenarios': rebuilt_q,
            'block_recourse_rel_diff': _rel_diff(local_q, rebuilt_q),
            'interface_profiles': _interface_profiles(model, network, kind),
            'shared_ess_schedule': _shared_ess_schedule(model, network, kind),
        }
        worst['settlement_split_rel'] = max(worst['settlement_split_rel'], entry['settlement']['split_rel_diff'])
        worst['covariance_vs_deviation_rel'] = max(worst['covariance_vs_deviation_rel'],
                                                   entry['settlement']['deviation_vs_covariance_rel_diff'])
        worst['per_scenario_settlement_rel'] = max(worst['per_scenario_settlement_rel'],
                                                   cost_checks['settlement_rel_diff'])
        if cost_checks['row18_rel_diff'] is not None:
            worst['per_scenario_row18_rel'] = max(worst['per_scenario_row18_rel'], cost_checks['row18_rel_diff'])
        worst['per_scenario_q_reconciliation_rel'] = max(worst['per_scenario_q_reconciliation_rel'],
                                                         entry['block_recourse_rel_diff'])
        weighted['covariance_recomputed'] += weight * covariance
        if kind == 'DSO':
            detail = dispersion[('DSO', node_id, year, day)]
            if detail is not None:
                metrics = _dispersion_block_metrics(detail, network, model)
                worst['dispersion_identity_rel'] = max(
                    worst['dispersion_identity_rel'],
                    metrics['identity_sum_omega_d2_p_equals_rms_sq_times_n_rel_diff'] or 0.0,
                    metrics['identity_sum_omega_d2_q_equals_rms_sq_times_n_rel_diff'] or 0.0)
                import model_construction_helpers as MCH
                # the prices beside d, so the W46 decomposition (p515_s51_coordinated_decomposition.
                # block_decomposition: market / operation parts, covariance) can be applied post hoc
                entry['dispersion'] = {**{k: v for k, v in detail.items() if k != 'per_scenario'},
                                       'per_scenario_d': detail['per_scenario'], **metrics,
                                       'pi_by_market_by_hour': [[float(network.cost_energy_p[s_m][p])
                                                                 for s_m in model.scenarios_market]
                                                                for p in model.periods],
                                       'pibar_by_hour': [float(MCH.expected_market_price(network, p))
                                                         for p in model.periods]}
                agg = per_dso.setdefault(str(node_id), {
                    'rms_mw_max_over_blocks': 0.0, 'max_abs_mw': 0.0, 'rms_mvar_max_over_blocks': 0.0,
                    'rms_share_of_mean_flow_max_over_blocks': None, 'E_abs_d_p_mwh_sum_over_blocks': 0.0,
                    'E_abs_d_p_mwh_weighted': 0.0, 'sum_omega_d2_p_mw2h_sum_over_blocks': 0.0,
                    'sum_omega_d2_p_mw2h_weighted': 0.0, 'E_abs_d_q_mvarh_sum_over_blocks': 0.0,
                    'row18_charge_sum_over_blocks': 0.0, 'row18_charge_weighted': 0.0,
                    'argmax_rms_block': None, 'n_blocks': 0})
                if detail['p']['rms_mw'] >= agg['rms_mw_max_over_blocks']:
                    agg['argmax_rms_block'] = f'{year}|{day}'
                agg['rms_mw_max_over_blocks'] = max(agg['rms_mw_max_over_blocks'], detail['p']['rms_mw'])
                agg['max_abs_mw'] = max(agg['max_abs_mw'], detail['p']['max_abs_mw'])
                agg['rms_mvar_max_over_blocks'] = max(agg['rms_mvar_max_over_blocks'], detail['q']['rms_mvar'])
                share = detail['p']['rms_share_of_mean_flow']
                if share is not None:
                    prev = agg['rms_share_of_mean_flow_max_over_blocks']
                    agg['rms_share_of_mean_flow_max_over_blocks'] = share if prev is None else max(prev, share)
                agg['E_abs_d_p_mwh_sum_over_blocks'] += metrics['E_abs_d_p_mwh']
                agg['E_abs_d_p_mwh_weighted'] += weight * metrics['E_abs_d_p_mwh']
                agg['sum_omega_d2_p_mw2h_sum_over_blocks'] += metrics['sum_omega_d2_p_mw2h']
                agg['sum_omega_d2_p_mw2h_weighted'] += weight * metrics['sum_omega_d2_p_mw2h']
                agg['E_abs_d_q_mvarh_sum_over_blocks'] += metrics['E_abs_d_q_mvarh']
                agg['row18_charge_sum_over_blocks'] += detail['row18_charge']
                agg['row18_charge_weighted'] += weight * detail['row18_charge']
                agg['n_blocks'] += 1
            readback = _row18_readback(model, network, alpha_in_force, floor_in_force)
            entry['row18_readback'] = readback
            row18_ok = row18_ok and readback['all_match']
            n_row18_wired += int(readback['wired'])
        per_block[key] = entry
    rc = srp._get_operational_recourse_components(planning, models)
    all_dso = {name: sum(v[name] for v in per_dso.values()) for name in (
        'E_abs_d_p_mwh_sum_over_blocks', 'E_abs_d_p_mwh_weighted', 'sum_omega_d2_p_mw2h_sum_over_blocks',
        'sum_omega_d2_p_mw2h_weighted', 'E_abs_d_q_mvarh_sum_over_blocks', 'row18_charge_sum_over_blocks',
        'row18_charge_weighted')}
    all_dso['rms_mw_max_over_all_dso_blocks'] = max([v['rms_mw_max_over_blocks'] for v in per_dso.values()] or [0.0])
    all_dso['max_abs_mw_over_all_dso_blocks'] = max([v['max_abs_mw'] for v in per_dso.values()] or [0.0])
    settlement_identity = {
        'definition': ('Addendum 38 (C): residual := T_TSO + sum_DSO T_DSO (interface_settlement_total) == the '
                       'price-deviation covariance (interface_settlement_deviation_total) + the leftover priced '
                       'interface consensus residual (interface_settlement_identity_residual = the contracted '
                       'parts, which cancel at consensus); the covariance is also recomputed here from the models'),
        'residual_interface_settlement_total': rc.get('interface_settlement_total'),
        'covariance_interface_settlement_deviation_total': rc.get('interface_settlement_deviation_total'),
        'covariance_recomputed_weighted_total': weighted['covariance_recomputed'],
        'covariance_recomputed_vs_production_rel_diff': _rel_diff(
            weighted['covariance_recomputed'], rc.get('interface_settlement_deviation_total'),
            scale=max(abs(rc.get('interface_settlement_total') or 0.0),
                      abs(rc.get('interface_settlement_contracted_total') or 0.0))),
        'leftover_identity_residual': rc.get('interface_settlement_identity_residual'),
        'contracted_total': rc.get('interface_settlement_contracted_total'),
        'tso_deviation_part': rc.get('interface_settlement_deviation_tso'),
        'dso_deviation_part': rc.get('interface_settlement_deviation_dso'),
    }
    checks = {name: bool(value <= tol) for name, value in worst.items()}
    checks['covariance_recomputed_vs_production'] = bool(
        (settlement_identity['covariance_recomputed_vs_production_rel_diff'] or 0.0) <= tol)
    checks['row18_readback_all_blocks'] = bool(row18_ok)
    voltage = {f'{k[0]}|{k[1]}|{k[2]}|{k[3]}': v
               for k, v in srp._get_operational_scenario_voltage_mismatch(planning, models).items()}
    summary = {
        'objective_convention': ('per-scenario obj = production base SMOPF objective per scenario (no settlement, '
                                 'no row 18, no voltage pin); Q per block = sum_s omega_s obj_s + (settlement - '
                                 'contracted) + row 18 charge = gross_operational_cost convention '
                                 '(_get_operational_recourse_block_components / weight)'),
        'dispersion_convention': ('d_{s,t} = p_int_{s,t} - pbar_t (the DSO block\'s own committed schedule); '
                                  'block-local, probability-weighted; *_sum_over_blocks = summed over the DSO\'s '
                                  '(year, day) blocks UNWEIGHTED (MWh per representative day summed, the W44 '
                                  'convention); *_weighted = weighted by _get_admm_block_weight (the Q weighting)'),
        'alpha_in_force': alpha_in_force, 'floor_in_force': floor_in_force,
        'n_blocks': len(per_block), 'n_dso_blocks_row18_wired': n_row18_wired,
        'per_dso': per_dso, 'all_dso': all_dso, 'settlement_identity': settlement_identity,
        'identity_worst_rel_diffs': worst, 'identity_rel_tol': tol, 'checks': checks,
        'all_checks_pass': bool(all(checks.values())),
        'recourse_components': {k: rc.get(k) for k in (
            'gross_operational_cost', 'net_operational_recourse', 'terminal_salvage_value', 'voltage_pin_total',
            'detector_penalty_total')},
        'voltage_pin_mismatch_max_rms_pu': max((v or {}).get('rms_pu', 0.0) for v in voltage.values()),
    }
    payload = {'schema': 'p515_s44_multiscenario_terminal_v1', 'summary': summary, 'blocks': per_block,
               'scenario_voltage_mismatch': voltage}
    return payload, summary


def write_multiscenario_terminal(planning, models, state, eval_dir, premium=None):
    """Writes `MULTISCENARIO_TERMINAL_FILE` (write-once) and returns the summary (+ path, sha256, sigma)."""
    t0 = time.time()
    payload, summary = multiscenario_terminal_capture(planning, models, state=state, premium=premium)
    payload['sigma_calibration'] = sigma_calibration_record(planning, state)
    path = os.path.join(eval_dir, MULTISCENARIO_TERMINAL_FILE)
    _write_once_json(path, payload)
    return {'status': 'written', 'path': os.path.relpath(path, REPO), 'sha256': sha256_file(path),
            'runtime_s': time.time() - t0, 'sigma_calibration': payload['sigma_calibration'], **summary}


# ==============================================================================
#  P5.15 Addendum 40 ruling 1 (W64): the ALPHA-ROW capture. ZERO SOLVES throughout.
#    * activation read-back and the initialisation identity, at the ADMM initialisation / activation point
#      (after the initialisation solve, before any ADMM-cycle solve);
#    * the per-cycle response record (`PER_CYCLE_RESPONSE_FIELDS`), every cycle;
#    * the terminal dual-based curtailment capture and the per-block coordination state.
#  Wired in `_child_real` for derived-instance / premium evaluations only (`capture_multiscenario`); every other
#  evaluation runs exactly as before.
# ==============================================================================
RESPONSE_TERMINAL_FILE = 'response_terminal.json'
PER_CYCLE_RESPONSE_FILE = 'per_cycle_response.jsonl'
ACTIVATION_READBACK_FILE = 'activation_readback.json'
INIT_IDENTITY_FILE = 'initialisation_identity.json'
INIT_IDENTITY_DIR_NAME = 'init_identity'   # <campaign_root>/init_identity/<eval dir name>.json, one per evaluation
# Committed diagnostic scripts whose DEFINITIONS the capture reuses BY IMPORT (never re-implemented): the W44
# coordination record and the W53 curtailment audit's dual helpers. Both arm a SolveProfileGuard at import; it is
# disarmed at once (`import_disarmed_diagnostic`, the p515_s51_2x2_limit_gate / single_block_ab precedent).
COORDINATION_MODULE = 'p515_s51_2x2_limit_gate'
CURTAILMENT_AUDIT_MODULE = 'p515_s53_curtailment_audit'
DECOMPOSITION_MODULE = 'p515_s51_coordinated_decomposition'   # pure json / math, arms nothing
# Sanity (reported here; the W64 smoke gate states its own threshold on it): stationarity of the two row-18 deviation
# Vars of one index gives, whatever the sign convention of the row dual, |zL(d+) + zL(d-)| = 2 * omega_s * alpha *
# pibar_t * baseMVA (the charge's coefficient
# on each Var, in the units of the objective IPOPT solved: p58's rescaled objective = base + scale * AL, and the AL
# terms do not contain d+/-), and |row dual| = |zL(d-) - zL(d+)| / 2.
ROW18_ZL_IDENTITY_REL_TOL = 1e-4


def import_disarmed_diagnostic(name):
    """Import a committed diagnostic script whose import arms a `SolveProfileGuard` as `GUARD`, and disarm it at
    once. Verifies that `OptSolver.solve` and `SystemCallSolver._execute_command` are exactly what they were before
    the import (so no guard of that module stays armed on top of the caller's) and that the module's guard counted
    nothing. A module already imported by this process is returned as is (it was disarmed then)."""
    import importlib
    from pyomo.opt.base.solvers import OptSolver
    from pyomo.opt.solver.shellcmd import SystemCallSolver
    if name in sys.modules:
        return sys.modules[name]
    before = (OptSolver.solve, SystemCallSolver._execute_command)
    cwd = os.getcwd()
    module = importlib.import_module(name)
    os.chdir(cwd)   # the diagnostic scripts chdir to the repo root at import; the child already runs there
    guard = getattr(module, 'GUARD', None)
    if guard is not None:
        guard.uninstall()
    after = (OptSolver.solve, SystemCallSolver._execute_command)
    if after != before:
        raise RuntimeError(f'importing {name} left a solve guard armed (OptSolver.solve / _execute_command changed)')
    if guard is not None and any(guard.counts.values()):
        raise RuntimeError(f'the import-time guard of {name} counted: {guard.counts}')
    return module


def _f(value):
    """A python float (or None) for JSON: pe.value and the network arrays may hand back numpy scalars."""
    return None if value is None else float(value)


def _block_list(planning):
    tn = planning.transmission_network
    blocks = [('TSO', None, tn, year, day) for year in tn.years for day in tn.days]
    for node_id, dn in planning.distribution_networks.items():
        blocks += [('DSO', node_id, dn, year, day) for year in dn.years for day in dn.days]
    return blocks


def _block_key(kind, node_id, year, day):
    return f'DSO|{node_id}|{year}|{day}' if kind == 'DSO' else f'TSO|{year}|{day}'


def _block_result(optimization_results, kind, node_id, year, day):
    try:
        return (optimization_results['tso'][year][day] if kind == 'TSO'
                else optimization_results['dso'][node_id][year][day])
    except (KeyError, TypeError, IndexError):
        return None


def _termination_record(result):
    """The block's LAST solve as production's SolverResults report it (`result.solver` survives
    `_release_solution_bookkeeping`); `succeeded` is production's own `solver_result_succeeded`."""
    from helper_functions import solver_result_succeeded
    if result is None or not hasattr(result, 'solver'):
        return {'available': False, 'status': None, 'termination_condition': None, 'succeeded': False,
                'message': None, 'ipopt_exit': None}
    # W65 (Addendum 40 ruling 1, G13): Pyomo maps BOTH IPOPT success exits ("Optimal Solution Found", converged to
    # `tol`, and "Solved To Acceptable Level", converged only to `acceptable_tol`) to termination_condition optimal;
    # the exit message (the .sol message, `pyomo.opt.plugins.sol`) is what tells them apart, and the G13 primal bound
    # depends on which tolerance the block's last solve met.
    message = getattr(result.solver, 'message', None)
    message = None if message is None else str(message)
    return {'available': True, 'status': str(result.solver.status),
            'termination_condition': str(result.solver.termination_condition),
            'succeeded': bool(solver_result_succeeded(result)),
            'message': message, 'ipopt_exit': ipopt_exit_class(message)}


def ipopt_exit_class(message):
    """'optimal' ("Optimal Solution Found"), 'acceptable' ("Solved To Acceptable Level"), 'other', or None."""
    if message is None:
        return None
    if 'Optimal Solution Found' in message:
        return 'optimal'
    if 'Solved To Acceptable Level' in message:
        return 'acceptable'
    return 'other'


def _curtailed_block_mwh(model, network, params):
    """sum_s omega_s x production's definitional RES curtailment at weight 1 (MWh per representative day):
    `model_construction_helpers.gen_curtailment_definitional_value` -- the same term the curtailment penalty prices."""
    import pyomo.environ as pe
    import model_construction_helpers as MCH
    total = 0.0
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            total += _scenario_probability(network, s_m, s_o) * float(pe.value(
                MCH.gen_curtailment_definitional_value(model, network, s_m, s_o, params, 1.0)))
    return total


def p_posthoc_block(per_scenario_d, probabilities, pibar_by_hour, include_q=True):
    """THE P(alpha) FORMULA on one DSO block (frozen spec v24 `formulas.P_posthoc`), from captured data only:
        P_b = sum_s omega_s sum_t pibar_t * (|d_p[s,t]| + |d_q[s,t]|)        (EUR per representative day)
    with d_p / d_q the per-scenario deviations of production's `_get_local_interface_dispersion` (MW / MVAr, already
    x baseMVA), omega_s the block's scenario probabilities and pibar_t the hourly probability-weighted mean price
    (`model_construction_helpers.expected_market_price`, = row18_premium without a floor). The horizon value is
    sum_b w_b P_b with w_b = admm_block_weight. At alpha > 0 it is the |d| form of charge / alpha (which prices
    d+ + d- >= |d|; equal at an exact minimal split). `include_q=False` gives the P leg alone."""
    total = 0.0
    for key, series in per_scenario_d.items():
        omega = probabilities[key]
        for t, pibar in enumerate(pibar_by_hour):
            dev = abs(series['d_p_mw'][t]) + (abs(series['d_q_mvar'][t]) if include_q else 0.0)
            total += omega * pibar * dev
    return total


def per_cycle_response_record(planning, tso_model, dso_models, results):
    """ONE CYCLE's response fields (`PER_CYCLE_RESPONSE_FIELDS`), read with `pe.value` off the cycle's own models
    right after its Boyd residuals were computed, and its own SolverResults. Zero solves. Formulas:
      E|d|, sum omega d^2, market / operation parts: `p515_s51_coordinated_decomposition.block_decomposition` on
        production's per-scenario d (the W46 definitions), summed over DSO blocks unweighted (W44/W46 convention)
        and weighted by `_get_admm_block_weight` (the Q weighting);
      row 18 charge: w_b x `row18_deviation_charge` (production's `_get_local_interface_dispersion`);
      covariance: w_b x production's settlement DEVIATION part (`_get_local_interface_settlement(part='deviation')`)
        over the DSO blocks -- the price-deviation covariance the DSOs earn (negative = earned);
      curtailed RES: w_b x `_curtailed_block_mwh`, DSO and TSO blocks separately;
      max |d|: max over DSO blocks of production's max_abs_mw;
      non-optimal terminations: blocks (TSO, DSO, ESSO) whose SolverResults fail production's
        `solver_result_succeeded` this cycle."""
    import pyomo.environ as pe  # noqa: F401
    import shared_resources_planning as srp
    import model_construction_helpers as MCH
    DEC = import_disarmed_diagnostic(DECOMPOSITION_MODULE)
    models = {'tso': tso_model, 'dso': dso_models}
    dispersion = srp._get_operational_interface_dispersion(planning, models)
    out = {name: 0.0 for name in (
        'E_abs_d_p_mwh_weighted', 'E_abs_d_p_mwh_unweighted', 'sum_omega_d2_p_mw2h_weighted',
        'sum_omega_d2_p_mw2h_unweighted', 'market_part_mw2h_weighted', 'operation_part_mw2h_weighted',
        'market_part_mw2h_unweighted', 'operation_part_mw2h_unweighted', 'row18_charge_weighted',
        'covariance_dso_weighted', 'curtailed_res_dso_mwh_weighted', 'curtailed_res_tso_mwh_weighted')}
    max_abs = 0.0
    for (kind, node_id, year, day), detail in dispersion.items():
        dn = planning.distribution_networks[node_id]
        network = dn.network[year][day]
        model = dso_models[node_id][year][day]
        weight = srp._get_admm_block_weight(dn, year, day)
        out['covariance_dso_weighted'] += weight * float(srp._get_local_interface_settlement(model, part='deviation'))
        out['curtailed_res_dso_mwh_weighted'] += weight * _curtailed_block_mwh(model, network, dn.params)
        if detail is None:
            continue
        probs = {f'{s_m}_{s_o}': _scenario_probability(network, s_m, s_o)
                 for s_m in model.scenarios_market for s_o in model.scenarios_operation}
        d_by = {k: v['d_p_mw'] for k, v in detail['per_scenario'].items()}
        pi = [[float(network.cost_energy_p[s_m][p]) for s_m in model.scenarios_market] for p in model.periods]
        pibar = [float(MCH.expected_market_price(network, p)) for p in model.periods]
        tot = DEC.block_decomposition(d_by, probs, pi, pibar)['totals']
        for name, src in (('E_abs_d_p_mwh', 'E_abs_d'), ('sum_omega_d2_p_mw2h', 'sum_omega_d2'),
                          ('market_part_mw2h', 'market_part'), ('operation_part_mw2h', 'operation_part')):
            out[f'{name}_unweighted'] += tot[src]
            out[f'{name}_weighted'] += weight * tot[src]
        out['row18_charge_weighted'] += weight * detail['row18_charge']
        max_abs = max(max_abs, detail['p']['max_abs_mw'])
    tn = planning.transmission_network
    for year in tn.years:
        for day in tn.days:
            out['curtailed_res_tso_mwh_weighted'] += (srp._get_admm_block_weight(tn, year, day) * _curtailed_block_mwh(
                tso_model[year][day], tn.network[year][day], tn.params))
    out['max_abs_d_p_mw'] = max_abs
    non_optimal = []
    for kind, node_id, _holder, year, day in _block_list(planning):
        if not _termination_record(_block_result(results, kind, node_id, year, day))['succeeded']:
            non_optimal.append(_block_key(kind, node_id, year, day))
    for node_id in planning.shared_ess_data.active_distribution_network_nodes:
        res = (results.get('esso') or {}).get(node_id) if isinstance(results, dict) else None
        if not _termination_record(res)['succeeded']:
            non_optimal.append(f'ESSO|{node_id}')
    out['n_non_optimal_block_terminations'] = len(non_optimal)
    out['non_optimal_blocks'] = non_optimal
    return out


def activation_readback(planning, tso_model, dso_models, alpha):
    """At ACTIVATION (right after `_prepare_transmission_objectives_for_admm`, i.e. after the initialisation solve,
    `_prepare_distribution_objectives_for_admm` -- which activates row 18 with the settlement weight -- and the TSO's
    counterpart, and BEFORE any ADMM-cycle solve), on EVERY block:
      DSO, alpha > 0: row 18 wired; `row18_alpha` == alpha exactly; every `row18_dev_{p,q}_def` row ACTIVE; no
                      deviation Var of the pair fixed; the row count = 2 x n_scenarios x n_periods;
      DSO, alpha = 0: row 18 ABSENT (no row18_alpha / rows / pair / charge / premium component);
      every block (TSO and DSO): penalty_gen_curtailment == 0 and interface_settlement_weight == 1 exactly;
      TSO: no row 18 structure (Addendum 38 (A)).
    Returns the per-block evidence and `all_ok`."""
    import pyomo.environ as pe
    row18_components = ('row18_alpha', 'row18_premium', 'row18_dev_p_up', 'row18_dev_p_down', 'row18_dev_q_up',
                        'row18_dev_q_down', 'row18_dev_p_def', 'row18_dev_q_def', 'row18_deviation_charge')
    per_block, failing = {}, []
    for kind, node_id, holder, year, day in _block_list(planning):
        model = tso_model[year][day] if kind == 'TSO' else dso_models[node_id][year][day]
        key = _block_key(kind, node_id, year, day)
        present = [c for c in row18_components if hasattr(model, c)]
        rec = {'penalty_gen_curtailment': _f(pe.value(model.penalty_gen_curtailment)),
               'interface_settlement_weight': _f(pe.value(model.interface_settlement_weight)),
               'row18_components_present': present}
        rec['penalty_and_weight_ok'] = (rec['penalty_gen_curtailment'] == 0.0
                                        and rec['interface_settlement_weight'] == 1.0)
        if kind == 'TSO':
            rec['row18_ok'] = not present
        elif alpha > 0.0:
            n_scen = len(model.scenarios_market) * len(model.scenarios_operation)
            expected_rows = 2 * n_scen * len(model.periods)
            wired = len(present) == len(row18_components)
            rec['alpha_on_model'] = _f(pe.value(model.row18_alpha)) if hasattr(model, 'row18_alpha') else None
            n_rows = n_active = n_fixed = 0
            if wired:
                for row_name, up_name, down_name in (('row18_dev_p_def', 'row18_dev_p_up', 'row18_dev_p_down'),
                                                     ('row18_dev_q_def', 'row18_dev_q_up', 'row18_dev_q_down')):
                    row, up, down = getattr(model, row_name), getattr(model, up_name), getattr(model, down_name)
                    for index in row:
                        n_rows += 1
                        n_active += int(row[index].active)
                        n_fixed += int(up[index].fixed) + int(down[index].fixed)
            rec.update({'n_rows': n_rows, 'n_rows_active': n_active, 'n_pair_vars_fixed': n_fixed,
                        'expected_rows': expected_rows})
            rec['row18_ok'] = (wired and rec['alpha_on_model'] == alpha and n_rows == expected_rows
                               and n_active == n_rows and n_fixed == 0)
        else:
            rec['row18_ok'] = not present
        rec['ok'] = rec['penalty_and_weight_ok'] and rec['row18_ok']
        if not rec['ok']:
            failing.append(key)
        per_block[key] = rec
    n_dso = sum(1 for k in per_block if k.startswith('DSO|'))
    expected_dso = sum(len(dn.years) * len(dn.days) for dn in planning.distribution_networks.values())
    return {'alpha': alpha, 'point': ('after _prepare_distribution_objectives_for_admm and '
                                      '_prepare_transmission_objectives_for_admm; before any ADMM-cycle solve'),
            'n_blocks': len(per_block), 'n_dso_blocks': n_dso, 'expected_n_dso_blocks': expected_dso,
            'failing_blocks': failing, 'all_ok': (not failing) and n_dso == expected_dso, 'per_block': per_block}


INIT_IDENTITY_FIELDS = ('gross_operational_cost', 'gross_operational_cost_including_settlement',
                        'interface_settlement_total', 'voltage_pin_total', 'terminal_salvage_value',
                        'net_operational_recourse')


def initialisation_identity_record(planning, tso_model, dso_models, esso_model):
    """The cycle-0 (initialisation) cost: production's `_get_operational_recourse_components` on the models as the
    initialisation solve left them, evaluated BEFORE `_prepare_distribution_objectives_for_admm` (row 18 still
    structurally inactive, settlement weight 0, the build-time curtailment penalty) -- so for one candidate it is a
    function of the initialisation solution only, which the Addendum 40 ruling 2 fix makes identical across alpha
    (the .nl identity). Recorded with float.hex so equality is BITWISE."""
    import shared_resources_planning as srp
    rc = srp._get_operational_recourse_components(planning, {'tso': tso_model, 'dso': dso_models, 'esso': esso_model})
    comps = {k: _f(rc.get(k)) for k in INIT_IDENTITY_FIELDS}
    return {'definition': ('gross_operational_cost per shared_resources_planning._get_operational_recourse_components '
                           'on the initialisation solution, before _prepare_distribution_objectives_for_admm'),
            'gross_operational_cost': comps['gross_operational_cost'],
            'gross_operational_cost_hex': float.hex(comps['gross_operational_cost']),
            'components': comps,
            'components_hex': {k: (float.hex(v) if v is not None else None) for k, v in comps.items()}}


def register_initialisation_identity(record, campaign_root, eval_dir, label, candidate_key, derived_identity):
    """Write this evaluation's initialisation record into `<campaign_root>/init_identity/<eval dir name>.json`
    (atomic, write-once: written under a temporary name and hard-linked into place), then compare it BITWISE with
    every record already there for the SAME candidate on the SAME instance (other evaluations of this campaign --
    concurrent siblings included, since each writes before it reads -- and any reference record the launcher placed
    there). Records of other candidates are listed and not compared. Returns the comparison; the caller refuses to
    continue on a mismatch."""
    directory = os.path.join(campaign_root, INIT_IDENTITY_DIR_NAME)
    os.makedirs(directory, exist_ok=True)
    name = f'{os.path.basename(os.path.normpath(eval_dir))}.json'
    path = os.path.join(directory, name)
    mine = {'label': label, 'eval_dir': os.path.relpath(eval_dir, REPO), 'candidate_key': candidate_key,
            'derived_instance': derived_identity, 'record': record, 'utc': _utc(), 'pid': os.getpid()}
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'x') as handle:
        json.dump(mine, handle, indent=1)
    try:
        os.link(tmp, path)   # raises FileExistsError: write-once
    finally:
        os.remove(tmp)
    compared, other_candidates = [], []
    for fname in sorted(os.listdir(directory)):
        if fname == name or not fname.endswith('.json'):
            continue
        with open(os.path.join(directory, fname)) as handle:
            other = json.load(handle)
        if other.get('candidate_key') != candidate_key or other.get('derived_instance') != derived_identity:
            other_candidates.append({'file': fname, 'label': other.get('label'),
                                     'candidate_key': other.get('candidate_key')})
            continue
        other_hex = (other.get('record') or {}).get('gross_operational_cost_hex')
        compared.append({'file': fname, 'label': other.get('label'), 'gross_operational_cost_hex': other_hex,
                         'equal': other_hex == record['gross_operational_cost_hex'],
                         'components_equal': {k: (other.get('record') or {}).get('components_hex', {}).get(k) == v
                                              for k, v in record['components_hex'].items()}})
    mismatches = [c for c in compared if not c['equal']]
    return {'path': os.path.relpath(path, REPO), 'gross_operational_cost': record['gross_operational_cost'],
            'gross_operational_cost_hex': record['gross_operational_cost_hex'], 'n_compared': len(compared),
            'compared': compared, 'other_candidates_not_compared': other_candidates, 'mismatches': mismatches,
            'all_equal': not mismatches}


class _AlphaRowHookControl:
    def __init__(self):
        self.closed = False
        self.stash = {}

    def close(self):
        """Called by the post-run hook first thing: the run is over, so the wrappers pass through from here on (a
        post-certification step that reached a wrapped function would otherwise be captured as a cycle)."""
        self.closed = True


def alpha_row_run_hooks(eval_dir, label, candidate_key, derived_identity, alpha, holder):
    """Context manager (installed INSIDE the s38 / s39 capture hooks, so it wraps their wrappers and is removed
    first): pass-through wrappers on six production functions of `shared_resources_planning`, each calling the
    original unchanged --
      create_transmission_network_model / create_shared_energy_storage_model: remember the planning object and the
        initialisation TSO / ESSO models;
      _prepare_distribution_objectives_for_admm: BEFORE it runs, the initialisation identity record;
      _prepare_transmission_objectives_for_admm: AFTER it runs, the activation read-back (ACTIVATION_READBACK_FILE)
        and the initialisation identity registered and compared (INIT_IDENTITY_FILE); either failing RAISES, so the
        evaluation stops before its first ADMM-cycle solve;
      get_admm_boyd_residual_metrics: remember the cycle's models (called once per cycle, after the ESSO solve);
      _admm_local_solves_succeeded: its first call is the initialisation check (passed through); every later call
        (once per cycle, right after the Boyd residuals) appends that cycle's `per_cycle_response_record` to
        PER_CYCLE_RESPONSE_FILE. A per-cycle capture error is recorded in the line, never raised.
    """
    from contextlib import contextmanager
    import resource as _resource
    import shared_resources_planning as srp

    names = ('create_transmission_network_model', 'create_shared_energy_storage_model',
             '_prepare_distribution_objectives_for_admm', '_prepare_transmission_objectives_for_admm',
             'get_admm_boyd_residual_metrics', '_admm_local_solves_succeeded')
    campaign_root = os.path.dirname(os.path.dirname(os.path.abspath(eval_dir)))
    sidecar = os.path.join(eval_dir, PER_CYCLE_RESPONSE_FILE)

    @contextmanager
    def _cm():
        control = _AlphaRowHookControl()
        st = control.stash
        st.update({'planning': None, 'tso': None, 'esso': None, 'dso': None, 'boyd_calls': 0, 'local_calls': 0,
                   'cycle_models': None, 't_last': None})
        originals = {n: getattr(srp, n) for n in names}
        try:
            import psutil
            proc = psutil.Process()
        except Exception:  # noqa: BLE001 -- RSS then recorded as None
            proc = None

        def w_create_tso(planning_problem, *args, **kwargs):
            out = originals['create_transmission_network_model'](planning_problem, *args, **kwargs)
            if not control.closed:
                st['planning'], st['tso'] = planning_problem, out[0]
            return out

        def w_create_esso(*args, **kwargs):
            out = originals['create_shared_energy_storage_model'](*args, **kwargs)
            if not control.closed:
                st['esso'] = out[0]
            return out

        def w_prep_dso(distribution_networks, models):
            if not control.closed:
                st['dso'] = models
                holder['initialisation_identity_record'] = initialisation_identity_record(
                    st['planning'], st['tso'], models, st['esso'])
            return originals['_prepare_distribution_objectives_for_admm'](distribution_networks, models)

        def w_prep_tso(transmission_network, model):
            out = originals['_prepare_transmission_objectives_for_admm'](transmission_network, model)
            if control.closed:
                return out
            readback = activation_readback(st['planning'], model, st['dso'], alpha)
            _write_once_json(os.path.join(eval_dir, ACTIVATION_READBACK_FILE), readback)
            holder['activation_readback'] = {k: v for k, v in readback.items() if k != 'per_block'}
            ident = register_initialisation_identity(holder['initialisation_identity_record'], campaign_root,
                                                     eval_dir, label, candidate_key, derived_identity)
            _write_once_json(os.path.join(eval_dir, INIT_IDENTITY_FILE),
                             {'record': holder['initialisation_identity_record'], 'comparison': ident})
            holder['initialisation_identity'] = ident
            print(f"[S44-CHILD] activation read-back all_ok={readback['all_ok']} (failing {readback['failing_blocks']}); "
                  f"initialisation gross {ident['gross_operational_cost']!r} ({ident['gross_operational_cost_hex']}) "
                  f"compared with {ident['n_compared']} record(s), all_equal={ident['all_equal']}", flush=True)
            if not readback['all_ok']:
                raise RuntimeError(f"ACTIVATION READ-BACK FAILED before the first ADMM cycle: failing blocks "
                                   f"{readback['failing_blocks']} (n_dso {readback['n_dso_blocks']} / expected "
                                   f"{readback['expected_n_dso_blocks']})")
            if not ident['all_equal']:
                raise RuntimeError(f"INITIALISATION IDENTITY FAILED before the first ADMM cycle: this evaluation's "
                                   f"initialisation gross cost {ident['gross_operational_cost_hex']} differs from "
                                   f"{ident['mismatches']}")
            st['t_last'] = time.time()
            return out

        def w_boyd(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_parameters):
            result = originals['get_admm_boyd_residual_metrics'](planning_problem, tso_model, dso_models, esso_model,
                                                                 consensus_vars, dual_vars, admm_parameters)
            if not control.closed:
                st['boyd_calls'] += 1
                st['cycle_models'] = (tso_model, dso_models)
            return result

        def w_local(planning_problem, results):
            ok = originals['_admm_local_solves_succeeded'](planning_problem, results)
            if control.closed:
                return ok
            n = st['local_calls']
            st['local_calls'] += 1
            if n == 0:
                return ok   # the initialisation check, before activation
            t_start = time.time()
            rec = {'response_cycle': n, 'boyd_calls_so_far': st['boyd_calls'],
                   'cycle_wall_s': (t_start - st['t_last']) if st['t_last'] is not None else None}
            try:
                if st['cycle_models'] is None or st['boyd_calls'] != n:
                    raise RuntimeError(f"cycle bookkeeping: {st['boyd_calls']} Boyd calls at local-solve call {n}")
                rec.update(per_cycle_response_record(planning_problem, st['cycle_models'][0], st['cycle_models'][1],
                                                     results))
                rec['response_captured'] = True
                rec['response_capture_error'] = None
            except Exception as error:  # noqa: BLE001 -- a capture defect must not destroy the run
                rec['response_captured'] = False
                rec['response_capture_error'] = f'{type(error).__name__}: {error}'
            rec['response_capture_s'] = time.time() - t_start
            rec['rss_bytes'] = proc.memory_info().rss if proc is not None else None
            rec['ru_maxrss_bytes'] = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            with open(sidecar, 'a') as handle:
                handle.write(json.dumps(rec, default=_json_default) + '\n')
                handle.flush()
            st['t_last'] = time.time()
            return ok

        wrappers = {'create_transmission_network_model': w_create_tso,
                    'create_shared_energy_storage_model': w_create_esso,
                    '_prepare_distribution_objectives_for_admm': w_prep_dso,
                    '_prepare_transmission_objectives_for_admm': w_prep_tso,
                    'get_admm_boyd_residual_metrics': w_boyd, '_admm_local_solves_succeeded': w_local}
        for n, w in wrappers.items():
            setattr(srp, n, w)
        try:
            yield control
        finally:
            for n, original in originals.items():
                setattr(srp, n, original)
    return _cm()


def read_per_cycle_response(eval_dir):
    """{cycle: record} from PER_CYCLE_RESPONSE_FILE (empty when absent)."""
    path = os.path.join(eval_dir, PER_CYCLE_RESPONSE_FILE)
    out = {}
    if os.path.isfile(path):
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    out[int(rec['response_cycle'])] = rec
    return out


def _flexibility_block(model, network, params):
    """Omega-weighted DSO flexibility legs (MWh per representative day): P up / P down / Q up / Q down over the
    fl_reg loads -- the legs of production's `network._compute_flexibility_used` (up + down), which is recomputed
    beside them as the reconciliation."""
    import pyomo.environ as pe
    import network as network_module
    legs = {'p_up_mwh': 0.0, 'p_down_mwh': 0.0, 'q_up_mvarh': 0.0, 'q_down_mvarh': 0.0}
    if params.fl_reg:
        base = network.baseMVA
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                omega = _scenario_probability(network, s_m, s_o)
                for c in model.loads:
                    if network.loads[c].fl_reg:
                        for p in model.periods:
                            legs['p_up_mwh'] += omega * float(pe.value(model.flex_p_up[c, s_m, s_o, p])) * base
                            legs['p_down_mwh'] += omega * float(pe.value(model.flex_p_down[c, s_m, s_o, p])) * base
                            legs['q_up_mvarh'] += omega * float(pe.value(model.flex_q_up[c, s_m, s_o, p])) * base
                            legs['q_down_mvarh'] += omega * float(pe.value(model.flex_q_down[c, s_m, s_o, p])) * base
    used = network_module._compute_flexibility_used(network, model, params)
    legs['production_flexibility_used'] = {'p': _f(used['p']), 'q': _f(used['q'])}
    legs['reconciliation_rel_diff_p'] = _rel_diff(legs['p_up_mwh'] + legs['p_down_mwh'], used['p'])
    return legs


def _row18_legs_block(model, network):
    """The row 18 charge split into its P and Q legs (EUR per representative day), the same form as
    `add_scenario_commitment_terms`, reconciled to `row18_deviation_charge`; plus the zL sanity identity
    (ROW18_ZL_IDENTITY_REL_TOL, reported). None where row 18 is not wired."""
    import pyomo.environ as pe
    if not hasattr(model, 'row18_deviation_charge'):
        return None
    alpha = float(pe.value(model.row18_alpha))
    base = network.baseMVA
    legs = {'charge_p': 0.0, 'charge_q': 0.0}
    worst_zl, worst_dual, n_checked, n_missing = 0.0, 0.0, 0, 0
    n_le = {'n_le_1e-6': 0, 'n_le_1e-4': 0, 'n_le_1e-2': 0}
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            omega = _scenario_probability(network, s_m, s_o)
            for p in model.periods:
                coef = omega * alpha * float(pe.value(model.row18_premium[p])) * base
                legs['charge_p'] += coef * float(pe.value(model.row18_dev_p_up[s_m, s_o, p] + model.row18_dev_p_down[s_m, s_o, p]))
                legs['charge_q'] += coef * float(pe.value(model.row18_dev_q_up[s_m, s_o, p] + model.row18_dev_q_down[s_m, s_o, p]))
                for row_name, up_name, down_name in (('row18_dev_p_def', 'row18_dev_p_up', 'row18_dev_p_down'),
                                                     ('row18_dev_q_def', 'row18_dev_q_up', 'row18_dev_q_down')):
                    zl_up = model.ipopt_zL_out.get(getattr(model, up_name)[s_m, s_o, p])
                    zl_dn = model.ipopt_zL_out.get(getattr(model, down_name)[s_m, s_o, p])
                    lam = model.dual.get(getattr(model, row_name)[s_m, s_o, p])
                    if zl_up is None or zl_dn is None or lam is None:
                        n_missing += 1
                        continue
                    n_checked += 1
                    scale = max(abs(2.0 * coef), 1e-12)
                    # |.| of the sum: independent of the sign convention the suffix reports bound multipliers in
                    rel_zl = abs(abs(zl_up + zl_dn) - 2.0 * coef) / scale
                    worst_zl = max(worst_zl, rel_zl)
                    worst_dual = max(worst_dual, abs(abs(lam) - abs(zl_dn - zl_up) / 2.0) / scale)
                    for name, bound in (('n_le_1e-6', 1e-6), ('n_le_1e-4', 1e-4), ('n_le_1e-2', 1e-2)):
                        n_le[name] += int(rel_zl <= bound)
    model_charge = float(pe.value(model.row18_deviation_charge))
    legs.update({'charge_model': model_charge,
                 'legs_vs_model_rel_diff': _rel_diff(legs['charge_p'] + legs['charge_q'], model_charge),
                 'zl_identity': {'n_checked': n_checked, 'n_missing_suffix_values': n_missing, **n_le,
                                 'worst_rel_zl_sum_vs_2coef': worst_zl, 'worst_rel_abs_dual_vs_half_zl_diff': worst_dual,
                                 'tol': ROW18_ZL_IDENTITY_REL_TOL,
                                 'holds': bool(n_checked > 0 and worst_zl <= ROW18_ZL_IDENTITY_REL_TOL
                                               and worst_dual <= ROW18_ZL_IDENTITY_REL_TOL)}})
    return legs


# W65 (Addendum 40 ruling 1, G13): IPOPT 3.14.18 defaults (`/usr/local/bin/ipopt --print-options`) for the options the
# DSO case files leave unset; the G13 primal bound reads the case-file value where one is set and these otherwise.
IPOPT_DEFAULTS_3_14_18 = {'tol': 1e-08, 'acceptable_tol': 1e-06, 'constr_viol_tol': 1e-04,
                          'acceptable_constr_viol_tol': 1e-02, 'bound_relax_factor': 1e-08,
                          'honor_original_bounds': 'no', 'nlp_scaling_method': 'gradient-based',
                          'nlp_scaling_max_gradient': 100.0}


def ipopt_options_in_force(params):
    """The block family's configured IPOPT options (`params.solver_params.options`, the case file) relevant to the
    G13 bound, each with its source ('case_file' or 'ipopt_default'). A retry's option_overrides are NOT reflected
    (the G13 gate requires zero retries)."""
    configured = dict(getattr(getattr(params, 'solver_params', None), 'options', None) or {})
    out = {}
    for name, default in IPOPT_DEFAULTS_3_14_18.items():
        if name in configured:
            out[name] = {'value': configured[name], 'source': 'case_file'}
        else:
            out[name] = {'value': default, 'source': 'ipopt_default'}
    return out


def _row18_primal_split_block(model, network):
    """W65 (G13), ZERO SOLVES: the REALIZED decomposition of the gap between the charge form and the |d| form of
    P on one DSO block, per unit alpha (EUR per representative day, unweighted), both legs:
        gap_b = sum_{s,t} omega_s pibar_t B [ (d+ + d-) - |d| ],   d = pg_adn - expected_interface_pf_p (P leg; the
                Q leg likewise) -- the SAME d `_get_local_interface_dispersion` reports, so sum_b w_b gap_b =
                P_charge - P_posthoc up to summation order;
    with r = d - (d+ - d-) the row-18 defining-row residual, and per index
        (d+ + d-) - |d| in [ 2 min(d+, d-) - |r|,  2 min(d+, d-) + |r| ],
    so |gap_b - split_b| <= residual_b, where split_b = sum omega pibar B 2 min(d+, d-) (its NEGATIVE part is the
    bound violation of the returned point -- honor_original_bounds = no, bound_relax_factor; its POSITIVE part is
    incomplete complementarity) and residual_b = sum omega pibar B |r|. bound_weight_b = sum_{s,t} omega_s pibar_t B
    (ONE leg), the multiplier of the per-index tolerance in the G13 bound. Maxima in per unit. None where row 18 is not
    wired."""
    import pyomo.environ as pe
    if not hasattr(model, 'row18_deviation_charge'):
        return None
    base = network.baseMVA
    acc = {'gap': 0.0, 'split': 0.0, 'split_negative': 0.0, 'split_positive': 0.0, 'residual': 0.0,
           'bound_weight': 0.0}
    mx = {'max_abs_row_residual_pu_p': 0.0, 'max_abs_row_residual_pu_q': 0.0, 'max_bound_violation_pu': 0.0,
          'max_min_split_pu': 0.0}
    legs = (('pg_adn', 'expected_interface_pf_p', 'row18_dev_p_up', 'row18_dev_p_down', 'max_abs_row_residual_pu_p'),
            ('qg_adn', 'expected_interface_pf_q', 'row18_dev_q_up', 'row18_dev_q_down', 'max_abs_row_residual_pu_q'))
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            omega = _scenario_probability(network, s_m, s_o)
            for p in model.periods:
                w = omega * float(pe.value(model.row18_premium[p])) * base
                acc['bound_weight'] += w
                for flow, expected, up_name, down_name, rkey in legs:
                    d = float(pe.value(getattr(model, flow)[s_m, s_o, p])) - float(pe.value(getattr(model, expected)[p]))
                    up = float(pe.value(getattr(model, up_name)[s_m, s_o, p]))
                    dn = float(pe.value(getattr(model, down_name)[s_m, s_o, p]))
                    r = d - (up - dn)
                    low = min(up, dn)
                    acc['gap'] += w * ((up + dn) - abs(d))
                    acc['split'] += w * 2.0 * low
                    acc['split_negative'] += w * 2.0 * max(0.0, -low)
                    acc['split_positive'] += w * 2.0 * max(0.0, low)
                    acc['residual'] += w * abs(r)
                    mx[rkey] = max(mx[rkey], abs(r))
                    mx['max_bound_violation_pu'] = max(mx['max_bound_violation_pu'], -min(up, dn, 0.0))
                    mx['max_min_split_pu'] = max(mx['max_min_split_pu'], low)
    acc['identity_abs_gap_minus_split_le_residual'] = bool(abs(acc['gap'] - acc['split']) <= acc['residual'] * (1.0 + 1e-9)
                                                           + 1e-12)
    return {**acc, **mx, 'units': 'EUR per unit alpha per representative day (unweighted); maxima per unit'}


def _node_index_of_bus(network, bus_id):
    for i, node in enumerate(network.nodes):
        if node.bus_i == bus_id:
            return i
    return None


def _curtailment_block(kind, node_id, year, day, model, network, params, weight, termination, CA):
    """Per (network, hour, scenario) curtailed RES volume and its value at the scenario's market price, and the
    DUAL-BASED entries of every curtaillable generator-hour-scenario above tolerance (the W53 audit's
    `analyse_srp1_models` capture, generalized to many scenarios, with its helpers and tolerances BY IMPORT).
    Units: MW per 1 h period = MWh; B = baseMVA; duals raw (the units of the objective IPOPT solved -- p58's rescaled
    objective = base + scale x AL, base = sum_s omega_s cost_s in EUR per representative day) with the EUR/MWh
    reading lmp = dual / (B x omega_s) beside them (sign as Pyomo's `dual` suffix returns it)."""
    import math
    import pyomo.environ as pe
    base = network.baseMVA
    tol = CA.TOL_FACTOR * base
    gens = ([g for g in model.generators if network.generators[g].is_curtaillable()]
            if (params.rg_curt and hasattr(model, 'pg_avail')) else [])
    gbus = CA._gen_bus(model) if gens else {}
    ref_bus = _node_index_of_bus(network, network.get_reference_node_id())
    row18 = hasattr(model, 'row18_dev_p_def')
    has_cap = hasattr(model, 'sg_capability')
    key = _block_key(kind, node_id, year, day)
    scenarios, entries, rows_by_hour = {}, [], {}
    e_net = e_plus = priced = 0.0
    for s_m in model.scenarios_market:
        price = [float(network.cost_energy_p[s_m][p]) for p in model.periods]
        for s_o in model.scenarios_operation:
            omega = _scenario_probability(network, s_m, s_o)
            v_net, v_plus = [0.0] * len(price), [0.0] * len(price)
            for ti, t in enumerate(model.periods):
                lmp_ref = None
                for g in gens:
                    var = model.pg[g, s_m, s_o, t]
                    av = float(pe.value(model.pg_avail[g, s_o, t]))
                    pg = float(pe.value(var))
                    c = (av - pg) * base
                    v_net[ti] += c
                    v_plus[ti] += max(c, 0.0)
                    if c <= tol:
                        continue
                    cap = model.sg_capability[g, s_m, s_o, t] if (has_cap and (g, s_m, s_o, t) in model.sg_capability) else None
                    cap_slack = (float(pe.value(cap.upper) - pe.value(cap.body))) if cap is not None else None
                    bus = gbus.get(g)
                    lam_bus = model.dual.get(model.node_balance_p[bus, s_m, s_o, t]) if bus is not None else None
                    if lmp_ref is None and ref_bus is not None:
                        lmp_ref = model.dual.get(model.node_balance_p[ref_bus, s_m, s_o, t])
                    entry = {
                        'block': key, 'network': 'TSO' if kind == 'TSO' else f'DSO{node_id}', 'year': str(year),
                        'day': str(day), 'scenario': f'{s_m}_{s_o}', 'omega': omega, 'hour': ti, 'gen': g,
                        'bus_index': bus, 'c_mw': c, 'pg_mw': pg * base, 'pg_avail_mw': av * base,
                        'qg_mvar': _f(pe.value(model.qg[g, s_m, s_o, t])) * base,
                        'sg_avail_mva': _f(pe.value(model.sg_avail[g, s_o, t])) * base,
                        'sg_mva': (math.sqrt(max(float(pe.value(model.sg_sqr[g, s_m, s_o, t])), 0.0)) * base
                                   if hasattr(model, 'sg_sqr') else None),
                        'sg_capability_slack_pu2': cap_slack,
                        'sg_capability_dual_raw': _f(model.dual.get(cap)) if cap is not None else None,
                        'class': ('capability_bound' if (cap is not None and cap_slack <= CA.CAP_SLACK_TOL)
                                  else 'interior'),
                        'pg_zU_raw': _f(model.ipopt_zU_out.get(var)),
                        'price_scenario_eur_mwh': price[ti],
                        'lmp_bus_dual_raw': _f(lam_bus), 'lmp_ref_bus_dual_raw': _f(lmp_ref),
                        'lmp_bus_eur_mwh': (float(lam_bus) / (base * omega)) if lam_bus is not None else None,
                        'lmp_ref_bus_eur_mwh': (float(lmp_ref) / (base * omega)) if lmp_ref is not None else None,
                        'ref_bus_index': ref_bus, 'termination_last_solve': termination,
                    }
                    if kind == 'DSO':
                        entry['d_p_mw'] = (float(pe.value(model.pg_adn[s_m, s_o, t]))
                                           - float(pe.value(model.expected_interface_pf_p[t]))) * base
                    if row18:
                        idx = (s_m, s_o, t)
                        entry.update({
                            'd_up_mw': float(pe.value(model.row18_dev_p_up[idx])) * base,
                            'd_down_mw': float(pe.value(model.row18_dev_p_down[idx])) * base,
                            'row18_dev_p_def_dual_raw': _f(model.dual.get(model.row18_dev_p_def[idx])),
                            'zL_d_up_raw': _f(model.ipopt_zL_out.get(model.row18_dev_p_up[idx])),
                            'zL_d_down_raw': _f(model.ipopt_zL_out.get(model.row18_dev_p_down[idx])),
                            'row18_premium_eur_mwh': float(pe.value(model.row18_premium[t])),
                            'row18_alpha': float(pe.value(model.row18_alpha))})
                    hour_key = f'{key}|{s_m}_{s_o}|{ti}'
                    if hour_key not in rows_by_hour:
                        rows = CA._network_hour_rows(model, s_m, s_o, t, ())
                        for fam_rows in rows.values():
                            for r in fam_rows:
                                r.pop('is_transformer', None)   # W53 caveat: not computed here (no transformer set)
                        rows_by_hour[hour_key] = rows
                    entry['network_hour_rows_key'] = hour_key
                    entry['voltage_bound_active'] = bool(rows_by_hour[hour_key]['voltage'])
                    entry['branch_limit_active'] = bool(rows_by_hour[hour_key]['branch'])
                    entries.append(entry)
            priced_s = sum(pr * v for pr, v in zip(price, v_plus))
            scenarios[f'{s_m}_{s_o}'] = {'omega': omega, 'V_net_mw': v_net, 'V_plus_mw': v_plus, 'price_eur_mwh': price,
                                         'priced_V_plus_eur': priced_s}
            e_net += omega * sum(v_net)
            e_plus += omega * sum(v_plus)
            priced += omega * priced_s
    production = _curtailed_block_mwh(model, network, params)
    block = {'kind': kind, 'node_id': node_id, 'year': str(year), 'day': str(day), 'admm_block_weight': weight,
             'baseMVA': base, 'tol_mw': tol, 'curtaillable_gens': gens, 'E_net_mwh': e_net, 'E_plus_mwh': e_plus,
             'priced_eur': priced, 'production_definitional_mwh': production,
             'reconciliation_abs_diff_mwh': abs(e_net - production), 'scenarios': scenarios,
             'termination_last_solve': termination}
    return block, entries, rows_by_hour


def response_terminal_capture(planning, models, optimization_results=None):
    """W64, ZERO SOLVES, on the run's own TERMINAL models (before the workbook and any post-certification step):
    (1) the dual-based curtailment capture, (2) the per-DSO-block coordination state (the W44 `coordination_record`,
    BY IMPORT: dual_pf_p/q_req, rho_pf, the TSO request, pbar, the effective objective scale, lambda_AL, penalties,
    settlement parts, per-scenario curtailment), (3) DSO flexibility legs, (4) the row 18 charge by leg (P / Q) with
    the zL sanity identity, (5) the P(alpha) |d| form per block (`p_posthoc_block`), (6) the voltage pin and the
    recourse components. Returns (payload, summary)."""
    import shared_resources_planning as srp
    LG = import_disarmed_diagnostic(COORDINATION_MODULE)
    CA = import_disarmed_diagnostic(CURTAILMENT_AUDIT_MODULE)
    admm = planning.params.admm
    alpha = float(admm.interface_deviation_premium.get('alpha') or 0.0)
    dispersion = srp._get_operational_interface_dispersion(planning, models)
    rc = srp._get_operational_recourse_components(planning, models)
    curtail_blocks, entries, rows_by_hour = {}, [], {}
    coordination, flexibility, row18_legs, p_posthoc = {}, {}, {}, {}
    tot = {'charge_p_weighted': 0.0, 'charge_q_weighted': 0.0, 'P_posthoc_weighted': 0.0,
           'P_posthoc_p_leg_weighted': 0.0}
    # W65 (G13): the realized (d+ + d-) - |d| decomposition, weighted, and the per-DSO IPOPT options in force
    split_tot = {k: 0.0 for k in ('gap', 'split', 'split_negative', 'split_positive', 'residual', 'bound_weight')}
    split_max = {k: 0.0 for k in ('max_abs_row_residual_pu_p', 'max_abs_row_residual_pu_q', 'max_bound_violation_pu',
                                  'max_min_split_pu')}
    split_identity_all = True
    ipopt_options_dso, dso_exit_counts = {}, {}
    per_network = {}
    zl = {'n_checked': 0, 'n_missing_suffix_values': 0, 'n_le_1e-6': 0, 'n_le_1e-4': 0, 'n_le_1e-2': 0,
          'worst_rel_zl_sum_vs_2coef': 0.0, 'worst_rel_abs_dual_vs_half_zl_diff': 0.0}
    for kind, node_id, holder, year, day in _block_list(planning):
        model = models['tso'][year][day] if kind == 'TSO' else models['dso'][node_id][year][day]
        network = holder.network[year][day]
        weight = srp._get_admm_block_weight(holder, year, day)
        key = _block_key(kind, node_id, year, day)
        term = _termination_record(_block_result(optimization_results, kind, node_id, year, day))
        blk, ents, rows = _curtailment_block(kind, node_id, year, day, model, network, holder.params, weight, term, CA)
        curtail_blocks[key] = blk
        entries += ents
        rows_by_hour.update(rows)
        net_name = 'TSO' if kind == 'TSO' else f'DSO{node_id}'
        agg = per_network.setdefault(net_name, {'E_plus_mwh_weighted': 0.0, 'E_net_mwh_weighted': 0.0,
                                                'priced_eur_weighted': 0.0, 'n_entries_above_tol': 0})
        agg['E_plus_mwh_weighted'] += weight * blk['E_plus_mwh']
        agg['E_net_mwh_weighted'] += weight * blk['E_net_mwh']
        agg['priced_eur_weighted'] += weight * blk['priced_eur']
        agg['n_entries_above_tol'] += len(ents)
        if kind != 'DSO':
            continue
        coordination[key] = LG.coordination_record_or_error(model, network, holder.params, key)
        coordination[key]['admm_block_weight'] = weight
        coordination[key]['termination_last_solve'] = term
        flexibility[key] = dict(_flexibility_block(model, network, holder.params), admm_block_weight=weight)
        legs = _row18_legs_block(model, network)
        if legs is not None:   # W65 (G13): computed separately, so the W64 leg sums above are untouched
            legs['primal_split'] = _row18_primal_split_block(model, network)
            for name in split_tot:
                split_tot[name] += weight * legs['primal_split'][name]
            for name in split_max:
                split_max[name] = max(split_max[name], legs['primal_split'][name])
            split_identity_all = split_identity_all and legs['primal_split']['identity_abs_gap_minus_split_le_residual']
        ipopt_options_dso.setdefault(f'DSO{node_id}', ipopt_options_in_force(holder.params))
        exit_class = str(term.get('ipopt_exit'))
        dso_exit_counts[exit_class] = dso_exit_counts.get(exit_class, 0) + 1
        row18_legs[key] = legs
        if legs is not None:
            tot['charge_p_weighted'] += weight * legs['charge_p']
            tot['charge_q_weighted'] += weight * legs['charge_q']
            for name in ('n_checked', 'n_missing_suffix_values', 'n_le_1e-6', 'n_le_1e-4', 'n_le_1e-2'):
                zl[name] += legs['zl_identity'][name]
            for name in ('worst_rel_zl_sum_vs_2coef', 'worst_rel_abs_dual_vs_half_zl_diff'):
                zl[name] = max(zl[name], legs['zl_identity'][name])
        detail = dispersion.get(('DSO', node_id, year, day))
        if detail is not None:
            import model_construction_helpers as MCH
            probs = {f'{s_m}_{s_o}': _scenario_probability(network, s_m, s_o)
                     for s_m in model.scenarios_market for s_o in model.scenarios_operation}
            pibar = [float(MCH.expected_market_price(network, p)) for p in model.periods]
            both = p_posthoc_block(detail['per_scenario'], probs, pibar, include_q=True)
            p_only = p_posthoc_block(detail['per_scenario'], probs, pibar, include_q=False)
            p_posthoc[key] = {'P_b': both, 'P_b_p_leg': p_only, 'admm_block_weight': weight}
            tot['P_posthoc_weighted'] += weight * both
            tot['P_posthoc_p_leg_weighted'] += weight * p_only
    flex_tot = {}
    for key, legs in flexibility.items():
        node = key.split('|')[1]
        agg = flex_tot.setdefault(f'DSO{node}', {k: 0.0 for k in ('p_up_mwh', 'p_down_mwh', 'q_up_mvarh',
                                                                    'q_down_mvarh')})
        for k in agg:
            agg[k] += legs['admm_block_weight'] * legs[k]
    charge_total = tot['charge_p_weighted'] + tot['charge_q_weighted']
    zl['tol'] = ROW18_ZL_IDENTITY_REL_TOL
    zl['holds'] = bool(zl['n_checked'] > 0 and zl['worst_rel_zl_sum_vs_2coef'] <= ROW18_ZL_IDENTITY_REL_TOL
                       and zl['worst_rel_abs_dual_vs_half_zl_diff'] <= ROW18_ZL_IDENTITY_REL_TOL)
    n_dso_entries = sum(1 for e in entries if e['network'] != 'TSO')
    summary = {
        'alpha_in_force': alpha,
        'units': ('volumes MWh (MW x 1 h periods) per representative day per block; *_weighted x admm_block_weight '
                  '(the Q weighting); money EUR'),
        'n_curtailment_entries_above_tol': len(entries), 'n_entries_dso': n_dso_entries,
        'n_entries_tso': len(entries) - n_dso_entries,
        'n_entries_with_row18_fields': sum(1 for e in entries if 'row18_dev_p_def_dual_raw' in e),
        'n_entries_with_bus_dual': sum(1 for e in entries if e.get('lmp_bus_dual_raw') is not None),
        'n_network_hours_with_rows': len(rows_by_hour),
        'curtailment_per_network': per_network,
        'n_coordination_blocks': len(coordination),
        'n_coordination_capture_errors': sum(1 for c in coordination.values() if 'capture_error' in c),
        'flexibility_per_dso_weighted': flex_tot,
        'row18_charge_p_weighted': tot['charge_p_weighted'], 'row18_charge_q_weighted': tot['charge_q_weighted'],
        'row18_charge_weighted_legs_total': charge_total,
        'row18_q_leg_share_of_charge': (tot['charge_q_weighted'] / charge_total) if charge_total else None,
        'P_charge_over_alpha': (charge_total / alpha) if alpha > 0.0 else None,
        'P_posthoc_weighted': tot['P_posthoc_weighted'], 'P_posthoc_p_leg_weighted': tot['P_posthoc_p_leg_weighted'],
        'row18_zl_identity': zl,
        'voltage_pin_total': _f(rc.get('voltage_pin_total')),
        'recourse_components': {k: _f(rc.get(k)) for k in (
            'gross_operational_cost', 'net_operational_recourse', 'terminal_salvage_value', 'voltage_pin_total',
            'interface_settlement_total', 'interface_settlement_deviation_total', 'detector_penalty_total')},
        'n_blocks_last_solve_not_succeeded': sum(1 for b in curtail_blocks.values()
                                                 if not b['termination_last_solve']['succeeded']),
        # W65 (G13): the realized gap P_charge - P_posthoc decomposed (weighted, EUR per unit alpha), the maxima of the
        # returned point's row residual / bound violation / split, the DSO last-solve IPOPT exits, the options in force
        'P_gap_decomposition_weighted': ({**split_tot, **split_max,
                                          'identity_abs_gap_minus_split_le_residual_all_blocks': split_identity_all}
                                         if alpha > 0.0 else None),
        'dso_last_solve_ipopt_exit_counts': dso_exit_counts,
        'ipopt_options_in_force_dso': ipopt_options_dso,
    }
    payload = {'schema': 'p515_s44_response_terminal_v1', 'summary': summary,   # in memory; v2 = its compact layout
               'curtailment_entries': entries, 'network_hour_rows': rows_by_hour,
               'curtailment_by_block': curtail_blocks, 'coordination_by_dso_block': coordination,
               'flexibility_by_dso_block': flexibility, 'row18_legs_by_dso_block': row18_legs,
               'p_posthoc_by_dso_block': p_posthoc,
               'constants': {'TOL_FACTOR': CA.TOL_FACTOR, 'CAP_SLACK_TOL': CA.CAP_SLACK_TOL, 'DUAL_TOL': CA.DUAL_TOL,
                             'ROW_SLACK_TOL': CA.ROW_SLACK_TOL, 'ROW18_ZL_IDENTITY_REL_TOL': ROW18_ZL_IDENTITY_REL_TOL,
                             'source': f'{CURTAILMENT_AUDIT_MODULE} (by import)'}}
    return payload, summary


# ==============================================================================
#  W65 (Addendum 40 ruling 1, G9): the COMPACT, LOSSLESS serialization of RESPONSE_TERMINAL_FILE. Nothing is capped,
#  subsampled or dropped: every captured entry is written; only the REPRESENTATION changes --
#    * no indentation, no spaces (separators ',' ':');
#    * curtailment_entries COLUMNAR: field names once, one column per field; the fields constant within a block
#      (RESPONSE_ENTRY_BLOCK_FIELDS) once per block in a block table, with the fields that block's entries lack;
#      network_hour_rows_key (== '{block}|{scenario}|{hour}', asserted per entry) rebuilt on decode;
#    * each coordination record's `hours` list COLUMNAR (field names once per block).
#  `write_response_terminal` re-reads the written file, decodes it and asserts it equals the in-memory payload's JSON
#  form EXACTLY (every entry, every field, every float) before returning; a mismatch raises (capture error).
#  `load_response_terminal` is the one reader (it also reads the v1 layout of the r1 smoke unchanged).
# ==============================================================================
RESPONSE_TERMINAL_SCHEMA = 'p515_s44_response_terminal_v2'
RESPONSE_ENTRIES_ENCODING = 'p515_s44_columnar_entries_v1'
RESPONSE_ROWS_ENCODING = 'p515_s44_columnar_rows_v1'
RESPONSE_ENTRY_BLOCK_FIELDS = ('block', 'network', 'year', 'day', 'ref_bus_index', 'termination_last_solve',
                               'row18_alpha')


def _hour_key(entry):
    return f"{entry['block']}|{entry['scenario']}|{entry['hour']}"


def encode_curtailment_entries(entries):
    """Lossless columnar form of the curtailment entries (see the section note). Raises if an assumption of the
    encoding (block-constant fields, the derivable hour key) does not hold, rather than writing a lossy form."""
    fields, seen = [], set()
    for e in entries:
        for k in e:
            if k not in seen and k not in RESPONSE_ENTRY_BLOCK_FIELDS and k != 'network_hour_rows_key':
                seen.add(k)
                fields.append(k)
    table, block_index, index_of = [], [], {}
    for e in entries:
        if e.get('network_hour_rows_key') != _hour_key(e):
            raise ValueError(f'entry hour key {e.get("network_hour_rows_key")!r} != {_hour_key(e)!r}')
        const = {k: e[k] for k in RESPONSE_ENTRY_BLOCK_FIELDS if k in e}
        absent = [k for k in fields if k not in e]
        sig = json.dumps([const, absent], sort_keys=True, default=str)
        i = index_of.get(e['block'])
        if i is None:
            i = index_of[e['block']] = len(table)
            table.append({'const': const, 'absent_fields': absent, '_sig': sig})
        elif table[i]['_sig'] != sig:
            raise ValueError(f"block-constant fields vary within block {e['block']}")
        block_index.append(i)
    return {'encoding': RESPONSE_ENTRIES_ENCODING, 'n': len(entries), 'fields': fields,
            'block_table': [{'const': b['const'], 'absent_fields': b['absent_fields']} for b in table],
            'block_index': block_index, 'columns': [[e.get(k) for e in entries] for k in fields],
            'rebuilt_on_decode': {'network_hour_rows_key': '{block}|{scenario}|{hour}'}}


def decode_curtailment_entries(encoded):
    if isinstance(encoded, list):   # the v1 layout (r1 smoke): already a list of dicts
        return encoded
    if encoded.get('encoding') != RESPONSE_ENTRIES_ENCODING:
        raise ValueError(f"unknown entries encoding {encoded.get('encoding')!r}")
    fields, columns, table = encoded['fields'], encoded['columns'], encoded['block_table']
    absent = [set(b['absent_fields']) for b in table]
    out = []
    for j, i in enumerate(encoded['block_index']):
        e = dict(table[i]['const'])
        for k, col in zip(fields, columns):
            if k not in absent[i]:
                e[k] = col[j]
        e['network_hour_rows_key'] = _hour_key(e)
        out.append(e)
    if len(out) != encoded['n']:
        raise ValueError(f"decoded {len(out)} entries, header says {encoded['n']}")
    return out


def encode_rows(rows):
    """Columnar form of a list of dicts sharing one key list (in order); returned unchanged otherwise."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        return rows
    keys = list(rows[0])
    if any(list(r) != keys for r in rows):
        return rows
    return {'encoding': RESPONSE_ROWS_ENCODING, 'n': len(rows), 'fields': keys,
            'columns': [[r[k] for r in rows] for k in keys]}


def decode_rows(value):
    if isinstance(value, dict) and value.get('encoding') == RESPONSE_ROWS_ENCODING:
        return [dict(zip(value['fields'], vals)) for vals in zip(*value['columns'])] if value['n'] else []
    return value


def encode_response_payload(payload):
    out = dict(payload)
    out['schema'] = RESPONSE_TERMINAL_SCHEMA
    out['curtailment_entries'] = encode_curtailment_entries(payload['curtailment_entries'])
    out['coordination_by_dso_block'] = {k: ({**c, 'hours': encode_rows(c['hours'])} if 'hours' in c else c)
                                        for k, c in payload['coordination_by_dso_block'].items()}
    return out


def decode_response_payload(stored):
    out = dict(stored)
    out['curtailment_entries'] = decode_curtailment_entries(stored['curtailment_entries'])
    out['coordination_by_dso_block'] = {k: ({**c, 'hours': decode_rows(c['hours'])} if 'hours' in c else c)
                                        for k, c in stored['coordination_by_dso_block'].items()}
    return out


def load_response_terminal(path):
    """THE reader of RESPONSE_TERMINAL_FILE: the payload with entries and coordination hours in list-of-dict form
    (v2 compact files decoded; v1 files returned as stored)."""
    with open(path) as handle:
        return decode_response_payload(json.load(handle))


def _write_once_json_compact(path, obj):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
    with open(path, 'x') as handle:
        json.dump(obj, handle, separators=(',', ':'), default=_json_default)


def write_response_terminal(planning, models, eval_dir, optimization_results=None):
    """Writes RESPONSE_TERMINAL_FILE (write-once, COMPACT and LOSSLESS -- W65) with its own cost: runtime, the process
    RSS before / after, the process peak before / after (so the capture's memory is MEASURED, not assumed), the file
    size, and the round-trip verification (the written file decoded == the in-memory payload's JSON form, exactly;
    raises otherwise)."""
    import resource as _resource
    try:
        import psutil
        proc = psutil.Process()
    except Exception:  # noqa: BLE001
        proc = None
    t0 = time.time()
    rss0 = proc.memory_info().rss if proc is not None else None
    peak0 = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    payload, summary = response_terminal_capture(planning, models, optimization_results=optimization_results)
    rss1 = proc.memory_info().rss if proc is not None else None
    peak1 = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    cost = {'runtime_s': time.time() - t0, 'rss_before_bytes': rss0, 'rss_after_bytes': rss1,
            'ru_maxrss_before_bytes': peak0, 'ru_maxrss_after_bytes': peak1,
            'ru_maxrss_units': 'bytes on macOS/BSD, kilobytes on Linux'}
    payload['capture_cost'] = cost
    path = os.path.join(eval_dir, RESPONSE_TERMINAL_FILE)
    t_w = time.time()
    _write_once_json_compact(path, encode_response_payload(payload))
    expected = json.loads(json.dumps(payload, default=_json_default))  # W74: the writer's own hook
    got = load_response_terminal(path)
    got['schema'] = expected['schema']   # the only intended difference: the layout's schema tag
    # canonical JSON text per section: exact (floats round-trip through repr) and NaN-safe (nan != nan in Python)
    mismatch = sorted(k for k in set(expected) | set(got)
                      if json.dumps(expected.get(k), sort_keys=True) != json.dumps(got.get(k), sort_keys=True))
    if mismatch:
        raise RuntimeError(f'{RESPONSE_TERMINAL_FILE}: the compact file does not decode to the captured payload '
                           f'(sections differing: {mismatch})')
    roundtrip = {'verified': True, 'sections_compared': sorted(expected), 'n_entries_written': len(
        got['curtailment_entries']), 'n_entries_captured': len(payload['curtailment_entries']),
        'write_and_verify_s': time.time() - t_w}
    del expected, got
    cost['file_bytes'] = os.path.getsize(path)
    cost['roundtrip'] = roundtrip
    # the write-and-verify step's own memory, measured (it holds the payload's JSON form and its decoded copy)
    cost['rss_after_write_verify_bytes'] = proc.memory_info().rss if proc is not None else None
    cost['ru_maxrss_after_write_verify_bytes'] = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    return {'status': 'written', 'path': os.path.relpath(path, REPO), 'sha256': sha256_file(path),
            'schema': RESPONSE_TERMINAL_SCHEMA, 'capture_cost': cost, **summary}


def assert_alpha_row_capture_paths():
    """RULE ELEVEN for the W64 capture, asserted on the code that will run, BEFORE any solve: every production
    function the hooks wrap or read exists and is called where the hooks assume; the suffixes the dual capture reads
    are declared on production models and survive `_release_solution_bookkeeping`; the diagnostic definitions reused
    by import load (disarmed) and expose what is used; the harness wires the capture before the workbook and
    post-certification. Raises AssertionError listing every missing path."""
    import inspect
    import shared_resources_planning as srp
    import network as network_module
    import p515_g_g1_g4_admm_gates as G
    import p58_rescale as R
    run_src = inspect.getsource(srp._run_operational_planning)
    prep_dso_src = inspect.getsource(srp._prepare_distribution_objectives_for_admm)
    prep_tso_src = inspect.getsource(srp._prepare_transmission_objectives_for_admm)
    net_src = inspect.getsource(network_module)
    child_src = inspect.getsource(_child_real)
    # executable lines only (production's comment there names the suffixes it leaves untouched)
    release_code = '\n'.join(ln for ln in inspect.getsource(network_module._release_solution_bookkeeping).splitlines()[1:]
                             if ln.strip() and not ln.strip().startswith('#'))
    LG = import_disarmed_diagnostic(COORDINATION_MODULE)
    CA = import_disarmed_diagnostic(CURTAILMENT_AUDIT_MODULE)
    DEC = import_disarmed_diagnostic(DECOMPOSITION_MODULE)
    boyd_at = run_src.find('boyd_metrics = get_admm_boyd_residual_metrics(')
    local_at = run_src.find('local_solves_ok = _admm_local_solves_succeeded(planning_problem, results)')
    checks = {
        **{f'production_fn_{n}': callable(getattr(srp, n, None)) for n in (
            'create_transmission_network_model', 'create_shared_energy_storage_model',
            '_prepare_distribution_objectives_for_admm', '_prepare_transmission_objectives_for_admm',
            'get_admm_boyd_residual_metrics', '_admm_local_solves_succeeded', '_get_operational_interface_dispersion',
            '_get_operational_recourse_components', '_get_local_interface_settlement', '_get_admm_block_weight',
            '_activate_row18_with_settlement')},
        'init_check_then_prepare_dso_then_prepare_tso': (
            0 <= run_src.find('if not _admm_local_solves_succeeded(planning_problem, results):')
            < run_src.find('_prepare_distribution_objectives_for_admm(distribution_networks, dso_models)')
            < run_src.find('_prepare_transmission_objectives_for_admm(transmission_network, tso_model)')),
        'tso_and_esso_built_before_prepare': (
            0 <= run_src.find('create_transmission_network_model(')
            < run_src.find('_prepare_distribution_objectives_for_admm(distribution_networks, dso_models)')
            and 0 <= run_src.find('create_shared_energy_storage_model(')
            < run_src.find('_prepare_distribution_objectives_for_admm(distribution_networks, dso_models)')),
        'boyd_once_per_cycle_before_local_check': (run_src.count('get_admm_boyd_residual_metrics(') == 1
                                                   and 0 <= boyd_at < local_at),
        'local_check_called_exactly_twice': run_src.count('_admm_local_solves_succeeded(') == 2,
        'row18_activated_in_prepare_dso': '_activate_row18_with_settlement(dso_model[year][day])' in prep_dso_src,
        'prepare_dso_sets_weight_and_penalty': ('interface_settlement_weight.set_value(1.00)' in prep_dso_src
                                                and 'penalty_gen_curtailment.set_value(0.00)' in prep_dso_src),
        'prepare_tso_sets_weight_and_penalty': ('interface_settlement_weight.set_value(1.00)' in prep_tso_src
                                                and 'penalty_gen_curtailment.set_value(0.00)' in prep_tso_src),
        'model_declares_dual_and_bound_suffixes': ('model.dual = pe.Suffix(direction=pe.Suffix.IMPORT_EXPORT)' in net_src
                                                   and 'model.ipopt_zL_out = pe.Suffix(direction=pe.Suffix.IMPORT)' in net_src
                                                   and 'model.ipopt_zU_out = pe.Suffix(direction=pe.Suffix.IMPORT)' in net_src),
        'release_bookkeeping_leaves_suffixes': all(t not in release_code for t in ('dual', 'ipopt_z', 'Suffix')),
        'solved_objective_is_p58_rescaled': hasattr(R, 'RESCALED_OBJECTIVE') and 'patched_admm_objectives' in dir(R),
        'coordination_record_by_import': callable(getattr(LG, 'coordination_record', None))
                                         and callable(getattr(LG, 'coordination_record_or_error', None)),
        'coordination_record_reads_duals_rho_request_pbar_scale': all(t in inspect.getsource(LG.coordination_record) for t in (
            "'dual_pf_p_req'", "'rho_pf'", "'p_pf_req_mw'", "'pbar_mw'", "'admm_objective_scale'")),
        'audit_helpers_by_import': all(callable(getattr(CA, n, None)) for n in ('_gen_bus', '_network_hour_rows',
                                                                                 '_active_rows')),
        'audit_constants_by_import': all(hasattr(CA, n) for n in ('TOL_FACTOR', 'CAP_SLACK_TOL', 'DUAL_TOL',
                                                                  'ROW_SLACK_TOL')),
        'decomposition_by_import': callable(getattr(DEC, 'block_decomposition', None)),
        'post_run_hook_gets_optimization_results': "hook_kwargs['optimization_results'] = _results" in inspect.getsource(
            G.run_admm_arm),
        'p_posthoc_inputs_captured': all(t in inspect.getsource(multiscenario_terminal_capture) for t in (
            "'per_scenario_d': detail['per_scenario']", "'pibar_by_hour'", "'admm_block_weight': weight")),
        'p_posthoc_formula_defined': callable(p_posthoc_block),
        'child_installs_alpha_row_hooks': 'alpha_row_run_hooks(' in child_src,
        'child_closes_hooks_in_post_run_hook': 'run_hooks.close()' in child_src,
        'child_writes_response_before_workbook_and_post_certification': (
            0 <= child_src.find('write_response_terminal(') < child_src.find('write_operational_workbook(')
            < child_src.find('run_post_certification(')),
        'child_merges_per_cycle_response': 'read_per_cycle_response(eval_dir)' in child_src,
        'child_response_capture_error_exits_2': "'response_terminal'" in child_src,
        # W65 (Addendum 40 ruling 1): the G13 bound inputs and the compact lossless layout (G9)
        'termination_record_carries_ipopt_exit': "'ipopt_exit': ipopt_exit_class(message)" in inspect.getsource(
            _termination_record),
        'primal_split_captured_per_dso_block': ("legs['primal_split'] = _row18_primal_split_block(model, network)"
                                                in inspect.getsource(response_terminal_capture)),
        'primal_split_terms': all(t in inspect.getsource(_row18_primal_split_block) for t in (
            "acc['gap']", "acc['split_negative']", "acc['split_positive']", "acc['residual']", "acc['bound_weight']")),
        'ipopt_options_in_force_captured': "'ipopt_options_in_force_dso'" in inspect.getsource(response_terminal_capture),
        'compact_writer_with_roundtrip': all(t in inspect.getsource(write_response_terminal) for t in (
            '_write_once_json_compact(path, encode_response_payload(payload))', 'load_response_terminal(path)',
            "'verified': True")),
        'no_entry_cap_in_encoder': ("'columns': [[e.get(k) for e in entries] for k in fields]"
                                    in inspect.getsource(encode_curtailment_entries)),
    }
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (W64 alpha-row capture): capture paths missing: {missing}')
    return checks


TERMINAL_PHASE_LOCK_NAME = '.terminal_phase.lock'
TERMINAL_PHASE_LOCK_POLL_S = 5.0
TERMINAL_PHASE_LOCK_TIMEOUT_S = 45 * 60.0


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def acquire_terminal_phase_lock(lock_path, timeout_s=TERMINAL_PHASE_LOCK_TIMEOUT_S,
                                poll_s=TERMINAL_PHASE_LOCK_POLL_S):
    """W47: at most ONE child of a derived-instance campaign runs its TERMINAL PHASE (the multi-scenario capture,
    production's workbook and the post-certification step) at a time. Measured zero-solve on the pilot's own models
    (p515_s52_pilot_checks, memory_by_stage): the workbook writer holds a +3.5-3.75 GiB transient for ~2 min and the
    certified-model pickle a +5 GiB transient, on top of the run's models; two children in that phase together
    would stack them. The lock (created O_EXCL in the campaign root) holds the holder's pid; a lock whose pid is
    dead is stale and is taken over (recorded). Waits up to `timeout_s`; on timeout the phase runs WITHOUT the lock
    (recorded) rather than losing its outputs. Returns the record."""
    t0 = time.time()
    stolen = []
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({'pid': os.getpid(), 'utc': _utc()}).encode())
            os.close(fd)
            return {'status': 'held', 'lock_path': os.path.relpath(lock_path, REPO), 'waited_s': time.time() - t0,
                    'stale_locks_taken_over': stolen}
        except FileExistsError:
            try:
                with open(lock_path) as handle:
                    holder = json.load(handle)
            except (OSError, ValueError):
                holder = {}
            if holder.get('pid') is not None and not _pid_alive(holder.get('pid')):
                stolen.append(holder)
                try:
                    os.remove(lock_path)
                except FileNotFoundError:
                    pass
                continue
            if time.time() - t0 > timeout_s:
                return {'status': 'timeout_ran_without_lock', 'lock_path': os.path.relpath(lock_path, REPO),
                        'waited_s': time.time() - t0, 'held_by': holder, 'stale_locks_taken_over': stolen}
            time.sleep(poll_s)


def release_terminal_phase_lock(lock_path, record):
    if (record or {}).get('status') != 'held' or not os.path.exists(lock_path):
        return False
    try:
        with open(lock_path) as handle:
            holder = json.load(handle)
    except (OSError, ValueError):
        holder = {}
    if holder.get('pid') == os.getpid():
        os.remove(lock_path)
        return True
    return False


def write_operational_workbook(planning, models, optimization_results, primal_evolution, state, execution_time):
    """Production's own operational-planning workbook of the terminal point
    (`SharedResourcesPlanning.write_operational_planning_results_to_excel`, the call `run_operational_planning`
    makes with print_results=True), fed the run's own per-block SolverResults and primal evolution. Written to
    `planning.results_dir` (the eval dir's results/, `_set_results_dir_for_arm`). Refuses to overwrite; fails
    loudly when production falls back to its timestamped backup name. Collects the writer's garbage afterwards
    (the openpyxl workbook is cyclic garbage: +3.5 GiB held until collected, measured)."""
    import gc
    t0 = time.time()
    filename = f'{planning.name}_distributed_terminal'
    path = os.path.join(planning.results_dir, f'{filename}.xlsx')
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
    planning.write_operational_planning_results_to_excel(
        models, optimization_results, filename=filename, primal_evolution=list(primal_evolution or []),
        admm_diagnostics=(state or {}).get('admm_diagnostics', []),
        solver_recovery_diagnostics=(state or {}).get('solver_recovery_diagnostics', []),
        execution_time=execution_time)
    write_s = time.time() - t0
    gc.collect()  # the writer's openpyxl workbook is cyclic garbage: release it before the next step
    if not os.path.isfile(path):
        raise RuntimeError(f'the operational workbook was not written at {path} (production may have used a '
                           f'timestamped backup name)')
    return {'status': 'written', 'path': os.path.relpath(path, REPO), 'sha256': sha256_file(path),
            'size_bytes': os.path.getsize(path), 'runtime_s': time.time() - t0, 'write_s': write_s,
            'writer': 'SharedResourcesPlanning.write_operational_planning_results_to_excel (production)',
            'point': 'terminal models of the run, BEFORE any post-certification step'}


# ==============================================================================
#  P5.15 Addendum 46 ruling 7 (W84, Planner ruling Q2): per-solve floor status and the tail state, EVERY evaluation
# ==============================================================================
def persist_convergence_depth_capture(state, eval_dir):
    """Writes (write-once) `NETWORK_IPOPT_SOLVE_RECORDS_FILE` -- production's `state['network_ipopt_solve_records']`,
    one JSON line per record, in production's order, unmodified -- and `CONVERGENCE_DEPTH_TAIL_STATE_FILE` --
    production's `state['convergence_depth_tail']` as one JSON document (null when the state carries none). Zero
    solves; the returned summary goes into the record. `load_network_ipopt_solve_records` /
    `load_convergence_depth_tail_state` read them back."""
    from collections import Counter
    st = state or {}
    records = st.get('network_ipopt_solve_records')
    tail_state = st.get('convergence_depth_tail')
    records_path = os.path.join(eval_dir, NETWORK_IPOPT_SOLVE_RECORDS_FILE)
    tail_path = os.path.join(eval_dir, CONVERGENCE_DEPTH_TAIL_STATE_FILE)
    for path in (records_path, tail_path):
        if os.path.exists(path):
            raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
    with open(records_path, 'w') as handle:
        for record in (records or []):
            handle.write(json.dumps(record, default=_json_default) + '\n')
    _write_once_json(tail_path, tail_state)
    recs = list(records or [])
    return {
        'status': 'written',
        'network_ipopt_solve_records_path': os.path.relpath(records_path, REPO),
        'network_ipopt_solve_records_sha256': sha256_file(records_path),
        'state_carried_records': records is not None,
        'n_records': len(recs),
        'records_per_round': {str(k): v for k, v in sorted(Counter(r.get('round') for r in recs).items(),
                                                            key=lambda kv: (kv[0] is None, kv[0] or 0))},
        'floor_status_tally': dict(sorted(Counter(
            f"{'TSO' if r.get('agent') == 'TSO' else 'DSO'}|{r.get('floor_status')}" for r in recs).items())),
        'exit_tally': dict(sorted(Counter(str(r.get('exit')) for r in recs).items())),
        'n_parse_problems': sum(1 for r in recs if r.get('parse_reason') is not None),
        'stale_network_ipopt_solve_records_discarded': st.get('stale_network_ipopt_solve_records_discarded'),
        'convergence_depth_tail_state_path': os.path.relpath(tail_path, REPO),
        'convergence_depth_tail_state_sha256': sha256_file(tail_path),
        'state_carried_tail_state': tail_state is not None,
    }


def load_network_ipopt_solve_records(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_convergence_depth_tail_state(path):
    with open(path) as handle:
        return json.load(handle)


# ==============================================================================
#  P5.15 Addendum 46 ruling 7 (W85, Planner ruling Q3 on W84): the same capture APPENDED PER ROUND, so a child that
#  raises or is killed mid-run keeps every completed round's records (the end-of-run write above is unchanged)
# ==============================================================================
class ConvergenceDepthAppender:
    """Per-round, append-only, fsync'd copy of the convergence-depth capture of ONE evaluation (W85).

    Two files in the eval dir, both opened in append mode and flushed + fsync'd after every write:
      - `NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE`: every record production's `_drain_network_ipopt_solve_records`
        returns for a round (0 = initialisation, k = cycle k), serialized EXACTLY as `persist_convergence_depth_capture`
        serializes `state['network_ipopt_solve_records']` (same `json.dumps(record, default=_json_default)` line, same
        order). Production drains nothing else into that list, so on a completed run this file is BYTE-IDENTICAL to
        the end-of-run `NETWORK_IPOPT_SOLVE_RECORDS_FILE` -- which `reconcile` checks (sha256) and records.
      - `CONVERGENCE_DEPTH_APPEND_EVENTS_FILE`: line 1 = the child's rule-eleven tail checklist (written before any
        solve); then one line per production tail event -- 'baseline' (`_capture_convergence_depth_tail_baseline`),
        'apply' (`_apply_convergence_depth_tail`: a cycle's record, or the restore at exit when cycle is None),
        'next_state' (`_convergence_depth_tail_next_state`: the AA-off predicate production stores into that cycle's
        record) -- and a 'drained' checkpoint per round (round, n_records, records-file size after the round).
        `reconstruct_convergence_depth_tail_state` rebuilds production's `state['convergence_depth_tail']` from it;
        `reconcile` checks it equals the state production returned.
    The production functions are wrapped (call-through, result returned unchanged, exceptions passed through) only
    inside `convergence_depth_append_hooks`; nothing is written for production's stale-record discard (round None),
    which production does not keep either. After `seal()` (called once the end-of-run write is reconciled) the
    wrappers write nothing more. `drain_on_failure` is the exception path: records still in the Network objects of
    the round in flight are drained (production's function, called directly) and appended. Zero solves."""

    def __init__(self, eval_dir, checklist):
        self.eval_dir = eval_dir
        self.records_path = os.path.join(eval_dir, NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE)
        self.events_path = os.path.join(eval_dir, CONVERGENCE_DEPTH_APPEND_EVENTS_FILE)
        for path in (self.records_path, self.events_path):
            if os.path.exists(path):
                raise RuntimeError(f'refusing to overwrite existing artifact: {path}')
        self.planning = None
        self.original_drain = None
        self.last_round = None
        self.rounds = []
        self.n_records = 0
        self.sealed = False
        self.drains_after_seal = 0
        self.failure_drain = None
        self.write_errors = []
        self._event({'event': 'checklist', 'utc': _utc(), 'pid': os.getpid(), 'checklist': checklist})

    @staticmethod
    def _append_lines(path, lines):
        with open(path, 'a') as handle:
            for line in lines:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _event(self, event):
        self._append_lines(self.events_path, [json.dumps(event, default=_json_default) + '\n'])

    def append_round(self, round_index, records, event='drained'):
        self._append_lines(self.records_path,
                           [json.dumps(record, default=_json_default) + '\n' for record in records])
        self.n_records += len(records)
        self.rounds.append(round_index)
        self._event({'event': event, 'round': round_index, 'n_records': len(records),
                     'records_file_bytes': os.path.getsize(self.records_path), 'utc': _utc()})

    def _write_error(self, where, error):
        """A capture write must never change the run: the error is recorded (and printed), the run goes on, and
        `reconcile` then fails (the child exits 2 after writing its record)."""
        self.write_errors.append({'where': where, 'error': f'{type(error).__name__}: {error}', 'utc': _utc()})
        print(traceback.format_exc(), file=sys.stderr, flush=True)

    def on_drain(self, planning_problem, round_index, records):
        self.planning = planning_problem
        if round_index is None:
            return  # production's stale-record discard: not kept by production, not kept here
        self.last_round = round_index
        if self.sealed:
            self.drains_after_seal += 1
            return
        try:
            self.append_round(round_index, records)
        except Exception as error:  # noqa: BLE001 -- see _write_error
            self._write_error(f'drain round {round_index}', error)

    def on_tail_event(self, event, **payload):
        if self.sealed:
            return
        try:
            self._event({'event': event, **payload})
        except Exception as error:  # noqa: BLE001 -- see _write_error
            self._write_error(f'tail event {event}', error)

    def drain_on_failure(self, reason):
        """Exception path (the child is failing): drain the round in flight -- 0 if no round was drained yet, else
        the last drained round + 1 (inferred: a failure after the last cycle's drain gives that round number with 0
        records) -- with production's own drain function, and append it. Never raises."""
        try:
            if self.sealed:
                self.failure_drain = {'status': 'skipped', 'why': 'sealed: the end-of-run write was reconciled'}
            elif self.planning is None or self.original_drain is None:
                self.failure_drain = {'status': 'skipped',
                                      'why': 'production never drained (the failure preceded the ADMM run)'}
            else:
                round_in_flight = 0 if self.last_round is None else self.last_round + 1
                records = self.original_drain(self.planning, round_in_flight)
                self.append_round(round_in_flight, records, event='drained_at_failure')
                self.failure_drain = {'status': 'drained', 'round_in_flight': round_in_flight,
                                      'n_records': len(records), 'reason': reason,
                                      'write_errors_during_run': list(self.write_errors)}
        except Exception as error:  # noqa: BLE001 -- the failure record must still be written
            self.failure_drain = {'status': 'error', 'error': f'{type(error).__name__}: {error}'}
        return self.failure_drain

    def reconcile(self, capture_summary, state):
        """After the end-of-run write: the appended records file must be byte-identical (sha256) to the end-of-run
        records file, and the tail state rebuilt from the events must equal (after JSON round-trip) the tail state
        production returned. `ok` False -> recorded; the child exits 2 after writing its record."""
        tail_state = (state or {}).get('convergence_depth_tail')
        append_sha = sha256_file(self.records_path) if os.path.exists(self.records_path) else None
        rebuilt = reconstruct_convergence_depth_tail_state(load_convergence_depth_append_events(self.events_path)[0])
        returned = json.loads(json.dumps(tail_state, default=_json_default)) if tail_state is not None else None
        out = {
            'records_append_path': os.path.relpath(self.records_path, REPO),
            'records_append_sha256': append_sha,
            'end_of_run_records_sha256': (capture_summary or {}).get('network_ipopt_solve_records_sha256'),
            'n_records_appended': self.n_records, 'rounds_appended': list(self.rounds),
            'events_path': os.path.relpath(self.events_path, REPO),
            'records_append_byte_identical_to_end_of_run_file': (
                append_sha is not None
                and append_sha == (capture_summary or {}).get('network_ipopt_solve_records_sha256')),
            'tail_state_rebuilt_from_events_equals_returned_state': rebuilt == returned,
            'write_errors': list(self.write_errors),
        }
        out['ok'] = ((capture_summary or {}).get('status') == 'written' and not self.write_errors
                     and out['records_append_byte_identical_to_end_of_run_file']
                     and out['tail_state_rebuilt_from_events_equals_returned_state'])
        if not out['tail_state_rebuilt_from_events_equals_returned_state']:
            out['tail_state_rebuilt'] = rebuilt
        return out

    def seal(self):
        self.sealed = True
        self._event({'event': 'sealed', 'utc': _utc(), 'n_records': self.n_records, 'rounds': list(self.rounds)})


@contextmanager
def convergence_depth_append_hooks(appender):
    """W85: wraps production's `_drain_network_ipopt_solve_records`, `_capture_convergence_depth_tail_baseline`,
    `_apply_convergence_depth_tail` and `_convergence_depth_tail_next_state` (module globals of
    `shared_resources_planning`, resolved at call time by `_run_operational_planning`) so each result is appended by
    `appender` as it is produced, then returned UNCHANGED. Restored on exit, even on error."""
    import shared_resources_planning as srp
    names = ('_drain_network_ipopt_solve_records', '_capture_convergence_depth_tail_baseline',
             '_apply_convergence_depth_tail', '_convergence_depth_tail_next_state')
    originals = {name: getattr(srp, name) for name in names}
    appender.original_drain = originals['_drain_network_ipopt_solve_records']

    def drain(planning_problem, round_index):
        records = originals['_drain_network_ipopt_solve_records'](planning_problem, round_index)
        appender.on_drain(planning_problem, round_index, records)
        return records

    def baseline(planning_problem, admm_parameters):
        result = originals['_capture_convergence_depth_tail_baseline'](planning_problem, admm_parameters)
        appender.on_tail_event('baseline', baseline=result, option=srp.CONVERGENCE_DEPTH_TAIL_OPTION,
                               declaration_in_force=dict(admm_parameters.convergence_depth_tail))
        return result

    def apply(planning_problem, admm_parameters, active, baseline_state, cycle):
        result = originals['_apply_convergence_depth_tail'](planning_problem, admm_parameters, active,
                                                             baseline_state, cycle)
        appender.on_tail_event('apply', cycle=cycle, record=result)
        return result

    def next_state(cycle_convergence, aa_enabled, aa_record):
        result = originals['_convergence_depth_tail_next_state'](cycle_convergence, aa_enabled, aa_record)
        appender.on_tail_event('next_state', value=result)
        return result

    wrappers = {'_drain_network_ipopt_solve_records': drain, '_capture_convergence_depth_tail_baseline': baseline,
                '_apply_convergence_depth_tail': apply, '_convergence_depth_tail_next_state': next_state}
    for name, fn in wrappers.items():
        setattr(srp, name, fn)
    try:
        yield appender
    finally:
        for name, fn in originals.items():
            setattr(srp, name, fn)


def _load_jsonl_tolerant(path):
    """(complete JSON lines, n lines that did not parse -- e.g. a last line cut by a kill mid-write)."""
    rows, bad = [], 0
    if not os.path.exists(path):
        return rows, bad
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad += 1
    return rows, bad


def load_convergence_depth_append_events(path):
    return _load_jsonl_tolerant(path)


def reconstruct_convergence_depth_tail_state(events):
    """Production's `state['convergence_depth_tail']` rebuilt from the append events (the assembly
    `_run_operational_planning` does): {'enabled': False} without a 'baseline' event; otherwise enabled, the tail value
    and option, the baseline, per_cycle = the 'apply' records of cycles 1..k, each followed by its 'next_state' value
    as 'aa_off_predicate_end_of_cycle', and restore_at_exit = the 'apply' record with cycle None (None if not
    reached). None if no event at all. For a run that failed this is the state up to the failure."""
    if not events:
        return None
    base = next((e for e in events if e.get('event') == 'baseline'), None)
    if base is None:
        return {'enabled': False}
    state = {'enabled': True, 'compl_inf_tol_tail': base['declaration_in_force']['compl_inf_tol'],
             'option': base['option'], 'baseline': base['baseline'], 'per_cycle': [], 'restore_at_exit': None}
    for e in events:
        if e.get('event') == 'apply':
            if e.get('cycle') is None:
                state['restore_at_exit'] = e['record']
            else:
                state['per_cycle'].append(dict(e['record']))
        elif e.get('event') == 'next_state' and state['per_cycle']:
            state['per_cycle'][-1]['aa_off_predicate_end_of_cycle'] = e['value']
    return state


def recover_convergence_depth_append(eval_dir):
    """What the per-round append preserved for an evaluation, read from its files alone (parent side, no model
    import) -- for a failure record. None when the child never created the files."""
    records_path = os.path.join(eval_dir, NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE)
    events_path = os.path.join(eval_dir, CONVERGENCE_DEPTH_APPEND_EVENTS_FILE)
    if not os.path.exists(records_path) and not os.path.exists(events_path):
        return None
    from collections import Counter
    records, bad_records = _load_jsonl_tolerant(records_path)
    events, bad_events = _load_jsonl_tolerant(events_path)
    header = events[0] if events and events[0].get('event') == 'checklist' else None
    drains = [e for e in events if e.get('event') in ('drained', 'drained_at_failure')]
    return {
        'records_append_path': os.path.relpath(records_path, REPO),
        'records_append_sha256': sha256_file(records_path) if os.path.exists(records_path) else None,
        'n_records': len(records), 'n_unparsable_record_lines': bad_records,
        'records_per_round': {str(k): v for k, v in sorted(Counter(r.get('round') for r in records).items(),
                                                            key=lambda kv: (kv[0] is None, kv[0] or 0))},
        'rounds_drained': [e.get('round') for e in drains if e.get('event') == 'drained'],
        'round_drained_at_failure': next((e.get('round') for e in drains if e.get('event') == 'drained_at_failure'),
                                         None),
        'floor_status_tally': dict(sorted(Counter(
            f"{'TSO' if r.get('agent') == 'TSO' else 'DSO'}|{r.get('floor_status')}" for r in records).items())),
        'events_path': os.path.relpath(events_path, REPO), 'n_events': len(events),
        'n_unparsable_event_lines': bad_events,
        'sealed': any(e.get('event') == 'sealed' for e in events),
        'tail_checklist_from_child': (header or {}).get('checklist'),
        'tail_state_rebuilt': reconstruct_convergence_depth_tail_state(events),
    }


def convergence_depth_tail_state_check(checklist, state):
    """W84: the tail state production RETURNED must agree with what the child asserted before the run
    (`assert_convergence_depth_tail_capture`): `enabled` identical, and when enabled the tail value identical to the
    declaration. `match` False -> recorded, and the child exits 2 after writing its record."""
    tail_state = (state or {}).get('convergence_depth_tail')
    tail_state = tail_state if isinstance(tail_state, dict) else {}
    expected = bool(checklist['tail_enabled_for_this_run'])
    enabled_in_state = tail_state.get('enabled')
    per_cycle = tail_state.get('per_cycle') or []
    match = enabled_in_state is expected
    if expected:
        match = match and tail_state.get('compl_inf_tol_tail') == checklist['compl_inf_tol_tail']
    return {
        'expected_enabled': expected, 'enabled_in_state': enabled_in_state,
        'expected_compl_inf_tol_tail': checklist['compl_inf_tol_tail'],
        'compl_inf_tol_tail_in_state': tail_state.get('compl_inf_tol_tail'),
        'match': match,
        'n_cycles_recorded': len(per_cycle),
        'cycles_tail_active': [p.get('cycle') for p in per_cycle if p.get('active')],
        'n_cycles_acted': sum(1 for p in per_cycle if p.get('acted')),
        'restore_at_exit_acted': (tail_state.get('restore_at_exit') or {}).get('acted'),
    }


def _child_real(args, spec, spec_path, entry, eval_dir, lock_content, env_caps, started, progress=None):
    """`progress` (Addendum 27, W5): a dict the caller (`main_child`) owns; filled with
    `case_file_sha256_in_child` and the configuration-hook `holder` as soon as each is
    known, so the exception-path record carries them (None when not reached)."""
    import pyomo.environ as pe  # noqa: F401
    import p515_g_g1_g4_admm_gates as G
    from p515_s40_polish_gap import _build_floor_rows

    if progress is None:
        progress = {}
    holder = {}
    progress['holder'] = holder
    capture_checklist = assert_record_capture_paths()
    # W84 (Addendum 46 ruling 7, Q3): whether THIS run has the convergence-depth tail, asserted and recorded up front.
    tail_checklist = assert_convergence_depth_tail_capture(spec)
    progress['convergence_depth_tail_checklist'] = tail_checklist
    print(f"[S44-CHILD] convergence-depth tail enabled for this run: {tail_checklist['tail_enabled_for_this_run']} "
          f"({tail_checklist['source']}; declared {tail_checklist['declared']})", flush=True)
    eff_overrides = validate_overrides(entry['overrides'] if 'overrides' in entry
                                       else (spec['configuration'].get('overrides') or {}))
    # Addendum 27 item 1: with a declared case-file AA dict, AA is on iff the effective (declaration +
    # override) dict says so; undeclared specs keep the override-only rule (their hook requires case-file AA off).
    case_file_aa = validate_case_file_anderson_acceleration(
        spec['configuration'].get('case_file_anderson_acceleration'))
    if case_file_aa is None:
        aa_on = bool((eff_overrides.get('anderson_acceleration') or {}).get('enabled'))
    else:
        aa_on = bool(effective_anderson_acceleration(case_file_aa, eff_overrides).get('enabled'))
    case_file_sha256_in_child = sha256_file(CASE_FILE)
    progress['case_file_sha256_in_child'] = case_file_sha256_in_child
    # Addendum 30 (W21): a declared ESS ageing baseline -> the ESS parameters file must hash to the spec's pin
    # before anything is built (the hook then checks the LOADED parameters against the declaration).
    ess_ageing = validate_ess_ageing_baseline(spec['configuration'].get('ess_ageing_baseline'))
    ess_params_sha256_in_child = None
    if ess_ageing is not None:
        ess_params_sha256_in_child = sha256_file(os.path.join(REPO, ESS_PARAMS_FILE_REL))
        progress['ess_params_sha256_in_child'] = ess_params_sha256_in_child
        pin = spec['configuration'].get('ess_params_file') or {}
        if pin.get('path') != ESS_PARAMS_FILE_REL or pin.get('sha256') != ess_params_sha256_in_child:
            raise RuntimeError(f'ess_ageing_baseline: {ESS_PARAMS_FILE_REL} sha256 {ess_params_sha256_in_child} != '
                               f'the spec pin {pin}')
    # Addenda 28-29 (W20): a model variant runs only under its explicit label, at spec AND entry level.
    model_variant = validate_model_variant(entry.get('model_variant'))
    if model_variant is not None and (spec.get('model_variant_label') != MODEL_VARIANT_LABEL
                                      or entry.get('model_variant_label') != MODEL_VARIANT_LABEL):
        raise RuntimeError(f'model_variant entry {entry["label"]!r} without the label {MODEL_VARIANT_LABEL!r} '
                           f'at spec and entry level')
    # Addendum 34 (W33): a flexibility-price multiplier != 1.0 runs only under its explicit label, at spec AND entry
    # level; 1.0 (explicit) runs the override path with no label (it is the baseline evaluation, same key).
    flex_m = validate_flex_price_multiplier(entry.get('flex_price_multiplier'))
    if flex_price_multiplier_in_key(flex_m) is not None and (spec.get('flex_price_label') != FLEX_PRICE_LABEL
                                                             or entry.get('flex_price_label') != FLEX_PRICE_LABEL):
        raise RuntimeError(f'flex_price_multiplier entry {entry["label"]!r} without the label {FLEX_PRICE_LABEL!r} '
                           f'at spec and entry level')
    post_request = entry.get('post_certification')
    post_checklist = None
    if post_request:
        post_checklist = assert_post_certification_capture_paths()
        if post_request.get('reference'):
            post_checklist['reference_hashes_verified_before_run'] = verify_reference_unchanged(
                post_request['reference'])
    ids = entry['working_dir_ids']
    for eid in ids.values():
        if os.path.exists(os.path.join(G.O.WORK_DIR, eid)):
            raise RuntimeError(f'working dir id already used (never reusable): {eid}')
    # Addendum 39 (W47): a declared derived instance is installed as the oracle baseline FIRST -- before
    # `instance_investment_years` and the floor-row precheck read it -- and a premium entry is validated.
    derived = validate_derived_instance(spec['configuration'].get('derived_instance'))
    premium = validate_interface_deviation_premium(entry.get('interface_deviation_premium'))
    # W90 (Addendum 48): option (b), only when the entry declares it (never keyed).
    release_bk = validate_release_solution_bookkeeping(entry.get('release_solution_bookkeeping'))
    # W98 (Addendum 51): the certification continuation, only when the entry declares it (it enters the eval key);
    # its preconditions (cap = N + continuation cycles, tail and AA on, production signatures, the replay reference's
    # hash) are asserted here, before any solve.
    continuation = validate_certification_continuation(entry.get('certification_continuation'))
    continuation_checklist = None
    if continuation is not None:
        import p515_s53_w98_continuation_hooks as W98C
        continuation_checklist = W98C.assert_continuation_preconditions(continuation, spec, tail_checklist, aa_on)
        progress['certification_continuation_checklist'] = continuation_checklist
    derived_installed = None
    if derived is not None:
        derived_installed = install_derived_instance(derived, eval_dir)
        progress['derived_instance_installed'] = derived_installed
        print(f"[S44-CHILD] derived instance {derived['instance_label']} installed: case sha256 "
              f"{derived_installed['case_sha256_in_child']} scenario checksum "
              f"{derived_installed['scenario_checksum_in_child']}", flush=True)
    capture_multiscenario = derived is not None or premium is not None
    # W64 (Addendum 40 ruling 1): the alpha-row capture -- rule eleven asserted here, before any solve.
    alpha_row_checklist = assert_alpha_row_capture_paths() if capture_multiscenario else None
    label = spec['configuration']['arm_label']
    investment_map = investment_map_from_canonical(entry['canonical'])
    # Addendum 27 (W14): the candidate carries its own SINGLE cohort year. It must be one of
    # THIS instance's investment years, read from the shared-ESS data rather than compared to
    # the 2025 literal `G.N.INVEST_YEAR`; anything else still raises before any solve.
    investment_year = investment_year_from_canonical(entry['canonical'])
    instance_years = instance_investment_years()
    if investment_year not in instance_years:
        raise RuntimeError(f'candidate investment year {investment_year} is not one of the instance '
                           f'investment years {instance_years}')

    _cc, floor_rows_by_node, _fc = _build_floor_rows(ids['precheck'])
    paths = {
        'recourse_jump': os.path.join(eval_dir, 'recourse_jump_sidecar_baseline.jsonl'),
        'ess_stride': os.path.join(eval_dir, 'ess_entry_stride_baseline.jsonl'),
        'floor': os.path.join(eval_dir, 'soh_floor_sidecar_baseline.jsonl'),
        'pf_stride': os.path.join(eval_dir, f'pf_entry_stride_{label}.jsonl'),
        'exempt': os.path.join(eval_dir, f'ess_exempt_until_state_{label}.jsonl'),
    }
    for p in list(paths.values()) + [os.path.join(eval_dir, f) for f in (
            POST_CERTIFICATION_FILE, HULL_BOUND_DETAIL_FILE, AA_SIDECAR_FILE, 'certified_models.pkl',
            RESPONSE_TERMINAL_FILE, PER_CYCLE_RESPONSE_FILE, ACTIVATION_READBACK_FILE, INIT_IDENTITY_FILE,
            NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE, CONVERGENCE_DEPTH_APPEND_EVENTS_FILE,
            NETWORK_IPOPT_SOLVE_RECORDS_FILE, CONVERGENCE_DEPTH_TAIL_STATE_FILE)]:
        if os.path.exists(p):
            raise RuntimeError(f'refusing to overwrite existing artifact: {p}')
    # W85 (Planner ruling Q3 on W84): the per-round append; its first line is the tail checklist, before any solve.
    appender = ConvergenceDepthAppender(eval_dir, tail_checklist)
    progress['convergence_depth_appender'] = appender
    hooks_ref = {}   # W64: the alpha-row hook control, closed by the post-run hook

    def post_run_hook(planning, sed, models, rows, report, out_dir, label, state=None, optimization_results=None,
                      primal_evolution=None):
        run_hooks = hooks_ref.get('control')
        if run_hooks is not None:   # W64: the run is over; the per-cycle / activation wrappers pass through from here
            run_hooks.close()
        # W84 (Addendum 46 ruling 7, Q2): per-solve floor status + tail state, persisted FIRST, for every evaluation.
        try:
            holder['convergence_depth_capture'] = persist_convergence_depth_capture(state, eval_dir)
        except Exception as error:  # noqa: BLE001 -- recorded loudly; the child exits 2 after writing its record
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            holder['convergence_depth_capture'] = {'status': 'error', 'error': f'{type(error).__name__}: {error}',
                                                   'traceback': tb}
        holder['convergence_depth_tail_state_check'] = convergence_depth_tail_state_check(tail_checklist, state)
        # W85: the per-round append must reproduce the end-of-run write exactly; then it is sealed.
        try:
            holder['convergence_depth_append'] = appender.reconcile(holder['convergence_depth_capture'], state)
        except Exception as error:  # noqa: BLE001 -- recorded loudly; the child exits 2 after writing its record
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            holder['convergence_depth_append'] = {'ok': False, 'error': f'{type(error).__name__}: {error}',
                                                  'traceback': tb}
        appender.seal()
        report['s34_recourse_jump_sidecar_path'] = os.path.relpath(paths['recourse_jump'], REPO)
        report['s34_ess_entry_stride_sidecar_path'] = os.path.relpath(paths['ess_stride'], REPO)
        report['s35ref_soh_floor_sidecar_path'] = os.path.relpath(paths['floor'], REPO)
        report['s38_pf_entry_stride_sidecar_path'] = os.path.relpath(paths['pf_stride'], REPO)
        report['s39_ess_exempt_until_state_sidecar_path'] = os.path.relpath(paths['exempt'], REPO)
        G.write_boyd_terminal_s35ref(planning, sed, models, rows, report, out_dir, label,
                                     floor_rows_by_node=floor_rows_by_node, floor_sidecar_path=paths['floor'])
        caps = sed.get_updated_capacities(models['esso'])
        holder['published_caps'] = {str(n): {str(y): v for y, v in per_year.items()} for n, per_year in caps.items()}
        if model_variant is not None:  # W20: read back from CLONES of the run's own ESSO models; read-only capture
            holder['model_variant_readback_terminal'] = model_variant_readback_models(
                models['esso'], sed, model_variant, investment_year, clone=True)
            holder['ageing_trajectory_terminal'] = ageing_trajectory_terminal(models['esso'], sed)
        if ess_ageing is not None:  # W21: read back from CLONES of the run's own ESSO models; read-only capture
            if model_variant is None:
                holder['ess_ageing_readback_terminal'] = ess_ageing_readback_models(
                    models['esso'], sed, ess_ageing, investment_year, clone=True)
            if 'ageing_trajectory_terminal' not in holder:
                holder['ageing_trajectory_terminal'] = ageing_trajectory_terminal(models['esso'], sed)
        if flex_m is not None:  # W33: read back from the run's OWN DSO models; read-only
            holder['flex_price_readback_terminal'] = flex_price_readback_run_models(
                models['dso'], planning, holder['_flex_price_original_arrays'], flex_m)
        st = state or {}
        holder['peak_rss_ru_maxrss_production'] = st.get('peak_rss_ru_maxrss')
        holder['peak_rss_platform_units'] = st.get('peak_rss_platform_units')
        # W47: a derived-instance campaign serializes its children's TERMINAL PHASE (capture, workbook,
        # post-certification) -- see `acquire_terminal_phase_lock`; released in the `finally` below.
        terminal_lock_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(eval_dir))),
                                          TERMINAL_PHASE_LOCK_NAME)
        terminal_lock = acquire_terminal_phase_lock(terminal_lock_path) if capture_multiscenario else None
        if terminal_lock is not None:
            holder['terminal_phase_lock'] = terminal_lock
            print(f"[S44-CHILD] terminal-phase lock: {terminal_lock['status']} after {terminal_lock['waited_s']:.0f}s",
                  flush=True)
        try:
            _terminal_phase(planning, models, rows, report, state, st, optimization_results, primal_evolution)
        finally:
            if terminal_lock is not None:
                holder['terminal_phase_lock']['released'] = release_terminal_phase_lock(terminal_lock_path,
                                                                                        terminal_lock)

    def _terminal_phase(planning, models, rows, report, state, st, optimization_results, primal_evolution):
        if capture_multiscenario:  # W47: zero solves, on the terminal models, BEFORE any post-certification step
            try:
                holder['multiscenario_terminal'] = write_multiscenario_terminal(planning, models, state, eval_dir,
                                                                                premium=premium)
            except Exception as error:  # noqa: BLE001 -- recorded loudly; the evaluation itself stands
                tb = traceback.format_exc()
                print(tb, file=sys.stderr, flush=True)
                holder['multiscenario_terminal'] = {'status': 'error', 'error': f'{type(error).__name__}: {error}',
                                                    'traceback': tb}
            try:   # W64: dual-based curtailment + coordination state, on the same terminal models, before the workbook
                holder['response_terminal'] = write_response_terminal(planning, models, eval_dir,
                                                                      optimization_results=optimization_results)
            except Exception as error:  # noqa: BLE001 -- recorded loudly; the evaluation itself stands
                tb = traceback.format_exc()
                print(tb, file=sys.stderr, flush=True)
                holder['response_terminal'] = {'status': 'error', 'error': f'{type(error).__name__}: {error}',
                                               'traceback': tb}
            print(f"[S44-CHILD] response terminal capture: {(holder['response_terminal'] or {}).get('status')} "
                  f"(entries {(holder['response_terminal'] or {}).get('n_curtailment_entries_above_tol')}, "
                  f"coordination blocks {(holder['response_terminal'] or {}).get('n_coordination_blocks')})", flush=True)
            try:
                holder['operational_workbook'] = write_operational_workbook(
                    planning, models, optimization_results, primal_evolution, state, report.get('wall_clock_s'))
            except Exception as error:  # noqa: BLE001 -- recorded loudly; the evaluation itself stands
                tb = traceback.format_exc()
                print(tb, file=sys.stderr, flush=True)
                holder['operational_workbook'] = {'status': 'error', 'error': f'{type(error).__name__}: {error}',
                                                  'traceback': tb}
            print(f"[S44-CHILD] multi-scenario terminal capture: "
                  f"{(holder['multiscenario_terminal'] or {}).get('status')} "
                  f"(checks pass: {(holder['multiscenario_terminal'] or {}).get('all_checks_pass')}); workbook: "
                  f"{(holder['operational_workbook'] or {}).get('status')}", flush=True)
        if aa_on:
            import p515_s43_aa_run as S43  # its sidecar builder, BY IMPORT, unchanged
            S43._build_aa_per_cycle_sidecar(rows, os.path.join(eval_dir, AA_SIDECAR_FILE))
            holder['aa_sidecar'] = {'path': os.path.relpath(os.path.join(eval_dir, AA_SIDECAR_FILE), REPO),
                                    **aa_sidecar_summary(rows)}
        if post_request:
            t_pc = time.time()
            try:
                pc, _detail = run_post_certification(planning=planning, models=models, rows=rows, report=report,
                                                     state=state, spec=spec, entry=entry, eval_dir=eval_dir)
            except Exception as error:  # noqa: BLE001 -- recorded loudly; the evaluation itself stands
                tb = traceback.format_exc()
                print(tb, file=sys.stderr, flush=True)
                pc = {'requested': post_request, 'status': 'error', 'evaluated': False,
                      'error': f'{type(error).__name__}: {error}', 'traceback': tb}
            pc['runtime_s'] = time.time() - t_pc
            pc['production_peak_rss_before_step'] = st.get('peak_rss_ru_maxrss')
            pc['process_ru_maxrss_after_step'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            _write_once_json(os.path.join(eval_dir, POST_CERTIFICATION_FILE), pc)
            holder['post_certification'] = pc
            print(f"[S44-CHILD] post-certification: status={pc.get('status')} "
                  f"reason={pc.get('skip_reason') or pc.get('error')}", flush=True)

    t0 = time.time()
    from contextlib import nullcontext
    derived_identity = derived_instance_identity(derived) if derived is not None else None
    alpha_row_hooks = (alpha_row_run_hooks(eval_dir, entry['label'], entry['key'], derived_identity,
                                           float((premium or {}).get('alpha') or 0.0), holder)
                       if capture_multiscenario else nullcontext())
    release_counter = _ReleaseBookkeepingCallCounter(holder) if release_bk is not None else nullcontext()  # W90
    # W98: entered FIRST, so its wrappers sit directly on production and every capture hook below wraps them (the tail
    # appender and the s39 penalty sidecar record the held values).
    continuation_cm = (W98C.continuation_hooks(eval_dir, continuation, holder, int(spec['cap']))
                       if continuation is not None else nullcontext())
    with continuation_cm, \
         G.s38_pf_capture_hooks(paths['recourse_jump'], paths['ess_stride'], paths['floor'],
                                paths['pf_stride'], floor_rows_by_node, stride=1), \
         G.s39_exempt_until_capture_hooks(paths['exempt']), \
         convergence_depth_append_hooks(appender), \
         release_counter, \
         alpha_row_hooks as hook_control:
        hooks_ref['control'] = hook_control
        report, report_path = G.run_admm_arm(
            label, eval_dir, k_override=None, investment_map=investment_map,
            num_max_iters_override=int(spec['cap']), eval_id=ids['run'], apply_rho=False,
            full_diagnostics_in_rows=True, post_run_hook=post_run_hook,
            pre_solve_hook=_config_hook_factory(spec, holder, overrides=eff_overrides, model_variant=model_variant,
                                                investment_year=investment_year,
                                                expected_floor_rows=floor_rows_by_node,
                                                flex_price_multiplier=flex_m,
                                                interface_deviation_premium=premium,
                                                release_solution_bookkeeping=release_bk),
            investment_year=investment_year)
    run_wall = time.time() - t0

    rows = report.get('cycle_trajectory') or []
    per_cycle_path = os.path.join(eval_dir, 'per_cycle_record.jsonl')
    if os.path.exists(per_cycle_path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {per_cycle_path}')
    # W64: derived-instance / premium evaluations merge the per-cycle RESPONSE record (by cycle) into the standard
    # per-cycle record; every other evaluation writes exactly its pre-W64 fields.
    response_by_cycle = read_per_cycle_response(eval_dir) if capture_multiscenario else {}
    per_cycle_fields = PER_CYCLE_RECORD_FIELDS if capture_multiscenario else PER_CYCLE_TRAJECTORY_FIELDS
    with open(per_cycle_path, 'w') as handle:
        for r in rows:
            merged = dict(r)
            if capture_multiscenario:
                merged.update(response_by_cycle.get(int(r.get('cycle')), {'response_captured': False,
                                                                          'response_capture_error': 'no response line'}))
            handle.write(json.dumps({k: merged.get(k) for k in per_cycle_fields}, default=_json_default) + '\n')

    with open(os.path.join(eval_dir, 'component_levels_terminal.json')) as handle:
        component_levels = json.load(handle)
    with open(os.path.join(eval_dir, 'boyd_terminal.json')) as handle:
        boyd_terminal = json.load(handle)
    self_ru = resource.getrusage(resource.RUSAGE_SELF)
    children_ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss = {
        'units': 'bytes on macOS/BSD, kilobytes on Linux (ru_maxrss)',
        'child_python_process_ru_maxrss': self_ru.ru_maxrss,
        'production_state_peak_rss_ru_maxrss': holder.get('peak_rss_ru_maxrss_production'),
        'solver_subprocesses_max_ru_maxrss': children_ru.ru_maxrss,
        'semantics': ('child_python_process = RUSAGE_SELF of the evaluation process at record time '
                      '(the evaluation\'s own peak); production_state = the same measure taken by '
                      'production at the end of run_operational_planning; solver_subprocesses = '
                      'RUSAGE_CHILDREN max over the IPOPT executables this evaluation launched; when a '
                      'post-certification step ran, child_python_process includes it (the ADMM run alone is '
                      'production_state)'),
    }
    wall = {'child_process_s': time.time() - started, 'run_admm_arm_s': run_wall,
            'run_admm_arm_reported_wall_clock_s': report.get('wall_clock_s')}
    variant_extra = {}
    if model_variant is not None:  # W20: only for variant entries, so every other record keeps its format
        variant_extra = {
            'model_variant': model_variant, 'model_variant_label': MODEL_VARIANT_LABEL,
            'model_variant_applied_in_child': holder.get('model_variant_applied'),
            'model_variant_readback_pre_run': holder.get('model_variant_readback_pre_run'),
            'model_variant_readback_terminal': holder.get('model_variant_readback_terminal'),
            'ageing_trajectory_terminal': holder.get('ageing_trajectory_terminal'),
        }
    if ess_ageing is not None:  # W21: only for declared specs, so every other record keeps its format
        variant_extra.update({
            'ess_ageing_baseline': ess_ageing,
            'ess_ageing_baseline_label': spec['configuration'].get('ess_ageing_baseline_label'),
            'ess_params_sha256_in_child': ess_params_sha256_in_child,
            'ess_ageing_verified_pre_run': holder.get('ess_ageing_verified_pre_run'),
            'ess_ageing_readback_terminal': holder.get('ess_ageing_readback_terminal'),
            'ageing_trajectory_terminal': holder.get('ageing_trajectory_terminal'),
        })
    if 'flex_price_multiplier' in entry:  # W33: only for flexibility-price entries, so every other record keeps its format
        variant_extra.update({
            **_flex_price_record_fields(entry),
            'flex_price_applied_in_child': holder.get('flex_price_applied'),
            'flex_price_readback_terminal': holder.get('flex_price_readback_terminal'),
        })
    if capture_multiscenario:  # W47: only for derived-instance specs / premium entries, so every other record keeps its format
        ms = holder.get('multiscenario_terminal') or {'status': 'not_reached'}
        variant_extra.update({
            **_derived_record_fields(spec, entry),
            'derived_instance_installed_in_child': derived_installed,
            'derived_instance_checks_in_child': holder.get('derived_instance_checks'),
            'interface_deviation_premium_applied_in_child': holder.get('interface_deviation_premium_applied'),
            'multiscenario_terminal': {k: v for k, v in ms.items() if k != 'traceback'},
            'multiscenario_terminal_error_traceback': ms.get('traceback'),
            'operational_workbook': holder.get('operational_workbook') or {'status': 'not_reached'},
            'terminal_phase_lock': holder.get('terminal_phase_lock'),
            'sigma_calibration': ms.get('sigma_calibration'),
            # W64 (Addendum 40 ruling 1): the alpha-row capture
            'alpha_row_capture_checklist_asserted_before_run': alpha_row_checklist,
            'activation_readback': holder.get('activation_readback') or {'status': 'not_reached'},
            'initialisation_identity': holder.get('initialisation_identity') or {'status': 'not_reached'},
            'response_terminal': {k: v for k, v in (holder.get('response_terminal') or {'status': 'not_reached'}).items()
                                  if k != 'traceback'},
            'response_terminal_error_traceback': (holder.get('response_terminal') or {}).get('traceback'),
            'per_cycle_response': {'path': os.path.relpath(os.path.join(eval_dir, PER_CYCLE_RESPONSE_FILE), REPO),
                                   'n_lines': len(response_by_cycle), 'n_trajectory_rows': len(rows),
                                   'n_captured': sum(1 for v in response_by_cycle.values()
                                                     if v.get('response_captured')),
                                   'cycles_match': sorted(response_by_cycle) == [int(r.get('cycle')) for r in rows]},
        })
    if release_bk is not None:  # W90: only for entries declaring option (b), so every other record keeps its format
        variant_extra.update({
            'release_solution_bookkeeping': release_bk,
            'release_solution_bookkeeping_applied_in_child': holder.get('release_solution_bookkeeping_applied'),
            'release_solution_bookkeeping_calls': holder.get('release_solution_bookkeeping_calls'),
        })
    if continuation is not None:  # W98: only for entries declaring the continuation, so every other record keeps its format
        variant_extra.update({
            'certification_continuation': continuation,
            'certification_continuation_checklist_asserted_before_run': continuation_checklist,
            'certification_continuation_summary': holder.get('certification_continuation'),
        })
    record = build_evaluation_record(
        spec=spec, spec_path=spec_path, spec_sha256=args.spec_sha256, entry=entry, report=report,
        component_levels=component_levels,
        floor_terminal=boyd_terminal.get('soh_floor_multiplier_and_efc_per_cohort_year_terminal'),
        published_caps=holder.get('published_caps'), peak_rss=peak_rss, wall=wall, eval_dir=eval_dir,
        extra={
            'record_capture_checklist_asserted_before_run': capture_checklist,
            # W84 (Addendum 46 ruling 7): tail declaration / in force / returned, and the persisted floor-status capture
            'convergence_depth_tail_checklist_asserted_before_run': tail_checklist,
            'convergence_depth_tail_applied_in_child': holder.get('convergence_depth_tail_applied'),
            'convergence_depth_tail_state_check': holder.get('convergence_depth_tail_state_check'),
            'convergence_depth_capture': (holder.get('convergence_depth_capture') or {'status': 'not_reached'}),
            'convergence_depth_per_round_append': (holder.get('convergence_depth_append')
                                                   or {'ok': False, 'status': 'not_reached'}),
            'eval_key': _entry_eval_key(entry),
            'evaluation_overrides_effective': eff_overrides,
            'post_certification_capture_checklist_asserted_before_run': post_checklist,
            'post_certification': post_certification_summary(holder.get('post_certification')),
            'post_certification_path': (os.path.relpath(os.path.join(eval_dir, POST_CERTIFICATION_FILE), REPO)
                                        if holder.get('post_certification') is not None else None),
            'aa_per_cycle': holder.get('aa_sidecar'),
            'configuration_checks_in_child': holder.get('configuration_checks'),
            'overrides_applied_in_child': holder.get('overrides_applied'),
            'anderson_acceleration_effective_in_child': holder.get('anderson_acceleration_effective'),
            'case_file_sha256_in_child': case_file_sha256_in_child,
            'thread_caps_seen_by_child': env_caps,
            'PYTHONHASHSEED_in_child': os.environ.get('PYTHONHASHSEED'),
            'nlp_solver_path_in_child': os.environ.get('NLP_SOLVER_PATH'),
            'campaign_lock_seen_by_child': lock_content,
            'child_pid': os.getpid(), 'parent_pid': os.getppid(),
            'report_path': os.path.relpath(report_path, REPO),
            'per_cycle_record_path': os.path.relpath(per_cycle_path, REPO),
            **variant_extra,
        })
    _write_once_json(os.path.join(eval_dir, 'evaluation_record.json'), record)
    manifest = {}
    for root, _dirs, files in os.walk(eval_dir):
        for fname in sorted(files):
            if fname in ('child_stdout.log', 'child_stderr.log', 'exit_code.txt', 'wait4_rusage.json'):
                continue  # still being written by the parent; the campaign manifest hashes them
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = sha256_file(fpath)
    _write_once_json(os.path.join(eval_dir, 'child_manifest_sha256.json'), manifest)
    print(f"[S44-CHILD] {entry['label']}: status={record['status']} cycles={record['cycles_run']} "
          f"certified_cost={record['certified_cost']} bar={record['bar']['value']} "
          f"peak_rss={self_ru.ru_maxrss}")
    capture_error = capture_multiscenario and any(
        (holder.get(k) or {}).get('status') != 'written' for k in ('multiscenario_terminal', 'operational_workbook',
                                                                   'response_terminal'))
    convergence_depth_error = (
        (holder.get('convergence_depth_capture') or {}).get('status') != 'written'
        or not (holder.get('convergence_depth_tail_state_check') or {}).get('match')
        or not (holder.get('convergence_depth_append') or {}).get('ok'))
    continuation_error = continuation is not None and not (holder.get('certification_continuation') or {}).get('ok')
    return {'post_certification_error': (holder.get('post_certification') or {}).get('status') == 'error',
            'multiscenario_capture_error': capture_error,
            'convergence_depth_capture_error': convergence_depth_error,
            'certification_continuation_error': continuation_error}


def main_child(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', action='store_true', required=True)
    parser.add_argument('--campaign-root', required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--eval-key', required=True)
    parser.add_argument('--stub-mode', default=None, choices=(None, 'ok', 'fail'))
    parser.add_argument('--stub-sleep-s', default='2')
    parser.add_argument('--stub-alloc-mb', default='64')
    parser.add_argument('--lock-path', default=CAMPAIGN_LOCK_PATH)
    args = parser.parse_args(argv)
    started = time.time()
    env_caps = _child_verify_env()
    spec_path, spec = load_frozen_spec(args.campaign_root, args.spec_sha256)
    lock_content = verify_child_lock(args.spec_sha256, lock_path=args.lock_path)
    entry = next((e for e in spec['candidates'] if _entry_eval_key(e) == args.eval_key), None)
    if entry is None:
        raise SystemExit(f'CHILD REFUSES: eval key {args.eval_key} not in the frozen spec')
    eval_dir = os.path.join(args.campaign_root, 'evals', entry['eval_dir'])
    if not os.path.isdir(eval_dir) or os.path.exists(os.path.join(eval_dir, 'evaluation_record.json')):
        raise SystemExit(f'CHILD REFUSES: eval dir missing or already holds a record: {eval_dir}')
    print(f"[S44-CHILD] pid={os.getpid()} ppid={os.getppid()} label={entry['label']} key={entry['key'][:16]} "
          f"caps={env_caps} PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}", flush=True)
    progress = {}
    try:
        if args.stub_mode is not None:
            _child_stub(args, spec, spec_path, entry, eval_dir, lock_content, env_caps, started)
        else:
            outcome = _child_real(args, spec, spec_path, entry, eval_dir, lock_content, env_caps, started,
                                  progress=progress)
            if outcome and outcome.get('post_certification_error'):
                print('[S44-CHILD] post-certification step FAILED (recorded in post_certification.json); '
                      'exiting 2', file=sys.stderr, flush=True)
                sys.exit(2)
            if outcome and outcome.get('multiscenario_capture_error'):  # W47
                print('[S44-CHILD] multi-scenario terminal capture / workbook FAILED (recorded in '
                      'evaluation_record.json); exiting 2', file=sys.stderr, flush=True)
                sys.exit(2)
            if outcome and outcome.get('convergence_depth_capture_error'):  # W84
                print('[S44-CHILD] convergence-depth capture FAILED or the returned tail state does not match the '
                      'declaration (recorded in evaluation_record.json); exiting 2', file=sys.stderr, flush=True)
                sys.exit(2)
            if outcome and outcome.get('certification_continuation_error'):  # W98
                print('[S44-CHILD] certification continuation summary not ok (recorded in evaluation_record.json); '
                      'exiting 2', file=sys.stderr, flush=True)
                sys.exit(2)
    except SystemExit:
        raise
    except BaseException as error:  # noqa: BLE001 -- recorded as a barrier with its cause, then exit 1
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        # W85: the round in flight is drained and appended before the record is written (never raises).
        appender = progress.get('convergence_depth_appender')
        failure_drain = (appender.drain_on_failure(f'{type(error).__name__}: {error}') if appender is not None
                         else {'status': 'skipped', 'why': 'no appender (the failure preceded it)'})
        record_path = os.path.join(eval_dir, 'evaluation_record.json')
        if not os.path.exists(record_path):
            _write_once_json(record_path, {
                'schema': RECORD_SCHEMA, 'campaign_id': spec['campaign_id'],
                'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': args.spec_sha256,
                'candidate_label': entry['label'], 'candidate_canonical': entry['canonical'],
                'candidate_key': entry['key'], 'status': 'error', 'barrier': True,
                'barrier_cause': f'{type(error).__name__}: {error}', 'traceback': tb,
                'wall_time_s': {'child_process_s': time.time() - started},
                # Addendum 27 (W5): same schema as a success record; None when the failure came first.
                'anderson_acceleration_effective_in_child': (
                    (progress.get('holder') or {}).get('anderson_acceleration_effective')),
                'case_file_sha256_in_child': progress.get('case_file_sha256_in_child'),
                # W84: whether the tail was enabled for this run, on every path (None when the failure came first).
                'convergence_depth_tail_checklist_asserted_before_run': progress.get(
                    'convergence_depth_tail_checklist'),
                # W85: what the per-round append preserved up to the failure (None when the failure preceded it).
                'convergence_depth_failure_drain': failure_drain,
                'convergence_depth_per_round_append_recovered': recover_convergence_depth_append(eval_dir),
                # W20: a model-variant entry's record carries the variant and its label on every path.
                **({'model_variant': entry['model_variant'], 'model_variant_label': MODEL_VARIANT_LABEL,
                    'model_variant_applied_in_child': (progress.get('holder') or {}).get('model_variant_applied')}
                   if entry.get('model_variant') is not None else {}),
                # W21: a declared-ESS-ageing spec's record carries the declaration, its label and the file hash seen.
                **({'ess_ageing_baseline': spec['configuration']['ess_ageing_baseline'],
                    'ess_ageing_baseline_label': spec['configuration'].get('ess_ageing_baseline_label'),
                    'ess_params_sha256_in_child': progress.get('ess_params_sha256_in_child'),
                    'ess_ageing_verified_pre_run': (progress.get('holder') or {}).get('ess_ageing_verified_pre_run')}
                   if spec['configuration'].get('ess_ageing_baseline') is not None else {}),
                # W33: a flexibility-price entry's record carries the multiplier (+ label) and what was applied.
                **({**_flex_price_record_fields(entry),
                    'flex_price_applied_in_child': (progress.get('holder') or {}).get('flex_price_applied')}
                   if 'flex_price_multiplier' in entry else {}),
                # W47: a derived-instance spec / premium entry's record carries its declaration and what was done.
                **({**_derived_record_fields(spec, entry),
                    'derived_instance_installed_in_child': progress.get('derived_instance_installed'),
                    'interface_deviation_premium_applied_in_child': (
                        (progress.get('holder') or {}).get('interface_deviation_premium_applied'))}
                   if _derived_record_fields(spec, entry) else {}),
                # W90: an entry declaring option (b) carries it and what was applied, on every path.
                **({'release_solution_bookkeeping': entry['release_solution_bookkeeping'],
                    'release_solution_bookkeeping_applied_in_child': (
                        (progress.get('holder') or {}).get('release_solution_bookkeeping_applied'))}
                   if 'release_solution_bookkeeping' in entry else {}),
                # W98: an entry declaring the continuation carries it, its checklist and its summary, on every path.
                **({'certification_continuation': entry['certification_continuation'],
                    'certification_continuation_checklist_asserted_before_run': progress.get(
                        'certification_continuation_checklist'),
                    'certification_continuation_summary': (progress.get('holder') or {}).get(
                        'certification_continuation')}
                   if 'certification_continuation' in entry else {}),
            })
        sys.exit(1)


if __name__ == '__main__':
    if '--child' in sys.argv[1:]:
        main_child(sys.argv[1:])
    else:
        raise SystemExit('p515_s44_campaign_harness.py is a library + child entry point; '
                         'campaigns are launched by their own script (e.g. p515_s44_gate.py)')
