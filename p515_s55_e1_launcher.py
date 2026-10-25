"""
P5.15 Addendum 83 ruling 3 (i), (v), (vi) -- task W183b: the E1 SEARCH LAUNCHER (matched degradation-aware and

================================================================================
v2 (W183g (b), PLANNER_BRIEF_2026-09-13.md Addendum 85 rulings 2, 3 and 4 (iv); v1 = sha256 320d503c..., cb36b22e)
================================================================================
  E1 POLL RULE B+ (ruling 2). The search loop is the launcher's OWN `run_continuation_e1`: a copy of
  `p515_s53_f2_certificate.run_continuation` (the committed module is not edited) with ONE inserted step, between
  `build_poll` and the completion-cap refusal: when fewer than n + 1 = 8 distinct feasible poll points remain after
  rounding and snapping and the incumbent's completion has MORE than 30 feasible points, the Halton index advances by
  one and the 2n directions are re-drawn, rounded and snapped (`C.build_poll`, unchanged, by reference), at most
  ADVANCE_WINDOW = 10 consecutive indices (the refused one included); the first index with >= 8 distinct feasible
  points is polled; the skipped indices and their snap tables are written into the poll record
  (`halton_indices_skipped`, `advance_rule` -- the only new record fields); if no index in the window yields 8 points,
  STOP_FOR_REVIEW_completion_cap as frozen (the terminating record is the record of the first examined index). A
  completion of <= 30 points is used as frozen. The Halton index is explicit state `k_next`: each launched poll
  examines k_next, k_next + 1, ... and polls at k_p; then k_next = k_p + 1; skipped indices are not launched polls
  (MAX_POLLS counts launched polls); `poll_index` stays the Halton k. Everything else is the frozen variant B by
  reference: `C.build_poll`, `C._certificate`, `PB.classify`, `PB.resolution`, `PB.BARRIER_STOP_*` (A7 confirmed,
  ruling 3), the acceptance threshold, the budget check after the cap check at the chosen index. k0 = 0 (t = 17) in
  both arms. The rule text is B_PLUS_RULE, recorded in every spec (extra.poll_design).
  DECLARED ROOTS (ruling 4 iv): `P515S55/e1_2x` (phases seeds -- all seven --, search_N, search_A; phase `cross`
  belongs to the fixed-plan launcher and is never created or read here) and the batch-gate root
  `P515S55/e1_2x_batch4` (ONE seeds phase: unit_A_2x, unit_N_2x, n7_05_2_A_2x, n7_05_2_N_2x at concurrency 4; no
  search phase; its records never seed an arm -- the cache is root-scoped). --freeze / --run refuse any other root.
  The dry-run gate of W183g (b) is `p515_s55_w183g_b_e1_v2_dryrun.py`.

v1 text follows (the search paragraph's "run_continuation BY IMPORT" is replaced by the v2 paragraph above).

degradation-neutral searches at twice the base flexibility price; STEP7_SIMULATION_PLAN_2026-10-25.md section 5 and
the section 11 amendments; Addendum 82 v2 ruling 1). HARNESS CODE: no production module is touched, and the campaign
harness `p515_s44_campaign_harness.py` is used exactly as committed (no harness change was needed).

WHAT E1 IS (plan section 5.1 / 5.2; Addendum 83 ruling 3)
  Two arms at flexibility price m = 2 on the final code head, variant B search throughout:
    A  baseline ageing: the declared ESS ageing baseline (C2 + phi_cal 0.985 + soh_min 0.70). NO `model_variant` is
       passed for A: an absent model variant IS the baseline in the harness, and passing one would put a
       'model_variant' item into every A key and break the key space of every committed A cell (rows 35, 40 ...).
    N  no ageing: `model_variant` = {ageing_enabled False, available_energy_soh_point 'end', calendar_retention_per_year
       1.0, eol_retention_r 0.50} -- the T8 / ext-v3 `e_no_ageing` declaration (Addendum 83 ruling 3 i: 0.50, so the
       committed no-ageing cell of the reference unit, E.21 row 54, carries the same key at the same price).
  Every entry declares `flex_price_multiplier` = 2.0 together with its model (N: `model_variant`; A: the baseline, i.e.
  none); both are in the evaluation key (`H.evaluation_key`), and the harness writes the labels
  (`model_variant_label` on N entries and at spec level when the spec holds an N entry; `flex_price_label` on every
  entry and at spec level). `rho0_scale` and `nrf_option` are NEVER declared (E1 uses neither) -- asserted.
  Common seed set S (fixed by the plan): x = 0; node 7 0.25 MVA / 1 MWh 2025 (the reference unit); node 7 0.5 / 2
  2025; the historical 2x incumbent ref:5ca4f86c (2030, node 5 0.25/0.5 + node 7 1/3.5; E.21 row 40). Seed costs are
  evaluated under each model at the production exit; x = 0 is evaluated ONCE (under A) and its N seed is an explicit
  ALIAS (below) -- the Advisor's reading of "7 distinct seed keys". N^max = 30 new search evaluations per arm, at most
  10 polls. Cross-evaluations (certified under rule v6) are NOT run by this launcher (a re-settling declaration, a
  separate later task).

THE SEARCH (variant B; plan section 5.1; Addendum 70 ruling 6)
  The poll is `p515_s53_f2_certificate.build_poll` / `run_continuation` BY IMPORT, unchanged: at unit mesh (Delta = 1
  at every poll) the 2n = 14 OrthoMADS directions +-h_j (Halton t = 17 + k, k from 0 for EACH arm), rounded, every
  infeasible rounded point SNAPPED to the incumbent's unit frame, the admissible unit box (completion) added only when
  fewer than n + 1 = 8 distinct feasible poll points remain, the completion cap 30 REFUSES the poll (stop for review,
  never truncated); acceptance only when F(inc) - F(x) > max(bar_x + bar_inc, sigma_Q), sigma_Q = 18,449.66 EUR
  (PB.classify / PB.resolution); on success the next poll is again a unit poll; a poll whose new evaluations would
  exceed what remains of N^max is not launched (budget exhausted). The barrier stop rule is the inherited Phase B
  rule A7 (PB.BARRIER_STOP_PER_POLL = 2, PB.BARRIER_STOP_OVERALL = 3), unchanged. F(x) = I(x) + Q(x), Q =
  gross_operational_cost, contracted interface settlement excluded; salvage reported by the harness, excluded.
  Incumbent rule: the best production-exit F among the budget-feasible, certified seeds of the arm's model (ties: the
  lower I, then the label).

THE ROOT AND ITS PHASES
  --root names the E1 root (any directory strictly under data/SRP1/Results/P515S55/; spec v1 names P515S55/e1_2x and
  P515S55/e1_screen_1x; the 4-slot batch root is not yet named). Each phase is its own write-once campaign directory
  inside the root, `campaign_s55_<root>_<phase>`, with its own frozen spec:
    seeds      the seed evaluations (default all 7; `--seeds` restricts them, e.g. the four node-7 seeds of the batch
               gate at --concurrency 4), at the production exit;
    search_N   the arm-N search (spec = the whole admissible domain under N at 2x; cache = root-scoped, below);
    search_A   the arm-A search (spec = the whole admissible domain under A at 2x).
  An existing non-empty root is REFUSED for --dry-run and for the first phase (seeds): only launch logs
  (`*launch*.log`) may be in it. A search phase is refused unless the root holds exactly the completed seeds phase
  (spec + campaign_results.json + manifest), at most the completed other search phase, launch logs, and nothing else.

THE CACHE (root-scoped; STEP4 2.6; plan 5.1 "per arm, per ageing model, seeded only by that arm's seed evaluations")
  `load_seed_cache` reads ONE file: <root>/campaign_s55_<root>_seeds/campaign_results.json. No glob over the results
  tree, no committed record of any other stage, never the other arm's search. Enforced, and refused otherwise: the
  results file, its spec and every point's eval_dir resolve (realpath) inside the root; the seeds spec in that
  directory is unique, hashes to the results' `campaign_spec_sha256` and records this root (`extra.e1_root`) and this
  launcher; every point's eval key is recomputed from its canonical candidate under the ARM's model at 2x and must
  equal the recorded key and a key of that seeds spec; points of the other model are skipped (never cache); the
  manifest hashes of the points' evaluation records hold; the files are git-tracked and clean.

THE SEED ALIAS (x = 0 under N)
  {'reuse_of': <the x = 0 / A / 2x eval key>, 'justification': 'G0', 'g0_artifact_sha256': ...}: accepted only when the
  G0 artifact exists, hashes to its pin (--g0-sha256) and reports NL identity of x = 0 under A and N at 2x (ESSO NL on
  every node, network NL in both builders), with its own x = 0 A / N keys equal to this launcher's; at --freeze /
  --run its production modules must also equal the current production (the final-head G0 artifact of W183d). The alias
  is recorded in the seeds spec and results under `aliases`, never as an evaluation; in arm N's search cache it is an
  entry marked `alias` (source kind 'alias_G0') that the poll can hit, and it never appears among the arm's `points`.

PINS (ruling 3 vi), recorded in every spec this launcher freezes; --run refuses if any differs
  the production head (last commit touching the tracked root-level *.py not matching ^p\\d) and every production
  module's sha256; the sha256 of every harness / launcher file it runs (the campaign harness, the gates module, this
  launcher, p56a_oracle.py, the settling modules settling_criterion*.py) and of every repo module imported in this
  process; every tracked input under data/SRP1 outside Results/ and Diagrams/ (case, params and cost files named);
  the pinned W2 investment-cost table and the Phase A tables (sigma_Q).

CONCURRENCY (ruling 3 v)
  `--concurrency N` is frozen into the spec (the harness slot count and the poll batch size). Memory preflight (the
  S47 rule): hw.memsize - (wired + anonymous + compressor-occupied pages) x page size >= N x 2.75 GiB; recorded at
  --dry-run and --freeze (non-gating), GATING at --run.

WHAT IS REUSED BY IMPORT, AND WHAT IS NOT
  p515_s47_phase_b_record (PB): Lattice, unit_costs_from_w2, w2_cross_check, sigma_q_from_tables, degradation_clause,
  resolution / classify and the constants. p515_s53_f2_certificate (C): build_poll, run_continuation, _certificate and
  their record schemas (variant B). NOT used: their cache, key, pin, precondition, spec-validation, freeze and run
  functions (PB asserts no model variant at ~874-875 and ~1383; F2 Phase B at ~362-363 and ~851-852). They are
  replaced by TRIPWIRES at import: any call raises and is counted; the counts are verified at exactly 0.

MODES (repo root, canonical interpreter, attached, ALONE, both streams captured, never detached)
  --dry-run --root R --concurrency N --g0-artifact P --g0-sha256 S --scratch D
        ZERO SOLVES, no model build, pickle blocked: the seed key set under both models at 2x and its assertions (7
        distinct keys to evaluate, exactly 1 alias, alias target = x = 0 / A / 2x), the key table, the base-key
        comparison with the committed final-code cells (E.21 rows 35, 40, 54), the configuration-block equality with
        the W142 extension v6, the seeds spec frozen IN SCRATCH by the harness's own `freeze_campaign_spec` (labels,
        concurrency, preflight, pins), the domain keys, the cache-scoping tests (the real root, and planted roots in
        scratch), rule eleven, the preconditions (reported). Writes <R>/w183b_e1_dryrun.json and its manifest.
  --freeze --phase {seeds,search_N,search_A} --root R --concurrency N --g0-artifact P --g0-sha256 S [--seeds L,...]
        ZERO SOLVES: freezes and validates the phase's campaign spec.
  --run --phase ... --root R --spec-sha256 X [--g0-artifact P --g0-sha256 S]
        runs the frozen phase (seeds: every entry; search: the variant-B poll) through `H.evaluate`.
  Exit codes: 0 done (search: poll failure at unit mesh with its certificate holding); 2 search budget / max polls; 3
  STOP_FOR_REVIEW (barrier rule or completion cap); 1 any refusal, guard, tripwire, read-back or precondition failure.

EXACT COMMAND OF THE W183b DRY RUN (the only mode run by W183b):
  mkdir data/SRP1/Results/P515S55/w183b_e1_dryrun && NLP_SOLVER_PATH=/usr/local/bin/ipopt \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_e1_launcher.py --dry-run \\
      --root data/SRP1/Results/P515S55/w183b_e1_dryrun --concurrency 1 \\
      --g0-artifact data/SRP1/Results/P515S55/w183_nl_reference_pre/w183_nl_digest.json.gz \\
      --g0-sha256 cd58078ef0e0c297b50373c65e93b23fd449b46fb8cc13104be35841ae3b6e70 --scratch <outside the repo> \\
      > data/SRP1/Results/P515S55/w183b_e1_dryrun/launch.log 2>&1
"""

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183b E1 launcher parent (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W183b E1 launcher: pickle.load called -- the parent never unpickles')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W183b E1 launcher: pickle.loads called -- the parent never unpickles')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer that refuses string flags)
import p515_s44_campaign_harness as H  # noqa: E402  (stdlib only at import)
import p515_s47_phase_b_record as PB  # noqa: E402  (pure lattice / direction / classification primitives only)
import p515_s53_f2_certificate as C  # noqa: E402  (variant B poll: build_poll / run_continuation / _certificate)

F51 = C.F51
pickle.load, pickle.loads = _blocked_load, _blocked_loads

import pyomo.environ as pe  # noqa: E402  (already loaded by the solve guard; used for the model-build tripwire only)

MODEL_BUILD_COUNTS = {'ConcreteModel.__init__': 0}
_ORIGINAL_CONCRETE_INIT = pe.ConcreteModel.__init__


def _blocked_concrete_init(self, *_a, **_k):
    MODEL_BUILD_COUNTS['ConcreteModel.__init__'] += 1
    raise RuntimeError('W183b E1 launcher: a pyomo ConcreteModel was constructed in the parent -- no model build')


pe.ConcreteModel.__init__ = _blocked_concrete_init

# ---- tripwires on the committed launchers' cache / key / precondition / freeze / run functions (never used here) ----
TRIPWIRE_COUNTS = {}
TRIPWIRED = {
    'PB': (PB, ('build_inputs', 'load_cache_source', 'merge_cache', 'c3_exclusion', '_check_pins',
                '_check_baseline_launcher_declaration', '_check_case_file_loads_to_declaration', 'memory_preflight',
                'rule_eleven', 'initial_incumbent', 'x0_entry', 'validate_spec', 'spec_matches_baseline', 'freeze',
                'run', 'eval_key_of', 'make_key_of', 'expected_first_poll', 'run_mads', 'campaign_root')),
    'F51': (F51, ('eval_key_of_canonical', 'make_key_of', 'check_pins', 'shared_constants',
                  '_spec_declares_the_same_configuration', 'load_cache_source', 'baseline_exclusion', 'budget_facts',
                  'incumbent_matches_committed_ladder', 'rule_eleven', 'memory_preflight', 'build_inputs',
                  'campaign_root', 'validate_spec', 'freeze', '_point_result', '_cache_entry_from_point', 'run')),
    'C': (C, ('dry_run', 'load_phase_b_source', 'continuation_facts', 'baseline_exclusion', 'budget_facts',
              'rule_eleven', 'shared_constants', 'check_pins', 'build_inputs', 'campaign_root', 'validate_spec',
              'freeze', 'run', 'make_key_of')),
}


def _make_tripwire(tag):
    def _tripwire(*_a, **_k):
        TRIPWIRE_COUNTS[tag] = TRIPWIRE_COUNTS.get(tag, 0) + 1
        raise RuntimeError(f'W183b E1 launcher: {tag} is not used by this launcher (cache / key / precondition / '
                           f'freeze / run functions of the committed launchers assert no model variant)')
    return _tripwire


for _mod_tag, (_mod, _names) in TRIPWIRED.items():
    for _name in _names:
        if not callable(getattr(_mod, _name, None)):
            raise RuntimeError(f'tripwire target {_mod_tag}.{_name} not found')
        TRIPWIRE_COUNTS[f'{_mod_tag}.{_name}'] = 0
        setattr(_mod, _name, _make_tripwire(f'{_mod_tag}.{_name}'))

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRINGS = ('p515_s55_e1_launcher',)
STAGE = ('P5.15 Addendum 83 ruling 3 (i)/(v)/(vi), W183b -- E1: matched degradation-aware (A) and degradation-neutral '
         '(N) variant-B searches at flexibility price x 2 on the final code (STEP7 plan section 5)')
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 83 rulings 3 (i), (v), (vi) and 6; campaign spec v1; W183 order item W183b',
    'PLANNER_BRIEF_2026-09-13.md Addendum 82 v2 ruling 1 (E1); Addendum 70 ruling 6 (variant B)',
    'STEP7_SIMULATION_PLAN_2026-10-25.md sections 5.1, 5.2, 11',
    'P5_15_W182E_CODE_PATHS.md (Blocked 1: no committed search launcher accepts a model variant)',
    'P5_15_W183_0_NL_REFERENCE.md (G0 at x = 0; unit_N_1x key = the ext-v6 e_no_ageing base key)',
    'PLANNER_BRIEF_2026-09-13.md Addendum 85 ruling 2 (E1 poll rule B+), ruling 3 (A7 confirmed), ruling 4 (i), (iv) '
    '(k0 = 0; the 4-slot batch seeds and the roots); W183g order item (b) -- launcher v2',
]
OBJECTIVE_CONVENTION = ('Q(x) = certified gross_operational_cost (contracted interface settlement excluded); F(x) = I(x) '
                        '+ Q(x); salvage and net recourse recorded by the harness, excluded from F')
RESULTS_P55_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55')
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
DRY_RUN_FILE = 'e1_launcher_v2_dryrun.json'                      # v1: w183b_e1_dryrun.json
DRY_RUN_MANIFEST = 'e1_launcher_v2_dryrun_manifest_sha256.json'  # v1: w183b_e1_dryrun_manifest_sha256.json
LOG_RX = re.compile(r'^[A-Za-z0-9_.-]*launch[A-Za-z0-9_.-]*\.log$')

# ---- the frozen configuration: the W142 extension v6 block (p515_s53_w142_resettle_ext_v6_campaign.py
#      `configuration`, ~439-451) minus the re-settling declaration -- literals, checked against the committed specs ----
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
ESS_AGEING_BASELINE = {'calendar_life_years': 15, 'cycle_life_nominal': 10000, 'depth_of_discharge_nominal': 0.8,
                       'minimum_soh': 0.7, 'calendar_retention_per_year': 0.985,
                       'calibration': {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8,
                                       'eol_retention_r': 0.8}}
ESS_AGEING_BASELINE_LABEL = 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)'
CONVERGENCE_DEPTH_TAIL = {'enabled': True, 'compl_inf_tol': 1e-06}
ARM_LABEL = 's39_D'
CAP = 500
REQUIRED_CONSECUTIVE_CYCLES = 10
EXT_V6_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w142_resettle_ext_v6')
EXT_V6_REFERENCE_SPEC = {
    'path': os.path.join(EXT_V6_ROOT_REL, 'campaign_s53_w142_resettle_ext_v6_e_no_ageing',
                         'campaign_spec_s53_w142_resettle_ext_v6_e_no_ageing_10f4aa7a.json'),
    'sha256': '10f4aa7a05b5022ecdd347d45aea1070dc1557237ad482e17a0ee1645846d022',
    'what': 'the committed W142 extension v6 e_no_ageing campaign spec (E.21 row 54): its configuration block'}
CONFIG_TEXT_KEYS = ('name', 'note')     # free text: the only configuration keys allowed to differ from ext v6

# ---- the two models and the price ----
FLEX_M = 2.0
MODELS = ('A', 'N')
MODEL_VARIANT_N = {'ageing_enabled': False, 'available_energy_soh_point': 'end', 'calendar_retention_per_year': 1.0,
                   'eol_retention_r': 0.5}
MODEL_DECLARATIONS = {'A': None, 'N': MODEL_VARIANT_N}
MODEL_TEXT = {
    'A': ('baseline ageing: the declared ESS ageing baseline (C2 + phi_cal 0.985 + soh_min 0.70); model_variant ABSENT '
          '(absent = the baseline in the harness; the key space of every committed A cell is kept)'),
    'N': ('no ageing: model_variant {ageing_enabled False, available_energy_soh_point end, calendar_retention_per_year '
          '1.0, eol_retention_r 0.50} -- the T8 / ext-v3 e_no_ageing declaration (Addendum 83 ruling 3 i)')}

# ---- the seed set S (plan 5.1), as lattice points z = (zP5, zE5, zP7, zE7, zP9, zE9, zY) ----
SEEDS = (
    ('x0', (0, 0, 0, 0, 0, 0, 0), {'5': [0.0, 0.0], '7': [0.0, 0.0], '9': [0.0, 0.0]}, 2025,
     'x = 0 (no storage)'),
    ('unit', (0, 0, 1, 2, 0, 0, 0), {'5': [0.0, 0.0], '7': [0.25, 1.0], '9': [0.0, 0.0]}, 2025,
     'the reference unit: node 7 0.25 MVA / 1 MWh, 2025'),
    ('n7_05_2', (0, 0, 2, 4, 0, 0, 0), {'5': [0.0, 0.0], '7': [0.5, 2.0], '9': [0.0, 0.0]}, 2025,
     'node 7 0.5 MVA / 2 MWh, 2025'),
    ('hist_inc', (1, 1, 4, 7, 0, 0, 1), {'5': [0.25, 0.5], '7': [1.0, 3.5], '9': [0.0, 0.0]}, 2030,
     'the historical 2x incumbent ref:5ca4f86c (2030; node 5 0.25/0.5 + node 7 1/3.5; E.21 row 40)'),
)
EXPECTED_SEED_CANDIDATE_KEYS = {    # committed candidate keys, asserted (unit: ext-v6 e_no_ageing; hist_inc: F2 Phase B)
    'unit': 'db77e1549af855bcb3521950c634db55fe6c155ab2de591b0441b5f2bf4b369a',
    'hist_inc': '59757776a2da3ffeb369b361520faa715a159976c3b98437b6be1d863278b8db'}
EXPECTED_N_KEYS_TO_EVALUATE = 7
ALIAS_SEED, ALIAS_MODEL, ALIAS_TARGET_MODEL = 'x0', 'N', 'A'

# ---- the search (variant B, by import) ----
N_MAX_NEW_EVALUATIONS = 30
MAX_POLLS = 10
HALTON_K0 = 0
COMPLETION_CAP = C.COMPLETION_CAP                       # 30
MIN_FEASIBLE_POLL_POINTS = C.MIN_FEASIBLE_POLL_POINTS   # n + 1 = 8
SIGMA_Q_EXPECTED_EUR = 18449.66
PHASES = ('seeds', 'search_N', 'search_A')
# ---- E1 poll rule B+ (Addendum 85 ruling 2) ----
ADVANCE_WINDOW = 10      # at most 10 consecutive Halton indices examined per launched poll (the refused one included)
B_PLUS_RULE = (
    'E1 POLL RULE B+ (PLANNER_BRIEF_2026-09-13.md Addendum 85 ruling 2; declared before any E1 solve; identical in '
    'both arms and at every poll): when fewer than n + 1 = 8 distinct feasible poll points remain after rounding and '
    'snapping, (i) if the completion of the incumbent has <= 30 feasible points it is used, as frozen; (ii) if it has '
    'more than 30, the Halton index advances by one and the 2n directions are re-drawn, rounded and snapped, at most '
    '10 consecutive indices (the refused index included), and the first index with at least n + 1 distinct feasible '
    'poll points is polled -- the skipped indices and their snap tables are written into the poll record '
    '(halton_indices_skipped, advance_rule); skipped indices are not launched polls; (iii) if no index in the window '
    'yields n + 1 points, STOP_FOR_REVIEW_completion_cap as frozen (the terminating record is the record of the first '
    'examined index, with the further examined indices listed in halton_indices_skipped). Everything else is the '
    'frozen variant B of Addendum 39 ruling 1 / spec v23: 2n = 14 OrthoMADS directions, the rounding, SNAP_RULE, '
    'Delta = 1 at every poll, acceptance of a poll point only above max{bar_x + bar_inc, sigma_Q = 18,449.66 EUR}, '
    'the A7 barrier rule (2 new barrier evaluations in one poll or 3 overall: stop for review), <= 30 new '
    'evaluations per arm, <= 10 launched polls, the Halton index continuing from the last polled index + 1; '
    'k0 = 0 (t = 17) in both arms. The rule reads the snap table only (no objective value).')
POLL_RECORD_FIELDS_E1 = tuple(C.POLL_RECORD_FIELDS) + ('halton_indices_skipped', 'advance_rule')
SKIPPED_INDEX_FIELDS = ('k', 'halton_t', 'halton_u', 'directions', 'snap_table', 'n_distinct_feasible_poll_points',
                        'completion', 'candidates')
SKIPPED_COMPLETION_FIELDS = ('n_feasible', 'over_cap', 'cap')
SKIPPED_CANDIDATE_FIELDS = ('label', 'poll_part', 'direction_index', 'disposition')
ADVANCE_RULE_FIELDS = ('enabled', 'window', 'first_index', 'polled_index', 'n_examined', 'text')
# ---- the frozen variant-B primitives BY REFERENCE (the unchanged objects of p515_s53_f2_certificate), bound under the
#      names C.run_continuation's body uses, so that run_continuation_e1's text differs from C.run_continuation only
#      where B+ requires it (the W183g (b) gate diffs the two sources) ----
build_poll, _certificate = C.build_poll, C._certificate
DELTA_UNIT, POLL_RULE, STOP_RULE = C.DELTA_UNIT, C.POLL_RULE, C.STOP_RULE
# ---- the declared roots (Addendum 85 ruling 4 (iv); spec v1) -- --freeze / --run refuse any other root ----
E1_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e1_2x')
E1_BATCH_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e1_2x_batch4')
BATCH_SEEDS = ('unit_A_2x', 'unit_N_2x', 'n7_05_2_A_2x', 'n7_05_2_N_2x')
BATCH_CONCURRENCY = 4
DECLARED_ROOTS = {
    E1_ROOT_REL: {
        'role': 'the E1 root: seeds, both searches, then the cross-evaluations',
        'phases': ['seeds', 'search_N', 'search_A', 'cross'],
        'phases_of_this_launcher': list(PHASES),
        'seeds': 'all seven (no --seeds subset)',
        'cross': ('owned by the fixed-plan launcher (Addendum 85 ruling 4 (vi), (xi)); created after both searches; '
                  'never created or read by this launcher'),
        'concurrency': ('declared per phase at --freeze; 4 slots for the search batches only if the 4-slot batch gate '
                        'is bitwise (ruling 4 iv), otherwise 1')},
    E1_BATCH_ROOT_REL: {
        'role': ('the 4-slot batch gate (ruling 4 iv): the four node-7 seed evaluations as ONE batch, compared with '
                 'the serial records of P515S55/e1_2x on PER_CYCLE_TRAJECTORY_FIELDS (W101/W142 replay method; the '
                 'comparison script is not part of this launcher), peak RSS recorded'),
        'phases': ['seeds'], 'phases_of_this_launcher': ['seeds'],
        'seeds': list(BATCH_SEEDS), 'concurrency': BATCH_CONCURRENCY,
        'never_seeds_an_arm': ('the cache is root-scoped (load_seed_cache reads only <root>/campaign_s55_<root>_seeds) '
                               'and no search phase is accepted in this root')},
}
POLL_DESIGN_RECORD = {
    'variant': 'B+ (Addendum 85 ruling 2) over variant B (Addendum 70 ruling 6; plan 5.1)',
    'module': ('p515_s55_e1_launcher.run_continuation_e1 (a copy of p515_s53_f2_certificate.run_continuation with the '
               'B+ advance inserted before the completion-cap refusal); p515_s53_f2_certificate.build_poll and '
               '_certificate by reference, unchanged'),
    'b_plus_rule': B_PLUS_RULE, 'advance_window': ADVANCE_WINDOW,
    'poll_rule': C.POLL_RULE, 'snap_rule': C.SNAP_RULE,
    'iteration_rule_as_applied': ('on success the next poll is again a unit poll (Delta = 1); the Halton index starts '
                                  'at k = 0 (t = 17) for EACH arm and continues from the last POLLED index + 1 (B+: '
                                  'indices skipped by the advance are not launched polls); at most '
                                  f'{N_MAX_NEW_EVALUATIONS} NEW evaluations per arm (seeds counted separately); a '
                                  'poll whose new evaluations would exceed what remains is not launched '
                                  '(evaluation_budget_exhausted); at most '
                                  f'{MAX_POLLS} launched polls (max_polls_reached)'),
    'resolution_rule': PB.RESOLUTION_RULE, 'stop_rule': C.STOP_RULE,
    'barrier_stop_per_poll': PB.BARRIER_STOP_PER_POLL, 'barrier_stop_overall': PB.BARRIER_STOP_OVERALL,
    'certificate_statement': C.CERTIFICATE_STATEMENT, 'halton_k0': HALTON_K0, 'n_max_new_evaluations':
    N_MAX_NEW_EVALUATIONS, 'max_polls': MAX_POLLS, 'completion_cap': COMPLETION_CAP,
    'min_feasible_poll_points': MIN_FEASIBLE_POLL_POINTS, 'delta': C.DELTA_UNIT, 'n_directions': C.N_DIRECTIONS,
    'incumbent_rule': ('the best production-exit F among the budget-feasible, certified seeds of the arm (x = 0 under N '
                       'through the G0 alias), ties: the lower I(x), then the label')}

# ---- pins ----
GIB = 1 << 30
MEMORY_PER_SLOT_BYTES = 11 * GIB // 4       # 2.75 GiB (the S47 rule, p515_s47_phase_b_record.py ~1063-1088)
MEMORY_RULE = ('hw.memsize - (wired + anonymous + compressor-occupied pages) x page size >= concurrency x 2.75 GiB '
               '(the S47 launcher rule)')
SETTLING_MODULES = ('settling_criterion.py', 'settling_criterion_v2.py', 'settling_criterion_v3.py',
                    'settling_criterion_v4.py', 'settling_criterion_v5.py', 'settling_criterion_v6.py')
RUN_FILES = ('p515_s44_campaign_harness.py', 'p515_g_g1_g4_admm_gates.py', SCRIPT_NAME, 'p56a_oracle.py') \
    + SETTLING_MODULES
NAMED_INPUTS = {'case_file': H.CASE_FILE_REL, 'ess_params_file': H.ESS_PARAMS_FILE_REL,
                'cost_file': PB.COST_FILE['path'],
                'network_params_files': [os.path.join('data', 'SRP1', d, f'{d}_params.json')
                                         for d in ('case9', 'case33_1', 'case33_2', 'case33_3')]}
SEARCH_TABLES = {'investment_cost_w2': dict(PB.INVESTMENT_COST_RESULTS), 'phase_a_tables': dict(PB.PHASE_A_TABLES)}
PRODUCTION_DEFINITION = 'tracked root-level *.py not matching ^p\\d (the W182f / W183-0 definition)'

# ---- the committed final-code cells whose (candidate, model, price) coincide with a seed (dry-run comparison) ----
W182F_RECORDS = {'path': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w182f_records', 'w182f_records.json'),
                 'sha256': '405c79c9e146600421a76443692fb23eb69d0d35d90358de14c899d66ddda085'}
COMMITTED_CELLS = (
    {'row': 35, 'cell': 'h_74eda68d', 'seed': 'unit', 'model': 'A', 'flex': 2.0, 'source': 'W182f Z3 row 35',
     'spec': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w142_resettle_v6',
                          'campaign_s53_w142_resettle_v6_h_74eda68d',
                          'campaign_spec_s53_w142_resettle_v6_h_74eda68d_63ca1061.json'),
     'spec_sha256': '63ca1061295f572e7bece04dfb4eb5891ce2aa05dcde6990d7c9916297d0162d',
     'eval_key': 'dca08628128736042fb1d1fa1a933b0b93f6d4cfea9ef8b4279e3c45d5a6139a'},
    {'row': 40, 'cell': 'ref:5ca4f86c (issuing run: W118 r2 re-settle)', 'seed': 'hist_inc', 'model': 'A', 'flex': 2.0,
     'source': 'W182f Z3 row 40',
     'spec': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w118_resettle',
                          'campaign_s53_w118_resettle_r2_f2_incumbent',
                          'campaign_spec_s53_w118_resettle_r2_f2_incumbent_b1dd9b82.json'),
     'spec_sha256': 'b1dd9b82655072dcb9c6318914bd5abc840b7bbea2d4f9b33e61db62b556ef30',
     'eval_key': '24c5ccb6f285219f112f54709f268a76b7eace266d5c3ea1320f92f06859c7f3'},
    {'row': 40, 'cell': 'ref:5ca4f86c (the original s51 F2 Phase B evaluation the row is named after)',
     'seed': 'hist_inc', 'model': 'A', 'flex': 2.0, 'source': 'the row label; s51 F2 Phase B spec 5ce295e1',
     'spec': os.path.join('data', 'SRP1', 'Results', 'P515S51', 'campaign_s51_f2_phase_b',
                          'campaign_spec_s51_f2_phase_b_5ce295e1.json'),
     'spec_sha256': '5ce295e17c5165ea2148d1240c918ebca05b8aafbeeb22e2ebfc03f9bf120094',
     'eval_key': '5ca4f86c3406c0424d3b63b8f39df4568f1be77f7f2d72642491539c0eb42663'},
    {'row': 54, 'cell': 'e_no_ageing', 'seed': 'unit', 'model': 'N', 'flex': 1.0, 'source': 'W182f Z3 row 54',
     'spec': EXT_V6_REFERENCE_SPEC['path'], 'spec_sha256': EXT_V6_REFERENCE_SPEC['sha256'],
     'eval_key': '1b10e9e161523f46b9e4c0a2b6f7400ae1d2b2d7a42ef311e0d4726a7effd7e0',
     'expected_base_key': '054976fb6f0c6adbd3175a23f60915e931e161f71e3bd59c8719e71d3ffa000a'},
)
WRAPPER_KEYS = ('certification_continuation', 'settling_continuation', 'settling_extension', 'settling_resettle')


# ======================================================================================================================
#  small helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W183b-E1] {msg}', flush=True)


def _abs(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def _rel(path):
    return os.path.relpath(path, REPO)


def _load(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _sha(rel):
    return H.sha256_file(_abs(rel))


def _sha_or_none(rel):
    return H.sha256_file(_abs(rel)) if os.path.isfile(_abs(rel)) else None


def _git_tracked_clean(rel):
    tracked = bool(H._git(['ls-files', '--', rel]).strip())
    dirty = bool(H._git(['status', '--porcelain', '--', rel]).strip())
    return tracked, not dirty


def _jsonable(obj):
    return json.loads(GRIO.dumps(obj))


def _write_once_json(path, obj):
    text = GRIO.dumps(obj, indent=1, sort_keys=True)
    with open(path, 'x') as handle:
        handle.write(text)
    return H.sha256_file(path)


def _inside(path, root):
    real, base = os.path.realpath(path), os.path.realpath(root)
    return real.startswith(base + os.sep)


def all_guards():
    """Every SolveProfileGuard instance reachable from the imported repo modules (each launcher module installs its
    own parent guard at import), deduplicated."""
    seen, out = set(), []
    for name, mod in sorted(sys.modules.items()):
        if not getattr(mod, '__file__', None) or not str(mod.__file__).startswith(REPO + os.sep):
            continue
        for attr, value in sorted(vars(mod).items()):
            if isinstance(value, SolveProfileGuard) and id(value) not in seen:
                seen.add(id(value))
                out.append((f'{name}.{attr}', value))
    if id(PARENT_GUARD) not in seen:
        out.append(('p515_s55_e1_launcher.PARENT_GUARD', PARENT_GUARD))
    return out


def guards_report():
    guards = {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in all_guards()}
    return {'solve_profile_guards': guards,
            'solve_guards_ok': all(not v['verify_0_failures'] for v in guards.values()),
            'pickle': {'counts': dict(PICKLE_COUNTS),
                       'blocked': pickle.load is _blocked_load and pickle.loads is _blocked_loads},
            'model_build_tripwire': dict(MODEL_BUILD_COUNTS),
            'committed_launcher_function_tripwires': {k: v for k, v in sorted(TRIPWIRE_COUNTS.items())},
            'all_zero': (all(not v['verify_0_failures'] for v in guards.values())
                         and not any(PICKLE_COUNTS.values()) and not any(MODEL_BUILD_COUNTS.values())
                         and not any(TRIPWIRE_COUNTS.values()))}


def _own_process_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and any(s in parts[1] for s in OWN_PROCESS_SUBSTRINGS) \
                and 'python' in parts[1]:
            hits.append(line.strip())
    return hits


# ======================================================================================================================
#  configuration, models, keys
# ======================================================================================================================
def configuration():
    """The W142 extension v6 configuration block (`p515_s53_w142_resettle_ext_v6_campaign.configuration`, ~439-451),
    copied minus the re-settling declaration: the case file (AA keep_memory declared), the ESS ageing baseline
    declared (minimum SoH 0.70, NOT overridden), the convergence-depth tight tail DECLARED ENABLED; no settling
    declaration of any kind, so the evaluation stops at the production exit."""
    return {'name': ('E1 SRP1 SEARCH (W183b, Addendum 83) -- the current production configuration as the W142 '
                     'extension v6 declares it: the case file (AA keep_memory declared), the ESS ageing baseline '
                     + ESS_AGEING_BASELINE_LABEL + ' declared (minimum SoH 0.70, NOT overridden), the convergence-depth '
                     'tight tail DECLARED ENABLED (compl_inf_tol 1e-6); NO re-settling / continuation declaration: '
                     'production exit (10 consecutive passing cycles, cap 500); flexibility price x 2 on every entry'),
            'arm_label': ARM_LABEL, 'overrides': {},
            'case_file_anderson_acceleration': dict(CASE_FILE_AA),
            'ess_ageing_baseline': json.loads(json.dumps(ESS_AGEING_BASELINE)),
            'ess_ageing_baseline_label': ESS_AGEING_BASELINE_LABEL,
            'convergence_depth_tail': dict(CONVERGENCE_DEPTH_TAIL),
            'note': ('every entry declares flex_price_multiplier 2.0 (keyed; FLEX PRICE label) and its ageing model: '
                     'N = model_variant no_ageing (keyed; MODEL VARIANT label at spec and entry level), A = the '
                     'declared baseline (no model_variant); rho0_scale and nrf_option never declared; no '
                     'post-certification; the W142 ext-v6 block minus settling_resettle')}


def key_kwargs(model, flex=FLEX_M):
    if model not in MODELS:
        raise ValueError(f'unknown model {model!r}')
    return dict(case_file_aa=CASE_FILE_AA, model_variant=MODEL_DECLARATIONS[model],
                ess_ageing_baseline=ESS_AGEING_BASELINE, flex_price_multiplier=flex,
                convergence_depth_tail=CONVERGENCE_DEPTH_TAIL)


def eval_key(canonical, model, flex=FLEX_M):
    """The evaluation key of a candidate under `model` at `flex` in THIS configuration (no rho0_scale, no nrf_option,
    no wrapper)."""
    return H.evaluation_key(H.candidate_key(canonical), {}, **key_kwargs(model, flex))


def launcher_key_inputs(canonical, model, flex=FLEX_M):
    """The normalised evaluation-key inputs of the launcher's key (the comparison form, `cell_key_inputs`)."""
    return {'candidate_key': H.candidate_key(canonical), 'overrides': {},
            'effective_anderson_acceleration': H.effective_anderson_acceleration(CASE_FILE_AA, {}),
            'model_variant': H.validate_model_variant(MODEL_DECLARATIONS[model]),
            'ess_ageing_baseline': H.validate_ess_ageing_baseline(ESS_AGEING_BASELINE),
            'flex_price_multiplier_in_key': H.flex_price_multiplier_in_key(flex),
            'convergence_depth_tail_in_key': H.convergence_depth_tail_in_key(CONVERGENCE_DEPTH_TAIL),
            'derived_instance': None, 'interface_deviation_premium': None,
            'rho0_scale_in_key': None, 'nrf_option_in_key': None}


def make_arm_key_of(lattice, model):
    memo = {}

    def key_of(z):
        z = lattice.canonical_z(z)
        if z not in memo:
            memo[z] = eval_key(canonical_of(lattice, z), model)
        return memo[z]
    return key_of


def canonical_of(lattice, z):
    return H.canonical_candidate(lattice.nodes_map(z), investment_year=lattice.year_of(z))


def entry_options(lattice, z, model):
    """The harness evaluation options of one E1 entry: the year, the price and the model (N: model_variant)."""
    opts = {'investment_year': lattice.year_of(z), 'flex_price_multiplier': FLEX_M}
    if MODEL_DECLARATIONS[model] is not None:
        opts['model_variant'] = copy.deepcopy(MODEL_DECLARATIONS[model])
    return opts


def seed_label(seed, model):
    return f'{seed}_{model}_{FLEX_M:g}x'


def domain_label(lattice, z, model):
    return f'{lattice.label(z)}_{model}_{FLEX_M:g}x'


# ======================================================================================================================
#  the lattice and the search tables (pinned)
# ======================================================================================================================
def build_lattice():
    ev, failures = {}, []
    for name, pin in SEARCH_TABLES.items():
        got = _sha_or_none(pin['path'])
        ev[f'{name}_sha256'] = got
        if got != pin['sha256']:
            failures.append(f"{name}: {pin['path']} sha256 {got} != pinned {pin['sha256']}")
    if failures:
        return None, ev, failures
    w2 = _load(SEARCH_TABLES['investment_cost_w2']['path'])
    costs = PB.unit_costs_from_w2(w2)
    years = tuple(sorted(costs))
    if years != (2025, 2030, 2035):
        failures.append(f'instance years {years} != (2025, 2030, 2035)')
    lattice = PB.Lattice(years, costs)
    ev['I_x_cross_check_vs_W2'] = PB.w2_cross_check(lattice, w2)
    if not ev['I_x_cross_check_vs_W2']['ok']:
        failures.append(f"I(x) closed form != W2 table: {ev['I_x_cross_check_vs_W2']}")
    ev['sigma_Q'] = PB.sigma_q_from_tables(_load(SEARCH_TABLES['phase_a_tables']['path']))
    if round(ev['sigma_Q']['sigma_Q_eur'], 2) != SIGMA_Q_EXPECTED_EUR:
        failures.append(f"sigma_Q {ev['sigma_Q']['sigma_Q_eur']} != {SIGMA_Q_EXPECTED_EUR}")
    ev['degradation_clause'] = PB.degradation_clause(lattice, ev['sigma_Q']['sigma_Q_eur'])
    if ev['degradation_clause']['triggered']:
        failures.append('STEP4 5.2 degradation clause triggers (not implemented)')
    return lattice, ev, failures


# ======================================================================================================================
#  the seed key set (both models at 2x) and the alias
# ======================================================================================================================
def seed_plan(lattice):
    """Every seed under both models: label, z, canonical, candidate key, eval key, I(x), disposition. x = 0 under N is
    the ALIAS of x = 0 under A (never an evaluation)."""
    rows = []
    for seed, z, nodes, year, text in SEEDS:
        can = canonical_of(lattice, z)
        if can != {'investment_year': year, 'nodes': nodes}:
            raise AssertionError(f'seed {seed}: lattice canonical {can} != declared {nodes} @ {year}')
        why = lattice.reasons(z)
        if why:
            raise AssertionError(f'seed {seed} infeasible: {why}')
        ckey = H.candidate_key(can)
        if seed in EXPECTED_SEED_CANDIDATE_KEYS and ckey != EXPECTED_SEED_CANDIDATE_KEYS[seed]:
            raise AssertionError(f'seed {seed}: candidate key {ckey} != committed {EXPECTED_SEED_CANDIDATE_KEYS[seed]}')
        for model in MODELS:
            row = {'seed': seed, 'seed_text': text, 'model': model, 'label': seed_label(seed, model), 'z': list(z),
                   'lattice_label': lattice.label(z), 'canonical': can, 'candidate_key': ckey,
                   'eval_key': eval_key(can, model), 'flex_price_multiplier': FLEX_M,
                   'model_variant': copy.deepcopy(MODEL_DECLARATIONS[model]),
                   'I_x_eur': lattice.investment_cost(z), 'budget_feasible': not why}
            if seed == ALIAS_SEED and model == ALIAS_MODEL:
                row['disposition'] = 'alias'
                row['alias_of'] = {'label': seed_label(seed, ALIAS_TARGET_MODEL),
                                   'eval_key': eval_key(can, ALIAS_TARGET_MODEL)}
            else:
                row['disposition'] = 'evaluate'
            rows.append(row)
    return rows


def seed_key_assertions(plan):
    evaluate = [r for r in plan if r['disposition'] == 'evaluate']
    aliases = [r for r in plan if r['disposition'] == 'alias']
    keys_to_evaluate = [r['eval_key'] for r in evaluate]
    x0_a = next(r for r in plan if r['seed'] == ALIAS_SEED and r['model'] == ALIAS_TARGET_MODEL)
    all_keys = [r['eval_key'] for r in plan]
    checks = {
        'len_set_keys_to_evaluate_eq_7': len(set(keys_to_evaluate)) == EXPECTED_N_KEYS_TO_EVALUATE,
        'len_keys_to_evaluate_eq_7_no_duplicates': len(keys_to_evaluate) == EXPECTED_N_KEYS_TO_EVALUATE,
        'exactly_one_alias': len(aliases) == 1,
        'alias_is_x0_under_N': (len(aliases) == 1 and aliases[0]['seed'] == ALIAS_SEED
                                and aliases[0]['model'] == ALIAS_MODEL),
        'alias_target_eq_x0_A_2x_key': len(aliases) == 1 and aliases[0]['alias_of']['eval_key'] == x0_a['eval_key'],
        'alias_target_is_evaluated': len(aliases) == 1 and aliases[0]['alias_of']['eval_key'] in keys_to_evaluate,
        'alias_own_key_not_evaluated': len(aliases) == 1 and aliases[0]['eval_key'] not in keys_to_evaluate,
        'all_8_seed_model_keys_pairwise_distinct': len(set(all_keys)) == len(all_keys) == 8,
        'every_key_carries_flex_2': all(r['eval_key'] != eval_key(r['canonical'], r['model'], flex=1.0)
                                        for r in plan),
        'N_keys_differ_from_A_keys': all(
            r['eval_key'] != eval_key(r['canonical'], 'A') for r in plan if r['model'] == 'N'),
        'every_seed_budget_feasible': all(r['budget_feasible'] for r in plan),
    }
    return checks, {'keys_to_evaluate': keys_to_evaluate, 'n_keys_to_evaluate': len(keys_to_evaluate),
                    'n_distinct_keys_to_evaluate': len(set(keys_to_evaluate)), 'n_aliases': len(aliases),
                    'alias': aliases[0] if aliases else None}


def key_table(plan, holders=None):
    out = []
    for r in plan:
        row = {k: r[k] for k in ('seed', 'model', 'label', 'lattice_label', 'candidate_key', 'eval_key',
                                 'disposition', 'I_x_eur')}
        row['alias_of_eval_key'] = (r.get('alias_of') or {}).get('eval_key')
        if holders is not None:
            row['committed_campaign_specs_holding_this_eval_key'] = holders.get(r['eval_key'], [])
        out.append(row)
    return out


def committed_eval_key_holders(keys):
    """eval key -> the git-tracked campaign specs (data/*campaign_spec_*.json) whose entries carry it. Informative
    (the root-scoped cache never reads them)."""
    wanted, out = set(keys), {}
    rels = [p for p in H._git(['ls-files', 'data/*campaign_spec_*.json']).splitlines() if p.strip()]
    for rel in sorted(rels):
        for e in _load(rel).get('candidates') or []:
            k = H._entry_eval_key(e)
            if k in wanted:
                out.setdefault(k, []).append(rel)
    return out, len(rels)


# ======================================================================================================================
#  G0 (the alias's justification)
# ======================================================================================================================
def production_modules():
    return sorted(p for p in H._git(['ls-files', '*.py']).splitlines()
                  if p.strip() and '/' not in p and not re.match(r'^p\d', p))


def production_now():
    mods = production_modules()
    return {'definition': PRODUCTION_DEFINITION, 'head': H._git(['log', '-1', '--format=%H', '--'] + mods),
            'modules_sha256': {m: _sha(m) for m in mods}}


def _load_g0(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(_abs(path), 'rt') as handle:
        return json.load(handle)


def g0_check(g0_path, g0_sha_pin, x0_keys, require_current_production):
    """The alias justification. `x0_keys` = {'A': key, 'N': key} of x = 0 at FLEX_M. Accepted only when the artifact
    exists, hashes to its pin and reports NL identity of x = 0 under A and N at FLEX_M (ESSO NL on every node; network
    NL in both builders), with its own x = 0 keys equal to this launcher's and its configuration / N declaration
    equal to this launcher's; with `require_current_production` (--freeze / --run) its production modules must also
    equal the current production."""
    tag = f'{FLEX_M:g}x'
    info = {'path': g0_path, 'sha256_pinned': g0_sha_pin, 'exists': os.path.isfile(_abs(g0_path))}
    checks = {'exists': info['exists']}
    if not info['exists']:
        return False, dict(info, checks=checks)
    info['sha256_on_disk'] = _sha(g0_path)
    info['git_tracked'], info['git_clean'] = _git_tracked_clean(g0_path)
    checks['sha256_equals_pin'] = info['sha256_on_disk'] == g0_sha_pin
    checks['git_tracked_and_clean'] = info['git_tracked'] and info['git_clean']
    if not checks['sha256_equals_pin']:
        return False, dict(info, checks=checks)
    d = _load_g0(g0_path)
    comp = (d.get('summary') or {}).get(f'i_x0_A_{tag}_vs_x0_N_{tag}') or {}
    seq, par = comp.get('sequential') or {}, comp.get('parallel') or {}
    keys = (d.get('summary') or {}).get('eval_keys') or {}
    spec_like = ((d.get('configuration') or {}).get('spec_like') or {}).get('configuration') or {}
    checks.update({
        'artifact_all_ok': d.get('all_ok') is True,
        'artifact_guard_zero_solves': ((d.get('checks') or {}).get('guard_zero_solves_verified') is True),
        f'x0_A_vs_N_{tag}_esso_nl_plain_identical_every_node': (
            bool(seq.get('esso_nl_plain_identical_by_node'))
            and all(v is True for v in seq['esso_nl_plain_identical_by_node'].values())),
        f'x0_A_vs_N_{tag}_esso_nl_all_four_files_identical_every_node': (
            bool(seq.get('esso_nl_all_four_files_identical_by_node'))
            and all(v is True for v in seq['esso_nl_all_four_files_identical_by_node'].values())),
        f'x0_A_vs_N_{tag}_network_nl_identical_sequential': (seq.get('network_nl_plain_identical_all') is True
                                                              and seq.get('network_nl_all_four_files_identical_all')
                                                              is True),
        f'x0_A_vs_N_{tag}_network_nl_identical_parallel': (par.get('network_nl_plain_identical_all') is True
                                                            and par.get('network_nl_all_four_files_identical_all')
                                                            is True),
        f'artifact_x0_A_{tag}_key_equals_alias_target': keys.get(f'x0_A_{tag}') == x0_keys['A'],
        f'artifact_x0_N_{tag}_key_equals_alias_key': keys.get(f'x0_N_{tag}') == x0_keys['N'],
        'artifact_configuration_equals_launcher': (
            spec_like.get('case_file_anderson_acceleration') == CASE_FILE_AA
            and spec_like.get('convergence_depth_tail') == CONVERGENCE_DEPTH_TAIL
            and spec_like.get('ess_ageing_baseline') == ESS_AGEING_BASELINE
            and (spec_like.get('ess_params_file') or {}).get('sha256') == _sha(H.ESS_PARAMS_FILE_REL)),
        'artifact_model_N_equals_launcher': ((d.get('configuration') or {}).get('models') or {}).get('N')
        == MODEL_VARIANT_N,
    })
    prod_now = production_now()
    art_prod = (d.get('production') or {}).get('modules_sha256') or {}
    differing = sorted(m for m in set(prod_now['modules_sha256']) | set(art_prod)
                       if prod_now['modules_sha256'].get(m) != art_prod.get(m))
    info['artifact_production'] = {'expected_commit': (d.get('production') or {}).get('expected_commit'),
                                   'git_head_at_run': d.get('git_head_at_run'),
                                   'equals_current_production': not differing,
                                   'modules_differing_from_current': differing,
                                   'current_production_head': prod_now['head']}
    info['differing_esso_rows_every_index_inactive'] = seq.get('esso_differing_constraints_every_index_inactive')
    info['artifact_eval_keys'] = {k: keys.get(k) for k in (f'x0_A_{tag}', f'x0_N_{tag}')}
    if require_current_production:
        checks['artifact_production_equals_current_production'] = not differing
    accepted = all(checks.values())
    return accepted, dict(info, checks=checks, accepted=accepted,
                          require_current_production=require_current_production)


def alias_record(plan, g0_path, g0_info):
    a = next(r for r in plan if r['disposition'] == 'alias')
    return {'alias_label': a['label'], 'alias_eval_key': a['eval_key'], 'reuse_of': a['alias_of']['eval_key'],
            'reuse_of_label': a['alias_of']['label'], 'justification': 'G0',
            'g0_artifact_path': g0_path, 'g0_artifact_sha256': g0_info.get('sha256_on_disk'),
            'g0_accepted': g0_info.get('accepted'), 'g0_checks': g0_info.get('checks'),
            'rule': ('x = 0 is evaluated once, under A; its N seed reuses that evaluation by the G0 identity (the '
                     'model IPOPT receives at x = 0 is byte-identical under A and N); never written as an evaluation')}


# ======================================================================================================================
#  pins (ruling 3 vi)
# ======================================================================================================================
def tracked_inputs():
    return sorted(p for p in H._git(['ls-files', 'data/SRP1']).splitlines()
                  if p.strip() and not p.startswith(('data/SRP1/Results/', 'data/SRP1/Diagrams/')))


def imported_repo_modules():
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if f and str(f).startswith(REPO + os.sep) and str(f).endswith('.py'):
            out[_rel(f)] = _sha(_rel(f))
    return out


def pins_now():
    prod = production_now()
    run_files = {f: _sha_or_none(f) for f in RUN_FILES}
    imported = imported_repo_modules()
    inputs = {f: _sha(f) for f in tracked_inputs()}
    tables = {name: {'path': pin['path'], 'sha256': _sha_or_none(pin['path'])} for name, pin in SEARCH_TABLES.items()}
    every = sorted(set(prod['modules_sha256']) | set(run_files) | set(imported) | set(inputs)
                   | {t['path'] for t in tables.values()})
    status = H._git(['status', '--porcelain', '--'] + every)
    untracked = [f for f in every if not H._git(['ls-files', '--', f]).strip()]
    return {'git_head_at_pin': H._git(['rev-parse', 'HEAD']), 'production': prod,
            'run_files_sha256': run_files, 'imported_repo_modules_sha256': imported,
            'inputs': {'definition': 'every tracked file under data/SRP1 outside Results/ and Diagrams/',
                       'sha256': inputs},
            'named_inputs': {k: ({'path': v, 'sha256': inputs.get(v)} if isinstance(v, str)
                                 else [{'path': p, 'sha256': inputs.get(p)} for p in v])
                             for k, v in NAMED_INPUTS.items()},
            'search_tables': tables,
            'uncommitted_changes': [ln for ln in status.splitlines() if ln.strip()],
            'untracked': untracked}


def pins_compare(frozen, now):
    """Every pinned item of `frozen` against `now` (the HEAD itself is not pinned: committing a phase's results moves
    it). Returns the list of differences (empty = identical)."""
    diffs = []
    if frozen['production']['head'] != now['production']['head']:
        diffs.append(f"production head {frozen['production']['head']} -> {now['production']['head']}")
    for group in ('modules_sha256',):
        a, b = frozen['production'][group], now['production'][group]
        for m in sorted(set(a) | set(b)):
            if a.get(m) != b.get(m):
                diffs.append(f'production module {m}: {a.get(m)} -> {b.get(m)}')
    for f, sha in sorted(frozen['run_files_sha256'].items()):
        if _sha_or_none(f) != sha:
            diffs.append(f'run file {f}: {sha} -> {_sha_or_none(f)}')
    for f, sha in sorted(frozen['imported_repo_modules_sha256'].items()):
        if _sha_or_none(f) != sha:
            diffs.append(f'imported module {f}: {sha} -> {_sha_or_none(f)}')
    a, b = frozen['inputs']['sha256'], now['inputs']['sha256']
    for f in sorted(set(a) | set(b)):
        if a.get(f) != b.get(f):
            diffs.append(f'input {f}: {a.get(f)} -> {b.get(f)}')
    for name, t in sorted(frozen['search_tables'].items()):
        if _sha_or_none(t['path']) != t['sha256']:
            diffs.append(f"search table {name}: {t['sha256']} -> {_sha_or_none(t['path'])}")
    return diffs


def pins_recorded_check(pins):
    """Every pin ruling 3 (vi) names is recorded with a sha256."""
    return {
        'production_head': bool(pins.get('production', {}).get('head')),
        'production_modules_sha256': bool(pins.get('production', {}).get('modules_sha256')),
        **{f'run_file:{f}': bool((pins.get('run_files_sha256') or {}).get(f)) for f in RUN_FILES},
        'case_file': bool(((pins.get('named_inputs') or {}).get('case_file') or {}).get('sha256')),
        'ess_params_file': bool(((pins.get('named_inputs') or {}).get('ess_params_file') or {}).get('sha256')),
        'cost_file': bool(((pins.get('named_inputs') or {}).get('cost_file') or {}).get('sha256')),
        'network_params_files': all(e.get('sha256') for e in (pins.get('named_inputs') or {}).get(
            'network_params_files') or [{}]),
        'cost_file_equals_pb_pin': (((pins.get('named_inputs') or {}).get('cost_file') or {}).get('sha256')
                                    == PB.COST_FILE['sha256']),
        'ess_params_equals_pb_pin': (((pins.get('named_inputs') or {}).get('ess_params_file') or {}).get('sha256')
                                     == PB.ESS_PARAMS_FILE['sha256']),
        'search_tables': all(t.get('sha256') for t in (pins.get('search_tables') or {}).values()),
        'imported_repo_modules': bool(pins.get('imported_repo_modules_sha256')),
    }


# ======================================================================================================================
#  memory preflight (S47 rule, re-implemented with the slot count as a parameter)
# ======================================================================================================================
def memory_preflight(concurrency):
    required = int(concurrency) * MEMORY_PER_SLOT_BYTES
    text = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    first = text.splitlines()[0]
    page = int(first.split('page size of')[1].split('bytes')[0].strip())
    raw = {}
    for line in text.splitlines()[1:]:
        if ':' in line:
            k, v = line.split(':', 1)
            v = v.strip().rstrip('.')
            if v.isdigit():
                raw[k.strip().strip('"')] = int(v)
    need = ('Pages wired down', 'Anonymous pages', 'Pages occupied by compressor')
    total = int(subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True, check=True).stdout)
    if any(raw.get(k) is None for k in need):
        return {'pass': False, 'missing': [k for k in need if raw.get(k) is None], 'rule': MEMORY_RULE,
                'concurrency': int(concurrency)}
    avail = total - sum(raw[k] for k in need) * page
    return {'utc': _utc(), 'concurrency': int(concurrency), 'per_slot_gib': MEMORY_PER_SLOT_BYTES / GIB,
            'available_gib': avail / GIB, 'required_gib': required / GIB, 'pass': avail >= required,
            'rule': MEMORY_RULE, 'hw_memsize_gib': total / GIB,
            'pages': {k: raw.get(k) for k in need + ('Pages free', 'Pages inactive')}, 'page_size': page}


# ======================================================================================================================
#  the root and its phases
# ======================================================================================================================
def resolve_root(arg):
    """(rel, abs) of a root strictly under data/SRP1/Results/P515S55/ (realpath; no escape by '..' or links)."""
    path = os.path.abspath(_abs(arg))
    base = os.path.realpath(_abs(RESULTS_P55_REL))
    real = os.path.realpath(path)
    if not real.startswith(base + os.sep):
        raise SystemExit(f'REFUSED: --root {arg} is not strictly under {RESULTS_P55_REL}/')
    return _rel(real), real


def declared_root_refusals(root_rel, phase, seed_labels, concurrency, all_seed_labels):
    """Addendum 85 ruling 4 (iv), for --freeze / --run: only a DECLARED root, only a phase of this launcher declared
    for it, the declared seed set (P515S55/e1_2x: all seven; P515S55/e1_2x_batch4: the four batch seeds) and, in the
    batch root, the declared concurrency 4. `seed_labels` = the seeds phase's entry labels (None for a search phase).
    Returns the refusals (empty = declared)."""
    decl = DECLARED_ROOTS.get(root_rel)
    if decl is None:
        return [f'root {root_rel} is not a declared E1 root (Addendum 85 ruling 4 iv): {sorted(DECLARED_ROOTS)}']
    out = []
    if phase not in decl['phases_of_this_launcher']:
        out.append(f"phase {phase} is not declared for {root_rel} (this launcher: {decl['phases_of_this_launcher']})")
    if phase == 'seeds' and seed_labels is not None:
        expected = sorted(BATCH_SEEDS) if root_rel == E1_BATCH_ROOT_REL else sorted(all_seed_labels)
        if sorted(seed_labels) != expected:
            out.append(f'seed set {sorted(seed_labels)} != the declared seed set of {root_rel}: {expected}')
    if root_rel == E1_BATCH_ROOT_REL and concurrency != BATCH_CONCURRENCY:
        out.append(f'concurrency {concurrency} != the declared {BATCH_CONCURRENCY} of the batch root {root_rel}')
    return out


def campaign_id(root_rel, phase):
    return f"s55_{H._sanitize_id(os.path.basename(root_rel))}_{H._sanitize_id(phase)}"


def phase_root(root_abs, root_rel, phase):
    return os.path.join(root_abs, f'campaign_{campaign_id(root_rel, phase)}')


def _phase_state(path):
    if not os.path.isdir(path):
        return {'exists': False}
    names = sorted(os.listdir(path))
    specs = [n for n in names if n.startswith('campaign_spec_') and n.endswith('.json')]
    return {'exists': True, 'names': names, 'specs': specs, 'results': RESULTS_FILE in names,
            'manifest': MANIFEST_FILE in names, 'complete': bool(specs) and RESULTS_FILE in names
            and MANIFEST_FILE in names, 'only_spec': len(specs) == 1 and names == specs}


def root_inventory(root_abs, root_rel):
    inv = {'root': root_rel, 'exists': os.path.isdir(root_abs), 'launch_logs': [], 'other': [], 'phases': {}}
    names = sorted(os.listdir(root_abs)) if inv['exists'] else []
    phase_names = {os.path.basename(phase_root(root_abs, root_rel, p)): p for p in PHASES}
    for n in names:
        if n in phase_names:
            continue
        if LOG_RX.match(n) and os.path.isfile(os.path.join(root_abs, n)):
            inv['launch_logs'].append(n)
        else:
            inv['other'].append(n)
    for p in PHASES:
        inv['phases'][p] = _phase_state(phase_root(root_abs, root_rel, p))
    inv['non_log_entries'] = inv['other'] + [os.path.basename(phase_root(root_abs, root_rel, p)) for p in PHASES
                                             if inv['phases'][p]['exists']]
    return inv


def root_refusals(inv, mode, phase=None):
    """Why this root cannot be used for (mode, phase); empty = usable."""
    out = []
    if inv['other']:
        out.append(f"root holds entries that are neither launch logs nor E1 phases: {inv['other']}")
    if mode == 'dry_run' or (mode == 'freeze' and phase == 'seeds'):
        if inv['non_log_entries']:
            out.append(f"existing non-empty root (only launch logs may be present): {inv['non_log_entries']}")
        return out
    if phase not in PHASES:
        return out + [f'unknown phase {phase}']
    own = inv['phases'][phase]
    if mode == 'freeze':
        if own['exists']:
            out.append(f'phase {phase} already exists (write-once)')
    elif mode == 'run':
        if not (own['exists'] and own['only_spec']):
            out.append(f'phase {phase} must hold exactly its frozen spec and nothing else: {own}')
    if phase.startswith('search'):
        if not inv['phases']['seeds'].get('complete'):
            out.append(f"the seeds phase is not complete in this root: {inv['phases']['seeds']}")
        other = 'search_A' if phase == 'search_N' else 'search_N'
        if inv['phases'][other]['exists'] and not inv['phases'][other].get('complete'):
            out.append(f'the other search phase {other} exists and is not complete (no concurrent arms)')
    return out


# ======================================================================================================================
#  the root-scoped cache
# ======================================================================================================================
def load_seed_cache(root_abs, root_rel, arm, lattice, g0=None, require_git=True):
    """-> (ok, info, cache). Reads ONE file, <root>/campaign_s55_<root>_seeds/campaign_results.json, and nothing
    else (see the module docstring). arm in {'N', 'A'}; arm N also gets the G0 alias of x = 0 (`g0` = the
    (accepted, info) of a fresh g0_check), as an entry marked `alias`."""
    seeds_dir = phase_root(root_abs, root_rel, 'seeds')
    results_path = os.path.join(seeds_dir, RESULTS_FILE)
    manifest_path = os.path.join(seeds_dir, MANIFEST_FILE)
    info = {'arm': arm, 'root': root_rel, 'the_only_file_read': _rel(results_path),
            'enforcement': ('one path, built from --root; realpath containment of the results file, its spec and every '
                            'point eval_dir in the root; spec sha256 == results campaign_spec_sha256; spec extra '
                            'e1_root == this root and campaign_script == this launcher; every point key recomputed '
                            'under the arm model at 2x == its recorded key and in the seeds spec; other-model points '
                            'skipped; manifest hashes of the evaluation records; git tracked and clean')}
    reasons, cache, skipped = [], {}, []
    if not os.path.isfile(results_path):
        return False, dict(info, reason=f'no seeds results in this root: {_rel(results_path)}'), {}
    for path in (results_path, manifest_path):
        if not _inside(path, root_abs):
            reasons.append(f'{_rel(path)} resolves outside the root')
        if not os.path.isfile(path):
            reasons.append(f'missing {_rel(path)}')
        elif require_git:
            tracked, clean = _git_tracked_clean(_rel(path))
            if not (tracked and clean):
                reasons.append(f'{_rel(path)} not git-tracked and clean')
    specs = [n for n in os.listdir(seeds_dir) if n.startswith('campaign_spec_') and n.endswith('.json')]
    if reasons or len(specs) != 1:
        return False, dict(info, reason='; '.join(reasons + ([] if len(specs) == 1 else [f'specs: {specs}']))), {}
    spec_path = os.path.join(seeds_dir, specs[0])
    results = json.load(open(results_path))
    spec = json.load(open(spec_path))
    manifest = json.load(open(manifest_path))
    info.update({'results_sha256': H.sha256_file(results_path), 'spec': _rel(spec_path),
                 'spec_sha256': H.sha256_file(spec_path), 'manifest_sha256': H.sha256_file(manifest_path)})
    extra = spec.get('extra') or {}
    run_checks = {
        'spec_inside_root': _inside(spec_path, root_abs),
        'spec_sha256_equals_results': results.get('campaign_spec_sha256') == info['spec_sha256'],
        'spec_records_this_root': extra.get('e1_root') == root_rel,
        'spec_frozen_by_this_launcher': extra.get('campaign_script') == SCRIPT_NAME,
        'spec_phase_is_seeds': extra.get('phase') == 'seeds',
        'results_phase_is_seeds': results.get('phase') == 'seeds',
        'spec_flex_is_2': all(e.get('flex_price_multiplier') == FLEX_M for e in spec.get('candidates') or []),
    }
    if require_git:
        tracked, clean = _git_tracked_clean(_rel(spec_path))
        run_checks['spec_git_tracked_and_clean'] = tracked and clean
    reasons += [f'run check failed: {k}' for k, v in run_checks.items() if not v]
    spec_keys = {e['eval_key']: e for e in spec.get('candidates') or []}
    key_of = make_arm_key_of(lattice, arm)
    point_checks = {}
    for label, p in sorted((results.get('points') or {}).items()):
        can = p.get('candidate_canonical')
        z = lattice.z_of_canonical(can) if can else None
        if p.get('model') != arm:
            skipped.append({'label': label, 'model': p.get('model'), 'why': 'other model: never cache for this arm'})
            continue
        ed = p.get('eval_dir') or ''
        rec_rel = os.path.join(ed, 'evaluation_record.json')
        pc = {
            'eval_dir_inside_root': bool(ed) and _inside(_abs(ed), root_abs),
            'on_the_lattice': z is not None,
            'eval_key_recomputes_under_the_arm_model': z is not None and p.get('eval_key') == key_of(z),
            'eval_key_in_the_seeds_spec': p.get('eval_key') in spec_keys,
            'spec_entry_model_matches': (p.get('eval_key') in spec_keys and (
                spec_keys[p['eval_key']].get('model_variant') == MODEL_DECLARATIONS[arm])),
            'record_hash_in_manifest_holds': (manifest.get(rec_rel) is not None
                                              and _sha_or_none(rec_rel) == manifest.get(rec_rel)),
            'Q_present_iff_certified': ((p.get('status') == 'certified') == (
                p.get('certified_cost_gross_settlement_excluded') is not None)),
        }
        point_checks[label] = pc
        if not all(pc.values()):
            reasons.append(f'point {label} refused: {[k for k, v in pc.items() if not v]}')
            continue
        cache[p['eval_key']] = {
            'label': label, 'status': p.get('status'), 'eval_key': p['eval_key'],
            'Q': p.get('certified_cost_gross_settlement_excluded'), 'bar': (p.get('bar') or {}).get('value'),
            'canonical': can, 'barrier_cause': p.get('barrier_cause'),
            'source': {'kind': 'e1_seed_evaluation', 'campaign': results.get('campaign_id'), 'root': root_rel,
                       'eval_dir': ed, 'results_sha256': info['results_sha256']}}
    alias_info = None
    if arm == ALIAS_MODEL:
        aliases = [a for a in results.get('aliases') or [] if a.get('alias_eval_key') == key_of(lattice.x0())]
        if len(aliases) != 1:
            reasons.append(f'arm N: expected exactly one x = 0 alias in the seeds results, found {len(aliases)}')
        else:
            a = aliases[0]
            target = next((p for p in (results.get('points') or {}).values()
                           if p.get('eval_key') == a.get('reuse_of')), None)
            x0_keys = {'A': eval_key(canonical_of(lattice, lattice.x0()), 'A'),
                       'N': eval_key(canonical_of(lattice, lattice.x0()), 'N')}
            g_ok = bool(g0 and g0[0])
            ac = {'reuse_of_is_x0_A_key': a.get('reuse_of') == x0_keys['A'],
                  'alias_key_is_x0_N_key': a.get('alias_eval_key') == x0_keys['N'],
                  'target_evaluated_in_this_root': target is not None and target.get('model') == ALIAS_TARGET_MODEL,
                  'target_eval_dir_inside_root': target is not None and _inside(_abs(target.get('eval_dir') or '/'),
                                                                                root_abs),
                  'g0_accepted_now': g_ok,
                  'g0_sha256_equals_alias_record': bool(g0) and (g0[1].get('sha256_on_disk')
                                                                 == a.get('g0_artifact_sha256'))}
            alias_info = {'alias': a, 'checks': ac}
            if not all(ac.values()):
                reasons.append(f'alias refused: {[k for k, v in ac.items() if not v]}')
            else:
                cache[a['alias_eval_key']] = {
                    'label': a['alias_label'], 'status': target.get('status'), 'eval_key': a['alias_eval_key'],
                    'Q': target.get('certified_cost_gross_settlement_excluded'),
                    'bar': (target.get('bar') or {}).get('value'), 'canonical': target.get('candidate_canonical'),
                    'barrier_cause': target.get('barrier_cause'),
                    'alias': {'reuse_of': a['reuse_of'], 'justification': 'G0',
                              'g0_artifact_sha256': a.get('g0_artifact_sha256')},
                    'source': {'kind': 'alias_G0', 'campaign': results.get('campaign_id'), 'root': root_rel,
                               'reuse_of_eval_dir': target.get('eval_dir')}}
    info.update({'run_checks': run_checks, 'point_checks': point_checks, 'skipped_other_model': skipped,
                 'alias': alias_info, 'n_entries': len(cache),
                 'n_evaluation_entries': sum(1 for v in cache.values() if 'alias' not in v),
                 'n_alias_entries': sum(1 for v in cache.values() if 'alias' in v)})
    if reasons:
        return False, dict(info, reason='; '.join(reasons)), {}
    return True, info, cache


def initial_incumbent(lattice, cache, sigma_q):
    """The best production-exit F among the budget-feasible certified cache entries (ties: lower I, then label)."""
    rows = []
    for key, rec in sorted(cache.items()):
        z = lattice.z_of_canonical(rec['canonical'])
        i_x = lattice.investment_cost(z)
        certified = rec.get('status') == 'certified' and rec.get('Q') is not None
        rows.append({'label': rec['label'], 'eval_key': key, 'z': list(z), 'I': i_x, 'Q': rec.get('Q'),
                     'F': (i_x + rec['Q']) if certified else None, 'bar': rec.get('bar'), 'status': rec.get('status'),
                     'budget_feasible': not lattice.reasons(z), 'alias': 'alias' in rec, 'source': rec.get('source')})
    ok = [r for r in rows if r['F'] is not None and r['budget_feasible']]
    if not ok:
        return None, {'rows': rows, 'reason': 'no certified budget-feasible seed'}
    best = min(ok, key=lambda r: (r['F'], r['I'], r['label']))
    for r in rows:
        if r['F'] is not None:
            res = PB.resolution(r['bar'], best['bar'], sigma_q)
            r['F_minus_F_inc_eur'] = r['F'] - best['F']
            r['resolution_eur'] = res if math.isfinite(res) else None
    inc = {'eval_key': best['eval_key'], 'z': tuple(best['z']), 'label': best['label'], 'I': best['I'],
           'Q': best['Q'], 'F': best['F'], 'bar': best['bar'], 'source': best['source']}
    return inc, {'rule': POLL_DESIGN_RECORD['incumbent_rule'], 'incumbent': _jsonable(inc), 'rows': rows}


# ======================================================================================================================
#  the E1 search loop: variant B with the B+ advance (Addendum 85 ruling 2)
# ======================================================================================================================
def _skipped_index_entry(record, completion):
    """One examined-and-refused Halton index of the B+ advance, from its `C.build_poll` record: the index, the Halton
    point, the 2n directions, the snap table, the distinct count, the completion size against the cap, and every
    candidate's identity and disposition (SKIPPED_INDEX_FIELDS)."""
    return {'k': record['poll_index'], 'halton_t': record['halton_t'], 'halton_u': record['halton_u'],
            'directions': record['directions'], 'snap_table': record['snap_table'],
            'n_distinct_feasible_poll_points': record['n_distinct_feasible_poll_points'],
            'completion': {f: completion[f] for f in SKIPPED_COMPLETION_FIELDS},
            'candidates': [{f: c[f] for f in SKIPPED_CANDIDATE_FIELDS} for c in record['candidates']]}


# run_continuation_e1 -- the E1 loop (v2). A COPY of p515_s53_f2_certificate.run_continuation (the committed module is
# not edited; its docstring is kept verbatim) with exactly three changes: (1) the signature (name; E1 defaults; the
# `advance` and `advance_window` parameters); (2) the Halton index as explicit state `k_next` (k = k_next at each
# launched poll, k_next = k_p + 1 after it) instead of k = k0 + i; (3) the B+ block between `build_poll` and the
# completion-cap refusal (marked >>> / <<<), which also writes the two new record fields `halton_indices_skipped` and
# `advance_rule`. With advance=False the loop is the imported loop plus the two fields at their trivial values
# (skipped [], enabled False, n_examined 1) -- the W183g (b) gate tests this against C.run_continuation itself.
# `on_poll_start` fires once per launched poll, on the chosen record. B_PLUS_RULE states the rule.
def run_continuation_e1(lattice, cache, key_of, incumbent, evaluate_fn, sigma_q, k0=HALTON_K0,
                        max_new_evaluations=N_MAX_NEW_EVALUATIONS, max_polls=MAX_POLLS, batch_size=1,
                        completion_cap=COMPLETION_CAP, log=_log, on_poll=None, on_poll_start=None, advance=True,
                        advance_window=ADVANCE_WINDOW):
    """The continuation loop (POLL_RULE, SNAP_RULE, ITERATION_RULE). `cache` (eval_key -> entry) is extended in
    place. `evaluate_fn(list[z]) -> list[cache entry]`. Structure, barrier rule, classification and record schema
    follow PB.run_mads; the poll set is `build_poll`'s."""
    inc = dict(incumbent)
    history, n_new, n_barrier_new = [], 0, 0
    termination, certificate = None, None
    k_next = k0

    def _finish(rec):
        history.append(rec)
        if on_poll:
            on_poll(rec)

    for i in range(max_polls):
        k = k_next
        record, completion = build_poll(lattice, cache, key_of, inc, k, sigma_q, completion_cap=completion_cap)
        # >>> B+ advance (Addendum 85 ruling 2) -- the only inserted step; the rest of the loop is C.run_continuation's
        first_k, skipped, n_examined = k, [], 1
        if advance and completion is not None and completion['over_cap']:
            pending = [_skipped_index_entry(record, completion)]
            for k_try in range(first_k + 1, first_k + advance_window):
                record_try, completion_try = build_poll(lattice, cache, key_of, inc, k_try, sigma_q,
                                                        completion_cap=completion_cap)
                n_examined += 1
                if completion_try is not None and completion_try['n_feasible'] != completion['n_feasible']:
                    raise RuntimeError(f'B+: the completion of {inc["label"]} changed with the Halton index '
                                       f'({completion["n_feasible"]} at k = {first_k}, '
                                       f'{completion_try["n_feasible"]} at k = {k_try})')
                if completion_try is None:      # >= n + 1 distinct feasible poll points: polled at this index
                    skipped, record, completion, k = pending, record_try, completion_try, k_try
                    break
                pending.append(_skipped_index_entry(record_try, completion_try))
            else:                               # (iii): the record stays the first examined index's
                skipped = pending[1:]
        record['halton_indices_skipped'] = skipped
        record['advance_rule'] = {'enabled': bool(advance), 'window': advance_window, 'first_index': first_k,
                                  'polled_index': None if (completion is not None and completion['over_cap']) else k,
                                  'n_examined': n_examined, 'text': B_PLUS_RULE}
        k_next = k + 1
        # <<< B+ advance
        cands = record['candidates']
        new = [c for c in cands if c['disposition'] == 'new_evaluation']
        if on_poll_start:
            on_poll_start(record)
        if completion is not None and completion['over_cap']:
            record['decision'] = 'stopped_for_review_completion_cap'
            for c in cands:
                if c['disposition'] in ('cache_hit', 'new_evaluation'):
                    c['outcome'] = 'not_evaluated_completion_cap'
            _finish(record)
            termination = {'reason': 'STOP_FOR_REVIEW_completion_cap', 'poll_size_reached': DELTA_UNIT,
                           'completion_n_feasible': completion['n_feasible'], 'cap': completion_cap,
                           'rule': POLL_RULE,
                           'detail': f"poll {k}: {record['n_distinct_feasible_poll_points']} distinct feasible poll "
                                     f"points < {MIN_FEASIBLE_POLL_POINTS} triggered the completion of "
                                     f"{inc['label']}, which has {completion['n_feasible']} feasible points > "
                                     f'{completion_cap}; refused, nothing of this poll evaluated, not truncated'}
            break
        if n_new + len(new) > max_new_evaluations:
            record['decision'] = 'not_launched_evaluation_budget'
            _finish(record)
            termination = {'reason': 'evaluation_budget_exhausted', 'poll_size_reached': DELTA_UNIT,
                           'detail': f'poll {k} needs {len(new)} new evaluations; {n_new} used of '
                                     f'{max_new_evaluations}'}
            break
        stop = None
        for b in range(0, len(new), batch_size):
            batch = new[b:b + batch_size]
            recs = evaluate_fn([tuple(c['z']) for c in batch])
            if len(recs) != len(batch):
                raise RuntimeError('evaluate_fn returned a different number of records')
            for c, rec in zip(batch, recs):
                if rec.get('eval_key') != c['eval_key']:
                    raise RuntimeError(f"evaluation record eval key {rec.get('eval_key')} != {c['eval_key']}")
                cache[c['eval_key']] = dict(rec)
                n_new += 1
                if rec['status'] != 'certified':
                    n_barrier_new += 1
            record['batches'].append([c['label'] for c in batch])
            barrier_this_poll = sum(1 for c in new if c['eval_key'] in cache
                                    and cache[c['eval_key']]['status'] != 'certified')
            if barrier_this_poll >= PB.BARRIER_STOP_PER_POLL or n_barrier_new >= PB.BARRIER_STOP_OVERALL:
                stop = {'reason': 'STOP_FOR_REVIEW_barrier_rule', 'barrier_new_this_poll': barrier_this_poll,
                        'barrier_new_overall': n_barrier_new, 'rule': STOP_RULE}
                break
        for c in cands:
            if c['disposition'] not in ('cache_hit', 'new_evaluation'):
                continue
            rec = cache.get(c['eval_key'])
            if rec is None:
                c['outcome'] = 'not_evaluated_stop_rule'
                continue
            c['source'] = rec.get('source')
            c['status'] = rec['status']
            c['barrier_cause'] = rec.get('barrier_cause')
            if rec['status'] != 'certified' or rec.get('Q') is None:
                c.update({'F_eur': None, 'outcome': 'barrier'})
                continue
            f_x = c['I_x_eur'] + rec['Q']
            res = PB.resolution(rec.get('bar'), inc['bar'], sigma_q)
            c.update({'Q_eur': rec['Q'], 'F_eur': f_x, 'bar_eur': rec.get('bar'),
                      'bar_sum_eur': (rec['bar'] + inc['bar']) if (rec.get('bar') is not None
                                                                     and inc['bar'] is not None) else None,
                      'resolution_eur': res if math.isfinite(res) else None, 'F_inc_minus_F_eur': inc['F'] - f_x,
                      'outcome': PB.classify(inc['F'], f_x, res)})
        if stop:
            record['decision'] = 'stopped_for_review'
            _finish(record)
            termination = dict(stop, poll_size_reached=DELTA_UNIT)
            break
        improvers = [c for c in cands if c['outcome'] == 'improvement']
        if improvers:
            best = min(improvers, key=lambda c: (c['F_eur'], c['I_x_eur'], c['label']))
            inc = {'eval_key': best['eval_key'], 'z': tuple(best['z']), 'label': best['label'], 'I': best['I_x_eur'],
                   'Q': best['Q_eur'], 'F': best['F_eur'], 'bar': best['bar_eur'], 'source': best['source']}
            record.update({'decision': 'success', 'next_incumbent': best['label'], 'next_poll_size': DELTA_UNIT})
            _finish(record)
            log(f"[S53-CERT] poll {k} (t={record['halton_t']}) incumbent={record['incumbent']['label']} "
                f"new={record['n_new_evaluations']} hits={record['n_cache_hits']} -> success, next {best['label']}")
            continue
        record.update({'decision': 'failure_at_unit_mesh', 'next_incumbent': inc['label'], 'next_poll_size': None})
        _finish(record)
        termination = {'reason': 'poll_failure_at_unit_mesh', 'poll_size_reached': DELTA_UNIT}
        certificate = _certificate(lattice, key_of, cache, inc, record, sigma_q)
        break
    if termination is None:
        termination = {'reason': 'max_polls_reached', 'poll_size_reached': DELTA_UNIT}
    last = history[-1] if history else None
    unresolved = [c for c in (last['candidates'] if last else []) if c['outcome'] == 'indeterminate']
    nb = lattice.neighbourhood(tuple(inc['z']))
    return {'termination': termination, 'termination_certificate': certificate,
            'incumbent': {kk: inc[kk] for kk in ('label', 'eval_key', 'z', 'I', 'Q', 'F', 'bar')},
            'n_polls': len(history), 'n_new_evaluations': n_new, 'n_barrier_new_evaluations': n_barrier_new,
            'final_poll_unresolved_indeterminate': [{kk: c[kk] for kk in ('label', 'F_eur', 'F_inc_minus_F_eur',
                                                                          'resolution_eur')} for c in unresolved],
            'final_poll_feasible_points': [c['label'] for c in (last['candidates'] if last else [])
                                           if c['disposition'] in ('cache_hit', 'new_evaluation')],
            'lattice_neighbourhood_of_incumbent': [
                {'label': lattice.label(z), 'eval_key': key_of(z), 'in_cache': key_of(z) in cache,
                 'polled_in_final_poll': any(c['eval_key'] == key_of(z) for c in (last['candidates'] if last else [])),
                 'I_x_eur': lattice.investment_cost(z)} for z in nb],
            'history': history}


# ======================================================================================================================
#  the harness record -> the E1 point / cache entry
# ======================================================================================================================
POINT_FIELDS = ('label', 'seed', 'model', 'status', 'eval_key', 'candidate_key', 'candidate_canonical',
                'certified_cost_gross_settlement_excluded', 'bar', 'barrier_cause', 'cycles_run',
                'certification_cycle', 'rule_ten_terminal_step_over_threshold', 'flex_price_multiplier',
                'flex_price_readback_all_match', 'model_variant', 'model_readback_all_match', 'storage_per_node',
                'wall_time_s', 'peak_rss', 'eval_dir', 'per_cycle_record', 'exit_code', 'git_head_at_record')


def _all_match(x):
    return x.get('all_match') if isinstance(x, dict) else None


def point_from_harness(rec, entry, model, seed=None):
    rec = rec or {}
    certified = rec.get('status') == 'certified'
    ed = rec.get('eval_dir')
    pc_rel = os.path.join(ed, 'per_cycle_record.jsonl') if ed else None
    if model == 'N':
        mr = {'pre_run': _all_match(rec.get('model_variant_readback_pre_run')),
              'terminal': _all_match(rec.get('model_variant_readback_terminal'))}
    else:
        mr = {'pre_run': _all_match((rec.get('ess_ageing_verified_pre_run') or {}).get('readback_pre_run')),
              'terminal': _all_match(rec.get('ess_ageing_readback_terminal'))}
    flex_rb = {'pre_run': _all_match((rec.get('flex_price_applied_in_child') or {}).get('readback_pre_run')),
               'terminal': _all_match(rec.get('flex_price_readback_terminal'))}
    return {'label': entry['label'], 'seed': seed, 'model': model, 'status': rec.get('status'),
            'eval_key': rec.get('eval_key', entry['eval_key']), 'candidate_key': entry['key'],
            'candidate_canonical': entry['canonical'],
            'certified_cost_gross_settlement_excluded': rec.get('certified_cost') if certified else None,
            'terminal_gross_operational_cost': rec.get('terminal_gross_operational_cost'),
            'terminal_net_operational_recourse': rec.get('terminal_net_operational_recourse'),
            'bar': {'value': (rec.get('bar') or {}).get('value')}, 'barrier_cause': rec.get('barrier_cause'),
            'cycles_run': rec.get('cycles_run'), 'certification_cycle': rec.get('certification_cycle'),
            'rule_ten_terminal_step_over_threshold': (rec.get('rule_ten') or {}).get('terminal_step_over_threshold'),
            'flex_price_multiplier': rec.get('flex_price_multiplier'), 'flex_price_label': rec.get('flex_price_label'),
            'flex_price_readback_all_match': flex_rb, 'model_variant': rec.get('model_variant'),
            'model_variant_label': rec.get('model_variant_label'), 'model_readback_all_match': mr,
            'storage_per_node': rec.get('storage_per_node'), 'wall_time_s': rec.get('wall_time_s'),
            'peak_rss': rec.get('peak_rss'), 'eval_dir': ed,
            'per_cycle_record': {'path': pc_rel, 'sha256': _sha_or_none(pc_rel) if pc_rel else None},
            'exit_code': (rec.get('parent_view') or {}).get('exit_code'),
            'git_head_at_record': H._git(['rev-parse', 'HEAD'])}


def point_problems(p):
    """A harness error or a read-back that does not match (the run exits 1)."""
    out = []
    if p['status'] not in ('certified', 'not_certified') or p['exit_code'] != 0:
        out.append('harness_error')
    if p['status'] in ('certified', 'not_certified'):
        if p['flex_price_multiplier'] != FLEX_M or p['flex_price_readback_all_match'].get('pre_run') is not True \
                or p['flex_price_readback_all_match'].get('terminal') is not True:
            out.append('flex_price_readback')
        if p['model_readback_all_match'].get('pre_run') is not True or \
                p['model_readback_all_match'].get('terminal') is not True:
            out.append('model_readback')
    return out


def cache_entry_from_point(p, root_rel, campaign):
    return {'label': p['label'], 'status': p['status'], 'eval_key': p['eval_key'],
            'Q': p['certified_cost_gross_settlement_excluded'], 'bar': p['bar']['value'],
            'canonical': p['candidate_canonical'], 'barrier_cause': p['barrier_cause'],
            'source': {'kind': 'e1_search_new_evaluation', 'campaign': campaign, 'root': root_rel,
                       'eval_dir': p['eval_dir']}}


# ======================================================================================================================
#  rule eleven (capture paths asserted before any run)
# ======================================================================================================================
def rule_eleven(lattice, sigma_q):
    import inspect
    harness_checks = H.assert_record_capture_paths()   # imports production modules; builds nothing
    build_src = inspect.getsource(H.build_evaluation_record)
    child_src = inspect.getsource(H._child_real)
    spn_src = inspect.getsource(H._storage_per_node)
    checks = {
        'record_status': "'status': status" in build_src,
        'record_Q_gross': "'certified_cost': report.get('gross_operational_cost')" in build_src,
        'record_bar': "'bar': bar" in build_src,
        'record_cycles_and_certification_cycle': ("'cycles_run': len(rows)" in build_src
                                                  and "'certification_cycle':" in build_src),
        'record_stopping_ratio_rule_ten': "'terminal_step_over_threshold':" in build_src,
        'record_candidate_hash': "'candidate_key': entry['key']" in build_src,
        'record_spec_hash': "'campaign_spec_sha256': spec_sha256" in build_src,
        'record_storage_per_node': "'storage_per_node': _storage_per_node(" in build_src,
        'storage_per_node_E_av_terminal': "'published_available_capacity_terminal'" in spn_src,
        'storage_per_node_soh_terminal': "'terminal_soh_per_active_cohort_year'" in spn_src,
        'record_wall_and_rss': "'wall_time_s': wall" in build_src and "'peak_rss': peak_rss" in build_src,
        'per_cycle_trajectory_written': "'per_cycle_record.jsonl'" in child_src,
        'model_variant_readback_terminal': ("holder['model_variant_readback_terminal'] = model_variant_readback_models("
                                            in child_src),
        'ageing_trajectory_terminal': "holder['ageing_trajectory_terminal'] = ageing_trajectory_terminal(" in child_src,
        'ess_ageing_readback_terminal': "holder['ess_ageing_readback_terminal'] = ess_ageing_readback_models(" in child_src,
        'flex_price_readback_terminal': "'flex_price_readback_terminal': holder.get('flex_price_readback_terminal')"
                                        in child_src,
        'parent_barrier_record_for_missing': "'status': 'harness_error', 'barrier': True" in inspect.getsource(
            H._barrier_record_for_missing),
    }
    # the E1 point record: every field produced from a synthetic harness record
    fake_entry = {'label': 'synthetic', 'eval_key': 'k' * 64, 'key': 'c' * 64,
                  'canonical': canonical_of(lattice, lattice.x0())}
    p = point_from_harness({'status': 'not_certified', 'eval_key': 'k' * 64, 'parent_view': {'exit_code': 0}},
                           fake_entry, 'N')
    checks['e1_point_record_fields'] = all(f in p for f in POINT_FIELDS)
    # the variant-B poll record: the REAL loop on synthetic records (no solve)
    key_of = make_arm_key_of(lattice, 'N')
    z_inc = SEEDS[0][1]   # x = 0: its first poll triggers the 14-point completion (under the cap), so every path runs
    inc = {'eval_key': key_of(z_inc), 'z': z_inc, 'label': lattice.label(z_inc), 'I': lattice.investment_cost(z_inc),
           'Q': 1.0e8, 'F': 1.0e8 + lattice.investment_cost(z_inc), 'bar': 1.0, 'source': 'synthetic'}
    cache = {inc['eval_key']: {'label': inc['label'], 'status': 'certified', 'Q': 1.0e8, 'bar': 1.0,
                               'canonical': canonical_of(lattice, z_inc), 'source': 'synthetic'}}

    def _barrier(batch):
        return [{'label': lattice.label(z), 'status': 'not_certified', 'eval_key': key_of(z), 'Q': None, 'bar': None,
                 'canonical': canonical_of(lattice, z), 'barrier_cause': 'synthetic', 'source': 'synthetic'}
                for z in batch]

    def _worse(batch):
        return [{'label': lattice.label(z), 'status': 'certified', 'eval_key': key_of(z), 'Q': 2.0e8, 'bar': 1.0,
                 'canonical': canonical_of(lattice, z), 'barrier_cause': None, 'source': 'synthetic'}
                for z in batch]
    try:
        out_b = C.run_continuation(lattice, dict(cache), key_of, dict(inc), _barrier, sigma_q, k0=HALTON_K0,
                                   max_new_evaluations=N_MAX_NEW_EVALUATIONS, max_polls=1, batch_size=1,
                                   completion_cap=COMPLETION_CAP, log=lambda m: None)
        poll = out_b['history'][0]
        checks['poll_record_fields'] = all(f in poll for f in C.POLL_RECORD_FIELDS)
        checks['candidate_record_fields'] = all(all(f in c for f in C.CANDIDATE_RECORD_FIELDS)
                                                for c in poll['candidates'])
        checks['snap_record_fields'] = all(all(f in s for f in C.SNAP_RECORD_FIELDS) for s in poll['snap_table'])
        checks['barrier_stop_rule_fires'] = out_b['termination']['reason'] == 'STOP_FOR_REVIEW_barrier_rule'
        out_w = C.run_continuation(lattice, dict(cache), key_of, dict(inc), _worse, sigma_q, k0=HALTON_K0,
                                   max_new_evaluations=N_MAX_NEW_EVALUATIONS, max_polls=1, batch_size=4,
                                   completion_cap=COMPLETION_CAP, log=lambda m: None)
        cert = out_w['termination_certificate'] or {}
        checks['poll_failure_certificate_fields'] = (out_w['termination']['reason'] == 'poll_failure_at_unit_mesh'
                                                     and all(f in cert for f in C.CERTIFICATE_FIELDS))
        checks['budget_rule_refuses_poll'] = C.run_continuation(
            lattice, dict(cache), key_of, dict(inc), _worse, sigma_q, k0=HALTON_K0, max_new_evaluations=0,
            max_polls=1, batch_size=1, completion_cap=COMPLETION_CAP,
            log=lambda m: None)['termination']['reason'] == 'evaluation_budget_exhausted'
        # v2: the launcher's OWN loop (run_continuation_e1, B+ advance ON), which --run uses
        out_be = run_continuation_e1(lattice, dict(cache), key_of, dict(inc), _barrier, sigma_q, k0=HALTON_K0,
                                     max_new_evaluations=N_MAX_NEW_EVALUATIONS, max_polls=1, batch_size=1,
                                     completion_cap=COMPLETION_CAP, log=lambda m: None, advance=True,
                                     advance_window=ADVANCE_WINDOW)
        checks['e1_loop_barrier_stop_rule_fires'] = out_be['termination']['reason'] == 'STOP_FOR_REVIEW_barrier_rule'
        checks['e1_loop_budget_rule_refuses_poll'] = run_continuation_e1(
            lattice, dict(cache), key_of, dict(inc), _worse, sigma_q, k0=HALTON_K0, max_new_evaluations=0,
            max_polls=1, batch_size=1, completion_cap=COMPLETION_CAP, log=lambda m: None, advance=True,
            advance_window=ADVANCE_WINDOW)['termination']['reason'] == 'evaluation_budget_exhausted'
        # ... from a STORAGE incumbent (the reference unit), whose frozen first poll the completion cap refuses
        z_u = SEEDS[1][1]
        inc_u = {'eval_key': key_of(z_u), 'z': z_u, 'label': lattice.label(z_u), 'I': lattice.investment_cost(z_u),
                 'Q': 1.0e8, 'F': 1.0e8 + lattice.investment_cost(z_u), 'bar': 1.0, 'source': 'synthetic'}
        cache_u = {inc_u['eval_key']: {'label': inc_u['label'], 'status': 'certified', 'Q': 1.0e8, 'bar': 1.0,
                                       'canonical': canonical_of(lattice, z_u), 'source': 'synthetic'}}
        out_u = run_continuation_e1(lattice, cache_u, key_of, dict(inc_u), _worse, sigma_q, k0=HALTON_K0,
                                    max_new_evaluations=N_MAX_NEW_EVALUATIONS, max_polls=1, batch_size=4,
                                    completion_cap=COMPLETION_CAP, log=lambda m: None, advance=True,
                                    advance_window=ADVANCE_WINDOW)
        pu = out_u['history'][0]
        skipped, rule = pu.get('halton_indices_skipped') or [], pu.get('advance_rule') or {}
        checks['e1_loop_poll_record_fields'] = all(f in pu for f in POLL_RECORD_FIELDS_E1)
        checks['e1_loop_candidate_record_fields'] = all(all(f in c for f in C.CANDIDATE_RECORD_FIELDS)
                                                        for c in pu['candidates'])
        checks['e1_loop_advance_ran_from_a_storage_incumbent'] = (
            bool(skipped) and rule.get('enabled') is True and rule.get('first_index') == HALTON_K0
            and [s['k'] for s in skipped] == list(range(HALTON_K0, pu['poll_index']))
            and rule.get('polled_index') == pu['poll_index'] and rule.get('n_examined') == len(skipped) + 1
            and pu['n_distinct_feasible_poll_points'] >= MIN_FEASIBLE_POLL_POINTS)
        checks['e1_loop_skipped_index_fields'] = bool(skipped) and all(
            all(f in s for f in SKIPPED_INDEX_FIELDS) and all(f in s['completion'] for f in SKIPPED_COMPLETION_FIELDS)
            and all(all(f in c for f in SKIPPED_CANDIDATE_FIELDS) for c in s['candidates'])
            and all(all(f in t for f in C.SNAP_RECORD_FIELDS) for t in s['snap_table']) for s in skipped)
        checks['e1_loop_advance_rule_fields'] = all(f in rule for f in ADVANCE_RULE_FIELDS)
        cert_u = out_u['termination_certificate'] or {}
        checks['e1_loop_poll_failure_certificate_fields'] = (
            out_u['termination']['reason'] == 'poll_failure_at_unit_mesh'
            and all(f in cert_u for f in C.CERTIFICATE_FIELDS))
    except Exception as error:  # noqa: BLE001
        checks['poll_record_fields'] = False
        checks['poll_exception'] = False
        harness_checks = dict(harness_checks, poll_exception=f'{type(error).__name__}: {error}')
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN (E1 launcher): capture paths missing: {missing}')
    return {'checks': checks, 'harness_record_capture_checklist': harness_checks,
            'point_fields': list(POINT_FIELDS)}


# ======================================================================================================================
#  spec construction / validation
# ======================================================================================================================
def configuration_vs_ext_v6(spec_configuration):
    """The frozen configuration against the committed W142 extension v6 specs: every key but the free text (name,
    note) equal to the e_no_ageing spec's; and the seven ext-v6 specs agree among themselves on those keys."""
    ref_sha = _sha_or_none(EXT_V6_REFERENCE_SPEC['path'])
    ref = _load(EXT_V6_REFERENCE_SPEC['path'])['configuration']
    keys = sorted((set(ref) | set(spec_configuration)) - set(CONFIG_TEXT_KEYS))
    diffs = {k: {'e1': spec_configuration.get(k), 'ext_v6': ref.get(k)} for k in keys
             if spec_configuration.get(k) != ref.get(k)}
    ext_specs = sorted(p for p in H._git(['ls-files', EXT_V6_ROOT_REL]).splitlines()
                       if os.path.basename(p).startswith('campaign_spec_'))
    among = {}
    for rel in ext_specs:
        cfg = _load(rel)['configuration']
        among[rel] = sorted(k for k in keys if cfg.get(k) != ref.get(k))
    return {'reference': dict(EXT_V6_REFERENCE_SPEC, sha256_on_disk=ref_sha),
            'reference_sha256_holds': ref_sha == EXT_V6_REFERENCE_SPEC['sha256'],
            'compared_keys': keys, 'text_keys_not_compared': list(CONFIG_TEXT_KEYS),
            'differing_keys': diffs, 'equal_except_text': not diffs,
            'ext_v6_specs': ext_specs, 'ext_v6_specs_differing_keys_from_reference': among,
            'ext_v6_specs_agree': len(ext_specs) == 7 and not any(among.values()),
            'ext_v6_reference_entry_carries_settling_resettle': 'settling_resettle' in (
                _load(EXT_V6_REFERENCE_SPEC['path'])['candidates'][0])}


def seeds_entries(lattice, plan, subset):
    rows = [r for r in plan if r['disposition'] == 'evaluate' and (subset is None or r['label'] in subset)]
    return [(r['label'], lattice.nodes_map(tuple(r['z'])), entry_options(lattice, tuple(r['z']), r['model']))
            for r in rows], rows


def domain_entries(lattice, arm):
    return [(domain_label(lattice, z, arm), lattice.nodes_map(z), entry_options(lattice, z, arm))
            for z in lattice.domain()]


def common_extra(root_rel, phase, concurrency, pins, preflight, lattice_ev, rule11, g0_info):
    return {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage': STAGE,
            'e1_root': root_rel, 'phase': phase, 'flex_price_multiplier': FLEX_M,
            'models': {m: {'model_variant': MODEL_DECLARATIONS[m], 'text': MODEL_TEXT[m]} for m in MODELS},
            'seeds': [{'seed': s, 'z': list(z), 'nodes': n, 'year': y, 'text': t} for s, z, n, y, t in SEEDS],
            'objective_convention': OBJECTIVE_CONVENTION, 'poll_design': POLL_DESIGN_RECORD,
            'concurrency': {'declared': int(concurrency), 'memory_rule': MEMORY_RULE,
                            'memory_per_slot_gib': MEMORY_PER_SLOT_BYTES / GIB,
                            'memory_preflight_at_freeze_non_gating': preflight,
                            'gating_at': '--run'},
            'pins': pins, 'search_tables': {k: lattice_ev.get(k) for k in ('sigma_Q', 'I_x_cross_check_vs_W2',
                                                                          'degradation_clause')},
            'rule_eleven_asserted_before_freeze': rule11['checks'], 'g0': g0_info,
            'options_never_declared': ['rho0_scale', 'nrf_option'] + list(WRAPPER_KEYS),
            'declared_roots': DECLARED_ROOTS, 'declared_root_of_this_spec': DECLARED_ROOTS.get(root_rel),
            'post_certification': 'none (production exit; as the Phase B searches)'}


def validate_spec(spec, phase, root_rel, concurrency, expected_keys):
    """The frozen spec ON DISK. `expected_keys`: label -> (eval key, model)."""
    cfg, extra = spec['configuration'], spec.get('extra') or {}
    ents = spec['candidates']
    has_n = any(m == 'N' for _k, m in expected_keys.values())
    vs6 = configuration_vs_ext_v6(cfg)
    checks = {
        'configuration_equals_ext_v6_minus_resettle': vs6['equal_except_text'],
        'n_entries': len(ents) == len(expected_keys),
        'labels_as_expected': sorted(e['label'] for e in ents) == sorted(expected_keys),
        'eval_keys_recompute': all(e['eval_key'] == expected_keys[e['label']][0] for e in ents
                                   if e['label'] in expected_keys),
        'every_entry_flex_2': all(e.get('flex_price_multiplier') == FLEX_M for e in ents),
        'every_entry_flex_label': all(e.get('flex_price_label') == H.FLEX_PRICE_LABEL for e in ents),
        'spec_flex_label': spec.get('flex_price_label') == H.FLEX_PRICE_LABEL,
        'N_entries_carry_the_variant_and_its_label': all(
            e.get('model_variant') == MODEL_VARIANT_N and e.get('model_variant_label') == H.MODEL_VARIANT_LABEL
            for e in ents if expected_keys.get(e['label'], (None, None))[1] == 'N'),
        'A_entries_carry_no_variant': all('model_variant' not in e and 'model_variant_label' not in e
                                          for e in ents if expected_keys.get(e['label'], (None, None))[1] == 'A'),
        'spec_variant_label_iff_N_entry': ((spec.get('model_variant_label') == H.MODEL_VARIANT_LABEL) == has_n
                                           and (has_n or 'model_variant_label' not in spec)),
        'no_rho0_scale_or_nrf_option': (not any(k in e for e in ents for k in ('rho0_scale', 'nrf_option',
                                                                                  'rho0_scale_label',
                                                                                  'nrf_option_label'))
                                        and 'rho0_scale_label' not in spec and 'nrf_option_label' not in spec),
        'no_wrapper_declaration': not any(k in e for e in ents for k in WRAPPER_KEYS),
        'no_post_certification': all(e.get('post_certification') is None for e in ents),
        'overrides_empty': cfg.get('overrides') == {} and all(e.get('overrides') == {} for e in ents),
        'cap_500': spec.get('cap') == CAP, 'ten_cycles': spec.get('required_consecutive_cycles') == 10,
        'concurrency_recorded': spec.get('concurrency') == int(concurrency)
        == (extra.get('concurrency') or {}).get('declared'),
        'memory_preflight_recorded': bool((extra.get('concurrency') or {}).get('memory_preflight_at_freeze_non_gating')),
        'pins_recorded': all(pins_recorded_check(extra.get('pins') or {}).values()),
        'e1_root_recorded': extra.get('e1_root') == root_rel, 'phase_recorded': extra.get('phase') == phase,
        'script_recorded': extra.get('campaign_script') == SCRIPT_NAME,
        'harness_sha256_recorded': (spec.get('harness') or {}).get('sha256') == H.sha256_file(H.HARNESS_PATH),
    }
    return checks, vs6


# ======================================================================================================================
#  the committed final-code cells: base keys
# ======================================================================================================================
def cell_key_inputs(cfg, entry):
    aa = cfg.get('case_file_anderson_acceleration')
    return {'candidate_key': entry['key'], 'overrides': entry.get('overrides') or {},
            'effective_anderson_acceleration': (H.effective_anderson_acceleration(aa, entry.get('overrides'))
                                                if aa is not None else None),
            'model_variant': H.validate_model_variant(entry.get('model_variant')),
            'ess_ageing_baseline': H.validate_ess_ageing_baseline(cfg.get('ess_ageing_baseline')),
            'flex_price_multiplier_in_key': H.flex_price_multiplier_in_key(entry.get('flex_price_multiplier')),
            'convergence_depth_tail_in_key': H.convergence_depth_tail_in_key(cfg.get('convergence_depth_tail')),
            'derived_instance': (H.derived_instance_identity(cfg['derived_instance'])
                                 if cfg.get('derived_instance') else None),
            'interface_deviation_premium': entry.get('interface_deviation_premium'),
            'rho0_scale_in_key': H.rho0_scale_in_key(entry.get('rho0_scale')),
            'nrf_option_in_key': H.nrf_option_in_key(entry.get('nrf_option'))}


def _cell_key(cfg, entry, with_wrappers):
    kw = dict(case_file_aa=cfg.get('case_file_anderson_acceleration'), model_variant=entry.get('model_variant'),
              ess_ageing_baseline=cfg.get('ess_ageing_baseline'),
              flex_price_multiplier=entry.get('flex_price_multiplier'),
              derived_instance=cfg.get('derived_instance'),
              interface_deviation_premium=entry.get('interface_deviation_premium'),
              convergence_depth_tail=cfg.get('convergence_depth_tail'), rho0_scale=entry.get('rho0_scale'),
              nrf_option=entry.get('nrf_option'))
    if with_wrappers:
        kw.update({w: entry.get(w) for w in WRAPPER_KEYS})
    return H.evaluation_key(entry['key'], entry.get('overrides') or {}, **kw)


def committed_cell_comparison(plan):
    """Informative (not forced): for each committed final-code cell whose (candidate, model, price) coincides with a
    seed, the cell's BASE key recomputed from its committed spec entry (the key without any wrapper declaration) against
    the launcher's key, and exactly which key inputs differ."""
    w182f_sha = _sha_or_none(W182F_RECORDS['path'])
    w182f_rows = {r['no']: r for r in _load(W182F_RECORDS['path'])['Z3_e21_code_heads']['rows']} \
        if w182f_sha == W182F_RECORDS['sha256'] else {}
    out = []
    for cell in COMMITTED_CELLS:
        spec_sha = _sha_or_none(cell['spec'])
        rec = {k: cell[k] for k in ('row', 'cell', 'seed', 'model', 'flex', 'source', 'spec', 'spec_sha256',
                                    'eval_key')}
        rec['spec_sha256_holds'] = spec_sha == cell['spec_sha256']
        rec['spec_git_tracked_and_clean'] = all(_git_tracked_clean(cell['spec']))
        w = w182f_rows.get(cell['row'])
        if cell['source'].startswith('W182f'):
            rec['w182f_row_names_this_spec_and_key'] = bool(w) and (w.get('campaign_spec') == cell['spec']
                                                                     and w.get('eval_key') == cell['eval_key'])
        spec = _load(cell['spec'])
        hits = [e for e in spec['candidates'] if e.get('eval_key') == cell['eval_key']]
        rec['entry_found_uniquely'] = len(hits) == 1
        if len(hits) != 1:
            out.append(rec)
            continue
        e, cfg = hits[0], spec['configuration']
        rec['wrappers_declared'] = [w_ for w_ in WRAPPER_KEYS if e.get(w_) is not None]
        rec['committed_eval_key_recomputes'] = _cell_key(cfg, e, True) == cell['eval_key']
        base = _cell_key(cfg, e, False)
        rec['committed_base_key'] = base
        if cell.get('expected_base_key'):
            rec['committed_base_key_equals_expected'] = base == cell['expected_base_key']
        seed_row = next(r for r in plan if r['seed'] == cell['seed'] and r['model'] == cell['model'])
        rec['launcher_label'] = seed_row['label']
        rec['launcher_key_2x'] = seed_row['eval_key']
        rec['launcher_disposition'] = seed_row['disposition']
        cell_in, launcher_in = cell_key_inputs(cfg, e), launcher_key_inputs(seed_row['canonical'], cell['model'])
        rec['base_key_equals_launcher_key'] = base == seed_row['eval_key']
        rec['differing_key_inputs'] = {k: {'committed_cell': cell_in.get(k), 'launcher': launcher_in.get(k)}
                                       for k in sorted(set(cell_in) | set(launcher_in))
                                       if cell_in.get(k) != launcher_in.get(k)}
        if cell['flex'] != FLEX_M:   # the cell is at another price: the launcher's key at the cell's price, beside
            k_same = eval_key(seed_row['canonical'], cell['model'], flex=cell['flex'])
            rec['launcher_key_at_the_cell_price'] = k_same
            rec['base_key_equals_launcher_key_at_the_cell_price'] = base == k_same
            li = launcher_key_inputs(seed_row['canonical'], cell['model'], flex=cell['flex'])
            rec['differing_key_inputs_at_the_cell_price'] = {
                k: {'committed_cell': cell_in.get(k), 'launcher': li.get(k)}
                for k in sorted(set(cell_in) | set(li)) if cell_in.get(k) != li.get(k)}
        rec['non_key_differences'] = {
            'cap': {'committed_cell': spec.get('cap'), 'launcher': CAP},
            'post_certification': {'committed_cell': e.get('post_certification'), 'launcher': None},
            'stop_rule': {'committed_cell': ('re-settle declaration: schema '
                                             + str((e.get('settling_resettle') or {}).get('schema'))
                                             + ', settling rule ' + str(((e.get('settling_resettle') or {})
                                                                         .get('settling_rule') or {}).get('class'))
                                             if e.get('settling_resettle') else 'production exit'),
                          'launcher': 'production exit (10 consecutive passing cycles)'}}
        out.append(rec)
    return {'w182f_records': dict(W182F_RECORDS, sha256_on_disk=w182f_sha), 'cells': out}


def first_poll_by_seed(lattice, sigma_q):
    """Informative, zero solves: the first variant-B poll (k = HALTON_K0) the search would list if each seed were the
    arm's initial incumbent (the incumbent is decided only by the seed evaluations): distinct feasible poll points,
    whether the completion is triggered, its size against the cap of 30 (a refusal stops the search for review), and
    how many points are not seeds. Pure `C.build_poll` with a cache holding only the seed keys."""
    out = {}
    for arm in MODELS:
        key_of = make_arm_key_of(lattice, arm)
        seed_keys = {key_of(z): s for s, z, _n, _y, _t in SEEDS}
        cache = {k: {'label': s} for k, s in seed_keys.items()}
        for seed, z, _n, _y, _t in SEEDS:
            inc = {'eval_key': key_of(z), 'z': z, 'label': lattice.label(z), 'I': lattice.investment_cost(z),
                   'Q': None, 'F': None, 'bar': None}
            record, completion = C.build_poll(lattice, cache, key_of, inc, HALTON_K0, sigma_q,
                                              completion_cap=COMPLETION_CAP)
            polled = [c for c in record['candidates'] if c['disposition'] in ('cache_hit', 'new_evaluation')]
            over = bool(completion and completion['over_cap'])
            out[f'{seed}_{arm}'] = {
                'incumbent': lattice.label(z), 'halton_t': record['halton_t'],
                'n_distinct_feasible_poll_points_after_snap': record['n_distinct_feasible_poll_points'],
                'completion_triggered': record['completion_triggered'],
                'completion_n_feasible': (completion or {}).get('n_feasible'),
                'completion_over_cap_30_STOP_FOR_REVIEW': over,
                'n_poll_points': None if over else len(polled),
                'n_poll_points_that_are_seeds': None if over else sum(c['disposition'] == 'cache_hit' for c in polled),
                'n_new_evaluations_needed': None if over else sum(c['disposition'] == 'new_evaluation'
                                                                  for c in polled)}
            by_k = []
            for k in range(HALTON_K0, HALTON_K0 + MAX_POLLS):   # the same incumbent at later Halton indices
                rk, ck = C.build_poll(lattice, cache, key_of, inc, k, sigma_q, completion_cap=COMPLETION_CAP)
                by_k.append({'k': k, 'halton_t': rk['halton_t'],
                             'n_distinct': rk['n_distinct_feasible_poll_points'],
                             'completion_triggered': rk['completion_triggered'],
                             'over_cap': bool(ck and ck['over_cap'])})
            out[f'{seed}_{arm}']['same_incumbent_at_k_0_to_9'] = by_k
    return out


# ======================================================================================================================
#  the planted cache-scoping test (scratch, outside the repo; no solve)
# ======================================================================================================================
def planted_scope_test(lattice, plan, scratch, g0):
    """Two sibling planted roots: r1 holds a seeds phase with A and N points and the alias; r2 holds another seeds
    phase with points of the same keys. The loader on r1 must return exactly r1's arm points (+ the alias for N),
    never r2's; a point of r1 whose eval_dir points into r2 must make the load refuse."""
    base = os.path.join(scratch, 'planted_scope')
    os.makedirs(base)
    out = {}

    def _plant(rname, bad_eval_dir=False):
        r_abs = os.path.join(base, rname)
        r_rel = f'planted_{rname}'
        sdir = phase_root(r_abs, r_rel, 'seeds')
        os.makedirs(sdir)
        cid = campaign_id(r_rel, 'seeds')
        spec = {'campaign_id': cid, 'candidates': [
            {'label': r['label'], 'eval_key': r['eval_key'], 'flex_price_multiplier': FLEX_M,
             **({'model_variant': r['model_variant']} if r['model_variant'] else {})}
            for r in plan if r['disposition'] == 'evaluate'],
            'extra': {'e1_root': r_rel, 'campaign_script': SCRIPT_NAME, 'phase': 'seeds'}}
        spec_path = os.path.join(sdir, 'campaign_spec_planted.json')
        with open(spec_path, 'w') as handle:
            json.dump(spec, handle)
        points, manifest = {}, {}
        for i, r in enumerate(x for x in plan if x['disposition'] == 'evaluate'):
            ed_abs = os.path.join(sdir, 'evals', r['label'])
            if bad_eval_dir and i == 0:
                ed_abs = os.path.join(base, 'r2', 'stray', r['label'])
            os.makedirs(ed_abs, exist_ok=True)
            rec_path = os.path.join(ed_abs, 'evaluation_record.json')
            with open(rec_path, 'w') as handle:
                json.dump({'synthetic': rname, 'label': r['label']}, handle)
            ed_rel = _rel(ed_abs)
            manifest[os.path.join(ed_rel, 'evaluation_record.json')] = H.sha256_file(rec_path)
            points[r['label']] = {'model': r['model'], 'eval_key': r['eval_key'], 'candidate_canonical': r['canonical'],
                                  'status': 'certified', 'certified_cost_gross_settlement_excluded':
                                  (1.0e8 if rname == 'r1' else 9.0e8) + i, 'bar': {'value': 1.0},
                                  'barrier_cause': None, 'eval_dir': ed_rel}
        alias = alias_record(plan, 'planted', {'sha256_on_disk': g0[1].get('sha256_on_disk'), 'accepted': True})
        results = {'campaign_id': cid, 'phase': 'seeds', 'campaign_spec_sha256': H.sha256_file(spec_path),
                   'points': points, 'aliases': [alias]}
        with open(os.path.join(sdir, RESULTS_FILE), 'w') as handle:
            json.dump(results, handle)
        with open(os.path.join(sdir, MANIFEST_FILE), 'w') as handle:
            json.dump(manifest, handle)
        return r_abs, r_rel

    g0_ok = (True, {'sha256_on_disk': g0[1].get('sha256_on_disk')})
    r2_abs, r2_rel = _plant('r2')
    r1_abs, r1_rel = _plant('r1')
    for arm in ('N', 'A'):
        ok, info, cache = load_seed_cache(r1_abs, r1_rel, arm, lattice, g0=g0_ok, require_git=False)
        expected = {r['eval_key'] for r in plan if r['model'] == arm}
        out[f'r1_{arm}'] = {
            'ok': ok, 'n_entries': len(cache), 'keys_equal_expected': set(cache) == expected,
            'every_entry_from_r1': all((v['source'].get('root') == r1_rel) for v in cache.values()),
            'no_r2_cost': all((v['Q'] or 0) < 9.0e8 for v in cache.values()),
            'n_alias_entries': sum(1 for v in cache.values() if 'alias' in v),
            'other_model_points_skipped': len(info.get('skipped_other_model') or [])}
    r3_abs, r3_rel = (os.path.join(base, 'r3'), 'planted_r3')
    _plant('r3', bad_eval_dir=True)
    ok3, info3, cache3 = load_seed_cache(r3_abs, r3_rel, 'A', lattice, g0=g0_ok, require_git=False)
    out['r3_stray_eval_dir_refused'] = {'ok': ok3, 'reason': info3.get('reason'), 'n_entries': len(cache3)}
    checks = {
        'r1_N_ok_and_exact': out['r1_N']['ok'] and out['r1_N']['keys_equal_expected'],
        'r1_N_has_the_alias_once': out['r1_N']['n_alias_entries'] == 1,
        'r1_A_ok_and_exact': out['r1_A']['ok'] and out['r1_A']['keys_equal_expected'],
        'r1_A_has_no_alias': out['r1_A']['n_alias_entries'] == 0,
        'never_reads_the_sibling_root': all(out[f'r1_{a}']['every_entry_from_r1'] and out[f'r1_{a}']['no_r2_cost']
                                            for a in ('N', 'A')),
        'stray_eval_dir_refuses_the_load': (not ok3) and 'eval_dir_inside_root' in (info3.get('reason') or ''),
    }
    shutil.rmtree(base)
    return {'checks': checks, 'detail': out}


# ======================================================================================================================
#  --dry-run
# ======================================================================================================================
def dry_run(args, root_rel, root_abs):
    started = time.time()
    res = {'schema': 'p515_s55_e1_launcher_dryrun_v2', 'stage': STAGE, 'authority': AUTHORITY, 'started_utc': _utc(),
           'argv': sys.argv, 'interpreter': sys.executable, 'git_head': H._git(['rev-parse', 'HEAD']),
           'objective_convention': OBJECTIVE_CONVENTION + ' (no objective value is computed by this dry run)',
           'root': root_rel, 'concurrency': args.concurrency}
    assertions, informative = {}, {}
    inv = root_inventory(root_abs, root_rel)
    res['root_inventory_at_start'] = inv
    ref = root_refusals(inv, 'dry_run')
    if ref:
        raise SystemExit(f'REFUSED (root): {ref}')
    scratch = os.path.abspath(args.scratch)
    if scratch.startswith(REPO + os.sep) or os.path.exists(scratch):
        raise SystemExit(f'REFUSED: --scratch must be a NEW directory outside the repository: {scratch}')
    os.makedirs(scratch)
    lattice, lev, lfail = build_lattice()
    if lfail:
        raise SystemExit(f'REFUSED (search tables): {lfail}')
    res['search_tables'] = lev
    # 1. the seed key set under both models at 2x
    plan = seed_plan(lattice)
    checks, summary = seed_key_assertions(plan)
    assertions['seed_keys'] = checks
    holders, n_specs = committed_eval_key_holders([r['eval_key'] for r in plan])
    res['seed_key_summary'] = summary
    res['key_table'] = key_table(plan, holders)
    res['committed_specs_scanned'] = n_specs
    res['prediction'] = {'source': 'Addendum 83 W183 predictions: "the E1 key dry run yields 7 distinct seed keys"',
                         'predicted_distinct_keys_to_evaluate': 7,
                         'observed_distinct_keys_to_evaluate': summary['n_distinct_keys_to_evaluate'],
                         'held': summary['n_distinct_keys_to_evaluate'] == 7}
    # 2. G0 and the alias
    x0_keys = {m: next(r['eval_key'] for r in plan if r['seed'] == 'x0' and r['model'] == m) for m in MODELS}
    g0_ok, g0_info = g0_check(args.g0_artifact, args.g0_sha256, x0_keys, require_current_production=False)
    res['g0'] = g0_info
    assertions['g0_alias_accepted'] = {'accepted_for_the_dry_run': g0_ok}
    informative['g0_artifact_production_equals_current'] = g0_info.get('artifact_production', {}).get(
        'equals_current_production')
    res['alias'] = alias_record(plan, args.g0_artifact, g0_info)
    # 3. configuration block and the would-be seeds spec, frozen IN SCRATCH by the harness's own function
    rule11 = rule_eleven(lattice, lev['sigma_Q']['sigma_Q_eur'])
    res['rule_eleven'] = rule11
    assertions['rule_eleven'] = {k: v for k, v in rule11['checks'].items()}
    pins = pins_now()
    pre = memory_preflight(args.concurrency)
    res['memory_preflight_non_gating'] = pre
    res['memory_preflight_informative'] = {str(n): memory_preflight(n) for n in (1, 4)}
    entries, rows = seeds_entries(lattice, plan, None)
    extra = common_extra(root_rel, 'seeds', args.concurrency, pins, pre, lev, rule11,
                         {'accepted': g0_ok, **g0_info})
    extra.update({'aliases': [res['alias']], 'seed_plan': key_table(plan), 'dry_run_scratch_freeze': True})
    scratch_root = os.path.join(scratch, f"campaign_{campaign_id(root_rel, 'seeds')}")
    spec_path, spec_sha, _ = H.freeze_campaign_spec(scratch_root, campaign_id(root_rel, 'seeds'), entries,
                                                     configuration(), CAP, args.concurrency, AUTHORITY,
                                                     REQUIRED_CONSECUTIVE_CYCLES, extra)
    with open(spec_path) as handle:
        spec = json.load(handle)
    expected = {r['label']: (r['eval_key'], r['model']) for r in rows}
    vchecks, vs6 = validate_spec(spec, 'seeds', root_rel, args.concurrency, expected)
    assertions['would_be_seeds_spec'] = vchecks
    assertions['pins_recorded'] = pins_recorded_check(spec['extra']['pins'])
    res['would_be_seeds_spec'] = {'frozen_in_scratch_by': 'p515_s44_campaign_harness.freeze_campaign_spec',
                                  'scratch_path': spec_path, 'sha256': spec_sha,
                                  'note': ('not written into the repository under a campaign_spec_* name (a dry-run '
                                           'spec must not enter the committed key space); its content is embedded'),
                                  'content': spec}
    res['configuration_vs_ext_v6'] = vs6
    assertions['configuration'] = {'equals_ext_v6_minus_resettle': vs6['equal_except_text'],
                                   'ext_v6_reference_sha256_holds': vs6['reference_sha256_holds'],
                                   'ext_v6_seven_specs_agree': vs6['ext_v6_specs_agree'],
                                   'ext_v6_reference_entry_had_the_resettle': vs6[
                                       'ext_v6_reference_entry_carries_settling_resettle']}
    res['pins'] = pins
    informative['pinned_files_uncommitted_or_untracked'] = {'uncommitted': pins['uncommitted_changes'],
                                                            'untracked': pins['untracked']}
    if not args.allow_uncommitted:
        assertions['pins_recorded']['every_pinned_file_committed_and_clean'] = (
            not pins['uncommitted_changes'] and not pins['untracked'])
    # 4. the domain keys of the two search arms
    dom = lattice.domain()
    keys = {arm: [make_arm_key_of(lattice, arm)(z) for z in dom] for arm in MODELS}
    res['domain_keys'] = {'n_points': len(dom), **{f'n_distinct_{a}': len(set(keys[a])) for a in MODELS},
                          'A_N_disjoint': not (set(keys['A']) & set(keys['N']))}
    assertions['domain_keys'] = {
        'domain_1705': len(dom) == 1705,
        'keys_distinct_per_arm': all(len(set(keys[a])) == len(dom) for a in MODELS),
        'A_and_N_key_sets_disjoint': res['domain_keys']['A_N_disjoint'],
        'every_seed_key_in_its_arm_domain': all(r['eval_key'] in set(keys[r['model']]) for r in plan)}
    # 5. cache scoping
    for arm in ('N', 'A'):
        ok, info, cache = load_seed_cache(root_abs, root_rel, arm, lattice, g0=(g0_ok, g0_info))
        res[f'cache_loader_on_this_root_{arm}'] = {'ok': ok, 'n_entries': len(cache), **info}
    planted = planted_scope_test(lattice, plan, scratch, (g0_ok, g0_info))
    res['cache_scope_planted_test'] = planted
    assertions['cache_scope'] = dict(planted['checks'], **{
        f'this_root_has_no_seeds_so_arm_{a}_refuses': res[f'cache_loader_on_this_root_{a}']['ok'] is False
        for a in ('N', 'A')})
    # 6. concurrency and preflight recorded (the preflight is non-gating here)
    assertions['concurrency_and_preflight'] = {
        'concurrency_in_spec': spec.get('concurrency') == args.concurrency,
        'preflight_recorded_in_spec': bool(spec['extra']['concurrency']['memory_preflight_at_freeze_non_gating']),
        'preflight_rule_is_n_x_2_75_gib': spec['extra']['concurrency']['memory_per_slot_gib'] == 2.75}
    informative['memory_preflight_pass_at_declared_concurrency'] = pre.get('pass')
    # 7. the committed final-code cells (informative)
    res['committed_cell_comparison'] = committed_cell_comparison(plan)
    res['first_poll_if_each_seed_were_the_incumbent'] = first_poll_by_seed(lattice, lev['sigma_Q']['sigma_Q_eur'])
    # 8. preconditions as --run would check them (reported, not gating here)
    res['preconditions_reported'] = {
        'check_campaign_preconditions_on_a_new_scratch_root': H.check_campaign_preconditions(
            os.path.join(scratch, 'not_created'), extra_clean_files=(SCRIPT_NAME,) + SETTLING_MODULES),
        'own_launcher_processes_alive': _own_process_alive(),
        'campaign_lock_present': os.path.exists(H.CAMPAIGN_LOCK_PATH)}
    shutil.rmtree(scratch)
    res['scratch_removed'] = not os.path.exists(scratch)
    res['informative'] = informative
    res['assertions'] = assertions
    res['repo_modules_imported_sha256_end'] = imported_repo_modules()
    g = guards_report()
    res['guards'] = g
    assertions['guards'] = {'zero_solves_pickle_build_tripwires': g['all_zero']}
    failing = sorted(f'{grp}.{k}' for grp, cs in assertions.items() for k, v in cs.items() if v is not True)
    res['failing_assertions'] = failing
    res['all_assertions_hold'] = not failing
    res['finished_utc'] = _utc()
    res['wall_clock_s'] = time.time() - started
    out_path = os.path.join(root_abs, DRY_RUN_FILE)
    sha = _write_once_json(out_path, res)
    manifest = {_rel(out_path): sha, f'{SCRIPT_NAME}': _sha(SCRIPT_NAME),
                'p515_s44_campaign_harness.py': _sha('p515_s44_campaign_harness.py'),
                'p515_s53_f2_certificate.py': _sha('p515_s53_f2_certificate.py'),
                'p515_s47_phase_b_record.py': _sha('p515_s47_phase_b_record.py'),
                args.g0_artifact: _sha_or_none(args.g0_artifact)}
    _write_once_json(os.path.join(root_abs, DRY_RUN_MANIFEST), manifest)
    _log(f'dry run: key table ({len(plan)} seed x model rows):')
    for r in res['key_table']:
        _log(f"  {r['label']:<16} {r['disposition']:<8} eval {r['eval_key'][:16]} "
             f"{('-> ' + r['alias_of_eval_key'][:16]) if r['alias_of_eval_key'] else ''} "
             f"holders {len(r['committed_campaign_specs_holding_this_eval_key'])}")
    _log(f"prediction 7 distinct seed keys: observed {summary['n_distinct_keys_to_evaluate']} "
         f"({'HELD' if res['prediction']['held'] else 'NOT HELD'}); aliases {summary['n_aliases']}")
    for c in res['committed_cell_comparison']['cells']:
        _log(f"  row {c['row']} {c['cell'][:40]}: base {str(c.get('committed_base_key'))[:16]} vs launcher "
             f"{str(c.get('launcher_key_2x'))[:16]} equal={c.get('base_key_equals_launcher_key')} "
             f"differing={sorted((c.get('differing_key_inputs') or {}).keys())}")
    for k, v in res['first_poll_if_each_seed_were_the_incumbent'].items():
        _log(f"  first poll if {k:<12} were the incumbent: distinct {v['n_distinct_feasible_poll_points_after_snap']}"
             f" completion {v['completion_triggered']} ({v['completion_n_feasible']}) over cap "
             f"{v['completion_over_cap_30_STOP_FOR_REVIEW']} new {v['n_new_evaluations_needed']}")
    _log(f"G0 accepted for the dry run: {g0_ok}; artifact production equals current: "
         f"{informative['g0_artifact_production_equals_current']}")
    _log(f"guards {g['all_zero']} solve {[(n, v['counts']) for n, v in g['solve_profile_guards'].items()]}")
    _log(f'wrote {_rel(out_path)} sha256 {sha}; failing assertions {failing}')
    return 0 if not failing else 1


# ======================================================================================================================
#  --freeze
# ======================================================================================================================
def freeze(args, root_rel, root_abs):
    phase = args.phase
    inv = root_inventory(root_abs, root_rel)
    refusals = root_refusals(inv, 'freeze', phase)
    croot = phase_root(root_abs, root_rel, phase)
    refusals += H.check_campaign_preconditions(croot, extra_clean_files=(SCRIPT_NAME,) + SETTLING_MODULES)
    lattice, lev, lfail = build_lattice()
    refusals += lfail
    if lattice is None:
        raise SystemExit(f'REFUSED: {refusals}')
    plan = seed_plan(lattice)
    s_checks, summary = seed_key_assertions(plan)
    refusals += [f'seed key check failed: {k}' for k, v in s_checks.items() if not v]
    x0_keys = {m: next(r['eval_key'] for r in plan if r['seed'] == 'x0' and r['model'] == m) for m in MODELS}
    try:
        rule11 = rule_eleven(lattice, lev['sigma_Q']['sigma_Q_eur'])
    except AssertionError as error:
        raise SystemExit(f'REFUSED: {error}')
    pins = pins_now()
    if pins['uncommitted_changes'] or pins['untracked']:
        refusals.append(f"pinned files not committed and clean: {pins['uncommitted_changes']} {pins['untracked']}")
    pre = memory_preflight(args.concurrency)
    cid = campaign_id(root_rel, phase)
    if phase == 'seeds':
        subset = set(args.seeds.split(',')) if args.seeds else None
        valid = {r['label'] for r in plan if r['disposition'] == 'evaluate'}
        if subset is not None and not subset <= valid:
            refusals.append(f'--seeds names unknown or alias labels: {sorted(subset - valid)}; valid {sorted(valid)}')
        entries, rows = seeds_entries(lattice, plan, subset)
        refusals += declared_root_refusals(root_rel, phase, [r['label'] for r in rows], args.concurrency,
                                           sorted(valid))
        needs_alias = any(r['seed'] == ALIAS_SEED and r['model'] == ALIAS_TARGET_MODEL for r in rows)
        g0_ok, g0_info = g0_check(args.g0_artifact, args.g0_sha256, x0_keys, require_current_production=True)
        if needs_alias and not g0_ok:
            refusals.append(f"G0 artifact not accepted for the x = 0 alias: {g0_info.get('checks')}")
        extra = common_extra(root_rel, phase, args.concurrency, pins, pre, lev, rule11, {'accepted': g0_ok, **g0_info})
        extra.update({'aliases': [alias_record(plan, args.g0_artifact, g0_info)] if needs_alias else [],
                      'seed_plan': key_table(plan), 'seed_subset': sorted(subset) if subset else 'all seven'})
        expected = {r['label']: (r['eval_key'], r['model']) for r in rows}
    else:
        arm = phase.split('_')[1]
        refusals += declared_root_refusals(root_rel, phase, None, args.concurrency, None)
        g0 = g0_check(args.g0_artifact, args.g0_sha256, x0_keys, require_current_production=True) \
            if arm == ALIAS_MODEL else (None, {'not_used_by_arm_A': True})
        ok, cinfo, cache = load_seed_cache(root_abs, root_rel, arm, lattice, g0=g0 if arm == 'N' else None)
        if not ok:
            refusals.append(f"seed cache refused: {cinfo.get('reason')}")
        inc, inc_rec = initial_incumbent(lattice, cache, lev['sigma_Q']['sigma_Q_eur']) if ok else (None, None)
        if ok and inc is None:
            refusals.append(f'no initial incumbent: {inc_rec}')
        entries = domain_entries(lattice, arm)
        extra = common_extra(root_rel, phase, args.concurrency, pins, pre, lev, rule11,
                             {'accepted': g0[0], **g0[1]} if arm == 'N' else g0[1])
        key_of = make_arm_key_of(lattice, arm)
        extra.update({'arm': arm, 'cache_source': cinfo,
                      'cache_table': {k: _jsonable(v) for k, v in sorted(cache.items())},
                      'initial_incumbent': inc_rec,
                      'domain_summary': {'n_points': len(lattice.domain())}})
        expected = {domain_label(lattice, z, arm): (key_of(z), arm) for z in lattice.domain()}
    if refusals:
        for r in refusals:
            _log(f'[FREEZE REFUSED] {r}')
        raise SystemExit(1)
    spec_path, spec_sha, _ = H.freeze_campaign_spec(croot, cid, entries, configuration(), CAP, args.concurrency,
                                                     AUTHORITY, REQUIRED_CONSECUTIVE_CYCLES, extra)
    with open(spec_path) as handle:
        spec = json.load(handle)
    vchecks, _vs6 = validate_spec(spec, phase, root_rel, args.concurrency, expected)
    g = guards_report()
    ok = all(vchecks.values()) and g['all_zero']
    _log(f'frozen {_rel(spec_path)} sha256 {spec_sha}; spec checks failing '
         f'{[k for k, v in vchecks.items() if not v]}; guards {g["all_zero"]}')
    _log(f'run with: --run --phase {phase} --root {root_rel} --spec-sha256 {spec_sha}')
    return 0 if ok else 1


# ======================================================================================================================
#  --run
# ======================================================================================================================
def _run_preconditions(args, root_rel, root_abs):
    phase = args.phase
    croot = phase_root(root_abs, root_rel, phase)
    inv = root_inventory(root_abs, root_rel)
    failures = root_refusals(inv, 'run', phase)
    if failures:
        return failures, None, None, None
    spec_path, spec = H.load_frozen_spec(croot, args.spec_sha256)
    failures += [f for f in H.check_campaign_preconditions(croot, extra_clean_files=(SCRIPT_NAME,) + SETTLING_MODULES)
                 if f != f'campaign root already exists (write-once): {croot}']
    extra = spec.get('extra') or {}
    if extra.get('phase') != phase or extra.get('e1_root') != root_rel or extra.get('campaign_script') != SCRIPT_NAME:
        failures.append('the frozen spec is not this launcher\'s spec of this phase in this root')
    diffs = pins_compare(extra['pins'], pins_now())
    failures += [f'PIN DIFFERS: {d}' for d in diffs]
    for what, frozen, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE)),
                              ('this launcher', extra.get('campaign_script_sha256'), _sha(SCRIPT_NAME))):
        if frozen != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    memory = memory_preflight(spec['concurrency'])
    if not memory['pass']:
        failures.append(f'memory preflight REFUSED (gating at --run): {memory}')
    return failures, spec_path, spec, memory


def _write_results(croot, results):
    _write_once_json(os.path.join(croot, RESULTS_FILE), results)
    manifest = {}
    for directory, _dirs, files in os.walk(croot):
        for fname in sorted(files):
            fpath = os.path.join(directory, fname)
            manifest[_rel(fpath)] = H.sha256_file(fpath)
    _write_once_json(os.path.join(croot, MANIFEST_FILE), manifest)


def run(args, root_rel, root_abs):
    started = time.time()
    phase = args.phase
    croot = phase_root(root_abs, root_rel, phase)
    lattice, lev, lfail = build_lattice()
    failures, spec_path, spec, memory = _run_preconditions(args, root_rel, root_abs)
    failures += lfail
    if spec is None or failures:
        for f in failures:
            _log(f'[RUN PRECONDITION FAILED] {f}')
        return 1
    extra = spec['extra']
    plan = seed_plan(lattice)
    x0_keys = {m: next(r['eval_key'] for r in plan if r['seed'] == 'x0' and r['model'] == m) for m in MODELS}
    try:
        rule11 = rule_eleven(lattice, lev['sigma_Q']['sigma_Q_eur'])
    except AssertionError as error:
        _log(f'[RUN PRECONDITION FAILED] {error}')
        return 1
    sigma_q = lev['sigma_Q']['sigma_Q_eur']
    by_label = {e['label']: e for e in spec['candidates']}
    declared = declared_root_refusals(root_rel, phase, list(by_label) if phase == 'seeds' else None,
                                      spec.get('concurrency'),
                                      [r['label'] for r in plan if r['disposition'] == 'evaluate'])
    if declared:
        for f in declared:
            _log(f'[RUN PRECONDITION FAILED] {f}')
        return 1
    if phase == 'seeds':
        g0 = g0_check(args.g0_artifact, args.g0_sha256, x0_keys, require_current_production=True) \
            if extra.get('aliases') else (None, {})
        if extra.get('aliases') and not (g0[0] and g0[1].get('sha256_on_disk')
                                         == extra['aliases'][0]['g0_artifact_sha256']):
            _log(f'[RUN PRECONDITION FAILED] G0 alias not accepted now: {g0[1].get("checks")}')
            return 1
    else:
        arm = extra['arm']
        g0 = g0_check(args.g0_artifact, args.g0_sha256, x0_keys, require_current_production=True) \
            if arm == ALIAS_MODEL else None
        ok, cinfo, cache = load_seed_cache(root_abs, root_rel, arm, lattice, g0=g0)
        if not ok or {k: _jsonable(v) for k, v in sorted(cache.items())} != extra['cache_table']:
            _log(f"[RUN PRECONDITION FAILED] seed cache refused or differs from the frozen table: {cinfo.get('reason')}")
            return 1
        inc, inc_rec = initial_incumbent(lattice, cache, sigma_q)
        if _jsonable(inc_rec) != extra['initial_incumbent']:
            _log('[RUN PRECONDITION FAILED] initial incumbent differs from the frozen one')
            return 1
    head = H._git(['rev-parse', 'HEAD'])
    lock = H.acquire_campaign_lock(spec['campaign_id'], args.spec_sha256)
    _log(f'campaign lock acquired: {lock}; HEAD {head}; memory {memory}')
    ctx = H.CampaignContext(croot, spec_path, args.spec_sha256, spec, log=_log)
    points, batch_infos = {}, []
    try:
        if phase == 'seeds':
            labels = [e['label'] for e in spec['candidates']]
            H.evaluate.last_batch_info = {}
            recs = H.evaluate(labels, ctx, heartbeat_tag='seeds')
            batch_infos.append({'labels': labels, **getattr(H.evaluate, 'last_batch_info', {})})
            seed_of = {r['label']: (r['seed'], r['model']) for r in plan}
            for lab, rec in zip(labels, recs):
                seed, model = seed_of[lab]
                points[lab] = point_from_harness(rec, by_label[lab], model, seed)
            result = None
        else:
            key_of = make_arm_key_of(lattice, arm)
            label_of_key = {e['eval_key']: e['label'] for e in spec['candidates']}
            state_path = os.path.join(croot, 'e1_search_state.json')
            history = []
            counter = {'n': 0}

            def evaluate_fn(batch):
                keys = [key_of(z) for z in batch]
                labels = [label_of_key[k] for k in keys]
                counter['n'] += 1
                _log(f'evaluating batch {counter["n"]}: {labels}')
                H.evaluate.last_batch_info = {}
                recs = H.evaluate(labels, ctx, heartbeat_tag=f'batch_{counter["n"]:03d}')
                batch_infos.append({'labels': labels, **getattr(H.evaluate, 'last_batch_info', {})})
                out = []
                for k, lab, rec in zip(keys, labels, recs):
                    p = point_from_harness(rec, by_label[lab], arm)
                    points[lab] = p
                    out.append(cache_entry_from_point(p, root_rel, spec['campaign_id']))
                return out

            def on_poll(record):
                history.append(record)
                H._atomic_write_json(state_path, {'spec_sha256': args.spec_sha256, 'utc': _utc(),
                                                  'history': history, 'new_points': points})

            result = run_continuation_e1(lattice, cache, key_of, inc, evaluate_fn, sigma_q, k0=HALTON_K0,
                                         max_new_evaluations=N_MAX_NEW_EVALUATIONS, max_polls=MAX_POLLS,
                                         batch_size=int(spec['concurrency']), completion_cap=COMPLETION_CAP,
                                         log=_log, on_poll=on_poll, advance=True, advance_window=ADVANCE_WINDOW)
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log('campaign lock released')
    problems = {lab: point_problems(p) for lab, p in points.items() if point_problems(p)}
    g = guards_report()
    results = {
        'stage': STAGE, 'phase': phase, 'campaign_id': spec['campaign_id'], 'e1_root': root_rel,
        'timestamp_utc': _utc(), 'git_head_at_run': head, 'campaign_spec_path': _rel(spec_path),
        'campaign_spec_sha256': args.spec_sha256, 'objective_convention': OBJECTIVE_CONVENTION,
        'flex_price_multiplier': FLEX_M, 'points': points, 'aliases': extra.get('aliases') or [],
        'problems': problems, 'batch_info': batch_infos, 'memory_preflight_gating': memory,
        'rule_eleven_asserted_before_run': rule11['checks'], 'guards': g, 'wall_clock_s': time.time() - started}
    if result is not None:
        results.update({'arm': arm, 'termination': result['termination'],
                        'termination_certificate': result['termination_certificate'],
                        'STOP_FOR_REVIEW': result['termination']['reason'].startswith('STOP_FOR_REVIEW'),
                        'initial_incumbent': extra['initial_incumbent'], 'final_incumbent': result['incumbent'],
                        'n_polls': result['n_polls'], 'n_new_evaluations': result['n_new_evaluations'],
                        'n_barrier_new_evaluations': result['n_barrier_new_evaluations'],
                        'final_poll_unresolved_indeterminate': result['final_poll_unresolved_indeterminate'],
                        'lattice_neighbourhood_of_incumbent': result['lattice_neighbourhood_of_incumbent'],
                        'poll_history': result['history']})
    _write_results(croot, results)
    _log(f'results written; problems {problems}; guards {g["all_zero"]}')
    if problems or not g['all_zero']:
        return 1
    if result is None:
        return 0
    reason = result['termination']['reason']
    if reason == 'poll_failure_at_unit_mesh':
        return 0 if (result['termination_certificate'] or {}).get('holds') else 1
    return 3 if reason.startswith('STOP_FOR_REVIEW') else 2


# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--root', required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--concurrency', type=int)
    parser.add_argument('--spec-sha256')
    parser.add_argument('--seeds', help='seeds phase only: comma-separated seed labels (default: all seven)')
    parser.add_argument('--g0-artifact', required=True)
    parser.add_argument('--g0-sha256', required=True)
    parser.add_argument('--scratch', help='--dry-run only: a NEW directory outside the repository')
    parser.add_argument('--allow-uncommitted', action='store_true',
                        help='--dry-run smoke only: do not assert that every pinned file is committed and clean')
    args = parser.parse_args()
    alive = _own_process_alive()
    if alive:
        raise SystemExit(f'REFUSED: another copy of this launcher is alive: {alive}')
    root_rel, root_abs = resolve_root(args.root)
    if args.dry_run:
        if args.phase or args.spec_sha256 or args.seeds or not args.scratch or not args.concurrency:
            parser.error('--dry-run takes --root, --concurrency, --g0-artifact, --g0-sha256, --scratch only')
        code = dry_run(args, root_rel, root_abs)
    elif args.freeze:
        if not args.phase or not args.concurrency or args.spec_sha256 or args.scratch or args.allow_uncommitted:
            parser.error('--freeze takes --phase, --root, --concurrency, --g0-artifact, --g0-sha256 [--seeds]')
        if args.seeds and args.phase != 'seeds':
            parser.error('--seeds is for the seeds phase only')
        code = freeze(args, root_rel, root_abs)
    else:
        if not args.phase or not args.spec_sha256 or args.concurrency or args.seeds or args.scratch \
                or args.allow_uncommitted:
            parser.error('--run takes --phase, --root, --spec-sha256, --g0-artifact, --g0-sha256 '
                         '(the concurrency is the frozen spec\'s)')
        code = run(args, root_rel, root_abs)
    g = guards_report()
    _log(f"final guards: all_zero={g['all_zero']} pickle={g['pickle']['counts']} build={g['model_build_tripwire']} "
         f"tripwires_nonzero={[k for k, v in g['committed_launcher_function_tripwires'].items() if v]}")
    for _n, guard in reversed(all_guards()):
        guard.uninstall()
    sys.exit(code if g['all_zero'] else 1)


if __name__ == '__main__':
    main()
