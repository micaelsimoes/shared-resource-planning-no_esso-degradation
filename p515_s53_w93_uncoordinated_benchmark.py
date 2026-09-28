"""
P5.15 Addendum 49 (Planner task W93) -- the UNCOORDINATED BENCHMARK harness.

WRITTEN DURING THE 3x3 PAIR; NOTHING HERE HAS BEEN RUN (Addendum 49 "Timing": the Worker writes the function and its
tests during the pair and runs nothing). Static checks only at W93 (py_compile, AST). W94 (Planner rulings on the W93
report): the coupling check's fixed side gains the interface-voltage pin; the lambda_t channel-scaling correction and
the min-over-starts note are recorded. Static checks only at W94 as well.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 49 (rulings 1-3, the consistency convention, the resolution rule, the
timing, and the 2026-09-26 clarification on the tie-breaker and on lambda_t recovery); TASKS.md "Addendum 49 order";
Planner task W93. The production function is `uncoordinated_benchmark.run_operational_planning_uncoordinated`
(a new module beside `shared_resources_planning.py`; the retired `_run_operational_planning_without_coordination`
is untouched, its source sha256 pin re-checked below).

INSTANCE (recorded in every artifact): x = 0 on SRP1, candidate key 8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57
(sha256 of the canonical candidate {5: [0, 0], 7: [0, 0], 9: [0, 0]} at 2025, `p515_s44_campaign_harness.candidate_key`).
COORDINATED ARM (W106, Addendum 54 Ruling 3) = the SETTLED x = 0 cell `d110bd1a5977df1e_x0` (campaign
s53_w101_srp1_cont_x0, spec d1d5c3bc under stage spec v39 8a612429; evidence commit e4b5c992): bitwise replay through the
W86 certificate (cycle 132), then certified by the settling rule at k* = 181, Q181 = gross_operational_cost
653,873,702.1876609 (`settling_decision.json` Q_k_star = evaluation_record certified_cost). Its terminal (cycle-181) models
were persisted (certified_models.pkl sha256 99ab1070..., ESSO esso_models_s39_D.pkl sha256 118487e1..., both hash-recorded
in the committed campaign and child manifests). It is NOT re-run. Constant: `COORDINATED` (= `SETTLED`).
W86 (UNWIRED, RETAINED; W106): W93/W94 were written against the W86 tight-tail re-certification cell
`5cfe69a615ae3708_x0` (campaign s53_w86_tail_recert, spec ddd6cd44, certified at cycle 132, gross_operational_cost
653,858,731.5686293; certified_models.pkl sha256 d816bb4b..., ESSO 15f38409...). Addendum 53 found that cell unsettled
(certified at the trough of the post-AA fall); Addendum 54 Ruling 3 moved every coordinated-side input -- the common-Q
reference, the lambda_t models, the coupling check's pinned targets, the warm start, the coordinated curtailment figure
-- to the settled cycle-181 models. The `W86` constant and `_load_w86_models` are kept, unwired and unused (unwire,
never delete); `COORDINATED_PREVIOUS_REFERENCE` carries the W86 figures for REPORTING beside the settled ones only.
Nothing ran under W93/W94 (their output root `w93_uncoordinated` was never created).

OBJECTIVE CONVENTION ON EVERY OUTPUT: Q = gross_operational_cost, settlement EXCLUDED (production
`_get_operational_recourse_components`), every block priced for the evaluation with production's ADMM-subproblem
pricing and the curtailment tie-breaker at the EVALUATION value 0 (ruled); salvage reported separately.

STAGES, IN THE ADDENDUM 49 ORDER, EACH ITS OWN ATTACHED PROCESS WITH AN EXACT DECLARED SOLVE COUNT
-------------------------------------------------------------------------------------------------
 1. lambda-look               ZERO solves. Units check on the S48 x = 0 persisted models (sha 03b62593...): the TN
                              bus duals must reproduce W28's committed LMP7 (x0_capture_analysis.json, dd0a86b3) and
                              W31's committed per-hour lmp/y0/pi/lambda_E (dn_plateau.json, d68c814d) to 1e-9 EUR/MWh,
                              and the consensus-dual Params (undone: sigma via admm_objective_scale, interface rating,
                              B) must agree with the nodal duals to the KKT tolerance -- Addendum 32's "bus-7 marginal
                              cost = DSO flexibility shadow price" figure. Then the lambda_t vs pi_t table on the
                              SETTLED cycle-181 models, per node and hour, flagging |lambda_t - pi_t| > 0.05 EUR/MWh AND
                              c_flex < pi_t. THIRD SOURCE (W106): the lambda_t sidecar `interface_duals_per_cycle.jsonl`
                              of the settled cell (`lambda_sidecar_cross_check`): row k*-1 = 180 divided by s_base must
                              equal the persisted consensus-dual Params BITWISE (the sidecar holds the dual the NEXT
                              cycle's solve uses; AA not accepted at 180/181), and row k* = 181, converted to EUR/MWh by
                              the same interface-P scaling, must agree with the Param-derived lambda_dso_full within
                              the consensus tolerance. The table is marked usable as the prediction only if the units
                              check passes, the settled identities hold AND the sidecar cross-check passes.
 2. common-q-gate             ZERO solves. `evaluate_common_q` (evaluation tie-breaker 0, require_unchanged) on the
                              SETTLED persisted models must reproduce the settled recourse components BITWISE -- the
                              reference is Q181 = 653,873,702.1876609 (settling_decision.json Q_k_star, equal to the
                              evaluation record's certified_cost and recourse gross, checked before loading); negative
                              control and discrimination: evaluation tie-breaker 1 must be refused under
                              require_unchanged and, repriced, must differ by exactly the curtailment it prices.
 3. tso-coupling-check        24 solves (2 couplings x 12 TSO blocks: 12 fixed + 12 penalty). Ruling 2's penalty-vs-
                              fixed check at the SAME targets = the certified coordinated DSO schedule: TN cost,
                              interface residual (P, Q, V), IPOPT iterations, dual infeasibility (scaled/unscaled) at
                              termination. Reported, not gating. THE CHECK'S FIXED SIDE PINS THE INTERFACE VOLTAGE (W94)
                              to the SAME target the penalty tracks (targets['v_kv'] / TN base kV), because production's
                              _add_tso_scenario_tracking_penalty tracks P, Q AND V: with P/Q fixed and V free the two
                              sides would be different problems and a TN-cost difference could come from the voltage
                              freedom instead of the scaling hazard the check measures. With the pin both sides are
                              the penalty's limit problem (P = P_req, Q = Q_req, V = V_req), so any difference is
                              numerical. The pin is CHECK-ONLY: the TSO ARM (stage 4) fixes P/Q and leaves V within its
                              normal bounds (ruling 2). Fixed rows per fixed-side block: 3 x 3 ADNs x 24 h = 216 (the
                              arm's: 2 x 3 x 24 = 144). Declared: uncoordinated_benchmark.
                              COUPLING_CHECK_FIXED_SIDE_VOLTAGE_PIN, recorded in the output.
 4. arm --arm A --start S     48 solves (36 DSO + 12 TSO), then the consistency re-evaluation 36 solves -> 84; if the
                              declared trigger fires, the one sequential pass adds 48 -> 132. A in {passive,
                              price_taker}, S in {cold, warm_from_certified, perturbed}: six runs.
 5. passive-tie-breaker --value V   48 solves (passive, cold, DSO decision tie-breaker V in {0.1, 10}); no
                              consistency step (declared: the variant exists to measure the interface schedule's
                              value-independence against the ruled 1 EUR/MWh cold run).
 6. report                    ZERO solves. Reads 1-5; claim = min(passive, price-taker) - coordinated with the bands;
                              curtailment table (author requirement): every arm and start on the coordinated figure's
                              weighting (EUR at 1 EUR/MWh, block-weighted) with raw MWh (day-weighted and per
                              representative day) alongside, beside the settled coordinated figure (record and
                              recomputed from the models by stage 2) and the W86 cycle-132 figure 589.18 (reported).
                              Each arm's Q is the MINIMUM over its three starts: CONSERVATIVE AGAINST THE COORDINATION
                              CLAIM -- the uncoordinated side gets its best local optimum of three, the coordinated side
                              is one certified cell (Planner ruling, W94). Recorded as Q_MIN_OVER_STARTS_NOTE.

LAMBDA_t UNITS (stage 1; LAMBDA_CHANNEL_SCALING): Addendum 49's clarification says to undo "sigma, S_ref, D5". On the
interface-P channel only sigma (through admm_objective_scale = sigma / block weight) and the interface rating apply;
S_ref and D5 act ONLY on the shared-ESS channel. Accepted by the Planner as a correction to that wording (W94); the
production source lines are cited in LAMBDA_CHANNEL_SCALING and in uncoordinated_benchmark.interface_price_terms.

Every solve stage arms `SolveProfileGuard(permitted=(('uncoordinated_benchmark.py', '_solve_block'),))` BEFORE any
production import and checks the cumulative count EXACTLY at every phase boundary (too few fails as loudly as too
many; a production retry tier raises the count and fails it). Zero-solve stages arm `permitted=()`. Each run refuses
unless: no campaign / legacy lock, no forbidden live process (the p515_s4* campaign children, any other p515_s53_*
stage -- the 3x3 pair launcher included), production and these files clean in git, its output directory absent
(write-once), its own lock acquired, and (W106) the frozen benchmark spec binds: the highest-version
`frozen_s53_benchmark_spec_v<N>_<hash8>.json` under the output root has a content sha256 starting with its <hash8>,
and its `code_sha256_binding` equals the sha256 of every listed file on disk and its coordinated models sha equals
`COORDINATED` (`frozen_spec_binding_failures`). Per-solve records are appended (flushed, fsynced) to
per_solve_record.jsonl as they happen, phase checkpoints to phase_checkpoints.jsonl, so a failure is diagnosable.
Capture-path checklists are asserted before any solve.

EXACT COMMANDS (W106; repo root; attached, ALONE, one at a time, both streams captured; never detached; the frozen
benchmark spec under the output root lists the same commands with the declared solve counts):
    mkdir -p data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w93_uncoordinated_benchmark.py --stage lambda-look > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/lambda_look.log 2>&1
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w93_uncoordinated_benchmark.py --stage common-q-gate > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/common_q_gate.log 2>&1
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w93_uncoordinated_benchmark.py --stage tso-coupling-check > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/tso_coupling_check.log 2>&1
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w93_uncoordinated_benchmark.py --stage arm --arm passive --start cold > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/arm_passive_cold.log 2>&1
      (and --arm passive --start warm_from_certified | perturbed; --arm price_taker --start cold | warm_from_certified | perturbed; log name arm_<arm>_<start>.log)
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w93_uncoordinated_benchmark.py --stage passive-tie-breaker --value 0.1 > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/passive_tie_breaker_0p1.log 2>&1
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w93_uncoordinated_benchmark.py --stage passive-tie-breaker --value 10 > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/passive_tie_breaker_10.log 2>&1
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w93_uncoordinated_benchmark.py --stage report > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/report.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/w106_uncoordinated_settled/<run_id>/ with <run_id>.json,
per_solve_record.jsonl (solve stages), phase_checkpoints.jsonl, manifest_sha256.json. Exit 0 = completed with every
gate of the stage passed; 1 = a gate failed or a solve failed; 2 = refused before doing anything (precondition).
(W93/W94's commands named data/SRP1/Results/P515S53/w93_uncoordinated/; that root was never created and is unwired.)
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
import pickle
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s44_campaign_harness as H  # noqa: E402 -- standard library only at import (no model code)
import gate_result_io as GRIO  # noqa: E402 -- W100 (Addendum 52): the one gate-result writer; stdlib only

# ======================================================================================================================
#  FROZEN CONFIGURATION (declared before any run; recorded in every artifact)
# ======================================================================================================================
STAGE = 'P5.15 Addendum 49 W93 -- uncoordinated benchmark: three arms, fixed interface, common Q'
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 49 (rulings 1-3, consistency convention, resolution, timing)',
    'PLANNER_BRIEF_2026-09-13.md Addendum 49 clarification 2026-09-26 (tie-breaker roles; lambda_t recovery)',
    'TASKS.md Addendum 49 order', 'Planner task W93',
    'PLANNER_BRIEF_2026-09-13.md Addendum 53 Ruling 4 (coordinated arm reported only as the settled x = 0 value)',
    'PLANNER_BRIEF_2026-09-13.md Addendum 54 Ruling 3 (common-Q gate and lambda_t look on the settled cycle-181 x = 0 '
    'models; the gate reference is the settled value)', 'Planner task W106',
]
SCRIPT_NAME = os.path.basename(__file__)
# W106 (Addendum 54 Ruling 3): the settled benchmark's output root (new; the frozen benchmark spec lives here too).
OUT_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w106_uncoordinated_settled')
# W106: UNWIRED, RETAINED -- the W93/W94 output root (never created: nothing ran there). Unwire, never delete.
_W93_OUT_ROOT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w93_uncoordinated')
FROZEN_SPEC_GLOB = 'frozen_s53_benchmark_spec_v*_*.json'   # W106: the frozen benchmark spec, under OUT_ROOT_REL
LOCK_PATH = os.path.join(REPO, '.p515_s53_w93_benchmark.lock')
EXTRA_FORBIDDEN_PROCESS_SUBSTRINGS = ('p515_s53_', 'p515_s44_campaign_harness', 'p515_s4', 'p515_g_g1_g4_admm_gates')
EXTRA_CLEAN_FILES = ('uncoordinated_benchmark.py', 'p515_s53_w93_uncoordinated_benchmark.py',
                     'p515_s53_w93_uncoordinated_benchmark_checks.py', 'p58_rescale.py', 'p513_solve_profile_guard.py',
                     'p515_s53_srp1_bitwise_gate.py')
EVAL_ID_PREFIX = 'p515s53w106_'  # isolated IPOPT log dir: p56a_oracle.WORK_DIR/<EVAL_ID_PREFIX><run_id>/logs
# W106: UNWIRED, RETAINED -- the W93/W94 IPOPT log-dir prefix (no directory with it exists: nothing ran).
_W93_EVAL_ID_PREFIX = 'p515s53w93_'

X0 = {'label': 'x0', 'nodes': {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}, 'investment_year': 2025,
      'candidate_key': '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57'}
# ---------------------------------------------------------------------------------------------------------------------
# W106 (Addendum 54 Ruling 3): THE COORDINATED CELL = the settled cycle-181 x = 0 models. Every coordinated-side input
# (common-Q reference, lambda_t models + sidecar, the coupling check's pinned targets, the warm start, the coordinated
# curtailment figure) reads COORDINATED. The sha256 pins are the ones hash-recorded in the committed campaign manifest
# (evidence commit e4b5c992) and child manifest; `common_capture_checklist` re-verifies them against both manifests,
# and the campaign manifest's blob at e4b5c992 against the working file, before any model is loaded.
# ---------------------------------------------------------------------------------------------------------------------
_SETTLED_CAMPAIGN = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w101_srp1_continuation',
                                 'campaign_s53_w101_srp1_cont_x0')
_SETTLED_EVAL = os.path.join(_SETTLED_CAMPAIGN, 'evals', 'd110bd1a5977df1e_x0')
SETTLED = {
    'description': ('W102 settled SRP1 x = 0 cell (settling continuation of the W86 re-certification: bitwise replay '
                    'through cycle 132, then certified by the settling rule at k* = 181; terminal models persisted)'),
    'campaign_spec_sha256': 'd1d5c3bcbdc261d297282382ce4a54263a811059d17230613a1765ee915ebd33',
    'stage_spec': {'path': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'frozen_s53_spec_v39_8a612429.json'),
                   'version': 39},
    'eval_key': 'd110bd1a5977df1e811a1b3963afc565de6daa65f76dcc096ead3fdc70b08546',
    'evidence_commit': 'e4b5c9928d96b413c77039c377e5f7a82a6549f5',
    'eval_dir': _SETTLED_EVAL,
    'evaluation_record': os.path.join(_SETTLED_EVAL, 'evaluation_record.json'),
    'post_certification': os.path.join(_SETTLED_EVAL, 'post_certification.json'),
    'child_manifest': os.path.join(_SETTLED_EVAL, 'child_manifest_sha256.json'),
    'campaign_manifest': os.path.join(_SETTLED_CAMPAIGN, 'campaign_manifest_sha256.json'),
    'settling_decision': os.path.join(_SETTLED_EVAL, 'settling_decision.json'),
    'certified_models': {'path': os.path.join(_SETTLED_EVAL, 'certified_models.pkl'),
                         'sha256': '99ab1070b0e61cc7818975ce898069d2060d2668bae67c166b58913e9923c33a'},
    'esso_models': {'path': os.path.join(_SETTLED_EVAL, 'esso_models_s39_D.pkl'),
                    'sha256': '118487e1853bbad176f324b1173df8c4198cb6c7709c25d7dce9f94ce4c199fa'},
    'pf_entry_stride': {'path': os.path.join(_SETTLED_EVAL, 'pf_entry_stride_s39_D.jsonl'),
                        'sha256': 'f71147584175a42f7c503e905e8e519841424b5a89bf33724dfce753769f498a'},
    'interface_duals': {'path': os.path.join(_SETTLED_EVAL, 'interface_duals_per_cycle.jsonl'),
                        'sha256': '10c986c79eebe9e6d84bbd9c3978b802b2760c00399cb427125a0b797b41e8cb'},
    # the common-Q reference (Addendum 54 Ruling 3: "the reference Q for the gate is the settled value")
    'certified_gross': 653873702.1876609,
    'certified_gross_source': ('settling_decision.json Q_k_star; equal (bitwise, checked) to evaluation_record '
                               'certified_cost, recourse_components.gross_operational_cost and post_certification '
                               'certified_cost'),
    'certification_cycle': 181,
    'certification_rule': ('settling criterion of stage spec v39 (Boyd residuals + three turning points, swings not '
                           'growing, one-period range <= tau = 4,539.07 EUR); branch oscillatory; band width 4,209.17'),
    # the coordinated curtailment figure at cycle 181 (component_decomposition_totals_weighted of the record)
    'res_curtailment_definitional_at_weight_1': 589.425936658749,
}
COORDINATED = SETTLED
# W106: REPORTED ONLY (never an input) -- the W86 cycle-132 figures, stated beside the settled ones.
COORDINATED_PREVIOUS_REFERENCE = {
    'cell': 'W86 5cfe69a615ae3708_x0 (campaign s53_w86_tail_recert, spec ddd6cd44) -- UNWIRED',
    'certification_cycle': 132, 'certified_gross': 653858731.5686293,
    'res_curtailment_definitional_at_weight_1': 589.1778397145646,
    'source': ('committed W86 evaluation_record.json (certified_cost; component_decomposition_totals_weighted.'
               'res_curtailment_definitional_at_weight_1)'),
}
# W106: UNWIRED, RETAINED -- the W93/W94 coordinated cell (Addendum 54 Ruling 3 moved every input to COORDINATED).
# Kept so the W93/W94 record stays readable and the unwired check can name it. Unwire, never delete.
_W86_EVAL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'tight_tail_w86', 'campaign_s53_w86_tail_recert',
                         'evals', '5cfe69a615ae3708_x0')
W86 = {
    'description': 'W86 tight-tail SRP1 re-certification of x = 0 (the coordinated arm; terminal models persisted)',
    'campaign_spec_sha256': 'ddd6cd4422b56f946eeeb427dff44ef572490bd34c606fad96eaf00fd4a84ec1',
    'eval_key': '5cfe69a615ae370835a5896eaa6899f1d9973b7adf649437c2ddd94d1d2836ee',
    'eval_dir': _W86_EVAL,
    'evaluation_record': os.path.join(_W86_EVAL, 'evaluation_record.json'),
    'post_certification': os.path.join(_W86_EVAL, 'post_certification.json'),
    'child_manifest': os.path.join(_W86_EVAL, 'child_manifest_sha256.json'),
    'certified_models': {'path': os.path.join(_W86_EVAL, 'certified_models.pkl'),
                         'sha256': 'd816bb4b1ade4b44ad4a77872443fe2668986663c524fc4239815d4657974e21'},
    'esso_models': {'path': os.path.join(_W86_EVAL, 'esso_models_s39_D.pkl'),
                    'sha256': '15f384090089e123a295d8a3f5d0d6b2dc4e953effed55b28eae456a38529d8d'},
    'pf_entry_stride': {'path': os.path.join(_W86_EVAL, 'pf_entry_stride_s39_D.jsonl'),
                        'sha256': '38631ea3378f5122b0cdb9aa0ea43ce21c88fd152dab55ee94893938d7d7bc51'},
    'certified_gross': 653858731.5686293,
    'certification_cycle': 132,
}
_S48_EVAL = os.path.join('data', 'SRP1', 'Results', 'P515S48', 'x0_capture', 'evals', 'd2c96b1480402a3b_x0')
S48 = {
    'description': 'S48 x = 0 capture (pre-tail), the models Addendum 32 Q4 analysed (units check only)',
    'eval_key': 'd2c96b1480402a3b61aca4abc188e41c6009eb582d8e6ccdd380e51651f996c7',
    'certified_models': {'path': os.path.join(_S48_EVAL, 'certified_models.pkl'),
                         'sha256': '03b62593a23f748c819f18dce52c88c6a9802b8af3c52033ae09a3b88d10afce'},
    'x0_capture_analysis': os.path.join('data', 'SRP1', 'Results', 'P515S48', 'x0_capture_analysis',
                                        'x0_capture_analysis.json'),
    'dn_plateau': os.path.join('data', 'SRP1', 'Results', 'P515S49', 'dn_plateau', 'dn_plateau.json'),
}
BITWISE_GATE_SCRIPT = 'p515_s53_srp1_bitwise_gate.py'   # holds PIN_BENCHMARK_SOURCE_SHA256 (read as text, not imported)

# The tie-breaker (Addendum 49 clarification 2026-09-26): evaluation 0 in every arm; decision 0 for the price-taker
# and the TSO (production), 1 EUR/MWh for the passive DSO only; value-independence re-solves at 0.1 and 10.
TIE_BREAKER = {'evaluation': 0.0,
               'decision': {'passive_dso': 1.0, 'price_taker_dso': 0.0, 'tso': 0.0},
               'passive_value_independence_variants': (0.1, 10.0),
               'authority': 'Addendum 49 clarification 2026-09-26'}
# The arms' network IPOPT complementarity tolerance: the tight-tail value the coordinated cell's terminal cycles
# were certified under (Addendum 46 r7 / 48), applied through production's own tail helpers. None = production
# default (compl_inf_tol not passed). DECLARED CHOICE FOR THE PLANNER TO CONFIRM (W93 report).
ARM_NETWORK_COMPL_INF_TOL = 1e-6
# The perturbed start (uncoordinated_benchmark.apply_perturbation): warm start from the certified coordinated solution,
# then every free Var v <- clip(v * (1 + delta * u), lb, ub), u ~ U[-1, 1) from
# numpy Generator(PCG64(SeedSequence([seed, crc32(block label)]))), one draw per free Var in sorted order.
PERTURBATION = {'seed': 20260926, 'delta': 0.05}
LAMBDA_NEQ_PI_TOL_EUR = 0.05                      # |lambda_t - pi_t| above which lambda_t != pi_t (W31 CONSENSUS_TOL)
UNITS_CHECK_TOL = {'repro_eur': 1e-9,             # reproduction of committed W28/W31 figures (W31 W25_REPRO_TOL)
                   'kkt_eur': 1e-3,               # Param-derived lambda vs the nodal dual (W31 KKT_TOL_EUR)
                   'consensus_eur': 0.05}         # DSO-side vs TSO-side lambda (W31 CONSENSUS_TOL_EUR)
CONSISTENCY_TOL = {'hard_tol_pu2': 1e-6, 'soft_excess_tol_pu2': 1e-6, 'thermal_tol_pu2': 1e-6}
NEGATIVE_CONTROL_EXPLAINED_ABS_TOL_EUR = 1e-5     # (Q at tie-breaker 1) - (Q at 0) - priced curtailment, ~100 ulp
COORDINATED_REPRODUCIBILITY_BAND_REL = 1.1e-4     # Addendum 49 "Resolution": the coordinated cell's band (0.011 %)
SRP1_DECLARED = {'dso_solves_per_arm': 36, 'tso_solves_per_arm': 12, 'arm_solves': 48, 'reevaluation_solves': 36,
                 'sequential_pass_solves': 48, 'coupling_check_solves': 24,
                 'coupling_check_fixed_side_solves': 12, 'coupling_check_penalty_side_solves': 12,
                 # fixed rows per TSO block = row families x 3 ADNs x 24 h (W94: the check's fixed side adds V)
                 'tso_arm_fixed_rows_per_block': 2 * 3 * 24,
                 'coupling_check_fixed_side_fixed_rows_per_block': 3 * 3 * 24,
                 'coupling_check_penalty_side_fixed_rows_per_block': 0}
# W94 (Planner ruling on W93 item 6): recorded in the docstring and every output.
Q_MIN_OVER_STARTS_NOTE = (
    "each uncoordinated arm's Q is the MINIMUM over its three starts (cold, warm_from_certified, perturbed); this is "
    'CONSERVATIVE AGAINST THE COORDINATION CLAIM: the uncoordinated side gets its best local optimum of three, while '
    'the coordinated side is one certified cell (W106: the settled cycle-181 cell d110bd1a5977df1e_x0, not re-run; '
    'W94 named the W86 cell 5cfe69a615ae3708_x0, now unwired)')
# W94: the lambda_t conversion on the interface-P channel (Planner-accepted correction to the Addendum 49 wording).
LAMBDA_CHANNEL_SCALING = {
    'addendum_49_wording': 'undo sigma, S_ref, D5 when converting lambda_t to EUR/MWh',
    'correction': ('on the interface-P channel only sigma (through admm_objective_scale = sigma / block weight) and the '
                   'interface rating apply; S_ref (shared_ess_reference_rating_mva) and D5 (the ESSO AL scale, '
                   'al_scale_esso) act ONLY on the shared-ESS channel and are therefore not undone'),
    'accepted_by': 'Planner, task W94 (correction to the Addendum 49 clarification wording)',
    'source': {
        'file': 'shared_resources_planning.py', 'at_commit': 'a8c58da0',
        'sigma_tso': 'update_transmission_model_to_admm L5137-5142: effective_scale = objective_scale / block_weight; '
                     'obj = copy(objective.expr) / effective_scale',
        'rating_tso': 'update_transmission_model_to_admm L5156-5157: constraint_p_req/q_req = (E - z) / '
                      'interface_transf_rating',
        'sref_tso': 'update_transmission_model_to_admm L5179: _admm_shared_ess_reference_mva(params) only in the '
                    'shared-ESS loop (shared_ess_rating)',
        'sigma_dso': 'update_distribution_models_to_admm L5402-5407 (same as TSO)',
        'rating_dso': 'update_distribution_models_to_admm L5428-5429 (same as TSO)',
        'sref_dso': 'update_distribution_models_to_admm L5410-5415: only in shared_ess_rating (ESS channel)',
        'd5': ('update_shared_energy_storage_model_to_admm L5486 (admm_esso_al_scale = al_scale_esso) and L5515-5518 '
               '(multiplies ONLY the ESSO AL terms); al_scale_esso is never passed to the TSO/DSO updates'),
        'implemented_in': 'uncoordinated_benchmark.interface_price_terms (docstring carries the same citation)',
    },
}
ARM_RUN_IDS = [f'arm_{a}_{s}' for a in ('passive', 'price_taker')
               for s in ('cold', 'warm_from_certified', 'perturbed')]
VARIANT_RUN_IDS = ['passive_tie_breaker_0p1', 'passive_tie_breaker_10']
PERMITTED_SOLVE_SITE = (('uncoordinated_benchmark.py', '_solve_block'),)
# W106: the files whose sha256 the frozen benchmark spec binds (a stage refuses when any differs from the spec).
FROZEN_SPEC_BOUND_FILES = (
    'p515_s53_w93_uncoordinated_benchmark.py', 'p515_s53_w93_uncoordinated_benchmark_checks.py',
    'uncoordinated_benchmark.py', 'shared_resources_planning.py', 'model_construction_helpers.py', 'network.py',
    'network_data.py', 'shared_energy_storage_data.py', 'helper_functions.py', 'definitions.py', 'p56a_oracle.py',
    'p58_rescale.py', 'p513_solve_profile_guard.py', 'gate_result_io.py', 'interface_dual_capture.py')
# W106: every quantity the report needs, the stage output that carries it and the key path inside it (rule eleven:
# asserted before any stage runs by the W106 zero-solve checks, and again by the report stage when it reads them).
REPORT_CAPTURE_PATHS = {
    'lambda_t_units_check_s48': ('lambda_look', ('units_check_s48', 'passed')),
    'lambda_t_identities_coordinated': ('lambda_look', ('identities_coordinated', 'passed')),
    'lambda_t_sidecar_cross_check': ('lambda_look', ('lambda_sidecar_cross_check', 'passed')),
    'lambda_t_prediction_usable': ('lambda_look', ('prediction_usable',)),
    'lambda_t_vs_pi_t_summary': ('lambda_look', ('summary_coordinated',)),
    'common_q_gate_status': ('common_q_gate', ('status',)),
    'common_q_reference_q181': ('common_q_gate', ('gate', 'gross_certified')),
    'coordinated_curtailment_recomputed': ('common_q_gate', ('evaluation_at_0', 'curtailment', 'totals')),
    'coordinated_curtailment_record': ('common_q_gate', ('negative_controls',
                                                         'certified_record_res_curtailment_definitional_at_weight_1')),
    'coupling_check_tn_cost_totals': ('tso_coupling_check', ('tn_cost_weighted_totals',)),
    'coupling_check_solve_summary': ('tso_coupling_check', ('solve_summary',)),
    'arm_cost': ('arm_*', ('arm_cost', 'gross_operational_cost')),
    'arm_curtailment_totals': ('arm_*', ('phase_A', 'evaluation', 'curtailment', 'totals')),
    'arm_consistency_max_abs_dv': ('arm_*', ('phase_B_consistency', 'max_abs_dv_dn_pu')),
    'arm_consistency_trigger': ('arm_*', ('phase_B_consistency', 'reevaluation', 'trigger_sequential_pass')),
    'variant_dso_interface_schedule': ('passive_tie_breaker_*', ('phase_A', 'dso_interface_schedule')),
    'variant_cost': ('passive_tie_breaker_*', ('arm_cost', 'gross_operational_cost')),
    'variant_curtailment_totals': ('passive_tie_breaker_*', ('phase_A', 'evaluation', 'curtailment', 'totals')),
}

_GUARD = None
_LOG_T0 = time.time()


# ======================================================================================================================
#  generic helpers
# ======================================================================================================================
def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{time.strftime("%H:%M:%S")} [W93 +{time.time() - _LOG_T0:8.1f}s] {msg}', flush=True)


def _abs(rel):
    return os.path.join(REPO, rel)


# W100 (Addendum 52): UNWIRED, RETAINED. `_write_json_once` / `_append_jsonl` now go through `gate_result_io` with
# GRIO.json_default_item (this function's behaviour verbatim) plus the string-flag refusal. Unwire, never delete.
def _json_default(o):
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if hasattr(o, 'item'):          # numpy scalars
        try:
            return o.item()
        except Exception:  # noqa: BLE001
            pass
    return str(o)


def _write_json_once(path, payload):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    with open(path, 'x') as handle:
        GRIO.dump(payload, handle, indent=1, sort_keys=True, default=GRIO.json_default_item)
    return path


def _append_jsonl(path, record):
    with open(path, 'a') as handle:
        handle.write(GRIO.dumps(record, sort_keys=True, default=GRIO.json_default_item) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def _write_manifest(run_dir):
    manifest = {}
    for root, _dirs, files in os.walk(run_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            if name == 'manifest_sha256.json':
                continue
            manifest[os.path.relpath(path, REPO)] = H.sha256_file(path)
    return _write_json_once(os.path.join(run_dir, 'manifest_sha256.json'), manifest)


def _git_tracked_clean(rel):
    try:
        subprocess.run(['git', 'ls-files', '--error-unmatch', rel], cwd=REPO, capture_output=True, check=True)
    except subprocess.CalledProcessError:
        return False
    return H._git(['status', '--porcelain', '--', rel]).strip() == ''


def _load_json(rel):
    with open(_abs(rel)) as handle:
        return json.load(handle)


def _verified_path(entry):
    path = _abs(entry['path'])
    got = H.sha256_file(path)
    if got != entry['sha256']:
        raise RuntimeError(f"{entry['path']}: sha256 {got} != declared {entry['sha256']}")
    return path


def _load_pickle_verified(entry):
    path = _verified_path(entry)
    _log(f"unpickling {entry['path']} (sha256 verified {entry['sha256'][:8]})")
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def _read_last_line(path, block=1 << 16):
    with open(path, 'rb') as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        data = b''
        position = end
        while position > 0:
            step = min(block, position)
            position -= step
            handle.seek(position)
            data = handle.read(step) + data
            stripped = data.rstrip(b'\n')
            if b'\n' in stripped:
                return stripped.rsplit(b'\n', 1)[1].decode()
        return data.rstrip(b'\n').decode()


def _install_guard(stage, bounded):
    global _GUARD
    from p513_solve_profile_guard import SolveProfileGuard
    permitted = PERMITTED_SOLVE_SITE if bounded else ()
    _GUARD = SolveProfileGuard(permitted, label=f'P5.15 W93 {stage}').install()
    return _GUARD


def _check_guard(expected, where):
    failures = _GUARD.verify(expected)
    if failures:
        raise RuntimeError(f'SolveProfileGuard at {where}: expected exactly {expected}: {failures}; '
                           f'counts {_GUARD.counts}')
    return {'where': where, 'expected': expected, 'counts': dict(_GUARD.counts), 'verified': True}


def _production():
    """Production and committed-diagnostic imports, AFTER the guard is armed."""
    import shared_resources_planning as srp
    import uncoordinated_benchmark as UB
    import p56a_oracle as O
    import p58_rescale as R
    return srp, UB, O, R


# ======================================================================================================================
#  preconditions, lock, provenance
# ======================================================================================================================
def check_preconditions(run_dir):
    failures = H.check_campaign_preconditions(run_dir, extra_clean_files=EXTRA_CLEAN_FILES)
    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan the process table: {error}')
        ps_output = ''
    excluded = {str(pid) for pid in H._ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[1] in excluded:
            continue
        if any(s in line for s in EXTRA_FORBIDDEN_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')
    if os.path.exists(LOCK_PATH):
        failures.append(f'W93 lock exists: {LOCK_PATH}')
    failures.extend(frozen_spec_binding_failures())
    return failures


def _latest_frozen_spec():
    """(path, version, hash8) of the highest-version frozen benchmark spec under the output root, or None."""
    import glob
    best = None
    for path in glob.glob(os.path.join(_abs(OUT_ROOT_REL), FROZEN_SPEC_GLOB)):
        match = re.fullmatch(r'frozen_s53_benchmark_spec_v(\d+)_([0-9a-f]{8})\.json', os.path.basename(path))
        if match and (best is None or int(match.group(1)) > best[1]):
            best = (path, int(match.group(1)), match.group(2))
    return best


def _frozen_spec_identity():
    latest = _latest_frozen_spec()
    if latest is None:
        return None
    return {'path': os.path.relpath(latest[0], REPO), 'version': latest[1], 'sha256': H.sha256_file(latest[0])}


def frozen_spec_binding_failures():
    """W106: a stage runs only under its frozen spec -- the highest-version spec exists, its content sha256 starts
    with the <hash8> in its name, every bound file's sha256 on disk equals the spec's pin, and the spec's coordinated
    models / reference Q equal COORDINATED."""
    latest = _latest_frozen_spec()
    if latest is None:
        return [f'no frozen benchmark spec {FROZEN_SPEC_GLOB} under {OUT_ROOT_REL}']
    path, _version, hash8 = latest
    failures = []
    sha = H.sha256_file(path)
    if not sha.startswith(hash8):
        failures.append(f'frozen spec {path}: content sha256 {sha} does not start with its name hash {hash8}')
    with open(path) as handle:
        spec = json.load(handle)
    pins = spec.get('code_sha256_binding') or {}
    if sorted(pins) != sorted(FROZEN_SPEC_BOUND_FILES):
        failures.append(f'frozen spec binds {sorted(pins)} != FROZEN_SPEC_BOUND_FILES {sorted(FROZEN_SPEC_BOUND_FILES)}')
    for name, pinned in pins.items():
        got = H.sha256_file(_abs(name))
        if got != pinned:
            failures.append(f'{name}: sha256 {got} != frozen spec pin {pinned}')
    coordinated = spec.get('coordinated') or {}
    if (coordinated.get('certified_models') or {}).get('sha256') != COORDINATED['certified_models']['sha256']:
        failures.append('frozen spec coordinated certified_models sha256 != COORDINATED')
    reference = coordinated.get('reference_q') or {}
    if reference.get('hex') != float(COORDINATED['certified_gross']).hex():
        failures.append('frozen spec coordinated reference Q != COORDINATED certified_gross')
    return failures


def acquire_lock(run_id):
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as handle:
        json.dump({'pid': os.getpid(), 'run_id': run_id, 'started_utc': _utc()}, handle)


def release_lock():
    if os.path.exists(LOCK_PATH):
        os.remove(LOCK_PATH)


def x0_candidate_key():
    canonical = H.canonical_candidate({n: v for n, v in X0['nodes'].items()},
                                      investment_year=X0['investment_year'])
    return canonical, H.candidate_key(canonical)


def provenance(extra=None):
    canonical, key = x0_candidate_key()
    record = {
        'stage': STAGE, 'authority': AUTHORITY, 'utc': _utc(),
        'git_head': H._git(['rev-parse', 'HEAD']),
        'script': SCRIPT_NAME, 'script_sha256': H.sha256_file(os.path.abspath(__file__)),
        'module_sha256': {name: H.sha256_file(_abs(name)) for name in (
            'uncoordinated_benchmark.py', 'shared_resources_planning.py', 'model_construction_helpers.py',
            'network.py', 'network_data.py', 'p56a_oracle.py', 'p58_rescale.py', 'p513_solve_profile_guard.py')},
        'interpreter': sys.executable,
        'nlp_solver_path': H._resolve_solver_path_from_dotenv(),
        'instance': {'problem': 'SRP1', 'label': X0['label'], 'canonical_candidate': canonical,
                     'candidate_key': key, 'candidate_key_declared': X0['candidate_key'],
                     'candidate_key_matches': key == X0['candidate_key']},
        'coordinated_cell': {k: v for k, v in COORDINATED.items() if k not in ('description',)},
        'coordinated_cell_description': COORDINATED['description'],
        'coordinated_previous_reference_reported_only': COORDINATED_PREVIOUS_REFERENCE,
        'frozen_benchmark_spec': _frozen_spec_identity(),
        'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED (production '
                                 '_get_operational_recourse_components); every block priced for the evaluation with '
                                 "production's ADMM-subproblem pricing and the curtailment tie-breaker at the "
                                 'EVALUATION value; salvage reported separately'),
        'tie_breaker': TIE_BREAKER,
        'arm_network_compl_inf_tol': ARM_NETWORK_COMPL_INF_TOL,
        'perturbation': PERTURBATION,
        'q_min_over_starts_note': Q_MIN_OVER_STARTS_NOTE,
        'lambda_channel_scaling': LAMBDA_CHANNEL_SCALING,
        'tolerances': {'lambda_neq_pi_eur': LAMBDA_NEQ_PI_TOL_EUR, 'units_check': UNITS_CHECK_TOL,
                       'consistency': CONSISTENCY_TOL,
                       'negative_control_explained_abs_eur': NEGATIVE_CONTROL_EXPLAINED_ABS_TOL_EUR},
    }
    if extra:
        record.update(extra)
    return record


# ======================================================================================================================
#  capture-path checklist (asserted before any solve / any evaluation)
# ======================================================================================================================
_ADMM_ASSIGN_RE = re.compile(r'(?:model|dso_model)\[year\]\[day\]\.(\w+)\s*=\s*pe\.(?:Param|Objective)\(')


def common_capture_checklist(srp, UB, R, planning=None):
    checks = {}
    canonical, key = x0_candidate_key()
    checks['instance_candidate_key_is_x0'] = key == X0['candidate_key']
    # W106: every entry below reads COORDINATED (the settled cycle-181 cell); W93/W94 read W86 here (unwired).
    cell = COORDINATED
    for name in ('evaluation_record', 'post_certification', 'child_manifest', 'campaign_manifest',
                 'settling_decision', 'interface_duals'):
        rel = cell[name]['path'] if isinstance(cell[name], dict) else cell[name]
        checks[f'coordinated_{name}_git_tracked_and_clean'] = _git_tracked_clean(rel)
    record = _load_json(cell['evaluation_record'])
    manifest = _load_json(cell['child_manifest'])
    campaign_manifest = _load_json(cell['campaign_manifest'])
    post = _load_json(cell['post_certification'])
    decision = _load_json(cell['settling_decision'])
    q_ref = float(cell['certified_gross'])
    checks['coordinated_record_candidate_key_is_x0'] = record.get('candidate_key') == X0['candidate_key']
    checks['coordinated_record_eval_key'] = record.get('eval_key') == cell['eval_key']
    checks['coordinated_record_campaign_spec'] = record.get('campaign_spec_sha256') == cell['campaign_spec_sha256']
    checks['coordinated_record_certified'] = record.get('status') == 'certified'
    checks['coordinated_record_certification_cycle'] = record.get('certification_cycle') == cell['certification_cycle']
    checks['coordinated_record_certified_gross_is_declared'] = (
        float(record.get('certified_cost')).hex() == q_ref.hex())
    checks['coordinated_record_recourse_components_present'] = isinstance(record.get('recourse_components'), dict)
    checks['coordinated_record_recourse_gross_is_declared'] = (
        float((record.get('recourse_components') or {}).get('gross_operational_cost')).hex() == q_ref.hex())
    # the reference Q's source (Addendum 54 Ruling 3): the settling decision's Q_k*, at k* = the certification cycle
    checks['coordinated_settling_decision_certified'] = decision.get('status') == 'certified'
    checks['coordinated_settling_decision_k_star_is_certification_cycle'] = (
        decision.get('k_star') == cell['certification_cycle'])
    checks['coordinated_settling_decision_q_k_star_is_reference_q_bitwise'] = (
        float(decision.get('Q_k_star')).hex() == q_ref.hex())
    checks['coordinated_post_certification_cycle'] = (
        (post.get('certification') or {}).get('certification_cycle') == cell['certification_cycle'])
    checks['coordinated_post_certification_cost_is_reference_q_bitwise'] = (
        float(post.get('certified_cost')).hex() == q_ref.hex())
    checks['coordinated_record_curtailment_figure_is_declared'] = (
        (record.get('component_decomposition_totals_weighted') or {}).get(
            'res_curtailment_definitional_at_weight_1') == cell['res_curtailment_definitional_at_weight_1'])
    checks['coordinated_persist_requested'] = (post.get('requested') or {}).get('persist_certified_models') is True
    checks['coordinated_persisted_sha_in_post_certification'] = (
        (post.get('persisted_models') or {}).get('sha256') == cell['certified_models']['sha256'])
    for name in ('certified_models', 'esso_models', 'pf_entry_stride', 'interface_duals'):
        checks[f'coordinated_{name}_sha_in_child_manifest'] = manifest.get(cell[name]['path']) == cell[name]['sha256']
        checks[f'coordinated_{name}_sha_in_campaign_manifest'] = (campaign_manifest.get(cell[name]['path'])
                                                                  == cell[name]['sha256'])
    for name in ('evaluation_record', 'settling_decision', 'post_certification'):
        disk = H.sha256_file(_abs(cell[name]))
        checks[f'coordinated_{name}_sha_in_child_manifest'] = manifest.get(cell[name]) == disk
        checks[f'coordinated_{name}_sha_in_campaign_manifest'] = campaign_manifest.get(cell[name]) == disk
    # the campaign manifest the pins come from is the one committed in the evidence commit (blob == working file)
    try:
        blob = subprocess.run(['git', 'show', f"{cell['evidence_commit']}:{cell['campaign_manifest']}"], cwd=REPO,
                              capture_output=True, check=True).stdout
        checks['coordinated_campaign_manifest_equals_evidence_commit_blob'] = (
            hashlib.sha256(blob).hexdigest() == H.sha256_file(_abs(cell['campaign_manifest'])))
    except subprocess.CalledProcessError:
        checks['coordinated_campaign_manifest_equals_evidence_commit_blob'] = False
    # the retired path is untouched (its pin, read as text from the committed bitwise gate -- not imported: that
    # module arms its own guard at import)
    gate_text = open(_abs(BITWISE_GATE_SCRIPT)).read()
    pin = re.search(r"PIN_BENCHMARK_SOURCE_SHA256 = '([0-9a-f]{64})'", gate_text)
    old_src = inspect.getsource(srp._run_operational_planning_without_coordination).rstrip('\n')
    checks['retired_path_source_sha_equals_bitwise_gate_pin'] = (
        pin is not None and hashlib.sha256(old_src.encode()).hexdigest() == pin.group(1))
    checks['new_module_does_not_call_retired_path'] = (
        '_run_operational_planning_without_coordination(' not in inspect.getsource(UB))
    # every component the coordination path adds is a declared consensus term
    added = set()
    for fn in (srp.update_transmission_model_to_admm, srp.update_distribution_models_to_admm):
        added |= set(_ADMM_ASSIGN_RE.findall(inspect.getsource(fn)))
    checks['coordination_components_found_in_source'] = len(added) >= 20
    checks['coordination_components_all_declared_consensus_terms'] = added <= set(UB.CONSENSUS_TERM_COMPONENTS)
    checks['p58_rescaled_objective_declared'] = R.RESCALED_OBJECTIVE in UB.CONSENSUS_TERM_COMPONENTS
    # the interface-P AL term is normalised by the interface rating and the objective by effective_scale (the scalings
    # interface_price_terms undoes); the ESS channel's S_ref / D5 are not on this channel
    dso_src = inspect.getsource(srp.update_distribution_models_to_admm)
    tso_src = inspect.getsource(srp.update_transmission_model_to_admm)
    checks['pf_channel_dso_rating_normalised'] = (
        'constraint_p_req = (dso_model[year][day].expected_interface_pf_p[p] - dso_model[year][day].p_pf_req[p]) '
        '/ interface_transf_rating' in dso_src)
    checks['pf_channel_tso_rating_normalised'] = (
        'constraint_p_req = (model[year][day].expected_interface_pf_p[dn, p] - model[year][day].p_pf_req[dn, p]) '
        '/ interface_transf_rating' in tso_src)
    checks['objective_divided_by_effective_scale_dso'] = (
        'obj = copy(dso_model[year][day].objective.expr) / effective_scale' in dso_src)
    checks['objective_divided_by_effective_scale_tso'] = (
        'obj = copy(model[year][day].objective.expr) / effective_scale' in tso_src)
    checks['dual_param_is_dual_var_over_s_base_dso'] = (
        "dual_pf_p_req[p].set_value(dual_pf['current'][node_id][year][day]['p'][p] / s_base)"
        in inspect.getsource(srp.update_distribution_coordination_models_and_solve_sequential))
    checks['dual_param_is_dual_var_over_s_base_tso'] = (
        "dual_pf_p_req[dn, p].set_value(dual_pf['current'][node_id][year][day]['p'][p] / s_base)"
        in inspect.getsource(srp.update_transmission_coordination_model_and_solve))
    # production functions the new module relies on
    for name in ('_prepare_transmission_objectives_for_admm', '_prepare_distribution_objectives_for_admm',
                 '_get_operational_recourse_components', '_get_operational_recourse_block_components',
                 '_add_tso_scenario_tracking_penalty', '_drain_network_ipopt_solve_records',
                 '_capture_convergence_depth_tail_baseline', '_apply_convergence_depth_tail',
                 '_set_row18_inactive_for_initialisation', '_get_admm_block_weight'):
        checks[f'srp_has_{name}'] = callable(getattr(srp, name, None))
    checks['recourse_components_reads_gross'] = "'gross_operational_cost': gross_operational_cost" in \
        inspect.getsource(srp._get_operational_recourse_components)
    if planning is not None:
        holders = [('TSO', planning.transmission_network)] + [
            (f'DSO{n}', planning.distribution_networks[n]) for n in sorted(planning.distribution_networks)]
        for label, holder in holders:
            options = holder.params.solver_params.options or {}
            checks[f'{label}_ipopt_output_file_configured'] = bool(options.get('output_file'))
            checks[f'{label}_ipopt_file_print_level_ge_5'] = int(options.get('file_print_level', 0)) >= 5
            checks[f'{label}_single_scenario'] = all(
                len(holder.network[y][d].prob_market_scenarios) == 1
                and len(holder.network[y][d].prob_operation_scenarios) == 1
                for y in holder.years for d in holder.days)
    return checks


def _assert_checklist(checks, where):
    false = sorted(k for k, v in checks.items() if v is not True)
    if false:
        raise RuntimeError(f'capture-path checklist FAILED at {where}: {false}')
    _log(f'capture-path checklist at {where}: {len(checks)} items, all True')


def _model_capture_checks(models, prefix, need_duals=True):
    """Every quantity the lambda look / structure / warm start reads exists on the persisted models."""
    import pyomo.environ as pe
    checks = {}
    tso_blocks = [b for yd in models['tso'].values() for b in yd.values()]
    dso_blocks = [b for n in models['dso'].values() for yd in n.values() for b in yd.values()]
    checks[f'{prefix}_n_tso_blocks_12'] = len(tso_blocks) == 12
    checks[f'{prefix}_n_dso_blocks_36'] = len(dso_blocks) == 36
    for kind, blocks in (('tso', tso_blocks), ('dso', dso_blocks)):
        for name in ('dual_pf_p_req', 'p_pf_req', 'rho_pf', 'admm_objective_scale', 'expected_interface_pf_p',
                     'expected_interface_vmag', 'node_balance_p'):
            checks[f'{prefix}_{kind}_all_have_{name}'] = all(hasattr(b, name) for b in blocks)
        active = {tuple(sorted(o.name for o in b.component_data_objects(pe.Objective, active=True))) for b in blocks}
        checks[f'{prefix}_{kind}_one_active_objective_p58_or_admm'] = active <= {('p58_rescaled_admm_objective',),
                                                                             ('admm_objective',)}
        if need_duals:
            checks[f'{prefix}_{kind}_dual_suffix_nonempty'] = all(hasattr(b, 'dual') and len(b.dual) > 0
                                                                  for b in blocks)
    return checks


# ======================================================================================================================
#  shared stage plumbing
# ======================================================================================================================
def _new_planning(O, run_id):
    eval_id = EVAL_ID_PREFIX + run_id
    if os.path.exists(os.path.join(O.WORK_DIR, eval_id)):
        raise RuntimeError(f'refusing: IPOPT log dir already exists: {os.path.join(O.WORK_DIR, eval_id)}')
    planning = O.fresh_planning(eval_id)
    return planning, os.path.join(O.WORK_DIR, eval_id, 'logs')


def _x0_candidate(srp, planning):
    candidate = planning.get_initial_candidate_solution()
    srp._rebuild_candidate_total_capacities(planning, candidate)
    nodes = {int(n): (float(candidate['investment'][n][_x0_year_key(planning)]['s']),
                      float(candidate['investment'][n][_x0_year_key(planning)]['e']))
             for n in candidate['investment']}
    canonical = H.canonical_candidate(nodes, investment_year=X0['investment_year'])
    if H.candidate_key(canonical) != X0['candidate_key']:
        raise RuntimeError(f'the constructed candidate is not x = 0: {canonical}')
    for node, years in candidate['total_capacity'].items():
        for year, cap in years.items():
            if cap['s'] != 0.0 or cap['e'] != 0.0:
                raise RuntimeError(f'x = 0 candidate has nonzero total capacity at {node} {year}: {cap}')
    return candidate


def _x0_year_key(planning):
    """The planning's own key for the investment year (the planning dicts are keyed as read from the case file)."""
    for year in planning.years:
        if int(year) == X0['investment_year']:
            return year
    raise RuntimeError(f"investment year {X0['investment_year']} is not a planning year {list(planning.years)}")


def apply_arm_solver_options(srp, planning):
    """The declared network compl_inf_tol, applied with production's own tail helpers; read back."""
    holders = srp._convergence_depth_tail_holders(planning)
    before = {label: dict(nd.params.solver_params.options or {}) for label, nd in holders}
    if ARM_NETWORK_COMPL_INF_TOL is None:
        return {'applied': False, 'options_in_force': before}
    admm = planning.params.admm
    admm.convergence_depth_tail = {'enabled': True, 'compl_inf_tol': float(ARM_NETWORK_COMPL_INF_TOL)}
    baseline = srp._capture_convergence_depth_tail_baseline(planning, admm)
    tail_record = srp._apply_convergence_depth_tail(planning, admm, True, baseline, 0)
    after = {label: dict(nd.params.solver_params.options or {}) for label, nd in holders}
    readback_ok = all(opts.get('compl_inf_tol') == float(ARM_NETWORK_COMPL_INF_TOL) for opts in after.values())
    if not readback_ok:
        raise RuntimeError(f'compl_inf_tol read-back failed: {after}')
    return {'applied': True, 'compl_inf_tol': float(ARM_NETWORK_COMPL_INF_TOL), 'baseline': baseline,
            'tail_record': tail_record, 'options_before': before, 'options_in_force': after,
            'recovery_options': {label: dict(nd.params.solver_params.recovery_options or {})
                                 for label, nd in holders}}


class SolveSink:
    """Appends every per-solve record to per_solve_record.jsonl immediately (flush + fsync) and keeps a compact
    in-memory summary."""

    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, 'per_solve_record.jsonl')
        if os.path.exists(self.path):
            raise RuntimeError(f'refusing to append to an existing {self.path}')
        self.records = []

    def __call__(self, record):
        record = dict(record)
        record['sequence'] = len(self.records) + 1
        _append_jsonl(self.path, record)
        last = record['attempts'][-1] if record['attempts'] else {}
        summary = last.get('final_summary') or {}
        self.records.append({
            'sequence': record['sequence'], 'block': record['block'], 'phase': record['phase'],
            'succeeded': record['succeeded'], 'termination_condition': record['termination_condition'],
            'n_attempts': record['n_attempts'], 'wall_s': record['wall_s'],
            'iterations': last.get('iterations'), 'exit': last.get('exit'),
            'mu_final': last.get('mu_final'), 'floor_status': last.get('floor_status'),
            'compl_inf_tol_in_force': last.get('compl_inf_tol_in_force'), 'tol_in_force': last.get('tol_in_force'),
            'dual_infeasibility_scaled': summary.get('scaled', {}).get('dual_infeasibility'),
            'dual_infeasibility_unscaled': summary.get('unscaled', {}).get('dual_infeasibility'),
            'overall_nlp_error_scaled': summary.get('scaled', {}).get('overall_nlp_error'),
            'constraint_violation_unscaled': summary.get('unscaled', {}).get('constraint_violation'),
            'active_objective_value': record['active_objective_value'],
        })
        status = 'ok' if record['succeeded'] else 'FAILED'
        _log(f"solve {record['sequence']:4d} {record['phase']:>36s} {record['block']:<24s} {status} "
             f"iter {last.get('iterations')} attempts {record['n_attempts']} {record['wall_s']:.1f}s")

    def summary(self, phase_prefix=None):
        rows = [r for r in self.records if phase_prefix is None or r['phase'].startswith(phase_prefix)]
        tol_ratio = [r['overall_nlp_error_scaled'] / r['tol_in_force'] for r in rows
                     if r['overall_nlp_error_scaled'] is not None and r['tol_in_force']]
        return {'n_solves': len(rows), 'all_succeeded': all(r['succeeded'] for r in rows),
                'n_retried': sum(1 for r in rows if r['n_attempts'] != 1),
                'iterations_max': max((r['iterations'] or 0 for r in rows), default=None),
                'iterations_sum': sum(r['iterations'] or 0 for r in rows),
                'dual_infeasibility_unscaled_max': max((r['dual_infeasibility_unscaled'] for r in rows
                                                        if r['dual_infeasibility_unscaled'] is not None),
                                                       default=None),
                'ipopt_terminal_error_over_tol_max': max(tol_ratio, default=None),
                'floor_status_counts': {s: sum(1 for r in rows if r['floor_status'] == s)
                                        for s in sorted({str(r['floor_status']) for r in rows})},
                'wall_s_sum': sum(r['wall_s'] for r in rows)}


def _checkpoint(run_dir, payload):
    payload = dict(payload)
    payload['utc'] = _utc()
    _append_jsonl(os.path.join(run_dir, 'phase_checkpoints.jsonl'), payload)


def _schedule_difference(a, b):
    """max |a - b| over every node/year/day/period for P [MW], Q [MVAr], V [kV]."""
    out = {'p_mw': 0.0, 'q_mvar': 0.0, 'v_kv': 0.0}
    for node_id in a:
        for year in a[node_id]:
            for day in a[node_id][year]:
                for key in out:
                    for x, y in zip(a[node_id][year][day][key], b[node_id][year][day][key]):
                        out[key] = max(out[key], abs(x - y))
    return out


# ======================================================================================================================
#  stage 1 -- lambda_t vs pi_t (zero solves)
# ======================================================================================================================
def units_check(rows, x0a, dnp):
    """Reproduction of W28/W31's committed figures from the S48 models, and the Param-vs-dual identities."""
    tol = UNITS_CHECK_TOL
    worst = {'lmp7_vs_w28': 0.0, 'pi_tn_vs_w28': 0.0, 'pi_vs_w31': 0.0, 'lmp_vs_w31': 0.0, 'y0_vs_w31': 0.0,
             'lambdaE_vs_w31': 0.0, 'lambda_dso_full_vs_y0': 0.0, 'lambda_tso_full_vs_lmp_delta_interior': 0.0,
             'lambda_dso_vs_lambda_tso': 0.0}
    counts = {k: 0 for k in worst}
    missing = []
    w31_hours = {}
    for key, block in dnp['task_a']['blocks'].items():
        for h in block['hours']:
            w31_hours[(key, int(h['hour']))] = h
    for r in rows:
        yd = f"{r['year']}_{r['day']}"
        if r['node_id'] == 7:
            w28 = x0a['blocks'].get(yd)
            if w28 is None:
                missing.append(f'W28 block {yd}')
            else:
                if r['lmp_tn_bus'] is None:
                    missing.append(f"lmp {r['node_id']} {yd} {r['hour']}")
                else:
                    worst['lmp7_vs_w28'] = max(worst['lmp7_vs_w28'], abs(r['lmp_tn_bus'] - w28['lmp7'][r['period']]))
                    counts['lmp7_vs_w28'] += 1
                worst['pi_tn_vs_w28'] = max(worst['pi_tn_vs_w28'], abs(r['pi_tn'] - w28['pi'][r['period']]))
                counts['pi_tn_vs_w28'] += 1
        h = w31_hours.get((f"{r['node_id']}_{yd}", r['hour']))
        if h is None:
            missing.append(f"W31 hour {r['node_id']} {yd} {r['hour']}")
        else:
            for name, mine, theirs in (('pi_vs_w31', r['pi'], h.get('pi')),
                                       ('lmp_vs_w31', r['lmp_tn_bus'], h.get('lmp_tso_bus')),
                                       ('y0_vs_w31', r['y0_dn_ref'], h.get('y0_dn_ref')),
                                       ('lambdaE_vs_w31', r['lambda_dso_full'] - r['pi'], h.get('lambda_E_over_B'))):
                if mine is None or theirs is None:
                    missing.append(f"{name} {r['node_id']} {yd} {r['hour']}")
                    continue
                worst[name] = max(worst[name], abs(mine - theirs))
                counts[name] += 1
        if r['y0_dn_ref'] is not None:
            worst['lambda_dso_full_vs_y0'] = max(worst['lambda_dso_full_vs_y0'], abs(r['lambda_dso_full'] - r['y0_dn_ref']))
            counts['lambda_dso_full_vs_y0'] += 1
        if r['lmp_tn_bus'] is not None and r['tso_interface_delta_interior']:
            worst['lambda_tso_full_vs_lmp_delta_interior'] = max(
                worst['lambda_tso_full_vs_lmp_delta_interior'], abs(r['lambda_tso_full'] - r['lmp_tn_bus']))
            counts['lambda_tso_full_vs_lmp_delta_interior'] += 1
        worst['lambda_dso_vs_lambda_tso'] = max(worst['lambda_dso_vs_lambda_tso'],
                                                abs(r['lambda_dso_full'] - r['lambda_tso_full']))
        counts['lambda_dso_vs_lambda_tso'] += 1
    limits = {'lmp7_vs_w28': tol['repro_eur'], 'pi_tn_vs_w28': tol['repro_eur'], 'pi_vs_w31': tol['repro_eur'],
              'lmp_vs_w31': tol['repro_eur'], 'y0_vs_w31': tol['repro_eur'], 'lambdaE_vs_w31': tol['kkt_eur'],
              'lambda_dso_full_vs_y0': tol['kkt_eur'], 'lambda_tso_full_vs_lmp_delta_interior': tol['kkt_eur'],
              'lambda_dso_vs_lambda_tso': tol['consensus_eur']}
    expected_counts = {'lmp7_vs_w28': 288, 'pi_tn_vs_w28': 288, 'pi_vs_w31': 864, 'lmp_vs_w31': 864,
                       'y0_vs_w31': 864, 'lambdaE_vs_w31': 864, 'lambda_dso_full_vs_y0': 864,
                       'lambda_dso_vs_lambda_tso': 864}
    items = {name: {'max_abs_eur_per_mwh': worst[name], 'limit': limits[name], 'n_compared': counts[name],
                    'n_expected': expected_counts.get(name),
                    'passed': worst[name] <= limits[name] and counts[name] > 0
                    and (expected_counts.get(name) is None or counts[name] == expected_counts[name])}
             for name in worst}
    return {'items': items, 'missing': missing[:50], 'n_missing': len(missing),
            'passed': all(i['passed'] for i in items.values()) and not missing,
            'what_it_reproduces': ("Addendum 32 Q4 (W28 dd0a86b3, W31 d68c814d): the TSO bus-7 marginal cost and the "
                                   "DN interface price, which W31 showed equal the DSO flexibility shadow price "
                                   "(mu_l + c_flex on P-down, mu_l on P-up) hour by hour; reproduced here from the "
                                   "nodal duals (to 1e-9) and from the consensus-dual Params with the ADMM scaling "
                                   "undone (to the KKT tolerance)")}


def flag_rows(rows):
    sources = ('lambda_dso_full', 'lambda_dso_linear', 'lambda_tso_full', 'lmp_tn_bus', 'y0_dn_ref')
    flagged = []
    for r in rows:
        r['lambda_t'] = r['lambda_dso_full']
        r['lambda_minus_pi'] = r['lambda_t'] - r['pi']
        r['lambda_neq_pi'] = abs(r['lambda_minus_pi']) > LAMBDA_NEQ_PI_TOL_EUR
        r['lambda_neq_pi_by_source'] = {s: (None if r[s] is None else abs(r[s] - r['pi']) > LAMBDA_NEQ_PI_TOL_EUR)
                                        for s in sources}
        r['cflex_lt_pi'] = r['c_flex'] < r['pi']
        r['coordination_proper_hour'] = bool(r['lambda_neq_pi'] and r['cflex_lt_pi'])
        values = [v for v in r['lambda_neq_pi_by_source'].values() if v is not None]
        r['sources_agree_on_neq'] = len(set(values)) == 1
        if r['coordination_proper_hour']:
            flagged.append({k: r[k] for k in ('node_id', 'year', 'day', 'hour', 'pi', 'c_flex', 'lambda_t',
                                              'lambda_minus_pi', 'lmp_tn_bus', 'y0_dn_ref')})
    return flagged


def summarise_flags(rows, planning):
    by_node, by_node_hour, by_node_year = {}, {}, {}
    weighted = {}
    for r in rows:
        n = r['node_id']
        by_node.setdefault(n, {'rows': 0, 'lambda_neq_pi': 0, 'cflex_lt_pi': 0, 'coordination_proper_hours': 0})
        by_node[n]['rows'] += 1
        by_node[n]['lambda_neq_pi'] += int(r['lambda_neq_pi'])
        by_node[n]['cflex_lt_pi'] += int(r['cflex_lt_pi'])
        by_node[n]['coordination_proper_hours'] += int(r['coordination_proper_hour'])
        key = f"{n}|{r['hour']}"
        by_node_hour[key] = by_node_hour.get(key, 0) + int(r['coordination_proper_hour'])
        key = f"{n}|{r['year']}"
        by_node_year[key] = by_node_year.get(key, 0) + int(r['coordination_proper_hour'])
        dn = planning.distribution_networks[n]
        day_weight = float(dn.years[r['year']]) * float(dn.days[r['day']])
        weighted[n] = weighted.get(n, 0.0) + day_weight * int(r['coordination_proper_hour'])
    return {'by_node': by_node, 'coordination_proper_hours_by_node_hour_of_day': by_node_hour,
            'coordination_proper_hours_by_node_year': by_node_year,
            'coordination_proper_hours_day_weighted_by_node': weighted,
            'n_rows': len(rows), 'n_coordination_proper_hours': sum(int(r['coordination_proper_hour']) for r in rows),
            'n_rows_sources_disagree_on_neq': sum(1 for r in rows if not r['sources_agree_on_neq'])}


def pf_capture_cross_check(rows):
    """Fourth, informational source (W106: was the third under W93): the per-cycle capture's lambda_dso at the
    certification cycle (the dual after that cycle's update) against the Param the last DSO solve used (dual before
    it). Reads the COORDINATED (settled) cell's stride file."""
    path = _verified_path(COORDINATED['pf_entry_stride'])
    last = json.loads(_read_last_line(path))
    by_key = {}
    for e in last.get('entries', []):
        if e.get('power_type') == 'p':
            by_key[(int(e['node_id']), str(e['year']), str(e['day']), int(e['period']))] = e
    worst_raw = worst_eur = 0.0
    n = 0
    for r in rows:
        e = by_key.get((r['node_id'], str(r['year']), str(r['day']), r['period']))
        if e is None:
            continue
        capture_param_units = float(e['lambda_dso']) / float(e['s_base_dso'])
        diff = capture_param_units - r['raw']['dual_pf_p_req_dso']
        eur = diff * r['raw']['eff_dso'] / r['raw']['rating_mva']
        worst_raw = max(worst_raw, abs(diff))
        worst_eur = max(worst_eur, abs(eur))
        n += 1
    return {'cycle': last.get('cycle'), 'n_compared': n, 'max_abs_param_units': worst_raw,
            'max_abs_eur_per_mwh': worst_eur, 'certification_cycle_declared': COORDINATED['certification_cycle'],
            'note': 'informational: the capture holds the dual AFTER the cycle update, the Param the dual the last '
                    'DSO solve used; they differ by one dual step'}


SIDECAR_ALIGNMENT_NOTE = (
    "interface_dual_capture: the sidecar row of cycle n holds dual_vars['pf'][agent]['current'] after cycle n's dual "
    "updates and before the AA step; 'The dual_vars value is the model Param dual_pf_p_req of the NEXT cycle's solve'. "
    'The persisted models are the cycle-k* models, so their Param = sidecar row k*-1 / s_base, bitwise (AA not accepted '
    'at k*-1 or k*: aa_per_cycle.jsonl). Row k* is the dual AFTER the last update = one dual step past the Param: '
    'converted by the same interface-P scaling (sigma via admm_objective_scale, the interface rating, B) it is the '
    'lambda_t the next cycle would price at, compared with the Param-derived lambda_dso_full (the Param plus the '
    'current AL gradient term) within the consensus tolerance')


def lambda_sidecar_cross_check(rows):
    """W106 (Addendum 54 Ruling 3 order): the THIRD lambda_t source -- the settled cell's per-cycle sidecar
    `interface_duals_per_cycle.jsonl` (sha-verified; read with production's `interface_dual_capture.read_sidecar`),
    rows k*-1 and k*, against the consensus-dual Params of the persisted cycle-k* models (`rows` from
    `uncoordinated_benchmark.interface_price_terms` on those models). Zero solves; pure reads."""
    import interface_dual_capture as IDC
    k = int(COORDINATED['certification_cycle'])
    header, lines = IDC.read_sidecar(_verified_path(COORDINATED['interface_duals']))
    out = {'certification_cycle': k, 'rows_used': [k - 1, k], 'alignment_note': SIDECAR_ALIGNMENT_NOTE,
           'tolerance_lambda_eur_per_mwh': UNITS_CHECK_TOL['consensus_eur'],
           'header_schema': None if header is None else header.get('schema')}
    missing = [c for c in (k - 1, k) if c not in lines or lines[c].get('captured') is not True]
    if header is None or missing:
        out.update({'passed': False, 'reason': f'sidecar header present {header is not None}; rows missing or '
                                               f'not captured {missing}'})
        return out
    index = {(int(b[0]), str(b[1]), str(b[2])): i for i, b in enumerate(header['blocks'])}
    meta = {(int(m['node_id']), str(m['year']), str(m['day'])): m for m in header['metadata']['per_block']}
    prev, last = lines[k - 1], lines[k]
    n = n_bitwise_dso = n_bitwise_tso = 0
    worst = {'align_dso_param_units': 0.0, 'align_tso_param_units': 0.0, 'metadata_scaling_abs': 0.0,
             'k_dso_vs_lambda_dso_full': 0.0, 'k_dso_vs_lambda_dso_linear': 0.0, 'k_tso_vs_lambda_tso_full': 0.0,
             'k_tso_vs_lambda_tso_linear': 0.0, 'k_dso_vs_k_tso': 0.0, 'k_dso_vs_y0_dn_ref': 0.0,
             'k_tso_vs_lmp_tn_bus_delta_interior': 0.0}
    per_row = []
    for r in rows:
        key = (int(r['node_id']), str(r['year']), str(r['day']))
        if key not in index or key not in meta:
            out.update({'passed': False, 'reason': f'block {key} absent from the sidecar header'})
            return out
        b, p, raw, m = index[key], int(r['period']), r['raw'], meta[key]
        s_dso, s_tso = float(m['dso_base_mva']), float(m['tso_base_mva'])
        for got, want in ((s_dso, raw['B_dn']), (s_tso, raw['B_tn']), (float(m['interface_rating_mva']),
                                                                       raw['rating_mva']),
                          (float(m['admm_objective_scale_dso']), raw['eff_dso']),
                          (float(m['admm_objective_scale_tso']), raw['eff_tso'])):
            worst['metadata_scaling_abs'] = max(worst['metadata_scaling_abs'], abs(got - want))
        a_dso = float(prev['lambda_pf_p_dso'][b][p]) / s_dso
        a_tso = float(prev['lambda_pf_p_tso'][b][p]) / s_tso
        n_bitwise_dso += int(a_dso.hex() == float(raw['dual_pf_p_req_dso']).hex())
        n_bitwise_tso += int(a_tso.hex() == float(raw['dual_pf_p_req_tso']).hex())
        worst['align_dso_param_units'] = max(worst['align_dso_param_units'], abs(a_dso - raw['dual_pf_p_req_dso']))
        worst['align_tso_param_units'] = max(worst['align_tso_param_units'], abs(a_tso - raw['dual_pf_p_req_tso']))
        r_pu_dn, r_pu_tn = raw['rating_mva'] / raw['B_dn'], raw['rating_mva'] / raw['B_tn']
        lam_k_dso = r['pi'] + raw['eff_dso'] * (float(last['lambda_pf_p_dso'][b][p]) / s_dso) / r_pu_dn / raw['B_dn']
        lam_k_tso = r['pi'] - raw['eff_tso'] * (float(last['lambda_pf_p_tso'][b][p]) / s_tso) / r_pu_tn / raw['B_tn']
        for name, x, y in (('k_dso_vs_lambda_dso_full', lam_k_dso, r['lambda_dso_full']),
                           ('k_dso_vs_lambda_dso_linear', lam_k_dso, r['lambda_dso_linear']),
                           ('k_tso_vs_lambda_tso_full', lam_k_tso, r['lambda_tso_full']),
                           ('k_tso_vs_lambda_tso_linear', lam_k_tso, r['lambda_tso_linear']),
                           ('k_dso_vs_k_tso', lam_k_dso, lam_k_tso),
                           ('k_dso_vs_y0_dn_ref', lam_k_dso, r['y0_dn_ref'])):
            if y is not None:
                worst[name] = max(worst[name], abs(x - y))
        if r['lmp_tn_bus'] is not None and r['tso_interface_delta_interior']:
            worst['k_tso_vs_lmp_tn_bus_delta_interior'] = max(worst['k_tso_vs_lmp_tn_bus_delta_interior'],
                                                              abs(lam_k_tso - r['lmp_tn_bus']))
        per_row.append({'node_id': r['node_id'], 'year': r['year'], 'day': r['day'], 'hour': r['hour'],
                        'lambda_sidecar_k_dso': lam_k_dso, 'lambda_sidecar_k_tso': lam_k_tso})
        n += 1
    alignment_bitwise = n_bitwise_dso == n and n_bitwise_tso == n
    agrees = worst['k_dso_vs_lambda_dso_full'] <= UNITS_CHECK_TOL['consensus_eur']
    out.update({
        'n_rows': n, 'n_alignment_bitwise_dso': n_bitwise_dso, 'n_alignment_bitwise_tso': n_bitwise_tso,
        'alignment_row_k_minus_1_bitwise_to_param': bool(alignment_bitwise),
        'row_k_agrees_with_lambda_dso_full_within_tolerance': bool(agrees),
        'max_abs': worst, 'per_row_lambda_sidecar_k': per_row,
        'passed': bool(n == 864 and alignment_bitwise and agrees and worst['metadata_scaling_abs'] == 0.0),
        'pass_rule': ('864 rows; row k*-1 / s_base == the persisted Params BITWISE on both sides; row k* lambda_dso '
                      '(EUR/MWh) within the consensus tolerance of the Param-derived lambda_dso_full; the header '
                      'scaling metadata (rating, both B, both admm_objective_scale) equal to the models\' exactly'),
    })
    return out


def stage_lambda_look(run_dir):
    srp, UB, O, R = _production()
    planning = O.load_baseline()['planning']          # read-only: network data, prices, weights
    checks = common_capture_checklist(srp, UB, R, planning)
    for name in ('x0_capture_analysis', 'dn_plateau'):
        checks[f's48_{name}_git_tracked_and_clean'] = _git_tracked_clean(S48[name])
    x0a = _load_json(S48['x0_capture_analysis'])
    dnp = _load_json(S48['dn_plateau'])
    checks['s48_x0_capture_analysis_names_pickle'] = x0a['inputs']['pickle']['sha256'] == S48['certified_models']['sha256']
    checks['s48_dn_plateau_names_pickle'] = dnp['inputs']['pickle']['sha256'] == S48['certified_models']['sha256']
    checks['s48_dn_plateau_has_864_hours'] = sum(len(b['hours']) for b in dnp['task_a']['blocks'].values()) == 864
    _assert_checklist(checks, 'lambda-look (before loading models)')

    s48_models = _load_pickle_verified(S48['certified_models'])
    s48_checks = _model_capture_checks(s48_models, 's48')
    _assert_checklist(s48_checks, 'lambda-look S48 models')
    rows_s48 = UB.interface_price_terms(planning, s48_models)
    del s48_models
    gc.collect()
    units = units_check(rows_s48, x0a, dnp)
    _log(f"units check: {'PASS' if units['passed'] else 'FAIL'} "
         f"{ {k: (v['max_abs_eur_per_mwh'], v['passed']) for k, v in units['items'].items()} }")
    _checkpoint(run_dir, {'phase': 'units_check', 'passed': units['passed']})

    # W106: the settled cycle-181 models (Addendum 54 Ruling 3); W93/W94 loaded the W86 models here (unwired)
    coordinated_models = _load_pickle_verified(COORDINATED['certified_models'])
    coordinated_checks = _model_capture_checks(coordinated_models, 'coordinated')
    _assert_checklist(coordinated_checks, 'lambda-look coordinated (settled) models')
    rows = UB.interface_price_terms(planning, coordinated_models)
    del coordinated_models
    gc.collect()
    identities_coordinated = units_check_identities_only(rows)
    sidecar = lambda_sidecar_cross_check(rows)
    _log(f"sidecar cross-check (third source, cycles {sidecar['rows_used']}): "
         f"{'PASS' if sidecar['passed'] else 'FAIL'} {sidecar.get('max_abs')}")
    capture = pf_capture_cross_check(rows)
    flagged = flag_rows(rows)
    summary = summarise_flags(rows, planning)
    guard = _check_guard(0, 'lambda-look end')
    result = provenance({
        'run': 'lambda_look', 'solve_profile_guard': guard,
        'capture_path_checklist': {**checks, **s48_checks, **coordinated_checks},
        'definitions': {
            'lambda_t': 'lambda_dso_full: pi_t + eff_dso * [dual_pf_p_req + rho_pf (E - z)/r_pu] / (r_pu * B_dn) -- '
                        'the interface price the DSO faced at the certified point (uncoordinated_benchmark.'
                        'interface_price_terms); the other sources are reported beside it',
            'lambda_neq_pi': f'|lambda_t - pi_t| > {LAMBDA_NEQ_PI_TOL_EUR} EUR/MWh',
            'coordination_proper_hour': 'lambda_neq_pi AND c_flex_t < pi_t (Addendum 49 ruling 1: these hours '
                                        'bound coordination proper from above)',
            'lambda_sources': ['consensus-dual Param dual_pf_p_req (DSO and TSO sides, persisted settled models)',
                               'TN interface-bus power-balance dual (TSO dual suffix)',
                               'DN reference-bus power-balance dual (DSO dual suffix, W31 y0)',
                               'lambda_t sidecar interface_duals_per_cycle.jsonl rows k*-1 and k* (W106 third source)',
                               'per-cycle capture pf_entry_stride lambda_dso (informational)'],
        },
        'units_check_s48': units,
        'identities_coordinated': identities_coordinated,
        'lambda_sidecar_cross_check': sidecar,
        'pf_capture_cross_check_coordinated': capture,
        'prediction_usable': bool(units['passed'] and identities_coordinated['passed'] and sidecar['passed']),
        'prediction_usable_rule': 'the table drives a prediction only if the S48 units check reproduces Addendum '
                                  '32 (W28/W31), the identities hold on the settled models (Addendum 49 '
                                  'clarification) and the lambda_t sidecar cross-check passes (W106: "cross-check all '
                                  'three")',
        'summary_coordinated': summary,
        'coordination_proper_hours_coordinated': flagged,
        'table_coordinated': rows,
    })
    _write_json_once(os.path.join(run_dir, 'lambda_look.json'), result)
    return 0 if result['prediction_usable'] else 1


def units_check_identities_only(rows):
    """The two Param-vs-dual identities and the consensus agreement on the coordinated (W106: settled) rows (no
    committed figures)."""
    tol = UNITS_CHECK_TOL
    dso = [abs(r['lambda_dso_full'] - r['y0_dn_ref']) for r in rows if r['y0_dn_ref'] is not None]
    tso = [abs(r['lambda_tso_full'] - r['lmp_tn_bus']) for r in rows
           if r['lmp_tn_bus'] is not None and r['tso_interface_delta_interior']]
    cons = [abs(r['lambda_dso_full'] - r['lambda_tso_full']) for r in rows]
    items = {'lambda_dso_full_vs_y0': (max(dso, default=None), tol['kkt_eur'], len(dso)),
             'lambda_tso_full_vs_lmp_delta_interior': (max(tso, default=None), tol['kkt_eur'], len(tso)),
             'lambda_dso_vs_lambda_tso': (max(cons, default=None), tol['consensus_eur'], len(cons))}
    out = {k: {'max_abs_eur_per_mwh': v[0], 'limit': v[1], 'n_compared': v[2],
               'passed': v[0] is not None and v[0] <= v[1]} for k, v in items.items()}
    return {'items': out, 'passed': all(i['passed'] for i in out.values()) and len(rows) == 864,
            'n_rows': len(rows)}


# ======================================================================================================================
#  stage 2 -- common-Q gate (zero solves)
# ======================================================================================================================
# W106: UNWIRED, RETAINED -- the W93/W94 loader of the W86 models; no stage and no check calls it (asserted by the
# W106 zero-solve checks). Unwire, never delete.
def _load_w86_models():
    payload = _load_pickle_verified(W86['certified_models'])
    esso = _load_pickle_verified(W86['esso_models'])
    return {'tso': payload['tso'], 'dso': payload['dso'], 'esso': esso}


def _load_coordinated_models():
    """W106: the coordinated cell's persisted terminal models (the settled cycle-181 x = 0 cell), sha-verified."""
    payload = _load_pickle_verified(COORDINATED['certified_models'])
    esso = _load_pickle_verified(COORDINATED['esso_models'])
    return {'tso': payload['tso'], 'dso': payload['dso'], 'esso': esso}


def stage_common_q_gate(run_dir):
    srp, UB, O, R = _production()
    planning = O.load_baseline()['planning']
    checks = common_capture_checklist(srp, UB, R, planning)
    _assert_checklist(checks, 'common-q-gate (before loading models)')
    record = _load_json(COORDINATED['evaluation_record'])
    models = _load_coordinated_models()
    checks_m = _model_capture_checks(models, 'coordinated', need_duals=False)
    _assert_checklist(checks_m, 'common-q-gate coordinated (settled) models')

    ev0 = UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=TIE_BREAKER['evaluation'],
                               require_unchanged=True)
    gate = UB.common_q_gate(ev0, record['recourse_components'])
    _log(f"common-Q gate: {gate['status']} evaluated {gate['gross_evaluated']!r} certified {gate['gross_certified']!r} "
         f"diff {gate['difference']!r}; {gate['n_fields_bitwise_equal']}/{gate['n_fields']} fields bitwise")
    ev0_again = UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=TIE_BREAKER['evaluation'],
                                     require_unchanged=True)
    idempotent = ev0_again['gross_operational_cost_hex'] == ev0['gross_operational_cost_hex']

    # negative control 1: a deliberately wrong configuration is REFUSED under require_unchanged
    refused, refusal_text = False, None
    try:
        UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=1.0, require_unchanged=True)
    except UB.CommonQConfigurationError as error:
        refused, refusal_text = True, str(error)[:500]
    # negative control 2 + discrimination: repriced at 1, the gate must FAIL, by exactly the curtailment it prices
    ev1 = UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=1.0, require_unchanged=False)
    gate1 = UB.common_q_gate(ev1, record['recourse_components'])
    priced = (ev0['curtailment']['totals']['TSO']['eur_at_1_block_weighted']
              + ev0['curtailment']['totals']['DSO']['eur_at_1_block_weighted'])
    certified_priced = (record.get('component_decomposition_totals_weighted') or {}).get(
        'res_curtailment_definitional_at_weight_1')
    delta_q = ev1['gross_operational_cost'] - ev0['gross_operational_cost']
    explained_residual = delta_q - priced
    discriminates = priced > 0.0
    negative = {
        'wrong_configuration_refused_under_require_unchanged': refused, 'refusal': refusal_text,
        'repriced_at_1_gate_status': gate1['status'], 'repriced_at_1_gate_failed': not gate1['passed'],
        'delta_q_eur': delta_q, 'priced_curtailment_at_1_eur': priced,
        'certified_record_res_curtailment_definitional_at_weight_1': certified_priced,
        'priced_vs_certified_record_abs': (None if certified_priced is None else abs(priced - certified_priced)),
        'explained_residual_eur': explained_residual,
        'explained_to_tolerance': abs(explained_residual) <= NEGATIVE_CONTROL_EXPLAINED_ABS_TOL_EUR,
        'gate_can_discriminate_0_from_1': discriminates,
        'note': ('if the x = 0 cell curtailed nothing the gate cannot discriminate 0 from 1 and the rule stands by '
                 'principle (Addendum 49 clarification)'),
    }
    negative_ok = refused and (not discriminates or (not gate1['passed'] and negative['explained_to_tolerance']))
    # W106: the reference is the settled value (Addendum 54 Ruling 3) -- stated and compared explicitly
    q_ref = float(COORDINATED['certified_gross'])
    reference_q = {'value': q_ref, 'hex': q_ref.hex(), 'source': COORDINATED['certified_gross_source'],
                   'certification_cycle': COORDINATED['certification_cycle'],
                   'evaluated_equals_reference_bitwise': float(ev0['gross_operational_cost']).hex() == q_ref.hex(),
                   'evaluated_minus_reference_eur': float(ev0['gross_operational_cost']) - q_ref,
                   'previous_reference_reported_only': COORDINATED_PREVIOUS_REFERENCE}
    guard = _check_guard(0, 'common-q-gate end')
    status = ('PASS' if (gate['passed'] and reference_q['evaluated_equals_reference_bitwise'] and idempotent
                         and negative_ok) else 'FAIL')
    result = provenance({
        'run': 'common_q_gate', 'status': status, 'solve_profile_guard': guard,
        'capture_path_checklist': {**checks, **checks_m}, 'reference_q': reference_q,
        'gate': gate, 'idempotent_repeat_bitwise': idempotent, 'negative_controls': negative,
        'evaluation_at_0': {k: ev0[k] for k in ('objective_convention', 'evaluation_curtailment_penalty',
                                                'gross_operational_cost', 'gross_operational_cost_hex',
                                                'pricing_changed_by_evaluation', 'curtailment')},
        'evaluation_at_1_gross': ev1['gross_operational_cost'],
    })
    _write_json_once(os.path.join(run_dir, 'common_q_gate.json'), result)
    return 0 if status == 'PASS' else 1


# ======================================================================================================================
#  stage 3 -- 24-solve penalty-vs-fixed TSO check
# ======================================================================================================================
def stage_tso_coupling_check(run_dir, run_id):
    srp, UB, O, R = _production()
    planning, logs_dir = _new_planning(O, run_id)
    solver_options = apply_arm_solver_options(srp, planning)
    candidate = _x0_candidate(srp, planning)
    checks = common_capture_checklist(srp, UB, R, planning)
    # W106: the pinned targets (P, Q and the V pin) come from the settled cycle-181 point (W93/W94: W86, unwired)
    certified = _load_pickle_verified(COORDINATED['certified_models'])
    checks.update(_model_capture_checks(certified, 'coordinated', need_duals=False))
    reference = UB.coordinated_reference_structure(planning, certified)
    targets = UB.get_dso_interface_schedule(planning, certified['dso'])
    certified_tso_metrics = UB.tso_block_metrics(planning, certified['tso'], targets)
    del certified
    gc.collect()
    declared = 2 * UB.declared_solve_count(planning)['tso']
    checks['declared_equals_srp1_literal'] = declared == SRP1_DECLARED['coupling_check_solves']
    checks['check_fixed_side_pins_voltage_declared'] = UB.COUPLING_CHECK_FIXED_SIDE_PIN_INTERFACE_VOLTAGE is True
    checks['tso_arm_does_not_pin_voltage_declared'] = UB.TSO_ARM_PIN_INTERFACE_VOLTAGE is False
    checks['voltage_target_capture_path_v_kv_present'] = all(
        len(targets[n][y][d]['v_kv']) == 24 for n in targets for y in targets[n] for d in targets[n][y])
    _assert_checklist(checks, 'tso-coupling-check (before any solve)')
    _check_guard(0, 'tso-coupling-check before solves')
    sink = SolveSink(run_dir)
    out = UB.run_tso_interface_coupling_check(planning, candidate, targets,
                                              tso_curtailment_penalty=TIE_BREAKER['decision']['tso'],
                                              reference_structure=reference, record_callback=sink)
    guard = _check_guard(declared, 'tso-coupling-check end')
    per_block = {}
    for coupling in (UB.TSO_COUPLING_FIXED, UB.TSO_COUPLING_TRACKING_PENALTY):
        solves = {r['block']: r for r in sink.records if r['phase'] == f'coupling_check:{coupling}'}
        for label, metrics in out[coupling]['metrics'].items():
            row = per_block.setdefault(label, {'certified_coordinated': certified_tso_metrics[label]})
            row[coupling] = {**metrics, **{k: solves[label][k] for k in (
                'iterations', 'exit', 'dual_infeasibility_scaled', 'dual_infeasibility_unscaled',
                'overall_nlp_error_scaled', 'n_attempts', 'termination_condition', 'mu_final', 'floor_status')}}
    totals = {c: sum(per_block[b][c]['block_gross_cost_weighted'] for b in per_block)
              for c in (UB.TSO_COUPLING_FIXED, UB.TSO_COUPLING_TRACKING_PENALTY)}
    totals['certified_coordinated'] = sum(per_block[b]['certified_coordinated']['block_gross_cost_weighted']
                                          for b in per_block)
    result = provenance({
        'run': run_id, 'solve_profile_guard': guard, 'declared_solves': declared,
        'capture_path_checklist': checks, 'solver_options': solver_options, 'ipopt_logs_dir': logs_dir,
        'targets': 'the coordinated DSO schedule at the settled cycle-181 point (W106; persisted DSO models of '
                   'd110bd1a5977df1e_x0: expected_interface_pf_p/q, expected_interface_vmag), the SAME for both '
                   'couplings',
        'fixed_side_voltage_pin': UB.COUPLING_CHECK_FIXED_SIDE_VOLTAGE_PIN,
        'tracking_penalty_note': ("production's _add_tso_scenario_tracking_penalty tracks V as well as P and Q; the "
                                  "check's fixed side therefore pins P, Q AND V at the same targets (W94), so the two "
                                  'sides are the same mathematical problem and any difference is numerical. The TSO '
                                  'ARM (stage 4) is NOT pinned: P/Q fixed, V within its normal bounds (ruling 2)'),
        'voltage_pin_records': {c: out[c]['voltage_pin'] for c in out},
        'declared_fixed_rows_per_block': {
            UB.TSO_COUPLING_FIXED: SRP1_DECLARED['coupling_check_fixed_side_fixed_rows_per_block'],
            UB.TSO_COUPLING_TRACKING_PENALTY: SRP1_DECLARED['coupling_check_penalty_side_fixed_rows_per_block']},
        'fixed_rows_match_declared': {
            c: all(r['fixed_rows'] == SRP1_DECLARED[k] for r in out[c]['structure'].values())
            for c, k in ((UB.TSO_COUPLING_FIXED, 'coupling_check_fixed_side_fixed_rows_per_block'),
                         (UB.TSO_COUPLING_TRACKING_PENALTY, 'coupling_check_penalty_side_fixed_rows_per_block'))},
        'per_block': per_block, 'tn_cost_weighted_totals': totals,
        'fixed_minus_penalty_tn_cost_weighted': totals[UB.TSO_COUPLING_FIXED] - totals[UB.TSO_COUPLING_TRACKING_PENALTY],
        'structure': {c: out[c]['structure'] for c in out},
        'solve_summary': {c: sink.summary(f'coupling_check:{c}') for c in out},
        'reported_not_gating': True,
    })
    _write_json_once(os.path.join(run_dir, 'tso_coupling_check.json'), result)
    return 0


# ======================================================================================================================
#  stage 4 / 5 -- arms (48 + 36 [+ 48]) and the passive tie-breaker variants (48)
# ======================================================================================================================
def stage_arm(run_dir, run_id, *, arm, start, dso_tie_breaker, consistency):
    srp, UB, O, R = _production()
    planning, logs_dir = _new_planning(O, run_id)
    solver_options = apply_arm_solver_options(srp, planning)
    candidate = _x0_candidate(srp, planning)
    checks = common_capture_checklist(srp, UB, R, planning)
    # W106: the warm-from-certified start and the reference structure are the settled cycle-181 point (W93: W86)
    certified = _load_pickle_verified(COORDINATED['certified_models'])
    esso = _load_pickle_verified(COORDINATED['esso_models'])
    checks.update(_model_capture_checks(certified, 'coordinated', need_duals=False))
    reference = UB.coordinated_reference_structure(planning, certified)
    warm = UB.extract_model_values(planning, certified) if start != UB.START_COLD else None
    coordinated_schedule = UB.get_dso_interface_schedule(planning, certified['dso'])
    del certified
    gc.collect()
    declared = UB.declared_solve_count(planning)
    checks['declared_arm_equals_srp1_literal'] = declared['total'] == SRP1_DECLARED['arm_solves']
    checks['declared_reevaluation_equals_srp1_literal'] = declared['dso'] == SRP1_DECLARED['reevaluation_solves']
    checks['declared_pass_equals_srp1_literal'] = declared['total'] == SRP1_DECLARED['sequential_pass_solves']
    checks['esso_salvage_is_zero_at_x0'] = planning.shared_ess_data.get_salvage_value(esso) == 0.0
    checks['tie_breaker_decision_declared'] = dso_tie_breaker is not None
    _assert_checklist(checks, f'{run_id} (before any solve)')
    _check_guard(0, f'{run_id} before solves')
    sink = SolveSink(run_dir)
    expected = 0
    phases = []

    # phase A -- the arm
    arm_out = UB.run_operational_planning_uncoordinated(
        planning, candidate, arm=arm, dso_curtailment_penalty=dso_tie_breaker,
        tso_curtailment_penalty=TIE_BREAKER['decision']['tso'], reference_structure=reference, start=start,
        warm_values=warm, perturbation=PERTURBATION if start == UB.START_PERTURBED else None, record_callback=sink)
    expected += declared['total']
    phases.append(_check_guard(expected, f'{run_id} phase A (arm)'))
    models = {'tso': arm_out['models']['tso'], 'dso': arm_out['models']['dso'], 'esso': esso}
    evaluation = UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=TIE_BREAKER['evaluation'],
                                      require_unchanged=False)
    _checkpoint(run_dir, {'phase': 'A', 'gross_operational_cost': evaluation['gross_operational_cost'],
                          'solves': expected, 'guard': dict(_GUARD.counts)})
    _log(f"{run_id}: phase A Q = {evaluation['gross_operational_cost']!r}")
    result = {
        'run': run_id, 'arm': arm, 'start': start,
        'decision_tie_breaker': {'dso': dso_tie_breaker, 'tso': TIE_BREAKER['decision']['tso']},
        'evaluation_tie_breaker': TIE_BREAKER['evaluation'],
        'capture_path_checklist': checks, 'solver_options': solver_options, 'ipopt_logs_dir': logs_dir,
        'declared_solves': {'arm': declared, 'reevaluation': declared['dso'] if consistency else 0,
                            'sequential_pass_if_triggered': declared['total'] if consistency else 0},
        'structure': arm_out['structure'], 'build': {k: _compact_build(v) for k, v in arm_out['build'].items()},
        'start_records': arm_out['start_records'],
        'phase_A': {'evaluation': _compact_evaluation(evaluation),
                    'dso_interface_schedule': arm_out['dso_interface_schedule'],
                    'tso_interface_schedule': arm_out['tso_interface_schedule'],
                    'tso_vs_dso_schedule_max_abs': _schedule_difference_tso_dso(arm_out),
                    'dso_schedule_vs_coordinated_max_abs': _schedule_difference(arm_out['dso_interface_schedule'],
                                                                                coordinated_schedule),
                    'solve_summary': sink.summary(f'{arm}:{start}')},
    }
    arm_cost = evaluation['gross_operational_cost']
    arm_cost_source = 'phase_A'

    if consistency:
        mismatch = UB.interface_voltage_mismatch(planning, arm_out['models']['tso'], arm_out['models']['dso'])
        reeval = UB.reevaluate_dso_at_actual_voltage(planning, arm_out['models']['dso'], mismatch['v_actual_dn_pu'],
                                                     tolerances=CONSISTENCY_TOL, record_callback=sink)
        expected += declared['dso']
        phases.append(_check_guard(expected, f'{run_id} phase B (consistency re-evaluation)'))
        _checkpoint(run_dir, {'phase': 'B', 'trigger': reeval['trigger_sequential_pass'],
                              'max_abs_dv_dn_pu': mismatch['max_abs_dv_dn_pu'], 'solves': expected})
        result['phase_B_consistency'] = {
            'max_abs_dv_dn_pu': mismatch['max_abs_dv_dn_pu'],
            'max_abs_dv_dn_pu_per_node_hour': mismatch['max_abs_dv_dn_pu_per_node_hour'],
            'dv_entries': mismatch['entries'],
            'reevaluation': reeval,
            'trigger_rule': ('any hard (outside the DSO model\'s own vmag_sqr band or reference-generator bounds) '
                             '> hard_tol, any thermal > thermal_tol, or any soft violation exceeding the arm\'s own '
                             'voltage slack at the same index by > soft_excess_tol'),
            'solve_summary': sink.summary('consistency:'),
        }
        if reeval['trigger_sequential_pass']:
            shift = UB.pin_dso_interface_voltage(planning, arm_out['models']['dso'], mismatch['v_actual_dn_pu'])
            UB.solve_dso_models(planning, arm_out['models']['dso'], phase='sequential_pass:dso', record_callback=sink)
            new_schedule = UB.get_dso_interface_schedule(planning, arm_out['models']['dso'])
            moved = UB.set_tso_interface_targets(planning, arm_out['models']['tso'], new_schedule)
            UB.solve_tso_model(planning, arm_out['models']['tso'], phase='sequential_pass:tso', record_callback=sink)
            expected += declared['total']
            phases.append(_check_guard(expected, f'{run_id} phase C (sequential pass)'))
            evaluation_c = UB.evaluate_common_q(planning, models,
                                                evaluation_curtailment_penalty=TIE_BREAKER['evaluation'],
                                                require_unchanged=False)
            mismatch_c = UB.interface_voltage_mismatch(planning, arm_out['models']['tso'], arm_out['models']['dso'])
            _checkpoint(run_dir, {'phase': 'C', 'gross_operational_cost': evaluation_c['gross_operational_cost'],
                                  'solves': expected})
            result['phase_C_sequential_pass'] = {
                'dso_voltage_shift_max_pu': shift, 'tso_target_move_max': moved,
                'evaluation': _compact_evaluation(evaluation_c),
                'effect_on_q_eur': evaluation_c['gross_operational_cost'] - evaluation['gross_operational_cost'],
                'max_abs_dv_dn_pu_after_pass': mismatch_c['max_abs_dv_dn_pu'],
                'dso_interface_schedule': new_schedule,
                'solve_summary': sink.summary('sequential_pass:'),
            }
            arm_cost = evaluation_c['gross_operational_cost']
            arm_cost_source = 'phase_C_sequential_pass'
    result['arm_cost'] = {'gross_operational_cost': arm_cost, 'source': arm_cost_source,
                          'objective_convention': evaluation['objective_convention']}
    result['solve_profile_guard'] = {'phases': phases, 'final': _check_guard(expected, f'{run_id} end')}
    result['solve_summary_all'] = sink.summary()
    _write_json_once(os.path.join(run_dir, f'{run_id}.json'), provenance(result))
    return 0


def _compact_build(build):
    out = {k: v for k, v in build.items() if k != 'blocks'}
    out['blocks'] = {label: {k: (len(v) if isinstance(v, list) else v) for k, v in rec.items()}
                     for label, rec in build.get('blocks', {}).items()}
    return out


def _compact_evaluation(evaluation):
    return {k: evaluation[k] for k in ('objective_convention', 'evaluation_curtailment_penalty',
                                       'gross_operational_cost', 'gross_operational_cost_hex', 'recourse_components',
                                       'block_components', 'salvage_block', 'pricing_before',
                                       'pricing_changed_by_evaluation', 'curtailment')}


def _schedule_difference_tso_dso(arm_out):
    """The TSO's achieved interface P/Q against the DSO schedule it was fixed to (the fixed rows' residual)."""
    worst = {'p_mw': 0.0, 'q_mvar': 0.0}
    dso, tso = arm_out['dso_interface_schedule'], arm_out['tso_interface_schedule']
    for node_id in dso:
        for year in dso[node_id]:
            for day in dso[node_id][year]:
                for key in worst:
                    for x, y in zip(dso[node_id][year][day][key], tso[node_id][year][day][key]):
                        worst[key] = max(worst[key], abs(x - y))
    return worst


# ======================================================================================================================
#  stage 6 -- report (zero solves)
# ======================================================================================================================
def stage_report(run_dir):
    root = _abs(OUT_ROOT_REL)

    def load(run_id, name=None):
        path = os.path.join(root, run_id, f'{name or run_id}.json')
        if not os.path.exists(path):
            return None
        with open(path) as handle:
            return json.load(handle)

    gate = load('common_q_gate')
    look = load('lambda_look')
    coupling = load('tso_coupling_check')
    arms = {run_id: load(run_id) for run_id in ARM_RUN_IDS}
    variants = {run_id: load(run_id) for run_id in VARIANT_RUN_IDS}
    missing = [k for k, v in list(arms.items()) + list(variants.items()) if v is None]
    missing += [k for k, v in (('common_q_gate', gate), ('lambda_look', look), ('tso_coupling_check', coupling))
                if v is None]
    q_coord = COORDINATED['certified_gross']       # W106: the settled Q181 (Addendum 53 Ruling 4; W93: W86 Q132)
    band_coord = COORDINATED_REPRODUCIBILITY_BAND_REL * q_coord
    per_arm = {}
    for arm in ('passive', 'price_taker'):
        costs = {}
        for start in ('cold', 'warm_from_certified', 'perturbed'):
            rec = arms.get(f'arm_{arm}_{start}')
            if rec is not None:
                costs[start] = rec['arm_cost']['gross_operational_cost']
        if costs:
            best_start = min(costs, key=costs.get)
            per_arm[arm] = {'q_by_start': costs, 'q_best': costs[best_start], 'best_start': best_start,
                            'multimodality_band_eur': max(costs.values()) - min(costs.values()),
                            'n_starts': len(costs),
                            'curtailment_mwh_day_weighted': {
                                s: arms[f'arm_{arm}_{s}']['phase_A']['evaluation']['curtailment']['totals']
                                for s in costs}}
    claim = None
    if gate is None or gate.get('status') != 'PASS':
        claim = {'computed': False, 'reason': 'common-Q gate has not PASSED (Addendum 49 ruling 3: the gate comes '
                                              'before any arm is reported)'}
    elif len(per_arm) == 2 and all(p['n_starts'] == 3 for p in per_arm.values()):
        best_arm = min(per_arm, key=lambda a: per_arm[a]['q_best'])
        benefit = per_arm[best_arm]['q_best'] - q_coord
        larger_band = max(per_arm[best_arm]['multimodality_band_eur'], band_coord)
        claim = {
            'computed': True, 'objective_convention': 'gross_operational_cost, settlement excluded',
            'q_min_over_starts_note': Q_MIN_OVER_STARTS_NOTE,
            'best_uncoordinated_arm': best_arm,
            'benefit_eur': benefit, 'benefit_relative': benefit / q_coord,
            'bands_eur': {'best_arm_multimodality': per_arm[best_arm]['multimodality_band_eur'],
                          'coordinated_reproducibility_0.011pct': band_coord,
                          'dso_band_step5': None},
            'larger_band_eur': larger_band,
            'dso_band_step5_note': 'the DSO band re-measured in Step 5 is not yet available; the claim must be '
                                   're-read against it when it is (Addendum 49 "Resolution")',
            'determinate': abs(benefit) > larger_band,
            'verdict': ('coordination beats the best uncoordinated arrangement' if benefit > larger_band else
                        ('the best uncoordinated arrangement beats coordination' if benefit < -larger_band else
                         'inside the band')),
            'decomposition': {
                'passive_minus_price_taker_eur': per_arm['passive']['q_best'] - per_arm['price_taker']['q_best'],
                'price_taker_minus_coordinated_eur': per_arm['price_taker']['q_best'] - q_coord,
                'passive_minus_coordinated_eur': per_arm['passive']['q_best'] - q_coord,
            },
        }
    else:
        claim = {'computed': False, 'reason': f'arm runs missing: {missing}'}
    value_independence = {}
    base = arms.get('arm_passive_cold')
    for run_id, rec in variants.items():
        if base is None or rec is None:
            continue
        value_independence[run_id] = {
            'dso_tie_breaker': rec['decision_tie_breaker']['dso'],
            'interface_schedule_max_abs_difference_vs_1': _schedule_difference(
                _keys_to_str(rec['phase_A']['dso_interface_schedule']),
                _keys_to_str(base['phase_A']['dso_interface_schedule'])),
            'q_difference_vs_1_eur': (rec['arm_cost']['gross_operational_cost']
                                      - base['phase_A']['evaluation']['gross_operational_cost'])}
    consistency = {run_id: {'max_abs_dv_dn_pu': (rec.get('phase_B_consistency') or {}).get('max_abs_dv_dn_pu'),
                            'trigger': ((rec.get('phase_B_consistency') or {}).get('reevaluation') or {}).get(
                                'trigger_sequential_pass'),
                            'pass_effect_eur': (rec.get('phase_C_sequential_pass') or {}).get('effect_on_q_eur')}
                   for run_id, rec in arms.items() if rec is not None}
    curtailment = curtailment_table(gate, arms, variants)
    capture = report_capture_check({'lambda_look': look, 'common_q_gate': gate, 'tso_coupling_check': coupling,
                                    **arms, **variants})
    guard = _check_guard(0, 'report end')
    result = provenance({
        'run': 'report', 'solve_profile_guard': guard, 'missing_inputs': missing,
        'report_capture_check': capture,
        'coordinated': {'q': q_coord, 'q_hex': float(q_coord).hex(),
                        'source': ('settled cycle-181 x = 0 cell d110bd1a5977df1e_x0 (not re-run; Addendum 53 Ruling 4 '
                                   '/ Addendum 54 Ruling 3): ' + COORDINATED['certified_gross_source']),
                        'reproducibility_band_eur': band_coord,
                        'settling_band_width_eur_reported_only': _load_json(COORDINATED['settling_decision']).get(
                            'band_width'),
                        'previous_reference_reported_only': COORDINATED_PREVIOUS_REFERENCE},
        'curtailment_table': curtailment,
        'common_q_gate_status': None if gate is None else gate.get('status'),
        'lambda_look_prediction_usable': None if look is None else look.get('prediction_usable'),
        'lambda_look_summary': None if look is None else look.get('summary_coordinated'),
        'tso_coupling_check': None if coupling is None else {
            'tn_cost_weighted_totals': coupling['tn_cost_weighted_totals'],
            'fixed_minus_penalty_tn_cost_weighted': coupling['fixed_minus_penalty_tn_cost_weighted'],
            'fixed_side_voltage_pin': coupling.get('fixed_side_voltage_pin'),
            'solve_summary': coupling['solve_summary']},
        'per_arm': per_arm, 'claim': claim, 'passive_tie_breaker_value_independence': value_independence,
        'consistency': consistency,
    })
    _write_json_once(os.path.join(run_dir, 'report.json'), result)
    return 0 if (claim.get('computed') and capture['all_present']) else 1


def _curtailment_row(totals):
    """One curtailment row from `uncoordinated_benchmark.curtailment_report` totals ({'TSO': .., 'DSO': ..})."""
    return {'eur_at_1_block_weighted': totals['TSO']['eur_at_1_block_weighted'] + totals['DSO']['eur_at_1_block_weighted'],
            'mwh_day_weighted': totals['TSO']['mwh_day_weighted'] + totals['DSO']['mwh_day_weighted'],
            'mwh_rep_day_sum': totals['TSO']['mwh_rep_day_sum'] + totals['DSO']['mwh_rep_day_sum'],
            'by_agent': totals}


def curtailment_table(gate, arms, variants):
    """W106 (TASKS.md Addendum 49, author requirement): curtailed energy per arm and start on the SAME weighting as
    the coordinated figure -- EUR at 1 EUR/MWh, block-weighted (years x days x discount), the convention of
    res_curtailment_definitional_at_weight_1 -- with raw MWh alongside (day-weighted years x days, undiscounted; and
    the plain sum over representative days). The coordinated row: the settled cycle-181 record figure, the same
    quantity recomputed from the persisted models by the common-Q gate (evaluation tie-breaker 0), and the W86
    cycle-132 record figure (589.18, reported only)."""
    coordinated = {'settled_cycle_181_record_eur_at_1_block_weighted':
                   COORDINATED['res_curtailment_definitional_at_weight_1'],
                   'previous_cycle_132_record_eur_at_1_block_weighted_reported_only':
                   COORDINATED_PREVIOUS_REFERENCE['res_curtailment_definitional_at_weight_1'],
                   'settled_cycle_181_recomputed_from_models': None}
    if gate is not None:
        coordinated['settled_cycle_181_recomputed_from_models'] = _curtailment_row(
            gate['evaluation_at_0']['curtailment']['totals'])
    rows = {}
    for run_id, rec in list(arms.items()) + list(variants.items()):
        if rec is not None:
            rows[run_id] = _curtailment_row(rec['phase_A']['evaluation']['curtailment']['totals'])
    return {'weighting': ('eur_at_1_block_weighted = res_curtailment_definitional_at_weight_1 convention (EUR at 1 '
                          'EUR/MWh, block-weighted: years x days x discount), i.e. weighted MWh-equivalent; raw: '
                          'mwh_day_weighted (years x days, undiscounted) and mwh_rep_day_sum'),
            'coordinated': coordinated, 'arms_and_variants_phase_A': rows}


def _dig(obj, path):
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return False, None
        obj = obj[key]
    return True, obj


def report_capture_check(outputs):
    """Every REPORT_CAPTURE_PATHS quantity present in the loaded stage outputs (`None` output = stage missing)."""
    present, absent = {}, []
    for quantity, (source, path) in REPORT_CAPTURE_PATHS.items():
        if source.endswith('*'):
            ids = [k for k in outputs if k.startswith(source[:-1])]
        else:
            ids = [source]
        for run_id in ids:
            ok, _value = _dig(outputs.get(run_id), path)
            present[f'{quantity}@{run_id}'] = ok
            if not ok:
                absent.append(f'{quantity}@{run_id}')
    return {'n_checked': len(present), 'absent': absent, 'all_present': not absent}


def _keys_to_str(schedule):
    return {str(n): {str(y): {str(d): v for d, v in days.items()} for y, days in years.items()}
            for n, years in schedule.items()}


# ======================================================================================================================
#  main
# ======================================================================================================================
def _run_id(args):
    if args.stage == 'lambda-look':
        return 'lambda_look'
    if args.stage == 'common-q-gate':
        return 'common_q_gate'
    if args.stage == 'tso-coupling-check':
        return 'tso_coupling_check'
    if args.stage == 'arm':
        return f'arm_{args.arm}_{args.start}'
    if args.stage == 'passive-tie-breaker':
        return 'passive_tie_breaker_' + {0.1: '0p1', 10.0: '10'}[args.value]
    if args.stage == 'report':
        return 'report'
    raise ValueError(args.stage)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--stage', required=True, choices=('lambda-look', 'common-q-gate', 'tso-coupling-check',
                                                           'arm', 'passive-tie-breaker', 'report'))
    parser.add_argument('--arm', choices=('passive', 'price_taker'))
    parser.add_argument('--start', choices=('cold', 'warm_from_certified', 'perturbed'))
    parser.add_argument('--value', type=float, choices=TIE_BREAKER['passive_value_independence_variants'])
    args = parser.parse_args(argv)
    if args.stage == 'arm' and (args.arm is None or args.start is None):
        parser.error('--stage arm needs --arm and --start')
    if args.stage != 'arm' and (args.arm is not None or args.start is not None):
        parser.error('--arm/--start only with --stage arm')
    if args.stage == 'passive-tie-breaker' and args.value is None:
        parser.error('--stage passive-tie-breaker needs --value')
    if args.stage != 'passive-tie-breaker' and args.value is not None:
        parser.error('--value only with --stage passive-tie-breaker')
    return args


def main(argv=None):
    args = parse_args(argv)
    run_id = _run_id(args)
    run_dir = os.path.join(_abs(OUT_ROOT_REL), run_id)
    failures = check_preconditions(run_dir)
    if failures:
        print(f'REFUSING TO RUN {run_id}:', *failures, sep='\n  ', flush=True)
        return 2
    bounded = args.stage in ('tso-coupling-check', 'arm', 'passive-tie-breaker')
    _install_guard(run_id, bounded)
    acquire_lock(run_id)
    try:
        os.makedirs(run_dir, exist_ok=False)
        _log(f'{STAGE} -- {run_id} (solve stage: {bounded}); output {os.path.relpath(run_dir, REPO)}')
        try:
            if args.stage == 'lambda-look':
                code = stage_lambda_look(run_dir)
            elif args.stage == 'common-q-gate':
                code = stage_common_q_gate(run_dir)
            elif args.stage == 'tso-coupling-check':
                code = stage_tso_coupling_check(run_dir, run_id)
            elif args.stage == 'arm':
                code = stage_arm(run_dir, run_id, arm=args.arm, start=args.start,
                                 dso_tie_breaker=TIE_BREAKER['decision'][f'{args.arm}_dso'], consistency=True)
            elif args.stage == 'passive-tie-breaker':
                code = stage_arm(run_dir, run_id, arm='passive', start='cold', dso_tie_breaker=float(args.value),
                                 consistency=False)
            else:
                code = stage_report(run_dir)
        except Exception as error:  # noqa: BLE001 -- recorded, then non-zero exit
            traceback.print_exc()
            _write_json_once(os.path.join(run_dir, 'failure.json'), provenance({
                'run': run_id, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
                'solve_profile_guard_counts': dict(_GUARD.counts) if _GUARD is not None else None,
                'record': getattr(error, 'record', None)}))
            code = 1
        _write_manifest(run_dir)
        _log(f'{run_id}: exit {code}; guard counts {dict(_GUARD.counts)}')
        return code
    finally:
        if _GUARD is not None:
            _GUARD.uninstall()
        release_lock()


if __name__ == '__main__':
    sys.exit(main())
