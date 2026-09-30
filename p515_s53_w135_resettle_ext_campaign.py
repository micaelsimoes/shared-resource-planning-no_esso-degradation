"""
P5.15 Addendum 58 Supplement, Planner task W135 -- the RE-SETTLING EXTENSION: claim group 3 (ageing E: six
model-variant arms of the ageing law on the current ESS parameters file 39106f93, minimum SoH 0.70 NOT overridden, at
the unit db77e154...) and the pb_y2025_n5 re-run under the settling rule v4 (7 cells). The 7 per-cell campaign
freezes, the frozen stage spec `frozen_s53_resettle_ext_spec_v1_<sha8>.json` (it EXTENDS the v4 stage spec
frozen_s53_resettle_spec_v4, which is in force for its 38 cells), the per-cell run (one cell per call, launch order
enforced, AFTER the v4 campaign's cell #38), and the zero-solve scorer for item E and the Phase B cell. BUILT IN W135;
RE-TARGETED TO CRITERION v4, CHECKED AND FROZEN IN W137 (Addendum 59: "W135's router patch in the same freeze"); NOTHING
IS RUN.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 58 and its Supplement (2026-09-29: ageing arms at minimum SoH 0.70,
option (b); per arm the year the floor binds beside the value; the 0.50-era ageing statements restated; the expert's
prediction; pb_y2025_n5 re-run under the new rule at the end of the priority queue); TASKS.md Addendum 58 section;
W134 (arm definitions, key, G9 replacement, floor-year capture, walls) as transcribed in Planner task W135; W132
(spec 139d1e62: the machinery reused here); W137 (Addendum 59 and its Supplement: criterion v4, the v4 campaign).

ORDER OF USE (W137, Addendum 59: the router patch in the v4 freeze):
  1. the prepared `p515_s53_w135_harness_router.patch` applied (router commit 1), then the v4 branch (router commit 2);
  2. the zero-solve checks (`p515_s53_w135_resettle_ext_checks.py`) run, output committed;
  3. --freeze-cells, commit; --freeze-spec, commit (both need the v4 stage spec frozen and committed, NOT the v4 #38);
  4. --run: every cell REFUSES until the v4 campaign's cell #38 (l_195156fa) has its results and manifest committed
     (`_hard_preconditions(..., require_v4_38=True)`); the dry run on e_c3_unit refuses on exactly that until then.
Every mode except --summarize refuses unless the harness is the pre-W135 harness plus the patch plus the v4 branch and
the v4 stage spec is frozen; --run also refuses until the v4 #38 is committed.

THE CELLS (`p515_s53_w135_resettle_ext_hooks.CELLS`, launch order CELL_ORDER):
  #1 e_c3_unit  #2 e_c2  #3 e_c4  #4 e_c2_calfade  #5 e_c3_midblock  #6 e_no_ageing   (item E, ungated, cap
     min(k0_run + 109, 300), criterion v4; the current configuration + the arm's model_variant)
  #7 pb_y2025_n5_v4 (item C, gated bitwise against 4a852725 through k0 = 110, cap 219; a NEW eval key and root; the
     W118 v2 run stays as it is)

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-cells                        ZERO SOLVES. The 7 campaign specs (write-once) in the W135 stage root.
  --freeze-spec                         ZERO SOLVES. The stage spec (write-once, named by its sha256): pins the 7
                                        committed campaign specs, the code, the checks output, the references, the
                                        predictions, and holds the EXACT launch command of every cell.
  --run --cell C --spec-sha256 S        THE RUN OF ONE CELL. Launch order enforced (every earlier W135 cell has
                                        results; the v4 #38 committed). Preconditions (the stage spec pins S for C; the
                                        zero-solve checks re-run inline except section M, which is in the committed
                                        output; the pre-launch assertion; the parent-side capture checklist; memory;
                                        solver; own-process check), then H.evaluate on the one entry, the gates, the
                                        cell report; results + manifest; the claim-completion point, if any.
  --run ... --preconditions-only        ZERO SOLVES. Every --run precondition, then STOP before the lock and the child.
  --summarize --after-cell C            ZERO SOLVES. The scorer over every cell with committed results up to C: the
                                        W117 claims (E and the pb Phase B claim), the restated 0.50-era statements,
                                        the predictions; write-once, named after C.

Exit codes: 0 done (every stopping gate holds); 1 any other gate / harness / guard / precondition failure; 3 every
stopping gate holds but the C2_calfade consistency check (G28) failed -- a recorded validation failed: STOP FOR THE
PLANNER before the next cell.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W135 re-settling extension launcher (never solves)').install()

import p515_s53_w132_resettle_v3_campaign as L132  # noqa: E402 -- generic gates / scorer (arms its guards)
import p515_s53_w135_resettle_ext_checks as K  # noqa: E402 -- the zero-solve checks (arms its guard)
import p515_s53_w135_resettle_ext_hooks as W  # noqa: E402
import p515_s46_ageing_mechanism as M46A  # noqa: E402 -- the committed AE formula `pv` (arms its own guard)
import p515_s53_w137_resettle_v4_campaign as L137  # noqa: E402 -- the v4 gates (arms its guards)
import p515_s53_w137_resettle_v4_hooks as V4  # noqa: E402
import settling_criterion_v4 as SC4  # noqa: E402
import gate_result_io as GRIO  # noqa: E402

H, L, X, W9, W98L, W101L, L118, W131 = L132.H, L132.L, L132.X, L132.W9, L132.W98L, L132.W101L, L132.L118, L132.W131
V = L132.V
R = V.R


def _dedupe_guards(pairs):
    seen, out = set(), []
    for name, guard in pairs:
        if id(guard) not in seen:
            seen.add(id(guard))
            out.append((name, guard))
    return tuple(out)


GUARDS = _dedupe_guards(tuple(K.GUARDS) + tuple(L137.GUARDS) + tuple(L132.GUARDS)
                        + (('s46_ageing_mechanism_imported', M46A.GUARD), ('w135_parent', PARENT_GUARD)))

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRINGS = ('p515_s53_w135_resettle_ext_campaign', 'p515_s53_w137_resettle_v4_campaign') \
    + L132.OWN_PROCESS_SUBSTRINGS
STAGE_TEXT = ('P5.15 Addendum 58 Supplement, W135 -- re-settling extension (7 cells): the ageing E arms as model '
              'variants of the ageing law on the current ESS parameters file (minimum SoH 0.70, not overridden), '
              'ungated first evaluations at the unit; pb_y2025_n5 replayed bitwise against its original record through '
              'k0 = 110; the certifying regime held after the run\'s first residual pass (AA off, tight tail on, rho '
              'frozen); settling rule v4 (no reset on a non-Optimal accepted solve; certification vetoed while a '
              'non-Optimal cycle lies in the last W cycles the test reads) until it certifies or the cap; W105 '
              'captures, t_sum, Q_cc, the IPOPT exit of every block, the SoH-floor sidecar and the terminal ageing '
              'trajectory')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = K.W135_ROOT_REL
SPEC_PREFIX = 'frozen_s53_resettle_ext_spec_v1_'
SPEC_SERIES = 'frozen_s53_resettle_ext_spec'
SPEC_VERSION = 1
EXTENDS_RELATION = ('extends: the v4 stage spec (frozen_s53_resettle_spec_v4) is in force, unchanged, for its 38 cells; '
                    'this series holds the 7 cells that spec does not (the ageing E arms and pb_y2025_n5, W134) and '
                    'reuses its criterion v4, the v4 state and W132\'s wrappers and scorer')


def extends():
    """The extended v4 stage spec: found by its series prefix (W.v4_stage_spec_rel: the one file in the v4 root whose
    name carries its own sha256 prefix); None path when it is not frozen yet."""
    rel, sha = W.v4_stage_spec_rel()
    return {'path': rel, 'sha256': sha, 'relation': EXTENDS_RELATION}
CAMPAIGN_IDS = {cell: f'{K.CAMPAIGN_ID_PREFIX}{cell}' for cell in W.CELL_ORDER}
CONCURRENCY = 1
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
TASKS = 'TASKS.md'
A28_REPORT = 'P5_15_ADDENDUM28_AGEING_REPORT.md'
A30_REPORT = 'P5_15_ADDENDUM30_PHASE_B_REPORT.md'
SOLVER_PATH = L132.SOLVER_PATH
PYTHON = L132.PYTHON
N_NETWORK_BLOCKS = 48
EXTRA_CLEAN_FILES = tuple(dict.fromkeys((SCRIPT_NAME, 'p515_s53_w135_resettle_ext_hooks.py',
                                         'p515_s53_w137_resettle_v4_campaign.py',
                                         'p515_s53_w135_resettle_ext_checks.py', K.PATCH_REL,
                                         'p515_s46_ageing_mechanism.py', 'p515_s46_variant_checks.py')
                                        + tuple(L132.EXTRA_CLEAN_FILES)))
CODE_PINNED = tuple(dict.fromkeys(K.CODE_PINNED_BY_CHECKS + (
    SCRIPT_NAME, 'p515_s53_w137_resettle_v4_campaign.py', 'p515_s53_w132_resettle_v3_campaign.py',
    'p515_s53_w118_resettle_campaign.py',
    'p515_s53_w131_prefreeze_diagnostics.py', 'p515_s53_w101_srp1_continuation_campaign.py',
    'p515_s53_w98_continuation_campaign.py', 'p515_s53_w98_continuation_hooks.py',
    'p515_s53_w86_tail_recert_campaign.py', 'p515_s53_w89_g6_final_attempt_reeval.py', 'p515_s46_ageing_mechanism.py')))
INLINE_SECTIONS = tuple((sid, fn) for sid, fn in K.SECTIONS if sid != 'M')   # M builds models: committed output only

# ---- verbatim text (checked against the committed files, whitespace-normalised, at every freeze) ----------------------
VERBATIM = {
    (BRIEF, 'supplement_option_b'): ('**Supplement (2026-09-29) — ageing arms at minimum SoH 0.70, option (b), '
                                     'overriding the Planner\'s (a).**'),
    (BRIEF, 'supplement_row'): ('The row reports, per arm, the year the floor binds (if any) beside the value; the '
                                'ageing statements measured at 0.50 (Addendum 30 era) are restated under the new row'),
    (BRIEF, 'supplement_prediction'): ('**Prediction:** the arms where the floor binds show a smaller value change per '
                                       'unit of cycle life than the 0.50-based elasticity implied, since end-of-life '
                                       'caps the value before cycle life does.'),
    (BRIEF, 'supplement_pb'): ('**pb_y2025_n5: re-run under the new rule at the end of the priority queue** (≈ 1.5 h), '
                               'so every Phase B certificate has the same strength and the flag leaves the table.'),
    (BRIEF, 'salvage_reporting_only'): 'Salvage is reporting-only and does not touch gross.',
    (BRIEF, 'ruling2_future_specs'): L132.VERBATIM[(BRIEF, 'ruling2_future_specs')],
    (TASKS, 'ageing_ruled'): '**Ageing arms RULED (Add. 58 supplement): option (b), minimum SoH 0.70**',
    (TASKS, 'pb_end_of_queue'): '**pb_y2025_n5: re-run under v3 at the END of the priority queue**',
    (BRIEF, 'a59_router_in_the_v4_freeze'): ("W135's router patch in the same freeze so the ageing and pb cells need no "
                                             "second one"),
    (BRIEF, 'a59_order'): ('v4 freeze → cell 1 re-run → resume in priority order (≈ 57 h v3 cells, 9 h ageing and '
                           'pb)'),
    (A28_REPORT, 'title_sign'): 'the ageing convention does not decide the sign',
    (A28_REPORT, 'none_pays'): 'Under none of them does the smallest unit pay determinately',
    (A28_REPORT, 'floor_050'): 'soh_min is 0.50 in every variant, and no floor row was active.',
    (A28_REPORT, 'elasticity'): 'Its elasticity is 0.60–0.62 over the two resolvable variants.',
    (A28_REPORT, 'midblock'): 'The mid-block variant moves almost nothing** (×1.009, against the Advisor\'s ~9 % estimate).',
    (A28_REPORT, 'break_even'): ('**Break-even needs no ageing at all.** Value then equals I within 485 €. Any physical '
                                 'fade leaves the smallest unit negative or indeterminate.'),
    (A28_REPORT, 'band_c2'): 'Under C2 the smallest unit needs ×1.08 more value; under no ageing it breaks even.',
    (A30_REPORT, 'band'): ('**Ageing-sensitivity band** (C3-era batch, labelled as such): the smallest unit needs ×1.21 '
                           'more value. The measured multipliers were 1.13 (C2), 1.06 (C4), 1.05 (C2 + fade), 1.01 '
                           '(mid-block) and 1.22 (no ageing, which only breaks even).'),
    (A30_REPORT, 'elasticity'): ('**Elasticity mechanism:** value rises with available energy at elasticity ≈ 0.6 at '
                                 'fixed power'),
    (A30_REPORT, 'floor_2035'): 'The 0.70 floor binds in the 2035 block at every point and at C\\*',
}

# ---- the references (committed; pinned at freeze) --------------------------------------------------------------------
REFERENCES = L132.REFERENCES                   # x0 7aa017f0 (d110bd1a, Q181), unit bd504ecf (3f084f2f, Q172), F2 pair
X0_REF = '7aa017f0'
UNIT_REF_EVAL_DIR = K.UNIT_SETTLED_EVAL_DIR    # 3f084f2f: the C2_calfade consistency reference
AGEING_MECHANISM = K.AGEING_MECHANISM
W117 = K.K132.W117
ARMS_WITH_K_DIFFERENT_FROM_C3 = ('C2', 'C4', 'C2_calfade')
AGED_ARMS = ('C3_unit', 'C2', 'C4', 'C2_calfade', 'C3_midblock')
E_REPORT_POINT_CELL = 'e_no_ageing'
GAP_REFUSED_LABEL = L132.GAP_REFUSED_LABEL
PF_SLOPE_WINDOW = L132.PF_SLOPE_WINDOW
ORIG_KEY_TO_CELL = {W.CELLS[c]['orig_eval_key']: c for c in W.CELL_ORDER}
PB_CLAIM_ID = 'C:y2025__n5_p0.25_e0.5'

# ---- the frozen definitions --------------------------------------------------------------------------------------------
DEFINITIONS = {
    'objective_convention': L132.DEFINITIONS['objective_convention'],
    'per_cell': dict(L132.DEFINITIONS['per_cell'], **{
        'arm': 'the E arm (W134 arm definitions table); None for pb',
        'model_variant': 'the arm\'s model_variant, applied in the child and read back (G9 replacement)',
        'floor_year': ('W.floor_year_reading at the end cycle (k* or the cap): the first block year whose node-7 floor '
                       'row (cohort 2025) is active in soh_floor_sidecar_baseline.jsonl (active = |SoH - soh_min| <= '
                       '1e-6); None when no floor row is active'),
        'AE': ('p515_s46_ageing_mechanism.pv(SoH_used per block): sum_y w_y SoH_used[y] / sum_y w_y, w_y = 1 / 1.02**(y '
               '- 2025); SoH_used = soh_used_for_available_energy of node 7, cohort 2025, from the record\'s '
               'ageing_trajectory_terminal'),
        'EFC': 'p515_s46_ageing_mechanism.pv(efc_per_day per block), the same source'}),
    'uncertified_form': L132.DEFINITIONS['uncertified_form'],
    'claims': dict(L132.DEFINITIONS['claims'], source=(
        'W117 claims (w117_triage_recompute.json, 3b2e76de): the 11 item-E claims and the Phase B claim '
        'C:y2025__n5_p0.25_e0.5, each with its W117 definition (form, I_ref, I_other, net_of_salvage); the cells are '
        'this campaign\'s (the original eval keys mapped to the W135 cells) and the settled x0 reference (d110bd1a for '
        '7aa017f0)')),
    'value': 'value_a = Q(0) - Q_a, Q(0) = the settled x0 Q181 (d110bd1a); value_Qcc likewise with Q_cc = Q + t_sum',
    'weighted_resolution': (
        'Worker generalisation of W132\'s rule to a difference sum_i c_i Q_i over three cells (the expert prediction): '
        'all certified -> resolution = sum_i |c_i| band_i (W132\'s sum of the two band widths when |c_i| = 1), gross '
        'verdict, Q_cc beside; any uncertified ungated cell -> indeterminate (slack undefined); an uncertified gated '
        'cell -> bar = 3 x max(|c_i| gap_i, |c_i| slack_i) over the uncertified cells, both terms (W132\'s form when '
        '|c_i| = 1). FOR PLANNER CONFIRMATION'),
}

# ---- the restated 0.50-era statements (item E), each with its source ---------------------------------------------------
RESTATED_STATEMENTS = {
    'S1_x_c3_multipliers': {
        'statement_050': ('x C3 measured (value_a / value_C3): C2 1.126, C4 1.058, C2 + phi_cal 0.985 1.046, C3 with '
                          'mid-block health 1.009, no ageing 1.216 (A28 table); rounded 1.13 / 1.06 / 1.05 / 1.01 / 1.22 '
                          '(A30 manuscript item 1)'),
        'source': [f'{A28_REPORT} section 2 table (82bda45a)', f'{A30_REPORT} section 6 item 1 (d04caba7)',
                   'ageing_mechanism.json b5eca2a2 table[].value_over_C3'],
        'restated_as': ('value_a / value_C3unit at 0.70, value = Q(0) - Q (gross; Q_cc beside); the vs_C3 claim of the '
                        'arm (W117 E:<arm>:vs_C3, W132 resolution rule) says whether the ratio differs from 1 '
                        'determinately')},
    'S2_I_over_value_sensitivity_band': {
        'statement_050': ('the smallest unit needs x1.21 more value (C3-era, I / value_C3); under C2 x1.08; under no ageing '
                          'it breaks even'),
        'source': [f'{A30_REPORT} section 6 item 1', f'{A28_REPORT} section 5 question 3',
                   'ageing_mechanism.json b5eca2a2 (value_eur) with I = 317,957.0085 (W117 I_other of the unit)'],
        'restated_as': 'I / value_a per arm at 0.70; beside it the arm\'s value - I claim verdict'},
    'S3_convention_does_not_decide_the_sign': {
        'statement_050': ('the ageing convention does not decide the sign: under none of the variants does the smallest '
                          'unit pay determinately'),
        'source': [f'{A28_REPORT} title and section 1 item 1'],
        'restated_as': ('W117\'s 11 E claims scored on W132\'s resolution rule; per arm the sign of value - I read as '
                        '+ (determinate positive), - (determinate negative), 0 (within resolution / indeterminate). The '
                        'statement HOLDS (restated) iff no arm is + (A28\'s sentence); "the convention decides the sign" '
                        'iff one arm is + and another - (reported beside). Worker operationalisation, FOR PLANNER '
                        'CONFIRMATION')},
    'S4_break_even_needs_no_ageing_at_all': {
        'statement_050': ('Break-even needs no ageing at all. Value then equals I within 485 EUR. Any physical fade '
                          'leaves the smallest unit negative or indeterminate.'),
        'source': [f'{A28_REPORT} section 2, readings'],
        'restated_as': ('no_ageing: value - I within resolution (0); every aged arm (C3 unit, C2, C4, C2_calfade, '
                        'C3_midblock): - or 0, never +. Both components reported')},
    'S5_elasticity_to_available_energy': {
        'statement_050': ('value is sub-proportional to available energy: elasticity ln(value ratio) / ln(AE ratio) 0.60 '
                          '(C2) - 0.62 (no ageing) over the two resolvable variants; approx 0.6 at fixed power'),
        'source': [f'{A28_REPORT} section 2 mechanism table', f'{A30_REPORT} section 6 item 2',
                   'ageing_mechanism.json b5eca2a2 formulas.elasticity; p515_s46_ageing_mechanism.py'],
        'restated_as': ('eps_AE_a = ln(value_a / value_C3) / ln(AE_a / AE_C3) at 0.70 (AE by the committed formula); an '
                        'arm is resolvable iff its vs_C3 claim is determinate (A28\'s asterisk rule restated)')},
    'S6_midblock_x1009': {
        'statement_050': 'the mid-block variant moves almost nothing (x1.009, against the Advisor\'s ~9 % estimate)',
        'source': [f'{A28_REPORT} section 2, readings', 'ageing_mechanism.json b5eca2a2 (C3_midblock value_over_C3)'],
        'restated_as': 'value_C3midblock / value_C3unit at 0.70 with the E:n7_4h_e1_C3_midblock:vs_C3 verdict'},
    'S7_floor_year_per_arm': {
        'statement_050': ('soh_min is 0.50 in every variant, and no floor row was active (A28); under the baseline the '
                          '0.70 floor binds in the 2035 block at every point (A30)'),
        'source': [f'{A28_REPORT} section 2 (Floor)', f'{A30_REPORT} section 1 item 2'],
        'restated_as': 'per arm the floor year (definitions.per_cell.floor_year) beside the value'},
}

# ---- predictions, recorded BEFORE any run (each with its source) --------------------------------------------------------
PREDICTIONS = {
    'expert_addendum58_supplement': {
        'statement': ('the arms where the floor binds show a smaller value change per unit of cycle life than the '
                      '0.50-based elasticity implied, since end-of-life caps the value before cycle life does'),
        'source': f'{BRIEF} Addendum 58 Supplement (2026-09-29), expert',
        'operationalisation': (
            'as the Planner accepted it (task W135): eps_k,a = ln(v_a / v_C3) / ln(k_a / k_C3), k = N * D / (-ln R) '
            '(N, D the file\'s calibration), for the aged arms whose k differs from C3: C2, C4, C2_calfade; v = value '
            '(gross; Q_cc beside). For each such arm WHOSE FLOOR BINDS (floor_year not None), the prediction HOLDS iff '
            'eps_k^0.70 < eps_k^0.50; an arm whose floor does not bind is not scored'),
        'references_050': {'C2': 0.105, 'C4': 0.085, 'C2_calfade': 0.039,
                           'source': ('ageing_mechanism.json b5eca2a2: ln(value_over_C3) / ln(k_a / k_C3), k_C3 = '
                                      '11,541.56, k_C2 = 35,851.36, k_C4 = 22,429.3 (recomputed at every freeze and '
                                      'checked to round to these)')},
        'determinacy': ('W132\'s resolution rule on the difference, generalised (definitions.weighted_resolution): '
                        'eps^0.70 < eps^0.50 <=> d = rho_a (Q0 - Q_C3) - (Q0 - Q_a) > 0 with rho_a = (k_a / '
                        'k_C3)**eps^0.50 (ln(k_a / k_C3) > 0 for all three arms); d = Q_a - rho_a Q_C3 - (1 - rho_a) Q0, '
                        'coefficients (1, -rho_a, -(1 - rho_a)) on (arm, C3 unit, x0)'),
        'weak_testability': ('the 0.50 references are themselves PENDING under W117: every E:<arm>:vs_C3 difference at '
                             '0.50 was within its resolution (W117 verdict_R1 "pending" on all 5 vs_C3 claims; A28 marks '
                             'C4, C2_calfade and mid-block "not resolvable"), so eps^0.50 has no resolved value and the '
                             'comparison is weakly testable; recorded, not a reason to skip the score')},
    'c2_calfade_consistency_planner': {
        'statement': ('on the current file C2_calfade IS the baseline (C2 calibration 0.80, phi_cal 0.985, \'end\', on, '
                      'soh_min 0.70): its run must reproduce the settled unit 3f084f2f\'s trajectory bitwise through '
                      'that run\'s certifying cycle 172, or explain the first difference. This validates the variant '
                      'plumbing'),
        'source': 'Planner task W135',
        'criterion': ('Q (per_cycle_record gross_operational_cost) at every cycle 1..172 equals 3f084f2f\'s EXACTLY '
                      '(W.trajectory_equality, JSON text of the float); every other shared per_cycle_record field '
                      'compared the same way, report-only; gate G28 (e_c2_calfade only); a failure stops the campaign '
                      'for the Planner (exit 3)'),
        'expected_equality_reasoning': [
            'MODEL: apply_model_variant writes eol 0.80 and phi 0.985 -- the values the file already loads -- and '
            're-applies production\'s EnergyStorageAgeingParameters.apply_to, which recomputes cl_eff = k from the '
            'same (N, D, R) floats: every ESS constant bit-identical; the two switches are set to their defaults '
            '(\'end\', True); soh_min is not a variant key. Checks M assert the freshly built ESSO models of the '
            'variant path and the baseline path are identical row by row (zero solves) before any run',
            'CONFIGURATION HOOK: the baseline path builds one probe set for the ESS-ageing readback; the variant path '
            'builds one probe set for the variant readback (the ESS-ageing readback is skipped); both discard the '
            'probes. No run model is touched',
            'IDENTITY: the eval key, eval dir and working-dir ids differ (the base key carries the variant; the '
            'resettle key wraps it) -- names only; W118 / W132 replayed originals bitwise under new keys',
            'HOLDS: the W101 unit run held from 113 (its N = 112); this run holds from k0_run + 1 = 104 if its first '
            'residual pass is 103. On 104..112 the unit\'s natural regime already was the held regime (AA off by '
            'production, tail passed active, rho frozen, every cycle a residual pass; checks O), and the AA hold '
            'passes a copy of production\'s Boyd metrics with all_boyd_pass already True -- so the holds change no value '
            'there; from 113 both runs hold',
            'STOP: settling rule v4 here (W137), v1 in W101; the rule reads the trajectory, it does not change it. W137 '
            'checks R: v4 on the unit\'s record (no non-Optimal cycle, W131) certifies at 172, and the dynamic cap '
            'min(103 + 109, 300) = 212 '
            'equals W101\'s N + 100 = 212',
            'SALVAGE: the health basis NORMALIZED_ABOVE_MINIMUM_SOH reads soh_min 0.70 from the file in both paths; '
            'salvage is reporting-only and outside gross (Addendum 58 Supplement)',
            'WHERE A DIFFERENCE MUST ARISE if one appears: (i) at cycle 104..112 with a hold_changed_value flag in the '
            'cycle line (the held regime differing from the natural one) -- a hold effect, not the variant; (ii) at '
            'cycle 1 with an identical model -- nondeterminism or a working-dir effect (then the W118 precedent is '
            'contradicted); (iii) at cycle 1 with a model difference -- impossible if checks M held; the diagnosis is '
            'recorded beside the first difference (G28 detail)']},
    'advisor_secondary': {'statement': None,
                          'note': ('NONE EXISTS: no Advisor prediction for the ageing arms at 0.70 or for the pb re-run '
                                   'was transcribed (TASKS.md Addendum 58 section and the brief\'s Addendum 58 '
                                   'Supplement searched); its absence is recorded'),
                          'source': 'Planner task W135 ("Advisor secondary: none exists. Record its absence")'},
    'walls': {'statement': 'E approx 7.7 h expected, 9.8 h worst (6 cells); pb approx 1.2 h',
              'source': 'W134, as transcribed in Planner task W135 (this spec\'s own estimate is expected_wall_time)'},
    'per_claim_numeric_predictions': {'statement': None,
                                      'note': ('NONE transcribed for the E claims or the pb claim beyond the expert\'s '
                                               'and the Planner\'s above (Planner task W135 names no other)'),
                                      'source': 'Worker search record (W135)'},
}


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


_read_jsonl = L132._read_jsonl
_committed_clean = L132._committed_clean


def guards_verify():
    return {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in GUARDS}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    g = guards_verify()
    _log(f'[W135] guards {g} {extra_msg}')
    for _n, guard in reversed(GUARDS):
        guard.uninstall()
    sys.exit(code if _guards_ok(g) else 1)


def _own_process_alive():
    excluded = {str(p) for p in H._ancestor_pids()}
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    hits = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] not in excluded and any(s in parts[1] for s in OWN_PROCESS_SUBSTRINGS):
            hits.append(line.strip())
    return hits


def _write_once_text(rel, text):
    with open(_abs(rel), 'x') as handle:
        handle.write(text)


def _norm(text):
    return ' '.join(text.split())


# ======================================================================================================================
#  configuration, entries, keys
# ======================================================================================================================
def configuration(cell):
    cfg = K.configuration_now()
    c = W.CELLS[cell]
    return {'name': ('W135 SRP1 RE-SETTLING EXTENSION (Addendum 58 Supplement) -- the current production configuration: '
                     'the case file (AA keep_memory declared), the ESS ageing baseline ' + cfg['ess_ageing_baseline_label']
                     + ' declared (minimum SoH 0.70, NOT overridden), the convergence-depth tight tail DECLARED ENABLED '
                     '(compl_inf_tol 1e-6), post-certification: persist the certified TSO/DSO models only'),
            'arm_label': 's39_D', 'overrides': {},
            'case_file_anderson_acceleration': cfg['case_file_anderson_acceleration'],
            'ess_ageing_baseline': cfg['ess_ageing_baseline'],
            'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'],
            'convergence_depth_tail': cfg['convergence_depth_tail'],
            'note': ('the entry adds settling_resettle (W135 schema; keyed)'
                     + (f"; model_variant = the {c['arm']} arm (keyed through the base key; MODEL VARIANT label at spec "
                        f"and entry level)" if c['arm'] else '')
                     + '; cap: gated N_old + 100, ungated 300 (the rule stops at min(k0_run + 109, 300)); '
                       'persist_certified_models as the W101 / W118 / W132 cells; no option (b); concurrency 1')}


def orig_entry(cell):
    return K.spec_entry(cell)[1]


def entries(cell):
    e = orig_entry(cell)
    nodes = {int(k): tuple(map(float, v)) for k, v in e['canonical']['nodes'].items()}
    opts = {'investment_year': e['canonical']['investment_year'],
            'post_certification': {'persist_certified_models': True, 'hull_polish': False, 'reference': None},
            'settling_resettle': W.declaration_for(cell)}
    if W.CELLS[cell]['arm'] is not None:
        opts['model_variant'] = W.arm_variant(cell)
    return [(cell, nodes, opts)]


def expected_keys(cell):
    key, overrides = K.candidate_of(cell)
    kw = K.key_kwargs(cell)
    spec, e = K.spec_entry(cell)
    ocfg = spec['configuration']
    base = H.evaluation_key(key, overrides, **kw)
    rkey = H.evaluation_key(key, overrides, settling_resettle=W.declaration_for(cell), **kw)
    base_orig = H.evaluation_key(e['key'], e['overrides'], case_file_aa=ocfg.get('case_file_anderson_acceleration'),
                                 model_variant=e.get('model_variant'),
                                 ess_ageing_baseline=ocfg.get('ess_ageing_baseline'),
                                 flex_price_multiplier=e.get('flex_price_multiplier'),
                                 convergence_depth_tail=ocfg.get('convergence_depth_tail'))
    return {'base_key_current_configuration': base, 'resettle_key': rkey,
            'base_key_original_configuration': base_orig, 'original_eval_key': e['eval_key']}


def campaign_root_rel(cell):
    return os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_IDS[cell]}')


def campaign_root(cell):
    return _abs(campaign_root_rel(cell))


def pre_launch_assertion(cell, spec=None):
    """The resettle key, recomputed now, is EXACTLY the expected one (and a frozen spec's entry carries it); it appears in
    no committed campaign spec OUTSIDE the W135 stage root (the rule: a pre-run check that scans committed artefacts
    excludes the run's own); the original configuration's key reproduces the original eval key."""
    k = expected_keys(cell)
    committed = L.committed_eval_keys(exclude_roots=(ROOT_REL,))
    e = orig_entry(cell)
    eval_dir_name = H.eval_dir_name(k['resettle_key'], cell)
    ids = H.eval_ids(CAMPAIGN_IDS[cell], k['resettle_key'])
    work = L._work_dir()
    entry = next((x for x in (spec or {}).get('candidates') or [] if x['label'] == cell), None)
    parts = {
        'base_key_original_configuration_equals_original_eval_key': k['base_key_original_configuration']
        == e['eval_key'] == W.CELLS[cell]['orig_eval_key'],
        'resettle_key_differs_from_original_and_base': k['resettle_key'] not in (e['eval_key'],
                                                                                 k['base_key_current_configuration']),
        'resettle_key_absent_from_committed_specs_outside_w135_root': k['resettle_key'] not in committed,
        'campaign_root_differs_from_original': os.path.abspath(campaign_root(cell)) != os.path.abspath(
            _abs(W.CELLS[cell]['orig_root'])),
        'eval_dir_name_differs_from_original': eval_dir_name != e['eval_dir'],
        'working_dir_ids_differ_from_original': not (set(ids.values()) & set(e['working_dir_ids'].values())),
        'working_dirs_absent': not any(os.path.exists(os.path.join(work, i)) for i in ids.values()),
    }
    if spec is not None:
        parts['frozen_entry_equals_recomputed'] = (entry is not None and entry.get('eval_key') == k['resettle_key']
                                                   and entry.get('eval_dir') == eval_dir_name
                                                   and entry.get('working_dir_ids') == ids)
    if not parts['resettle_key_absent_from_committed_specs_outside_w135_root']:
        parts_detail = {'holders_outside': committed.get(k['resettle_key'])}
    else:
        parts_detail = {}
    return {'holds': all(parts.values()), 'parts': parts, **k, 'eval_dir_name': eval_dir_name, 'working_dir_ids': ids,
            'n_committed_keys_scanned': len(committed), 'excluded_roots': [ROOT_REL], **parts_detail}


def validate_campaign_spec(cell, spec):
    ref = _load(L132.W101_X0_SPEC_REL)          # the W101 x0 cell: the same C2 + tight-tail configuration
    want = configuration(cell)
    cfg = spec['configuration']
    ents = spec['candidates']
    e = ents[0] if len(ents) == 1 else {}
    oe = orig_entry(cell)
    mv = W.arm_variant(cell)
    checks = {f'configuration:{k}': cfg.get(k) == want[k] for k in
              ('arm_label', 'overrides', 'case_file_anderson_acceleration', 'ess_ageing_baseline',
               'ess_ageing_baseline_label', 'convergence_depth_tail')}
    checks.update({
        'configuration:case_file_sha256': cfg.get('case_file_sha256') == K.K132.CASE_FILE_SHA256,
        'configuration:ess_params_file_is_the_pinned_file': (cfg.get('ess_params_file') or {}).get('sha256')
        == W.ESS_PARAMS_SHA256,
        'configuration:minimum_soh_070_not_overridden': (cfg.get('ess_ageing_baseline') or {}).get('minimum_soh')
        == W.FLOOR_ROW_LOWER_EXPECTED,
        'configuration:as_the_w101_x0_cell': all(cfg.get(k) == ref['configuration'].get(k) for k in (
            'arm_label', 'overrides', 'case_file_anderson_acceleration', 'ess_ageing_baseline',
            'ess_ageing_baseline_label', 'convergence_depth_tail', 'apply_rho', 'full_diagnostics_in_rows',
            'case_file_sha256')),
        'one_entry': len(ents) == 1, 'entry_label': e.get('label') == cell,
        'entry_canonical_and_key_as_original': e.get('canonical') == oe['canonical'] and e.get('key') == oe['key'],
        'entry_overrides_empty': e.get('overrides') == {},
        'entry_post_certification_persist_only': e.get('post_certification') == {
            'persist_certified_models': True, 'hull_polish': False, 'reference': None},
        'entry_no_flex_multiplier': 'flex_price_multiplier' not in e,
        'entry_resettle_is_the_w135_declaration': e.get('settling_resettle') == W.declaration_for(cell),
        'entry_model_variant_is_the_arm': (json.dumps(e.get('model_variant'), sort_keys=True)
                                           == json.dumps(mv, sort_keys=True)),
        'variant_label_at_spec_and_entry_iff_variant': (
            (mv is None and 'model_variant_label' not in e and 'model_variant_label' not in spec)
            or (mv is not None and e.get('model_variant_label') == H.MODEL_VARIANT_LABEL
                and spec.get('model_variant_label') == H.MODEL_VARIANT_LABEL)),
        'entry_has_no_other_continuation': not any(x in e for x in ('settling_continuation', 'certification_continuation',
                                                                    'settling_extension', 'release_solution_bookkeeping',
                                                                    'interface_deviation_premium')),
        'no_early_stop_anywhere_in_the_entry': 'early_stop' not in json.dumps(e),
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_IDS[cell],
        'cap_is_the_cell_cap': spec.get('cap') == W.spec_cap(cell),
        'concurrency_1': spec.get('concurrency') == CONCURRENCY,
        'required_consecutive_cycles_10': spec.get('required_consecutive_cycles') == 10,
        'bar_window_as_w101': spec.get('bar_window_cycles') == ref['bar_window_cycles'],
        'thread_caps_as_w101': spec.get('thread_caps') == ref['thread_caps'],
        'interpreter_as_w101': spec.get('interpreter') == ref['interpreter'],
        'solver_path_as_w101': (spec.get('nlp_solver_path_env') or {}).get('NLP_SOLVER_PATH') == SOLVER_PATH,
        'script_recorded': (spec.get('extra') or {}).get('campaign_script') == SCRIPT_NAME,
        'cell_recorded': (spec.get('extra') or {}).get('cell') == cell,
    })
    return checks


def parent_capture_checklist(cell, spec):
    """The child's capture checklist (W.assert_resettle_preconditions), asserted in the PARENT too (before the lock)."""
    decl = spec['candidates'][0]['settling_resettle']
    tail_on = bool((spec['configuration'].get('convergence_depth_tail') or {}).get('enabled'))
    aa_on = bool((spec['configuration'].get('case_file_anderson_acceleration') or {}).get('enabled'))
    try:
        checks = W.assert_resettle_preconditions(decl, spec, {'tail_enabled_for_this_run': tail_on}, aa_on)
        return True, checks
    except Exception as error:  # noqa: BLE001
        return False, {'error': f'{type(error).__name__}: {error}'}


def harness_routed():
    """The harness is the pre-W135 harness plus the prepared patch plus the v4 branch (W137; sha256 pinned) and routes a
    W135 declaration here."""
    sha = H.sha256_file(H.HARNESS_PATH)
    try:
        routed = H.resettle_hooks_module(W.declaration_for(W.CELL_ORDER[0])) is W
    except Exception:  # noqa: BLE001
        routed = False
    return {'ok': bool(sha == K.HARNESS_POST_V4_SHA256 and routed), 'harness_sha256': sha,
            'expected_post_v4_sha256': K.HARNESS_POST_V4_SHA256,
            'post_patch_sha256_router_commit_1': K.HARNESS_POST_PATCH_SHA256, 'routes_w135_declarations_here': routed}


# ======================================================================================================================
#  walls and the claim-completion points
# ======================================================================================================================
def wall_time_estimate():
    """Per cell. E: s = the settled unit 3f084f2f's mean per-cycle wall (the same candidate, the current configuration,
    concurrency 1: W101); expected cycles = its k* 172 (proxy: the arms' own k* is unknown), worst = min(103 + 109,
    300) = 212 (its first residual pass as the proxy for k0_run). pb: s = the W118 r2 pb_y2025_n5 mean per-cycle wall
    (the same candidate and configuration, concurrency 1); expected = its 167 cycles (proxy), worst = the cap 219. Plus
    the unit campaign's measured overhead (campaign wall - sum of the iteration walls) and 240 s for the inline checks.
    W134's transcribed walls beside."""
    unit_walls = L132._iteration_walls(UNIT_REF_EVAL_DIR)
    unit_root = os.path.dirname(os.path.dirname(UNIT_REF_EVAL_DIR))
    unit_campaign_s = _load(os.path.join(unit_root, RESULTS_FILE))['wall_clock_s']
    overhead = (unit_campaign_s - sum(unit_walls)) + 240.0
    pb_eval = os.path.join(K.K132.W118_ROOT_REL, K.K132.W118_CELL_DIRS['pb_y2025_n5'])
    pb_walls = L132._iteration_walls(pb_eval)
    unit_k0 = K.UNIT_SETTLED['first_residual_pass']
    per, tot_exp, tot_worst = {}, 0.0, 0.0
    for cell in W.CELL_ORDER:
        c = W.CELLS[cell]
        if c['gated']:
            s = sum(pb_walls) / len(pb_walls)
            exp_cycles, cap = len(pb_walls), W.spec_cap(cell)
            basis = f'W118 r2 pb_y2025_n5 ({pb_eval})'
        else:
            s = sum(unit_walls) / len(unit_walls)
            exp_cycles, cap = K.UNIT_SETTLED['k_star'], min(unit_k0 + W.CAP_AFTER_K0, c['cap_ceiling'])
            basis = f'W101 settled unit 3f084f2f ({UNIT_REF_EVAL_DIR})'
        e_s, w_s = exp_cycles * s + overhead, cap * s + overhead
        per[cell] = {'basis': basis, 's_per_cycle_estimate': s, 'expected_cycles': exp_cycles, 'cap_cycles': cap,
                     'expected_h': e_s / 3600.0, 'worst_case_h': w_s / 3600.0}
        tot_exp += e_s
        tot_worst += w_s
    e_exp = sum(per[c]['expected_h'] for c in W.E_CELLS)
    e_worst = sum(per[c]['worst_case_h'] for c in W.E_CELLS)
    return {'basis': wall_time_estimate.__doc__, 'overhead_s_per_cell': overhead, 'per_cell': per,
            'total_expected_h': tot_exp / 3600.0, 'total_worst_case_h': tot_worst / 3600.0,
            'e_cells_expected_h': e_exp, 'e_cells_worst_case_h': e_worst,
            'w134_transcribed': PREDICTIONS['walls']['statement']}


def claims_dataset():
    """The W117 claims this extension scores: the 11 item-E claims and the Phase B claim of pb_y2025_n5, each with its
    W117 definition; the original eval keys mapped to the W135 cells, 7aa017f0 to the settled x0 reference."""
    w117, sha = K.K132._pinned_json(W117)
    out, skipped = [], []
    for cl in w117['claims']:
        if cl['item'] != 'E' and cl['claim_id'] != PB_CLAIM_ID:
            continue
        cells, ok = {}, True
        for side in ('ref', 'other'):
            key = cl[side]['eval_key']
            if key in ORIG_KEY_TO_CELL:
                cells[side] = {'source': 'w135', 'cell': ORIG_KEY_TO_CELL[key]}
            elif key[:8] == X0_REF:
                cells[side] = {'source': 'reference', 'ref': X0_REF}
            else:
                ok = False
        if not ok:
            skipped.append(cl['claim_id'])
            continue
        out.append({'claim_id': cl['claim_id'], 'item': cl['item'], 'statement': cl['statement'],
                    'claim_type': cl['claim_type'], 'form': cl['form'], 'net_of_salvage': cl['net_of_salvage'],
                    'I_ref': cl['I_ref'], 'I_other': cl['I_other'], 'I_source': cl['I_source'],
                    'ref_label': cl['ref']['label'], 'other_label': cl['other']['label'],
                    'ref_eval_key_w117': cl['ref']['eval_key'], 'other_eval_key_w117': cl['other']['eval_key'],
                    'ref': cells['ref'], 'other': cells['other'], 'old_verdict_R1_w117': cl['verdict_R1'],
                    'old_d_Q_w117': cl['d_Q'], 'old_d_Qcc_w117': cl['d_Qcc'], 'note': cl.get('note')})
    return {'source': W117['path'], 'sha256': sha, 'commit': W117['commit'], 'claims': out, 'n_claims': len(out),
            'skipped_not_in_the_set': skipped}


def completion_points(dataset):
    pos = {c: i for i, c in enumerate(W.CELL_ORDER)}
    per_claim = {}
    for cl in dataset['claims']:
        cells = [s['cell'] for s in (cl['ref'], cl['other']) if s['source'] == 'w135']
        per_claim[cl['claim_id']] = {'item': cl['item'], 'cells': cells,
                                     'complete_after': (max(cells, key=pos.get) if cells else None)}
    items = {}
    for v in per_claim.values():
        if v['complete_after'] is None:
            continue
        cur = items.get(v['item'])
        if cur is None or pos[v['complete_after']] > pos[cur]:
            items[v['item']] = v['complete_after']
    points = {}
    for item, cell in items.items():
        points.setdefault(cell, []).append(item)
    ordered = [{'after_cell_index_1_based': pos[c] + 1, 'after_cell': c, 'items_complete': sorted(points[c])}
               for c in sorted(points, key=pos.get)]
    return {'rule': ('a claim is complete after the last of its W135 cells in the launch order; an item after the last '
                     'cell of any of its claims (Addendum 58: reports at each claim\'s completion)'),
            'report_points': ordered, 'per_item': {k: {'after_cell': v, 'index_1_based': pos[v] + 1}
                                                   for k, v in items.items()},
            'per_claim': per_claim}


def launch_command(cell, spec_sha, preconditions_only=False):
    log = f'run_{cell}_launch.log' if not preconditions_only else f'run_{cell}_preconditions_only.log'
    return ('cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && set -o noclobber && '
            f'NLP_SOLVER_PATH={SOLVER_PATH} {PYTHON} -u {SCRIPT_NAME} --run --cell {cell} --spec-sha256 {spec_sha}'
            + (' --preconditions-only' if preconditions_only else '')
            + f' > {os.path.join(ROOT_REL, log)} 2>&1')


def summarize_command(cell):
    return ('cd /Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation && set -o noclobber && '
            f'{PYTHON} -u {SCRIPT_NAME} --summarize --after-cell {cell} > '
            f'{os.path.join(ROOT_REL, f"summarize_after_{cell}_launch.log")} 2>&1')


# ======================================================================================================================
#  post-run gates (cell-keyed restatements of W132's; W132's generic ones reused)
# ======================================================================================================================
LINE_FIELDS_REQUIRED = L132.LINE_FIELDS_REQUIRED
_decision = L132._decision


def hold_checks(cell, eval_dir, rec):
    lines = _read_jsonl(os.path.join(eval_dir, W.CYCLE_FILE))
    summ = rec.get('settling_resettle_summary') or {}
    by = {x['cycle']: x for x in lines}
    cycles = sorted(by)
    fp = summ.get('first_residual_pass_run')
    pre = [by[c] for c in cycles if fp is None or c <= fp]
    held = [by[c] for c in cycles if fp is not None and c > fp]
    rho_fp = ((by.get(fp) or {}).get('rho') or {}).get('rho_after') if fp else None
    dec = _decision(eval_dir) or {}
    last = cycles[-1] if cycles else None
    c = W.CELLS[cell]
    parts = {
        'one_line_per_cycle_contiguous': cycles == list(range(1, (rec.get('cycles_run') or 0) + 1)),
        'first_pass_as_declared_for_gated': (not c['gated']) or fp == c['k0'],
        'first_pass_equals_the_rule_first_k0_v2': fp == summ.get('first_k0_v2'),
        'no_hold_acted_through_first_pass': all(
            (x.get('holds') or {}) == {'aa': False, 'tail_apply': False, 'tail_next': False, 'rho': False}
            or ((x.get('holds') or {}).get('aa') is None and x.get('gross') is None) for x in pre),
        'aa_held_off_after_first_pass': all((x.get('aa') is None and x.get('gross') is None)
                                            or ((x.get('aa') or {}).get('hold') is True
                                                and (x.get('aa') or {}).get('action') == R.AA_OFF_ACTION) for x in held),
        'tail_held_on_after_first_pass': all((x.get('tail_apply') or {}).get('active_passed') is True
                                             and (x.get('tail_next') or {}).get('returned') is True for x in held),
        'rho_frozen_after_first_pass_at_its_values': fp is None or (bool(rho_fp) and all(
            (x.get('rho') or {}).get('hold') is True and (x.get('rho') or {}).get('rho_after') == rho_fp
            and not (x.get('rho') or {}).get('changed_channels') for x in held)),
        'certificate_length_disabled_except_the_rule_end': all(
            x.get('certificate_length_in_force_at_cycle_end') == W.CERTIFICATION_DISABLED_THRESHOLD
            or (x['cycle'] == last and x.get('certificate_length_in_force_at_cycle_end') == W.SETTLING_END_THRESHOLD
                and summ.get('stopped_by') in ('settling_rule', 'rule_cap')) for x in lines),
        'replay_equal_every_gated_cycle': ((not c['gated'])
                                           or (all(by[k].get('replay_equal') is True
                                                   for k in range(1, c['k0'] + 1) if k in by)
                                               and all(k in by for k in range(1, c['k0'] + 1)))),
        'summary_ok': summ.get('ok') is True,
        'summary_schema_is_w135': summ.get('schema') == W.SCHEMA,
        'decision_present': bool(dec),
    }
    return all(parts.values()), {'parts': parts, 'first_pass': fp, 'rho_at_first_pass': rho_fp}


def stopping_check(cell, rec, eval_dir):
    summ = rec.get('settling_resettle_summary') or {}
    dec = _decision(eval_dir) or {}
    k = rec.get('cycles_run')
    spec_cap = W.spec_cap(cell)
    if dec.get('status') == 'certified':
        ok = k == dec.get('k_star') and summ.get('stopped_by') == 'settling_rule'
    elif dec.get('status') == 'uncertified':
        if W.CELLS[cell]['gated']:
            ok = k == dec.get('k_cap') == spec_cap and summ.get('stopped_by') == 'cap'
        else:
            ok = k == dec.get('k_cap') and summ.get('stopped_by') == ('rule_cap' if k < spec_cap else 'cap')
    else:
        ok = False
    return bool(ok), {'cycles_run': k, 'decision_status': dec.get('status'), 'k_star': dec.get('k_star'),
                      'k_cap': dec.get('k_cap'), 'stopped_by': summ.get('stopped_by'), 'spec_cap': spec_cap}


def settling_replay_check(cell, eval_dir):
    """The pure rule v4 replayed on the run's per_cycle_record (Q, boyd) with the in-cycle t_sum and all_optimal_k
    reproduces every in-cycle rule record and the decision (W137's G17 on the W135 declaration)."""
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, W.CYCLE_FILE))}
    rule = K.K137.pure_rule(W.declaration_for(cell))
    pure = []
    for r in rows:
        ln = lines.get(r['cycle']) or {}
        pure.append(rule.observe(r['cycle'], r['gross_operational_cost'],
                                 bool(r['boyd_all_pass'] and r['local_solves_ok']), ln.get('t_sum'),
                                 bool(ln.get('all_optimal_k'))))
    in_cycle = [(lines.get(r['cycle']) or {}).get('settling') for r in rows]
    dec = _decision(eval_dir) or {}
    pd = rule.decision or {}
    keys = L137.DECISION_KEYS_REPLAYED
    parts = {'every_cycle_record_reproduced': [K._jt(a) for a in in_cycle] == [K._jt(b) for b in pure],
             'decision_reproduced': bool(dec) and all(K._jt(dec.get(k)) == K._jt(pd.get(k)) for k in keys),
             'decision_file_present': bool(dec), 'decision_is_version_4': dec.get('version') == 4}
    return all(parts.values()), {'parts': parts}


def replay_gate_full(cell, eval_dir):
    """Gated cell (pb): rows 1..k0 of the run's per_cycle_record.jsonl against the ORIGINAL record's rows: EVERY field."""
    c = W.CELLS[cell]
    ref = {r['cycle']: r for r in _read_jsonl(_abs(W.reference_path(cell)))}
    path = os.path.join(eval_dir, 'per_cycle_record.jsonl')
    run = {r['cycle']: r for r in _read_jsonl(path)} if os.path.isfile(path) else {}
    first, detail = None, None
    for k in range(1, c['k0'] + 1):
        a, b = run.get(k), ref.get(k)
        if a is None:
            first, detail = k, {'missing_in_run': True}
            break
        diff = sorted(f for f in set(a) | set(b) if json.dumps(a.get(f), sort_keys=True) != json.dumps(b.get(f),
                                                                                                        sort_keys=True))
        if diff:
            ga, gb = a.get('gross_operational_cost'), b.get('gross_operational_cost')
            first, detail = k, {'fields_differing': diff, 'gross_difference_run_minus_recorded':
                                (ga - gb) if (ga is not None and gb is not None) else None}
            break
    return {'bitwise_through_k0': first is None, 'k0': c['k0'], 'first_divergence_cycle': first, 'divergence': detail}


def overlap_check(cell, rec):
    summ = rec.get('settling_resettle_summary') or {}
    ov = summ.get('overlap_k0_plus_1_to_N_old') or []
    c = W.CELLS[cell]
    if not c['gated']:
        return len(ov) == 0, {'n': len(ov)}
    ok = [o['cycle'] for o in ov] == list(range(c['k0'] + 1, c['N_old'] + 1)) and all(
        o.get('Q_new_minus_Q_old') is not None for o in ov)
    return bool(ok), {'n': len(ov)}


def _end_cycle(eval_dir, rec):
    dec = _decision(eval_dir) or {}
    end = dec.get('k_star') if dec.get('status') == 'certified' else dec.get('k_cap')
    return end if end is not None else rec.get('cycles_run')


def floor_capture_check(cell, eval_dir, rec):
    """G26: the SoH-floor sidecar holds the end cycle's line with every block of the unit node's 2025 cohort (E), or
    of the pb cell's node 5 2025 cohort (pb, report-only reading)."""
    path = os.path.join(eval_dir, W.FLOOR_SIDECAR_FILE)
    if not os.path.isfile(path):
        return False, {'error': f'{W.FLOOR_SIDECAR_FILE} absent'}
    lines = _read_jsonl(path)
    node = W.UNIT_NODE if W.CELLS[cell]['item'] == 'E' else 5
    reading, ok = W.floor_year_reading(lines, _end_cycle(eval_dir, rec), node=node)
    return bool(ok and len(lines) == (rec.get('cycles_run') or -1)), {'reading': reading, 'n_lines': len(lines)}


def ageing_series(rec):
    """Node 7, cohort 2025: per block (y 0..2) SoH_used, SoH_end, EFC/day, from the record's terminal ageing capture
    (the run's own ESSO models, read-only). (series, ok)."""
    att = rec.get('ageing_trajectory_terminal') or {}
    cells = [c for c in ((att.get('nodes') or {}).get(str(W.UNIT_NODE)) or {}).get('cells') or []
             if int(c.get('y_inv', -1)) == W.UNIT_COHORT_INDEX]
    cells = sorted(cells, key=lambda c: int(c['y']))
    ok = ([int(c['y']) for c in cells] == [0, 1, 2]
          and [int(c['block_year']) for c in cells] == list(W.INSTANCE_YEARS)
          and all(str(c.get('investment_year')) == str(W.UNIT_YEAR) for c in cells))
    return {'soh_used': [c['soh_used_for_available_energy'] for c in cells],
            'soh_end': [c['soh_end'] for c in cells], 'efc_per_day': [c['efc_per_day'] for c in cells],
            'block_years': [c['block_year'] for c in cells], 'phi_cal_in_model': [c.get('phi_cal_in_model') for c in cells],
            'cl_eff': [c.get('cl_eff') for c in cells],
            'salvage_value_node7': ((att.get('nodes') or {}).get(str(W.UNIT_NODE)) or {}).get('salvage_value'),
            'available_energy_soh_point': att.get('available_energy_soh_point'),
            'ageing_enabled': att.get('ageing_enabled')}, bool(ok)


def c2_calfade_consistency(eval_dir):
    """G28 (e_c2_calfade only): the Planner's consistency criterion against the settled unit 3f084f2f through its
    certifying cycle 172, with the diagnosis of a first difference (the cycle line's regime and hold flags beside the
    W101 unit's line at the same cycle)."""
    run_rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    ref_rows = _read_jsonl(_abs(os.path.join(UNIT_REF_EVAL_DIR, 'per_cycle_record.jsonl')))
    through = K.UNIT_SETTLED['k_star']
    eq = W.trajectory_equality(run_rows, ref_rows, through)
    diag = None
    fd = eq['first_difference']
    if fd is not None and fd.get('cycle') is not None:
        c = fd['cycle']
        lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, W.CYCLE_FILE))}
        unit_lines = {x['cycle']: x for x in _read_jsonl(_abs(os.path.join(
            UNIT_REF_EVAL_DIR, 'settling_continuation_cycle_record.jsonl')))}
        x = lines.get(c) or {}
        prev = {k: (lines.get(k) or {}) for k in range(max(1, c - 3), c)}
        diag = {'cycle': c, 'regime_in_this_run': x.get('regime'), 'holds_in_this_run': x.get('holds'),
                'hold_changed_value_this_cycle': {g: (x.get(g) or {}).get('hold_changed_value')
                                                  for g in ('aa', 'tail_apply', 'tail_next')},
                'hold_changed_value_previous_3': {k: {g: (v.get(g) or {}).get('hold_changed_value')
                                                      for g in ('aa', 'tail_apply', 'tail_next')}
                                                  for k, v in prev.items()},
                'holds_in_the_unit_run': (unit_lines.get(c) or {}).get('holds'),
                'first_residual_pass_this_run': next((k for k in sorted(lines) if (lines[k].get('boyd_k'))), None),
                'reading': ('cycle 1 with an identical model -> nondeterminism / working-dir effect; cycles 104..112 with '
                            'a hold_changed_value flag -> the hold regime (not the variant); see the prediction\'s '
                            'expected_equality_reasoning')}
    q_ks = {'run_last_cycle': run_rows[-1]['cycle'] if run_rows else None,
            'run_Q_last': run_rows[-1]['gross_operational_cost'] if run_rows else None,
            'unit_Q_172': next((r['gross_operational_cost'] for r in ref_rows if r['cycle'] == through), None)}
    return bool(eq['reproduced']), {**eq, 'diagnosis': diag, **q_ks,
                                    'reference': {'eval_dir': UNIT_REF_EVAL_DIR,
                                                  'per_cycle_record_sha256': _sha(os.path.join(
                                                      UNIT_REF_EVAL_DIR, 'per_cycle_record.jsonl'))}}


GATE_SCOPE = {
    'G8_persistence_on_production_certificate': ('every cell: persistence follows status_production_trajectory '
                                                 '(Addendum 59; W137)'),
    'G9_ess_ageing_readback': 'pb_y2025_n5_v4 only (no model variant: the harness reads the ESS ageing baseline back)',
    'G9v_variant_readback_and_floor': ('the six E cells only: the G9 REPLACEMENT (W.variant_readback_gate; the harness '
                                       'does not compute the ESS-ageing readback for a variant entry)'),
    'G19_replay_bitwise_1_k0_every_field': 'pb_y2025_n5_v4 only (the E cells have no replay reference: SKIPPED)',
    'G23_overlap_recorded': 'pb: k0+1..N_old recorded; E: none (asserted empty)',
    'G26_floor_sidecar_end_cycle': 'every cell (E: node 7 cohort 2025, the floor year; pb: node 5, report-only)',
    'G27_ageing_trajectory_terminal': 'the six E cells only',
    'G28_c2_calfade_reproduces_3f084f2f_through_172': ('e_c2_calfade only; a failure is a failed Planner validation: '
                                                       'exit 3, STOP for the Planner'),
    'G6_v37_optimal_and_four_metrics': 'every cell; REPORTED, does NOT stop the campaign (Planner ruling at W128)',
    'all_other_gates': 'every cell',
}
NON_STOPPING_GATES = ('G6_v37_optimal_and_four_metrics', 'G28_c2_calfade_reproduces_3f084f2f_through_172')
PREDICTION_GATES = ('G28_c2_calfade_reproduces_3f084f2f_through_172',)


def cell_gates(cell, entry, eval_dir):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    c = W.CELLS[cell]
    gates, detail = {}, {'exit_code': exit_code, 'gate_scope': GATE_SCOPE}
    gates['G1_harness_clean'] = (exit_code == 0 and bool(rec) and rec.get('status') in ('certified', 'not_certified')
                                 and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json')))
    if not rec or rec.get('status') == 'error':
        detail['barrier'] = {k: rec.get(k) for k in ('status', 'barrier_cause')}
        detail['settling_resettle_summary'] = rec.get('settling_resettle_summary')
        return gates, detail, rec
    gates['G2_eval_key'] = rec.get('eval_key') == entry['eval_key'] == expected_keys(cell)['resettle_key']
    ch, d = L.evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'])
    detail['w86_evaluation_checks'] = {'checks': ch, 'detail': d}
    gates['G3_append_reconcile'] = ch.get('append_reconciles_byte_identical', False)
    gates['G4_tail_state_check'] = ch.get('tail_state_check_matches', False)
    records = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    gates['G5_solve_profile_reconciled_per_event'], detail['G5'] = W101L.solve_profile_check(rec, len(records))
    g6 = W98L.g6_v37_evaluate_records(records, rec.get('cycles_run'), b=N_NETWORK_BLOCKS)
    gates['G6_v37_optimal_and_four_metrics'] = g6['gate_pass']
    detail['G6_v37'] = g6
    gates['G7_append_sealed'] = ch.get('append_sealed_after_reconcile', False)
    gates['G8_persistence_on_production_certificate'], detail['G8'] = V4.persistence_check_production_certificate(
        rec, eval_dir)
    if c['arm'] is None:
        gates['G9_ess_ageing_readback'] = ch.get('ess_ageing_readback_all_match', False)
    else:
        gates['G9v_variant_readback_and_floor'], detail['G9v'] = W.variant_readback_gate(
            rec, W.declaration_for(cell))
    xc = X.acceptance_cross_check(eval_dir)
    gates['G11_acceptance_cross_check'] = xc['agrees']
    detail['G11'] = xc
    gates['G13_holds_inert_through_first_pass_held_after'], detail['G13'] = hold_checks(cell, eval_dir, rec)
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    gates['G14_all_block_capture'], detail['G14'] = W101L.block_capture(eval_dir, rec, rows)
    gates['G15_stopping_consistent'], detail['G15'] = stopping_check(cell, rec, eval_dir)
    gates['G16_lambda_sidecar_complete'], detail['G16'] = W101L.lambda_sidecar_check(eval_dir, rec)
    gates['G17_rule_v4_replay_reproduces_in_cycle'], detail['G17'] = settling_replay_check(cell, eval_dir)
    gates['G18_line_fields_every_cycle'], detail['G18'] = L137.line_fields_check(eval_dir)
    if c['gated']:
        rg = replay_gate_full(cell, eval_dir)
        gates['G19_replay_bitwise_1_k0_every_field'] = rg['bitwise_through_k0']
        detail['G19'] = rg
    else:
        detail['G19'] = {'skipped': 'ungated E cell (first evaluation of a model-variant arm): no replay reference'}
    gates['G20_production_counters_in_per_cycle_record'] = all(
        'consecutive_converged_cycles' in r and 'boyd_all_pass' in r for r in rows)
    gates['G21_creep_captures_complete'], detail['G21'] = L118.creep_capture_check(eval_dir, rec, rows)
    gates['G22_t_sum_in_cycle_equals_stride_and_terminal'], detail['G22'] = L118.t_sum_check(eval_dir)
    gates['G23_overlap_recorded'], detail['G23'] = overlap_check(cell, rec)
    try:
        gates['G24_exit_capture_equals_the_solve_records_and_esso_logs'], detail['G24'] = L132.exit_crosscheck(
            eval_dir, rec=rec)
    except Exception as error:  # noqa: BLE001 -- recorded; the gate FAILS
        gates['G24_exit_capture_equals_the_solve_records_and_esso_logs'] = False
        detail['G24'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    gates['G25_record_status_follows_the_settling_decision'], detail['G25'] = L132.status_label_check(rec, eval_dir)
    gates['G26_floor_sidecar_end_cycle'], detail['G26'] = floor_capture_check(cell, eval_dir, rec)
    if c['arm'] is not None:
        series, ok = ageing_series(rec)
        gates['G27_ageing_trajectory_terminal'] = ok
        detail['G27'] = series
    if c['arm'] == W.BASELINE_EQUIVALENT_ARM:
        try:
            gates['G28_c2_calfade_reproduces_3f084f2f_through_172'], detail['G28'] = c2_calfade_consistency(eval_dir)
        except Exception as error:  # noqa: BLE001
            gates['G28_c2_calfade_reproduces_3f084f2f_through_172'] = False
            detail['G28'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    return gates, detail, rec


# ======================================================================================================================
#  the per-cell report
# ======================================================================================================================
def cell_report(cell, eval_dir, rec):
    """W132's per-cell report (definitions.per_cell) plus, for an E cell, the arm, its variant and closed-form k, the
    floor year at the end cycle, the per-block SoH / EFC and AE / EFC by the committed formula."""
    rows = {r['cycle']: r for r in _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))}
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, W.CYCLE_FILE))}
    dec = _decision(eval_dir) or {}
    c = W.CELLS[cell]
    summ = rec.get('settling_resettle_summary') or {}
    q = {k: r['gross_operational_cost'] for k, r in rows.items()}
    last = max(rows)
    end = dec.get('k_star') if dec.get('status') == 'certified' else dec.get('k_cap')
    end = end if end is not None else last
    steps = [q[k] - q[k - 1] for k in sorted(q) if (k - 1) in q and q[k] is not None and q[k - 1] is not None]
    fp = summ.get('first_residual_pass_run')
    after = [x for k, x in lines.items() if fp is not None and k > fp]
    win = [k for k in range(end - PF_SLOPE_WINDOW + 1, end + 1) if k in rows
           and rows[k].get('boyd_pf_primal_ratio') is not None]
    certified = dec.get('status') == 'certified'
    q_end = dec.get('Q_k_star') if certified else dec.get('Q_at_cap')
    t_end = dec.get('t_sum_k_star') if certified else dec.get('t_sum_at_cap')
    rep = {'cell': cell, 'item': c['item'], 'arm': c['arm'], 'claim_group': W.GROUP_OF_ITEM[c['item']],
           'gated': c['gated'], 'eval_key': rec.get('eval_key'), 'candidate_key': rec.get('candidate_key'),
           'candidate_canonical': rec.get('candidate_canonical'), 'original_eval_key': c['orig_eval_key'],
           'k0_run': fp, 'first_k0_v2': summ.get('first_k0_v2'), 'criterion_version': dec.get('version'),
           'k0_original': c['k0'] if c['gated'] else None, 'N_old': c['N_old'] if c['gated'] else None,
           'rule_cap': dec.get('cap'), 'cap_ceiling': c['cap_ceiling'], 'cycles_run': rec.get('cycles_run'),
           'status': dec.get('status'), 'record_status': rec.get('status'), 'k_star': dec.get('k_star'),
           'end_cycle': end, 'branch': dec.get('branch'), 'band': dec.get('band'), 'band_width': dec.get('band_width'),
           'range_over_tau': dec.get('range_over_tau'), 'Q_k_star': dec.get('Q_k_star'),
           't_sum_k_star': dec.get('t_sum_k_star'), 'Q_cc_k_star': dec.get('Q_cc_k_star'),
           'Q_end': q_end, 't_sum_end': t_end, 'Q_cc_end': (q_end + t_end) if (q_end is not None and t_end is not None)
           else None, 'Q_net_end': (rows.get(end) or {}).get('recourse'),
           'terminal_salvage_value_end': (rows.get(end) or {}).get('terminal_salvage_value'),
           'T': dec.get('T'), 'A': dec.get('A'), 'P_hat': dec.get('P_hat'),
           'lapse_events': dec.get('lapse_events'), 'gap_refusals': dec.get('gap_refusals'),
           'non_optimal_cycles': dec.get('non_optimal_cycles'), 'W': dec.get('W'), 'window': dec.get('window'),
           'vetoes': dec.get('vetoes'), 'n_vetoes': dec.get('n_vetoes'),
           'window_all_optimal': dec.get('window_all_optimal'), 'out_of_window_reads': dec.get('out_of_window_reads'),
           'terminal_step_abs': abs(steps[-1]) if steps else None,
           'terminal_step_over_EPS0': (abs(steps[-1]) / SC4.EPS0) if steps else None,
           'rule_ten_last_cycle': ((rows[last]['objective_change_abs'] / rows[last]['objective_tolerance'])
                                   if rows[last].get('objective_change_abs') is not None
                                   and rows[last].get('objective_tolerance') else None),
           'boyd_lapses_after_k0': sum(1 for x in after if not x.get('boyd_k')),
           'pf_primal_slope_last_50': L132._ols_slope(win, [rows[k]['boyd_pf_primal_ratio'] for k in win]),
           'pf_primal_slope_window': [win[0], win[-1]] if win else None,
           'objective_convention': DEFINITIONS['objective_convention']}
    if not certified:
        rep.update({k: dec.get(k) for k in ('k_cap', 'reasons', 'band_window', 'drift_rate_mean_dQ_last_25',
                                            'dQ_cc_rate_mean_last_25', 'Q_at_cap', 't_sum_at_cap', 'Q_cc_at_cap',
                                            'gap_clause_refused_at_cap')})
        rep['t_by_node_at_cap'] = (lines.get(end) or {}).get('t_by_node')
        rep['gap_refused'] = bool(dec.get('gap_refusals'))
        rep['label'] = GAP_REFUSED_LABEL if rep['gap_refused'] else 'uncertified at the cap'
    if c['gated']:
        ref = {r['cycle']: r for r in _read_jsonl(_abs(W.reference_path(cell)))}
        q_n_old = ref[c['N_old']]['gross_operational_cost']
        ov = summ.get('overlap_k0_plus_1_to_N_old') or []
        rep.update({'Q_N_old': q_n_old, 's_signed': (q_end - q_n_old) if q_end is not None else None,
                    'overlap': ov, 'overlap_at_N_old': next((o for o in ov if o['cycle'] == c['N_old']), None),
                    'replay_bitwise_through': summ.get('replay_bitwise_through_cycle'),
                    'replay_first_divergence': summ.get('replay_first_divergence')})
    floor_path = os.path.join(eval_dir, W.FLOOR_SIDECAR_FILE)
    if os.path.isfile(floor_path):
        node = W.UNIT_NODE if c['item'] == 'E' else 5
        rep['floor_reading'] = W.floor_year_reading(_read_jsonl(floor_path), end, node=node)[0]
        rep['floor_year'] = rep['floor_reading'].get('floor_year')
    if c['arm'] is not None:
        series, ok = ageing_series(rec)
        mv = W.arm_variant(cell)
        cal = ((rec.get('ess_ageing_verified_pre_run') or {}).get('loaded') or {}).get('calibration') or {}
        rep.update({'model_variant': mv, 'ageing_series': series, 'ageing_series_ok': ok,
                    'closed_form': (W.closed_form_expected(mv, cal['cycles_n'], cal['reference_dod_d'])
                                    if cal.get('cycles_n') is not None else None),
                    'AE': M46A.pv(series['soh_used']) if ok else None,
                    'EFC': M46A.pv(series['efc_per_day']) if ok else None})
    rep['view'] = L132.view_from_report(rep)
    return rep


# ======================================================================================================================
#  the scorer (pure below the loaders): the claims, the restated statements, the predictions
# ======================================================================================================================
def _sign(scored):
    """+ / - / 0 of a scored claim's primary difference: +/- only when its verdict is determinate."""
    if not scored or str(scored.get('verdict', '')).startswith('not scored'):
        return None
    if scored.get('verdict') != 'determinate':
        return '0'
    return '+' if scored.get('d_primary', 0.0) > 0 else '-'


def resolve_weighted(d_q, d_cc, terms):
    """definitions.weighted_resolution: terms = [(coefficient, view)]."""
    unc = [(cf, v) for cf, v in terms if v.get('status') != 'certified']
    m_cc = abs(d_cc) if (d_q > 0) == (d_cc > 0) else -abs(d_cc)
    if not unc:
        res = sum(abs(cf) * v['band'] for cf, v in terms)
        return {'rule': 'settled (weighted sum of the bands)', 'resolution': res,
                'verdict': 'determinate' if abs(d_q) > res else 'within resolution',
                'verdict_Qcc_report_only': 'determinate' if m_cc > res else 'within resolution',
                'margin_over_resolution': abs(d_q) / res if res else None}
    comps = []
    for cf, v in unc:
        if v.get('slack') is None or v.get('gap') is None:
            return {'rule': 'uncertified_form', 'bar': None, 'verdict': 'indeterminate (slack undefined)',
                    'note': 'an ungated uncertified cell has no s = Q(cap) - Q_N_old'}
        comps += [abs(cf) * v['gap'], abs(cf) * v['slack']]
    bar = 3.0 * max(comps)
    det = abs(d_q) > bar and m_cc > bar
    return {'rule': 'uncertified_form (weighted)', 'bar': bar, 'bar_components': comps, 'margin_Q': abs(d_q),
            'margin_Qcc': m_cc, 'verdict': 'determinate' if det else 'within the uncertified bar'}


def epsilon_050_references(am, k_by_arm):
    """eps_k^0.50 = ln(value_over_C3) / ln(k_a / k_C3) from the committed 0.50-era mechanism table."""
    by = {t['variant']: t for t in am['table']}
    out = {}
    for arm in ARMS_WITH_K_DIFFERENT_FROM_C3:
        vr = by[arm]['value_over_C3']
        out[arm] = math.log(vr) / math.log(k_by_arm[arm] / k_by_arm['C3_unit'])
    return out


def score_item_e(views, reports, refs, dataset, am, k_by_arm):
    """Item E at 0.70: the W117 claims, the restated statements S1-S7, the expert's prediction. Pure (given the loads)."""
    x0 = refs[X0_REF]
    q0, q0cc = x0['Q'], x0['Q_cc']
    arms = {W.CELLS[c]['arm']: c for c in W.E_CELLS}
    claims = {}
    for cl in dataset['claims']:
        if cl['item'] != 'E':
            continue

        def view(side):
            s = cl[side]
            return refs[s['ref']] if s['source'] == 'reference' else views.get(s['cell'], {})
        claims[cl['claim_id']] = L132.score_claim(cl, view('ref'), view('other'))

    def vmi_claim(arm):
        return claims.get('E:C3_unit_value_minus_I') if arm == 'C3_unit' else claims.get(
            f'E:n7_4h_e1_{arm}:value_minus_I')

    def vs_c3_claim(arm):
        return claims.get(f'E:n7_4h_e1_{arm}:vs_C3')
    unit_I = next(cl['I_other'] for cl in dataset['claims'] if cl['claim_id'] == 'E:C3_unit_value_minus_I')
    per_arm = {}
    for arm, cell in arms.items():
        v = views.get(cell) or {}
        rep = reports.get(cell) or {}
        if v.get('Q') is None:
            per_arm[arm] = {'cell': cell, 'status': 'no result yet'}
            continue
        val = q0 - v['Q']
        val_cc = (q0cc - v['Q_cc']) if v.get('Q_cc') is not None else None
        per_arm[arm] = {'cell': cell, 'status': v.get('status'), 'Q': v['Q'], 'value': val, 'value_Qcc': val_cc,
                        'I': unit_I, 'value_minus_I': val - unit_I, 'I_over_value': (unit_I / val) if val else None,
                        'floor_year': rep.get('floor_year'), 'AE': rep.get('AE'), 'EFC': rep.get('EFC'),
                        'k': k_by_arm[arm], 'band': v.get('band'),
                        'value_minus_I_claim': {k: (vmi_claim(arm) or {}).get(k) for k in ('verdict', 'd_primary')},
                        'value_minus_I_sign': _sign(vmi_claim(arm))}
    c3 = per_arm.get('C3_unit') or {}
    for arm, row in per_arm.items():
        if 'value' not in row or 'value' not in c3:
            continue
        row['x_C3'] = row['value'] / c3['value'] if c3['value'] else None
        if arm != 'C3_unit':
            sc = vs_c3_claim(arm) or {}
            row['vs_C3_claim'] = {k: sc.get(k) for k in ('verdict', 'd_primary', 'gross')}
            row['vs_C3_resolvable'] = sc.get('verdict') == 'determinate'
            if row.get('AE') and c3.get('AE') and abs(row['AE'] / c3['AE'] - 1.0) > 1e-9 and row['x_C3'] and row['x_C3'] > 0:
                row['eps_AE'] = math.log(row['x_C3']) / math.log(row['AE'] / c3['AE'])
    am_by = {t['variant']: t for t in am['table']}
    am_name = {'C3_unit': 'C3 (baseline)', 'C2': 'C2', 'C4': 'C4', 'C2_calfade': 'C2_calfade',
               'C3_midblock': 'C3_midblock', 'no_ageing': 'no_ageing'}
    s = {}
    s['S1_x_c3_multipliers'] = {a: {'x_C3_070': (per_arm.get(a) or {}).get('x_C3'),
                                    'x_C3_050': am_by[am_name[a]]['value_over_C3'],
                                    'vs_C3_verdict_070': ((per_arm.get(a) or {}).get('vs_C3_claim') or {}).get('verdict')}
                                for a in arms if a != 'C3_unit'}
    s['S2_I_over_value_sensitivity_band'] = {a: {'I_over_value_070': (per_arm.get(a) or {}).get('I_over_value'),
                                                 'I_over_value_050': unit_I / am_by[am_name[a]]['value_eur'],
                                                 'value_minus_I_verdict_070': ((per_arm.get(a) or {}).get(
                                                     'value_minus_I_claim') or {}).get('verdict')} for a in arms}
    signs = {a: (per_arm.get(a) or {}).get('value_minus_I_sign') for a in arms}
    complete = all(v is not None for v in signs.values())
    s['S3_convention_does_not_decide_the_sign'] = {
        'signs_value_minus_I': signs, 'complete': complete,
        'holds_restated_no_arm_pays_determinately': (not any(v == '+' for v in signs.values())) if complete else None,
        'convention_decides_the_sign_plus_and_minus_present': (('+' in signs.values()) and ('-' in signs.values()))
        if complete else None}
    aged = {a: signs.get(a) for a in AGED_ARMS}
    s['S4_break_even_needs_no_ageing_at_all'] = {
        'no_ageing_value_minus_I_sign': signs.get('no_ageing'), 'aged_signs': aged,
        'no_ageing_within_resolution': signs.get('no_ageing') == '0' if signs.get('no_ageing') else None,
        'no_aged_arm_positive': (not any(v == '+' for v in aged.values())) if all(v is not None for v in aged.values())
        else None}
    eps = {a: (per_arm.get(a) or {}).get('eps_AE') for a in arms if a != 'C3_unit'}
    resolv = {a: (per_arm.get(a) or {}).get('vs_C3_resolvable') for a in arms if a != 'C3_unit'}
    res_eps = [e for a, e in eps.items() if resolv.get(a) and e is not None]
    s['S5_elasticity_to_available_energy'] = {
        'eps_AE_070': eps, 'resolvable_070': resolv,
        'band_over_resolvable_070': [min(res_eps), max(res_eps)] if res_eps else None,
        'eps_AE_050': {a: am_by[am_name[a]]['elasticity'] for a in arms if a != 'C3_unit'}}
    mid = per_arm.get('C3_midblock') or {}
    s['S6_midblock_x1009'] = {'x_C3_070': mid.get('x_C3'), 'x_C3_050': am_by['C3_midblock']['value_over_C3'],
                              'vs_C3_verdict_070': (mid.get('vs_C3_claim') or {}).get('verdict')}
    s['S7_floor_year_per_arm'] = {a: {'floor_year_070': (per_arm.get(a) or {}).get('floor_year'),
                                      'floor_050': 'no floor row active (soh_min 0.50)'} for a in arms}
    # the expert's prediction
    eps050 = epsilon_050_references(am, k_by_arm)
    pred = {}
    for arm in ARMS_WITH_K_DIFFERENT_FROM_C3:
        row, cell = per_arm.get(arm) or {}, arms[arm]
        va, vc = views.get(cell) or {}, views.get(arms['C3_unit']) or {}
        if 'value' not in row or 'value' not in c3:
            pred[arm] = {'scored': False, 'reason': 'no result yet'}
            continue
        lk = math.log(k_by_arm[arm] / k_by_arm['C3_unit'])
        rho = (k_by_arm[arm] / k_by_arm['C3_unit']) ** eps050[arm]
        eps070 = math.log(row['value'] / c3['value']) / lk if (row['value'] > 0 and c3['value'] > 0) else None
        d_q = va['Q'] - rho * vc['Q'] - (1.0 - rho) * q0
        d_cc = (va['Q_cc'] - rho * vc['Q_cc'] - (1.0 - rho) * q0cc) if (va.get('Q_cc') is not None
                                                                        and vc.get('Q_cc') is not None) else None
        floor_binds = row.get('floor_year') is not None
        r = resolve_weighted(d_q, d_cc if d_cc is not None else d_q, [(1.0, va), (-rho, vc), (-(1.0 - rho), x0)])
        pred[arm] = {'scored': floor_binds, 'floor_year': row.get('floor_year'),
                     'reason': None if floor_binds else 'the floor does not bind for this arm: not scored',
                     'eps_k_050': eps050[arm], 'eps_k_070': eps070, 'ln_k_ratio': lk, 'rho': rho,
                     'value_threshold_implied_by_eps050': rho * c3['value'], 'value_070': row['value'],
                     'd_Q': d_q, 'd_Qcc': d_cc, 'resolution': r,
                     'holds': (eps070 is not None and eps070 < eps050[arm]) if floor_binds else None,
                     'verdict': r['verdict'] if floor_binds else 'not scored'}
    return {'claims': claims, 'per_arm': per_arm, 'statements': s, 'expert_prediction': pred,
            'epsilon_050_references_recomputed': eps050, 'Q0': q0, 'Q0_cc': q0cc, 'I_unit': unit_I}


def reference_views():
    return L132.reference_views()


# ======================================================================================================================
#  self-tests (zero solves; committed inputs; synthetic eval dirs from the real wrappers)
# ======================================================================================================================
def _k_by_arm_now():
    ess = _load(W.ESS_PARAMS_REL)
    cal = ess['ageing']['calibration']
    return {a: W.closed_form_expected(dict(mv, ageing_enabled=True), cal['cycles_n'], cal['reference_dod_d'])['k']
            for a, mv in W.ARMS.items()}


def _synthetic_variant_record(cell):
    """A synthetic record whose variant readbacks are built from the closed forms (for the G9-replacement self-test)."""
    mv = W.arm_variant(cell)
    cal = {'status': 'ACTIVE', 'cycles_n': 10000, 'reference_dod_d': 0.8, 'eol_retention_r': 0.8}
    cf = W.closed_form_expected(mv, cal['cycles_n'], cal['reference_dod_d'])
    rb = {'k': cf['k'], 'phi_cal_in_model': cf['phi_cal_in_model'],
          'available_energy_soh_point': cf['available_energy_soh_point'], 'd_row_form': cf['d_row_form'],
          'floor_row_lower': 0.7}
    chk = {'k': True, 'n_years_from_d_row_equals_data': True, 'phi_cal_in_model': True, 'd_row_form': True,
           'available_energy_soh_point': True, 'soh_row_exp_form': True}
    readback = {'expected': {k: cf[k] for k in ('k', 'phi_cal_in_model', 'available_energy_soh_point', 'ageing_enabled',
                                                'd_row_form')},
                'per_node': {n: {'readback': dict(rb), 'checks': dict(chk)} for n in ('5', '7', '9')},
                'all_match': True}
    ess_decl = {'minimum_soh': 0.7, 'calibration': cal}
    return {'model_variant': mv, 'model_variant_label': H.MODEL_VARIANT_LABEL,
            'model_variant_applied_in_child': {'checks': {'x': True}, 'after': {'per_ess': [{'soh_min': 0.7}] * 9}},
            'model_variant_readback_pre_run': readback, 'model_variant_readback_terminal': json.loads(json.dumps(readback)),
            'ess_ageing_baseline': ess_decl, 'ess_params_sha256_in_child': W.ESS_PARAMS_SHA256,
            'ess_ageing_verified_pre_run': {'loaded': ess_decl, 'checks': {'y': True}}}


def _synthetic_run_dir(tmp, cell, variant):
    """W132's `_synthetic_run_dir` restated for a W135 cell (K.drive on the W135 declaration)."""
    c = W.CELLS[cell]
    fp_at = 50
    n_nonopt = (c['N_old'] + 20) if c['gated'] else fp_at + 18     # W132's placement: k0 + 18
    d = K.drive(cell, 'creep' if variant == 'creep' else 'certify', first_pass_at=fp_at,
                nonopt=({n_nonopt: ('ESSO|9',)} if variant == 'nonopt' else None))
    files, st = d['files'], d['state']
    lines = {x['cycle']: x for x in files[W.CYCLE_FILE]}
    creep = {x['cycle']: x for x in files[W.CREEP_FILE]}
    if variant == 'tamper_t_sum_30':
        lines[30]['t_sum'] = lines[30]['t_sum'] + 1.0
    if variant == 'tamper_hold_flag':
        lines[max(lines) - 5]['aa']['hold'] = False
    if variant == 'tamper_decision':
        files[W.DECISION_FILE][0]['k_star'] = files[W.DECISION_FILE][0]['k_star'] - 1
    if variant == 'tamper_exit':
        lines[n_nonopt]['all_optimal_k'] = False
    for fname, objs in files.items():
        with open(os.path.join(tmp, fname), 'w') as handle:
            if fname == W.DECISION_FILE:
                GRIO.dump(objs[0], handle, default=GRIO.json_default, indent=1, sort_keys=True)
            elif fname == W.CYCLE_FILE:
                for k in sorted(lines):
                    handle.write(GRIO.dumps(lines[k], default=GRIO.json_default) + '\n')
            else:
                for o in objs:
                    handle.write(GRIO.dumps(o, default=GRIO.json_default) + '\n')
    ref = st.reference if st.gated else {}
    rows, g_rows = [], []
    g_orig = ({r['cycle']: r for r in json.load(open(_abs(os.path.join(W.original_eval_dir(cell), 'g_s39_D.json'))))
               ['cycle_trajectory']} if st.gated else {})
    for k in sorted(lines):
        x = lines[k]
        if st.gated and k <= c['N_old']:
            row = dict(ref[k])
            g = dict(g_orig[k])
        else:
            row = {'cycle': k, 'local_solves_ok': True, 'recourse': x['net_operational_recourse'],
                   'gross_operational_cost': x['gross'], 'terminal_salvage_value': x['terminal_salvage_value'],
                   'objective_change_abs': abs(x['step'] or 0.0), 'objective_tolerance': 65000.0,
                   'objective_change_ratio': abs(x['step'] or 0.0) / 65000.0, 'cycle_convergence': x['boyd_k'],
                   'consecutive_converged_cycles': x['consecutive_converged_cycles_tracked'],
                   'boyd_all_pass': x['boyd_k'], 'boyd_stop': x['boyd_k'],
                   **{f: x['boyd_ratios'][f] for f in R.BOYD_RATIO_FIELDS},
                   **{f'boyd_{g_}_channel_pass': creep[k]['boyd'][g_]['channel_pass'] for g_ in R.CHANNELS},
                   **{f'rho_{g_}_after': x['rho']['rho_after'][g_] for g_ in R.CHANNELS},
                   **{f'rho_{g_}_action': x['rho']['actions'][g_] for g_ in R.CHANNELS},
                   'rho_freeze_active': True, 'efc_per_day_max': 1.0}
            g = {'cycle': k}
        for g_ in R.CHANNELS:
            for f, v in creep[k]['boyd'][g_].items():
                g[f'boyd_{g_}_{f}'] = v
        rows.append(row)
        g_rows.append(g)
    if variant == 'tamper_row_40':
        rows[39] = dict(rows[39], objective_change_abs=(rows[39]['objective_change_abs'] or 0.0) + 1.0)
    with open(os.path.join(tmp, 'per_cycle_record.jsonl'), 'w') as handle:
        for r in rows:
            handle.write(GRIO.dumps(r, default=GRIO.json_default) + '\n')
    with open(os.path.join(tmp, 'g_s39_D.json'), 'w') as handle:
        json.dump({'cycle_trajectory': g_rows}, handle)
    kk = K.K118
    n_e = len(kk.NODES) * len(kk.YEARS) * len(kk.DAYS) * kk.PERIODS
    with open(os.path.join(tmp, 'pf_entry_stride_s39_D.jsonl'), 'w') as handle:
        for k in sorted(lines):
            t = lines[k]['t_sum'] if variant != 'tamper_t_sum_30' or k != 30 else lines[k]['t_sum'] - 1.0
            gap = t / n_e
            ents = [{'node_id': n, 'year': str(y), 'day': d_, 'power_type': 'p', 'period': p, 'x_dso': 10.0 + gap,
                     'z_tso_current': 10.0, 'lambda_dso': 0.0, 's_base_dso': 100.0, 'rho_pf': 1.0, 'r': 0.0}
                    for n in kk.NODES for y in kk.YEARS for d_ in kk.DAYS for p in range(kk.PERIODS)]
            handle.write(json.dumps({'cycle': k, 'identity_holds': True, 'production_boyd_pf_r': 0.0,
                                     'entries': ents}) + '\n')
    last = max(lines)
    detail = {'t_tso_plus_t_dso_terminal': lines[last]['t_sum'], 'cycles_run': last,
              'interface_reporting_detail': {str(n): {str(y): {d_: {'periods': {str(p): {'price_per_mwh': 1.0}
                                                                                  for p in range(kk.PERIODS)}}
                                                               for d_ in kk.DAYS} for y in kk.YEARS}
                                             for n in kk.NODES},
              'interface_consensus_residual_per_dso': {str(n): {'periods': {f'{y}|{d_}|{p}': {'admm_block_weight': 1.0}
                                                                            for y in kk.YEARS for d_ in kk.DAYS
                                                                            for p in range(kk.PERIODS)},
                                                                'sum_pi_baseMVA_residual_weighted': 0.0}
                                                       for n in kk.NODES}}
    with open(os.path.join(tmp, 'interface_settlement_detail_s31c.json'), 'w') as handle:
        json.dump(detail, handle)
    rec = H._apply_settling_resettle_status({'cycles_run': last, 'settling_resettle_summary': st.summary(),
                                             'status': 'certified', 'barrier': False, 'barrier_cause': None,
                                             'certified_cost': 1.0, 'certification_cycle': last,
                                             'terminal_gross_operational_cost': 1.0})
    return rec, rows


def post_run_evaluator_self_tests():
    """The cell-keyed evaluators (G13, G15, G17, G18, G19, G21, G22, G23, G25, the cell report) on synthetic eval dirs
    built by the REAL wrappers with W135 declarations, tampered negative controls; the G9 replacement, the floor-year
    reader and the C2_calfade evaluator on synthetic / committed inputs; W132's scorer self-tests (reused)."""
    out = {}
    cases = (('pb_y2025_n5_v4', 'pass', None, True), ('pb_y2025_n5_v4', 'tampered_row_40', 'tamper_row_40', False),
             ('pb_y2025_n5_v4', 'tampered_t_sum_30', 'tamper_t_sum_30', False),
             ('pb_y2025_n5_v4', 'tampered_hold_flag', 'tamper_hold_flag', False),
             ('pb_y2025_n5_v4', 'tampered_decision', 'tamper_decision', False),
             ('e_c2', 'non_optimal_exit_is_not_a_lapse', 'nonopt', True),
             ('e_c2', 'tampered_exit_flag', 'tamper_exit', False),
             ('e_c3_unit', 'ungated_rule_cap', 'creep', True))
    for cell, name, variant, expect in cases:
        tmp = tempfile.mkdtemp(prefix='w135_selftest_')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rec, rows = _synthetic_run_dir(tmp, cell, variant)
            h_ok, _h = hold_checks(cell, tmp, rec)
            s_ok, _s = stopping_check(cell, rec, tmp)
            k_ok, _k = settling_replay_check(cell, tmp)
            f_ok, _f = L137.line_fields_check(tmp)
            c_ok, _c = L118.creep_capture_check(tmp, rec, rows)
            t_ok, t_d = L118.t_sum_check(tmp, os.path.relpath(os.path.join(tmp, 'pf_entry_stride_s39_D.jsonl'), REPO),
                                         os.path.relpath(os.path.join(tmp, 'interface_settlement_detail_s31c.json'),
                                                         REPO))
            o_ok, _o = overlap_check(cell, rec)
            l_ok, _l = L132.status_label_check(rec, tmp)
            rg = replay_gate_full(cell, tmp) if W.CELLS[cell]['gated'] else {'bitwise_through_k0': True}
            allg = {'holds': h_ok, 'stopping': s_ok, 'rule_replay': k_ok, 'line_fields': f_ok, 'creep': c_ok,
                    't_sum': t_ok, 'overlap': o_ok, 'status_label': l_ok, 'replay_full': rg['bitwise_through_k0']}
            if expect:
                ok = all(allg.values())
            elif variant == 'tamper_row_40':
                ok = (not rg['bitwise_through_k0']) and rg['first_divergence_cycle'] == 40 and all(
                    v for k_, v in allg.items() if k_ != 'replay_full')
            elif variant == 'tamper_t_sum_30':
                ok = (not t_ok) and t_d['max_abs_diff_vs_stride'] >= 0.99
            elif variant == 'tamper_hold_flag':
                ok = (not h_ok) and all(v for k_, v in allg.items() if k_ != 'holds')
            elif variant == 'tamper_exit':
                ok = (not k_ok) and all(v for k_, v in allg.items() if k_ != 'rule_replay')
            else:
                ok = (not k_ok) and (not s_ok)
            rep = cell_report(cell, tmp, rec) if expect else None
            out[name] = {'ok': bool(ok), 'gates': allg, 'expect_all_pass': expect,
                         'report_on_synthetic': ({k: rep.get(k) for k in ('status', 'k_star', 'branch', 'band_width',
                                                                          's_signed', 't_sum_end', 'k0_run',
                                                                          'n_vetoes', 'non_optimal_cycles',
                                                                          'label', 'arm')} if rep else None)}
        except Exception as error:  # noqa: BLE001
            out[name] = {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        finally:
            shutil.rmtree(tmp)
    # the G9 replacement on synthetic records (closed forms) + negative controls, every E cell
    g9 = {}
    for cell in W.E_CELLS:
        rec = _synthetic_variant_record(cell)
        decl = W.declaration_for(cell)
        pos, _d = W.variant_readback_gate(rec, decl)
        bad = copy.deepcopy(rec)
        bad['model_variant_readback_terminal']['per_node']['9']['readback']['floor_row_lower'] = 0.5
        neg_floor = not W.variant_readback_gate(bad, decl)[0]
        bad = copy.deepcopy(rec)
        bad['model_variant'] = W.arm_variant('e_c4' if cell != 'e_c4' else 'e_c2')
        neg_var = not W.variant_readback_gate(bad, decl)[0]
        bad = copy.deepcopy(rec)
        bad['model_variant_applied_in_child']['after']['per_ess'] = [{'soh_min': 0.5}] * 9
        neg_soh = not W.variant_readback_gate(bad, decl)[0]
        g9[cell] = {'ok': bool(pos and neg_floor and neg_var and neg_soh), 'positive': pos, 'floor_050_refused': neg_floor,
                    'other_variant_refused': neg_var, 'soh_min_after_apply_050_refused': neg_soh}
    out['G9v_variant_readback_gate_synthetic'] = {'ok': all(v['ok'] for v in g9.values()), 'cells': g9}
    # the C2_calfade evaluator on the committed unit against itself (reproduced) and against a planted difference
    try:
        unit = _read_jsonl(_abs(os.path.join(UNIT_REF_EVAL_DIR, 'per_cycle_record.jsonl')))
        eq = W.trajectory_equality(unit, unit, K.UNIT_SETTLED['k_star'])
        t = copy.deepcopy(unit)
        t[103]['gross_operational_cost'] = math.nextafter(t[103]['gross_operational_cost'], -math.inf)
        ne = W.trajectory_equality(t, unit, K.UNIT_SETTLED['k_star'])
        out['G28_evaluator_on_the_committed_unit'] = {'ok': bool(eq['reproduced'] and not ne['reproduced']
                                                                 and ne['first_difference']['cycle'] == 104),
                                                      'planted_first_difference': ne['first_difference']}
    except Exception as error:  # noqa: BLE001
        out['G28_evaluator_on_the_committed_unit'] = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
    # the floor-year reader on the committed unit (2035 at 172)
    try:
        fy, ok = W.floor_year_reading(_read_jsonl(_abs(os.path.join(UNIT_REF_EVAL_DIR, W.FLOOR_SIDECAR_FILE))),
                                      K.UNIT_SETTLED['k_star'])
        out['G26_floor_year_on_the_committed_unit'] = {'ok': bool(ok and fy['floor_year'] == 2035), 'reading': fy}
    except Exception as error:  # noqa: BLE001
        out['G26_floor_year_on_the_committed_unit'] = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
    out['item_e_scorer'] = item_e_scorer_self_tests()
    sc, sc_ok = L132.scorer_self_tests()
    out['w132_scorer_reused'] = {'ok': sc_ok, 'tests': sc}
    return out, all(v.get('ok') for v in out.values())


def _mechanism_050_inputs():
    """The 0.50-era rows as the committed mechanism table holds them (ageing_mechanism.json b5eca2a2, manifest-free:
    its sha256 is pinned and it is git-clean)."""
    am = _load(AGEING_MECHANISM['path'])
    if _sha(AGEING_MECHANISM['path']) != AGEING_MECHANISM['sha256'] or not _committed_clean(AGEING_MECHANISM['path']):
        raise RuntimeError('ageing_mechanism.json not as pinned / committed')
    return am


def item_e_scorer_self_tests():
    """The item-E scorer on the 0.50-ERA committed inputs reproduces the 0.50-era statements: the x C3 multipliers,
    I / value 1.21 (C3) and 1.08 (C2), the AE elasticities of the mechanism table (via the committed `pv` on the S46
    records' SoH_used), the mid-block x1.009, and the expert's references 0.105 / 0.085 / 0.039."""
    res = {}
    try:
        am = _mechanism_050_inputs()
        by = {t['variant']: t for t in am['table']}
        unit_I = next(cl['I_other'] for cl in claims_dataset()['claims'] if cl['claim_id'] == 'E:C3_unit_value_minus_I')
        vals = {'C3_unit': by['C3 (baseline)']['value_eur'], **{a: by[a]['value_eur'] for a in
                                                                ('C2', 'C4', 'C2_calfade', 'C3_midblock', 'no_ageing')}}
        xc3 = {a: vals[a] / vals['C3_unit'] for a in vals}
        res['x_c3_reproduced'] = {'ok': all(abs(xc3[a] - (by[a]['value_over_C3'] if a != 'C3_unit' else 1.0)) < 1e-12
                                            for a in xc3)
                                  and [round(xc3[a], 3) for a in ('C2', 'C4', 'C2_calfade', 'C3_midblock', 'no_ageing')]
                                  == [1.126, 1.058, 1.046, 1.009, 1.216], 'x_C3': xc3}
        band = {'C3_unit': unit_I / vals['C3_unit'], 'C2': unit_I / vals['C2']}
        res['I_over_value_reproduced'] = {'ok': round(band['C3_unit'], 2) == 1.21 and round(band['C2'], 2) == 1.08,
                                          'I_over_value': band}
        ae_ok = {}
        for a in ('C2', 'C4', 'C2_calfade', 'C3_midblock', 'no_ageing'):
            recp = os.path.join(W.original_eval_dir(W.CELL_OF_ARM[a]), 'evaluation_record.json')
            series, ok = ageing_series(_load(recp))
            ae = M46A.pv(series['soh_used'])
            eps = math.log(xc3[a]) / math.log(ae / by['C3 (baseline)']['AE'])
            ae_ok[a] = {'ok': bool(ok and abs(ae - by[a]['AE']) < 1e-12 and abs(eps - by[a]['elasticity']) < 1e-9),
                        'AE': ae, 'eps_AE': eps, 'committed_eps': by[a]['elasticity']}
        res['eps_AE_reproduced_via_committed_pv'] = {'ok': all(v['ok'] for v in ae_ok.values()), 'arms': ae_ok}
        eps050 = epsilon_050_references(am, _k_by_arm_now())
        want = PREDICTIONS['expert_addendum58_supplement']['references_050']
        res['expert_references_050_round_as_transcribed'] = {
            'ok': all(round(eps050[a], 3) == want[a] for a in ARMS_WITH_K_DIFFERENT_FROM_C3), 'eps050': eps050}
        # resolve_weighted reduces to W132's resolve when |c| = 1
        a_, b_ = {'status': 'certified', 'band': 10.0}, {'status': 'certified', 'band': 5.0}
        u = {'status': 'uncertified', 'gap': 100.0, 'slack': 40.0}
        ug = {'status': 'uncertified', 'gap': 100.0, 'slack': None}
        red = {'settled': resolve_weighted(16.0, 16.0, [(1.0, a_), (-1.0, b_)])['resolution']
               == L132.resolve(16.0, 16.0, (a_, b_))['resolution'],
               'uncertified': resolve_weighted(301.0, 301.0, [(1.0, a_), (-1.0, u)])['bar']
               == L132.resolve(301.0, 301.0, (a_, u))['bar'],
               'ungated_uncertified': resolve_weighted(1e9, 1e9, [(1.0, a_), (-1.0, ug)])['verdict']
               == 'indeterminate (slack undefined)',
               'weighted': resolve_weighted(1.0, 1.0, [(1.0, a_), (-2.0, b_), (0.5, a_)])['resolution'] == 25.0}
        res['weighted_resolution_reduces_to_w132'] = {'ok': all(red.values()), 'tests': red}
    except Exception as error:  # noqa: BLE001
        res['error'] = {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    return {'ok': all(v.get('ok') for v in res.values()), 'tests': res}


# ======================================================================================================================
#  checks state, verbatim, the v4 spec and its #38
# ======================================================================================================================
def _checks_file_state():
    rel = os.path.join(K.OUT_DIR_REL, K.OUT_FILE)
    doc = _load(rel) if os.path.isfile(_abs(rel)) else {}
    return {'path': rel, 'sha256': _sha(rel) if doc else None,
            'manifest': os.path.join(K.OUT_DIR_REL, K.OUT_MANIFEST),
            'committed_clean': _committed_clean(rel) if doc else False,
            'all_hold': doc.get('all_hold'), 'all_hold_including_typing_test': doc.get('all_hold_including_typing_test'),
            'section_M_holds': ((doc.get('sections') or {}).get('M') or {}).get('holds'),
            'typing_test': doc.get('W_bool_typing_test'),
            'guards_verify_0_failures': {k: v.get('verify_0_failures') for k, v in (doc.get('guards') or {}).items()},
            'code_sha256_at_check': doc.get('code_sha256')}


def _run_checks_inline():
    with contextlib.redirect_stdout(io.StringIO()):
        return K.run_all_checks(INLINE_SECTIONS)


def verbatim_check():
    files = sorted({f for f, _k in VERBATIM})
    texts = {f: _norm(open(_abs(f), encoding='utf-8').read()) for f in files}
    found = {f'{f}:{k}': _norm(v) in texts[f] for (f, k), v in VERBATIM.items()}
    return {'files': {f: {'sha256_at_freeze': _sha(f), 'git_state': L._git_state(f)} for f in files},
            'found_whitespace_normalised': found, 'all_found': all(found.values())}


def _hard_preconditions(tag, require_v4_38):
    """Refuse at once (before any key is computed) unless the harness routes a W135 declaration here and the v4 stage
    spec is frozen and committed; with `require_v4_38` (--run, the preconditions-only dry run included) ALSO unless the
    v4 campaign's cell #38 (l_195156fa) has its results and manifest committed -- THE LAUNCH GATE (W137: the freeze
    modes do not need #38, the launcher does)."""
    spec_ok, sd = K.v4_spec_state()
    done, d = K.v4_done()
    hr = harness_routed()
    if hr['ok'] and spec_ok and (done or not require_v4_38):
        return
    if not spec_ok:
        _log(f'[{tag} PRECONDITION FAILED] the v4 stage spec is not frozen / committed / naming l_195156fa last: '
             f'{sd["parts"]}')
    if require_v4_38 and not done:
        _log(f'[{tag} PRECONDITION FAILED] the v4 campaign cell #38 ({W.V4_LAST_CELL}) results / manifest not committed '
             f'(the launch gate of this extension): {d["parts"]}')
    if not hr['ok']:
        _log(f'[{tag} PRECONDITION FAILED] the harness is not the pre-W135 harness + {K.PATCH_REL} + the v4 branch, or '
             f'does not route a W135 declaration here: {hr}')
    _finish(1)


def _common_checks(require_v4_38):
    failures = []
    spec_ok, sd = K.v4_spec_state()
    if not spec_ok:
        failures.append(f'the v4 stage spec is not frozen / committed / naming l_195156fa last: {sd["parts"]}')
    if require_v4_38:
        done, d = K.v4_done()
        if not done:
            failures.append(f'the v4 campaign cell #38 ({W.V4_LAST_CELL}) results / manifest not committed: '
                            f'{d["parts"]}')
    hr = harness_routed()
    if not hr['ok']:
        failures.append(f'the harness is not the pre-W135 harness + the router patch, or does not route here: {hr}')
    for rel in EXTRA_CLEAN_FILES + tuple(H.PRODUCTION_FILES_TO_CHECK_CLEAN):
        if os.path.isfile(_abs(rel)) and not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    for cell in W.CELL_ORDER:
        rel = W.reference_path(cell)
        if _sha(rel) != W.CELLS[cell]['per_cycle_record_sha256'] or not _committed_clean(rel):
            failures.append(f'{cell} original record not as committed: {rel}')
    ext = extends()
    for rel in tuple(r for r in (ext['path'],) if r) + (W117['path'], AGEING_MECHANISM['path'], L132.W118_SUMMARY,
                                                          os.path.join(UNIT_REF_EVAL_DIR, 'per_cycle_record.jsonl'),
                                                          A28_REPORT, A30_REPORT):
        if not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    others = _own_process_alive()
    if others:
        failures.append(f'another copy of a W137 / W135 / W132 / W118 / W105 / W101 / W98 launcher is alive: {others}')
    return failures


def _checks_output_ok(failures):
    cf = _checks_file_state()
    if not (cf['all_hold_including_typing_test'] is True and cf['committed_clean'] and cf['section_M_holds'] is True
            and all(v == [] for v in (cf['guards_verify_0_failures'] or {'x': None}).values())):
        failures.append(f'the committed zero-solve checks output must exist, be clean and all hold: {cf}')
    for rel in K.CODE_PINNED_BY_CHECKS:
        if (cf.get('code_sha256_at_check') or {}).get(rel) != _sha(rel):
            failures.append(f'{rel} differs from the version the committed checks ran on')
    return cf


# ======================================================================================================================
#  --freeze-cells
# ======================================================================================================================
def freeze_cells(started):
    tag = 'W135-FREEZE-CELLS'
    _hard_preconditions(tag, require_v4_38=False)
    failures = _common_checks(require_v4_38=False)
    cf = _checks_output_ok(failures)
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline re-run of the zero-solve checks fails: {L132._failing_check_items(checks_inline)}')
    post_tests, post_ok = post_run_evaluator_self_tests()
    if not post_ok:
        failures.append(f'post-run evaluator / scorer self-tests fail: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    for cell in W.CELL_ORDER:
        failures += H.check_campaign_preconditions(campaign_root(cell), extra_clean_files=EXTRA_CLEAN_FILES)
        pre = pre_launch_assertion(cell)
        if not pre['holds']:
            failures.append(f'pre-launch assertion fails for {cell}: {pre["parts"]}')
    if os.path.isdir(_abs(ROOT_REL)) and any(f.startswith(SPEC_PREFIX) for f in os.listdir(_abs(ROOT_REL))):
        failures.append('a stage spec already exists: the cell specs are frozen BEFORE the stage spec')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    all_ok = True
    checks_pin = {'path': cf['path'], 'sha256': cf['sha256']}
    for cell in W.CELL_ORDER:
        pre = pre_launch_assertion(cell)
        c = W.CELLS[cell]
        extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage_text': STAGE_TEXT,
                 'stage_spec': ('frozen_s53_resettle_ext_spec_v1 (frozen AFTER this campaign spec) pins this spec by '
                                'its sha256 and holds the exact launch command'),
                 'extends': extends(), 'label': W.LABEL, 'cell': cell, 'item': c['item'], 'arm': c['arm'],
                 'model_variant': W.arm_variant(cell), 'claim_group': W.GROUP_OF_ITEM[c['item']],
                 'original': {'campaign_id': c['orig_campaign_id'], 'eval_key': c['orig_eval_key'],
                              'eval_dir': W.original_eval_dir(cell), 'role': ('replay reference (gated through k0)'
                                                                              if c['gated'] else
                                                                              'provenance only (0.50-era; ungated)')},
                 'expected_eval_key': pre['resettle_key'], 'objective_convention': DEFINITIONS['objective_convention'],
                 'solve_claim': {'parent': 'never solves (every launcher guard permitted=(), verify(0))',
                                 'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
                 'zero_solve_checks_output': checks_pin, 'pre_launch_assertion_at_freeze': pre,
                 'cell_order': list(W.CELL_ORDER), 'code_sha256': {rel: _sha(rel) for rel in CODE_PINNED}}
        spec_path, spec_sha, spec = H.freeze_campaign_spec(
            campaign_root(cell), CAMPAIGN_IDS[cell], entries(cell), configuration=configuration(cell),
            cap=W.spec_cap(cell), concurrency=CONCURRENCY,
            authority=[f'{BRIEF} Addendum 58 Supplement', f'{BRIEF} Addendum 58', 'Planner task W135 (W134 findings)',
                       f'{BRIEF} Addendum 59 and its Supplement', 'Planner task W137',
                       f'{extends()["path"]} (extended)'], required_consecutive_cycles=10, extra=extra)
        checks = validate_campaign_spec(cell, spec)
        pre_frozen = pre_launch_assertion(cell, spec)
        cap_ok, _cap = parent_capture_checklist(cell, spec)
        ok = all(checks.values()) and pre_frozen['holds'] and cap_ok
        all_ok = all_ok and ok
        e = spec['candidates'][0]
        _log(f'[{tag}] {cell}: frozen {os.path.relpath(spec_path, REPO)} sha256={spec_sha} eval_key={e["eval_key"]} '
             f'cap={spec["cap"]} checks={all(checks.values())} failing={[k for k, v in checks.items() if not v]} '
             f'pre-launch={pre_frozen["holds"]} capture-checklist={cap_ok}')
    _finish(0 if all_ok else 1, f'freeze-cells {"OK" if all_ok else "NOT OK"} -- next: commit, then --freeze-spec')


def cell_spec_state():
    out = {}
    for cell in W.CELL_ORDER:
        root = campaign_root(cell)
        files = sorted(f for f in os.listdir(root) if f.startswith('campaign_spec_')) if os.path.isdir(root) else []
        if len(files) != 1:
            out[cell] = {'error': f'campaign specs in {root}: {files}'}
            continue
        rel = os.path.relpath(os.path.join(root, files[0]), REPO)
        sha = _sha(rel)
        spec = _load(rel)
        checks = validate_campaign_spec(cell, spec)
        pre = pre_launch_assertion(cell, spec)
        out[cell] = {'path': rel, 'sha256': sha, 'name_carries_sha': files[0].endswith(f'_{sha[:8]}.json'),
                     'committed_clean': _committed_clean(rel), 'checks_all': all(checks.values()),
                     'checks_failing': [k for k, v in checks.items() if not v], 'pre_launch_holds': pre['holds'],
                     'eval_key': spec['candidates'][0]['eval_key'], 'eval_dir': spec['candidates'][0]['eval_dir'],
                     'harness_sha256': spec['harness']['sha256'], 'git_head': spec.get('git_head'),
                     'root_holds_only_the_spec': sorted(os.listdir(root)) == [files[0]]}
    return out


# ======================================================================================================================
#  --freeze-spec
# ======================================================================================================================
def _find_stage_spec():
    hits = sorted(f for f in os.listdir(_abs(ROOT_REL)) if f.startswith(SPEC_PREFIX) and f.endswith('.json')) \
        if os.path.isdir(_abs(ROOT_REL)) else []
    if len(hits) != 1:
        return None, None
    rel = os.path.join(ROOT_REL, hits[0])
    sha = _sha(rel)
    if hits[0] != f'{SPEC_PREFIX}{sha[:8]}.json':
        raise RuntimeError(f'stage spec file name does not carry its sha256 prefix: {rel} {sha}')
    return rel, sha


def load_stage_spec():
    rel, sha = _find_stage_spec()
    if rel is None:
        raise RuntimeError('frozen W135 re-settling extension stage spec not found')
    return rel, sha, _load(rel)


def stage_spec_content(checks_inline, cf, post_tests, verb, solver, prov, mem, wall, specs, o_section, dataset, points,
                       refs, ref_inputs, am_eps050, k_by_arm):
    code = {rel: _sha(rel) for rel in CODE_PINNED}
    production = {rel: _sha(rel) for rel in H.PRODUCTION_FILES_TO_CHECK_CLEAN if os.path.isfile(_abs(rel))}
    cells = {}
    for i, cell in enumerate(W.CELL_ORDER):
        c = W.CELLS[cell]
        o = o_section['cells'][cell]
        s = specs[cell]
        cells[cell] = {
            'launch_index_1_based': i + 1, 'item': c['item'], 'arm': c['arm'], 'model_variant': W.arm_variant(cell),
            'k_closed_form': k_by_arm.get(c['arm']) if c['arm'] else None,
            'claim_group': W.GROUP_OF_ITEM[c['item']], 'gated': c['gated'],
            'original': {'campaign_id': c['orig_campaign_id'], 'eval_key': c['orig_eval_key'],
                         'eval_dir': W.original_eval_dir(cell), 'label': c['orig_label'],
                         'per_cycle_record': o['per_cycle_record'], 'per_cycle_record_sha256': o['per_cycle_record_sha256'],
                         'git_head': o['original_git_head'], 'cycles': o['orig_cycles']},
            'candidate_key': K.candidate_of(cell)[0], 'N_old': c['N_old'], 'k0_original_first_residual_pass': c['k0'],
            'original_lapses_after_k0': c['original_lapses_after_k0'], 'cap_rule': W.cap_rule(cell),
            'spec_cap': W.spec_cap(cell), 'cap_ceiling': c['cap_ceiling'],
            'v2_run_stays': c.get('v2_run_stays'),
            'declaration': W.declaration_for(cell), 'campaign_id': CAMPAIGN_IDS[cell],
            'campaign_root': campaign_root_rel(cell), 'configuration': configuration(cell), 'keys': expected_keys(cell),
            'campaign_spec': {k: s[k] for k in ('path', 'sha256', 'eval_key', 'eval_dir', 'harness_sha256', 'git_head')},
            'launch_command': launch_command(cell, s['sha256']),
            'preconditions_only_command': launch_command(cell, s['sha256'], preconditions_only=True),
            'expected_wall_time': wall['per_cell'][cell]}
    return {
        'schema': 'p515_s53_resettle_ext_spec_v1', 'series': SPEC_SERIES, 'version': SPEC_VERSION,
        'stage_text': STAGE_TEXT, 'extends': extends(), 'predecessor': None,
        'predecessor_note': ('first version of this series; it EXTENDS (does not replace) the v4 stage spec '
                             'frozen_s53_resettle_spec_v4, whose path and sha256 are recorded in `extends` (W137; the '
                             'code was built in W135 against v3 139d1e62 and re-targeted to v4 before this freeze)'),
        'authority': [f'{BRIEF} Addendum 59 and its Supplement (criterion v4; W135 router patch in the v4 freeze)',
                      f'{BRIEF} Addendum 58 Supplement (ageing arms at 0.70, option (b); pb re-run)',
                      f'{BRIEF} Addendum 58 (Rulings 1-3; order)', 'TASKS.md Addendum 58 section',
                      'W134 (arm definitions, key, G9 replacement, floor-year capture, walls), as transcribed in '
                      'Planner task W135', 'Planner task W135', 'Planner task W137'],
        'frozen_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'objective_convention': DEFINITIONS['objective_convention'],
        'pins': {'code_sha256': code, 'production_sha256': production, 'solver': solver,
                 'harness': {'pre_w135': K.PRE_W135_HARNESS, 'post_patch_sha256': K.HARNESS_POST_PATCH_SHA256,
                             'post_v4_branch_sha256': K.HARNESS_POST_V4_SHA256,
                             'patch': {'path': K.PATCH_REL, 'sha256': _sha(K.PATCH_REL)},
                             'v4_branch_lines': list(K.V4_BRANCH_LINES)},
                 'zero_solve_checks_output': {'path': cf['path'], 'sha256': cf['sha256'], 'manifest': cf['manifest']},
                 'w117': {'path': W117['path'], 'sha256': W117['sha256']},
                 'ageing_mechanism': {**AGEING_MECHANISM},
                 'references_inputs_sha256': ref_inputs,
                 'unit_reference_3f084f2f': {'eval_dir': UNIT_REF_EVAL_DIR,
                                             'per_cycle_record_sha256': K.UNIT_SETTLED['per_cycle_record_sha256']},
                 'v4_cell_38_launch_gate': dict(zip(('results', 'manifest'), W.v4_last_cell_files_rel())),
                 'ess_params_file': {'path': W.ESS_PARAMS_REL, 'sha256': W.ESS_PARAMS_SHA256},
                 'campaign_specs': {c: {'path': specs[c]['path'], 'sha256': specs[c]['sha256']} for c in W.CELL_ORDER}},
        'production_since_originals': prov,
        'cells': cells, 'cell_order': list(W.CELL_ORDER),
        'launch_order': {'order': list(W.CELL_ORDER),
                         'enforced': ('the v4 campaign #38 results committed (the launch gate; --run refuses '
                                      'before), then every earlier W135 cell has results before --run'),
                         'one_cell_per_call': True},
        'launch_commands': {c: cells[c]['launch_command'] for c in W.CELL_ORDER},
        'summarize_commands_at_the_completion_points': {p['after_cell']: summarize_command(p['after_cell'])
                                                        for p in points['report_points']},
        'claim_completion_points': points,
        'inputs_in_force_now': o_section['inputs_now'], 'references': refs,
        'configuration': {'current_production': ('case file (AA keep_memory declared) + ESS ageing baseline C2 declared '
                                                 '(minimum SoH 0.70) + tight tail {enabled True, compl_inf_tol 1e-6} '
                                                 'declared; E entries + model_variant (arm)'),
                          'persistence': 'persist_certified_models True, hull_polish False, no reference',
                          'concurrency': CONCURRENCY, 'option_b_release_solution_bookkeeping': 'absent',
                          'checked_field_by_field': 'validate_campaign_spec at --freeze-cells, --freeze-spec and --run'},
        'stop_rule': {'module': 'settling_criterion_v4', 'class': 'settling_criterion_v4.SettlingRuleV4',
                      'version': SC4.VERSION, 'constants': SC4.constants(W.P_MAX), 'readings': SC4.READINGS,
                      'sub_test_reads': SC4.SUB_TEST_READS,
                      'as_v4': 'the v4 declaration settling_rule (V4.settling_rule_declaration()), unchanged',
                      'caps': {'E (ungated)': 'min(k0_run + 109, 300) (dynamic)', 'pb (gated)': 'N_old + 100 = 219'}},
        'replay_gate': {'applies_to': list(W.GATED_CELLS), 'skipped_for': list(W.UNGATED_CELLS),
                        'as_w132': 'in-cycle bitwise through k0 (abort on the first difference); G19 post-run'},
        'holds_after_first_residual_pass': {'AA': 'off', 'tail': 'on', 'rho': 'frozen', 'same_as': 'W101 / W118 / W132'},
        'model_variant': {'arms': W.ARMS, 'k_closed_form_by_arm': k_by_arm,
                          'floor_row_lower_expected': W.FLOOR_ROW_LOWER_EXPECTED,
                          'applied': ('in the child by the harness configuration hook (apply_model_variant) after the '
                                      'file-vs-declaration check; read back from probe models before any solve (the '
                                      'harness refuses on mismatch) and from clones of the run\'s own ESSO models '
                                      'after the run'),
                          'g9_replacement': W.variant_readback_gate.__doc__},
        'floor_year': {'reader': W.floor_year_reading.__doc__, 'capture': W.FLOOR_SIDECAR_FILE},
        'definitions': DEFINITIONS,
        'scorer': {'claims_dataset': dataset, 'formulas': DEFINITIONS['claims'],
                   'uncertified_form': DEFINITIONS['uncertified_form'],
                   'restated_statements': RESTATED_STATEMENTS,
                   'epsilon_050_references_recomputed': am_eps050,
                   'functions': ['score_item_e', 'resolve_weighted', 'epsilon_050_references', 'cell_report',
                                 'L132.score_claim', 'L132.resolve', 'L132.view_from_report', 'reference_views']},
        'predictions_recorded_before_any_run': PREDICTIONS,
        'gates': {'as_w132': ('G1-G7, G11, G13-G16, G20-G25 as W132 (cell-keyed ones restated for the W135 table); '
                              'G8 on production\'s certificate, G17 / G18 on the v4 rule records (as W137)'),
                  'G9': 'pb only (ESS ageing readback)', 'G9v': 'E only: the G9 replacement',
                  'G19': 'pb only', 'G26': 'floor sidecar at the end cycle', 'G27': 'E: terminal ageing trajectory',
                  'G28': 'e_c2_calfade: reproduces 3f084f2f through 172 (exit 3 on failure)', 'scope': GATE_SCOPE,
                  'non_stopping': list(NON_STOPPING_GATES), 'prediction_gates': list(PREDICTION_GATES),
                  'post_run_evaluator_and_scorer_self_tests': post_tests},
        'zero_solve_checks': {'script': 'p515_s53_w135_resettle_ext_checks.py', 'committed_output': cf,
                              'inline_rerun_at_freeze': {
                                  'all_hold': checks_inline['all_hold'],
                                  'per_section': {k: v['holds'] for k, v in checks_inline['sections'].items()},
                                  'not_rerun_inline': ['M (builds models; in the committed output)', 'W (typing)']}},
        'verbatim_text': {'quotes': {f'{f}:{k}': v for (f, k), v in VERBATIM.items()}, 'check': verb},
        'labelling_and_identity': {'label': W.LABEL, 'model_variant_label': H.MODEL_VARIANT_LABEL,
                                   'evaluation_key': ('sha256({base_evaluation_key (carries model_variant for E), '
                                                      'settling_resettle (W135 declaration)}); no key formula change; '
                                                      'every other key byte-identical to the pre-W135 harness (checks '
                                                      'K)')},
        'harness_change': ('p515_s44_campaign_harness.py: the prepared patch (router commit 1, W137): a W135 '
                           'declaration dispatches to p515_s53_w135_resettle_ext_hooks; then the v4 branch (router '
                           'commit 2); every existing branch identical (checks D)'),
        'memory_preflight': {'rule': 'W86 rule at concurrency 1', 'measured_at_freeze_non_gating': mem},
        'solve_profile_declared': {'parent': 'never solves: every launcher guard permitted=(), verify(0)',
                                   'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
        'expected_wall_time': wall,
        'walls': {'this_spec_expected_h': wall['total_expected_h'], 'this_spec_worst_case_h': wall['total_worst_case_h'],
                  'e_cells_expected_h': wall['e_cells_expected_h'], 'e_cells_worst_case_h': wall['e_cells_worst_case_h'],
                  'w134_transcribed': PREDICTIONS['walls']['statement'],
                  'single_run_over_4h': {c: wall['per_cell'][c]['worst_case_h'] for c in W.CELL_ORDER
                                         if wall['per_cell'][c]['worst_case_h'] > 4.0}},
        'dry_run_command': launch_command(W.CELL_ORDER[0], specs[W.CELL_ORDER[0]]['sha256'], preconditions_only=True),
        'zero_solve_smoke': ('no zero-solve path exercises _child_real end to end; the wiring is exercised by checks H '
                             '(the real wrappers; the real install with the patched dispatch), M (the child\'s own '
                             'configuration hook on freshly built models); THE FIRST REAL CYCLE OF THE FIRST LAUNCH IS '
                             'THE SMOKE'),
    }


def freeze_spec(started):
    tag = 'W135-SPEC'
    _hard_preconditions(tag, require_v4_38=False)
    failures = _common_checks(require_v4_38=False)
    os.makedirs(_abs(ROOT_REL), exist_ok=True)
    existing = sorted(f for f in os.listdir(_abs(ROOT_REL)) if f.startswith(SPEC_PREFIX))
    if existing:
        failures.append(f'the stage spec already exists (write-once): {existing}')
    cf = _checks_output_ok(failures)
    specs = cell_spec_state()
    for cell, s in specs.items():
        if 'error' in s or not (s['name_carries_sha'] and s['committed_clean'] and s['checks_all']
                                and s['pre_launch_holds'] and s['root_holds_only_the_spec']):
            failures.append(f'campaign spec of {cell} not frozen / committed / valid: {s}')
        elif s['harness_sha256'] != H.sha256_file(H.HARNESS_PATH):
            failures.append(f'{cell}: harness changed since its campaign spec froze')
        elif _load(s['path'])['extra'].get('code_sha256') != {rel: _sha(rel) for rel in CODE_PINNED}:
            failures.append(f'{cell}: code changed since its campaign spec froze')
    prov = K.production_since_originals(CODE_PINNED)
    if not prov['ok']:
        failures.append(f'uncommitted files this run uses: {prov["uncommitted_files_this_run_uses"]}')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline re-run of the zero-solve checks fails: {L132._failing_check_items(checks_inline)}')
    post_tests, post_ok = post_run_evaluator_self_tests()
    if not post_ok:
        failures.append(f'post-run evaluator / scorer self-tests fail: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    verb = verbatim_check()
    if not verb['all_found']:
        failures.append(f'verbatim quotes not found: {verb["found_whitespace_normalised"]}')
    solver = W101L.solver_check()
    if not solver['ok']:
        failures.append(f'solver path: {solver}')
    if failures:
        for f in failures:
            _log(f'[{tag} PRECONDITION FAILED] {f}')
        _finish(1)
    o_section = checks_inline['sections']['O']['result']
    dataset = claims_dataset()
    points = completion_points(dataset)
    refs, ref_inputs = reference_views()
    mem = L.memory_preflight(1)
    wall = wall_time_estimate()
    k_by_arm = _k_by_arm_now()
    am_eps050 = epsilon_050_references(_mechanism_050_inputs(), k_by_arm)
    content = stage_spec_content(checks_inline, cf, post_tests, verb, solver, prov, mem, wall, specs, o_section,
                                 dataset, points, refs, ref_inputs, am_eps050, k_by_arm)
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(ROOT_REL, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError('the stage spec\'s written bytes do not hash to its name')
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] wrote {rel} sha256={sha} (extends the v4 stage spec {(extends()["sha256"] or "")[:8]})')
    _log(f"[{tag}] checks per section: { {k: v['holds'] for k, v in checks_inline['sections'].items()} }")
    _log(f"[{tag}] post-run evaluator / scorer self-tests: { {k: v.get('ok') for k, v in post_tests.items()} }")
    _log(f'[{tag}] claims scored: {dataset["n_claims"]} (skipped {dataset["skipped_not_in_the_set"]})')
    for p in points['report_points']:
        _log(f"[{tag}] completion point after #{p['after_cell_index_1_based']} {p['after_cell']}: {p['items_complete']}")
    for cell in W.CELL_ORDER:
        w = wall['per_cell'][cell]
        _log(f"[{tag}] {cell}: cap {W.spec_cap(cell)} expected {w['expected_h']:.2f} h worst {w['worst_case_h']:.2f} h; "
             f"LAUNCH: {content['cells'][cell]['launch_command']}")
    _log(f"[{tag}] total expected {wall['total_expected_h']:.2f} h, worst {wall['total_worst_case_h']:.2f} h "
         f"(W134: {PREDICTIONS['walls']['statement']})")
    _finish(0, '-- next: commit, then the dry run on the first cell')


# ======================================================================================================================
#  --run
# ======================================================================================================================
def run(started, cell, spec_sha256, preconditions_only=False):
    tag = f'W135-RUN-{cell}' + ('-PRECONDITIONS-ONLY' if preconditions_only else '')
    _hard_preconditions(tag, require_v4_38=True)     # THE LAUNCH GATE: the v4 #38 committed
    failures = _common_checks(require_v4_38=True)
    ss_rel, ss_sha, ss = load_stage_spec()
    if not _committed_clean(ss_rel):
        failures.append('the stage spec is not committed / clean')
    if ss['pins']['code_sha256'] != {rel: _sha(rel) for rel in ss['pins']['code_sha256']}:
        failures.append('code changed since the stage spec froze')
    pin = ss['pins']['campaign_specs'].get(cell) or {}
    if pin.get('sha256') != spec_sha256:
        failures.append(f'the stage spec pins {pin.get("sha256")} for {cell}, not {spec_sha256}')
    idx = W.CELL_ORDER.index(cell)
    for prev in W.CELL_ORDER[:idx]:
        if not os.path.isfile(os.path.join(campaign_root(prev), RESULTS_FILE)):
            failures.append(f'launch order: {prev} (#{W.CELL_ORDER.index(prev) + 1}) has no results yet')
    root = campaign_root(cell)
    spec_path, spec = H.load_frozen_spec(root, spec_sha256)
    failures += [f for f in H.check_campaign_preconditions(root, extra_clean_files=EXTRA_CLEAN_FILES)
                 if f != f'campaign root already exists (write-once): {root}']
    if sorted(os.listdir(root)) != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only its frozen spec; holds {sorted(os.listdir(root))}')
    if not _committed_clean(os.path.relpath(spec_path, REPO)):
        failures.append('campaign spec not committed / clean')
    checks = validate_campaign_spec(cell, spec)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    for what, pinned, now in (('harness', spec['harness']['sha256'], H.sha256_file(H.HARNESS_PATH)),
                              ('script', spec['extra'].get('campaign_script_sha256'), _sha(SCRIPT_NAME)),
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE)),
                              ('ESS parameters file', (spec['configuration'].get('ess_params_file') or {}).get('sha256'),
                               _sha(W.ESS_PARAMS_REL))):
        if pinned != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'the zero-solve checks do not all hold now -- failing: '
                        f'{L132._failing_check_items(checks_inline)}')
    pre = pre_launch_assertion(cell, spec)
    if not pre['holds']:
        failures.append(f'pre-launch assertion fails on the frozen spec: {pre["parts"]}')
    cap_ok, cap_checks = parent_capture_checklist(cell, spec)
    if not cap_ok:
        failures.append(f'parent-side capture checklist fails: {cap_checks}')
    solver = W101L.solver_check()
    if not solver['ok'] or solver['sha256'] != ss['pins']['solver']['sha256']:
        failures.append(f'solver path / binary differs: {solver}')
    mem = L.memory_preflight(1)
    _log(f"[{tag}] memory preflight: available {mem.get('available_gib')} GiB required {mem['required_gib']:.2f} GiB -> "
         f"{'PASS' if mem['pass'] else 'REFUSE'}")
    if not mem['pass']:
        failures.append(f"memory preflight REFUSED: {mem.get('available_gib')} < {mem['required_gib']}")
    if failures:
        for fl in failures:
            _log(f'[{tag} PRECONDITION FAILED] {fl}')
        _finish(1)
    entry = spec['candidates'][0]
    if preconditions_only:
        _log(f"[{tag}] every precondition holds: campaign spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256} "
             f"(pinned by the stage spec {ss_rel} sha256={ss_sha}); launch index {idx + 1} of {len(W.CELL_ORDER)}; "
             f"the v4 #38 committed; harness routed ({harness_routed()['harness_sha256'][:8]}); checks per section "
             f"{({k: v['holds'] for k, v in checks_inline['sections'].items()})}; pre-launch parts {pre['parts']}; parent "
             f"capture checklist {len(cap_checks)} items all True; eval_key {entry['eval_key']}; cap {spec['cap']}; "
             f"model_variant {entry.get('model_variant')}; solver {solver['resolved'].get('NLP_SOLVER_PATH')} "
             f"sha256={solver['sha256']}")
        _log(f'[{tag}] STOPPED before the campaign lock and the child (no lock taken, no evaluation, zero solves)')
        _finish(0, 'preconditions-only OK')
    lock = H.acquire_campaign_lock(spec['campaign_id'], spec_sha256)
    _log(f"[{tag}] {STAGE_TEXT}; cell {cell} (#{idx + 1}) eval_key {entry['eval_key']}; cap {spec['cap']}; lock {lock}")
    try:
        ctx = H.CampaignContext(root, spec_path, spec_sha256, spec, log=_log)
        H.evaluate.last_batch_info = {}
        records = H.evaluate([entry['label']], ctx)
        batch = dict(getattr(H.evaluate, 'last_batch_info', {}) or {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
    eval_dir = os.path.join(root, 'evals', entry['eval_dir'])
    try:
        gates, detail, rec = cell_gates(cell, entry, eval_dir)
    except Exception as error:  # noqa: BLE001 -- recorded; the gates FAIL
        gates, detail, rec = {'cell_gates_ran': False}, {'error': f'{type(error).__name__}: {error}',
                                                         'traceback': traceback.format_exc()}, {}
    report = None
    try:
        if _decision(eval_dir) is not None and os.path.isfile(os.path.join(eval_dir, 'per_cycle_record.jsonl')):
            report = cell_report(cell, eval_dir, rec)
    except Exception as error:  # noqa: BLE001
        report = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    summ = (rec or {}).get('settling_resettle_summary') or {}
    points = ss['claim_completion_points']['report_points']
    point = next((p for p in points if p['after_cell'] == cell), None)
    stopping = [k for k, v in gates.items() if not v and k not in NON_STOPPING_GATES]
    prediction_failed = [k for k in PREDICTION_GATES if k in gates and not gates[k]]
    g = guards_verify()
    results = {'stage': STAGE_TEXT, 'cell': cell, 'launch_index_1_based': idx + 1, 'utc': _utc(),
               'git_head_at_run': H._git(['rev-parse', 'HEAD']),
               'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
               'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': ss['objective_convention'],
               'eval_key': entry['eval_key'], 'candidate_key': entry['key'], 'arm': W.CELLS[cell]['arm'],
               'model_variant': entry.get('model_variant'), 'original_eval_key': W.CELLS[cell]['orig_eval_key'],
               'replay_divergence_abort': summ.get('replay_first_divergence'),
               'gates': gates, 'gates_pass': all(gates.values()), 'stopping_gate_failures': stopping,
               'prediction_gate_failures': prediction_failed,
               'gate_detail': detail, 'cell_report': report, 'claim_completion_point': point,
               'pre_launch_assertion': pre, 'parent_capture_checklist': cap_checks, 'memory_preflight_at_run': mem,
               'solver': solver, 'batch_info': batch, 'parent_view': (records[0] or {}).get('parent_view'),
               'guards': g, 'wall_clock_s': time.time() - started}
    H._write_once_json(os.path.join(root, RESULTS_FILE), results)
    H._write_once_json(os.path.join(root, MANIFEST_FILE), W9._manifest_of([root]))
    if summ.get('replay_first_divergence'):
        _log(f"[{tag}] REPLAY DIVERGED -- cell ABORTED: {summ['replay_first_divergence']}")
    _log(f"[{tag}] gates {gates}")
    if isinstance(report, dict) and 'error' not in report:
        _log(f"[{tag}] settling v4: status {report.get('status')} k* {report.get('k_star')} branch {report.get('branch')} "
             f"band_width {report.get('band_width')} Q_end {report.get('Q_end')} t_sum(end) {report.get('t_sum_end')} "
             f"k0_run {report.get('k0_run')} non-Optimal cycles {report.get('non_optimal_cycles')} cycles "
             f"{report.get('cycles_run')} floor_year {report.get('floor_year')} AE {report.get('AE')} record status "
             f"{report.get('record_status')}")
    if prediction_failed:
        _log(f'[{tag}] PLANNER VALIDATION FAILED: {prediction_failed} -- {detail.get("G28")} -- STOP FOR THE PLANNER '
             f'before the next cell (exit 3)')
    if point is not None:
        _log(f"[{tag}] CLAIM-COMPLETION POINT after #{point['after_cell_index_1_based']} {cell}: items "
             f"{point['items_complete']} complete -- report (Addendum 58). Scorer: {summarize_command(cell)}")
    ok_rest = not stopping and _guards_ok(g) and isinstance(report, dict) and 'error' not in report
    code = (3 if prediction_failed else 0) if ok_rest else 1
    _finish(code, f'wall={time.time() - started:.0f}s')


# ======================================================================================================================
#  --summarize
# ======================================================================================================================
def summarize(started, after_cell):
    tag = f'W135-SUMMARY-after-{after_cell}'
    ss_rel, ss_sha, ss = load_stage_spec()
    idx = W.CELL_ORDER.index(after_cell)
    reports, missing, inputs = {}, [], {}
    for cell in W.CELL_ORDER[:idx + 1]:
        path = os.path.join(campaign_root(cell), RESULTS_FILE)
        rel = os.path.relpath(path, REPO)
        if not os.path.isfile(path) or not _committed_clean(rel):
            missing.append(cell)
            continue
        inputs[rel] = _sha(rel)
        reports[cell] = (json.load(open(path)).get('cell_report') or {})
    out_rel = os.path.join(ROOT_REL, f'w135_summary_after_{idx + 1:02d}_{after_cell}.json')
    if missing or os.path.exists(_abs(out_rel)):
        _log(f'[{tag} PRECONDITION FAILED] cells without committed results {missing} or the summary exists')
        _finish(1)
    refs, ref_inputs = reference_views()
    if ref_inputs != ss['pins']['references_inputs_sha256']:
        _log(f'[{tag} PRECONDITION FAILED] the references changed since the stage spec froze')
        _finish(1)
    am = _mechanism_050_inputs()
    dataset = ss['scorer']['claims_dataset']
    views = {c: r.get('view') or L132.view_from_report(r) for c, r in reports.items()}
    k_by_arm = ss['model_variant']['k_closed_form_by_arm']
    item_e = score_item_e(views, reports, refs, dataset, am, k_by_arm)
    pb_scored = []
    for cl in dataset['claims']:
        if cl['claim_id'] != PB_CLAIM_ID:
            continue

        def view(side):
            s = cl[side]
            return refs[s['ref']] if s['source'] == 'reference' else views.get(s['cell'], {})
        pb_scored.append(L132.score_claim(cl, view('ref'), view('other')))
    doc = {'stage': STAGE_TEXT, 'utc': _utc(), 'after_cell': after_cell, 'after_cell_index_1_based': idx + 1,
           'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': DEFINITIONS['objective_convention'],
           'reports': reports, 'references': refs, 'item_E': item_e, 'phase_b_claim': pb_scored,
           'claims_complete': sorted([cid for cid, s in item_e['claims'].items()
                                      if not str(s.get('verdict', '')).startswith('not scored')]
                                     + [s['claim_id'] for s in pb_scored
                                        if not str(s.get('verdict', '')).startswith('not scored')]),
           'restated_statements_definitions': RESTATED_STATEMENTS, 'definitions': DEFINITIONS,
           'predictions': PREDICTIONS,
           'c2_calfade_consistency': ((json.load(open(os.path.join(campaign_root('e_c2_calfade'), RESULTS_FILE)))
                                       .get('gate_detail') or {}).get('G28')
                                      if 'e_c2_calfade' in reports else 'no result yet'),
           'inputs_sha256': {**inputs, **ref_inputs, AGEING_MECHANISM['path']: _sha(AGEING_MECHANISM['path'])}}
    H._write_once_json(_abs(out_rel), doc)
    man = os.path.join(ROOT_REL, f'w135_summary_after_{idx + 1:02d}_{after_cell}_manifest_sha256.json')
    H._write_once_json(_abs(man), {out_rel: _sha(out_rel), **doc['inputs_sha256']})
    _log(f'[{tag}] wrote {out_rel}; claims complete {len(doc["claims_complete"])}')
    _finish(0)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-cells', action='store_true')
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument('--cell', choices=W.CELL_ORDER, default=None)
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--after-cell', choices=W.CELL_ORDER, default=None)
    parser.add_argument('--preconditions-only', action='store_true',
                        help='with --run: every --run precondition, then stop before the lock and the child')
    args = parser.parse_args()
    if args.preconditions_only and not args.run:
        parser.error('--preconditions-only requires --run')
    started = time.time()
    try:
        if args.freeze_cells:
            freeze_cells(started)
        elif args.freeze_spec:
            freeze_spec(started)
        elif args.summarize:
            if not args.after_cell:
                parser.error('--summarize requires --after-cell')
            summarize(started, args.after_cell)
        else:
            if not (args.spec_sha256 and args.cell):
                parser.error('--run requires --cell and --spec-sha256')
            run(started, args.cell, args.spec_sha256, preconditions_only=args.preconditions_only)
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        _finish(1, 'EXCEPTION')


if __name__ == '__main__':
    main()
