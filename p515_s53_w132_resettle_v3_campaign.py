"""
P5.15 Addendum 58, Planner task W132 -- the v3 RE-SETTLING CAMPAIGN (claim groups 1, 2 and 4; 38 cells): the 38 per-cell
campaign freezes, the frozen stage spec `frozen_s53_resettle_spec_v3_<sha8>.json` (the resettle series, predecessor v2
fc791891), the per-cell run (one cell per call, priority order enforced), and the zero-solve scorer. BUILT AND FROZEN IN
W132; NO RUN IS LAUNCHED BY THE WORKER.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 58 (Ruling 1: the F2 reporting form and the 3 x max(gap, slack) rule;
Ruling 2: a certifying cycle requires "Optimal Solution Found" on every block; Ruling 3: gross primary, net of salvage
beside; the machine-time order and the reports at each claim's completion) and Addendum 57 (the gap clause, the amended
monotone branch, the campaign rulings); TASKS.md Addendum 58 section (the Advisor's design review of the remaining cells
and the Planner rulings (i)-(iv), commit 82fa1b61); W131 (7a5c5717); Planner task W132.

THE CELLS (`p515_s53_w132_resettle_v3_hooks.CELLS`, launch order CELL_ORDER = the priority order):
  group 1  B 2a0ba8b2 0dd237f0 4649234b; D c52e1670 4a82a64a 3632b0ae 36686489 d3709599 a12d95a2 f759dd48 c7fee8be
           9246ed01 (S47 a1a_baseline; C2; m = 1; gated)
  group 2  H aa8a76d7 f9eae48f (m 1.5) 50dea31c 74eda68d (m 2); I 5a6a88b4; J f3aa335e a11d7966 5f3cccb4 (gated)
  group 4  C 156ce2d1 6597a79d (gated); G 37b5c499 47dce43c 48749148 9abf31d4 (UNGATED first C2 evaluations);
           L 7b199ef9 e1da0984 0ee93aca df1a5525 2ab0ce2d 8e4c220e 7db09f6c 76c78064 45aa25a6 7c455554 b2251bc5
           195156fa (gated; m = 2)
Group 3 (ageing E) is NOT in this spec. 7aa017f0 / bd504ecf are covered by the settled references (not re-run).

MODES (repo root, canonical interpreter; attached, ALONE, both streams captured, never detached):
  --freeze-cells                        ZERO SOLVES. The 38 campaign specs (write-once), in the W132 stage root.
  --freeze-spec                         ZERO SOLVES. The stage spec (write-once, named by its sha256): pins the 38
                                        committed campaign specs and holds the EXACT launch command of every cell.
  --run --cell C --spec-sha256 S        THE RUN OF ONE CELL (NOT RUN IN W132). Priority order enforced (every earlier
                                        cell has results). Preconditions (the stage spec pins S for C; the zero-solve
                                        checks re-run inline; the pre-launch assertion; the parent-side capture
                                        checklist; memory; solver; own-process check), then H.evaluate on the one entry,
                                        the gates, the cell report; results + manifest; the claim-completion point, if
                                        this cell is one.
  --run ... --preconditions-only        ZERO SOLVES. Every --run precondition, then STOP before the lock and the child.
  --summarize --after-cell C            ZERO SOLVES. The scorer over every cell with committed results up to C: the
                                        per-claim differences (gross and Q_cc; net beside on G rows), the D fit, the
                                        predictions; write-once, named after C.

Exit codes: 0 done (every gate holds, or only G6 fails -- G6 does not stop the campaign, Planner ruling at W128); 1 any
other gate / harness / guard / precondition failure.
"""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import statistics
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

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W132 v3 re-settling launcher (never solves)').install()

import numpy as np  # noqa: E402

import p515_s53_w101_srp1_continuation_campaign as W101L  # noqa: E402 -- generic gates (arms its guards)
import p515_s53_w118_resettle_campaign as L118  # noqa: E402 -- t_sum / creep evaluators (arms its guards)
import p515_s53_w132_resettle_v3_checks as K  # noqa: E402 -- the zero-solve checks (arms its guards)
import p515_s53_w132_resettle_v3_hooks as V  # noqa: E402
import p515_s53_w131_prefreeze_diagnostics as W131  # noqa: E402 -- final-attempt readers for G24 (arms its guard)
import settling_criterion_v3 as SC3  # noqa: E402
import gate_result_io as GRIO  # noqa: E402

H, L, X, W9, W98L = W101L.H, W101L.L, W101L.X, W101L.W9, W101L.W98L
R = V.R


def _dedupe_guards(pairs):
    seen, out = set(), []
    for name, guard in pairs:
        if id(guard) not in seen:
            seen.add(id(guard))
            out.append((name, guard))
    return tuple(out)


GUARDS = _dedupe_guards(tuple(K.GUARDS) + tuple(W101L.GUARDS) + tuple(L118.GUARDS)
                        + (('w131_imported', W131._GUARD), ('w132_parent', PARENT_GUARD)))

SCRIPT_NAME = os.path.basename(__file__)
OWN_PROCESS_SUBSTRINGS = ('p515_s53_w132_resettle_v3_campaign', 'p515_s53_w118_resettle_campaign',
                          'p515_s53_w105_c_star_extension_campaign', 'p515_s53_w101_srp1_continuation_campaign',
                          'p515_s53_w98_continuation_campaign')
STAGE_TEXT = ('P5.15 Addendum 58, W132 -- v3 re-settling campaign (38 cells: claim groups 1, 2, 4) under the current '
              'production configuration: gated cells replayed bitwise against their original record through the first '
              'residual pass k0 (abort on the first divergence), the G cells as first C2 evaluations; the certifying '
              'regime held after the run\'s first residual pass (AA off, tight tail on, rho frozen); settling rule v3 '
              '(reading alpha: a non-Optimal accepted solve, ESSO included, is a lapse; reading gamma report-only; gap '
              'clause; amended monotone branch) until it certifies or the cap; W105 captures, per-cycle t_sum and Q_cc, '
              'and the IPOPT exit of every block of every cycle')
_P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
ROOT_REL = K.W132_ROOT_REL
SPEC_PREFIX = 'frozen_s53_resettle_spec_v3_'
SPEC_SERIES = 'frozen_s53_resettle_spec'
SPEC_VERSION = 3
PREDECESSOR_REL = os.path.join(_P53, 'w118_resettle', 'frozen_s53_resettle_spec_v2_fc791891.json')
CAMPAIGN_IDS = {cell: f'{K.CAMPAIGN_ID_PREFIX}{cell}' for cell in V.CELL_ORDER}
CONCURRENCY = 1
RESULTS_FILE = 'campaign_results.json'
MANIFEST_FILE = 'campaign_manifest_sha256.json'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
TASKS = 'TASKS.md'
SOLVER_PATH = '/usr/local/bin/ipopt'
PYTHON = '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python'
N_NETWORK_BLOCKS = 48
W101_X0_SPEC_REL = K.REF_CELLS['x0'][1]
COST_FILE_REL = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx')
EXTRA_CLEAN_FILES = (SCRIPT_NAME, 'p515_s53_w132_resettle_v3_hooks.py', 'p515_s53_w132_resettle_v3_checks.py',
                     'settling_criterion_v3.py', 'settling_criterion_v2.py', 'settling_criterion.py',
                     'p515_s53_w118_resettle_hooks.py', 'p515_s53_w118_resettle_checks.py',
                     'p515_s53_w118_resettle_campaign.py', 'p515_s53_w131_prefreeze_diagnostics.py',
                     'interface_dual_capture.py', 'gate_result_io.py', 'p515_s53_w105_settling_extension_hooks.py',
                     'p515_s53_w105_extension_checks.py', 'p515_s53_w101_srp1_continuation_campaign.py',
                     'p515_s53_w101_settling_continuation_hooks.py', 'p515_s53_w101_continuation_checks.py',
                     'p515_s53_w98_continuation_campaign.py', 'p515_s53_w98_continuation_hooks.py',
                     'p515_s53_w98_continuation_checks.py', 'p515_s53_w112_consensus_gap.py',
                     'p515_s53_w86_tail_recert_campaign.py', H.ESS_PARAMS_FILE_REL, 'shared_energy_storage_parameters.py',
                     'shared_energy_storage_data.py', 'network.py', 'helper_functions.py', COST_FILE_REL)
CODE_PINNED = tuple(dict.fromkeys(K.CODE_PINNED_BY_CHECKS + (
    SCRIPT_NAME, 'p515_s53_w118_resettle_campaign.py', 'p515_s53_w131_prefreeze_diagnostics.py',
    'p515_s53_w101_srp1_continuation_campaign.py', 'p515_s53_w98_continuation_campaign.py',
    'p515_s53_w98_continuation_hooks.py', 'p515_s53_w86_tail_recert_campaign.py',
    'p515_s53_w89_g6_final_attempt_reeval.py')))

# ---- verbatim text (checked against the committed brief / TASKS.md, whitespace-normalised, at every freeze) ------------
VERBATIM = {
    (BRIEF, 'ruling2_future_specs'): ('**Future specs:** a certifying cycle requires `Optimal Solution Found` on every '
                                      'block; a non-Optimal accepted solve makes the cycle non-certifying (retry, or the '
                                      'count restarts).'),
    (BRIEF, 'ruling1_form'): ('Report both F2 cells as "objective settled; interface-consensus gap unresolved (degenerate '
                              'dual, documented)", with margins in gross and Q_cc; the F2 conclusion (storage pays at ×2; '
                              'a two-node plan emerges) stands if the +102 k€ margin exceeds 3 × max(gap, slack) in both '
                              'terms, with the caveat stated.'),
    (BRIEF, 'ruling3_net_beside'): ('**Primary stays gross** (author\'s earlier decision, Addendum 25 era); the '
                                    'year-ladder row carries the net-of-salvage figure beside it'),
    (BRIEF, 'priority_order'): ('Phase A ladders (the affine slope) → flexibility ladder (break-even ×2) → ageing → the '
                                'remainder'),
    (BRIEF, 'reports_at_claims'): ('campaign in priority order, reports at each claim\'s completion (no stop between '
                                   'claims unless a prediction fails)'),
    (TASKS, 'ruling_ii_alpha'): ('**criterion v3 = reading α** (`boyd_k AND all_optimal_k` inside the rule — a '
                                 'non-Optimal accepted solve, ESSO included, is a lapse; \'the count restarts\'), γ '
                                 'report-only, **no retry tier** (a solve-path change)'),
    (TASKS, 'ruling_iii_caps'): ('the two L cells get their ruled cap N_old + 100 above the 300 implementation ceiling'),
    (TASKS, 'ruling_iv_order'): ('order: group 1 (B, D) → group 2 → group 4 (C, G, L) → group 3 E last, pending the '
                                 'author\'s soh_min ruling'),
    (TASKS, 'lattice_scope'): 'Lattice check scoped to E/P for ladder points (no substitution)',
    (TASKS, 'dead_zone'): ('Dead-zone candidates (m = 2, ESS P ≥ 1.25 MVA): `5f3cccb4`, `0ee93aca`, `45aa25a6`, '
                           '`7c455554`, `b2251bc5` (≈ 11.5 h to caps); `2ab0ce2d` borderline. Walls ≈ 64 h expected (44 '
                           'cells), 77 h worst'),
}

# ---- the references the new cells are scored against (committed; pinned at freeze) ------------------------------------
REFERENCES = {
    '7aa017f0': {'name': 'x0 settled (W101 d110bd1a, certified at 181)', 'kind': 'w101',
                 'eval_dir': os.path.join(_P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0', 'evals',
                                          'd110bd1a5977df1e_x0')},
    'bd504ecf': {'name': 'unit settled (W101 3f084f2f, certified at 172)', 'kind': 'w101',
                 'eval_dir': os.path.join(_P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_n7_4h_e1', 'evals',
                                          '3f084f2ffaeef2b7_n7_4h_e1')},
    '5ca4f86c': {'name': 'F2 incumbent (W118 r2 f2_incumbent, uncertified at 281, gap clause)', 'kind': 'w118',
                 'cell': 'f2_incumbent'},
    'e28de4ac': {'name': 'F2 challenger (W118 r2 f2_challenger, uncertified at 261, gap clause)', 'kind': 'w118',
                 'cell': 'f2_challenger'},
}
W118_SUMMARY = os.path.join(_P53, 'w118_resettle', 'w118_resettle_summary.json')
W118_SUMMARY_MANIFEST = os.path.join(_P53, 'w118_resettle', 'w118_resettle_summary_manifest_sha256.json')
BASELINE_TABLES = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'baseline_tables', 'baseline_tables.json')
W2_TABLE = os.path.join('data', 'SRP1', 'Results', 'P515S45', 'investment_cost', 'investment_cost_results.json')
CLAIM_ITEMS = ('B', 'C', 'G', 'H', 'I', 'J', 'L', 'CHECK')
GAP_REFUSED_LABEL = 'objective settled; interface-consensus gap unresolved (degenerate dual)'
PF_SLOPE_WINDOW = 50
D_FIT_NODE7_LABELS = ('n7_2h_e1', 'n7_4h_e1', 'n7_2h_e2', 'n7_4h_e2', 'n7_2h_e3', 'n7_4h_e3', 'n7_2h_e4', 'n7_4h_e4',
                      'n7_2h_e5', 'n7_4h_e5')

# ---- the frozen definitions ---------------------------------------------------------------------------------------------
DEFINITIONS = {
    'objective_convention': ('Q = gross_operational_cost, settlement EXCLUDED (primary; Addendum 58 Ruling 3); Q_net = '
                             'net_operational_recourse = Q - terminal salvage credit, reported BESIDE gross on every G '
                             'row and on any claim stated net; t_sum = the priced interface-consensus gap sum w * pi * '
                             '(p_DSO - p_TSO) (EUR); Q_cc = Q + t_sum, a FIRST-ORDER consensus-consistent DIAGNOSTIC'),
    'per_cell': {
        'end_cycle': 'k* (certified) or the cap (uncertified); every per-cell value below is read at the end cycle',
        'k_star': 'the cycle the settling rule v3 (reading alpha) certified; None when uncertified at its cap',
        'first_k0_v2': 'the run\'s first residual pass under the version-2 definition (holds and dynamic cap keyed on it)',
        'first_k0_alpha': 'the first k0 of the reading-alpha rule',
        'band': '[min, max] Q over the certifying window (uncertified: over the last L = 60 cycles)',
        's': 'gated cells: s = Q(end) - Q_N_old (Q_N_old = the ORIGINAL record\'s last cycle); ungated: undefined',
        'Q_net': 'net_operational_recourse of the end cycle (per_cycle_record recourse)',
        'non_optimal_cycles': 'cycles whose all_optimal_k is False (every block\'s final exit; ESSO included)',
        'gamma_report_only': 'reading gamma\'s decision (report-only)',
        'terminal_step_to_threshold': ('|dQ_last| / EPS0; range / tau at k* (certified); production\'s rule ten '
                                       '(objective_change_abs / objective_tolerance) of the last row (CLAUDE.md)'),
        'pf_primal_slope_last_50': ('OLS slope of boyd_pf_primal_ratio on the cycle over the last 50 cycles ending at the '
                                    'end cycle (per_cycle_record; report-only)'),
    },
    'uncertified_form': {
        'frozen_v2_form': ('band [min, max] over the last L = 60 cycles; drift rate = mean dQ over the last 25; dQ_cc '
                           'rate = mean (dQ + dt_sum) over the same cycles; t_sum and Q_cc at the cap; reasons'),
        'gap_refused_cell': ('a cell uncertified at its cap with at least one gap-clause refusal (decision gap_refusals '
                             'non-empty) -- Addendum 58 Ruling 1, generalised'),
        'gap_refused_label': GAP_REFUSED_LABEL,
        'gap_refused_carries': ('the label; the pf-primal slope over the last 50 cycles; t_by_node at the cap; the drift '
                                'rate; the dQ_cc rate; gap = |t_sum at the cap|; slack = |s| (gated)'),
        'determinacy_rule': ('a difference involving an uncertified cell is DETERMINATE iff its margin exceeds BAR = 3 x '
                             'max(|gap_u|, |slack_u|) over the uncertified cell(s) u of the difference in BOTH gross and '
                             'Q_cc terms (Addendum 58 Ruling 1; as W130 applied it: 3 x max(9,234.42, 1,461.49) = '
                             '27,703.27); margin_Qcc = |d_Qcc| if sign(d_Qcc) == sign(d_Q), else -|d_Qcc|; an ungated '
                             'uncertified cell has no slack -> INDETERMINATE (slack undefined)'),
    },
    'claims': {
        'source': ('W117 claims (w117_triage_recompute.json, 3b2e76de): item, claim id, statement, form, claim type, '
                   'I_ref, I_other, net_of_salvage, per claim; the cells are the re-settled ones (this campaign), the '
                   'settled references (x0 d110bd1a for 7aa017f0; unit 3f084f2f for bd504ecf) and the W118 F2 pair '
                   '(5ca4f86c incumbent, e28de4ac challenger); a claim is scored only when every cell it names is in '
                   'that set'),
        'form_F': 'd = [Q(o) + I(o) - SV(o)] - [Q(r) + I(r) - SV(r)], SV = salvage only for the net figure',
        'form_value': 'd = [Q(r) - Q(o)] - [I(o) - I(r)] (value - I; delta-value - delta-I)',
        'd_Qcc': 'the same with every Q replaced by Q_cc = Q + t_sum',
        'net_beside': 'd_net: the F form with Q_net (G rows: primary per the claim\'s own convention, the other beside)',
        'resolution_settled_vs_settled': ('both cells certified (settled): resolution = band_r + band_o (the sum of the '
                                          'two band widths); determinate iff |d| > resolution (gross verdict; Q_cc '
                                          'beside, report-only)'),
        'resolution_with_an_uncertified_cell': 'the determinacy rule of the uncertified form (above)',
        'verdict_words': ('determinate | within resolution | indeterminate (slack undefined) | not scored (a cell has '
                          'no result yet)'),
    },
    'D_fit': {
        'point_set': ('the committed Phase A table baseline_tables.json node7_fit.labels: the 10 node-7 S47 A1a points '
                      'n7_{2h,4h}_e{1..5} (value = Q(0) - Q(x)), with Q(0) the settled x0 (W101 d110bd1a, Q181) and '
                      'n7_4h_e1 the settled unit (W101 3f084f2f, Q172); the other 9 are the D cells of this campaign. '
                      'The OLS fit itself uses NO B point; the committed breakeven block also cites ONE B point, '
                      'first_unit_best_node = n5_4h_e1 (2a0ba8b2, the argmin of F over the 30 A1a points), recomputed '
                      'on its settled value'),
        'formula': ('value = a + b E + c P, OLS; SE = sqrt(diag(sigma^2 (X^T X)^-1)), sigma^2 = RSS / (n - 3); e* = b + '
                    'c/4 - p_cost/4; se_e* = sqrt(se_b^2 + (se_c/4)^2); margin = e_cost - e*; first_unit = (V - 0.25 '
                    'p_cost) / 1.0 (p515_s47_baseline_tables.py; W117); p_cost, e_cost from the W2 table candidate '
                    'n7_4h_e1'),
        'slack_bound_report_only': ('e* is linear in the values y_i = Q0 - Q_i: e* = sum_i w_i y_i + const; the '
                                    'stopping-slack bound sum_i |w_i| band_i + |sum_i w_i| band_0 is REPORTED beside the '
                                    'SE (a Worker formula, for Planner confirmation; no verdict is read off it)'),
    },
}

# ---- predictions, recorded BEFORE any run (each with its source) --------------------------------------------------------
PREDICTIONS = {
    'dead_zone_candidates': {
        'cells': list(V.DEAD_ZONE_CANDIDATES),
        'statement': ('dead-zone candidates (m = 2, ESS P >= 1.25 MVA): 5f3cccb4, 0ee93aca, 45aa25a6, 7c455554, b2251bc5 '
                      '(~11.5 h to caps)'),
        'operationalisation': ('each is UNCERTIFIED at its cap as a gap-refused cell (at least one gap-clause refusal: '
                               'the F2 pattern) -- Worker operationalisation, for Planner confirmation'),
        'source': 'Advisor design review of the remaining cells, as transcribed in TASKS.md (Addendum 58 section, 82fa1b61)'},
    'dead_zone_borderline': {
        'cells': list(V.DEAD_ZONE_BORDERLINE), 'statement': '2ab0ce2d borderline',
        'operationalisation': 'outcome recorded (certified / gap-refused / other), not scored held or missed',
        'context': ('W131 Task 2 (7a5c5717): in its original run t_sum closed inside tau/2 over 334-337 with AA accepted '
                    'at 325-326 (H_regime supported)'),
        'source': 'Advisor design review, as transcribed in TASKS.md (Addendum 58 section, 82fa1b61)'},
    'gate_ability': {
        'statement': ('every gated cell (groups 1 and 2, C, L) replays bitwise against its original record through its '
                      'first residual pass k0 (G19)'),
        'cells': list(V.GATED_CELLS),
        'source': ('Advisor design review: gate-ability (TASKS.md Addendum 58 section: "per-cell configuration and '
                   'gate-ability"; the gated / ungated split as given in Planner task W132)')},
    'walls': {
        'statement': 'walls ~64 h expected (44 cells), 77 h worst',
        'note': ('the 44 runs include the 6 E arms (not in this spec) and the D affine-fit cells; this spec\'s own '
                 'per-cell estimate is its expected_wall_time block'),
        'source': 'Advisor design review, as transcribed in TASKS.md (Addendum 58 section, 82fa1b61)'},
    'per_claim_numeric_predictions': {
        'statement': None,
        'note': ('NONE transcribed: TASKS.md Addendum 58 section (82fa1b61 and its diff), PLANNER_BRIEF Addenda 57-58 and '
                 'the W131 commit were searched; the design review as transcribed carries the cell list, per-cell '
                 'configuration, gate-ability, the cap-ceiling exceedances, the dead-zone candidates and the walls, and '
                 'no per-claim value, sign or k* prediction'),
        'source': 'Worker search record (W132)'},
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


def _read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _committed_clean(rel):
    st = L._git_state(rel)
    return bool(st.get('git_tracked') and st.get('git_clean'))


def guards_verify():
    return {n: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for n, g in GUARDS}


def _guards_ok(g):
    return all(not v['verify_0_failures'] for v in g.values())


def _finish(code, extra_msg=''):
    g = guards_verify()
    _log(f'[W132] guards {g} {extra_msg}')
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
    c = V.CELLS[cell]
    return {'name': ('W132 SRP1 RE-SETTLING v3 (Addendum 58) -- the current production configuration: the case file (AA '
                     'keep_memory declared), the ESS ageing baseline ' + cfg['ess_ageing_baseline_label'] + ' declared, '
                     'the convergence-depth tight tail DECLARED ENABLED (compl_inf_tol 1e-6, from AA-off + 1), '
                     'post-certification: persist the certified TSO/DSO models only'),
            'arm_label': 's39_D', 'overrides': {},
            'case_file_anderson_acceleration': cfg['case_file_anderson_acceleration'],
            'ess_ageing_baseline': cfg['ess_ageing_baseline'],
            'ess_ageing_baseline_label': cfg['ess_ageing_baseline_label'],
            'convergence_depth_tail': cfg['convergence_depth_tail'],
            'note': ('the entry adds settling_resettle (v3 schema; keyed); cap: gated N_old + 100 (per-cell ceiling), '
                     'ungated 300 (the rule stops at min(k0_run + 109, 300)); persist_certified_models as the W101 / W118 '
                     'cells; no option (b); concurrency 1'
                     + (f"; flexibility price x {c['flex_price_multiplier']} per entry" if c['flex_price_multiplier']
                        is not None else ''))}


def orig_entry(cell):
    return K.spec_entry(cell)[1]


def entries(cell):
    e = orig_entry(cell)
    nodes = {int(k): tuple(map(float, v)) for k, v in e['canonical']['nodes'].items()}
    opts = {'investment_year': e['canonical']['investment_year'],
            'post_certification': {'persist_certified_models': True, 'hull_polish': False, 'reference': None},
            'settling_resettle': V.declaration_for(cell)}
    if V.CELLS[cell]['flex_price_multiplier'] is not None:
        opts['flex_price_multiplier'] = V.CELLS[cell]['flex_price_multiplier']
    return [(cell, nodes, opts)]


def expected_keys(cell):
    spec, e, kw = K.resettle_kwargs(cell)
    ocfg = spec['configuration']
    base = H.evaluation_key(e['key'], e['overrides'], **kw)
    key = H.evaluation_key(e['key'], e['overrides'], settling_resettle=V.declaration_for(cell), **kw)
    base_orig = H.evaluation_key(e['key'], e['overrides'], case_file_aa=ocfg.get('case_file_anderson_acceleration'),
                                 ess_ageing_baseline=ocfg.get('ess_ageing_baseline'),
                                 flex_price_multiplier=e.get('flex_price_multiplier'),
                                 convergence_depth_tail=ocfg.get('convergence_depth_tail'))
    return {'base_key_current_configuration': base, 'resettle_key': key,
            'base_key_original_configuration': base_orig, 'original_eval_key': e['eval_key']}


def campaign_root_rel(cell):
    return os.path.join(ROOT_REL, f'campaign_{CAMPAIGN_IDS[cell]}')


def campaign_root(cell):
    return _abs(campaign_root_rel(cell))


def pre_launch_assertion(cell, spec=None):
    """The re-settling key, recomputed now, is EXACTLY the expected one (and a frozen spec's entry carries it); it appears
    in no committed campaign spec OUTSIDE the W132 stage root (the rule: a pre-run check that scans committed artefacts
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
        == e['eval_key'] == V.CELLS[cell]['orig_eval_key'],
        'resettle_key_differs_from_original_and_base': k['resettle_key'] not in (e['eval_key'],
                                                                                 k['base_key_current_configuration']),
        'resettle_key_absent_from_committed_specs_outside_w132_root': k['resettle_key'] not in committed,
        'campaign_root_differs_from_original': os.path.abspath(campaign_root(cell)) != os.path.abspath(
            _abs(V.CELLS[cell]['orig_root'])),
        'eval_dir_name_differs_from_original': eval_dir_name != e['eval_dir'],
        'working_dir_ids_differ_from_original': not (set(ids.values()) & set(e['working_dir_ids'].values())),
        'working_dirs_absent': not any(os.path.exists(os.path.join(work, i)) for i in ids.values()),
    }
    if spec is not None:
        parts['frozen_entry_equals_recomputed'] = (entry is not None and entry.get('eval_key') == k['resettle_key']
                                                   and entry.get('eval_dir') == eval_dir_name
                                                   and entry.get('working_dir_ids') == ids)
    return {'holds': all(parts.values()), 'parts': parts, **k, 'eval_dir_name': eval_dir_name, 'working_dir_ids': ids,
            'n_committed_keys_scanned': len(committed), 'excluded_roots': [ROOT_REL]}


def validate_campaign_spec(cell, spec):
    ref = _load(W101_X0_SPEC_REL)          # the W101 x0 cell: the same C2 + tight-tail configuration, concurrency 1
    want = configuration(cell)
    cfg = spec['configuration']
    ents = spec['candidates']
    e = ents[0] if len(ents) == 1 else {}
    oe = orig_entry(cell)
    checks = {f'configuration:{k}': cfg.get(k) == want[k] for k in
              ('arm_label', 'overrides', 'case_file_anderson_acceleration', 'ess_ageing_baseline',
               'ess_ageing_baseline_label', 'convergence_depth_tail')}
    checks.update({
        'configuration:case_file_sha256': cfg.get('case_file_sha256') == K.CASE_FILE_SHA256,
        'configuration:ess_params_file_C2': (cfg.get('ess_params_file') or {}).get('sha256') == K.ESS_PARAMS_SHA256_C2,
        'configuration:as_the_w101_x0_cell': all(cfg.get(k) == ref['configuration'].get(k) for k in (
            'arm_label', 'overrides', 'case_file_anderson_acceleration', 'ess_ageing_baseline',
            'ess_ageing_baseline_label', 'convergence_depth_tail', 'apply_rho', 'full_diagnostics_in_rows',
            'case_file_sha256')),
        'one_entry': len(ents) == 1, 'entry_label': e.get('label') == cell,
        'entry_canonical_and_key_as_original': e.get('canonical') == oe['canonical'] and e.get('key') == oe['key'],
        'entry_overrides_empty': e.get('overrides') == {},
        'entry_post_certification_persist_only': e.get('post_certification') == {
            'persist_certified_models': True, 'hull_polish': False, 'reference': None},
        'entry_flex_multiplier': e.get('flex_price_multiplier') == V.CELLS[cell]['flex_price_multiplier'],
        'entry_resettle_is_the_v3_declaration': e.get('settling_resettle') == V.declaration_for(cell),
        'entry_has_no_other_continuation': not any(x in e for x in ('settling_continuation', 'certification_continuation',
                                                                    'settling_extension', 'release_solution_bookkeeping',
                                                                    'model_variant')),
        'no_early_stop_anywhere_in_the_entry': 'early_stop' not in json.dumps(e),
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_IDS[cell],
        'cap_is_the_cell_cap': spec.get('cap') == V.spec_cap(cell),
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
    """The child's capture checklist, asserted in the PARENT too (before the lock)."""
    decl = spec['candidates'][0]['settling_resettle']
    tail_on = bool((spec['configuration'].get('convergence_depth_tail') or {}).get('enabled'))
    aa_on = bool((spec['configuration'].get('case_file_anderson_acceleration') or {}).get('enabled'))
    try:
        checks = V.assert_resettle_preconditions(decl, spec, {'tail_enabled_for_this_run': tail_on}, aa_on)
        return True, checks
    except Exception as error:  # noqa: BLE001
        return False, {'error': f'{type(error).__name__}: {error}'}


# ======================================================================================================================
#  walls and the claim-completion points
# ======================================================================================================================
def _iteration_walls(eval_dir_rel):
    txt = open(_abs(os.path.join(eval_dir_rel, 'child_stdout.log')), errors='replace').read()
    return [float(m) for m in re.findall(r'\[INFO\] \t - Iteration \d+: ([0-9.]+) s', txt)]


def wall_time_estimate():
    """Per cell (W118's basis, fc791891): s = the original run's mean per-cycle wall x kappa (kappa = W104's C* mean at
    concurrency 1 / the W86 recert's C* mean at concurrency 3), plus W110's measured child + launcher overhead and one
    inline re-run of the zero-solve checks. Expected cycles: the cap for the dead-zone candidates (the Advisor's
    prediction), k0 + 60 for every other gated cell (the W118 / W128-W129 outcome: certified at k0 + 57..72), the
    original k0 + 60 for the G cells (a proxy: k0_run is unknown); worst = the cap (G: min(original k0 + 109, 300)).
    Calibration beside: W118's ten cells, measured wall / its own expected."""
    c104 = os.path.join(_P53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_c_star', 'evals',
                        '4bf36c151fd10613_c_star')
    c86 = os.path.join(_P53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'evals', '96c5aa50cc229cc1_c_star')
    kappa = statistics.mean(_iteration_walls(c104)) / statistics.mean(_iteration_walls(c86))
    w110 = os.path.join(_P53, 'w105_c_star_extension', 'campaign_s53_w105_c_star_ext_r2')
    w110_eval = os.path.join(w110, 'evals', '8864266d3c064ed0_c_star_ext')
    lines = _read_jsonl(_abs(os.path.join(w110_eval, 'settling_extension_cycle_record.jsonl')))
    sum_walls = sum(x['t_end_s'] - x['t_start_s'] for x in lines)
    child = _load(os.path.join(w110_eval, 'evaluation_record.json'))['wall_time_s']['child_process_s']
    campaign = _load(os.path.join(w110, 'campaign_results.json'))['wall_clock_s']
    overhead = (child - sum_walls) + (campaign - child) + 240.0
    per, tot_exp, tot_worst = {}, 0.0, 0.0
    for cell in V.CELL_ORDER:
        c = V.CELLS[cell]
        walls = _iteration_walls(V.original_eval_dir(cell))
        s = statistics.mean(walls) * kappa
        cap = V.spec_cap(cell) if c['gated'] else min(c['k0'] + V.CAP_AFTER_K0, c['cap_ceiling'])
        exp_cycles = cap if cell in V.DEAD_ZONE_CANDIDATES else min(c['k0'] + 60, cap)
        e_s, w_s = exp_cycles * s + overhead, cap * s + overhead
        per[cell] = {'original_mean_cycle_s': statistics.mean(walls), 'original_cycles': len(walls),
                     's_per_cycle_estimate': s, 'expected_cycles': exp_cycles, 'cap_cycles': cap,
                     'expected_h': e_s / 3600.0, 'worst_case_h': w_s / 3600.0}
        tot_exp += e_s
        tot_worst += w_s
    calib = {}
    w118 = _load(PREDECESSOR_REL)
    for cell, sub in K.W118_CELL_DIRS.items():
        root = sub.split('/evals/')[0]
        res = _load(os.path.join(K.W118_ROOT_REL, root, RESULTS_FILE))
        est = w118['expected_wall_time']['per_cell'][cell]
        cyc = res['cell_report']['cycles_run']
        calib[cell] = {'measured_h': res['wall_clock_s'] / 3600.0, 'cycles_run': cyc,
                       'estimate_at_the_run_cycles_h': (cyc * est['s_per_cycle_estimate']
                                                        + w118['expected_wall_time']['overhead_s_per_cell']) / 3600.0}
        calib[cell]['measured_over_estimate'] = calib[cell]['measured_h'] / calib[cell]['estimate_at_the_run_cycles_h']
    return {'basis': wall_time_estimate.__doc__, 'kappa_concurrency_1_over_3': kappa,
            'overhead_s_per_cell': overhead, 'per_cell': per, 'total_expected_h': tot_exp / 3600.0,
            'total_worst_case_h': tot_worst / 3600.0,
            'w118_calibration_measured_over_estimate': calib,
            'advisor_transcribed': PREDICTIONS['walls']['statement']}


def claims_dataset():
    """The W117 claims this campaign scores: every claim of items B, C, G, H, I, J, L and the CHECK row whose cells are
    all in {the 38 re-settled cells, the settled references, the W118 F2 pair}. Each claim keeps its W117 definition
    (id, statement, form, type, I_ref, I_other, net_of_salvage, recorded); its cells are mapped to their settled source."""
    w117, sha = K._pinned_json(K.W117)
    by_key = {V.CELLS[c]['orig_eval_key']: c for c in V.CELL_ORDER}
    out, skipped = [], []
    for cl in w117['claims']:
        if cl['item'] not in CLAIM_ITEMS:
            continue
        cells = {}
        ok = True
        for side in ('ref', 'other'):
            key = cl[side]['eval_key']
            p8 = key[:8]
            if key in by_key:
                cells[side] = {'source': 'w132', 'cell': by_key[key]}
            elif p8 in REFERENCES:
                cells[side] = {'source': 'reference', 'ref': p8}
            elif p8 in ('d110bd1a', '3f084f2f'):
                cells[side] = {'source': 'reference', 'ref': {'d110bd1a': '7aa017f0', '3f084f2f': 'bd504ecf'}[p8]}
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
                    'note': cl.get('note')})
    return {'source': K.W117['path'], 'sha256': sha, 'commit': K.W117['commit'], 'claims': out,
            'n_claims': len(out), 'skipped_not_in_the_settled_set': skipped}


def completion_points(dataset):
    """For every claim and item: the cell after which it is complete (its last re-settled cell in CELL_ORDER)."""
    pos = {c: i for i, c in enumerate(V.CELL_ORDER)}
    per_claim = {}
    for cl in dataset['claims']:
        cells = [s['cell'] for s in (cl['ref'], cl['other']) if s['source'] == 'w132']
        per_claim[cl['claim_id']] = {'item': cl['item'], 'cells': cells,
                                     'complete_after': (max(cells, key=pos.get) if cells else None)}
    items = {}
    for cid, v in per_claim.items():
        if v['complete_after'] is None:
            continue
        cur = items.get(v['item'])
        if cur is None or pos[v['complete_after']] > pos[cur]:
            items[v['item']] = v['complete_after']
    items['D'] = max((c for c in V.CELL_ORDER if V.CELLS[c]['item'] == 'D'), key=pos.get)
    # an item whose W117-PENDING claims complete earlier than the item gets its own point (B: the 3 pending B claims
    # complete after the third cell; the D cells' B claims were determinate in W117)
    pending = {}
    for cl in dataset['claims']:
        v = per_claim[cl['claim_id']]
        if cl['old_verdict_R1_w117'] == 'pending' and v['complete_after'] is not None:
            cur = pending.get(cl['item'])
            if cur is None or pos[v['complete_after']] > pos[cur]:
                pending[cl['item']] = v['complete_after']
    labelled = dict(items)
    for item, cell in pending.items():
        if cell != items.get(item):
            labelled[f'{item} (its W117-pending claims)'] = cell
    points = {}
    for item, cell in labelled.items():
        points.setdefault(cell, []).append(item)
    ordered = [{'after_cell_index_1_based': pos[c] + 1, 'after_cell': c, 'items_complete': sorted(points[c])}
               for c in sorted(points, key=pos.get)]
    return {'rule': ('a claim is complete after the last of its re-settled cells in the launch order; an item after the '
                     'last cell of any of its claims (D: after the last D cell); the Planner reports at each point '
                     '(Addendum 58: "reports at each claim\'s completion")'),
            'report_points': ordered, 'per_item': {k: {'after_cell': v, 'index_1_based': pos[v] + 1}
                                                   for k, v in labelled.items()},
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
#  post-run gates
# ======================================================================================================================
LINE_FIELDS_REQUIRED = L118.LINE_FIELDS_REQUIRED + ('ipopt_exit_by_block', 'ipopt_exit_counts', 'all_optimal_k',
                                                    'non_optimal_blocks')
SETTLING_FIELDS_REQUIRED = L118.SETTLING_FIELDS_REQUIRED + ('boyd_k_v2', 'all_optimal_k', 'boyd_k_v3', 'first_k0_alpha',
                                                            'gamma')


def _decision(eval_dir):
    p = os.path.join(eval_dir, V.DECISION_FILE)
    return json.load(open(p)) if os.path.isfile(p) else None


def hold_checks(cell, eval_dir, rec):
    lines = _read_jsonl(os.path.join(eval_dir, V.CYCLE_FILE))
    summ = rec.get('settling_resettle_summary') or {}
    by = {x['cycle']: x for x in lines}
    cycles = sorted(by)
    fp = summ.get('first_residual_pass_run')
    pre = [by[c] for c in cycles if fp is None or c <= fp]
    held = [by[c] for c in cycles if fp is not None and c > fp]
    rho_fp = ((by.get(fp) or {}).get('rho') or {}).get('rho_after') if fp else None
    dec = _decision(eval_dir) or {}
    last = cycles[-1] if cycles else None
    c = V.CELLS[cell]
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
            x.get('certificate_length_in_force_at_cycle_end') == V.CERTIFICATION_DISABLED_THRESHOLD
            or (x['cycle'] == last and x.get('certificate_length_in_force_at_cycle_end') == V.SETTLING_END_THRESHOLD
                and summ.get('stopped_by') in ('settling_rule', 'rule_cap')) for x in lines),
        'replay_equal_every_gated_cycle': ((not c['gated'])
                                           or (all(by[k].get('replay_equal') is True
                                                   for k in range(1, c['k0'] + 1) if k in by)
                                               and all(k in by for k in range(1, c['k0'] + 1)))),
        'summary_ok': summ.get('ok') is True,
        'decision_present': bool(dec),
    }
    return all(parts.values()), {'parts': parts, 'first_pass': fp, 'rho_at_first_pass': rho_fp}


def stopping_check(cell, rec, eval_dir):
    summ = rec.get('settling_resettle_summary') or {}
    dec = _decision(eval_dir) or {}
    k = rec.get('cycles_run')
    spec_cap = V.spec_cap(cell)
    if dec.get('status') == 'certified':
        ok = k == dec.get('k_star') and summ.get('stopped_by') == 'settling_rule'
    elif dec.get('status') == 'uncertified':
        if V.CELLS[cell]['gated']:
            ok = k == dec.get('k_cap') == spec_cap and summ.get('stopped_by') == 'cap'
        else:
            ok = k == dec.get('k_cap') and summ.get('stopped_by') == ('rule_cap' if k < spec_cap else 'cap')
    else:
        ok = False
    return bool(ok), {'cycles_run': k, 'decision_status': dec.get('status'), 'k_star': dec.get('k_star'),
                      'k_cap': dec.get('k_cap'), 'stopped_by': summ.get('stopped_by'), 'spec_cap': spec_cap}


def settling_replay_check(cell, eval_dir):
    """The pure rule v3 replayed on the run's per_cycle_record (Q, boyd) with the in-cycle t_sum and all_optimal_k
    reproduces every in-cycle rule record and the decision."""
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, V.CYCLE_FILE))}
    rule = K.pure_rule(V.declaration_for(cell))
    pure = []
    for r in rows:
        ln = lines.get(r['cycle']) or {}
        pure.append(rule.observe(r['cycle'], r['gross_operational_cost'],
                                 bool(r['boyd_all_pass'] and r['local_solves_ok']), ln.get('t_sum'),
                                 bool(ln.get('all_optimal_k'))))
    in_cycle = [(lines.get(r['cycle']) or {}).get('settling') for r in rows]
    dec = _decision(eval_dir) or {}
    pd = rule.decision or {}
    keys = ('status', 'k_star', 'branch', 'k0', 'N', 'T', 'A', 'P_hat', 'W', 'window', 'band', 'band_width', 'k_cap',
            'reasons', 'range', 't_sum_k_star', 'gap_refusals', 'drift_rate_mean_dQ_last_25', 'dQ_cc_rate_mean_last_25',
            'version', 'first_k0_alpha', 'non_optimal_cycles', 'lapse_events', 'gamma_report_only')
    parts = {'every_cycle_record_reproduced': [K._jt(a) for a in in_cycle] == [K._jt(b) for b in pure],
             'decision_reproduced': bool(dec) and all(K._jt(dec.get(k)) == K._jt(pd.get(k)) for k in keys),
             'decision_file_present': bool(dec), 'decision_is_version_3': dec.get('version') == 3}
    return all(parts.values()), {'parts': parts}


def replay_gate_full(cell, eval_dir):
    """Gated cells: rows 1..k0 of the run's per_cycle_record.jsonl against the ORIGINAL record's rows: EVERY field."""
    c = V.CELLS[cell]
    ref = {r['cycle']: r for r in _read_jsonl(_abs(V.reference_path(cell)))}
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


def line_fields_check(eval_dir):
    lines = _read_jsonl(os.path.join(eval_dir, V.CYCLE_FILE))
    missing = {}
    for x in lines:
        m = [f for f in LINE_FIELDS_REQUIRED if f not in x]
        if x.get('gross') is not None:
            m += [f for f in ('blocks_captured',) if f not in x]
        s = x.get('settling') or {}
        m += [f'settling.{f}' for f in SETTLING_FIELDS_REQUIRED if f not in s]
        m += [f'boyd_ratios.{f}' for f in R.BOYD_RATIO_FIELDS if f not in (x.get('boyd_ratios') or {})]
        if len(x.get('ipopt_exit_by_block') or {}) != 51:
            m.append('ipopt_exit_by_block(51)')
        if m:
            missing[str(x['cycle'])] = m
    return not missing and bool(lines), {'n_lines': len(lines), 'missing_by_cycle': dict(list(missing.items())[:10])}


def overlap_check(cell, rec):
    summ = rec.get('settling_resettle_summary') or {}
    ov = summ.get('overlap_k0_plus_1_to_N_old') or []
    c = V.CELLS[cell]
    if not c['gated']:
        return len(ov) == 0, {'n': len(ov)}
    ok = [o['cycle'] for o in ov] == list(range(c['k0'] + 1, c['N_old'] + 1)) and all(
        o.get('Q_new_minus_Q_old') is not None for o in ov)
    return bool(ok), {'n': len(ov)}


def _net_key(agent, year, day):
    if agent == 'TSO':
        return f'TSO|{year}|{day}'
    return f'DSO|{int(agent[3:])}|{year}|{day}'


def exit_crosscheck(eval_dir, lines=None, rec=None):
    """G24: every cycle line's 51 exit classes against the independent records -- the 48 network blocks' final attempts
    in network_ipopt_solve_records.jsonl (their EXIT parsed from the IPOPT logs; W131's reader, which cross-checks every
    record against its log byte range) and the 3 ESSO final attempts from the per-solve ESSO logs (W131's reader); the
    initialisation round against the summary's round-0 capture."""
    eval_rel = os.path.relpath(eval_dir, REPO)
    lines = lines if lines is not None else {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, V.CYCLE_FILE))}
    n = max(lines)
    net, net_meta = W131.network_final_attempts(eval_rel, n)
    recs = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    logs_dir = os.path.dirname(recs[0]['log_path'])
    esso, esso_meta = W131.esso_final_attempts(eval_rel, logs_dir, n)
    disagree = []
    compared = 0
    for (rnd, agent, _network, year, day), v in net.items():
        want = H.ipopt_exit_class(v['final_exit'])
        key = _net_key(agent, year, day)
        if rnd == 0:
            got_nonopt = key in ((((rec or {}).get('settling_resettle_summary') or {}).get('exit_capture_init_round_0')
                                  or {}).get('non_optimal_blocks') or {})
            if got_nonopt != (want != V.OPTIMAL):
                disagree.append({'round': 0, 'block': key, 'record': want, 'init_non_optimal': got_nonopt})
            continue
        got = ((lines.get(rnd) or {}).get('ipopt_exit_by_block') or {}).get(key, {}).get('class')
        compared += 1
        if got != want:
            disagree.append({'round': rnd, 'block': key, 'record': want, 'line': got})
    for (k, node), v in esso.items():
        if k == 0:
            continue
        want = H.ipopt_exit_class(v['final_exit'])
        got = ((lines.get(k) or {}).get('ipopt_exit_by_block') or {}).get(f'ESSO|{node}', {}).get('class')
        compared += 1
        if got != want:
            disagree.append({'round': k, 'block': f'ESSO|{node}', 'record': want, 'line': got})
    all_opt_consistent = all(x.get('all_optimal_k') == all(v['class'] == V.OPTIMAL for v in
                                                            (x.get('ipopt_exit_by_block') or {}).values())
                             for x in lines.values())
    parts = {'no_disagreement': not disagree, 'compared_51_per_cycle': compared == 51 * n,
             'network_log_crosscheck_clean': net_meta['log_byte_crosscheck']['n_disagree'] == 0,
             'esso_every_solve_found': not esso_meta['missing'],
             'all_optimal_k_is_the_conjunction': all_opt_consistent}
    return all(parts.values()), {'parts': parts, 'n_compared': compared, 'disagreements_first20': disagree[:20],
                                 'n_disagreements': len(disagree), 'network_meta': {
                                     k: net_meta[k] for k in ('n_records', 'n_blocks_x_rounds', 'attempt_counts')},
                                 'esso_meta': {k: esso_meta[k] for k in ('n_solves', 'missing', 'n_with_retry_logs',
                                                                         'exit_counts_final')}}


def status_label_check(rec, eval_dir):
    dec = _decision(eval_dir) or {}
    want = 'certified' if dec.get('status') == 'certified' else 'not_certified'
    ok = (rec.get('status') == want and rec.get('status_production_trajectory') is not None
          and (rec.get('certified_cost') is None) == (want != 'certified'))
    return bool(ok), {'record_status': rec.get('status'), 'decision_status': dec.get('status'),
                      'production_trajectory_view': rec.get('status_production_trajectory')}


GATE_SCOPE = {
    'G19_replay_bitwise_1_k0_every_field': 'gated cells only (the G cells have no replay reference: SKIPPED)',
    'G23_overlap_recorded': 'gated cells: k0+1..N_old recorded; ungated: none (asserted empty)',
    'G6_v37_optimal_and_four_metrics': 'every cell; REPORTED, does NOT stop the campaign (Planner ruling at W128)',
    'all_other_gates': 'every cell',
}
NON_STOPPING_GATES = ('G6_v37_optimal_and_four_metrics',)


def cell_gates(cell, entry, eval_dir):
    rec_path = os.path.join(eval_dir, 'evaluation_record.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    exit_code = (int(open(os.path.join(eval_dir, 'exit_code.txt')).read().strip())
                 if os.path.isfile(os.path.join(eval_dir, 'exit_code.txt')) else None)
    gates, detail = {}, {'exit_code': exit_code, 'gate_scope': GATE_SCOPE}
    gates['G1_harness_clean'] = (exit_code == 0 and bool(rec) and rec.get('status') in ('certified', 'not_certified')
                                 and not os.path.exists(os.path.join(eval_dir, 'parent_barrier_record.json')))
    if not rec or rec.get('status') == 'error':
        detail['barrier'] = {k: rec.get(k) for k in ('status', 'barrier_cause')}
        detail['settling_resettle_summary'] = rec.get('settling_resettle_summary')
        return gates, detail, rec
    gates['G2_eval_key'] = rec.get('eval_key') == entry['eval_key'] == expected_keys(cell)['resettle_key']
    c, d = L.evaluation_checks(eval_dir, rec, entry['working_dir_ids']['run'])
    detail['w86_evaluation_checks'] = {'checks': c, 'detail': d}
    gates['G3_append_reconcile'] = c.get('append_reconciles_byte_identical', False)
    gates['G4_tail_state_check'] = c.get('tail_state_check_matches', False)
    records = _read_jsonl(os.path.join(eval_dir, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE))
    gates['G5_solve_profile_reconciled_per_event'], detail['G5'] = W101L.solve_profile_check(rec, len(records))
    g6 = W98L.g6_v37_evaluate_records(records, rec.get('cycles_run'), b=N_NETWORK_BLOCKS)
    gates['G6_v37_optimal_and_four_metrics'] = g6['gate_pass']
    detail['G6_v37'] = g6
    gates['G7_append_sealed'] = c.get('append_sealed_after_reconcile', False)
    gates['G8_persistence_as_w101'], detail['G8'] = W101L.persistence_check(rec, eval_dir)
    gates['G9_ess_ageing_readback'] = c.get('ess_ageing_readback_all_match', False)
    xc = X.acceptance_cross_check(eval_dir)
    gates['G11_acceptance_cross_check'] = xc['agrees']
    detail['G11'] = xc
    gates['G13_holds_inert_through_first_pass_held_after'], detail['G13'] = hold_checks(cell, eval_dir, rec)
    rows = _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))
    gates['G14_all_block_capture'], detail['G14'] = W101L.block_capture(eval_dir, rec, rows)
    gates['G15_stopping_consistent'], detail['G15'] = stopping_check(cell, rec, eval_dir)
    gates['G16_lambda_sidecar_complete'], detail['G16'] = W101L.lambda_sidecar_check(eval_dir, rec)
    gates['G17_rule_v3_replay_reproduces_in_cycle'], detail['G17'] = settling_replay_check(cell, eval_dir)
    gates['G18_line_fields_every_cycle'], detail['G18'] = line_fields_check(eval_dir)
    if V.CELLS[cell]['gated']:
        rg = replay_gate_full(cell, eval_dir)
        gates['G19_replay_bitwise_1_k0_every_field'] = rg['bitwise_through_k0']
        detail['G19'] = rg
    else:
        detail['G19'] = {'skipped': 'ungated cell (first C2 evaluation): no replay reference'}
    gates['G20_production_counters_in_per_cycle_record'] = all(
        'consecutive_converged_cycles' in r and 'boyd_all_pass' in r for r in rows)
    gates['G21_creep_captures_complete'], detail['G21'] = L118.creep_capture_check(eval_dir, rec, rows)
    gates['G22_t_sum_in_cycle_equals_stride_and_terminal'], detail['G22'] = L118.t_sum_check(eval_dir)
    gates['G23_overlap_recorded'], detail['G23'] = overlap_check(cell, rec)
    try:
        gates['G24_exit_capture_equals_the_solve_records_and_esso_logs'], detail['G24'] = exit_crosscheck(
            eval_dir, rec=rec)
    except Exception as error:  # noqa: BLE001 -- recorded; the gate FAILS
        gates['G24_exit_capture_equals_the_solve_records_and_esso_logs'] = False
        detail['G24'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
    gates['G25_record_status_follows_the_settling_decision'], detail['G25'] = status_label_check(rec, eval_dir)
    return gates, detail, rec


# ======================================================================================================================
#  the per-cell report
# ======================================================================================================================
def _ols_slope(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def cell_report(cell, eval_dir, rec):
    """The frozen per-cell report (DEFINITIONS['per_cell'] and the uncertified form) from the run's artefacts."""
    rows = {r['cycle']: r for r in _read_jsonl(os.path.join(eval_dir, 'per_cycle_record.jsonl'))}
    lines = {x['cycle']: x for x in _read_jsonl(os.path.join(eval_dir, V.CYCLE_FILE))}
    dec = _decision(eval_dir) or {}
    c = V.CELLS[cell]
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
    rep = {'cell': cell, 'item': c['item'], 'claim_group': V.GROUP_OF_ITEM[c['item']], 'gated': c['gated'],
           'eval_key': rec.get('eval_key'), 'candidate_key': rec.get('candidate_key'),
           'candidate_canonical': rec.get('candidate_canonical'), 'original_eval_key': c['orig_eval_key'],
           'k0_run': fp, 'first_k0_v2': summ.get('first_k0_v2'), 'first_k0_alpha': summ.get('first_k0_alpha'),
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
           'non_optimal_cycles': dec.get('non_optimal_cycles'), 'gamma_report_only': dec.get('gamma_report_only'),
           'terminal_step_abs': abs(steps[-1]) if steps else None,
           'terminal_step_over_EPS0': (abs(steps[-1]) / SC3.EPS0) if steps else None,
           'rule_ten_last_cycle': ((rows[last]['objective_change_abs'] / rows[last]['objective_tolerance'])
                                   if rows[last].get('objective_change_abs') is not None
                                   and rows[last].get('objective_tolerance') else None),
           'boyd_lapses_after_k0': sum(1 for x in after if not x.get('boyd_k')),
           'pf_primal_ratio_after_k0': {'max': max((x.get('boyd_pf_primal_ratio') or 0.0) for x in after) if after
                                        else None, 'last': after[-1].get('boyd_pf_primal_ratio') if after else None},
           'pf_primal_slope_last_50': _ols_slope(win, [rows[k]['boyd_pf_primal_ratio'] for k in win]),
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
        ref = {r['cycle']: r for r in _read_jsonl(_abs(V.reference_path(cell)))}
        q_n_old = ref[c['N_old']]['gross_operational_cost']
        ov = summ.get('overlap_k0_plus_1_to_N_old') or []
        rep.update({'Q_N_old': q_n_old, 's_signed': (q_end - q_n_old) if q_end is not None else None,
                    's_resolution_band_width': dec.get('band_width'),
                    'overlap': ov, 'overlap_at_N_old': next((o for o in ov if o['cycle'] == c['N_old']), None),
                    'replay_bitwise_through': summ.get('replay_bitwise_through_cycle'),
                    'replay_first_divergence': summ.get('replay_first_divergence')})
    rep['view'] = view_from_report(rep)
    return rep


def view_from_report(rep):
    """The scorer's view of a cell (new or reference): status, Q, t, Q_cc, Q_net, salvage, band, gap, slack."""
    certified = rep.get('status') == 'certified'
    q, t = rep.get('Q_end'), rep.get('t_sum_end')
    s = rep.get('s_signed')
    return {'status': rep.get('status'), 'Q': q, 't': t, 'Q_cc': (q + t) if (q is not None and t is not None) else None,
            'Q_net': rep.get('Q_net_end'), 'salvage': rep.get('terminal_salvage_value_end'),
            'band': rep.get('band_width'), 'gated': rep.get('gated'),
            'gap': None if certified or t is None else abs(t),
            'slack': None if certified else (abs(s) if (rep.get('gated') and s is not None) else None),
            'gap_refused': rep.get('gap_refused'), 'label': rep.get('label')}


# ======================================================================================================================
#  the references and the scorer (pure below the loaders)
# ======================================================================================================================
def reference_views():
    """The settled references as scorer views, from committed artefacts (sha256 recorded)."""
    out = {}
    inputs = {}
    for p8, ref in REFERENCES.items():
        if ref['kind'] == 'w101':
            ed = ref['eval_dir']
            dec = _load(os.path.join(ed, 'settling_decision.json'))
            rows = {r['cycle']: r for r in _read_jsonl(_abs(os.path.join(ed, 'per_cycle_record.jsonl')))}
            det = _load(os.path.join(ed, 'interface_settlement_detail_s31c.json'))
            k = dec['k_star']
            if det['cycles_run'] != k or max(rows) != k:
                raise RuntimeError(f'{ed}: the reference did not end at its k* ({k})')
            q, t = rows[k]['gross_operational_cost'], det['t_tso_plus_t_dso_terminal']
            out[p8] = {'status': dec['status'], 'Q': q, 't': t, 'Q_cc': q + t, 'Q_net': rows[k]['recourse'],
                       'salvage': rows[k]['terminal_salvage_value'], 'band': dec['band_width'], 'gated': True,
                       'gap': None, 'slack': None, 'k_star': k, 'name': ref['name']}
            for f in ('settling_decision.json', 'per_cycle_record.jsonl', 'interface_settlement_detail_s31c.json'):
                inputs[os.path.join(ed, f)] = _sha(os.path.join(ed, f))
        else:
            summ = _load(W118_SUMMARY)
            r = summ['reports'][ref['cell']]
            q, t = r['Q_at_cap'], r['t_sum_at_cap']
            out[p8] = {'status': r['status'], 'Q': q, 't': t, 'Q_cc': q + t, 'Q_net': r['net_operational_recourse_last'],
                       'salvage': r['terminal_salvage_value_last'], 'band': r['band_width'], 'gated': True,
                       'gap': abs(t), 'slack': abs(r['s_signed']), 'k_cap': r['k_cap'], 'name': ref['name'],
                       'gap_refused': bool(r.get('gap_refusals')), 'label': GAP_REFUSED_LABEL}
            inputs[W118_SUMMARY] = _sha(W118_SUMMARY)
    return out, inputs


def _verdict_settled(d, res):
    return 'determinate' if abs(d) > res else 'within resolution'


def resolve(d_q, d_cc, views):
    """The resolution rule of a difference: settled-vs-settled (sum of bands; gross verdict, Q_cc beside) or the
    uncertified-form determinacy rule (3 x max(|gap|, |slack|) over the uncertified cells, both terms)."""
    unc = [v for v in views if v.get('status') != 'certified']
    m_cc = abs(d_cc) if (d_q > 0) == (d_cc > 0) else -abs(d_cc)
    if not unc:
        res = sum(v['band'] for v in views)
        return {'rule': 'settled_vs_settled', 'resolution': res, 'verdict': _verdict_settled(d_q, res),
                'verdict_Qcc_report_only': 'determinate' if m_cc > res else 'within resolution',
                'margin_over_resolution': abs(d_q) / res if res else None}
    comps = []
    for v in unc:
        if v.get('slack') is None or v.get('gap') is None:
            return {'rule': 'uncertified_form', 'bar': None, 'verdict': 'indeterminate (slack undefined)',
                    'note': 'an ungated uncertified cell has no s = Q(cap) - Q_N_old'}
        comps += [v['gap'], v['slack']]
    bar = 3.0 * max(comps)
    det = abs(d_q) > bar and m_cc > bar
    return {'rule': 'uncertified_form', 'bar': bar, 'bar_components': comps, 'margin_Q': abs(d_q), 'margin_Qcc': m_cc,
            'verdict': 'determinate' if det else 'within the uncertified bar',
            'margin_over_bar_Q': abs(d_q) / bar if bar else None, 'margin_over_bar_Qcc': m_cc / bar if bar else None}


def score_claim(cl, rv, ov):
    """One claim (W117 definition) on two views (ref rv, other ov). Pure."""
    Ir, Io = cl['I_ref'], cl['I_other']

    def d(qr, qo):
        if cl['form'] == 'F':
            return (qo + Io) - (qr + Ir)
        return (qr - qo) - (Io - Ir)
    out = {'claim_id': cl['claim_id'], 'item': cl['item'], 'statement': cl['statement'], 'form': cl['form'],
           'claim_type': cl['claim_type'], 'net_of_salvage': cl['net_of_salvage'],
           'ref': {k: rv.get(k) for k in ('status', 'Q', 't', 'Q_cc', 'Q_net', 'salvage', 'band', 'gap', 'slack')},
           'other': {k: ov.get(k) for k in ('status', 'Q', 't', 'Q_cc', 'Q_net', 'salvage', 'band', 'gap', 'slack')},
           'I_ref': Ir, 'I_other': Io, 'objective_convention_primary': 'net of salvage' if cl['net_of_salvage']
           else 'gross (settlement excluded)'}
    if any(v.get('Q') is None or v.get('Q_cc') is None for v in (rv, ov)):
        out.update({'verdict': 'not scored (a cell has no result yet)'})
        return out
    d_q = d(rv['Q'], ov['Q'])
    d_cc = d(rv['Q_cc'], ov['Q_cc'])
    out.update({'d_Q': d_q, 'd_Qcc': d_cc, 'd_Qcc_minus_d_Q': d_cc - d_q, 'sign_same_in_Q_and_Qcc': (d_q > 0) == (d_cc > 0),
                'gross': resolve(d_q, d_cc, (rv, ov))})
    if cl['form'] == 'F' and (cl['item'] == 'G' or cl['net_of_salvage']):
        if rv.get('Q_net') is not None and ov.get('Q_net') is not None:
            d_n = d(rv['Q_net'], ov['Q_net'])
            d_ncc = d(rv['Q_net'] + rv['t'], ov['Q_net'] + ov['t'])
            out['net_of_salvage'] = {'d_net': d_n, 'd_net_cc': d_ncc, 'resolution': resolve(d_n, d_ncc, (rv, ov))}
    if cl['net_of_salvage']:
        primary = (out.get('net_of_salvage') or {}).get('resolution')
        primary_d = (out.get('net_of_salvage') or {}).get('d_net')
    else:
        primary, primary_d = out['gross'], d_q
    out['verdict'] = (primary or {}).get('verdict') if primary else 'not scored (no net figure)'
    out['d_primary'] = primary_d
    out['sign_primary_positive'] = (primary_d > 0) if primary_d is not None else None
    return out


def d_fit(values, q0_band, bands, p_cost, e_cost, labels_ep):
    """The D affine fit (p515_s47_baseline_tables.py / W117). `values` = {'Q': {label: value}, 'Q_cc': {...}} with value
    = Q(0) - Q(x); `labels_ep` = {label: (E, P)}; `bands` = {label: band width}. Pure."""
    labels = list(D_FIT_NODE7_LABELS)
    X_ = np.array([[1.0, labels_ep[lb][0], labels_ep[lb][1]] for lb in labels])
    out = {}
    for conv in ('Q', 'Q_cc'):
        y = np.array([values[conv][lb] for lb in labels])
        coef, *_ = np.linalg.lstsq(X_, y, rcond=None)
        res = y - X_ @ coef
        s2 = float(res @ res) / (len(y) - 3)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X_.T @ X_)))
        a, b, c = (float(v) for v in coef)
        e_star = b + c / 4.0 - p_cost / 4.0
        out[conv] = {'a': a, 'b_per_MWh': b, 'c_per_MVA': c, 'se': [float(v) for v in se],
                     'breakeven_marginal_4h_energy_cost': e_star, 'margin_to_energy_cost_per_MWh': e_cost - e_star,
                     'se_breakeven_independent_terms_approx': float(np.sqrt(se[1] ** 2 + (se[2] / 4.0) ** 2)),
                     'residual_rms': float(np.sqrt(res @ res / len(y))), 'residual_max_abs': float(abs(res).max())}
    # e* = w . y + const: w = rows b and c/4 of (X^T X)^-1 X^T
    pinv = np.linalg.inv(X_.T @ X_) @ X_.T
    w = pinv[1] + pinv[2] / 4.0
    bound = float(sum(abs(wi) * bands[lb] for wi, lb in zip(w, labels)) + abs(float(w.sum())) * q0_band)
    out['slack_bound_report_only'] = {'value': bound, 'weights': {lb: float(wi) for wi, lb in zip(w, labels)},
                                      'formula': DEFINITIONS['D_fit']['slack_bound_report_only']}
    out['point_set'] = labels
    return out


def score_predictions(reports):
    out = {}
    dz = {}
    for cell in V.DEAD_ZONE_CANDIDATES:
        r = reports.get(cell)
        if r is None:
            dz[cell] = {'held': None, 'status': 'no result yet'}
            continue
        dz[cell] = {'held': r.get('status') == 'uncertified' and bool(r.get('gap_refused')), 'status': r.get('status'),
                    'gap_refused': r.get('gap_refused'), 'k_star': r.get('k_star')}
    out['dead_zone_candidates'] = dz
    out['dead_zone_borderline'] = {c: ({'status': reports[c].get('status'), 'gap_refused': reports[c].get('gap_refused'),
                                        'k_star': reports[c].get('k_star')} if c in reports else 'no result yet')
                                   for c in V.DEAD_ZONE_BORDERLINE}
    out['gate_ability'] = {c: ({'held': reports[c].get('replay_bitwise_through') == V.CELLS[c]['k0']
                                and not reports[c].get('replay_first_divergence')} if c in reports else 'no result yet')
                           for c in V.GATED_CELLS}
    return out


# ======================================================================================================================
#  scorer self-tests (zero solves; committed inputs)
# ======================================================================================================================
def scorer_self_tests():
    res = {}
    w117, _sha117 = K._pinned_json(K.W117)
    # (1) score_claim reproduces W117's d_Q and d_Qcc on every W117 claim, fed W117's own cell values
    mism = []
    for cl in w117['claims']:
        def v(side):
            s = cl[side]
            return {'status': 'certified', 'Q': s['Q'], 'Q_cc': s['Q_cc'], 't': s['t_sum'],
                    'Q_net': s['Q'] - s['salvage'], 'salvage': s['salvage'], 'band': 0.0}
        sc = score_claim({**cl}, v('ref'), v('other'))
        d_q = sc['net_of_salvage']['d_net'] if cl['net_of_salvage'] else sc['d_Q']
        if not (d_q == cl['d_Q'] or abs(d_q - cl['d_Q']) <= 1e-6) or (not cl['net_of_salvage']
                                                                        and sc['d_Qcc'] != cl['d_Qcc']):
            mism.append({'claim': cl['claim_id'], 'mine': d_q, 'w117': cl['d_Q'], 'mine_cc': sc['d_Qcc'],
                         'w117_cc': cl['d_Qcc']})
    res['S1_claim_formula_reproduces_w117'] = {'ok': not mism, 'n_claims': len(w117['claims']),
                                               'mismatches_first10': mism[:10]}
    # (2) d_fit reproduces the committed Phase A fit (old certificates) and W117's D fits
    bt = _load(BASELINE_TABLES)
    unit = _load(W2_TABLE)['candidates']['n7_4h_e1']
    p_cost, e_cost = unit['I_new_power_eur'] / 0.25, unit['I_new_energy_eur'] / 1.0
    cells = w117['cells']
    by_label = {c['label']: c for c in cells.values() if 'campaign_s47_a1a_baseline' in c['eval_dir']}
    x0 = cells['7aa017f0']
    vals = {conv: {lb: x0[conv] - by_label[lb][conv] for lb in D_FIT_NODE7_LABELS} for conv in ('Q', 'Q_cc')}
    ep = {lb: _ep_of(lb) for lb in D_FIT_NODE7_LABELS}
    fit = d_fit(vals, 0.0, {lb: 0.0 for lb in D_FIT_NODE7_LABELS}, p_cost, e_cost, ep)
    n7 = bt['node7_fit']
    w117_fits = w117['D_break_even']['fits']
    ok2 = (abs(fit['Q']['b_per_MWh'] - n7['b']) < 1e-6 and abs(fit['Q']['a'] - n7['a']) < 1e-6
           and abs(fit['Q']['c_per_MVA'] - n7['c']) < 1e-6
           and abs(fit['Q']['breakeven_marginal_4h_energy_cost'] - bt['breakeven']['marginal_4h_eur_per_mwh']) < 1e-6
           and abs(fit['Q']['se_breakeven_independent_terms_approx'] - bt['breakeven']['marginal_4h_se_eur_per_mwh'])
           < 1e-6 and all(abs(fit[c]['breakeven_marginal_4h_energy_cost']
                               - w117_fits[c]['breakeven_marginal_4h_energy_cost']) < 1e-6 for c in ('Q', 'Q_cc')))
    res['S2_d_fit_reproduces_the_committed_phase_a_fit'] = {
        'ok': bool(ok2), 'mine': {c: {k: fit[c][k] for k in ('a', 'b_per_MWh', 'c_per_MVA',
                                                             'breakeven_marginal_4h_energy_cost')} for c in ('Q', 'Q_cc')},
        'committed': {'a': n7['a'], 'b': n7['b'], 'c': n7['c'], 'breakeven': bt['breakeven']['marginal_4h_eur_per_mwh']}}
    # (3) the W130 F2 plan-vs-corner bar: the W118 incumbent (uncertified) against a certified cell -> 27,703.27
    refs, _ = reference_views()
    inc = refs['5ca4f86c']
    corner = {'status': 'certified', 'Q': 1.0, 'Q_cc': 1.0, 't': 0.0, 'band': 100.0}
    r3 = resolve(103578.54, 112885.66, (inc, corner))
    res['S3_w130_bar_reproduced'] = {'ok': bool(abs(r3['bar'] - 27703.27) < 0.01 and r3['verdict'] == 'determinate'),
                                     'bar': r3['bar'], 'w130_bar': 27703.27}
    # (4) the resolution rules on synthetic views
    a = {'status': 'certified', 'band': 10.0}
    b = {'status': 'certified', 'band': 5.0}
    u = {'status': 'uncertified', 'gap': 100.0, 'slack': 40.0}
    ug = {'status': 'uncertified', 'gap': 100.0, 'slack': None}
    t4 = {'settled_sum': resolve(16.0, 16.0, (a, b))['resolution'] == 15.0 and resolve(16.0, 1.0, (a, b))['verdict']
          == 'determinate' and resolve(14.0, 14.0, (a, b))['verdict'] == 'within resolution',
          'uncertified_both_terms': resolve(301.0, 299.0, (a, u))['verdict'] == 'within the uncertified bar'
          and resolve(301.0, 301.0, (a, u))['verdict'] == 'determinate' and resolve(301.0, 301.0, (a, u))['bar'] == 300.0,
          'sign_flip_never_determinate': resolve(1000.0, -1000.0, (a, u))['verdict'] == 'within the uncertified bar',
          'ungated_uncertified_indeterminate': resolve(1e9, 1e9, (a, ug))['verdict'] == 'indeterminate (slack undefined)',
          'two_uncertified_take_the_max': resolve(1.0, 1.0, (u, {'status': 'uncertified', 'gap': 5.0, 'slack': 200.0}))[
              'bar'] == 600.0}
    res['S4_resolution_rules'] = {'ok': all(t4.values()), 'tests': t4}
    return res, all(v['ok'] for v in res.values())


def _ep_of(label):
    """(E, P) of a node-7 A1a label n7_{d}h_e{E}: P = E / d."""
    m = re.match(r'n7_(\d)h_e(\d)$', label)
    d, e = int(m.group(1)), float(m.group(2))
    return e, e / d


# ======================================================================================================================
#  post-run evaluator self-tests (synthetic eval dirs from the REAL wrappers; tampered negative controls)
# ======================================================================================================================
def _synthetic_run_dir(tmp, cell, variant):
    d = K.drive(cell, 'creep' if variant == 'creep' else 'certify',
                nonopt=({V.CELLS[cell]['N_old'] + 20: ('ESSO|9',)} if variant == 'nonopt' else None))
    files, st = d['files'], d['state']
    lines = {x['cycle']: x for x in files[V.CYCLE_FILE]}
    creep = {x['cycle']: x for x in files[V.CREEP_FILE]}
    if variant == 'tamper_t_sum_30':
        lines[30]['t_sum'] = lines[30]['t_sum'] + 1.0
    if variant == 'tamper_hold_flag':
        lines[max(lines) - 5]['aa']['hold'] = False
    if variant == 'tamper_decision':
        files[V.DECISION_FILE][0]['k_star'] = files[V.DECISION_FILE][0]['k_star'] - 1
    if variant == 'tamper_exit':
        c = V.CELLS[cell]['N_old'] + 20
        lines[c]['all_optimal_k'] = False
    for fname, objs in files.items():
        with open(os.path.join(tmp, fname), 'w') as handle:
            if fname == V.DECISION_FILE:
                GRIO.dump(objs[0], handle, default=GRIO.json_default, indent=1, sort_keys=True)
            elif fname == V.CYCLE_FILE:
                for c in sorted(lines):
                    handle.write(GRIO.dumps(lines[c], default=GRIO.json_default) + '\n')
            else:
                for o in objs:
                    handle.write(GRIO.dumps(o, default=GRIO.json_default) + '\n')
    ref = st.reference if st.gated else {}
    rows, g_rows = [], []
    g_orig = ({r['cycle']: r for r in json.load(open(_abs(os.path.join(V.original_eval_dir(cell), 'g_s39_D.json'))))
               ['cycle_trajectory']} if st.gated else {})
    for c in sorted(lines):
        x = lines[c]
        if st.gated and c <= V.CELLS[cell]['N_old']:
            row = dict(ref[c])
            g = dict(g_orig[c])
        else:
            row = {'cycle': c, 'local_solves_ok': True, 'recourse': x['net_operational_recourse'],
                   'gross_operational_cost': x['gross'], 'terminal_salvage_value': x['terminal_salvage_value'],
                   'objective_change_abs': abs(x['step'] or 0.0), 'objective_tolerance': 65000.0,
                   'objective_change_ratio': abs(x['step'] or 0.0) / 65000.0, 'cycle_convergence': x['boyd_k'],
                   'consecutive_converged_cycles': x['consecutive_converged_cycles_tracked'],
                   'boyd_all_pass': x['boyd_k'], 'boyd_stop': x['boyd_k'],
                   **{f: x['boyd_ratios'][f] for f in R.BOYD_RATIO_FIELDS},
                   **{f'boyd_{g_}_channel_pass': creep[c]['boyd'][g_]['channel_pass'] for g_ in R.CHANNELS},
                   **{f'rho_{g_}_after': x['rho']['rho_after'][g_] for g_ in R.CHANNELS},
                   **{f'rho_{g_}_action': x['rho']['actions'][g_] for g_ in R.CHANNELS},
                   'rho_freeze_active': True, 'efc_per_day_max': 1.0}
            g = {'cycle': c}
        for g_ in R.CHANNELS:
            for f, v in creep[c]['boyd'][g_].items():
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
    n_e = len(K.K118.NODES) * len(K.K118.YEARS) * len(K.K118.DAYS) * K.K118.PERIODS
    with open(os.path.join(tmp, 'pf_entry_stride_s39_D.jsonl'), 'w') as handle:
        for c in sorted(lines):
            t = lines[c]['t_sum'] if variant != 'tamper_t_sum_30' or c != 30 else lines[c]['t_sum'] - 1.0
            gap = t / n_e
            ents = [{'node_id': n, 'year': str(y), 'day': d_, 'power_type': 'p', 'period': p, 'x_dso': 10.0 + gap,
                     'z_tso_current': 10.0, 'lambda_dso': 0.0, 's_base_dso': 100.0, 'rho_pf': 1.0, 'r': 0.0}
                    for n in K.K118.NODES for y in K.K118.YEARS for d_ in K.K118.DAYS for p in range(K.K118.PERIODS)]
            handle.write(json.dumps({'cycle': c, 'identity_holds': True, 'production_boyd_pf_r': 0.0,
                                     'entries': ents}) + '\n')
    last = max(lines)
    detail = {'t_tso_plus_t_dso_terminal': lines[last]['t_sum'], 'cycles_run': last,
              'interface_reporting_detail': {str(n): {str(y): {d_: {'periods': {str(p): {'price_per_mwh': 1.0}
                                                                                  for p in range(K.K118.PERIODS)}}
                                                               for d_ in K.K118.DAYS} for y in K.K118.YEARS}
                                             for n in K.K118.NODES},
              'interface_consensus_residual_per_dso': {str(n): {'periods': {f'{y}|{d_}|{p}': {'admm_block_weight': 1.0}
                                                                            for y in K.K118.YEARS for d_ in K.K118.DAYS
                                                                            for p in range(K.K118.PERIODS)},
                                                                'sum_pi_baseMVA_residual_weighted': 0.0}
                                                       for n in K.K118.NODES}}
    with open(os.path.join(tmp, 'interface_settlement_detail_s31c.json'), 'w') as handle:
        json.dump(detail, handle)
    # the record as the harness writer leaves it: production's trajectory view, then the settling-status fix
    rec = H._apply_settling_resettle_status({'cycles_run': last, 'settling_resettle_summary': st.summary(),
                                             'status': 'certified', 'barrier': False, 'barrier_cause': None,
                                             'certified_cost': 1.0, 'certification_cycle': last,
                                             'terminal_gross_operational_cost': 1.0})
    return rec, rows


def post_run_evaluator_self_tests():
    """The NEW / adapted post-run evaluators (G13, G15, G17, G18, G19, G22, G23, G25, the cell report) on synthetic eval
    dirs built by the real wrappers, with tampered negative controls; G24's comparison on a committed W118 cell (W131's
    readers) with a planted disagreement."""
    out = {}
    cases = (('b_2a0ba8b2', 'pass', None, True), ('b_2a0ba8b2', 'tampered_row_40', 'tamper_row_40', False),
             ('b_2a0ba8b2', 'tampered_t_sum_30', 'tamper_t_sum_30', False),
             ('b_2a0ba8b2', 'tampered_hold_flag', 'tamper_hold_flag', False),
             ('b_2a0ba8b2', 'tampered_decision', 'tamper_decision', False),
             ('h_74eda68d', 'non_optimal_lapse', 'nonopt', True),
             ('h_74eda68d', 'tampered_exit_flag', 'tamper_exit', False),
             ('g_37b5c499', 'ungated_rule_cap', 'creep', True))
    for cell, name, variant, expect in cases:
        tmp = tempfile.mkdtemp(prefix='w132_selftest_')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rec, rows = _synthetic_run_dir(tmp, cell, variant)
            h_ok, _h = hold_checks(cell, tmp, rec)
            s_ok, _s = stopping_check(cell, rec, tmp)
            k_ok, _k = settling_replay_check(cell, tmp)
            f_ok, _f = line_fields_check(tmp)
            c_ok, _c = L118.creep_capture_check(tmp, rec, rows)
            t_ok, t_d = L118.t_sum_check(tmp, os.path.relpath(os.path.join(tmp, 'pf_entry_stride_s39_D.jsonl'), REPO),
                                         os.path.relpath(os.path.join(tmp, 'interface_settlement_detail_s31c.json'),
                                                         REPO))
            o_ok, _o = overlap_check(cell, rec)
            l_ok, _l = status_label_check(rec, tmp)
            rg = replay_gate_full(cell, tmp) if V.CELLS[cell]['gated'] else {'bitwise_through_k0': True}
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
                                                                          'first_k0_alpha', 'non_optimal_cycles',
                                                                          'pf_primal_slope_last_50', 'label')}
                                                 if rep else None)}
        except Exception as error:  # noqa: BLE001
            out[name] = {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        finally:
            shutil.rmtree(tmp)
    # G24's comparison on a committed W118 cell: lines synthesised from W131's own final-attempt readers (positive),
    # then one class flipped (negative)
    try:
        cell = 'pb_y2025_n5'
        eval_rel = os.path.join(K.W118_ROOT_REL, K.W118_CELL_DIRS[cell])
        n = len(_read_jsonl(_abs(os.path.join(eval_rel, 'per_cycle_record.jsonl'))))
        net, _m = W131.network_final_attempts(eval_rel, n)
        recs = _read_jsonl(_abs(os.path.join(eval_rel, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE)))
        esso, _e = W131.esso_final_attempts(eval_rel, os.path.dirname(recs[0]['log_path']), n)
        lines = {k: {'cycle': k, 'ipopt_exit_by_block': {}} for k in range(1, n + 1)}
        init_nonopt = {}
        for (rnd, agent, _nw, year, day), v in net.items():
            cls = H.ipopt_exit_class(v['final_exit'])
            if rnd == 0:
                if cls != V.OPTIMAL:
                    init_nonopt[_net_key(agent, year, day)] = {'class': cls}
                continue
            lines[rnd]['ipopt_exit_by_block'][_net_key(agent, year, day)] = {'class': cls}
        for (k, node), v in esso.items():
            if k:
                lines[k]['ipopt_exit_by_block'][f'ESSO|{node}'] = {'class': H.ipopt_exit_class(v['final_exit'])}
        for x in lines.values():
            x['all_optimal_k'] = all(b['class'] == V.OPTIMAL for b in x['ipopt_exit_by_block'].values())
        rec = {'settling_resettle_summary': {'exit_capture_init_round_0': {'non_optimal_blocks': init_nonopt}}}
        pos_ok, pos = exit_crosscheck(_abs(eval_rel), lines=lines, rec=rec)
        tampered = json.loads(json.dumps(lines))
        tampered = {int(k): v for k, v in tampered.items()}
        tampered[167]['ipopt_exit_by_block']['TSO|2035|Spring']['class'] = V.OPTIMAL
        neg_ok, neg = exit_crosscheck(_abs(eval_rel), lines=tampered, rec=rec)
        out['G24_exit_crosscheck_on_committed_w118_cell'] = {
            'ok': bool(pos_ok and not neg_ok and neg['n_disagreements'] == 1
                       and lines[167]['all_optimal_k'] is False),
            'positive': pos['parts'], 'negative_disagreements': neg['disagreements_first20'],
            'cycle_167_non_optimal_blocks': sorted(k for k, v in lines[167]['ipopt_exit_by_block'].items()
                                                   if v['class'] != V.OPTIMAL)}
    except Exception as error:  # noqa: BLE001
        out['G24_exit_crosscheck_on_committed_w118_cell'] = {'ok': False, 'error': f'{type(error).__name__}: {error}',
                                                             'traceback': traceback.format_exc()}
    sc, sc_ok = scorer_self_tests()
    out['scorer'] = {'ok': sc_ok, 'tests': sc}
    return out, all(v.get('ok') for v in out.values())


# ======================================================================================================================
#  checks state, verbatim, basis
# ======================================================================================================================
def _checks_file_state():
    rel = os.path.join(K.OUT_DIR_REL, K.OUT_FILE)
    doc = _load(rel) if os.path.isfile(_abs(rel)) else {}
    return {'path': rel, 'sha256': _sha(rel) if doc else None,
            'manifest': os.path.join(K.OUT_DIR_REL, K.OUT_MANIFEST),
            'committed_clean': _committed_clean(rel) if doc else False,
            'all_hold': doc.get('all_hold'), 'all_hold_including_typing_test': doc.get('all_hold_including_typing_test'),
            'typing_test': doc.get('W_bool_typing_test'),
            'guards_verify_0_failures': {k: v.get('verify_0_failures') for k, v in (doc.get('guards') or {}).items()},
            'code_sha256_at_check': doc.get('code_sha256')}


def _run_checks_inline():
    with contextlib.redirect_stdout(io.StringIO()):
        return K.run_all_checks()


def _failing_check_items(checks):
    out = {}
    for sid, sec in ((checks or {}).get('sections') or {}).items():
        if (sec or {}).get('holds') is True:
            continue
        r = (sec or {}).get('result') or {}
        items = []
        if 'error' in r:
            items.append(f"error: {r['error']}")
        for k, v in (r.get('parts') or {}).items():
            if v is not True:
                items.append(f'parts.{k}')
        for k, v in (r.get('tests') or {}).items():
            if isinstance(v, dict) and v.get('holds', v.get('ok')) is not True:
                items.append(f'tests.{k}')
        for k, v in (r.get('cells') or {}).items() if isinstance(r.get('cells'), dict) else []:
            if isinstance(v, dict) and v.get('ok') is False:
                items.append(f'cells.{k}')
        out[sid] = items or ['holds is not True (no itemised field)']
    return out


def verbatim_check():
    texts = {f: _norm(open(_abs(f), encoding='utf-8').read()) for f in (BRIEF, TASKS)}
    found = {f'{f}:{k}': _norm(v) in texts[f] for (f, k), v in VERBATIM.items()}
    return {'files': {f: {'sha256_at_freeze': _sha(f), 'git_state': L._git_state(f)} for f in (BRIEF, TASKS)},
            'found_whitespace_normalised': found, 'all_found': all(found.values())}


def _common_checks():
    failures = []
    for rel in EXTRA_CLEAN_FILES + tuple(H.PRODUCTION_FILES_TO_CHECK_CLEAN):
        if os.path.isfile(_abs(rel)) and not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    for cell in V.CELL_ORDER:
        rel = V.reference_path(cell)
        if _sha(rel) != V.CELLS[cell]['per_cycle_record_sha256'] or not _committed_clean(rel):
            failures.append(f'{cell} original record not as committed: {rel}')
    for rel in (PREDECESSOR_REL, K.W131['path'], K.W117['path'], W118_SUMMARY, BASELINE_TABLES, W2_TABLE):
        if not _committed_clean(rel):
            failures.append(f'{rel} not committed / clean')
    others = _own_process_alive()
    if others:
        failures.append(f'another copy of a W132 / W118 / W105 / W101 / W98 launcher is alive: {others}')
    return failures


def _checks_output_ok(failures):
    cf = _checks_file_state()
    if not (cf['all_hold_including_typing_test'] is True and cf['committed_clean']
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
    tag = 'W132-FREEZE-CELLS'
    failures = _common_checks()
    cf = _checks_output_ok(failures)
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'inline re-run of the zero-solve checks fails: {_failing_check_items(checks_inline)}')
    post_tests, post_ok = post_run_evaluator_self_tests()
    if not post_ok:
        failures.append(f'post-run evaluator / scorer self-tests fail: {[k for k, v in post_tests.items() if not v.get("ok")]}')
    for cell in V.CELL_ORDER:
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
    for cell in V.CELL_ORDER:
        pre = pre_launch_assertion(cell)
        c = V.CELLS[cell]
        extra = {'campaign_script': SCRIPT_NAME, 'campaign_script_sha256': _sha(SCRIPT_NAME), 'stage_text': STAGE_TEXT,
                 'stage_spec': ('frozen_s53_resettle_spec_v3 (frozen AFTER this campaign spec) pins this spec by its '
                                'sha256 and holds the exact launch command'),
                 'label': V.LABEL, 'cell': cell, 'item': c['item'], 'claim_group': V.GROUP_OF_ITEM[c['item']],
                 'original': {'campaign_id': c['orig_campaign_id'], 'eval_key': c['orig_eval_key'],
                              'eval_dir': V.original_eval_dir(cell), 'N_old': c['N_old'], 'k0': c['k0']},
                 'expected_eval_key': pre['resettle_key'], 'objective_convention': DEFINITIONS['objective_convention'],
                 'solve_claim': {'parent': 'never solves (every launcher guard permitted=(), verify(0))',
                                 'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
                 'zero_solve_checks_output': checks_pin, 'pre_launch_assertion_at_freeze': pre,
                 'cell_order': list(V.CELL_ORDER), 'code_sha256': {rel: _sha(rel) for rel in CODE_PINNED}}
        spec_path, spec_sha, spec = H.freeze_campaign_spec(
            campaign_root(cell), CAMPAIGN_IDS[cell], entries(cell), configuration=configuration(cell),
            cap=V.spec_cap(cell), concurrency=CONCURRENCY,
            authority=[f'{BRIEF} Addendum 58', f'{BRIEF} Addendum 57', 'TASKS.md Addendum 58 Planner rulings (82fa1b61)',
                       'Planner task W132'], required_consecutive_cycles=10, extra=extra)
        checks = validate_campaign_spec(cell, spec)
        pre_frozen = pre_launch_assertion(cell, spec)
        ok = all(checks.values()) and pre_frozen['holds']
        all_ok = all_ok and ok
        e = spec['candidates'][0]
        _log(f'[{tag}] {cell}: frozen {os.path.relpath(spec_path, REPO)} sha256={spec_sha} eval_key={e["eval_key"]} '
             f'cap={spec["cap"]} checks={all(checks.values())} failing={[k for k, v in checks.items() if not v]} '
             f'pre-launch={pre_frozen["holds"]}')
    _finish(0 if all_ok else 1, f'freeze-cells {"OK" if all_ok else "NOT OK"} -- next: commit, then --freeze-spec')


def cell_spec_state():
    """{cell: {path, sha256, committed_clean, checks, pre_launch}} of the 38 frozen campaign specs."""
    out = {}
    for cell in V.CELL_ORDER:
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
        raise RuntimeError('frozen v3 re-settling stage spec not found')
    return rel, sha, _load(rel)


def stage_spec_content(checks_inline, cf, post_tests, verb, solver, prov, mem, wall, specs, o_section, dataset, points,
                       refs, ref_inputs):
    code = {rel: _sha(rel) for rel in CODE_PINNED}
    production = {rel: _sha(rel) for rel in H.PRODUCTION_FILES_TO_CHECK_CLEAN if os.path.isfile(_abs(rel))}
    pred_sha = _sha(PREDECESSOR_REL)
    cells = {}
    for i, cell in enumerate(V.CELL_ORDER):
        c = V.CELLS[cell]
        o = o_section['cells'][cell]
        s = specs[cell]
        cells[cell] = {
            'launch_index_1_based': i + 1, 'item': c['item'], 'claim_group': V.GROUP_OF_ITEM[c['item']],
            'gated': c['gated'], 'prefix': c['orig_eval_key'][:8],
            'original': {**o['original'], 'label': c['orig_label']},
            'm_flex_price_multiplier': c['flex_price_multiplier'] if c['flex_price_multiplier'] is not None else 1.0,
            'investment_year': o['investment_year'], 'canonical': o['canonical'], 'candidate_key': o['candidate_key'],
            'ess_params_file_sha256_at_original_run': (o['original']['ess_params_file_at_run'] or {}).get('sha256'),
            'ess_params_file_sha256_now': o_section['inputs_now']['ess_params_file']['sha256'],
            'N_old': c['N_old'], 'k0_original_first_residual_pass': c['k0'],
            'original_lapses_after_k0': c['original_lapses_after_k0'], 'Q_N_old': o['Q_N_old'],
            'cap_rule': V.cap_rule(cell), 'spec_cap': V.spec_cap(cell), 'cap_ceiling': c['cap_ceiling'],
            'e_over_p': o['e_over_p'], 'lattice_e_over_p_legal': o['parts']['lattice_e_over_p_in_2_4_every_storage_node'],
            'substituted_plan': None, 'I_cited': o['I'], 'I_source': o['I_source'],
            't_sum_terminal_original': o['t_sum_terminal_original'],
            'dead_zone_candidate': o['dead_zone_candidate'], 'dead_zone_borderline': o['dead_zone_borderline'],
            'identity_vs_original': o['identity_vs_original'],
            'declaration': V.declaration_for(cell), 'campaign_id': CAMPAIGN_IDS[cell],
            'campaign_root': campaign_root_rel(cell), 'configuration': configuration(cell), 'keys': expected_keys(cell),
            'campaign_spec': {k: s[k] for k in ('path', 'sha256', 'eval_key', 'eval_dir', 'harness_sha256', 'git_head')},
            'launch_command': launch_command(cell, s['sha256']),
            'preconditions_only_command': launch_command(cell, s['sha256'], preconditions_only=True),
            'expected_wall_time': wall['per_cell'][cell]}
    return {
        'schema': 'p515_s53_resettle_spec_v3', 'series': SPEC_SERIES, 'version': SPEC_VERSION, 'stage_text': STAGE_TEXT,
        'predecessor': {'version': 2, 'path': PREDECESSOR_REL, 'sha256': pred_sha,
                        'status': 'the W118 ten-cell campaign spec (run W123-W129); unchanged, cited'},
        'authority': [f'{BRIEF} Addendum 58 (Rulings 1-3; order)', f'{BRIEF} Addendum 57 (Decisions 2-3)',
                      'TASKS.md Addendum 58: Advisor design review of the remaining cells + Planner rulings (i)-(iv), '
                      '82fa1b61', 'W131 pre-freeze diagnostics 7a5c5717', 'Planner task W132'],
        'frozen_utc': _utc(), 'git_head': H._git(['rev-parse', 'HEAD']),
        'objective_convention': DEFINITIONS['objective_convention'],
        'pins': {'code_sha256': code, 'production_sha256': production, 'solver': solver,
                 'zero_solve_checks_output': {'path': cf['path'], 'sha256': cf['sha256'], 'manifest': cf['manifest']},
                 'w131': {'path': K.W131['path'], 'sha256': K.W131['sha256']},
                 'w117': {'path': K.W117['path'], 'sha256': K.W117['sha256']},
                 'references_inputs_sha256': ref_inputs,
                 'baseline_tables': {'path': BASELINE_TABLES, 'sha256': _sha(BASELINE_TABLES)},
                 'w2_table': {'path': W2_TABLE, 'sha256': _sha(W2_TABLE)},
                 'campaign_specs': {c: {'path': specs[c]['path'], 'sha256': specs[c]['sha256']} for c in V.CELL_ORDER}},
        'production_since_originals': prov,
        'cells': cells, 'cell_order': list(V.CELL_ORDER),
        'not_in_this_spec': {'group_3_ageing_E': ['c6b53015', '65a5da77', '98e28570', 'ed4a1acc', '06f092d1', '7eb1ce62'],
                             'reason_E': 'held for the author\'s soh_min ruling (TASKS.md Addendum 58)',
                             'covered_by_settled_references': {'7aa017f0': 'x0 settled d110bd1a',
                                                               'bd504ecf': 'unit settled 3f084f2f'}},
        'launch_order': {'order': list(V.CELL_ORDER), 'enforced': 'every earlier cell has results before --run',
                         'one_cell_per_call': True},
        'launch_commands': {c: cells[c]['launch_command'] for c in V.CELL_ORDER},
        'summarize_commands_at_the_completion_points': {p['after_cell']: summarize_command(p['after_cell'])
                                                        for p in points['report_points']},
        'claim_completion_points': points,
        'inputs_in_force_now': o_section['inputs_now'], 'references': refs,
        'configuration': {'current_production': ('case file (AA keep_memory declared) + ESS ageing baseline C2 declared '
                                                 '+ tight tail {enabled True, compl_inf_tol 1e-6} declared'),
                          'tail_rule': 'production: the tail acts from AA-off + 1 (next-state = this cycle converged)',
                          'persistence': 'persist_certified_models True, hull_polish False, no reference (as W101/W118)',
                          'concurrency': CONCURRENCY, 'option_b_release_solution_bookkeeping': 'absent',
                          'checked_field_by_field': 'validate_campaign_spec at --freeze-cells, --freeze-spec and --run'},
        'stop_rule': {
            'module': 'settling_criterion_v3', 'class': 'settling_criterion_v3.SettlingRuleV3', 'version': SC3.VERSION,
            'versions_1_and_2_unchanged': 'settling_criterion.py and settling_criterion_v2.py byte-identical (checks V0)',
            'constants': SC3.constants(V.P_MAX), 'readings': SC3.READINGS, 'algorithm': SC3.__doc__,
            'failing_reasons': list(SC3.FAILING_REASONS),
            'p_max': {'value': V.P_MAX, 'L': V.L_MONO, 'source': 'W102 x0 P_hat 29, W103 unit P_hat 30 (as fc791891)'},
            'reading_alpha': 'the decision: boyd_k_v3 = boyd_k AND all_optimal_k; a non-Optimal cycle is a lapse',
            'reading_gamma': 'report-only: no reset; a branch verdict vetoed while a non-Optimal cycle is in the window',
            'retry_tier': None,
            'N_and_holds_and_dynamic_cap': 'keyed on the first residual pass under the version-2 definition; both k0s '
                                           'recorded',
            'caps': {'gated': 'N_old + 100 (fixed; the per-cell ceiling in the cell table)',
                     'ungated': 'min(k0_run + 109, 300) (dynamic)',
                     'above_300': {'l_2ab0ce2d': 437, 'l_b2251bc5': 320}},
            'all_optimal_k_capture': ('the ninth pass-through wrapper on shared_resources_planning.'
                                      '_admm_local_solves_succeeded(planning_problem, results): each block\'s final '
                                      'accepted attempt classified by p515_s44_campaign_harness.ipopt_exit_class('
                                      'str(result.solver.message)); 51 entries (12 TSO, 36 DSO, 3 ESSO) per cycle line '
                                      '(ipopt_exit_by_block) and all_optimal_k; asserted before the first solve '
                                      '(exit-capture source facts) and on real production (checks H12); cross-checked '
                                      'post-run against the solve records and the ESSO logs (G24)'),
            'boyd_k': 'boyd_metrics[\'all_boyd_pass\'] AND local_solves_ok, from THIS cycle (AA wrapper)',
            'Q_k': 'gross_operational_cost (recourse wrapper); None when any local solve failed',
            't_sum_k': 'as fc791891 (in-cycle; validated post-run against the pf stride and the terminal identity, G22)',
            'mechanism': ('decision in the recourse wrapper; certification (or the dynamic cap of an ungated cell below '
                          'the spec cap) sets the certificate length to 0 -> production\'s own exit test ends the loop '
                          'at the end of THIS cycle; the decision file is written once'),
            'early_stop': 'ABSENT (validator refuses the key; checklist)'},
        'replay_gate': {
            'applies_to': list(V.GATED_CELLS), 'skipped_for': list(V.UNGATED_CELLS),
            'in_cycle': ('every cycle k <= k0 (the original first residual pass), at the end of the cycle: '
                         + ', '.join(R.REPLAY_GATED_FIELDS) + ' as JSON text against the ORIGINAL record row k; the first '
                         'difference writes the cycle line and ABORTS the cell; at k0 the run\'s first residual pass '
                         'must be k0'),
            'post_run': 'G19: every field of per_cycle_record rows 1..k0 bitwise',
            'overlap_report_only': 'k0+1..N_old: Q_new - Q_old and relative (G23)',
            'j_5f3cccb4': 'original first residual pass 159, Boyd lapse at 160, passes 161..170: gate 1..159, holds from 160'},
        'holds_after_first_residual_pass': {
            'AA': 'off (production\'s own off branch), every cycle > k0_run, even across a lapse',
            'tail': 'on', 'rho': 'frozen', 'same_as': 'W101 / W105 / W118'},
        'captures': {'w105': 'Q by block and component; ESS schedule movement; every raw Boyd field',
                     'w101_all_block_line': 'recourse_blocks_all.jsonl', 't_sum_and_Q_cc': 'resettle_cycle_record.jsonl',
                     'ipopt_exit_by_block': 'resettle_cycle_record.jsonl (51 entries) + all_optimal_k',
                     'lambda_t': 'interface_duals_per_cycle.jsonl (harness default)',
                     'asserted_before_any_solve': 'assert_resettle_preconditions (child) and parent_capture_checklist'},
        'record_status_label': ('p515_s44_campaign_harness._apply_settling_resettle_status: the record status follows '
                                'the settling decision (W123 / W125 defect fixed at the writer; G25)'),
        'definitions': DEFINITIONS,
        'scorer': {'claims_dataset': dataset, 'formulas': DEFINITIONS['claims'], 'D_fit': DEFINITIONS['D_fit'],
                   'uncertified_form': DEFINITIONS['uncertified_form'],
                   'functions': ['score_claim', 'resolve', 'd_fit', 'score_predictions', 'view_from_report',
                                 'reference_views', 'cell_report']},
        'predictions_recorded_before_any_run': PREDICTIONS,
        'gates': {
            'G1-G9, G11, G14, G16': 'as W101 / W118 (W86 evaluation_checks; solve profile; G6 v37; persistence; lambda)',
            'G13': 'holds inert through the first residual pass, held after', 'G15': 'stopping consistent',
            'G17': 'the pure rule v3 replays the in-cycle records and the decision', 'G18': 'line fields (+ exits)',
            'G19': 'gated: replay 1..k0 bitwise every field', 'G20': 'production counters',
            'G21': 'creep captures complete', 'G22': 't_sum in-cycle == stride formula; terminal identity',
            'G23': 'overlap recorded (gated) / absent (ungated)',
            'G24': 'the 51 exits of every cycle == the network solve records\' final attempts and the ESSO logs',
            'G25': 'the record status follows the settling decision', 'scope': GATE_SCOPE,
            'non_stopping': list(NON_STOPPING_GATES), 'post_run_evaluator_and_scorer_self_tests': post_tests},
        'zero_solve_checks': {'script': 'p515_s53_w132_resettle_v3_checks.py', 'committed_output': cf,
                              'inline_rerun_at_freeze': {
                                  'all_hold': checks_inline['all_hold'],
                                  'per_section': {k: v['holds'] for k, v in checks_inline['sections'].items()}},
                              'rerun_before_launch': ('--run re-runs every section and refuses unless all hold (the W100 '
                                                      'typing test is in the committed output only)')},
        'verbatim_text': {'quotes': {f'{f}:{k}': v for (f, k), v in VERBATIM.items()}, 'check': verb},
        'labelling_and_identity': {'label': V.LABEL,
                                   'evaluation_key': ('sha256({base_evaluation_key, settling_resettle (v3 '
                                                      'declaration)}); every other key byte-identical to the pre-W132 '
                                                      'harness (checks K)')},
        'harness_change': ('p515_s44_campaign_harness.py: resettle_hooks_module dispatch on the v3 declaration schema; '
                           '_apply_settling_resettle_status at the record writer; every pre-W132 key byte-identical '
                           '(checks K)'),
        'memory_preflight': {'rule': 'W86 rule at concurrency 1', 'measured_at_freeze_non_gating': mem},
        'solve_profile_declared': {'parent': 'never solves: every launcher guard permitted=(), verify(0)',
                                   'child': 'RECONCILED PER EVENT (G5): 51 x (cycles_run + 1) + every retry attempted'},
        'expected_wall_time': wall,
        'walls': {'this_spec_expected_h': wall['total_expected_h'], 'this_spec_worst_case_h': wall['total_worst_case_h'],
                  'advisor_transcribed': PREDICTIONS['walls']['statement'],
                  'single_run_over_4h': {c: wall['per_cell'][c]['worst_case_h'] for c in V.CELL_ORDER
                                         if wall['per_cell'][c]['worst_case_h'] > 4.0}},
        'dry_run_command': launch_command(V.CELL_ORDER[0], specs[V.CELL_ORDER[0]]['sha256'], preconditions_only=True),
        'zero_solve_smoke': ('no zero-solve path exercises _child_real end to end (it builds and solves); the wiring is '
                             'exercised by checks H (the real wrappers; the real install; the exit wrapper on real '
                             'production); THE FIRST REAL CYCLE OF THE FIRST LAUNCH IS THE SMOKE'),
    }


def freeze_spec(started):
    tag = 'W132-SPEC'
    failures = _common_checks()
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
        failures.append(f'inline re-run of the zero-solve checks fails: {_failing_check_items(checks_inline)}')
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
    content = stage_spec_content(checks_inline, cf, post_tests, verb, solver, prov, mem, wall, specs, o_section,
                                 dataset, points, refs, ref_inputs)
    text = GRIO.dumps(content, indent=1, sort_keys=True, default=H._json_default) + '\n'
    sha = hashlib.sha256(text.encode()).hexdigest()
    rel = os.path.join(ROOT_REL, f'{SPEC_PREFIX}{sha[:8]}.json')
    _write_once_text(rel, text)
    if _sha(rel) != sha:
        raise RuntimeError('the stage spec\'s written bytes do not hash to its name')
    _log(f'[{tag}] {STAGE_TEXT}')
    _log(f'[{tag}] wrote {rel} sha256={sha} (predecessor v2 {_sha(PREDECESSOR_REL)[:8]})')
    _log(f"[{tag}] checks per section: { {k: v['holds'] for k, v in checks_inline['sections'].items()} }")
    _log(f"[{tag}] post-run evaluator / scorer self-tests: { {k: v.get('ok') for k, v in post_tests.items()} }")
    _log(f'[{tag}] claims scored: {dataset["n_claims"]} (skipped, not in the settled set: '
         f'{len(dataset["skipped_not_in_the_settled_set"])})')
    for p in points['report_points']:
        _log(f"[{tag}] completion point after #{p['after_cell_index_1_based']} {p['after_cell']}: {p['items_complete']}")
    for cell in V.CELL_ORDER:
        w = wall['per_cell'][cell]
        _log(f"[{tag}] {cell}: cap {V.spec_cap(cell)} expected {w['expected_h']:.2f} h worst {w['worst_case_h']:.2f} h; "
             f"LAUNCH: {content['cells'][cell]['launch_command']}")
    _log(f"[{tag}] total expected {wall['total_expected_h']:.2f} h, worst {wall['total_worst_case_h']:.2f} h; memory "
         f"at freeze (non-gating): available {mem.get('available_gib')} GiB")
    _finish(0, '-- next: commit, then the dry run on the first cell')


# ======================================================================================================================
#  --run
# ======================================================================================================================
def run(started, cell, spec_sha256, preconditions_only=False):
    tag = f'W132-RUN-{cell}' + ('-PRECONDITIONS-ONLY' if preconditions_only else '')
    failures = _common_checks()
    ss_rel, ss_sha, ss = load_stage_spec()
    if not _committed_clean(ss_rel):
        failures.append('the stage spec is not committed / clean')
    if ss['pins']['code_sha256'] != {rel: _sha(rel) for rel in ss['pins']['code_sha256']}:
        failures.append('code changed since the stage spec froze')
    pin = ss['pins']['campaign_specs'].get(cell) or {}
    if pin.get('sha256') != spec_sha256:
        failures.append(f'the stage spec pins {pin.get("sha256")} for {cell}, not {spec_sha256}')
    idx = V.CELL_ORDER.index(cell)
    for prev in V.CELL_ORDER[:idx]:
        if not os.path.isfile(os.path.join(campaign_root(prev), RESULTS_FILE)):
            failures.append(f'priority order: {prev} (#{V.CELL_ORDER.index(prev) + 1}) has no results yet')
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
                              ('case file', spec['configuration']['case_file_sha256'], H.sha256_file(H.CASE_FILE))):
        if pinned != now:
            failures.append(f'{what} sha256 differs from the frozen spec')
    checks_inline = _run_checks_inline()
    if not checks_inline['all_hold']:
        failures.append(f'the zero-solve checks do not all hold now -- failing: {_failing_check_items(checks_inline)}')
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
             f"(pinned by the stage spec {ss_rel} sha256={ss_sha}); launch index {idx + 1} of {len(V.CELL_ORDER)}; "
             f"checks per section {({k: v['holds'] for k, v in checks_inline['sections'].items()})}; pre-launch parts "
             f"{pre['parts']}; parent capture checklist {len(cap_checks)} items all True; eval_key {entry['eval_key']}; "
             f"cap {spec['cap']}; solver {solver['resolved'].get('NLP_SOLVER_PATH')} sha256={solver['sha256']}")
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
    g = guards_verify()
    results = {'stage': STAGE_TEXT, 'cell': cell, 'launch_index_1_based': idx + 1, 'utc': _utc(),
               'git_head_at_run': H._git(['rev-parse', 'HEAD']),
               'campaign_spec_path': os.path.relpath(spec_path, REPO), 'campaign_spec_sha256': spec_sha256,
               'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': ss['objective_convention'],
               'eval_key': entry['eval_key'], 'candidate_key': entry['key'],
               'original_eval_key': V.CELLS[cell]['orig_eval_key'],
               'replay_divergence_abort': summ.get('replay_first_divergence'),
               'gates': gates, 'gates_pass': all(gates.values()), 'stopping_gate_failures': stopping,
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
        _log(f"[{tag}] settling v3: status {report.get('status')} k* {report.get('k_star')} branch {report.get('branch')} "
             f"band_width {report.get('band_width')} s {report.get('s_signed')} t_sum(end) {report.get('t_sum_end')} "
             f"k0_run {report.get('k0_run')} first_k0_alpha {report.get('first_k0_alpha')} non-Optimal cycles "
             f"{report.get('non_optimal_cycles')} cycles {report.get('cycles_run')} record status "
             f"{report.get('record_status')}")
    if point is not None:
        _log(f"[{tag}] CLAIM-COMPLETION POINT after #{point['after_cell_index_1_based']} {cell}: items "
             f"{point['items_complete']} complete -- report (Addendum 58). Scorer: {summarize_command(cell)}")
    code = 0 if (not stopping and _guards_ok(g) and isinstance(report, dict) and 'error' not in report) else 1
    _finish(code, f'wall={time.time() - started:.0f}s')


# ======================================================================================================================
#  --summarize
# ======================================================================================================================
def summarize(started, after_cell):
    tag = f'W132-SUMMARY-after-{after_cell}'
    ss_rel, ss_sha, ss = load_stage_spec()
    idx = V.CELL_ORDER.index(after_cell)
    reports, missing = {}, []
    inputs = {}
    for cell in V.CELL_ORDER[:idx + 1]:
        path = os.path.join(campaign_root(cell), RESULTS_FILE)
        rel = os.path.relpath(path, REPO)
        if not os.path.isfile(path) or not _committed_clean(rel):
            missing.append(cell)
            continue
        inputs[rel] = _sha(rel)
        reports[cell] = (json.load(open(path)).get('cell_report') or {})
    out_rel = os.path.join(ROOT_REL, f'w132_summary_after_{idx + 1:02d}_{after_cell}.json')
    if missing or os.path.exists(_abs(out_rel)):
        _log(f'[{tag} PRECONDITION FAILED] cells without committed results {missing} or the summary exists')
        _finish(1)
    refs, ref_inputs = reference_views()
    if ref_inputs != ss['pins']['references_inputs_sha256']:
        _log(f'[{tag} PRECONDITION FAILED] the references changed since the stage spec froze')
        _finish(1)
    dataset = ss['scorer']['claims_dataset']
    views = {c: r.get('view') or view_from_report(r) for c, r in reports.items()}
    scored = []
    for cl in dataset['claims']:
        def view(side):
            s = cl[side]
            return refs[s['ref']] if s['source'] == 'reference' else views.get(s['cell'], {})
        scored.append(score_claim(cl, view('ref'), view('other')))
    # the D fit, when all its points are in
    d_cells = {V.CELLS[c]['orig_label']: c for c in V.CELL_ORDER if V.CELLS[c]['item'] == 'D'}
    d_out = None
    if all(c in views for c in d_cells.values()):
        unit = _load(W2_TABLE)['candidates']['n7_4h_e1']
        p_cost, e_cost = unit['I_new_power_eur'] / 0.25, unit['I_new_energy_eur'] / 1.0
        x0, un = refs['7aa017f0'], refs['bd504ecf']
        pts = {lb: (un if lb == 'n7_4h_e1' else views[d_cells[lb]]) for lb in D_FIT_NODE7_LABELS}
        vals = {'Q': {lb: x0['Q'] - pts[lb]['Q'] for lb in D_FIT_NODE7_LABELS},
                'Q_cc': {lb: x0['Q_cc'] - pts[lb]['Q_cc'] for lb in D_FIT_NODE7_LABELS}}
        d_out = d_fit(vals, x0['band'], {lb: pts[lb]['band'] for lb in D_FIT_NODE7_LABELS}, p_cost, e_cost,
                      {lb: _ep_of(lb) for lb in D_FIT_NODE7_LABELS})
        d_out['statuses'] = {lb: pts[lb]['status'] for lb in D_FIT_NODE7_LABELS}
        d_out['all_points_certified'] = all(v == 'certified' for v in d_out['statuses'].values())
        d_out['first_unit_n7_eur_per_mwh'] = (x0['Q'] - un['Q'] - 0.25 * p_cost) / 1.0
        b = views.get('b_2a0ba8b2')
        d_out['first_unit_best_node_n5_4h_e1_eur_per_mwh'] = ((x0['Q'] - b['Q'] - 0.25 * p_cost) / 1.0) if b else None
        d_out['committed_baseline_tables'] = {k: _load(BASELINE_TABLES)[k] for k in ('node7_fit', 'breakeven')}
    doc = {'stage': STAGE_TEXT, 'utc': _utc(), 'after_cell': after_cell, 'after_cell_index_1_based': idx + 1,
           'stage_spec': {'path': ss_rel, 'sha256': ss_sha}, 'objective_convention': DEFINITIONS['objective_convention'],
           'reports': reports, 'references': refs, 'claims_scored': scored,
           'claims_complete': [s['claim_id'] for s in scored if not str(s.get('verdict', '')).startswith('not scored')],
           'D_fit': d_out, 'predictions_scored': score_predictions(reports), 'definitions': DEFINITIONS,
           'predictions': PREDICTIONS, 'inputs_sha256': {**inputs, **ref_inputs}}
    H._write_once_json(_abs(out_rel), doc)
    man = os.path.join(ROOT_REL, f'w132_summary_after_{idx + 1:02d}_{after_cell}_manifest_sha256.json')
    H._write_once_json(_abs(man), {out_rel: _sha(out_rel), **doc['inputs_sha256']})
    _log(f'[{tag}] wrote {out_rel}; claims complete {len(doc["claims_complete"])}; D fit '
         f'{"in" if d_out else "not yet"}')
    _finish(0)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-cells', action='store_true')
    mode.add_argument('--freeze-spec', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--summarize', action='store_true')
    parser.add_argument('--cell', choices=V.CELL_ORDER, default=None)
    parser.add_argument('--spec-sha256', default=None)
    parser.add_argument('--after-cell', choices=V.CELL_ORDER, default=None)
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
