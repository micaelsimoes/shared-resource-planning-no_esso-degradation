"""P5.15 S56 step 4 -- POST-RUN zero-solve diagnostics D1-D5 on the committed main-run evidence (spec v2 7ca210fb,
commit 5f1db156; P5_15_S56_STEP4_RESULT_REPORT.md). Planner task of 2026-11-23: discriminate the mechanisms behind
step 4's terminal offsets using only existing captures. No solve, no model build, no production change, no ADMM.

  D1  offset series per coupled run (A V-1, A V-2, B V-1, B V-2) and per block, every cycle >= 21: e_b, price_b, the
      expectation / deviation dual parts (P, Q, V separately), o1 = e - price, o2 = e - price + dual,
      o3 = e - price - dual (= e - PR); geometric fits on the tail-depth window, last 3 cycles held out.
  D2  dual consistency mu~_T + mu~_D (EUR per MW / MVAr / kV), terminal cycle and the last 10 cycles.
  D3  ADMM-only boxes at the terminal cycle: the per-scenario TSO copies against pc_init +- rating (P, Q); beside:
      DSO reference-generator boxes and the interface voltage rows.
  D4  P7 recomputed POST HOC with the corrected sign (Q - price + dual, and Q - price), beside the frozen outcome.
  D5  M1 rounding check: T and D exactly (fractions.Fraction) from the stored doubles; |pr_price + T| <= c u sum|.|.

Every definition, constant and prediction is in the frozen spec (`--freeze` writes it; the run reads it). This
process never solves: SolveProfileGuard(permitted=()) is installed before any project import and verified at 0.
Imports (never copies) the step 4 evaluator's loaders (`Run`, `eval_dir`), `priced_residual`, `settling_bar` and
`v_base`, and the launcher's spec / root helpers and tail-depth rule.

Usage (canonical interpreter):
  p515_s56_step4_postdiag.py --freeze                    write the frozen spec into OUT_DIR (refuses if one exists)
  p515_s56_step4_postdiag.py --check                     capture-path assertion only (no diagnostic computed)
  p515_s56_step4_postdiag.py --spec <frozen spec path>   the run (results JSON into OUT_DIR, never overwritten)

Objective convention (every table): EUR of total discounted cost. Coupled arms: Q_b = settlement excluded in full
(production primary; pin excluded, salvage separate). e_b = Q_b - w_b C_central (step 3 gross EUR/day x w_b).
price_b = -sum_agents(block) w x settlement_total_unweighted = +sum w omega pi (P_T - P_D). dual = the TSO-side dual
terms of the evaluator's priced_residual (expectation P/Q/V + omega-weighted deviation P/Q/V).
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPT = os.path.basename(__file__)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 step 4 post-run diagnostics D1-D5 (zero-solve)').install()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import traceback  # noqa: E402
from fractions import Fraction  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s56_step4_main as L  # noqa: E402  (launcher helpers; installs its own permitted=() guard)
import p515_s56_step4_main_evaluate as E  # noqa: E402  (evaluator; installs its own permitted=() guard)

# ======================================================================================================================
#  constants (all frozen into the spec)
# ======================================================================================================================
RES = os.path.join('data', 'SRP1', 'Results', 'P515S56')
OUT_DIR = os.path.join(RES, 'step4_postdiag_v1')
SPEC_PREFIX = 'frozen_step4_postdiag_spec_v1_'
STEP4_DIR = os.path.join(RES, 'step4_main_v2')
STEP4_SPEC = os.path.join(STEP4_DIR, 'frozen_step4_main_spec_v2_7ca210fb.json')
STEP4_EVALUATION = os.path.join(STEP4_DIR, 'step4_main_evaluation_20261123T163553.json')
STEP4_MANIFEST = os.path.join(STEP4_DIR, 'step4_main_v2_manifest_sha256.json')
STEP4_REPORT = 'P5_15_S56_STEP4_RESULT_REPORT.md'
DIAG_RESULTS = os.path.join(RES, 'step4_diag_v1', 'run_v1', 'step4_diag_results.json')
RUNS_GIT_HEAD = 'cb0628340686753862bef1ec90c941fe0535a8fb'
C_ROUND = 64
U_ROUND = Fraction(1, 2 ** 53)
MIN_CYCLE = 21
HOLDOUT = 3
D2_LAST = 10
ACTIVE_TOL_REL = 1e-6
NEAR_ACTIVE_REL = 1e-3
VALIDATION_REL = 1e-12
P7_FACTOR = 5.0
Q3_PLATEAU_TOTAL_REL = 0.03
Q3_DECAY_PER_CYCLE = 0.03
INSTANCES = ('A', 'B')
COUPLED = ('V1', 'V2')
ALL_ARMS = ('V1', 'V2', 'pinned_aa_off', 'pinned_aa_on')
RUN_FILES = ('per_cycle_record.jsonl', H.S56_CYCLE_CAPTURE_FILE, 'interface_duals_per_cycle.jsonl',
             'evaluation_record.json', H.MULTISCENARIO_TERMINAL_FILE, 'interface_settlement_detail_s31c.json',
             'interface_voltage_terminal.json', H.PER_CYCLE_RESPONSE_FILE, 'exit_code.txt')
HAND_VALUES = {
    'A|V1|2025|Winter|o1': {'cycles': [38, 39, 40, 41, 42, 43, 44, 45, 46],
                            'values': [349, 323, 299, 279, 261, 242, 225, 210, 197]},
    'A|V2|2025|Winter|o1': {'cycles': [32, 40], 'values': [308, 207]},
    'B|V1|2035|Summer|o3': {'cycles': [101, 107], 'values': [960, 827]},
    'A|V1|2025|Winter|o2': {'cycles': [46], 'values': [156]},
}

# the production lines each sign convention / formula is read from (verified verbatim before computing)
MCH, SRP = 'model_construction_helpers.py', 'shared_resources_planning.py'
CODE_LINES = [
    (MCH, 1771, 'settlement += probability * c_p[p] * network.baseMVA * model.pc_adn[dn, s_m, s_o, p]',
     'TSO block: accumulates prob x pi x baseMVA x P_T'),
    (MCH, 1772, 'return -settlement', 'TSO interface_settlement = -sum omega pi baseMVA P_T'),
    (MCH, 1779, 'settlement += probability * c_p[p] * network.baseMVA * model.pg_adn[s_m, s_o, p]',
     'DSO block: accumulates prob x pi x baseMVA x P_D'),
    (MCH, 1780, 'return settlement', 'DSO interface_settlement = +sum omega pi baseMVA P_D'),
    (MCH, 1847, 'obj += model.interface_settlement_weight * model.interface_settlement',
     'the settlement is in each local objective (weight 1 in the ADMM path)'),
    (SRP, 1390, "settlement_part = 'total' if _per_scenario_interface_coupling_wired(models) else 'contracted'",
     'per-block Q (capture q_blocks): the whole settlement subtracted while the coupling is wired'),
    (SRP, 4876, 'gross_in_full = gross_including_settlement - settlement_total - voltage_pin_total',
     'Q in full = sum of local costs at each side\'s own copy, settlement and pin excluded'),
    (SRP, 4802, 'pr_price += weight * omega * price * d_p', 'production pr_price = +sum w omega pi (P_T - P_D)'),
    (SRP, 5847, 'effective_scale = objective_scale / block_weight', 'TSO ADMM objective scale'),
    (SRP, 5852, 'obj = copy(model[year][day].objective.expr) / effective_scale', 'TSO local cost / effective scale'),
    (SRP, 5866, 'constraint_p_req = (model[year][day].expected_interface_pf_p[dn, p] - model[year][day].p_pf_req[dn, p]) '
                '/ interface_transf_rating', 'TSO expectation term argument (P_T - P_D)/R'),
    (SRP, 5868, 'obj += model[year][day].dual_pf_p_req[dn, p] * constraint_p_req', 'TSO term +mu_T (P_T - P_D)/R'),
    (SRP, 6120, 'effective_scale = objective_scale / block_weight', 'DSO ADMM objective scale'),
    (SRP, 6146, 'constraint_p_req = (dso_model[year][day].expected_interface_pf_p[p] - dso_model[year][day].p_pf_req[p]) '
                '/ interface_transf_rating', 'DSO expectation term argument (P_D - P_T)/R'),
    (SRP, 6148, 'obj += (dso_model[year][day].dual_pf_p_req[p]) * constraint_p_req', 'DSO term +mu_D (P_D - P_T)/R'),
    (SRP, 6824, "model[year][day].dual_pf_p_req[dn, p].set_value(dual_pf['current'][node_id][year][day]['p'][p] / s_base)",
     'TSO model Param = stored dual / s_base'),
    (SRP, 7153, "model[year][day].dual_pf_p_req[p].set_value(dual_pf['current'][node_id][year][day]['p'][p] / s_base)",
     'DSO model Param = stored dual / s_base'),
    (SRP, 9248, "error_p_pf_req_tso = interface_vars['pf']['tso']['current'][node_id][year][day]['p'][p] - "
                "interface_vars['pf']['dso']['current'][node_id][year][day]['p'][p]", 'TSO residual = T - D'),
    (SRP, 9251, "dual_vars['pf']['tso']['current'][node_id][year][day]['p'][p] += rho_pf_tso * error_p_pf_req_tso / "
                "interface_rating * tso_s_base", 'mu_T += rho_T (T - D) s_base_T / R'),
    (SRP, 9258, "error_p_pf_req_dso = interface_vars['pf']['dso']['current'][node_id][year][day]['p'][p] - "
                "interface_vars['pf']['tso']['current'][node_id][year][day]['p'][p]", 'DSO residual = D - T'),
    (SRP, 9261, "dual_vars['pf']['dso']['current'][node_id][year][day]['p'][p] += rho_pf_dso * error_p_pf_req_dso / "
                "interface_rating * dso_s_base", 'mu_D += rho_D (D - T) s_base_D / R (its own rho)'),
    (SRP, 9250, "dual_vars['vmag']['tso']['current'][node_id][year][day][p] += rho_v_tso * error_v_req_tso",
     'mu_V,T += rho_v,T (V_T - V_D)'),
    (SRP, 9260, "dual_vars['vmag']['dso']['current'][node_id][year][day][p] += rho_v_dso * error_v_req_dso",
     'mu_V,D += rho_v,D (V_D - V_T)'),
    (SRP, 4505, 'extra += w * (dual * c + (rho / 2) * c ** 2)', 'TSO deviation term omega mu_s (d_T,s - d_D,s)/R'),
    (SRP, 4537, 'd_extra += w * (dual * c + (rho / 2) * c ** 2)', 'DSO deviation term omega mu_s (d_D,s - d_T,s)/R'),
    (SRP, 4607, 'dual[index].set_value(dual_src[kind][k][p] / base)', 'deviation model Param = stored / base'),
    (SRP, 4690, 'tso_dual[kind][k][p] += rho_v_tso * error_tso', 'deviation V: mu_T += rho_v,T (d_T - d_D)'),
    (SRP, 4691, 'dso_dual[kind][k][p] += rho_v_dso * error_dso', 'deviation V: mu_D += rho_v,D (d_D - d_T)'),
    (SRP, 4693, 'tso_dual[kind][k][p] += rho_pf_tso * error_tso / interface_rating * tso_s_base',
     'deviation P/Q: mu_T += rho_T (d_T - d_D) s_base / R'),
    (SRP, 4694, 'dso_dual[kind][k][p] += rho_pf_dso * error_dso / interface_rating * dso_s_base',
     'deviation P/Q: mu_D += rho_D (d_D - d_T) s_base / R'),
    (SRP, 4478, 'copy_var.setlb(-rating)', 'per-scenario TSO delta copy box: lower -rating (pu)'),
    (SRP, 4479, 'copy_var.setub(rating)', 'per-scenario TSO delta copy box: upper +rating (pu)'),
    (SRP, 4962, 'tso_model[year][day].interface_delta_p[dn, s_m, s_o, p].setlb(-interface_transf_rating)',
     'alias delta copy box (P)'),
    (SRP, 4964, 'tso_model[year][day].interface_delta_q[dn, s_m, s_o, p].setlb(-interface_transf_rating)',
     'alias delta copy box (Q)'),
    (SRP, 4933, 'fix_or_set(tso_model[year][day].pc[adn_load_idx, s_m, s_o, p], interface_pf_p)',
     'pc fixed at the DSO initial expectation (pc_init); P_T,s = pc_init + delta_s'),
    (SRP, 4922, 'tso_model[year][day].slack_v_sqr_down[adn_node_idx, s_m, s_o, p].fix(0.00)',
     'TSO interface-bus voltage slacks fixed at 0 in the ADMM model (the V row is hard there)'),
]

PREDICTIONS_VERBATIM = {
    'Q1': ('on A V-1 and V-2, o1 > 0 at every tail-depth cycle and decreasing. The fitted limit of o2 is in [0, 100] '
           'EUR and its held-out error is <= 50 EUR.'),
    'Q2': 'on A, terminal o2 in [100, 230] EUR (the Advisor\'s hand value is +156).',
    'Q3': ('on B 2035|Summer, terminal o1 in [700, 850] EUR. Two outcomes are recorded as competing, not predicted: o1 '
           'decays at <= 3 % per cycle (B open), or it plateaus.'),
    'Q4': 'on A, the D2 EUR value of the dual-sum mismatch at the terminal cycle is <= 50 EUR (consistent duals).',
    'Q5': ('no ADMM-only box is active at the terminal cycle on A or B (D-3 margins were >= 37 MW and >= 85 MVAr at '
           'the central solution).'),
    'Q6': 'D5 holds on every cycle of every coupled run, and the sign of (pr_price + T) is mixed.',
    'policy': 'A failed prediction is recorded, not a stop: this is a zero-solve diagnostic stage.',
}
PREDICTIONS_OPERATIONAL = {
    'Q1': ('runs A|V1 and A|V2, block 2025|Winter, tail-depth cycles T (evaluator rule). (a) o1(k) > 0 for every k in '
           'T and o1(k) < o1(k-1) for every k in T with k-1 in T; (b) E.settling_bar(o2 series, cycles=T, W=20, h=3) '
           'is valid (one-sign steps, >= 6 steps, r < 1, r_fit < 1) with Q_hat_inf in [0, 100] and H <= 50. '
           'Held iff (a) and (b) on both runs. A run with no valid settling_bar fit fails clause (b) (recorded as '
           'not evaluable); the 3-parameter fit is reported beside, non-deciding'),
    'Q2': ('decisive on A|V1 (the arm the hand value +156 and the frozen P7 come from): o2(K) in [100, 230], K = the '
           'last cycle. A|V2 reported beside with its own in/out flag, non-deciding'),
    'Q3': ('decisive on B|V1, block 2035|Summer (the hand window 101-107 exists only on V-1): o1(K) in [700, 850]. '
           'Competing outcomes, classified on o1 over B|V1 tail-depth cycles K1..K (recorded, never a verdict): '
           'total_rel = (o1(K) - o1(K1)) / |o1(K1)|; g = (o1(K) / o1(K1))^(1/(K - K1)) - 1 (None if a sign change or '
           'a zero); "plateau" if |total_rel| <= 0.03; else "decays at <= 3 %/cycle (B open)" if -0.03 <= g < 0; '
           'else "decays faster than 3 %/cycle" if g < -0.03; else "grows or changes sign". B|V2 beside'),
    'Q4': ('decisive on A|V1 at its terminal cycle: D2 EUR mismatch (absolute form) = sum over every coordinate of '
           '|(mu~_T + mu~_D) x r| (expectation P/Q/V; deviation P/Q/V weighted by omega_s) <= 50. Signed form and '
           'A|V2 beside, non-deciding'),
    'Q5': ('decisive scope = the D3 task definition: the per-scenario TSO copies P_T,s = pc_init + delta_s against '
           'pc_init +- rating, i.e. |delta_s| <= R (P in MW, Q in MVAr), at the terminal state of every run of A and B '
           '(8 runs: coupled arms all 9 scenario copies; pinned arms the alias copy that every scenario references). '
           'A bound is ACTIVE when margin = R - |delta| <= 1e-6 x max(1, R). Held iff no active bound. Beside, '
           'non-deciding ("Q5 extended"): DSO reference-generator P/Q boxes and the interface V rows (coupled arms)'),
    'Q6': ('on every coupled run (A|V1, A|V2, B|V1, B|V2) and every cycle 1..K: |pr_price + T| <= c u S_T with '
           'c = 64, u = 2^-53, S_T = sum_agents |w_a s_a| (exact rational arithmetic on the stored doubles; '
           'pr_price = per_cycle_record priced_residual_price_part); AND on each coupled run both signs of '
           '(pr_price + T) occur among its cycles (zeros excluded). Held iff both parts hold on all four runs'),
}

DEFINITIONS = {
    'objective_convention': ('EUR of total discounted cost. Coupled arms: Q_b(k) = capture q_blocks per_year_day[b] '
                             'settlement_excluded_in_full (production primary; pin excluded, salvage separate). '
                             'e_b(k) = Q_b(k) - w_b C_central (step 4 spec ii_central_and_allowance.blocks[b]'
                             '.w_b_C_central_eur; 2025|Summer is MERGED-ONLY, reported not calibrated)'),
    'price_b': ('price_b(k) = -sum_{agents a in block b} w_a x settlement_total_unweighted_a (capture per_agent) = '
                '+sum w omega pi (P_T - P_D) (sign from the CODE_LINES); computed by E.priced_residual (V-1) and by '
                'the per-channel decomposition below (both runs), cross-validated'),
    'dual_parts': ('TSO-side dual terms, EUR, as E.priced_residual: fac_pq = w x admm_objective_scale_tso / '
                   '(tso_base_mva x interface_rating_mva), fac_v = w x admm_objective_scale_tso / v_b^2 (v_b from '
                   'E.v_base). exp_P = sum_{dn,t} lambda_pf_p_tso x (Pbar_T - Pbar_D) x fac_pq; exp_Q likewise with '
                   'Q; exp_V = sum dual_expectation_v.tso x (Vbar_T - Vbar_D) x fac_v; dev_X = sum_{dn,s,t} omega_s '
                   'lambda_dev_x_tso,s x (X_T,s - X_D,s) x fac, X_side,s = Xbar_side + d_side,s (V-2: no V '
                   'deviation channel, dev_V absent). dual = exp_P + exp_Q + exp_V + dev_P + dev_Q + dev_V'),
    'o_series': 'o1 = e - price; o2 = e - price + dual; o3 = e - price - dual = e - PR (PR = price + dual)',
    'D1_fit': ('window = tail-depth cycles (E/L rule: every solve record of round k has compl_inf_tol_in_force == '
               '1e-6). PRIMARY: E.settling_bar(series, cycles=window, W=20, h=3): r_fit from OLS of ln|step| on the '
               'window minus its last 3 cycles; held-out H = max abs error of its predictions on the last 3; r from '
               'all steps; projected limit Q_hat_inf = X(K) + d_K r / (1 - r). Recorded beside: the limit implied by '
               'the held-out fit alone, X(k0) + d_k0 r_fit / (1 - r_fit) (k0 the last fitted cycle); and LSQ3: '
               'X(k) = L + A r^(k - k_first) by least squares on the window minus its last 3 (r grid 0.001..0.999 '
               'step 0.001 + golden refinement, linear L, A), held-out error on the last 3, boundary flag when r hits '
               'the grid edge'),
    'D2': ('per coordinate: mu~_T = w_T x admm_objective_scale_tso x lambda_T / (tso_base_mva x R_MVA) (P, Q; EUR per '
           'MW / MVAr), w_T x scale_T x lambda_T / v_b,T^2 (V; EUR per kV); mu~_D likewise with w_D (capture per_agent '
           'DSO weight), admm_objective_scale_dso, dso_base_mva, v_b,D (capture DSO v / interface_voltage_terminal '
           'dso_pu). Each mu~ is the EUR coefficient the side\'s objective places on ITS OWN copy (CODE_LINES 5852/'
           '5868, 6125/6148, 4505/4537 with the Param = stored / base). Residual r = x_T - x_D on the channel the dual '
           'update uses (expectation: Pbar_T - Pbar_D; deviation: d_T,s - d_D,s). EUR value per coordinate: '
           '(mu~_T + mu~_D) x r (deviation: x omega_s). Aggregates per run / cycle / block / channel / kind: max '
           '|mu~_T + mu~_D|, omega-RMS (sqrt(sum_s omega_s mean_{dn,t} x^2); expectation: RMS over dn, t), max '
           '|mu~_T|, EUR signed sum, EUR absolute sum, count of stored duals with lambda_T == -lambda_D exactly. '
           'Plus rho_T == rho_D (pf and v) per block at every captured cycle'),
    'D2_equal_and_opposite': ('"equal and opposite" := mu~_D = -mu~_T at every coordinate, i.e. the DSO\'s EUR '
                              'coefficient on its own copy x_D is minus the TSO\'s EUR coefficient on its own copy '
                              'x_T -- what one Lagrangian term lambda (x_T - x_D) would give both sides. From the '
                              'update lines (9251/9261, 9250/9260, 4690-4694): mu_T += rho_T r s_base_T / R and '
                              'mu_D += rho_D (-r) s_base_D / R from mu = 0, so mu_T + mu_D = 0 exactly when rho_T = '
                              'rho_D and s_base_T = s_base_D at every update (and the EUR conversions agree when w_T '
                              'scale_T = w_D scale_D)'),
    'D3': ('terminal state. TSO P/Q: interface_settlement_detail_s31c.json delta_p_mw / delta_q_mvar per (DN, block, '
           'period, scenario) (= interface_delta x s_base, the copy each scenario references) against [-R, +R] '
           '(R = interface_rating_mva), i.e. P_T,s against anchor_p_mw (= pc_init) +- R; margin = R - |delta|; '
           'active iff margin <= 1e-6 x max(1, R); near-active iff margin / R < 1e-3. Cross-check (coupled): anchor + '
           'delta_s against the capture terminal Pbar_T + d_T,s. Beside: (i) the same TSO box at every cycle >= 21 '
           'from the capture with the terminal anchor (pc is fixed); (ii) DSO reference generator P_D,s, Q_D,s '
           '(capture) against step 3 relaxation_limits ref_gen_p_mw / ref_gen_q_mvar (the D-3 model box for DN7 P '
           'was +-100.001: 0.001 MW wider); (iii) interface V rows: V-1 per scenario (capture, pu = kV / v_b), '
           'V-2 and pinned the expectation (interface_voltage_terminal), against v_min_pu / v_max_pu (TSO) and '
           'dn_ref_node_v_pu (DSO); active iff margin <= 1e-6 pu'),
    'D4': ('cycles k >= 21 of each coupled run, per block: frozen P7 = sum |e + PR| vs sum |e| (factor = sum|e| / '
           'sum|e + PR|, held iff >= 5); POST HOC corrected (a) sum |e - price + dual| = sum |o2|, (b) sum |e - '
           'price| = sum |o1|, same factor and threshold. Beside: the same on the tail-depth window. Labelled post '
           'hoc; never replaces the frozen outcome'),
    'D5': ('every cycle 1..K of every coupled run. T = sum_a w_a s_a (s_a = settlement_total_unweighted), D = sum_a '
           'w_a (s_a - c_a) (c_a = settlement_contracted_unweighted), each exactly as fractions.Fraction of the '
           'stored doubles; also math.fsum and the evaluator\'s float sum. Test (i): |pr_price + T| <= c u S_T, '
           'S_T = sum_a |w_a s_a| (exact). Beside (ii): |(gross_contracted_excluded - gross) - D| <= c u S_ii, '
           'S_ii = sum_a (|q_full_a| + |q_cont_a| + |w_a s_a| + |w_a c_a|). Beside: S_fine = sum_{b,dn,s,t} w '
           'omega |pi| (|P_T,s| + |P_D,s|) (prices from interface_settlement_detail, P from the capture) and the '
           'recomputed s_a against the stored s_a. Sign of (pr_price + T) per cycle; Spearman rank correlation '
           '(average ranks) of (pr_price + T) and |pr_price + T| with |T| and with the PF residual norm sqrt('
           'boyd_pf_r_expectation_part^2 + boyd_pf_r_deviation_part^2)'),
}


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def _utc():
    return L._utc()


def _sha(path):
    return L._sha(path)


def _log(msg):
    print(f'[S56 step4 postdiag] {msg}', flush=True)


def _fin(x):
    """JSON-safe float: inf / nan -> None."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, Fraction):
        return float(obj)
    return _fin(obj)


def write_json_new(path, obj):
    obj = _clean(obj)
    GRIO.check(obj)
    with open(path, 'x', encoding='utf-8') as handle:
        GRIO.dump(obj, handle, indent=1, default=GRIO.json_default_item)


def load_step4_spec():
    spec, sha, rel = L._spec(STEP4_SPEC)   # verifies the name tag and applies the spec's absolute roots
    return spec, sha, rel


def run_dirs(step4_spec):
    out = {}
    for inst in INSTANCES:
        for arm in ALL_ARMS:
            root = L.root_path('main', inst)
            out[f'{inst}|{arm}'] = E.eval_dir(root, L.label_of('main', arm))
    return out


def input_files(step4_spec):
    files = [STEP4_SPEC, STEP4_EVALUATION, STEP4_MANIFEST, STEP4_REPORT, DIAG_RESULTS]
    for key, ev in run_dirs(step4_spec).items():
        for f in RUN_FILES:
            files.append(os.path.relpath(os.path.join(ev, f), REPO))
        recs = os.path.join(ev, H.NETWORK_IPOPT_SOLVE_RECORDS_FILE)
        if not os.path.exists(recs):
            recs = os.path.join(ev, H.NETWORK_IPOPT_SOLVE_RECORDS_APPEND_FILE)
        files.append(os.path.relpath(recs, REPO))
    return files


def code_line_check():
    out, bad = [], []
    cache = {}
    for f, n, snippet, meaning in CODE_LINES:
        if f not in cache:
            with open(os.path.join(REPO, f), encoding='utf-8') as handle:
                cache[f] = handle.read().split('\n')
        line = cache[f][n - 1] if n - 1 < len(cache[f]) else ''
        ok = snippet in line
        out.append({'file': f, 'line': n, 'snippet': snippet, 'meaning': meaning, 'verbatim': ok})
        if not ok:
            bad.append(f'{f}:{n} does not contain {snippet!r} (reads {line.strip()!r})')
    return out, bad


# ======================================================================================================================
#  spec
# ======================================================================================================================
def build_spec():
    step4_spec, step4_sha, _ = load_step4_spec()
    lines, bad = code_line_check()
    if bad:
        raise RuntimeError(f'code-line check failed at freeze: {bad}')
    files = input_files(step4_spec)
    inv = {f: _sha(f) for f in files}
    runs = {}
    for key, ev in run_dirs(step4_spec).items():
        rec = E._load(os.path.join(ev, 'evaluation_record.json'))
        init = os.path.join(L.root_path('main', key.split('|')[0]), 'init_identity',
                            os.path.basename(ev) + '.json')
        ii = E._load(init) if os.path.exists(init) else {}
        runs[key] = {'eval_dir': os.path.relpath(ev, REPO), 'cycles_run': rec.get('cycles_run'),
                     'candidate_key': ii.get('candidate_key'), 'derived_instance': ii.get('derived_instance')}
    return {
        'schema': 'p515_s56_step4_postdiag_spec_v1', 'version': 1, 'predecessor': None,
        'stage': 'P5.15 S56 step 4 post-run zero-solve diagnostics D1-D5',
        'authority': 'Planner task 2026-11-23 (bounded zero-solve diagnostic task on the committed step 4 main-run '
                     'evidence; commit 5f1db156, P5_15_S56_STEP4_RESULT_REPORT.md)',
        'script': SCRIPT, 'script_sha256_at_freeze': _sha(SCRIPT), 'git_head_at_freeze': L._git('rev-parse', 'HEAD').strip(),
        'runs_git_head': RUNS_GIT_HEAD,
        'production_unchanged_since_runs': ('git log cb062834..HEAD -- shared_resources_planning.py '
                                            'model_construction_helpers.py p515_s44_campaign_harness.py '
                                            'interface_dual_capture.py is empty at freeze (checked by the Worker); '
                                            'the CODE_LINES are re-verified verbatim at run time'),
        'out_dir': OUT_DIR, 'objective_convention': DEFINITIONS['objective_convention'],
        'evidence': {'step4_spec': {'path': STEP4_SPEC, 'sha256': step4_sha}, 'step4_evaluation': STEP4_EVALUATION,
                     'step4_manifest': STEP4_MANIFEST, 'step4_report': STEP4_REPORT,
                     'step4_diag_results': DIAG_RESULTS, 'runs': runs},
        'input_sha256_inventory': inv,
        'constants': {'c': C_ROUND, 'u': '2^-53', 'u_float': float(U_ROUND), 'min_cycle': MIN_CYCLE,
                      'holdout_cycles': HOLDOUT, 'd2_last_cycles': D2_LAST, 'active_tol_rel': ACTIVE_TOL_REL,
                      'near_active_rel': NEAR_ACTIVE_REL, 'validation_rel': VALIDATION_REL, 'p7_factor': P7_FACTOR,
                      'q3_plateau_total_rel': Q3_PLATEAU_TOTAL_REL, 'q3_decay_per_cycle': Q3_DECAY_PER_CYCLE},
        'definitions': DEFINITIONS,
        'code_lines_sign_conventions': lines,
        'predictions_verbatim': PREDICTIONS_VERBATIM,
        'predictions_operational': PREDICTIONS_OPERATIONAL,
        'hand_values_to_compare': HAND_VALUES,
        'facts_recorded_at_freeze': {
            'Q5_background_margins': ('P5_15_S56_STEP4_DIAG_REPORT.md table (central solution, D-3): 2025|Winter P '
                                      '47.33 MW (TSO delta box), 37.02 MW (DSO ref-gen), Q 98.66 / 89.93 MVAr; '
                                      '2035|Summer P 27.34 MW (TSO delta box), 0.563 MW (DSO7 ref-gen), Q 97.95 / '
                                      '85.53 MVAr; V at its bound strictly (-4.6e-10 / -1.9e-10 pu) inside '
                                      'tolerance. The task\'s ">= 37 MW and >= 85 MVAr" holds for the TSO delta box '
                                      'and for the 2025|Winter DSO box, not for the 2035|Summer DSO7 P box'),
        },
        'inputs_used': ['the evaluator loaders E.Run / E.eval_dir / E.probabilities, E.priced_residual, '
                        'E.settling_bar, E.v_base (imported)', 'L.tail_depth_rounds via E.Run', 'L._spec / '
                        'apply_spec_roots / root_path / label_of'],
        'not_permitted': ['any solve (guard permitted=(), verified 0)', 'any model build', 'any production change',
                          'any edit to p515_s56_step4_main_evaluate.py, the launcher or existing artifacts',
                          're-running the step 4 evaluator onto its output path', 'any ADMM run', 'TASKS.md',
                          '.claude/worktrees/'],
        'failure_policy': ('a failed prediction is recorded, never a stop. A capture-path, input-hash or code-line '
                           'mismatch, a cross-validation failure or a guard count != 0 is a harness fault: the run '
                           'refuses / records it and the Worker reports'),
        'outputs': {'results': f'{OUT_DIR}/postdiag_results_<UTC stamp>.json (open mode x)',
                    'logs': f'{OUT_DIR}/postdiag_run_stdout.log, postdiag_run_stderr.log',
                    'manifest': f'{OUT_DIR}/step4_postdiag_v1_manifest_sha256.json'},
    }


def freeze():
    os.makedirs(OUT_DIR, exist_ok=True)
    existing = [f for f in os.listdir(OUT_DIR) if f.startswith(SPEC_PREFIX)]
    if existing:
        _log(f'REFUSED: a spec already exists: {existing}')
        return 2
    spec = build_spec()
    GRIO.check(spec)
    text = json.dumps(spec, indent=1, sort_keys=True)
    sha = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(OUT_DIR, f'{SPEC_PREFIX}{sha[:8]}.json')
    with open(path, 'x', encoding='utf-8') as handle:
        handle.write(text)
    _log(f'wrote {path} (sha256 {sha}); guard {GUARD.counts}')
    return 0 if not GUARD.verify(0) else 1


# ======================================================================================================================
#  capture-path assertion (rule eleven): every file and field D1-D5 need, before any computation
# ======================================================================================================================
def assert_capture_path(runs, step4_spec):
    missing = []

    def need(cond, what):
        if not cond:
            missing.append(what)

    blocks = step4_spec['ii_central_and_allowance']['blocks']
    for inst in INSTANCES:
        for bk in step4_spec['i_arms_instances_cap']['instances'][inst]['blocks']:
            need(isinstance((blocks.get(bk) or {}).get('w_b_C_central_eur'), float), f'step4 spec w_b_C_central {bk}')
    diag = E._load(DIAG_RESULTS)
    for bk in ('2025|Winter', '2035|Summer', '2025|Summer'):
        rl = ((diag.get('phase_A') or {}).get(bk) or {}).get('relaxation_limits') or {}
        for n in ('5', '7', '9'):
            for f in ('ref_gen_p_mw', 'ref_gen_q_mvar', 'dn_ref_node_v_pu', 'tn_bus_v_pu', 'interface_rating_mva'):
                need(f in (rl.get(n) or {}), f'diag relaxation_limits {bk} DN{n} {f}')
    ev_json = E._load(STEP4_EVALUATION)
    for inst in INSTANCES:
        for bk in L.CALIBRATED[inst]:
            need('sum_abs_Q_plus_PR_minus_wC' in ((ev_json['instances'][inst].get('P7') or {}).get(bk) or {}),
                 f'evaluation P7 {inst} {bk}')
    for key, run in runs.items():
        inst, arm = key.split('|')
        for f in RUN_FILES:
            need(os.path.exists(os.path.join(run.ev, f)), f'{key}: file {f}')
        K = run.rec.get('cycles_run')
        need(len(run.rows) == K and len(run.cap) == K and len(run.duals) == K,
             f'{key}: per_cycle_record / capture / duals lengths == cycles_run')
        need(all(x.get('captured') is True and x.get('error') is None for x in run.cap), f'{key}: every capture line captured')
        need(all(x.get('captured') is True for x in run.duals), f'{key}: every dual line captured')
        need([x['cycle'] for x in run.cap] == list(range(1, K + 1)), f'{key}: capture cycles 1..K')
        need([x['cycle'] for x in run.duals] == list(range(1, K + 1)), f'{key}: dual cycles 1..K')
        need([x['cycle'] for x in run.rows] == list(range(1, K + 1)), f'{key}: record cycles 1..K')
        meta = run.dual_header.get('metadata', {}).get('per_block') or []
        need(len(meta) == len(run.blocks), f'{key}: dual metadata per block')
        for m in meta:
            for f in ('node_id', 'year', 'day', 'interface_rating_mva', 'tso_base_mva', 'dso_base_mva',
                      'admm_objective_scale_tso', 'admm_objective_scale_dso'):
                need(f in m, f'{key}: dual metadata field {f}')
        need([tuple(b) for b in run.dual_header['blocks']] == run.blocks, f'{key}: dual and capture block order equal')
        isd = E._load(os.path.join(run.ev, 'interface_settlement_detail_s31c.json'))
        ird = isd.get('interface_reporting_detail') or {}
        ivt = E._load(os.path.join(run.ev, 'interface_voltage_terminal.json'))
        need(isd.get('cycles_run') == K, f'{key}: settlement detail written at the terminal cycle')
        need(ivt.get('summary', {}).get('cycle') == K, f'{key}: voltage terminal written at the terminal cycle')
        for (node, year, day) in run.blocks:
            per = (((ird.get(str(node)) or {}).get(str(year)) or {}).get(day) or {}).get('periods') or {}
            need(len(per) == 24, f'{key}: settlement detail periods {node}|{year}|{day}')
            for p, v in per.items():
                for f in ('delta_p_mw', 'delta_q_mvar', 'anchor_p_mw', 'anchor_q_mvar', 'price_per_mwh_by_scenario'):
                    need(f in v, f'{key}: settlement detail {node}|{year}|{day}|{p} {f}')
            ents = [e for e in ivt.get('entries') or [] if e['node_id'] == node and int(e['year']) == year
                    and e['day'] == day]
            need(len(ents) == 24 and all(all(f in e for f in ('tso_pu', 'dso_pu', 'v_min_pu', 'v_max_pu', 'period'))
                                         for e in ents), f'{key}: voltage terminal entries {node}|{year}|{day}')
        for r in run.rows:
            for f in ('priced_residual_price_part', 'priced_residual_dual_part', 'gross_operational_cost',
                      'boyd_pf_r_expectation_part', 'cycle'):
                if run.coupled or f in ('gross_operational_cost', 'cycle'):
                    need(f in r, f'{key}: per_cycle_record field {f} (cycle {r.get("cycle")})')
            if run.coupled:
                need('gross_operational_cost_contracted_excluded' in r and 'boyd_pf_r_deviation_part' in r,
                     f'{key}: per_cycle_record contracted-excluded / deviation residual (cycle {r.get("cycle")})')
        for ln in run.cap:
            need(all(k in ln for k in ('primal_expectation', 'dual_expectation_v', 'rho', 'q_blocks')),
                 f'{key}: capture fields (cycle {ln["cycle"]})')
            for side in ('tso', 'dso'):
                need(all(k in ln['primal_expectation'][side] for k in ('p', 'q', 'v')), f'{key}: primal_expectation {side}')
            for a, v in ln['q_blocks']['per_agent'].items():
                need(all(k in v for k in ('weight', 'settlement_total_unweighted', 'settlement_contracted_unweighted',
                                          'settlement_excluded_in_full', 'contracted_excluded')),
                     f'{key}: per_agent fields {a}')
            if run.coupled:
                kinds = run.cap_header['deviation_kinds']
                need(kinds == (['p', 'q', 'v'] if arm == 'V1' else ['p', 'q']), f'{key}: deviation kinds {kinds}')
                for side in ('tso', 'dso'):
                    need(sorted(ln['primal_deviation'][side]) == sorted(kinds), f'{key}: primal_deviation {side}')
                need('voltage_pin' in ln, f'{key}: voltage_pin (cycle {ln["cycle"]})')
        for du in run.duals:
            for f in ('lambda_pf_p_tso', 'lambda_pf_p_dso', 'lambda_pf_q_tso', 'lambda_pf_q_dso'):
                need(f in du, f'{key}: dual field {f}')
            if run.coupled:
                for kind in run.cap_header['deviation_kinds']:
                    for side in ('tso', 'dso'):
                        need(f'lambda_dev_{kind}_{side}' in du, f'{key}: dual field lambda_dev_{kind}_{side}')
        need(bool(run.tail_cycles), f'{key}: tail-depth cycles')
        need(all(bk in run.probs for bk in {f'{y}|{d}' for _, y, d in run.blocks}), f'{key}: probabilities per block')
    if missing:
        raise RuntimeError(f'capture path missing ({len(missing)} items; first 30): {missing[:30]}')
    return {'checked_runs': sorted(runs), 'ok': True}


# ======================================================================================================================
#  shared readings
# ======================================================================================================================
def scen_keys(run):
    return [f'{a}_{b}' for a, b in run.dual_header['deviation_channel']['scenarios']] if run.coupled else None


def block_indices(run, bk):
    return [bi for bi, (_n, y, d) in enumerate(run.blocks) if f'{y}|{d}' == bk]


def meta_of(run):
    return {(m['node_id'], m['year'], m['day']): m for m in run.dual_header['metadata']['per_block']}


def v_base_dso(run):
    """DSO interface base kV per node: capture terminal DSO v / interface_voltage_terminal dso_pu (constant check)."""
    vt = E._load(os.path.join(run.ev, 'interface_voltage_terminal.json'))['entries']
    last = run.cap[-1]['primal_expectation']['dso']['v']
    ratios = {}
    for e in vt:
        bi = run.blocks.index((e['node_id'], int(e['year']), e['day']))
        ratios.setdefault(e['node_id'], []).append(last[bi][e['period']] / e['dso_pu'])
    return {node: {'v_b_kv': rs[0], 'constant': max(rs) - min(rs) <= 1e-9 * max(abs(max(rs)), 1.0),
                   'min': min(rs), 'max': max(rs)} for node, rs in ratios.items()}


def weight_of(ln, label):
    return ln['q_blocks']['per_agent'][label]['weight']


def channel_parts(run, bk, vbT):
    """Per cycle: price and the TSO-side dual parts per channel (same formulas as E.priced_residual, split by channel;
    a missing V deviation kind (V-2) gives dev_v None)."""
    meta = meta_of(run)
    scen = scen_keys(run)
    om = run.probs[bk]
    duals_by_cycle = {x['cycle']: x for x in run.duals}
    has_v_dev = 'v' in run.cap_header['deviation_kinds']
    out = {}
    for ln in run.cap:
        k = ln['cycle']
        du = duals_by_cycle[k]
        pa = ln['q_blocks']['per_agent']
        price = -sum(v['weight'] * v['settlement_total_unweighted'] for lab, v in pa.items()
                     if lab.split('|', 1)[1] == bk)
        acc = {'exp_p': 0.0, 'exp_q': 0.0, 'exp_v': 0.0, 'dev_p': 0.0, 'dev_q': 0.0, 'dev_v': 0.0}
        pe_, pd_ = ln['primal_expectation'], ln['primal_deviation']
        dv_tso = ln['dual_expectation_v']['tso']
        for bi in block_indices(run, bk):
            node, year, day = run.blocks[bi]
            m = meta[(node, year, day)]
            w = pa[f'TSO|{year}|{day}']['weight']
            fac_pq = w * m['admm_objective_scale_tso'] / (m['tso_base_mva'] * m['interface_rating_mva'])
            fac_v = w * m['admm_objective_scale_tso'] / (vbT[node]['v_b_kv'] ** 2)
            for t in range(len(pe_['tso']['p'][bi])):
                acc['exp_p'] += du['lambda_pf_p_tso'][bi][t] * (pe_['tso']['p'][bi][t] - pe_['dso']['p'][bi][t]) * fac_pq
                acc['exp_q'] += du['lambda_pf_q_tso'][bi][t] * (pe_['tso']['q'][bi][t] - pe_['dso']['q'][bi][t]) * fac_pq
                acc['exp_v'] += dv_tso[bi][t] * (pe_['tso']['v'][bi][t] - pe_['dso']['v'][bi][t]) * fac_v
                for si, s in enumerate(scen):
                    for kind, fac in (('p', fac_pq), ('q', fac_pq), ('v', fac_v)):
                        if kind == 'v' and not has_v_dev:
                            continue
                        xT = pe_['tso'][kind][bi][t] + pd_['tso'][kind][bi][si][t]
                        xD = pe_['dso'][kind][bi][t] + pd_['dso'][kind][bi][si][t]
                        acc[f'dev_{kind}'] += om[s] * du[f'lambda_dev_{kind}_tso'][bi][si][t] * (xT - xD) * fac
        if not has_v_dev:
            acc['dev_v'] = None
        dual = sum(v for v in acc.values() if v is not None)
        out[k] = {'price': price, **acc, 'expectation_part': acc['exp_p'] + acc['exp_q'] + acc['exp_v'],
                  'deviation_part': acc['dev_p'] + acc['dev_q'] + (acc['dev_v'] or 0.0), 'dual': dual,
                  'PR': price + dual}
    return out


def validate_parts(run, parts_by_block, vbT):
    """V-1: every block / cycle against the imported E.priced_residual (price, expectation, deviation parts).
    Every coupled run: sum over blocks of dev_p against production priced_residual_dual_part (rel 1e-9, the
    evaluator's p7_validation rule)."""
    out = {'against_E_priced_residual': None, 'dev_p_vs_production': None}
    if run.arm == 'V1':
        worst = 0.0
        bad = []
        for bk, parts in parts_by_block.items():
            ref = E.priced_residual(run, bk, vbT)
            for k, p in parts.items():
                r = ref[k]
                scale = max(1.0, r['A'])
                for a, b in (('price', 'price'), ('expectation_part', 'expectation_part'),
                             ('deviation_part', 'deviation_part'), ('PR', 'PR'), ('dev_p', 'deviation_p_part')):
                    d = abs(p[a] - r[b]) / scale
                    worst = max(worst, d)
                    if d > VALIDATION_REL:
                        bad.append({'block': bk, 'cycle': k, 'field': a, 'ours': p[a], 'E': r[b]})
        out['against_E_priced_residual'] = {'worst_abs_diff_over_A': worst, 'n_failed': len(bad),
                                            'failed_first_5': bad[:5], 'held': not bad}
    worst = 0.0
    bad = []
    for r in run.rows:
        k = r['cycle']
        s = sum(parts[k]['dev_p'] for parts in parts_by_block.values())
        rel = E._rel(s, r['priced_residual_dual_part'])
        worst = max(worst, rel)
        if rel > 1e-9:
            bad.append({'cycle': k, 'ours': s, 'production': r['priced_residual_dual_part'], 'rel': rel})
    out['dev_p_vs_production'] = {'worst_rel': worst, 'n_failed': len(bad), 'failed_first_5': bad[:5],
                                  'held': not bad}
    out['held'] = (out['dev_p_vs_production']['held']
                   and (out['against_E_priced_residual'] is None or out['against_E_priced_residual']['held']))
    return out


# ======================================================================================================================
#  D1 fits
# ======================================================================================================================
def lsq3(series, window, h=HOLDOUT):
    fit = [k for k in window[:-h] if k in series]
    held = [k for k in window[-h:] if k in series]
    if len(fit) < 4:
        return {'valid': False, 'reason': 'fewer than 4 fitted points'}
    k0 = fit[0]
    ys = [series[k] for k in fit]

    def solve(r):
        xs = [r ** (k - k0) for k in fit]
        n = len(xs)
        sx, sxx = sum(xs), sum(x * x for x in xs)
        sy, sxy = sum(ys), sum(x * y for x, y in zip(xs, ys))
        det = n * sxx - sx * sx
        if det == 0.0:
            return None
        a = (n * sxy - sx * sy) / det
        lim = (sy - a * sx) / n
        sse = sum((lim + a * x - y) ** 2 for x, y in zip(xs, ys))
        return sse, lim, a

    best = None
    for i in range(1, 1000):
        r = i / 1000.0
        s = solve(r)
        if s is not None and (best is None or s[0] < best[0]):
            best = (s[0], r)
    if best is None:
        return {'valid': False, 'reason': 'degenerate'}
    lo, hi = max(1e-6, best[1] - 0.001), min(1.0 - 1e-9, best[1] + 0.001)
    g = (math.sqrt(5) - 1) / 2
    a_, b_ = lo, hi
    c_, d_ = b_ - g * (b_ - a_), a_ + g * (b_ - a_)
    for _ in range(80):
        sc, sd = solve(c_), solve(d_)
        if sc is None or sd is None:
            break
        if sc[0] < sd[0]:
            b_, d_ = d_, c_
            c_ = b_ - g * (b_ - a_)
        else:
            a_, c_ = c_, d_
            d_ = a_ + g * (b_ - a_)
    r = (a_ + b_) / 2
    sse, lim, amp = solve(r)
    pred = [{'cycle': k, 'predicted': lim + amp * r ** (k - k0), 'observed': series[k],
             'abs_error': abs(lim + amp * r ** (k - k0) - series[k])} for k in held]
    return {'valid': True, 'model': 'X(k) = L + A r^(k - k_first)', 'k_first': k0, 'fit_cycles': fit,
            'r': r, 'L_limit': lim, 'A': amp, 'sse': sse, 'heldout': pred,
            'H': max(x['abs_error'] for x in pred) if pred else None,
            'r_at_grid_boundary': best[1] <= 0.001 or best[1] >= 0.999}


def fits(series, window):
    sb = E.settling_bar(series, window, W=20, h=HOLDOUT)
    out = {'settling_bar': sb}
    if sb.get('fit_r_heldout') is not None and sb['fit_r_heldout'] < 1.0:
        win = [k for k in sorted(window) if k in series]
        win = win[-min(20, len(win)):]
        fitted = [k for k in win[:-HOLDOUT] if (k - 1) in win]
        if fitted:
            k0 = fitted[-1]
            d0 = series[k0] - series[k0 - 1]
            rf = sb['fit_r_heldout']
            out['limit_from_heldout_fit'] = {'k0': k0, 'd_k0': d0, 'r_fit': rf,
                                             'limit': series[k0] + d0 * rf / (1.0 - rf)}
    out['lsq3'] = lsq3(series, [k for k in sorted(window) if k in series])
    return out


def d1(runs, step4_spec, parts_all):
    blocks = step4_spec['ii_central_and_allowance']['blocks']
    out = {}
    for key, run in runs.items():
        if not run.coupled:
            continue
        inst = key.split('|')[0]
        out[key] = {}
        for bk in step4_spec['i_arms_instances_cap']['instances'][inst]['blocks']:
            wc = blocks[bk]['w_b_C_central_eur']
            q = run.q_series(bk)
            parts = parts_all[key][bk]
            pins = {ln['cycle']: (ln.get('voltage_pin') or {}).get('per_year_day', {}).get(bk) for ln in run.cap}
            rows = []
            ser = {n: {} for n in ('e', 'o1', 'o2', 'o3', 'price', 'dual')}
            for k in sorted(q):
                if k < MIN_CYCLE:
                    continue
                p = parts[k]
                e = q[k] - wc
                o1, o2, o3 = e - p['price'], e - p['price'] + p['dual'], e - p['price'] - p['dual']
                rows.append({'cycle': k, 'Q_b': q[k], 'e_b': e, 'price_b': p['price'], 'exp_p': p['exp_p'],
                             'exp_q': p['exp_q'], 'exp_v': p['exp_v'], 'dev_p': p['dev_p'], 'dev_q': p['dev_q'],
                             'dev_v': p['dev_v'], 'dual': p['dual'], 'PR': p['PR'], 'o1': o1, 'o2': o2, 'o3': o3,
                             'voltage_pin_block': pins.get(k), 'tail_depth': k in run.tail_cycles})
                for n, v in (('e', e), ('o1', o1), ('o2', o2), ('o3', o3), ('price', p['price']), ('dual', p['dual'])):
                    ser[n][k] = v
            T = [k for k in run.tail_cycles if k >= MIN_CYCLE]
            rec = {'calibrated': blocks[bk]['calibrated'], 'w_b_C_central_eur': wc,
                   'C_central_status': 'calibrated' if blocks[bk]['calibrated'] else 'MERGED-ONLY (not calibrated)',
                   'tail_depth_cycles': T, 'K': max(q), 'rows_ge_21': rows,
                   'fits_tail_window': {n: fits(ser[n], T) for n in ('e', 'o1', 'o2', 'o3')}}
            o1T = [ser['o1'][k] for k in T]
            rec['o1_tail'] = {'all_positive': all(v > 0 for v in o1T),
                              'strictly_decreasing': all(ser['o1'][k] < ser['o1'][k - 1] for k in T if (k - 1) in T),
                              'steps': {str(k): ser['o1'][k] - ser['o1'][k - 1] for k in T if (k - 1) in T}}
            K1, K = T[0], T[-1]
            a, b = ser['o1'][K1], ser['o1'][K]
            total_rel = (b - a) / abs(a) if a != 0 else None
            g = ((b / a) ** (1.0 / (K - K1)) - 1.0) if (a != 0 and b != 0 and (a > 0) == (b > 0) and K > K1) else None
            if total_rel is not None and abs(total_rel) <= Q3_PLATEAU_TOTAL_REL:
                cls = 'plateau'
            elif g is not None and -Q3_DECAY_PER_CYCLE <= g < 0:
                cls = 'decays at <= 3 %/cycle (B open)'
            elif g is not None and g < -Q3_DECAY_PER_CYCLE:
                cls = 'decays faster than 3 %/cycle'
            else:
                cls = 'grows or changes sign'
            rec['o1_tail_classification'] = {'K1': K1, 'K': K, 'o1_K1': a, 'o1_K': b, 'total_rel': total_rel,
                                             'mean_per_cycle_rel_change': g, 'class': cls}
            hv = {}
            for hk, h in HAND_VALUES.items():
                hi, ha, hb, hd, hs = hk.split('|')
                if f'{hi}|{ha}' == key and f'{hb}|{hd}' == bk:
                    hv[hs] = [{'cycle': c, 'hand': v, 'computed': ser[hs].get(c),
                               'diff': (ser[hs][c] - v) if c in ser[hs] else None}
                              for c, v in zip(h['cycles'], h['values'])]
            rec['hand_value_comparison'] = hv
            out[key][bk] = rec
    return out


# ======================================================================================================================
#  D2 dual consistency
# ======================================================================================================================
def d2(runs):
    out = {}
    for key, run in runs.items():
        if not run.coupled:
            continue
        meta = meta_of(run)
        vbT, vbD = E.v_base(run), v_base_dso(run)
        scen = scen_keys(run)
        kinds = run.cap_header['deviation_kinds']
        cap_by = {x['cycle']: x for x in run.cap}
        du_by = {x['cycle']: x for x in run.duals}
        K = len(run.cap)
        # rho equality at every captured cycle
        rho_mismatch = []
        for ln in run.cap:
            for (node, year, day) in run.blocks:
                t, d = ln['rho'][f'TSO|{year}|{day}'], ln['rho'][f'DSO{node}|{year}|{day}']
                for f in ('rho_pf', 'rho_v'):
                    if t[f] != d[f]:
                        rho_mismatch.append({'cycle': ln['cycle'], 'block': f'{node}|{year}|{day}', 'field': f,
                                             'tso': t[f], 'dso': d[f]})
        per_cycle = {}
        for k in range(max(1, K - D2_LAST + 1), K + 1):
            ln, du = cap_by[k], du_by[k]
            pe_, pd_ = ln['primal_expectation'], ln['primal_deviation']
            res = {}
            tot_signed = tot_abs = 0.0
            for bi, (node, year, day) in enumerate(run.blocks):
                bk = f'{year}|{day}'
                m = meta[(node, year, day)]
                wT = weight_of(ln, f'TSO|{year}|{day}')
                wD = weight_of(ln, f'DSO{node}|{year}|{day}')
                R = m['interface_rating_mva']
                facT = {'p': wT * m['admm_objective_scale_tso'] / (m['tso_base_mva'] * R),
                        'v': wT * m['admm_objective_scale_tso'] / vbT[node]['v_b_kv'] ** 2}
                facD = {'p': wD * m['admm_objective_scale_dso'] / (m['dso_base_mva'] * R),
                        'v': wD * m['admm_objective_scale_dso'] / vbD[node]['v_b_kv'] ** 2}
                facT['q'], facD['q'] = facT['p'], facD['p']
                om = run.probs[bk]
                for kind in ('p', 'q', 'v'):
                    # expectation
                    if kind == 'v':
                        lt, ld = ln['dual_expectation_v']['tso'][bi], ln['dual_expectation_v']['dso'][bi]
                    else:
                        lt, ld = du[f'lambda_pf_{kind}_tso'][bi], du[f'lambda_pf_{kind}_dso'][bi]
                    agg = res.setdefault(f'{bk}|exp|{kind}', {'max_abs_sum': 0.0, 'sumsq': 0.0, 'n': 0,
                                                              'max_abs_mu_T': 0.0, 'eur_signed': 0.0, 'eur_abs': 0.0,
                                                              'n_bitwise_opposite': 0, 'n_coords': 0})
                    for t in range(len(lt)):
                        mT, mD = facT[kind] * lt[t], facD[kind] * ld[t]
                        s = mT + mD
                        r = pe_['tso'][kind][bi][t] - pe_['dso'][kind][bi][t]
                        agg['max_abs_sum'] = max(agg['max_abs_sum'], abs(s))
                        agg['sumsq'] += s * s
                        agg['n'] += 1
                        agg['max_abs_mu_T'] = max(agg['max_abs_mu_T'], abs(mT))
                        agg['eur_signed'] += s * r
                        agg['eur_abs'] += abs(s * r)
                        agg['n_bitwise_opposite'] += int(lt[t] == -ld[t])
                        agg['n_coords'] += 1
                    # deviation
                    if kind not in kinds:
                        continue
                    lT, lD = du[f'lambda_dev_{kind}_tso'][bi], du[f'lambda_dev_{kind}_dso'][bi]
                    agg = res.setdefault(f'{bk}|dev|{kind}', {'max_abs_sum': 0.0, 'sumsq_omega': 0.0, 'n_per_s': 0,
                                                              'max_abs_mu_T': 0.0, 'eur_signed': 0.0, 'eur_abs': 0.0,
                                                              'n_bitwise_opposite': 0, 'n_coords': 0})
                    for si, sk in enumerate(scen):
                        for t in range(len(lT[si])):
                            mT, mD = facT[kind] * lT[si][t], facD[kind] * lD[si][t]
                            s = mT + mD
                            r = pd_['tso'][kind][bi][si][t] - pd_['dso'][kind][bi][si][t]
                            agg['max_abs_sum'] = max(agg['max_abs_sum'], abs(s))
                            agg['sumsq_omega'] += om[sk] * s * s
                            agg['max_abs_mu_T'] = max(agg['max_abs_mu_T'], abs(mT))
                            agg['eur_signed'] += om[sk] * s * r
                            agg['eur_abs'] += abs(om[sk] * s * r)
                            agg['n_bitwise_opposite'] += int(lT[si][t] == -lD[si][t])
                            agg['n_coords'] += 1
                    agg['n_per_s'] += len(lT[0])
            for name, agg in res.items():
                if 'sumsq' in agg:
                    agg['omega_rms_sum'] = math.sqrt(agg.pop('sumsq') / agg.pop('n'))
                else:
                    agg['omega_rms_sum'] = math.sqrt(agg.pop('sumsq_omega') / agg.pop('n_per_s'))
                agg['unit'] = {'p': 'EUR/MW', 'q': 'EUR/MVAr', 'v': 'EUR/kV'}[name.rsplit('|', 1)[1]]
                agg['max_abs_sum_over_max_abs_mu_T'] = (agg['max_abs_sum'] / agg['max_abs_mu_T']
                                                        if agg['max_abs_mu_T'] > 0 else None)
                tot_signed += agg['eur_signed']
                tot_abs += agg['eur_abs']
            per_cycle[str(k)] = {'per_block_kind_channel': res, 'eur_signed_total': tot_signed,
                                 'eur_abs_total': tot_abs}
        out[key] = {'K': K, 'cycles': list(range(max(1, K - D2_LAST + 1), K + 1)), 'per_cycle': per_cycle,
                    'terminal': per_cycle[str(K)], 'v_base_tso': vbT, 'v_base_dso': vbD,
                    'rho_T_equals_rho_D_every_cycle': not rho_mismatch, 'rho_mismatch_first_10': rho_mismatch[:10],
                    'n_rho_mismatch': len(rho_mismatch),
                    'eur_abs_total_last_10': {c: per_cycle[c]['eur_abs_total'] for c in per_cycle},
                    'eur_signed_total_last_10': {c: per_cycle[c]['eur_signed_total'] for c in per_cycle}}
    return out


# ======================================================================================================================
#  D3 ADMM-only boxes
# ======================================================================================================================
def _box_entry(store, fam, margin, where, unit, bound_abs, half):
    e = store.setdefault(fam, {'unit': unit, 'n': 0, 'min_margin': math.inf, 'min_margin_at': None, 'n_active': 0,
                               'n_near_active': 0, 'n_outside_strict': 0})
    e['n'] += 1
    if margin < e['min_margin']:
        e['min_margin'], e['min_margin_at'] = margin, where
    if margin <= ACTIVE_TOL_REL * max(1.0, bound_abs):
        e['n_active'] += 1
    if half and margin / half < NEAR_ACTIVE_REL:
        e['n_near_active'] += 1
    if margin < 0:
        e['n_outside_strict'] += 1


def d3(runs, diag):
    out = {}
    for key, run in runs.items():
        meta = meta_of(run)
        isd = E._load(os.path.join(run.ev, 'interface_settlement_detail_s31c.json'))['interface_reporting_detail']
        ivt = E._load(os.path.join(run.ev, 'interface_voltage_terminal.json'))['entries']
        tso, beside_dso, beside_v, cross, every = {}, {}, {}, {'max_abs_p_mw': 0.0, 'max_abs_q_mvar': 0.0}, {}
        last = run.cap[-1]
        anchors = {}
        for bi, (node, year, day) in enumerate(run.blocks):
            R = meta[(node, year, day)]['interface_rating_mva']
            per = isd[str(node)][str(year)][day]['periods']
            for p_str, v in per.items():
                p = int(p_str)
                for sk in v['delta_p_mw']:
                    where = f'DN{node} {year}|{day} s{sk} t{p}'
                    _box_entry(tso, 'TSO_P_delta_box', R - abs(v['delta_p_mw'][sk]), where, 'MW', R, R)
                    _box_entry(tso, 'TSO_Q_delta_box', R - abs(v['delta_q_mvar'][sk]), where, 'MVAr', R, R)
                    anchors[(bi, p, sk)] = (v['anchor_p_mw'][sk], v['anchor_q_mvar'][sk])
                    if run.coupled:
                        si = scen_keys(run).index(sk)
                        PT = (last['primal_expectation']['tso']['p'][bi][p]
                              + last['primal_deviation']['tso']['p'][bi][si][p])
                        QT = (last['primal_expectation']['tso']['q'][bi][p]
                              + last['primal_deviation']['tso']['q'][bi][si][p])
                    else:
                        PT = last['primal_expectation']['tso']['p'][bi][p]
                        QT = last['primal_expectation']['tso']['q'][bi][p]
                    cross['max_abs_p_mw'] = max(cross['max_abs_p_mw'],
                                                abs(v['anchor_p_mw'][sk] + v['delta_p_mw'][sk] - PT))
                    cross['max_abs_q_mvar'] = max(cross['max_abs_q_mvar'],
                                                  abs(v['anchor_q_mvar'][sk] + v['delta_q_mvar'][sk] - QT))
        rec = {'tso_delta_box_terminal': tso, 'cross_check_anchor_plus_delta_vs_capture_terminal': cross}
        if run.coupled:
            # beside (i): the TSO box at every cycle >= 21 from the capture, terminal anchor (pc fixed)
            for ln in run.cap:
                if ln['cycle'] < MIN_CYCLE:
                    continue
                for bi, (node, year, day) in enumerate(run.blocks):
                    R = meta[(node, year, day)]['interface_rating_mva']
                    for si, sk in enumerate(scen_keys(run)):
                        for p in range(len(ln['primal_expectation']['tso']['p'][bi])):
                            ap, aq = anchors[(bi, p, sk)]
                            PT = ln['primal_expectation']['tso']['p'][bi][p] + ln['primal_deviation']['tso']['p'][bi][si][p]
                            QT = ln['primal_expectation']['tso']['q'][bi][p] + ln['primal_deviation']['tso']['q'][bi][si][p]
                            where = f'cycle {ln["cycle"]} DN{node} {year}|{day} s{sk} t{p}'
                            _box_entry(every, 'TSO_P_delta_box', R - abs(PT - ap), where, 'MW', R, R)
                            _box_entry(every, 'TSO_Q_delta_box', R - abs(QT - aq), where, 'MVAr', R, R)
            rec['beside_tso_delta_box_every_cycle_ge_21'] = every
            # beside (ii): DSO reference generator against step 3 relaxation_limits; (iii) V rows per scenario
            vbT, vbD = E.v_base(run), v_base_dso(run)
            has_v_dev = 'v' in run.cap_header['deviation_kinds']
            for bi, (node, year, day) in enumerate(run.blocks):
                rl = diag['phase_A'][f'{year}|{day}']['relaxation_limits'][str(node)]
                for si, sk in enumerate(scen_keys(run)):
                    for p in range(len(last['primal_expectation']['dso']['p'][bi])):
                        where = f'DN{node} {year}|{day} s{sk} t{p}'
                        PD = last['primal_expectation']['dso']['p'][bi][p] + last['primal_deviation']['dso']['p'][bi][si][p]
                        QD = last['primal_expectation']['dso']['q'][bi][p] + last['primal_deviation']['dso']['q'][bi][si][p]
                        for fam, val, (lo, hi), unit in (('DSO_P_refgen_box', PD, rl['ref_gen_p_mw'], 'MW'),
                                                         ('DSO_Q_refgen_box', QD, rl['ref_gen_q_mvar'], 'MVAr')):
                            _box_entry(beside_dso, fam, min(val - lo, hi - val), where, unit,
                                       max(abs(lo), abs(hi)), (hi - lo) / 2)
                        if has_v_dev:
                            VT = (last['primal_expectation']['tso']['v'][bi][p]
                                  + last['primal_deviation']['tso']['v'][bi][si][p]) / vbT[node]['v_b_kv']
                            VD = (last['primal_expectation']['dso']['v'][bi][p]
                                  + last['primal_deviation']['dso']['v'][bi][si][p]) / vbD[node]['v_b_kv']
                            lo, hi = rl['tn_bus_v_pu']
                            _box_entry(beside_v, 'TSO_V_row_per_scenario', min(VT - lo, hi - VT), where, 'pu', 1.0,
                                       (hi - lo) / 2)
                            lo, hi = rl['dn_ref_node_v_pu']
                            _box_entry(beside_v, 'DSO_V_row_per_scenario', min(VD - lo, hi - VD), where, 'pu', 1.0,
                                       (hi - lo) / 2)
            rec['beside_dso_refgen_box_terminal'] = beside_dso
        for e in ivt:
            where = f'DN{e["node_id"]} {e["year"]}|{e["day"]} t{e["period"]} (expectation)'
            lo, hi = e['v_min_pu'], e['v_max_pu']
            _box_entry(beside_v, 'TSO_V_row_expectation', min(e['tso_pu'] - lo, hi - e['tso_pu']), where, 'pu', 1.0,
                       (hi - lo) / 2)
            _box_entry(beside_v, 'DSO_V_row_expectation', min(e['dso_pu'] - lo, hi - e['dso_pu']), where, 'pu', 1.0,
                       (hi - lo) / 2)
        rec['beside_v_rows_terminal'] = beside_v
        rec['active_tso_delta_box'] = sum(e['n_active'] for e in tso.values())
        rec['active_beside_extended'] = {f: e['n_active'] for d_ in (beside_dso, beside_v) for f, e in d_.items()}
        out[key] = rec
    return out


# ======================================================================================================================
#  D4 P7 post hoc
# ======================================================================================================================
def d4(d1res, ev_json):
    out = {'label': 'POST HOC (corrected sign); the frozen P7 outcome stands and is not replaced', 'per_run': {}}
    for key, blocks in d1res.items():
        inst = key.split('|')[0]
        out['per_run'][key] = {}
        for bk, rec in blocks.items():
            rows = rec['rows_ge_21']
            T = set(rec['tail_depth_cycles'])

            def agg(rs):
                rhs = sum(abs(r['e_b']) for r in rs)
                lf = sum(abs(r['e_b'] + r['PR']) for r in rs)
                la = sum(abs(r['o2']) for r in rs)
                lb = sum(abs(r['o1']) for r in rs)
                f = (lambda lhs: (rhs / lhs) if lhs > 0 else None)
                return {'n_cycles': len(rs), 'sum_abs_e': rhs,
                        'frozen_sum_abs_e_plus_PR': lf, 'frozen_factor': f(lf),
                        'frozen_held': lf <= rhs / P7_FACTOR,
                        'posthoc_a_sum_abs_e_minus_price_plus_dual': la, 'posthoc_a_factor': f(la),
                        'posthoc_a_held': la <= rhs / P7_FACTOR,
                        'posthoc_b_sum_abs_e_minus_price': lb, 'posthoc_b_factor': f(lb),
                        'posthoc_b_held': lb <= rhs / P7_FACTOR}
            o = {'cycles_ge_21': agg(rows), 'tail_depth_window_beside': agg([r for r in rows if r['cycle'] in T])}
            fz = ((ev_json['instances'][inst].get('P7') or {}).get(bk)) if key.endswith('V1') else None
            if fz is not None:
                o['frozen_evaluation_record'] = {k: fz[k] for k in ('n_cycles', 'sum_abs_Q_plus_PR_minus_wC',
                                                                     'sum_abs_Q_minus_wC', 'reduction_factor', 'held')}
                o['recomputed_frozen_matches_evaluation'] = (
                    o['cycles_ge_21']['n_cycles'] == fz['n_cycles']
                    and E._rel(o['cycles_ge_21']['frozen_sum_abs_e_plus_PR'], fz['sum_abs_Q_plus_PR_minus_wC']) <= 1e-9
                    and E._rel(o['cycles_ge_21']['sum_abs_e'], fz['sum_abs_Q_minus_wC']) <= 1e-9)
            out['per_run'][key][bk] = o
    return out


# ======================================================================================================================
#  D5 rounding
# ======================================================================================================================
def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for m in range(i, j + 1):
            ranks[order[m]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    sab = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    saa = sum((x - ma) ** 2 for x in ra)
    sbb = sum((y - mb) ** 2 for y in rb)
    return sab / math.sqrt(saa * sbb) if saa > 0 and sbb > 0 else None


def d5(runs, ev_json):
    out = {}
    for key, run in runs.items():
        if not run.coupled:
            continue
        isd = E._load(os.path.join(run.ev, 'interface_settlement_detail_s31c.json'))['interface_reporting_detail']
        scen = scen_keys(run)
        rows_by = {r['cycle']: r for r in run.rows}
        per = []
        worst_recompute_rel = 0.0
        for ln in run.cap:
            k = ln['cycle']
            r = rows_by[k]
            pa = ln['q_blocks']['per_agent']
            T_exact = sum((Fraction(v['weight']) * Fraction(v['settlement_total_unweighted']) for v in pa.values()),
                          Fraction(0))
            S_T = sum((abs(Fraction(v['weight']) * Fraction(v['settlement_total_unweighted'])) for v in pa.values()),
                      Fraction(0))
            D_exact = sum((Fraction(v['weight']) * (Fraction(v['settlement_total_unweighted'])
                                                    - Fraction(v['settlement_contracted_unweighted']))
                           for v in pa.values()), Fraction(0))
            S_ii = sum(((abs(Fraction(v['settlement_excluded_in_full'])) + abs(Fraction(v['contracted_excluded']))
                         + abs(Fraction(v['weight']) * Fraction(v['settlement_total_unweighted']))
                         + abs(Fraction(v['weight']) * Fraction(v['settlement_contracted_unweighted'])))
                        for v in pa.values()), Fraction(0))
            T_eval = sum(v['weight'] * v['settlement_total_unweighted'] for v in pa.values())   # E.m1's order
            T_fsum = math.fsum(v['weight'] * v['settlement_total_unweighted'] for v in pa.values())
            pr = Fraction(r['priced_residual_price_part'])
            diff_i = pr + T_exact
            bound_i = C_ROUND * U_ROUND * S_T
            lhs2 = Fraction(r['gross_operational_cost_contracted_excluded']) - Fraction(r['gross_operational_cost'])
            diff_ii = lhs2 - D_exact
            bound_ii = C_ROUND * U_ROUND * S_ii
            # S_fine and the recomputed per-agent settlements (prices from the settlement detail; P from the capture)
            pe_, pd_ = ln['primal_expectation'], ln['primal_deviation']
            s_fine = 0.0
            recomputed = {}
            for bi, (node, year, day) in enumerate(run.blocks):
                per_p = isd[str(node)][str(year)][day]['periods']
                om = run.probs[f'{year}|{day}']
                w = pa[f'TSO|{year}|{day}']['weight']
                for t in range(len(pe_['tso']['p'][bi])):
                    prices = per_p[str(t)]['price_per_mwh_by_scenario']
                    for si, sk in enumerate(scen):
                        PT = pe_['tso']['p'][bi][t] + pd_['tso']['p'][bi][si][t]
                        PD = pe_['dso']['p'][bi][t] + pd_['dso']['p'][bi][si][t]
                        s_fine += w * om[sk] * abs(prices[sk]) * (abs(PT) + abs(PD))
                        recomputed[f'TSO|{year}|{day}'] = recomputed.get(f'TSO|{year}|{day}', 0.0) - om[sk] * prices[sk] * PT
                        recomputed[f'DSO{node}|{year}|{day}'] = (recomputed.get(f'DSO{node}|{year}|{day}', 0.0)
                                                                 + om[sk] * prices[sk] * PD)
            for lab, val in recomputed.items():
                worst_recompute_rel = max(worst_recompute_rel, E._rel(val, pa[lab]['settlement_total_unweighted']))
            rnorm = math.sqrt(r['boyd_pf_r_expectation_part'] ** 2 + r['boyd_pf_r_deviation_part'] ** 2)
            per.append({'cycle': k, 'T_exact': float(T_exact), 'T_eval_float': T_eval, 'T_fsum': T_fsum,
                        'T_eval_minus_exact': float(Fraction(T_eval) - T_exact),
                        'T_fsum_minus_exact': float(Fraction(T_fsum) - T_exact),
                        'pr_price': r['priced_residual_price_part'], 'pr_plus_T': float(diff_i),
                        'S_T': float(S_T), 'bound_i': float(bound_i), 'ratio_i': float(abs(diff_i) / bound_i),
                        'holds_i': abs(diff_i) <= bound_i, 'pr_plus_T_eval_float_abs': abs(r['priced_residual_price_part']
                                                                                              + T_eval),
                        'D_exact': float(D_exact), 'lhs_ii': float(lhs2), 'diff_ii': float(diff_ii),
                        'S_ii': float(S_ii), 'bound_ii': float(bound_ii), 'ratio_ii': float(abs(diff_ii) / bound_ii),
                        'holds_ii': abs(diff_ii) <= bound_ii, 'S_fine': s_fine,
                        'bound_i_with_S_fine': float(C_ROUND * U_ROUND) * s_fine,
                        'abs_T': abs(float(T_exact)), 'pf_residual_norm': rnorm})
        signs = [(1 if x['pr_plus_T'] > 0 else -1) for x in per if x['pr_plus_T'] != 0.0]
        d = [x['pr_plus_T'] for x in per]
        ad = [abs(x) for x in d]
        absT = [x['abs_T'] for x in per]
        rn = [x['pf_residual_norm'] for x in per]
        m1 = ev_json['instances'][key.split('|')[0]]['runs'][key.split('|')[1]]['M1']
        evaluator_worst = m1['beside_not_deciding_worst']['i_abs_diff']
        ours_worst = max(x['pr_plus_T_eval_float_abs'] for x in per)
        out[key] = {'per_cycle': per, 'n_cycles': len(per),
                    'all_hold_i': all(x['holds_i'] for x in per), 'n_fail_i': sum(not x['holds_i'] for x in per),
                    'worst_ratio_i': max(x['ratio_i'] for x in per),
                    'all_hold_ii_beside': all(x['holds_ii'] for x in per),
                    'worst_ratio_ii': max(x['ratio_ii'] for x in per),
                    'n_positive': signs.count(1), 'n_negative': signs.count(-1),
                    'n_zero': sum(1 for x in per if x['pr_plus_T'] == 0.0), 'sign_mixed': len(set(signs)) == 2,
                    'spearman_signed_vs_abs_T': spearman(d, absT), 'spearman_signed_vs_residual': spearman(d, rn),
                    'spearman_abs_vs_abs_T': spearman(ad, absT), 'spearman_abs_vs_residual': spearman(ad, rn),
                    'worst_recomputed_settlement_rel': worst_recompute_rel,
                    'max_T_eval_minus_exact_abs': max(abs(x['T_eval_minus_exact']) for x in per),
                    'max_T_fsum_minus_exact_abs': max(abs(x['T_fsum_minus_exact']) for x in per),
                    'evaluator_worst_i_abs_diff': evaluator_worst, 'ours_worst_with_evaluator_float_sum': ours_worst,
                    'reproduces_evaluator_M1_beside': ours_worst == evaluator_worst}
    return out


# ======================================================================================================================
#  predictions
# ======================================================================================================================
def evaluate_predictions(d1r, d2r, d3r, d5r):
    q = {}
    # Q1
    per = {}
    for key in ('A|V1', 'A|V2'):
        rec = d1r[key]['2025|Winter']
        sb = rec['fits_tail_window']['o2']['settling_bar']
        a = rec['o1_tail']['all_positive'] and rec['o1_tail']['strictly_decreasing']
        b_eval = bool(sb.get('valid'))
        b = b_eval and 0.0 <= sb['Q_hat_inf'] <= 100.0 and sb['H'] <= 50.0
        per[key] = {'o1_positive': rec['o1_tail']['all_positive'], 'o1_decreasing': rec['o1_tail']['strictly_decreasing'],
                    'o2_fit_valid': b_eval, 'o2_limit': sb.get('Q_hat_inf'), 'o2_heldout_H': sb.get('H'),
                    'o2_settling_bar_label': sb.get('label'), 'o2_reason_if_invalid': sb.get('reason'),
                    'lsq3_beside': {k: rec['fits_tail_window']['o2']['lsq3'].get(k) for k in ('r', 'L_limit', 'H',
                                                                                             'r_at_grid_boundary')},
                    'clause_a': a, 'clause_b': b}
    q['Q1'] = {'per_run': per, 'held': all(v['clause_a'] and v['clause_b'] for v in per.values())}
    # Q2
    o2 = {key: d1r[key]['2025|Winter']['rows_ge_21'][-1]['o2'] for key in ('A|V1', 'A|V2')}
    q['Q2'] = {'decisive_run': 'A|V1', 'o2_K': o2, 'in_range': {k: 100.0 <= v <= 230.0 for k, v in o2.items()},
               'held': 100.0 <= o2['A|V1'] <= 230.0}
    # Q3
    o1 = {key: d1r[key]['2035|Summer']['rows_ge_21'][-1]['o1'] for key in ('B|V1', 'B|V2')}
    q['Q3'] = {'decisive_run': 'B|V1', 'o1_K': o1, 'in_range': {k: 700.0 <= v <= 850.0 for k, v in o1.items()},
               'held': 700.0 <= o1['B|V1'] <= 850.0,
               'competing_outcome_recorded': {k: d1r[k]['2035|Summer']['o1_tail_classification'] for k in ('B|V1', 'B|V2')}}
    # Q4
    v = {key: d2r[key]['terminal']['eur_abs_total'] for key in ('A|V1', 'A|V2')}
    q['Q4'] = {'decisive_run': 'A|V1', 'eur_abs_terminal': v,
               'eur_signed_terminal': {key: d2r[key]['terminal']['eur_signed_total'] for key in ('A|V1', 'A|V2')},
               'held': v['A|V1'] <= 50.0}
    # Q5
    act = {key: r['active_tso_delta_box'] for key, r in d3r.items()}
    mins = {key: {f: (e['min_margin'], e['min_margin_at']) for f, e in r['tso_delta_box_terminal'].items()}
            for key, r in d3r.items()}
    q['Q5'] = {'active_count_per_run': act, 'min_margin_per_run': mins, 'held': all(v == 0 for v in act.values()),
               'extended_beside_active_counts': {key: r['active_beside_extended'] for key, r in d3r.items()}}
    # Q6
    q['Q6'] = {'per_run': {key: {'all_hold_i': r['all_hold_i'], 'n_fail_i': r['n_fail_i'],
                                 'worst_ratio_i': r['worst_ratio_i'], 'sign_mixed': r['sign_mixed'],
                                 'n_positive': r['n_positive'], 'n_negative': r['n_negative'], 'n_zero': r['n_zero']}
                           for key, r in d5r.items()},
               'held': all(r['all_hold_i'] and r['sign_mixed'] for r in d5r.values()) and len(d5r) == 4}
    return q


# ======================================================================================================================
#  main
# ======================================================================================================================
def load_runs(step4_spec):
    runs = {}
    for inst in INSTANCES:
        for arm in ALL_ARMS:
            runs[f'{inst}|{arm}'] = E.Run(step4_spec, 'main', inst, arm)
    return runs


def check_only():
    step4_spec, _sha4, _ = load_step4_spec()
    lines, bad = code_line_check()
    if bad:
        raise RuntimeError(f'code-line check failed: {bad}')
    runs = load_runs(step4_spec)
    res = assert_capture_path(runs, step4_spec)
    _log(f'capture path OK ({len(res["checked_runs"])} runs); code lines {len(lines)} verbatim; guard {GUARD.counts}')
    return 0 if not GUARD.verify(0) else 1


def run_main(spec_path):
    spec = E._load(spec_path)
    sha = _sha(spec_path)
    tag = os.path.basename(spec_path).rsplit('_', 1)[-1].split('.')[0]
    if not sha.startswith(tag) or not os.path.basename(spec_path).startswith(SPEC_PREFIX):
        raise RuntimeError(f'spec {spec_path}: sha256 {sha[:8]} does not match its name tag {tag}')
    if os.path.abspath(os.path.dirname(spec_path)) != os.path.abspath(OUT_DIR):
        raise RuntimeError(f'spec is not in {OUT_DIR}')
    res = {'schema': 'p515_s56_step4_postdiag_results_v1', 'utc_start': _utc(),
           'spec': {'path': os.path.relpath(os.path.abspath(spec_path), REPO), 'sha256': sha},
           'script': SCRIPT, 'script_sha256': _sha(SCRIPT),
           'script_sha256_equals_frozen': _sha(SCRIPT) == spec['script_sha256_at_freeze'],
           'git_head': L._git('rev-parse', 'HEAD').strip(), 'objective_convention': spec['objective_convention'],
           'errors': {}}
    # inputs unchanged since the freeze
    changed = [f for f, h in spec['input_sha256_inventory'].items() if _sha(f) != h]
    if changed:
        raise RuntimeError(f'input files changed since the freeze: {changed}')
    res['inputs_unchanged'] = True
    lines, bad = code_line_check()
    if bad:
        raise RuntimeError(f'code-line check failed: {bad}')
    res['code_lines_verbatim'] = len(lines)
    step4_spec, _s, _r = load_step4_spec()
    runs = load_runs(step4_spec)
    res['capture_path'] = assert_capture_path(runs, step4_spec)
    _log('capture path asserted; computing')
    ev_json = E._load(STEP4_EVALUATION)
    diag = E._load(DIAG_RESULTS)
    # per-channel parts + validation
    parts_all, validation, vbs = {}, {}, {}
    for key, run in runs.items():
        if not run.coupled:
            continue
        vbT = E.v_base(run)
        vbs[key] = {'tso': vbT, 'tso_constant': all(v['constant'] for v in vbT.values())}
        bks = sorted({f'{y}|{d}' for _, y, d in run.blocks})
        parts_all[key] = {bk: channel_parts(run, bk, vbT) for bk in bks}
        validation[key] = validate_parts(run, parts_all[key], vbT)
    res['v_base'] = vbs
    res['validation_parts'] = validation
    if not all(v['held'] for v in validation.values()):
        res['errors']['validation_parts'] = 'per-channel parts do not reproduce E.priced_residual / production'
    for name, fn in (('D1', lambda: d1(runs, step4_spec, parts_all)), ('D2', lambda: d2(runs)),
                     ('D3', lambda: d3(runs, diag))):
        try:
            res[name] = fn()
            _log(f'{name} done')
        except Exception:  # noqa: BLE001
            res['errors'][name] = traceback.format_exc()
    try:
        res['D4'] = d4(res['D1'], ev_json)
        _log('D4 done')
    except Exception:  # noqa: BLE001
        res['errors']['D4'] = traceback.format_exc()
    try:
        res['D5'] = d5(runs, ev_json)
        _log('D5 done')
    except Exception:  # noqa: BLE001
        res['errors']['D5'] = traceback.format_exc()
    try:
        res['predictions'] = evaluate_predictions(res['D1'], res['D2'], res['D3'], res['D5'])
        for qn, qv in res['predictions'].items():
            _log(f'{qn}: held={qv["held"]}')
    except Exception:  # noqa: BLE001
        res['errors']['predictions'] = traceback.format_exc()
    res['guard'] = {'counts': dict(GUARD.counts), 'verify_failures': GUARD.verify(0),
                    'evaluator_guard_counts': dict(E.GUARD.counts), 'launcher_guard_counts': dict(L.GUARD.counts)}
    res['utc_end'] = _utc()
    path = os.path.join(OUT_DIR, f'postdiag_results_{L._stamp()}.json')
    write_json_new(path, res)
    _log(f'results -> {path}; errors {list(res["errors"])}; guard {res["guard"]["counts"]} '
         f'verify {res["guard"]["verify_failures"]}')
    for e in res['errors'].values():
        print(e, file=sys.stderr)
    return 0 if not res['errors'] and not res['guard']['verify_failures'] else 2


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze', action='store_true')
    g.add_argument('--check', action='store_true')
    g.add_argument('--spec')
    a = ap.parse_args()
    if a.freeze:
        return freeze()
    if a.check:
        return check_only()
    return run_main(a.spec)


if __name__ == '__main__':
    sys.exit(main())
