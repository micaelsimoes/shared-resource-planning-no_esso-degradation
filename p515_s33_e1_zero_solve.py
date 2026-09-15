"""P5.15 Step 3.3(a) Addendum 14 -- E1 zero-solve diagnostic (SolveProfileGuard armed at zero).

Answers Q1 (interface V vs bounds), Q2 (ESSO objective scale vs TSO/DSO sigma, D5 evidence) and
Q3 (low-priority checks: ess tso-prev vs proximal centre; PF ||y|| units) using ONLY artifacts
already serialized under data/SRP1/Results/P515S32_run/ plus the static case9 bus table and
SRP1_params.json. No production, case-file or harness edit. No solve of any kind -- the guard is
armed for the whole run and verified with expected_solves=0.

Reads (all pre-existing, none written by this script):
  data/SRP1/Results/P515S32_run/g_baseline.json
  data/SRP1/Results/P515S32_run/component_levels_terminal.json
  data/SRP1/Results/P515S32_run/interface_settlement_detail_s31c.json
  data/SRP1/Results/P515S32_run/stdout_baseline.log
  data/SRP1/Results/P515S32_run/esso_models_baseline.pkl   (unpickled; pe.value() reads only)
  data/SRP1/Results/P515S32_run/frozen_snapshots_baseline.jsonl
  data/SRP1/Results/P515S31C_run/stdout_baseline.log        (cross-check only: same sigma?)
  data/SRP1/case9/case9_2025.json                           (bus v_min/v_max/baseKV, nodes 5/7/9)
  data/SRP1/SRP1_params.json                                (gamma_tso, for context)
  shared_resources_planning.py                              (line citations only, not imported
                                                              for computation beyond pe.value())

Output (write-once, refuses if present): data/SRP1/Results/P515S33/E1/e1_zero_solve.json
"""
import json
import os
import re
import sys
from math import sqrt

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RUN = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32_run')
S31C_RUN = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31C_run')
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S33', 'E1')
OUT = os.path.join(OUT_DIR, 'e1_zero_solve.json')

BOUND_ENTRIES_TOL_AT_BOUND = 1e-6
BOUND_ENTRIES_TOL_NEAR = 0.005
TARGET_DELTAS_PU = (0.04, 0.06, 0.09)


def _q1_interface_voltage(g, rows):
    """Q1: interface V vs bounds. Evidence: g_baseline.json (cycle_trajectory,
    fields boyd_v_norm_z and worst_v_primal_*), case9_2025.json (bus table)."""

    # ------------------------------------------------------------------
    # Bounds (production case data; not reconstructed, not solved).
    # ------------------------------------------------------------------
    case9_path = os.path.join(REPO, 'data', 'SRP1', 'case9', 'case9_2025.json')
    case9 = json.load(open(case9_path))
    bounds = {}
    for node in case9['nodes']:
        if node.get('bus_i') in (5, 7, 9):
            bounds[node['bus_i']] = {'v_min_pu': node['Vmin'], 'v_max_pu': node['Vmax'],
                                      'base_kv': node['baseKV']}

    # ------------------------------------------------------------------
    # Per-entry data availability search (explicit, so absence is documented).
    # ------------------------------------------------------------------
    searched = [
        {'artifact': 'g_baseline.json cycle_trajectory', 'found': 'worst_v_primal_* (single '
         'worst-disagreement entry per cycle: node/year/day/period/tso/dso/base/rho) and '
         'aggregate boyd_v_norm_z / boyd_v_norm_x (Euclidean norm over all 864 entries); '
         'no full per-entry array.'},
        {'artifact': 'component_levels_terminal.json blocks[*].unweighted/weighted.voltage_slack',
         'found': 'aggregate weighted voltage-bound-violation slack penalty per TSO block '
         '(near-zero, e.g. -1.09e-1 unweighted / -5.02e1 weighted for TSO|2025|Spring); '
         'confirms no bound violation at terminal but gives no per-entry magnitude or '
         'distance.'},
        {'artifact': 'interface_settlement_detail_s31c.json',
         'found': 'per (kind,node,year,day) P/Q settlement detail only; no voltage field.'},
        {'artifact': 'stdout_baseline.log grep [DIAG]',
         'found': '48 [DIAG][PF MAX] lines, 0 [DIAG][V MAX] lines -- the V-branch print in '
         '_print_worst_primal_residual_diagnostics (shared_resources_planning.py:5790-5804) is '
         'gated on primal V residual exceeding params.tol[\'consensus\'][\'v\'], which held in '
         'only 11/150 cycles per the s32 report (V primal passes from cycle 10); consistent '
         'with 0 hits.'},
        {'artifact': 'stdout_baseline.log grep [VOLTAGE SLACK',
         'found': '0 lines -- the per-node vmag/v_min/v_max transition printer '
         '(_print_tso_voltage_slack_transitions, shared_resources_planning.py:5878) is gated on '
         'abs(voltage_delta) > 0.1*objective_tolerance; observed [SLACK COMPONENTS] voltage '
         'deltas are O(1e-8) to O(1e-13), below that gate on every occurrence checked, so the '
         'branch never fired.'},
        {'artifact': 'esso_models_baseline.pkl', 'found': 'ESSO (shared-storage) models only; '
         'no TSO/DSO node voltages.'},
        {'artifact': 'frozen_snapshots_baseline.jsonl', 'found': '2 records, both cycle-7 '
         'FrozenSMOPF captures (DSO node7 case33_2, TSO case9), not the terminal (cycle-150) '
         'state.'},
        {'artifact': 'results/ directory', 'found': 'empty except results/FrozenSMOPF/ (the '
         'same 2 cycle-7 pickles referenced above); no spreadsheets or per-entry V dumps.'},
    ]
    conclusion = ('No per-entry terminal (cycle-150) TSO/DSO interface voltage array is '
                  'serialized anywhere in the run artifacts. The best available evidence is '
                  '(a) the single worst-primal-residual V entry recorded per cycle, and (b) the '
                  'aggregate Euclidean norm of the TSO interface-V consensus vector (864 '
                  'entries) per cycle.')

    # ------------------------------------------------------------------
    # (a) worst_v_primal envelope across all 150 cycles.
    # ------------------------------------------------------------------
    worst_series = []
    for r in rows:
        if r.get('worst_v_primal_tso') is None:
            continue
        base = r['worst_v_primal_base']
        worst_series.append({
            'cycle': r['cycle'], 'node': r['worst_v_primal_node'], 'year': r['worst_v_primal_year'],
            'day': r['worst_v_primal_day'], 'period': r['worst_v_primal_period'],
            'tso_pu': r['worst_v_primal_tso'] / base, 'dso_pu': r['worst_v_primal_dso'] / base,
        })
    max_entry = max(worst_series, key=lambda e: max(e['tso_pu'], e['dso_pu']))
    min_entry = min(worst_series, key=lambda e: min(e['tso_pu'], e['dso_pu']))
    terminal = worst_series[-1]

    dist_to_vmax = 1.1 - max(max_entry['tso_pu'], max_entry['dso_pu'])
    dist_to_vmin = min(min_entry['tso_pu'], min_entry['dso_pu']) - 0.9
    binding_bound = 'Vmax (1.1 pu)' if dist_to_vmax < dist_to_vmin else 'Vmin (0.9 pu)'

    entries_at_or_near_bound = {
        'computable': False,
        'reason': 'no per-entry array serialized (see searched list); the worst-primal-residual '
        'sample (one entry/cycle, 150 samples, NOT the true per-cycle extremum of the 864-entry '
        'population) never sits within 0.005 pu of either bound: min observed distance to Vmax '
        f'across the sample is {dist_to_vmax:.4f} pu (cycle {max_entry["cycle"]}, node '
        f'{max_entry["node"]}), min observed distance to Vmin is {dist_to_vmin:.4f} pu (cycle '
        f'{min_entry["cycle"]}, node {min_entry["node"]}). This bounds (does not determine) the '
        'true population extremum: it can only be AT LEAST this close (the true worst entry is '
        'unknown and could be closer to a bound than the worst-DISAGREEMENT entry sampled here).',
    }

    # ------------------------------------------------------------------
    # (b) aggregate norm trend -- direction, coherence, rate.
    # ------------------------------------------------------------------
    n_v = round(rows[0]['boyd_v_norm_z'] and (
        # n recovered from eps_dual = sqrt(n)*eps_abs + eps_rel*norm_y (same identity as
        # p515_s32_supplementary.py); avoids hard-coding 864.
        ((rows[0]['boyd_v_eps_dual'] - rows[0]['boyd_eps_rel'] * rows[0]['boyd_v_norm_y'])
         / rows[0]['boyd_eps_abs']) ** 2
    ))

    rms_per_entry = [{'cycle': r['cycle'], 'rms_pu': r['boyd_v_norm_z'] / sqrt(n_v)}
                      for r in rows]
    window_25_150 = [e for e in rms_per_entry if 25 <= e['cycle'] <= 150]
    rms_start, rms_end = window_25_150[0]['rms_pu'], window_25_150[-1]['rms_pu']
    n_steps = window_25_150[-1]['cycle'] - window_25_150[0]['cycle']
    rate_pu_per_cycle = (rms_end - rms_start) / n_steps

    norm_z_deltas = [rows[i]['boyd_v_norm_z'] - rows[i - 1]['boyd_v_norm_z']
                      for i in range(len(rows)) if rows[i]['cycle'] >= 26]
    direction = 'up (rising)' if sum(1 for d in norm_z_deltas if d > 0) == len(norm_z_deltas) \
        else ('mixed' if any(d > 0 for d in norm_z_deltas) else 'down (falling)')
    monotone_fraction = sum(1 for d in norm_z_deltas if d > 0) / len(norm_z_deltas)

    # Coherence(k) = (||z||(k)-||z||(k-1)) / ||dz||(k); ||dz|| recovered from
    # s_proximal_part / gamma (gamma = 1 on every channel per SRP1_params.json), same
    # identity as p515_s32_e0_coherence.py. sqrt(n_active/n) ~= coherence if the moving
    # subset shares the aggregate's typical per-entry magnitude (~1 pu) and the static
    # subset does not move at all.
    gamma = json.load(open(os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')))['admm'][
        'proximal_regularization']['tso']['gamma']
    coherences = []
    for i, r in enumerate(rows):
        if r['cycle'] < 26:
            continue
        dz = r['boyd_v_s_proximal_part'] / gamma['v']
        if dz <= 0:
            continue
        coh = (r['boyd_v_norm_z'] - rows[i - 1]['boyd_v_norm_z']) / dz
        coherences.append(coh)
    coherence_mean = sum(coherences) / len(coherences)
    implied_active_fraction = coherence_mean ** 2

    # ------------------------------------------------------------------
    # (c) cycles-to-bound extrapolation (illustrative; uses the aggregate mean
    # per-entry rate, and a refined estimate assuming only the coherence-implied
    # active fraction actually moves, at correspondingly higher per-entry speed).
    # ------------------------------------------------------------------
    rate_refined = rate_pu_per_cycle / max(implied_active_fraction, 1e-9) \
        if implied_active_fraction > 0 else None
    projection = []
    for delta in TARGET_DELTAS_PU:
        projection.append({
            'delta_pu': delta,
            'cycles_naive_mean_rate': delta / rate_pu_per_cycle,
            'cycles_refined_active_subset_rate': (delta / rate_refined) if rate_refined else None,
        })
    projection.append({
        'delta_pu': round(dist_to_vmax, 6),
        'note': 'distance from the largest worst-primal-residual pu value observed in 150 '
        'cycles (cycle 150) to Vmax; NOT the true population-worst distance (unavailable, see '
        'entries_at_or_near_bound).',
        'cycles_naive_mean_rate': dist_to_vmax / rate_pu_per_cycle,
        'cycles_refined_active_subset_rate': (dist_to_vmax / rate_refined) if rate_refined else None,
    })

    return {
        'bounds_source': {'file': 'data/SRP1/case9/case9_2025.json', 'nodes': bounds},
        'per_entry_data_availability': {'searched': searched, 'conclusion': conclusion},
        'worst_primal_residual_envelope': {
            'source': 'data/SRP1/Results/P515S32_run/g_baseline.json cycle_trajectory '
            '(worst_v_primal_*, all 150 cycles)',
            'n_cycles_sampled': len(worst_series),
            'terminal_cycle_150': terminal,
            'max_pu_entry': max_entry,
            'min_pu_entry': min_entry,
            'distance_to_vmax_pu_at_max_entry': dist_to_vmax,
            'distance_to_vmin_pu_at_min_entry': dist_to_vmin,
            'binding_bound_of_the_two': binding_bound,
            'caveat': 'this envelope is built from the single WORST-DISAGREEMENT (|tso-dso|) '
            'entry per cycle, not the worst-MAGNITUDE entry; it is a lower bound on how close '
            'the true population gets to either limit, not the true minimum distance.',
        },
        'entries_at_or_near_bound': entries_at_or_near_bound,
        'aggregate_norm_trend': {
            'source': 'data/SRP1/Results/P515S32_run/g_baseline.json cycle_trajectory '
            '(boyd_v_norm_z, boyd_v_s_proximal_part); n_entries recovered via the eps_dual '
            'identity, same as p515_s32_supplementary.py / p515_s32_e0_coherence.py',
            'n_entries_v': n_v,
            'direction': direction,
            'monotone_increasing_fraction_cycles_26_150': monotone_fraction,
            'rms_per_entry_pu_cycle_1': rms_per_entry[0]['rms_pu'],
            'rms_per_entry_pu_cycle_25': [e for e in rms_per_entry if e['cycle'] == 25][0]['rms_pu'],
            'rms_per_entry_pu_cycle_150': rms_per_entry[-1]['rms_pu'],
            'mean_rate_pu_per_cycle_cycles_25_150': rate_pu_per_cycle,
            'coherence_mean_cycles_26_150': coherence_mean,
            'implied_active_entry_fraction_sqrt_coherence_sq': implied_active_fraction,
            'implied_active_entry_count': round(implied_active_fraction * n_v),
            'interpretation': 'coherence ~= sqrt(n_active/n) holds if the active subset shares '
            'the aggregate RMS (~1 pu) and the static subset does not move; '
            f'coherence {coherence_mean:.3f} => active fraction {implied_active_fraction:.3f} '
            f'(~{round(implied_active_fraction * n_v)}/{n_v} entries), i.e. roughly half.',
        },
        'cycles_to_bound_projection': {
            'method': 'delta_pu / rate; naive = aggregate mean rate over all entries; refined = '
            'rate scaled by 1/implied_active_fraction, i.e. assumes only the moving subset '
            'accounts for all the norm growth and moves at that higher per-entry speed. '
            'Illustrative only -- extrapolates a 125-cycle-observed linear trend, not a '
            'guaranteed asymptote.',
            'targets': projection,
        },
    }


def _q2_esso_objective_scale():
    """Q2: effective_scale/sigma for TSO/DSO vs ESSO base objective magnitude (D5)."""
    import pickle
    import pyomo.environ as pe

    stdout_path = os.path.join(RUN, 'stdout_baseline.log')
    scale_line = None
    with open(stdout_path) as fh:
        for line in fh:
            if line.startswith('[ADMM OF SCALE] n='):
                scale_line = line.strip()
                break
    m = re.search(r'selected=([0-9.eE+-]+)', scale_line)
    objective_scale = float(m.group(1))

    s31c_scale = None
    s31c_stdout = os.path.join(S31C_RUN, 'stdout_baseline.log')
    if os.path.exists(s31c_stdout):
        with open(s31c_stdout) as fh:
            for line in fh:
                if line.startswith('[ADMM OF SCALE] n='):
                    m2 = re.search(r'selected=([0-9.eE+-]+)', line)
                    s31c_scale = float(m2.group(1))
                    break

    comp = json.load(open(os.path.join(RUN, 'component_levels_terminal.json')))
    blocks = comp['blocks']
    effective_scale_by_block = {}
    for key, b in blocks.items():
        w = b['admm_block_weight']
        effective_scale_by_block[key] = {
            'kind': b['kind'], 'node_id': b.get('node_id'), 'admm_block_weight': w,
            'effective_scale': objective_scale / w,
        }
    scales = [v['effective_scale'] for v in effective_scale_by_block.values()]

    esso_pkl = os.path.join(RUN, 'esso_models_baseline.pkl')
    with open(esso_pkl, 'rb') as fh:
        esso_models = pickle.load(fh)

    esso_terminal = {}
    for node_id, m_ in esso_models.items():
        base = pe.value(m_.objective.expr)
        admm = pe.value(m_.admm_objective.expr) if hasattr(m_, 'admm_objective') else None
        rho = pe.value(m_.rho) if hasattr(m_, 'rho') else None
        feas_pen = pe.value(m_.feasibility_penalty) if hasattr(m_, 'feasibility_penalty') else None
        max_gap = 0.0
        max_dual = 0.0
        for y in m_.years:
            for d in m_.days:
                for p in m_.periods:
                    gp = abs(pe.value(m_.es_pnet[y, d, p]) - pe.value(m_.p_req[y, d, p]))
                    gq = abs(pe.value(m_.es_qnet[y, d, p]) - pe.value(m_.q_req[y, d, p]))
                    max_gap = max(max_gap, gp, gq)
                    max_dual = max(max_dual, abs(pe.value(m_.dual_p_req[y, d, p])),
                                    abs(pe.value(m_.dual_q_req[y, d, p])))
        esso_terminal[str(node_id)] = {
            'objective_expr_base': base,
            'admm_objective_expr': admm,
            'al_contribution_admm_minus_base': (admm - base) if admm is not None else None,
            'feasibility_penalty_component': feas_pen,
            'rho_esso': rho,
            'max_abs_pnet_minus_p_req_pu': max_gap,
            'max_abs_dual_p_or_q_req': max_dual,
            'if_divided_by_mean_effective_scale': base / (sum(scales) / len(scales)),
        }

    return {
        'objective_scale_selected': {
            'value': objective_scale,
            'source': f'{os.path.relpath(stdout_path, REPO)}:1 line "{scale_line}" '
            '(printed once, before the ADMM loop starts, by '
            '_compute_common_admm_objective_scale, shared_resources_planning.py:3049-3122, '
            'called once at shared_resources_planning.py:2412)',
            'definition': 'max over 48 TSO/DSO (year,day) blocks of |admm_block_weight * '
            'raw_pre_ADMM_objective|; computed once for the whole run, before ADMM starts, '
            'from an initial (non-augmented) solve.',
            's31c_same_instance_cross_check': {
                'value': s31c_scale,
                'source': f'{os.path.relpath(s31c_stdout, REPO)}:1 (P5.15 s31c gate run, same '
                'C* instance) -- included only as an independent-run cross-check, not part of '
                'the s32 baseline this E1 task audits.',
                'identical_to_s32': (s31c_scale == objective_scale) if s31c_scale is not None else None,
            },
        },
        'effective_scale_by_block': {
            'source': 'reconstructed: objective_scale (above) / admm_block_weight '
            '(data/SRP1/Results/P515S32_run/component_levels_terminal.json blocks[*].'
            'admm_block_weight); NOT stored as a field itself in component_levels_terminal.json. '
            'Formula per shared_resources_planning.py:3978-3983 (TSO) and :4130-4136 (DSO).',
            'min': min(scales), 'max': max(scales), 'n_blocks': len(scales),
            'by_block': effective_scale_by_block,
        },
        'esso_no_division_by_sigma': {
            'source': 'update_shared_energy_storage_model_to_admm, shared_resources_planning.py'
            ':4209 ("obj = copy(models[node_id].objective.expr)", no /effective_scale term, no '
            'admm_objective_scale Param on the ESSO model -- contrast with TSO '
            'shared_resources_planning.py:3982-3983 and DSO shared_resources_planning.py:'
            '4133-4136).',
        },
        'esso_objective_terminal': {
            'source': 'data/SRP1/Results/P515S32_run/esso_models_baseline.pkl, unpickled and '
            'read via pe.value() only (no solve); base objective = model.objective.expr = '
            'model.feasibility_penalty (PENALTY_ESSO_SLACK=1e3 * slack_es_pnet_{up,down} sum + '
            'EPS_ESSO_THROUGHPUT=1e-5 * throughput sum, shared_energy_storage_data.py:770-795; '
            'both constants defined in definitions.py:63,74). No degradation/investment cost '
            'term exists in this objective (investments are fixed parameters per '
            'PLANNER_BRIEF_2026-09-13.md governing decisions).',
            'by_node': esso_terminal,
        },
        'implied_relative_weighting': {
            'statement': 'the factor by which ESSO terms would shrink if divided by the same '
            'sigma the network blocks use is sigma itself: effective_scale ranges '
            f'{min(scales):.3e} to {max(scales):.3e} across the 48 TSO/DSO blocks. Applied to '
            'the ESSO base objective (~-5.72e-3 at every node, see esso_objective_terminal), '
            'the divided value would be on the order of -2.5e-8 -- 6 orders of magnitude '
            'smaller than it already is.',
            'observation_not_yet_interpretation': 'the ESSO base objective is not a genuine '
            'economic cost signal at this revision (see esso_no_division_by_sigma / '
            'esso_objective_terminal): it is dominated by IPOPT bound-multiplier tolerance on '
            'slack_es_pnet_{up,down} sitting at ~-1e-8 each times PENALTY_ESSO_SLACK=1e3 times '
            '~576 slack entries, i.e. a numerical floor artifact, not a priced degradation or '
            'investment term. Both the base objective (~-5.7e-3) and the AL contribution '
            '(admm_objective - objective, ~-1.1e-6) are far below the O(1e2-1e3) scale TSO/DSO '
            'blocks reach after dividing by their own effective_scale.',
            'conclusion_for_planner': 'D5 is a real structural asymmetry (ESSO is not divided by '
            'sigma, TSO/DSO are), but at this terminal iterate it does not translate into the '
            'ESSO base objective distorting or dominating the AL/consensus penalty inside the '
            'ESSO subproblem, because the base objective term is itself negligible (no economic '
            'signal to inflate or protect). Whether this held throughout the run, or only near '
            'the terminal iterate, is not established by this E1 check (single terminal '
            'snapshot only).',
        },
    }


def _q3a_ess_tso_prev_vs_proximal_centre(g, rows):
    """Q3(a): consensus_vars['ess']['tso']['prev'] update site vs
    _update_tso_proximal_centres_after_solve's ESS centre. Code-only (file:line), with
    magnitude corroboration from g_baseline.json."""

    rms_step_sampled = []
    gamma = json.load(open(os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')))['admm'][
        'proximal_regularization']['tso']['gamma']
    for r in rows:
        if r['cycle'] not in (1, 10, 25, 50, 75, 100, 125, 150):
            continue
        eps_dual = r['boyd_ess_eps_dual']
        norm_y = r['boyd_ess_norm_y']
        eps_abs = rows[0]['boyd_eps_abs']
        eps_rel = rows[0]['boyd_eps_rel']
        sqrt_n = (eps_dual - eps_rel * norm_y) / eps_abs
        s_prox = r['boyd_ess_s_proximal_part']
        rms_step_sampled.append({
            'cycle': r['cycle'],
            'rms_step_per_entry': (s_prox / (gamma['ess'] * sqrt_n)) if sqrt_n > 0 else None,
        })

    return {
        'consensus_vars_ess_tso_prev_write_site': {
            'file_line': 'shared_resources_planning.py:6394-6395, inside '
            '_update_shared_energy_storage_variables (parameter name shared_ess_vars there is '
            'bound to consensus_vars[\'ess\'] by the caller, shared_resources_planning.py:206: '
            '"self.update_shared_energy_storage_variables(..., consensus_vars[\'ess\'], ...)")',
            'code': "shared_ess_vars['tso']['prev'][node_id][year][day]['p'][p] = "
            "copy(shared_ess_vars['tso']['current'][node_id][year][day]['p'][p])  # then "
            "['current'][...] is overwritten with the just-solved TSO value on the next two "
            "lines, gated on _solver_result_succeeded(results['tso'][year][day])",
            'call_site': 'invoked via update_and_check_convergence(update_tn=True), '
            'shared_resources_planning.py:2496-2503, inside the main ADMM loop '
            '(shared_resources_planning.py:2448 onward), immediately after the TSO solve '
            'block and _update_tso_proximal_centres_after_solve.',
        },
        'tso_proximal_centre_write_site': {
            'file_line': 'shared_resources_planning.py:4419-4420, inside '
            '_update_tso_proximal_centres_after_solve',
            'code': 'local_model.prox_ess_p_prev[e, p].set_value(current_ess_p); '
            'local_model.prox_ess_q_prev[e, p].set_value(current_ess_q)  # "Successful solution '
            'becomes next proximal centre." (comment at line 4418); gated on the same '
            'per-block success check as the surrounding loop (shared_resources_planning.py:4255 '
            '"A failed TSO block keeps its previous proximal centre.")',
            'call_site': 'shared_resources_planning.py:2495, called immediately after the TSO '
            'solve at line 2481-2489, and BEFORE the consensus_vars update at line 2496-2503.',
        },
        'cycle_ordering_within_one_ADMM_cycle_k': [
            '1. DSO solve (uses consensus_vars[\'ess\'] as of end of cycle k-1).',
            '2. update_and_check_convergence(update_dns=True): updates DSO current/prev only.',
            '3. TSO solve (line 2481-2489): the TSO model\'s own proximal penalty uses '
            'prox_ess_p_prev/q_prev, which since the end of cycle k-1 holds the cycle (k-1) TSO '
            'solution -- this IS the "centre" for this solve.',
            '4. _update_tso_proximal_centres_after_solve (line 2495): prox_ess_p_prev/q_prev <- '
            'cycle-k TSO solution (becomes the centre for cycle k+1).',
            '5. update_and_check_convergence(update_tn=True) (line 2496-2503): consensus_vars'
            '[\'ess\'][\'tso\'][\'prev\'] <- OLD current (= cycle k-1 TSO solution, i.e. exactly '
            'the centre used in step 3); consensus_vars[\'ess\'][\'tso\'][\'current\'] <- cycle-k '
            'TSO solution.',
            '6. ESSO solve + update_and_check_convergence(update_sess=True).',
            '7. get_admm_boyd_residual_metrics (line 2534) is called AFTER all of the above, so '
            'it reads consensus_vars[\'ess\'][\'tso\'][\'current\'] = x_TSO^k and '
            '[\'prev\'] = x_TSO^{k-1} = the centre that step 3 regularized against.',
        ],
        'answer': 'No divergence found at cycle-k granularity, under normal operation (TSO '
        'solve succeeds). Both trackers are written in the same call (_update_tso_proximal_'
        'centres_after_solve at step 4, then the consensus_vars copy at step 5), in the same '
        'order, gated on the identical per-block success flag. Because get_admm_boyd_residual_'
        'metrics is called only after the full cycle-k update sequence (step 7), '
        'consensus_vars[\'ess\'][\'tso\'][\'prev\'] at read time equals x_TSO^{k-1}, the exact '
        'value that was the proximal centre for cycle k\'s TSO solve (prox_ess_p_prev/q_prev '
        'before step 4 overwrote it). The Boyd quantity gamma*a*(x_TSO^k - x_TSO^{k-1}) computed '
        'at shared_resources_planning.py:5706-5712 is therefore numerically identical to the '
        'actual proximal displacement (x_TSO^k - centre), to floating-point precision -- not an '
        'approximation of it under a different convention.',
        'corroborating_magnitude_check': {
            'source': 'data/SRP1/Results/P515S32_run/g_baseline.json, boyd_ess_s_proximal_part '
            '(sampled cycles), same identity p515_s32_supplementary.py uses '
            '(rms_step_per_entry = s_proximal_part / (gamma_ess * sqrt(n)))',
            'rms_step_per_entry_sampled': rms_step_sampled,
            'interpretation': 'values are O(1e-5), i.e. small per-cycle displacements, not O(1) '
            'absolute ESS-consensus levels (which the pu-normalized ESS coordinate would reach '
            'if x_TSO_prev were stuck at a stale/zero value from a broken tracker); this is '
            'consistent with (not proof of) the code-level finding that the tracker is updated '
            'correctly every cycle.',
        },
        'caveat': 'this reconciles the DIAGNOSTIC (Boyd s) computation with the ACTUAL TSO-'
        'internal proximal regularization; it does not by itself explain the ESS drift dynamics '
        '(see P5_15_S32_BOYD_GATE_REPORT.md hypothesis table, row (d) vs F5) -- it rules out '
        '"the diagnostic is reading a stale/mismatched centre" as a confound.',
    }


def _q3b_pf_dual_units():
    """Q3(b): PF ||y|| units vs the DSO augmented-Lagrangian multiplier's actual model units."""
    return {
        'dso_al_constraint_definition': {
            'file_line': 'shared_resources_planning.py:4145,4156-4159',
            'code': 'interface_transf_rating = get_interface_branch_rating() / s_base  # DSO '
            'own p.u.\nconstraint_p_req = (expected_interface_pf_p[p] - p_pf_req[p]) / '
            'interface_transf_rating\nobj += dual_pf_p_req[p] * constraint_p_req + '
            '(rho_pf/2) * constraint_p_req**2',
            'note': 'constraint_p_req is dimensionless, normalized by the INTERFACE RATING '
            '(150 or 200 MVA in this instance), expressed via the DSO p.u. base -- not by '
            's_base (100 MVA) alone.',
        },
        'dual_pf_p_req_param_set_site': {
            'file_line': 'shared_resources_planning.py:4980-4981 (sequential DSO update path)',
            'code': "dual_pf_p_req[p].set_value(dual_pf['current'][node_id][year][day]['p'][p] "
            "/ s_base)  # s_base = DSO's own baseMVA (100 in this instance)",
        },
        'dual_vars_pf_dso_accumulation_site': {
            'file_line': 'shared_resources_planning.py:6348-6351',
            'code': "error_p_pf_req_dso = interface_vars['pf']['dso']['current'][...] - "
            "interface_vars['pf']['tso']['current'][...]  # both stored in MW\n"
            "dual_vars['pf']['dso']['current'][...] += rho_pf_dso * error_p_pf_req_dso / "
            "interface_rating * dso_s_base",
            'note': 'interface_vars[\'pf\'][\'dso\'/\'tso\'][\'current\'] are stored in MW '
            '(shared_resources_planning.py:6312-6316: pe.value(expected_interface_pf_p) * '
            's_base), so error_p_pf_req_dso is in MW and error/interface_rating is exactly '
            'constraint_p_req.',
        },
        'boyd_y_pf_site': {
            'file_line': 'shared_resources_planning.py:5667',
            'code': 'y_pf = lambda_dso_pf / s_base_dso  # s_base_dso = dso_network.baseMVA',
        },
        'algebraic_trace': [
            '1. update increment to dual_vars[...]["current"] (raw, MW-scale storage) = '
            'rho_pf * (error_MW/rating_MW) * s_base = rho_pf * constraint_p_req * s_base.',
            '2. dual_pf_p_req[p] (the Param actually multiplying constraint_p_req inside the '
            'DSO objective) = (accumulated raw dual_vars value) / s_base = accumulated sum of '
            'rho_pf * constraint_p_req increments -- the s_base factor introduced in step 1 is '
            'exactly cancelled by the /s_base in the Param set-value call. dual_pf_p_req is '
            'therefore the correct dual-ascent multiplier conjugate to constraint_p_req '
            '(rating-normalized), with NO leftover s_base or rating factor.',
            '3. Boyd\'s y_pf = lambda_dso_pf / s_base_dso applies the IDENTICAL divisor as step '
            '2 to the SAME raw stored value, so y_pf = dual_pf_p_req[p] exactly -- the value '
            'actually used inside the DSO\'s own augmented-Lagrangian term.',
            '4. r_pf (Boyd primal residual) = (x_dso_pf - z_tso_pf) / interface_rating (MW/MW) '
            '= constraint_p_req itself (same rating-normalization, confirmed by comparing units '
            'in step 1). s_pf (Boyd dual residual) uses dz_pf / interface_rating, same '
            'normalization again.',
        ],
        'answer': 'No unit mismatch. y_pf, r_pf and s_pf are all in the SAME rating-normalized '
        'units: the two s_base multiplications/divisions (the *dso_s_base in the accumulation '
        'step and the /s_base in both the Param set-value call and Boyd\'s y_pf) are a '
        'self-cancelling round trip through an intermediate MW-scale storage convention, not an '
        'independent second normalization. y_pf reproduces dual_pf_p_req[p] -- the multiplier '
        'the DSO subproblem actually optimizes against -- to floating-point precision.',
        'impact_on_eps_dual': 'none (no correction needed); this corroborates, at the algebra '
        'level, the s32-report\'s independent "mapping diagnostic" finding ("The V/PF Boyd '
        'mapping is consistent with the dual update to machine precision", '
        'WORKER_REPORT_S32_MAPPING_DIAG.md / p515_s32_mapping_diag.py) rather than adding a new '
        'discrepancy.',
        'caveat': 'this is a static code trace (file:line + algebra), not a numeric replay of '
        'cycle-150 values against an independently rebuilt DSO model; the existing mapping '
        'diagnostic already did the numeric replay and is cited above as corroboration.',
    }


def main():
    if os.path.exists(OUT):
        raise RuntimeError(f'refusing to overwrite {OUT}')
    if os.path.isdir(OUT_DIR) and os.listdir(OUT_DIR):
        raise RuntimeError(f'refusing to write into non-empty existing directory {OUT_DIR}')
    os.makedirs(OUT_DIR, exist_ok=True)

    guard = SolveProfileGuard(permitted=(), label='P5.15 s33 E1 zero-solve').install()
    try:
        g = json.load(open(os.path.join(RUN, 'g_baseline.json')))
        rows = g['cycle_trajectory']

        q1 = _q1_interface_voltage(g, rows)
        q2 = _q2_esso_objective_scale()
        q3a = _q3a_ess_tso_prev_vs_proximal_centre(g, rows)
        q3b = _q3b_pf_dual_units()

        out = {
            'stage': 'P5.15 Step 3.3(a) Addendum 14 -- E1 zero-solve diagnostic',
            'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 14; '
                         'P5_15_S32_BOYD_GATE_REPORT.md sections 3 and 5 item 1',
            'instance': g.get('instance'),
            'source_run_dir': os.path.relpath(RUN, REPO),
            'q1_interface_voltage_vs_bounds': q1,
            'q2_esso_objective_scale_vs_sigma': q2,
            'q3a_ess_tso_prev_vs_proximal_centre': q3a,
            'q3b_pf_dual_units': q3b,
        }
    finally:
        guard.uninstall()

    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)

    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)

    print(f'Wrote {OUT}')
    print('guard counts:', guard.counts)
    print()
    print('Q1 binding bound:', q1['worst_primal_residual_envelope']['binding_bound_of_the_two'],
          'min distance', round(min(q1['worst_primal_residual_envelope']['distance_to_vmax_pu_at_max_entry'],
                                     q1['worst_primal_residual_envelope']['distance_to_vmin_pu_at_min_entry']), 4), 'pu')
    print('Q1 aggregate direction:', q1['aggregate_norm_trend']['direction'],
          'rate', q1['aggregate_norm_trend']['mean_rate_pu_per_cycle_cycles_25_150'], 'pu/cycle')
    print('Q2 objective_scale', q2['objective_scale_selected']['value'],
          'effective_scale range', q2['effective_scale_by_block']['min'], '-',
          q2['effective_scale_by_block']['max'])
    print('Q2 esso base objective (node 5)',
          q2['esso_objective_terminal']['by_node']['5']['objective_expr_base'])
    print('Q3a answer:', q3a['answer'][:120], '...')
    print('Q3b answer:', q3b['answer'][:120], '...')
    return 0


if __name__ == '__main__':
    sys.exit(main())
