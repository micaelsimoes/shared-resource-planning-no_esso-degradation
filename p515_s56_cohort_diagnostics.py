"""P5.15 S56 -- detectors for the ESSO storage-cohort formulation (arrangement A, `cohort_formulation`
'independent_units'; S56_COHORT_FORMULATION_PROPOSAL.md, Expert rulings E2/E5/E6 in TASKS.md S56).

Read-only: every function reads the VALUES of a built (normally solved) ESSO node model and returns plain dicts. Nothing
is solved, fixed or modified. Works on 'legacy_h3' models too (they carry no per-cohort SoC: the SoC is then IMPLIED,
integrated from the per-cohort powers by the same recursion, `implied_cohort_soc`).

Units: MW / MVAr / MWh (the ESSO's), dt = `model_construction_helpers.period_duration_hours(model)`.

DEFINITIONS (one (block y, day d) of one node; c = cohort y_inv active in y; t = period):
  (i)   within-cohort simultaneity, per cohort:  W_c = sum_t min(pch_c,t, pdch_c,t) * dt   [MWh]
                                                 and max_t min(pch_c,t, pdch_c,t)          [MW]
  (ii)  inter-cohort gross transfer:             X = sum_t min(sum_c pch_c,t, sum_c pdch_c,t) * dt   [MWh]
        conversion loss:                         L = (1/eta_dch - eta_ch) * X   [MWh]
        (X counts the within-cohort simultaneity too; X - sum_c W_c is the strictly inter-cohort part, reported as
        `inter_cohort_transfer_net_of_within_mwh`, a lower bound on the inter-cohort part when both occur.)
        eta: the common efficiencies of the active cohorts; if they differ, the mean is used and flagged.
  (iii) per cohort: energy-balance residual  max_t |e_t - e_{t-1} - (eta_ch pch_t - pdch_t / eta_dch) dt|,
                    e_{-1} = INIT * E_av (the row-3 recursion; on SoC values when the model has them);
        window margin   min_t min(e_t - MIN * E_av, MAX * E_av - e_t)    [MWh; < 0 = violation]
        circle margin   min_t (s_c - sqrt((pch - pdch)^2 + q^2))          [MVA; < 0 = violation]
        closure residual e_T - INIT * E_av                                [MWh] (and the closure slacks)
  (iv)  ESSO slacks: P pair `slack_es_pnet_{up,down}`, Q pair `slack_es_qnet_{up,down}` (when built): max and sum.
  energy accounting (per (y, d), from values): grid energy into the cohorts G = sum_c sum_t (pch - pdch) dt;
        losses Lc = sum_c sum_t ((1 - eta_ch) pch + (1/eta_dch - 1) pdch) dt; stored change S = sum_c (e_T - e_0^-)
        with e_0^- = INIT * E_av; residual G - Lc - S.
"""
from math import sqrt

import pyomo.environ as pe

from definitions import ENERGY_STORAGE_MAX_ENERGY_STORED, ENERGY_STORAGE_MIN_ENERGY_STORED, \
    ENERGY_STORAGE_RELATIVE_INIT_SOC
from model_construction_helpers import period_duration_hours


def _v(x):
    return float(pe.value(x))


def active_cohorts(model, y):
    """Cohorts active in block y by the ESSO's own test (non-zero investment and inside the calendar-life window)."""
    out = []
    for y_inv in model.years:
        if model._esso_cohort_inactive.get(y_inv, False):
            continue
        if model.es_s_rated_per_unit[y_inv, y].fixed or model.es_e_rated_per_unit[y_inv, y].fixed:
            continue
        out.append(y_inv)
    return out


def cohort_efficiencies(shared_ess_data, node_id, y_inv):
    years = list(shared_ess_data.years)
    idx = shared_ess_data.get_shared_energy_storage_idx(node_id)
    ses = shared_ess_data.shared_energy_storages[years[y_inv]][idx]
    return ses.eff_ch, ses.eff_dch


def implied_cohort_soc(model, shared_ess_data, node_id, y_inv, y, d):
    """The per-cohort stored energy integrated from the cohort's power values by the row-3 recursion, starting from
    INIT * E_av (for legacy models, which have no SoC variable)."""
    dt = period_duration_hours(model)
    eta_ch, eta_dch = cohort_efficiencies(shared_ess_data, node_id, y_inv)
    e = ENERGY_STORAGE_RELATIVE_INIT_SOC * _v(model.es_e_available_per_unit[y_inv, y])
    out = []
    for p in model.periods:
        e += (eta_ch * _v(model.es_pch_per_unit[y_inv, y, d, p]) * dt
              - _v(model.es_pdch_per_unit[y_inv, y, d, p]) * dt / eta_dch)
        out.append(e)
    return out


def cohort_block_day(model, shared_ess_data, node_id, y, d):
    """Detectors (i)-(iii) and the energy accounting for one (block y, day d)."""
    dt = period_duration_hours(model)
    cohorts = active_cohorts(model, y)
    has_soc = hasattr(model, 'es_soc_per_unit')
    has_q = hasattr(model, 'es_q_per_unit')
    per = {}
    effs = {}
    grid = 0.0
    losses = 0.0
    stored = 0.0
    sum_ch = [0.0 for _ in model.periods]
    sum_dch = [0.0 for _ in model.periods]
    for y_inv in cohorts:
        eta_ch, eta_dch = cohort_efficiencies(shared_ess_data, node_id, y_inv)
        effs[y_inv] = (eta_ch, eta_dch)
        e_av = _v(model.es_e_available_per_unit[y_inv, y])
        s_c = _v(model.es_s_rated_per_unit[y_inv, y])
        soc_on = has_soc and not model.es_soc_per_unit[y_inv, y, d, model.periods[0]].fixed
        soc = ([_v(model.es_soc_per_unit[y_inv, y, d, p]) for p in model.periods] if soc_on
               else implied_cohort_soc(model, shared_ess_data, node_id, y_inv, y, d))
        e_prev = ENERGY_STORAGE_RELATIVE_INIT_SOC * e_av
        within = 0.0
        within_max = 0.0
        bal_res = 0.0
        win = float('inf')
        circ = float('inf')
        thr = 0.0
        for k, p in enumerate(model.periods):
            pch = _v(model.es_pch_per_unit[y_inv, y, d, p])
            pdch = _v(model.es_pdch_per_unit[y_inv, y, d, p])
            q = _v(model.es_q_per_unit[y_inv, y, d, p]) if has_q else 0.0
            sum_ch[k] += pch
            sum_dch[k] += pdch
            m = min(pch, pdch)
            within += m * dt
            within_max = max(within_max, m)
            bal_res = max(bal_res, abs(soc[k] - e_prev - (eta_ch * pch * dt - pdch * dt / eta_dch)))
            e_prev = soc[k]
            win = min(win, soc[k] - ENERGY_STORAGE_MIN_ENERGY_STORED * e_av, ENERGY_STORAGE_MAX_ENERGY_STORED * e_av - soc[k])
            circ = min(circ, s_c - sqrt((pch - pdch) ** 2 + q ** 2))
            grid += (pch - pdch) * dt
            losses += ((1.0 - eta_ch) * pch + (1.0 / eta_dch - 1.0) * pdch) * dt
            thr += (eta_ch * pch + pdch / eta_dch) * dt
        e0 = ENERGY_STORAGE_RELATIVE_INIT_SOC * e_av
        stored += soc[-1] - e0
        rec = {'e_available_mwh': e_av, 's_rated_mva': s_c, 'soc_source': 'variable' if soc_on else 'implied',
               'within_cohort_simultaneity_mwh': within, 'within_cohort_simultaneity_max_mw': within_max,
               'energy_balance_residual_max_mwh': bal_res, 'window_margin_min_mwh': win,
               'window_fraction_min': min(soc) / e_av if e_av > 0 else None,
               'window_fraction_max': max(soc) / e_av if e_av > 0 else None,
               'circle_margin_min_mva': circ, 'closure_residual_mwh': soc[-1] - e0,
               'throughput_stored_side_mwh': thr, 'soc_mwh': soc}
        if soc_on and hasattr(model, 'slack_es_soc_final_per_unit_up'):
            rec['closure_slack_up_mwh'] = _v(model.slack_es_soc_final_per_unit_up[y_inv, y, d])
            rec['closure_slack_down_mwh'] = _v(model.slack_es_soc_final_per_unit_down[y_inv, y, d])
        per[str(y_inv)] = rec
    eff_set = sorted(set(effs.values()))
    if eff_set:
        eta_ch = sum(e[0] for e in effs.values()) / len(effs)
        eta_dch = sum(e[1] for e in effs.values()) / len(effs)
    else:
        eta_ch = eta_dch = float('nan')
    transfer = sum(min(a, b) for a, b in zip(sum_ch, sum_dch)) * dt
    within_total = sum(r['within_cohort_simultaneity_mwh'] for r in per.values())
    return {'cohorts': [str(c) for c in cohorts], 'n_active': len(cohorts), 'per_cohort': per,
            'efficiencies_common': len(eff_set) <= 1, 'eta_ch': eta_ch, 'eta_dch': eta_dch,
            'inter_cohort_gross_transfer_mwh': transfer,
            'inter_cohort_transfer_net_of_within_mwh': transfer - within_total,
            'conversion_loss_mwh': (1.0 / eta_dch - eta_ch) * transfer if cohorts else 0.0,
            'transfer_hourly_max_mw': max((min(a, b) for a, b in zip(sum_ch, sum_dch)), default=0.0),
            'energy_accounting': {'grid_into_cohorts_mwh': grid, 'conversion_losses_mwh': losses,
                                  'stored_change_mwh': stored, 'residual_mwh': grid - losses - stored}}


def esso_slacks(model):
    """(iv): the P and Q ESSO slack pairs (MW / MVAr), max and sum over (y, d, p)."""
    out = {}
    for label, up, down in (('p', 'slack_es_pnet_up', 'slack_es_pnet_down'), ('q', 'slack_es_qnet_up', 'slack_es_qnet_down')):
        if not hasattr(model, up):
            out[label] = None
            continue
        vals = [(_v(getattr(model, up)[i]), _v(getattr(model, down)[i])) for i in getattr(model, up).index_set()]
        out[label] = {'max_abs': max(max(a, b) for a, b in vals), 'sum': sum(a + b for a, b in vals),
                      'max_net_abs': max(abs(a - b) for a, b in vals), 'n': len(vals)}
    return out


def slack_at(model, y, d, p):
    out = {'p_up': _v(model.slack_es_pnet_up[y, d, p]), 'p_down': _v(model.slack_es_pnet_down[y, d, p])}
    if hasattr(model, 'slack_es_qnet_up'):
        out['q_up'] = _v(model.slack_es_qnet_up[y, d, p])
        out['q_down'] = _v(model.slack_es_qnet_down[y, d, p])
    return out


def cohort_degradation(model, shared_ess_data, node_id):
    """Per (cohort, block) inside the lifetime: D as solved vs D recomputed from the cohort's OWN power values by the
    degradation law (`energy_storage_capacity_degradation`): D = 365 * n * avg / (2 * cl_eff * E_rated), avg =
    sum_d (num_days/365) sum_p (eta_ch pch + pdch / eta_dch) dt; n = the installation block's width (as the law)."""
    years = list(shared_ess_data.years)
    days = list(shared_ess_data.days)
    idx = shared_ess_data.get_shared_energy_storage_idx(node_id)
    dt = period_duration_hours(model)
    out = {}
    for y_inv in model.years:
        if model._esso_cohort_inactive.get(y_inv, False):
            continue
        ses = shared_ess_data.shared_energy_storages[years[y_inv]][idx]
        n = shared_ess_data.years[years[y_inv]]
        e_rated = _v(model.es_e_investment_fixed[y_inv])
        for y in model.years:
            if model.es_D_per_unit[y_inv, y].fixed and model.es_s_rated_per_unit[y_inv, y].fixed:
                continue
            avg = 0.0
            for d, day in enumerate(days):
                w = shared_ess_data.days[day] / 365.00
                for p in model.periods:
                    avg += w * (ses.eff_ch * _v(model.es_pch_per_unit[y_inv, y, d, p]) * dt
                                + _v(model.es_pdch_per_unit[y_inv, y, d, p]) * dt / ses.eff_dch)
            d_own = 365.00 * n * avg / (2 * ses.cl_eff * e_rated)
            d_solved = _v(model.es_D_per_unit[y_inv, y])
            out[f'{years[y_inv]}|{years[y]}'] = {
                'D_solved': d_solved, 'D_from_own_throughput': d_own, 'abs_diff': abs(d_solved - d_own),
                'avg_throughput_mwh_per_day': avg, 'efc_per_day': avg / (2 * e_rated) if e_rated else None,
                'soh_cumul': _v(model.es_soh_per_unit_cumul[y_inv, y]),
                'e_available_mwh': _v(model.es_e_available_per_unit[y_inv, y])}
    return out


def node_report(model, shared_ess_data, node_id, blocks=None):
    """All detectors for one node model: every (y, d) with >= 1 active cohort (or only `blocks` = [(y, d), ...])."""
    out = {'slacks': esso_slacks(model), 'block_day': {}, 'degradation': cohort_degradation(model, shared_ess_data, node_id)}
    years = list(shared_ess_data.years)
    days = list(shared_ess_data.days)
    pairs = blocks if blocks is not None else [(y, d) for y in model.years for d in model.days]
    for y, d in pairs:
        if not active_cohorts(model, y):
            continue
        out['block_day'][f'{years[y]}|{days[d]}'] = cohort_block_day(model, shared_ess_data, node_id, y, d)
    bd = [v for v in out['block_day'].values() if v['n_active'] >= 2]
    pc = [r for v in bd for r in v['per_cohort'].values()]
    out['summary_multi_cohort'] = {
        'n_block_days': len(bd),
        'within_cohort_simultaneity_max_mwh': max((r['within_cohort_simultaneity_mwh'] for r in pc), default=None),
        'within_cohort_simultaneity_max_mw': max((r['within_cohort_simultaneity_max_mw'] for r in pc), default=None),
        'inter_cohort_gross_transfer_max_mwh': max((v['inter_cohort_gross_transfer_mwh'] for v in bd), default=None),
        'energy_balance_residual_max_mwh': max((r['energy_balance_residual_max_mwh'] for r in pc), default=None),
        'window_margin_min_mwh': min((r['window_margin_min_mwh'] for r in pc), default=None),
        'circle_margin_min_mva': min((r['circle_margin_min_mva'] for r in pc), default=None),
        'closure_residual_max_abs_mwh': max((abs(r['closure_residual_mwh']) for r in pc), default=None),
        'energy_accounting_residual_max_abs_mwh': max((abs(v['energy_accounting']['residual_mwh']) for v in bd), default=None),
    }
    return out
