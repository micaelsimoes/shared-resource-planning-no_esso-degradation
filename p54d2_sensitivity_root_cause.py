"""
Stage P5.4-D2 -- shared-ESS capacity-sensitivity root-cause audit.

Sections
  D2.2  inventory of every S / E dependence, classified A/B/C/D
  D2.3  capacity-dependent variable-bound audit incl. IPOPT zL/zU
  D2.4  envelope-theorem decomposition of dQ/dS and dQ/dE
  D2.5  controlled central finite differences for S and E
  D2.9  active-set / local-branch stability for every FD pair

Sign convention (measured, not assumed -- see p54d2_sign_calibration.py):
    dual[param == var]  = +dQ/dparam
    dQ/d(upper bound)   = +zU
    dQ/d(lower bound)   = +zL

so the complete parametric derivative is

    dQ/dtheta = dual[fixing row]
              + sum over capacity-dependent bounds of ( zU*du/dtheta + zL*dl/dtheta )
              + direct-expression contributions

while the production Benders extraction uses ONLY the first term.

    python p54d2_sensitivity_root_cause.py [--case dso9_2030_Winter] [--quick]
"""

import argparse
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from definitions import (ENERGY_STORAGE_MAX_ENERGY_STORED,  # noqa: E402
                         ENERGY_STORAGE_MIN_ENERGY_STORED,
                         ENERGY_STORAGE_RELATIVE_INIT_SOC,
                         ESS_COMPLEMENTARITY_TOLERANCE,
                         SHARED_ESS_ZERO_CAPACITY_TOLERANCE)
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D2')

# every shared-ESS operational/state variable, for the D2.3 bound audit
ESS_VARIABLES = ('shared_es_pch', 'shared_es_pdch', 'shared_es_pch_hat',
                 'shared_es_pdch_hat', 'shared_es_pnet', 'shared_es_qnet',
                 'shared_es_soc', 'slack_shared_es_soc_final_up',
                 'slack_shared_es_soc_final_down')

# rows whose activity is tracked for the D2.9 active-set comparison
TRACKED_ROWS = ('sess_converter_capability', 'sess_active_sum_limit',
                'sess_comp', 'sess_soc_limit_upper', 'sess_soc_limit_lower')

FD_STEPS = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def quiet(fn, *a, **k):
    console = io.StringIO()
    with redirect_stdout(console):
        out = fn(*a, **k)
    return out, console.getvalue()


# ---------------------------------------------------------------------------
#  Model helpers
# ---------------------------------------------------------------------------
def ess_entries(model, name, e):
    if not hasattr(model, name):
        return []
    comp = getattr(model, name)
    return list(mch._component_entries_for_shared_ess(comp, e))


def solve_at(network, params, e, s_pu, e_pu):
    """Build a fresh production model, configure capacity exactly as production
    does, and solve it through the real production path."""
    def _build():
        m = network.build_model(params)
        m.shared_es_s_rated_fixed[e].set_value(s_pu)
        m.shared_es_e_rated_fixed[e].set_value(e_pu)
        mch.configure_shared_ess_operational_state(m, e, s_pu, e_pu)
        r = network.run_smopf(m, params, print_header=False)
        return m, r
    (m, r), _log = quiet(_build)
    ok = bool(srp._solver_result_succeeded(r))
    return m, r, ok


def objective_of(model):
    return float(pe.value(model.objective))


def bound_multiplier_terms(model, e):
    """D2.3 / D2.4: the variable-bound contribution to dQ/dS.

    _SHARED_ESS_RATED_BOUNDED_VARIABLES gives, per variable, the lower and upper
    bound as multiples of S: lb = lower_factor * S, ub = upper_factor * S, so
    dl/dS = lower_factor and du/dS = upper_factor.
    """
    zL = getattr(model, 'ipopt_zL_out', None)
    zU = getattr(model, 'ipopt_zU_out', None)
    per_variable, total = {}, 0.0
    for name, lower_factor, upper_factor in mch._SHARED_ESS_RATED_BOUNDED_VARIABLES:
        if not hasattr(model, name):
            continue
        sum_lo = sum_hi = 0.0
        n_active_lo = n_active_hi = 0
        worst = None
        for entry in ess_entries(model, name, e):
            l_mult = float(zL.get(entry, 0.0)) if zL is not None else 0.0
            u_mult = float(zU.get(entry, 0.0)) if zU is not None else 0.0
            sum_lo += l_mult * lower_factor
            sum_hi += u_mult * upper_factor
            if abs(l_mult) > 1e-12:
                n_active_lo += 1
            if abs(u_mult) > 1e-12:
                n_active_hi += 1
            contribution = l_mult * lower_factor + u_mult * upper_factor
            if worst is None or abs(contribution) > abs(worst[0]):
                worst = (contribution, str(entry), float(pe.value(entry)),
                         list(entry.bounds), l_mult, u_mult)
        per_variable[name] = {
            'lower_factor': lower_factor, 'upper_factor': upper_factor,
            'n_entries': len(ess_entries(model, name, e)),
            'n_nonzero_zL': n_active_lo, 'n_nonzero_zU': n_active_hi,
            'sum_zL_times_dl_dS': sum_lo,
            'sum_zU_times_du_dS': sum_hi,
            'contribution': sum_lo + sum_hi,
            'worst_entry': ({'contribution': worst[0], 'name': worst[1],
                             'value': worst[2], 'bounds': worst[3],
                             'zL': worst[4], 'zU': worst[5]} if worst else None),
        }
        total += sum_lo + sum_hi
    return {'per_variable': per_variable, 'total_bound_contribution': total}


def unbounded_variable_audit(model, e):
    """D2.3: which shared-ESS variables carry bounds at all, and do those bounds
    depend on S or E?"""
    rated = {n: (lo, hi) for n, lo, hi in mch._SHARED_ESS_RATED_BOUNDED_VARIABLES}
    s_pu = float(pe.value(model.shared_es_s_rated_fixed[e]))
    e_pu = float(pe.value(model.shared_es_e_rated_fixed[e]))
    zL = getattr(model, 'ipopt_zL_out', None)
    zU = getattr(model, 'ipopt_zU_out', None)
    out = {}
    for name in ESS_VARIABLES:
        entries = ess_entries(model, name, e)
        if not entries:
            out[name] = {'present': False}
            continue
        sample = entries[0]
        lo, hi = sample.bounds
        depends_on_s = name in rated
        # detect an E-scaled bound numerically as well, rather than trusting the list
        e_scaled = (e_pu > 0 and hi is not None
                    and abs(hi - e_pu * ENERGY_STORAGE_MAX_ENERGY_STORED) < 1e-15)
        n_lo = sum(1 for x in entries if zL is not None and abs(float(zL.get(x, 0.0))) > 1e-12)
        n_hi = sum(1 for x in entries if zU is not None and abs(float(zU.get(x, 0.0))) > 1e-12)
        out[name] = {
            'present': True, 'n_entries': len(entries),
            'sample_bounds': [lo, hi],
            'bound_depends_on_S': depends_on_s,
            'bound_depends_on_E': bool(e_scaled),
            'S_factors': rated.get(name),
            'n_entries_with_active_lower_bound': n_lo,
            'n_entries_with_active_upper_bound': n_hi,
            'max_abs_zL': max((abs(float(zL.get(x, 0.0))) for x in entries), default=0.0)
            if zL is not None else None,
            'max_abs_zU': max((abs(float(zU.get(x, 0.0))) for x in entries), default=0.0)
            if zU is not None else None,
        }
    out['_capacity'] = {'S_pu': s_pu, 'E_pu': e_pu}
    return out


def direct_parameter_dependence(model, e):
    """D2.2 category B: does any ACTIVE row other than the fixing rows reference
    the mutable capacity parameters directly?"""
    fixed_names = (f'shared_es_s_rated_fixed[{e}]', f'shared_es_e_rated_fixed[{e}]')
    hits = []
    for con in model.component_objects(pe.Constraint, active=True):
        local = con.local_name
        if local in ('shared_energy_storage_s_sensitivities',
                     'shared_energy_storage_e_sensitivities'):
            continue
        for index in con:
            data = con[index]
            if not data.active:
                continue
            body = str(data.body)
            if any(f in body for f in fixed_names):
                hits.append({'component': local, 'index': str(index),
                             'body': body[:200]})
                break
    return {'n_active_rows_using_fixed_params': len(hits), 'rows': hits[:20]}


def active_set_signature(model, e):
    """D2.9: which capacity-relevant rows are at their bound, plus a global
    count of active variable bounds, so a basin change is detectable."""
    sig = {}
    for name in TRACKED_ROWS:
        entries = ess_entries(model, name, e)
        n_tight = 0
        for entry in entries:
            try:
                body = float(pe.value(entry.body))
            except Exception:
                continue
            up = entry.upper
            lo = entry.lower
            if up is not None and abs(body - float(pe.value(up))) <= 1e-8 * max(1.0, abs(float(pe.value(up)))):
                n_tight += 1
            elif lo is not None and abs(body - float(pe.value(lo))) <= 1e-8 * max(1.0, abs(float(pe.value(lo)))):
                n_tight += 1
        sig[name] = {'n_rows': len(entries), 'n_tight': n_tight}
    zL = getattr(model, 'ipopt_zL_out', None)
    zU = getattr(model, 'ipopt_zU_out', None)
    sig['_global'] = {
        'n_active_lower_bounds': sum(1 for _, v in (zL.items() if zL else []) if abs(float(v)) > 1e-9),
        'n_active_upper_bounds': sum(1 for _, v in (zU.items() if zU else []) if abs(float(v)) > 1e-9),
    }
    return sig


def sensitivity_decomposition(model, e, theta):
    """D2.4: split dQ/dtheta into fixing-row, bound and direct terms."""
    row = (model.shared_energy_storage_s_sensitivities[e] if theta == 'S'
           else model.shared_energy_storage_e_sensitivities[e])
    fixing = model.dual.get(row)
    fixing = float(fixing) if fixing is not None else None
    if theta == 'S':
        bounds = bound_multiplier_terms(model, e)
    else:
        # No shared-ESS variable bound is written from E: SOC limits are symbolic
        # rows against shared_es_e_rated. Verified numerically in D2.3.
        bounds = {'per_variable': {}, 'total_bound_contribution': 0.0,
                  'note': 'no capacity-dependent variable bound is written from E'}
    direct = direct_parameter_dependence(model, e)
    total = (fixing or 0.0) + bounds['total_bound_contribution']
    return {
        'theta': theta,
        'fixing_row_dual': fixing,
        'bound_contribution': bounds['total_bound_contribution'],
        'bound_detail': bounds['per_variable'],
        'direct_expression_dependence': direct,
        'corrected_total_derivative': total,
        'benders_uses_only_fixing_row': True,
        'fixing_row_share_of_total': (fixing / total) if total not in (0.0, None) else None,
    }


# ---------------------------------------------------------------------------
#  Cases
# ---------------------------------------------------------------------------
def build_case(planning, candidate, agent, ident, year, day):
    if agent == 'dso':
        holder = planning.distribution_networks[int(ident)]
        network = holder.network[int(year)][day]
        params = holder.params
        e = network.get_shared_energy_storage_idx(network.get_reference_node_id())
        node_for_capacity = int(ident)
    else:
        holder = planning.transmission_network
        network = holder.network[int(year)][day]
        params = holder.params
        e = int(ident)
        node_for_capacity = holder.active_distribution_network_nodes[e]
    s_pu = abs(candidate['total_capacity'][node_for_capacity][int(year)]['s']) / network.baseMVA
    e_pu = abs(candidate['total_capacity'][node_for_capacity][int(year)]['e']) / network.baseMVA
    tag = f'{agent}/{ident}/{year}/{day}'
    return {'tag': tag, 'network': network, 'params': params, 'e': e,
            's_pu': s_pu, 'e_pu': e_pu, 'base_mva': network.baseMVA}


def finite_differences(case, theta, steps):
    """D2.5 + D2.9: central differences with active-set tracking."""
    network, params, e = case['network'], case['params'], case['e']
    s0, e0 = case['s_pu'], case['e_pu']
    base_model, _r, ok = solve_at(network, params, e, s0, e0)
    if not ok:
        return {'base_solved': False}
    q0 = objective_of(base_model)
    base_sig = active_set_signature(base_model, e)
    decomposition = sensitivity_decomposition(base_model, e, theta)

    rows = []
    for rel in steps:
        theta0 = s0 if theta == 'S' else e0
        h = rel * theta0
        if theta == 'S':
            up = solve_at(network, params, e, s0 + h, e0)
            dn = solve_at(network, params, e, s0 - h, e0)
        else:
            up = solve_at(network, params, e, s0, e0 + h)
            dn = solve_at(network, params, e, s0, e0 - h)
        (mu, ru, oku), (md, rd, okd) = up, dn
        rec = {'relative_step': rel, 'step': h,
               'up_solved': oku, 'down_solved': okd}
        if not (oku and okd):
            rec['usable'] = False
            rows.append(rec)
            continue
        qu, qd = objective_of(mu), objective_of(md)
        sig_u, sig_d = active_set_signature(mu, e), active_set_signature(md, e)
        same_basin = (sig_u == sig_d == base_sig)
        central = (qu - qd) / (2 * h)
        rec.update({
            'Q_up': qu, 'Q_down': qd, 'Q_base': q0,
            'delta_Q': qu - qd,
            'central_difference': central,
            'fixing_dual_prediction': decomposition['fixing_row_dual'],
            'corrected_prediction': decomposition['corrected_total_derivative'],
            'rel_err_vs_fixing_dual': (
                abs(central - decomposition['fixing_row_dual'])
                / max(abs(decomposition['fixing_row_dual']), 1e-30)
                if decomposition['fixing_row_dual'] is not None else None),
            'rel_err_vs_corrected': (
                abs(central - decomposition['corrected_total_derivative'])
                / max(abs(decomposition['corrected_total_derivative']), 1e-30)),
            'termination_up': str(getattr(ru.solver, 'termination_condition', None)),
            'termination_down': str(getattr(rd.solver, 'termination_condition', None)),
            'active_set_up': sig_u, 'active_set_down': sig_d,
            'same_active_set_as_base': same_basin,
            'usable': same_basin,
        })
        rows.append(rec)
    return {'base_solved': True, 'Q_base': q0, 'theta0': s0 if theta == 'S' else e0,
            'base_active_set': base_sig, 'decomposition': decomposition,
            'finite_differences': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='dso:9:2030:Winter')
    parser.add_argument('--steps', default=','.join(str(s) for s in FD_STEPS))
    parser.add_argument('--out', default='p54d2_report.json')
    args = parser.parse_args()
    steps = tuple(float(x) for x in args.steps.split(','))

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-D2', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'sign_convention': {
                  'source': 'p54d2_sign_calibration.py (measured, not assumed)',
                  'dual_param_equals_var': '+dQ/dparam',
                  'dQ_du': '+zU', 'dQ_dl': '+zL'},
              'rated_bounded_variables': [
                  list(t) for t in mch._SHARED_ESS_RATED_BOUNDED_VARIABLES],
              'soc_limit_semantics': {
                  'lower': 'shared_es_soc >= shared_es_e_rated * '
                           f'{ENERGY_STORAGE_MIN_ENERGY_STORED} (symbolic row)',
                  'upper': 'shared_es_soc <= shared_es_e_rated * '
                           f'{ENERGY_STORAGE_MAX_ENERGY_STORED} (symbolic row)',
                  'initial_soc_value': f'{ENERGY_STORAGE_RELATIVE_INIT_SOC} * E '
                                       '(start point only, not a bound)'}}

    console = io.StringIO()
    with redirect_stdout(console):
        planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
        planning.read_planning_problem()
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)

    results = {}
    for spec in args.cases.split(','):
        agent, ident, year, day = spec.split(':')
        case = build_case(planning, candidate, agent, ident, year, day)
        print(f"\n[D2] case {case['tag']}  S={case['s_pu']:.6e}  E={case['e_pu']:.6e}",
              flush=True)

        base_model, _r, ok = solve_at(case['network'], case['params'], case['e'],
                                      case['s_pu'], case['e_pu'])
        if not ok:
            results[case['tag']] = {'base_solved': False}
            print('    base solve FAILED')
            continue

        entry = {'s_pu': case['s_pu'], 'e_pu': case['e_pu'],
                 'base_mva': case['base_mva'],
                 'D2_3_variable_bounds': unbounded_variable_audit(base_model, case['e']),
                 'D2_2_direct_parameter_dependence': direct_parameter_dependence(
                     base_model, case['e'])}

        print('    D2.5 S ...', flush=True)
        entry['S'] = finite_differences(case, 'S', steps)
        print('    D2.5 E ...', flush=True)
        entry['E'] = finite_differences(case, 'E', steps)
        results[case['tag']] = entry

        for theta in ('S', 'E'):
            d = entry[theta]['decomposition']
            print(f"    {theta}: fixing_dual={d['fixing_row_dual']:.6e} "
                  f"bound_contrib={d['bound_contribution']:.6e} "
                  f"corrected={d['corrected_total_derivative']:.6e}")
            for r in entry[theta]['finite_differences']:
                if not r.get('usable'):
                    print(f"       rel={r['relative_step']:<6g} UNUSABLE "
                          f"(solved={r['up_solved']}/{r['down_solved']}, "
                          f"same_set={r.get('same_active_set_as_base')})")
                    continue
                print(f"       rel={r['relative_step']:<6g} central={r['central_difference']:+.6e} "
                      f"err_vs_dual={r['rel_err_vs_fixing_dual']:.3e} "
                      f"err_vs_corrected={r['rel_err_vs_corrected']:.3e}")

    report['cases'] = results
    out = os.path.join(OUT_DIR, args.out)
    with open(out, 'w') as h:
        json.dump(report, h, indent=1, default=str)
    print(f'\n[D2] report -> {out}')


if __name__ == '__main__':
    main()
