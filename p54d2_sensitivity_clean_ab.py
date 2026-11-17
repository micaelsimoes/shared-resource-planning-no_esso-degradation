"""
Stage P5.4-D2.6/D2.7/D2.8/D2.10 -- sensitivity-clean bound reformulation A/B.

D2.2/D2.3 established that four shared-ESS variable bounds are rewritten
numerically from the fixed capacity parameter:

    pch  in [0*S, 1*S]      pdch in [0*S, 1*S]
    pnet in [-1*S, 1*S]     qnet in [-1*S, 1*S]

Every one of them is redundant with a symbolic constraint that is already in the
model:

    pch <= S, pdch <= S   <=  pch + pdch <= S  with pch, pdch >= 0   (sess_active_sum_limit)
                          <=  pch = S*pch_hat  with pch_hat <= 1     (P5.4-H1 link)
    |pnet| <= S           <=  pnet^2 + qnet^2 <= S^2                 (sess_converter_capability)
    |qnet| <= S           <=  same

So the capacity-dependent numerical bounds can be dropped without changing the
feasible set, which routes ALL S-dependence through symbolic rows and the
`shared_es_s_rated` variable -- and therefore through the fixing-row dual that
Benders actually reads.

  A -- current production: capacity-dependent numerical bounds retained
  B -- sensitivity-clean:  only those four bounds relaxed to capacity-independent
       values. Nonnegativity of pch/pdch, the [0,1] hat bounds, the symbolic
       active-sum and capability rows, all SOC/energy rows and the zero-capacity
       gating are untouched.

D2.7 (E analogue): the SOC limits are ALREADY symbolic rows against
`shared_es_e_rated`, and `shared_es_soc` carries no E-scaled numerical bound.
That is verified numerically here rather than assumed, and no E-side change is
proposed.

    python p54d2_sensitivity_clean_ab.py [--cases ...]
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
                         SHARED_ESS_ZERO_CAPACITY_TOLERANCE)
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

from p54d2_sensitivity_root_cause import (active_set_signature, build_case,  # noqa: E402
                                          bound_multiplier_terms, ess_entries,
                                          objective_of, sensitivity_decomposition)

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P54D2')

DEFAULT_CASES = ('dso:5:2025:Winter,dso:5:2030:Summer,'
                 'dso:7:2025:Spring,dso:7:2030:Winter,'
                 'dso:9:2030:Winter,dso:9:2035:Summer,'
                 'tso:0:2025:Winter,tso:1:2030:Summer')


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def make_sensitivity_clean(model, e):
    """Variant B: drop only the four redundant capacity-dependent bounds.

    pch/pdch keep their lower bound of 0 (nonnegativity, capacity-independent);
    their upper bound and both pnet/qnet bounds are removed, since the symbolic
    rows already imply them.
    """
    changed = []
    for name in ('shared_es_pch', 'shared_es_pdch'):
        for entry in ess_entries(model, name, e):
            changed.append((entry.name, list(entry.bounds)))
            entry.setlb(0.0)          # nonnegativity retained, independent of S
            entry.setub(None)         # implied by sess_active_sum_limit / H1 link
    for name in ('shared_es_pnet', 'shared_es_qnet'):
        for entry in ess_entries(model, name, e):
            changed.append((entry.name, list(entry.bounds)))
            entry.setlb(None)         # implied by sess_converter_capability
            entry.setub(None)
    return {'n_bounds_relaxed': len(changed), 'sample': changed[:4]}


def solve_variant(network, params, e, s_pu, e_pu, clean):
    console = io.StringIO()
    with redirect_stdout(console):
        m = network.build_model(params)
        m.shared_es_s_rated_fixed[e].set_value(s_pu)
        m.shared_es_e_rated_fixed[e].set_value(e_pu)
        mch.configure_shared_ess_operational_state(m, e, s_pu, e_pu)
        info = make_sensitivity_clean(m, e) if clean else {'n_bounds_relaxed': 0}
        r = network.run_smopf(m, params, print_header=False)
    ok = bool(srp._solver_result_succeeded(r))
    return m, r, ok, info, console.getvalue()


def physical_profile(model, e):
    """Everything D2.8 requires compared between A and B."""
    out = {}
    for name in ('shared_es_pch', 'shared_es_pdch', 'shared_es_pnet',
                 'shared_es_qnet', 'shared_es_soc', 'shared_es_pch_hat',
                 'shared_es_pdch_hat'):
        out[name] = [float(pe.value(x)) for x in ess_entries(model, name, e)]
    for name in ('pg', 'qg', 'vmag'):
        if hasattr(model, name):
            comp = getattr(model, name)
            out[name] = [float(pe.value(comp[i])) for i in comp]
    for name in ('expected_shared_ess_p', 'expected_shared_ess_q'):
        if hasattr(model, name):
            comp = getattr(model, name)
            out[name] = [float(pe.value(comp[i])) for i in comp]
    return out


def max_abs_diff(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def binding_report(model, e):
    """Is S or E operationally binding at this solution? (D2.10)"""
    s = float(pe.value(model.shared_es_s_rated_fixed[e]))
    e_cap = float(pe.value(model.shared_es_e_rated_fixed[e]))
    sum_slack = min(
        (float(pe.value(x.upper)) - float(pe.value(x.body))
         for x in ess_entries(model, 'sess_active_sum_limit', e)), default=None)
    cap_slack = min(
        (float(pe.value(x.upper)) - float(pe.value(x.body))
         for x in ess_entries(model, 'sess_converter_capability', e)), default=None)
    socs = [float(pe.value(x)) for x in ess_entries(model, 'shared_es_soc', e)]
    soc_hi = e_cap * ENERGY_STORAGE_MAX_ENERGY_STORED
    soc_lo = e_cap * ENERGY_STORAGE_MIN_ENERGY_STORED
    return {
        'S_pu': s, 'E_pu': e_cap,
        'min_active_sum_slack': sum_slack,
        'min_capability_slack': cap_slack,
        'S_binding': bool(sum_slack is not None and sum_slack <= 1e-8 * max(s, 1e-30))
        or bool(cap_slack is not None and cap_slack <= 1e-8 * max(s ** 2, 1e-30)),
        'max_soc': max(socs) if socs else None,
        'min_soc': min(socs) if socs else None,
        'soc_upper_limit': soc_hi, 'soc_lower_limit': soc_lo,
        'soc_upper_slack': (soc_hi - max(socs)) if socs else None,
        'soc_lower_slack': (min(socs) - soc_lo) if socs else None,
        'E_binding': bool(socs and (
            (soc_hi - max(socs)) <= 1e-8 * max(e_cap, 1e-30)
            or (min(socs) - soc_lo) <= 1e-8 * max(e_cap, 1e-30))),
    }


def e_side_bound_check(model, e):
    """D2.7: confirm numerically that no shared-ESS variable bound is written
    from the energy capacity."""
    e_cap = float(pe.value(model.shared_es_e_rated_fixed[e]))
    findings = {}
    for name in ('shared_es_soc', 'slack_shared_es_soc_final_up',
                 'slack_shared_es_soc_final_down'):
        entries = ess_entries(model, name, e)
        if not entries:
            findings[name] = {'present': False}
            continue
        lo, hi = entries[0].bounds
        findings[name] = {
            'present': True, 'bounds': [lo, hi],
            'upper_equals_E_times_max_fraction': (
                hi is not None and e_cap > 0
                and abs(hi - e_cap * ENERGY_STORAGE_MAX_ENERGY_STORED) < 1e-15),
            'upper_equals_E': (hi is not None and e_cap > 0
                               and abs(hi - e_cap) < 1e-15),
            'bound_depends_on_E': False,
        }
        findings[name]['bound_depends_on_E'] = bool(
            findings[name]['upper_equals_E_times_max_fraction']
            or findings[name]['upper_equals_E'])
    findings['_soc_limits_are_symbolic_rows'] = (
        hasattr(model, 'sess_soc_limit_upper') and hasattr(model, 'sess_soc_limit_lower'))
    findings['_soc_upper_row_body'] = str(
        model.sess_soc_limit_upper[ess_entries(model, 'shared_es_soc', e)[0].index()].expr
    )[:200] if hasattr(model, 'sess_soc_limit_upper') else None
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default=DEFAULT_CASES)
    parser.add_argument('--out', default='p54d2_sensitivity_clean_ab.json')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.4-D2.6/7/8/10', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'redundancy_proof': {
                  'pch_le_S': 'pch + pdch <= S with pdch >= 0; also pch = S*pch_hat, pch_hat <= 1',
                  'pdch_le_S': 'pch + pdch <= S with pch >= 0; also pdch = S*pdch_hat, pdch_hat <= 1',
                  'abs_pnet_le_S': 'pnet^2 + qnet^2 <= S^2',
                  'abs_qnet_le_S': 'pnet^2 + qnet^2 <= S^2',
                  'retained_in_B': ['pch >= 0', 'pdch >= 0', 'hat bounds [0,1]',
                                    'sess_active_sum_limit', 'sess_converter_capability',
                                    'all SOC/energy rows', 'zero-capacity gating']}}

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
        tag = case['tag']
        print(f'\n[D2.6/8] {tag}', flush=True)

        mA, rA, okA, _iA, _l = solve_variant(case['network'], case['params'], case['e'],
                                             case['s_pu'], case['e_pu'], clean=False)
        mB, rB, okB, iB, _l = solve_variant(case['network'], case['params'], case['e'],
                                            case['s_pu'], case['e_pu'], clean=True)
        entry = {'A_solved': okA, 'B_solved': okB,
                 'n_bounds_relaxed_in_B': iB.get('n_bounds_relaxed')}
        if not (okA and okB):
            results[tag] = entry
            print(f'    A={okA} B={okB} -- skipping comparison')
            continue

        qa, qb = objective_of(mA), objective_of(mB)
        pa, pb = physical_profile(mA, case['e']), physical_profile(mB, case['e'])
        decA = sensitivity_decomposition(mA, case['e'], 'S')
        decB = sensitivity_decomposition(mB, case['e'], 'S')
        decAe = sensitivity_decomposition(mA, case['e'], 'E')
        decBe = sensitivity_decomposition(mB, case['e'], 'E')

        entry.update({
            'objective_A': qa, 'objective_B': qb,
            'objective_abs_diff': abs(qa - qb),
            'objective_rel_diff': abs(qa - qb) / max(abs(qa), 1e-30),
            'max_abs_diff': {k: max_abs_diff(pa.get(k), pb.get(k)) for k in pa},
            'same_ess_active_set': (
                json.dumps(active_set_signature(mA, case['e']), sort_keys=True)
                == json.dumps(active_set_signature(mB, case['e']), sort_keys=True)),
            'S_decomposition_A': {k: decA[k] for k in (
                'fixing_row_dual', 'bound_contribution', 'corrected_total_derivative',
                'fixing_row_share_of_total')},
            'S_decomposition_B': {k: decB[k] for k in (
                'fixing_row_dual', 'bound_contribution', 'corrected_total_derivative',
                'fixing_row_share_of_total')},
            'E_decomposition_A': {k: decAe[k] for k in (
                'fixing_row_dual', 'bound_contribution', 'corrected_total_derivative')},
            'E_decomposition_B': {k: decBe[k] for k in (
                'fixing_row_dual', 'bound_contribution', 'corrected_total_derivative')},
            'B_bound_contribution_is_zero': abs(decB['bound_contribution']) < 1e-12,
            'binding_A': binding_report(mA, case['e']),
            'D2_7_E_side_bounds': e_side_bound_check(mA, case['e']),
        })
        results[tag] = entry

        print(f"    obj A={qa:.10e} B={qb:.10e} rel_diff={entry['objective_rel_diff']:.3e} "
              f"same_active_set={entry['same_ess_active_set']}")
        print(f"    max|dpch|={entry['max_abs_diff'].get('shared_es_pch'):.3e} "
              f"max|dpnet|={entry['max_abs_diff'].get('shared_es_pnet'):.3e} "
              f"max|dsoc|={entry['max_abs_diff'].get('shared_es_soc'):.3e} "
              f"max|dpg|={entry['max_abs_diff'].get('pg'):.3e}")
        print(f"    S: A fix={decA['fixing_row_dual']:+.4e} bound={decA['bound_contribution']:+.4e} "
              f"(fixing share {decA['fixing_row_share_of_total']:.3f})")
        print(f"       B fix={decB['fixing_row_dual']:+.4e} bound={decB['bound_contribution']:+.4e} "
              f"-> bound term eliminated: {entry['B_bound_contribution_is_zero']}")
        b = entry['binding_A']
        print(f"    binding: S={b['S_binding']} (sum_slack={b['min_active_sum_slack']:.3e}) "
              f"E={b['E_binding']} (soc_up_slack={b['soc_upper_slack']:.3e})")

    report['cases'] = results
    n = len(results)
    ok_cases = [v for v in results.values() if v.get('A_solved') and v.get('B_solved')]
    report['summary'] = {
        'n_cases': n, 'n_compared': len(ok_cases),
        'max_objective_rel_diff': max((v['objective_rel_diff'] for v in ok_cases), default=None),
        'all_B_bound_contribution_zero': all(
            v['B_bound_contribution_is_zero'] for v in ok_cases) if ok_cases else None,
        'n_cases_S_binding': sum(1 for v in ok_cases if v['binding_A']['S_binding']),
        'n_cases_E_binding': sum(1 for v in ok_cases if v['binding_A']['E_binding']),
        'fixing_share_of_total_A': [v['S_decomposition_A']['fixing_row_share_of_total']
                                    for v in ok_cases],
    }
    out = os.path.join(OUT_DIR, args.out)
    with open(out, 'w') as h:
        json.dump(report, h, indent=1, default=str)
    print(f"\n[D2] summary: {json.dumps(report['summary'], indent=1, default=str)}")
    print(f'[D2] report -> {out}')


if __name__ == '__main__':
    main()
