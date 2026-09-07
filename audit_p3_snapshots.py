import argparse
import hashlib
import json
import math
import pickle
import statistics

import pyomo.environ as pe
from pyomo.core.expr.calculus.derivatives import Modes, differentiate
from pyomo.core.expr.visitor import identify_variables


VARIABLE_FAMILIES = (
    'shared_es_s_rated', 'shared_es_e_rated', 'shared_es_pch',
    'shared_es_pdch', 'shared_es_sch', 'shared_es_sdch',
    'shared_es_pnet', 'shared_es_qnet', 'shared_es_soc',
    'expected_shared_ess_p', 'expected_shared_ess_q',
    'expected_interface_vmag', 'expected_interface_pf_p',
    'expected_interface_pf_q', 'e', 'f', 'vmag', 'vmag_sqr',
    'voltage_product_real', 'voltage_product_imag', 'r', 'r_sqr',
    'pij', 'qij', 'pji', 'qji', 'pg', 'qg', 'flex_p_up',
    'flex_p_down', 'flex_q_up', 'flex_q_down',
)

PARAMETER_FAMILIES = (
    'shared_es_s_rated_fixed', 'shared_es_e_rated_fixed',
    'vmag_req', 'dual_vmag_req', 'p_pf_req', 'q_pf_req',
    'dual_pf_p_req', 'dual_pf_q_req', 'p_ess_req', 'q_ess_req',
    'dual_ess_p_req', 'dual_ess_q_req', 'p_ess_prev', 'q_ess_prev',
    'dual_ess_p_prev', 'dual_ess_q_prev',
)

CONSTRAINT_FAMILIES = (
    'sess_pnet_def', 'sess_snet_def', 'sess_pch_link',
    'sess_pdch_link', 'sess_s_limit', 'sess_phi_limit_lower',
    'sess_phi_limit_upper', 'sess_soc_def', 'sess_soc_limit_upper',
    'sess_soc_limit_lower', 'sess_soc_final', 'sess_comp',
    'voltage_mag_def', 'voltage_mag_sqr_def',
    'voltage_product_real_def', 'voltage_product_imag_def',
    'r_sqr_def', 'pij_def', 'qij_def', 'pji_def', 'qji_def',
    'sg_capability', 'node_balance_p', 'node_balance_q',
    'branch_flow_limit', 'branch_flow_limit_ji',
)


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def stats(values):
    values = [value for value in (finite(v) for v in values) if value is not None]
    if not values:
        return {'count': 0}
    absolute = [abs(value) for value in values]
    nonzero = [value for value in absolute if value > 1e-14]
    return {
        'count': len(values),
        'min': min(values),
        'max': max(values),
        'abs_max': max(absolute),
        'median_abs': statistics.median(absolute),
        'min_nonzero_abs': min(nonzero) if nonzero else None,
        'near_zero_count': sum(value <= 1e-10 for value in absolute),
    }


def component_values(component, active_only=False):
    result = {}
    for index in component:
        item = component[index]
        if active_only and not item.active:
            continue
        value = finite(pe.value(item, exception=False))
        if value is not None:
            result[str(index)] = value
    return result


def variable_summary(model, name):
    if not hasattr(model, name):
        return None
    component = getattr(model, name)
    values, lower, upper = [], [], []
    fixed = 0
    active = 0
    for item in component.values():
        if not item.active:
            continue
        active += 1
        fixed += int(item.fixed)
        values.append(pe.value(item, exception=False))
        lower.append(pe.value(item.lb, exception=False) if item.lb is not None else None)
        upper.append(pe.value(item.ub, exception=False) if item.ub is not None else None)
    return {
        'active': active,
        'fixed': fixed,
        'values': stats(values),
        'lower_bounds': stats(lower),
        'upper_bounds': stats(upper),
    }


def constraint_residual(item):
    body = finite(pe.value(item.body, exception=False))
    lower = finite(pe.value(item.lower, exception=False)) if item.has_lb() else None
    upper = finite(pe.value(item.upper, exception=False)) if item.has_ub() else None
    if body is None:
        return None
    if item.equality:
        return body - lower
    violation = 0.0
    if lower is not None:
        violation = max(violation, lower - body)
    if upper is not None:
        violation = max(violation, body - upper)
    return violation


def gradient_stats(item):
    variables = list(identify_variables(item.body, include_fixed=False))
    if not variables:
        return 0.0, None, None, 0
    derivatives = differentiate(item.body, wrt_list=variables, mode=Modes.reverse_numeric)
    magnitudes = [abs(float(value)) for value in derivatives if finite(value) is not None]
    nonzero = [value for value in magnitudes if value > 1e-14]
    return (
        max(magnitudes) if magnitudes else 0.0,
        min(nonzero) if nonzero else None,
        max(nonzero) if nonzero else None,
        len(nonzero),
    )


def constraint_summary(model, name):
    if not hasattr(model, name):
        return None
    component = getattr(model, name)
    residuals, inf_norms, nonzero_mins, nonzero_maxs, multipliers = [], [], [], [], []
    active_count = equality_count = zero_gradient_count = 0
    dual = getattr(model, 'dual', None)
    for item in component.values():
        if not item.active:
            continue
        active_count += 1
        equality_count += int(item.equality)
        residuals.append(constraint_residual(item))
        inf_norm, nonzero_min, nonzero_max, nonzero_count = gradient_stats(item)
        inf_norms.append(inf_norm)
        if nonzero_count == 0:
            zero_gradient_count += 1
        if nonzero_min is not None:
            nonzero_mins.append(nonzero_min)
            nonzero_maxs.append(nonzero_max)
        if dual is not None and item in dual:
            multipliers.append(dual[item])
    return {
        'active': active_count,
        'equalities': equality_count,
        'max_abs_residual_or_violation': max((abs(v) for v in residuals if v is not None), default=None),
        'gradient_inf_norm': stats(inf_norms),
        'min_nonzero_derivative': min(nonzero_mins) if nonzero_mins else None,
        'max_nonzero_derivative': max(nonzero_maxs) if nonzero_maxs else None,
        'zero_gradient_rows': zero_gradient_count,
        'multipliers': stats(multipliers),
    }


def suffix_summary(model, suffix_name, variable_name):
    suffix = getattr(model, suffix_name, None)
    component = getattr(model, variable_name, None)
    if suffix is None or component is None:
        return None
    return stats(suffix[item] for item in component.values() if item in suffix)


def snapshot(path):
    with open(path, 'rb') as file:
        payload = pickle.load(file)
    model = payload['model']
    with open(path, 'rb') as file:
        digest = hashlib.sha256(file.read()).hexdigest()
    return payload, {
        'path': path,
        'sha256': digest,
        'metadata': payload.get('metadata', {}),
        'model_counts': {
            'active_variables': sum(1 for item in model.component_data_objects(pe.Var, active=True) if not item.fixed),
            'fixed_variables': sum(1 for item in model.component_data_objects(pe.Var, active=True) if item.fixed),
            'active_constraints': sum(1 for _ in model.component_data_objects(pe.Constraint, active=True)),
            'active_equalities': sum(1 for item in model.component_data_objects(pe.Constraint, active=True) if item.equality),
        },
        'variables': {name: variable_summary(model, name) for name in VARIABLE_FAMILIES if hasattr(model, name)},
        'parameters': {name: stats(component_values(getattr(model, name)).values()) for name in PARAMETER_FAMILIES if hasattr(model, name)},
        'constraints': {name: constraint_summary(model, name) for name in CONSTRAINT_FAMILIES if hasattr(model, name)},
        'warm_start_suffixes': {
            name: {
                suffix: suffix_summary(model, suffix, name)
                for suffix in ('ipopt_zL_in', 'ipopt_zU_in')
            }
            for name in VARIABLE_FAMILIES if hasattr(model, name)
        },
    }


def compare(left_payload, right_payload):
    left, right = left_payload['model'], right_payload['model']
    output = {}
    for category, names in (('variables', VARIABLE_FAMILIES), ('parameters', PARAMETER_FAMILIES)):
        category_output = {}
        for name in names:
            if not hasattr(left, name) or not hasattr(right, name):
                continue
            left_values = component_values(getattr(left, name))
            right_values = component_values(getattr(right, name))
            shared = left_values.keys() & right_values.keys()
            category_output[name] = stats(right_values[index] - left_values[index] for index in shared)
        output[category] = category_output
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('failure')
    parser.add_argument('success')
    arguments = parser.parse_args()
    failure_payload, failure_summary = snapshot(arguments.failure)
    success_payload, success_summary = snapshot(arguments.success)
    print(json.dumps({
        'failure': failure_summary,
        'success': success_summary,
        'success_minus_failure': compare(failure_payload, success_payload),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
