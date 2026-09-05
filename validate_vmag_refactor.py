import argparse
from copy import deepcopy
import inspect
import json
import os
import re
import time

import pyomo.environ as pe

from centralized_coordination import combine_networks
from helper_functions import solver_result_succeeded, solver_result_summary
from model_construction_helpers import configure_shared_ess_operational_state
from model_construction_helpers import (
    dn_interface_expected_vmag_rule,
    get_vmag_node_indices,
    tn_interface_expected_vmag_rule,
    voltage_setpoint_cons_rule,
)
from network import _process_results
from shared_resources_planning import SharedResourcesPlanning


CONTROL_YEAR = 2025
CONTROL_DAY = 'Spring'


def _json_default(value):
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


def _solver_log_path(network, params, recovery=False):
    output_file = params.solver_params.options.get('output_file')
    if not output_file:
        return None
    configured_path = os.path.join(network.logs_dir, output_file)
    stem, extension = os.path.splitext(configured_path)
    day = ''.join(
        character if character.isalnum() or character in ('-', '_') else '_'
        for character in str(network.day)
    )
    suffix = f'_{network.year}_{day}'
    if recovery:
        suffix += '_recovery'
    return f'{stem}{suffix}{extension}'


def _file_size(path):
    return os.path.getsize(path) if path and os.path.exists(path) else 0


def _read_appended_text(path, initial_size):
    if not path or not os.path.exists(path):
        return ''
    with open(path, 'rb') as stream:
        stream.seek(initial_size)
        return stream.read().decode('utf-8', errors='replace')


def _last_match(pattern, text, cast=float):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return cast(matches[-1]) if matches else None


def _parse_ipopt_log(text):
    return {
        'iterations': _last_match(r'Number of Iterations\.{4}:\s*(\d+)', text, int),
        'objective': _last_match(r'^Objective\.{15}:\s*[+\-\d.eE]+\s+([+\-\d.eE]+)', text),
        'dual_infeasibility': _last_match(r'^Dual infeasibility\.{6}:\s*[+\-\d.eE]+\s+([+\-\d.eE]+)', text),
        'primal_infeasibility_scaled': _last_match(r'^Constraint violation\.{4}:\s*([+\-\d.eE]+)', text),
        'primal_infeasibility_unscaled': _last_match(r'^Constraint violation\.{4}:\s*[+\-\d.eE]+\s+([+\-\d.eE]+)', text),
        'complementarity': _last_match(r'^Complementarity\.{9}:\s*[+\-\d.eE]+\s+([+\-\d.eE]+)', text),
        'runtime_seconds': _last_match(r'^Total seconds in IPOPT\s*=\s*([+\-\d.eE]+)', text),
        'exit': (_last_match(r'^EXIT:\s*(.+)$', text, str) or '').strip() or None,
    }


def _configure_inactive_shared_ess(model):
    for storage in model.shared_energy_storages:
        configure_shared_ess_operational_state(
            model,
            storage,
            pe.value(model.shared_es_s_rated_fixed[storage]),
            pe.value(model.shared_es_e_rated_fixed[storage]),
        )


def _capture_control(label, network, params):
    model = network.build_model(params)
    _configure_inactive_shared_ess(model)
    if network.is_transmission:
        model.active_distribution_networks = range(
            len(network.active_distribution_network_nodes)
        )

    primary_log = _solver_log_path(network, params)
    recovery_log = _solver_log_path(network, params, recovery=True)
    primary_start = _file_size(primary_log)
    recovery_start = _file_size(recovery_log)
    wall_start = time.perf_counter()
    result = network.run_smopf(model, params, from_warm_start=False)
    wall_runtime = time.perf_counter() - wall_start

    if not solver_result_succeeded(result):
        raise RuntimeError(f'{label} control failed: {solver_result_summary(result)}')

    processed = network.process_results(model, params, results=result)
    interface = network.process_results_interface(model)
    primary_metrics = _parse_ipopt_log(_read_appended_text(primary_log, primary_start))
    recovery_text = _read_appended_text(recovery_log, recovery_start)

    return {
        'label': label,
        'network': network.name,
        'year': network.year,
        'day': network.day,
        'from_warm_start': False,
        'solver_summary': solver_result_summary(result),
        'primary': primary_metrics,
        'recovery_attempted': bool(recovery_text),
        'recovery': _parse_ipopt_log(recovery_text) if recovery_text else None,
        'wall_runtime_seconds': wall_runtime,
        'model_objective': pe.value(model.objective),
        'capacity_sensitivity_duals': {
            's': {
                str(storage): model.dual.get(
                    model.shared_energy_storage_s_sensitivities[storage]
                )
                for storage in model.shared_energy_storages
            },
            'e': {
                str(storage): model.dual.get(
                    model.shared_energy_storage_e_sensitivities[storage]
                )
                for storage in model.shared_energy_storages
            },
        },
        'interface': interface,
        'processed_results': processed,
    }


def capture_controls(output_path):
    planning = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
    planning.read_planning_problem()
    controls = {
        'scenario_checksum': planning.scenario_metadata['combined_scenario_checksum'],
        'dso': _capture_control(
            'DSO node 7',
            planning.distribution_networks[7].network[CONTROL_YEAR][CONTROL_DAY],
            planning.distribution_networks[7].params,
        ),
        'tso': _capture_control(
            'TSO with active DSO interfaces',
            planning.transmission_network.network[CONTROL_YEAR][CONTROL_DAY],
            planning.transmission_network.params,
        ),
    }
    with open(output_path, 'w', encoding='utf-8') as stream:
        json.dump(controls, stream, indent=2, default=_json_default)
    print(json.dumps({
        key: value if key == 'scenario_checksum' else {
            'solver_summary': value['solver_summary'],
            'objective': value['model_objective'],
            'primary': value['primary'],
            'recovery_attempted': value['recovery_attempted'],
        }
        for key, value in controls.items()
    }, indent=2))


def _add_expected_interface_voltage(model, network):
    if network.is_transmission:
        model.active_distribution_networks = range(
            len(network.active_distribution_network_nodes)
        )
        model.expected_interface_vmag = pe.Var(
            model.active_distribution_networks,
            model.periods,
            domain=pe.NonNegativeReals,
            initialize=1.0,
        )
        model.expected_interface_vmag_def = pe.Constraint(
            model.active_distribution_networks,
            model.periods,
            rule=lambda m, dn, p: tn_interface_expected_vmag_rule(
                m, dn, p, network
            ),
        )
    else:
        model.expected_interface_vmag = pe.Var(
            model.periods,
            domain=pe.NonNegativeReals,
            initialize=1.0,
        )
        model.expected_interface_vmag_def = pe.Constraint(
            model.periods,
            rule=lambda m, p: dn_interface_expected_vmag_rule(m, p, network),
        )


def _model_structure(label, model, network):
    scenarios = (
        len(model.scenarios_market)
        * len(model.scenarios_operation)
        * len(model.periods)
    )
    expected_vmag = len(model.vmag_nodes) * scenarios
    expected_full = len(model.nodes) * scenarios
    assert len(model.vmag) == expected_vmag
    assert len(model.voltage_mag_def) == expected_vmag
    assert len(model.e) == expected_full
    assert len(model.f) == expected_full
    assert len(model.vmag_sqr) == expected_full
    assert len(model.voltage_mag_sqr_def) == expected_full
    assert len(model.voltage_magnitude_lower_cons) == expected_full
    assert len(model.voltage_magnitude_upper_cons) == expected_full
    if hasattr(model, 'slack_v_sqr_down'):
        assert len(model.slack_v_sqr_down) == expected_full
        assert len(model.slack_v_sqr_up) == expected_full

    _add_expected_interface_voltage(model, network)
    for expression in model.vmag_adn.values():
        assert expression.expr.parent_component() is model.vmag
        assert expression.expr.index()[0] in model.vmag_nodes

    return {
        'label': label,
        'physical_nodes': len(model.nodes),
        'vmag_nodes': list(model.vmag_nodes),
        'vmag': len(model.vmag),
        'vmag_sqr': len(model.vmag_sqr),
        'voltage_mag_def': len(model.voltage_mag_def),
        'voltage_mag_sqr_def': len(model.voltage_mag_sqr_def),
        'expected_interface_vmag_def': len(model.expected_interface_vmag_def),
    }


def validate_construction():
    planning = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
    planning.read_planning_problem()

    dso_network = planning.distribution_networks[7].network[CONTROL_YEAR][CONTROL_DAY]
    dso_model = dso_network.build_model(planning.distribution_networks[7].params)
    assert list(dso_model.vmag_nodes) == [
        dso_network.get_node_idx(dso_network.get_reference_node_id())
    ]

    tso_network = planning.transmission_network.network[CONTROL_YEAR][CONTROL_DAY]
    tso_model = tso_network.build_model(planning.transmission_network.params)
    assert list(tso_model.vmag_nodes) == [
        tso_network.get_node_idx(bus_id)
        for bus_id in tso_network.active_distribution_network_nodes
    ]

    no_interface_tso = deepcopy(tso_network)
    no_interface_tso.active_distribution_network_nodes = []
    no_interface_model = no_interface_tso.build_model(planning.transmission_network.params)
    assert list(no_interface_model.vmag_nodes) == []

    centralized = combine_networks(
        planning.transmission_network,
        planning.distribution_networks,
    )
    centralized_network = centralized.network[CONTROL_YEAR][CONTROL_DAY]
    centralized_model = centralized_network.build_model(centralized.params)
    assert list(centralized_model.vmag_nodes) == []

    duplicate_tso = deepcopy(tso_network)
    duplicate_tso.active_distribution_network_nodes = [5, 5]
    try:
        get_vmag_node_indices(duplicate_tso)
    except ValueError as error:
        duplicate_error = str(error)
    else:
        raise AssertionError('Duplicate TSO interface metadata was accepted.')

    unmappable_tso = deepcopy(tso_network)
    unmappable_tso.active_distribution_network_nodes = ['missing-bus']
    try:
        get_vmag_node_indices(unmappable_tso)
    except ValueError as error:
        unmappable_error = str(error)
    else:
        raise AssertionError('Unmappable TSO interface metadata was accepted.')

    setpoint_source = inspect.getsource(voltage_setpoint_cons_rule)
    assert 'm.vmag_sqr[' in setpoint_source
    assert 'm.vmag[' not in setpoint_source
    result_source = inspect.getsource(_process_results)
    assert 'v_mag = sqrt(e**2 + f**2)' in result_source

    structures = [
        _model_structure('DSO', dso_model, dso_network),
        _model_structure('TSO with interfaces', tso_model, tso_network),
        _model_structure('TSO without interfaces', no_interface_model, no_interface_tso),
        _model_structure('centralized combined', centralized_model, centralized_network),
    ]
    print(json.dumps({
        'structures': structures,
        'duplicate_metadata_error': duplicate_error,
        'unmappable_metadata_error': unmappable_error,
        'pv_setpoint_uses_vmag_sqr': True,
        'result_processing_reconstructs_from_e_f': True,
        'all_assertions_passed': True,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-controls', metavar='OUTPUT_JSON')
    parser.add_argument('--validate-construction', action='store_true')
    args = parser.parse_args()
    if args.capture_controls:
        capture_controls(args.capture_controls)
        return
    if args.validate_construction:
        validate_construction()
        return
    parser.error('one validation action is required')


if __name__ == '__main__':
    main()
