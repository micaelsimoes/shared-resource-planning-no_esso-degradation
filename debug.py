import os
import math
import argparse
import pickle
import pyomo.environ as pe
from helper_functions import solver_result_summary
from model_construction_helpers import get_vmag_node_indices
from shared_resources_planning import SharedResourcesPlanning


def replay_frozen_dso_block(planning_problem, frozen_filename):

    with open(frozen_filename, 'rb') as file:
        payload = pickle.load(file)

    metadata = payload['metadata']

    node_id = metadata['node_id']
    year = metadata['year']
    day = metadata['day']

    # Every experiment begins from exactly the same frozen state.
    model = payload['model'].clone()

    distribution_network = planning_problem.distribution_networks[node_id]
    network = distribution_network.network[year][day]
    params = distribution_network.params

    apply_debug_nonreference_vmag_elimination(model, network)
    print(
        '[FROZEN SMOPF] '
        f'node={node_id} | '
        f'network={network.name} | '
        f'year={year} | '
        f'day={day} | '
        f'original ADMM cycle={metadata["cycle"]} | '
        f'warm_start={metadata["from_warm_start"]}'
    )

    result = network.run_smopf(model, params, from_warm_start=metadata['from_warm_start'])

    print(f'[FROZEN SMOPF] Result: {solver_result_summary(result)}')

    return model, result


def _active_nlp_counts(model):

    active_variables = sum(
        1
        for variable in model.component_data_objects(pe.Var, active=True, descend_into=True)
        if not variable.fixed
    )
    active_equalities = sum(
        1
        for constraint in model.component_data_objects(pe.Constraint, active=True, descend_into=True)
        if constraint.equality
    )
    return active_variables, active_equalities


def apply_debug_nonreference_vmag_elimination(model, network):
    """Remove non-reference ``vmag`` instances from a frozen clone's NLP."""

    retained_nodes = set(get_vmag_node_indices(network))
    variables_before, equalities_before = _active_nlp_counts(model)
    fixed_nonreference_vmag = 0
    deactivated_voltage_mag_def = 0

    for i in model.nodes:
        if i in retained_nodes:
            continue

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    vmag_sqr_value = float(pe.value(model.vmag_sqr[i, s_m, s_o, p]))
                    model.vmag[i, s_m, s_o, p].fix(math.sqrt(max(vmag_sqr_value, 0.0)))
                    fixed_nonreference_vmag += 1

                    voltage_mag_def = model.voltage_mag_def[i, s_m, s_o, p]
                    if voltage_mag_def.active:
                        voltage_mag_def.deactivate()
                        deactivated_voltage_mag_def += 1

    variables_after, equalities_after = _active_nlp_counts(model)
    retained_vmag = sum(
        not model.vmag[i, s_m, s_o, p].fixed
        for i in retained_nodes
        for s_m in model.scenarios_market
        for s_o in model.scenarios_operation
        for p in model.periods
    )
    retained_voltage_mag_def = sum(
        model.voltage_mag_def[i, s_m, s_o, p].active
        for i in retained_nodes
        for s_m in model.scenarios_market
        for s_o in model.scenarios_operation
        for p in model.periods
    )
    active_voltage_mag_sqr_def = sum(
        constraint.active for constraint in model.voltage_mag_sqr_def.values()
    )

    print(
        '[FROZEN SMOPF][DEBUG NONREFERENCE VMAG] '
        f'retained_node_indices={sorted(retained_nodes)} | '
        f'fixed_nonreference_vmag={fixed_nonreference_vmag} | '
        f'deactivated_voltage_mag_def={deactivated_voltage_mag_def} | '
        f'retained_vmag={retained_vmag} | '
        f'retained_voltage_mag_def={retained_voltage_mag_def} | '
        f'active_voltage_mag_sqr_def={active_voltage_mag_sqr_def}'
    )
    print(
        '[FROZEN SMOPF][NLP COUNTS] '
        f'active_variables_before={variables_before} | '
        f'active_variables_after={variables_after} | '
        f'active_equalities_before={equalities_before} | '
        f'active_equalities_after={equalities_after}'
    )


def apply_tighter_vmag_bounds(model, network):

    tightened = 0

    min_lb = float('inf')
    max_ub = -float('inf')

    for i in model.nodes:

        node = network.nodes[i]

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:

                    v = model.vmag[i, s_m, s_o, p]
                    slack_down = model.slack_v_sqr_down[i, s_m, s_o, p]
                    slack_up = model.slack_v_sqr_up[i, s_m, s_o, p]
                    down_ub = slack_down.ub
                    up_ub = slack_up.ub

                    # Bounds implied by the existing voltage constraints.
                    if down_ub is not None:
                        implied_lb = math.sqrt(max(0.0, node.v_min ** 2 - float(down_ub)))
                    else:
                        implied_lb = 0.0

                    if up_ub is not None:
                        implied_ub = math.sqrt(node.v_max ** 2 + float(up_ub))
                    else:
                        implied_ub = None

                    # Do not weaken any existing variable bounds.
                    new_lb = implied_lb
                    if v.lb is not None:
                        new_lb = max(new_lb, float(v.lb))

                    new_ub = implied_ub
                    if v.ub is not None:
                        if new_ub is None:
                            new_ub = float(v.ub)
                        else:
                            new_ub = min(new_ub, float(v.ub))

                    value = float(pe.value(v))
                    if (value < new_lb - 1e-8 or (new_ub is not None and value > new_ub + 1e-8)):
                        print(
                            '[WARNING][VMAG BOUNDS] '
                            f'Initial value outside proposed bounds | '
                            f'node={i}, sm={s_m}, so={s_o}, p={p} | '
                            f'value={value:.8f} | '
                            f'lb={new_lb:.8f} | '
                            f'ub={new_ub}'
                        )

                    v.setlb(new_lb)
                    if new_ub is not None:
                        v.setub(new_ub)
                    tightened += 1
                    min_lb = min(min_lb, new_lb)
                    if new_ub is not None:
                        max_ub = max(max_ub, new_ub)

    print(
        '[FROZEN SMOPF][VMAG BOUNDS] '
        f'tightened={tightened} | '
        f'min_lb={min_lb:.6f} | '
        f'max_ub={max_ub:.6f}'
    )


def apply_tighter_vmag_sqr_bounds(model, network):

    tightened = 0

    min_lb = float('inf')
    max_ub = -float('inf')

    for i in model.nodes:

        node = network.nodes[i]

        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:

                    v_sqr = model.vmag_sqr[i, s_m, s_o, p]
                    slack_down = model.slack_v_sqr_down[i, s_m, s_o, p]
                    slack_up = model.slack_v_sqr_up[i, s_m, s_o, p]
                    down_ub = slack_down.ub
                    up_ub = slack_up.ub

                    # Bounds implied directly by the existing
                    # voltage inequality constraints:
                    # vmag_sqr + slack_down >= v_min^2
                    # vmag_sqr - slack_up   <= v_max^2

                    if down_ub is not None:
                        implied_lb = max(0.0, node.v_min ** 2 - float(down_ub))
                    else:
                        implied_lb = 0.0

                    if up_ub is not None:
                        implied_ub = (node.v_max ** 2 + float(up_ub))
                    else:
                        implied_ub = None

                    # Do not weaken pre-existing bounds.
                    new_lb = implied_lb
                    if v_sqr.lb is not None:
                        new_lb = max(new_lb, float(v_sqr.lb))

                    new_ub = implied_ub
                    if v_sqr.ub is not None:
                        if new_ub is None:
                            new_ub = float(v_sqr.ub)
                        else:
                            new_ub = min(new_ub, float(v_sqr.ub))

                    value = float(pe.value(v_sqr))
                    if (value < new_lb - 1e-8 or (new_ub is not None and value > new_ub + 1e-8)):
                        print(
                            '[WARNING][VMAG_SQR BOUNDS] '
                            f'Initial value outside proposed bounds | '
                            f'node={i}, sm={s_m}, so={s_o}, p={p} | '
                            f'value={value:.8f} | '
                            f'lb={new_lb:.8f} | '
                            f'ub={new_ub}'
                        )

                    v_sqr.setlb(new_lb)

                    if new_ub is not None:
                        v_sqr.setub(new_ub)
                    tightened += 1
                    min_lb = min(min_lb, new_lb)
                    if new_ub is not None:
                        max_ub = max(max_ub, new_ub)

    print(
        '[FROZEN SMOPF][VMAG_SQR BOUNDS] '
        f'tightened={tightened} | '
        f'min_lb={min_lb:.6f} | '
        f'max_ub={max_ub:.6f}'
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', required=True)
    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('-p', '--pickle', required=True)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, 'data', args.data_dir)
    planning_problem = SharedResourcesPlanning(data_dir, args.file)
    planning_problem.read_planning_problem()

    replay_frozen_dso_block(planning_problem, args.pickle)


if __name__ == '__main__':
    main()
