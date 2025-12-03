from copy import copy
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from model_construction_helpers import *


def get_pq_map(network, params, num_steps=8, print_pq_map=False):

    print(f'[INFO] - Determining PQ map, network {network.name}, {network.day}...')

    model, initial_solution = _get_pq_initial_solution(network, params)
    vertices_dict = _get_pq_map_vertices(network, model, num_steps, params)
    hull_dict = dict()
    inequalities = dict()
    for p in model.periods:
        convex_hull = ConvexHull(vertices_dict[p])
        inequalities[p] = _get_pq_map_inequalities(vertices_dict[p][convex_hull.vertices])
        hull_dict[p] = convex_hull

    pq_map = dict()
    for p in model.periods:
        pq_map[p] = {
            'initial_solution': initial_solution[p],
            'inequalities': inequalities[p]
        }

    if print_pq_map:
        _plot_pq_map(network, num_steps, vertices_dict, hull_dict)

    return pq_map


def _plot_pq_map(network, num_steps, vertices_dict, hull_dict):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    instants = range(network.num_instants)

    for instant in instants:

        vertices = vertices_dict[instant]
        hull = hull_dict[instant]
        polygon = vertices[hull.vertices]

        # Build 3-D polygon: [ [x,y,z], ... ]
        poly3d = [[(instant, p[0], p[1]) for p in polygon]]

        color = cm.viridis(instant / len(instants))
        face = Poly3DCollection(poly3d, alpha=0.60, facecolor=color, edgecolor=color, linewidths=0.8)
        ax.add_collection3d(face)

    ax.set_xlabel("Time, [h]")
    ax.set_ylabel("Active Power, [MW]")
    ax.set_zlabel("Reactive Power, [MVAr]")
    ax.set_xlim(0, len(instants))
    ax.xaxis.set_major_locator(MultipleLocator(4))
    plt.tight_layout()

    filename = os.path.join(network.diagrams_dir, f'{network.name}_{network.year}_{network.day}_FOR_n={num_steps}.pdf')
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.30)


def _get_pq_map_vertices(network, model, num_steps, params):

    s_base = network.baseMVA
    ref_gen_idx = network.get_reference_gen_idx()

    # New objective function (PQ maps)
    obj = model.objective.expr
    model.alpha = pe.Var(domain=pe.Reals, initialize=0.00, bounds=(-1.00, 1.00))
    model.beta = pe.Var(domain=pe.Reals, initialize=0.00, bounds=(-1.00, 1.00))
    for p in model.periods:
        for s_m in model.scenarios_market:
            omega_market = network.prob_market_scenarios[s_m]
            for s_o in model.scenarios_operation:
                omega_oper = network.prob_operation_scenarios[s_o]
                obj += model.alpha * model.pg[ref_gen_idx, s_m, s_o, p] * omega_oper * omega_market
                obj += model.beta * model.qg[ref_gen_idx, s_m, s_o, p] * omega_oper * omega_market
    model.objective.expr = obj


    vertices = dict()
    for p in model.periods:
        vertices[p] = list()

    for n in range(num_steps + 1):

        alpha = n/num_steps
        beta = 1 - alpha

        model.alpha.fix(alpha)
        model.beta.fix(beta)
        network.run_smopf(model, params, from_warm_start=True, print_header=False)
        for p in model.periods:
            pg = pe.value(model.expected_interface_pf_p[p]) * network.baseMVA
            qg = pe.value(model.expected_interface_pf_q[p]) * network.baseMVA
            vertices[p].append((pg, qg))

        model.alpha.fix(-alpha)
        model.beta.fix(-beta)
        network.run_smopf(model, params, from_warm_start=True, print_header=False)
        for p in model.periods:
            pg = pe.value(model.expected_interface_pf_p[p]) * network.baseMVA
            qg = pe.value(model.expected_interface_pf_q[p]) * network.baseMVA
            vertices[p].append((pg, qg))

    for p in model.periods:
        vertices[p] = np.array(vertices[p])

    return vertices


def _get_pq_map_inequalities(vertices):

    inequalities = []
    centroid = np.mean(vertices, axis=0)
    n = len(vertices)

    for i in range(n):

        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        # Perpendicular vector (normal)
        dx = x2 - x1
        dy = y2 - y1
        a = -dy
        b = dx
        c = a * x1 + b * y1

        # Flip direction if needed
        if a * centroid[0] + b * centroid[1] > c:
            a, b, c = -a, -b, -c

        inequalities.append({'Pg': a, 'Qg': b, 'c': c})

    return inequalities


def _get_pq_initial_solution(network, params):

    model = network.build_model(params)
    ref_node_id = network.get_reference_node_id()
    ref_gen_idx = network.get_reference_gen_idx()
    adn_node_idx = network.get_node_idx(ref_node_id)

    # Add expected interface power flow variables
    model.expected_interface_vmag = pe.Var(model.periods, domain=pe.NonNegativeReals, initialize=1.00)
    model.expected_interface_pf_p = pe.Var(model.periods, domain=pe.Reals, initialize=0.00)
    model.expected_interface_pf_q = pe.Var(model.periods, domain=pe.Reals, initialize=0.00)
    model.interface_expected_values = pe.ConstraintList()
    for p in model.periods:
        expected_vmag = 0.00
        expected_pf_p = 0.00
        expected_pf_q = 0.00
        for s_m in model.scenarios_market:
            omega_market = network.prob_market_scenarios[s_m]
            for s_o in model.scenarios_operation:
                omega_oper = network.prob_operation_scenarios[s_o]
                expected_vmag += omega_market * omega_oper * model.e[adn_node_idx, s_m, s_o, p]
                expected_pf_p += omega_market * omega_oper * model.pg[ref_gen_idx, s_m, s_o, p]
                expected_pf_q += omega_market * omega_oper * model.qg[ref_gen_idx, s_m, s_o, p]
        model.interface_expected_values.add(model.expected_interface_vmag[p] <= expected_vmag + SMALL_TOLERANCE)
        model.interface_expected_values.add(model.expected_interface_vmag[p] >= expected_vmag - SMALL_TOLERANCE)
        model.interface_expected_values.add(model.expected_interface_pf_p[p] <= expected_pf_p + EQUALITY_TOLERANCE)
        model.interface_expected_values.add(model.expected_interface_pf_p[p] >= expected_pf_p - EQUALITY_TOLERANCE)
        model.interface_expected_values.add(model.expected_interface_pf_q[p] <= expected_pf_q + EQUALITY_TOLERANCE)
        model.interface_expected_values.add(model.expected_interface_pf_q[p] >= expected_pf_q - EQUALITY_TOLERANCE)

    # Regularization -- Added to OF to minimize deviations from scenarios to expected values
    s_base = network.baseMVA
    obj = model.objective.expr
    model.penalty_regularization = pe.Var(domain=pe.NonNegativeReals)
    model.penalty_regularization.fix(PENALTY_REGULARIZATION)
    for p in model.periods:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                obj += model.penalty_regularization * (model.e[adn_node_idx, s_m, s_o, p] - model.expected_interface_vmag[p]) ** 2
                obj += model.penalty_regularization * s_base * (model.pg[ref_gen_idx, s_m, s_o, p] - model.expected_interface_pf_p[p]) ** 2
                obj += model.penalty_regularization * s_base * (model.qg[ref_gen_idx, s_m, s_o, p] - model.expected_interface_pf_q[p]) ** 2
    model.objective.expr = obj

    network.run_smopf(model, params, from_warm_start=True, print_header=False)

    solution = dict()
    for p in model.periods:
        solution[p] = {
            'Vg': pe.value(model.expected_interface_vmag[p]),
            'Pg': pe.value(model.expected_interface_pf_p[p]) * s_base,
            'Qg': pe.value(model.expected_interface_pf_q[p]) * s_base
        }

    return model, solution


def update_of_to_settlement(network, model, params):

    s_base = network.baseMVA
    ref_node_id = network.get_reference_node_id()
    ref_node_idx = network.get_node_idx(ref_node_id)
    ref_gen_idx = network.get_reference_gen_idx()

    # New objective function (settlement)
    obj = copy(model.objective.expr)
    model.penalty_regularization = pe.Param(initialize=PENALTY_REGULARIZATION)
    model.interface_vmag_req = pe.Var(model.periods, domain=pe.NonNegativeReals, initialize=1.00)
    model.interface_pf_p_req = pe.Var(model.periods, domain=pe.Reals, initialize=0.00)
    model.interface_pf_q_req = pe.Var(model.periods, domain=pe.Reals, initialize=0.00)
    for s_m in model.scenarios_market:
        for s_o in model.scenarios_operation:
            for p in model.periods:
                obj += model.penalty_regularization * (model.e[ref_node_idx, s_m, s_o, p] - model.interface_vmag_req[p]) ** 2
                obj += model.penalty_regularization * s_base * (model.pg[ref_gen_idx, s_m, s_o, p] - model.interface_pf_p_req[p]) ** 2
                obj += model.penalty_regularization * s_base * (model.qg[ref_gen_idx, s_m, s_o, p] - model.interface_pf_q_req[p]) ** 2
    model.objective.expr = obj
