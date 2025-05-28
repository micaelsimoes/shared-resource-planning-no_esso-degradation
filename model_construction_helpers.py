import pyomo.environ as pe
from definitions import *


# Voltage constraints, e
def voltage_rule_e(model, i, s_m, s_o, p, params):
    e_val = model.e[i, s_m, s_o, p]
    if params.slacks.grid_operation.voltage:
        e_val += model.slack_e[i, s_m, s_o, p]
    return model.e_actual[i, s_m, s_o, p] == e_val


# Voltage constraints, f
def voltage_rule_f(model, i, s_m, s_o, p, params):
    f_val = model.f[i, s_m, s_o, p]
    if params.slacks.grid_operation.voltage:
        f_val += model.slack_f[i, s_m, s_o, p]
    return model.f_actual[i, s_m, s_o, p] == f_val


# Voltage constraints, magnitude
def voltage_magnitude_rule(model, i, s_m, s_o, p, network, params):
    node = network.nodes[i]
    e = model.e[i, s_m, s_o, p]
    f = model.f[i, s_m, s_o, p]
    vmag_sq = e ** 2 + f ** 2
    if node.type == BUS_PV and params.enforce_vg:
        vg = network.generators[network.get_gen_idx(node.bus_i)].vg[p]
        return pe.inequality(vg ** 2 - EQUALITY_TOLERANCE, vmag_sq, vg ** 2 + EQUALITY_TOLERANCE)
    else:
        return pe.inequality(node.v_min ** 2, vmag_sq, node.v_max ** 2)



