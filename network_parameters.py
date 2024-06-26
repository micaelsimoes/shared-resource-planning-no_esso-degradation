from helper_functions import *
from solver_parameters import SolverParameters


# ======================================================================================================================
#   Class NetworkParameters
# ======================================================================================================================
class NetworkParameters:

    def __init__(self):
        self.obj_type = OBJ_MIN_COST
        self.transf_reg = True
        self.es_reg = True
        self.fl_reg = True
        self.rg_curt = False
        self.l_curt = False
        self.enforce_vg = False
        self.slack_line_limits = False
        self.slack_voltage_limits = False
        self.voltage_relax = True
        self.ess_relax = True
        self.fl_relax = False
        self.node_balance_relax = False
        self.branch_current_relax = False
        self.gen_v_relax = False
        self.interface_pf_relax = False
        self.interface_ess_relax = False
        self.slacks_used = False
        self.print_to_screen = False
        self.plot_diagram = False
        self.print_results_to_file = False
        self.solver_params = SolverParameters()

    def read_parameters_from_file(self, filename):
        _read_network_parameters_from_file(self, filename)


def _read_network_parameters_from_file(parameters, filename):

    params_data = convert_json_to_dict(read_json_file(filename))

    if params_data['obj_type'] == 'COST':
        parameters.obj_type = OBJ_MIN_COST
    elif params_data['obj_type'] == 'CONGESTION_MANAGEMENT':
        parameters.obj_type = OBJ_CONGESTION_MANAGEMENT
    else:
        print('[ERROR] Invalid objective type. Exiting...')
        exit(ERROR_PARAMS_FILE)
    parameters.transf_reg = params_data['transf_reg']
    parameters.es_reg = params_data['es_reg']
    parameters.fl_reg = params_data['fl_reg']
    parameters.rg_curt = params_data['rg_curt']
    parameters.l_curt = params_data['l_curt']
    parameters.enforce_vg = params_data['enforce_vg']
    parameters.slack_line_limits = params_data['slack_line_limits']
    parameters.slack_voltage_limits = params_data['slack_voltage_limits']
    parameters.voltage_relax = params_data['voltage_relax']
    parameters.ess_relax = params_data['ess_relax']
    parameters.fl_relax = params_data['fl_relax']
    parameters.node_balance_relax = params_data['node_balance_relax']
    parameters.branch_current_relax = params_data['branch_current_relax']
    parameters.gen_v_relax = params_data['gen_v_relax']
    parameters.interface_pf_relax = params_data['interface_pf_relax']
    parameters.interface_ess_relax = params_data['interface_ess_relax']
    parameters.solver_params.read_solver_parameters(params_data['solver'])
    parameters.print_to_screen = params_data['print_to_screen']
    parameters.plot_diagram = params_data['plot_diagram']
    parameters.print_results_to_file = params_data['print_results_to_file']

    if any([parameters.slack_voltage_limits, parameters.voltage_relax, parameters.slack_line_limits,
            parameters.ess_relax, parameters.fl_relax, parameters.node_balance_relax, parameters.branch_current_relax,
            parameters.interface_pf_relax, parameters.interface_ess_relax]):
        parameters.slacks_used = True
