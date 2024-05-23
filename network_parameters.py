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
        self.relax_equalities = False
        self.slack_flexibility = False      # Slacks only applicable if relax_equalities are "False"
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
    parameters.transf_reg = bool(params_data['transf_reg'])
    parameters.es_reg = bool(params_data['es_reg'])
    parameters.fl_reg = bool(params_data['fl_reg'])
    parameters.rg_curt = bool(params_data['rg_curt'])
    parameters.l_curt = bool(params_data['l_curt'])
    parameters.enforce_vg = bool(params_data['enforce_vg'])
    parameters.relax_equalities = bool(params_data['relax_equalities'])
    if not parameters.relax_equalities:
        parameters.slacks_flexibility = bool(params_data['slacks_flexibility'])
        parameters.slacks_ess_charging = bool(params_data['slacks_ess_charging'])
        parameters.slacks_ess_soc = bool(params_data['slacks_ess_soc'])
        parameters.slacks_ess_complementarity = bool(params_data['slacks_ess_complementarity'])
        parameters.slacks_ess_day_balance = bool(params_data['slacks_ess_day_balance'])
    parameters.solver_params.read_solver_parameters(params_data['solver'])
    parameters.print_to_screen = params_data['print_to_screen']
    parameters.plot_diagram = params_data['plot_diagram']
    parameters.print_results_to_file = params_data['print_results_to_file']
