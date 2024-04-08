import os
from network import Network
from network_parameters import NetworkParameters


# ======================================================================================================================
#   Class NETWORK DATA -- Contains information of the Network over the planning period (years, days)
# ======================================================================================================================
class NetworkData:

    def __init__(self):
        self.name = str()
        self.data_dir = str()
        self.results_dir = str()
        self.diagrams_dir = str()
        self.years = dict()
        self.days = dict()
        self.num_instants = int()
        self.discount_factor = float()
        self.network = dict()
        self.params_file = str()
        self.params = NetworkParameters()
        self.cost_energy_p = dict()
        self.prob_market_scenarios = dict()
        self.is_transmission = False

    def build_model(self):
        network_models = dict()
        for year in self.years:
            network_models[year] = dict()
            for day in self.days:
                network_models[year][day] = self.network[year][day].build_model(self.params)
        return network_models

    def optimize(self, model, from_warm_start=False):
        print(f'[INFO] \t\t\t - Running SMOPF, Network {self.name}...')
        results = dict()
        for year in self.years:
            results[year] = dict()
            for day in self.days:
                print(f'[INFO] \t\t\t\t - Year {year}, Day {day}...')
                results[year][day] = self.network[year][day].run_smopf(model[year][day], self.params, from_warm_start=from_warm_start)
        return results

    def update_model_with_candidate_solution(self, model, candidate_solution):
        _update_model_with_candidate_solution(self, model, candidate_solution)

    def read_network_data(self):
        _read_network_data(self)

    def read_network_parameters(self):
        filename = os.path.join(self.data_dir, self.name, self.params_file)
        self.params.read_parameters_from_file(filename)


# ======================================================================================================================
#  NETWORK DATA read function
# ======================================================================================================================
def _read_network_data(network_planning):

    for year in network_planning.years:

        network_planning.network[year] = dict()

        for day in network_planning.days:

            # Create Network object
            network_planning.network[year][day] = Network()
            network_planning.network[year][day].name = network_planning.name
            network_planning.network[year][day].data_dir = network_planning.data_dir
            network_planning.network[year][day].results_dir = network_planning.results_dir
            network_planning.network[year][day].diagrams_dir = network_planning.diagrams_dir
            network_planning.network[year][day].year = int(year)
            network_planning.network[year][day].day = day
            network_planning.network[year][day].num_instants = network_planning.num_instants
            network_planning.network[year][day].is_transmission = network_planning.is_transmission
            network_planning.network[year][day].prob_market_scenarios = network_planning.prob_market_scenarios
            network_planning.network[year][day].cost_energy_p = network_planning.cost_energy_p[year][day]
            network_planning.network[year][day].operational_data_file = f'{network_planning.name}_{year}.xlsx'

            # Read info from file(s)
            network_planning.network[year][day].read_network_from_json_file()
            network_planning.network[year][day].read_network_operational_data_from_file()

            if network_planning.params.print_to_screen:
                network_planning.network[year][day].print_network_to_screen()
            if network_planning.params.plot_diagram:
                network_planning.network[year][day].plot_diagram()


# ======================================================================================================================
#  OTHER (auxiliary) functions
# ======================================================================================================================
def _update_model_with_candidate_solution(network, model, candidate_solution):
    if network.is_transmission:
        for year in network.years:
            for day in network.days:
                s_base = network.network[year][day].baseMVA
                for node_id in network.active_distribution_network_nodes:
                    shared_ess_idx = network.network[year][day].get_shared_energy_storage_idx(node_id)
                    model[year][day].shared_es_s_rated[shared_ess_idx].fix(abs(candidate_solution[node_id][year]['s']) / s_base)
                    model[year][day].shared_es_e_rated[shared_ess_idx].fix(abs(candidate_solution[node_id][year]['e']) / s_base)
    else:
        tn_node_id = network.tn_connection_nodeid
        for year in network.years:
            for day in network.days:
                s_base = network.network[year][day].baseMVA
                ref_node_id = network.network[year][day].get_reference_node_id()
                shared_ess_idx = network.network[year][day].get_shared_energy_storage_idx(ref_node_id)
                model[year][day].shared_es_s_rated[shared_ess_idx].fix(abs(candidate_solution[tn_node_id][year]['s']) / s_base)
                model[year][day].shared_es_e_rated[shared_ess_idx].fix(abs(candidate_solution[tn_node_id][year]['e']) / s_base)
