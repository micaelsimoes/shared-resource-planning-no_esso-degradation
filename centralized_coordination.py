from copy import copy
from definitions import *


def combine_networks(transmission_network, distribution_networks):

    print('[INFO] - Combining networks...')

    combined_network = copy(transmission_network)
    combined_network.name += '_combined'

    tn_node_mapping = dict()
    for year in combined_network.years:
        tn_node_mapping[year] = dict()
        for day in combined_network.days:
            tn_node_mapping[year][day] = dict()

    # Reassign IDs
    for year in combined_network.years:
        for day in combined_network.days:



            # Nodes
            new_nodes = list()
            for node in combined_network.network[year][day].nodes:
                new_node = copy(node)
                new_node.old_bus_i = node.bus_i
                new_node.bus_i = f'TN_{new_node.old_bus_i}'
                tn_node_mapping[year][day][new_node.old_bus_i] = new_node.bus_i
                new_nodes.append(new_node)
            combined_network.network[year][day].nodes = new_nodes

            # Loads (Only include loads that are NOT ADNs)
            new_loads = list()
            for load in combined_network.network[year][day].loads:
                if load.bus not in combined_network.network[year][day].active_distribution_network_nodes:
                    new_load = copy(load)
                    new_load.old_load_id = load.load_id
                    new_load.load_id = f'TN_{new_load.old_load_id}'
                    new_load.bus = tn_node_mapping[year][day][load.bus]
                    new_loads.append(new_load)
            combined_network.network[year][day].loads = new_loads

            # Branches
            new_branches = list()
            for branch in combined_network.network[year][day].branches:
                new_branch = copy(branch)
                new_branch.old_branch_id = branch.branch_id
                new_branch.branch_id = f'TN_{new_branch.old_branch_id}'
                new_branch.fbus = tn_node_mapping[year][day][branch.fbus]
                new_branch.tbus = tn_node_mapping[year][day][branch.tbus]
                new_branches.append(new_branch)
            combined_network.network[year][day].branches = new_branches

            # Generators
            new_generators = list()
            for generator in combined_network.network[year][day].generators:
                new_generator = copy(generator)
                new_generator.old_gen_id = generator.gen_id
                new_generator.gen_id = f'TN_{new_generator.old_gen_id}'
                new_generator.bus = tn_node_mapping[year][day][generator.bus]
                new_generators.append(new_generator)
            combined_network.network[year][day].generators = new_generators

            # Energy Storages
            new_energy_storages = list()
            for energy_storage in combined_network.network[year][day].energy_storages:
                new_energy_storage = copy(energy_storage)
                new_energy_storage.old_storage_id = energy_storage.storage_id
                new_energy_storage.storage_id = f'TN_{new_energy_storage.old_storage_id}'
                new_energy_storage.bus = tn_node_mapping[year][day][energy_storage.bus]
                new_energy_storages.append(new_energy_storage)
            combined_network.network[year][day].energy_storages = new_energy_storages

            # Shared Energy Storages (empty)
            combined_network.network[year][day].shared_energy_storages = list()

            # ADN nodes (empty)
            combined_network.network[year][day].active_distribution_network_nodes = list()

    # Add ADNs to combined network
    for adn_node_id in distribution_networks:
        for year in combined_network.years:
            for day in combined_network.days:

                distribution_network = distribution_networks[adn_node_id].network[year][day]
                ref_node_id = distribution_network.get_reference_node_id()
                local_node_mapping = dict()

                # Nodes (do not add reference bus)
                for node in distribution_network.nodes:
                    if node.type != BUS_REF:
                        new_node = copy(node)
                        new_node.old_bus_i = node.bus_i
                        new_node.bus_i = f'ADN_{adn_node_id}_{new_node.old_bus_i}'
                        local_node_mapping[new_node.old_bus_i] = new_node.bus_i
                        combined_network.network[year][day].nodes.append(new_node)

                # Loads
                for load in distribution_network.loads:
                    new_load = copy(load)
                    new_load.old_load_id = load.load_id
                    new_load.load_id = f'ADN_{adn_node_id}_{new_load.old_load_id}'
                    new_load.bus = local_node_mapping[load.bus]
                    combined_network.network[year][day].loads.append(new_load)

                # Branches (update those connected to ref node)
                for branch in distribution_network.branches:
                    new_branch = copy(branch)
                    new_branch.old_branch_id = branch.branch_id
                    new_branch.branch_id = f'ADN_{adn_node_id}_{new_branch.old_branch_id}'
                    if new_branch.fbus == ref_node_id:
                        new_branch.fbus = tn_node_mapping[year][day][adn_node_id]
                    else:
                        new_branch.fbus = local_node_mapping[branch.fbus]
                    if new_branch.tbus == ref_node_id:
                        new_branch.tbus = tn_node_mapping[year][day][adn_node_id]
                    else:
                        new_branch.tbus = local_node_mapping[branch.tbus]
                    combined_network.network[year][day].branches.append(new_branch)

                # Generators (do not add reference generator)
                for generator in distribution_network.generators:
                    new_generator = copy(generator)
                    if new_generator.bus != ref_node_id:
                        new_generator.old_gen_id = generator.gen_id
                        new_generator.gen_id = f'ADN_{adn_node_id}_{new_generator.old_gen_id}'
                        new_generator.bus = local_node_mapping[new_generator.bus]
                        combined_network.network[year][day].generators.append(new_generator)

                for energy_storage in distribution_network.energy_storages:
                    new_energy_storage = copy(energy_storage)
                    new_energy_storage.old_storage_id = energy_storage.storage_id
                    new_energy_storage.storage_id = f'ADN_{adn_node_id}_{new_energy_storage.old_storage_id}'
                    new_energy_storage.bus = local_node_mapping[energy_storage.bus]
                    combined_network.network[year][day].energy_storages.append(new_energy_storage)

    return combined_network
