import networkx as nx
from part2.graph.construct_network import construct_network
from part2.model.FRLM import FRLM
from part2.utils.visualization import visualize_network

def locate_stations(tmja, ald, p, R):
    cn1 = construct_network(tmja, ald)
    x, y, demand_satisfied = FRLM(cn1, p, R) # p number of stations R= autonomie
    nx.set_node_attributes(cn1, 0, "is_Station")
    nx.set_node_attributes(cn1, x, "is_Station")
    visualize_network(cn1)
    cn1.nodes.data

    return cn1


def station_info(cn1):
    # percentage of demand answered
    # count by region

    pass

def station_assign_size(cn1):
    pass