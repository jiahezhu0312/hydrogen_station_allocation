import numpy as np
import networkx as nx

from collections import defaultdict



def create_graph(model, cluster_centers, tmja):
    """Consturct basic French transportation network based on"""
    # fill the roads which do not go two ways in the dataset ?
    
    n_roads = tmja.shape[0]
    edges_with_weight = [
        (model.labels_[i], model.labels_[i + n_roads], tmja["longueur"].iloc[i])
        for i in range(tmja.shape[0])
        if model.labels_[i] != model.labels_[i + n_roads]
    ]


    edges_flow = defaultdict(int)
    for i in range(tmja.shape[0]):
        edge = (model.labels_[i], model.labels_[i + tmja.shape[0]])
        edges_flow[edge] = max(edges_flow[edge], tmja["TMJA_truck"].iloc[i])
    
    nodes_region = dict()
    for i, node in enumerate(model.labels_):
        if i < tmja.shape[0]:
            nodes_region[node] = tmja['region_name'].iloc[i]
        else:
            nodes_region[node] = tmja['region_name'].iloc[i - n_roads]


    roads = nx.Graph()
    roads.add_nodes_from(range(max(model.labels_)))
    roads.add_weighted_edges_from(edges_with_weight)
    
    nx.set_edge_attributes(roads, edges_flow, "traffic flow")

    nx.set_node_attributes(roads, cluster_centers, "coordinates")
    nx.set_node_attributes(roads, nodes_region, "region_name")
    
    #adding roads to connect the full network
    clusters = [[] for _ in range(max(model.labels_) + 1)]
    for i in range(n_roads * 2):
        clusters[model.labels_[i]].append(
            [tmja["xD"].iloc[i], tmja["yD"].iloc[i]]
            if i < n_roads
            else [tmja["xF"].iloc[i - n_roads], tmja["yF"].iloc[i - n_roads]]
        )
    cluster_coord_arr = np.array(
         [np.mean(np.array(clusters[i]), axis=0) for i in range(max(model.labels_) + 1)]
    )
    dist_matrix = np.linalg.norm(cluster_coord_arr - cluster_coord_arr[:, None], axis=-1)
    possible_edges = {index: v for index, v in np.ndenumerate(dist_matrix)}
    new_edges = [(edge[0], edge[1], possible_edges[edge]) for edge in nx.k_edge_augmentation(roads, 1, possible_edges)]
    roads.add_weighted_edges_from(new_edges)
    print("number of additional roads", len(new_edges))
    new_edges_flow = defaultdict(int)
    for node0, node1, _ in new_edges:
        degree_w = roads.degree([node0, node1], weight="traffic flow")
        degree = roads.degree([node0, node1])
        new_edges_flow[(node0, node1)] = np.mean([dw[1] / d[1] for (dw, d) in zip(degree_w, degree)])
    nx.set_edge_attributes(roads, new_edges_flow, "traffic flow")
    
    return roads