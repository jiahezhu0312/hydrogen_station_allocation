import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(roads):
    for node in roads.nodes:
        if "is_Station" in roads.nodes[node]:
            return visualize_network_with_stations(roads)
            
        else:
            visualize_network_no_stations(roads)
        break


def visualize_network_no_stations(roads):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    is_OD = np.array(list(dict(roads.nodes(data="is_OD")).values()))
    # is_nor = np.array(list(dict(roads.nodes(data='region_name')).values() ))
    cluster_centers = dict(roads.nodes(data="coordinates"))
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": [
            "red" if n else "blue" for n in is_OD
        ],  # ['red' if n=='Normandie' else 'blue' for n in is_nor],#
        "node_size": is_OD * 50 + 10,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()


def visualize_network_with_stations(roads):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    is_OD = np.array(list(dict(roads.nodes(data="is_OD")).values()))
    # is_nor = np.array(list(dict(roads.nodes(data='region_name')).values() ))
    cluster_centers = dict(roads.nodes(data="coordinates"))
    results = dict(roads.nodes(data="is_Station"))
    candidate_sites = list(results.values())
    print(
        f"We allocate {len([ele for ele in candidate_sites if ele >0])} stations in this network"
    )

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    node_color = ["red" if n else "blue" for n in is_OD]
    for ind in range(len(candidate_sites)):
        if node_color[ind] != "red":
            if candidate_sites[ind] > 0:
                node_color[ind] = "green"
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": node_color,
        "node_size": (is_OD + np.array(candidate_sites)) * 50,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()

def visualize_stations(roads):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    cluster_centers = dict(roads.nodes(data="coordinates"))
    results = dict(roads.nodes(data="is_Station"))
    candidate_sites = list(results.values())
    print(
        f"We allocate {len([ele for ele in candidate_sites if ele >0])} stations in this network"
    )

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    node_color = ["green" if n else "grey" for n in candidate_sites]
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": node_color,
        "node_size": (np.array(candidate_sites)) * 50,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()



def visualize_network(roads):
    for node in roads.nodes:
        if "is_Station" in roads.nodes[node]:
            return visualize_network_with_stations(roads)
            
        else:
            visualize_network_no_stations(roads)
        break


def visualize_network_no_stations(roads):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    is_OD = np.array(list(dict(roads.nodes(data="is_OD")).values()))
    # is_nor = np.array(list(dict(roads.nodes(data='region_name')).values() ))
    cluster_centers = dict(roads.nodes(data="coordinates"))
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": [
            "red" if n else "blue" for n in is_OD
        ],  # ['red' if n=='Normandie' else 'blue' for n in is_nor],#
        "node_size": is_OD * 50 + 10,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()


def visualize_network_with_stations(roads):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    is_OD = np.array(list(dict(roads.nodes(data="is_OD")).values()))
    # is_nor = np.array(list(dict(roads.nodes(data='region_name')).values() ))
    cluster_centers = dict(roads.nodes(data="coordinates"))
    results = dict(roads.nodes(data="is_Station"))
    candidate_sites = list(results.values())
    print(
        f"We allocate {len([ele for ele in candidate_sites if ele >0])} stations in this network"
    )

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    node_color = ["red" if n else "blue" for n in is_OD]
    for ind in range(len(candidate_sites)):
        if node_color[ind] != "red":
            if candidate_sites[ind] > 0:
                node_color[ind] = "green"
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": node_color,
        "node_size": (is_OD + np.array(candidate_sites)) * 50,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()

def visualize_stations(roads):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    cluster_centers = dict(roads.nodes(data="coordinates"))
    results = dict(roads.nodes(data="is_Station"))
    candidate_sites = list(results.values())
    print(
        f"We allocate {len([ele for ele in candidate_sites if ele >0])} stations in this network"
    )

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    node_color = ["green" if n else "grey" for n in candidate_sites]
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": node_color,
        "node_size": (np.array(candidate_sites)) * 50,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()



def visualize_network(roads):
    for node in roads.nodes:
        if "is_Station" in roads.nodes[node]:
            return visualize_network_with_stations(roads)
            
        else:
            visualize_network_no_stations(roads)
        break

def visualize_network_no_stations(roads):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    is_OD = np.array(list(dict(roads.nodes(data="is_OD")).values()))
    # is_nor = np.array(list(dict(roads.nodes(data='region_name')).values() ))
    cluster_centers = dict(roads.nodes(data="coordinates"))
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": [
            "red" if n else "blue" for n in is_OD
        ],  # ['red' if n=='Normandie' else 'blue' for n in is_nor],#
        "node_size": is_OD * 50 + 10,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()

def visualize_network_with_stations(roads):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    is_OD = np.array(list(dict(roads.nodes(data="is_OD")).values()))
    # is_nor = np.array(list(dict(roads.nodes(data='region_name')).values() ))
    cluster_centers = dict(roads.nodes(data="coordinates"))
    results = dict(roads.nodes(data="is_Station"))
    candidate_sites = list(results.values())
    print(
        f"We allocate {len([ele for ele in candidate_sites if ele >0])} stations in this network"
    )
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    node_color = ["red" if n else "blue" for n in is_OD]
    for ind in range(len(candidate_sites)):
        if node_color[ind] != "red":
            if candidate_sites[ind] > 0:
                node_color[ind] = "green"
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": node_color,
        "node_size": (is_OD + np.array(candidate_sites)) * 50,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )
    plt.show()
def visualize_network_scenario2(roads):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    is_OD = np.array(list(dict(roads.nodes(data="is_OD")).values()))
    # is_nor = np.array(list(dict(roads.nodes(data='region_name')).values() ))
    cluster_centers = dict(roads.nodes(data="coordinates"))
    competitors = dict(roads.nodes(data="competitor"))
    airliquide = dict(roads.nodes(data="airliquide"))

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    node_color = []
    node_size = []
    for ind in range(len(competitors)):
        if is_OD[ind]:
            node_color.append("red")
            node_size.append(20)
        elif competitors[ind] > 0:
            node_color.append("blue")
            node_size.append(20 * competitors[ind])
        elif airliquide[ind] > 0:
            node_color.append("green")
            node_size.append(20 * airliquide[ind])
        else:
            node_color.append("grey")
            node_size.append(0)

    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": node_color,
        "node_size": node_size,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()
    
def visualize_scenario1_stations(roads, prefix):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    cluster_centers = dict(roads.nodes(data="coordinates"))
    results = dict(roads.nodes(data=prefix + "station_size"))
    candidate_sites = list(results.values())
    print(candidate_sites)
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    node_color = ["green" if n else "blue" for n in candidate_sites]
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": node_color,
        "node_size": (np.array(candidate_sites)) * 20,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()

    return nstations

    
def visualize_network_with_stations_size(roads, prefix):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    is_OD = np.array(list(dict(roads.nodes(data="is_OD")).values()))
    # is_nor = np.array(list(dict(roads.nodes(data='region_name')).values() ))
    cluster_centers = dict(roads.nodes(data="coordinates"))
    results = dict(roads.nodes(data=prefix + "station_size"))
    candidate_sites = list(results.values())
    nstations = len([ele for ele in candidate_sites if ele >0])
    print(
        f"We allocate {nstations} stations in this network"
    )

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

    node_color = ["red" if n else "blue" for n in is_OD]
    for ind in range(len(candidate_sites)):
        if node_color[ind] != "red":
            if candidate_sites[ind] > 0:
                node_color[ind] = "green"
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": node_color,
        "node_size": (is_OD + np.array(candidate_sites)) * 20,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()

    return nstations


def visualize_scenario2(roads, prefix):
    edges, weights = zip(*nx.get_edge_attributes(roads, "traffic flow").items())
    weights = np.array(weights) + 1
    weights = np.log(weights)
    is_OD = np.array(list(dict(roads.nodes(data="is_OD")).values()))
    # is_nor = np.array(list(dict(roads.nodes(data='region_name')).values() ))
    cluster_centers = dict(roads.nodes(data="coordinates"))
    results = dict(roads.nodes(data=prefix + "station_size"))
    candidate_sites = list(results.values())
    nstations = len([ele for ele in candidate_sites if ele >0])
    print(
        f"We allocate {nstations} stations in this network"
    )

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    airliquide = dict(roads.nodes(data=prefix + "airliquide"))
    node_color = []
    node_size = []
    for ind in range(len(airliquide)):
        if not(airliquide[ind]):
            node_color.append("purple")
            node_size.append(20 * candidate_sites[ind])
        elif airliquide[ind]:
            node_color.append("green")
            node_size.append(20 * candidate_sites[ind])
        else:
            node_color.append("grey")
            node_size.append(0)
    
    options = {
        "edge_color": weights,
        "width": 4,
        "edge_cmap": plt.cm.Wistia,
        "with_labels": False,
        "node_color": node_color,
        "node_size": (np.array(candidate_sites)) * 20,
    }
    nx.draw(
        roads,
        pos=cluster_centers,
        **options,
    )

    plt.show()

    return nstations
