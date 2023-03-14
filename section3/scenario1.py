import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from section3.utils.visualization import visualize_network_with_stations_size

flux_to_refueling = 0.002
fuel_by_refueling = 28  # kg


def scenario_1(cn1, x, timesteps=3):
    w_x = x.copy()
    res = dict()
    percentage_of_hydrogen_truck_list = [0.05, 0.1, 0.15]
    for i, percentage_of_hydrogen_truck in enumerate([0.05, 0.1, 0.15]):
        if i >= timesteps:
            break
        nx.set_node_attributes(cn1, w_x, "is_Station")
        flux_by_station = {
            node: cn1.degree(node, weight="traffic flow")
            / np.ceil(cn1.degree(node) / 2)
            for node in w_x
        }
        # dictionary with the estimated demand of H2 by day in kg by station
        h2day_demand_nodes = {
            node: x_val
            * flux_by_station[node]
            * flux_to_refueling
            * percentage_of_hydrogen_truck
            * fuel_by_refueling
            for (node, x_val) in w_x.items()
        }
        # dictionary with the expected size of station by node
        h2station_nodes = {
            node: 1 * (h2day >= 1000 and h2day <= 1800)
            + 2 * (h2day > 1800 and h2day <= 2800)
            + 3 * (h2day > 2800)
            for (node, h2day) in h2day_demand_nodes.items()
        }
        # dictionary with the demand answered by our model
        h2day_nodes = {
            node: (h2station_nodes[node] == 1) * min(1000, h2day_demand_nodes[node])
            + (h2station_nodes[node] == 2) * min(2000, h2day_demand_nodes[node])
            + (h2station_nodes[node] == 3) * min(4000, h2day_demand_nodes[node])
            for node in h2station_nodes
        }
        h2profit_nodes = {
            node: (h2station_nodes[node] == 1) * (h2day_nodes[node] - 900)
            + (h2station_nodes[node] == 2) * (h2day_nodes[node] - 1600)
            + (h2station_nodes[node] == 3) * (h2day_nodes[node] - 2400)
            for node in h2station_nodes
        }
        df_coor = pd.DataFrame.from_dict(
            nx.get_node_attributes(cn1, "coordinates"),
            orient="index",
            columns=["x", "y"],
        )
        for _ in range(int(0.3 * 200)):
            max_profit = max(h2profit_nodes.values())
            best_node = [
                node
                for (node, profit) in h2profit_nodes.items()
                if profit == max_profit
            ][0]
            res[best_node] = h2station_nodes[best_node]
            df_dist = (
                (df_coor.x - df_coor.loc[best_node, "x"]) ** 2
                + (df_coor.y - df_coor.loc[best_node, "y"]) ** 2
            ).pow(0.5)
            df_nearest = df_dist[df_dist <= 20000].index.values
            for key in df_nearest:
                w_x.pop(key, None)
                h2profit_nodes.pop(key, None)
            # remove neighbors from x and h2profit_nodes
        nx.set_node_attributes(cn1, 0, "station_size")
        nx.set_node_attributes(cn1, res, "station_size")
        nx.set_node_attributes(cn1, 0, "h2day")
        nx.set_node_attributes(cn1, h2day_nodes, "h2day")
        nx.set_node_attributes(cn1, 0, "kg_profits")
        nx.set_node_attributes(cn1, h2profit_nodes, "kg_profit")
        nx.set_node_attributes(cn1, 0, "h2day_demand")
        nx.set_node_attributes(cn1, h2day_demand_nodes, "h2day_demand")
        visualize_network_with_stations_size(cn1)
        

def scenario_1_metrics(roads):
    station_size = dict(roads.nodes(data="station_size"))
    counter = Counter(station_size.values())
    print(f"#Station by size, 1: {counter[1]}, 2: {counter[2]}, 3: {counter[3]}")
    h2day_d = dict(roads.nodes(data="h2day"))
    h2day = filter(lambda item: item is not None, h2day_d.values())
    print(f"Number of tons covered by our network: {sum(h2day) / 1000}")
    kg_profit = dict(roads.nodes(data="kg_profit"))
    kg_profit = filter(lambda item: item is not None, kg_profit.values())
    print(f"Number of tons sold in profit in our network: {sum(kg_profit) / 1000}")
    h2day_demand = dict(roads.nodes(data="h2day_demand"))
    h2day_demand = list(filter(lambda item: item is not None, h2day_demand.values()))
    print(f"total estimated demand in our network: {sum(h2day_demand) / 1000}")
    fig, axs = plt.subplots((3), figsize=(16, 16))
    ax = axs[0]
    ax.hist(h2day_demand)
    ax.set_title("Total expected demand by station")
    ax.set_xlabel("Count")
    ax.set_ylabel("Expected demand")
    ax = axs[1]
    region_nodes = dict(roads.nodes(data="region_name"))
    h2day_by_region = defaultdict(int)
    for node in h2day_d:
        h2day_by_region[region_nodes[node]] += h2day_d[node] / 1000
    ax.bar(*zip(*h2day_by_region.items()))
    ax.set_ylabel("Demand in t")
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    ax = axs[2]
    stations_by_region = defaultdict(int)
    for node in h2day_d:
        stations_by_region[region_nodes[node]] += int(h2day_d[node] > 0)
    ax.bar(*zip(*stations_by_region.items()))
    ax.set_ylabel("Number of stations")
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    
    
        


