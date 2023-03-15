import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from utils.visualization import visualize_scenario2

flux_to_refueling = 0.0012
fuel_by_refueling = 32  # kg


def scenario_2(cn1, x, timesteps=4, visualization=False, metrics=False):
    w_x = x.copy()
    res_size = dict()
    res_h2day = dict()
    res_profit = dict()
    res_airliquide = dict()
    percentage_of_hydrogen_truck_list = [0.04, 0.09, 0.168, 0.24]
    demand_treshold = [150000, 370000, 7500000, 1100000]
    cur_demand_sum = 0
    for i, percentage_of_hydrogen_truck in enumerate(percentage_of_hydrogen_truck_list):
        if i >= timesteps:
            break
        flux_by_station = {
            node: cn1.degree(node, weight="traffic flow")
            / np.ceil(cn1.degree(node) / 2)
            for node in x
        }
        # dictionary with the estimated demand of H2 by day in kg by station
        h2day_demand_all_nodes = {
            node: x_val
            * flux_by_station[node]
            * flux_to_refueling
            * percentage_of_hydrogen_truck
            * fuel_by_refueling
            for (node, x_val) in x.items()
        }
        h2station_all_nodes = {
            node: 1 * (h2day >= 900 and h2day <= 1600)
            + 2 * (h2day > 1600 and h2day <= 2400)
            + 3 * (h2day > 2400)
            for (node, h2day) in h2day_demand_all_nodes.items()
        }
        h2day_all_nodes = {
            node: (h2station_all_nodes[node] == 1) * min(1000, h2day_demand_all_nodes[node])
            + (h2station_all_nodes[node] == 2) * min(2000, h2day_demand_all_nodes[node])
            + (h2station_all_nodes[node] == 3) * min(4000, h2day_demand_all_nodes[node])
            for node in h2station_all_nodes
        }
        h2profit_all_nodes = {
            node: (h2station_all_nodes[node] == 1) * (h2day_all_nodes[node] - 900)
            + (h2station_all_nodes[node] == 2) * (h2day_all_nodes[node] - 1600)
            + (h2station_all_nodes[node] == 3) * (h2day_all_nodes[node] - 2400)
            for node in h2station_all_nodes
        }
        for node in res_size:
            cur_demand_sum -= res_h2day[node]
            res_size[node] = h2station_all_nodes[node]
            res_h2day[node] = h2day_all_nodes[node]
            res_profit[node] = h2profit_all_nodes[node]
            cur_demand_sum += res_h2day[node]
        
        
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
            node: 1 * (h2day >= 900 and h2day <= 1600)
            + 2 * (h2day > 1600 and h2day <= 2400)
            + 3 * (h2day > 2400)
            for (node, h2day) in h2day_demand_nodes.items()
        }
        h2station_nodes = {k: v for k, v in h2station_nodes.items() if v != 0}
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
        while cur_demand_sum < demand_treshold[i] and len(h2profit_nodes.values()):
            max_profit = max(h2profit_nodes.values())
            best_node = [
                node
                for (node, profit) in h2profit_nodes.items()
                if profit == max_profit
            ][0]
            res_size[best_node] = h2station_nodes[best_node]
            res_h2day[best_node] = h2day_nodes[best_node]
            res_profit[best_node] = h2profit_nodes[best_node]
            res_airliquide[best_node] = (random.random() < 0.5)
            cur_demand_sum += h2day_nodes[best_node]
            df_dist = (
                (df_coor.x - df_coor.loc[best_node, "x"]) ** 2
                + (df_coor.y - df_coor.loc[best_node, "y"]) ** 2
            ).pow(0.5)
            df_nearest = df_dist[df_dist <= 15000].index.values
            for key in df_nearest:
                w_x.pop(key, None)
                h2profit_nodes.pop(key, None)
            # remove neighbors from x and h2profit_nodes
        nx.set_node_attributes(cn1, 0, "S3P2_station_size")
        nx.set_node_attributes(cn1, res_size, "S3P2_station_size")
        nx.set_node_attributes(cn1, 0, "S3P2_h2day")
        nx.set_node_attributes(cn1, res_h2day, "S3P2_h2day")
        nx.set_node_attributes(cn1, 0, "S3P2_kg_profit")
        nx.set_node_attributes(cn1, res_profit, "S3P2_kg_profit")
        nx.set_node_attributes(cn1, 0, "S3P2_h2day_demand")
        nx.set_node_attributes(cn1, h2day_demand_all_nodes, "S3P2_h2day_demand")
        nx.set_node_attributes(cn1, False, "S3P2_airliquide")
        nx.set_node_attributes(cn1, res_airliquide, "S3P2_airliquide")
        nx.set_node_attributes(cn1, x, "is_Station")
        if visualization:
            visualize_scenario2(cn1, 'S3P2_')
        if metrics:
            scenario_2_metrics(cn1)

    #do scenario in the for loop + demand decrease in size with time
    #question on scaling for the last date ?
        
        

def scenario_2_metrics(roads):
    station_size = dict(roads.nodes(data="S3P2_station_size"))
    counter = Counter(station_size.values())
    print(f"#Station by size, 1: {counter[1]}, 2: {counter[2]}, 3: {counter[3]}")
    h2day_d = dict(roads.nodes(data="S3P2_h2day"))
    h2day = list(filter(lambda item: item is not None, h2day_d.values()))
    print(f"Number of tons covered by our network: {sum(h2day) / 1000}")
    kg_profit = dict(roads.nodes(data="S3P2_kg_profit"))
    kg_profit = filter(lambda item: item is not None, kg_profit.values())
    print(f"Number of tons sold in profit in our network: {sum(kg_profit) / 1000}")
    h2day_demand = dict(roads.nodes(data="S3P2_h2day_demand"))
    h2day_demand = list(filter(lambda item: item is not None, h2day_demand.values()))
    print(f"total estimated demand in our network: {sum(h2day_demand) / 1000}")
    fig, axs = plt.subplots((3), figsize=(16, 16))
    ax = axs[0]
    ax.hist([x for x in h2day if x > 0])
    ax.set_title("Expected output by station")
    ax.set_xlabel("Expected output by station")
    ax.set_ylabel("Count")
    ax.grid()
    region_nodes = dict(roads.nodes(data="region_name"))
    ax = axs[1]
    size_list = ["small", "medium", "large"]
    size_values = [counter[1], counter[2], counter[3]]
    ax.bar(size_list, size_values)
    ax.set_ylabel("Number of stations")
    ax.set_xlabel("Station size")
    fig.tight_layout()
    ax = axs[2]
    stations_by_region = defaultdict(int)
    for node in h2day_d:
        stations_by_region[region_nodes[node]] += int(h2day_d[node] > 0)
    ax.bar(*zip(*stations_by_region.items()))
    ax.set_ylabel("Number of stations")
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()