import networkx as nx
import numpy as np
import pandas as pd
from section3.utils.visualization import visualize_network_with_stations_size

flux_to_refueling = 0.008
percentage_of_hydrogen_truck = 0.1
traffic_increase = 1.06
fuel_by_refueling = 28  # kg


def scenario_1(cn1, x):
    w_x = x.copy()
    res = dict()
    for years in [10, 15, 20]:
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
            * traffic_increase**years
            for (node, x_val) in w_x.items()
        }
        # dictionary with the expected size of station by node
        h2station_nodes = {
            node: 1 * (h2day >= 900 and h2day <= 1600)
            + 2 * (h2day > 1600 and h2day <= 2400)
            + 3 * (h2day > 2400)
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
            df_nearest = df_dist[df_dist <= 30000].index.values
            for key in df_nearest:
                w_x.pop(key, None)
                h2profit_nodes.pop(key, None)
            # remove neighbors from x and h2profit_nodes
        nx.set_node_attributes(cn1, 0, "station_size")
        nx.set_node_attributes(cn1, res, "station_size")
        visualize_network_with_stations_size(cn1)
        


