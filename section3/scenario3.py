import pyproj
import numpy as np
from collections import Counter, defaultdict
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
from utils.summary_statistics import phase_summary, print_table
from utils.output_dataframe import df_phase

class Scenario3():
    def __init__(self, data_path, cn, x) -> None:

      
        existent_stations = self._read_existent_stations(data_path)

        self.conversion_planned = self._match_stations(cn, existent_stations)
        nx.set_node_attributes(cn, self.conversion_planned, "S3P3_conversion_planned")

        self.define_station_size(cn,x)
        rival = self._check_rival(cn)
        nx.set_node_attributes(cn, rival, "S3P3_rival_station")

        self.graph = cn
        self.ncompetitor_stations = len(existent_stations)
    
        self.rival_stations = dict(self.graph.nodes(data='S3P3_rival_station'))

    def define_station_size(self, cn1, x, timesteps=4):
        flux_to_refueling = 0.0012
        fuel_by_refueling = 32
        base_year = 2025# k
        w_x = x.copy()
        res_size = dict()
        res_h2day = dict()
        res_profit = dict()
        percentage_of_hydrogen_truck_list = [0.04, 0.09, 0.168, 0.24]
        demand_treshold = [150000, 370000, 7500000, 1100000]
        station_size_all_phase = []
        summary = []
        df_all_phase = []
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
                * fuel_by_refueling / (1 + self.conversion_planned[node] + 0.5 * sum(self.conversion_planned[nei] for nei in cn1.neighbors(node)))
                for (node, x_val) in x.items()
            }
            h2station_all_nodes = {
                node: 1 * (h2day >= 1000 and h2day <= 1800)
                + 2 * (h2day > 1800 and h2day <= 3000)
                + 3 * (h2day > 3000)
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
                * fuel_by_refueling / (1 + self.conversion_planned[node] + 0.5 * sum(self.conversion_planned[nei] for nei in cn1.neighbors(node)))
                for (node, x_val) in w_x.items()
            }
            # dictionary with the expected size of station by node
            h2station_nodes = {
                node: 1 * (h2day >= 1000 and h2day <= 1800)
                + 2 * (h2day > 1800 and h2day <= 3000)
                + 3 * (h2day > 3000)
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
            nx.set_node_attributes(cn1, 0, "S3P3_station_size")
            nx.set_node_attributes(cn1, res_size, "S3P3_station_size")
            nx.set_node_attributes(cn1, 0, "S3P3_h2day")
            nx.set_node_attributes(cn1, res_h2day, "S3P3_h2day")
            nx.set_node_attributes(cn1, 0, "S3P3_kg_profit")
            nx.set_node_attributes(cn1, res_profit, "S3P3_kg_profit")
            nx.set_node_attributes(cn1, 0, "S3P3_h2day_demand")
            nx.set_node_attributes(cn1, h2day_demand_all_nodes, "S3P3_h2day_demand")
            nx.set_node_attributes(cn1, x, "is_Station")
            station_size_all_phase.append(dict(cn1.nodes(data='S3P3_station_size')))
            summary.append(self.call_phase_summary(cn1, station_size_all_phase, base_year + i * 5), )
            df_all_phase.append(df_phase(cn1, 'S3P3'))
        print_table(summary)   
        return df_all_phase
    
    
    def call_phase_summary(self, cn, station_size_all_phase, phase):
        station_size = dict(cn.nodes(data='S3P3_station_size'))
        fulfilled_demand = dict(cn.nodes(data='S3P3_h2day'))
        profit_ton = dict(cn.nodes(data='S3P3_kg_profit'))
        operation_rate = 0.97
        return phase_summary(
                                    station_size,
                                    fulfilled_demand,
                                    profit_ton,
                                    operation_rate,
                                    station_size_all_phase,
                                    phase

                                )


    def summary(self):
        station_size = dict(self.graph.nodes(data="S3P3_station_size"))
        counter = Counter(station_size.values())
        station_size_nc = dict(self.graph.nodes(data="S3P1_station_size"))
        counter_nc = Counter(station_size_nc.values())
        print(f"#Station by size scenario 1, 1: {counter_nc[1]}, 2: {counter_nc[2]}, 3: {counter_nc[3]}")
        print(f"#Station by size scenario 3, 1: {counter[1]}, 2: {counter[2]}, 3: {counter[3]}")
        h2day_d = dict(self.graph.nodes(data="S3P3_h2day"))
        h2day = filter(lambda item: item is not None, h2day_d.values())
        h2day_d_nc = dict(self.graph.nodes(data="S3P1_h2day"))
        h2day_nc = filter(lambda item: item is not None, h2day_d_nc.values())
        print(f"Number of tons covered by our network scenario 1: {sum(h2day_nc) / 1000}")
        print(f"Number of tons covered by our network scenario 3: {sum(h2day) / 1000}")
        kg_profit = dict(self.graph.nodes(data="S3P3_kg_profit"))
        kg_profit = filter(lambda item: item is not None, kg_profit.values())
        print(f"Number of tons sold in profit in our network: {sum(kg_profit) / 1000}")
        # h2day_demand_nc = dict(self.graph.nodes(data="S3P1_h2day_demand"))
        # h2day_demand_nc = list(filter(lambda item: item is not None, h2day_demand_nc.values()))
        # print(f"total estimated demand in our network  scenario 1: {sum(h2day_demand_nc) / 1000}")
        # h2day_demand_market = dict(self.graph.nodes(data="S3P3_h2day_demand_market"))
        # h2day_demand_market = list(filter(lambda item: item is not None, h2day_demand_market.values()))
        # print(f"total estimated market demand in our network 3: {sum(h2day_demand_market) / 1000}")
        fig, axs = plt.subplots((2), figsize=(32, 18))
      

        ax = axs[0]
        region_nodes = dict(self.graph.nodes(data="region_name"))
        h2day_by_region = defaultdict(int)
        h2day_by_region_nc = defaultdict(int)

        for node in h2day_d:
            h2day_by_region[region_nodes[node]] += h2day_d[node] / 1000
            h2day_by_region_nc[region_nodes[node]] +=  h2day_d_nc[node] / 1000

        regions = [k for k in h2day_by_region]
        h2day_by_region_both = {
            'Competitor': [h2day_by_region[r] for r in regions],
            'no Competitor': [h2day_by_region_nc[r] for r in regions],
        }
        # ax.bar(*zip(*h2day_by_region.items()))
        x = np.arange(12) 
        width = 0.4  # the width of the bars
        multiplier = 0
        for attribute, measurement in h2day_by_region_both.items():
            offset = width * multiplier
            
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            #ax.bar_label(rects)
            multiplier += 1
        ax.set_xticks(x + width, regions)
        ax.legend(loc='upper left')
        ax.set_ylabel("Demand in t")
        ax.tick_params(axis='x', rotation=45)



        ax = axs[1]
        stations_by_region = defaultdict(int)
        for node in h2day_d:
            stations_by_region[region_nodes[node]] += int(h2day_d[node] > 0)
        rects = ax.bar(*zip(*stations_by_region.items()))
        ax.set_ylabel("Number of stations")
        ax.tick_params(axis='x', rotation=45)
        #ax.bar_label(rects)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))

        ax = axes[0]
        def func(pct, allvals):
            absolute = int(np.round(pct/100.*np.sum(allvals)))
            return f"{pct:.1f}%\n({absolute:d} stations)"

        labels = ['Small', 'Medium', 'Large']
        sizes = [counter[1], counter[2], counter[3]]
        ax.pie(sizes, labels=labels,autopct=lambda pct: func(pct, sizes), textprops={'fontsize': 14})
        ax.set_title('Competitor')
        ax = axes[1]

        labels_nc = ['Small', 'Medium', 'Large']
        sizes_nc = [counter_nc[1], counter_nc[2], counter_nc[3]]

        wedges, texts,autotexts =ax.pie(sizes_nc, labels=labels_nc,autopct=lambda pct: func(pct, sizes_nc), textprops={'fontsize': 14})
        ax.set_title('No Competitor')
    
        fig.tight_layout()         

    def describe(self):
        
        nstations = [k for k,v in dict(self.graph.nodes(data='S3P3_station_size')).items() if v > 0]

        print(f'We plan to deploy {len(nstations)} H2 stations')
        print(f'The existent player has {self.ncompetitor_stations} in France Metropole')
        print(f'{sum(self.conversion_planned.values())} stations lies within our transportation network')
        print(f'{sum(self.rival_stations.values())} stations or  {sum(self.rival_stations.values())/len(nstations):.3f} percent collapse with our deployment plan')


    def plot(self):
        edges, weights = zip(*nx.get_edge_attributes(self.graph, "traffic flow").items())
        weights = np.array(weights) + 1
        weights = np.log(weights)
        is_OD = np.array(list(dict(self.graph.nodes(data="is_OD")).values()))
        # is_nor = np.array(list(dict(roads.nodes(data='region_name')).values() ))
        cluster_centers = dict(self.graph.nodes(data="coordinates"))
        results = dict(self.graph.nodes(data="station_size"))
        candidate_sites = list(results.values())
        is_competitor = list(dict(self.graph.nodes(data='S3P3_conversion_planned')).values())
     

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))

        node_color = ["red" if n else "blue" for n in is_OD]
        for ind in range(len(candidate_sites)):
            if node_color[ind] != "red":
                if is_competitor[ind] > 0:
                    node_color[ind] = "purple"
                   
                    
        options = {
            "edge_color": weights,
            "width": 4,
            "edge_cmap": plt.cm.Wistia,
            "with_labels": False,
            "node_color": node_color,
            "node_size": (is_OD + (np.array(is_competitor)) )* 20,
        }
        nx.draw(
            self.graph,
            pos=cluster_centers,
            **options,
        )

        plt.show()

    def _to_2Dcoords(self, coords, transformer):
    
        x, y = float(coords.split(',')[0]), float(coords.split(',')[-1])
        # Create a PyProj transformer object

        # Transform the input coordinates to the output CRS
        x, y = transformer.transform(x, y)  # Example input coordinates (longitude, latitude)

        return x, y

    def _read_existent_stations(self, data_path):
                # Define the input and output coordinate reference systems
        in_crs = 'EPSG:4326'
        out_crs = 'EPSG:2154 '  # Lambert Conformal Conic projection for France
        transformer = pyproj.Transformer.from_crs(in_crs, out_crs)
        existent_stations = pd.read_csv(data_path + 'Données de stations TE_DV.csv')
        existent_stations['X'] = existent_stations.Coordinates.map(lambda x: self._to_2Dcoords(x, transformer)[0])
        existent_stations['Y'] = existent_stations.Coordinates.map(lambda x: self._to_2Dcoords(x, transformer)[1])
        existent_stations['Coordinates_point'] = existent_stations.apply(lambda x: Point(x.X, x.Y), axis=1)
        existent_stations = existent_stations[(existent_stations.X > 0 ) & (existent_stations.X < 1.1e6)]
        existent_stations = existent_stations[existent_stations['H2 Conversion']==1]
        return existent_stations
    

    def _match_stations(self, cn, existent_stations):
        pr_point = gpd.GeoSeries(
        {i: Point(P) for i, P in dict(cn.nodes(data='coordinates')).items()}
        )

        is_station = {i: 0 for i, P in dict(cn.nodes(data='coordinates')).items()}
        for ele in existent_stations.Coordinates_point:
            if pr_point.distance(ele).min() <= 10000:
                n = pr_point.distance(ele).argmin()
                is_station[n] += 1
            
        return is_station
    
    def _check_rival(self, cn):
        rival = {}
        for n in range(len(cn.nodes)):
            rival[n] = 0
            if cn.nodes[n]['S3P3_conversion_planned'] == 1 and cn.nodes[n]['S3P3_station_size'] > 0:
                rival[n] = 1
        return rival