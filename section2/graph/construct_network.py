import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point

from section2.utils.visualization import visualize_network
from section2.preprocessing.region import add_region, filter_on_region
from section2.preprocessing.cluster_points import cluster_nodes, cluster_nodes_ald

from section2.graph.trasportation_network import create_graph

def construct_network(tmja, ald, region=None):
    code2region = {
            '52': 'Pays de la Loire'
            ,'24': 'Centre-Val de Loire'
            ,'28': 'Normandie'
            ,'11': 'Île-de-France'
            ,'32': 'Hauts-de-France'
            ,'44': 'Grand Est'
            ,'75': 'Nouvelle-Aquitaine'
            ,'53': 'Bretagne'
            ,'84': 'Auvergne-Rhône-Alpes'
            ,'76': 'Occitanie'
            ,'93': 'Provence-Alpes-Côte d\'Azur'
            ,'27': 'Bourgogne-Franche-Comté'
        }
    tmja, ald = add_region(tmja, ald)
    if not region:
        tmja_r, ald_r = tmja, ald
        ald_rc = cluster_nodes_ald(ald_r, 80000)
        roads_r= make_graph(tmja_r, ald_rc,distance_threshold=15000 )
    else:
        tmja_r, ald_r = filter_on_region(tmja, ald, code2region[str(region)])
        ald_rc = cluster_nodes_ald(ald_r, 50000)
        roads_r= make_graph(tmja_r, ald_rc,distance_threshold=5000 )

    largest_cc = max(nx.connected_components(roads_r), key=len)
    cn = roads_r.subgraph(largest_cc)

    # visualize_network(cn)

    return cn
    
def make_graph(tmja, ald,distance_threshold=5000):
    model, cluster_centers = cluster_nodes(tmja, distance_threshold)
    roads =  create_graph(model, cluster_centers, tmja)
    is_OD = match_OD(ald, cluster_centers)
    nx.set_node_attributes(roads, {i:is_OD[i] for i in range(len(is_OD))}, "is_OD")
    visualize_network(roads)
    return roads

def match_OD(ald, cluster_centers):
    cluster_centers_point = gpd.GeoSeries(
    {i: Point(P) for i, P in cluster_centers.items()}
    )

    is_OD = np.zeros(len(cluster_centers_point))
    for ele in ald.centroid:
        n = cluster_centers_point.distance(ele).argmin()
        is_OD[n] = 1
    
    return is_OD

