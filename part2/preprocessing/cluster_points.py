import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import AgglomerativeClustering

def cluster_nodes(tmja, distance_threshold):
    """Cluster based on distance on tmja."""
    model = AgglomerativeClustering(
        linkage="complete", distance_threshold=distance_threshold, n_clusters=None
    )
    X = np.concatenate((list(zip(tmja.xD, tmja.yD)), list(zip(tmja.xF, tmja.yF))))
    model.fit(X)
    n_roads = tmja.shape[0]
    clusters = [[] for _ in range(max(model.labels_) + 1)]
    for i in range(n_roads * 2):
        clusters[model.labels_[i]].append(
            [tmja["xD"].iloc[i], tmja["yD"].iloc[i]]
            if i < n_roads
            else [tmja["xF"].iloc[i - n_roads], tmja["yF"].iloc[i - n_roads]]
        )
    # strange multiple times the same route ?are there two way roads
    # probably two way roads
    cluster_centers = {
        i: np.mean(np.array(clusters[i]), axis=0) for i in range(max(model.labels_) + 1)
    }
    return model, cluster_centers

def cluster_nodes_ald(ald, distance_threshold=5000):
    """Cluster based on distance on ald."""
    model = AgglomerativeClustering(
        linkage="complete", distance_threshold=distance_threshold, n_clusters=None
    )
    X = list(ald.centroid.map(lambda x: list(x.coords[0])))
    model.fit(X)
    n_hubs = ald.shape[0]
    clusters = [[] for _ in range(max(model.labels_) + 1)]
    for i in range(n_hubs):
        clusters[model.labels_[i]].append(
            list(ald['centroid'].iloc[i].coords[0])
        )
    cluster_centers = [
        Point(np.mean(np.array(clusters[i]), axis=0)) for i in range(max(model.labels_) + 1)
    ]
    return pd.DataFrame(cluster_centers
                        ,columns=['centroid'])
