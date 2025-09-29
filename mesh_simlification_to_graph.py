import os
import math
import pickle
from typing import List, Dict

import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

import pymesh


def _faces_to_edges_unique(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for f in faces:
        f = list(f)
        for i in range(len(f)):
            u = int(f[i]); v = int(f[(i + 1) % len(f)])
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
    if not edges:
        return np.zeros((0, 2), dtype=int)
    return np.array(sorted(edges), dtype=int)


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def mesh_to_vertex_graph(mesh) -> nx.Graph:
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    E = _faces_to_edges_unique(F)
    G = nx.Graph()
    for i, xyz in enumerate(V):
        G.add_node(i, coords=xyz.astype(float))
    for i, j in E:
        G.add_edge(int(i), int(j), weight=_euclid(V[i], V[j]))
    return G


def truncated_dijkstra(G: nx.Graph, start: int, max_radius: float) -> Dict[int, float]:
    import heapq
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > max_radius:
            continue
        for v, attr in G[u].items():
            nd = d + float(attr.get("weight", 1.0))
            if nd <= max_radius and nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def create_local_patches(G: nx.Graph, max_radius: float) -> List[List[int]]:
    remaining = set(G.nodes())
    patches: List[List[int]] = []
    while remaining:
        seed = remaining.pop()
        reach = truncated_dijkstra(G, seed, max_radius)
        patch_nodes = list(reach.keys())
        patches.append(patch_nodes)
        remaining.difference_update(patch_nodes)
    return patches


def compute_mean_curvature(mesh) -> np.ndarray:
    mesh.add_attribute("vertex_mean_curvature")
    mc = mesh.get_attribute("vertex_mean_curvature")
    if mc.ndim == 1:
        mc = mc[:, None]
    return mc  # (N,1)


def load_properties_txt(props_dir: str, base_name: str) -> np.ndarray:
    path = os.path.join(props_dir, base_name + ".txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Properties file not found: {path}")
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr  # (N,P)


def build_feature_matrix(mesh, properties: np.ndarray, mean_curv: np.ndarray) -> np.ndarray:
    V = np.asarray(mesh.vertices)  # (N,3)
    if properties.shape[0] != V.shape[0]:
        raise ValueError(f"Properties rows ({properties.shape[0]}) != vertices ({V.shape[0]})")
    if mean_curv.shape[0] != V.shape[0]:
        raise ValueError(f"Mean curvature length ({mean_curv.shape[0]}) != vertices ({V.shape[0]})")
    if mean_curv.ndim == 1:
        mean_curv = mean_curv[:, None]
    return np.column_stack((V, properties, mean_curv))  # (N, 3+P+1)


def _fit_agglomerative(X: np.ndarray, linkage: str, distance_threshold: float):
    """Compat helper for sklearn versions (metric vs affinity)."""
    try:
        # Newer sklearn
        return AgglomerativeClustering(
            n_clusters=None, metric="euclidean", linkage=linkage, distance_threshold=distance_threshold
        ).fit(X)
    except TypeError:
        # Older sklearn (e.g., 0.24)
        return AgglomerativeClustering(
            n_clusters=None, affinity="euclidean", linkage=linkage, distance_threshold=distance_threshold
        ).fit(X)


def cluster_patch(data_patch: np.ndarray, distance_threshold: float = 0.5,
                  linkage: str = "average") -> np.ndarray:
    if len(data_patch) < 2:
        return np.zeros(len(data_patch), dtype=int)
    X = MinMaxScaler().fit_transform(data_patch)
    model = _fit_agglomerative(X, linkage=linkage, distance_threshold=distance_threshold)
    return model.labels_


def label_from_patches(data: np.ndarray, patches: List[List[int]],
                       distance_threshold: float = 0.5,
                       linkage: str = "average") -> np.ndarray:
    labels = np.full(data.shape[0], -1, dtype=int)
    next_label = 0
    for patch in patches:
        if len(patch) < 2:
            labels[np.array(patch)] = next_label
            next_label += 1
            continue
        lp = cluster_patch(data[patch, :], distance_threshold, linkage)
        # Si l'agglomératif renvoie un seul cluster, on le garde comme un seul label
        if len(np.unique(lp)) < 2:
            labels[np.array(patch)] = next_label
            next_label += 1
            continue
        for l in np.unique(lp):
            idx = np.array(patch)[lp == l]
            labels[idx] = next_label
            next_label += 1
    return labels


def compute_centroids(global_labels: np.ndarray, data: np.ndarray) -> np.ndarray:
    cents = []
    for l in np.unique(global_labels):
        if l < 0:
            continue
        pts = data[global_labels == l]
        if len(pts) == 0:
            continue
        cents.append(np.mean(pts, axis=0))
    if not cents:
        return np.zeros((0, data.shape[1]), dtype=float)
    return np.vstack(cents)


def find_optimal_k(centroids: np.ndarray, feat_from: int = 3, max_k: int = 10) -> int:
    n = len(centroids)
    if n <= 3:
        return max(1, min(2, n - 1))
    X = centroids[:, feat_from:] if centroids.shape[1] > feat_from else centroids
    best_k, best_val = 3, math.inf
    for k in range(3, max_k + 1):
        if k >= n:
            break
        nn = NearestNeighbors(n_neighbors=k).fit(X)
        dists, _ = nn.kneighbors(X)
        val = float(np.mean(dists[:, 1:]))
        if val < best_val:
            best_val, best_k = val, k
    return best_k


def build_centroid_graph(centroids: np.ndarray, k: int, feat_from: int = 3) -> nx.Graph:
    G = nx.Graph()
    n = len(centroids)
    if n == 0:
        return G
    for i, c in enumerate(centroids):
        G.add_node(i, feat=c.astype(float), xyz=c[:3].astype(float))
    if n == 1 or k <= 0:
        return G
    X = centroids[:, feat_from:] if centroids.shape[1] > feat_from else centroids
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X)
    dists, inds = nn.kneighbors(X)
    for i in range(n):
        for d, j in zip(dists[i][1:], inds[i][1:]):  # skip self
            j = int(j); w = float(d)
            if G.has_edge(i, j):
                G[i][j]["weight"] = min(G[i][j]["weight"], w)
            else:
                G.add_edge(i, j, weight=w)
    return G


class SingleCentroidGraph:
    """
    Version minimaliste:
      - __init__ construit tout
      - .graph : Graph NetworkX (nœuds 'feat', 'xyz' ; arêtes 'weight')
      - .centroids : np.ndarray (C, D)
      - .save_pickle(path) : optionnel
    """
    def __init__(self,
                 mesh_path: str,
                 properties_dir: str,
                 radius: float = 3.0,
                 dist_th: float = 0.5,
                 linkage: str = "average",
                 max_k: int = 10,
                 feat_from: int = 3):  
        # Load mesh
        mesh = pymesh.load_mesh(mesh_path)
        base_name = os.path.splitext(os.path.basename(mesh_path))[0]

        # load  proprties from txt file
        props = load_properties_txt(properties_dir, base_name)
        # compute mean curvatures
        mean_curv = compute_mean_curvature(mesh)

        # Features Matrix: [xyz | propreties | mean_curv]
        data = build_feature_matrix(mesh, props, mean_curv)

        # meah to graph 
        Gv = mesh_to_vertex_graph(mesh)
        # local patches
        patches = create_local_patches(Gv, max_radius=radius)

        # Clustering intra-patch -> global labels
        labels = label_from_patches(data, patches, distance_threshold=dist_th, linkage=linkage)

        #  centroid of a cluster 
        centroids = compute_centroids(labels, data)

        # Graphe de centroides (k-NN sur les features à partir de feat_from)
        if len(centroids) > 0:
            k_opt = find_optimal_k(centroids, feat_from=feat_from, max_k=max_k)
            Gc = build_centroid_graph(centroids, k=k_opt, feat_from=feat_from)
        else:
            Gc = nx.Graph()

        self.graph = Gc
        self.centroids = centroids

    def save_pickle(self, out_path: str):
        dirpath = os.path.dirname(out_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(self.graph, f)
