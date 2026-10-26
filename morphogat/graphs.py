"""Sparse, section-local, symmetrized nearest-neighbor graphs."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_edges(features, k, valid=None):
    features = np.asarray(features)
    n = len(features)
    eligible = np.arange(n) if valid is None else np.flatnonzero(valid)
    if len(eligible) < 2:
        return np.empty((2, 0), dtype=np.int64)
    count = min(k, len(eligible) - 1)
    # Request one extra neighbor and remove the actual center explicitly. This
    # also handles equal feature rows whose nearest-neighbor tie order varies.
    model = NearestNeighbors(n_neighbors=count + 1, metric="euclidean", n_jobs=1)
    local = features[eligible]
    indices = model.fit(local).kneighbors(local, return_distance=False)
    edges = []
    for center, neighbors in enumerate(indices):
        selected = [j for j in neighbors if j != center][:count]
        edges.extend((int(eligible[j]), int(eligible[center])) for j in selected)
    forward = np.asarray(edges, dtype=np.int64).T
    return np.unique(np.concatenate([forward, forward[::-1]], axis=1), axis=1)


def build_graphs(transformed, config, include_protein=False):
    graphs = {
        "spatial": knn_edges(transformed["spatial"], config.spatial_k),
        "rna": knn_edges(transformed["rna_pca"], config.rna_k),
        "morphology": knn_edges(transformed["f"], config.morphology_k, transformed["mask"]),
    }
    if include_protein:
        if transformed["y"] is None:
            raise ValueError("Protein graph construction requires training protein measurements")
        protein = knn_edges(transformed["y"], config.protein_k)
        graphs["protein"] = np.unique(np.concatenate([graphs["spatial"], protein], axis=1), axis=1)
    return graphs


def tensor_section(transformed, graph_config, device, include_protein=False):
    import torch
    graphs = build_graphs(transformed, graph_config, include_protein)
    result = dict(transformed)
    for key in ["x", "y", "f", "mask"]:
        if result[key] is not None:
            result[key] = torch.as_tensor(result[key], device=device)
    result["graphs"] = {key: torch.as_tensor(value, dtype=torch.long, device=device)
                        for key, value in graphs.items()}
    return result
