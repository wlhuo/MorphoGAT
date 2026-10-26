import numpy as np
import pytest

from morphogat.config import GraphConfig
from morphogat.graphs import build_graphs, knn_edges
from morphogat.metrics import concordance


def test_knn_symmetry_no_self_and_neighbor_cap():
    edges = knn_edges(np.array([[0], [1], [4]], dtype=float), 10)
    pairs = set(map(tuple, edges.T))
    assert len(pairs) == 6
    assert all(a != b and (b, a) in pairs for a, b in pairs)
    assert knn_edges(np.array([[0.0]]), 10).shape == (2, 0)
    equal_edges = knn_edges(np.zeros((5, 2)), 2)
    assert not np.any(equal_edges[0] == equal_edges[1])


def test_masked_morphology_and_no_protein_graph():
    values = {"spatial": np.arange(8).reshape(4, 2), "rna_pca": np.zeros((4, 2)),
              "f": np.zeros((4, 65)), "mask": np.array([True, False, True, False]), "y": None}
    graphs = build_graphs(values, GraphConfig())
    assert "protein" not in graphs
    assert set(graphs["morphology"].ravel()) == {0, 2}
    with pytest.raises(ValueError, match="requires training"):
        build_graphs(values, GraphConfig(), include_protein=True)


def test_metrics_ties_constants_and_identifier_dimensions():
    true = np.array([[0, 1], [1, 1], [1, 1], [3, 1]], dtype=float)
    pred = true.copy()
    per_protein, _, summary = concordance(pred, true, ["p1", "constant"], ["a", "b", "c", "d"])
    assert per_protein.loc[0, "spearman"] == pytest.approx(1)
    assert np.isnan(per_protein.loc[1, "spearman"])
    assert summary["valid_proteins_spearman"] == 1
    assert summary["median_protein_spearman"] == pytest.approx(1)
    assert summary["rmse"] == 0
    with pytest.raises(ValueError, match="dimensions"):
        concordance(pred, true, ["p1"], ["a", "b", "c", "d"])
