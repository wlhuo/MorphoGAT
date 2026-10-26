import numpy as np
import pandas as pd
import pytest

from morphogat.analysis import plot_domains, plot_protein_metrics, plot_spatial


def test_metric_and_spatial_figures(tmp_path):
    pytest.importorskip("matplotlib")
    table = pd.DataFrame({"protein": ["CD3", "CD4", "CD8"], "spearman": [0.1, 0.5, 0.8]})
    table.to_csv(tmp_path / "metrics.csv", index=False)
    boxplot = tmp_path / "boxplot.png"
    plot_protein_metrics([tmp_path / "metrics.csv"], ["example"], boxplot)
    assert boxplot.stat().st_size > 1000
    assert boxplot.with_suffix(".csv").exists()
    fixture = tmp_path / "predictions.npz"
    np.savez_compressed(fixture, spatial=np.array([[0, 0], [1, 0], [0, 1]]),
                        proteins=np.array(["CD3"]), prediction=np.array([[0.1], [0.5], [0.9]]),
                        fusion_gates=np.array([[0.2], [0.5], [0.7]]),
                        view_weights=np.full((3, 3), 1 / 3))
    plot_spatial(fixture, tmp_path / "protein.png", protein="CD3")
    plot_spatial(fixture, tmp_path / "gate.png", readout="mean_gate")
    assert (tmp_path / "protein.png").stat().st_size > 1000
    domains = pd.DataFrame({"barcode": ["a", "b", "c"], "x": [0, 1, 0], "y": [0, 0, 1],
                            "cluster": ["0", "1", "0"], "label": ["A", "B", "A"]})
    domains.to_csv(tmp_path / "domains.csv", index=False)
    plot_domains([tmp_path / "domains.csv"], ["example"], tmp_path / "domains.png")
    assert (tmp_path / "domains.png").stat().st_size > 1000
    with pytest.raises(ValueError, match="not in predicted panel"):
        plot_spatial(fixture, tmp_path / "absent.png", protein="absent")
