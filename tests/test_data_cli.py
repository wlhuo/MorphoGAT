import json

import numpy as np
import pandas as pd
import pytest
import yaml

from morphogat.cli import main
from morphogat.config import Config, PreprocessConfig
from morphogat.data import load_section, read_manifest, read_matrix
from morphogat.demo import make_demo
from morphogat.table_evaluation import evaluate_tables


def test_demo_and_check_data_workflow(tmp_path):
    demo = tmp_path / "demo"
    make_demo(demo)
    output = tmp_path / "checked"
    main(["check-data", "--manifest", str(demo / "manifest.yaml"), "--config", str(demo / "config.yaml"),
          "--output", str(output)])
    report = json.loads((output / "data_report.json").read_text())
    assert len(report) == 5
    assert all(row["valid_images"] == row["spots"] == 36 for row in report)
    assert all(row["proteins"] == 4 for row in report)


def test_prediction_loader_does_not_open_protein(tmp_path):
    pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["g1", "g2"]).to_csv(tmp_path / "rna.csv")
    pd.DataFrame([[0, 0], [1, 1]], index=["a", "b"], columns=["x", "y"]).to_csv(tmp_path / "spatial.csv")
    spec = {"id": "external", "rna": "rna.csv", "spatial": "spatial.csv", "protein": "DOES_NOT_EXIST.csv"}
    section = load_section(spec, tmp_path, PreprocessConfig(), include_protein=False)
    assert section.protein is None and len(section.spot_ids) == 2


def test_duplicate_source_partition_is_rejected(tmp_path):
    values = {"sections": [{"id": "train", "split": "train", "npz": "same.npz"},
                           {"id": "test", "split": "test", "npz": "same.npz"}]}
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="reused"):
        read_manifest(path)


def test_table_metrics_realign_identifiers(tmp_path):
    prediction = pd.DataFrame([[1, 2], [2, 4], [3, 6]], index=["a", "b", "c"], columns=["p1", "p2"])
    prediction.to_csv(tmp_path / "prediction.csv")
    prediction.iloc[::-1, ::-1].to_csv(tmp_path / "truth.csv")
    summary = evaluate_tables(tmp_path / "prediction.csv", tmp_path / "truth.csv", tmp_path / "metrics")
    assert summary["rmse"] == 0
    prediction.rename(columns={"p2": "p3"}).to_csv(tmp_path / "mismatch.csv")
    with pytest.raises(ValueError, match="identifier"):
        evaluate_tables(tmp_path / "prediction.csv", tmp_path / "mismatch.csv", tmp_path / "failed")


def test_unknown_configuration_and_invalid_heads():
    with pytest.raises((ValueError, TypeError)):
        Config.from_dict({"training": {"learning_rte": 0.1}})
    with pytest.raises(ValueError, match="divisible"):
        Config.from_dict({"model": {"widths": [31, 16, 16]}})


def test_numeric_barcodes_are_preserved(tmp_path):
    path = tmp_path / "counts.csv"
    path.write_text("barcode,g1\n0001,2\n0002,3\n")
    assert read_matrix(path).index.tolist() == ["0001", "0002"]


def test_section_ids_cannot_escape_output_root(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"sections": [{"id": "..", "split": "train", "npz": "counts.npz"}]}))
    with pytest.raises(ValueError, match="Section IDs"):
        read_manifest(path)
