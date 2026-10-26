import json

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from morphogat.config import Config
from morphogat.demo import make_demo
from morphogat.inference import load_checkpoint, run_prediction
from morphogat.training import train


def test_train_reload_predict_and_evaluate(tmp_path):
    data = tmp_path / "demo"
    make_demo(data)
    config = Config.load(data / "config.yaml")
    config.training.epochs = 2
    config.training.threads = 1
    run = tmp_path / "run"
    summary = train(data / "manifest.yaml", config, run)
    assert summary["best_epoch"] in (1, 2)
    assert (run / "best.pt").is_file()
    _, prep, _, _ = load_checkpoint(run / "best.pt", "cpu")
    assert len(prep.proteins) == 4
    first = run_prediction(run / "best.pt", data / "manifest.yaml", "external", tmp_path / "predict", "cpu")
    second = run_prediction(run / "best.pt", data / "manifest.yaml", "test", tmp_path / "evaluate", "cpu", True)
    assert first[0]["retained_spots"] == 36
    assert 0 <= second[0]["valid_proteins_spearman"] <= 4
    assert np.isfinite(second[0]["rmse"])
    with np.load(tmp_path / "predict" / "demo_4" / "predictions.npz") as values:
        assert values["prediction"].shape == (36, 4)
    # The sidecar integrity check catches mixed models/preprocessors.
    with (run / "preprocessor.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(run / "best.pt", "cpu")
