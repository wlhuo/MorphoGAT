"""Write the two executable walkthrough notebooks using the public CLI."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def write(name, cells):
    path = ROOT / "notebooks" / name
    path.parent.mkdir(exist_ok=True)
    value = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"},
                                          "language_info": {"name": "python", "version": "3.11"}},
             "nbformat": 4, "nbformat_minor": 5}
    for index, cell in enumerate(value["cells"]):
        cell["id"] = f"cell-{index:02d}"
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    setup = '''from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd().resolve()
if not (ROOT / "morphogat").is_dir():
    ROOT = ROOT.parent
assert (ROOT / "morphogat").is_dir(), "Open this notebook inside the source checkout"

def run(*arguments):
    subprocess.run([sys.executable, "-m", "morphogat", *map(str, arguments)], cwd=ROOT, check=True)
'''
    write("01_train_and_evaluate.ipynb", [
        markdown("# MorphoGAT: paired training and evaluation\n\nInstall the project dependencies in this notebook's Python environment first.\nThis walkthrough uses simulated sections to verify the workflow.\nUse new output directories when rerunning cells; existing experiment files are protected."),
        code(setup),
        markdown("## Make and validate the demonstration sections\n\nThe real-data workflow uses the same manifest format with raw counts and registered tissue images."),
        code('''DATA = ROOT / "data" / "notebook_demo"
MODEL = ROOT / "runs" / "notebook_model"
TEST = ROOT / "runs" / "notebook_test"
run("make-demo", "--output", DATA)
run("check-data", "--manifest", DATA / "manifest.yaml", "--config", DATA / "config.yaml",
    "--output", ROOT / "runs" / "notebook_check")
'''),
        markdown("## Fit the paired model\n\nThe small demo runs four CPU epochs. For study data use `configs/reference.yaml` and an explicit validation partition."),
        code('''run("train", "--manifest", DATA / "manifest.yaml", "--config", DATA / "config.yaml",
    "--output", MODEL, "--device", "cpu")
'''),
        markdown("## Evaluate the independent RNA-to-protein path\n\nHeld-out proteins are used as targets only. The auxiliary protein encoder is not called."),
        code('''run("evaluate", "--checkpoint", MODEL / "best.pt", "--manifest", DATA / "manifest.yaml",
    "--split", "test", "--output", TEST, "--device", "cpu")
import pandas as pd
pd.read_csv(TEST / "demo_3" / "protein_metrics.csv")
'''),
        code('''import matplotlib.pyplot as plt
history = pd.read_csv(MODEL / "history.csv")
history.plot(x="epoch", y=["train_prediction", "validation_rmse"], marker="o")
plt.ylabel("Normalized protein RMSE")
plt.show()
'''),
        markdown("## Study experiments\n\nSee `docs/EXPERIMENTS.md` for fixed-split repetitions, the four ablations, baseline-table evaluation and manuscript-style boxplots. The demo's numerical scores are not study results."),
    ])
    write("02_predict_and_spatial_readouts.ipynb", [
        markdown("# MorphoGAT: external prediction and spatial readouts\n\nRun Notebook 01 first, or point the variables below to an existing checkpoint and manifest. Prediction requires RNA, coordinates and valid images according to the saved configuration; it never reads a protein target file."),
        code(setup),
        code('''import numpy as np
import pandas as pd
from IPython.display import Image, display

DATA = ROOT / "data" / "notebook_demo"
MODEL = ROOT / "runs" / "notebook_model"
OUT = ROOT / "runs" / "notebook_external"
run("predict", "--checkpoint", MODEL / "best.pt", "--manifest", DATA / "manifest.yaml",
    "--split", "external", "--output", OUT, "--device", "cpu")
PREDICTIONS = OUT / "demo_4" / "predictions.npz"
pd.read_csv(OUT / "demo_4" / "predicted_protein.csv").head()
'''),
        markdown("## Protein, graph weight and fusion gate maps\n\nThese readouts describe relative input use. The toy image and counts do not establish a biological interpretation."),
        code('''run("plot-spatial", "--predictions", PREDICTIONS, "--protein", "CD3", "--output", OUT / "CD3.png")
run("plot-spatial", "--predictions", PREDICTIONS, "--readout", "morphology", "--output", OUT / "morphology.png")
run("plot-spatial", "--predictions", PREDICTIONS, "--readout", "mean_gate", "--output", OUT / "gate.png")
for name in ["CD3.png", "morphology.png", "gate.png"]:
    display(Image(filename=str(OUT / name), width=550))
'''),
        code('''weights = pd.read_csv(OUT / "demo_4" / "view_weights_and_gates.csv", index_col=0)
assert np.allclose(weights[["spatial", "rna", "morphology"]].sum(axis=1), 1)
weights.describe()
'''),
        markdown("## Optional spatial domain analysis\n\nInstall `requirements-analysis.txt`. On study data use `cluster --predictions ... --labels ... --resolution ... --neighbors ... --output ...`. Annotation CSVs contain `barcode,label`. Set clustering parameters without optimizing against held-out labels; the original figure settings must be supplied for exact reproduction."),
    ])


if __name__ == "__main__":
    main()
