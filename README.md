# MorphoGAT

Morphology-aware graph attention for spatial protein inference.



MorphoGAT predicts a fixed panel of protein abundances from spatial RNA counts,
spatial coordinates and registered tissue images. It combines three RNA graph
views, spot-specific view weights and gated residual image fusion. A separate
protein encoder supplies auxiliary supervision during paired training. 



## Installation

Use Python 3.11 in a fresh environment. Python 3.10 is supported by the
package metadata. The dependency ranges target the PyTorch 2.5 / PyG 2.6 APIs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


## Quick start



```bash
python -m morphogat make-demo --output data/demo
python -m morphogat check-data --manifest data/demo/manifest.yaml --config data/demo/config.yaml --output runs/demo_check
python -m morphogat train --manifest data/demo/manifest.yaml --config data/demo/config.yaml --output runs/demo_model --device cpu
python -m morphogat evaluate --checkpoint runs/demo_model/best.pt --manifest data/demo/manifest.yaml --split test --output runs/demo_test --device cpu
python -m morphogat predict --checkpoint runs/demo_model/best.pt --manifest data/demo/manifest.yaml --split external --output runs/demo_external --device cpu
python -m morphogat plot-spatial --predictions runs/demo_external/demo_4/predictions.npz --protein CD3 --output runs/demo_external/CD3.png
```

## Training with paired data



```bash
python -m morphogat check-data --manifest examples/manifest.yaml --config configs/reference.yaml --output runs/data_check
python -m morphogat train --manifest examples/manifest.yaml --config configs/reference.yaml --output runs/reference --seed 0 --device cuda:0
```


## Prediction and evaluation

Keep `best.pt` and `preprocessor.npz` together when moving a model. The checkpoint
verifies the preprocessing file checksum and gene/protein ordering.

```bash
python -m morphogat predict --checkpoint runs/reference/best.pt --manifest examples/manifest.yaml --split external --output runs/external
python -m morphogat evaluate --checkpoint runs/reference/best.pt --manifest examples/manifest.yaml --split test --output runs/test
```


## Ablations and downstream analysis

```bash
python scripts/run_experiments.py --manifest examples/manifest.yaml --config configs/reference.yaml --output runs/ablations --seeds 0 1 2 --device cuda:0
python -m morphogat cluster --predictions runs/external/external_section/predictions.npz --labels data/external_section/labels.csv --resolution 0.5 --neighbors 10 --output runs/domains
python -m morphogat plot-spatial --predictions runs/external/external_section/predictions.npz --readout morphology --output runs/morphology_weight.png
```


## Tests and source release

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q -ra
python scripts/build_release.py
```

