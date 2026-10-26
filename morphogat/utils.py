"""Reproducibility and portable experiment records."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random

import numpy as np


def save_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    versions = {}
    for name in ["numpy", "scipy", "pandas", "scikit-learn", "torch", "torch-geometric",
                 "Pillow", "PyYAML", "matplotlib"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def setup_runtime(config):
    # Must be set before initializing a CUDA context.
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(config.deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def new_output_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path
