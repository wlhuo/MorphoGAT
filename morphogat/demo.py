"""Small paired sections for installation and workflow checks."""
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .config import Config
from .utils import new_output_directory


def make_demo(output_path, seed=7):
    output = new_output_directory(output_path).resolve()
    rng = np.random.default_rng(seed)
    sections = []
    genes = np.asarray([f"GENE{i:03d}" for i in range(28)])
    proteins = np.asarray(["CD3", "CD4", "CD8", "CD19"])
    weights = rng.normal(0, 0.2, (len(genes), len(proteins)))
    for index, split in enumerate(["train", "train", "validation", "test", "external"]):
        n, name = 36, f"demo_{index}"
        ids = np.asarray([f"{name}_{i:03d}" for i in range(n)])
        x, y = np.meshgrid(np.arange(6), np.arange(6))
        coords = np.column_stack([24 + 18 * x.ravel(), 24 + 18 * y.ravel()]).astype(np.float32)
        counts = rng.poisson(3 + rng.gamma(2, 1, (n, len(genes)))).astype(np.float32)
        protein = np.maximum(0, np.log1p(counts) @ weights + 3 + rng.normal(0, 0.1, (n, len(proteins)))) * 10
        values = {"counts": counts, "genes": genes, "spot_ids": ids, "spatial": coords}
        if split != "external":
            values.update(protein=protein.astype(np.float32), proteins=proteins)
        np.savez_compressed(output / f"{name}.npz", **values)
        pixels = np.zeros((144, 144, 3), dtype=np.uint8)
        yy, xx = np.indices(pixels.shape[:2])
        pixels[..., 0] = np.clip(160 + xx // 2 + rng.normal(0, 4, xx.shape), 0, 255)
        pixels[..., 1] = np.clip(90 + yy // 2 + rng.normal(0, 4, xx.shape), 0, 255)
        pixels[..., 2] = np.clip(170 + (xx + yy) // 5, 0, 255)
        Image.fromarray(pixels).save(output / f"{name}.png")
        sections.append({"id": name, "split": split, "npz": f"{name}.npz", "image": f"{name}.png",
                         "spatial_is_pixel": True, "image_scale": 1.0})
    (output / "manifest.yaml").write_text(yaml.safe_dump({"sections": sections}, sort_keys=False), encoding="utf-8")
    config = Config()
    config.preprocess.min_genes = 1
    config.preprocess.pca_components = 8
    config.preprocess.patch_size = 64
    config.model.widths = [32, 16, 16]
    config.model.feature_gate_heads = 2
    config.model.feature_gate_width = 16
    config.model.scorer_width = 16
    config.model.decoder_width = 32
    config.model.protein_branch_width = 8
    config.training.epochs = 4
    config.training.device = "cpu"
    (output / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    print(f"Demo written to {output}. These are simulated data for workflow checks.")
    return output / "manifest.yaml"
