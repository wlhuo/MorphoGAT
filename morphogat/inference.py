"""Portable checkpoint loading, independent prediction, and paired evaluation."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import load_split, read_manifest
from .graphs import tensor_section
from .metrics import concordance
from .model import MorphoGAT
from .preprocessing import Preprocessor, quality_control
from .utils import new_output_directory, save_json, setup_runtime, sha256


def load_checkpoint(path, device="auto"):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = Config.from_dict(checkpoint["config"])
    config.training.device = device
    runtime_device = setup_runtime(config.training)
    prep_path = path.parent / "preprocessor.npz"
    if sha256(prep_path) != checkpoint["preprocessor_sha256"]:
        raise ValueError("Preprocessor does not match this checkpoint")
    prep = Preprocessor.load(prep_path)
    if prep.genes.tolist() != checkpoint["genes"] or prep.proteins.tolist() != checkpoint["proteins"]:
        raise ValueError("Checkpoint feature ordering does not match preprocessor")
    model = MorphoGAT(len(prep.genes), len(prep.proteins), config.model, checkpoint["variant"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(runtime_device).eval()
    return model, prep, config, runtime_device


def save_predictions(folder, section, transformed, outputs, prep):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {key: value.detach().cpu().numpy() for key, value in outputs.items()}
    pd.DataFrame(arrays["prediction"], index=section.spot_ids, columns=prep.proteins).to_csv(
        folder / "predicted_protein.csv", index_label="barcode")
    weights = pd.DataFrame(arrays["view_weights"], columns=["spatial", "rna", "morphology"],
                           index=section.spot_ids)
    weights["mean_gate"] = arrays["fusion_gates"].mean(axis=1)
    weights["image_valid"] = section.image_valid
    weights.to_csv(folder / "view_weights_and_gates.csv", index_label="barcode")
    np.savez_compressed(folder / "predictions.npz", **arrays, spot_ids=section.spot_ids,
                        proteins=prep.proteins, spatial=section.spatial, image_valid=section.image_valid,
                        scale=np.asarray("training-fitted log1p(B / a)"))


def run_prediction(checkpoint_path, manifest_path, split, output_path, device="auto", evaluate=False):
    model, prep, config, runtime_device = load_checkpoint(checkpoint_path, device)
    specs, base = read_manifest(manifest_path)
    sections = load_split(specs, base, config.preprocess, split, include_protein=evaluate)
    if not sections:
        raise ValueError(f"No sections in split {split}")
    output = new_output_directory(output_path)
    results = []
    for raw in sections:
        section = quality_control(raw, config.preprocess)
        values = prep.transform(section, include_protein=evaluate)
        graph = tensor_section(values, config.graph, runtime_device, include_protein=False)
        with torch.inference_mode():
            outputs = model.predict(graph)
        if not torch.isfinite(outputs["prediction"]).all():
            raise FloatingPointError(f"{section.name}: non-finite predictions")
        folder = output / section.name
        save_predictions(folder, section, values, outputs, prep)
        summary = {"section": section.name, "input_aligned_spots": raw.input_spots,
                   "retained_spots": len(section.spot_ids), "valid_images": int(section.image_valid.sum())}
        if evaluate:
            prediction = outputs["prediction"].cpu().numpy()
            protein_metrics, spot_metrics, scores = concordance(prediction, values["y"], prep.proteins, section.spot_ids)
            protein_metrics.to_csv(folder / "protein_metrics.csv", index=False)
            spot_metrics.to_csv(folder / "spot_metrics.csv", index=False)
            pd.DataFrame(values["y"], index=section.spot_ids, columns=prep.proteins).to_csv(
                folder / "measured_protein_normalized.csv", index_label="barcode")
            summary.update(scores)
        save_json(folder / "summary.json", summary)
        results.append(summary)
        print(summary, flush=True)
    save_json(output / "summary.json", results)
    pd.DataFrame(results).to_csv(output / "summary.csv", index=False)
    save_json(output / "provenance.json", {"checkpoint_sha256": sha256(checkpoint_path),
                                           "manifest_sha256": sha256(manifest_path),
                                           "split": split, "paired_evaluation": evaluate})
    return results


