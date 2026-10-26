"""Section-level training; validation uses the deployment RNA-only path."""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .data import load_split, read_manifest
from .graphs import tensor_section
from .losses import joint_loss, rmse
from .model import MorphoGAT
from .preprocessing import Preprocessor, quality_control
from .utils import environment, new_output_directory, save_json, setup_runtime, sha256


def train(manifest_path, config, output_path, variant="full"):
    config.validate()
    device = setup_runtime(config.training)
    specs, base = read_manifest(manifest_path)
    train_raw = load_split(specs, base, config.preprocess, "train")
    validation_raw = load_split(specs, base, config.preprocess, "validation")
    if not train_raw or not validation_raw:
        raise ValueError("Explicit paired train and validation sections are required; test is never used for selection")
    training = [quality_control(s, config.preprocess) for s in train_raw]
    validation = [quality_control(s, config.preprocess) for s in validation_raw]
    prep = Preprocessor(config.preprocess).fit(training, seed=config.training.seed)
    train_graphs = [tensor_section(prep.transform(s), config.graph, device, True) for s in training]
    # Validation proteins are targets only, never graph inputs or latent teachers.
    validation_graphs = [tensor_section(prep.transform(s), config.graph, device, False) for s in validation]
    output = new_output_directory(output_path)
    prep.save(output / "preprocessor.npz")
    prep_hash = sha256(output / "preprocessor.npz")
    save_json(output / "config.json", config.to_dict())
    save_json(output / "environment.json", {**environment(), "device": str(device),
                                            "cuda_runtime": torch.version.cuda})
    qc_report = [{"id": s.name, "input_aligned_spots": s.input_spots,
                  "retained_spots": len(s.spot_ids), "valid_images": int(s.image_valid.sum())}
                 for s in training + validation]
    save_json(output / "qc.json", qc_report)
    # Keep source paths in the run record, not in portable model checkpoints.
    save_json(output / "manifest.json", {"manifest": str(Path(manifest_path).resolve()), "sections": specs})
    (output / "genes.txt").write_text("\n".join(prep.genes) + "\n", encoding="utf-8")
    (output / "proteins.txt").write_text("\n".join(prep.proteins) + "\n", encoding="utf-8")
    model = MorphoGAT(len(prep.genes), len(prep.proteins), config.model, variant).to(device)
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    options = config.training
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": options.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=options.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=options.scheduler_patience, min_lr=options.min_lr)
    best, stale, best_epoch, records = float("inf"), 0, 0, []
    rng = np.random.default_rng(options.seed)
    for epoch in range(1, options.epochs + 1):
        started = time.monotonic()
        model.train()
        totals = {key: 0.0 for key in ["loss", "rna", "protein", "prediction", "cross", "alignment"]}
        for i in rng.permutation(len(train_graphs)):
            section = train_graphs[i]
            optimizer.zero_grad(set_to_none=True)
            output_values = model(section)
            loss, terms = joint_loss(output_values, section, options, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, section {section['name']}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            totals["loss"] += loss.item()
            for key, value in terms.items():
                totals[key] += value.item()
            del loss, terms, output_values
        model.eval()
        with torch.inference_mode():
            validation_rmse = float(np.mean([rmse(model.predict(s)["prediction"], s["y"]).item()
                                            for s in validation_graphs]))
        if not np.isfinite(validation_rmse):
            raise FloatingPointError("Non-finite validation error")
        scheduler.step(validation_rmse)
        record = {"epoch": epoch, **{f"train_{k}": v / len(train_graphs) for k, v in totals.items()},
                  "validation_rmse": validation_rmse, "learning_rate": optimizer.param_groups[0]["lr"],
                  "seconds": time.monotonic() - started}
        records.append(record)
        pd.DataFrame(records).to_csv(output / "history.csv", index=False)
        if validation_rmse < best - options.min_delta:
            best, stale, best_epoch = validation_rmse, 0, epoch
            checkpoint = {"format_version": 1, "package_version": "0.1.0", "config": config.to_dict(),
                          "variant": variant, "genes": prep.genes.tolist(), "proteins": prep.proteins.tolist(),
                          "preprocessor_sha256": prep_hash, "epoch": epoch,
                          "validation_rmse": best, "train_ids": [s.name for s in training],
                          "validation_ids": [s.name for s in validation],
                          "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            torch.save(checkpoint, output / "best.pt")
        else:
            stale += 1
        print(f"epoch={epoch:03d} loss={record['train_loss']:.5f} val_rmse={validation_rmse:.5f} "
              f"lr={record['learning_rate']:.2g}", flush=True)
        if epoch >= options.early_start and stale >= options.patience:
            break
    summary = {"best_epoch": best_epoch, "validation_rmse": best, "epochs_run": len(records),
               "variant": variant, "seed": options.seed, "genes": len(prep.genes),
               "proteins": len(prep.proteins), "checkpoint": "best.pt"}
    save_json(output / "training_summary.json", summary)
    return summary
