"""Section-level objective, including the manuscript's stop-gradient paths."""
import torch
from torch.nn import functional as F


def rmse(prediction, target):
    return torch.sqrt(F.mse_loss(prediction, target) + 1e-8)


def joint_loss(output, section, config, epoch):
    terms = {
        "rna": rmse(output["rna_reconstruction"], section["x"]),
        "protein": rmse(output["protein_reconstruction"], section["y"]),
        "prediction": rmse(output["prediction"], section["y"]),
        "cross": rmse(output["cross_rna"], section["x"]),
        "alignment": F.mse_loss(output["fused_latent"], output["protein_latent"].detach()),
    }
    ramp = min(1.0, epoch / config.align_warmup) if config.align_warmup else 1.0
    total = (config.lambda_rna * terms["rna"] + config.lambda_protein * terms["protein"]
             + config.lambda_prediction * terms["prediction"] + config.lambda_cross * terms["cross"]
             + config.lambda_alignment * ramp * terms["alignment"])
    return total, terms
