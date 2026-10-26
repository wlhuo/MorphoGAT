"""Protein-wise and spot-wise concordance, with explicit constant-vector handling."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def correlation(a, b, ranked=False):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    if ranked:
        a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def concordance(prediction, truth, proteins, spot_ids):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if prediction.shape != truth.shape or prediction.shape != (len(spot_ids), len(proteins)):
        raise ValueError("Prediction, target and identifier dimensions differ")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Metrics require finite predictions and targets")
    protein_rows, spot_rows = [], []
    for i, name in enumerate(proteins):
        a, b = prediction[:, i], truth[:, i]
        protein_rows.append({"protein": name, "spearman": correlation(a, b, True),
                             "pearson": correlation(a, b), "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                             "spots": len(a)})
    for i, name in enumerate(spot_ids):
        a, b = prediction[i], truth[i]
        spot_rows.append({"barcode": name, "spearman": correlation(a, b, True),
                          "pearson": correlation(a, b), "rmse": float(np.sqrt(np.mean((a - b) ** 2)))})
    per_protein, per_spot = pd.DataFrame(protein_rows), pd.DataFrame(spot_rows)
    summary = {"rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
               "proteins": len(proteins), "spots": len(spot_ids)}
    for key in ("spearman", "pearson"):
        valid = per_protein[key].dropna()
        summary[f"median_protein_{key}"] = float(valid.median()) if len(valid) else None
        summary[f"mean_protein_{key}"] = float(valid.mean()) if len(valid) else None
        summary[f"valid_proteins_{key}"] = len(valid)
    return per_protein, per_spot, summary
