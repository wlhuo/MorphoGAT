"""Evaluation of baseline outputs, without loading a model or protein encoder."""
from .data import read_matrix
from .metrics import concordance
from .utils import new_output_directory, save_json


def evaluate_tables(prediction_path, truth_path, output_path):
    predicted, truth = read_matrix(prediction_path), read_matrix(truth_path)
    if set(predicted.index) != set(truth.index) or set(predicted.columns) != set(truth.columns):
        raise ValueError("Prediction and target CSVs must have identical spot and protein identifier sets")
    truth = truth.loc[predicted.index, predicted.columns]
    per_protein, per_spot, summary = concordance(predicted.to_numpy(), truth.to_numpy(),
                                                predicted.columns, predicted.index)
    output = new_output_directory(output_path)
    per_protein.to_csv(output / "protein_metrics.csv", index=False)
    per_spot.to_csv(output / "spot_metrics.csv", index=False)
    save_json(output / "summary.json", summary)
    return summary
