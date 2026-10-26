"""Run a fixed split across seeds and ablations, then collect held-out metrics."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morphogat.config import Config
from morphogat.inference import run_prediction
from morphogat.model import VARIANTS
from morphogat.training import train
from morphogat.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/reference.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        for variant in args.variants:
            config = Config.load(args.config)
            config.training.seed, config.training.device = seed, args.device
            folder = Path(args.output) / f"{variant}_seed{seed}"
            train(args.manifest, config, folder / "model", variant)
            results = run_prediction(folder / "model" / "best.pt", args.manifest, "test",
                                     folder / "test", args.device, True)
            rows.extend({"seed": seed, "variant": variant, **result} for result in results)
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(args.output) / "all_runs.csv", index=False)
    metrics = ["median_protein_spearman", "mean_protein_spearman", "rmse"]
    frame.groupby(["variant", "section"])[metrics].agg(["mean", "std", "count"]).to_csv(
        Path(args.output) / "across_seed_summary.csv")
    save_json(Path(args.output) / "experiment_settings.json", vars(args))


if __name__ == "__main__":
    main()
