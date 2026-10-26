"""Command-line interfaces with optional dependencies loaded on demand."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="morphogat", description="MorphoGAT spatial protein inference")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("make-demo", help="Create a small simulated paired dataset")
    demo.add_argument("--output", required=True)
    demo.add_argument("--seed", type=int, default=7)
    check = commands.add_parser("check-data", help="Validate inputs, QC and training-fitted preprocessing")
    check.add_argument("--manifest", required=True)
    check.add_argument("--config", default="configs/reference.yaml")
    check.add_argument("--output", required=True)
    train = commands.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--config", default="configs/reference.yaml")
    train.add_argument("--output", required=True)
    train.add_argument("--variant", choices=["full", "no_histology", "no_image_fusion",
                                             "no_morphology_graph", "no_adaptive_gating"], default="full")
    train.add_argument("--seed", type=int)
    train.add_argument("--device")
    for name in ["predict", "evaluate"]:
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--split", choices=["train", "validation", "test", "external"],
                             default="external" if name == "predict" else "test")
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="auto")
    tables = commands.add_parser("evaluate-tables", help="Score aligned normalized protein CSV matrices")
    tables.add_argument("--predicted", required=True)
    tables.add_argument("--measured", required=True)
    tables.add_argument("--output", required=True)
    cluster = commands.add_parser("cluster")
    cluster.add_argument("--predictions", required=True)
    cluster.add_argument("--labels")
    cluster.add_argument("--output", required=True)
    cluster.add_argument("--neighbors", type=int, default=10)
    cluster.add_argument("--resolution", type=float, default=0.5)
    cluster.add_argument("--seed", type=int, default=0)
    cluster.add_argument("--min-label-size", type=int, default=1)
    boxplot = commands.add_parser("plot-proteins")
    boxplot.add_argument("--metrics", nargs="+", required=True)
    boxplot.add_argument("--names", nargs="+", required=True)
    boxplot.add_argument("--output", required=True)
    spatial = commands.add_parser("plot-spatial")
    spatial.add_argument("--predictions", required=True)
    selection = spatial.add_mutually_exclusive_group(required=True)
    selection.add_argument("--protein")
    selection.add_argument("--readout", choices=["spatial", "rna", "morphology", "mean_gate"])
    spatial.add_argument("--output", required=True)
    spatial.add_argument("--image")
    spatial.add_argument("--image-scale", type=float, default=1.0)
    domains = commands.add_parser("plot-domains", help="Compare annotation and method-specific spatial domains")
    domains.add_argument("--tables", nargs="+", required=True)
    domains.add_argument("--names", nargs="+", required=True)
    domains.add_argument("--output", required=True)
    domains.add_argument("--image")
    domains.add_argument("--image-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.command == "make-demo":
        from .demo import make_demo
        make_demo(args.output, args.seed)
    elif args.command == "check-data":
        check_data(args)
    elif args.command == "train":
        from .config import Config
        from .training import train as train_model
        config = Config.load(args.config)
        if args.seed is not None:
            config.training.seed = args.seed
        if args.device:
            config.training.device = args.device
        print(json.dumps(train_model(args.manifest, config, args.output, args.variant), indent=2))
    elif args.command in {"predict", "evaluate"}:
        from .inference import run_prediction
        run_prediction(args.checkpoint, args.manifest, args.split, args.output, args.device,
                        evaluate=args.command == "evaluate")
    elif args.command == "evaluate-tables":
        # Table metrics do not require a model runtime.
        from .table_evaluation import evaluate_tables
        print(evaluate_tables(args.predicted, args.measured, args.output))
    elif args.command == "cluster":
        from .analysis import cluster_domains
        print(cluster_domains(args.predictions, args.labels, args.output, args.resolution,
                              args.neighbors, args.seed, args.min_label_size))
    elif args.command == "plot-proteins":
        from .analysis import plot_protein_metrics
        plot_protein_metrics(args.metrics, args.names, args.output)
    elif args.command == "plot-spatial":
        from .analysis import plot_spatial
        plot_spatial(args.predictions, args.output, args.protein, args.readout, args.image, args.image_scale)
    elif args.command == "plot-domains":
        from .analysis import plot_domains
        plot_domains(args.tables, args.names, args.output, args.image, args.image_scale)


def check_data(args):
    from .config import Config
    from .data import load_section, read_manifest
    from .preprocessing import Preprocessor, quality_control
    from .utils import new_output_directory, save_json
    config = Config.load(args.config)
    specs, base = read_manifest(args.manifest)
    sections = []
    for spec in specs:
        section = load_section(spec, base, config.preprocess, include_protein=spec["split"] != "external")
        sections.append((spec["split"], quality_control(section, config.preprocess)))
    prep = Preprocessor(config.preprocess).fit([s for split, s in sections if split == "train"],
                                              config.training.seed)
    report = []
    for split, section in sections:
        values = prep.transform(section, include_protein=split != "external")
        report.append({"id": section.name, "split": split, "spots": len(section.spot_ids),
                       "valid_images": int(section.image_valid.sum()),
                       "genes": values["x"].shape[1], "proteins": len(prep.proteins)})
    output = new_output_directory(args.output)
    save_json(output / "data_report.json", report)
    prep.save(output / "preprocessor.npz")
    print(json.dumps(report, indent=2))
