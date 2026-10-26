"""Downstream domain analysis and publication figure inputs."""
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import new_output_directory, save_json


def cluster_domains(prediction_path, labels_path, output_path, resolution=0.5,
                    neighbors=10, seed=0, min_label_size=1):
    if resolution <= 0 or neighbors < 1 or min_label_size < 1:
        raise ValueError("Resolution, neighbor count and minimum label size must be positive")
    try:
        import anndata
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("Install requirements-analysis.txt for Leiden domain analysis") from exc
    from sklearn.metrics import adjusted_rand_score
    with np.load(prediction_path, allow_pickle=False) as data:
        prediction, ids, spatial = data["prediction"], data["spot_ids"], data["spatial"]
        proteins = data["proteins"]
    if len(ids) < 3:
        raise ValueError("Domain analysis requires at least three spots")
    adata = anndata.AnnData(prediction.astype(np.float32))
    adata.obs_names, adata.var_names = ids.astype(str), proteins.astype(str)
    adata.obsm["spatial"] = spatial
    # Use the predicted normalized protein matrix directly; annotations never
    # influence neighbor construction, resolution selection or cluster fitting.
    sc.pp.neighbors(adata, n_neighbors=min(neighbors, len(ids) - 1), use_rep="X", random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, random_state=seed, key_added="cluster",
                 flavor="igraph", n_iterations=2, directed=False)
    frame = pd.DataFrame({"barcode": ids, "x": spatial[:, 0], "y": spatial[:, 1],
                          "cluster": adata.obs["cluster"].astype(str).to_numpy()}).set_index("barcode")
    report = {"resolution": resolution, "neighbors": neighbors, "seed": seed,
              "min_label_size": min_label_size, "spots": len(ids),
              "clusters": int(frame.cluster.nunique()), "ari": None, "evaluated_spots": 0}
    if labels_path:
        from .data import read_matrix
        labels = read_matrix(labels_path)
        if "label" not in labels:
            raise ValueError("Annotation CSV must contain barcode and label columns")
        frame["label"] = labels["label"].reindex(frame.index)
        counts = frame["label"].value_counts()
        eligible = frame["label"].notna() & frame["label"].isin(counts[counts >= min_label_size].index)
        report["evaluated_spots"] = int(eligible.sum())
        if eligible.sum() >= 2:
            report["ari"] = float(adjusted_rand_score(frame.loc[eligible, "label"],
                                                      frame.loc[eligible, "cluster"]))
    output = new_output_directory(output_path)
    frame.to_csv(output / "domains.csv")
    adata.write_h5ad(output / "domains.h5ad")
    save_json(output / "domain_metrics.json", report)
    return report


def plot_protein_metrics(paths, names, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(paths) != len(names) or len(set(names)) != len(names):
        raise ValueError("Provide one unique method label per metric file")
    tables = [pd.read_csv(path).set_index("protein") for path in paths]
    if any(not table.index.is_unique for table in tables):
        raise ValueError("Duplicate protein names in a metric table")
    if any(set(table.index) != set(tables[0].index) for table in tables):
        raise ValueError("Method comparisons require the same protein panel")
    values = [table.spearman.dropna().to_numpy() for table in tables]
    if any(len(value) == 0 for value in values):
        raise ValueError("Each method must have at least one defined Spearman correlation")
    figure, axis = plt.subplots(figsize=(max(5, len(paths) * 1.3), 4))
    boxes = axis.boxplot(values, labels=names, patch_artist=True, widths=0.55, showfliers=True)
    colors = plt.get_cmap("Set2").colors
    for i, box in enumerate(boxes["boxes"]):
        box.set_facecolor(colors[i % len(colors)])
    axis.set_ylabel("Protein-wise Spearman correlation")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    rows = []
    for name, table in zip(names, tables):
        valid = table.spearman.dropna()
        rows.append({"method": name, "proteins": len(table), "valid_proteins": len(valid),
                     "median_spearman": float(valid.median()), "mean_spearman": float(valid.mean())})
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)


def plot_spatial(prediction_path, output_path, protein=None, readout=None, image_path=None,
                 image_scale=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    with np.load(prediction_path, allow_pickle=False) as data:
        coords = data["spatial"].copy()
        if protein is not None:
            matches = np.flatnonzero(data["proteins"] == protein)
            if not len(matches):
                raise ValueError(f"Protein not in predicted panel: {protein}")
            values, title = data["prediction"][:, matches[0]], protein
        elif readout in {"spatial", "rna", "morphology"}:
            values = data["view_weights"][:, ["spatial", "rna", "morphology"].index(readout)]
            title = f"{readout.capitalize()} graph weight"
        elif readout == "mean_gate":
            values, title = data["fusion_gates"].mean(axis=1), "Mean image fusion gate"
        else:
            raise ValueError("Select a protein or a view/gate readout")
    figure, axis = plt.subplots(figsize=(6, 5))
    if image_path:
        # Caller must confirm that spatial coordinates are full-resolution pixels.
        coords = coords * image_scale
        with Image.open(image_path) as image:
            axis.imshow(image)
    scatter = axis.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=12, linewidths=0)
    if not image_path:
        axis.invert_yaxis()
    axis.set_aspect("equal")
    axis.set_title(title)
    axis.set_axis_off()
    figure.colorbar(scatter, ax=axis, shrink=0.7)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_domains(paths, names, output_path, image_path=None, image_scale=1.0):
    """Plot a shared annotation map and aligned method-specific domain maps."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    if len(paths) != len(names) or len(set(names)) != len(names):
        raise ValueError("Provide one unique method label per domain table")
    tables = [pd.read_csv(path, index_col=0, dtype={"barcode": str}) for path in paths]
    for table in tables:
        if not table.index.is_unique or not {"x", "y", "cluster"} <= set(table.columns):
            raise ValueError("Domain CSVs need unique barcode rows and x,y,cluster columns")
        if set(table.index) != set(tables[0].index):
            raise ValueError("Domain comparisons require identical barcode sets")
    tables = [table.reindex(tables[0].index) for table in tables]
    for table in tables[1:]:
        if not np.allclose(table[["x", "y"]], tables[0][["x", "y"]]):
            raise ValueError("Domain comparisons require matching coordinates")
    panels = []
    if "label" in tables[0] and tables[0]["label"].notna().any():
        panels.append(("Reference", tables[0]["label"], tables[0]))
    for path, name, table in zip(paths, names, tables):
        title = name
        metrics_file = Path(path).parent / "domain_metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            if metrics.get("ari") is not None:
                title += f" (ARI {metrics['ari']:.3f})"
        panels.append((title, table["cluster"], table))
    columns = min(3, len(panels))
    rows = (len(panels) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.7 * rows), squeeze=False)
    background = None
    if image_path:
        with Image.open(image_path) as source:
            background = np.asarray(source.convert("RGB"))
    for axis, (title, values, table) in zip(axes.ravel(), panels):
        coords = table[["x", "y"]].to_numpy()
        if background is not None:
            axis.imshow(background)
            coords = coords * image_scale
        valid = values.notna().to_numpy()
        labels = values[valid].astype(str)
        categories = sorted(labels.unique())
        for index, category in enumerate(categories):
            mask = valid.copy()
            mask[valid] = labels.to_numpy() == category
            axis.scatter(coords[mask, 0], coords[mask, 1], s=9,
                         color=plt.get_cmap("tab20")(index % 20), linewidths=0, label=category)
        if background is None:
            axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.set_title(title)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False,
                    fontsize=7, ncol=min(4, max(1, len(categories))))
    for axis in axes.ravel()[len(panels):]:
        axis.set_visible(False)
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
