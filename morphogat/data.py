"""Input formats, spot alignment, and section-level partition validation."""
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .images import extract_descriptors, read_positions


def unique_names(values, label):
    names = np.asarray(values).astype(str)
    if names.ndim != 1 or not len(names) or any(not n or n == "nan" for n in names):
        raise ValueError(f"{label} must be a nonempty one-dimensional identifier array")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate {label}; resolve identifiers before loading")
    return names


def count_matrix(values, label):
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{label} must be a finite, nonnegative raw count matrix")
    return result


@dataclass
class Section:
    name: str
    counts: np.ndarray
    genes: np.ndarray
    spot_ids: np.ndarray
    spatial: np.ndarray
    image_features: np.ndarray
    image_valid: np.ndarray
    protein: np.ndarray | None = None
    proteins: np.ndarray | None = None
    input_spots: int = 0

    def subset(self, indices):
        return replace(self, counts=self.counts[indices], spot_ids=self.spot_ids[indices],
                       spatial=self.spatial[indices], image_features=self.image_features[indices],
                       image_valid=self.image_valid[indices],
                       protein=None if self.protein is None else self.protein[indices])

    def validate(self):
        self.genes = unique_names(self.genes, "genes")
        self.spot_ids = unique_names(self.spot_ids, "spots")
        self.counts = count_matrix(self.counts, "RNA")
        n = len(self.spot_ids)
        if self.counts.shape != (n, len(self.genes)):
            raise ValueError(f"{self.name}: RNA matrix and identifier dimensions differ")
        self.spatial = np.asarray(self.spatial, dtype=np.float32)
        if self.spatial.shape != (n, 2) or not np.isfinite(self.spatial).all():
            raise ValueError(f"{self.name}: finite (spots, 2) spatial coordinates required")
        self.image_features = np.asarray(self.image_features, dtype=np.float32)
        raw_mask = np.asarray(self.image_valid)
        if raw_mask.shape != (n,) or not np.isin(raw_mask, [0, 1]).all():
            raise ValueError(f"{self.name}: image validity must be a binary vector")
        self.image_valid = raw_mask.astype(bool)
        if (self.image_features.shape != (n, 65) or
                not np.isfinite(self.image_features[self.image_valid]).all()):
            raise ValueError(f"{self.name}: valid image descriptors must be finite and 65-dimensional")
        self.image_features[~self.image_valid] = 0
        if self.protein is not None:
            self.proteins = unique_names(self.proteins, "proteins")
            self.protein = count_matrix(self.protein, "protein")
            if self.protein.shape != (n, len(self.proteins)):
                raise ValueError(f"{self.name}: protein dimensions differ")
        return self


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sections"), list):
        raise ValueError("Manifest must contain a sections list")
    sections = manifest["sections"]
    names, sources = set(), {}
    for spec in sections:
        name = spec.get("id", "")
        if (not isinstance(name, str) or not name or name in {".", ".."}
                or name.strip() != name or name.endswith(".")
                or Path(name).name != name or any(c in name for c in '/\\:')):
            raise ValueError("Section IDs must be nonempty filenames without path separators")
        if name in names:
            raise ValueError(f"Duplicate section ID: {name}")
        names.add(name)
        if spec.get("split") not in {"train", "validation", "test", "external"}:
            raise ValueError(f"{name}: split must be train, validation, test or external")
        source = spec.get("npz", spec.get("rna"))
        if not source:
            raise ValueError(f"{name}: npz or rna path is required")
        resolved = str((path.parent / source).resolve()).casefold()
        if resolved in sources:
            raise ValueError(f"Input reused by sections {sources[resolved]} and {name}")
        sources[resolved] = name
    return sections, path.parent


def read_matrix(path):
    if Path(path).suffix.lower() == ".h5ad":
        try:
            import anndata
        except ImportError as exc:
            raise ImportError("Install requirements-analysis.txt to read h5ad files") from exc
        adata = anndata.read_h5ad(path)
        values = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        frame = pd.DataFrame(values, index=adata.obs_names.astype(str),
                             columns=adata.var_names.astype(str))
    else:
        header = pd.read_csv(path, nrows=0).columns
        if len(set(header)) != len(header) or any(str(x).endswith(".1") for x in header[1:]):
            # Pandas may disambiguate duplicate headers automatically.
            import csv
            with Path(path).open(encoding="utf-8-sig", newline="") as stream:
                original = next(csv.reader(stream))
            if len(original) != len(set(original)):
                raise ValueError(f"Duplicate matrix columns: {path}")
        frame = pd.read_csv(path, index_col=0, dtype={header[0]: str})
    frame.index = frame.index.astype(str)
    unique_names(frame.index, f"row identifiers in {path}")
    unique_names(frame.columns, f"column identifiers in {path}")
    return frame


def load_section(spec, base, preprocess, include_protein=True):
    path = lambda key: base / spec[key]
    if "npz" in spec:
        with np.load(path("npz"), allow_pickle=False) as data:
            counts, genes, ids = data["counts"], data["genes"], data["spot_ids"]
            spatial = data["spatial"]
            features = data["image_features"] if "image_features" in data else np.zeros((len(ids), 65))
            valid = data["image_valid"] if "image_valid" in data else np.zeros(len(ids), bool)
            protein = data["protein"] if include_protein and "protein" in data else None
            proteins = data["proteins"] if protein is not None else None
    else:
        rna = read_matrix(path("rna"))
        # Never open protein files on the prediction-only path.
        prot = read_matrix(path("protein")) if include_protein and spec.get("protein") else None
        ids = rna.index if prot is None else rna.index[rna.index.isin(prot.index)]
        if len(ids) == 0:
            raise ValueError(f"{spec['id']}: RNA and protein have no matching barcodes")
        rna = rna.loc[ids]
        coords = read_matrix(path("spatial"))
        if not {"x", "y"} <= set(coords.columns) or not ids.isin(coords.index).all():
            raise ValueError(f"{spec['id']}: spatial table must contain x,y for every retained barcode")
        spatial = coords.loc[ids, ["x", "y"]].to_numpy()
        counts, genes = rna.to_numpy(), rna.columns.to_numpy()
        protein = None if prot is None else prot.loc[ids].to_numpy()
        proteins = None if prot is None else prot.columns.to_numpy()
        features, valid = np.zeros((len(ids), 65), dtype=np.float32), np.zeros(len(ids), bool)
    ids = np.asarray(ids).astype(str)
    if spec.get("image"):
        if spec.get("positions"):
            centers = read_positions(path("positions")).reindex(ids).to_numpy(dtype=float)
        elif spec.get("spatial_is_pixel", False):
            centers = np.asarray(spatial, dtype=float)
        else:
            raise ValueError("An image requires positions or explicit spatial_is_pixel: true")
        scale = spec.get("image_scale", 1.0)
        if spec.get("scalefactors"):
            if not spec.get("scale_key"):
                raise ValueError("Provide scale_key when using a scalefactors JSON file")
            with path("scalefactors").open(encoding="utf-8") as stream:
                scale = json.load(stream)[spec["scale_key"]]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Image scale must be finite and positive")
        features, valid = extract_descriptors(path("image"), centers * scale, preprocess.patch_size)
    return Section(spec["id"], counts, genes, ids, spatial, features, valid,
                   protein, proteins, len(ids)).validate()


def load_split(specs, base, config, split, include_protein=True):
    return [load_section(spec, base, config, include_protein)
            for spec in specs if spec["split"] == split]
