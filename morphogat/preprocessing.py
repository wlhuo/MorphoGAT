"""Training-fitted molecular, image, and PCA transformations."""
from dataclasses import asdict
from pathlib import Path
import re

import numpy as np
from sklearn.decomposition import PCA

from .config import PreprocessConfig


def is_isotype(name):
    compact = re.sub(r"[^a-z0-9]", "", name.lower())
    return "isotype" in compact and ("mouse" in compact or "rat" in compact)


def quality_control(section, config):
    library = section.counts.sum(axis=1, dtype=np.float64)
    detected = (section.counts > 0).sum(axis=1)
    mito = np.array([g.upper().startswith("MT-") for g in section.genes])
    fraction = section.counts[:, mito].sum(axis=1) / np.maximum(library, 1)
    keep = ((library > 0) & (detected >= config.min_genes)
            & (fraction <= config.max_mito_fraction))
    if keep.sum() < 2:
        raise ValueError(f"{section.name}: fewer than two spots survive RNA quality control")
    result = section.subset(keep)
    if not config.allow_partial_images and not result.image_valid.all():
        raise ValueError(f"{section.name}: {int((~result.image_valid).sum())} retained spots lack valid images; "
                         "provide aligned images or explicitly enable allow_partial_images")
    return result


def stats(matrix):
    mean = np.mean(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(matrix, axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1
    return mean, std


class Preprocessor:
    def __init__(self, config):
        self.config = config

    def rna_log(self, section):
        index = {g: i for i, g in enumerate(section.genes)}
        missing = [g for g in self.genes if g not in index]
        if missing and not self.config.allow_missing_genes:
            raise ValueError(f"{section.name}: {len(missing)} training genes are absent, e.g. {missing[:5]}")
        counts = np.zeros((len(section.spot_ids), len(self.genes)), dtype=np.float32)
        present = [(j, index[g]) for j, g in enumerate(self.genes) if g in index]
        if present:
            dest, source = zip(*present)
            counts[:, dest] = section.counts[:, source]
        # The denominator uses every input gene, before training feature selection.
        library = section.counts.sum(axis=1, dtype=np.float64)
        return np.log1p(counts * (self.config.library_size / library)[:, None]).astype(np.float32)

    def protein_counts(self, section):
        if section.protein is None:
            raise ValueError(f"{section.name}: paired protein counts are required")
        index = {p: i for i, p in enumerate(section.proteins)}
        missing = [p for p in self.proteins if p not in index]
        if missing:
            raise ValueError(f"{section.name}: missing training-panel proteins: {missing}")
        return section.protein[:, [index[p] for p in self.proteins]]

    def fit(self, training_sections, seed=0):
        if not training_sections or any(s.protein is None for s in training_sections):
            raise ValueError("At least one paired training section is required")
        genes = sorted(set.intersection(*(set(s.genes) for s in training_sections)))
        proteins = sorted(p for p in set.intersection(*(set(s.proteins) for s in training_sections))
                          if not is_isotype(p))
        if not genes or not proteins:
            raise ValueError("Training sections have no shared genes or non-isotype proteins")
        total = sum(len(s.spot_ids) for s in training_sections)
        detected = np.zeros(len(genes), dtype=np.int64)
        for section in training_sections:
            index = {g: i for i, g in enumerate(section.genes)}
            detected += (section.counts[:, [index[g] for g in genes]] > 0).sum(axis=0)
        keep = np.array([n / total >= self.config.min_gene_fraction or
                         any(p.lower().startswith(g.lower()) for p in proteins)
                         for g, n in zip(genes, detected)])
        self.genes = np.asarray(genes)[keep]
        self.proteins = np.asarray(proteins)
        if not len(self.genes):
            raise ValueError("No genes survive prevalence filtering")
        log_rna = np.concatenate([self.rna_log(s) for s in training_sections])
        self.rna_mean, self.rna_std = stats(log_rna)
        raw_protein = np.concatenate([self.protein_counts(s) for s in training_sections])
        self.protein_scale = np.exp(np.log1p(raw_protein.astype(np.float64)).mean(axis=0)).astype(np.float32)
        valid_images = [s.image_features[s.image_valid] for s in training_sections if s.image_valid.any()]
        if valid_images:
            self.image_mean, self.image_std = stats(np.concatenate(valid_images))
        else:
            self.image_mean, self.image_std = np.zeros(65, np.float32), np.ones(65, np.float32)
        normalized = self.standardize(log_rna, self.rna_mean, self.rna_std)
        n_components = min(self.config.pca_components, normalized.shape[1], normalized.shape[0] - 1)
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        pca.fit(normalized)
        self.pca_mean = pca.mean_.astype(np.float32)
        self.pca_components = np.ascontiguousarray(pca.components_, dtype=np.float32)
        return self

    def standardize(self, matrix, mean, std):
        return np.clip((matrix - mean) / std, -self.config.clip, self.config.clip).astype(np.float32)

    def transform(self, section, include_protein=True):
        x = self.standardize(self.rna_log(section), self.rna_mean, self.rna_std)
        f = self.standardize(section.image_features, self.image_mean, self.image_std)
        f[~section.image_valid] = 0
        y = None
        if include_protein:
            y = np.log1p(self.protein_counts(section) / self.protein_scale).astype(np.float32)
        return {"x": x, "y": y, "f": f, "mask": section.image_valid,
                "rna_pca": (x - self.pca_mean) @ self.pca_components.T,
                "spatial": section.spatial, "spot_ids": section.spot_ids, "name": section.name}

    def save(self, path):
        import json
        values = {key: getattr(self, key) for key in
                  ["genes", "proteins", "rna_mean", "rna_std", "protein_scale",
                   "image_mean", "image_std", "pca_mean", "pca_components"]}
        np.savez_compressed(path, config=np.asarray(json.dumps(asdict(self.config))), **values)

    @classmethod
    def load(cls, path):
        import json
        with np.load(Path(path), allow_pickle=False) as values:
            result = cls(PreprocessConfig(**json.loads(str(values["config"]))))
            for key in values.files:
                if key != "config":
                    setattr(result, key, values[key].copy())
        return result
