import numpy as np
import pytest

from morphogat.config import PreprocessConfig
from morphogat.data import Section


@pytest.fixture
def qc_config():
    return PreprocessConfig(min_genes=1, pca_components=2, allow_partial_images=True)


@pytest.fixture
def paired_section():
    rng = np.random.default_rng(8)
    return Section("paired", rng.poisson(3, (8, 5)).astype(np.float32),
                   np.array(["CD3", "GENE1", "GENE2", "GENE3", "MT-A"]),
                   np.array([f"s{i}" for i in range(8)]), rng.random((8, 2)).astype(np.float32),
                   rng.random((8, 65)).astype(np.float32), np.array([1, 1, 1, 1, 1, 1, 0, 0]),
                   rng.poisson(4, (8, 3)).astype(np.float32),
                   np.array(["CD3_TotalSeq", "CD19", "Mouse IgG1 Isotype Control"]), 8).validate()
