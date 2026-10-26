from dataclasses import replace

import numpy as np
import pytest

from morphogat.preprocessing import Preprocessor, quality_control


def test_protein_axis_isotype_and_roundtrip(paired_section, qc_config, tmp_path):
    prep = Preprocessor(qc_config).fit([paired_section])
    assert prep.proteins.tolist() == ["CD19", "CD3_TotalSeq"]
    expected = np.exp(np.log1p(paired_section.protein[:, [1, 0]]).mean(axis=0))
    np.testing.assert_allclose(prep.protein_scale, expected, rtol=1e-6)
    normalized = prep.transform(paired_section)
    np.testing.assert_allclose(normalized["y"], np.log1p(paired_section.protein[:, [1, 0]] / expected), rtol=1e-6)
    prep.save(tmp_path / "preprocessor.npz")
    loaded = Preprocessor.load(tmp_path / "preprocessor.npz")
    for key in ("x", "y", "f", "rna_pca"):
        np.testing.assert_allclose(loaded.transform(paired_section)[key], normalized[key], rtol=1e-6, atol=1e-6)


def test_fit_excludes_heldout_and_invalid_images(paired_section, qc_config):
    section = paired_section
    altered = replace(section, image_features=section.image_features.copy())
    altered.image_features[~section.image_valid] = 100000
    prep = Preprocessor(qc_config).fit([section])
    other = Preprocessor(qc_config).fit([altered])
    np.testing.assert_array_equal(prep.image_mean, other.image_mean)
    assert np.all(prep.transform(altered)["f"][~section.image_valid] == 0)
    old_stats = prep.rna_mean.copy(), prep.protein_scale.copy(), prep.pca_components.copy()
    heldout = replace(section, name="heldout", counts=section.counts * 20,
                      protein=section.protein * 100)
    prep.transform(heldout)
    np.testing.assert_array_equal(prep.rna_mean, old_stats[0])
    np.testing.assert_array_equal(prep.protein_scale, old_stats[1])
    np.testing.assert_array_equal(prep.pca_components, old_stats[2])


def test_library_denominator_uses_unselected_genes(paired_section, qc_config):
    prep = Preprocessor(qc_config).fit([paired_section])
    prep.genes = np.array(["CD3"])
    expected = np.log1p(paired_section.counts[:, :1] * 10000 / paired_section.counts.sum(axis=1)[:, None])
    np.testing.assert_allclose(prep.rna_log(paired_section), expected, rtol=1e-6)


def test_missing_features_require_explicit_policy(paired_section, qc_config):
    prep = Preprocessor(qc_config).fit([paired_section])
    heldout = replace(paired_section, counts=paired_section.counts[:, 1:], genes=paired_section.genes[1:])
    with pytest.raises(ValueError, match="training genes are absent"):
        prep.transform(heldout)
    prep.config.allow_missing_genes = True
    assert np.isfinite(prep.transform(heldout)["x"]).all()


def test_qc_before_feature_selection_and_complete_image_default(paired_section, qc_config):
    counts = paired_section.counts.copy()
    counts[0] = 0
    counts[1] = [1, 1, 1, 1, 1000]
    section = replace(paired_section, counts=counts)
    result = quality_control(section, qc_config)
    assert "s0" not in result.spot_ids and "s1" not in result.spot_ids
    qc_config.allow_partial_images = False
    with pytest.raises(ValueError, match="lack valid images"):
        quality_control(section, qc_config)


def test_rare_gene_target_prefix_exception(paired_section, qc_config):
    qc_config.min_gene_fraction = 1.0
    counts = paired_section.counts.copy()
    counts[:, 0] = 0
    section = replace(paired_section, counts=counts)
    prep = Preprocessor(qc_config).fit([section])
    assert "CD3" in prep.genes


def test_bad_counts_and_masks_rejected(paired_section):
    counts = paired_section.counts.copy()
    counts[0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        replace(paired_section, counts=counts).validate()
    with pytest.raises(ValueError, match="binary"):
        replace(paired_section, image_valid=np.full(8, 2)).validate()
