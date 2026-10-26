import numpy as np
import pandas as pd
from PIL import Image

from morphogat.images import FEATURE_NAMES, describe_patch, extract_descriptors, read_positions


def test_white_patch_descriptor():
    feature = describe_patch(np.full((64, 64, 3), 255, np.uint8))
    assert feature.shape == (65,)
    assert len(FEATURE_NAMES) == 65
    np.testing.assert_allclose(feature[:3], 1)
    np.testing.assert_allclose(feature[3:6], 0)
    np.testing.assert_allclose(feature[6:15], 1)
    np.testing.assert_allclose(feature[15:17], 0)
    for histogram in feature[17:].reshape(3, 16):
        assert histogram[-1] == 1 and histogram.sum() == 1


def test_boundary_padding_and_invalid_centers(tmp_path):
    path = tmp_path / "black.png"
    Image.new("RGB", (10, 10), "black").save(path)
    features, mask = extract_descriptors(path, [[0, 0], [5, 5], [-1, 0], [10, 5], [np.nan, 0]], 4)
    assert mask.tolist() == [True, True, False, False, False]
    np.testing.assert_allclose(features[0, :3], 0.75)
    np.testing.assert_allclose(features[1, :3], 0)
    assert np.all(features[~mask] == 0)


def test_position_order_and_legacy_format(tmp_path):
    legacy = tmp_path / "positions.csv"
    legacy.write_text("s2,1,0,0,20,30\ns1,1,0,0,40,50\n", encoding="utf-8")
    positions = read_positions(legacy).reindex(["s1", "s2"])
    np.testing.assert_array_equal(positions.to_numpy(), [[50, 40], [30, 20]])
    current = tmp_path / "current.csv"
    pd.DataFrame({"barcode": ["a"], "x": [3], "y": [4]}).to_csv(current, index=False)
    np.testing.assert_array_equal(read_positions(current).to_numpy(), [[3, 4]])
