"""
Unit tests for dracox Draco mesh decompression.
"""

import os

import msgpack
import numpy as np

# Test data path
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "tile_test_data.msgpack")


def test_draco_primitive():
    from dracox import handle_draco_primitive

    with open(TEST_DATA_PATH, "rb") as f:
        test_data = msgpack.unpack(f)

    """Test decompressing Draco data from saved test data."""
    # Extract test data
    primitive = test_data["primitive"]
    views = test_data["views"]
    access = test_data["access"]

    # Should successfully decompress
    result = handle_draco_primitive(primitive, views, access)
    assert result is True, "Should successfully handle Draco primitive"

    # Check that data was decompressed (access array should be modified)
    # The primitive has POSITION, NORMAL, TEXCOORD_0 attributes
    position_idx = primitive["attributes"]["POSITION"]
    assert access[position_idx] is not None

    # check that vertices are as expected
    positions = access[position_idx]
    assert positions.shape == (162, 3)
    assert np.allclose(
        np.ptp(positions, axis=0), [6.4768610e06, 6.5278600e06, 6.5023605e06], rtol=0.01
    )

    # Check indices were decompressed
    indices_idx = primitive["indices"]
    indices = access[indices_idx]
    assert indices.shape == (251, 3)
    assert indices.dtype.kind in "iu"
    # make sure indices aren't all zeros
    assert np.ptp(indices) > 0
    # make sure indices aren't out of bounds
    assert positions[indices].shape == (251, 3, 3)

    # make sure the UV coordinates are populated with some data
    uv = access[primitive["attributes"]["TEXCOORD_0"]]
    assert uv.shape == (len(positions), 2)
    uv_ptp = np.ptp(uv, axis=0)
    assert (uv_ptp > 0.9).all()
    assert (uv_ptp < 2.0).all()


if __name__ == "__main__":
    test_draco_primitive()
