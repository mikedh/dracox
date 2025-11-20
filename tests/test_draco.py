"""
Unit tests for dracox Draco mesh decompression.
"""

import os

import msgpack
import pytest

# Test data path
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "tile_test_data.msgpack")

# Check if dracox is available
try:
    import dracox

    DRACOX_AVAILABLE = True
except ImportError:
    DRACOX_AVAILABLE = False


@pytest.fixture
def test_data():
    """Load test data from msgpack file."""
    if not os.path.exists(TEST_DATA_PATH):
        pytest.skip(f"Test data file not found: {TEST_DATA_PATH}")

    with open(TEST_DATA_PATH, "rb") as f:
        data = msgpack.unpack(f, raw=False)
    return data


@pytest.mark.skipif(not DRACOX_AVAILABLE, reason="dracox extension not built")
def test_load_draco_compressed_data(test_data):
    """Test decompressing Draco data from saved test data."""
    # Extract test data
    primitive = test_data["primitive"]
    views = [bytearray(v) for v in test_data["views"]]
    access = test_data["access"]

    # Should successfully decompress
    result = dracox.handle_draco_primitive(primitive, views, access)
    assert result is True, "Should successfully handle Draco primitive"

    # Check that data was decompressed (access array should be modified)
    # The primitive has POSITION, NORMAL, TEXCOORD_0 attributes
    position_idx = primitive["attributes"]["POSITION"]
    assert access[position_idx] is not None

    # Verify we got the expected number of vertices
    import numpy as np

    positions = np.array(access[position_idx])
    assert len(positions) == 162, f"Expected 162 vertices, got {len(positions)}"

    # Check indices were decompressed
    indices_idx = primitive["indices"]
    indices = np.array(access[indices_idx])
    # Indices are stored as Nx3 array (N faces, 3 vertices each)
    assert len(indices) == 251, f"Expected 251 faces, got {len(indices)}"
    assert indices.shape[1] == 3, f"Expected 3 vertices per face, got {indices.shape[1]}"


@pytest.mark.skipif(not DRACOX_AVAILABLE, reason="dracox extension not built")
def test_handle_draco_primitive_no_extension():
    """Test handle_draco_primitive returns False when no extension present."""
    primitive = {"attributes": {"POSITION": 0}}
    views = []
    access = []

    result = dracox.handle_draco_primitive(primitive, views, access)
    assert result is False, "Should return False when no extension present"


@pytest.mark.skipif(not DRACOX_AVAILABLE, reason="dracox extension not built")
def test_handle_draco_primitive_different_extension():
    """Test handle_draco_primitive ignores other extensions."""
    primitive = {"attributes": {"POSITION": 0}, "extensions": {"KHR_materials_unlit": {}}}
    views = []
    access = []

    result = dracox.handle_draco_primitive(primitive, views, access)
    assert result is False, "Should return False for non-Draco extensions"


@pytest.mark.skipif(not DRACOX_AVAILABLE, reason="dracox extension not built")
def test_decode_draco_buffer_invalid_data():
    """Test decode_draco_buffer with invalid data raises error."""
    with pytest.raises(RuntimeError):
        dracox.decode_draco_buffer(b"invalid data", [("POSITION", 0)])


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v"])
