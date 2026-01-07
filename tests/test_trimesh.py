"""
Test dracox integration with trimesh for DRACO-compressed glTF export.

Requires: trimesh, lxml
"""

import os
import time

import numpy as np
import pytest

# Import dracox to register handlers
import dracox  # noqa: F401

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("lxml")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "cycloidal.3DXML")


def test_draco_glb_roundtrip():
    """Test loading 3DXML, exporting as GLB with/without DRACO, and roundtrip."""
    # Verify handlers are registered
    from trimesh.exchange.gltf.extensions import _handlers

    assert "KHR_draco_mesh_compression" in _handlers.get(
        "primitive_export", {}
    ), "Draco export handler not registered"

    # Load the 3DXML file
    scene = trimesh.load(DATA_PATH)
    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) > 0

    # Verify we have meshes to test
    mesh_count = sum(
        1 for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)
    )
    assert mesh_count > 0, "Scene should contain at least one mesh"

    # Export as normal GLB (no DRACO)
    glb_normal = scene.export(file_type="glb")
    assert isinstance(glb_normal, bytes)
    assert len(glb_normal) > 0

    # Export as DRACO-compressed GLB
    glb_draco = scene.export(file_type="glb", extension_draco=True)
    assert isinstance(glb_draco, bytes)
    assert len(glb_draco) > 0

    # DRACO should produce smaller output
    compression_ratio = len(glb_draco) / len(glb_normal)
    assert compression_ratio < 0.8, (
        f"DRACO compression should reduce size significantly, "
        f"got {compression_ratio:.2%} of original "
        f"({len(glb_draco)} vs {len(glb_normal)} bytes)"
    )

    # Reload normal GLB and verify roundtrip
    scene_normal = trimesh.load(trimesh.util.wrap_as_stream(glb_normal), file_type="glb")
    assert isinstance(scene_normal, trimesh.Scene)

    # Reload DRACO GLB and verify roundtrip
    scene_draco = trimesh.load(trimesh.util.wrap_as_stream(glb_draco), file_type="glb")
    assert isinstance(scene_draco, trimesh.Scene)

    # Verify mesh counts match
    assert len(scene_draco.geometry) == len(scene.geometry)

    # Verify mesh data is preserved by comparing surface area
    # Area should be within 1% after DRACO compression
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue

        orig_area = geom.area

        # Check normal GLB has matching mesh
        assert name in scene_normal.geometry, f"Missing {name} in normal GLB"
        normal_geom = scene_normal.geometry[name]
        assert isinstance(normal_geom, trimesh.Trimesh)
        assert np.isclose(normal_geom.area, orig_area, rtol=0.001), (
            f"Normal GLB area mismatch for {name}: {normal_geom.area} vs {orig_area}"
        )

        # Check DRACO GLB has matching mesh with similar area
        assert name in scene_draco.geometry, f"Missing {name} in DRACO GLB"
        draco_geom = scene_draco.geometry[name]
        assert isinstance(draco_geom, trimesh.Trimesh)
        assert np.isclose(draco_geom.area, orig_area, rtol=0.01), (
            f"DRACO area mismatch for {name}: {draco_geom.area} vs {orig_area} "
            f"(diff: {abs(draco_geom.area - orig_area) / orig_area * 100:.2f}%)"
        )


def test_draco_timing():
    """Benchmark export/import times for normal vs DRACO GLB."""
    scene = trimesh.load(DATA_PATH)

    iterations = 3

    # Test normal GLB (no compression)
    normal_export_times = []
    normal_import_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        glb_normal = scene.export(file_type="glb")
        normal_export_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        trimesh.load(trimesh.util.wrap_as_stream(glb_normal), file_type="glb")
        normal_import_times.append(time.perf_counter() - t0)

    # Test DRACO GLB (compression level 7 is hardcoded in encoder)
    draco_export_times = []
    draco_import_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        glb_draco = scene.export(file_type="glb", extension_draco=True)
        draco_export_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        trimesh.load(trimesh.util.wrap_as_stream(glb_draco), file_type="glb")
        draco_import_times.append(time.perf_counter() - t0)

    # Print timing results
    normal_size = len(glb_normal)
    draco_size = len(glb_draco)
    ratio = draco_size / normal_size * 100

    print("\n" + "=" * 70)
    print("DRACO Compression Benchmark")
    print("=" * 70)
    print(f"{'Mode':<10} {'Size':>12} {'Export':>12} {'Import':>12}")
    print("-" * 70)
    print(
        f"{'normal':<10} {normal_size:>10} (100%) "
        f"{np.mean(normal_export_times)*1000:>8.1f}ms "
        f"{np.mean(normal_import_times)*1000:>8.1f}ms"
    )
    print(
        f"{'draco':<10} {draco_size:>10} ({ratio:>3.0f}%) "
        f"{np.mean(draco_export_times)*1000:>8.1f}ms "
        f"{np.mean(draco_import_times)*1000:>8.1f}ms"
    )
    print("=" * 70)

    # Basic sanity checks
    assert draco_size < normal_size, "DRACO should compress"


if __name__ == "__main__":
    test_draco_glb_roundtrip()
    test_draco_timing()
