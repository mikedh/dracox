"""
# dracox

Support library for `trimesh` providing minimal decompress-only
handling for glTF's `KHR_draco_mesh_compression` using the large
`draco` codebase.
"""

from typing import Dict, Optional

__version__ = "0.0.1"


def _draco_mesh_compression(ext_data: Dict, **kwargs) -> Optional[Dict]:
    """
    Handle KHR_draco_mesh_compression extension for a glTF primitive.

    Registered as a handler for scope="primitive".

    Parameters
    ----------
    ext_data : dict
        The extension data from KHR_draco_mesh_compression
    **kwargs
        Must contain 'views' and 'access' from the loader

    Returns
    -------
    result : dict or None
        Dict with decompressed geometry data, or None on failure
    """
    views = kwargs.get("views")
    access = kwargs.get("access")
    primitive = kwargs.get("primitive")

    if views is None or access is None or primitive is None:
        return None

    # lazily import our C extension
    from .dracox_ext import decode_draco_buffer

    # Get the compressed data from the bufferView
    buffer_view_index = ext_data["bufferView"]
    compressed_data = views[buffer_view_index]

    # Build attribute map from Draco attribute IDs to names
    attribute_map = [
        (attr_name, attr_id) for attr_name, attr_id in ext_data["attributes"].items()
    ]

    # Decompress using dracox
    decompressed = decode_draco_buffer(compressed_data, attribute_map)

    # Update the access array with decompressed data
    for attr_name in ext_data["attributes"].keys():
        if attr_name not in decompressed:
            continue
        # append the decompressed data as a new accessor
        primitive["attributes"][attr_name] = len(access)
        access.append(decompressed[attr_name])

    # Handle indices if present
    if "indices" in primitive and "indices" in decompressed:
        primitive["indices"] = len(access)
        access.append(decompressed["indices"])

    return {"decompressed": True}


def _register_handlers():
    """Register dracox handlers with trimesh's gltf extension system."""
    try:
        from trimesh.exchange.gltf.extensions import register_handler

        register_handler("KHR_draco_mesh_compression", scope="primitive_preprocess")(
            _draco_mesh_compression
        )
    except ImportError:
        # trimesh not available, skip registration
        pass


# Register on import
_register_handlers()

__all__ = ["_draco_mesh_compression"]
