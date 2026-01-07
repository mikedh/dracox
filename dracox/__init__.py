"""
# dracox

Support library for `trimesh` providing Draco compression/decompression
for glTF's `KHR_draco_mesh_compression` extension.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

__version__ = "0.0.1"

if TYPE_CHECKING:
    from trimesh.exchange.gltf.extensions import (
        PrimitiveExportContext,
        PrimitivePreprocessContext,
    )


def _draco_decode(ctx: "PrimitivePreprocessContext") -> Optional[Dict[str, Any]]:
    """
    Handle KHR_draco_mesh_compression extension for decoding a glTF primitive.

    Registered as a handler for scope="primitive_preprocess".

    Parameters
    ----------
    ctx
        PrimitivePreprocessContext with:
        - data: The KHR_draco_mesh_compression extension data
        - views: List of buffer views from the glTF
        - accessors: List of accessors (mutable, will be appended to)
        - primitive: The primitive dict (mutable, indices/attributes will be updated)

    Returns
    -------
    result
        Dict with {"decompressed": True}, or None on failure.
    """
    # lazily import our C extension
    from .dracox_ext import decode_draco_buffer

    ext_data = ctx["data"]
    views = ctx["views"]
    accessors = ctx["accessors"]
    primitive = ctx["primitive"]

    # Get the compressed data from the bufferView
    buffer_view_index = ext_data["bufferView"]
    compressed_data = views[buffer_view_index]

    # Build attribute map from Draco attribute IDs to names
    attribute_map = [
        (attr_name, attr_id) for attr_name, attr_id in ext_data["attributes"].items()
    ]

    # Decompress using dracox
    decompressed = decode_draco_buffer(compressed_data, attribute_map)

    # Update the accessors array with decompressed data
    for attr_name in ext_data["attributes"].keys():
        if attr_name not in decompressed:
            continue
        # append the decompressed data as a new accessor
        primitive["attributes"][attr_name] = len(accessors)
        accessors.append(decompressed[attr_name])

    # Handle indices if present
    if "indices" in primitive and "indices" in decompressed:
        primitive["indices"] = len(accessors)
        accessors.append(decompressed["indices"])

    return {"decompressed": True}


def _draco_encode(ctx: "PrimitiveExportContext") -> Optional[Dict[str, Any]]:
    """
    Handle KHR_draco_mesh_compression extension for encoding a mesh primitive.

    Registered as a handler for scope="primitive_export".

    Parameters
    ----------
    ctx
        PrimitiveExportContext with:
        - mesh: trimesh.Trimesh being exported
        - name: Mesh name
        - tree: glTF tree being built (mutable)
        - buffer_items: Buffer data being built (mutable)
        - primitive: Primitive dict being built (mutable)
        - include_normals: Whether to include normals

    Returns
    -------
    result
        Dict with extension data for KHR_draco_mesh_compression, or None on failure.
    """
    # lazily import our C extension
    from .dracox_ext import encode_draco_buffer

    mesh = ctx["mesh"]
    buffer_items = ctx["buffer_items"]
    primitive = ctx["primitive"]
    include_normals = ctx["include_normals"]

    # Get mesh data
    vertices = mesh.vertices.astype("float32")
    faces = mesh.faces.astype("uint32")

    # Get optional normals
    normals = None
    if include_normals and hasattr(mesh, "vertex_normals"):
        normals = mesh.vertex_normals.astype("float32")

    # Get optional texture coordinates
    texcoords = None
    if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
        texcoords = mesh.visual.uv.astype("float32")

    # Encode using dracox
    result = encode_draco_buffer(
        vertices=vertices,
        faces=faces,
        normals=normals,
        texcoords=texcoords,
    )

    # Pad buffer to 4-byte alignment (GLTF requirement)
    compressed = result["buffer"]
    padding = (4 - len(compressed) % 4) % 4
    if padding > 0:
        compressed = compressed + b'\x00' * padding

    # Add compressed buffer to buffer_items
    # The bufferView index will be the position in the OrderedDict
    buffer_view_index = len(buffer_items)
    buf_key = f"draco_{buffer_view_index}"
    buffer_items[buf_key] = compressed

    # Build extension data with integer bufferView index
    extension_data = {
        "bufferView": buffer_view_index,
        "attributes": dict(result["attributes"]),  # Convert from nanobind dict
    }

    # Store in primitive extensions
    if "extensions" not in primitive:
        primitive["extensions"] = {}
    primitive["extensions"]["KHR_draco_mesh_compression"] = extension_data

    return extension_data


def _register_handlers():
    """Register dracox handlers with trimesh's gltf extension system."""
    try:
        from trimesh.exchange.gltf.extensions import register_handler

        # Register decode handler for import
        register_handler("KHR_draco_mesh_compression", scope="primitive_preprocess")(
            _draco_decode
        )

        # Register encode handler for export (only if encoder is available)
        try:
            from .dracox_ext import encode_draco_buffer  # noqa: F401

            register_handler("KHR_draco_mesh_compression", scope="primitive_export")(
                _draco_encode
            )
        except ImportError:
            # Encoder not available (decode-only build)
            pass

    except ImportError:
        # trimesh not available, skip registration
        pass


# Register on import
_register_handlers()

__all__ = ["_draco_decode", "_draco_encode"]
