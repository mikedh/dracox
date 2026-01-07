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
    tree = ctx["tree"]
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

    # Replace uncompressed buffers with empty stubs
    # Per glTF spec, accessor data MAY be empty when draco extension is present
    accessors = tree["accessors"]
    buffer_keys = list(buffer_items.keys())

    # Find accessor indices that belong to this primitive
    accessor_indices = []
    if "indices" in primitive:
        accessor_indices.append(primitive["indices"])
    for attr_idx in primitive.get("attributes", {}).values():
        accessor_indices.append(attr_idx)

    # Replace buffers for these accessors with minimal 4-byte stubs
    # The accessor's bufferView field tells us which buffer to stub
    accessor_list = list(accessors.values()) if hasattr(accessors, 'values') else accessors
    for acc_idx in accessor_indices:
        if acc_idx < len(accessor_list):
            accessor = accessor_list[acc_idx]
            # Get the bufferView index from the accessor
            bv_idx = accessor.get("bufferView")
            if bv_idx is not None and bv_idx < len(buffer_keys):
                key = buffer_keys[bv_idx]
                # Replace with 4-byte stub (minimum for alignment)
                buffer_items[key] = b'\x00\x00\x00\x00'
            # Update accessor count to 0 since data is in draco buffer
            accessor["count"] = 0

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

def handle_draco_primitive(primitive, views, access):
    """
    Handle KHR_draco_mesh_compression for a glTF primitive.

    Parameters
    ----------
    primitive : dict
        The primitive dict with extensions data
    views : list
        List of buffer views (bytes)
    access : list
        List of accessors (will be modified in-place)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    from .dracox_ext import decode_draco_buffer

    ext_data = primitive.get("extensions", {}).get("KHR_draco_mesh_compression")
    if ext_data is None:
        return False

    buffer_view_index = ext_data["bufferView"]
    compressed_data = views[buffer_view_index]
    attribute_map = [(name, id) for name, id in ext_data["attributes"].items()]

    decompressed = decode_draco_buffer(compressed_data, attribute_map)

    for attr_name in ext_data["attributes"].keys():
        if attr_name in decompressed:
            primitive["attributes"][attr_name] = len(access)
            access.append(decompressed[attr_name])

    if "indices" in primitive and "indices" in decompressed:
        primitive["indices"] = len(access)
        access.append(decompressed["indices"])

    return True


__all__ = ["_draco_decode", "_draco_encode", "handle_draco_primitive"]
