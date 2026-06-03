"""
# dracox

Support library for `trimesh` providing Draco compression/decompression
for glTF's `KHR_draco_mesh_compression` extension.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

__version__ = "0.0.2"

# Module-level encoding settings. Set these before calling trimesh .export()
# to control Draco compression quality per-thread.
import threading as _threading

_settings = _threading.local()


def set_encoder_options(
    compression_level: int = 7,
    quantization_position: int = 16,
    quantization_normal: int = 14,
    quantization_tex_coord: int = 14,
) -> None:
    """Set Draco encoder options for the current thread."""
    _settings.compression_level = compression_level
    _settings.quantization_position = quantization_position
    _settings.quantization_normal = quantization_normal
    _settings.quantization_tex_coord = quantization_tex_coord


def clear_encoder_options() -> None:
    """Reset encoder options to defaults."""
    _settings.compression_level = None
    _settings.quantization_position = None
    _settings.quantization_normal = None
    _settings.quantization_tex_coord = None

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
        uv = mesh.visual.uv.copy()[:, :2].astype("float32")
        # Flip Y to match glTF convention (trimesh does this in its own export)
        uv[:, 1] = 1.0 - uv[:, 1]
        texcoords = uv

    # Encode using dracox with thread-local settings
    encode_kwargs = {}
    if (cl := getattr(_settings, "compression_level", None)) is not None:
        encode_kwargs["compression_level"] = cl
    if (qp := getattr(_settings, "quantization_position", None)) is not None:
        encode_kwargs["quantization_position"] = qp
    if (qn := getattr(_settings, "quantization_normal", None)) is not None:
        encode_kwargs["quantization_normal"] = qn
    if (qt := getattr(_settings, "quantization_tex_coord", None)) is not None:
        encode_kwargs["quantization_tex_coord"] = qt

    result = encode_draco_buffer(
        vertices=vertices,
        faces=faces,
        normals=normals,
        texcoords=texcoords,
        **encode_kwargs,
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
    # Per glTF spec: accessor count MUST remain correct (viewers read it to know
    # how many elements the Draco buffer contains). Only the raw buffer is stubbed.
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


__all__ = ["_draco_decode", "_draco_encode", "handle_draco_primitive", "set_encoder_options", "clear_encoder_options"]
