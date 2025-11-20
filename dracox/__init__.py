"""
# dracox

Support library for `trimesh` providing minimal decompress-only
handling for glTF's `KHR_draco_mesh_compression` using the large
`draco` codebase.
"""

from typing import Dict, List

__version__ = "0.0.1"


def handle_draco_primitive(
    primitive: Dict, views: List[bytes], access: List[Dict]
) -> bool:
    """
    Handle KHR_draco_mesh_compression extension for a glTF primitive.

    If the primitive uses Draco compression, decompress the data and
    update the accessor array in-place with decompressed geometry.

    Parameters
    ------------
    primitive : dict
      A GLTF primitive object that may contain Draco extension
    views : list
      Buffer views containing raw data
    access : list
      Accessor array to be updated with decompressed data

    Returns
    ---------
    success : bool
      True if Draco decompression was successful, False otherwise
    """
    # get the draco-compressed data if it exists
    draco_field = primitive.get("extensions", {}).get("KHR_draco_mesh_compression", None)

    if draco_field is None:
        return False

    # lazily import our C extension
    from .dracox_ext import decode_draco_buffer

    # Get the compressed data from the bufferView
    buffer_view_index = draco_field["bufferView"]
    compressed_data = views[buffer_view_index]

    # Build attribute map from Draco attribute IDs to names
    attribute_map = [
        (attr_name, attr_id) for attr_name, attr_id in draco_field["attributes"].items()
    ]

    # Decompress using dracox
    decompressed = decode_draco_buffer(compressed_data, attribute_map)

    # Update the access array with decompressed data
    # The extension spec says accessors must match decompressed data
    for attr_name in draco_field["attributes"].keys():
        if attr_name in decompressed:
            # Find the accessor index for this attribute
            if attr_name in primitive["attributes"]:
                accessor_idx = primitive["attributes"][attr_name]
                # Replace the accessor data with decompressed data
                data = decompressed[attr_name]
                print(
                    f"[dracox] Setting accessor[{accessor_idx}] ({attr_name}): dtype={data.dtype}, shape={data.shape}, min={data.min(axis=0) if len(data.shape) > 1 else data.min()}, max={data.max(axis=0) if len(data.shape) > 1 else data.max()}"
                )
                access[accessor_idx] = data

    # Handle indices if present in Draco extension
    if "indices" in primitive and "indices" in decompressed:
        indices_accessor_idx = primitive["indices"]
        data = decompressed["indices"]
        print(
            f"[dracox] Setting accessor[{indices_accessor_idx}] (indices): dtype={data.dtype}, shape={data.shape}"
        )
        access[indices_accessor_idx] = data

    return True


__all__ = ["handle_draco_primitive"]
