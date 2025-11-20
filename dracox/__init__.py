"""
dracox: Minimal Draco mesh decompression for glTF KHR_draco_mesh_compression

This module provides a simple interface for decompressing Draco-compressed
mesh data from glTF files.
"""

try:
    from .dracox_ext import decode_draco_buffer
    _AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"dracox C++ extension not available: {e}")
    decode_draco_buffer = None
    _AVAILABLE = False

__version__ = "0.0.1"


def handle_draco_primitive(primitive, views, access, log=None):
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
    log : logging.Logger, optional
      Logger for warnings and debug messages
      
    Returns
    ---------
    success : bool
      True if Draco decompression was successful, False otherwise
    """
    # Check if this primitive uses Draco compression
    if "extensions" not in primitive:
        return False
    if "KHR_draco_mesh_compression" not in primitive["extensions"]:
        return False
    
    if not _AVAILABLE:
        if log:
            log.warning(
                "KHR_draco_mesh_compression extension found but dracox not available. "
                "Install with: pip install dracox"
            )
        return False
    
    try:
        draco_ext = primitive["extensions"]["KHR_draco_mesh_compression"]
        
        # Get the compressed data from the bufferView
        buffer_view_index = draco_ext["bufferView"]
        compressed_data = views[buffer_view_index]
        
        # Build attribute map from Draco attribute IDs to names
        attribute_map = [
            (attr_name, attr_id)
            for attr_name, attr_id in draco_ext["attributes"].items()
        ]
        
        # Decompress using dracox
        decompressed = decode_draco_buffer(
            bytes(compressed_data), attribute_map
        )
        
        # Update the access array with decompressed data
        # The extension spec says accessors must match decompressed data
        for attr_name in draco_ext["attributes"].keys():
            if attr_name in decompressed:
                # Find the accessor index for this attribute
                if attr_name in primitive["attributes"]:
                    accessor_idx = primitive["attributes"][attr_name]
                    # Replace the accessor data with decompressed data
                    access[accessor_idx] = decompressed[attr_name]
        
        # Handle indices if present in Draco extension
        if "indices" in primitive and "indices" in decompressed:
            indices_accessor_idx = primitive["indices"]
            access[indices_accessor_idx] = decompressed["indices"]
        
        if log:
            log.debug(f"Decompressed Draco data with {len(decompressed)} attributes")
        return True
            
    except Exception as e:
        if log:
            log.warning(f"Failed to decompress Draco data: {e}")
        return False


__all__ = ['decode_draco_buffer', 'handle_draco_primitive']
