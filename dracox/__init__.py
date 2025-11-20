"""
dracox: Minimal Draco mesh decompression for glTF KHR_draco_mesh_compression

This module provides a simple interface for decompressing Draco-compressed
mesh data from glTF files.
"""

try:
    from .dracox_ext import decode_draco_buffer
    __all__ = ['decode_draco_buffer']
except ImportError as e:
    import warnings
    warnings.warn(f"dracox C++ extension not available: {e}")
    __all__ = []

__version__ = "0.0.1"
