// Minimal nanobind extension for Draco mesh decompression
// Specifically for glTF KHR_draco_mesh_compression extension

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "draco/compression/decode.h"
#include "draco/core/decoder_buffer.h"
#include "draco/mesh/mesh.h"
#include "draco/point_cloud/point_cloud.h"

namespace nb = nanobind;
using namespace nb::literals;

// Decode Draco-compressed data and return mesh data as numpy arrays
nb::dict decode_draco_buffer(
    nb::bytes compressed_data,
    const std::vector<std::tuple<std::string, int>>& attribute_map) {
    
    // Get the raw buffer from Python bytes
    const char* buffer_data = compressed_data.c_str();
    size_t buffer_size = compressed_data.size();
    
    // Create Draco decoder buffer
    draco::DecoderBuffer decoder_buffer;
    decoder_buffer.Init(buffer_data, buffer_size);
    
    // Create decoder and decode mesh
    draco::Decoder decoder;
    auto decode_result = decoder.DecodeMeshFromBuffer(&decoder_buffer);
    
    if (!decode_result.ok()) {
        throw std::runtime_error("Failed to decode Draco mesh: " + 
                                 decode_result.status().error_msg_string());
    }
    
    std::unique_ptr<draco::Mesh> mesh = std::move(decode_result).value();
    
    // Prepare the result dictionary
    nb::dict result;
    
    // Extract indices (faces)
    const int num_faces = mesh->num_faces();
    auto* faces_array = new uint32_t[num_faces * 3];
    
    for (int i = 0; i < num_faces; ++i) {
        const draco::Mesh::Face& face = mesh->face(draco::FaceIndex(i));
        faces_array[i * 3 + 0] = face[0].value();
        faces_array[i * 3 + 1] = face[1].value();
        faces_array[i * 3 + 2] = face[2].value();
    }
    
    size_t faces_shape[2] = {static_cast<size_t>(num_faces), 3};
    result["indices"] = nb::ndarray<nb::numpy, uint32_t>(
        faces_array, 2, faces_shape, nb::capsule(faces_array, [](void *p) noexcept {
            delete[] static_cast<uint32_t*>(p);
        }));
    
    // Extract attributes based on the attribute map
    // The attribute_map contains tuples of (attribute_name, attribute_id)
    for (const auto& [attr_name, attr_id] : attribute_map) {
        const draco::PointAttribute* attr = mesh->GetAttributeByUniqueId(attr_id);
        
        if (attr == nullptr) {
            continue;  // Skip if attribute not found
        }
        
        const int num_values = mesh->num_points();
        const int num_components = attr->num_components();
        const draco::DataType data_type = attr->data_type();
        
        // Allocate output array based on data type
        if (data_type == draco::DT_FLOAT32) {
            auto* data_array = new float[num_values * num_components];
            
            // Extract attribute data
            for (int i = 0; i < num_values; ++i) {
                draco::AttributeValueIndex val_index = attr->mapped_index(draco::PointIndex(i));
                attr->GetValue(val_index, data_array + i * num_components);
            }
            
            if (num_components == 1) {
                size_t shape[1] = {static_cast<size_t>(num_values)};
                result[attr_name.c_str()] = nb::ndarray<nb::numpy, float>(
                    data_array, 1, shape, nb::capsule(data_array, [](void *p) noexcept {
                        delete[] static_cast<float*>(p);
                    }));
            } else {
                size_t shape[2] = {static_cast<size_t>(num_values), 
                                  static_cast<size_t>(num_components)};
                result[attr_name.c_str()] = nb::ndarray<nb::numpy, float>(
                    data_array, 2, shape, nb::capsule(data_array, [](void *p) noexcept {
                        delete[] static_cast<float*>(p);
                    }));
            }
        } else if (data_type == draco::DT_UINT8 || data_type == draco::DT_INT8) {
            auto* data_array = new uint8_t[num_values * num_components];
            
            for (int i = 0; i < num_values; ++i) {
                draco::AttributeValueIndex val_index = attr->mapped_index(draco::PointIndex(i));
                attr->GetValue(val_index, data_array + i * num_components);
            }
            
            if (num_components == 1) {
                size_t shape[1] = {static_cast<size_t>(num_values)};
                result[attr_name.c_str()] = nb::ndarray<nb::numpy, uint8_t>(
                    data_array, 1, shape, nb::capsule(data_array, [](void *p) noexcept {
                        delete[] static_cast<uint8_t*>(p);
                    }));
            } else {
                size_t shape[2] = {static_cast<size_t>(num_values), 
                                  static_cast<size_t>(num_components)};
                result[attr_name.c_str()] = nb::ndarray<nb::numpy, uint8_t>(
                    data_array, 2, shape, nb::capsule(data_array, [](void *p) noexcept {
                        delete[] static_cast<uint8_t*>(p);
                    }));
            }
        } else if (data_type == draco::DT_UINT16 || data_type == draco::DT_INT16) {
            auto* data_array = new uint16_t[num_values * num_components];
            
            for (int i = 0; i < num_values; ++i) {
                draco::AttributeValueIndex val_index = attr->mapped_index(draco::PointIndex(i));
                attr->GetValue(val_index, data_array + i * num_components);
            }
            
            if (num_components == 1) {
                size_t shape[1] = {static_cast<size_t>(num_values)};
                result[attr_name.c_str()] = nb::ndarray<nb::numpy, uint16_t>(
                    data_array, 1, shape, nb::capsule(data_array, [](void *p) noexcept {
                        delete[] static_cast<uint16_t*>(p);
                    }));
            } else {
                size_t shape[2] = {static_cast<size_t>(num_values), 
                                  static_cast<size_t>(num_components)};
                result[attr_name.c_str()] = nb::ndarray<nb::numpy, uint16_t>(
                    data_array, 2, shape, nb::capsule(data_array, [](void *p) noexcept {
                        delete[] static_cast<uint16_t*>(p);
                    }));
            }
        } else if (data_type == draco::DT_UINT32 || data_type == draco::DT_INT32) {
            auto* data_array = new uint32_t[num_values * num_components];
            
            for (int i = 0; i < num_values; ++i) {
                draco::AttributeValueIndex val_index = attr->mapped_index(draco::PointIndex(i));
                attr->GetValue(val_index, data_array + i * num_components);
            }
            
            if (num_components == 1) {
                size_t shape[1] = {static_cast<size_t>(num_values)};
                result[attr_name.c_str()] = nb::ndarray<nb::numpy, uint32_t>(
                    data_array, 1, shape, nb::capsule(data_array, [](void *p) noexcept {
                        delete[] static_cast<uint32_t*>(p);
                    }));
            } else {
                size_t shape[2] = {static_cast<size_t>(num_values), 
                                  static_cast<size_t>(num_components)};
                result[attr_name.c_str()] = nb::ndarray<nb::numpy, uint32_t>(
                    data_array, 2, shape, nb::capsule(data_array, [](void *p) noexcept {
                        delete[] static_cast<uint32_t*>(p);
                    }));
            }
        }
    }
    
    return result;
}

NB_MODULE(dracox_ext, m) {
    m.doc() = "Minimal Draco mesh decompression for glTF KHR_draco_mesh_compression";
    
    m.def("decode_draco_buffer", &decode_draco_buffer,
          "compressed_data"_a, "attribute_map"_a,
          "Decode Draco-compressed mesh data and return numpy arrays.\n\n"
          "Args:\n"
          "    compressed_data: bytes containing Draco-compressed mesh\n"
          "    attribute_map: list of tuples (attribute_name, attribute_id)\n\n"
          "Returns:\n"
          "    dict with 'indices' and attribute arrays (e.g., 'POSITION', 'NORMAL')");
}
