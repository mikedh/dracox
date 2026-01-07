// Minimal nanobind extension for Draco mesh compression/decompression
// Specifically for glTF KHR_draco_mesh_compression extension

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "draco/compression/decode.h"
#include "draco/compression/encode.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/encoder_buffer.h"
#include "draco/mesh/mesh.h"
#include "draco/mesh/triangle_soup_mesh_builder.h"
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

// Encode mesh data to Draco-compressed buffer using TriangleSoupMeshBuilder
nb::dict encode_draco_buffer(
    nb::ndarray<nb::numpy, float, nb::shape<-1, 3>> vertices,
    nb::ndarray<nb::numpy, uint32_t, nb::shape<-1, 3>> faces,
    std::optional<nb::ndarray<nb::numpy, float, nb::shape<-1, 3>>> normals,
    std::optional<nb::ndarray<nb::numpy, float, nb::shape<-1, 2>>> texcoords,
    int compression_level = 7) {

    const size_t num_vertices = vertices.shape(0);
    const size_t num_faces = faces.shape(0);

    // Use TriangleSoupMeshBuilder for simpler mesh construction
    draco::TriangleSoupMeshBuilder builder;
    builder.Start(num_faces);

    // Add position attribute
    const int pos_att_id = builder.AddAttribute(
        draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32);

    // Add normal attribute if provided
    int norm_att_id = -1;
    if (normals.has_value()) {
        norm_att_id = builder.AddAttribute(
            draco::GeometryAttribute::NORMAL, 3, draco::DT_FLOAT32);
    }

    // Add texcoord attribute if provided
    int tex_att_id = -1;
    if (texcoords.has_value()) {
        tex_att_id = builder.AddAttribute(
            draco::GeometryAttribute::TEX_COORD, 2, draco::DT_FLOAT32);
    }

    // Get data pointers
    const float* vert_data = vertices.data();
    const uint32_t* face_data = faces.data();
    const float* norm_data = normals.has_value() ? normals->data() : nullptr;
    const float* tex_data = texcoords.has_value() ? texcoords->data() : nullptr;

    // Add faces with their vertex data
    for (size_t fi = 0; fi < num_faces; ++fi) {
        uint32_t i0 = face_data[fi * 3 + 0];
        uint32_t i1 = face_data[fi * 3 + 1];
        uint32_t i2 = face_data[fi * 3 + 2];

        draco::FaceIndex face_index(fi);

        // Set position for each vertex of the face
        builder.SetAttributeValuesForFace(
            pos_att_id, face_index,
            vert_data + i0 * 3,
            vert_data + i1 * 3,
            vert_data + i2 * 3);

        // Set normals if provided
        if (norm_data != nullptr) {
            builder.SetAttributeValuesForFace(
                norm_att_id, face_index,
                norm_data + i0 * 3,
                norm_data + i1 * 3,
                norm_data + i2 * 3);
        }

        // Set texcoords if provided
        if (tex_data != nullptr) {
            builder.SetAttributeValuesForFace(
                tex_att_id, face_index,
                tex_data + i0 * 2,
                tex_data + i1 * 2,
                tex_data + i2 * 2);
        }
    }

    // Finalize mesh construction
    std::unique_ptr<draco::Mesh> mesh = builder.Finalize();
    if (mesh == nullptr) {
        throw std::runtime_error("Failed to build Draco mesh");
    }

    // Track attribute IDs for the extension
    nb::dict attributes;

    // Get unique IDs from the finalized mesh
    const draco::PointAttribute* pos_attr = mesh->GetNamedAttribute(
        draco::GeometryAttribute::POSITION);
    if (pos_attr != nullptr) {
        attributes["POSITION"] = static_cast<int>(pos_attr->unique_id());
    }

    if (normals.has_value()) {
        const draco::PointAttribute* norm_attr = mesh->GetNamedAttribute(
            draco::GeometryAttribute::NORMAL);
        if (norm_attr != nullptr) {
            attributes["NORMAL"] = static_cast<int>(norm_attr->unique_id());
        }
    }

    if (texcoords.has_value()) {
        const draco::PointAttribute* tex_attr = mesh->GetNamedAttribute(
            draco::GeometryAttribute::TEX_COORD);
        if (tex_attr != nullptr) {
            attributes["TEXCOORD_0"] = static_cast<int>(tex_attr->unique_id());
        }
    }

    // Create encoder and set options
    draco::Encoder encoder;
    encoder.SetSpeedOptions(10 - compression_level, 10 - compression_level);
    encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, 14);
    encoder.SetAttributeQuantization(draco::GeometryAttribute::NORMAL, 10);
    encoder.SetAttributeQuantization(draco::GeometryAttribute::TEX_COORD, 12);

    // Encode the mesh
    draco::EncoderBuffer buffer;
    auto status = encoder.EncodeMeshToBuffer(*mesh, &buffer);

    if (!status.ok()) {
        throw std::runtime_error("Failed to encode Draco mesh: " +
                                 status.error_msg_string());
    }

    // Create result dictionary
    nb::dict result;

    // Copy buffer data to Python bytes
    result["buffer"] = nb::bytes(buffer.data(), buffer.size());
    result["attributes"] = attributes;

    return result;
}

NB_MODULE(dracox_ext, m) {
    m.doc() = "Draco mesh compression/decompression for glTF KHR_draco_mesh_compression";

    m.def("decode_draco_buffer", &decode_draco_buffer,
          "compressed_data"_a, "attribute_map"_a,
          "Decode Draco-compressed mesh data and return numpy arrays.\n\n"
          "Args:\n"
          "    compressed_data: bytes containing Draco-compressed mesh\n"
          "    attribute_map: list of tuples (attribute_name, attribute_id)\n\n"
          "Returns:\n"
          "    dict with 'indices' and attribute arrays (e.g., 'POSITION', 'NORMAL')");

    m.def("encode_draco_buffer", &encode_draco_buffer,
          "vertices"_a, "faces"_a, "normals"_a = nb::none(),
          "texcoords"_a = nb::none(), "compression_level"_a = 7,
          "Encode mesh data to Draco-compressed buffer.\n\n"
          "Args:\n"
          "    vertices: (N, 3) float32 array of vertex positions\n"
          "    faces: (M, 3) uint32 array of face indices\n"
          "    normals: optional (N, 3) float32 array of vertex normals\n"
          "    texcoords: optional (N, 2) float32 array of texture coordinates\n"
          "    compression_level: 0-10, higher = better compression (default 7)\n\n"
          "Returns:\n"
          "    dict with 'buffer' (compressed bytes) and 'attributes' (Draco IDs)");
}
