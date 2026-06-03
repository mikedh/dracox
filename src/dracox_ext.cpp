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

// Encode mesh data to Draco-compressed buffer preserving original vertex indices.
// Uses draco::Mesh directly (NOT TriangleSoupMeshBuilder) to avoid vertex
// deduplication that corrupts UV seams and normals at hard edges.
nb::dict encode_draco_buffer(
    nb::ndarray<nb::numpy, float, nb::shape<-1, 3>> vertices,
    nb::ndarray<nb::numpy, uint32_t, nb::shape<-1, 3>> faces,
    std::optional<nb::ndarray<nb::numpy, float, nb::shape<-1, 3>>> normals,
    std::optional<nb::ndarray<nb::numpy, float, nb::shape<-1, 2>>> texcoords,
    int compression_level = 7,
    int quantization_position = 16,
    int quantization_normal = 14,
    int quantization_tex_coord = 14) {

    const int num_vertices = static_cast<int>(vertices.shape(0));
    const int num_faces = static_cast<int>(faces.shape(0));

    // Build mesh directly — preserves original vertex indices (no deduplication).
    // This avoids TriangleSoupMeshBuilder which re-deduplicates vertices based on
    // quantized values, corrupting UV seams.
    auto mesh = std::unique_ptr<draco::Mesh>(new draco::Mesh());
    mesh->set_num_points(num_vertices);

    // Add position attribute (identity_mapping=false, we set mapping manually)
    draco::GeometryAttribute pos_ga;
    pos_ga.Init(draco::GeometryAttribute::POSITION, nullptr, 3, draco::DT_FLOAT32,
                false, sizeof(float) * 3, 0);
    const int pos_att_id = mesh->AddAttribute(pos_ga, false, num_vertices);

    // Add normal attribute if provided
    int norm_att_id = -1;
    if (normals.has_value()) {
        draco::GeometryAttribute norm_ga;
        norm_ga.Init(draco::GeometryAttribute::NORMAL, nullptr, 3, draco::DT_FLOAT32,
                     false, sizeof(float) * 3, 0);
        norm_att_id = mesh->AddAttribute(norm_ga, false, num_vertices);
    }

    // Add texcoord attribute if provided
    int tex_att_id = -1;
    if (texcoords.has_value()) {
        draco::GeometryAttribute tex_ga;
        tex_ga.Init(draco::GeometryAttribute::TEX_COORD, nullptr, 2, draco::DT_FLOAT32,
                    false, sizeof(float) * 2, 0);
        tex_att_id = mesh->AddAttribute(tex_ga, false, num_vertices);
    }

    // Set attribute values and point mapping
    const float* vert_data = vertices.data();
    const float* norm_data = normals.has_value() ? normals->data() : nullptr;
    const float* tex_data = texcoords.has_value() ? texcoords->data() : nullptr;

    draco::PointAttribute* pos_att = mesh->attribute(pos_att_id);
    draco::PointAttribute* norm_att = norm_att_id >= 0 ? mesh->attribute(norm_att_id) : nullptr;
    draco::PointAttribute* tex_att = tex_att_id >= 0 ? mesh->attribute(tex_att_id) : nullptr;

    for (int i = 0; i < num_vertices; ++i) {
        const draco::AttributeValueIndex avi(i);
        const draco::PointIndex pi(i);

        pos_att->SetAttributeValue(avi, vert_data + i * 3);
        pos_att->SetPointMapEntry(pi, avi);

        if (norm_att) {
            norm_att->SetAttributeValue(avi, norm_data + i * 3);
            norm_att->SetPointMapEntry(pi, avi);
        }
        if (tex_att) {
            tex_att->SetAttributeValue(avi, tex_data + i * 2);
            tex_att->SetPointMapEntry(pi, avi);
        }
    }

    // Add faces using original indices
    const uint32_t* face_data = faces.data();
    for (int fi = 0; fi < num_faces; ++fi) {
        draco::Mesh::Face face;
        face[0] = draco::PointIndex(face_data[fi * 3 + 0]);
        face[1] = draco::PointIndex(face_data[fi * 3 + 1]);
        face[2] = draco::PointIndex(face_data[fi * 3 + 2]);
        mesh->AddFace(face);
    }

#ifdef DRACO_ATTRIBUTE_DEDUPLICATION_SUPPORTED
    // Deduplicate points that share identical values across ALL attributes.
    // Unlike DeduplicateAttributeValues(), this preserves UV seams and
    // normal splits — only truly redundant points are merged.
    mesh->DeduplicatePointIds();
#endif

    // Track attribute IDs for the extension
    nb::dict attributes;
    attributes["POSITION"] = static_cast<int>(pos_att->unique_id());
    if (norm_att) {
        attributes["NORMAL"] = static_cast<int>(norm_att->unique_id());
    }
    if (tex_att) {
        attributes["TEXCOORD_0"] = static_cast<int>(tex_att->unique_id());
    }

    // Create encoder and set options
    draco::Encoder encoder;
    encoder.SetSpeedOptions(10 - compression_level, 10 - compression_level);
    encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, quantization_position);
    encoder.SetAttributeQuantization(draco::GeometryAttribute::NORMAL, quantization_normal);
    encoder.SetAttributeQuantization(draco::GeometryAttribute::TEX_COORD, quantization_tex_coord);

    // Encode the mesh
    draco::EncoderBuffer buffer;
    auto status = encoder.EncodeMeshToBuffer(*mesh, &buffer);

    if (!status.ok()) {
        throw std::runtime_error("Failed to encode Draco mesh: " +
                                 status.error_msg_string());
    }

    // Create result dictionary
    nb::dict result;
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
          "quantization_position"_a = 16, "quantization_normal"_a = 14,
          "quantization_tex_coord"_a = 14,
          "Encode mesh data to Draco-compressed buffer.\n\n"
          "Args:\n"
          "    vertices: (N, 3) float32 array of vertex positions\n"
          "    faces: (M, 3) uint32 array of face indices\n"
          "    normals: optional (N, 3) float32 array of vertex normals\n"
          "    texcoords: optional (N, 2) float32 array of texture coordinates\n"
          "    compression_level: 0-10, higher = better compression (default 7)\n"
          "    quantization_position: quantization bits for positions (default 16)\n"
          "    quantization_normal: quantization bits for normals (default 14)\n"
          "    quantization_tex_coord: quantization bits for UVs (default 14)\n\n"
          "Returns:\n"
          "    dict with 'buffer' (compressed bytes) and 'attributes' (Draco IDs)");
}
