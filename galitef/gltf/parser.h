#ifndef GLM_H
#define GLM_H
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtx/hash.hpp>
#endif

#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/glm_element_traits.hpp>

#ifndef STB_H
#define STB_H
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb_image.h>
#endif

#include <iostream>

enum TextureType {
	BASE = 0,
	METALLIC_ROUGHNESS = 1,
	NORMAL = 2,
	EMISSIVE = 3,
	OCCLUSION = 4
};

std::string getMimeType(fastgltf::MimeType mimeType) {
	switch (mimeType)
	{
	case fastgltf::MimeType::None:
		return std::string("None");
		break;
	case fastgltf::MimeType::JPEG:
		return std::string("JPEG");
		break;
	case fastgltf::MimeType::PNG:
		return std::string("PNG");
		break;
	case fastgltf::MimeType::KTX2:
		return std::string("KTX2");
		break;
	case fastgltf::MimeType::DDS:
		return std::string("DDS");
		break;
	default:
		break;
	}
}

class Texture {
public:
	TextureType type;
	stbi_uc* pixels;
	int texWidth, texHeight, texChannels;
	std::array<float, 4> factor = { 0, 0, 0, 0 };
	uint64_t textureIndex, texCoordIndex;

	bool load(fastgltf::Asset& asset, uint64_t textureIndex, TextureType texType, std::array<float, 4> factor, uint64_t texCoordIndex) {
		std::cout << "trying to load texture of type " << texType;
		if (!asset.textures[textureIndex].imageIndex.has_value()) {
			std::cout << "...not found, marking as such.\n";
			return false;
		}
		fastgltf::Image& image = asset.images[asset.textures[textureIndex].imageIndex.value()];
		fastgltf::sources::BufferView bufferViewView = std::get<fastgltf::sources::BufferView>(image.data);
		fastgltf::BufferView& bufferView = asset.bufferViews[bufferViewView.bufferViewIndex];
		//bufferView.

		fastgltf::Buffer& buffer = asset.buffers[bufferView.bufferIndex];
		auto& byteView = std::get<fastgltf::sources::ByteView>(buffer.data);
		int requiredChannels = 4;
		pixels = stbi_load_from_memory((stbi_uc*)byteView.bytes.data() + bufferView.byteOffset, bufferView.byteLength, &texWidth, &texHeight, &texChannels, requiredChannels);
		if (!pixels) {
			//stbi__jpeg_load((stbi_uc*)byteView.bytes.data() + bufferView.byteOffset, bufferView.byteLength, &texWidth, &texHeight, &texChannels, requiredChannels);
			//stbi_memory
			std::cout << "... error! detected channels: " << texChannels << ", requested channels: " << requiredChannels << "\n"
				<< "failure reason from stbi: " << stbi_failure_reason() << std::endl
				<< "extra info:\n\tname:" << image.name << std::endl
				<< "\tmime type: "<< getMimeType(bufferViewView.mimeType)<<std::endl;
			//routine to get handle to jpeg data in memory
			throw std::runtime_error("failed to load image!!");
		}

		this->factor = factor;
		this->type = texType;
		this->texCoordIndex = texCoordIndex;
		std::cout << "... done!\n";
		return true;
	}
};

class Material {
public:
	Texture baseColorTex, metalRoughTex, normalTex, emissiveTex, occlusionTex;
	bool isEmissiveTexPresent = false, isOcclusionTexPresent = false;
};

class Model {
public:
	std::vector<std::uint32_t> indices;
	std::vector<glm::vec3> pos;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec4> tangents;
	std::vector<glm::vec2> texCoords;
	Material mat;
	bool hasTangents = false;
	fastgltf::TRS transformData;
};

class GLTFParser {
public:

	fastgltf::PrimitiveType renderingMode;
	Model model;

	void validateGLTF(fastgltf::Asset& asset) {
		fastgltf::Error validResult = fastgltf::validate(asset);

		switch (validResult)
		{
		case fastgltf::Error::None:
			break;
		case fastgltf::Error::InvalidPath:
			throw std::runtime_error("invalid path!");
			break;
		case fastgltf::Error::MissingExtensions:
			throw std::runtime_error("missing extension!");
			break;
		case fastgltf::Error::UnknownRequiredExtension:
			throw std::runtime_error("unknown required extension!");
			break;
		case fastgltf::Error::InvalidJson:
			throw std::runtime_error("invalid json!");
			break;
		case fastgltf::Error::InvalidGltf:
			throw std::runtime_error("invalid gltf!");
			break;
		case fastgltf::Error::InvalidOrMissingAssetField:
			throw std::runtime_error("invalid or missing asset field!");
			break;
		case fastgltf::Error::InvalidGLB:
			throw std::runtime_error("invalid glb!");
			break;
		case fastgltf::Error::MissingField:
			throw std::runtime_error("missing field!");
			break;
		case fastgltf::Error::MissingExternalBuffer:
			throw std::runtime_error("missing external buffer!");
			break;
		case fastgltf::Error::UnsupportedVersion:
			throw std::runtime_error("unsupported version!");
			break;
		case fastgltf::Error::InvalidURI:
			throw std::runtime_error("invalid uri!");
			break;
		case fastgltf::Error::InvalidFileData:
			throw std::runtime_error("invalid file data!");
			break;
		default:
			break;
		}
	}

	void parse(std::filesystem::path path) {
		fastgltf::Parser parser;

		fastgltf::GltfDataBuffer data;
		data.loadFromFile(path);

		auto assetRef = parser.loadGltfBinary(&data, path.parent_path(), fastgltf::Options::None);

		if (auto error = assetRef.error(); error != fastgltf::Error::None) {
			throw std::runtime_error("failed to load asset!");
		}
		//if (isDebugEnv) {
		validateGLTF(assetRef.get());
		//}

		fastgltf::Asset& asset = assetRef.get();

		std::cout << "successfully loaded & parsed gltf file!" << std::endl;

		for (auto& node : asset.scenes[0].nodeIndices) {

			auto currNode = asset.nodes[node];
			auto meshIndex = currNode.meshIndex;
			auto cameraIndex = currNode.cameraIndex;
			auto skinIndex = currNode.skinIndex;
			auto lightIndex = currNode.lightIndex;

			std::cout << "NODE DATA:\n\n";

			if (meshIndex.has_value()) std::cout << "mesh present...\n";
			if (cameraIndex.has_value()) std::cout << "camera present...\n";
			if (skinIndex.has_value()) std::cout << "skin present...\n";
			if (lightIndex.has_value()) std::cout << "light present...\n";

			//compute trs matrix
			model.transformData = std::get<fastgltf::TRS>(currNode.transform);

			std::cout << "\nNODE DATA COMPLETE.\n";

			if (meshIndex.has_value()) {
				fastgltf::Mesh currMesh = asset.meshes[meshIndex.value()];

				for (auto& primitive : currMesh.primitives) {
					renderingMode = primitive.type;
					std::cout << "size: " << primitive.attributes.size() << "\n";
					for (auto attrib : primitive.attributes) {
						std::cout << "ACCESSOR DATA:\n" <<
							attrib.first << "	" << attrib.second << "\n";
						if (attrib.first == "NORMAL") {
							auto& accessor = asset.accessors[attrib.second];
							model.normals.resize(accessor.count);

							std::size_t idx = 0;
							fastgltf::iterateAccessor<glm::vec3>(asset, accessor, [&](glm::vec3 index) {
								model.normals[idx++] = index;
								});
						}
						else if (attrib.first == "TANGENT") {
							auto& accessor = asset.accessors[attrib.second];
							model.tangents.resize(accessor.count);

							std::size_t idx = 0;
							fastgltf::iterateAccessor<glm::vec4>(asset, accessor, [&](glm::vec4 index) {
								model.tangents[idx++] = index;
								});
							model.hasTangents = true;
							
						}
						else if (attrib.first == "POSITION") {
							auto& accessor = asset.accessors[attrib.second];
							model.pos.resize(accessor.count);

							std::size_t idx = 0;
							fastgltf::iterateAccessor<glm::vec3>(asset, accessor, [&](glm::vec3 index) {
								model.pos[idx++] = index;
								});
						}
						else if (attrib.first == "TEXCOORD_0") {
							auto& accessor = asset.accessors[attrib.second];
							model.texCoords.resize(accessor.count);

							std::size_t idx = 0;
							fastgltf::iterateAccessor<glm::vec2>(asset, accessor, [&](glm::vec2 index) {
								model.texCoords[idx++] = index;
								});
						}
					}
				}
				fastgltf::Primitive& currPrim = currMesh.primitives[0];
				if (currPrim.indicesAccessor.has_value()) {
					auto& accessor = asset.accessors[currPrim.indicesAccessor.value()];
					model.indices.resize(accessor.count);

					std::size_t idx = 0;
					fastgltf::iterateAccessor<std::uint32_t>(asset, accessor, [&](std::uint32_t index) {
						model.indices[idx++] = index;
						});
				}

				auto& currMaterial = asset.materials[currPrim.materialIndex.value()];

				model.mat.baseColorTex.load(
					asset,
					currMaterial.pbrData.baseColorTexture.value().textureIndex,
					TextureType::BASE,
					currMaterial.pbrData.baseColorFactor,
					currMaterial.pbrData.baseColorTexture.value().texCoordIndex
				);

				model.mat.metalRoughTex.load(
					asset,
					currMaterial.pbrData.metallicRoughnessTexture.value().textureIndex,
					TextureType::METALLIC_ROUGHNESS,
					std::array<float, 4>{
					currMaterial.pbrData.metallicFactor,
						currMaterial.pbrData.roughnessFactor,
						0,
						0
				},
					currMaterial.pbrData.metallicRoughnessTexture.value().texCoordIndex
				);

				model.mat.normalTex.load(
					asset,
					currMaterial.normalTexture.value().textureIndex,
					TextureType::NORMAL,
					std::array<float, 4>{currMaterial.normalTexture.value().scale},
					currMaterial.normalTexture.value().texCoordIndex
				);

				if (currMaterial.emissiveTexture.has_value()) {
					model.mat.isEmissiveTexPresent = model.mat.emissiveTex.load(
						asset,
						currMaterial.emissiveTexture.value().textureIndex,
						TextureType::EMISSIVE,
						std::array<float, 4>{
						currMaterial.emissiveFactor[0],
							currMaterial.emissiveFactor[1],
							currMaterial.emissiveFactor[2],
							1
					},
						currMaterial.emissiveTexture.value().texCoordIndex
					);

				}

				if (currMaterial.occlusionTexture.has_value()) {
					model.mat.isOcclusionTexPresent = model.mat.occlusionTex.load(
						asset,
						currMaterial.occlusionTexture.value().textureIndex,
						TextureType::OCCLUSION,
						std::array<float, 4>{currMaterial.occlusionTexture.value().strength},
						currMaterial.occlusionTexture.value().texCoordIndex
					);
				}

			}
		}
	}
};

