#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>

#include <glm.hpp>
#include <fastgltf/glm_element_traits.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <filesystem>
#include <stdexcept>
#include <iostream>

#ifdef _DEBUG
	bool isDebugEnv = true;
#else
	bool isDebugEnv = false;
#endif

struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec3 texCoord;
};

enum TextureType {
	BASE,
	METALLIC_ROUGHNESS,
	NORMAL,
	EMISSIVE,
	OCCLUSION
};

class Texture {
public:
	TextureType type;
	stbi_uc* pixels;
	int texWidth, texHeight, texChannels;
	std::array<float, 4> *factor;
	uint64_t textureIndex, texCoordIndex;

	void load(fastgltf::Asset &asset, uint64_t textureIndex, TextureType texType, std::array<float, 4> factor, uint64_t texCoordIndex) {
		std::cout << "trying to load texture of type " << texType;
		fastgltf::Image& image = asset.images[asset.textures[textureIndex].imageIndex.value()];
		//std::cout << "data index: " << image.data.index() << "\n";
		fastgltf::sources::BufferView bufferViewView = std::get<fastgltf::sources::BufferView>(image.data);
		//std::cout << bufferView.bufferViewIndex << "\n";
		fastgltf::BufferView &bufferView = asset.bufferViews[bufferViewView.bufferViewIndex];
		//bufferView.

		fastgltf::Buffer& buffer = asset.buffers[bufferView.bufferIndex];
		auto& byteView = std::get<fastgltf::sources::ByteView>(buffer.data);
		int requiredChannels = 3;
		//if (texType == 3) requiredChannels = 3;
		pixels = stbi_load_from_memory((stbi_uc*)byteView.bytes.data() + bufferView.byteOffset, bufferView.byteLength, &texWidth, &texHeight, &texChannels, requiredChannels);
		if (stbi_failure_reason()) {
			std::cout << "... error!\n";
			std::cout << stbi_failure_reason() << std::endl;
			throw std::runtime_error("failed to load image!!");
		}
		/*
		if (pixels == nullptr) {
		}
		*/

		this->factor = &factor;
		this->type = texType;
		this->texCoordIndex = texCoordIndex;
		std::cout << "... done!\n";
	}
};

class Material {
public:
	Texture baseColor, metalRough, normal, emissive, occlusion;
	//std::array<float, 3> emissiveFactor;
	//float emissiveStrength;
};

class Model {
public:
	std::vector<std::uint32_t> indices;
	std::vector<glm::vec3> pos;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> texCoords;
	Material mat;
};

void validateGLTF(fastgltf::Asset &asset) {
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

void testMain() {
	std::filesystem::path path = "./models/BoomBox.glb";
	fastgltf::Parser parser;

	fastgltf::GltfDataBuffer data;
	data.loadFromFile(path);

	auto asset = parser.loadGltfBinary(&data, path.parent_path(), fastgltf::Options::None);

	if (auto error = asset.error(); error != fastgltf::Error::None) {
		throw std::runtime_error("failed to load asset!");
	}
	if (isDebugEnv) {
		validateGLTF(asset.get());
	}

	std::cout <<"successfully loaded & parsed gltf file!"<<std::endl;

	fastgltf::PrimitiveType renderingMode;
	Model model;
	Texture baseColorTex, metalRoughnessTex, normalTex, emissiveTex, occlusionTex;
	for (auto& node : asset->scenes[0].nodeIndices) {
		//process nodes
		//	try to get hold of all the relevant parts first
		auto currNode = asset->nodes[node];
		auto meshIndex = currNode.meshIndex;
		auto cameraIndex = currNode.cameraIndex;
		auto skinIndex = currNode.skinIndex;
		auto lightIndex = currNode.lightIndex;

		std::cout << "NODE DATA:\n\n";

		if (meshIndex.has_value()) std::cout << "mesh present...\n";
		if (cameraIndex.has_value()) std::cout << "camera present...\n";
		if (skinIndex.has_value()) std::cout << "skin present...\n";
		if (lightIndex.has_value()) std::cout << "light present...\n";

		std::cout << "\nNODE DATA COMPLETE.\n";

		if (meshIndex.has_value()) {
			fastgltf::Mesh currMesh = asset->meshes[meshIndex.value()];

			//output primitive attribute data
			for (auto& primitive : currMesh.primitives) {
				renderingMode = primitive.type;
				std::cout << "size: " << primitive.attributes.size() << "\n";
				for (auto attrib : primitive.attributes) {
					std::cout << "ACCESSOR DATA:\n" <<
						attrib.first << "	" << attrib.second << "\n";
					if (attrib.first == "NORMAL") {
						auto& accessor = asset->accessors[attrib.second];
						model.normals.resize(accessor.count);

						std::size_t idx = 0;
						fastgltf::iterateAccessor<glm::vec3>(asset.get(), accessor, [&](glm::vec3 index) {
							model.normals[idx++] = index;
							});
					}
					else if (attrib.first == "POSITION") {
						auto& accessor = asset->accessors[attrib.second];
						model.pos.resize(accessor.count);

						std::size_t idx = 0;
						fastgltf::iterateAccessor<glm::vec3>(asset.get(), accessor, [&](glm::vec3 index) {
							model.pos[idx++] = index;
							});
					}
					else if (attrib.first == "TEXCOORD_0") {
						auto& accessor = asset->accessors[attrib.second];
						model.texCoords.resize(accessor.count);

						std::size_t idx = 0;
						fastgltf::iterateAccessor<glm::vec2>(asset.get(), accessor, [&](glm::vec2 index) {
							model.texCoords[idx++] = index;
							});
					}
				}
			}
			fastgltf::Primitive& currPrim = currMesh.primitives[0];
			if (currPrim.indicesAccessor.has_value()) {
				auto& accessor = asset->accessors[currPrim.indicesAccessor.value()];
				model.indices.resize(accessor.count);

				std::size_t idx = 0;
				fastgltf::iterateAccessor<std::uint32_t>(asset.get(), accessor, [&](std::uint32_t index) {
					model.indices[idx++] = index;
					});
			}

			auto& currMaterial = asset->materials[currPrim.materialIndex.value()];
			//model.mat.baseColor.factor[0] = currMaterial.pbrData.baseColorFactor[0];
			//model.mat.baseColor.factor[1] = currMaterial.pbrData.baseColorFactor[1];
			//model.mat.baseColor.factor[2] = currMaterial.pbrData.baseColorFactor[2];
			//model.mat.baseColor.factor[3] = currMaterial.pbrData.baseColorFactor[3];
			//model.mat.baseColor.

			baseColorTex.load(
				asset.get(), 
				currMaterial.pbrData.baseColorTexture.value().textureIndex, 
				TextureType::BASE, 
				currMaterial.pbrData.baseColorFactor,
				currMaterial.pbrData.baseColorTexture.value().texCoordIndex
			);

			metalRoughnessTex.load(
				asset.get(),
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

			normalTex.load(
				asset.get(),
				currMaterial.normalTexture.value().textureIndex,
				TextureType::NORMAL,
				std::array<float,4>{currMaterial.normalTexture.value().scale},
				currMaterial.normalTexture.value().texCoordIndex
			);

			emissiveTex.load(
				asset.get(),
				currMaterial.emissiveTexture.value().textureIndex,
				TextureType::EMISSIVE,
				std::array<float,4>{
					currMaterial.emissiveFactor[0],
					currMaterial.emissiveFactor[1],
					currMaterial.emissiveFactor[2],
					1
				},
				currMaterial.emissiveTexture.value().texCoordIndex
			);
			occlusionTex.load(
				asset.get(),
				currMaterial.occlusionTexture.value().textureIndex,
				TextureType::OCCLUSION,
				std::array<float, 4>{currMaterial.occlusionTexture.value().strength},
				currMaterial.occlusionTexture.value().texCoordIndex
			);
		}
	}
	assert(1 == 1);
}

int main() {
	try {
		testMain();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}