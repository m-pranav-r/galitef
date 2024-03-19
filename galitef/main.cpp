#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifndef GLM_H

#define GLM_H
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtx/hash.hpp>

#endif

#ifndef PARSER_H
#define PARSER_H
#include "gltf/parser.h"
#endif

#include "app/app.h"

int main() {
	try {
		GLTFParser model;
		model.parse("./models/BoomBox.glb");
		GalitefApp app(800, 600, model.model);
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}