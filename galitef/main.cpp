#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifndef PARSER_H
#define PARSER_H
#include "gltf/parser.h"
#endif

#include "app/app.h"

int main() {
	try {
		GLTFParser model;
		model.parse("./models/WaterBottle.glb");
		GalitefApp app(800, 600, model.model);
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}