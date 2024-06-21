#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifndef PARSER_H
#define PARSER_H
#include "gltf/parser.h"
#endif

#include "app/app.h"

int main(int argc, char* argv[]) {
	for (int i = 0; i < argc; i++) {
		std::cout << argv[i] << " \n";
	}
	uint32_t width = 100, height = 100;
	sscanf_s(argv[1], "%d", &width);
	sscanf_s(argv[2], "%d", &height);
	try {
		GLTFParser model;
		model.parse(std::filesystem::path(argv[3]));
		GalitefApp app(width, height, model.model);
		app.run(argv[4]);
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}