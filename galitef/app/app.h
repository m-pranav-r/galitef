#pragma once
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
#include <glm/gtx/quaternion.hpp>
#endif

#ifndef PARSER_H
#define PARSER_H
#include "../gltf/parser.h"
#endif

#ifndef STB_H
#define STB_H
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#endif

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <set>
#include <optional>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <array>
#include <unordered_map>


#ifdef _DEBUG
bool isDebugEnv = true;
#else
bool isDebugEnv = false;
#endif

struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec4 tangent;
	glm::vec2 texCoord;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, normal);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, tangent);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec3 camPos;
};

struct CubemapBufferObject {
	glm::mat4 model;
	glm::mat4 view[6];
	glm::mat4 proj;
};

struct MaterialBufferObject {
	float roughnessFactor;
	float metallicFactor;
	glm::vec4 baseColorFactor;
};

struct QueueFamilyIndices {
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> transferFamily;
	std::optional<uint32_t> graphicsFamily;

	bool isComplete() {
		return presentFamily.has_value() && transferFamily.has_value() && graphicsFamily.has_value();
	}
};

std::vector<float> offscreenVertices = {
	// back face
	-1.0f, -1.0f, -1.0f,
	 1.0f,  1.0f, -1.0f,
	 1.0f, -1.0f, -1.0f,
	 1.0f,  1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f,  1.0f, -1.0f,
	// front face
	-1.0f, -1.0f,  1.0f,
	 1.0f, -1.0f,  1.0f,
	 1.0f,  1.0f,  1.0f,
	 1.0f,  1.0f,  1.0f,
	-1.0f,  1.0f,  1.0f,
	-1.0f, -1.0f,  1.0f,
	// left face
	-1.0f,  1.0f,  1.0f,
	-1.0f,  1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f,  1.0f,
	-1.0f,  1.0f,  1.0f,
	// right face
	 1.0f,  1.0f,  1.0f,
	 1.0f, -1.0f, -1.0f,
	 1.0f,  1.0f, -1.0f,
	 1.0f, -1.0f, -1.0f,
	 1.0f,  1.0f,  1.0f,
	 1.0f, -1.0f,  1.0f,
	 // bottom face
	 -1.0f, -1.0f, -1.0f,
	  1.0f, -1.0f, -1.0f,
	  1.0f, -1.0f,  1.0f,
	  1.0f, -1.0f,  1.0f,
	 -1.0f, -1.0f,  1.0f,
	 -1.0f, -1.0f, -1.0f,
	 // top face
	 -1.0f,  1.0f, -1.0f,
	  1.0f,  1.0f , 1.0f,
	  1.0f,  1.0f, -1.0f,
	  1.0f,  1.0f,  1.0f,
	 -1.0f,  1.0f, -1.0f,
	 -1.0f,  1.0f,  1.0f,
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

const std::vector<const char*> validationlayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
	VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
	VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME,
	VK_KHR_MULTIVIEW_EXTENSION_NAME,
#ifdef _DEBUG_NVIDIA
	VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME,
	VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME
#endif
};

const int MAX_FRAMES_IN_FLIGHT = 2;

static std::vector<char> readFile(const std::string& fileName) {
	std::ifstream file(fileName, std::ios::ate | std::ios::binary);


	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}
	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();

	return buffer;
}

class TemporaryCamera {
public:
	glm::vec3 velocity;
	glm::vec3 position;
	float pitch = 0.f;
	float yaw = 0.f;

	double prevFrameXPos = -1, prevFrameYPos = -1;
	
	glm::mat4 getViewMatrix() {
		glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.0f), position);
		glm::mat4 cameraRotation = getRotationMatrix();
		return glm::inverse(cameraTranslation * cameraRotation);
	}
	glm::mat4 getRotationMatrix() {
		glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3{ 1.f, 0.f, 0.f });
		glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3({ 0.f, -1.f, 0.f }));

		return glm::mat4_cast(yawRotation) * glm::mat4_cast(pitchRotation);
	}

	void processGLFWKeyboardEvent(GLFWwindow* window, int key, int scancode, int action, int mods) {
		if (action == GLFW_PRESS) {
			if (key == GLFW_KEY_W) velocity.z = -1;
			if (key == GLFW_KEY_A) velocity.x = -1;
			if (key == GLFW_KEY_S) velocity.z = 1;
			if (key == GLFW_KEY_D) velocity.x = 1;
			if (key == GLFW_KEY_E) velocity.y = 1;
			if (key == GLFW_KEY_Q) velocity.y = -1;
			if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, 1);
		}

		if (action == GLFW_RELEASE) {
			if (key == GLFW_KEY_W) velocity.z = 0;
			if (key == GLFW_KEY_A) velocity.x = 0;
			if (key == GLFW_KEY_S) velocity.z = 0;
			if (key == GLFW_KEY_D) velocity.x = 0;
			if (key == GLFW_KEY_E) velocity.y = 0;
			if (key == GLFW_KEY_Q) velocity.y = 0;
		}
	}

	void processGLFWMouseEvent(GLFWwindow* window, double x, double y) {
		if (prevFrameXPos == -1) {
			prevFrameXPos = x;
			prevFrameYPos = y;
		}
		yaw += (x - prevFrameXPos) / 2000.f;
		pitch -= (y - prevFrameYPos) / 2000.f;

		prevFrameXPos = x;
		prevFrameYPos = y;
	}

	void update() {
		glm::mat4 cameraRotation = getRotationMatrix();
		position += glm::vec3(cameraRotation * glm::vec4(velocity * 0.005f, 0.f));
	}
};

TemporaryCamera camera;

class GalitefApp {
public:
	GalitefApp(uint32_t width, uint32_t height, Model parsedModel) : WIDTH(width), HEIGHT(height), model(parsedModel) {};

	void run(const char* path) {
		hdriPath = path;
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	uint32_t WIDTH, HEIGHT;
	Model model;

	const char* hdriPath;

	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;
	VkSurfaceKHR surface;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkQueue transferQueue;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSetLayout descriptorSetLayout;
	std::vector<VkDescriptorSet> descriptorSets;
	VkPipeline graphicsPipeline;
	VkCommandPool commandPool;
	VkCommandPool transferPool;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;
	VkDescriptorPool descriptorPool;
	uint32_t mipLevels;

	//make five of these
	VkImage baseColorImage, metallicRoughnessImage, normalImage, occlusionImage, emissiveImage;
	VkImageView baseColorImageView, metallicRoughnessImageView, normalImageView, occlusionImageView, emissiveImageView;
	VkSampler baseColorSampler, metallicRoughnessSampler, normalSampler, occlusionSampler, emissiveSampler;
	VkDeviceMemory baseColorMemory, metallicRoughnessMemory, normalMemory, occlusionMemory, emissiveMemory;
	//VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

	VkImage colorImage;
	VkDeviceMemory colorImageMemory;
	VkImageView colorImageView;
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	VkImage hdriImage;
	VkDeviceMemory hdriImageMemory;
	VkImageView hdriImageView;
	VkSampler hdriSampler;

	VkFormat cubemapFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
	VkFormat brdfFormat = VK_FORMAT_R16G16_UNORM;

	VkImage cubemap;
	VkDeviceMemory cubemapMemory;
	VkImageView cubemapImageView;
	VkSampler cubemapSampler;

	VkImage diffuseCubemap;
	VkDeviceMemory diffuseCubemapMemory;
	VkImageView diffuseCubemapImageView;
	VkSampler diffuseCubemapSampler;

	VkImage prefilterCubemap;
	VkDeviceMemory prefilterCubemapMemory;
	VkImageView prefilterCubemapImageViewPerMip[5], prefilterCubemapImageView;
	VkSampler prefilterCubemapSampler;

	VkImage brdfLUT;
	VkDeviceMemory brdfLUTMemory;
	VkImageView brdfLUTImageView;
	VkSampler brdfLUTSampler;

	VkImage offscreenDepthImage;
	VkDeviceMemory offscreenDepthImageMemory;
	VkImageView offscreenDepthImageView;

	VkImage diffuseDepthImage;
	VkDeviceMemory diffuseDepthImageMemory;
	VkImageView diffuseDepthImageView;

	VkBuffer offscreenVertexBuffer;
	VkDeviceMemory offscreenVertexBufferMemory;

	VkPipelineLayout cubemapPipelineLayout, diffuseCubemapPipelineLayout;

	VkPipeline cubemapPipeline, diffuseCubemapPipeline;
	VkDescriptorPool cubemapDescPool;

	VkDescriptorPool cubemapCreateDescriptorPool;
	VkDescriptorSetLayout cubemapCreateDescriptorSetLayout;
	VkDescriptorSet cubemapCreateDescriptorSet;
	VkRenderPass cubemapCreateRenderPass;
	VkFramebuffer cubemapCreateFramebuffer;
	VkPipeline cubemapCreatePipeline;
	VkPipelineLayout cubemapCreatePipelineLayout;
	VkCommandBuffer cubemapCreateCommandBuffer;

	VkDescriptorPool diffuseDescriptorPool;
	VkDescriptorSetLayout diffuseDescriptorSetLayout;
	VkDescriptorSet diffuseDescriptorSet;
	VkRenderPass diffuseRenderPass;
	VkFramebuffer diffuseFramebuffer;

	VkDescriptorPool prefilterDescriptorPool;
	VkDescriptorSetLayout prefilterDescriptorSetLayout;
	VkDescriptorSet prefilterDescriptorSet;
	VkPipelineLayout prefilterPipelineLayout;
	VkRenderPass prefilterRenderPass;
	VkFramebuffer prefilterFramebufferPerMip[5];
	VkPipeline prefilterPipeline;
	VkFence prefilterMipRenderedFence;

	
	VkDescriptorSetLayout cubemapDescriptorSetLayout;

	VkDescriptorPool brdfDescPool;
	VkDescriptorSetLayout brdfDescriptorSetLayout;
	VkBuffer brdfVertexBuffer;
	VkDeviceMemory brdfVertexMemory;
	VkPipelineLayout brdfPipelineLayout;
	VkPipeline brdfPipeline;

	std::vector<VkBuffer> uniformBuffers, materialBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory, materialBuffersMemory;
	std::vector<void*> uniformBuffersMapped, materialBuffersMapped;

	VkBuffer offscreenUniformBuffer;
	VkDeviceMemory offscreenUniformBufferMemory;
	void* offscreenUniformBufferMapped;

	VkDescriptorSet cubemapDescriptorSet, diffuseCubemapDescriptorSet;

	glm::mat4 TRS;

	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore >imageAvailableSemaphores;
	std::vector<VkSemaphore >renderFinishedSemaphores;
	std::vector<VkFence >inFlightFences;
	uint32_t currentFrame = 0;

	bool framebufferResized = false;

	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
		glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {camera.processGLFWKeyboardEvent(window, key, scancode, action, mods); });
		glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {camera.processGLFWMouseEvent(window, x, y); });
		camera.velocity = glm::vec3(0.f);
		camera.position = glm::vec3(2.f);
		camera.pitch = 0;
		camera.yaw = 0;
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<GalitefApp*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();

		createSwapChain();
		createImageViews();

		createDescriptorSetLayout();
		createGraphicsPipeline();
		createCommandPool();
		createColorResources();
		createDepthResources();

		createTextureImages();
		createTextureImageViews();
		createTextureSamplers();

			createHDRIResources();
			createCubemap();
			performOffscreenCubemapRender();
			handleCubemapTransitions();
			generateCubemapMipMaps();
			createDiffuseMap();
			performOffscreenDiffuseMapRender();
			createPrefilterMap();
			performPrefilterMapRender();
			createBRDFLUT();
			performBRDFLUTRender();

		loadModel();
		createVertexBuffer();
		createIndexBuffer();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
		createCubemapRenderStuff();
		std::cout << "done!\n\n\n";
	}

	void createInstance() {
		if (isDebugEnv && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers enabled, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_API_VERSION_1_3;
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_API_VERSION_1_3;
		appInfo.apiVersion = VK_API_VERSION_1_3;

		appInfo.pNext = nullptr;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();

		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (isDebugEnv) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationlayers.size());
			createInfo.ppEnabledLayerNames = validationlayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else {
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : extensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormatHDR(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_R16G16B16A16_SFLOAT && availableFormat.colorSpace == VK_COLOR_SPACE_HDR10_HLG_EXT) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}

			if ((queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) && !(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
				indices.transferFamily = i;
			}

			VkBool32 presentSupport = VK_FALSE;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (presentSupport == VK_TRUE) {
				indices.presentFamily = i;
			}
			i++;
		}
		return indices;
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

		return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
	}

	void pickPhysicalDevice() {

		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPU with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data());

		for (const auto& device : physicalDevices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				//msaaSamples = getMaxUsableSampleCount();
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
			throw std::runtime_error("failed to find a suitable GPU!");
	}

	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageExtent = extent;
		createInfo.presentMode = presentMode;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;// surfaceFormat.colorSpace;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 1;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;

			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}
		}
	}

	void createDescriptorSetLayout() {

		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding materialLayoutBinding{};
		materialLayoutBinding.binding = 1;
		materialLayoutBinding.descriptorCount = 1;
		materialLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		materialLayoutBinding.pImmutableSamplers = nullptr;
		materialLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding baseColorLayoutBinding{};
		baseColorLayoutBinding.binding = 2;
		baseColorLayoutBinding.descriptorCount = 1;
		baseColorLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		baseColorLayoutBinding.pImmutableSamplers = nullptr;
		baseColorLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding metallicRoughnessLayoutBinding{};
		metallicRoughnessLayoutBinding.binding = 3;
		metallicRoughnessLayoutBinding.descriptorCount = 1;
		metallicRoughnessLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		metallicRoughnessLayoutBinding.pImmutableSamplers = nullptr;
		metallicRoughnessLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding normalLayoutBinding{};
		normalLayoutBinding.binding = 4;
		normalLayoutBinding.descriptorCount = 1;
		normalLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		normalLayoutBinding.pImmutableSamplers = nullptr;
		normalLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding occlusionLayoutBinding{};
		occlusionLayoutBinding.binding = 5;
		occlusionLayoutBinding.descriptorCount = 1;
		occlusionLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		occlusionLayoutBinding.pImmutableSamplers = nullptr;
		occlusionLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding emissiveLayoutbinding{};
		emissiveLayoutbinding.binding = 6;
		emissiveLayoutbinding.descriptorCount = 1;
		emissiveLayoutbinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		emissiveLayoutbinding.pImmutableSamplers = nullptr;
		emissiveLayoutbinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding cubemapLayoutbinding{};
		cubemapLayoutbinding.binding = 7;
		cubemapLayoutbinding.descriptorCount = 1;
		cubemapLayoutbinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		cubemapLayoutbinding.pImmutableSamplers = nullptr;
		cubemapLayoutbinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding diffuseCubemapLayoutbinding{};
		diffuseCubemapLayoutbinding.binding = 8;
		diffuseCubemapLayoutbinding.descriptorCount = 1;
		diffuseCubemapLayoutbinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		diffuseCubemapLayoutbinding.pImmutableSamplers = nullptr;
		diffuseCubemapLayoutbinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding prefilterCubemapLayoutbinding{};
		prefilterCubemapLayoutbinding.binding = 9;
		prefilterCubemapLayoutbinding.descriptorCount = 1;
		prefilterCubemapLayoutbinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		prefilterCubemapLayoutbinding.pImmutableSamplers = nullptr;
		prefilterCubemapLayoutbinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding brdfLayoutbinding{};
		brdfLayoutbinding.binding = 10;
		brdfLayoutbinding.descriptorCount = 1;
		brdfLayoutbinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		brdfLayoutbinding.pImmutableSamplers = nullptr;
		brdfLayoutbinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


		std::vector<VkDescriptorSetLayoutBinding> bindings = { 
			uboLayoutBinding, 
			materialLayoutBinding,
			baseColorLayoutBinding,
			metallicRoughnessLayoutBinding,
			normalLayoutBinding,
			cubemapLayoutbinding,
			diffuseCubemapLayoutbinding,
			prefilterCubemapLayoutbinding,
			brdfLayoutbinding
		};

		if (model.mat.isEmissiveTexPresent) bindings.push_back(occlusionLayoutBinding);
		if (model.mat.isOcclusionTexPresent) bindings.push_back(emissiveLayoutbinding);

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}

	void createGraphicsPipeline() {
		std::vector<char> vertShaderCode, fragShaderCode;
		if (model.hasTangents) {
			vertShaderCode = readFile("shader/pbr/vert.spv");
			fragShaderCode = readFile("shader/pbr/frag.spv");
		}
		else {
			vertShaderCode = readFile("shader/pbr/vert-no-tangents.spv");
			fragShaderCode = readFile("shader/pbr/frag-no-tangents.spv");
		}

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapChainImageFormat,
			.depthAttachmentFormat = findDepthFormat()
		};

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = NULL;
		pipelineInfo.pNext = &pipelineRenderingCreateInfo;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
	}

	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.presentFamily.value(), indices.graphicsFamily.value(), indices.transferFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.pNext = nullptr;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceMultiviewFeatures multiviewFeatures = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
			.multiview = VK_TRUE,
		};

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.sampleRateShading = VK_TRUE;

		VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeatures = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
			.pNext = &multiviewFeatures,
			.dynamicRendering = VK_TRUE,
		};

#ifdef _DEBUG_NVIDIA
		VkDeviceDiagnosticsConfigCreateInfoNV aftermathAttachInfo{};
		aftermathAttachInfo.sType = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV;
		aftermathAttachInfo.flags =
			VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV |
			VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
			VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV |
			VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV;

		dynamicRenderingFeatures.pNext = &aftermathAttachInfo;
#endif

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();;
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.pNext = &dynamicRenderingFeatures;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (isDebugEnv) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationlayers.size());
			createInfo.ppEnabledLayerNames = validationlayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
		vkGetDeviceQueue(device, indices.transferFamily.value(), 0, &transferQueue);
	}

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	/*
	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 3> attachments = {
				colorImageView,
				depthImageView,
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}
	*/

	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}

		VkCommandPoolCreateInfo transferPoolInfo{};
		transferPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		transferPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		transferPoolInfo.queueFamilyIndex = queueFamilyIndices.transferFamily.value();

		if (vkCreateCommandPool(device, &transferPoolInfo, nullptr, &transferPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create transfer command pool!");
		}
	}

	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}
		throw std::runtime_error("failed to find supported format!");
	}

	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	VkFormat findDepthFormat() {
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	VkSampleCountFlagBits getMaxUsableSampleCount() {
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

		VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;

		if (counts & VK_SAMPLE_COUNT_64_BIT) return VK_SAMPLE_COUNT_64_BIT;
		if (counts & VK_SAMPLE_COUNT_32_BIT) return VK_SAMPLE_COUNT_32_BIT;
		if (counts & VK_SAMPLE_COUNT_16_BIT) return VK_SAMPLE_COUNT_16_BIT;
		if (counts & VK_SAMPLE_COUNT_8_BIT) return VK_SAMPLE_COUNT_8_BIT;
		if (counts & VK_SAMPLE_COUNT_4_BIT) return VK_SAMPLE_COUNT_4_BIT;
		if (counts & VK_SAMPLE_COUNT_2_BIT) return VK_SAMPLE_COUNT_2_BIT;

		return VK_SAMPLE_COUNT_1_BIT;
	}

	void createDepthResources() {
		VkFormat depthFormat = findDepthFormat();
		createImage(
			swapChainExtent.width,
			swapChainExtent.height,
			1,
			VK_SAMPLE_COUNT_1_BIT,
			depthFormat,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			depthImage,
			depthImageMemory
		);
		depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
		transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);

	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if (typeFilter & (1 << i) && ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		QueueFamilyIndices queueFamilies = findQueueFamilies(physicalDevice);
		bufferInfo.queueFamilyIndexCount = 2;
		uint32_t queueFamilyIndices[] = { queueFamilies.graphicsFamily.value(), queueFamilies.transferFamily.value() };
		bufferInfo.pQueueFamilyIndices = queueFamilyIndices;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate vertex buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling,
		VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = static_cast<uint32_t>(width);
		imageInfo.extent.height = static_cast<uint32_t>(height);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples = numSamples;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	void createTextureImages() {
		std::vector<Texture> textures = {
			model.mat.baseColorTex,
			model.mat.normalTex,
			model.mat.metalRoughTex,
		};

		if (model.mat.isEmissiveTexPresent) textures.push_back(model.mat.emissiveTex);
		if (model.mat.isOcclusionTexPresent) textures.push_back(model.mat.occlusionTex);

		for (auto& texture : textures) {
			mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texture.texWidth, texture.texHeight)))) + 1;

			VkDeviceSize imageSize = texture.texWidth * texture.texHeight * 4;

			if (!texture.pixels) {
				throw std::runtime_error("failed to load texture image!");
			}

			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;

			createBuffer(
				imageSize,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				stagingBuffer,
				stagingBufferMemory
			);

			void* data;
			if (vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data) != VK_SUCCESS) {
				throw std::runtime_error("failed to map texture memory!");
			}
			memcpy(data, texture.pixels, static_cast<size_t>(imageSize));
			vkUnmapMemory(device, stagingBufferMemory);

			stbi_image_free(texture.pixels);

			VkImage* textureImage = nullptr;// = &baseColorImage;
			VkDeviceMemory* textureImageMemory = nullptr;// = &baseColorMemory;
			VkFormat textureImageFormat;

			switch (texture.type) {
				case BASE:
				{
					textureImage = &baseColorImage;
					textureImageMemory = &baseColorMemory;
					textureImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
					break;
				}
				case METALLIC_ROUGHNESS:{
					textureImage = &metallicRoughnessImage;
					textureImageMemory = &metallicRoughnessMemory;
					textureImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
					break;
				}
				case NORMAL: {
					textureImage = &normalImage;
					textureImageMemory = &normalMemory;
					textureImageFormat = VK_FORMAT_R8G8B8A8_UNORM;
					break;
				}
				case OCCLUSION:
				{
					textureImage = &occlusionImage;
					textureImageMemory = &occlusionMemory;
					textureImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
					break;
				}
				case EMISSIVE: {
					textureImage = &emissiveImage;
					textureImageMemory = &emissiveMemory;
					textureImageFormat = VK_FORMAT_R8G8B8A8_SRGB;
					break;
				}
			}

			createImage(
				texture.texWidth,
				texture.texHeight,
				mipLevels,
				VK_SAMPLE_COUNT_1_BIT,
				textureImageFormat,
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				*textureImage,
				*textureImageMemory
			);

			transitionImageLayout(*textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
			copyBufferToImage(stagingBuffer, *textureImage, static_cast<uint32_t>(texture.texWidth), static_cast<uint32_t>(texture.texHeight));

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vkFreeMemory(device, stagingBufferMemory, nullptr);

			generateMipmaps(*textureImage, VK_FORMAT_R8G8B8A8_SRGB, texture.texWidth, texture.texHeight, mipLevels);
		}
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		VkPipelineStageFlags srcStage, dstStage;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (hasStencilComponent(format)) {
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}
		else {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else {
			throw std::runtime_error("unsupported layout transition!");
		}

		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.image = image;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		vkCmdPipelineBarrier(
			commandBuffer,
			srcStage, dstStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);
	}

	void transitionImageLayout(VkCommandBuffer commandBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
		VkImageMemoryBarrier barrier{};
		VkPipelineStageFlags srcStage, dstStage;

		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (hasStencilComponent(format)) {
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}
		else {
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else {
			throw std::runtime_error("unsupported layout transition!");
		}

		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.image = image;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		vkCmdPipelineBarrier(
			commandBuffer,
			srcStage, dstStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);
	}

	void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
		VkFormatProperties formatproperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatproperties);

		if (!(formatproperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; i++) {
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			vkCmdBlitImage(
				commandBuffer,
				image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				VK_FILTER_LINEAR
			);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);
	}

	void createColorResources() {
		VkFormat colorFormat = swapChainImageFormat;

		createImage(
			swapChainExtent.width,
			swapChainExtent.height,
			1,
			VK_SAMPLE_COUNT_1_BIT,
			colorFormat,
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			colorImage,
			colorImageMemory
		);

		colorImageView = createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
	}

	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = mipLevels;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;

		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image view!");
		}

		return imageView;
	}

	void createTextureImageViews() {
		baseColorImageView = createImageView(baseColorImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		metallicRoughnessImageView = createImageView(metallicRoughnessImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		normalImageView = createImageView(normalImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		if (model.mat.isOcclusionTexPresent) occlusionImageView = createImageView(occlusionImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		if (model.mat.isEmissiveTexPresent) emissiveImageView = createImageView(emissiveImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);

	}

	void createTextureSamplers() {
		std::vector<Texture> textures = {
			model.mat.baseColorTex,
			model.mat.normalTex,
			model.mat.metalRoughTex,
		};

		if (model.mat.isEmissiveTexPresent) textures.push_back(model.mat.emissiveTex);
		if (model.mat.isOcclusionTexPresent) textures.push_back(model.mat.occlusionTex);

		for (auto& texture : textures) {
			VkSampler* textureSampler = nullptr;// &baseColorSampler;

			switch (texture.type) {
			case BASE:
			{
				textureSampler = &baseColorSampler;
				break;
			}
			case METALLIC_ROUGHNESS: {
				textureSampler = &metallicRoughnessSampler;
				break;
			}
			case NORMAL: {
				textureSampler = &normalSampler;
				break;
			}
			case OCCLUSION:
			{
				textureSampler = &occlusionSampler;
				break;
			}
			case EMISSIVE: {
				textureSampler = &emissiveSampler;
				break;
			}
			}

			VkSamplerCreateInfo samplerInfo{};
			samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerInfo.magFilter = VK_FILTER_LINEAR;
			samplerInfo.minFilter = VK_FILTER_LINEAR;
			samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.anisotropyEnable = VK_TRUE;

			VkPhysicalDeviceProperties properties{};
			vkGetPhysicalDeviceProperties(physicalDevice, &properties);

			samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
			samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
			samplerInfo.unnormalizedCoordinates = VK_FALSE;
			samplerInfo.compareEnable = VK_FALSE;
			samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
			samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			samplerInfo.minLod = 0.0f;
			samplerInfo.maxLod = static_cast<float>(mipLevels);
			samplerInfo.mipLodBias = 0.0f;

			if (vkCreateSampler(device, &samplerInfo, nullptr, textureSampler) != VK_SUCCESS) {
				throw std::runtime_error("failed to create texture sampler!");
			}
		}
	}

	void createHDRIResources() {
		stbi_set_flip_vertically_on_load(true);
		int width, height, noComponents;
		float* pixels = stbi_loadf(hdriPath, &width, &height, &noComponents, 3);
		if (!pixels) {
			throw std::runtime_error("failed to load hdri!");
		}

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		VkDeviceSize imageSize = width * height * noComponents * sizeof(float);

		createBuffer(
			imageSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer,
			stagingBufferMemory
		);

		void* data;
		if (vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data) != VK_SUCCESS) {
			throw std::runtime_error("failed to map hdri texture memory!");
		}
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		stbi_image_free(pixels);

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = static_cast<uint32_t>(width);
		imageInfo.extent.height = static_cast<uint32_t>(height);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = VK_FORMAT_R32G32B32_SFLOAT;
		imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

		if (vkCreateImage(device, &imageInfo, nullptr, &hdriImage) != VK_SUCCESS) {
			throw std::runtime_error("failed to create hdri image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, hdriImage, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &hdriImageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate hdri image memory!");
		}

		vkBindImageMemory(device, hdriImage, hdriImageMemory, 0);

		transitionImageLayout(
			hdriImage,
			VK_FORMAT_R32G32B32_SFLOAT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1
		);

		copyBufferToImage(
			stagingBuffer,
			hdriImage,
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		);

		transitionImageLayout(
			hdriImage,
			VK_FORMAT_R32G32B32_SFLOAT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			1
		);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		VkSamplerCreateInfo hdriSamplerInfo{};
		hdriSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		hdriSamplerInfo.magFilter = VK_FILTER_LINEAR;
		hdriSamplerInfo.minFilter = VK_FILTER_LINEAR;
		hdriSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		hdriSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		hdriSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		hdriSamplerInfo.anisotropyEnable = VK_TRUE;
		hdriSamplerInfo.maxAnisotropy = 1.0f;
		hdriSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		hdriSamplerInfo.unnormalizedCoordinates = VK_FALSE;
		hdriSamplerInfo.compareEnable = VK_FALSE;
		hdriSamplerInfo.compareOp = VK_COMPARE_OP_NEVER;
		hdriSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		hdriSamplerInfo.minLod = 0.0f;
		hdriSamplerInfo.maxLod = 1.0f;
		hdriSamplerInfo.mipLodBias = 0.0f;

		if (vkCreateSampler(device, &hdriSamplerInfo, nullptr, &hdriSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create hdri sampler!");
		}

		VkImageViewCreateInfo hdriViewInfo{};
		hdriViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		hdriViewInfo.image = hdriImage;
		hdriViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		hdriViewInfo.format = VK_FORMAT_R32G32B32_SFLOAT;
		hdriViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		hdriViewInfo.subresourceRange.baseMipLevel = 0;
		hdriViewInfo.subresourceRange.levelCount = 1;
		hdriViewInfo.subresourceRange.baseArrayLayer = 0;
		hdriViewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(device, &hdriViewInfo, nullptr, &hdriImageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create hdri view!");
		}
	}

	void createCubemap() {
		VkImageCreateInfo cubemapInfo{};
		cubemapInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		cubemapInfo.imageType = VK_IMAGE_TYPE_2D;
		cubemapInfo.extent.width = 1024;
		cubemapInfo.extent.height = 1024;
		cubemapInfo.extent.depth = 1;
		cubemapInfo.mipLevels = 5;
		cubemapInfo.arrayLayers = 6;
		cubemapInfo.format = cubemapFormat;
		cubemapInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		cubemapInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		cubemapInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		cubemapInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		cubemapInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		cubemapInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

		if (vkCreateImage(device, &cubemapInfo, nullptr, &cubemap) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, cubemap, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &cubemapMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate cubemap image memory!");
		}

		vkBindImageMemory(device, cubemap, cubemapMemory, 0);

		VkImageViewCreateInfo cubemapImageViewInfo{};
		cubemapImageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		cubemapImageViewInfo.image = cubemap;
		cubemapImageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		cubemapImageViewInfo.format = cubemapFormat;
		cubemapImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		cubemapImageViewInfo.subresourceRange.baseMipLevel = 0;
		cubemapImageViewInfo.subresourceRange.levelCount = 1;
		cubemapImageViewInfo.subresourceRange.baseArrayLayer = 0;
		cubemapImageViewInfo.subresourceRange.layerCount = 6;

		if (vkCreateImageView(device, &cubemapImageViewInfo, nullptr, &cubemapImageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap image view!");
		}

		VkSamplerCreateInfo cubemapSamplerInfo{};
		cubemapSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		cubemapSamplerInfo.magFilter = VK_FILTER_LINEAR;
		cubemapSamplerInfo.minFilter = VK_FILTER_LINEAR;
		cubemapSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		cubemapSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		cubemapSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		cubemapSamplerInfo.anisotropyEnable = VK_TRUE;
		cubemapSamplerInfo.maxAnisotropy = 1.0f;
		cubemapSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		cubemapSamplerInfo.unnormalizedCoordinates = VK_FALSE;
		cubemapSamplerInfo.compareEnable = VK_FALSE;
		cubemapSamplerInfo.compareOp = VK_COMPARE_OP_NEVER;
		cubemapSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		cubemapSamplerInfo.minLod = 0.0f;
		cubemapSamplerInfo.maxLod = 5.0f;
		cubemapSamplerInfo.mipLodBias = 0.0f;
		cubemapSamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

		if (vkCreateSampler(device, &cubemapSamplerInfo, nullptr, &cubemapSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap sampler!");
		}

		//depth attachment
		VkImageCreateInfo depthImageInfo{};
		depthImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		depthImageInfo.imageType = VK_IMAGE_TYPE_2D;
		depthImageInfo.extent.width = 1024;
		depthImageInfo.extent.height = 1024;
		depthImageInfo.extent.depth = 1;
		depthImageInfo.mipLevels = 1;
		depthImageInfo.arrayLayers = 6;
		depthImageInfo.format = findDepthFormat();
		depthImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		depthImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		depthImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		depthImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		depthImageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

		if (vkCreateImage(device, &depthImageInfo, nullptr, &offscreenDepthImage) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap depth image!");
		}

		VkMemoryRequirements depthMemRequirements;
		vkGetImageMemoryRequirements(device, offscreenDepthImage, &depthMemRequirements);

		VkMemoryAllocateInfo depthAllocInfo{};
		depthAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		depthAllocInfo.allocationSize = depthMemRequirements.size;
		depthAllocInfo.memoryTypeIndex = findMemoryType(depthMemRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &depthAllocInfo, nullptr, &offscreenDepthImageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, offscreenDepthImage, offscreenDepthImageMemory, 0);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = offscreenDepthImage;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		viewInfo.format = findDepthFormat();
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 6;

		if (vkCreateImageView(device, &viewInfo, nullptr, &offscreenDepthImageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image view!");
		}
	}

	void performOffscreenCubemapRender() {
		//create uniform buffers
		//																																				dont forget to HANDLE DELETION LATER
		VkBufferCreateInfo offscreenUniformBufferInfo;
		VkDeviceSize uboBufferSize = sizeof(CubemapBufferObject);

		createBuffer(
			uboBufferSize,
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			offscreenUniformBuffer,
			offscreenUniformBufferMemory
		);
		vkMapMemory(device, offscreenUniformBufferMemory, 0, uboBufferSize, 0, &offscreenUniformBufferMapped);

		//setup descriptors
		//pool
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = 1;
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = 1;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 2;
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = 1;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &cubemapCreateDescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen descriptor pool!");
		}

		//layout
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding cubemapLayoutBinding{};
		cubemapLayoutBinding.binding = 1;
		cubemapLayoutBinding.descriptorCount = 1;
		cubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		cubemapLayoutBinding.pImmutableSamplers = nullptr;
		cubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
			uboLayoutBinding,
			cubemapLayoutBinding
		};

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 2;
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &cubemapCreateDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen descriptor set layout!");
		}

		//sets
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = cubemapCreateDescriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &cubemapCreateDescriptorSetLayout;

		if (vkAllocateDescriptorSets(device, &allocInfo, &cubemapCreateDescriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate offscreen descriptor set!");
		}

		VkDescriptorBufferInfo cubemapBufferInfo{};
		cubemapBufferInfo.buffer = offscreenUniformBuffer;
		cubemapBufferInfo.offset = 0;
		cubemapBufferInfo.range = sizeof(CubemapBufferObject);

		VkDescriptorImageInfo cubemapImageInfo{};
		cubemapImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		cubemapImageInfo.imageView = hdriImageView;
		cubemapImageInfo.sampler = hdriSampler;

		std::array<VkWriteDescriptorSet, 2> descWrites{};

		descWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descWrites[0].dstSet = cubemapCreateDescriptorSet;
		descWrites[0].dstBinding = 0;
		descWrites[0].dstArrayElement = 0;
		descWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descWrites[0].descriptorCount = 1;
		descWrites[0].pBufferInfo = &cubemapBufferInfo;

		descWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descWrites[1].dstSet = cubemapCreateDescriptorSet;
		descWrites[1].dstBinding = 1;
		descWrites[1].dstArrayElement = 0;
		descWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descWrites[1].descriptorCount = 1;
		descWrites[1].pImageInfo = &cubemapImageInfo;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descWrites.size()), descWrites.data(), 0, nullptr);

		//make offscreen pipeline

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_FALSE;
		depthStencil.depthWriteEnable = VK_FALSE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		auto vertShaderCode = readFile("shader/make-cubemap/cube.vert.spv");
		auto fragShaderCode = readFile("shader/make-cubemap/cube.frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkDeviceSize bufferSize = sizeof(offscreenVertices[0]) * offscreenVertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer,
			stagingBufferMemory
		);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, offscreenVertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			offscreenVertexBuffer,
			offscreenVertexBufferMemory
		);

		copyBuffer(stagingBuffer, offscreenVertexBuffer, bufferSize);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(float) * 3;
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription attributeDescription{};

		attributeDescription.binding = 0;
		attributeDescription.location = 0;
		attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescription.offset = 0;

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &cubemapCreateDescriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &cubemapCreatePipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen pipeline layout!");
		}

		/*
		VkRenderPassMultiviewCreateInfo multiviewInfo{};
		multiviewInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO_KHR;
		*/

		std::array<VkAttachmentDescription, 2> attachments;

		attachments[0].format = cubemapFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[0].flags = 0;

		attachments[1].format = findDepthFormat();
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].flags = 0;

		VkAttachmentReference colorReference = {
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		};

		VkAttachmentReference depthReference = {
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		};

		VkSubpassDescription subpassDesc = {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorReference,
			.pDepthStencilAttachment = &depthReference
		};

		VkSubpassDependency dependency;

		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 2;
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDesc;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		uint32_t viewMask = 0b00111111;

		VkRenderPassMultiviewCreateInfo multiViewInfo{};
		multiViewInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
		multiViewInfo.subpassCount = 1;
		multiViewInfo.pViewMasks = &viewMask;
		multiViewInfo.correlationMaskCount = 0;

		renderPassInfo.pNext = &multiViewInfo;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &cubemapCreateRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap render pass!");
		}

		std::array<VkImageView, 2> framebufferAttachments;
		framebufferAttachments[0] = cubemapImageView;
		framebufferAttachments[1] = offscreenDepthImageView;

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = cubemapCreateRenderPass;
		framebufferInfo.attachmentCount = 2;
		framebufferInfo.pAttachments = framebufferAttachments.data();
		framebufferInfo.width = 1024;
		framebufferInfo.height = 1024;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &cubemapCreateFramebuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap framebuffer!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = cubemapCreatePipelineLayout;
		pipelineInfo.renderPass = cubemapCreateRenderPass;
		pipelineInfo.pNext = NULL;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &cubemapCreatePipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);

		//make command buffers

		VkCommandBufferAllocateInfo commandBufferAllocInfo{};
		commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferAllocInfo.commandPool = commandPool;
		commandBufferAllocInfo.commandBufferCount = 1;
		commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(device, &commandBufferAllocInfo, &cubemapCreateCommandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen command buffers!");
		}

		vkResetCommandBuffer(cubemapCreateCommandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		//vkQueueWaitIdle(graphicsQueue);

		std::array<VkClearValue, 2>clearValues{};
		clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo mvRenderpassbeginInfo{};
		mvRenderpassbeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		mvRenderpassbeginInfo.renderPass = cubemapCreateRenderPass;
		mvRenderpassbeginInfo.renderArea.offset = { 0, 0 };
		mvRenderpassbeginInfo.renderArea.extent = { 1024, 1024 };
		mvRenderpassbeginInfo.clearValueCount = 2;
		mvRenderpassbeginInfo.pClearValues = clearValues.data();
		mvRenderpassbeginInfo.framebuffer = cubemapCreateFramebuffer;

		if (vkBeginCommandBuffer(cubemapCreateCommandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording cubemap command buffer!");
		}

		vkCmdBeginRenderPass(cubemapCreateCommandBuffer, &mvRenderpassbeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(cubemapCreateCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, cubemapCreatePipeline);

		VkBuffer vertexBuffers[] = { offscreenVertexBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(cubemapCreateCommandBuffer, 0, 1, vertexBuffers, offsets);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(1024);
		viewport.height = static_cast<float>(1024);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(cubemapCreateCommandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent.height = 1024;
		scissor.extent.width = 1024;
		vkCmdSetScissor(cubemapCreateCommandBuffer, 0, 1, &scissor);
		vkCmdBindDescriptorSets(cubemapCreateCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, cubemapCreatePipelineLayout, 0, 1, &cubemapCreateDescriptorSet, 0, nullptr);
		vkCmdDraw(cubemapCreateCommandBuffer, 36, 1, 0, 0);

		CubemapBufferObject cbo{};
		cbo.model = glm::mat4(1.0);
		cbo.proj = glm::perspective((float)(3.1417 / 2.0), 1.0f, 0.1f, 10.0f);

		//0
		glm::mat4 view0 = glm::mat4(1.0f);
		view0 = glm::rotate(view0, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		view0 = glm::rotate(view0, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[0] = view0;
		
		//1
		glm::mat4 view1 = glm::mat4(1.0f);
		view1 = glm::rotate(view1, glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		view1 = glm::rotate(view1, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[1] = view1;

		//2
		glm::mat4 view2 = glm::mat4(1.0f);
		view2 = glm::rotate(view2, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[2] = view2;

		//3
		glm::mat4 view3 = glm::mat4(1.0f);
		view3 = glm::rotate(view3, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[3] = view3;

		//4
		glm::mat4 view4 = glm::mat4(1.0f);
		view4 = glm::rotate(view4, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[4] = view4;


		//5
		glm::mat4 view5 = glm::mat4(1.0f);
		view5 = glm::rotate(view5, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		cbo.view[5] = view5;

		memcpy(offscreenUniformBufferMapped, &cbo, sizeof(CubemapBufferObject));

		vkCmdEndRenderPass(cubemapCreateCommandBuffer);

		if (vkEndCommandBuffer(cubemapCreateCommandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record cubemap command buffer for rendering~!");
		}

		//queue submit info
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cubemapCreateCommandBuffer;
		submitInfo.signalSemaphoreCount = 0;

		VkResult dubious = vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr);
		if (dubious != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw cubemap command buffer for rendering~!\n");
		}
	}

	void handleCubemapTransitions() {

		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = cubemap;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 6;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		for (uint32_t i = 1; i < 5; i++) {
			barrier.subresourceRange.baseMipLevel = i;
			barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);
		}

		endSingleTimeCommands(commandBuffer);
	}

	void generateCubemapMipMaps() {
		VkFormatProperties formatproperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, cubemapFormat, &formatproperties);

		if (!(formatproperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = cubemap;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 6;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = 1024;
		int32_t mipHeight = 1024;

		for (uint32_t i = 1; i < 5; i++) {
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };
			blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 6;
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 6;

			vkCmdBlitImage(
				commandBuffer,
				cubemap, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				cubemap, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				VK_FILTER_LINEAR
			);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier
			);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
		}

		barrier.subresourceRange.baseMipLevel = 4;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);
	}

	void createDiffuseMap() {
		VkImageCreateInfo diffuseCubemapInfo{};
		diffuseCubemapInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		diffuseCubemapInfo.imageType = VK_IMAGE_TYPE_2D;
		diffuseCubemapInfo.extent.width = 32;
		diffuseCubemapInfo.extent.height = 32;
		diffuseCubemapInfo.extent.depth = 1;
		diffuseCubemapInfo.mipLevels = 1;
		diffuseCubemapInfo.arrayLayers = 6;
		diffuseCubemapInfo.format = cubemapFormat;
		diffuseCubemapInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		diffuseCubemapInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		diffuseCubemapInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		diffuseCubemapInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		diffuseCubemapInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		diffuseCubemapInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

		if (vkCreateImage(device, &diffuseCubemapInfo, nullptr, &diffuseCubemap) != VK_SUCCESS) {
			throw std::runtime_error("failed to create diffuse cubemap image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, diffuseCubemap, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &diffuseCubemapMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate diffuseCubemap image memory!");
		}

		vkBindImageMemory(device, diffuseCubemap, diffuseCubemapMemory, 0);

		VkImageViewCreateInfo diffuseCubemapImageViewInfo{};
		diffuseCubemapImageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		diffuseCubemapImageViewInfo.image = diffuseCubemap;
		diffuseCubemapImageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		diffuseCubemapImageViewInfo.format = cubemapFormat;
		diffuseCubemapImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		diffuseCubemapImageViewInfo.subresourceRange.baseMipLevel = 0;
		diffuseCubemapImageViewInfo.subresourceRange.levelCount = 1;
		diffuseCubemapImageViewInfo.subresourceRange.baseArrayLayer = 0;
		diffuseCubemapImageViewInfo.subresourceRange.layerCount = 6;

		if (vkCreateImageView(device, &diffuseCubemapImageViewInfo, nullptr, &diffuseCubemapImageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create diffuse cubemap image view!");
		}

		VkSamplerCreateInfo diffuseCubemapSamplerInfo{};
		diffuseCubemapSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		diffuseCubemapSamplerInfo.magFilter = VK_FILTER_LINEAR;
		diffuseCubemapSamplerInfo.minFilter = VK_FILTER_LINEAR;
		diffuseCubemapSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		diffuseCubemapSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		diffuseCubemapSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		diffuseCubemapSamplerInfo.anisotropyEnable = VK_TRUE;
		diffuseCubemapSamplerInfo.maxAnisotropy = 1.0f;
		diffuseCubemapSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		diffuseCubemapSamplerInfo.unnormalizedCoordinates = VK_FALSE;
		diffuseCubemapSamplerInfo.compareEnable = VK_FALSE;
		diffuseCubemapSamplerInfo.compareOp = VK_COMPARE_OP_NEVER;
		diffuseCubemapSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		diffuseCubemapSamplerInfo.minLod = 0.0f;
		diffuseCubemapSamplerInfo.maxLod = 1.0f;
		diffuseCubemapSamplerInfo.mipLodBias = 0.0f;
		diffuseCubemapSamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

		if (vkCreateSampler(device, &diffuseCubemapSamplerInfo, nullptr, &diffuseCubemapSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create diffuse cubemap sampler!");
		}
	}

	void performOffscreenDiffuseMapRender() {

		//setup descriptors
		//pool
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = 1;
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = 1;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 2;
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = 1;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &diffuseDescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen diffuse descriptor pool!");
		}

		//layout
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding cubemapLayoutBinding{};
		cubemapLayoutBinding.binding = 1;
		cubemapLayoutBinding.descriptorCount = 1;
		cubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		cubemapLayoutBinding.pImmutableSamplers = nullptr;
		cubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
			uboLayoutBinding,
			cubemapLayoutBinding
		};

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 2;
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &diffuseDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen diffuse descriptor set layout!");
		}

		//sets
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = diffuseDescriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &diffuseDescriptorSetLayout;

		if (vkAllocateDescriptorSets(device, &allocInfo, &diffuseDescriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate offscreen diffuse descriptor set!");
		}

		VkDescriptorBufferInfo cubemapBufferInfo{};
		cubemapBufferInfo.buffer = offscreenUniformBuffer;
		cubemapBufferInfo.offset = 0;
		cubemapBufferInfo.range = sizeof(CubemapBufferObject);

		VkDescriptorImageInfo cubemapImageInfo{};
		cubemapImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		cubemapImageInfo.imageView = cubemapImageView;
		cubemapImageInfo.sampler = cubemapSampler;

		std::array<VkWriteDescriptorSet, 2> descWrites{};

		descWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descWrites[0].dstSet = diffuseDescriptorSet;
		descWrites[0].dstBinding = 0;
		descWrites[0].dstArrayElement = 0;
		descWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descWrites[0].descriptorCount = 1;
		descWrites[0].pBufferInfo = &cubemapBufferInfo;

		descWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descWrites[1].dstSet = diffuseDescriptorSet;
		descWrites[1].dstBinding = 1;
		descWrites[1].dstArrayElement = 0;
		descWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descWrites[1].descriptorCount = 1;
		descWrites[1].pImageInfo = &cubemapImageInfo;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descWrites.size()), descWrites.data(), 0, nullptr);

		//make offscreen pipeline
		//sytart
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_FALSE;
		depthStencil.depthWriteEnable = VK_FALSE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		auto vertShaderCode = readFile("shader/make-diffuse/diffuse.vert.spv");
		auto fragShaderCode = readFile("shader/make-diffuse/diffuse.frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(float) * 3;
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription attributeDescription{};

		attributeDescription.binding = 0;
		attributeDescription.location = 0;
		attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescription.offset = 0;

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &diffuseDescriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &diffuseCubemapPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen diffuse pipeline layout!");
		}

		std::array<VkAttachmentDescription, 2> attachments;

		attachments[0].format = cubemapFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[0].flags = 0;

		attachments[1].format = findDepthFormat();
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].flags = 0;

		VkAttachmentReference colorReference = {
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		};

		VkAttachmentReference depthReference = {
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		};

		VkSubpassDescription subpassDesc = {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorReference,
			.pDepthStencilAttachment = &depthReference
		};

		VkSubpassDependency dependency;

		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 2;
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDesc;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		uint32_t viewMask = 0b00111111;

		VkRenderPassMultiviewCreateInfo multiViewInfo{};
		multiViewInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
		multiViewInfo.subpassCount = 1;
		multiViewInfo.pViewMasks = &viewMask;
		multiViewInfo.correlationMaskCount = 0;

		renderPassInfo.pNext = &multiViewInfo;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &diffuseRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap render pass!");
		}

		std::array<VkImageView, 2> framebufferAttachments;
		framebufferAttachments[0] = diffuseCubemapImageView;
		framebufferAttachments[1] = offscreenDepthImageView;

		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = diffuseRenderPass;
		framebufferInfo.attachmentCount = 2;
		framebufferInfo.pAttachments = framebufferAttachments.data();
		framebufferInfo.width = 32;
		framebufferInfo.height = 32;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &diffuseFramebuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create diffuse cubemap framebuffer!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = diffuseCubemapPipelineLayout;
		pipelineInfo.renderPass = diffuseRenderPass;
		pipelineInfo.pNext = NULL;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &diffuseCubemapPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen diffuse graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		//end

		//make command buffers
		VkCommandBuffer offscreenCommandBuffer;

		VkCommandBufferAllocateInfo commandBufferAllocInfo{};
		commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferAllocInfo.commandPool = commandPool;
		commandBufferAllocInfo.commandBufferCount = 1;
		commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(device, &commandBufferAllocInfo, &offscreenCommandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen command buffers!");
		}

		vkResetCommandBuffer(offscreenCommandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		//vkQueueWaitIdle(graphicsQueue);

		std::array<VkClearValue, 2>clearValues{};
		clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo mvRenderpassbeginInfo{};
		mvRenderpassbeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		mvRenderpassbeginInfo.renderPass = diffuseRenderPass;
		mvRenderpassbeginInfo.renderArea.offset = { 0, 0 };
		mvRenderpassbeginInfo.renderArea.extent = { 32, 32 };
		mvRenderpassbeginInfo.clearValueCount = 2;
		mvRenderpassbeginInfo.pClearValues = clearValues.data();
		mvRenderpassbeginInfo.framebuffer = diffuseFramebuffer;

		if (vkBeginCommandBuffer(offscreenCommandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording cubemap command buffer!");
		}

		vkCmdBeginRenderPass(offscreenCommandBuffer, &mvRenderpassbeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(offscreenCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, diffuseCubemapPipeline);

		VkBuffer vertexBuffers[] = { offscreenVertexBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(offscreenCommandBuffer, 0, 1, vertexBuffers, offsets);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(32);
		viewport.height = static_cast<float>(32);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(offscreenCommandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent.height = 32;
		scissor.extent.width = 32;
		vkCmdSetScissor(offscreenCommandBuffer, 0, 1, &scissor);
		vkCmdBindDescriptorSets(offscreenCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, diffuseCubemapPipelineLayout, 0, 1, &diffuseDescriptorSet, 0, nullptr);
		vkCmdDraw(offscreenCommandBuffer, 36, 1, 0, 0);

		CubemapBufferObject cbo{};
		cbo.model = glm::mat4(1.0);
		cbo.proj = glm::perspective((float)(3.1417 / 2.0), 1.0f, 0.1f, 10.0f);

		//0
		glm::mat4 view0 = glm::mat4(1.0f);
		view0 = glm::rotate(view0, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		view0 = glm::rotate(view0, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[0] = view0;

		//1
		glm::mat4 view1 = glm::mat4(1.0f);
		view1 = glm::rotate(view1, glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		view1 = glm::rotate(view1, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[1] = view1;

		//2
		glm::mat4 view2 = glm::mat4(1.0f);
		view2 = glm::rotate(view2, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[2] = view2;

		//3
		glm::mat4 view3 = glm::mat4(1.0f);
		view3 = glm::rotate(view3, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[3] = view3;

		//4
		glm::mat4 view4 = glm::mat4(1.0f);
		view4 = glm::rotate(view4, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[4] = view4;


		//5
		glm::mat4 view5 = glm::mat4(1.0f);
		view5 = glm::rotate(view5, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		cbo.view[5] = view5;

		memcpy(offscreenUniformBufferMapped, &cbo, sizeof(CubemapBufferObject));

		vkCmdEndRenderPass(offscreenCommandBuffer);

		if (vkEndCommandBuffer(offscreenCommandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record diffuse cubemap command buffer for rendering~!");
		}

		//queue submit info
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &offscreenCommandBuffer;
		submitInfo.signalSemaphoreCount = 0;

		VkResult dubious = vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr);
		if (dubious != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw diffuse cubemap command buffer for rendering~!\n");
		}
	}

	void createPrefilterMap() {
		VkImageCreateInfo prefilterCubemapInfo{};
		prefilterCubemapInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		prefilterCubemapInfo.imageType = VK_IMAGE_TYPE_2D;
		prefilterCubemapInfo.extent.width = 128;
		prefilterCubemapInfo.extent.height = 128;
		prefilterCubemapInfo.extent.depth = 1;
		prefilterCubemapInfo.mipLevels = 5;
		prefilterCubemapInfo.arrayLayers = 6;
		prefilterCubemapInfo.format = cubemapFormat;
		prefilterCubemapInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		prefilterCubemapInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		prefilterCubemapInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		prefilterCubemapInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		prefilterCubemapInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		prefilterCubemapInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

		if (vkCreateImage(device, &prefilterCubemapInfo, nullptr, &prefilterCubemap) != VK_SUCCESS) {
			throw std::runtime_error("failed to create prefilter cubemap image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, prefilterCubemap, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &prefilterCubemapMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate prefilterCubemap image memory!");
		}

		vkBindImageMemory(device, prefilterCubemap, prefilterCubemapMemory, 0);

		VkImageViewCreateInfo prefilterCubemapImageViewInfo{};
		prefilterCubemapImageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		prefilterCubemapImageViewInfo.image = prefilterCubemap;
		prefilterCubemapImageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		prefilterCubemapImageViewInfo.format = cubemapFormat;
		prefilterCubemapImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		prefilterCubemapImageViewInfo.subresourceRange.levelCount = 1;
		prefilterCubemapImageViewInfo.subresourceRange.baseArrayLayer = 0;
		prefilterCubemapImageViewInfo.subresourceRange.layerCount = 6;
		
		for (int i = 0; i < 5; i++) {
			prefilterCubemapImageViewInfo.subresourceRange.baseMipLevel = i;
			if (vkCreateImageView(device, &prefilterCubemapImageViewInfo, nullptr, &prefilterCubemapImageViewPerMip[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create prefilter cubemap image view(s)!");
			}
		}

		//for final use
		prefilterCubemapImageViewInfo.subresourceRange.baseMipLevel = 0;
		prefilterCubemapImageViewInfo.subresourceRange.levelCount = 5;
		if (vkCreateImageView(device, &prefilterCubemapImageViewInfo, nullptr, &prefilterCubemapImageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create prefilter cubemap image view!");
		}

		VkSamplerCreateInfo prefilterCubemapSamplerInfo{};
		prefilterCubemapSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		prefilterCubemapSamplerInfo.magFilter = VK_FILTER_LINEAR;
		prefilterCubemapSamplerInfo.minFilter = VK_FILTER_LINEAR;
		prefilterCubemapSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		prefilterCubemapSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		prefilterCubemapSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		prefilterCubemapSamplerInfo.anisotropyEnable = VK_TRUE;
		prefilterCubemapSamplerInfo.maxAnisotropy = 1.0f;
		prefilterCubemapSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		prefilterCubemapSamplerInfo.unnormalizedCoordinates = VK_FALSE;
		prefilterCubemapSamplerInfo.compareEnable = VK_FALSE;
		prefilterCubemapSamplerInfo.compareOp = VK_COMPARE_OP_NEVER;
		prefilterCubemapSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		prefilterCubemapSamplerInfo.minLod = 0.0f;
		prefilterCubemapSamplerInfo.maxLod = 5.0f;
		prefilterCubemapSamplerInfo.mipLodBias = 0.0f;
		prefilterCubemapSamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

		if (vkCreateSampler(device, &prefilterCubemapSamplerInfo, nullptr, &prefilterCubemapSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create prefilter cubemap sampler!");
		}
	}

	void performPrefilterMapRender() {

		//setup descriptors
		//pool
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = 1;
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = 1;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 2;
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = 1;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &prefilterDescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create prefilter descriptor pool!");
		}

		//layout
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding cubemapLayoutBinding{};
		cubemapLayoutBinding.binding = 1;
		cubemapLayoutBinding.descriptorCount = 1;
		cubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		cubemapLayoutBinding.pImmutableSamplers = nullptr;
		cubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
			uboLayoutBinding,
			cubemapLayoutBinding
		};

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 2;
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &prefilterDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create prefilter descriptor set layout!");
		}

		//sets
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = prefilterDescriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &prefilterDescriptorSetLayout;

		if (vkAllocateDescriptorSets(device, &allocInfo, &prefilterDescriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate offscreen diffuse descriptor set!");
		}

		VkDescriptorBufferInfo cubemapBufferInfo{};
		cubemapBufferInfo.buffer = offscreenUniformBuffer;
		cubemapBufferInfo.offset = 0;
		cubemapBufferInfo.range = sizeof(CubemapBufferObject);

		VkDescriptorImageInfo cubemapImageInfo{};
		cubemapImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		cubemapImageInfo.imageView = cubemapImageView;
		cubemapImageInfo.sampler = cubemapSampler;

		std::array<VkWriteDescriptorSet, 2> descWrites{};

		descWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descWrites[0].dstSet = prefilterDescriptorSet;
		descWrites[0].dstBinding = 0;
		descWrites[0].dstArrayElement = 0;
		descWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descWrites[0].descriptorCount = 1;
		descWrites[0].pBufferInfo = &cubemapBufferInfo;

		descWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descWrites[1].dstSet = prefilterDescriptorSet;
		descWrites[1].dstBinding = 1;
		descWrites[1].dstArrayElement = 0;
		descWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descWrites[1].descriptorCount = 1;
		descWrites[1].pImageInfo = &cubemapImageInfo;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descWrites.size()), descWrites.data(), 0, nullptr);

		//make offscreen pipeline
		//sytart
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_FALSE;
		depthStencil.depthWriteEnable = VK_FALSE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		auto vertShaderCode = readFile("shader/make-prefiltermap/prefilter.vert.spv");
		auto fragShaderCode = readFile("shader/make-prefiltermap/prefilter.frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(float) * 3;
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription attributeDescription{};

		attributeDescription.binding = 0;
		attributeDescription.location = 0;
		attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescription.offset = 0;

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPushConstantRange pushConstant;
		pushConstant.offset = 0;
		pushConstant.size = sizeof(float);
		pushConstant.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &prefilterDescriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		pipelineLayoutInfo.pPushConstantRanges = &pushConstant;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &prefilterPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen diffuse pipeline layout!");
		}

		std::array<VkAttachmentDescription, 2> attachments;

		attachments[0].format = cubemapFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[0].flags = 0;

		attachments[1].format = findDepthFormat();
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].flags = 0;

		VkAttachmentReference colorReference = {
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		};

		VkAttachmentReference depthReference = {
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		};

		VkSubpassDescription subpassDesc = {
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorReference,
			.pDepthStencilAttachment = &depthReference
		};

		VkSubpassDependency dependency;

		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 2;
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpassDesc;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		uint32_t viewMask = 0b00111111;

		VkRenderPassMultiviewCreateInfo multiViewInfo{};
		multiViewInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
		multiViewInfo.subpassCount = 1;
		multiViewInfo.pViewMasks = &viewMask;
		multiViewInfo.correlationMaskCount = 0;

		renderPassInfo.pNext = &multiViewInfo;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &prefilterRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create prefilter render pass!");
		}

		const uint32_t PREFILTER_MAP_BASE_DIMENSION = 128;

		for (int i = 0; i < 5; i++) {
			std::array<VkImageView, 2> framebufferAttachments;
			framebufferAttachments[0] = prefilterCubemapImageViewPerMip[i];
			framebufferAttachments[1] = offscreenDepthImageView;

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = prefilterRenderPass;
			framebufferInfo.attachmentCount = 2;
			framebufferInfo.pAttachments = framebufferAttachments.data();
			framebufferInfo.width = PREFILTER_MAP_BASE_DIMENSION * pow(0.5, i);
			framebufferInfo.height = PREFILTER_MAP_BASE_DIMENSION * pow(0.5, i);
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &prefilterFramebufferPerMip[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create diffuse cubemap framebuffer!");
			}
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = prefilterPipelineLayout;
		pipelineInfo.renderPass = prefilterRenderPass;
		pipelineInfo.pNext = NULL;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &prefilterPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen diffuse graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		//end

		//make command buffers
		VkCommandBuffer offscreenCommandBuffer;

		VkCommandBufferAllocateInfo commandBufferAllocInfo{};
		commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferAllocInfo.commandPool = commandPool;
		commandBufferAllocInfo.commandBufferCount = 1;
		commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(device, &commandBufferAllocInfo, &offscreenCommandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen command buffers!");
		}

		vkResetCommandBuffer(offscreenCommandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		//vkQueueWaitIdle(graphicsQueue);

		std::array<VkClearValue, 2>clearValues{};
		clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
		clearValues[1].depthStencil = { 1.0f, 0 };

		CubemapBufferObject cbo{};
		cbo.model = glm::mat4(1.0);
		cbo.proj = glm::perspective((float)(3.1417 / 2.0), 1.0f, 0.1f, 10.0f);

		//0
		glm::mat4 view0 = glm::mat4(1.0f);
		view0 = glm::rotate(view0, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		view0 = glm::rotate(view0, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[0] = view0;

		//1
		glm::mat4 view1 = glm::mat4(1.0f);
		view1 = glm::rotate(view1, glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		view1 = glm::rotate(view1, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[1] = view1;

		//2
		glm::mat4 view2 = glm::mat4(1.0f);
		view2 = glm::rotate(view2, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[2] = view2;

		//3
		glm::mat4 view3 = glm::mat4(1.0f);
		view3 = glm::rotate(view3, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[3] = view3;

		//4
		glm::mat4 view4 = glm::mat4(1.0f);
		view4 = glm::rotate(view4, glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		cbo.view[4] = view4;


		//5
		glm::mat4 view5 = glm::mat4(1.0f);
		view5 = glm::rotate(view5, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		cbo.view[5] = view5;

		memcpy(offscreenUniformBufferMapped, &cbo, sizeof(CubemapBufferObject));

		uint32_t workingDimension = PREFILTER_MAP_BASE_DIMENSION;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		vkCreateFence(device, &fenceInfo, nullptr, &prefilterMipRenderedFence);


		for (int i = 0; i < 5; i++) {

			vkWaitForFences(device, 1, &prefilterMipRenderedFence, VK_TRUE, UINT64_MAX);

			VkRenderPassBeginInfo mvRenderpassbeginInfo{};
			mvRenderpassbeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			mvRenderpassbeginInfo.renderPass = prefilterRenderPass;
			mvRenderpassbeginInfo.renderArea.offset = { 0, 0 };
			mvRenderpassbeginInfo.renderArea.extent = { workingDimension, workingDimension };
			mvRenderpassbeginInfo.clearValueCount = 2;
			mvRenderpassbeginInfo.pClearValues = clearValues.data();
			mvRenderpassbeginInfo.framebuffer = prefilterFramebufferPerMip[i];

			if (vkBeginCommandBuffer(offscreenCommandBuffer, &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording cubemap command buffer!");
			}

			vkCmdBeginRenderPass(offscreenCommandBuffer, &mvRenderpassbeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(offscreenCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, prefilterPipeline);

			VkBuffer vertexBuffers[] = { offscreenVertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(offscreenCommandBuffer, 0, 1, vertexBuffers, offsets);

			VkViewport viewport{};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = static_cast<float>(workingDimension);
			viewport.height = static_cast<float>(workingDimension);
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			vkCmdSetViewport(offscreenCommandBuffer, 0, 1, &viewport);

			VkRect2D scissor{};
			scissor.offset = { 0, 0 };
			scissor.extent.height = workingDimension;
			scissor.extent.width = workingDimension;
			vkCmdSetScissor(offscreenCommandBuffer, 0, 1, &scissor);
			vkCmdBindDescriptorSets(offscreenCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, prefilterPipelineLayout, 0, 1, &prefilterDescriptorSet, 0, nullptr);
			float roughness = (float)i/(float)5;
			vkCmdPushConstants(offscreenCommandBuffer, prefilterPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &roughness);
			vkCmdDraw(offscreenCommandBuffer, 36, 1, 0, 0);

			vkCmdEndRenderPass(offscreenCommandBuffer);

			if (vkEndCommandBuffer(offscreenCommandBuffer) != VK_SUCCESS) {
				throw std::runtime_error("failed to record prefilter cubemap command buffer for rendering~!");
			}

			//queue submit info
			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.waitSemaphoreCount = 0;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &offscreenCommandBuffer;
			submitInfo.signalSemaphoreCount = 0;

			vkResetFences(device, 1, &prefilterMipRenderedFence);

			VkResult dubious = vkQueueSubmit(graphicsQueue, 1, &submitInfo, prefilterMipRenderedFence);
			if (dubious != VK_SUCCESS) {
				throw std::runtime_error("failed to submit draw prefilter cubemap command buffer for rendering~!\n");
			}

			workingDimension /= 2;
		}
	}

	void createBRDFLUT() {
		VkImageCreateInfo brdfLUTInfo{};
		brdfLUTInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		brdfLUTInfo.imageType = VK_IMAGE_TYPE_2D;
		brdfLUTInfo.extent.width = 512;
		brdfLUTInfo.extent.height = 512;
		brdfLUTInfo.extent.depth = 1;
		brdfLUTInfo.mipLevels = 1;
		brdfLUTInfo.arrayLayers = 1;
		brdfLUTInfo.format = brdfFormat;
		brdfLUTInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		brdfLUTInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		brdfLUTInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		brdfLUTInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		brdfLUTInfo.samples = VK_SAMPLE_COUNT_1_BIT;

		if (vkCreateImage(device, &brdfLUTInfo, nullptr, &brdfLUT) != VK_SUCCESS) {
			throw std::runtime_error("failed to create brdfLUT image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, brdfLUT, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &brdfLUTMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate brdfLUT image memory!");
		}

		vkBindImageMemory(device, brdfLUT, brdfLUTMemory, 0);

		VkImageViewCreateInfo brdfLUTImageViewInfo{};
		brdfLUTImageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		brdfLUTImageViewInfo.image = brdfLUT;
		brdfLUTImageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		brdfLUTImageViewInfo.format = brdfFormat;
		brdfLUTImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		brdfLUTImageViewInfo.subresourceRange.levelCount = 1;
		brdfLUTImageViewInfo.subresourceRange.baseArrayLayer = 0;
		brdfLUTImageViewInfo.subresourceRange.layerCount = 1;
		brdfLUTImageViewInfo.subresourceRange.baseMipLevel = 0;
		brdfLUTImageViewInfo.subresourceRange.levelCount = 1;
		if (vkCreateImageView(device, &brdfLUTImageViewInfo, nullptr, &brdfLUTImageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create brdfLUT image view!");
		}

		VkSamplerCreateInfo brdfLUTSamplerInfo{};
		brdfLUTSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		brdfLUTSamplerInfo.magFilter = VK_FILTER_LINEAR;
		brdfLUTSamplerInfo.minFilter = VK_FILTER_LINEAR;
		brdfLUTSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		brdfLUTSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		brdfLUTSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		brdfLUTSamplerInfo.anisotropyEnable = VK_TRUE;
		brdfLUTSamplerInfo.maxAnisotropy = 1.0f;
		brdfLUTSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		brdfLUTSamplerInfo.unnormalizedCoordinates = VK_FALSE;
		brdfLUTSamplerInfo.compareEnable = VK_FALSE;
		brdfLUTSamplerInfo.compareOp = VK_COMPARE_OP_NEVER;
		brdfLUTSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		brdfLUTSamplerInfo.minLod = 0.0f;
		brdfLUTSamplerInfo.maxLod = 1.0f;
		brdfLUTSamplerInfo.mipLodBias = 0.0f;
		brdfLUTSamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

		if (vkCreateSampler(device, &brdfLUTSamplerInfo, nullptr, &brdfLUTSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create brdfLUT sampler!");
		}

		transitionImageLayout(
			brdfLUT,
			brdfFormat,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			1
		);
	}

	void performBRDFLUTRender() {

		//handle descriptors
		//start
		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 0;
		poolInfo.maxSets = 1;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &brdfDescPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create brdf descriptor pool!");
		}

		//layout

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 0;

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &brdfDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create brdf descriptor set layout!");
		}

		//make pipeline
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_FALSE;
		depthStencil.depthWriteEnable = VK_FALSE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		//make brdf shaders
		auto vertShaderCode = readFile("shader/make-brdf/brdf.vert.spv");
		auto fragShaderCode = readFile("shader/make-brdf/brdf.frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		//make vertex buffer here
		std::vector<float> brdfVertices = {
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};

		VkDeviceSize brdfBufferSize = sizeof(float) * brdfVertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(
			brdfBufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer,
			stagingBufferMemory
		);

		void* brdfData;
		vkMapMemory(device, stagingBufferMemory, 0, brdfBufferSize, 0, &brdfData);
		memcpy(brdfData, brdfVertices.data(), (size_t)brdfBufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(
			brdfBufferSize,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			brdfVertexBuffer,
			brdfVertexMemory
		);

		copyBuffer(stagingBuffer, brdfVertexBuffer, brdfBufferSize);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

		VkVertexInputBindingDescription bindingDescription;
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(float) * 5;
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		std::array<VkVertexInputAttributeDescription, 2> attributeDescription;

		attributeDescription[0].binding = 0;
		attributeDescription[0].location = 0;
		attributeDescription[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescription[0].offset = 0;

		attributeDescription[1].binding = 0;
		attributeDescription[1].location = 1;
		attributeDescription[1].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescription[1].offset = sizeof(float) * 3;

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = 2;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescription.data();

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 0;
		//pipelineLayoutInfo.pSetLayouts = &brdfDescriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &brdfPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create brdf pipeline layout!");
		}

		VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &brdfFormat,
			.depthAttachmentFormat = findDepthFormat()
		};

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = brdfPipelineLayout;
		pipelineInfo.renderPass = NULL;
		pipelineInfo.pNext = &pipelineRenderingCreateInfo;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &brdfPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create brdfPipeline!");
		}

		//add vertex data here

		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		//end

		//make command buffers
		VkCommandBuffer brdfCommandBuffer;

		VkCommandBufferAllocateInfo commandBufferAllocInfo{};
		commandBufferAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferAllocInfo.commandPool = commandPool;
		commandBufferAllocInfo.commandBufferCount = 1;
		commandBufferAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(device, &commandBufferAllocInfo, &brdfCommandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create brdf command buffers!");
		}

		vkResetCommandBuffer(brdfCommandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		//vkQueueWaitIdle(graphicsQueue);

		std::array<VkClearValue, 2>clearValues{};
		clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
		clearValues[1].depthStencil = { 1.0f, 0 };

		uint32_t BRDF_MAP_DIMENSION = 512;
	
		if (vkBeginCommandBuffer(brdfCommandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin brdf command buffer!");
		}

		VkRenderingAttachmentInfoKHR colorAttachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = brdfLUTImageView,
			.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = clearValues[0]
		};

		VkRenderingAttachmentInfoKHR depthAttachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = depthImageView,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.clearValue = clearValues[1]
		};

		VkRenderingInfoKHR renderingInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
			.renderArea = {0, 0, BRDF_MAP_DIMENSION, BRDF_MAP_DIMENSION},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachment,
			.pDepthAttachment = &depthAttachment,
		};

		vkCmdBeginRendering(brdfCommandBuffer, &renderingInfo);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = BRDF_MAP_DIMENSION;
		viewport.height = BRDF_MAP_DIMENSION;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(brdfCommandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = { BRDF_MAP_DIMENSION, BRDF_MAP_DIMENSION };
		vkCmdSetScissor(brdfCommandBuffer, 0, 1, &scissor);

		vkCmdBindPipeline(brdfCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, brdfPipeline);

		VkBuffer brdfBuffers[] = { brdfVertexBuffer };
		VkDeviceSize offsets[] = { 0 };

		vkCmdBindVertexBuffers(brdfCommandBuffer, 0, 1, brdfBuffers, offsets);
		//vkCmdBindDescriptorSets(brdfCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, brdfPipelineLayout, 0, 1, &brdfDescriptorSet, 0, nullptr);
		vkCmdDraw(brdfCommandBuffer, 6, 1, 0, 0);

		vkCmdEndRendering(brdfCommandBuffer);

		vkEndCommandBuffer(brdfCommandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &brdfCommandBuffer;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit brdf draw command buffer!");
		}

		brdfCommandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier colorImageChangeToReadOnlyBarrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = brdfLUT,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		vkCmdPipelineBarrier(
			brdfCommandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&colorImageChangeToReadOnlyBarrier
		);

		endSingleTimeCommands(brdfCommandBuffer);
	}

	void createCubemapRenderStuff() {

		//handle descriptors
		//start
		std::array<VkDescriptorPoolSize, 2> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = 1;
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = 1;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = 2;
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = 1;

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &cubemapDescPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap descriptor pool!");
		}

		//layout
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding cubemapLayoutBinding{};
		cubemapLayoutBinding.binding = 1;
		cubemapLayoutBinding.descriptorCount = 1;
		cubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		cubemapLayoutBinding.pImmutableSamplers = nullptr;
		cubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
			uboLayoutBinding,
			cubemapLayoutBinding
		};

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = 2;
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &cubemapDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap descriptor set layout!");
		}

		//sets
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = cubemapDescPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &cubemapDescriptorSetLayout;

		if (vkAllocateDescriptorSets(device, &allocInfo, &cubemapDescriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate cubemap descriptor set!");
		}

		VkDescriptorBufferInfo uniformBufferInfo{};
		uniformBufferInfo.buffer = uniformBuffers[0];
		uniformBufferInfo.offset = 0;
		uniformBufferInfo.range = sizeof(UniformBufferObject);

		VkDescriptorImageInfo cubemapImageInfo{};
		cubemapImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		cubemapImageInfo.imageView = cubemapImageView;
		cubemapImageInfo.sampler = cubemapSampler;

		std::array<VkWriteDescriptorSet, 2> descWrites{};

		descWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descWrites[0].dstSet = cubemapDescriptorSet;
		descWrites[0].dstBinding = 0;
		descWrites[0].dstArrayElement = 0;
		descWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descWrites[0].descriptorCount = 1;
		descWrites[0].pBufferInfo = &uniformBufferInfo;

		descWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descWrites[1].dstSet = cubemapDescriptorSet;
		descWrites[1].dstBinding = 1;
		descWrites[1].dstArrayElement = 0;
		descWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descWrites[1].descriptorCount = 1;
		descWrites[1].pImageInfo = &cubemapImageInfo;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descWrites.size()), descWrites.data(), 0, nullptr);

		//make pipeline
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_FALSE;
		depthStencil.depthWriteEnable = VK_FALSE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		//make cubemap shaders
		auto vertShaderCode = readFile("shader/cubemap/cubemap.vert.spv");
		auto fragShaderCode = readFile("shader/cubemap/cubemap.frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(float) * 3;
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription attributeDescription{};

		attributeDescription.binding = 0;
		attributeDescription.location = 0;
		attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescription.offset = 0;

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.vertexAttributeDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &cubemapDescriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &cubemapPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap pipeline layout!");
		}

		VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
			.colorAttachmentCount = 1,
			.pColorAttachmentFormats = &swapChainImageFormat,
			.depthAttachmentFormat = findDepthFormat()
		};

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = cubemapPipelineLayout;
		pipelineInfo.renderPass = NULL;
		pipelineInfo.pNext = &pipelineRenderingCreateInfo;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &cubemapPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create cubemap graphics pipeline!");
		}

		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
	}

	void loadModel() {
		std::cout << "trying to load model...";
		for (int i = 0; i < model.pos.size(); i++) {
			Vertex vertex{};

			vertex.pos = model.pos[i];
			vertex.normal = model.normals[i];
			if (model.hasTangents) vertex.tangent = model.tangents[i];
			else vertex.tangent = glm::vec4(0.0);
			vertex.texCoord = model.texCoords[i];

			vertices.push_back(vertex);
		}
		for (int i = 0; i < model.indices.size(); i++) {
			indices.push_back(model.indices[i]);
		}

		glm::mat4 S = glm::scale(glm::mat4(1.0), glm::vec3(model.transformData.scale[0], model.transformData.scale[1], model.transformData.scale[2]));

		glm::mat4 R = glm::mat4_cast(glm::quat(model.transformData.rotation[3], model.transformData.rotation[0], model.transformData.rotation[1], model.transformData.rotation[2]));

		TRS = glm::translate(S * R, glm::vec3(model.transformData.translation[0], model.transformData.translation[1], model.transformData.translation[2]));
		//glm::mat4 R = glm::rotate(glm::mat4(1.0), glm::vec4(model.transformData.rotation[0], model.transformData.rotation[1], model.transformData.rotation[2], model.transformData.rotation[3])));
		//glm::mat4 R = glm::rotate(glm::mat4(1.0f), )

		//indices = &model.indices;
		std::cout << " done!\n";
	}

	void createVertexBuffer() {
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer,
			stagingBufferMemory
		);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			vertexBuffer,
			vertexBufferMemory
		);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = { width, height, 1 };

		vkCmdCopyBufferToImage(
			commandBuffer,
			buffer,
			image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&region
		);

		endSingleTimeCommands(commandBuffer);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	void createIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer,
			stagingBufferMemory
		);

		void* data;
		VkResult indexMapResult = vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		if (indexMapResult != VK_SUCCESS) {
			throw std::runtime_error("failed to map index buffer!");
		}
		assert(bufferSize == sizeof(indices[0]) * indices.size());
		memcpy(data, indices.data(), (size_t)bufferSize);

		//try copying memory in chunks
		
		/*
		const int MEMORY_CHUNKS_SPLIT_SIZE = 2;
		for (int i = 0; i < MEMORY_CHUNKS_SPLIT_SIZE; i++) {
			memcpy(reinterpret_cast<void*>(&data + (bufferSize * i / MEMORY_CHUNKS_SPLIT_SIZE)), reinterpret_cast<void*>(indices->data() + (bufferSize * i / MEMORY_CHUNKS_SPLIT_SIZE)), (size_t)bufferSize / MEMORY_CHUNKS_SPLIT_SIZE);
		}
		*/
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(
			bufferSize,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			indexBuffer,
			indexBufferMemory
		);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
		vkDestroyCommandPool(device, transferPool, nullptr);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(
				bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				uniformBuffers[i],
				uniformBuffersMemory[i]
			);
			vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}

		bufferSize = sizeof(MaterialBufferObject);

		materialBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		materialBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		materialBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(
				bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				materialBuffers[i],
				materialBuffersMemory[i]
			);
			vkMapMemory(device, materialBuffersMemory[i], 0, bufferSize, 0, &materialBuffersMapped[i]);
		}
	}

	void createDescriptorPool() {
		std::vector<VkDescriptorPoolSize> poolSizes{};

		uint16_t noUniforms = 2;

		for (uint16_t i = 0; i < noUniforms; i++) {
			VkDescriptorPoolSize poolSize = {
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
			};
			poolSizes.push_back(poolSize);
		}

		uint16_t noTextures = 7;
		if (model.mat.isEmissiveTexPresent) noTextures++;
		if (model.mat.isOcclusionTexPresent) noTextures++;

		for (uint16_t i = 0; i < noTextures; i++) {
			VkDescriptorPoolSize poolSize = {
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)
			};
			poolSizes.push_back(poolSize);
		}

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}

	}

	void createDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();
		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);

		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			VkDescriptorBufferInfo uniformBufferInfo{};
			uniformBufferInfo.buffer = uniformBuffers[i];
			uniformBufferInfo.offset = 0;
			uniformBufferInfo.range = sizeof(UniformBufferObject);

			VkDescriptorBufferInfo materialBufferInfo{};
			materialBufferInfo.buffer = materialBuffers[i];
			materialBufferInfo.offset = 0;
			materialBufferInfo.range = sizeof(MaterialBufferObject);

			VkDescriptorImageInfo baseColorImageInfo{};
			baseColorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			baseColorImageInfo.imageView = baseColorImageView;
			baseColorImageInfo.sampler = baseColorSampler;

			VkDescriptorImageInfo metallicRoughnessImageInfo{};
			metallicRoughnessImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			metallicRoughnessImageInfo.imageView = metallicRoughnessImageView;
			metallicRoughnessImageInfo.sampler = metallicRoughnessSampler;

			VkDescriptorImageInfo normalImageInfo{};
			normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			normalImageInfo.imageView = normalImageView;
			normalImageInfo.sampler = normalSampler;

			VkDescriptorImageInfo occlusionImageInfo{};
			occlusionImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			occlusionImageInfo.imageView = occlusionImageView;
			occlusionImageInfo.sampler = occlusionSampler;

			VkDescriptorImageInfo emissiveImageInfo{};
			emissiveImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			emissiveImageInfo.imageView = emissiveImageView;
			emissiveImageInfo.sampler = emissiveSampler;

			VkDescriptorImageInfo cubemapInfo{};
			cubemapInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			cubemapInfo.imageView = cubemapImageView;
			cubemapInfo.sampler = cubemapSampler;

			VkDescriptorImageInfo diffuseCubemapInfo{};
			diffuseCubemapInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			diffuseCubemapInfo.imageView = diffuseCubemapImageView;
			diffuseCubemapInfo.sampler = diffuseCubemapSampler;

			VkDescriptorImageInfo prefilterCubemapInfo{};
			prefilterCubemapInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			prefilterCubemapInfo.imageView = prefilterCubemapImageView;
			prefilterCubemapInfo.sampler = prefilterCubemapSampler;

			VkDescriptorImageInfo brdfInfo{};
			brdfInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			brdfInfo.imageView = brdfLUTImageView;
			brdfInfo.sampler = brdfLUTSampler;

			std::array<VkWriteDescriptorSet, 11> descWrites{};

			descWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[0].dstSet = descriptorSets[i];
			descWrites[0].dstBinding = 0;
			descWrites[0].dstArrayElement = 0;
			descWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descWrites[0].descriptorCount = 1;
			descWrites[0].pBufferInfo = &uniformBufferInfo;

			descWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[1].dstSet = descriptorSets[i];
			descWrites[1].dstBinding = 1;
			descWrites[1].dstArrayElement = 0;
			descWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descWrites[1].descriptorCount = 1;
			descWrites[1].pBufferInfo = &materialBufferInfo;

			descWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[2].dstSet = descriptorSets[i];
			descWrites[2].dstBinding = 2;
			descWrites[2].dstArrayElement = 0;
			descWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descWrites[2].descriptorCount = 1;
			descWrites[2].pImageInfo = &baseColorImageInfo;

			descWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[3].dstSet = descriptorSets[i];
			descWrites[3].dstBinding = 3;
			descWrites[3].dstArrayElement = 0;
			descWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descWrites[3].descriptorCount = 1;
			descWrites[3].pImageInfo = &metallicRoughnessImageInfo;

			descWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[4].dstSet = descriptorSets[i];
			descWrites[4].dstBinding = 4;
			descWrites[4].dstArrayElement = 0;
			descWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descWrites[4].descriptorCount = 1;
			descWrites[4].pImageInfo = &normalImageInfo;

			descWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[5].dstSet = descriptorSets[i];
			descWrites[5].dstBinding = 7;
			descWrites[5].dstArrayElement = 0;
			descWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descWrites[5].descriptorCount = 1;
			descWrites[5].pImageInfo = &cubemapInfo;

			descWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[6].dstSet = descriptorSets[i];
			descWrites[6].dstBinding = 8;
			descWrites[6].dstArrayElement = 0;
			descWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descWrites[6].descriptorCount = 1;
			descWrites[6].pImageInfo = &diffuseCubemapInfo;
			
			descWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[7].dstSet = descriptorSets[i];
			descWrites[7].dstBinding = 9;
			descWrites[7].dstArrayElement = 0;
			descWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descWrites[7].descriptorCount = 1;
			descWrites[7].pImageInfo = &prefilterCubemapInfo;

			descWrites[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[8].dstSet = descriptorSets[i];
			descWrites[8].dstBinding = 10;
			descWrites[8].dstArrayElement = 0;
			descWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descWrites[8].descriptorCount = 1;
			descWrites[8].pImageInfo = &brdfInfo;

			uint32_t writeCount = 9;

			descWrites[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[9].dstSet = descriptorSets[i];
			descWrites[9].dstBinding = 5;
			descWrites[9].dstArrayElement = 0;
			descWrites[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descWrites[9].descriptorCount = 1;
			descWrites[9].pImageInfo = &occlusionImageInfo;
			if (model.mat.isOcclusionTexPresent) writeCount++;
			
			descWrites[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descWrites[10].dstSet = descriptorSets[i];
			descWrites[10].dstBinding = 6;
			descWrites[10].dstArrayElement = 0;
			descWrites[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descWrites[10].descriptorCount = 1;
			descWrites[10].pImageInfo = &emissiveImageInfo;
			if (model.mat.isEmissiveTexPresent) writeCount++;

			vkUpdateDescriptorSets(device, writeCount, descWrites.data(), 0, nullptr);
		}
	}

	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command buffers!");
		}
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		std::array<VkClearValue, 2>clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderingAttachmentInfoKHR colorAttachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = swapChainImageViews[imageIndex],
			.imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = clearValues[0]
		};

		VkRenderingAttachmentInfoKHR depthAttachment = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = depthImageView,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL_KHR,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.clearValue = clearValues[1]
		};

		VkRenderingInfoKHR renderingInfo = {
			.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
			.renderArea = {0, 0, swapChainExtent.width, swapChainExtent.height},
			.layerCount = 1,
			.colorAttachmentCount = 1,
			.pColorAttachments = &colorAttachment,
			.pDepthAttachment = &depthAttachment,
		};

		//color image transition
		VkImageMemoryBarrier colorImageStartTransitionBarrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = swapChainImages[imageIndex],
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&colorImageStartTransitionBarrier
		);
		
		//depth image transition
		VkImageMemoryBarrier depthImageStartTransitionBarrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = depthImage,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&depthImageStartTransitionBarrier
		);

		vkCmdBeginRendering(commandBuffer, &renderingInfo);


		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		//render cubemap		
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, cubemapPipeline);

		VkBuffer cubemapBuffers[] = { offscreenVertexBuffer };
		VkDeviceSize offsets[] = { 0 };

		vkCmdBindVertexBuffers(commandBuffer, 0, 1, cubemapBuffers, offsets);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, cubemapPipelineLayout, 0, 1, &cubemapDescriptorSet, 0, nullptr);
		vkCmdDraw(commandBuffer, 36, 1, 0, 0);

		//render model
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		VkBuffer vertexBuffers[] = { vertexBuffer };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

		vkCmdEndRendering(commandBuffer);

		//transitionImageLayout();

		//transition images to present_src
		VkImageMemoryBarrier colorImageEndTransitionBarrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = swapChainImages[imageIndex],
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&colorImageEndTransitionBarrier
		);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

	}

	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create sync objects for a frame!");
			}
		}
	}

	void drawFrame() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		updateUniformBuffer(currentFrame);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapchains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapchains;

		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void updateUniformBuffer(uint32_t currentImage) {
		camera.update();

		UniformBufferObject ubo{};
		ubo.model = TRS;
		ubo.view = camera.getViewMatrix();
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.01f, 100.0f);
		ubo.proj[1][1] *= -1;
		ubo.camPos = camera.position;

		memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));

		MaterialBufferObject mat{};
		mat.baseColorFactor = glm::vec4(model.mat.baseColorTex.factor[0], model.mat.baseColorTex.factor[1], model.mat.baseColorTex.factor[2], model.mat.baseColorTex.factor[3]);
		mat.metallicFactor = model.mat.metalRoughTex.factor[0];
		mat.roughnessFactor = model.mat.metalRoughTex.factor[1];

		memcpy(materialBuffersMapped[currentFrame], &mat, sizeof(mat));
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(device);
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		createSwapChain();
		createImageViews();
		createColorResources();
		createDepthResources();
		//createFramebuffers();
	}

	bool checkValidationLayerSupport() {
		uint32_t layerCount = 0;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationlayers) {
			bool layerFound = true;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}
			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (isDebugEnv) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);

		return extensions;
	}

	void cleanupSwapChain() {
		vkDestroyImageView(device, colorImageView, nullptr);
		vkDestroyImage(device, colorImage, nullptr);
		vkFreeMemory(device, colorImageMemory, nullptr);
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		for (auto swapChainImageView : swapChainImageViews) {
			vkDestroyImageView(device, swapChainImageView, nullptr);
		}
		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void cleanupPBR() {
		
		vkDestroyPipelineLayout(device, cubemapPipelineLayout, nullptr);
		vkDestroyPipeline(device, cubemapPipeline, nullptr);
		vkDestroyDescriptorSetLayout(device, cubemapDescriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(device, cubemapDescPool, nullptr);

		vkDestroyPipeline(device, brdfPipeline, nullptr);
		vkDestroyPipelineLayout(device, brdfPipelineLayout, nullptr);
		vkFreeMemory(device, brdfVertexMemory, nullptr);
		vkDestroyBuffer(device, brdfVertexBuffer, nullptr);
		vkDestroyDescriptorSetLayout(device, brdfDescriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(device, brdfDescPool, nullptr);

		vkDestroySampler(device, brdfLUTSampler, nullptr);
		vkDestroyImageView(device, brdfLUTImageView, nullptr);
		vkDestroyImage(device, brdfLUT, nullptr);
		vkFreeMemory(device, brdfLUTMemory, nullptr);

		vkDestroyPipeline(device, prefilterPipeline, nullptr);
		vkDestroyFramebuffer(device, prefilterFramebufferPerMip[4], nullptr);
		vkDestroyFramebuffer(device, prefilterFramebufferPerMip[3], nullptr);
		vkDestroyFramebuffer(device, prefilterFramebufferPerMip[2], nullptr);
		vkDestroyFramebuffer(device, prefilterFramebufferPerMip[1], nullptr);
		vkDestroyFramebuffer(device, prefilterFramebufferPerMip[0], nullptr);
		vkDestroyRenderPass(device, prefilterRenderPass, nullptr);
		vkDestroyPipelineLayout(device, prefilterPipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, prefilterDescriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(device, prefilterDescriptorPool, nullptr);

		vkDestroySampler(device, prefilterCubemapSampler, nullptr);
		vkDestroyImageView(device, prefilterCubemapImageView, nullptr);
		vkDestroyImageView(device, prefilterCubemapImageViewPerMip[0], nullptr);
		vkDestroyImageView(device, prefilterCubemapImageViewPerMip[1], nullptr);
		vkDestroyImageView(device, prefilterCubemapImageViewPerMip[2], nullptr);
		vkDestroyImageView(device, prefilterCubemapImageViewPerMip[3], nullptr);
		vkDestroyImageView(device, prefilterCubemapImageViewPerMip[4], nullptr);
		vkDestroyImage(device, prefilterCubemap, nullptr);
		vkFreeMemory(device, prefilterCubemapMemory, nullptr);
		vkDestroyFence(device, prefilterMipRenderedFence, nullptr);

		vkDestroyPipeline(device, cubemapCreatePipeline, nullptr);
		vkDestroyFramebuffer(device, cubemapCreateFramebuffer, nullptr);
		vkDestroyRenderPass(device, cubemapCreateRenderPass, nullptr);
		vkDestroyPipelineLayout(device, cubemapCreatePipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, cubemapCreateDescriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(device, cubemapCreateDescriptorPool, nullptr);

		vkDestroyBuffer(device, offscreenUniformBuffer, nullptr);
		vkFreeMemory(device, offscreenUniformBufferMemory, nullptr);

		vkDestroyBuffer(device, offscreenVertexBuffer, nullptr);
		vkFreeMemory(device, offscreenVertexBufferMemory, nullptr);

		vkDestroySampler(device, cubemapSampler, nullptr);
		vkDestroyImageView(device, cubemapImageView, nullptr);
		vkDestroyImage(device, cubemap, nullptr);
		vkFreeMemory(device, cubemapMemory, nullptr);

		vkDestroyPipeline(device, diffuseCubemapPipeline, nullptr);
		vkDestroyFramebuffer(device, diffuseFramebuffer, nullptr);
		vkDestroyRenderPass(device, diffuseRenderPass, nullptr);
		vkDestroyPipelineLayout(device, diffuseCubemapPipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, diffuseDescriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(device, diffuseDescriptorPool, nullptr);

		vkDestroySampler(device, diffuseCubemapSampler, nullptr);
		vkDestroyImageView(device, diffuseCubemapImageView, nullptr);
		vkDestroyImage(device, diffuseCubemap, nullptr);
		vkFreeMemory(device, diffuseCubemapMemory, nullptr);

		vkDestroyImageView(device, offscreenDepthImageView, nullptr);
		vkDestroyImage(device, offscreenDepthImage, nullptr);
		vkFreeMemory(device, offscreenDepthImageMemory, nullptr);

		vkDestroySampler(device, hdriSampler, nullptr);
		vkDestroyImageView(device, hdriImageView, nullptr);
		vkDestroyImage(device, hdriImage, nullptr);
		vkFreeMemory(device, hdriImageMemory, nullptr);
	}

	void cleanup() {
		cleanupSwapChain();

		vkDestroySampler(device, occlusionSampler, nullptr);
		vkDestroyImageView(device, occlusionImageView, nullptr);
		vkDestroyImage(device, occlusionImage, nullptr);
		vkFreeMemory(device, occlusionMemory, nullptr);

		vkDestroySampler(device, emissiveSampler, nullptr);
		vkDestroyImageView(device, emissiveImageView, nullptr);
		vkDestroyImage(device, emissiveImage, nullptr);
		vkFreeMemory(device, emissiveMemory, nullptr);

		vkDestroySampler(device, normalSampler, nullptr);
		vkDestroyImageView(device, normalImageView, nullptr);
		vkDestroyImage(device, normalImage, nullptr);
		vkFreeMemory(device, normalMemory, nullptr);

		vkDestroySampler(device, metallicRoughnessSampler, nullptr);
		vkDestroyImageView(device, metallicRoughnessImageView, nullptr);
		vkDestroyImage(device, metallicRoughnessImage, nullptr);
		vkFreeMemory(device, metallicRoughnessMemory, nullptr);

		vkDestroySampler(device, baseColorSampler, nullptr);
		vkDestroyImageView(device, baseColorImageView, nullptr);
		vkDestroyImage(device, baseColorImage, nullptr);
		vkFreeMemory(device, baseColorMemory, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer(device, materialBuffers[i], nullptr);
			vkFreeMemory(device, materialBuffersMemory[i], nullptr);

			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);

		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);

		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		cleanupPBR();

		vkDestroyDevice(device, nullptr);

		if (isDebugEnv) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData) {

		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl << std::endl;

		return VK_FALSE;
	}

	VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
		const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
		const VkAllocationCallbacks* pAllocator,
		VkDebugUtilsMessengerEXT* pDebugMessenger) {
		auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
		if (func != nullptr) {
			return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
		}
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}

	void DestroyDebugUtilsMessengerEXT(VkInstance instance,
		VkDebugUtilsMessengerEXT debugMessenger,
		const VkAllocationCallbacks* pAllocator) {

		auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
		if (func != nullptr) {
			return func(instance, debugMessenger, pAllocator);
		}
		throw std::runtime_error("failed to destroy degub utils messenger!");
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity =
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType =
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;

	}

	void setupDebugMessenger() {
		if (!isDebugEnv) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}
};