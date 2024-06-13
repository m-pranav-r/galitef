#version 450

layout(location = 0) in vec3 aPos;

layout(set = 0, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec3 camPos;
} ubo;

layout(location = 0) out vec3 localPos;

void main(){
	localPos = aPos;
	gl_Position = ubo.proj * ubo.view * vec4(localPos, 1.0);
}