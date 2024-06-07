#version 450 

layout(set = 0, binding = 0) uniform UniformBufferObject{
	mat4 model;
	mat4 view;
	mat4 proj;
	vec3 camPos;
} ubo;

layout(location = 0) out vec2 uv;

void main(){
	float x = float((gl_VertexIndex & 1) << 2);
	float y = float((gl_VertexIndex & 2) << 1);
	uv = vec2(x, y) * 0.5;
	gl_Position = vec4(x - 1.0, y - 1.0, 0, 1);
}