#version 450

layout(location = 0) in vec3 texCoords;

layout(binding = 1) uniform samplerCube cubemap;

layout(location = 0) out vec4 outColor;

void main(){
	vec3 color = textureLod(cubemap, texCoords, 1.2).rgb;
	/*
	color = color / (color + vec3(1.0));
	color = pow(color, vec3(1.0/2.2));
	*/
	outColor = vec4(color, 1.0);
}