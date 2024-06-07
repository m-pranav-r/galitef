#version 450

layout(location = 0) in vec2 uv;

layout(binding = 1) uniform sampler2D hdriTex;

layout(location = 0) out vec4 outColor;

const vec2 invAtan = vec2(0.1591, 0.3183);

vec2 SampleSphericalMap(vec3 v){
	vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
	uv *= invAtan;
	uv += 0.5;
	return uv;
}

void main(){
	vec3 color = texture(hdriTex, uv).rgb;

	outColor = vec4(color, 1.0);
}