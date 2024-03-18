#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec3 worldPos;

layout(binding = 2) uniform sampler2D baseColor;
layout(binding = 3) uniform sampler2D metallicRoughness;
layout(binding = 4) uniform sampler2D normals;
layout(binding = 5) uniform sampler2D occlusion;
layout(binding = 6) uniform sampler2D emissive;

layout (set = 0, binding = 1) uniform Material {
	uniform float roughnessFactor;
	uniform float metallicFactor;
	uniform vec4 baseColorFactor;
	//uniform float alphaRoughness;
} mat;

layout(location = 0) out vec4 outColor;

const float M_PI = 3.141592653589793;

void main(){
	float perceptualRoughness;
	float metallic;
	vec3 diffuseColor;
	vec4 baseColor;
	vec3 f0 = vec3(0.0);

	perceptualRoughness = mat.roughnessFactor;
	metallic = mat.metallicFactor;

	vec4 mrSample = texture(metallicRoughness, texCoord);
	perceptualRoughness *= mrSample.g;
	metallic *= mrSample.b;
	baseColor = mat.baseColorFactor;

	diffuseColor = baseColor.rgb * (vec3(1.0) - f0);
	diffuseColor *= 1.0 - metallic;

	float alphaRoughness = perceptualRoughness * perceptualRoughness;

	vec3 specularColor = mix(f0, baseColor.rgb, metallic);

	float reflectance0 = max(max(specularColor.r, specularColor.g), specularColor.b);

	float reflectance90 = clamp(reflectance0 * 25.0, 0.0, 1.0);
	vec3 specularEnvironmentR0 = specularColor.rbg;
	vec3 specularEnvironmantR90 = vec3(1.0, 1.0, 1.0) * reflectance90;

	vec3 n = normal;
	vec3 v = normalize(vec3(0.0) - worldPos);
	vec3 l = normalize(vec3(0.0, 1.0, 0.0));							//light direction!!
	vec3 h = normalize(l+v);
	vec3 reflection = -normalize(reflect(v, n));
	reflection.y *= -1.0f;

	float NdotL = clamp(dot(n, l), 0.001, 1.0);
	float NdotV = clamp(abs(dot(n, v)), 0.001, 1.0);
	float NdotH = clamp(dot(n, h), 0.0, 1.0);
	float LdotH = clamp(dot(l, h), 0.0, 1.0);
	float VdotH = clamp(dot(v, h), 0.0, 1.0);

	vec3 F = vec3(reflectance0 + (reflectance90 - reflectance0)) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);

	float attenuationL = 2.0 * NdotL / (NdotL + sqrt(alphaRoughness * alphaRoughness + (1.0 - alphaRoughness * alphaRoughness) * (NdotL * NdotL)));
	float attenuationV = 2.0 * NdotV / (NdotV + sqrt(alphaRoughness * alphaRoughness + (1.0 - alphaRoughness * alphaRoughness) * (NdotV * NdotV)));

	float G = attenuationL * attenuationV;

	float roughnessSq = alphaRoughness * alphaRoughness;
	float f = (NdotH * roughnessSq - NdotH) * NdotH + 1.0;
	float D = roughnessSq / (M_PI * f * f);

	const vec3 u_LightColor = vec3(1.0);

	vec3 diffuseContrib = (1.0 - F) * diffuseColor / M_PI;
	vec3 specContrib = F * G * D / (4.0 * NdotL * NdotV);

	vec3 color = NdotL * u_LightColor * (diffuseColor + specContrib);

	//ADD OCCLUSIVE SUPPORT LATER

	//ADD EMISSIVE SUPPORT LATER

	outColor = vec4(color, 1.0);
}