#version 450
#define OCCLUSION
#define EMISSIVE

//#extension GL_EXT_debug_printf : enable

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec3 worldPos;
layout(location = 4) in vec3 camPos;

layout(binding = 2) uniform sampler2D baseColorTex;
layout(binding = 3) uniform sampler2D metallicRoughness;
layout(binding = 4) uniform sampler2D normals;
#ifdef OCCLUSION
layout(binding = 5) uniform sampler2D occlusion;
#endif
#ifdef EMISSIVE
layout(binding = 6) uniform sampler2D emissive;
#endif
layout(binding = 7) uniform samplerCube cubemap;
layout(binding = 8) uniform samplerCube irradianceCubemap;
layout(binding = 9) uniform samplerCube prefilterCubemap;
layout(binding = 10) uniform sampler2D brdfLUT;

layout (set = 0, binding = 1) uniform Material {
	uniform float roughnessFactor;
	uniform float metallicFactor;
	uniform vec4 baseColorFactor;
} mat;

layout(location = 0) out vec4 outColor;

const float M_PI = 3.14159265359;

float D_GGX(vec3 n, vec3 h, float roughness){
	float alpha = roughness * roughness;
	float alpha_squared = alpha * alpha;
	float NdotH = max(dot(n, h), 0.0);
	float NdotH2 = NdotH * NdotH;

	float num = alpha_squared;
	float denom = (NdotH2 * (alpha_squared - 1.0) + 1.0);
	denom = M_PI * denom * denom;

	return num / denom;
}

float G_SchlickGGX(float NdotV, float roughness){
	float r = roughness + 1.0;
	float k = (r * r) / 8.0;

	float num = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return num / denom;
}

float G_Smith(vec3 n, vec3 v, vec3 l, float roughness){
	float NdotV = max(dot(n, v), 0.0);
	float NdotL = max(dot(n, l), 0.0);
	float ggx2 = G_SchlickGGX(NdotV, roughness);
	float ggx1 = G_SchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

vec3 F_Schlick(float cosTheta, vec3 F0){
	float term = clamp(1 - cosTheta, 0.0, 1.0);
	float term_squared = term * term;
	return F0 + (1 - F0) * term_squared * term_squared * term;
}

vec3 F_SchlickLagarde(float cosTheta, vec3 F0, float roughness){
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main_alt(){
	outColor = texture(occlusion, texCoord);
}

void main(){
	vec3 baseColor = pow(texture(baseColorTex, texCoord).rgb, vec3(2.2));
	vec4 mrSample = pow(texture(metallicRoughness, texCoord), vec4(1/2.2));
	float roughness = mat.roughnessFactor * mrSample.g;
	float metallic = mat.metallicFactor * mrSample.b;
#ifdef OCCLUSION
	float ao = texture(occlusion, texCoord).r;
#endif
#ifdef EMISSIVE
	vec3 emissiveColor = pow(texture(emissive, texCoord).rgb, vec3(2.2));
#endif

	vec3 F0 = mix(vec3(0.04), baseColor, metallic);
	
	vec2 uv_dx = dFdx(texCoord);
	vec2 uv_dy = dFdy(texCoord);

	vec3 T = (uv_dy.t * dFdx(pos) - uv_dx.t * dFdy(pos)) / (uv_dx.s * uv_dy.t - uv_dy.s * uv_dx.t);
	vec3 N = normalize(normal);
	T = normalize(T - dot(T, N) * N);
	vec3 B = cross(N, T);
	mat3 TBN_frag = mat3(T, B, N);

	vec3 n = pow(texture(normals, texCoord).rgb, vec3(2.2));
	n = normalize(TBN_frag * n);

	
	vec3 v = normalize(camPos - worldPos);

	vec3 Lo = vec3(0.0);

	vec3 F = F_SchlickLagarde(max(dot(n, v), 0.0), F0, roughness);

	vec3 kS = F;
	vec3 kD = 1.0 - kS;
	kD *= 1.0 - metallic;

	vec3 irradiance = texture(irradianceCubemap, n).rgb;
	vec3 diffuse = irradiance * baseColor;

	vec3 r = reflect(-v, n);
	const float MAX_REFLECTION_LOD = 4.0;
	vec3 prefilteredColor = textureLod(prefilterCubemap, r, roughness * MAX_REFLECTION_LOD).rgb;
	vec2 brdf = texture(brdfLUT, vec2(max(dot(n, v), 0.0), roughness)).rg;
	vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);
#ifdef OCCLUSION
	vec3 ambient = (kD * diffuse + specular) * ao;
#else
	vec3 ambient = (kD * diffuse + specular);
#endif
#ifdef EMISSIVE
	vec3 color = ambient + Lo + emissiveColor;
#else
	vec3 color = ambient + Lo;
#endif
	
	/*
	color = color / (color + vec3(1.0));
	color = pow(color, vec3(1.0/2.2));
	*/

	outColor = vec4(color, 1.0);
}