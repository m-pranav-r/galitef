#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;
layout(location = 3) in vec3 worldPos;
layout(location = 4) in mat4 modelMatrix;

layout(binding = 2) uniform sampler2D baseColorTex;
layout(binding = 3) uniform sampler2D metallicRoughness;
layout(binding = 4) uniform sampler2D normals;
layout(binding = 5) uniform sampler2D occlusion;
layout(binding = 6) uniform sampler2D emissive;

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

void main(){
	vec3 baseColor = pow(texture(baseColorTex, texCoord).rgb, vec3(2.2));
	vec4 mrSample = texture(metallicRoughness, texCoord);
	float roughness = mrSample.g;
	float metallic = mrSample.b;
	float ao = texture(occlusion, texCoord).r;

	vec3 F0 = mix(vec3(0.04), baseColor, metallic);

	//vec3 n = normalize(texture(normals, texCoord).rgb * 2.0 - 1.0);
	vec3 n = normalize(normal);
	vec3 viewVector = vec4(modelMatrix * vec4(2.0, 2.0, 2.0, 1.0)).xyz;
	vec3 v = normalize(viewVector - worldPos);

	vec3 Lo = vec3(0.0);
	vec3 lights[4] = vec3[4](
							vec3(-10.0,  10.0,  10.0),  
							vec3( 10.0,  10.0,  10.0), 
							vec3(-10.0, -10.0,  10.0),
							vec3( 10.0, -10.0,  10.0)
					);
	//per light shit
	for(int i = 0; i < 3; i++){
		//vec3 lightVector = -vec4(modelMatrix * vec4(lights[i], 1.0)).xyz;
		vec3 l = normalize(lights[i] - worldPos);
		vec3 h = normalize(l + v);

		float distance = length(lights[i] - worldPos);
		float attenuation = 1.0 / (distance * distance);
		vec3 radiance = vec3(300.0) * attenuation;

		//calculate brdf terms
		float D = D_GGX(n, h, roughness);
		float G = G_Smith(n, v, l, roughness);
		vec3 F = F_Schlick(clamp(dot(h, v), 0.0, 1.0), F0);

		vec3 numer = D * G * F;
		float denom = 4 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0) + 0.0001;
		vec3 specular = numer / denom;

		vec3 kS = F;
		vec3 kD = vec3(1.0) - kS;
		kD *= 1.0 - metallic;

		Lo += ((kD * baseColor / M_PI) + specular) * radiance * max(dot(n, l), 0.0);
	}
	
	vec3 ambient = vec3(0.03) * baseColor * ao;
	vec3 color = ambient + Lo;

	color = color / (color + vec3(1.0));
	color = pow(color, vec3(1.0/2.2));

	outColor = vec4(color, 1.0);

}