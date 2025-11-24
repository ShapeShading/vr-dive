#include <metal_stdlib>
using namespace metal;

struct SceneUniforms {
  float time;
  uint layerCount;
  float2 padding;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct PongWarInstanceState {
  float4 positionAndScale;
  float4 color;
  float4 motion;
};

struct PongWarUniforms {
  float pulseAmplitude;
  float pulseSpeed;
  float cubeRotationSpeed;
  float sphereBobSpeed;
  float sphereBobAmount;
  float sphereGlow;
  float noiseAmount;
  float padding;
};

struct PongWarVertexOut {
  float4 position [[position]];
  float3 normal;
  float3 worldPos;
  float4 color;
  float type;
};

float3x3 rotationY(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return float3x3(
    float3(c, 0.0, -s),
    float3(0.0, 1.0, 0.0),
    float3(s, 0.0, c)
  );
}

vertex PongWarVertexOut pongWarVertexShader(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    const device PongWarInstanceState *instances [[buffer(1)]],
    constant SceneUniforms &scene [[buffer(2)]],
    constant float4x4 *viewProjectionMatrices [[buffer(3)]],
    constant PongWarUniforms &uniforms [[buffer(4)]],
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]]) {
  PongWarVertexOut out;
  PongWarInstanceState state = instances[instanceID];
  float3 meshPos = vertices[vertexID].position;
  float3 meshNormal = vertices[vertexID].normal;
  float time = scene.time;
  float type = state.motion.z;

  float wobble = sin(time * uniforms.pulseSpeed + state.motion.x) * state.motion.w;
  float scale = state.positionAndScale.w * (type < 0.5 ? (1.0 + uniforms.pulseAmplitude * wobble) : 1.0);
  float3 worldPos = state.positionAndScale.xyz;
  float3 rotatedPos = meshPos;
  float3 rotatedNormal = meshNormal;

  if (type < 0.5) {
    float rotation = time * uniforms.cubeRotationSpeed + state.motion.y;
    float3x3 rotMatrix = rotationY(rotation);
    rotatedPos = rotMatrix * meshPos;
    rotatedNormal = rotMatrix * meshNormal;
  } else {
    float bob = sin(time * uniforms.sphereBobSpeed + state.motion.y) * uniforms.sphereBobAmount;
    worldPos.y += bob;
    rotatedPos += normalize(meshPos) * uniforms.noiseAmount * wobble;
  }

  float3 finalPos = worldPos + rotatedPos * scale;
  float4 worldPosition = float4(finalPos, 1.0);

  uint layers = max(scene.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1);
  float4x4 viewProj = viewProjectionMatrices[viewIndex];

  out.position = viewProj * worldPosition;
  out.normal = normalize(rotatedNormal);
  out.worldPos = finalPos;
  out.color = state.color;
  out.type = type;
  return out;
}

fragment float4 pongWarFragmentShader(
    PongWarVertexOut in [[stage_in]],
    constant SceneUniforms &scene [[buffer(0)]],
    constant PongWarUniforms &uniforms [[buffer(1)]]) {
  float3 normal = normalize(in.normal);
  float3 lightDir = normalize(float3(-0.3, 0.9, -0.2));
  float3 viewDir = normalize(-in.worldPos);
  float ndotl = max(dot(normal, lightDir), 0.0);
  float3 halfVector = normalize(lightDir + viewDir);
  float typeFactor = clamp(in.type, 0.0, 1.0);
  float specPower = 24.0 + (42.0 - 24.0) * typeFactor;
  float spec = pow(max(dot(normal, halfVector), 0.0), specPower);
  float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);

  float3 baseColor = in.color.rgb;
  float glow = in.type < 0.5 ? 0.1 : uniforms.sphereGlow;
  float3 litColor = baseColor * (0.25 + ndotl * 0.75);
  float3 color = litColor + spec * float3(1.0) + fresnel * float3(0.5, 0.7, 1.0) * glow;

  return float4(color, 1.0);
}
