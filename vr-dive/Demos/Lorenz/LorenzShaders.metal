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

struct LorenzParticleState {
  float4 positionAndScale;
  float4 seedAndPhase;
};

struct LorenzUniforms {
  float deltaTime;
  float globalTime;
  uint particleCount;
  float sigma;
  float beta;
  float rho;
  float damping;
  float worldScale;
  float resetRadius;
  float noiseAmplitude;
  float padding;
};

struct LorenzVertexOut {
  float4 position [[position]];
  float3 normal [[flat]];
  float3 worldPos;
  float objectScale;
  // uint layer [[render_target_array_index]]; // Removed
};

vertex LorenzVertexOut lorenzVertexShader(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  const device LorenzParticleState *states [[buffer(1)]],
  constant SceneUniforms &uniforms [[buffer(2)]],
  constant float4x4 *viewProjectionMatrices [[buffer(3)]],
  uint vertexID [[vertex_id]],
  uint instanceID [[instance_id]]) {
  LorenzVertexOut out;
  uint layers = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1);
  LorenzParticleState state = states[instanceID];
  MeshVertex vtx = vertices[vertexID];

  float phase = state.seedAndPhase.w;
  float c = cos(phase);
  float s = sin(phase);

  float3 rotatedMeshPos =
      float3(vtx.position.x * c - vtx.position.z * s, vtx.position.y,
             vtx.position.x * s + vtx.position.z * c);
  float3 rotatedMeshNormal =
      float3(vtx.normal.x * c - vtx.normal.z * s, vtx.normal.y,
             vtx.normal.x * s + vtx.normal.z * c);

  float3 simPos = state.positionAndScale.xyz;
  float3 reorientedPos = float3(simPos.x, simPos.z, simPos.y);
  float3 offset = float3(0.0, -0.2, -1.5);
  float3 finalCenterPos = reorientedPos + offset;

  float3 position = finalCenterPos + rotatedMeshPos * state.positionAndScale.w;
  float4 worldPos = float4(position, 1.0);
  float4x4 viewProjection = viewProjectionMatrices[viewIndex];

  out.position = viewProjection * worldPos;
  out.normal = normalize(
      float3(rotatedMeshNormal.x, rotatedMeshNormal.z, rotatedMeshNormal.y));
  out.worldPos = position;
  out.objectScale = state.positionAndScale.w;
  // out.layer = viewIndex; // Removed
  return out;
}

fragment float4 lorenzFragmentShader(LorenzVertexOut in [[stage_in]],
                                     constant SceneUniforms &uniforms
                                     [[buffer(0)]]) {
  float3 normal = normalize(in.normal);
  float3 lightDir = normalize(float3(-0.2, 0.9, -0.3));
  float3 viewDir = normalize(-in.worldPos);
  float ndotl = max(dot(normal, lightDir), 0.0);
  float3 halfVec = normalize(lightDir + viewDir);
  float spec = pow(max(dot(normal, halfVec), 0.0), 48.0);
  float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 5.0);
  float scaleFactor = saturate((in.objectScale - 0.04) * 8.0);

  float3 baseColor = abs(normal) * 0.8 + 0.2;
  baseColor = mix(baseColor, float3(0.9, 0.95, 1.0), scaleFactor * 0.5);

  float glow = scaleFactor * 0.3;
  float3 metallic = baseColor * (0.25 + ndotl * 0.75) + glow;
  float3 color = metallic + spec * float3(0.9, 0.95, 1.0) +
                 fresnel * float3(0.25, 0.55, 0.95);
  return float4(color, 1.0);
}

inline float3 lorenzDerivative(float3 p, float sigma, float beta, float rho) {
  float3 d;
  d.x = sigma * (p.y - p.x);
  d.y = p.x * (rho - p.z) - p.y;
  d.z = p.x * p.y - beta * p.z;
  return d;
}

kernel void integrateLorenzAttractor(device LorenzParticleState *states
                                     [[buffer(0)]],
                                     constant LorenzUniforms &uniforms
                                     [[buffer(1)]],
                                     uint id [[thread_position_in_grid]]) {
  if (id >= uniforms.particleCount) {
    return;
  }

  LorenzParticleState state = states[id];
  float3 worldPosition = state.positionAndScale.xyz;
  float3 position = worldPosition / max(uniforms.worldScale, 1e-4f);
  float dt = clamp(uniforms.deltaTime, 1.0f / 10000.0f, 1.0f / 30.0f);

  float3 k1 =
      lorenzDerivative(position, uniforms.sigma, uniforms.beta, uniforms.rho);
  float3 k2 = lorenzDerivative(position + 0.5f * dt * k1, uniforms.sigma,
                               uniforms.beta, uniforms.rho);
  float3 k3 = lorenzDerivative(position + 0.5f * dt * k2, uniforms.sigma,
                               uniforms.beta, uniforms.rho);
  float3 k4 = lorenzDerivative(position + dt * k3, uniforms.sigma,
                               uniforms.beta, uniforms.rho);
  float3 next = position + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);

  float3 seeds = state.seedAndPhase.xyz;
  next *= uniforms.damping;

  bool invalid = !(isfinite(next.x) && isfinite(next.y) && isfinite(next.z));
  float len = length(next);
  if (invalid || len > uniforms.resetRadius) {
    float sign = fmod(seeds.x + seeds.y, 2.0f) > 1.0f ? 1.0f : -1.0f;
    next = float3(sign * (8.0f + sin(seeds.x) * 0.5f),
                  8.5f + cos(seeds.y) * 1.5f, 27.0f + sin(seeds.z) * 1.5f);
  }

  float3 worldNext = next * uniforms.worldScale;
  float3 tangent = next - position;
  if (length(tangent) < 1e-4f || !isfinite(tangent.x)) {
    tangent = k1;
  }
  float rotation = atan2(tangent.z, tangent.x);

  state.positionAndScale.xyz = worldNext;
  state.seedAndPhase.w = rotation;
  states[id] = state;
}
