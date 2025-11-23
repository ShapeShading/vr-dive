#include <metal_stdlib>
using namespace metal;

struct BackgroundUniforms {
  float time;
  float intensity;
  float2 padding;
  float4x4 viewToWorldLeft;
  float4x4 viewToWorldRight;
};

struct SceneUniforms {
  float4x4 viewProjectionMatrixLeft;
  float4x4 viewProjectionMatrixRight;
  float time;
  uint layerCount;
  float2 padding;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct FourWingParticleState {
  float4 positionAndScale;
  float4 seedAndPhase;
};

struct FourWingUniforms {
  float deltaTime;
  float globalTime;
  uint particleCount;
  float a;
  float b;
  float c;
  float damping;
  float worldScale;
  float resetRadius;
  float noiseAmplitude;
  float padding;
};

struct BackgroundVertexOut {
  float4 position [[position]];
  float4 clipPosition;
  float2 uv;
  uint layer [[render_target_array_index]];
};

struct FourWingVertexOut {
  float4 position [[position]];
  float3 normal [[flat]];  // Use flat interpolation for uniform color per triangle
  float3 worldPos;
  float objectScale;
  uint layer [[render_target_array_index]];
};

static inline BackgroundVertexOut
makeBackgroundVertex(uint vertexID, uint instanceID,
                     constant BackgroundUniforms &uniforms) {
  BackgroundVertexOut out;
  float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
  out.position = float4(positions[vertexID], 0.0, 1.0);
  out.clipPosition = out.position;
  out.uv = positions[vertexID] * 0.5 + 0.5;
  out.layer = instanceID;
  return out;
}

vertex BackgroundVertexOut fourWingBackgroundVertexShader(
    uint vertexID [[vertex_id]], uint instanceID [[instance_id]],
    constant BackgroundUniforms &uniforms [[buffer(0)]]) {
  return makeBackgroundVertex(vertexID, instanceID, uniforms);
}

fragment float4 fourWingBackgroundFragmentShader(
    BackgroundVertexOut in [[stage_in]],
    constant BackgroundUniforms &uniforms [[buffer(0)]]) {
  return float4(0.005, 0.008, 0.012, 1.0);
}

vertex FourWingVertexOut fourWingVertexShader(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    const device FourWingParticleState *states [[buffer(1)]],
    constant SceneUniforms &uniforms [[buffer(2)]], uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]]) {
  FourWingVertexOut out;
  uint layers = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1);
  FourWingParticleState state = states[instanceID];
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
  float4x4 viewProjection = viewIndex == 0 ? uniforms.viewProjectionMatrixLeft
                                           : uniforms.viewProjectionMatrixRight;

  out.position = viewProjection * worldPos;
  out.normal = normalize(
      float3(rotatedMeshNormal.x, rotatedMeshNormal.z, rotatedMeshNormal.y));
  out.worldPos = position;
  out.objectScale = state.positionAndScale.w;
  out.layer = viewIndex;
  return out;
}

fragment float4 fourWingFragmentShader(FourWingVertexOut in [[stage_in]],
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

  // Color based on normal direction
  float3 baseColor = abs(normal) * 0.8 + 0.2;  // Map normal xyz to rgb
  baseColor = mix(baseColor, float3(0.85, 0.92, 1.0), scaleFactor * 0.5);

  float glow = scaleFactor * 0.3;
  float3 metallic = baseColor * (0.25 + ndotl * 0.75) + glow;
  float3 color = metallic + spec * float3(0.9, 0.95, 1.0) +
                 fresnel * float3(0.35, 0.45, 0.95);
  return float4(color, 1.0);
}

inline float3 fourWingDerivative(float3 p, float a, float b, float c) {
  float3 d;
  d.x = a * p.x + p.y * p.z;
  d.y = b * p.x + c * p.y - p.x * p.z;
  d.z = -p.z - p.x * p.y;
  return d;
}

kernel void integrateFourWingAttractor(device FourWingParticleState *states
                                       [[buffer(0)]],
                                       constant FourWingUniforms &uniforms
                                       [[buffer(1)]],
                                       uint id [[thread_position_in_grid]]) {
  if (id >= uniforms.particleCount) {
    return;
  }

  FourWingParticleState state = states[id];
  float3 worldPosition = state.positionAndScale.xyz;
  float3 position = worldPosition / max(uniforms.worldScale, 1e-4f);
  float dt = clamp(uniforms.deltaTime, 1.0f / 10000.0f, 1.0f / 30.0f);

  float3 k1 = fourWingDerivative(position, uniforms.a, uniforms.b, uniforms.c);
  float3 k2 = fourWingDerivative(position + 0.5f * dt * k1, uniforms.a,
                                 uniforms.b, uniforms.c);
  float3 k3 = fourWingDerivative(position + 0.5f * dt * k2, uniforms.a,
                                 uniforms.b, uniforms.c);
  float3 k4 = fourWingDerivative(position + dt * k3, uniforms.a, uniforms.b,
                                 uniforms.c);
  float3 next = position + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);

  float3 seeds = state.seedAndPhase.xyz;
  // Noise removed - particles follow pure attractor trajectory
  // float noisePhase = uniforms.globalTime * 0.25f;
  // float3 noise = float3(sin(seeds.x * 0.017f + noisePhase),
  //                       cos(seeds.y * 0.013f + noisePhase * 1.2f),
  //                       sin(seeds.z * 0.019f - noisePhase * 0.8f)) *
  //                uniforms.noiseAmplitude;
  // next += noise * dt;
  next *= uniforms.damping;

  bool invalid = !(isfinite(next.x) && isfinite(next.y) && isfinite(next.z));
  float len = length(next);
  if (invalid || len > uniforms.resetRadius) {
    float sign = fmod(seeds.x + seeds.y, 2.0f) > 1.0f ? 1.0f : -1.0f;
    next = float3(sign * 0.1f, 0.1f, 0.1f);
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
