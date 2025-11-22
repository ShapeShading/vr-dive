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

struct SimulationUniforms {
  float deltaTime;
  float globalTime;
  uint objectCount;
  uint padding;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct ObjectState {
  float4 positionAndType;
  float4 motionAndPhase;
  float4 scaleAndPadding;
  float4 homeAndJitter;
};

struct BackgroundVertexOut {
  float4 position [[position]];
  float4 clipPosition;
  float2 uv;
  uint layer [[render_target_array_index]];
};

struct ObjectVertexOut {
  float4 position [[position]];
  float3 normal;
  float3 worldPos;
  uint layer [[render_target_array_index]];
  float objectType;
};

vertex BackgroundVertexOut backgroundVertexShader(
    uint vertexID [[vertex_id]], uint instanceID [[instance_id]],
    constant BackgroundUniforms &uniforms [[buffer(0)]]) {
  BackgroundVertexOut out;
  float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
  out.position = float4(positions[vertexID], 0.0, 1.0);
  out.clipPosition = out.position;
  out.uv = positions[vertexID] * 0.5 + 0.5;
  out.layer = instanceID;
  return out;
}

float3 shallowSeaGradient(float3 worldDir, float t, float intensity) {
  float upFactor = saturate(worldDir.y * 0.5 + 0.5);
  float lateral = saturate(length(worldDir.xz));
  float3 surfaceColor = float3(0.1, 0.32, 0.45);
  float3 depthColor = float3(0.005, 0.04, 0.08);
  float wave = 0.03 * sin(t * 0.35 + worldDir.x * 5.0 + worldDir.z * 3.0);
  float mixFactor = clamp(upFactor + wave, 0.0, 1.0);
  float glow = smoothstep(0.2, 0.7, upFactor) * 0.12;
  float3 tint = float3(0.05, -0.02, -0.04) * lateral;
  float3 color = mix(depthColor, surfaceColor, mixFactor) + glow;
  return (color + tint) * intensity;
}

fragment float4 backgroundFragmentShader(BackgroundVertexOut in [[stage_in]],
                                         constant BackgroundUniforms &uniforms
                                         [[buffer(0)]]) {
  float4x4 viewToWorld =
      in.layer == 0 ? uniforms.viewToWorldLeft : uniforms.viewToWorldRight;
  float2 ndc = in.clipPosition.xy / in.clipPosition.w;
  float4 eyeRay = float4(ndc, -1.0, 0.0);
  float3 worldDir = normalize((viewToWorld * eyeRay).xyz);
  float3 color =
      shallowSeaGradient(worldDir, uniforms.time, uniforms.intensity);
  return float4(color, 1.0);
}

vertex ObjectVertexOut objectVertexShader(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    const device ObjectState *states [[buffer(1)]],
    constant SceneUniforms &uniforms [[buffer(2)]], uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]]) {
  ObjectVertexOut out;
  uint layers = max(uniforms.layerCount, 1u);
  uint objectIndex = instanceID;
  uint viewIndex = min((uint)amplificationID, layers - 1);
  MeshVertex vtx = vertices[vertexID];
  ObjectState state = states[objectIndex];

  float3 position =
      state.positionAndType.xyz + vtx.position * state.scaleAndPadding.xyz;
  float4 worldPos = float4(position, 1.0);
  float4x4 viewProjection = viewIndex == 0 ? uniforms.viewProjectionMatrixLeft
                                           : uniforms.viewProjectionMatrixRight;
  out.position = viewProjection * worldPos;
  out.normal = vtx.normal;
  out.worldPos = position;
  out.objectType = state.positionAndType.w;
  out.layer = viewIndex;
  return out;
}

float3 objectColor(float type, float3 pos, float time) {
  if (type > 0.5) {
    float hue = 0.52 + 0.05 * sin(time + pos.x);
    return mix(float3(0.1, 0.35, 0.45), float3(0.18, 0.5, 0.62),
               0.5 + 0.5 * sin(hue));
  }
  float variation =
      0.2 * sin(pos.x * 2.0 + time) + 0.1 * cos(pos.y * 3.0 - time * 0.5);
  return float3(0.05, 0.2, 0.28) + variation;
}

fragment float4 objectFragmentShader(ObjectVertexOut in [[stage_in]],
                                     constant SceneUniforms &uniforms
                                     [[buffer(0)]]) {
  float3 normal = normalize(in.normal);
  float3 lightDir = normalize(float3(0.4, 0.9, -0.2));
  float ndotl = abs(dot(normal, lightDir));
  float rim = pow(1.0 - abs(dot(normal, float3(0, 0, 1))), 2.0);
  float3 baseColor = objectColor(in.objectType, in.worldPos, uniforms.time);
  float3 color = baseColor * (0.35 + ndotl * 0.65) + rim * 0.1;
  return float4(color, 0.95);
}

kernel void simulateObjects(device ObjectState *states [[buffer(0)]],
                            constant SimulationUniforms &uniforms [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
  if (id >= uniforms.objectCount) {
    return;
  }

  ObjectState state = states[id];
  float dt = uniforms.deltaTime;
  float scaledDt = dt * 0.1125;
  float type = state.positionAndType.w;

  float3 amplitude = state.motionAndPhase.xyz;
  float phase = state.motionAndPhase.w + scaledDt * (0.35 + 0.35 * type);
  float3 home = state.homeAndJitter.xyz;
  float jitterRadius = state.homeAndJitter.w;

  float3 swirl = float3(sin(phase * 1.3 + home.x), cos(phase * 0.8 + home.y),
                        sin(phase * 1.15 + home.z)) *
                 amplitude;

  float3 current = float3(
      sin(uniforms.globalTime * 0.18 + home.y) * jitterRadius * 0.35, 0.0,
      cos(uniforms.globalTime * 0.15 + home.x) * jitterRadius * 0.4);

  float lift = mix(0.02, 0.09, type);
  float bob = sin(phase * 0.6 + home.z) * jitterRadius * 0.8;

  float3 position = home;
  position.x += swirl.x + current.x;
  position.y += swirl.y + current.y + bob + lift;
  position.z += swirl.z + current.z;

  float3 displacement = position - home;
  float maxRadius =
      jitterRadius + max(max(amplitude.x, amplitude.y), amplitude.z);
  float lenSq = dot(displacement, displacement);
  if (lenSq > maxRadius * maxRadius) {
    float len = sqrt(lenSq + 1e-5);
    displacement *= (maxRadius / len);
    position = home + displacement;
  }

  state.positionAndType = float4(position, type);
  state.motionAndPhase.w = phase;
  states[id] = state;
}
