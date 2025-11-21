#include <metal_stdlib>
using namespace metal;

struct BackgroundUniforms {
  float time;
  float intensity;
  float2 padding;
};

struct SceneUniforms {
  float4x4 viewProjectionMatrix;
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
  float4 velocityAndPhase;
  float4 scaleAndPadding;
};

struct BackgroundVertexOut {
  float4 position [[position]];
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
  uint vertexID [[vertex_id]],
  uint instanceID [[instance_id]],
  constant BackgroundUniforms &uniforms [[buffer(0)]]) {
  BackgroundVertexOut out;
  float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
  out.position = float4(positions[vertexID], 0.0, 1.0);
  out.uv = positions[vertexID] * 0.5 + 0.5;
  out.layer = instanceID;
  return out;
}

float3 shallowSeaGradient(float2 uv, float t, float intensity) {
  float depthFactor = smoothstep(0.0, 1.0, uv.y);
  float3 topColor = float3(0.05, 0.19, 0.24);
  float3 bottomColor = float3(0.01, 0.07, 0.11);
  float gentlePulse = 0.03 * sin(t * 0.3 + uv.x * 2.0);
  return mix(bottomColor, topColor, depthFactor) * (intensity + gentlePulse);
}

fragment float4 backgroundFragmentShader(
  BackgroundVertexOut in [[stage_in]],
  constant BackgroundUniforms &uniforms [[buffer(0)]]) {
  float3 color = shallowSeaGradient(in.uv, uniforms.time, uniforms.intensity);
  float caustic = 0.08 * sin((in.uv.x + uniforms.time * 0.1) * 12.0) *
                  sin((in.uv.y + uniforms.time * 0.15) * 8.0);
  color += caustic;

  if (in.layer == 0) {
    color *= float3(1.0, 0.98, 0.95);
  } else {
    color *= float3(0.95, 0.98, 1.0);
  }

  return float4(color, 1.0);
}

vertex ObjectVertexOut objectVertexShader(
  const device MeshVertex *vertices [[buffer(0)]],
  const device ObjectState *states [[buffer(1)]],
  constant SceneUniforms &uniforms [[buffer(2)]],
  uint vertexID [[vertex_id]],
  uint instanceID [[instance_id]]) {
  ObjectVertexOut out;
  uint layers = max(uniforms.layerCount, 1u);
  uint objectIndex = instanceID / layers;
  uint layer = instanceID % layers;
  MeshVertex vtx = vertices[vertexID];
  ObjectState state = states[objectIndex];

  float3 position = state.positionAndType.xyz + vtx.position * state.scaleAndPadding.xyz;
  float4 worldPos = float4(position, 1.0);
  out.position = uniforms.viewProjectionMatrix * worldPos;
  out.normal = vtx.normal;
  out.worldPos = position;
  out.objectType = state.positionAndType.w;
  out.layer = layer;
  return out;
}

float3 objectColor(float type, float3 pos, float time) {
  if (type > 0.5) {
    float hue = 0.52 + 0.05 * sin(time + pos.x);
    return mix(float3(0.1, 0.35, 0.45), float3(0.18, 0.5, 0.62), 0.5 + 0.5 * sin(hue));
  }
  float variation = 0.2 * sin(pos.x * 2.0 + time) + 0.1 * cos(pos.y * 3.0 - time * 0.5);
  return float3(0.05, 0.2, 0.28) + variation;
}

fragment float4 objectFragmentShader(
  ObjectVertexOut in [[stage_in]],
  constant SceneUniforms &uniforms [[buffer(0)]]) {
  float3 lightDir = normalize(float3(0.4, 0.9, -0.2));
  float ndotl = max(dot(in.normal, lightDir), 0.0);
  float rim = pow(1.0 - max(dot(in.normal, float3(0, 0, 1)), 0.0), 2.0);
  float3 baseColor = objectColor(in.objectType, in.worldPos, uniforms.time);
  float3 color = baseColor * (0.35 + ndotl * 0.65) + rim * 0.1;
  return float4(color, 0.95);
}

kernel void simulateObjects(
  device ObjectState *states [[buffer(0)]],
  constant SimulationUniforms &uniforms [[buffer(1)]],
  uint id [[thread_position_in_grid]]) {
  if (id >= uniforms.objectCount) {
    return;
  }

  ObjectState state = states[id];
  float dt = uniforms.deltaTime;
  float type = state.positionAndType.w;

  float3 velocity = state.velocityAndPhase.xyz;
  float phase = state.velocityAndPhase.w + dt * (0.5 + 0.5 * type);

  float3 offset = float3(sin(phase * 1.3), cos(phase * 0.8), sin(phase * 1.1)) * 0.02;
  float buoyancy = mix(0.2, 0.35, type);
  velocity.y += buoyancy * dt;
  velocity.x += sin(uniforms.globalTime * 0.2 + state.positionAndType.z) * 0.01 * dt;

  float3 position = state.positionAndType.xyz + velocity * dt + offset;

  if (position.y > 1.2) {
    position.y = -0.9;
  }
  if (position.x > 2.0 || position.x < -2.0) {
    velocity.x *= -0.8;
  }
  if (position.z > -0.3 || position.z < -3.5) {
    velocity.z *= -0.6;
  }

  state.positionAndType = float4(position, type);
  state.velocityAndPhase = float4(velocity, phase);
  states[id] = state;
}
