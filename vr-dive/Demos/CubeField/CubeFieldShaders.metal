#include <metal_stdlib>
using namespace metal;

struct SceneUniforms {
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

struct ObjectVertexOut {
  float4 position [[position]];
  float3 normal;
  float3 worldPos;
  // uint layer [[render_target_array_index]]; // Removed
  float objectType;
};

vertex ObjectVertexOut objectVertexShader(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  const device ObjectState *states [[buffer(1)]],
  constant SceneUniforms &uniforms [[buffer(2)]],
  constant float4x4 *viewProjectionMatrices [[buffer(3)]],
  uint vertexID [[vertex_id]],
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
  float4x4 viewProjection = viewProjectionMatrices[viewIndex];
  out.position = viewProjection * worldPos;
  out.normal = vtx.normal;
  out.worldPos = position;
  out.objectType = state.positionAndType.w;
  // out.layer = viewIndex; // Removed
  return out;
}

float3 objectColor(float type, float3 pos) {
  float variant = clamp(round(type), 0.0, 2.0);
  const float3 penumbra = float3(0.038, 0.048, 0.06);
  const float3 cool = float3(0.08, 0.09, 0.11);
  const float3 warm = float3(0.12, 0.11, 0.1);
  const float3 highlight = float3(0.16, 0.17, 0.19);

  float heightMix = saturate((pos.y + 5.0) / 10.0);
  float depthMix = saturate((-pos.z - 0.5) / 4.0);
  float edgeLift = smoothstep(0.0, 1.0, depthMix);

  float3 base;
  if (variant < 0.5) {
    base = mix(penumbra, cool, 0.35 + 0.45 * heightMix);
  } else if (variant < 1.5) {
    base = mix(penumbra, warm, 0.4 + 0.3 * depthMix + 0.15 * heightMix);
  } else {
    base = mix(cool, highlight, 0.4 + 0.4 * heightMix + 0.2 * depthMix);
  }

  base += float3(0.01, 0.012, 0.014) * edgeLift;
  return base;
}

fragment float4 objectFragmentShader(ObjectVertexOut in [[stage_in]],
                                     constant SceneUniforms &uniforms
                                     [[buffer(0)]],
                                     bool isFrontFacing [[front_facing]]) {
  float3 normal = normalize(in.normal);
  normal = isFrontFacing ? normal : -normal;
  float3 lightDir = normalize(float3(0.4, 0.9, -0.2));
  float ndotl = abs(dot(normal, lightDir));
  float rim = pow(1.0 - abs(dot(normal, float3(0, 0, 1))), 2.0);
  float3 baseColor = objectColor(in.objectType, in.worldPos);
  float3 color = baseColor * (0.35 + ndotl * 0.65) + rim * 0.08;
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
  float scaledDt = dt * 0.11;
  float shapeType = clamp(state.positionAndType.w, 0.0, 2.0);

  float3 amplitude = state.motionAndPhase.xyz;
  float phase = state.motionAndPhase.w;
  float3 home = state.homeAndJitter.xyz;
  float jitterRadius = state.homeAndJitter.w;

  float3 flowWeights;
  float phaseSpeed;
  float lift;
  float bobScale;
  float jitterScale;

  if (shapeType < 0.5) {
    flowWeights = float3(0.9, 0.8, 0.9);
    phaseSpeed = 0.32;
    lift = 0.05;
    bobScale = 0.75;
    jitterScale = 0.35;
  } else if (shapeType < 1.5) {
    flowWeights = float3(0.55, 1.6, 0.55);
    phaseSpeed = 0.24;
    lift = 0.12;
    bobScale = 1.05;
    jitterScale = 0.48;
  } else {
    flowWeights = float3(1.25, 1.1, 1.25);
    phaseSpeed = 0.42;
    lift = 0.1;
    bobScale = 1.2;
    jitterScale = 0.55;
  }

  phase += scaledDt * phaseSpeed;

  float3 swirl = float3(
                    sin(phase * (0.9 + 0.15 * flowWeights.x) + home.x),
                    cos(phase * (0.7 + 0.2 * flowWeights.y) + home.y),
                    sin(phase * (1.05 + 0.15 * flowWeights.z) + home.z)) *
                 (amplitude * flowWeights);

  float3 current = float3(
      sin(uniforms.globalTime * 0.2 + home.y) * jitterRadius * jitterScale,
      0.0,
      cos(uniforms.globalTime * 0.17 + home.x) * jitterRadius * jitterScale);

  if (shapeType >= 0.5 && shapeType < 1.5) {
    current.y = sin(uniforms.globalTime * 0.1 + home.z) * jitterRadius * 0.5;
  }

  float bob = sin(phase * (0.4 + 0.15 * flowWeights.y) + home.z) * jitterRadius * bobScale;

  float3 position = home;
  position.x += swirl.x + current.x;
  position.y += swirl.y * 0.5 + current.y + bob + lift;
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

  state.positionAndType = float4(position, shapeType);
  state.motionAndPhase.w = phase;
  states[id] = state;
}
