#include <metal_stdlib>
using namespace metal;

struct SimoneSceneUniforms {
  float time;
  uint layerCount;
  float2 padding;
};

struct SimoneMeshVertex {
  float3 position;
  float3 normal;
};

struct SimoneParticleState {
  float4 positionAndScale;
  float4 seedAndPhase; // x=seed index, y=orbit progress, z=brightness, w=phase
};

struct SimoneVertexOut {
  float4 position [[position]];
  float3 normal [[flat]];
  float3 worldPos;
  float objectScale;
  float seedIndex [[flat]];
  float progress [[flat]];
  float brightness [[flat]];
};

vertex SimoneVertexOut simoneOrbitPointVertex(
    ushort amplificationID [[amplification_id]],
    const device SimoneMeshVertex *vertices [[buffer(0)]],
    const device SimoneParticleState *states [[buffer(1)]],
    constant SimoneSceneUniforms &uniforms [[buffer(2)]],
    constant float4x4 *viewProjectionMatrices [[buffer(3)]],
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]]) {
  SimoneVertexOut out;
  uint layers = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1);
  SimoneParticleState state = states[instanceID];
  SimoneMeshVertex meshVertex = vertices[vertexID];

  float phase = state.seedAndPhase.w;
  float c = cos(phase);
  float s = sin(phase);
  float3 rotatedMeshPos = float3(
      meshVertex.position.x * c - meshVertex.position.z * s,
      meshVertex.position.y,
      meshVertex.position.x * s + meshVertex.position.z * c);
  float3 rotatedMeshNormal = float3(
      meshVertex.normal.x * c - meshVertex.normal.z * s,
      meshVertex.normal.y,
      meshVertex.normal.x * s + meshVertex.normal.z * c);

  float3 simPos = state.positionAndScale.xyz;
  float3 reorientedPos = float3(simPos.x, simPos.z, simPos.y);
  float3 finalCenterPos = reorientedPos + float3(0.0f, -0.2f, -1.5f);
  float3 worldPos = finalCenterPos + rotatedMeshPos * state.positionAndScale.w;

  out.position = viewProjectionMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.normal = normalize(float3(rotatedMeshNormal.x, rotatedMeshNormal.z, rotatedMeshNormal.y));
  out.worldPos = worldPos;
  out.objectScale = state.positionAndScale.w;
  out.seedIndex = state.seedAndPhase.x;
  out.progress = state.seedAndPhase.y;
  out.brightness = state.seedAndPhase.z;
  return out;
}

fragment float4 simoneOrbitPointFragment(
    SimoneVertexOut in [[stage_in]],
    constant SimoneSceneUniforms &uniforms [[buffer(0)]]) {
  float3 normal = normalize(in.normal);
  float3 lightDir = normalize(float3(-0.35f, 0.85f, -0.25f));
  float3 viewDir = normalize(-in.worldPos);
  float ndotl = max(dot(normal, lightDir), 0.0f);
  float rim = pow(1.0f - max(dot(normal, viewDir), 0.0f), 3.0f);
  float spec = pow(max(dot(normalize(lightDir + viewDir), normal), 0.0f), 32.0f);

  float huePhase = fract(in.seedIndex * 0.173f + in.progress * 0.72f + sin(uniforms.time * 0.18f) * 0.08f);
  float3 paletteA = float3(0.25f, 0.55f, 1.00f);
  float3 paletteB = float3(0.95f, 0.35f, 0.82f);
  float3 paletteC = float3(1.00f, 0.86f, 0.38f);
  float3 paletteD = float3(0.35f, 1.00f, 0.72f);

  float3 coolMix = mix(paletteA, paletteB, smoothstep(0.0f, 0.55f, huePhase));
  float3 warmMix = mix(paletteC, paletteD, smoothstep(0.45f, 1.0f, huePhase));
  float3 baseColor = mix(coolMix, warmMix, smoothstep(0.25f, 0.95f, in.brightness));

  float shade = 0.38f + ndotl * 0.72f;
  float3 color = baseColor * shade;
  color += spec * float3(1.0f, 0.95f, 0.75f) * 0.35f;
  color += rim * baseColor * 0.55f;
  color *= 1.15f + in.brightness * 0.55f;

  return float4(color, 1.0f);
}
