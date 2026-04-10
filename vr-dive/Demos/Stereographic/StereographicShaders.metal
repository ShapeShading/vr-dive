#include <metal_stdlib>
using namespace metal;

struct StereoSceneUniforms {
  float time;
  uint layerCount;
  float2 padding;
};

struct StereographicVertex {
  float3 position;
  float3 normal;
  float4 color;
};

struct StereoVertexOut {
  float4 position [[position]];
  float3 normal;
  float4 color;
};

vertex StereoVertexOut stereographicVertexShader(
    ushort amplificationID [[amplification_id]],
    const device StereographicVertex *vertices [[buffer(0)]],
    constant StereoSceneUniforms &uniforms [[buffer(2)]],
    constant float4x4 *viewProjectionMatrices [[buffer(3)]],
    uint vertexID [[vertex_id]]) {
  StereoVertexOut out;

  uint layers = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1);

  StereographicVertex vtx = vertices[vertexID];

  out.position = viewProjectionMatrices[viewIndex] * float4(vtx.position, 1.0);
  out.normal = normalize(vtx.normal);
  out.color = vtx.color;
  return out;
}

fragment float4 stereographicFragmentShader(StereoVertexOut in [[stage_in]],
                                            bool isFrontFacing [[front_facing]]) {
  float3 normal = isFrontFacing ? normalize(in.normal) : -normalize(in.normal);
  float3 lightDir = normalize(float3(0.35, 0.85, -0.25));
  float diffuse = max(dot(normal, lightDir), 0.0);
  float rim = pow(1.0 - max(dot(normal, float3(0.0, 0.0, 1.0)), 0.0), 2.0);
  float brightness = 0.22 + diffuse * 0.7 + rim * 0.2;

  float3 color = in.color.rgb * brightness;
  return float4(color, 1.0);
}
