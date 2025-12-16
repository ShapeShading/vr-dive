#include <metal_stdlib>
using namespace metal;

struct SceneUniforms {
  float time;
  uint layerCount;
  float2 padding;
};

struct MeshVertex {
  float3 position [[attribute(0)]];
  float3 normal [[attribute(1)]];
};

struct PagodaInstance {
  float4x4 modelMatrix;
  float4 color;
};

struct VertexOut {
  float4 position [[position]];
  float3 worldPosition;
  float3 normal;
  float4 color;
};

vertex VertexOut pagodaVertexShader(
    MeshVertex in [[stage_in]],
    constant PagodaInstance *instances [[buffer(1)]],
    constant SceneUniforms &uniforms [[buffer(2)]],
    constant float4x4 *viewProjections [[buffer(3)]],
    uint instanceId [[instance_id]], uint ampId [[amplification_id]]) {
  VertexOut out;
  PagodaInstance instance = instances[instanceId];

  float4 worldPos = instance.modelMatrix * float4(in.position, 1.0);
  out.worldPosition = worldPos.xyz;
  out.normal = (instance.modelMatrix * float4(in.normal, 0.0)).xyz;
  out.color = instance.color;

  out.position = viewProjections[ampId] * worldPos;

  return out;
}

fragment float4 pagodaFragmentShader(VertexOut in [[stage_in]],
                                     constant SceneUniforms &uniforms
                                     [[buffer(0)]]) {
  float3 N = normalize(in.normal);
  float3 L = normalize(float3(1.0, 2.0, 1.0)); // Simple directional light

  float diffuse = max(dot(N, L), 0.0);
  float ambient = 0.3;

  float3 finalColor = in.color.rgb * (diffuse + ambient);

  return float4(finalColor, in.color.a);
}

struct SolidOut {
  float4 position [[position]];
  float3 worldPosition;
  float3 normal;
};

vertex SolidOut pagodaSolidVertexShader(
    MeshVertex in [[stage_in]],
    constant SceneUniforms &uniforms [[buffer(2)]],
    constant float4x4 *viewProjections [[buffer(3)]],
    uint ampId [[amplification_id]]) {
  SolidOut out;
  float4 worldPos = float4(in.position, 1.0);
  out.worldPosition = worldPos.xyz;
  out.normal = in.normal;
  out.position = viewProjections[ampId] * worldPos;
  return out;
}

fragment float4 pagodaSolidFragmentShader(SolidOut in [[stage_in]],
                                          constant SceneUniforms &uniforms [[buffer(0)]],
                                          constant float4 &partColor [[buffer(1)]]) {
  float3 N = normalize(in.normal);
  float3 L = normalize(float3(1.0, 2.0, 1.0));
  float diffuse = max(dot(N, L), 0.0);
  float ambient = 0.35;
  float3 base = partColor.rgb;
  float3 finalColor = base * (diffuse + ambient);
  return float4(finalColor, partColor.a);
}
