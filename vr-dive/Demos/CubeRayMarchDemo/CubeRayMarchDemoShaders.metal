// CubeRayMarchDemoShaders.metal
// Standard mesh shading for CPU-generated probe spheres, rays, cube frame and torus.
#include <metal_stdlib>
using namespace metal;

struct CRMVertex {
  float3 position;
  float3 normal;
  float3 color;
};

struct CRMUniforms {
  float  time;
  uint   viewCount;
  float2 pad0;
  float4 objectCenter;
  float4 lightPosition;
};

struct CRMOut {
  float4 clipPos [[position]];
  float3 worldPos;
  float3 normal;
  float3 color;
  uint   viewIndex [[flat]];
};

vertex CRMOut cubeRayMarchVertex(
  ushort amplificationID [[amplification_id]],
  const device CRMVertex *vertices [[buffer(0)]],
  constant CRMUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  CRMVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position + uniforms.objectCenter.xyz;

  CRMOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.normal = vtx.normal;
  out.color = vtx.color;
  out.viewIndex = viewIndex;
  return out;
}

fragment float4 cubeRayMarchFragment(
  CRMOut in [[stage_in]],
  constant CRMUniforms &uniforms [[buffer(0)]],
  constant float4x4 *viewToWorld [[buffer(1)]],
  bool frontFacing [[front_facing]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = viewToWorld[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
  float3 lightWorld = uniforms.objectCenter.xyz + uniforms.lightPosition.xyz;

  float3 N = normalize(in.normal);
  if (!frontFacing) N = -N;
  float3 L = normalize(lightWorld - in.worldPos);
  float3 V = normalize(camWorld - in.worldPos);
  float3 H = normalize(L + V);

  float diffuse = max(dot(N, L), 0.0f);
  float specular = pow(max(dot(N, H), 0.0f), 42.0f);
  float rim = pow(1.0f - max(dot(N, V), 0.0f), 2.3f);

  float3 color = in.color * (0.16f + 0.84f * diffuse);
  color += in.color * rim * 0.20f;
  color += float3(1.0f, 0.98f, 0.92f) * specular * 0.30f;
  return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}
