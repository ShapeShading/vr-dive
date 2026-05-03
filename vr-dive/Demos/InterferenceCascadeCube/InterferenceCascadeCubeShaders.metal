// InterferenceCascadeCubeShaders.metal
//
// Original cube-portal interference field scene.
// Visual inspiration requested from ShaderToy scSGD1:
// https://www.shadertoy.com/view/scSGD1
// This implementation is original and does not reuse source code from the
// reference shader.

#include <metal_stdlib>
using namespace metal;

#define ICC_STEPS       42
#define ICC_MAX_DIST    22.0f
#define ICC_SCENE_SCALE 11.0f

struct InterferenceCascadeCubeUniforms {
  float time;
  uint viewCount;
  float cubeScale;
  float travelSpeed;
  float4 objectCenter;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct InterferenceCascadeCubeVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex InterferenceCascadeCubeVertexOut interferenceCascadeCubeVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant InterferenceCascadeCubeUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  InterferenceCascadeCubeVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float2 icc_rot(float2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float4 icc_palette(float x) {
  return 0.5f + 0.5f * cos(x + float4(0.0f, 1.0f, 2.0f, 0.0f));
}

static float icc_logRepeatMetric(float x, float time, thread float &radiusA, thread float &radiusB) {
  float safeX = max(x, 1e-3f);
  float layer = floor(time - log(safeX) / 0.48f);
  radiusA = pow(0.69f, layer - time + 0.4f);
  radiusB = radiusA * 0.67f;
  return layer;
}

static float icc_map(float3 p, float time) {
  p.z -= 2.7f;
  p.xz = icc_rot(p.xz, 0.3f);
  p.zy = icc_rot(p.zy, sin(time * 0.25f) * 0.3f + 0.9f);
  p.xy = icc_rot(p.xy, time * 0.25f);
  p.x = abs(p.x) - 1.0f;

  float radiusA;
  float radiusB;
  icc_logRepeatMetric(abs(p.x) + 0.05f, time, radiusA, radiusB);

  float wave = smoothstep(0.88f, 1.0f, cos(length(p.xy + p.z) * 2.1f - time));
  float lobe = abs(length(p - float3(radiusA, 0.0f, 0.0f)) - radiusA * 0.23f);
  float lobe2 = abs(length(p - float3(radiusB, 0.0f, 0.0f)) - radiusA * 0.15f);

  float3 q = p;
  q.yx = icc_rot(q.yx, atan2(q.y, q.x));
  q.zx = icc_rot(q.zx, atan2(max(q.z, 0.0f), q.x));
  float corridor = length(q.xy) - (0.13f + 0.05f * wave);
  float spine = abs(q.z) - (0.08f + 0.02f * wave);

  return min(min(lobe, lobe2), min(corridor, spine));
}

static bool icc_boxHit(
  float3 ro, float3 rd, float3 bmin, float3 bmax,
  thread float &tNear, thread float &tFar)
{
  float3 t0 = (bmin - ro) / rd;
  float3 t1 = (bmax - ro) / rd;
  float3 lo = min(t0, t1);
  float3 hi = max(t0, t1);
  tNear = max(max(lo.x, lo.y), lo.z);
  tFar = min(min(hi.x, hi.y), hi.z);
  return tFar >= max(tNear, 0.0f);
}

fragment float4 interferenceCascadeCubeFragment(
  InterferenceCascadeCubeVertexOut in [[stage_in]],
  constant InterferenceCascadeCubeUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center = uniforms.objectCenter.xyz;
  float3 roLocal = (camWorld - center) / uniforms.cubeScale;
  float3 rdLocal = normalize(in.worldPos - camWorld);

  float tEntry;
  float tExit;
  if (!icc_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float time = uniforms.time * uniforms.travelSpeed;
  float3 ro = (roLocal + rdLocal * max(tEntry, 0.0f)) * ICC_SCENE_SCALE;
  float3 rd = normalize(rdLocal);

  float t = 0.0f;
  float4 accum = float4(0.0f);
  for (int i = 0; i < ICC_STEPS; ++i) {
    float3 p = ro + rd * t;
    float radiusA;
    float radiusB;
    icc_logRepeatMetric(abs(p.x) + 0.05f, time, radiusA, radiusB);

    float d = icc_map(p, time);
    float lxy = max(length(p.xy), 1e-3f);
    float pulse = smoothstep(0.9f, 1.0f, cos(lxy * 1.8f + p.z * 0.65f - time));
    float4 glow = (0.008f + 0.018f * pulse)
          * (1.0f + cos(lxy * 2.0f + float(i) * 0.2f + float4(0.0f, 1.0f, 2.0f, 0.0f)))
          / (0.45f + 1.25f * lxy);
    float density = exp(-15.0f * abs(d));
    float4 tint = icc_palette(lxy + p.z * 0.4f + time * 0.5f);
    accum += tint * glow * density * 1.7f;

    t += clamp(d * 0.82f + 0.035f, 0.05f, 0.9f);
    if (t > ICC_MAX_DIST) { break; }
  }

  float3 color = tanh(accum.xyz * 0.95f);
  color = pow(color, float3(0.88f));
  return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}