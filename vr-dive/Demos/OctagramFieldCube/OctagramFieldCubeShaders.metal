// OctagramFieldCubeShaders.metal
//
// Original cube-portal raymarch scene of repeating animated box clusters.
// Visual inspiration requested from ShaderToy tlVGDt:
// https://www.shadertoy.com/view/tlVGDt
// This implementation is original and does not reuse source code from the
// reference shader.

#include <metal_stdlib>
using namespace metal;

#define OFC_MAX_STEPS      56
#define OFC_HIT_EPS        0.015f
#define OFC_MAX_DIST       42.0f
#define OFC_CELL_SIZE      3.6f
#define OFC_SCENE_SCALE    10.0f

struct OctagramFieldCubeUniforms {
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

struct OctagramFieldCubeVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex OctagramFieldCubeVertexOut octagramFieldCubeVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant OctagramFieldCubeUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  OctagramFieldCubeVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float2 ofc_rot(float2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float ofc_hash13(float3 p) {
  p = fract(p * 0.1031f);
  p += dot(p, p.yzx + 19.19f);
  return fract((p.x + p.y) * p.z);
}

static float ofc_sdBox(float3 p, float3 b) {
  float3 q = abs(p) - b;
  return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

static float ofc_clusterSdf(float3 p, float time, thread float3 &emissive) {
  float3 cellId = floor((p + 0.5f * OFC_CELL_SIZE) / OFC_CELL_SIZE);
  float3 q = fmod(p + 0.5f * OFC_CELL_SIZE, OFC_CELL_SIZE) - 0.5f * OFC_CELL_SIZE;
  float phase = time + 6.28318f * ofc_hash13(cellId);

  q.xy = ofc_rot(q.xy, 0.55f + 0.8f * sin(phase * 0.7f));
  q.yz = ofc_rot(q.yz, 0.30f + 0.6f * cos(phase * 0.5f));
  q.xz = ofc_rot(q.xz, 0.25f + 0.5f * sin(phase * 0.9f));

  float swing = 0.72f + 0.30f * sin(phase * 1.4f);
  float shell = abs(ofc_sdBox(q, float3(0.78f))) - 0.10f;
  float armX0 = ofc_sdBox(q - float3(swing, 0.0f, 0.0f), float3(0.34f, 0.10f, 0.10f));
  float armX1 = ofc_sdBox(q + float3(swing, 0.0f, 0.0f), float3(0.34f, 0.10f, 0.10f));
  float armY0 = ofc_sdBox(q - float3(0.0f, swing, 0.0f), float3(0.10f, 0.34f, 0.10f));
  float armY1 = ofc_sdBox(q + float3(0.0f, swing, 0.0f), float3(0.10f, 0.34f, 0.10f));
  float armZ0 = ofc_sdBox(q - float3(0.0f, 0.0f, swing), float3(0.10f, 0.10f, 0.34f));
  float armZ1 = ofc_sdBox(q + float3(0.0f, 0.0f, swing), float3(0.10f, 0.10f, 0.34f));
  float core = ofc_sdBox(q, float3(0.17f));

  float d = min(shell, min(core, min(min(armX0, armX1), min(min(armY0, armY1), min(armZ0, armZ1)))));

  float hue = fract(0.17f * cellId.x + 0.11f * cellId.y + 0.07f * cellId.z + 0.1f * sin(phase));
  emissive = mix(float3(0.08f, 0.25f, 0.85f), float3(0.95f, 0.35f, 0.12f), hue);
  emissive *= 0.8f + 0.6f * sin(phase * 1.3f);
  return d;
}

static float ofc_sceneSdf(float3 p, float time, thread float3 &emissive) {
  return ofc_clusterSdf(p, time, emissive);
}

static float3 ofc_sceneNormal(float3 p, float time) {
  const float eps = 0.03f;
  float3 throwaway;
  float dx = ofc_sceneSdf(p + float3(eps, 0, 0), time, throwaway)
           - ofc_sceneSdf(p - float3(eps, 0, 0), time, throwaway);
  float dy = ofc_sceneSdf(p + float3(0, eps, 0), time, throwaway)
           - ofc_sceneSdf(p - float3(0, eps, 0), time, throwaway);
  float dz = ofc_sceneSdf(p + float3(0, 0, eps), time, throwaway)
           - ofc_sceneSdf(p - float3(0, 0, eps), time, throwaway);
  return normalize(float3(dx, dy, dz));
}

static bool ofc_boxHit(
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

fragment float4 octagramFieldCubeFragment(
  OctagramFieldCubeVertexOut in [[stage_in]],
  constant OctagramFieldCubeUniforms &uniforms [[buffer(0)]],
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
  if (!ofc_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float startT = max(tEntry, 0.0f);
  float3 entryLocal = roLocal + rdLocal * startT;
  float3 ro = entryLocal * OFC_SCENE_SCALE;
  float3 rd = rdLocal;
  float time = uniforms.time * uniforms.travelSpeed;

  float t = 0.0f;
  float3 accum = float3(0.0f);
  float transmittance = 1.0f;
  bool hit = false;
  float3 hitPos = float3(0.0f);
  float3 hitEmissive = float3(0.0f);

  for (int i = 0; i < OFC_MAX_STEPS; ++i) {
    if (t > OFC_MAX_DIST) break;
    float3 pos = ro + rd * t;
    float3 emissive;
    float d = ofc_sceneSdf(pos, time, emissive);

    float glow = exp(-max(d, 0.0f) * 7.5f);
    accum += transmittance * emissive * glow * 0.028f;
    transmittance *= 0.985f;

    if (d < OFC_HIT_EPS) {
      hit = true;
      hitPos = pos;
      hitEmissive = emissive;
      break;
    }

    t += clamp(d * 0.72f, 0.045f, 0.72f);
  }

  float3 color = accum;
  float depthFade = exp(-0.03f * t);
  color += mix(float3(0.01f, 0.02f, 0.05f), float3(0.02f, 0.05f, 0.10f), 0.5f + 0.5f * rd.y)
         * (0.20f + 0.12f * sin(time));

  if (hit) {
    float3 normal = ofc_sceneNormal(hitPos, time);
    float3 lightDir = normalize(float3(-0.45f, 0.65f, 0.60f));
    float diffuse = max(dot(normal, lightDir), 0.0f);
    float rim = pow(clamp(1.0f - max(dot(-rd, normal), 0.0f), 0.0f, 1.0f), 2.5f);
    float3 lit = hitEmissive * (0.55f + 1.25f * diffuse) + rim * float3(0.8f, 0.9f, 1.2f);
    color += lit * depthFade;
  }

  color = pow(max(color, 0.0f), float3(0.86f));
  return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}