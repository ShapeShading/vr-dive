// WaveLatticeCubeShaders.metal
//
// Original cube-portal wave lattice scene.
// Visual inspiration requested from ShaderToy Ml2XRD:
// https://www.shadertoy.com/view/Ml2XRD
// This implementation is original and does not reuse source code from the
// reference shader.

#include <metal_stdlib>
using namespace metal;

#define WLC_MAX_STEPS   96
#define WLC_MAX_DIST    36.0f
#define WLC_HIT_EPS     0.0012f
#define WLC_SCENE_SCALE 10.0f

struct WaveLatticeCubeUniforms {
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

struct WaveLatticeCubeVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex WaveLatticeCubeVertexOut waveLatticeCubeVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant WaveLatticeCubeUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  WaveLatticeCubeVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float2 wlc_rot(float2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return float2(p.x * c - p.y * s, p.x * s + p.y * c);
}

static float wlc_map(float3 p) {
  float k1 = 1.75f;
  float k2 = (sin(p.x * k1) + sin(p.z * k1)) * 0.7f;
  float k3 = (sin(p.y * k1) + sin(p.z * k1)) * 0.7f;

  float3 n0 = normalize(float3(0.0f, 1.0f, 0.0f));
  float3 n1 = normalize(float3(1.0f, 0.0f, 1.0f));
  float w1 = 3.8f - dot(abs(p), n0) + k2;
  float w2 = 3.6f - dot(abs(p), n1) + k3;

  float2 j0 = float2(sin((p.z + p.x) * 1.8f) * 0.25f, cos((p.z + p.x) * 0.9f) * 0.45f);
  float2 j1 = float2(sin((p.z - p.y) * 1.7f) * 0.25f, cos((p.x + p.y) * 1.1f) * 0.28f);
  float s1 = length(fmod(p.xy + j0 + 100.0f, 2.0f) - 1.0f) - 0.19f;
  float s2 = length(fmod(p.yz + j1 + 100.5f, 2.0f) - 1.0f) - 0.18f;
  float s3 = length(fmod(p.xz + j0.yx + 101.0f, 2.2f) - 1.1f) - 0.14f;

  return min(w1, min(w2, min(s1, min(s2, s3))));
}

static float3 wlc_normal(float3 p) {
  const float e = 0.002f;
  return normalize(float3(
    wlc_map(p + float3(e, 0, 0)) - wlc_map(p - float3(e, 0, 0)),
    wlc_map(p + float3(0, e, 0)) - wlc_map(p - float3(0, e, 0)),
    wlc_map(p + float3(0, 0, e)) - wlc_map(p - float3(0, 0, e))));
}

static bool wlc_boxHit(
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

fragment float4 waveLatticeCubeFragment(
  WaveLatticeCubeVertexOut in [[stage_in]],
  constant WaveLatticeCubeUniforms &uniforms [[buffer(0)]],
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
  if (!wlc_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float time = uniforms.time * uniforms.travelSpeed;
  float3 ro = (roLocal + rdLocal * max(tEntry, 0.0f)) * WLC_SCENE_SCALE;
  float3 rd = normalize(rdLocal);
  rd.xz = wlc_rot(rd.xz, time * 0.23f);
  rd = rd.yzx;
  rd.xz = wlc_rot(rd.xz, time * 0.20f);
  rd = normalize(rd.yzx);

  float t = 0.0f;
  float tt = 0.0f;
  bool hit = false;
  for (int i = 0; i < WLC_MAX_STEPS; ++i) {
    float3 p = ro + rd * t;
    tt = wlc_map(p);
    if (tt < WLC_HIT_EPS) {
      hit = true;
      break;
    }
    t += tt * 0.45f;
    if (t > WLC_MAX_DIST) { break; }
  }

  float3 p = ro + rd * t;
  float3 col = sqrt(max(float3(t * 0.08f), 0.0f));
  float accent = max(0.0f, wlc_map(p - float3(0.08f)) - tt);
  float3 color = 0.05f * t + abs(rd) * col + accent;

  if (hit) {
    float3 normal = wlc_normal(p);
    float rim = pow(clamp(1.0f - max(dot(-rd, normal), 0.0f), 0.0f, 1.0f), 2.0f);
    float diffuse = max(dot(normal, normalize(float3(0.6f, 0.8f, 0.4f))), 0.0f);
    color += float3(0.12f, 0.22f, 0.38f) * diffuse;
    color += float3(0.5f, 0.35f, 0.8f) * rim * 0.4f;
  } else {
    float skyMix = clamp(0.5f + 0.5f * rd.y, 0.0f, 1.0f);
    color = mix(float3(0.01f, 0.01f, 0.03f), float3(0.10f, 0.06f, 0.18f), skyMix);
  }

  color = clamp(color, 0.0f, 1.0f);
  return float4(color, 1.0f);
}