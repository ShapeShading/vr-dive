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
#define WLC_HIT_EPS     0.0010f
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

static float2 wlc_mod(float2 x, float y) {
  return x - y * floor(x / y);
}

static float wlc_map(float3 p) {
  float3 n = float3(0.0f, 1.0f, 0.0f);
  float k1 = 1.9f;
  float k2 = (sin(p.x * k1) + sin(p.z * k1)) * 0.8f;
  float k3 = (sin(p.y * k1) + sin(p.z * k1)) * 0.8f;

  float w1 = 4.0f - dot(abs(p), normalize(n)) + k2;
  float w2 = 4.0f - dot(abs(p), normalize(n.yzx)) + k3;

  float2 j0 = float2(sin((p.z + p.x) * 2.0f) * 0.3f, cos((p.z + p.x) * 1.0f) * 0.5f);
  float2 j1 = float2(sin((p.z + p.x) * 2.0f) * 0.3f, cos((p.z + p.x) * 1.0f) * 0.3f);
  float s1 = length(wlc_mod(p.xy + j0, 2.0f) - 1.0f) - 0.2f;
  float s2 = length(wlc_mod(0.5f + p.yz + j1, 2.0f) - 1.0f) - 0.2f;

  return min(w1, min(w2, min(s1, s2)));
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
  ro += float3(0.0f, 0.0f, time * 3.8f);
  float3 rd = normalize(rdLocal);
  rd.xz = wlc_rot(rd.xz, time * 0.23f);
  rd = rd.yzx;
  rd.xz = wlc_rot(rd.xz, time * 0.20f);
  rd = normalize(rd.yzx);

  float t = 0.0f;
  float tt = 0.0f;
  for (int i = 0; i < WLC_MAX_STEPS; ++i) {
    float3 p = ro + rd * t;
    tt = wlc_map(p);
    if (tt < WLC_HIT_EPS) {
      break;
    }
    t += tt * 0.45f;
    if (t > WLC_MAX_DIST) { break; }
  }

  float3 p = ro + rd * t;
  float3 col = sqrt(max(float3(t * 0.1f), 0.0f));
  float accent = max(0.0f, wlc_map(p - float3(0.1f)) - tt);
  float3 color = 0.05f * t + abs(rd) * col + accent;

  color = clamp(color, 0.0f, 1.0f);
  return float4(color, 1.0f);
}