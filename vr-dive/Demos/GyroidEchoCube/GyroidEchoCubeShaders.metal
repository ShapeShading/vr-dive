// GyroidEchoCubeShaders.metal
//
// Original cube-portal reflective gyroid scene.
// Visual inspiration requested from ShaderToy tXtyW8:
// https://www.shadertoy.com/view/tXtyW8
// This implementation is original and does not reuse source code from the
// reference shader.

#include <metal_stdlib>
using namespace metal;

#define GEC_FAR         28.0f
#define GEC_PI          3.14159265f
#define GEC_MAX_STEPS   84
#define GEC_BOUNCES     2
#define GEC_SCENE_SCALE 10.0f

struct GyroidEchoCubeUniforms {
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

struct GyroidEchoCubeVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

struct GecHit {
  float dist;
  int material;
};

vertex GyroidEchoCubeVertexOut gyroidEchoCubeVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant GyroidEchoCubeUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  GyroidEchoCubeVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float3x3 gec_lookAt(float3 dir) {
  float3 up = float3(0.0f, 1.0f, 0.0f);
  float3 rt = normalize(cross(dir, up));
  float3 uu = cross(rt, dir);
  return float3x3(rt, uu, dir);
}

static float gec_gyroid(float3 p) {
  return dot(cos(p), sin(p.zxy)) + 0.85f;
}

static GecHit gec_map(float3 p) {
  GecHit hit;
  hit.dist = 1e5f;
  hit.material = 0;

  float d0 = gec_gyroid(p);
  if (d0 < hit.dist) {
    hit.dist = d0;
    hit.material = 1;
  }

  float d1 = gec_gyroid(p - float3(0.0f, 0.0f, GEC_PI));
  if (d1 < hit.dist) {
    hit.dist = d1;
    hit.material = 2;
  }

  return hit;
}

static GecHit gec_raymarch(float3 ro, float3 rd) {
  GecHit result;
  result.dist = 0.0f;
  result.material = 0;
  for (int i = 0; i < GEC_MAX_STEPS; ++i) {
    GecHit sample = gec_map(ro + rd * result.dist);
    if (abs(sample.dist) < 0.001f) {
      result.material = sample.material;
      break;
    }
    result.dist += clamp(sample.dist * 0.82f, 0.01f, 0.6f);
    result.material = sample.material;
    if (result.dist > GEC_FAR) { break; }
  }
  return result;
}

static float gec_ao(float3 p, float3 sn) {
  float occ = 0.0f;
  for (int i = 0; i < 4; ++i) {
    float t = float(i) * 0.08f;
    float d = gec_map(p + sn * t).dist;
    occ += t - d;
  }
  return clamp(1.0f - occ, 0.0f, 1.0f);
}

static float3 gec_normal(float3 p) {
  float2 e = float2(0.5773f, -0.5773f) * 0.001f;
  return normalize(
    e.xyy * gec_map(p + e.xyy).dist +
    e.yyx * gec_map(p + e.yyx).dist +
    e.yxy * gec_map(p + e.yxy).dist +
    e.xxx * gec_map(p + e.xxx).dist);
}

static float3 gec_trace(float3 ro, float3 rd, float time) {
  float3 color = float3(0.0f);
  float3 throughput = float3(1.0f);

  for (int bounce = 0; bounce < GEC_BOUNCES; ++bounce) {
    GecHit hit = gec_raymarch(ro, rd);
    if (hit.dist > GEC_FAR) {
      float skyMix = clamp(0.5f + 0.5f * rd.y, 0.0f, 1.0f);
      float3 sky = mix(float3(0.08f, 0.06f, 0.12f), float3(0.25f, 0.16f, 0.30f), skyMix);
      color += throughput * sky;
      break;
    }

    float fog = 1.0f - exp(-0.010f * hit.dist * hit.dist);
    throughput *= 1.0f - fog;

    float3 p = ro + rd * hit.dist;
    float3 sn = normalize(gec_normal(p) + pow(abs(cos(p * 48.0f)), float3(16.0f)) * 0.08f);

    float3 lp = float3(10.0f, -10.0f, -10.0f + ro.z);
    float3 ld = normalize(lp - p);
    float diff = max(0.0f, 0.45f + 1.7f * dot(sn, ld));
    float diff2 = pow(length(sin(sn * 2.1f) * 0.5f + 0.5f), 2.0f);
    float diff3 = max(0.0f, 0.5f + 0.5f * dot(sn, float3(0.0f, 1.0f, 0.0f)));

    float spec = max(0.0f, dot(reflect(-ld, sn), -rd));
    float fres = 1.0f - max(0.0f, dot(-rd, sn));
    float freck = dot(cos(p * 23.0f), float3(1.0f));

    float3 light = float3(0.0f);
    light += float3(0.4f, 0.6f, 0.9f) * diff;
    light += float3(0.5f, 0.1f, 0.1f) * diff2;
    light += float3(0.9f, 0.1f, 0.4f) * diff3;
    light += float3(0.18f, 0.16f, 0.18f) * pow(spec, 6.0f) * 2.2f;

    float3 albedo = float3(0.0f);
    if (hit.material == 1) {
      albedo = float3(0.2f, 0.1f, 0.9f) * max(0.6f, step(2.5f, freck));
    } else {
      albedo = float3(0.6f, 0.3f, 0.1f) * max(0.8f, step(-2.5f, freck));
    }

    float ao = gec_ao(p, sn);
    float3 localColor = light * albedo * ao;
    localColor += albedo * (0.08f + 0.18f * diff3);
    color += throughput * localColor;

    rd = reflect(rd, sn);
    ro = p + sn * 0.012f;
    throughput *= 0.22f + 0.28f * fres;
  }

  return color;
}

static bool gec_boxHit(
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

fragment float4 gyroidEchoCubeFragment(
  GyroidEchoCubeVertexOut in [[stage_in]],
  constant GyroidEchoCubeUniforms &uniforms [[buffer(0)]],
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
  if (!gec_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float time = uniforms.time * uniforms.travelSpeed;
  float3 ro = (roLocal + rdLocal * max(tEntry, 0.0f)) * GEC_SCENE_SCALE;
  float3 rd = rdLocal;

  rd.xy = float2(
    cos(sin(time * 0.2f)) * rd.x - sin(sin(time * 0.2f)) * rd.y,
    sin(sin(time * 0.2f)) * rd.x + cos(sin(time * 0.2f)) * rd.y);
  float3 ta = float3(cos(time * 0.4f), sin(time * 0.4f), 4.0f);
  rd = gec_lookAt(normalize(ta)) * rd;

  float3 color = gec_trace(float3(GEC_PI * 0.5f, 0.0f, -time * 5.0f), rd, time);
  color = pow(max(color, 0.0f), float3(0.4545f));
  return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}