// OrbitalSphereCubeShaders.metal
//
// Original cube-portal orbital sphere scene.
// Visual inspiration requested from ShaderToy llj3Rz:
// https://www.shadertoy.com/view/llj3Rz
// This implementation is original and does not reuse source code from the
// reference shader. The original used external resources; this version uses
// only procedural background, surface, and wire shading.

#include <metal_stdlib>
using namespace metal;

#define OSC_PI           3.14159265f
#define OSC_SCENE_SCALE  7.5f
#define OSC_SPHERE_RAD   1.05f

struct OrbitalSphereCubeUniforms {
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

struct OrbitalSphereCubeVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex OrbitalSphereCubeVertexOut orbitalSphereCubeVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant OrbitalSphereCubeUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  OrbitalSphereCubeVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float osc_hash21(float2 p) {
  float3 p3 = fract(float3(p.x, p.y, p.x) * 0.1031f);
  p3 += dot(p3, p3.yzx + 33.33f);
  return fract((p3.x + p3.y) * p3.z);
}

static float osc_sphereIntersect(float3 ro, float3 rd, float4 sph) {
  float3 oc = ro - sph.xyz;
  float b = dot(rd, oc);
  float c = dot(oc, oc) - sph.w * sph.w;
  float h = b * b - c;
  if (h <= 0.0f) return -1.0f;
  return -b - sqrt(h);
}

static float osc_sphereDistance(float3 ro, float3 rd, float4 sph) {
  float3 oc = ro - sph.xyz;
  float b = dot(oc, rd);
  float h = dot(oc, oc) - b * b;
  return sqrt(max(0.0f, h)) - sph.w;
}

static float osc_sphereSoftShadow(float3 ro, float3 rd, float4 sph, float k) {
  float3 oc = sph.xyz - ro;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - sph.w * sph.w;
  float h = b * b - c;
  return (b < 0.0f) ? 1.0f : 1.0f - smoothstep(0.0f, 1.0f, k * h / max(b, 1e-4f));
}

static float3 osc_sphereNormal(float3 pos, float4 sph) {
  return normalize((pos - sph.xyz) / sph.w);
}

static float osc_noise2(float2 p) {
  float2 i = floor(p);
  float2 f = fract(p);
  f = f * f * (3.0f - 2.0f * f);
  float a = osc_hash21(i + float2(0.0f, 0.0f));
  float b = osc_hash21(i + float2(1.0f, 0.0f));
  float c = osc_hash21(i + float2(0.0f, 1.0f));
  float d = osc_hash21(i + float2(1.0f, 1.0f));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

static float2 osc_voronoi(float2 x) {
  float2 n = floor(x);
  float2 f = fract(x);
  float3 m = float3(8.0f);
  for (int j = -1; j <= 1; ++j) {
    for (int i = -1; i <= 1; ++i) {
      float2 g = float2(float(i), float(j));
      float2 o = float2(osc_hash21(n + g), osc_hash21(n + g + 17.0f));
      float2 r = g - f + o;
      float d = dot(r, r);
      if (d < m.x) {
        m = float3(d, o);
      }
    }
  }
  return float2(sqrt(m.x), m.y + m.z);
}

static float3 osc_background(float3 d, float3 lightDir) {
  return float3(0.0f);
}

static float osc_heightMap(float3 pos, float time) {
  float2 r = pos.xz;
  float swell = 1.0f - 2.0f / (1.0f + 0.32f * dot(r, r));
  return pos.y - swell;
}

static float osc_rayMarchPlane(float3 ro, float3 rd, float tmax, float time) {
  float t = 0.0f;
  float h = (1.0f - ro.y) / rd.y;
  if (h > 0.0f) { t = h; }
  for (int i = 0; i < 24; ++i) {
    float3 pos = ro + t * rd;
    float dh = osc_heightMap(pos, time);
    if (dh < 0.001f || t > tmax) { break; }
    t += dh;
  }
  return t;
}

static bool osc_boxHit(
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

fragment float4 orbitalSphereCubeFragment(
  OrbitalSphereCubeVertexOut in [[stage_in]],
  constant OrbitalSphereCubeUniforms &uniforms [[buffer(0)]],
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
  if (!osc_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float time = uniforms.time * uniforms.travelSpeed;
  float3 ro = (roLocal + rdLocal * max(tEntry, 0.0f)) * OSC_SCENE_SCALE;
  float3 rd = rdLocal;
  float3 lightDir = normalize(float3(0.85f, 0.35f, 0.65f));
  float4 sphere = float4(0.0f, 0.0f, 0.0f, OSC_SPHERE_RAD);

  float3 color = osc_background(rd, lightDir);
  float sphereHit = osc_sphereIntersect(ro, rd, sphere);
  float tmax = sphereHit > 0.0f ? sphereHit : 20.0f;

  if (sphereHit > 0.0f) {
    float3 pos = ro + sphereHit * rd;
    float3 nor = osc_sphereNormal(pos, sphere);

    float am = 0.1f * time;
    float2 pr = float2(cos(am), sin(am));
    float3 tnor = nor;
    tnor.xz = float2(pr.x * tnor.x - pr.y * tnor.z, pr.y * tnor.x + pr.x * tnor.z);

    float am2 = 0.08f * time - (1.0f - nor.y * nor.y);
    pr = float2(cos(am2), sin(am2));
    float3 tnor2 = nor;
    tnor2.xz = float2(pr.x * tnor2.x - pr.y * tnor2.z, pr.y * tnor2.x + pr.x * tnor2.z);

    float fre = clamp(1.0f + dot(nor, rd), 0.0f, 1.0f);
    float lat = asin(clamp(tnor.y, -1.0f, 1.0f));
    float lon = atan2(tnor.z, tnor.x);
    float2 uv = float2(lon / (2.0f * OSC_PI) + 0.5f, lat / OSC_PI + 0.5f);
    float sheen = 0.5f + 0.5f * sin(8.0f * uv.x + 3.5f * uv.y + time * 0.15f);
    float horizonBand = smoothstep(0.15f, 0.95f, 1.0f - abs(nor.y));
    float3 mat = mix(float3(0.015f, 0.02f, 0.04f), float3(0.02f, 0.10f, 0.18f), 0.35f * sheen);
    mat += horizonBand * float3(0.01f, 0.05f, 0.10f) * 0.45f;

    float dif = clamp(dot(nor, lightDir), 0.0f, 1.0f);
    float3 lin = float3(1.0f, 1.15f, 1.35f) * (0.08f + 0.55f * dif);
    color = mat * lin;
    color += 0.75f * fre * fre * float3(0.72f, 0.82f, 1.0f) * (0.45f + 0.55f * dif);

    float spe = clamp(dot(reflect(rd, nor), lightDir), 0.0f, 1.0f);
    float tspe = 0.18f * pow(spe, 6.0f) + 0.65f * pow(spe, 28.0f);
    color += float3(0.70f, 0.80f, 1.0f) * tspe * (0.2f + 0.8f * dif);
  }

  float planeHit = osc_rayMarchPlane(ro, rd, tmax, time);
  if (planeHit < tmax) {
    float3 pos = ro + planeHit * rd;
    float2 scp = sin(2.0f * 6.2831f * pos.xz);
    float3 wire = float3(0.0f);
    wire += exp(-12.0f * abs(scp.x));
    wire += exp(-12.0f * abs(scp.y));
    wire += 0.45f * exp(-4.0f * abs(scp.x));
    wire += 0.45f * exp(-4.0f * abs(scp.y));
    wire *= 0.16f + 0.95f * osc_sphereSoftShadow(pos, lightDir, sphere, 4.0f);
    color += wire * float3(0.40f, 0.95f, 0.72f) * 0.52f * exp(-0.035f * planeHit * planeHit);
  }

  if (dot(rd, sphere.xyz - ro) > 0.0f) {
    float d = osc_sphereDistance(ro, rd, sphere);
    float3 glow = float3(0.0f);
    glow += float3(0.40f, 0.75f, 1.00f) * 0.75f * exp(-2.3f * abs(d)) * step(0.0f, d);
    glow += float3(0.45f, 0.80f, 1.00f) * 0.22f * exp(-9.0f * abs(d));
    glow += float3(0.82f, 0.90f, 1.00f) * 0.30f * exp(-110.0f * abs(d));
    color += glow * 1.35f;
  }

  color *= smoothstep(0.0f, 2.5f, time + 0.3f);
  color = clamp(color, 0.0f, 1.0f);
  return float4(color, 1.0f);
}