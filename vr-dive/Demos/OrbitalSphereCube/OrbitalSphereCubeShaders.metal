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
  /// Pattern-space navigation transform (identity in normal mode).
  float4x4 patternTransform;
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
  if (tmax <= 0.0f) {
    return tmax + 1.0f;
  }

  const int coarseSteps = 48;
  float prevT = 0.0f;
  float prevH = osc_heightMap(ro, time);
  float stepSize = tmax / float(coarseSteps);

  for (int i = 1; i <= coarseSteps; ++i) {
    float t = min(tmax, stepSize * float(i));
    float h = osc_heightMap(ro + t * rd, time);
    if ((prevH <= 0.0f && h >= 0.0f) || (prevH >= 0.0f && h <= 0.0f)) {
      float a = prevT;
      float b = t;
      float ha = prevH;
      for (int j = 0; j < 6; ++j) {
        float mid = 0.5f * (a + b);
        float hm = osc_heightMap(ro + mid * rd, time);
        if ((ha <= 0.0f && hm <= 0.0f) || (ha >= 0.0f && hm >= 0.0f)) {
          a = mid;
          ha = hm;
        } else {
          b = mid;
        }
      }
      return 0.5f * (a + b);
    }
    prevT = t;
    prevH = h;
  }

  return tmax + 1.0f;
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
  // Real camera in cube-local space (used for box boundary hit — keeps the cube fixed)
  float3 boxRoLocal = (camWorld - center) / uniforms.cubeScale;
  float3 boxRdLocal = normalize(in.worldPos - camWorld);

  // Box boundary test uses the REAL camera so the cube mesh boundary stays fixed.
  float tEntry;
  float tExit;
  if (!osc_boxHit(boxRoLocal, boxRdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  // Entry point on the box surface in cube-local space (real camera perspective).
  float3 realEntry = boxRoLocal + boxRdLocal * max(tEntry, 0.0f);

  // Apply pattern navigation ONLY to the scene entry point and ray direction.
  // This shifts what's rendered inside the box without moving the box boundary.
  float3 roLocal = (uniforms.patternTransform * float4(realEntry, 1.0f)).xyz;
  float3 rdLocal = normalize(float3(uniforms.patternTransform * float4(boxRdLocal, 0.0f)));

  float time = uniforms.time * uniforms.travelSpeed;
  float3 ro = roLocal * OSC_SCENE_SCALE;
  float3 rd = rdLocal;
  float3 lightDir = normalize(float3(0.85f, 0.35f, 0.65f));
  float4 sphere = float4(0.0f, 0.0f, 0.0f, OSC_SPHERE_RAD);

  float3 color = osc_background(rd, lightDir);
  float sphereHit = osc_sphereIntersect(ro, rd, sphere);
  float tmax = sphereHit > 0.0f ? sphereHit : 20.0f;

  if (sphereHit > 0.0f) {
    float3 pos = ro + sphereHit * rd;
    float3 nor = osc_sphereNormal(pos, sphere);
    (void)pos;
    (void)nor;
    color = float3(0.06f, 0.20f, 0.38f);
  }

  float planeHit = osc_rayMarchPlane(ro, rd, tmax, time);
  if (planeHit < tmax) {
    float3 pos = ro + planeHit * rd;
    float2 scp = sin(2.0f * 6.2831f * pos.xz);
    float3 wire = float3(0.0f);
    wire += exp(-24.0f * abs(scp.x));
    wire += exp(-24.0f * abs(scp.y));
    color += wire * float3(0.40f, 0.95f, 0.72f) * 0.75f * exp(-0.03f * planeHit * planeHit);
  }

  if (dot(rd, sphere.xyz - ro) > 0.0f) {
    float d = osc_sphereDistance(ro, rd, sphere);
    float3 glow = float3(0.0f);
    glow += float3(0.40f, 0.75f, 1.00f) * 0.55f * exp(-3.8f * abs(d)) * step(0.0f, d);
    glow += float3(0.45f, 0.80f, 1.00f) * 0.18f * exp(-13.0f * abs(d));
    glow += float3(0.82f, 0.90f, 1.00f) * 0.22f * exp(-150.0f * abs(d));
    color += glow * 1.8f;
  }

  color *= smoothstep(0.0f, 2.5f, time + 0.3f);
  color = clamp(color, 0.0f, 1.0f);
  return float4(color, 1.0f);
}