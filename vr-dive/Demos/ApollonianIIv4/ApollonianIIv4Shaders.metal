// ApollonianIIv4Shaders.metal
//
// Source reference:
// https://www.shadertoy.com/view/WlcXR2
// "Apollonian II" by inigo quilez - iq/2016
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported
//
// Adapted for vr-dive: renders inside a view-independent 2 metre cube container.
// Camera is driven by the visionOS head pose; the original ShaderToy camera orbit
// is replaced by standard box-intersection ray marching.

#include <metal_stdlib>
using namespace metal;

// Scene is scaled so that the repeating Apollonian cell (period 2) comfortably
// fills the cube. Smaller values = larger structures visible from outside.
#define AP_SCENE_SCALE   1.0f
#define AP_MAXD          20.0f
#define AP_MAX_STEPS     256

struct ApollonianIIv4Uniforms {
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

struct ApollonianIIv4VertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------
vertex ApollonianIIv4VertexOut apollonianIIv4Vertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant ApollonianIIv4Uniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  ApollonianIIv4VertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

// ---------------------------------------------------------------------------
// Box intersection helper
// ---------------------------------------------------------------------------
static bool ap_boxHit(
  float3 ro, float3 rd, float3 bmin, float3 bmax,
  thread float &tNear, thread float &tFar)
{
  float3 t0 = (bmin - ro) / rd;
  float3 t1 = (bmax - ro) / rd;
  float3 lo = min(t0, t1);
  float3 hi = max(t0, t1);
  tNear = max(max(lo.x, lo.y), lo.z);
  tFar  = min(min(hi.x, hi.y), hi.z);
  return tFar >= max(tNear, 0.0f);
}

static float ap_edgeDistance(float3 p) {
  float3 a = abs(p);
  if (a.x > a.y && a.x > a.z) return min(1.0f - a.y, 1.0f - a.z);
  if (a.y > a.z) return min(1.0f - a.x, 1.0f - a.z);
  return min(1.0f - a.x, 1.0f - a.y);
}

static float3 ap_faceNormal(float3 p) {
  float3 a = abs(p);
  if (a.x > a.y && a.x > a.z) return float3(sign(p.x), 0.0f, 0.0f);
  if (a.y > a.z) return float3(0.0f, sign(p.y), 0.0f);
  return float3(0.0f, 0.0f, sign(p.z));
}

// GLSL-compatible mod: x - y*floor(x/y), always non-negative when y > 0.
// Metal's fmod() truncates toward zero and gives wrong results for negative x.
static float3 glsl_mod(float3 x, float y) {
  return x - y * floor(x / y);
}

// ---------------------------------------------------------------------------
// Apollonian SDF  (direct port of map() from ShaderToy WlcXR2)
// Returns float3( distance, adr, k*0.5 )
// ---------------------------------------------------------------------------
static float3 ap_map(float3 ppp, float iTime)
{
  float3 p = ppp;

  // Move scene forward instead of moving camera — original IQ trick.
  p.z += iTime * 0.5f;

  float i = 0.0f, s = 1.0f, k = 1.0f;

  // Repeat Apollonian fractal — 6 iterations.
  while (i++ < 6.0f) {
    float3 pp = glsl_mod(p - 1.0f, 2.0f) - 1.0f;
    p = pp;
    k = 1.0f / dot(pp, p);
    p *= k;
    s *= k;
  }

  float a1 = dot(p.xy, p.xy);
  float a2 = dot(p.yz, p.yz);
  float a3 = dot(p.zx, p.zx);

  float d1 = sqrt(min(min(a1, a2), a3)) - 0.11f;
  float d2 = abs(p.y);
  float dmi = d2;
  float adr = 0.7f * fract((0.5f * p.y + 0.5f) * 8.0f);

  if (d1 < d2) {
    dmi = d1;
    adr = 0.0f;
  }

  return float3(0.5f * dmi / s, adr, k * 0.5f);
}

// ---------------------------------------------------------------------------
// Ray march  (port of trace())
// ---------------------------------------------------------------------------
static float3 ap_trace(float3 ro, float3 rd, float tMax, float iTime)
{
  float t = 0.01f;
  float2 info = float2(0.0f);
  for (int i = 0; i < AP_MAX_STEPS; ++i) {
    float precis = 0.001f * t;
    float3 r = ap_map(ro + rd * t, iTime);
    float h = r.x;
    info = r.yz;
    if (h < precis || t > tMax) break;
    t += h;
  }
  if (t > tMax) t = -1.0f;
  return float3(t, info);
}

// ---------------------------------------------------------------------------
// Normal  (port of calcNormal())
// ---------------------------------------------------------------------------
static float3 ap_normal(float3 pos, float t, float iTime)
{
  float precis = 0.0001f * t * 0.57f;
  float2 e = float2(precis, -precis);
  return normalize(
    e.xyy * ap_map(pos + e.xyy, iTime).x +
    e.yyx * ap_map(pos + e.yyx, iTime).x +
    e.yxy * ap_map(pos + e.yxy, iTime).x +
    e.xxx * ap_map(pos + e.xxx, iTime).x);
}

// Spherical Fibonacci direction (port of forwardSF())
static float3 ap_forwardSF(float i, float n)
{
  const float PI  = 3.141592653589793f;
  const float PHI = 1.618033988749895f;
  float phi = 2.0f * PI * fract(i / PHI);
  float zi = 1.0f - (2.0f * i + 1.0f) / n;
  float sinTheta = sqrt(1.0f - zi * zi);
  return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, zi);
}

// AO  (port of calcAO())
static float ap_ao(float3 pos, float3 nor, float iTime)
{
  float ao = 0.0f;
  for (int i = 0; i < 16; ++i) {
    float3 w = ap_forwardSF(float(i), 16.0f);
    w *= sign(dot(w, nor));
    float h = float(i) / 15.0f;
    ao += clamp(ap_map(pos + nor * 0.1f + h, iTime).x * 2.0f, 0.0f, 1.0f);
  }
  ao /= 16.0f;
  return clamp(ao * 16.0f, 0.0f, 1.0f);
}

// ---------------------------------------------------------------------------
// Shade a hit point  (port of render() body)
// ---------------------------------------------------------------------------
static float3 ap_shade(float3 ro, float3 rd, float3 res, float iTime)
{
  float3 col = float3(0.0f);
  float t = res.x;
  if (t > 0.0f) {
    float3 pos = ro + t * rd;
    float3 nor = ap_normal(pos, t, iTime);
    float fre = clamp(1.0f + dot(rd, nor), 0.0f, 1.0f);
    float occ = pow(clamp(res.z * 2.0f, 0.0f, 1.0f), 1.2f);
    occ = 1.5f * (0.1f + 0.9f * occ) * ap_ao(pos, nor, iTime);
    float3 lin = float3(1.0f, 1.0f, 1.5f)
                 * (2.0f + fre * fre * float3(1.8f, 1.0f, 1.0f))
                 * occ * (1.0f - 0.5f * abs(nor.y));

    col = 0.5f + 0.5f * cos(6.2831f * res.y + float3(0.0f, 1.0f, 2.0f));
    col = col * lin;
    col += 0.6f * pow(1.0f - fre, 32.0f) * occ * float3(0.5f, 1.0f, 1.5f);
    col *= exp(-0.3f * t);
  }
  col.z += 0.01f;
  return sqrt(col);
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------
fragment float4 apollonianIIv4Fragment(
  ApollonianIIv4VertexOut in [[stage_in]],
  constant ApollonianIIv4Uniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center = uniforms.objectCenter.xyz;
  float3 roLocal = (camWorld - center) / uniforms.cubeScale;
  float3 rdLocal = normalize(in.worldPos - camWorld);

  float tEntry, tExit;
  if (!ap_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  // Subtle edge glow on the container faces
  float3 entryLocal = roLocal + rdLocal * max(tEntry, 0.0f);
  float3 faceNrm = ap_faceNormal(entryLocal);
  float fresnel = pow(1.0f - max(0.0f, dot(-rdLocal, faceNrm)), 2.2f);
  float edge = smoothstep(0.16f, 0.02f, ap_edgeDistance(entryLocal));

  // Map entry point to scene space and march
  float iTime = uniforms.time * uniforms.travelSpeed;
  float3 roScene = entryLocal * AP_SCENE_SCALE;
  float3 rdScene = normalize(rdLocal);
  float travelLimit = min((tExit - max(tEntry, 0.0f)) * AP_SCENE_SCALE, AP_MAXD);

  float3 res = ap_trace(roScene, rdScene, travelLimit, iTime);

  float3 color = ap_shade(roScene, rdScene, res, iTime);

  // If ray missed (hit back wall / travelLimit), darken to near-black
  if (res.x < 0.0f) {
    color = float3(0.0f, 0.0f, 0.01f);
  }

  // Very subtle container boundary hints — barely visible
  color += float3(0.04f, 0.08f, 0.16f) * fresnel * 0.08f;
  color += float3(0.20f, 0.30f, 0.50f) * edge * 0.04f;

  return float4(color, 1.0f);
}
