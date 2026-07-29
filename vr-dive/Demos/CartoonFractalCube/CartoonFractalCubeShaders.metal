// CartoonFractalCubeShaders.metal
//
// Original cube-portal cartoon fractal scene with normal-based color and dark edges.
// Visual inspiration requested from ShaderToy XsBXWt:
// https://www.shadertoy.com/view/XsBXWt
// This implementation is original and does not reuse source code from the
// reference shader.

#include <metal_stdlib>
using namespace metal;

#define CFC_RAY_STEPS   150
#define CFC_MAX_DIST    25.0f
#define CFC_DETAIL_EPS  0.001f
#define CFC_SCENE_SCALE 12.0f

struct CartoonFractalCubeUniforms {
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

struct CartoonFractalCubeVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

struct CfcMapSample {
  float dist;
  float edgeShape;
};

vertex CartoonFractalCubeVertexOut cartoonFractalCubeVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant CartoonFractalCubeUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  CartoonFractalCubeVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float2 cfc_rot(float2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float cfc_mod(float x, float y) {
  return x - y * floor(x / y);
}

static float4 cfc_formula(float4 p) {
  p.xz = abs(p.xz + 1.0f) - abs(p.xz - 1.0f) - p.xz;
  p.y -= 0.25f;
  p.xy = cfc_rot(p.xy, 0.61086524f);
  float denom = clamp(dot(p.xyz, p.xyz), 0.2f, 1.0f);
  p *= 2.0f / denom;
  return p;
}

static CfcMapSample cfc_map(float3 pos, float time) {
  pos.y += sin(pos.z - time * 0.6f) * 0.15f;

  float3 tpos = pos;
  tpos.z = abs(3.0f - cfc_mod(tpos.z, 6.0f));
  float4 p = float4(tpos, 1.0f);
  float edgeShape = 0.0f;
  for (int i = 0; i < 4; ++i) {
    p = cfc_formula(p);
    edgeShape += exp(-1.4f * abs(p.y));
  }

  float fractal = (length(max(float2(0.0f), p.yz - 1.5f)) - 1.0f) / p.w;

  float ribs = max(abs(pos.x + 1.0f) - 0.3f, pos.y - 0.35f);
  ribs = max(ribs, -max(abs(pos.x + 1.0f) - 0.1f, pos.y - 0.5f));

  float stripes = abs(0.25f - cfc_mod(pos.z, 0.5f));
  ribs = max(ribs, -max(abs(stripes) - 0.2f, pos.y - 0.3f));
  ribs = max(ribs, -max(abs(stripes) - 0.01f, -pos.y + 0.32f));

  CfcMapSample sample;
  sample.dist = min(fractal, ribs);
  sample.edgeShape = edgeShape;
  return sample;
}

static float cfc_de(float3 pos, float time) {
  return cfc_map(pos, time).dist;
}

static float3 cfc_normal(float3 p, float time) {
  float e = CFC_DETAIL_EPS * 4.0f;
  float dx = cfc_de(p + float3(e, 0, 0), time) - cfc_de(p - float3(e, 0, 0), time);
  float dy = cfc_de(p + float3(0, e, 0), time) - cfc_de(p - float3(0, e, 0), time);
  float dz = cfc_de(p + float3(0, 0, e), time) - cfc_de(p - float3(0, 0, e), time);
  return normalize(float3(dx, dy, dz));
}

static float cfc_edgeMetric(float3 p, float time, float det) {
  float3 e = float3(0.0f, det * 5.0f, 0.0f);
  float d1 = cfc_de(p - e.yxx, time);
  float d2 = cfc_de(p + e.yxx, time);
  float d3 = cfc_de(p - e.xyx, time);
  float d4 = cfc_de(p + e.xyx, time);
  float d5 = cfc_de(p - e.xxy, time);
  float d6 = cfc_de(p + e.xxy, time);
  float d = cfc_de(p, time);
  float edge = abs(d - 0.5f * (d2 + d1))
             + abs(d - 0.5f * (d4 + d3))
             + abs(d - 0.5f * (d6 + d5));
  return min(1.0f, pow(edge, 0.55f) * 15.0f);
}

static float3 cfc_path(float time) {
  float ti = time * 1.5f;
  return float3(
    sin(ti),
    (1.0f - sin(ti * 2.0f)) * 0.5f,
    ti * 5.0f) * 0.5f;
}

static void cfc_applyPath(thread float3 &ro, thread float3 &rd, float time) {
  float3 go = cfc_path(time);
  float3 adv = cfc_path(time + 0.7f);
  float3 advec = normalize(adv - go);

  float an = adv.x - go.x;
  an *= min(1.0f, abs(adv.z - go.z)) * sign(adv.z - go.z) * 0.7f;
  rd.xy = cfc_rot(rd.xy, an);

  an = advec.y * 1.7f;
  rd.yz = cfc_rot(rd.yz, an);

  an = atan2(advec.x, advec.z);
  rd.xz = cfc_rot(rd.xz, an);

  ro += float3(-1.0f, 0.7f, 0.0f) + go;
}

static float3 cfc_sky(float3 rd, float time) {
  float3 skyDir = rd;
  skyDir.y -= 0.02f;
  float sunSize = 6.2f;
  float angle = atan2(skyDir.x, skyDir.y) + time * 1.5f;
  float spoke = abs(0.2f - cfc_mod(angle, 0.4f));
  float radial = length(skyDir.xy);
  float sun = pow(clamp(1.0f - radial * sunSize - spoke, 0.0f, 1.0f), 0.1f);
  float sunBorder = pow(clamp(1.0f - radial * (sunSize - 0.2f) - spoke, 0.0f, 1.0f), 0.1f);
  float rays = pow(clamp(1.0f - radial * (sunSize - 4.5f) - 0.5f * spoke, 0.0f, 1.0f), 3.0f);
  float y = mix(0.45f, 1.2f, pow(smoothstep(0.0f, 1.0f, 0.75f - skyDir.y), 2.0f)) * (1.0f - sunBorder * 0.5f);

  float3 backg = float3(0.5f, 0.0f, 1.0f)
    * ((1.0f - sun) * (1.0f - rays) * y + (1.0f - sunBorder) * rays * float3(1.0f, 0.8f, 0.15f) * 3.0f);
  backg += float3(1.0f, 0.9f, 0.1f) * sun;
  backg = max(backg, rays * float3(1.0f, 0.9f, 0.5f));
  return backg;
}

static bool cfc_boxHit(
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

fragment float4 cartoonFractalCubeFragment(
  CartoonFractalCubeVertexOut in [[stage_in]],
  constant CartoonFractalCubeUniforms &uniforms [[buffer(0)]],
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
  if (!cfc_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float time = uniforms.time * uniforms.travelSpeed * 0.5f;
  float3 ro = (roLocal + rdLocal * max(tEntry, 0.0f)) * CFC_SCENE_SCALE;
  float3 rd = rdLocal;
  cfc_applyPath(ro, rd, time);

  float dist = 0.0f;
  float d = 100.0f;
  float det = CFC_DETAIL_EPS;
  float3 p = ro;
  bool hit = false;
  for (int i = 0; i < CFC_RAY_STEPS; ++i) {
    if (d <= det || dist >= CFC_MAX_DIST) { break; }
    p = ro + dist * rd;
    d = cfc_de(p, time);
    det = CFC_DETAIL_EPS * exp(0.13f * dist);
    dist += d;
    if (d <= det) { hit = true; }
  }

  float3 sky = cfc_sky(rd, time);
  float3 color = sky;
  if (hit && dist < CFC_MAX_DIST) {
    p -= (det - d) * rd;
    float3 norm = cfc_normal(p, time);
    float edge = cfc_edgeMetric(p, time, det);
    float3 base = (1.0f - abs(norm)) * max(0.0f, 1.0f - edge * 0.8f);
    float3 warm = float3(1.0f, 0.9f, 0.3f);
    float fade = exp(-0.004f * dist * dist);
    color = mix(warm, base, fade);
    color = mix(color, sky, smoothstep(18.0f, CFC_MAX_DIST, dist));
    color *= float3(1.0f, 0.9f, 0.85f);
  }

  color = pow(max(color, 0.0f), float3(1.4f)) * 1.2f;
  float luminance = dot(color, float3(0.299f, 0.587f, 0.114f));
  color = mix(float3(luminance), color, 0.65f);
  color = clamp(color, 0.0f, 1.0f);
  return float4(color, 1.0f);
}