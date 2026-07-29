// SynthwaveSunsetShaders.metal
//
// Source: ShaderToy "another synthwave sunset thing"
// https://www.shadertoy.com/view/tsScRK
//
// VR adaptation notes:
// - Converted from mainImage full-screen rendering to a view-aligned box mesh.
// - Stereo, audio texture sampling, and AA branches are removed for visionOS cost.
// - The raymarch scene uses the real headset view rays while preserving the original
//   trinoise terrain / sun / starfield look.

#include <metal_stdlib>
using namespace metal;

struct SynthwaveSunsetUniforms {
  float time;
  uint viewCount;
  float sceneSpeed;
  float _pad;
  float4 objectCenter;
  float4 viewerOffset;
  float4 boxHalfExtents;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct SynthwaveVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex SynthwaveVertexOut synthwaveSunsetVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant SynthwaveSunsetUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.boxHalfExtents.xyz + uniforms.objectCenter.xyz;

  SynthwaveVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

#define SW_SPEED 10.0f

static float sw_amp(float2 p) {
  return smoothstep(1.0f, 8.0f, abs(p.x));
}

static float sw_pow512(float a) {
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  return a * a;
}

static float sw_pow1d5(float a) {
  return a * sqrt(max(a, 0.0f));
}

static float sw_hash21(float2 co) {
  return fract(sin(dot(co, float2(1.9898f, 7.233f))) * 45758.5433f);
}

static float sw_hash(float2 uv, float t) {
  float a = sw_amp(uv);
  float w = a > 0.0f
    ? (1.0f - 0.4f * sw_pow512(0.51f + 0.49f * sin((0.02f * (uv.y + 0.5f * uv.x) - t) * 2.0f)))
    : 0.0f;
  return a > 0.0f ? a * sw_pow1d5(sw_hash21(uv)) * w : 0.0f;
}

static float sw_edgeMin(float dx, float2 da, float2 db, float2 uv) {
  uv.x += 5.0f;
  float3 c = fract(round(float3(uv, uv.x + uv.y)) * (float3(0.0f, 1.0f, 2.0f) + 0.61803398875f));
  float a1 = sw_hash21(float2(c.y, c.x + 17.0f)) > 0.6f ? 0.15f : 1.0f;
  float a2 = sw_hash21(float2(c.x, c.z + 23.0f)) > 0.6f ? 0.15f : 1.0f;
  float a3 = sw_hash21(float2(c.z, c.y + 31.0f)) > 0.6f ? 0.15f : 1.0f;
  return min(min((1.0f - dx) * db.y * a3, da.x * a2), da.y * a1);
}

static float2 sw_trinoise(float2 uv, float t) {
  const float sq = 1.22474487139f;  // sqrt(3/2)
  uv.x *= sq;
  uv.y -= 0.5f * uv.x;
  float2 d = fract(uv);
  uv -= d;

  bool c = dot(d, float2(1.0f)) > 1.0f;
  float2 dd = 1.0f - d;
  float2 da = c ? dd : d;
  float2 db = c ? d : dd;

  float nn = sw_hash(uv + float(c), t);
  float n2 = sw_hash(uv + float2(1.0f, 0.0f), t);
  float n3 = sw_hash(uv + float2(0.0f, 1.0f), t);

  float nmid = mix(n2, n3, d.y);
  float ns = mix(nn, c ? n2 : n3, da.y);
  float dx = da.x / max(db.y, 1e-4f);
  return float2(mix(ns, nmid, dx), sw_edgeMin(dx, da, db, uv + d));
}

static float2 sw_map(float3 p, float t) {
  float2 n = sw_trinoise(p.xz, t);
  return float2(p.y - 2.0f * n.x, n.y);
}

static float3 sw_grad(float3 p, float t, float eps) {
  const float2 base = float2(1.0f, 0.0f);
  float2 e = base * eps;
  float a = sw_map(p, t).x;
  return float3(
    sw_map(p + e.xyy, t).x - a,
    sw_map(p + e.yxy, t).x - a,
    sw_map(p + e.yyx, t).x - a) / e.x;
}

static float2 sw_intersect(float3 ro, float3 rd, float t, int maxSteps, float maxDist, float hitScale) {
  float d = 0.0f;
  float h = 0.0f;
  for (int i = 0; i < maxSteps; ++i) {
    float3 p = ro + d * rd;
    float2 s = sw_map(p, t);
    h = s.x;
    d += h * 0.5f;
    if (abs(h) < hitScale * max(d, 1.0f)) {
      return float2(d, s.y);
    }
    if (d > maxDist || p.y > 2.5f) {
      break;
    }
  }
  return float2(-1.0f);
}

static float sw_frontDetail(float3 rd) {
  float front = smoothstep(-0.15f, 0.35f, rd.z);
  float side = 1.0f - smoothstep(0.35f, 0.95f, abs(rd.x));
  float down = 1.0f - smoothstep(0.2f, 0.75f, rd.y);
  return clamp(max(front * side, 0.18f) * mix(0.7f, 1.0f, down), 0.18f, 1.0f);
}

static void sw_addSun(float3 rd, float3 ld, thread float3 &col) {
  float sun = smoothstep(0.21f, 0.2f, distance(rd, ld));
  if (sun > 0.0f) {
    float yd = rd.y - ld.y;
    float a = sin(3.1f * exp(-yd * 14.0f));
    sun *= smoothstep(-0.8f, 0.0f, a);
    col = mix(col, float3(1.0f, 0.8f, 0.4f) * 0.75f, sun);
  }
}

static float sw_starNoise(float3 rd) {
  float c = 0.0f;
  float3 p = normalize(rd) * 300.0f;
  for (float i = 0.0f; i < 2.0f; i += 1.0f) {
    float3 q = fract(p) - 0.5f;
    float3 id = floor(p);
    float c2 = smoothstep(0.5f, 0.0f, length(q));
    c2 *= step(sw_hash21(id.xz / max(abs(id.y), 1.0f)), 0.06f - i * i * 0.005f);
    c += c2;
    p = p * 0.6f + 0.5f * p * float3x3(
      float3(3.0f / 5.0f, 0.0f, 4.0f / 5.0f),
      float3(0.0f, 1.0f, 0.0f),
      float3(-4.0f / 5.0f, 0.0f, 3.0f / 5.0f));
  }
  c *= c;
  float g = dot(sin(rd * 10.512f), cos(rd.yzx * 10.512f));
  c *= smoothstep(-3.14f, -0.9f, g) * 0.5f + 0.5f * smoothstep(-0.3f, 1.0f, g);
  return c * c;
}

static float3 sw_sky(float3 rd, float3 ld, bool mask, float detail) {
  float haze = exp2(-(3.2f + 1.2f * detail) * (abs(rd.y) - 0.18f * dot(rd, ld)));
  float st = (mask && detail > 0.55f) ? sw_starNoise(rd) * detail * (1.0f - min(haze, 1.0f)) : 0.0f;
  float horizonGlow = exp2(-(0.08f + 0.08f * detail) * abs(length(rd.xz) / max(abs(rd.y), 0.08f)));
  float3 back = float3(0.32f, 0.08f, 0.52f) * (1.0f - 0.18f * horizonGlow * max(sign(rd.y), 0.0f));
  float3 col = clamp(mix(back, float3(0.7f, 0.1f, 0.4f), haze) + st, 0.0f, 1.0f);
  if (mask) {
    float3 sunCol = col;
    sw_addSun(rd, ld, sunCol);
    col = mix(col, sunCol, 0.4f + 0.6f * detail);
  }
  return col;
}

fragment float4 synthwaveSunsetFragment(
  SynthwaveVertexOut in [[stage_in]],
  constant SynthwaveSunsetUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 worldRay = normalize(in.worldPos - camWorld);
  float3 rd = normalize(float3(worldRay.x, worldRay.y, -worldRay.z));
  float detail = sw_frontDetail(rd);

  float t = fmod(uniforms.time, 4000.0f);
  float3 viewerOffset = uniforms.viewerOffset.xyz;
  float3 ro = float3(
    viewerOffset.x * 8.0f,
    1.0f + viewerOffset.y * 3.0f,
    -160.0f + t * uniforms.sceneSpeed + viewerOffset.z * 8.0f);

  bool traceTerrain = (rd.y < 0.32f) || (detail > 0.55f);
  int maxSteps = int(mix(36.0f, 96.0f, detail));
  float maxDist = mix(70.0f, 130.0f, detail);
  float hitScale = mix(0.008f, 0.0035f, detail);
  float2 hit = traceTerrain ? sw_intersect(ro, rd, t, maxSteps, maxDist, hitScale) : float2(-1.0f);
  float d = hit.x;
  float3 ld = normalize(float3(0.0f, 0.125f + 0.05f * sin(0.1f * t), 1.0f));

  float3 sky = sw_sky(rd, ld, d < 0.0f, detail);
  float3 col = sky;

  if (d > 0.0f) {
    float3 fog = exp2(-d * mix(float3(0.18f, 0.13f, 0.32f), float3(0.14f, 0.1f, 0.28f), detail));
    float3 p = ro + d * rd;
    float3 n = normalize(sw_grad(p, t, mix(0.03f, 0.012f, detail)));

    float diff = max(dot(n, ld) + 0.1f * n.y, 0.0f);
    col = float3(0.1f, 0.11f, 0.18f) * diff;

    float3 reflectedRay = reflect(rd, n);
    float3 reflectedColor = sw_sky(reflectedRay, ld, detail > 0.55f, detail * 0.85f);
    float fresnel = (0.02f + 0.45f * detail) * pow(max(1.0f + dot(rd, n), 0.0f), 5.0f);
    col = mix(col, reflectedColor, fresnel);
    col = mix(col, float3(0.8f, 0.1f, 0.92f), smoothstep(0.03f, 0.0f, hit.y) * detail);
    col = mix(sky, col, fog);
  }

  col = sqrt(clamp(col, 0.0f, 1.0f));
  return float4(col, 1.0f);
}