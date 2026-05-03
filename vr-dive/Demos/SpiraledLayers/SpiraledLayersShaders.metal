// SpiraledLayersShaders.metal
//
// Source reference:
// https://www.shadertoy.com/view/Ns3XWf
// "Spiraled Layers" shader (original author unknown)
// License: see original ShaderToy page
//
// Adapted for vr-dive: renders inside a view-independent 2 metre cube container.
// The original fixed ShaderToy camera is replaced by visionOS head-pose ray marching.
// Each ray is box-intersected with the cube, then marched through the scene in local space
// scaled by SPL_SCENE_SCALE. A +3 scene-x offset centres the spiral content in the cube.

#include <metal_stdlib>
using namespace metal;

// ── Tuning constants ──────────────────────────────────────────────────────────
#define SPL_PI          3.14159265358979323846f
#define SPL_STEPS       100          // ray march iterations (original: 200)
#define SPL_SHAD_STEPS  16           // soft-shadow iterations (original: 64)
#define SPL_MDIST       80.0f        // max march distance in scene units
#define SPL_SCENE_SCALE 5.0f         // local [-1,1] → scene units; matches original apparent scale

// ── Structs ───────────────────────────────────────────────────────────────────
struct SpiraledLayersUniforms {
  float  time;
  uint   viewCount;
  float  cubeScale;
  float  padding;
  float4 objectCenter;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct SpiraledLayersVertexOut {
  float4 clipPos  [[position]];
  float3 worldPos;
  uint   viewIndex [[flat]];
};

// ── Vertex shader ─────────────────────────────────────────────────────────────
vertex SpiraledLayersVertexOut spiraledLayersVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant SpiraledLayersUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex  = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  SpiraledLayersVertexOut out;
  out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos  = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

// ── Box intersection (slab method; handles inside-box case) ───────────────────
static bool spl_boxHit(
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

// ── Scene helpers (ported 1-to-1 from GLSL) ──────────────────────────────────

// rot(a) = mat2(cos,sin,-sin,cos); used as p *= rot(a) (row-vector × matrix → rotate by -a)
static float2x2 spl_rot(float a) {
  float c = cos(a), s = sin(a);
  return float2x2(float2(c, s), float2(-s, c));
}

// Extruded 1-D SDF: extrude line SDF 's' into a slab of half-height h along p.y
static float spl_ext(float3 p, float s, float h) {
  float2 b = float2(s, abs(p.y) - h);
  return min(max(b.x, b.y), 0.0f) + length(max(b, 0.0f));
}

// Pseudo-random hash (original: h11)
static float spl_h11(float a) {
  a += 0.65343f;
  return fract(fract(a * a * 12.9898f) * 43758.5453123f);
}

// Distance to next lane-boundary plane along the ray; returns large value when
// rd.z ≈ 0 to prevent divide-by-zero (ray parallel to boundary → no artifact)
static float spl_diplane(float3 p, float3 b, float3 rd) {
  if (abs(rd.z) < 1e-6f) return 1e6f;
  float3 dir = sign(rd) * b;
  float3 rc  = (dir - p) / rd;
  return rc.z + 0.01f;
}

// Tiled repetition clamped to [lima, limb] cells
static float spl_lim(float p, float s, float lima, float limb) {
  return p - s * clamp(round(p / s), lima, limb);
}
static float spl_idlim(float p, float s, float lima, float limb) {
  return clamp(round(p / s), lima, limb);
}
static float spl_lim2(float p, float s, float limb) {
  return p - s * min(round(p / s), limb);
}
static float spl_idlim2(float p, float s, float limb) {
  return min(round(p / s), limb);
}

// Spiral SDF — port of spiral() from ShaderToy Ns3XWf
static float spl_spiral(float2 p, float t, float m, float scale, float size, float expand) {
  size -= expand - 0.01f;
  t     = max(t, 0.0f);

  // Offset spiral to the left
  p.x += SPL_PI * -t * (m + m * (-t - 1.0f));
  t   -= 0.25f;

  float2 po = p;

  // Move spiral up and counter-rotate
  p.y += -t * m - m * 0.5f;
  p    = p * spl_rot(t * SPL_PI * 2.0f + SPL_PI * 0.5f);

  // Polar map
  float theta = atan2(p.y, p.x);
  theta = clamp(theta, -SPL_PI, SPL_PI);
  p     = float2(theta, length(p));

  // Create spiral: offset radial by angle
  p.y += theta * scale * 0.5f;

  // Tile spiral rings
  float py = p.y;
  p.y = spl_lim(p.y, m, 0.0f, floor(t));

  // Line SDF of the spiral
  float a = abs(p.y) - size;

  // Moving outer spiral segment
  p.y  = py;
  p.x -= SPL_PI;
  p.y -= (floor(t) + 1.5f) * m - m * 0.5f;
  float b = max(abs(p.y), abs(p.x) - (SPL_PI * 2.0f) * fract(t) + size);

  // Unrolled line SDF
  a = min(a, b - size);
  b = abs(po.y) - size;
  b = max(po.x * 30.0f, b);

  a = min(a, b);
  return a;
}

// Scene SDF — port of map() from ShaderToy Ns3XWf
// Returns float3(distance, id_flag, c_for_AO_and_soft_shadow)
//   id_flag = 1 → real spiral surface; 0 → lane-boundary artifact plane
//   c_for_AO = SDF value without boundary clipping (used for soft shadows / AO)
// rdg: active ray direction needed by spl_diplane for artifact removal
static float3 spl_map(float3 p, float iTime, float3 rdg) {
  float2 a = float2(1.0f);
  float2 b = float2(1.0f);
  float  c = 0.0f;
  float  t = iTime;

  const float size     = 0.062f;
  const float scale    = size - 0.01f;          // 0.052
  const float expand   = 0.04f;
  const float m2       = size * 6.0f;
  const float m        = SPL_PI * scale;         // ≈ 0.1634
  const float ltime    = 10.0f;
  const float width    = 0.5f;
  const float count    = 6.0f;
  const float modwidth = width * 2.0f + 0.10f;  // ≈ 1.1

  // Scroll upward with time; offset in x to frame the scene
  p.y -= (t / ltime) * size * 6.0f;
  p.x -= 3.0f;

  // Tile in z; randomise per-lane timing
  float id3 = spl_idlim(p.z, modwidth, -count, count);
  t        += spl_h11(id3 * 0.76f) * 8.0f;
  p.z       = spl_lim(p.z, modwidth, -count, count);

  float  to = t;
  float3 po = p;

  // ── Spiral 1 ──
  float stack = -floor(t / ltime);
  float id2   = spl_idlim2(p.y, m2, stack);
  t  += id2 * ltime;
  p.y = spl_lim2(p.y, m2, stack);
  a.x = spl_spiral(p.xy, t, m, scale, size, expand);
  c   = a.x;
  a.x = min(a.x, max(p.y + size * 5.0f, p.x));  // artifact removal

  // ── Spiral 2 ──
  p = po;  t = to;
  p.y += size * 2.0f;
  t   -= ltime / 3.0f;
  stack = -floor(t / ltime);
  id2   = spl_idlim2(p.y, m2, stack);
  t  += id2 * ltime;
  p.y = spl_lim2(p.y, m2, stack);
  b.x = spl_spiral(p.xy, t, m, scale, size, expand);
  c   = min(c, b.x);
  a   = (a.x < b.x) ? a : b;
  a.x = min(a.x, max(p.y + size * 5.0f, p.x));  // artifact removal

  // ── Spiral 3 ──
  p = po;  t = to;
  p.y += size * 4.0f;
  t   -= 2.0f * ltime / 3.0f;
  stack = -floor(t / ltime);
  id2   = spl_idlim2(p.y, m2, stack);
  t  += id2 * ltime;
  p.y = spl_lim2(p.y, m2, stack);
  b.x = spl_spiral(p.xy, t, m, scale, size, expand);
  c   = min(c, b.x);
  a   = (a.x < b.x) ? a : b;
  a.x = min(a.x, max(p.y + size * 5.0f, p.x));  // artifact removal

  // Extrude spirals in z (lane direction). po.yzx = (po.y, po.z, po.x).
  float halfLane = width - expand * 0.5f + 0.02f;
  a.x = spl_ext(float3(po.y, po.z, po.x), a.x, halfLane) - expand;
  c   = spl_ext(float3(po.y, po.z, po.x), c,   halfLane) - expand;

  // Lane-boundary artifact removal via diplane
  b.x = spl_diplane(po, float3(modwidth) * 0.5f, rdg);
  b.y = 0.0f;
  a   = (a.x < b.x) ? a : b;

  return float3(a, c);
}

// Surface normal via forward-difference gradient
static float3 spl_norm(float3 p, float iTime, float3 rdg) {
  const float e = 0.01f;
  float base = spl_map(p, iTime, rdg).x;
  return normalize(base - float3(
    spl_map(p - float3(e, 0.0f, 0.0f), iTime, rdg).x,
    spl_map(p - float3(0.0f, e, 0.0f), iTime, rdg).x,
    spl_map(p - float3(0.0f, 0.0f, e), iTime, rdg).x));
}

// ── Fragment shader ───────────────────────────────────────────────────────────
fragment float4 spiraledLayersFragment(
  SpiraledLayersVertexOut in [[stage_in]],
  constant SpiraledLayersUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w     = v2wMats[vi];
  float3 camWorld  = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center   = uniforms.objectCenter.xyz;
  float  halfSize = uniforms.cubeScale;

  // Ray in local cube space [-1, 1]^3
  float3 roLocal = (camWorld - center) / halfSize;
  float3 rdLocal = normalize(in.worldPos - camWorld);

  float tNear, tFar;
  if (!spl_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tNear, tFar)) {
    discard_fragment();
  }
  float tStart = max(tNear, 0.0f);

  // Map to scene space. +3 in x centres the spiral content (map() applies p.x -= 3 internally).
  const float3 SCENE_OFFSET = float3(3.0f, 0.0f, 0.0f);
  float3 ro  = roLocal * SPL_SCENE_SCALE + SCENE_OFFSET;
  float3 rd  = normalize(rdLocal);
  float  tMaxScene = min(tFar, 100.0f) * SPL_SCENE_SCALE;

  float  iTime = uniforms.time;
  float3 rdg   = rd;  // diplane artifact-removal direction tracks the active marching ray

  // ── Ray march ──
  float3 p   = ro;
  float3 d   = float3(0.0f);
  float  dO  = tStart * SPL_SCENE_SCALE;
  bool   hit = false;

  for (int i = 0; i < SPL_STEPS; ++i) {
    p  = ro + rd * dO;
    d  = spl_map(p, iTime, rdg);
    dO += d.x;
    if (d.x < 0.001f) { hit = true; break; }
    if (dO > SPL_MDIST || dO > tMaxScene) break;
  }

  float3 col;

  if (hit && d.y != 0.0f) {
    // ── Lighting ──
    float3 ld = normalize(float3(0.5f, 0.4f, 0.9f));
    float3 n  = spl_norm(p, iTime, rdg);

    // Soft shadow (rdg updated to ld so diplane uses light direction)
    rdg = ld;
    float shadow = 1.0f;
    float h      = 0.09f;
    for (int i = 0; i < SPL_SHAD_STEPS; ++i) {
      float3 dd = spl_map(p + ld * h + n * 0.005f, iTime, rdg);
      if (dd.x < 0.001f && dd.y == 0.0f) break;   // lane boundary — not a real occluder
      if (dd.x < 0.001f) { shadow = 0.0f; break; } // real surface blocks light
      shadow = min(shadow, dd.z * 30.0f);
      if (h > 7.0f) break;
      h += dd.x;
    }
    shadow = max(shadow, 0.8f);

    // AO: smoothstep(-a, a, map(p + n*a).z) at two offsets (matches original macro)
    float ao = max(
      smoothstep(-0.05f, 0.05f, spl_map(p + n * 0.05f, iTime, rdg).z) *
      smoothstep(-0.10f, 0.10f, spl_map(p + n * 0.10f, iTime, rdg).z),
      0.1f);

    // Normal-based colour with hue rotation in xz plane (matches original n.xz *= rot(4π/3))
    n.xz = n.xz * spl_rot(4.0f * SPL_PI / 3.0f);
    col  = n * 0.5f + 0.5f;
    col  = col * shadow * ao;

  } else {
    // Background: vertical gradient (purple-blue sky, matches original render())
    col = mix(float3(0.355f, 0.129f, 0.894f),
              float3(0.278f, 0.953f, 1.000f),
              clamp((rd.y + 0.05f) * 2.0f, -0.15f, 1.5f));
  }

  // Gamma ≈ 2.0 approximation (matches original sqrt(col))
  col = sqrt(max(col, 0.0f));
  return float4(col, 1.0f);
}
