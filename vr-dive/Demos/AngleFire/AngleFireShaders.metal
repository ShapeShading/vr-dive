// AngleFireShaders.metal
//
// Source reference:
// https://www.shadertoy.com/view/3XXSDB
// "Angel" by @XorDev — an experiment based on "3D Fire":
//   https://www.shadertoy.com/view/3XXSWS
// License: see original ShaderToy page
//
// Adapted for vr-dive: renders inside a view-independent 2 metre cube container.
// The original fixed ShaderToy camera is replaced by visionOS head-pose ray marching.
// A box intersection constrains the volumetric march; the fire column (cylinder SDF
// along the Y axis) is centred at the box origin and visible from all directions.

#include <metal_stdlib>
using namespace metal;

// ── Tuning ────────────────────────────────────────────────────────────────────
// Maps local box coords [-1,1] → scene units. Cylinder radius is 0.5 in scene
// units; turbulence displaces ~±5 units. Scale 4 gives the best view density.
#define AF_SCENE_SCALE  4.0f
#define AF_STEPS        100
#define AF_MAX_T        30.0f   // max march distance cap in local box units

// ── Structs ───────────────────────────────────────────────────────────────────
struct AngleFireUniforms {
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

struct AngleFireVertexOut {
  float4 clipPos  [[position]];
  float3 worldPos;
  uint   viewIndex [[flat]];
};

// ── Vertex shader ─────────────────────────────────────────────────────────────
vertex AngleFireVertexOut angleFireVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant AngleFireUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx   = vertices[vertexID];
  uint viewIndex   = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos  = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  AngleFireVertexOut out;
  out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos  = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

// ── Box intersection (slab method; handles inside-box camera) ─────────────────
static bool af_boxHit(
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

// ── Fragment shader ───────────────────────────────────────────────────────────
fragment float4 angleFireFragment(
  AngleFireVertexOut in [[stage_in]],
  constant AngleFireUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi         = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w   = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center  = uniforms.objectCenter.xyz;
  float  halfSize = uniforms.cubeScale;

  // Ray in local cube space [-1, 1]^3
  float3 roLocal = (camWorld - center) / halfSize;
  float3 rdLocal = normalize(in.worldPos - camWorld);

  float tNear, tFar;
  if (!af_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tNear, tFar)) {
    discard_fragment();
  }

  // Map to scene space.  The fire column is a cylinder of radius 0.5 along Y,
  // centred at the scene origin — visible from all viewing angles.
  float3 ro = roLocal * AF_SCENE_SCALE;
  float3 rd = normalize(rdLocal);

  float iTime = uniforms.time;

  // ── Volumetric ray march — port of mainImage() from ShaderToy 3XXSDB ─────
  //
  // Each outer step:
  //   1. Twist p.xz with a y-dependent non-orthogonal rotation (the "angel" shape)
  //   2. Add 10-octave turbulence driven by time
  //   3. Evaluate distorted cylinder SDF as the march step size
  //   4. Accumulate additive glow: brightness ∝ 1/SDF, colour cycles with depth z
  //
  // The original zeros O with `O *= i` (i=0 at start); here we just init to 0.

  float4 O   = float4(0.0f);
  float  z   = max(tNear, 0.0f) * AF_SCENE_SCALE;       // scene-space march depth
  float  zEnd = min(tFar, AF_MAX_T) * AF_SCENE_SCALE;
  float  d;

  for (int i = 0; i < AF_STEPS && z < zEnd; i++) {
    float3 p = ro + rd * z;

    // Twist shape: y-dependent rotation in xz plane.
    // GLSL original: p.xz *= mat2(cos(p.y*.5 + vec4(0,33,11,0)))
    // mat2(a,b,c,d) in GLSL is col-major: col0=(a,b), col1=(c,d).
    // p.xz = p.xz * M: new.x = dot(p.xz, col0), new.z = dot(p.xz, col1).
    float py = p.y * 0.5f;
    float2x2 twistMat = float2x2(
      float2(cos(py),        cos(py + 33.0f)),   // col0
      float2(cos(py + 11.0f), cos(py)));          // col1
    p.xz = p.xz * twistMat;

    // Turbulence distortion loop (10 octaves, d: 1.0 → ~9.3 via ×1.25 each step).
    // GLSL: for(d=1.; d<9.; d/=.8)  p += cos((p.yzx - t*vec3(3,1,0))*d)/d;
    float3 tdir = float3(3.0f, 1.0f, 0.0f);
    for (d = 1.0f; d < 9.0f; d /= 0.8f)
      p += cos((p.yzx - iTime * tdir) * d) / d;

    // Distorted cylinder SDF (radius 0.5) as the step size.
    // Small d near the cylinder → many samples → bright glow.
    z += d = (0.1f + abs(length(p.xz) - 0.5f)) / 20.0f;

    // Additive glow: depth-based rainbow colour, amplitude ∝ 1/SDF.
    O += (sin(z + float4(2.0f, 3.0f, 4.0f, 0.0f)) + 1.1f) / d;
  }

  // Tanh tone mapping — matches original `tanh(O / 4e3)`.
  float3 col = tanh(O.rgb / 4000.0f);
  return float4(col, 1.0f);
}
