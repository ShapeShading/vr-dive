// GlowingMountainLinesShaders.metal
//
// Source reference:
// https://www.shadertoy.com/view/wcjyDm
// "CC0: Glowing mountain lines"
// Uses XorDev's dot noise: https://www.shadertoy.com/view/wfsyRX
// License: CC0
//
// Adapted for vr-dive: renders inside a view-independent 2 metre cube container.
// The original fixed ShaderToy camera is replaced by visionOS head-pose rays.
// A box intersection constrains the march; the DDA depth state is initialised
// at the box entry so the mountain layer stack is visible from all directions.
//
// Key adaptation notes:
//   - The original uses a DDA that alternates sampling Z.x and Z.z depth arms.
//     Both arms initialise at the box entry depth, with Z.z carrying the
//     time-based phase offset (matching fract(-iTime)/I.z in the original).
//   - Step size is capped at 0.5/max(|I|, 0.05) to prevent degenerate huge
//     jumps when the ray is nearly parallel to an axis. Samples that land
//     outside the box depth range are skipped, so the 120-step budget is
//     never wasted computing outside the visible volume.
//   - I[j] (ray direction component used for the perspective anti-aliasing
//     term) naturally becomes large when the ray is shallow in that direction,
//     which suppresses those arm contributions — matching the original intent.

#include <metal_stdlib>
using namespace metal;

// Maps local box [-1,1] to scene units. Mountains occupy roughly z ∈ [0, 10].
// With tFar ≤ 2 (local) and SCALE = 5: scene depth = 2 × 5 = 10 units. ✓
#define GML_SCENE_SCALE  5.0f
#define GML_STEPS        120

struct GlowingMountainLinesUniforms {
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

struct GlowingMountainLinesVertexOut {
  float4 clipPos  [[position]];
  float3 worldPos;
  uint   viewIndex [[flat]];
};

// ── Vertex shader ─────────────────────────────────────────────────────────────
vertex GlowingMountainLinesVertexOut glowingMountainLinesVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant GlowingMountainLinesUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx  = vertices[vertexID];
  uint viewIndex  = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  GlowingMountainLinesVertexOut out;
  out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos  = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

// ── Box intersection (slab method; handles inside-box camera) ─────────────────
static bool gml_boxHit(
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
fragment float4 glowingMountainLinesFragment(
  GlowingMountainLinesVertexOut in [[stage_in]],
  constant GlowingMountainLinesUniforms &uniforms [[buffer(0)]],
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
  if (!gml_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tNear, tFar)) {
    discard_fragment();
  }

  float tStart = max(tNear, 0.0f);
  float iTime  = uniforms.time;

  // I = normalized ray direction (same role as I in the original ShaderToy)
  float3 I = rdLocal;

  // Scene-space depth bounds
  float zStart = tStart * GML_SCENE_SCALE;
  float zEnd   = tFar   * GML_SCENE_SCALE;

  // Camera origin in scene space. Must be added to every sampled point so that
  // the left and right eyes sample different scene positions — without this the
  // two eyes see identical images and there is no stereo depth perception.
  float3 roScene = roLocal * GML_SCENE_SCALE;

  // ── DDA state vector Z ────────────────────────────────────────────────────
  // Original: Z = fract(-T)/I.z  where T = vec3(0,0,iTime)
  //   → Z.x = 0,  Z.y = 0,  Z.z = fract(-iTime)/I.z
  //
  // VR adaptation: both arms start at box entry depth (zStart / 0.2 = zStart*5),
  // with Z.z carrying the time-based animation phase offset.
  float3 T    = float3(0.0f, 0.0f, iTime);
  float  absIz = max(abs(I.z), 0.1f);

  float3 Z;
  Z.x = zStart * 5.0f;                                    // z_x = 0.2*Z.x = zStart
  Z.y = 0.0f;                                              // unused (j never == 1)
  Z.z = zStart * 5.0f + fract(-iTime) / absIz;            // z_z = zStart + phase

  // Per-iteration DDA step: original Z += 0.5/abs(I)
  // Capped to prevent degenerate huge jumps for near-axis rays.
  float3 dZ = 0.5f / max(abs(I), float3(0.05f));

  // ── Ray march — port of mainImage() from ShaderToy wcjyDm ────────────────
  float4 o  = float4(0.0f);  // accumulated volumetric colour (original: vec4 o)
  float4 O4 = float4(0.0f);  // per-sample colour (original: vec4 O)

  for (int i = 0; i < GML_STEPS; i++) {
    // Pick whichever DDA arm is currently at the smaller depth (original: j = Z.x<Z.z ? 0 : 2).
    // This ensures mountain layers are sampled in depth order and the sampling density
    // adapts to the ray direction — eyes with different I.x/I.z ratios get the same
    // layer count.  The previous j^=2 alternation broke this, causing unequal line
    // counts between left/right eyes and near-field artefacts.
    int j = (Z.x < Z.z) ? 0 : 2;
    float z = 0.2f * Z[j];
    Z += dZ;  // step both arms every iteration (matches original Z+=.5/abs(I))

    // Skip samples outside the box depth range
    if (z < zStart || z > zEnd) continue;

    // 3D world position. roScene carries the per-eye camera offset so that
    // left/right eyes sample different scene points at the same depth → stereo.
    // Original: p = z*I + 0.2*T (camera at origin); VR: p = roScene + z*I + 0.2*T
    float3 p = roScene + z * I + 0.2f * T;

    // Height-field base distance (original: d = p.y)
    float d = p.y;

    // Per-sample colour from position (original: O = 1+sin(.5*p.x+p.z+vec4(2,7,0,2)))
    // This is set as the for-loop init (before the noise octave loop).
    float baseArg = 0.5f * p.x + p.z;
    O4 = 1.0f + sin(baseArg + float4(2.0f, 7.0f, 0.0f, 2.0f));

    // ── 3-octave fractal noise (port of inner for-loop) ──────────────────
    // Original structure:
    //   for(O=...; a>.1; p.xy*=mat2(6,8,-8,6)/8.)
    //     d += a + a*dot(sin(p), cos(p.yzx*1.62)), a*=.5;
    //
    // GLSL mat2(6,8,-8,6) is column-major: col0=(6,8), col1=(-8,6).
    // Row-vector left-multiplication: new.x = 6*p.x+8*p.y, new.y = -8*p.x+6*p.y (then /8).
    float a = 0.6f;
    while (a > 0.1f) {
      // Dot noise by XorDev: d += a + a*dot(sin(p), cos(p.yzx*1.62))
      float3 pyzx = float3(p.y, p.z, p.x) * 1.62f;
      d += a + a * dot(sin(p), cos(pyzx));
      a *= 0.5f;
      // Rotate & scale p.xy (for-loop update step)
      float px = p.x, py = p.y;
      p.x = (6.0f * px + 8.0f * py) / 8.0f;
      p.y = (-8.0f * px + 6.0f * py) / 8.0f;
    }
    // After loop: a ≈ 0.075 (unused); reassign to cosh-based depth falloff.

    // ── Accumulate volumetric contribution ────────────────────────────────
    // Original: a=cosh(8.-z);
    //   o += O.w / (abs(d)*5e2 + .3/I[j]/I[j] + 5./(8.<z?1.:a))
    //            / (8.>z?1.:a) * O;
    //
    // Near-field (z<8):  second divisor = 1, cosh term suppresses bright near-z
    // Far-field  (z>8):  second divisor = cosh(z-8) → rapid falloff
    a = cosh(8.0f - z);
    bool  isFar  = (z > 8.0f);
    float Ij_sq  = max(I[j] * I[j], 0.001f);         // perspective anti-alias term
    float denom  = abs(d) * 500.0f
                 + 0.3f / Ij_sq
                 + 5.0f / (isFar ? 1.0f : a);
    float fadeFar = isFar ? a : 1.0f;
    o += O4.w / denom / fadeFar * O4;
  }

  // Tanh tone mapping (original: O = tanh(o))
  float4 col = tanh(o);
  return float4(col.rgb, 1.0f);
}
