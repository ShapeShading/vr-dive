// MagnetarShaders.metal
//
// Source reference:
// https://www.shadertoy.com/view/NclXWn
// "Magnetar" — reworked from https://www.shadertoy.com/view/XfK3zV
// License: see original ShaderToy page
//
// Adapted for vr-dive: renders inside a view-independent 2 metre cube container.
// The original ShaderToy camera orbit is replaced by visionOS head-pose ray marching.

#include <metal_stdlib>
using namespace metal;

#define MAG_STEPS  100
#define MAG_FAR    40.0f
#define MAG_NEAR   1e-3f

struct MagnetarUniforms {
  float time;
  uint  viewCount;
  float cubeScale;
  float padding;
  float4 objectCenter;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct MagnetarVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint   viewIndex [[flat]];
};

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------
vertex MagnetarVertexOut magnetarVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant MagnetarUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  MagnetarVertexOut out;
  out.clipPos  = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

// ---------------------------------------------------------------------------
// Box intersection  (slab method, handles inside-box case)
// ---------------------------------------------------------------------------
static bool mag_boxHit(
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

// ---------------------------------------------------------------------------
// Oscillate helper  (GLSL: O(x,a,b) = (cos(x*6.2832)*.5+.5)*(a-b)+b)
// ---------------------------------------------------------------------------
static float mag_oscillate(float x, float a, float b) {
  return (cos(x * 6.2832f) * 0.5f + 0.5f) * (a - b) + b;
}

// ---------------------------------------------------------------------------
// SDF  (port of map() from ShaderToy NclXWn)
// ---------------------------------------------------------------------------
static float mag_map(float3 p, float T)
{
  float density = mag_oscillate(T / 8.0f, 10.0f, 30.0f);
  float b = (dot(p, p) - 1.0f) / density;  // coordinate transform

  // Inside or right at the unit sphere b≤0: the coordinate transform p/=b
  // would divide by zero or flip the sign of all coordinates, producing
  // completely wrong tube distances (white blob / red arcs artifact).
  // Return a large step so the march passes cleanly through the sphere.
  if (b <= 0.05f) return 1.0f;

  p /= b;
  float x = T + round(p.x - T);   // tile and move along x
  p.x -= x;

  float s = min(b, length(p.xz - round(p.xz)) + 0.05f);  // tubes
  return s;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------
fragment float4 magnetarFragment(
  MagnetarVertexOut in [[stage_in]],
  constant MagnetarUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center = uniforms.objectCenter.xyz;
  float halfSize = uniforms.cubeScale;

  // Ray in local cube space  [-1, 1]^3
  float3 roLocal = (camWorld - center) / halfSize;
  float3 rdLocal = normalize(in.worldPos - camWorld);

  float tNear, tFar;
  if (!mag_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tNear, tFar)) {
    discard_fragment();
  }

  // Start march from the entry point (or camera if inside the box)
  float tStart = max(tNear, 0.0f);

  // SCENE_SCALE = 6: local [-1,1] → scene [-6,6]. The unit sphere (r=1) sits at
  // the origin; interesting field-line content is at r=1.5–5. With scale=1.2
  // (the old value) the box barely reached r=1.2 — almost entirely inside the
  // sphere where b=(|p|²-1)/density≤0 → p/=b blows up → white blob artifact.
  const float SCENE_SCALE = 6.0f;

  float3 ro = roLocal * SCENE_SCALE;
  float3 rd = normalize(rdLocal);

  float T = uniforms.time * 0.25f;  // matches original T = iTime/4.

  // Accumulated colour (emission-only volumetric, matches original c += min(.001/s, s))
  float3 c = float3(0.0f);
  float d = tStart * SCENE_SCALE;
  float travelMax = tFar * SCENE_SCALE;

  for (int i = 0; i < MAG_STEPS; ++i) {
    if (d >= travelMax) break;
    float3 p = ro + rd * d;
    float s = mag_map(p, T);

    if (s < MAG_NEAR) break;

    c += min(0.001f / s, s);

    // Adaptive step: coord-transform shrinks steps near sphere surface.
    // b is re-evaluated at the current march position (same as original B macro).
    float density = mag_oscillate(T / 8.0f, 10.0f, 30.0f);
    float b = (dot(p, p) - 1.0f) / density;
    d += s * clamp(b, 0.3f, 2.0f);
  }

  // Colour tint — original: c *= vec3(.7,.8,.9) / min(sqrt(length(uv)), 1.)
  // uv is 2D pixel offset from screen centre; analogous in VR is the angular
  // distance of the ray from the viewing axis (length of rd.xy).
  float radialUV = length(rd.xy);  // 0 at straight-ahead, ~0.7 at 45°
  c *= float3(0.7f, 0.8f, 0.9f) / max(sqrt(radialUV), 0.15f);

  // tanh(c^3) for contrast + HDR limiting
  float3 col = tanh(c * c * c);

  return float4(col, 1.0f);
}
