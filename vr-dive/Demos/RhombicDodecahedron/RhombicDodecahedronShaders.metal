#include <metal_stdlib>
using namespace metal;

// Layout must match the Swift struct RhombicDodecahedronUniforms.
struct RhombicDodecahedronUniforms {
  float  time;
  uint   viewCount;
  float  roomScale;           // half-distance from centre to each face
  uint   reflectionBounces;
  float4 objectCenter;        // xyz = world position of dodecahedron centre
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

// ── Vertex shader ────────────────────────────────────────────────────────────
// Renders a bounding sphere positioned at objectCenter.
// Passes the interpolated world-space position to the fragment for ray setup.

struct RhombicVertexOut {
  float4 clipPos   [[position]];
  float3 worldPos;
  uint   viewIndex [[flat]];
};

vertex RhombicVertexOut rhombicDodecahedronVertex(
    ushort                               amplificationID [[amplification_id]],
    const device MeshVertex             *vertices        [[buffer(0)]],
    constant RhombicDodecahedronUniforms &uniforms       [[buffer(1)]],
    constant float4x4                   *vpMatrices      [[buffer(2)]],
    uint                                 vertexID        [[vertex_id]])
{
  MeshVertex vtx   = vertices[vertexID];
  float3 worldPos  = vtx.position + uniforms.objectCenter.xyz;
  uint viewIndex   = min((uint)amplificationID, uniforms.viewCount - 1u);

  RhombicVertexOut out;
  out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos  = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

// ── Ray vs rhombic dodecahedron ──────────────────────────────────────────────
// The 12 face normals (normalised). Pre-computed literals avoid Metal's
// restriction on global constructors (no function calls at global scope).
// 1/sqrt(2) ≈ 0.70710678118654752
#define INV_SQRT2 0.70710678118654752f

constant float3 kRhombicNormals[12] = {
  { INV_SQRT2,  INV_SQRT2, 0.0f},
  { INV_SQRT2, -INV_SQRT2, 0.0f},
  {-INV_SQRT2,  INV_SQRT2, 0.0f},
  {-INV_SQRT2, -INV_SQRT2, 0.0f},
  { INV_SQRT2, 0.0f,  INV_SQRT2},
  { INV_SQRT2, 0.0f, -INV_SQRT2},
  {-INV_SQRT2, 0.0f,  INV_SQRT2},
  {-INV_SQRT2, 0.0f, -INV_SQRT2},
  {0.0f,  INV_SQRT2,  INV_SQRT2},
  {0.0f,  INV_SQRT2, -INV_SQRT2},
  {0.0f, -INV_SQRT2,  INV_SQRT2},
  {0.0f, -INV_SQRT2, -INV_SQRT2},
};

// Returns the parametric interval [tEntry, tExit] for a ray (origin, dir).
// If tEntry > tExit or tExit < 0 the ray misses.
struct RayHit {
  float tEntry;
  float tExit;
  int   entryFace;  // index of the face the ray enters through, or -1
};

RayHit rhombicRayHit(float3 origin, float3 dir, float s)
{
  float tEntry    = -1e9f;
  float tExit     =  1e9f;
  int   entryFace = -1;

  for (int i = 0; i < 12; i++) {
    float3 n     = kRhombicNormals[i];
    float  denom = dot(n, dir);
    float  dist  = s - dot(n, origin);  // positive when origin is inside this half-space

    if (abs(denom) < 1e-7f) {
      if (dist < 0.0f) {
        // Origin outside a parallel half-space → ray always misses
        RayHit miss; miss.tEntry = 1.0f; miss.tExit = -1.0f; miss.entryFace = -1;
        return miss;
      }
      continue;
    }

    float t = dist / denom;
    if (denom < 0.0f) {               // ray enters this half-space
      if (t > tEntry) { tEntry = t; entryFace = i; }
    } else {                          // ray exits this half-space
      if (t < tExit) { tExit = t; }
    }
  }

  RayHit r; r.tEntry = tEntry; r.tExit = tExit; r.entryFace = entryFace;
  return r;
}

// ── Mirror bounce tracing ────────────────────────────────────────────────────
struct BounceResult {
  float3 dir;
  float3 lastNormal;
  int    bounceCount;
  float  minEdgeDist;  // min distance-to-edge across all bounce points (0 = on edge)
};

// Simple integer hash for per-pixel noise — no global constructor, no texture.
inline uint pcgHash(uint v)
{
  uint state = v * 747796405u + 2891336453u;
  uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

BounceResult traceMirrorBounces(float3 pos, float3 dir, float scale,
                                uint maxBounces, uint noiseSeed)
{
  // Hard cap: never exceed 20 bounces regardless of the uniform.
  const uint kHardMax = 20u;
  maxBounces = min(maxBounces, kHardMax);

  // Dither: ~10 % of pixels skip 1 bounce to spread compute load.
  // pcgHash returns uniform bits; (rng % 10u == 0u) is true ~10 % of the time.
  uint rng = pcgHash(noiseSeed);
  uint budget = maxBounces - ((rng % 10u == 0u) ? 1u : 0u);

  BounceResult r;
  r.dir         = dir;
  r.lastNormal  = float3(0, 1, 0);
  r.bounceCount = 0;
  r.minEdgeDist = 1.0f;

  for (uint b = 0u; b < budget; b++) {
    float tMin    = 1e9f;
    int   hitFace = -1;

    for (int i = 0; i < 12; i++) {
      float denom = dot(kRhombicNormals[i], dir);
      if (denom > 1e-5f) {
        float t = (scale - dot(kRhombicNormals[i], pos)) / denom;
        if (t > 1e-4f && t < tMin) { tMin = t; hitFace = i; }
      }
    }

    if (hitFace < 0) break;

    pos          = pos + dir * tMin;
    r.lastNormal = kRhombicNormals[hitFace];
    dir          = reflect(dir, kRhombicNormals[hitFace]);
    r.dir        = dir;
    r.bounceCount++;

    // Distance to nearest edge: (scale - dot(n_j, p)) / scale.
    float minGap = 1.0f;
    for (int j = 0; j < 12; j++) {
      if (j == hitFace) continue;
      float gap = (scale - dot(kRhombicNormals[j], pos)) / scale;
      if (gap < minGap) minGap = gap;
    }
    if (minGap < r.minEdgeDist) r.minEdgeDist = minGap;
  }

  return r;
}

// ── Colour mapping ───────────────────────────────────────────────────────────
// Edges are sharp bright white; faces are a uniform dark colour. No glow.
float3 bounceColor(BounceResult bounce)
{
  const float kEdgeWidth = 0.015f;

  // step() = hard edge, no glow.
  float onEdge = step(bounce.minEdgeDist, kEdgeWidth);

  float3 faceColor = float3(0.04f, 0.04f, 0.06f);
  float3 edgeColor = float3(1.00f, 0.97f, 0.93f);

  return mix(faceColor, edgeColor, onEdge);
}

// ── Fragment shader ──────────────────────────────────────────────────────────
// Fragment receives worldPos on the bounding sphere surface.
// We reconstruct the ray from the eye through that point, intersect it with
// the rhombic dodecahedron, and trace mirror bounces inside.
// Fragments that miss the dodecahedron are discarded so the object has the
// correct silhouette.  Custom depth is written at the entry surface so the
// object composites correctly against other geometry.

struct RhombicFragOut {
  float4 color [[color(0)]];
  float  depth [[depth(any)]];
};

fragment RhombicFragOut rhombicDodecahedronFragment(
    RhombicVertexOut                      in                    [[stage_in]],
    constant RhombicDodecahedronUniforms &uniforms              [[buffer(0)]],
    constant float4x4                    *viewToWorldTransforms [[buffer(1)]],
    constant float4x4                    *vpMatrices            [[buffer(2)]])
{
  uint viewIndex = min(in.viewIndex, uniforms.viewCount - 1u);

  // Eye world position: column 3 of the view-to-world matrix.
  float3 eyeWorld = viewToWorldTransforms[viewIndex][3].xyz;
  float3 rayDir   = normalize(in.worldPos - eyeWorld);

  // Work in dodecahedron-local space (centred at objectCenter).
  float3 localEye = eyeWorld - uniforms.objectCenter.xyz;

  RayHit hit = rhombicRayHit(localEye, rayDir, uniforms.roomScale);

  // Discard fragments that miss the dodecahedron.
  if (hit.tEntry > hit.tExit || hit.tExit < 0.0f) {
    discard_fragment();
  }

  // Start at the entry surface (or from inside if the eye is already inside).
  float  tStart     = max(hit.tEntry, 0.001f);
  float3 entryLocal = localEye + rayDir * tStart;

  // Per-pixel noise seed derived from fragment position and view index,
  // used to dither bounce budget across pixels.
  uint noiseSeed = uint(in.clipPos.x) * 1973u + uint(in.clipPos.y) * 9277u + viewIndex * 26699u;

  BounceResult bounce = traceMirrorBounces(
      entryLocal, rayDir, uniforms.roomScale, uniforms.reflectionBounces, noiseSeed);

  float3 color = bounceColor(bounce);

  // Compute depth at the entry surface so depth-compositing with other geometry works.
  float3 entryWorld = entryLocal + uniforms.objectCenter.xyz;
  float4 clipEntry  = vpMatrices[viewIndex] * float4(entryWorld, 1.0f);
  float  ndcDepth   = clipEntry.z / clipEntry.w;

  RhombicFragOut out;
  out.color = float4(color, 1.0f);
  out.depth = clamp(ndcDepth, 0.0f, 1.0f);
  return out;
}
