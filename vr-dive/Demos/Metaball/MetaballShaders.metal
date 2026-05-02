#include <metal_stdlib>
using namespace metal;

// Must match MetaballUniforms in Swift.
struct MetaballUniforms {
  float  time;
  uint   viewCount;
  float  boundingRadius;
  float  padding;
  float4 objectCenter;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

// ── Vertex shader ─────────────────────────────────────────────────────────────
// Renders the bounding sphere; passes interpolated world position to fragment.

struct MetaballVertexOut {
  float4 clipPos   [[position]];
  float3 worldPos;
  uint   viewIndex [[flat]];
};

vertex MetaballVertexOut metaballVertex(
    ushort                     amplificationID [[amplification_id]],
    const device MeshVertex   *vertices        [[buffer(0)]],
    constant MetaballUniforms &uniforms         [[buffer(1)]],
    constant float4x4         *vpMatrices       [[buffer(2)]],
    uint                       vertexID         [[vertex_id]])
{
  MeshVertex vtx  = vertices[vertexID];
  float3 worldPos = vtx.position + uniforms.objectCenter.xyz;
  uint viewIndex  = min((uint)amplificationID, uniforms.viewCount - 1u);

  MetaballVertexOut out;
  out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos  = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

// ── Metaball SDF ──────────────────────────────────────────────────────────────
// Fixed ball count as a preprocessor constant so local arrays can use it.
#define BALL_COUNT 18

constant uint  kBallCount   = BALL_COUNT;
constant float kBallRadius  = 0.065f;  // world-space radius of each water drop
constant float kSmoothK     = 0.085f;  // merging smoothness (larger = blobs connect sooner)
constant float kMotionRange = 0.255f;  // max distance from centre each ball wanders

// Polynomial smooth-min.
float smin(float a, float b, float k)
{
  float h = saturate(0.5f + 0.5f * (b - a) / k);
  return mix(b, a, h) - k * h * (1.0f - h);
}

// Procedural per-ball position in dodecahedron-local space.
// Two sin components per axis give organic, non-repetitive motion.
float3 ballCenter(uint idx, float time)
{
  float i = float(idx);
  float3 p = float3(
    sin(time * 0.22f + i * 2.094f) * 0.70f + sin(time * 0.51f + i * 1.234f) * 0.30f,
    sin(time * 0.18f + i * 1.745f) * 0.70f + sin(time * 0.37f + i * 2.456f) * 0.30f,
    sin(time * 0.25f + i * 2.618f) * 0.70f + sin(time * 0.43f + i * 3.678f) * 0.30f
  );
  return p * kMotionRange;
}

// SDF using pre-computed centres (avoids recomputing ballCenter every march step).
float sdfFromCenters(float3 p, thread float3 *centers)
{
  float d = 1e9f;
  for (uint i = 0u; i < kBallCount; i++) {
    float di = length(p - centers[i]) - kBallRadius;
    d = smin(d, di, kSmoothK);
  }
  return d;
}

// Central-difference gradient (6 SDF evaluations → accurate normal).
float3 normalFromCenters(float3 p, thread float3 *centers)
{
  const float e = 0.003f;
  return normalize(float3(
    sdfFromCenters(p + float3(e, 0, 0), centers) - sdfFromCenters(p - float3(e, 0, 0), centers),
    sdfFromCenters(p + float3(0, e, 0), centers) - sdfFromCenters(p - float3(0, e, 0), centers),
    sdfFromCenters(p + float3(0, 0, e), centers) - sdfFromCenters(p - float3(0, 0, e), centers)
  ));
}

// ── Ray-sphere intersection ───────────────────────────────────────────────────
// Returns (tEntry, tExit); miss when tEntry > tExit.
float2 raySphereHit(float3 ro, float3 rd, float r)
{
  float b = dot(ro, rd);
  float c = dot(ro, ro) - r * r;
  float h = b * b - c;
  if (h < 0.0f) return float2(1.0f, -1.0f);
  h = sqrt(h);
  return float2(-b - h, -b + h);
}

// ── Direction-based shading ───────────────────────────────────────────────────
// Maps surface normal to a smooth HSV colour, plus a thin specular edge.
float3 directionShade(float3 normal, float3 viewDir)
{
  // Hue from azimuth of the normal (0–360° → 0–1).
  float az  = atan2(normal.z, normal.x);           // -π … π
  float hue = az / (2.0f * M_PI_F) + 0.5f;

  // Saturation peaks at the equator, fades at poles.
  float el  = asin(clamp(normal.y, -1.0f, 1.0f));  // -π/2 … π/2
  float sat = 0.80f + 0.20f * cos(el * 2.0f);

  // Brightness: constant base so all directions are clearly coloured.
  float val = 0.72f;

  // HSV → RGB
  float h6 = fract(hue) * 6.0f;
  float f  = fract(h6);
  float p  = val * (1.0f - sat);
  float q  = val * (1.0f - sat * f);
  float tv = val * (1.0f - sat * (1.0f - f));
  int   s6 = int(h6);
  float3 rgb;
  if      (s6 == 0) rgb = float3(val, tv,  p  );
  else if (s6 == 1) rgb = float3(q,   val, p  );
  else if (s6 == 2) rgb = float3(p,   val, tv );
  else if (s6 == 3) rgb = float3(p,   q,   val);
  else if (s6 == 4) rgb = float3(tv,  p,   val);
  else              rgb = float3(val, p,   q  );

  // Thin specular highlight so the surface still reads as 3-D.
  float3 L    = normalize(float3(0.6f, 0.8f, 0.3f));
  float3 H    = normalize(L + viewDir);
  float  spec = pow(saturate(dot(normal, H)), 60.0f) * 0.55f;

  return saturate(rgb + spec);
}

// ── Fragment shader ───────────────────────────────────────────────────────────
struct MetaballFragOut {
  float4 color [[color(0)]];
  float  depth [[depth(any)]];
};

fragment MetaballFragOut metaballFragment(
    MetaballVertexOut             in                    [[stage_in]],
    constant MetaballUniforms    &uniforms              [[buffer(0)]],
    constant float4x4            *viewToWorldTransforms [[buffer(1)]],
    constant float4x4            *vpMatrices            [[buffer(2)]])
{
  uint viewIndex = min(in.viewIndex, uniforms.viewCount - 1u);

  // Eye world position from view-to-world matrix column 3.
  float3 eyeWorld = viewToWorldTransforms[viewIndex][3].xyz;
  float3 rayDir   = normalize(in.worldPos - eyeWorld);

  // Work in object-local space.
  float3 localEye = eyeWorld - uniforms.objectCenter.xyz;

  // Clip marching to the bounding sphere.
  float2 bounds = raySphereHit(localEye, rayDir, uniforms.boundingRadius);
  if (bounds.x > bounds.y || bounds.y < 0.0f) {
    discard_fragment();
  }

  // Pre-compute all ball centres once per pixel (avoids redundant trig in loop).
  float3 centers[BALL_COUNT];
  for (uint i = 0u; i < kBallCount; i++) {
    centers[i] = ballCenter(i, uniforms.time);
  }

  // Sphere-tracing ray march.
  const int   kMaxSteps     = 64;
  const float kHitThreshold = 0.0018f;

  float  t      = max(bounds.x, 0.001f);
  float  tMax   = bounds.y;
  float3 hitPos = localEye;
  bool   hit    = false;

  for (int step = 0; step < kMaxSteps; step++) {
    float3 p = localEye + rayDir * t;
    float  d = sdfFromCenters(p, centers);

    if (d < kHitThreshold) {
      hitPos = p;
      hit    = true;
      break;
    }

    // Step by 85 % of the SDF distance (conservative to avoid over-stepping).
    t += max(d * 0.85f, kHitThreshold);
    if (t >= tMax) break;
  }

  if (!hit) {
    discard_fragment();
  }

  float3 normal   = normalFromCenters(hitPos, centers);
  float3 worldHit = hitPos + uniforms.objectCenter.xyz;
  float3 viewDir  = normalize(eyeWorld - worldHit);

  float3 color = directionShade(normal, viewDir);

  // Write depth at the actual surface for correct compositing.
  float4 clipHit = vpMatrices[viewIndex] * float4(worldHit, 1.0f);
  float  ndcZ    = clipHit.z / clipHit.w;

  MetaballFragOut out;
  out.color = float4(color, 1.0f);
  out.depth = clamp(ndcZ, 0.0f, 1.0f);
  return out;
}
