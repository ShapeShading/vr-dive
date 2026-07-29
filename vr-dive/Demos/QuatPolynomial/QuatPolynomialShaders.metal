#include <metal_stdlib>
using namespace metal;

struct SceneUniforms {
  float  time;
  uint   layerCount;
  float2 padding;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct QuatPolynomialParticleState {
  float4 positionAndScale;
  float4 color;
};

struct QuatPolynomialUniforms {
  float time;
  float speed;
  float worldScale;
  uint  particleCount;
};

// ── Complex arithmetic ────────────────────────────────────────────────────────
inline float2 cmul(float2 a, float2 b) {
  return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
inline float2 cdiv(float2 a, float2 b) {
  float d = max(dot(b, b), 1e-24f);
  return float2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) / d;
}
inline float2 cpow3(float2 c) {
  float2 c2 = cmul(c, c);
  return cmul(c2, c);
}

// f(x) = x^5 + a2*x^2 + a1*x + a0
inline float2 evalPoly(float2 x, float2 a2, float2 a1, float2 a0) {
  float2 x2 = cmul(x, x);
  float2 x3 = cmul(x2, x);
  float2 x5 = cmul(x3, x2);
  return x5 + cmul(a2, x2) + cmul(a1, x) + a0;
}

// ── HSV → RGB ─────────────────────────────────────────────────────────────────
inline float3 hsv2rgb(float h, float s, float v) {
  float h6 = fract(h) * 6.0f;
  float f  = fract(h6);
  float p  = v * (1.0f - s);
  float q  = v * (1.0f - s * f);
  float t  = v * (1.0f - s * (1.0f - f));
  int   i  = int(h6);
  if (i == 0) return float3(v, t, p);
  if (i == 1) return float3(q, v, p);
  if (i == 2) return float3(p, v, t);
  if (i == 3) return float3(p, q, v);
  if (i == 4) return float3(t, p, v);
  return float3(v, p, q);
}

// ── Compute shader ────────────────────────────────────────────────────────────
// Each particle encodes a specific (theta1, theta2, root, circle_point) tuple.
// The polynomial f(x) = x^5 + t1^3*x^2 + t2^2*x + 1 has 5 complex roots.
// Each complex root α+βi generates a circle in quaternion space:
//   (α, β·cos φ, β·sin φ)  for φ ∈ [0, 2π]
// sweeping (θ1, θ2) ∈ S¹×S¹ traces the full quaternion root variety in 3D.

#define THETA1_GRID 64
#define THETA2_GRID 64
#define POLY_DEGREE  5
#define CIRCLE_PTS  20

kernel void computeQuatPolyParticles(
    device   QuatPolynomialParticleState *states   [[buffer(0)]],
    constant QuatPolynomialUniforms      &uniforms [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
  if (id >= uniforms.particleCount) return;

  // ── Decode (θ1, θ2, root, φ) from flat particle id ──────────────────────
  uint phi_idx    = id % uint(CIRCLE_PTS);
  uint rem        = id / uint(CIRCLE_PTS);
  uint root_idx   = rem % uint(POLY_DEGREE);
  rem             = rem / uint(POLY_DEGREE);
  uint theta2_idx = rem % uint(THETA2_GRID);
  uint theta1_idx = rem / uint(THETA2_GRID);

  // ── Parameter angles, slowly rotating over time ──────────────────────────
  float s      = uniforms.speed * uniforms.time;
  float theta1 = (float(theta1_idx) / float(THETA1_GRID)) * 2.0f * M_PI_F + s;
  float theta2 = (float(theta2_idx) / float(THETA2_GRID)) * 2.0f * M_PI_F
               + s * 0.618033988f;   // golden-ratio rate avoids resonance

  // ── Polynomial coefficients: f(x) = x^5 + t1³x² + t2²x + 1 ─────────────
  float2 t1 = float2(cos(theta1), sin(theta1));
  float2 t2 = float2(cos(theta2), sin(theta2));
  float2 a2 = cpow3(t1);        // t1^3 — 3-fold θ1 symmetry
  float2 a1 = cmul(t2, t2);     // t2^2 — 2-fold θ2 symmetry
  float2 a0 = float2(1.0f, 0.0f);

  // ── Durand-Kerner / Weierstrass simultaneous root finding ────────────────
  // Initial guesses: equally spaced on unit circle with slight offset.
  float2 roots[POLY_DEGREE];
  float2 new_roots[POLY_DEGREE];
  for (int k = 0; k < POLY_DEGREE; k++) {
    float angle = (2.0f * M_PI_F * float(k) + 0.31f) / float(POLY_DEGREE);
    roots[k] = float2(1.1f * cos(angle), 1.1f * sin(angle));
  }

  for (int iter = 0; iter < 12; iter++) {
    for (int k = 0; k < POLY_DEGREE; k++) {
      float2 fk    = evalPoly(roots[k], a2, a1, a0);
      float2 denom = float2(1.0f, 0.0f);
      for (int j = 0; j < POLY_DEGREE; j++) {
        if (j != k) denom = cmul(denom, roots[k] - roots[j]);
      }
      float dlen2  = dot(denom, denom);
      new_roots[k] = (dlen2 > 1e-20f) ? roots[k] - cdiv(fk, denom) : roots[k];
    }
    for (int k = 0; k < POLY_DEGREE; k++) roots[k] = new_roots[k];
  }

  // ── Quaternion circle ─────────────────────────────────────────────────────
  float2 root    = roots[root_idx];
  float  alpha   = root.x;   // real part  → x coordinate
  float  beta    = root.y;   // imag part  → circle radius in yz plane

  // Validity: discard unconverged or blown-up roots
  float rootMag  = length(root);
  float residual = length(evalPoly(root, a2, a1, a0));
  bool  valid    = (rootMag < 3.5f) && (residual < 0.06f)
                && !isnan(root.x) && !isnan(root.y);

  float phi   = 2.0f * M_PI_F * float(phi_idx) / float(CIRCLE_PTS);
  float3 lp   = float3(alpha, beta * cos(phi), beta * sin(phi));
  float3 wp   = lp * uniforms.worldScale + float3(0.0f, -0.1f, -1.2f);
  float  scale = valid ? 0.10f : 0.0f;

  // ── Colour: hue from θ1, shifted per root sheet ───────────────────────────
  float hue = float(theta1_idx) / float(THETA1_GRID)
            + float(root_idx)   / float(POLY_DEGREE);
  float sat = 0.80f;
  float val = 0.70f + 0.30f * abs(sin(float(theta2_idx) / float(THETA2_GRID) * M_PI_F));
  float3 rgb = hsv2rgb(hue, sat, val);

  states[id].positionAndScale = float4(wp, scale);
  states[id].color            = float4(rgb, 1.0f);
}

// ── Vertex / fragment shaders ─────────────────────────────────────────────────
struct VertexOut {
  float4 clipPos        [[position]];
  float3 normal         [[flat]];
  float4 particleColor  [[flat]];
};

vertex VertexOut quatPolyVertexShader(
    ushort                                    amplificationID [[amplification_id]],
    const device MeshVertex                  *vertices        [[buffer(0)]],
    const device QuatPolynomialParticleState *states          [[buffer(1)]],
    constant SceneUniforms                   &uniforms        [[buffer(2)]],
    constant float4x4                        *vpMatrices      [[buffer(3)]],
    uint vertexID   [[vertex_id]],
    uint instanceID [[instance_id]])
{
  uint layers    = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1u);

  QuatPolynomialParticleState state = states[instanceID];
  MeshVertex vtx = vertices[vertexID];

  float3 center = state.positionAndScale.xyz;
  float  scale  = state.positionAndScale.w;
  float3 pos    = center + vtx.position * scale;

  VertexOut out;
  out.clipPos       = vpMatrices[viewIndex] * float4(pos, 1.0f);
  out.normal        = vtx.normal;
  out.particleColor = state.color;
  return out;
}

fragment float4 quatPolyFragmentShader(
    VertexOut               in       [[stage_in]],
    constant SceneUniforms &uniforms [[buffer(0)]])
{
  float3 n        = normalize(in.normal);
  float3 lightDir = normalize(float3(-0.2f, 0.8f, -0.4f));
  float  ndotl    = max(dot(n, lightDir), 0.0f) * 0.65f + 0.35f;
  return float4(in.particleColor.rgb * ndotl, 1.0f);
}
