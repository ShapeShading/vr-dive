#include <metal_stdlib>
using namespace metal;

// ── Structs (must match HuashanTypes.swift) ───────────────────────────────────
struct SplatPoint {
  packed_float3 position;  // 12 bytes
  packed_float3 scale;     // 12 bytes
  uchar4 colorRGBA;        //  4 bytes (r,g,b, opacity)
  uchar4 rotWXYZ;          //  4 bytes (w,x,y,z each = component*128+128)
};

struct PerEyeUniforms {
  float4x4 vpMatrix;
  float4x4 viewMatrix;
  float2   focalXY;
  float2   viewportSize;      // display viewport (may be larger than texture)
  float2   renderTargetSize;  // actual texture size
};

struct HuashanUniforms {
  PerEyeUniforms eye0;
  PerEyeUniforms eye1;
  uint  splatCount;
  uint  viewCount;
  float splatScale;
  float _pad;
};

// ── Helper: build 3×3 rotation matrix from unit quaternion stored as (x,y,z,w) ─
inline float3x3 quatToMatrix(float4 q) {
  // q.xyzw = (x, y, z, w)  — .sog WebP RGBA maps to (x,y,z,w) order
  float x = q.x, y = q.y, z = q.z, w = q.w;
  float x2 = x+x, y2 = y+y, z2 = z+z;
  float wx = w*x2, wy = w*y2, wz = w*z2;
  float xx = x*x2, xy = x*y2, xz = x*z2;
  float yy = y*y2, yz = y*z2, zz = z*z2;
  // column-major float3x3(col0, col1, col2)
  return float3x3(
    float3(1-(yy+zz),   xy+wz,       xz-wy),
    float3(  xy-wz,   1-(xx+zz),     yz+wx),
    float3(  xz+wy,     yz-wx,     1-(xx+yy))
  );
}

// ── Compute 2D screen-space covariance via Jacobian projection ─────────────────
// Sigma3D: 3×3 world-space covariance
// posView: view-space position of the splat centre
// focalXY: (fx, fy) in pixels
// returns float3(cov00, cov01, cov11)  — upper-triangle of 2×2 sym matrix
inline float3 computeCov2D(float3x3 Sigma3D,
                            float3   posView,
                            float2   focalXY,
                            float4x4 viewMatrix) {
  // Extract 3×3 rotation part of view matrix (world → view)
  float3x3 W = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);

  // View-space covariance: Sigma_v = W * Sigma3D * W^T
  float3x3 Sv = W * Sigma3D * transpose(W);

  float tz  = posView.z;
  float tx  = posView.x;
  float ty  = posView.y;
  float tz2 = tz * tz;

  float J00 =  focalXY.x / tz;
  float J11 =  focalXY.y / tz;
  float J02 = -focalXY.x * tx / tz2;
  float J12 = -focalXY.y * ty / tz2;

  // Sigma2D = J * Sv * J^T  (2×2, symmetric → 3 values)
  // J = [[J00, 0, J02], [0, J11, J12]]   (2×3)
  // Using row-vectors of J:
  float3 jRow0 = float3(J00, 0.0, J02);
  float3 jRow1 = float3(0.0, J11, J12);

  // Sv * J^T:  columns are Sv * jRow0^T and Sv * jRow1^T
  float3 col0 = Sv * jRow0;  // Sv * J^T[:,0]
  float3 col1 = Sv * jRow1;  // Sv * J^T[:,1]

  float c00 = dot(jRow0, col0);
  float c01 = dot(jRow0, col1);
  float c11 = dot(jRow1, col1);

  // Small regularizer to prevent degenerate splats
  return float3(c00 + 0.3, c01, c11 + 0.3);
}

// ── Vertex shader output ───────────────────────────────────────────────────────
struct VertexOut {
  float4 clipPos       [[position]];
  float2 uv;           // normalised ellipse coords, ±1 at 3σ boundary
  float4 color;        // pre-multiplied: rgba with r,g,b pre-multiplied by alpha
};

// ── Vertex shader: 3DGS ellipse billboard with covariance ────────────────────
vertex VertexOut huashanVertexShader(
    ushort             amplificationID [[amplification_id]],
    uint               vertexID        [[vertex_id]],
    uint               instanceID      [[instance_id]],
    const device SplatPoint    *splats         [[buffer(0)]],
    const device uint32_t      *sortedIndices  [[buffer(1)]],
    constant HuashanUniforms   &uniforms       [[buffer(2)]])
{
  VertexOut out;
  out.uv      = float2(0);
  out.clipPos = float4(0, 0, -1, 1);  // default: invisible (reverse-Z)
  out.color   = float4(0);

  // amplificationID → per-eye VP (avoids relying on uniforms.viewCount layout)
  PerEyeUniforms eye = (amplificationID == 0) ? uniforms.eye0 : uniforms.eye1;

  // Sample raw buffer at fixed physical stride (not sorted indices).
  // Sorted indices change every ~1s as background sort completes, causing visible
  // flicker when sparsely sampling. Physical-index sampling is perfectly stable.
  uint splatIdx = instanceID * 16u;  // stride=16 → ~77k splats, within GPU budget
  if (splatIdx >= uniforms.splatCount) { return out; }
  SplatPoint sp = splats[splatIdx];

  // ── Colour & opacity ─────────────────────────────────────────────────────
  float alpha = float(sp.colorRGBA.a) / 255.0;
  if (alpha < 0.02) { return out; }  // skip near-invisible splats

  float3 srgb   = float3(sp.colorRGBA.rgb) / 255.0;
  float3 linRGB = pow(max(srgb, 0.0), float3(2.2));
  // Store as pre-multiplied alpha so fragment can just return it
  out.color = float4(linRGB * alpha, alpha);

  // ── View-space position ───────────────────────────────────────────────────
  float3 posScene = float3(sp.position);
  float4 posView4 = eye.viewMatrix * float4(posScene, 1.0);
  float3 posView  = posView4.xyz;
  if (posView.z >= 0.0) { return out; }  // behind camera

  float4 clipCenter = eye.vpMatrix * float4(posScene, 1.0);
  if (clipCenter.w <= 0.0) { return out; }

  // ── Decode quaternion (xyzw stored as uchar4: component*128+128) ──────────
  float4 q;
  q.x = (float(sp.rotWXYZ.x) - 128.0) / 128.0;
  q.y = (float(sp.rotWXYZ.y) - 128.0) / 128.0;
  q.z = (float(sp.rotWXYZ.z) - 128.0) / 128.0;
  q.w = (float(sp.rotWXYZ.w) - 128.0) / 128.0;
  q   = normalize(q);  // guard against quantisation error

  // ── Scale: .splat format stores linear scale (already exp'd by converter) ──
  float3 sc = float3(sp.scale);

  // ── 3D covariance Σ = R·S²·Rᵀ in scene space ─────────────────────────────
  float3x3 R    = quatToMatrix(q);
  float3x3 S2   = float3x3(
    float3(sc.x*sc.x, 0, 0),
    float3(0, sc.y*sc.y, 0),
    float3(0, 0, sc.z*sc.z));
  float3x3 Sig3 = R * S2 * transpose(R);

  // ── Project to 2D screen-space covariance ────────────────────────────────
  float3 cov2d = computeCov2D(Sig3, posView, eye.focalXY, eye.viewMatrix);
  float c00 = cov2d.x, c01 = cov2d.y, c11 = cov2d.z;

  // ── Eigendecomposition of 2×2 symmetric cov → half-axes in pixels ─────────
  float mid   = 0.5 * (c00 + c11);
  float disc  = sqrt(max(0.25*(c00-c11)*(c00-c11) + c01*c01, 0.0));
  float lam1  = mid + disc;     // larger eigenvalue
  float lam2  = mid - disc;     // smaller eigenvalue
  float r1    = sqrt(max(lam1, 0.0));  // half-axis radii in pixels
  float r2    = sqrt(max(lam2, 0.0));
  // Eigenvector for lam1:
  float2 v1   = normalize(float2(c01, lam1 - c00) + float2(1e-6));

  // Billboard corners in pixel-space (±3σ extent, clamped to avoid huge splats)
  float maxR  = 8.0;   // max half-axis in pixels; limits fragment fill rate
  r1 = min(r1 * 3.0, maxR);
  r2 = min(r2 * 3.0, maxR);

  float2 corners[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
  uint   triVtx[6]  = { 0, 1, 2, 1, 3, 2 };
  float2 uv  = corners[triVtx[vertexID % 6]];

  // Local billboard coordinate in pixels, oriented along principal axes
  float2 v2 = float2(-v1.y, v1.x);
  float2 pxOffset = uv.x * r1 * v1 + uv.y * r2 * v2;

  // Convert pixel offset to NDC offset (renderTargetSize in physical pixels)
  float2 ndcOffset = pxOffset / (eye.renderTargetSize * 0.5) * clipCenter.w;

  out.clipPos = clipCenter + float4(ndcOffset, 0.0, 0.0);
  out.uv      = uv;
  return out;
}

// ── Fragment shader ───────────────────────────────────────────────────────────
fragment float4 huashanFragmentShader(VertexOut in [[stage_in]])
{
  float r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard_fragment(); }
  float gauss = exp(-0.5 * r2 * 9.0);  // billboard is ±3σ; at |uv|=1 → exp(-4.5)≈0.01
  if (gauss < 0.02) { discard_fragment(); }
  // Pre-multiplied alpha: BOTH rgb and alpha must be scaled by gauss
  // in.color = float4(linRGB * alpha, alpha) from vertex shader
  return in.color * gauss;
}

