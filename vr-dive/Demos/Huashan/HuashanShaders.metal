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

// ── Per-splat precomputed data (matches HuashanSplatPrecomp in HuashanTypes.swift) ─
struct SplatPrecomp {
  float4 clipPos0;   // clip position eye 0
  float4 clipPos1;   // clip position eye 1
  float4 axesU;      // xy = NDC axis-U eye0,  zw = NDC axis-U eye1
  float4 axesV;      // xy = NDC axis-V eye0,  zw = NDC axis-V eye1
  float4 color;      // pre-multiplied RGBA
};

// ── Compute kernel: precompute per-splat 2D covariance for both eyes ──────────
// Runs once per splat per frame (NOT amplified). Results stored in precompBuffer.
// Vertex shader just reads the result — no heavy math in VS.
kernel void huashanComputePrecomp(
    uint                         gid      [[thread_position_in_grid]],
    const device SplatPoint     *splats   [[buffer(0)]],
    constant HuashanUniforms    &uniforms [[buffer(1)]],
    device SplatPrecomp         *output   [[buffer(2)]],
    const device uint           *sortedIndices [[buffer(3)]])
{
  // Thread gid handles one entry of the sampled back-to-front sorted splat list.
  uint splatIdx = sortedIndices[gid];
  if (splatIdx >= uniforms.splatCount) { return; }

  SplatPrecomp out;
  // Default: invisible (behind camera in reverse-Z)
  out.clipPos0 = float4(0, 0, -1, 1);
  out.clipPos1 = float4(0, 0, -1, 1);
  out.axesU    = float4(0);
  out.axesV    = float4(0);
  out.color    = float4(0);

  SplatPoint sp = splats[splatIdx];

  float alpha = float(sp.colorRGBA.a) / 255.0;
  if (alpha < 0.008) { output[gid] = out; return; }

  float3 srgb   = float3(sp.colorRGBA.rgb) / 255.0;
  float3 linRGB = pow(max(srgb, 0.0), float3(2.2));
  out.color = float4(linRGB * alpha, alpha);

  // Most raw scales are already linear, but a tiny fraction are huge outliers.
  // Clamp only the tail so we keep real anisotropic detail without giant blob splats.
  float4 q;
  q.x = (float(sp.rotWXYZ.x) - 128.0) / 128.0;
  q.y = (float(sp.rotWXYZ.y) - 128.0) / 128.0;
  q.z = (float(sp.rotWXYZ.z) - 128.0) / 128.0;
  q.w = (float(sp.rotWXYZ.w) - 128.0) / 128.0;
  q   = normalize(q);

  float3x3 R    = quatToMatrix(q);
  float3 sc     = clamp(abs(float3(sp.scale)), float3(0.0005), float3(1.15));
  float3x3 S2   = float3x3(
    float3(sc.x*sc.x, 0, 0),
    float3(0, sc.y*sc.y, 0),
    float3(0, 0, sc.z*sc.z));
  float3x3 Sig3 = R * S2 * transpose(R);

  float3 posScene = float3(sp.position);
  float maxR = 1.45;

  // Compute for each eye
  for (uint eyeIdx = 0; eyeIdx < 2; eyeIdx++) {
    PerEyeUniforms eye = (eyeIdx == 0) ? uniforms.eye0 : uniforms.eye1;

    float4 posView4 = eye.viewMatrix * float4(posScene, 1.0);
    float3 posView  = posView4.xyz;
    float viewDepth = -posView.z;

    float4 clipCenter = eye.vpMatrix * float4(posScene, 1.0);
    if (clipCenter.w <= 0.0) { continue; }

    float2 clipNDC = clipCenter.xy / clipCenter.w;
    float centerDist2 = dot(clipNDC, clipNDC);

    float3 cov2d = computeCov2D(Sig3, posView, eye.focalXY, eye.viewMatrix);
    float c00 = cov2d.x, c01 = cov2d.y, c11 = cov2d.z;

    float mid  = 0.5 * (c00 + c11);
    float disc = sqrt(max(0.25*(c00-c11)*(c00-c11) + c01*c01, 0.0));
    float lam1 = mid + disc;
    float lam2 = mid - disc;
    float r1   = min(sqrt(max(lam1, 0.0)) * 3.0, maxR);
    float r2   = min(sqrt(max(lam2, 0.0)) * 3.0, maxR);
    r2 = max(r2, r1 * 0.08);  // keep elongated structure but avoid collapsing to circles
    r1 = max(r1, 0.02);
    r2 = max(r2, 0.008);

    // Stable far-field thinning: preserve central splats, thin peripheral distant tiny ones first.
    uint hash = splatIdx * 1664525u + 1013904223u;
    bool peripheral = centerDist2 > 0.40;
    bool farPeripheral = centerDist2 > 0.85;
    bool edgePeripheral = centerDist2 > 1.40;
    if (peripheral && viewDepth > 2.35 && r1 < 0.30 && (hash & 1u) != 0u) { continue; }
    if (farPeripheral && viewDepth > 2.60 && r1 < 0.20 && (hash & 3u) != 0u) { continue; }
    if (farPeripheral && viewDepth > 2.80 && r1 < 0.12 && (hash & 7u) != 0u) { continue; }
    if (edgePeripheral && viewDepth > 3.00 && r1 < 0.08 && (hash & 15u) != 0u) { continue; }

    float2 v1  = normalize(float2(c01, lam1 - c00) + float2(1e-6));
    float2 v2  = float2(-v1.y, v1.x);

    // Axes in NDC space: axisU/V such that vertex offset = (uv.x*axisU + uv.y*axisV) * w
    float2 axisU = v1 * r1 / (eye.renderTargetSize * 0.5);
    float2 axisV = v2 * r2 / (eye.renderTargetSize * 0.5);

    if (eyeIdx == 0) {
      out.clipPos0  = clipCenter;
      out.axesU.xy  = axisU;
      out.axesV.xy  = axisV;
    } else {
      out.clipPos1  = clipCenter;
      out.axesU.zw  = axisU;
      out.axesV.zw  = axisV;
    }
  }

  output[gid] = out;
}

// ── Vertex shader output ───────────────────────────────────────────────────────
struct VertexOut {
  float4 clipPos       [[position]];
  float2 uv;           // normalised ellipse coords, ±1 at 3σ boundary
  float4 color;        // pre-multiplied RGBA
};

// ── Vertex shader: reads precomputed data, trivial billboard projection ───────
vertex VertexOut huashanVertexShader(
    ushort                    amplificationID [[amplification_id]],
    uint                      vertexID        [[vertex_id]],
    uint                      instanceID      [[instance_id]],
    const device SplatPrecomp *precomp        [[buffer(0)]],
    constant HuashanUniforms  &uniforms       [[buffer(1)]])
{
  VertexOut out;
  out.clipPos = float4(0, 0, -1, 1);
  out.uv      = float2(0);
  out.color   = float4(0);

  float2 corners[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
  uint   triVtx[6]  = { 0, 1, 2, 1, 3, 2 };
  float2 uv = corners[triVtx[vertexID % 6]];

  if (instanceID >= uniforms.splatCount) { return out; }

  SplatPrecomp sp = precomp[instanceID];
  out.color = sp.color;
  if (out.color.a < 0.02) { return out; }

  float4 clipCenter = (amplificationID == 0) ? sp.clipPos0 : sp.clipPos1;
  if (clipCenter.w <= 0.0) { return out; }

  float2 axisU = (amplificationID == 0) ? sp.axesU.xy : sp.axesU.zw;
  float2 axisV = (amplificationID == 0) ? sp.axesV.xy : sp.axesV.zw;

  float2 ndcOffset = (uv.x * axisU + uv.y * axisV) * clipCenter.w;
  out.clipPos = clipCenter + float4(ndcOffset, 0.0, 0.0);
  out.uv      = uv;
  return out;
}

// ── Fragment shader ───────────────────────────────────────────────────────────
fragment float4 huashanFragmentShader(VertexOut in [[stage_in]])
{
  float r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard_fragment(); }
  float gauss = exp(-0.5 * r2 * 6.5);
  if (gauss < 0.015) { discard_fragment(); }
  return in.color * gauss;
}

