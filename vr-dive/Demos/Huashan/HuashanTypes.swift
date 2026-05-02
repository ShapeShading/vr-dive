import Foundation
import simd

// ─── GPU-side struct (matches .splat binary layout, 32 bytes) ───────────────
// Uses individual Floats instead of SIMD3<Float> to avoid Swift's 16-byte padding
// that would make the stride 40 instead of 32. Must match SplatPoint in HuashanShaders.metal
struct HuashanSplatPoint {
  var px, py, pz: Float  // 12 bytes (position)
  var sx, sy, sz: Float  // 12 bytes (scale, linear)
  var colorRGBA: SIMD4<UInt8>  //  4 bytes (sRGB 0-255 + opacity 0-255)
  var rotWXYZ: SIMD4<UInt8>  //  4 bytes (xyzw each = component*128+128)
}  // stride = 32 ✓

// ─── Per-eye uniforms passed to vertex shader ────────────────────────────────
struct HuashanPerEyeUniforms {
  var vpMatrix: simd_float4x4
  var viewMatrix: simd_float4x4  // scene → view
  var focalXY: SIMD2<Float>  // focal lengths in pixels (render target pixels)
  var viewportSize: SIMD2<Float>  // display viewport size (may be larger than texture)
  var renderTargetSize: SIMD2<Float>  // actual texture/render target size in pixels
}

// ─── Per-splat precomputed data: written by compute shader, read by vertex shader
// 5 × float4 = 80 bytes, fully float4-aligned
struct HuashanSplatPrecomp {
  var clipPos0: SIMD4<Float>  // clip position eye 0
  var clipPos1: SIMD4<Float>  // clip position eye 1
  var axesU: SIMD4<Float>  // xy = NDC axis-U eye0,  zw = NDC axis-U eye1
  var axesV: SIMD4<Float>  // xy = NDC axis-V eye0,  zw = NDC axis-V eye1
  var color: SIMD4<Float>  // pre-multiplied RGBA (linear)
}  // stride = 80 ✓

// ─── Global uniforms ─────────────────────────────────────────────────────────
struct HuashanUniforms {
  var eye0: HuashanPerEyeUniforms
  var eye1: HuashanPerEyeUniforms
  var splatCount: UInt32
  var viewCount: UInt32
  var splatScale: Float  // global scale multiplier for debugging
  var _pad: Float = 0
}
