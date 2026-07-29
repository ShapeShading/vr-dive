import simd

/// Must stay in sync with the Metal struct GlassBoxUniforms in GlassBoxShaders.metal.
struct GlassBoxUniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float  // uniform world-space scale applied to the local BOXDIMS
  var _pad: Float  // padding to keep float4 aligned
  var objectCenter: SIMD4<Float>  // xyz = world position of box centre
  /// Pattern-space navigation transform (identity in normal mode).
  /// Applied to the box-local eye/ray to simulate moving the virtual viewpoint
  /// through the interior while the box stays fixed in world space.
  var patternTransform: simd_float4x4
}
