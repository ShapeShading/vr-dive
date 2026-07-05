import simd

/// Must stay in sync with the Metal struct LunarSurfaceUniforms in
/// LunarSurfaceShaders.metal.
struct LunarSurfaceUniforms {
  var time: Float
  var viewCount: UInt32
  var _pad0: Float
  var _pad1: Float
  var objectCenter: SIMD4<Float>
  var boxHalfExtents: SIMD4<Float>
  /// Pattern-space navigation transform (identity in normal mode).
  /// Applied directly to the moon-local ray origin/direction so the gamepad
  /// "箱内移动" mode lets the player walk across the crater field / look around.
  var patternTransform: simd_float4x4
  /// xy = world-space (x,z) center the baked height map texture is currently
  /// centered on, z = the texture's world-space half-extent, w unused.
  var heightMapParams: SIMD4<Float>
}
