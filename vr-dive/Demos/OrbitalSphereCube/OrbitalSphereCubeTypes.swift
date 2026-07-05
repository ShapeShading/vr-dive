import simd

/// Must stay in sync with the Metal struct OrbitalSphereCubeUniforms in
/// OrbitalSphereCubeShaders.metal.
struct OrbitalSphereCubeUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var travelSpeed: Float
  var objectCenter: SIMD4<Float>
  /// Pattern-space navigation transform (identity in normal mode).
  /// Applied to the cube-local eye/ray so the rendered scene shifts
  /// while the cube stays fixed in world space.
  var patternTransform: simd_float4x4
}
