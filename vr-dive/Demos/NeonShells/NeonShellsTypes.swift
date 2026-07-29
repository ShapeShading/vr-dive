import simd

/// Must stay in sync with the Metal struct NeonShellsUniforms in NeonShellsShaders.metal.
struct NeonShellsUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
