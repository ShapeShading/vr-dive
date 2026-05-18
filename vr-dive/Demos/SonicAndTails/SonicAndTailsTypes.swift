import simd

/// Must stay in sync with the Metal struct SonicAndTailsUniforms in SonicAndTailsShaders.metal.
struct SonicAndTailsUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
