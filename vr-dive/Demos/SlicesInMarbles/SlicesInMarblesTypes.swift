import simd

/// Must stay in sync with the Metal struct SlicesInMarblesUniforms in
/// SlicesInMarblesShaders.metal.
struct SlicesInMarblesUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
