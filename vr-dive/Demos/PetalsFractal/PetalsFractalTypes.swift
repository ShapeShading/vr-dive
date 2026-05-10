import simd

/// Must stay in sync with the Metal struct PetalsFractalUniforms in
/// PetalsFractalShaders.metal.
struct PetalsFractalUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
