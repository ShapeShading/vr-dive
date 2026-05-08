import simd

/// Must stay in sync with the Metal struct TorusFanUniforms in TorusFanShaders.metal.
struct TorusFanUniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float
  var _pad: Float
  var objectCenter: SIMD4<Float>
}