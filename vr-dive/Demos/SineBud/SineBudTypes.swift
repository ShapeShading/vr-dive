import simd

/// Must stay in sync with the Metal struct SineBudUniforms in SineBudShaders.metal.
struct SineBudUniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
