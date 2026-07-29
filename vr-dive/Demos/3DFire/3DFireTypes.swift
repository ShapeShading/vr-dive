import simd

/// Must stay in sync with the Metal struct ThreeDFireUniforms in 3DFireShaders.metal.
struct ThreeDFireUniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
