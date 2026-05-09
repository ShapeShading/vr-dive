import simd

/// Must stay in sync with the Metal struct PlayingMarbleUniforms in PlayingMarbleShaders.metal.
struct PlayingMarbleUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
