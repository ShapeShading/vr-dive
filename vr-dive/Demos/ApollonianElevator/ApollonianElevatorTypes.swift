import simd

/// Must stay in sync with the Metal struct ApollonianElevatorUniforms in ApollonianElevatorShaders.metal.
struct ApollonianElevatorUniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}