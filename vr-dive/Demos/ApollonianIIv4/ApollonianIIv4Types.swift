import simd

/// Must stay in sync with the Metal struct ApollonianIIv4Uniforms in
/// ApollonianIIv4Shaders.metal.
struct ApollonianIIv4Uniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var travelSpeed: Float
  var objectCenter: SIMD4<Float>
}
