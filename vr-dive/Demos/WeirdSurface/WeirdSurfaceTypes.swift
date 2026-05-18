import simd

/// Must stay in sync with the Metal struct WeirdSurfaceUniforms in WeirdSurfaceShaders.metal.
struct WeirdSurfaceUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
