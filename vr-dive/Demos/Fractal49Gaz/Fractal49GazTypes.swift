import simd

/// Must stay in sync with the Metal struct Fractal49GazUniforms in Fractal49GazShaders.metal.
struct Fractal49GazUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}