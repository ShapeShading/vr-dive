import simd

/// Must stay in sync with the Metal struct Fractal77GazUniforms in Fractal77GazShaders.metal.
struct Fractal77GazUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}