import simd

/// Must stay in sync with the Metal struct FireTornadoUniforms in FireTornadoShaders.metal.
struct FireTornadoUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
