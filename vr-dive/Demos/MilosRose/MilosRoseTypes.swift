import simd

/// Must stay in sync with the Metal struct MilosRoseUniforms in MilosRoseShaders.metal.
struct MilosRoseUniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}