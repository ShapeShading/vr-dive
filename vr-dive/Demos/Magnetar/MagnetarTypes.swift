import simd

/// Must stay in sync with the Metal struct MagnetarUniforms in
/// MagnetarShaders.metal.
struct MagnetarUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
