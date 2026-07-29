import simd

// Layout must match StarTrailsUniforms in StarTrailsShaders.metal.
struct StarTrailsUniforms {
  var time: Float
  var viewCount: UInt32
  var pad0: Float = 0
  var pad1: Float = 0
  var objectCenter: SIMD4<Float>
}
