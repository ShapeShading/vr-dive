import simd

/// Must stay in sync with the Metal struct MirrorLoopingUniforms in MirrorLoopingShaders.metal.
struct MirrorLoopingUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}