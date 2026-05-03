import simd

/// Must stay in sync with the Metal struct InterferenceCascadeCubeUniforms in
/// InterferenceCascadeCubeShaders.metal.
struct InterferenceCascadeCubeUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var travelSpeed: Float
  var objectCenter: SIMD4<Float>
}