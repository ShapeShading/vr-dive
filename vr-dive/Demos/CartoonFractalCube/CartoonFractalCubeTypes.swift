import simd

/// Must stay in sync with the Metal struct CartoonFractalCubeUniforms in
/// CartoonFractalCubeShaders.metal.
struct CartoonFractalCubeUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var travelSpeed: Float
  var objectCenter: SIMD4<Float>
}
