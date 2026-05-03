import simd

/// Must stay in sync with the Metal struct PathTilesCubeUniforms in
/// PathTilesCubeShaders.metal.
struct PathTilesCubeUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var travelSpeed: Float
  var objectCenter: SIMD4<Float>
}