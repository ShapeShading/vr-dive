import simd

/// Must stay in sync with the Metal struct TunnelUniforms in TunnelShaders.metal.
struct TunnelUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var travelSpeed: Float
  var objectCenter: SIMD4<Float>
}
