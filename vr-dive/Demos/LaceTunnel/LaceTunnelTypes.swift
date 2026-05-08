import simd

/// Must stay in sync with the Metal struct LaceTunnelUniforms in LaceTunnelShaders.metal.
struct LaceTunnelUniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float  // uniform world-space scale applied to the local ±1 half-extents
  var _pad: Float      // padding to keep float4 aligned
  var objectCenter: SIMD4<Float>  // xyz = world position of box centre
}
