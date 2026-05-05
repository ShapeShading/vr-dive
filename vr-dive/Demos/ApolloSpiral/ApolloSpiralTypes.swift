import simd

/// Must stay in sync with the Metal struct ApolloSpiralUniforms in
/// ApolloSpiralShaders.metal.
struct ApolloSpiralUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
