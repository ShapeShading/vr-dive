import simd

/// Must stay in sync with the Metal struct NearLoxodromeUniforms in
/// NearLoxodromeShaders.metal.
struct NearLoxodromeUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}