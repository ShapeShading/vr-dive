import simd

/// Layout must stay in sync with the Metal struct DynamicBoxUniforms in shaders.
struct DynamicBoxUniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float
  var _pad: Float
  var objectCenter: SIMD4<Float>
  var patternTransform: simd_float4x4
}
