import simd

/// Must stay in sync with the Metal struct EtherUniforms in EtherShaders.metal.
struct EtherUniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
