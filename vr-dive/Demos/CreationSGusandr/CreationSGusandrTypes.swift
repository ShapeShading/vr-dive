import simd

/// Must stay in sync with the Metal struct CreationSGusandrUniforms in CreationSGusandrShaders.metal.
struct CreationSGusandrUniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}