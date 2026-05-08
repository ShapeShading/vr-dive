import simd

/// Must stay in sync with the Metal struct TorusKnotInR4Uniforms in TorusKnotInR4Shaders.metal.
struct TorusKnotInR4Uniforms {
  var time: Float
  var viewCount: UInt32
  var boxScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}