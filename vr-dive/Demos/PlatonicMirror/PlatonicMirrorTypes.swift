import simd

/// Must stay in sync with the Metal struct PlatonicMirrorUniforms in PlatonicMirrorShaders.metal.
struct PlatonicMirrorUniforms {
  var time: Float
  var viewCount: UInt32
  var solidScale: Float  // world metres per local unit
  var _pad: Float
  var objectCenter: SIMD4<Float>  // xyz = world position of solid centre
}
