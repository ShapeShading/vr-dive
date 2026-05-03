import simd

/// Must stay in sync with the Metal struct SynthwaveSunsetUniforms in
/// SynthwaveSunsetShaders.metal.
struct SynthwaveSunsetUniforms {
  var time: Float
  var viewCount: UInt32
  var sceneSpeed: Float
  var _pad: Float
  var objectCenter: SIMD4<Float>
  var viewerOffset: SIMD4<Float>
  var boxHalfExtents: SIMD4<Float>
}
