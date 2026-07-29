import simd

/// Must stay in sync with the Metal struct WaveySpheresUniforms in
/// WaveySpheresShaders.metal.
struct WaveySpheresUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var travelSpeed: Float
  var objectCenter: SIMD4<Float>
}
