import simd

/// Must stay in sync with the Metal struct DigitalLinesUniforms in DigitalLinesShaders.metal.
struct DigitalLinesUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
