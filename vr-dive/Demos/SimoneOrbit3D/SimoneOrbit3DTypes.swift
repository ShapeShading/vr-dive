import simd

/// Must stay in sync with the Metal struct SimoneOrbit3DUniforms in
/// SimoneOrbit3DShaders.metal.
struct SimoneOrbit3DUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var simoneParameters: SIMD4<Float>
  var objectCenter: SIMD4<Float>
}