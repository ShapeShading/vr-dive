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

/// Must match Metal's `packed_float3 position; float brightness;` (16 bytes total).
struct OrbitPointVertex {
  var x: Float
  var y: Float
  var z: Float
  var brightness: Float
}
