import simd

struct QuatPolynomialParticleState {
  var positionAndScale: SIMD4<Float>  // xyz = world position, w = visual scale
  var color: SIMD4<Float>
}

struct QuatPolynomialUniforms {
  var time: Float
  var speed: Float
  var worldScale: Float
  var particleCount: UInt32
}
