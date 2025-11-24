import simd

struct PongWarInstanceState {
  var positionAndScale: SIMD4<Float>
  var color: SIMD4<Float>
  var motion: SIMD4<Float>
}

struct PongWarUniforms {
  var pulseAmplitude: Float
  var pulseSpeed: Float
  var cubeRotationSpeed: Float
  var sphereBobSpeed: Float
  var sphereBobAmount: Float
  var sphereGlow: Float
  var noiseAmount: Float
  var padding: Float = 0
}
