import simd

struct FourWingParticleState {
  var positionAndScale: SIMD4<Float>
  var seedAndPhase: SIMD4<Float>
}

struct FourWingUniforms {
  var deltaTime: Float
  var globalTime: Float
  var particleCount: UInt32
  var a: Float  // 0.2
  var b: Float  // 0.01
  var c: Float  // -0.4
  var damping: Float
  var worldScale: Float
  var resetRadius: Float
  var noiseAmplitude: Float
  var padding: Float = 0
}
