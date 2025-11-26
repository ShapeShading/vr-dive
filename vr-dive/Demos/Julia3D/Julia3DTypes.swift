import simd

struct Julia3DParticleState {
  var positionAndScale: SIMD4<Float>
  var seedAndPhase: SIMD4<Float>
}

struct Julia3DUniforms {
  var deltaTime: Float
  var globalTime: Float
  var particleCount: UInt32
  var sigma: Float
  var beta: Float
  var rho: Float
  var damping: Float
  var worldScale: Float
  var resetRadius: Float
  var noiseAmplitude: Float
  var padding: Float = 0
}
