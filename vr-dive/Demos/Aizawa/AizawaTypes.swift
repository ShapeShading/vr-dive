import simd

struct AizawaParticleState {
  var positionAndScale: SIMD4<Float>
  var seedAndPhase: SIMD4<Float>
}

struct AizawaUniforms {
  var deltaTime: Float
  var globalTime: Float
  var particleCount: UInt32
  var a: Float  // 0.95
  var b: Float  // 0.7
  var c: Float  // 0.6
  var d: Float  // 3.5
  var e: Float  // 0.25
  var f: Float  // 0.1
  var damping: Float
  var worldScale: Float
  var resetRadius: Float
  var noiseAmplitude: Float
  var padding: Float = 0
}
