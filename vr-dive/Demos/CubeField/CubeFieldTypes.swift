import simd

struct ObjectState {
  var positionAndType: SIMD4<Float>
  var motionAndPhase: SIMD4<Float>
  var scaleAndPadding: SIMD4<Float>
  var homeAndJitter: SIMD4<Float>
}

struct SimulationUniforms {
  var deltaTime: Float
  var globalTime: Float
  var objectCount: UInt32
  var padding: UInt32 = 0
}
