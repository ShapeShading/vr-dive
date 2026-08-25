import simd

/// CPU/GPU contract for the world-space Mandelbulb box.
struct InfiniteMandelbulbZoomUniforms {
  var zoomPhase: Float
  var viewCount: UInt32
  var generation: UInt32
  var maxRaySteps: UInt32

  var fractalIterations: UInt32
  var surfaceEpsilon: Float
  var boxScale: Float
  var padding: Float

  var objectCenter: SIMD4<Float>
  var patternTransform: simd_float4x4
}
