import simd

/// CPU/GPU contract for the continuously rebased Mandelbulb zoom.
struct InfiniteMandelbulbZoomUniforms {
  var zoomPhase: Float
  var zoomDirection: Float
  var viewCount: UInt32
  var generation: UInt32
  var maxRaySteps: UInt32
  var fractalIterations: UInt32
  var surfaceEpsilon: Float
  var padding: Float = 0
  var objectCenterAndScale: SIMD4<Float>
}

struct InfiniteMandelbulbZoomViewUniform {
  var viewToWorld: simd_float4x4
  var projectionInverse: simd_float4x4
}
