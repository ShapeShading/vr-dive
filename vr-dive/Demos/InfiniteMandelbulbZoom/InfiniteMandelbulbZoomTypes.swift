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
  /// Reverse-Z depth written by the screen-space composite pass. A value of
  /// exactly zero is the far-plane clear value and is not valid presentation
  /// geometry for CompositorServices reprojection on device.
  var compositeDepth: Float
  /// Canonical view-space camera position and fractal scale. Keeping the scene
  /// view-relative prevents navigation/world-origin changes from moving it off-screen.
  var cameraAndScale: SIMD4<Float>
}
