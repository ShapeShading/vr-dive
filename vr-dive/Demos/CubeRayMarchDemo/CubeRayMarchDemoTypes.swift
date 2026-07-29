import simd

/// Synced with CRMUniforms in CubeRayMarchDemoShaders.metal
struct CRMMeshUniforms {
  var time: Float
  var viewCount: UInt32
  var pad0: SIMD2<Float> = .zero
  var objectCenter: SIMD4<Float>
  var lightPosition: SIMD4<Float>
}
