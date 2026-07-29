import simd

/// Must stay in sync with the Metal struct ReflectiveWythoffPolyhedraUniforms in
/// ReflectiveWythoffPolyhedraShaders.metal.
struct ReflectiveWythoffPolyhedraUniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
