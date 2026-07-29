import simd

/// Must stay in sync with the Metal struct CrystalCubeLatticinioCore1Uniforms in CrystalCubeLatticinioCore1Shaders.metal.
struct CrystalCubeLatticinioCore1Uniforms {
  var time: Float
  var viewCount: UInt32
  var cubeScale: Float
  var padding: Float
  var objectCenter: SIMD4<Float>
}
