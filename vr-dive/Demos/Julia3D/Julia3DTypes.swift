import simd

struct Julia3DUniforms {
  // Global animation
  var globalTime: Float
  var maxRaySteps: UInt32
  var iterationCount: UInt32
  var padding: UInt32 = 0

  // Julia set parameter c (quaternion)
  var juliaC: SIMD4<Float>

  // Rendering parameters
  var worldScale: Float
  var escapeRadius: Float
  var surfaceEpsilon: Float
  var maxDistance: Float

  // Shading controls
  var ambientStrength: Float
  var glowStrength: Float
  var aoStrength: Float
  var animationSpeed: Float
}

struct Julia3DViewUniform {
  var viewToWorld: simd_float4x4
  var projectionInverse: simd_float4x4
}
