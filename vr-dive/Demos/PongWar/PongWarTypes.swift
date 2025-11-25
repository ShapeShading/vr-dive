import simd

struct PongWarInstanceState {
  var positionAndScale: SIMD4<Float>
  var color: SIMD4<Float>
  var edgeData: SIMD4<Float>
}

struct PongWarUniforms {
  var edgeHighlight: Float
  var baseGlow: Float
  var ballGlow: Float
  var padding: Float = 0
}
