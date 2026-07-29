import simd

struct MetaballUniforms {
  var time: Float
  var viewCount: UInt32
  var boundingRadius: Float
  var padding: Float = 0
  var objectCenter: SIMD4<Float>
}
