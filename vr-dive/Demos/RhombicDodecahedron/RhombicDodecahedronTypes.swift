import simd

struct RhombicDodecahedronUniforms {
  var time: Float
  var viewCount: UInt32
  var roomScale: Float  // half-distance from centre to each face
  var reflectionBounces: UInt32
  var objectCenter: SIMD4<Float>  // xyz = world position of dodecahedron centre
}
