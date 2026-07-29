import Metal
import simd

struct MeshGeometryFactory {
  static func makeIcosahedron(
    device: MTLDevice,
    size: Float = 0.03
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    let phi: Float = 1.618_034
    let rawRadius = sqrt(1 + phi * phi)
    let scale = size / rawRadius

    let positions: [SIMD3<Float>] = [
      SIMD3<Float>(-1, phi, 0),
      SIMD3<Float>(1, phi, 0),
      SIMD3<Float>(-1, -phi, 0),
      SIMD3<Float>(1, -phi, 0),
      SIMD3<Float>(0, -1, phi),
      SIMD3<Float>(0, 1, phi),
      SIMD3<Float>(0, -1, -phi),
      SIMD3<Float>(0, 1, -phi),
      SIMD3<Float>(phi, 0, -1),
      SIMD3<Float>(phi, 0, 1),
      SIMD3<Float>(-phi, 0, -1),
      SIMD3<Float>(-phi, 0, 1),
    ].map { $0 * scale }

    let vertices: [MeshVertex] = positions.map { position in
      let normal = simd_normalize(position)
      return MeshVertex(position: position, normal: normal)
    }

    let indices: [UInt16] = [
      0, 11, 5,
      0, 5, 1,
      0, 1, 7,
      0, 7, 10,
      0, 10, 11,
      1, 5, 9,
      5, 11, 4,
      11, 10, 2,
      10, 7, 6,
      7, 1, 8,
      3, 9, 4,
      3, 4, 2,
      3, 2, 6,
      3, 6, 8,
      3, 8, 9,
      4, 9, 5,
      2, 4, 11,
      6, 2, 10,
      8, 6, 7,
      9, 8, 1,
    ]

    let vertexBuffer = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<MeshVertex>.stride * vertices.count,
      options: [.storageModeShared]
    )!

    let indexBuffer = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: [.storageModeShared]
    )!

    return (vertexBuffer, indexBuffer, indices.count)
  }

  static func makeOctahedron(
    device: MTLDevice,
    size: Float = 0.03
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    let h = size
    let positions: [SIMD3<Float>] = [
      SIMD3<Float>(0, h, 0),  // Top
      SIMD3<Float>(h, 0, 0),  // Right
      SIMD3<Float>(0, 0, h),  // Front
      SIMD3<Float>(-h, 0, 0),  // Left
      SIMD3<Float>(0, 0, -h),  // Back
      SIMD3<Float>(0, -h, 0),  // Bottom
    ]

    let vertices: [MeshVertex] = positions.map { position in
      let normal = simd_normalize(position)
      return MeshVertex(position: position, normal: normal)
    }

    let indices: [UInt16] = [
      // Top pyramid
      0, 2, 1,
      0, 3, 2,
      0, 4, 3,
      0, 1, 4,
      // Bottom pyramid
      5, 1, 2,
      5, 2, 3,
      5, 3, 4,
      5, 4, 1,
    ]

    let vertexBuffer = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<MeshVertex>.stride * vertices.count,
      options: [.storageModeShared]
    )!

    let indexBuffer = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: [.storageModeShared]
    )!

    return (vertexBuffer, indexBuffer, indices.count)
  }
}
