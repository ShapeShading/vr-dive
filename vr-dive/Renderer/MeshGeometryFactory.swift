import Metal
import simd

struct MeshGeometryFactory {
  static func makeOctahedron(
    device: MTLDevice,
    size: Float = 0.03
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    let h = size
    let positions: [SIMD3<Float>] = [
      SIMD3<Float>(0, h, 0),   // Top
      SIMD3<Float>(h, 0, 0),   // Right
      SIMD3<Float>(0, 0, h),   // Front
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
