import Metal
import simd

final class MetaballRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .metaball
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int
  private let maxViewCount: Int

  // Bounding sphere radius — slightly larger than the maximum metaball cluster extent.
  // Balls wander up to 0.255 m + 0.065 m radius = 0.32 m, so 0.42 m gives comfortable margin.
  private let boundingRadius: Float = 0.42

  // World-space position: straight ahead, roughly eye-level.
  private let objectCenter = SIMD3<Float>(0.0, 0.0, -1.0)

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    let geo = MetaballRenderer.makeUVSphere(
      device: device, radius: boundingRadius, latSegments: 24, lonSegments: 48)
    vertexBuffer = geo.vertexBuffer
    indexBuffer = geo.indexBuffer
    indexCount = geo.indexCount

    pipelineState = try MetaballRenderer.makePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    depthStencilState = MetaballRenderer.makeDepthStencilState(device: device)
  }

  func updateSimulation(_ context: PatternSimulationContext) {}
  func resetToInitialState() {}

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)

    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    var uniforms = MetaballUniforms(
      time: context.time,
      viewCount: UInt32(context.viewData.viewCount),
      boundingRadius: boundingRadius,
      padding: 0,
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0)
    )

    encoder.setVertexBytes(
      &uniforms, length: MemoryLayout<MetaballUniforms>.stride, index: 1)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 2)
      }
    }

    encoder.setFragmentBytes(
      &uniforms, length: MemoryLayout<MetaballUniforms>.stride, index: 0)

    var viewToWorld = context.viewData.viewToWorldTransforms
    if viewToWorld.isEmpty { viewToWorld = [matrix_identity_float4x4] }
    viewToWorld.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setFragmentBytes(base, length: $0.count, index: 1)
      }
    }

    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setFragmentBytes(base, length: $0.count, index: 2)
      }
    }

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexCount,
      indexType: .uint16,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0
    )
  }
}

extension MetaballRenderer {
  // UV sphere centred at the origin.  latSegments × lonSegments quads.
  // (latSegments+1) × (lonSegments+1) ≤ 25×49 = 1225 vertices → fits UInt16.
  fileprivate static func makeUVSphere(
    device: MTLDevice,
    radius: Float,
    latSegments: Int,
    lonSegments: Int
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    var vertices: [MeshVertex] = []
    var indices: [UInt16] = []
    vertices.reserveCapacity((latSegments + 1) * (lonSegments + 1))
    indices.reserveCapacity(latSegments * lonSegments * 6)

    for lat in 0...latSegments {
      let theta = Float(lat) * .pi / Float(latSegments)
      let sinTheta = sin(theta)
      let cosTheta = cos(theta)
      for lon in 0...lonSegments {
        let phi = Float(lon) * 2 * .pi / Float(lonSegments)
        let n = SIMD3<Float>(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta)
        vertices.append(MeshVertex(position: n * radius, normal: n))
      }
    }

    let stride = UInt16(lonSegments + 1)
    for lat in 0..<latSegments {
      for lon in 0..<lonSegments {
        let a = UInt16(lat) * stride + UInt16(lon)
        let b = a + stride
        indices.append(contentsOf: [a, b, a + 1, b, b + 1, a + 1])
      }
    }

    let vBuf = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<MeshVertex>.stride * vertices.count,
      options: .storageModeShared)!
    let iBuf = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: .storageModeShared)!
    return (vBuf, iBuf, indices.count)
  }

  fileprivate static func makePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "metaballVertex")
    descriptor.fragmentFunction = library.makeFunction(name: "metaballFragment")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float

    let vd = MTLVertexDescriptor()
    vd.attributes[0].format = .float3
    vd.attributes[0].offset = 0
    vd.attributes[0].bufferIndex = 0
    vd.attributes[1].format = .float3
    vd.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vd.attributes[1].bufferIndex = 0
    vd.layouts[0].stride = MemoryLayout<MeshVertex>.stride
    descriptor.vertexDescriptor = vd

    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let desc = MTLDepthStencilDescriptor()
    desc.depthCompareFunction = .greater  // reverse-Z
    desc.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: desc)!
  }
}
