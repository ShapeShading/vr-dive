import Metal
import simd

final class QuatPolynomialRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .quatPolynomial
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let renderPipelineState: MTLRenderPipelineState
  private let computePipelineState: MTLComputePipelineState
  private let depthStencilState: MTLDepthStencilState
  private let meshVertexBuffer: MTLBuffer
  private let meshIndexBuffer: MTLBuffer
  private let meshIndexCount: Int
  private let particleStateBuffer: MTLBuffer
  private let particleCount: Int
  private let maxViewCount: Int

  // Grid layout — must match the #define constants in the metal file.
  private static let theta1Grid  = 64
  private static let theta2Grid  = 64
  private static let polyDegree  = 5
  private static let circlePoints = 20
  // Total: 64 × 64 × 5 × 20 = 409 600 particles

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount  = max(1, maxViewCount)
    self.particleCount = Self.theta1Grid * Self.theta2Grid * Self.polyDegree * Self.circlePoints

    renderPipelineState  = try QuatPolynomialRenderer.makeRenderPipeline(
      device: device, library: library, maxViewCount: self.maxViewCount)
    computePipelineState = try QuatPolynomialRenderer.makeComputePipeline(
      device: device, library: library)
    depthStencilState    = QuatPolynomialRenderer.makeDepthStencilState(device: device)

    let geo = MeshGeometryFactory.makeOctahedron(device: device)
    meshVertexBuffer = geo.vertexBuffer
    meshIndexBuffer  = geo.indexBuffer
    meshIndexCount   = geo.indexCount

    particleStateBuffer = device.makeBuffer(
      length: MemoryLayout<QuatPolynomialParticleState>.stride * particleCount,
      options: .storageModeShared)!
  }

  func resetToInitialState() {}

  func updateSimulation(_ context: PatternSimulationContext) {
    guard !context.isPaused else { return }

    var uniforms = QuatPolynomialUniforms(
      time: context.time,
      speed: context.speedMultiplier * 0.15,
      worldScale: 0.22,
      particleCount: UInt32(particleCount)
    )

    guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder()
    else { return }

    encoder.setComputePipelineState(computePipelineState)
    encoder.setBuffer(particleStateBuffer, offset: 0, index: 0)
    encoder.setBytes(&uniforms, length: MemoryLayout<QuatPolynomialUniforms>.stride, index: 1)

    let threadsPerGroup = min(computePipelineState.maxTotalThreadsPerThreadgroup, 128)
    let threadgroups = MTLSize(
      width: (particleCount + threadsPerGroup - 1) / threadsPerGroup,
      height: 1, depth: 1)
    encoder.dispatchThreadgroups(
      threadgroups,
      threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(renderPipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)

    encoder.setVertexBuffer(meshVertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(particleStateBuffer, offset: 0, index: 1)

    context.applyViewConfiguration(on: encoder)

    var sceneUniforms = SceneUniforms(
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount))
    encoder.setVertexBytes(
      &sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 2)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 3)
      }
    }

    encoder.setFragmentBytes(
      &sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 0)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: meshIndexCount,
      indexType: .uint16,
      indexBuffer: meshIndexBuffer,
      indexBufferOffset: 0,
      instanceCount: particleCount
    )
  }
}

extension QuatPolynomialRenderer {
  fileprivate static func makeRenderPipeline(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction   = library.makeFunction(name: "quatPolyVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "quatPolyFragmentShader")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .triangle

    let vd = MTLVertexDescriptor()
    vd.attributes[0].format      = .float3
    vd.attributes[0].offset      = 0
    vd.attributes[0].bufferIndex = 0
    vd.attributes[1].format      = .float3
    vd.attributes[1].offset      = MemoryLayout<SIMD3<Float>>.stride
    vd.attributes[1].bufferIndex = 0
    vd.layouts[0].stride         = MemoryLayout<MeshVertex>.stride
    descriptor.vertexDescriptor  = vd

    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  fileprivate static func makeComputePipeline(
    device: MTLDevice,
    library: MTLLibrary
  ) throws -> MTLComputePipelineState {
    let fn = library.makeFunction(name: "computeQuatPolyParticles")!
    return try device.makeComputePipelineState(function: fn)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let desc = MTLDepthStencilDescriptor()
    desc.depthCompareFunction = .greater   // reverse-Z
    desc.isDepthWriteEnabled  = true
    return device.makeDepthStencilState(descriptor: desc)!
  }
}
