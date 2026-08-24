import Metal
import simd

/// An endlessly rebased zoom into a deliberately self-similar Mandelbulb hierarchy.
///
/// `zoomPhase` never grows beyond one. At a layer boundary the renderer increments a
/// wrapping generation counter and returns the GPU to canonical local coordinates.
/// Consequently shader arithmetic retains the same precision at layer one and at a
/// conceptually unlimited depth.
final class InfiniteMandelbulbZoomRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .infiniteMandelbulbZoom
  let preferredClearColor = MTLClearColor(red: 0.002, green: 0.004, blue: 0.012, alpha: 1)

  private static let raymarchWidth = 384
  private static let raymarchHeight = 304

  private let pipelineState: MTLRenderPipelineState
  private let computePipelineState: MTLComputePipelineState
  private let depthStencilState: MTLDepthStencilState
  private let raymarchTexture: MTLTexture
  private let maxViewCount: Int

  private var zoomPhase: Float = 0
  private var generation: UInt32 = 0
  private var lastSimulationTime: Float?
  private var latestContext: PatternSimulationContext?
  private var didLogRaymarchConfiguration = false

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)
    pipelineState = try Self.makePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    computePipelineState = try device.makeComputePipelineState(
      function: library.makeFunction(name: "infiniteMandelbulbZoomCompute")!)
    depthStencilState = Self.makeDepthStencilState(device: device)

    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba16Float,
      width: Self.raymarchWidth,
      height: Self.raymarchHeight,
      mipmapped: false)
    textureDescriptor.textureType = .type2DArray
    textureDescriptor.arrayLength = self.maxViewCount
    textureDescriptor.storageMode = .private
    textureDescriptor.usage = [.shaderRead, .shaderWrite]
    guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
      throw InfiniteMandelbulbZoomRendererError.textureAllocationFailed
    }
    texture.label = "Infinite Mandelbulb Zoom Raymarch"
    raymarchTexture = texture
  }

  func synchronizeState(_ context: PatternSimulationContext) {
    latestContext = context
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    defer { lastSimulationTime = context.time }
    guard let lastSimulationTime else { return }

    let deltaTime = max(0, min(context.time - lastSimulationTime, 1.0 / 20.0))
    let delta = deltaTime * context.infiniteZoomRate * max(context.speedMultiplier, 0)
      * context.infiniteZoomDirection
    zoomPhase += delta

    while zoomPhase >= 1 {
      zoomPhase -= 1
      generation &+= 1
    }
    while zoomPhase < 0 {
      zoomPhase += 1
      generation &-= 1
    }
  }

  func resetToInitialState() {
    zoomPhase = 0
    generation = 0
    lastSimulationTime = nil
  }

  func encodeComputePrepass(commandBuffer: MTLCommandBuffer, context: PatternRenderContext) {
    var views = makeViewUniforms(context: context)
    if views.isEmpty {
      views = [
        InfiniteMandelbulbZoomViewUniform(
          viewToWorld: matrix_identity_float4x4,
          projectionInverse: matrix_identity_float4x4)
      ]
    }
    var uniforms = makeUniforms(viewCount: views.count)

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
    encoder.label = "Infinite Mandelbulb Zoom Raymarch"
    encoder.setComputePipelineState(computePipelineState)
    encoder.setBytes(
      &uniforms, length: MemoryLayout<InfiniteMandelbulbZoomUniforms>.stride, index: 0)
    views.withUnsafeBytes { bytes in
      guard let base = bytes.baseAddress else { return }
      encoder.setBytes(base, length: bytes.count, index: 1)
    }
    encoder.setTexture(raymarchTexture, index: 0)

    let threadWidth = computePipelineState.threadExecutionWidth
    let threadHeight = max(
      1, min(8, computePipelineState.maxTotalThreadsPerThreadgroup / threadWidth))
    encoder.dispatchThreads(
      MTLSize(width: Self.raymarchWidth, height: Self.raymarchHeight, depth: views.count),
      threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1))
    encoder.endEncoding()

    if !didLogRaymarchConfiguration {
      didLogRaymarchConfiguration = true
      print(
        "[InfiniteMandelbulbZoom] Offscreen raymarch active: \(Self.raymarchWidth)x\(Self.raymarchHeight)x\(views.count)"
      )
    }
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    var views = makeViewUniforms(context: context)
    if views.isEmpty {
      views = [
        InfiniteMandelbulbZoomViewUniform(
          viewToWorld: matrix_identity_float4x4,
          projectionInverse: matrix_identity_float4x4)
      ]
    }

    var uniforms = makeUniforms(viewCount: views.count)

    context.applyViewConfiguration(on: encoder)
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)

    encoder.setVertexBytes(
      &uniforms, length: MemoryLayout<InfiniteMandelbulbZoomUniforms>.stride, index: 0)
    encoder.setFragmentTexture(raymarchTexture, index: 0)
    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
  }

  private func makeUniforms(viewCount: Int) -> InfiniteMandelbulbZoomUniforms {
    let settings = latestContext
    let quality = settings?.infiniteZoomQuality ?? .balanced
    return InfiniteMandelbulbZoomUniforms(
      zoomPhase: zoomPhase,
      zoomDirection: settings?.infiniteZoomDirection ?? 1,
      viewCount: UInt32(viewCount),
      generation: generation,
      maxRaySteps: quality.raySteps,
      fractalIterations: quality.fractalIterations,
      surfaceEpsilon: quality == .detailed ? 0.0009 : 0.0013,
      cameraAndScale: SIMD4<Float>(0, 0, 2.75, 1.0))
  }

  private func makeViewUniforms(context: PatternRenderContext) -> [InfiniteMandelbulbZoomViewUniform] {
    let count = min(context.viewData.viewCount, maxViewCount)
    guard count > 0 else { return [] }
    return (0..<count).map { index in
      let viewToWorld = context.viewData.viewToWorldTransforms.indices.contains(index)
        ? context.viewData.viewToWorldTransforms[index] : matrix_identity_float4x4
      let viewProjection = context.viewData.viewProjectionMatrices.indices.contains(index)
        ? context.viewData.viewProjectionMatrices[index] : matrix_identity_float4x4
      let projection = viewProjection * viewToWorld
      let determinant = simd_determinant(projection)
      return InfiniteMandelbulbZoomViewUniform(
        viewToWorld: viewToWorld,
        projectionInverse: abs(determinant) > 1e-6 ? simd_inverse(projection) : matrix_identity_float4x4)
    }
  }
}

private enum InfiniteMandelbulbZoomRendererError: Error {
  case textureAllocationFailed
}

extension InfiniteMandelbulbZoomRenderer {
  fileprivate static func makePipelineState(
    device: MTLDevice, library: MTLLibrary, maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "infiniteMandelbulbZoomVertex")
    descriptor.fragmentFunction = library.makeFunction(name: "infiniteMandelbulbZoomFragment")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .triangle
    descriptor.maxVertexAmplificationCount = max(1, maxViewCount)
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    // This is a full-screen ray-marching pass: its triangle lies at depth zero,
    // the same as the reverse-Z clear value. A greater test would reject every
    // fragment and present a black immersive view.
    descriptor.depthCompareFunction = .always
    descriptor.isDepthWriteEnabled = false
    return device.makeDepthStencilState(descriptor: descriptor)!
  }
}
