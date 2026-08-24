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

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let maxViewCount: Int

  private var zoomPhase: Float = 0
  private var generation: UInt32 = 0
  private var lastSimulationTime: Float?
  private var latestContext: PatternSimulationContext?

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)
    pipelineState = try Self.makePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    depthStencilState = Self.makeDepthStencilState(device: device)
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

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    let settings = latestContext
    let quality = settings?.infiniteZoomQuality ?? .balanced
    var views = makeViewUniforms(context: context)
    if views.isEmpty {
      views = [
        InfiniteMandelbulbZoomViewUniform(
          viewToWorld: matrix_identity_float4x4,
          projectionInverse: matrix_identity_float4x4)
      ]
    }

    var uniforms = InfiniteMandelbulbZoomUniforms(
      zoomPhase: zoomPhase,
      zoomDirection: settings?.infiniteZoomDirection ?? 1,
      viewCount: UInt32(views.count),
      generation: generation,
      maxRaySteps: quality.raySteps,
      fractalIterations: quality.fractalIterations,
      surfaceEpsilon: quality == .detailed ? 0.00065 : 0.001,
      cameraAndScale: SIMD4<Float>(0, 0, 2.75, 1.0))

    context.applyViewConfiguration(on: encoder)
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)

    encoder.setVertexBytes(
      &uniforms, length: MemoryLayout<InfiniteMandelbulbZoomUniforms>.stride, index: 0)
    encoder.setFragmentBytes(
      &uniforms, length: MemoryLayout<InfiniteMandelbulbZoomUniforms>.stride, index: 0)
    views.withUnsafeBytes { bytes in
      guard let base = bytes.baseAddress else { return }
      encoder.setFragmentBytes(base, length: bytes.count, index: 1)
    }
    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
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
