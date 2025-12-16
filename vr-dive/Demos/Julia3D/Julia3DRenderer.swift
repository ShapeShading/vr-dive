import Metal
import simd

final class Julia3DRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .julia3D
  let preferredClearColor = MTLClearColor(red: 0.01, green: 0.01, blue: 0.04, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let maxViewCount: Int

  private var uniforms: Julia3DUniforms
  private var juliaC = SIMD4<Float>(-0.35, 0.65, 0.11, 0.0)

  init(device: MTLDevice, library: MTLLibrary, particleCount: Int, maxViewCount: Int) throws {
    _ = particleCount  // legacy argument retained for factory compatibility
    self.maxViewCount = max(1, maxViewCount)

    pipelineState = try Julia3DRenderer.makePipelineState(
      device: device,
      library: library,
      maxViewCount: self.maxViewCount
    )
    depthStencilState = Julia3DRenderer.makeDepthStencilState(device: device)

    uniforms = Julia3DUniforms(
      globalTime: 0,
      maxRaySteps: 160,
      iterationCount: 16,
      juliaC: juliaC,
      worldScale: 1.35,
      escapeRadius: 6.0,
      surfaceEpsilon: 0.0008,
      maxDistance: 40.0,
      ambientStrength: 0.28,
      glowStrength: 0.55,
      aoStrength: 0.4,
      animationSpeed: 0.12
    )
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    uniforms.globalTime = context.time
    juliaC = animateJuliaParameter(time: context.time)
    uniforms.juliaC = juliaC
  }

  func resetToInitialState() {
    uniforms.globalTime = 0
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    var sceneUniforms = SceneUniforms(
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount)
    )
    var viewUniforms = makeViewUniforms(context: context)
    if viewUniforms.isEmpty {
      viewUniforms = [
        Julia3DViewUniform(
          viewToWorld: matrix_identity_float4x4,
          projectionInverse: matrix_identity_float4x4
        )
      ]
      sceneUniforms.layerCount = 1
    }

    context.applyViewConfiguration(on: encoder)

    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)

    encoder.setVertexBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 0)
    encoder.setVertexBytes(
      viewUniforms,
      length: MemoryLayout<Julia3DViewUniform>.stride * viewUniforms.count,
      index: 1
    )

    var drawUniforms = uniforms
    encoder.setFragmentBytes(
      viewUniforms,
      length: MemoryLayout<Julia3DViewUniform>.stride * viewUniforms.count,
      index: 0
    )
    encoder.setFragmentBytes(&drawUniforms, length: MemoryLayout<Julia3DUniforms>.stride, index: 1)

    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
  }

  private func animateJuliaParameter(time: Float) -> SIMD4<Float> {
    let t = time * uniforms.animationSpeed
    let cx = -0.35 + 0.25 * sin(t * 0.7)
    let cy = 0.6 + 0.15 * cos(t * 0.5)
    let cz = 0.15 * sin(t * 0.9 + 1.2)
    let cw = 0.05 * cos(t * 0.6 + 0.3)
    return SIMD4<Float>(cx, cy, cz, cw)
  }

  private func makeViewUniforms(context: PatternRenderContext) -> [Julia3DViewUniform] {
    let desiredViewCount = min(context.viewData.viewCount, maxViewCount)
    guard desiredViewCount > 0 else { return [] }

    var result: [Julia3DViewUniform] = []
    result.reserveCapacity(desiredViewCount)

    for index in 0..<desiredViewCount {
      let viewToWorld =
        index < context.viewData.viewToWorldTransforms.count
        ? context.viewData.viewToWorldTransforms[index]
        : matrix_identity_float4x4
      let viewProjection =
        index < context.viewData.viewProjectionMatrices.count
        ? context.viewData.viewProjectionMatrices[index]
        : matrix_identity_float4x4
      let projectionMatrix = viewProjection * viewToWorld
      let inverseProjection = projectionMatrix.inverseOrIdentity
      result.append(
        Julia3DViewUniform(viewToWorld: viewToWorld, projectionInverse: inverseProjection))
    }

    return result
  }
}

extension simd_float4x4 {
  fileprivate var inverseOrIdentity: simd_float4x4 {
    let determinant = simd_determinant(self)
    if abs(determinant) < 1e-6 {
      return matrix_identity_float4x4
    }
    return simd_inverse(self)
  }
}

extension Julia3DRenderer {
  fileprivate static func makePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "julia3DRaymarchVertex")
    descriptor.fragmentFunction = library.makeFunction(name: "julia3DRaymarchFragment")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.colorAttachments[0].isBlendingEnabled = true
    descriptor.colorAttachments[0].rgbBlendOperation = .add
    descriptor.colorAttachments[0].alphaBlendOperation = .add
    descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
    descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
    descriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
    descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .triangle
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = false
    return device.makeDepthStencilState(descriptor: descriptor)!
  }
}
