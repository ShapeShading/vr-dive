import Metal
import simd

final class SimoneOrbit3DRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .simoneOrbit3D
  let preferredClearColor = MTLClearColor(red: 0.002, green: 0.003, blue: 0.006, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let orbitBuffer: MTLBuffer
  private let maxViewCount: Int

  // Orbit simulation config
  private static let pointsPerSeed = 4000
  private static let warmupSteps   = 500
  private static let seedVectors: [SIMD3<Float>] = [
    SIMD3<Float>( 0.12, -0.09,  0.04),
    SIMD3<Float>(-0.18,  0.06,  0.11),
    SIMD3<Float>( 0.07,  0.15, -0.13),
  ]
  private static let totalPoints = pointsPerSeed * seedVectors.count

  private let orbitScale: Float = 0.72   // maps sin/cos [-1,1] to local space
  private let cubeScale:  Float = 0.46   // world-space object size
  private let objectCenter = SIMD3<Float>(0.0, 0.0, -2.0)

  private var animationTime: Float = 0
  private var lastSimulationTime: Float?
  private var currentPreset: SimoneOrbit3DPreset = .preset01

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    orbitBuffer = device.makeBuffer(
      length: MemoryLayout<OrbitPointVertex>.stride * Self.totalPoints,
      options: .storageModeShared)!

    pipelineState = try SimoneOrbit3DRenderer.makePipelineState(
      device: device,
      library: library,
      maxViewCount: self.maxViewCount)
    depthStencilState = SimoneOrbit3DRenderer.makeDepthStencilState(device: device)
  }

  func synchronizeState(_ context: PatternSimulationContext) {
    currentPreset = context.simoneOrbit3DPreset
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    defer { lastSimulationTime = context.time }
    guard let lastSimulationTime else { return }
    let deltaTime = max(0, min(context.time - lastSimulationTime, 1.0 / 20.0))
    animationTime += deltaTime * max(context.speedMultiplier, 0.0) * 0.18
  }

  func resetToInitialState() {
    animationTime = 0
    lastSimulationTime = nil
  }

  // MARK: - CPU orbit simulation

  private func simoneMap(_ p: SIMD3<Float>, _ params: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3<Float>(
      sin(p.x * p.x - p.y * p.y - p.z * p.z + params.x),
      cos(2 * p.x * p.y + params.y),
      sin(2 * p.x * p.z + params.z))
  }

  private func rebuildOrbit() {
    let params = currentPreset.parameters
    let ptr = orbitBuffer.contents().bindMemory(
      to: OrbitPointVertex.self, capacity: Self.totalPoints)
    var writeIndex = 0
    let inv = 1.0 / Float(Self.pointsPerSeed)
    for seed in Self.seedVectors {
      var s = seed
      for _ in 0..<Self.warmupSteps { s = simoneMap(s, params) }
      for i in 0..<Self.pointsPerSeed {
        s = simoneMap(s, params)
        ptr[writeIndex] = OrbitPointVertex(
          x: s.x * orbitScale,
          y: s.y * orbitScale,
          z: s.z * orbitScale,
          brightness: 0.35 + 0.65 * Float(i) * inv)
        writeIndex += 1
      }
    }
  }

  // MARK: - Render

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    rebuildOrbit()

    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(orbitBuffer, offset: 0, index: 0)

    var uniforms = SimoneOrbit3DUniforms(
      time: animationTime,
      viewCount: UInt32(context.viewData.viewCount),
      cubeScale: cubeScale,
      padding: 0,
      simoneParameters: SIMD4<Float>(
        currentPreset.parameters.x,
        currentPreset.parameters.y,
        currentPreset.parameters.z,
        0),
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0))

    encoder.setVertexBytes(
      &uniforms,
      length: MemoryLayout<SimoneOrbit3DUniforms>.stride,
      index: 1)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 2)
      }
    }

    encoder.setFragmentBytes(
      &uniforms, length: MemoryLayout<SimoneOrbit3DUniforms>.stride, index: 0)

    encoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: Self.totalPoints)
  }
}

extension SimoneOrbit3DRenderer {
  fileprivate static func makePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction  = library.makeFunction(name: "simoneOrbitPointVertex")
    desc.fragmentFunction = library.makeFunction(name: "simoneOrbitPointFragment")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.depthAttachmentPixelFormat      = .depth32Float

    // Additive blending: overlapping points accumulate brightness
    let ca = desc.colorAttachments[0]!
    ca.isBlendingEnabled          = true
    ca.rgbBlendOperation          = .add
    ca.alphaBlendOperation        = .add
    ca.sourceRGBBlendFactor       = .one
    ca.sourceAlphaBlendFactor     = .one
    ca.destinationRGBBlendFactor  = .one
    ca.destinationAlphaBlendFactor = .one

    desc.maxVertexAmplificationCount = max(maxViewCount, 1)
    return try device.makeRenderPipelineState(descriptor: desc)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let desc = MTLDepthStencilDescriptor()
    desc.depthCompareFunction = .greater  // reversed-Z convention
    desc.isDepthWriteEnabled  = false     // points don't occlude each other
    return device.makeDepthStencilState(descriptor: desc)!
  }
}