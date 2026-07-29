import Metal
import simd

final class SimoneOrbit3DRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .simoneOrbit3D
  let preferredClearColor = MTLClearColor(red: 0.002, green: 0.003, blue: 0.006, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let meshVertexBuffer: MTLBuffer
  private let meshIndexBuffer: MTLBuffer
  private let meshIndexCount: Int
  private let particleStateBuffer: MTLBuffer
  private let maxViewCount: Int

  private static let pointsPerSeed = 9000
  private static let rawStepsPerSeed = 24000
  private static let maxSkipSteps = 5
  private static let warmupSteps = 700
  private static let seedVectors: [SIMD3<Float>] = [
    SIMD3<Float>(0.12, -0.09, 0.04),
    SIMD3<Float>(-0.18, 0.06, 0.11),
    SIMD3<Float>(0.07, 0.15, -0.13),
    SIMD3<Float>(-0.11, -0.17, 0.08),
    SIMD3<Float>(0.19, 0.03, -0.16),
    SIMD3<Float>(-0.04, 0.21, 0.14),
  ]
  private static let totalPoints = pointsPerSeed * seedVectors.count
  private static let targetMeanRadius: Float = 0.72
  private static let acceptedMinDistance: Float = 0.010

  private let orbitScale: Float = 0.72
  private let cubeScale: Float = 3.68
  private let objectCenter = SIMD3<Float>(0.0, 1.1, -7.0)

  private var animationTime: Float = 0
  private var lastSimulationTime: Float?
  private var currentPreset: SimoneOrbit3DPreset = .preset01
  private var needsRebuildParticleStates = true
  private var loggedFirstEncode = false

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    pipelineState = try SimoneOrbit3DRenderer.makePipelineState(
      device: device,
      library: library,
      maxViewCount: self.maxViewCount)
    depthStencilState = SimoneOrbit3DRenderer.makeDepthStencilState(device: device)

    let geometry = MeshGeometryFactory.makeOctahedron(device: device, size: 1.0)
    meshVertexBuffer = geometry.vertexBuffer
    meshIndexBuffer = geometry.indexBuffer
    meshIndexCount = geometry.indexCount

    particleStateBuffer = device.makeBuffer(
      length: MemoryLayout<AizawaParticleState>.stride * Self.totalPoints,
      options: .storageModeShared)!
  }

  func synchronizeState(_ context: PatternSimulationContext) {
    if currentPreset != context.simoneOrbit3DPreset {
      currentPreset = context.simoneOrbit3DPreset
      needsRebuildParticleStates = true
    }
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
    needsRebuildParticleStates = true
    loggedFirstEncode = false
  }

  private func simoneMap(_ p: SIMD3<Float>, _ params: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3<Float>(
      sin(p.x * p.x - p.y * p.y - p.z * p.z + params.x),
      cos(2 * p.x * p.y + params.y),
      sin(2 * p.x * p.z + params.z))
  }

  private func remapForAizawaVisiblePath(_ world: SIMD3<Float>) -> SIMD3<Float> {
    SIMD3<Float>(world.x, world.z + 1.5, world.y + 0.2)
  }

  private func collectAdaptiveLocalPoints(
    seed: SIMD3<Float>,
    params: SIMD3<Float>
  ) -> [SIMD3<Float>] {
    var state = seed
    for _ in 0..<Self.warmupSteps {
      state = simoneMap(state, params)
    }

    var accepted: [SIMD3<Float>] = []
    accepted.reserveCapacity(Self.pointsPerSeed)
    var fallback: [SIMD3<Float>] = []
    fallback.reserveCapacity(Self.rawStepsPerSeed)

    var previousAccepted = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
    var skippedSteps = 0

    for _ in 0..<Self.rawStepsPerSeed {
      state = simoneMap(state, params)
      let local = state * orbitScale
      fallback.append(local)

      let shouldAccept: Bool
      if accepted.isEmpty {
        shouldAccept = true
      } else {
        let distance = simd_length(local - previousAccepted)
        shouldAccept = distance >= Self.acceptedMinDistance || skippedSteps >= Self.maxSkipSteps
      }

      if shouldAccept {
        accepted.append(local)
        previousAccepted = local
        skippedSteps = 0
        if accepted.count == Self.pointsPerSeed {
          break
        }
      } else {
        skippedSteps += 1
      }
    }

    if accepted.count < Self.pointsPerSeed, !fallback.isEmpty {
      let remaining = Self.pointsPerSeed - accepted.count
      for i in 0..<remaining {
        let t = Float(i + 1) / Float(remaining + 1)
        let fallbackIndex = min(Int(t * Float(fallback.count - 1)), fallback.count - 1)
        accepted.append(fallback[fallbackIndex])
      }
    }

    if accepted.count > Self.pointsPerSeed {
      accepted.removeSubrange(Self.pointsPerSeed..<accepted.count)
    }

    return accepted
  }

  private func rebuildParticleStates() {
    let params = currentPreset.parameters
    let ptr = particleStateBuffer.contents().bindMemory(
      to: AizawaParticleState.self,
      capacity: Self.totalPoints)

    var localPoints = Array(repeating: SIMD3<Float>.zero, count: Self.totalPoints)
    var localSteps = Array(repeating: Float(0), count: Self.totalPoints)

    for seedIndex in Self.seedVectors.indices {
      let acceptedPoints = collectAdaptiveLocalPoints(
        seed: Self.seedVectors[seedIndex], params: params)
      guard !acceptedPoints.isEmpty else { continue }

      let base = seedIndex * Self.pointsPerSeed
      var writeIndex = 0
      var previousLocal = acceptedPoints[0]
      for pointIndex in 0..<acceptedPoints.count where writeIndex < Self.pointsPerSeed {
        let local = acceptedPoints[pointIndex]
        localPoints[base + writeIndex] = local
        localSteps[base + writeIndex] = pointIndex == 0 ? 0 : simd_length(local - previousLocal)
        previousLocal = local
        writeIndex += 1
      }

      while writeIndex < Self.pointsPerSeed {
        localPoints[base + writeIndex] = previousLocal
        localSteps[base + writeIndex] = 0
        writeIndex += 1
      }
    }

    var centroid = SIMD3<Float>.zero
    for point in localPoints {
      centroid += point
    }
    centroid /= Float(Self.totalPoints)

    var meanRadius: Float = 0
    for point in localPoints {
      meanRadius += simd_length(point - centroid)
    }
    meanRadius /= Float(Self.totalPoints)
    let radiusScale = Self.targetMeanRadius / max(meanRadius, 0.0001)

    var meanStep: Float = 0
    for step in localSteps {
      meanStep += step
    }
    meanStep /= Float(Self.totalPoints)
    let safeMeanStep = max(meanStep, 0.0001)

    let inv = 1.0 / Float(Self.pointsPerSeed)
    for index in 0..<Self.totalPoints {
      let point = (localPoints[index] - centroid) * radiusScale
      let world = point * cubeScale + objectCenter
      let seedIndex = index / Self.pointsPerSeed
      let normalizedProgress = Float(index % Self.pointsPerSeed) * inv
      let brightness = 0.30 + 0.70 * normalizedProgress
      let phase = animationTime * (0.7 + 0.11 * Float(seedIndex)) + Float(index) * 0.009
      let densityRatio = simd_clamp(localSteps[index] / safeMeanStep, 0.35, 2.4)
      let scale = (0.0028 + 0.0036 * brightness) * (0.80 + 0.55 * densityRatio)
      let visiblePosition = remapForAizawaVisiblePath(world)

      ptr[index] = AizawaParticleState(
        positionAndScale: SIMD4<Float>(
          visiblePosition.x,
          visiblePosition.y,
          visiblePosition.z,
          scale),
        seedAndPhase: SIMD4<Float>(
          Float(seedIndex),
          normalizedProgress,
          brightness,
          phase))
    }
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    if needsRebuildParticleStates {
      rebuildParticleStates()
      needsRebuildParticleStates = false
    }

    if !loggedFirstEncode {
      loggedFirstEncode = true
      print("[Simone] safe point path active particles=\(Self.totalPoints) cubeScale=\(cubeScale)")
    }

    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)

    encoder.setVertexBuffer(meshVertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(particleStateBuffer, offset: 0, index: 1)

    context.applyViewConfiguration(on: encoder)

    var sceneUniforms = SceneUniforms(
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount))
    var viewMatrices = context.viewData.viewProjectionMatrices
    if viewMatrices.isEmpty { viewMatrices = [matrix_identity_float4x4] }

    encoder.setVertexBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 2)
    viewMatrices.withUnsafeBytes {
      if let baseAddress = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(baseAddress, length: $0.count, index: 3)
      }
    }
    encoder.setFragmentBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 0)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: meshIndexCount,
      indexType: .uint16,
      indexBuffer: meshIndexBuffer,
      indexBufferOffset: 0,
      instanceCount: Self.totalPoints)
  }
}

extension SimoneOrbit3DRenderer {
  fileprivate static func makePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "simoneOrbitPointVertex")
    desc.fragmentFunction = library.makeFunction(name: "simoneOrbitPointFragment")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.colorAttachments[0].isBlendingEnabled = false
    desc.depthAttachmentPixelFormat = .depth32Float
    desc.inputPrimitiveTopology = .triangle

    let vertexDescriptor = MTLVertexDescriptor()
    vertexDescriptor.attributes[0].format = .float3
    vertexDescriptor.attributes[0].offset = 0
    vertexDescriptor.attributes[0].bufferIndex = 0
    vertexDescriptor.attributes[1].format = .float3
    vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vertexDescriptor.attributes[1].bufferIndex = 0
    vertexDescriptor.layouts[0].stride = MemoryLayout<MeshVertex>.stride
    desc.vertexDescriptor = vertexDescriptor
    desc.maxVertexAmplificationCount = max(maxViewCount, 1)

    return try device.makeRenderPipelineState(descriptor: desc)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let desc = MTLDepthStencilDescriptor()
    desc.depthCompareFunction = .greater
    desc.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: desc)!
  }
}
