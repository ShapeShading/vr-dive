import Metal
import simd

final class FourWingRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .fourWingAttractor
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let objectPipelineState: MTLRenderPipelineState
  private let computePipelineState: MTLComputePipelineState
  private let depthStencilState: MTLDepthStencilState
  private let meshVertexBuffer: MTLBuffer
  private let meshIndexBuffer: MTLBuffer
  private let meshIndexCount: Int
  private let particleStateBuffer: MTLBuffer
  private var lastSimulationTimestamp: Float = 0
  private let particleCount: Int
  private let device: MTLDevice
  private let a: Float = 0.2
  private let b: Float = 0.01
  private let c: Float = -0.4
  private let worldScale: Float = 4.8  // Scale for Four-Wing attractor (4x larger for better visibility)
  private let resetRadius: Float = 15.0  // Adjusted for Four-Wing's smaller scale
  private let noiseAmplitude: Float = 0.05  // Reduced noise for stability
  private let minDeltaTime: Float = 1.0 / 600.0
  private let maxDeltaTime: Float = 1.0 / 45.0
  private let damping: Float = 1.0
  private let maxViewCount: Int

  init(device: MTLDevice, library: MTLLibrary, particleCount: Int, maxViewCount: Int) throws {
    self.particleCount = particleCount
    self.device = device
    self.maxViewCount = max(1, maxViewCount)
    objectPipelineState = try FourWingRenderer.makeObjectPipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    computePipelineState = try FourWingRenderer.makeComputePipelineState(
      device: device, library: library)
    depthStencilState = FourWingRenderer.makeDepthStencilState(device: device)

    let geometry = MeshGeometryFactory.makeOctahedron(device: device)
    meshVertexBuffer = geometry.vertexBuffer
    meshIndexBuffer = geometry.indexBuffer
    meshIndexCount = geometry.indexCount

    particleStateBuffer = FourWingRenderer.makeInitialParticleStates(
      device: device,
      count: particleCount,
      worldScale: worldScale
    )
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    let elapsed = max(0, context.time - lastSimulationTimestamp)
    let clampedDeltaTime = min(max(elapsed, minDeltaTime), maxDeltaTime) * 0.2
    guard clampedDeltaTime > 0 else { return }

    var uniforms = FourWingUniforms(
      deltaTime: clampedDeltaTime,
      globalTime: context.time,
      particleCount: UInt32(particleCount),
      a: a,
      b: b,
      c: c,
      damping: damping,
      worldScale: worldScale,
      resetRadius: resetRadius,
      noiseAmplitude: noiseAmplitude,
      padding: 0
    )

    guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else { return }

    encoder.setComputePipelineState(computePipelineState)
    encoder.setBuffer(particleStateBuffer, offset: 0, index: 0)
    encoder.setBytes(&uniforms, length: MemoryLayout<FourWingUniforms>.stride, index: 1)

    let threadsPerGroup = min(computePipelineState.maxTotalThreadsPerThreadgroup, 64)
    let threadgroups = MTLSize(
      width: (particleCount + threadsPerGroup - 1) / threadsPerGroup,
      height: 1,
      depth: 1
    )

    encoder.dispatchThreadgroups(
      threadgroups, threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    lastSimulationTimestamp = context.time
  }

  func resetToInitialState() {
    let newBuffer = FourWingRenderer.makeInitialParticleStates(
      device: device,
      count: particleCount,
      worldScale: worldScale
    )
    memcpy(
      particleStateBuffer.contents(),
      newBuffer.contents(),
      MemoryLayout<FourWingParticleState>.stride * particleCount
    )
    lastSimulationTimestamp = 0
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encodeObjects(with: encoder, context: context)
  }

  private func encodeObjects(with encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(objectPipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)

    encoder.setVertexBuffer(meshVertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(particleStateBuffer, offset: 0, index: 1)

    context.applyViewConfiguration(on: encoder)

    var sceneUniforms = SceneUniforms(
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount)
    )
    var viewMatrices = context.viewData.viewProjectionMatrices
    if viewMatrices.isEmpty {
      viewMatrices = [matrix_identity_float4x4]
    }

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
      instanceCount: particleCount
    )
  }
}

extension FourWingRenderer {
  fileprivate static func makeObjectPipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws
    -> MTLRenderPipelineState
  {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "fourWingVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "fourWingFragmentShader")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.colorAttachments[0].isBlendingEnabled = false  // Disable blending for opaque rendering
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .triangle

    let vertexDescriptor = MTLVertexDescriptor()
    vertexDescriptor.attributes[0].format = .float3
    vertexDescriptor.attributes[0].offset = 0
    vertexDescriptor.attributes[0].bufferIndex = 0
    vertexDescriptor.attributes[1].format = .float3
    vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vertexDescriptor.attributes[1].bufferIndex = 0
    vertexDescriptor.layouts[0].stride = MemoryLayout<MeshVertex>.stride
    descriptor.vertexDescriptor = vertexDescriptor
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)

    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  fileprivate static func makeComputePipelineState(device: MTLDevice, library: MTLLibrary) throws
    -> MTLComputePipelineState
  {
    let function = library.makeFunction(name: "integrateFourWingAttractor")!
    return try device.makeComputePipelineState(function: function)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = true  // Enable depth write for correct occlusion
    return device.makeDepthStencilState(descriptor: descriptor)!
  }

  fileprivate static func makeInitialParticleStates(
    device: MTLDevice, count: Int, worldScale: Float
  ) -> MTLBuffer {
    // Four-Wing parameters (same as shader)
    let a: Float = 0.2
    let b: Float = 0.01
    let c: Float = -0.4
    let dt: Float = 0.01  // Step for trajectory

    // Create 16 trajectory groups
    let numGroups = 16
    let particlesPerGroup = count / numGroups

    var states: [FourWingParticleState] = []
    states.reserveCapacity(count)

    for groupIndex in 0..<numGroups {
      // Random starting point in attractor space (not scaled)
      var x: Float = Float.random(in: -0.5...0.5)
      var y: Float = Float.random(in: -0.5...0.5)
      var z: Float = Float.random(in: -0.5...0.5)

      let groupSize =
        (groupIndex == numGroups - 1) ? (count - groupIndex * particlesPerGroup) : particlesPerGroup

      for _ in 0..<groupSize {
        // Current position in attractor space
        let attractorPosition = SIMD3<Float>(x, y, z)
        // Scale to world space
        let worldPosition = attractorPosition * worldScale

        let isSpecial = Float.random(in: 0...1) < 0.03
        let scale: Float
        if isSpecial {
          scale = Float.random(in: 0.18...0.27)
        } else {
          scale = Float.random(in: 0.09...0.15)
        }

        let rotation = atan2(attractorPosition.z, attractorPosition.x)
        let seed = SIMD3<Float>(
          Float.random(in: 0...10_000),
          Float.random(in: 0...10_000),
          Float.random(in: 0...10_000)
        )
        states.append(
          FourWingParticleState(
            positionAndScale: SIMD4<Float>(
              worldPosition.x, worldPosition.y, worldPosition.z, scale),
            seedAndPhase: SIMD4<Float>(seed.x, seed.y, seed.z, rotation)
          )
        )

        // Integrate Four-Wing equations in attractor space
        let dx = a * x + y * z
        let dy = b * x + c * y - x * z
        let dz = -z - x * y

        x += dx * dt
        y += dy * dt
        z += dz * dt
      }
    }

    return device.makeBuffer(
      bytes: states,
      length: MemoryLayout<FourWingParticleState>.stride * states.count,
      options: [.storageModeShared]
    )!
  }
}
