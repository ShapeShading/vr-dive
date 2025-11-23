import Metal
import simd

final class LorenzRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .lorenzAttractor
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
  private let sigma: Float = 10.0
  private let beta: Float = 8.0 / 3.0
  private let rho: Float = 28.0
  private let worldScale: Float = 0.24  // Larger scale (4x previous 0.06)
  private let resetRadius: Float = 60.0  // Increased reset radius
  private let noiseAmplitude: Float = 0.18
  private let minDeltaTime: Float = 1.0 / 600.0
  private let maxDeltaTime: Float = 1.0 / 45.0
  private let damping: Float = 1.0  // No damping to preserve chaotic motion
  private let maxViewCount: Int

  init(device: MTLDevice, library: MTLLibrary, particleCount: Int, maxViewCount: Int) throws {
    self.particleCount = particleCount
    self.device = device
    self.maxViewCount = max(1, maxViewCount)
    objectPipelineState = try LorenzRenderer.makeObjectPipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    computePipelineState = try LorenzRenderer.makeComputePipelineState(
      device: device, library: library)
    depthStencilState = LorenzRenderer.makeDepthStencilState(device: device)

    let geometry = MeshGeometryFactory.makeOctahedron(device: device)
    meshVertexBuffer = geometry.vertexBuffer
    meshIndexBuffer = geometry.indexBuffer
    meshIndexCount = geometry.indexCount

    particleStateBuffer = LorenzRenderer.makeInitialParticleStates(
      device: device,
      count: particleCount,
      worldScale: worldScale
    )
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    let elapsed = max(0, context.time - lastSimulationTimestamp)
    // Speed up simulation for better visual effect
    let clampedDeltaTime = min(max(elapsed, minDeltaTime), maxDeltaTime) * 0.2
    guard clampedDeltaTime > 0 else { return }

    var uniforms = LorenzUniforms(
      deltaTime: clampedDeltaTime,
      globalTime: context.time,
      particleCount: UInt32(particleCount),
      sigma: sigma,
      beta: beta,
      rho: rho,
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
    encoder.setBytes(&uniforms, length: MemoryLayout<LorenzUniforms>.stride, index: 1)

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
    let newBuffer = LorenzRenderer.makeInitialParticleStates(
      device: device,
      count: particleCount,
      worldScale: worldScale
    )
    // Copy new initial states to existing buffer
    memcpy(
      particleStateBuffer.contents(),
      newBuffer.contents(),
      MemoryLayout<LorenzParticleState>.stride * particleCount
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

extension LorenzRenderer {
  fileprivate static func makeObjectPipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws
    -> MTLRenderPipelineState
  {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "lorenzVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "lorenzFragmentShader")
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
    let function = library.makeFunction(name: "integrateLorenzAttractor")!
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
    // Lorenz parameters (same as shader)
    let sigma: Float = 10.0
    let beta: Float = 8.0 / 3.0
    let rho: Float = 28.0
    let dt: Float = 0.001  // Small step for trajectory

    // Create 16 trajectory groups
    let numGroups = 16
    let particlesPerGroup = count / numGroups

    var states: [LorenzParticleState] = []
    states.reserveCapacity(count)

    for groupIndex in 0..<numGroups {
      // Random starting point in attractor space (not scaled)
      var x: Float = Float.random(in: -10...10)
      var y: Float = Float.random(in: -10...10)
      var z: Float = Float.random(in: 20...35)

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
          LorenzParticleState(
            positionAndScale: SIMD4<Float>(
              worldPosition.x, worldPosition.y, worldPosition.z, scale),
            seedAndPhase: SIMD4<Float>(seed.x, seed.y, seed.z, rotation)
          )
        )

        // Integrate Lorenz equations in attractor space
        let dx = sigma * (y - x)
        let dy = x * (rho - z) - y
        let dz = x * y - beta * z

        x += dx * dt
        y += dy * dt
        z += dz * dt
      }
    }

    return device.makeBuffer(
      bytes: states,
      length: MemoryLayout<LorenzParticleState>.stride * states.count,
      options: [.storageModeShared]
    )!
  }
}
