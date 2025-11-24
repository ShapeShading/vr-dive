import Metal
import simd

final class PongWarRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .pongWar
  let preferredClearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.04, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let cubeVertexBuffer: MTLBuffer
  private let cubeIndexBuffer: MTLBuffer
  private let sphereVertexBuffer: MTLBuffer
  private let sphereIndexBuffer: MTLBuffer
  private let cubeInstanceBuffer: MTLBuffer
  private let sphereInstanceBuffer: MTLBuffer
  private let cubeIndexCount: Int
  private let sphereIndexCount: Int
  private let cubeInstanceCount: Int
  private let sphereInstanceCount: Int
  private var cubeStates: [PongWarInstanceState]
  private var sphereStates: [PongWarInstanceState]
  private let initialCubeStates: [PongWarInstanceState]
  private let initialSphereStates: [PongWarInstanceState]
  private let device: MTLDevice
  private let maxViewCount: Int
  private var lastSimulationTimestamp: Float = 0
  private let gridDimension: Int = 8
  private let worldCubeSize: Float = 4.0
  private let gridCenter = SIMD3<Float>(0, 0.35, -4.5)
  private let sphereOrbitRadius: Float = 3.2
  private let sphereRadius: Float = 0.32
  private var effectUniforms = PongWarUniforms(
    pulseAmplitude: 0.08,
    pulseSpeed: 1.8,
    cubeRotationSpeed: 0.55,
    sphereBobSpeed: 0.9,
    sphereBobAmount: 0.35,
    sphereGlow: 0.42,
    noiseAmount: 0.25
  )

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.device = device
    self.maxViewCount = max(1, maxViewCount)
    pipelineState = try PongWarRenderer.makePipelineState(
      device: device,
      library: library,
      maxViewCount: self.maxViewCount
    )
    depthStencilState = PongWarRenderer.makeDepthStencilState(device: device)

    let cubeGeometry = PongWarRenderer.makeCubeGeometry(device: device)
    cubeVertexBuffer = cubeGeometry.vertexBuffer
    cubeIndexBuffer = cubeGeometry.indexBuffer
    cubeIndexCount = cubeGeometry.indexCount

    let sphereGeometry = PongWarRenderer.makeSphereGeometry(device: device)
    sphereVertexBuffer = sphereGeometry.vertexBuffer
    sphereIndexBuffer = sphereGeometry.indexBuffer
    sphereIndexCount = sphereGeometry.indexCount

    let cubeStates = PongWarRenderer.makeCubeInstances(
      gridDimension: gridDimension,
      worldSize: worldCubeSize,
      center: gridCenter
    )
    let sphereStates = PongWarRenderer.makeSphereInstances(
      count: 8,
      orbitRadius: sphereOrbitRadius,
      center: gridCenter,
      radius: sphereRadius
    )
    self.cubeStates = cubeStates
    self.sphereStates = sphereStates
    initialCubeStates = cubeStates
    initialSphereStates = sphereStates

    cubeInstanceCount = cubeStates.count
    sphereInstanceCount = sphereStates.count

    let cubeLength = MemoryLayout<PongWarInstanceState>.stride * cubeInstanceCount
    cubeInstanceBuffer = device.makeBuffer(
      bytes: cubeStates,
      length: cubeLength,
      options: [.storageModeShared]
    )!

    let sphereLength = MemoryLayout<PongWarInstanceState>.stride * sphereInstanceCount
    sphereInstanceBuffer = device.makeBuffer(
      bytes: sphereStates,
      length: sphereLength,
      options: [.storageModeShared]
    )!
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    let elapsed = max(0, context.time - lastSimulationTimestamp)
    guard elapsed >= 0 else { return }

    updateSphereStates(time: context.time)
    lastSimulationTimestamp = context.time
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.back)
    encoder.setFrontFacing(.counterClockwise)
    encoder.setTriangleFillMode(.fill)

    context.applyViewConfiguration(on: encoder)

    var sceneUniforms = SceneUniforms(
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount)
    )
    var viewMatrices = context.viewData.viewProjectionMatrices
    if viewMatrices.isEmpty {
      viewMatrices = [matrix_identity_float4x4]
    }

    var uniforms = effectUniforms

    encoder.setVertexBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 2)
    viewMatrices.withUnsafeBytes {
      if let baseAddress = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(baseAddress, length: $0.count, index: 3)
      }
    }
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<PongWarUniforms>.stride, index: 4)
    encoder.setFragmentBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 0)
    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<PongWarUniforms>.stride, index: 1)

    encoder.setVertexBuffer(cubeVertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(cubeInstanceBuffer, offset: 0, index: 1)
    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: cubeIndexCount,
      indexType: .uint16,
      indexBuffer: cubeIndexBuffer,
      indexBufferOffset: 0,
      instanceCount: cubeInstanceCount
    )

    encoder.setVertexBuffer(sphereVertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(sphereInstanceBuffer, offset: 0, index: 1)
    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: sphereIndexCount,
      indexType: .uint16,
      indexBuffer: sphereIndexBuffer,
      indexBufferOffset: 0,
      instanceCount: sphereInstanceCount
    )
  }

  func resetToInitialState() {
    cubeStates = initialCubeStates
    sphereStates = initialSphereStates

    cubeStates.withUnsafeBytes { buffer in
      guard let baseAddress = buffer.baseAddress, buffer.count > 0 else { return }
      memcpy(cubeInstanceBuffer.contents(), baseAddress, buffer.count)
    }

    sphereStates.withUnsafeBytes { buffer in
      guard let baseAddress = buffer.baseAddress, buffer.count > 0 else { return }
      memcpy(sphereInstanceBuffer.contents(), baseAddress, buffer.count)
    }

    lastSimulationTimestamp = 0
  }

  private func updateSphereStates(time: Float) {
    guard !sphereStates.isEmpty else { return }
    let orbitSpeed: Float = 0.4
    let bobSpeed: Float = 0.9
    let radialPulse: Float = 0.2

    for index in 0..<sphereStates.count {
      var state = sphereStates[index]
      let baseAngle = state.motion.x
      let bobPhase = state.motion.y
      let angle = baseAngle + time * orbitSpeed
      let vertical = sin(time * bobSpeed + bobPhase) * 0.6
      let radius = sphereOrbitRadius + cos(time * 0.3 + bobPhase) * radialPulse
      let position = SIMD3<Float>(
        gridCenter.x + cos(angle) * radius,
        gridCenter.y + vertical,
        gridCenter.z + sin(angle) * radius
      )
      state.positionAndScale = SIMD4<Float>(position, sphereRadius)
      sphereStates[index] = state
    }

    sphereStates.withUnsafeBytes { buffer in
      guard let baseAddress = buffer.baseAddress, buffer.count > 0 else { return }
      memcpy(sphereInstanceBuffer.contents(), baseAddress, buffer.count)
    }
  }
}

private typealias MeshBuffers = (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int)

private extension PongWarRenderer {
  static func makePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "pongWarVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "pongWarFragmentShader")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
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

  static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: descriptor)!
  }

  static func makeCubeGeometry(device: MTLDevice) -> MeshBuffers {
    let vertices: [MeshVertex] = [
      // Front
      MeshVertex(position: [-0.5, -0.5, 0.5], normal: [0, 0, 1]),
      MeshVertex(position: [0.5, -0.5, 0.5], normal: [0, 0, 1]),
      MeshVertex(position: [0.5, 0.5, 0.5], normal: [0, 0, 1]),
      MeshVertex(position: [-0.5, 0.5, 0.5], normal: [0, 0, 1]),
      // Back
      MeshVertex(position: [-0.5, -0.5, -0.5], normal: [0, 0, -1]),
      MeshVertex(position: [0.5, -0.5, -0.5], normal: [0, 0, -1]),
      MeshVertex(position: [0.5, 0.5, -0.5], normal: [0, 0, -1]),
      MeshVertex(position: [-0.5, 0.5, -0.5], normal: [0, 0, -1]),
      // Left
      MeshVertex(position: [-0.5, -0.5, -0.5], normal: [-1, 0, 0]),
      MeshVertex(position: [-0.5, -0.5, 0.5], normal: [-1, 0, 0]),
      MeshVertex(position: [-0.5, 0.5, 0.5], normal: [-1, 0, 0]),
      MeshVertex(position: [-0.5, 0.5, -0.5], normal: [-1, 0, 0]),
      // Right
      MeshVertex(position: [0.5, -0.5, -0.5], normal: [1, 0, 0]),
      MeshVertex(position: [0.5, -0.5, 0.5], normal: [1, 0, 0]),
      MeshVertex(position: [0.5, 0.5, 0.5], normal: [1, 0, 0]),
      MeshVertex(position: [0.5, 0.5, -0.5], normal: [1, 0, 0]),
      // Top
      MeshVertex(position: [-0.5, 0.5, 0.5], normal: [0, 1, 0]),
      MeshVertex(position: [0.5, 0.5, 0.5], normal: [0, 1, 0]),
      MeshVertex(position: [0.5, 0.5, -0.5], normal: [0, 1, 0]),
      MeshVertex(position: [-0.5, 0.5, -0.5], normal: [0, 1, 0]),
      // Bottom
      MeshVertex(position: [-0.5, -0.5, 0.5], normal: [0, -1, 0]),
      MeshVertex(position: [0.5, -0.5, 0.5], normal: [0, -1, 0]),
      MeshVertex(position: [0.5, -0.5, -0.5], normal: [0, -1, 0]),
      MeshVertex(position: [-0.5, -0.5, -0.5], normal: [0, -1, 0]),
    ]

    let indices: [UInt16] = [
      0, 1, 2, 0, 2, 3,
      4, 5, 6, 4, 6, 7,
      8, 9, 10, 8, 10, 11,
      12, 13, 14, 12, 14, 15,
      16, 17, 18, 16, 18, 19,
      20, 21, 22, 20, 22, 23,
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

  static func makeSphereGeometry(device: MTLDevice) -> MeshBuffers {
    let stacks = 18
    let slices = 24
    var vertices: [MeshVertex] = []
    var indices: [UInt16] = []

    for stack in 0...stacks {
      let v = Float(stack) / Float(stacks)
      let phi = v * Float.pi
      let y = cos(phi)
      let sinPhi = sin(phi)
      for slice in 0...slices {
        let u = Float(slice) / Float(slices)
        let theta = u * 2 * Float.pi
        let x = sinPhi * cos(theta)
        let z = sinPhi * sin(theta)
        let normal = simd_normalize(SIMD3<Float>(x, y, z))
        vertices.append(MeshVertex(position: normal * 0.5, normal: normal))
      }
    }

    let rowLength = slices + 1
    for stack in 0..<stacks {
      for slice in 0..<slices {
        let topLeft = stack * rowLength + slice
        let bottomLeft = (stack + 1) * rowLength + slice
        let topRight = topLeft + 1
        let bottomRight = bottomLeft + 1
        indices.append(contentsOf: [
          UInt16(topLeft), UInt16(bottomLeft), UInt16(topRight),
          UInt16(topRight), UInt16(bottomLeft), UInt16(bottomRight),
        ])
      }
    }

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

  static func makeCubeInstances(
    gridDimension: Int,
    worldSize: Float,
    center: SIMD3<Float>
  ) -> [PongWarInstanceState] {
    guard gridDimension > 0 else { return [] }
    let total = gridDimension * gridDimension * gridDimension
    var instances: [PongWarInstanceState] = []
    instances.reserveCapacity(total)

    let cellSize = worldSize / Float(gridDimension)
    let start = -worldSize * 0.5 + cellSize * 0.5

    for x in 0..<gridDimension {
      for y in 0..<gridDimension {
        for z in 0..<gridDimension {
          let position = SIMD3<Float>(
            center.x + start + Float(x) * cellSize,
            center.y + start + Float(y) * cellSize,
            center.z + start + Float(z) * cellSize
          )
          let color = SIMD3<Float>(
            Float.random(in: 0.2...0.95),
            Float.random(in: 0.2...0.95),
            Float.random(in: 0.2...0.95)
          )
          let normalizedColor = simd_normalize(color + SIMD3<Float>(0.08, 0.12, 0.16))
          let phase = Float.random(in: 0...(.pi * 2))
          let offset = Float.random(in: -0.5...0.5)
          let wobble = Float.random(in: 0.1...0.5)

          instances.append(
            PongWarInstanceState(
              positionAndScale: SIMD4<Float>(position, cellSize),
              color: SIMD4<Float>(normalizedColor, 1),
              motion: SIMD4<Float>(phase, offset, 0, wobble)
            )
          )
        }
      }
    }

    return instances
  }

  static func makeSphereInstances(
    count: Int,
    orbitRadius: Float,
    center: SIMD3<Float>,
    radius: Float
  ) -> [PongWarInstanceState] {
    guard count > 0 else { return [] }
    var instances: [PongWarInstanceState] = []
    instances.reserveCapacity(count)

    for index in 0..<count {
      let fraction = Float(index) / Float(count)
      let angle = fraction * 2 * Float.pi
      let height = (index % 2 == 0 ? 0.8 : -0.8) * radius * 2.4
      let position = SIMD3<Float>(
        center.x + cos(angle) * orbitRadius,
        center.y + height,
        center.z + sin(angle) * orbitRadius
      )
      let color = SIMD3<Float>(
        0.55 + 0.45 * sin(angle + 0.3),
        0.35 + 0.55 * sin(angle * 1.7 + 1.2),
        0.55 + 0.35 * cos(angle * 0.8 + 0.4)
      )
      instances.append(
        PongWarInstanceState(
          positionAndScale: SIMD4<Float>(position, radius),
          color: SIMD4<Float>(color, 1),
          motion: SIMD4<Float>(angle, Float(index) * 0.37, 1, 0)
        )
      )
    }

    return instances
  }
}
