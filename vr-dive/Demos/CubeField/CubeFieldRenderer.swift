import Metal
import simd

private enum CubeFieldShapeVariant: Int, CaseIterable {
  case block = 0
  case column = 1
  case crystal = 2
}

private struct CubeFieldShapeRange {
  let variant: CubeFieldShapeVariant
  let start: Int
  let count: Int
}

private typealias MeshBuffers = (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int)

final class CubeFieldRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .cubeField
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let objectPipelineState: MTLRenderPipelineState
  private let computePipelineState: MTLComputePipelineState
  private let depthStencilState: MTLDepthStencilState
  private let objectStateBuffer: MTLBuffer
  private var lastSimulationTimestamp: Float = 0
  private let objectCount: Int
  private let device: MTLDevice
  private let maxViewCount: Int
  private let shapeMeshes: [CubeFieldShapeVariant: MeshBuffers]
  private var shapeRanges: [CubeFieldShapeRange]

  init(device: MTLDevice, library: MTLLibrary, objectCount: Int, maxViewCount: Int) throws {
    self.objectCount = objectCount
    self.device = device
    self.maxViewCount = max(1, maxViewCount)
    objectPipelineState = try CubeFieldRenderer.makeObjectPipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    computePipelineState = try CubeFieldRenderer.makeComputePipelineState(
      device: device, library: library)
    depthStencilState = CubeFieldRenderer.makeDepthStencilState(device: device)

    shapeMeshes = [
      .block: CubeFieldRenderer.makeCubeGeometry(device: device),
      .column: CubeFieldRenderer.makeColumnGeometry(device: device),
      .crystal: CubeFieldRenderer.makeCrystalGeometry(device: device)
    ]
    shapeRanges = CubeFieldRenderer.makeShapeRanges(totalCount: objectCount)

    objectStateBuffer = CubeFieldRenderer.makeInitialObjectStates(
      device: device,
      ranges: shapeRanges
    )
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    let deltaTime = max(0, context.time - lastSimulationTimestamp)
    guard deltaTime > 0 else { return }

    var uniforms = SimulationUniforms(
      deltaTime: min(deltaTime, 1.0 / 30.0),
      globalTime: context.time,
      objectCount: UInt32(objectCount)
    )

    guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else { return }

    encoder.setComputePipelineState(computePipelineState)
    encoder.setBuffer(objectStateBuffer, offset: 0, index: 0)
    encoder.setBytes(&uniforms, length: MemoryLayout<SimulationUniforms>.stride, index: 1)

    let threadWidth = min(computePipelineState.maxTotalThreadsPerThreadgroup, 32)
    let threadsPerThreadgroup = MTLSize(width: threadWidth, height: 1, depth: 1)
    let threadgroups = MTLSize(
      width: (objectCount + threadWidth - 1) / threadWidth,
      height: 1,
      depth: 1
    )

    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    lastSimulationTimestamp = context.time
  }

  func resetToInitialState() {
    let states = CubeFieldRenderer.generateInitialStates(for: shapeRanges)
    states.withUnsafeBytes { buffer in
      guard let baseAddress = buffer.baseAddress, buffer.count > 0 else { return }
      memcpy(
        objectStateBuffer.contents(),
        baseAddress,
        min(buffer.count, MemoryLayout<ObjectState>.stride * objectCount)
      )
    }
    lastSimulationTimestamp = 0
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encodeObjects(with: encoder, context: context)
  }

  private func encodeObjects(with encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    guard !shapeRanges.isEmpty else { return }

    encoder.setRenderPipelineState(objectPipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)  // Disable culling to see back faces
    encoder.setFrontFacing(.counterClockwise)
    encoder.setTriangleFillMode(.fill)

    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(objectStateBuffer, offset: 0, index: 1)

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

    for range in shapeRanges {
      guard range.count > 0,
        let mesh = shapeMeshes[range.variant]
      else { continue }

      encoder.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
      encoder.drawIndexedPrimitives(
        type: .triangle,
        indexCount: mesh.indexCount,
        indexType: .uint16,
        indexBuffer: mesh.indexBuffer,
        indexBufferOffset: 0,
        instanceCount: range.count,
        baseVertex: 0,
        baseInstance: range.start
      )
    }
  }
}

extension CubeFieldRenderer {
  fileprivate static func makeObjectPipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws
    -> MTLRenderPipelineState
  {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "objectVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "objectFragmentShader")
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

  fileprivate static func makeComputePipelineState(device: MTLDevice, library: MTLLibrary) throws
    -> MTLComputePipelineState
  {
    let function = library.makeFunction(name: "simulateObjects")!
    return try device.makeComputePipelineState(function: function)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: descriptor)!
  }

  fileprivate static func makeCubeGeometry(device: MTLDevice) -> MeshBuffers {
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
      // Front face (counter-clockwise from outside, +Z looking at -Z)
      0, 1, 2, 0, 2, 3,
      // Back face (counter-clockwise from outside, -Z looking at +Z) - FIXED
      4, 5, 6, 4, 6, 7,
      // Left face (counter-clockwise from outside, -X looking at +X)
      8, 9, 10, 8, 10, 11,
      // Right face (counter-clockwise from outside, +X looking at -X)
      12, 13, 14, 12, 14, 15,
      // Top face (counter-clockwise from outside, +Y looking at -Y)
      16, 17, 18, 16, 18, 19,
      // Bottom face (counter-clockwise from outside, -Y looking at +Y) - FIXED
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

  fileprivate static func makeInitialObjectStates(
    device: MTLDevice,
    ranges: [CubeFieldShapeRange]
  ) -> MTLBuffer {
    let states = generateInitialStates(for: ranges)
    guard !states.isEmpty else {
      return device.makeBuffer(
        length: MemoryLayout<ObjectState>.stride,
        options: [.storageModeShared]
      )!
    }
    return device.makeBuffer(
      bytes: states,
      length: MemoryLayout<ObjectState>.stride * states.count,
      options: [.storageModeShared]
    )!
  }

  private static func generateInitialStates(
    for ranges: [CubeFieldShapeRange]
  ) -> [ObjectState] {
    var states: [ObjectState] = []
    states.reserveCapacity(ranges.reduce(0) { $0 + $1.count })
    for range in ranges {
      guard range.count > 0 else { continue }
      for _ in 0..<range.count {
        states.append(makeState(for: range.variant))
      }
    }
    return states
  }

  private static func makeState(for variant: CubeFieldShapeVariant) -> ObjectState {
    let scaleBoost: Float = 8.0
    let zRange: ClosedRange<Float> = -3.4...(-0.6)
    let basePosition: SIMD3<Float>
    let motionAmplitude: SIMD3<Float>
    let scale: SIMD3<Float>
    let jitterRadius: Float

    switch variant {
    case .block:
      basePosition = SIMD3<Float>(
        Float.random(in: -1.6...1.6),
        Float.random(in: -0.9...1.1),
        Float.random(in: zRange)
      )
      motionAmplitude = SIMD3<Float>(
        Float.random(in: 0.008...0.04),
        Float.random(in: 0.012...0.05),
        Float.random(in: 0.008...0.04)
      )
      scale = SIMD3<Float>(
        Float.random(in: 0.18...0.4),
        Float.random(in: 0.18...0.42),
        Float.random(in: 0.18...0.35)
      )
      jitterRadius = Float.random(in: 0.05...0.12)

    case .column:
      basePosition = SIMD3<Float>(
        Float.random(in: -1.2...1.2),
        Float.random(in: -0.8...1.3),
        Float.random(in: -3.1...(-0.8))
      )
      motionAmplitude = SIMD3<Float>(
        Float.random(in: 0.006...0.02),
        Float.random(in: 0.02...0.06),
        Float.random(in: 0.006...0.02)
      )
      scale = SIMD3<Float>(
        Float.random(in: 0.12...0.2),
        Float.random(in: 0.65...1.2),
        Float.random(in: 0.12...0.2)
      )
      jitterRadius = Float.random(in: 0.08...0.18)

    case .crystal:
      basePosition = SIMD3<Float>(
        Float.random(in: -1.7...1.7),
        Float.random(in: -0.4...1.5),
        Float.random(in: -3.4...(-0.4))
      )
      let amplitudeRange: ClosedRange<Float> = 0.015...0.05
      let amp = Float.random(in: amplitudeRange)
      motionAmplitude = SIMD3<Float>(amp, amp * 1.2, amp)
      let uniformScale = Float.random(in: 0.18...0.3)
      scale = SIMD3<Float>(repeating: uniformScale)
      jitterRadius = Float.random(in: 0.09...0.2)
    }

    let phase = Float.random(in: 0...(.pi * 2))

    let scaledPosition = basePosition * scaleBoost
    let scaledAmplitude = motionAmplitude * scaleBoost
    let scaledScale = scale * scaleBoost
    let scaledJitter = jitterRadius * scaleBoost
    let typeValue = Float(variant.rawValue)

    return ObjectState(
      positionAndType: SIMD4<Float>(scaledPosition, typeValue),
      motionAndPhase: SIMD4<Float>(scaledAmplitude, phase),
      scaleAndPadding: SIMD4<Float>(scaledScale, 0),
      homeAndJitter: SIMD4<Float>(scaledPosition, scaledJitter)
    )
  }

  private static func makeShapeRanges(totalCount: Int) -> [CubeFieldShapeRange] {
    guard totalCount > 0 else { return [] }

    let weights: [(CubeFieldShapeVariant, Float)] = [
      (.block, 0.5),
      (.column, 0.25),
      (.crystal, 0.25),
    ]

    var remaining = totalCount
    var counts: [CubeFieldShapeVariant: Int] = [:]

    for (index, entry) in weights.enumerated() {
      let variant = entry.0
      if index == weights.count - 1 {
        counts[variant] = remaining
        remaining = 0
      } else {
        let desired = Int(round(Float(totalCount) * entry.1))
        let clamped = max(0, min(desired, remaining - max(0, weights.count - index - 1)))
        counts[variant] = clamped
        remaining -= clamped
      }
    }

    if remaining > 0 {
      counts[weights[0].0, default: 0] += remaining
    }

    var start = 0
    var ranges: [CubeFieldShapeRange] = []
    for (variant, _) in weights {
      let count = counts[variant, default: 0]
      guard count > 0 else { continue }
      ranges.append(CubeFieldShapeRange(variant: variant, start: start, count: count))
      start += count
    }
    return ranges
  }

  fileprivate static func makeColumnGeometry(device: MTLDevice) -> MeshBuffers {
    let sides = 6
    let height: Float = 1.2
    let topRadius: Float = 0.32
    let bottomRadius: Float = 0.38
    var vertices: [MeshVertex] = []
    var indices: [UInt16] = []

    for i in 0..<sides {
      let angle = (Float(i) / Float(sides)) * 2 * Float.pi
      let dir = SIMD3<Float>(cos(angle), 0, sin(angle))
      let topPos = SIMD3<Float>(dir.x * topRadius, height * 0.5, dir.z * topRadius)
      let bottomPos = SIMD3<Float>(dir.x * bottomRadius, -height * 0.5, dir.z * bottomRadius)
      let normal = simd_normalize(SIMD3<Float>(dir.x, 0.15, dir.z))
      vertices.append(MeshVertex(position: topPos, normal: normal))
      vertices.append(MeshVertex(position: bottomPos, normal: normal))
    }

    for i in 0..<sides {
      let next = (i + 1) % sides
      let topCurrent = UInt16(i * 2)
      let bottomCurrent = topCurrent + 1
      let topNext = UInt16(next * 2)
      let bottomNext = topNext + 1
      indices.append(contentsOf: [
        topCurrent, bottomCurrent, topNext,
        topNext, bottomCurrent, bottomNext,
      ])
    }

    let topCenterIndex = UInt16(vertices.count)
    vertices.append(
      MeshVertex(position: SIMD3<Float>(0, height * 0.5, 0), normal: SIMD3<Float>(0, 1, 0))
    )
    let bottomCenterIndex = UInt16(vertices.count)
    vertices.append(
      MeshVertex(position: SIMD3<Float>(0, -height * 0.5, 0), normal: SIMD3<Float>(0, -1, 0))
    )

    for i in 0..<sides {
      let next = (i + 1) % sides
      let topVertex = UInt16(i * 2)
      let nextTop = UInt16(next * 2)
      indices.append(contentsOf: [
        topCenterIndex, nextTop, topVertex,
      ])

      let bottomVertex = UInt16(i * 2 + 1)
      let nextBottom = UInt16(next * 2 + 1)
      indices.append(contentsOf: [
        bottomCenterIndex, bottomVertex, nextBottom,
      ])
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

  fileprivate static func makeCrystalGeometry(device: MTLDevice) -> MeshBuffers {
    MeshGeometryFactory.makeOctahedron(device: device, size: 0.5)
  }
}
