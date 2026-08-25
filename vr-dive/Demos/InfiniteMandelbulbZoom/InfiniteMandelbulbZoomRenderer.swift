import Metal
import simd

/// A world-space Dynamic Box whose interior contains an endlessly rebased
/// Mandelbulb hierarchy. The box remains fixed in the scene while the shared
/// pattern-navigation transform lets the controller move through its contents.
final class InfiniteMandelbulbZoomRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .infiniteMandelbulbZoom
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int

  // Match Dynamic Box so this pattern has the same placement, scale and
  // controller-navigation behavior as the other box-contained demos.
  private let boxScale: Float = 0.84
  private let objectCenter = SIMD3<Float>(0, -0.05, -1.1)

  private var zoomPhase: Float = 0
  private var generation: UInt32 = 0
  private var lastSimulationTime: Float?
  private var latestContext: PatternSimulationContext?
  private var didLogConfiguration = false

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    let geometry = Self.makeBox(
      device: device,
      localHalfExtents: SIMD3<Float>(0.95, 0.95, 1.25) * 1.02)
    vertexBuffer = geometry.vertexBuffer
    indexBuffer = geometry.indexBuffer
    indexCount = geometry.indexCount

    pipelineState = try Self.makePipelineState(
      device: device,
      library: library,
      maxViewCount: max(1, maxViewCount))
    depthStencilState = Self.makeDepthStencilState(device: device)
  }

  func synchronizeState(_ context: PatternSimulationContext) {
    latestContext = context
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    defer { lastSimulationTime = context.time }
    guard let lastSimulationTime else { return }

    let deltaTime = max(0, min(context.time - lastSimulationTime, 1.0 / 20.0))
    zoomPhase += deltaTime * context.infiniteZoomRate
      * max(context.speedMultiplier, 0) * context.infiniteZoomDirection

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
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    context.applyViewConfiguration(on: encoder)

    var uniforms = makeUniforms(context: context)
    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
    encoder.setVertexBytes(
      &uniforms,
      length: MemoryLayout<InfiniteMandelbulbZoomUniforms>.stride,
      index: 1)

    var viewProjectionMatrices = context.viewData.viewProjectionMatrices
    if viewProjectionMatrices.isEmpty {
      viewProjectionMatrices = [matrix_identity_float4x4]
    }
    viewProjectionMatrices.withUnsafeBytes { bytes in
      if let baseAddress = bytes.baseAddress, bytes.count > 0 {
        encoder.setVertexBytes(baseAddress, length: bytes.count, index: 2)
      }
    }

    encoder.setFragmentBytes(
      &uniforms,
      length: MemoryLayout<InfiniteMandelbulbZoomUniforms>.stride,
      index: 0)

    var viewToWorldTransforms = context.viewData.viewToWorldTransforms
    if viewToWorldTransforms.isEmpty {
      viewToWorldTransforms = [matrix_identity_float4x4]
    }
    viewToWorldTransforms.withUnsafeBytes { bytes in
      if let baseAddress = bytes.baseAddress, bytes.count > 0 {
        encoder.setFragmentBytes(baseAddress, length: bytes.count, index: 1)
      }
    }

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexCount,
      indexType: .uint16,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0)

    if !didLogConfiguration {
      didLogConfiguration = true
      print(
        "[InfiniteMandelbulbZoom] World-space box active: center=\(objectCenter), scale=\(boxScale), views=\(context.viewData.viewCount), controllerNavigation=true"
      )
    }
  }

  private func makeUniforms(context: PatternRenderContext) -> InfiniteMandelbulbZoomUniforms {
    let quality = latestContext?.infiniteZoomQuality ?? .balanced
    return InfiniteMandelbulbZoomUniforms(
      zoomPhase: zoomPhase,
      viewCount: UInt32(max(context.viewData.viewCount, 1)),
      generation: generation,
      maxRaySteps: quality.raySteps,
      fractalIterations: quality.fractalIterations,
      surfaceEpsilon: quality == .detailed ? 0.0010 : 0.0015,
      boxScale: boxScale,
      padding: 0,
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0),
      patternTransform: context.patternNavigationTransform)
  }
}

extension InfiniteMandelbulbZoomRenderer {
  fileprivate static func makeBox(
    device: MTLDevice,
    localHalfExtents e: SIMD3<Float>
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    typealias Vertex = MeshVertex
    let (x, y, z) = (e.x, e.y, e.z)
    let faces: [(positions: [SIMD3<Float>], normal: SIMD3<Float>)] = [
      ([[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], [0, 0, 1]),
      ([[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]], [0, 0, -1]),
      ([[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]], [1, 0, 0]),
      ([[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]], [-1, 0, 0]),
      ([[-x, y, z], [x, y, z], [x, y, -z], [-x, y, -z]], [0, 1, 0]),
      ([[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]], [0, -1, 0]),
    ]

    var vertices: [Vertex] = []
    vertices.reserveCapacity(24)
    var indices: [UInt16] = []
    indices.reserveCapacity(36)
    for face in faces {
      let base = UInt16(vertices.count)
      for position in face.positions {
        vertices.append(Vertex(position: position, normal: face.normal))
      }
      indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }

    let vertexBuffer = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<Vertex>.stride * vertices.count,
      options: .storageModeShared)!
    let indexBuffer = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: .storageModeShared)!
    return (vertexBuffer, indexBuffer, indices.count)
  }

  fileprivate static func makePipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "infiniteMandelbulbZoomVertex")
    descriptor.fragmentFunction = library.makeFunction(name: "infiniteMandelbulbZoomFragment")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float

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

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: descriptor)!
  }
}
