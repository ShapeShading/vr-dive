import Metal
import simd

// ApollonianElevatorRenderer.swift
//
// Cube-container adaptation of ShaderToy "Xtlyzl" by coyote.
// The visible container is a 2 m × 2 m × 2 m cube. Rays march from the
// entry point on the cube surface, or from the eye when the camera is inside.

final class ApollonianElevatorRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .apollonianElevator
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int
  private let maxViewCount: Int

  private let boxScale: Float = 1.0
  private let objectCenter = SIMD3<Float>(0.0, 0.0, -1.5)

  private var animationTime: Float = 0
  private var lastSimulationTime: Float?

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    let geo = ApollonianElevatorRenderer.makeBox(
      device: device,
      localHalfExtents: SIMD3<Float>(repeating: 1.0) * 1.02)
    vertexBuffer = geo.vertexBuffer
    indexBuffer = geo.indexBuffer
    indexCount = geo.indexCount

    pipelineState = try ApollonianElevatorRenderer.makePipelineState(
      device: device,
      library: library,
      maxViewCount: self.maxViewCount)
    depthStencilState = ApollonianElevatorRenderer.makeDepthStencilState(device: device)
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    defer { lastSimulationTime = context.time }
    guard let lastSimulationTime else { return }
    let deltaTime = max(0, min(context.time - lastSimulationTime, 1.0 / 20.0))
    animationTime += deltaTime * max(context.speedMultiplier, 0)
  }

  func resetToInitialState() {
    animationTime = 0
    lastSimulationTime = nil
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    var uniforms = ApollonianElevatorUniforms(
      time: animationTime,
      viewCount: UInt32(context.viewData.viewCount),
      boxScale: boxScale,
      padding: 0,
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0))

    encoder.setVertexBytes(
      &uniforms,
      length: MemoryLayout<ApollonianElevatorUniforms>.stride,
      index: 1)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 2)
      }
    }

    encoder.setFragmentBytes(
      &uniforms,
      length: MemoryLayout<ApollonianElevatorUniforms>.stride,
      index: 0)

    var viewToWorld = context.viewData.viewToWorldTransforms
    if viewToWorld.isEmpty { viewToWorld = [matrix_identity_float4x4] }
    viewToWorld.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setFragmentBytes(base, length: $0.count, index: 1)
      }
    }

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexCount,
      indexType: .uint16,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0)
  }
}

extension ApollonianElevatorRenderer {
  fileprivate static func makeBox(
    device: MTLDevice,
    localHalfExtents e: SIMD3<Float>
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    typealias V = MeshVertex
    let (x, y, z) = (e.x, e.y, e.z)
    let faces: [(positions: [SIMD3<Float>], normal: SIMD3<Float>)] = [
      ([[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], [0, 0, 1]),
      ([[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]], [0, 0, -1]),
      ([[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]], [1, 0, 0]),
      ([[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]], [-1, 0, 0]),
      ([[-x, y, z], [x, y, z], [x, y, -z], [-x, y, -z]], [0, 1, 0]),
      ([[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]], [0, -1, 0]),
    ]

    var vertices: [V] = []
    vertices.reserveCapacity(24)
    var indices: [UInt16] = []
    indices.reserveCapacity(36)

    for face in faces {
      let base = UInt16(vertices.count)
      for p in face.positions {
        vertices.append(V(position: p, normal: face.normal))
      }
      indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }

    let vertexBuffer = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<V>.stride * vertices.count,
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
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "apollonianElevatorVertex")
    desc.fragmentFunction = library.makeFunction(name: "apollonianElevatorFragment")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.depthAttachmentPixelFormat = .depth32Float

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
