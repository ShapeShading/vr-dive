import Metal
import simd

// WaveySpheresRenderer.swift
//
// Source reference requested by user:
// https://www.shadertoy.com/view/WX3cR4
// This is an original Metal adaptation for vr-dive that reinterprets the
// reference as a 3D raymarched volume inside a 2 meter cube container.

final class WaveySpheresRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .waveySpheres
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int
  private let maxViewCount: Int

  // Mesh vertices span [-1, 1], so a scale of 1.0 yields a 2 meter cube.
  private let cubeScale: Float = 1.0
  private let travelSpeed: Float = 0.7
  private let objectCenter = SIMD3<Float>(0.0, 0.0, -1.75)

  private var animationTime: Float = 0
  private var lastSimulationTime: Float?

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    let geo = WaveySpheresRenderer.makeBox(device: device)
    vertexBuffer = geo.vertexBuffer
    indexBuffer = geo.indexBuffer
    indexCount = geo.indexCount

    pipelineState = try WaveySpheresRenderer.makePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    depthStencilState = WaveySpheresRenderer.makeDepthStencilState(device: device)
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
    encoder.setCullMode(.back)
    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    var uniforms = WaveySpheresUniforms(
      time: animationTime,
      viewCount: UInt32(context.viewData.viewCount),
      cubeScale: cubeScale,
      travelSpeed: travelSpeed,
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0))

    encoder.setVertexBytes(
      &uniforms, length: MemoryLayout<WaveySpheresUniforms>.stride, index: 1)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 2)
      }
    }

    encoder.setFragmentBytes(
      &uniforms, length: MemoryLayout<WaveySpheresUniforms>.stride, index: 0)

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

extension WaveySpheresRenderer {
  fileprivate static func makeBox(
    device: MTLDevice
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    typealias V = MeshVertex
    let x: Float = 1.0
    let y: Float = 1.0
    let z: Float = 1.0
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
      for position in face.positions {
        vertices.append(V(position: position, normal: face.normal))
      }
      indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }

    let vBuf = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<MeshVertex>.stride * vertices.count,
      options: .storageModeShared)!
    let iBuf = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: .storageModeShared)!
    return (vBuf, iBuf, indices.count)
  }

  fileprivate static func makePipelineState(
    device: MTLDevice, library: MTLLibrary, maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "waveySpheresVertex")
    desc.fragmentFunction = library.makeFunction(name: "waveySpheresFragment")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.depthAttachmentPixelFormat = .depth32Float

    let vd = MTLVertexDescriptor()
    vd.attributes[0].format = .float3
    vd.attributes[0].offset = 0
    vd.attributes[0].bufferIndex = 0
    vd.attributes[1].format = .float3
    vd.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vd.attributes[1].bufferIndex = 0
    vd.layouts[0].stride = MemoryLayout<MeshVertex>.stride
    desc.vertexDescriptor = vd

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
