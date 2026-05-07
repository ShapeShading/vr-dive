import Metal
import simd

// GlassBoxRenderer.swift
//
// Renders a glass box with bilinear-patch internal structure using ray marching.
// Architecture follows RhombicDodecahedronRenderer: a bounding sphere mesh acts
// as the container; the fragment shader does all ray-marching work.
//
// Local BOXDIMS = (0.95, 0.95, 1.25).  Circumscribed sphere radius ≈ 1.84.
// World-space box size = BOXDIMS * boxScale.  The sphere radius matches accordingly.

final class GlassBoxRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .glassBox
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int
  private let maxViewCount: Int

  // World-space scale and placement.
  private let boxScale: Float = 0.84
  private let objectCenter = SIMD3<Float>(0.0, -0.05, -1.1)
  private var animationTime: Float = 0
  private var lastSimulationTime: Float?

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    // Box mesh in local BOXDIMS space (slightly enlarged so the rasterised mesh
    // fully covers all visible pixels before the fragment shader takes over).
    let geo = GlassBoxRenderer.makeBox(
      device: device, localHalfExtents: SIMD3<Float>(0.95, 0.95, 1.25) * 1.02)
    vertexBuffer = geo.vertexBuffer
    indexBuffer = geo.indexBuffer
    indexCount = geo.indexCount

    pipelineState = try GlassBoxRenderer.makePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    depthStencilState = GlassBoxRenderer.makeDepthStencilState(device: device)
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

    // Vertex buffers
    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    var uniforms = GlassBoxUniforms(
      time: animationTime,
      viewCount: UInt32(context.viewData.viewCount),
      boxScale: boxScale,
      _pad: 0,
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0))

    encoder.setVertexBytes(&uniforms, length: MemoryLayout<GlassBoxUniforms>.stride, index: 1)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 2)
      }
    }

    // Fragment buffers
    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<GlassBoxUniforms>.stride, index: 0)

    var viewToWorld = context.viewData.viewToWorldTransforms
    if viewToWorld.isEmpty { viewToWorld = [matrix_identity_float4x4] }
    viewToWorld.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setFragmentBytes(base, length: $0.count, index: 1)
      }
    }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setFragmentBytes(base, length: $0.count, index: 2)
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

// MARK: - Geometry & Pipeline factory
extension GlassBoxRenderer {

  /// Build a box mesh with the given half-extents in local (BOXDIMS) space.
  /// 6 faces × 4 verts = 24 vertices, 6 × 2 triangles = 36 indices.
  /// Normals point outward so back-face culling shows the correct faces
  /// when viewed from outside.
  fileprivate static func makeBox(
    device: MTLDevice, localHalfExtents e: SIMD3<Float>
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    typealias V = MeshVertex
    let (x, y, z) = (e.x, e.y, e.z)
    // Each face: 4 corner positions + outward normal, wound CCW from outside.
    let faces: [(positions: [SIMD3<Float>], normal: SIMD3<Float>)] = [
      ([[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], [0, 0, 1]),  // +Z
      ([[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]], [0, 0, -1]),  // -Z
      ([[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]], [1, 0, 0]),  // +X
      ([[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]], [-1, 0, 0]),  // -X
      ([[-x, y, z], [x, y, z], [x, y, -z], [-x, y, -z]], [0, 1, 0]),  // +Y
      ([[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]], [0, -1, 0]),  // -Y
    ]
    var vertices: [V] = []
    vertices.reserveCapacity(24)
    var indices: [UInt16] = []
    indices.reserveCapacity(36)
    for face in faces {
      let base = UInt16(vertices.count)
      for p in face.positions { vertices.append(V(position: p, normal: face.normal)) }
      indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }
    let vBuf = device.makeBuffer(
      bytes: vertices, length: MemoryLayout<V>.stride * vertices.count,
      options: .storageModeShared)!
    let iBuf = device.makeBuffer(
      bytes: indices, length: MemoryLayout<UInt16>.stride * indices.count,
      options: .storageModeShared)!
    return (vBuf, iBuf, indices.count)
  }

  fileprivate static func makePipelineState(
    device: MTLDevice, library: MTLLibrary, maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "glassBoxVertex")
    desc.fragmentFunction = library.makeFunction(name: "glassBoxFragment")
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
    desc.depthCompareFunction = .greater  // reverse-Z: near=1, far=0
    desc.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: desc)!
  }
}
