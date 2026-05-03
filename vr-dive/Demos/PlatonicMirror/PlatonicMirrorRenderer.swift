import Metal
import simd

// PlatonicMirrorRenderer.swift
//
// Renders a Platonic solid with internal mirror reflections using ray marching.
// A UV-sphere mesh acts as the container; the fragment shader does all ray work.
//
// Shader source: "Let's self reflect" by mrange
// https://www.shadertoy.com/view/XfyXRV  (License: CC0)

final class PlatonicMirrorRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .platonicMirror
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int
  private let maxViewCount: Int

  // Local bounding sphere radius.  The Platonic solid with poly_zoom=2 fits
  // within roughly ±2.8 local units, so 3.1 gives comfortable clearance.
  private static let localBoundRadius: Float = 3.1

  // World-space placement: 0.35 m per local unit, placed ~1.8 m in front.
  private let solidScale: Float = 0.35
  private let objectCenter = SIMD3<Float>(0.0, -0.15, -1.8)

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    // Vertices are stored in local units; the vertex shader scales by solidScale.
    let geo = PlatonicMirrorRenderer.makeUVSphere(
      device: device,
      radius: PlatonicMirrorRenderer.localBoundRadius,
      latSegments: 24,
      lonSegments: 48)
    vertexBuffer = geo.vertexBuffer
    indexBuffer = geo.indexBuffer
    indexCount = geo.indexCount

    pipelineState = try PlatonicMirrorRenderer.makePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    depthStencilState = PlatonicMirrorRenderer.makeDepthStencilState(device: device)
  }

  func updateSimulation(_ context: PatternSimulationContext) {}
  func resetToInitialState() {}

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.back)
    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    var uniforms = PlatonicMirrorUniforms(
      time: context.time,
      viewCount: UInt32(context.viewData.viewCount),
      solidScale: solidScale,
      _pad: 0,
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0))

    encoder.setVertexBytes(&uniforms, length: MemoryLayout<PlatonicMirrorUniforms>.stride, index: 1)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 2)
      }
    }

    // Fragment: uniforms, viewToWorld, viewProjection
    encoder.setFragmentBytes(
      &uniforms, length: MemoryLayout<PlatonicMirrorUniforms>.stride, index: 0)

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

// MARK: - Geometry & pipeline factory
extension PlatonicMirrorRenderer {

  fileprivate static func makeUVSphere(
    device: MTLDevice, radius: Float, latSegments: Int, lonSegments: Int
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    var vertices: [MeshVertex] = []
    var indices: [UInt16] = []
    vertices.reserveCapacity((latSegments + 1) * (lonSegments + 1))
    indices.reserveCapacity(latSegments * lonSegments * 6)

    for lat in 0...latSegments {
      let theta = Float(lat) * .pi / Float(latSegments)
      let sinT = sin(theta)
      let cosT = cos(theta)
      for lon in 0...lonSegments {
        let phi = Float(lon) * 2 * .pi / Float(lonSegments)
        let n = SIMD3<Float>(cos(phi) * sinT, cosT, sin(phi) * sinT)
        vertices.append(MeshVertex(position: n * radius, normal: n))
      }
    }

    let stride = UInt16(lonSegments + 1)
    for lat in 0..<latSegments {
      for lon in 0..<lonSegments {
        let a = UInt16(lat) * stride + UInt16(lon)
        let b = a + stride
        indices.append(contentsOf: [a, b, a + 1, b, b + 1, a + 1])
      }
    }

    let vBuf = device.makeBuffer(
      bytes: vertices, length: MemoryLayout<MeshVertex>.stride * vertices.count,
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
    desc.vertexFunction = library.makeFunction(name: "platonicMirrorVertex")
    desc.fragmentFunction = library.makeFunction(name: "platonicMirrorFragment")
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
    desc.depthCompareFunction = .greater  // reverse-Z: near = 1, far = 0
    desc.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: desc)!
  }
}
