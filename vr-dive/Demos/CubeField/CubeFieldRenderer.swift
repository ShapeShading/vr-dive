import Metal
import simd

final class CubeFieldRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .cubeField
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let backgroundPipelineState: MTLRenderPipelineState
  private let objectPipelineState: MTLRenderPipelineState
  private let computePipelineState: MTLComputePipelineState
  private let depthStencilState: MTLDepthStencilState
  private let meshVertexBuffer: MTLBuffer
  private let meshIndexBuffer: MTLBuffer
  private let meshIndexCount: Int
  private let objectStateBuffer: MTLBuffer
  private var lastSimulationTimestamp: Float = 0
  private let objectCount: Int

  init(device: MTLDevice, library: MTLLibrary, objectCount: Int) throws {
    self.objectCount = objectCount
    backgroundPipelineState = try CubeFieldRenderer.makeBackgroundPipelineState(
      device: device, library: library)
    objectPipelineState = try CubeFieldRenderer.makeObjectPipelineState(
      device: device, library: library)
    computePipelineState = try CubeFieldRenderer.makeComputePipelineState(
      device: device, library: library)
    depthStencilState = CubeFieldRenderer.makeDepthStencilState(device: device)

    let geometry = CubeFieldRenderer.makeCubeGeometry(device: device)
    meshVertexBuffer = geometry.vertexBuffer
    meshIndexBuffer = geometry.indexBuffer
    meshIndexCount = geometry.indexCount

    objectStateBuffer = CubeFieldRenderer.makeInitialObjectStates(
      device: device, count: objectCount)
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

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encodeBackground(with: encoder, context: context)
    encodeObjects(with: encoder, context: context)
  }

  private func encodeBackground(
    with encoder: MTLRenderCommandEncoder, context: PatternRenderContext
  ) {
    var uniforms = BackgroundUniforms(time: context.time, intensity: 0.85)
    let transforms = context.viewData.viewToWorldTransforms
    uniforms.viewToWorldLeft = transforms[0]
    uniforms.viewToWorldRight = transforms[min(context.viewData.viewCount - 1, 1)]

    encoder.setRenderPipelineState(backgroundPipelineState)
    context.applyViewConfiguration(on: encoder)
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<BackgroundUniforms>.stride, index: 0)
    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<BackgroundUniforms>.stride, index: 0)
    encoder.drawPrimitives(
      type: .triangle, vertexStart: 0, vertexCount: 3, instanceCount: context.viewData.viewCount)
  }

  private func encodeObjects(with encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(objectPipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)
    encoder.setTriangleFillMode(.fill)

    encoder.setVertexBuffer(meshVertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(objectStateBuffer, offset: 0, index: 1)

    context.applyViewConfiguration(on: encoder)

    var sceneUniforms = SceneUniforms(
      viewProjectionMatrixLeft: context.viewData.leftViewProjection,
      viewProjectionMatrixRight: context.viewData.rightViewProjection,
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount)
    )

    encoder.setVertexBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 2)
    encoder.setFragmentBytes(&sceneUniforms, length: MemoryLayout<SceneUniforms>.stride, index: 0)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: meshIndexCount,
      indexType: .uint16,
      indexBuffer: meshIndexBuffer,
      indexBufferOffset: 0,
      instanceCount: objectCount
    )
  }
}

extension CubeFieldRenderer {
  fileprivate static func makeBackgroundPipelineState(device: MTLDevice, library: MTLLibrary) throws
    -> MTLRenderPipelineState
  {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "backgroundVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "backgroundFragmentShader")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .triangle
    descriptor.maxVertexAmplificationCount = Renderer.maxViewCount
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  fileprivate static func makeObjectPipelineState(device: MTLDevice, library: MTLLibrary) throws
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
    descriptor.maxVertexAmplificationCount = Renderer.maxViewCount

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
    descriptor.depthCompareFunction = .less
    descriptor.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: descriptor)!
  }

  fileprivate static func makeCubeGeometry(device: MTLDevice) -> (
    vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int
  ) {
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
      4, 6, 5, 4, 7, 6,
      8, 9, 10, 8, 10, 11,
      12, 14, 13, 12, 15, 14,
      16, 17, 18, 16, 18, 19,
      20, 22, 21, 20, 23, 22,
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

  fileprivate static func makeInitialObjectStates(device: MTLDevice, count: Int) -> MTLBuffer {
    var states: [ObjectState] = []
    states.reserveCapacity(count)
    let scaleBoost: Float = 8.0

    for i in 0..<count {
      let type: Float = (i % 7 == 0) ? 1.0 : 0.0
      let zRange: ClosedRange<Float> = -3.2...(-0.8)
      let position = SIMD3<Float>(
        Float.random(in: -1.5...1.5),
        Float.random(in: -0.9...1.1),
        Float.random(in: zRange)
      )
      let jitterBase: ClosedRange<Float> = 0.008...0.04
      let motionAmplitude = SIMD3<Float>(
        Float.random(in: jitterBase) * (type > 0.5 ? 1.2 : 1.0),
        Float.random(in: jitterBase) * 1.5,
        Float.random(in: jitterBase)
      )
      let scale = SIMD3<Float>(
        Float.random(in: 0.15...0.35) * (type > 0.5 ? 1.8 : 1.0),
        Float.random(in: 0.15...0.45),
        Float.random(in: 0.15...0.35)
      )
      let phase = Float.random(in: 0...(.pi * 2))
      let jitterRadius = Float.random(in: 0.04...0.12)

      let scaledPosition = position * scaleBoost
      let scaledAmplitude = motionAmplitude * scaleBoost
      let scaledScale = scale * scaleBoost
      let scaledJitter = jitterRadius * scaleBoost

      states.append(
        ObjectState(
          positionAndType: SIMD4<Float>(scaledPosition, type),
          motionAndPhase: SIMD4<Float>(scaledAmplitude, phase),
          scaleAndPadding: SIMD4<Float>(scaledScale, 0),
          homeAndJitter: SIMD4<Float>(scaledPosition, scaledJitter)
        )
      )
    }

    return device.makeBuffer(
      bytes: states,
      length: MemoryLayout<ObjectState>.stride * states.count,
      options: [.storageModeShared]
    )!
  }
}
