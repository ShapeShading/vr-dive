import ARKit
import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI

struct BackgroundUniforms {
  var time: Float
  var intensity: Float
  var padding: SIMD2<Float> = .zero
  var viewToWorldLeft: simd_float4x4 = matrix_identity_float4x4
  var viewToWorldRight: simd_float4x4 = matrix_identity_float4x4
}

struct SceneUniforms {
  var viewProjectionMatrixLeft: simd_float4x4
  var viewProjectionMatrixRight: simd_float4x4
  var time: Float
  var layerCount: UInt32
  var padding: SIMD2<Float> = .zero
}

struct SimulationUniforms {
  var deltaTime: Float
  var globalTime: Float
  var objectCount: UInt32
  var padding: UInt32 = 0
}

struct MeshVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
}

struct ObjectState {
  var positionAndType: SIMD4<Float>
  var motionAndPhase: SIMD4<Float>
  var scaleAndPadding: SIMD4<Float>
  var homeAndJitter: SIMD4<Float>
}

struct VRConfiguration: CompositorLayerConfiguration {
  func makeConfiguration(
    capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration
  ) {
    let supportsFoveation = capabilities.supportsFoveation
    configuration.isFoveationEnabled = supportsFoveation

    _ = capabilities.supportedLayouts(
      options: supportsFoveation ? [.foveationEnabled] : [])
    configuration.layout = .layered

    configuration.colorFormat = .rgba16Float
    configuration.depthFormat = .depth32Float
  }
}

class Renderer {
  let layerRenderer: LayerRenderer
  let device: MTLDevice
  let commandQueue: MTLCommandQueue
  let backgroundPipelineState: MTLRenderPipelineState
  let objectPipelineState: MTLRenderPipelineState
  let computePipelineState: MTLComputePipelineState
  let depthStencilState: MTLDepthStencilState
  let meshVertexBuffer: MTLBuffer
  let meshIndexBuffer: MTLBuffer
  let meshIndexCount: Int
  let objectStateBuffer: MTLBuffer
  let arSession: ARKitSession
  let worldTracking: WorldTrackingProvider
  let gameManager: GameManager
  private var lastKnownDeviceAnchor: ARKit.DeviceAnchor?
  private var lastSimulationTimestamp: Float = 0
  private var rigTransform: simd_float4x4 = matrix_identity_float4x4
  private var lastRigUpdateTime: Float = 0
  private let objectCount = 48
  private static let largeObjectScale: Float = 8.0

  var startTime: Date = Date()
  private static let attosecondsPerSecond = 1_000_000_000_000_000_000.0
  private static let maxViewCount = 2

  private struct ViewRenderingData {
    var leftViewProjection: simd_float4x4
    var rightViewProjection: simd_float4x4
    var viewports: [MTLViewport]
    var renderTargetIndices: [UInt32]
    var viewToWorldTransforms: [simd_float4x4]
    var viewCount: Int
  }

  init(_ layerRenderer: LayerRenderer) {
    self.layerRenderer = layerRenderer
    self.device = layerRenderer.device
    self.commandQueue = self.device.makeCommandQueue()!

    let library = device.makeDefaultLibrary()!
    self.backgroundPipelineState = try! Renderer.makeBackgroundPipelineState(
      device: device, library: library)
    self.objectPipelineState = try! Renderer.makeObjectPipelineState(
      device: device, library: library)
    self.computePipelineState = try! Renderer.makeComputePipelineState(
      device: device, library: library)
    self.depthStencilState = Renderer.makeDepthStencilState(device: device)

    let geometry = Renderer.makeCubeGeometry(device: device)
    self.meshVertexBuffer = geometry.vertexBuffer
    self.meshIndexBuffer = geometry.indexBuffer
    self.meshIndexCount = geometry.indexCount

    self.objectStateBuffer = Renderer.makeInitialObjectStates(device: device, count: objectCount)

    self.arSession = ARKitSession()
    self.worldTracking = WorldTrackingProvider()
    self.gameManager = GameManager()
  }

  func startRenderLoop() {
    print("[Renderer] Starting render loop...")
    Task {
      do {
        try await arSession.run([worldTracking])
        print("[Renderer] ARSession started successfully")
      } catch {
        print("[Renderer] Failed to start ARSession: \(error)")
      }
    }

    let renderThread = Thread {
      self.renderLoop()
    }
    renderThread.name = "Render Thread"
    renderThread.start()
    print("[Renderer] Render thread started")
  }

  func renderLoop() {
    print("[Renderer] Render loop started, layerRenderer state: \(layerRenderer.state)")
    var frameCount = 0
    while true {
      if layerRenderer.state == .invalidated {
        print("[Renderer] Layer renderer invalidated, exiting")
        break
      }
      guard layerRenderer.state == .running else {
        if frameCount == 0 {
          print(
            "[Renderer] Waiting for layerRenderer to be running, current state: \(layerRenderer.state)"
          )
        }
        Thread.sleep(forTimeInterval: 0.01)
        continue
      }

      if frameCount == 0 {
        print("[Renderer] First frame rendering...")
      }
      if frameCount % 60 == 0 {
        print("[Renderer] Frame \(frameCount) rendered")
      }
      frameCount += 1

      guard let frame = layerRenderer.queryNextFrame() else { continue }

      frame.startUpdate()

      let animationTime = Float(Date().timeIntervalSince(startTime))
      let predictedTiming = frame.predictTiming()
      let presentationTimestamp = presentationTimeInterval(from: predictedTiming)

      frame.endUpdate()

      let drawables = frame.queryDrawables()
      guard !drawables.isEmpty else { continue }

      var deviceAnchor: ARKit.DeviceAnchor?
      if worldTracking.state == .running {
        deviceAnchor = worldTracking.queryDeviceAnchor(
          atTimestamp: presentationTimestamp ?? CACurrentMediaTime()
        )
        if let validAnchor = deviceAnchor {
          lastKnownDeviceAnchor = validAnchor
        }
      }

      let anchorToUse = deviceAnchor ?? lastKnownDeviceAnchor
      if anchorToUse == nil, frameCount % 120 == 0 {
        print("[Renderer] Waiting for reliable world tracking data...")
      }

      simulateObjectsIfNeeded(currentTime: animationTime)
      if let anchor = anchorToUse {
        updateRigTransformIfNeeded(
          deviceAnchorTransform: anchor.originFromAnchorTransform,
          currentTime: animationTime
        )
      }

      autoreleasepool {
        frame.startSubmission()
        defer { frame.endSubmission() }

        for drawable in drawables {
          if let anchor = anchorToUse {
            render(drawable: drawable, deviceAnchor: anchor, time: animationTime)
          } else {
            presentDrawableWithoutRendering(drawable)
          }
        }
      }
    }
  }

  private func presentationTimeInterval(from timing: LayerRenderer.Frame.Timing?) -> TimeInterval? {
    guard let instant = timing?.presentationTime else { return nil }
    let duration = LayerRenderer.Clock.Instant.epoch.duration(to: instant)
    let components = duration.components
    let seconds = Double(components.seconds)
    let attoseconds = Double(components.attoseconds) / Renderer.attosecondsPerSecond
    return seconds + attoseconds
  }

  private func render(
    drawable: LayerRenderer.Drawable,
    deviceAnchor: ARKit.DeviceAnchor,
    time: Float
  ) {
    let viewCount = resolvedViewCount(for: drawable)
    guard viewCount > 0 else { return }
    guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

    drawable.deviceAnchor = deviceAnchor

    guard let colorTexture = drawable.colorTextures.first else { return }
    let viewData = makeViewRenderingData(
      drawable: drawable,
      deviceAnchor: deviceAnchor,
      colorTexture: colorTexture,
      viewCount: viewCount
    )

    if let renderPassDescriptor = makeRenderPassDescriptor(
      for: drawable, viewCount: viewData.viewCount),
      let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
    {
      encodeBackgroundPass(with: encoder, viewData: viewData, time: time)
      encodeObjectPass(with: encoder, viewData: viewData, time: time)
      encoder.endEncoding()
    }

    drawable.encodePresent(commandBuffer: commandBuffer)
    commandBuffer.commit()
  }

  private func presentDrawableWithoutRendering(_ drawable: LayerRenderer.Drawable) {
    guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
    drawable.encodePresent(commandBuffer: commandBuffer)
    commandBuffer.commit()
  }

  private func makeRenderPassDescriptor(for drawable: LayerRenderer.Drawable, viewCount: Int)
    -> MTLRenderPassDescriptor?
  {
    guard let colorTexture = drawable.colorTextures.first else { return nil }
    let descriptor = MTLRenderPassDescriptor()
    let colorAttachment = descriptor.colorAttachments[0] ?? MTLRenderPassColorAttachmentDescriptor()
    descriptor.colorAttachments[0] = colorAttachment
    colorAttachment.texture = colorTexture
    colorAttachment.loadAction = .clear
    colorAttachment.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    colorAttachment.storeAction = .store
    descriptor.renderTargetArrayLength = max(min(colorTexture.arrayLength, viewCount), 1)

    if let depthTexture = drawable.depthTextures.first {
      let depthAttachment = descriptor.depthAttachment ?? MTLRenderPassDepthAttachmentDescriptor()
      descriptor.depthAttachment = depthAttachment
      depthAttachment.texture = depthTexture
      depthAttachment.loadAction = .clear
      depthAttachment.storeAction = .store
      depthAttachment.clearDepth = 1.0
    }

    if let rateMap = drawable.rasterizationRateMaps.first {
      descriptor.rasterizationRateMap = rateMap
    }

    return descriptor
  }

  private func encodeBackgroundPass(
    with encoder: MTLRenderCommandEncoder,
    viewData: ViewRenderingData,
    time: Float
  ) {
    var uniforms = BackgroundUniforms(time: time, intensity: 0.85)
    uniforms.viewToWorldLeft = viewData.viewToWorldTransforms[0]
    uniforms.viewToWorldRight = viewData.viewToWorldTransforms[min(viewData.viewCount - 1, 1)]

    encoder.setRenderPipelineState(backgroundPipelineState)
    applyViewConfiguration(on: encoder, using: viewData)
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<BackgroundUniforms>.stride, index: 0)
    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<BackgroundUniforms>.stride, index: 0)
    encoder.drawPrimitives(
      type: .triangle,
      vertexStart: 0,
      vertexCount: 3,
      instanceCount: viewData.viewCount
    )
  }

  private func applyViewConfiguration(
    on encoder: MTLRenderCommandEncoder, using viewData: ViewRenderingData
  ) {
    if !viewData.viewports.isEmpty {
      encoder.setViewports(viewData.viewports)
    }

    if viewData.viewCount > 1 {
      var viewMappings = (0..<viewData.viewCount).map {
        MTLVertexAmplificationViewMapping(
          viewportArrayIndexOffset: UInt32($0),
          renderTargetArrayIndexOffset: viewData.renderTargetIndices[$0]
        )
      }
      encoder.setVertexAmplificationCount(viewData.viewCount, viewMappings: &viewMappings)
    } else {
      encoder.setVertexAmplificationCount(1, viewMappings: nil)
    }
  }

  private func encodeObjectPass(
    with encoder: MTLRenderCommandEncoder,
    viewData: ViewRenderingData,
    time: Float
  ) {
    encoder.setRenderPipelineState(objectPipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)
    encoder.setTriangleFillMode(.fill)

    encoder.setVertexBuffer(meshVertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(objectStateBuffer, offset: 0, index: 1)

    applyViewConfiguration(on: encoder, using: viewData)

    var sceneUniforms = SceneUniforms(
      viewProjectionMatrixLeft: viewData.leftViewProjection,
      viewProjectionMatrixRight: viewData.rightViewProjection,
      time: time,
      layerCount: UInt32(viewData.viewCount)
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

  private func resolvedViewCount(for drawable: LayerRenderer.Drawable) -> Int {
    let textureArrayLength = drawable.colorTextures.first?.arrayLength ?? 1
    let viewsCount = drawable.views.count
    let limitedByTextures = min(textureArrayLength, Renderer.maxViewCount)
    let limitedByViews = min(viewsCount, Renderer.maxViewCount)
    let resolved = min(limitedByTextures, max(limitedByViews, 1))
    return max(resolved, 1)
  }

  private func makeViewRenderingData(
    drawable: LayerRenderer.Drawable,
    deviceAnchor: ARKit.DeviceAnchor,
    colorTexture: any MTLTexture,
    viewCount: Int
  ) -> ViewRenderingData {
    var viewports: [MTLViewport] = []
    var matrices = Array(repeating: matrix_identity_float4x4, count: Renderer.maxViewCount)
    var renderTargetIndices: [UInt32] = []
    var viewToWorldTransforms: [simd_float4x4] = []
    let desiredViewCount = max(min(viewCount, Renderer.maxViewCount), 1)
    let availableViews = drawable.views
    let sampledViewCount = min(desiredViewCount, availableViews.count)

    if sampledViewCount > 0 {
      let deviceMatrix = deviceAnchor.originFromAnchorTransform
      for index in 0..<sampledViewCount {
        let view = availableViews[index]
        let localTransform = view.transform
        let worldFromEye = deviceMatrix * localTransform
        let adjustedWorldFromEye = rigTransform * worldFromEye
        let viewMatrix = adjustedWorldFromEye.inverse
        let projection = projectionMatrix(for: drawable, view: view, viewIndex: index)
        matrices[index] = projection * viewMatrix
        viewports.append(view.textureMap.viewport)
        renderTargetIndices.append(UInt32(view.textureMap.textureIndex))
        viewToWorldTransforms.append(adjustedWorldFromEye)
      }
    }

    if viewports.isEmpty {
      matrices[0] = fallbackViewProjection(for: colorTexture)
      let fallbackViewport = MTLViewport(
        originX: 0,
        originY: 0,
        width: Double(colorTexture.width),
        height: Double(colorTexture.height),
        znear: 0,
        zfar: 1
      )
      viewports.append(fallbackViewport)
      renderTargetIndices.append(0)
      viewToWorldTransforms.append(rigTransform)
    }

    let fallbackMatrix = matrices[max(min(sampledViewCount - 1, Renderer.maxViewCount - 1), 0)]
    let fallbackRenderIndex = renderTargetIndices.last ?? 0
    let fallbackTransform = viewToWorldTransforms.last ?? matrix_identity_float4x4
    if sampledViewCount < desiredViewCount {
      for index in sampledViewCount..<desiredViewCount {
        matrices[index] = fallbackMatrix
        renderTargetIndices.append(fallbackRenderIndex)
        viewToWorldTransforms.append(fallbackTransform)
      }
    }

    while viewports.count < desiredViewCount {
      viewports.append(viewports.last ?? viewports[0])
    }

    while renderTargetIndices.count < desiredViewCount {
      renderTargetIndices.append(renderTargetIndices.last ?? 0)
    }

    while viewToWorldTransforms.count < desiredViewCount {
      viewToWorldTransforms.append(viewToWorldTransforms.last ?? matrix_identity_float4x4)
    }

    let rightIndex = desiredViewCount > 1 ? 1 : 0
    return ViewRenderingData(
      leftViewProjection: matrices[0],
      rightViewProjection: matrices[rightIndex],
      viewports: Array(viewports.prefix(desiredViewCount)),
      renderTargetIndices: Array(renderTargetIndices.prefix(desiredViewCount)),
      viewToWorldTransforms: Array(viewToWorldTransforms.prefix(desiredViewCount)),
      viewCount: desiredViewCount
    )
  }

  private func projectionMatrix(
    for drawable: LayerRenderer.Drawable,
    view: LayerRenderer.Drawable.View,
    viewIndex: Int
  ) -> simd_float4x4 {
    if #available(visionOS 2.0, *) {
      return drawable.computeProjection(convention: .rightUpBack, viewIndex: viewIndex)
    } else {
      let tangents = view.tangents
      let depthRange = drawable.depthRange
      let projective = ProjectiveTransform3D(
        leftTangent: Double(tangents[0]),
        rightTangent: Double(tangents[1]),
        topTangent: Double(tangents[2]),
        bottomTangent: Double(tangents[3]),
        nearZ: Double(depthRange.x),
        farZ: Double(depthRange.y),
        reverseZ: true
      )
      return simd_float4x4(projective.matrix)
    }
  }

  private func fallbackViewProjection(for colorTexture: any MTLTexture) -> simd_float4x4 {
    let aspect = Float(colorTexture.width) / Float(max(colorTexture.height, 1))
    let projection = simd_float4x4.perspective(
      fovY: 60 * (.pi / 180),
      aspect: aspect,
      nearZ: 0.05,
      farZ: 20
    )
    let view = simd_float4x4.lookAt(
      eye: SIMD3<Float>(0, 0, 0.4),
      center: SIMD3<Float>(0, 0, -1),
      up: SIMD3<Float>(0, 1, 0)
    )
    return projection * view
  }

  private func simulateObjectsIfNeeded(currentTime: Float) {
    let deltaTime = max(0, currentTime - lastSimulationTimestamp)
    guard deltaTime > 0 else { return }

    var uniforms = SimulationUniforms(
      deltaTime: min(deltaTime, 1.0 / 30.0),
      globalTime: currentTime,
      objectCount: UInt32(objectCount)
    )

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
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

    lastSimulationTimestamp = currentTime
  }

  private func updateRigTransformIfNeeded(deviceAnchorTransform: simd_float4x4, currentTime: Float)
  {
    let delta = max(0, currentTime - lastRigUpdateTime)
    guard delta > 0 else { return }
    rigTransform = gameManager.updateRigState(
      deltaTime: delta, headTransform: deviceAnchorTransform)
    lastRigUpdateTime = currentTime
  }
}

extension Renderer {
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

    let scaleBoost = Renderer.largeObjectScale

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

extension simd_float4x4 {
  fileprivate static func perspective(fovY: Float, aspect: Float, nearZ: Float, farZ: Float)
    -> simd_float4x4
  {
    let yScale = 1 / tan(fovY * 0.5)
    let xScale = yScale / max(aspect, 0.1)
    let zRange = farZ - nearZ
    let zScale = -(farZ + nearZ) / zRange
    let wzScale = -(2 * farZ * nearZ) / zRange

    return simd_float4x4(
      SIMD4<Float>(xScale, 0, 0, 0),
      SIMD4<Float>(0, yScale, 0, 0),
      SIMD4<Float>(0, 0, zScale, -1),
      SIMD4<Float>(0, 0, wzScale, 0)
    )
  }

  fileprivate static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>)
    -> simd_float4x4
  {
    let forward = simd_normalize(center - eye)
    let right = simd_normalize(simd_cross(forward, up))
    let correctedUp = simd_cross(right, forward)

    let translation = SIMD3<Float>(
      -simd_dot(right, eye),
      -simd_dot(correctedUp, eye),
      simd_dot(forward, eye)
    )

    return simd_float4x4(
      SIMD4<Float>(right.x, correctedUp.x, -forward.x, 0),
      SIMD4<Float>(right.y, correctedUp.y, -forward.y, 0),
      SIMD4<Float>(right.z, correctedUp.z, -forward.z, 0),
      SIMD4<Float>(translation.x, translation.y, translation.z, 1)
    )
  }

  fileprivate init(_ matrix: simd_double4x4) {
    self.init(
      columns: (
        SIMD4<Float>(matrix.columns.0),
        SIMD4<Float>(matrix.columns.1),
        SIMD4<Float>(matrix.columns.2),
        SIMD4<Float>(matrix.columns.3)
      ))
  }
}

extension SIMD4 where Scalar == Float {
  fileprivate init(_ vector: SIMD4<Double>) {
    self.init(Float(vector.x), Float(vector.y), Float(vector.z), Float(vector.w))
  }
}
