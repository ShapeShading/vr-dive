import ARKit
import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI

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
  let arSession: ARKitSession
  let worldTracking: WorldTrackingProvider
  let gameManager: GameManager
  let patternCoordinator: PatternCoordinator
  private var patternControllers: [VisualPatternKind: VisualPatternController] = [:]
  private var activePatternKind: VisualPatternKind
  private var lastKnownDeviceAnchor: ARKit.DeviceAnchor?
  private var rigTransform: simd_float4x4 = matrix_identity_float4x4
  private var lastRigUpdateTime: Float = 0
  private static let cubeObjectCount = 48
  private static let lorenzParticleCount = 400000

  var startTime: Date = Date()
  private static let attosecondsPerSecond = 1_000_000_000_000_000_000.0
  static let maxViewCount = 2

  init(_ layerRenderer: LayerRenderer, patternCoordinator: PatternCoordinator) {
    self.layerRenderer = layerRenderer
    self.device = layerRenderer.device
    self.commandQueue = self.device.makeCommandQueue()!
    self.patternCoordinator = patternCoordinator

    let library = device.makeDefaultLibrary()!
    let controllers = Renderer.makePatternControllers(
      device: device,
      library: library,
      cubeCount: Renderer.cubeObjectCount,
      lorenzCount: Renderer.lorenzParticleCount
    )
    self.patternControllers = controllers
    let requestedPattern = patternCoordinator.currentPattern()
    if controllers[requestedPattern] != nil {
      self.activePatternKind = requestedPattern
    } else if let fallback = controllers.keys.first {
      self.activePatternKind = fallback
      patternCoordinator.setPattern(fallback)
    } else {
      fatalError("No render patterns available")
    }

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

  private func resolveActivePatternController() -> VisualPatternController? {
    let desiredPattern = patternCoordinator.currentPattern()
    if desiredPattern != activePatternKind, let controller = patternControllers[desiredPattern] {
      activePatternKind = desiredPattern
      print("[Renderer] Switching to pattern: \(desiredPattern.rawValue)")
      return controller
    }

    if let controller = patternControllers[activePatternKind] {
      return controller
    }

    if let fallback = patternControllers.values.first {
      activePatternKind = fallback.identifier
      return fallback
    }

    return nil
  }

  private func render(
    drawable: LayerRenderer.Drawable,
    deviceAnchor: ARKit.DeviceAnchor,
    time: Float
  ) {
    let viewCount = resolvedViewCount(for: drawable)
    guard viewCount > 0 else { return }

    guard let pattern = resolveActivePatternController() else { return }
    let simulationContext = PatternSimulationContext(commandQueue: commandQueue, time: time)
    pattern.updateSimulation(simulationContext)

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
      for: drawable,
      viewCount: viewData.viewCount,
      clearColor: pattern.preferredClearColor),
      let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
    {
      let renderContext = PatternRenderContext(viewData: viewData, time: time)
      pattern.encodeFrame(encoder: encoder, context: renderContext)
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

  private func makeRenderPassDescriptor(
    for drawable: LayerRenderer.Drawable,
    viewCount: Int,
    clearColor: MTLClearColor
  ) -> MTLRenderPassDescriptor? {
    guard let colorTexture = drawable.colorTextures.first else { return nil }
    let descriptor = MTLRenderPassDescriptor()
    let colorAttachment = descriptor.colorAttachments[0] ?? MTLRenderPassColorAttachmentDescriptor()
    descriptor.colorAttachments[0] = colorAttachment
    colorAttachment.texture = colorTexture
    colorAttachment.loadAction = .clear
    colorAttachment.clearColor = clearColor
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

  private func updateRigTransformIfNeeded(deviceAnchorTransform: simd_float4x4, currentTime: Float)
  {
    let delta = max(0, currentTime - lastRigUpdateTime)
    guard delta > 0 else { return }
    rigTransform = gameManager.updateRigState(
      deltaTime: delta, headTransform: deviceAnchorTransform)
    lastRigUpdateTime = currentTime
  }

  private static func makePatternControllers(
    device: MTLDevice,
    library: MTLLibrary,
    cubeCount: Int,
    lorenzCount: Int
  ) -> [VisualPatternKind: VisualPatternController] {
    var controllers: [VisualPatternKind: VisualPatternController] = [:]
    do {
      controllers[.cubeField] = try CubeFieldRenderer(
        device: device, library: library, objectCount: cubeCount)
    } catch {
      fatalError("Failed to build cube pattern: \(error)")
    }

    if let lorenz = try? LorenzRenderer(
      device: device,
      library: library,
      particleCount: lorenzCount
    ) {
      controllers[.lorenzAttractor] = lorenz
    } else {
      print("[Renderer] Lorenz attractor pattern unavailable; continuing with base pattern only.")
    }

    return controllers
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
