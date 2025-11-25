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

    let layoutOptions: LayerRenderer.Capabilities.SupportedLayoutsOptions = supportsFoveation
      ? [.foveationEnabled]
      : []
    let supportedLayouts = capabilities.supportedLayouts(options: layoutOptions)
    if supportedLayouts.contains(.layered) {
      configuration.layout = .layered
    } else if let fallbackLayout = supportedLayouts.first {
      configuration.layout = fallbackLayout
      print("[VRConfiguration] Falling back to supported layout: \(fallbackLayout)")
    } else {
      configuration.layout = .layered
      print("[VRConfiguration] No supported layouts reported; defaulting to layered")
    }

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
  private let maxViewCount: Int
  private var lastKnownDeviceAnchor: ARKit.DeviceAnchor?
  private var rigTransform: simd_float4x4 = matrix_identity_float4x4
  private var lastRigUpdateTime: Float = 0
  private static let cubeObjectCount = 48
  private static let lorenzParticleCount = 400000
  private static let fourWingParticleCount = 400000
  private static let aizawaParticleCount = 400000
  private var didLogDrawableLayout = false

  var startTime: Date = Date()
  private static let attosecondsPerSecond = 1_000_000_000_000_000_000.0

  init(_ layerRenderer: LayerRenderer, patternCoordinator: PatternCoordinator) {
    self.layerRenderer = layerRenderer
    self.device = layerRenderer.device
    self.commandQueue = self.device.makeCommandQueue()!
    self.patternCoordinator = patternCoordinator
    self.maxViewCount = max(1, layerRenderer.properties.viewCount)

    let library = device.makeDefaultLibrary()!
    let controllers = Renderer.makePatternControllers(
      device: device,
      library: library,
      cubeCount: Renderer.cubeObjectCount,
      lorenzCount: Renderer.lorenzParticleCount,
      fourWingCount: Renderer.fourWingParticleCount,
      aizawaCount: Renderer.aizawaParticleCount,
      maxViewCount: maxViewCount
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
      } else if frameCount % 60 == 0 {
        print("[Renderer] Frame \(frameCount) rendered")
      }

      guard let frame = layerRenderer.queryNextFrame() else { continue }

      var shouldSkipFrame = false

      autoreleasepool {
        frame.startUpdate()

        let animationTime = Float(Date().timeIntervalSince(startTime))
        let predictedTiming = frame.predictTiming()
        let presentationTimestamp = presentationTimeInterval(from: predictedTiming)
        let drawables = frame.queryDrawables()

        guard !drawables.isEmpty else {
          frame.endUpdate()
          shouldSkipFrame = true
          return
        }

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
        if let anchor = anchorToUse {
          updateRigTransformIfNeeded(
            deviceAnchorTransform: anchor.originFromAnchorTransform,
            currentTime: animationTime
          )
        } else if frameCount % 120 == 0 {
          print("[Renderer] Waiting for reliable world tracking data...")
        }

        let pattern = resolveActivePatternController()

        if patternCoordinator.shouldReset() {
          pattern?.resetToInitialState()
          patternCoordinator.clearResetFlag()
        }

        if let activePattern = pattern, !patternCoordinator.isPaused() {
          let simulationContext = PatternSimulationContext(
            commandQueue: commandQueue,
            time: animationTime,
            speedMultiplier: patternCoordinator.speedMultiplier()
          )
          activePattern.updateSimulation(simulationContext)
        }

        var pendingCommands: [(LayerRenderer.Drawable, MTLCommandBuffer)] = []
        for drawable in drawables {
          if let commandBuffer = encodeDrawable(
            drawable: drawable,
            pattern: pattern,
            deviceAnchor: anchorToUse,
            time: animationTime
          ) {
            pendingCommands.append((drawable, commandBuffer))
          }
        }

        frame.endUpdate()

        guard !pendingCommands.isEmpty else {
          shouldSkipFrame = true
          return
        }

        frame.startSubmission()

        for (drawable, commandBuffer) in pendingCommands {
          drawable.deviceAnchor = anchorToUse
          drawable.encodePresent(commandBuffer: commandBuffer)
          commandBuffer.commit()
        }

        frame.endSubmission()
      }

      if shouldSkipFrame {
        continue
      }

      frameCount += 1
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

  private func encodeDrawable(
    drawable: LayerRenderer.Drawable,
    pattern: VisualPatternController?,
    deviceAnchor: ARKit.DeviceAnchor?,
    time: Float
  ) -> MTLCommandBuffer? {
    let viewCount = resolvedViewCount(for: drawable)
    guard viewCount > 0 else { return nil }
    guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }
    guard let colorTexture = drawable.colorTextures.first else { return nil }

    if !didLogDrawableLayout {
      didLogDrawableLayout = true
      print(
        "[Renderer] Drawable layout: colorTextures=\(drawable.colorTextures.count) depthTextures=\(drawable.depthTextures.count) views=\(drawable.views.count)"
      )
      for (index, texture) in drawable.colorTextures.enumerated() {
        print(
          "[Renderer]  colorTexture[\(index)] size=\(texture.width)x\(texture.height) layers=\(texture.arrayLength)"
        )
      }
      for (index, depthTexture) in drawable.depthTextures.enumerated() {
        print(
          "[Renderer]  depthTexture[\(index)] size=\(depthTexture.width)x\(depthTexture.height) layers=\(depthTexture.arrayLength)"
        )
      }
      for (index, view) in drawable.views.enumerated() {
        let viewport = view.textureMap.viewport
        let textureMap = view.textureMap
        print(
          "[Renderer]  view[\(index)] texIndex=\(textureMap.textureIndex) slice=\(textureMap.sliceIndex) viewport=(\(viewport.originX), \(viewport.originY), \(viewport.width), \(viewport.height))"
        )
      }
    }

    let clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

    guard
      let descriptor = makeRenderPassDescriptor(
        for: drawable,
        viewCount: viewCount,
        clearColor: clearColor
      ),
      let sceneEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)
    else { return nil }

    if let activePattern = pattern, let anchor = deviceAnchor ?? lastKnownDeviceAnchor {
      let viewData = makeViewRenderingData(
        drawable: drawable,
        deviceAnchor: anchor,
        colorTexture: colorTexture,
        viewCount: viewCount
      )
      let sceneContext = PatternRenderContext(
        viewData: viewData,
        time: time
      )
      activePattern.encodeFrame(encoder: sceneEncoder, context: sceneContext)
    }

    sceneEncoder.endEncoding()
    return commandBuffer
  }

  private func makeRenderPassDescriptor(
    for drawable: LayerRenderer.Drawable,
    viewCount: Int,
    clearColor: MTLClearColor
  ) -> MTLRenderPassDescriptor? {
    guard let colorTexture = drawable.colorTextures.first else { return nil }
    let descriptor = MTLRenderPassDescriptor()

    descriptor.colorAttachments[0].texture = colorTexture
    descriptor.colorAttachments[0].loadAction = .clear
    descriptor.colorAttachments[0].clearColor = clearColor
    descriptor.colorAttachments[0].storeAction = .store

    if let depthTexture = drawable.depthTextures.first {
      descriptor.depthAttachment.texture = depthTexture
      descriptor.depthAttachment.loadAction = .clear
      descriptor.depthAttachment.storeAction = .store
      descriptor.depthAttachment.clearDepth = 0.0
    }

    descriptor.rasterizationRateMap = drawable.rasterizationRateMaps.first

    // Always clear every slice in the drawable's texture, otherwise untouched
    // foveation tiles stay black.
    descriptor.renderTargetArrayLength = colorTexture.arrayLength

    return descriptor
  }

  private func resolvedViewCount(for drawable: LayerRenderer.Drawable) -> Int {
    let textureArrayLength = drawable.colorTextures.first?.arrayLength ?? 1
    let viewsCount = drawable.views.count
    let limitedByTextures = min(textureArrayLength, maxViewCount)
    let limitedByViews = min(viewsCount, maxViewCount)
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
    var matrices = Array(repeating: matrix_identity_float4x4, count: maxViewCount)
    var renderTargetLayers: [UInt32] = []
    var viewToWorldTransforms: [simd_float4x4] = []
    let desiredViewCount = max(min(viewCount, maxViewCount), 1)
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
        let textureMap = view.textureMap
        viewports.append(textureMap.viewport)
        renderTargetLayers.append(UInt32(textureMap.sliceIndex))
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
      renderTargetLayers.append(0)
      viewToWorldTransforms.append(rigTransform)
    }

    let fallbackMatrix = matrices[max(min(sampledViewCount - 1, maxViewCount - 1), 0)]
    let fallbackRenderLayer = renderTargetLayers.last ?? 0
    let fallbackTransform = viewToWorldTransforms.last ?? matrix_identity_float4x4
    if sampledViewCount < desiredViewCount {
      for index in sampledViewCount..<desiredViewCount {
        matrices[index] = fallbackMatrix
        renderTargetLayers.append(fallbackRenderLayer)
        viewToWorldTransforms.append(fallbackTransform)
      }
    }

    while viewports.count < desiredViewCount {
      viewports.append(viewports.last ?? viewports[0])
    }

    while renderTargetLayers.count < desiredViewCount {
      renderTargetLayers.append(renderTargetLayers.last ?? 0)
    }

    while viewToWorldTransforms.count < desiredViewCount {
      viewToWorldTransforms.append(viewToWorldTransforms.last ?? matrix_identity_float4x4)
    }

    return ViewRenderingData(
      viewProjectionMatrices: Array(matrices.prefix(desiredViewCount)),
      viewports: Array(viewports.prefix(desiredViewCount)),
      renderTargetLayers: Array(renderTargetLayers.prefix(desiredViewCount)),
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
    lorenzCount: Int,
    fourWingCount: Int,
    aizawaCount: Int,
    maxViewCount: Int
  ) -> [VisualPatternKind: VisualPatternController] {
    var controllers: [VisualPatternKind: VisualPatternController] = [:]
    do {
      controllers[.cubeField] = try CubeFieldRenderer(
        device: device, library: library, objectCount: cubeCount, maxViewCount: maxViewCount)
    } catch {
      fatalError("Failed to build cube pattern: \(error)")
    }

    if let pongWar = try? PongWarRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.pongWar] = pongWar
    } else {
      print("[Renderer] PongWar pattern unavailable.")
    }

    if let lorenz = try? LorenzRenderer(
      device: device,
      library: library,
      particleCount: lorenzCount,
      maxViewCount: maxViewCount
    ) {
      controllers[.lorenzAttractor] = lorenz
    } else {
      print("[Renderer] Lorenz attractor pattern unavailable; continuing with base pattern only.")
    }

    if let fourWing = try? FourWingRenderer(
      device: device,
      library: library,
      particleCount: fourWingCount,
      maxViewCount: maxViewCount
    ) {
      controllers[.fourWingAttractor] = fourWing
    } else {
      print("[Renderer] Four-Wing attractor pattern unavailable.")
    }

    if let aizawa = try? AizawaRenderer(
      device: device,
      library: library,
      particleCount: aizawaCount,
      maxViewCount: maxViewCount
    ) {
      controllers[.aizawaAttractor] = aizawa
    } else {
      print("[Renderer] Aizawa attractor pattern unavailable.")
    }

    if let pagoda = try? PagodaRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.pagoda] = pagoda
    } else {
      print("[Renderer] Pagoda pattern unavailable.")
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
