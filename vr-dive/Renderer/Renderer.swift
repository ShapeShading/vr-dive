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

    let layoutOptions: LayerRenderer.Capabilities.SupportedLayoutsOptions =
      supportsFoveation ? [.foveationEnabled] : []
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
  private typealias PatternControllerBuilder = () -> VisualPatternController?

  let layerRenderer: LayerRenderer
  let device: MTLDevice
  let library: MTLLibrary
  let commandQueue: MTLCommandQueue
  let arSession: ARKitSession
  let worldTracking: WorldTrackingProvider
  let gameManager: GameManager
  let patternCoordinator: PatternCoordinator
  private var patternControllers: [VisualPatternKind: VisualPatternController] = [:]
  private var deferredPatternBuilders: [VisualPatternKind: PatternControllerBuilder] = [:]
  private var activePatternKind: VisualPatternKind
  private let maxViewCount: Int
  private var lastKnownDeviceAnchor: ARKit.DeviceAnchor?
  private var rigTransform: simd_float4x4 = matrix_identity_float4x4
  private var lastRigUpdateTime: Float = 0
  private static let cubeObjectCount = 48
  private static let lorenzParticleCount = 400000
  private static let fourWingParticleCount = 400000
  private static let aizawaParticleCount = 400000
  private static let julia3DParticleCount = 400000
  private var didLogDrawableLayout = false

  var startTime: Date = Date()
  private static let attosecondsPerSecond = 1_000_000_000_000_000_000.0

  init(_ layerRenderer: LayerRenderer, patternCoordinator: PatternCoordinator) {
    self.layerRenderer = layerRenderer
    self.device = layerRenderer.device
    self.library = device.makeDefaultLibrary()!
    self.commandQueue = self.device.makeCommandQueue()!
    self.patternCoordinator = patternCoordinator
    self.maxViewCount = max(1, layerRenderer.properties.viewCount)

    let patternSetup = Renderer.makePatternControllers(
      device: device,
      library: self.library,
      cubeCount: Renderer.cubeObjectCount,
      lorenzCount: Renderer.lorenzParticleCount,
      fourWingCount: Renderer.fourWingParticleCount,
      aizawaCount: Renderer.aizawaParticleCount,
      julia3DCount: Renderer.julia3DParticleCount,
      maxViewCount: maxViewCount
    )
    var controllers = patternSetup.controllers
    self.deferredPatternBuilders = patternSetup.deferredBuilders

    self.arSession = ARKitSession()
    self.worldTracking = WorldTrackingProvider()
    self.gameManager = GameManager()

    // Add Tetris3D after gameManager is initialized
    Renderer.addTetris3D(
      to: &controllers,
      device: device,
      library: self.library,
      maxViewCount: maxViewCount,
      gameManager: self.gameManager
    )

    // Add Snake3D
    Renderer.addSnake3D(
      to: &controllers,
      device: device,
      library: self.library,
      maxViewCount: maxViewCount,
      gameManager: self.gameManager
    )

    self.patternControllers = controllers
    let requestedPattern = patternCoordinator.currentPattern()
    if controllers[requestedPattern] != nil || deferredPatternBuilders[requestedPattern] != nil {
      self.activePatternKind = requestedPattern
    } else if let fallback = controllers.keys.first {
      self.activePatternKind = fallback
      patternCoordinator.setPattern(fallback)
    } else if let deferredFallback = deferredPatternBuilders.keys.first {
      self.activePatternKind = deferredFallback
      patternCoordinator.setPattern(deferredFallback)
    } else {
      fatalError("No render patterns available")
    }
  }

  func startRenderLoop() {
    print("[Renderer] Starting render loop...")

    // Pre-warm GPU pipelines for all registered pattern controllers.
    // This triggers Metal's JIT shader compilation before the first rendered frame,
    // preventing compositor watchdog timeouts caused by slow first-frame compilation.
    warmupPipelines()

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

  /// Submits a minimal 1×1 offscreen render pass for every registered pattern controller
  /// so that Metal compiles their pipelines before the compositor needs the first real frame.
  private func warmupPipelines() {
    let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
    colorDesc.usage = [.renderTarget]
    colorDesc.storageMode = .private
    guard let colorTex = device.makeTexture(descriptor: colorDesc) else { return }

    let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .depth32Float, width: 1, height: 1, mipmapped: false)
    depthDesc.usage = [.renderTarget]
    depthDesc.storageMode = .private
    guard let depthTex = device.makeTexture(descriptor: depthDesc) else { return }

    let passDesc = MTLRenderPassDescriptor()
    passDesc.colorAttachments[0].texture = colorTex
    passDesc.colorAttachments[0].loadAction = .clear
    passDesc.colorAttachments[0].storeAction = .dontCare
    passDesc.depthAttachment.texture = depthTex
    passDesc.depthAttachment.loadAction = .clear
    passDesc.depthAttachment.clearDepth = 0.0
    passDesc.depthAttachment.storeAction = .dontCare

    guard let cmdBuf = commandQueue.makeCommandBuffer(),
      let enc = cmdBuf.makeRenderCommandEncoder(descriptor: passDesc)
    else { return }
    enc.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()
    print("[Renderer] Pipeline warmup complete")
  }

  func renderLoop() {
    print("[Renderer] Render loop started, layerRenderer state: \(layerRenderer.state)")
    var frameCount = 0
    var didLogFirstFrame = false
    var lastMissingAnchorLogTime: CFTimeInterval = 0
    while true {
      if layerRenderer.state == .invalidated {
        print("[Renderer] Layer renderer invalidated, exiting")
        break
      }

      // Also check for paused state during transition
      if layerRenderer.state == .paused {
        Thread.sleep(forTimeInterval: 0.05)
        continue
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

      if !didLogFirstFrame {
        print("[Renderer] First frame rendering...")
        didLogFirstFrame = true
      } else if frameCount > 0 && frameCount % 60 == 0 {
        print("[Renderer] Frame \(frameCount) rendered")
      }

      guard let frame = layerRenderer.queryNextFrame() else {
        // Check state again after failed query
        if layerRenderer.state != .running {
          continue
        }
        Thread.sleep(forTimeInterval: 0.001)
        continue
      }

      // Double-check state after getting frame but before processing
      // This catches the transition that happens between queryNextFrame and startUpdate
      guard layerRenderer.state == .running else {
        continue
      }

      var shouldSkipFrame = false

      autoreleasepool {
        var shouldEndUpdate = false
        var didStartSubmission = false

        // Final state check before any frame operations
        guard layerRenderer.state == .running else {
          shouldSkipFrame = true
          return
        }

        frame.startUpdate()
        shouldEndUpdate = true

        // Check state immediately after startUpdate - if transitioning, end gracefully
        guard layerRenderer.state == .running else {
          if shouldEndUpdate {
            frame.endUpdate()
          }
          shouldSkipFrame = true
          return
        }

        let animationTime = Float(Date().timeIntervalSince(startTime))
        let predictedTiming = frame.predictTiming()
        guard predictedTiming != nil else {
          // Frame is no longer valid; do not call endUpdate on an invalid frame.
          shouldEndUpdate = false
          shouldSkipFrame = true
          return
        }
        let presentationTimestamp = presentationTimeInterval(from: predictedTiming)
        let drawables = frame.queryDrawables()

        guard !drawables.isEmpty else {
          // queryDrawables can invalidate the frame during immersive dismissal.
          shouldEndUpdate = false
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
        } else if CFAbsoluteTimeGetCurrent() - lastMissingAnchorLogTime >= 1.0 {
          lastMissingAnchorLogTime = CFAbsoluteTimeGetCurrent()
          print("[Renderer] Waiting for reliable world tracking data...")
        }

        let pattern = resolveActivePatternController()
        let isPaused = patternCoordinator.isPaused()
        let simulationContext = PatternSimulationContext(
          commandQueue: commandQueue,
          time: animationTime,
          speedMultiplier: patternCoordinator.speedMultiplier(),
          isPaused: isPaused,
          originCellInspectionEnabled: patternCoordinator.originCellInspectionEnabled(),
          rayMarchingProbeDimTarget: patternCoordinator.rayMarchingProbeDimTarget(),
          huashanSampleRatio: patternCoordinator.huashanSampleRatio(),
          simoneOrbit3DPreset: patternCoordinator.simoneOrbit3DPreset()
        )
        pattern?.synchronizeState(simulationContext)

        if patternCoordinator.shouldReset() {
          pattern?.resetToInitialState()
          patternCoordinator.clearResetFlag()
        }

        if let activePattern = pattern, !isPaused {
          activePattern.updateSimulation(simulationContext)
        }

        guard let validAnchor = anchorToUse else {
          if shouldEndUpdate {
            frame.endUpdate()
            shouldEndUpdate = false
          }
          completeEmptySubmissionIfPossible(for: frame, drawables: drawables)
          shouldSkipFrame = true
          return
        }

        var pendingCommands: [(LayerRenderer.Drawable, MTLCommandBuffer)] = []
        for drawable in drawables {
          if let commandBuffer = encodeDrawable(
            drawable: drawable,
            pattern: pattern,
            deviceAnchor: validAnchor,
            time: animationTime
          ) {
            pendingCommands.append((drawable, commandBuffer))
          }
        }

        if shouldEndUpdate {
          frame.endUpdate()
          shouldEndUpdate = false
        }

        guard !pendingCommands.isEmpty else {
          completeEmptySubmissionIfPossible(for: frame, drawables: drawables)
          shouldSkipFrame = true
          return
        }

        // Check state before submission - if not running, skip submission entirely
        guard layerRenderer.state == .running else {
          shouldSkipFrame = true
          return
        }

        frame.startSubmission()
        didStartSubmission = true

        // Check state after startSubmission - if transitioning, end submission gracefully
        guard layerRenderer.state == .running else {
          if didStartSubmission {
            frame.endSubmission()
          }
          shouldSkipFrame = true
          return
        }

        for (drawable, commandBuffer) in pendingCommands {
          drawable.deviceAnchor = validAnchor
          drawable.encodePresent(commandBuffer: commandBuffer)
          commandBuffer.commit()
        }

        if didStartSubmission {
          frame.endSubmission()
        }
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

  private func completeEmptySubmissionIfPossible(
    for frame: LayerRenderer.Frame,
    drawables: [LayerRenderer.Drawable] = []
  ) {
    guard layerRenderer.state == .running else { return }
    frame.startSubmission()
    for drawable in drawables {
      guard let cmdBuf = commandQueue.makeCommandBuffer() else { continue }
      drawable.encodePresent(commandBuffer: cmdBuf)
      cmdBuf.commit()
    }
    frame.endSubmission()
  }

  private func resolveActivePatternController() -> VisualPatternController? {
    let desiredPattern = patternCoordinator.currentPattern()
    if desiredPattern != activePatternKind {
      if let controller = patternControllers[desiredPattern]
        ?? loadDeferredPatternController(for: desiredPattern)
      {
        activePatternKind = desiredPattern
        print("[Renderer] Switching to pattern: \(desiredPattern.rawValue)")
        return controller
      }
    }

    if let controller = patternControllers[activePatternKind]
      ?? loadDeferredPatternController(for: activePatternKind)
    {
      return controller
    }

    if let fallback = patternControllers.values.first {
      activePatternKind = fallback.identifier
      return fallback
    }

    if let deferredFallback = deferredPatternBuilders.keys.first,
      let controller = loadDeferredPatternController(for: deferredFallback)
    {
      activePatternKind = deferredFallback
      return controller
    }

    return nil
  }

  private func loadDeferredPatternController(for kind: VisualPatternKind)
    -> VisualPatternController?
  {
    if let existing = patternControllers[kind] {
      return existing
    }

    guard let builder = deferredPatternBuilders.removeValue(forKey: kind),
      let controller = builder()
    else {
      return nil
    }

    patternControllers[kind] = controller
    return controller
  }

  private func encodeDrawable(
    drawable: LayerRenderer.Drawable,
    pattern: VisualPatternController?,
    deviceAnchor: ARKit.DeviceAnchor?,
    time: Float
  ) -> MTLCommandBuffer? {
    let anchorToUse = deviceAnchor ?? lastKnownDeviceAnchor
    guard anchorToUse != nil else { return nil }
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

    // ⚠️ clearColor 必须是纯黑，不得改为读取 pattern?.preferredClearColor。
    // foveation 的 rasterizationRateMap 将渲染目标分成 tile，未被几何体覆盖的
    // tile 会被 clear 到此颜色。非黑色 clearColor 会造成可见的彩色瓦片伪影。
    // 见 notes/05-08-tile-artifacts-and-stereo-bugs.md
    let clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

    // ── Compute pre-pass (before render encoder is created) ──────────────────
    if let activePattern = pattern, let anchor = anchorToUse {
      let viewData = makeViewRenderingData(
        drawable: drawable,
        deviceAnchor: anchor,
        colorTexture: colorTexture,
        viewCount: viewCount
      )
      let prepassContext = PatternRenderContext(
        viewData: viewData,
        time: time,
        renderTargetWidth: colorTexture.width,
        renderTargetHeight: colorTexture.height
      )
      activePattern.encodeComputePrepass(commandBuffer: commandBuffer, context: prepassContext)
    }

    guard
      let descriptor = makeRenderPassDescriptor(
        for: drawable,
        viewCount: viewCount,
        clearColor: clearColor
      ),
      let sceneEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)
    else { return nil }

    if let activePattern = pattern, let anchor = anchorToUse {
      let viewData = makeViewRenderingData(
        drawable: drawable,
        deviceAnchor: anchor,
        colorTexture: colorTexture,
        viewCount: viewCount
      )
      let sceneContext = PatternRenderContext(
        viewData: viewData,
        time: time,
        renderTargetWidth: colorTexture.width,
        renderTargetHeight: colorTexture.height
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
    // Safety-first gate: some single-view fallback layouts on visionOS simulator
    // still assert if setVertexAmplificationCount(1, nil) is called, even though
    // Metal exposes the capability query. For now only enable amplification on
    // actual multi-view draws where the device explicitly supports the count.
    let supportsVertexAmplification =
      desiredViewCount > 1
      && device.supportsVertexAmplificationCount(desiredViewCount)
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
        // ⚠️ 必须用 sliceIndex，不是 textureIndex。
        // layered layout 下两眼在同一 texture 的不同 array slice（左眼 slice=0，右眼 slice=1）。
        // textureIndex 对两眼都是 0，会导致两眼画面叠到左眼，右眼黑屏。
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
      viewCount: desiredViewCount,
      supportsVertexAmplification: supportsVertexAmplification
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
    julia3DCount: Int,
    maxViewCount: Int
  ) -> (
    controllers: [VisualPatternKind: VisualPatternController],
    deferredBuilders: [VisualPatternKind: PatternControllerBuilder]
  ) {
    var controllers: [VisualPatternKind: VisualPatternController] = [:]
    var deferredBuilders: [VisualPatternKind: PatternControllerBuilder] = [:]
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

    let polychoronConfigs: [(VisualPatternKind, RegularPolychoronKind, String)] = [
      (.fiveCellProjection, .fiveCell, "5-cell"),
      (.eightCellProjection, .eightCell, "8-cell"),
      (.sixteenCellProjection, .sixteenCell, "16-cell"),
      (.twentyFourCellProjection, .twentyFourCell, "24-cell"),
      (.oneHundredTwentyCellProjection, .oneHundredTwentyCell, "120-cell"),
      (.sixHundredCellProjection, .sixHundredCell, "600-cell"),
    ]

    for (patternKind, polychoronKind, label) in polychoronConfigs {
      if let renderer = try? StereographicRenderer(
        device: device,
        library: library,
        patternKind: patternKind,
        polychoronKind: polychoronKind,
        maxViewCount: maxViewCount
      ) {
        controllers[patternKind] = renderer
      } else {
        print("[Renderer] \(label) projection pattern unavailable.")
      }
    }

    deferredBuilders[.lorenzAttractor] = {
      if let lorenz = try? LorenzRenderer(
        device: device,
        library: library,
        particleCount: lorenzCount,
        maxViewCount: maxViewCount
      ) {
        return lorenz
      }
      print("[Renderer] Lorenz attractor pattern unavailable; continuing with base pattern only.")
      return nil
    }

    deferredBuilders[.fourWingAttractor] = {
      if let fourWing = try? FourWingRenderer(
        device: device,
        library: library,
        particleCount: fourWingCount,
        maxViewCount: maxViewCount
      ) {
        return fourWing
      }
      print("[Renderer] Four-Wing attractor pattern unavailable.")
      return nil
    }

    deferredBuilders[.aizawaAttractor] = {
      if let aizawa = try? AizawaRenderer(
        device: device,
        library: library,
        particleCount: aizawaCount,
        maxViewCount: maxViewCount
      ) {
        return aizawa
      }
      print("[Renderer] Aizawa attractor pattern unavailable.")
      return nil
    }

    if let pagoda = try? PagodaSolidRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.pagoda] = pagoda
    } else {
      print("[Renderer] Pagoda pattern unavailable.")
    }

    if let julia3D = try? Julia3DRenderer(
      device: device,
      library: library,
      particleCount: julia3DCount,
      maxViewCount: maxViewCount
    ) {
      controllers[.julia3D] = julia3D
    } else {
      print("[Renderer] Julia3D pattern unavailable.")
    }

    if let rhombic = try? RhombicDodecahedronRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.rhombicDodecahedron] = rhombic
    } else {
      print("[Renderer] RhombicDodecahedron pattern unavailable.")
    }

    if let metaball = try? MetaballRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.metaball] = metaball
    } else {
      print("[Renderer] Metaball pattern unavailable.")
    }

    if let quatPoly = try? QuatPolynomialRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.quatPolynomial] = quatPoly
    } else {
      print("[Renderer] QuatPolynomial pattern unavailable.")
    }

    deferredBuilders[.huashan] = {
      if let huashan = try? HuashanSplatRenderer(
        device: device,
        library: library,
        maxViewCount: maxViewCount
      ) {
        return huashan
      }
      print("[Renderer] Huashan 3DGS pattern unavailable (missing huashan.splat?).")
      return nil
    }

    if let glassBox = try? GlassBoxRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.glassBox] = glassBox
    } else {
      print("[Renderer] GlassBox pattern unavailable.")
    }

    if let platonicMirror = try? PlatonicMirrorRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.platonicMirror] = platonicMirror
    } else {
      print("[Renderer] PlatonicMirror pattern unavailable.")
    }

    if let synthwaveSunset = try? SynthwaveSunsetRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.synthwaveSunset] = synthwaveSunset
    } else {
      print("[Renderer] SynthwaveSunset pattern unavailable.")
    }

    if let tunnel = try? TunnelRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.tunnel] = tunnel
    } else {
      print("[Renderer] Tunnel pattern unavailable.")
    }

    if let cubicSpaceDivision = try? CubicSpaceDivisionRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.cubicSpaceDivision] = cubicSpaceDivision
    } else {
      print("[Renderer] CubicSpaceDivision pattern unavailable.")
    }

    if let voxelEdges = try? VoxelEdgesRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.voxelEdges] = voxelEdges
    } else {
      print("[Renderer] VoxelEdges pattern unavailable.")
    }

    if let pathTilesCube = try? PathTilesCubeRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.pathTilesCube] = pathTilesCube
    } else {
      print("[Renderer] PathTilesCube pattern unavailable.")
    }

    if let cartoonFractalCube = try? CartoonFractalCubeRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.cartoonFractalCube] = cartoonFractalCube
    } else {
      print("[Renderer] CartoonFractalCube pattern unavailable.")
    }

    if let gyroidEchoCube = try? GyroidEchoCubeRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.gyroidEchoCube] = gyroidEchoCube
    } else {
      print("[Renderer] GyroidEchoCube pattern unavailable.")
    }

    if let waveLatticeCube = try? WaveLatticeCubeRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.waveLatticeCube] = waveLatticeCube
    } else {
      print("[Renderer] WaveLatticeCube pattern unavailable.")
    }

    if let waveySpheres = try? WaveySpheresRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.waveySpheres] = waveySpheres
    } else {
      print("[Renderer] Wavey spheres pattern unavailable.")
    }

    if let fractalFlythrough = try? FractalFlythroughRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.fractalFlythrough] = fractalFlythrough
    } else {
      print("[Renderer] Fractal Flythrough pattern unavailable.")
    }

    if let apollonianIIv4 = try? ApollonianIIv4Renderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.apollonianIIv4] = apollonianIIv4
    } else {
      print("[Renderer] Apollonian-II-v4 pattern unavailable.")
    }

    if let magnetar = try? MagnetarRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.magnetar] = magnetar
    } else {
      print("[Renderer] Magnetar pattern unavailable.")
    }

    if let spiraledLayers = try? SpiraledLayersRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.spiraledLayers] = spiraledLayers
    } else {
      print("[Renderer] Spiraled Layers pattern unavailable.")
    }

    if let angleFire = try? AngleFireRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.angleFire] = angleFire
    } else {
      print("[Renderer] Angle Fire pattern unavailable.")
    }

    if let glowingMountainLines = try? GlowingMountainLinesRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.glowingMountainLines] = glowingMountainLines
    } else {
      print("[Renderer] Glowing Mountain Lines pattern unavailable.")
    }

    if let rayMarchingDemo = try? RayMarchingDemoRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.rayMarchingDemo] = rayMarchingDemo
      print("[Renderer] RayMarchingDemo pattern added.")
    } else {
      print("[Renderer] Ray Marching Demo pattern unavailable.")
    }

    if let cubeRayMarchDemo = try? CubeRayMarchDemoRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.cubeRayMarchDemo] = cubeRayMarchDemo
      print("[Renderer] CubeRayMarchDemo pattern added.")
    } else {
      print("[Renderer] Cube Ray March Demo pattern unavailable.")
    }

    if let hyperbolicGroupLimitSet = try? HyperbolicGroupLimitSetRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.hyperbolicGroupLimitSet] = hyperbolicGroupLimitSet
      print("[Renderer] HyperbolicGroupLimitSet pattern added.")
    } else {
      print("[Renderer] Hyperbolic Group Limit Set pattern unavailable.")
    }

    if let apolloSpiral = try? ApolloSpiralRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.apolloSpiral] = apolloSpiral
      print("[Renderer] ApolloSpiral pattern added.")
    } else {
      print("[Renderer] Apollo Spiral pattern unavailable.")
    }

    if let voxelTunnel = try? VoxelTunnelRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.voxelTunnel] = voxelTunnel
      print("[Renderer] VoxelTunnel pattern added.")
    } else {
      print("[Renderer] Voxel tunnel pattern unavailable.")
    }

    if let nearLoxodrome = try? NearLoxodromeRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.nearLoxodrome] = nearLoxodrome
      print("[Renderer] NearLoxodrome pattern added.")
    } else {
      print("[Renderer] Near Loxodrome pattern unavailable.")
    }

    if let shield = try? ShieldRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.shield] = shield
      print("[Renderer] Shield pattern added.")
    } else {
      print("[Renderer] Shield pattern unavailable.")
    }

    if let digitalLines = try? DigitalLinesRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.digitalLines] = digitalLines
      print("[Renderer] Digital Lines pattern added.")
    } else {
      print("[Renderer] Digital Lines pattern unavailable.")
    }

    if let cloudyCrystal = try? CloudyCrystalRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.cloudyCrystal] = cloudyCrystal
      print("[Renderer] Cloudy Crystal pattern added.")
    } else {
      print("[Renderer] Cloudy Crystal pattern unavailable.")
    }

    if let shaderdoughFairy = try? ShaderdoughFairyRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.shaderdoughFairy] = shaderdoughFairy
      print("[Renderer] Shaderdough Fairy pattern added.")
    } else {
      print("[Renderer] Shaderdough Fairy pattern unavailable.")
    }

    if let crystalCubeLatticinioCore1 = try? CrystalCubeLatticinioCore1Renderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.crystalCubeLatticinioCore1] = crystalCubeLatticinioCore1
      print("[Renderer] Crystal Cube Latticinio core 1 pattern added.")
    } else {
      print("[Renderer] Crystal Cube Latticinio core 1 pattern unavailable.")
    }

    if let fireTornado = try? FireTornadoRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.fireTornado] = fireTornado
      print("[Renderer] Fire Tornado pattern added.")
    } else {
      print("[Renderer] Fire Tornado pattern unavailable.")
    }

    if let reflectiveWythoffPolyhedra = try? ReflectiveWythoffPolyhedraRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.reflectiveWythoffPolyhedra] = reflectiveWythoffPolyhedra
      print("[Renderer] Reflective Wythoff polyhedra pattern added.")
    } else {
      print("[Renderer] Reflective Wythoff polyhedra pattern unavailable.")
    }

    if let apollonian = try? ApollonianRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.apollonian] = apollonian
      print("[Renderer] apollonian pattern added.")
    } else {
      print("[Renderer] apollonian pattern unavailable.")
    }

    if let apollonianTwist = try? ApollonianTwistRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.apollonianTwist] = apollonianTwist
      print("[Renderer] Apollonian Twist pattern added.")
    } else {
      print("[Renderer] Apollonian Twist pattern unavailable.")
    }

    if let simoneOrbit3D = try? SimoneOrbit3DRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.simoneOrbit3D] = simoneOrbit3D
      print("[Renderer] Simone Orbit 3D pattern added.")
    } else {
      print("[Renderer] Simone Orbit 3D pattern unavailable.")
    }

    if let steampunkOrb = try? SteampunkOrbRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.steampunkOrb] = steampunkOrb
      print("[Renderer] Steampunk Orb pattern added.")
    } else {
      print("[Renderer] Steampunk Orb pattern unavailable.")
    }

    if let apollonianWires = try? ApollonianWiresRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.apollonianWires] = apollonianWires
      print("[Renderer] Apollonian Wires pattern added.")
    } else {
      print("[Renderer] Apollonian Wires pattern unavailable.")
    }

    if let kuKo = try? KuKoRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.kuKo] = kuKo
      print("[Renderer] KuKo pattern added.")
    } else {
      print("[Renderer] KuKo pattern unavailable.")
    }

    if let sonicAndTails = try? SonicAndTailsRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.sonicAndTails] = sonicAndTails
      print("[Renderer] Sonic & Tails pattern added.")
    } else {
      print("[Renderer] Sonic & Tails pattern unavailable.")
    }

    if let followYourLight = try? FollowYourLightRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.followYourLight] = followYourLight
      print("[Renderer] Follow Your Light pattern added.")
    } else {
      print("[Renderer] Follow Your Light pattern unavailable.")
    }

    if let weirdSurface = try? WeirdSurfaceRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.weirdSurface] = weirdSurface
      print("[Renderer] Weird Surface pattern added.")
    } else {
      print("[Renderer] Weird Surface pattern unavailable.")
    }

    if let neonShells = try? NeonShellsRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.neonShells] = neonShells
      print("[Renderer] Neon Shells pattern added.")
    } else {
      print("[Renderer] Neon Shells pattern unavailable.")
    }

    if let magneticLinesThatDrawInGold = try? MagneticLinesThatDrawInGoldRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.magneticLinesThatDrawInGold] = magneticLinesThatDrawInGold
      print("[Renderer] Magnetic lines that draw in gold pattern added.")
    } else {
      print("[Renderer] Magnetic lines that draw in gold pattern unavailable.")
    }

    if let lanterns = try? LanternsRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.lanterns] = lanterns
      print("[Renderer] Lanterns pattern added.")
    } else {
      print("[Renderer] Lanterns pattern unavailable.")
    }

    if let laceTunnel = try? LaceTunnelRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.laceTunnel] = laceTunnel
      print("[Renderer] Lace Tunnel pattern added.")
    } else {
      print("[Renderer] Lace Tunnel pattern unavailable.")
    }

    if let torusFan = try? TorusFanRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.torusFan] = torusFan
      print("[Renderer] Torus fan pattern added.")
    } else {
      print("[Renderer] Torus fan pattern unavailable.")
    }

    if let apollonianElevator = try? ApollonianElevatorRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.apollonianElevator] = apollonianElevator
      print("[Renderer] Apollonian Elevator pattern added.")
    } else {
      print("[Renderer] Apollonian Elevator pattern unavailable.")
    }

    if let torusKnotInR4 = try? TorusKnotInR4Renderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.torusKnotInR4] = torusKnotInR4
      print("[Renderer] Torus Knot in R4 pattern added.")
    } else {
      print("[Renderer] Torus Knot in R4 pattern unavailable.")
    }

    if let threeDFire = try? ThreeDFireRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.threeDFire] = threeDFire
      print("[Renderer] 3D Fire pattern added.")
    } else {
      print("[Renderer] 3D Fire pattern unavailable.")
    }

    if let bubbleRings = try? BubbleRingsRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.bubbleRings] = bubbleRings
      print("[Renderer] Bubble rings pattern added.")
    } else {
      print("[Renderer] Bubble rings pattern unavailable.")
    }

    if let ether = try? EtherRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.ether] = ether
      print("[Renderer] Ether pattern added.")
    } else {
      print("[Renderer] Ether pattern unavailable.")
    }

    if let fiberSpiral = try? FiberSpiralRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.fiberSpiral] = fiberSpiral
      print("[Renderer] Fiber Spiral pattern added.")
    } else {
      print("[Renderer] Fiber Spiral pattern unavailable.")
    }

    if let saturdayTorus = try? SaturdayTorusRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.saturdayTorus] = saturdayTorus
      print("[Renderer] Saturday Torus pattern added.")
    } else {
      print("[Renderer] Saturday Torus pattern unavailable.")
    }

    if let tesseractCornerFractal = try? TesseractCornerFractalRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.tesseractCornerFractal] = tesseractCornerFractal
      print("[Renderer] Tesseract Corner Fractal pattern added.")
    } else {
      print("[Renderer] Tesseract Corner Fractal pattern unavailable.")
    }

    if let hexwaves = try? HexwavesRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.hexwaves] = hexwaves
      print("[Renderer] hexwaves pattern added.")
    } else {
      print("[Renderer] hexwaves pattern unavailable.")
    }

    if let milosRose = try? MilosRoseRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.milosRose] = milosRose
      print("[Renderer] Milo's Rose pattern added.")
    } else {
      print("[Renderer] Milo's Rose pattern unavailable.")
    }

    if let recursiveLotus = try? RecursiveLotusRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.recursiveLotus] = recursiveLotus
      print("[Renderer] Recursive Lotus pattern added.")
    } else {
      print("[Renderer] Recursive Lotus pattern unavailable.")
    }

    if let blueFlower = try? BlueFlowerRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.blueFlower] = blueFlower
      print("[Renderer] Blue Flower pattern added.")
    } else {
      print("[Renderer] Blue Flower pattern unavailable.")
    }

    if let flowerTest = try? FlowerTestRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.flowerTest] = flowerTest
      print("[Renderer] Flower Test pattern added.")
    } else {
      print("[Renderer] Flower Test pattern unavailable.")
    }

    if let floreus = try? FloreusRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.floreus] = floreus
      print("[Renderer] Floreus pattern added.")
    } else {
      print("[Renderer] Floreus pattern unavailable.")
    }

    if let sineBud = try? SineBudRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.sineBud] = sineBud
      print("[Renderer] Sine bud pattern added.")
    } else {
      print("[Renderer] Sine bud pattern unavailable.")
    }

    if let saturdayWeirdness = try? SaturdayWeirdnessRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.saturdayWeirdness] = saturdayWeirdness
      print("[Renderer] Saturday weirdness pattern added.")
    } else {
      print("[Renderer] Saturday weirdness pattern unavailable.")
    }

    if let soulstone = try? SoulstoneRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.soulstone] = soulstone
      print("[Renderer] Soulstone pattern added.")
    } else {
      print("[Renderer] Soulstone pattern unavailable.")
    }

    if let boxOfStars = try? BoxOfStarsRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.boxOfStars] = boxOfStars
      print("[Renderer] Box of Stars pattern added.")
    } else {
      print("[Renderer] Box of Stars pattern unavailable.")
    }

    if let mirrorLooping = try? MirrorLoopingRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.mirrorLooping] = mirrorLooping
      print("[Renderer] Mirror Looping pattern added.")
    } else {
      print("[Renderer] Mirror Looping pattern unavailable.")
    }

    if let greatDodecaheadroll = try? GreatDodecaheadrollRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.greatDodecaheadroll] = greatDodecaheadroll
      print("[Renderer] Great Dodecaheadroll pattern added.")
    } else {
      print("[Renderer] Great Dodecaheadroll pattern unavailable.")
    }

    if let playingMarble = try? PlayingMarbleRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.playingMarble] = playingMarble
      print("[Renderer] Playing marble pattern added.")
    } else {
      print("[Renderer] Playing marble pattern unavailable.")
    }

    if let novaMarble = try? NovaMarbleRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.novaMarble] = novaMarble
      print("[Renderer] Nova Marble pattern added.")
    } else {
      print("[Renderer] Nova Marble pattern unavailable.")
    }

    if let dirtBall = try? DirtBallRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.dirtBall] = dirtBall
      print("[Renderer] Dirt Ball pattern added.")
    } else {
      print("[Renderer] Dirt Ball pattern unavailable.")
    }

    if let fractal49Gaz = try? Fractal49GazRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.fractal49Gaz] = fractal49Gaz
      print("[Renderer] Fractal 49_gaz pattern added.")
    } else {
      print("[Renderer] Fractal 49_gaz pattern unavailable.")
    }

    if let starryPlanes = try? StarryPlanesRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.starryPlanes] = starryPlanes
      print("[Renderer] Starry planes pattern added.")
    } else {
      print("[Renderer] Starry planes pattern unavailable.")
    }

    if let fractal77Gaz = try? Fractal77GazRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.fractal77Gaz] = fractal77Gaz
      print("[Renderer] Fractal 77_gaz pattern added.")
    } else {
      print("[Renderer] Fractal 77_gaz pattern unavailable.")
    }

    if let poincareBallHoneycomb = try? PoincareBallHoneycombRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.poincareBallHoneycomb] = poincareBallHoneycomb
      print("[Renderer] Poincare Ball Honeycomb pattern added.")
    } else {
      print("[Renderer] Poincare Ball Honeycomb pattern unavailable.")
    }

    if let goldenApollian = try? GoldenApollianRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.goldenApollian] = goldenApollian
      print("[Renderer] Golden apollian pattern added.")
    } else {
      print("[Renderer] Golden apollian pattern unavailable.")
    }

    if let anotherMarble = try? AnotherMarbleRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.anotherMarble] = anotherMarble
      print("[Renderer] Another Marble pattern added.")
    } else {
      print("[Renderer] Another Marble pattern unavailable.")
    }

    if let petalsFractal = try? PetalsFractalRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.petalsFractal] = petalsFractal
      print("[Renderer] Petals Fractal pattern added.")
    } else {
      print("[Renderer] Petals Fractal pattern unavailable.")
    }

    if let marbleMovingRemix = try? MarbleMovingRemixRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.marbleMovingRemix] = marbleMovingRemix
      print("[Renderer] marble moving remix pattern added.")
    } else {
      print("[Renderer] marble moving remix pattern unavailable.")
    }

    if let tunnelingThroughApollianFrac = try? TunnelingThroughApollianFracRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.tunnelingThroughApollianFrac] = tunnelingThroughApollianFrac
      print("[Renderer] Tunneling through apollian frac pattern added.")
    } else {
      print("[Renderer] Tunneling through apollian frac pattern unavailable.")
    }

    if let slicesInMarbles = try? SlicesInMarblesRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.slicesInMarbles] = slicesInMarbles
      print("[Renderer] slices in marbles pattern added.")
    } else {
      print("[Renderer] slices in marbles pattern unavailable.")
    }

    if let logSphericalKIFSZoomer = try? LogSphericalKIFSZoomerRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.logSphericalKIFSZoomer] = logSphericalKIFSZoomer
      print("[Renderer] Log Spherical KIFS Zoomer pattern added.")
    } else {
      print("[Renderer] Log Spherical KIFS Zoomer pattern unavailable.")
    }

    if let fractalCity = try? FractalCityRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.fractalCity] = fractalCity
      print("[Renderer] Fractal city pattern added.")
    } else {
      print("[Renderer] Fractal city pattern unavailable.")
    }

    if let interferenceCascadeCube = try? InterferenceCascadeCubeRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.interferenceCascadeCube] = interferenceCascadeCube
    } else {
      print("[Renderer] InterferenceCascadeCube pattern unavailable.")
    }

    if let orbitalSphereCube = try? OrbitalSphereCubeRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.orbitalSphereCube] = orbitalSphereCube
    } else {
      print("[Renderer] OrbitalSphereCube pattern unavailable.")
    }

    if let starTrails = try? StarTrailsRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.starTrails] = starTrails
      print("[Renderer] StarTrails pattern added.")
    } else {
      print("[Renderer] StarTrails pattern unavailable.")
    }

    if let particleRain = try? ParticleRainRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount
    ) {
      controllers[.particleRain] = particleRain
      print("[Renderer] ParticleRain pattern added.")
    } else {
      print("[Renderer] ParticleRain pattern unavailable.")
    }

    return (controllers: controllers, deferredBuilders: deferredBuilders)
  }

  private static func addTetris3D(
    to controllers: inout [VisualPatternKind: VisualPatternController],
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int,
    gameManager: GameManager
  ) {
    let tetris = Tetris3DRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount,
      gameManager: gameManager
    )
    controllers[.tetris3D] = tetris
    print("[Renderer] Tetris3D pattern added.")
  }

  private static func addSnake3D(
    to controllers: inout [VisualPatternKind: VisualPatternController],
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int,
    gameManager: GameManager
  ) {
    let snake = Snake3DRenderer(
      device: device,
      library: library,
      maxViewCount: maxViewCount,
      gameManager: gameManager
    )
    controllers[.snake3D] = snake
    print("[Renderer] Snake3D pattern added.")
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
