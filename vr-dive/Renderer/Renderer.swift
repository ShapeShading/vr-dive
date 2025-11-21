import ARKit
import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI

// Simple uniforms struct matching the Metal shader
struct Uniforms {
  var time: Float
  var padding: SIMD3<Float>
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
  let pipelineState: MTLRenderPipelineState
  let arSession: ARKitSession
  let worldTracking: WorldTrackingProvider

  var startTime: Date = Date()
  private static let attosecondsPerSecond = 1_000_000_000_000_000_000.0

  init(_ layerRenderer: LayerRenderer) {
    self.layerRenderer = layerRenderer
    self.device = layerRenderer.device
    self.commandQueue = self.device.makeCommandQueue()!

    let library = device.makeDefaultLibrary()!
    let vertexFunction = library.makeFunction(name: "vertexShader")
    let fragmentFunction = library.makeFunction(name: "fragmentShader")

    let pipelineDescriptor = MTLRenderPipelineDescriptor()
    pipelineDescriptor.vertexFunction = vertexFunction
    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
    pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
    pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
    pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
    pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
    pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .zero
    pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
    pipelineDescriptor.inputPrimitiveTopology = .triangle

    self.pipelineState = try! device.makeRenderPipelineState(descriptor: pipelineDescriptor)

    self.arSession = ARKitSession()
    self.worldTracking = WorldTrackingProvider()
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

      autoreleasepool {
        frame.startUpdate()

        let animationTime = Float(Date().timeIntervalSince(startTime))
        let predictedTiming = frame.predictTiming()
        let presentationTimestamp = presentationTimeInterval(from: predictedTiming)

        frame.endUpdate()

        let drawables = frame.queryDrawables()
        guard !drawables.isEmpty else { return }

        var deviceAnchor: ARKit.DeviceAnchor?
        if worldTracking.state == .running {
          deviceAnchor = worldTracking.queryDeviceAnchor(
            atTimestamp: presentationTimestamp ?? CACurrentMediaTime()
          )
        }

        if deviceAnchor == nil && frameCount % 120 == 0 {
          print("[Renderer] Waiting for reliable world tracking data...")
        }

        frame.startSubmission()
        defer { frame.endSubmission() }

        for drawable in drawables {
          if let anchor = deviceAnchor {
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
    guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

    drawable.deviceAnchor = deviceAnchor

    if let renderPassDescriptor = makeRenderPassDescriptor(for: drawable),
      let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
    {
      encoder.setRenderPipelineState(pipelineState)

      var uniforms = Uniforms(time: time, padding: SIMD3<Float>(repeating: 0))
      encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
      encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
      encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3, instanceCount: 2)
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

  private func makeRenderPassDescriptor(for drawable: LayerRenderer.Drawable)
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
    descriptor.renderTargetArrayLength = colorTexture.arrayLength

    if let depthTexture = drawable.depthTextures.first {
      let depthAttachment = descriptor.depthAttachment ?? MTLRenderPassDepthAttachmentDescriptor()
      descriptor.depthAttachment = depthAttachment
      depthAttachment.texture = depthTexture
      depthAttachment.loadAction = .clear
      depthAttachment.storeAction = .store
      depthAttachment.clearDepth = 1.0
    }

    return descriptor
  }
}
