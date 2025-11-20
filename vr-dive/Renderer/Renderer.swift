import ARKit
import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI

// Must match Metal struct
struct Uniforms {
  var viewMatrix: (simd_float4x4, simd_float4x4)
  var projectionMatrix: (simd_float4x4, simd_float4x4)
  var time: Float
  var cameraPosition: simd_float3
  var projectileData:
    (
      simd_float4, simd_float4, simd_float4, simd_float4, simd_float4, simd_float4, simd_float4,
      simd_float4, simd_float4, simd_float4
    )  // Array of 10 float4
  var projectileCount: Int32
  var padding: (Int32, Int32, Int32)  // Padding to match alignment if needed, usually 16 bytes alignment
}

struct VRConfiguration: CompositorLayerConfiguration {
  func makeConfiguration(
    capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration
  ) {
    let supportsFoveation = capabilities.supportsFoveation
    configuration.isFoveationEnabled = supportsFoveation

    let supportedLayouts = capabilities.supportedLayouts(
      options: supportsFoveation ? [.foveationEnabled] : [])
    configuration.layout = supportedLayouts.contains(.dedicated) ? .dedicated : .shared

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

  var gameManager = GameManager()
  var startTime: Date = Date()

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
    pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float
    pipelineDescriptor.inputPrimitiveTopology = .triangle

    self.pipelineState = try! device.makeRenderPipelineState(descriptor: pipelineDescriptor)

    self.arSession = ARKitSession()
    self.worldTracking = WorldTrackingProvider()
  }

  func startRenderLoop() {
    Task {
      try? await arSession.run([worldTracking])

      let renderThread = Thread {
        self.renderLoop()
      }
      renderThread.name = "Render Thread"
      renderThread.start()
    }
  }

  func renderLoop() {
    while true {
      if layerRenderer.state == .invalidated { break }
      guard layerRenderer.state == .running else {
        Thread.sleep(forTimeInterval: 0.01)
        continue
      }

      guard let frame = layerRenderer.queryNextFrame() else { continue }

      frame.startUpdate()

      // Update Game Logic
      let time = Float(Date().timeIntervalSince(startTime))

      // Get Head Pose
      // We need to predict where the head will be at display time
      // let timing = frame.predictTimeline()
      // let displayTime = timing.presentationTime
      let displayTime = CACurrentMediaTime()

      // Get the device anchor
      // Note: In a real app we use the timestamp. For simplicity we query latest.
      // But CompositorServices requires us to use the frame's time for correct prediction.

      // Actually, we should use the `worldTracking.queryDeviceAnchor(atTimestamp: ...)`
      // But `queryDeviceAnchor` returns an optional DeviceAnchor.

      let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: displayTime)
      let headTransform = deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4

      gameManager.update(time: time, deltaTime: 0.016, headTransform: headTransform)

      frame.endUpdate()

      guard let drawable = frame.queryDrawables().first else { continue }

      let commandBuffer = commandQueue.makeCommandBuffer()!
      let renderPassDescriptor = MTLRenderPassDescriptor()
      renderPassDescriptor.colorAttachments[0].texture = drawable.colorTextures[0]
      renderPassDescriptor.colorAttachments[0].loadAction = .clear
      renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(
        red: 0, green: 0, blue: 0, alpha: 0)
      renderPassDescriptor.colorAttachments[0].storeAction = .store
      renderPassDescriptor.renderTargetArrayLength = 2  // Explicitly set array length for stereo

      if let depthTexture = drawable.depthTextures.first {
        renderPassDescriptor.depthAttachment.texture = depthTexture
        renderPassDescriptor.depthAttachment.loadAction = .clear
        renderPassDescriptor.depthAttachment.storeAction = .store
        renderPassDescriptor.depthAttachment.clearDepth = 0.0
      }

      let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
      encoder.setRenderPipelineState(pipelineState)

      // Setup Uniforms
      // We need to handle stereo rendering (2 views)
      // CompositorLayer usually gives us 2 views in `drawable.views`

      let views = drawable.views
      // Ensure we have at least 1 view, usually 2 for VR
      let view0 = views.count > 0 ? views[0] : nil
      let view1 = views.count > 1 ? views[1] : view0

      guard let v0 = view0, let v1 = view1 else { continue }

      var uniforms = Uniforms(
        viewMatrix: (v0.transform.inverse, v1.transform.inverse),
        projectionMatrix: (makeProjectionMatrix(view: v0), makeProjectionMatrix(view: v1)),
        time: time,
        cameraPosition: simd_make_float3(headTransform.columns.3) + gameManager.cameraPosition,  // Add virtual offset
        projectileData: (
          simd_float4(), simd_float4(), simd_float4(), simd_float4(), simd_float4(), simd_float4(),
          simd_float4(), simd_float4(), simd_float4(), simd_float4()
        ),
        projectileCount: Int32(gameManager.projectiles.count),
        padding: (0, 0, 0)
      )

      // Fill projectiles
      // This is ugly manual filling because Swift tuples aren't arrays.

      // In a real app, use a pointer to buffer.
      // I'll just fill the first few manually or use `withUnsafeMutableBytes`.

      var projArray = [simd_float4](repeating: simd_float4(0, 0, 0, 0), count: 10)
      for i in 0..<min(10, gameManager.projectiles.count) {
        let p = gameManager.projectiles[i]
        projArray[i] = simd_float4(p.position.x, p.position.y, p.position.z, p.active ? 1.0 : 0.0)
      }

      // Copy to tuple (hacky)
      uniforms.projectileData = (
        projArray[0], projArray[1], projArray[2], projArray[3], projArray[4], projArray[5],
        projArray[6], projArray[7], projArray[8], projArray[9]
      )

      // Let's use `encoder.setVertexBytes` to pass uniforms.
      encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
      encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)

      // Draw 3 vertices, 2 instances (one per eye)
      encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3, instanceCount: 2)

      encoder.endEncoding()

      drawable.encodePresent(commandBuffer: commandBuffer)
      commandBuffer.commit()
    }
  }

  func makeProjectionMatrix(view: LayerRenderer.Drawable.View) -> float4x4 {
    let tangents = view.tangents
    // Assuming tangents order: x=left, y=right, z=top, w=bottom
    let left = tangents.x
    let right = tangents.y
    let top = tangents.z
    let bottom = tangents.w

    let nearDepth: Float = 0.05
    let farDepth: Float = 100.0

    let xScale = 2.0 / (right - left)
    let yScale = 2.0 / (top - bottom)
    let xOffset = -(right + left) * xScale * 0.5
    let yOffset = -(top + bottom) * yScale * 0.5

    let zScale = farDepth / (nearDepth - farDepth)
    let zOffset = (farDepth * nearDepth) / (nearDepth - farDepth)

    return float4x4(
      [xScale, 0, 0, 0],
      [0, yScale, 0, 0],
      [xOffset, yOffset, zScale, -1],
      [0, 0, zOffset, 0]
    )
  }
}
