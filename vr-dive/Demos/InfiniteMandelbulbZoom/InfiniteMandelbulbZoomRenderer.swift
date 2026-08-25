import Foundation
import Metal
import simd

/// An endlessly rebased zoom into a deliberately self-similar Mandelbulb hierarchy.
///
/// `zoomPhase` never grows beyond one. At a layer boundary the renderer increments a
/// wrapping generation counter and returns the GPU to canonical local coordinates.
/// Consequently shader arithmetic retains the same precision at layer one and at a
/// conceptually unlimited depth.
final class InfiniteMandelbulbZoomRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .infiniteMandelbulbZoom
  let preferredClearColor = MTLClearColor(red: 0.002, green: 0.004, blue: 0.012, alpha: 1)

  private static let raymarchWidth = 256
  private static let raymarchHeight = 200
  private static let raymarchUpdateInterval: UInt32 = 3
  private static let diagnosticBytesPerSample = 256

  private let pipelineState: MTLRenderPipelineState
  private let computePipelineState: MTLComputePipelineState
  private let depthStencilState: MTLDepthStencilState
  private let raymarchTexture: MTLTexture
  private let fragmentCoverageBuffer: MTLBuffer
  private let maxViewCount: Int

  private var zoomPhase: Float = 0
  private var generation: UInt32 = 0
  private var lastSimulationTime: Float?
  private var latestContext: PatternSimulationContext?
  private var didLogRaymarchConfiguration = false
  private var didLogCompositeConfiguration = false
  private var hasRaymarchContent = false
  private var prepassFrameIndex: UInt32 = 0
  private var computeDispatchCount: UInt64 = 0
  private var postpassFrameCount: UInt64 = 0

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)
    pipelineState = try Self.makePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    computePipelineState = try device.makeComputePipelineState(
      function: library.makeFunction(name: "infiniteMandelbulbZoomCompute")!)
    depthStencilState = Self.makeDepthStencilState(device: device)

    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba16Float,
      width: Self.raymarchWidth,
      height: Self.raymarchHeight,
      mipmapped: false)
    textureDescriptor.textureType = .type2DArray
    textureDescriptor.arrayLength = self.maxViewCount
    textureDescriptor.storageMode = .private
    textureDescriptor.usage = [.shaderRead, .shaderWrite]
    guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
      throw InfiniteMandelbulbZoomRendererError.textureAllocationFailed
    }
    texture.label = "Infinite Mandelbulb Zoom Raymarch"
    raymarchTexture = texture

    guard
      let coverageBuffer = device.makeBuffer(
        length: MemoryLayout<UInt32>.stride,
        options: .storageModeShared)
    else {
      throw InfiniteMandelbulbZoomRendererError.diagnosticBufferAllocationFailed
    }
    coverageBuffer.label = "Infinite Mandelbulb Zoom Fragment Coverage"
    coverageBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    fragmentCoverageBuffer = coverageBuffer
  }

  func synchronizeState(_ context: PatternSimulationContext) {
    latestContext = context
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    defer { lastSimulationTime = context.time }
    guard let lastSimulationTime else { return }

    let deltaTime = max(0, min(context.time - lastSimulationTime, 1.0 / 20.0))
    let delta = deltaTime * context.infiniteZoomRate * max(context.speedMultiplier, 0)
      * context.infiniteZoomDirection
    zoomPhase += delta

    while zoomPhase >= 1 {
      zoomPhase -= 1
      generation &+= 1
    }
    while zoomPhase < 0 {
      zoomPhase += 1
      generation &-= 1
    }
  }

  func resetToInitialState() {
    zoomPhase = 0
    generation = 0
    lastSimulationTime = nil
    hasRaymarchContent = false
  }

  func encodeComputePrepass(commandBuffer: MTLCommandBuffer, context: PatternRenderContext) {
    let viewCount = min(max(context.viewData.viewCount, 1), maxViewCount)
    let shouldUpdate = !hasRaymarchContent
      || prepassFrameIndex.isMultiple(of: Self.raymarchUpdateInterval)
    prepassFrameIndex &+= 1
    guard shouldUpdate else { return }

    var uniforms = makeUniforms(
      viewCount: viewCount,
      depthRange: context.viewData.depthRange)

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
    encoder.label = "Infinite Mandelbulb Zoom Raymarch"
    encoder.setComputePipelineState(computePipelineState)
    encoder.setBytes(
      &uniforms, length: MemoryLayout<InfiniteMandelbulbZoomUniforms>.stride, index: 0)
    encoder.setTexture(raymarchTexture, index: 0)

    let threadWidth = computePipelineState.threadExecutionWidth
    let threadHeight = max(
      1, min(8, computePipelineState.maxTotalThreadsPerThreadgroup / threadWidth))
    encoder.dispatchThreads(
      MTLSize(width: Self.raymarchWidth, height: Self.raymarchHeight, depth: viewCount),
      threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1))
    encoder.endEncoding()
    hasRaymarchContent = true
    computeDispatchCount &+= 1

    let dispatchID = computeDispatchCount
    if dispatchID <= 3 || dispatchID.isMultiple(of: 120) {
      encodeDiagnostics(
        commandBuffer: commandBuffer,
        dispatchID: dispatchID,
        viewCount: viewCount,
        uniforms: uniforms)
    }

    if !didLogRaymarchConfiguration {
      didLogRaymarchConfiguration = true
      print(
        "[InfiniteMandelbulbZoom] Offscreen raymarch active: \(Self.raymarchWidth)x\(Self.raymarchHeight)x\(viewCount), updating every \(Self.raymarchUpdateInterval) frames, texture=\(raymarchTexture.pixelFormat.rawValue)/storage=\(raymarchTexture.storageMode.rawValue)/hazard=\(raymarchTexture.hazardTrackingMode.rawValue)"
      )
    }
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    let viewCount = min(max(context.viewData.viewCount, 1), maxViewCount)
    var uniforms = makeUniforms(
      viewCount: viewCount,
      depthRange: context.viewData.depthRange)

    context.applyViewConfiguration(on: encoder)
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)

    encoder.setVertexBytes(
      &uniforms, length: MemoryLayout<InfiniteMandelbulbZoomUniforms>.stride, index: 0)
    encoder.setFragmentTexture(raymarchTexture, index: 0)
    encoder.setFragmentBuffer(fragmentCoverageBuffer, offset: 0, index: 0)
    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

    if !didLogCompositeConfiguration {
      didLogCompositeConfiguration = true
      let viewportSummary = context.viewData.viewports.enumerated().map { index, viewport in
        "v\(index)=\(Int(viewport.width))x\(Int(viewport.height))"
      }.joined(separator: ",")
      print(
        "[InfiniteMandelbulbZoom] Composite encoded: views=\(viewCount), vertexAmplification=\(context.viewData.supportsVertexAmplification), layers=\(context.viewData.renderTargetLayers), viewports=[\(viewportSummary)], depthRange=\(context.viewData.depthRange), compositeDepth=\(String(format: "%.6f", uniforms.compositeDepth))"
      )
    }
  }

  func encodePostpass(
    commandBuffer: MTLCommandBuffer,
    context: PatternRenderContext,
    colorTexture: MTLTexture,
    depthTexture: MTLTexture?
  ) {
    postpassFrameCount &+= 1
    let frameID = postpassFrameCount
    guard frameID <= 3 || frameID.isMultiple(of: 360) else { return }

    let viewCount = min(
      max(context.viewData.viewCount, 1),
      max(min(colorTexture.arrayLength, maxViewCount), 1))
    let width = colorTexture.width
    let height = colorTexture.height
    let samplePoints: [(String, MTLOrigin)] = [
      ("C", MTLOrigin(x: width / 2, y: height / 2, z: 0)),
      ("L", MTLOrigin(x: width / 4, y: height / 2, z: 0)),
      ("R", MTLOrigin(x: width * 3 / 4, y: height / 2, z: 0)),
      ("B", MTLOrigin(x: width / 2, y: height / 4, z: 0)),
      ("T", MTLOrigin(x: width / 2, y: height * 3 / 4, z: 0)),
    ]
    let sampleCount = samplePoints.count * viewCount
    guard
      let colorReadbackBuffer = colorTexture.device.makeBuffer(
        length: sampleCount * Self.diagnosticBytesPerSample,
        options: .storageModeShared)
    else {
      print("[InfiniteMandelbulbZoom] Drawable #\(frameID) could not allocate readback")
      return
    }

    let depthReadbackBuffer = depthTexture.flatMap {
      $0.device.makeBuffer(
        length: sampleCount * Self.diagnosticBytesPerSample,
        options: .storageModeShared)
    }
    guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
      print("[InfiniteMandelbulbZoom] Drawable #\(frameID) could not create blit encoder")
      return
    }

    colorReadbackBuffer.label = "Infinite Mandelbulb Zoom Drawable Color #\(frameID)"
    depthReadbackBuffer?.label = "Infinite Mandelbulb Zoom Drawable Depth #\(frameID)"
    blitEncoder.label = "Infinite Mandelbulb Zoom Drawable Readback"
    for slice in 0..<viewCount {
      for (pointIndex, sample) in samplePoints.enumerated() {
        let offset = (slice * samplePoints.count + pointIndex)
          * Self.diagnosticBytesPerSample
        blitEncoder.copy(
          from: colorTexture,
          sourceSlice: slice,
          sourceLevel: 0,
          sourceOrigin: sample.1,
          sourceSize: MTLSize(width: 1, height: 1, depth: 1),
          to: colorReadbackBuffer,
          destinationOffset: offset,
          destinationBytesPerRow: Self.diagnosticBytesPerSample,
          destinationBytesPerImage: Self.diagnosticBytesPerSample)
        if let depthTexture, let depthReadbackBuffer {
          blitEncoder.copy(
            from: depthTexture,
            sourceSlice: slice,
            sourceLevel: 0,
            sourceOrigin: sample.1,
            sourceSize: MTLSize(width: 1, height: 1, depth: 1),
            to: depthReadbackBuffer,
            destinationOffset: offset,
            destinationBytesPerRow: Self.diagnosticBytesPerSample,
            destinationBytesPerImage: Self.diagnosticBytesPerSample)
        }
      }
    }
    blitEncoder.endEncoding()

    let textureSummary = "\(width)x\(height)x\(viewCount), format=\(colorTexture.pixelFormat.rawValue), storage=\(colorTexture.storageMode.rawValue)"
    commandBuffer.addCompletedHandler { [fragmentCoverageBuffer] buffer in
      let coverage = fragmentCoverageBuffer.contents().load(as: UInt32.self)
      let texelSummary = Self.makeTexelSummary(
        buffer: colorReadbackBuffer,
        viewCount: viewCount,
        sampleNames: samplePoints.map(\.0))
      print(
        "[InfiniteMandelbulbZoom] Drawable #\(frameID) completed: status=\(buffer.status.rawValue), fragmentCoverage=\(coverage), target=\(textureSummary)"
      )
      print("[InfiniteMandelbulbZoom] Drawable #\(frameID) texels: \(texelSummary)")
      if let depthReadbackBuffer {
        let depthSummary = Self.makeDepthSummary(
          buffer: depthReadbackBuffer,
          viewCount: viewCount,
          sampleNames: samplePoints.map(\.0))
        print("[InfiniteMandelbulbZoom] Drawable #\(frameID) depths: \(depthSummary)")
      } else {
        print("[InfiniteMandelbulbZoom] Drawable #\(frameID) depths: unavailable")
      }
    }
  }

  private func encodeDiagnostics(
    commandBuffer: MTLCommandBuffer,
    dispatchID: UInt64,
    viewCount: Int,
    uniforms: InfiniteMandelbulbZoomUniforms
  ) {
    let samplePoints: [(String, MTLOrigin)] = [
      ("C", MTLOrigin(x: Self.raymarchWidth / 2, y: Self.raymarchHeight / 2, z: 0)),
      ("L", MTLOrigin(x: Self.raymarchWidth / 4, y: Self.raymarchHeight / 2, z: 0)),
      ("R", MTLOrigin(x: Self.raymarchWidth * 3 / 4, y: Self.raymarchHeight / 2, z: 0)),
      ("B", MTLOrigin(x: Self.raymarchWidth / 2, y: Self.raymarchHeight / 4, z: 0)),
      ("T", MTLOrigin(x: Self.raymarchWidth / 2, y: Self.raymarchHeight * 3 / 4, z: 0)),
    ]
    let sampleCount = samplePoints.count * viewCount
    guard
      let readbackBuffer = raymarchTexture.device.makeBuffer(
        length: sampleCount * Self.diagnosticBytesPerSample,
        options: .storageModeShared),
      let blitEncoder = commandBuffer.makeBlitCommandEncoder()
    else {
      print("[InfiniteMandelbulbZoom] Diagnostic #\(dispatchID) could not allocate readback")
      return
    }

    readbackBuffer.label = "Infinite Mandelbulb Zoom Diagnostic #\(dispatchID)"
    blitEncoder.label = "Infinite Mandelbulb Zoom Diagnostic Readback"
    for slice in 0..<viewCount {
      for (pointIndex, sample) in samplePoints.enumerated() {
        let offset = (slice * samplePoints.count + pointIndex)
          * Self.diagnosticBytesPerSample
        blitEncoder.copy(
          from: raymarchTexture,
          sourceSlice: slice,
          sourceLevel: 0,
          sourceOrigin: sample.1,
          sourceSize: MTLSize(width: 1, height: 1, depth: 1),
          to: readbackBuffer,
          destinationOffset: offset,
          destinationBytesPerRow: Self.diagnosticBytesPerSample,
          destinationBytesPerImage: Self.diagnosticBytesPerSample)
      }
    }
    blitEncoder.endEncoding()

    let encodedAt = ProcessInfo.processInfo.systemUptime
    let qualitySummary = "steps=\(uniforms.maxRaySteps),iterations=\(uniforms.fractalIterations),phase=\(String(format: "%.4f", uniforms.zoomPhase))"
    commandBuffer.addScheduledHandler { buffer in
      let delayMilliseconds = (ProcessInfo.processInfo.systemUptime - encodedAt) * 1_000
      print(
        "[InfiniteMandelbulbZoom] Diagnostic #\(dispatchID) scheduled: status=\(buffer.status.rawValue), queueDelayMs=\(String(format: "%.2f", delayMilliseconds)), \(qualitySummary)"
      )
    }
    commandBuffer.addCompletedHandler { buffer in
      let wallMilliseconds = (ProcessInfo.processInfo.systemUptime - encodedAt) * 1_000
      let gpuMilliseconds = max(0, buffer.gpuEndTime - buffer.gpuStartTime) * 1_000
      let errorSummary = buffer.error.map {
        "\(($0 as NSError).domain)(\(($0 as NSError).code)): \($0.localizedDescription)"
      } ?? "none"
      let texelSummary = Self.makeTexelSummary(
        buffer: readbackBuffer,
        viewCount: viewCount,
        sampleNames: samplePoints.map(\.0))
      print(
        "[InfiniteMandelbulbZoom] Diagnostic #\(dispatchID) completed: status=\(buffer.status.rawValue), wallMs=\(String(format: "%.2f", wallMilliseconds)), gpuMs=\(String(format: "%.2f", gpuMilliseconds)), error=\(errorSummary)"
      )
      print("[InfiniteMandelbulbZoom] Diagnostic #\(dispatchID) texels: \(texelSummary)")
    }
  }

  private static func makeTexelSummary(
    buffer: MTLBuffer,
    viewCount: Int,
    sampleNames: [String]
  ) -> String {
    (0..<viewCount).map { slice in
      let samples = sampleNames.enumerated().map { sampleIndex, name in
        let offset = (slice * sampleNames.count + sampleIndex) * diagnosticBytesPerSample
        let values = buffer.contents().advanced(by: offset)
          .bindMemory(to: UInt16.self, capacity: 4)
        let rgba = (0..<4).map { component in
          Float(Float16(bitPattern: values[component]))
        }
        return "\(name)=(\(rgba.map { String(format: "%.4f", $0) }.joined(separator: ",")))"
      }.joined(separator: " ")
      return "eye\(slice){\(samples)}"
    }.joined(separator: "; ")
  }

  private static func makeDepthSummary(
    buffer: MTLBuffer,
    viewCount: Int,
    sampleNames: [String]
  ) -> String {
    (0..<viewCount).map { slice in
      let samples = sampleNames.enumerated().map { sampleIndex, name in
        let offset = (slice * sampleNames.count + sampleIndex) * diagnosticBytesPerSample
        let depth = buffer.contents().advanced(by: offset).load(as: Float.self)
        return "\(name)=\(String(format: "%.6f", depth))"
      }.joined(separator: " ")
      return "eye\(slice){\(samples)}"
    }.joined(separator: "; ")
  }

  private func makeUniforms(
    viewCount: Int,
    depthRange: SIMD2<Float>
  ) -> InfiniteMandelbulbZoomUniforms {
    let settings = latestContext
    let quality = settings?.infiniteZoomQuality ?? .balanced
    return InfiniteMandelbulbZoomUniforms(
      zoomPhase: zoomPhase,
      zoomDirection: settings?.infiniteZoomDirection ?? 1,
      viewCount: UInt32(viewCount),
      generation: generation,
      maxRaySteps: quality.raySteps,
      fractalIterations: quality.fractalIterations,
      surfaceEpsilon: quality == .detailed ? 0.0009 : 0.0013,
      compositeDepth: Self.reverseZDepth(
        distance: 2.75,
        far: depthRange.x,
        near: depthRange.y),
      cameraAndScale: SIMD4<Float>(0, 0, 2.75, 1.0))
  }

  /// Maps a view-space distance to Metal's [0, 1] reverse-Z depth range.
  /// Keeping the composite plane away from exact zero lets the device
  /// compositor reconstruct valid geometry for late-stage reprojection.
  private static func reverseZDepth(distance: Float, far: Float, near: Float) -> Float {
    guard distance.isFinite, far.isFinite, near.isFinite,
      distance > 0, far > near, near > 0
    else {
      return 0.02
    }
    let depth = near * (far / distance - 1) / (far - near)
    return min(max(depth, 0.001), 0.999)
  }

}

private enum InfiniteMandelbulbZoomRendererError: Error {
  case textureAllocationFailed
  case diagnosticBufferAllocationFailed
}

extension InfiniteMandelbulbZoomRenderer {
  fileprivate static func makePipelineState(
    device: MTLDevice, library: MTLLibrary, maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "infiniteMandelbulbZoomVertex")
    descriptor.fragmentFunction = library.makeFunction(name: "infiniteMandelbulbZoomFragment")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .triangle
    descriptor.maxVertexAmplificationCount = max(1, maxViewCount)
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    // The physical Vision Pro compositor uses this depth texture for late-stage
    // reprojection. Leaving the full-screen pass at the reverse-Z clear value
    // (zero/far plane) produces valid color texels that are discarded at
    // presentation. Always accept the pass and store its valid plane depth.
    descriptor.depthCompareFunction = .always
    descriptor.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: descriptor)!
  }
}
