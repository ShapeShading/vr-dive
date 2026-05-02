import Metal
import simd

final class HuashanSplatRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .huashan
  let preferredClearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1)

  // MARK: - GPU resources
  private let splatBuffer: MTLBuffer  // SplatPoint × N  (read-only)
  private let sortedIndexBuffers: [MTLBuffer]  // triple-buffered uint32 × N
  private let renderPipelineState: MTLRenderPipelineState
  private let computePipelineState: MTLComputePipelineState
  private let depthStencilState: MTLDepthStencilState
  private let precompBuffer: MTLBuffer  // SplatPrecomp × renderCount (written by compute, read by vertex)
  private let splatCount: Int
  private let renderCount: Int   // = splatCount / renderStride
  private let renderStride: UInt32
  private let maxViewCount: Int

  // MARK: - Sort state
  private var currentSortBuf = 0  // which buffer the GPU is currently reading
  private var gpuSortBuf = 0  // which buffer most recently filled by CPU sort
  private var sortAvailable = false  // true once first sort completes
  private let sortQueue = DispatchQueue(label: "vr-dive.huashan.sort", qos: .userInitiated)
  private var sortInFlight = false
  // Camera state for sort (updated each frame from render thread, read on sort thread)
  private var sortCameraPos: SIMD3<Float> = .zero
  private var sortCameraFwd: SIMD3<Float> = SIMD3(0, 0, -1)
  private let sortLock = NSLock()
  // Positions extracted once at load time for fast sort
  private let splatPositions: [SIMD3<Float>]

  // MARK: - Scene placement
  // Translates + scales the splat cloud so it appears in front of and below the user.
  // Adjust tx/ty/tz/s to tune initial viewing position.
  private let sceneTransform: simd_float4x4  // scene-space → world-space
  private let sceneInverse: simd_float4x4  // world-space → scene-space

  // MARK: - Init
  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    // Load .splat file from app bundle
    guard let url = Bundle.main.url(forResource: "huashan", withExtension: "splat") else {
      throw NSError(
        domain: "Huashan", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "huashan.splat not found in bundle"])
    }
    let data = try Data(contentsOf: url, options: .mappedIfSafe)

    let stride = MemoryLayout<HuashanSplatPoint>.stride  // 32 bytes
    let count = data.count / stride
    guard count > 0 else {
      throw NSError(
        domain: "Huashan", code: 2,
        userInfo: [NSLocalizedDescriptionKey: "huashan.splat is empty"])
    }
    self.splatCount = count

    // Upload splat data to GPU (shared storage for CPU access during sort)
    guard
      let buf = device.makeBuffer(
        bytes: (data as NSData).bytes,
        length: count * stride,
        options: .storageModeShared)
    else {
      throw NSError(
        domain: "Huashan", code: 3,
        userInfo: [NSLocalizedDescriptionKey: "Failed to allocate splat buffer"])
    }
    self.splatBuffer = buf

    // Layout diagnostic — helps detect Swift/Metal struct padding mismatch
    print(
      "[Huashan] PerEyeUniforms stride=\(MemoryLayout<HuashanPerEyeUniforms>.stride) HuashanUniforms stride=\(MemoryLayout<HuashanUniforms>.stride) (Metal expects PerEye=160, splatCount at offset 320)"
    )
    print("[Huashan] SplatPrecomp stride=\(MemoryLayout<HuashanSplatPrecomp>.stride) (expect 80)")

    // Precomp buffer: one SplatPrecomp per splat
    let precompSize = count * MemoryLayout<HuashanSplatPrecomp>.stride
    guard let pcBuf = device.makeBuffer(length: precompSize, options: .storageModePrivate) else {
      throw NSError(
        domain: "Huashan", code: 6,
        userInfo: [NSLocalizedDescriptionKey: "Failed to allocate precomp buffer"])
    }
    self.precompBuffer = pcBuf
    let splatStride: UInt32 = 16         // ~77k splats — proven stable
    self.renderStride = splatStride
    self.renderCount  = count / Int(splatStride)

    // Extract positions once for fast CPU sort
    var positions = [SIMD3<Float>](repeating: .zero, count: count)
    let rawPtr = buf.contents().assumingMemoryBound(to: HuashanSplatPoint.self)
    for i in 0..<count {
      positions[i] = SIMD3<Float>(rawPtr[i].px, rawPtr[i].py, rawPtr[i].pz)
    }
    self.splatPositions = positions

    // Triple-buffered sort index buffers
    let idxSize = count * MemoryLayout<UInt32>.stride
    var sortBufs = [MTLBuffer]()
    for _ in 0..<3 {
      guard let b = device.makeBuffer(length: idxSize, options: .storageModeShared) else {
        throw NSError(
          domain: "Huashan", code: 4,
          userInfo: [NSLocalizedDescriptionKey: "Failed to allocate sort buffer"])
      }
      // Default: sequential order (no sort yet)
      let ptr = b.contents().assumingMemoryBound(to: UInt32.self)
      for i in 0..<count { ptr[i] = UInt32(i) }
      sortBufs.append(b)
    }
    self.sortedIndexBuffers = sortBufs

    // Scene transform: scale 1.0 — positions ±10 scene = ±10m world
    // Camera at origin, mountain entirely ahead at -80..-100m.
    // At this distance with corrected focal (~694px RT), median splat (scale≈0.4)
    // has natural radius ≈ 9 physical px — shape visible, fill rate safe.
    let s: Float = 1.0
    let tx: Float = 0
    let ty: Float = 0
    let tz: Float = -90
    sceneTransform = simd_float4x4(
      columns: (
        SIMD4<Float>(s, 0, 0, 0),
        SIMD4<Float>(0, s, 0, 0),
        SIMD4<Float>(0, 0, s, 0),
        SIMD4<Float>(tx, ty, tz, 1)
      ))
    let si: Float = 1 / s
    sceneInverse = simd_float4x4(
      columns: (
        SIMD4<Float>(si, 0, 0, 0),
        SIMD4<Float>(0, si, 0, 0),
        SIMD4<Float>(0, 0, si, 0),
        SIMD4<Float>(-tx * si, -ty * si, -tz * si, 1)
      ))

    // Depth stencil: always pass (3DGS is sorted back-to-front; depth test would break blending)
    // Write depth at splat center so compositor detects rendered content.
    let dsd = MTLDepthStencilDescriptor()
    dsd.depthCompareFunction = .always
    dsd.isDepthWriteEnabled = true
    guard let dss = device.makeDepthStencilState(descriptor: dsd) else {
      throw NSError(
        domain: "Huashan", code: 5,
        userInfo: [NSLocalizedDescriptionKey: "Failed to create depth stencil"])
    }
    self.depthStencilState = dss

    // Render + compute pipelines
    let pipelines = try HuashanSplatRenderer.makePipelines(
      device: device, library: library, maxViewCount: self.maxViewCount)
    self.renderPipelineState = pipelines.render
    self.computePipelineState = pipelines.compute
  }

  // MARK: - VisualPatternController

  func resetToInitialState() {}

  func updateSimulation(_ context: PatternSimulationContext) {
    // Nothing to simulate — the scene is static, only sorting needed
  }

  // MARK: - encodeComputePrepass
  func encodeComputePrepass(commandBuffer: MTLCommandBuffer, context: PatternRenderContext) {
    guard splatCount > 0 else { return }
    var uniforms = makeUniforms(context: context)
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
    encoder.setComputePipelineState(computePipelineState)
    encoder.setBuffer(splatBuffer, offset: 0, index: 0)
    encoder.setBytes(&uniforms, length: MemoryLayout<HuashanUniforms>.stride, index: 1)
    encoder.setBuffer(precompBuffer, offset: 0, index: 2)
    var step = renderStride
    encoder.setBytes(&step, length: MemoryLayout<UInt32>.size, index: 3)
    let w = min(computePipelineState.maxTotalThreadsPerThreadgroup, 256)
    encoder.dispatchThreadgroups(
      MTLSize(width: (renderCount + w - 1) / w, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    encoder.endEncoding()
  }

  // MARK: - encodeFrame
  private var diagFrameCount = 0
  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    guard splatCount > 0 else { return }

    var uniforms = makeUniforms(context: context)

    // Camera for background sort
    let viewData = context.viewData
    let v2w0 = viewData.viewToWorldTransforms[0]
    let camPosWorld = SIMD3<Float>(v2w0.columns.3.x, v2w0.columns.3.y, v2w0.columns.3.z)
    let fwdWorld4 = v2w0 * SIMD4<Float>(0, 0, -1, 0)
    let camPos4 = sceneInverse * SIMD4<Float>(camPosWorld.x, camPosWorld.y, camPosWorld.z, 1)
    let camPos = SIMD3<Float>(camPos4.x, camPos4.y, camPos4.z)
    let camFwd4 = sceneInverse * SIMD4<Float>(fwdWorld4.x, fwdWorld4.y, fwdWorld4.z, 0)
    let camFwd = normalize(SIMD3<Float>(camFwd4.x, camFwd4.y, camFwd4.z))
    triggerSortIfNeeded(cameraPos: camPos, cameraForward: camFwd)

    diagFrameCount += 1
    if diagFrameCount == 2 {
      let centerClip = uniforms.eye0.vpMatrix * SIMD4<Float>(0, 0, 0, 1)
      let ndc = SIMD3<Float>(
        centerClip.x / centerClip.w, centerClip.y / centerClip.w, centerClip.z / centerClip.w)
      print(
        "[Huashan] scene-origin ndc=\(ndc)  splatCount=\(splatCount) renderCount=\(renderCount) stride=\(renderStride)")
    }

    encoder.setRenderPipelineState(renderPipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(precompBuffer, offset: 0, index: 0)
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<HuashanUniforms>.stride, index: 1)

    encoder.drawPrimitives(
      type: .triangle,
      vertexStart: 0,
      vertexCount: 6,
      instanceCount: renderCount)
  }

  // MARK: - Uniform builder
  private func makeUniforms(context: PatternRenderContext) -> HuashanUniforms {
    let viewData = context.viewData
    let eyeCount = min(viewData.viewCount, maxViewCount)
    func makeEye(_ i: Int) -> HuashanPerEyeUniforms {
      let idx = min(i, eyeCount - 1)
      let vp = viewData.viewProjectionMatrices[idx]
      let v2w = viewData.viewToWorldTransforms[idx]
      // Use renderTarget size for focal so r1 comes out in physical pixels
      let rtW = Float(context.renderTargetWidth)
      let rtH = Float(context.renderTargetHeight)
      let p = vp * v2w
      return HuashanPerEyeUniforms(
        vpMatrix: vp * sceneTransform,
        viewMatrix: v2w.inverse * sceneTransform,
        focalXY: SIMD2(abs(p[0][0] * rtW * 0.5), abs(p[1][1] * rtH * 0.5)),
        viewportSize: SIMD2(rtW, rtH),
        renderTargetSize: SIMD2(rtW, rtH)
      )
    }
    return HuashanUniforms(
      eye0: makeEye(0), eye1: makeEye(1),
      splatCount: UInt32(splatCount), viewCount: UInt32(eyeCount),
      splatScale: 1.0, _pad: 0)
  }

  // MARK: - Background sort
  private func triggerSortIfNeeded(cameraPos: SIMD3<Float>, cameraForward: SIMD3<Float>) {
    sortLock.lock()
    sortCameraPos = cameraPos
    sortCameraFwd = cameraForward
    let alreadyRunning = sortInFlight
    sortLock.unlock()

    guard !alreadyRunning else { return }

    sortLock.lock()
    sortInFlight = true
    let pos = sortCameraPos
    let fwd = sortCameraFwd
    sortLock.unlock()

    let positions = self.splatPositions
    let n = self.splatCount

    // Pick the buffer that neither the GPU nor previous sort is using
    let writeBuf = (gpuSortBuf + 1) % 3
    let targetBuf = self.sortedIndexBuffers[writeBuf]

    sortQueue.async { [weak self] in
      guard let self else { return }

      // Compute depths (signed distance along camera forward — larger = farther)
      var depths = [Float](repeating: 0, count: n)
      for i in 0..<n {
        depths[i] = dot(positions[i] - pos, fwd)
      }

      // Build index array sorted by descending depth (far → near = back-to-front)
      var indices = [UInt32](0..<UInt32(n))
      indices.sort { depths[Int($0)] > depths[Int($1)] }

      // Copy to GPU buffer
      indices.withUnsafeBytes { rawBytes in
        memcpy(targetBuf.contents(), rawBytes.baseAddress!, n * MemoryLayout<UInt32>.stride)
      }

      // Atomically publish the new buffer
      self.sortLock.lock()
      self.gpuSortBuf = writeBuf
      self.sortAvailable = true
      self.sortInFlight = false
      self.sortLock.unlock()
    }
  }
}

// MARK: - Pipeline factory
extension HuashanSplatRenderer {
  fileprivate static func makePipelines(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> (render: MTLRenderPipelineState, compute: MTLComputePipelineState) {
    // Render pipeline
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "huashanVertexShader")
    desc.fragmentFunction = library.makeFunction(name: "huashanFragmentShader")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.depthAttachmentPixelFormat = .depth32Float
    desc.inputPrimitiveTopology = .triangle

    let att = desc.colorAttachments[0]!
    att.isBlendingEnabled = true
    att.rgbBlendOperation = .add
    att.alphaBlendOperation = .add
    att.sourceRGBBlendFactor = .one
    att.sourceAlphaBlendFactor = .one
    att.destinationRGBBlendFactor = .oneMinusSourceAlpha
    att.destinationAlphaBlendFactor = .oneMinusSourceAlpha

    desc.maxVertexAmplificationCount = max(maxViewCount, 1)
    let renderPSO = try device.makeRenderPipelineState(descriptor: desc)

    // Compute pipeline
    guard let computeFn = library.makeFunction(name: "huashanComputePrecomp") else {
      throw NSError(
        domain: "Huashan", code: 7,
        userInfo: [NSLocalizedDescriptionKey: "huashanComputePrecomp not found in library"])
    }
    let computePSO = try device.makeComputePipelineState(function: computeFn)

    return (render: renderPSO, compute: computePSO)
  }
}
