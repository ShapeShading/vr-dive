import Metal
import simd

final class HuashanSplatRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .huashan
  let preferredClearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1)

  // MARK: - GPU resources
  private let splatBuffer: MTLBuffer      // SplatPoint × N  (read-only)
  private let sortedIndexBuffers: [MTLBuffer]  // triple-buffered uint32 × N
  private let renderPipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let splatCount: Int
  private let maxViewCount: Int

  // MARK: - Sort state
  private var currentSortBuf = 0          // which buffer the GPU is currently reading
  private var gpuSortBuf = 0             // which buffer most recently filled by CPU sort
  private var sortAvailable = false       // true once first sort completes
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
  private let sceneTransform: simd_float4x4   // scene-space → world-space
  private let sceneInverse: simd_float4x4     // world-space → scene-space

  // MARK: - Init
  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    // Load .splat file from app bundle
    guard let url = Bundle.main.url(forResource: "huashan", withExtension: "splat") else {
      throw NSError(domain: "Huashan", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "huashan.splat not found in bundle"])
    }
    let data = try Data(contentsOf: url, options: .mappedIfSafe)

    let stride = MemoryLayout<HuashanSplatPoint>.stride  // 32 bytes
    let count = data.count / stride
    guard count > 0 else {
      throw NSError(domain: "Huashan", code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "huashan.splat is empty"])
    }
    self.splatCount = count

    // Upload splat data to GPU (shared storage for CPU access during sort)
    guard let buf = device.makeBuffer(bytes: (data as NSData).bytes,
                                      length: count * stride,
                                      options: .storageModeShared) else {
      throw NSError(domain: "Huashan", code: 3,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to allocate splat buffer"])
    }
    self.splatBuffer = buf

    // Layout diagnostic — helps detect Swift/Metal struct padding mismatch
    print("[Huashan] PerEyeUniforms stride=\(MemoryLayout<HuashanPerEyeUniforms>.stride) HuashanUniforms stride=\(MemoryLayout<HuashanUniforms>.stride) (Metal expects PerEye=160, splatCount at offset 320)")

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
        throw NSError(domain: "Huashan", code: 4,
                      userInfo: [NSLocalizedDescriptionKey: "Failed to allocate sort buffer"])
      }
      // Default: sequential order (no sort yet)
      let ptr = b.contents().assumingMemoryBound(to: UInt32.self)
      for i in 0..<count { ptr[i] = UInt32(i) }
      sortBufs.append(b)
    }
    self.sortedIndexBuffers = sortBufs

    // Scene transform: scale 0.15 + place scene 8m ahead, near eye level
    // Raw splat positions span ≈ -10..+10. At scale 0.15 → ±1.5m world units.
    // ty=-1: scene centre slightly below eye level (mountain base near floor)
    let s: Float = 0.15
    let tx: Float = 0, ty: Float = -1, tz: Float = -8
    sceneTransform = simd_float4x4(columns: (
      SIMD4<Float>(s,  0,  0, 0),
      SIMD4<Float>(0,  s,  0, 0),
      SIMD4<Float>(0,  0,  s, 0),
      SIMD4<Float>(tx, ty, tz, 1)
    ))
    let si: Float = 1 / s
    sceneInverse = simd_float4x4(columns: (
      SIMD4<Float>(si,  0,   0,  0),
      SIMD4<Float>( 0, si,   0,  0),
      SIMD4<Float>( 0,  0,  si,  0),
      SIMD4<Float>(-tx * si, -ty * si, -tz * si, 1)
    ))

    // Depth stencil: always pass (3DGS is sorted back-to-front; depth test would break blending)
    // Write depth at splat center so compositor detects rendered content.
    let dsd = MTLDepthStencilDescriptor()
    dsd.depthCompareFunction = .always
    dsd.isDepthWriteEnabled = true
    guard let dss = device.makeDepthStencilState(descriptor: dsd) else {
      throw NSError(domain: "Huashan", code: 5,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to create depth stencil"])
    }
    self.depthStencilState = dss

    // Render pipeline
    self.renderPipelineState = try HuashanSplatRenderer.makeRenderPipeline(
      device: device, library: library, maxViewCount: self.maxViewCount)
  }

  // MARK: - VisualPatternController

  func resetToInitialState() {}

  func updateSimulation(_ context: PatternSimulationContext) {
    // Nothing to simulate — the scene is static, only sorting needed
  }

  // MARK: - encodeFrame
  private var diagFrameCount = 0
  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    guard splatCount > 0 else { return }

    let viewData = context.viewData
    let eyeCount = min(viewData.viewCount, maxViewCount)

    // ── Build uniforms ────────────────────────────────────────────────────────
    func makeEyeUniforms(_ i: Int) -> HuashanPerEyeUniforms {
      let idx = min(i, eyeCount - 1)
      let vp     = viewData.viewProjectionMatrices[idx]
      let v2w    = viewData.viewToWorldTransforms[idx]
      let viewMat = v2w.inverse

      // Focal lengths from projection matrix
      let vport  = viewData.viewports[idx]
      let w = Float(vport.width), h = Float(vport.height)
      let p = vp * v2w  // recover projection-only matrix
      let fx = p[0][0] * w * 0.5
      let fy = p[1][1] * h * 0.5

      // Actual render target texture size (not the larger display viewport)
      // Use a fixed physical resolution; the drawable logs show 1888×1792 on device.
      // On simulator it equals the viewport. We pass both to the shader.
      return HuashanPerEyeUniforms(
        vpMatrix: vp * sceneTransform,        // scene → clip
        viewMatrix: viewMat * sceneTransform, // scene → view
        focalXY: SIMD2(abs(fx), abs(fy)),
        viewportSize: SIMD2(w, h),
        renderTargetSize: SIMD2(
          Float(context.renderTargetWidth),
          Float(context.renderTargetHeight)
        )
      )
    }

    var uniforms = HuashanUniforms(
      eye0: makeEyeUniforms(0),
      eye1: makeEyeUniforms(1),
      splatCount: UInt32(splatCount),
      viewCount: UInt32(eyeCount),
      splatScale: 1.0,
      _pad: 0
    )

    // ── Update camera for background sort (in scene space) ───────────────────
    let v2w0 = viewData.viewToWorldTransforms[0]
    // Camera world position and forward direction — directly from viewToWorldTransform
    let camPosWorld = SIMD3<Float>(v2w0.columns.3.x, v2w0.columns.3.y, v2w0.columns.3.z)
    let fwdWorld4   = v2w0 * SIMD4<Float>(0, 0, -1, 0)
    // Transform to scene space (where splatPositions live)
    let camPos4 = sceneInverse * SIMD4<Float>(camPosWorld.x, camPosWorld.y, camPosWorld.z, 1)
    let camPos  = SIMD3<Float>(camPos4.x, camPos4.y, camPos4.z)
    let camFwd4 = sceneInverse * SIMD4<Float>(fwdWorld4.x, fwdWorld4.y, fwdWorld4.z, 0)
    let camFwd  = normalize(SIMD3<Float>(camFwd4.x, camFwd4.y, camFwd4.z))

    triggerSortIfNeeded(cameraPos: camPos, cameraForward: camFwd)

    // ── First-frame diagnostics ───────────────────────────────────────────────
    diagFrameCount += 1
    if diagFrameCount == 2 {  // frame 2: world tracking reliable
      let e0 = uniforms.eye0
      let vm = e0.viewMatrix
      // What does the camera-forward ray hit in NDC?
      let centerClip = e0.vpMatrix * SIMD4<Float>(0, 0, 0, 1)  // scene origin
      let centerNDC  = SIMD3<Float>(centerClip.x/centerClip.w, centerClip.y/centerClip.w, centerClip.z/centerClip.w)
      print("[Huashan] scene-origin clip=\(centerClip)  ndc=\(centerNDC)")
      let sampleStride = max(1, splatCount / 5)
      var depths = [Float]()
      for i in stride(from: 0, to: splatCount, by: sampleStride) {
        let p = splatPositions[i]
        let vz = vm.columns.0.z*p.x + vm.columns.1.z*p.y + vm.columns.2.z*p.z + vm.columns.3.z
        depths.append(vz)
      }
      let posInView  = depths.filter { $0 < 0 }.count
      let posInScene = depths.count
      print("[Huashan] splatCount=\(splatCount) renderCount=\(splatCount/8)")
      print("[Huashan] camPosWorld=\(camPosWorld)  camPosScene=\(camPos)")
      print("[Huashan] eyeCount=\(eyeCount)  viewport=\(uniforms.eye0.viewportSize)  focal=\(uniforms.eye0.focalXY)")
      print("[Huashan] sample depths (view-z, expect <0): \(depths.map { String(format:"%.2f",$0) })")
      print("[Huashan] splats in front of camera: \(posInView)/\(posInScene)")
      // Check a few splat positions raw
      let raw0 = splatPositions[0]
      let raw1 = splatPositions[splatCount/2]
      print("[Huashan] raw scene positions: splat[0]=\(raw0)  splat[mid]=\(raw1)")
      // VP-project first visible splat
      let vp = e0.vpMatrix
      let clip = vp * SIMD4<Float>(raw0.x, raw0.y, raw0.z, 1)
      print("[Huashan] splat[0] clip=\(clip)  ndc=(\(clip.x/clip.w), \(clip.y/clip.w), \(clip.z/clip.w))")
    }


    let sortBuf = sortedIndexBuffers[sortAvailable ? gpuSortBuf : 0]

    // ── Encode draw ───────────────────────────────────────────────────────────
    encoder.setRenderPipelineState(renderPipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)

    context.applyViewConfiguration(on: encoder)
    // NOTE: Do NOT override the viewport here. visionOS uses Variable Rate Rasterization
    // (rasterizationRateMap): the logical viewport (e.g. 4338×3478) is non-linearly
    // mapped to the physical texture (e.g. 1888×1792). setViewport must match the
    // drawable's textureMap.viewport exactly, which applyViewConfiguration already sets.

    encoder.setVertexBuffer(splatBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(sortBuf, offset: 0, index: 1)
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<HuashanUniforms>.stride, index: 2)

    // stride=16: ~77k splats, safe GPU budget (stride=4/309k crashes watchdog)
    let renderCount = max(1, splatCount / 16)
    print("[Huashan] DRAW instanceCount=\(renderCount) vertexCount=6")
    encoder.drawPrimitives(type: .triangle,
                            vertexStart: 0,
                            vertexCount: 6,
                            instanceCount: renderCount)
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
  fileprivate static func makeRenderPipeline(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction   = library.makeFunction(name: "huashanVertexShader")
    desc.fragmentFunction = library.makeFunction(name: "huashanFragmentShader")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.depthAttachmentPixelFormat = .depth32Float
    desc.inputPrimitiveTopology = .triangle

    // Alpha blending: pre-multiplied alpha (src=one, dst=oneMinusSourceAlpha)
    // Required for correct 3DGS Gaussian compositing.
    let att = desc.colorAttachments[0]!
    att.isBlendingEnabled             = true
    att.rgbBlendOperation             = .add
    att.alphaBlendOperation           = .add
    att.sourceRGBBlendFactor          = .one
    att.sourceAlphaBlendFactor        = .one
    att.destinationRGBBlendFactor     = .oneMinusSourceAlpha
    att.destinationAlphaBlendFactor   = .oneMinusSourceAlpha

    desc.maxVertexAmplificationCount = max(maxViewCount, 1)
    return try device.makeRenderPipelineState(descriptor: desc)
  }
}
