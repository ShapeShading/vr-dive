import Foundation
import Metal
import simd

final class HuashanSplatRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .huashan
  let preferredClearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1)

  // MARK: - GPU resources
  private let splatBuffer: MTLBuffer  // SplatPoint × N  (read-only)
  private let sortedIndexBuffers: [MTLBuffer]  // triple-buffered uint32 × renderCount
  private let renderPipelineState: MTLRenderPipelineState
  private let computePipelineState: MTLComputePipelineState
  private let depthStencilState: MTLDepthStencilState
  private let precompBuffer: MTLBuffer  // SplatPrecomp × renderCount (written by compute, read by vertex)
  private let splatCount: Int
  private let renderCount: Int
  private let renderStride: UInt32
  private let maxViewCount: Int

  // MARK: - Sort state
  private var currentSortBuf = 0  // which buffer the GPU is currently reading
  private var gpuSortBuf = 0  // which buffer most recently filled by CPU sort
  private var sortAvailable = false  // true once first sort completes
  private let sortQueue = DispatchQueue(label: "vr-dive.huashan.sort", qos: .userInitiated)
  private var sortInFlight = false
  private var lastSortedCameraPos: SIMD3<Float> = SIMD3(repeating: .greatestFiniteMagnitude)
  private var lastSortedCameraFwd: SIMD3<Float> = SIMD3(0, 0, -1)
  private var lastSortTime: CFTimeInterval = 0
  // Camera state for sort (updated each frame from render thread, read on sort thread)
  private var sortCameraPos: SIMD3<Float> = .zero
  private var sortCameraFwd: SIMD3<Float> = SIMD3(0, 0, -1)
  private let sortLock = NSLock()
  // Sampled positions/indices extracted once at load time for fast sort
  private let sampledSplatPositions: [SIMD3<Float>]
  private let sampledSplatIndices: [UInt32]

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

    let recordStride = MemoryLayout<HuashanSplatPoint>.stride  // 32 bytes
    let count = data.count / recordStride
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
        length: count * recordStride,
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

    let splatStride: UInt32 = 2  // aggressive half-density background sampling
    self.renderStride = splatStride

    // Extract a mixed-density sample set for CPU sorting.
    // Keep a center-front ROI at full density and the rest at half density.
    var sampledPositions = [SIMD3<Float>]()
    var sampledIndices = [UInt32]()
    let rawPtr = buf.contents().assumingMemoryBound(to: HuashanSplatPoint.self)
    let dataCenterX: Float = -0.15462685
    let dataCenterY: Float = 0.09616375
    let dataCenterZ: Float = 0.07551384
    let sceneScale: Float = 0.26
    let sceneTranslateZ: Float = -1.25
    let denseCenterWorldZ: Float = -1.6
    let denseHalfWidthX: Float = 3.4
    let denseHalfWidthY: Float = 3.5
    for i in 0..<count {
      let px = rawPtr[i].px
      let py = rawPtr[i].py
      let pz = rawPtr[i].pz
      let worldX = (px - dataCenterX) * sceneScale
      let worldY = (py - dataCenterY) * sceneScale
      let worldZ = (pz - dataCenterZ) * sceneScale + sceneTranslateZ
      let inDenseROI =
        abs(worldX) < denseHalfWidthX && abs(worldY) < denseHalfWidthY
        && worldZ > denseCenterWorldZ
      if inDenseROI || i % Int(splatStride) == 0 {
        sampledPositions.append(SIMD3<Float>(px, py, pz))
        sampledIndices.append(UInt32(i))
      }
    }
    self.renderCount = sampledIndices.count
    self.sampledSplatPositions = sampledPositions
    self.sampledSplatIndices = sampledIndices

    // Precomp buffer: one SplatPrecomp per rendered sampled splat
    let precompSize = renderCount * MemoryLayout<HuashanSplatPrecomp>.stride
    guard let pcBuf = device.makeBuffer(length: precompSize, options: .storageModePrivate) else {
      throw NSError(
        domain: "Huashan", code: 6,
        userInfo: [NSLocalizedDescriptionKey: "Failed to allocate precomp buffer"])
    }
    self.precompBuffer = pcBuf

    // Triple-buffered sort index buffers
    let idxSize = renderCount * MemoryLayout<UInt32>.stride
    var sortBufs = [MTLBuffer]()
    for _ in 0..<3 {
      guard let b = device.makeBuffer(length: idxSize, options: .storageModeShared) else {
        throw NSError(
          domain: "Huashan", code: 4,
          userInfo: [NSLocalizedDescriptionKey: "Failed to allocate sort buffer"])
      }
      // Default: sequential sampled order (no sort yet)
      let ptr = b.contents().assumingMemoryBound(to: UInt32.self)
      for i in 0..<renderCount { ptr[i] = sampledIndices[i] }
      sortBufs.append(b)
    }
    self.sortedIndexBuffers = sortBufs

    // Place the cloud directly in front of the user in local immersive space.
    // Bias a bit closer/larger so local structure occupies more retinal area.
    let dataCenter = SIMD3<Float>(-0.15462685, 0.09616375, 0.07551384)
    let centerShift = simd_float4x4(
      columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(-dataCenter.x, -dataCenter.y, -dataCenter.z, 1)
      ))
    let scale: Float = 0.26
    let scaleMatrix = simd_float4x4(
      columns: (
        SIMD4<Float>(scale, 0, 0, 0),
        SIMD4<Float>(0, scale, 0, 0),
        SIMD4<Float>(0, 0, scale, 0),
        SIMD4<Float>(0, 0, 0, 1)
      ))
    let placeInFront = simd_float4x4(
      columns: (
        SIMD4<Float>(1, 0, 0, 0),
        SIMD4<Float>(0, 1, 0, 0),
        SIMD4<Float>(0, 0, 1, 0),
        SIMD4<Float>(0.0, -0.03, -1.25, 1)
      ))
    sceneTransform = placeInFront * scaleMatrix * centerShift
    sceneInverse = simd_inverse(sceneTransform)

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
    let sortBuf = sortedIndexBuffers[sortAvailable ? gpuSortBuf : 0]
    encoder.setComputePipelineState(computePipelineState)
    encoder.setBuffer(splatBuffer, offset: 0, index: 0)
    encoder.setBytes(&uniforms, length: MemoryLayout<HuashanUniforms>.stride, index: 1)
    encoder.setBuffer(precompBuffer, offset: 0, index: 2)
    encoder.setBuffer(sortBuf, offset: 0, index: 3)
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
        "[Huashan] scene-origin ndc=\(ndc)  splatCount=\(splatCount) renderCount=\(renderCount) stride=\(renderStride)"
      )
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
    let now = CFAbsoluteTimeGetCurrent()
    let movement = simd_length(cameraPos - lastSortedCameraPos)
    let forwardDot = simd_dot(cameraForward, lastSortedCameraFwd)
    let needsFirstSort = !sortAvailable
    let needsResort = movement > 0.26 || forwardDot < 0.98
    if !needsFirstSort && (!needsResort || now - lastSortTime < 0.40) {
      return
    }

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

    let positions = self.sampledSplatPositions
    let sampledIndices = self.sampledSplatIndices
    let n = self.renderCount

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

      // Build sampled splat index array sorted by descending depth (far → near = back-to-front)
      var order = [Int](0..<n)
      order.sort { depths[$0] > depths[$1] }
      var indices = [UInt32](repeating: 0, count: n)
      for i in 0..<n {
        indices[i] = sampledIndices[order[i]]
      }

      // Copy to GPU buffer
      indices.withUnsafeBytes { rawBytes in
        memcpy(targetBuf.contents(), rawBytes.baseAddress!, n * MemoryLayout<UInt32>.stride)
      }

      // Atomically publish the new buffer
      self.sortLock.lock()
      self.gpuSortBuf = writeBuf
      self.sortAvailable = true
      self.sortInFlight = false
      self.lastSortedCameraPos = pos
      self.lastSortedCameraFwd = fwd
      self.lastSortTime = now
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
