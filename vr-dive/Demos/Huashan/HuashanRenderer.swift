import Foundation
import Metal
import simd

private struct HuashanPackedPosition {
  var x: Float
  var y: Float
  var z: Float

  var simd: SIMD3<Float> { SIMD3<Float>(x, y, z) }
}

private struct HuashanChunkKey: Hashable {
  var x: Int32
  var y: Int32
  var z: Int32
}

private struct HuashanSplatChunk {
  var startIndex: Int
  var count: Int
  var center: SIMD3<Float>
  var radius: Float
}

final class HuashanSplatRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .huashan
  let preferredClearColor = MTLClearColor(red: 0.05, green: 0.05, blue: 0.1, alpha: 1)
  private static let maxSampledSplatCount = 550_000
  private static let maxVisibleSplatCount = 400_000
  private static let warmStartVisibleSplatCount = 120_000
  private static let minPerChunkQuota = 96
  private static let refillBatchSize = 32
  private static let minVisibleCosine: Float = 0.35
  private static let minVisibleCosine2: Float = minVisibleCosine * minVisibleCosine
  private static let maxVisibleDistanceScene: Float = 3.6 / 0.26
  private static let maxVisibleDistanceScene2: Float =
    maxVisibleDistanceScene * maxVisibleDistanceScene
  private static let chunkCellSizeScene: Float = 0.22 / 0.26

  // MARK: - GPU resources
  private let splatBuffer: MTLBuffer  // SplatPoint × N  (read-only)
  private let sortedIndexBuffers: [MTLBuffer]  // triple-buffered uint32 × visibleCapacity
  private let renderPipelineState: MTLRenderPipelineState
  private let computePipelineState: MTLComputePipelineState
  private let depthStencilState: MTLDepthStencilState
  private let precompBuffer: MTLBuffer  // SplatPrecomp × visibleCapacity (written by compute, read by vertex)
  private let splatCount: Int
  private var renderCount: Int
  private let visibleCapacity: Int
  private var renderStride: UInt32
  private let maxViewCount: Int
  private var currentSampleRatio: Float = PatternCoordinator.defaultHuashanSampleRatio

  // MARK: - Sort state
  private var currentSortBuf = 0  // which buffer the GPU is currently reading
  private var gpuSortBuf = 0  // which buffer most recently filled by CPU sort
  private var sortAvailable = false  // true once first sort completes
  private let sortQueue = DispatchQueue(label: "vr-dive.huashan.sort", qos: .userInitiated)
  private var sortInFlight = false
  private var lastSortedCameraPos: SIMD3<Float> = SIMD3(repeating: .greatestFiniteMagnitude)
  private var lastSortedCameraFwd: SIMD3<Float> = SIMD3(0, 0, -1)
  private var lastSortTime: CFTimeInterval = 0
  private var sortedVisibleCounts: [Int] = []
  private var sortedVisibleChunkCounts: [Int] = []
  private var sortedBudgetHits: [Bool] = []
  private var sortedSortDurationsMS: [Float] = []
  private var lastDiagnosticsLogTime: CFTimeInterval = 0
  // Camera state for sort (updated each frame from render thread, read on sort thread)
  private var sortCameraPos: SIMD3<Float> = .zero
  private var sortCameraFwd: SIMD3<Float> = SIMD3(0, 0, -1)
  private let sortLock = NSLock()
  // Sampled positions/indices extracted once at load time for fast sort
  private var sampledSplatPositions: [HuashanPackedPosition]
  private var sampledSplatIndices: [UInt32]
  private var sampleChunks: [HuashanSplatChunk]
  private var sortDepths: [Float]
  private var sortOrder: [Int]
  private var sortScratchIndices: [UInt32]
  private var chunkDepths: [Float]
  private var chunkOrder: [Int]
  private var chunkVisibleCounts: [Int]
  private var chunkVisibleOffsets: [Int]

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
    print("[Huashan] SplatPrecomp stride=\(MemoryLayout<HuashanSplatPrecomp>.stride) (expect 64)")

    let initialSamples = Self.buildSampleSet(
      rawPtr: buf.contents().assumingMemoryBound(to: HuashanSplatPoint.self),
      splatCount: count,
      sampleRatio: currentSampleRatio)
    self.renderCount = initialSamples.positions.count
    self.visibleCapacity = HuashanSplatRenderer.maxVisibleSplatCount
    self.sampledSplatPositions = initialSamples.positions
    self.sampledSplatIndices = initialSamples.indices
    self.sampleChunks = initialSamples.chunks
    self.renderStride = initialSamples.renderStride
    self.sortDepths = [Float](repeating: 0, count: renderCount)
    self.sortOrder = Array(0..<renderCount)
    self.sortScratchIndices = [UInt32](repeating: 0, count: visibleCapacity)
    self.chunkDepths = [Float](repeating: 0, count: initialSamples.chunks.count)
    self.chunkOrder = Array(0..<initialSamples.chunks.count)
    self.chunkVisibleCounts = [Int](repeating: 0, count: initialSamples.chunks.count)
    self.chunkVisibleOffsets = [Int](repeating: 0, count: initialSamples.chunks.count)
    let initialVisibleCount = min(
      renderCount,
      visibleCapacity,
      HuashanSplatRenderer.warmStartVisibleSplatCount)
    self.sortedVisibleCounts = [initialVisibleCount, initialVisibleCount, initialVisibleCount]
    self.sortedVisibleChunkCounts = [
      initialSamples.chunks.count, initialSamples.chunks.count, initialSamples.chunks.count,
    ]
    self.sortedBudgetHits = [false, false, false]
    self.sortedSortDurationsMS = [0, 0, 0]

    // Precomp buffer: one SplatPrecomp per rendered sampled splat
    let precompSize = visibleCapacity * MemoryLayout<HuashanSplatPrecomp>.stride
    guard let pcBuf = device.makeBuffer(length: precompSize, options: .storageModePrivate) else {
      throw NSError(
        domain: "Huashan", code: 6,
        userInfo: [NSLocalizedDescriptionKey: "Failed to allocate precomp buffer"])
    }
    self.precompBuffer = pcBuf

    // Triple-buffered sort index buffers
    let idxSize = visibleCapacity * MemoryLayout<UInt32>.stride
    var sortBufs = [MTLBuffer]()
    for _ in 0..<3 {
      guard let b = device.makeBuffer(length: idxSize, options: .storageModeShared) else {
        throw NSError(
          domain: "Huashan", code: 4,
          userInfo: [NSLocalizedDescriptionKey: "Failed to allocate sort buffer"])
      }
      // Default: sequential sampled order (no sort yet)
      let ptr = b.contents().assumingMemoryBound(to: UInt32.self)
      for i in 0..<min(visibleCapacity, self.sampledSplatIndices.count) {
        ptr[i] = self.sampledSplatIndices[i]
      }
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

  func synchronizeState(_ context: PatternSimulationContext) {
    let targetRatio = min(
      max(context.huashanSampleRatio, PatternCoordinator.minHuashanSampleRatio),
      PatternCoordinator.maxHuashanSampleRatio)
    guard abs(targetRatio - currentSampleRatio) > 0.001 else { return }

    sortLock.lock()
    let rebuildBlocked = sortInFlight
    sortLock.unlock()
    guard !rebuildBlocked else { return }

    rebuildSampleSet(sampleRatio: targetRatio)
  }

  func resetToInitialState() {}

  func updateSimulation(_ context: PatternSimulationContext) {
    // Nothing to simulate — the scene is static, only sorting needed
  }

  // MARK: - encodeComputePrepass
  func encodeComputePrepass(commandBuffer: MTLCommandBuffer, context: PatternRenderContext) {
    guard splatCount > 0 else { return }
    let sortState = currentSortState()
    guard sortState.visibleCount > 0 else { return }
    var uniforms = makeUniforms(context: context)
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
    let sortBuf = sortedIndexBuffers[sortState.bufferIndex]
    encoder.setComputePipelineState(computePipelineState)
    encoder.setBuffer(splatBuffer, offset: 0, index: 0)
    encoder.setBytes(&uniforms, length: MemoryLayout<HuashanUniforms>.stride, index: 1)
    encoder.setBuffer(precompBuffer, offset: 0, index: 2)
    encoder.setBuffer(sortBuf, offset: 0, index: 3)
    let w = min(computePipelineState.maxTotalThreadsPerThreadgroup, 256)
    encoder.dispatchThreadgroups(
      MTLSize(width: (sortState.visibleCount + w - 1) / w, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    encoder.endEncoding()
  }

  // MARK: - encodeFrame
  private var diagFrameCount = 0
  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    guard splatCount > 0 else { return }

    // Always update camera and kick background sort, even before first sort completes.
    let viewData = context.viewData
    let v2w0 = viewData.viewToWorldTransforms[0]
    let camPosWorld = SIMD3<Float>(v2w0.columns.3.x, v2w0.columns.3.y, v2w0.columns.3.z)
    let fwdWorld4 = v2w0 * SIMD4<Float>(0, 0, -1, 0)
    let camPos4 = sceneInverse * SIMD4<Float>(camPosWorld.x, camPosWorld.y, camPosWorld.z, 1)
    let camPos = SIMD3<Float>(camPos4.x, camPos4.y, camPos4.z)
    let camFwd4 = sceneInverse * SIMD4<Float>(fwdWorld4.x, fwdWorld4.y, fwdWorld4.z, 0)
    let camFwd = normalize(SIMD3<Float>(camFwd4.x, camFwd4.y, camFwd4.z))
    triggerSortIfNeeded(cameraPos: camPos, cameraForward: camFwd)

    var uniforms = makeUniforms(context: context)
    let sortState = currentSortState()
    logDiagnosticsIfNeeded(sortState: sortState, now: CFAbsoluteTimeGetCurrent())
    guard sortState.visibleCount > 0 else { return }

    diagFrameCount += 1
    if diagFrameCount == 2 {
      let centerClip = uniforms.eye0.vpMatrix * SIMD4<Float>(0, 0, 0, 1)
      let ndc = SIMD3<Float>(
        centerClip.x / centerClip.w, centerClip.y / centerClip.w, centerClip.z / centerClip.w)
      print(
        "[Huashan] scene-origin ndc=\(ndc)  splatCount=\(splatCount) renderCount=\(renderCount) visibleCount=\(sortState.visibleCount) visibleCapacity=\(visibleCapacity) visibleChunks=\(sortState.visibleChunkCount) chunks=\(sampleChunks.count) stride=\(renderStride) sampleRatio=\(Int((currentSampleRatio * 100).rounded()))%"
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
      instanceCount: sortState.visibleCount)
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
    let heavySampleSet = renderCount >= 320_000
    let movementThreshold: Float = heavySampleSet ? 0.54 : 0.26
    let forwardThreshold: Float = heavySampleSet ? 0.955 : 0.98
    let minSortInterval: CFTimeInterval = heavySampleSet ? 0.90 : 0.40
    let needsResort = movement > movementThreshold || forwardDot < forwardThreshold
    if !needsFirstSort && (!needsResort || now - lastSortTime < minSortInterval) {
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
    let sampleChunks = self.sampleChunks
    let visibleCapacity = self.visibleCapacity

    // Pick the buffer that neither the GPU nor previous sort is using
    let writeBuf = (gpuSortBuf + 1) % 3
    let targetBuf = self.sortedIndexBuffers[writeBuf]

    sortQueue.async { [weak self] in
      guard let self else { return }
      let sortStart = CFAbsoluteTimeGetCurrent()

      // Chunk coarse culling trims most splats before fine depth sorting.
      var visibleChunkCount = 0
      for chunkIndex in sampleChunks.indices {
        let chunk = sampleChunks[chunkIndex]
        let dx = chunk.center.x - pos.x
        let dy = chunk.center.y - pos.y
        let dz = chunk.center.z - pos.z
        let dist2 = dx * dx + dy * dy + dz * dz
        let dist = sqrt(max(dist2, 1e-4))
        let depth = dx * fwd.x + dy * fwd.y + dz * fwd.z
        if depth + chunk.radius <= 0.0 {
          continue
        }
        if dist - chunk.radius > HuashanSplatRenderer.maxVisibleDistanceScene {
          continue
        }
        if (depth + chunk.radius) / dist < HuashanSplatRenderer.minVisibleCosine {
          continue
        }

        self.chunkDepths[chunkIndex] = depth
        self.chunkOrder[visibleChunkCount] = chunkIndex
        visibleChunkCount += 1
      }

      // Sort chunks back-to-front, then locally sort splats inside each chunk.
      // This avoids one global 200k-element sort while preserving approximate blend order.
      self.chunkOrder[0..<visibleChunkCount].sort { self.chunkDepths[$0] > self.chunkDepths[$1] }

      var localVisibleTotal = 0
      var maxChunkVisibleCount = 0
      for chunkListIndex in 0..<visibleChunkCount {
        let chunkIndex = self.chunkOrder[chunkListIndex]
        let chunk = sampleChunks[chunkIndex]
        let chunkEnd = chunk.startIndex + chunk.count
        self.chunkVisibleOffsets[chunkIndex] = localVisibleTotal

        var chunkVisibleCount = 0
        for sampleIndex in chunk.startIndex..<chunkEnd {
          let sample = positions[sampleIndex]
          let dx = sample.x - pos.x
          let dy = sample.y - pos.y
          let dz = sample.z - pos.z
          let dist2 = dx * dx + dy * dy + dz * dz
          if dist2 <= 1e-4 {
            continue
          }

          let depth = dx * fwd.x + dy * fwd.y + dz * fwd.z
          if depth <= 0.0 {
            continue
          }
          // dist2 > maxDist^2 → equivalent to dist > maxDist (avoids sqrt)
          if dist2 > HuashanSplatRenderer.maxVisibleDistanceScene2 {
            continue
          }
          // depth/dist < minCosine → depth^2 < dist2*minCosine^2 (valid when depth > 0)
          if depth * depth < dist2 * HuashanSplatRenderer.minVisibleCosine2 {
            continue
          }

          self.sortDepths[sampleIndex] = depth
          self.sortOrder[localVisibleTotal + chunkVisibleCount] = sampleIndex
          chunkVisibleCount += 1
        }

        self.chunkVisibleCounts[chunkIndex] = chunkVisibleCount
        localVisibleTotal += chunkVisibleCount
        maxChunkVisibleCount = max(maxChunkVisibleCount, chunkVisibleCount)
      }

      let perChunkQuota =
        visibleChunkCount > 0
        ? min(
          max(HuashanSplatRenderer.minPerChunkQuota, visibleCapacity / visibleChunkCount),
          visibleCapacity)
        : visibleCapacity
      var visibleCount = 0
      var budgetHit = false

      // First pass: guarantee each visible chunk a modest quota to keep the full silhouette.
      for chunkListIndex in 0..<visibleChunkCount {
        let chunkIndex = self.chunkOrder[chunkListIndex]
        let chunkVisibleCount = self.chunkVisibleCounts[chunkIndex]
        guard chunkVisibleCount > 0 else { continue }

        let base = self.chunkVisibleOffsets[chunkIndex]
        self.sortOrder[base..<(base + chunkVisibleCount)].sort {
          self.sortDepths[$0] > self.sortDepths[$1]
        }

        let emitCount = min(perChunkQuota, chunkVisibleCount, visibleCapacity - visibleCount)
        guard emitCount > 0 else {
          budgetHit = true
          break
        }

        for localIndex in 0..<emitCount {
          self.sortScratchIndices[visibleCount] = sampledIndices[self.sortOrder[base + localIndex]]
          visibleCount += 1
        }
        self.chunkVisibleOffsets[chunkIndex] = base + emitCount

        if visibleCount >= visibleCapacity {
          budgetHit = true
          break
        }
      }

      // Second pass: rotate extra budget across chunks in small batches instead of
      // draining one chunk at a time, so distant silhouette chunks continue to contribute.
      if visibleCount < visibleCapacity {
        var refillRemaining = true
        while visibleCount < visibleCapacity && refillRemaining {
          refillRemaining = false
          refillLoop: for chunkListIndex in 0..<visibleChunkCount {
            let chunkIndex = self.chunkOrder[chunkListIndex]
            let chunkVisibleCount = self.chunkVisibleCounts[chunkIndex]
            guard chunkVisibleCount > 0 else { continue }

            let base = self.chunkVisibleOffsets[chunkIndex] - min(perChunkQuota, chunkVisibleCount)
            let resume = self.chunkVisibleOffsets[chunkIndex]
            let chunkEnd = base + chunkVisibleCount
            guard resume < chunkEnd else { continue }

            refillRemaining = true
            let batchEnd = min(resume + HuashanSplatRenderer.refillBatchSize, chunkEnd)
            for localIndex in resume..<batchEnd {
              self.sortScratchIndices[visibleCount] = sampledIndices[self.sortOrder[localIndex]]
              visibleCount += 1
              if visibleCount >= visibleCapacity {
                budgetHit = true
                self.chunkVisibleOffsets[chunkIndex] = localIndex + 1
                break refillLoop
              }
            }
            self.chunkVisibleOffsets[chunkIndex] = batchEnd
          }
        }

        if visibleCount >= visibleCapacity {
          budgetHit = true
        }
      }

      // Copy to GPU buffer
      _ = self.sortScratchIndices.withUnsafeBytes { rawBytes in
        memcpy(
          targetBuf.contents(),
          rawBytes.baseAddress!,
          visibleCount * MemoryLayout<UInt32>.stride)
      }

      // Atomically publish the new buffer
      self.sortLock.lock()
      self.gpuSortBuf = writeBuf
      self.sortedVisibleCounts[writeBuf] = visibleCount
      self.sortedVisibleChunkCounts[writeBuf] = visibleChunkCount
      self.sortedBudgetHits[writeBuf] = budgetHit
      self.sortedSortDurationsMS[writeBuf] = Float(
        (CFAbsoluteTimeGetCurrent() - sortStart) * 1000.0)
      self.sortAvailable = true
      self.sortInFlight = false
      self.lastSortedCameraPos = pos
      self.lastSortedCameraFwd = fwd
      self.lastSortTime = now
      self.sortLock.unlock()
    }
  }

  private func currentSortState() -> (
    bufferIndex: Int,
    visibleCount: Int,
    visibleChunkCount: Int,
    budgetHit: Bool,
    sortDurationMS: Float
  ) {
    sortLock.lock()
    let bufferIndex = sortAvailable ? gpuSortBuf : 0
    let visibleCount =
      sortedVisibleCounts.isEmpty ? visibleCapacity : sortedVisibleCounts[bufferIndex]
    let visibleChunkCount =
      sortedVisibleChunkCounts.isEmpty ? sampleChunks.count : sortedVisibleChunkCounts[bufferIndex]
    let budgetHit = sortedBudgetHits.isEmpty ? false : sortedBudgetHits[bufferIndex]
    let sortDurationMS = sortedSortDurationsMS.isEmpty ? 0 : sortedSortDurationsMS[bufferIndex]
    sortLock.unlock()
    return (bufferIndex, visibleCount, visibleChunkCount, budgetHit, sortDurationMS)
  }

  private func logDiagnosticsIfNeeded(
    sortState: (
      bufferIndex: Int,
      visibleCount: Int,
      visibleChunkCount: Int,
      budgetHit: Bool,
      sortDurationMS: Float
    ),
    now: CFTimeInterval
  ) {
    guard now - lastDiagnosticsLogTime >= 2.0 else { return }
    lastDiagnosticsLogTime = now

    let utilization =
      visibleCapacity > 0
      ? Float(sortState.visibleCount) / Float(visibleCapacity)
      : 0
    let utilizationPct = Int((utilization * 100).rounded())
    let sortDurationText = String(format: "%.2f", sortState.sortDurationMS)
    print(
      "[Huashan] sampleRatio=\(Int((currentSampleRatio * 100).rounded()))% visibleChunks=\(sortState.visibleChunkCount)/\(sampleChunks.count) visibleSplats=\(sortState.visibleCount)/\(visibleCapacity) util=\(utilizationPct)% budgetHit=\(sortState.budgetHit) sort=\(sortDurationText)ms"
    )
  }

  private func rebuildSampleSet(sampleRatio: Float) {
    let rebuiltSamples = Self.buildSampleSet(
      rawPtr: splatBuffer.contents().assumingMemoryBound(to: HuashanSplatPoint.self),
      splatCount: splatCount,
      sampleRatio: sampleRatio)
    let nextVisibleCount = min(visibleCapacity, rebuiltSamples.positions.count)
    let warmStartVisibleCount = min(
      nextVisibleCount, HuashanSplatRenderer.warmStartVisibleSplatCount)

    sampledSplatPositions = rebuiltSamples.positions
    sampledSplatIndices = rebuiltSamples.indices
    sampleChunks = rebuiltSamples.chunks
    renderCount = rebuiltSamples.positions.count
    renderStride = rebuiltSamples.renderStride
    currentSampleRatio = sampleRatio
    sortDepths = [Float](repeating: 0, count: renderCount)
    sortOrder = Array(0..<renderCount)
    sortScratchIndices = [UInt32](repeating: 0, count: visibleCapacity)
    for i in 0..<nextVisibleCount { sortScratchIndices[i] = rebuiltSamples.indices[i] }
    chunkDepths = [Float](repeating: 0, count: rebuiltSamples.chunks.count)
    chunkOrder = Array(0..<rebuiltSamples.chunks.count)
    chunkVisibleCounts = [Int](repeating: 0, count: rebuiltSamples.chunks.count)
    chunkVisibleOffsets = [Int](repeating: 0, count: rebuiltSamples.chunks.count)

    for bufferIndex in sortedIndexBuffers.indices {
      let ptr = sortedIndexBuffers[bufferIndex].contents().assumingMemoryBound(to: UInt32.self)
      for i in 0..<warmStartVisibleCount {
        ptr[i] = sortScratchIndices[i]
      }
      sortedVisibleCounts[bufferIndex] = warmStartVisibleCount
      sortedVisibleChunkCounts[bufferIndex] = rebuiltSamples.chunks.count
      sortedBudgetHits[bufferIndex] = false
      sortedSortDurationsMS[bufferIndex] = 0
    }

    sortLock.lock()
    sortAvailable = false
    lastSortedCameraPos = SIMD3(repeating: .greatestFiniteMagnitude)
    lastSortedCameraFwd = SIMD3(0, 0, -1)
    sortLock.unlock()

    print(
      "[Huashan] rebuilt sample set ratio=\(Int((sampleRatio * 100).rounded()))% renderCount=\(renderCount) chunks=\(sampleChunks.count) stride=\(renderStride)"
    )
  }
}

// MARK: - Pipeline factory
extension HuashanSplatRenderer {
  fileprivate static func buildSampleSet(
    rawPtr: UnsafePointer<HuashanSplatPoint>,
    splatCount: Int,
    sampleRatio: Float
  ) -> (
    positions: [HuashanPackedPosition],
    indices: [UInt32],
    chunks: [HuashanSplatChunk],
    renderStride: UInt32
  ) {
    let clampedRatio = min(
      max(sampleRatio, PatternCoordinator.minHuashanSampleRatio),
      PatternCoordinator.maxHuashanSampleRatio)
    let rawTargetSampleCount = max(1, Int((Float(splatCount) * clampedRatio).rounded()))
    let targetSampleCount = min(rawTargetSampleCount, maxSampledSplatCount)
    let keepRatio = min(Float(targetSampleCount) / Float(splatCount), 1.0)
    let renderStride =
      keepRatio > 0
      ? UInt32(max(1, Int((1.0 / keepRatio).rounded())))
      : UInt32(splatCount)

    if targetSampleCount >= splatCount {
      var sampledPositions = [HuashanPackedPosition]()
      var sampledIndices = [UInt32]()
      sampledPositions.reserveCapacity(splatCount)
      sampledIndices.reserveCapacity(splatCount)

      for i in 0..<splatCount {
        sampledPositions.append(
          HuashanPackedPosition(x: rawPtr[i].px, y: rawPtr[i].py, z: rawPtr[i].pz))
        sampledIndices.append(UInt32(i))
      }

      let chunkedSamples = buildSpatialChunks(
        sampledPositions: sampledPositions,
        sampledIndices: sampledIndices)
      return (
        positions: chunkedSamples.positions,
        indices: chunkedSamples.indices,
        chunks: chunkedSamples.chunks,
        renderStride: renderStride
      )
    }

    let chunkedSamples = buildSpatialSampleSet(
      rawPtr: rawPtr,
      splatCount: splatCount,
      targetSampleCount: targetSampleCount)
    return (
      positions: chunkedSamples.positions,
      indices: chunkedSamples.indices,
      chunks: chunkedSamples.chunks,
      renderStride: renderStride
    )
  }

  fileprivate static func buildSpatialSampleSet(
    rawPtr: UnsafePointer<HuashanSplatPoint>,
    splatCount: Int,
    targetSampleCount: Int
  ) -> (positions: [HuashanPackedPosition], indices: [UInt32], chunks: [HuashanSplatChunk]) {
    var buckets: [HuashanChunkKey: [Int]] = [:]
    buckets.reserveCapacity(max(1, splatCount / 384))

    let invCell = 1.0 / chunkCellSizeScene
    for sampleIndex in 0..<splatCount {
      let point = rawPtr[sampleIndex]
      let key = HuashanChunkKey(
        x: Int32(floor(point.px * invCell)),
        y: Int32(floor(point.py * invCell)),
        z: Int32(floor(point.pz * invCell)))
      buckets[key, default: []].append(sampleIndex)
    }

    let sortedKeys = buckets.keys.sorted {
      if $0.z != $1.z { return $0.z < $1.z }
      if $0.y != $1.y { return $0.y < $1.y }
      return $0.x < $1.x
    }

    var sampledPositions = [HuashanPackedPosition]()
    var sampledIndices = [UInt32]()
    var chunks = [HuashanSplatChunk]()
    sampledPositions.reserveCapacity(targetSampleCount)
    sampledIndices.reserveCapacity(targetSampleCount)
    chunks.reserveCapacity(sortedKeys.count)

    let guaranteePerChunk = targetSampleCount >= sortedKeys.count ? 1 : 0
    var sampledCount = 0
    var sourceCountSeen = 0

    for (chunkListIndex, key) in sortedKeys.enumerated() {
      guard let bucket = buckets[key], !bucket.isEmpty else { continue }

      let bucketCount = bucket.count
      let remainingBudget = targetSampleCount - sampledCount
      if remainingBudget <= 0 { break }

      let remainingSourceCount = max(splatCount - sourceCountSeen, 1)
      let takeCount: Int
      if chunkListIndex == sortedKeys.count - 1 {
        takeCount = min(bucketCount, remainingBudget)
      } else {
        let proportional = Int(
          (Float(bucketCount) * Float(remainingBudget) / Float(remainingSourceCount)).rounded(.down)
        )
        takeCount = min(bucketCount, remainingBudget, max(guaranteePerChunk, proportional))
      }

      sourceCountSeen += bucketCount
      guard takeCount > 0 else { continue }

      let startIndex = sampledPositions.count
      var center = SIMD3<Float>(repeating: 0)
      for sourceIndex in bucket {
        center += SIMD3<Float>(
          rawPtr[sourceIndex].px, rawPtr[sourceIndex].py, rawPtr[sourceIndex].pz)
      }
      center /= Float(bucketCount)

      if takeCount >= bucketCount {
        for sourceIndex in bucket {
          sampledPositions.append(
            HuashanPackedPosition(
              x: rawPtr[sourceIndex].px,
              y: rawPtr[sourceIndex].py,
              z: rawPtr[sourceIndex].pz))
          sampledIndices.append(UInt32(sourceIndex))
        }
      } else {
        let step = Float(bucketCount) / Float(takeCount)
        var cursor = step * 0.5
        var lastLocalIndex = -1
        for _ in 0..<takeCount {
          var localIndex = min(bucketCount - 1, Int(cursor))
          if localIndex <= lastLocalIndex {
            localIndex = min(bucketCount - 1, lastLocalIndex + 1)
          }
          let sourceIndex = bucket[localIndex]
          sampledPositions.append(
            HuashanPackedPosition(
              x: rawPtr[sourceIndex].px,
              y: rawPtr[sourceIndex].py,
              z: rawPtr[sourceIndex].pz))
          sampledIndices.append(UInt32(sourceIndex))
          lastLocalIndex = localIndex
          cursor += step
        }
      }

      var radius: Float = 0
      for sourceIndex in bucket {
        let position = SIMD3<Float>(
          rawPtr[sourceIndex].px, rawPtr[sourceIndex].py, rawPtr[sourceIndex].pz)
        radius = max(radius, simd_length(position - center))
      }

      chunks.append(
        HuashanSplatChunk(
          startIndex: startIndex,
          count: sampledPositions.count - startIndex,
          center: center,
          radius: max(radius, chunkCellSizeScene * 0.5)))
      sampledCount = sampledPositions.count
    }

    return (positions: sampledPositions, indices: sampledIndices, chunks: chunks)
  }

  fileprivate static func buildSpatialChunks(
    sampledPositions: [HuashanPackedPosition],
    sampledIndices: [UInt32]
  ) -> (positions: [HuashanPackedPosition], indices: [UInt32], chunks: [HuashanSplatChunk]) {
    var buckets: [HuashanChunkKey: [Int]] = [:]
    buckets.reserveCapacity(max(1, sampledPositions.count / 384))

    for sampleIndex in sampledPositions.indices {
      let position = sampledPositions[sampleIndex]
      let invCell = 1.0 / chunkCellSizeScene
      let key = HuashanChunkKey(
        x: Int32(floor(position.x * invCell)),
        y: Int32(floor(position.y * invCell)),
        z: Int32(floor(position.z * invCell)))
      buckets[key, default: []].append(sampleIndex)
    }

    let sortedKeys = buckets.keys.sorted {
      if $0.z != $1.z { return $0.z < $1.z }
      if $0.y != $1.y { return $0.y < $1.y }
      return $0.x < $1.x
    }

    var reorderedPositions = [HuashanPackedPosition]()
    var reorderedIndices = [UInt32]()
    var chunks = [HuashanSplatChunk]()
    reorderedPositions.reserveCapacity(sampledPositions.count)
    reorderedIndices.reserveCapacity(sampledIndices.count)
    chunks.reserveCapacity(sortedKeys.count)

    for key in sortedKeys {
      guard let bucket = buckets[key], !bucket.isEmpty else { continue }
      let startIndex = reorderedPositions.count
      var center = SIMD3<Float>(repeating: 0)
      for sourceIndex in bucket {
        let position = sampledPositions[sourceIndex]
        reorderedPositions.append(position)
        reorderedIndices.append(sampledIndices[sourceIndex])
        center += position.simd
      }

      center /= Float(bucket.count)
      var radius: Float = 0
      for offset in 0..<bucket.count {
        let position = reorderedPositions[startIndex + offset].simd
        radius = max(radius, simd_length(position - center))
      }

      chunks.append(
        HuashanSplatChunk(
          startIndex: startIndex,
          count: bucket.count,
          center: center,
          radius: max(radius, chunkCellSizeScene * 0.5)))
    }

    return (positions: reorderedPositions, indices: reorderedIndices, chunks: chunks)
  }

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
