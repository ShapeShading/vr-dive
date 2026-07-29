import Metal
import MetalKit
import simd

// LunarSurfaceRenderer.swift
//
// Renders an immersive lunar-surface scene: a procedurally ray-marched cratered
// ground underfoot, and a sky containing the Earth, the Sun, and the Milky Way.
// A large inward-facing box surrounds the viewer (same technique as
// SynthwaveSunsetRenderer) so every direction the player looks resolves to either
// ground or sky.
//
// Texture credits (bundled under Demos/LunarSurface/Textures):
// - Earth: NASA Apollo 17 "Blue Marble" (AS17-148-22727), public domain.
// - Milky Way: ESA/Gaia/DPAC all-sky map, CC BY-SA 3.0 IGO — attribution required
//   ("Gaia Data Processing and Analysis Consortium (DPAC); A. Moitinho / A. F.
//   Silva / M. Barros / C. Barata, University of Lisbon; H. Savietto, Fork
//   Research" — see README credits).

final class LunarSurfaceRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .lunarSurface
  let preferredClearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

  private let pipelineState: MTLRenderPipelineState
  private let bakeHeightPipelineState: MTLComputePipelineState
  private let depthStencilState: MTLDepthStencilState
  private let vertexBuffer: MTLBuffer
  private let indexBuffer: MTLBuffer
  private let indexCount: Int
  private let maxViewCount: Int
  private let earthTexture: MTLTexture
  private let milkywayTexture: MTLTexture
  private let heightMapTexture: MTLTexture

  private let boxHalfExtents = SIMD3<Float>(1.0, 1.0, 1.0)
  private let objectCenter = SIMD3<Float>(0.0, 0.0, 0.0)

  // The coarse heightmap tile covers a (2*halfRange)^2 world-space square
  // around `bakedCenter`, big enough to comfortably exceed the ray march's
  // maxDist (55). It's only re-baked once the player has walked far enough
  // (via "箱内移动" pattern navigation) that the old tile no longer covers
  // the visible area — see encodeComputePrepass.
  private static let heightMapResolution = 512
  private static let heightMapHalfRange: Float = 70.0
  private static let heightMapRecenterThreshold: Float = heightMapHalfRange * 0.35
  private var bakedCenter: SIMD2<Float>?

  private var animationTime: Float = 0
  private var lastSimulationTime: Float?

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    self.maxViewCount = max(1, maxViewCount)

    let geo = LunarSurfaceRenderer.makeBox(device: device)
    vertexBuffer = geo.vertexBuffer
    indexBuffer = geo.indexBuffer
    indexCount = geo.indexCount

    let loader = MTKTextureLoader(device: device)
    // Both source images are ordinary sRGB-encoded photos/maps. Decoding them
    // as sRGB here (rather than treating the raw gamma-encoded bytes as linear)
    // is required so dark regions (e.g. empty space in the Milky Way map) come
    // out near-black instead of washed-out grey once the shader's final
    // `sqrt()` display-gamma pass is applied on top.
    let loaderOptions: [MTKTextureLoader.Option: Any] = [
      .SRGB: true,
      .generateMipmaps: true,
      .origin: MTKTextureLoader.Origin.topLeft,
    ]
    // The Milky Way map is sampled as a full 360° equirectangular wrap
    // (address::repeat across u). MTKTextureLoader's mip generation doesn't
    // know the texture tiles horizontally, so each downsampled level blends
    // across the u=0/u=1 border as if it were a hard edge instead of a
    // seamless wrap — that mismatch shows up as a persistent bright/dark
    // seam line running the full height of the sky (ground to zenith) at
    // whatever longitude the border falls on. The nebulosity/star content is
    // slow-varying enough that skipping mipmaps entirely (always sample the
    // full-resolution base level) avoids the bad seam with no visible loss.
    let milkywayLoaderOptions: [MTKTextureLoader.Option: Any] = [
      .SRGB: true,
      .generateMipmaps: false,
      .origin: MTKTextureLoader.Origin.topLeft,
    ]

    guard let earthURL = Bundle.main.url(forResource: "earth_blue_marble", withExtension: "jpg")
    else {
      throw NSError(
        domain: "LunarSurfaceRenderer", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "earth_blue_marble.jpg not found in bundle"])
    }
    guard
      let milkywayURL = Bundle.main.url(forResource: "milkyway_equirect", withExtension: "png")
    else {
      throw NSError(
        domain: "LunarSurfaceRenderer", code: 2,
        userInfo: [NSLocalizedDescriptionKey: "milkyway_equirect.png not found in bundle"])
    }

    earthTexture = try loader.newTexture(URL: earthURL, options: loaderOptions)
    milkywayTexture = try loader.newTexture(URL: milkywayURL, options: milkywayLoaderOptions)

    let heightMapDesc = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .r16Float,
      width: LunarSurfaceRenderer.heightMapResolution,
      height: LunarSurfaceRenderer.heightMapResolution,
      mipmapped: false)
    heightMapDesc.usage = [.shaderRead, .shaderWrite]
    heightMapDesc.storageMode = .private
    guard let heightMapTexture = device.makeTexture(descriptor: heightMapDesc) else {
      throw NSError(
        domain: "LunarSurfaceRenderer", code: 3,
        userInfo: [NSLocalizedDescriptionKey: "Failed to create height map texture"])
    }
    self.heightMapTexture = heightMapTexture

    guard let bakeFunction = library.makeFunction(name: "lunarBakeHeightKernel") else {
      throw NSError(
        domain: "LunarSurfaceRenderer", code: 4,
        userInfo: [NSLocalizedDescriptionKey: "lunarBakeHeightKernel not found in library"])
    }
    bakeHeightPipelineState = try device.makeComputePipelineState(function: bakeFunction)

    pipelineState = try LunarSurfaceRenderer.makePipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    depthStencilState = LunarSurfaceRenderer.makeDepthStencilState(device: device)
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    defer { lastSimulationTime = context.time }
    guard let lastSimulationTime else { return }
    let deltaTime = max(0, min(context.time - lastSimulationTime, 1.0 / 20.0))
    animationTime += deltaTime * max(context.speedMultiplier, 0)
  }

  func resetToInitialState() {
    animationTime = 0
    lastSimulationTime = nil
    bakedCenter = nil
  }

  /// Virtual eye position (world-space) after applying the pattern-navigation
  /// transform to the local eye point — matches exactly what the fragment
  /// shader computes for `ro`. The local eye point itself now includes the
  /// real camera's offset from the (fixed) box center, so normal-mode flight
  /// shifts the baked heightmap's center just like it shifts the shader's
  /// march origin — otherwise walking far away via normal-mode navigation
  /// would leave the height bake sampling the wrong area.
  private func virtualEyeXZ(context: PatternRenderContext) -> SIMD2<Float> {
    let v2w = context.viewData.viewToWorldTransforms.first ?? matrix_identity_float4x4
    let camWorld = SIMD3<Float>(v2w.columns.3.x, v2w.columns.3.y, v2w.columns.3.z)
    let eyeOffset = camWorld - objectCenter
    let baseRo = SIMD4<Float>(eyeOffset.x, 1.7 + eyeOffset.y, eyeOffset.z, 1)
    let ro = context.patternNavigationTransform * baseRo
    return SIMD2<Float>(ro.x, ro.z)
  }

  func encodeComputePrepass(commandBuffer: MTLCommandBuffer, context: PatternRenderContext) {
    let currentEye = virtualEyeXZ(context: context)
    if let bakedCenter,
      simd_length(currentEye - bakedCenter) < LunarSurfaceRenderer.heightMapRecenterThreshold
    {
      return  // Existing bake still covers the visible area — skip re-baking.
    }
    bakedCenter = currentEye

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
    encoder.setComputePipelineState(bakeHeightPipelineState)
    encoder.setTexture(heightMapTexture, index: 0)

    var uniforms = LunarSurfaceUniforms(
      time: animationTime,
      viewCount: UInt32(max(context.viewData.viewCount, 1)),
      _pad0: 0,
      _pad1: 0,
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0),
      boxHalfExtents: SIMD4<Float>(boxHalfExtents.x, boxHalfExtents.y, boxHalfExtents.z, 0),
      patternTransform: context.patternNavigationTransform,
      heightMapParams: SIMD4<Float>(
        currentEye.x, currentEye.y, LunarSurfaceRenderer.heightMapHalfRange, 0))
    encoder.setBytes(&uniforms, length: MemoryLayout<LunarSurfaceUniforms>.stride, index: 0)

    let w = bakeHeightPipelineState.threadExecutionWidth
    let h = max(1, bakeHeightPipelineState.maxTotalThreadsPerThreadgroup / w)
    let threadsPerGroup = MTLSize(width: w, height: h, depth: 1)
    let resolution = LunarSurfaceRenderer.heightMapResolution
    let groups = MTLSize(
      width: (resolution + w - 1) / w,
      height: (resolution + h - 1) / h,
      depth: 1)
    encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
    encoder.endEncoding()
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setRenderPipelineState(pipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.back)
    context.applyViewConfiguration(on: encoder)

    encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)

    let currentEye = bakedCenter ?? virtualEyeXZ(context: context)

    var uniforms = LunarSurfaceUniforms(
      time: animationTime,
      viewCount: UInt32(context.viewData.viewCount),
      _pad0: 0,
      _pad1: 0,
      objectCenter: SIMD4<Float>(objectCenter.x, objectCenter.y, objectCenter.z, 0),
      boxHalfExtents: SIMD4<Float>(boxHalfExtents.x, boxHalfExtents.y, boxHalfExtents.z, 0),
      patternTransform: context.patternNavigationTransform,
      heightMapParams: SIMD4<Float>(
        currentEye.x, currentEye.y, LunarSurfaceRenderer.heightMapHalfRange, 0))

    encoder.setVertexBytes(&uniforms, length: MemoryLayout<LunarSurfaceUniforms>.stride, index: 1)

    var vpMatrices = context.viewData.viewProjectionMatrices
    if vpMatrices.isEmpty { vpMatrices = [matrix_identity_float4x4] }
    vpMatrices.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(base, length: $0.count, index: 2)
      }
    }

    encoder.setFragmentBytes(
      &uniforms, length: MemoryLayout<LunarSurfaceUniforms>.stride, index: 0)

    var viewToWorld = context.viewData.viewToWorldTransforms
    if viewToWorld.isEmpty { viewToWorld = [matrix_identity_float4x4] }
    viewToWorld.withUnsafeBytes {
      if let base = $0.baseAddress, $0.count > 0 {
        encoder.setFragmentBytes(base, length: $0.count, index: 1)
      }
    }

    encoder.setFragmentTexture(earthTexture, index: 0)
    encoder.setFragmentTexture(milkywayTexture, index: 1)
    encoder.setFragmentTexture(heightMapTexture, index: 2)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexCount,
      indexType: .uint16,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0)
  }
}

// MARK: - Geometry & Pipeline factory
extension LunarSurfaceRenderer {
  fileprivate static func makeBox(
    device: MTLDevice
  ) -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    typealias V = MeshVertex
    let x: Float = 1.0
    let y: Float = 1.0
    let z: Float = 1.0
    let faces: [(positions: [SIMD3<Float>], normal: SIMD3<Float>)] = [
      ([[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], [0, 0, 1]),
      ([[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]], [0, 0, -1]),
      ([[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]], [1, 0, 0]),
      ([[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]], [-1, 0, 0]),
      ([[-x, y, z], [x, y, z], [x, y, -z], [-x, y, -z]], [0, 1, 0]),
      ([[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]], [0, -1, 0]),
    ]

    var vertices: [V] = []
    vertices.reserveCapacity(24)
    var indices: [UInt16] = []
    indices.reserveCapacity(36)
    for face in faces {
      let base = UInt16(vertices.count)
      for position in face.positions {
        vertices.append(V(position: position, normal: face.normal))
      }
      indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
    }

    let vBuf = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<MeshVertex>.stride * vertices.count,
      options: .storageModeShared)!
    let iBuf = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: .storageModeShared)!
    return (vBuf, iBuf, indices.count)
  }

  fileprivate static func makePipelineState(
    device: MTLDevice, library: MTLLibrary, maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "lunarSurfaceVertex")
    desc.fragmentFunction = library.makeFunction(name: "lunarSurfaceFragment")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.depthAttachmentPixelFormat = .depth32Float

    let vd = MTLVertexDescriptor()
    vd.attributes[0].format = .float3
    vd.attributes[0].offset = 0
    vd.attributes[0].bufferIndex = 0
    vd.attributes[1].format = .float3
    vd.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vd.attributes[1].bufferIndex = 0
    vd.layouts[0].stride = MemoryLayout<MeshVertex>.stride
    desc.vertexDescriptor = vd

    desc.maxVertexAmplificationCount = max(maxViewCount, 1)
    return try device.makeRenderPipelineState(descriptor: desc)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let desc = MTLDepthStencilDescriptor()
    desc.depthCompareFunction = .greater
    desc.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: desc)!
  }
}
