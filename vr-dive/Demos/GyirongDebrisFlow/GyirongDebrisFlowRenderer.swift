import Foundation
import Metal
import simd

/// A metre-for-metre reconstruction of the Gyirong–Rasuwagadhi valley using
/// public DEM and OSM data. Hydrology is an intentionally coarse shallow-water
/// approximation suitable for an initial interactive VR reconstruction.
final class GyirongDebrisFlowRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .gyirongDebrisFlow
  let preferredClearColor = MTLClearColor(red: 0.39, green: 0.42, blue: 0.44, alpha: 1)

  private static let navigationSpeedScale: Float = 250
  /// Twenty-four times the previous count.  Individual clasts are now much
  /// smaller, so density rather than oversized geometry makes the bore legible.
  private static let particleCount = 196_608
  /// Four decorrelated tetrahedral fragments are rendered for every physical
  /// carrier. This raises visible density to 786,432 clasts while keeping the
  /// expensive water/terrain integration at 196,608 particles.
  private static let particleVisualReplicaCount = 4
  private static let particleVerticesPerReplica = 12
  private static let simulationSpeed: Float = 12
  private static let breachStartTime: Float = 156
  private static let routedTravelDuration: Float = 790
  private static let maximumPendingTime: Float = 2.0
  private static let maximumSubstep: Float = 0.45
  private static let maximumSubstepsPerFrame = 3
  /// Initial view: roughly 420 m above the Chinese approach, with the border
  /// gate centred 650 m ahead toward Nepal.
  private static let portGroundY: Float = -420
  private static let portForwardZ: Float = -650
  private static let terrainTileCellSize = 32
  // OSM way 904894059 (Gyirong Port national gate), queried 2026-08-27.
  // The previous model incorrectly treated the nearby CCTV/plaza reference
  // coordinate as the building centre, placing the gate about 149 m too far
  // north and rotating its long facade into the mountainside.
  private static let gateLatitude = 28.279_511
  private static let gateLongitude = 85.377_742
  /// Principal axes of the measured 83 x 51 m OSM footprint in scene x/z.
  /// Across follows the 70.47-degree facade bearing. Positive forward points
  /// south-southeast from the Chinese apron through the gate toward Nepal.
  private static let gateAcross = SIMD2<Float>(-0.334_228, 0.942_492)
  private static let gateForward = SIMD2<Float>(-0.942_492, -0.334_228)

  private let metadata: GyirongSceneMetadata
  private let heights: [Float]
  /// Presents the geographic scene from the Chinese approach: the gate is
  /// centred 650 m ahead, its road axis points into the valley, and the mapped
  /// river appears on the viewer's right as in the supplied aerial imagery.
  private let scenePresentationTransform: simd_float4x4
  private let terrainTiles: [GyirongTerrainTile]
  private let outerTerrainLOD: GyirongMeshLOD
  private let skyVertexBuffer: MTLBuffer
  private let skyIndexBuffer: MTLBuffer
  private let skyIndexCount: Int
  private let buildingVertexBuffer: MTLBuffer
  private let buildingIndexBuffer: MTLBuffer
  private let buildingIndexCount: Int
  private let waterVertexBuffer: MTLBuffer
  private let waterIndexBuffer: MTLBuffer
  private let waterIndexCount: Int
  private let terrainHeightTexture: MTLTexture
  private let flowGuideTexture: MTLTexture
  private let flowPathBuffer: MTLBuffer
  private let flowPathPointCount: Int
  private let flowSourceUV: SIMD2<Float>
  private let waterStateTextures: [MTLTexture]
  private let particleBuffer: MTLBuffer

  private let meshPipelineState: MTLRenderPipelineState
  private let skyPipelineState: MTLRenderPipelineState
  private let waterPipelineState: MTLRenderPipelineState
  private let particlePipelineState: MTLRenderPipelineState
  private let opaqueDepthState: MTLDepthStencilState
  private let skyDepthState: MTLDepthStencilState
  private let transparentDepthState: MTLDepthStencilState
  private let resetWaterPipelineState: MTLComputePipelineState
  private let resetParticlePipelineState: MTLComputePipelineState
  private let stepWaterPipelineState: MTLComputePipelineState
  private let updateParticlePipelineState: MTLComputePipelineState

  private var currentWaterIndex = 0
  private var simulationTime: Float = 0
  private var pendingSimulationTime: Float = 0
  private var lastSimulationTime: Float?
  private var needsReset = true
  private var didLogConfiguration = false

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int) throws {
    let scene = try Self.loadSceneData()
    metadata = scene.metadata
    scenePresentationTransform = Self.makeScenePresentationTransform(metadata: scene.metadata)
    let mappedFlowPath = Self.makeMappedRiverFlowPath(metadata: scene.metadata)
    guard let mappedSource = mappedFlowPath.first, mappedFlowPath.count >= 2 else {
      throw GyirongDebrisFlowError.invalidData("mapped river flow path")
    }
    flowSourceUV = mappedSource
    flowPathPointCount = mappedFlowPath.count
    let riverConditionedHeights = Self.conditionTerrainForMappedRiver(
      metadata: scene.metadata,
      heights: scene.heights,
      path: mappedFlowPath)
    let conditionedHeights = Self.conditionTerrainForPortFacility(
      metadata: scene.metadata,
      heights: riverConditionedHeights)
    heights = conditionedHeights

    terrainTiles = try Self.makeTerrainTiles(
      device: device,
      metadata: scene.metadata,
      heights: conditionedHeights,
      tileCellSize: Self.terrainTileCellSize,
      flowPath: mappedFlowPath)
    outerTerrainLOD = try Self.makeOuterTerrainLOD(
      device: device,
      metadata: scene.metadata,
      heights: conditionedHeights)

    let skyDome = try Self.makeSkyDome(device: device)
    skyVertexBuffer = skyDome.vertexBuffer
    skyIndexBuffer = skyDome.indexBuffer
    skyIndexCount = skyDome.indexCount

    let buildings = try Self.makeBuildings(
      device: device,
      metadata: scene.metadata,
      heights: conditionedHeights)
    buildingVertexBuffer = buildings.vertexBuffer
    buildingIndexBuffer = buildings.indexBuffer
    buildingIndexCount = buildings.indexCount

    let waterMesh = try Self.makeWaterMesh(device: device, metadata: scene.metadata)
    waterVertexBuffer = waterMesh.vertexBuffer
    waterIndexBuffer = waterMesh.indexBuffer
    waterIndexCount = waterMesh.indexCount

    terrainHeightTexture = try Self.makeTerrainHeightTexture(
      device: device,
      metadata: scene.metadata,
      heights: conditionedHeights)
    flowGuideTexture = try Self.makeFlowGuideTexture(
      device: device,
      metadata: scene.metadata,
      heights: conditionedHeights,
      path: mappedFlowPath)
    flowPathBuffer = try Self.makeFlowPathBuffer(
      device: device,
      path: mappedFlowPath)
    waterStateTextures = try [0, 1].map {
      try Self.makeWaterStateTexture(
        device: device,
        metadata: scene.metadata,
        label: "Gyirong water state \($0)")
    }

    guard
      let particles = device.makeBuffer(
        length: Self.particleCount * MemoryLayout<GyirongFlowParticle>.stride,
        options: .storageModePrivate)
    else {
      throw GyirongDebrisFlowError.resourceAllocationFailed("particle buffer")
    }
    particles.label = "Gyirong flow particles"
    particleBuffer = particles

    skyPipelineState = try Self.makeSkyPipeline(
      device: device, library: library, maxViewCount: maxViewCount)
    meshPipelineState = try Self.makeMeshPipeline(
      device: device, library: library, maxViewCount: maxViewCount)
    waterPipelineState = try Self.makeWaterPipeline(
      device: device, library: library, maxViewCount: maxViewCount)
    particlePipelineState = try Self.makeParticlePipeline(
      device: device, library: library, maxViewCount: maxViewCount)
    skyDepthState = Self.makeSkyDepthState(device: device)
    opaqueDepthState = Self.makeDepthState(device: device, writesDepth: true)
    transparentDepthState = Self.makeDepthState(device: device, writesDepth: false)

    resetWaterPipelineState = try Self.makeComputePipeline(
      device: device, library: library, name: "gyirongResetWater")
    resetParticlePipelineState = try Self.makeComputePipeline(
      device: device, library: library, name: "gyirongResetParticles")
    stepWaterPipelineState = try Self.makeComputePipeline(
      device: device, library: library, name: "gyirongStepWater")
    updateParticlePipelineState = try Self.makeComputePipeline(
      device: device, library: library, name: "gyirongUpdateParticles")
  }

  func updateSimulation(_ context: PatternSimulationContext) {
    defer { lastSimulationTime = context.time }
    guard let lastSimulationTime else { return }
    let realDelta = max(0, min(context.time - lastSimulationTime, 1.0 / 20.0))
    let simulatedDelta = realDelta * max(context.speedMultiplier, 0) * Self.simulationSpeed
    pendingSimulationTime = min(
      pendingSimulationTime + simulatedDelta,
      Self.maximumPendingTime)
  }

  func resetToInitialState() {
    currentWaterIndex = 0
    simulationTime = 0
    pendingSimulationTime = 0
    lastSimulationTime = nil
    needsReset = true
  }

  func encodeComputePrepass(commandBuffer: MTLCommandBuffer, context: PatternRenderContext) {
    if needsReset {
      encodeWaterReset(commandBuffer: commandBuffer, context: context)
      encodeParticleReset(commandBuffer: commandBuffer, context: context)
      needsReset = false
    }

    var advancedTime: Float = 0
    var substeps = 0
    while pendingSimulationTime > 0.0001 && substeps < Self.maximumSubstepsPerFrame {
      let delta = min(pendingSimulationTime, Self.maximumSubstep)
      encodeWaterStep(commandBuffer: commandBuffer, context: context, delta: delta)
      simulationTime += delta
      pendingSimulationTime -= delta
      advancedTime += delta
      substeps += 1
    }

    if advancedTime > 0 {
      encodeParticleUpdate(
        commandBuffer: commandBuffer,
        context: context,
        delta: advancedTime)
    }
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    context.applyViewConfiguration(on: encoder)
    let navigation = acceleratedNavigationTransform(context.patternNavigationTransform)
    var uniforms = makeUniforms(
      context: context,
      navigationInverse: simd_inverse(navigation) * scenePresentationTransform,
      simulationDelta: 0)
    var viewProjectionMatrices = context.viewData.viewProjectionMatrices
    if viewProjectionMatrices.isEmpty {
      viewProjectionMatrices = [matrix_identity_float4x4]
    }
    let cameraScene = sceneCameraPosition(context: context, navigationTransform: navigation)
    // Draw a real, highly tessellated world-space dome before the scene. It
    // follows exactly the same presentation/navigation and stereo projection
    // path as the terrain. It writes a finite, very distant reverse-Z depth so
    // Compositor Services can reproject every opaque sky pixel on device;
    // closer terrain then replaces that depth normally.
    encoder.pushDebugGroup("Gyirong overcast world dome")
    encoder.setRenderPipelineState(skyPipelineState)
    encoder.setDepthStencilState(skyDepthState)
    encoder.setCullMode(.none)
    encoder.setVertexBuffer(skyVertexBuffer, offset: 0, index: 0)
    setVertexSharedData(
      encoder: encoder,
      uniforms: &uniforms,
      viewProjectionMatrices: viewProjectionMatrices)
    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: skyIndexCount,
      indexType: .uint16,
      indexBuffer: skyIndexBuffer,
      indexBufferOffset: 0)
    encoder.popDebugGroup()

    encoder.pushDebugGroup("Gyirong terrain and buildings")
    encoder.setRenderPipelineState(meshPipelineState)
    encoder.setDepthStencilState(opaqueDepthState)
    // The terrain must remain visible from the initial aerial viewpoint and
    // while flying below steep overhang-like triangles in the coarse DEM.
    encoder.setCullMode(.none)
    setVertexSharedData(
      encoder: encoder,
      uniforms: &uniforms,
      viewProjectionMatrices: viewProjectionMatrices)
    encoder.setVertexBuffer(outerTerrainLOD.vertexBuffer, offset: 0, index: 0)
    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: outerTerrainLOD.indexCount,
      indexType: .uint16,
      indexBuffer: outerTerrainLOD.indexBuffer,
      indexBufferOffset: 0)
    for tile in terrainTiles {
      let lod = selectLOD(for: tile, cameraScene: cameraScene)
      encoder.setVertexBuffer(lod.vertexBuffer, offset: 0, index: 0)
      encoder.drawIndexedPrimitives(
        type: .triangle,
        indexCount: lod.indexCount,
        indexType: .uint16,
        indexBuffer: lod.indexBuffer,
        indexBufferOffset: 0)
    }

    if buildingIndexCount > 0 {
      encoder.setCullMode(.none)
      encoder.setVertexBuffer(buildingVertexBuffer, offset: 0, index: 0)
      encoder.drawIndexedPrimitives(
        type: .triangle,
        indexCount: buildingIndexCount,
        indexType: .uint32,
        indexBuffer: buildingIndexBuffer,
        indexBufferOffset: 0)
    }
    encoder.popDebugGroup()

    encoder.pushDebugGroup("Gyirong shallow water")
    encoder.setRenderPipelineState(waterPipelineState)
    encoder.setDepthStencilState(transparentDepthState)
    encoder.setCullMode(.none)
    encoder.setVertexBuffer(waterVertexBuffer, offset: 0, index: 0)
    setVertexSharedData(
      encoder: encoder,
      uniforms: &uniforms,
      viewProjectionMatrices: viewProjectionMatrices)
    encoder.setVertexTexture(terrainHeightTexture, index: 0)
    encoder.setVertexTexture(waterStateTextures[currentWaterIndex], index: 1)
    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: waterIndexCount,
      indexType: .uint16,
      indexBuffer: waterIndexBuffer,
      indexBufferOffset: 0)
    encoder.popDebugGroup()

    encoder.pushDebugGroup("Gyirong droplets and debris")
    encoder.setRenderPipelineState(particlePipelineState)
    encoder.setDepthStencilState(opaqueDepthState)
    encoder.setCullMode(.none)
    encoder.setVertexBuffer(particleBuffer, offset: 0, index: 0)
    setVertexSharedData(
      encoder: encoder,
      uniforms: &uniforms,
      viewProjectionMatrices: viewProjectionMatrices)
    encoder.setVertexTexture(terrainHeightTexture, index: 0)
    encoder.setVertexTexture(waterStateTextures[currentWaterIndex], index: 1)
    encoder.setVertexTexture(flowGuideTexture, index: 2)
    encoder.setFragmentBytes(
      &uniforms,
      length: MemoryLayout<GyirongDebrisFlowUniforms>.stride,
      index: 0)
    encoder.drawPrimitives(
      type: .triangle,
      vertexStart: 0,
      vertexCount: Self.particleCount
        * Self.particleVisualReplicaCount
        * Self.particleVerticesPerReplica)
    encoder.popDebugGroup()

    if !didLogConfiguration {
      didLogConfiguration = true
      let event = metadata.event
      print(
        "[GyirongDebrisFlow] 1:1 aerial scene active: terrain=\(metadata.terrain.width)x\(metadata.terrain.height) in \(terrainTiles.count) tiled LODs + outer apron, continuousPortGround=true, terrainSkirts=4m, physicalSkyDome=200km-48x24-world-space-depth-writing-first, gateAlignedChineseApproach=true, gateFacadeBearing=70.47deg, OSMBuildings=\(metadata.buildings.count), calibratedGateOSM904894059=true, scalePeople=52, flowPathPoints=\(flowPathPointCount), mappedLendeCenterline=true, riverTerrainConditioned=true, portFloodBranches=right62-left26-portal12, waterFiniteGuard=true, physicalParticles=\(Self.particleCount), visibleClasts=\(Self.particleCount * Self.particleVisualReplicaCount), avalanche≈0-6s, breach≈13s, routedFloodArrival≈79s, peakScenario=\(Int(event.scenarioPeakDischargeCubicMetersPerSecond))m3/s, volumeScenario=\(String(format: "%.1f", event.scenarioReleasedVolumeCubicMeters / 1_000_000))Mm3, source=(\(event.sourceLatitude),\(event.sourceLongitude)), port=(\(event.portLatitude),\(event.portLongitude)), gaugeScenario=\(event.reportedMonitoringWaterLevelMeters)m, navigation=\(Int(Self.navigationSpeedScale))x base / 4000x single / 64000x both"
      )
    }
  }

  private func setVertexSharedData(
    encoder: MTLRenderCommandEncoder,
    uniforms: inout GyirongDebrisFlowUniforms,
    viewProjectionMatrices: [simd_float4x4]
  ) {
    encoder.setVertexBytes(
      &uniforms,
      length: MemoryLayout<GyirongDebrisFlowUniforms>.stride,
      index: 1)
    viewProjectionMatrices.withUnsafeBytes { bytes in
      if let baseAddress = bytes.baseAddress, bytes.count > 0 {
        encoder.setVertexBytes(baseAddress, length: bytes.count, index: 2)
      }
    }
  }

  private func acceleratedNavigationTransform(_ transform: simd_float4x4) -> simd_float4x4 {
    var result = transform
    result.columns.3.x *= Self.navigationSpeedScale
    result.columns.3.y *= Self.navigationSpeedScale
    result.columns.3.z *= Self.navigationSpeedScale
    return result
  }

  private func makeUniforms(
    context: PatternRenderContext,
    navigationInverse: simd_float4x4,
    simulationDelta: Float
  ) -> GyirongDebrisFlowUniforms {
    let terrain = metadata.terrain
    let event = metadata.event
    let portU = Float((event.portLongitude - terrain.west) / (terrain.east - terrain.west))
    let portV = Float((event.portLatitude - terrain.south) / (terrain.north - terrain.south))
    let gateU = Float((Self.gateLongitude - terrain.west) / (terrain.east - terrain.west))
    let gateV = Float((Self.gateLatitude - terrain.south) / (terrain.north - terrain.south))
    let gateElevation = Self.sampleHeight(
      heights,
      width: terrain.width,
      height: terrain.height,
      u: gateU,
      v: gateV)
    return GyirongDebrisFlowUniforms(
      viewCount: UInt32(max(context.viewData.viewCount, 1)),
      gridWidth: UInt32(terrain.width),
      gridHeight: UInt32(terrain.height),
      flags: UInt32(flowPathPointCount),
      simulationTime: simulationTime,
      simulationDelta: simulationDelta,
      navigationSpeedScale: Self.navigationSpeedScale,
      particleCount: UInt32(Self.particleCount),
      terrainSizeAndDatum: SIMD4<Float>(
        Float(terrain.physicalHeightMeters),
        Float(terrain.physicalWidthMeters),
        Float(terrain.portDatumElevationMeters),
        Float(event.reportedMonitoringWaterLevelMeters)),
      portSourceUV: SIMD4<Float>(portU, portV, flowSourceUV.x, flowSourceUV.y),
      sceneOrigin: SIMD4<Float>(0, Self.portGroundY, Self.portForwardZ, 0),
      floodScenario: SIMD4<Float>(
        Self.breachStartTime,
        Self.routedTravelDuration,
        Float(event.scenarioHydraulicBoreDepthMeters),
        Float(event.scenarioSprayHeightMeters)),
      floodHydrology: SIMD4<Float>(
        Float(event.scenarioPeakDischargeCubicMetersPerSecond),
        Float(event.scenarioReleasedVolumeCubicMeters),
        100,
        0),
      facilityUVAndElevation: SIMD4<Float>(gateU, gateV, gateElevation, 230),
      facilityFrame: SIMD4<Float>(
        -Self.gateAcross.y,
        Self.gateAcross.x,
        -Self.gateForward.y,
        Self.gateForward.x),
      navigationInverse: navigationInverse)
  }

  private func makeComputeUniforms(
    context: PatternRenderContext,
    delta: Float
  ) -> GyirongDebrisFlowUniforms {
    makeUniforms(
      context: context,
      navigationInverse: matrix_identity_float4x4,
      simulationDelta: delta)
  }

  private func encodeWaterReset(
    commandBuffer: MTLCommandBuffer,
    context: PatternRenderContext
  ) {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
    encoder.label = "Reset Gyirong water"
    encoder.setComputePipelineState(resetWaterPipelineState)
    encoder.setTexture(waterStateTextures[0], index: 0)
    encoder.setTexture(waterStateTextures[1], index: 1)
    encoder.setTexture(flowGuideTexture, index: 2)
    var uniforms = makeComputeUniforms(context: context, delta: 0)
    encoder.setBytes(
      &uniforms,
      length: MemoryLayout<GyirongDebrisFlowUniforms>.stride,
      index: 0)
    dispatch2D(
      encoder: encoder,
      pipeline: resetWaterPipelineState,
      width: metadata.terrain.width,
      height: metadata.terrain.height)
    encoder.endEncoding()
  }

  private func encodeParticleReset(
    commandBuffer: MTLCommandBuffer,
    context: PatternRenderContext
  ) {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
    encoder.label = "Reset Gyirong flow particles"
    encoder.setComputePipelineState(resetParticlePipelineState)
    encoder.setBuffer(particleBuffer, offset: 0, index: 0)
    var uniforms = makeComputeUniforms(context: context, delta: 0)
    encoder.setBytes(
      &uniforms,
      length: MemoryLayout<GyirongDebrisFlowUniforms>.stride,
      index: 1)
    dispatch1D(
      encoder: encoder,
      pipeline: resetParticlePipelineState,
      count: Self.particleCount)
    encoder.endEncoding()
  }

  private func encodeWaterStep(
    commandBuffer: MTLCommandBuffer,
    context: PatternRenderContext,
    delta: Float
  ) {
    let nextWaterIndex = 1 - currentWaterIndex
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
    encoder.label = "Step Gyirong shallow water"
    encoder.setComputePipelineState(stepWaterPipelineState)
    encoder.setTexture(waterStateTextures[currentWaterIndex], index: 0)
    encoder.setTexture(waterStateTextures[nextWaterIndex], index: 1)
    encoder.setTexture(terrainHeightTexture, index: 2)
    encoder.setTexture(flowGuideTexture, index: 3)
    var uniforms = makeComputeUniforms(context: context, delta: delta)
    encoder.setBytes(
      &uniforms,
      length: MemoryLayout<GyirongDebrisFlowUniforms>.stride,
      index: 0)
    dispatch2D(
      encoder: encoder,
      pipeline: stepWaterPipelineState,
      width: metadata.terrain.width,
      height: metadata.terrain.height)
    encoder.endEncoding()
    currentWaterIndex = nextWaterIndex
  }

  private func encodeParticleUpdate(
    commandBuffer: MTLCommandBuffer,
    context: PatternRenderContext,
    delta: Float
  ) {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
    encoder.label = "Advect Gyirong droplets"
    encoder.setComputePipelineState(updateParticlePipelineState)
    encoder.setBuffer(particleBuffer, offset: 0, index: 0)
    encoder.setTexture(terrainHeightTexture, index: 0)
    encoder.setTexture(waterStateTextures[currentWaterIndex], index: 1)
    encoder.setTexture(flowGuideTexture, index: 2)
    encoder.setBuffer(flowPathBuffer, offset: 0, index: 2)
    var uniforms = makeComputeUniforms(context: context, delta: delta)
    encoder.setBytes(
      &uniforms,
      length: MemoryLayout<GyirongDebrisFlowUniforms>.stride,
      index: 1)
    dispatch1D(
      encoder: encoder,
      pipeline: updateParticlePipelineState,
      count: Self.particleCount)
    encoder.endEncoding()
  }

  private func dispatch2D(
    encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState,
    width: Int,
    height: Int
  ) {
    let threadWidth = pipeline.threadExecutionWidth
    let threadHeight = max(1, pipeline.maxTotalThreadsPerThreadgroup / threadWidth)
    encoder.dispatchThreads(
      MTLSize(width: width, height: height, depth: 1),
      threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1))
  }

  private func dispatch1D(
    encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState,
    count: Int
  ) {
    let width = pipeline.threadExecutionWidth
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
  }

  private func sceneCameraPosition(
    context: PatternRenderContext,
    navigationTransform: simd_float4x4
  ) -> SIMD3<Float> {
    let viewToWorld = context.viewData.viewToWorldTransforms.first ?? matrix_identity_float4x4
    let cameraWorld = SIMD4<Float>(
      viewToWorld.columns.3.x,
      viewToWorld.columns.3.y,
      viewToWorld.columns.3.z,
      1)
    let cameraPresentedScene = navigationTransform * cameraWorld
    let cameraScene = simd_inverse(scenePresentationTransform) * cameraPresentedScene
    return SIMD3<Float>(cameraScene.x, cameraScene.y, cameraScene.z)
  }

  private func selectLOD(
    for tile: GyirongTerrainTile,
    cameraScene: SIMD3<Float>
  ) -> GyirongMeshLOD {
    // Preserve the full half-cell interpolated surface throughout the river
    // gorge.  Coarser triangles can span across a narrow channel and occlude
    // the flow even when the camera itself is several kilometres away.
    if tile.isFlowCorridor { return tile.lods[0] }
    let horizontalDistance = simd_distance(
      SIMD2<Float>(cameraScene.x, cameraScene.z),
      tile.centerSceneXZ)
    if horizontalDistance < 1_500 { return tile.lods[0] }
    if horizontalDistance < 4_500 { return tile.lods[1] }
    if horizontalDistance < 10_000 { return tile.lods[2] }
    return tile.lods[3]
  }

  private func terrainHeight(atSceneX northMeters: Float, sceneZ: Float) -> Float? {
    let terrain = metadata.terrain
    let event = metadata.event
    let portU = Float((event.portLongitude - terrain.west) / (terrain.east - terrain.west))
    let portV = Float((event.portLatitude - terrain.south) / (terrain.north - terrain.south))
    let eastMeters = -(sceneZ - Self.portForwardZ)
    let u = portU + eastMeters / Float(terrain.physicalWidthMeters)
    let v = portV + northMeters / Float(terrain.physicalHeightMeters)
    guard u >= 0, u <= 1, v >= 0, v <= 1 else { return nil }
    let elevation = Self.sampleHeight(
      heights,
      width: terrain.width,
      height: terrain.height,
      u: u,
      v: v)
    return elevation - Float(terrain.portDatumElevationMeters) + Self.portGroundY
  }
}

private enum GyirongDebrisFlowError: Error {
  case missingResource(String)
  case invalidData(String)
  case resourceAllocationFailed(String)
  case missingFunction(String)
}

extension GyirongDebrisFlowRenderer {
  fileprivate static func makeScenePresentationTransform(
    metadata: GyirongSceneMetadata
  ) -> simd_float4x4 {
    let terrain = metadata.terrain
    let event = metadata.event
    let portU = Float((event.portLongitude - terrain.west) / (terrain.east - terrain.west))
    let portV = Float((event.portLatitude - terrain.south) / (terrain.north - terrain.south))
    let gateU = Float((gateLongitude - terrain.west) / (terrain.east - terrain.west))
    let gateV = Float((gateLatitude - terrain.south) / (terrain.north - terrain.south))
    let gateScene = SIMD2<Float>(
      (gateV - portV) * Float(terrain.physicalHeightMeters),
      -(gateU - portU) * Float(terrain.physicalWidthMeters) + portForwardZ)

    // Convert the geographic north/west plane into gate-local presentation
    // coordinates. Negative local across is the river/east side and maps to
    // screen-right; positive local forward is the Nepal valley and maps ahead.
    var transform = matrix_identity_float4x4
    transform.columns.0 = SIMD4<Float>(-gateAcross.x, 0, -gateForward.x, 0)
    transform.columns.2 = SIMD4<Float>(-gateAcross.y, 0, -gateForward.y, 0)
    transform.columns.3 = SIMD4<Float>(
      simd_dot(gateScene, gateAcross),
      0,
      simd_dot(gateScene, gateForward) - 650,
      1)
    return transform
  }

  fileprivate static func loadSceneData() throws -> (
    metadata: GyirongSceneMetadata,
    heights: [Float]
  ) {
    guard let metadataURL = Bundle.main.url(forResource: "gyirong_scene", withExtension: "json")
    else {
      throw GyirongDebrisFlowError.missingResource("gyirong_scene.json")
    }
    let metadata = try JSONDecoder().decode(
      GyirongSceneMetadata.self,
      from: Data(contentsOf: metadataURL))
    guard metadata.version == 1 else {
      throw GyirongDebrisFlowError.invalidData("unsupported scene version")
    }

    let terrainName = URL(fileURLWithPath: metadata.terrain.heightFile)
    guard
      let terrainURL = Bundle.main.url(
        forResource: terrainName.deletingPathExtension().lastPathComponent,
        withExtension: terrainName.pathExtension)
    else {
      throw GyirongDebrisFlowError.missingResource(metadata.terrain.heightFile)
    }
    let terrainData = try Data(contentsOf: terrainURL, options: .mappedIfSafe)
    let expectedCount = metadata.terrain.width * metadata.terrain.height
    guard terrainData.count == expectedCount * MemoryLayout<Float>.stride else {
      throw GyirongDebrisFlowError.invalidData("terrain byte count")
    }
    var heights = [Float](repeating: 0, count: expectedCount)
    _ = heights.withUnsafeMutableBytes { destination in
      terrainData.copyBytes(to: destination)
    }
    return (metadata, heights)
  }

  /// A real, closed scene object used as the sky background. The dome is
  /// intentionally tessellated instead of using six enormous cube faces, so
  /// it follows the same variable-rate rasterization path as normal geometry.
  fileprivate static func makeSkyDome(
    device: MTLDevice
  ) throws -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    let longitudeSegments = 48
    let latitudeSegments = 24
    var vertices: [SIMD3<Float>] = []
    vertices.reserveCapacity((longitudeSegments + 1) * (latitudeSegments + 1))
    for latitude in 0...latitudeSegments {
      let latitudeAngle = -Float.pi / 2
        + Float(latitude) / Float(latitudeSegments) * Float.pi
      let ringRadius = cos(latitudeAngle)
      let y = sin(latitudeAngle)
      for longitude in 0...longitudeSegments {
        let longitudeAngle = Float(longitude) / Float(longitudeSegments) * 2 * Float.pi
        vertices.append(
          SIMD3<Float>(
            ringRadius * cos(longitudeAngle),
            y,
            ringRadius * sin(longitudeAngle)))
      }
    }

    var indices: [UInt16] = []
    indices.reserveCapacity(longitudeSegments * latitudeSegments * 6)
    let rowStride = longitudeSegments + 1
    for latitude in 0..<latitudeSegments {
      for longitude in 0..<longitudeSegments {
        let a = UInt16(latitude * rowStride + longitude)
        let b = a + 1
        let c = UInt16((latitude + 1) * rowStride + longitude)
        let d = c + 1
        indices.append(contentsOf: [a, c, b, b, c, d])
      }
    }
    guard
      let vertexBuffer = device.makeBuffer(
        bytes: vertices,
        length: vertices.count * MemoryLayout<SIMD3<Float>>.stride,
        options: .storageModeShared),
      let indexBuffer = device.makeBuffer(
        bytes: indices,
        length: indices.count * MemoryLayout<UInt16>.stride,
        options: .storageModeShared)
    else {
      throw GyirongDebrisFlowError.resourceAllocationFailed("physical sky dome")
    }
    vertexBuffer.label = "Gyirong physical sky dome vertices"
    indexBuffer.label = "Gyirong physical sky dome indices"
    return (vertexBuffer, indexBuffer, indices.count)
  }

  fileprivate static func makeTerrainTiles(
    device: MTLDevice,
    metadata: GyirongSceneMetadata,
    heights: [Float],
    tileCellSize: Int,
    flowPath: [SIMD2<Float>]
  ) throws -> [GyirongTerrainTile] {
    let terrain = metadata.terrain
    let event = metadata.event
    let portU = Float((event.portLongitude - terrain.west) / (terrain.east - terrain.west))
    let portV = Float((event.portLatitude - terrain.south) / (terrain.north - terrain.south))
    var tiles: [GyirongTerrainTile] = []
    for rowStart in stride(from: 0, to: terrain.height - 1, by: tileCellSize) {
      let rowEnd = min(rowStart + tileCellSize, terrain.height - 1)
      for columnStart in stride(from: 0, to: terrain.width - 1, by: tileCellSize) {
        let columnEnd = min(columnStart + tileCellSize, terrain.width - 1)
        let centerUV = SIMD2<Float>(
          Float(columnStart + columnEnd) * 0.5 / Float(terrain.width - 1),
          Float(rowStart + rowEnd) * 0.5 / Float(terrain.height - 1))
        let tileWidth = Float(columnEnd - columnStart)
          / Float(terrain.width - 1) * Float(terrain.physicalWidthMeters)
        let tileHeight = Float(rowEnd - rowStart)
          / Float(terrain.height - 1) * Float(terrain.physicalHeightMeters)
        let riverDistance = distanceToPathMeters(
          uv: centerUV,
          path: flowPath,
          physicalScale: SIMD2<Float>(
            Float(terrain.physicalWidthMeters),
            Float(terrain.physicalHeightMeters)))
        let isFlowCorridor = riverDistance < hypot(tileWidth, tileHeight) * 0.5 + 550
        let lods = try [1, 2, 4, 8].map { sampleStep in
          try makeTerrainTileLOD(
            device: device,
            metadata: metadata,
            heights: heights,
            columnStart: columnStart,
            columnEnd: columnEnd,
            rowStart: rowStart,
            rowEnd: rowEnd,
            sampleStep: sampleStep,
            addsSkirt: !isFlowCorridor)
        }
        let centerU = Float(columnStart + columnEnd) * 0.5 / Float(terrain.width - 1)
        let centerV = Float(rowStart + rowEnd) * 0.5 / Float(terrain.height - 1)
        let center = SIMD2<Float>(
          (centerV - portV) * Float(terrain.physicalHeightMeters),
          -(centerU - portU) * Float(terrain.physicalWidthMeters) + portForwardZ)
        tiles.append(
          GyirongTerrainTile(
            centerSceneXZ: center,
            isFlowCorridor: isFlowCorridor,
            lods: lods))
      }
    }
    return tiles
  }

  /// A very coarse continuation beyond the downloaded DEM. It is anchored to
  /// the real boundary elevations and gradually introduces broad ridges, so a
  /// distant viewer sees a mountain horizon instead of a vertical map edge.
  fileprivate static func makeOuterTerrainLOD(
    device: MTLDevice,
    metadata: GyirongSceneMetadata,
    heights: [Float]
  ) throws -> GyirongMeshLOD {
    let terrain = metadata.terrain
    let event = metadata.event
    let columnCount = 29
    let rowCount = 17
    let minimumU: Float = -0.375
    let maximumU: Float = 1.375
    let minimumV: Float = -0.5
    let maximumV: Float = 1.5
    let portU = Float((event.portLongitude - terrain.west) / (terrain.east - terrain.west))
    let portV = Float((event.portLatitude - terrain.south) / (terrain.north - terrain.south))
    let widthMeters = Float(terrain.physicalWidthMeters)
    let heightMeters = Float(terrain.physicalHeightMeters)
    let datum = Float(terrain.portDatumElevationMeters)
    var vertices: [GyirongMeshVertex] = []
    vertices.reserveCapacity(columnCount * rowCount)

    for row in 0..<rowCount {
      let v = simd_mix(minimumV, maximumV, Float(row) / Float(rowCount - 1))
      for column in 0..<columnCount {
        let u = simd_mix(minimumU, maximumU, Float(column) / Float(columnCount - 1))
        let clampedU = min(max(u, 0), 1)
        let clampedV = min(max(v, 0), 1)
        let eastMeters = (u - portU) * widthMeters
        let northMeters = (v - portV) * heightMeters
        let edgeEast = (u - clampedU) * widthMeters
        let edgeNorth = (v - clampedV) * heightMeters
        let outsideDistance = hypot(edgeEast, edgeNorth)
        let boundaryElevation = sampleHeight(
          heights,
          width: terrain.width,
          height: terrain.height,
          u: clampedU,
          v: clampedV)
        let ridgeBlend = smoothstep(0, 7_500, outsideDistance)
        let broadRidges =
          sin(eastMeters / 2_700 + northMeters / 5_100) * 430
          + cos(northMeters / 3_600 - eastMeters / 6_200) * 310
        let elevation = boundaryElevation
          + broadRidges * ridgeBlend
          - outsideDistance * 0.018
        vertices.append(
          GyirongMeshVertex(
            position: SIMD3<Float>(
              northMeters,
              elevation - datum + portGroundY,
              -eastMeters + portForwardZ),
            normal: SIMD3<Float>(0, 1, 0),
            color: SIMD4<Float>(0.18, 0.20, 0.17, 1)))
      }
    }

    for row in 0..<rowCount {
      for column in 0..<columnCount {
        let left = vertices[row * columnCount + max(column - 1, 0)].position
        let right = vertices[row * columnCount + min(column + 1, columnCount - 1)].position
        let down = vertices[max(row - 1, 0) * columnCount + column].position
        let up = vertices[min(row + 1, rowCount - 1) * columnCount + column].position
        var normal = simd_cross(right - left, up - down)
        if simd_length_squared(normal) < 0.0001 {
          normal = SIMD3<Float>(0, 1, 0)
        } else {
          normal = simd_normalize(normal)
          if normal.y < 0 { normal = -normal }
        }
        let index = row * columnCount + column
        vertices[index].normal = normal
        let elevation = vertices[index].position.y + datum - portGroundY
        vertices[index].color = terrainColor(elevation: elevation, normal: normal)
      }
    }

    var indices: [UInt16] = []
    for row in 0..<(rowCount - 1) {
      let v0 = simd_mix(minimumV, maximumV, (Float(row) + 0.5) / Float(rowCount - 1))
      for column in 0..<(columnCount - 1) {
        let u0 = simd_mix(
          minimumU,
          maximumU,
          (Float(column) + 0.5) / Float(columnCount - 1))
        // Overlap the measured DEM slightly. The apron is intentionally coarse,
        // so meeting it on the exact boundary can expose a sub-pixel crack as
        // either mesh changes LOD.
        guard u0 < 0.025 || u0 > 0.975 || v0 < 0.04 || v0 > 0.96 else { continue }
        let a = UInt16(row * columnCount + column)
        let b = a + 1
        let c = UInt16((row + 1) * columnCount + column)
        let d = c + 1
        indices.append(contentsOf: [a, c, b, b, c, d])
      }
    }

    guard
      let vertexBuffer = device.makeBuffer(
        bytes: vertices,
        length: vertices.count * MemoryLayout<GyirongMeshVertex>.stride,
        options: .storageModeShared),
      let indexBuffer = device.makeBuffer(
        bytes: indices,
        length: indices.count * MemoryLayout<UInt16>.stride,
        options: .storageModeShared)
    else {
      throw GyirongDebrisFlowError.resourceAllocationFailed("outer terrain apron")
    }
    vertexBuffer.label = "Gyirong coarse outer mountain vertices"
    indexBuffer.label = "Gyirong coarse outer mountain indices"
    return GyirongMeshLOD(
      vertexBuffer: vertexBuffer,
      indexBuffer: indexBuffer,
      indexCount: indices.count,
      sampleStep: 16)
  }

  fileprivate static func makeTerrainTileLOD(
    device: MTLDevice,
    metadata: GyirongSceneMetadata,
    heights: [Float],
    columnStart: Int,
    columnEnd: Int,
    rowStart: Int,
    rowEnd: Int,
    sampleStep: Int,
    addsSkirt: Bool
  ) throws -> GyirongMeshLOD {
    let terrain = metadata.terrain
    let event = metadata.event
    func samples(from start: Int, through end: Int) -> [Float] {
      // The nearest LOD evaluates a Catmull-Rom surface at half-cell spacing.
      // It preserves every source DEM sample while replacing each visibly
      // large ~100 m triangle with eight smoothly curved sub-triangles.
      let increment: Float = sampleStep == 1 ? 0.5 : Float(sampleStep)
      var result = stride(
        from: Float(start), through: Float(end), by: increment).map { $0 }
      if result.last != Float(end) { result.append(Float(end)) }
      return result
    }
    let columns = samples(from: columnStart, through: columnEnd)
    let rows = samples(from: rowStart, through: rowEnd)
    let portU = Float((event.portLongitude - terrain.west) / (terrain.east - terrain.west))
    let portV = Float((event.portLatitude - terrain.south) / (terrain.north - terrain.south))
    let eastCell = Float(terrain.physicalWidthMeters) / Float(terrain.width - 1)
    let northCell = Float(terrain.physicalHeightMeters) / Float(terrain.height - 1)
    let datum = Float(terrain.portDatumElevationMeters)

    var vertices: [GyirongMeshVertex] = []
    vertices.reserveCapacity(columns.count * rows.count)
    for row in rows {
      for column in columns {
        let u = Float(column) / Float(terrain.width - 1)
        let v = Float(row) / Float(terrain.height - 1)
        let elevation = sampleStep == 1
          ? sampleHeightBicubic(
            heights, width: terrain.width, height: terrain.height, u: u, v: v)
          : sampleHeight(
            heights, width: terrain.width, height: terrain.height, u: u, v: v)
        let du = 1 / Float(terrain.width - 1)
        let dv = 1 / Float(terrain.height - 1)
        let left = sampleHeightBicubic(
          heights, width: terrain.width, height: terrain.height, u: u - du, v: v)
        let right = sampleHeightBicubic(
          heights, width: terrain.width, height: terrain.height, u: u + du, v: v)
        let down = sampleHeightBicubic(
          heights, width: terrain.width, height: terrain.height, u: u, v: v - dv)
        let up = sampleHeightBicubic(
          heights, width: terrain.width, height: terrain.height, u: u, v: v + dv)
        let eastSlope = (right - left) / max(2 * eastCell, 0.001)
        let northSlope = (up - down) / max(2 * northCell, 0.001)
        let normal = simd_normalize(SIMD3<Float>(-northSlope, 1, eastSlope))
        let northMeters = (v - portV) * Float(terrain.physicalHeightMeters)
        let eastMeters = (u - portU) * Float(terrain.physicalWidthMeters)
        let position = SIMD3<Float>(
          northMeters,
          elevation - datum + portGroundY,
          -eastMeters + portForwardZ)
        let color = terrainColor(elevation: elevation, normal: normal)
        vertices.append(GyirongMeshVertex(position: position, normal: normal, color: color))
      }
    }

    var indices: [UInt16] = []
    indices.reserveCapacity((columns.count - 1) * (rows.count - 1) * 6)
    for row in 0..<(rows.count - 1) {
      for column in 0..<(columns.count - 1) {
        let a = UInt16(row * columns.count + column)
        let b = a + 1
        let c = UInt16((row + 1) * columns.count + column)
        let d = c + 1
        indices.append(contentsOf: [a, c, b, b, c, d])
      }
    }

    // Adjacent tiles choose LOD independently. Their shared edge therefore
    // contains T-junctions whenever one side samples every 1/2/4 vertices and
    // the other every 2/4/8. A vertical skirt closes that otherwise visible
    // opening without forcing all 66 tiles to use the highest-detail mesh.
    func appendSkirt(along edge: [Int]) {
      guard edge.count >= 2 else { return }
      let skirtStart = vertices.count
      for topIndex in edge {
        var skirtVertex = vertices[topIndex]
        // Only a shallow overlap is needed to hide sub-pixel T-junctions.
        // The former 160 m drop exposed each tile edge as a giant vertical
        // wall when viewed from inside the valley.
        skirtVertex.position.y -= 4
        vertices.append(skirtVertex)
      }
      for index in 0..<(edge.count - 1) {
        let topA = UInt16(edge[index])
        let topB = UInt16(edge[index + 1])
        let skirtA = UInt16(skirtStart + index)
        let skirtB = UInt16(skirtStart + index + 1)
        indices.append(contentsOf: [topA, skirtA, topB, topB, skirtA, skirtB])
      }
    }
    if addsSkirt {
      let columnCount = columns.count
      let rowCount = rows.count
      appendSkirt(along: (0..<columnCount).map { $0 })
      appendSkirt(along: (0..<rowCount).map { $0 * columnCount + columnCount - 1 })
      appendSkirt(
        along: (0..<columnCount).reversed().map { (rowCount - 1) * columnCount + $0 })
      appendSkirt(along: (0..<rowCount).reversed().map { $0 * columnCount })
    }

    guard
      let vertexBuffer = device.makeBuffer(
        bytes: vertices,
        length: vertices.count * MemoryLayout<GyirongMeshVertex>.stride,
        options: .storageModeShared),
      let indexBuffer = device.makeBuffer(
        bytes: indices,
        length: indices.count * MemoryLayout<UInt16>.stride,
        options: .storageModeShared)
    else {
      throw GyirongDebrisFlowError.resourceAllocationFailed("terrain tile LOD")
    }
    vertexBuffer.label =
      "Gyirong terrain tile \(columnStart),\(rowStart) vertices LOD \(sampleStep)"
    indexBuffer.label =
      "Gyirong terrain tile \(columnStart),\(rowStart) indices LOD \(sampleStep)"
    return GyirongMeshLOD(
      vertexBuffer: vertexBuffer,
      indexBuffer: indexBuffer,
      indexCount: indices.count,
      sampleStep: sampleStep)
  }

  fileprivate static func terrainColor(
    elevation: Float,
    normal: SIMD3<Float>
  ) -> SIMD4<Float> {
    let upward = max(normal.y, 0)
    let steepness = 1 - upward
    let forest = SIMD3<Float>(0.055, 0.17, 0.065)
    let alpineMeadow = SIMD3<Float>(0.19, 0.29, 0.12)
    let darkRock = SIMD3<Float>(0.25, 0.235, 0.22)
    let paleRock = SIMD3<Float>(0.39, 0.36, 0.33)
    let snow = SIMD3<Float>(0.90, 0.94, 0.96)
    let glacier = SIMD3<Float>(0.61, 0.79, 0.88)

    let vegetationLimit = smoothstep(4_150, 3_250, elevation)
    let meadowAmount = smoothstep(2_450, 3_600, elevation)
    var vegetation = simd_mix(
      forest,
      alpineMeadow,
      SIMD3<Float>(repeating: meadowAmount))
    vegetation = simd_mix(
      vegetation,
      darkRock,
      SIMD3<Float>(repeating: min(steepness * 1.75, 1)))
    var color = simd_mix(
      paleRock,
      vegetation,
      SIMD3<Float>(repeating: vegetationLimit))

    // Snow remains bright and distinct from the green lower valley. Steep
    // faces shed snow, while flatter high basins retain blue-white glacier ice.
    let retainedSnow = smoothstep(4_700, 5_350, elevation)
      * smoothstep(0.22, 0.58, upward)
    color = simd_mix(color, snow, SIMD3<Float>(repeating: retainedSnow))
    let iceAmount = smoothstep(5_250, 5_850, elevation)
      * smoothstep(0.48, 0.82, upward)
    color = simd_mix(color, glacier, SIMD3<Float>(repeating: iceAmount * 0.72))
    // Alpha is a material tag here (the mesh pass always outputs opaque).
    // Terrain uses zero so the fragment shader can add metre-scale vegetation
    // variation without accidentally mottling buildings and people.
    return SIMD4<Float>(color, 0)
  }

  fileprivate static func smoothstep(_ edge0: Float, _ edge1: Float, _ value: Float) -> Float {
    let t = min(max((value - edge0) / (edge1 - edge0), 0), 1)
    return t * t * (3 - 2 * t)
  }

  fileprivate static func makeBuildings(
    device: MTLDevice,
    metadata: GyirongSceneMetadata,
    heights: [Float]
  ) throws -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    let terrain = metadata.terrain
    let event = metadata.event
    let portU = Float((event.portLongitude - terrain.west) / (terrain.east - terrain.west))
    let portV = Float((event.portLatitude - terrain.south) / (terrain.north - terrain.south))
    let datum = Float(terrain.portDatumElevationMeters)
    var vertices: [GyirongMeshVertex] = []
    var indices: [UInt32] = []

    func appendBox(
      center: SIMD3<Float>,
      size: SIMD3<Float>,
      wallColor: SIMD4<Float>,
      roofColor: SIMD4<Float>? = nil,
      horizontalRight: SIMD2<Float> = SIMD2<Float>(1, 0),
      horizontalForward: SIMD2<Float> = SIMD2<Float>(0, 1)
    ) {
      let half = size * 0.5
      let x0 = -half.x
      let x1 = half.x
      let y0 = -half.y
      let y1 = half.y
      let z0 = -half.z
      let z1 = half.z
      func orientPosition(_ local: SIMD3<Float>) -> SIMD3<Float> {
        center + SIMD3<Float>(
          horizontalRight.x * local.x + horizontalForward.x * local.z,
          local.y,
          horizontalRight.y * local.x + horizontalForward.y * local.z)
      }
      func orientNormal(_ local: SIMD3<Float>) -> SIMD3<Float> {
        simd_normalize(
          SIMD3<Float>(
            horizontalRight.x * local.x + horizontalForward.x * local.z,
            local.y,
            horizontalRight.y * local.x + horizontalForward.y * local.z))
      }
      let faces: [(SIMD3<Float>, [SIMD3<Float>], SIMD4<Float>)] = [
        (SIMD3<Float>(0, 0, 1), [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], wallColor),
        (SIMD3<Float>(0, 0, -1), [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], wallColor),
        (SIMD3<Float>(1, 0, 0), [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], wallColor),
        (SIMD3<Float>(-1, 0, 0), [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], wallColor),
        (SIMD3<Float>(0, 1, 0), [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], roofColor ?? wallColor),
        (SIMD3<Float>(0, -1, 0), [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], wallColor),
      ]
      for (normal, positions, color) in faces {
        let base = UInt32(vertices.count)
        for position in positions {
          vertices.append(
            GyirongMeshVertex(
              position: orientPosition(position),
              normal: orientNormal(normal),
              color: color))
        }
        indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
      }
    }

    func position(longitude: Double, latitude: Double, elevation: Float) -> SIMD3<Float> {
      let u = Float((longitude - terrain.west) / (terrain.east - terrain.west))
      let v = Float((latitude - terrain.south) / (terrain.north - terrain.south))
      return SIMD3<Float>(
        (v - portV) * Float(terrain.physicalHeightMeters),
        elevation - datum + portGroundY,
        -(u - portU) * Float(terrain.physicalWidthMeters) + portForwardZ)
    }

    let gateU = Float((gateLongitude - terrain.west) / (terrain.east - terrain.west))
    let gateV = Float((gateLatitude - terrain.south) / (terrain.north - terrain.south))
    let gateElevation = sampleHeight(
      heights,
      width: terrain.width,
      height: terrain.height,
      u: gateU,
      v: gateV)
    let gateCenter = position(
      longitude: gateLongitude,
      latitude: gateLatitude,
      elevation: gateElevation)

    for building in metadata.buildings {
      let footprint = building.footprintLonLat.filter { $0.count >= 2 }
      guard footprint.count >= 3 else { continue }
      let centerLongitude = footprint.reduce(0.0) { $0 + $1[0] } / Double(footprint.count)
      let centerLatitude = footprint.reduce(0.0) { $0 + $1[1] } / Double(footprint.count)
      let u = Float((centerLongitude - terrain.west) / (terrain.east - terrain.west))
      let v = Float((centerLatitude - terrain.south) / (terrain.north - terrain.south))
      let distanceFromBorderGate = simd_length(
        SIMD2<Float>(
          (v - gateV) * Float(terrain.physicalHeightMeters),
          (u - gateU) * Float(terrain.physicalWidthMeters)))
      // OSM currently contains only several tiny auxiliary footprints at the
      // gate. Replace that incomplete cluster with the calibrated facility
      // below instead of drawing both on top of one another.
      if distanceFromBorderGate < 95 { continue }
      let floorElevation = sampleHeight(
        heights,
        width: terrain.width,
        height: terrain.height,
        u: u,
        v: v)
      let wallColor = SIMD4<Float>(0.47, 0.42, 0.35, 1.25)
      let roofColor = SIMD4<Float>(0.52, 0.18, 0.10, 1.45)
      let floorPositions = footprint.map {
        position(longitude: $0[0], latitude: $0[1], elevation: floorElevation)
      }
      let roofPositions = floorPositions.map {
        $0 + SIMD3<Float>(0, building.heightMeters, 0)
      }

      let roofBase = UInt32(vertices.count)
      for roofPosition in roofPositions {
        vertices.append(
          GyirongMeshVertex(
            position: roofPosition,
            normal: SIMD3<Float>(0, 1, 0),
            color: roofColor))
      }
      for index in 1..<(roofPositions.count - 1) {
        indices.append(contentsOf: [roofBase, roofBase + UInt32(index), roofBase + UInt32(index + 1)])
      }

      for index in floorPositions.indices {
        let next = (index + 1) % floorPositions.count
        let p0 = floorPositions[index]
        let p1 = floorPositions[next]
        let p2 = roofPositions[next]
        let p3 = roofPositions[index]
        let edge = p1 - p0
        var normal = SIMD3<Float>(-edge.z, 0, edge.x)
        if simd_length_squared(normal) > 0.0001 {
          normal = simd_normalize(normal)
        } else {
          normal = SIMD3<Float>(0, 0, 1)
        }
        let base = UInt32(vertices.count)
        vertices.append(GyirongMeshVertex(position: p0, normal: normal, color: wallColor))
        vertices.append(GyirongMeshVertex(position: p1, normal: normal, color: wallColor))
        vertices.append(GyirongMeshVertex(position: p2, normal: normal, color: wallColor))
        vertices.append(GyirongMeshVertex(position: p3, normal: normal, color: wallColor))
        indices.append(contentsOf: [base, base + 1, base + 2, base, base + 2, base + 3])
      }
    }

    // Chinese-side Gyirong border facility. OSM way 904894059 supplies the
    // measured footprint centre, 83 x 51 m principal axes and facade bearing.
    // Height remains an image-derived estimate using four window rows and
    // full-size freight vehicles, not a surveyed value. The satellite and
    // map geometry places the broad Chinese apron north of the gate and
    // the river immediately along its eastern edge.
    let limestone = SIMD4<Float>(0.63, 0.60, 0.54, 1.25)
    let limestoneRoof = SIMD4<Float>(0.38, 0.18, 0.11, 1.45)
    let concrete = SIMD4<Float>(0.34, 0.36, 0.36, 1.25)
    let paving = SIMD4<Float>(0.43, 0.45, 0.44, 1)
    let windowGlass = SIMD4<Float>(0.045, 0.095, 0.12, 2)
    let paleTrim = SIMD4<Float>(0.76, 0.73, 0.67, 1.25)

    func facilityPosition(_ local: SIMD3<Float>) -> SIMD3<Float> {
      gateCenter + SIMD3<Float>(
        gateAcross.x * local.x + gateForward.x * local.z,
        local.y,
        gateAcross.y * local.x + gateForward.y * local.z)
    }

    func appendFacilityBox(
      center: SIMD3<Float>,
      size: SIMD3<Float>,
      wallColor: SIMD4<Float>,
      roofColor: SIMD4<Float>? = nil
    ) {
      appendBox(
        center: facilityPosition(center),
        size: size,
        wallColor: wallColor,
        roofColor: roofColor,
        horizontalRight: gateAcross,
        horizontalForward: gateForward)
    }

    // A solid, gently recessed structural slab closes the deliberately omitted
    // coarse DEM triangles. Its 160 x 126 m outline follows the flat inspection
    // area visible in the supplied aerial/satellite views; it is not an
    // elevated parking deck.
    appendFacilityBox(
      center: SIMD3<Float>(32, -1.7, -15),
      size: SIMD3<Float>(160, 3.4, 126),
      wallColor: concrete,
      roofColor: paving)

    // A continuous road ribbon runs through the portal and both sides of the
    // reconstructed valley. Its elevation matches the terrain-conditioning
    // grade below, so it neither floats above nor disappears into coarse DEM
    // triangles.
    let roadColor = SIMD4<Float>(0.30, 0.31, 0.30, 1)
    let roadStart = vertices.count
    let roadSegments = 47
    for segment in 0...roadSegments {
      let progress = Float(segment) / Float(roadSegments)
      let forward = simd_mix(Float(-520), Float(420), progress)
      let roadElevation = forward * 0.012 + 0.16
      for across in [-8 as Float, 8] {
        vertices.append(
          GyirongMeshVertex(
            position: facilityPosition(SIMD3<Float>(across, roadElevation, forward)),
            normal: SIMD3<Float>(0, 1, 0),
            color: roadColor))
      }
    }
    for segment in 0..<roadSegments {
      let a = UInt32(roadStart + segment * 2)
      let b = a + 1
      let c = a + 2
      let d = a + 3
      indices.append(contentsOf: [a, c, b, b, c, d])
    }

    // Main gate: approximately 84 m wide, 30 m deep and 27 m high. The portal
    // remains open beneath the central lintel, matching the frontal reference.
    appendFacilityBox(
      center: SIMD3<Float>(-25, 13.5, 0),
      size: SIMD3<Float>(34, 27, 30),
      wallColor: limestone,
      roofColor: limestoneRoof)
    appendFacilityBox(
      center: SIMD3<Float>(25, 13.5, 0),
      size: SIMD3<Float>(34, 27, 30),
      wallColor: limestone,
      roofColor: limestoneRoof)
    appendFacilityBox(
      center: SIMD3<Float>(0, 21, 0),
      size: SIMD3<Float>(16, 12, 30),
      wallColor: limestone,
      roofColor: limestoneRoof)

    // Recessed dark glazing gives all four inferred storeys a readable scale.
    // Windows are geometry rather than a texture so they remain stable in
    // stereo and at oblique viewing angles.
    let floorWindowY: [Float] = [4.1, 9.6, 15.1, 20.6]
    for wingCenterX in [-25 as Float, 25] {
      for floorY in floorWindowY {
        for column in -2...2 {
          let windowX = wingCenterX + Float(column) * 5.5
          for facadeZ in [-15.11 as Float, 15.11] {
            appendFacilityBox(
              center: SIMD3<Float>(windowX, floorY, facadeZ),
              size: SIMD3<Float>(2.8, 3.1, 0.16),
              wallColor: windowGlass)
          }
        }
        let outerX = wingCenterX < 0 ? -42.11 as Float : 42.11
        for sideColumn in -1...1 {
          appendFacilityBox(
            center: SIMD3<Float>(outerX, floorY, Float(sideColumn) * 7.1),
            size: SIMD3<Float>(0.16, 3.1, 3.0),
            wallColor: windowGlass)
        }
      }
    }
    for facadeZ in [-15.11 as Float, 15.11] {
      for column in -1...1 {
        appendFacilityBox(
          center: SIMD3<Float>(Float(column) * 4.2, 21.0, facadeZ),
          size: SIMD3<Float>(2.6, 3.0, 0.16),
          wallColor: windowGlass)
      }
    }
    // Horizontal floor bands break up the large facade and make the inferred
    // floor spacing visible even from the initial aerial viewpoint.
    for bandY in [6.7 as Float, 12.2, 17.7, 24.9] {
      for wingCenterX in [-25 as Float, 25] {
        for facadeZ in [-15.16 as Float, 15.16] {
          appendFacilityBox(
            center: SIMD3<Float>(wingCenterX, bandY, facadeZ),
            size: SIMD3<Float>(34, 0.38, 0.20),
            wallColor: paleTrim)
        }
      }
    }
    for wingCenterX in [-25 as Float, 25] {
      for offsetX in [-16.7 as Float, -8.35, 0, 8.35, 16.7] {
        for facadeZ in [-15.25 as Float, 15.25] {
          appendFacilityBox(
            center: SIMD3<Float>(wingCenterX + offsetX, 13.4, facadeZ),
            size: SIMD3<Float>(0.36, 26.5, 0.34),
            wallColor: paleTrim)
        }
      }
      for facadeZ in [-15.28 as Float, 15.28] {
        appendFacilityBox(
          center: SIMD3<Float>(wingCenterX, 27.15, facadeZ),
          size: SIMD3<Float>(34.5, 0.9, 0.46),
          wallColor: SIMD4<Float>(0.74, 0.71, 0.65, 1.25))
      }
    }
    // A raised central plaque and deep lintel match the monumental hierarchy
    // visible in the port photographs without inventing unreadable signage.
    appendFacilityBox(
      center: SIMD3<Float>(0, 24.2, -15.32),
      size: SIMD3<Float>(12.5, 2.5, 0.42),
      wallColor: SIMD4<Float>(0.42, 0.11, 0.075, 1.25))
    appendFacilityBox(
      center: SIMD3<Float>(0, 15.15, -15.34),
      size: SIMD3<Float>(16.8, 0.9, 0.5),
      wallColor: paleTrim)

    // Flat inspection apron details from the aerial view: a central approach,
    // parking-lane separators, low booths, river-edge barrier and a west-side
    // service building. These replace the former invented elevated deck.
    appendFacilityBox(
      center: SIMD3<Float>(90, 5.5, -30),
      size: SIMD3<Float>(26, 11, 24),
      wallColor: SIMD4<Float>(0.52, 0.50, 0.46, 1.25),
      roofColor: limestoneRoof)
    for laneX in [-36 as Float, -12, 12, 36, 60, 84] {
      for segment in 0..<3 {
        appendFacilityBox(
          center: SIMD3<Float>(laneX, 0.08, -20 - Float(segment) * 25),
          size: SIMD3<Float>(0.22, 0.10, 14),
          wallColor: SIMD4<Float>(0.82, 0.82, 0.76, 1))
      }
    }
    appendFacilityBox(
      center: SIMD3<Float>(12, 0.10, -30),
      size: SIMD3<Float>(0.28, 0.12, 88),
      wallColor: SIMD4<Float>(0.82, 0.64, 0.12, 1))
    for boothX in [-36 as Float, -12, 12, 36] {
      appendFacilityBox(
        center: SIMD3<Float>(boothX + 24, 1.65, -42),
        size: SIMD3<Float>(4.2, 3.3, 7.0),
        wallColor: SIMD4<Float>(0.56, 0.60, 0.60, 1.25),
        roofColor: SIMD4<Float>(0.16, 0.40, 0.56, 1.45))
    }
    for segment in 0..<4 {
      appendFacilityBox(
        center: SIMD3<Float>(-46.5, 1.0, -61 + Float(segment) * 29),
        size: SIMD3<Float>(1.1, 2.0, 22),
        wallColor: SIMD4<Float>(0.64, 0.65, 0.61, 1.25))
    }

    // Human-scale markers reconstructed from the dozens of people visible in
    // the pre-impact CCTV. They are deliberately simple 1.7 m columns: their
    // purpose is scale and distribution, not identification of individuals.
    let personColors = [
      SIMD4<Float>(0.88, 0.31, 0.12, 1),
      SIMD4<Float>(0.92, 0.69, 0.12, 1),
      SIMD4<Float>(0.16, 0.48, 0.70, 1),
    ]
    for index in 0..<52 {
      let a = Float(index)
      let xSeed = fmod(a * 0.618_033_9 + 0.17, 1)
      let zSeed = fmod(a * 0.414_213_6 + 0.41, 1)
      let markerPosition: SIMD3<Float>
      if index < 22 {
        markerPosition = SIMD3<Float>(
          -35 + xSeed * 138,
          0.85,
          -70 + zSeed * 38)
      } else if index < 40 {
        markerPosition = SIMD3<Float>(
          6 + xSeed * 14,
          0.85,
          -64 + zSeed * 44)
      } else {
        markerPosition = SIMD3<Float>(
          62 + xSeed * 38,
          0.85,
          -62 + zSeed * 45)
      }
      appendFacilityBox(
        center: markerPosition,
        size: SIMD3<Float>(0.34, 1.7, 0.34),
        wallColor: personColors[index % personColors.count])
    }

    let drawableIndexCount = indices.count
    if vertices.isEmpty {
      vertices.append(
        GyirongMeshVertex(
          position: .zero,
          normal: SIMD3<Float>(0, 1, 0),
          color: SIMD4<Float>(repeating: 0)))
    }
    if indices.isEmpty { indices.append(0) }
    guard
      let vertexBuffer = device.makeBuffer(
        bytes: vertices,
        length: vertices.count * MemoryLayout<GyirongMeshVertex>.stride,
        options: .storageModeShared),
      let indexBuffer = device.makeBuffer(
        bytes: indices,
        length: indices.count * MemoryLayout<UInt32>.stride,
        options: .storageModeShared)
    else {
      throw GyirongDebrisFlowError.resourceAllocationFailed("buildings")
    }
    vertexBuffer.label = "Gyirong OSM buildings"
    indexBuffer.label = "Gyirong OSM building indices"
    return (vertexBuffer, indexBuffer, drawableIndexCount)
  }

  fileprivate static func makeWaterMesh(
    device: MTLDevice,
    metadata: GyirongSceneMetadata
  ) throws -> (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int) {
    let width = metadata.terrain.width
    let height = metadata.terrain.height
    var vertices: [GyirongWaterVertex] = []
    vertices.reserveCapacity(width * height)
    for row in 0..<height {
      for column in 0..<width {
        vertices.append(
          GyirongWaterVertex(
            uv: SIMD2<Float>(
              Float(column) / Float(width - 1),
              Float(row) / Float(height - 1))))
      }
    }
    var indices: [UInt16] = []
    indices.reserveCapacity((width - 1) * (height - 1) * 6)
    for row in 0..<(height - 1) {
      for column in 0..<(width - 1) {
        let a = UInt16(row * width + column)
        let b = a + 1
        let c = UInt16((row + 1) * width + column)
        let d = c + 1
        indices.append(contentsOf: [a, c, b, b, c, d])
      }
    }
    guard
      let vertexBuffer = device.makeBuffer(
        bytes: vertices,
        length: vertices.count * MemoryLayout<GyirongWaterVertex>.stride,
        options: .storageModeShared),
      let indexBuffer = device.makeBuffer(
        bytes: indices,
        length: indices.count * MemoryLayout<UInt16>.stride,
        options: .storageModeShared)
    else {
      throw GyirongDebrisFlowError.resourceAllocationFailed("water mesh")
    }
    return (vertexBuffer, indexBuffer, indices.count)
  }

  fileprivate static func sampleHeight(
    _ heights: [Float],
    width: Int,
    height: Int,
    u: Float,
    v: Float
  ) -> Float {
    let x = min(max(u, 0), 1) * Float(width - 1)
    let y = min(max(v, 0), 1) * Float(height - 1)
    let x0 = Int(floor(x))
    let y0 = Int(floor(y))
    let x1 = min(x0 + 1, width - 1)
    let y1 = min(y0 + 1, height - 1)
    let fx = x - Float(x0)
    let fy = y - Float(y0)
    let a = simd_mix(heights[y0 * width + x0], heights[y0 * width + x1], fx)
    let b = simd_mix(heights[y1 * width + x0], heights[y1 * width + x1], fx)
    return simd_mix(a, b, fy)
  }

  fileprivate static func sampleHeightBicubic(
    _ heights: [Float],
    width: Int,
    height: Int,
    u: Float,
    v: Float
  ) -> Float {
    func cubic(_ p0: Float, _ p1: Float, _ p2: Float, _ p3: Float, _ t: Float) -> Float {
      let t2 = t * t
      let t3 = t2 * t
      return 0.5 * (
        2 * p1
          + (-p0 + p2) * t
          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
          + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
    }

    let x = min(max(u, 0), 1) * Float(width - 1)
    let y = min(max(v, 0), 1) * Float(height - 1)
    let centerX = Int(floor(x))
    let centerY = Int(floor(y))
    let tx = x - Float(centerX)
    let ty = y - Float(centerY)
    var rows = SIMD4<Float>(repeating: 0)
    var neighborhoodMinimum = Float.greatestFiniteMagnitude
    var neighborhoodMaximum = -Float.greatestFiniteMagnitude
    for rowOffset in -1...2 {
      let sampleY = min(max(centerY + rowOffset, 0), height - 1)
      var samples = SIMD4<Float>(repeating: 0)
      for columnOffset in -1...2 {
        let sampleX = min(max(centerX + columnOffset, 0), width - 1)
        let value = heights[sampleY * width + sampleX]
        samples[columnOffset + 1] = value
        neighborhoodMinimum = min(neighborhoodMinimum, value)
        neighborhoodMaximum = max(neighborhoodMaximum, value)
      }
      rows[rowOffset + 1] = cubic(samples.x, samples.y, samples.z, samples.w, tx)
    }
    // Catmull-Rom can overshoot at sharp Himalayan ridges. Clamping to the
    // contributing samples keeps the interpolation faithful to the source DEM.
    return min(
      max(cubic(rows.x, rows.y, rows.z, rows.w, ty), neighborhoodMinimum),
      neighborhoodMaximum)
  }

  fileprivate static func makeTerrainHeightTexture(
    device: MTLDevice,
    metadata: GyirongSceneMetadata,
    heights: [Float]
  ) throws -> MTLTexture {
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .r32Float,
      width: metadata.terrain.width,
      height: metadata.terrain.height,
      mipmapped: false)
    descriptor.storageMode = .shared
    descriptor.usage = [.shaderRead]
    guard let texture = device.makeTexture(descriptor: descriptor) else {
      throw GyirongDebrisFlowError.resourceAllocationFailed("terrain texture")
    }
    heights.withUnsafeBytes { bytes in
      texture.replace(
        region: MTLRegionMake2D(0, 0, metadata.terrain.width, metadata.terrain.height),
        mipmapLevel: 0,
        withBytes: bytes.baseAddress!,
        bytesPerRow: metadata.terrain.width * MemoryLayout<Float>.stride)
    }
    texture.label = "Gyirong DEM metres"
    return texture
  }

  fileprivate static func makeMappedRiverFlowPath(
    metadata: GyirongSceneMetadata
  ) -> [SIMD2<Float>] {
    // The generated path is already sampled at equal 65 m intervals.  Its
    // upper section follows a no-uphill DEM route from the reported source to
    // the main valley; the downstream section is OSM way 937405875 itself.
    // Do not move it laterally at runtime: the former ±700 m cross-section
    // search erased genuine bends and could move the guide onto another slope.
    var path: [SIMD2<Float>] = []
    path.reserveCapacity(metadata.flowPathUV.count)
    for point in metadata.flowPathUV {
      guard point.count >= 2 else { continue }
      let uv = SIMD2<Float>(point[0], point[1])
      guard uv.x >= 0, uv.x <= 1, uv.y >= 0, uv.y <= 1 else { continue }
      if let previous = path.last, simd_length_squared(uv - previous) < 1e-12 {
        continue
      }
      path.append(uv)
    }
    return path
  }

  /// Hydrologically condition only a narrow band around the mapped river.
  /// The source DEM is about 90–100 m per cell, so a correctly mapped canyon
  /// centreline can sample an adjacent wall and appear to climb tens of metres.
  /// A monotonically descending floor and a 220 m transition bank remove those
  /// false dams while leaving the surrounding mountain mass unchanged.
  fileprivate static func conditionTerrainForMappedRiver(
    metadata: GyirongSceneMetadata,
    heights: [Float],
    path: [SIMD2<Float>]
  ) -> [Float] {
    guard path.count >= 2 else { return heights }
    let terrain = metadata.terrain
    let physicalScale = SIMD2<Float>(
      Float(terrain.physicalWidthMeters),
      Float(terrain.physicalHeightMeters))
    var floorElevations = path.map {
      sampleHeight(
        heights,
        width: terrain.width,
        height: terrain.height,
        u: $0.x,
        v: $0.y)
    }
    for index in 1..<floorElevations.count {
      let segmentLength = simd_length((path[index] - path[index - 1]) * physicalScale)
      let minimumDrop = max(segmentLength * 0.0008, 0.02)
      floorElevations[index] = min(
        floorElevations[index],
        floorElevations[index - 1] - minimumDrop)
    }

    let eastCell = Float(terrain.physicalWidthMeters) / Float(terrain.width - 1)
    let northCell = Float(terrain.physicalHeightMeters) / Float(terrain.height - 1)
    let maximumDistance: Float = 220
    let columnRadius = Int(ceil(maximumDistance / eastCell)) + 1
    let rowRadius = Int(ceil(maximumDistance / northCell)) + 1
    var result = heights
    for index in path.indices {
      let gridX = path[index].x * Float(terrain.width - 1)
      let gridY = path[index].y * Float(terrain.height - 1)
      let centerX = Int(round(gridX))
      let centerY = Int(round(gridY))
      for row in max(centerY - rowRadius, 0)...min(centerY + rowRadius, terrain.height - 1) {
        for column in max(centerX - columnRadius, 0)...min(
          centerX + columnRadius, terrain.width - 1)
        {
          let distance = hypot(
            (Float(column) - gridX) * eastCell,
            (Float(row) - gridY) * northCell)
          guard distance <= maximumDistance else { continue }
          let bank = pow(distance / maximumDistance, 1.35) * 34
          let conditionedElevation = floorElevations[index] - 2.5 + bank
          let sampleIndex = row * terrain.width + column
          result[sampleIndex] = min(result[sampleIndex], conditionedElevation)
        }
      }
    }
    return result
  }

  /// Lower only DEM samples that intrude into the constructed border terrace.
  /// The platform is less than three source DEM cells wide, so leaving the raw
  /// samples untouched creates one giant mountain triangle through the gate.
  /// Low river cells are never raised by this operation.
  fileprivate static func conditionTerrainForPortFacility(
    metadata: GyirongSceneMetadata,
    heights: [Float]
  ) -> [Float] {
    let terrain = metadata.terrain
    let gateU = Float((gateLongitude - terrain.west) / (terrain.east - terrain.west))
    let gateV = Float((gateLatitude - terrain.south) / (terrain.north - terrain.south))
    let gateElevation = sampleHeight(
      heights,
      width: terrain.width,
      height: terrain.height,
      u: gateU,
      v: gateV)
    var result = heights
    for row in 0..<terrain.height {
      let v = Float(row) / Float(terrain.height - 1)
      for column in 0..<terrain.width {
        let u = Float(column) / Float(terrain.width - 1)
        let local = portFacilityLocalCoordinates(
          metadata: metadata,
          u: u,
          v: v)
        let sampleIndex = row * terrain.width + column

        let terraceAcrossDistance = max(max(-48 - local.x, local.x - 112), 0)
        let terraceForwardDistance = max(max(-78 - local.y, local.y - 48), 0)
        let terraceDistance = hypot(terraceAcrossDistance, terraceForwardDistance)
        // OSM geometry places the broad Chinese inspection apron north
        // (negative local-forward) of the gate. Give that approach a very
        // slight downhill grade while keeping the gate threshold fixed.
        if terraceDistance < 80 {
          let terraceRise = min(max(local.y, -78), 48) * 0.012
          let terraceElevation = gateElevation + terraceRise
          let blend = smoothstep(0, 80, terraceDistance)
          let target = simd_mix(terraceElevation, result[sampleIndex], blend)
          result[sampleIndex] = min(result[sampleIndex], target)
        }

        // The source DEM has ~90–100 m cells and samples the adjacent canyon
        // walls across the narrow border road. Cut a continuous 60 m floor and
        // a 180 m blended shoulder through the gate axis. Only excessive high
        // terrain is lowered; the mapped river and other low cells are never
        // raised.
        let roadAcrossDistance = max(abs(local.x - 18) - 30, 0)
        let roadForwardDistance = max(max(-520 - local.y, local.y - 420), 0)
        let roadDistance = hypot(roadAcrossDistance, roadForwardDistance)
        if roadDistance < 180 {
          let roadElevation = gateElevation + min(max(local.y, -520), 420) * 0.012
          let shoulderRise = pow(roadDistance / 180, 1.35) * 52
          let carvedElevation = roadElevation + shoulderRise
          let blend = smoothstep(135, 180, roadDistance)
          let target = simd_mix(carvedElevation, result[sampleIndex], blend)
          result[sampleIndex] = min(result[sampleIndex], target)
        }
      }
    }
    return result
  }

  fileprivate static func portFacilityLocalCoordinates(
    metadata: GyirongSceneMetadata,
    u: Float,
    v: Float
  ) -> SIMD2<Float> {
    let terrain = metadata.terrain
    let gateU = Float((gateLongitude - terrain.west) / (terrain.east - terrain.west))
    let gateV = Float((gateLatitude - terrain.south) / (terrain.north - terrain.south))
    let deltaScene = SIMD2<Float>(
      (v - gateV) * Float(terrain.physicalHeightMeters),
      -(u - gateU) * Float(terrain.physicalWidthMeters))
    return SIMD2<Float>(
      simd_dot(deltaScene, gateAcross),
      simd_dot(deltaScene, gateForward))
  }

  fileprivate static func distanceToPathMeters(
    uv: SIMD2<Float>,
    path: [SIMD2<Float>],
    physicalScale: SIMD2<Float>
  ) -> Float {
    guard path.count >= 2 else { return .greatestFiniteMagnitude }
    var nearestDistance = Float.greatestFiniteMagnitude
    for index in 0..<(path.count - 1) {
      let segment = (path[index + 1] - path[index]) * physicalScale
      let offset = (uv - path[index]) * physicalScale
      let t = min(
        max(simd_dot(offset, segment) / max(simd_length_squared(segment), 0.0001), 0),
        1)
      nearestDistance = min(nearestDistance, simd_length(offset - segment * t))
    }
    return nearestDistance
  }

  fileprivate static func makeFlowGuideTexture(
    device: MTLDevice,
    metadata: GyirongSceneMetadata,
    heights: [Float],
    path: [SIMD2<Float>]
  ) throws -> MTLTexture {
    let terrain = metadata.terrain
    guard path.count >= 2 else {
      throw GyirongDebrisFlowError.invalidData("flow path")
    }
    let physicalScale = SIMD2<Float>(
      Float(terrain.physicalWidthMeters),
      Float(terrain.physicalHeightMeters))
    var cumulative = [Float](repeating: 0, count: path.count)
    for index in 1..<path.count {
      cumulative[index] = cumulative[index - 1]
        + simd_length((path[index] - path[index - 1]) * physicalScale)
    }
    let totalLength = max(cumulative.last ?? 1, 1)
    let pathElevations = path.map {
      sampleHeight(
        heights,
        width: terrain.width,
        height: terrain.height,
        u: $0.x,
        v: $0.y)
    }
    var guide = [SIMD4<Float>](
      repeating: SIMD4<Float>(0, 0, -1, 0),
      count: terrain.width * terrain.height)

    for row in 0..<terrain.height {
      for column in 0..<terrain.width {
        let uv = SIMD2<Float>(
          Float(column) / Float(terrain.width - 1),
          Float(row) / Float(terrain.height - 1))
        var nearestDistance = Float.greatestFiniteMagnitude
        var nearestProgress: Float = 0
        var nearestTangent = SIMD2<Float>(-1, 0)
        var nearestValleyElevation = pathElevations[0]
        for index in 0..<(path.count - 1) {
          let a = path[index]
          let segment = path[index + 1] - a
          let segmentMeters = segment * physicalScale
          let pointMeters = (uv - a) * physicalScale
          let lengthSquared = max(simd_length_squared(segmentMeters), 0.0001)
          let t = min(max(simd_dot(pointMeters, segmentMeters) / lengthSquared, 0), 1)
          let deltaMeters = pointMeters - segmentMeters * t
          let distance = simd_length(deltaMeters)
          guard distance < nearestDistance else { continue }
          nearestDistance = distance
          nearestProgress = (
            cumulative[index] + simd_length(segmentMeters) * t
          ) / totalLength
          nearestValleyElevation = simd_mix(
            pathElevations[index], pathElevations[index + 1], t)
          nearestTangent = simd_length_squared(segmentMeters) > 0.001
            ? simd_normalize(segmentMeters)
            : nearestTangent
        }
        let ordinaryValleyWidth = simd_mix(
          Float(115), Float(175), smoothstep(0.82, 0.96, nearestProgress))
        // The supplied satellite view shows the river immediately east of a
        // broad constructed apron. Widen only the final two percent of the
        // guide so the arriving bore can overtop the bank and spread across
        // the gate/plaza instead of remaining a thin ribbon beside it.
        let portOverflow = smoothstep(0.965, 0.995, nearestProgress)
        let valleyWidth = simd_mix(ordinaryValleyWidth, Float(340), portOverflow)
        let heightAboveValley = max(
          heights[row * terrain.width + column] - nearestValleyElevation,
          0)
        let horizontalMask = exp(-pow(nearestDistance / valleyWidth, 2))
        let verticalMask = exp(-pow(max(heightAboveValley - 18, 0) / 58, 2))
        let corridor = horizontalMask * verticalMask
        guide[row * terrain.width + column] = SIMD4<Float>(
          corridor,
          nearestProgress,
          nearestTangent.x,
          nearestTangent.y)
      }
    }

    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba32Float,
      width: terrain.width,
      height: terrain.height,
      mipmapped: false)
    descriptor.storageMode = .shared
    descriptor.usage = [.shaderRead]
    guard let texture = device.makeTexture(descriptor: descriptor) else {
      throw GyirongDebrisFlowError.resourceAllocationFailed("flow guide texture")
    }
    guide.withUnsafeBytes { bytes in
      texture.replace(
        region: MTLRegionMake2D(0, 0, terrain.width, terrain.height),
        mipmapLevel: 0,
        withBytes: bytes.baseAddress!,
        bytesPerRow: terrain.width * MemoryLayout<SIMD4<Float>>.stride)
    }
    texture.label = "Gyirong DEM-aware flood corridor"
    return texture
  }

  fileprivate static func makeFlowPathBuffer(
    device: MTLDevice,
    path: [SIMD2<Float>]
  ) throws -> MTLBuffer {
    guard path.count >= 2,
      let buffer = device.makeBuffer(
        bytes: path,
        length: path.count * MemoryLayout<SIMD2<Float>>.stride,
        options: .storageModeShared)
    else {
      throw GyirongDebrisFlowError.resourceAllocationFailed("flow path buffer")
    }
    buffer.label = "Gyirong flood particle path"
    return buffer
  }

  fileprivate static func makeWaterStateTexture(
    device: MTLDevice,
    metadata: GyirongSceneMetadata,
    label: String
  ) throws -> MTLTexture {
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
      pixelFormat: .rgba16Float,
      width: metadata.terrain.width,
      height: metadata.terrain.height,
      mipmapped: false)
    descriptor.storageMode = .private
    descriptor.usage = [.shaderRead, .shaderWrite]
    guard let texture = device.makeTexture(descriptor: descriptor) else {
      throw GyirongDebrisFlowError.resourceAllocationFailed(label)
    }
    texture.label = label
    return texture
  }

  fileprivate static func makeMeshPipeline(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    try makeRenderPipeline(
      device: device,
      library: library,
      vertexName: "gyirongMeshVertex",
      fragmentName: "gyirongMeshFragment",
      maxViewCount: maxViewCount,
      blending: false)
  }

  fileprivate static func makeSkyPipeline(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    try makeRenderPipeline(
      device: device,
      library: library,
      vertexName: "gyirongSkyVertex",
      fragmentName: "gyirongSkyFragment",
      maxViewCount: maxViewCount,
      blending: false)
  }

  fileprivate static func makeWaterPipeline(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    try makeRenderPipeline(
      device: device,
      library: library,
      vertexName: "gyirongWaterVertex",
      fragmentName: "gyirongWaterFragment",
      maxViewCount: maxViewCount,
      blending: true)
  }

  fileprivate static func makeParticlePipeline(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    try makeRenderPipeline(
      device: device,
      library: library,
      vertexName: "gyirongParticleVertex",
      fragmentName: "gyirongParticleFragment",
      maxViewCount: maxViewCount,
      blending: false)
  }

  fileprivate static func makeRenderPipeline(
    device: MTLDevice,
    library: MTLLibrary,
    vertexName: String,
    fragmentName: String,
    maxViewCount: Int,
    blending: Bool,
    additive: Bool = false
  ) throws -> MTLRenderPipelineState {
    guard let vertexFunction = library.makeFunction(name: vertexName) else {
      throw GyirongDebrisFlowError.missingFunction(vertexName)
    }
    guard let fragmentFunction = library.makeFunction(name: fragmentName) else {
      throw GyirongDebrisFlowError.missingFunction(fragmentName)
    }
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = vertexFunction
    descriptor.fragmentFunction = fragmentFunction
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)
    if blending {
      let attachment = descriptor.colorAttachments[0]!
      attachment.isBlendingEnabled = true
      attachment.rgbBlendOperation = .add
      attachment.alphaBlendOperation = .add
      attachment.sourceRGBBlendFactor = .sourceAlpha
      attachment.sourceAlphaBlendFactor = .one
      attachment.destinationRGBBlendFactor = additive ? .one : .oneMinusSourceAlpha
      attachment.destinationAlphaBlendFactor = additive ? .one : .oneMinusSourceAlpha
    }
    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  fileprivate static func makeDepthState(
    device: MTLDevice,
    writesDepth: Bool
  ) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = writesDepth
    return device.makeDepthStencilState(descriptor: descriptor)!
  }

  fileprivate static func makeSkyDepthState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: descriptor)!
  }

  fileprivate static func makeComputePipeline(
    device: MTLDevice,
    library: MTLLibrary,
    name: String
  ) throws -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else {
      throw GyirongDebrisFlowError.missingFunction(name)
    }
    return try device.makeComputePipelineState(function: function)
  }
}
