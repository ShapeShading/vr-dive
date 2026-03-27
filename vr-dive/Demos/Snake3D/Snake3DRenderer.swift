import Metal
import simd

// MARK: - Pipeline creation helpers (private typealias)
private typealias MeshBuffers = (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int)

// MARK: - Snake3DRenderer

final class Snake3DRenderer: VisualPatternController {

  // MARK: Protocol
  let identifier: VisualPatternKind = .snake3D
  let preferredClearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.05, alpha: 1)

  // MARK: Metal
  private let device: MTLDevice
  private let maxViewCount: Int
  private let bodyPipelineState: MTLRenderPipelineState
  private let foodPipelineState: MTLRenderPipelineState
  private let borderPipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState

  // Shared cube geometry
  private let cubeMesh: MeshBuffers

  // Instance buffers
  private var bodyBuffer: MTLBuffer
  private var foodBuffer: MTLBuffer
  private let maxSegments = 512
  private let maxFoods = 8

  // Border line buffer (static)
  private var borderBuffer: MTLBuffer!
  private var borderVertexCount: Int = 0

  // MARK: Game
  private let gameLogic: Snake3DGameLogic
  private weak var gameManager: GameManager?

  // MARK: World Rotation State
  // The world quaternion transforms all geometry so the snake head always
  // points toward the camera (-Z).  When the player turns, we compose an
  // *inverse* rotation onto worldQuat so the rendered world rotates opposite
  // to the snake's direction change.
  private var currentWorldQuat: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)  // identity
  private var rotationStartQuat: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
  private var targetWorldQuat: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
  private var rotationProgress: Float = 1.0  // 1.0 = done
  private let rotationDuration: Float = 0.3

  // MARK: Input debounce
  private var lastDpadTime: TimeInterval = 0
  private var lastRollTime: TimeInterval = 0
  private var lastOptionsTime: TimeInterval = 0
  private let dpadDelay: TimeInterval = 0.31  // slightly above move interval
  private let rollDelay: TimeInterval = 0.31
  private let optionsDelay: TimeInterval = 0.5

  // MARK: Fixed anchor in view space: slightly in front of origin
  private let headAnchor = SIMD3<Float>(0.0, 0.0, -0.6)
  private let displayRight = SIMD3<Float>(1.0, 0.0, 0.0)
  private let displayUp = SIMD3<Float>(0.0, 1.0, 0.0)
  private let displayForward = SIMD3<Float>(0.0, 0.0, -1.0)

  // MARK: Last update time
  private var lastUpdateTime: TimeInterval = 0

  // MARK: - Init

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int, gameManager: GameManager) {
    self.device = device
    self.maxViewCount = max(1, maxViewCount)
    self.gameManager = gameManager
    self.gameLogic = Snake3DGameLogic()

    cubeMesh = Snake3DRenderer.makeCubeGeometry(device: device)

    bodyBuffer = device.makeBuffer(
      length: MemoryLayout<SnakeSegmentInstance>.stride * 512,
      options: .storageModeShared)!

    foodBuffer = device.makeBuffer(
      length: MemoryLayout<FoodInstance>.stride * 8,
      options: .storageModeShared)!

    bodyPipelineState = try! Snake3DRenderer.makeBodyPipeline(
      device: device, library: library, maxViewCount: max(1, maxViewCount))
    foodPipelineState = try! Snake3DRenderer.makeFoodPipeline(
      device: device, library: library, maxViewCount: max(1, maxViewCount))
    borderPipelineState = try! Snake3DRenderer.makeBorderPipeline(
      device: device, library: library, maxViewCount: max(1, maxViewCount))
    depthStencilState = Snake3DRenderer.makeDepthStencilState(device: device)

    buildBorderGeometry()
  }

  // MARK: - VisualPatternController Protocol

  func updateSimulation(_ context: PatternSimulationContext) {
    let currentTime = TimeInterval(context.time)
    let deltaTime = Float(max(0, currentTime - lastUpdateTime))
    lastUpdateTime = currentTime

    // Advance rotation interpolation
    if rotationProgress < 1.0 {
      rotationProgress = min(rotationProgress + deltaTime / rotationDuration, 1.0)
      currentWorldQuat = simd_slerp(rotationStartQuat, targetWorldQuat, rotationProgress)
    }

    // Handle input
    processInput(currentTime: currentTime)

    // Update game tick
    gameLogic.update(currentTime: currentTime)

    // Upload instance data
    uploadBodyInstances()
    uploadFoodInstances()
  }

  func resetToInitialState() {
    gameLogic.reset()
    currentWorldQuat = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    rotationStartQuat = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    targetWorldQuat = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    rotationProgress = 1.0
    lastUpdateTime = 0
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)

    context.applyViewConfiguration(on: encoder)

    var uniforms = makeSceneUniforms(
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount))
    var viewMatrices = context.viewData.viewProjectionMatrices
    if viewMatrices.isEmpty { viewMatrices = [matrix_identity_float4x4] }

    // Draw border
    drawBorder(encoder: encoder, uniforms: &uniforms, viewMatrices: viewMatrices)

    // Draw snake body
    let segCount = gameLogic.state.segments.count
    if segCount > 0 {
      encoder.setRenderPipelineState(bodyPipelineState)
      encoder.setVertexBuffer(cubeMesh.vertexBuffer, offset: 0, index: 0)
      encoder.setVertexBuffer(bodyBuffer, offset: 0, index: 1)
      encoder.setVertexBytes(&uniforms, length: MemoryLayout<Snake3DSceneUniforms>.stride, index: 2)
      viewMatrices.withUnsafeBytes { raw in
        if let base = raw.baseAddress, raw.count > 0 {
          encoder.setVertexBytes(base, length: raw.count, index: 3)
        }
      }
      encoder.setFragmentBytes(
        &uniforms, length: MemoryLayout<Snake3DSceneUniforms>.stride, index: 0)
      encoder.drawIndexedPrimitives(
        type: .triangle,
        indexCount: cubeMesh.indexCount,
        indexType: .uint16,
        indexBuffer: cubeMesh.indexBuffer,
        indexBufferOffset: 0,
        instanceCount: min(segCount, maxSegments))
    }

    // Draw food
    let foodCount = gameLogic.state.foods.count
    if foodCount > 0 {
      encoder.setRenderPipelineState(foodPipelineState)
      encoder.setVertexBuffer(cubeMesh.vertexBuffer, offset: 0, index: 0)
      encoder.setVertexBuffer(foodBuffer, offset: 0, index: 1)
      encoder.setVertexBytes(&uniforms, length: MemoryLayout<Snake3DSceneUniforms>.stride, index: 2)
      viewMatrices.withUnsafeBytes { raw in
        if let base = raw.baseAddress, raw.count > 0 {
          encoder.setVertexBytes(base, length: raw.count, index: 3)
        }
      }
      encoder.setFragmentBytes(
        &uniforms, length: MemoryLayout<Snake3DSceneUniforms>.stride, index: 0)
      encoder.drawIndexedPrimitives(
        type: .triangle,
        indexCount: cubeMesh.indexCount,
        indexType: .uint16,
        indexBuffer: cubeMesh.indexBuffer,
        indexBufferOffset: 0,
        instanceCount: min(foodCount, maxFoods))
    }
  }

  // MARK: - Input Processing

  private func processInput(currentTime: TimeInterval) {
    guard let manager = gameManager else { return }
    let input = manager.getTetrisInput()

    if rotationProgress < 0.98 {
      return
    }

    // Options button = reset
    if input.buttonTriangle && currentTime - lastOptionsTime > optionsDelay {
      lastOptionsTime = currentTime
    }

    // D-pad turns are always interpreted in camera/display space.
    let dpadCanFire = currentTime - lastDpadTime > dpadDelay
    if dpadCanFire, !gameLogic.state.isGameOver {
      var turned = false
      if input.dpadUp {
        applyDisplayRotation(axis: displayRight, angle: -.pi / 2, updatesDirection: true)
        turned = true
      } else if input.dpadDown {
        applyDisplayRotation(axis: displayRight, angle: .pi / 2, updatesDirection: true)
        turned = true
      } else if input.dpadLeft {
        applyDisplayRotation(axis: displayUp, angle: .pi / 2, updatesDirection: true)
        turned = true
      } else if input.dpadRight {
        applyDisplayRotation(axis: displayUp, angle: -.pi / 2, updatesDirection: true)
        turned = true
      }
      if turned { lastDpadTime = currentTime }
    }

    // □ / ○ only roll the snake around the camera forward axis.
    let rollCanFire = currentTime - lastRollTime > rollDelay
    if rollCanFire, !gameLogic.state.isGameOver {
      if input.buttonSquare {
        applyDisplayRotation(axis: displayForward, angle: -.pi / 2, updatesDirection: false)
        lastRollTime = currentTime
      } else if input.buttonCircle {
        applyDisplayRotation(axis: displayForward, angle: .pi / 2, updatesDirection: false)
        lastRollTime = currentTime
      }
    }

    // Reset
    let optionsPressed = input.buttonCross  // □ for now; Options not exposed in TetrisInput
    // We reuse buttonCross (×) as game-over restart trigger since Options isn't in current input struct
    if gameLogic.state.isGameOver && optionsPressed && currentTime - lastOptionsTime > optionsDelay
    {
      resetToInitialState()
      lastOptionsTime = currentTime
    }
  }

  // MARK: - Rotation Logic

  private func applyDisplayRotation(axis: SIMD3<Float>, angle: Float, updatesDirection: Bool) {
    let displayRotation = simd_quatf(angle: angle, axis: simd_normalize(axis))
    rotationStartQuat = currentWorldQuat
    targetWorldQuat = simd_normalize(displayRotation * targetWorldQuat)
    rotationProgress = 0.0

    if updatesDirection {
      let worldForward = simd_inverse(targetWorldQuat).act(displayForward)
      gameLogic.requestTurn(to: nearestDirection(for: worldForward))
    }
  }

  private func nearestDirection(for vector: SIMD3<Float>) -> SnakeDirection {
    let dirs: [(SnakeDirection, SIMD3<Float>)] = [
      (.posX, SIMD3<Float>(1, 0, 0)),
      (.negX, SIMD3<Float>(-1, 0, 0)),
      (.posY, SIMD3<Float>(0, 1, 0)),
      (.negY, SIMD3<Float>(0, -1, 0)),
      (.posZ, SIMD3<Float>(0, 0, 1)),
      (.negZ, SIMD3<Float>(0, 0, -1)),
    ]
    let best = dirs.max { a, b in
      simd_dot(a.1, vector) < simd_dot(b.1, vector)
    }
    return best?.0 ?? .posZ
  }

  // MARK: - Scene Uniforms

  private func makeSceneUniforms(time: Float, layerCount: UInt32) -> Snake3DSceneUniforms {
    let rotMat = simd_matrix4x4(currentWorldQuat)

    // The anchor translation moves all geometry so the snake head ends at headAnchor.
    // headWorldPos = gridToWorld(head grid pos)
    let headGrid = gameLogic.state.segments.first ?? SIMD3<Int>(10, 10, 10)
    let headWorld = gridToWorld(headGrid)
    // After rotation: rotated head = rotMat * headWorld
    // We want rotated head + anchorTranslation = headAnchor
    // => anchorTranslation = headAnchor - (rotMat * headWorld).xyz
    let rotatedHead = (rotMat * SIMD4<Float>(headWorld.x, headWorld.y, headWorld.z, 1.0)).xyz
    let anchorTranslation = headAnchor - rotatedHead

    return Snake3DSceneUniforms(
      worldRotation: rotMat,
      anchorTranslation: anchorTranslation,
      time: time,
      layerCount: layerCount)
  }

  // MARK: - Grid to World Conversion

  private func gridToWorld(_ grid: SIMD3<Int>) -> SIMD3<Float> {
    let cell = Snake3DState.cellSize
    let g = Float(Snake3DState.gridSize)
    // Centre the grid at origin
    return SIMD3<Float>(
      (Float(grid.x) - g * 0.5 + 0.5) * cell,
      (Float(grid.y) - g * 0.5 + 0.5) * cell,
      (Float(grid.z) - g * 0.5 + 0.5) * cell
    )
  }

  // MARK: - Instance Data Upload

  private func uploadBodyInstances() {
    let segs = gameLogic.getSegmentPositions()
    let ptr = bodyBuffer.contents().assumingMemoryBound(to: SnakeSegmentInstance.self)
    // Head color = bright green, tail dims
    let headColor = SIMD3<Float>(0.1, 1.0, 0.2)
    for (i, seg) in segs.prefix(maxSegments).enumerated() {
      let worldPos = gridToWorld(seg.gridPos)
      let t = seg.normalizedIndex  // 0=head, 1=tail
      // Interpolate from bright green → dark green
      let color = mix(headColor, headColor * 0.2, t)
      ptr[i] = SnakeSegmentInstance(
        position: worldPos,
        color: color,
        normalizedIndex: t,
        scale: Snake3DState.blockSize)
    }
  }

  private func uploadFoodInstances() {
    let foods = gameLogic.getFoodPositions()
    let ptr = foodBuffer.contents().assumingMemoryBound(to: FoodInstance.self)
    for (i, grid) in foods.prefix(maxFoods).enumerated() {
      ptr[i] = FoodInstance(
        position: gridToWorld(grid),
        phase: Float(i) * 1.3)
    }
  }

  // MARK: - Border Geometry

  private func buildBorderGeometry() {
    let g = Float(Snake3DState.gridSize)
    let c = Snake3DState.cellSize
    // The grid spans [0, g) cells; in world space centred at 0 that's [-g/2, g/2] * cellSize
    let half = g * c * 0.5

    // 8 corners of the bounding box
    let corners: [SIMD3<Float>] = [
      SIMD3(-half, -half, -half), SIMD3(half, -half, -half),
      SIMD3(half, half, -half), SIMD3(-half, half, -half),
      SIMD3(-half, -half, half), SIMD3(half, -half, half),
      SIMD3(half, half, half), SIMD3(-half, half, half),
    ]

    // 12 edges = 24 line-segment endpoints
    let edges: [(Int, Int)] = [
      (0, 1), (1, 2), (2, 3), (3, 0),  // back face
      (4, 5), (5, 6), (6, 7), (7, 4),  // front face
      (0, 4), (1, 5), (2, 6), (3, 7),  // connecting edges
    ]

    var verts: [SnakeBorderVertex] = []
    for (a, b) in edges {
      verts.append(SnakeBorderVertex(position: corners[a]))
      verts.append(SnakeBorderVertex(position: corners[b]))
    }

    borderVertexCount = verts.count
    borderBuffer = device.makeBuffer(
      bytes: verts,
      length: MemoryLayout<SnakeBorderVertex>.stride * verts.count,
      options: .storageModeShared)!
  }

  private func drawBorder(
    encoder: MTLRenderCommandEncoder,
    uniforms: inout Snake3DSceneUniforms,
    viewMatrices: [simd_float4x4]
  ) {
    guard borderVertexCount > 0 else { return }
    encoder.setRenderPipelineState(borderPipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setVertexBuffer(borderBuffer, offset: 0, index: 0)
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<Snake3DSceneUniforms>.stride, index: 1)
    viewMatrices.withUnsafeBytes { raw in
      if let base = raw.baseAddress, raw.count > 0 {
        encoder.setVertexBytes(base, length: raw.count, index: 2)
      }
    }
    encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: borderVertexCount)
  }
}

// MARK: - SIMD helpers

private func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
  return a + (b - a) * t
}

extension SIMD4<Float> {
  fileprivate var xyz: SIMD3<Float> { SIMD3<Float>(x, y, z) }
}

// MARK: - Pipeline & Geometry Creation

extension Snake3DRenderer {

  fileprivate static func makeBodyPipeline(
    device: MTLDevice, library: MTLLibrary, maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "snake3DBodyVertexShader")
    desc.fragmentFunction = library.makeFunction(name: "snake3DBodyFragmentShader")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.depthAttachmentPixelFormat = .depth32Float
    desc.inputPrimitiveTopology = .triangle
    desc.maxVertexAmplificationCount = max(maxViewCount, 1)

    let vd = MTLVertexDescriptor()
    vd.attributes[0].format = .float3
    vd.attributes[0].offset = 0
    vd.attributes[0].bufferIndex = 0
    vd.attributes[1].format = .float3
    vd.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vd.attributes[1].bufferIndex = 0
    vd.layouts[0].stride = MemoryLayout<SnakeMeshVertex>.stride
    desc.vertexDescriptor = vd

    return try device.makeRenderPipelineState(descriptor: desc)
  }

  fileprivate static func makeFoodPipeline(
    device: MTLDevice, library: MTLLibrary, maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "snake3DFoodVertexShader")
    desc.fragmentFunction = library.makeFunction(name: "snake3DFoodFragmentShader")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.depthAttachmentPixelFormat = .depth32Float
    desc.inputPrimitiveTopology = .triangle
    desc.maxVertexAmplificationCount = max(maxViewCount, 1)

    let vd = MTLVertexDescriptor()
    vd.attributes[0].format = .float3
    vd.attributes[0].offset = 0
    vd.attributes[0].bufferIndex = 0
    vd.attributes[1].format = .float3
    vd.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vd.attributes[1].bufferIndex = 0
    vd.layouts[0].stride = MemoryLayout<SnakeMeshVertex>.stride
    desc.vertexDescriptor = vd

    return try device.makeRenderPipelineState(descriptor: desc)
  }

  fileprivate static func makeBorderPipeline(
    device: MTLDevice, library: MTLLibrary, maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let desc = MTLRenderPipelineDescriptor()
    desc.vertexFunction = library.makeFunction(name: "snake3DBorderVertexShader")
    desc.fragmentFunction = library.makeFunction(name: "snake3DBorderFragmentShader")
    desc.colorAttachments[0].pixelFormat = .rgba16Float
    desc.colorAttachments[0].isBlendingEnabled = true
    desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
    desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
    desc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
    desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
    desc.depthAttachmentPixelFormat = .depth32Float
    desc.inputPrimitiveTopology = .line
    desc.maxVertexAmplificationCount = max(maxViewCount, 1)
    return try device.makeRenderPipelineState(descriptor: desc)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let desc = MTLDepthStencilDescriptor()
    desc.depthCompareFunction = .greater
    desc.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: desc)!
  }

  fileprivate static func makeCubeGeometry(device: MTLDevice) -> MeshBuffers {
    let h: Float = 0.5
    let vertices: [SnakeMeshVertex] = [
      // Front (+Z)
      SnakeMeshVertex(position: [-h, -h, h], normal: [0, 0, 1]),
      SnakeMeshVertex(position: [h, -h, h], normal: [0, 0, 1]),
      SnakeMeshVertex(position: [h, h, h], normal: [0, 0, 1]),
      SnakeMeshVertex(position: [-h, h, h], normal: [0, 0, 1]),
      // Back (-Z)
      SnakeMeshVertex(position: [-h, -h, -h], normal: [0, 0, -1]),
      SnakeMeshVertex(position: [h, -h, -h], normal: [0, 0, -1]),
      SnakeMeshVertex(position: [h, h, -h], normal: [0, 0, -1]),
      SnakeMeshVertex(position: [-h, h, -h], normal: [0, 0, -1]),
      // Left (-X)
      SnakeMeshVertex(position: [-h, -h, -h], normal: [-1, 0, 0]),
      SnakeMeshVertex(position: [-h, -h, h], normal: [-1, 0, 0]),
      SnakeMeshVertex(position: [-h, h, h], normal: [-1, 0, 0]),
      SnakeMeshVertex(position: [-h, h, -h], normal: [-1, 0, 0]),
      // Right (+X)
      SnakeMeshVertex(position: [h, -h, -h], normal: [1, 0, 0]),
      SnakeMeshVertex(position: [h, -h, h], normal: [1, 0, 0]),
      SnakeMeshVertex(position: [h, h, h], normal: [1, 0, 0]),
      SnakeMeshVertex(position: [h, h, -h], normal: [1, 0, 0]),
      // Top (+Y)
      SnakeMeshVertex(position: [-h, h, h], normal: [0, 1, 0]),
      SnakeMeshVertex(position: [h, h, h], normal: [0, 1, 0]),
      SnakeMeshVertex(position: [h, h, -h], normal: [0, 1, 0]),
      SnakeMeshVertex(position: [-h, h, -h], normal: [0, 1, 0]),
      // Bottom (-Y)
      SnakeMeshVertex(position: [-h, -h, h], normal: [0, -1, 0]),
      SnakeMeshVertex(position: [h, -h, h], normal: [0, -1, 0]),
      SnakeMeshVertex(position: [h, -h, -h], normal: [0, -1, 0]),
      SnakeMeshVertex(position: [-h, -h, -h], normal: [0, -1, 0]),
    ]

    let indices: [UInt16] = [
      0, 1, 2, 0, 2, 3,  // Front
      4, 5, 6, 4, 6, 7,  // Back
      8, 9, 10, 8, 10, 11,  // Left
      12, 13, 14, 12, 14, 15,  // Right
      16, 17, 18, 16, 18, 19,  // Top
      20, 21, 22, 20, 22, 23,  // Bottom
    ]

    let vb = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<SnakeMeshVertex>.stride * vertices.count,
      options: .storageModeShared)!
    let ib = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: .storageModeShared)!

    return (vb, ib, indices.count)
  }
}
