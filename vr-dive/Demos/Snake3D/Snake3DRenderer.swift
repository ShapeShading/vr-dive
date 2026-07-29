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
  private let transparentDepthStencilState: MTLDepthStencilState

  // Shared cube geometry
  private let cubeMesh: MeshBuffers

  // Instance buffers
  private var bodyBuffer: MTLBuffer
  private var foodBuffer: MTLBuffer
  private var guideBuffer: MTLBuffer
  private let maxSegments = 512
  private let maxFoods = 4096
  private let maxGuideInstances = Snake3DState.gridSize * 8

  // Border line buffer (static)
  private var borderBuffer: MTLBuffer!
  private var borderInstanceCount: Int = 0
  private var guideInstanceCount: Int = 0

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
  private let rotationDuration: Float = 1.2

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
  // Game-local time: advances only when updateSimulation runs (pauses when paused)
  private var gameTime: TimeInterval = 0

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
      length: MemoryLayout<FoodInstance>.stride * maxFoods,
      options: .storageModeShared)!

    guideBuffer = device.makeBuffer(
      length: MemoryLayout<SnakeGuideInstance>.stride * maxGuideInstances,
      options: .storageModeShared)!

    bodyPipelineState = try! Snake3DRenderer.makeBodyPipeline(
      device: device, library: library, maxViewCount: max(1, maxViewCount))
    foodPipelineState = try! Snake3DRenderer.makeFoodPipeline(
      device: device, library: library, maxViewCount: max(1, maxViewCount))
    borderPipelineState = try! Snake3DRenderer.makeBorderPipeline(
      device: device, library: library, maxViewCount: max(1, maxViewCount))
    depthStencilState = Snake3DRenderer.makeDepthStencilState(device: device)
    transparentDepthStencilState = Snake3DRenderer.makeTransparentDepthStencilState(device: device)

    buildBorderGeometry()
    updateGuideGeometry()
  }

  // MARK: - VisualPatternController Protocol

  func updateSimulation(_ context: PatternSimulationContext) {
    let currentTime = TimeInterval(context.time)
    let deltaTime = Float(max(0, currentTime - lastUpdateTime))
    lastUpdateTime = currentTime
    gameTime += TimeInterval(deltaTime)

    // Advance rotation interpolation
    if rotationProgress < 1.0 {
      rotationProgress = min(rotationProgress + deltaTime / rotationDuration, 1.0)
      let easedProgress = smoothRotationProgress(rotationProgress)
      currentWorldQuat = simd_slerp(rotationStartQuat, targetWorldQuat, easedProgress)
    }

    // Handle input
    processInput(currentTime: currentTime)

    // Update game tick
    gameLogic.update(currentTime: gameTime)

    // Upload instance data
    uploadBodyInstances()
    uploadFoodInstances()
    updateGuideGeometry()
  }

  func resetToInitialState() {
    gameLogic.reset()
    gameTime = 0
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

    // Draw opaque helper geometry first so the snake remains the primary visual.
    drawBorder(encoder: encoder, uniforms: &uniforms, viewMatrices: viewMatrices)
    drawGuides(encoder: encoder, uniforms: &uniforms, viewMatrices: viewMatrices)

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
        applyDisplayRotation(axis: displayUp, angle: -.pi / 2, updatesDirection: true)
        turned = true
      } else if input.dpadRight {
        applyDisplayRotation(axis: displayUp, angle: .pi / 2, updatesDirection: true)
        turned = true
      }
      if turned { lastDpadTime = currentTime }
    }

    // □ / ○ only roll the snake around the camera forward axis.
    let rollCanFire = currentTime - lastRollTime > rollDelay
    if rollCanFire, !gameLogic.state.isGameOver {
      if input.buttonSquare {
        applyDisplayRotation(axis: displayForward, angle: .pi / 2, updatesDirection: false)
        lastRollTime = currentTime
      } else if input.buttonCircle {
        applyDisplayRotation(axis: displayForward, angle: -.pi / 2, updatesDirection: false)
        lastRollTime = currentTime
      }
    }

    // Reset
    let optionsPressed = input.buttonCross
    // We reuse buttonCross (×) as game-over restart trigger since Options isn't in current input struct
    if gameLogic.state.isGameOver && optionsPressed && currentTime - lastOptionsTime > optionsDelay
    {
      resetToInitialState()
      lastOptionsTime = currentTime
    }

    // R1 = boost forward speed
    gameLogic.isBoosting = input.buttonR1
  }

  // MARK: - Rotation Logic

  private func smoothRotationProgress(_ t: Float) -> Float {
    let clamped = max(0, min(1, t))
    return clamped * clamped * (3 - 2 * clamped)
  }

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
    let headGrid = gameLogic.getInterpolatedHeadPosition(currentTime: TimeInterval(time))
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
    return gridToWorld(SIMD3<Float>(Float(grid.x), Float(grid.y), Float(grid.z)))
  }

  private func gridToWorld(_ grid: SIMD3<Float>) -> SIMD3<Float> {
    let cell = Snake3DState.cellSize
    let g = Float(Snake3DState.gridSize)
    return SIMD3<Float>(
      (grid.x - g * 0.5 + 0.5) * cell,
      (grid.y - g * 0.5 + 0.5) * cell,
      (grid.z - g * 0.5 + 0.5) * cell
    )
  }

  // MARK: - Instance Data Upload

  private func uploadBodyInstances() {
    let segs = gameLogic.getSegmentPositions()
    let ptr = bodyBuffer.contents().assumingMemoryBound(to: SnakeSegmentInstance.self)
    let headColor = SIMD3<Float>(0.1, 1.0, 0.2)
    for (i, seg) in segs.prefix(maxSegments).enumerated() {
      let worldPos = gridToWorld(seg.gridPos)
      let t = seg.normalizedIndex
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
    let head =
      gameLogic.state.segments.first
      ?? SIMD3<Int>(Snake3DState.gridSize / 2, Snake3DState.gridSize / 2, Snake3DState.gridSize / 2)
    let direction = gameLogic.state.pendingDirection ?? gameLogic.state.direction
    let ptr = foodBuffer.contents().assumingMemoryBound(to: FoodInstance.self)
    for (i, grid) in foods.prefix(maxFoods).enumerated() {
      let isHit = isGuideDashHit(cell: grid, head: head, direction: direction)
      ptr[i] = FoodInstance(
        position: gridToWorld(grid),
        phase: Float(i) * 1.3,
        hit: isHit ? 1.0 : 0.0,
        colorIndex: Float(i % 4))
    }
  }

  private func isGuideDashHit(cell: SIMD3<Int>, head: SIMD3<Int>, direction: SnakeDirection) -> Bool
  {
    let period = 4
    // A cell is "on a dash" if any of the 4 spiral rails covers it.
    // Rail i covers positions with (pos - headVal) ≡ i*2 (mod period) in forward direction,
    // or (pos % period) == i*2 for perpendicular axes.
    func anyRailHits(value: Int, headValue: Int, isForward: Bool, negDir: Bool) -> Bool {
      for railIdx in 0..<4 {
        let offset = railIdx * 2
        if isForward {
          if negDir {
            let diff = headValue - value
            if diff >= 0 && diff % period == offset { return true }
          } else {
            let diff = value - headValue
            if diff >= 0 && diff % period == offset { return true }
          }
        } else {
          if value % period == offset { return true }
        }
      }
      return false
    }

    let xForward = direction == .posX || direction == .negX
    let yForward = direction == .posY || direction == .negY
    let zForward = direction == .posZ || direction == .negZ

    let xOnDash = anyRailHits(
      value: cell.x, headValue: head.x, isForward: xForward, negDir: direction == .negX)
    let yOnDash = anyRailHits(
      value: cell.y, headValue: head.y, isForward: yForward, negDir: direction == .negY)
    let zOnDash = anyRailHits(
      value: cell.z, headValue: head.z, isForward: zForward, negDir: direction == .negZ)

    let onXLine = cell.y == head.y && cell.z == head.z && xOnDash
    let onYLine = cell.x == head.x && cell.z == head.z && yOnDash
    let onZLine = cell.x == head.x && cell.y == head.y && zOnDash
    return onXLine || onYLine || onZLine
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

    var instances: [SnakeGuideInstance] = []
    let borderColor = SIMD4<Float>(0.28, 0.32, 0.38, 1.0)
    let borderThickness: Float = 0.01

    func addSegment(_ a: SIMD3<Float>, _ b: SIMD3<Float>, color: SIMD4<Float>, thickness: Float) {
      let center = (a + b) * 0.5
      let delta = b - a
      let scale = SIMD3<Float>(
        max(abs(delta.x), thickness),
        max(abs(delta.y), thickness),
        max(abs(delta.z), thickness)
      )
      instances.append(SnakeGuideInstance(position: center, scale: scale, color: color))
    }

    for (a, b) in edges {
      addSegment(corners[a], corners[b], color: borderColor, thickness: borderThickness)
    }

    borderInstanceCount = instances.count
    borderBuffer = device.makeBuffer(
      bytes: instances,
      length: MemoryLayout<SnakeGuideInstance>.stride * instances.count,
      options: .storageModeShared)!
  }

  private func updateGuideGeometry() {
    let g = Float(Snake3DState.gridSize)
    let xHintColor = SIMD4<Float>(0.46, 0.34, 0.18, 1.0)  // darker hint axis (X control cue)
    let yHintColor = SIMD4<Float>(0.72, 0.96, 0.78, 1.0)  // brighter hint axis (Y control cue)
    let neutralColor = SIMD4<Float>(0.30, 0.36, 0.44, 1.0)
    let dashLength = Snake3DState.cellSize * 0.55
    let dashThickness = Snake3DState.blockSize * 0.08
    let head =
      gameLogic.state.segments.first
      ?? SIMD3<Int>(Snake3DState.gridSize / 2, Snake3DState.gridSize / 2, Snake3DState.gridSize / 2)
    let direction = gameLogic.state.pendingDirection ?? gameLogic.state.direction
    let ptr = guideBuffer.contents().assumingMemoryBound(to: SnakeGuideInstance.self)
    var instanceIndex = 0

    func addDash(_ cell: SIMD3<Int>, scale: SIMD3<Float>, color: SIMD4<Float>) {
      let position = SIMD3<Float>(
        (Float(cell.x) - g * 0.5 + 0.5) * Snake3DState.cellSize,
        (Float(cell.y) - g * 0.5 + 0.5) * Snake3DState.cellSize,
        (Float(cell.z) - g * 0.5 + 0.5) * Snake3DState.cellSize
      )
      ptr[instanceIndex] = SnakeGuideInstance(position: position, scale: scale, color: color)
      instanceIndex += 1
    }

    func axisFamily(_ dir: SnakeDirection) -> Int {
      switch dir {
      case .posX, .negX: return 0
      case .posY, .negY: return 1
      case .posZ, .negZ: return 2
      }
    }

    let displayRightInGrid = simd_inverse(currentWorldQuat).act(displayRight)
    let displayUpInGrid = simd_inverse(currentWorldQuat).act(displayUp)
    let rightFamily = axisFamily(nearestDirection(for: displayRightInGrid))
    let upFamily = axisFamily(nearestDirection(for: displayUpInGrid))

    func colorForWorldAxisFamily(_ family: Int) -> SIMD4<Float> {
      if family == upFamily {
        return yHintColor
      }
      if family == rightFamily {
        return xHintColor
      }
      return neutralColor
    }

    let xColor = colorForWorldAxisFamily(0)
    let yColor = colorForWorldAxisFamily(1)
    let zColor = colorForWorldAxisFamily(2)
    let xScale = SIMD3<Float>(dashLength, dashThickness, dashThickness)
    let yScale = SIMD3<Float>(dashThickness, dashLength, dashThickness)
    let zScale = SIMD3<Float>(dashThickness, dashThickness, dashLength)

    let cell = Snake3DState.cellSize
    let hx = Float(head.x)
    let hy = Float(head.y)
    let hz = Float(head.z)
    let size = Snake3DState.gridSize

    func gridWorldF(_ x: Float, _ y: Float, _ z: Float) -> SIMD3<Float> {
      SIMD3<Float>(
        (x - g * 0.5 + 0.5) * cell,
        (y - g * 0.5 + 0.5) * cell,
        (z - g * 0.5 + 0.5) * cell
      )
    }

    // Spiral rails: 4 corners arranged in rotational order around each axis.
    // Each rail i has dash phase offset i*2, period 8 → collectively same coverage
    // as stride-2 but each rail is 4× less dense, creating a helix appearance.
    let spiralCorners: [(Float, Float)] = [(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)]
    let period = 4

    // Dash positions for one rail along an axis.
    // fullGrid=true → cover whole axis with phase offset; false → forward from head.
    func dashIndices(headVal: Int, railIdx: Int, fullGrid: Bool, negDir: Bool) -> [Int] {
      let offset = railIdx * 2
      if fullGrid {
        return Array(stride(from: offset, to: size, by: period))
      } else if negDir {
        let start = headVal - offset
        guard start >= 0 else { return [] }
        return Array(stride(from: start, through: 0, by: -period))
      } else {
        let start = headVal + offset
        guard start < size else { return [] }
        return Array(stride(from: start, to: size, by: period))
      }
    }

    let xForward = direction == .posX || direction == .negX
    let yForward = direction == .posY || direction == .negY
    let zForward = direction == .posZ || direction == .negZ
    let xNeg = direction == .negX
    let yNeg = direction == .negY
    let zNeg = direction == .negZ

    for railIdx in 0..<4 {
      let (pA, pB) = spiralCorners[railIdx]

      // X-axis rail (corners in YZ plane)
      for x in dashIndices(headVal: head.x, railIdx: railIdx, fullGrid: !xForward, negDir: xNeg) {
        let pos = gridWorldF(Float(x), hy + pA, hz + pB)
        ptr[instanceIndex] = SnakeGuideInstance(position: pos, scale: xScale, color: xColor)
        instanceIndex += 1
      }
      // Y-axis rail (corners in XZ plane)
      for y in dashIndices(headVal: head.y, railIdx: railIdx, fullGrid: !yForward, negDir: yNeg) {
        let pos = gridWorldF(hx + pA, Float(y), hz + pB)
        ptr[instanceIndex] = SnakeGuideInstance(position: pos, scale: yScale, color: yColor)
        instanceIndex += 1
      }
      // Z-axis rail (corners in XY plane)
      for z in dashIndices(headVal: head.z, railIdx: railIdx, fullGrid: !zForward, negDir: zNeg) {
        let pos = gridWorldF(hx + pA, hy + pB, Float(z))
        ptr[instanceIndex] = SnakeGuideInstance(position: pos, scale: zScale, color: zColor)
        instanceIndex += 1
      }
    }

    guideInstanceCount = instanceIndex
  }

  private func drawBorder(
    encoder: MTLRenderCommandEncoder,
    uniforms: inout Snake3DSceneUniforms,
    viewMatrices: [simd_float4x4]
  ) {
    guard borderInstanceCount > 0 else { return }
    encoder.setRenderPipelineState(borderPipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setVertexBuffer(cubeMesh.vertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(borderBuffer, offset: 0, index: 1)
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<Snake3DSceneUniforms>.stride, index: 2)
    viewMatrices.withUnsafeBytes { raw in
      if let base = raw.baseAddress, raw.count > 0 {
        encoder.setVertexBytes(base, length: raw.count, index: 3)
      }
    }
    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: cubeMesh.indexCount,
      indexType: .uint16,
      indexBuffer: cubeMesh.indexBuffer,
      indexBufferOffset: 0,
      instanceCount: borderInstanceCount)
  }

  private func drawGuides(
    encoder: MTLRenderCommandEncoder,
    uniforms: inout Snake3DSceneUniforms,
    viewMatrices: [simd_float4x4]
  ) {
    guard guideInstanceCount > 0 else { return }
    encoder.setRenderPipelineState(borderPipelineState)
    encoder.setDepthStencilState(depthStencilState)
    encoder.setVertexBuffer(cubeMesh.vertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(guideBuffer, offset: 0, index: 1)
    encoder.setVertexBytes(&uniforms, length: MemoryLayout<Snake3DSceneUniforms>.stride, index: 2)
    viewMatrices.withUnsafeBytes { raw in
      if let base = raw.baseAddress, raw.count > 0 {
        encoder.setVertexBytes(base, length: raw.count, index: 3)
      }
    }
    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: cubeMesh.indexCount,
      indexType: .uint16,
      indexBuffer: cubeMesh.indexBuffer,
      indexBufferOffset: 0,
      instanceCount: guideInstanceCount)
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
    desc.colorAttachments[0].isBlendingEnabled = false
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

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let desc = MTLDepthStencilDescriptor()
    desc.depthCompareFunction = .greater
    desc.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: desc)!
  }

  fileprivate static func makeTransparentDepthStencilState(device: MTLDevice)
    -> MTLDepthStencilState
  {
    let desc = MTLDepthStencilDescriptor()
    desc.depthCompareFunction = .greater
    desc.isDepthWriteEnabled = false
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
