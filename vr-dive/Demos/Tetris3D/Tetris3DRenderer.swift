import Metal
import simd

private typealias MeshBuffers = (vertexBuffer: MTLBuffer, indexBuffer: MTLBuffer, indexCount: Int)

struct TetrisMeshVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
}

struct TetrisGroundVertex {
  var position: SIMD3<Float>
  var uv: SIMD2<Float>
}

final class Tetris3DRenderer: VisualPatternController {
  let identifier: VisualPatternKind = .tetris3D
  let preferredClearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.04, alpha: 1)

  private let device: MTLDevice
  private let blockPipelineState: MTLRenderPipelineState
  private let groundPipelineState: MTLRenderPipelineState
  private let depthStencilState: MTLDepthStencilState
  private let maxViewCount: Int

  private let cubeMesh: MeshBuffers
  private var blockStateBuffer: MTLBuffer
  private let maxBlockCount = 3000

  private var groundVertexBuffer: MTLBuffer?
  private var groundVertexCount: Int = 0

  // Ghost piece (drop preview) buffer
  private var ghostBuffer: MTLBuffer?
  private var ghostVertexCount: Int = 0

  // Direction marker buffer (dynamic, updated each frame)
  private var directionMarkerBuffer: MTLBuffer?
  private var directionMarkerVertexCount: Int = 0

  // Camera-relative direction (0 = -Z, 1 = +X, 2 = +Z, 3 = -X)
  private var currentFacingDirection: Int = 0

  private let gameLogic: Tetris3DGameLogic
  private var lastUpdateTime: TimeInterval = 0
  private weak var gameManager: GameManager?

  // Input state - D-pad timing
  private var lastDpadTime: TimeInterval = 0
  private var lastMoveUpTime: TimeInterval = 0
  private var lastFastDropTime: TimeInterval = 0
  private var lastSwitchTime: TimeInterval = 0
  private var lastRotateTime: TimeInterval = 0
  private let buttonDelay: TimeInterval = 0.15
  private let actionDelay: TimeInterval = 0.25

  init(device: MTLDevice, library: MTLLibrary, maxViewCount: Int, gameManager: GameManager) {
    self.device = device
    self.maxViewCount = max(1, maxViewCount)
    self.gameLogic = Tetris3DGameLogic()
    self.gameManager = gameManager

    // Create cube geometry
    cubeMesh = Tetris3DRenderer.makeCubeGeometry(device: device)

    // Create block state buffer
    blockStateBuffer = device.makeBuffer(
      length: MemoryLayout<TetrisBlockState>.stride * maxBlockCount,
      options: [.storageModeShared]
    )!

    // Create pipelines
    blockPipelineState = try! Tetris3DRenderer.makeBlockPipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    groundPipelineState = try! Tetris3DRenderer.makeGroundPipelineState(
      device: device, library: library, maxViewCount: self.maxViewCount)
    depthStencilState = Tetris3DRenderer.makeDepthStencilState(device: device)

    // Create ground
    createGroundGeometry()
  }

  private func createGroundGeometry() {
    // Ground grid matching the tetris grid exactly
    let blockSize = Tetris3DGameLogic.blockSize
    let gridWidth = 6  // Match Tetris3DState.gridWidth
    let gridDepth = 6  // Match Tetris3DState.gridDepth
    let gridOffsetX = -Float(gridWidth) * blockSize * 0.5
    let gridOffsetZ = Tetris3DGameLogic.gridOffsetZ - Float(gridDepth) * blockSize * 0.5

    let groundY: Float = -0.01
    let gridSizeX = Float(gridWidth) * blockSize
    let gridSizeZ = Float(gridDepth) * blockSize

    // UV covers exactly 6x6 grid cells
    let groundVertices: [TetrisGroundVertex] = [
      TetrisGroundVertex(
        position: SIMD3<Float>(gridOffsetX, groundY, gridOffsetZ),
        uv: SIMD2<Float>(0, 0)),
      TetrisGroundVertex(
        position: SIMD3<Float>(gridOffsetX + gridSizeX, groundY, gridOffsetZ),
        uv: SIMD2<Float>(Float(gridWidth), 0)),
      TetrisGroundVertex(
        position: SIMD3<Float>(gridOffsetX + gridSizeX, groundY, gridOffsetZ + gridSizeZ),
        uv: SIMD2<Float>(Float(gridWidth), Float(gridDepth))),

      TetrisGroundVertex(
        position: SIMD3<Float>(gridOffsetX, groundY, gridOffsetZ),
        uv: SIMD2<Float>(0, 0)),
      TetrisGroundVertex(
        position: SIMD3<Float>(gridOffsetX + gridSizeX, groundY, gridOffsetZ + gridSizeZ),
        uv: SIMD2<Float>(Float(gridWidth), Float(gridDepth))),
      TetrisGroundVertex(
        position: SIMD3<Float>(gridOffsetX, groundY, gridOffsetZ + gridSizeZ),
        uv: SIMD2<Float>(0, Float(gridDepth))),
    ]

    groundVertexCount = groundVertices.count

    groundVertexBuffer = device.makeBuffer(
      bytes: groundVertices,
      length: MemoryLayout<TetrisGroundVertex>.stride * groundVertices.count,
      options: [.storageModeShared]
    )

    // Create direction arrow (pointing "forward" = -Z)
    createDirectionMarker()
  }

  private func createDirectionMarker() {
    // Create buffer for direction marker (will be updated dynamically)
    // Arrow shape needs 3 vertices
    directionMarkerBuffer = device.makeBuffer(
      length: MemoryLayout<TetrisGroundVertex>.stride * 3,
      options: [.storageModeShared]
    )
    directionMarkerVertexCount = 3
  }

  // Update direction marker based on camera position and max block height
  private func updateDirectionMarker(cameraPosition: SIMD3<Float>?) {
    guard let buffer = directionMarkerBuffer else { return }

    let blockSize = Tetris3DGameLogic.blockSize
    let gridCenter = gameLogic.gridCenter
    let maxHeight = gameLogic.getMaxBlockHeight()

    // Determine facing direction from camera position
    var forwardDir = SIMD3<Float>(0, 0, -1)  // Default forward
    if let camPos = cameraPosition {
      // Vector from grid center to camera
      let toCamera = SIMD2<Float>(camPos.x - gridCenter.x, camPos.z - gridCenter.z)

      // Determine which quadrant the camera is in
      // Arrow should point away from camera (toward "forward" from player's perspective)
      if abs(toCamera.x) > abs(toCamera.y) {
        // Camera is more on X axis
        if toCamera.x > 0 {
          forwardDir = SIMD3<Float>(-1, 0, 0)  // Forward is -X
          currentFacingDirection = 3
        } else {
          forwardDir = SIMD3<Float>(1, 0, 0)  // Forward is +X
          currentFacingDirection = 1
        }
      } else {
        // Camera is more on Z axis
        if toCamera.y > 0 {
          forwardDir = SIMD3<Float>(0, 0, -1)  // Forward is -Z
          currentFacingDirection = 0
        } else {
          forwardDir = SIMD3<Float>(0, 0, 1)  // Forward is +Z
          currentFacingDirection = 2
        }
      }
    }

    // Position arrow above the highest block, entirely outside the grid, pointing outward
    let arrowY = maxHeight + 0.05
    let gridHalfSize = Float(6) * blockSize * 0.5 + 0.08  // Just outside grid edge

    // Arrow positioned outside grid, pointing outward (away from grid)
    let arrowBase = gridCenter + forwardDir * gridHalfSize  // At grid edge
    let arrowTip = arrowBase + forwardDir * 0.15  // Arrow points outward (away from grid)

    // Perpendicular direction for arrow width
    let perpDir = SIMD3<Float>(-forwardDir.z, 0, forwardDir.x)
    let arrowWidth: Float = 0.06  // Smaller arrow

    let vertices: [TetrisGroundVertex] = [
      TetrisGroundVertex(
        position: SIMD3<Float>(arrowTip.x, arrowY, arrowTip.z),
        uv: SIMD2<Float>(0.5, 0)),
      TetrisGroundVertex(
        position: SIMD3<Float>(
          arrowBase.x - perpDir.x * arrowWidth, arrowY, arrowBase.z - perpDir.z * arrowWidth),
        uv: SIMD2<Float>(0, 1)),
      TetrisGroundVertex(
        position: SIMD3<Float>(
          arrowBase.x + perpDir.x * arrowWidth, arrowY, arrowBase.z + perpDir.z * arrowWidth),
        uv: SIMD2<Float>(1, 1)),
    ]

    memcpy(buffer.contents(), vertices, MemoryLayout<TetrisGroundVertex>.stride * 3)
  }

  // Get the current facing direction for input mapping
  func getCurrentFacingDirection() -> Int {
    return currentFacingDirection
  }

  // MARK: - VisualPatternController Protocol

  func updateSimulation(_ context: PatternSimulationContext) {
    let currentTime = TimeInterval(context.time)
    let deltaTime = currentTime - lastUpdateTime

    // Handle input
    if let manager = gameManager {
      let input = manager.getTetrisInput()

      // D-pad controls block movement
      let canMoveDpad = currentTime - lastDpadTime > buttonDelay
      let dpadPressed = input.dpadUp || input.dpadDown || input.dpadLeft || input.dpadRight

      // Right side buttons (PS5 layout)
      // △ = moveUp, × = fastDrop, □ = switchPiece, ○ = randomRotate
      let shouldMoveUp = input.buttonTriangle && (currentTime - lastMoveUpTime > actionDelay)
      let shouldFastDrop = input.buttonCross && (currentTime - lastFastDropTime > actionDelay)
      let shouldSwitch = input.buttonSquare && (currentTime - lastSwitchTime > actionDelay)
      let shouldRandomRotate = input.buttonCircle && (currentTime - lastRotateTime > actionDelay)

      if dpadPressed && canMoveDpad { lastDpadTime = currentTime }
      if shouldMoveUp { lastMoveUpTime = currentTime }
      if shouldFastDrop { lastFastDropTime = currentTime }
      if shouldSwitch { lastSwitchTime = currentTime }
      if shouldRandomRotate { lastRotateTime = currentTime }

      gameLogic.handleInput(
        dpadUp: input.dpadUp && canMoveDpad,
        dpadDown: input.dpadDown && canMoveDpad,
        dpadLeft: input.dpadLeft && canMoveDpad,
        dpadRight: input.dpadRight && canMoveDpad,
        fastDrop: shouldFastDrop,
        moveUp: shouldMoveUp,
        switchPiece: shouldSwitch,
        randomRotate: shouldRandomRotate,
        facingDirection: currentFacingDirection
      )
    }

    if lastUpdateTime > 0 {
      gameLogic.update(currentTime: currentTime, deltaTime: Float(deltaTime))
    }
    lastUpdateTime = currentTime

    // Update block state buffer
    let states = gameLogic.getBlockStates()
    if !states.isEmpty {
      let buffer = blockStateBuffer.contents().assumingMemoryBound(to: TetrisBlockState.self)
      for (index, state) in states.enumerated() where index < maxBlockCount {
        buffer[index] = state
      }
    }
  }

  func resetToInitialState() {
    gameLogic.reset()
    lastUpdateTime = 0
  }

  func encodeFrame(encoder: MTLRenderCommandEncoder, context: PatternRenderContext) {
    let states = gameLogic.getBlockStates()

    encoder.setDepthStencilState(depthStencilState)
    encoder.setCullMode(.none)
    encoder.setFrontFacing(.counterClockwise)
    encoder.setTriangleFillMode(.fill)

    context.applyViewConfiguration(on: encoder)

    var sceneUniforms = TetrisSceneUniforms(
      time: context.time,
      layerCount: UInt32(context.viewData.viewCount),
      padding: SIMD2<Float>(0, 0)
    )

    var viewMatrices = context.viewData.viewProjectionMatrices
    if viewMatrices.isEmpty {
      viewMatrices = [matrix_identity_float4x4]
    }

    // Extract camera position from view transform
    var cameraPosition: SIMD3<Float>? = nil
    if let viewToWorld = context.viewData.viewToWorldTransforms.first {
      cameraPosition = SIMD3<Float>(
        viewToWorld.columns.3.x, viewToWorld.columns.3.y, viewToWorld.columns.3.z)
    }

    // Update direction marker based on camera and block height
    updateDirectionMarker(cameraPosition: cameraPosition)

    // Render ground
    if let groundBuffer = groundVertexBuffer {
      encoder.setRenderPipelineState(groundPipelineState)
      encoder.setVertexBuffer(groundBuffer, offset: 0, index: 0)
      encoder.setVertexBytes(
        &sceneUniforms, length: MemoryLayout<TetrisSceneUniforms>.stride, index: 1)
      viewMatrices.withUnsafeBytes {
        if let baseAddress = $0.baseAddress, $0.count > 0 {
          encoder.setVertexBytes(baseAddress, length: $0.count, index: 2)
        }
      }
      encoder.setFragmentBytes(
        &sceneUniforms, length: MemoryLayout<TetrisSceneUniforms>.stride, index: 0)
      encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: groundVertexCount)
    }

    // Render direction marker (arrow) - now uses directionMarkerBuffer
    if let arrowBuffer = directionMarkerBuffer, directionMarkerVertexCount > 0 {
      encoder.setRenderPipelineState(groundPipelineState)
      encoder.setVertexBuffer(arrowBuffer, offset: 0, index: 0)
      encoder.setVertexBytes(
        &sceneUniforms, length: MemoryLayout<TetrisSceneUniforms>.stride, index: 1)
      viewMatrices.withUnsafeBytes {
        if let baseAddress = $0.baseAddress, $0.count > 0 {
          encoder.setVertexBytes(baseAddress, length: $0.count, index: 2)
        }
      }
      encoder.setFragmentBytes(
        &sceneUniforms, length: MemoryLayout<TetrisSceneUniforms>.stride, index: 0)
      encoder.drawPrimitives(
        type: .triangle, vertexStart: 0, vertexCount: directionMarkerVertexCount)
    }

    // Render ghost pieces (drop preview) as semi-transparent blocks
    let ghostPositions = gameLogic.getGhostPiecePositions()
    if !ghostPositions.isEmpty, let piece = gameLogic.state.currentPiece {
      let effectiveSize = Tetris3DGameLogic.blockSize - Tetris3DGameLogic.margin

      // Update ghost states in block buffer (use upper part of buffer)
      let ghostStartIndex = min(states.count, maxBlockCount - ghostPositions.count)
      let buffer = blockStateBuffer.contents().assumingMemoryBound(to: TetrisBlockState.self)

      for (i, pos) in ghostPositions.enumerated() {
        let index = ghostStartIndex + i
        if index < maxBlockCount {
          buffer[index] = TetrisBlockState(
            positionAndType: SIMD4<Float>(pos, Float(piece.type.rawValue)),
            motionAndPhase: SIMD4<Float>(0, 0, 0, 0.25),  // Semi-transparent
            scaleAndPadding: SIMD4<Float>(effectiveSize, effectiveSize * 0.1, effectiveSize, 0),  // Flat
            homeAndJitter: SIMD4<Float>(0, 0, 0, 0)
          )
        }
      }

      // Draw ghost blocks
      encoder.setRenderPipelineState(blockPipelineState)
      encoder.setVertexBuffer(cubeMesh.vertexBuffer, offset: 0, index: 0)
      encoder.setVertexBuffer(
        blockStateBuffer, offset: ghostStartIndex * MemoryLayout<TetrisBlockState>.stride, index: 1)
      encoder.setVertexBytes(
        &sceneUniforms, length: MemoryLayout<TetrisSceneUniforms>.stride, index: 2)
      viewMatrices.withUnsafeBytes {
        if let baseAddress = $0.baseAddress, $0.count > 0 {
          encoder.setVertexBytes(baseAddress, length: $0.count, index: 3)
        }
      }
      encoder.setFragmentBytes(
        &sceneUniforms, length: MemoryLayout<TetrisSceneUniforms>.stride, index: 0)
      encoder.drawIndexedPrimitives(
        type: .triangle,
        indexCount: cubeMesh.indexCount,
        indexType: .uint16,
        indexBuffer: cubeMesh.indexBuffer,
        indexBufferOffset: 0,
        instanceCount: min(ghostPositions.count, maxBlockCount - ghostStartIndex)
      )
    }

    // Render blocks
    guard !states.isEmpty else { return }

    encoder.setRenderPipelineState(blockPipelineState)
    encoder.setVertexBuffer(cubeMesh.vertexBuffer, offset: 0, index: 0)
    encoder.setVertexBuffer(blockStateBuffer, offset: 0, index: 1)
    encoder.setVertexBytes(
      &sceneUniforms, length: MemoryLayout<TetrisSceneUniforms>.stride, index: 2)
    viewMatrices.withUnsafeBytes {
      if let baseAddress = $0.baseAddress, $0.count > 0 {
        encoder.setVertexBytes(baseAddress, length: $0.count, index: 3)
      }
    }
    encoder.setFragmentBytes(
      &sceneUniforms, length: MemoryLayout<TetrisSceneUniforms>.stride, index: 0)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: cubeMesh.indexCount,
      indexType: .uint16,
      indexBuffer: cubeMesh.indexBuffer,
      indexBufferOffset: 0,
      instanceCount: min(states.count, maxBlockCount)
    )
  }
}

// MARK: - Pipeline and Geometry Creation

extension Tetris3DRenderer {
  fileprivate static func makeBlockPipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "tetrisBlockVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "tetrisBlockFragmentShader")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.colorAttachments[0].isBlendingEnabled = true
    descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
    descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .triangle

    let vertexDescriptor = MTLVertexDescriptor()
    vertexDescriptor.attributes[0].format = .float3
    vertexDescriptor.attributes[0].offset = 0
    vertexDescriptor.attributes[0].bufferIndex = 0
    vertexDescriptor.attributes[1].format = .float3
    vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD3<Float>>.stride
    vertexDescriptor.attributes[1].bufferIndex = 0
    vertexDescriptor.layouts[0].stride = MemoryLayout<TetrisMeshVertex>.stride
    descriptor.vertexDescriptor = vertexDescriptor
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)

    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  fileprivate static func makeGroundPipelineState(
    device: MTLDevice,
    library: MTLLibrary,
    maxViewCount: Int
  ) throws -> MTLRenderPipelineState {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "tetrisGroundVertexShader")
    descriptor.fragmentFunction = library.makeFunction(name: "tetrisGroundFragmentShader")
    descriptor.colorAttachments[0].pixelFormat = .rgba16Float
    descriptor.colorAttachments[0].isBlendingEnabled = true
    descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
    descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
    descriptor.depthAttachmentPixelFormat = .depth32Float
    descriptor.inputPrimitiveTopology = .triangle
    descriptor.maxVertexAmplificationCount = max(maxViewCount, 1)

    return try device.makeRenderPipelineState(descriptor: descriptor)
  }

  fileprivate static func makeDepthStencilState(device: MTLDevice) -> MTLDepthStencilState {
    let descriptor = MTLDepthStencilDescriptor()
    descriptor.depthCompareFunction = .greater
    descriptor.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: descriptor)!
  }

  fileprivate static func makeCubeGeometry(device: MTLDevice) -> MeshBuffers {
    let vertices: [TetrisMeshVertex] = [
      // Front
      TetrisMeshVertex(position: [-0.5, -0.5, 0.5], normal: [0, 0, 1]),
      TetrisMeshVertex(position: [0.5, -0.5, 0.5], normal: [0, 0, 1]),
      TetrisMeshVertex(position: [0.5, 0.5, 0.5], normal: [0, 0, 1]),
      TetrisMeshVertex(position: [-0.5, 0.5, 0.5], normal: [0, 0, 1]),
      // Back
      TetrisMeshVertex(position: [-0.5, -0.5, -0.5], normal: [0, 0, -1]),
      TetrisMeshVertex(position: [0.5, -0.5, -0.5], normal: [0, 0, -1]),
      TetrisMeshVertex(position: [0.5, 0.5, -0.5], normal: [0, 0, -1]),
      TetrisMeshVertex(position: [-0.5, 0.5, -0.5], normal: [0, 0, -1]),
      // Left
      TetrisMeshVertex(position: [-0.5, -0.5, -0.5], normal: [-1, 0, 0]),
      TetrisMeshVertex(position: [-0.5, -0.5, 0.5], normal: [-1, 0, 0]),
      TetrisMeshVertex(position: [-0.5, 0.5, 0.5], normal: [-1, 0, 0]),
      TetrisMeshVertex(position: [-0.5, 0.5, -0.5], normal: [-1, 0, 0]),
      // Right
      TetrisMeshVertex(position: [0.5, -0.5, -0.5], normal: [1, 0, 0]),
      TetrisMeshVertex(position: [0.5, -0.5, 0.5], normal: [1, 0, 0]),
      TetrisMeshVertex(position: [0.5, 0.5, 0.5], normal: [1, 0, 0]),
      TetrisMeshVertex(position: [0.5, 0.5, -0.5], normal: [1, 0, 0]),
      // Top
      TetrisMeshVertex(position: [-0.5, 0.5, 0.5], normal: [0, 1, 0]),
      TetrisMeshVertex(position: [0.5, 0.5, 0.5], normal: [0, 1, 0]),
      TetrisMeshVertex(position: [0.5, 0.5, -0.5], normal: [0, 1, 0]),
      TetrisMeshVertex(position: [-0.5, 0.5, -0.5], normal: [0, 1, 0]),
      // Bottom
      TetrisMeshVertex(position: [-0.5, -0.5, 0.5], normal: [0, -1, 0]),
      TetrisMeshVertex(position: [0.5, -0.5, 0.5], normal: [0, -1, 0]),
      TetrisMeshVertex(position: [0.5, -0.5, -0.5], normal: [0, -1, 0]),
      TetrisMeshVertex(position: [-0.5, -0.5, -0.5], normal: [0, -1, 0]),
    ]

    let indices: [UInt16] = [
      0, 1, 2, 0, 2, 3,  // Front
      4, 5, 6, 4, 6, 7,  // Back
      8, 9, 10, 8, 10, 11,  // Left
      12, 13, 14, 12, 14, 15,  // Right
      16, 17, 18, 16, 18, 19,  // Top
      20, 21, 22, 20, 22, 23,  // Bottom
    ]

    let vertexBuffer = device.makeBuffer(
      bytes: vertices,
      length: MemoryLayout<TetrisMeshVertex>.stride * vertices.count,
      options: [.storageModeShared]
    )!
    let indexBuffer = device.makeBuffer(
      bytes: indices,
      length: MemoryLayout<UInt16>.stride * indices.count,
      options: [.storageModeShared]
    )!

    return (vertexBuffer, indexBuffer, indices.count)
  }
}
