import Foundation
import simd

class Tetris3DGameLogic {
  var state: Tetris3DState

  // Timing
  private var lastDropTime: TimeInterval = 0
  private var dropInterval: TimeInterval = 1.0  // 1 second per drop

  // Input state - pause dropping while controlling
  private var isInputActive: Bool = false

  // Input handling
  private var lastMoveTime: TimeInterval = 0
  private let moveDelay: TimeInterval = 0.15

  init() {
    self.state = Tetris3DState()
    spawnNewPiece()
  }

  // MARK: - Game Loop

  func update(currentTime: TimeInterval, deltaTime: Float) {
    guard !state.isGameOver else { return }

    // Handle line clearing animation (slower for more visible effect)
    if !state.clearingLines.isEmpty {
      state.clearAnimationProgress += deltaTime * 0.8  // Slower animation
      if state.clearAnimationProgress >= 1.0 {
        completeClearLines()
      }
      return
    }

    // Auto-drop piece (paused while controlling)
    if !isInputActive && currentTime - lastDropTime >= dropInterval {
      if !movePiece(offset: SIMD3<Int>(0, -1, 0)) {
        lockPiece()
        checkForCompleteLines()
        spawnNewPiece()
      }
      lastDropTime = currentTime
    }

    // Reset drop timer when input ends to give player time
    if isInputActive {
      lastDropTime = currentTime
    }
  }

  // MARK: - Input Handling

  // facingDirection: 0 = -Z, 1 = +X, 2 = +Z, 3 = -X
  func handleInput(
    dpadUp: Bool, dpadDown: Bool, dpadLeft: Bool, dpadRight: Bool,
    fastDrop: Bool, moveUp: Bool, switchPiece: Bool, randomRotate: Bool,
    facingDirection: Int = 0
  ) {
    guard !state.isGameOver, state.clearingLines.isEmpty else { return }

    let currentTime = Date().timeIntervalSince1970

    // Track if any d-pad input is active - pause auto-drop while controlling
    isInputActive = dpadUp || dpadDown || dpadLeft || dpadRight

    // Map d-pad to world directions based on facing
    // facingDirection: 0 = camera facing -Z (default), 1 = +X, 2 = +Z, 3 = -X
    // dpadUp = move forward (away from camera)
    // dpadDown = move backward (toward camera)
    // dpadLeft/Right = strafe

    var moveX: Int = 0
    var moveZ: Int = 0

    // Calculate movement based on facing direction
    if dpadUp {
      switch facingDirection {
      case 0: moveZ = -1  // Forward is -Z
      case 1: moveX = 1  // Forward is +X
      case 2: moveZ = 1  // Forward is +Z
      case 3: moveX = -1  // Forward is -X
      default: moveZ = -1
      }
    }
    if dpadDown {
      switch facingDirection {
      case 0: moveZ = 1  // Backward is +Z
      case 1: moveX = -1  // Backward is -X
      case 2: moveZ = -1  // Backward is -Z
      case 3: moveX = 1  // Backward is +X
      default: moveZ = 1
      }
    }
    if dpadLeft {
      switch facingDirection {
      case 0: moveX = -1  // Left is -X
      case 1: moveZ = -1  // Left is -Z
      case 2: moveX = 1  // Left is +X
      case 3: moveZ = 1  // Left is +Z
      default: moveX = -1
      }
    }
    if dpadRight {
      switch facingDirection {
      case 0: moveX = 1  // Right is +X
      case 1: moveZ = 1  // Right is +Z
      case 2: moveX = -1  // Right is -X
      case 3: moveZ = -1  // Right is -Z
      default: moveX = 1
      }
    }

    // Handle D-pad movement
    if currentTime - lastMoveTime >= moveDelay {
      if moveX != 0 || moveZ != 0 {
        movePiece(offset: SIMD3<Int>(moveX, 0, moveZ))
        lastMoveTime = currentTime
      }
    }

    // Handle move up (△) - move piece up 2 blocks to buy more time
    if moveUp {
      movePiece(offset: SIMD3<Int>(0, 2, 0))
    }

    // Handle random rotate (○) - randomly change orientation
    if randomRotate {
      applyRandomRotation()
    }

    // Handle fast drop (×)
    if fastDrop {
      hardDrop()
    }

    // Handle piece switch (buttonSquare/B)
    if switchPiece {
      cyclePieceType()
    }
  }

  // MARK: - Piece Movement

  @discardableResult
  private func movePiece(offset: SIMD3<Int>) -> Bool {
    let newPosition = state.currentPosition &+ offset

    if isValidPosition(
      position: newPosition, piece: state.currentPiece!, rotation: state.currentRotation)
    {
      state.currentPosition = newPosition
      return true
    }
    return false
  }

  private func rotatePiece() {
    guard let piece = state.currentPiece else { return }
    let rotated = piece.rotatedY()

    if isValidPosition(
      position: state.currentPosition, piece: rotated, rotation: state.currentRotation)
    {
      state.currentPiece = rotated
    }
  }

  // Apply a random rotation to the current piece (24 possible orientations in 3D)
  private func applyRandomRotation() {
    guard let piece = state.currentPiece else { return }

    // Try up to 10 random rotations to find a valid one
    for _ in 0..<10 {
      var rotated = piece

      // Apply random number of rotations around each axis
      let xRotations = Int.random(in: 0...3)
      let yRotations = Int.random(in: 0...3)
      let zRotations = Int.random(in: 0...3)

      for _ in 0..<xRotations { rotated = rotated.rotatedX() }
      for _ in 0..<yRotations { rotated = rotated.rotatedY() }
      for _ in 0..<zRotations { rotated = rotated.rotatedZ() }

      if isValidPosition(
        position: state.currentPosition, piece: rotated, rotation: state.currentRotation)
      {
        state.currentPiece = rotated
        return
      }
    }
  }

  private func hardDrop() {
    while movePiece(offset: SIMD3<Int>(0, -1, 0)) {
      // Keep dropping
    }
    lockPiece()
    checkForCompleteLines()
    spawnNewPiece()
  }

  private func cyclePieceType() {
    guard let current = state.currentPiece else { return }
    let allTypes = TetrominoType.allCases
    let currentIndex = allTypes.firstIndex(of: current.type) ?? 0
    let nextIndex = (currentIndex + 1) % allTypes.count
    let newType = allTypes[nextIndex]

    let newPiece = Tetromino(type: newType)
    if isValidPosition(
      position: state.currentPosition, piece: newPiece, rotation: state.currentRotation)
    {
      state.currentPiece = newPiece
    }
  }

  // MARK: - Collision Detection

  private func isValidPosition(position: SIMD3<Int>, piece: Tetromino, rotation: Int) -> Bool {
    for block in piece.blocks {
      let worldPos = position &+ block

      // Check bounds
      if worldPos.x < 0 || worldPos.x >= state.gridWidth
        || worldPos.y < 0 || worldPos.y >= state.gridHeight
        || worldPos.z < 0 || worldPos.z >= state.gridDepth
      {
        return false
      }

      // Check collision with placed blocks
      if state.grid[worldPos.x][worldPos.y][worldPos.z] {
        return false
      }
    }
    return true
  }

  // MARK: - Piece Management

  private func spawnNewPiece() {
    let randomType = TetrominoType.allCases.randomElement()!
    state.currentPiece = Tetromino(type: randomType)
    state.currentPosition = SIMD3<Int>(
      state.gridWidth / 2, state.gridHeight - 2, state.gridDepth / 2)
    state.currentRotation = 0

    // Check if game over
    if !isValidPosition(
      position: state.currentPosition, piece: state.currentPiece!, rotation: state.currentRotation)
    {
      state.isGameOver = true
      print("[Tetris3D] Game Over! Score: \(state.score)")
    }
  }

  private func lockPiece() {
    guard let piece = state.currentPiece else { return }

    for block in piece.blocks {
      let worldPos = state.currentPosition &+ block
      if worldPos.x >= 0 && worldPos.x < state.gridWidth
        && worldPos.y >= 0 && worldPos.y < state.gridHeight
        && worldPos.z >= 0 && worldPos.z < state.gridDepth
      {
        state.grid[worldPos.x][worldPos.y][worldPos.z] = true
        state.colorGrid[worldPos.x][worldPos.y][worldPos.z] = piece.type.color
      }
    }

    state.currentPiece = nil
  }

  // MARK: - Line Clearing

  private func checkForCompleteLines() {
    var completeLayers: [Int] = []

    // Check each Y layer
    for y in 0..<state.gridHeight {
      var isComplete = true
      for x in 0..<state.gridWidth {
        for z in 0..<state.gridDepth {
          if !state.grid[x][y][z] {
            isComplete = false
            break
          }
        }
        if !isComplete { break }
      }
      if isComplete {
        completeLayers.append(y)
      }
    }

    if !completeLayers.isEmpty {
      state.clearingLines = completeLayers
      state.clearAnimationProgress = 0.0
      print("[Tetris3D] Clearing \(completeLayers.count) lines: \(completeLayers)")
    }
  }

  private func completeClearLines() {
    let linesToClear = state.clearingLines.sorted()

    for y in linesToClear {
      // Clear the line
      for x in 0..<state.gridWidth {
        for z in 0..<state.gridDepth {
          state.grid[x][y][z] = false
          state.colorGrid[x][y][z] = SIMD4<Float>(0, 0, 0, 0)
        }
      }
    }

    // Drop blocks above cleared lines
    for clearY in linesToClear.reversed() {
      for y in (clearY + 1)..<state.gridHeight {
        for x in 0..<state.gridWidth {
          for z in 0..<state.gridDepth {
            state.grid[x][y - 1][z] = state.grid[x][y][z]
            state.colorGrid[x][y - 1][z] = state.colorGrid[x][y][z]
          }
        }
      }
      // Clear top row
      for x in 0..<state.gridWidth {
        for z in 0..<state.gridDepth {
          state.grid[x][state.gridHeight - 1][z] = false
          state.colorGrid[x][state.gridHeight - 1][z] = SIMD4<Float>(0, 0, 0, 0)
        }
      }
    }

    // Update score
    let cleared = linesToClear.count
    state.linesCleared += cleared
    state.score += cleared * 100 * state.level

    // Level up every 10 lines
    state.level = state.linesCleared / 10 + 1
    dropInterval = max(0.1, 1.0 - Double(state.level - 1) * 0.1)

    state.clearingLines = []
    state.clearAnimationProgress = 0.0
  }

  // MARK: - Reset

  func reset() {
    state = Tetris3DState()
    lastDropTime = 0
    lastMoveTime = 0
    dropInterval = 1.0
    spawnNewPiece()
  }

  // MARK: - Rendering Constants

  static let blockSize: Float = 0.12  // 12cm blocks (larger)
  static let margin: Float = 0.008  // 8mm margin
  static let gridOffsetZ: Float = -2.0  // Position in front of user

  var gridOffsetX: Float {
    -Float(state.gridWidth) * Self.blockSize * 0.5
  }
  var gridOffsetZValue: Float {
    Self.gridOffsetZ - Float(state.gridDepth) * Self.blockSize * 0.5
  }

  // Get grid center position
  var gridCenter: SIMD3<Float> {
    SIMD3<Float>(0, 0, Self.gridOffsetZ)
  }

  // Get the maximum height of placed blocks
  func getMaxBlockHeight() -> Float {
    var maxY = 0
    for x in 0..<state.gridWidth {
      for z in 0..<state.gridDepth {
        for y in (0..<state.gridHeight).reversed() {
          if state.grid[x][y][z] {
            maxY = max(maxY, y + 1)
            break
          }
        }
      }
    }
    return Float(maxY) * Self.blockSize
  }

  // MARK: - Rendering Data

  func getBlockStates() -> [TetrisBlockState] {
    var states: [TetrisBlockState] = []
    let blockSize = Self.blockSize
    let margin = Self.margin
    let effectiveSize = blockSize - margin

    // Calculate grid offset to center it
    let gridOffsetX = self.gridOffsetX
    let gridOffsetY: Float = 0.0  // Ground level
    let gridOffsetZ = self.gridOffsetZValue

    // Add placed blocks
    for x in 0..<state.gridWidth {
      for y in 0..<state.gridHeight {
        for z in 0..<state.gridDepth {
          if state.grid[x][y][z] {
            let worldPos = SIMD3<Float>(
              gridOffsetX + Float(x) * blockSize + blockSize * 0.5,
              gridOffsetY + Float(y) * blockSize + blockSize * 0.5,
              gridOffsetZ + Float(z) * blockSize + blockSize * 0.5
            )

            var alpha: Float = 1.0
            var xOffset: Float = 0.0
            var yOffset: Float = 0.0
            var zOffset: Float = 0.0

            // Apply clear animation - blocks scatter outward and fade
            if state.clearingLines.contains(y) {
              let t = state.clearAnimationProgress
              // Fade out with easing
              alpha = 1.0 - (t * t)  // Ease in fade

              // Calculate scatter direction (outward from center)
              let centerX = Float(state.gridWidth - 1) * 0.5
              let centerZ = Float(state.gridDepth - 1) * 0.5
              let dirX = (Float(x) - centerX) / centerX  // -1 to 1
              let dirZ = (Float(z) - centerZ) / centerZ  // -1 to 1

              // Scatter outward as animation progresses
              let scatterAmount = t * t * blockSize * 0.8
              xOffset = dirX * scatterAmount
              zOffset = dirZ * scatterAmount
              // Also drop down slightly
              yOffset = -t * t * blockSize * 0.3
            }

            // Determine type from color
            let color = state.colorGrid[x][y][z]
            let typeValue = colorToType(color)

            let animatedPos = SIMD3<Float>(
              worldPos.x + xOffset, worldPos.y + yOffset, worldPos.z + zOffset)

            states.append(
              TetrisBlockState(
                positionAndType: SIMD4<Float>(animatedPos, typeValue),
                motionAndPhase: SIMD4<Float>(0, 0, 0, alpha),
                scaleAndPadding: SIMD4<Float>(effectiveSize, effectiveSize, effectiveSize, 0),
                homeAndJitter: SIMD4<Float>(0, 0, 0, 0)
              ))
          }
        }
      }
    }

    // Add current falling piece
    if let piece = state.currentPiece {
      for block in piece.blocks {
        let gridPos = state.currentPosition &+ block
        if gridPos.x >= 0 && gridPos.x < state.gridWidth
          && gridPos.y >= 0 && gridPos.y < state.gridHeight
          && gridPos.z >= 0 && gridPos.z < state.gridDepth
        {
          let worldPos = SIMD3<Float>(
            gridOffsetX + Float(gridPos.x) * blockSize + blockSize * 0.5,
            gridOffsetY + Float(gridPos.y) * blockSize + blockSize * 0.5,
            gridOffsetZ + Float(gridPos.z) * blockSize + blockSize * 0.5
          )

          states.append(
            TetrisBlockState(
              positionAndType: SIMD4<Float>(worldPos, Float(piece.type.rawValue)),
              motionAndPhase: SIMD4<Float>(0, 0, 0, 1.0),
              scaleAndPadding: SIMD4<Float>(effectiveSize, effectiveSize, effectiveSize, 0),
              homeAndJitter: SIMD4<Float>(0, 0, 0, 0)
            ))
        }
      }
    }

    return states
  }

  // Get ghost piece positions (where piece will land)
  // Returns positions for flat markers at ground level showing drop location
  func getGhostPiecePositions() -> [SIMD3<Float>] {
    guard let piece = state.currentPiece else { return [] }

    // Find where piece would land
    var ghostY = state.currentPosition.y
    while ghostY > 0 {
      let testPos = SIMD3<Int>(state.currentPosition.x, ghostY - 1, state.currentPosition.z)
      if !isValidPosition(position: testPos, piece: piece, rotation: state.currentRotation) {
        break
      }
      ghostY -= 1
    }

    let blockSize = Self.blockSize
    let gridOffsetX = self.gridOffsetX
    let gridOffsetZ = self.gridOffsetZValue

    // Find highest block at each ghost position to render marker above it
    var positions: [SIMD3<Float>] = []
    for block in piece.blocks {
      let gridPos = SIMD3<Int>(state.currentPosition.x, ghostY, state.currentPosition.z) &+ block
      if gridPos.x >= 0 && gridPos.x < state.gridWidth
        && gridPos.z >= 0 && gridPos.z < state.gridDepth
      {
        // Find highest occupied cell in this column
        var highestY: Float = 0.02  // Default just above ground
        for y in 0..<state.gridHeight {
          if state.grid[gridPos.x][y][gridPos.z] {
            highestY = Float(y + 1) * blockSize + 0.02
          }
        }

        let worldPos = SIMD3<Float>(
          gridOffsetX + Float(gridPos.x) * blockSize + blockSize * 0.5,
          highestY,  // Above the highest block in this column
          gridOffsetZ + Float(gridPos.z) * blockSize + blockSize * 0.5
        )
        positions.append(worldPos)
      }
    }
    return positions
  }

  private func colorToType(_ color: SIMD4<Float>) -> Float {
    // Match color to tetromino type
    for type in TetrominoType.allCases {
      let tc = type.color
      if abs(color.x - tc.x) < 0.1 && abs(color.y - tc.y) < 0.1 && abs(color.z - tc.z) < 0.1 {
        return Float(type.rawValue)
      }
    }
    return 0
  }
}
