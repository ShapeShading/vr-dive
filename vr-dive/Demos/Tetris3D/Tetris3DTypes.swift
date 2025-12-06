import simd

// MARK: - Tetromino Shapes (7 classic shapes)

enum TetrominoType: Int, CaseIterable {
  case I = 0  // Cyan
  case O = 1  // Yellow
  case T = 2  // Purple
  case S = 3  // Green
  case Z = 4  // Red
  case J = 5  // Blue
  case L = 6  // Orange

  var color: SIMD4<Float> {
    switch self {
    case .I: return SIMD4<Float>(0.0, 0.9, 0.9, 1.0)  // Cyan
    case .O: return SIMD4<Float>(0.9, 0.9, 0.0, 1.0)  // Yellow
    case .T: return SIMD4<Float>(0.7, 0.0, 0.9, 1.0)  // Purple
    case .S: return SIMD4<Float>(0.0, 0.9, 0.0, 1.0)  // Green
    case .Z: return SIMD4<Float>(0.9, 0.0, 0.0, 1.0)  // Red
    case .J: return SIMD4<Float>(0.0, 0.0, 0.9, 1.0)  // Blue
    case .L: return SIMD4<Float>(0.9, 0.5, 0.0, 1.0)  // Orange
    }
  }

  // Define block positions relative to center (rotation point)
  var blocks: [SIMD3<Int>] {
    switch self {
    case .I:
      return [
        SIMD3<Int>(-1, 0, 0),
        SIMD3<Int>(0, 0, 0),
        SIMD3<Int>(1, 0, 0),
        SIMD3<Int>(2, 0, 0),
      ]
    case .O:
      return [
        SIMD3<Int>(0, 0, 0),
        SIMD3<Int>(1, 0, 0),
        SIMD3<Int>(0, 0, 1),
        SIMD3<Int>(1, 0, 1),
      ]
    case .T:
      return [
        SIMD3<Int>(0, 0, 0),
        SIMD3<Int>(-1, 0, 0),
        SIMD3<Int>(1, 0, 0),
        SIMD3<Int>(0, 0, 1),
      ]
    case .S:
      return [
        SIMD3<Int>(0, 0, 0),
        SIMD3<Int>(1, 0, 0),
        SIMD3<Int>(0, 0, 1),
        SIMD3<Int>(-1, 0, 1),
      ]
    case .Z:
      return [
        SIMD3<Int>(0, 0, 0),
        SIMD3<Int>(-1, 0, 0),
        SIMD3<Int>(0, 0, 1),
        SIMD3<Int>(1, 0, 1),
      ]
    case .J:
      return [
        SIMD3<Int>(0, 0, 0),
        SIMD3<Int>(-1, 0, 0),
        SIMD3<Int>(1, 0, 0),
        SIMD3<Int>(-1, 0, 1),
      ]
    case .L:
      return [
        SIMD3<Int>(0, 0, 0),
        SIMD3<Int>(-1, 0, 0),
        SIMD3<Int>(1, 0, 0),
        SIMD3<Int>(1, 0, 1),
      ]
    }
  }
}

// MARK: - Game State

struct Tetris3DState {
  var gridWidth: Int = 6
  var gridHeight: Int = 18  // Taller for easier gameplay
  var gridDepth: Int = 6

  // Current falling piece
  var currentPiece: Tetromino?
  var currentPosition: SIMD3<Int> = SIMD3<Int>(5, 19, 5)
  var currentRotation: Int = 0

  // Grid of placed blocks (true = occupied)
  var grid: [[[Bool]]]
  var colorGrid: [[[SIMD4<Float>]]]

  // Game state
  var isGameOver: Bool = false
  var score: Int = 0
  var level: Int = 1
  var linesCleared: Int = 0

  // Animation state for clearing lines
  var clearingLines: [Int] = []
  var clearAnimationProgress: Float = 0.0

  init() {
    grid = Array(
      repeating: Array(repeating: Array(repeating: false, count: gridDepth), count: gridHeight),
      count: gridWidth)
    colorGrid = Array(
      repeating: Array(
        repeating: Array(repeating: SIMD4<Float>(0, 0, 0, 0), count: gridDepth), count: gridHeight),
      count: gridWidth)
  }
}

struct Tetromino {
  var type: TetrominoType
  var blocks: [SIMD3<Int>]

  init(type: TetrominoType) {
    self.type = type
    self.blocks = type.blocks
  }

  // Rotate around Y axis (horizontal rotation)
  func rotatedY() -> Tetromino {
    var rotated = self
    rotated.blocks = blocks.map { block in
      // Rotate 90 degrees clockwise around Y axis
      SIMD3<Int>(-block.z, block.y, block.x)
    }
    return rotated
  }

  // Rotate around X axis (tilt forward/backward)
  func rotatedX() -> Tetromino {
    var rotated = self
    rotated.blocks = blocks.map { block in
      // Rotate 90 degrees around X axis
      SIMD3<Int>(block.x, -block.z, block.y)
    }
    return rotated
  }

  // Rotate around Z axis (roll left/right)
  func rotatedZ() -> Tetromino {
    var rotated = self
    rotated.blocks = blocks.map { block in
      // Rotate 90 degrees around Z axis
      SIMD3<Int>(-block.y, block.x, block.z)
    }
    return rotated
  }
}

// MARK: - Rendering Data (matches Metal shader structures)

struct TetrisBlockState {
  var positionAndType: SIMD4<Float>  // xyz = position, w = type (for color)
  var motionAndPhase: SIMD4<Float>  // xyz = unused, w = alpha
  var scaleAndPadding: SIMD4<Float>  // xyz = scale
  var homeAndJitter: SIMD4<Float>  // unused, for compatibility
}

struct TetrisSceneUniforms {
  var time: Float
  var layerCount: UInt32
  var padding: SIMD2<Float>
}

struct TetrisSimulationUniforms {
  var deltaTime: Float
  var globalTime: Float
  var objectCount: UInt32
  var padding: UInt32 = 0
}
