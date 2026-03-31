import simd

// MARK: - Direction (6 faces of a cube)

enum SnakeDirection: Int, CaseIterable {
  case posZ = 0
  case negZ = 1
  case posX = 2
  case negX = 3
  case posY = 4
  case negY = 5

  var delta: SIMD3<Int> {
    switch self {
    case .posZ: return SIMD3<Int>(0, 0, 1)
    case .negZ: return SIMD3<Int>(0, 0, -1)
    case .posX: return SIMD3<Int>(1, 0, 0)
    case .negX: return SIMD3<Int>(-1, 0, 0)
    case .posY: return SIMD3<Int>(0, 1, 0)
    case .negY: return SIMD3<Int>(0, -1, 0)
    }
  }

  var opposite: SnakeDirection {
    switch self {
    case .posZ: return .negZ
    case .negZ: return .posZ
    case .posX: return .negX
    case .negX: return .posX
    case .posY: return .negY
    case .negY: return .posY
    }
  }
}

// MARK: - Game State

struct Snake3DState {
  static let gridSize: Int = 80
  static let cellSize: Float = 0.2  // enlarged grid spacing for a larger play volume
  static let blockSize: Float = 0.192  // keep guide cells and snake blocks visually consistent

  var segments: [SIMD3<Int>]  // index 0 = head
  var direction: SnakeDirection
  var pendingDirection: SnakeDirection?  // queued turn applied on next step
  var foods: [SIMD3<Int>]
  var score: Int
  var isGameOver: Bool
  var pendingGrow: Int  // how many segments to add

  init() {
    let mid = Snake3DState.gridSize / 2
    segments = [
      SIMD3<Int>(mid, mid, mid),
      SIMD3<Int>(mid, mid, mid + 1),
      SIMD3<Int>(mid, mid, mid + 2),
      SIMD3<Int>(mid, mid, mid + 3),
      SIMD3<Int>(mid, mid, mid + 4),
    ]
    direction = .negZ
    pendingDirection = nil
    foods = []
    score = 0
    isGameOver = false
    pendingGrow = 0
  }
}

// MARK: - GPU Types

/// One snake segment instance on GPU
struct SnakeSegmentInstance {
  var position: SIMD3<Float>  // world position (pre-rotated on CPU)
  var colorAndIndex: SIMD4<Float>  // rgb = color, a = normalized segment index (0=head)
  var scale: Float
  var padding: SIMD3<Float> = .zero

  init(position: SIMD3<Float>, color: SIMD3<Float>, normalizedIndex: Float, scale: Float) {
    self.position = position
    self.colorAndIndex = SIMD4<Float>(color.x, color.y, color.z, normalizedIndex)
    self.scale = scale
  }
}

/// One food instance on GPU
struct FoodInstance {
  var position: SIMD3<Float>
  var phase: Float  // animation phase offset
  var hit: Float
  var padding: SIMD3<Float> = .zero
}

/// Mesh vertex shared by all snake geometry
struct SnakeMeshVertex {
  var position: SIMD3<Float>
  var normal: SIMD3<Float>
}

/// Scene uniforms passed to shaders each frame
struct Snake3DSceneUniforms {
  var worldRotation: simd_float4x4  // cumulative world rotation (applied to all geometry)
  var anchorTranslation: SIMD3<Float>  // offset so snake head is at fixed view point
  var time: Float
  var layerCount: UInt32
  var padding: SIMD3<Float> = .zero
}

/// Border line vertex
struct SnakeGuideInstance {
  var position: SIMD3<Float>
  var scale: SIMD3<Float>
  var color: SIMD4<Float>
}
