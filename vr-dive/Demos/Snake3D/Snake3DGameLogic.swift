import Foundation
import simd

final class Snake3DGameLogic {

  // MARK: - Public State
  private(set) var state: Snake3DState

  // MARK: - Constants
  static let maxFoodCount = 256
  static let initialMoveInterval: TimeInterval = 0.4
  static let minMoveInterval: TimeInterval = 0.15
  static let speedUpPerScore = 5  // every N score points, speed up
  static let speedUpFactor: Double = 0.95

  // MARK: - Private
  private var lastMoveTime: TimeInterval = 0
  private var moveInterval: TimeInterval = initialMoveInterval
  private var hasStartedMoving = false

  // MARK: - Init

  init() {
    state = Snake3DState()
    spawnFoodIfNeeded()
  }

  // MARK: - Game Loop

  func update(currentTime: TimeInterval) {
    guard !state.isGameOver else { return }
    if !hasStartedMoving {
      hasStartedMoving = true
      lastMoveTime = currentTime
      return
    }

    while currentTime - lastMoveTime >= moveInterval {
      lastMoveTime += moveInterval
      step()
      if state.isGameOver {
        break
      }
    }
  }

  // MARK: - Input

  /// Called by renderer when a direction button is pressed.
  /// `newDirection` is expressed in the snake's *current* local coordinate frame.
  /// The game logic just records the new world axis direction.
  func requestTurn(to newDirection: SnakeDirection) {
    guard newDirection != state.direction,
      newDirection != state.direction.opposite
    else { return }
    state.pendingDirection = newDirection
  }

  func reset() {
    state = Snake3DState()
    lastMoveTime = 0
    moveInterval = Self.initialMoveInterval
    hasStartedMoving = false
    spawnFoodIfNeeded()
  }

  // MARK: - Step

  private func step() {
    // Apply pending turn
    if let pending = state.pendingDirection {
      state.direction = pending
      state.pendingDirection = nil
    }

    let head = state.segments[0]
    let newHead = head &+ state.direction.delta

    // Collision: self
    if state.segments.contains(newHead) {
      state.isGameOver = true
      return
    }

    // Move: prepend new head
    state.segments.insert(newHead, at: 0)

    // Check food
    if let foodIndex = state.foods.firstIndex(of: newHead) {
      // Eat food
      state.foods.remove(at: foodIndex)
      state.score += 1
      state.pendingGrow += 2  // grow 2 extra segments per food
      updateMoveInterval()
    }

    // Remove tail unless growing
    if state.pendingGrow > 0 {
      state.pendingGrow -= 1
    } else {
      state.segments.removeLast()
    }

    spawnFoodIfNeeded()
  }

  // MARK: - Food

  private func spawnFoodIfNeeded() {
    let g = Snake3DState.gridSize
    var attempts = 0
    while state.foods.count < Self.maxFoodCount, attempts < 1000 {
      attempts += 1
      let candidate = SIMD3<Int>(
        Int.random(in: 0..<g),
        Int.random(in: 0..<g),
        Int.random(in: 0..<g)
      )
      if !state.segments.contains(candidate) && !state.foods.contains(candidate) {
        state.foods.append(candidate)
      }
    }
  }

  // MARK: - Speed

  private func updateMoveInterval() {
    let speedUps = state.score / Self.speedUpPerScore
    let interval = Self.initialMoveInterval * pow(Self.speedUpFactor, Double(speedUps))
    moveInterval = max(interval, Self.minMoveInterval)
  }

  // MARK: - Rendering Data

  /// Returns segment instances (position in *logic grid* coordinates, not yet rotated).
  /// The renderer will apply world rotation on the CPU before uploading to GPU,
  /// or pass it via uniforms.
  func getSegmentPositions() -> [(gridPos: SIMD3<Int>, normalizedIndex: Float)] {
    let count = max(state.segments.count, 1)
    let denominator = Float(max(count - 1, 1))
    return state.segments.enumerated().map { i, pos in
      (pos, Float(i) / denominator)
    }
  }

  func getInterpolatedSegmentPositions(currentTime: TimeInterval) -> [(
    gridPos: SIMD3<Float>, normalizedIndex: Float
  )] {
    let count = max(state.segments.count, 1)
    let denominator = Float(max(count - 1, 1))
    let alpha = interpolationAlpha(currentTime: currentTime)
    let predicted = predictedSegmentsForCurrentDirection()

    return state.segments.enumerated().map { index, pos in
      let start = SIMD3<Float>(Float(pos.x), Float(pos.y), Float(pos.z))
      let targetInt = predicted[min(index, predicted.count - 1)]
      let target = SIMD3<Float>(Float(targetInt.x), Float(targetInt.y), Float(targetInt.z))
      let interpolated = simd_mix(start, target, SIMD3<Float>(repeating: alpha))
      return (interpolated, Float(index) / denominator)
    }
  }

  func getInterpolatedHeadPosition(currentTime: TimeInterval) -> SIMD3<Float> {
    guard let head = getInterpolatedSegmentPositions(currentTime: currentTime).first?.gridPos else {
      return SIMD3<Float>(
        Float(Snake3DState.gridSize / 2), Float(Snake3DState.gridSize / 2),
        Float(Snake3DState.gridSize / 2))
    }
    return head
  }

  func getFoodPositions() -> [SIMD3<Int>] {
    return state.foods
  }

  private func interpolationAlpha(currentTime: TimeInterval) -> Float {
    guard hasStartedMoving, moveInterval > 0 else { return 0 }
    let progress = max(0, min(1, (currentTime - lastMoveTime) / moveInterval))
    return Float(progress)
  }

  private func predictedSegmentsForCurrentDirection() -> [SIMD3<Int>] {
    guard let head = state.segments.first else { return [] }
    let nextHead = head &+ state.direction.delta
    var predicted = [nextHead]

    if state.segments.count > 1 {
      predicted.append(contentsOf: state.segments.dropLast())
    }

    return predicted
  }
}
