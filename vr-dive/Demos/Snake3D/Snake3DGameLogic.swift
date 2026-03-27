import Foundation
import simd

final class Snake3DGameLogic {

  // MARK: - Public State
  private(set) var state: Snake3DState

  // MARK: - Constants
  static let maxFoodCount = 3
  static let initialMoveInterval: TimeInterval = 0.4
  static let minMoveInterval: TimeInterval = 0.15
  static let speedUpPerScore = 5  // every N score points, speed up
  static let speedUpFactor: Double = 0.95

  // MARK: - Private
  private var lastMoveTime: TimeInterval = 0
  private var moveInterval: TimeInterval = initialMoveInterval

  // MARK: - Init

  init() {
    state = Snake3DState()
    spawnFoodIfNeeded()
  }

  // MARK: - Game Loop

  func update(currentTime: TimeInterval) {
    guard !state.isGameOver else { return }
    guard currentTime - lastMoveTime >= moveInterval else { return }
    lastMoveTime = currentTime
    step()
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
    while state.foods.count < Self.maxFoodCount, attempts < 200 {
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

  func getFoodPositions() -> [SIMD3<Int>] {
    return state.foods
  }
}
