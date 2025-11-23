import Foundation
import Observation

enum VisualPatternKind: String, CaseIterable, Identifiable {
  case cubeField
  case lorenzAttractor
  case fourWingAttractor
  case aizawaAttractor

  var id: String { rawValue }

  var displayName: String {
    switch self {
    case .cubeField:
      return "立方体"
    case .lorenzAttractor:
      return "Lorenz 吸引子"
    case .fourWingAttractor:
      return "Four-Wing Attractor"
    case .aizawaAttractor:
      return "Aizawa 吸引子"
    }
  }
}

final class PatternCoordinator {
  private let queue = DispatchQueue(label: "vr-dive.pattern.coordinator", attributes: .concurrent)
  private var _current: VisualPatternKind = .lorenzAttractor
  private var _isPaused: Bool = false
  private var _shouldReset: Bool = false

  func currentPattern() -> VisualPatternKind {
    queue.sync { _current }
  }

  func setPattern(_ pattern: VisualPatternKind) {
    queue.async(flags: .barrier) { self._current = pattern }
  }
  
  func isPaused() -> Bool {
    queue.sync { _isPaused }
  }
  
  func setPaused(_ paused: Bool) {
    queue.async(flags: .barrier) { self._isPaused = paused }
  }
  
  func shouldReset() -> Bool {
    queue.sync { _shouldReset }
  }
  
  func triggerReset() {
    queue.async(flags: .barrier) { self._shouldReset = true }
  }
  
  func clearResetFlag() {
    queue.async(flags: .barrier) { self._shouldReset = false }
  }
}

@MainActor
@Observable
final class PatternMenuModel {
  var selectedPattern: VisualPatternKind {
    didSet {
      coordinator.setPattern(selectedPattern)
    }
  }
  
  var isPaused: Bool = false {
    didSet {
      coordinator.setPaused(isPaused)
    }
  }

  private let coordinator: PatternCoordinator

  init(coordinator: PatternCoordinator) {
    self.coordinator = coordinator
    self.selectedPattern = coordinator.currentPattern()
    self.isPaused = coordinator.isPaused()
  }

  func refreshFromCoordinator() {
    selectedPattern = coordinator.currentPattern()
    isPaused = coordinator.isPaused()
  }
  
  func reset() {
    coordinator.triggerReset()
  }
}
