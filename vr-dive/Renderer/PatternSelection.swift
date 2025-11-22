import Foundation
import Observation

enum VisualPatternKind: String, CaseIterable, Identifiable {
  case cubeField
  case lorenzAttractor

  var id: String { rawValue }

  var displayName: String {
    switch self {
    case .cubeField:
      return "立方体"
    case .lorenzAttractor:
      return "Lorenz 吸引子"
    }
  }
}

final class PatternCoordinator {
  private let queue = DispatchQueue(label: "vr-dive.pattern.coordinator", attributes: .concurrent)
  private var _current: VisualPatternKind = .lorenzAttractor

  func currentPattern() -> VisualPatternKind {
    queue.sync { _current }
  }

  func setPattern(_ pattern: VisualPatternKind) {
    queue.async(flags: .barrier) { self._current = pattern }
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

  private let coordinator: PatternCoordinator

  init(coordinator: PatternCoordinator) {
    self.coordinator = coordinator
    self.selectedPattern = coordinator.currentPattern()
  }

  func refreshFromCoordinator() {
    selectedPattern = coordinator.currentPattern()
  }
}
