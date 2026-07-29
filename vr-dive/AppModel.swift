//
//  AppModel.swift
//  vr-dive
//
//  Created by chen on 2025/11/21.
//

import SwiftUI

/// Maintains app-wide state
@MainActor
@Observable
class AppModel {
  let immersiveSpaceID = "ImmersiveSpace"
  enum ImmersiveSpaceState {
    case closed
    case inTransition
    case open
  }
  var immersiveSpaceState = ImmersiveSpaceState.closed {
    didSet {
      print("[AppModel] immersiveSpaceState changed: \(oldValue) -> \(immersiveSpaceState)")
    }
  }

  let patternCoordinator = PatternCoordinator()
  let gameManager = GameManager()
  var patternMenuModel: PatternMenuModel

  init() {
    self.patternMenuModel = PatternMenuModel(coordinator: patternCoordinator)
  }
}
