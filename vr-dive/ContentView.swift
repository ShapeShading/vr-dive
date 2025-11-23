//
//  ContentView.swift
//  vr-dive
//
//  Created by chen on 2025/11/21.
//

import Observation
import RealityKit
import SwiftUI

struct ContentView: View {
  @Environment(AppModel.self) private var appModel

  var body: some View {
    VStack(spacing: 24) {
      PatternMenuView(model: appModel.patternMenuModel)
      ToggleImmersiveSpaceButton()
      ControlButtonsView(model: appModel.patternMenuModel)
    }
    .padding(.horizontal, 24)
    .padding(.vertical, 28)
    .frame(maxWidth: 420)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
  }
}

struct PatternMenuView: View {
  @Bindable var model: PatternMenuModel

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("图案切换")
        .font(.headline)
      Picker("当前图案", selection: $model.selectedPattern) {
        ForEach(VisualPatternKind.allCases) { pattern in
          Text(pattern.displayName).tag(pattern)
        }
      }
      .pickerStyle(.menu)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }
}

struct ControlButtonsView: View {
  @Bindable var model: PatternMenuModel

  var body: some View {
    HStack(spacing: 12) {
      Button(action: {
        model.reset()
      }) {
        Label("Reset", systemImage: "arrow.counterclockwise")
      }
      .buttonStyle(.bordered)

      Button(action: {
        model.isPaused.toggle()
      }) {
        Label(
          model.isPaused ? "Resume" : "Pause",
          systemImage: model.isPaused ? "play.fill" : "pause.fill")
      }
      .buttonStyle(.bordered)
    }
    .frame(maxWidth: .infinity, alignment: .center)
  }
}

#Preview(windowStyle: .automatic) {
  ContentView()
    .environment(AppModel())
}
