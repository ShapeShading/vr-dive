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
    VStack(spacing: 28) {
      HStack(alignment: .bottom, spacing: 20) {
        PatternMenuView(model: appModel.patternMenuModel)
        ToggleImmersiveSpaceButton()
      }
      ControlButtonsView(model: appModel.patternMenuModel)
    }
    .padding(.horizontal, 28)
    .padding(.vertical, 32)
    .frame(maxWidth: 560, minHeight: 280)
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
    VStack(spacing: 18) {
      HStack(spacing: 18) {
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

        Button(action: {
          model.toggleSpeed()
        }) {
          Label(
            model.speedMultiplier > 1.0 ? "x8" : "x1",
            systemImage: model.speedMultiplier > 1.0 ? "hare.fill" : "tortoise.fill")
        }
        .buttonStyle(.bordered)
      }

      if model.selectedPattern.supportsOriginCellInspection {
        Toggle(isOn: $model.originCellInspectionEnabled) {
          VStack(alignment: .leading, spacing: 2) {
            Text("原点胞高亮")
            Text("自动暂停并加粗高亮包含原点的一个胞")
              .font(.caption)
              .foregroundStyle(.secondary)
          }
        }
        .toggleStyle(.switch)
        .padding(.top, 2)
      }
    }
    .frame(maxWidth: .infinity, alignment: .center)
  }
}

#Preview(windowStyle: .automatic) {
  ContentView()
    .environment(AppModel())
}
