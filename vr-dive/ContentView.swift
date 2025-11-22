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

#Preview(windowStyle: .automatic) {
  ContentView()
    .environment(AppModel())
}
