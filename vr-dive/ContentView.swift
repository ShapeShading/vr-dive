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
      HStack(alignment: .bottom, spacing: 20) {
        PatternMenuView(model: appModel.patternMenuModel)
        ToggleImmersiveSpaceButton()
      }
      ControlButtonsView(model: appModel.patternMenuModel, gameManager: appModel.gameManager)
    }
    .padding(.horizontal, 28)
    .padding(.vertical, 32)
    .frame(maxWidth: 980, minHeight: 440)
    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
  }
}

struct PatternMenuView: View {
  @Bindable var model: PatternMenuModel

  private var nextPattern: VisualPatternKind {
    let all = VisualPatternKind.allCases
    let idx = all.firstIndex(of: model.selectedPattern) ?? 0
    return all[(idx + 1) % all.count]
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("图案切换")
        .font(.headline)

      HStack(spacing: 10) {
        Picker("当前图案", selection: $model.selectedPattern) {
          ForEach(VisualPatternKind.allCases) { pattern in
            Text(pattern.displayName).tag(pattern)
          }
        }
        .pickerStyle(.menu)

        Button(action: { model.selectedPattern = nextPattern }) {
          VStack(spacing: 1) {
            Image(systemName: "chevron.right")
              .font(.caption.weight(.semibold))
            Text(nextPattern.displayName)
              .font(.system(size: 9))
              .lineLimit(1)
          }
          .frame(minWidth: 64)
        }
        .buttonStyle(.bordered)
      }
    }
    .frame(minWidth: 360, maxWidth: .infinity, alignment: .leading)
  }
}

struct ControlButtonsView: View {
  @Bindable var model: PatternMenuModel
  var gameManager: GameManager

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
            model.speedMultiplier > 1.0 ? "x5" : "x1",
            systemImage: model.speedMultiplier > 1.0 ? "hare.fill" : "tortoise.fill")
        }
        .buttonStyle(.bordered)

        Button(action: {
          gameManager.togglePatternNavigation()
        }) {
          Label(
            gameManager.isPatternNavigationActive ? "箱内移动 ON" : "箱内移动 OFF",
            systemImage: gameManager.isPatternNavigationActive
              ? "scope" : "scope")
        }
        .buttonStyle(.bordered)
        .tint(gameManager.isPatternNavigationActive ? .blue : nil)
      }

      if model.selectedPattern == .rayMarchingDemo {
        Button(action: {
          model.cycleRayMarchingProbeDimTarget()
        }) {
          Label(model.rayMarchingProbeDimTarget.buttonTitle, systemImage: "circle.lefthalf.filled")
        }
        .buttonStyle(.bordered)
      }

      if model.selectedPattern == .huashan {
        VStack(alignment: .leading, spacing: 8) {
          Text("华山点比例")
            .font(.headline)

          HStack(spacing: 14) {
            Button(action: {
              model.adjustHuashanSampleRatio(by: -0.05)
            }) {
              Label("减少 5%", systemImage: "minus")
            }
            .buttonStyle(.bordered)

            Text(model.huashanSampleRatioPercentText)
              .font(.system(.body, design: .monospaced).weight(.semibold))
              .frame(minWidth: 52)

            Button(action: {
              model.adjustHuashanSampleRatio(by: 0.05)
            }) {
              Label("增加 5%", systemImage: "plus")
            }
            .buttonStyle(.bordered)
          }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
      }

      if model.selectedPattern == .simoneOrbit3D {
        HStack(alignment: .top, spacing: 24) {
          VStack(alignment: .leading, spacing: 10) {
            Text("3D Simone Orbit")
              .font(.headline)

            Text(model.simoneOrbit3DPrincipleText)
              .font(.caption)
              .foregroundStyle(.secondary)

            Picker("参数预设", selection: $model.simoneOrbit3DPreset) {
              ForEach(SimoneOrbit3DPreset.allCases) { preset in
                Text(preset.pickerTitle).tag(preset)
              }
            }
            .pickerStyle(.menu)

            HStack(spacing: 10) {
              Button {
                model.simoneOrbit3DPreset = model.simoneOrbit3DPreset.previous()
              } label: {
                Label("上一个", systemImage: "chevron.left")
              }
              .buttonStyle(.bordered)

              Button {
                model.simoneOrbit3DPreset = model.simoneOrbit3DPreset.next()
              } label: {
                Label("下一个", systemImage: "chevron.right")
              }
              .buttonStyle(.bordered)
            }
          }
          .frame(maxWidth: .infinity, alignment: .leading)

          VStack(alignment: .leading, spacing: 10) {
            let params = model.simoneOrbit3DPreset.parameters
            Text(
              "当前参数: a=\(params.x, specifier: "%.2f")  b=\(params.y, specifier: "%.2f")  c=\(params.z, specifier: "%.2f")"
            )
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(.secondary)

            Text("搜索摘要: \(model.simoneOrbit3DPreset.metricsSummary)")
              .font(.system(.caption, design: .monospaced))
              .foregroundStyle(.secondary)
              .fixedSize(horizontal: false, vertical: true)
          }
          .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
      }

      if model.selectedPattern == .dynamicBox {
        VStack(alignment: .leading, spacing: 8) {
          Text("动态着色器加载")
            .font(.headline)

          HStack(spacing: 10) {
            TextField("着色器名称", text: $model.dynamicBoxShaderInput)
              .textFieldStyle(.roundedBorder)
              .frame(width: 160)

            Button(action: {
              model.loadDynamicBoxShader()
            }) {
              Label("加载", systemImage: "arrow.down.doc")
            }
            .buttonStyle(.bordered)
          }

          Text("状态: \(model.dynamicBoxStatus)")
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .task {
          // Poll status from coordinator while this view is visible
          while !Task.isCancelled {
            model.refreshDynamicBoxStatus()
            try? await Task.sleep(for: .milliseconds(200))
          }
        }
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
