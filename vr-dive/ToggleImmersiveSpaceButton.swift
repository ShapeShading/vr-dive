//
//  ToggleImmersiveSpaceButton.swift
//  vr-dive
//
//  Created by chen on 2025/11/21.
//

import SwiftUI

struct ToggleImmersiveSpaceButton: View {

  @Environment(AppModel.self) private var appModel

  @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
  @Environment(\.openImmersiveSpace) private var openImmersiveSpace

  var body: some View {
    Button {
      Task { @MainActor in
        switch appModel.immersiveSpaceState {
        case .open:
          appModel.immersiveSpaceState = .inTransition
          await dismissImmersiveSpace()
          if appModel.immersiveSpaceState == .inTransition {
            // ImmersiveSpace.onDisappear() didn’t run (or hasn’t yet), so unblock the UI here.
            appModel.immersiveSpaceState = .closed
          }
        // Don't set immersiveSpaceState to .closed because there
        // are multiple paths to ImmersiveView.onDisappear().
        // Only set .closed in ImmersiveView.onDisappear().

        case .closed:
          appModel.immersiveSpaceState = .inTransition
          switch await openImmersiveSpace(id: appModel.immersiveSpaceID) {
          case .opened:
            // Don't set immersiveSpaceState to .open because there
            // may be multiple paths to ImmersiveView.onAppear().
            // Only set .open in ImmersiveView.onAppear().
            break

          case .userCancelled, .error:
            // On error, we need to mark the immersive space
            // as closed because it failed to open.
            fallthrough
          @unknown default:
            // On unknown response, assume space did not open.
            appModel.immersiveSpaceState = .closed
          }

        case .inTransition:
          // This case should not ever happen because button is disabled for this case.
          break
        }
      }
    } label: {
      Text(appModel.immersiveSpaceState == .open ? "退出沉浸模式" : "进入沉浸模式")
    }
    .disabled(appModel.immersiveSpaceState == .inTransition)
    .buttonStyle(.bordered)
    .animation(.none, value: 0)
    .font(.headline)
    .fontWeight(.semibold)
    .padding(.vertical, 4)
  }
}
