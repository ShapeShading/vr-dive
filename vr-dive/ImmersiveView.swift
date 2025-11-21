//
//  ImmersiveView.swift
//  vr-dive
//
//  Created by chen on 2025/11/21.
//

import RealityKit
import RealityKitContent
import SwiftUI

struct ImmersiveView: View {
  @Environment(AppModel.self) var appModel

  var body: some View {
    RealityView { content in
      // Add the initial RealityKit content
      if let immersiveContentEntity = try? await Entity(
        named: "Immersive", in: realityKitContentBundle)
      {
        content.add(immersiveContentEntity)
      }
    }
    .onAppear {
      appModel.immersiveSpaceState = .open
    }
    .onDisappear {
      appModel.immersiveSpaceState = .closed
    }
  }
}

#Preview(immersionStyle: .full) {
  ImmersiveView()
    .environment(AppModel())
}
