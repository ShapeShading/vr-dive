//
//  vr_diveApp.swift
//  vr-dive
//
//  Created by chen on 2025/11/21.
//

import CompositorServices
import SwiftUI

@main
struct vr_diveApp: App {

  @State private var appModel = AppModel()
  @State private var avPlayerViewModel = AVPlayerViewModel()

  var body: some Scene {
    WindowGroup {
      if avPlayerViewModel.isPlaying {
        AVPlayerView(viewModel: avPlayerViewModel)
      } else {
        ContentView()
          .environment(appModel)
      }
    }
    .defaultSize(width: 480, height: 260)

    ImmersiveSpace(id: appModel.immersiveSpaceID) {
      CompositorLayer(configuration: VRConfiguration()) { layerRenderer in
        let renderer = Renderer(layerRenderer, patternCoordinator: appModel.patternCoordinator)
        renderer.startRenderLoop()
      }
      .onAppear {
        appModel.immersiveSpaceState = .open
      }
      .onDisappear {
        appModel.immersiveSpaceState = .closed
      }
    }
    .immersionStyle(selection: .constant(.full), in: .full)
  }
}
