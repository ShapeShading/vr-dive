import Foundation
import GameController
import SwiftUI
import simd

struct Projectile {
  var position: SIMD3<Float>
  var velocity: SIMD3<Float>
  var active: Bool
  var radius: Float
}

@Observable
class GameManager {
  var cameraPosition: SIMD3<Float> = [0, 0, 0]
  var cameraForward: SIMD3<Float> = [0, 0, -1]
  var cameraRight: SIMD3<Float> = [1, 0, 0]

  var projectiles: [Projectile] = []

  private var lastUpdateTime: TimeInterval = 0

  init() {
    setupControllerObserver()
    // Init projectiles pool
    for _ in 0..<10 {
      projectiles.append(
        Projectile(position: [0, 0, 0], velocity: [0, 0, 0], active: false, radius: 0.1))
    }
  }

  func setupControllerObserver() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(controllerDidConnect), name: .GCControllerDidConnect, object: nil)
  }

  @objc func controllerDidConnect(notification: Notification) {
    guard let controller = notification.object as? GCController else { return }
    controller.extendedGamepad?.valueChangedHandler = { [weak self] (gamepad, element) in
      self?.handleInput(gamepad: gamepad)
    }
  }

  func handleInput(gamepad: GCExtendedGamepad) {
    // Movement
    let leftStick = gamepad.leftThumbstick
    let speed: Float = 0.1

    // Simple movement relative to camera orientation (which we need to update from ARKit or just manage here if we were fully virtual, but in VR we usually move the "rig" or offset)
    // For this demo, we will update a "world offset" or "virtual position".

    let moveDir = cameraForward * leftStick.yAxis.value + cameraRight * leftStick.xAxis.value
    cameraPosition += moveDir * speed

    // Shooting
    if gamepad.buttonA.isPressed {
      shoot()
    }
  }

  func shoot() {
    // Find inactive projectile
    for i in 0..<projectiles.count {
      if !projectiles[i].active {
        projectiles[i].active = true
        projectiles[i].position = cameraPosition + cameraForward * 0.5
        projectiles[i].velocity = cameraForward * 0.5
        break
      }
    }
  }

  func update(time: Float, deltaTime: TimeInterval, headTransform: simd_float4x4) {
    // Update camera vectors from head transform
    // headTransform is Camera -> World
    cameraForward = -simd_make_float3(headTransform.columns.2)
    cameraRight = simd_make_float3(headTransform.columns.0)

    // Update projectiles
    for i in 0..<projectiles.count {
      if projectiles[i].active {
        projectiles[i].position += projectiles[i].velocity

        // Simple collision with "fish" (hardcoded pos in shader for now, let's mirror it here)
        let fishPos = SIMD3<Float>(sin(time) * 3.0, 0.0, cos(time) * 3.0 + 5.0)
        if distance(projectiles[i].position, fishPos) < 0.6 {
          // Bounce
          projectiles[i].velocity *= -1.0
        }

        // Deactivate if too far
        if length(projectiles[i].position - cameraPosition) > 20.0 {
          projectiles[i].active = false
        }
      }
    }
  }
}
