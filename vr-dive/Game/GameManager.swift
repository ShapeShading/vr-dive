import Foundation
import GameController
import SwiftUI
import simd

@Observable
class GameManager {
  private struct ControllerState {
    var leftStick: SIMD2<Float> = .zero
    var rightStick: SIMD2<Float> = .zero
    var buttonA: Bool = false
  }

  private let controllerQueue = DispatchQueue(label: "vr-dive.controller.state")
  private var controllerState = ControllerState()
  private var lastInputLogTime: TimeInterval = 0

  private let movementSpeed: Float = 1.2
  private let verticalSpeed: Float = 0.8
  private let yawSpeed: Float = .pi / 2.0
  private let deadZone: Float = 0.12

  private(set) var playerOffset: SIMD3<Float> = .zero
  private(set) var yawAngle: Float = 0
  private(set) var rigTransform: simd_float4x4 = matrix_identity_float4x4

  init() {
    setupControllerObserver()
  }

  func setupControllerObserver() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(controllerDidConnect), name: .GCControllerDidConnect, object: nil)
    NotificationCenter.default.addObserver(
      self, selector: #selector(controllerDidDisconnect), name: .GCControllerDidDisconnect,
      object: nil)

    GCController.startWirelessControllerDiscovery(completionHandler: nil)
    for controller in GCController.controllers() {
      register(controller: controller)
    }
  }

  @objc private func controllerDidConnect(notification: Notification) {
    guard let controller = notification.object as? GCController else { return }
    register(controller: controller)
  }

  @objc private func controllerDidDisconnect(notification: Notification) {
    guard let controller = notification.object as? GCController else { return }
    print("[GameManager] Controller disconnected: \(controller.vendorName ?? "Unknown Controller")")
  }

  private func register(controller: GCController) {
    print("[GameManager] Controller connected: \(controller.vendorName ?? "Unknown Controller")")
    guard let gamepad = controller.extendedGamepad else {
      print("[GameManager] Connected controller has no extended gamepad profile")
      return
    }

    gamepad.valueChangedHandler = { [weak self] gamepad, element in
      self?.handleInput(gamepad: gamepad, element: element)
    }
  }

  private func handleInput(gamepad: GCExtendedGamepad, element: GCControllerElement) {
    let leftStick = SIMD2<Float>(
      gamepad.leftThumbstick.xAxis.value, gamepad.leftThumbstick.yAxis.value)
    let rightStick = SIMD2<Float>(
      gamepad.rightThumbstick.xAxis.value, gamepad.rightThumbstick.yAxis.value)
    let buttonA = gamepad.buttonA.isPressed

    controllerQueue.sync {
      controllerState.leftStick = leftStick
      controllerState.rightStick = rightStick
      controllerState.buttonA = buttonA
    }

    logInputEvent(element: element, leftStick: leftStick, rightStick: rightStick, buttonA: buttonA)
  }

  private func logInputEvent(
    element: GCControllerElement, leftStick: SIMD2<Float>, rightStick: SIMD2<Float>,
    buttonA: Bool
  ) {
    let now = Date().timeIntervalSince1970
    guard now - lastInputLogTime > 0.05 else { return }
    lastInputLogTime = now

    let elementName = String(describing: type(of: element))
    let formattedLeft = String(format: "(%.2f, %.2f)", leftStick.x, leftStick.y)
    let formattedRight = String(format: "(%.2f, %.2f)", rightStick.x, rightStick.y)
    print(
      "[GameManager] Input \(elementName) left=\(formattedLeft) right=\(formattedRight) A=\(buttonA)"
    )
  }

  func updateRigState(deltaTime: Float, headTransform: simd_float4x4) -> simd_float4x4 {
    controllerQueue.sync {
      _ = headTransform  // head pose available for future drift-compensation tweaks
      let planarInput = applyDeadZone(controllerState.rightStick)
      let verticalYawInput = applyDeadZone(controllerState.leftStick)

      yawAngle -= verticalYawInput.x * yawSpeed * deltaTime
      yawAngle = wrapAngle(yawAngle)
      let basis = movementBasisVectors()

      var displacement = SIMD3<Float>.zero
      displacement -= basis.forward * planarInput.y  // first-person: push forward to move forward (scene pulls back)
      displacement -= basis.right * planarInput.x    // first-person: push left to strafe left (scene moves right)
      playerOffset += displacement * movementSpeed * deltaTime
      playerOffset.y -= verticalYawInput.y * verticalSpeed * deltaTime

      rigTransform = buildRigTransform()
      return rigTransform
    }
  }

  func currentRigTransform() -> simd_float4x4 {
    controllerQueue.sync { rigTransform }
  }

  private func movementBasisVectors() -> (forward: SIMD3<Float>, right: SIMD3<Float>) {
    let cosYaw = cos(yawAngle)
    let sinYaw = sin(yawAngle)
    let forward = SIMD3<Float>(-sinYaw, 0, -cosYaw)
    let right = SIMD3<Float>(cosYaw, 0, -sinYaw)
    return (forward, right)
  }

  private func applyDeadZone(_ input: SIMD2<Float>) -> SIMD2<Float> {
    let magnitude = simd_length(input)
    guard magnitude > deadZone else { return .zero }
    let scaled = (magnitude - deadZone) / (1 - deadZone)
    return (input / max(magnitude, 0.0001)) * scaled
  }

  private func wrapAngle(_ angle: Float) -> Float {
    var value = angle
    let twoPi: Float = .pi * 2
    value = fmod(value, twoPi)
    if value > .pi {
      value -= twoPi
    } else if value < -.pi {
      value += twoPi
    }
    return value
  }

  private func buildRigTransform() -> simd_float4x4 {
    let cosYaw = cos(-yawAngle)
    let sinYaw = sin(-yawAngle)
    let rotation = simd_float4x4(
      SIMD4<Float>(cosYaw, 0, sinYaw, 0),
      SIMD4<Float>(0, 1, 0, 0),
      SIMD4<Float>(-sinYaw, 0, cosYaw, 0),
      SIMD4<Float>(0, 0, 0, 1)
    )

    let translation = simd_float4x4(
      SIMD4<Float>(1, 0, 0, 0),
      SIMD4<Float>(0, 1, 0, 0),
      SIMD4<Float>(0, 0, 1, 0),
      SIMD4<Float>(-playerOffset.x, -playerOffset.y, -playerOffset.z, 1)
    )

    return translation * rotation
  }
}
