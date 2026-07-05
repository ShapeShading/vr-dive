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
    var buttonB: Bool = false
    var buttonX: Bool = false
    var buttonY: Bool = false
    var boostActive: Bool = false
    // D-pad for Tetris
    var dpadUp: Bool = false
    var dpadDown: Bool = false
    var dpadLeft: Bool = false
    var dpadRight: Bool = false
    var leftShoulder: Bool = false
    var rightShoulder: Bool = false
    // Rising-edge counter for square (□) button — incremented in handleInput,
    // drained in updateRigState. Using a counter (rather than a single bool)
    // means every physical/UI press is honored even if several presses land
    // between two updateRigState ticks (e.g. during a frame-rate stall) —
    // a bool latch would silently collapse multiple presses into at most one
    // toggle, making the switch feel like it "doesn't work" under lag.
    var squareJustPressedCount: Int = 0
  }

  private let controllerQueue = DispatchQueue(label: "vr-dive.controller.state")
  // GCController's `valueChangedHandler` defaults to firing on the MAIN
  // queue/run loop. During a heavy render stall (this app can stall for
  // hundreds of ms on some patterns), or while SwiftUI/visionOS is running
  // the main run loop in a restricted "tracking" mode (e.g. mid-gesture),
  // queued main-queue blocks — including a stick's release-to-center event —
  // can sit undelivered indefinitely. That's what caused input to appear
  // "stuck" (still moving after releasing the stick) until an unrelated pinch
  // forced the main run loop back to a common mode and flushed the queue.
  // Routing the handler to its own dedicated background queue makes every
  // value-changed event (press AND release) get delivered promptly,
  // independent of main-thread/run-loop state.
  private let controllerHandlerQueue = DispatchQueue(label: "vr-dive.controller.handler")
  private var controllerState = ControllerState()
  private var lastInputLogTime: TimeInterval = 0
  private var lastPatternNavLogTime: TimeInterval = 0

  private let movementSpeed: Float = 1.2
  private let yawSpeed: Float = .pi / 2.0
  private let deadZone: Float = 0.2
  private let residualStickClamp: Float = 0.025
  private let boostMovementMultiplier: Float = 5.0
  private let boostYawMultiplier: Float = 2.0
  // L1 + R1 held together — an extra multiplier stacked on top of the
  // regular single-shoulder boost, for covering huge distances quickly
  // (e.g. the space elevator's ~25km shaft).
  private let superBoostMovementMultiplier: Float = 32.0

  private(set) var playerOffset: SIMD3<Float> = .zero
  private(set) var yawAngle: Float = 0
  private(set) var rigTransform: simd_float4x4 = matrix_identity_float4x4

  // Pattern navigation state (square / □ button activates)
  private var patternNavOffset: SIMD3<Float> = .zero
  private var patternNavYaw: Float = 0
  private(set) var isPatternNavigationActive: Bool = false
  private(set) var patternNavTransform: simd_float4x4 = matrix_identity_float4x4

  init() {
    setupControllerObserver()
  }

  // MARK: - Tetris Input (PS5 Controller)
  // △ = buttonY = 向上移动
  // × = buttonA = 快速下落
  // □ = buttonX = 切换方块类型
  // ○ = buttonB = 随机旋转朝向

  struct TetrisInput {
    var dpadUp: Bool
    var dpadDown: Bool
    var dpadLeft: Bool
    var dpadRight: Bool
    var buttonCross: Bool  // × = 快速下落
    var buttonTriangle: Bool  // △ = 向上移动
    var buttonSquare: Bool  // □ = 切换方块类型
    var buttonCircle: Bool  // ○ = 随机旋转朝向
    var buttonR1: Bool  // R1 = 加速前进
  }

  func getTetrisInput() -> TetrisInput {
    controllerQueue.sync {
      TetrisInput(
        dpadUp: controllerState.dpadUp,
        dpadDown: controllerState.dpadDown,
        dpadLeft: controllerState.dpadLeft,
        dpadRight: controllerState.dpadRight,
        buttonCross: controllerState.buttonA,  // PS5 × maps to buttonA
        buttonTriangle: controllerState.buttonY,  // PS5 △ maps to buttonY
        buttonSquare: controllerState.buttonX,  // PS5 □ maps to buttonX
        buttonCircle: controllerState.buttonB,  // PS5 ○ maps to buttonB
        buttonR1: controllerState.rightShoulder
      )
    }
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

    // See `controllerHandlerQueue` doc comment above: without this, input
    // change events (including releases) are delivered on the main queue and
    // can get stuck behind a stalled render loop or an in-progress gesture.
    controller.handlerQueue = controllerHandlerQueue

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
    let buttonB = gamepad.buttonB.isPressed
    let buttonX = gamepad.buttonX.isPressed
    let buttonY = gamepad.buttonY.isPressed
    let boostActive = gamepad.leftShoulder.isPressed || gamepad.rightShoulder.isPressed
    let leftShoulder = gamepad.leftShoulder.isPressed
    let rightShoulder = gamepad.rightShoulder.isPressed

    // D-pad
    let dpadUp = gamepad.dpad.up.isPressed
    let dpadDown = gamepad.dpad.down.isPressed
    let dpadLeft = gamepad.dpad.left.isPressed
    let dpadRight = gamepad.dpad.right.isPressed

    controllerQueue.sync {
      let prevButtonX = controllerState.buttonX  // capture before update for edge detection
      controllerState.leftStick = leftStick
      controllerState.rightStick = rightStick
      controllerState.buttonA = buttonA
      controllerState.buttonB = buttonB
      controllerState.buttonX = buttonX
      controllerState.buttonY = buttonY
      controllerState.boostActive = boostActive
      controllerState.dpadUp = dpadUp
      controllerState.dpadDown = dpadDown
      controllerState.dpadLeft = dpadLeft
      controllerState.dpadRight = dpadRight
      controllerState.leftShoulder = leftShoulder
      controllerState.rightShoulder = rightShoulder
      // Rising edge: square button just pressed this event
      if buttonX && !prevButtonX {
        controllerState.squareJustPressedCount += 1
        print("[GameManager] Square (□) rising edge detected")
      }
    }

    logInputEvent(
      element: element, leftStick: leftStick, rightStick: rightStick, buttonA: buttonA,
      boost: boostActive)
  }

  private func logInputEvent(
    element: GCControllerElement, leftStick: SIMD2<Float>, rightStick: SIMD2<Float>,
    buttonA: Bool, boost: Bool
  ) {
    let now = Date().timeIntervalSince1970
    guard now - lastInputLogTime > 0.05 else { return }
    lastInputLogTime = now

    let elementName = String(describing: type(of: element))
    let formattedLeft = String(format: "(%.2f, %.2f)", leftStick.x, leftStick.y)
    let formattedRight = String(format: "(%.2f, %.2f)", rightStick.x, rightStick.y)
    print(
      "[GameManager] Input \(elementName) left=\(formattedLeft) right=\(formattedRight) A=\(buttonA) boost=\(boost)"
    )
  }

  /// Call from any thread (e.g. UI button) to toggle pattern navigation mode.
  /// Uses the same latch mechanism as the gamepad square button so the change
  /// is applied atomically on the next render-loop tick.
  func togglePatternNavigation() {
    controllerQueue.sync {
      controllerState.squareJustPressedCount += 1
    }
  }

  func updateRigState(deltaTime: Float, headTransform: simd_float4x4) -> simd_float4x4 {
    controllerQueue.sync {
      let primaryStickInput = applyDeadZone(controllerState.leftStick)
      let secondaryStickInput = applyDeadZone(controllerState.rightStick)
      // L1 + R1 held together stacks an extra 10x on top of the normal boost
      // (useful for quickly traversing the space elevator's huge shaft).
      let superBoostActive = controllerState.leftShoulder && controllerState.rightShoulder
      let movementMultiplier =
        (controllerState.boostActive ? boostMovementMultiplier : 1.0)
        * (superBoostActive ? superBoostMovementMultiplier : 1.0)
      let yawMultiplier = controllerState.boostActive ? boostYawMultiplier : 1.0

      let forwardInput = primaryStickInput.y
      let yawInput = primaryStickInput.x
      let strafeInput = secondaryStickInput.x
      let verticalInput = secondaryStickInput.y

      // Square button (□) — drain the rising-edge counter. Each press toggles
      // the mode once; draining all pending presses (instead of a single
      // bool flag) means a burst of presses that lands between two ticks
      // (e.g. during a lag spike) still nets out to the correct final state
      // instead of silently losing all but one press.
      if controllerState.squareJustPressedCount > 0 {
        let presses = controllerState.squareJustPressedCount
        controllerState.squareJustPressedCount = 0
        if presses % 2 == 1 {
          isPatternNavigationActive.toggle()
        }
        print(
          "[GameManager] Pattern nav mode: \(isPatternNavigationActive ? "ON" : "OFF") (drained \(presses) press(es))"
        )
      }

      let turnSpeedReduction = 1.0 - abs(yawInput) * 0.5

      if isPatternNavigationActive {
        // ── Pattern navigation mode ──────────────────────────────────────────
        // LEFT X  = rotate virtual scene view (patternNavYaw)
        // LEFT Y / RIGHT X / RIGHT Y = translate, following the direction the
        //   player is CURRENTLY virtually facing (real head direction plus the
        //   patternNavYaw twist from the left stick), so pushing forward always
        //   walks toward what's rendered in front of you instead of a fixed
        //   box-relative direction.
        // Camera rig stays frozen.
        patternNavYaw -= yawInput * yawSpeed * yawMultiplier * turnSpeedReduction * deltaTime
        patternNavYaw = wrapAngle(patternNavYaw)

        // Build world-space direction vectors from the real head transform,
        // additionally rotated by patternNavYaw so "forward" always tracks the
        // CURRENT virtual look direction the player sees (frozen rig yaw +
        // live head rotation + the virtual patternNavYaw twist applied on top
        // in the shader via patternTransform). Without adding patternNavYaw
        // here, movement direction stayed pinned to a fixed box-relative
        // direction even after turning with the left stick, since yawAngle
        // alone never changes while pattern nav is active — only patternNavYaw
        // does. Y-axis rotations compose additively, so `yawAngle +
        // patternNavYaw` is exactly the combined rotation applied to `rd` in
        // the shader (R(-patternNavYaw) * R(-yawAngle) == R(-(yawAngle +
        // patternNavYaw))).
        let effectiveYaw = yawAngle + patternNavYaw
        let cosYaw = cos(-effectiveYaw)
        let sinYaw = sin(-effectiveYaw)
        let rigRot = simd_float3x3(
          SIMD3<Float>(cosYaw, 0, sinYaw),
          SIMD3<Float>(0, 1, 0),
          SIMD3<Float>(-sinYaw, 0, cosYaw)
        )
        let headRight = SIMD3<Float>(
          headTransform.columns.0.x, headTransform.columns.0.y, headTransform.columns.0.z)
        let headUp = SIMD3<Float>(
          headTransform.columns.1.x, headTransform.columns.1.y, headTransform.columns.1.z)
        let headForward = -SIMD3<Float>(
          headTransform.columns.2.x, headTransform.columns.2.y, headTransform.columns.2.z)
        let worldForward = rigRot * headForward
        let worldRight = rigRot * headRight
        let worldUp = rigRot * headUp

        patternNavOffset -=
          worldForward * forwardInput * movementSpeed * movementMultiplier * deltaTime
        patternNavOffset -=
          worldRight * strafeInput * movementSpeed * movementMultiplier * deltaTime
        patternNavOffset -= worldUp * verticalInput * movementSpeed * movementMultiplier * deltaTime

        patternNavTransform = buildPatternNavTransform()

        // Throttled log — shows both sticks so right-stick capture is verifiable
        let anyInput = abs(forwardInput) + abs(yawInput) + abs(strafeInput) + abs(verticalInput)
        let nowNav = Date().timeIntervalSince1970
        if anyInput > 0 && nowNav - lastPatternNavLogTime > 2.0 {
          lastPatternNavLogTime = nowNav
          let o = patternNavOffset
          print(
            String(
              format:
                "[GameManager] patNav L=(%.2f,%.2f) R=(%.2f,%.2f) offset=(%.2f,%.2f,%.2f) yaw=%.2f",
              yawInput, forwardInput, strafeInput, verticalInput,
              o.x, o.y, o.z, patternNavYaw))
        }
        // rigTransform unchanged
      } else {
        // ── Normal mode: camera rig update (original behaviour) ──────────────
        yawAngle -= yawInput * yawSpeed * yawMultiplier * turnSpeedReduction * deltaTime
        yawAngle = wrapAngle(yawAngle)

        let cosYaw = cos(-yawAngle)
        let sinYaw = sin(-yawAngle)
        let rigRotation = simd_float3x3(
          SIMD3<Float>(cosYaw, 0, sinYaw),
          SIMD3<Float>(0, 1, 0),
          SIMD3<Float>(-sinYaw, 0, cosYaw)
        )

        let headRight = SIMD3<Float>(
          headTransform.columns.0.x, headTransform.columns.0.y, headTransform.columns.0.z)
        let headUp = SIMD3<Float>(
          headTransform.columns.1.x, headTransform.columns.1.y, headTransform.columns.1.z)
        let headForward = -SIMD3<Float>(
          headTransform.columns.2.x, headTransform.columns.2.y, headTransform.columns.2.z)

        let worldForward = rigRotation * headForward
        let worldRight = rigRotation * headRight
        let worldUp = rigRotation * headUp

        var displacement = SIMD3<Float>.zero
        displacement -= worldForward * forwardInput
        displacement -= worldRight * strafeInput
        displacement -= worldUp * verticalInput

        playerOffset += displacement * movementSpeed * movementMultiplier * deltaTime

        rigTransform = buildRigTransform()
      }

      return rigTransform
    }
  }

  func currentRigTransform() -> simd_float4x4 {
    controllerQueue.sync { rigTransform }
  }

  private func applyDeadZone(_ input: SIMD2<Float>) -> SIMD2<Float> {
    let magnitude = simd_length(input)
    guard magnitude > deadZone else { return .zero }
    let scaled = (magnitude - deadZone) / (1 - deadZone)
    var filtered = (input / max(magnitude, 0.0001)) * scaled
    if abs(filtered.x) < residualStickClamp {
      filtered.x = 0
    }
    if abs(filtered.y) < residualStickClamp {
      filtered.y = 0
    }
    return filtered
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

  private func buildPatternNavTransform() -> simd_float4x4 {
    let cosYaw = cos(-patternNavYaw)
    let sinYaw = sin(-patternNavYaw)
    // Same convention as buildRigTransform: T(-offset) * R(-yaw), i.e. rotate
    // first, then subtract the (unrotated) world-space offset. This keeps
    // patternNavOffset in an absolute reference frame so that turning
    // (patternNavYaw changing) never re-rotates the already-accumulated
    // offset — it only changes look direction, exactly like normal-mode view
    // rotation. The previous `rotation * translation` order instead rotated
    // the offset itself every time the yaw changed, which made turning after
    // having moved feel like an unwanted orbit/strafe around the origin.
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
      SIMD4<Float>(-patternNavOffset.x, -patternNavOffset.y, -patternNavOffset.z, 1)
    )
    return translation * rotation
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
