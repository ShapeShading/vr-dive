# VR Dive

A simple VR diving experience for Apple Vision Pro using CompositorServices and Metal Raymarching.

## Features

- **Immersive Rendering**: Uses Metal and Raymarching (SDFs) to render a 3D underwater scene.
- **Multiple Objects**: Renders seabed, corals, fish, ancient ruins (box), and strange rings (torus).
- **Different Shapes & Colors**:
  - **Seabed**: Sand color, wavy plane.
  - **Corals**: Pink/Purple spheres.
  - **Fish**: Orange, deformed sphere animation.
  - **Ruins**: Wood/Brown box.
  - **Ring**: Green rotating torus.
  - **Projectiles**: Red glowing spheres.
- **手柄支持**：兼容蓝牙游戏手柄（如 PS5 DualSense）。
  - **右摇杆**：以第一人称视角控制平移——向前推，画面后移（角色前进）；向左推，画面右移（角色左移）。
  - **左摇杆**：同样采取第一人称方向——向前推，画面下移（角色上浮）；向左推，画面右转（角色左转）。
  - **× 键（Button A）**：预留给后续交互。

## 手柄映射与调试

- DualSense 输入会在 Xcode 控制台输出（前缀为 `[GameManager]`），可用来确认模拟器或真机是否正确转发事件。
- 运动通过额外挂载的“机位变换”实现，你可以一边自由转头观察环境，一边用左右摇杆微调潜艇的位置与朝向。

## How to Verify

### 1. Rendering Verification

1.  Build and run the app on **Apple Vision Pro Simulator**.
2.  Enter the "Immersive Space" by tapping the button in the window.
3.  Look around. You should see:
    - A blue underwater environment.
    - A sandy seabed below.
    - Pink coral spheres scattered around.
    - An orange "fish" swimming in a circle.
    - A wooden box (ruins) at position `(-3, -1, 8)`.
    - A green rotating torus at position `(3, 1, 8)`.

### 2. Controller Verification (PS5 DualSense)

1.  **Connect Controller**:
    - **Simulator**: Connect a supported game controller to your Mac via Bluetooth or USB. The Simulator should automatically detect it.
    - **Device**: Pair the PS5 controller with the Vision Pro via Bluetooth settings.
2.  **移动体验**：

- **右摇杆**控制第一人称平移（向前推=角色前进，向左推=角色左移）。
- **左摇杆**的上下用于垂直移动、左右用于旋转，方向与第一人称保持一致（向前推=上浮，向左推=左转）。
- Xcode 控制台会输出 `[GameManager] Input ...` 日志，可用来验证事件是否成功传递。

## Technical Details

- **Renderer**: Custom `CompositorLayer` renderer using Metal.
- **Shaders**: Raymarching Signed Distance Fields (SDF) in `Shaders.metal`.
- **Input**: `GameController` framework for handling gamepad input.
