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
- **Controller Support**: Supports Bluetooth Game Controllers (e.g., PS5 DualSense).
  - **Left Stick**: Move camera (swim).
  - **Button A (Cross)**: Shoot projectiles.

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
2.  **Movement**:
    - Use the **Left Thumbstick** to move the camera forward/backward and strafe left/right.
    - You should see the scene moving relative to you.
3.  **Shooting**:
    - Press the **Cross (X)** button (mapped to Button A).
    - You should see red spherical projectiles firing from your position in the direction you are looking.
    - Projectiles will bounce off the "fish" if they hit it.

## Technical Details

- **Renderer**: Custom `CompositorLayer` renderer using Metal.
- **Shaders**: Raymarching Signed Distance Fields (SDF) in `Shaders.metal`.
- **Input**: `GameController` framework for handling gamepad input.
