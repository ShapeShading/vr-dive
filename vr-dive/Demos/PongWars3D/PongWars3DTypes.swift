import simd

// Ball state for each of the 8 balls
struct BallState {
  var position: SIMD4<Float>  // xyz = position, w = region index (0-7)
  var velocity: SIMD4<Float>  // xyz = velocity, w = radius
  var colorAndPadding: SIMD4<Float>  // rgb = color, a = padding
}

// Voxel grid is 16x16x16 = 4096 voxels
// Each voxel stores its current color/owner (0-7 for 8 regions)
struct VoxelData {
  var ownerAndFlags: UInt32  // lower 8 bits = owner index, upper bits = flags
}

// Simulation uniforms passed to compute shader
struct PongWarsSimulationUniforms {
  var deltaTime: Float
  var globalTime: Float
  var gridSize: UInt32  // 16
  var ballCount: UInt32  // 8
  var worldSize: Float  // 4.0 meters
  var voxelSize: Float  // worldSize / gridSize = 0.25 meters
  var padding1: Float
  var padding2: Float
}

// Edge rendering uniforms
struct PongWarsSceneUniforms {
  var time: Float
  var layerCount: UInt32
  var padding: SIMD2<Float>
}
