#include <metal_stdlib>
using namespace metal;

struct BallState {
  float4 position;      // xyz = position, w = region index
  float4 velocity;      // xyz = velocity, w = radius
  float4 colorAndPadding; // rgb = color, a = padding
};

struct VoxelData {
  uint ownerAndFlags;
};

struct PongWarsSimulationUniforms {
  float deltaTime;
  float globalTime;
  uint gridSize;
  uint ballCount;
  float worldSize;
  float voxelSize;
  float padding1;
  float padding2;
};

struct PongWarsSceneUniforms {
  float time;
  uint layerCount;
  float2 padding;
};

struct EdgeVertex {
  float3 position;
  float3 color;
};

struct EdgeVertexOut {
  float4 position [[position]];
  float3 color;
};

// Helper function to get voxel index from 3D coordinates
uint getVoxelIndex(uint3 coord, uint gridSize) {
  return coord.x + coord.y * gridSize + coord.z * gridSize * gridSize;
}

// Helper function to get 3D coordinates from voxel index
uint3 getVoxelCoord(uint index, uint gridSize) {
  uint z = index / (gridSize * gridSize);
  uint remainder = index % (gridSize * gridSize);
  uint y = remainder / gridSize;
  uint x = remainder % gridSize;
  return uint3(x, y, z);
}

// Check if a voxel is at the boundary (has at least one face exposed to different color)
bool isEdgeVoxel(uint3 coord, uint gridSize, device VoxelData *voxels) {
  uint centerIndex = getVoxelIndex(coord, gridSize);
  uint centerOwner = voxels[centerIndex].ownerAndFlags & 0xFF;
  
  // Check 6 neighbors (faces)
  int3 offsets[6] = {
    int3(1, 0, 0), int3(-1, 0, 0),
    int3(0, 1, 0), int3(0, -1, 0),
    int3(0, 0, 1), int3(0, 0, -1)
  };
  
  for (int i = 0; i < 6; i++) {
    int3 neighborCoord = int3(coord) + offsets[i];
    if (neighborCoord.x < 0 || neighborCoord.x >= int(gridSize) ||
        neighborCoord.y < 0 || neighborCoord.y >= int(gridSize) ||
        neighborCoord.z < 0 || neighborCoord.z >= int(gridSize)) {
      return true; // Boundary of world
    }
    
    uint neighborIndex = getVoxelIndex(uint3(neighborCoord), gridSize);
    uint neighborOwner = voxels[neighborIndex].ownerAndFlags & 0xFF;
    if (neighborOwner != centerOwner) {
      return true; // Different color neighbor
    }
  }
  
  return false;
}

// Compute shader to simulate ball physics and voxel conversion
kernel void simulatePongWarsBalls(
  device BallState *balls [[buffer(0)]],
  device VoxelData *voxels [[buffer(1)]],
  constant PongWarsSimulationUniforms &uniforms [[buffer(2)]],
  uint id [[thread_position_in_grid]])
{
  if (id >= uniforms.ballCount) {
    return;
  }
  
  BallState ball = balls[id];
  float dt = uniforms.deltaTime;
  float halfWorld = uniforms.worldSize * 0.5;
  float radius = ball.velocity.w;
  
  // Update position
  float3 newPos = ball.position.xyz + ball.velocity.xyz * dt;
  float3 vel = ball.velocity.xyz;
  
  // Check boundary collisions
  bool collided = false;
  
  for (int axis = 0; axis < 3; axis++) {
    if (newPos[axis] - radius < -halfWorld) {
      newPos[axis] = -halfWorld + radius;
      vel[axis] = abs(vel[axis]);
      collided = true;
    } else if (newPos[axis] + radius > halfWorld) {
      newPos[axis] = halfWorld - radius;
      vel[axis] = -abs(vel[axis]);
      collided = true;
    }
  }
  
  // Check voxel collisions and convert voxels
  // Get voxel coordinates at ball position
  float3 worldToGrid = (newPos + halfWorld) / uniforms.voxelSize;
  int3 voxelCoord = int3(floor(worldToGrid));
  
  // Check surrounding voxels (3x3x3 region around ball)
  uint ballOwner = uint(ball.position.w);
  
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dz = -1; dz <= 1; dz++) {
        int3 checkCoord = voxelCoord + int3(dx, dy, dz);
        
        if (checkCoord.x >= 0 && checkCoord.x < int(uniforms.gridSize) &&
            checkCoord.y >= 0 && checkCoord.y < int(uniforms.gridSize) &&
            checkCoord.z >= 0 && checkCoord.z < int(uniforms.gridSize)) {
          
          uint voxelIndex = getVoxelIndex(uint3(checkCoord), uniforms.gridSize);
          uint currentOwner = voxels[voxelIndex].ownerAndFlags & 0xFF;
          
          // Calculate voxel center
          float3 voxelCenter = (float3(checkCoord) + 0.5) * uniforms.voxelSize - halfWorld;
          float3 toVoxel = voxelCenter - newPos;
          float distSq = dot(toVoxel, toVoxel);
          float voxelRadiusSq = uniforms.voxelSize * uniforms.voxelSize * 0.75; // 0.75 ≈ 3/4, collision threshold for cube
          
          // If ball is close to voxel and voxel is not owned by ball
          if (distSq < voxelRadiusSq && currentOwner != ballOwner) {
            // Convert voxel to ball's color (atomic operation to prevent race conditions)
            atomic_store_explicit((device atomic_uint*)&voxels[voxelIndex].ownerAndFlags, ballOwner, memory_order_relaxed);
            
            // Bounce ball based on which side was hit
            float3 normal = normalize(toVoxel);
            float maxComp = max(abs(normal.x), max(abs(normal.y), abs(normal.z)));
            
            if (abs(normal.x) == maxComp) {
              vel.x = -vel.x;
            } else if (abs(normal.y) == maxComp) {
              vel.y = -vel.y;
            } else {
              vel.z = -vel.z;
            }
            
            collided = true;
          }
        }
      }
    }
  }
  
  // Add slight randomness to prevent getting stuck
  if (collided) {
    vel += float3(
      (fract(sin(uniforms.globalTime * 12.9898 + float(id))) - 0.5) * 0.1,
      (fract(sin(uniforms.globalTime * 78.233 + float(id))) - 0.5) * 0.1,
      (fract(sin(uniforms.globalTime * 45.164 + float(id))) - 0.5) * 0.1
    );
  }
  
  // Clamp speed
  float speed = length(vel);
  float minSpeed = 1.0;
  float maxSpeed = 3.0;
  if (speed < 0.001) {
    // Prevent zero velocity - reinitialize with random direction
    vel = float3(
      (fract(sin(uniforms.globalTime * 91.8732 + float(id))) - 0.5) * 2.0,
      (fract(sin(uniforms.globalTime * 63.2941 + float(id))) - 0.5) * 2.0,
      (fract(sin(uniforms.globalTime * 37.5648 + float(id))) - 0.5) * 2.0
    );
    vel = normalize(vel) * minSpeed;
  } else if (speed < minSpeed) {
    vel = normalize(vel) * minSpeed;
  } else if (speed > maxSpeed) {
    vel = normalize(vel) * maxSpeed;
  }
  
  // Update ball state
  ball.position.xyz = newPos;
  ball.velocity.xyz = vel;
  balls[id] = ball;
}

// Vertex shader for rendering edges
vertex EdgeVertexOut edgeVertexShader(
  ushort amplificationID [[amplification_id]],
  const device EdgeVertex *vertices [[buffer(0)]],
  constant PongWarsSceneUniforms &uniforms [[buffer(1)]],
  constant float4x4 *viewProjectionMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  EdgeVertexOut out;
  uint layers = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1);
  
  EdgeVertex vtx = vertices[vertexID];
  float4x4 viewProjection = viewProjectionMatrices[viewIndex];
  
  out.position = viewProjection * float4(vtx.position, 1.0);
  out.color = vtx.color;
  
  return out;
}

// Fragment shader for rendering edges
fragment float4 edgeFragmentShader(
  EdgeVertexOut in [[stage_in]],
  constant PongWarsSceneUniforms &uniforms [[buffer(0)]])
{
  return float4(in.color, 0.8);
}
