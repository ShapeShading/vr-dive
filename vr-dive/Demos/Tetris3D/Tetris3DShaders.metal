#include <metal_stdlib>
using namespace metal;

// Use same structures as CubeField for compatibility

struct TetrisSceneUniforms {
  float time;
  uint layerCount;
  float2 padding;
};

struct TetrisSimulationUniforms {
  float deltaTime;
  float globalTime;
  uint objectCount;
  uint padding;
};

struct TetrisMeshVertex {
  float3 position;
  float3 normal;
};

struct TetrisBlockState {
  float4 positionAndType; // xyz = position, w = type (0-6 for tetromino colors)
  float4 motionAndPhase;  // w = alpha for animation
  float4 scaleAndPadding; // xyz = scale
  float4 homeAndJitter;   // unused
};

struct TetrisVertexOut {
  float4 position [[position]];
  float3 normal;
  float3 worldPos;
  float3 localPos; // For edge detection
  float blockType;
  float alpha;
};

// Tetromino colors
float3 tetrisBlockColor(float type) {
  int t = int(clamp(round(type), 0.0, 6.0));
  if (t == 0)
    return float3(0.0, 0.9, 0.9); // I - Cyan
  if (t == 1)
    return float3(0.9, 0.9, 0.0); // O - Yellow
  if (t == 2)
    return float3(0.7, 0.0, 0.9); // T - Purple
  if (t == 3)
    return float3(0.0, 0.9, 0.0); // S - Green
  if (t == 4)
    return float3(0.9, 0.0, 0.0); // Z - Red
  if (t == 5)
    return float3(0.0, 0.0, 0.9); // J - Blue
  return float3(0.9, 0.5, 0.0);   // L - Orange
}

vertex TetrisVertexOut tetrisBlockVertexShader(
    ushort amplificationID [[amplification_id]],
    const device TetrisMeshVertex *vertices [[buffer(0)]],
    const device TetrisBlockState *states [[buffer(1)]],
    constant TetrisSceneUniforms &uniforms [[buffer(2)]],
    constant float4x4 *viewProjectionMatrices [[buffer(3)]],
    uint vertexID [[vertex_id]], uint instanceID [[instance_id]]) {

  TetrisVertexOut out;
  uint layers = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1u);

  TetrisMeshVertex vtx = vertices[vertexID];
  TetrisBlockState state = states[instanceID];

  float3 position =
      state.positionAndType.xyz + vtx.position * state.scaleAndPadding.xyz;
  float4 worldPos = float4(position, 1.0);
  float4x4 viewProjection = viewProjectionMatrices[viewIndex];

  out.position = viewProjection * worldPos;
  out.normal = vtx.normal;
  out.worldPos = position;
  out.localPos = vtx.position; // -0.5 to 0.5 range
  out.blockType = state.positionAndType.w;
  out.alpha = state.motionAndPhase.w;

  return out;
}

fragment float4 tetrisBlockFragmentShader(TetrisVertexOut in [[stage_in]],
                                          constant TetrisSceneUniforms &uniforms
                                          [[buffer(0)]],
                                          bool isFrontFacing [[front_facing]]) {

  float3 normal = normalize(in.normal);
  normal = isFrontFacing ? normal : -normal;

  float3 lightDir = normalize(float3(0.4, 0.9, -0.2));
  float ndotl = max(dot(normal, lightDir), 0.0);

  // Add specular highlight
  float3 viewDir = normalize(-in.worldPos);
  float3 halfVec = normalize(lightDir + viewDir);
  float spec = pow(max(dot(normal, halfVec), 0.0), 32.0) * 0.4;

  // Rim lighting for depth
  float rim = pow(1.0 - abs(dot(normal, float3(0, 0, 1))), 2.0);

  float3 baseColor = tetrisBlockColor(in.blockType);

  // Edge darkening - darken edges of each face
  float3 absLocal = abs(in.localPos);
  float edgeDist = max(max(absLocal.x, absLocal.y), absLocal.z);
  float edgeFactor = smoothstep(0.35, 0.48, edgeDist); // Dark border at edges
  float3 edgeColor = baseColor * 0.3;                  // Dark edge color
  baseColor = mix(baseColor, edgeColor, edgeFactor);

  float3 color = baseColor * (0.35 + ndotl * 0.65) + spec + rim * 0.08;

  return float4(color, in.alpha);
}

// Ground plane shader
struct TetrisGroundVertexOut {
  float4 position [[position]];
  float2 uv;
  float3 worldPos;
};

struct TetrisGroundVertex {
  float3 position;
  float2 uv;
};

vertex TetrisGroundVertexOut tetrisGroundVertexShader(
    ushort amplificationID [[amplification_id]],
    const device TetrisGroundVertex *vertices [[buffer(0)]],
    constant TetrisSceneUniforms &uniforms [[buffer(1)]],
    constant float4x4 *viewProjectionMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]]) {

  TetrisGroundVertexOut out;
  uint layers = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1u);

  TetrisGroundVertex vtx = vertices[vertexID];
  float4x4 viewProjection = viewProjectionMatrices[viewIndex];

  out.position = viewProjection * float4(vtx.position, 1.0);
  out.uv = vtx.uv;
  out.worldPos = vtx.position;

  return out;
}

fragment float4 tetrisGroundFragmentShader(
    TetrisGroundVertexOut in [[stage_in]],
    constant TetrisSceneUniforms &uniforms [[buffer(0)]]) {

  // Grid pattern
  float2 grid = abs(fract(in.uv * 10.0) - 0.5);
  float gridLine = min(grid.x, grid.y);
  gridLine = smoothstep(0.02, 0.05, gridLine);

  float3 baseColor = float3(0.08, 0.1, 0.12);
  float3 lineColor = float3(0.2, 0.25, 0.3);

  float3 finalColor = mix(lineColor, baseColor, gridLine);

  return float4(finalColor, 0.7);
}
