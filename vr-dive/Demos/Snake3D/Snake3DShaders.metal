#include <metal_stdlib>
using namespace metal;

// MARK: - Shared Types

struct Snake3DSceneUniforms {
  float4x4 worldRotation;
  float3   anchorTranslation;
  float    time;
  uint     layerCount;
  float3   padding;
};

struct SnakeMeshVertex {
  float3 position;
  float3 normal;
};

struct SnakeSegmentInstance {
  float3 position;        // grid-space world position (before rotation)
  float4 colorAndIndex;   // rgb=color, a=normalizedIndex
  float  scale;
  float3 padding;
};

struct FoodInstance {
  float3 position;
  float  phase;
  float  hit;
  float3 padding;
};

struct SnakeGuideInstance {
  float3 position;
  float3 scale;
  float4 color;
};

// MARK: - Snake Body Shader

struct SnakeBodyVertexOut {
  float4 position [[position]];
  float3 normal;
  float3 worldPos;
  float3 localPos;
  float4 color;
};

// Apply world rotation transform: rotate around head, then offset to anchor
float3 applyWorldTransform(float3 pos,
                           float4x4 rotation,
                           float3 anchor) {
  float4 rotated = rotation * float4(pos, 1.0);
  return rotated.xyz + anchor;
}

vertex SnakeBodyVertexOut snake3DBodyVertexShader(
    ushort amplificationID      [[amplification_id]],
    const device SnakeMeshVertex *vertices        [[buffer(0)]],
    const device SnakeSegmentInstance *instances  [[buffer(1)]],
    constant Snake3DSceneUniforms &uniforms        [[buffer(2)]],
    constant float4x4 *viewProjectionMatrices      [[buffer(3)]],
    uint vertexID   [[vertex_id]],
    uint instanceID [[instance_id]])
{
  SnakeBodyVertexOut out;
  uint layers    = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1u);

  SnakeMeshVertex      vtx  = vertices[vertexID];
  SnakeSegmentInstance inst = instances[instanceID];

  // Local cube vertex scaled by block size, centred on instance position
  float3 localPos = vtx.position * inst.scale;
  float3 worldPos = inst.position + localPos;

  // Apply world rotation (rotates the whole scene so head stays at front)
  float3 transformed = applyWorldTransform(worldPos,
                                           uniforms.worldRotation,
                                           uniforms.anchorTranslation);

  float4x4 vp = viewProjectionMatrices[viewIndex];
  out.position = vp * float4(transformed, 1.0);

  // Rotate normal too (only rotation, not translation)
  float3 rotatedNormal = (uniforms.worldRotation * float4(vtx.normal, 0.0)).xyz;
  out.normal   = rotatedNormal;
  out.worldPos = transformed;
  out.localPos = vtx.position;
  out.color    = inst.colorAndIndex;
  return out;
}

fragment float4 snake3DBodyFragmentShader(SnakeBodyVertexOut in [[stage_in]],
                                          constant Snake3DSceneUniforms &uniforms [[buffer(0)]],
                                          bool isFrontFacing [[front_facing]])
{
  float3 normal = normalize(in.normal);
  normal = isFrontFacing ? normal : -normal;

  float3 lightDir = normalize(float3(0.3, 1.0, -0.5));
  float ndotl = max(dot(normal, lightDir), 0.0);

  float3 viewDir  = normalize(-in.worldPos);
  float3 halfVec  = normalize(lightDir + viewDir);
  float  spec     = pow(max(dot(normal, halfVec), 0.0), 32.0) * 0.3;

  // Edge darkening on cube faces
  float3 absLocal    = abs(in.localPos);
  float  edgeDist    = max(max(absLocal.x, absLocal.y), absLocal.z);
  float  edgeFactor  = smoothstep(0.35, 0.48, edgeDist);

  float3 baseColor = in.color.rgb;
  // Tail attenuation: dim toward tail
  float  t = in.color.a;   // 0=head, 1=tail
  baseColor = mix(baseColor, baseColor * 0.25, t * 0.7);

  float3 edgeColor = baseColor * 0.25;
  baseColor = mix(baseColor, edgeColor, edgeFactor);

  float3 color = baseColor * (0.3 + ndotl * 0.7) + spec;
  return float4(color, 1.0);
}

// MARK: - Food Shader

struct FoodVertexOut {
  float4 position [[position]];
  float3 normal;
  float3 worldPos;
  float3 localPos;
  float  time;
  float  phase;
  float  hit;
};

vertex FoodVertexOut snake3DFoodVertexShader(
    ushort amplificationID  [[amplification_id]],
    const device SnakeMeshVertex *vertices  [[buffer(0)]],
    const device FoodInstance   *instances  [[buffer(1)]],
    constant Snake3DSceneUniforms &uniforms [[buffer(2)]],
    constant float4x4 *viewProjectionMatrices [[buffer(3)]],
    uint vertexID   [[vertex_id]],
    uint instanceID [[instance_id]])
{
  FoodVertexOut out;
  uint layers    = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1u);

  SnakeMeshVertex vtx  = vertices[vertexID];
  FoodInstance    inst = instances[instanceID];

  // Pulsing scale
  float pulse = 1.0 + 0.12 * sin(uniforms.time * 3.0 + inst.phase);
  float blockSize = 0.192;
  float3 worldPos = inst.position + vtx.position * (blockSize * pulse);

  float3 transformed = applyWorldTransform(worldPos,
                                           uniforms.worldRotation,
                                           uniforms.anchorTranslation);

  float4x4 vp = viewProjectionMatrices[viewIndex];
  out.position = vp * float4(transformed, 1.0);
  out.normal   = (uniforms.worldRotation * float4(vtx.normal, 0.0)).xyz;
  out.worldPos = transformed;
  out.localPos = vtx.position;
  out.time     = uniforms.time;
  out.phase    = inst.phase;
  out.hit      = inst.hit;
  return out;
}

fragment float4 snake3DFoodFragmentShader(FoodVertexOut in [[stage_in]],
                                          constant Snake3DSceneUniforms &uniforms [[buffer(0)]],
                                          bool isFrontFacing [[front_facing]])
{
  float3 normal  = normalize(in.normal);
  normal = isFrontFacing ? normal : -normal;

  float3 lightDir = normalize(float3(0.3, 1.0, -0.5));
  float  ndotl    = max(dot(normal, lightDir), 0.0);

  // Red by default, switches to bright cyan when touched by guide dashes.
  float  glow     = 0.7 + 0.3 * sin(in.time * 3.0 + in.phase);
  float3 normalColor = float3(1.0, 0.15, 0.1) * glow;
  float3 hitColor = float3(0.15, 0.95, 1.0) * (0.85 + 0.15 * glow);
  float3 baseColor = mix(normalColor, hitColor, saturate(in.hit));

  float3 absLocal   = abs(in.localPos);
  float  edgeDist   = max(max(absLocal.x, absLocal.y), absLocal.z);
  float  edgeFactor = smoothstep(0.35, 0.48, edgeDist);
  baseColor = mix(baseColor, baseColor * 0.3, edgeFactor);

  float3 color = baseColor * (0.35 + ndotl * 0.65);
  return float4(color, 1.0);
}

// MARK: - Border (wireframe box)

struct BorderVertexOut {
  float4 position [[position]];
  float3 normal;
  float4 color;
};

vertex BorderVertexOut snake3DBorderVertexShader(
    ushort amplificationID [[amplification_id]],
    const device SnakeMeshVertex *vertices         [[buffer(0)]],
    const device SnakeGuideInstance *instances     [[buffer(1)]],
    constant Snake3DSceneUniforms  &uniforms       [[buffer(2)]],
    constant float4x4 *viewProjectionMatrices      [[buffer(3)]],
    uint vertexID [[vertex_id]],
    uint instanceID [[instance_id]])
{
  BorderVertexOut out;
  uint layers    = max(uniforms.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1u);

  SnakeMeshVertex vtx = vertices[vertexID];
  SnakeGuideInstance inst = instances[instanceID];

  float3 worldPos = inst.position + vtx.position * inst.scale;
  float3 transformed = applyWorldTransform(worldPos,
                                           uniforms.worldRotation,
                                           uniforms.anchorTranslation);

  float4x4 vp = viewProjectionMatrices[viewIndex];
  out.position = vp * float4(transformed, 1.0);
  out.normal = (uniforms.worldRotation * float4(vtx.normal, 0.0)).xyz;
  out.color = inst.color;
  return out;
}

fragment float4 snake3DBorderFragmentShader(BorderVertexOut in [[stage_in]],
                                            bool isFrontFacing [[front_facing]])
{
  float3 normal = normalize(in.normal);
  normal = isFrontFacing ? normal : -normal;
  float3 lightDir = normalize(float3(0.25, 1.0, -0.4));
  float ndotl = max(dot(normal, lightDir), 0.0);
  float3 color = in.color.rgb * (0.55 + ndotl * 0.45);
  return float4(color, 1.0);
}
