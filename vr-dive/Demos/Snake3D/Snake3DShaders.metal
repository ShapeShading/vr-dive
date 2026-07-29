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
  float  colorIndex;
  float2 padding;
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

  // Diagonal gradient to break up visual merging between adjacent segments.
  float3 bodyGradDir = normalize(float3(0.6, 0.8, 0.4));
  float  bodyGradT   = saturate(dot(in.localPos, bodyGradDir) * 2.0 + 0.5);
  baseColor = mix(baseColor * 0.88, baseColor * 1.12, bodyGradT);

  float3 edgeColor = baseColor * 0.25;
  baseColor = mix(baseColor, edgeColor, edgeFactor);

  float3 color = baseColor * (0.5 + ndotl * 0.6) + spec;
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
  float  colorIndex;
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

  // Shrink block only when very close to camera to avoid obstructing view.
  // Block center distance determines scale; normal size beyond ~0.6 m.
  float3 centerTransformed = applyWorldTransform(inst.position,
                                                  uniforms.worldRotation,
                                                  uniforms.anchorTranslation);
  float  camDist    = length(centerTransformed);
  float  proximity  = saturate(1.0 - camDist / 0.6);  // 1=touching cam, 0=>=0.6m
  float  normalSize = 0.192;
  float  minSize    = 0.04;
  float  blockSize  = mix(normalSize, minSize, proximity * proximity);
  float3 worldPos = inst.position + vtx.position * blockSize;

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
  out.colorIndex = inst.colorIndex;
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

  // Per-food base color from palette (orange / blue / purple / hot-pink)
  float3 paletteColor;
  int ci = clamp(int(in.colorIndex), 0, 3);
  if      (ci == 0) paletteColor = float3(1.00, 0.55, 0.05);  // orange
  else if (ci == 1) paletteColor = float3(0.20, 0.50, 1.00);  // blue
  else if (ci == 2) paletteColor = float3(0.75, 0.15, 1.00);  // purple
  else              paletteColor = float3(1.00, 0.20, 0.55);  // hot pink

  // Fixed-direction diagonal gradient so adjacent same-color blocks stay distinguishable.
  // Project local position onto a fixed diagonal; this produces a unique shade per face corner.
  float3 gradDir = normalize(float3(0.6, 0.8, 0.4));
  float  gradT   = dot(in.localPos, gradDir) * 2.0 + 0.5; // roughly [-0.5, 1.5]
  gradT = saturate(gradT);
  float3 gradColor = mix(paletteColor * 0.82, paletteColor * 1.18, gradT);

  float3 normalColor = gradColor;
  float3 hitColor = float3(0.15, 0.95, 1.0);
  float3 baseColor = mix(normalColor, hitColor, saturate(in.hit));

  float3 absLocal   = abs(in.localPos);
  float  edgeDist   = max(max(absLocal.x, absLocal.y), absLocal.z);
  float  edgeFactor = smoothstep(0.35, 0.48, edgeDist);
  baseColor = mix(baseColor, baseColor * 0.3, edgeFactor);

  // Distance-based brightness: closer = brighter, farther = darker.
  float dist = length(in.worldPos);
  float distFade = saturate(1.0 - dist * 0.13);  // full at ~0m, half at ~5m
  float distScale = mix(0.55, 1.25, distFade);

  float3 color = baseColor * (0.55 + ndotl * 0.55) * distScale;
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
  float3 color = in.color.rgb * (0.65 + ndotl * 0.45);
  return float4(color, 1.0);
}
