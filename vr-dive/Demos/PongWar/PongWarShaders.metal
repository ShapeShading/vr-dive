#include <metal_stdlib>
using namespace metal;

struct SceneUniforms {
  float time;
  uint layerCount;
  float2 padding;
};

struct MeshVertex {
  float3 position;
  float3 normal;
  float3 axisMask; // 棱的方向
  float3 faceMask; // faceMask.x = 该棱所属的两个面的掩码
};

struct PongWarInstanceState {
  float4 positionAndScale;
  float4 color;
  float4 edgeData;
};

struct PongWarUniforms {
  float edgeHighlight;
  float baseGlow;
  float ballGlow;
  float edgeHalfThickness;
};

struct PongWarVertexOut {
  float4 position [[position]];
  float3 normal;
  float3 worldPos;
  float3 color;
  float type;
  float interior;
  float thickness;
  float2 crossCoord;
};

vertex PongWarVertexOut pongWarVertexShader(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    const device PongWarInstanceState *instances [[buffer(1)]],
    constant SceneUniforms &scene [[buffer(2)]],
    constant float4x4 *viewProjectionMatrices [[buffer(3)]],
    constant PongWarUniforms &uniforms [[buffer(4)]],
    uint vertexID [[vertex_id]], uint instanceID [[instance_id]]) {
  PongWarVertexOut out;
  PongWarInstanceState state = instances[instanceID];
  MeshVertex meshVertex = vertices[vertexID];
  float3 meshPos = meshVertex.position;
  float3 meshNormal = meshVertex.normal;
  float3 axisMask = meshVertex.axisMask;
  float edgeFaceMask = meshVertex.faceMask.x; // 该棱所属的两个面
  float time = scene.time;

  float isSphere = state.edgeData.w;
  float3 scaledPos = meshPos;
  float3 scaledNormal = meshNormal;
  float thickness = 0.0;
  float2 crossCoord = float2(0.0);

  if (isSphere < 0.5) {
    // 立方体：根据边界掩码判断这条棱是否需要显示
    float nearness = state.edgeData.x;     // 小球接近度 (0=远, 1=近)
    float boundaryMask = state.edgeData.y; // 6个面的颜色边界状态
    float outerMask = state.edgeData.z; // 6个面的外边界状态（动态）
    int bmask = int(boundaryMask);
    int omask = int(outerMask);
    int emask = int(edgeFaceMask);

    // 如果棱所属的两个面中至少有一个是颜色边界面，则显示边界棱
    bool isBoundaryEdge = (bmask & emask) != 0;
    // 如果棱所属的两个面中至少有一个是外边界面，则显示外边界棱
    bool isOuterEdge = (omask & emask) != 0;

    if (!isBoundaryEdge && !isOuterEdge) {
      // 既不是颜色边界也不是外边界，不显示
      out.position = float4(0, 0, -1, 0);
      out.normal = float3(0);
      out.worldPos = float3(0);
      out.color = float3(0);
      out.type = 0;
      out.interior = 1;
      out.thickness = 0;
      out.crossCoord = float2(0);
      return out;
    }

    // 粗细控制：
    // - 外边界棱：1.0（延伸到立方体边缘，无margin）
    // - 颜色边界棱：0.95（略微分开，避免重合）
    if (isOuterEdge && !isBoundaryEdge) {
      thickness = 1.0; // 外边界棱，满尺寸到边缘
    } else {
      // 颜色边界棱：略小于满尺寸，保持微小间隙
      thickness = 0.95;
    }

    // 不需要margin，棱直接延伸到立方体边缘
    // 这样相邻不同颜色的棱会紧靠在一起

    // 缩放棱的粗细
    float3 axisScale = axisMask + (float3(1.0) - axisMask) * thickness;
    scaledPos = meshPos * axisScale;
    scaledNormal = normalize(meshNormal / fmax(axisScale, float3(0.0001)));
    scaledPos *= state.positionAndScale.w;

    // 计算截面坐标用于抗锯齿 - 用原始 meshPos，不是缩放后的
    float halfThickness = uniforms.edgeHalfThickness;
    if (axisMask.x > 0.5) {
      // X轴方向的棱，截面在 YZ 平面
      crossCoord = abs(float2(meshPos.y, meshPos.z)) / halfThickness;
    } else if (axisMask.y > 0.5) {
      // Y轴方向的棱，截面在 XZ 平面
      crossCoord = abs(float2(meshPos.x, meshPos.z)) / halfThickness;
    } else {
      // Z轴方向的棱，截面在 XY 平面
      crossCoord = abs(float2(meshPos.x, meshPos.y)) / halfThickness;
    }
  } else {
    // 球体
    float wobble = sin(time * 1.2 + state.edgeData.x * 0.37);
    float radius = state.positionAndScale.w * (1.0 + wobble * 0.05);
    scaledPos *= radius;
    thickness = 1.0;
  }

  float3 worldPos = state.positionAndScale.xyz;
  float3 finalPos = worldPos + scaledPos;
  float4 worldPosition = float4(finalPos, 1.0);

  uint layers = max(scene.layerCount, 1u);
  uint viewIndex = min((uint)amplificationID, layers - 1);
  float4x4 viewProj = viewProjectionMatrices[viewIndex];

  out.position = viewProj * worldPosition;
  out.normal = normalize(scaledNormal);
  out.worldPos = finalPos;
  out.color = state.color.rgb;
  out.type = isSphere;
  // interior: 0 = 颜色边界棱, 0.5 = 外边界棱
  out.interior = isSphere < 0.5 ? (thickness < 0.4 ? 0.5 : 0.0) : 0.0;
  out.thickness = thickness;
  out.crossCoord = crossCoord;
  return out;
}

fragment float4 pongWarFragmentShader(PongWarVertexOut in [[stage_in]],
                                      constant SceneUniforms &scene
                                      [[buffer(0)]],
                                      constant PongWarUniforms &uniforms
                                      [[buffer(1)]]) {
  float3 normal = normalize(in.normal);
  float3 lightDir = normalize(float3(-0.35, 0.85, -0.12));
  float3 viewDir = normalize(-in.worldPos);
  float ndotl = max(dot(normal, lightDir), 0.0);
  float3 halfVector = normalize(lightDir + viewDir);
  float specPower = 32.0 + (12.0 - 32.0) * in.type;
  float spec = pow(max(dot(normal, halfVector), 0.0), specPower);
  float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);

  // 外边界棱用较暗的颜色，颜色边界棱用正常颜色
  float outerDim = in.interior > 0.3 ? 0.5 : 1.0; // 外边界棱变暗
  float3 baseColor = in.color * outerDim;

  // 提高基础亮度，让颜色更鲜明
  float ambient = 0.5;         // 提高环境光
  float diffuse = ndotl * 0.5; // 漫反射
  float3 lit = baseColor * (ambient + diffuse);

  // 高光和菲涅尔效果
  float3 specColor = float3(0.4) * spec;
  float3 fresnelColor = baseColor * fresnel * 0.2;

  float3 color = lit + specColor + fresnelColor;

  // 边缘增亮抗锯齿：在边缘处略微增亮，减少视觉上的锯齿感
  if (in.type < 0.5) {
    float edgeFactor = 1.0 - abs(dot(normal, viewDir));
    float boost = edgeFactor * edgeFactor * 0.15; // 边缘轻微增亮
    color = color * (1.0 + boost);
  }

  // 对于小球，增加自发光效果
  if (in.type > 0.5) {
    color = baseColor * 0.8 + color * 0.4; // 小球更亮
  }

  return float4(color, 1.0);
}
