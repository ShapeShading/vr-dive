#include <metal_stdlib>
using namespace metal;

struct Uniforms {
  float time;
  float3 padding;
};

struct VertexOut {
  float4 position [[position]];
  float2 uv;
  uint layer [[render_target_array_index]];
};

// Simple test shaders

vertex VertexOut vertexShader(uint vertexID [[vertex_id]],
                              uint instanceID [[instance_id]],
                              constant Uniforms &uniforms [[buffer(0)]]) {
  VertexOut out;
  // Full screen triangle
  float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
  out.position = float4(positions[vertexID], 0.0, 1.0);
  out.uv = positions[vertexID] * 0.5 + 0.5;
  out.layer = instanceID;
  return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               constant Uniforms &uniforms [[buffer(0)]]) {
  // Bright test pattern with animated colors
  float t = uniforms.time * 0.5;

  // Create a colorful gradient
  float3 col =
      float3(0.5 + 0.5 * sin(t + in.uv.x * 3.14159),
             0.5 + 0.5 * cos(t + in.uv.y * 3.14159), 0.5 + 0.5 * sin(t * 2.0));

  // Make colors brighter and more saturated
  col = col * 0.8 + 0.2;

  // Different tint per eye to verify stereo
  if (in.layer == 0) {
    col.r = mix(col.r, 1.0, 0.3); // Left eye more red
  } else {
    col.b = mix(col.b, 1.0, 0.3); // Right eye more blue
  }

  return float4(col, 1.0);
}
