#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Quaternion Julia set ray marcher based on Inigo Quilez's article:
// https://iquilezles.org/articles/juliasets3d/
// Sphere tracing + distance estimation + gradient normals
// ============================================================================

struct SceneUniforms {
  float time;
  uint layerCount;
  float2 padding;
};

struct Julia3DUniforms {
  float globalTime;
  uint maxRaySteps;
  uint iterationCount;
  uint padding;
  float4 juliaC;
  float worldScale;
  float escapeRadius;
  float surfaceEpsilon;
  float maxDistance;
  float ambientStrength;
  float glowStrength;
  float aoStrength;
  float animationSpeed; // passed for reference/debug
};

struct Julia3DViewUniform {
  float4x4 viewToWorld;
  float4x4 projectionInverse;
};

struct Julia3DVertexOut {
  float4 position [[position]];
  float2 uv;
  uint viewIndex [[flat]];
};

// ============================================================================
// Quaternion helpers
// ============================================================================

inline float4 qsqr(float4 q) {
  return float4(
    q.x * q.x - q.y * q.y - q.z * q.z - q.w * q.w,
    2.0 * q.x * q.y,
    2.0 * q.x * q.z,
    2.0 * q.x * q.w
  );
}

inline float qlength2(float4 q) {
  return dot(q, q);
}

// Distance estimation
float juliaDE(float3 pos, float4 c, uint maxIter, float escapeRadius) {
  float4 z = float4(pos, 0.0);
  float dz2 = 1.0;
  float escapeR2 = escapeRadius * escapeRadius;

  for (uint i = 0; i < maxIter; ++i) {
    dz2 *= 4.0 * qlength2(z);
    z = qsqr(z) + c;
    if (qlength2(z) > escapeR2) break;
  }

  float r = max(0.0001f, length(z));
  float dr = max(1e-6f, sqrt(dz2));
  return 0.5 * r * log(r) / dr;
}

float juliaIterations(float3 pos, float4 c, uint maxIter, float escapeRadius) {
  float4 z = float4(pos, 0.0);
  float escapeR2 = escapeRadius * escapeRadius;

  for (uint i = 0; i < maxIter; ++i) {
    z = qsqr(z) + c;
    float mag2 = qlength2(z);
    if (mag2 > escapeR2) {
      float logMag = log2(max(1.0f, log2(sqrt(mag2))));
      return (float(i) - logMag) / float(maxIter);
    }
  }
  return 1.0;
}

float3 calcNormal(float3 pos, constant Julia3DUniforms &uniforms) {
  const float eps = 0.002;
  float3 e = float3(eps, -eps, 0.0);
  float dx = juliaDE(pos + float3(e.x, 0.0, 0.0), uniforms.juliaC, uniforms.iterationCount, uniforms.escapeRadius)
          - juliaDE(pos - float3(e.x, 0.0, 0.0), uniforms.juliaC, uniforms.iterationCount, uniforms.escapeRadius);
  float dy = juliaDE(pos + float3(0.0, e.x, 0.0), uniforms.juliaC, uniforms.iterationCount, uniforms.escapeRadius)
          - juliaDE(pos - float3(0.0, e.x, 0.0), uniforms.juliaC, uniforms.iterationCount, uniforms.escapeRadius);
  float dz = juliaDE(pos + float3(0.0, 0.0, e.x), uniforms.juliaC, uniforms.iterationCount, uniforms.escapeRadius)
          - juliaDE(pos - float3(0.0, 0.0, e.x), uniforms.juliaC, uniforms.iterationCount, uniforms.escapeRadius);
  return normalize(float3(dx, dy, dz));
}

float ambientOcclusion(float3 pos, float3 normal, constant Julia3DUniforms &uniforms) {
  float ao = 0.0;
  float step = 0.02;
  float weight = 0.5;
  for (int i = 1; i <= 4; ++i) {
    float dist = juliaDE(pos + normal * step * float(i), uniforms.juliaC, uniforms.iterationCount, uniforms.escapeRadius);
    ao += (step * float(i) - dist) * weight;
    weight *= 0.5;
  }
  return clamp(1.0 - ao * uniforms.aoStrength, 0.0, 1.0);
}

float hash31(float3 p) {
  p = fract(p * float3(443.897, 441.423, 437.195));
  p += dot(p, p.yzx + 19.19);
  return fract((p.x + p.y) * p.z);
}

// ============================================================================
// Vertex shader: full-screen triangle with multi-view support
// ============================================================================

vertex Julia3DVertexOut julia3DRaymarchVertex(
  ushort amplificationID [[amplification_id]],
  uint vertexID [[vertex_id]],
  constant SceneUniforms &scene [[buffer(0)]],
  constant Julia3DViewUniform *viewUniforms [[buffer(1)]]
) {
  Julia3DVertexOut out;

  const float2 positions[3] = {
    float2(-1.0, -1.0),
    float2(3.0, -1.0),
    float2(-1.0, 3.0)
  };

  out.position = float4(positions[vertexID], 0.0, 1.0);
  out.uv = out.position.xy * 0.5 + 0.5;

  uint layers = max(scene.layerCount, 1u);
  out.viewIndex = min((uint)amplificationID, layers - 1);
  (void)viewUniforms; // silence unused warning when layerCount == 0
  return out;
}

// ============================================================================
// Fragment shader: sphere trace + shading
// ============================================================================

fragment float4 julia3DRaymarchFragment(
  Julia3DVertexOut in [[stage_in]],
  constant Julia3DViewUniform *viewUniforms [[buffer(0)]],
  constant Julia3DUniforms &uniforms [[buffer(1)]]
) {
  const float3 background = float3(0.005, 0.01, 0.025);
  uint viewIndex = min(in.viewIndex, (uint)0xFFFF);
  Julia3DViewUniform viewUniform = viewUniforms[viewIndex];

  float2 ndc = in.uv * 2.0 - 1.0;
  float4 clip = float4(ndc, 1.0, 1.0);
  float4 viewPos = viewUniform.projectionInverse * clip;
  viewPos /= viewPos.w;

  float4 cameraWorld = viewUniform.viewToWorld * float4(0.0, 0.0, 0.0, 1.0);
  float3 rayOrigin = cameraWorld.xyz;
  float4 worldTarget = viewUniform.viewToWorld * float4(viewPos.xyz, 1.0);
  float3 rayDir = normalize(worldTarget.xyz - rayOrigin);

  float travel = 0.0;
  float3 pos = rayOrigin;
  bool hit = false;

  for (uint step = 0; step < uniforms.maxRaySteps; ++step) {
    float3 fractalPos = pos / uniforms.worldScale;
    float dist = juliaDE(fractalPos, uniforms.juliaC, uniforms.iterationCount, uniforms.escapeRadius);
    dist *= uniforms.worldScale;

    if (dist < uniforms.surfaceEpsilon) {
      hit = true;
      break;
    }

    travel += dist;
    pos += rayDir * dist;

    if (travel > uniforms.maxDistance) {
      break;
    }
  }

  if (!hit) {
    float stars = pow(hash31(float3(in.uv, uniforms.globalTime * 0.05)), 32.0);
    float3 color = background + stars * 0.75;
    return float4(color, 1.0);
  }

  float3 fractalHitPos = pos / uniforms.worldScale;
  float3 normal = calcNormal(fractalHitPos, uniforms);
  float iterVal = juliaIterations(fractalHitPos, uniforms.juliaC, uniforms.iterationCount, uniforms.escapeRadius);

  float3 lightDir = normalize(float3(-0.4, 0.8, -0.5));
  float diff = clamp(dot(normal, lightDir), 0.0, 1.0);
  float3 viewDir = normalize(rayOrigin - pos);
  float3 halfVec = normalize(lightDir + viewDir);
  float spec = pow(max(dot(normal, halfVec), 0.0), 48.0);

  float ao = ambientOcclusion(fractalHitPos, normal, uniforms);

  // Palette inspired by IQ article
  float3 baseColor = 0.55 + 0.45 * cos(6.28318 * (iterVal + float3(0.0, 0.33, 0.67)));
  baseColor = pow(baseColor, float3(0.85));

  float3 shaded = baseColor * (uniforms.ambientStrength * ao + diff * ao);
  shaded += spec * float3(0.9, 0.95, 1.0) * 0.6;

  float glow = uniforms.glowStrength * exp(-travel * 0.04);
  shaded += glow * baseColor;

  float fog = clamp(travel / uniforms.maxDistance, 0.0, 1.0);
  float3 color = mix(shaded, background, fog);

  return float4(color, 1.0);
}
