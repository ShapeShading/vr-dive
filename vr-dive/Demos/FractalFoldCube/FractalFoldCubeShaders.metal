// FractalFoldCubeShaders.metal
//
// Original cube-portal volumetric fold-fractal scene.
// Visual inspiration requested from ShaderToy NXfGzH:
// https://www.shadertoy.com/view/NXfGzH
// This implementation is original and does not reuse source code from the
// reference shader.

#include <metal_stdlib>
using namespace metal;

#define FFC_STEPS        56
#define FFC_MAX_DIST     38.0f
#define FFC_SCENE_SCALE  9.5f
#define FFC_REPEAT       26.0f

struct FractalFoldCubeUniforms {
  float time;
  uint viewCount;
  float cubeScale;
  float travelSpeed;
  float4 objectCenter;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct FractalFoldCubeVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex FractalFoldCubeVertexOut fractalFoldCubeVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant FractalFoldCubeUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  FractalFoldCubeVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float2 ffc_rot(float2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float3 ffc_foldField(float3 p, float time, thread float &scaleAccum) {
  scaleAccum = 1.0f;
  float phase = time * 0.4f;
  float3 axis = normalize(tan(phase + p * 0.03f + float3(4.0f, 2.0f, 0.0f)));
  p = reflect(-p, axis) + float3(1.0f, -3.0f, 1.0f);

  for (int j = 0; j < 10; ++j) {
    p *= 1.18f;
    scaleAccum *= 1.18f;
    p.y += 2.4f;
    if (p.y > p.z) { p = p.xzy; }
    p.xz = float2(p.z, -p.x);

    p *= 1.14f;
    scaleAccum *= 1.14f;
    p.y += 2.1f;
    if (p.x < p.y) { p = p.yxz; }
  }

  p = fmod(p - 0.5f * FFC_REPEAT, FFC_REPEAT) - 0.5f * FFC_REPEAT;
  return p;
}

static float4 ffc_sampleVolume(float3 p, float time) {
  float scaleAccum;
  float3 q = ffc_foldField(p, time, scaleAccum);
  float line = length(q.xz) / max(scaleAccum, 1e-3f);
  float tube = exp(-5.0f * line);
  float shell = exp(-10.0f * abs(line - 0.22f));
  float vertical = exp(-0.05f * abs(q.y));
  float density = max(tube, 0.75f * shell) * (0.72f + 0.28f * vertical);

  float huePhase = 0.15f * q.y + 0.08f * q.x + time * 0.9f;
  float3 tintA = 0.5f + 0.5f * sin(float3(0.0f, 2.0f, 4.0f) + huePhase);
  float3 tintB = float3(0.08f, 0.32f, 0.95f) + 0.35f * sin(float3(1.0f, 2.8f, 4.7f) + time + q.z * 0.05f);
  float3 color = mix(tintA, tintB, 0.45f) * (0.6f + 0.4f * vertical) * density;
  return float4(color, density);
}

static bool ffc_boxHit(
  float3 ro, float3 rd, float3 bmin, float3 bmax,
  thread float &tNear, thread float &tFar)
{
  float3 t0 = (bmin - ro) / rd;
  float3 t1 = (bmax - ro) / rd;
  float3 lo = min(t0, t1);
  float3 hi = max(t0, t1);
  tNear = max(max(lo.x, lo.y), lo.z);
  tFar = min(min(hi.x, hi.y), hi.z);
  return tFar >= max(tNear, 0.0f);
}

fragment float4 fractalFoldCubeFragment(
  FractalFoldCubeVertexOut in [[stage_in]],
  constant FractalFoldCubeUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center = uniforms.objectCenter.xyz;
  float3 roLocal = (camWorld - center) / uniforms.cubeScale;
  float3 rdLocal = normalize(in.worldPos - camWorld);

  float tEntry;
  float tExit;
  if (!ffc_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float3 entryLocal = roLocal + rdLocal * max(tEntry, 0.0f);
  float3 ro = entryLocal * FFC_SCENE_SCALE;
  float3 rd = rdLocal;
  float time = uniforms.time * uniforms.travelSpeed;

  float stepLen = FFC_MAX_DIST / float(FFC_STEPS);
  float3 accum = float3(0.0f);
  float opacity = 0.0f;
  float distTravelled = 0.0f;

  for (int i = 0; i < FFC_STEPS; ++i) {
    float3 p = ro + rd * distTravelled;
    float4 sample = ffc_sampleVolume(p, time - float(i) * 0.015f);
    float density = clamp(sample.w * 0.36f, 0.0f, 0.92f);
    float contrib = (1.0f - opacity) * density;
    accum += sample.xyz * contrib * (2.2f + 0.04f * distTravelled);
    opacity += contrib;
    if (opacity > 0.98f) { break; }
    distTravelled += stepLen;
  }

  float3 sky = float3(0.02f, 0.03f, 0.06f) + 0.08f * float3(0.4f, 0.6f, 1.0f) * max(rd.y * 0.5f + 0.5f, 0.0f);
  float fade = exp(-0.014f * distTravelled);
  float3 color = accum * fade + sky * (1.0f - min(opacity, 1.0f));
  color = tanh(color / 1.1f);
  color = pow(max(color, 0.0f), float3(0.88f));
  return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}