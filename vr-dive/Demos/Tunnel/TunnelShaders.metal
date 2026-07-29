// TunnelShaders.metal
//
// Original implementation for a cube-contained tunnel field.
// Visual inspiration requested from ShaderToy 4dfGDr.
// Reference link: https://www.shadertoy.com/view/4dfGDr
// The source from the reference shader is not reused here. This is a
// clean-room Metal implementation that preserves the high-level idea of an
// animated segmented tunnel rendered after the ray enters the cube container.

#include <metal_stdlib>
using namespace metal;

#define TU_PI              3.14159265359f
#define TU_TWO_PI          6.28318530718f
#define TU_MAX_STEPS       192
#define TU_MAX_DIST        96.0f
#define TU_EPS             0.0010f
#define TU_SCENE_SCALE     14.0f

struct TunnelUniforms {
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

struct TunnelVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex TunnelVertexOut tunnelVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant TunnelUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  TunnelVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float tuPeriodic(float value, float period, float dutyCycle) {
  float normalized = value / period;
  float centered = abs(normalized - floor(normalized) - 0.5f) - dutyCycle * 0.5f;
  return centered * period;
}

static float tuCount(float value, float period) {
  return floor(value / period);
}

static float3 tuPalette(float t) {
  constexpr float3 stops[13] = {
    float3(0.00f, 0.00f, 0.00f),
    float3(0.01f, 0.06f, 0.20f),
    float3(0.03f, 0.08f, 0.28f),
    float3(0.10f, 0.03f, 0.36f),
    float3(0.28f, 0.02f, 0.47f),
    float3(0.52f, 0.03f, 0.54f),
    float3(0.72f, 0.07f, 0.44f),
    float3(0.88f, 0.12f, 0.20f),
    float3(0.98f, 0.30f, 0.12f),
    float3(0.99f, 0.63f, 0.28f),
    float3(0.99f, 0.90f, 0.55f),
    float3(0.94f, 0.98f, 0.80f),
    float3(0.96f, 1.00f, 0.92f)
  };

  float x = clamp(t, 0.0f, 0.999f) * 12.0f;
  int idx = min((int)floor(x), 11);
  float f = fract(x);
  return mix(stops[idx], stops[idx + 1], f);
}

static float tuDistanceField(float3 p, float time) {
  float radial = length(p.xy);
  float angle = atan2(p.y, p.x);

  float radialBand = tuCount(radial, 3.0f);
  float depthBand = tuCount(p.z, 1.0f);
  float temporalGate = 0.80f + 0.12f * cos(time / 3.0f);
  float swirl = time * 0.3f * sin(radialBand + 1.0f) * sin(depthBand * 13.73f);
  float sweptAngle = angle + swirl;
  float angularPeriod = max(radial, 0.001f) * (TU_TWO_PI / 6.0f);

  float radialShell = tuPeriodic(radial, 3.0f, 0.26f);
  float axialSlice = tuPeriodic(p.z, 1.0f, temporalGate);
  float spokeSlice = tuPeriodic(sweptAngle * radial, angularPeriod, temporalGate);
  return min(max(max(radialShell, axialSlice), spokeSlice), 0.25f);
}

static float3 tuBackground(float3 rd) {
  float horizon = exp(-9.0f * abs(rd.y));
  float3 col = float3(0.001f, 0.0008f, 0.004f);
  col += horizon * float3(0.01f, 0.004f, 0.02f);
  return col;
}

static bool tuBoxHit(
  float3 ro, float3 rd, float3 bmin, float3 bmax,
  thread float &tNear, thread float &tFar)
{
  float3 invDir = 1.0f / max(abs(rd), 1e-4f) * sign(rd);
  float3 t0 = (bmin - ro) * invDir;
  float3 t1 = (bmax - ro) * invDir;
  float3 lo = min(t0, t1);
  float3 hi = max(t0, t1);
  tNear = max(max(lo.x, lo.y), lo.z);
  tFar = min(min(hi.x, hi.y), hi.z);
  return tFar >= max(tNear, 0.0f);
}

static bool tuTrace(
  float3 ro, float3 rd, float time, thread float &hitT, thread float &stepsTaken)
{
  float travel = 0.0f;
  float marchedSteps = float(TU_MAX_STEPS);

  for (int i = 0; i < TU_MAX_STEPS; ++i) {
    float3 pos = ro + rd * travel;
    float distanceToSurface = tuDistanceField(pos, time);
    marchedSteps = float(i);
    if (abs(distanceToSurface) < TU_EPS) {
      hitT = travel;
      stepsTaken = marchedSteps;
      return true;
    }

    travel += distanceToSurface;
    if (travel > TU_MAX_DIST) break;
  }

  hitT = travel;
  stepsTaken = marchedSteps;
  return false;
}

fragment float4 tunnelFragment(
  TunnelVertexOut in [[stage_in]],
  constant TunnelUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center = uniforms.objectCenter.xyz;
  float cubeScale = uniforms.cubeScale;
  float3 rdWorld = normalize(in.worldPos - camWorld);

  float3 roLocal = (camWorld - center) / cubeScale;
  float3 rdLocal = rdWorld;

  float tEntry;
  float tExit;
  if (!tuBoxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float3 entryLocal = roLocal + rdLocal * max(tEntry, 0.0f);
  float sceneTime = uniforms.time * uniforms.travelSpeed;
  float3 sceneRo = entryLocal * TU_SCENE_SCALE;
  sceneRo.z += sceneTime;
  float3 sceneRd = rdLocal;

  float sceneNear;
  float sceneFar;
  if (!tuBoxHit(sceneRo, sceneRd, float3(-48.0f), float3(48.0f), sceneNear, sceneFar)) {
    return float4(tuBackground(sceneRd), 1.0f);
  }

  sceneRo += sceneRd * max(sceneNear, 0.0f);

  float hitT;
  float stepsTaken;
  bool hit = tuTrace(sceneRo, sceneRd, sceneTime, hitT, stepsTaken);
  if (!hit) {
    return float4(tuBackground(sceneRd), 1.0f);
  }

  float progress = clamp(stepsTaken / float(TU_MAX_STEPS), 0.0f, 1.0f);
  float3 color = tuPalette(progress);
  color = pow(max(color, 0.0f), float3(1.08f));
  return float4(color, 1.0f);
}