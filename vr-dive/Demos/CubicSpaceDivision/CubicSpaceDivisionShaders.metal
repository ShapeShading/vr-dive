// CubicSpaceDivisionShaders.metal
//
// Original implementation for a cube-contained cubic space division scene.
// Visual inspiration requested from ShaderToy 4ltyWl.
// Reference link: https://www.shadertoy.com/view/4ltyWl
// The source from the reference shader is not reused here. This is a
// clean-room Metal implementation adapted for the app's cube portal.

#include <metal_stdlib>
using namespace metal;

#define CSD_PI                3.14159265359f
#define CSD_MAX_STEPS         140
#define CSD_MIN_DIST          0.02f
#define CSD_MAX_DIST          180.0f
#define CSD_EPS               0.0009f
#define CSD_CELL_SPACING      20.0f
#define CSD_EDGE_SIZE         0.08f
#define CSD_EDGE_SMOOTH       0.05f
#define CSD_SCENE_SCALE       36.0f

struct CubicSpaceDivisionUniforms {
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

struct CubicSpaceDivisionVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

struct CSDTraceResult {
  float depth;
  bool hit;
};

vertex CubicSpaceDivisionVertexOut cubicSpaceDivisionVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant CubicSpaceDivisionUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  CubicSpaceDivisionVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float3x3 csdRotateX(float angle) {
  float s = sin(angle);
  float c = cos(angle);
  return float3x3(
    float3(1.0f, 0.0f, 0.0f),
    float3(0.0f, c, -s),
    float3(0.0f, s, c));
}

static float3x3 csdRotateZ(float angle) {
  float s = sin(angle);
  float c = cos(angle);
  return float3x3(
    float3(c, -s, 0.0f),
    float3(s, c, 0.0f),
    float3(0.0f, 0.0f, 1.0f));
}

static float3 csdMod(float3 x, float3 y) {
  return x - y * floor(x / y);
}

static float csdBoxSDF(float3 p, float3 size) {
  float3 d = abs(p) - size * 0.5f;
  float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0f);
  float outsideDistance = length(max(d, 0.0f));
  return insideDistance + outsideDistance;
}

static float csdColumnSDF(float3 p, float thickness) {
  float2 d = abs(p.xz) - float2(thickness * 0.5f);
  float insideDistance = min(max(d.x, d.y), 0.0f);
  float outsideDistance = length(max(d, 0.0f));
  return insideDistance + outsideDistance;
}

static float csdColumnsSDF(float3 p, float spacing) {
  float3 cell = float3(spacing, 100000.0f, spacing);
  float3 q = csdMod(p, cell) - 0.5f * cell;
  return csdColumnSDF(q, 1.0f);
}

static float csdSceneSDF(float3 p) {
  float columns1 = csdColumnsSDF(p, CSD_CELL_SPACING);
  float columns2 = csdColumnsSDF(csdRotateZ(CSD_PI * 0.5f) * p, CSD_CELL_SPACING);
  float columns3 = csdColumnsSDF(csdRotateX(CSD_PI * 0.5f) * p, CSD_CELL_SPACING);
  float columns = min(columns1, min(columns2, columns3));

  float3 repeated = csdMod(p, float3(CSD_CELL_SPACING)) - 0.5f * CSD_CELL_SPACING;
  float box = csdBoxSDF(repeated, float3(3.0f));
  return min(columns, box);
}

static float3 csdEstimateNormal(float3 p) {
  float2 e = float2(CSD_EPS, 0.0f);
  return normalize(float3(
    csdSceneSDF(p + e.xyy) - csdSceneSDF(p - e.xyy),
    csdSceneSDF(p + e.yxy) - csdSceneSDF(p - e.yxy),
    csdSceneSDF(p + e.yyx) - csdSceneSDF(p - e.yyx)));
}

static float3 csdDiffuse(float3 kd, float3 p, float3 eye, float3 lightDir, float3 intensity) {
  float3 n = csdEstimateNormal(p);
  float3 l = normalize(lightDir);
  float dotLN = dot(l, n);
  if (dotLN < 0.0f) return float3(0.0f);
  return intensity * (kd * dotLN);
}

static float3 csdFog(float3 rgb, float distance) {
  float fogAmount = 1.0f - exp(-distance * pow(0.03f, 1.4f));
  return mix(rgb, float3(0.80f, 0.82f, 0.86f), fogAmount);
}

static bool csdBoxHit(
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

static CSDTraceResult csdRayMarch(float3 eye, float3 direction) {
  float depth = CSD_MIN_DIST;

  for (int i = 0; i < CSD_MAX_STEPS; ++i) {
    float dist = csdSceneSDF(eye + depth * direction);

    if (dist < CSD_EPS) {
      return CSDTraceResult { depth, true };
    }

    depth += dist;
    if (depth >= CSD_MAX_DIST) {
      return CSDTraceResult { CSD_MAX_DIST, false };
    }
  }

  return CSDTraceResult { CSD_MAX_DIST, false };
}

static float3 csdComputeColor(float3 eye, float3 direction) {
  CSDTraceResult trace = csdRayMarch(eye, direction);
  if (!trace.hit) {
    return float3(0.95f);
  }

  float3 p = eye + trace.depth * direction;
  float3 ambient = float3(0.44f, 0.46f, 0.50f);
  float3 kd = float3(0.12f, 0.13f, 0.15f);
  float3 color = ambient;
  color += csdDiffuse(kd, p, eye, float3(0.3f, 0.5f, -1.0f), float3(1.6f));
  color = csdFog(color, trace.depth);
  return pow(max(color, 0.0f), float3(1.5f));
}

fragment float4 cubicSpaceDivisionFragment(
  CubicSpaceDivisionVertexOut in [[stage_in]],
  constant CubicSpaceDivisionUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center = uniforms.objectCenter.xyz;
  float3 rdWorld = normalize(in.worldPos - camWorld);
  float3 roLocal = (camWorld - center) / uniforms.cubeScale;
  float3 rdLocal = rdWorld;

  float tEntry;
  float tExit;
  if (!csdBoxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float sceneTime = uniforms.time * uniforms.travelSpeed;
  float3 sceneEye = float3(0.7f, 0.83f, 1.8f) * 25.0f + float3(7.0f, 8.0f, 3.0f - sceneTime * 4.0f);
  float3 sceneDir = normalize(float3(rdLocal.x, rdLocal.y, rdLocal.z));

  float3 color = csdComputeColor(sceneEye, sceneDir);
  return float4(color, 1.0f);
}