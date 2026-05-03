// VoxelEdgesShaders.metal
//
// Original implementation for a voxel edges cube portal.
// Visual inspiration was requested from ShaderToy 4dfGzs.
// Reference link: https://www.shadertoy.com/view/4dfGzs
// Shading article reference: https://iquilezles.org/articles/voxellines
// The original source license forbids reuse in products/projects, so no source
// code from the original work is copied here. This shader is a clean-room
// implementation of a voxel landscape shown in a constant glowing-edge style.

#include <metal_stdlib>
using namespace metal;

#define VE_VOXEL_SIZE      1.44225f
#define VE_WORLD_MIN       int3(-25, -6, -31)
#define VE_WORLD_MAX       int3(25, 24, 25)
#define VE_MAX_TRACE_STEPS 58
#define VE_MAX_TRACE_DIST  62.0f

struct VoxelEdgesUniforms {
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

struct VoxelEdgesVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex VoxelEdgesVertexOut voxelEdgesVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant VoxelEdgesUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  VoxelEdgesVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float ve_hash21(float2 p) {
  p = fract(p * float2(0.1031f, 0.1030f));
  p += dot(p, p.yx + 19.19f);
  return fract((p.x + p.y) * p.x);
}

static float ve_noise2(float2 p) {
  float2 i = floor(p);
  float2 f = fract(p);
  f = f * f * (3.0f - 2.0f * f);
  float a = ve_hash21(i + float2(0.0f, 0.0f));
  float b = ve_hash21(i + float2(1.0f, 0.0f));
  float c = ve_hash21(i + float2(0.0f, 1.0f));
  float d = ve_hash21(i + float2(1.0f, 1.0f));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

static float ve_hash31(float3 p) {
  p = fract(p * 0.1031f);
  p += dot(p, p.yzx + 31.32f);
  return fract((p.x + p.y) * p.z);
}

static float ve_noise3(float3 p) {
  float3 i = floor(p);
  float3 f = fract(p);
  f = f * f * (3.0f - 2.0f * f);

  float n000 = ve_hash31(i + float3(0.0f, 0.0f, 0.0f));
  float n100 = ve_hash31(i + float3(1.0f, 0.0f, 0.0f));
  float n010 = ve_hash31(i + float3(0.0f, 1.0f, 0.0f));
  float n110 = ve_hash31(i + float3(1.0f, 1.0f, 0.0f));
  float n001 = ve_hash31(i + float3(0.0f, 0.0f, 1.0f));
  float n101 = ve_hash31(i + float3(1.0f, 0.0f, 1.0f));
  float n011 = ve_hash31(i + float3(0.0f, 1.0f, 1.0f));
  float n111 = ve_hash31(i + float3(1.0f, 1.0f, 1.0f));

  float nx00 = mix(n000, n100, f.x);
  float nx10 = mix(n010, n110, f.x);
  float nx01 = mix(n001, n101, f.x);
  float nx11 = mix(n011, n111, f.x);
  float nxy0 = mix(nx00, nx10, f.y);
  float nxy1 = mix(nx01, nx11, f.y);
  return mix(nxy0, nxy1, f.z);
}

static float ve_fbm(float2 p) {
  float sum = 0.0f;
  float amp = 0.5f;
  float2 pp = p;
  for (int i = 0; i < 3; ++i) {
    sum += amp * ve_noise2(pp);
    pp = float2(1.7f * pp.x - 1.2f * pp.y, 1.2f * pp.x + 1.7f * pp.y);
    amp *= 0.5f;
  }
  return sum;
}

static float ve_terrainHeight(float2 xz, float time) {
  float2 p = xz * 0.085f + float2(0.0f, time * 0.18f);
  float broad = ve_fbm(p);
  float detail = ve_fbm(p * 2.1f + float2(7.1f, -3.7f));
  float dunes = 0.9f * sin(xz.x * 0.07f + time * 0.4f) + 0.6f * cos(xz.y * 0.05f - time * 0.3f);
  return -1.0f + 10.0f * broad + 3.5f * detail + dunes;
}

static float ve_terrainField(float3 p, float time) {
  float3 samplePos = p * 0.1f;
  samplePos.xz *= 0.6f;

  float animTime = 0.5f + 0.15f * time;
  float ft = fract(animTime);
  float it = floor(animTime);
  ft = smoothstep(0.7f, 1.0f, ft);
  animTime = it + ft;
  float speed = 1.4f;

  float field = 0.5f * ve_noise3(samplePos * 1.00f + float3(0.0f, 1.0f, 0.0f) * speed * animTime);
  field += 0.25f * ve_noise3(samplePos * 2.02f + float3(0.0f, 2.0f, 0.0f) * speed * animTime);
  field += 0.125f * ve_noise3(samplePos * 4.01f);
  return 25.0f * field - 10.0f;
}

static bool ve_insideWorld(int3 cell) {
  return all(cell >= VE_WORLD_MIN) && all(cell <= VE_WORLD_MAX);
}

static float ve_voxelValue(int3 cell, float time, float3 cameraOrigin) {
  if (!ve_insideWorld(cell)) return false;
  float3 p = float3(cell) + 0.5f;
  float density = ve_terrainField(p, time) + 0.25f * p.y;
  float cameraClear = step(length(cameraOrigin - p), 5.0f);
  density = mix(density, 1.0f, cameraClear);
  return density <= 0.5f ? 1.0f : 0.0f;
}

static bool ve_voxelSolid(int3 cell, float time, float3 cameraOrigin) {
  return ve_voxelValue(cell, time, cameraOrigin) > 0.5f;
}

static bool ve_isFrontBoundaryOutside(int3 cell, float3 viewDir) {
  bool outsideXMin = cell.x < VE_WORLD_MIN.x && viewDir.x > 0.0f;
  bool outsideXMax = cell.x > VE_WORLD_MAX.x && viewDir.x < 0.0f;
  bool outsideYMin = cell.y < VE_WORLD_MIN.y && viewDir.y > 0.0f;
  bool outsideYMax = cell.y > VE_WORLD_MAX.y && viewDir.y < 0.0f;
  bool outsideZMin = cell.z < VE_WORLD_MIN.z && viewDir.z > 0.0f;
  bool outsideZMax = cell.z > VE_WORLD_MAX.z && viewDir.z < 0.0f;
  return outsideXMin || outsideXMax || outsideYMin || outsideYMax || outsideZMin || outsideZMax;
}

static float ve_neighborValue(int3 cell, float time, float3 cameraOrigin, float3 viewDir) {
  if (ve_insideWorld(cell)) {
    return ve_voxelValue(cell, time, cameraOrigin);
  }
  return ve_isFrontBoundaryOutside(cell, viewDir) ? 0.0f : 1.0f;
}

static float3 ve_sky(float3 rd) {
  float up = clamp(rd.y * 0.5f + 0.5f, 0.0f, 1.0f);
  float horizon = exp(-8.0f * abs(rd.y));
  float3 col = mix(float3(0.005f, 0.006f, 0.010f), float3(0.018f, 0.014f, 0.012f), up);
  col += horizon * float3(0.020f, 0.010f, 0.004f);
  return col;
}

static bool ve_boxHit(
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

static bool ve_trace(
  float3 ro, float3 rd, float time, float3 cameraOrigin, thread float &hitT,
  thread float3 &hitPos, thread float3 &hitNormal, thread int3 &hitCell)
{
  float3 dir = normalize(rd);
  float3 cellf = floor(ro);
  int3 cell = int3(cellf);
  float3 stepDir = sign(dir);
  int3 stepI = int3(stepDir);
  float3 nextBoundary = cellf + step(float3(0.0f), stepDir);

  float3 invDir = 1.0f / max(abs(dir), 1e-4f);
  float3 tDelta = invDir;
  float3 tMax = abs((nextBoundary - ro) * invDir);

  float currentT = 0.0f;
  float3 currentNormal = float3(0.0f, 1.0f, 0.0f);

  for (int i = 0; i < VE_MAX_TRACE_STEPS; ++i) {
    if (ve_insideWorld(cell) && ve_voxelSolid(cell, time, cameraOrigin)) {
      hitT = currentT;
      hitPos = ro + dir * currentT;
      hitNormal = currentNormal;
      hitCell = cell;
      return true;
    }

    if (tMax.x < tMax.y && tMax.x < tMax.z) {
      currentT = tMax.x;
      tMax.x += tDelta.x;
      cell.x += stepI.x;
      currentNormal = float3(-stepDir.x, 0.0f, 0.0f);
    } else if (tMax.y < tMax.z) {
      currentT = tMax.y;
      tMax.y += tDelta.y;
      cell.y += stepI.y;
      currentNormal = float3(0.0f, -stepDir.y, 0.0f);
    } else {
      currentT = tMax.z;
      tMax.z += tDelta.z;
      cell.z += stepI.z;
      currentNormal = float3(0.0f, 0.0f, -stepDir.z);
    }

    if (currentT > VE_MAX_TRACE_DIST) break;
  }

  return false;
}

static float2 ve_faceUV(float3 pos, float3 normal) {
  float3 local = fract(pos);
  if (abs(normal.x) > 0.5f) return local.zy;
  if (abs(normal.y) > 0.5f) return local.xz;
  return local.xy;
}

static void ve_faceAxes(float3 normal, thread int3 &axisA, thread int3 &axisB) {
  if (abs(normal.x) > 0.5f) {
    axisA = int3(0, 1, 0);
    axisB = int3(0, 0, 1);
  } else if (abs(normal.y) > 0.5f) {
    axisA = int3(1, 0, 0);
    axisB = int3(0, 0, 1);
  } else {
    axisA = int3(1, 0, 0);
    axisB = int3(0, 1, 0);
  }
}

static float ve_max4(float4 v) {
  return max(max(v.x, v.y), max(v.z, v.w));
}

static float ve_contourMask(float2 uv, float4 va, float4 vb, float4 vc, float4 vd) {
  float2 st = 1.0f - uv;
  float4 edgeWeights = smoothstep(0.85f, 0.99f, float4(uv.x, st.x, uv.y, st.y))
    * (1.0f - va + va * vc);
  float4 cornerWeights = smoothstep(
    0.85f, 0.99f,
    float4(uv.x * uv.y, st.x * uv.y, st.x * st.y, uv.x * st.y))
    * (1.0f - vb + vd * vb);
  return ve_max4(max(edgeWeights, cornerWeights));
}

static bool ve_sceneHit(
  float3 ro, float3 rd,
  thread float &tNear, thread float &tFar)
{
  float3 bmin = float3(VE_WORLD_MIN) * VE_VOXEL_SIZE;
  float3 bmax = (float3(VE_WORLD_MAX) + 1.0f) * VE_VOXEL_SIZE;
  return ve_boxHit(ro, rd, bmin, bmax, tNear, tFar);
}

fragment float4 voxelEdgesFragment(
  VoxelEdgesVertexOut in [[stage_in]],
  constant VoxelEdgesUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center = uniforms.objectCenter.xyz;
  float cubeScale = uniforms.cubeScale;
  float3 worldDir = normalize(in.worldPos - camWorld);
  float3 rd = normalize(worldDir);

  float3 roLocal = (camWorld - center) / cubeScale;
  float3 rdLocal = rd;

  float tEntry;
  float tExit;
  if (!ve_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float3 entryLocal = roLocal + rdLocal * max(tEntry, 0.0f);
  float sceneTime = uniforms.time * uniforms.travelSpeed;
  float3 sceneRo = entryLocal * 18.0f + float3(0.0f, 6.0f, -18.0f);
  float3 sceneRd = rdLocal;

  float sceneTNear;
  float sceneTFar;
  if (!ve_sceneHit(sceneRo, sceneRd, sceneTNear, sceneTFar)) {
    return float4(ve_sky(sceneRd), 1.0f);
  }

  float startT = max(sceneTNear, 0.0f);
  sceneRo += sceneRd * startT;

  float hitT;
  float3 hitPos;
  float3 hitNormal;
  int3 hitCell = int3(0);
  float3 color = ve_sky(sceneRd);

  float3 voxelRo = sceneRo / VE_VOXEL_SIZE;
  float3 voxelRd = sceneRd;
  float3 cameraOrigin = voxelRo;

  if (ve_trace(voxelRo, voxelRd, sceneTime, cameraOrigin, hitT, hitPos, hitNormal, hitCell)) {
    float3 sceneHitPos = hitPos * VE_VOXEL_SIZE;
    float2 uv = ve_faceUV(hitPos, hitNormal);

    int3 normalCell = int3(int(hitNormal.x), int(hitNormal.y), int(hitNormal.z));
    int3 axisA;
    int3 axisB;
    ve_faceAxes(hitNormal, axisA, axisB);

    float4 vc = float4(
      ve_neighborValue(hitCell + normalCell + axisA, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell + normalCell - axisA, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell + normalCell + axisB, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell + normalCell - axisB, sceneTime, cameraOrigin, voxelRd));
    float4 vd = float4(
      ve_neighborValue(hitCell + normalCell + axisA + axisB, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell + normalCell - axisA + axisB, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell + normalCell - axisA - axisB, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell + normalCell + axisA - axisB, sceneTime, cameraOrigin, voxelRd));
    float4 va = float4(
      ve_neighborValue(hitCell + axisA, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell - axisA, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell + axisB, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell - axisB, sceneTime, cameraOrigin, voxelRd));
    float4 vb = float4(
      ve_neighborValue(hitCell + axisA + axisB, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell - axisA + axisB, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell - axisA - axisB, sceneTime, cameraOrigin, voxelRd),
      ve_neighborValue(hitCell + axisA - axisB, sceneTime, cameraOrigin, voxelRd));

    float sixNeighborOccupancy =
      ve_neighborValue(hitCell + int3(1, 0, 0), sceneTime, cameraOrigin, voxelRd) +
      ve_neighborValue(hitCell + int3(-1, 0, 0), sceneTime, cameraOrigin, voxelRd) +
      ve_neighborValue(hitCell + int3(0, 1, 0), sceneTime, cameraOrigin, voxelRd) +
      ve_neighborValue(hitCell + int3(0, -1, 0), sceneTime, cameraOrigin, voxelRd) +
      ve_neighborValue(hitCell + int3(0, 0, 1), sceneTime, cameraOrigin, voxelRd) +
      ve_neighborValue(hitCell + int3(0, 0, -1), sceneTime, cameraOrigin, voxelRd);
    bool fullyEnclosed = sixNeighborOccupancy > 5.5f;

    bool boundaryFace = ve_isFrontBoundaryOutside(hitCell + normalCell, voxelRd);

    float contourGlow = ve_contourMask(uv, va, vb, vc, vd);
    float edgeDistance = min(min(uv.x, 1.0f - uv.x), min(uv.y, 1.0f - uv.y));
    float faceEdge = 1.0f - smoothstep(0.02f, 0.06f, edgeDistance);
    float fineEdge = 1.0f - smoothstep(0.008f, 0.022f, edgeDistance);
    contourGlow = max(contourGlow, faceEdge * (1.0f - ve_max4(va)) * 0.3f);
    float hiddenEdgeMask = boundaryFace ? 0.0f : faceEdge * (1.0f - contourGlow);

    float sky = 0.5f + 0.5f * hitNormal.y;
    float ambient = clamp(0.5f + 0.02f * sceneHitPos.y, 0.15f, 1.0f);
    float fog = exp(-0.032f * (hitT * VE_VOXEL_SIZE + startT));
    float heightTint = clamp((sceneHitPos.y + 6.0f) / 20.0f, 0.0f, 1.0f);

    float3 fill = mix(float3(0.035f, 0.038f, 0.040f), float3(0.080f, 0.076f, 0.070f), heightTint);
    fill *= 0.38f + 0.34f * sky + 0.28f * ambient;
    fill *= 1.0f - 0.72f * contourGlow;

    if (boundaryFace) {
      float boundaryLight = 0.55f + 0.25f * sky + 0.20f * ambient;
      float3 boundaryFill = mix(float3(0.070f, 0.066f, 0.060f), float3(0.125f, 0.115f, 0.100f), heightTint);
      fill = boundaryFill * boundaryLight;
      fill += fineEdge * 0.06f * float3(0.50f, 0.38f, 0.24f);
    }

    float3 hiddenTint = mix(float3(0.014f, 0.013f, 0.012f), float3(0.028f, 0.024f, 0.020f), heightTint);
    fill = mix(hiddenTint, fill, clamp(0.25f + 0.75f * contourGlow, 0.0f, 1.0f));
    fill -= hiddenEdgeMask * 0.035f * float3(1.0f, 0.9f, 0.8f);
    float3 hiddenEdgeGlow = hiddenEdgeMask * mix(float3(0.050f, 0.030f, 0.012f), float3(0.085f, 0.050f, 0.018f), heightTint);

    float3 edgeGlow = mix(float3(3.2f, 0.55f, 0.02f), float3(5.4f, 0.95f, 0.10f), heightTint);
    edgeGlow *= (0.45f + 0.55f * ambient) * contourGlow;

    color = fullyEnclosed ? ve_sky(sceneRd) : (fill + hiddenEdgeGlow + edgeGlow);
    color *= fog;
    color += 0.22f * ve_sky(sceneRd) * (1.0f - fog);
    color = pow(max(color, 0.0f), float3(0.82f));
  }

  return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}