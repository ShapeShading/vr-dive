// VoxelTerrainCubeShaders.metal
//
// Original implementation for a voxel terrain cube portal.
// Visual inspiration was requested from ShaderToy 4dfGzs.
// The original source license forbids reuse in products/projects, so no source
// code from the original work is copied here. This shader is a clean-room
// implementation of a lit voxel landscape viewed through a cube container.

#include <metal_stdlib>
using namespace metal;

#define VT_VOXEL_SIZE      1.44225f   // cbrt(3): each voxel has ~3x the original volume
#define VT_WORLD_MIN       int3(-25, -6, -31)
#define VT_WORLD_MAX       int3(25, 15, 25)
#define VT_MAX_TRACE_STEPS 58
#define VT_MAX_TRACE_DIST  62.0f

struct VoxelTerrainCubeUniforms {
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

struct VoxelTerrainCubeVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex VoxelTerrainCubeVertexOut voxelTerrainCubeVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant VoxelTerrainCubeUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  VoxelTerrainCubeVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float vt_hash21(float2 p) {
  p = fract(p * float2(0.1031f, 0.1030f));
  p += dot(p, p.yx + 19.19f);
  return fract((p.x + p.y) * p.x);
}

static float vt_noise2(float2 p) {
  float2 i = floor(p);
  float2 f = fract(p);
  f = f * f * (3.0f - 2.0f * f);
  float a = vt_hash21(i + float2(0.0f, 0.0f));
  float b = vt_hash21(i + float2(1.0f, 0.0f));
  float c = vt_hash21(i + float2(0.0f, 1.0f));
  float d = vt_hash21(i + float2(1.0f, 1.0f));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

static float vt_fbm(float2 p) {
  float sum = 0.0f;
  float amp = 0.5f;
  float2 pp = p;
  for (int i = 0; i < 3; ++i) {
    sum += amp * vt_noise2(pp);
    pp = float2(1.7f * pp.x - 1.2f * pp.y, 1.2f * pp.x + 1.7f * pp.y);
    amp *= 0.5f;
  }
  return sum;
}

static float vt_terrainHeight(float2 xz, float time) {
  float2 p = xz * 0.085f + float2(0.0f, time * 0.18f);
  float broad = vt_fbm(p);
  float detail = vt_fbm(p * 2.1f + float2(7.1f, -3.7f));
  float dunes = 0.9f * sin(xz.x * 0.07f + time * 0.4f) + 0.6f * cos(xz.y * 0.05f - time * 0.3f);
  return -1.0f + 10.0f * broad + 3.5f * detail + dunes;
}

static bool vt_insideWorld(int3 cell) {
  return all(cell >= VT_WORLD_MIN) && all(cell <= VT_WORLD_MAX);
}

static bool vt_voxelSolid(int3 cell, float time) {
  if (!vt_insideWorld(cell)) return false;
  float h = vt_terrainHeight(float2(float(cell.x), float(cell.z)), time);
  if (float(cell.y) > h) return false;

  float cave = vt_noise2(float2(cell.x, cell.z) * 0.21f + float2(13.0f, -7.0f));
  if (cell.y > 2 && cave > 0.78f) return false;

  return true;
}

static float3 vt_sky(float3 rd) {
  float up = clamp(rd.y * 0.5f + 0.5f, 0.0f, 1.0f);
  float horizon = exp(-8.0f * abs(rd.y));
  float3 col = mix(float3(0.03f, 0.04f, 0.08f), float3(0.26f, 0.18f, 0.10f), up);
  col += horizon * float3(0.45f, 0.18f, 0.02f);
  return col;
}

static bool vt_boxHit(
  float3 ro, float3 rd, float3 bmin, float3 bmax,
  thread float &tNear, thread float &tFar)
{
  float3 inv = 1.0f / max(abs(rd), 1e-5f) * sign(rd);
  float3 t0 = (bmin - ro) / rd;
  float3 t1 = (bmax - ro) / rd;
  float3 lo = min(t0, t1);
  float3 hi = max(t0, t1);
  tNear = max(max(lo.x, lo.y), lo.z);
  tFar = min(min(hi.x, hi.y), hi.z);
  return tFar >= max(tNear, 0.0f);
}

static bool vt_trace(
  float3 ro, float3 rd, float time, thread float &hitT,
  thread float3 &hitPos, thread float3 &hitNormal)
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

  for (int i = 0; i < VT_MAX_TRACE_STEPS; ++i) {
    if (vt_insideWorld(cell) && vt_voxelSolid(cell, time)) {
      hitT = currentT;
      hitPos = ro + dir * currentT;
      hitNormal = currentNormal;
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

    if (currentT > VT_MAX_TRACE_DIST) break;
  }

  return false;
}

static float2 vt_faceUV(float3 pos, float3 normal) {
  float3 local = fract(pos);
  if (abs(normal.x) > 0.5f) return local.zy;
  if (abs(normal.y) > 0.5f) return local.xz;
  return local.xy;
}

static bool vt_sceneHit(
  float3 ro, float3 rd,
  thread float &tNear, thread float &tFar)
{
  float3 bmin = float3(VT_WORLD_MIN) * VT_VOXEL_SIZE;
  float3 bmax = (float3(VT_WORLD_MAX) + 1.0f) * VT_VOXEL_SIZE;
  return vt_boxHit(ro, rd, bmin, bmax, tNear, tFar);
}

fragment float4 voxelTerrainCubeFragment(
  VoxelTerrainCubeVertexOut in [[stage_in]],
  constant VoxelTerrainCubeUniforms &uniforms [[buffer(0)]],
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
  if (!vt_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float3 entryLocal = roLocal + rdLocal * max(tEntry, 0.0f);
  float sceneTime = uniforms.time * uniforms.travelSpeed;
  float3 sceneRo = entryLocal * 18.0f + float3(0.0f, 6.0f, -18.0f);
  float3 sceneRd = rdLocal;

  float sceneTNear;
  float sceneTFar;
  if (!vt_sceneHit(sceneRo, sceneRd, sceneTNear, sceneTFar)) {
    return float4(vt_sky(sceneRd), 1.0f);
  }

  float startT = max(sceneTNear, 0.0f);
  sceneRo += sceneRd * startT;

  float hitT;
  float3 hitPos;
  float3 hitNormal;
  float3 color = vt_sky(sceneRd);

  float3 voxelRo = sceneRo / VT_VOXEL_SIZE;
  float3 voxelRd = sceneRd;

  if (vt_trace(voxelRo, voxelRd, sceneTime, hitT, hitPos, hitNormal)) {
    float3 sceneHitPos = hitPos * VT_VOXEL_SIZE;
    float2 uv = vt_faceUV(hitPos, hitNormal);
    float edge = min(min(uv.x, 1.0f - uv.x), min(uv.y, 1.0f - uv.y));
    float wire = 1.0f - smoothstep(0.04f, 0.12f, edge);

    float3 sunDir = normalize(float3(-0.4f, 0.45f, 0.7f));
    float diffuse = max(dot(hitNormal, sunDir), 0.0f);
    float sky = 0.5f + 0.5f * hitNormal.y;
    float ambient = clamp(0.5f + 0.02f * sceneHitPos.y, 0.15f, 1.0f);
    float fog = exp(-0.032f * (hitT * VT_VOXEL_SIZE + startT));

    float heightTint = clamp((sceneHitPos.y + 6.0f) / 20.0f, 0.0f, 1.0f);
    float3 base = mix(float3(0.10f, 0.16f, 0.18f), float3(0.18f, 0.26f, 0.30f), heightTint);
    float3 light = 1.9f * diffuse * float3(1.0f, 0.88f, 0.68f)
                 + 0.9f * sky * float3(0.30f, 0.24f, 0.16f)
                 + 0.35f * ambient * float3(0.08f, 0.10f, 0.14f);
    float3 glow = wire * float3(1.8f, 0.42f, 0.05f);

    color = base * light + glow;
    color *= fog;
    color += 0.25f * vt_sky(sceneRd) * (1.0f - fog);
    color = pow(max(color, 0.0f), float3(0.82f));
  }

  return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}