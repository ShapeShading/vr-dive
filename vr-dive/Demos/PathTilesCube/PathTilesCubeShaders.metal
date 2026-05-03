// PathTilesCubeShaders.metal
//
// Original cube-portal raymarch scene of rounded pillars and a drifting path.
// Visual inspiration requested from ShaderToy s3fGR8:
// https://www.shadertoy.com/view/s3fGR8
// This implementation is original and does not reuse source code from the
// reference shader.

#include <metal_stdlib>
using namespace metal;

#define PTC_MAX_STEPS    52
#define PTC_MAX_DIST     42.0f
#define PTC_HIT_EPS      0.03f
#define PTC_SCENE_SCALE  11.5f
#define PTC_TILE_SIZE    2.6f

struct PathTilesCubeUniforms {
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

struct PathTilesCubeVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex PathTilesCubeVertexOut pathTilesCubeVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant PathTilesCubeUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  PathTilesCubeVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

static float ptc_hash21(float2 p) {
  float3 p3 = fract(float3(p.x, p.y, p.x) * 0.1031f);
  p3 += dot(p3, p3.yzx + 33.33f);
  return fract((p3.x + p3.y) * p3.z);
}

static float2 ptc_rot(float2 p, float a) {
  float c = cos(a);
  float s = sin(a);
  return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float ptc_roundBox(float3 p, float3 b, float r) {
  return length(max(abs(p) - b, 0.0f)) - r;
}

static float2 ptc_path(float z, float time) {
  float zz = z * 0.12f + time * 0.22f;
  return float2(sin(zz) * cos(zz * 0.53f) * 6.5f, 0.0f);
}

static float ptc_tileHeight(float2 tileId, float time) {
  float seeded = ptc_hash21(tileId * 0.73f);
  float pulse = 0.5f + 0.5f * sin(time * 0.9f + seeded * 6.28318f);
  return 0.8f + pow(seeded, 1.9f) * 7.5f + pulse * 0.7f;
}

static float ptc_tileField(float3 p, float time, thread float &matId) {
  float floorY = -3.0f;
  float d = p.y - floorY;
  matId = 2.0f;

  float2 field = p.xz / PTC_TILE_SIZE;
  float2 baseId = floor(field);
  float2 local = fract(field) - 0.5f;
  float best = d;
  float bestMat = 2.0f;

  int x0 = local.x > 0.0f ? 0 : -1;
  int x1 = x0 + 1;
  int z0 = local.y > 0.0f ? 0 : -1;
  int z1 = z0 + 1;

  for (int iz = z0; iz <= z1; ++iz) {
    for (int ix = x0; ix <= x1; ++ix) {
      float2 tileId = baseId + float2(float(ix), float(iz));
      float2 center = (tileId + 0.5f) * PTC_TILE_SIZE;
      float2 laneCenter = ptc_path(center.y, time);
      float laneDist = abs(center.x - laneCenter.x);
      if (laneDist < 1.35f) {
        continue;
      }

      float h = ptc_tileHeight(tileId, time);
      float3 q = p - float3(center.x, floorY + h * 0.5f, center.y);
      q.xz = ptc_rot(q.xz, 0.2f * sin(time + tileId.x * 0.7f + tileId.y * 0.35f));
      float pillar = ptc_roundBox(q, float3(0.58f, h * 0.5f, 0.58f), 0.12f);
      if (pillar < best) {
        best = pillar;
        bestMat = 0.0f;
      }

      float cap = ptc_roundBox(
        p - float3(center.x, floorY + h + 0.35f, center.y),
        float3(0.42f, 0.12f, 0.42f), 0.08f);
      if (cap < best) {
        best = cap;
        bestMat = 1.0f;
      }
    }
  }

  matId = bestMat;
  return best;
}

static float ptc_sceneSdf(float3 p, float time, thread float &matId) {
  return ptc_tileField(p, time, matId);
}

static float3 ptc_normal(float3 p, float time) {
  const float e = 0.05f;
  float matId;
  float dx = ptc_sceneSdf(p + float3(e, 0, 0), time, matId) - ptc_sceneSdf(p - float3(e, 0, 0), time, matId);
  float dy = ptc_sceneSdf(p + float3(0, e, 0), time, matId) - ptc_sceneSdf(p - float3(0, e, 0), time, matId);
  float dz = ptc_sceneSdf(p + float3(0, 0, e), time, matId) - ptc_sceneSdf(p - float3(0, 0, e), time, matId);
  return normalize(float3(dx, dy, dz));
}

static bool ptc_boxHit(
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

static float3 ptc_sky(float3 rd) {
  float t = clamp(rd.y * 0.5f + 0.5f, 0.0f, 1.0f);
  float3 horizon = float3(0.95f, 0.97f, 1.00f);
  float3 mid = float3(0.20f, 0.45f, 0.75f);
  float3 zenith = float3(0.02f, 0.05f, 0.12f);
  float3 col = mix(horizon, mid, smoothstep(0.0f, 0.5f, t));
  col = mix(col, zenith, smoothstep(0.4f, 1.0f, t));
  col += 0.15f * float3(1.0f, 0.75f, 0.45f) / (1.0f + 18.0f * abs(rd.y - 0.08f));
  return col;
}

fragment float4 pathTilesCubeFragment(
  PathTilesCubeVertexOut in [[stage_in]],
  constant PathTilesCubeUniforms &uniforms [[buffer(0)]],
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
  if (!ptc_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  float startT = max(tEntry, 0.0f);
  float3 entryLocal = roLocal + rdLocal * startT;
  float3 ro = entryLocal * PTC_SCENE_SCALE;
  float3 rd = rdLocal;
  float time = uniforms.time * uniforms.travelSpeed;

  float t = 0.0f;
  float matId = 2.0f;
  bool hit = false;
  float3 sp = float3(0.0f);
  for (int i = 0; i < PTC_MAX_STEPS; ++i) {
    if (t > PTC_MAX_DIST) { break; }
    float dist = ptc_sceneSdf(ro + rd * t, time, matId);
    if (dist < PTC_HIT_EPS) {
      hit = true;
      sp = ro + rd * t;
      break;
    }
    t += clamp(dist * 0.82f, 0.05f, 1.15f);
  }

  float farMix = smoothstep(0.0f, 1.0f, t / PTC_MAX_DIST);
  float3 sky = ptc_sky(rd);
  float3 color = sky;

  if (hit) {
    float3 sn = ptc_normal(sp, time);
    float3 lightPos = ro + float3(0.0f, 7.0f, 12.0f);
    float3 lv = lightPos - sp;
    float ldist = max(length(lv), 0.001f);
    float3 ldir = lv / ldist;
    float diff = max(dot(sn, ldir), 0.0f);
    float spec = pow(max(dot(reflect(-ldir, sn), -rd), 0.0f), 10.0f);
    float fres = pow(clamp(1.0f + dot(rd, sn), 0.0f, 1.0f), 1.4f);

    float2 cellId = floor(sp.xz / PTC_TILE_SIZE);
    float seed = ptc_hash21(cellId);
    float3 base = mix(float3(0.10f, 0.12f, 0.15f), float3(0.00f, 0.80f, 1.00f), step(0.52f, seed));
    base = mix(base, float3(1.0f, 0.0f, 0.4f), step(0.82f, seed));
    if (matId > 1.5f) {
      base = float3(0.18f, 0.18f, 0.20f);
    } else if (matId > 0.5f) {
      base = float3(1.0f, 0.85f, 0.0f);
    }

    float atten = 1.0f / (1.0f + 0.014f * ldist * ldist);
    float pathGlow = 1.0f / (1.0f + 0.12f * abs(sp.x - ptc_path(sp.z, time).x));
    float3 lit = base * (0.18f + 1.25f * diff) + spec * float3(0.9f, 0.5f, 0.2f);
    lit += fres * 0.15f * sky;
    lit += pathGlow * float3(0.08f, 0.14f, 0.22f);
    color = lit * atten;
    color = mix(color, sky, farMix);
  }

  float vignette = 1.0f - smoothstep(0.7f, 1.8f, length(in.worldPos.xy - center.xy));
  color *= mix(0.82f, 1.0f, vignette);
  color = pow(max(color, 0.0f), float3(0.9f));
  return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}