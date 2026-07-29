// WaveySpheresShaders.metal
//
// Source reference requested by user:
// https://www.shadertoy.com/view/WX3cR4
// This is an original Metal adaptation for vr-dive that preserves the
// wave-driven color spirit of the reference while rebuilding the scene as a
// fully view-independent 3D volume inside a 2 meter cube container.

#include <metal_stdlib>
using namespace metal;

#define WS_PI            3.14159265f
// Grid spacing from reference (xz = .3, LAYER_DISTANCE = 5)
#define WS_XZ_SPACING    0.30f
#define WS_LAYER_DIST    5.00f
#define WS_SCENE_SCALE   5.50f
#define WS_MAX_STEPS     72

struct WaveySpheresUniforms {
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

struct WaveySpheresVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex WaveySpheresVertexOut waveySpheresVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant WaveySpheresUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

  WaveySpheresVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

// 14-color palette: port from reference
static constant float3 WS_COLS[14] = {
  float3(47.0f, 75.0f, 162.0f) / 255.0f,
  float3(233.0f, 71.0f, 245.0f) / 255.0f,
  float3(128.0f, 63.0f, 224.0f) / 255.0f,
  float3(61.0f, 199.0f, 220.0f) / 255.0f,
  float3(222.0f, 51.0f, 150.0f) / 255.0f,
  float3(160.0f, 220.0f, 70.0f) / 255.0f,
  float3(245.0f, 140.0f, 60.0f) / 255.0f,
  float3(38.0f, 178.0f, 133.0f) / 255.0f,
  float3(220.0f, 50.0f, 50.0f) / 255.0f,
  float3(240.0f, 220.0f, 80.0f) / 255.0f,
  float3(180.0f, 90.0f, 240.0f) / 255.0f,
  float3(80.0f, 210.0f, 255.0f) / 255.0f,
  float3(245.0f, 80.0f, 220.0f) / 255.0f,
  float3(70.0f, 200.0f, 100.0f) / 255.0f,
};

// get_color: port from reference (t in [0, 1])
static float3 ws_getColor(float t) {
  float x = clamp(t, 0.0f, 0.9999f) * 13.0f;
  uint i0 = (uint)floor(x);
  uint i1 = min(i0 + 1u, 13u);
  return mix(WS_COLS[i0], WS_COLS[i1], x - floor(x));
}

// hash41: port from reference
static float4 ws_hash41(float p) {
  float4 p4 = fract(float4(p) * float4(0.1031f, 0.1030f, 0.0973f, 0.1099f));
  p4 += dot(p4, p4.wzxy + 33.33f);
  return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

// get_height: exact port from reference
// id = (x, z) grid cell index, layer = y grid cell index, t = iTime
static float ws_getHeight(float2 id, float layer, float t) {
  float4 h = ws_hash41(layer) * 1000.0f;
  float o = 0.0f;
  o += sin((id.x + h.x) * 0.2f + t) * 0.3f;
  o += sin((id.y + h.y) * 0.2f + t) * 0.3f;
  o += sin((-id.x + id.y + h.z) * 0.3f + t) * 0.3f;
  o += sin((id.x + id.y + h.z) * 0.3f + t) * 0.4f;
  o += sin((id.x - id.y + h.w) * 0.8f + t) * 0.1f;
  return o;
}

// map: port from reference
// Spheres on a grid: xz spacing = WS_XZ_SPACING, y spacing = WS_LAYER_DIST
// Each sphere's y position is offset by get_height(); radius varies with that offset
static float ws_map(float3 p, float t) {
  float3 s = float3(WS_XZ_SPACING, WS_LAYER_DIST, WS_XZ_SPACING);
  float3 id = round(p / s);
  float ho = ws_getHeight(id.xz, id.y, t);
  float3 q = p;
  q.y += ho;
  q -= s * id;
  return length(q) - (smoothstep(1.3f, -1.3f, ho) * 0.03f + 0.0001f);
}

static bool ws_boxHit(
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

static float ws_edgeDistance(float3 p) {
  float3 a = abs(p);
  if (a.x > a.y && a.x > a.z) return min(1.0f - a.y, 1.0f - a.z);
  if (a.y > a.z) return min(1.0f - a.x, 1.0f - a.z);
  return min(1.0f - a.x, 1.0f - a.y);
}

static float3 ws_faceNormal(float3 p) {
  float3 a = abs(p);
  if (a.x > a.y && a.x > a.z) return float3(sign(p.x), 0.0f, 0.0f);
  if (a.y > a.z) return float3(0.0f, sign(p.y), 0.0f);
  return float3(0.0f, 0.0f, sign(p.z));
}

fragment float4 waveySpheresFragment(
  WaveySpheresVertexOut in [[stage_in]],
  constant WaveySpheresUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]])
{
  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

  float3 center = uniforms.objectCenter.xyz;
  float3 roLocal = (camWorld - center) / uniforms.cubeScale;
  float3 rdLocal = normalize(in.worldPos - camWorld);

  float tEntry, tExit;
  if (!ws_boxHit(roLocal, rdLocal, float3(-1.0f), float3(1.0f), tEntry, tExit)) {
    discard_fragment();
  }

  // Cube entry shell (edge + fresnel hint)
  float3 entryLocal = roLocal + rdLocal * max(tEntry, 0.0f);
  float3 faceNrm = ws_faceNormal(entryLocal);
  float fresnel = pow(1.0f - max(0.0f, dot(-rdLocal, faceNrm)), 2.0f);
  float edge = smoothstep(0.18f, 0.02f, ws_edgeDistance(entryLocal));
  float3 shellColor = float3(0.04f, 0.06f, 0.09f)
    + float3(0.25f, 0.55f, 0.78f) * fresnel * 0.45f
    + float3(0.85f, 0.95f, 1.00f) * edge * 0.32f;

  // iTime equivalent
  float t = uniforms.time * uniforms.travelSpeed;

  // Phase variables: port of reference's mainImage preamble
  float phase = t * 0.2f;
  float y = sin(phase);
  float ny = smoothstep(-1.0f, 1.0f, y);

  // Single cycling color per frame: port of reference's c = get_color(...)
  // One full palette pass every 5 * 2*PI ≈ 31 s
  float3 c = ws_getColor(fract(t / (5.0f * 2.0f * WS_PI)));

  // Virtual camera in scene space: port of reference's ro = vec3(0, y*LAYER_DISTANCE*.5, -t)
  float3 virtualCam = float3(0.0f, y * WS_LAYER_DIST * 0.5f, -t);

  // Ray origin: cube entry point scaled to scene space + virtual camera offset
  // This maps the bounded cube into the reference's infinite flying-camera scene
  float3 roScene = entryLocal * WS_SCENE_SCALE + virtualCam;
  float3 rdScene = normalize(rdLocal);
  float travelLength = (tExit - max(tEntry, 0.0f)) * WS_SCENE_SCALE;

  // Glow accumulation: direct port of reference's main loop
  float3 col = float3(0.0f);
  float d = 0.0f;
  for (int i = 0; i < WS_MAX_STEPS; ++i) {
    if (d >= travelLength) break;
    float3 p = roScene + rdScene * d;
    float dt = ws_map(p, t);
    // Step modulation from reference: dt*(cos(ny*PI*2.)*.3+.5)
    dt = max(dt * (cos(ny * WS_PI * 2.0f) * 0.3f + 0.5f), 1e-3f);
    col += (0.1f / dt) * c;
    d += dt * 0.8f;
  }

  // Tonemap: port of reference's tanh(col * .01)
  col = tanh(col * 0.01f);

  // Overlay subtle cube shell on dark areas
  col += shellColor * (1.0f - smoothstep(0.0f, 0.3f, dot(col, float3(0.3f, 0.6f, 0.1f)))) * 0.15f;

  return float4(col, 1.0f);
}