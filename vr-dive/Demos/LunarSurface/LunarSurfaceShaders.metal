// LunarSurfaceShaders.metal
//
// Immersive lunar-surface scene: a cratered ground underfoot (procedural ray-marched
// height field), the Earth and the Sun in the sky, and the Milky Way as a textured
// background. A large inward-facing box surrounds the viewer (same technique as
// SynthwaveSunset) so every direction the player looks resolves to either the ground
// or the sky.
//
// Texture credits (baked into the app bundle, see LunarSurfaceRenderer.swift):
// - Earth: NASA Apollo 17 "Blue Marble" (AS17-148-22727), public domain.
// - Milky Way: ESA/Gaia/DPAC all-sky map, CC BY-SA 3.0 IGO — attribution required.

#include <metal_stdlib>
using namespace metal;

// ─── TUNABLE PARAMETERS ────────────────────────────────────────────────────
// Every "knob" that's been adjusted in response to visual feedback so far
// lives here, grouped by feature, instead of scattered as inline literals
// through the functions below. When asked to tweak a size/speed/color, look
// here first — and if a new adjustable value is added elsewhere in this
// file, add a named constant for it here too rather than leaving a bare
// literal in the function body, so future rounds of tuning stay this easy.
// See also /memories/repo/vr-dive-lunarsurface-notes.md in the repo's agent
// memory for a change log of every round of adjustments made against these.

// -- Space elevator: overall shaft/base/top -----------------------------
constexpr constant float2 LS_TOWER_XZ = float2(20.0f, -30.0f);  // fixed world (x,z) position
constexpr constant float LS_TOWER_BASE_H = 14.0f;               // bottom cabin half-less height (full height above ground)
constexpr constant float LS_TOWER_SHAFT_H = 25000.0f;           // shaft length, ~25km
constexpr constant float LS_TOWER_HUB_EXTRA_RADIUS = 6.0f;      // bottom cabin radius = ring outer radius + this
constexpr constant float LS_TOWER_TOPCAP_EXTRA_RADIUS = 6.0f;   // top cabin radius = ring outer radius + this
constexpr constant float LS_TOWER_TOPCAP_HALF_THICK = 18.0f;    // top cabin half-height (36m tall total)

// -- Space elevator: legs (the six "cable" pillars) ----------------------
constexpr constant float LS_PILLAR_RING_RADIUS = 100.0f;  // distance of each leg's axis from the tower center
constexpr constant float LS_PILLAR_RADIUS = 10.0f;        // leg cylinder radius
constexpr constant float LS_CABLE_STRAND_COUNT = 8.0f;    // number of twisted "cable" strands wrapping each leg
constexpr constant float LS_CABLE_SPIRAL_PITCH = 1200.0f; // meters per full revolution of the twist (larger = slower/subtler)

// -- Space elevator: periodic reinforcing rings --------------------------
constexpr constant float LS_RING_RADIAL_HALF_WIDTH = 12.0f;  // ring band extends this far in/out from LS_PILLAR_RING_RADIUS
constexpr constant float LS_RING_SPACING = 1000.0f;          // vertical distance between adjacent rings
constexpr constant float LS_RING_HALF_THICK = 40.0f;         // ring vertical half-thickness (80m tall total)

// -- Space elevator: climber car -----------------------------------------
constexpr constant float LS_CLIMBER_HALF_THICK = 4.0f;      // 8m tall total
constexpr constant float LS_CLIMBER_CYCLE_SPEED = 0.02f;    // how fast it travels the full shaft, in cycles/sec of `uniforms.time`
constexpr constant float LS_CLIMBER_WINDOW_COUNT = 28.0f;   // number of round "airplane window" ports around its middle band

// -- Space elevator: top beacon ------------------------------------------
constexpr constant float LS_BEACON_RANGE = 20.0f;       // distance from the very top over which the beacon glow fades in
constexpr constant float LS_BEACON_BLINK_SPEED = 3.0f;  // radians/sec fed into sin() for the blink

// -- Space elevator: surface colors --------------------------------------
constexpr constant float3 LS_TOWER_BAND_COLOR_A = float3(0.30f, 0.32f, 0.35f);  // hub/leg/topcap banding, light stripe
constexpr constant float3 LS_TOWER_BAND_COLOR_B = float3(0.15f, 0.16f, 0.18f);  // hub/leg/topcap banding, dark stripe
constexpr constant float3 LS_RING_COLOR = float3(0.22f, 0.23f, 0.26f);          // reinforcing ring: darker steel collar
constexpr constant float3 LS_TOPCAP_COLOR = float3(0.26f, 0.27f, 0.30f);        // top counterweight/cabin module
constexpr constant float3 LS_CLIMBER_GRAY_BRIGHT = float3(0.5f, 0.51f, 0.53f);  // climber fuselage, equator band
constexpr constant float3 LS_CLIMBER_GRAY_DARK = float3(0.14f, 0.14f, 0.15f);   // climber fuselage, top/bottom caps
constexpr constant float3 LS_CLIMBER_WINDOW_COLOR = float3(0.04f, 0.06f, 0.09f);
constexpr constant float3 LS_BEACON_COLOR = float3(1.0f, 0.15f, 0.1f);

// -- Moon terrain: near-field (marched) albedo ---------------------------
constexpr constant float3 LS_NEAR_ALBEDO_A = float3(0.085f, 0.08f, 0.075f);
constexpr constant float3 LS_NEAR_ALBEDO_B = float3(0.13f, 0.125f, 0.115f);
constexpr constant float LS_NEAR_FBM_SCALE = 6.0f;
constexpr constant int LS_NEAR_FBM_OCTAVES = 2;
constexpr constant float LS_MARCH_MAX_DIST = 55.0f;  // near-field ray march distance budget

// -- Moon terrain: far-field (flat-plane fallback) albedo ----------------
// Kept visually consistent with the near-field albedo above (same base
// colors) so there's no visible seam where the march gives way to the
// flat-plane approximation.
constexpr constant float LS_FAR_MOTTLE_SCALE = 0.05f;
constexpr constant int LS_FAR_MOTTLE_OCTAVES = 3;
constexpr constant float LS_FAR_SHADE_MULT = 0.65f;
constexpr constant float LS_FAR_SHADE_CLAMP_MIN = 0.3f;
constexpr constant float LS_FAR_SHADE_CLAMP_MAX = 1.35f;
constexpr constant float LS_FAR_MAX_DIST = 20000.0f;

// -- Sky: sun + Earth ------------------------------------------------------
constexpr constant float LS_SUN_DISC_INNER = 0.9993f;   // dot(rd,sunDir) smoothstep start — disc edge softness
constexpr constant float LS_SUN_DISC_OUTER = 0.9998f;
constexpr constant float LS_SUN_DISC_BRIGHTNESS = 6.0f;
constexpr constant float LS_EARTH_ANGULAR_RADIUS = 0.10f;     // apparent size in the sky
constexpr constant float LS_EARTH_DISC_EDGE_INNER = 0.98f;    // disc cutoff softness (no atmosphere glow, see Round 8/11 notes)
constexpr constant float LS_EARTH_DISC_EDGE_OUTER = 1.02f;

// -- Player/camera -----------------------------------------------------------
constexpr constant float LS_MIN_HEIGHT_ABOVE_GROUND = 0.5f;  // clamp so the camera can't render as if underground

struct LunarSurfaceUniforms {
  float time;
  uint viewCount;
  float _pad0;
  float _pad1;
  float4 objectCenter;
  float4 boxHalfExtents;
  float4x4 patternTransform;
  // xy = world-space (x,z) center the baked height map texture is currently
  // centered on (recentres as the player walks in "箱内移动" mode), z = the
  // texture's world-space half-extent, w unused.
  float4 heightMapParams;
};

struct MeshVertex {
  float3 position;
  float3 normal;
};

struct LSVertexOut {
  float4 clipPos [[position]];
  float3 worldPos;
  uint viewIndex [[flat]];
};

vertex LSVertexOut lunarSurfaceVertex(
  ushort amplificationID [[amplification_id]],
  const device MeshVertex *vertices [[buffer(0)]],
  constant LunarSurfaceUniforms &uniforms [[buffer(1)]],
  constant float4x4 *vpMatrices [[buffer(2)]],
  uint vertexID [[vertex_id]])
{
  MeshVertex vtx = vertices[vertexID];
  uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
  float3 worldPos = vtx.position * uniforms.boxHalfExtents.xyz + uniforms.objectCenter.xyz;

  LSVertexOut out;
  out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
  out.worldPos = worldPos;
  out.viewIndex = viewIndex;
  return out;
}

// ─── Hash / noise helpers ─────────────────────────────────────────────────────
static float ls_hash21(float2 p) {
  p = fract(p * float2(123.34f, 456.21f));
  p += dot(p, p + 45.32f);
  return fract(p.x * p.y);
}

// Trig-free hash (no sin()) — the ray march below can call this thousands of
// times per pixel, and transcendental functions are far more expensive than
// plain multiply/fract ALU ops on the GPU. This form still gives a good-enough
// pseudo-random distribution for crater placement.
static float2 ls_hash22(float2 p) {
  float3 p3 = fract(float3(p.x, p.y, p.x) * float3(0.1031f, 0.1030f, 0.0973f));
  p3 += dot(p3, p3.yzx + 33.33f);
  return fract((p3.xx + p3.yz) * p3.zy);
}

static float ls_noise2(float2 p) {
  float2 i = floor(p);
  float2 f = fract(p);
  float a = ls_hash21(i);
  float b = ls_hash21(i + float2(1.0f, 0.0f));
  float c = ls_hash21(i + float2(0.0f, 1.0f));
  float d = ls_hash21(i + float2(1.0f, 1.0f));
  float2 u = f * f * (3.0f - 2.0f * f);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

static float ls_fbm2(float2 p, int octaves) {
  float sum = 0.0f;
  float amp = 0.5f;
  float freq = 1.0f;
  for (int i = 0; i < octaves; i++) {
    sum += amp * ls_noise2(p * freq);
    freq *= 2.03f;
    amp *= 0.5f;
  }
  return sum;
}

// ─── Procedural crater field ──────────────────────────────────────────────────
// One randomly-placed crater per grid cell (searched over the 3x3 neighbourhood),
// bowl-shaped depression with a slightly raised rim. Multiple layers at different
// scales give large maria-scale craters down to small pockmarks.
static float ls_craterLayer(float2 p, float cellSize, float radiusScale, float depth, float rimHeight) {
  float2 pc = p / cellSize;
  float2 cell = floor(pc);
  float h = 0.0f;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      float2 c = cell + float2(float(i), float(j));
      float2 rnd = ls_hash22(c);
      float2 center = (c + rnd) * cellSize;
      float r = cellSize * radiusScale * (0.4f + 0.6f * ls_hash21(c + 91.7f));
      float dist = length(p - center);
      float t = dist / max(r, 1e-4f);
      float bowl = smoothstep(1.0f, 0.0f, t);
      float depression = -depth * bowl * bowl;
      float rim = rimHeight * exp(-pow((t - 1.05f) * 6.0f, 2.0f));
      h += depression + rim;
    }
  }
  return h;
}

// Cheap version used for the inner ray-march loop: only the two largest crater
// layers plus a low-octave rolling-ground fbm. This is called up to ~100 times
// per pixel, so keeping it lightweight matters far more than at the (single)
// final hit point.
static float ls_terrainHeightCoarse(float2 p) {
  float h = 0.0f;
  h += ls_craterLayer(p, 9.0f, 0.42f, 0.95f, 0.16f);
  h += ls_craterLayer(p, 3.1f, 0.38f, 0.32f, 0.07f);
  h += (ls_fbm2(p * 0.15f, 2) - 0.5f) * 0.6f;   // gentle rolling ground
  return h;
}

// Samples the pre-baked coarse heightmap texture (see lunarBakeHeightKernel)
// instead of evaluating the procedural noise directly — used inside the
// march loop where it's called up to ~96 times per pixel.
static float ls_sampleHeightCoarse(
  float2 p, float2 center, float halfRange, texture2d<float> heightMap, sampler smp)
{
  float2 uv = (p - center) / (2.0f * halfRange) + 0.5f;
  return heightMap.sample(smp, uv).r;
}

// ─── Compute pre-pass: bake the coarse height field into a texture ───────────
// The terrain is static (time-invariant), so re-baking is only needed when the
// player has walked far enough (via "箱内移动" pattern navigation) that the
// previously-baked tile no longer covers the visible area — see
// LunarSurfaceRenderer.encodeComputePrepass. Baking ~512x512 texels in a
// compute kernel is vastly cheaper than evaluating the same noise millions of
// times (once per march step per screen pixel, per eye) in the fragment shader.
kernel void lunarBakeHeightKernel(
  texture2d<float, access::write> heightMap [[texture(0)]],
  constant LunarSurfaceUniforms &uniforms [[buffer(0)]],
  uint2 gid [[thread_position_in_grid]])
{
  uint w = heightMap.get_width();
  uint h = heightMap.get_height();
  if (gid.x >= w || gid.y >= h) return;

  float2 uv = (float2(gid) + 0.5f) / float2(float(w), float(h));
  float2 center = uniforms.heightMapParams.xy;
  float halfRange = uniforms.heightMapParams.z;
  float2 p = center + (uv - 0.5f) * (2.0f * halfRange);

  float height = ls_terrainHeightCoarse(p);
  heightMap.write(float4(height, 0.0f, 0.0f, 0.0f), gid);
}

// Cheaper normal for the near-field hit point, built from finite differences
// of the pre-baked COARSE heightmap texture (4 cheap texture fetches) instead
// of the procedural full-detail ls_terrainHeight (4 calls, each running two
// crater-layer 3x3-cell searches + an fbm) — the fine pockmark/regolith layer
// was already de-emphasized twice per earlier user feedback, so folding it
// into the per-pixel normal too wasn't buying much visible detail for its
// cost. Large/medium crater shape (from ls_terrainHeightCoarse, baked into
// the texture) still fully drives the normal/shading.
static float3 ls_terrainNormalCoarse(
  float3 p, float eps, float2 hmCenter, float hmHalfRange,
  texture2d<float> heightMap, sampler smp)
{
  float2 e = float2(eps, 0.0f);
  float hL = ls_sampleHeightCoarse(p.xz - e.xy, hmCenter, hmHalfRange, heightMap, smp);
  float hR = ls_sampleHeightCoarse(p.xz + e.xy, hmCenter, hmHalfRange, heightMap, smp);
  float hD = ls_sampleHeightCoarse(p.xz - e.yx, hmCenter, hmHalfRange, heightMap, smp);
  float hU = ls_sampleHeightCoarse(p.xz + e.yx, hmCenter, hmHalfRange, heightMap, smp);
  return normalize(float3(hL - hR, 2.0f * eps, hD - hU));
}

// Height-field sphere-tracing (approximate SDF, same convention as other
// terrain demos in this project: step by the height delta, clamped). The
// height lookup samples a pre-baked heightmap texture (see
// lunarBakeHeightKernel below) instead of calling the procedural noise
// directly — that texture fetch is dramatically cheaper than re-evaluating
// several octaves of crater/fbm noise on every one of the ~96 march steps.
static float ls_marchTerrain(
  float3 ro, float3 rd, float maxDist,
  float2 hmCenter, float hmHalfRange,
  texture2d<float> heightMap, sampler smp,
  thread float3 &hitPos)
{
  float t = 0.1f;
  for (int i = 0; i < 96; i++) {
    float3 p = ro + rd * t;
    float h = p.y - ls_sampleHeightCoarse(p.xz, hmCenter, hmHalfRange, heightMap, smp);
    if (h < 0.0025f * max(t, 1.0f)) {
      hitPos = p;
      return t;
    }
    // Allow a larger minimum stride farther from the camera (cheaper, coarser
    // marching where detail is fogged out anyway) so grazing/horizon rays
    // don't burn the full iteration budget.
    float minStep = 0.02f + t * 0.015f;
    t += clamp(h, minStep, 3.0f) * 0.7f;
    if (t > maxDist) break;
  }
  return -1.0f;
}

// ─── Space elevator tether (simple analytic geometry, ⟨流浪地球⟩-style) ────────
// A capped vertical cylinder test, reused for the tower's base drum, its long
// thin shaft, and the climber pod — all as plain closed-form ray/quadric
// intersections (no marching needed), so this costs only a handful of extra
// ALU ops per pixel regardless of how enormous `yMax` is.
static float ls_intersectCappedCylinder(
  float3 ro, float3 rd, float2 center, float radius, float yMin, float yMax)
{
  float2 oc = ro.xz - center;
  float a = dot(rd.xz, rd.xz);
  float t0 = -1.0f;
  float t1 = -1.0f;
  bool hasSide = false;
  if (a > 1e-8f) {
    float b = dot(oc, rd.xz);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - a * c;
    if (disc >= 0.0f) {
      float s = sqrt(disc);
      t0 = (-b - s) / a;
      t1 = (-b + s) / a;
      hasSide = true;
    }
  } else if (dot(oc, oc) <= radius * radius) {
    // Ray runs parallel to the tower's axis and is within its radius — the
    // side surface bounds don't constrain t at all; only the y-slab below does.
    t0 = -1e9f;
    t1 = 1e9f;
    hasSide = true;
  }
  if (!hasSide) return -1.0f;

  float tySlabMin;
  float tySlabMax;
  if (abs(rd.y) > 1e-8f) {
    float tA = (yMin - ro.y) / rd.y;
    float tB = (yMax - ro.y) / rd.y;
    tySlabMin = min(tA, tB);
    tySlabMax = max(tA, tB);
  } else if (ro.y >= yMin && ro.y <= yMax) {
    tySlabMin = -1e9f;
    tySlabMax = 1e9f;
  } else {
    return -1.0f;
  }

  float tMin = max(t0, tySlabMin);
  float tMax = min(t1, tySlabMax);
  if (tMin > tMax || tMax < 0.0f) return -1.0f;
  return tMin >= 0.0f ? tMin : -1.0f;
}

// Like ls_intersectCappedCylinder, but hollow — only counts a hit if the
// resolved point's XZ distance from `center` is also >= innerRadius. Used for
// the space elevator's reinforcing collars and climber platform: those need
// an open middle so looking up through the leg ring (e.g. to see the climber
// or the sky beyond) isn't blocked by a solid "cap" disc filling the whole
// cross-section down to the axis.
static float ls_intersectAnnulusBand(
  float3 ro, float3 rd, float2 center, float innerRadius, float outerRadius,
  float yMin, float yMax)
{
  float tOuter = ls_intersectCappedCylinder(ro, rd, center, outerRadius, yMin, yMax);
  if (tOuter < 0.0f) return -1.0f;
  float3 p = ro + rd * tOuter;
  float r = length(p.xz - center);
  return (r >= innerRadius) ? tOuter : -1.0f;
}

// ─── Sky: star field + Milky Way texture + Sun ───────────────────────────────
static float ls_starField(float3 rd) {
  float3 p = rd * 400.0f;
  float3 id = floor(p);
  float3 f = fract(p) - 0.5f;
  float h = ls_hash21(id.xy * 7.0f + id.z * 13.0f);
  float star = smoothstep(0.5f, 0.0f, length(f)) * step(0.9965f, h);
  float twinkle = 0.6f + 0.4f * sin(h * 6.2831f * 40.0f);
  return star * twinkle;
}

static float2 ls_equirectUV(float3 rd) {
  float u = atan2(rd.z, rd.x) / (2.0f * M_PI_F) + 0.5f;
  float v = acos(clamp(rd.y, -1.0f, 1.0f)) / M_PI_F;
  return float2(u, v);
}

static float3 ls_sky(float3 rd, float3 sunDir, texture2d<float> milkyway, sampler smp) {
  float2 uv = ls_equirectUV(rd);
  // Texture is loaded with sRGB decode (see LunarSurfaceRenderer), so this is
  // already linear light: true black sky stays at 0, only the galaxy band /
  // nebulosity is bright. A small boost keeps the band reading clearly without
  // lifting the black point. The source map isn't a true black-background
  // radiance image (it's a scientific brightness/colour map with a fairly
  // bright diffuse floor even in "empty" sky), so on top of the sRGB decode
  // we apply an extra contrast curve: subtract a small black-point, then
  // raise to a power >1. This crushes the diffuse floor toward true black
  // while keeping the actual galaxy band / nebulosity relatively bright.
  float3 galaxyRaw = milkyway.sample(smp, uv).rgb;
  float3 galaxy = pow(max(galaxyRaw - 0.045f, 0.0f), 1.8f) * 3.2f;

  float darkness = 1.0f - saturate(dot(galaxy, float3(0.333f)) * 5.0f);
  float stars = ls_starField(rd) * darkness;
  float3 col = galaxy + float3(stars);

  float sunDot = dot(rd, sunDir);
  float sunDisc = smoothstep(LS_SUN_DISC_INNER, LS_SUN_DISC_OUTER, sunDot);
  // Airless body: the sun should read as a crisp, hard-edged disc with no
  // halo at all (that hazy glow comes from atmospheric scattering, which
  // the Moon doesn't have), per user feedback ("太阳周围的光晕直接去掉").
  col += float3(1.0f, 0.95f, 0.85f) * sunDisc * LS_SUN_DISC_BRIGHTNESS;

  return col;
}

// Earth rendered as a billboard disc in the direction `earthDir`, textured with
// the source photo (already framed as a lit disc against black space).
static float3 ls_earth(
  float3 rd, float3 earthDir, float angularRadius,
  texture2d<float> earthTex, sampler smp, thread float &mask)
{
  mask = 0.0f;
  float cosA = dot(rd, earthDir);
  if (cosA <= 0.05f) return float3(0.0f);

  float3 right = normalize(cross(float3(0.0f, 1.0f, 0.0f), earthDir));
  float3 up = cross(earthDir, right);
  float3 proj = rd - earthDir * cosA;
  float2 local = float2(dot(proj, right), dot(proj, up)) / cosA;
  float2 uvOffset = local / angularRadius;
  float r = length(uvOffset);
  if (r > 1.08f) return float3(0.0f);

  float2 uv = uvOffset * 0.5f + 0.5f;
  float3 col = earthTex.sample(smp, uv).rgb;

  // No atmosphere glow at all, per user feedback ("地球周围的光晕直接去掉") —
  // just the lit disc itself, cut off cleanly at its edge.
  mask = smoothstep(LS_EARTH_DISC_EDGE_OUTER, LS_EARTH_DISC_EDGE_INNER, r);
  return col;
}

// ─── Fragment shader ──────────────────────────────────────────────────────────
fragment float4 lunarSurfaceFragment(
  LSVertexOut in [[stage_in]],
  constant LunarSurfaceUniforms &uniforms [[buffer(0)]],
  constant float4x4 *v2wMats [[buffer(1)]],
  texture2d<float> earthTex [[texture(0)]],
  texture2d<float> milkywayTex [[texture(1)]],
  texture2d<float> heightMapTex [[texture(2)]])
{
  constexpr sampler wrapSampler(address::repeat, filter::linear, mip_filter::linear);
  constexpr sampler clampSampler(address::clamp_to_edge, filter::linear);

  uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
  float4x4 v2w = v2wMats[vi];
  float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
  float3 camForward = -float3(v2w[2].x, v2w[2].y, v2w[2].z);

  float3 baseRd = normalize(in.worldPos - camWorld);

  // This box-surface point is behind the camera relative to its forward
  // direction. With a convex inward-facing box the camera normally only ever
  // rasterizes the one exit face it's actually looking at (any face behind
  // it is already frustum-clipped by the GPU before the fragment shader
  // runs), but this is a cheap explicit guard so no fragment ever pays for
  // the terrain march / sky shading below unless it's genuinely in front of
  // the viewer.
  if (dot(baseRd, camForward) <= 0.0f) {
    return float4(0.0f, 0.0f, 0.0f, 1.0f);
  }

  // Normal-mode flight (rig navigation) moves the real/game-world camera away
  // from the box, which is fixed in place — folding that offset into the
  // march origin makes the terrain/sky actually recede as you back away
  // instead of looking glued to the viewer (the box's own screen-space size
  // already shrinks with distance, but the ray-marched content needs this
  // explicit shift to match, since it isn't a static texture).
  float3 eyeOffset = camWorld - uniforms.objectCenter.xyz;
  float3 baseRo = float3(0.0f, 1.7f, 0.0f) + eyeOffset;  // eye height above the crater field datum

  // Pattern-space navigation: shifts/rotates the virtual viewpoint so the gamepad
  // "箱内移动" mode lets the player walk across the terrain and look around freely.
  float3 ro = (uniforms.patternTransform * float4(baseRo, 1.0f)).xyz;
  float3 rd = normalize(float3(uniforms.patternTransform * float4(baseRd, 0.0f)));

  float3 sunDir = normalize(float3(0.5f, 0.30f, -0.35f));
  float3 earthDir = normalize(float3(-0.35f, 0.55f, -0.6f));

  float2 hmCenter = uniforms.heightMapParams.xy;
  float hmHalfRange = uniforms.heightMapParams.z;

  // Prevent the camera from ever rendering as if it were underground (e.g.
  // after pattern-nav movement carries it into/through a crater wall) —
  // clamp the ray origin to stay at least LS_MIN_HEIGHT_ABOVE_GROUND above
  // the actual terrain height at its own XZ position.
  float groundYAtRo = ls_sampleHeightCoarse(ro.xz, hmCenter, hmHalfRange, heightMapTex, clampSampler);
  ro.y = max(ro.y, groundYAtRo + LS_MIN_HEIGHT_ABOVE_GROUND);

  // ── Space elevator tether ──────────────────────────────────────────────────
  // Fixed position in moon-local space, a modest distance from spawn so the
  // player can see the whole base and look up along the shaft. Six-legged
  // ring lattice (hub + six pillars + periodic reinforcing collars + a small
  // climber platform riding INSIDE the legs + a large counterweight/cabin
  // capping the very top), still built entirely from closed-form ray/quadric
  // intersections — no marching — so it costs only a handful of extra ALU
  // ops per pixel regardless of shaft height.
  float2 towerXZ = LS_TOWER_XZ;
  float towerGroundY = ls_sampleHeightCoarse(towerXZ, hmCenter, hmHalfRange, heightMapTex, clampSampler);
  float towerBaseH = LS_TOWER_BASE_H;
  float towerShaftH = LS_TOWER_SHAFT_H;
  float towerShaftTopY = towerGroundY + towerBaseH + towerShaftH;
  float pillarRingRadius = LS_PILLAR_RING_RADIUS;
  float pillarRadius = LS_PILLAR_RADIUS;
  // Reinforcing rings: radial + vertical thickness both pushed well past
  // double their previous size and past the pillar's own radius (10m), so
  // the collar reads as clearly THICKER than the legs it wraps, per user
  // request ("厚度也增加一倍以上, 而且要求比钢索(柱子)粗").
  float ringInnerRadius = pillarRingRadius - LS_RING_RADIAL_HALF_WIDTH;
  float ringOuterRadius = pillarRingRadius + LS_RING_RADIAL_HALF_WIDTH;  // collar protrudes OUTSIDE the legs

  // Central foundation hub / bottom cabin: drawn bigger than the whole
  // leg+ring footprint (not just the legs), per user request.
  float hubRadius = ringOuterRadius + LS_TOWER_HUB_EXTRA_RADIUS;
  float tTower = ls_intersectCappedCylinder(
    ro, rd, towerXZ, hubRadius, towerGroundY, towerGroundY + towerBaseH);
  int towerPart = 0;  // 0=hub, 1=leg, 2=reinforcing ring, 3=climber platform, 4=top counterweight
  float2 towerHitCenter = towerXZ;  // axis to use for this hit's surface normal

  // Six pillar legs arranged in a ring around the hub, like bamboo stalks.
  for (int i = 0; i < 6; i++) {
    float ang = float(i) * (6.28318530718f / 6.0f);
    float2 legXZ = towerXZ + pillarRingRadius * float2(cos(ang), sin(ang));
    float tLeg = ls_intersectCappedCylinder(
      ro, rd, legXZ, pillarRadius, towerGroundY, towerShaftTopY);
    if (tLeg > 0.0f && (tTower < 0.0f || tLeg < tTower)) {
      tTower = tLeg;
      towerPart = 1;
      towerHitCenter = legXZ;
    }
  }

  // Periodic reinforcing rings binding the six legs together — a true hollow
  // collar (innerRadius < legRadius < outerRadius) so it wraps around the
  // OUTSIDE of the legs without filling in the open shaft interior, unlike a
  // solid drum which would opaquely block the view straight up/down through
  // the middle (including the climber platform below).
  //
  // The full shaft is ~25km tall, so at this spacing that's ~250 rings —
  // too many to unconditionally test on every pixel. Instead, first solve
  // for where (if at all) this ray's XZ path stays within the ring's outer
  // radius (same quadratic as the capped-cylinder test, just unbounded in Y),
  // then only loop over the small subrange of ring indices whose Y-band could
  // actually fall within that span. Most pixels (sky, distant ground) miss
  // the radius entirely and skip the loop altogether; only pixels whose ray
  // actually grazes the tower's footprint pay for (usually a handful of)
  // ring tests.
  float ringSpacing = LS_RING_SPACING;
  // Vertical thickness: previously ±12 (24m total) read as "paper thin"
  // against this structure's mega-scale (100m ring radius, 25km shaft,
  // 1000m ring spacing) — bumped up to a real floor-slab-like thickness
  // per user request ("圆环应该有高度, 至少一层楼高").
  float ringHalfThick = LS_RING_HALF_THICK;
  int numRings = int(towerShaftH / ringSpacing);
  float ringBaseY = towerGroundY + towerBaseH;
  int ringLo = 0;
  int ringHi = -1;  // empty range unless the ray actually crosses the footprint
  {
    float2 oc = ro.xz - towerXZ;
    float aXZ = dot(rd.xz, rd.xz);
    if (aXZ > 1e-8f) {
      float b = dot(oc, rd.xz);
      float c = dot(oc, oc) - ringOuterRadius * ringOuterRadius;
      float disc = b * b - aXZ * c;
      if (disc >= 0.0f) {
        float s = sqrt(disc);
        float tNear = max((-b - s) / aXZ, 0.0f);
        float tFar = (-b + s) / aXZ;
        if (tFar >= 0.0f) {
          float yA = ro.y + rd.y * tNear;
          float yB = ro.y + rd.y * tFar;
          float yLo = min(yA, yB);
          float yHi = max(yA, yB);
          ringLo = max(0, int(floor((yLo - ringBaseY) / ringSpacing)) - 1);
          ringHi = min(numRings - 1, int(ceil((yHi - ringBaseY) / ringSpacing)) + 1);
        }
      }
    } else if (dot(oc, oc) <= ringOuterRadius * ringOuterRadius) {
      // Ray runs parallel to the shaft's axis, inside the ring radius — it
      // can graze the whole vertical run, so every ring is in play.
      ringLo = 0;
      ringHi = numRings - 1;
    }
  }
  for (int r = ringLo; r <= ringHi; r++) {
    float ringY = ringBaseY + float(r + 1) * ringSpacing;
    float tRing = ls_intersectAnnulusBand(
      ro, rd, towerXZ, ringInnerRadius, ringOuterRadius, ringY - ringHalfThick, ringY + ringHalfThick);
    if (tRing > 0.0f && (tTower < 0.0f || tRing < tTower)) {
      tTower = tRing;
      towerPart = 2;
      towerHitCenter = towerXZ;
    }
  }

  // Climber platform: a real disc spanning all the way out to the legs (same
  // outer radius as the reinforcing rings) so it visibly grips/rides on the
  // six legs rather than floating disconnected in the open middle — the legs
  // are what physically carry it, not empty space. Being a single instance
  // (not a repeating structure like the rings), a solid disc here is fine:
  // it only ever occludes the shaft at its own current height, just like a
  // real elevator car would. 8m tall per user request, with a diagonal
  // yellow/black hazard-stripe pattern (wrapped around its own axis + height)
  // so it reads as a distinct moving car rather than just another ring.
  float climberRadius = ringOuterRadius;
  float climberHalfThick = LS_CLIMBER_HALF_THICK;
  float climberCycle = fmod(uniforms.time * LS_CLIMBER_CYCLE_SPEED, 2.0f);
  float climberPhase = climberCycle < 1.0f ? climberCycle : 2.0f - climberCycle;
  float climberY = towerGroundY + towerBaseH + climberPhase * towerShaftH;
  float tPlatform = ls_intersectCappedCylinder(
    ro, rd, towerXZ, climberRadius, climberY - climberHalfThick, climberY + climberHalfThick);
  if (tPlatform > 0.0f && (tTower < 0.0f || tPlatform < tTower)) {
    tTower = tPlatform;
    towerPart = 3;
    towerHitCenter = towerXZ;
  }

  // Top counterweight/active cabin module: drawn bigger than the whole
  // leg+ring footprint (matching the enlarged bottom cabin), per user request.
  float topCapRadius = ringOuterRadius + LS_TOWER_TOPCAP_EXTRA_RADIUS;
  float topCapHalfThick = LS_TOWER_TOPCAP_HALF_THICK;
  float tTopCap = ls_intersectCappedCylinder(
    ro, rd, towerXZ, topCapRadius, towerShaftTopY - topCapHalfThick, towerShaftTopY + topCapHalfThick);
  if (tTopCap > 0.0f && (tTower < 0.0f || tTopCap < tTower)) {
    tTower = tTopCap;
    towerPart = 4;
    towerHitCenter = towerXZ;
  }

  float3 hitPos;
  float maxDist = LS_MARCH_MAX_DIST;
  float t = (rd.y < 0.25f)
    ? ls_marchTerrain(ro, rd, maxDist, hmCenter, hmHalfRange, heightMapTex, clampSampler, hitPos)
    : -1.0f;

  float3 col;
  bool towerHit = tTower > 0.0f && (t < 0.0f || tTower < t);
  if (towerHit) {
    float3 p = ro + rd * tTower;
    float2 radial = p.xz - towerHitCenter;
    float3 n = normalize(float3(radial.x, 0.0f, radial.y));

    // Darker base metal than before, per user feedback — and a much smaller
    // flat ambient floor below so the sun-facing side reads clearly bright
    // and the far side reads clearly dark, instead of everything looking
    // similarly lit.
    float band = step(0.5f, fract(p.y * (1.0f / 40.0f)));
    float3 baseColor = mix(LS_TOWER_BAND_COLOR_A, LS_TOWER_BAND_COLOR_B, band);
    if (towerPart == 1) {
      // Leg pillar surface: a slow spiraling twisted-steel-cable look —
      // several strands wrapping around the pillar as it climbs, rather
      // than a plain smooth tube (now that each leg is a hefty 10m-radius
      // column, a flat surface would read as far too plain/featureless). A
      // large spiral pitch keeps the twist reading as slow, not a busy
      // corkscrew.
      float angRad = atan2(radial.y, radial.x);
      float wrap = angRad * (LS_CABLE_STRAND_COUNT / 6.28318530718f) - p.y * (LS_CABLE_STRAND_COUNT / LS_CABLE_SPIRAL_PITCH);
      float d = fract(wrap) - 0.5f;  // -0.5..0.5, 0 at each strand's center
      float bump = cos(d * 3.14159265f);  // rounded highlight down each strand
      float groove = smoothstep(0.42f, 0.5f, abs(d));  // thin dark seam between strands
      baseColor *= (0.82f + 0.18f * bump);
      baseColor = mix(baseColor, baseColor * 0.3f, groove);
    } else if (towerPart == 2) {
      baseColor = LS_RING_COLOR;
    } else if (towerPart == 3) {
      // Climber car: mostly-gray fuselage shaded by its own relative height
      // (brighter equator band tapering darker toward the top/bottom caps)
      // instead of the previous diagonal hazard stripes — those wrapped
      // around the axis and, combined with the side surface's radial
      // per-pixel normal, ended up reading as a "starburst" pattern rather
      // than a distinct car. A ring of small round "airplane window" ports
      // runs around the middle band for detail while keeping the overall
      // look gray.
      float localY = p.y - climberY;
      float hFrac = clamp(localY / climberHalfThick, -1.0f, 1.0f);
      float3 climberGray = mix(LS_CLIMBER_GRAY_BRIGHT, LS_CLIMBER_GRAY_DARK, abs(hFrac));

      float angRad2 = atan2(radial.y, radial.x);
      float cellAngle = fract(angRad2 * (LS_CLIMBER_WINDOW_COUNT / 6.28318530718f) + 0.5f) - 0.5f;
      float arcDist = cellAngle * (6.28318530718f * climberRadius / LS_CLIMBER_WINDOW_COUNT);
      float windowDist = length(float2(arcDist, localY));
      float windowMask = smoothstep(0.85f, 0.55f, windowDist) * step(abs(hFrac), 0.55f);
      baseColor = mix(climberGray, LS_CLIMBER_WINDOW_COLOR, windowMask);
    } else if (towerPart == 4) {
      baseColor = LS_TOPCAP_COLOR;
    }

    float diff = max(dot(n, sunDir), 0.0f);
    float3 lit = baseColor * (diff * 1.1f + 0.04f);
    if (towerPart == 3) {
      lit += baseColor * 0.35f;  // platform reads as lit/powered, not just sun-lit
    }

    // Blinking aircraft-style warning beacon on the top counterweight module
    // (the tallest, most distant point on the whole structure).
    float distToTop = abs(p.y - towerShaftTopY);
    float beacon = (towerPart == 4)
      ? smoothstep(LS_BEACON_RANGE, 0.0f, distToTop) * (0.6f + 0.4f * sin(uniforms.time * LS_BEACON_BLINK_SPEED))
      : 0.0f;
    lit += LS_BEACON_COLOR * beacon;

    // The Moon has no atmosphere, so unlike the terrain's fog (an artistic
    // depth cue) the tether should stay crisp for kilometers — only a very
    // gentle falloff so it doesn't look like a flat cardboard cutout at
    // extreme distance up the shaft.
    float fog = exp2(-tTower * 0.0008f);
    float3 skyTint = ls_sky(rd, sunDir, milkywayTex, wrapSampler) * 0.15f;
    col = mix(skyTint, lit, fog);
  } else if (t > 0.0f) {
    float3 n = ls_terrainNormalCoarse(hitPos, 0.35f, hmCenter, hmHalfRange, heightMapTex, clampSampler);
    float diff = max(dot(n, sunDir), 0.0f);
    float earthLight = max(dot(n, earthDir), 0.0f) * 0.05f;
    float opposition = pow(max(dot(-rd, sunDir), 0.0f), 30.0f) * 0.25f;  // lunar heiligenschein

    // Slightly darker than before + one fewer fbm octave (per user feedback:
    // reduce near-field compute further, and the whole surface should read a
    // touch darker overall).
    float3 albedo = mix(
      LS_NEAR_ALBEDO_A, LS_NEAR_ALBEDO_B,
      ls_fbm2(hitPos.xz * LS_NEAR_FBM_SCALE, LS_NEAR_FBM_OCTAVES));

    float3 lit = albedo * (diff * float3(1.15f, 1.08f, 0.95f) + earthLight * float3(0.5f, 0.6f, 0.8f) + 0.01f);
    lit += albedo * opposition;

    float fog = exp2(-t * 0.012f);
    float3 skyAtHorizon = ls_sky(rd, sunDir, milkywayTex, wrapSampler) * 0.15f;
    col = mix(skyAtHorizon, lit, fog);
  } else {
    // Far-field fallback: the height-field march above only searches out to
    // `maxDist`, so past that range (or at grazing angles where it burns its
    // whole step budget without converging) it can fail even though we're
    // still clearly looking at the ground — without this, the ground used to
    // vanish into full sky a short distance from the player. Approximate the
    // far terrain as a flat plane at the local ground datum (a single sample
    // from the same baked heightmap, taken under the camera) so the ground
    // always extends to the horizon at the correct apparent scale.
    float tFar = -1.0f;
    if (rd.y < -0.0005f) {
      float groundDatum = ls_sampleHeightCoarse(ro.xz, hmCenter, hmHalfRange, heightMapTex, clampSampler);
      float candidate = (groundDatum - ro.y) / rd.y;
      if (candidate > 0.0f) {
        tFar = min(candidate, LS_FAR_MAX_DIST);
      }
    }

    if (tFar > 0.0f) {
      float3 farPos = ro + rd * tFar;
      // A bit more variation than before (extra octave + a second, smaller
      // crater layer) since the far ground was reading as almost solid
      // color — still just brightness modulation on a flat plane, not real
      // displaced geometry, so this stays cheap (two 3x3-cell loops total).
      float mottle = ls_fbm2(farPos.xz * LS_FAR_MOTTLE_SCALE, LS_FAR_MOTTLE_OCTAVES);
      float craterShadeBig = ls_craterLayer(farPos.xz, 55.0f, 0.42f, 1.0f, 0.3f);
      float craterShadeMed = ls_craterLayer(farPos.xz, 20.0f, 0.4f, 0.5f, 0.15f);
      float shade = clamp(
        1.0f + (craterShadeBig + craterShadeMed) * LS_FAR_SHADE_MULT,
        LS_FAR_SHADE_CLAMP_MIN, LS_FAR_SHADE_CLAMP_MAX);

      float3 albedo = mix(LS_NEAR_ALBEDO_A, LS_NEAR_ALBEDO_B, mottle);
      albedo *= shade;
      float diff = max(dot(float3(0.0f, 1.0f, 0.0f), sunDir), 0.0f);
      float3 lit = albedo * (diff * float3(1.15f, 1.08f, 0.95f) + 0.01f);

      // The Moon has no atmosphere, so — unlike a hazy Earth horizon — the
      // ground should stay fully OPAQUE all the way out instead of fading
      // into background sky/stars by distance. No fog blend here; a sphere's
      // surface doesn't turn see-through just because it's far away.
      col = lit;
    } else {
      col = ls_sky(rd, sunDir, milkywayTex, wrapSampler);
      float earthMask;
      float3 earthCol = ls_earth(rd, earthDir, LS_EARTH_ANGULAR_RADIUS, earthTex, clampSampler, earthMask);
      col = mix(col, earthCol, earthMask);
    }
  }

  col = clamp(col, 0.0f, 6.0f);
  col = col / (1.0f + col);
  col = sqrt(col);
  return float4(col, 1.0f);
}
