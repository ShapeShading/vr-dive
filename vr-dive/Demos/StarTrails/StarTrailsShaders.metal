// StarTrailsShaders.metal
//
// Star-trail long-exposure effect: particles orbit the Z-axis inside a 2 m
// cube, each leaving a short curved arc behind them.
//
// Performance design:
//  - Integer Murmur3 hash (no sin() in hash)
//  - Ray pre-rotated into star-field frame once — only 1 cos/sin per pixel
//  - Very tight distance cutoff keeps the halo narrow and the lines crisp
//  - Each arc is approximated by 3 short line segments; this preserves the
//    circular silhouette without going back to volumetric ray marching
//  - No atan2 or ray-march loop on the hot path

#include <metal_stdlib>
using namespace metal;

// ─── Structs ─────────────────────────────────────────────────────────────────

// Layout must match StarTrailsUniforms in StarTrailsTypes.swift.
struct StarTrailsUniforms {
    float  time;
    uint   viewCount;
    float  pad0;
    float  pad1;
    float4 objectCenter;  // xyz = world-space box centre
};

struct STVertex {
    float3 position;
    float3 normal;
};

struct STVertexOut {
    float4 clipPos   [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ─── Vertex shader ────────────────────────────────────────────────────────────
vertex STVertexOut starTrailsVertex(
    ushort                       amplificationID [[amplification_id]],
    const device STVertex       *vertices        [[buffer(0)]],
    constant StarTrailsUniforms &uniforms        [[buffer(1)]],
    constant float4x4           *vpMatrices      [[buffer(2)]],
    uint                         vertexID        [[vertex_id]])
{
    STVertex vtx   = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position + uniforms.objectCenter.xyz;

    STVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Fast integer hash (Murmur3 finalizer), result in [0, 1).
// No transcendental functions — replacing the sin()-based hash cuts ~20 k
// sin() calls per pixel at the original orbit/step counts.
static float st_h(uint n) {
    n ^= (n >> 16u);
    n *= 0x85ebca6bu;
    n ^= (n >> 13u);
    n *= 0xc2b2ae35u;
    n ^= (n >> 16u);
    return float(n) * (1.0f / 4294967296.0f);
}

// Ray vs axis-aligned box.  Sets tNear/tFar and returns true on hit.
// IEEE infinity handles zero ray-direction components correctly.
static bool st_boxHit(float3 ro, float3 rd, float3 halfExtents,
                      thread float &tNear, thread float &tFar)
{
    float3 invRd = 1.0f / rd;          // ±inf for zero components is intentional
    float3 t1 = (-halfExtents - ro) * invRd;
    float3 t2 = ( halfExtents - ro) * invRd;
    float3 tMin = min(t1, t2);
    float3 tMax = max(t1, t2);
    tNear = max(max(tMin.x, tMin.y), tMin.z);
    tFar  = min(min(tMax.x, tMax.y), tMax.z);
    if (tNear > tFar || tFar < 0.0f) return false;
    tNear = max(tNear, 0.0f);
    return true;
}

// HSV → RGB.
static float3 st_hsv2rgb(float h, float s, float v) {
    float3 c = clamp(abs(fract(h + float3(1.0f, 2.0f/3.0f, 1.0f/3.0f)) * 6.0f - 3.0f) - 1.0f,
                     0.0f, 1.0f);
    return v * mix(float3(1.0f), c, s);
}

// Closest distance between a ray ro + rd * t (t >= 0) and a segment [a, b].
// Returns squared distance and outputs the closest ray/segment parameters.
static float st_raySegmentDistanceSq(
    float3 ro, float3 rd, float3 a, float3 b,
    thread float &rayT, thread float &segU)
{
    float3 ab = b - a;
    float abLen2 = max(dot(ab, ab), 1e-6f);
    float3 ao = ro - a;
    float rdAb = dot(rd, ab);
    float rdAo = dot(rd, ao);
    float abAo = dot(ab, ao);
    float denom = abLen2 - rdAb * rdAb;

    float u;
    if (denom > 1e-6f) {
        u = clamp((abAo - rdAb * rdAo) / denom, 0.0f, 1.0f);
        rayT = rdAb * u - rdAo;
    } else {
        u = clamp(abAo / abLen2, 0.0f, 1.0f);
        rayT = dot(a + ab * u - ro, rd);
    }

    if (rayT < 0.0f) {
        rayT = 0.0f;
        u = clamp(abAo / abLen2, 0.0f, 1.0f);
    } else {
        float3 q = ro + rd * rayT;
        u = clamp(dot(q - a, ab) / abLen2, 0.0f, 1.0f);
        rayT = max(dot(a + ab * u - ro, rd), 0.0f);
    }

    segU = u;
    float3 diff = ro + rd * rayT - (a + ab * u);
    return dot(diff, diff);
}

// ─── Fragment shader ──────────────────────────────────────────────────────────
fragment float4 starTrailsFragment(
    STVertexOut                  in       [[stage_in]],
    constant StarTrailsUniforms &uniforms [[buffer(0)]],
    constant float4x4           *v2wMats  [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w  = v2wMats[vi];
    float3 cam    = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 center = uniforms.objectCenter.xyz;

    // Ray in box-local space (box half-extents = 1 m).
    float3 ro = cam - center;
    float3 rd = normalize(in.worldPos - cam);

    const float3 BOX_HALF = float3(1.0f);
    float tNear, tFar;
    if (!st_boxHit(ro, rd, BOX_HALF, tNear, tFar)) discard_fragment();

    const int   N_ORBITS = 28;
    const int   ARC_SEGMENTS = 3;
    const float D_CUTOFF = 0.016f;   // 1.6 cm line radius cutoff
    const float D_CUTOFF2 = D_CUTOFF * D_CUTOFF;
    const float TWO_PI   = 6.28318f;
    float3 accumColor = float3(0.0f);

    for (int i = 0; i < N_ORBITS; ++i) {
        uint base = uint(i) * 8u;
        float h1 = st_h(base + 0u);  // radius
        float h2 = st_h(base + 1u);  // z base
        float h3 = st_h(base + 2u);  // trail angle / colour split
        float h4 = st_h(base + 3u);  // angular velocity
        float h5 = st_h(base + 4u);  // initial phase
        float h6 = st_h(base + 5u);  // z wobble amplitude
        float h7 = st_h(base + 6u);  // z wobble frequency
        float h8 = st_h(base + 7u);  // colour / brightness variant

        float radius = 0.16f + 0.58f * h1;
        float zBase = -0.70f + 1.40f * h2;
        float omega = 0.22f + 0.48f * h4;
        float angle = omega * uniforms.time + h5 * TWO_PI;
        float zAmp = 0.02f + 0.09f * h6;
        float zFreq = 1.0f + floor(h7 * 2.99f);
        float wave = angle * zFreq + h8 * TWO_PI;
        float trailAngle = 0.22f + 0.24f * h3;
        float subAngle = trailAngle / float(ARC_SEGMENTS);

        float sA = sin(angle);
        float cA = cos(angle);
        float sW = sin(wave);
        float cW = cos(wave);
        float cStep = cos(subAngle);
        float sStep = sin(subAngle);
        float waveStep = subAngle * zFreq;
        float cWaveStep = cos(waveStep);
        float sWaveStep = sin(waveStep);

        float3 col;
        if (h3 < 0.52f) {
            col = st_hsv2rgb(0.58f + 0.11f * h1, 0.10f + 0.40f * h6, 1.0f);
        } else if (h3 < 0.84f) {
            col = st_hsv2rgb(0.08f + 0.10f * h8, 0.08f + 0.34f * h2, 1.0f);
        } else {
            float hue = (h8 < 0.5f) ? (0.48f + 0.06f * h1) : (0.76f + 0.05f * h1);
            col = st_hsv2rgb(hue, 0.55f + 0.35f * h6, 1.0f);
        }

        float brightness = 0.75f + 3.3f * h8 * h8;

        float currC = cA;
        float currS = sA;
        float currCW = cW;
        float currSW = sW;
        float3 p0 = float3(radius * currC, radius * currS, zBase + zAmp * currSW);

        float bestGlow = 0.0f;
        float bestRayT = tFar;
        float bestAlong = 1.0f;

        for (int seg = 0; seg < ARC_SEGMENTS; ++seg) {
            float nextC = currC * cStep + currS * sStep;
            float nextS = currS * cStep - currC * sStep;
            float nextCW = currCW * cWaveStep + currSW * sWaveStep;
            float nextSW = currSW * cWaveStep - currCW * sWaveStep;
            float3 p1 = float3(radius * nextC, radius * nextS, zBase + zAmp * nextSW);

            float rayT;
            float segU;
            float d2 = st_raySegmentDistanceSq(ro, rd, p0, p1, rayT, segU);
            if (rayT >= tNear && rayT <= tFar && d2 <= D_CUTOFF2) {
                float glow = exp(-d2 * 110000.0f) + exp(-d2 * 16000.0f) * 0.05f;
                if (glow > bestGlow) {
                    bestGlow = glow;
                    bestRayT = rayT;
                    bestAlong = (float(seg) + segU) / float(ARC_SEGMENTS);
                }
            }

            currC = nextC;
            currS = nextS;
            currCW = nextCW;
            currSW = nextSW;
            p0 = p1;
        }

        if (bestGlow == 0.0f) continue;

        float trailFade = 0.24f + 0.76f * pow(1.0f - bestAlong, 0.45f);
        float depthFade = exp(-(bestRayT - tNear) * 0.42f);
        accumColor += col * brightness * bestGlow * trailFade * depthFade;
    }

    float3 mapped = 1.0f - exp(-accumColor * 2.6f);
    mapped = max(mapped, float3(0.002f, 0.001f, 0.004f));
    return float4(mapped, 1.0f);
}
