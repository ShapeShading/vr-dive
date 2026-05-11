// ParticleRainShaders.metal
//
// Falling light-streak particles inside a 2 m cube.
// Each particle is a purely-vertical capsule (10–30 cm long) falling along -Y,
// wrapping back to the top when it exits the bottom.  No rotation.
//
// Performance design:
//  - Integer Murmur3 hash (no sin() in hash)
//  - Narrow horizontal cutoff keeps the streaks crisp instead of foggy
//  - Fewer particles reduce overdraw and keep the pattern readable
//  - N_STEPS=24, N_PARTS=42

#include <metal_stdlib>
using namespace metal;

// ─── Structs ─────────────────────────────────────────────────────────────────

// Layout must match ParticleRainUniforms in ParticleRainTypes.swift.
struct ParticleRainUniforms {
    float  time;
    uint   viewCount;
    float  pad0;
    float  pad1;
    float4 objectCenter;
};

struct PRVertex {
    float3 position;
    float3 normal;
};

struct PRVertexOut {
    float4 clipPos   [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ─── Vertex shader ────────────────────────────────────────────────────────────
vertex PRVertexOut particleRainVertex(
    ushort                        amplificationID [[amplification_id]],
    const device PRVertex        *vertices        [[buffer(0)]],
    constant ParticleRainUniforms &uniforms       [[buffer(1)]],
    constant float4x4            *vpMatrices      [[buffer(2)]],
    uint                          vertexID        [[vertex_id]])
{
    PRVertex vtx   = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position + uniforms.objectCenter.xyz;

    PRVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Integer hash (Murmur3 finalizer), result ∈ [0, 1).
static float pr_h(uint n) {
    n ^= (n >> 16u);
    n *= 0x85ebca6bu;
    n ^= (n >> 13u);
    n *= 0xc2b2ae35u;
    n ^= (n >> 16u);
    return float(n) * (1.0f / 4294967296.0f);
}

// Ray vs axis-aligned box.
static bool pr_boxHit(float3 ro, float3 rd, float3 halfExtents,
                      thread float &tNear, thread float &tFar)
{
    float3 invRd = 1.0f / rd;
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
static float3 pr_hsv2rgb(float h, float s, float v) {
    float3 c = clamp(abs(fract(h + float3(1.0f, 2.0f/3.0f, 1.0f/3.0f)) * 6.0f - 3.0f) - 1.0f,
                     0.0f, 1.0f);
    return v * mix(float3(1.0f), c, s);
}

// Closest distance between a ray ro + rd * t (t >= 0) and a segment [a, b].
// Returns squared distance and outputs the closest ray/segment parameters.
static float pr_raySegmentDistanceSq(
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
fragment float4 particleRainFragment(
    PRVertexOut                   in       [[stage_in]],
    constant ParticleRainUniforms &uniforms [[buffer(0)]],
    constant float4x4            *v2wMats  [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w  = v2wMats[vi];
    float3 cam    = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 center = uniforms.objectCenter.xyz;

    float3 ro = cam - center;
    float3 rd = normalize(in.worldPos - cam);

    float tNear, tFar;
    if (!pr_boxHit(ro, rd, float3(1.0f), tNear, tFar)) discard_fragment();

    const int   N_PARTS   = 48;
    const float LINE_CUTOFF = 0.016f;
    const float LINE_CUTOFF2 = LINE_CUTOFF * LINE_CUTOFF;
    const float BOX_H = 2.0f;

    float3 accumColor = float3(0.0f);

    for (int i = 0; i < N_PARTS; ++i) {
        uint base = uint(i) * 6u;
        float h1 = pr_h(base + 0u);  // x position
        float h2 = pr_h(base + 1u);  // z position
        float h3 = pr_h(base + 2u);  // fall speed
        float h4 = pr_h(base + 3u);  // trail length
        float h5 = pr_h(base + 4u);  // initial phase (start height)
        float h6 = pr_h(base + 5u);  // colour category / brightness

        float sx = -0.84f + 1.68f * h1;
        float sz = -0.84f + 1.68f * h2;

        // Cheap XY rejection before any segment math.
        float dxh = ro.x + rd.x * tNear - sx;
        float dzh = ro.z + rd.z * tNear - sz;
        float dh2Hint = dxh * dxh + dzh * dzh;
        if (dh2Hint > 0.90f) continue;

        float speed = 0.28f + 0.46f * h3;        // 0.28 – 0.74 m/s
        float trailLen = 0.10f + 0.20f * h4;     // 10 – 30 cm
        float yHead = 1.0f - fmod(h5 * BOX_H + speed * uniforms.time, BOX_H);

        float3 col;
        if (h6 < 0.30f) {
            col = pr_hsv2rgb(0.59f + 0.05f * h3, 0.08f + 0.18f * h4, 1.0f);
        } else if (h6 < 0.58f) {
            col = pr_hsv2rgb(0.50f + 0.07f * h1, 0.32f + 0.28f * h2, 1.0f);
        } else if (h6 < 0.82f) {
            col = pr_hsv2rgb(0.73f + 0.06f * h3, 0.28f + 0.30f * h4, 1.0f);
        } else {
            col = pr_hsv2rgb(0.11f + 0.04f * h1, 0.34f + 0.30f * h3, 1.0f);
        }

        float brightness = 1.0f + 2.3f * h6 * h6;

        for (int copy = -1; copy <= 1; ++copy) {
            float yOffset = float(copy) * BOX_H;
            float3 head = float3(sx, yHead + yOffset, sz);
            float3 tail = float3(sx, yHead + trailLen + yOffset, sz);

            float rayT;
            float segU;
            float d2 = pr_raySegmentDistanceSq(ro, rd, head, tail, rayT, segU);
            if (rayT < tNear || rayT > tFar || d2 > LINE_CUTOFF2) continue;

            float glow = exp(-d2 * 130000.0f) + exp(-d2 * 18000.0f) * 0.05f;
            float fade = 0.22f + 0.78f * pow(1.0f - segU, 0.55f);
            float depthFade = exp(-(rayT - tNear) * 0.28f);
            accumColor += col * brightness * glow * fade * depthFade;
        }
    }

    float3 mapped = 1.0f - exp(-accumColor * 2.8f);
    mapped = max(mapped, float3(0.001f, 0.001f, 0.002f));
    return float4(mapped, 1.0f);
}
