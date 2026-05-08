// BubbleRingsShaders.metal
// Adapted from ShaderToy "WdB3Dw".
// Source: https://www.shadertoy.com/view/WdB3Dw
// Uses pieces from:
// - HG_SDF helpers: https://www.shadertoy.com/view/Xs3GRB
// - Spectrum palette by IQ: https://www.shadertoy.com/view/ll2GD3
// - Main SDF reference: https://www.shadertoy.com/view/wsfGDS
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Metal adaptation notes:
// - The original shader uses a synthetic look-at camera. This version uses the
//   real per-eye world ray intersected with a 2 m cube container.
// - Outside the cube, marching starts at the visible cube surface; inside the
//   cube, marching starts at the eye.
// - The glow accumulation continues beyond the container entry plane, so the
//   simulated rings are not clipped by the cube volume.

#include <metal_stdlib>
using namespace metal;

struct BubbleRingsUniforms {
    float  time;
    uint   viewCount;
    float  boxScale;
    float  padding;
    float4 objectCenter;
};

struct MeshVertex {
    float3 position;
    float3 normal;
};

struct BubbleRingsVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float BR_PI = 3.14159265359f;
static constant float3 BR_BOX_HALF = float3(1.0f);
static constant float BR_SCENE_SCALE = 2.4f;
static constant float BR_ITER = 82.0f;
static constant float BR_FUDGE_FACTOR = 0.8f;
static constant float BR_INTERSECTION_PRECISION = 0.001f;
static constant float BR_MAX_DIST = 20.0f;
static constant float BR_TRACE_EPSILON = 0.002f;

vertex BubbleRingsVertexOut bubbleRingsVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant BubbleRingsUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    BubbleRingsVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static void brRotate(thread float2 &p, float a) {
    float c = cos(a);
    float s = sin(a);
    p = c * p + s * float2(p.y, -p.x);
}

static float brSmax(float a, float b, float r) {
    float2 u = max(float2(r + a, r + b), float2(0.0f));
    return min(-r, max(a, b)) + length(u);
}

static float3 brPal(float t, float3 a, float3 b, float3 c, float3 d) {
    return a + b * cos(6.28318f * (c * t + d));
}

static float3 brSpectrum(float n) {
    return brPal(
        n,
        float3(0.5f, 0.5f, 0.5f),
        float3(0.5f, 0.5f, 0.5f),
        float3(1.0f, 1.0f, 1.0f),
        float3(0.0f, 0.33f, 0.67f));
}

static float4 brInverseStereographic(float3 p, thread float &k) {
    k = 2.0f / (1.0f + dot(p, p));
    return float4(k * p, k - 1.0f);
}

static float brFTorus(float4 p4) {
    float xyLen = max(length(p4.xy), 1.0e-5f);
    float zwLen = max(length(p4.zw), 1.0e-5f);
    float d1 = xyLen / zwLen - 1.0f;
    float d2 = zwLen / xyLen - 1.0f;
    float d = d1 < 0.0f ? -d1 : d2;
    return d / BR_PI;
}

static float brFixDistance(float d, float k) {
    float sn = sign(d);
    d = abs(d);
    d = d / max(k, 1.0e-5f) * 1.82f;
    d += 1.0f;
    d = pow(d, 0.5f);
    d -= 1.0f;
    d *= 5.0f / 3.0f;
    return d * sn;
}

static float brMap(float3 p, float time) {
    float k;
    float4 p4 = brInverseStereographic(p, k);

    float2 zy = p4.zy;
    brRotate(zy, time * -BR_PI / 2.0f);
    p4.z = zy.x;
    p4.y = zy.y;

    float2 xw = float2(p4.x, p4.w);
    brRotate(xw, time * -BR_PI / 2.0f);
    p4.x = xw.x;
    p4.w = xw.y;

    float d = brFTorus(p4);
    d = abs(d);
    d -= 0.2f;
    d = brFixDistance(d, k);
    d = brSmax(d, length(p) - 1.85f, 0.2f);
    return d;
}

static float2 brBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

fragment float4 bubbleRingsFragment(
    BubbleRingsVertexOut in [[stage_in]],
    constant BubbleRingsUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float cubeScale = max(uniforms.boxScale, 1.0e-4f);
    float3 eye = (camWorld - center) / cubeScale;
    float3 hit = (in.worldPos - center) / cubeScale;
    float3 rd = normalize(hit - eye);

    bool insideBox = all(abs(eye) < BR_BOX_HALF - 1.0e-3f);
    float2 tBox = brBoxIntersect(eye, rd, BR_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 rayOrigin = (eye + rd * (tStart + BR_TRACE_EPSILON)) * BR_SCENE_SCALE;
    float time = fmod(uniforms.time / 2.0f, 1.0f);

    float rayLength = 0.0f;
    float distance = 0.0f;
    float3 color = float3(0.0f);

    for (int step = 0; step < int(BR_ITER); ++step) {
        rayLength += max(BR_INTERSECTION_PRECISION, abs(distance) * BR_FUDGE_FACTOR);
        float3 rayPosition = rayOrigin + rd * rayLength;
        distance = brMap(rayPosition, time);

        float3 c = float3(max(0.0f, 0.01f - abs(distance)) * 0.5f);
        c *= float3(1.4f, 2.1f, 1.7f);
        c += float3(0.6f, 0.25f, 0.7f) * BR_FUDGE_FACTOR / 160.0f;
        c *= 1.0f - smoothstep(7.0f, 20.0f, length(rayPosition));

        float rl = 1.0f - smoothstep(0.1f, BR_MAX_DIST, rayLength);
        c *= rl;
        c *= brSpectrum(rl * 6.0f - 0.6f);

        color += c;
        if (rayLength > BR_MAX_DIST) {
            break;
        }
    }

    color = pow(max(color, 0.0f), float3(1.0f / 1.8f)) * 2.0f;
    color = pow(max(color, 0.0f), float3(2.0f)) * 3.0f;
    color = pow(max(color, 0.0f), float3(1.0f / 2.2f));

    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}