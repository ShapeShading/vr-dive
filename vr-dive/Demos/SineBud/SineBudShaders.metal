// SineBudShaders.metal
// Adapted from ShaderToy "Sine bud".
// Source: https://www.shadertoy.com/view/Mcl3Wn
//
// Metal adaptation notes:
// - The original shader is a compact screen-space ray marcher using macros.
//   This version reconstructs the real per-eye world ray, intersects it with a
//   2 m cube container, and marches from the visible cube surface or from the
//   eye when the viewer is inside the cube.
// - The iterative field continues beyond the entry plane, so the simulated bud
//   is not clipped to the cube volume.
// - GLSL macros for rotation, polar transforms and wave evaluation are expanded
//   into explicit Metal helper functions.

#include <metal_stdlib>
using namespace metal;

struct SineBudUniforms {
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

struct SineBudVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float SB_PI = 3.1416f;
static constant float SB_PI_2 = 1.5708f;
static constant float3 SB_BOX_HALF = float3(1.0f);
static constant float SB_TRACE_EPSILON = 0.0015f;
static constant int SB_STEPS = 200;
static constant float SB_MAX_DIST = 100.0f;
static constant float SB_SCENE_SCALE = 8.0f;

vertex SineBudVertexOut sineBudVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant SineBudUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    SineBudVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 sbRotate2D(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float2 sbFaceUV(float3 p) {
    float3 ap = abs(p);
    float2 uv;
    if (ap.x >= ap.y && ap.x >= ap.z) {
        uv = p.zy;
    } else if (ap.y >= ap.z) {
        uv = p.xz;
    } else {
        uv = p.xy;
    }
    return clamp(uv * 0.5f + 0.5f, 0.0f, 1.0f);
}

static float2 sbBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float sbWave(float3 p, uint axisA, uint axisB) {
    float px = p.x;
    float2 pair;
    if (axisA == 1u && axisB == 2u) {
        pair = p.yz;
    } else if (axisA == 2u && axisB == 1u) {
        pair = p.zy;
    } else if (axisA == 0u && axisB == 2u) {
        pair = p.xz;
    } else if (axisA == 2u && axisB == 0u) {
        pair = p.zx;
    } else if (axisA == 0u && axisB == 1u) {
        pair = p.xy;
    } else {
        pair = p.yx;
    }
    float2 target = abs(sin(px + float2(0.0f, SB_PI_2)));
    return length(float3(pair - target, 0.0f));
}

static float3 sbPolar(float3 v) {
    return float3(atan2(v.x, v.y), length(v.xy), sqrt(abs(v.z)));
}

fragment float4 sineBudFragment(
    SineBudVertexOut in [[stage_in]],
    constant SineBudUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < SB_BOX_HALF - 1.0e-3f);
    float2 tBox = sbBoxIntersect(eye, rd, SB_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 ro = (eye + rd * (tStart + SB_TRACE_EPSILON)) * SB_SCENE_SCALE;

    float t = uniforms.time / 5.0f;
    float2 m = -float2(t - SB_PI_2 * 0.5f, 0.6f);

    float d = 0.0f;
    float s = 1.0f;
    float3 k = float3(0.0f);
    float3 c = float3(0.0f);
    bool hitBud = false;

    for (int step = 0; step < SB_STEPS; ++step) {
        float3 p = ro + rd * d;
        p.yz = sbRotate2D(p.yz, m.y);
        p.xz = sbRotate2D(p.xz, m.x);
        p = abs(p) - cos(t + SB_PI) * 0.5f - 0.5f;

        float axisSphereX = length(abs(p) - float3(1.0f, 0.0f, 0.0f)) - 0.07f;
        float axisSphereY = length(abs(p) - float3(0.0f, 1.0f, 0.0f)) - 0.07f;
        float axisSphereZ = length(abs(p) - float3(0.0f, 0.0f, 1.0f)) - 0.07f;
        s = min(s, min(axisSphereX, min(axisSphereY, axisSphereZ)));

        float3 q = p;
        p = sbPolar(q);
        k.x = min(sbWave(p, 1u, 2u), sbWave(p, 2u, 1u));
        p = sbPolar(q.yzx);
        k.y = min(sbWave(p, 1u, 2u), sbWave(p, 2u, 1u));
        p = sbPolar(q.zxy);
        k.z = min(sbWave(p, 1u, 2u), sbWave(p, 2u, 1u));

        s = min(s, min(k.z, min(k.x, k.y)));
        if (s < 0.001f || d > SB_MAX_DIST) {
            hitBud = s < 0.001f;
            break;
        }
        d += s * 0.3f;
    }

    if (hitBud) {
        c += max(cos(d * SB_PI * 2.0f) - s * sqrt(max(d, 0.0f)) - k, 0.0f);
    }
    float3 color = hitBud ? (c + c.brg + c * c) : float3(0.0f);

    float2 q = sbFaceUV(hit);
    float vignette = 1.0f - 0.35f * dot(q * 2.0f - 1.0f, q * 2.0f - 1.0f);
    color *= vignette;
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}