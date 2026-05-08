// FiberSpiralShaders.metal
// Adapted from ShaderToy "XsdSW7" by Stephane Cuillerdier (Aiekick), 2015.
// Source: https://www.shadertoy.com/view/XsdSW7
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Metal adaptation notes:
// - The original shader uses a synthetic camera moving forward along +Z. This
//   version uses the real per-eye world ray intersected with a 2 m cube.
// - Outside the cube, marching starts at the visible cube surface; inside the
//   cube, marching starts at the eye.
// - The fractal field is traced beyond the container entry plane, so the
//   simulated structure is not clipped by the cube volume.
// - GLSL `mod(p, 4.) - 2.` is expanded with floor-based modulo because Metal's
//   `fmod` does not match GLSL's negative-input wrap behaviour.

#include <metal_stdlib>
using namespace metal;

struct FiberSpiralUniforms {
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

struct FiberSpiralVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float3 FS_BOX_HALF = float3(1.0f);
static constant float FS_RATIO = 0.5f;
static constant float FS_SCENE_SCALE = 4.0f;
static constant float FS_TRACE_EPSILON = 0.002f;
static constant float FS_MAX_DIST = 30.0f;
static constant int FS_TRACE_STEPS = 250;

vertex FiberSpiralVertexOut fiberSpiralVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant FiberSpiralUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    FiberSpiralVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 fsPath(float z) {
    return float2(cos(z), sin(z));
}

static float2 fsRotate(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float3 fsMod(float3 x, float y) {
    return x - y * floor(x / y);
}

static float fsFractus(float3 p) {
    float2 z = p.xy;
    float2 c = float2(0.28f, -0.56f) * 2.0f * FS_RATIO;
    float k = 1.0f;
    float h = 1.0f;

    for (int i = 0; i < 7; ++i) {
        h *= 4.0f * k;
        k = dot(z, z);
        if (k > 4.0f) {
            break;
        }
        z = float2(z.x * z.x - z.y * z.y, 2.0f * z.x * z.y) + c;
    }

    float safeK = max(k, 1.0e-5f);
    float safeH = max(h, 1.0e-5f);
    return sqrt(safeK / safeH) * log(safeK);
}

static float fsDf(float3 p) {
    p.xy += fsPath(p.z * 0.2f) * 1.5f;
    p.xy = fsRotate(p.xy, p.z * 0.2f);
    p = fsMod(p, 4.0f) - 2.0f;
    return fsFractus(p);
}

static float3 fsNor(float3 p, float prec) {
    float2 e = float2(prec, 0.0f);
    return normalize(float3(
        fsDf(p + e.xyy) - fsDf(p - e.xyy),
        fsDf(p + e.yxy) - fsDf(p - e.yxy),
        fsDf(p + e.yyx) - fsDf(p - e.yyx)));
}

static float fsSoftShadow(float3 ro, float3 rd, float mint, float tmax) {
    float res = 1.0f;
    float t = mint;
    for (int i = 0; i < 18; ++i) {
        float h = fsDf(ro + rd * t);
        res = min(res, 8.0f * h / max(t, 1.0e-3f));
        t += h * 0.25f;
        if (h < 0.001f || t > tmax) {
            break;
        }
    }
    return clamp(res, 0.0f, 1.0f);
}

static float fsCao(float3 pos, float3 nor) {
    float occ = 0.0f;
    float sca = 1.0f;
    for (int i = 0; i < 10; ++i) {
        float hr = 0.01f + 0.12f * float(i) / 4.0f;
        float3 aopos = nor * hr + pos;
        float dd = fsDf(aopos);
        occ += -(dd - hr) * sca;
        sca *= 0.95f;
    }
    return clamp(1.0f - 3.0f * occ, 0.0f, 1.0f);
}

static float3 fsLighting(float3 p, float3 lp, float3 rd, float prec) {
    float3 l = lp - p;
    float dist = max(length(l), 0.01f);
    float atten = exp(-0.0001f * dist) - 0.5f;
    l /= dist;

    float3 n = fsNor(p, prec);
    float3 r = reflect(-l, n);
    float dif = clamp(dot(l, n), 0.0f, 1.0f);
    float spe = pow(clamp(dot(r, -rd), 0.0f, 1.0f), 8.0f);
    float fre = pow(clamp(1.0f + dot(n, rd), 0.0f, 1.0f), 2.0f);
    float dom = smoothstep(-1.0f, 1.0f, r.y);

    dif *= fsSoftShadow(p, l, 0.01f, 1.0f);

    float3 lin = float3(0.08f, 0.32f, 0.47f) * fsCao(p, n);
    lin += dif * float3(1.0f, 1.0f, 0.84f);
    lin += 2.5f * spe * dif * float3(1.0f, 1.0f, 0.84f);
    lin += 2.5f * fre * float3(1.0f);
    lin += 0.5f * dom * float3(1.0f);

    return lin * atten * fsCao(p, n);
}

static float2 fsBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

fragment float4 fiberSpiralFragment(
    FiberSpiralVertexOut in [[stage_in]],
    constant FiberSpiralUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < FS_BOX_HALF - 1.0e-3f);
    float2 tBox = fsBoxIntersect(eye, rd, FS_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float time = uniforms.time * 5.0f;
    float3 ro = (eye + rd * (tStart + FS_TRACE_EPSILON)) * FS_SCENE_SCALE + float3(0.0f, 0.0f, time);

    float d = 0.0f;
    float s = 0.01f;
    float3 p = ro;
    for (int i = 0; i < FS_TRACE_STEPS; ++i) {
        if (s < 0.0025f * d || d > FS_MAX_DIST) {
            break;
        }
        s = fsDf(p);
        d += s * 0.2f;
        p = ro + rd * d;
    }

    float3 color = float3(0.47f, 0.6f, 0.76f) * fsLighting(p, ro, rd, 0.1f);
    color = mix(color, float3(0.5f, 0.49f, 0.72f), 1.0f - exp(-0.01f * d * d));
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}