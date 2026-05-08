// Fractal49GazShaders.metal
// "Fractal 49_gaz" — cube-container adaptation of ShaderToy fdSGzt.
// Source: https://www.shadertoy.com/view/fdSGzt
// Original source header notes this shader was forked from Xs3yRM and licensed
// under CC-BY-NC-SA-3.0. This adaptation preserves the core iterative fractal
// accumulation while replacing the synthetic screen camera with a real per-eye
// world ray entering a visible 2 m cube container.

#include <metal_stdlib>
using namespace metal;

struct Fractal49GazUniforms {
    float  time;
    uint   viewCount;
    float  cubeScale;
    float  padding;
    float4 objectCenter;
};

struct MeshVertex {
    float3 position;
    float3 normal;
};

struct Fractal49GazVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct FractalAccum {
    float3 color;
    float glow;
};

static constant float3 FG_BOX_HALF = float3(1.0f);
static constant float3 FG_AXIS_BASE = float3(0.26726124f, 0.53452248f, 0.80178373f);
static constant float3 FG_EQ = float3(0.577350269f);

static float3 hue(float h) {
    return cos(h * 6.3f + float3(0.0f, 23.0f, 21.0f)) * 0.5f + 0.5f;
}

static float3 rotateAroundAxis(float3 p, float3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mix(axis * dot(p, axis), p, c) + s * cross(p, axis);
}

static float2 fgBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 fgFaceUV(float3 p) {
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

static float3 sceneWarp(float3 p, float time) {
    float3 axis = normalize(float3(1.0f, 2.0f * sin(time * 0.1f), 3.0f));
    p = rotateAroundAxis(p, axis, time * 0.2f);
    p.xz = float2(
        cos(time * 0.17f) * p.x + sin(time * 0.17f) * p.z,
        -sin(time * 0.17f) * p.x + cos(time * 0.17f) * p.z);
    return p;
}

static FractalAccum traceFractal(float3 ro, float3 rd, float time) {
    FractalAccum accum;
    accum.color = float3(0.0f);
    accum.glow = 0.0f;

    float g = 1.5f;
    float lastError = 0.0f;

    for (int stepIndex = 0; stepIndex < 90; ++stepIndex) {
        float iteration = float(stepIndex + 1);
        float3 p = g * rd - float3(-0.2f, 0.3f, 2.5f);
        p += ro;
        p = sceneWarp(p, time);

        float s = 5.0f;
        p = p / max(dot(p, p), 1.0e-4f) + 1.0f;
        float e = 0.0f;
        for (int fractalIndex = 0; fractalIndex < 8; ++fractalIndex) {
            p = 1.0f - abs(p - 1.0f);
            e = 1.6f / min(dot(p, p), 1.5f);
            s *= e;
            p *= e;
        }

        lastError = length(cross(p, FG_EQ)) / s - 5.0e-4f;
        float3 tint = mix(float3(1.0f), hue(log(max(s, 1.0e-4f)) * 0.3f), 0.8f);
        float falloff = exp(-12.0f * iteration * iteration * max(lastError, 0.0f));
        accum.color += 0.1f * tint * falloff;
        accum.glow += 0.015f / (0.0005f + lastError * lastError);

        g += lastError;
        if (g > 18.0f) {
            break;
        }
    }

    return accum;
}

vertex Fractal49GazVertexOut fractal49GazVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant Fractal49GazUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    Fractal49GazVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

fragment float4 fractal49GazFragment(
    Fractal49GazVertexOut in [[stage_in]],
    constant Fractal49GazUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 eye = (cameraWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 rd = normalize(surfacePos - eye);

    bool insideOuter = all(abs(eye) < FG_BOX_HALF - 1.0e-3f);
    float2 tOuter = fgBoxIntersect(eye, rd, FG_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 localOrigin = eye + rd * (tStart + 0.001f);

    const float sceneScale = 2.6f;
    float3 sceneRo = localOrigin * sceneScale;
    float3 sceneRd = normalize(rd);

    FractalAccum accum = traceFractal(sceneRo, sceneRd, uniforms.time);

    float3 axis = normalize(float3(1.0f, 2.0f * sin(uniforms.time * 0.1f), 3.0f));
    float horizon = pow(clamp(1.0f - abs(sceneRd.y), 0.0f, 1.0f), 2.0f);
    float facing = pow(clamp(dot(sceneRd, FG_AXIS_BASE) * 0.5f + 0.5f, 0.0f, 1.0f), 3.0f);
    float swirl = pow(clamp(dot(sceneRd, axis) * 0.5f + 0.5f, 0.0f, 1.0f), 2.0f);

    float3 background = mix(
        float3(0.03f, 0.02f, 0.06f),
        float3(0.24f, 0.08f, 0.32f),
        horizon);
    background += hue(uniforms.time * 0.08f + swirl * 0.25f) * (0.08f + 0.22f * facing);
    background += float3(0.14f, 0.28f, 0.1f) * swirl * 0.3f;

    float2 faceUV = fgFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.16f * dot(faceUV, faceUV);

    float glow = accum.glow * 0.007f;
    float3 color = background * 0.75f + accum.color;
    color += glow * float3(0.45f, 0.82f, 0.52f);
    color *= vignette;

    color = clamp(color, 0.0f, 1.0f);
    color = pow(color, float3(0.92f));
    return float4(color, 1.0f);
}