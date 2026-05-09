// Fractal77GazShaders.metal
// "Fractal 77_gaz" — cube-container adaptation of ShaderToy fdy3WG.
// Source: https://www.shadertoy.com/view/fdy3WG
// This adaptation preserves the original axis-rotated recursive fold, distance
// accumulation, hue ramp, and strong post-power response while replacing the
// screen-space ray with a real per-eye world ray entering a visible 2 m cube.

#include <metal_stdlib>
using namespace metal;

struct Fractal77GazUniforms {
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

struct Fractal77GazVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct Fractal77Accum {
    float3 color;
    float energy;
    float feature;
};

static constant float3 F77_BOX_HALF = float3(1.0f);
static constant float3 F77_AXIS = float3(0.26726124f, 0.53452248f, 0.80178373f);

static float3 hue77(float h) {
    return cos(h * 6.3f + float3(0.0f, 23.0f, 21.0f)) * 0.5f + 0.5f;
}

static float3 f77Tonemap(float3 color) {
    return color / (1.0f + color);
}

static float3 rotateAroundAxis77(float3 p, float3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mix(axis * dot(p, axis), p, c) + s * cross(p, axis);
}

static float2 f77BoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 f77FaceUV(float3 p) {
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

static float3 reorderFold(float3 p) {
    return p.x < p.y ? p.zxy : p.zyx;
}

static Fractal77Accum traceFractal77(float3 ro, float3 rd, float time) {
    Fractal77Accum accum;
    accum.color = float3(0.0f);
    accum.energy = 0.0f;
    accum.feature = 0.0f;

    float g = 0.0f;
    for (int stepIndex = 0; stepIndex < 99; ++stepIndex) {
        float iteration = float(stepIndex + 1);
        float3 p = ro + g * rd;
        p.z -= 0.6f;
        p = rotateAroundAxis77(p, F77_AXIS, time * 0.3f);

        float s = 4.0f;
        float e = 0.0f;
        for (int fractalIndex = 0; fractalIndex < 8; ++fractalIndex) {
            p = abs(p);
            p = reorderFold(p);
            e = 1.8f / min(dot(p, p), 1.3f);
            s *= e;
            p = p * e - float3(12.0f, 3.0f, 3.0f);
        }

        e = length(p.xz) / max(s, 1.0e-4f);
        float3 tint = mix(float3(0.04f, 0.055f, 0.09f), hue77(log(max(s, 1.0e-4f))), 0.88f);
        float ridge = exp(-34.0f * e);
        float body = exp(-7.5f * e);
        float depthFade = exp(-0.035f * iteration);
        accum.color += tint * depthFade * (0.018f * body + 0.11f * ridge);
        accum.energy += ridge * depthFade * 0.0022f;
        accum.feature += ridge * depthFade * 0.09f;

        g += e;
        if (g > 18.0f) {
            break;
        }
    }

    accum.color = pow(f77Tonemap(clamp(accum.color * 1.45f, 0.0f, 6.0f)), float3(1.2f));
    accum.feature = clamp(accum.feature, 0.0f, 1.0f);
    return accum;
}

vertex Fractal77GazVertexOut fractal77GazVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant Fractal77GazUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    Fractal77GazVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

fragment float4 fractal77GazFragment(
    Fractal77GazVertexOut in [[stage_in]],
    constant Fractal77GazUniforms &uniforms [[buffer(0)]],
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

    bool insideOuter = all(abs(eye) < F77_BOX_HALF - 1.0e-3f);
    float2 tOuter = f77BoxIntersect(eye, rd, F77_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 localOrigin = eye + rd * (tStart + 0.001f);

    const float sceneScale = 2.8f;
    float3 sceneRo = localOrigin * sceneScale;
    float3 sceneRd = normalize(rd);

    Fractal77Accum accum = traceFractal77(sceneRo, sceneRd, uniforms.time);

    float axisFacing = pow(clamp(dot(sceneRd, F77_AXIS) * 0.5f + 0.5f, 0.0f, 1.0f), 3.0f);
    float horizon = pow(clamp(1.0f - abs(sceneRd.y), 0.0f, 1.0f), 2.0f);
    float3 background = mix(
        float3(0.015f, 0.02f, 0.04f),
        float3(0.16f, 0.08f, 0.23f),
        horizon);
    background += hue77(uniforms.time * 0.1f + axisFacing * 0.3f) * (0.008f + 0.02f * axisFacing);

    float2 faceUV = f77FaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.16f * dot(faceUV, faceUV);
    float patternPresence = accum.feature;

    float3 pattern = accum.color * (1.08f + 0.45f * patternPresence);
    float3 color = background * (0.06f + 0.04f * (1.0f - patternPresence)) + pattern;
    color += accum.energy * float3(0.55f, 0.78f, 1.05f) * 0.001f;
    color *= vignette;
    color = f77Tonemap(max(color, 0.0f));
    return float4(color, 1.0f);
}