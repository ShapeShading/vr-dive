// FloreusShaders.metal
// Adapted from ShaderToy "Floreus" by Jaenam.
// Source: https://www.shadertoy.com/view/33fyWB
// License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
//
// Metal adaptation notes:
// - The original shader is a compact forward ray accumulator defined in screen
//   space. This version evaluates the same iterative field along the real per-
//   eye world ray after intersecting a 2 m cube container.
// - Outside the cube, marching starts at the visible cube surface; inside the
//   cube, marching starts at the eye.
// - The fractal accumulation continues beyond the entry plane, so the visual
//   field is not clipped to the cube volume.
// - GLSL macro rotations and implicit initialization tricks are expanded into
//   explicit Metal helpers and explicit initial values.

#include <metal_stdlib>
using namespace metal;

struct FloreusUniforms {
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

struct FloreusVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float3 FL_BOX_HALF = float3(1.0f);
static constant float FL_TRACE_EPSILON = 0.0015f;
static constant int FL_STEPS = 140;
static constant int FL_INNER_STEPS = 5;
static constant float FL_SCENE_SCALE = 2.4f;

vertex FloreusVertexOut floreusVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant FloreusUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    FloreusVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 flRotate2D(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float2 flFaceUV(float3 p) {
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

static float2 flBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

fragment float4 floreusFragment(
    FloreusVertexOut in [[stage_in]],
    constant FloreusUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < FL_BOX_HALF - 1.0e-3f);
    float2 tBox = flBoxIntersect(eye, rd, FL_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 ro = (eye + rd * (tStart + FL_TRACE_EPSILON)) * FL_SCENE_SCALE;

    float4 accum = float4(0.0f);
    float d = 0.0f;
    float time = uniforms.time;

    for (int step = 0; step < FL_STEPS; ++step) {
        float3 q = ro + rd * d;
        float3 p = q;
        p.z -= 0.3f;
        p.xz = flRotate2D(p.xz, 0.2f * time);

        float w = 5.0f;
        for (int inner = 0; inner < FL_INNER_STEPS; ++inner) {
            float2 folded = min(abs(p.xz), abs(p.xy));
            float numer = length(float2(0.7f) - folded);
            p = sin(p);
            float denom = max(dot(p, p + p), 1.0e-4f);
            float l = numer / denom;
            p *= l;
            w *= l;
        }

        float s = max(length(q) - 0.2f, length(p) / max(w, 1.0e-4f));
        d += s;
        accum += float4(3.0f, 2.0f, 1.0f, 1.0f) * 20.0f * d / max(s, 1.0e-4f);
    }

    float4 color = tanh(accum / 3.5e6f);
    float2 q = flFaceUV(hit);
    float vignette = 1.0f - 0.35f * dot(q * 2.0f - 1.0f, q * 2.0f - 1.0f);
    color.rgb *= vignette;
    return float4(clamp(color.rgb, 0.0f, 1.0f), 1.0f);
}