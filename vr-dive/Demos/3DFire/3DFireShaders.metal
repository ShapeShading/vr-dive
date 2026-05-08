// 3DFireShaders.metal
// Adapted from ShaderToy "3XXSWS" by XorDev.
// Source: https://www.shadertoy.com/view/3XXSWS
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Metal adaptation notes:
// - The original shader uses a screen-space ray from a fixed synthetic camera.
//   This version uses the real per-eye world ray intersected with a 2 m cube.
// - Outside the cube, marching starts at the visible cube surface; inside the
//   cube, marching starts at the eye.
// - The fire volume is integrated beyond the cube entry plane, so the simulated
//   turbulence is not clipped by the container volume.
// - GLSL `p.xz *= mat2(...)` is expanded explicitly to avoid row/column-major
//   ambiguity between GLSL and Metal matrix multiplication semantics.

#include <metal_stdlib>
using namespace metal;

struct ThreeDFireUniforms {
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

struct ThreeDFireVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float3 FIRE_BOX_HALF = float3(1.0f);
static constant float FIRE_EPSILON = 0.002f;
static constant float FIRE_SCENE_SCALE = 4.0f;
static constant int FIRE_TRACE_STEPS = 50;

vertex ThreeDFireVertexOut threeDFireVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant ThreeDFireUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    ThreeDFireVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 fireBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 fireTwistXZ(float2 value, float4 twistCos, float divisor) {
    return float2(
        value.x * twistCos.x + value.y * twistCos.y,
        value.x * twistCos.z + value.y * twistCos.w) / divisor;
}

fragment float4 threeDFireFragment(
    ThreeDFireVertexOut in [[stage_in]],
    constant ThreeDFireUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < FIRE_BOX_HALF - 1.0e-3f);
    float2 tBox = fireBoxIntersect(eye, rd, FIRE_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 marchOrigin = eye + rd * (tStart + FIRE_EPSILON);

    float time = uniforms.time;
    float z = 0.0f;
    float4 color = float4(0.0f);

    // Keep the original `p.z += 5. + cos(t)` motion, but center the volume in
    // container space by offsetting the ray origin by -5 along z beforehand.
    float3 rayOrigin = marchOrigin * FIRE_SCENE_SCALE + float3(0.0f, 0.0f, -5.0f);

    for (int i = 0; i < FIRE_TRACE_STEPS; ++i) {
        float3 p = rayOrigin + rd * z;
        p.z += 5.0f + cos(time);

        float4 twistCos = cos(p.y * 0.5f + float4(0.0f, 33.0f, 11.0f, 0.0f));
        float divisor = max(p.y * 0.1f + 1.0f, 0.1f);
        p.xz = fireTwistXZ(p.xz, twistCos, divisor);

        for (float frequency = 2.0f; frequency < 15.0f; frequency /= 0.6f) {
            p += cos((p.yzx - float3(time / 0.1f, time, frequency)) * frequency) / frequency;
        }

        float stepSize = 0.01f + abs(length(p.xz) + p.y * 0.3f - 0.5f) / 7.0f;
        z += stepSize;

        // In the original GLSL, `O += ... / d` happens in the loop increment,
        // after `d` has been overwritten with the raymarch step size. Using the
        // last turbulence frequency here makes the fire much darker than the
        // reference, so accumulate against the actual step size instead.
        color += (sin(z / 3.0f + float4(7.0f, 2.0f, 3.0f, 0.0f)) + 1.1f) / stepSize;
    }

    color = tanh(color / 1000.0f);
    return float4(color.rgb, 1.0f);
}