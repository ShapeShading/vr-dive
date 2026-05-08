// EtherShaders.metal
// Adapted from ShaderToy "t3XXWj" by XorDev.
// Source: https://www.shadertoy.com/view/t3XXWj
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Metal adaptation notes:
// - The original shader uses a fixed screen-space camera ray. This version uses
//   the real per-eye world ray intersected with a 2 m cube container.
// - Outside the cube, marching starts at the visible cube surface; inside the
//   cube, marching starts at the eye.
// - The ether volume is integrated beyond the cube entry plane, so the
//   simulated turbulence is not clipped by the container volume.
// - In the original GLSL, the glow accumulation divides by the raymarch step
//   size after `d` has been reassigned; this port preserves that order.

#include <metal_stdlib>
using namespace metal;

struct EtherUniforms {
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

struct EtherVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float3 ETHER_BOX_HALF = float3(1.0f);
static constant float ETHER_EPSILON = 0.002f;
static constant float ETHER_SCENE_SCALE = 4.0f;
static constant int ETHER_TRACE_STEPS = 80;

vertex EtherVertexOut etherVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant EtherUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    EtherVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 etherBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

fragment float4 etherFragment(
    EtherVertexOut in [[stage_in]],
    constant EtherUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < ETHER_BOX_HALF - 1.0e-3f);
    float2 tBox = etherBoxIntersect(eye, rd, ETHER_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 marchOrigin = eye + rd * (tStart + ETHER_EPSILON);

    float time = uniforms.time;
    float z = 0.0f;
    float4 color = float4(0.0f);

    float3 rayOrigin = marchOrigin * ETHER_SCENE_SCALE;

    for (int i = 0; i < ETHER_TRACE_STEPS; ++i) {
        float3 p = rayOrigin + rd * z;
        p.z -= 5.0f * time;

        for (float frequency = 1.0f; frequency < 15.0f; frequency /= 0.6f) {
            p += 0.6f * cos(p.yzx * frequency - float3(time * 0.6f, 0.0f, time)) / frequency;
        }

        float stepSize = 0.01f + abs(p.y * 0.3f + dot(cos(p), sin(p.yzx * 0.6f)) + 2.0f) / 3.0f;
        z += stepSize;
        color += max(sin(z * 0.4f + time + float4(6.0f, 2.0f, 4.0f, 0.0f)) + 0.7f, 0.2f) / stepSize;
    }

    color = tanh(color / 2000.0f);
    return float4(color.rgb, 1.0f);
}