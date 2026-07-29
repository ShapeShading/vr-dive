// ApollonianElevatorShaders.metal
// Adapted from ShaderToy "Xtlyzl" by coyote.
// Source: https://www.shadertoy.com/view/Xtlyzl
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Metal adaptation notes:
// - The original shader uses an implicit screen-space camera and a compact macro.
// - This version uses the real per-eye world ray intersected with a 2 m cube.
// - Marching starts at the visible cube surface, or at the eye when the camera
//   is inside the cube.
// - GLSL `mod(p - 1., 2.) - 1.` is expanded explicitly with floor-based modulo,
//   since Metal's `fmod` does not match GLSL's negative-input behavior.

#include <metal_stdlib>
using namespace metal;

struct ApollonianElevatorUniforms {
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

struct ApollonianElevatorVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float3 AE_BOX_HALF = float3(1.0f);
static constant float AE_SCENE_SCALE = 3.2f;
static constant float3 AE_SCENE_OFFSET = float3(1.0f, 1.0f, 1.0f);
static constant float AE_EPSILON = 0.002f;
static constant float AE_MIN_STEP = 0.0005f;
static constant float AE_MAX_TRACE_DISTANCE = 14.0f;
static constant int AE_MAX_TRACE_STEPS = 100;

vertex ApollonianElevatorVertexOut apollonianElevatorVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant ApollonianElevatorUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    ApollonianElevatorVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float3 aeMod(float3 x, float y) {
    return x - y * floor(x / y);
}

static float aeMap(float3 p, float time) {
    p.y += 0.2f * time;

    float scale = 1.0f;
    for (int i = 0; i < 7; ++i) {
        p = aeMod(p - 1.0f, 2.0f) - 1.0f;
        float invRadius2 = max(dot(p, p), 1.0e-4f);
        float k = 1.5f / invRadius2;
        p *= k;
        scale *= k;
    }

    return length(p) / scale - 0.01f;
}

static float2 aeBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float3 aeNormal(float3 p, float time) {
    float2 e = float2(0.003f, 0.0f);
    return normalize(float3(
        aeMap(p + e.xyy, time) - aeMap(p - e.xyy, time),
        aeMap(p + e.yxy, time) - aeMap(p - e.yxy, time),
        aeMap(p + e.yyx, time) - aeMap(p - e.yyx, time)));
}

fragment float4 apollonianElevatorFragment(
    ApollonianElevatorVertexOut in [[stage_in]],
    constant ApollonianElevatorUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float scale = max(uniforms.boxScale, 1.0e-4f);
    float3 eye = (camWorld - center) / scale;
    float3 hit = (in.worldPos - center) / scale;
    float3 rd = normalize(hit - eye);

    bool insideBox = all(abs(eye) < AE_BOX_HALF - 1.0e-3f);
    float2 tBox = aeBoxIntersect(eye, rd, AE_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float tEnd = tBox.y;
    if (tEnd <= tStart) {
        discard_fragment();
    }

    float3 marchOrigin = eye + rd * (tStart + AE_EPSILON);
    float3 ro = marchOrigin * AE_SCENE_SCALE + AE_SCENE_OFFSET;
    float traceLimit = min(AE_MAX_TRACE_DISTANCE, (tEnd - tStart) * AE_SCENE_SCALE);

    float distanceTraveled = 0.0f;
    float prevDistanceField = 0.0f;
    float stepDistance = 0.0f;
    float3 p = ro;

    for (int i = 0; i < AE_MAX_TRACE_STEPS; ++i) {
        p = ro + rd * distanceTraveled;
        float distanceField = aeMap(p, uniforms.time);
        prevDistanceField = distanceField;
        if (distanceField < AE_MIN_STEP || distanceTraveled > traceLimit) {
            break;
        }
        stepDistance = max(distanceField, AE_MIN_STEP);
        distanceTraveled += stepDistance;
    }

    if (distanceTraveled > traceLimit) {
        discard_fragment();
    }

    float3 hitPoint = ro + rd * distanceTraveled;
    float previousField = aeMap(hitPoint - rd * max(stepDistance, 0.02f), uniforms.time);

    // Preserve the original compact color idea while stabilizing the divisor for
    // arbitrary 3D viewing directions through the container.
    float zDenominator = hitPoint.z >= 0.0f ? max(hitPoint.z, 0.35f) : min(hitPoint.z, -0.35f);
    float3 color = (hitPoint * previousField - 2.0f) / zDenominator + 1.0f;

    float3 normal = aeNormal(hitPoint, uniforms.time);
    float3 lightDir = normalize(float3(0.45f, 0.82f, -0.34f));
    float diffuse = max(dot(normal, lightDir), 0.0f);
    float fresnel = pow(1.0f - max(dot(-rd, normal), 0.0f), 2.2f);
    color *= 0.35f + 0.65f * diffuse;
    color += float3(0.08f, 0.10f, 0.14f) * fresnel * 0.35f;
    color += float3(0.03f, 0.02f, 0.05f) * clamp(1.0f - distanceTraveled / traceLimit, 0.0f, 1.0f);

    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}