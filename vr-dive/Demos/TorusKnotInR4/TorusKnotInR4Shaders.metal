// TorusKnotInR4Shaders.metal
// Adapted from ShaderToy "tsBGzt" by S.Guillitte.
// Source: https://www.shadertoy.com/view/tsBGzt
// Based on https://www.shadertoy.com/view/4ds3zn by inigo quilez (iq), 2013.
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Metal adaptation notes:
// - The original shader uses a synthetic orbit camera. This version uses the
//   real per-eye world ray intersected with a 2 m cube container.
// - Outside the cube, marching starts at the visible cube surface; inside the
//   cube, marching starts at the eye.
// - The fractal field is traced beyond the cube entry plane, so the simulated
//   structure is not clipped by the container volume.

#include <metal_stdlib>
using namespace metal;

struct TorusKnotInR4Uniforms {
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

struct TorusKnotInR4VertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct TK4State {
    float4 orb;
    float ss;
    float blendSelector;
};

static constant float TK4_G = 0.85f;
static constant float3 TK4_CSIZE = float3(1.0f, 0.8f, 1.1f);
static constant float3 TK4_BOX_HALF = float3(1.0f);
static constant float TK4_SCENE_SCALE = 2.6f;
static constant float TK4_MAX_DISTANCE = 100.0f;
static constant float TK4_TRACE_EPSILON = 0.002f;
static constant int TK4_TRACE_STEPS = 200;

vertex TorusKnotInR4VertexOut torusKnotInR4Vertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant TorusKnotInR4Uniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    TorusKnotInR4VertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float3 tk4Fold(float3 p) {
    p = (-1.0f + 2.0f * fract(0.5f * p * TK4_CSIZE + 0.5f)) / TK4_CSIZE;
    p = clamp(p, -TK4_CSIZE, TK4_CSIZE) * 2.0f - p;
    return p;
}

static float tk4MapWithIterations(float3 p, int iterationCount, thread TK4State &state) {
    float scale = 1.0f;
    state.orb = float4(1000.0f);

    for (int i = 0; i < iterationCount; ++i) {
        p = tk4Fold(p);
        float r2 = max(dot(p, p), 1.0e-5f);
        state.orb = min(state.orb, float4(abs(p), r2));

        float k = max(state.ss / r2, 0.1f) * TK4_G;
        p *= k;
        scale *= k;
    }

    float d1 = 0.2f * (
        length(p.xz) * abs(p.y) +
        length(p.xy) * abs(p.z) +
        length(p.yz) * abs(p.x)) / scale;
    float d2 = 0.25f * abs(p.y) / scale;
    return state.blendSelector * d1 + (1.0f - state.blendSelector) * d2;
}

static float tk4Map(float3 p, thread TK4State &state) {
    return tk4MapWithIterations(p, 8, state);
}

static float tk4MapApprox(float3 p, thread TK4State &state) {
    return tk4MapWithIterations(p, 3, state);
}

static float2 tk4BoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float tk4Trace(float3 ro, float3 rd, thread TK4State &state, thread float4 &orbOut) {
    const float precis = 0.001f;
    const float approx = 0.1f;

    float h = precis * 2.0f;
    float t = 0.0f;
    orbOut = float4(1000.0f);

    for (int i = 0; i < TK4_TRACE_STEPS; ++i) {
        if (t > TK4_MAX_DISTANCE) {
            break;
        }

        if (abs(h) > approx || t > 20.0f) {
            t += h;
            h = tk4MapApprox(ro + rd * t, state);
            orbOut = state.orb;
            continue;
        }

        if (abs(h) < precis * (1.0f + 0.2f * t)) {
            break;
        }

        t += h;
        h = tk4Map(ro + rd * t, state);
        orbOut = state.orb;
    }

    return t > TK4_MAX_DISTANCE ? -1.0f : t;
}

static float3 tk4CalcNormal(float3 pos, thread TK4State &state) {
    float3 eps = float3(0.0001f, 0.0f, 0.0f);
    float dx = tk4Map(pos + eps.xyy, state) - tk4Map(pos - eps.xyy, state);
    float dy = tk4Map(pos + eps.yxy, state) - tk4Map(pos - eps.yxy, state);
    float dz = tk4Map(pos + eps.yyx, state) - tk4Map(pos - eps.yyx, state);
    return normalize(float3(dx, dy, dz));
}

fragment float4 torusKnotInR4Fragment(
    TorusKnotInR4VertexOut in [[stage_in]],
    constant TorusKnotInR4Uniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < TK4_BOX_HALF - 1.0e-3f);
    float2 tBox = tk4BoxIntersect(eye, rd, TK4_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 marchOrigin = eye + rd * (tStart + TK4_TRACE_EPSILON);

    TK4State state;
    float baseTime = uniforms.time;
    float morph = cos(0.1f * baseTime);
    state.blendSelector = step(0.0f, morph);
    state.ss = 1.3f - 0.2f * morph;
    state.orb = float4(1000.0f);

    float3 ro = marchOrigin * TK4_SCENE_SCALE;
    float4 trap = float4(1000.0f);
    float t = tk4Trace(ro, rd, state, trap);
    if (t <= 0.0f) {
        discard_fragment();
    }

    float3 pos = ro + t * rd;
    float3 nor = tk4CalcNormal(pos, state);

    float3 light1 = normalize(float3(0.577f, 0.577f, -0.577f));
    float3 light2 = normalize(float3(-0.707f, 0.0f, 0.707f));
    float key = clamp(dot(light1, nor), 0.0f, 1.0f);
    float bac = clamp(0.2f + 0.8f * dot(light2, nor), 0.0f, 1.0f);
    float amb = 0.7f + 0.3f * nor.y;
    float ao = pow(clamp(trap.w * 2.0f, 0.0f, 1.0f), 1.2f);

    float3 brdf = float3(0.40f) * amb * ao;
    brdf += float3(1.00f) * key * ao;
    brdf += float3(0.40f) * bac * ao;

    float3 rgb =
        0.4f * abs(sin(4.5f + float3(trap.w, trap.y * trap.y, 2.0f - trap.w))) +
        0.6f * sin(float3(-0.5f, -0.2f, 0.8f) + 1.3f + trap.x * 9.5f);

    float3 color = rgb * brdf * exp(-0.2f * t);
    color = sqrt(max(color, 0.0f));
    color = mix(color, smoothstep(0.0f, 1.0f, color), 0.25f);

    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}