// TorusFanShaders.metal
// Adapted from Stephane Cuillerdier (Aiekick), 2015.
// Source: https://www.shadertoy.com/view/4tS3Dc
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// This version renders the original torus-field effect inside a transparent
// 2 m × 2 m × 2 m cube container. The original ShaderToy samples a cubemap for
// reflection and background. This project does not bind a cubemap texture for
// container demos, so reflection is approximated with a procedural environment.

#include <metal_stdlib>
using namespace metal;

struct TorusFanUniforms {
    float  time;
    uint   viewCount;
    float  boxScale;
    float  _pad;
    float4 objectCenter;
};

struct TorusFanMeshVertex {
    float3 position;
    float3 normal;
};

struct TorusFanVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct TorusFanState {
    float3 dstepf;
    float3 colFact;
};

vertex TorusFanVertexOut torusFanVertex(
    ushort                      amplificationID [[amplification_id]],
    const device TorusFanMeshVertex *vertices   [[buffer(0)]],
    constant TorusFanUniforms  &uniforms        [[buffer(1)]],
    constant float4x4          *vpMatrices      [[buffer(2)]],
    uint                        vertexID        [[vertex_id]])
{
    TorusFanMeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    TorusFanVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float tf_dota(thread TorusFanState &state, float3 a, float3 b) {
    state.dstepf.y += state.colFact.y;
    return dot(a, b);
}

static float3 tf_nora(thread TorusFanState &state, float3 a) {
    state.dstepf.z += state.colFact.z;
    return normalize(a);
}

static float2 tf_uvMap(float3 p) {
    p = normalize(p);
    return float2(
        0.5f + atan2(p.z, p.x) / 6.28318530718f,
        0.5f - asin(clamp(p.y, -1.0f, 1.0f)) / 3.14159265359f);
}

static float2 tf_getTemp(thread TorusFanState &state, float3 p, float time) {
    p *= 2.0f;
    float2 p2 = tf_uvMap(p);
    float2 coef = float2(30.0f, 100.0f * (sin(time * 0.1f) * 0.5f + 0.5f));
    float r = fract(p2.x * coef.x + cos(p2.y * coef.y));
    return float2(tf_dota(state, p, p) * 100.0f * r, r);
}

static float3 tf_getHotColor(float temperature) {
    float safeTemperature = max(temperature, 1.0f);
    float3 col = float3(255.0f);
    col.x = 56100000.0f * pow(safeTemperature, -1.5f) + 148.0f;
    col.y = 100.04f * log(safeTemperature) - 623.6f;
    if (safeTemperature > 6500.0f) {
        col.y = 35200000.0f * pow(safeTemperature, -1.5f) + 184.0f;
    }
    col.z = 194.18f * log(safeTemperature) - 1448.6f;
    col = clamp(col, 0.0f, 255.0f) / 255.0f;
    if (safeTemperature < 1000.0f) {
        col *= safeTemperature / 1000.0f;
    }
    return col;
}

static float tf_sdTorus(float3 p, float2 t) {
    float2 q = float2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

static float3 tf_rotateSample(float3 p, float time) {
    float angleY = time * 0.35f;
    float cy = cos(angleY);
    float sy = sin(angleY);
    float2 xz = float2(cy * p.x - sy * p.z, sy * p.x + cy * p.z);
    p.x = xz.x;
    p.z = xz.y;

    float angleX = time * 0.19f;
    float cx = cos(angleX);
    float sx = sin(angleX);
    float2 yz = float2(cx * p.y - sx * p.z, sx * p.y + cx * p.z);
    p.y = yz.x;
    p.z = yz.y;
    return p;
}

static float4 tf_map(thread TorusFanState &state, float3 p, float time) {
    state.dstepf.x += state.colFact.x;
    float3 samplePoint = tf_rotateSample(p, time);
    float2 temp = tf_getTemp(state, samplePoint, time);
    float3 col = tf_getHotColor(temp.x);
    float disp = tf_dota(state, col, -state.dstepf.xyx);
    float dist = tf_sdTorus(samplePoint, float2(6.0f, 3.0f)) - disp;
    return float4(dist, col);
}

static float3 tf_nor(thread TorusFanState &state, float3 p, float prec, float time) {
    float2 e = float2(prec, 0.0f);
    return normalize(float3(
        tf_map(state, p + e.xyy, time).x - tf_map(state, p - e.xyy, time).x,
        tf_map(state, p + e.yxy, time).x - tf_map(state, p - e.yxy, time).x,
        tf_map(state, p + e.yyx, time).x - tf_map(state, p - e.yyx, time).x));
}

static float3 tf_environment(float3 dir) {
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 5.0f);
    float3 sky = mix(float3(0.03f, 0.04f, 0.08f), float3(0.15f, 0.22f, 0.36f), skyMix);
    float3 glow = float3(1.0f, 0.52f, 0.16f) * horizon;
    return sky + glow * 0.45f;
}

static float4 tf_boxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    float nearT = max(max(tMin.x, tMin.y), tMin.z);
    float farT = min(min(tMax.x, tMax.y), tMax.z);
    return float4(nearT, farT, 0.0f, 0.0f);
}

fragment float4 torusFanFragment(
    TorusFanVertexOut           in       [[stage_in]],
    constant TorusFanUniforms  &uniforms [[buffer(0)]],
    constant float4x4          *v2wMats  [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center = uniforms.objectCenter.xyz;
    float scale = uniforms.boxScale;
    float3 eye = (camWorld - center) / scale;
    float3 hit = (in.worldPos - center) / scale;

    TorusFanState state;
    state.dstepf = float3(0.0f);
    state.colFact = float3(0.0006f, 0.0004f, 0.17f);
    state.colFact.y = state.colFact.y * (sin(uniforms.time * 2.0f) * 0.5f + 0.5f) + state.colFact.y;
    state.colFact.x = state.colFact.y * (sin(uniforms.time) * 0.5f + 0.5f) + state.colFact.x / 3.0f;
    state.colFact.z = 0.5f * (sin(uniforms.time * 0.5f) * 0.5f + 0.5f) + 0.1f;

    float3 rd = tf_nora(state, hit - eye);
    const float3 halfBox = float3(1.0f);
    bool insideBox = all(abs(eye) < halfBox - 1e-3f);
    float4 tBox = tf_boxIntersect(eye, rd, halfBox);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float tEnd = tBox.y;
    if (tEnd <= tStart) {
        discard_fragment();
    }

    const float modelScale = 8.0f;
    float3 ro = (eye + rd * tStart) * modelScale;
    float maxDistance = min(50.0f, (tEnd - tStart) * modelScale);

    float distanceTraveled = 0.0f;
    float stepDistance = 0.001f;
    float3 p = ro;
    for (int i = 0; i < 500; i++) {
        if (stepDistance < 0.001f || stepDistance > 50.0f || distanceTraveled > maxDistance) {
            break;
        }
        float4 mapped = tf_map(state, p, uniforms.time);
        stepDistance = mapped.x * (stepDistance > 0.001f ? 0.3f : 0.05f);
        distanceTraveled += stepDistance;
        p = ro + rd * distanceTraveled;
    }

    if (distanceTraveled >= maxDistance || stepDistance > 50.0f) {
        discard_fragment();
    }

    float3 normal = tf_nor(state, p, 0.01f, uniforms.time);
    float3 reflected = reflect(rd, normal);
    float3 environment = tf_environment(reflected) * 0.6f;
    float4 surface = tf_map(state, p, uniforms.time);
    float3 color = environment + float3(pow(0.55f, 15.0f));
    color = mix(color, surface.yzw, 0.5f);
    color += state.dstepf;
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}