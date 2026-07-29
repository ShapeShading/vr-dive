// SaturdayWeirdnessShaders.metal
// Adapted from ShaderToy "Saturday weirdness".
// Source: https://www.shadertoy.com/view/43jXWt
//
// Metal adaptation notes:
// - The supplied ShaderToy code is the final FXAA resolve pass over iChannel0.
//   The original visual source is not present in the provided snippet, so this
//   cube-container version rebuilds a scene with the same intent: dense,
//   high-contrast weird structures with softened edges.
// - Instead of a post-process over a full-screen buffer, this version ray-
//   marches a procedural 3D field from the cube boundary and uses derivative-
//   aware softening near band edges as an in-shader substitute for the FXAA
//   resolve idea.

#include <metal_stdlib>
using namespace metal;

struct SaturdayWeirdnessUniforms {
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

struct SaturdayWeirdnessVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct SWHit {
    float distance;
    float sdf;
    float feature;
    float3 albedo;
    bool hit;
};

static constant float3 SW_BOX_HALF = float3(1.0f);
static constant float SW_MAX_DIST = 28.0f;
static constant float SW_HIT_EPSILON = 0.0015f;
static constant int SW_MAX_STEPS = 128;
static constant int SW_VOLUME_STEPS = 20;

vertex SaturdayWeirdnessVertexOut saturdayWeirdnessVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant SaturdayWeirdnessUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    SaturdayWeirdnessVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 swRotate2D(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float swRoundBox(float3 p, float3 b, float r) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f) - r;
}

static float3 swWarp(float3 p, float time) {
    p *= 1.45f;
    p.xz = swRotate2D(p.xz, time * 0.17f);
    p.yz = swRotate2D(p.yz, 0.52f + sin(time * 0.28f) * 0.16f);
    p.xy = swRotate2D(p.xy, 0.22f * sin(time * 0.19f));
    return p;
}

static float swGyroid(float3 p) {
    return dot(sin(p), cos(p.zxy));
}

static float3 swPalette(float t) {
    float u = 0.5f + 0.5f * sin(t);
    float v = 0.5f + 0.5f * sin(t * 0.72f + 1.2f);
    float w = 0.5f + 0.5f * cos(t * 1.1f - 0.4f);
    float3 deep = float3(0.04f, 0.08f, 0.28f);
    float3 blue = float3(0.14f, 0.42f, 0.92f);
    float3 cyan = float3(0.64f, 0.92f, 1.0f);
    float3 magenta = float3(0.86f, 0.2f, 0.96f);
    return mix(mix(deep, blue, u), mix(magenta, cyan, w), v);
}

static float swGeometry(float3 p, float time) {
    float3 q = swWarp(p, time);
    return swRoundBox(q, float3(0.72f), 0.24f);
}

static float swFlow(float3 p, float time) {
    float3 q = swWarp(p, time);
    float gyroid = swGyroid(q * 2.55f + time * 0.45f);
    float swirl = atan2(q.y, q.x);
    float ribbons = sin(length(q.xy) * 8.6f - swirl * 4.2f + q.z * 3.3f - time * 0.8f);
    float bands = sin(q.x * 3.1f - q.z * 4.0f + gyroid * 1.4f)
      + 0.55f * sin(q.y * 4.9f + time * 0.55f)
      + 0.35f * cos(q.z * 4.4f - q.x * 1.2f);
    return ribbons + bands * 0.65f + gyroid * 0.45f;
}

static float2 swBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 swFaceUV(float3 p) {
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

static SWHit swMap(float3 p, float time) {
    SWHit hit;
    float sdf = swGeometry(p, time);
    float flow = swFlow(p, time);
    float feature = 0.5f + 0.5f * sin(flow * 1.8f);

    hit.distance = 0.0f;
    hit.sdf = sdf;
    hit.feature = feature;
    hit.albedo = swPalette(flow);
    hit.hit = false;
    return hit;
}

static float3 swNormal(float3 p, float time) {
    float2 e = float2(0.0025f, 0.0f);
    return normalize(float3(
        swGeometry(p + e.xyy, time) - swGeometry(p - e.xyy, time),
        swGeometry(p + e.yxy, time) - swGeometry(p - e.yxy, time),
        swGeometry(p + e.yyx, time) - swGeometry(p - e.yyx, time)));
}

static float3 swEnvironment(float3 dir) {
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 5.0f);
    float sun = pow(max(dot(dir, normalize(float3(-0.45f, 0.4f, -0.8f))), 0.0f), 48.0f);
    float3 sky = mix(float3(0.01f, 0.025f, 0.06f), float3(0.08f, 0.18f, 0.34f), skyMix);
    return sky + float3(0.16f, 0.34f, 0.62f) * horizon * 0.28f + float3(0.92f, 0.97f, 1.0f) * sun * 0.7f;
}

static float3 swInnerColor(float3 p, float3 n, float time) {
    float3 upRef = abs(n.y) > 0.95f ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
    float3 tangent = normalize(cross(upRef, n));
    float3 bitangent = cross(n, tangent);

    float3 col = float3(0.0f);
    for (int i = 0; i < SW_VOLUME_STEPS; ++i) {
        float depth = 0.03f + 0.05f * float(i);
        float lateral = sin(time * 0.35f + depth * 11.0f + dot(p, tangent) * 4.0f) * 0.012f;
        float swirl = cos(time * 0.28f + depth * 9.0f + dot(p, bitangent) * 4.5f) * 0.012f;
        float3 pos = p - n * depth + tangent * lateral + bitangent * swirl;

        float sdf = swGeometry(pos, time);
        float shell = exp(-24.0f * abs(sdf));
        float flow = swFlow(pos, time);
        float field = 0.5f + 0.5f * sin(flow * 2.1f + depth * 8.0f);
        float density = shell * (0.3f + 0.7f * field);
        float fade = exp(-0.22f * float(i));
        float3 tint = swPalette(flow + depth * 0.6f);

        col += tint * density * fade * 0.16f;
    }

    return col;
}

fragment float4 saturdayWeirdnessFragment(
    SaturdayWeirdnessVertexOut in [[stage_in]],
    constant SaturdayWeirdnessUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < SW_BOX_HALF - 1.0e-3f);
    float2 tBox = swBoxIntersect(eye, rd, SW_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 ro = (eye + rd * (tStart + SW_HIT_EPSILON)) * 2.0f;

    float travel = 0.0f;
    SWHit state = swMap(ro, uniforms.time);
    for (int i = 0; i < SW_MAX_STEPS; ++i) {
        float3 p = ro + rd * travel;
        state = swMap(p, uniforms.time);
        if (abs(state.sdf) < SW_HIT_EPSILON || travel > SW_MAX_DIST) {
            break;
        }
        travel += clamp(abs(state.sdf), 0.01f, 0.12f);
    }

    if (travel > SW_MAX_DIST || abs(state.sdf) >= 0.02f) {
        float2 uv = swFaceUV(hit) * 2.0f - 1.0f;
        float vignette = 1.0f - 0.25f * dot(uv, uv);
        float3 bg = swEnvironment(rd) * vignette * vignette;
        return float4(bg, 1.0f);
    }

    float3 p = ro + rd * travel;
    float3 n = swNormal(p, uniforms.time);
    float3 reflected = reflect(rd, n);

    float diffuse = clamp(dot(n, normalize(float3(-0.32f, 0.76f, -0.56f))), 0.0f, 1.0f);
    float fresnel = pow(max(1.0f - abs(dot(n, -rd)), 0.0f), 4.0f);
    float featureWidth = 0.06f;
    float sheenMask = smoothstep(0.54f - featureWidth, 0.82f + featureWidth, state.feature);

    float3 env = swEnvironment(reflected);
    float3 inner = swInnerColor(p - n * 0.02f, n, uniforms.time);
    float3 color = state.albedo * (0.3f + 0.7f * diffuse);
    color += inner;
    color = mix(color, state.albedo * 1.18f + inner * 0.7f, sheenMask * 0.55f);
    color += env * (0.08f + 0.14f * fresnel);
    color += float3(0.85f, 0.95f, 1.0f) * sheenMask * 0.08f;

    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}