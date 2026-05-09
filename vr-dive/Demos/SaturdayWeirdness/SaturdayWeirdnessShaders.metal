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
    float edge;
    float3 albedo;
    bool hit;
};

static constant float3 SW_BOX_HALF = float3(1.0f);
static constant float SW_MAX_DIST = 28.0f;
static constant float SW_HIT_EPSILON = 0.0015f;
static constant int SW_MAX_STEPS = 180;

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

static float swGyroid(float3 p) {
    return dot(sin(p), cos(p.zxy));
}

static float3 swPalette(float t) {
    return 0.45f + 0.4f * cos(6.2831853f * (float3(0.12f, 0.27f, 0.43f) + t * float3(1.0f, 0.8f, 0.6f)));
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

    float3 q = p * 1.55f;
    q.xz = swRotate2D(q.xz, time * 0.2f);
    q.yz = swRotate2D(q.yz, 0.65f + sin(time * 0.33f) * 0.25f);

    float core = swRoundBox(q, float3(0.7f), 0.28f);
    float gyroid = swGyroid(q * 3.1f + time * 0.7f);
    float grooves = sin(q.x * 7.0f) + sin(q.y * 7.6f) + sin(q.z * 6.8f);
    float detail = (abs(gyroid) - 0.24f) * 0.045f + grooves * 0.012f;
    float sdf = core + detail;

    float edgeSignal = abs(detail) + abs(gyroid) * 0.08f;
    float colorPhase = 0.14f * q.y + 0.1f * q.z + sin(time * 0.45f) * 0.2f;

    hit.distance = 0.0f;
    hit.sdf = sdf;
    hit.edge = edgeSignal;
    hit.albedo = swPalette(colorPhase);
    hit.hit = false;
    return hit;
}

static float3 swNormal(float3 p, float time) {
    float2 e = float2(0.0025f, 0.0f);
    return normalize(float3(
        swMap(p + e.xyy, time).sdf - swMap(p - e.xyy, time).sdf,
        swMap(p + e.yxy, time).sdf - swMap(p - e.yxy, time).sdf,
        swMap(p + e.yyx, time).sdf - swMap(p - e.yyx, time).sdf));
}

static float3 swEnvironment(float3 dir) {
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 5.0f);
    float sun = pow(max(dot(dir, normalize(float3(-0.45f, 0.4f, -0.8f))), 0.0f), 48.0f);
    float3 sky = mix(float3(0.015f, 0.02f, 0.035f), float3(0.16f, 0.2f, 0.28f), skyMix);
    return sky + float3(0.95f, 0.55f, 0.2f) * horizon * 0.35f + float3(1.0f, 0.92f, 0.74f) * sun;
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
        travel += clamp(abs(state.sdf), 0.008f, 0.14f);
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

    float diffuse = clamp(dot(n, normalize(float3(-0.4f, 0.7f, -0.6f))), 0.0f, 1.0f);
    float fresnel = pow(max(1.0f - abs(dot(n, -rd)), 0.0f), 4.0f);
    float edgeWidth = max(fwidth(state.edge), 0.01f);
    float edgeMask = 1.0f - smoothstep(0.06f - edgeWidth, 0.06f + edgeWidth, state.edge);

    float3 env = swEnvironment(reflected);
    float3 color = state.albedo * (0.2f + 0.8f * diffuse);
    color = mix(color, state.albedo * 1.6f + env * 0.25f, edgeMask);
    color += env * fresnel * 0.45f;

    float glow = exp(-0.08f * travel) * (0.25f + 0.75f * edgeMask);
    color += swPalette(0.18f * p.y - 0.1f * p.z + uniforms.time * 0.08f) * glow * 0.22f;

    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}