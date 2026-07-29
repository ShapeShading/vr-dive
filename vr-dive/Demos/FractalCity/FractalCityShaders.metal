// FractalCityShaders.metal
// "Fractal city" — cube-container adaptation of ShaderToy 3ljyWz.
// Source: https://www.shadertoy.com/view/3ljyWz
// Source note: camera phase data and fold-based distance estimator come from the linked shader.

#include <metal_stdlib>
using namespace metal;

struct FractalCityUniforms {
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

struct FractalCityVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct FCPhase {
    float interval;
    float3 pos0;
    float3 pos1;
    float3 dir0;
    float3 dir1;
    float up;
};

struct FCState {
    float timeInPhase;
    float phaseT;
    FCPhase phase;
};

static constant float FC_ZOOM = 2.5f;
static constant float FC_HIT_EPS = 0.001f;
static constant float FC_MAX_H = 8.0f;
static constant int FC_MAX_STEPS = 100;
static constant float3 FC_BOX_HALF = float3(1.0f);
static constant FCPhase FC_PHASES[] = {
    {8.0f, float3(0.9f, 1.2f, 0.4f),  float3(0.6f, 1.0f, 0.8f),  float3(1.0f, 0.0f, 1.0f), float3(1.0f, 1.0f, 0.0f),  0.0f},
    {9.0f, float3(0.0f, 0.3f, 0.6f),  float3(0.0f, 0.0f, 0.6f),  float3(0.0f, 1.0f, 1.0f), float3(1.0f, 1.0f, 1.0f),  2.0f},
    {8.0f, float3(0.0f, 0.0f, 0.4f),  float3(0.0f, 0.0f, 1.2f),  float3(1.0f, 0.0f, 1.0f), float3(1.0f, 1.0f, 1.0f), -3.0f},
    {8.0f, float3(0.0f, 0.4f, 0.7f),  float3(0.4f, 0.0f, 0.7f),  float3(1.0f, 1.0f, 0.0f), float3(0.0f, 1.0f, 1.0f),  0.0f},
    {7.0f, float3(0.6f, 0.6f, 0.3f),  float3(0.6f, 0.8f, 0.0f),  float3(1.0f, 0.0f, 1.0f), float3(1.0f, 1.0f, 0.0f),  3.0f},
    {9.0f, float3(-0.8f, 0.4f, 0.6f), float3(-0.8f, 0.6f, 0.0f), float3(0.0f, 0.0f, 1.0f), float3(1.0f, 0.0f, 1.0f),  1.0f},
};

vertex FractalCityVertexOut fractalCityVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant FractalCityUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    FractalCityVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 fcFold45(float2 p) {
    return (p.y > p.x) ? p.yx : p;
}

static float fcMap(float3 p) {
    const float scale = 2.1f;
    const float off0 = 0.8f;
    const float off1 = 0.3f;
    const float off2 = 0.83f;
    const float3 off = float3(2.0f, 0.2f, 0.1f);
    float s = 1.0f;

    for (int i = 0; i < 20; ++i) {
        p.xy = abs(p.xy);
        p.xy = fcFold45(p.xy);
        p.y -= off0;
        p.y = -abs(p.y);
        p.y += off0;
        p.x += off1;
        p.xz = fcFold45(p.xz);
        p.x -= off2;
        p.xz = fcFold45(p.xz);
        p.x += off1;
        p -= off;
        p *= scale;
        p += off;
        s *= scale;
    }

    return length(p) / s;
}

static float2 fcBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 fcFaceUV(float3 p) {
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

static FCState fcPhaseState(float time) {
    FCState state;
    float cycle = 0.0f;
    constexpr int count = int(sizeof(FC_PHASES) / sizeof(FCPhase));
    for (int i = 0; i < count; ++i) {
        cycle += FC_PHASES[i].interval;
    }

    float wrapped = fmod(time, cycle);
    if (wrapped < 0.0f) {
        wrapped += cycle;
    }

    float accum = 0.0f;
    state.phase = FC_PHASES[0];
    state.timeInPhase = wrapped;
    state.phaseT = 0.0f;

    for (int i = 0; i < count; ++i) {
        float nextAccum = accum + FC_PHASES[i].interval;
        if (wrapped <= nextAccum || i == count - 1) {
            state.phase = FC_PHASES[i];
            state.timeInPhase = wrapped - accum;
            state.phaseT = clamp((wrapped - accum) / max(FC_PHASES[i].interval, 1.0e-4f), 0.0f, 1.0f);
            return state;
        }
        accum = nextAccum;
    }

    return state;
}

fragment float4 fractalCityFragment(
    FractalCityVertexOut in [[stage_in]],
    constant FractalCityUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 rdLocal = normalize(surfacePos - eye);

    bool insideCube = all(abs(eye) < FC_BOX_HALF - 1.0e-3f);
    float2 tCube = fcBoxIntersect(eye, rdLocal, FC_BOX_HALF);
    if (!insideCube && tCube.x > tCube.y) {
        discard_fragment();
    }

    float tStart = insideCube ? 0.0f : max(tCube.x, 0.0f);
    float3 localOrigin = eye + rdLocal * (tStart + 0.001f);

    FCState state = fcPhaseState(uniforms.time);
    FCPhase phase = state.phase;
    float t = state.phaseT;

    float3 baseRo = mix(phase.pos0, phase.pos1, t) * FC_ZOOM;
    float3 w = normalize(mix(phase.dir0, phase.dir1, t));
    float3 up = float3(sin(phase.up), cos(phase.up), 0.0f);
    float3 u = normalize(cross(w, up));
    float3 v = cross(u, w);

    float3 ro = baseRo + (localOrigin.x * u + localOrigin.y * v + localOrigin.z * w) * 1.2f;
    float3 rd = normalize(rdLocal.x * u + rdLocal.y * v + rdLocal.z * w);

    float h = 0.0f;
    float d = 0.0f;
    int stepCount = 1;
    float3 p = ro;
    bool hit = false;
    for (int i = 1; i < FC_MAX_STEPS; ++i) {
        stepCount = i;
        p = ro + rd * h;
        float3 samplePoint = p / FC_ZOOM;
        d = fcMap(samplePoint);
        if (d < FC_HIT_EPS) {
            hit = true;
            p = samplePoint;
            break;
        }
        if (h > FC_MAX_H) {
            p = samplePoint;
            break;
        }
        h += d;
    }

    float3 col = 30.0f * (cos(p * 1.2f) * 0.5f + 0.5f) / float(max(stepCount, 1));
    if (!hit) {
        float horizon = pow(max(1.0f - abs(rdLocal.y), 0.0f), 4.0f);
        float3 bg = mix(float3(0.008f, 0.012f, 0.022f), float3(0.045f, 0.06f, 0.09f), clamp(rdLocal.y * 0.5f + 0.5f, 0.0f, 1.0f));
        col = mix(bg, col, 0.35f);
        col += horizon * float3(0.03f, 0.025f, 0.05f);
    }

    float fog = smoothstep(2.0f, FC_MAX_H, h);
    col = mix(col, float3(0.012f, 0.016f, 0.03f), fog * 0.45f);
    col = clamp(col, 0.0f, 1.0f);

    float2 faceUV = fcFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.18f * dot(faceUV, faceUV);
    col = pow(col, float3(0.85f));
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}