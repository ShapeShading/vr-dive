// MagneticLinesThatDrawInGoldShaders.metal
// "Magnetic lines that draw in gold" — cube-portal adaptation of Shadertoy "3cBXRy"
// Original: https://www.shadertoy.com/view/3cBXRy
// Source note from the original shader: "2025-3-14 / apollo".
//
// Metal adaptation notes:
// - The original shader is a screen-space ray marcher with mouse-driven
//   orientation tweaks. This version replaces that camera with the real per-eye
//   world ray from a 2 m cube portal.
// - Outside the cube, marching starts at the visible container surface.
// - Inside the cube, marching starts at the eye.
// - The authored magnetic-gold line structure remains unbounded in scene space,
//   so it is not clipped by the cube bounds.

#include <metal_stdlib>
using namespace metal;

struct MagneticLinesThatDrawInGoldUniforms {
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

struct MagneticLinesThatDrawInGoldVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float MLDG_PI = 3.1415926535f;
static constant float MLDG_MAX_TRACE_DISTANCE = 15.0f;
static constant int MLDG_MAX_TRACE_STEPS = 256;
static constant int MLDG_SHADOW_STEPS = 80;
static constant int MLDG_AO_STEPS = 24;
static constant float MLDG_SCENE_SCALE = 0.23f;
static constant float3 MLDG_SCENE_OFFSET = float3(0.0f, 0.02f, 0.0f);
static constant float3 MLDG_BOX_HALF = float3(1.0f);

vertex MagneticLinesThatDrawInGoldVertexOut magneticLinesThatDrawInGoldVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant MagneticLinesThatDrawInGoldUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    MagneticLinesThatDrawInGoldVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 mldgRotate(float2 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float2 mldgHash2(float n) {
    return fract(sin(float2(n, n + 1.0f)) * float2(43758.5453123f, 22578.1459123f));
}

static float mldgApollo(float3 p) {
    float j = 1.0f;
    float k;
    float rxy;
    float r0 = length(p) - 0.6f;

    for (int i = 0; i < 9; ++i) {
        p = 2.0f * clamp(p, -2.0f, 2.0f) - p;
        k = max(1.0f, 0.70968f / dot(p, p));
        p *= k;
        j = j * k + 0.05f;
    }

    rxy = length(p.xy);
    return max(r0, max(rxy - 0.92784f, abs(rxy * p.z) / max(length(p), 1.0e-5f)) / j - 1.0e-4f);
}

static float mldgMap(float3 p, float time) {
    p.xy = mldgRotate(p.xy, 2.0f);
    p.xz = mldgRotate(p.xz, MLDG_PI / 3.0f + 9.8f);
    p.yz = mldgRotate(p.yz, 0.516f + 8.4f);
    p.xy = mldgRotate(p.xy, time + 2.0f);
    return mldgApollo(p);
}

static float mldgCalcShadow(float3 ro, float3 rd, float k, float time) {
    float res = 1.0f;
    float t = 0.01f;
    for (int i = 0; i < MLDG_SHADOW_STEPS; ++i) {
        float3 pos = ro + t * rd;
        float h = mldgMap(pos, time);
        res = min(res, k * max(h, 0.0f) / t);
        if (res < 1.0e-4f || pos.y > 10.0f) {
            break;
        }
        t += clamp(h, 0.01f, 5.0f);
    }
    return res;
}

static float mldgCalcOcclusion(float3 pos, float3 nor, float ra, float time) {
    float occ = 0.0f;
    for (int i = 0; i < MLDG_AO_STEPS; ++i) {
        float fi = float(i);
        float h = 0.01f + 4.0f * pow(fi / float(MLDG_AO_STEPS - 1), 2.0f);
        float2 an = mldgHash2(ra + fi * 13.1f) * float2(3.14159f, 6.2831f);
        float3 dir = float3(sin(an.x) * sin(an.y), sin(an.x) * cos(an.y), cos(an.x));
        dir *= sign(dot(dir, nor));
        occ += clamp(5.0f * mldgMap(pos + h * dir, time) / h, -1.0f, 1.0f);
    }
    return clamp(occ / float(MLDG_AO_STEPS), 0.0f, 1.0f);
}

static float mldgTrace(float3 ro, float3 rd, float time, thread float &hitDistance) {
    float t = 0.0f;
    float d = 1.0f;
    for (int i = 0; i < MLDG_MAX_TRACE_STEPS && t < MLDG_MAX_TRACE_DISTANCE; ++i) {
        float3 p = ro + rd * t;
        d = mldgMap(p, time);
        if (d < 3.0e-4f) {
            hitDistance = t;
            return d;
        }
        t += d * 0.29f;
    }

    hitDistance = t;
    return d;
}

static float3 mldgNormal(float3 p, float d, float time) {
    float3 e = float3(0.0f, 1.0e-5f, 0.0f);
    return normalize(float3(
        mldgMap(p + e.yxx, time),
        mldgMap(p + e, time),
        mldgMap(p + e.xxy, time)) - d);
}

static float3 mldgBackground(float3 rd) {
    float t = clamp(rd.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float3 low = float3(0.01f, 0.007f, 0.003f);
    float3 high = float3(0.08f, 0.05f, 0.02f);
    float horizon = pow(1.0f - abs(rd.z), 3.0f);
    return mix(low, high, t) + horizon * float3(0.08f, 0.05f, 0.02f);
}

static float mldgBoxHit(float3 ro, float3 rd, float3 halfExtents, thread float3 &nn, bool entering) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 dr = 1.0f / rd;
    float3 n = ro * dr;
    float3 k = halfExtents * abs(dr);
    float3 pin = -k - n;
    float3 pout = k - n;
    float tin = max(pin.x, max(pin.y, pin.z));
    float tout = min(pout.x, min(pout.y, pout.z));
    if (tin > tout) {
        return -1.0f;
    }
    if (entering) {
        nn = -sign(rd) * step(pin.zxy, pin.xyz) * step(pin.yzx, pin.xyz);
        return tin;
    }
    nn = sign(rd) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);
    return tout;
}

fragment float4 magneticLinesThatDrawInGoldFragment(
    MagneticLinesThatDrawInGoldVertexOut in [[stage_in]],
    constant MagneticLinesThatDrawInGoldUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];
    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float sceneScale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / sceneScale;
    float3 rd = normalize(in.worldPos - camWorld);

    bool insideBox = all(abs(eye) < float3(0.999f));
    float3 faceNormal;
    float entryT = insideBox ? 0.0f : mldgBoxHit(eye, rd, MLDG_BOX_HALF, faceNormal, true);
    if (!insideBox && entryT < 0.0f) {
        discard_fragment();
    }

    float3 marchOrigin = insideBox ? (eye + rd * 0.002f) : (eye + rd * (entryT + 0.002f));
    float time = uniforms.time;
    float3 ro = marchOrigin * MLDG_SCENE_SCALE + MLDG_SCENE_OFFSET;

    float hitT;
    float d = mldgTrace(ro, rd, time, hitT);
    float3 color = mldgBackground(rd);

    if (d < 3.0e-4f && hitT < MLDG_MAX_TRACE_DISTANCE) {
        float3 p = ro + rd * hitT;
        float3 s = normalize(float3(-1.0f, 2.0f, -3.0f));
        float shd = mldgCalcShadow(p - rd * max(d, 1.0e-4f), s, 100.0f, time);
        float ao = mldgCalcOcclusion(p - rd * max(d, 1.0e-4f), s, 1.0f, time);
        float3 n = mldgNormal(p, d, time);
        float f = 0.5f + 0.5f * dot(n, s);
        float g = max(dot(n, s), 0.0f);
        float c = 1.0f + pow(f, 200.0f) - f * 0.3f;

        float3 gold = g * c;
        gold = mix(gold * float3(3.0f, 2.0f, 1.0f), float3(0.5f), 1.0f - 1.0f / max(1.0f, hitT * hitT * 0.01f));
        gold *= shd;
        gold *= min(0.2f * exp(-29.0f * p.z), 1.5f);
        gold *= mix(0.55f, 1.0f, ao);
        gold = mix(gold, float3(0.5f), smoothstep(5.0f, 15.0f, hitT));
        color = tanh(gold);
    }

    float3 surfacePos = insideBox ? eye : (eye + rd * entryT);
    float3 surfaceNormal = float3(0.0f, 0.0f, 1.0f);
    float3 absSurface = abs(surfacePos);
    if (absSurface.x > absSurface.y && absSurface.x > absSurface.z) {
        surfaceNormal = float3(sign(surfacePos.x), 0.0f, 0.0f);
    } else if (absSurface.y > absSurface.z) {
        surfaceNormal = float3(0.0f, sign(surfacePos.y), 0.0f);
    } else {
        surfaceNormal = float3(0.0f, 0.0f, sign(surfacePos.z));
    }

    float fresnel = pow(1.0f - max(dot(-rd, surfaceNormal), 0.0f), 2.0f);
    color += float3(0.08f, 0.05f, 0.02f) * fresnel * 0.06f;
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}