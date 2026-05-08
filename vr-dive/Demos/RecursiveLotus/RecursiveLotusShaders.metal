// RecursiveLotusShaders.metal
// Adapted from ShaderToy "Recursive Lotus".
// Source: https://www.shadertoy.com/view/3d2Szm
//
// Original description:
// Self-similar flower based on the log-spherical mapping.
// Accompanying blog post: https://www.osar.fr/notes/logspherical/
//
// Metal adaptation notes:
// - The original shader constructed a synthetic orbit camera from fragCoord.
//   This version reconstructs the real per-eye world ray, intersects it with a
//   2 m cube container, and starts marching at the visible cube surface or at
//   the eye when the viewer is inside the cube.
// - The log-spherical tiled field is evaluated beyond the cube entry plane, so
//   the simulated lotus is not clipped to the container volume.
// - GLSL globals and inout rotations are rewritten as explicit Metal helpers.

#include <metal_stdlib>
using namespace metal;

struct RecursiveLotusUniforms {
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

struct RecursiveLotusVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct RLParams {
    float density;
    float height;
    float gTime;
    float vcut;
    float lpscale;
    float cameraY;
    float cameraTy;
    float interpos;
    float shorten;
    float lineWidth;
    float rotXY;
    float rotYZ;
    float radius;
    float rhoOffset;
};

static constant float RL_PI = 3.14159265358979323846f;
static constant float3 RL_BOX_HALF = float3(1.0f);
static constant float RL_TRACE_EPSILON = 0.0015f;
static constant float RL_HIT_EPSILON = 0.0001f;
static constant float RL_TMAX = 4.8f;
static constant float RL_SCENE_SCALE = 1.45f;
static constant int RL_MAX_STEPS = 64;

vertex RecursiveLotusVertexOut recursiveLotusVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant RecursiveLotusUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    RecursiveLotusVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float rlOsc(float time, float v1, float v2) {
    return (sin(time * 0.25f) * 0.5f + 0.5f) * (v2 - v1) + v1;
}

static RLParams rlMakeParams(float time) {
    RLParams params;
    params.gTime = time + rlOsc(time, 0.0f, 4.0f);
    params.density = 26.0f;
    params.height = rlOsc(time, 0.0f, 0.41f);
    params.cameraY = rlOsc(time, 0.4f, 1.07f);
    params.vcut = floor(params.density * 0.25f) * 2.0f + 0.9f;
    params.lpscale = floor(params.density) / RL_PI;
    params.cameraTy = -0.17f;
    params.interpos = -0.5f;
    params.shorten = 1.0f;
    params.lineWidth = 0.017f;
    params.rotXY = 0.0f;
    params.rotYZ = 0.785f;
    params.radius = 0.05f;
    params.rhoOffset = 0.0f;
    return params;
}

static float2 rlRotateAxis(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return c * p + s * float2(p.y, -p.x);
}

static float2 rlFaceUV(float3 p) {
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

static float2 rlBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float rlGain(float x, float k) {
    float a = 0.5f * pow(2.0f * ((x < 0.5f) ? x : 1.0f - x), k);
    return (x < 0.5f) ? a : 1.0f - a;
}

static float3 rlGain(float3 v, float k) {
    return float3(rlGain(v.x, k), rlGain(v.y, k), rlGain(v.z, k));
}

static void rlTile(
    float3 p,
    thread const RLParams &params,
    thread float3 &sp,
    thread float3 &tp,
    thread float3 &rp,
    thread float &mul)
{
    float r = max(length(p), 1.0e-5f);
    float3 q = float3(
        log(r),
        acos(clamp(p.y / r, -1.0f, 1.0f)),
        atan2(p.z, p.x));

    float xshrink =
        1.0f / (abs(q.y - RL_PI) + 1.0e-4f) +
        1.0f / (abs(q.y) + 1.0e-4f) -
        1.0f / RL_PI;

    q.y += params.height;
    q.z += q.x * 0.3f;
    mul = r / max(params.lpscale * xshrink, 1.0e-5f);
    q *= params.lpscale;
    sp = q;

    q.x -= params.rhoOffset + params.gTime;
    q = fract(q * 0.5f) * 2.0f - 1.0f;
    q.x *= xshrink;
    tp = q;
    q.xy = rlRotateAxis(q.xy, params.rotXY);
    q.yz = rlRotateAxis(q.yz, params.rotYZ);
    rp = q;
}

static float rlSdf(float3 p, thread const RLParams &params) {
    float3 sp, tp, rp;
    float mul;
    rlTile(p, params, sp, tp, rp, mul);

    float spheres = abs(rp.x) - 0.012f;
    float leaves = max(spheres, max(-rp.y, rp.z));
    leaves = max(leaves, params.vcut - sp.y);
    spheres = max(spheres, params.vcut - sp.y + 1.07f);
    float ret = min(leaves, spheres);

    float3 pi = rp;
    pi.x += params.interpos;
    float interS = abs(pi.x) - 0.02f;
    float interL = max(interS, max(-rp.y, rp.z));
    interL = max(interL, params.vcut - sp.y + 2.0f);
    interS = max(interS, params.vcut - sp.y + 3.0f);
    ret = min(ret, min(interL, interS));

    float ol = abs(rp.y) - params.radius * 0.8f;
    ol = min(ol, abs(rp.z) - params.radius * 0.8f);
    ret = max(ret, -ol);

    return ret * mul / params.shorten;
}

static float3 rlColor(float3 p, thread const RLParams &params) {
    float3 sp, tp, rp;
    float mul;
    rlTile(p, params, sp, tp, rp, mul);

    float ol = abs(rp.y) - params.radius;
    ol = min(ol, abs(rp.z) - params.radius);

    float3 pi = rp;
    pi.x += params.interpos;
    float inter = abs(pi.x) - 0.02f;
    inter = max(inter, params.vcut - sp.y + 2.0f);

    float dark = smoothstep(params.density * 0.25f, params.density * 0.5f, params.density - sp.y);
    dark *= dark;

    if (ol < params.lineWidth) {
        return float3(0.6f, 0.6f, 0.8f) * dark;
    }
    if (inter < 0.02f) {
        return float3(0.1f, 0.35f, 0.05f) * dark;
    }
    return float3(0.1f, 0.15f, 0.25f) * dark;
}

static float3 rlCalcNormal(float3 pos, thread const RLParams &params) {
    float2 e = float2(1.0f, -1.0f) * 0.5773f;
    const float eps = 0.0005f;
    return normalize(
        e.xyy * rlSdf(pos + e.xyy * eps, params) +
        e.yyx * rlSdf(pos + e.yyx * eps, params) +
        e.yxy * rlSdf(pos + e.yxy * eps, params) +
        e.xxx * rlSdf(pos + e.xxx * eps, params));
}

fragment float4 recursiveLotusFragment(
    RecursiveLotusVertexOut in [[stage_in]],
    constant RecursiveLotusUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < RL_BOX_HALF - 1.0e-3f);
    float2 tBox = rlBoxIntersect(eye, rd, RL_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 marchOrigin = eye + rd * (tStart + RL_TRACE_EPSILON);

    RLParams params = rlMakeParams(uniforms.time);
    float3 ro = marchOrigin * RL_SCENE_SCALE;
    float3 sceneRd = rd;

    float angle = 0.11f * sin(params.gTime * 0.05f);
    ro.xz = float2(cos(angle) * ro.x - sin(angle) * ro.z, sin(angle) * ro.x + cos(angle) * ro.z);
    float2 rotatedXZ = float2(
        cos(angle) * sceneRd.x - sin(angle) * sceneRd.z,
        sin(angle) * sceneRd.x + cos(angle) * sceneRd.z);
    sceneRd = normalize(float3(rotatedXZ.x, sceneRd.y, rotatedXZ.y));

    float2 q = rlFaceUV(hit);
    float3 bg = float3(0.06f, 0.08f, 0.11f) * 0.3f;
    bg *= 1.0f - smoothstep(0.1f, 2.0f, length(q * 2.0f - 1.0f));

    float3 tot = bg;
    float t = 0.0f;
    float3 pos = ro;
    int iout = 0;
    for (int i = 0; i < RL_MAX_STEPS; ++i) {
        pos = ro + t * sceneRd;
        float h = rlSdf(pos, params);
        if (h < RL_HIT_EPSILON || t > RL_TMAX) {
            break;
        }
        t += h;
        iout = i;
    }

    float fSteps = float(iout) / float(RL_MAX_STEPS);
    float3 col = float3(0.0f);
    if (t < RL_TMAX) {
        float3 nor = rlCalcNormal(pos, params);
        float dif = clamp(dot(nor, float3(0.57703f)), 0.0f, 1.0f);
        float amb = 0.5f + 0.5f * dot(nor, float3(0.0f, 1.0f, 0.0f));
        float3 lotus = rlColor(pos, params);
        col = lotus * amb + lotus * dif;
    }

    float gloamt = smoothstep(0.04f, 0.5f, length(pos));
    float gainPre = 1.0f - gloamt * 0.6f;
    float gainK = 1.5f + gloamt * 2.5f;
    col += rlGain(fSteps * float3(0.7f, 0.8f, 0.9f) * gainPre, gainK);

    col = mix(col, bg, smoothstep(0.2f + params.cameraY, 1.6f + params.cameraY, t));
    col = sqrt(max(col, 0.0f));
    tot += col;
    return float4(clamp(tot, 0.0f, 1.0f), 1.0f);
}