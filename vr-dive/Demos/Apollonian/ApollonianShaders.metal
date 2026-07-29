// ApollonianShaders.metal
// "apollonian" — cube-portal adaptation of Shadertoy "3l2czd"
// Original: https://www.shadertoy.com/view/3l2czd
// Original authorship note in source: created by inigo quilez - iq/2013,
// modified by jorge2017a1. License: CC BY-NC-SA 3.0.
//
// Metal adaptation notes:
// - The original mainImage/mainVR shader uses a synthetic camera. This version
//   uses the real per-eye world ray from the 2 m cube container.
// - Outside the cube, marching starts at the visible cube surface; inside the
//   cube, marching starts at the eye.
// - The Apollonian field extends beyond the cube and is not clipped by the
//   container's back face.

#include <metal_stdlib>
using namespace metal;

struct ApollonianUniforms {
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

struct ApollonianVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct ApollonianMapData {
    float dist;
    float4 orb;
};

static constant float AP_MAX_DISTANCE = 30.0f;
static constant int AP_MAX_TRACE_STEPS = 200;
static constant float AP_SCENE_SCALE = 2.7f;
static constant float3 AP_SCENE_OFFSET = float3(1.64f, 2.4f, -0.6f);
static constant float3 AP_BOX_HALF = float3(1.0f);

vertex ApollonianVertexOut apollonianVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant ApollonianUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    ApollonianVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float apIntersectSDF(float distA, float distB) {
    return max(distA, distB);
}

static float apSdSphere(float3 p, float s) {
    return length(p) - s;
}

static float apBoxHit(float3 ro, float3 rd, float3 halfExtents, thread float3 &nn, bool entering) {
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

static ApollonianMapData apMap1(float3 p) {
    float scale = 1.0f;
    float4 orb = float4(1000.0f);

    for (int i = 0; i < 8; ++i) {
        p = -1.0f + 2.0f * fract(0.5f * p + 0.5f);

        float r2 = dot(p, p);
        orb = min(orb, float4(abs(p), r2));

        float k = 1.75f / r2;
        p *= k;
        scale *= k;
    }

    ApollonianMapData result;
    result.dist = 0.25f * abs(p.y) / scale;
    result.orb = orb;
    return result;
}

static ApollonianMapData apMap(float3 p) {
    ApollonianMapData fractal = apMap1(p);
    float sphere = apSdSphere(p, 2.0f);

    ApollonianMapData result;
    result.dist = apIntersectSDF(sphere, fractal.dist);
    result.orb = fractal.orb;
    return result;
}

static float apMapDistance(float3 p) {
    return apMap(p).dist;
}

static float apTrace(float3 ro, float3 rd, thread float4 &orbOut) {
    float t = 0.01f;
    orbOut = float4(1000.0f);

    for (int i = 0; i < AP_MAX_TRACE_STEPS; ++i) {
        float precis = 0.001f * t;
        ApollonianMapData hit = apMap(ro + rd * t);
        orbOut = hit.orb;
        if (hit.dist < precis || t > AP_MAX_DISTANCE) {
            break;
        }
        t += hit.dist;
    }

    if (t > AP_MAX_DISTANCE) {
        return -1.0f;
    }
    return t;
}

static float3 apCalcNormal(float3 pos, float t) {
    float precis = 0.001f * t;
    float2 e = float2(1.0f, -1.0f) * precis;
    return normalize(
        e.xyy * apMapDistance(pos + e.xyy) +
        e.yyx * apMapDistance(pos + e.yyx) +
        e.yxy * apMapDistance(pos + e.yxy) +
        e.xxx * apMapDistance(pos + e.xxx));
}

static float3 apRender(float3 ro, float3 rd, float anim) {
    float4 trap;
    float t = apTrace(ro, rd, trap);
    if (t <= 0.0f) {
        float horizon = pow(1.0f - abs(rd.y), 3.0f);
        float3 bg = mix(float3(0.005f, 0.007f, 0.012f), float3(0.02f, 0.03f, 0.06f), rd.y * 0.5f + 0.5f);
        return bg + horizon * float3(0.02f, 0.03f, 0.05f);
    }

    float3 pos = ro + t * rd;
    float3 nor = apCalcNormal(pos, t);

    float3 light1 = normalize(float3(0.577f, 0.577f, -0.577f));
    float3 light2 = normalize(float3(-0.707f, 0.0f, 0.707f));
    float key = clamp(dot(light1, nor), 0.0f, 1.0f);
    float bac = clamp(0.2f + 0.8f * dot(light2, nor), 0.0f, 1.0f);
    float amb = 0.7f + 0.3f * nor.y;
    float ao = pow(clamp(trap.w * 2.0f, 0.0f, 1.0f), 1.2f);

    float3 brdf = float3(0.40f) * amb * ao;
    brdf += float3(1.0f) * key * ao;
    brdf += float3(0.40f) * bac * ao;

    float3 rgb = float3(1.0f);
    rgb = mix(rgb, float3(1.0f, 0.80f, 0.2f), clamp(6.0f * trap.y, 0.0f, 1.0f));
    rgb = mix(rgb, float3(1.0f, 0.55f, 0.0f), pow(clamp(1.0f - 2.0f * trap.z, 0.0f, 1.0f), 8.0f));

    float3 col = rgb * brdf * exp(-0.2f * t);
    col += 0.06f * anim * pow(max(1.0f + dot(rd, nor), 0.0f), 4.0f) * float3(1.0f, 0.7f, 0.25f);
    return sqrt(max(col, 0.0f));
}

fragment float4 apollonianFragment(
    ApollonianVertexOut in [[stage_in]],
    constant ApollonianUniforms &uniforms [[buffer(0)]],
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
    float entryT = insideBox ? 0.0f : apBoxHit(eye, rd, AP_BOX_HALF, faceNormal, true);
    if (!insideBox && entryT < 0.0f) {
        discard_fragment();
    }

    float3 marchOrigin = insideBox ? (eye + rd * 0.002f) : (eye + rd * (entryT + 0.002f));

    float time = uniforms.time * 0.25f;
    float anim = 1.1f + 0.5f * smoothstep(-0.3f, 0.3f, cos(0.4f * time));

    float3 roScene = marchOrigin * AP_SCENE_SCALE + AP_SCENE_OFFSET;
    float3 color = apRender(roScene, rd, anim);

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

    float fresnel = pow(1.0f - max(dot(-rd, surfaceNormal), 0.0f), 2.2f);
    color += float3(0.04f, 0.08f, 0.14f) * fresnel * 0.08f;

    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}