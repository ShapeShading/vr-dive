// BlueFlowerShaders.metal
// Adapted from ShaderToy "Blue Flower".
// Source: https://www.shadertoy.com/view/ttG3Dd
//
// Metal adaptation notes:
// - The original shader layered many petal shells directly in screen space.
//   This version converts the effect into a world-space SDF ray march that is
//   sampled from the real per-eye view ray after intersecting a 2 m cube.
// - Outside the cube, marching starts at the visible cube surface; inside the
//   cube, marching starts at the eye.
// - The flower field continues beyond the container entry plane, so the
//   simulated content is not clipped to the cube volume.
// - GLSL matrix macros and `fwidth`-based edge blending are rewritten as
//   explicit Metal helpers and shell thickness terms.

#include <metal_stdlib>
using namespace metal;

struct BlueFlowerUniforms {
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

struct BlueFlowerVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct BFHitInfo {
    float distance;
    float radius;
    float3 rotatedPoint;
    float3 normal;
};

static constant float3 BF_BOX_HALF = float3(1.0f);
static constant float3 BF_BACKGROUND = float3(0.8f, 0.85f, 0.9f);
static constant float3 BF_LIGHT = float3(0.26726124f, 0.53452248f, 0.80178373f);
static constant float3 BF_LIGHT_COLOR = float3(0.9f, 0.8f, 0.5f);
static constant int BF_LAYERS = 20;
static constant int BF_TRACE_STEPS = 96;
static constant float BF_TRACE_EPSILON = 0.0012f;
static constant float BF_HIT_EPSILON = 0.0015f;
static constant float BF_MAX_DISTANCE = 5.0f;
static constant float BF_SHELL_THICKNESS = 0.012f;

vertex BlueFlowerVertexOut blueFlowerVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant BlueFlowerUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    BlueFlowerVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 bfRotate2D(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float3 bfHash33(float3 p3) {
    p3 = fract(p3 * float3(0.1031f, 0.1030f, 0.0973f));
    p3 += dot(p3, p3.yxz + 33.33f);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

static float2 bfFaceUV(float3 p) {
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

static float2 bfBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float3 bfRotatePoint(float3 p, float time, int layerIndex) {
    float localTime = time * 0.05f + float(layerIndex) * 0.3f;
    p.zy = bfRotate2D(p.zy, 2.2f);
    p.xz = bfRotate2D(p.xz, -0.1f);
    p.yx = bfRotate2D(p.yx, localTime);
    p.xz = bfRotate2D(p.xz, sin(localTime * 8.145f) * 0.2f);
    p.zy = bfRotate2D(p.zy, sin(localTime * 6.587f) * 0.2f);
    return p;
}

static BFHitInfo bfLayerInfo(float3 p, float time, int layerIndex) {
    BFHitInfo info;
    float layer = float(layerIndex);
    float radius = (layer * layer + 1.0f) * 0.05f;
    float3 pr = bfRotatePoint(p, time, layerIndex);
    float at = atan2(pr.x, pr.y);
    float petalRadius = (sin(at * 5.0f) * 0.6f + 0.2f) * radius;

    float shell = abs(length(p) - radius) - BF_SHELL_THICKNESS;
    float petals = abs(pr.z) - petalRadius;
    info.distance = max(shell, petals);
    info.radius = radius;
    info.rotatedPoint = pr;
    info.normal = normalize(p);
    return info;
}

static BFHitInfo bfMap(float3 p, float time) {
    BFHitInfo best;
    best.distance = 1.0e9f;
    best.radius = 0.0f;
    best.rotatedPoint = p;
    best.normal = float3(0.0f, 1.0f, 0.0f);

    for (int layer = 0; layer < BF_LAYERS; ++layer) {
        BFHitInfo candidate = bfLayerInfo(p, time, layer);
        if (candidate.distance < best.distance) {
            best = candidate;
        }
    }
    return best;
}

static float bfDistance(float3 p, float time) {
    return bfMap(p, time).distance;
}

static float3 bfCalcNormal(float3 p, float time) {
    float e = 0.002f;
    return normalize(float3(
        bfDistance(p + float3(e, 0.0f, 0.0f), time) - bfDistance(p - float3(e, 0.0f, 0.0f), time),
        bfDistance(p + float3(0.0f, e, 0.0f), time) - bfDistance(p - float3(0.0f, e, 0.0f), time),
        bfDistance(p + float3(0.0f, 0.0f, e), time) - bfDistance(p - float3(0.0f, 0.0f, e), time)));
}

fragment float4 blueFlowerFragment(
    BlueFlowerVertexOut in [[stage_in]],
    constant BlueFlowerUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < BF_BOX_HALF - 1.0e-3f);
    float2 tBox = bfBoxIntersect(eye, rd, BF_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 ro = eye + rd * (tStart + BF_TRACE_EPSILON);

    float zoom = exp(-uniforms.time * 0.1f);
    float scale = mix(20.0f, 70.0f, zoom) / 55.0f;
    ro *= scale;

    float totalDistance = 0.0f;
    float stepDistance = 0.0f;
    float3 pos = ro;
    BFHitInfo info;
    info.distance = 1.0e9f;
    int stepsTaken = 0;
    bool didHit = false;

    for (int step = 0; step < BF_TRACE_STEPS; ++step) {
        stepsTaken = step;
        pos = ro + rd * totalDistance;
        info = bfMap(pos, uniforms.time);
        stepDistance = max(info.distance * 0.6f, BF_TRACE_EPSILON);
        if (info.distance < BF_HIT_EPSILON) {
            didHit = true;
            break;
        }
        totalDistance += stepDistance;
        if (totalDistance > BF_MAX_DISTANCE) {
            break;
        }
    }

    float2 q = bfFaceUV(hit);
    float3 color = BF_BACKGROUND;

    if (didHit) {
        float3 normal = bfCalcNormal(pos, uniforms.time);
        float at = atan2(info.rotatedPoint.x, info.rotatedPoint.y);
        float stripe = cos(at * 5.0f) * 20.0f;
        stripe *= smoothstep(0.1f, 0.0f, abs(stripe / 30.0f));
        float dotl = max(0.0f, dot(BF_LIGHT, normal) + stripe * 0.05f);

        float ao = 1.0f - min(1.0f, exp(pos.z / max(info.radius, 1.0e-4f) - 1.0f));
        float distFog = max(0.0f, 2.0f - info.rotatedPoint.z);
        float3 albedo = float3(0.2f, 0.3f, 0.8f);
        color = albedo * BF_LIGHT_COLOR * dotl * 3.0f + albedo * BF_BACKGROUND * 0.4f * ao;
        color = mix(BF_BACKGROUND, color, exp(-distFog * 0.1f));
    }

    color = pow(clamp(color, 0.0f, 1.0f), float3(1.0f / 2.2f));
    float2 uu = q - 0.5f;
    color = mix(color, float3(0.0f), dot(uu, uu) * 0.5f);
    color += (
        bfHash33(float3(hit.x * 256.0f, hit.y * 256.0f, uniforms.time + hit.z * 256.0f)) -
        0.5f) * 0.02f;
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}