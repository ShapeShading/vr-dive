// ApollonianTwistShaders.metal
// 3D cube-container adaptation of a twisted Apollian field.
//
// This combines the existing 3D Apollonian raymarch structure with the
// rotating 4D Apollian fold used by the planar "Apollian with a twist" code.

#include <metal_stdlib>
using namespace metal;

struct ApollonianTwistUniforms {
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

struct ApollonianTwistVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct ApollonianTwistMapData {
    float dist;
    float4 trap;
    float detail;
};

static constant float3 AT_BOX_HALF = float3(1.0f);
static constant float AT_SCENE_SCALE = 1.3125f;
static constant float AT_MAX_DISTANCE = 7.5f;
static constant int AT_MAX_TRACE_STEPS = 190;
static constant int AT_APOLLIAN_ITERS = 12;

vertex ApollonianTwistVertexOut apollonianTwistVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant ApollonianTwistUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    ApollonianTwistVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 atRotate(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float atPsin(float x) {
    return 0.5f + 0.5f * sin(x);
}

static float atTanhApprox(float x) {
    float x2 = x * x;
    return clamp(x * (27.0f + x2) / (27.0f + 9.0f * x2), -1.0f, 1.0f);
}

static float2 atBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float atApollian(float4 p, float s, thread float4 &trap, thread float &detail) {
    float scale = 1.0f;
    trap = float4(1000.0f);
    detail = 0.0f;

    for (int i = 0; i < AT_APOLLIAN_ITERS; ++i) {
        p = -1.0f + 2.0f * fract(0.5f * p + 0.5f);
        float r2 = max(dot(p, p), 1.0e-5f);
        trap = min(trap, float4(abs(p.x), abs(p.y), abs(p.z), r2));
        float iterWeight = float(i + 1) / float(AT_APOLLIAN_ITERS);
        detail = max(detail, iterWeight * exp(-1.25f * r2));
        float k = s / r2;
        p *= k;
        scale *= k;
    }

    detail = clamp(max(detail, clamp(log2(max(scale, 1.0f)) / 24.0f, 0.0f, 1.0f)), 0.0f, 1.0f);

    return abs(p.y) / scale;
}

static ApollonianTwistMapData atCluster(
    float3 p,
    float time,
    float phase,
    float thickness,
    float radius)
{
    float tm = 0.22f * time;
    float3 q = p;
    // Remove the local position-coupled twist terms that generated broad ripple bands.
    q.xy = atRotate(q.xy, tm * 0.40f + 0.55f * phase);
    q.yz = atRotate(q.yz, tm * 0.24f - 0.28f * phase);
    q.xz = atRotate(q.xz, tm * 0.16f + 0.22f * phase);

    float r = 0.32f;
    float3 off = float3(
        r * atPsin(tm * sqrt(3.0f) + 0.9f * phase),
        r * atPsin(tm * sqrt(1.5f) - 0.7f * phase),
        r * atPsin(tm * sqrt(2.0f) + 0.5f * phase));

    float4 pp = float4(q + off, 0.0f);
    pp.w = 0.055f * (1.0f - atTanhApprox(0.82f * length(pp.xyz)));
    pp.yz = atRotate(pp.yz, tm * 0.52f + 0.14f * phase);
    pp.xz = atRotate(pp.xz, tm * 0.35f - 0.10f * phase);
    pp.xw = atRotate(pp.xw, -tm * 0.46f + 0.38f * phase);
    pp.yw = atRotate(pp.yw, tm * 0.64f - 0.22f * phase);

    const float zoom = 4.10f;
    pp /= zoom;

    float4 trap;
    float detail;
    float fractal = atApollian(pp, 1.24f, trap, detail) * zoom - thickness;
    float bound = length(p) - radius;

    ApollonianTwistMapData result;
    result.dist = max(fractal, bound);
    result.trap = trap;
    result.detail = detail;
    return result;
}

static ApollonianTwistMapData atMap(float3 p, float time) {
    ApollonianTwistMapData result = atCluster(p, time, 0.0f, 0.0048f, 1.08f);

    float3 sat1Center = float3(1.05f, 0.28f, -0.22f);
    float sat1Scale = 0.34f;
    float3 sat1Local = (p - sat1Center) / sat1Scale;
    float sat1Gate = (length(sat1Local) - 1.05f) * sat1Scale;
    if (sat1Gate < 0.20f) {
        ApollonianTwistMapData sat1 = atCluster(sat1Local, time, 1.7f, 0.0040f, 0.92f);
        sat1.dist *= sat1Scale;
        if (sat1.dist < result.dist) {
            result = sat1;
        }
    }

    float3 sat2Center = float3(-0.90f, -0.42f, 0.30f);
    float sat2Scale = 0.29f;
    float3 sat2Local = (p - sat2Center) / sat2Scale;
    float sat2Gate = (length(sat2Local) - 1.05f) * sat2Scale;
    if (sat2Gate < 0.18f) {
        ApollonianTwistMapData sat2 = atCluster(sat2Local, time, -2.0f, 0.0038f, 0.88f);
        sat2.dist *= sat2Scale;
        if (sat2.dist < result.dist) {
            result = sat2;
        }
    }
    return result;
}

static float atMapDistance(float3 p, float time) {
    return atMap(p, time).dist;
}

static float atTrace(
    float3 ro,
    float3 rd,
    float time,
    float maxDistance,
    thread float4 &trapOut)
{
    float t = 0.0f;
    trapOut = float4(1000.0f);

    for (int i = 0; i < AT_MAX_TRACE_STEPS; ++i) {
        ApollonianTwistMapData hit = atMap(ro + rd * t, time);
        trapOut = min(trapOut, hit.trap);
        // The compact outer container reduces shaded area at the same distance;
        // in exchange we keep a denser fold count here.
        float precis = 0.00035f + 0.00022f * t;
        if (hit.dist < precis || t > maxDistance || t > AT_MAX_DISTANCE) {
            break;
        }
        t += clamp(hit.dist * 0.68f, 0.0025f, 0.14f);
    }

    if (t > min(maxDistance, AT_MAX_DISTANCE)) {
        return -1.0f;
    }
    return t;
}

static float3 atCalcNormal(float3 pos, float time, float eps) {
    float2 e = float2(eps, -eps);
    return normalize(
        e.xyy * atMapDistance(pos + e.xyy, time) +
        e.yyx * atMapDistance(pos + e.yyx, time) +
        e.yxy * atMapDistance(pos + e.yxy, time) +
        e.xxx * atMapDistance(pos + e.xxx, time));
}

static float3 atBackground(float3 rd) {
    float up = rd.y * 0.5f + 0.5f;
    float3 sky = mix(float3(0.00012f, 0.00014f, 0.00018f), float3(0.0010f, 0.0012f, 0.0016f), up);
    float glow = pow(max(1.0f - abs(rd.y), 0.0f), 3.2f);
    sky += glow * float3(0.00065f, 0.00040f, 0.00095f);
    return sqrt(max(sky, 0.0f));
}

static float3 atRender(float3 ro, float3 rd, float time, float maxDistance) {
    float4 trap;
    float t = atTrace(ro, rd, time, maxDistance, trap);
    if (t <= 0.0f) {
        return atBackground(rd);
    }

    float3 pos = ro + rd * t;
    ApollonianTwistMapData hit = atMap(pos, time);
    float3 nor = atCalcNormal(pos, time, max(0.0014f, 0.00045f * t));

    float3 light1 = normalize(float3(0.55f, 0.72f, -0.42f));
    float3 light2 = normalize(float3(-0.48f, 0.28f, 0.83f));
    float key = clamp(dot(nor, light1), 0.0f, 1.0f);
    float fill = clamp(0.2f + 0.8f * dot(nor, light2), 0.0f, 1.0f);
    float rim = pow(1.0f - max(dot(-rd, nor), 0.0f), 2.5f);

    float ao = pow(clamp(trap.w * 1.55f, 0.0f, 1.0f), 0.72f);
    float detail = clamp(hit.detail, 0.0f, 1.0f);
    float3 deepGreen = float3(0.010f, 0.055f, 0.028f);
    float3 midGreen = float3(0.15f, 0.33f, 0.12f);
    float3 gold = float3(1.00f, 0.82f, 0.26f);
    float3 warmWhite = float3(0.985f, 0.985f, 0.965f);
    float3 base = mix(deepGreen, midGreen, sqrt(detail));
    float3 brightCore = mix(gold, warmWhite, smoothstep(0.82f, 1.0f, detail));
    float3 highlight = mix(midGreen, brightCore, pow(detail, 1.35f));

    float keyBand = 0.04f + 1.10f * pow(key, 1.48f);
    float fillBand = 0.02f + 0.13f * pow(fill, 1.14f);
    float whiteLift = smoothstep(0.86f, 1.0f, detail) * (0.30f + 0.70f * key) * ao;

    float3 col = base * keyBand * ao;
    col += highlight * fillBand * ao;
    col += highlight * rim * (0.025f + 0.12f * detail);
    col += brightCore * (0.05f + 0.18f * detail)
        * (exp(-18.0f * trap.x) + 0.45f * exp(-30.0f * trap.y));
    col = mix(col, warmWhite, 0.72f * whiteLift);
    col *= exp(-0.11f * t);
    float3 shadowFloor = base * (0.010f + 0.006f * ao) + float3(0.00055f, 0.00070f, 0.00055f);
    col = max((col - 0.17f) * 1.52f + 0.17f, shadowFloor);
    return sqrt(max(col, 0.0f));
}

fragment float4 apollonianTwistFragment(
    ApollonianTwistVertexOut in [[stage_in]],
    constant ApollonianTwistUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (cameraWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 viewDir = normalize(surfacePos - eye);

    bool insideOuter = all(abs(eye) < AT_BOX_HALF - 1.0e-3f);
    float2 tOuter = atBoxIntersect(eye, viewDir, AT_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float tEnd = tOuter.y;
    if (tEnd <= tStart) {
        discard_fragment();
    }

    float3 localOrigin = eye + viewDir * (tStart + 0.001f);
    float3 sceneOrigin = localOrigin * AT_SCENE_SCALE;
    float maxDistance = max((tEnd - tStart) * AT_SCENE_SCALE, 0.0f);

    float time = uniforms.time;
    float3 col = atRender(sceneOrigin, viewDir, time, maxDistance);

    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}