// NearLoxodromeShaders.metal
// Source adaptation: Shadertoy "Near Loxodrome"
// https://www.shadertoy.com/view/NcX3RX
//
// The original shader uses a set of helper macros and types to raymarch a
// spherical spiral / rails composition. This version expands those ideas into
// explicit Metal code and renders them through a 2 m cube portal.

#include <metal_stdlib>
using namespace metal;

struct NearLoxodromeUniforms {
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

struct NearLoxodromeVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct NLObject {
    float distance;
    float id;
    float3 position;
};

vertex NearLoxodromeVertexOut nearLoxodromeVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant NearLoxodromeUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    NearLoxodromeVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static constant float NL_PI = 3.14159265359f;
static constant float NL_FAR = 24.0f;
static constant float NL_PORTAL_DEPTH = 14.0f;
static constant float NL_EPS = 0.0008f;
static constant int NL_MAX_TRACE_STEPS = 176;
static constant int NL_MAX_SHADOW_STEPS = 40;
static constant float3 NL_BOX_HALF = float3(1.0f, 1.0f, 1.0f);
static constant float NL_SCENE_SCALE = 0.82f;

static constant float NL_ID_NONE = -1.0f;
static constant float NL_ID_FIGURE = 0.0f;
static constant float NL_ID_FIGURE_INSIDE = 1.0f;
static constant float NL_ID_FIGURE_OUTSIDE = 2.0f;
static constant float NL_ID_RAILS = 3.0f;
static constant float NL_ID_CENTER = 4.0f;

static float nlRoundBox(float3 p, float3 b, float r) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f) - r;
}

static float nlBoxHit(float3 ro, float3 rd, float3 halfExtents, thread float3 &nn, bool entering) {
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

static float nlRepeat(thread float &x, float c) {
    float cell = floor((x + 0.5f * c) / c);
    x = fmod(x + 0.5f * c, c) - 0.5f * c;
    return cell;
}

static float2 nlRepeat2(thread float2 &p, float c) {
    float2 cell = floor((p + 0.5f * c) / c);
    p = fmod(p + 0.5f * c, c) - 0.5f * c;
    return cell;
}

static float3 nlSpiralLocal(float3 p, float branches, float radius, float stepScale, float phase) {
    float r = max(length(p), 1e-4f);
    float latitude = asin(clamp(p.y / r, -1.0f, 1.0f));
    float longitude = atan2(p.z, p.x);

    float pitch = max(stepScale, 0.12f);
    float loxo = branches * log(max(tan(NL_PI * 0.25f + latitude * 0.5f), 1e-4f)) / pitch + phase;
    float branchPeriod = 2.0f * NL_PI / branches;
    float branchCoord = (longitude - loxo) / branchPeriod;
    float branchIndex = round(branchCoord);
    float deltaLon = (branchCoord - branchIndex) * branchPeriod;

    float radial = r - radius;
    float along = latitude * radius * 4.2f / max(stepScale, 0.2f)
                + branchIndex * branchPeriod * radius * max(cos(latitude), 0.18f);
    float across = deltaLon * radius * max(cos(latitude), 0.12f);
    return float3(radial, along, across);
}

static NLObject nlMakeObject(float distance, float id, float3 position) {
    NLObject object;
    object.distance = distance;
    object.id = id;
    object.position = position;
    return object;
}

static void nlUnion(thread NLObject &a, NLObject b) {
    if (b.distance < a.distance) {
        a = b;
    }
}

static float nlMap(float3 p, constant NearLoxodromeUniforms &uniforms, thread NLObject &object, thread float &glow) {
    float3 sceneP = p / NL_SCENE_SCALE;
    object = nlMakeObject(NL_FAR, NL_ID_NONE, sceneP);

    float radius = 1.0f;
    float branches = 3.0f;
    float stepScale = 0.5f + 0.30f * sin(0.1f * uniforms.time);
    float3 q = nlSpiralLocal(sceneP, branches, radius, stepScale, 0.2f * uniforms.time);

    NLObject figure = nlMakeObject(NL_FAR, NL_ID_FIGURE, q);
    {
        float slab = abs(q.z) - 0.075f;
        float inward = q.x;
        float d = max(slab, inward);
        float figureId = q.z < -0.01f ? NL_ID_FIGURE_INSIDE : (q.z > 0.01f ? NL_ID_FIGURE_OUTSIDE : NL_ID_FIGURE);
        figure = nlMakeObject(d, figureId, q);
    }
    nlUnion(object, figure);

    NLObject rails = nlMakeObject(NL_FAR, NL_ID_RAILS, q);
    {
        float3 rq = q;
        rq.x -= 0.2f;
        rq.x = abs(rq.x) - 0.115f;
        float d1 = length(rq.xz) - 0.024f;

        float ry = rq.y;
        nlRepeat(ry, 0.3f);
        rq.y = ry;
        float d2 = max(length(rq.yz) - 0.016f, rq.x);

        rails = nlMakeObject(min(d1, d2), NL_ID_RAILS, rq);
    }
    nlUnion(object, rails);

    NLObject center = nlMakeObject(NL_FAR, NL_ID_CENTER, sceneP);
    {
        float3 cq = sceneP;
        cq.y = abs(cq.y) - radius - 0.2f;
        float d = nlRoundBox(cq, float3(0.0f, 0.1f, 0.0f), 0.025f);
        glow += 0.001f / (0.01f + d * d);
        center = nlMakeObject(d, NL_ID_CENTER, cq);
    }
    nlUnion(object, center);

    if (object.id != NL_ID_NONE) {
        object.distance *= 0.4f * NL_SCENE_SCALE;
    }
    return object.distance;
}

static float nlMapOnly(float3 p, constant NearLoxodromeUniforms &uniforms) {
    NLObject object;
    float glow = 0.0f;
    return nlMap(p, uniforms, object, glow);
}

static float3 nlMapNormal(float3 p, constant NearLoxodromeUniforms &uniforms, float eps) {
    float2 e = float2(eps, -eps);
    float v1 = nlMapOnly(p + e.xxx, uniforms);
    float v2 = nlMapOnly(p + e.xyy, uniforms);
    float v3 = nlMapOnly(p + e.yxy, uniforms);
    float v4 = nlMapOnly(p + e.yyx, uniforms);
    return normalize(float3(v1 - v2 - v3 - v4) + 2.0f * float3(v2, v3, v4));
}

static float nlSoftShadow(float3 origin, float3 direction, constant NearLoxodromeUniforms &uniforms, float farDist, float k) {
    float shade = 1.0f;
    float t = 0.02f;
    for (int i = 0; i < NL_MAX_SHADOW_STEPS && t < farDist; ++i) {
        float d = abs(nlMapOnly(origin + direction * t, uniforms));
        shade = min(shade, smoothstep(0.0f, 1.0f, k * d / max(t, 1e-4f)));
        t += min(max(d, 0.01f), farDist / float(NL_MAX_SHADOW_STEPS) * 2.0f);
    }
    return min(max(shade, 0.0f) + 0.5f, 1.0f);
}

static float3 nlMaterial(NLObject object) {
    float3 q = object.position;
    if (object.id == NL_ID_FIGURE) {
        return float3(1.0f);
    }
    if (object.id == NL_ID_FIGURE_INSIDE) {
        float2 grid = q.xy;
        float2 cell = nlRepeat2(grid, 0.14f);
        return fmod(cell.x + cell.y, 2.0f) == 0.0f ? float3(1.0f) : float3(1.0f, 0.0f, 0.0f);
    }
    if (object.id == NL_ID_FIGURE_OUTSIDE) {
        float2 grid = q.xy;
        float2 cell = nlRepeat2(grid, 0.14f);
        return fmod(cell.x + cell.y, 2.0f) == 0.0f ? float3(1.0f) : float3(0.0f, 1.0f, 0.0f);
    }
    if (object.id == NL_ID_RAILS) {
        float railY = q.y;
        float cell = nlRepeat(railY, 0.14f);
        return fmod(cell, 2.0f) == 0.0f ? float3(0.5f) : float3(1.0f, 0.6f, 0.2f);
    }
    return float3(1.0f);
}

fragment float4 nearLoxodromeFragment(
    NearLoxodromeVertexOut in [[stage_in]],
    constant NearLoxodromeUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center = uniforms.objectCenter.xyz;
    float scale = uniforms.cubeScale;
    float3 eye = (camWorld - center) / scale;
    float3 rdWorld = normalize(in.worldPos - camWorld);
    float3 rd = rdWorld / max(scale, 1e-4f);
    float3 rdUnit = normalize(rd);

    bool insideBox = all(abs(eye) < (NL_BOX_HALF - 1e-3f));
    float3 entryNormal;
    float entryT = nlBoxHit(eye, rd, NL_BOX_HALF, entryNormal, !insideBox);
    if (entryT < 0.0f) {
        discard_fragment();
    }

    float3 entryPoint = eye + rd * entryT;
    float3 faceNormal = insideBox ? -entryNormal : entryNormal;
    float2 faceCoords = entryPoint.xy * faceNormal.z / NL_BOX_HALF.xy
                      + entryPoint.yz * faceNormal.x / NL_BOX_HALF.yz
                      + entryPoint.zx * faceNormal.y / NL_BOX_HALF.zx;
    float edgeCoord = max(abs(faceCoords.x), abs(faceCoords.y));
    float edgeGlow = smoothstep(0.84f, 0.985f, edgeCoord);
    float faceFade = 1.0f - smoothstep(0.92f, 1.02f, edgeCoord);

    float3 marchOrigin = insideBox ? (eye + rd * 0.002f) : (entryPoint + rd * 0.002f);
    float maxDistance;
    if (insideBox) {
        maxDistance = NL_FAR;
    } else {
        float3 exitNormal;
        float throughCube = nlBoxHit(marchOrigin, rd, NL_BOX_HALF, exitNormal, false);
        if (throughCube < 0.0f) {
            throughCube = 4.0f;
        }
        maxDistance = throughCube + NL_PORTAL_DEPTH;
    }

    float glow = 0.0f;
    float distanceTraveled = 0.01f;
    NLObject hitObject = nlMakeObject(NL_FAR, NL_ID_NONE, marchOrigin);
    bool hit = false;

    for (int i = 0; i < NL_MAX_TRACE_STEPS; ++i) {
        float3 pos = marchOrigin + rd * distanceTraveled;
        float d = nlMap(pos, uniforms, hitObject, glow);
        if (abs(d) < NL_EPS) {
            hit = true;
            break;
        }
        distanceTraveled += max(abs(d) * 0.55f, NL_EPS * 0.5f);
        if (distanceTraveled > maxDistance) {
            break;
        }
    }

    float3 glassBase = mix(float3(0.014f, 0.016f, 0.024f), float3(0.05f, 0.08f, 0.14f), 1.0f - faceFade);
    glassBase += edgeGlow * float3(0.08f, 0.12f, 0.20f);
    float3 color = glassBase;

    if (hit) {
        float3 hitPos = marchOrigin + rd * distanceTraveled;
        float3 normal = nlMapNormal(hitPos, uniforms, 0.006f);
        float3 material = nlMaterial(hitObject);
        float3 lightDir = normalize(float3(1.0f, 1.0f, -1.0f));
        float shadow = nlSoftShadow(hitPos + normal * 0.02f, lightDir, uniforms, 8.0f, 128.0f);
        float diff = max(dot(lightDir, normal), 0.0f);
        float back = max(dot(-lightDir, normal), 0.0f);
        float spec = pow(max(dot(reflect(lightDir, normal), rdUnit), 0.0f), 64.0f);
        color = material * (0.2f + 0.2f * back + 0.8f * diff * shadow) + 0.8f * spec * shadow;
    }

    color += glow * float3(1.0f, 0.8f, 0.6f);
    float fresnel = pow(clamp(1.0f - abs(dot(faceNormal, rdUnit)), 0.0f, 1.0f), 3.0f);
    color += fresnel * float3(0.05f, 0.08f, 0.12f);
    color = clamp(tanh(color), 0.0f, 1.0f);
    return float4(color, 1.0f);
}