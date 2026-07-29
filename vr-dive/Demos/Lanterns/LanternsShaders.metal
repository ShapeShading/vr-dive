// LanternsShaders.metal
// Original Lanterns implementation for vr-dive.
// Request referenced https://www.shadertoy.com/view/4sB3D1, but that source has
// restrictive terms prohibiting reuse/adaptation. This shader is an original
// lantern-field implementation designed for the same 3D cube-portal behavior.

#include <metal_stdlib>
using namespace metal;

struct LanternsUniforms {
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

struct LanternsVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct LanternInfo {
    float3 center;
    float groundY;
    float radius;
    float stemRadius;
    float3 glowColor;
};

struct LanternSceneSample {
    float dist;
    float material;
    float glow;
    float3 glowColor;
    float2 cell;
};

static constant float3 LAN_BOX_HALF = float3(1.0f);
static constant float LAN_SCENE_SCALE = 3.12f;
static constant float3 LAN_SCENE_OFFSET = float3(0.0f, 0.0f, -1.0f);
static constant float LAN_MAX_DISTANCE = 28.0f;
static constant int LAN_MAX_STEPS = 144;
static constant float LAN_EPSILON = 0.0012f;

vertex LanternsVertexOut lanternsVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant LanternsUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    LanternsVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float3 lanHash33(float2 p) {
    float n = dot(p, float2(41.0f, 289.0f));
    return fract(sin(float3(n, n + 1.0f, n + 2.0f)) * float3(43758.5453f, 22578.1459f, 19642.3490f));
}

static float lanBoxHit(float3 ro, float3 rd, float3 halfExtents, thread float3 &nn, bool entering) {
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

static float lanSdRoundBox(float3 p, float3 b, float r) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f) - r;
}

static float lanSdSphere(float3 p, float r) {
    return length(p) - r;
}

static float lanSdCapsule(float3 p, float3 a, float3 b, float r) {
    float3 pa = p - a;
    float3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h) - r;
}

static float lanSdTorus(float3 p, float2 t) {
    float2 q = float2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

static LanternInfo lanMakeLantern(float2 cell, float time) {
    float3 seed = lanHash33(cell);
    float2 offset = (seed.xy - 0.5f) * 0.38f;
    float headHeight = -0.10f + 0.55f * seed.z;
    float bob = 0.10f * sin(time * (0.7f + 0.5f * seed.x) + seed.y * 6.28318f + dot(cell, float2(0.7f, 1.1f)));

    LanternInfo info;
    info.radius = 0.10f + 0.045f * seed.z;
    info.center = float3(cell.x + offset.x, headHeight + bob, cell.y + offset.y);
    info.groundY = -1.05f;
    info.stemRadius = mix(0.018f, 0.032f, seed.x);
    info.glowColor = mix(float3(1.0f, 0.50f, 0.16f), float3(1.0f, 0.78f, 0.35f), seed.y);
    return info;
}

static LanternSceneSample lanMap(float3 p, float time) {
    LanternSceneSample sample;
    sample.dist = 1.0e6f;
    sample.material = 0.0f;
    sample.glow = 0.0f;
    sample.glowColor = float3(0.0f);
    sample.cell = float2(0.0f);

    float2 baseCell = floor(p.xz);
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            float2 cell = baseCell + float2(float(ox), float(oy));
            LanternInfo info = lanMakeLantern(cell, time);
            float3 q = p - info.center;

            float head = lanSdSphere(q, info.radius);
            float headCore = lanSdSphere(q, info.radius * 0.68f);
            float3 stemBase = float3(info.center.x, info.groundY, info.center.z);
            float3 stemTop = info.center - float3(0.0f, info.radius * 0.92f, 0.0f);
            float stem = lanSdCapsule(p, stemBase, stemTop, info.stemRadius);
            float collar = lanSdTorus(q - float3(0.0f, info.radius * 0.10f, 0.0f), float2(info.radius * 0.16f, info.radius * 0.055f));
            float lantern = min(min(head, stem), collar);

            if (lantern < sample.dist) {
                sample.dist = lantern;
                sample.material = head < min(stem, collar) ? 0.0f : (stem < collar ? 1.0f : 2.0f);
                sample.glowColor = info.glowColor;
                sample.cell = cell;
            }

            float inner = max(headCore, -head);
            float localGlow = 0.05f / (0.025f + inner * inner);
            sample.glow += localGlow;
            sample.glowColor += info.glowColor * localGlow;
        }
    }

    // Keep a weak ground reference well below the main lantern cluster so it
    // doesn't mask most front-facing rays before they reach the lanterns.
    float floorPlane = p.y + 1.05f;
    if (floorPlane < sample.dist) {
        sample.dist = floorPlane;
        sample.material = 3.0f;
        sample.cell = floor(p.xz);
    }

    sample.glowColor /= max(sample.glow, 1.0e-4f);
    return sample;
}

static float lanMapDistance(float3 p, float time) {
    return lanMap(p, time).dist;
}

static float3 lanNormal(float3 p, float time) {
    float2 e = float2(0.0015f, -0.0015f);
    return normalize(
        e.xyy * lanMapDistance(p + e.xyy, time) +
        e.yyx * lanMapDistance(p + e.yyx, time) +
        e.yxy * lanMapDistance(p + e.yxy, time) +
        e.xxx * lanMapDistance(p + e.xxx, time));
}

static float lanAmbientOcclusion(float3 p, float3 n, float time) {
    float occlusion = 0.0f;
    float weight = 1.0f;
    for (int i = 0; i < 5; ++i) {
        float h = 0.04f + 0.12f * float(i);
        float d = lanMapDistance(p + n * h, time);
        occlusion += (h - d) * weight;
        weight *= 0.6f;
    }
    return clamp(1.0f - 1.8f * occlusion, 0.0f, 1.0f);
}

static float3 lanBackground(float3 rd) {
    float t = clamp(rd.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float3 low = float3(0.01f, 0.014f, 0.022f);
    float3 high = float3(0.05f, 0.07f, 0.11f);
    float stars = pow(max(0.0f, sin(rd.x * 91.0f) * sin(rd.y * 117.0f) * sin(rd.z * 83.0f)), 18.0f);
    return mix(low, high, t) + stars * 0.18f;
}

fragment float4 lanternsFragment(
    LanternsVertexOut in [[stage_in]],
    constant LanternsUniforms &uniforms [[buffer(0)]],
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
    float entryT = insideBox ? 0.0f : lanBoxHit(eye, rd, LAN_BOX_HALF, faceNormal, true);
    if (!insideBox && entryT < 0.0f) {
        discard_fragment();
    }

    float time = uniforms.time;
    float3 marchOrigin = insideBox ? (eye + rd * 0.002f) : (eye + rd * (entryT + 0.002f));

    float travel = 0.0f;
    LanternSceneSample hitSample;
    hitSample.dist = 0.0f;
    float3 pos = marchOrigin * LAN_SCENE_SCALE + LAN_SCENE_OFFSET;
    bool hit = false;
    for (int i = 0; i < LAN_MAX_STEPS; ++i) {
        float3 worldPoint = marchOrigin + rd * travel;
        pos = worldPoint * LAN_SCENE_SCALE + LAN_SCENE_OFFSET;
        hitSample = lanMap(pos, time);
        float worldDist = hitSample.dist / LAN_SCENE_SCALE;
        if (worldDist < LAN_EPSILON) {
            hit = true;
            break;
        }
        if (travel > LAN_MAX_DISTANCE) {
            break;
        }
        travel += worldDist * 0.72f;
    }

    float3 color = lanBackground(rd);
    float fogGlow = min(hitSample.glow, 0.35f);
    color += hitSample.glowColor * fogGlow * 0.035f;

    if (hit) {
        float3 n = lanNormal(pos, time);
        float ao = lanAmbientOcclusion(pos, n, time);
        float3 view = -rd;
        float3 warmLightDir = normalize(float3(-0.5f, 0.9f, -0.3f));
        float diffuse = max(dot(n, warmLightDir), 0.0f);
        float rim = pow(1.0f - max(dot(n, view), 0.0f), 3.0f);

        float3 baseColor;
        if (hitSample.material < 0.5f) {
            baseColor = hitSample.glowColor * 0.45f + float3(0.35f, 0.12f, 0.06f);
        } else if (hitSample.material < 1.5f) {
            baseColor = float3(0.18f, 0.12f, 0.06f);
        } else if (hitSample.material < 2.5f) {
            baseColor = float3(0.12f, 0.08f, 0.04f);
        } else {
            float tile = 0.5f + 0.5f * sin(dot(floor(hitSample.cell), float2(1.0f, 7.0f)));
            baseColor = mix(float3(0.06f, 0.05f, 0.04f), float3(0.12f, 0.09f, 0.06f), tile);
        }

        float3 localEmission = hitSample.glowColor * min(hitSample.glow, 1.5f) * (hitSample.material < 2.5f ? 0.95f : 0.04f);
        float3 lighting = float3(0.05f, 0.06f, 0.08f) * ao;
        lighting += diffuse * float3(0.9f, 0.75f, 0.55f) * ao;
        lighting += rim * hitSample.glowColor * 0.25f;

        color = baseColor * lighting + localEmission;
        color *= exp(-0.012f * travel * travel);
    }

    float3 surfacePos = insideBox ? eye : (eye + rd * entryT);
    float3 absSurface = abs(surfacePos);
    float3 surfaceNormal = absSurface.x > absSurface.y && absSurface.x > absSurface.z
        ? float3(sign(surfacePos.x), 0.0f, 0.0f)
        : (absSurface.y > absSurface.z
            ? float3(0.0f, sign(surfacePos.y), 0.0f)
            : float3(0.0f, 0.0f, sign(surfacePos.z)));
    float fresnel = pow(1.0f - max(dot(-rd, surfaceNormal), 0.0f), 2.0f);
    color += float3(0.12f, 0.07f, 0.03f) * fresnel * 0.06f;

    color = pow(clamp(color, 0.0f, 1.0f), float3(0.44f));
    return float4(color, 1.0f);
}