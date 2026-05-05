// ApolloSpiralShaders.metal
// Source adaptation: Shadertoy "Apollo Spiral"
// https://www.shadertoy.com/view/WXVGRG
//
// The original shader is a screen-space raymarch / accumulation effect.
// This version adapts the core fractal and spiral field into a 3D cube-contained
// volume so the result can be observed from any direction in immersive space.
// The cube acts as a portal when viewed from outside so deeper structure can
// remain visible instead of being clipped at the back face.

#include <metal_stdlib>
using namespace metal;

struct ApolloSpiralUniforms {
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

struct ApolloSpiralVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

vertex ApolloSpiralVertexOut apolloSpiralVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant ApolloSpiralUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    ApolloSpiralVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static constant float3 AP_BOX_HALF = float3(1.0f, 1.0f, 1.0f);
static constant int AP_MAX_VOLUME_STEPS = 480;
static constant float AP_MIN_STEP = 0.007f;
static constant float AP_MAX_STEP = 0.045f;
static constant float AP_PATTERN_SCALE = 2.15f;
static constant float AP_OUTSIDE_VIEW_DEPTH = 10.0f;

static float2 apRot2d(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

// Adapted from the Shadertoy source fractal() function.
static float apFractal(float3 p) {
    float w = 4.0f;
    for (int iter = 0; iter < 6; ++iter) {
        p = cos(p - 0.5f);
        float l = 2.0f / max(dot(p, p), 0.18f);
        p *= l;
        w *= l;
    }
    return length(p) / w;
}

// Adapted from the first Apollo Spiral volume attempt. Keep the original
// volume-field logic and only tune the scale / accumulation parameters.
static float apSpiralField(float3 p, float time) {
    float3 q = p * AP_PATTERN_SCALE;
    q.z += time * 2.0f;
    q.xy = apRot2d(q.xy, 0.05f * time + q.z * 0.2f);

    float s = sin(4.0f + q.y + q.x);
    for (float n = 5.0f; n < 16.0f; n += n) {
        s -= abs(dot(cos(q * n), float3(1.0f))) / n;
    }
    return abs(min(apFractal(q), s));
}

static float3 apPalette(float glow, float3 p) {
    float3 c = pow(float3(glow), float3(1.0f, 2.0f, 12.0f)) * 6.0f;
    c = tanh(mix(c, c.yzx, clamp(length(p.xy) * 0.8f, 0.0f, 1.0f)));
    return c;
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

fragment float4 apolloSpiralFragment(
    ApolloSpiralVertexOut in [[stage_in]],
    constant ApolloSpiralUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < (AP_BOX_HALF - 1e-3f));
    float3 entryNormal;
    float entryT = apBoxHit(eye, rd, AP_BOX_HALF, entryNormal, !insideBox);
    if (entryT < 0.0f) {
        discard_fragment();
    }

    float3 entryPoint = eye + rd * entryT;
    float3 faceNormal = insideBox ? -entryNormal : entryNormal;
    float2 faceCoords = entryPoint.xy * faceNormal.z / AP_BOX_HALF.xy
                      + entryPoint.yz * faceNormal.x / AP_BOX_HALF.yz
                      + entryPoint.zx * faceNormal.y / AP_BOX_HALF.zx;
    float edgeCoord = max(abs(faceCoords.x), abs(faceCoords.y));
    float edgeGlow = smoothstep(0.84f, 0.985f, edgeCoord);
    float faceFade = 1.0f - smoothstep(0.92f, 1.02f, edgeCoord);

    float3 marchOrigin = insideBox ? (eye + rd * 0.0015f) : (entryPoint + rd * 0.0015f);
    float exitT;
    if (insideBox) {
        exitT = max(entryT - 0.0015f, 0.0f);
    } else {
        float3 exitNormal;
        float throughCube = apBoxHit(marchOrigin, rd, AP_BOX_HALF, exitNormal, false);
        if (throughCube < 0.0f) {
            throughCube = 4.0f;
        }
        exitT = throughCube + AP_OUTSIDE_VIEW_DEPTH;
    }

    float3 glassBase = mix(float3(0.012f, 0.015f, 0.025f), float3(0.05f, 0.09f, 0.16f), 1.0f - faceFade);
    glassBase += edgeGlow * float3(0.10f, 0.18f, 0.30f);

    float t = 0.0f;
    float transmittance = 1.0f;
    float3 accum = float3(0.0f);

    for (int i = 0; i < AP_MAX_VOLUME_STEPS; ++i) {
        if (t >= exitT || transmittance < 0.00035f) {
            break;
        }

        float3 pos = marchOrigin + rd * t;
        float field = apSpiralField(pos, uniforms.time);
        float density = exp(-14.0f * field);
        density *= smoothstep(26.0f, 0.15f, length(pos));

        float glow = 1.0f / (0.045f + field * 10.0f);
        float3 sampleCol = apPalette(glow * 0.060f, pos);
        sampleCol *= mix(float3(0.9f, 0.55f, 0.35f), float3(0.45f, 0.85f, 1.25f), clamp(length(pos.xy) * 0.42f + 0.15f, 0.0f, 1.0f));

        float alpha = clamp(density * 0.050f, 0.0f, 0.08f);
        accum += transmittance * sampleCol * alpha;
        transmittance *= (1.0f - alpha);

        float stepSize = clamp(0.010f + field * 0.10f, AP_MIN_STEP, AP_MAX_STEP);
        t += stepSize;
    }

    float3 col = accum + transmittance * glassBase;
    float fresnel = pow(clamp(1.0f - abs(dot(faceNormal, rdUnit)), 0.0f, 1.0f), 3.2f);
    col += fresnel * float3(0.06f, 0.10f, 0.16f);
    col = mix(col, glassBase + edgeGlow * float3(0.15f, 0.22f, 0.34f), 0.10f);
    col = clamp(tanh(col), 0.0f, 1.0f);
    return float4(col, 1.0f);
}