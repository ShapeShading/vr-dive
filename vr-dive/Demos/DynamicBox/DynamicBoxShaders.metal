// DynamicBoxShaders.metal
// Vertex shader (fixed) + default fragment shader (3D grid of light points).
// The fragment shader can be hot-reloaded at runtime via the shader server.

#include <metal_stdlib>
using namespace metal;

// ─── Uniforms (must match DynamicBoxUniforms in DynamicBoxTypes.swift) ─────────
struct DynamicBoxUniforms {
    float  time;
    uint   viewCount;
    float  boxScale;
    float  _pad;
    float4 objectCenter;
    float4x4 patternTransform;
};

// ─── Vertex input / output ────────────────────────────────────────────────────
struct MeshVertex {
    float3 position;
    float3 normal;
};

struct DynamicBoxVertexOut {
    float4 clipPos   [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ─── Vertex shader (shared – same as GlassBox) ────────────────────────────────
vertex DynamicBoxVertexOut dynamicBoxVertex(
    ushort                     amplificationID [[amplification_id]],
    const device MeshVertex   *vertices        [[buffer(0)]],
    constant DynamicBoxUniforms &uniforms      [[buffer(1)]],
    constant float4x4         *vpMatrices      [[buffer(2)]],
    uint                       vertexID        [[vertex_id]])
{
    MeshVertex vtx  = vertices[vertexID];
    uint viewIndex  = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    DynamicBoxVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEFAULT FRAGMENT SHADER — 3D Grid of Light Points
// ═══════════════════════════════════════════════════════════════════════════════

#define DB_PI      3.14159265f
#define DB_BOXDIMS float3(0.95f, 0.95f, 1.25f)

// ─── Ray vs axis-aligned box (same logic as GlassBox) ─────────────────────────
static float db_boxHit(float3 ro, float3 rd, float3 r, thread float3 &nn, bool entering) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 dr = 1.0f / rd;
    float3 n  = ro * dr;
    float3 k  = r  * abs(dr);
    float3 pin  = -k - n;
    float3 pout =  k - n;
    float tin  = max(pin.x,  max(pin.y,  pin.z));
    float tout = min(pout.x, min(pout.y, pout.z));
    if (tin > tout) return -1.0f;
    if (entering) {
        nn = -sign(rd) * step(pin.zxy,  pin.xyz)  * step(pin.yzx,  pin.xyz);
        return tin;
    } else {
        nn =  sign(rd) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);
        return tout;
    }
}

// ─── HSV → RGB ─────────────────────────────────────────────────────────────────
static float3 db_hsv2rgb(float3 c) {
    float4 K = float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0f - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), c.y);
}

// ─── Default fragment: 3D grid of glowing points ──────────────────────────────
fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut        in         [[stage_in]],
    constant DynamicBoxUniforms &uniforms [[buffer(0)]],
    constant float4x4         *v2wMats    [[buffer(1)]],
    constant float4x4         *vpMatrices [[buffer(2)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);

    // Camera world position
    float4x4 v2w    = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    // Box-local space
    float3 center  = uniforms.objectCenter.xyz;
    float  sc      = uniforms.boxScale;
    float3 boxEye  = (camWorld - center) / sc;
    float3 boxRd   = normalize(in.worldPos - camWorld);

    // Box entry test
    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    float3 marchOrigin;
    float3 bgColor = float3(0.0f, 0.0f, 0.015f);

    if (!insideBox) {
        float3 entryNormal;
        float  tEnter = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (tEnter < 0.0f) {
            return float4(bgColor, 1.0f);
        }
        marchOrigin = boxEye + boxRd * (tEnter + 1e-3f);
    } else {
        marchOrigin = boxEye;
    }

    float3 exitNormal;
    float  tExit = db_boxHit(marchOrigin, boxRd, DB_BOXDIMS, exitNormal, false);
    if (tExit <= 0.0f) discard_fragment();

    // Apply pattern navigation transform
    float3 eye = (uniforms.patternTransform * float4(marchOrigin, 1.0f)).xyz;
    float3 rd  = normalize(float3(uniforms.patternTransform * float4(boxRd, 0.0f)));

    // ─── 3D Grid Spheres ───────────────────────────────────────────────────
    // Tiny spheres at integer grid positions, visible beyond the box boundary.
    float gridSpacing = 0.18f;
    float sphereR     = 0.006f;
    float maxMarch    = max(tExit * 2.0f, 30.0f); // 20+ meters beyond box

    float  accumA = 0.0f;
    float3 accumC = float3(0.0f);

    float stepSize = gridSpacing * 0.5f;
    int   maxSteps = int(maxMarch / stepSize) + 4;

    for (int i = 0; i < min(maxSteps, 512); i++) {
        float t = (float(i) + 0.5f) * stepSize;
        if (t > maxMarch) break;
        float3 p = eye + rd * t;

        float3 gp = round(p / gridSpacing) * gridSpacing;
        float3 oc = eye - gp;

        float b  = dot(oc, rd);
        float c2 = dot(oc, oc);
        float disc = b * b - (c2 - sphereR * sphereR);

        if (disc > 0.0f) {
            float sqrtD = sqrt(disc);
            float tHit  = -b - sqrtD;
            if (tHit < 0.001f) tHit = -b + sqrtD;
            if (tHit > 0.0f && tHit < t + stepSize && tHit < maxMarch) {
                float3 hitPos = eye + rd * tHit;
                float3 n      = normalize(hitPos - gp);

                float dif = max(0.0f, dot(n, rd));
                float amb = 0.4f + 0.6f * abs(n.y);
                float rim = 1.0f - max(dot(-rd, n), 0.0f);
                rim = pow(rim, 4.0f) * 0.5f;

                float hue = fract(dot(gp, float3(0.271f, 0.583f, 0.791f)) * 0.25f + uniforms.time * 0.02f);
                float3 albedo = db_hsv2rgb(float3(hue, 0.8f, 1.2f));

                float3 col = albedo * (dif * 0.9f + amb * 0.5f) + float3(0.3f, 0.5f, 1.0f) * rim;
                float alpha = 1.0f;
                accumC += col * alpha * (1.0f - accumA);
                accumA += alpha * (1.0f - accumA);
                if (accumA > 0.99f) break;
            }
        }
    }

    float3 finalColor = mix(bgColor, accumC, accumA);
    return float4(finalColor, 1.0f);
}
