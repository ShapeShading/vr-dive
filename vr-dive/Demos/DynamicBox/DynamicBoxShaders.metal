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
    float3 bgColor = float3(0.02f, 0.02f, 0.05f);

    if (!insideBox) {
        float3 entryNormal;
        float  tEnter = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (tEnter < 0.0f) {
            // Missed the box entirely – render dark background with subtle grid glow
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

    // ─── 3D Grid Point Rendering ───────────────────────────────────────────
    float gridSpacing = 0.12f;                     // distance between grid points
    float pointRadius = 0.025f;                    // visual radius of each point
    float glowFalloff = 120.0f;                    // sharpness of glow
    float maxMarch    = tExit + 0.5f;              // march a bit past exit for edge glow

    float3 pos    = eye;
    float  accumA = 0.0f;
    float3 accumC = float3(0.0f);

    // Ray-march through the grid in steps of gridSpacing/2
    float stepSize = gridSpacing * 0.5f;
    int   maxSteps = int(maxMarch / stepSize) + 2;

    for (int i = 0; i < min(maxSteps, 128); i++) {
        float t = (float(i) + 0.5f) * stepSize;
        if (t > maxMarch) break;
        float3 p = eye + rd * t;

        // Snap to nearest grid point
        float3 gp = round(p / gridSpacing) * gridSpacing;
        float  dist = length(p - gp);

        if (dist < pointRadius * 2.5f) {
            // Glow intensity – Gaussian-ish falloff
            float glow = exp(-dist * dist * glowFalloff);
            // Pulse animation
            float pulse = 0.7f + 0.3f * sin(uniforms.time * 2.0f + dot(gp, float3(3.7f, 5.1f, 7.3f)));
            glow *= pulse;

            // Color based on grid position
            float hue = fract(dot(gp, float3(0.373f, 0.617f, 0.819f)) * 0.4f + uniforms.time * 0.05f);
            float3 rgb = db_hsv2rgb(float3(hue, 0.7f, 1.0f));

            // Alpha compositing (front-to-back with emission)
            float alpha = glow * 0.35f;
            accumC += rgb * alpha * (1.0f - accumA);
            accumA += alpha * (1.0f - accumA);

            if (accumA > 0.98f) break;
        }
    }

    // Fade box edges
    float3 absPos  = abs(pos);
    float  edgeDist = min(min(DB_BOXDIMS.x - absPos.x,
                               DB_BOXDIMS.y - absPos.y),
                               DB_BOXDIMS.z - absPos.z);
    float edgeFade = smoothstep(0.0f, 0.08f, edgeDist);

    float3 finalColor = mix(bgColor, accumC, accumA) * edgeFade;
    return float4(finalColor, 1.0f);
}
