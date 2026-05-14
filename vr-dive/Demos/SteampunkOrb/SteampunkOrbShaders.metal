// SteampunkOrbShaders.metal
// 3D visionOS adaptation of "Steampunk Orb" by Jaenam (ShaderToy WXfcWN).
//
// Original GLSL source:
//   https://www.shadertoy.com/view/WXfcWN
//   © 2025 Jaenam — CC BY-NC-SA 4.0
//   https://x.com/Jaenam97/status/1974927996898390144
//
// Ported to Metal / visionOS cube-container ray march by the vr-dive project.
// Rendering strategy: rasterise the 6 faces of a world-space cube; for each
// fragment reconstruct the ray from the camera through the cube surface and
// march inward.  Inside-camera support is handled by setting tStart = 0.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared types (must match SteampunkOrbTypes.swift)
// ---------------------------------------------------------------------------

struct SteampunkOrbUniforms {
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

struct SteampunkOrbVertexOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ---------------------------------------------------------------------------
// Vertex
// ---------------------------------------------------------------------------

vertex SteampunkOrbVertexOut steampunkOrbVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant SteampunkOrbUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    SteampunkOrbVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Rotation matrix for 2-component plane — equivalent to GLSL #define R(a)
static float2x2 soRotate(float a) {
    float c = cos(a), s = sin(a);
    // Metal: column-major. col0=(c,s), col1=(-s,c)  →  same as GLSL mat2(c,-s,s,c)
    return float2x2(float2(c, s), float2(-s, c));
}

// Axis-aligned box intersection. Returns (tNear, tFar).
static float2 soBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv  = 1.0f / rd;
    float3 t0   = (-halfExt - ro) * inv;
    float3 t1   = ( halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

// ---------------------------------------------------------------------------
// Core raymarch — translated directly from the GLSL original.
//
// The original iterates 120 steps in a for-loop, accumulates colour via
//   O += 25*d/s
// and finishes with tanh tone-mapping.
//
// Key translation notes (GLSL → Metal):
//   • p.yz *= R(t*.1)   →  p.yz = soRotate(t*.1) * p.yz
//   • mat2 mul order: GLSL col-vec "p *= M" == Metal "p = M * p"
//     (Metal matrix × vector: result_i = row_i · vector, same convention)
//   • float d,i,s,w,l are all zero-initialised in GLSL — explicit here.
//   • O*=i at loop start with i=0 clears O to zero (already zero-init here).
//   • "for(int i; i++ < 5; ...)" inner loop starts i=0, body runs while i<5.
// ---------------------------------------------------------------------------

static float3 steampunkOrbTrace(float3 ro, float3 rd, float t) {
    float4 O = float4(0.0f);
    float  d = 0.0f;

    // Bounding sphere radius for the orb — used as a gate to skip the
    // expensive fold when the ray is clearly outside the structure.
    const float GATE_R = 0.32f;
    const float GATE_MARGIN = 0.01f;

    for (float i = 0.0f; i < 80.0f; i += 1.0f) {
        // q = original ray position; p = rotated copy for folding
        float3 q = ro + rd * d;

        // Fast gate: if outside bounding sphere, step by sphere distance
        // without evaluating the fold — large step, no colour accumulation.
        float gateDist = length(q) - GATE_R;
        if (gateDist > GATE_MARGIN) {
            d += gateDist * 0.9f;
            continue;
        }

        float3 p = q;

        // Apply two time-driven rotations (same as original p.yz*=R, p.xz*=R)
        p.yz = soRotate(t * 0.1f) * p.yz;
        p.xz = soRotate(t * 0.1f) * p.xz;

        // Apollonian fold — 5 iterations
        float w = 8.0f;
        float l = 1.0f;
        for (int j = 0; j < 5; j++) {
            p  = sin(p);
            l  = 1.8f / dot(p, p);
            p *= l;
            w *= l;
        }

        // SDF: outer sphere ∪ folded length
        float s = max(length(q) - 0.3f, length(p.xz) / w);

        d   += s;
        O   += 25.0f * d / s;
    }

    // tanh tone-map: O channels mapped to (1,2,3) tint, scalar denom 2e7
    float3 col = tanh(float3(1.0f, 2.0f, 3.0f) * O.xyz / 2.0e7f);
    return col;
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

fragment float4 steampunkOrbFragment(
    SteampunkOrbVertexOut in [[stage_in]],
    constant SteampunkOrbUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center    = uniforms.objectCenter.xyz;
    float  scale     = max(uniforms.cubeScale, 1.0e-4f);
    float3 halfExt   = float3(1.0f);   // local-space ±1 cube

    // Camera in local cube space
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 eye         = (cameraWorld - center) / scale;

    // Surface point in local cube space
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 viewDir    = normalize(surfacePos - eye);

    // Box intersection to find ray segment
    bool   insideBox = all(abs(eye) < halfExt - 1.0e-3f);
    float2 tBox      = soBoxIntersect(eye, viewDir, halfExt);

    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float tEnd   = tBox.y;
    if (tEnd <= tStart) {
        discard_fragment();
    }

    // Ray origin at cube entry (+ small offset to avoid self-intersection)
    float3 ro = eye + viewDir * (tStart + 0.001f);

    // Scale scene: with cubeScale=4 the cube spans ±4 m in world space.
    // sceneScale=0.45 maps the cube local ±1 → ±0.45 scene units, placing
    // the 0.3-radius orb near the centre and filling most of the cube volume.
    const float sceneScale = 0.45f;
    float3 roScene = ro * sceneScale;

    float3 col = steampunkOrbTrace(roScene, viewDir, uniforms.time);

    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}
