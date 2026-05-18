// FollowYourLightShaders.metal
// 3D visionOS adaptation of "Follow your light" (ShaderToy 73s3zs).
//
// Original GLSL:
//   https://www.shadertoy.com/view/73s3zs
//   "Follow your light" by Noztol
//   Inspired by and rewrite of shadertoy.com/view/WcdczB
//   Ported to Metal / visionOS cube-container by the vr-dive project.
//
// Technique: Volumetric accumulation ray march (28 steps) through a winding
//   tunnel with a glowing orb, using palette-based color accumulation.
//
// GLSL → Metal translation notes:
//   • vec3(12.0 * cos(z * vec2(0.1, 0.12)), z)  →  float3(12*cos(z*0.1), 12*cos(z*0.12), z)
//     (GLSL uses a vec2 element-wise constructor; Metal needs explicit components)
//   • animTime + 16.0 * rayPos  →  float scalar + float3: Metal broadcasts the scalar ✓
//   • length(rayPos.xy - pathCenter.x - 6.0)  →  Metal broadcasts scalar subtraction ✓
//   • for(float i = 1.0; i <= 28.0; i++)  →  int loop; cast to float where needed

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared types  (must match FollowYourLightTypes.swift)
// ---------------------------------------------------------------------------

struct FollowYourLightUniforms {
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

struct FollowYourLightVertexOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ---------------------------------------------------------------------------
// Path helper
// ---------------------------------------------------------------------------

// Winding tunnel centre — GLSL: vec3(12.0 * cos(z * vec2(0.1, 0.12)), z)
// cos applied element-wise to each frequency component
static float3 fylGetPathPosition(float z) {
    return float3(12.0f * cos(z * 0.1f),
                  12.0f * cos(z * 0.12f),
                  z);
}

// Axis-aligned box intersection; returns (tNear, tFar)
static float2 fylBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
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
// Vertex
// ---------------------------------------------------------------------------

vertex FollowYourLightVertexOut followYourLightVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant FollowYourLightUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    FollowYourLightVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment — volumetric accumulation ray march
// ---------------------------------------------------------------------------

fragment float4 followYourLightFragment(
    FollowYourLightVertexOut in [[stage_in]],
    constant FollowYourLightUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center  = uniforms.objectCenter.xyz;
    float  scale   = max(uniforms.cubeScale, 1.0e-4f);
    float3 halfExt = float3(1.0f);   // cube local ±1

    // Camera and surface in local cube space
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 eye         = (cameraWorld - center) / scale;
    float3 surfacePos  = (in.worldPos - center) / scale;
    float3 viewDir     = normalize(surfacePos - eye);

    // Box intersection
    bool   insideBox = all(abs(eye) < halfExt - 1.0e-3f);
    float2 tBox      = fylBoxIntersect(eye, viewDir, halfExt);
    if (!insideBox && tBox.x > tBox.y) { discard_fragment(); }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float tEnd   = tBox.y;
    if (tEnd <= tStart) { discard_fragment(); }

    // Map cube-local entry point into scene space.
    // sceneScale=15 maps cube ±1 → ±15 scene units, matching the original's 30-unit march budget.
    // The virtual camera follows the winding tunnel path at animTime.
    const float sceneScale = 15.0f;
    float animTime = uniforms.time * 4.0f + 5.0f + 5.0f * sin(uniforms.time * 0.3f);
    float3 virtualCam = fylGetPathPosition(animTime);

    // Flip z: ShaderToy content runs in +Z; cube local -Z must map to scene +Z.
    float3 ro_entry = (eye + viewDir * (tStart + 0.001f));
    float3 ro = float3(ro_entry.x, ro_entry.y, -ro_entry.z) * sceneScale + virtualCam;
    float3 rd = float3(viewDir.x, viewDir.y, -viewDir.z);

    // March budget bounded by the box traversal distance (capped at 30 to match original)
    float maxTotalDist = min((tEnd - tStart) * sceneScale, 30.0f);

    // -----------------------------------------------------------------------
    // Volumetric accumulation loop (28 steps — matching original)
    // -----------------------------------------------------------------------
    float  stepDist = 1.0f;
    float  totalDist = 0.0f;
    float  orbDist = 1.0f;
    float3 accum = float3(0.0f);
    float3 rayPos = ro;

    float sineTime = sin(uniforms.time);   // original: sineTime = sin(iTime)

    for (int i = 1; i <= 28; i++) {
        if (totalDist >= maxTotalDist) break;

        // 1. March ray forward
        rayPos += rd * stepDist;

        // 2. Path centre at current z
        float3 pathCenter = fylGetPathPosition(rayPos.z);

        // 3. Orb geometry — orb drifts with sineTime, slightly ahead of animTime
        float3 orbCenter = float3(
            pathCenter.x + sineTime,
            pathCenter.y + sineTime * 2.0f,
            6.0f + animTime + sineTime * 2.0f);
        orbDist = length(rayPos - orbCenter) - 0.01f;

        // 4. Tunnel wall geometry
        float baseRadius = cos(rayPos.z * 0.6f) * 2.0f + 4.0f;

        // Two distance measures combined to make tunnel cross-section irregular
        // GLSL: length(rayPos.xy - pathCenter.x - 6.0)
        //   → Metal: scalar broadcast across float2 ✓
        float tunnelStructure = min(
            length(rayPos.xy - pathCenter.x - 6.0f),
            length((rayPos - pathCenter).xy));

        // Scalar broadcast in Metal: float + float3 = float3 ✓
        float largeScoops   = abs(dot(sin(0.4f * rayPos),               float3(0.25f))) / 0.1f;
        float detailTexture = abs(dot(sin(animTime + 16.0f * rayPos),   float3(0.22f))) / 2.0f;

        float tunnelDist = baseRadius - tunnelStructure + largeScoops + detailTexture;

        // 5. Adaptive step size
        stepDist   = min(orbDist, 0.01f + 0.3f * abs(tunnelDist));
        totalDist += stepDist;

        // 6. Colour accumulation — palette cycles per loop index
        float  fi      = float(i);
        float3 palette = 1.0f + cos(fi * 0.7f + float3(6.0f, 1.0f, 2.0f));
        accum += (palette / stepDist + 10.0f * palette / max(orbDist, 0.6f)) / fi;
    }

    // Tone-map: squash then tanh (matches original)
    float3 col = accum * accum / 2000.0f;
    return float4(tanh(col), 1.0f);
}
