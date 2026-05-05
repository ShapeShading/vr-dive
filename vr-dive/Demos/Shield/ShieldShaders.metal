// ShieldShaders.metal
// "Shield" by @XorDev — https://www.shadertoy.com/view/cltfRf
//
// Faithful 3D stereo port for visionOS.
//
// The original GLSL accumulates 100 concentric sphere shells in 2D screen space:
//   for(i=0; i<1; i+=.01) { p = screenNDC * i; ...sphere_distortion; ...hex; }
//
// Here each iteration analytically intersects the per-eye ray with the sphere
// shell of radius i.  Because left/right eye positions differ, each shell is
// sampled at a slightly different 3D point — producing real stereo parallax.
// The hex formula, sphere distortion, z-weighting, and tanh tonemap are
// unchanged from the original.

#include <metal_stdlib>
using namespace metal;

struct ShieldUniforms {
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

struct ShieldVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

vertex ShieldVertexOut shieldVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant ShieldUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    ShieldVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — direct 3D port of the original mainImage loop.
//
// Object / scene space:
//   • The cube container has half-extents of 1.0 (mesh) × cubeScale in world.
//   • We work in scene space: scene = (world − center) / cubeScale.
//   • The 100 sphere shells span radii 0.01 … 1.0 in scene space, which
//     matches the original's i range exactly.
// ---------------------------------------------------------------------------

fragment float4 shieldFragment(
    ShieldVertexOut in [[stage_in]],
    constant ShieldUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center   = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 rd = normalize(in.worldPos - camWorld); // per-eye ray direction → stereo parallax

    // World-fixed basis for UV.  Using worldUp = (0,1,0) keeps hex orientation
    // stable regardless of head tilt.  Gram-Schmidt from camera-to-center.
    float3 worldFwd   = normalize(center - camWorld);
    float3 worldUp    = float3(0.0f, 1.0f, 0.0f);
    float3 fixedUp    = normalize(worldUp - dot(worldUp, worldFwd) * worldFwd);
    float3 fixedRight = normalize(cross(worldFwd, fixedUp));

    // UV base: direct spherical projection of the ray direction (no perspective
    // division).  In the loop p = uvBase * i, so:
    //   inner shells (small i) → tiny p → z≈1 → bright center  ✓
    //   outer shells (large i) → large p → z→0 → dim edges      ✓
    // Because rd differs between eyes for the same worldPos, uvBase differs
    // between eyes and the parallax magnitude scales with i — inner shells
    // appear far away, outer shells appear closer, creating a 3D sphere. ✓
    float2 uvBase = float2(dot(rd, fixedRight), dot(rd, fixedUp)) * 4.4f;

    float t = uniforms.time;
    float4 O = float4(0.0f);

    for (int n = 1; n <= 100; n++) {
        float i = float(n) * 0.01f;

        float2 p = uvBase * i;

        // ---- Sphere distortion (identical to original) --------------------
        float  z = max(1.0f - dot(p, p), 0.0f);
        p /= 0.2f + sqrt(z) * 0.3f;

        // ---- Hex-grid scroll (identical to original) ----------------------
        p.x  = p.x / 0.9f + t;
        p.y += fract(ceil(p.x) * 0.5f) + t * 0.2f;

        // ---- Hex cell distance (identical to original) --------------------
        float2 v       = abs(fract(p) - 0.5f);
        float  hexDist = abs(max(v.x * 1.5f + v, v + v).y - 1.0f)
                       + 0.1f - i * 0.09f;

        // ---- Color accumulation (identical to original) -------------------
        O += float4(2.0f, 3.0f, 5.0f, 1.0f) / 2000.0f * z / hexDist;
    }

    O = tanh(O * O);
    return float4(O.rgb, 1.0f);
}