// NeonShellsShaders.metal
// 3D visionOS adaptation of "Neon Shells" (ShaderToy scjSRt).
//
// Original GLSL:
//   https://www.shadertoy.com/view/scjSRt
//   Ported to Metal / visionOS cube-container by the vr-dive project.
//
// Technique: Volumetric accumulation ray march (50 steps) through an infinite
//   fractal lattice. In the original 2D shader, p = t * r so the twist amount
//   uses the radial distance from the virtual camera origin, not the segment
//   distance from a box entry point. The cube adaptation preserves that by
//   evaluating a world-space field whose shell twist is driven by |scenePos|.
//   Color is accumulated from a per-step palette and tone-mapped with tanh.
//
// GLSL → Metal translation notes:
//   • GLSL `p.xz *= mat2(cos(A + vec4(0,11,33,0)))` is row-vector form (v = v * M).
//     Metal equivalent: `p.xz = nsRot(A) * p.xz` (col-vector form).
//     Matrix M has row-vec result:
//       new.x = p.x*cos(A)    + p.z*cos(A+11)
//       new.z = p.x*cos(A+33) + p.z*cos(A)
//     Metal M*v with col0=[c0,c2], col1=[c1,c0]:
//       (M*v).x = c0*v.x + c1*v.z = cos(A)*p.x + cos(A+11)*p.z  ✓
//       (M*v).y = c2*v.x + c0*v.z = cos(A+33)*p.x + cos(A)*p.z  ✓
//   • `p.yz + p.x` — GLSL scalar broadcast to float2; Metal supports this ✓
//   • `p = (p.x < p.y) ? p.zxy : p.zyx` — scalar bool ternary on float3;
//     Metal supports scalar-condition ternary selecting float3 values ✓
//   • `fract()` — identical semantics in both GLSL and Metal ✓
//   • Original `r.xy *= mat2(...)` is preserved by rotating sample positions in
//     xy. Since p = t*r in the source, rotating r is equivalent to rotating p.
//   • `5e1` in GLSL = 50.0 in Metal.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared types  (must match NeonShellsTypes.swift)
// ---------------------------------------------------------------------------

struct NeonShellsUniforms {
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

struct NeonShellsVertexOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ---------------------------------------------------------------------------
// Rotation helper
//
// GLSL original: `p.xz *= mat2(cos(A + vec4(0,11,33,0)))` (row-vec form)
// Metal col-vec form: `p.xz = nsRot(A) * p.xz`
//
// float2x2 is column-major:  nsRot(A) = float2x2(col0, col1)
//   col0 = [cos(A),     cos(A+33)]
//   col1 = [cos(A+11),  cos(A)  ]
//
// Multiply M*v:
//   result.x = cos(A)*v.x    + cos(A+11)*v.z
//   result.y = cos(A+33)*v.x + cos(A)*v.z      ← assigned back to p.xz ✓
// ---------------------------------------------------------------------------

static float2x2 nsRot(float a) {
    float c0 = cos(a), c1 = cos(a + 11.0f), c2 = cos(a + 33.0f);
    return float2x2(float2(c0, c2), float2(c1, c0));
}

// Same matrix family as the original camera rotation:
// GLSL r.xy *= mat2(cos(a + vec4(0,11,33,0)))  ==  rotate sample position.xy.
static float2x2 nsCameraRot(float a) {
    float c0 = cos(a), c1 = cos(a + 11.0f), c2 = cos(a + 33.0f);
    return float2x2(float2(c0, c2), float2(c1, c0));
}

// ---------------------------------------------------------------------------
// Axis-aligned box intersection; returns (tNear, tFar)
// ---------------------------------------------------------------------------

static float2 nsBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
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

vertex NeonShellsVertexOut neonShellsVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant NeonShellsUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    NeonShellsVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment — volumetric fractal-lattice accumulation march
// ---------------------------------------------------------------------------

fragment float4 neonShellsFragment(
    NeonShellsVertexOut in [[stage_in]],
    constant NeonShellsUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center  = uniforms.objectCenter.xyz;
    float  scale   = max(uniforms.cubeScale, 1.0e-4f);
    float3 halfExt = float3(1.0f);

    // Camera and surface in local cube space
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 eye         = (cameraWorld - center) / scale;
    float3 surfacePos  = (in.worldPos - center) / scale;
    float3 viewDir     = normalize(surfacePos - eye);

    // Box intersection
    bool   insideBox = all(abs(eye) < halfExt - 1.0e-3f);
    float2 tBox      = nsBoxIntersect(eye, viewDir, halfExt);
    if (!insideBox && tBox.x > tBox.y) { discard_fragment(); }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float tEnd   = tBox.y;
    if (tEnd <= tStart) { discard_fragment(); }

    // Map cube-local ray into scene space.
    // sceneScale=1.6 keeps the repeating unit-sized folds legible inside the
    // 2 m cube while staying closer to the original camera-origin composition.
    const float sceneScale = 1.6f;

    // Flip z: ShaderToy content runs in +Z; cube local -Z must map to scene +Z.
    float3 ro_entry = (eye + viewDir * (tStart + 0.001f));
    float3 ro_s = float3(ro_entry.x, ro_entry.y, -ro_entry.z) * sceneScale;
    float3 rd_s = float3(viewDir.x, viewDir.y, -viewDir.z);

    // Maximum march budget in scene units
    float maxDist = (tEnd - tStart) * sceneScale;

    // -----------------------------------------------------------------------
    // Volumetric accumulation loop — 50 iterations matching original
    //
    // Original loop structure:
    //   for (O*=i; i++<50.; t+=v)   where i=0 initially, so first iter i=1.
    //   O*=0 zeros the output before the first iteration.
    // -----------------------------------------------------------------------
    float  t = 0.0f;
    float4 O = float4(0.0f);

    for (int ii = 1; ii <= 50; ii++) {
        if (t > maxDist) break;

        // Position along the ray in scene space.
        float3 scenePos = ro_s + rd_s * t;

        // Preserve the original camera rotation. In the source, p = t * r, so
        // rotating r.xy is exactly equivalent to rotating p.xy before the rest
        // of the field evaluation.
        scenePos.xy = nsCameraRot(uniforms.time * 0.1f) * scenePos.xy;

        // Original shader invariant: p = t * r with |r|=1, therefore t = |p|.
        // The shell twist must use radial distance from the virtual origin,
        // not box-entry march distance. This is the key shape-restoring change.
        float3 p = scenePos;
        float radialT = length(scenePos);

        // Rotate xz plane by radial distance from the virtual camera origin.
        // GLSL: p.xz *= mat2(cos(t*0.5 + vec4(0,11,33,0))) with t = |p|.
        p.xz = nsRot(radialT * 0.5f) * p.xz;

        // Time-based forward movement (fly through the fractal tunnel)
        p.z -= uniforms.time * 0.2f;

        // Fractal fold — GLSL: p = fract(p.zyx - 0.5) - 0.5
        // Swizzle p.zyx is an rvalue; Metal fract() is identical to GLSL ✓
        p = fract(p.zyx - 0.5f) - 0.5f;

        // Inner fold: 7 iterations of abs + conditional swap + scale + shift
        // GLSL: p = (p.x < p.y) ? p.zxy : p.zyx
        // Metal: scalar bool ternary selecting between two float3 swizzles ✓
        for (int j = 0; j < 7; j++) {
            p = abs(p);
            p = (p.x < p.y) ? p.zxy : p.zyx;
            p = p * 1.5f;
            p.x -= 1.0f;
        }

        // Distance measure.
        // length(p.yz + p.x): p.yz is float2, p.x is float — scalar broadcast ✓
        float v = abs(min(length(p.yz + p.x), length(p.xy)) + 0.005f) / 50.0f;

        // Guard against degenerate step that would cause NaN / Inf or stall.
        // Keep the floor tiny so the march shape stays faithful to the source.
        v = max(v, 1.0e-5f);

        // Color accumulation — palette cycles with loop index
        // GLSL: exp(sin(i * 0.1 * vec4(1,2,3,1))) / v
        float fi = float(ii);
        float4 phase = fi * 0.1f * float4(1.0f, 2.0f, 3.0f, 1.0f);
        O += exp(sin(phase)) / v;

        t += v;
    }

    // Tone-map: tanh(O / 2e4) — matches original
    float3 col = tanh(O.rgb / 2.0e4f);
    return float4(col, 1.0f);
}
