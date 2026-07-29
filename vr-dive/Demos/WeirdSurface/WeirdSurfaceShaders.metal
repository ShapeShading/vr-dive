// WeirdSurfaceShaders.metal
// 3D visionOS adaptation of "Weird Surface" (ShaderToy NXsGzX).
//
// Original GLSL:
//   https://www.shadertoy.com/view/NXsGzX
//   Klein bottle figure-8 implicit surface by Noztol
//   Ported to Metal / visionOS cube-container by the vr-dive project.
//
// Technique: Newton-step ray marching with bisection refinement.
//   Multi-layer alpha compositing lets the camera see through the
//   translucent surface to inner shells.  Auto-rotation replaces mouse.
//
// GLSL → Metal translation notes:
//   • GLSL `p.yz *= mat2(c,-s,s,c)` (row-vec × col-major mat2)
//     = Metal `p.yz = float2x2(float2(c,-s), float2(s,c)) * p.yz` (col-vec × mat) — same result.
//   • `atan(z, x)` two-arg GLSL form → `atan2(z, x)` in Metal.
//   • `PI` → `M_PI_F`.
//   • Original unrotates the normal to world space for shading; in the cube version
//     rd_s is already in the rotated scene space, so the unrotation is omitted.
//   • Original scales q_ro/q_rd by 1.4 (zoom trick); sceneScale handles this instead.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared types  (must match WeirdSurfaceTypes.swift)
// ---------------------------------------------------------------------------

struct WeirdSurfaceUniforms {
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

struct WeirdSurfaceVertexOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ---------------------------------------------------------------------------
// Rotation helper
// GLSL: p.yz *= mat2(c,-s,s,c)   (row-vec form)
// Metal: p.yz = wsRot(a) * p.yz  (col-vec form — same transformation)
//   float2x2 is column-major: col0=float2(c,-s), col1=float2(s,c)
//   M*v = (c*v.x + s*v.y,  -s*v.x + c*v.y) ✓
// ---------------------------------------------------------------------------

static float2x2 wsRot(float a) {
    float s = sin(a), c = cos(a);
    return float2x2(float2(c, -s), float2(s, c));
}

// ---------------------------------------------------------------------------
// Implicit surface — figure-8 Klein bottle equation
// ---------------------------------------------------------------------------

static float wsMapV(float3 p) {
    float x = p.x, y = p.y, z = p.z;
    float r2 = x*x + y*y + z*z;
    float a  = r2 + 2.0f*y - 1.0f;
    float b  = r2 - 2.0f*y - 1.0f;
    return a * (b*b - 8.0f*z*z) + 16.0f*x*z*b;
}

// ---------------------------------------------------------------------------
// Axis-aligned box intersection; returns (tNear, tFar)
// ---------------------------------------------------------------------------

static float2 wsBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
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

vertex WeirdSurfaceVertexOut weirdSurfaceVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant WeirdSurfaceUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    WeirdSurfaceVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment — Newton-step + bisection implicit surface marcher
// ---------------------------------------------------------------------------

fragment float4 weirdSurfaceFragment(
    WeirdSurfaceVertexOut in [[stage_in]],
    constant WeirdSurfaceUniforms &uniforms [[buffer(0)]],
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
    float2 tBox      = wsBoxIntersect(eye, viewDir, halfExt);
    if (!insideBox && tBox.x > tBox.y) { discard_fragment(); }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float tEnd   = tBox.y;
    if (tEnd <= tStart) { discard_fragment(); }

    // Scene space: cube ±1 → ±3.5 scene units.
    // Klein bottle bounding sphere has radius ~3; the cube contains it with margin.
    const float sceneScale = 3.5f;

    // Flip z: aligns cube local-space convention with ShaderToy scene +Z forward.
    float3 ro_entry = (eye + viewDir * (tStart + 0.001f));
    float3 ro_s = float3(ro_entry.x, ro_entry.y, -ro_entry.z) * sceneScale;
    float3 rd_s = float3(viewDir.x, viewDir.y, -viewDir.z);

    // Auto-rotation — mirrors original: t1 = t2 = iTime * 0.3 (no mouse in visionOS)
    float rotT = uniforms.time * 0.3f;

    // Rotate scene ray so the bottle spins inside the fixed cube.
    // GLSL: q.yz *= rot(t1); q.xz *= rot(t2)
    // Metal: q.yz = wsRot(t1) * q.yz; q.xz = wsRot(t2) * q.xz  (same transformation)
    ro_s.yz = wsRot(rotT) * ro_s.yz;
    ro_s.xz = wsRot(rotT) * ro_s.xz;
    rd_s.yz = wsRot(rotT) * rd_s.yz;
    rd_s.xz = wsRot(rotT) * rd_s.xz;

    // Maximum march distance in scene units (bounded by cube exit)
    float t_exit = (tEnd - tStart) * sceneScale;

    float3 col   = float3(0.0f);
    float  alpha = 1.0f;

    float t      = 0.0f;
    float v_curr = wsMapV(ro_s + rd_s * t);

    // -----------------------------------------------------------------------
    // Newton-step march — 130 iterations matching original
    // -----------------------------------------------------------------------
    for (int i = 0; i < 130; i++) {
        if (alpha < 0.01f || t > t_exit) break;

        float3 q = ro_s + rd_s * t;

        // Directional derivative estimate along rd_s
        float eps      = 0.01f;
        float v_eps    = wsMapV(q + rd_s * eps);
        float dirDeriv = abs(v_eps - v_curr) / eps;
        if (dirDeriv < 0.0001f) dirDeriv = 1.0f;

        // Newton step distance estimate
        float de = abs(v_curr) / dirDeriv;
        float dt = clamp(de * 0.5f, 0.01f, 0.4f);

        float  t_next = t + dt;
        float3 q_next = ro_s + rd_s * t_next;
        float  v_next = wsMapV(q_next);

        if (v_curr * v_next < 0.0f) {
            // Sign change detected — bisection refinement (8 iterations)
            float ta = t, tb = t_next;
            float va = v_curr;

            for (int b = 0; b < 8; b++) {
                float tm = (ta + tb) * 0.5f;
                float vm = wsMapV(ro_s + rd_s * tm);
                if (va * vm <= 0.0f) {
                    tb = tm;
                } else {
                    ta = tm;
                    va = vm;
                }
            }

            float  t_hit = (ta + tb) * 0.5f;
            float3 q_hit = ro_s + rd_s * t_hit;

            // Surface normal via central difference (step 0.005)
            const float2 e = float2(0.005f, 0.0f);
            float3 g_hit = float3(
                wsMapV(q_hit + e.xyy) - wsMapV(q_hit - e.xyy),
                wsMapV(q_hit + e.yxy) - wsMapV(q_hit - e.yxy),
                wsMapV(q_hit + e.yyx) - wsMapV(q_hit - e.yyx));
            float3 n_obj = normalize(g_hit);

            // Normal and rd_s are already in the same (rotated) space — no unrotation needed.
            if (dot(n_obj, rd_s) > 0.0f) n_obj = -n_obj;

            // Spherical UV coordinates for grid pattern
            float r      = length(q_hit);
            float theta  = acos(clamp(q_hit.y / max(r, 1.0e-5f), -1.0f, 1.0f)) / M_PI_F;
            // GLSL atan(z, x) two-arg = Metal atan2(z, x)
            float phi    = atan2(q_hit.z, q_hit.x) / (2.0f * M_PI_F);

            float g1   = abs(fract(theta * 12.0f) - 0.5f);
            float g2   = abs(fract(phi   * 24.0f) - 0.5f);
            float grid = smoothstep(0.04f, 0.0f, min(g1, g2));

            float  ndotv     = clamp(dot(n_obj, -rd_s), 0.0f, 1.0f);
            float  fresnel   = pow(1.0f - ndotv, 3.0f);

            float3 baseCol    = float3(0.05f, 0.10f, 0.30f);
            float3 gridCol    = float3(0.30f, 0.60f, 1.00f);
            float3 surfaceCol = mix(baseCol, gridCol, grid)
                              + fresnel * float3(0.5f, 0.8f, 1.0f);

            float surfaceAlpha = clamp(0.4f + grid * 0.6f + fresnel * 0.3f, 0.0f, 1.0f);

            col   += alpha * surfaceAlpha * surfaceCol;
            alpha *= (1.0f - surfaceAlpha);

            // Step slightly past surface to continue looking for inner layers
            t      = t_hit + 0.005f;
            v_curr = wsMapV(ro_s + rd_s * t);
        } else {
            t      = t_next;
            v_curr = v_next;
        }
    }

    // Dark background composited with remaining alpha; tone-map with 1-exp
    float3 bg = float3(0.02f, 0.03f, 0.06f);
    col += alpha * bg;
    col  = 1.0f - exp(-col * 1.5f);

    return float4(col, 1.0f);
}
