// ApollonianWiresShaders.metal
// 3D visionOS adaptation of an Apollonian wire fractal (ShaderToy Wlsfzs).
//
// Original GLSL source:
//   https://www.shadertoy.com/view/Wlsfzs
//   Author unknown — ported to Metal / visionOS cube-container ray march
//   by the vr-dive project.
//
// Rendering strategy: rasterise the 6 faces of a world-space cube; each
// fragment reconstructs a ray from the camera through the cube surface and
// marches inward.  Inside-camera support is handled by setting tStart = 0.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared types (must match ApollonianWiresTypes.swift)
// ---------------------------------------------------------------------------

struct ApollonianWiresUniforms {
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

struct ApollonianWiresVertexOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ---------------------------------------------------------------------------
// Vertex
// ---------------------------------------------------------------------------

vertex ApollonianWiresVertexOut apollonianWiresVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant ApollonianWiresUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    ApollonianWiresVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Rotation matrix — mirrors GLSL `#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))`.
//
// GLSL convention: `p.xz *= rot(a)` is a row-vector × matrix operation.
// Metal convention: `M * v` is matrix × column-vector.
// With the same column-major construction float2x2(float2(c,s), float2(-s,c)):
//   (M*v)[0] = c*v.x + (-s)*v.z = same as GLSL result.x  ✓
//   (M*v)[1] = s*v.x +   c*v.z  = same as GLSL result.z  ✓
static float2x2 awRot(float a) {
    float c = cos(a), s = sin(a);
    return float2x2(float2(c, s), float2(-s, c));
}

// Axis-aligned box intersection. Returns (tNear, tFar).
static float2 awBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
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
// Distance field — direct port of GLSL map().
//
// Translation notes:
//   p.xz *= rot(t) → p.xz = awRot(t) * p.xz   (see convention comment above)
//   p.xy *= rot(t) → p.xy = awRot(t) * p.xy
//   Both rotations are sequential; the second uses the x already modified by the first.
//   dot(p,p) is clamped to avoid division by zero in the fold.
// ---------------------------------------------------------------------------

static float awMap(float3 p, float t) {
    p.xz = awRot(t * 0.5f) * p.xz;
    p.xy = awRot(t * 0.5f) * p.xy;

    float s = 2.0f;
    p = abs(p);

    bool modeA = (fract(t * 0.5f) < 0.7f);
    for (int i = 0; i < 12; i++) {
        p = 1.0f - abs(p - 1.0f);
        float dd = max(dot(p, p), 1.0e-6f);
        float r2;
        if (modeA) {
            r2 = 1.2f / dd;
        } else {
            r2 = (i % 3 == 1) ? 1.3f : 1.3f / dd;
        }
        p  *= r2;
        s  *= r2;
    }

    return length(cross(p, normalize(float3(1.0f)))) / s - 0.003f;
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

fragment float4 apollonianWiresFragment(
    ApollonianWiresVertexOut in [[stage_in]],
    constant ApollonianWiresUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center  = uniforms.objectCenter.xyz;
    float  scale   = max(uniforms.cubeScale, 1.0e-4f);
    float3 halfExt = float3(1.0f);    // local-space ±1 cube

    // Camera and surface point in local cube space
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 eye         = (cameraWorld - center) / scale;
    float3 surfacePos  = (in.worldPos - center) / scale;
    float3 viewDir     = normalize(surfacePos - eye);

    // Box intersection to clip the ray to the cube volume
    bool   insideBox = all(abs(eye) < halfExt - 1.0e-3f);
    float2 tBox      = awBoxIntersect(eye, viewDir, halfExt);

    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float tEnd   = tBox.y;
    if (tEnd <= tStart) {
        discard_fragment();
    }

    // Entry point (tiny offset to avoid self-intersection artifacts)
    float3 ro = eye + viewDir * (tStart + 0.001f);

    // Scale to scene space where the fractal has interesting structure.
    // The original camera sits at x ∈ [3, 8]; sceneScale = 5 maps the cube
    // surface (±1 local) to ±5 scene units — right in that range.
    const float sceneScale = 5.0f;
    float3 roScene = ro * sceneScale;
    float  maxMarchDist = (tEnd - tStart) * sceneScale;

    // Ray march — matches original loop structure: i tracks iterations for brightness.
    float t = uniforms.time;
    float3 p = roScene;
    float h = 0.0f;
    float hitIter = 120.0f;

    for (float i = 1.0f; i < 120.0f; i += 1.0f) {
        p = roScene + viewDir * h;
        float d = awMap(p, t);
        if (d < 0.0001f) {
            hitIter = i;
            break;
        }
        h += d;
        if (h > maxMarchDist) {
            // Passed through the cube without hitting — return background
            return float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    // Color from original: 30 * vec3(cos(p*0.8)*0.5+0.5) / i
    float3 col = 30.0f * (cos(p * 0.8f) * 0.5f + 0.5f) / hitIter;
    col = clamp(col, 0.0f, 1.0f);
    return float4(col, 1.0f);
}
