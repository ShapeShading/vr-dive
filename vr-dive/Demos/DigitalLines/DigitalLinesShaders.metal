// DigitalLinesShaders.metal
// "Digital Lines" — cube-portal adaptation of Shadertoy "scf3zB"
// Original: https://www.shadertoy.com/view/scf3zB
// Adapted for visionOS Metal stereo rendering.
//
// Design:
//   A 2 m cube acts as the portal container.  For every fragment on the cube
//   surface (or back-faces when the camera is inside), the fragment shader
//   projects the view ray onto a world-fixed 2D plane and evaluates the
//   original Shadertoy's dodecahedron wireframe loop at that UV.
//   The inner scene is unbounded — the dodecahedron extends wherever it
//   happens to be in UV space regardless of where the cube surface is.

#include <metal_stdlib>
using namespace metal;

struct DigitalLinesUniforms {
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

struct DigitalLinesVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ---------------------------------------------------------------------------
// Constants — dodecahedron geometry (from original shader).
// phi = golden ratio = (1 + sqrt(5)) / 2 ≈ 1.618
// ---------------------------------------------------------------------------
static constant float DL_PHI  = 1.6180339887f;
static constant float DL_INVP = 0.6180339887f;   // 1 / phi
static constant float3 DL_BOX_HALF = float3(1.0f);

static constant float3 DL_VERTS[20] = {
    {-1.0f,-1.0f,-1.0f}, { 1.0f,-1.0f,-1.0f}, { 1.0f, 1.0f,-1.0f}, {-1.0f, 1.0f,-1.0f},
    {-1.0f,-1.0f, 1.0f}, { 1.0f,-1.0f, 1.0f}, { 1.0f, 1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f},
    { 0.0f,-DL_INVP,-DL_PHI}, { 0.0f, DL_INVP,-DL_PHI},
    { 0.0f, DL_INVP, DL_PHI}, { 0.0f,-DL_INVP, DL_PHI},
    {-DL_INVP,-DL_PHI, 0.0f}, { DL_INVP,-DL_PHI, 0.0f},
    { DL_INVP, DL_PHI, 0.0f}, {-DL_INVP, DL_PHI, 0.0f},
    {-DL_PHI, 0.0f,-DL_INVP}, { DL_PHI, 0.0f,-DL_INVP},
    { DL_PHI, 0.0f, DL_INVP}, {-DL_PHI, 0.0f, DL_INVP}
};

// 30 edges × 2 vertex indices = 60 entries.
static constant int DL_EDGES[60] = {
     0, 8,  1, 8,  8, 9,  2, 9,  3, 9,
     0,16,  3,16, 16,19,  4,19,  7,19,
     1,17,  2,17, 17,18,  5,18,  6,18,
     4,12,  5,12, 12,13,  0,13,  1,13,
     2,14,  3,14, 14,15,  6,15,  7,15,
     4,11,  5,11, 10,11,  6,10,  7,10
};

// ---------------------------------------------------------------------------
// Helper functions (translated from original GLSL, all names prefixed "dl")
// ---------------------------------------------------------------------------

// Three-stop colour ramp: a→b for t in [0,0.5], b→c for t in [0.5,1].
static float3 dlLerp3(float3 a, float3 b, float3 c, float t) {
    if (t < 0.5f) return mix(a, b, t * 2.0f);
    else          return mix(b, c, (t - 0.5f) * 2.0f);
}

// Rotation matrices — Metal float3x3 is column-major, matching GLSL mat3().
// GLSL mat3(1,0,0, 0,c,-s, 0,s,c): col0=(1,0,0), col1=(0,c,-s), col2=(0,s,c)
static float3x3 dlRotX(float a) {
    float s = sin(a), c = cos(a);
    return float3x3(float3(1,0,0), float3(0,c,-s), float3(0,s,c));
}
// GLSL mat3(c,0,s, 0,1,0, -s,0,c): col0=(c,0,s), col1=(0,1,0), col2=(-s,0,c)
static float3x3 dlRotY(float a) {
    float s = sin(a), c = cos(a);
    return float3x3(float3(c,0,s), float3(0,1,0), float3(-s,0,c));
}

static float2 dlBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float dlRaySegmentDistance(float3 ro, float3 rd, float3 a, float3 b, thread float &rayT) {
    float3 seg = b - a;
    float3 w0 = ro - a;
    float segLen2 = max(dot(seg, seg), 1.0e-5f);
    float bDot = dot(rd, seg);
    float dDot = dot(rd, w0);
    float eDot = dot(seg, w0);
    float denom = segLen2 - bDot * bDot;

    float sc = 0.0f;
    float tc = 0.0f;
    if (abs(denom) > 1.0e-5f) {
        sc = clamp((bDot * eDot - segLen2 * dDot) / denom, 0.0f, 10.0f);
        tc = clamp((eDot + bDot * sc) / segLen2, 0.0f, 1.0f);
    } else {
        tc = clamp(eDot / segLen2, 0.0f, 1.0f);
    }

    sc = max(dot(a + seg * tc - ro, rd), 0.0f);
    float3 closestRay = ro + rd * sc;
    float3 closestSeg = a + seg * tc;
    rayT = sc;
    return length(closestRay - closestSeg);
}

static float dlRayPointDistance(float3 ro, float3 rd, float3 p, thread float &rayT) {
    rayT = max(dot(p - ro, rd), 0.0f);
    float3 closestRay = ro + rd * rayT;
    return length(closestRay - p);
}

// ---------------------------------------------------------------------------
// Vertex shader — positions the cube container in world space.
// ---------------------------------------------------------------------------
vertex DigitalLinesVertexOut digitalLinesVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant DigitalLinesUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    DigitalLinesVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — rebuild the original layered dodecahedron as true 3D
// wireframe geometry anchored to the cube center.  Each eye traces through the
// same local-space structure, so stereo comes from real ray/segment distances.
// ---------------------------------------------------------------------------
fragment float4 digitalLinesFragment(
    DigitalLinesVertexOut in [[stage_in]],
    constant DigitalLinesUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center   = uniforms.objectCenter.xyz;
    float  scale    = max(uniforms.cubeScale, 1.0e-4f);
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 eye = (camWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 rd = normalize(surfacePos - eye);

    bool insideOuter = all(abs(eye) < DL_BOX_HALF - 1.0e-3f);
    float2 tOuter = dlBoxIntersect(eye, rd, DL_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 ro = eye;

    float3 colorA = float3(1.1f, 0.2f, 0.0f);   // _ColorA
    float3 colorB = float3(1.0f, 1.2f, 0.5f);   // _ColorB
    float3 colorC = float3(0.0f, 0.8f, 1.2f);   // _ColorC
    float  globalTime = -uniforms.time * 0.3f;

    float3 col = float3(0.0f);

    for (int i = 0; i < 8; i++) {
        float fi = float(i);
        float layerProgress = fract((fi / 8.0f) - fract(globalTime));
        float layerScale    = pow(2.1f, layerProgress * 2.6f) * 0.14f;
        float mask          = sin(layerProgress * 3.14159265f);

        float3 layerCol = dlLerp3(colorA, colorB, colorC, layerProgress);

        float3x3 transform = dlRotX(uniforms.time * 0.3f + fi) * dlRotY(uniforms.time * 0.2f);
        float lineWidth = mix(0.018f, 0.006f, layerProgress);

        for (int n = 0; n < 30; n++) {
            float3 p1 = transform * (DL_VERTS[DL_EDGES[n * 2    ]] * layerScale);
            float3 p2 = transform * (DL_VERTS[DL_EDGES[n * 2 + 1]] * layerScale);

            float rayT = 0.0f;
            float d = dlRaySegmentDistance(ro, rd, p1, p2, rayT);
            if (rayT < tStart) {
                continue;
            }
            float line = smoothstep(lineWidth, 0.0f, d);
            float depthFade = exp(-0.28f * rayT);
            col += layerCol * line * mask * depthFade;

            if (n < 20) {
                float3 pStar   = transform * (DL_VERTS[n] * layerScale);
                float starT = 0.0f;
                float dStar = dlRayPointDistance(ro, rd, pStar, starT);
                if (starT < tStart) {
                    continue;
                }
                float  sparkle = sin(uniforms.time * 10.0f + fi) * 0.5f + 0.5f;
                float star = smoothstep(0.03f, 0.0f, dStar);
                col += layerCol * star * mask * sparkle * exp(-0.22f * starT) * 0.55f;
            }
        }
    }

    float3 glowCol = mix(colorA, colorC, sin(uniforms.time) * 0.5f + 0.5f);
    float centerT = max(dot(-ro, rd), 0.0f);
    float centerDist = length(ro + rd * centerT);
    col += glowCol * (1.5f * 0.008f / (centerDist + 0.1f)) * exp(-0.35f * centerT);

    col = tanh(col);
    return float4(col, 1.0f);
}
