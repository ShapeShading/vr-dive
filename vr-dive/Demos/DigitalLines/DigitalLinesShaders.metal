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

// Star / sparkle glyph centred at uv origin.
static float dlStar(float2 uv, float size) {
    float d    = length(uv);
    float rays = max(0.0f, 1.0f - abs(uv.x * uv.y * 1000.0f));
    return (0.005f * size / d + rays * 0.05f * size) * smoothstep(0.15f, 0.0f, d);
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
// Fragment shader — faithful 3D port of the original mainImage() loop.
//
// Instead of screen UV, we derive an equivalent 2D coordinate from the per-eye
// ray direction using world-fixed axes (up = world Y).  This gives:
//   • Correct stereo parallax: each eye's slightly different ray direction
//     produces a slightly different UV → the dodecahedron is perceived in 3D.
//   • Stable orientation: the wireframe rows/columns don't tilt with head pose.
//   • Unbounded content: the dodecahedron exists in all UV directions; nothing
//     is clipped at the cube boundary.
// ---------------------------------------------------------------------------
fragment float4 digitalLinesFragment(
    DigitalLinesVertexOut in [[stage_in]],
    constant DigitalLinesUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center   = uniforms.objectCenter.xyz;
    float  scale    = uniforms.cubeScale;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 ro = (camWorld - center) / scale;       // camera in scene/object space
    float3 rd = normalize(in.worldPos - camWorld); // ray direction (world → fragment)

    // Camera forward = direction from camera toward cube centre.
    float  roLen = length(ro);
    float3 fwd   = roLen > 1e-4f ? normalize(-ro) : float3(0.0f, 0.0f, -1.0f);
    float  fwdComp = dot(rd, fwd);
    if (fwdComp <= 1e-5f) discard_fragment();

    // --- Perspective UV in world-fixed frame --------------------------------
    // Project rd onto the plane perpendicular to fwd, divide by fwdComp to
    // get a tangent-space coordinate (equivalent to the original's NDC UV).
    // Using world Y keeps the pattern orientation stable across head poses.
    float3 rdPerp     = rd - fwdComp * fwd;
    float3 worldUp    = float3(0.0f, 1.0f, 0.0f);
    float3 fixedUp    = normalize(worldUp - dot(worldUp, fwd) * fwd);  // Gram-Schmidt
    float3 fixedRight = normalize(cross(fwd, fixedUp));
    // Scale factor ≈ 1.0: the 1 m half-extent cube at 2.1 m subtends ±~0.47
    // in tangent space, matching the original shader's y range of [-0.5, 0.5].
    float2 uv = float2(dot(rdPerp, fixedRight), dot(rdPerp, fixedUp)) / fwdComp;

    // --- Original mainImage logic (ported from GLSL) -----------------------
    float3 colorA = float3(1.1f, 0.2f, 0.0f);   // _ColorA
    float3 colorB = float3(1.0f, 1.2f, 0.5f);   // _ColorB
    float3 colorC = float3(0.0f, 0.8f, 1.2f);   // _ColorC
    // _Speed = 0.3
    float  globalTime = -uniforms.time * 0.3f;

    float3 col = float3(0.0f);

    // _Layers = 8
    for (int i = 0; i < 8; i++) {
        float fi = float(i);
        float layerProgress = fract((fi / 8.0f) - fract(globalTime));
        float layerScale    = pow(2.55f, layerProgress * 3.0f) * 0.1f;
        float mask          = sin(layerProgress * 3.14159265f);

        float3 layerCol = dlLerp3(colorA, colorB, colorC, layerProgress);

        float3x3 transform = dlRotX(uniforms.time * 0.3f + fi) * dlRotY(uniforms.time * 0.2f);

        for (int n = 0; n < 30; n++) {
            float3 p1 = transform * (DL_VERTS[DL_EDGES[n * 2    ]] * layerScale);
            float3 p2 = transform * (DL_VERTS[DL_EDGES[n * 2 + 1]] * layerScale);

            // Simple perspective projection (camera at z = 5 in scene space).
            float z1 = 1.0f / (5.0f - p1.z);
            float z2 = 1.0f / (5.0f - p2.z);
            float2 a = p1.xy * z1;
            float2 b = p2.xy * z2;

            // Closest point on segment a→b to uv; compute distance.
            float2 pa = uv - a;
            float2 ba = b - a;
            float  h  = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
            float  d  = length(pa - ba * h);

            float line = smoothstep(0.003f, 0.0f, d);
            col += layerCol * line * mask * (p1.z + 3.0f) * 0.5f;

            // Stars at the first 20 vertices (same as original).
            if (n < 20) {
                float3 pStar   = transform * (DL_VERTS[n] * layerScale);
                float2 starPos = pStar.xy * (1.0f / (5.0f - pStar.z));
                float  sparkle = sin(uniforms.time * 10.0f + fi) * 0.5f + 0.5f;
                col += layerCol * dlStar(uv - starPos, 0.5f) * mask * sparkle;
            }
        }
    }

    // Centre glow (_Glow = 1.5).
    float3 glowCol = mix(colorA, colorC, sin(uniforms.time) * 0.5f + 0.5f);
    col += glowCol * (1.5f * 0.02f / (length(uv) + 0.1f));

    // tanh tonemap (identical to original).
    col = tanh(col);
    return float4(col, 1.0f);
}
