// KuKoShaders.metal
// 3D visionOS adaptation of "KuKo Day 384" (ShaderToy NXfGDl).
//
// Original GLSL source:
//   https://www.shadertoy.com/view/NXfGDl
//   DDA algorithm from @xor: https://www.shadertoy.com/view/XctSz8
//   Ported to Metal / visionOS cube-container by the vr-dive project.
//
// Technique: DDA voxel traversal with volumetric Henyey-Greenstein lighting.
// The cube is a window into an infinite voxel field that drifts over time.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared types (must match KuKoTypes.swift)
// ---------------------------------------------------------------------------

struct KuKoUniforms {
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

struct KuKoVertexOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ---------------------------------------------------------------------------
// Scene functions — direct translations of GLSL originals
// ---------------------------------------------------------------------------

// Boolean solid test (returns true = solid, stop marching)
static bool kukMap(float3 p) {
    return dot(sin(p * 0.13f), cos(p.yzx * 0.4384f)) + p.y * 0.0561f > 0.9f;
}

// Scalar density field (for volumetric light accumulation)
static float kukMap2(float3 p) {
    return dot(sin(p * 0.13f), cos(p.yzx * 0.4384f)) + p.y * 0.061f;
}

// Scalar hash for palette index
static float kukHash(float3 p) {
    return fract(sin(dot(p, float3(127.1f, 311.7f, 411.7f))) * 43758.5453f);
}

// Vector hash — used for sub-voxel jitter (@Shane technique)
static float3 kukHash33(float3 p) {
    float n = sin(dot(p, float3(7.0f, 157.0f, 113.0f)));
    return fract(float3(2097152.0f, 262144.0f, 32768.0f) * n);
}

// Axis-aligned box intersection; returns (tNear, tFar)
static float2 kukBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
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

vertex KuKoVertexOut kuKoVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant KuKoUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    KuKoVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment — DDA voxel traversal + Henyey-Greenstein volumetric lighting
// ---------------------------------------------------------------------------

fragment float4 kuKoFragment(
    KuKoVertexOut in [[stage_in]],
    constant KuKoUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center  = uniforms.objectCenter.xyz;
    float  scale   = max(uniforms.cubeScale, 1.0e-4f);
    float3 halfExt = float3(1.0f);   // local-space ±1 cube

    // Camera and surface in local cube space
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 eye         = (cameraWorld - center) / scale;
    float3 surfacePos  = (in.worldPos - center) / scale;
    float3 viewDir     = normalize(surfacePos - eye);

    // Box intersection
    bool   insideBox = all(abs(eye) < halfExt - 1.0e-3f);
    float2 tBox      = kukBoxIntersect(eye, viewDir, halfExt);
    if (!insideBox && tBox.x > tBox.y) { discard_fragment(); }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float tEnd   = tBox.y;
    if (tEnd <= tStart) { discard_fragment(); }

    // Entry point in local cube space → scene space.
    // sceneScale maps the cube ±1 to ±8 scene units.
    // The time-based offset mirrors the original flying camera:
    //   ro = vec3(-2, 2, iTime * SPEED)
    const float sceneScale = 8.0f;
    const float SPEED      = 7.0f;
    float t   = uniforms.time;
    // Flip z: ShaderToy/GLSL content runs in +Z; cube local -Z (viewDir.z ≈ -1
    // when looking at the front face) must map to scene +Z.
    float3 ro_entry = (eye + viewDir * (tStart + 0.001f));
    float3 ro = float3(ro_entry.x, ro_entry.y, -ro_entry.z) * sceneScale
                + float3(-2.0f, 2.0f, t * SPEED);

    // Ray direction stays normalized — DDA step sizes handle the rest.
    float3 rd = float3(viewDir.x, viewDir.y, -viewDir.z);

    // Prevent division-by-zero in DDA step size computation
    // (mirrors GLSL: rd += vec3(rd.x==0, rd.y==0, rd.z==0) * 1e-5)
    if (rd.x == 0.0f) rd.x = 1.0e-5f;
    if (rd.y == 0.0f) rd.y = 1.0e-5f;
    if (rd.z == 0.0f) rd.z = 1.0e-5f;

    // Colour palette (only indices 0–2 are reached; full array retained for fidelity)
    float3 colArr[7] = {
        float3(0.100f, 0.100f, 0.100f),
        float3(0.659f, 0.000f, 0.353f),
        float3(0.518f, 0.043f, 0.322f),
        float3(0.349f, 0.027f, 0.000f),
        float3(0.383f, 0.782f, 1.000f),
        float3(0.542f, 0.549f, 0.625f),
        float3(0.277f, 0.133f, 0.137f)
    };

    // ---------------------------------------------------------------------------
    // DDA initialisation
    // ---------------------------------------------------------------------------
    float3 axisDir    = sign(rd);
    float3 stepDir    = 1.0f / abs(rd);                      // t-length per unit step per axis
    float3 vox        = floor(ro);
    // Initial crossing depths: how far to first boundary on each axis
    float3 xyCrossing = ((vox - ro) * axisDir + 0.6f) * stepDir;

    // State — initialised to safe defaults
    float3 axis     = float3(0.0f);
    float3 p        = float3(0.0f);
    float3 vox2     = float3(0.0f);  // saved voxel before step (original uses vox2 but only saves it)

    float accum         = 0.0f;
    float voxDt         = 0.0f;
    float att           = 0.0f;
    float steps         = 0.0f;
    float transmittance = 1.0f;
    float stepL         = 0.0f;
    float henyey        = 0.0f;
    float newXyCros     = 0.0f;
    float edge          = 0.0f;
    float3 lightAcc     = float3(0.0f);

    // Light position in scene space (mirrors original: L = vec3(-2, 2, iTime*SPEED + 15))
    float3 L = float3(-2.0f, 2.0f, t * SPEED + 15.0f);

    // ---------------------------------------------------------------------------
    // DDA traversal — translated directly from GLSL mainImage loop
    // ---------------------------------------------------------------------------
    for (int i = 1; i < 80; i++) {
        // Hit test: if current voxel is solid, stop
        if (kukMap(vox)) break;

        // Sub-voxel jitter to suppress stepping artefacts (@Shane)
        xyCrossing += kukHash33(vox + ro) * stepDir * 0.005f;

        // Attenuation from light distance
        voxDt  = length((vox + 0.5f) - L) * 0.2f;
        att    = 1.0f / (80.0f + voxDt * voxDt);
        accum += att;
        steps += 1.0f;

        // Axis selection: which boundary is crossed next?
        axis = xyCrossing.x < xyCrossing.z
             ? (xyCrossing.x < xyCrossing.y ? float3(1,0,0) : float3(0,1,0))
             : (xyCrossing.z < xyCrossing.y ? float3(0,0,1) : float3(0,1,0));

        // World position at boundary crossing
        p = ro + dot(xyCrossing, axis) * rd;

        // Henyey-Greenstein phase function (g = 0.45)
        float angleL = dot(rd, normalize(p - L));
        float g      = 0.45f;
        henyey = (1.0f - g) / (4.0f * M_PI_F
                 * pow(1.0f + g*g - 2.0f*g*cos(angleL), 2.5f));

        // Advance voxel
        vox2       = vox;
        vox       += axis * axisDir;
        newXyCros  = dot(xyCrossing, axis);
        xyCrossing += axis * stepDir;

        // Volumetric shadow / light accumulation
        stepL          = dot(xyCrossing, axis) - newXyCros * 0.07f;
        transmittance *= exp(-0.000712f * stepL);
        lightAcc      += max(kukMap2(p), 0.0f) * henyey * transmittance * stepL * att;

        // Edge detection via crossing depths
        float dx = xyCrossing.x;
        float dy = xyCrossing.y + 0.5f;
        float dz = xyCrossing.z - 0.2f;
        float d2 = min(dy, max(dz, dy));
        edge = 1.0f - smoothstep(-5.0f, 0.4f, d2 - dz);
    }

    // ---------------------------------------------------------------------------
    // Surface shading — same as original post-loop code
    // ---------------------------------------------------------------------------
    float3 nor = axisDir * axis;
    float3 hit = p + rd * dot(xyCrossing - stepDir, axis);
    float  NoV = clamp(dot(nor, -hit), 0.0f, 1.0f);
    float  rim = pow(1.0f - NoV, 3.0f) * 10.0f;

    // Grid lines per unit voxel
    const float SCALE = 1.0f;
    float3 grid    = max(float3(0.7f) - 2.0f * abs(fract(p + 0.5f)) * SCALE, 0.0f);
    float3 newGrid = mix(float3(0.061f), float3(0.0f), grid.x + grid.y + grid.z);

    // Random palette index: hash * 2 + 0.4 → float in [0.4, 2.4) → int 0/1/2
    float rand = kukHash(floor(vox)) * 2.0f + 0.4f;
    int   ci   = max(0, min(6, int(rand)));

    float3 col = colArr[ci] * 2.0f
               + float3(0.902f, 0.059f, 0.678f) * edge
                 * (1.5f + sin(t * 2.0f) * 0.5f + 0.5f) * rim;

    col  = lightAcc * col;
    col -= newGrid;

    float fog = steps / 80.0f;
    col = mix(col, float3(0.2f, 0.24f, 0.3f), fog);

    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}
