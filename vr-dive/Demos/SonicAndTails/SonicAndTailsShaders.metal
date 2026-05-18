// SonicAndTailsShaders.metal
// 3D visionOS adaptation of "Sonic & Tails" (ShaderToy s3X3WX).
//
// Original GLSL:
//   https://www.shadertoy.com/view/s3X3WX
//   "Sonic & Tails" by Noztol
//   Ported to Metal / visionOS cube-container by the vr-dive project.
//
// Technique: SDF ray marching inside a procedural spiral pipe with golden rings,
//   volumetric neon-rail glow, ring halos, and a cloud-backed sky for missed rays.
//
// GLSL → Metal translation notes:
//   • mat3 is column-major in both; constructor args map 1-to-1 into float3x3 columns.
//   • GLSL `v.xz *= mat2(c,-s,s,c)` (row-vec) = Metal `v.xz = float2x2(float2(c,s), float2(-s,c)) * v.xz`
//   • GLSL `atan(y, x)` two-arg form = Metal `atan2(y, x)`.
//   • `iTime` inside map() is passed explicitly as parameter `t`.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Shared types  (must match SonicAndTailsTypes.swift)
// ---------------------------------------------------------------------------

struct SonicAndTailsUniforms {
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

struct SonicAndTailsVertexOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ---------------------------------------------------------------------------
// FBM rotation matrix
// Original: mat3(0.00, 1.60, 1.20, -1.60, 0.72, -0.96, -1.20, -0.96, 1.28)
// GLSL column-major col0=(0.00,1.60,1.20), col1=(-1.60,0.72,-0.96), col2=(-1.20,-0.96,1.28)
// ---------------------------------------------------------------------------

constant float3x3 satM3 = float3x3(
    float3( 0.00f,  1.60f,  1.20f),
    float3(-1.60f,  0.72f, -0.96f),
    float3(-1.20f, -0.96f,  1.28f)
);

// ---------------------------------------------------------------------------
// Procedural noise
// ---------------------------------------------------------------------------

static float satHash(float n) {
    return fract(sin(n) * 43758.5453123f);
}

static float satNoise(float3 x) {
    float3 p = floor(x);
    float3 f = fract(x);
    f = f * f * (3.0f - 2.0f * f);
    float n = p.x + p.y * 57.0f + 113.0f * p.z;
    return mix(
        mix(mix(satHash(n +   0.0f), satHash(n +   1.0f), f.x),
            mix(satHash(n +  57.0f), satHash(n +  58.0f), f.x), f.y),
        mix(mix(satHash(n + 113.0f), satHash(n + 114.0f), f.x),
            mix(satHash(n + 170.0f), satHash(n + 171.0f), f.x), f.y), f.z);
}

static float satFbm(float3 p) {
    float f = 0.5000f * satNoise(p); p = satM3 * p * 1.1f;
    f += 0.2500f * satNoise(p);      p = satM3 * p * 1.2f;
    f += 0.1666f * satNoise(p);      p = satM3 * p;
    f += 0.0834f * satNoise(p);
    return f;
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

// Spiral tunnel centre at path parameter z
static float3 satP(float z) {
    return float3(cos(z * 0.02f) * 12.0f + cos(z * 0.05f) * 6.0f,
                  sin(z * 0.015f) * 8.0f,
                  z);
}

static float satSdTorus(float3 p, float2 t) {
    float2 q = float2(length(p.xy) - t.x, p.z);
    return length(q) - t.y;
}

// SDF map — returns float2(dist, materialID)
// material 1.0 = track wall, 2.0 = golden ring
static float2 satMap(float3 p, float t) {
    float3 path  = satP(p.z);
    float3 loc   = p - path;

    // Pipe interior (PIPE_RAD = 10) capped by track floor
    float pipe   = 10.0f - length(loc.xy);
    float dTrack = max(pipe, loc.y - 3.8f);
    float2 res   = float2(dTrack, 1.0f);

    // Golden rings every 20 units along z
    float zIndex = floor(p.z / 20.0f) * 20.0f;
    for (float i = -1.0f; i <= 1.0f; i++) {
        float  currZ  = zIndex + i * 20.0f;
        float3 rPos   = satP(currZ);
        rPos.x += sin(currZ * 0.15f) * 6.0f;  // RING_FREQ=0.15, RING_AMP=6
        rPos.y -= 4.5f;
        float3 rLocal = p - rPos;

        // GLSL: rLocal.xz *= mat2(c,-s,s,c)  (row-vec × col-major mat2)
        // Metal col-vec equiv: rLocal.xz = float2x2(col0=(c,s), col1=(-s,c)) * rLocal.xz
        float rot = t * 4.0f;
        float c = cos(rot), s = sin(rot);
        rLocal.xz = float2x2(float2(c, s), float2(-s, c)) * rLocal.xz;

        float dRing = satSdTorus(rLocal, float2(1.8f, 0.3f));
        if (dRing < res.x) res = float2(dRing, 2.0f);
    }
    return res;
}

static float3 satGetNormal(float3 p, float t) {
    const float2 e = float2(0.01f, 0.0f);
    return normalize(float3(
        satMap(p + e.xyy, t).x - satMap(p - e.xyy, t).x,
        satMap(p + e.yxy, t).x - satMap(p - e.yxy, t).x,
        satMap(p + e.yyx, t).x - satMap(p - e.yyx, t).x));
}

// Axis-aligned box intersection; returns (tNear, tFar)
static float2 satBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
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

vertex SonicAndTailsVertexOut sonicAndTailsVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant SonicAndTailsUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    SonicAndTailsVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment — SDF ray marching, track/ring/sky shading
// ---------------------------------------------------------------------------

fragment float4 sonicAndTailsFragment(
    SonicAndTailsVertexOut in [[stage_in]],
    constant SonicAndTailsUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center  = uniforms.objectCenter.xyz;
    float  scale   = max(uniforms.cubeScale, 1.0e-4f);
    float3 halfExt = float3(1.0f);   // cube local ±1

    // Camera and surface position in local cube space
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 eye         = (cameraWorld - center) / scale;
    float3 surfacePos  = (in.worldPos - center) / scale;
    float3 viewDir     = normalize(surfacePos - eye);

    // Box intersection
    bool   insideBox = all(abs(eye) < halfExt - 1.0e-3f);
    float2 tBox      = satBoxIntersect(eye, viewDir, halfExt);
    if (!insideBox && tBox.x > tBox.y) { discard_fragment(); }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float tEnd   = tBox.y;
    if (tEnd <= tStart) { discard_fragment(); }

    // Map cube-local entry point into scene space.
    // sceneScale=10 maps cube ±1 → ±10 scene units (≈ one tunnel diameter).
    // Virtual camera follows the tunnel path with the original lateral swing,
    // offset to the track floor level — mirrors the original ro setup.
    // sceneScale=20: tunnel radius 10 occupies ±0.5 local units, leaving the cube
    // centre open so rays through the open tunnel exit and show the far face.
    const float sceneScale = 20.0f;
    const float T          = uniforms.time * 14.0f;      // path parameter
    float camSwing         = sin(T * 0.15f) * 6.0f;     // RING_FREQ * RING_AMP
    float3 camPos          = satP(T) + float3(camSwing, -4.5f, 0.0f);

    // Flip z: ShaderToy content runs in +Z; cube local -Z must map to scene +Z.
    float3 ro_entry = (eye + viewDir * (tStart + 0.001f));
    float3 ro = float3(ro_entry.x, ro_entry.y, -ro_entry.z) * sceneScale + camPos;
    float3 rd = float3(viewDir.x, viewDir.y, -viewDir.z);

    // Maximum scene-space march distance bounded by the cube exit point
    float maxDist = (tEnd - tStart) * sceneScale;

    float  d        = 0.0f;
    float  glowLine = 0.0f;
    float  glowRing = 0.0f;
    float2 res      = float2(1.0f, 1.0f);

    // -----------------------------------------------------------------------
    // Ray march — 90 steps (matching original)
    // -----------------------------------------------------------------------
    for (int i = 0; i < 90; i++) {
        float3 p = ro + rd * d;
        res = satMap(p, uniforms.time);

        // 1. Unified track glow — neon rail lines + horizontal ring bands
        float3 path = satP(p.z);
        float3 loc  = p - path;
        if (length(loc.xy) < 12.0f && loc.y < 4.0f) {
            // GLSL atan(x, y) two-arg = Metal atan2(x, y)
            float angle    = atan2(loc.x, loc.y);
            float rails    = smoothstep(0.12f, 0.0f, abs(abs(angle) - 0.5f));
            float rings    = smoothstep(0.45f, 0.5f, abs(fract(p.z * 0.15f) - 0.5f));
            float lineMask = max(rails, rings);
            glowLine += lineMask * 0.06f / (res.x + 0.1f);
        }

        // 2. Volumetric golden-ring halo
        float  zIdx = floor(p.z / 20.0f) * 20.0f;
        float3 rPos = satP(zIdx);
        rPos.x += sin(zIdx * 0.15f) * 6.0f;
        rPos.y -= 4.5f;
        glowRing += 0.02f / (length(p - rPos) + 0.5f);

        if (res.x < 0.001f || d > maxDist) break;
        d += res.x * 0.75f;
    }

    // -----------------------------------------------------------------------
    // Surface / sky shading
    // -----------------------------------------------------------------------
    float3 col;
    float3 sunDir = normalize(float3(0.1f, 0.25f, 0.9f));

    if (d >= maxDist) {
        // Ray exited cube without hitting geometry → sky & clouds
        float sundot = clamp(dot(rd, sunDir), 0.0f, 1.0f);
        float tt     = pow(1.0f - 0.7f * rd.y, 15.0f);
        col = 0.8f * (float3(0.6f, 0.8f, 1.2f) * tt
                    + float3(0.05f, 0.2f, 0.5f) * (1.0f - tt));
        col += 0.5f * float3(1.6f, 1.4f, 1.0f) * pow(sundot, 300.0f);

        float4 cloudSum = float4(0.0f);
        for (int q = 0; q < 20; q++) {
            float h = (float(q) * 15.0f + 320.0f - ro.y) / rd.y;
            if (h > 0.0f && h < 8000.0f) {
                float3 cp  = ro + h * rd
                           + float3(0.0f, -uniforms.time * 8.0f, uniforms.time * 15.0f);
                float  den = smoothstep(0.5f, 1.0f, satFbm(cp * 0.0018f));
                float3 cCol = mix(float3(1.1f), float3(0.4f, 0.4f, 0.45f), den);
                den *= (1.0f - cloudSum.w);
                cloudSum += float4(cCol * den, den);
                if (cloudSum.w > 0.98f) break;
            }
        }
        col = mix(col, cloudSum.rgb, cloudSum.w * (1.0f - tt));
    } else {
        float3 p = ro + rd * d;
        float3 n = satGetNormal(p, uniforms.time);

        if (res.y > 1.5f) {
            // Material: Golden Ring
            float spec = pow(max(dot(reflect(rd, n), sunDir), 0.0f), 32.0f);
            col = float3(1.0f, 0.8f, 0.1f) + spec;
        } else {
            // Material: Neon Track
            float3 path  = satP(p.z);
            float3 loc   = p - path;
            float  angle = atan2(loc.x, loc.y);

            float rings = smoothstep(0.35f, 0.5f, abs(fract(p.z * 0.15f) - 0.5f));
            float rails = smoothstep(0.40f, 0.5f, abs(fract(angle * 0.418f) - 0.5f));
            float mask  = max(rails, rings);

            float3 neonBlue = float3(0.0f, 0.8f, 5.0f) * (1.0f - mask);
            float3 neonRed  = float3(4.5f, 0.2f, 0.0f) * (4.0f * mask);

            col  = mix(neonBlue, neonRed, mask);
            col  = mix(col, neonRed * 1.5f, smoothstep(3.5f, 3.8f, loc.y));
            col += pow(1.0f - max(dot(n, -rd), 0.0f), 4.0f) * float3(0.5f, 0.8f, 1.0f);
        }
    }

    // Combine volumetric glows
    col += float3(1.8f, 0.1f, 0.0f) * glowLine;  // neon rail bloom
    col += float3(1.0f, 0.6f, 0.0f) * glowRing;  // golden-ring bloom

    // tanh tone-map (matches original)
    return float4(tanh(col), 1.0f);
}
