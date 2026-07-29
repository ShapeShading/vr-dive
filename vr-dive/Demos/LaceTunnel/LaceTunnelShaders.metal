// LaceTunnelShaders.metal
// Adapted from "Lace Tunnel" by Stephane Cuillerdier (Aiekick), 2015.
// https://www.shadertoy.com/view/4sGSzc
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Renders a lace-pattern tunnel inside a 2 m × 2 m × 2 m cube container.
// Container ray-marching: from outside the box, the ray enters at the cube
// surface; when the camera is inside the box the ray starts at the camera.
// The tunnel SDF extends indefinitely in tunnel space; the cube limits the
// visible portion.  Pattern does NOT clip at the container boundary.

#include <metal_stdlib>
using namespace metal;

// ─── Uniforms ─────────────────────────────────────────────────────────────────
// Layout must match LaceTunnelUniforms in LaceTunnelTypes.swift.
struct LaceTunnelUniforms {
    float  time;
    uint   viewCount;
    float  boxScale;    // world-space scale: worldPos = localPos * boxScale + center
    float  _pad;
    float4 objectCenter;  // xyz = world-space box centre
};

struct LaceMeshVertex {
    float3 position;
    float3 normal;
};

struct LaceTunnelVertexOut {
    float4 clipPos   [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ─── Vertex shader ────────────────────────────────────────────────────────────
vertex LaceTunnelVertexOut laceTunnelVertex(
    ushort                       amplificationID [[amplification_id]],
    const device LaceMeshVertex *vertices        [[buffer(0)]],
    constant LaceTunnelUniforms &uniforms        [[buffer(1)]],
    constant float4x4           *vpMatrices      [[buffer(2)]],
    uint                         vertexID        [[vertex_id]])
{
    LaceMeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    LaceTunnelVertexOut out;
    out.clipPos  = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ─── Tunnel SDF helpers ───────────────────────────────────────────────────────
// Source: Stephane Cuillerdier (Aiekick) 2015, https://www.shadertoy.com/view/4sGSzc
// Functions prefixed lt_ to avoid name collisions with other shaders in the lib.

// Tunnel centreline at depth t (slow helix, radius 2).
static float2 lt_path(float t) {
    return float2(cos(t * 0.2f), sin(t * 0.2f)) * 2.0f;
}

// Diagonal scale matrices (column-major, same layout as GLSL mat3).
// mx = diag(1,7,7)  my = diag(7,1,7)  mz = diag(7,7,1)
constant float3x3 lt_mx = float3x3(float3(1,0,0), float3(0,7,0), float3(0,0,7));
constant float3x3 lt_my = float3x3(float3(7,0,0), float3(0,1,0), float3(0,0,7));
constant float3x3 lt_mz = float3x3(float3(7,0,0), float3(0,7,0), float3(0,0,1));

// One-tweet cellular distance (Shane's technique, via Aiekick).
static float lt_func(float3 p) {
    p = fract(p / 68.6f) - 0.5f;
    return min(min(abs(p.x), abs(p.y)), abs(p.z)) + 0.1f;
}

// Warped cellular pattern.
// mz*mx*my = diag(49,49,49) = 49·I, so the matrix chain simplifies to 49.
static float3 lt_effect(float3 p) {
    p *= 49.0f * sin(p.zxy);  // p *= (mz*mx*my)*sin(p.zxy), with product = 49·I
    return float3(min(min(lt_func(p * lt_mx), lt_func(p * lt_my)), lt_func(p * lt_mz)) / 0.6f);
}

// Surface displacement amount + colour (w = dist, xyz = black-line colour).
static float4 lt_displacement(float3 p) {
    float3 col = 1.0f - lt_effect(p * 0.8f);
    col = clamp(col, -0.5f, 1.0f);
    float dist = dot(col, float3(0.023f));
    col = step(col, float3(0.82f));  // black line on shape
    return float4(dist, col);
}

// Main SDF: returns (signed distance, rgb colour).
// Tunnel axis is +Z, tube radius ≈ 4, centreline follows lt_path(z).
static float4 lt_map(float3 p) {
    p.xy -= lt_path(p.z);
    float4 disp = lt_displacement(sin(p.zxy * 2.0f) * 0.8f);
    p += sin(p.zxy * 0.5f) * 1.5f;
    float l = length(p.xy) - 4.0f;
    return float4(max(-l + 0.09f, l) - disp.x, disp.yzw);
}

// Finite-difference surface normal.
static float3 lt_nor(float3 pos) {
    const float eps = 0.1f;
    float2 e = float2(eps, 0.0f);
    return normalize(float3(
        lt_map(pos + e.xyy).x - lt_map(pos - e.xyy).x,
        lt_map(pos + e.yxy).x - lt_map(pos - e.yxy).x,
        lt_map(pos + e.yyx).x - lt_map(pos - e.yyx).x));
}

// Lighting: point light at lightPos.  Returns (rgb colour, light distance).
static float4 lt_light(float3 ro, float3 rd, float d, float3 lightPos) {
    float3 p       = ro + rd * d;
    float3 n       = lt_nor(p);
    float3 lightDir = lightPos - p;
    float  lightLen = length(lightDir);
    lightDir /= lightLen;
    float  amb  = 0.6f;
    float  diff = clamp(dot(n, lightDir), 0.0f, 1.0f);
    float3 brdf = amb * float3(0.2f, 0.5f, 0.3f);  // material colour
    brdf += diff * 0.6f;
    brdf = mix(brdf, lt_map(p).yzw, 0.5f);          // blend lighting & lace pattern
    return float4(brdf, lightLen);
}

// ─── Box intersection ─────────────────────────────────────────────────────────
// ro and rd in object/local space; halfExt are the box half-extents.
// Returns (tNear, tFar); if tNear > tFar the ray misses.
static float2 lt_boxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv  = 1.0f / rd;
    float3 t0   = (-halfExt - ro) * inv;
    float3 t1   = ( halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(max(max(tMin.x, tMin.y), tMin.z),
                  min(min(tMax.x, tMax.y), tMax.z));
}

// ─── Fragment shader ──────────────────────────────────────────────────────────
fragment float4 laceTunnelFragment(
    LaceTunnelVertexOut          in       [[stage_in]],
    constant LaceTunnelUniforms &uniforms [[buffer(0)]],
    constant float4x4           *v2wMats  [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);

    // Camera world position from view-to-world transform (column 3).
    float4x4 v2w    = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    // Convert to object (local) space: local = (world − centre) / scale.
    float3 center = uniforms.objectCenter.xyz;
    float  sc     = uniforms.boxScale;
    float3 eye    = (camWorld    - center) / sc;  // camera in object space
    float3 hit    = (in.worldPos - center) / sc;  // cube surface in object space
    float3 rd     = normalize(hit - eye);          // ray direction (unit)

    // Box half-extents in object space are exactly ±1 (unit-cube mesh × 1).
    const float3 halfBox = float3(1.0f);
    bool   insideBox = all(abs(eye) < halfBox - 1e-3f);
    float2 tBox      = lt_boxIntersect(eye, rd, halfBox);

    if (!insideBox && tBox.x > tBox.y) discard_fragment();

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float tEnd   = tBox.y;
    if (tEnd <= tStart) discard_fragment();

    // ── Map to tunnel space ──────────────────────────────────────────────────
    // Scale object space up so the 2 m cube shows a meaningful slice of the
    // tunnel (tube radius 4, path offset ±2).  Advance z by time so the
    // tunnel scrolls toward the viewer (matches original: camera.z = time).
    const float SCALE = 5.0f;
    const float SPEED = 1.0f;  // tunnel units / second (same rate as original)

    // roT: ray origin in tunnel space (at cube entry point or at camera).
    // rdT == rd: direction is unchanged under uniform scale + translation.
    // maxd: maximum march distance in tunnel space.
    float3 roT  = (eye + rd * tStart) * SCALE + float3(0.0f, 0.0f, uniforms.time * SPEED);
    float  maxd = min(40.0f, (tEnd - tStart) * SCALE);

    // ── Ray march ────────────────────────────────────────────────────────────
    // Translated directly from Aiekick's loop.
    // Special break condition for thin surfaces: st=0 on first iter yields
    // NaN in the log term → 0 < NaN = false → safe to skip on iter 0.
    float st = 0.0f;
    float d  = 0.0f;
    float ao = 0.0f;
    for (int i = 0; i < 150; i++) {
        if (st < 0.025f * log(d * d / st / 1e5f) || d > maxd) break;
        float4 m = lt_map(roT + rd * d);
        st = m.x;
        d += st * 0.6f;
        ao += 1.0f;
    }

    // ── Shade ────────────────────────────────────────────────────────────────
    if (d >= maxd) {
        discard_fragment();  // ray exited box without hitting → let scene show through
    }

    // Point light at the ray origin (camera-like light, same as original).
    float4 li  = lt_light(roT, rd, d, roT);
    float3 col = li.xyz / (li.w * 0.2f);                        // cheap distance attenuation
    col = mix(float3(1.0f - ao / 100.0f), col, 0.5f);           // low-cost AO
    col = mix(col, float3(0.0f), 1.0f - exp(-0.003f * d * d));  // depth fog
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}
