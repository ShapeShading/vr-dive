#include <metal_stdlib>
using namespace metal;

struct SimoneOrbit3DUniforms {
    float  time;
    uint   viewCount;
    float  cubeScale;
    float  padding;
    float4 simoneParameters;
    float4 objectCenter;
};

// ── Per-point input from CPU orbit buffer ───────────────────────────────────
// Swift OrbitPointVertex must match: { Float, Float, Float, Float } = 16 bytes
struct OrbitPoint {
    packed_float3 position;   // orbit-local space, pre-scaled
    float         brightness; // 0..1
};

struct OrbitPointVaryings {
    float4 clipPos    [[position]];
    float  brightness [[flat]];
    float  pointSize  [[point_size]];
    uint   viewIndex  [[flat]];
};

vertex OrbitPointVaryings simoneOrbitPointVertex(
    ushort amplificationID [[amplification_id]],
    const device OrbitPoint *points [[buffer(0)]],
    constant SimoneOrbit3DUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    OrbitPoint p = points[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = float3(p.position) * uniforms.cubeScale + uniforms.objectCenter.xyz;

    OrbitPointVaryings out;
    out.clipPos    = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.brightness = p.brightness;
    out.pointSize  = 10.0f;
    out.viewIndex  = viewIndex;
    return out;
}

// ── Fragment: soft additive disc ─────────────────────────────────────────────
fragment float4 simoneOrbitPointFragment(
    OrbitPointVaryings in [[stage_in]],
    constant SimoneOrbit3DUniforms &uniforms [[buffer(0)]],
    float2 pointCoord [[point_coord]])
{
    float2 uv = pointCoord * 2.0f - 1.0f;
    float r = length(uv);
    if (r > 1.0f) discard_fragment();

    float disc = 1.0f - r;
    float falloff = disc * disc * disc;
    float intensity = in.brightness * falloff * 0.25f;

    // cool blue → warm white-gold as brightness increases
    float3 cool = float3(0.30f, 0.62f, 1.00f);
    float3 warm = float3(1.00f, 0.93f, 0.68f);
    float3 color = mix(cool, warm, in.brightness * in.brightness);

    return float4(color * intensity, intensity);
}
