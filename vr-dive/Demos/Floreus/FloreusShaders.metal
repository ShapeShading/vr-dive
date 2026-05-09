// FloreusShaders.metal
// Adapted from ShaderToy "Floreus" by Jaenam.
// Source: https://www.shadertoy.com/view/33fyWB
// License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
//
// Metal adaptation notes:
// - The original shader is a compact forward ray accumulator defined in screen
//   space. This version evaluates the same iterative field along the real per-
//   eye world ray after intersecting a 2 m cube container.
// - Outside the cube, marching starts at the visible cube surface; inside the
//   cube, marching starts at the eye.
// - The fractal accumulation continues beyond the entry plane, so the visual
//   field is not clipped to the cube volume.
// - GLSL macro rotations and implicit initialization tricks are expanded into
//   explicit Metal helpers and explicit initial values.

#include <metal_stdlib>
using namespace metal;

struct FloreusUniforms {
    float  time;
    uint   viewCount;
    float  boxScale;
    float  padding;
    float4 objectCenter;
};

struct MeshVertex {
    float3 position;
    float3 normal;
};

struct FloreusVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct FlField {
    float sdf;
    float density;
    float3 tint;
};

static constant float3 FL_BOX_HALF = float3(1.0f);
static constant float FL_TRACE_EPSILON = 0.0015f;
static constant int FL_STEPS = 72;
static constant int FL_DETAIL_STEPS = 3;
static constant float FL_MAX_DIST = 4.8f;
static constant float FL_MIN_STEP = 0.012f;
static constant float FL_MAX_STEP = 0.085f;
static constant float3 FL_SATELLITE_CENTERS[4] = {
    float3(0.82f, 0.0f, 0.12f),
    float3(-0.86f, 0.05f, -0.06f),
    float3(0.1f, 0.84f, 0.18f),
    float3(-0.14f, -0.88f, 0.16f)
};
static constant float3 FL_SATELLITE_AXES[4] = {
    float3(1.0f, 0.0f, 0.14f),
    float3(-1.0f, 0.06f, -0.1f),
    float3(0.12f, 1.0f, 0.18f),
    float3(-0.15f, -1.0f, 0.16f)
};
static constant float FL_SATELLITE_SIZES[4] = {0.24f, 0.25f, 0.22f, 0.22f};
static constant float FL_SATELLITE_HUES[4] = {0.45f, 0.95f, 1.45f, 1.95f};

vertex FloreusVertexOut floreusVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant FloreusUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    FloreusVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 flRotate2D(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float2 flFaceUV(float3 p) {
    float3 ap = abs(p);
    float2 uv;
    if (ap.x >= ap.y && ap.x >= ap.z) {
        uv = p.zy;
    } else if (ap.y >= ap.z) {
        uv = p.xz;
    } else {
        uv = p.xy;
    }
    return clamp(uv * 0.5f + 0.5f, 0.0f, 1.0f);
}

static float2 flBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float flCapsule(float3 p, float3 a, float3 b, float r) {
    float3 pa = p - a;
    float3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h) - r;
}

static float3 flToLocal(float3 p, float3 axis) {
    float3 forward = normalize(axis);
    float3 upRef = abs(forward.y) > 0.95f ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 1.0f, 0.0f);
    float3 right = normalize(cross(upRef, forward));
    float3 up = cross(forward, right);
    return float3(dot(p, right), dot(p, up), dot(p, forward));
}

static float flPetalShell(float3 p, float petals, float radius, float thickness, float curl) {
    float angle = atan2(p.y, p.x);
    float radial = length(p.xy);
    float lobe = radius * (0.56f + 0.44f * cos(petals * angle));
    float bow = p.z + curl * smoothstep(0.0f, radius + 0.3f, radial) *
        (0.4f + 0.6f * abs(cos(angle * petals * 0.5f)));
    return length(float2(radial - lobe, bow * 1.75f)) - thickness;
}

static float flDetailField(float3 p, float time) {
    p.xz = flRotate2D(p.xz, 0.18f * time);
    float response = 0.0f;
    float weight = 1.0f;
    for (int inner = 0; inner < FL_DETAIL_STEPS; ++inner) {
        float2 folded = min(abs(p.xz), abs(p.xy));
        float l = length(float2(0.65f) - folded) / max(dot(p, p + p), 0.3f);
        p = sin(p * 1.25f);
        p *= l;
        response += weight * exp(-2.2f * length(p));
        weight *= 0.55f;
    }
    return response;
}

static void flAccumulate(thread FlField &field, float sdf, float density, float3 tint) {
    if (sdf < field.sdf) {
        field.sdf = sdf;
        field.tint = tint;
    }
    field.density += density;
}

static void flAddBloom(
    thread FlField &field,
    float3 p,
    float3 center,
    float3 axis,
    float size,
    float time,
    float hueShift
) {
    float3 local = flToLocal(p - center, axis) / size;

    float3 petalA = local;
    petalA.xy = flRotate2D(petalA.xy, hueShift + time * 0.08f);
    float shellA = flPetalShell(petalA, 6.0f, 0.62f, 0.06f, 0.16f);

    float3 petalB = local;
    petalB.xy = flRotate2D(petalB.xy, 1.0472f + hueShift * 0.6f);
    petalB.yz = flRotate2D(petalB.yz, 0.85f);
    float shellB = flPetalShell(petalB, 4.0f, 0.34f, 0.038f, -0.1f);

    float core = length(local) - 0.14f;
    float sdf = min(min(shellA, shellB), core) * size;

    float density =
        0.82f * exp(-12.0f * abs(shellA)) +
        0.46f * exp(-16.0f * abs(shellB)) +
        0.34f * exp(-18.0f * abs(core));

    float shimmer = 0.5f + 0.5f * sin(hueShift * 3.0f + time * 0.45f);
    float3 tint = mix(float3(1.1f, 0.72f, 0.25f), float3(1.85f, 1.55f, 1.08f), shimmer);
    flAccumulate(field, sdf, density, tint);
}

static void flAddBranch(
    thread FlField &field,
    float3 p,
    float3 a,
    float3 b,
    float time,
    float thickness,
    float hueShift
) {
    float sdf = flCapsule(p, a, b, thickness * (0.9f + 0.1f * sin(time * 0.6f + hueShift)));
    float wave = 0.5f + 0.5f * sin(dot(normalize(b - a), p) * 13.0f + time * 1.4f + hueShift);
    float density = exp(-16.0f * abs(sdf)) * (0.22f + 0.28f * wave);
    float3 tint = mix(float3(0.82f, 0.55f, 0.18f), float3(1.35f, 1.08f, 0.62f), wave);
    flAccumulate(field, sdf, density, tint);
}

static FlField flMap(float3 p, float time) {
    FlField field;
    field.sdf = 1.0e9f;
    field.density = 0.0f;
    field.tint = float3(1.0f, 0.85f, 0.5f);

    float3 q = p;
    q.xz = flRotate2D(q.xz, time * 0.08f);
    q.yz = flRotate2D(q.yz, -time * 0.05f);

    flAddBloom(field, q, float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 1.0f), 0.5f, time, 0.0f);

    for (int i = 0; i < 4; ++i) {
        flAddBloom(field, q, FL_SATELLITE_CENTERS[i], FL_SATELLITE_AXES[i], FL_SATELLITE_SIZES[i], time, FL_SATELLITE_HUES[i]);
        flAddBranch(field, q, float3(0.0f), FL_SATELLITE_CENTERS[i] * 0.82f, time, 0.028f, FL_SATELLITE_HUES[i]);
    }

    field.density += flDetailField(q * 1.45f, time) * (0.08f + 0.14f * exp(-3.0f * abs(field.sdf)));
    field.density = min(field.density, 1.8f);
    return field;
}

fragment float4 floreusFragment(
    FloreusVertexOut in [[stage_in]],
    constant FloreusUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float cubeScale = max(uniforms.boxScale, 1.0e-4f);
    float3 eye = (camWorld - center) / cubeScale;
    float3 hit = (in.worldPos - center) / cubeScale;
    float3 rd = normalize(hit - eye);

    bool insideBox = all(abs(eye) < FL_BOX_HALF - 1.0e-3f);
    float2 tBox = flBoxIntersect(eye, rd, FL_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 entry = eye + rd * (tStart + FL_TRACE_EPSILON);
    float3 ro = entry * 1.05f;
    float3 marchDir = normalize(rd);

    float3 accum = float3(0.0f);
    float transmittance = 1.0f;
    float travel = 0.0f;
    float time = uniforms.time;

    for (int step = 0; step < FL_STEPS; ++step) {
        float3 pos = ro + marchDir * travel;
        FlField field = flMap(pos, time);

        float density = field.density;
        accum += transmittance * field.tint * density * 0.05f;

        transmittance *= exp(-density * 0.09f);
        if (transmittance < 0.02f || travel > FL_MAX_DIST) {
            break;
        }

        float stepMix = clamp(abs(field.sdf) * 1.6f, 0.0f, 1.0f);
        travel += mix(FL_MIN_STEP, FL_MAX_STEP, stepMix);
    }

    float3 color = 1.0f - exp(-accum * 1.15f);
    color = pow(color, float3(0.94f));

    float2 q = flFaceUV(hit);
    float vignette = 1.0f - 0.18f * dot(q * 2.0f - 1.0f, q * 2.0f - 1.0f);
    color *= vignette;
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}