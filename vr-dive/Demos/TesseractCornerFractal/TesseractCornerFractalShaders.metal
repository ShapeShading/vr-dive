// TesseractCornerFractalShaders.metal
// Adapted from ShaderToy "Tesseract Corner Fractal".
// Source: https://www.shadertoy.com/view/7fs3Wf
//
// Metal adaptation notes:
// - The original shader used a fixed screen-space camera in a 4D field.
//   This version reconstructs the real per-eye world ray, intersects it with a
//   2 m cube container, and starts marching at the visible cube surface or at
//   the eye when the viewer is inside the cube.
// - The simulated 4D fractal is traced beyond the container boundary, so the
//   content is not clipped to the cube volume.
// - GLSL's row-vector style `p *= m` is implemented explicitly for Metal.

#include <metal_stdlib>
using namespace metal;

struct TesseractCornerFractalUniforms {
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

struct TesseractCornerFractalVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float3 TCF_BOX_HALF = float3(1.0f);
static constant float TCF_TRACE_EPSILON = 0.0015f;
static constant float TCF_HIT_EPSILON = 0.00012f;
static constant float TCF_SCENE_SCALE = 2.2f;
static constant float TCF_MAX_DISTANCE = 18.0f;
static constant int TCF_TRACE_STEPS = 96;

vertex TesseractCornerFractalVertexOut tesseractCornerFractalVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant TesseractCornerFractalUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    TesseractCornerFractalVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 tcfBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float4 tcfMulRow(float4 v, float4x4 m) {
    return float4(dot(v, m[0]), dot(v, m[1]), dot(v, m[2]), dot(v, m[3]));
}

static float4x4 tcfRotationPair(float t1, float t2) {
    return float4x4(
        float4(cos(t1), sin(t1), 0.0f, 0.0f),
        float4(-sin(t1), cos(t1), 0.0f, 0.0f),
        float4(0.0f, 0.0f, cos(t2), sin(t2)),
        float4(0.0f, 0.0f, -sin(t2), cos(t2)));
}

static float4x4 tcfPermutation() {
    return float4x4(
        float4(0.0f, 1.0f, 0.0f, 0.0f),
        float4(0.0f, 0.0f, 1.0f, 0.0f),
        float4(0.0f, 0.0f, 0.0f, 1.0f),
        float4(1.0f, 0.0f, 0.0f, 0.0f));
}

static float4x4 tcfBuildRotation(float time) {
    float4x4 rot = float4x4(1.0f);
    float t1 = time * 0.3f;
    float t2 = time * 0.3f;

    for (int j = 0; j < 8; ++j) {
        rot = rot * tcfRotationPair(t1, t2);
        rot = rot * tcfPermutation();
        t1 /= -1.237415f;
        t2 /= 1.348912f;
    }
    return rot;
}

static float tcfMax4(float4 v) {
    return max(max(v.x, v.y), max(v.z, v.w));
}

static float tcfSdf(float4 p, float4x4 m) {
    float q = 1.0f;
    float d = 1.0e9f;
    for (int n = 0; n < 4; ++n) {
        p = tcfMulRow(p, m);
        p = abs(p);

        float cornerA = max(p.x, max(p.y, p.z));
        float cornerB = max(p.y, max(p.z, p.w));
        float cornerC = max(p.z, max(p.w, p.x));
        float cornerD = max(p.w, max(p.x, p.y));

        float outer = tcfMax4(p) - 1.0f;
        float inner = 0.8f - min(min(cornerA, cornerB), min(cornerC, cornerD));
        d = min(d, max(outer, inner) / q);

        p = (p - 0.9f) * 2.1f;
        q *= 2.1f;
    }
    return d;
}

static float3 tcfCalcNormal(float4 p, float4x4 rot) {
    float e = 0.001f;
    float dx = tcfSdf(p + float4(e, 0.0f, 0.0f, 0.0f), rot) - tcfSdf(p - float4(e, 0.0f, 0.0f, 0.0f), rot);
    float dy = tcfSdf(p + float4(0.0f, e, 0.0f, 0.0f), rot) - tcfSdf(p - float4(0.0f, e, 0.0f, 0.0f), rot);
    float dz = tcfSdf(p + float4(0.0f, 0.0f, e, 0.0f), rot) - tcfSdf(p - float4(0.0f, 0.0f, e, 0.0f), rot);
    return normalize(float3(dx, dy, dz));
}

fragment float4 tesseractCornerFractalFragment(
    TesseractCornerFractalVertexOut in [[stage_in]],
    constant TesseractCornerFractalUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < TCF_BOX_HALF - 1.0e-3f);
    float2 tBox = tcfBoxIntersect(eye, rd, TCF_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 marchOrigin = eye + rd * (tStart + TCF_TRACE_EPSILON);

    float4x4 rot = tcfBuildRotation(uniforms.time);
    float wOrigin = 0.35f * sin(uniforms.time * 0.37f);
    float wDirection = 0.22f * sin(uniforms.time * 0.19f + dot(marchOrigin, float3(1.1f, -0.8f, 0.6f)));

    float4 rayPos = float4(marchOrigin * TCF_SCENE_SCALE, wOrigin);
    float4 rayDir = normalize(float4(rd, wDirection));

    float brightness = 1.0f;
    float traveled = 0.0f;
    float stepDistance = 0.0f;
    bool hitSurface = false;
    int stepsTaken = 0;

    for (int i = 0; i < TCF_TRACE_STEPS; ++i) {
        stepsTaken = i;
        stepDistance = tcfSdf(rayPos, rot);
        if (stepDistance < TCF_HIT_EPSILON) {
            hitSurface = true;
            break;
        }

        float safeStep = max(stepDistance, TCF_HIT_EPSILON * 1.25f);
        traveled += safeStep;
        rayPos += rayDir * safeStep;
        brightness /= 1.07f;

        if (traveled > TCF_MAX_DISTANCE) {
            break;
        }
    }

    float3 baseFog = mix(
        float3(0.02f, 0.03f, 0.06f),
        float3(0.16f, 0.22f, 0.34f),
        clamp(0.5f + 0.5f * rd.y, 0.0f, 1.0f));
    float3 color = baseFog * (0.25f + 0.75f * brightness);

    if (hitSurface) {
        float3 normal = tcfCalcNormal(rayPos, rot);
        float3 lightDir = normalize(float3(0.45f, 0.72f, -0.53f));
        float diffuse = clamp(dot(normal, lightDir), 0.0f, 1.0f);
        float backLight = clamp(0.2f + 0.8f * dot(normal, normalize(float3(-0.4f, 0.3f, 0.85f))), 0.0f, 1.0f);
        float rim = pow(1.0f - clamp(dot(normal, -rd), 0.0f, 1.0f), 2.5f);

        float3 stripe = 0.5f + 0.5f * cos(2.8f * rayPos.w + float3(0.0f, 1.8f, 3.6f));
        float3 albedo = mix(float3(0.09f, 0.12f, 0.22f), float3(0.78f, 0.86f, 1.0f), stripe);
        float attenuation = exp(-0.065f * traveled * traveled);

        color = albedo * (0.18f + 0.95f * diffuse + 0.35f * backLight);
        color += float3(0.45f, 0.55f, 0.9f) * rim * 0.55f;
        color *= attenuation * (0.5f + 0.5f * brightness);
    }

    color = sqrt(clamp(color, 0.0f, 1.0f));
    return float4(color, 1.0f);
}