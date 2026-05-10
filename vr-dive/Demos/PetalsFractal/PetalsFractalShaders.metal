// PetalsFractalShaders.metal
// "Petals Fractal" — cube-container adaptation of ShaderToy lt2GWw.
// Source: https://www.shadertoy.com/view/lt2GWw
// Source shader uses a rotating remote ray origin and a reciprocal-fold fractal field.
// This adaptation reconstructs a real per-eye world ray entering a 2 m cube,
// then maps that ray into the original volumetric scene so the pattern remains
// visible from all viewing directions and also when the viewer is inside.

#include <metal_stdlib>
using namespace metal;

struct PetalsFractalUniforms {
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

struct PetalsFractalVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant int PF_MAX_RAY_STEPS = 64;
static constant float PF_SCALE = 0.3f;
static constant float PF_SIZE = 0.45f;
static constant float PF_INTENSITY = 1.5f;
static constant float3 PF_BOX_HALF = float3(1.0f);
static constant float PF_SCENE_SCALE = 48.0f;

vertex PetalsFractalVertexOut petalsFractalVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant PetalsFractalUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    PetalsFractalVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float3 pfRotationCoord(float3 n, float t) {
    float s = sin(t);
    float c = cos(t);
    float3x3 rotate = float3x3(
        float3(c, 0.0f, s),
        float3(0.0f, 1.0f, 0.0f),
        float3(-s, 0.0f, c));
    return rotate * n;
}

static float pfPattern(float3 p) {
    p *= PF_SCALE;
    for (int i = 0; i < 10; ++i) {
        p = abs(p) / max(dot(p, p), 1.0e-4f) - float3(PF_SIZE);
    }
    return dot(p, p) * PF_INTENSITY;
}

static float pfRender(float3 posOnRay, float3 rayDir) {
    float t = 0.0f;
    float maxDist = 30.0f;
    float d = 0.1f;

    for (int i = 0; i < PF_MAX_RAY_STEPS; ++i) {
        if (abs(d) < 0.0001f || t > maxDist) {
            break;
        }
        t += d;
        posOnRay += rayDir / (d + 0.35f);
        d = pfPattern(posOnRay);
    }

    return d;
}

static float3 pfEnvironment(float3 dir, float time) {
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 4.0f);
    float sun = pow(max(dot(dir, normalize(float3(0.35f, 0.4f, -0.85f))), 0.0f), 48.0f);
    float shimmer = 0.5f + 0.5f * sin((dir.x + dir.z) * 10.0f + time * 0.25f);
    float3 sky = mix(float3(0.01f, 0.012f, 0.02f), float3(0.08f, 0.10f, 0.16f), skyMix);
    sky += float3(0.12f, 0.10f, 0.08f) * horizon * 0.25f * shimmer;
    sky += float3(1.0f, 0.96f, 0.9f) * sun;
    return clamp(sky, 0.0f, 1.6f);
}

static float2 pfBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 pfFaceUV(float3 p) {
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

fragment float4 petalsFractalFragment(
    PetalsFractalVertexOut in [[stage_in]],
    constant PetalsFractalUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 viewDir = normalize(surfacePos - eye);

    bool insideOuter = all(abs(eye) < PF_BOX_HALF - 1.0e-3f);
    float2 tOuter = pfBoxIntersect(eye, viewDir, PF_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 localOrigin = eye + viewDir * (tStart + 0.001f);

    float time = uniforms.time;
    float rotTime = time * 0.5f;

    // Keep the fractal anchored at the cube center so it stays spatially fixed
    // relative to the container and is visible from all directions.
    float3 sceneDir = normalize(pfRotationCoord(viewDir, rotTime));
    float3 sceneOrigin = pfRotationCoord(localOrigin * PF_SCENE_SCALE, rotTime);

    float t = pfRender(sceneOrigin, sceneDir);
    float3 col = float3(0.5f * t * t * t, 0.6f * t * t, 0.7f * t);
    col = min(col, 1.0f) - 0.28f * log(col + 1.0f);
    col = sqrt(max(col, 0.0f));

    float trail = pow(clamp(1.0f - abs(dot(viewDir, float3(0.0f, 0.0f, 1.0f))), 0.0f, 1.0f), 2.0f);
    float3 bg = pfEnvironment(viewDir, time);
    bg += float3(0.12f, 0.08f, 0.18f) * trail * 0.08f;

    float2 faceUV = pfFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.18f * dot(faceUV, faceUV);
    col += bg * 0.18f;
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}