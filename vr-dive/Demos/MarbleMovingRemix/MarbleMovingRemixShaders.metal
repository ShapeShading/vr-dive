// MarbleMovingRemixShaders.metal
// "marble moving remix" — cube-container adaptation of ShaderToy wstBRB.
// Source: https://www.shadertoy.com/view/wstBRB
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
// Original source note: fork of "Playing marble" by guil.
//
// Adaptation notes:
// - The original GLSL uses a synthetic screen camera orbiting a reflective
//   marble sphere and samples iChannel0 for environment lookup.
// - This version reconstructs a real per-eye ray from the visible 2 m cube,
//   starts at the cube surface when outside or at the eye when inside, and
//   replaces the texture environment dependency with a procedural sky.

#include <metal_stdlib>
using namespace metal;

struct MarbleMovingRemixUniforms {
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

struct MarbleMovingRemixVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float MMR_ZOOM = 1.0f;
static constant float MMR_SPHERE_RADIUS = 2.0f;
static constant float3 MMR_BOX_HALF = float3(1.0f);

vertex MarbleMovingRemixVertexOut marbleMovingRemixVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant MarbleMovingRemixUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    MarbleMovingRemixVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 mmrCSqr(float2 a) {
    return float2(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y);
}

static float2 mmrRot(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float2 mmrSphereIntersect(float3 ro, float3 rd, float4 sph) {
    float3 oc = ro - sph.xyz;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sph.w * sph.w;
    float h = b * b - c;
    if (h < 0.0f) {
        return float2(-1.0f);
    }
    h = sqrt(h);
    return float2(-b - h, -b + h);
}

static float mmrMap(float3 p, float time) {
    float res = 0.0f;
    float3 c = p;
    float growth = sin(time * 0.35432f) * 0.7f + 1.5f;
    float shift = sin(time * 0.2443f) * 0.3f;

    for (int i = 0; i < 10; ++i) {
        p = growth * abs(p) / max(dot(p, p), 1.0e-4f) - 0.4f + shift;
        p.yz = mmrCSqr(p.yz);
        p = p.zxy;
        res += exp(-19.0f * abs(dot(p, c)));
    }
    return res * 0.5f;
}

static float3 mmrEnvironment(float3 dir, float time) {
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 4.0f);
    float sun = pow(max(dot(dir, normalize(float3(0.28f, 0.42f, -0.86f))), 0.0f), 56.0f);
    float shimmer = 0.5f + 0.5f * sin((dir.x + dir.z) * 12.0f + time * 0.18f);
    float3 sky = mix(float3(0.01f, 0.02f, 0.045f), float3(0.12f, 0.22f, 0.34f), skyMix);
    sky += float3(0.06f, 0.16f, 0.28f) * horizon * shimmer * 0.35f;
    sky += float3(1.0f, 0.95f, 0.88f) * sun;
    return clamp(sky, 0.0f, 2.0f);
}

static float2 mmrBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 mmrFaceUV(float3 p) {
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

static float3 mmrRaymarch(float3 ro, float3 rd, float2 tminmax, float time) {
    float t = tminmax.x;
    const float dt = 0.02f;
    float3 col = float3(0.0f);
    float c = 0.0f;
    for (int i = 0; i < 64; ++i) {
        t += dt * exp(-2.0f * c);
        if (t > tminmax.y) {
            break;
        }

        c = mmrMap(ro + t * rd, time);
        col = 0.99f * col + 0.08f * float3(c * c, c, c * c * c);
    }
    return col;
}

fragment float4 marbleMovingRemixFragment(
    MarbleMovingRemixVertexOut in [[stage_in]],
    constant MarbleMovingRemixUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 rd = normalize(surfacePos - eye);

    bool insideOuter = all(abs(eye) < MMR_BOX_HALF - 1.0e-3f);
    float2 tOuter = mmrBoxIntersect(eye, rd, MMR_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 ro = (eye + rd * (tStart + 0.001f)) * (MMR_ZOOM * 4.0f);

    float orbitX = 0.1f * uniforms.time;
    float orbitY = 0.12f * sin(uniforms.time * 0.13f);
    ro.yz = mmrRot(ro.yz, orbitY);
    ro.xz = mmrRot(ro.xz, orbitX);

    float3 marchDir = normalize(rd);
    marchDir.yz = mmrRot(marchDir.yz, orbitY);
    marchDir.xz = mmrRot(marchDir.xz, orbitX);

    float2 tmm = mmrSphereIntersect(ro, marchDir, float4(0.0f, 0.0f, 0.0f, MMR_SPHERE_RADIUS));
    float3 col;

    if (tmm.x < 0.0f && tmm.y < 0.0f) {
        col = mmrEnvironment(marchDir, uniforms.time);
    } else {
        float tNear = max(tmm.x, 0.0f);
        float tFar = max(tmm.y, tNear);
        col = mmrRaymarch(ro, marchDir, float2(tNear, tFar), uniforms.time);

        float tSurface = (tmm.x > 0.0f) ? tmm.x : tmm.y;
        float3 hitPos = ro + tSurface * marchDir;
        float3 reflectedNormal = hitPos / MMR_SPHERE_RADIUS;
        float3 reflected = reflect(marchDir, reflectedNormal);
        float fre = pow(0.5f + clamp(dot(reflected, marchDir), 0.0f, 1.0f), 3.0f) * 1.3f;
        col += mmrEnvironment(reflected, uniforms.time) * fre;
    }

    col = 0.5f * log(1.0f + col);
    col = clamp(col, 0.0f, 1.0f);
    col.b = col.g * 3.0f;

    float2 faceUV = mmrFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.18f * dot(faceUV, faceUV);
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}