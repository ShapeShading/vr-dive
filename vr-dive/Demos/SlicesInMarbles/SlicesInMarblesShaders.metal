// SlicesInMarblesShaders.metal
// "slices in marbles" — cube-container adaptation of ShaderToy tdXGWM.
// Source: https://www.shadertoy.com/view/tdXGWM
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Adaptation notes:
// - The original shader renders a glass marble with milky refracted slices
//   using a synthetic screen camera and iChannel0 environment texture.
// - This version reconstructs a real per-eye ray from the 2 m cube container,
//   supports starting from the cube surface or from inside the cube, and
//   replaces the external texture with a procedural environment.

#include <metal_stdlib>
using namespace metal;

struct SlicesInMarblesUniforms {
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

struct SlicesInMarblesVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant int SIM_DETAIL = 5;
static constant int SIM_INNER_DEPTH = 64;
static constant float SIM_COLOR_CONTRAST = 45.0f;
static constant float SIM_MILKY_LIGHT = 80.0f;
static constant float SIM_OPACITY_OF_COLOR = 1.0f;
static constant float SIM_ZOOM = 1.0f;
static constant float SIM_SPHERE_RADIUS = 2.0f;
static constant float3 SIM_BOX_HALF = float3(1.0f);

vertex SlicesInMarblesVertexOut slicesInMarblesVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant SlicesInMarblesUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    SlicesInMarblesVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 simCSqr(float2 a) {
    return float2(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y);
}

static float2 simRot(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float2 simSphereIntersect(float3 ro, float3 rd, float4 sph) {
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

static float3 simMap(float3 p) {
    float res = 0.0f;
    float3 c = p;
    for (int i = 0; i < SIM_DETAIL; ++i) {
        p = 0.7f * abs(p) / max(dot(p, p), 1.0e-4f) - 0.7f;
        p.yz = simCSqr(p.yz);
        p = p.zxy;
        res += exp(-19.0f * abs(dot(p, c))) + 0.02f;
    }

    float3 normP = normalize(select(float3(0.0f, 0.0f, 1.0f), p, dot(p, p) > 1.0e-6f));
    return res * SIM_COLOR_CONTRAST * 0.013f * (normP + (1.0f - SIM_OPACITY_OF_COLOR) * float3(1.0f));
}

static float3 simEnvironment(float3 dir, float time) {
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 4.0f);
    float sun = pow(max(dot(dir, normalize(float3(0.28f, 0.45f, -0.84f))), 0.0f), 56.0f);
    float shimmer = 0.5f + 0.5f * sin((dir.x - dir.z) * 12.0f + time * 0.2f);
    float3 sky = mix(float3(0.012f, 0.02f, 0.04f), float3(0.12f, 0.18f, 0.30f), skyMix);
    sky += float3(0.12f, 0.16f, 0.22f) * horizon * shimmer * 0.30f;
    sky += float3(1.0f, 0.95f, 0.9f) * sun;
    return clamp(sky, 0.0f, 2.0f);
}

static float2 simBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 simFaceUV(float3 p) {
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

static float3 simRaymarch(float3 ro, float3 rd, float2 tminmax) {
    float t = tminmax.x;
    const float dt = 0.02f;
    float3 col = float3(0.0f);
    float3 c = float3(0.0f);

    for (int i = 0; i < SIM_INNER_DEPTH; ++i) {
        t += dt * exp(-2.0f * length(c));
        if (t > tminmax.y) {
            break;
        }

        float3 pos = refract(ro, (ro + t * rd) / 0.7f, -0.012f);
        c = simMap(pos);
        float gr = SIM_MILKY_LIGHT * 0.013824f / float(SIM_INNER_DEPTH);
        col = 0.995f * col + 0.09f * c + float3(gr);
    }

    return col;
}

fragment float4 slicesInMarblesFragment(
    SlicesInMarblesVertexOut in [[stage_in]],
    constant SlicesInMarblesUniforms &uniforms [[buffer(0)]],
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

    bool insideOuter = all(abs(eye) < SIM_BOX_HALF - 1.0e-3f);
    float2 tOuter = simBoxIntersect(eye, rd, SIM_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 ro = (eye + rd * (tStart + 0.001f)) * (SIM_ZOOM * 4.0f);

    float orbitX = 0.1f * uniforms.time;
    float orbitY = 0.12f * sin(uniforms.time * 0.11f);
    ro.yz = simRot(ro.yz, orbitY);
    ro.xz = simRot(ro.xz, orbitX);

    float3 marchDir = normalize(rd);
    marchDir.yz = simRot(marchDir.yz, orbitY);
    marchDir.xz = simRot(marchDir.xz, orbitX);

    float2 tmm = simSphereIntersect(ro, marchDir, float4(0.0f, 0.0f, 0.0f, SIM_SPHERE_RADIUS));
    float3 col;
    if (tmm.x < 0.0f && tmm.y < 0.0f) {
        col = simEnvironment(marchDir, uniforms.time);
    } else {
        float tNear = max(tmm.x, 0.0f);
        float tFar = max(tmm.y, tNear);
        col = simRaymarch(ro, marchDir, float2(tNear, tFar));

        float tSurface = (tmm.x > 0.0f) ? tmm.x : tmm.y;
        float3 reflectedNormal = (ro + tSurface * marchDir) / SIM_SPHERE_RADIUS;
        float3 reflected = reflect(marchDir, reflectedNormal);
        float fre = pow(0.5f + clamp(dot(reflected, marchDir), 0.0f, 1.0f), 3.0f) * 1.3f;
        col += simEnvironment(reflected, uniforms.time) * fre;
    }

    col = 0.5f * log(1.0f + col);
    col = clamp(col, 0.0f, 1.0f);

    float2 faceUV = simFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.18f * dot(faceUV, faceUV);
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}