// AnotherMarbleShaders.metal
// "Another Marble" — cube-container adaptation of ShaderToy lsG3D3.
// Source: https://www.shadertoy.com/view/lsG3D3
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Adaptation notes:
// - The original GLSL uses a synthetic screen camera orbiting a glass sphere and
//   samples an environment cubemap via iChannel0.
// - This version reconstructs a real per-eye ray from the visible 2 m cube,
//   begins marching from the cube surface when outside or from the eye when
//   inside, and replaces the cubemap dependency with a procedural environment.
// - The marble volume is evaluated in scene space and is not clipped by the cube.

#include <metal_stdlib>
using namespace metal;

struct AnotherMarbleUniforms {
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

struct AnotherMarbleVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float AM_ZOOM = 1.25f;
static constant float AM_SIZE = 0.19f;
static constant float AM_SPHERE_RADIUS = 2.0f;
static constant float3 AM_BOX_HALF = float3(1.0f);

vertex AnotherMarbleVertexOut anotherMarbleVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant AnotherMarbleUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    AnotherMarbleVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 amCSqr(float2 a) {
    return float2(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y);
}

static float2 amRot(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float3 amACESFilm(float3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

static float2 amSphereIntersect(float3 ro, float3 rd, float4 sph) {
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

static float amMap(float3 p, float time) {
    float res = 0.0f;
    float st = cos(time * 0.1f) * 0.4f;
    float3 c = p;
    for (int i = 0; i < 6; ++i) {
        p = 0.4f * abs(p) / max(dot(p, p), 1.0e-4f) - 0.3f + st;
        p.yz = amCSqr(p.yz);
        res += exp(-20.0f * abs(dot(p, c)));
    }
    return res * 0.325f;
}

static float3 amEnvironment(float3 dir, float time) {
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 4.0f);
    float sun = pow(max(dot(dir, normalize(float3(0.32f, 0.44f, -0.84f))), 0.0f), 64.0f);
    float shimmer = 0.5f + 0.5f * sin((dir.x - dir.z) * 12.0f + time * 0.2f);
    float3 sky = mix(float3(0.015f, 0.02f, 0.04f), float3(0.18f, 0.26f, 0.34f), skyMix);
    sky += horizon * float3(0.20f, 0.18f, 0.16f) * 0.35f * shimmer;
    sky += float3(1.0f, 0.95f, 0.88f) * sun;
    return clamp(sky, 0.0f, 2.0f);
}

static float2 amBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 amFaceUV(float3 p) {
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

static float3 amRaymarch(float3 ro, float3 rd, float2 tminmax, float time) {
    float t = tminmax.x;
    float m = cos(time * 0.1f) - 5.0f;
    float safeTMin = max(tminmax.x, 0.02f);
    float dt = (tminmax.y / safeTMin) * 0.25f;
    float3 col = float3(0.0f);
    float c = 0.0f;

    for (int i = 0; i < 192; ++i) {
        t += dt * exp(m * c);
        if (t > tminmax.y) {
            break;
        }
        float3 pos = (ro + t * rd) * AM_SIZE;
        c = amMap(pos, time);
        col += float3(
            c * (c + 0.5f) * c * c - pos.z,
            c * c * c - pos.y,
            c * c - pos.x) + rd * rd * c * c;
    }
    return col * 0.003f;
}

fragment float4 anotherMarbleFragment(
    AnotherMarbleVertexOut in [[stage_in]],
    constant AnotherMarbleUniforms &uniforms [[buffer(0)]],
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

    bool insideOuter = all(abs(eye) < AM_BOX_HALF - 1.0e-3f);
    float2 tOuter = amBoxIntersect(eye, rd, AM_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 ro = (eye + rd * (tStart + 0.001f)) * (AM_ZOOM * 3.2f);

    float orbit = -0.1f * uniforms.time;
    ro.yz = amRot(ro.yz, 0.18f * sin(uniforms.time * 0.07f));
    ro.xz = amRot(ro.xz, orbit);
    float3 marchDir = normalize(rd);
    marchDir.yz = amRot(marchDir.yz, 0.18f * sin(uniforms.time * 0.07f));
    marchDir.xz = amRot(marchDir.xz, orbit);

    float2 tmm = amSphereIntersect(ro, marchDir, float4(0.0f, 0.0f, 0.0f, AM_SPHERE_RADIUS));
    float3 col;
    if (tmm.x < 0.0f && tmm.y < 0.0f) {
        col = amEnvironment(marchDir, uniforms.time) * 2.0f;
    } else {
        float tNear = max(tmm.x * 0.6f, 0.0f);
        float tFar = max(tmm.y, tNear);
        col = amRaymarch(ro, marchDir, float2(tNear, tFar), uniforms.time);

        float tSurface = (tmm.x > 0.0f) ? tmm.x : tmm.y;
        float3 surfaceNormal = (ro + tSurface * marchDir) / AM_SPHERE_RADIUS;
        float3 reflected = reflect(marchDir, surfaceNormal);
        float fre = pow(0.5f + clamp(dot(reflected, marchDir), 0.0f, 1.0f), 3.0f) * 1.2f;
        col += amEnvironment(reflected, uniforms.time) * fre;
    }

    col = amACESFilm(col);
    col *= col;
    col *= col;

    float2 faceUV = amFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.18f * dot(faceUV, faceUV);
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}