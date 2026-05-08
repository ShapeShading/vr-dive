// NovaMarbleShaders.metal
// "Nova Marble" — cube-container adaptation of ShaderToy "MtdGD8"
// Source: https://www.shadertoy.com/view/MtdGD8
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Source notes:
// - The original shader is a variant of Playing marble with a time-warped
//   volumetric fractal and a bluer accumulation ramp.
// - This version keeps the glass sphere, internal volume march and reflective
//   shell behavior, but reconstructs the ray from the real per-eye camera and
//   enters through a visible 2 m cube container.
// - The marble itself is evaluated in scene space and is not clipped by the
//   cube bounds.

#include <metal_stdlib>
using namespace metal;

struct NovaMarbleUniforms {
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

struct NovaMarbleVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float NM_SPHERE_RADIUS = 2.0f;
static constant float3 NM_BOX_HALF = float3(1.0f);

vertex NovaMarbleVertexOut novaMarbleVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant NovaMarbleUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    NovaMarbleVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 csqr(float2 a) {
    return float2(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y);
}

static float2 nmRotate(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float3 nmRotateScene(float3 p, float time) {
    p.yz = nmRotate(p.yz, time * 0.27f);
    p.xz = nmRotate(p.xz, time * 0.25f + 0.35f);
    return p;
}

static float2 sphereIntersect(float3 ro, float3 rd, float4 sph) {
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

static float nmMap(float3 p, float time) {
    float res = 0.0f;
    float3 c = p;
    float wobbleA = cos(time * 0.15f + 1.6f) * 0.15f;
    float wobbleB = cos(time * 0.15f) * 0.15f;
    for (int i = 0; i < 10; ++i) {
        p = 0.7f * abs(p + wobbleA) / max(dot(p, p), 1.0e-4f) - 0.7f + wobbleB;
        p.yz = csqr(p.yz);
        p = p.zxy;
        res += exp(-19.0f * abs(dot(p, c)));
    }
    return res * 0.5f;
}

static float3 nmEnvironment(float3 dir, float time) {
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 5.0f);
    float sun = pow(max(dot(dir, normalize(float3(0.33f, 0.46f, -0.82f))), 0.0f), 64.0f);
    float shimmer = 0.5f + 0.5f * sin((dir.x + dir.z) * 13.0f + time * 0.35f);
    float3 sky = mix(float3(0.015f, 0.025f, 0.06f), float3(0.12f, 0.18f, 0.34f), skyMix);
    sky += float3(0.08f, 0.16f, 0.34f) * horizon * shimmer * 0.45f;
    sky += float3(1.0f, 0.9f, 0.82f) * sun;
    return clamp(sky, 0.0f, 2.5f);
}

static float2 nmBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 nmFaceUV(float3 p) {
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

static float3 raymarch(float3 ro, float3 rd, float2 tminmax, float time) {
    float t = tminmax.x;
    float dt = 0.1f - 0.075f * cos(time * 0.025f);
    float3 col = float3(0.0f);
    float c = 0.0f;
    for (int i = 0; i < 64; ++i) {
        t += dt * exp(-2.0f * c);
        if (t > tminmax.y) {
            break;
        }

        c = nmMap(ro + t * rd, time);
        col = 0.99f * col + 0.08f * float3(c * c * c, c * c, c);
    }
    return col;
}

fragment float4 novaMarbleFragment(
    NovaMarbleVertexOut in [[stage_in]],
    constant NovaMarbleUniforms &uniforms [[buffer(0)]],
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

    bool insideOuter = all(abs(eye) < NM_BOX_HALF - 1.0e-3f);
    float2 tOuter = nmBoxIntersect(eye, rd, NM_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    const float sceneScale = 2.2f;
    float3 ro = (eye + rd * (tStart + 0.001f)) * sceneScale;
    float3 marchDir = normalize(rd) * 0.975f;
    ro = nmRotateScene(ro, uniforms.time);
    marchDir = normalize(nmRotateScene(marchDir, uniforms.time));

    float2 tmm = sphereIntersect(ro, marchDir, float4(0.0f, 0.0f, 0.0f, NM_SPHERE_RADIUS));
    float3 col = float3(0.0f);

    if (tmm.x < 0.0f && tmm.y < 0.0f) {
        col = nmEnvironment(marchDir, uniforms.time);
    } else {
        float tNear = max(tmm.x, 0.0f);
        float tFar = max(tmm.y, tNear);
        col = raymarch(ro, marchDir, float2(tNear, tFar), uniforms.time);

        float tSurface = (tmm.x > 0.0f) ? tmm.x : tmm.y;
        float3 hitPos = ro + tSurface * marchDir;
        float3 nor = hitPos / NM_SPHERE_RADIUS;
        float3 reflected = reflect(marchDir, nor);
        float fre = pow(0.5f + clamp(dot(reflected, marchDir), 0.0f, 1.0f), 3.0f) * 1.3f;
        col += nmEnvironment(reflected, uniforms.time) * fre;
    }

    float2 faceUV = nmFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.2f * dot(faceUV, faceUV);
    col = 0.5f * log(1.0f + col);
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}