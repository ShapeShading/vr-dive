// PlayingMarbleShaders.metal
// "Playing marble" — cube-container adaptation of ShaderToy "MtX3Ws"
// Source: https://www.shadertoy.com/view/MtX3Ws
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Source notes:
// - The original shader ray-marches a glowing fractal volume inside a glassy
//   sphere and reflects an environment map on the outer shell.
// - This version keeps the sphere intersection, internal volumetric march and
//   reflective shell behavior, but replaces the synthetic screen camera with a
//   real per-eye world ray entering a 2 m cube container.
// - The marble itself lives in scene space and is not clipped by the cube.
//   When the viewer is outside the cube, marching begins at the visible cube
//   face. When the viewer is inside, marching begins at the eye.

#include <metal_stdlib>
using namespace metal;

struct PlayingMarbleUniforms {
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

struct PlayingMarbleVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float PM_SPHERE_RADIUS = 2.0f;
static constant float3 PM_BOX_HALF = float3(1.0f);

vertex PlayingMarbleVertexOut playingMarbleVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant PlayingMarbleUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    PlayingMarbleVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 csqr(float2 a) {
    return float2(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y);
}

static float2 pmRotate(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float3 pmRotateScene(float3 p, float time) {
    p.yz = pmRotate(p.yz, time * 0.23f);
    p.xz = pmRotate(p.xz, time * 0.19f + 0.35f);
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

static float pmMap(float3 p) {
    float res = 0.0f;
    float3 c = p;
    for (int i = 0; i < 10; ++i) {
        p = 0.7f * abs(p) / max(dot(p, p), 1.0e-4f) - 0.7f;
        p.yz = csqr(p.yz);
        p = p.zxy;
        res += exp(-19.0f * abs(dot(p, c)));
    }
    return res * 0.5f;
}

static float3 pmEnvironment(float3 dir, float time) {
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 5.0f);
    float sun = pow(max(dot(dir, normalize(float3(0.35f, 0.45f, -0.82f))), 0.0f), 64.0f);
    float shimmer = 0.5f + 0.5f * sin((dir.x + dir.z) * 14.0f + time * 0.3f);
    float3 sky = mix(float3(0.02f, 0.03f, 0.05f), float3(0.15f, 0.21f, 0.3f), skyMix);
    sky += float3(0.1f, 0.18f, 0.28f) * horizon * shimmer * 0.35f;
    sky += float3(1.0f, 0.92f, 0.8f) * sun;
    return clamp(sky, 0.0f, 2.5f);
}

static float2 pmBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 pmFaceUV(float3 p) {
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

static float3 raymarch(float3 ro, float3 rd, float2 tminmax) {
    float t = tminmax.x;
    const float dt = 0.02f;
    float3 col = float3(0.0f);
    float c = 0.0f;
    for (int i = 0; i < 64; ++i) {
        t += dt * exp(-2.0f * c);
        if (t > tminmax.y) {
            break;
        }

        c = pmMap(ro + t * rd);
        col = 0.99f * col + 0.08f * float3(c * c, c, c * c * c);
    }
    return col;
}

fragment float4 playingMarbleFragment(
    PlayingMarbleVertexOut in [[stage_in]],
    constant PlayingMarbleUniforms &uniforms [[buffer(0)]],
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

    bool insideOuter = all(abs(eye) < PM_BOX_HALF - 1.0e-3f);
    float2 tOuter = pmBoxIntersect(eye, rd, PM_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    const float sceneScale = 2.2f;
    float3 ro = (eye + rd * (tStart + 0.001f)) * sceneScale;
    float3 marchDir = normalize(rd);
    ro = pmRotateScene(ro, uniforms.time);
    marchDir = normalize(pmRotateScene(marchDir, uniforms.time));

    float2 tmm = sphereIntersect(ro, marchDir, float4(0.0f, 0.0f, 0.0f, PM_SPHERE_RADIUS));
    float3 col = float3(0.0f);

    if (tmm.x < 0.0f && tmm.y < 0.0f) {
        col = pmEnvironment(marchDir, uniforms.time);
    } else {
        float tNear = max(tmm.x, 0.0f);
        float tFar = max(tmm.y, tNear);
        col = raymarch(ro, marchDir, float2(tNear, tFar));

        float tSurface = (tmm.x > 0.0f) ? tmm.x : tmm.y;
        float3 hitPos = ro + tSurface * marchDir;
        float3 nor = hitPos / PM_SPHERE_RADIUS;
        float3 reflected = reflect(marchDir, nor);
        float fre = pow(0.5f + clamp(dot(reflected, marchDir), 0.0f, 1.0f), 3.0f) * 1.3f;
        col += pmEnvironment(reflected, uniforms.time) * fre;
    }

    float2 faceUV = pmFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.2f * dot(faceUV, faceUV);
    col = 0.5f * log(1.0f + col);
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}