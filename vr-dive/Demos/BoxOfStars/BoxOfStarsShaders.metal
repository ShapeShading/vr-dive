// BoxOfStarsShaders.metal
// Adapted from ShaderToy "Box of Stars".
// Source: https://www.shadertoy.com/view/NcsSz4
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Metal adaptation notes:
// - The original shader already renders a glass box with a volumetric star field.
//   This version preserves that internal effect, but reconstructs the ray from
//   the real per-eye camera and uses the visible 2 m cube as the outer entry
//   container for all viewing directions.
// - The internal glass box and volumetric pillars are simulated beyond the outer
//   cube boundary so the effect itself is not clipped by the container volume.
// - GLSL matrix/vector multiplication and out parameters are rewritten to match
//   Metal semantics.

#include <metal_stdlib>
using namespace metal;

struct BoxOfStarsUniforms {
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

struct BoxOfStarsVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float BOS_TSHIFT = 53.0f;
static constant float BOS_PI = 3.1415926f;
static constant float BOS_IOR = 1.33f;
static constant float3 BOS_DIMS = float3(0.75f, 0.75f, 1.25f);
static constant float3 BOS_OUTER_BOX_HALF = float3(1.0f);

vertex BoxOfStarsVertexOut boxOfStarsVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant BoxOfStarsUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    BoxOfStarsVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float dot2(float3 v) {
    return dot(v, v);
}

static float2 bosRotate(float2 v, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

static float3 bosRotateZVec(float3 v, float angle) {
    float2 xy = bosRotate(v.xy, angle);
    return float3(xy.x, xy.y, v.z);
}

static float segShadow(float3 ro, float3 rd, float3 pa, float sh) {
    float dm = dot(rd.yz, rd.yz);
    float k1 = (ro.x - pa.x) * dm;
    float k2 = (ro.x + pa.x) * dm;
    float2 k5 = (ro.yz + pa.yz) * dm;
    float k3 = dot(ro.yz + pa.yz, rd.yz);
    float2 k4 = (pa.yz + pa.yz) * rd.yz;
    float2 k6 = (pa.yz + pa.yz) * dm;

    for (int i = 0; i < 4; ++i) {
        float2 s = float2(float(i & 1), float(i >> 1));
        float t = dot(s, k4) - k3;
        if (t > 0.0f) {
            float3 term = float3(clamp(-rd.x * t, k1, k2), k5 - k6 * s) + rd * t;
            sh = min(sh, dot2(term) / max(t * t, 1.0e-6f));
        }
    }
    return sh;
}

static float boxSoftShadow(float3 ro, float3 rd, float3 rad, float sk) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 m = 1.0f / rd;
    float3 n = m * ro;
    float3 k = abs(m) * rad;

    float3 t1 = -n - k;
    float3 t2 = -n + k;

    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);
    if (tN < tF && tF > 0.0f) {
        return 0.0f;
    }

    float sh = 1.0f;
    sh = segShadow(ro.xyz, rd.xyz, rad.xyz, sh);
    sh = segShadow(ro.yzx, rd.yzx, rad.yzx, sh);
    sh = segShadow(ro.zxy, rd.zxy, rad.zxy, sh);
    sh = clamp(sk * sqrt(sh), 0.0f, 1.0f);
    return sh * sh * (3.0f - 2.0f * sh);
}

static float boxIntersect(float3 ro, float3 rd, float3 r, thread float3 &nn, bool entering) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 dr = 1.0f / rd;
    float3 n = ro * dr;
    float3 k = r * abs(dr);

    float3 pin = -k - n;
    float3 pout = k - n;
    float tin = max(pin.x, max(pin.y, pin.z));
    float tout = min(pout.x, min(pout.y, pout.z));
    if (tin > tout) {
        return -1.0f;
    }

    if (entering) {
        nn = -sign(rd) * step(pin.zxy, pin.xyz) * step(pin.yzx, pin.xyz);
        return tin;
    }

    nn = sign(rd) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);
    return tout;
}

static float2 outerBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float3 bosToOriginalScene(float3 v) {
    // The source shader is z-up. Rotate the reconstructed VR scene so the
    // apparent floor sits below the viewer and the glass box stands upright.
    return float3(v.x, -v.z, v.y);
}

static float3 bgcol(float3 rd) {
    return mix(float3(0.01f), float3(0.336f, 0.458f, 0.668f),
               1.0f - pow(abs(rd.z + 0.25f), 1.3f));
}

static float3 background(float3 ro, float3 rd, float3 l_dir, thread float &alpha) {
    float t = (-BOS_DIMS.z - ro.z) / rd.z;
    alpha = 0.0f;
    float3 bgc = bgcol(rd);
    if (t < 0.0f) {
        return bgc;
    }

    float2 uv = ro.xy + t * rd.xy;
    float3 lightDir = normalize(bosRotateZVec(l_dir + float3(0.0f, 0.0f, 1.0f), BOS_PI * 0.65f));
    float shad = boxSoftShadow(ro + t * rd, lightDir, BOS_DIMS, 1.5f);
    float aofac = smoothstep(-0.95f, 0.75f, length(abs(uv) - min(abs(uv), float2(0.45f))));
    aofac = min(aofac, smoothstep(-0.65f, 1.0f, shad));
    float lght = max(dot(normalize(ro + t * rd + float3(0.0f, 0.0f, -5.0f)),
                         normalize(bosRotateZVec(l_dir - float3(0.0f, 0.0f, 1.0f), BOS_PI * 0.65f))),
                     0.0f);
    float3 col = mix(float3(0.4f), float3(0.71f, 0.772f, 0.895f), lght * lght * aofac + 0.05f) * aofac;
    alpha = 1.0f - smoothstep(7.0f, 10.0f, length(uv));
    return mix(col * length(col) * 0.8f, bgc, smoothstep(7.0f, 10.0f, length(uv)));
}

static float3 hash33(float3 p3) {
    p3 = fract(p3 * float3(0.1031f, 0.1030f, 0.0973f));
    p3 += dot(p3, p3.yxz + 33.33f);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

static float3 waveDeform(float3 pos, float tOffset) {
    float3 deformed = pos;
    deformed.xz = bosRotate(deformed.xz, 0.4f);
    deformed += cos(deformed.zxy);
    deformed.xz = bosRotate(deformed.xz, 0.4f);
    deformed += cos(deformed.zxy * 2.0f - tOffset * 2.0f) * 0.5f;
    return deformed;
}

static float3 boxedStars(float3 ro, float3 rd, float time) {
    float tmod = fmod((time + BOS_TSHIFT) * 0.33f, 1000.0f);
    float starAccum = 0.0f;
    const int maxSteps = 150;
    const float stepSize = 0.0075f;
    float vt = 0.0f;
    float3 pillarAccum = float3(0.0f);
    const float pillarWidth = 2.0f;
    const float3 ambientBG = float3(0.0345f, 0.036f, 0.0915f);

    for (int i = 0; i < maxSteps; ++i) {
        float3 vp = ro + rd * vt;
        vp.yz = vp.zy;
        float3 p = vp * 10.0f;
        float3 id = floor(p);
        float3 q = fract(p) - 0.5f;
        float3 h = hash33(id);

        float3 pillarPos = p * 0.3f;
        float3 lightPillar = waveDeform(pillarPos + float3(0.0f, tmod, 0.0f), tmod);

        float2 cosPair = cos(lightPillar.xz);
        float innerBeamsDist = length(cosPair);
        float radialBound = length(pillarPos.xz) - pillarWidth;
        float pillarDist = max(innerBeamsDist, radialBound);
        float density = 0.02f / max(pillarDist, 0.001f);
        float3 pillarGradient = mix(float3(0.0f, 0.1f, 1.0f), float3(0.9f, 0.0f, 1.0f),
                                    smoothstep(-3.5f, 3.5f, pillarPos.y));

        if (pillarDist < 1.0f) {
            float3 gradxDens = pillarGradient * density;
            const float px = 1.2f;
            pillarAccum += clamp(float3(pow(gradxDens.x, px), pow(gradxDens.y, px), pow(gradxDens.z, px)), 0.0f, 1.0f);
        }

        if (h.z < 0.08f) {
            float3 movement = sin(tmod * (h * 2.0f + 1.0f)) * 0.15f;
            float3 offset = (h - 0.5f) * 0.3f + movement;
            float d = length(q - offset);
            float glow = pow(clamp(1.0f - d * 10.0f, 0.0f, 1.0f), 4.0f) * 2.0f;
            float fade = clamp(1.0f - vt * 1.2f, 0.0f, 1.0f);
            starAccum += glow * fade;
        }

        vt += stepSize;
    }

    float3 pillarColor = clamp(tanh(pillarAccum * 0.5f), 0.0f, 1.0f);
    float3 finalColor = ambientBG + float3(starAccum) + pillarColor;
    return clamp(finalColor, 0.0f, 1.0f);
}

fragment float4 boxOfStarsFragment(
    BoxOfStarsVertexOut in [[stage_in]],
    constant BoxOfStarsUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float cubeScale = max(uniforms.boxScale, 1.0e-4f);
    float3 eye = (camWorld - center) / cubeScale;
    float3 hit = (in.worldPos - center) / cubeScale;
    float3 rdOuter = normalize(hit - eye);

    bool insideOuter = all(abs(eye) < BOS_OUTER_BOX_HALF - 1.0e-3f);
    float2 tOuter = outerBoxIntersect(eye, rdOuter, BOS_OUTER_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    const float sceneScale = 2.0f;
    float3 roScene = bosToOriginalScene((eye + rdOuter * (tStart + 0.001f)) * sceneScale);
    float3 rdScene = normalize(bosToOriginalScene(rdOuter));

    float3 l_dir = normalize(bosRotateZVec(float3(0.0f, 1.0f, 0.0f), 0.5f));

    bool insideGlass = all(abs(roScene) < BOS_DIMS - 1.0e-3f);
    float3 ni = insideGlass ? -rdScene : float3(0.0f, 0.0f, 1.0f);
    float tGlass = 0.0f;
    float fadeborders = 1.0f;
    float3 glassSurface = roScene;

    if (!insideGlass) {
        tGlass = boxIntersect(roScene, rdScene, BOS_DIMS, ni, true);
        if (tGlass > 0.0f) {
            glassSurface = roScene + tGlass * rdScene;
            float2 coords = glassSurface.xy * ni.z / BOS_DIMS.xy
                + glassSurface.yz * ni.x / BOS_DIMS.yz
                + glassSurface.zx * ni.y / BOS_DIMS.zx;
            fadeborders = (1.0f - smoothstep(0.915f, 1.05f, abs(coords.x)))
                * (1.0f - smoothstep(0.915f, 1.05f, abs(coords.y)));
        }
    }

    if (!insideGlass && tGlass <= 0.0f) {
        float alpha;
        float3 bg = background(roScene, rdScene, l_dir, alpha);
        return float4(clamp(bg, 0.0f, 1.0f), 1.0f);
    }

    float R0 = (BOS_IOR - 1.0f) / (BOS_IOR + 1.0f);
    R0 *= R0;
    float3 nr = ni;
    float3 rdr = reflect(rdScene, nr);
    float talpha;
    float3 reflcol = background(glassSurface, rdr, l_dir, talpha);
    float3 interiorCol = boxedStars(glassSurface + rdScene * 0.001f, rdScene, uniforms.time);
    float fresnel = R0 + (1.0f - R0) * pow(1.0f - clamp(dot(-rdScene, nr), 0.0f, 1.0f), 5.0f);
    float3 col = mix(interiorCol * fadeborders, reflcol, pow(fresnel, 1.5f));
    col = clamp(col, 0.0f, 1.0f);

    return float4(col, 1.0f);
}