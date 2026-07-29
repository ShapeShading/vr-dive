// StarryPlanesShaders.metal
// "Starry planes" — cube-container adaptation of ShaderToy MfjyWK.
// Source: https://www.shadertoy.com/view/MfjyWK
// Original shader is marked CC0. This adaptation preserves the core stacked
// plane-marcher idea, star-shaped masks, curved path offsets, and ACES-like tone
// mapping while replacing the screen-space flythrough camera with a real per-eye
// ray that enters a visible 2 m cube container.

#include <metal_stdlib>
using namespace metal;

struct StarryPlanesUniforms {
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

struct StarryPlanesVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float SP_PI = 3.14159265359f;
static constant float SP_PLANE_DIST = 0.5f;
static constant float SP_FURTHEST = 16.0f;
static constant float SP_FADE_FROM = 8.0f;
static constant float2 SP_PATH_A = float2(0.31f, 0.41f);
static constant float2 SP_PATH_B = float2(1.0f, 0.70710678f);
static constant float3 SP_BOX_HALF = float3(1.0f);

static float2 spRotate(float2 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float3 acesApprox(float3 v) {
    v = max(v, 0.0f);
    v *= 0.6f;
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), 0.0f, 1.0f);
}

static float3 offsetCurve(float z) {
    return float3(SP_PATH_B * sin(SP_PATH_A * z), z);
}

static float3 dOffsetCurve(float z) {
    return float3(SP_PATH_A * SP_PATH_B * cos(SP_PATH_A * z), 1.0f);
}

static float3 ddOffsetCurve(float z) {
    return float3(-SP_PATH_A * SP_PATH_A * SP_PATH_B * sin(SP_PATH_A * z), 0.0f);
}

static float4 alphaBlend(float4 back, float4 front) {
    float w = front.w + back.w * (1.0f - front.w);
    if (w <= 0.0f) {
        return float4(0.0f);
    }
    float3 xyz = (front.xyz * front.w + back.xyz * back.w * (1.0f - front.w)) / w;
    return float4(xyz, w);
}

static float pmin(float a, float b, float k) {
    float h = clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return mix(b, a, h) - k * h * (1.0f - h);
}

static float pabs(float a, float k) {
    return -pmin(a, -a, k);
}

static float star5(float2 p, float r, float rf, float sm) {
    p = -p;
    const float2 k1 = float2(0.809016994375f, -0.587785252292f);
    const float2 k2 = float2(-0.809016994375f, -0.587785252292f);
    p.x = abs(p.x);
    p -= 2.0f * max(dot(k1, p), 0.0f) * k1;
    p -= 2.0f * max(dot(k2, p), 0.0f) * k2;
    p.x = pabs(p.x, sm);
    p.y -= r;
    float2 ba = rf * float2(-k1.y, k1.x) - float2(0.0f, 1.0f);
    float h = clamp(dot(p, ba) / dot(ba, ba), 0.0f, r);
    return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}

static float3 palette(float n) {
    return 0.5f + 0.5f * sin(float3(0.0f, 1.0f, 2.0f) + n);
}

static float2 spBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 spFaceUV(float3 p) {
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

static float4 planeColor(
    float3 ro,
    float3 rd,
    float3 pp,
    float3 npp,
    float pd,
    float planeIndex
) {
    float aa = 3.0f * pd * distance(pp.xy, npp.xy);
    float2 p2 = pp.xy - offsetCurve(pp.z).xy;
    float2 doff = ddOffsetCurve(pp.z).xz;
    float2 ddoff = dOffsetCurve(pp.z).xz;
    float dd = dot(doff, ddoff);
    p2 = spRotate(p2, dd * SP_PI * 5.0f);

    float d0 = star5(p2, 0.45f, 1.6f, 0.2f) - 0.02f;
    float d1 = d0 - 0.01f;
    float d2 = length(p2);
    float colp = SP_PI * 100.0f;
    float colaa = aa * 200.0f;

    float stripe = mix(
        0.5f / max(d2 * d2, 1.0e-4f),
        1.0f,
        smoothstep(-0.5f + colaa, 0.5f + colaa, sin(d2 * colp)));
    float3 col = palette(0.5f * planeIndex + 2.0f * d2) * stripe / max(3.0f * d2 * d2, 1.0e-1f);
    col = mix(col, float3(2.0f), smoothstep(aa, -aa, d1));
    float alpha = smoothstep(aa, -aa, -d0);

    float starField = exp(-28.0f * abs(d0)) + 0.45f / (1.0f + 60.0f * d2 * d2);
    col += float3(1.6f, 1.7f, 2.2f) * starField * 0.12f;
    return float4(col, alpha);
}

static float3 colorAlongPath(float3 ww, float3 uu, float3 vv, float3 ro, float2 p, float time) {
    float2 np = p + float2(0.0015f, 0.0015f);
    float rdd = 1.75f;

    float3 rd = normalize(p.x * uu + p.y * vv + rdd * ww);
    float3 nrd = normalize(np.x * uu + np.y * vv + rdd * ww);
    if (abs(rd.z) < 1.0e-4f || abs(nrd.z) < 1.0e-4f || abs(ww.z) < 1.0e-4f) {
        return float3(0.0f);
    }

    float nz = floor(ro.z / SP_PLANE_DIST);
    float4 accum = float4(0.0f);
    float3 advancingOrigin = ro;
    float apd = 0.0f;

    for (int stepIndex = 1; stepIndex <= 16; ++stepIndex) {
        if (accum.w > 0.95f) {
            break;
        }

        float planeIndex = nz + float(stepIndex);
        float pz = SP_PLANE_DIST * planeIndex;
        float lpd = (pz - advancingOrigin.z) / rd.z;
        float npd = (pz - advancingOrigin.z) / nrd.z;
        float cpd = (pz - advancingOrigin.z) / ww.z;
        if (lpd <= 0.0f || npd <= 0.0f || cpd <= 0.0f) {
            continue;
        }

        float3 pp = advancingOrigin + rd * lpd;
        float3 npp = advancingOrigin + nrd * npd;
        float3 cp = advancingOrigin + ww * cpd;

        apd += lpd;

        float dz = pp.z - ro.z;
        float fadeIn = smoothstep(SP_PLANE_DIST * SP_FURTHEST, SP_PLANE_DIST * SP_FADE_FROM, dz);
        float fadeOut = smoothstep(0.0f, SP_PLANE_DIST * 0.1f, dz);
        float fadeOutRI = smoothstep(0.0f, SP_PLANE_DIST * 1.0f, dz);
        float ri = mix(1.0f, 0.9f, fadeOutRI * fadeIn);

        float4 pcol = planeColor(ro, rd, pp, npp, apd, planeIndex);
        pcol.xyz *= mix(0.92f, 1.12f, sin(cp.z * 0.35f + time * 0.4f) * 0.5f + 0.5f);
        pcol.xyz *= ri;
        pcol.w *= fadeOut * fadeIn;
        accum = alphaBlend(accum, pcol);
        advancingOrigin = pp;
    }

    return accum.xyz * accum.w;
}

vertex StarryPlanesVertexOut starryPlanesVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant StarryPlanesUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    StarryPlanesVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

fragment float4 starryPlanesFragment(
    StarryPlanesVertexOut in [[stage_in]],
    constant StarryPlanesUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 eye = (cameraWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 viewDir = normalize(surfacePos - eye);

    bool insideOuter = all(abs(eye) < SP_BOX_HALF - 1.0e-3f);
    float2 tOuter = spBoxIntersect(eye, viewDir, SP_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 localOrigin = eye + viewDir * (tStart + 0.001f);

    float time = uniforms.time * 0.9f;
    float pathTime = SP_PLANE_DIST * time * 1.7f;
    float3 pathOrigin = offsetCurve(pathTime);
    float3 tangent = normalize(dOffsetCurve(pathTime));
    float3 curvature = ddOffsetCurve(pathTime);
    float3 binormal = normalize(cross(float3(0.0f, 1.0f, 0.0f) + curvature, tangent));
    if (!all(isfinite(binormal)) || length(binormal) < 1.0e-4f) {
        binormal = normalize(cross(float3(1.0f, 0.0f, 0.0f), tangent));
    }
    float3 normal = cross(tangent, binormal);

    float3 sceneOrigin = localOrigin * 2.4f;
    sceneOrigin.xy += pathOrigin.xy;
    sceneOrigin.z += pathTime;

    float3 ww = tangent;
    float3 uu = binormal;
    float3 vv = normal;
    float2 p = float2(dot(viewDir, uu), dot(viewDir, vv)) / max(dot(viewDir, ww), 0.25f);

    float3 col = colorAlongPath(ww, uu, vv, sceneOrigin, p, time);

    float trail = pow(clamp(1.0f - abs(dot(viewDir, ww)), 0.0f, 1.0f), 2.0f);
    float haze = 0.06f / (0.12f + abs(viewDir.y));
    float3 background = float3(0.01f, 0.015f, 0.03f);
    background += palette(pathTime * 0.35f + trail) * (0.04f + 0.08f * trail);
    background += float3(0.6f, 0.7f, 1.0f) * haze * 0.08f;

    float2 faceUV = spFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.18f * dot(faceUV, faceUV);

    col += background;
    col = acesApprox(col);
    col = sqrt(max(col, 0.0f));
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}