// GoldenApollianShaders.metal
// "Golden apollian" — cube-container adaptation of ShaderToy WlcfRS.
// Source: https://www.shadertoy.com/view/WlcfRS
// License: CC0.
//
// Adaptation notes:
// - The original shader renders a forward-moving screen-space camera through a
//   stack of procedural planes. This version reconstructs a real per-eye ray,
//   starts it at the visible 2 m cube surface, and maps that ray into the
//   original path-following camera frame so the content remains visible from all
//   directions and also when the viewer is inside the cube.
// - The procedural plane stack is evaluated in scene space beyond the container,
//   so the effect itself is not clipped by cube bounds.

#include <metal_stdlib>
using namespace metal;

struct GoldenApollianUniforms {
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

struct GoldenApollianVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct GAEffect {
    float lw;
    float tw;
    float sk;
    float cs;
};

static constant float GA_PI = 3.141592654f;
static constant float GA_TAU = 6.283185307f;
static constant float GA_PLANE_PERIOD = 5.0f;
static constant float3 GA_STD_GAMMA = float3(2.2f, 2.2f, 2.2f);
static constant float3 GA_PLANE_COL = float3(1.0f, 1.2f, 1.5f);
static constant float3 GA_BASE_RING_COL = float3(1.0f, 0.772459f, 0.435275f);
static constant float3 GA_SUN_COL = float3(1.0f, 0.8f, 0.88f);
static constant float GA_SCENE_SCALE = 0.35f;
static constant float3 GA_BOX_HALF = float3(1.0f);
static constant int GA_FURTHEST = 9;
static constant int GA_FADE_FROM = 5;

vertex GoldenApollianVertexOut goldenApollianVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant GoldenApollianUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    GoldenApollianVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static GAEffect gaEffectForIndex(int idx) {
    switch (idx) {
        case 0: return GAEffect{0.125f, 0.0f, 0.0f, 0.0f};
        case 1: return GAEffect{0.125f, 0.0f, 0.0f, 1.0f};
        case 2: return GAEffect{0.125f, 0.0f, 1.0f, 1.0f};
        case 3: return GAEffect{0.125f, 1.0f, 1.0f, 1.0f};
        case 4: return GAEffect{0.125f, 1.0f, 1.0f, 0.0f};
        default: return GAEffect{0.125f, 1.0f, 0.0f, 0.0f};
    }
}

static float2 gaRotate(float2 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float gaPSin(float x) {
    return 0.5f + 0.5f * sin(x);
}

static float gaHash(float x) {
    x += 100.0f;
    return fract(sin(x * 12.9898f) * 13758.5453f);
}

static float2 gaToPolar(float2 p) {
    return float2(length(p), atan2(p.y, p.x));
}

static float2 gaToRect(float2 p) {
    return float2(p.x * cos(p.y), p.x * sin(p.y));
}

static float gaTanhApprox(float x) {
    float x2 = x * x;
    return clamp(x * (27.0f + x2) / (27.0f + 9.0f * x2), -1.0f, 1.0f);
}

static float gaPMin(float a, float b, float k) {
    float h = clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return mix(b, a, h) - k * h * (1.0f - h);
}

static float gaCircle(float2 p, float r) {
    return length(p) - r;
}

static float gaHex(float2 p, float r) {
    const float3 k = float3(-0.86602540378f, 0.5f, 0.57735026919f);
    p = p.yx;
    p = abs(p);
    p -= 2.0f * min(dot(k.xy, p), 0.0f) * k.xy;
    p -= float2(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}

static float gaL2(float3 v) {
    return dot(v, v);
}

static float gaModMirror1(thread float &p, float size) {
    float halfsize = size * 0.5f;
    float c = floor((p + halfsize) / size);
    p = fmod(p + halfsize, size);
    if (p < 0.0f) {
        p += size;
    }
    p -= halfsize;
    p *= fmod(c, 2.0f) * 2.0f - 1.0f;
    return c;
}

static float gaSabs(float x, float k) {
    float ax = abs(x);
    float quad = (0.5f / k) * x * x + k * 0.5f;
    return mix(quad, ax, step(0.0f, ax - k));
}

static float gaSmoothKaleidoscope(thread float2 &p, float sm, float rep) {
    float2 polar = gaToPolar(p);
    float angle = polar.y;
    float rn = gaModMirror1(angle, GA_TAU / rep);
    polar.y = angle;
    float sa = GA_PI / rep - gaSabs(GA_PI / rep - abs(polar.y), sm);
    polar.y = copysign(sa, polar.y);
    p = gaToRect(polar);
    return rn;
}

static float4 gaAlphaBlend(float4 back, float4 front) {
    float w = front.w + back.w * (1.0f - front.w);
    if (w <= 0.0f) {
        return float4(0.0f);
    }
    float3 xyz = (front.xyz * front.w + back.xyz * back.w * (1.0f - front.w)) / w;
    return float4(xyz, w);
}

static float3 gaAlphaBlend3(float3 back, float4 front) {
    return mix(back, front.xyz, front.w);
}

static float gaApollian(float4 p, float s, GAEffect effect) {
    float scale = 1.0f;
    for (int i = 0; i < 7; ++i) {
        p = -1.0f + 2.0f * fract(0.5f * p + 0.5f);
        float r2 = dot(p, p);
        float k = s / max(r2, 1.0e-5f);
        p *= k;
        scale *= k;
    }

    float lw = 0.00125f * effect.lw;
    float d0 = abs(p.y) - lw * scale;
    float d1 = abs(gaCircle(p.xz, 0.005f * scale)) - lw * scale;
    float d = d0;
    d = mix(d, min(d, d1), effect.tw);
    return d / scale;
}

static float3 gaOffset(float z) {
    float a = z;
    float2 p = -0.075f * (
        float2(cos(a), sin(a * sqrt(2.0f)))
        + float2(cos(a * sqrt(0.75f)), sin(a * sqrt(0.5f))));
    return float3(p, z);
}

static float3 gaDOffset(float z) {
    float eps = 0.1f;
    return 0.5f * (gaOffset(z + eps) - gaOffset(z - eps)) / eps;
}

static float3 gaDDOffset(float z) {
    float eps = 0.1f;
    return 0.125f * (gaDOffset(z + eps) - gaDOffset(z - eps)) / eps;
}

static float gaWeird(float2 p, float h, float time, GAEffect effect) {
    float z = 4.0f;
    float tm = 0.1f * time + h * 10.0f;
    p = gaRotate(p, tm * 0.5f);
    float r = 0.5f;
    float4 off = float4(
        r * gaPSin(tm * sqrt(3.0f)),
        r * gaPSin(tm * sqrt(1.5f)),
        r * gaPSin(tm * sqrt(2.0f)),
        0.0f);
    float4 pp = float4(p.x, p.y, 0.0f, 0.0f) + off;
    pp.w = 0.125f * (1.0f - gaTanhApprox(length(pp.xyz)));
    pp.yz = gaRotate(pp.yz, tm);
    pp.xz = gaRotate(pp.xz, tm * sqrt(0.5f));
    pp /= z;
    float d = gaApollian(pp, 0.8f + h, effect);
    return d * z;
}

static float gaCircles(float2 p) {
    float2 pp = gaToPolar(p);
    const float ss = 2.0f;
    pp.x = fract(pp.x / ss) * ss;
    p = gaToRect(pp);
    return gaCircle(p, 1.0f);
}

static float2 gaDf2(float2 p, float h, float time, GAEffect effect) {
    float2 wp = p;
    float rep = 2.0f * round(mix(5.0f, 15.0f, h * h));
    float ss = 0.05f * 6.0f / rep;

    if (effect.sk > 0.0f) {
        gaSmoothKaleidoscope(wp, ss, rep);
    }

    float d0 = gaWeird(wp, h, time, effect);
    float d1 = gaHex(p, 0.25f) - 0.1f;
    float d2 = gaCircles(p);
    const float lw = 0.0125f;
    d2 = abs(d2) - lw;
    float d = d0;

    if (effect.cs > 0.0f) {
        d = gaPMin(d, d2, 0.1f);
    }

    d = gaPMin(d, abs(d1) - lw, 0.1f);
    d = max(d, -(d1 + lw));
    return float2(d, d1 + lw);
}

static float2 gaDf3(float3 p, float3 off, float s, float2x2 rot, float h, float time, GAEffect effect) {
    float2 p2 = p.xy - off.xy;
    p2 = rot * p2;
    return gaDf2(p2 / s, h, time, effect) * s;
}

static float3 gaSkyColor(float3 rd) {
    float ld = max(dot(rd, float3(0.0f, 0.0f, 1.0f)), 0.0f);
    return GA_SUN_COL * gaTanhApprox(3.0f * pow(ld, 100.0f));
}

static float2x2 gaRotMatrix(float a) {
    float c = cos(a);
    float s = sin(a);
    return float2x2(float2(c, -s), float2(s, c));
}

static float4 gaPlane(
    float3 ro,
    float3 rd,
    float3 pp,
    float pd,
    float3 off,
    float aa,
    float planeNumber,
    float time
) {
    int pi = int(fmod(floor(planeNumber / GA_PLANE_PERIOD), 6.0f));
    if (pi < 0) {
        pi += 6;
    }
    GAEffect effect = gaEffectForIndex(pi);

    float h = gaHash(planeNumber);
    float s = 0.25f * mix(0.5f, 0.25f, h);

    const float3 nor = float3(0.0f, 0.0f, -1.0f);
    const float3 loff = 2.0f * float3(0.125f, 0.0625f, -0.125f);
    float3 lp1 = ro + loff;
    float3 lp2 = ro + loff * float3(-2.0f, 1.0f, 1.0f);

    float2x2 rot = gaRotMatrix(GA_TAU * h);
    float2 d2 = gaDf3(pp, off, s, rot, h, time, effect);

    float3 ld1 = normalize(lp1 - pp);
    float3 ld2 = normalize(lp2 - pp);
    float dif1 = pow(max(dot(nor, ld1), 0.0f), 5.0f);
    float dif2 = pow(max(dot(nor, ld2), 0.0f), 5.0f);
    float3 ref = reflect(rd, nor);
    float spe1 = pow(max(dot(ref, ld1), 0.0f), 30.0f);
    float spe2 = pow(max(dot(ref, ld2), 0.0f), 30.0f);

    const float boff = 0.00625f;
    float dbt = boff / max(rd.z, 1.0e-4f);

    float3 bpp = ro + (pd + dbt) * rd;
    float3 srd1 = normalize(lp1 - bpp);
    float3 srd2 = normalize(lp2 - bpp);
    float bl21 = gaL2(lp1 - bpp);
    float bl22 = gaL2(lp2 - bpp);

    float st1 = -boff / min(srd1.z, -1.0e-4f);
    float st2 = -boff / min(srd2.z, -1.0e-4f);

    float3 spp1 = bpp + st1 * srd1;
    float3 spp2 = bpp + st2 * srd2;

    float2 bd = gaDf3(bpp, off, s, rot, h, time, effect);
    float2 sd1 = gaDf3(spp1, off, s, rot, h, time, effect);
    float2 sd2 = gaDf3(spp2, off, s, rot, h, time, effect);

    float3 col = float3(0.0f);
    const float ss = 200.0f;
    col += 0.1125f * GA_PLANE_COL * dif1 * (1.0f - exp(-ss * max(sd1.x, 0.0f))) / max(bl21, 1.0e-4f);
    col += 0.05625f * GA_PLANE_COL * dif2 * (1.0f - exp(-ss * max(sd2.x, 0.0f))) / max(bl22, 1.0e-4f);

    float3 ringCol = GA_BASE_RING_COL;
    ringCol *= clamp(0.1f + 2.5f * (0.1f + 0.25f * ((dif1 * dif1 / max(bl21, 1.0e-4f)) + (dif2 * dif2 / max(bl22, 1.0e-4f)))), 0.0f, 1.0f);
    ringCol += sqrt(GA_BASE_RING_COL) * spe1 * 2.0f;
    ringCol += sqrt(GA_BASE_RING_COL) * spe2 * 2.0f;
    col = mix(col, ringCol, smoothstep(-aa, aa, -d2.x));

    float ha = smoothstep(-aa, aa, bd.y);
    return float4(col, mix(0.0f, 1.0f, ha));
}

static float3 gaPostProcess(float3 col, float2 q) {
    col = clamp(col, 0.0f, 1.0f);
    col = pow(col, 1.0f / GA_STD_GAMMA);
    col = col * 0.6f + 0.4f * col * col * (3.0f - 2.0f * col);
    col = mix(col, float3(dot(col, float3(0.33f))), -0.4f);
    col *= 0.5f + 0.5f * pow(max(19.0f * q.x * q.y * (1.0f - q.x) * (1.0f - q.y), 0.0f), 0.7f);
    return col;
}

static float2 gaBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 gaFaceUV(float3 p) {
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

static float3 gaColorAlongPath(float3 ww, float3 uu, float3 vv, float3 ro, float2 p, float time) {
    float2 np = p + float2(0.0015f, 0.0015f);
    float rdd = 2.0f - 0.5f * gaTanhApprox(length(p));

    float3 rd = normalize(p.x * uu + p.y * vv + rdd * ww);
    float3 nrd = normalize(np.x * uu + np.y * vv + rdd * ww);
    if (abs(rd.z) < 1.0e-4f || abs(nrd.z) < 1.0e-4f) {
        return gaSkyColor(rd);
    }

    const float planeDist = 0.25f;
    const float fadeDist = planeDist * float(GA_FURTHEST - GA_FADE_FROM);
    float nz = floor(ro.z / planeDist);
    float3 skyCol = gaSkyColor(rd);

    float4 accum = float4(0.0f);
    const float cutOff = 0.95f;
    for (int i = 1; i <= GA_FURTHEST; ++i) {
        float planeIndex = nz + float(i);
        float pz = planeDist * planeIndex;
        float pd = (pz - ro.z) / rd.z;

        if (pd > 0.0f && accum.w < cutOff) {
            float3 pp = ro + rd * pd;
            float3 npp = ro + nrd * pd;
            float aa = 3.0f * length(pp - npp);
            float3 off = gaOffset(pp.z);

            float4 pcol = gaPlane(ro, rd, pp, pd, off, aa, planeIndex, time);
            float dz = pp.z - ro.z;
            float fadeIn = exp(-2.5f * max((dz - planeDist * float(GA_FADE_FROM)) / max(fadeDist, 1.0e-4f), 0.0f));
            float fadeOut = smoothstep(0.0f, planeDist * 0.1f, dz);
            pcol.xyz = mix(skyCol, pcol.xyz, fadeIn);
            pcol.w *= fadeOut;
            pcol = clamp(pcol, 0.0f, 1.0f);
            accum = gaAlphaBlend(pcol, accum);
        } else {
            break;
        }
    }

    return gaAlphaBlend3(skyCol, accum);
}

fragment float4 goldenApollianFragment(
    GoldenApollianVertexOut in [[stage_in]],
    constant GoldenApollianUniforms &uniforms [[buffer(0)]],
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

    bool insideOuter = all(abs(eye) < GA_BOX_HALF - 1.0e-3f);
    float2 tOuter = gaBoxIntersect(eye, viewDir, GA_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 localOrigin = eye + viewDir * (tStart + 0.001f);

    float time = uniforms.time;
    float tm = time * 0.125f;
    float3 pathOrigin = gaOffset(tm);
    float3 tangent = normalize(gaDOffset(tm));
    float3 curvature = gaDDOffset(tm);
    float3 binormal = normalize(cross(normalize(float3(0.0f, 1.0f, 0.0f) + curvature), tangent));
    if (!all(isfinite(binormal)) || length(binormal) < 1.0e-4f) {
        binormal = normalize(cross(float3(1.0f, 0.0f, 0.0f), tangent));
    }
    float3 normal = cross(tangent, binormal);

    float3 sceneOrigin = localOrigin * GA_SCENE_SCALE;
    sceneOrigin.xy += pathOrigin.xy;
    sceneOrigin.z += tm;

    float3 ww = tangent;
    float3 uu = binormal;
    float3 vv = normal;
    float2 p = float2(dot(viewDir, uu), dot(viewDir, vv)) / max(dot(viewDir, ww), 0.22f);

    float3 col = gaColorAlongPath(ww, uu, vv, sceneOrigin, p, time);

    float trail = pow(clamp(1.0f - abs(dot(viewDir, ww)), 0.0f, 1.0f), 2.0f);
    float haze = 0.04f / (0.12f + abs(viewDir.y));
    float3 background = float3(0.006f, 0.008f, 0.014f);
    background += GA_BASE_RING_COL * (0.035f + 0.10f * trail);
    background += float3(0.8f, 0.68f, 0.52f) * haze * 0.06f;

    float2 faceUV = gaFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.18f * dot(faceUV, faceUV);

    col += background;
    col = gaPostProcess(col, gaFaceUV(surfacePos));
    col = sqrt(max(col, 0.0f));
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}