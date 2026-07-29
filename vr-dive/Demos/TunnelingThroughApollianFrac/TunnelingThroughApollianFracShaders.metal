// TunnelingThroughApollianFracShaders.metal
// "Tunneling through apollian fractals" — cube-container adaptation of ShaderToy tlcBWH.
// Source: https://www.shadertoy.com/view/tlcBWH
// License: CC0.
//
// Adaptation notes:
// - The original shader uses a synthetic tunnel camera that advances along a
//   procedural 3D path and alpha-composites a stack of patterned planes.
// - This version reconstructs a real per-eye ray from the visible 2 m cube,
//   then maps that ray into the original tunnel camera frame so the content is
//   visible from all viewing directions and when the viewer is inside the cube.

#include <metal_stdlib>
using namespace metal;

struct TunnelingThroughApollianFracUniforms {
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

struct TunnelingThroughApollianFracVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float TT_PI = 3.141592654f;
static constant float TT_TAU = 6.283185307f;
static constant float3 TT_STD_GAMMA = float3(2.2f);
static constant float TT_SCENE_SCALE = 0.5f;
static constant float3 TT_BOX_HALF = float3(1.0f);
static constant int TT_FURTHEST = 8;
static constant int TT_FADE_FROM = 5;

vertex TunnelingThroughApollianFracVertexOut tunnelingThroughApollianFracVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant TunnelingThroughApollianFracUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    TunnelingThroughApollianFracVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 ttRotate(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float ttHash(float x) {
    return fract(sin(x * 12.9898f) * 13758.5453f);
}

static float2 ttToPolar(float2 p) {
    return float2(length(p), atan2(p.y, p.x));
}

static float2 ttToRect(float2 p) {
    return float2(p.x * cos(p.y), p.x * sin(p.y));
}

static float ttTanhApprox(float x) {
    float x2 = x * x;
    return clamp(x * (27.0f + x2) / (27.0f + 9.0f * x2), -1.0f, 1.0f);
}

static float ttPMin(float a, float b, float k) {
    float h = clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return mix(b, a, h) - k * h * (1.0f - h);
}

static float3 ttHsv2rgb(float3 c) {
    const float4 K = float4(1.0f, 0.6666667f, 0.3333333f, 3.0f);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0f - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), c.y);
}

static float ttApollian(float4 p, float s) {
    float scale = 1.0f;
    for (int i = 0; i < 7; ++i) {
        p = -1.0f + 2.0f * fract(0.5f * p + 0.5f);
        float r2 = dot(p, p);
        float k = s / max(r2, 1.0e-5f);
        p *= k;
        scale *= k;
    }
    return abs(p.y) / scale;
}

static float ttHex(float2 p, float r) {
    const float3 k = float3(-0.86602540378f, 0.5f, 0.57735026919f);
    p = p.yx;
    p = abs(p);
    p -= 2.0f * min(dot(k.xy, p), 0.0f) * k.xy;
    p -= float2(clamp(p.x, -k.z * r, k.z * r), r);
    return length(p) * sign(p.y);
}

static float ttCircle(float2 p, float r) {
    return length(p) - r;
}

static float ttModMirror1(thread float &p, float size) {
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

static float ttSabs(float x, float k) {
    float ax = abs(x);
    float quad = (0.5f / k) * x * x + k * 0.5f;
    return mix(quad, ax, step(0.0f, ax - k));
}

static float ttSmoothKaleidoscope(thread float2 &p, float sm, float rep) {
    float2 polar = ttToPolar(p);
    float angle = polar.y;
    float rn = ttModMirror1(angle, TT_TAU / rep);
    polar.y = angle;
    float sa = TT_PI / rep - ttSabs(TT_PI / rep - abs(polar.y), sm);
    polar.y = copysign(sa, polar.y);
    p = ttToRect(polar);
    return rn;
}

static float4 ttAlphaBlend(float4 back, float4 front) {
    float w = front.w + back.w * (1.0f - front.w);
    if (w <= 0.0f) {
        return float4(0.0f);
    }
    float3 xyz = (front.xyz * front.w + back.xyz * back.w * (1.0f - front.w)) / w;
    return float4(xyz, w);
}

static float3 ttAlphaBlend3(float3 back, float4 front) {
    return mix(back, front.xyz, front.w);
}

static float3 ttOffset(float z) {
    float a = z;
    float2 p = -0.10f * (
        float2(cos(a), sin(a * sqrt(2.0f)))
        + float2(cos(a * sqrt(0.75f)), sin(a * sqrt(0.5f))));
    return float3(p, z);
}

static float3 ttDOffset(float z) {
    float eps = 0.1f;
    return 0.5f * (ttOffset(z + eps) - ttOffset(z - eps)) / eps;
}

static float3 ttDDOffset(float z) {
    float eps = 0.1f;
    return 0.125f * (ttDOffset(z + eps) - ttDOffset(z - eps)) / eps;
}

static float ttWeird(float2 p, float h, float time) {
    float z = 4.0f;
    float tm = 0.1f * time + h * 10.0f;
    p = ttRotate(p, tm * 0.5f);
    float r = 0.5f;
    float4 off = float4(
        r * (0.5f + 0.5f * sin(tm * sqrt(3.0f))),
        r * (0.5f + 0.5f * sin(tm * sqrt(1.5f))),
        r * (0.5f + 0.5f * sin(tm * sqrt(2.0f))),
        0.0f);
    float4 pp = float4(p.x, p.y, 0.0f, 0.0f) + off;
    pp.w = 0.125f * (1.0f - ttTanhApprox(length(pp.xyz)));
    pp.yz = ttRotate(pp.yz, tm);
    pp.xz = ttRotate(pp.xz, tm * sqrt(0.5f));
    pp /= z;
    return ttApollian(pp, 0.8f + h) * z;
}

static float ttCircles(float2 p) {
    float2 pp = ttToPolar(p);
    const float ss = 0.25f;
    pp.x = fract(pp.x * ss) / ss;
    p = ttToRect(pp);
    return ttCircle(p, 1.0f);
}

static float ttOnionize(float d) {
    d = abs(d) - 0.02f;
    d = abs(d) - 0.005f;
    d = abs(d) - 0.0025f;
    return d;
}

static float2 ttDf(float2 p, float h, float time) {
    float2 wp = p;
    float rep = 10.0f;
    float ss = 0.05f * 6.0f / rep;
    ttSmoothKaleidoscope(wp, ss, rep);

    float d0 = ttWeird(wp, h, time);
    d0 = ttOnionize(d0);
    float d1 = ttHex(p, 0.25f) - 0.1f;
    float d2 = ttCircles(p);
    const float lw = 0.0125f;
    d2 = abs(d2) - lw;
    float d = ttPMin(ttPMin(d0, d2, 0.1f), abs(d1) - lw, 0.05f);
    return float2(d, d1 + lw);
}

static float4 ttPlane(float3 ro, float3 rd, float3 pp, float3 off, float aa, float n, float time) {
    float h = ttHash(n);
    float s = 0.25f * mix(0.5f, 0.25f, h);
    float dd = length(pp - ro);

    const float3 nor = float3(0.0f, 0.0f, 1.0f);
    const float3 loff = float3(0.125f, 0.0625f, -0.125f);
    float3 lp1 = ro + loff;
    float3 lp2 = ro + loff * float3(-1.0f, 1.0f, 1.0f);
    float3 ld1 = normalize(pp - lp1);
    float3 ld2 = normalize(pp - lp2);
    float ref1 = pow(max(dot(nor, ld1), 0.0f), 20.0f);
    float ref2 = pow(max(dot(nor, ld2), 0.0f), 20.0f);
    float3 col1 = float3(0.75f, 0.5f, 1.0f);
    float3 col2 = float3(1.0f, 0.5f, 0.75f);

    float2 p = (pp - off * float3(1.0f, 1.0f, 0.0f)).xy;
    p = ttRotate(p, TT_TAU * h);
    float2 d2 = ttDf(p / s, h, time) * s;

    float ha = smoothstep(-aa, aa, d2.y);
    float d = d2.x;
    float4 col = float4(0.0f);

    float l = length(10.0f * p);
    float ddf = 1.0f / (1.0f + 2.0f * dd);
    float hue = fract(0.75f * l - 0.1f * time) + 0.45f;
    float sat = 0.75f * ttTanhApprox(2.0f * l) * ddf;
    float vue = sqrt(ddf);
    float3 bcol = ttHsv2rgb(float3(hue, sat, vue));
    col.xyz = mix(col.xyz, bcol, smoothstep(-aa, aa, -d));
    float glow = exp(-(10.0f + 100.0f * ttTanhApprox(l)) * 10.0f * max(d, 0.0f) * ddf);
    col.xyz += 0.5f * sqrt(bcol.zxy) * glow;
    col.w = ha * mix(0.75f, 1.0f, ha * glow);
    col.xyz += 0.125f * col.w * (col1 * ref1 + col2 * ref2);

    return col;
}

static float3 ttSkyColor(float3 rd) {
    float ld = max(dot(rd, float3(0.0f, 0.0f, 1.0f)), 0.0f);
    return 1.25f * float3(1.0f, 0.75f, 0.85f) * ttTanhApprox(3.0f * pow(ld, 100.0f));
}

static float3 ttColorAlongPath(float3 ww, float3 uu, float3 vv, float3 ro, float2 p, float time) {
    float2 np = p + float2(0.0015f, 0.0015f);
    float rdd = 2.0f + 0.5f * ttTanhApprox(length(p));
    float3 rd = normalize(p.x * uu + p.y * vv + rdd * ww);
    float3 nrd = normalize(np.x * uu + np.y * vv + rdd * ww);
    if (abs(rd.z) < 1.0e-4f || abs(nrd.z) < 1.0e-4f) {
        return ttSkyColor(rd);
    }

    const float planeDist = 0.25f;
    const float fadeDist = planeDist * float(TT_FURTHEST - TT_FADE_FROM);
    float nz = floor(ro.z / planeDist);
    float3 skyCol = ttSkyColor(rd);

    float4 accum = float4(0.0f);
    const float cutOff = 0.95f;
    for (int i = 1; i <= TT_FURTHEST; ++i) {
        float planeIndex = nz + float(i);
        float pz = planeDist * planeIndex;
        float pd = (pz - ro.z) / rd.z;
        if (pd > 0.0f && accum.w < cutOff) {
            float3 pp = ro + rd * pd;
            float3 npp = ro + nrd * pd;
            float aa = 3.0f * length(pp - npp);
            float3 off = ttOffset(pp.z);

            float4 pcol = ttPlane(ro, rd, pp, off, aa, planeIndex, time);
            float dz = pp.z - ro.z;
            float fadeIn = exp(-2.5f * max((dz - planeDist * float(TT_FADE_FROM)) / max(fadeDist, 1.0e-4f), 0.0f));
            float fadeOut = smoothstep(0.0f, planeDist * 0.1f, dz);
            pcol.xyz = mix(skyCol, pcol.xyz, fadeIn);
            pcol.w *= fadeOut;
            pcol = clamp(pcol, 0.0f, 1.0f);
            accum = ttAlphaBlend(pcol, accum);
        } else if (pd > 0.0f) {
            break;
        }
    }

    return ttAlphaBlend3(skyCol, accum);
}

static float3 ttPostProcess(float3 col, float2 q) {
    col = clamp(col, 0.0f, 1.0f);
    col = pow(col, 1.0f / TT_STD_GAMMA);
    col = col * 0.6f + 0.4f * col * col * (3.0f - 2.0f * col);
    col = mix(col, float3(dot(col, float3(0.33f))), -0.4f);
    col *= 0.5f + 0.5f * pow(max(19.0f * q.x * q.y * (1.0f - q.x) * (1.0f - q.y), 0.0f), 0.7f);
    return col;
}

static float2 ttBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 ttFaceUV(float3 p) {
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

fragment float4 tunnelingThroughApollianFracFragment(
    TunnelingThroughApollianFracVertexOut in [[stage_in]],
    constant TunnelingThroughApollianFracUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 cameraWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (cameraWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 viewDir = normalize(surfacePos - eye);

    bool insideOuter = all(abs(eye) < TT_BOX_HALF - 1.0e-3f);
    float2 tOuter = ttBoxIntersect(eye, viewDir, TT_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 localOrigin = eye + viewDir * (tStart + 0.001f);

    float time = uniforms.time;
    float tm = time * 0.2f;
    float3 pathOrigin = ttOffset(tm);
    float3 tangent = normalize(ttDOffset(tm));
    float3 curvature = ttDDOffset(tm);
    float3 binormal = normalize(cross(normalize(float3(0.0f, 1.0f, 0.0f) + curvature), tangent));
    if (!all(isfinite(binormal)) || length(binormal) < 1.0e-4f) {
        binormal = normalize(cross(float3(1.0f, 0.0f, 0.0f), tangent));
    }
    float3 normal = cross(tangent, binormal);

    float3 sceneOrigin = localOrigin * TT_SCENE_SCALE;
    sceneOrigin.xy += pathOrigin.xy;
    sceneOrigin.z += tm;

    float3 ww = tangent;
    float3 uu = binormal;
    float3 vv = normal;
    float2 p = float2(dot(viewDir, uu), dot(viewDir, vv)) / max(dot(viewDir, ww), 0.22f);

    float3 col = ttColorAlongPath(ww, uu, vv, sceneOrigin, p, time);
    float trail = pow(clamp(1.0f - abs(dot(viewDir, ww)), 0.0f, 1.0f), 2.0f);
    float3 background = float3(0.008f, 0.01f, 0.016f);
    background += float3(0.18f, 0.08f, 0.24f) * (0.04f + 0.08f * trail);

    float2 faceUV = ttFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.18f * dot(faceUV, faceUV);

    col += background;
    col = ttPostProcess(col, ttFaceUV(surfacePos));
    col = sqrt(max(col, 0.0f));
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}