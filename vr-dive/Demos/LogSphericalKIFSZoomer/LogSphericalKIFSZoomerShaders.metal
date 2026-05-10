// LogSphericalKIFSZoomerShaders.metal
// "Log Spherical KIFS Zoomer" — cube-container adaptation of ShaderToy ctcGRf.
// Source: https://www.shadertoy.com/view/ctcGRf
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Adaptation notes:
// - The original shader ray-marches an unbounded log-spherical repeated KIFS scene
//   from a synthetic camera placed far from the origin.
// - This version reconstructs a real per-eye ray from a visible 2 m cube, starts
//   from the cube surface when the viewer is outside, keeps the actual SDF scene
//   in a larger decoupled scene space, and supports viewing from inside the cube.

#include <metal_stdlib>
using namespace metal;

struct LogSphericalKIFSZoomerUniforms {
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

struct LogSphericalKIFSZoomerVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float LKZ_GAMMA = 2.2f;
static constant int LKZ_MAX_STEPS = 90;
static constant float LKZ_MAX_DIST = 100.0f;
static constant float LKZ_MIN_DIST = 10.0f;
static constant float LKZ_GLOW_INT = 1.0f;
static constant float LKZ_PP_ACES = 1.0f;
static constant float LKZ_PP_CONT = 0.5f;
static constant float LKZ_PP_VIGN = 1.3f;
static constant float LKZ_AO_OCC = 0.5f;
static constant float LKZ_AO_SCA = 0.3f;
static constant float LKZ_PI = 3.14159265f;
static constant float LKZ_DENS = 0.9f;
static constant float LKZ_SCENE_SCALE = 26.0f;
static constant float3 LKZ_BOX_HALF = float3(1.0f);
static constant float3 LKZ_AMB_COL = float3(0.03f, 0.05f, 0.1f) * 5.5f;
static constant float3 LKZ_SUN_COL = float3(1.0f, 0.7f, 0.4f) * 1.2f;
static constant float3 LKZ_SKY_COL = float3(0.3f, 0.5f, 1.0f) * 0.04f;
static constant float LKZ_SPEC_EXP = 4.0f;

vertex LogSphericalKIFSZoomerVertexOut logSphericalKIFSZoomerVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant LogSphericalKIFSZoomerUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    LogSphericalKIFSZoomerVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float lkzSmooth(float a, float b, float t) {
    return smoothstep(a, b, t);
}

static float lkzSin3(float x) {
    float s = sin(x);
    return s * s * s;
}

static float2 lkzRot2D(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return c * p + s * float2(p.y, -p.x);
}

static float3 lkzRot(float3 p, float3 r) {
    p.xz = lkzRot2D(p.xz, r.y);
    p.yx = lkzRot2D(p.yx, r.z);
    p.zy = lkzRot2D(p.zy, r.x);
    return p;
}

static float2 lkzBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float lkzSdKMC(float3 p, int iters, float3 fTra, float3 fRot, float4 para) {
    int i = 0;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float r = dot(p, p);

    for (i = 0; i < iters && r < 1.0e6f; ++i) {
        if (i > 0) {
            p -= fTra;
            p = lkzRot(p, fRot);
        }

        p = abs(p);
        if (p.x - p.y < 0.0f) { x1 = p.y; p.y = p.x; p.x = x1; }
        if (p.x - p.z < 0.0f) { x1 = p.z; p.z = p.x; p.x = x1; }
        if (p.y - p.z < 0.0f) { y1 = p.z; p.z = p.y; p.y = y1; }

        p.z -= 0.5f * para.x * (para.y - 1.0f) / para.y;
        p.z = -abs(p.z);
        p.z += 0.5f * para.x * (para.y - 1.0f) / para.y;

        p.x = para.y * p.x - para.z * (para.y - 1.0f);
        p.y = para.y * p.y - para.w * (para.y - 1.0f);
        p.z = para.y * p.z;
        r = dot(p, p);
    }

    return length(p) * pow(para.y, float(-i));
}

static float3 lkzHsv2rgbSmooth(float3 c) {
    float3 rgb = clamp(abs(fmod(c.x * 6.0f + float3(0.0f, 4.0f, 2.0f), 6.0f) - 3.0f) - 1.0f, 0.0f, 1.0f);
    rgb = rgb * rgb * (3.0f - 2.0f * rgb);
    return c.z * mix(float3(1.0f), rgb, c.y);
}

static float3 lkzPalette(int index, float time) {
    switch (index) {
        case 0:
            return float3(1.0f, 1.0f, 1.0f);
        case 1:
            return float3(1.0f, 0.8f, 0.6f);
        case 2:
            return float3(0.6f, 0.8f, 1.0f);
        case 3:
            return lkzHsv2rgbSmooth(float3(fract(time / 21.0f), 0.65f, 0.8f));
        default:
            return float3(0.0f);
    }
}

static float2 lkzSDF(float3 p, float depth, float time) {
    float d = LKZ_MAX_DIST;
    float col = 0.0f;

    p = abs(lkzRot(p, float3(10.5f - depth)));

    float sphere = length(p - float3(1.8f + sin(time / 3.0f + depth) * 0.6f, 0.0f, 0.0f)) - 0.1f;
    col = mix(col, 1.7f, step(sphere, d));
    d = min(sphere, d);

    float torus = length(float2(length(p.yz) - 1.2f, p.x)) - 0.01f;
    col = mix(col, 1.3f, step(torus, d));
    d = min(torus, d);

    float menger = lkzSdKMC(
        p * 2.9f,
        8,
        float3(sin(time / 53.0f)) * 0.4f,
        float3(lkzSin3(time / 64.0f) * LKZ_PI),
        float4(2.0f, 3.5f, 4.5f, 5.5f)) / 2.9f;
    col = mix(col, floor(fmod(length(p) * 1.5f, 4.0f)) + 0.5f, step(menger, d));
    d = min(menger, d);

    return float2(d, col);
}

static float2 lkzMap(float3 p, float time) {
    float r = max(length(p), 1.0e-5f);
    float theta = acos(clamp(p.z / r, -1.0f, 1.0f));
    float phi = atan2(p.y, p.x);
    p = float3(log(r), theta, phi);

    float t = time / 10.0f;
    p.x -= t;
    float scale = floor(p.x * LKZ_DENS) + t * LKZ_DENS;
    p.x = fmod(p.x, 1.0f / LKZ_DENS);
    if (p.x < 0.0f) {
        p.x += 1.0f / LKZ_DENS;
    }

    float erho = exp(p.x);
    float sintheta = sin(p.y);
    p = float3(
        erho * sintheta * cos(p.z),
        erho * sintheta * sin(p.z),
        erho * cos(p.y));

    float2 sdf = lkzSDF(p, scale, time);
    sdf.x *= exp(scale / LKZ_DENS);
    return sdf;
}

static float3 lkzNormal(float3 p, float depth, float time) {
    float h = max(depth * 0.0025f, 0.002f);
    const float2 k = float2(1.0f, -1.0f);
    return normalize(
        k.xyy * lkzMap(p + k.xyy * h, time).x +
        k.yyx * lkzMap(p + k.yyx * h, time).x +
        k.yxy * lkzMap(p + k.yxy * h, time).x +
        k.xxx * lkzMap(p + k.xxx * h, time).x);
}

static float lkzCalcAO(float3 p, float3 n, float time) {
    float occ = LKZ_AO_OCC;
    float sca = LKZ_AO_SCA;
    for (int i = 0; i < 5; ++i) {
        float h = 0.001f + 0.150f * float(i) / 4.0f;
        float d = lkzMap(p + h * n, time).x;
        occ += (h - d) * sca;
        sca *= 0.95f;
    }
    return lkzSmooth(0.0f, 1.0f, 1.0f - 1.5f * occ);
}

static float3 lkzShade(float3 col, float mat, float3 p, float3 n, float3 rd, float3 lp, float time) {
    float3 lidi = normalize(lp - p);
    float amoc = lkzCalcAO(p, n, time);
    float diff = max(dot(n, lidi), 0.0f);
    float spec = pow(diff, max(1.0f, LKZ_SPEC_EXP * mat));
    float refl = pow(max(0.0f, dot(lidi, reflect(rd, n))), max(1.0f, LKZ_SPEC_EXP * 3.0f * mat));
    return col * (amoc * LKZ_AMB_COL + (1.0f - mat) * diff * LKZ_SUN_COL + mat * (spec + refl) * LKZ_SUN_COL);
}

static float4 lkzPostProcess(float3 col, float2 uv) {
    float3 aces = (col * (2.51f * col + 0.03f)) / (col * (2.43f * col + 0.59f) + 0.14f);
    col = mix(col, aces, LKZ_PP_ACES);
    col = mix(col, smoothstep(float3(0.0f), float3(1.0f), col), LKZ_PP_CONT);
    col *= lkzSmooth(LKZ_PP_VIGN, -LKZ_PP_VIGN / 5.0f, dot(uv, uv));
    col = pow(max(col, 0.0f), float3(1.0f / LKZ_GAMMA));
    return float4(col, 1.0f);
}

struct LKZMarchResult {
    float distance;
    float steps;
    float material;
};

static LKZMarchResult lkzRayMarch(float3 ro, float3 rd, float time) {
    float col = 0.0f;
    float dO = mix(LKZ_MIN_DIST, LKZ_MAX_DIST / 2.0f, lkzSmooth(0.9f, 1.0f, sin(time / 24.0f) * 0.5f + 0.5f));
    int steps = 0;

    for (int i = 0; i < LKZ_MAX_STEPS; ++i) {
        steps = i;
        float3 p = ro + rd * dO;
        float2 dS = lkzMap(p, time);
        col = dS.y;
        dO += min(dS.x, length(p) / 12.0f);
        if (dO > LKZ_MAX_DIST || dS.x < max(dO * 0.0025f, 0.002f)) {
            break;
        }
    }

    LKZMarchResult result;
    result.distance = (steps == 0) ? LKZ_MIN_DIST : dO;
    result.steps = float(steps);
    result.material = col;
    return result;
}

static float2 lkzFaceUV(float3 p) {
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

fragment float4 logSphericalKIFSZoomerFragment(
    LogSphericalKIFSZoomerVertexOut in [[stage_in]],
    constant LogSphericalKIFSZoomerUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 rdLocal = normalize(surfacePos - eye);

    bool insideCube = all(abs(eye) < LKZ_BOX_HALF - 1.0e-3f);
    float2 tCube = lkzBoxIntersect(eye, rdLocal, LKZ_BOX_HALF);
    if (!insideCube && tCube.x > tCube.y) {
        discard_fragment();
    }

    float tStart = insideCube ? 0.0f : max(tCube.x, 0.0f);
    float3 localOrigin = eye + rdLocal * (tStart + 0.001f);

    float3 ro = localOrigin * LKZ_SCENE_SCALE;
    float3 rd = normalize(rdLocal);

    float3 bg = LKZ_SKY_COL;
    float3 col = bg;
    float3 p = float3(0.0f);
    LKZMarchResult rmd = lkzRayMarch(ro, rd, uniforms.time);

    if (rmd.distance <= LKZ_MIN_DIST) {
        col = lkzPalette(int(floor(rmd.material)), uniforms.time) / 8.0f;
    } else if (rmd.distance < LKZ_MAX_DIST) {
        p = ro + rd * rmd.distance;
        float3 n = lkzNormal(p, rmd.distance, uniforms.time);
        float shine = fract(rmd.material);
        col = lkzPalette(int(floor(abs(rmd.material))), uniforms.time);
        col = lkzShade(col, shine, p, n, rd, float3(0.0f), uniforms.time);
    }

    float disFac = lkzSmooth(0.0f, 1.0f, pow(rmd.distance / LKZ_MAX_DIST, 2.0f));
    col = mix(col, bg, disFac);
    col += pow(rmd.steps / float(LKZ_MAX_STEPS), 2.5f) * normalize(LKZ_AMB_COL)
        * (LKZ_GLOW_INT + ((rmd.distance < LKZ_MAX_DIST)
            ? 3.0f * lkzSmooth(0.995f, 1.0f, sin(uniforms.time / 2.0f - length(p) / 20.0f))
            : 0.0f));

    float2 faceUV = lkzFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.18f * dot(faceUV, faceUV);
    float4 outCol = lkzPostProcess(col, faceUV * 0.95f);
    outCol.rgb *= vignette;
    return float4(clamp(outCol.rgb, 0.0f, 1.0f), 1.0f);
}