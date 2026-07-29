// FlowerTestShaders.metal
// Adapted from ShaderToy "Flower Test" by inigo quilez, 2013.
// Source: https://www.shadertoy.com/view/MltSRf
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Metal adaptation notes:
// - The original shader constructed a synthetic orbit camera. This version
//   reconstructs the real per-eye world ray, intersects it with a 2 m cube
//   container, and starts marching at the visible cube surface or at the eye
//   when the viewer is inside the cube.
// - The flower SDF is evaluated beyond the cube entry plane, so the simulated
//   bloom is not clipped to the container volume.
// - Unused GLSL primitive helpers were omitted; the port keeps the operators and
//   distance logic that actually drive the original flower and shading path.

#include <metal_stdlib>
using namespace metal;

struct FlowerTestUniforms {
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

struct FlowerTestVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct FTRayHit {
    float t;
    float material;
};

static constant float3 FT_BOX_HALF = float3(1.0f);
static constant float FT_TRACE_EPSILON = 0.0015f;
static constant float FT_SCENE_SCALE = 2.3f;
static constant float FT_PRECIS = 0.06f;
static constant float FT_TMIN = 0.0f;
static constant float FT_TMAX = 20.0f;
static constant float3 FT_SKY_A = float3(0.7f, 0.9f, 1.0f);
static constant float3 FT_LIGHT_DIR = float3(-0.57207757f, 0.66742384f, -0.4767313f);

vertex FlowerTestVertexOut flowerTestVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant FlowerTestUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    FlowerTestVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 ftRotate2D(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float2 ftFaceUV(float3 p) {
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

static float2 ftBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float ftFlower(float3 p, float r, float time) {
    float q = length(p);
    p -= float3(
        sin(p.x * 15.1f),
        sin(p.y * 25.1f),
        sin(p.z * 15.0f)) * 0.01f;

    float3 n = normalize(p);
    q = length(p);

    float rho = atan2(length(float2(n.x, n.z)), n.y) * 20.0f + q * 15.01f;
    float theta = atan2(n.x, n.z) * 6.0f + p.y * 3.0f + rho * 1.50f;
    float poleMask = 1.3f - abs(dot(n, float3(0.0f, 1.0f, 0.0f)));

    return length(p) - (
        r +
        sin(theta) * 0.3f * poleMask +
        sin(rho - time * 2.0f) * 0.3f * poleMask);
}

static float2 ftMap(float3 pos, float time) {
    return float2(ftFlower(pos, 0.750f, time), 15.1f);
}

static FTRayHit ftCastRay(float3 ro, float3 rd, float time) {
    float t = FT_TMIN;
    float material = -1.0f;

    for (int i = 0; i < 400; ++i) {
        float2 res = ftMap(ro + rd * t, time);
        if (res.x < FT_PRECIS || t > FT_TMAX) {
            material = res.y;
            break;
        }
        t += res.x * 0.05f;
    }

    if (t > FT_TMAX) {
        material = -1.0f;
    }

    FTRayHit hit;
    hit.t = t;
    hit.material = material;
    return hit;
}

static float3 ftCalcNormal(float3 pos, float time) {
    float3 eps = float3(0.001f, 0.0f, 0.0f);
    float dx = ftMap(pos + eps.xyy, time).x - ftMap(pos - eps.xyy, time).x;
    float dy = ftMap(pos + eps.yxy, time).x - ftMap(pos - eps.yxy, time).x;
    float dz = ftMap(pos + eps.yyx, time).x - ftMap(pos - eps.yyx, time).x;
    return normalize(float3(dx, dy, dz));
}

static float ftCalcAO(float3 pos, float3 nor, float time) {
    float occ = 0.0f;
    float sca = 1.0f;
    for (int i = 0; i < 15; ++i) {
        float hr = 0.05f + 0.12f * float(i) / 4.0f;
        float3 aopos = nor * hr + pos;
        float dd = ftMap(aopos, time).x;
        occ += -(dd - hr) * sca;
        sca *= 0.95f;
    }
    return clamp(1.0f - 3.0f * occ, 0.0f, 1.0f);
}

static float3 ftRender(float3 ro, float3 rd, float time) {
    float3 col = FT_SKY_A + rd.y * 0.8f;
    FTRayHit hit = ftCastRay(ro, rd, time);

    if (hit.material > -0.5f) {
        float3 pos = ro + hit.t * rd;
        float3 nor = ftCalcNormal(pos, time);
        float3 ref = reflect(rd, nor);

        col = 0.50f + 0.3f * sin(
            float3(2.3f - pos.y / 2.0f, 2.15f - pos.y / 4.0f, -1.30f) * (hit.material - 1.0f));

        if (hit.material < 1.5f) {
            float checker = fmod(floor(5.0f * pos.z) + floor(5.0f * pos.x), 2.0f);
            col = 0.4f + 0.1f * checker * float3(1.0f);
        }

        float occ = ftCalcAO(pos, nor, time);
        float amb = 0.0f;
        float dif = clamp(dot(nor, FT_LIGHT_DIR), 0.0f, 1.0f);
        float bac = 0.0f;
        float dom = smoothstep(-0.1f, 0.1f, ref.y);
        float fre = 0.750f;
        float spe = 0.0f;

        float3 lin = float3(0.0f);
        lin += 1.20f * dif * float3(1.00f, 0.85f, 0.55f);
        lin += 1.20f * spe * float3(1.00f, 0.85f, 0.55f) * dif;
        lin += 0.20f * amb * float3(0.50f, 0.70f, 1.00f) * occ;
        lin += 0.30f * dom * float3(0.50f, 0.70f, 1.00f) * occ;
        lin += 0.30f * bac * float3(0.25f, 0.25f, 0.25f) * occ;
        lin += 0.40f * fre * float3(1.00f, 1.00f, 1.00f) * occ;
        col *= lin;
        col = mix(col, float3(0.8f, 0.9f, 1.0f), 1.0f - exp(-0.002f * hit.t * hit.t));
    }

    return clamp(col, 0.0f, 1.0f);
}

fragment float4 flowerTestFragment(
    FlowerTestVertexOut in [[stage_in]],
    constant FlowerTestUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float cubeScale = max(uniforms.boxScale, 1.0e-4f);
    float3 eye = (camWorld - center) / cubeScale;
    float3 hit = (in.worldPos - center) / cubeScale;
    float3 rd = normalize(hit - eye);

    bool insideBox = all(abs(eye) < FT_BOX_HALF - 1.0e-3f);
    float2 tBox = ftBoxIntersect(eye, rd, FT_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 ro = (eye + rd * (tStart + FT_TRACE_EPSILON)) * FT_SCENE_SCALE;

    float sceneTime = 15.0f + uniforms.time * 3.0f;
    float3 rotatedRo = ro;
    float3 rotatedRd = rd;
    float spin = -0.3f * sceneTime;
    rotatedRo.xz = ftRotate2D(rotatedRo.xz, spin);
    rotatedRd.xz = ftRotate2D(rotatedRd.xz, spin);
    rotatedRo.xy = ftRotate2D(rotatedRo.xy, 0.2f * sin(sceneTime * 0.17f));
    rotatedRd.xy = ftRotate2D(rotatedRd.xy, 0.2f * sin(sceneTime * 0.17f));

    float3 col = ftRender(rotatedRo, normalize(rotatedRd), sceneTime);
    col = pow(col, float3(0.4545f));

    float2 q = ftFaceUV(hit);
    float vignette = 1.0f - 0.35f * dot(q * 2.0f - 1.0f, q * 2.0f - 1.0f);
    col *= vignette;
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}