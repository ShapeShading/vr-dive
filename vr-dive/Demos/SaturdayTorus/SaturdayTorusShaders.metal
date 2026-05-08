// SaturdayTorusShaders.metal
// Adapted from ShaderToy "Saturday Torus".
// Source: https://www.shadertoy.com/view/fd33zn
// Shader header declares: License CC0: Saturday Torus.
// Embedded helpers retained from the original source:
// - rayTorus / torusNormal: MIT, Inigo Quilez
// - tanh_approx: original source marks author/license as unknown
//
// Metal adaptation notes:
// - The original ShaderToy used screen-space fragCoord and a fixed camera.
//   This version reconstructs the real per-eye world ray and starts it at the
//   visible 2 m cube surface, or at the eye when the camera is inside.
// - The torus itself is not clipped to the container bounds after entry.
// - GLSL macros and mat2 rotation helpers are expanded into explicit Metal
//   constants and functions.

#include <metal_stdlib>
using namespace metal;

struct SaturdayTorusUniforms {
    float  time;
    uint   viewCount;
    float  boxScale;
    float  padding;
    float4 objectCenter;
};

struct SaturdayTorusMeshVertex {
    float3 position;
    float3 normal;
};

struct SaturdayTorusVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float ST_PI = 3.141592654f;
static constant float ST_TAU = 6.28318530718f;
static constant float3 ST_BOX_HALF = float3(1.0f);
static constant float2 ST_TORUS = 0.55f * float2(1.0f, 0.75f);

vertex SaturdayTorusVertexOut saturdayTorusVertex(
    ushort amplificationID [[amplification_id]],
    const device SaturdayTorusMeshVertex *vertices [[buffer(0)]],
    constant SaturdayTorusUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    SaturdayTorusMeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    SaturdayTorusVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 stRotate(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float3 stRotateScene(float3 p, float time) {
    p.xy = stRotate(p.xy, 0.35f * sin(time * 0.23f));
    p.xz = stRotate(p.xz, 0.55f + time * 0.08f);
    p.yz = stRotate(p.yz, 0.65f);
    return p;
}

static float2 stBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float rayTorus(float3 ro, float3 rd, float2 tor) {
    float po = 1.0f;

    float Ra2 = tor.x * tor.x;
    float ra2 = tor.y * tor.y;

    float m = dot(ro, ro);
    float n = dot(ro, rd);

    float h = n * n - m + (tor.x + tor.y) * (tor.x + tor.y);
    if (h < 0.0f) {
        return -1.0f;
    }

    float k = (m - ra2 - Ra2) * 0.5f;
    float k3 = n;
    float k2 = n * n + Ra2 * rd.z * rd.z + k;
    float k1 = k * n + Ra2 * ro.z * rd.z;
    float k0 = k * k + Ra2 * ro.z * ro.z - Ra2 * ra2;

    if (abs(k3 * (k3 * k3 - k2) + k1) < 0.01f) {
        po = -1.0f;
        float tmp = k1;
        k1 = k3;
        k3 = tmp;
        k0 = 1.0f / max(k0, 1.0e-6f);
        k1 *= k0;
        k2 *= k0;
        k3 *= k0;
    }

    float c2 = 2.0f * k2 - 3.0f * k3 * k3;
    float c1 = k3 * (k3 * k3 - k2) + k1;
    float c0 = k3 * (k3 * (-3.0f * k3 * k3 + 4.0f * k2) - 8.0f * k1) + 4.0f * k0;

    c2 /= 3.0f;
    c1 *= 2.0f;
    c0 /= 3.0f;

    float Q = c2 * c2 + c0;
    float R = 3.0f * c0 * c2 - c2 * c2 * c2 - c1 * c1;

    h = R * R - Q * Q * Q;
    float z = 0.0f;
    if (h < 0.0f) {
        float sQ = sqrt(max(Q, 0.0f));
        float denom = max(sQ * Q, 1.0e-6f);
        z = 2.0f * sQ * cos(acos(clamp(R / denom, -1.0f, 1.0f)) / 3.0f);
    } else {
        float sQ = pow(sqrt(max(h, 0.0f)) + abs(R), 1.0f / 3.0f);
        float safeSQ = max(sQ, 1.0e-6f);
        z = copysign(abs(safeSQ + Q / safeSQ), R);
    }
    z = c2 - z;

    float d1 = z - 3.0f * c2;
    float d2 = z * z - 3.0f * c0;
    if (abs(d1) < 1.0e-4f) {
        if (d2 < 0.0f) {
            return -1.0f;
        }
        d2 = sqrt(d2);
    } else {
        if (d1 < 0.0f) {
            return -1.0f;
        }
        d1 = sqrt(d1 * 0.5f);
        d2 = c1 / d1;
    }

    float result = 1.0e20f;

    h = d1 * d1 - z + d2;
    if (h > 0.0f) {
        h = sqrt(h);
        float t1 = -d1 - h - k3;
        t1 = (po < 0.0f) ? 2.0f / t1 : t1;
        float t2 = -d1 + h - k3;
        t2 = (po < 0.0f) ? 2.0f / t2 : t2;
        if (t1 > 0.0f) {
            result = t1;
        }
        if (t2 > 0.0f) {
            result = min(result, t2);
        }
    }

    h = d1 * d1 - z - d2;
    if (h > 0.0f) {
        h = sqrt(h);
        float t1 = d1 - h - k3;
        t1 = (po < 0.0f) ? 2.0f / t1 : t1;
        float t2 = d1 + h - k3;
        t2 = (po < 0.0f) ? 2.0f / t2 : t2;
        if (t1 > 0.0f) {
            result = min(result, t1);
        }
        if (t2 > 0.0f) {
            result = min(result, t2);
        }
    }

    return result < 1.0e19f ? result : -1.0f;
}

static float3 torusNormal(float3 pos, float2 tor) {
    float sq = dot(pos, pos) - tor.y * tor.y;
    return normalize(pos * (sq - tor.x * tor.x * float3(1.0f, 1.0f, -1.0f)));
}

static float tanhApprox(float x) {
    float x2 = x * x;
    return clamp(x * (27.0f + x2) / (27.0f + 9.0f * x2), -1.0f, 1.0f);
}

static float2 cubeFaceUV(float3 p) {
    float3 ap = abs(p);
    float2 uv = float2(0.5f);
    if (ap.x >= ap.y && ap.x >= ap.z) {
        uv = p.zy;
    } else if (ap.y >= ap.z) {
        uv = p.xz;
    } else {
        uv = p.xy;
    }
    return clamp(uv * 0.5f + 0.5f, 0.0f, 1.0f);
}

static float3 postProcess(float3 col, float2 q) {
    col = clamp(col, 0.0f, 1.0f);
    col = pow(col, 1.0f / float3(2.2f));
    col = col * 0.6f + 0.4f * col * col * (3.0f - 2.0f * col);
    col = mix(col, float3(dot(col, float3(0.33f))), -0.4f);
    float vignette = 0.5f + 0.5f * pow(max(19.0f * q.x * q.y * (1.0f - q.x) * (1.0f - q.y), 0.0f), 0.7f);
    return col * vignette;
}

static float3 saturdayTorusColor(float3 ro, float3 rd, float2 q, float time) {
    float td = rayTorus(ro, rd, ST_TORUS);

    float3 background = mix(
        float3(0.02f, 0.02f, 0.025f),
        float3(0.14f, 0.14f, 0.16f),
        0.25f + 0.75f * clamp(0.5f + 0.5f * rd.z, 0.0f, 1.0f));
    background *= 0.7f + 0.3f * (0.5f + 0.5f * cos(ST_TAU * q.x + time * 0.3f));

    if (td <= 0.0f) {
        return background;
    }

    float3 tpos = ro + rd * td;
    float3 tnor = -torusNormal(tpos, ST_TORUS);
    float3 tref = reflect(rd, tnor);

    float3 lp1 = ro;
    lp1.xy = stRotate(lp1.xy, 0.85f + 0.2f * sin(time * 0.5f));
    lp1.xz = stRotate(lp1.xz, -0.5f + 0.15f * cos(time * 0.4f));

    float3 ldif1 = lp1 - tpos;
    float ldd1 = max(dot(ldif1, ldif1), 1.0e-4f);
    float ldl1 = sqrt(ldd1);
    float3 ld1 = ldif1 / ldl1;
    float3 sro = tpos + 0.05f * tnor;
    float sd = rayTorus(sro, ld1, ST_TORUS);

    float dif1 = max(dot(tnor, ld1), 0.0f);
    float spe1 = pow(max(dot(tref, ld1), 0.0f), 10.0f);
    float r = length(tpos.xy);
    float denom = max(r + 0.5f * abs(tpos.z), 1.0e-3f);
    float a = atan2(tpos.y, tpos.x) - ST_PI * tpos.z / denom - ST_TAU * time / 45.0f;
    float s = mix(0.05f, 0.5f, tanhApprox(2.0f * abs(td - 0.75f)));
    float3 bcol0 = float3(0.3f);
    float3 bcol1 = float3(0.025f);
    float stripe = smoothstep(-s, s, sin(9.0f * a));
    float3 tcol = mix(bcol0, bcol1, stripe);

    float fresnel = sqrt(abs(dot(rd, tnor)));
    float3 col = background * 0.2f;
    col += tcol * mix(0.2f, 1.0f, dif1 / ldd1) + 0.25f * spe1;
    col *= fresnel;

    if (sd > 0.0f && sd < ldl1) {
        float3 spos = sro + ld1 * sd;
        float3 snor = -torusNormal(spos, ST_TORUS);
        col *= mix(1.0f, 0.0f, pow(abs(dot(ld1, snor)), 3.0f * tanhApprox(sd)));
    }

    return max(col, 0.0f);
}

fragment float4 saturdayTorusFragment(
    SaturdayTorusVertexOut in [[stage_in]],
    constant SaturdayTorusUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < ST_BOX_HALF - 1.0e-3f);
    float2 tBox = stBoxIntersect(eye, rd, ST_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 ro = eye + rd * (tStart + 1.0e-3f);
    float2 q = cubeFaceUV(hit);

    float3 sceneRo = stRotateScene(ro, uniforms.time);
    float3 sceneRd = normalize(stRotateScene(rd, uniforms.time));

    float3 col = saturdayTorusColor(sceneRo, sceneRd, q, uniforms.time);
    col = postProcess(col, q);
    return float4(col, 1.0f);
}