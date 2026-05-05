// HyperbolicGroupLimitSetShaders.metal
// Source adaptation: Zhao Liang, Shadertoy "Limit set of rank 4 hyperbolic Coxeter groups"
// https://www.shadertoy.com/view/NstSDs
//
// This version renders the original sphere/plane hyperbolic limit-set scene
// inside a 2 m cube container so the effect can be viewed from any direction
// in immersive space.

#include <metal_stdlib>
using namespace metal;

struct HyperbolicGroupLimitSetUniforms {
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

struct HGLSVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct HGLSMirrors {
    float3 A;
    float3 B;
    float3 D;
    float4 C;
};

vertex HGLSVertexOut hyperbolicGroupLimitSetVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant HyperbolicGroupLimitSetUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    HGLSVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static constant float HG_PI = 3.141592653f;
static constant float3 HG_BOX_HALF = float3(1.0f, 1.0f, 1.0f);
static constant float HG_SCENE_SCALE = 0.82f;
static constant int HG_MAX_TRACE_STEPS = 96;
static constant int HG_MAX_REFLECTIONS = 240;
static constant float HG_MIN_TRACE_DIST = 0.0015f;
static constant float HG_PRECISION = 0.00012f;
static constant float3 HG_PQR = float3(3.0f, 3.0f, 7.0f);
static constant float3 HG_CHECKER1 = float3(0.0f, 0.0f, 0.05f);
static constant float3 HG_CHECKER2 = float3(0.2f, 0.2f, 0.2f);
static constant float3 HG_MATERIAL = float3(1.25f, 0.34f, 0.18f);
static constant float3 HG_FUNDCOL = float3(0.3f, 1.0f, 8.0f);
static constant float HG_LIGHTENING_FACTOR = 8.0f;

static float hg_dihedral(float x) {
    return cos(HG_PI / x);
}

static float2 hg_rot2d(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float3 hg_rotateX(float3 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

static float3 hg_rotateY(float3 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static HGLSMirrors hg_setupMirrors() {
    float cp = hg_dihedral(HG_PQR.x);
    float sp = sqrt(max(1.0f - cp * cp, 0.0f));
    float cq = hg_dihedral(HG_PQR.y);
    float cr = hg_dihedral(HG_PQR.z);

    HGLSMirrors mirrors;
    mirrors.A = float3(0.0f, 0.0f, 1.0f);
    mirrors.B = float3(0.0f, sp, -cp);
    mirrors.D = float3(1.0f, 0.0f, 0.0f);

    float r = 1.0f / cr;
    float k = r * cq / sp;
    float3 cen = float3(1.0f, k, 0.0f);
    mirrors.C = float4(cen, r) / sqrt(dot(cen, cen) - r * r);
    return mirrors;
}

static float hg_distABCD(float3 p, HGLSMirrors mirrors) {
    float dA = abs(dot(p, mirrors.A));
    float dB = abs(dot(p, mirrors.B));
    float dD = abs(dot(p, mirrors.D));
    float dC = abs(length(p - mirrors.C.xyz) - mirrors.C.w);
    return min(dA, min(dB, min(dC, dD)));
}

static bool hg_tryReflectPlane(thread float3 &p, float3 n, thread int &count) {
    float k = dot(p, n);
    if (k >= 0.0f) {
        return true;
    }
    p -= 2.0f * k * n;
    count += 1;
    return false;
}

static bool hg_tryReflectSphere(thread float3 &p, float4 sphere, thread int &count, thread float &orb) {
    float3 q = p - sphere.xyz;
    float d2 = dot(q, q);
    if (d2 == 0.0f) {
        return true;
    }
    float k = (sphere.w * sphere.w) / d2;
    if (k < 1.0f) {
        return true;
    }
    p = k * q + sphere.xyz;
    count += 1;
    orb *= k;
    return false;
}

static bool hg_iterateSpherePoint(thread float3 &p, thread int &count, thread float &orb, HGLSMirrors mirrors) {
    for (int iter = 0; iter < HG_MAX_REFLECTIONS; ++iter) {
        bool inA = hg_tryReflectPlane(p, mirrors.A, count);
        bool inB = hg_tryReflectPlane(p, mirrors.B, count);
        bool inC = hg_tryReflectSphere(p, mirrors.C, count, orb);
        bool inD = hg_tryReflectPlane(p, mirrors.D, count);
        p = normalize(p);
        if (inA && inB && inC && inD) {
            return true;
        }
    }
    return false;
}

static float3 hg_chooseColor(bool found, int count, float orb) {
    float3 col;
    if (found) {
        if (count == 0) {
            return HG_FUNDCOL;
        } else if (count >= 180) {
            col = HG_MATERIAL;
        } else {
            col = ((count & 1) == 0) ? HG_CHECKER1 : HG_CHECKER2;
        }
    } else {
        col = HG_MATERIAL;
    }

    float t = float(count) / float(HG_MAX_REFLECTIONS);
    float orbMix = 1.0f - t * smoothstep(0.0f, 1.0f, log(max(orb, 1e-6f)) / 32.0f);
    col = mix(HG_MATERIAL * HG_LIGHTENING_FACTOR, col, orbMix);
    return col;
}

static float hg_sdSphere(float3 p, float radius) {
    return length(p) - radius;
}

static float hg_sdPlane(float3 p) {
    return p.y + 1.0f;
}

static float3 hg_planeToSphere(float2 p) {
    float pp = dot(p, p);
    return float3(2.0f * p.x, pp - 1.0f, 2.0f * p.y) / (1.0f + pp);
}

static float2 hg_map(float3 p) {
    float3 q = p / HG_SCENE_SCALE;
    float dSphere = hg_sdSphere(q, 1.0f) * HG_SCENE_SCALE;
    float dPlane = hg_sdPlane(q) * HG_SCENE_SCALE;
    return (dSphere < dPlane) ? float2(dSphere, 0.0f) : float2(dPlane, 1.0f);
}

static float3 hg_getNormal(float3 p) {
    const float2 e = float2(0.001f, 0.0f);
    return normalize(float3(
        hg_map(p + e.xyy).x - hg_map(p - e.xyy).x,
        hg_map(p + e.yxy).x - hg_map(p - e.yxy).x,
        hg_map(p + e.yyx).x - hg_map(p - e.yyx).x));
}

static float2 hg_raymarch(float3 ro, float3 rd, float maxDist) {
    float t = HG_MIN_TRACE_DIST;
    float2 h = float2(-1.0f, -1.0f);
    for (int i = 0; i < HG_MAX_TRACE_STEPS; ++i) {
        h = hg_map(ro + t * rd);
        if (h.x < max(0.00035f, HG_PRECISION * max(t, 1.0f))) {
            return float2(t, h.y);
        }
        if (t > maxDist) {
            break;
        }
        t += max(h.x, 0.0006f);
    }
    return float2(-1.0f, -1.0f);
}

static float hg_calcOcclusion(float3 p, float3 n) {
    float occ = 0.0f;
    float sca = 1.0f;
    for (int i = 0; i < 5; ++i) {
        float h = 0.01f + 0.12f * float(i) / 4.0f;
        float d = hg_map(p + h * n).x;
        occ += (h - d) * sca;
        sca *= 0.75f;
    }
    return clamp(1.0f - occ, 0.0f, 1.0f);
}

static float3 hg_getColor(float3 ro, float3 rd, float3 pos, float3 nor, float3 lp, float3 basecol) {
    float3 ld = lp - pos;
    float lDist = max(length(ld), 0.001f);
    ld /= lDist;
    float ao = hg_calcOcclusion(pos, nor);
    float diff = clamp(dot(nor, ld), 0.0f, 1.0f);
    float atten = 2.0f / (1.0f + lDist * lDist * 0.08f);
    float spec = pow(max(dot(reflect(-ld, nor), -rd), 0.0f), 32.0f);
    float fres = clamp(1.0f + dot(rd, nor), 0.0f, 1.0f);

    float3 col = basecol * (0.18f + 0.82f * diff);
    col += basecol * float3(1.0f, 0.8f, 0.3f) * spec * 0.75f;
    col += basecol * 0.35f * pow(fres, 5.0f);
    col *= ao * atten;
    col += basecol * clamp(0.8f + 0.2f * nor.y, 0.0f, 1.0f) * 0.25f;
    return col;
}

static float hg_boxHit(float3 ro, float3 rd, float3 halfExtents, thread float3 &nn, bool entering) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 dr = 1.0f / rd;
    float3 n = ro * dr;
    float3 k = halfExtents * abs(dr);
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

fragment float4 hyperbolicGroupLimitSetFragment(
    HGLSVertexOut in [[stage_in]],
    constant HyperbolicGroupLimitSetUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center = uniforms.objectCenter.xyz;
    float scale = uniforms.cubeScale;
    float3 eye = (camWorld - center) / scale;
    float3 rd = normalize(in.worldPos - camWorld);

    bool insideBox = all(abs(eye) < (HG_BOX_HALF - 1e-3f));
    float3 entryNormal;
    float entryT = hg_boxHit(eye, rd, HG_BOX_HALF, entryNormal, !insideBox);
    if (entryT < 0.0f) {
        discard_fragment();
    }
    float3 entryPoint = eye + rd * entryT;
    float3 faceNormal = insideBox ? -entryNormal : entryNormal;

    float2 faceCoords = entryPoint.xy * faceNormal.z / HG_BOX_HALF.xy
                      + entryPoint.yz * faceNormal.x / HG_BOX_HALF.yz
                      + entryPoint.zx * faceNormal.y / HG_BOX_HALF.zx;
    float edgeCoord = max(abs(faceCoords.x), abs(faceCoords.y));
    float edgeGlow = smoothstep(0.84f, 0.985f, edgeCoord);
    float borderMask = 1.0f - smoothstep(0.92f, 1.02f, edgeCoord);

    float3 start = entryPoint + rd * 0.0015f;
    float3 exitNormal;
    float exitT = hg_boxHit(start, rd, HG_BOX_HALF, exitNormal, false);
    if (exitT < 0.0f) {
        exitT = 4.0f;
    }

    HGLSMirrors mirrors = hg_setupMirrors();
    float rotationX = 0.74f * HG_PI;
    float rotationY = uniforms.time * 0.12f;

    float2 res = hg_raymarch(start, rd, exitT);
    float3 glassBase = mix(float3(0.02f, 0.03f, 0.05f), float3(0.08f, 0.16f, 0.22f), 1.0f - borderMask);
    glassBase += edgeGlow * float3(0.10f, 0.20f, 0.28f);

    if (res.x < 0.0f) {
        float falloff = exp(-0.35f * exitT * exitT);
        float3 missCol = mix(glassBase, float3(0.0f), 1.0f - falloff);
        return float4(clamp(missCol, 0.0f, 1.0f), 1.0f);
    }

    float t = res.x;
    float id = res.y;
    float3 pos = start + t * rd;
    float3 q = pos / HG_SCENE_SCALE;
    float3 nor = hg_getNormal(pos);
    float3 lp = float3(0.55f, 0.92f, -0.35f);
    lp.xz = hg_rot2d(lp.xz, uniforms.time * 0.18f);

    int count = 0;
    float orb = 1.0f;
    bool found;
    float edist;

    if (id < 0.5f) {
        float3 samplePoint = hg_rotateY(hg_rotateX(normalize(q), rotationX), rotationY);
        found = hg_iterateSpherePoint(samplePoint, count, orb, mirrors);
        edist = hg_distABCD(samplePoint, mirrors);
    } else {
        float3 samplePoint = hg_planeToSphere(q.xz);
        samplePoint = hg_rotateY(hg_rotateX(samplePoint, rotationX), rotationY);
        found = hg_iterateSpherePoint(samplePoint, count, orb, mirrors);
        edist = hg_distABCD(samplePoint, mirrors);
    }

    float3 basecol = hg_chooseColor(found, count, orb);
    float3 col = hg_getColor(start, rd, pos, nor, lp, basecol);
    col = mix(col, float3(0.0f), (1.0f - smoothstep(0.0f, 0.007f, edist)) * 0.85f);
    col = mix(col, glassBase * 0.9f, 1.0f - exp(-0.05f * t * t));

    float fresnel = pow(clamp(1.0f - abs(dot(faceNormal, rd)), 0.0f, 1.0f), 3.0f);
    col += fresnel * float3(0.08f, 0.12f, 0.18f);
    col = mix(col, glassBase + edgeGlow * float3(0.12f, 0.20f, 0.28f), 0.12f);
    col = mix(col, 1.0f - exp(-col), 0.35f);
    col = sqrt(max(col, 0.0f));
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}