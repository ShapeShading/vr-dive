// PoincareBallHoneycombShaders.metal
// Exploratory proof-of-concept for a Poincare-ball honeycomb.
//
// Source note:
// - This shader is built from the Coxeter mirror construction already used in
//   HyperbolicGroupLimitSetShaders.metal, but adapted here to fold interior
//   points of the ball instead of only boundary samples.
// - The mirror set uses the hyperbolic tetrahedral Coxeter data [3,3,7] as a
//   compact demonstrator for 3D hyperbolic tessellation structure.

#include <metal_stdlib>
using namespace metal;

struct PoincareBallHoneycombUniforms {
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

struct PoincareBallHoneycombVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct PBHMirrors {
    float3 A;
    float3 B;
    float3 D;
    float4 C;
};

struct PBHFoldResult {
    float3 point;
    float4 distances;
    float  orb;
    int    count;
    uint   converged;
};

static constant float PBH_PI = 3.141592653f;
static constant float PBH_TAU = 6.283185307f;
static constant float3 PBH_BOX_HALF = float3(1.0f);
static constant float PBH_MODEL_SCALE = 0.9f;
static constant float3 PBH_PQR = float3(3.0f, 3.0f, 7.0f);
static constant int PBH_MAX_FOLDS = 48;
static constant int PBH_VOLUME_STEPS = 72;

vertex PoincareBallHoneycombVertexOut poincareBallHoneycombVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant PoincareBallHoneycombUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    PoincareBallHoneycombVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float pbhDihedral(float x) {
    return cos(PBH_PI / x);
}

static float3 pbhRotateX(float3 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

static float3 pbhRotateY(float3 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static float3 pbhRotateScene(float3 p, float time) {
    p = pbhRotateX(p, 0.72f);
    p = pbhRotateY(p, time * 0.08f + 0.22f);
    return p;
}

static PBHMirrors pbhSetupMirrors() {
    float cp = pbhDihedral(PBH_PQR.x);
    float sp = sqrt(max(1.0f - cp * cp, 0.0f));
    float cq = pbhDihedral(PBH_PQR.y);
    float cr = pbhDihedral(PBH_PQR.z);

    PBHMirrors mirrors;
    mirrors.A = float3(0.0f, 0.0f, 1.0f);
    mirrors.B = float3(0.0f, sp, -cp);
    mirrors.D = float3(1.0f, 0.0f, 0.0f);

    float r = 1.0f / cr;
    float k = r * cq / sp;
    float3 cen = float3(1.0f, k, 0.0f);
    mirrors.C = float4(cen, r) / sqrt(dot(cen, cen) - r * r);
    return mirrors;
}

static float2 pbhBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 pbhSphereIntersect(float3 ro, float3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) {
        return float2(-1.0f, -1.0f);
    }
    h = sqrt(h);
    return float2(-b - h, -b + h);
}

static bool pbhReflectPlane(thread float3 &p, float3 n, thread int &count) {
    float k = dot(p, n);
    if (k >= 0.0f) {
        return false;
    }
    p -= 2.0f * k * n;
    count += 1;
    return true;
}

static bool pbhReflectSphere(thread float3 &p, float4 sphere, thread int &count, thread float &orb) {
    float3 q = p - sphere.xyz;
    float d2 = dot(q, q);
    float r2 = sphere.w * sphere.w;
    if (d2 >= r2 || d2 <= 1.0e-6f) {
        return false;
    }
    float k = r2 / d2;
    p = sphere.xyz + q * k;
    orb *= k;
    count += 1;
    return true;
}

static float4 pbhMirrorDistances(float3 p, PBHMirrors mirrors) {
    float dA = abs(dot(p, mirrors.A));
    float dB = abs(dot(p, mirrors.B));
    float dD = abs(dot(p, mirrors.D));
    float dC = abs(length(p - mirrors.C.xyz) - mirrors.C.w);
    return float4(dA, dB, dC, dD);
}

static float2 pbhTwoSmallest(float4 d) {
    float4 s = d;
    if (s.x > s.y) { float t = s.x; s.x = s.y; s.y = t; }
    if (s.z > s.w) { float t = s.z; s.z = s.w; s.w = t; }
    if (s.x > s.z) { float t = s.x; s.x = s.z; s.z = t; }
    if (s.y > s.w) { float t = s.y; s.y = s.w; s.w = t; }
    if (s.y > s.z) { float t = s.y; s.y = s.z; s.z = t; }
    return s.xy;
}

static PBHFoldResult pbhFoldPoint(float3 p, PBHMirrors mirrors) {
    PBHFoldResult result;
    result.point = p;
    result.orb = 1.0f;
    result.count = 0;
    result.converged = 0u;

    for (int iter = 0; iter < PBH_MAX_FOLDS; ++iter) {
        bool changed = false;
        changed = pbhReflectPlane(result.point, mirrors.A, result.count) || changed;
        changed = pbhReflectPlane(result.point, mirrors.B, result.count) || changed;
        changed = pbhReflectSphere(result.point, mirrors.C, result.count, result.orb) || changed;
        changed = pbhReflectPlane(result.point, mirrors.D, result.count) || changed;

        float len2 = dot(result.point, result.point);
        if (len2 > 0.9998f * 0.9998f) {
            result.point *= 0.9998f * rsqrt(max(len2, 1.0e-6f));
        }

        if (!changed) {
            result.converged = 1u;
            break;
        }
    }

    result.distances = pbhMirrorDistances(result.point, mirrors);
    return result;
}

static float3 pbhPalette(float t) {
    return 0.55f + 0.45f * cos(PBH_TAU * (t + float3(0.0f, 0.13f, 0.31f)));
}

static float3 pbhBackground(float3 rd) {
    float sky = clamp(rd.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(rd.y), 0.0f), 5.0f);
    float3 bg = mix(float3(0.01f, 0.014f, 0.024f), float3(0.045f, 0.065f, 0.11f), sky);
    bg += horizon * float3(0.035f, 0.055f, 0.095f);
    return bg;
}

fragment float4 poincareBallHoneycombFragment(
    PoincareBallHoneycombVertexOut in [[stage_in]],
    constant PoincareBallHoneycombUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center = uniforms.objectCenter.xyz;
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 rdLocal = normalize(surfacePos - eye);

    bool insideCube = all(abs(eye) < PBH_BOX_HALF - 1.0e-3f);
    float2 tCube = pbhBoxIntersect(eye, rdLocal, PBH_BOX_HALF);
    if (!insideCube && tCube.x > tCube.y) {
        discard_fragment();
    }

    float tStart = insideCube ? 0.0f : max(tCube.x, 0.0f);
    float3 localOrigin = eye + rdLocal * (tStart + 0.001f);

    float3 ro = pbhRotateScene(localOrigin / PBH_MODEL_SCALE, uniforms.time);
    float3 rd = normalize(pbhRotateScene(rdLocal, uniforms.time));

    float2 tBall = pbhSphereIntersect(ro, rd, 1.0f);
    float3 bg = pbhBackground(rd);
    bool insideBall = dot(ro, ro) < 0.999f;
    if ((!insideBall && tBall.y <= 0.0f) || tBall.x > tBall.y) {
        return float4(bg, 1.0f);
    }

    float tBallStart = insideBall ? 0.0f : max(tBall.x, 0.0f);
    float tBallEnd = tBall.y;
    if (tBallEnd <= tBallStart) {
        return float4(bg, 1.0f);
    }

    float shellOverlay = 0.0f;
    float3 shellColor = float3(0.0f);
    if (!insideBall) {
        float3 shellPos = ro + rd * tBallStart;
        float3 shellNormal = normalize(shellPos);
        float fresnel = pow(max(1.0f - dot(-rd, shellNormal), 0.0f), 4.0f);
        shellColor = mix(float3(0.05f, 0.10f, 0.18f), float3(0.20f, 0.38f, 0.62f), 0.35f + 0.65f * fresnel);
        shellOverlay = 0.12f + 0.18f * fresnel;
    }

    PBHMirrors mirrors = pbhSetupMirrors();
    float stepSize = (tBallEnd - tBallStart) / float(PBH_VOLUME_STEPS);
    float transmittance = 1.0f;
    float3 volumeColor = float3(0.0f);

    for (int stepIndex = 0; stepIndex < PBH_VOLUME_STEPS; ++stepIndex) {
        float t = tBallStart + (float(stepIndex) + 0.5f) * stepSize;
        float3 p = ro + rd * t;

        PBHFoldResult folded = pbhFoldPoint(p, mirrors);
        float2 nearest = pbhTwoSmallest(folded.distances);
        float faceField = nearest.x;
        float edgeField = length(nearest);
        float interiorDepth = clamp(1.0f - length(p), 0.0f, 1.0f);
        float boundaryFade = smoothstep(0.018f, 0.13f, interiorDepth);
        float faceDensity = exp(-52.0f * faceField);
        float edgeDensity = exp(-96.0f * edgeField);
        float shellDensity = exp(-120.0f * abs(length(p) - 1.0f));

        float hueParam = 0.025f * float(folded.count) + 0.045f * log(max(folded.orb, 1.0e-4f));
        float3 tint = mix(float3(0.08f, 0.26f, 0.78f), pbhPalette(hueParam), 0.78f);
        tint = mix(tint, float3(1.0f, 0.93f, 0.82f), clamp(edgeDensity * 0.35f, 0.0f, 1.0f));

        float density = boundaryFade * (0.020f * faceDensity + 0.065f * edgeDensity);
        float shellAmount = 0.018f * shellDensity;

        volumeColor += transmittance * tint * density * stepSize * 15.0f;
        volumeColor += transmittance * float3(0.09f, 0.18f, 0.30f) * shellAmount * stepSize * 8.0f;
        transmittance *= exp(-(density + shellAmount) * stepSize * 11.0f);
        if (transmittance < 0.02f) {
            break;
        }
    }

    float3 col = bg * transmittance + volumeColor;
    col += shellColor * shellOverlay;
    col = 1.0f - exp(-1.45f * max(col, 0.0f));
    col = sqrt(max(col, 0.0f));
    return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}