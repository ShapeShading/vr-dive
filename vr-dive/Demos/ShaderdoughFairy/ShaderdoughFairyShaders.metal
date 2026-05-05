// ShaderdoughFairyShaders.metal
// "Shaderdough fairy" — cube-portal adaptation of Shadertoy "4lGyW1"
// Original: https://www.shadertoy.com/view/4lGyW1
//
// Metal adaptation notes:
// - The original GLSL builds a synthetic screen-space camera and marches a
//   glowing twisted icosahedral field from that camera.
// - This version uses the actual per-eye world ray instead. When the viewer is
//   outside the 2 m cube, marching starts at the visible cube surface. When the
//   viewer is inside the cube, marching starts at the eye position.
// - The scene itself is fixed in scene space around the cube centre and is not
//   clipped by the cube bounds, so head motion reveals true stereo parallax
//   instead of a 2D image attached to the cube wall.

#include <metal_stdlib>
using namespace metal;

struct ShaderdoughFairyUniforms {
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

struct ShaderdoughFairyVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct ShaderdoughFairyIcosahedronData {
    float3 nc;
    float3 pca;
};

struct ShaderdoughFairyModel {
    float dist;
    float3 colour;
    float id;
};

static constant float SF_PI = 3.14159265359f;
static constant float SF_PHI = 1.618033988749895f;
static constant float SF_MAX_TRACE_DISTANCE = 6.0f;
static constant float SF_INTERSECTION_PRECISION = 0.001f;
static constant float SF_FUDGE_FACTOR = 0.2f;

vertex ShaderdoughFairyVertexOut shaderdoughFairyVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant ShaderdoughFairyUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    ShaderdoughFairyVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float3x3 sfRotationMatrix(float3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0f - c;
    return float3x3(
        float3(oc * axis.x * axis.x + c,
               oc * axis.x * axis.y - axis.z * s,
               oc * axis.z * axis.x + axis.y * s),
        float3(oc * axis.x * axis.y + axis.z * s,
               oc * axis.y * axis.y + c,
               oc * axis.y * axis.z - axis.x * s),
        float3(oc * axis.z * axis.x - axis.y * s,
               oc * axis.y * axis.z + axis.x * s,
               oc * axis.z * axis.z + c)
    );
}

static float3 sfPalette(float t, float3 a, float3 b, float3 c, float3 d) {
    return a + b * cos(6.28318f * (c * t + d));
}

static float3 sfSpectrum(float n) {
    return sfPalette(
        n,
        float3(0.5f, 0.5f, 0.5f),
        float3(0.5f, 0.5f, 0.5f),
        float3(1.0f, 1.0f, 1.0f),
        float3(0.0f, 0.33f, 0.67f));
}

static float2 sfRotate2D(float2 p, float a) {
    return cos(a) * p + sin(a) * float2(p.y, -p.x);
}

static float sfPReflect(thread float3 &p, float3 planeNormal, float offset) {
    float localT = dot(p, planeNormal) + offset;
    if (localT < 0.0f) {
        p = p - (2.0f * localT) * planeNormal;
    }
    return sign(localT);
}

static float sfPModPolar(thread float2 &p, float repetitions) {
    float angle = 2.0f * SF_PI / repetitions;
    float a = atan2(p.y, p.x) + angle / 2.0f;
    float r = length(p);
    float c = floor(a / angle);
    a = fmod(a, angle) - angle / 2.0f;
    p = float2(cos(a), sin(a)) * r;
    if (abs(c) >= (repetitions / 2.0f)) {
        c = abs(c);
    }
    return c;
}

static ShaderdoughFairyIcosahedronData sfInitIcosahedron() {
    float cospin = cos(SF_PI / 5.0f);
    float scospin = sqrt(0.75f - cospin * cospin);
    ShaderdoughFairyIcosahedronData data;
    data.nc = float3(-0.5f, -cospin, scospin);
    data.pca = normalize(float3(0.0f, scospin, cospin));
    return data;
}

static void sfPModIcosahedron(thread float3 &p, ShaderdoughFairyIcosahedronData data) {
    p = abs(p);
    sfPReflect(p, data.nc, 0.0f);
    p.xy = abs(p.xy);
    sfPReflect(p, data.nc, 0.0f);
    p.xy = abs(p.xy);
    sfPReflect(p, data.nc, 0.0f);
}

static float sfSplitPlane(float a, float b, float3 p, float3 plane) {
    float split = max(sign(dot(p, plane)), 0.0f);
    return mix(a, b, split);
}

static float sfIcosahedronIndex(float3 p) {
    float3 sp = sign(p);
    float x = sp.x * 0.5f + 0.5f;
    float y = sp.y * 0.5f + 0.5f;
    float z = sp.z * 0.5f + 0.5f;

    float3 plane = float3(-1.0f - SF_PHI, -1.0f, SF_PHI);
    float idx = x + y * 2.0f + z * 4.0f;
    idx = sfSplitPlane(idx, 8.0f + y + z * 2.0f, p, plane * sp);
    idx = sfSplitPlane(idx, 12.0f + x + y * 2.0f, p, plane.yzx * sp);
    idx = sfSplitPlane(idx, 16.0f + z + x * 2.0f, p, plane.zxy * sp);
    return idx;
}

static float3 sfIcosahedronVertex(float3 p) {
    float3 v = float3(SF_PHI, 1.0f, 0.0f);
    float3 sp = sign(p);
    float3 v1 = v.xyz * sp;
    float3 v2 = v.yzx * sp;
    float3 v3 = v.zxy * sp;

    float3 plane = float3(1.0f, SF_PHI, -SF_PHI - 1.0f);
    float split = max(sign(dot(p, plane.xyz * sp)), 0.0f);
    float3 result = mix(v2, v1, split);
    plane = mix(plane.yzx * -sp, plane.zxy * sp, split);
    split = max(sign(dot(p, plane)), 0.0f);
    result = mix(result, v3, split);
    return normalize(result);
}

static float4 sfIcosahedronAxisDistance(float3 p, ShaderdoughFairyIcosahedronData data) {
    float3 iv = sfIcosahedronVertex(p);
    float3 originalIv = iv;

    float3 pn = normalize(p);
    sfPModIcosahedron(pn, data);
    sfPModIcosahedron(iv, data);

    float boundaryDist = dot(pn, float3(1.0f, 0.0f, 0.0f));
    float boundaryMax = dot(iv, float3(1.0f, 0.0f, 0.0f));
    boundaryDist /= boundaryMax;

    float roundDist = length(iv - pn);
    float roundMax = length(iv - float3(0.0f, 0.0f, 1.0f));
    roundDist /= roundMax;
    roundDist = -roundDist + 1.0f;

    float blend = 1.0f - boundaryDist;
    blend = pow(blend, 6.0f);
    float dist = mix(roundDist, boundaryDist, blend);

    return float4(originalIv, dist);
}

static void sfPTwistIcosahedron(thread float3 &p, float amount, ShaderdoughFairyIcosahedronData data) {
    float4 a = sfIcosahedronAxisDistance(p, data);
    float3 axis = a.xyz;
    float dist = a.w;
    float3x3 m = sfRotationMatrix(axis, dist * amount);
    p = p * m;
}

static ShaderdoughFairyModel sfInflatedIcosahedron(float3 p, float localTime, ShaderdoughFairyIcosahedronData data) {
    float idx = sfIcosahedronIndex(p);
    float d = dot(p, data.pca) - 0.9f;
    d = mix(d, length(p) - 0.9f, 1.0f);

    if (idx == 3.0f) {
        idx = 2.0f;
    }
    idx /= 10.0f;
    idx = fract(idx + localTime);

    ShaderdoughFairyModel result;
    result.dist = d * 0.6f;
    result.colour = sfSpectrum(idx);
    result.id = 1.0f;
    return result;
}

static ShaderdoughFairyModel sfModel(float3 p, float localTime, ShaderdoughFairyIcosahedronData data) {
    float rate = SF_PI / 6.0f;
    float a = atan2(1.0f, SF_PHI + 1.0f);

    p.yz = sfRotate2D(p.yz, a);
    p.yx = sfRotate2D(p.yx, localTime * 2.1f + rate);
    p.yz = sfRotate2D(p.yz, a);

    float3 twistCenter = float3(0.7f, 0.0f, 0.0f);
    twistCenter.yx = sfRotate2D(twistCenter.yx, localTime * 2.1f + rate);
    twistCenter.yz = sfRotate2D(twistCenter.yz, a);

    p += twistCenter;
    sfPTwistIcosahedron(p, 10.5f, data);
    p -= twistCenter;

    p.yz = sfRotate2D(p.yz, -a);
    p.xy = sfRotate2D(p.xy, -SF_PI * 0.5f);
    float2 polarXY = p.xy;
    sfPModPolar(polarXY, 3.0f);
    p.xy = polarXY;
    p.xy = sfRotate2D(p.xy, -SF_PI * 0.5f);
    p.yz = sfRotate2D(p.yz, -a);

    return sfInflatedIcosahedron(p, localTime, data);
}

static ShaderdoughFairyModel sfMap(float3 p, float localTime, ShaderdoughFairyIcosahedronData data) {
    return sfModel(p, localTime, data);
}

fragment float4 shaderdoughFairyFragment(
    ShaderdoughFairyVertexOut in [[stage_in]],
    constant ShaderdoughFairyUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];
    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float sceneScale = max(uniforms.cubeScale, 1.0e-4f);
    float3 roScene = (camWorld - center) / sceneScale;
    float3 rdScene = normalize(in.worldPos - camWorld);
    float3 surfaceScene = (in.worldPos - center) / sceneScale;

    bool insideBox = all(abs(roScene) < float3(0.999f));
    float3 marchOrigin = insideBox ? (roScene + rdScene * 0.002f) : (surfaceScene + rdScene * 0.002f);

    // Keep the underlying fairy shape fixed at the scene origin. Only the
    // model's own authored deformation/rotation uses time, so the content stays
    // in world space instead of following the viewer like a 2D image.
    float localTime = (uniforms.time - 0.25f) * 0.5f;
    ShaderdoughFairyIcosahedronData ico = sfInitIcosahedron();
    float fairyScale = 2.0f;

    float3 color = pow(float3(0.15f, 0.0f, 0.2f), float3(2.2f));
    float travel = 0.0f;
    int iter = int(20.0f / SF_FUDGE_FACTOR);

    for (int i = 0; i < iter; ++i) {
        if (travel > SF_MAX_TRACE_DISTANCE) {
            break;
        }

        float3 samplePoint = (marchOrigin + rdScene * travel) * fairyScale;
        ShaderdoughFairyModel sample = sfMap(samplePoint, localTime, ico);
        float h = abs(sample.dist) / fairyScale;
        travel += max(SF_INTERSECTION_PRECISION, h * SF_FUDGE_FACTOR);
        color += sample.colour * pow(max(0.0f, (0.02f - h)) * 19.5f, 10.0f) * 150.0f;
        color += sample.colour * 0.001f * SF_FUDGE_FACTOR;
    }

    color = pow(color, float3(1.0f / 1.8f)) * 1.5f;
    color = pow(color, float3(1.5f));
    color *= 3.5f;
    return float4(color, 1.0f);
}
