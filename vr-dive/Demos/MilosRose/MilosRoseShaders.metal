// MilosRoseShaders.metal
// Adapted from ShaderToy "Milo's Rose".
// Source: https://www.shadertoy.com/view/XsdyWr
//
// Metal adaptation notes:
// - The original shader built a synthetic camera from fragCoord. This version
//   reconstructs the real per-eye world ray, intersects it with a 2 m cube
//   container, and starts marching at the visible cube surface or at the eye
//   when the viewer is inside the cube.
// - The rose SDF is evaluated beyond the container entry plane, so the
//   simulated petals are not clipped to the cube volume.
// - GLSL matrix constructors and implicit scalar/vector operations are expanded
//   into explicit Metal helpers.

#include <metal_stdlib>
using namespace metal;

struct MilosRoseUniforms {
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

struct MilosRoseVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float MR_PI2 = 6.28318530718f;
static constant float3 MR_BOX_HALF = float3(1.0f);
static constant float MR_STOP_THRESHOLD = 0.01f;
static constant float MR_GRAD_STEP = 0.01f;
static constant float MR_TRACE_EPSILON = 0.0015f;
static constant float MR_CLIP_FAR = 14.0f;
static constant float MR_SCENE_SCALE = 0.55f;
static constant int MR_MAX_ITERATIONS = 128;

vertex MilosRoseVertexOut milosRoseVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant MilosRoseUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    MilosRoseVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float3x3 mrRotationXY(float2 angle) {
    float2 c = cos(angle);
    float2 s = sin(angle);

    return float3x3(
        float3(c.y, 0.0f, -s.y),
        float3(s.y * s.x, c.x, c.y * s.x),
        float3(s.y * c.x, -s.x, c.y * c.x));
}

static float2 mrFaceUV(float3 p) {
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

static float2 mrBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float mrOpI(float d1, float d2) {
    return max(d1, d2);
}

static float mrOpU(float d1, float d2) {
    return min(d1, d2);
}

static float mrOpS(float d1, float d2) {
    return max(-d1, d2);
}

static float mrSdPetal(float3 p, float s) {
    p = p * float3(0.8f, 1.5f, 0.8f) + float3(0.1f, 0.0f, 0.0f);
    float2 q = float2(length(p.xz), p.y);

    float lower = length(q) - 1.0f;
    lower = mrOpS(length(q) - 0.97f, lower);
    lower = mrOpI(lower, q.y);

    float upper = length(q - float2(s, 0.0f)) + 1.0f - s;
    upper = mrOpS(upper, length(q - float2(s, 0.0f)) + 0.97f - s);
    upper = mrOpI(upper, -q.y);
    upper = mrOpI(upper, q.x - 2.0f);

    float region = length(p - float3(1.0f, 0.0f, 0.0f)) - 1.0f;
    return mrOpI(mrOpU(upper, lower), region);
}

static float mrMap(float3 p) {
    float d = 1000.0f;
    float s = 2.0f;
    float3x3 r = mrRotationXY(float2(0.1f, MR_PI2 * 0.618034f));
    r = r * float3x3(
        float3(1.08f, 0.0f, 0.0f),
        float3(0.0f, 0.995f, 0.0f),
        float3(0.0f, 0.0f, 1.08f));

    for (int i = 0; i < 21; ++i) {
        d = mrOpU(d, mrSdPetal(p, s));
        p = r * p;
        p += float3(0.0f, -0.02f, 0.0f);
        s *= 1.05f;
    }

    return d;
}

static float3 mrGradient(float3 pos) {
    float3 dx = float3(MR_GRAD_STEP, 0.0f, 0.0f);
    float3 dy = float3(0.0f, MR_GRAD_STEP, 0.0f);
    float3 dz = float3(0.0f, 0.0f, MR_GRAD_STEP);
    return normalize(float3(
        mrMap(pos + dx) - mrMap(pos - dx),
        mrMap(pos + dy) - mrMap(pos - dy),
        mrMap(pos + dz) - mrMap(pos - dz)));
}

static float mrRayMarching(float3 origin, float3 dir, float start, float end) {
    float depth = start;
    for (int i = 0; i < MR_MAX_ITERATIONS; ++i) {
        float dist = mrMap(origin + dir * depth);
        if (dist < MR_STOP_THRESHOLD) {
            return depth;
        }
        depth += dist * 0.3f;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}

static float3 mrShading(float3 v, float3 n, float3 eye) {
    const float3 lightPos = float3(20.0f, 50.0f, 20.0f);
    float3 ev = normalize(v - eye);
    float3 matColor = float3(0.65f, 0.0f, 0.0f);
    float3 vl = normalize(lightPos - v);

    float diffuse = dot(vl, n) * 0.5f + 0.5f;
    float rim = pow(1.0f - max(dot(n, -ev), 0.0f), 2.0f) * 0.15f;
    float ao = clamp(v.y * 0.5f + 0.5f, 0.0f, 1.0f);
    return (matColor * diffuse + rim) * ao;
}

fragment float4 milosRoseFragment(
    MilosRoseVertexOut in [[stage_in]],
    constant MilosRoseUniforms &uniforms [[buffer(0)]],
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

    bool insideBox = all(abs(eye) < MR_BOX_HALF - 1.0e-3f);
    float2 tBox = mrBoxIntersect(eye, rd, MR_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 marchOrigin = eye + rd * (tStart + MR_TRACE_EPSILON);

    float3 sceneEye = marchOrigin / MR_SCENE_SCALE;
    float3 sceneDir = rd;
    float3x3 sceneRot = mrRotationXY(float2(-1.0f + 0.1f * sin(uniforms.time * 0.23f), 1.0f + 0.15f * cos(uniforms.time * 0.17f)));
    sceneEye = sceneRot * sceneEye;
    sceneDir = sceneRot * sceneDir;

    float depth = mrRayMarching(sceneEye, sceneDir, 0.0f, MR_CLIP_FAR);
    float3 pos = sceneEye + sceneDir * depth;

    float2 q = mrFaceUV(hit);
    float radial = 1.2f - length(q - 0.5f);

    float3 color;
    if (depth >= MR_CLIP_FAR) {
        float glow = 0.5f + 0.5f * cos(3.0f * atan2(sceneDir.y, sceneDir.x) + uniforms.time * 0.35f);
        color = mix(float3(0.08f, 0.0f, 0.06f), float3(0.2f, 0.0f, 0.1f), glow);
    } else {
        float3 normal = mrGradient(pos);
        color = mrShading(pos, normal, sceneEye);
    }

    color *= radial;
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}