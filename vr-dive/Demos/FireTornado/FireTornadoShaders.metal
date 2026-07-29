// FireTornadoShaders.metal
// "Fire Tornado" — cube-portal adaptation of Shadertoy "wfSBzV"
// Original: https://www.shadertoy.com/view/wfSBzV
//
// Metal adaptation notes:
// - The original shader raymarches a fire volume from a fixed screen-space
//   camera at z = -10.
// - This version replaces that camera with the real per-eye world ray. When the
//   viewer is outside the cube, marching starts at the visible cube surface.
//   When the viewer is inside the cube, marching starts at the eye.
// - The fire volume is fixed in scene space around the cube centre and is not
//   clipped to the cube bounds, so user motion produces real stereo and spatial
//   parallax instead of a 2D image on the container wall.

#include <metal_stdlib>
using namespace metal;

struct FireTornadoUniforms {
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

struct FireTornadoVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct FireTornadoSample {
    float dist;
    float3 glow;
};

static constant float FT_EPSILON = 1.0e-6f;
static constant float3 FT_BG_LOW = float3(0.02f, 0.01f, 0.005f);
static constant float3 FT_BG_HIGH = float3(0.12f, 0.04f, 0.01f);

vertex FireTornadoVertexOut fireTornadoVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant FireTornadoUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    FireTornadoVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 ftRotate(float2 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float ftFbm(float3 p, float time) {
    float amp = 1.0f;
    float fre = 1.0f;
    float n = 0.0f;
    for (int i = 0; i < 4; ++i) {
        n += abs(dot(cos(p * fre), float3(0.1f, 0.2f, 0.3f))) * amp;
        amp *= 0.9f;
        fre *= 1.3f;
        p.xz = ftRotate(p.xz, p.y * 0.1f + time * 0.3f);
        p.y -= time * 4.0f;
    }
    return n;
}

static float ftSdBox(float3 p, float3 b) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

static float ftBoxHit(float3 ro, float3 rd, float3 r, thread float3 &nn, bool entering) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 dr = 1.0f / rd;
    float3 n = ro * dr;
    float3 k = r * abs(dr);
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

static FireTornadoSample ftFireBall(float3 p, float time) {
    p.y += 1.0f;
    float3 q = p;

    float h = 5.0f;
    float range = smoothstep(-h, h, p.y);
    float w = range * 4.0f + 1.0f;
    float thick = range * 4.0f + 1.0f;
    q.xz = ftRotate(q.xz, q.y - time * 2.0f);

    float d = ftSdBox(q, float3(w, h, thick));
    float d1 = ftSdBox(q - float3(0.0f, 1.0f, 0.0f), float3(w, h, thick) * float3(0.7f, 2.0f, 0.7f));
    d = max(d, -d1);
    d += ftFbm(p * 3.0f, time) * 0.5f;
    d = abs(d) * 0.1f + 0.01f;

    float3 phase = float3(3.0f, 2.0f, 1.0f) + (p.y + p.z) * 0.5f - time * 2.0f;
    float3 c = sin(phase) * 0.5f + 0.5f;

    FireTornadoSample result;
    result.dist = d;
    result.glow = pow(1.3f / max(d, 1.0e-4f), 2.0f) * c;
    return result;
}

static float3 ftBackground(float3 rd) {
    float t = clamp(rd.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(1.0f - abs(rd.z), 3.0f);
    return mix(FT_BG_LOW, FT_BG_HIGH, t) + horizon * float3(0.08f, 0.02f, 0.0f);
}

fragment float4 fireTornadoFragment(
    FireTornadoVertexOut in [[stage_in]],
    constant FireTornadoUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];
    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float sceneScale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / sceneScale;
    float3 rd = normalize(in.worldPos - camWorld);

    bool insideBox = all(abs(eye) < float3(0.999f));
    float3 faceNormal;
    float entryT = insideBox ? 0.0f : ftBoxHit(eye, rd, float3(1.0f), faceNormal, true);
    if (!insideBox && entryT < 0.0f) {
        discard_fragment();
    }

    float3 marchOrigin = insideBox ? (eye + rd * 0.002f) : (eye + rd * (entryT + 0.002f));

    // Shrink the authored scene so the tornado fits the 2 m cube more naturally,
    // while still allowing the effect to extend beyond the cube without clipping.
    float fireScale = 10.0f;
    float maxDistance = 18.0f;
    float travel = 0.1f;
    float3 color = float3(0.0f);

    for (int i = 0; i < 100; ++i) {
        if (travel > maxDistance) {
            break;
        }

        float3 worldPoint = marchOrigin + rd * travel;
        FireTornadoSample sample = ftFireBall(worldPoint * fireScale, uniforms.time);
        float dist = sample.dist / fireScale;
        color += sample.glow;

        if (dist < FT_EPSILON) {
            break;
        }
        travel += dist;
    }

    color = tanh(color / 9.0e4f);
    color += ftBackground(rd) * (1.0f - clamp(length(color) * 1.8f, 0.0f, 1.0f));
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}
