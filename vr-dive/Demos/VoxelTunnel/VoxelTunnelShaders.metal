// VoxelTunnelShaders.metal
// Source adaptation: @lsdlive, Shadertoy "Voxel tunnel"
// https://www.shadertoy.com/view/MscBRs
// Original algorithm note in the source references fb39ca4 and Shane's DDA
// implementation. This version adapts the effect to a 2.4 m cube container and
// supports rays entering from outside or starting from inside the cube.
//
// GLSL -> Metal adaptation notes:
// - Use an explicit 2D rotation matrix instead of GLSL constructor tricks.
// - Keep the glow accumulator thread-local instead of a mutable global.
// - Treat the visible cube face as a portal when the camera is outside so the
//   tunnel can continue beyond the box instead of being clipped at the back face.

#include <metal_stdlib>
using namespace metal;

struct VoxelTunnelUniforms {
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

struct VoxelTunnelVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

vertex VoxelTunnelVertexOut voxelTunnelVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant VoxelTunnelUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    VoxelTunnelVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static constant float3 VT_BOX_HALF = float3(1.0f, 1.0f, 1.0f);
static constant float VT_SCENE_SCALE = 8.0f;
static constant float VT_OUTSIDE_VIEW_DEPTH = 14.0f;
static constant int VT_MAX_DDA_STEPS = 448;

static float2 vtR2d(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float2 vtPath(float t) {
    float a = sin(t * 0.2f + 1.5f);
    float b = sin(t * 0.2f);
    return float2(2.0f * a, a * b);
}

static float vtDe(float3 p, float time, thread float &glow) {
    p.z += time * 6.0f;
    p.xy -= vtPath(p.z);

    float d = -length(p.xy) + 4.0f;

    p.xy += float2(cos(p.z + time) * sin(time), cos(p.z + time));
    p.z -= 6.0f + time * 6.0f;

    float3 octSign = sign(p + float3(1e-4f));
    float oct = dot(p, normalize(octSign)) - 1.0f;
    d = min(d, oct);

    glow += 0.015f / (0.01f + d * d);
    return d;
}

static float vtBoxHit(float3 ro, float3 rd, float3 halfExtents, thread float3 &nn, bool entering) {
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

fragment float4 voxelTunnelFragment(
    VoxelTunnelVertexOut in [[stage_in]],
    constant VoxelTunnelUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center = uniforms.objectCenter.xyz;
    float scale = uniforms.cubeScale;
    float3 eye = (camWorld - center) / scale;
    float3 rdWorld = normalize(in.worldPos - camWorld);
    float3 rd = normalize(rdWorld);

    bool insideBox = all(abs(eye) < (VT_BOX_HALF - 1e-3f));
    float3 faceNormal;
    float faceT = vtBoxHit(eye, rd, VT_BOX_HALF, faceNormal, !insideBox);
    if (faceT < 0.0f) {
        discard_fragment();
    }

    float3 facePoint = eye + rd * faceT;
    float2 faceCoords = facePoint.xy * faceNormal.z / VT_BOX_HALF.xy
                      + facePoint.yz * faceNormal.x / VT_BOX_HALF.yz
                      + facePoint.zx * faceNormal.y / VT_BOX_HALF.zx;
    float edgeCoord = max(abs(faceCoords.x), abs(faceCoords.y));
    float edgeGlow = smoothstep(0.84f, 0.985f, edgeCoord);
    float faceFade = 1.0f - smoothstep(0.92f, 1.02f, edgeCoord);

    float3 start = insideBox ? (eye + rd * 0.002f) : (facePoint + rd * 0.002f);
    float maxDistance;
    if (insideBox) {
        maxDistance = max(faceT - 0.002f, 0.0f);
    } else {
        float3 exitNormal;
        float exitT = vtBoxHit(start, rd, VT_BOX_HALF, exitNormal, false);
        float throughCube = exitT > 0.0f ? exitT : 4.0f;
        maxDistance = throughCube + VT_OUTSIDE_VIEW_DEPTH;
    }

    float3 sceneRo = start * VT_SCENE_SCALE;
    float3 sceneRd = normalize(rd);
    sceneRo.z = -sceneRo.z;
    sceneRd.z = -sceneRd.z;
    sceneRd.xy = vtR2d(sceneRd.xy, sin(-sceneRo.x / 3.14f) * 0.3f);

    float3 voxel = floor(sceneRo) + 0.5f;
    float3 mask = float3(0.0f);
    float3 drd = 1.0f / max(abs(sceneRd), float3(1e-4f));
    float3 raySign = sign(sceneRd + float3(1e-4f));
    float3 side = drd * (raySign * (voxel - sceneRo) + 0.5f);

    float glow = 0.0f;
    bool hit = false;
    float travel = 0.0f;
    float traveledSceneLimit = maxDistance * VT_SCENE_SCALE;

    for (int i = 0; i < VT_MAX_DDA_STEPS; ++i) {
        float d = vtDe(voxel, uniforms.time, glow);
        if (d < 0.0f) {
            hit = true;
            break;
        }

        mask = step(side, side.yzx) * step(side, side.zxy);
        side += drd * mask;
        voxel += raySign * mask;
        travel = length(voxel - sceneRo);
        if (travel > traveledSceneLimit) {
            break;
        }
    }

    float axisMix = length(mask * float3(1.0f, 0.5f, 0.75f));
    float3 color = mix(float3(0.2f, 0.2f, 0.7f), float3(0.2f, 0.1f, 0.2f), axisMix);
    color += glow * 0.4f;
    color.r += sin(uniforms.time) * 0.2f + sin(-voxel.z * 0.5f - uniforms.time * 6.0f);
    color = mix(color, float3(0.2f, 0.1f, 0.2f), 1.0f - exp(-0.00025f * travel * travel));

    float3 glassBase = mix(float3(0.012f, 0.014f, 0.022f), float3(0.05f, 0.08f, 0.14f), 1.0f - faceFade);
    glassBase += edgeGlow * float3(0.08f, 0.13f, 0.22f);
    float hitMix = hit ? 1.0f : 0.28f;
    color = mix(glassBase, color, hitMix);
    float fresnel = pow(clamp(1.0f - abs(dot(faceNormal, rd)), 0.0f, 1.0f), 3.0f);
    color += fresnel * float3(0.05f, 0.08f, 0.12f);
    color = clamp(tanh(color), 0.0f, 1.0f);
    return float4(color, 1.0f);
}