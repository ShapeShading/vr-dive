// ShieldShaders.metal
// "Shield" by @XorDev — https://www.shadertoy.com/view/cltfRf
//
// Faithful 3D stereo port for visionOS.
//
// The original GLSL accumulates 100 concentric sphere shells in 2D screen space:
//   for(i=0; i<1; i+=.01) { p = screenNDC * i; ...sphere_distortion; ...hex; }
//
// Here each iteration analytically intersects the per-eye ray with the sphere
// shell of radius i.  Because left/right eye positions differ, each shell is
// sampled at a slightly different 3D point — producing real stereo parallax.
// The hex formula, sphere distortion, z-weighting, and tanh tonemap are
// unchanged from the original.

#include <metal_stdlib>
using namespace metal;

struct ShieldUniforms {
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

struct ShieldVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float3 SH_BOX_HALF = float3(1.0f);

static float2 shBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float shRaySphereHit(float3 ro, float3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) {
        return -1.0f;
    }

    float s = sqrt(h);
    float tNear = -b - s;
    float tFar = -b + s;
    if (tNear > 1.0e-4f) {
        return tNear;
    }
    if (tFar > 1.0e-4f) {
        return tFar;
    }
    return -1.0f;
}

vertex ShieldVertexOut shieldVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant ShieldUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    ShieldVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader — rebuild the original shell stack as actual 3D shells
// centered on the cube.  Each eye now traces its own ray through the shell
// volume, so stereo disparity comes from real 3D hit points rather than a
// shared screen-space UV projection.
// ---------------------------------------------------------------------------

fragment float4 shieldFragment(
    ShieldVertexOut in [[stage_in]],
    constant ShieldUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 eye = (camWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 rd = normalize(surfacePos - eye);

    bool insideOuter = all(abs(eye) < SH_BOX_HALF - 1.0e-3f);
    float2 tOuter = shBoxIntersect(eye, rd, SH_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 ro = eye;
    const float3 fixedRight = float3(1.0f, 0.0f, 0.0f);
    const float3 fixedUp = float3(0.0f, 1.0f, 0.0f);

    float t = uniforms.time;
    float4 O = float4(0.0f);

    for (int n = 1; n <= 100; n++) {
        float i = float(n) * 0.01f;
        float hitT = shRaySphereHit(ro, rd, i);
        if (hitT < max(tStart, 0.0f)) {
            continue;
        }

        float3 hit = ro + rd * hitT;
        float2 p = float2(dot(hit, fixedRight), dot(hit, fixedUp)) / max(i, 1.0e-4f);

        float  z = max(1.0f - dot(p, p), 0.0f);
        if (z <= 0.0f) {
            continue;
        }
        p /= 0.2f + sqrt(z) * 0.3f;

        p.x  = p.x / 0.9f + t;
        p.y += fract(ceil(p.x) * 0.5f) + t * 0.2f;

        float2 v       = abs(fract(p) - 0.5f);
        float  hexDist = abs(max(v.x * 1.5f + v, v + v).y - 1.0f)
                       + 0.1f - i * 0.09f;

        O += float4(2.0f, 3.0f, 5.0f, 1.0f) / 2000.0f * z / hexDist;
    }

    O = tanh(O * O);
    return float4(O.rgb, 1.0f);
}