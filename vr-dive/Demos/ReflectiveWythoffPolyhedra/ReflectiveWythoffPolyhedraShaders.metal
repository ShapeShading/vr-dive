// ReflectiveWythoffPolyhedraShaders.metal
// "Reflective Wythoff polyhedra" — cube-portal adaptation of Shadertoy "ctVGRR"
// Original: https://www.shadertoy.com/view/ctVGRR
//
// Source notes:
// - The original shader constructs a reflective Wythoff polyhedron from three
//   mirror planes and ray-traces it from a synthetic orbit camera.
// - This version keeps the fold / distance / trace / bounce logic, but replaces
//   the screen-space camera with the real per-eye world ray hitting a 2 m cube.
// - When the viewer is outside the cube, marching starts at the visible cube
//   face. When the viewer is inside the cube, marching starts at the eye.
// - The solid itself lives in scene space and is not clipped by the cube bounds,
//   so motion around the portal produces real stereo parallax.
// - The original sampled two textures for walls and sky. This Metal version
//   uses procedural wall and sky shading instead, so it remains self-contained.

#include <metal_stdlib>
using namespace metal;

struct ReflectiveWythoffPolyhedraUniforms {
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

struct ReflectiveWythoffPolyhedraVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct RWState {
    float3x3 mirrors;
    float3x3 triangleVertices;
    float3 baseVertex;
};

static constant float RW_PI = 3.141592654f;
static constant float RW_EDGE_THICKNESS = 0.05f;
static constant int RW_MAX_TRACE_STEPS = 128;
static constant int RW_MAX_RAY_BOUNCES = 12;
static constant float RW_EPSILON = 1.0e-4f;
static constant float RW_FAR = 20.0f;
static constant float RW_SIZE = 0.675f;
static constant float3 RW_PQR = float3(2.0f, 3.0f, 5.0f);
static constant float3 RW_TRUNCATION = float3(1.0f, 1.0f, 1.0f);
static constant float3 RW_BOX_HALF = float3(1.0f);

vertex ReflectiveWythoffPolyhedraVertexOut reflectiveWythoffPolyhedraVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant ReflectiveWythoffPolyhedraUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    ReflectiveWythoffPolyhedraVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float rwMin3(float x, float y, float z) {
    return min(x, min(y, z));
}

static float rwMax3(float x, float y, float z) {
    return max(x, max(y, z));
}

static float rwHash21(float2 p) {
    p = fract(p * float2(123.34f, 456.21f));
    p += dot(p, p + 34.45f);
    return fract(p.x * p.y);
}

static RWState rwInitState() {
    RWState state;

    float3 c = cos(RW_PI / RW_PQR);
    float sp = sin(RW_PI / RW_PQR.x);

    float3 m1 = float3(1.0f, 0.0f, 0.0f);
    float3 m2 = float3(-c.x, sp, 0.0f);
    float x3 = -c.z;
    float y3 = -(c.y + c.x * c.z) / sp;
    float z3 = sqrt(max(1.0f - x3 * x3 - y3 * y3, 0.0f));
    float3 m3 = float3(x3, y3, z3);

    state.mirrors = float3x3(m1, m2, m3);

    float3 t0 = normalize(cross(m2, m3));
    float3 t1 = normalize(cross(m3, m1));
    float3 t2 = normalize(cross(m1, m2));
    state.triangleVertices = float3x3(t0, t1, t2);

    float determinant = dot(m1, cross(m2, m3));
    float3x3 inverseTranspose = float3x3(
        cross(m2, m3),
        cross(m3, m1),
        cross(m1, m2)) / determinant;
    state.baseVertex = normalize(inverseTranspose * RW_TRUNCATION) * RW_SIZE;
    return state;
}

static float rwBoxHit(float3 ro, float3 rd, float3 halfExtents, thread float3 &nn, bool entering) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 invRD = 1.0f / rd;
    float3 roOverRD = ro * invRD;
    float3 extentsOverRD = halfExtents * abs(invRD);
    float3 pin = -extentsOverRD - roOverRD;
    float3 pout = extentsOverRD - roOverRD;
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

static float3 rwFold(float3 p, thread const RWState &state) {
    for (int outer = 0; outer < 5; ++outer) {
        for (int mirrorIndex = 0; mirrorIndex < 3; ++mirrorIndex) {
            float3 mirror = state.mirrors[mirrorIndex];
            p -= 2.0f * min(dot(p, mirror), 0.0f) * mirror;
        }
    }
    return p;
}

static float3 rwDistEdges(float3 p, thread const RWState &state) {
    p = rwFold(p, state) - state.baseVertex;

    float3 edgeDistance;
    for (int mirrorIndex = 0; mirrorIndex < 3; ++mirrorIndex) {
        float3 mirror = state.mirrors[mirrorIndex];
        float3 q = p - min(0.0f, dot(p, mirror)) * mirror;
        edgeDistance[mirrorIndex] = dot(q, q);
    }
    return sqrt(edgeDistance);
}

static float rwMap(float3 p, thread const RWState &state) {
    p = rwFold(p, state) - state.baseVertex;
    return rwMax3(
        dot(p, state.triangleVertices[0]),
        dot(p, state.triangleVertices[1]),
        dot(p, state.triangleVertices[2]));
}

static float rwTrace(float3 pos, float3 rd, bool outside, thread const RWState &state) {
    float t = 0.0f;
    float sgn = outside ? 1.0f : -1.0f;
    for (int stepIndex = 0; stepIndex < RW_MAX_TRACE_STEPS; ++stepIndex) {
        float d = rwMap(pos + t * rd, state);
        if (abs(d) < RW_EPSILON) {
            return t;
        }
        if (t > RW_FAR) {
            break;
        }
        t += sgn * d * 0.9f;
    }
    return RW_FAR;
}

static float3 rwGetNormal(float3 pos, thread const RWState &state) {
    float3 eps = float3(0.001f, 0.0f, 0.0f);
    return normalize(float3(
        rwMap(pos + eps.xyy, state) - rwMap(pos - eps.xyy, state),
        rwMap(pos + eps.yxy, state) - rwMap(pos - eps.yxy, state),
        rwMap(pos + eps.yyx, state) - rwMap(pos - eps.yyx, state)));
}

static float3 rwWallAlbedo(float2 uv, float time) {
    float2 tile = uv * 6.0f;
    float2 cell = fract(tile) - 0.5f;
    float panel = smoothstep(0.48f, 0.10f, max(abs(cell.x), abs(cell.y)));
    float brushed = 0.5f + 0.5f * sin(tile.x * 4.2f + tile.y * 2.7f + time * 0.35f);
    float grain = rwHash21(floor(tile * 2.0f));
    float mixValue = clamp(panel * 0.7f + brushed * 0.2f + grain * 0.1f, 0.0f, 1.0f);
    return mix(float3(0.08f, 0.09f, 0.11f), float3(0.36f, 0.38f, 0.42f), mixValue);
}

static float4 rwWallColor(float3 dir, float3 nor, float3 eds, float time) {
    float d = rwMin3(eds.x, eds.y, eds.z);

    float3 albedo = rwWallAlbedo(eds.xy * 2.0f, time) * 0.5f;
    float lighting = 0.2f + max(dot(nor, normalize(float3(0.8f, 0.5f, 0.0f))), 0.0f);

    if (dot(dir, nor) < 0.0f) {
        float f = clamp(d * 1000.0f - 3.0f, 0.0f, 1.0f);
        albedo = mix(float3(0.01f), albedo, f);
        return float4(albedo * lighting, f);
    }

    float m = rwMax3(eds.x, eds.y, eds.z);
    float2 a = fract(float2(d, m) * 40.6f + time * float2(0.03f, -0.05f)) - 0.5f;
    float aa = dot(a, a);
    float b = 0.2f / (aa + 0.2f);
    float lightShape = (1.0f - clamp(d * 100.0f - 2.0f, 0.0f, 1.0f)) * b;

    float3 emissive = float3(3.5f, 1.8f, 1.0f);
    return float4(mix(albedo * lighting, emissive, lightShape), 0.0f);
}

static float3 rwBackground(float3 dir, float time) {
    float t = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float3 sky = mix(float3(0.02f, 0.025f, 0.035f), float3(0.16f, 0.19f, 0.24f), t);

    float3 sunDir = normalize(float3(0.5f, 0.35f, 0.2f));
    float sun = pow(max(dot(dir, sunDir), 0.0f), 72.0f);
    float horizon = pow(1.0f - abs(dir.y), 5.0f);
    float shimmer = 0.5f + 0.5f * sin((dir.x + dir.z) * 12.0f + time * 0.6f);

    sky += sun * float3(2.5f, 2.1f, 1.4f);
    sky += horizon * shimmer * float3(0.12f, 0.08f, 0.05f);
    return sky;
}

fragment float4 reflectiveWythoffPolyhedraFragment(
    ReflectiveWythoffPolyhedraVertexOut in [[stage_in]],
    constant ReflectiveWythoffPolyhedraUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];
    RWState state = rwInitState();

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float sceneScale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / sceneScale;
    float3 surfacePos = (in.worldPos - center) / sceneScale;
    float3 rd = normalize(surfacePos - eye);

    bool insideBox = all(abs(eye) < float3(0.999f));
    float3 faceNormal;
    float entryT = insideBox ? 0.0f : rwBoxHit(eye, rd, RW_BOX_HALF, faceNormal, true);
    if (!insideBox && entryT < 0.0f) {
        discard_fragment();
    }

    float3 pos = insideBox ? (eye + rd * 0.002f) : (eye + rd * (entryT + 0.002f));
    float3 color = float3(0.0f);
    float3 transmittance = float3(1.0f);

    if (rwMap(pos, state) > 0.0f) {
        float t = rwTrace(pos, rd, true, state);
        if (t >= RW_FAR) {
            float3 bg = rwBackground(rd, uniforms.time);
            bg = bg / (bg * 0.5f + 0.5f);
            return float4(clamp(bg, 0.0f, 1.0f), 1.0f);
        }

        pos += t * rd;
        float3 nor = rwGetNormal(pos, state);
        float3 reflectedDir = reflect(rd, nor);
        float3 bgColor = rwBackground(reflectedDir, uniforms.time);
        float fresnel = 0.04f + 0.96f * pow(1.0f - max(dot(rd, -nor), 0.0f), 5.0f);
        color += bgColor * fresnel;

        float3 eds = rwDistEdges(pos, state);
        float d = rwMin3(eds.x, eds.y, eds.z);
        if (d < RW_EDGE_THICKNESS) {
            float4 wc = rwWallColor(rd, nor, eds, uniforms.time);
            float3 result = color * wc.a + wc.rgb;
            result = result / (result * 0.5f + 0.5f);
            return float4(clamp(result, 0.0f, 1.0f), 1.0f);
        }
    }

    for (int bounceIndex = 0; bounceIndex < RW_MAX_RAY_BOUNCES; ++bounceIndex) {
        float t = rwTrace(pos, rd, false, state);
        if (t >= RW_FAR) {
            color += transmittance * rwBackground(rd, uniforms.time);
            break;
        }

        pos += t * rd;
        float3 eds = rwDistEdges(pos, state);
        float3 nor = rwGetNormal(pos, state);
        float d = rwMin3(eds.x, eds.y, eds.z);
        if (d < RW_EDGE_THICKNESS) {
            color += transmittance * rwWallColor(rd, nor, eds, uniforms.time).rgb;
            break;
        }

        rd = reflect(rd, nor);
        pos += rd * 0.005f;
        transmittance *= float3(0.4f, 0.7f, 0.7f);
    }

    color = color / (color * 0.5f + 0.5f);
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}