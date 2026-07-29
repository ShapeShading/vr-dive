// MirrorLoopingShaders.metal
// "Mirror Looping" — cube-container adaptation of ShaderToy "XXdGDH"
// Source: https://www.shadertoy.com/view/XXdGDH
//
// Source notes:
// - The original shader builds a reflective Wythoff polyhedron from three
//   mirror planes, then traces repeated reflections through its interior.
// - This version preserves the fold / edge / trace / bounce logic, but replaces
//   the original screen-space orbit camera with the real per-eye world ray
//   entering a 2 m cube container.
// - The original sampled an environment map, a wall texture and controller
//   state from auxiliary channels. This Metal version uses procedural sky and
//   wall shading, and animates the truncation parameters directly over time.

#include <metal_stdlib>
using namespace metal;

struct MirrorLoopingUniforms {
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

struct MirrorLoopingVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct MLState {
    float3x3 mirrors;
    float3x3 triangle;
    float3 baseVertex;
};

static constant float ML_PI = 3.141592654f;
static constant float ML_EDGE_THICKNESS = 0.05f;
static constant int ML_MAX_TRACE_STEPS = 128;
static constant int ML_MAX_RAY_BOUNCES = 12;
static constant float ML_EPSILON = 1.0e-4f;
static constant float ML_FAR = 20.0f;
static constant float ML_SIZE = 1.35f;
static constant float3 ML_PQR = float3(2.0f, 3.0f, 3.0f);
static constant float3 ML_BOX_HALF = float3(1.0f);

vertex MirrorLoopingVertexOut mirrorLoopingVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant MirrorLoopingUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    MirrorLoopingVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float mlMin3(float x, float y, float z) {
    return min(x, min(y, z));
}

static float mlMax3(float x, float y, float z) {
    return max(x, max(y, z));
}

static float mlHash21(float2 p) {
    p = fract(p * float2(123.34f, 456.21f));
    p += dot(p, p + 34.45f);
    return fract(p.x * p.y);
}

static float3 mlWallAlbedo(float2 uv, float time) {
    float2 grid = uv * 4.5f;
    float2 cell = fract(grid) - 0.5f;
    float panel = smoothstep(0.48f, 0.1f, max(abs(cell.x), abs(cell.y)));
    float weave = 0.5f + 0.5f * sin(grid.x * 3.7f + grid.y * 2.8f + time * 0.4f);
    float grain = mlHash21(floor(grid * 2.0f));
    float tone = clamp(panel * 0.65f + weave * 0.25f + grain * 0.1f, 0.0f, 1.0f);
    return mix(float3(0.07f, 0.08f, 0.1f), float3(0.37f, 0.42f, 0.48f), tone);
}

static MLState mlInitState(float time) {
    MLState state;
    float3 c = cos(ML_PI / ML_PQR);
    float sp = sin(ML_PI / ML_PQR.x);
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
    state.triangle = float3x3(t0, t1, t2);

    float3 truncation = float3(
        0.5f * sin(time * 1.5f) + 0.5f,
        0.5f * sin(time * 0.8f) + 0.5f,
        0.5f * sin(time * 0.3f) + 0.5f);

    float determinant = dot(m1, cross(m2, m3));
    float3x3 inverseTranspose = float3x3(
        cross(m2, m3),
        cross(m3, m1),
        cross(m1, m2)) / determinant;
    state.baseVertex = normalize(inverseTranspose * truncation) * ML_SIZE;
    return state;
}

static float2 mlOuterBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float3 mlFold(float3 p, thread const MLState &state) {
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) {
            float3 mirror = state.mirrors[j];
            p -= 2.0f * min(dot(p, mirror), 0.0f) * mirror;
        }
    }
    return p;
}

static float mlMap(float3 p, thread const MLState &state) {
    p = mlFold(p, state) - state.baseVertex;
    return mlMax3(
        dot(p, state.triangle[0]),
        dot(p, state.triangle[1]),
        dot(p, state.triangle[2]));
}

static float3 mlDistEdges(float3 p, thread const MLState &state) {
    p = mlFold(p, state) - state.baseVertex;
    float3 ed;
    for (int i = 0; i < 3; ++i) {
        float3 mirror = state.mirrors[i];
        float3 q = p - min(0.0f, dot(p, mirror)) * mirror;
        ed[i] = dot(q, q);
    }
    return sqrt(ed);
}

static float mlTrace(float3 pos, float3 rd, bool outside, thread const MLState &state) {
    float t = 0.0f;
    float sgn = outside ? 1.0f : -1.0f;
    for (int i = 0; i < ML_MAX_TRACE_STEPS; ++i) {
        float d = mlMap(pos + t * rd, state);
        if (abs(d) < ML_EPSILON) {
            return t;
        }
        if (t > ML_FAR) {
            break;
        }
        t += sgn * d * 0.9f;
    }
    return ML_FAR;
}

static float3 mlGetNormal(float3 pos, thread const MLState &state) {
    float3 eps = float3(0.001f, 0.0f, 0.0f);
    return normalize(float3(
        mlMap(pos + eps.xyy, state) - mlMap(pos - eps.xyy, state),
        mlMap(pos + eps.yxy, state) - mlMap(pos - eps.yxy, state),
        mlMap(pos + eps.yyx, state) - mlMap(pos - eps.yyx, state)));
}

static float3 mlBackground(float3 dir, float time) {
    float t = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float3 sky = mix(float3(0.015f, 0.02f, 0.03f), float3(0.18f, 0.23f, 0.32f), t);
    float3 sunDir = normalize(float3(0.45f, 0.35f, 0.2f));
    float sun = pow(max(dot(dir, sunDir), 0.0f), 80.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 5.0f);
    float shimmer = 0.5f + 0.5f * sin((dir.x + dir.z) * 12.0f + time * 0.6f);
    sky += sun * float3(2.5f, 2.1f, 1.4f);
    sky += horizon * shimmer * float3(0.12f, 0.08f, 0.05f);
    return 2.2f * sky / max(1.0f - dot(sky, float3(0.2126f, 0.7152f, 0.0722f)) * 0.35f, 0.15f);
}

static float4 mlWallColor(float3 dir, float3 nor, float3 eds, float time) {
    float d = mlMin3(eds.x, eds.y, eds.z);
    float3 albedo = pow(mlWallAlbedo(eds.xy * 2.0f, time), float3(2.2f)) * 0.5f;
    float lighting = 0.2f + max(dot(nor, normalize(float3(0.8f, 0.5f, 0.0f))), 0.0f);

    if (dot(dir, nor) < 0.0f) {
        float f = clamp(d * 1000.0f - 3.0f, 0.0f, 1.0f);
        albedo = mix(float3(0.01f), albedo, f);
        return float4(albedo * lighting, f);
    }

    float m = mlMax3(eds.x, eds.y, eds.z);
    float2 a = fract(float2(d, m) * 40.6f + time * float2(0.03f, -0.05f)) - 0.5f;
    float aa = dot(a, a);
    float b = 0.2f / (aa + 0.2f);
    float lightShape = (1.0f - clamp(d * 100.0f - 2.0f, 0.0f, 1.0f)) * b;
    float3 emissive = float3(3.5f, 1.8f, 1.0f);
    return float4(mix(albedo * lighting, emissive, lightShape), 0.0f);
}

fragment float4 mirrorLoopingFragment(
    MirrorLoopingVertexOut in [[stage_in]],
    constant MirrorLoopingUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];
    MLState state = mlInitState(uniforms.time);

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 rd = normalize(surfacePos - eye);

    bool insideOuter = all(abs(eye) < ML_BOX_HALF - 1.0e-3f);
    float2 tOuter = mlOuterBoxIntersect(eye, rd, ML_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float entryT = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    float3 pos = eye + rd * (entryT + 0.002f);
    float3 color = float3(0.0f);
    float3 transmittance = float3(1.0f);

    if (mlMap(pos, state) > 0.0f) {
        float t = mlTrace(pos, rd, true, state);
        if (t >= ML_FAR) {
            float3 bg = mlBackground(rd, uniforms.time);
            bg = bg / (bg * 0.5f + 0.5f);
            return float4(clamp(bg, 0.0f, 1.0f), 1.0f);
        }

        pos += t * rd;
        float3 nor = mlGetNormal(pos, state);
        float3 reflDir = reflect(rd, nor);
        float3 bgColor = mlBackground(reflDir, uniforms.time);
        float fresnel = 0.04f + 0.96f * pow(1.0f - max(dot(rd, -nor), 0.0f), 5.0f);
        color += bgColor * fresnel;

        float3 eds = mlDistEdges(pos, state);
        float d = mlMin3(eds.x, eds.y, eds.z);
        if (d < ML_EDGE_THICKNESS) {
            float4 wc = mlWallColor(rd, nor, eds, uniforms.time);
            float3 result = color * wc.a + wc.rgb;
            result = result / (result * 0.5f + 0.5f);
            return float4(clamp(result, 0.0f, 1.0f), 1.0f);
        }
    }

    for (int i = 0; i < ML_MAX_RAY_BOUNCES; ++i) {
        float t = mlTrace(pos, rd, false, state);
        if (t >= ML_FAR) {
            color += transmittance * mlBackground(rd, uniforms.time);
            break;
        }

        pos += t * rd;
        float3 eds = mlDistEdges(pos, state);
        float3 nor = mlGetNormal(pos, state);
        float d = mlMin3(eds.x, eds.y, eds.z);
        if (d < ML_EDGE_THICKNESS) {
            color += transmittance * mlWallColor(rd, nor, eds, uniforms.time).rgb;
            break;
        }

        rd = reflect(rd, nor);
        pos += rd * 0.005f;
        transmittance *= float3(0.4f, 0.7f, 0.7f);
    }

    color = color / (color * 0.5f + 0.5f);
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}