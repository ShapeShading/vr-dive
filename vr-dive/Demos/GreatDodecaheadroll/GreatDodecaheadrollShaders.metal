// GreatDodecaheadrollShaders.metal
// "Great Dodecaheadroll" — cube-container adaptation of ShaderToy "tf23DD"
// Source: https://www.shadertoy.com/view/tf23DD
//
// Source notes:
// - The original shader ray-marches a transparent great dodecahedron from a
//   fixed screen-space camera and uses repeated side flips plus refraction to
//   create the rolling glassy look.
// - This Metal version keeps the pyramid-based SDF, surface normal, Fresnel and
//   refraction behavior, but reconstructs the ray from the real per-eye camera
//   entering a visible 2 m cube container.
// - When the viewer is outside the cube, marching begins at the cube surface;
//   when the viewer is inside, marching begins at the eye. The polyhedron is
//   evaluated in scene space and is not clipped by the container bounds.

#include <metal_stdlib>
using namespace metal;

struct GreatDodecaheadrollUniforms {
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

struct GreatDodecaheadrollVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float GD_PI = 3.14159265359f;
static constant float GD_SIN36 = 0.58778525229f;
static constant float GD_COS36 = 0.80901699437f;
static constant float GD_SIN72 = 0.95105651629f;
static constant float GD_COS72 = 0.30901699437f;
static constant float3 GD_BOX_HALF = float3(1.0f);

vertex GreatDodecaheadrollVertexOut greatDodecaheadrollVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant GreatDodecaheadrollUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    GreatDodecaheadrollVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 gdRotate(float2 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float pyr(float3 p, float incline, float radius) {
    float d12 = 0.0f;
    p.z = abs(p.z);
    float o = p.y * incline;

    d12 = max(d12, p.z * GD_SIN72 + abs(p.x * GD_COS72 + o));
    d12 = max(d12, p.z * GD_SIN36 + abs(p.x * GD_COS36 - o));
    d12 = max(d12, abs(p.x + o)) - radius;
    return d12 / 1.4142f;
}

static float gdMap(float3 p, float time) {
    const float incline = 0.5f;
    const float radius = 1.0f;

    float angleX = time * 0.25f;
    float angleY = angleX;

    p.yz = gdRotate(p.yz, angleY);
    p.xy = gdRotate(p.xy, angleX);

    float d12 = pyr(p, incline, radius);
    float nextDistance;
    float3 p0;
    p.x = -p.x;

    for (int ki = 0; ki < 5; ++ki) {
        float k = float(ki);
        p0 = p;
        p.xz = gdRotate(p.xz, GD_PI / 2.5f * k);
        p.xy = gdRotate(p.xy, GD_PI * -0.352416382f);
        nextDistance = pyr(p, incline, radius);
        d12 = min(d12, nextDistance);
        p = p0;
    }

    return d12;
}

static float3 gdNormal(float3 p, float time) {
    float2 e = float2(1.0e-2f, 0.0f);
    float d = gdMap(p, time);
    float3 n = d - float3(
        gdMap(p - e.xyy, time),
        gdMap(p - e.yxy, time),
        gdMap(p - e.yyx, time));
    return normalize(n);
}

static float2 gdBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 gdFaceUV(float3 p) {
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

static float3 gdEnvironment(float3 rd, float time) {
    rd = normalize(rd);
    float skyMix = clamp(rd.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(rd.y), 0.0f), 4.0f);
    float sun = pow(max(dot(rd, normalize(float3(0.4f, 0.5f, -0.7f))), 0.0f), 40.0f);
    float shimmer = 0.5f + 0.5f * sin((rd.x - rd.z) * 9.0f + time * 0.4f);
    float3 sky = mix(float3(0.02f, 0.03f, 0.05f), float3(0.16f, 0.2f, 0.28f), skyMix);
    sky += float3(0.9f, 0.55f, 0.35f) * horizon * 0.35f * shimmer;
    sky += float3(1.0f, 0.92f, 0.74f) * sun;
    return sky;
}

fragment float4 greatDodecaheadrollFragment(
    GreatDodecaheadrollVertexOut in [[stage_in]],
    constant GreatDodecaheadrollUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float scale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / scale;
    float3 surfacePos = (in.worldPos - center) / scale;
    float3 rd = normalize(surfacePos - eye);

    bool insideOuter = all(abs(eye) < GD_BOX_HALF - 1.0e-3f);
    float2 tOuter = gdBoxIntersect(eye, rd, GD_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    const float sceneScale = 1.75f;
    float3 ro = (eye + rd * (tStart + 0.001f)) * sceneScale;
    float3 marchDir = normalize(rd);

    float3 col = float3(0.0f);
    float3 p = ro;
    float at = 0.0f;
    float side = 1.0f;
    int stepsTaken = 0;

    for (int i = 0; i < 80; ++i) {
        float d = gdMap(p, uniforms.time) * side;
        stepsTaken = i;

        if (d < 1.0e-3f) {
            float3 n = gdNormal(p, uniforms.time) * side;
            float3 l = normalize(ro - float3(5.0f));
            float3 r = normalize(marchDir);
            if (dot(l, n) < 0.0f) {
                l = -l;
            }

            float3 h = normalize(l - r);
            float fres = pow(1.0f - max(0.0f, dot(-marchDir, n)), 5.0f);
            float diff = pow(max(0.0f, dot(l, n)), 4.0f);

            col += diff * (
                1.8f * pow(max(0.0f, dot(h, n)), 12.0f) +
                1.6f * pow(max(0.0f, dot(marchDir, n)), 18.0f));
            col += 0.5f * fres;

            side *= -1.0f;
            float eta = 1.0f + 0.45f * side;
            float3 refracted = refract(marchDir, n, eta);
            if (length_squared(refracted) < 1.0e-8f) {
              refracted = reflect(marchDir, n);
            }
            marchDir = normalize(refracted);
            d = 9.0e-2f;
        }

        if (d > 20.0f) {
            break;
        }

        p += marchDir * d;
        at += 0.01f / max(d, 1.0e-3f);
    }

    float3 tint = float3(-marchDir * cos(uniforms.time / 5.0f) - 0.5f);
    col += at * 0.001f + float(stepsTaken) / 800.0f;
    col = mix(col, tint * 1.25f, 0.4f);

    if (stepsTaken < 2 && gdMap(ro, uniforms.time) > 0.3f) {
        float3 env = gdEnvironment(marchDir, uniforms.time);
        float2 faceUV = gdFaceUV(surfacePos) * 2.0f - 1.0f;
        float vignette = 1.0f - 0.25f * dot(faceUV, faceUV);
        col += env * vignette * 0.35f;
    }

    return float4(clamp(col * 1.33f, 0.0f, 1.0f), 1.0f);
}