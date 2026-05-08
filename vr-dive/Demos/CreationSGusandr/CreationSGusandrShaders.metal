// CreationSGusandrShaders.metal
// Adapted from ShaderToy "Creation S gusandr".
// Source: https://www.shadertoy.com/view/ssXXzn
//
// Metal adaptation notes:
// - The original ShaderToy uses a multi-pass setup: one buffer provides the
//   outer shell normal and hit distance, another texture shapes the interior,
//   and a cubemap supplies the environment.
// - This version folds the effect into one pass for the cube-container demo.
//   It ray-marches a procedural shell directly, computes a local surface normal,
//   then performs the refracted medium march inside that shell.
// - The visible container is only the entry surface. Once the ray enters from
//   the cube boundary, the simulated structure is allowed to continue beyond the
//   2 m cube so the effect is not clipped by the container itself.

#include <metal_stdlib>
using namespace metal;

struct CreationSGusandrUniforms {
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

struct CreationSGusandrVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct CSGHit {
    float distance;
    float mediumDistance;
    float density;
    float3 normal;
    bool hit;
};

static constant float CSG_FIXED_STEP_SIZE = 0.0175f;
static constant int   CSG_FIXED_STEPS = 100;
static constant float CSG_MAX_DISTANCE = 50.0f;
static constant float CSG_EPSILON = 0.025f;
static constant float CSG_EPSILON_MEDIUM = 0.75f;
static constant float CSG_MEDIUM_ETA = 0.5757575f;
static constant float3 CSG_BOX_HALF = float3(1.0f);

vertex CreationSGusandrVertexOut creationSGusandrVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant CreationSGusandrUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    CreationSGusandrVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 csgRotate2D(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float hash13(float3 p3)
{
    p3 = fract(p3 * 0.1031f);
    p3 += dot(p3, p3.yzx + 19.19f);
    return fract((p3.x + p3.y) * p3.z);
}

static float3 hash33(float3 p3)
{
    p3 = fract(p3 * float3(0.1031f, 0.1030f, 0.0973f));
    p3 += dot(p3, p3.yxz + 19.19f);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

static float hash21(float2 p)
{
    float3 p3 = fract(float3(p.xyx) * float3(0.1031f, 0.1030f, 0.0973f));
    p3 += dot(p3, p3.yzx + 19.19f);
    return fract((p3.x + p3.y) * p3.z);
}

static float valueNoise(float2 p)
{
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0f - 2.0f * f);

    float a = hash21(i + float2(0.0f, 0.0f));
    float b = hash21(i + float2(1.0f, 0.0f));
    float c = hash21(i + float2(0.0f, 1.0f));
    float d = hash21(i + float2(1.0f, 1.0f));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

static float terrainTexture(float2 p)
{
    float n = 0.0f;
    float amplitude = 0.55f;
    float frequency = 1.0f;
    for (int i = 0; i < 4; ++i) {
        n += valueNoise(p * frequency) * amplitude;
        frequency *= 2.07f;
        amplitude *= 0.5f;
    }
    return clamp(n, 0.0f, 1.0f);
}

static float dot2(float3 p)
{
    return dot(p, p);
}

static float worley(float3 p)
{
    float d = 10.0f;
    float3 n = floor(p);

    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        float3 neighbor = n + float3((float)x, (float)y, (float)z);
        float3 centerPosition = neighbor + hash33(neighbor);
        d = min(d, dot2(centerPosition - p) + 0.7f);
    }

    return d;
}

static float terrainDeposit(float3 p)
{
    p.xz -= worley(p * 2.342f) * 0.2f;
    p.y += 0.45f;
    p *= 0.15f;
    p.xz *= 0.75f;
    p.y += sin(p.x * 12.0f) * 0.05f - 0.05f;
    p.xz += sin(p.y * 12.0f + float2(0.0f, 0.8f)) * 0.01f + 0.01f;
    p.y *= 0.35f;

    float tx = terrainTexture(p.xz) + 0.05f;
    float terrain = smoothstep(0.3f, 0.0f, tx * 0.2f + terrainTexture(p.xz * 2.0f) * 0.5f) * 1.5f;
    return terrain;
}

static float densityField(float3 p)
{
    p.xz -= worley(p * 2.342f) * 0.2f;
    p.y += 0.45f;
    p *= 0.15f;
    p.xz *= 0.75f;
    float d = p.y * 1.5f;
    p.y += sin(p.x * 12.0f) * 0.05f - 0.05f;
    p.xz += sin(p.y * 12.0f + float2(0.0f, 0.8f)) * 0.01f + 0.01f;
    p.y *= 0.35f;

    float edge0 = p.y * 8.0f + (sin(p.y) * 0.5f + 0.5f) * 0.3f;
    float terrain = smoothstep(edge0 + 0.2f, edge0, terrainTexture(p.xz * 2.0f)) * 1.25f;

    d += terrain;
    d += sin(terrain * 3.14159265f - 0.5f) * 1.5f;
    d += p.y + 0.4f;
    d -= smoothstep(0.1f, -0.05f, p.y - d * 0.005f);

    return max(d, 0.0f);
}

static float shellSDF(float3 p, float time)
{
    float3 q = p;
    q.xz = csgRotate2D(q.xz, time * 0.18f);

    float radial = length(q.xz);
    float lobe = sin(atan2(q.z, q.x) * 7.0f + q.y * 3.4f - time * 0.7f) * 0.11f;
    float ripples = sin(q.y * 8.0f - radial * 6.0f + time * 1.2f) * 0.045f;
    float cellWarp = (worley(q * 1.35f + time * 0.1f) - 0.95f) * 0.09f;
    float taper = mix(1.35f, 0.52f, smoothstep(-1.1f, 1.3f, q.y));
    float body = length(float2(radial - lobe, (q.y + 0.18f) * 0.82f)) - taper;
    float stem = length(float2(radial, q.y + 1.15f)) - 0.32f;
    return min(body + ripples + cellWarp, stem);
}

static float3 shellNormal(float3 p, float time)
{
    float2 e = float2(0.003f, 0.0f);
    return normalize(float3(
        shellSDF(p + e.xyy, time) - shellSDF(p - e.xyy, time),
        shellSDF(p + e.yxy, time) - shellSDF(p - e.yxy, time),
        shellSDF(p + e.yyx, time) - shellSDF(p - e.yyx, time)));
}

static float2 boxIntersect(float3 ro, float3 rd, float3 halfExt)
{
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float3 environmentColor(float3 dir)
{
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 4.0f);
    float sun = pow(max(dot(dir, normalize(float3(-0.35f, 0.55f, -0.75f))), 0.0f), 42.0f);
    float3 sky = mix(float3(0.04f, 0.06f, 0.10f), float3(0.18f, 0.24f, 0.34f), skyMix);
    float3 glow = float3(0.95f, 0.62f, 0.32f) * horizon * 0.28f;
    return sky + glow + float3(1.0f, 0.82f, 0.6f) * sun * 0.9f;
}

static CSGHit traceShell(float3 ro, float3 rd, float time)
{
    CSGHit hit;
    hit.distance = CSG_MAX_DISTANCE;
    hit.mediumDistance = 0.0f;
    hit.density = 0.0f;
    hit.normal = float3(0.0f, 1.0f, 0.0f);
    hit.hit = false;

    float t = 0.0f;
    for (int i = 0; i < 160; ++i) {
        float3 p = ro + rd * t;
        float sd = shellSDF(p, time);
        if (abs(sd) < CSG_EPSILON * 0.35f) {
            hit.distance = t;
            hit.normal = shellNormal(p, time);
            hit.hit = true;
            break;
        }
        t += clamp(abs(sd) * 0.55f, 0.015f, 0.22f);
        if (t > CSG_MAX_DISTANCE) {
            break;
        }
    }

    return hit;
}

static void traceMedium(
    float3 hitPosition,
    float3 rayDir,
    float3 normal,
    float time,
    thread CSGHit &hit)
{
    float roughETA = CSG_MEDIUM_ETA + hash13(hitPosition * 44.0f) * 0.02f;
    float3 refr = refract(rayDir, normal, roughETA);
    if (length_squared(refr) < 1.0e-6f) {
        refr = reflect(rayDir, normal);
    }

    float t = CSG_FIXED_STEP_SIZE;
    float d = 0.0f;
    for (int i = 0; i < CSG_FIXED_STEPS; ++i) {
        float3 p = hitPosition + refr * t;
        if (shellSDF(p, time) > 0.015f && t > 0.05f) {
            break;
        }

        float dd = densityField(p);
        d += dd;
        t += CSG_FIXED_STEP_SIZE * max(dd, 0.35f) * (0.9f + hash13(p * 22.2f) * 0.3f);
        if (dd < CSG_EPSILON_MEDIUM || t > 4.5f) {
            break;
        }
    }

    hit.density = d;
    hit.mediumDistance = t;
}

static float2 faceUV(float3 p) {
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

fragment float4 creationSGusandrFragment(
    CreationSGusandrVertexOut in [[stage_in]],
    constant CreationSGusandrUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 camRight = normalize(float3(v2w[0].x, v2w[0].y, v2w[0].z));
    float3 camUp = normalize(float3(v2w[1].x, v2w[1].y, v2w[1].z));

    float cubeScale = max(uniforms.boxScale, 1.0e-4f);
    float3 eye = (camWorld - center) / cubeScale;
    float3 hitOnCube = (in.worldPos - center) / cubeScale;
    float3 rd = normalize(hitOnCube - eye);

    bool insideBox = all(abs(eye) < CSG_BOX_HALF - 1.0e-3f);
    float2 tBox = boxIntersect(eye, rd, CSG_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 ro = eye + rd * (tStart + CSG_EPSILON * 0.5f);

    CSGHit shellHit = traceShell(ro, rd, uniforms.time);
    if (!shellHit.hit) {
        float2 uv = faceUV(hitOnCube) * 2.0f - 1.0f;
        float vignette = 1.0f - 0.28f * dot(uv, uv);
        float3 bg = environmentColor(rd) * vignette * vignette;
        return float4(bg, 1.0f);
    }

    float3 shellPosition = ro + rd * shellHit.distance;
    traceMedium(shellPosition + shellHit.normal * CSG_EPSILON, rd, shellHit.normal, uniforms.time, shellHit);

    float roughETA = CSG_MEDIUM_ETA + hash13(shellPosition * 44.0f) * 0.02f;
    float3 refr = refract(rd, shellHit.normal, roughETA);
    if (length_squared(refr) < 1.0e-6f) {
        refr = reflect(rd, shellHit.normal);
    }

    float3 reflected = reflect(rd, shellHit.normal);
    float3 env = environmentColor(reflected);
    float fresnel = smoothstep(0.65f, 0.2f, -dot(shellHit.normal, rd));

    float3 innerColor = float3(0.25f, 0.75f, 1.0f);
    float den = max(0.0001f, shellHit.density) * 0.0075f;
    float3 innerP = shellPosition + refr * shellHit.mediumDistance;
    float deposit = terrainDeposit(innerP);
    float3 lightPos = eye - camRight * 6.0f - camUp * 15.0f;
    float3 toLight = normalize(lightPos - innerP);
    float lighting = 0.35f + 0.65f * clamp(dot(toLight, shellHit.normal), 0.0f, 1.0f);

    float3 volumetric = innerColor * (den + deposit * 0.375f) + float3(deposit * 0.2f);
    volumetric *= volumetric * smoothstep(0.0f, 0.65f, shellHit.mediumDistance);
    volumetric *= lighting;

    float rim = smoothstep(-0.05f, 0.2f, shellHit.normal.y);
    float3 color = env * fresnel * 0.75f * rim + volumetric;
    color += innerColor * pow(max(1.0f - abs(dot(shellHit.normal, rd)), 0.0f), 6.0f) * 0.08f;

    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}