// SoulstoneShaders.metal
// Adapted from ShaderToy "Soulstone".
// Source: https://www.shadertoy.com/view/llSBRD
//
// Metal adaptation notes:
// - The original GLSL shader uses cubemap and 2D texture lookups for reflection,
//   triplanar detail and UI-like glow modulation. This Metal version replaces
//   them with a procedural sky, procedural triplanar noise and analytic glow.
// - The visible 2 m cube is only the entry container. Rays are reconstructed
//   from the real per-eye camera pose, begin at the cube surface when viewed
//   from outside, and continue marching the crystal beyond the container bounds.
// - GLSL helpers relying on swizzle l-values and `uintBitsToFloat` are recast
//   into Metal-safe helper functions.

#include <metal_stdlib>
using namespace metal;

struct SoulstoneUniforms {
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

struct SoulstoneVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct SoulstoneIntersection {
    float totalDistance;
    float mediumDistance;
    float sdf;
    float density;
    int materialID;
    bool hit;
};

static constant float SOULSTONE_TAU = 6.2831853f;
static constant float SOULSTONE_CRYSTAL_SCALE = 1.0f;
static constant float SOULSTONE_VERTICAL_ANISOTROPY = 1.3f;
static constant int SOULSTONE_MAX_STEPS = 50;
static constant float SOULSTONE_FIXED_STEP_SIZE = 0.05f;
static constant float SOULSTONE_MAX_DISTANCE = 15.0f;
static constant float SOULSTONE_EPSILON = 0.01f;
static constant float SOULSTONE_EPSILON_NORMAL = 0.1f;
static constant float3 SOULSTONE_BOX_HALF = float3(1.0f);
static constant int SOULSTONE_MATERIAL_NONE = -1;
static constant int SOULSTONE_MATERIAL_CRYSTAL = 1;

vertex SoulstoneVertexOut soulstoneVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant SoulstoneUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    SoulstoneVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float ssSaturate(float x) {
    return clamp(x, 0.0f, 1.0f);
}

static float2 ssRotate2D(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float ssHash11(float n) {
    return fract(sin(n * 127.1f) * 43758.5453123f);
}

static float ssHash31(float3 p) {
    return fract(sin(dot(p, float3(0.09123898f, 0.0231233f, 0.0532234f))) * 100000.0f);
}

static float3 ssHash33(float3 p) {
    return fract(sin(float3(
        dot(p, float3(127.1f, 311.7f, 74.7f)),
        dot(p, float3(269.5f, 183.3f, 246.1f)),
        dot(p, float3(113.5f, 271.9f, 124.6f)))) * 43758.5453123f);
}

static float2 ssComplexSquare(float2 a) {
    return float2(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y);
}

static float3 ssRotateX(float3 p, float t) {
    p.yz = ssRotate2D(p.yz, t);
    return p;
}

static float3 ssRotateY(float3 p, float t) {
    p.xz = ssRotate2D(p.xz, t);
    return p;
}

static float3 ssPalette(float t, float3 a, float3 b, float3 c, float3 d) {
    return clamp(a + b * cos(6.28318f * (c * t + d)), 0.0f, 1.0f);
}

static float ssNoise2(float2 p) {
    float2 i = floor(p);
    float2 f = fract(p);
    float2 u = f * f * (3.0f - 2.0f * f);
    float a = ssHash31(float3(i, 0.13f));
    float b = ssHash31(float3(i + float2(1.0f, 0.0f), 0.13f));
    float c = ssHash31(float3(i + float2(0.0f, 1.0f), 0.13f));
    float d = ssHash31(float3(i + float2(1.0f, 1.0f), 0.13f));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

static float3 ssProceduralTexture(float2 uv) {
    float value = 0.0f;
    float amp = 0.55f;
    float freq = 1.0f;
    for (int i = 0; i < 4; ++i) {
        value += ssNoise2(uv * freq) * amp;
        freq *= 2.07f;
        amp *= 0.5f;
    }
    float cracks = abs(sin(uv.x * 6.0f) + cos(uv.y * 7.0f));
    float tone = clamp(value * 0.9f + cracks * 0.15f, 0.0f, 1.0f);
    return float3(tone, tone * tone, sqrt(tone));
}

static float3 ssTriplanar(float3 p, float3 n) {
    float3 an = abs(n);
    float sum = max(an.x + an.y + an.z, 1.0e-4f);
    an /= sum;
    float3 c0 = ssProceduralTexture(p.xy).rgb * an.z;
    float3 c1 = ssProceduralTexture(p.yz).rgb * an.x;
    float3 c2 = ssProceduralTexture(p.xz).rgb * an.y;
    return c0 + c1 + c2;
}

static float2 ssBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 ssFaceUV(float3 p) {
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

static float3 ssEnvironment(float3 dir) {
    dir = normalize(dir);
    float skyMix = clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.y), 0.0f), 4.0f);
    float sun = pow(max(dot(dir, normalize(float3(-0.35f, 0.4f, -0.85f))), 0.0f), 44.0f);
    float3 sky = mix(float3(0.05f, 0.01f, 0.04f), float3(0.18f, 0.03f, 0.12f), skyMix);
    return sky + float3(1.0f, 0.42f, 0.18f) * horizon * 0.35f + float3(1.0f, 0.75f, 0.45f) * sun;
}

static float ssSeed(float time) {
    return floor(time * 0.5f);
}

static float ssCrystalSDFSimple(float3 p, float time) {
    float d = 0.0f;
    float seed = ssSeed(time);
    float sides = 6.0f;
    float sideAmpl = SOULSTONE_TAU / sides;

    for (int i = 0; i < 6; ++i) {
        float fi = float(i);
        float angle = mix(fi, fi + 1.0f, ssHash11(fi + seed * 3.17f)) * sideAmpl;
        float3 offset = float3(cos(angle), 0.0f, sin(angle));
        float3 axis = normalize(offset);
        offset = offset * SOULSTONE_CRYSTAL_SCALE / SOULSTONE_VERTICAL_ANISOTROPY;
        d = max(d, dot(p - offset, axis));
    }

    float3 capOffset = float3(0.0f, 2.0f, 0.0f);
    for (int i = 0; i < 6; ++i) {
        float fi = float(i);
        float angle = mix(fi, fi + 1.0f, ssHash11(fi + 17.0f + seed * 2.13f)) * sideAmpl;
        float randomLift = ssHash11(fi + 31.0f + seed * 5.71f);
        float3 axis = normalize(float3(cos(angle), 0.5f + randomLift, sin(angle)));
        d = max(d, dot(p - capOffset * SOULSTONE_CRYSTAL_SCALE * SOULSTONE_VERTICAL_ANISOTROPY, axis));
        d = max(d, dot(p + capOffset * SOULSTONE_CRYSTAL_SCALE * SOULSTONE_VERTICAL_ANISOTROPY, -axis));
    }

    return d;
}

static float ssCurvatureModifier(float3 p, float w, float time) {
    float2 e = float2(-1.0f, 1.0f) * w;
    float t1 = ssCrystalSDFSimple(p + e.yxx, time);
    float t2 = ssCrystalSDFSimple(p + e.xxy, time);
    float t3 = ssCrystalSDFSimple(p + e.xyx, time);
    float t4 = ssCrystalSDFSimple(p + e.yyy, time);
    return (0.25f / e.y) * (t1 + t2 + t3 + t4 - 4.0f * ssCrystalSDFSimple(p, time));
}

static float ssCrystalSDF(float3 p, float time) {
    return ssCrystalSDFSimple(p, time);
}

static float3 ssCrystalNormal(float3 p, float epsilon, float time) {
    float3 eps = float3(epsilon, -epsilon, 0.0f);
    float dX = ssCrystalSDF(p + eps.xzz, time) - ssCrystalSDF(p + eps.yzz, time);
    float dY = ssCrystalSDF(p + eps.zxz, time) - ssCrystalSDF(p + eps.zyz, time);
    float dZ = ssCrystalSDF(p + eps.zzx, time) - ssCrystalSDF(p + eps.zzy, time);
    return normalize(float3(dX, dY, dZ));
}

static float ssDensity(float3 p, float time) {
    float3 p0 = p;
    float3 pp = p + fmod(time, 2.0f) * 0.35f;
    p *= 0.3f;

    for (int i = 0; i < 4; ++i) {
        p = 0.7f * abs(p) / max(dot(p, p), 0.18f) - 0.95f;
        p.yz = ssComplexSquare(p.yz);
        p = p.zxy;
    }

    p = pp + p * 0.5f;
    float d = 0.0f;
    float seed = ssSeed(time);

    for (int i = 0; i < 6; ++i) {
        float fi = float(i);
        float3 hash = ssHash33(float3(fi, seed, 1.0f));
        float3 axis = normalize(hash * float3(2.0f, 4.0f, 2.0f) - float3(1.0f, 1.0f, 1.0f));
        float3 offset = float3(0.0f, ssHash11(fi + 71.0f + seed) * 2.0f - 1.0f, 0.0f);
        float proj = dot(p - offset, axis);
        d += smoothstep(0.1f, 0.0f, abs(proj));
    }

    float pulse = ssSaturate(1.0f - length(p0 * (1.0f + sin(time * 2.0f) * 0.5f)));
    d = d * 0.5f + pulse * (0.75f + d * 0.25f);
    return d * d + 0.05f;
}

static SoulstoneIntersection ssRaymarch(float3 ro, float3 rd, float time) {
    SoulstoneIntersection outData;
    outData.sdf = 0.0f;
    outData.materialID = SOULSTONE_MATERIAL_NONE;
    outData.density = 0.0f;
    outData.totalDistance = 0.0f;
    outData.mediumDistance = 0.0f;
    outData.hit = false;

    for (int j = 0; j < SOULSTONE_MAX_STEPS; ++j) {
        float3 p = ro + rd * outData.totalDistance;
        outData.sdf = ssCrystalSDFSimple(p, time) * 0.9f;
        outData.totalDistance += outData.sdf;

        if (outData.sdf < SOULSTONE_EPSILON || outData.totalDistance > SOULSTONE_MAX_DISTANCE) {
            break;
        }
    }

    if (outData.sdf < SOULSTONE_EPSILON && outData.totalDistance <= SOULSTONE_MAX_DISTANCE) {
        float t = SOULSTONE_FIXED_STEP_SIZE;
        float d = 0.0f;
        float3 hitPosition = ro + rd * (outData.totalDistance + SOULSTONE_FIXED_STEP_SIZE);

        float3 normal = ssCrystalNormal(hitPosition, 1.0f, time);
        float3 refr = refract(rd, normal, 0.9f);
        if (length_squared(refr) < 1.0e-6f) {
            refr = reflect(rd, normal);
        }

        for (int i = 0; i < 50; ++i) {
            float3 p = hitPosition + refr * t;
            if (ssCrystalSDFSimple(p, time) > SOULSTONE_EPSILON) {
                break;
            }

            d += ssDensity(p, time);
            t += SOULSTONE_FIXED_STEP_SIZE;
        }

        outData.density = d;
        outData.materialID = SOULSTONE_MATERIAL_CRYSTAL;
        outData.totalDistance *= 0.99f;
        outData.mediumDistance = t;
        outData.hit = true;
    }

    return outData;
}

static float3 ssGradient(float factor) {
    float3 a = float3(0.478f, 0.45f, 0.5f);
    float3 b = float3(0.5f);
    float3 c = float3(0.1688f, 0.748f, 0.1748f);
    float3 d = float3(0.1318f, 0.388f, 0.1908f);
    return ssPalette(factor, a, b, c, d);
}

fragment float4 soulstoneFragment(
    SoulstoneVertexOut in [[stage_in]],
    constant SoulstoneUniforms &uniforms [[buffer(0)]],
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
    float3 hit = (in.worldPos - center) / cubeScale;
    float3 rd = normalize(hit - eye);

    bool insideBox = all(abs(eye) < SOULSTONE_BOX_HALF - 1.0e-3f);
    float2 tBox = ssBoxIntersect(eye, rd, SOULSTONE_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 ro = (eye + rd * (tStart + SOULSTONE_EPSILON)) * 2.05f;
    ro = ssRotateY(ro, uniforms.time * 0.22f);
    ro = ssRotateX(ro, sin(uniforms.time * 0.31f) * 0.22f);
    float3 marchDir = ssRotateY(rd, uniforms.time * 0.22f);
    marchDir = ssRotateX(marchDir, sin(uniforms.time * 0.31f) * 0.22f);

    SoulstoneIntersection isect = ssRaymarch(ro, marchDir, uniforms.time);
    if (isect.materialID > 0) {
        float3 p = ro + marchDir * isect.totalDistance;
        float3 lightPos = ro - camRight * 2.0f + camUp * 2.0f;
        float3 normal = ssCrystalNormal(p, SOULSTONE_EPSILON_NORMAL, uniforms.time);
        float3 toLight = normalize(lightPos - p);

        float3 tx = ssTriplanar(p * 0.85f - p.zzz * 0.3f, normal);
        float curvature = ssCurvatureModifier(p, 0.1f + tx.r * 0.85f, uniforms.time);
        normal = normalize(normal - float3(curvature * 0.3f) + (tx * 0.25f - 0.125f));

        float rim = pow(smoothstep(0.0f, 1.0f, 1.0f - dot(normal, -marchDir)), 7.0f);
        float3 H = normalize(toLight - marchDir);
        float specular = pow(max(0.0f, dot(H, normal)), tx.r * 5.0f + curvature * 25.0f);

        float3 reflected = reflect(marchDir, normal);
        float3 refl = ssEnvironment(reflected);

        float glowFactor = isect.density * 0.04f;
        float3 glow = mix(float3(1.0f, 0.15f, 0.15f), float3(1.0f, 0.45f, 0.15f), isect.density * 0.05f) * glowFactor;
        glow *= smoothstep(0.5f, 1.0f, curvature) * 1.5f + 1.0f;
        glow *= 1.0f + pow(exp(-isect.mediumDistance), 2.0f) * 4.0f;

        float transmission = exp(-isect.mediumDistance * 0.35f);
        float3 soulGlow = ssGradient(isect.density * 0.04f + transmission * 0.2f) * isect.density * 0.018f;
        float2 glowUV = ssFaceUV(hit) * 2.0f - 1.0f;
        glowUV.y += sin(uniforms.time * 2.0f) * 0.1f;
        float3 glowColor = float3(1.0f, 0.7f, 0.15f);
        float2 scaled = glowUV * 0.7f;
        float3 fx = glowColor * pow(ssSaturate(1.0f - length(scaled * float2(0.75f, 0.9f))), 2.0f);
        fx += glowColor * pow(ssSaturate(1.0f - length(scaled * float2(0.5f, 1.0f))), 2.0f);
        fx += glowColor * pow(ssSaturate(1.0f - length(scaled * float2(0.25f, 7.0f))), 2.0f) * 0.25f;
        fx += glowColor * pow(ssSaturate(1.0f - length(scaled * float2(0.1f, 7.0f))), 2.0f) * 0.15f;
        float intensity = pow(ssProceduralTexture(float2(uniforms.time * 0.03f, 0.0f)).r, 2.0f);

        float3 color = (refl + specular) * float3(0.15f, 0.1f, 0.1f) * rim;
        color += rim * curvature * 0.15f * float3(0.1f, 0.4f, 0.8f);
        color += glow + soulGlow + fx * fx * fx * intensity * 0.2f;
        return float4(clamp(color, 0.0f, 1.0f), 1.0f);
    }

    float2 uv = ssFaceUV(hit) * 2.0f - 1.0f;
    float vignette = 1.0f - pow(length(uv + ssHash31(ro + float3(uniforms.time)) * 0.2f) / 2.0f, 2.0f);
    float3 bg = float3(0.15f, 0.025f, 0.1f) * vignette * vignette * 0.25f;
    bg += ssEnvironment(marchDir) * 0.18f;
    return float4(clamp(bg, 0.0f, 1.0f), 1.0f);
}