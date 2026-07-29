// CloudyCrystalShaders.metal
// "Cloudy crystal" — cube-portal adaptation of Shadertoy "fdlSDl"
// Original: https://www.shadertoy.com/view/fdlSDl
// License in source: CC0
//
// Metal adaptation notes:
// - The original GLSL is a screen-space effect with its own synthetic camera.
// - This version replaces that camera with the real per-eye world ray coming
//   from the cube surface (or the viewer position when inside the cube).
// - The crystal scene is fixed in scene space around the cube centre, so user
//   head motion produces stereo parallax and positional motion instead of a
//   flat 2D image on the cube wall.
// - Unlike the original post-process, this version intentionally omits the
//   screen-space vignette because it would reintroduce a 2D overlay artifact.

#include <metal_stdlib>
using namespace metal;

struct CloudyCrystalUniforms {
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

struct CloudyCrystalVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float CC_PI = 3.141592654f;
static constant float CC_MISS = 1.0e4f;
static constant float CC_REFRACT_INDEX = 0.85f;
static constant float3 CC_LIGHT_POS = 2.0f * float3(1.5f, 2.0f, 1.0f);
static constant float3 CC_SUN_COL = float3(8.0f, 7.0f, 6.0f) / 8.0f;

vertex CloudyCrystalVertexOut cloudyCrystalVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant CloudyCrystalUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    CloudyCrystalVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float ccTanhApprox(float x) {
    float x2 = x * x;
    return clamp(x * (27.0f + x2) / (27.0f + 9.0f * x2), -1.0f, 1.0f);
}

static float3 ccHsv2Rgb(float3 c) {
    const float4 K = float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0f - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), c.y);
}

static float ccL2(float3 x) {
    return dot(x, x);
}

// IQ quartic sphere intersector translated directly from GLSL.
static float ccRaySphere4(float3 ro, float3 rd, float ra) {
    float r2 = ra * ra;
    float3 d2 = rd * rd;
    float3 d3 = d2 * rd;
    float3 o2 = ro * ro;
    float3 o3 = o2 * ro;
    float ka = 1.0f / dot(d2, d2);
    float k3 = ka * dot(ro, d3);
    float k2 = ka * dot(o2, d2);
    float k1 = ka * dot(o3, rd);
    float k0 = ka * (dot(o2, o2) - r2 * r2);
    float c2 = k2 - k3 * k3;
    float c1 = k1 + 2.0f * k3 * k3 * k3 - 3.0f * k3 * k2;
    float c0 = k0 - 3.0f * k3 * k3 * k3 * k3 + 6.0f * k3 * k3 * k2 - 4.0f * k3 * k1;
    float p = c2 * c2 + c0 / 3.0f;
    float q = c2 * c2 * c2 - c2 * c0 + c1 * c1;
    float h = q * q - p * p * p;
    if (h < 0.0f) {
        return CC_MISS;
    }
    float sh = sqrt(h);
    float s = sign(q + sh) * pow(abs(q + sh), 1.0f / 3.0f);
    float t = sign(q - sh) * pow(abs(q - sh), 1.0f / 3.0f);
    float2 w = float2(s + t, s - t);
    float2 v = float2(w.x + c2 * 4.0f, w.y * sqrt(3.0f)) * 0.5f;
    float r = length(v);
    return -abs(v.y) / sqrt(max(r + v.x, 1.0e-6f)) - c1 / max(r, 1.0e-6f) - k3;
}

static float3 ccSphere4Normal(float3 pos) {
    return normalize(pos * pos * pos);
}

static float ccIRaySphere4(float3 ro, float3 rd, float ra) {
    float3 rro = ro + rd * ra * 4.0f;
    float3 rrd = -rd;
    float rt = ccRaySphere4(rro, rrd, ra);
    if (rt == CC_MISS) {
        return CC_MISS;
    }
    float3 rpos = rro + rrd * rt;
    return length(rpos - ro);
}

static float ccRayPlane(float3 ro, float3 rd, float4 p) {
    return -(dot(ro, p.xyz) + p.w) / dot(rd, p.xyz);
}

static float2 ccCSqr(float2 a) {
    return float2(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y);
}

static float3x3 ccRotX(float a) {
    float s = sin(a);
    float c = cos(a);
    return float3x3(float3(1.0f, 0.0f, 0.0f), float3(0.0f, c, -s), float3(0.0f, s, c));
}

static float3x3 ccRotY(float a) {
    float s = sin(a);
    float c = cos(a);
    return float3x3(float3(c, 0.0f, s), float3(0.0f, 1.0f, 0.0f), float3(-s, 0.0f, c));
}

static float ccMarbleDf(float3 p) {
    float res = 0.0f;
    float3 c = p;
    float scale = 0.72f;
    constexpr int maxIter = 10;
    for (int i = 0; i < maxIter; ++i) {
        p = scale * abs(p) / dot(p, p) - scale;
        p.yz = ccCSqr(p.yz);
        p = p.zxy;
        res += exp(-19.0f * abs(dot(p, c)));
    }
    return res;
}

static float3 ccMarbleMarch(float3 ro, float3 rd, float2 tminmax) {
    float t = tminmax.x;
    float dt = 0.02f;
    float3 col = float3(0.0f);
    float c = 0.0f;
    constexpr int maxIter = 64;
    for (int i = 0; i < maxIter; ++i) {
        t += dt * exp(-2.0f * c);
        if (t > tminmax.y) {
            break;
        }
        float3 pos = ro + t * rd;
        c = ccMarbleDf(pos);
        c *= 0.5f;
        float dist = abs(pos.x + pos.y - 0.15f) * 10.0f;
        float3 dcol = float3(c * c * c - c * dist, c * c - c, c);
        col += dcol;
    }
    float scale = 0.005f;
    float td = (t - tminmax.x) / max(tminmax.y - tminmax.x, 1.0e-4f);
    col *= exp(-10.0f * td);
    col *= scale;
    return col;
}

static float3 ccSkyColor(float3 ro, float3 rd) {
    const float3 skyCol1 = pow(float3(0.2f, 0.4f, 0.6f), float3(0.25f));
    const float3 skyCol2 = pow(float3(0.4f, 0.7f, 1.0f), float3(2.0f));
    float3 sunDir = normalize(CC_LIGHT_POS);
    float sunDot = max(dot(rd, sunDir), 0.0f);
    float3 final = float3(0.0f);

    final += mix(skyCol1, skyCol2, rd.y);
    final += 0.5f * CC_SUN_COL * pow(sunDot, 20.0f);
    final += 4.0f * CC_SUN_COL * pow(sunDot, 400.0f);

    float tp = ccRayPlane(ro, rd, float4(float3(0.0f, 1.0f, 0.0f), 0.505f));
    if (tp > 0.0f) {
        float3 pos = ro + tp * rd;
        float3 ld = normalize(CC_LIGHT_POS - pos);
        float ts4 = ccRaySphere4(pos, ld, 0.5f);
        float3 spos = pos + ld * ts4;
        float its4 = ccIRaySphere4(spos, ld, 0.5f);
        float sha = ts4 == CC_MISS ? 1.0f : (1.0f - ccTanhApprox(its4 * 1.5f / (0.5f + 0.5f * ts4)));
        float3 nor = float3(0.0f, 1.0f, 0.0f);
        float3 icol = 1.5f * skyCol1 + 4.0f * CC_SUN_COL * sha * dot(-rd, nor);
        float2 ppos = pos.xz * 0.75f;
        ppos = fract(ppos + 0.5f) - 0.5f;
        float pd = min(abs(ppos.x), abs(ppos.y));
        float3 pcol = mix(float3(0.4f), float3(0.3f), exp(-60.0f * pd));

        float3 col = clamp(icol * pcol, 0.0f, 1.25f);
        float f = exp(-10.0f * (max(tp - 10.0f, 0.0f) / 100.0f));
        return mix(final, col, f);
    }
    return final;
}

static float3 ccRender1(float3 ro, float3 rd) {
    float its4 = ccIRaySphere4(ro, rd, 0.5f);
    return ccMarbleMarch(ro, rd, float2(0.0f, its4));
}

static float3 ccRender(float3 ro, float3 rd) {
    float3 skyCol = ccSkyColor(ro, rd);
    float ts4 = ccRaySphere4(ro, rd, 0.5f);
    if (ts4 >= CC_MISS) {
        return skyCol;
    }

    float3 pos = ro + ts4 * rd;
    float3 nor = ccSphere4Normal(pos);
    float3 refr = refract(rd, nor, CC_REFRACT_INDEX);
    float3 refl = reflect(rd, nor);
    float3 rcol = ccSkyColor(pos, refl);
    float fre = mix(0.0f, 1.0f, pow(1.0f - dot(-rd, nor), 4.0f));

    float3 lv = CC_LIGHT_POS - pos;
    float ll2 = ccL2(lv);
    float ll = sqrt(ll2);
    float3 ld = lv / ll;

    float dm = min(1.0f, 40.0f / ll2);
    float dif = pow(max(dot(nor, ld), 0.0f), 8.0f) * dm;
    float spe = pow(max(dot(reflect(-ld, nor), -rd), 0.0f), 100.0f);
    float lin = mix(0.0f, 1.0f, dif);
    float3 lcol = 2.0f * sqrt(CC_SUN_COL);

    float3 col = ccRender1(pos, refr);
    float3 diff = ccHsv2Rgb(float3(0.7f, fre, 0.075f * lin)) * lcol;
    col += fre * rcol + diff + spe * lcol;
    if (all(refr == float3(0.0f))) {
        col = float3(1.0f, 0.0f, 0.0f);
    }
    return col;
}

// The cube itself is only the portal. The crystal scene is centred at the
// cube origin and is not clipped to the cube bounds.
fragment float4 cloudyCrystalFragment(
    CloudyCrystalVertexOut in [[stage_in]],
    constant CloudyCrystalUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];
    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    // Scene space is cube-centred. cubeScale = 1.0, but keep the transform
    // explicit so the renderer remains consistent with other cube portal demos.
    float sceneScale = max(uniforms.cubeScale, 1.0e-4f);
    float3 roScene = (camWorld - center) / sceneScale;
    float3 rdScene = normalize(in.worldPos - camWorld);
    float3 surfaceScene = (in.worldPos - center) / sceneScale;

    bool insideBox = all(abs(roScene) < float3(0.999f));
    float3 marchOrigin = insideBox ? (roScene + rdScene * 0.002f) : (surfaceScene + rdScene * 0.002f);

    // Replace the original screen-space orbit camera with a scene that is fixed
    // in world space. A mild local-space rotation preserves the source's motion
    // without tying image structure to the viewer.
    float3x3 sceneRot = ccRotY(CC_PI * 0.5f + sin(uniforms.time * 0.05f))
                      * ccRotX(0.5f + 0.125f * sin(uniforms.time * 0.05f * sqrt(0.5f)));
    float3 localRo = sceneRot * marchOrigin;
    float3 localRd = normalize(sceneRot * rdScene);

    float3 col = ccRender(localRo, localRd);

    // Post-process from the source, minus the screen-space vignette.
    col = clamp(col, 0.0f, 1.0f);
    col = pow(col, float3(1.0f / 2.2f));
    col = col * 0.6f + 0.4f * col * col * (3.0f - 2.0f * col);
    col = mix(col, float3(dot(col, float3(0.33f))), -0.4f);

    return float4(col, 1.0f);
}
