// PlatonicMirrorShaders.metal
//
// Source: "Let's self reflect" by mrange
// https://www.shadertoy.com/view/XfyXRV
// License: CC0
//
// VR adaptation: UV-sphere mesh container; fragment shader reconstructs a world-space
// ray from the actual VR camera position, transforms it to solid-local space, then runs
// the full Platonic-solid ray-march with internal mirror reflections.
//
// Original poly-fold technique by knighty: https://www.shadertoy.com/view/MsKGzw

#include <metal_stdlib>
using namespace metal;

// ─── Uniforms ─────────────────────────────────────────────────────────────────
// Layout must match PlatonicMirrorUniforms in PlatonicMirrorTypes.swift.
struct PlatonicMirrorUniforms {
    float  time;
    uint   viewCount;
    float  solidScale;   // world metres per local unit
    float  _pad;
    float4 objectCenter; // xyz = world position
};

struct MeshVertex { float3 position; float3 normal; };

struct PlatonicVertexOut {
    float4 clipPos  [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ─── Vertex shader ────────────────────────────────────────────────────────────
vertex PlatonicVertexOut platonicMirrorVertex(
    ushort                          amplificationID [[amplification_id]],
    const device MeshVertex        *vertices        [[buffer(0)]],
    constant PlatonicMirrorUniforms &uniforms        [[buffer(1)]],
    constant float4x4              *vpMatrices       [[buffer(2)]],
    uint                            vertexID         [[vertex_id]])
{
    MeshVertex vtx  = vertices[vertexID];
    uint viewIndex  = min((uint)amplificationID, uniforms.viewCount - 1u);
    // vtx.position is in local units; scale to world space
    float3 worldPos = vtx.position * uniforms.solidScale + uniforms.objectCenter.xyz;

    PlatonicVertexOut out;
    out.clipPos  = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ─── Tunable constants ────────────────────────────────────────────────────────
#define PM_PI           3.141592654f
#define ROTATION_SPEED  0.25f

// Polyhedron parameters (matching the original ShaderToy defaults)
#define POLY_TYPE    3        // [2,5]
#define POLY_U       1.0f
#define POLY_V       0.5f
#define POLY_W       1.0f
#define POLY_ZOOM    2.0f
#define INNER_SPHERE 1.0f
#define REFR_INDEX   0.9f
#define RREFR_INDEX  (1.0f / REFR_INDEX)
#define PM_BOUND_RADIUS 3.1f

// Ray-march budgets
#define MAX_BOUNCES2   4      // increased for deeper internal mirror recursion
#define MAX_MARCHES2   16     // slightly higher to support the extra bounces
#define TOLERANCE2     0.002f
#define NORM_OFF2      0.007f
#define MAX_MARCHES3   24     // allow more iterations for distant views
#define TOLERANCE3     0.002f
#define NORM_OFF3      0.007f

// ─── Context (replaces GLSL mutable globals g_rot / g_gd) ────────────────────
struct PlatonicCtx {
    float3x3 g_rot;   // per-frame rotation applied inside the solid
    float2   g_gd;    // accumulated minimum (edge_dist, corner_or_inner_dist)
    // Precomputed poly-fold constants (depend only on POLY_TYPE, U, V, W)
    float3   poly_nc;
    float3   poly_pab;
    float3   poly_pbc;  // normalised
    float3   poly_pca;  // normalised
    float3   poly_p;    // normalised blended vertex direction
};

// ─── Utilities ────────────────────────────────────────────────────────────────
// License: WTFPL, author: sam hocevar
inline float3 pm_hsv2rgb(float3 c) {
    float4 K = float4(1.0f, 2.0f/3.0f, 1.0f/3.0f, 3.0f);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0f - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), c.y);
}

// License: MIT, author: Inigo Quilez — rotation matrix aligning z → d
inline float3x3 pm_buildRot(float3 d, float3 z) {
    float3 v = cross(z, d);
    float  c = dot(z, d);
    if (c < -0.9999f) {
        return float3x3(float3(1,0,0), float3(0,-1,0), float3(0,0,-1));
    }
    float k = 1.0f / (1.0f + c);
    return float3x3(
        float3(v.x*v.x*k + c,   v.y*v.x*k - v.z, v.z*v.x*k + v.y),
        float3(v.x*v.y*k + v.z, v.y*v.y*k + c,   v.z*v.y*k - v.x),
        float3(v.x*v.z*k - v.y, v.y*v.z*k + v.x, v.z*v.z*k + c  )
    );
}

// License: Unknown, author: Matt Taylor — ACES filmic tone-map approximation
inline float3 pm_aces(float3 v) {
    v = max(v, 0.0f); v *= 0.6f;
    const float a=2.51f, b=0.03f, c=2.43f, d=0.59f, e=0.14f;
    return clamp((v*(a*v+b))/(v*(c*v+d)+e), 0.0f, 1.0f);
}

inline float pm_box2(float2 p, float2 b) {
    float2 dd = abs(p) - b;
    return length(max(dd, 0.0f)) + min(max(dd.x, dd.y), 0.0f);
}

// ─── Polyhedron fold (License: Unknown, author: knighty) ─────────────────────
inline void poly_fold(thread float3 &pos, float3 nc) {
    for (int i = 0; i < POLY_TYPE; ++i) {
        pos.xy = abs(pos.xy);
        pos -= 2.0f * min(0.0f, dot(pos, nc)) * nc;
    }
}

// Evaluate the three shape SDF components: (plane, edge, corner)
inline float3 pm_shape(float3 pos, thread const PlatonicCtx &ctx) {
    pos = pos * ctx.g_rot;     // row-vector × matrix (same as GLSL pos *= g_rot)
    pos /= POLY_ZOOM;
    poly_fold(pos, ctx.poly_nc);
    pos -= ctx.poly_p;

    // poly_plane: maximum of three half-space signed distances
    float d0 = max(max(dot(pos, ctx.poly_pab),
                       dot(pos, ctx.poly_pbc)),
                       dot(pos, ctx.poly_pca));

    // poly_edge: distance to the three fold edges
    float3 pa = pos - min(0.0f, pos.x)                  * float3(1,0,0);
    float3 pb = pos - min(0.0f, pos.y)                  * float3(0,1,0);
    float3 pc = pos - min(0.0f, dot(pos, ctx.poly_nc)) * ctx.poly_nc;
    float d1 = sqrt(min(min(dot(pa,pa), dot(pb,pb)), dot(pc,pc))) - 2e-3f;

    // poly_corner: small sphere at folded-space origin
    float d2 = length(pos) - 0.0125f;

    return float3(d0, d1, d2) * POLY_ZOOM;
}

// ─── Distance fields ──────────────────────────────────────────────────────────
// df2: used inside the solid for bounce ray-marching
static float pm_df2(float3 p, thread PlatonicCtx &ctx) {
    float3 ds = pm_shape(p, ctx);
    float d2  = ds.y - 5e-3f;
    float d0  = min(-ds.x, d2);            // inside the solid OR near edge
    float d1  = length(p) - INNER_SPHERE;  // inner sphere
    ctx.g_gd  = min(ctx.g_gd, float2(d2, d1));
    return min(d0, d1);
}

// df3: used outside the solid for the primary ray-march
static float pm_df3(float3 p, thread PlatonicCtx &ctx) {
    float3 ds = pm_shape(p, ctx);
    ctx.g_gd  = min(ctx.g_gd, ds.yz);
    return min(min(ds.x, ds.y), ds.z);
}

// ─── Normal estimation ────────────────────────────────────────────────────────
static float3 pm_normal2(float3 pos, thread PlatonicCtx &ctx) {
    const float2 eps = float2(NORM_OFF2, 0.0f);
    float3 n;
    n.x = pm_df2(pos+eps.xyy,ctx) - pm_df2(pos-eps.xyy,ctx);
    n.y = pm_df2(pos+eps.yxy,ctx) - pm_df2(pos-eps.yxy,ctx);
    n.z = pm_df2(pos+eps.yyx,ctx) - pm_df2(pos-eps.yyx,ctx);
    return normalize(n);
}

static float3 pm_normal3(float3 pos, thread PlatonicCtx &ctx) {
    const float2 eps = float2(NORM_OFF3, 0.0f);
    float3 n;
    n.x = pm_df3(pos+eps.xyy,ctx) - pm_df3(pos-eps.xyy,ctx);
    n.y = pm_df3(pos+eps.yxy,ctx) - pm_df3(pos-eps.yxy,ctx);
    n.z = pm_df3(pos+eps.yyx,ctx) - pm_df3(pos-eps.yyx,ctx);
    return normalize(n);
}

// ─── Ray marchers ─────────────────────────────────────────────────────────────
static float pm_rayMarch2(float3 ro, float3 rd, float tinit, thread PlatonicCtx &ctx) {
    float t = tinit;
    float2 dti = float2(1e10f, 0.0f);  // backstep: (min_d, t_at_min_d)
    int i;
    for (i = 0; i < MAX_MARCHES2; ++i) {
        float d = pm_df2(ro + rd*t, ctx);
        if (d < dti.x) { dti = float2(d, t); }
        if (d < TOLERANCE2) break;
        t += d;
    }
    if (i == MAX_MARCHES2) { t = dti.y; }
    return t;
}

static float pm_rayMarch3(float3 ro, float3 rd, float tinit, float maxRayLen,
                          thread PlatonicCtx &ctx, thread int &iter) {
    float t = tinit;
    int i;
    for (i = 0; i < MAX_MARCHES3; ++i) {
        float d = pm_df3(ro + rd*t, ctx);
        if (d < TOLERANCE3 || t > maxRayLen) break;
        t += d;
    }
    iter = i;
    return t;
}

static bool pm_sphereHit(float3 ro, float3 rd, float radius,
                         thread float &tNear, thread float &tFar) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) {
        return false;
    }

    float root = sqrt(h);
    tNear = -b - root;
    tFar = -b + root;
    return tFar >= max(tNear, 0.0f);
}

// ─── Procedural background environment ───────────────────────────────────────
// sunDir = normalize(-rayOrigin) where original rayOrigin = (0,1,-5)
static constant float3 PM_SUN_DIR = float3(0.0f, -0.19612f, 0.98058f); // normalize(0,-1,5)

static float3 pm_render0(float3 ro, float3 rd,
                         float3 sunCol, float3 botCol, float3 topCol)
{
    float3 col = float3(0);
    float srd  = sign(rd.y);
    float tp   = -(ro.y - 6.0f) / (abs(rd.y) + 1e-6f);

    if (srd < 0.0f) {
        col += botCol * exp(-0.5f * length((ro + tp*rd).xz));
    }
    if (srd > 0.0f) {
        float3 pos = ro + tp*rd;
        float2 pp  = pos.xz;
        float db   = pm_box2(pp, float2(5.0f, 9.0f)) - 3.0f;
        col += topCol * rd.y*rd.y * smoothstep(0.25f, 0.0f, db);
        col += 0.2f  * topCol * exp(-0.5f * max(db, 0.0f));
        col += 0.05f * sqrt(topCol) * max(-db, 0.0f);
    }
    col += sunCol / (1.001f - dot(PM_SUN_DIR, rd));
    return col;
}

// ─── Internal bounce reflections (render2) ───────────────────────────────────
static float3 pm_render2(float3 ro, float3 rd, float db, thread PlatonicCtx &ctx,
                         float3 sunCol, float3 botCol, float3 topCol,
                         float3 glowCol1, float3 beerCol)
{
    float3 agg  = float3(0);
    float  ragg = 1.0f;
    float  tagg = 0.0f;

    for (int bounce = 0; bounce < MAX_BOUNCES2; ++bounce) {
        if (ragg < 0.1f) break;

        ctx.g_gd    = float2(1e3f);
        float  t2   = pm_rayMarch2(ro, rd, min(db + 0.05f, 0.3f), ctx);
        float2 gd2  = ctx.g_gd;
        tagg       += t2;

        float3 p2  = ro + rd*t2;
        float3 n2  = pm_normal2(p2, ctx);
        float3 r2  = reflect(rd, n2);
        float3 rr2 = refract(rd, n2, RREFR_INDEX);
        float fre2 = 1.0f + dot(n2, rd);

        float3 beer = ragg * exp(0.2f * beerCol * tagg);

        // Edge-proximity glow
        float denom = max(gd2.x, 5e-4f + tagg*tagg * 2e-4f / max(ragg, 1e-6f));
        agg += glowCol1 * beer * ((1.0f + tagg*tagg*4e-2f) * 6.0f / denom);

        float3 ocol = 0.2f * beer * pm_render0(p2, rr2, sunCol, botCol, topCol);
        if (gd2.y <= TOLERANCE2) {
            ragg *= 1.0f - 0.9f * fre2;   // hit inner sphere: partial absorption
        } else {
            agg  += ocol;
            ragg *= 0.8f;
        }
        ro = p2;
        rd = r2;
        db = gd2.x;
    }
    return agg;
}

// ─── Outer-surface render (render3) ──────────────────────────────────────────
static float3 pm_render3(float3 ro, float3 rd, float maxRayLen, thread PlatonicCtx &ctx) {
    // Colour constants (matching original ShaderToy verbatim)
    float3 sunCol   = pm_hsv2rgb(float3(0.06f, 0.90f, 1e-2f));
    float3 botCol   = pm_hsv2rgb(float3(0.66f, 0.80f, 0.5f));
    float3 topCol   = pm_hsv2rgb(float3(0.60f, 0.90f, 1.0f));
    float3 glowCol0 = pm_hsv2rgb(float3(0.05f, 0.7f,  1e-3f));
    float3 glowCol1 = pm_hsv2rgb(float3(0.95f, 0.7f,  1e-3f));
    float3 beerCol  = -pm_hsv2rgb(float3(0.65f, 0.7f, 2.0f));  // negative → absorption

    float3 skyCol = pm_render0(ro, rd, sunCol, botCol, topCol);
    float3 col    = skyCol;

    ctx.g_gd = float2(1e3f);
    int   iter;
    float t1   = pm_rayMarch3(ro, rd, 0.1f, maxRayLen, ctx, iter);
    float2 gd1 = ctx.g_gd;

    float3 p1  = ro + t1*rd;
    float  fre1 = 0.0f;  // default for miss

    // Smooth-step fade for rays near the march limit (silhouette anti-alias)
    float ifo = mix(0.5f, 1.0f,
                    smoothstep(1.0f, 0.9f, float(iter) / float(MAX_MARCHES3)));

    if (t1 < maxRayLen) {
        // Only compute expensive derivatives on hit fragments
        float3 n1  = pm_normal3(p1, ctx);
        float3 r1  = reflect(rd, n1);
        float3 rr1 = refract(rd, n1, REFR_INDEX);
        fre1 = 1.0f + dot(rd, n1);
        fre1 *= fre1;

        col = pm_render0(p1, r1, sunCol, botCol, topCol) * (0.5f + 0.5f*fre1) * ifo;

        if (gd1.x > TOLERANCE3 && gd1.y > TOLERANCE3 && length(rr1) > 1e-4f) {
            float3 icol = pm_render2(p1, rr1, gd1.x, ctx,
                                     sunCol, botCol, topCol, glowCol1, beerCol);
            col += icol * (1.0f - 0.75f*fre1) * ifo;
        }
    }

    // Outer edge-proximity glow (safe for miss: gd1.x == 1e3, fre1 == 0)
    col += (glowCol0 + 1.0f*fre1*glowCol0) / max(gd1.x, 3e-4f);
    return col;
}

// ─── Fragment shader ──────────────────────────────────────────────────────────
fragment float4 platonicMirrorFragment(
    PlatonicVertexOut               in       [[stage_in]],
    constant PlatonicMirrorUniforms &uniforms [[buffer(0)]],
    constant float4x4               *v2wMats  [[buffer(1)]],
    constant float4x4               *vpMats   [[buffer(2)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w    = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center = uniforms.objectCenter.xyz;
    float  sc     = uniforms.solidScale;

    // Ray in solid-local space.  Direction is scale-invariant for uniform scale.
    float3 ro = (camWorld - center) / sc;
    float3 rd = normalize(in.worldPos - camWorld);

    // Cull back-hemisphere fragments: the sphere mesh is convex, so any fragment
    // whose outward normal faces away from the camera is occluded by the front face.
    // discard_fragment() avoids the full ray-march for ~half the sphere pixels.
    float3 fragNormal = normalize(in.worldPos - center);
    if (dot(rd, fragNormal) > 0.0f) { discard_fragment(); }

    // Per-frame rotation (matches mainImage in the original ShaderToy)
    float  a   = uniforms.time * ROTATION_SPEED;
    float  sq2 = sqrt(0.5f);
    float3 r0  = float3(1.0f,             sin(sq2*a),          sin(a));
    float3 r1  = float3(cos(sq2*0.913f*a), cos(0.913f*a),      1.0f);

    // Precompute poly-fold constants (depend only on compile-time parameters)
    float cospin  = cos(PM_PI / float(POLY_TYPE));
    float scospin = sqrt(max(0.75f - cospin*cospin, 0.0f));
    float3 nc     = float3(-0.5f, -cospin, scospin);
    float3 pab    = float3(0.0f, 0.0f, 1.0f);
    float3 pbc_   = float3(scospin, 0.0f, 0.5f);
    float3 pca_   = float3(0.0f, scospin, cospin);

    PlatonicCtx ctx;
    ctx.g_rot    = pm_buildRot(normalize(r0), normalize(r1));
    ctx.g_gd     = float2(1e3f);
    ctx.poly_nc  = nc;
    ctx.poly_pab = pab;
    ctx.poly_pbc = normalize(pbc_);
    ctx.poly_pca = normalize(pca_);
    ctx.poly_p   = normalize(POLY_U*pab + POLY_V*pbc_ + POLY_W*pca_);

    float sphereEntry;
    float sphereExit;
    float primaryMaxRayLen = 8.0f;
    if (pm_sphereHit(ro, rd, PM_BOUND_RADIUS, sphereEntry, sphereExit)) {
        primaryMaxRayLen = max(sphereExit + 0.35f, 1.5f);
    }
    float3 col = pm_render3(ro, rd, primaryMaxRayLen, ctx);

    // ACES filmic tone-mapping + gamma
    col = pm_aces(col);
    col = sqrt(max(col, 0.0f));
    return float4(col, 1.0f);
}
