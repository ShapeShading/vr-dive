// GlassBoxShaders.metal
// Adapted from ShaderToy "NslGRN" by Danil (2021+), CC BY-NC-SA 3.0.
// https://www.shadertoy.com/view/NslGRN
//
// Renders a glass box with bilinear-patch internal structure using ray marching.
// A bounding sphere mesh (built by GlassBoxRenderer.swift) is the container.
// The vertex shader transforms each sphere vertex to world space.
// The fragment shader reconstructs a world-space ray and runs the full glass-box
// ray-marching pipeline in box-local space.

#include <metal_stdlib>
using namespace metal;

// ─── Uniforms ─────────────────────────────────────────────────────────────────
// Layout must match GlassBoxUniforms in GlassBoxTypes.swift.
struct GlassBoxUniforms {
    float  time;
    uint   viewCount;
    float  boxScale;
    float  _pad;
    float4 objectCenter;  // xyz = world-space box centre
};

struct MeshVertex {
    float3 position;
    float3 normal;
};

struct GlassBoxVertexOut {
    float4 clipPos   [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

// ─── Vertex shader ────────────────────────────────────────────────────────────
vertex GlassBoxVertexOut glassBoxVertex(
    ushort                     amplificationID [[amplification_id]],
    const device MeshVertex   *vertices        [[buffer(0)]],
    constant GlassBoxUniforms &uniforms        [[buffer(1)]],
    constant float4x4         *vpMatrices      [[buffer(2)]],
    uint                       vertexID        [[vertex_id]])
{
    MeshVertex vtx  = vertices[vertexID];
    uint viewIndex  = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    GlassBoxVertexOut out;
    out.clipPos   = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos  = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

// ─── Constants ────────────────────────────────────────────────────────────────
#define GB_PI      3.14159265f
#define GB_BOXDIMS float3(0.75f, 0.75f, 1.25f)
#define GB_IOR     1.33f

// ─── Rotation matrices (column-major, same as GLSL) ──────────────────────────
static float3x3 gb_rotx(float a) {
    float s = sin(a), c = cos(a);
    return float3x3(float3(1,0,0), float3(0,c,s), float3(0,-s,c));
}
static float3x3 gb_roty(float a) {
    float s = sin(a), c = cos(a);
    return float3x3(float3(c,0,s), float3(0,1,0), float3(-s,0,c));
}
static float3x3 gb_rotz(float a) {
    float s = sin(a), c = cos(a);
    return float3x3(float3(c,s,0), float3(-s,c,0), float3(0,0,1));
}

// ─── Color palette (adapted from ShaderToy getColor) ─────────────────────────
static float3 gb_getColor(float3 p, float t_time) {
    p = abs(p);
    p *= 1.25f;
    float dp = dot(p, p);
    if (dp < 1e-8f) return float3(0.3f, 0.4f, 0.5f);
    p = 0.5f * p / dp;
    p += 0.072f * t_time;   // continuous color animation
    float t = 0.13f * length(p);
    float3 col = float3(0.3f, 0.4f, 0.5f);
    col += 0.12f * cos(6.28318f * t * 1.0f   + float3(0.0f, 0.8f, 1.1f));
    col += 0.11f * cos(6.28318f * t * 3.1f   + float3(0.3f, 0.4f, 0.1f));
    col += 0.10f * cos(6.28318f * t * 5.1f   + float3(0.1f, 0.7f, 1.1f));
    col += 0.10f * cos(6.28318f * t * 17.1f  + float3(0.2f, 0.6f, 0.7f));
    col += 0.10f * cos(6.28318f * t * 31.1f  + float3(0.1f, 0.6f, 0.7f));
    col += 0.10f * cos(6.28318f * t * 65.1f  + float3(0.0f, 0.5f, 0.8f));
    col += 0.10f * cos(6.28318f * t * 115.1f + float3(0.1f, 0.4f, 0.7f));
    col += 0.10f * cos(6.28318f * t * 265.1f + float3(1.1f, 1.4f, 2.7f));
    return clamp(col, 0.0f, 1.0f);
}

// ─── calcColor ────────────────────────────────────────────────────────────────
static void gb_calcColor(
    float3 ro, float3 rd, float d, float len, bool si, float td, float t_time,
    thread float4 &colx, thread float4 &colsi)
{
    float3 pos = ro + rd * d;
    float a = 1.0f - smoothstep(len - 0.075f, len + 1e-5f, length(pos));
    colx = float4(gb_getColor(pos, t_time), a);
    if (si) {
        pos = ro + rd * td;
        float ta = 1.0f - smoothstep(len - 0.075f, len + 1e-5f, length(pos));
        colsi = float4(gb_getColor(pos, t_time), ta);
    }
}

// ─── Bilinear-patch intersection ──────────────────────────────────────────────
// Direct translation of iBilinearPatch from the original GLSL.
// Polynomial coefficients are simplified using a=b=c=e=f=0.
static bool gb_bilinearPatch(
    float3 ro, float3 rd, float4 ps, float4 ph, float sz,
    thread float  &t,     thread float3 &norm,
    thread bool   &si,    thread float  &tsi,   thread float3 &normsi,
    thread float  &fade,  thread float  &fadesi)
{
    float3 va = float3(0.0f, 0.0f, ph.x + ph.w - ph.y - ph.z);
    float3 vb = float3(0.0f, ps.w - ps.y, ph.z - ph.x);
    float3 vc = float3(ps.z - ps.x, 0.0f, ph.y - ph.x);
    float3 vd = float3(ps.xy, ph.x);

    t = -1.0f; tsi = -1.0f; si = false; fade = 1.0f; fadesi = 1.0f;
    norm = normsi = float3(0, 1, 0);

    float tmp = 1.0f / (vb.y * vc.x);
    float dd  = va.z * tmp;
    float gg  = (vc.z * vb.y - vd.y * va.z) * tmp;
    float hh  = (vb.z * vc.x - va.z * vd.x) * tmp;
    float jj  = (vd.x * vd.y * va.z + vd.z * vb.y * vc.x) * tmp
              - (vd.y * vb.z * vc.x + vd.x * vc.z * vb.y) * tmp;

    // Quadratic: p*t^2 + q*t + r = 0  (simplified with a=b=c=e=f=0)
    float p = dd * rd.x * rd.z;
    float q = dd * (ro.x * rd.z + ro.z * rd.x) + gg * rd.x + hh * rd.z - rd.y;
    float r = dd *  ro.x * ro.z               + gg * ro.x + hh * ro.z  - ro.y + jj;

    // Inline normal computation (gradient in permuted xzy space)
    // grad = pos.zxz*(dd,dd,0) + (gg,hh,-1)  =>  (pos.z*dd+gg, pos.x*dd+hh, -1)
#define GB_NORM(pos_) (-normalize(float3((pos_).z * dd + gg, (pos_).x * dd + hh, -1.0f)))

    if (abs(p) < 1e-6f) {
        if (abs(q) < 1e-12f) return false;
        float tt = -r / q;
        if (tt <= 0.0f) return false;
        float3 pos = ro + tt * rd;
        if (length(pos) > sz) return false;
        t = tt;
        norm = GB_NORM(pos);
        return true;
    }

    float sq = q * q - 4.0f * p * r;
    if (sq < 0.0f) return false;
    float s  = sqrt(sq);
    float t0 = (-q + s) / (2.0f * p);
    float t1 = (-q - s) / (2.0f * p);
    float tt1 = min(t0 < 0.0f ? t1 : t0, t1 < 0.0f ? t0 : t1);
    float tt2 = max(t0 > 0.0f ? t1 : t0, t1 > 0.0f ? t0 : t1);
    float tt0 = tt1;
    if (tt0 <= 0.0f) return false;

    float3 pos = ro + tt0 * rd;
    bool ru = step(sz, length(pos)) > 0.5f;
    if (ru) { tt0 = tt2; pos = ro + tt0 * rd; }
    if (tt0 <= 0.0f) return false;
    if (step(sz, length(pos)) > 0.5f) return false;

    if ((tt2 > 0.0f) && !ru && !(step(sz, length(ro + tt2 * rd)) > 0.5f)) {
        si     = true;
        fadesi = s;
        tsi    = tt2;
        float3 tp = ro + tsi * rd;
        normsi = GB_NORM(tp);
    }
    fade = s;
    t    = tt0;
    norm = GB_NORM(pos);
    return true;
#undef GB_NORM
}

// ─── Ray vs axis-aligned box ──────────────────────────────────────────────────
// entering=true  → returns near-t and sets nn to entry normal
// entering=false → returns far-t  and sets nn to exit normal
static float gb_boxHit(float3 ro, float3 rd, float3 r, thread float3 &nn, bool entering) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 dr = 1.0f / rd;
    float3 n  = ro * dr;
    float3 k  = r  * abs(dr);
    float3 pin  = -k - n;
    float3 pout =  k - n;
    float tin  = max(pin.x,  max(pin.y,  pin.z));
    float tout = min(pout.x, min(pout.y, pout.z));
    if (tin > tout) return -1.0f;
    if (entering) {
        nn = -sign(rd) * step(pin.zxy,  pin.xyz)  * step(pin.yzx,  pin.xyz);
        return tin;
    } else {
        nn =  sign(rd) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);
        return tout;
    }
}

// ─── Background visible through refracted ray (simplified NO_SHADOW path) ────
static float3 gb_background(float3 ro, float3 rd, float3 l_dir, thread float &alpha) {
    float3 bgc = mix(float3(0.01f), float3(0.336f, 0.458f, 0.668f),
                     1.0f - pow(abs(rd.z + 0.25f), 1.3f));
    float t = (-GB_BOXDIMS.z - ro.z) / rd.z;
    alpha = 0.0f;
    if (t < 0.0f) return bgc;
    float2 uv  = ro.xy + t * rd.xy;
    float aofac = smoothstep(-0.95f, 0.75f, length(abs(uv) - min(abs(uv), float2(0.45f))));
    float lght  = max(dot(normalize(ro + t * rd + float3(0,0,-5)),
                         normalize(l_dir - float3(0,0,1)) * gb_rotz(GB_PI * 0.65f)), 0.0f);
    float3 col = mix(float3(0.4f), float3(0.71f, 0.772f, 0.895f),
                     lght * lght * aofac + 0.05f) * aofac;
    alpha = 1.0f - smoothstep(7.0f, 10.0f, length(uv));
    return mix(col * length(col) * 0.8f, bgc, smoothstep(7.0f, 10.0f, length(uv)));
}

// ─── Internal bilinear-patch rendering (3 rotated layers) ────────────────────
static float4 gb_insides(float3 ro, float3 rd, float3 nor_c, float3 l_dir,
                          float t_time, thread float &tout)
{
    tout = -1.0f;

    // Remap ro/rd into the canonical face coordinate system
    if (abs(nor_c.x) > 0.5f) {
        rd = rd.xzy * nor_c.x;
        ro = ro.xzy * nor_c.x;
    } else if (abs(nor_c.z) > 0.5f) {
        l_dir = l_dir * gb_roty(GB_PI);
        rd    = rd.yxz * nor_c.z;
        ro    = ro.yxz * nor_c.z;
    } else if (abs(nor_c.y) > 0.5f) {
        l_dir = l_dir * gb_rotz(-GB_PI * 0.5f);
        rd    = rd    * nor_c.y;
        ro    = ro    * nor_c.y;
    }

    // Apply time rotation AFTER face remapping so all faces animate with the same phase.
    // Moving this here (from the fragment shader) ensures the rotation is in each face's
    // local frame (z = into the box), giving consistent visual behaviour on every face.
    {
        float faceRot = t_time * 0.8999f * 0.25f;
        ro = ro * gb_rotz(faceRot);
        rd = rd * gb_rotz(faceRot);
    }

    const float curvature = 0.5f;
    const float bil_size  = 1.0f;
    float4 ps = float4(-bil_size, -bil_size,  bil_size,  bil_size) * curvature;
    float4 ph = float4(-bil_size,  bil_size,  bil_size, -bil_size) * curvature;

    float4 colx[3]   = { float4(0), float4(0), float4(0) };
    float3 dx[3]     = { float3(-1), float3(-1), float3(-1) };
    float4 colxsi[3] = { float4(0), float4(0), float4(0) };
    int    order[3]  = { 0, 1, 2 };

    for (int i = 0; i < 3; i++) {
        if (abs(nor_c.x) > 0.5f) {
            ro = ro * gb_rotz(-GB_PI * (1.0f / 3.0f));
            rd = rd * gb_rotz(-GB_PI * (1.0f / 3.0f));
        } else if (abs(nor_c.z) > 0.5f) {
            ro = ro * gb_rotz( GB_PI * (1.0f / 3.0f));
            rd = rd * gb_rotz( GB_PI * (1.0f / 3.0f));
        } else if (abs(nor_c.y) > 0.5f) {
            ro = ro * gb_rotx( GB_PI * (1.0f / 3.0f));
            rd = rd * gb_rotx( GB_PI * (1.0f / 3.0f));
        }

        float3 normnew; float tnew;
        bool   si;      float tsi;   float3 normsi;
        float  fade;    float fadesi;

        if (gb_bilinearPatch(ro, rd, ps, ph, bil_size,
                             tnew, normnew, si, tsi, normsi, fade, fadesi)) {
            if (tnew > 0.0f) {
                float4 tcol(0), tcolsi(0);
                gb_calcColor(ro, rd, tnew, bil_size, si, tsi, t_time, tcol, tcolsi);
                if (tcol.a > 0.0f) {
                    dx[i] = float3(tnew, si ? 1.0f : 0.0f, tsi);

                    float dif = clamp(dot(normnew, l_dir), 0.0f, 1.0f);
                    float amb = clamp(0.5f + 0.5f * dot(normnew, l_dir), 0.0f, 1.0f);
                    float3 shad = float3(0.32f, 0.43f, 0.54f) * amb
                                + float3(1.0f,  0.9f,  0.7f)  * dif;
                    float3 tcr  = float3(1.0f, 0.21f, 0.11f);
                    float  ta   = clamp(length(tcol.rgb), 0.0f, 1.0f);
                    tcol        = clamp(tcol * tcol * 2.0f, 0.0f, 1.0f);
                    float4 tv   = float4(
                        tcol.rgb * shad * 1.4f
                        + 3.0f * (tcr * tcol.rgb) * clamp(1.0f - (amb + dif), 0.0f, 1.0f),
                        min(tcol.a, ta));
                    tv.rgb      = clamp(2.0f * tv.rgb * tv.rgb, 0.0f, 1.0f);
                    tv         *= min(fade * 5.0f, 1.0f);
                    colx[i]     = tv;

                    if (si) {
                        dif  = clamp(dot(normsi, l_dir), 0.0f, 1.0f);
                        amb  = clamp(0.5f + 0.5f * dot(normsi, l_dir), 0.0f, 1.0f);
                        shad = float3(0.32f, 0.43f, 0.54f) * amb
                             + float3(1.0f,  0.9f,  0.7f)  * dif;
                        float ta2   = clamp(length(tcolsi.rgb), 0.0f, 1.0f);
                        tcolsi      = clamp(tcolsi * tcolsi * 2.0f, 0.0f, 1.0f);
                        float4 tv2  = float4(
                            tcolsi.rgb * shad
                            + 3.0f * (tcr * tcolsi.rgb) * clamp(1.0f - (amb+dif), 0.0f, 1.0f),
                            min(tcolsi.a, ta2));
                        tv2.rgb     = clamp(2.0f * tv2.rgb * tv2.rgb, 0.0f, 1.0f);
                        tv2.rgb    *= min(fadesi * 5.0f, 1.0f);
                        colxsi[i]   = tv2;
                    }
                }
            }
        }
    }

    // Sort dx[] descending by x (farthest first) — bubble sort 3 elements
    // Inline the swap macro: {TYPE swap(a,b)} => TYPE _t=a;a=b;b=_t;
    if (dx[0].x < dx[1].x) {
        float3 _f = dx[0];    dx[0]    = dx[1];    dx[1]    = _f;
        int    _i = order[0]; order[0] = order[1]; order[1] = _i;
    }
    if (dx[1].x < dx[2].x) {
        float3 _f = dx[1];    dx[1]    = dx[2];    dx[2]    = _f;
        int    _i = order[1]; order[1] = order[2]; order[2] = _i;
    }
    if (dx[0].x < dx[1].x) {
        float3 _f = dx[0];    dx[0]    = dx[1];    dx[1]    = _f;
        int    _i = order[0]; order[0] = order[1]; order[1] = _i;
    }

    tout = max(max(dx[0].x, dx[1].x), dx[2].x);

    float a = 1.0f;
    if (dx[0].y < 0.5f) { a = colx[order[0]].a; }

    // Self-intersection compositing
    bool rul[3] = {
        (dx[0].y > 0.5f) && (dx[1].x <= 0.0f),
        (dx[1].y > 0.5f) && (dx[0].x > dx[1].z),
        (dx[2].y > 0.5f) && (dx[1].x > dx[2].z)
    };
    for (int k = 0; k < 3; k++) {
        if (rul[k]) {
            float4 tsi2 = colxsi[order[k]];
            float4 tcx  = colx[order[k]];
            float4 tv   = mix(tsi2, tcx, tcx.a);
            colx[order[k]] = mix(float4(0), tv, max(tcx.a, tsi2.a));
        }
    }

    float a1 = (dx[1].y < 0.5f) ? colx[order[1]].a
             : ((dx[1].z > dx[0].x) ? colx[order[1]].a : 1.0f);
    float a2 = (dx[2].y < 0.5f) ? colx[order[2]].a
             : ((dx[2].z > dx[1].x) ? colx[order[2]].a : 1.0f);
    float3 col = mix(mix(colx[order[0]].rgb, colx[order[1]].rgb, a1),
                     colx[order[2]].rgb, a2);
    a = max(max(a, a1), a2);
    return float4(col, a);
}

// ─── Fragment shader ──────────────────────────────────────────────────────────
fragment float4 glassBoxFragment(
    GlassBoxVertexOut          in         [[stage_in]],
    constant GlassBoxUniforms &uniforms   [[buffer(0)]],
    constant float4x4         *v2wMats    [[buffer(1)]],
    constant float4x4         *vpMatrices [[buffer(2)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);

    // Camera world position from viewToWorld transform column 3
    float4x4 v2w     = v2wMats[vi];
    float3 camWorld  = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    // Ray in world space → convert to box-local space
    float3 center    = uniforms.objectCenter.xyz;
    float  sc        = uniforms.boxScale;
    float3 eye       = (camWorld   - center) / sc;
    float3 rd        = normalize(in.worldPos - camWorld); // dir unchanged under uniform scale

    // Light direction (world-agnostic, same as original ShaderToy)
    float3 l_dir = normalize(float3(0, 1, 0)) * gb_rotz(0.5f);

    // (Time rotation is now applied inside gb_insides, after per-face remapping,
    //  so that all faces show the same rotational phase.)

    bool insideBox = all(abs(eye) < (GB_BOXDIMS - 1e-3f));

    // Box intersection: outside cameras use the entry face; inside cameras use the
    // exit face, then flip the face normal so the local wall frame still points
    // inward into the box interior.
    float3 boxNormal;
    float  t = gb_boxHit(eye, rd, GB_BOXDIMS, boxNormal, !insideBox);
    if (t < 0.0f) discard_fragment();

    float3 ni = insideBox ? -boxNormal : boxNormal;

    float3 ro        = eye + t * rd;
    float2 coords    = ro.xy * ni.z / GB_BOXDIMS.xy
                     + ro.yz * ni.x / GB_BOXDIMS.yz
                     + ro.zx * ni.y / GB_BOXDIMS.zx;
    float fadeborders = (1.0f - smoothstep(0.915f, 1.05f, abs(coords.x)))
                      * (1.0f - smoothstep(0.915f, 1.05f, abs(coords.y)));

    // Mirror-only setup: keep the surface reflection term, but do not refract the
    // ray into the box. This preserves apparent depth across faces instead of
    // pulling the contents toward the entry face.
    // theta=0 simplification: n=(0,0,1), nr = ni (identity transform)
    float3 nr = ni;
    float  R0 = (GB_IOR - 1.0f) / (GB_IOR + 1.0f);
    R0 *= R0;

    float  talpha_bg = 0.0f;
    float3 reflcol = insideBox ? float3(0.0f) : gb_background(ro, reflect(rd, nr), l_dir, talpha_bg);
    float3 rd2     = rd;

    float  accum    = 1.0f;
    float3 no2      = ni;
    float3 ro_refr  = ro + rd2 * 1e-3f;
    float4 colo[2]  = { float4(0), float4(0) };

    for (int j = 0; j < 2; j++) {
        float  tb;
        // Project ro_refr onto the current face coordinate system
        float2 coords2  = ro_refr.xy * no2.z + ro_refr.yz * no2.x + ro_refr.zx * no2.y;
        float3 eye2     = float3(coords2, -1.0f);
        float3 rd2t     = rd2.yzx * no2.x + rd2.zxy * no2.y + rd2.xyz * no2.z;
        rd2t.z          = -rd2t.z;

        float4 internalcol = gb_insides(eye2, rd2t, no2, l_dir, uniforms.time, tb);
        if (tb > 0.0f) {
            internalcol.rgb *= accum;
            colo[j]          = internalcol;
        }

        if ((tb <= 0.0f) || (internalcol.a < 1.0f)) {
            float3 no2_exit;
            float  tout   = gb_boxHit(ro_refr, rd2, GB_BOXDIMS, no2_exit, false);
            rd2           = reflect(rd2, -no2_exit);
            ro_refr       = ro_refr + tout * rd2;
            ro_refr.z     = max(ro_refr.z, -0.999f);
            no2           = no2_exit;
        }
    }

    float  fresnel = insideBox
                   ? 0.0f
                   : R0 + (1.0f - R0) * pow(max(1.0f - dot(-rd, nr), 0.0f), 5.0f);
    float3 col     = mix(
        mix(colo[1].rgb * colo[1].a, colo[0].rgb, colo[0].a) * fadeborders,
        reflcol,
        pow(fresnel, 1.5f));
    col = clamp(col, 0.0f, 1.0f);
    return float4(col, 1.0f);
}
