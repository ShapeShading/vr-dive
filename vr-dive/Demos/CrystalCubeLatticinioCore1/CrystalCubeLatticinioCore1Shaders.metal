// CrystalCubeLatticinioCore1Shaders.metal
// "Crystal Cube Latticinio core 1" — cube-portal adaptation of Shadertoy "Wfy3Wm"
// Original: https://www.shadertoy.com/view/Wfy3Wm
//
// Metal adaptation notes:
// - The original GLSL already renders a glass cube with internal bilinear patch
//   geometry, but it does so from a synthetic screen-space orbit camera.
// - This version replaces that camera with the real per-eye world ray from the
//   application. Outside the cube, rendering starts from the visible container
//   surface. Inside the cube, rendering uses the currently viewed inner face as
//   the portal and marches inward from that face.
// - The internal crystal lattice field remains fixed in scene space and is not
//   clipped to the container bounds, so user motion produces true stereo and
//   positional parallax rather than a 2D surface image.

#include <metal_stdlib>
using namespace metal;

struct CrystalCubeLatticinioCore1Uniforms {
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

struct CrystalCubeLatticinioCore1VertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float3 CCLC_BOX_DIMS = float3(1.0f, 1.0f, 1.0f);
static constant float CCLC_PI = 3.1415926f;
static constant float CCLC_IOR = 1.33f;
static constant float CCLC_HUE_SHIFT = 0.0f;
static constant float CCLC_INTERIOR_SCENE_SCALE = 1.22f;
static constant float CCLC_COLOR_FREQ = 2.5f;
static constant float CCLC_COLOR_SPEED = 0.5f;
static constant float CCLC_COLOR_WARP = 0.0f;
static constant float CCLC_CONTRAST = 1.2f;

vertex CrystalCubeLatticinioCore1VertexOut crystalCubeLatticinioCore1Vertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant CrystalCubeLatticinioCore1Uniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    CrystalCubeLatticinioCore1VertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float3 cclcRotXRow(float3 v, float a) {
    float s = sin(a);
    float c = cos(a);
    return float3(v.x, c * v.y + s * v.z, -s * v.y + c * v.z);
}

static float3 cclcRotYRow(float3 v, float a) {
    float s = sin(a);
    float c = cos(a);
    return float3(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}

static float3 cclcRotZRow(float3 v, float a) {
    float s = sin(a);
    float c = cos(a);
    return float3(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}

static float3 cclcPalette(float t) {
    float3 a = float3(0.5f, 0.5f, 0.5f);
    float3 b = float3(0.5f, 0.5f, 0.5f);
    float3 c = float3(1.0f, 1.0f, 1.0f);
    float3 d = float3(0.263f, 0.416f, 0.557f) + CCLC_HUE_SHIFT;
    return a + b * cos(6.28318f * (c * t + d));
}

static float3 cclcGetColor(float3 p, float time) {
    p = abs(p);

    float warp = sin(p.x * 3.0f + time * CCLC_COLOR_SPEED)
               * cos(p.y * 3.0f - time * CCLC_COLOR_SPEED);
    p += CCLC_COLOR_WARP * warp;

    float t = length(p) * CCLC_COLOR_FREQ;
    t += 0.5f * sin(p.x * 5.0f + time * CCLC_COLOR_SPEED);
    t += 0.3f * cos(p.y * 7.0f - time * CCLC_COLOR_SPEED);

    float3 col = cclcPalette(t);
    col *= 0.8f + 0.4f * sin(10.0f * t);
    col = pow(clamp(col, 0.0f, 1.0f), float3(CCLC_CONTRAST));
    return clamp(col, 0.0f, 1.0f);
}

static void cclcCalcColor(
    float3 ro,
    float3 rd,
    float3 nor,
    float d,
    float len,
    int idx,
    bool si,
    float td,
    float time,
    thread float4 &colx,
    thread float4 &colsi)
{
    float3 pos = ro + rd * d;
    float a = 1.0f - smoothstep(len - 0.15f * 0.5f, len + 0.00001f, length(pos));
    float3 col = cclcGetColor(pos, time);
    colx = float4(col, a);
    if (si) {
        pos = ro + rd * td;
        float ta = 1.0f - smoothstep(len - 0.15f * 0.5f, len + 0.00001f, length(pos));
        col = cclcGetColor(pos, time);
        colsi = float4(col, ta);
    }
}

static bool cclcIBilinearPatch(
    float3 ro,
    float3 rd,
    float4 ps,
    float4 ph,
    float sz,
    thread float &t,
    thread float3 &norm,
    thread bool &si,
    thread float &tsi,
    thread float3 &normsi,
    thread float &fade,
    thread float &fadesi)
{
    float3 va = float3(0.0f, 0.0f, ph.x + ph.w - ph.y - ph.z);
    float3 vb = float3(0.0f, ps.w - ps.y, ph.z - ph.x);
    float3 vc = float3(ps.z - ps.x, 0.0f, ph.y - ph.x);
    float3 vd = float3(ps.x, ps.y, ph.x);
    t = -1.0f;
    tsi = -1.0f;
    si = false;
    fade = 1.0f;
    fadesi = 1.0f;
    norm = float3(0.0f, 1.0f, 0.0f);
    normsi = float3(0.0f, 1.0f, 0.0f);

    float tmp = 1.0f / (vb.y * vc.x);
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    float d = va.z * tmp;
    float e = 0.0f;
    float f = 0.0f;
    float g = (vc.z * vb.y - vd.y * va.z) * tmp;
    float h = (vb.z * vc.x - va.z * vd.x) * tmp;
    float i = -1.0f;
    float j = (vd.x * vd.y * va.z + vd.z * vb.y * vc.x) * tmp
            - (vd.y * vb.z * vc.x + vd.x * vc.z * vb.y) * tmp;

    float p = dot(float3(a, b, c), rd.xzy * rd.xzy) + dot(float3(d, e, f), rd.xzy * rd.zyx);
    float q = dot(2.0f * ro.xzy * rd.xyz, float3(a, b, c))
            + dot(ro.xzz * rd.zxy, float3(d, d, e))
            + dot(ro.yyx * rd.zxy, float3(e, f, f))
            + dot(float3(g, h, i), rd.xzy);
    float r = dot(float3(a, b, c), ro.xzy * ro.xzy)
            + dot(float3(d, e, f), ro.xzy * ro.zyx)
            + dot(float3(g, h, i), ro.xzy)
            + j;

    if (abs(p) < 0.000001f) {
        float tt = -r / q;
        if (tt <= 0.0f) {
            return false;
        }
        t = tt;
        float3 pos = ro + t * rd;
        if (length(pos) > sz) {
            return false;
        }
        float3 grad = 2.0f * pos.xzy * float3(a, b, c)
                    + pos.zxz * float3(d, d, e)
                    + pos.yyx * float3(f, e, f)
                    + float3(g, h, i);
        norm = -normalize(grad);
        return true;
    }

    float sq = q * q - 4.0f * p * r;
    if (sq < 0.0f) {
        return false;
    }

    float s = sqrt(sq);
    float t0 = (-q + s) / (2.0f * p);
    float t1 = (-q - s) / (2.0f * p);
    float tt1 = min(t0 < 0.0f ? t1 : t0, t1 < 0.0f ? t0 : t1);
    float tt2 = max(t0 > 0.0f ? t1 : t0, t1 > 0.0f ? t0 : t1);
    float tt0 = tt1;
    if (tt0 <= 0.0f) {
        return false;
    }

    float3 pos = ro + tt0 * rd;
    bool ru = step(sz, length(pos)) > 0.5f;
    if (ru) {
        tt0 = tt2;
        pos = ro + tt0 * rd;
    }
    if (tt0 <= 0.0f) {
        return false;
    }
    bool ru2 = step(sz, length(pos)) > 0.5f;
    if (ru2) {
        return false;
    }

    if ((tt2 > 0.0f) && (!ru) && !(step(sz, length(ro + tt2 * rd)) > 0.5f)) {
        si = true;
        fadesi = s;
        tsi = tt2;
        float3 tpos = ro + tsi * rd;
        float3 tgrad = 2.0f * tpos.xzy * float3(a, b, c)
                     + tpos.zxz * float3(d, d, e)
                     + tpos.yyx * float3(f, e, f)
                     + float3(g, h, i);
        normsi = -normalize(tgrad);
    }

    fade = s;
    t = tt0;
    float3 grad = 2.0f * pos.xzy * float3(a, b, c)
                + pos.zxz * float3(d, d, e)
                + pos.yyx * float3(f, e, f)
                + float3(g, h, i);
    norm = -normalize(grad);
    return true;
}

static float cclcDot2(float3 v) {
    return dot(v, v);
}

static float cclcSegShadow(float3 ro, float3 rd, float3 pa, float sh) {
    float dm = dot(rd.yz, rd.yz);
    float k1 = (ro.x - pa.x) * dm;
    float k2 = (ro.x + pa.x) * dm;
    float2 k5 = (ro.yz + pa.yz) * dm;
    float k3 = dot(ro.yz + pa.yz, rd.yz);
    float2 k4 = (pa.yz + pa.yz) * rd.yz;
    float2 k6 = (pa.yz + pa.yz) * dm;

    for (int i = 0; i < 4; ++i) {
        float2 s = float2(float(i & 1), float(i >> 1));
        float t = dot(s, k4) - k3;
        if (t > 0.0f) {
            sh = min(sh, cclcDot2(float3(clamp(-rd.x * t, k1, k2), k5 - k6 * s) + rd * t) / (t * t));
        }
    }
    return sh;
}

static float cclcBoxSoftShadow(float3 ro, float3 rd, float3 rad, float sk) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 rdd = rd;
    float3 roo = ro;

    float3 m = 1.0f / rdd;
    float3 n = m * roo;
    float3 k = abs(m) * rad;

    float3 t1 = -n - k;
    float3 t2 = -n + k;

    float tN = max(max(t1.x, t1.y), t1.z);
    float tF = min(min(t2.x, t2.y), t2.z);

    if (tN < tF && tF > 0.0f) {
        return 0.0f;
    }

    float sh = 1.0f;
    sh = cclcSegShadow(roo.xyz, rdd.xyz, rad.xyz, sh);
    sh = cclcSegShadow(roo.yzx, rdd.yzx, rad.yzx, sh);
    sh = cclcSegShadow(roo.zxy, rdd.zxy, rad.zxy, sh);
    sh = clamp(sk * sqrt(sh), 0.0f, 1.0f);
    return sh * sh * (3.0f - 2.0f * sh);
}

static float cclcBox(float3 ro, float3 rd, float3 r, thread float3 &nn, bool entering) {
    rd += 0.0001f * (1.0f - abs(sign(rd)));
    float3 dr = 1.0f / rd;
    float3 n = ro * dr;
    float3 k = r * abs(dr);

    float3 pin = -k - n;
    float3 pout = k - n;
    float tin = max(pin.x, max(pin.y, pin.z));
    float tout = min(pout.x, min(pout.y, pout.z));
    if (tin > tout) {
        return -1.0f;
    }
    if (entering) {
        nn = -sign(rd) * step(pin.zxy, pin.xyz) * step(pin.yzx, pin.xyz);
    } else {
        nn = sign(rd) * step(pout.xyz, pout.zxy) * step(pout.xyz, pout.yzx);
    }
    return entering ? tin : tout;
}

static float3 cclcBgCol(float3 rd) {
    return mix(float3(0.01f), float3(0.336f, 0.458f, 0.668f), 1.0f - pow(abs(rd.z + 0.25f), 1.3f));
}

static float3 cclcBackground(float3 ro, float3 rd, float3 lDir, thread float &alpha) {
    float t = (-CCLC_BOX_DIMS.z - ro.z) / rd.z;
    alpha = 0.0f;
    float3 bgc = cclcBgCol(rd);
    if (t < 0.0f) {
        return bgc;
    }

    float2 uv = ro.xy + t * rd.xy;
    float3 shadowDir = cclcRotZRow(normalize(lDir + float3(0.0f, 0.0f, 1.0f)), CCLC_PI * 0.65f);
    float shad = cclcBoxSoftShadow(ro + t * rd, shadowDir, CCLC_BOX_DIMS, 1.5f);

    float aofac = smoothstep(-0.95f, 0.75f, length(abs(uv) - min(abs(uv), float2(0.45f))));
    aofac = min(aofac, smoothstep(-0.65f, 1.0f, shad));
    float lght = max(dot(normalize(ro + t * rd + float3(0.0f, 0.0f, -5.0f)),
                         cclcRotZRow(normalize(lDir - float3(0.0f, 0.0f, 1.0f)), CCLC_PI * 0.65f)), 0.0f);
    float3 col = mix(float3(0.4f), float3(0.71f, 0.772f, 0.895f), lght * lght * aofac + 0.05f) * aofac;
    alpha = 1.0f - smoothstep(7.0f, 10.0f, length(uv));
    return mix(col * length(col) * 0.8f, bgc, smoothstep(7.0f, 10.0f, length(uv)));
}

static float4 cclcInsides(float3 ro, float3 rd, float3 norC, float3 lDir, float time, thread float &tout) {
    tout = -1.0f;

    if (abs(norC.x) > 0.5f) {
        rd = rd.xzy * norC.x;
        ro = ro.xzy * norC.x;
    } else if (abs(norC.z) > 0.5f) {
        lDir = cclcRotYRow(lDir, CCLC_PI);
        rd = rd.yxz * norC.z;
        ro = ro.yxz * norC.z;
    } else if (abs(norC.y) > 0.5f) {
        lDir = cclcRotZRow(lDir, -CCLC_PI * 0.5f);
        rd = rd * norC.y;
        ro = ro * norC.y;
    }

    // Shrink the authored lattice inside the same 2 m portal container.
    ro *= CCLC_INTERIOR_SCENE_SCALE;
    rd *= CCLC_INTERIOR_SCENE_SCALE;

    float curvature = 0.5f;
    float bilSize = 1.0f;
    float4 ps = float4(-bilSize, -bilSize, bilSize, bilSize) * curvature;
    float4 ph = float4(-bilSize, bilSize, bilSize, -bilSize) * curvature;

    float4 colx[3];
    float3 dx[3];
    float4 colxsi[3];
    int order[3];
    for (int k = 0; k < 3; ++k) {
        colx[k] = float4(0.0f);
        dx[k] = float3(-1.0f);
        colxsi[k] = float4(0.0f);
        order[k] = k;
    }

    for (int i = 0; i < 3; ++i) {
        if (abs(norC.x) > 0.5f) {
            ro = cclcRotZRow(ro, -CCLC_PI / 3.0f);
            rd = cclcRotZRow(rd, -CCLC_PI / 3.0f);
        } else if (abs(norC.z) > 0.5f) {
            ro = cclcRotZRow(ro, CCLC_PI / 3.0f);
            rd = cclcRotZRow(rd, CCLC_PI / 3.0f);
        } else if (abs(norC.y) > 0.5f) {
            ro = cclcRotXRow(ro, CCLC_PI / 3.0f);
            rd = cclcRotXRow(rd, CCLC_PI / 3.0f);
        }

        float3 normnew;
        float tnew;
        bool si;
        float tsi;
        float3 normsi;
        float fade;
        float fadesi;

        if (cclcIBilinearPatch(ro, rd, ps, ph, bilSize, tnew, normnew, si, tsi, normsi, fade, fadesi) && tnew > 0.0f) {
            float4 tcol = float4(0.0f);
            float4 tcolsi = float4(0.0f);
            cclcCalcColor(ro, rd, normnew, tnew, bilSize, i, si, tsi, time, tcol, tcolsi);
            if (tcol.a > 0.0f) {
                dx[i] = float3(tnew, float(si), tsi);

                float dif = clamp(dot(normnew, lDir), 0.0f, 1.0f);
                float amb = clamp(0.5f + 0.5f * dot(normnew, lDir), 0.0f, 1.0f);
                float3 shad = float3(0.32f, 0.43f, 0.54f) * amb + float3(1.0f, 0.9f, 0.7f) * dif;
                float3 tcr = float3(1.0f, 0.21f, 0.11f);
                float ta = clamp(length(tcol.rgb), 0.0f, 1.0f);
                tcol = clamp(tcol * tcol * 2.0f, 0.0f, 1.0f);
                float4 tvalx = float4(
                    (tcol.rgb * shad * 1.4f + 3.0f * (tcr * tcol.rgb) * clamp(1.0f - (amb + dif), 0.0f, 1.0f)),
                    min(tcol.a, ta));
                tvalx.rgb = clamp(2.0f * tvalx.rgb * tvalx.rgb, 0.0f, 1.0f);
                tvalx *= min(fade * 5.0f, 1.0f);
                colx[i] = tvalx;

                if (si) {
                    dif = clamp(dot(normsi, lDir), 0.0f, 1.0f);
                    amb = clamp(0.5f + 0.5f * dot(normsi, lDir), 0.0f, 1.0f);
                    shad = float3(0.32f, 0.43f, 0.54f) * amb + float3(1.0f, 0.9f, 0.7f) * dif;
                    ta = clamp(length(tcolsi.rgb), 0.0f, 1.0f);
                    tcolsi = clamp(tcolsi * tcolsi * 2.0f, 0.0f, 1.0f);
                    float4 tvalxsi = float4(
                        tcolsi.rgb * shad + 3.0f * (tcr * tcolsi.rgb) * clamp(1.0f - (amb + dif), 0.0f, 1.0f),
                        min(tcolsi.a, ta));
                    tvalxsi.rgb = clamp(2.0f * tvalxsi.rgb * tvalxsi.rgb, 0.0f, 1.0f);
                    tvalxsi.rgb *= min(fadesi * 5.0f, 1.0f);
                    colxsi[i] = tvalxsi;
                }
            }
        }
    }

    if (dx[0].x < dx[1].x) { float3 tv = dx[0]; dx[0] = dx[1]; dx[1] = tv; int to = order[0]; order[0] = order[1]; order[1] = to; }
    if (dx[1].x < dx[2].x) { float3 tv = dx[1]; dx[1] = dx[2]; dx[2] = tv; int to = order[1]; order[1] = order[2]; order[2] = to; }
    if (dx[0].x < dx[1].x) { float3 tv = dx[0]; dx[0] = dx[1]; dx[1] = tv; int to = order[0]; order[0] = order[1]; order[1] = to; }

    tout = max(max(dx[0].x, dx[1].x), dx[2].x);

    bool rul0 = (dx[0].y > 0.5f) && (dx[1].x <= 0.0f);
    bool rul1 = (dx[1].y > 0.5f) && (dx[0].x > dx[1].z);
    bool rul2 = (dx[2].y > 0.5f) && (dx[1].x > dx[2].z);
    bool rul[3] = {rul0, rul1, rul2};
    for (int k = 0; k < 3; ++k) {
        if (rul[k]) {
            float4 tcolxsi = colxsi[order[k]];
            float4 tcolx = colx[order[k]];
            float4 tvalx = mix(tcolxsi, tcolx, tcolx.a);
            colx[order[k]] = mix(float4(0.0f), tvalx, max(tcolx.a, tcolxsi.a));
        }
    }

    float a1 = (dx[1].y < 0.5f) ? colx[order[1]].a : ((dx[1].z > dx[0].x) ? colx[order[1]].a : 1.0f);
    float a2 = (dx[2].y < 0.5f) ? colx[order[2]].a : ((dx[2].z > dx[1].x) ? colx[order[2]].a : 1.0f);
    float3 col = mix(mix(colx[order[0]].rgb, colx[order[1]].rgb, a1), colx[order[2]].rgb, a2);
    float a = max(max(colx[order[0]].a, a1), a2);
    return float4(col, a);
}

fragment float4 crystalCubeLatticinioCore1Fragment(
    CrystalCubeLatticinioCore1VertexOut in [[stage_in]],
    constant CrystalCubeLatticinioCore1Uniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float sceneScale = max(uniforms.cubeScale, 1.0e-4f);
    float3 eye = (camWorld - center) / sceneScale;
    float3 surfaceWorld = in.worldPos;
    float3 viewRd = normalize(surfaceWorld - camWorld);

    float3 lightDir = normalize(float3(0.0f, 1.0f, 0.0f));
    lightDir = cclcRotZRow(lightDir, 0.5f);

    bool insideBox = all(abs(eye) < float3(0.999f));
    float3 ni;
    float hitT;
    float3 facePoint;
    float3 portalDir;
    if (insideBox) {
        hitT = cclcBox(eye, viewRd, CCLC_BOX_DIMS, ni, false);
        if (hitT < 0.0f) {
            discard_fragment();
        }
        facePoint = eye + hitT * viewRd;
        ni = -ni;
        portalDir = -viewRd;
    } else {
        hitT = cclcBox(eye, viewRd, CCLC_BOX_DIMS, ni, true);
        if (hitT < 0.0f) {
            float alpha;
            return float4(cclcBackground(eye, viewRd, lightDir, alpha), 1.0f);
        }
        facePoint = eye + hitT * viewRd;
        portalDir = viewRd;
    }

    float2 coords = facePoint.xy * ni.z / CCLC_BOX_DIMS.xy
                  + facePoint.yz * ni.x / CCLC_BOX_DIMS.yz
                  + facePoint.zx * ni.y / CCLC_BOX_DIMS.zx;
    float fadeBorders = (1.0f - smoothstep(0.915f, 1.05f, abs(coords.x)))
                      * (1.0f - smoothstep(0.915f, 1.05f, abs(coords.y)));

    float time = uniforms.time;
    float R0 = (CCLC_IOR - 1.0f) / (CCLC_IOR + 1.0f);
    R0 *= R0;

    float3 ro = facePoint + portalDir * 0.002f;
    float3 nr = ni;
    float3 rdr = reflect(portalDir, nr);
    float talpha;
    float3 reflcol = cclcBackground(ro, rdr, lightDir, talpha);

    float3 rd2 = refract(portalDir, nr, 1.0f / CCLC_IOR);
    if (all(rd2 == float3(0.0f))) {
        rd2 = reflect(portalDir, nr);
    }

    float accum = 1.0f;
    float3 no2 = ni;
    float3 roRefr = ro;
    float4 colo[2];
    colo[0] = float4(0.0f);
    colo[1] = float4(0.0f);

    for (int j = 0; j < 2; ++j) {
        float tb;
        float2 coords2 = roRefr.xy * no2.z + roRefr.yz * no2.x + roRefr.zx * no2.y;
        float3 eye2 = float3(coords2, -1.0f);
        float3 rd2trans = rd2.yzx * no2.x + rd2.zxy * no2.y + rd2.xyz * no2.z;
        rd2trans.z = -rd2trans.z;

        float4 internalcol = cclcInsides(eye2, rd2trans, no2, lightDir, time, tb);
        if (tb > 0.0f) {
            internalcol.rgb *= accum;
            colo[j] = internalcol;
        }

        if ((tb <= 0.0f) || (internalcol.a < 1.0f)) {
            float tout = cclcBox(roRefr, rd2, CCLC_BOX_DIMS, no2, false);
            no2 = nr.zyx * no2.x + nr.xzy * no2.y + nr.yxz * no2.z;
            float3 rout = roRefr + tout * rd2;
            float3 rdout = refract(rd2, -no2, CCLC_IOR);
            float fresnel2 = R0 + (1.0f - R0) * pow(1.0f - dot(rdout, no2), 1.3f);
            rd2 = reflect(rd2, -no2);
            roRefr = rout;
            roRefr.z = max(roRefr.z, -0.999f);
            accum *= fresnel2;
        }
    }

    float fresnel = R0 + (1.0f - R0) * pow(1.0f - dot(-portalDir, nr), 5.0f);
    float3 col = mix(mix(colo[1].rgb * colo[1].a, colo[0].rgb, colo[0].a) * fadeBorders,
                     reflcol,
                     pow(fresnel, 1.5f));
    col = clamp(col, 0.0f, 1.0f);
    return float4(col, 1.0f);
}
