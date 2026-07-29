// DirtBallShaders.metal
// "Dirt Ball" — cube-container adaptation of ShaderToy "MsVcRy"
// Source: https://www.shadertoy.com/view/MsVcRy
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Source notes:
// - The original shader combines a cut sphere, floor lighting, cloudy sky,
//   exterior glow, and an internal fractal volume.
// - This version keeps those scene components, but replaces the synthetic orbit
//   camera with a real per-eye world ray entering a visible 2 m cube container.
// - The dirt-ball scene is evaluated in its own scene space and is not clipped
//   by the cube bounds.

#include <metal_stdlib>
using namespace metal;

struct DirtBallUniforms {
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

struct DirtBallVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

struct Scene {
    float t;
    float id;
    float3 n;
    float stn;
    float stf;
};

static constant float DB_FAR = 20.0f;
static constant float DB_EPS = 0.005f;
static constant float DB_SPHERE_EXTERIOR = 1.0f;
static constant float DB_SPHERE_INTERIOR = 2.0f;
static constant float DB_FLOOR = 3.0f;
static constant float DB_SR = 0.2f;
static constant float3 DB_CA = float3(0.5f);
static constant float3 DB_CB = float3(0.5f);
static constant float3 DB_CC = float3(1.0f);
static constant float3 DB_CD = float3(0.0f, 0.33f, 0.67f);
static constant float3 DB_BOX_HALF = float3(1.0f);
static constant float4 DB_SPHERE = float4(0.0f, 0.0f, 0.0f, 1.0f);

vertex DirtBallVertexOut dirtBallVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant DirtBallUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.cubeScale + uniforms.objectCenter.xyz;

    DirtBallVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 dbRotate(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

static float3 palette(float t, float3 a, float3 b, float3 c, float3 d) {
    return a + b * cos(6.28318f * (c * t + d));
}

static float3 glowColour(float time) {
    return palette(time * 0.1f, DB_CA, DB_CB, DB_CC, DB_CD);
}

static float2 csqr(float2 a) {
    return float2(a.x * a.x - a.y * a.y, 2.0f * a.x * a.y);
}

static float noise(float3 rp) {
    float3 ip = floor(rp);
    rp -= ip;
    float3 s = float3(7.0f, 157.0f, 113.0f);
    float4 h = float4(0.0f, s.y, s.z, s.y + s.z) + dot(ip, s);
    rp = rp * rp * (3.0f - 2.0f * rp);
    h = mix(fract(sin(h) * 43758.5f), fract(sin(h + s.x) * 43758.5f), rp.x);
    h.xy = mix(h.xz, h.yw, rp.y);
    return mix(h.x, h.y, rp.z);
}

static float fbm(float3 x) {
    float r = 0.0f;
    float w = 1.0f;
    float s = 1.0f;
    for (int i = 0; i < 5; ++i) {
        w *= 0.5f;
        s *= 2.0f;
        r += w * noise(s * x);
    }
    return r;
}

static float tex(float3 rp, float time) {
    rp.xy = dbRotate(rp.xy, time);
    if (rp.x > 0.3f && rp.x < 0.5f) {
        return 0.0f;
    }
    return 1.0f;
}

static float pattern(float3 rp, float time) {
    float3 f = abs(rp);
    f = step(f.zxy, f) * step(f.yzx, f);
    float2 face = f.x > 0.5f ? rp.yz / max(rp.x, 1.0e-4f)
        : f.y > 0.5f ? rp.xz / max(rp.y, 1.0e-4f)
        : rp.xy / max(rp.z, 1.0e-4f);
    return tex(float3(face, 0.0f), time);
}

static float4 sphIntersect(float3 ro, float3 rd, float4 sph, float time) {
    float3 oc = ro - sph.xyz;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sph.w * sph.w;
    float h = b * b - c;
    if (h < 0.0f) {
        return float4(0.0f);
    }
    h = sqrt(h);
    float tN = -b - h;
    float tNF = tN;
    if (pattern(ro + rd * tNF, time) == 0.0f) {
        tNF = 0.0f;
    }
    float tF = -b + h;
    float tFF = tF;
    if (pattern(ro + rd * tFF, time) == 0.0f) {
        tFF = 0.0f;
    }
    return float4(tNF, tFF, tN, tF);
}

static float3 sphNormal(float3 pos, float4 sph) {
    return normalize(pos - sph.xyz);
}

static float sphSoftShadow(float3 ro, float3 rd, float4 sph, float k, float time) {
    float3 oc = ro - sph.xyz;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sph.w * sph.w;
    float h = b * b - c;
    float d = sqrt(max(0.0f, sph.w * sph.w - h)) - sph.w;
    float tN = -b - sqrt(max(h, 0.0f));
    float tF = -b + sqrt(max(h, 0.0f));
    if ((pattern(ro + rd * tN, time) + pattern(ro + rd * tF, time)) == 0.0f) {
        return 1.0f;
    }
    if (tN > 0.0f) {
        return smoothstep(0.0f, 1.0f, 4.0f * k * d / tN);
    }
    return 1.0f;
}

static float sphOcclusion(float3 pos, float3 nor, float4 sph) {
    float3 r = sph.xyz - pos;
    float l = length(r);
    float d = dot(nor, r);
    float res = d;
    if (d < sph.w) {
        res = pow(clamp((d + sph.w) / (2.0f * sph.w), 0.0f, 1.0f), 1.5f) * sph.w;
    }
    return clamp(res * (sph.w * sph.w) / (l * l * l), 0.0f, 1.0f);
}

static float planeIntersection(float3 ro, float3 rd, float3 n, float3 o) {
    return dot(o - ro, n) / dot(rd, n);
}

static float mapVolume(float3 rp) {
    return min(length(rp) - DB_SPHERE.w, rp.y + 1.0f);
}

static float fractal(float3 rp, float time) {
    float res = 0.0f;
    float x = 0.8f + sin(time * 0.2f) * 0.3f;
    rp.yz = dbRotate(rp.yz, time);
    float3 c = rp;
    for (int i = 0; i < 10; ++i) {
        rp = x * abs(rp) / max(dot(rp, rp), 1.0e-4f) - x;
        rp.yz = csqr(rp.yz);
        rp = rp.zxy;
        res += exp(-99.0f * abs(dot(rp, c)));
    }
    return res;
}

static float3 fractalMarch(float3 ro, float3 rd, float maxt, float time) {
    float3 pc = float3(0.0f);
    float t = 0.0f;
    float ns = 0.0f;
    for (int i = 0; i < 64; ++i) {
        float3 rp = ro + t * rd;
        float lt = length(rp) - DB_SR;
        ns = fractal(rp, time);
        if (lt < DB_EPS || t > maxt) {
            break;
        }
        t += 0.02f * exp(-2.0f * ns);
        float3 glow = glowColour(time);
        pc = 0.99f * (pc + 0.08f * glow * ns) / (1.0f + lt * lt);
        pc += 0.1f * glow / (1.0f + lt * lt);
    }
    return pc;
}

static float3 vMarch(float3 ro, float3 rd, float time) {
    float3 pc = float3(0.0f);
    float t = 0.0f;
    for (int i = 0; i < 96; ++i) {
        float3 rp = ro + rd * t;
        float ns = mapVolume(rp);
        float fz = pattern(rp, time);
        if ((ns < DB_EPS && fz > 0.0f) || t > DB_FAR) {
            break;
        }

        float3 ld = normalize(-rp);
        float lt = length(rp);
        if (sphIntersect(rp, ld, DB_SPHERE, time).x == 0.0f || lt < DB_SPHERE.w) {
            float ltn = lt - DB_SR;
            pc += glowColour(time) * 0.1f / (1.0f + ltn * ltn * 12.0f);
        }
        t += 0.05f;
    }
    return pc;
}

static float3 clouds(float3 rd, float time) {
    float ct = time / 14.0f;
    float2 uv = rd.xz / (rd.y + 0.6f);
    float nz = fbm(float3(uv.yx * 1.4f + float2(ct, 0.0f), ct)) * 1.5f;
    return clamp(pow(float3(nz), float3(4.0f)) * rd.y, 0.0f, 1.0f);
}

static float3 pri(float3 x) {
    float3 h = fract(x / 2.0f) - 0.5f;
    return x * 0.5f + h * (1.0f - 2.0f * abs(h));
}

static float checkersTextureGradTri(float3 p, float3 ddx, float3 ddy, float time) {
    p.z += time;
    float3 w = max(abs(ddx), abs(ddy)) + 0.01f;
    float3 i = (pri(p + w) - 2.0f * pri(p) + pri(p - w)) / (w * w);
    return 0.5f - 0.5f * i.x * i.y * i.z;
}

static float3 texCoords(float3 p) {
    return 5.0f * p;
}

static Scene drawScene(float3 ro, float3 rd, float time) {
    Scene scene;
    scene.t = DB_FAR;
    scene.id = 0.0f;
    scene.n = float3(0.0f);
    scene.stn = 0.0f;
    scene.stf = 0.0f;

    float3 fo = float3(0.0f, -1.0f, 0.0f);
    float3 fn = float3(0.0f, 1.0f, 0.0f);
    float ft = planeIntersection(ro, rd, fn, fo);
    if (ft > 0.0f && ft < DB_FAR) {
        scene.t = ft;
        scene.id = DB_FLOOR;
        scene.n = fn;
    }

    float4 si = sphIntersect(ro, rd, DB_SPHERE, time);
    if (si.x > 0.0f && si.x < scene.t) {
        float3 rp = ro + rd * si.x;
        scene.t = si.x;
        scene.id = DB_SPHERE_EXTERIOR;
        scene.n = sphNormal(rp, DB_SPHERE);
    } else if (si.y > 0.0f && si.y < scene.t) {
        float3 rp = ro + rd * si.y;
        scene.t = si.y;
        scene.id = DB_SPHERE_INTERIOR;
        scene.n = -sphNormal(rp, DB_SPHERE);
    }

    scene.stn = si.z;
    scene.stf = si.w;
    return scene;
}

static float2 dbBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float2 dbFaceUV(float3 p) {
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

fragment float4 dirtBallFragment(
    DirtBallVertexOut in [[stage_in]],
    constant DirtBallUniforms &uniforms [[buffer(0)]],
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

    bool insideOuter = all(abs(eye) < DB_BOX_HALF - 1.0e-3f);
    float2 tOuter = dbBoxIntersect(eye, rd, DB_BOX_HALF);
    if (!insideOuter && tOuter.x > tOuter.y) {
        discard_fragment();
    }

    float tStart = insideOuter ? 0.0f : max(tOuter.x, 0.0f);
    const float sceneScale = 2.0f;
    float3 ro = (eye + rd * (tStart + 0.001f)) * sceneScale;
    float3 marchDir = normalize(rd);
    ro.xz = dbRotate(ro.xz, uniforms.time * 0.4f);
    marchDir.xz = dbRotate(marchDir.xz, uniforms.time * 0.4f);

    Scene scene = drawScene(ro, marchDir, uniforms.time);
    float3 pc = clouds(marchDir, uniforms.time) * glowColour(uniforms.time);
    float3 gc = float3(0.0f);
    float3 lp = float3(4.0f, 5.0f, -2.0f);

    float3 rp = ro + marchDir * scene.t;
    float3 ld = normalize(lp - rp);
    float lt = length(lp - rp);
    float atten = 1.0f / (1.0f + lt * lt * 0.051f);

    if (scene.stn > 0.0f) {
        gc = fractalMarch(ro + marchDir * scene.stn, marchDir, scene.stf - scene.stn, uniforms.time);
        pc = gc;
    }

    if (scene.id == DB_FLOOR) {
        float3 uvw = texCoords(rp * 0.15f);
        float3 ddxUVW = dfdx(uvw);
        float3 ddyUVW = dfdy(uvw);
        float fc = checkersTextureGradTri(uvw, ddxUVW, ddyUVW, uniforms.time);
        float diff = max(dot(ld, scene.n), 0.05f);
        float ao = 1.0f - sphOcclusion(rp, scene.n, DB_SPHERE);
        float spec = pow(max(dot(reflect(-ld, scene.n), -marchDir), 0.0f), 32.0f);
        float sh = sphSoftShadow(rp, ld, DB_SPHERE, 2.0f, uniforms.time);

        pc += glowColour(uniforms.time) * fc * diff * atten;
        pc += float3(1.0f) * spec;
        pc *= ao * sh;

        float3 gld = normalize(-rp);
        if (sphIntersect(rp, gld, DB_SPHERE, uniforms.time).x == 0.0f) {
            pc += glowColour(uniforms.time) / (1.0f + length(rp) * length(rp));
        }
    }

    if (scene.id == DB_SPHERE_EXTERIOR) {
        float ao = 0.5f + 0.5f * scene.n.y;
        float spec = pow(max(dot(reflect(-ld, scene.n), -marchDir), 0.0f), 32.0f);
        float fres = pow(clamp(dot(scene.n, marchDir) + 1.0f, 0.0f, 1.0f), 2.0f);
        pc *= 0.4f * (1.0f - fres);
        pc += float3(1.0f) * fres * 0.2f;
        pc *= ao;
        pc += float3(1.0f) * spec;
    }

    if (scene.id == DB_SPHERE_INTERIOR) {
        float ao = 0.5f + 0.5f * scene.n.y;
        float ilt = length(rp) - DB_SR;
        pc += glowColour(uniforms.time) * ao / (1.0f + ilt * ilt);
    }

    pc += vMarch(ro, marchDir, uniforms.time);

    float2 faceUV = dbFaceUV(surfacePos) * 2.0f - 1.0f;
    float vignette = 1.0f - 0.18f * dot(faceUV, faceUV);
    return float4(clamp(pc * 2.0f * vignette, 0.0f, 1.0f), 1.0f);
}