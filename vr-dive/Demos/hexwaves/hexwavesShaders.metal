// hexwavesShaders.metal
// Adapted from ShaderToy "hexwaves" by mattz.
// Source: https://www.shadertoy.com/view/XsBczc
// License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported.
//
// Metal adaptation notes:
// - The original shader used a fixed camera plus a cubemap sampler.
//   This version reconstructs the real per-eye world ray, intersects it with a
//   2 m cube container, and starts tracing at the visible cube surface or at
//   the eye when the viewer is inside the cube.
// - The hex grid traversal continues beyond the container boundary, so the
//   simulated wave field is not clipped to the cube volume.
// - The cubemap reflection is replaced with a procedural environment because
//   this project does not bind `iChannel0` for container demos.

#include <metal_stdlib>
using namespace metal;

struct HexwavesUniforms {
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

struct HexwavesVertexOut {
    float4 clipPos [[position]];
    float3 worldPos;
    uint   viewIndex [[flat]];
};

static constant float HW_HEX_FACTOR = 0.8660254037844386f;
static constant float3 HW_FOG_COLOR = float3(0.9f, 0.95f, 1.0f);
static constant float3 HW_BOX_HALF = float3(1.0f);
static constant float HW_TRACE_EPSILON = 0.001f;

vertex HexwavesVertexOut hexwavesVertex(
    ushort amplificationID [[amplification_id]],
    const device MeshVertex *vertices [[buffer(0)]],
    constant HexwavesUniforms &uniforms [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]],
    uint vertexID [[vertex_id]])
{
    MeshVertex vtx = vertices[vertexID];
    uint viewIndex = min((uint)amplificationID, uniforms.viewCount - 1u);
    float3 worldPos = vtx.position * uniforms.boxScale + uniforms.objectCenter.xyz;

    HexwavesVertexOut out;
    out.clipPos = vpMatrices[viewIndex] * float4(worldPos, 1.0f);
    out.worldPos = worldPos;
    out.viewIndex = viewIndex;
    return out;
}

static float2 hwHexFromCart(float2 p) {
    return float2(p.x / HW_HEX_FACTOR, p.y);
}

static float2 hwCartFromHex(float2 g) {
    return float2(g.x * HW_HEX_FACTOR, g.y);
}

static float hwMod(float x, float y) {
    return x - y * floor(x / y);
}

static float2 hwMod2(float2 x, float2 y) {
    return x - y * floor(x / y);
}

static float2 hwRotate(float2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

static float3 hwRotateScene(float3 p, float time) {
    p.xy = hwRotate(p.xy, -0.14f * time);
    p.xz = hwRotate(p.xz, -0.35f - 0.2f * cos(0.031513f * time));
    p.yz = hwRotate(p.yz, 0.17f * sin(0.21f * time) + 0.4f);
    return p;
}

static float2 hwFaceUV(float3 p) {
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

static float2 hwBoxIntersect(float3 ro, float3 rd, float3 halfExt) {
    float3 inv = 1.0f / rd;
    float3 t0 = (-halfExt - ro) * inv;
    float3 t1 = (halfExt - ro) * inv;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);
    return float2(
        max(max(tMin.x, tMin.y), tMin.z),
        min(min(tMax.x, tMax.y), tMax.z));
}

static float hexDist(float2 p) {
    p = abs(p);
    return max(dot(p, float2(HW_HEX_FACTOR, 0.5f)), p.y) - 1.0f;
}

static float2 nearestHexCell(float2 pos) {
    float2 gpos = hwHexFromCart(pos);
    float2 hexInt = floor(gpos);

    float sy = step(2.0f, hwMod(hexInt.x + 1.0f, 4.0f));
    hexInt += hwMod2(float2(hexInt.x, hexInt.y + sy), float2(2.0f));

    float2 gdiff = gpos - hexInt;
    if (dot(abs(gdiff), float2(HW_HEX_FACTOR * HW_HEX_FACTOR, 0.5f)) > 1.0f) {
        float2 delta = float2(gdiff.x < 0.0f ? -1.0f : 1.0f, gdiff.y < 0.0f ? -1.0f : 1.0f);
        hexInt += delta * float2(2.0f, 1.0f);
    }

    return hexInt;
}

static float2 alignNormal(float2 h, float2 d) {
    float s = dot(h, hwCartFromHex(d)) < 0.0f ? -1.0f : 1.0f;
    return h * s;
}

static float3 rayHexIntersect(float2 ro, float2 rd, float2 h) {
    float2 n = hwCartFromHex(h);
    float denom = dot(n, rd);
    if (abs(denom) < 1.0e-5f) {
        return float3(h, 1.0e20f);
    }

    float u = (1.0f - dot(n, ro)) / denom;
    return float3(h, u > 0.0f ? u : 1.0e20f);
}

static float3 rayMin(float3 a, float3 b) {
    return a.z < b.z ? a : b;
}

static float3 hash32(float2 p) {
    float3 p3 = fract(float3(p.x, p.y, p.x) * float3(0.1031f, 0.1030f, 0.0973f));
    p3 += dot(p3, p3.yxz + 19.19f);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

static float heightForPos(float2 pos, float time) {
    pos += float2(2.0f * sin(time * 0.3f + 0.2f), 2.0f * cos(time * 0.1f + 0.5f));
    float x2 = dot(pos, pos);
    float x = sqrt(x2);
    return 6.0f * cos(x * 0.2f + time) * exp(-x2 / 128.0f);
}

static float3 hwEnvironment(float3 dir, float time) {
    dir = normalize(dir);
    float skyMix = clamp(0.5f + 0.5f * dir.z, 0.0f, 1.0f);
    float horizon = pow(max(1.0f - abs(dir.z), 0.0f), 3.0f);
    float rings = 0.5f + 0.5f * cos(7.0f * atan2(dir.y, dir.x) + 4.0f * dir.z + time * 0.7f);
    float3 sky = mix(float3(0.25f, 0.32f, 0.44f), HW_FOG_COLOR, skyMix);
    float3 glow = mix(float3(0.15f, 0.08f, 0.22f), float3(0.85f, 0.95f, 1.0f), rings);
    return sky + glow * horizon * 0.35f;
}

static float4 surface(float3 rd, float2 cell, float4 hitNT, float bdist, float time) {
    float fogc = exp(-length(hitNT.w * rd) * 0.02f);

    float3 n = hitNT.xyz;
    float3 noise = (hash32(cell) - 0.5f) * 0.15f;
    n = normalize(n + noise);

    float borderScale = 0.006f * max(1.0f, 0.1f * hitNT.w);
    const float borderSize = 0.04f;
    float border = smoothstep(0.0f, borderScale * hitNT.w + 1.0e-4f, abs(bdist) - borderSize);
    border = mix(border, 0.75f, smoothstep(18.0f, 45.0f, hitNT.w));

    float3 L = normalize(float3(3.0f, 1.0f, 4.0f));
    float diffamb = clamp(dot(n, L), 0.0f, 1.0f) * 0.8f + 0.2f;

    float3 color = float3(1.0f);
    color = mix(float3(0.1f, 0.0f, 0.08f), color, border);
    color *= diffamb;

    color = mix(color, hwEnvironment(reflect(rd, n), time), 0.4f * border);
    color = mix(HW_FOG_COLOR, color, fogc);

    return float4(color, border);
}

static float3 shade(float3 ro, float3 rd, float time) {
    float3 color = HW_FOG_COLOR;
    float2 curCell = nearestHexCell(ro.xy);

    float2 h0 = alignNormal(float2(0.0f, 1.0f), rd.xy);
    float2 h1 = alignNormal(float2(1.0f, 0.5f), rd.xy);
    float2 h2 = alignNormal(float2(1.0f, -0.5f), rd.xy);

    float cellHeight = heightForPos(hwCartFromHex(curCell), time);
    float alpha = 1.0f;

    for (int i = 0; i < 80; ++i) {
        bool hit = false;
        float4 hitNT = float4(0.0f);
        float bdist = 1.0e5f;

        float2 curCenter = hwCartFromHex(curCell);
        float2 rdelta = ro.xy - curCenter;

        float3 ht = rayHexIntersect(rdelta, rd.xy, h0);
        ht = rayMin(ht, rayHexIntersect(rdelta, rd.xy, h1));
        ht = rayMin(ht, rayHexIntersect(rdelta, rd.xy, h2));

        float tz = 1.0e20f;
        if (abs(rd.z) > 1.0e-5f) {
            tz = (cellHeight - ro.z) / rd.z;
        }

        if (ro.z > cellHeight && rd.z < 0.0f && tz > 0.0f && tz < ht.z) {
            hit = true;
            hitNT = float4(0.0f, 0.0f, 1.0f, tz);
            float2 pinter = ro.xy + rd.xy * tz;
            bdist = hexDist(pinter - curCenter);
        } else {
            curCell += 2.0f * ht.xy;

            float2 n = hwCartFromHex(ht.xy);
            curCenter = hwCartFromHex(curCell);

            float prevCellHeight = cellHeight;
            cellHeight = heightForPos(curCenter, time);

            float3 pIntersect = ro + rd * ht.z;
            if (pIntersect.z < cellHeight) {
                hitNT = float4(n, 0.0f, ht.z);
                hit = true;

                bdist = cellHeight - pIntersect.z;
                bdist = min(bdist, pIntersect.z - prevCellHeight);

                float2 p = pIntersect.xy - curCenter;
                p -= n * dot(p, n);
                bdist = min(bdist, abs(length(p) - 0.5f / HW_HEX_FACTOR));
            }
        }

        if (hit) {
            float4 hitColor = surface(rd, curCell, hitNT, bdist, time);
            color = mix(color, hitColor.xyz, alpha);
            alpha *= 0.17f * hitColor.w;

            ro = ro + rd * hitNT.w;
            rd = reflect(rd, hitNT.xyz);
            ro += 1.0e-3f * hitNT.xyz;

            h0 = alignNormal(float2(0.0f, 1.0f), rd.xy);
            h1 = alignNormal(float2(1.0f, 0.5f), rd.xy);
            h2 = alignNormal(float2(1.0f, -0.5f), rd.xy);
        }

        if (alpha < 0.01f) {
            break;
        }
    }

    color = mix(color, HW_FOG_COLOR, alpha);
    return color;
}

fragment float4 hexwavesFragment(
    HexwavesVertexOut in [[stage_in]],
    constant HexwavesUniforms &uniforms [[buffer(0)]],
    constant float4x4 *viewToWorld [[buffer(1)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = viewToWorld[vi];

    float3 center = uniforms.objectCenter.xyz;
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float cubeScale = max(uniforms.boxScale, 1.0e-4f);
    float3 eye = (camWorld - center) / cubeScale;
    float3 hit = (in.worldPos - center) / cubeScale;
    float3 rd = normalize(hit - eye);

    bool insideBox = all(abs(eye) < HW_BOX_HALF - 1.0e-3f);
    float2 tBox = hwBoxIntersect(eye, rd, HW_BOX_HALF);
    if (!insideBox && tBox.x > tBox.y) {
        discard_fragment();
    }

    float tStart = insideBox ? 0.0f : max(tBox.x, 0.0f);
    float3 marchOrigin = eye + rd * (tStart + HW_TRACE_EPSILON);

    float3 sceneRo = hwRotateScene(marchOrigin * 8.0f + float3(0.0f, 0.0f, 2.0f), uniforms.time);
    float3 sceneRd = normalize(hwRotateScene(rd, uniforms.time));

    float3 color = shade(sceneRo, sceneRd, uniforms.time);
    color = sqrt(max(color, 0.0f));

    float2 q = hwFaceUV(hit);
    color *= pow(max(16.0f * q.x * q.y * (1.0f - q.x) * (1.0f - q.y), 0.0f), 0.1f);
    return float4(clamp(color, 0.0f, 1.0f), 1.0f);
}