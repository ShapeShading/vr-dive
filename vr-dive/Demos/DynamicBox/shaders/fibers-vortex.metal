// fibers-vortex.metal
//
// Spiral thread columns that twist into a turbulent vortex,
// with rough matte highlights.

struct FiberHit {
    float d;
    float3 tangent;
    float id;
};

static float hash13(float3 p) {
    p = fract(p * 0.1031f);
    p += dot(p, p.yzx + 33.33f);
    return fract((p.x + p.y) * p.z);
}

static FiberHit mapVortexFibers(float3 p, float t) {
    FiberHit best;
    best.d = 1e9f;
    best.tangent = float3(0.0f, 1.0f, 0.0f);
    best.id = 0.0f;

    const int strandCount = 18;
    for (int i = 0; i < strandCount; i++) {
        float fi = float(i);
        float lane = fi / float(strandCount - 1);

        float spinRate = 5.0f + 1.2f * sin(t * 0.2f + fi);
        float spin = p.y * spinRate + fi * (DB_PI * 2.0f / float(strandCount)) + t * (0.8f + 0.05f * fi);

        float radius = 0.22f + 0.28f * lane + 0.04f * sin(p.y * 3.0f + fi * 1.7f + t);
        float2 cXZ = radius * float2(cos(spin), sin(spin));
        float yOffset = 0.03f * sin(fi + t + p.y * 4.0f);
        float3 c = float3(cXZ.x, p.y + yOffset, cXZ.y);

        float fiberRadius = mix(0.014f, 0.008f, lane);
        float d = length(p - c) - fiberRadius;

        if (d < best.d) {
            float drdy = 0.12f * cos(p.y * 3.0f + fi * 1.7f + t);
            float dxdy = drdy * cos(spin) - radius * sin(spin) * spinRate;
            float dzdy = drdy * sin(spin) + radius * cos(spin) * spinRate;
            float dydy = 1.0f + 0.12f * cos(fi + t + p.y * 4.0f);

            best.d = d;
            best.tangent = normalize(float3(dxdy, dydy, dzdy));
            best.id = fi;
        }
    }

    return best;
}

static float mapVortexDistance(float3 p, float t) {
    return mapVortexFibers(p, t).d;
}

static float3 calcVortexNormal(float3 p, float t) {
    float2 e = float2(0.003f, 0.0f);
    return normalize(float3(
        mapVortexDistance(p + e.xyy, t) - mapVortexDistance(p - e.xyy, t),
        mapVortexDistance(p + e.yxy, t) - mapVortexDistance(p - e.yxy, t),
        mapVortexDistance(p + e.yyx, t) - mapVortexDistance(p - e.yyx, t)
    ));
}

static float3 shadeVortexFiber(
    float3 p,
    float3 n,
    float3 rd,
    float3 tangent,
    float id,
    float march,
    float t)
{
    float3 lightA = normalize(float3(0.30f, 0.90f, 0.25f));
    float3 lightB = normalize(float3(-0.50f, 0.35f, 0.80f));
    float3 v = -rd;

    float diffuse = 0.16f + 0.66f * max(dot(n, lightA), 0.0f) + 0.28f * max(dot(n, lightB), 0.0f);

    float3 hA = normalize(v + lightA);
    float rough = pow(max(dot(n, hA), 0.0f), 9.0f) * 0.18f;
    float anis = pow(max(dot(tangent, hA), 0.0f), 18.0f) * 0.10f;
    float rim = pow(1.0f - max(dot(v, n), 0.0f), 2.0f);

    float tone = 0.5f + 0.5f * sin(id * 0.57f + t * 0.35f);
    float3 base = mix(float3(0.70f, 0.63f, 0.56f), float3(0.44f, 0.39f, 0.34f), tone);

    float dust = 0.76f + 0.24f * hash13(p * 48.0f + id * 0.19f + t * 0.4f);

    float3 color = base * diffuse;
    color += float3(0.92f) * rough;
    color += base * (0.20f * rim + anis);
    color *= dust;

    float fog = exp(-march * 0.06f);
    return color * fog;
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut        in         [[stage_in]],
    constant DynamicBoxUniforms &uniforms [[buffer(0)]],
    constant float4x4         *v2wMats    [[buffer(1)]],
    constant float4x4         *vpMatrices [[buffer(2)]])
{
    (void)vpMatrices;

    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);

    float4x4 v2w = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center = uniforms.objectCenter.xyz;
    float sc = uniforms.boxScale;
    float3 boxEye = (camWorld - center) / sc;
    float3 boxRd = normalize(in.worldPos - camWorld);

    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    float3 marchOrigin;

    if (!insideBox) {
        float3 entryNormal;
        float tEnter = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (tEnter < 0.0f) {
            return float4(0.02f, 0.017f, 0.014f, 1.0f);
        }
        marchOrigin = boxEye + boxRd * (tEnter + 1e-3f);
    } else {
        marchOrigin = boxEye;
    }

    float3 exitNormal;
    float tExit = db_boxHit(marchOrigin, boxRd, DB_BOXDIMS, exitNormal, false);
    if (tExit <= 0.0f) discard_fragment();

    float3 ro = (uniforms.patternTransform * float4(marchOrigin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRd, 0.0f)));

    float t = uniforms.time;
    float march = 0.0f;
    float maxMarch = 30.0f;

    for (int i = 0; i < 128; i++) {
        float3 p = ro + rd * march;
        FiberHit fh = mapVortexFibers(p, t);

        if (fh.d < 0.0026f) {
            float3 n = calcVortexNormal(p, t);
            float3 col = shadeVortexFiber(p, n, rd, fh.tangent, fh.id, march, t);
            return float4(col, 1.0f);
        }

        march += clamp(fh.d * 0.72f, 0.005f, 0.06f);
        if (march > maxMarch) break;
    }

    float glow = exp(-0.8f * length(ro.xz));
    float3 bg = mix(float3(0.012f, 0.010f, 0.009f), float3(0.030f, 0.020f, 0.015f), glow);
    return float4(bg, 1.0f);
}
