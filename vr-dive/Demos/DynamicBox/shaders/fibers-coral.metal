// fibers-coral.metal
//
// Coral-like filament colony: branching upward tendrils with
// rough matte thread shading.

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

static FiberHit mapCoralFibers(float3 p, float t) {
    FiberHit best;
    best.d = 1e9f;
    best.tangent = float3(0.0f, 1.0f, 0.0f);
    best.id = 0.0f;

    const int branchCount = 14;
    const float yMin = -0.92f;
    const float yMax = 0.95f;

    for (int i = 0; i < branchCount; i++) {
        float fi = float(i);
        float a = fi * (DB_PI * 2.0f / float(branchCount));

        float y = clamp(p.y, yMin, yMax);
        float h = (y - yMin) / (yMax - yMin);

        float flare = 0.06f + 0.55f * pow(max(h, 0.0001f), 1.45f);
        float branch = 0.09f * sin(6.0f * h + t * 0.60f + fi * 1.30f) * (0.3f + 0.7f * h);

        float swayX = 0.06f * sin(y * 3.5f + t * 0.40f + fi);
        float swayZ = 0.06f * cos(y * 3.2f - t * 0.50f + fi * 0.70f);

        float radial = flare + branch;
        float3 c = float3(
            cos(a) * radial + swayX,
            y,
            sin(a) * radial + swayZ
        );

        float r = mix(0.014f, 0.008f, h) + 0.0015f * sin(fi + t * 0.2f);
        float d = length(p - c) - r;

        if (d < best.d) {
            float dhdy = 1.0f / (yMax - yMin);
            float dFlareDy = 0.55f * 1.45f * pow(max(h, 0.0001f), 0.45f) * dhdy;
            float wave = sin(6.0f * h + t * 0.60f + fi * 1.30f);
            float dWaveDy = 6.0f * dhdy * cos(6.0f * h + t * 0.60f + fi * 1.30f);
            float dBranchDy = 0.09f * (dWaveDy * (0.3f + 0.7f * h) + wave * 0.7f * dhdy);
            float dRadialDy = dFlareDy + dBranchDy;

            float dxdy = cos(a) * dRadialDy + 0.21f * cos(y * 3.5f + t * 0.40f + fi);
            float dzdy = sin(a) * dRadialDy - 0.192f * sin(y * 3.2f - t * 0.50f + fi * 0.70f);

            best.d = d;
            best.tangent = normalize(float3(dxdy, 1.0f, dzdy));
            best.id = fi;
        }
    }

    return best;
}

static float mapCoralDistance(float3 p, float t) {
    return mapCoralFibers(p, t).d;
}

static float3 calcCoralNormal(float3 p, float t) {
    float2 e = float2(0.003f, 0.0f);
    return normalize(float3(
        mapCoralDistance(p + e.xyy, t) - mapCoralDistance(p - e.xyy, t),
        mapCoralDistance(p + e.yxy, t) - mapCoralDistance(p - e.yxy, t),
        mapCoralDistance(p + e.yyx, t) - mapCoralDistance(p - e.yyx, t)
    ));
}

static float3 shadeCoralFiber(
    float3 p,
    float3 n,
    float3 rd,
    float3 tangent,
    float id,
    float march,
    float t)
{
    float3 lightA = normalize(float3(0.40f, 0.88f, 0.28f));
    float3 lightB = normalize(float3(-0.34f, 0.25f, 0.91f));
    float3 v = -rd;

    float diffuse = 0.20f + 0.62f * max(dot(n, lightA), 0.0f) + 0.22f * max(dot(n, lightB), 0.0f);

    float3 hA = normalize(v + lightA);
    float rough = pow(max(dot(n, hA), 0.0f), 8.0f) * 0.20f;
    float anis = pow(max(dot(tangent, hA), 0.0f), 16.0f) * 0.08f;
    float rim = pow(1.0f - max(dot(v, n), 0.0f), 2.2f);

    float tint = 0.5f + 0.5f * sin(id * 0.91f + t * 0.22f);
    float3 base = mix(float3(0.80f, 0.62f, 0.48f), float3(0.58f, 0.44f, 0.34f), tint);

    float grain = 0.74f + 0.26f * hash13(p * 52.0f + id * 0.33f + t * 0.3f);

    float3 color = base * diffuse;
    color += float3(0.94f) * rough;
    color += base * (0.24f * rim + anis);
    color *= grain;

    float fog = exp(-march * 0.05f);
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
            return float4(0.02f, 0.015f, 0.012f, 1.0f);
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

    for (int i = 0; i < 120; i++) {
        float3 p = ro + rd * march;
        FiberHit fh = mapCoralFibers(p, t);

        if (fh.d < 0.0027f) {
            float3 n = calcCoralNormal(p, t);
            float3 col = shadeCoralFiber(p, n, rd, fh.tangent, fh.id, march, t);
            return float4(col, 1.0f);
        }

        march += clamp(fh.d * 0.72f, 0.005f, 0.06f);
        if (march > maxMarch) break;
    }

    float horizon = 0.5f + 0.5f * rd.y;
    float3 bg = mix(float3(0.014f, 0.012f, 0.010f), float3(0.026f, 0.020f, 0.016f), horizon);
    return float4(bg, 1.0f);
}
