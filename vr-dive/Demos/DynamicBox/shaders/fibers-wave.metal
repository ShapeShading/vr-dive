// fibers-wave.metal
//
// Dense wave-driven fiber bundles with frosted thread shading.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: ±8 共 17 条丝线沿 X 轴排列，每条丝线的 y/z 位置由不同相位的
//       sin/cos 组合决定，形成水平方向的「波浪丝束」；解析求出切线供
//       各向异性高光使用。
// 关键参数:
//   - fiberRadius = 0.012（固定，不随丝线变化）。
//   - phase = fi * 0.63：相邻丝线的相位差，决定波浪起伏的错落感。
// 性能特征: 每次 SDF 求值遍历 17 条丝线，法线额外 6 次；march 120
//           步/maxD=30，与 fibers-coral 接近，中等偏高开销。
// 已知限制/优化方向:
//   - 目前所有丝线共享同一组频率 (2.10f/1.70f)，只有相位不同，波形略显
//     规律；如需更自然的水波感可以让频率也随 fi 轻微抖动。

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

static FiberHit mapWaveFibers(float3 p, float t) {
    FiberHit best;
    best.d = 1e9f;
    best.tangent = float3(1.0f, 0.0f, 0.0f);
    best.id = 0.0f;

    const float fiberRadius = 0.012f;
    for (int i = -8; i <= 8; i++) {
        float fi = float(i);
        float phase = fi * 0.63f;

        float y = 0.28f * sin(p.x * 2.10f + phase + t * 0.90f) + fi * 0.065f;
        float z = 0.24f * cos(p.x * 1.70f - phase * 1.20f + t * 0.55f);

        float3 c = float3(p.x, y, z);
        float d = length(p - c) - fiberRadius;
        if (d < best.d) {
            float dy = 0.28f * 2.10f * cos(p.x * 2.10f + phase + t * 0.90f);
            float dz = -0.24f * 1.70f * sin(p.x * 1.70f - phase * 1.20f + t * 0.55f);
            best.d = d;
            best.tangent = normalize(float3(1.0f, dy, dz));
            best.id = fi;
        }
    }

    return best;
}

static float mapWaveDistance(float3 p, float t) {
    return mapWaveFibers(p, t).d;
}

static float3 calcWaveNormal(float3 p, float t) {
    float2 e = float2(0.003f, 0.0f);
    return normalize(float3(
        mapWaveDistance(p + e.xyy, t) - mapWaveDistance(p - e.xyy, t),
        mapWaveDistance(p + e.yxy, t) - mapWaveDistance(p - e.yxy, t),
        mapWaveDistance(p + e.yyx, t) - mapWaveDistance(p - e.yyx, t)
    ));
}

static float3 shadeFrostedFiber(
    float3 p,
    float3 n,
    float3 rd,
    float3 tangent,
    float fiberId,
    float march,
    float t)
{
    float3 lightA = normalize(float3(0.45f, 0.85f, 0.35f));
    float3 lightB = normalize(float3(-0.60f, 0.40f, 0.70f));
    float3 v = -rd;

    float difA = max(dot(n, lightA), 0.0f);
    float difB = max(dot(n, lightB), 0.0f);
    float diffuse = 0.18f + difA * 0.62f + difB * 0.30f;

    float3 h = normalize(lightA + v);
    float roughSpec = pow(max(dot(n, h), 0.0f), 10.0f) * 0.16f;

    float rim = pow(1.0f - max(dot(v, n), 0.0f), 2.1f);
    float fiberScatter = pow(max(dot(tangent, h), 0.0f), 22.0f) * 0.08f;

    float hueBand = 0.5f + 0.5f * sin(fiberId * 0.44f + t * 0.2f);
    float3 baseA = float3(0.58f, 0.70f, 0.78f);
    float3 baseB = float3(0.36f, 0.50f, 0.64f);
    float3 base = mix(baseA, baseB, hueBand);

    float grain = 0.78f + 0.22f * hash13(p * 45.0f + fiberId * 0.37f + t);

    float3 color = base * diffuse;
    color += float3(0.95f) * roughSpec;
    color += base * (0.22f * rim + fiberScatter);
    color *= grain;

    float fog = exp(-march * 0.055f);
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
            return float4(0.015f, 0.02f, 0.03f, 1.0f);
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
        FiberHit fh = mapWaveFibers(p, t);

        if (fh.d < 0.0028f) {
            float3 n = calcWaveNormal(p, t);
            float3 col = shadeFrostedFiber(p, n, rd, fh.tangent, fh.id, march, t);
            return float4(col, 1.0f);
        }

        march += clamp(fh.d * 0.75f, 0.005f, 0.06f);
        if (march > maxMarch) break;
    }

    float sky = 0.5f + 0.5f * rd.y;
    float3 bg = mix(float3(0.012f, 0.015f, 0.022f), float3(0.02f, 0.028f, 0.04f), sky);
    return float4(bg, 1.0f);
}
