// apollonian.metal – Apollonian Gasket (3D Sphere Fractal)
//
// Renders an Apollonian gasket – a fractal of nested spheres – using
// ray marching. Each iteration inverts space through spheres, creating
// an infinitely detailed packing of spheres within spheres.
//
// Based on the standard Apollonian fractal DE popularized by
// Syntopia / FractalForums.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 标准 Apollonian gasket 距离估计器 (DE)。每次迭代对点做球面反演
//       (q = q / |q|²)，再取绝对值做镜像折叠，最后整体平移 + 缩放，
//       14 次迭代后用 length(q)/dr 作为距离场。
// 关键参数:
//   - scale = 3.0：每次迭代的缩放倍数，决定球体嵌套的密度。
//   - 迭代次数 14：越多细节越丰富，但法线计算(6 次求值)成本也线性增加。
//   - march 步进直接用原始 d（未做安全系数缩放），因为 DE 本身已是保守估计。
// 性能特征: 单点 DE 迭代 14 次 + 有限差分法线 ×6 次 DE 调用；march 上限
//           120 步、maxD=25。整体开销中等偏高，属于本目录里较重的分形。
// 已知限制/优化方向:
//   - 目前没有做「orbit trap」之类按迭代轨迹着色，只用 palette(length(p))，
//     细节层次的色彩变化有限，可尝试引入 orbit trap 提升可读性。
//   - 可尝试提前判断 m 过小时 break（已有），或增加自适应步进系数以降低
//     远处采样开销。

// ─── Apollonian DE ────────────────────────────────────────────────────────────
// Standard Apollonian gasket: iterative sphere inversion + fold + scale.
static float apollonianDE(float3 p, float t) {
    float3 q = p;
    float  dr = 1.0f;
    float  scale = 3.0f;

    for (int i = 0; i < 14; i++) {
        // Sphere inversion: reflect across unit sphere
        float m = dot(q, q);
        if (m < 1e-8f) break;

        q = q / m;
        dr = dr / m * 2.0f + 1.0f;

        // Symmetry fold
        q = abs(q);

        // Offset then scale
        q = q + float3(0.5f, 0.5f, 0.5f);
        q = q * scale;
        dr = dr * scale + 1.0f;
    }

    return length(q) / dr - 0.05f;
}

// ─── Normal ───────────────────────────────────────────────────────────────────
static float3 calcNormal(float3 p, float t) {
    float2 e = float2(0.002f, 0.0f);
    return normalize(float3(
        apollonianDE(p + e.xyy, t) - apollonianDE(p - e.xyy, t),
        apollonianDE(p + e.yxy, t) - apollonianDE(p - e.yxy, t),
        apollonianDE(p + e.yyx, t) - apollonianDE(p - e.yyx, t)
    ));
}

// ─── Color palette ────────────────────────────────────────────────────────────
static float3 palette(float t) {
    float3 a = float3(0.5f, 0.5f, 0.5f);
    float3 b = float3(0.5f, 0.5f, 0.5f);
    float3 c = float3(1.0f, 0.7f, 0.4f);
    float3 d = float3(0.00f, 0.15f, 0.20f);
    return a + b * cos(6.28318f * (c * t + d));
}

// ─── Fragment shader ──────────────────────────────────────────────────────────
fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut        in         [[stage_in]],
    constant DynamicBoxUniforms &uniforms [[buffer(0)]],
    constant float4x4         *v2wMats    [[buffer(1)]],
    constant float4x4         *vpMatrices [[buffer(2)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);

    float4x4 v2w    = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center  = uniforms.objectCenter.xyz;
    float  sc      = uniforms.boxScale;
    float3 boxEye  = (camWorld - center) / sc;
    float3 boxRd   = normalize(in.worldPos - camWorld);

    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    float3 marchOrigin;
    float3 bgColor = float3(0.0f, 0.0f, 0.02f);

    if (!insideBox) {
        float3 entryNormal;
        float  tEnter = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (tEnter < 0.0f) return float4(bgColor, 1.0f);
        marchOrigin = boxEye + boxRd * (tEnter + 1e-3f);
    } else {
        marchOrigin = boxEye;
    }

    float3 exitNormal;
    float  tExit = db_boxHit(marchOrigin, boxRd, DB_BOXDIMS, exitNormal, false);
    if (tExit <= 0.0f) discard_fragment();

    float3 ro = (uniforms.patternTransform * float4(marchOrigin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRd, 0.0f)));

    // Slow rotation for visual interest
    float t = uniforms.time;
    float ca = cos(t * 0.07f), sa = sin(t * 0.07f);
    ro.xz = float2(ro.x*ca - ro.z*sa, ro.x*sa + ro.z*ca);
    rd.xz = float2(rd.x*ca - rd.z*sa, rd.x*sa + rd.z*ca);
    float cb = cos(t * 0.05f), sb = sin(t * 0.05f);
    ro.xy = float2(ro.x*cb - ro.y*sb, ro.x*sb + ro.y*cb);
    rd.xy = float2(rd.x*cb - rd.y*sb, rd.x*sb + rd.y*cb);

    // ─── Ray march ─────────────────────────────────────────────────────────
    float march = 0.0f;
    float maxD  = 25.0f;

    for (int i = 0; i < 120; i++) {
        float3 p = ro + rd * march;
        float d = apollonianDE(p, t);
        if (d < 0.003f) {
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.5f, 1.0f, 0.3f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 3.0f) * 0.6f;

            // Color by iteration depth + position
            float colT = length(p) * 0.3f + t * 0.03f;
            float3 col = palette(colT) * (dif * 1.2f + amb * 0.4f) + float3(0.2f, 0.4f, 1.0f) * rim;
            col *= exp(-march * 0.25f);

            return float4(col, 1.0f);
        }
        march += d;
        if (march > maxD) break;
    }

    return float4(bgColor, 1.0f);
}
