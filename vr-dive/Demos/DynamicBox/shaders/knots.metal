// knots.metal – Mathematical Knot Surfaces
//
// Renders 3D knot surfaces: trefoil knot, torus knot, and figure-8
// knot. These are parametric curves thickened into tubes using SDF
// techniques. Colors flow along the knot path.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 用参数方程近似求「torus knot」曲线上离查询点最近的参数 u（先粗
//       筛 20 个采样点，再做 2 次二分式精修），以该点为中心画细管
//       (tubeR) 得到 SDF；同时叠加两条不同 (p,q) 参数的纽结 (2,3) 与
//       (3,5) 并取 min，实现「互相缠绕」的视觉效果。
// 关键参数:
//   - tubeR = 0.04f / 0.03f：两条纽结管的粗细，决定「丝线」的视觉重量。
//   - R = 0.55：纽结缠绕的主半径；(p,q) 决定缠绕圈数比例（经典纽结
//     分类）。
// 性能特征: 每次 SDF 求值需 20(粗筛) + 5×2(精修) ≈ 30 次三角函数评估，
//           法线额外 6 次；march 80 步/maxD=25，是本目录里参数曲线类
//           shader 中最贵的一种，建议关注其 perf 抽样日志。
// 已知限制/优化方向:
//   - 粗筛用固定 20 段可能在纽结缠绕较快的区域漏检最近点，如画面出现
//     断裂可提高采样段数或改用解析式最近点近似。

// ─── Torus knot SDF ───────────────────────────────────────────────────────────
// A (p,q) torus knot wraps around a torus p times in one direction
// and q times in the other. The trefoil is (2,3).
static float torusKnotSDF(float3 p, float2 pq, float tubeR, float t) {
    float p1 = pq.x, q1 = pq.y;

    // Approximate closest point on the torus knot curve
    // We do a few iteration steps to refine the closest point
    float theta = atan2(p.z, p.x);
    float3 best = float3(1e10f, 0, 0);

    float R = 0.55f; // major radius

    for (int i = 0; i < 20; i++) {
        float u = theta + float(i) / 20.0f * 6.28318f;

        // Parametric torus knot
        float cu = cos(u), su = sin(u);
        float v = u * q1 / p1;
        float cv = cos(v), sv = sin(v);

        float3 kp = float3(
            (R + 0.2f * cv) * cu,
            0.2f * sv,
            (R + 0.2f * cv) * su
        );

        float3 d = p - kp;
        float  dist = length(d);
        if (dist < best.x) {
            best = float3(dist, u, 0);
        }
    }

    // Refine around the best guess
    float u = best.y;
    float step = 0.1f;
    for (int i = 0; i < 5; i++) {
        float cu = cos(u), su = sin(u);
        float v = u * q1 / p1;
        float cv = cos(v), sv = sin(v);
        float3 kp = float3(
            (R + 0.2f * cv) * cu, 0.2f * sv, (R + 0.2f * cv) * su);
        float3 d = p - kp;
        float dist = length(d);

        // Try nearby parameters
        float u2 = u + step;
        cu = cos(u2); su = sin(u2);
        v = u2 * q1 / p1; cv = cos(v); sv = sin(v);
        float3 kp2 = float3((R + 0.2f*cv)*cu, 0.2f*sv, (R + 0.2f*cv)*su);
        float d2 = length(p - kp2);

        if (d2 < dist) {
            u = u2; dist = d2;
        } else {
            u2 = u - step;
            cu = cos(u2); su = sin(u2);
            v = u2 * q1 / p1; cv = cos(v); sv = sin(v);
            kp2 = float3((R + 0.2f*cv)*cu, 0.2f*sv, (R + 0.2f*cv)*su);
            d2 = length(p - kp2);
            if (d2 < dist) { u = u2; dist = d2; }
        }
        step *= 0.5f;
    }

    return length(p - float3((R + 0.2f*cos(u*q1/p1))*cos(u),
                              0.2f*sin(u*q1/p1),
                              (R + 0.2f*cos(u*q1/p1))*sin(u))) - tubeR;
}

// ─── Combined: one trefoil (2,3) and one (3,5) knot interlinked ──────────────
static float knotsSDF(float3 p, float t) {
    // Slow rotation
    float ca = cos(t * 0.12f), sa = sin(t * 0.12f);
    float3 q = p;
    q.xz = float2(q.x*ca - q.z*sa, q.x*sa + q.z*ca);

    float d1 = torusKnotSDF(q, float2(2, 3), 0.04f, t);
    float d2 = torusKnotSDF(q + float3(0.3f, 0, 0), float2(3, 5), 0.03f, t);
    return min(d1, d2);
}

static float3 calcNormal(float3 p, float t) {
    float2 e = float2(0.003f, 0.0f);
    return normalize(float3(
        knotsSDF(p + e.xyy, t) - knotsSDF(p - e.xyy, t),
        knotsSDF(p + e.yxy, t) - knotsSDF(p - e.yxy, t),
        knotsSDF(p + e.yyx, t) - knotsSDF(p - e.yyx, t)
    ));
}

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
    float3 bgColor = float3(0.01f, 0.0f, 0.02f);

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

    float t = uniforms.time;
    float march = 0.0f;
    float maxD = 25.0f;

    for (int i = 0; i < 80; i++) {
        float3 p = ro + rd * march;
        float d = knotsSDF(p, t);
        if (d < 0.004f) {
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.4f, 0.7f, 0.6f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 3.0f) * 0.6f;

            // Metallic color along the knot
            float hue = fract(march * 0.3f + t * 0.04f);
            float3 col;
            { float4 K = float4(1,2/3.f,1/3.f,3); float3 pp = abs(fract(hue+K.xyz)*6-K.www); col = clamp(pp-K.xxx,0,1); }
            col = col * (dif * 1.3f + amb * 0.3f) + float3(0.2f, 0.4f, 0.9f) * rim;
            col *= exp(-march * 0.15f);
            return float4(col, 1.0f);
        }
        march += max(d, 0.01f);
        if (march > maxD) break;
    }
    return float4(bgColor, 1.0f);
}
