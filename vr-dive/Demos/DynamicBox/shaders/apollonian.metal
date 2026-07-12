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
// 思路: 经典「domain-repeat Apollonian」DE：每次迭代先把点折回
//       [-1,1]³ 的周期性瓦片 (q = -1+2*fract(0.5q+0.5))，再做一次
//       「钳位球面反演」(k = max(fixedRadius2/r², 1.0); q*=k; scale*=k)。
//       周期折叠保证 q 永远不会跑飞，钳位反演保证 scale 单调增长，最终
//       用 length(q)/scale 作为距离场，能稳定生成"球中球"的无限堆叠。
// 关键参数:
//   - fixedRadius2 = 1.0：球面反演的触发半径平方，越大单次反演放大倍数
//     越强，细节越密。
//   - 迭代次数 10：domain-repeat 变体收敛快，不需要像早期实现那样堆到
//     14 次也能出丰富细节。
//   - 末尾 -0.02 是"球体半径"，决定每个小球的粗细/棱角锐度。
// 性能特征: 单点 DE 迭代 10 次（每次 1 次 fract + 1 次 max + 2 次乘法），
//           法线额外 6 次 DE 调用；march 120 步/maxD=25，中等开销。
// 已知限制/优化方向（⚠️ 曾踩坑记录）:
//   - 最初的实现每次迭代无条件做 `q = q/dot(q,q)`（无周期折叠、无钳位），
//     导致 q 在若干次迭代后要么迅速发散到极大值、要么塌缩到 0，且额外的
//     `scale *= 3` 与位移叠加会进一步放大这种不稳定性。结果是距离场在
//     摄像机所在的盒子尺度内几乎处处远离 0，march 命中率极低，画面看起来
//     「几乎是空的」。改用带周期折叠 + 钳位反演的经典写法后问题解决。
//   - 目前没有做「orbit trap」之类按迭代轨迹着色，只用 palette(length(p))，
//     细节层次的色彩变化有限，可尝试引入 orbit trap 提升可读性。

// ─── Apollonian DE ────────────────────────────────────────────────────────────
// Domain-repeat Apollonian gasket: fold space into a periodic tile, then
// apply a clamped sphere inversion each iteration. This variant is numerically
// stable (bounded q, monotonically growing scale) and reliably fills space.
static float apollonianDE(float3 p, float t) {
    float3 q = p;
    float  scale = 1.0f;
    const float fixedRadius2 = 1.0f;

    for (int i = 0; i < 10; i++) {
        // Domain repeat: fold space back into a [-1,1]^3 tile.
        q = -1.0f + 2.0f * fract(0.5f * q + 0.5f);

        // Clamped sphere inversion (only ever grows scale, keeps q bounded).
        float r2 = dot(q, q);
        float k = max(fixedRadius2 / max(r2, 1e-6f), 1.0f);
        q *= k;
        scale *= k;
    }

    return length(q) / scale - 0.02f;
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
