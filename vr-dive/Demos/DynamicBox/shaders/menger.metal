// menger.metal – Menger Sponge Fractal
//
// A classic 3D fractal: a cube with recursively removed sub-cubes,
// creating an infinitely detailed sponge-like structure.
// Rendered with ray marching and distance estimation.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 标准 Menger sponge DE：每级先取绝对值 + 3 次冒泡排序让坐标单调，
//       再用 3 个 max() 组合出十字形镂空区域，从实心方块中「挖空」，最后
//       缩放 3 倍进入下一级，共 6 级。
// 关键参数:
//   - 挖空阈值 1/3：十字形镂空的宽度比例，决定海绵孔洞的疏密（标准值，
//     不建议随意更改否则会破坏自相似性）。
//   - 迭代 6 级：级数越高细节越多，但每级都线性放大法线计算的误差敏感度。
// 性能特征: 每次 DE 6 次迭代（排序 + 两次 max 合并），法线 6 次求值；
//           march 120 步/maxD=25，中等开销，是分形中较「便宜」的一种。
// 已知限制/优化方向:
//   - 目前只有全局慢速旋转 (ca/sa)，可尝试让镂空阈值随时间轻微呼吸，
//     做出「海绵在生长/收缩」的动态效果。

// ─── Menger sponge DE ─────────────────────────────────────────────────────────
static float mengerDE(float3 p) {
    float3 q = p;
    float  d = 1.0f;

    for (int i = 0; i < 6; i++) {
        q = abs(q);
        if (q.x < q.y) { float t = q.x; q.x = q.y; q.y = t; }
        if (q.x < q.z) { float t = q.x; q.x = q.z; q.z = t; }
        if (q.y < q.z) { float t = q.y; q.y = q.z; q.z = t; }

        // Three cross-shaped holes (closest to the current folded point)
        float d1 = max(q.y - 1.0f / 3.0f, q.z - 1.0f / 3.0f);
        float d2 = max(q.x - 1.0f / 3.0f, q.z - 1.0f / 3.0f);
        float d3 = max(q.x - 1.0f / 3.0f, q.y - 1.0f / 3.0f);
        float cross = min(d1, min(d2, d3));

        // Subtract cross from the solid cube
        d = max(d, -cross);

        // Scale for next iteration
        q = q * 3.0f - 2.0f;
        d *= 3.0f;
    }

    return (length(q) - 2.0f) / d;
}

// ─── Normal ───────────────────────────────────────────────────────────────────
static float3 calcNormal(float3 p) {
    float2 e = float2(0.0015f, 0.0f);
    return normalize(float3(
        mengerDE(p + e.xyy) - mengerDE(p - e.xyy),
        mengerDE(p + e.yxy) - mengerDE(p - e.yxy),
        mengerDE(p + e.yyx) - mengerDE(p - e.yyx)
    ));
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
    float3 bgColor = float3(0.01f, 0.01f, 0.02f);

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

    // ─── Ray march ─────────────────────────────────────────────────────────
    float t     = uniforms.time;
    float march = 0.0f;
    float maxD  = 25.0f;

    // Slow rotation
    float ca = cos(t * 0.1f), sa = sin(t * 0.1f);
    ro.xz = float2(ro.x*ca - ro.z*sa, ro.x*sa + ro.z*ca);
    rd.xz = float2(rd.x*ca - rd.z*sa, rd.x*sa + rd.z*ca);

    for (int i = 0; i < 120; i++) {
        float3 p = ro + rd * march;
        float d = mengerDE(p);
        if (d < 0.003f) {
            float3 n = calcNormal(p);
            float3 light = normalize(float3(0.5f, 1.0f, 0.4f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.2f + 0.8f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 3.0f) * 0.5f;

            // Gold-ish metallic color with blue edges
            float3 col = float3(0.8f, 0.6f, 0.2f) * (dif * 1.2f + amb * 0.3f)
                       + float3(0.2f, 0.4f, 1.0f) * rim;
            // Fresnel edge glow
            col += float3(0.3f, 0.5f, 0.8f) * rim * 0.5f;
            col *= exp(-march * 0.25f);

            return float4(col, 1.0f);
        }
        march += d;
        if (march > maxD) break;
    }

    return float4(bgColor, 1.0f);
}
