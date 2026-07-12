// sierpinski5cell.metal – 4D Sierpinski Simplex (5-Cell IFS Fractal)
//
// A different fractal-construction technique from the escape-time Mandelbulb/
// Julia family: an Iterated Function System (IFS) toward the five vertices of
// a 4D simplex ("5-cell"), the 4D analog of the Sierpinski tetrahedron. Each
// iteration contracts toward whichever of the five vertices is nearest,
// building the classic chaos-game fractal directly as a distance estimator.
// The fractal's W-axis slice is driven directly by time (not by perspective
// projection), so you see genuinely different 4D cross-sections over time
// rather than a rotated view of the same shape.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 5-cell（4D 单纯形）的 5 个顶点各自是单位四维向量；每次迭代找出
//       离当前点最近的顶点，做「朝该顶点折叠再放大 2 倍」的经典
//       Sierpinski IFS 变换 (q = (q - vBest) * 2)，累计 scale，最终
//       distance = length(q)/scale 作为 SDF。W 分量在进入折叠前先加上一个
//       随时间缓慢变化的偏移，从而在不同时间切出 4D 分形的不同 3D 截面。
// 关键参数:
//   - 5 个顶点使用精确单位长度的字面量常量（而非 normalize() 调用），
//     避免 file-scope 初始化调用函数导致的 llvm.global_ctors 问题。
//   - 迭代 8 次、缩放系数 2：控制细节层级与整体尺寸的收敛速度。
//   - wOffset = 0.6*sin(t*0.15)：4D 切片随时间缓慢平移，比单纯旋转更能
//     体现「切开 4D 物体看到不同结构」的观感。
// 性能特征: 每次 DE 迭代 8 次，每次比较 5 个顶点距离（5 次 distance +
//           4 次条件判断），无三角函数；march 85 步/maxD=25，中等偏轻。
// 已知限制/优化方向:
//   - 5 个顶点不是严格正则的 4D 单纯形（为避免额外 sqrt 计算而手工挑选
//     了单位向量），视觉上依然呈现清晰的自相似枝干结构；如需完全正则
//     单纯形可换成精确等距顶点坐标。

// ─── Sierpinski 5-cell DE ──────────────────────────────────────────────────────
static float sierpinskiDE(float3 p, float wOffset) {
    float4 q = float4(p, wOffset);
    float  scale = 1.0f;
    const float s = 2.0f;

    // Unit-length vertices of a (non-regular but sufficiently symmetric)
    // 4D simplex, precomputed as literals to avoid file-scope normalize().
    float4 v0 = float4( 0.5f,  0.5f,  0.5f,  0.5f);
    float4 v1 = float4( 0.5f, -0.5f, -0.5f,  0.5f);
    float4 v2 = float4(-0.5f,  0.5f, -0.5f,  0.5f);
    float4 v3 = float4(-0.5f, -0.5f,  0.5f,  0.5f);
    float4 v4 = float4( 0.0f,  0.0f,  0.0f, -1.0f);

    for (int i = 0; i < 8; i++) {
        float4 vBest = v0;
        float  dBest = distance(q, v0);
        float  d1 = distance(q, v1); if (d1 < dBest) { dBest = d1; vBest = v1; }
        float  d2 = distance(q, v2); if (d2 < dBest) { dBest = d2; vBest = v2; }
        float  d3 = distance(q, v3); if (d3 < dBest) { dBest = d3; vBest = v3; }
        float  d4 = distance(q, v4); if (d4 < dBest) { dBest = d4; vBest = v4; }

        q = (q - vBest) * s;
        scale *= s;
    }

    return length(q) / scale - 0.015f;
}

static float3 calcNormal(float3 p, float wOffset) {
    float2 e = float2(0.0018f, 0.0f);
    return normalize(float3(
        sierpinskiDE(p + e.xyy, wOffset) - sierpinskiDE(p - e.xyy, wOffset),
        sierpinskiDE(p + e.yxy, wOffset) - sierpinskiDE(p - e.yxy, wOffset),
        sierpinskiDE(p + e.yyx, wOffset) - sierpinskiDE(p - e.yyx, wOffset)
    ));
}

static float3 palette(float t) {
    float3 a = float3(0.5f, 0.45f, 0.4f);
    float3 b = float3(0.5f, 0.5f, 0.5f);
    float3 c = float3(0.8f, 1.0f, 0.6f);
    float3 d = float3(0.15f, 0.05f, 0.3f);
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
    float3 bgColor = float3(0.02f, 0.01f, 0.015f);

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
    float wOffset = 0.6f * sin(t * 0.15f);
    float march = 0.0f;
    float maxD  = 25.0f;

    for (int i = 0; i < 85; i++) {
        float3 p = ro + rd * march;
        float d = sierpinskiDE(p, wOffset);
        if (d < 0.003f) {
            float3 n = calcNormal(p, wOffset);
            float3 light = normalize(float3(0.35f, 0.9f, 0.25f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = pow(1.0f - max(dot(-rd, n), 0.0f), 3.2f) * 0.5f;

            float colT = length(p) * 0.7f + t * 0.04f + wOffset * 0.3f;
            float3 col = palette(colT) * (dif * 1.05f + amb * 0.45f) + float3(0.3f, 0.9f, 0.5f) * rim;
            col *= exp(-march * 0.22f);
            return float4(col, 1.0f);
        }
        march += max(d * 0.75f, 0.0015f);
        if (march > maxD) break;
    }
    return float4(bgColor, 1.0f);
}
