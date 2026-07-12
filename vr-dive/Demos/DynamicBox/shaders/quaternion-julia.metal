// quaternion-julia.metal – Quaternion Julia Set (4D Escape-Time Fractal)
//
// A classic higher-dimensional fractal distinct from the power-8 Mandelbulb:
// iterates z → z² + c directly in quaternion algebra (4D hypercomplex
// numbers) instead of spherical coordinates. The 3-space ray-marched point
// becomes the (x, y, z) part of a quaternion with w = 0; a slowly drifting
// quaternion constant `c` reshapes the Julia set continuously over time.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 标准四元数 Julia set DE：把 march 采样点 p 当作四元数
//       z=(p.x,p.y,p.z,0)，每次迭代先用四元数导数递推 dz' = 2·z·dz
//       （四元数乘法非交换，必须用 qmul 而非逐分量乘），再算 z = z² + c；
//       c 由时间参数缓慢漂移，让同一段 DE 代码呈现不断变化的 Julia 形态。
// 关键参数:
//   - c 基准值 (-0.2, 0.55, 0.2, 0.15)，附加 ±0.12 的时间调制，保持在
//     已知能产生丰富连通分形边界的范围内漂移，避免退化成单点或全空。
//   - 迭代 11 次（原 9 次）、逃逸阈值 |z|²>16：实测上一版 9 次迭代在
//     shader-performance.log 中的慢帧样本约 34–57ms（均值约 44ms），与
//     mandelbulb（约 33ms）、menger（约 30–40ms）、sierpinski5cell（约
//     32–35ms）同属「正常」区间，明显低于当时被判定需要优化的 nebula
//     （109–144ms）等 shader；因此有余量把迭代数提高到 11 次以加密分形
//     边界细节，同时把 march 步进系数从 0.8 提到 0.85（配合更精确的 DE
//     导数更快收敛）来部分抵消额外迭代的开销。
// 性能特征: 每次迭代 1 次 qsquare（4 次乘加）+ 1 次 qmul（16 次乘加）
//           求导数，共 11 次；法线额外 6 次 DE 调用；march 90 步/maxD=25。
//           预期新的慢帧均值比旧版本略高，但仍应显著低于 100ms 量级。
// 已知限制/优化方向:
//   - 目前固定在 w=0 的 3D 切片上取值，可尝试让 w 随时间独立偏移（类似
//     hyper4d 的 4D 旋转）以展示四元数 Julia 集在 w 方向上的其他切片。
//   - 若后续性能抽样显示慢帧均值明显超过同类 shader（如接近或超过
//     nebula 的 100ms+ 量级），应优先回退迭代数到 9–10 次，而不是继续
//     加密细节。

// ─── Quaternion helpers ────────────────────────────────────────────────────────
static float4 qmul(float4 a, float4 b) {
    return float4(
        a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
        a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
        a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
        a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x
    );
}

static float4 qsquare(float4 q) {
    return float4(
        q.x*q.x - q.y*q.y - q.z*q.z - q.w*q.w,
        2.0f*q.x*q.y,
        2.0f*q.x*q.z,
        2.0f*q.x*q.w
    );
}

// ─── Quaternion Julia DE ───────────────────────────────────────────────────────
static float juliaDE(float3 p, float4 c) {
    float4 z  = float4(p, 0.0f);
    float4 dz = float4(1.0f, 0.0f, 0.0f, 0.0f);
    float  m2 = dot(z, z);

    for (int i = 0; i < 11; i++) {
        if (m2 > 16.0f) break;
        dz = 2.0f * qmul(z, dz);
        z  = qsquare(z) + c;
        m2 = dot(z, z);
    }

    float r = sqrt(m2);
    return 0.5f * r * log(max(r, 1e-6f)) / max(length(dz), 1e-6f);
}

static float3 calcNormal(float3 p, float4 c) {
    float2 e = float2(0.0012f, 0.0f);
    return normalize(float3(
        juliaDE(p + e.xyy, c) - juliaDE(p - e.xyy, c),
        juliaDE(p + e.yxy, c) - juliaDE(p - e.yxy, c),
        juliaDE(p + e.yyx, c) - juliaDE(p - e.yyx, c)
    ));
}

static float3 palette(float t) {
    float3 a = float3(0.48f, 0.42f, 0.5f);
    float3 b = float3(0.5f, 0.5f, 0.5f);
    float3 c = float3(0.9f, 0.75f, 1.0f);
    float3 d = float3(0.1f, 0.3f, 0.45f);
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
    float3 bgColor = float3(0.01f, 0.0f, 0.03f);

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
    float4 c = float4(-0.2f, 0.55f, 0.2f, 0.15f)
             + float4(0.12f * sin(t * 0.10f), 0.10f * cos(t * 0.13f),
                      0.09f * sin(t * 0.08f), 0.08f * cos(t * 0.11f));

    float march = 0.0f;
    float maxD  = 25.0f;

    for (int i = 0; i < 90; i++) {
        float3 p = ro + rd * march;
        float d = juliaDE(p, c);
        if (d < 0.0018f) {
            float3 n = calcNormal(p, c);
            float3 light = normalize(float3(0.4f, 0.9f, 0.35f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = pow(1.0f - max(dot(-rd, n), 0.0f), 3.5f) * 0.6f;

            float colT = length(p) * 0.6f + t * 0.05f + n.z * 0.2f;
            float3 col = palette(colT) * (dif * 1.1f + amb * 0.45f) + float3(0.6f, 0.4f, 1.0f) * rim;
            col *= exp(-march * 0.28f);
            return float4(col, 1.0f);
        }
        march += max(d * 0.85f, 0.0015f);
        if (march > maxD) break;
    }
    return float4(bgColor, 1.0f);
}
