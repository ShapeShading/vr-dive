// hyper-menger.metal – 4D Menger Sponge (Tesseract Cross-Fold Fractal)
//
// Generalizes the classic 3D Menger sponge fold into 4D: a hypercube has its
// axis-aligned "cross" tunnels removed on every iteration across all four
// coordinates, then the whole 4D lattice is rotated through four planes and
// perspective-projected into the 3D ray-marched box, so the sponge's visible
// silhouette continuously morphs as W-axis structure rotates into view.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 复用 hyper4d.metal 的「4D 旋转 + 透视投影」框架，但把内部的 4D
//       distance field 换成 menger.metal 那种「排序 + 十字镂空 + 缩放」
//       折叠，并把排序/镂空扩展到 4 个分量（六次两两比较排序），让镂空
//       图样贯穿 W 轴，从而呈现比 3D Menger 更复杂的镂空结构。
// 关键参数:
//   - 挖空阈值 1/3、缩放 3 倍：与经典 Menger 保持一致的自相似比例。
//   - 迭代 5 级（比 3D 版本的 6 级少一级）：4D 排序 + 镂空每级开销更高，
//     故收紧迭代数以维持可接受帧率。
//   - 四个旋转角速度 (0.15/0.19/0.09/0.12) 各自独立，产生非同步的 4D
//     旋转观感。
// 性能特征: 每次 DE 5 次迭代（每次 6 次比较交换 + 2 次 max 合并），法线
//           4D 梯度需要额外 3 次 map 求值；march 90 步/maxD=25。
// 已知限制/优化方向:
//   - w 分量目前和 hyper4d 一样由透视投影导出（伪 4D），如需真正独立的
//     W 自由度，可暴露一个随时间变化的 w 偏移量。

// ─── 4D rotation helpers ──────────────────────────────────────────────────────
static float4 hmRotXY(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x*c - p.y*s, p.x*s + p.y*c, p.z, p.w);
}
static float4 hmRotXW(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x*c - p.w*s, p.y, p.z, p.x*s + p.w*c);
}
static float4 hmRotYZ(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x, p.y*c - p.z*s, p.y*s + p.z*c, p.w);
}
static float4 hmRotZW(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x, p.y, p.z*c - p.w*s, p.z*s + p.w*c);
}

// ─── 4D Menger DE ──────────────────────────────────────────────────────────────
static float menger4DE(float4 p) {
    float4 q = p;
    float  d = 1.0f;

    for (int i = 0; i < 5; i++) {
        q = abs(q);
        if (q.x < q.y) { float tmp = q.x; q.x = q.y; q.y = tmp; }
        if (q.x < q.z) { float tmp = q.x; q.x = q.z; q.z = tmp; }
        if (q.x < q.w) { float tmp = q.x; q.x = q.w; q.w = tmp; }
        if (q.y < q.z) { float tmp = q.y; q.y = q.z; q.z = tmp; }
        if (q.y < q.w) { float tmp = q.y; q.y = q.w; q.w = tmp; }
        if (q.z < q.w) { float tmp = q.z; q.z = q.w; q.w = tmp; }

        float d1 = max(q.y - 1.0f / 3.0f, q.z - 1.0f / 3.0f);
        float d2 = max(q.z - 1.0f / 3.0f, q.w - 1.0f / 3.0f);
        float cross = min(d1, d2);
        d = max(d, -cross);

        q = q * 3.0f - 2.0f;
        d *= 3.0f;
    }

    return (length(q) - 2.0f) / d;
}

static float3 project4D(float4 p, float fov) {
    float wScale = 1.0f / (1.0f + p.w * fov);
    return p.xyz * wScale;
}

static float3 calcNormal(float3 p3, float4 p4) {
    float2 e = float2(0.002f, 0.0f);
    float4 dx = float4(e.x, 0, 0, 0);
    float4 dy = float4(0, e.x, 0, 0);
    float4 dz = float4(0, 0, e.x, 0);
    return normalize(float3(
        menger4DE(p4 + dx) - menger4DE(p4 - dx),
        menger4DE(p4 + dy) - menger4DE(p4 - dy),
        menger4DE(p4 + dz) - menger4DE(p4 - dz)
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
    float3 bgColor = float3(0.0f, 0.02f, 0.02f);

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
    float maxD  = 25.0f;

    for (int i = 0; i < 90; i++) {
        float3 p3 = ro + rd * march;

        float4 p4 = float4(p3, 0.0f);
        p4 = hmRotXY(p4, t * 0.15f);
        p4 = hmRotZW(p4, t * 0.19f);
        p4 = hmRotXW(p4, t * 0.09f);
        p4 = hmRotYZ(p4, t * 0.12f);

        float d = menger4DE(p4);
        float projScale = 1.0f / (1.0f + p4.w * 0.4f);
        d *= projScale;

        if (d < 0.004f) {
            float3 n = calcNormal(p3, p4);
            float3 light = normalize(float3(0.4f, 0.85f, 0.3f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.25f + 0.7f * abs(n.y);
            float rim = pow(1.0f - max(dot(-rd, n), 0.0f), 3.0f) * 0.55f;

            float hue = fract(length(p4) * 0.22f + t * 0.03f + p4.w * 0.12f);
            float4 K = float4(1.0f, 2.0f/3.0f, 1.0f/3.0f, 3.0f);
            float3 pp = abs(fract(hue + K.xyz) * 6.0f - K.www);
            float3 col = clamp(pp - K.xxx, 0.0f, 1.0f);
            col = col * (dif * 1.0f + amb * 0.5f) + float3(0.2f, 0.8f, 0.9f) * rim;
            col *= exp(-march * 0.28f);
            return float4(col, 1.0f);
        }
        march += d;
        if (march > maxD) break;
    }

    return float4(bgColor, 1.0f);
}
