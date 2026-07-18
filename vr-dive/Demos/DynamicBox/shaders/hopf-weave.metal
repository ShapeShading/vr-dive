// hopf-weave.metal - S3 Hopf Fibration Weave
//
// A ray-march-friendly artistic projection of the Hopf fibration. It does not
// enumerate individual 3D tubes: that would overlap knots.metal's expensive
// closest-curve search and become costly under ray-marched normal evaluation.
// Instead, nine analytic, differently oriented torus latitude layers stand in
// for S3 latitude tori. Their surface coordinates carry counter-winding
// Villarceau-style light bands, evoking the linked Hopf circles revealed by
// stereographic projection of S3.
// This is deliberately distinct from fibers-vortex.metal: the geometry is a
// compact set of nested TORI, not vertical cylindrical shells, and the bands
// run in two alternating linked families rather than one axial helix.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: Hopf 纤维化把 4D 中的 3-球面 S3 按圆 S1 分解；S2 上固定纬度的
//       原像在 S3 中是一个环面，立体投影后成为 R3 中嵌套的环面，纤维则是
//       环面上彼此链接的 Villarceau 圆。直接对几十根圆管取 min 会在每个
//       march 点和法线 6 次重采样中放大为高开销，故此 shader 只 ray march
//       9 个解析 torus SDF；每层先做不同的旋转、偏移和轻微椭圆拉伸，
//       避免旧版三个同轴近半径环面挤成一圈；命中后在环面局部双角 (u,v) 以
//       5u±3v+phase 的两组反向绕行条纹着色，视觉上给出两族链接纤维。
//       这样保持 Hopf 投影的核心「嵌套环面 + 链接圆」特征，同时避免逐管
//       距离场遍历，也不复刻已有 vortex 圆柱螺旋、torus knot 或 hyper4d
//       超环面+球格点的机制。
// 关键参数:
//   - 9 个环面投影层：新增一组三层更大尺度环面；层间主半径、Y/Z 偏移、
//     X/Y 倾角与椭圆比的差异被拉大，轮廓会分成多组交叠的花形环面。
//   - weave orders (5,3)：两组 5u+3v 与 5u-3v 的窄亮线交错，模拟
//     Hopf circle 在每层环面上的不同绕行方向，而不是单方向螺旋。
//   - phase 只以 0.07 rad/s 缓慢推进，整体旋转以 0.05 rad/s 进行，让
//     编织关系可读而不会像 kaleidoscope 或 vortex 一样高速翻动。
// 性能特征: map() 固定评估 9 个解析 torus SDF；无参数曲线最近点搜索、
//           无分形迭代、无体积累积；march 72 步/maxD=25，预计明显轻于
//           knots（约 16 次曲线采样/DE）和 quaternion-julia（11 次迭代）。
// 已知限制/优化方向:
//   - 条纹目前是表面发光图样而不是几何上独立的 Hopf 圆管；如需更强的
//     立体丝线触感，可只对最近环面加入很小的 thickness 调制，但不应把
//     每条纤维改为独立 tube SDF，否则会破坏当前的性能目标。

// ─── Rotation helpers ────────────────────────────────────────────────────────
static float3 rotateY(float3 p, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static float3 rotateX(float3 p, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

static float3 rotateZ(float3 p, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

// ─── Hopf latitude tori ──────────────────────────────────────────────────────
struct HopfHit {
    float distance;
    int layer;
    float u;
    float v;
};

static HopfHit hopfTorusField(float3 p) {
    const float majorRadii[9] = { 0.20f, 0.35f, 0.49f, 0.64f, 0.78f, 0.91f, 1.02f, 1.16f, 1.29f };
    const float minorRadii[9] = { 0.082f, 0.076f, 0.070f, 0.064f, 0.058f, 0.054f, 0.049f, 0.045f, 0.041f };
    const float yOffsets[9] = { -0.24f, 0.15f, -0.07f, 0.22f, -0.16f, 0.05f, 0.29f, -0.27f, 0.11f };
    const float zOffsets[9] = { 0.14f, -0.20f, 0.23f, -0.09f, 0.13f, -0.19f, 0.27f, 0.08f, -0.25f };
    const float xTilts[9] = { 0.16f, -0.42f, 0.58f, -0.67f, 0.38f, -0.24f, 0.71f, -0.55f, 0.47f };
    const float zTilts[9] = { -0.22f, 0.32f, -0.19f, 0.48f, -0.56f, 0.35f, -0.41f, 0.62f, -0.33f };
    const float radialStretch[9] = { 0.82f, 1.18f, 0.76f, 1.25f, 0.85f, 1.11f, 0.72f, 1.29f, 0.90f };

    HopfHit hit;
    hit.distance = 1e6f;
    hit.layer = 0;
    hit.u = 0.0f;
    hit.v = 0.0f;

    for (int layer = 0; layer < 9; layer++) {
        float3 q = p - float3(0.0f, yOffsets[layer], zOffsets[layer]);
        q = rotateZ(rotateX(q, xTilts[layer]), zTilts[layer]);
        q.x *= radialStretch[layer];
        float radial = length(q.xz);
        float u = atan2(q.z, q.x);
        float qx = radial - majorRadii[layer];
        float d = length(float2(qx, q.y)) - minorRadii[layer];
        if (d < hit.distance) {
            hit.distance = d;
            hit.layer = layer;
            hit.u = u;
            hit.v = atan2(q.y, qx);
        }
    }
    return hit;
}

static float map(float3 p) {
    return hopfTorusField(p).distance;
}

static float3 calcNormal(float3 p) {
    float2 e = float2(0.0015f, 0.0f);
    return normalize(float3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

static float3 palette(float t) {
    float3 a = float3(0.18f, 0.26f, 0.34f);
    float3 b = float3(0.50f, 0.45f, 0.38f);
    float3 c = float3(0.75f, 0.95f, 1.0f);
    float3 d = float3(0.12f, 0.32f, 0.55f);
    return a + b * cos(6.28318f * (c * t + d));
}

// ─── Fragment shader ─────────────────────────────────────────────────────────
fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut         in         [[stage_in]],
    constant DynamicBoxUniforms &uniforms [[buffer(0)]],
    constant float4x4          *v2wMats   [[buffer(1)]],
    constant float4x4          *vpMatrices [[buffer(2)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center = uniforms.objectCenter.xyz;
    float sc = uniforms.boxScale;
    float3 boxEye = (camWorld - center) / sc;
    float3 boxRd = normalize(in.worldPos - camWorld);
    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    float3 origin;
    float3 bgColor = float3(0.004f, 0.008f, 0.018f);

    if (!insideBox) {
        float3 entryNormal;
        float entry = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (entry < 0.0f) return float4(bgColor, 1.0f);
        origin = boxEye + boxRd * (entry + 1e-3f);
    } else {
        origin = boxEye;
    }

    float3 exitNormal;
    if (db_boxHit(origin, boxRd, DB_BOXDIMS, exitNormal, false) <= 0.0f) discard_fragment();

    float3 ro = (uniforms.patternTransform * float4(origin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRd, 0.0f)));
    float time = uniforms.time;

    // Gentle motion makes the S3 projection legible without turning it into a
    // conventional spinning torus demo.
    ro = rotateX(rotateY(ro, time * 0.05f), -0.13f);
    rd = rotateX(rotateY(rd, time * 0.05f), -0.13f);

    float march = 0.0f;
    for (int i = 0; i < 72; i++) {
        float3 p = ro + rd * march;
        HopfHit hit = hopfTorusField(p);
        if (hit.distance < 0.0025f) {
            float3 n = calcNormal(p);
            float phase = time * 0.07f + float(hit.layer) * 0.83f;

            // The two signs represent the two visibly interlaced Hopf-circle
            // families on every latitude torus.
            float bandA = abs(sin(hit.u * 5.0f + hit.v * 3.0f + phase));
            float bandB = abs(sin(hit.u * 5.0f - hit.v * 3.0f - phase * 0.8f));
            float lineA = 1.0f - smoothstep(0.16f, 0.29f, bandA);
            float lineB = 1.0f - smoothstep(0.16f, 0.29f, bandB);
            float fiber = max(lineA, lineB) * 0.34f;
            float crossing = lineA * lineB;

            float3 light = normalize(float3(-0.35f, 0.8f, 0.5f));
            float dif = max(dot(n, light), 0.0f);
            float rim = pow(1.0f - max(dot(-rd, n), 0.0f), 2.5f);
            float colorT = float(hit.layer) * 0.22f + hit.v * 0.11f + phase * 0.14f;

            float3 base = palette(colorT) * (0.18f + dif * 0.62f);
            float3 fiberColor = palette(colorT + 0.08f) * (0.84f + crossing * 0.14f);
            float3 col = mix(base, fiberColor, fiber);
            col += float3(0.30f, 0.72f, 1.0f) * rim * (0.18f + fiber * 0.24f);
            col *= exp(-march * 0.18f);
            return float4(col, 1.0f);
        }
        march += max(hit.distance * 0.82f, 0.003f);
        if (march > 25.0f) break;
    }

    return float4(bgColor, 1.0f);
}
