// fibers-wave.metal
//
// Several semi-transparent wavy sheets stacked in Y. Each sheet is almost
// invisible on its own; what you actually see are very thin, sharply
// defined, naturally curved fiber lines that live ON the sheet surface,
// like a fine mesh of threads draped over a rippling membrane.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 先定义若干层「水平波浪曲面」（sheetHeight 高度场），每层曲面本身
//       只贡献极低的 alpha（近乎透明）；再在曲面的局部 2D 坐标 (x,z) 上
//       定义一个「扭曲后的周期场」sheetFlow，取其小数部分距离最近整数的
//       三角波，阈值化成很窄的细线掩码 fiberLineMask——这就是曲面上
//       「很细、显示清晰」的丝线。掩码额外算了一个更窄的 core 高光带，
//       让每根线中心更亮、边缘更暗，模拟圆形丝线的高光而不是一块扁平色
//       斑。sheetFlow 在计算周期坐标前先用 sin() 对坐标做了扭曲，因此
//       细线本身沿曲面走向也是自然弯曲的曲线。体积 march 采用「基于距离
//       场的自适应步进」：远离所有曲面时用最近曲面距离直接跳步前进，
//       只有贴近曲面薄壳时才精细采样，命中时按 alpha 做前向合成。
// 关键参数:
//   - kSheetCount = 5、层间距 0.5：曲面层数与纵深范围。
//   - fiberLineMask 的 halfWidth = 0.05：线宽（相比早期版本的 0.125
//     明显更窄，对应"非常细的线"）；aa 按 halfWidth 比例缩放，线宽变化
//     时抗锯齿过渡区也同步变窄，避免线条模糊发虚。
//   - coreHalfWidth = halfWidth*0.35：线中心高光带宽度，制造圆润的丝线
//     高光而非扁平色块。
//   - baseAlpha=0.025（曲面本身）/ fiberAlpha=0.85（细线，更不透明，
//     配合更窄的线宽依旧要保持"显示清晰"）。
// 性能特征: 自适应距离步进（远离曲面时用 `hit.absDist` 直接跳步），
//           上限 130 次迭代 / maxD=25；相比早期固定步长(≤220 步)的暴力
//           体积 march，典型场景下迭代次数大幅降低。
// 已知限制/优化方向（⚠️ 曾踩坑记录）:
//   - 早期版本用「固定小步长(0.026)+最多220步」的暴力体积 march，
//     在 5 层曲面场景下实测单帧最高耗时超过 200ms；改为「远处用最近
//     曲面距离自适应跳步、只在贴近薄壳时精细采样」后大幅降低了平均/
//     最坏情况耗时。
//   - 曲面用「点到高度场的竖直距离」近似代替真实最近距离，曲面坡度较大
//     处会有轻微误差；本 demo 幅度刻意控制得较小以规避明显瑕疵。

// ─── Thin curved fiber-line mask ──────────────────────────────────────────
// `body` is 1 inside a thin band around integer values of `flow` (line
// width controlled by `halfWidth`, antialiasing scaled to it so thinner
// lines stay crisp instead of blurring out). `core` is a much narrower
// highlight band down the middle of the line, used to fake a rounded
// thread cross-section instead of a flat painted stripe.
static float fiberLineMask(float flow, float halfWidth, thread float &core) {
    float tri = abs(fract(flow + 0.5f) - 0.5f);
    float aa = halfWidth * 0.4f;
    float body = 1.0f - smoothstep(halfWidth - aa, halfWidth + aa, tri);
    core = 1.0f - smoothstep(0.0f, halfWidth * 0.4f, tri);
    return body;
}

// ─── Sheet height field ────────────────────────────────────────────────────
static float sheetHeight(float2 uv, float t, float phase) {
    return 0.16f * sin(uv.x * 1.3f + phase + t * 0.15f) * cos(uv.y * 1.05f - phase * 0.6f)
         + 0.07f * sin(uv.x * 2.4f - uv.y * 1.8f + t * 0.22f + phase * 1.4f);
}

// ─── Fiber flow field on a sheet (warped so iso-lines curve naturally) ────
static float sheetFlow(float2 uv, float t, float phase) {
    float warped = uv.x + 0.35f * sin(uv.y * 1.7f + phase + t * 0.10f);
    return warped * 3.2f + 0.6f * sin(uv.y * 2.3f - phase * 0.8f + t * 0.08f);
}

#define FIBER_WAVE_SHEET_COUNT 5

struct SheetHit {
    float absDist;
    int   k;
};

static SheetHit nearestWaveSheet(float3 p, float t) {
    SheetHit best;
    best.absDist = 1e9f;
    best.k = 0;

    for (int k = 0; k < FIBER_WAVE_SHEET_COUNT; k++) {
        float phase = float(k) * 1.7f;
        float y0 = (float(k) - 2.0f) * 0.5f;
        float h = sheetHeight(p.xz, t, phase);
        float d = abs(p.y - y0 - h);
        if (d < best.absDist) {
            best.absDist = d;
            best.k = k;
        }
    }
    return best;
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
    float maxMarch = 25.0f;
    const float shellHalf = 0.014f;
    const float lineHalfWidth = 0.05f;

    float3 accumC = float3(0.0f);
    float  accumA = 0.0f;
    float  march = 0.0f;

    for (int i = 0; i < 130; i++) {
        if (march > maxMarch) break;
        float3 p = ro + rd * march;

        SheetHit hit = nearestWaveSheet(p, t);
        if (hit.absDist < shellHalf) {
            float phase = float(hit.k) * 1.7f;
            float flow = sheetFlow(p.xz, t, phase);
            float core = 0.0f;
            float lineMask = fiberLineMask(flow, lineHalfWidth, core);

            float falloff = 1.0f - hit.absDist / shellHalf;
            float baseAlpha = 0.025f;
            float fiberAlpha = 0.85f;
            float alpha = clamp(mix(baseAlpha, fiberAlpha, lineMask) * falloff, 0.0f, 1.0f);

            float depthTone = float(hit.k) / float(FIBER_WAVE_SHEET_COUNT - 1);
            float3 sheetColor = mix(float3(0.28f, 0.40f, 0.54f), float3(0.55f, 0.68f, 0.80f), depthTone);
            float3 fiberColor = mix(float3(0.72f, 0.82f, 0.94f), float3(0.90f, 0.86f, 0.70f), depthTone);
            float3 coreColor = float3(0.98f, 0.99f, 1.0f);
            float3 threadColor = mix(fiberColor, coreColor, core);
            float3 col = mix(sheetColor, threadColor, lineMask);

            accumC += col * alpha * (1.0f - accumA);
            accumA += alpha * (1.0f - accumA);
            if (accumA > 0.97f) break;

            march += shellHalf * 2.2f + 0.01f;
        } else {
            march += clamp(hit.absDist * 0.85f, 0.02f, 0.4f);
        }
    }

    float sky = 0.5f + 0.5f * rd.y;
    float3 bg = mix(float3(0.012f, 0.015f, 0.022f), float3(0.02f, 0.028f, 0.04f), sky);
    float3 finalColor = accumC + bg * (1.0f - accumA);
    return float4(finalColor, 1.0f);
}
