// fibers-wave.metal
//
// Several semi-transparent wavy sheets stacked in Y. Each sheet is almost
// invisible on its own; what you actually see are dense, thin, naturally
// curved fiber lines that live ON the sheet surface (~25% of its area),
// like a woven mesh of threads draped over a rippling membrane.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 先定义若干层「水平波浪曲面」（sheetHeight 高度场），每层曲面本身
//       只贡献极低的 alpha（近乎透明）；再在曲面的局部 2D 坐标 (x,z) 上
//       定义一个「扭曲后的周期场」sheetFlow，取其小数部分距离最近整数的
//       三角波，阈值化成占周期 25% 宽度的细线掩码 fiberLineMask——这就是
//       曲面上「很细、比较密集、约占 1/4 面积」的丝线。sheetFlow 在计算
//       周期坐标前先用 sin() 对坐标做了扭曲，因此细线本身沿曲面走向也是
//       自然弯曲的曲线，而不是笔直条纹。体积 march 沿途在每一步找「最近
//       的曲面」，命中曲面薄壳时按 alpha 做前向合成，多层前后叠加，透过
//       前层能看到后层，呈现有纵深的半透明多层薄纱效果。
// 关键参数:
//   - kSheetCount = 5、层间距 0.5：曲面层数与纵深范围（覆盖盒子 y 方向
//     约 [-1,1]）。
//   - fiberLineMask 的 halfWidth = 0.125：对应 25% 占空比（占面积 1/4）。
//   - sheetFlow 里的 0.35*sin(uv.y*1.7+phase) 扭曲项：决定细线弯曲的
//     幅度/频率；去掉它细线会退化成笔直条纹。
//   - baseAlpha=0.03（曲面本身，近乎透明）/ fiberAlpha=0.65（细线，明显
//     更不透明）：两者之差就是「曲面透明、细线可见」的核心对比。
// 性能特征: 固定步长体积 march（stepSize=0.026），每步检测 5 层曲面的
//           「到最近曲面距离」，命中薄壳时再算一次 flow 场；≤220 步/
//           maxD=25，中等偏高开销，建议关注其 perf 抽样日志。
// 已知限制/优化方向:
//   - 曲面用「点到高度场的竖直距离」近似代替真实最近距离，曲面坡度较大
//     处会有轻微误差；本 demo 幅度刻意控制得较小以规避明显瑕疵。
//   - 目前每层曲面权重相同，如需更强纵深感可让远处层的 alpha 随深度
//     衰减得更快。

// ─── Thin curved fiber-line mask ──────────────────────────────────────────
// 1 inside a thin band around integer values of `flow`; band half-width in
// units of one period, so halfWidth=0.125 gives ~25% coverage per period.
static float fiberLineMask(float flow, float halfWidth) {
    float tri = abs(fract(flow + 0.5f) - 0.5f);
    float aa  = 0.02f;
    return 1.0f - smoothstep(halfWidth - aa, halfWidth + aa, tri);
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

    float stepSize = 0.026f;
    float maxMarch = 25.0f;
    int   maxSteps = int(maxMarch / stepSize) + 2;

    const float shellHalf = 0.016f;

    float3 accumC = float3(0.0f);
    float  accumA = 0.0f;

    for (int i = 0; i < min(maxSteps, 220); i++) {
        float dist = (float(i) + 0.5f) * stepSize;
        if (dist > maxMarch) break;
        float3 p = ro + rd * dist;

        SheetHit hit = nearestWaveSheet(p, t);
        if (hit.absDist < shellHalf) {
            float phase = float(hit.k) * 1.7f;
            float flow = sheetFlow(p.xz, t, phase);
            float lineMask = fiberLineMask(flow, 0.125f);

            float falloff = 1.0f - hit.absDist / shellHalf;
            float baseAlpha = 0.03f;
            float fiberAlpha = 0.65f;
            float alpha = clamp(mix(baseAlpha, fiberAlpha, lineMask) * falloff * (stepSize * 30.0f), 0.0f, 1.0f);

            float depthTone = float(hit.k) / float(FIBER_WAVE_SHEET_COUNT - 1);
            float3 sheetColor = mix(float3(0.28f, 0.40f, 0.54f), float3(0.55f, 0.68f, 0.80f), depthTone);
            float3 fiberColor = mix(float3(0.86f, 0.91f, 0.98f), float3(0.98f, 0.95f, 0.86f), depthTone);
            float3 col = mix(sheetColor, fiberColor, lineMask);

            accumC += col * alpha * (1.0f - accumA);
            accumA += alpha * (1.0f - accumA);
            if (accumA > 0.97f) break;
        }
    }

    float sky = 0.5f + 0.5f * rd.y;
    float3 bg = mix(float3(0.012f, 0.015f, 0.022f), float3(0.02f, 0.028f, 0.04f), sky);
    float3 finalColor = accumC + bg * (1.0f - accumA);
    return float4(finalColor, 1.0f);
}
