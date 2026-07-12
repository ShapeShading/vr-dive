// fibers-vortex.metal
//
// Several semi-transparent cylindrical shells nested around the Y axis,
// each gently undulating (not perfectly circular). What you actually see
// are very thin, sharply defined, spiraling fiber lines living ON each
// shell — a vortex-like woven mesh twisting around the central axis.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 先定义若干层「同心圆柱曲面」（半径 rk 随 y 轻微波动，制造有机的
//       不规则感），曲面本身只贡献极低 alpha；再在曲面的局部坐标
//       (方位角 theta, 高度 y) 上定义一个「随高度扭转」的周期场
//       vortexFlow —— theta 项乘以圈数、y 项乘以扭转速率并叠加一个正弦
//       扭曲，取其小数部分距离最近整数的三角波，阈值化成很窄的细线掩码，
//       这就是「很细、显示清晰」的丝线；额外算了一个更窄的 core 高光带
//       让线中心更亮，模拟圆形丝线而不是扁平色斑。由于 flow 场含有额外
//       正弦扭曲项，细线本身沿曲面走向天然带有波动曲率，而不是笔直螺旋
//       线。体积 march 采用「基于距离场的自适应步进」：远离所有圆柱壳时
//       用最近壳距离直接跳步前进，只有贴近薄壳时才精细采样。
// 关键参数:
//   - kVortexShellCount = 4、基础半径 0.24 起、层间距 0.17：同心圆柱的
//     半径分布，决定漩涡的"层数"与稀疏程度。
//   - vortexFlow 里 theta*turns + y*twistRate：turns 控制每层螺旋圈数，
//     twistRate 控制随高度扭转的速度，两者共同决定螺旋的"扭紧"程度。
//   - fiberLineMask 的 halfWidth = 0.05（相比早期版本的 0.125 明显更
//     窄）；coreHalfWidth = halfWidth*0.4 制造线中心高光。
// 性能特征: 自适应距离步进（远离曲面时用 `hit.absDist` 直接跳步），
//           上限 140 次迭代 / maxD=25；相比早期固定步长(≤230 步)的暴力
//           体积 march（实测单帧最高约 98ms），迭代次数明显降低。
// 已知限制/优化方向:
//   - 目前每层壳的扭转速率相同，只有半径不同；如需更强的"漩涡吸入"感，
//     可以让 twistRate 也随层数 k 递增，形成内层转得更快的差速漩涡。

// ─── Thin curved fiber-line mask ──────────────────────────────────────────
static float fiberLineMask(float flow, float halfWidth, thread float &core) {
    float tri = abs(fract(flow + 0.5f) - 0.5f);
    float aa = halfWidth * 0.4f;
    float body = 1.0f - smoothstep(halfWidth - aa, halfWidth + aa, tri);
    core = 1.0f - smoothstep(0.0f, halfWidth * 0.4f, tri);
    return body;
}

// ─── Spiraling fiber flow field on a cylindrical shell ────────────────────
// theta: azimuth angle around Y; y: height. Warped so the spiral bands
// curve naturally instead of running as straight helices.
static float vortexFlow(float theta, float y, float t, float phase) {
    float turns = 3.0f;
    float twistRate = 1.6f;
    float warp = 0.5f * sin(y * 1.4f + phase + t * 0.15f);
    return (theta + warp) * turns + y * twistRate + phase + t * 0.25f;
}

#define FIBER_VORTEX_SHELL_COUNT 4

struct ShellHit {
    float absDist;
    int   k;
    float theta;
};

static ShellHit nearestVortexShell(float3 p, float t) {
    ShellHit best;
    best.absDist = 1e9f;
    best.k = 0;
    best.theta = 0.0f;

    float theta = atan2(p.z, p.x);
    float radius = length(p.xz);

    for (int k = 0; k < FIBER_VORTEX_SHELL_COUNT; k++) {
        float phase = float(k) * 2.1f;
        float rk = 0.24f + 0.17f * float(k) + 0.035f * sin(p.y * 2.2f + phase + t * 0.2f);
        float d = abs(radius - rk);
        if (d < best.absDist) {
            best.absDist = d;
            best.k = k;
            best.theta = theta;
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
            return float4(0.02f, 0.017f, 0.014f, 1.0f);
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
    const float shellHalf = 0.013f;
    const float lineHalfWidth = 0.05f;

    float3 accumC = float3(0.0f);
    float  accumA = 0.0f;
    float  march = 0.0f;

    for (int i = 0; i < 140; i++) {
        if (march > maxMarch) break;
        float3 p = ro + rd * march;

        ShellHit hit = nearestVortexShell(p, t);
        if (hit.absDist < shellHalf) {
            float phase = float(hit.k) * 2.1f;
            float flow = vortexFlow(hit.theta, p.y, t, phase);
            float core = 0.0f;
            float lineMask = fiberLineMask(flow, lineHalfWidth, core);

            float falloff = 1.0f - hit.absDist / shellHalf;
            float baseAlpha = 0.022f;
            float fiberAlpha = 0.82f;
            float alpha = clamp(mix(baseAlpha, fiberAlpha, lineMask) * falloff, 0.0f, 1.0f);

            float depthTone = float(hit.k) / float(FIBER_VORTEX_SHELL_COUNT - 1);
            float3 sheetColor = mix(float3(0.42f, 0.20f, 0.34f), float3(0.30f, 0.34f, 0.62f), depthTone);
            float3 fiberColor = mix(float3(0.85f, 0.48f, 0.66f), float3(0.60f, 0.68f, 0.92f), depthTone);
            float3 coreColor = float3(1.0f, 0.97f, 0.99f);
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

    float glow = exp(-0.8f * length(ro.xz));
    float3 bg = mix(float3(0.012f, 0.010f, 0.009f), float3(0.030f, 0.020f, 0.015f), glow);
    float3 finalColor = accumC + bg * (1.0f - accumA);
    return float4(finalColor, 1.0f);
}
