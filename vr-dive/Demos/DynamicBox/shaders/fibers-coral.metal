// fibers-coral.metal
//
// Several semi-transparent lobed spherical shells (like a stack of nested
// coral fans). Each shell is almost invisible on its own; what you actually
// see are dense, thin, naturally curved fiber lines living ON each shell
// (~25% of its area), running roughly along meridians like veins in a
// coral fan / leaf.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 先定义若干层「带花瓣状波纹的球形曲面」——半径不是常数，而是随
//       方位角 phi 与极角 theta 调制 (numLobes*phi 产生若干个"瓣"，
//       乘以 sin(theta) 让波纹只在赤道附近明显、两极趋于平滑)，形成像
//       珊瑚扇/贝壳一样的分瓣轮廓；曲面本身只贡献极低 alpha。再在曲面的
//       局部坐标 (phi, theta) 上定义一个「沿经线走向、按纬度扭曲」的周期
//       场 coralFlow，取其小数部分距离最近整数的三角波，阈值化成占周期
//       25% 宽度的细线掩码——这就是「很细、比较密集、约占 1/4 面积」的
//       丝线；扭曲项让细线沿曲面走向天然带有波动曲率，而不是笔直经线。
//       体积 march 每步找「到最近球形曲面的距离」，命中薄壳时按 alpha
//       做前向合成，多层叠加呈现半透明珊瑚扇的纵深感。
// 关键参数:
//   - kCoralShellCount = 4、基础半径 0.30 起、层间距 0.16：同心球壳的
//     半径分布。
//   - numLobes = 6、rippleAmp = 0.09：花瓣数量与波纹幅度，决定"珊瑚扇"
//     轮廓的分瓣程度。
//   - coralFlow 里 phi*turns + warpAmp*sin(theta*warpFreq+phase)：细线
//     沿经线方向排布，warp 项让线条随纬度自然弯曲。
//   - fiberLineMask 的 halfWidth = 0.125：对应 25% 占空比。
// 性能特征: 固定步长体积 march（stepSize=0.026），每步检测 4 层球壳的
//           「到最近壳距离」（含 atan2/acos），命中薄壳时再算一次 flow
//           场；≤210 步/maxD=25，中等偏高开销。
// 已知限制/优化方向:
//   - 目前花瓣调制只作用在半径上，若想要更强的"珊瑚"分支感，可以让
//     rippleAmp 或 numLobes 也随壳层 k 变化，形成外层更密的分瓣。

// ─── Thin curved fiber-line mask ──────────────────────────────────────────
static float fiberLineMask(float flow, float halfWidth) {
    float tri = abs(fract(flow + 0.5f) - 0.5f);
    float aa  = 0.02f;
    return 1.0f - smoothstep(halfWidth - aa, halfWidth + aa, tri);
}

// ─── Lobed shell radius field ──────────────────────────────────────────────
static float coralRadius(float phi, float theta, float t, float phase, float baseR) {
    const float numLobes = 6.0f;
    const float rippleAmp = 0.09f;
    float ripple = rippleAmp * sin(numLobes * phi + phase + t * 0.12f) * sin(theta);
    return baseR + ripple;
}

// ─── Meridian fiber flow field on a shell (warped to curve naturally) ────
static float coralFlow(float phi, float theta, float t, float phase) {
    float turns = 5.0f;
    float warp = 0.6f * sin(theta * 2.4f + phase + t * 0.10f);
    return (phi + warp) * turns + phase + t * 0.06f;
}

#define FIBER_CORAL_SHELL_COUNT 4

struct CoralShellHit {
    float absDist;
    int   k;
    float phi;
    float theta;
};

static CoralShellHit nearestCoralShell(float3 p, float t) {
    CoralShellHit best;
    best.absDist = 1e9f;
    best.k = 0;
    best.phi = 0.0f;
    best.theta = 0.0f;

    float radius = length(p);
    float phi = atan2(p.z, p.x);
    float theta = acos(clamp(p.y / max(radius, 1e-4f), -1.0f, 1.0f));

    for (int k = 0; k < FIBER_CORAL_SHELL_COUNT; k++) {
        float phase = float(k) * 1.9f;
        float baseR = 0.30f + 0.16f * float(k);
        float rk = coralRadius(phi, theta, t, phase, baseR);
        float d = abs(radius - rk);
        if (d < best.absDist) {
            best.absDist = d;
            best.k = k;
            best.phi = phi;
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
            return float4(0.02f, 0.015f, 0.012f, 1.0f);
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

    for (int i = 0; i < min(maxSteps, 210); i++) {
        float dist = (float(i) + 0.5f) * stepSize;
        if (dist > maxMarch) break;
        float3 p = ro + rd * dist;

        CoralShellHit hit = nearestCoralShell(p, t);
        if (hit.absDist < shellHalf) {
            float phase = float(hit.k) * 1.9f;
            float flow = coralFlow(hit.phi, hit.theta, t, phase);
            float lineMask = fiberLineMask(flow, 0.125f);

            float falloff = 1.0f - hit.absDist / shellHalf;
            float baseAlpha = 0.028f;
            float fiberAlpha = 0.62f;
            float alpha = clamp(mix(baseAlpha, fiberAlpha, lineMask) * falloff * (stepSize * 30.0f), 0.0f, 1.0f);

            float depthTone = float(hit.k) / float(FIBER_CORAL_SHELL_COUNT - 1);
            float3 sheetColor = mix(float3(0.55f, 0.30f, 0.22f), float3(0.40f, 0.30f, 0.46f), depthTone);
            float3 fiberColor = mix(float3(0.95f, 0.70f, 0.55f), float3(0.85f, 0.70f, 0.95f), depthTone);
            float3 col = mix(sheetColor, fiberColor, lineMask);

            accumC += col * alpha * (1.0f - accumA);
            accumA += alpha * (1.0f - accumA);
            if (accumA > 0.97f) break;
        }
    }

    float horizon = 0.5f + 0.5f * rd.y;
    float3 bg = mix(float3(0.014f, 0.012f, 0.010f), float3(0.026f, 0.020f, 0.016f), horizon);
    float3 finalColor = accumC + bg * (1.0f - accumA);
    return float4(finalColor, 1.0f);
}
