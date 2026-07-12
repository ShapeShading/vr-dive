// fractree.metal – Procedural Fractal Tree
//
// Renders a recursive branching tree using ray marching with
// iterative branch approximation. Each branch splits into two
// at each level, creating a natural fractal structure.
// The tree sways gently in an invisible wind.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 用「折叠空间 + 圆柱体 SDF」近似分形树。每一级迭代先算主干圆柱
//       (trunk) 和沿 foldDir 方向的分支圆柱 (branch)，取 min 后，再对
//       空间做取绝对值 + 轴交换的折叠，模拟真实分叉产生的自相似结构。
// 关键参数:
//   - foldDir 由 sway = sin(t*0.5 + i*1.5) 计算，让每一级分支随时间摆动，
//     模拟风吹效果；i 越大摆动相位差越大，看起来更自然。
//   - rep = 0.35：每级迭代前沿 y 轴平移的距离，决定分支的分布密度。
//   - 迭代 7 级，每级 q *= 1.5 放大，配合 r *= 1.5 保持是有效距离场。
// 性能特征: 每次 DE 求值仅 7 次循环（三角函数调用可控），march 步进用
//           max(d*0.8, 0.01) 做保守推进，100 步 / maxD=25。
// 已知限制/优化方向:
//   - 树冠和树干目前共享同一套折叠逻辑，层次感可通过在高层级减小
//     branch 半径来增强「越往上越细」的真实感。
//   - 可尝试给顶端加入叶片/花朵状 SDF 提升识别度。

// ─── Tree SDF ─────────────────────────────────────────────────────────────────
// Approximates a branching tree by folding space toward branches.
static float treeSDF(float3 p, float t) {
    float3 q = p;
    float  r = 0.0f;

    // Trunk and branching
    for (int i = 0; i < 7; i++) {
        // Vertical trunk
        float trunk = length(q.xz) - 0.04f;
        float tip   = q.y - 0.6f;
        float seg   = max(trunk, tip);

        // Branch split: fold space into two symmetric branches
        float sway = 0.3f * sin(t * 0.5f + float(i) * 1.5f);
        float3 foldDir = normalize(float3(cos(sway + float(i) * 2.4f), 0.6f, sin(sway + float(i) * 1.8f)));

        // Branch cylinder along foldDir
        float3 proj = q - foldDir * dot(q, foldDir);
        float branch = length(proj) - 0.025f;
        float bTip = dot(q, foldDir) - 0.5f;
        branch = max(branch, bTip);

        seg = min(seg, branch);

        if (i == 0) r = seg;

        // Fold space for next level: reflect and scale toward branches
        float rep = 0.35f;
        q.y -= rep;
        q = abs(q);
        // Swap axes to create branching
        float tmp = q.y; q.y = q.z; q.z = tmp;

        // Scale down for finer detail
        q = q * 1.5f;
        r *= 1.5f;
    }

    return r / 3.0f - 0.01f;
}

static float3 calcNormal(float3 p, float t) {
    float2 e = float2(0.003f, 0.0f);
    return normalize(float3(
        treeSDF(p + e.xyy, t) - treeSDF(p - e.xyy, t),
        treeSDF(p + e.yxy, t) - treeSDF(p - e.yxy, t),
        treeSDF(p + e.yyx, t) - treeSDF(p - e.yyx, t)
    ));
}

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
    float3 bgColor = float3(0.01f, 0.005f, 0.02f);

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
    float maxD = 25.0f;

    for (int i = 0; i < 100; i++) {
        float3 p = ro + rd * march;
        float d = treeSDF(p, t);
        if (d < 0.005f) {
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.3f, 0.8f, 0.2f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 4.0f) * 0.4f;

            // Warm bark-like color with green tips
            float height = p.y + 0.5f;
            float3 col = mix(float3(0.4f, 0.25f, 0.1f), float3(0.2f, 0.6f, 0.15f), smoothstep(0.0f, 1.0f, height));
            col = col * (dif * 1.0f + amb * 0.5f) + float3(0.1f, 0.3f, 0.1f) * rim;
            col *= exp(-march * 0.2f);
            return float4(col, 1.0f);
        }
        march += max(d * 0.8f, 0.01f);
        if (march > maxD) break;
    }
    return float4(bgColor, 1.0f);
}
