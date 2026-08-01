// fractree.metal – Procedural Fractal Tree
//
// Renders a recursive branching tree using ray marching. A vertical trunk
// supports three explicit branching tiers, so every visible branch remains
// attached to the tree instead of becoming a disconnected folded fragment.
// The tree sways gently in an invisible wind.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 在真实树的局部坐标中构造主干、一级粗枝、二级细枝和三级嫩枝。
//       每根枝条用有限长圆柱 SDF，起点落在父枝上，末端逐级收缩并向上，
//       因此得到连续的树冠轮廓，而不是互不相连的空间折叠碎片。
// 关键参数:
//   - trunkRadius = 0.09、branchRadius = 0.045/0.028/0.015：从树干到嫩枝
//     逐层变细；分叉角约 38°，让树冠填充盒子但保留清晰的负空间。
//   - sway = sin(t*0.5 + phase) 只偏移枝条方向，主干保持稳定，模拟风吹。
// 性能特征: 每次 DE 检查 1 根树干 + 14 根有限枝条，march 96 步/maxD=25。
// 已知限制/优化方向: 当前使用固定三层分叉而非递归调用，便于保证实时性能；
//       如需更密树冠，可增加末端叶片，但不应牺牲父枝连接关系。

// ─── Tree SDF ─────────────────────────────────────────────────────────────────
static float capsuleSDF(float3 p, float3 a, float3 b, float radius) {
    float3 ba = b - a;
    float3 pa = p - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-5f), 0.0f, 1.0f);
    return length(pa - ba * h) - radius;
}

static float3 branchDirection(float3 parent, float angle, float sway) {
    float3 radial = normalize(float3(cos(angle), 0.0f, sin(angle)));
    return normalize(radial * (0.62f + 0.06f * sway) + float3(0.0f, 0.78f, 0.0f));
}

static float treeSDF(float3 p, float t) {
    float sway = sin(t * 0.5f) * 0.10f;
    float d = capsuleSDF(p, float3(0.0f, -0.85f, 0.0f), float3(0.0f, 0.78f, 0.0f), 0.09f);

    // First tier: four broad branches leave the upper trunk.
    for (int i = 0; i < 4; i++) {
        float angle = float(i) * 1.5708f + 0.35f;
        float localSway = sin(t * 0.5f + float(i) * 1.7f) * 0.12f;
        float3 a = float3(0.0f, 0.34f, 0.0f);
        float3 dir = branchDirection(float3(0.0f), angle + localSway, sway);
        float3 b = a + dir * 0.62f;
        d = min(d, capsuleSDF(p, a, b, 0.052f));

        // Second tier: two attached limbs on each broad branch.
        for (int side = 0; side < 2; side++) {
            float sideAngle = angle + (side == 0 ? 0.82f : -0.82f);
            float tierSway = sin(t * 0.62f + float(i) * 1.3f + float(side)) * 0.10f;
            float3 childDir = branchDirection(float3(0.0f), sideAngle + tierSway, sway);
            float3 childA = mix(a, b, 0.66f);
            float3 childB = childA + childDir * 0.39f;
            d = min(d, capsuleSDF(p, childA, childB, 0.032f));

            // Fine twigs keep the silhouette irregular without losing the parent.
            float3 twigDir = branchDirection(float3(0.0f), sideAngle + (side == 0 ? 0.42f : -0.42f) + sway, sway);
            float3 twigA = mix(childA, childB, 0.62f);
            float3 twigB = twigA + twigDir * 0.23f;
            d = min(d, capsuleSDF(p, twigA, twigB, 0.017f));
        }
    }

    // A small crown bud gives the top of the tree a readable terminal point.
    d = min(d, length(p - float3(0.0f, 0.82f, 0.0f)) - 0.075f);
    return d;
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
