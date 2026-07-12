// liquids.metal – Animated Fluid / Liquid Surface
//
// Renders an undulating liquid/organic surface using domain-warped
// layered noise. The surface moves like a living fluid with
// iridescent reflections. Inspired by fluid simulation visuals.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 用 3 阶 fbm (基于 hash-based value noise) 对采样点做多重 domain
//       warp (先按 sin/cos 位移 q，再叠加两层不同频率/相位的 fbm)，最终
//       以 `field - 0.45` 作为隐式等值面，模拟流体扰动表面。
// 关键参数:
//   - 0.3f/0.2f 系数的正弦位移：一次形变的幅度，决定表面起伏剧烈程度。
//   - fbm 混合比例 0.7/0.3：两层噪声的权重，影响细节层次的丰富度。
//   - isosurface 阈值 0.45：值越大，「液面」越薄/越少被 march 命中。
// 性能特征: 每个 fbm 调用 3 octave noise（原 4，因 perf 抽样显示单帧最高
//           110ms 而收紧），每次 map 求值调用 2 次 fbm，法线额外 6 次；map
//           非严格 SDF，用 max(fabs(d),0.02) 兜底步进，60 步（原 80）/
//           maxD=25，属于中等开销。
// 已知限制/优化方向:
//   - 由于不是真实 SDF，命中判定用 `d < 0.008 && d > -0.05` 双边阈值，
//     表面可能出现锯齿；如需更光滑可提高步进采样密度或改用解析梯度。

// ─── Noise helpers ────────────────────────────────────────────────────────────
static float hash(float3 p) {
    p = fract(p * 0.3183099f + 0.1f);
    p *= 17.0f;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

static float noise(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    float a = hash(i);
    float b = hash(i + float3(1,0,0));
    float c = hash(i + float3(0,1,0));
    float d = hash(i + float3(1,1,0));
    float e = hash(i + float3(0,0,1));
    float f_ = hash(i + float3(1,0,1));
    float g = hash(i + float3(0,1,1));
    float h = hash(i + float3(1,1,1));
    return mix(mix(mix(a,b,f.x), mix(c,d,f.x), f.y),
               mix(mix(e,f_,f.x), mix(g,h,f.x), f.y), f.z);
}

static float fbm(float3 p) {
    float v = 0.0f, a = 0.5f;
    p = p * 2.0f;
    for (int i = 0; i < 3; i++) {
        v += a * noise(p);
        p = p * 2.0f + float3(50, 100, 150);
        a *= 0.5f;
    }
    return v;
}

// ─── Domain-warped fluid SDF ──────────────────────────────────────────────────
static float fluidSDF(float3 p, float t) {
    // Primary deformation
    float3 q = p;
    q.x += 0.3f * sin(t * 0.3f + p.y * 1.5f + p.z * 1.2f);
    q.z += 0.3f * cos(t * 0.4f + p.x * 1.1f + p.y * 1.3f);
    q.y += 0.2f * sin(t * 0.5f + p.x * 0.9f + p.z * 1.4f);

    // Secondary domain warp
    float w1 = fbm(q + t * 0.1f);
    float w2 = fbm(q * 1.5f - t * 0.08f + float3(30, 30, 30));

    // The fluid surface is an isosurface of the warped field
    float field = w1 * 0.7f + w2 * 0.3f;
    return field - 0.45f;
}

static float3 calcNormal(float3 p, float t) {
    float2 e = float2(0.005f, 0.0f);
    return normalize(float3(
        fluidSDF(p + e.xyy, t) - fluidSDF(p - e.xyy, t),
        fluidSDF(p + e.yxy, t) - fluidSDF(p - e.yxy, t),
        fluidSDF(p + e.yyx, t) - fluidSDF(p - e.yyx, t)
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
    float3 bgColor = float3(0.0f, 0.0f, 0.015f);

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

    for (int i = 0; i < 60; i++) {
        float3 p = ro + rd * march;
        float d = fluidSDF(p, t);
        if (d < 0.008f && d > -0.05f) {
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.6f, 0.5f, 0.8f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.2f + 0.8f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 3.0f) * 0.8f;
            float spec = pow(max(dot(reflect(-light, n), -rd), 0.0f), 32.0f) * 2.0f;

            // Deep ocean colors shifting with surface motion
            float3 col = mix(float3(0.0f, 0.1f, 0.3f), float3(0.1f, 0.5f, 0.7f),
                            smoothstep(-0.5f, 0.5f, p.y));
            col += float3(0.0f, 0.3f, 0.5f) * amb;
            col += float3(0.8f, 0.9f, 1.0f) * spec;
            col += float3(0.2f, 0.5f, 0.8f) * rim;
            col *= exp(-march * 0.15f);
            return float4(col, 1.0f);
        }
        march += max(fabs(d), 0.02f);
        if (march > maxD) break;
    }
    return float4(bgColor, 1.0f);
}
