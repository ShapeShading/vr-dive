// petal-particles.metal – Outward-Radiating Petal Particle Bloom
//
// A flower-like burst of glowing particles: each particle is born at the
// center, eases outward along a curved path that swings sideways and back
// (tracing a petal-shaped arc), then fades near the tip and loops again —
// giving a continuous stream of "radiating particles" rather than a solid
// surface. A slow global envelope makes the whole bloom gently open and
// partially close forever, so the flower itself feels like it keeps slowly
// unfolding instead of sitting at a fixed size.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 把 3D 点转成 (r, theta, y)，theta 对 6 瓣做角度折叠（domain-fold）
//       得到瓣内局部角 la；在这一个瓣内用固定 7 次的小循环生成 7 条独立
//       粒子轨迹（复用 hyper4d 的 sdSpherePack4D / apollonian 的 domain-
//       repeat 思路：靠折叠免去对花瓣数量的循环，只需循环「每瓣粒子数」，
//       O(1) 复杂度，不随花瓣数增加而变慢）。每个粒子的年龄
//       ageFrac=fract(time*speed+phase) 循环推进，半径按 ageFrac^0.55
//       缓动增长（先快后慢，模拟“缓慢展开/收尾”），角度按
//       lobeAmp*sin(ageFrac*π) 先摆向一侧再摆回瓣中心线，从而让大量不同
//       年龄的粒子共同勾勒出一条花瓣弧线；粒子在局部坐标下用
//       length(tangential,y-particleY,radial)-size 的球形距离场表示。
// 关键参数:
//   - petalCount=6：花瓣数量，只影响角度折叠周期，不影响性能。
//   - kParticles=7（每瓣粒子数）：决定花瓣弧线的连续感，7 条足够连续又
//     不至于过密。
//   - bloomT = 0.5+0.5*sin(time*0.10)、maxRadius = mix(0.22,0.60,bloomT)：
//     整朵花的「慢呼吸」包络，周期约 63s，比单个粒子的生命周期慢一个
//     数量级，制造「持续放射粒子」与「花整体缓慢开合」两种时间尺度叠加
//     的效果。
//   - size = mix(0.12, 0.028, grown)：粒子从中心诞生时更大更亮，随着
//     ageFrac 增长逐渐收缩，视觉上等价于「远离中心逐渐消散」。
// 性能特征: 每次 DE 求值 1 次中心核心球 + 7 次粒子距离（每次几次三角
//           函数 + hash1），复杂度与 quaternion-julia（11 次迭代）、
//           apollonian（10 次迭代）同量级；march 90 步/maxD=25。
// 已知限制/优化方向:
//   - 粒子用不透明「最近命中」渲染（与本目录其它 shader 一致），没有真正
//     的半透明叠加发光；如需更浓密的粒子云观感，可在命中点周围额外做一
//     次体积雾状的辉光累积（会显著增加开销，建议先看 perf log 再决定）。
//   - 若后续性能采样显示这个 shader 偏慢，可优先把 kParticles 从 7 降到
//     5，或将 march 步数从 90 降到 75。

// ─── Small hash ────────────────────────────────────────────────────────────────
static float hash1(float n) {
    return fract(sin(n) * 43758.5453123f);
}

// ─── Petal particle bloom DE ───────────────────────────────────────────────────
static float petalParticlesDE(float3 p, float time) {
    const float petalCount  = 6.0f;
    const float sectorAngle = 6.2831853f / petalCount;
    const float halfSector  = sectorAngle * 0.5f;

    float theta = atan2(p.z, p.x) + time * 0.03f;
    float r = length(p.xz);
    float y = p.y;

    float la = fmod(theta + halfSector, sectorAngle) - halfSector;

    // Slow global bloom envelope: the whole flower gently opens and
    // partially closes forever instead of sitting at a fixed size.
    float bloomT = 0.5f + 0.5f * sin(time * 0.10f);
    float maxRadius = mix(0.22f, 0.60f, bloomT);

    float best = length(p) - 0.045f; // small glowing core at the flower's heart

    for (int i = 0; i < 7; i++) {
        float seed = float(i) * 1.37f + 5.0f;
        float speed = 0.06f + 0.02f * hash1(seed);
        float phase = hash1(seed + 4.1f);
        float ageFrac = fract(time * speed + phase);

        // Ease-out growth: quick start, slows near the petal tip.
        float grown = pow(ageFrac, 0.55f);
        float particleR = grown * maxRadius;

        // Petal lobe sway: swings toward one side then eases back to the
        // sector's centerline, tracing a petal-shaped outward arc.
        float lobeAmp = halfSector * 0.55f;
        float side = hash1(seed + 8.0f) > 0.5f ? 1.0f : -1.0f;
        float sway = lobeAmp * sin(ageFrac * 3.14159265f) * side;
        float particleY = 0.14f * sin(ageFrac * 3.14159265f + hash1(seed + 2.0f) * 6.2831853f);

        float size = mix(0.12f, 0.028f, grown);

        float tangential = r * (la - sway);
        float radial = r - particleR;
        float3 delta = float3(tangential, y - particleY, radial);
        float d = length(delta) - size;
        best = min(best, d);
    }

    return best;
}

static float3 calcNormal(float3 p, float time) {
    float2 e = float2(0.0016f, 0.0f);
    return normalize(float3(
        petalParticlesDE(p + e.xyy, time) - petalParticlesDE(p - e.xyy, time),
        petalParticlesDE(p + e.yxy, time) - petalParticlesDE(p - e.yxy, time),
        petalParticlesDE(p + e.yyx, time) - petalParticlesDE(p - e.yyx, time)
    ));
}

static float3 palette(float t) {
    float3 a = float3(0.55f, 0.35f, 0.45f);
    float3 b = float3(0.45f, 0.45f, 0.4f);
    float3 c = float3(1.0f, 0.9f, 0.6f);
    float3 d = float3(0.05f, 0.2f, 0.4f);
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
    float3 bgColor = float3(0.008f, 0.008f, 0.02f);

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
        float3 p = ro + rd * march;
        float d = petalParticlesDE(p, t);
        if (d < 0.003f) {
            float3 n = calcNormal(p, t);
            float rim = pow(1.0f - max(dot(-rd, n), 0.0f), 2.0f);

            float hue = atan2(p.z, p.x) * 0.6f / 6.2831853f + length(p) * 0.6f + t * 0.02f;
            float3 core = palette(hue);
            float glow = 0.55f + rim * 0.9f;
            float3 col = core * glow;
            col *= exp(-march * 0.22f);
            return float4(col, 1.0f);
        }
        march += max(d * 0.7f, 0.0025f);
        if (march > maxD) break;
    }
    return float4(bgColor, 1.0f);
}
