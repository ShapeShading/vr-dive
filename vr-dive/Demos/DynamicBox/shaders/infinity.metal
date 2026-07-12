// infinity.metal – Infinite Mirror Corridor
//
// Renders an endless recursive mirrored hallway with floating
// colored lights. Each reflection is darker and tinted differently,
// creating a deep infinite tunnel effect.
//
// The corridor has glowing wall panels and floating energy orbs
// that recede into infinite depth.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 用 `round(p.z/segLen)` 沿 Z 轴做无限重复 (domain repetition)，把
//       每个「走廊段」折回原点附近计算；每段由「外墙盒子 - 内部镂空盒子」
//       构成走廊边框，再叠加网格线、发光面板、中心能量球三种细节 SDF，
//       取 min 合并。
// 关键参数:
//   - segLen = 0.8：每个走廊段的长度，决定重复频率。
//   - corridorW/corridorH = 0.3：走廊内部净空半宽/半高，配合
//     wallThick=0.05 决定通道大小。
//   - hue 由 segment 编号 + 时间决定，让每一段颜色随深度渐变，增强
//     「无限」错觉。
// 性能特征: map() 内只有常数次运算（无迭代循环），march 150 步/maxD=25，
//           是该目录里较轻量的 shader，可承受更多细节层。
// 已知限制/优化方向:
//   - 目前 grid/panel 的镂空判定用多个 max() 拼接，可读性一般，如需增加
//     走廊分岔可以在这里扩展成真正的多分支网络（类似 maze3d 的做法）。

// ─── Scene SDF ────────────────────────────────────────────────────────────────
static float map(float3 p, float t) {
    // Repeat space along Z axis to create infinite corridor segments
    float zSeg = p.z;
    float segLen = 0.8f;
    float segment = round(zSeg / segLen);
    p.z -= segment * segLen;

    // Corridor walls: box with hole in the middle
    float wallThick = 0.05f;
    float corridorW = 0.3f;
    float corridorH = 0.3f;

    // Outer walls
    float outer = max(abs(p.x) - corridorW, max(abs(p.y) - corridorH, abs(p.z) - segLen * 0.5f));
    // Inner hole
    float inner = max(-abs(p.x) + corridorW - wallThick, max(-abs(p.y) + corridorH - wallThick, 0.0f));
    // Frame: outer minus inner
    float wall = max(outer, inner);

    // Floor/ceiling grid lines
    float grid = 1e10f;
    {
        float gx = abs(p.x) - 0.02f;
        float gz = abs(p.z) - segLen * 0.45f;
        float gy1 = abs(p.y - corridorH + 0.02f) - 0.01f;
        float gy2 = abs(p.y + corridorH - 0.02f) - 0.01f;
        grid = min(min(max(gx, gy1), max(gx, gy2)), max(gz, max(abs(p.y) - corridorH + 0.02f, 0.0f)));
    }

    // Energy orb at center of each segment
    float orb = length(p - float3(0, 0, 0)) - 0.08f;

    // Glow panel on the far wall
    float panel = max(abs(p.x) - 0.15f, max(abs(p.y) - 0.12f, abs(p.z) - segLen * 0.4f + wallThick));

    return min(min(wall, orb), panel);
}

// ─── Normal ───────────────────────────────────────────────────────────────────
static float3 calcNormal(float3 p, float t) {
    float2 e = float2(0.002f, 0.0f);
    return normalize(float3(
        map(p + e.xyy, t) - map(p - e.xyy, t),
        map(p + e.yxy, t) - map(p - e.yxy, t),
        map(p + e.yyx, t) - map(p - e.yyx, t)
    ));
}

// ─── HSV → RGB ────────────────────────────────────────────────────────────────
static float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0f, 2.0f/3.0f, 1.0f/3.0f, 3.0f);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0f - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), c.y);
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
    float3 bgColor = float3(0.0f, 0.0f, 0.0f);

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

    // ─── Ray march ─────────────────────────────────────────────────────────
    float t     = uniforms.time;
    float march = 0.0f;
    float maxD  = 25.0f;

    for (int i = 0; i < 150; i++) {
        float3 p = ro + rd * march;
        float d = map(p, t);

        if (d < 0.004f) {
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.5f, 1.0f, 0.3f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * abs(n.y);

            // Determine segment and what we hit
            float segLen = 0.8f;
            float segment = round(p.z / segLen);
            float hue = fract(segment * 0.15f + t * 0.02f);

            float3 col;
            float dist = length(p);

            // Energy orb: bright glowing sphere
            float orbD = length(p) - 0.08f;
            if (abs(orbD) < 0.01f) {
                col = hsv2rgb(float3(hue, 0.9f, 1.5f)) * 3.0f;
            } else {
                // Wall: dark with colored edge glow
                col = float3(0.05f, 0.05f, 0.08f);
                float rim = 1.0f - max(dot(-rd, n), 0.0f);
                rim = pow(rim, 3.0f) * 0.5f;
                col += hsv2rgb(float3(hue, 0.6f, 0.8f)) * rim * 2.0f;
                col += float3(0.3f, 0.5f, 0.8f) * dif * 0.3f;
            }

            // Distance fade (deeper = darker)
            col *= exp(-march * 0.4f);

            return float4(col, 1.0f);
        }
        march += d;
        if (march > maxD) break;
    }

    return float4(bgColor, 1.0f);
}
