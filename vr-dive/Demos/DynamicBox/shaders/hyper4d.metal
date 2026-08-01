// hyper4d.metal – 4D Hypersphere Projection
//
// Renders 4D geometric shapes projected into 3D via ray marching.
// The 4D scene rotates continuously in 4D space (XY, XZ, YZ, XW, YW, ZW
// planes), and the 3D ray-march samples the 4D distance field.
//
// This creates morphing 3D shapes that smoothly transform as 4D rotation
// brings different cross-sections into view.
//
// ─── 设计方案 ────────────────────────────────────────────────────────────
// 思路: 在 4D 空间构建两个隐式几何体 (4D 超环面 sdHyperTorus + 4D 球
//       格点 sdSpherePack4D)，实际在 fragment 主循环中依次应用 rotXY→
//       rotZW→rotXW→rotYZ 四个旋转平面让 4D 坐标随时间旋转，再把当前
//       3D march 采样点通过透视投影公式 (w 分量由 3D 距离反推) 映射回
//       4D，取 4D distance field 当作 3D 等效距离场使用（并乘以投影
//       Jacobian `1/(1+p.w*fov)` 修正步长）。
// 关键参数:
//   - fov = 0.4：4D→3D 透视投影的强度，越大 W 轴形变越明显。
//   - 四个旋转的角速度各自独立 (0.17/0.23/0.11/0.13)，避免所有旋转同步
//     导致画面呆板。
//   - sdHyperTorus 的 R1/R2/r 决定超环面「粗细/半径比」；
//     sdSpherePack4D 的格点间距 0.5 决定球体阵列密度。
// 性能特征: 每次法线计算需要额外 3 次 4D map 求值（比 3D 版本略贵），
//           march 100 步/maxD=25。整体是本目录概念最复杂的 shader。
// 已知限制/优化方向:
//   - w 分量目前由透视投影反推而非独立自由度，效果上偏「伪 4D」，若要
//     更真实的 4D 旋转体验可以让 w 与 3D 位置解耦（例如引入独立的 w
//     uniform 或时间驱动的 w 偏移）。

// ─── 4D rotation helpers ──────────────────────────────────────────────────────
static float4 rotXY(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x*c - p.y*s, p.x*s + p.y*c, p.z, p.w);
}
static float4 rotXZ(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x*c - p.z*s, p.y, p.x*s + p.z*c, p.w);
}
static float4 rotXW(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x*c - p.w*s, p.y, p.z, p.x*s + p.w*c);
}
static float4 rotYZ(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x, p.y*c - p.z*s, p.y*s + p.z*c, p.w);
}
static float4 rotYW(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x, p.y*c - p.w*s, p.z, p.y*s + p.w*c);
}
static float4 rotZW(float4 p, float a) {
    float s = sin(a), c = cos(a);
    return float4(p.x, p.y, p.z*c - p.w*s, p.z*s + p.w*c);
}

// ─── 4D SDF: distance from a 4D point to a 4D hyper-torus ────────────────────
// A 3-torus in 4D: (sqrt(x²+y²) - R1)² + (sqrt(z²+w²) - R2)² = r²
static float sdHyperTorus(float4 p, float t) {
    float R1 = 0.6f;
    float R2 = 0.35f;
    float r  = 0.08f;

    float d1 = length(p.xy) - R1;
    float d2 = length(p.zw) - R2;
    return length(float2(d1, d2)) - r;
}

// ─── 4D SDF: 4D sphere packing (hyper-balls at lattice points) ───────────────
static float sdSpherePack4D(float4 p, float t) {
    float4 gp = round(p / 0.5f) * 0.5f;
    float  r  = 0.12f + 0.03f * sin(t * 0.5f + dot(gp, float4(1.3f, 2.7f, 3.1f, 4.9f)));
    return length(p - gp) - r;
}

// ─── Combined 4D SDF ─────────────────────────────────────────────────────────
static float map(float4 p, float t) {
    float d1 = sdHyperTorus(p, t);
    float d2 = sdSpherePack4D(p, t);
    return min(d1, d2);
}

// ─── 4D → 3D perspective projection ──────────────────────────────────────────
// Projects a 4D point into 3D using perspective divide by (1 + w * fovFactor).
// Points with large |w| are pushed toward the center (perspective foreshortening).
static float3 project4D(float4 p, float fov) {
    float wScale = 1.0f / (1.0f + p.w * fov);
    return p.xyz * wScale;
}

// ─── 3D normal via 4D distance gradient ──────────────────────────────────────
static float3 calcNormal(float3 p3, float4 p4, float t) {
    float2 e = float2(0.002f, 0.0f);
    float4 dx = float4(e.x, 0, 0, 0);
    float4 dy = float4(0, e.x, 0, 0);
    float4 dz = float4(0, 0, e.x, 0);
    return normalize(float3(
        map(p4 + dx, t) - map(p4 - dx, t),
        map(p4 + dy, t) - map(p4 - dy, t),
        map(p4 + dz, t) - map(p4 - dz, t)
    ));
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
    float3 bgColor = float3(0.0f, 0.0f, 0.03f);

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

    // ─── 4D rotation (animates over time) ──────────────────────────────────
    float t = uniforms.time;
    float4 p4;

    // ─── Ray march ─────────────────────────────────────────────────────────
    float march = 0.0f;
    float maxD  = 25.0f;

    for (int i = 0; i < 100; i++) {
        float3 p3 = ro + rd * march;

        // Build 4D coordinate from 3D position + implicit w from projection
        // For perspective projection, w is derived from the 3D distance
        float fov = 0.45f;
        float w = (1.0f - fov * length(p3)) / fov;
        // Alternative: start with w=0 and use 4D rotation to mix it in

        p4 = float4(p3, 0.0f);

        // Apply 4D rotations
        p4 = rotXY(p4, t * 0.17f);
        p4 = rotZW(p4, t * 0.23f);
        p4 = rotXW(p4, t * 0.11f);
        p4 = rotYZ(p4, t * 0.13f);

        // Perspective projection from 4D → 3D
        float3 q3 = project4D(p4, 0.4f);

        // The 4D SDF evaluated at the 4D point
        float d = map(p4, t);

        // Scale SDF by the projection Jacobian for correct step size
        float projScale = 1.0f / (1.0f + p4.w * 0.4f);
        d *= projScale;

        if (d < 0.004f) {
            float3 n = calcNormal(q3, p4, t);
            float3 light = normalize(float3(0.5f, 1.0f, 0.3f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * abs(n.y);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 4.0f) * 0.7f;

            // Iridescent color shifting with 4D position
            float hue = fract(length(p4) * 0.3f + t * 0.04f + p4.w * 0.1f);
            float3 col;
            {
                float4 K = float4(1.0f, 2.0f/3.0f, 1.0f/3.0f, 3.0f);
                float3 pp = abs(fract(hue + K.xyz) * 6.0f - K.www);
                col = clamp(pp - K.xxx, 0.0f, 1.0f);
            }
            col = col * (dif * 1.0f + amb * 0.5f) + float3(0.3f, 0.6f, 1.0f) * rim;
            col *= exp(-march * 0.3f);

            return float4(col, 1.0f);
        }
        march += d;
        if (march > maxD) break;
    }

    return float4(bgColor, 1.0f);
}
