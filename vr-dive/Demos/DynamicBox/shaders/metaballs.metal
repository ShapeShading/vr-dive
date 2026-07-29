// metaballs.metal – Organic Metaballs
//
// Renders animated blobby metaballs using sphere-based SDF with
// smooth blending. Multiple metaballs orbit and merge together
// like living cells, with iridescent surface coloring.

// ─── Metaball SDF (smooth union) ─────────────────────────────────────────────
static float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return mix(d2, d1, h) - k * h * (1.0f - h);
}

// ─── Scene SDF ────────────────────────────────────────────────────────────────
static float map(float3 p, float t) {
    float d = 1e10f;
    float k = 0.3f; // blend factor

    // Central cluster
    float3 c = float3(0.0f);
    float3 p1 = p - (c + float3(0.25f * cos(t * 0.7f), 0.15f * sin(t * 0.5f), 0.2f * sin(t * 0.6f)));
    d = opSmoothUnion(d, length(p1) - 0.25f, k);

    float3 p2 = p - (c + float3(-0.2f * cos(t * 0.4f + 1.0f), 0.1f * sin(t * 0.8f + 2.0f), -0.15f * cos(t * 0.3f + 3.0f)));
    d = opSmoothUnion(d, length(p2) - 0.22f, k);

    float3 p3 = p - (c + float3(0.1f * sin(t * 0.6f + 1.5f), -0.25f * cos(t * 0.5f + 0.5f), 0.1f * sin(t * 0.7f + 2.5f)));
    d = opSmoothUnion(d, length(p3) - 0.2f, k);

    // Orbiting satellites
    float a1 = t * 0.5f;
    float3 s1 = float3(0.5f * cos(a1), 0.2f * sin(a1 * 1.3f), 0.4f * sin(a1 * 0.7f));
    float3 p4 = p - s1;
    d = opSmoothUnion(d, length(p4) - 0.15f, k * 0.7f);

    float a2 = t * 0.4f + 2.0f;
    float3 s2 = float3(0.4f * sin(a2 * 0.8f), 0.5f * cos(a2), 0.3f * sin(a2 * 1.1f));
    float3 p5 = p - s2;
    d = opSmoothUnion(d, length(p5) - 0.12f, k * 0.5f);

    return d;
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
    float3 bgColor = float3(0.0f, 0.0f, 0.02f);

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

    for (int i = 0; i < 80; i++) {
        float3 p = ro + rd * march;
        float d = map(p, t);
        if (d < 0.003f) {
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.6f, 0.8f, 0.5f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 3.0f) * 0.6f;

            // Iridescent cell-like coloring
            float hue = fract(length(p) * 0.5f + t * 0.04f);
            float3 col;
            {
                float4 K = float4(1.0f, 2.0f/3.0f, 1.0f/3.0f, 3.0f);
                float3 pp = abs(fract(hue + K.xyz) * 6.0f - K.www);
                col = clamp(pp - K.xxx, 0.0f, 1.0f);
            }
            // Glass-like translucency
            col = col * (dif * 1.0f + amb * 0.6f) + float3(0.2f, 0.4f, 0.8f) * rim;
            col *= exp(-march * 0.2f);

            return float4(col, 1.0f);
        }
        march += d;
        if (march > maxD) break;
    }

    return float4(bgColor, 1.0f);
}
