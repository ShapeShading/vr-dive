// kaleido-ribbons.metal - Flowing kaleidoscopic ribbons
//
// Design: fold polar angle into twelve mirrored sectors, then bend a set of
// ribbon tubes through the depth of the box. The ribbon centerline drifts in
// time, so the pattern reads as a living woven object rather than a rigid IFS.
// Key parameters: 12 sectors, 0.16m ribbon radius, and three depth layers.
// Performance: one analytic distance function and 80 march steps.
// Known limit: the strongest detail is near the center; distant layers fade.

static float3 palette(float t) {
    float3 a = float3(0.45f, 0.38f, 0.50f);
    float3 b = float3(0.55f, 0.48f, 0.40f);
    float3 c = float3(1.0f, 0.8f, 0.55f);
    float3 d = float3(0.02f, 0.18f, 0.34f);
    return a + b * cos(6.28318f * (c * t + d));
}

static float2 foldPolar(float2 p, float sectors) {
    float angle = atan2(p.y, p.x);
    float sector = 6.2831853f / sectors;
    angle = fmod(angle + 0.5f * sector, sector);
    angle = abs(angle - 0.5f * sector);
    return float2(cos(angle), sin(angle)) * length(p);
}

static float ribbonDE(float3 p, float time) {
    float2 q = foldPolar(p.xz, 12.0f);
    float radial = length(q);
    float best = 1e5f;

    for (int layer = 0; layer < 3; layer++) {
        float z = (float(layer) - 1.0f) * 0.42f;
        float phase = time * (0.45f + float(layer) * 0.12f) + float(layer) * 1.9f;
        float center = 0.45f + 0.16f * sin(q.y * 5.0f + phase) + 0.06f * cos(p.y * 4.0f - phase);
        float2 delta = float2(radial - center, p.y - 0.12f * sin(q.x * 8.0f + phase));
        float tube = length(delta) - 0.105f;
        float depth = abs(p.z - z) - 0.13f;
        best = min(best, max(tube, depth));
    }

    float core = length(p) - 0.18f;
    return min(best, core);
}

static float3 calcNormal(float3 p, float time) {
    float2 e = float2(0.0015f, 0.0f);
    return normalize(float3(
        ribbonDE(p + e.xyy, time) - ribbonDE(p - e.xyy, time),
        ribbonDE(p + e.yxy, time) - ribbonDE(p - e.yxy, time),
        ribbonDE(p + e.yyx, time) - ribbonDE(p - e.yyx, time)));
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut        in         [[stage_in]],
    constant DynamicBoxUniforms &uniforms [[buffer(0)]],
    constant float4x4         *v2wMats   [[buffer(1)]],
    constant float4x4         *vpMatrices [[buffer(2)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);
    float4x4 v2w = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);
    float3 center = uniforms.objectCenter.xyz;
    float sc = uniforms.boxScale;
    float3 boxEye = (camWorld - center) / sc;
    float3 boxRd = normalize(in.worldPos - camWorld);
    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    float3 bg = float3(0.008f, 0.012f, 0.035f);
    float3 origin;

    if (!insideBox) {
        float3 entryNormal;
        float entry = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (entry < 0.0f) return float4(bg, 1.0f);
        origin = boxEye + boxRd * (entry + 1e-3f);
    } else {
        origin = boxEye;
    }
    float3 exitNormal;
    if (db_boxHit(origin, boxRd, DB_BOXDIMS, exitNormal, false) <= 0.0f) discard_fragment();

    float3 ro = (uniforms.patternTransform * float4(origin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRd, 0.0f)));
    float time = uniforms.time;
    float march = 0.0f;

    for (int i = 0; i < 80; i++) {
        float3 p = ro + rd * march;
        float d = ribbonDE(p, time);
        if (d < 0.003f) {
            float3 n = calcNormal(p, time);
            float3 light = normalize(float3(-0.4f, 0.8f, 0.6f));
            float dif = max(dot(n, light), 0.0f);
            float rim = pow(1.0f - max(dot(-rd, n), 0.0f), 2.5f);
            float3 col = palette(length(p) * 0.8f + time * 0.05f + n.y * 0.2f);
            col *= 0.35f + dif * 1.1f;
            col += float3(0.2f, 0.55f, 1.0f) * rim * 0.75f;
            col *= exp(-march * 0.22f);
            return float4(col, 1.0f);
        }
        march += max(d * 0.72f, 0.008f);
        if (march > 25.0f) break;
    }
    return float4(bg, 1.0f);
}
