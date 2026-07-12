// kaleido-crystal.metal - Drifting crystal kaleidoscope
//
// Design: combine mirrored octahedral folds with animated plane cuts and a
// twisted shell. The result is a faceted crystal lattice whose silhouette
// changes as the cuts slide through it, rather than merely rotating in place.
// Key parameters: 8-fold fold symmetry, 5 animated cuts, and 9 DE iterations.
// Performance: compact folded DE, 72 march steps, and six normal samples.
// Known limit: sharp cuts can alias at very close range; the DE keeps them stable.

static float3 palette(float t) {
    float3 a = float3(0.42f, 0.48f, 0.54f);
    float3 b = float3(0.52f, 0.42f, 0.46f);
    float3 c = float3(0.7f, 0.95f, 1.0f);
    float3 d = float3(0.02f, 0.20f, 0.38f);
    return a + b * cos(6.28318f * (c * t + d));
}

static float crystalDE(float3 p, float time) {
    float3 q = p;
    float dr = 1.0f;
    for (int i = 0; i < 9; i++) {
        q = abs(q);
        if (q.x < q.y) { float v = q.x; q.x = q.y; q.y = v; }
        if (q.x < q.z) { float v = q.x; q.x = q.z; q.z = v; }
        q.xy = q.xy * float2(0.94f, 1.06f);
        float cut = sin(q.x * 4.0f + time * 0.5f) * 0.12f
                  + cos(q.z * 3.0f - time * 0.37f) * 0.10f;
        q.y -= cut;
        q = q * 1.72f - float3(0.72f, 0.52f, 0.63f);
        dr = dr * 1.72f + 1.0f;
    }
    float shell = length(q) - 0.055f;
    float facets = max(abs(q.x) - 0.045f, max(abs(q.y) - 0.045f, abs(q.z) - 0.045f));
    return min(shell, facets * 0.85f) / dr;
}

static float3 calcNormal(float3 p, float time) {
    float2 e = float2(0.0012f, 0.0f);
    return normalize(float3(
        crystalDE(p + e.xyy, time) - crystalDE(p - e.xyy, time),
        crystalDE(p + e.yxy, time) - crystalDE(p - e.yxy, time),
        crystalDE(p + e.yyx, time) - crystalDE(p - e.yyx, time)));
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
    float3 boxEye = (camWorld - uniforms.objectCenter.xyz) / uniforms.boxScale;
    float3 boxRd = normalize(in.worldPos - camWorld);
    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    float3 bg = float3(0.004f, 0.018f, 0.025f);
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

    for (int i = 0; i < 72; i++) {
        float3 p = ro + rd * march;
        float d = crystalDE(p, time);
        if (d < 0.003f) {
            float3 n = calcNormal(p, time);
            float3 light = normalize(float3(-0.5f, 0.75f, 0.65f));
            float dif = max(dot(n, light), 0.0f);
            float spec = pow(max(dot(reflect(-light, n), -rd), 0.0f), 24.0f);
            float rim = pow(1.0f - max(dot(-rd, n), 0.0f), 2.3f);
            float3 col = palette(length(p) * 1.2f + time * 0.045f + n.z * 0.3f);
            col *= 0.28f + dif * 1.0f;
            col += float3(0.65f, 0.9f, 1.0f) * (spec * 0.9f + rim * 0.55f);
            col *= exp(-march * 0.18f);
            return float4(col, 1.0f);
        }
        march += max(d * 0.78f, 0.008f);
        if (march > 25.0f) break;
    }
    return float4(bg, 1.0f);
}
