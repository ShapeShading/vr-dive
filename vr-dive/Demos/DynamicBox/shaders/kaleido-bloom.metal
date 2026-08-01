// kaleido-bloom.metal - Breathing kaleidoscopic bloom
//
// Design: twelve mirrored radial sectors carry three offset petal shells. Each
// petal swells and twists with a different phase, producing a slow breathing
// flower with translucent-looking edge color and changing silhouettes.
// Key parameters: 12-fold symmetry, 3 petal shells, and a 0.08m shell radius.
// Performance: analytic shell distance, 80 march steps, no texture lookup.
// Known limit: the bloom is intentionally centered and does not form a tunnel.

static float3 palette(float t) {
    float3 a = float3(0.35f, 0.42f, 0.55f);
    float3 b = float3(0.60f, 0.45f, 0.35f);
    float3 c = float3(0.9f, 1.0f, 0.7f);
    float3 d = float3(0.0f, 0.23f, 0.47f);
    return a + b * cos(6.28318f * (c * t + d));
}

static float petalDE(float3 p, float time) {
    float angle = atan2(p.z, p.x);
    float sector = 6.2831853f / 12.0f;
    angle = fmod(angle + sector * 0.5f, sector);
    angle = abs(angle - sector * 0.5f);

    float radial = length(p.xz);
    float d = 1e5f;
    for (int shell = 0; shell < 3; shell++) {
        float phase = time * (0.35f + float(shell) * 0.09f) + float(shell) * 2.1f;
        float lobe = 0.10f * sin(angle * 18.0f + phase) + 0.045f * cos(angle * 36.0f - phase);
        float radius = 0.24f + float(shell) * 0.19f + lobe;
        float vertical = 0.17f + 0.08f * sin(angle * 10.0f + phase * 1.4f);
        float2 shellPoint = float2(radial - radius, p.y - 0.08f * sin(angle * 14.0f + phase));
        float shellD = length(shellPoint) - vertical;
        d = min(d, shellD);
    }

    float stem = length(float2(radial, p.y)) - 0.13f;
    return min(d, stem);
}

static float3 calcNormal(float3 p, float time) {
    float2 e = float2(0.0015f, 0.0f);
    return normalize(float3(
        petalDE(p + e.xyy, time) - petalDE(p - e.xyy, time),
        petalDE(p + e.yxy, time) - petalDE(p - e.yxy, time),
        petalDE(p + e.yyx, time) - petalDE(p - e.yyx, time)));
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
    float3 bg = float3(0.015f, 0.006f, 0.025f);
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
        float d = petalDE(p, time);
        if (d < 0.003f) {
            float3 n = calcNormal(p, time);
            float3 light = normalize(float3(0.3f, 0.9f, -0.4f));
            float dif = max(dot(n, light), 0.0f);
            float rim = pow(1.0f - max(dot(-rd, n), 0.0f), 2.0f);
            float colT = atan2(p.z, p.x) * 0.55f + length(p) * 0.9f + time * 0.035f;
            float3 col = palette(colT) * (0.32f + dif * 1.15f);
            col += float3(1.0f, 0.35f, 0.7f) * rim * 0.65f;
            col *= exp(-march * 0.20f);
            return float4(col, 1.0f);
        }
        march += max(d * 0.70f, 0.008f);
        if (march > 25.0f) break;
    }
    return float4(bg, 1.0f);
}
