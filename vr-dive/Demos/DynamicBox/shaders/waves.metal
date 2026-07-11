// waves.metal – Colorful Wave Interference
//
// A sample fragment shader for DynamicBox showing cosine-based
// color waves that drift through the box volume.
//
// The function MUST be named `dynamicBoxFragment`.

static float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0f - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0f, 1.0f), c.y);
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

    if (!insideBox) {
        float3 entryNormal;
        float  tEnter = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (tEnter < 0.0f) {
            return float4(0.01f, 0.01f, 0.03f, 1.0f);
        }
        marchOrigin = boxEye + boxRd * (tEnter + 1e-3f);
    } else {
        marchOrigin = boxEye;
    }

    float3 exitNormal;
    float  tExit = db_boxHit(marchOrigin, boxRd, DB_BOXDIMS, exitNormal, false);
    if (tExit <= 0.0f) discard_fragment();

    float3 eye = (uniforms.patternTransform * float4(marchOrigin, 1.0f)).xyz;
    float3 rd  = normalize(float3(uniforms.patternTransform * float4(boxRd, 0.0f)));

    // ─── Wave Interference ─────────────────────────────────────────────────
    float t = uniforms.time;
    float maxMarch = 25.0f;

    float3 pos = eye;
    float  accumA = 0.0f;
    float3 accumC = float3(0.0f);

    float stepSize = 0.03f;
    int   maxSteps = int(maxMarch / stepSize) + 2;

    for (int i = 0; i < min(maxSteps, 200); i++) {
        float dist = (float(i) + 0.5f) * stepSize;
        if (dist > maxMarch) break;
        float3 p = eye + rd * dist;

        // Layered sine waves in different directions
        float w = sin(p.x * 4.0f + t * 0.7f) * sin(p.y * 5.0f + t * 0.5f) * sin(p.z * 3.0f + t * 0.9f);
        w += 0.5f * sin(p.x * 8.0f - t * 1.1f) * sin(p.z * 7.0f + t * 0.6f);
        w += 0.3f * sin(p.y * 9.0f + t * 0.8f) * sin((p.x + p.z) * 6.0f - t * 0.4f);
        w = abs(w);

        if (w > 0.05f) {
            float hue = fract(w * 3.0f + t * 0.03f);
            float3 rgb = hsv2rgb(float3(hue, 0.8f, w * 2.5f));

            float alpha = w * 0.15f;
            accumC += rgb * alpha * (1.0f - accumA);
            accumA += alpha * (1.0f - accumA);

            if (accumA > 0.98f) break;
        }
    }

    float3 bgColor = float3(0.01f, 0.01f, 0.03f);
    float3 finalColor = mix(bgColor, accumC, accumA);
    return float4(finalColor, 1.0f);
}
