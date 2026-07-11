// default.metal – 3D Grid of Light Points
//
// This is the default fragment shader for DynamicBox.
// It renders glowing points on a regular 3D grid inside the bounding box.
//
// The function MUST be named `dynamicBoxFragment` and accept the standard
// signature shown below.  The structs and `db_boxHit` helper are prepended
// automatically by the renderer; you only need to write the fragment function
// body and your own helpers.

// ─── HSV → RGB helper ─────────────────────────────────────────────────────────
static float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);
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

    // Camera world position
    float4x4 v2w    = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    // Box-local space
    float3 center  = uniforms.objectCenter.xyz;
    float  sc      = uniforms.boxScale;
    float3 boxEye  = (camWorld - center) / sc;
    float3 boxRd   = normalize(in.worldPos - camWorld);

    // Box entry test
    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    float3 marchOrigin;
    float3 bgColor = float3(0.02f, 0.02f, 0.05f);

    if (!insideBox) {
        float3 entryNormal;
        float  tEnter = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (tEnter < 0.0f) {
            return float4(bgColor, 1.0f);
        }
        marchOrigin = boxEye + boxRd * (tEnter + 1e-3f);
    } else {
        marchOrigin = boxEye;
    }

    float3 exitNormal;
    float  tExit = db_boxHit(marchOrigin, boxRd, DB_BOXDIMS, exitNormal, false);
    if (tExit <= 0.0f) discard_fragment();

    // Apply pattern navigation transform
    float3 eye = (uniforms.patternTransform * float4(marchOrigin, 1.0f)).xyz;
    float3 rd  = normalize(float3(uniforms.patternTransform * float4(boxRd, 0.0f)));

    // ─── 3D Grid Point Rendering ───────────────────────────────────────────
    float gridSpacing = 0.12f;
    float pointRadius = 0.025f;
    float glowFalloff = 120.0f;
    float maxMarch    = tExit + 0.5f;

    float3 pos    = eye;
    float  accumA = 0.0f;
    float3 accumC = float3(0.0f);

    float stepSize = gridSpacing * 0.5f;
    int   maxSteps = int(maxMarch / stepSize) + 2;

    for (int i = 0; i < min(maxSteps, 128); i++) {
        float t = (float(i) + 0.5f) * stepSize;
        if (t > maxMarch) break;
        float3 p = eye + rd * t;

        float3 gp = round(p / gridSpacing) * gridSpacing;
        float  dist = length(p - gp);

        if (dist < pointRadius * 2.5f) {
            float glow = exp(-dist * dist * glowFalloff);
            float pulse = 0.7f + 0.3f * sin(uniforms.time * 2.0f + dot(gp, float3(3.7f, 5.1f, 7.3f)));
            glow *= pulse;

            float hue = fract(dot(gp, float3(0.373f, 0.617f, 0.819f)) * 0.4f + uniforms.time * 0.05f);
            float3 rgb = hsv2rgb(float3(hue, 0.7f, 1.0f));

            float alpha = glow * 0.35f;
            accumC += rgb * alpha * (1.0f - accumA);
            accumA += alpha * (1.0f - accumA);

            if (accumA > 0.98f) break;
        }
    }

    // Fade box edges
    float3 absPos  = abs(pos);
    float  edgeDist = min(min(DB_BOXDIMS.x - absPos.x,
                               DB_BOXDIMS.y - absPos.y),
                               DB_BOXDIMS.z - absPos.z);
    float edgeFade = smoothstep(0.0f, 0.08f, edgeDist);

    float3 finalColor = mix(bgColor, accumC, accumA) * edgeFade;
    return float4(finalColor, 1.0f);
}
