// cosine-cross-orbit.metal — cosine perturbation orbit
//
// Adapted from the compact shader at:
// https://fragcoord.xyz/s/q91w1o8q
//
// This keeps the original mix/cross-product transform, nine cosine orbit
// perturbations, and accumulated color energy, while replacing the source's
// screen-space ray with DynamicBox's real per-eye ray.

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut         in          [[stage_in]],
    constant DynamicBoxUniforms &uniforms   [[buffer(0)]],
    constant float4x4           *v2wMats    [[buffer(1)]],
    constant float4x4           *vpMatrices  [[buffer(2)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);

    float3 cameraWorld = v2wMats[vi][3].xyz;
    float3 boxEye = (cameraWorld - uniforms.objectCenter.xyz) / uniforms.boxScale;
    float3 boxRay = normalize(in.worldPos - cameraWorld);
    float3 background = float3(0.001f, 0.003f, 0.012f);

    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    float3 rayOrigin;
    if (!insideBox) {
        float3 entryNormal;
        float entry = db_boxHit(boxEye, boxRay, DB_BOXDIMS, entryNormal, true);
        if (entry < 0.0f) return float4(background, 1.0f);
        rayOrigin = boxEye + boxRay * (entry + 1e-3f);
    } else {
        rayOrigin = boxEye;
    }

    float3 ro = (uniforms.patternTransform * float4(rayOrigin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRay, 0.0f)));
    float time = uniforms.time;

    // Increase procedural-domain frequency so the pattern is approximately
    // one eighth of its previous visible size.
    const float patternScale = 8.0f;

    float4 accumulated = float4(0.0f);
    float z = 0.0f;
    float orbitScale = 1.0f;

    for (int i = 1; i <= 90; i++) {
        float3 p = (ro + z * rd) * patternScale;
        // The source uses +9 in a screen-space coordinate system. Compress
        // that offset to the local scale of the DynamicBox volume.
        p.z += 2.8f * patternScale;

        // The compact source leaves these loop variables implicit. Explicit
        // initialization keeps the Metal version deterministic across GPUs.
        float3 a = float3(0.0f);
        float phase = orbitScale - time;
        a -= 0.6f;
        a = mix(dot(a, p) * a, p, cos(phase))
            - sin(phase) * cross(a, p);

        orbitScale = sqrt(max(length(a - a.zxy), 1e-5f));

        // Nine cosine perturbations, matching d++ < 9 in the source.
        for (int j = 1; j <= 9; j++) {
            float fj = float(j);
            a += cos(a * fj + time).yzx / fj;
        }

        // Safe version of dot(a, a / a), whose compact form can divide by
        // zero at isolated points.
        float3 safeA = float3(
            a.x + (abs(a.x) < 1e-5f ? 1e-5f : 0.0f),
            a.y + (abs(a.y) < 1e-5f ? 1e-5f : 0.0f),
            a.z + (abs(a.z) < 1e-5f ? 1e-5f : 0.0f));
        float field = dot(a, a / safeA);

        float safeScale = max(orbitScale, 1e-4f);
        float energyStep = sqrt(max(field, -field * 0.1f)) * safeScale / 14.0f;
        z += max(energyStep / patternScale, 0.0015f / patternScale);

        float colorScale = max(safeScale, 0.001f);
        accumulated += float4(colorScale, 1.0f, z * patternScale / 5.0f, 1.0f)
            / (colorScale * colorScale) * float(i);
    }

    float4 color = tanh(accumulated * accumulated / 4e6f);
    return float4(max(color.rgb, 0.0f), 1.0f);
}
