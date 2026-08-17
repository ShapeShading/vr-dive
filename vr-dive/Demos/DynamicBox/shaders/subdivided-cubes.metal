// subdivided-cubes.metal — random recursive cube volume
//
// Adapted from the compact shader at:
// https://fragcoord.xyz/s/ctrzdrsa
//
// The source uses a screen-space camera. DynamicBox supplies a real per-eye
// ray, while the procedural core remains the same: recursively subdivide a
// cube, stop on a pseudorandom branch, and accumulate colored volume glow.

static float scRandom(float3 cell) {
    // Same hash shape as fract(dot(P, sin(P.zxy))) from the source shader.
    return fract(dot(cell, sin(cell.zxy)));
}

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
    float3 background = float3(0.001f, 0.004f, 0.012f);

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

    float3 exitNormal;
    float exitDistance = db_boxHit(rayOrigin, boxRay, DB_BOXDIMS, exitNormal, false);
    if (exitDistance <= 0.0f) discard_fragment();

    float3 ro = (uniforms.patternTransform * float4(rayOrigin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRay, 0.0f)));

    float4 accumulated = float4(0.0f);
    float travel = 0.0f;
    float time = uniforms.time;

    for (int i = 0; i < 50; i++) {
        float3 point = ro + rd * travel;
        point.z += time;

        float subdivision = 1.0f;
        float lastRandom = 0.5f;

        // Randomly subdivide cubes into 8 smaller cubes. The source stops
        // when the cell hash exceeds .7; keep the final hash for coloring.
        for (float level = 1.0f; level < 16.0f; level *= 2.0f) {
            float3 cell = ceil(point);
            lastRandom = scRandom(cell);
            if (lastRandom > 0.7f) break;
            point *= 2.0f;
            subdivision = level * 2.0f;
        }

        float3 local = abs(fract(point) - 0.5f);
        float density = abs(max(max(local.x, local.y), local.z)
            - 0.3f / subdivision) + 0.01f;

        float colorBand = ceil(lastRandom * 9.0f + time);
        accumulated += exp(sin(colorBand + float4(0.0f, 2.0f, 4.0f, 0.0f)))
            / density;

        // Preserve the source shader's ray step while limiting traversal to
        // the DynamicBox volume through exitDistance.
        travel += density * 0.3f;
        if (travel >= exitDistance) break;
    }

    float4 color = tanh(accumulated / 700.0f);
    color *= color * color;
    return float4(max(color.rgb, 0.0f), 1.0f);
}
