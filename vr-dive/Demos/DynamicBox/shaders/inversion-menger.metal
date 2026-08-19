// inversion-menger.metal — spherical-inversion Menger glow
//
// Adapted from the compact raymarch demo at:
// https://fragcoord.xyz/s/c5y70a5f
//
// The original shader uses a screen-space camera. DynamicBox supplies a real
// per-eye ray instead, so the same mirror, spherical inversion, box fold, and
// Menger fold are evaluated along the ray inside the shared box volume.

// Equivalent to the source shader's A(x, y) macro. Keep the larger component
// first; this is the inexpensive Menger-style coordinate fold used by the
// original pattern.
static float2 imSortDescending(float2 pair) {
    float a = min(pair.x - pair.y, 0.0f);
    pair.x -= a;
    pair.y += a;
    return pair;
}

static float3 imFoldedPoint(float3 point, float time, thread float &radiusSquared) {
    // Rotate and mirror. The four phase offsets intentionally match the
    // mat2(cos(u_time*.2 + vec4(0,11,33,0))) construction in the source.
    float a = time * 0.2f;
    float c00 = cos(a);
    float c01 = cos(a + 11.0f);
    float c10 = cos(a + 33.0f);
    float c11 = cos(a);
    point.xz = -abs(float2(
        point.x * c00 + point.z * c10,
        point.x * c01 + point.z * c11));

    // Spherical inversion.
    radiusSquared = max(dot(point, point), 1e-5f);
    point /= radiusSquared;

    for (int i = 0; i < 9; i++) {
        // Box fold.
        point = clamp(point, -0.2f, 0.2f) * 2.0f - point;

        // Menger fold.
        point.xy = imSortDescending(point.xy);
        point.xz = imSortDescending(point.xz);
        point.yz = imSortDescending(point.yz);

        // Scale, translate, and rotate the folded cell.
        point = point * 2.5f + float3(-3.0f, -0.5f, 0.0f);
        point.yz = float2(point.z, -point.y);
    }

    return point;
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut        in         [[stage_in]],
    constant DynamicBoxUniforms &uniforms [[buffer(0)]],
    constant float4x4          *v2wMats  [[buffer(1)]],
    constant float4x4          *vpMatrices [[buffer(2)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);

    float3 cameraWorld = v2wMats[vi][3].xyz;
    float3 boxEye = (cameraWorld - uniforms.objectCenter.xyz) / uniforms.boxScale;
    float3 boxRay = normalize(in.worldPos - cameraWorld);
    float3 background = float3(0.002f, 0.004f, 0.012f);

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

    // Tie the sampling interval to the actual ray/box segment. Without this,
    // the old adaptive step count under-sampled oblique edge rays and never
    // reached the upper part of the pattern consistently.
    float3 exitNormal;
    float exitDistance = db_boxHit(rayOrigin, boxRay, DB_BOXDIMS, exitNormal, false);
    if (exitDistance <= 0.0f) return float4(background, 1.0f);
    float marchLength = exitDistance + 1.2f;

    // Navigation is applied before the procedural field, just like the other
    // DynamicBox shaders. The pattern transform is rigid in normal use.
    float3 ro = (uniforms.patternTransform * float4(rayOrigin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRay, 0.0f)));

    // Increase the procedural-domain frequency so the visible pattern is
    // approximately one third of its previous size.
    const float patternScale = 3.0f;

    // The source shader's synthetic camera covers a shorter interval than the
    // physical DynamicBox ray. Uniform samples remove the soft/empty edge
    // artifacts, while the extra 1.2 units preserve the visible overrun past
    // the nominal box depth.
    const int sampleCount = 112;
    float4 accumulated = float4(0.0f);
    float time = uniforms.time;

    for (int i = 0; i < sampleCount; i++) {
        float travel = marchLength * (float(i) + 0.5f) / float(sampleCount);
        float3 point = (ro + rd * travel) * patternScale;
        float radiusSquared;
        point = imFoldedPoint(point, time, radiusSquared);

        float glowDistance = abs(length(point) / 4000.0f) + 1e-4f;
        float phase = float(i) * 0.1f + travel * patternScale * 9.0f;
        float fieldWeight = 0.35f + 0.65f * min(radiusSquared, 1.0f);
        accumulated += exp(sin(phase + float4(0.0f, 2.0f, 3.0f, 0.0f)))
            * fieldWeight / sqrt(glowDistance);
    }

    float4 color = tanh(accumulated / 3200.0f);
    return float4(max(color.rgb * color.rgb, 0.0f), 1.0f);
}
