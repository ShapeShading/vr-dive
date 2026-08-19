// holofoil-dice.metal — holographic foil dice
//
// Adapted from:
// https://fragcoord.xyz/s/hy0rz4rv
//
// Original author: Jaenam. License: CC BY-NC-SA 4.0.
// https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// The compact GLSL macro A() is expanded into a Metal helper. The original
// uses three passes with Z = -1, 0, and 1 to build separate RGB channels.

static float2 hfCosineMatrix(float2 value, float angle) {
    // GLSL `v *= mat2(...)` is row-vector multiplication. The explicit form
    // avoids Metal's column-major matrix convention changing the rotation.
    float c0 = cos(angle);
    float c11 = cos(angle + 11.0f);
    float c33 = cos(angle + 33.0f);
    return float2(value.x * c0 + value.y * c33,
                  value.x * c11 + value.y * c0);
}

static float hfHash3(float3 p) {
    return fract(sin(dot(p, float3(127.1f, 311.7f, 74.7f))) * 43758.5f);
}

static float hfHashAngle(float3 p) {
    return fract(sin(dot(p, float3(43.7f, 78.2f, 123.4f))) * 127.1f)
        * 6.2831853f;
}

static float hfChannel(
    float3 patternOrigin,
    float3 patternDirection,
    float time,
    float channelOffset)
{
    float accumulated = 0.0f;
    float travel = 0.0f;
    float angle = time * 0.5f;

    for (int i = 1; i <= 80; i++) {
        // Original: p = vec3((I + I - r.xy) / r.y * d, d - 8.)
        // DynamicBox supplies the equivalent stereo ray, with patternOrigin
        // already positioned at the box entry and scaled into dice space.
        float3 p = patternOrigin + patternDirection * travel;
        p.xz = hfCosineMatrix(p.xz, angle);
        p.xy = hfCosineMatrix(p.xy, angle);

        float3 cell = floor(p * 6.0f);
        float3 local = fract(p * 6.0f) - 0.5f;

        float pointRadius = hfHash3(cell) * 0.3f + 0.1f;
        float highlight = step(length(local), pointRadius);
        float sparkleAngle = hfHashAngle(cell);

        float edge = 1.0f;
        float subdivision = 2.0f;
        for (int j = 0; j < 3; j++) {
            // GLSL mod(p * sc, 2.) for a possibly negative p is expressed
            // with fract so it retains GLSL's non-negative remainder.
            float3 folded = abs(fract(p * subdivision * 0.5f) * 2.0f - 1.0f);
            edge = min(edge,
                min(max(folded.x, folded.y),
                    min(max(folded.y, folded.z), max(folded.x, folded.z)))
                / subdivision);
            subdivision *= 0.6f;
        }

        // Rounded cube / die boundary with a three-dimensional diagonal
        // support term, matching dot(abs(p), vec3(.577)) * .9.
        float cubeDistance = max(
            max(max(abs(p.x), abs(p.y)), abs(p.z)),
            dot(abs(p), float3(0.577f)) * 0.9f) - 3.0f;

        float field = max(max(cubeDistance, edge - 0.1f), abs(sin(cubeDistance)) - 0.3f)
            + channelOffset * 0.02f - float(i) / 130.0f;
        float stepDistance = 0.01f + 0.15f * abs(field);
        travel += stepDistance;

        // GLSL smoothstep(.02, .01, s), expanded to avoid relying on the
        // undefined edge0 > edge1 behavior of some Metal implementations.
        float sparkle = 1.0f - smoothstep(0.01f, 0.02f, stepDistance);
        float pulse = 0.5f + 0.5f * sin(float(i) * 0.3f + channelOffset * 5.0f);
        float foil = sparkle * 4.0f * highlight
            * sin(sparkleAngle + float(i) * 0.4f + channelOffset * 5.0f);
        accumulated += 1.6f / stepDistance * (pulse + foil);
    }

    return accumulated;
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
    float3 background = float3(0.0f);

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

    float3 transformedOrigin = (uniforms.patternTransform
        * float4(rayOrigin, 1.0f)).xyz;
    float3 transformedDirection = normalize(float3(uniforms.patternTransform
        * float4(boxRay, 0.0f)));

    // The source dice occupies roughly a radius-3 domain. Map that domain to
    // the DynamicBox while keeping the ray direction in normalized units.
    const float diceScale = 5.5f;
    float3 patternOrigin = transformedOrigin * diceScale;
    float3 patternDirection = transformedDirection;

    float3 raw = float3(
        hfChannel(patternOrigin, patternDirection, uniforms.time, -1.0f),
        hfChannel(patternOrigin, patternDirection, uniforms.time,  0.0f),
        hfChannel(patternOrigin, patternDirection, uniforms.time,  1.0f));

    float3 color = tanh(raw * raw / 1e7f);
    return float4(max(color, 0.0f), 1.0f);
}
