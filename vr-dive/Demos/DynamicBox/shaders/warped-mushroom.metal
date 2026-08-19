// warped-mushroom.metal — warped sphere mushroom
//
// Adapted from:
// https://fragcoord.xyz/s/muke6asz
//
// The source is a compact screen-space ray marcher. This version uses the
// DynamicBox stereo ray and expands its two warped-sphere distances into a
// stable bounded distance field for the stalk and cap.

static float2 wmRotate(float2 value, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float2(c * value.x - s * value.y,
                  s * value.x + c * value.y);
}

static float wmDistance(float3 point, float time, thread float3 &rotatedPoint) {
    float3 q = point;
    q.yz = wmRotate(q.yz, time * 0.2f);
    rotatedPoint = q;

    // Warped sphere distances from the original: one for the stalk and one
    // for the cap. The -1 turns the source's unit-radius march into an SDF.
    float stalkHeight = 0.9f + 0.2f * q.y / (q.y + 2.0f);
    float stalk = length(float3(stalkHeight, q.xz)) - 1.0f;

    float capHeight = q.y - 2.5f + length(q) + q.x * q.y * 0.1f;
    float cap = length(float3(capHeight, q.xz * 0.5f)) - 1.0f;

    return min(stalk, cap);
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
    float3 background = float3(0.001f, 0.003f, 0.012f);

    // Start at the actual DynamicBox entry point. Using boxEye directly would
    // spend the fixed march budget outside the box when the camera is outside.
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
    if (exitDistance <= 0.0f) return float4(background, 1.0f);

    // Keep the warped object inside the DynamicBox while preserving its
    // original proportions and camera-facing raymarch behavior.
    const float mushroomScale = 0.24f;
    float3 ro = (uniforms.patternTransform * float4(rayOrigin, 1.0f)).xyz
        / mushroomScale;
    float3 rd = normalize(float3(uniforms.patternTransform
        * float4(boxRay, 0.0f)));

    float travel = 0.0f;
    float maxTravel = (exitDistance + 0.6f) / mushroomScale;
    float time = uniforms.time;

    for (int i = 0; i < 140; i++) {
        float3 point = ro + rd * travel;
        float3 q;
        float distance = wmDistance(point, time, q);

        if (distance < 0.0025f) {
            // The original color is a blue-white glow modulated by repeated
            // spots and a soft back-light term.
            float3 spots = fract(q / 0.3f) * 3.0f - 0.9f;
            float spotLength = max(length(spots), 0.7f);
            float bodyGlow = max((3.0f - length(point)) / spotLength * 0.1f,
                                 0.1f);
            float3 color = float3(1.0f, 3.0f, 8.0f) * bodyGlow;

            // Approximate a normal with tetrahedral finite differences so the
            // cap and stalk receive a readable blue rim light.
            float e = 0.0015f;
            float3 nx;
            float dx = wmDistance(point + float3(e, 0, 0), time, nx)
                     - wmDistance(point - float3(e, 0, 0), time, nx);
            float dy = wmDistance(point + float3(0, e, 0), time, nx)
                     - wmDistance(point - float3(0, e, 0), time, nx);
            float dz = wmDistance(point + float3(0, 0, e), time, nx)
                     - wmDistance(point - float3(0, 0, e), time, nx);
            float3 normal = normalize(float3(dx, dy, dz));
            float rim = pow(1.0f - max(dot(-rd, normal), 0.0f), 2.0f);
            color += float3(0.08f, 0.35f, 1.0f) * rim * 0.45f;

            return float4(color * exp(-travel * 0.12f), 1.0f);
        }

        travel += max(distance * 0.72f, 0.003f);
        if (travel > maxTravel) break;
    }

    return float4(background, 1.0f);
}
