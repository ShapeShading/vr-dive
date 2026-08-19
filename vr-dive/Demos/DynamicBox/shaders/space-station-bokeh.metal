// space-station-bokeh.metal — single-pass Apollonian bokeh
//
// Adapted from the Bokeh pass at:
// https://fragcoord.xyz/s/spajnznt
//
// The source samples a u_pass1 texture containing the Space Station fractal.
// DynamicBox runtime shaders do not have a texture input, so this version
// reconstructs an Apollonian fractal volume procedurally and applies the same
// golden-angle bokeh accumulation to that field.

static float3 sbPalette(float t) {
    return float3(0.35f, 0.55f, 0.85f)
        + float3(0.55f, 0.35f, 0.25f)
        * cos(6.28318f * (t + float3(0.0f, 0.17f, 0.33f)));
}

// Procedural replacement for the source shader's u_pass1 image. This is a
// bounded Apollonian-style inversion field with a soft shell so the Bokeh
// samples reproduce a continuous fractal instead of isolated star points.
static float4 sbFractalPass(float3 point, float time) {
    point.z += time * 0.08f;

    float angle = time * 0.12f;
    float s = sin(angle), c = cos(angle);
    point.xz = float2(c * point.x - s * point.z,
                      s * point.x + c * point.z);

    float3 q = point * 1.35f;
    float scale = 1.0f;
    float orbit = 1e4f;
    for (int i = 0; i < 8; i++) {
        q = -1.0f + 2.0f * fract(q * 0.5f + 0.5f);
        float radiusSquared = max(dot(q, q), 1e-5f);
        float inversion = max(1.0f / radiusSquared, 1.0f);
        q *= inversion;
        scale *= inversion;
        orbit = min(orbit, min(min(abs(q.x), abs(q.y)), abs(q.z)));
    }

    float distanceEstimate = length(q) / max(scale, 1e-5f) - 0.018f;
    float shell = exp(-abs(distanceEstimate) * 72.0f);
    float innerGlow = exp(-max(distanceEstimate, 0.0f) * 13.0f) * 0.16f;

    // A broad ring gives the fractal the "station" silhouette without
    // replacing the actual fractal field with sparse point lights.
    float radial = length(point.xy);
    float ringDistance = length(float2(radial - 0.58f, point.z * 1.4f)) - 0.035f;
    float ringGlow = exp(-abs(ringDistance) * 45.0f) * 0.18f;

    float brightness = shell + innerGlow + ringGlow;
    float hue = fract(orbit * 2.1f + time * 0.025f);
    float3 color = sbPalette(hue) * brightness;
    color += float3(0.12f, 0.38f, 1.0f) * shell * 0.32f;
    return float4(color, brightness);
}

// GLSL's `u *= mat2(...)` multiplies a row vector by a column-major matrix.
// Spell the multiplication out so Metal's matrix/vector convention cannot
// silently transpose the golden-angle transform.
static float2 sbGoldenAngle(float2 value) {
    float m00 = -0.737f;
    float m01 =  0.061f - 0.737f;
    float m10 =  1.413f - 0.737f;
    float m11 = -0.737f;
    return float2(value.x * m00 + value.y * m01,
                  value.x * m10 + value.y * m11);
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
    float marchLength = exitDistance + 0.55f;

    float3 ro = (uniforms.patternTransform * float4(rayOrigin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRay, 0.0f)));
    float3 tangent = abs(rd.y) < 0.9f ? float3(0.0f, 1.0f, 0.0f)
                                     : float3(1.0f, 0.0f, 0.0f);
    float3 right = normalize(cross(rd, tangent));
    float3 up = normalize(cross(right, rd));

    // Equivalent to the source's (I + I - resolution) / 2e3, expressed in
    // the local transverse coordinates of the actual stereo ray.
    float2 u = float2(dot(ro, right), dot(ro, up)) * 0.035f;
    float sampleScale = 1.0f;
    float3 accumulated = float3(0.0f);

    const int bokehSamples = 16;
    const int depthSamples = 14;
    for (int sample = 0; sample < bokehSamples; sample++) {
        u = sbGoldenAngle(u);
        float2 offset = u * sampleScale * 1.6f;

        for (int depth = 0; depth < depthSamples; depth++) {
            float depthT = marchLength * (float(depth) + 0.5f)
                / float(depthSamples);
            float3 samplePoint = ro + rd * depthT
                + right * offset.x + up * offset.y;
            float4 field = sbFractalPass(samplePoint, uniforms.time);
            accumulated += field.rgb;
        }

        // GLSL `i += 1.0 / i`, with i initialized to 1.
        sampleScale += 1.0f / sampleScale;
    }

    float3 color = sqrt(max(accumulated
        / float(bokehSamples * depthSamples), 0.0f)) * 1.35f;
    return float4(color + background, 1.0f);
}
