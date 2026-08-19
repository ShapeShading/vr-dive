// galaxy.metal — spiral galaxy with Worley/FBM cloud detail
//
// Adapted from:
// https://fragcoord.xyz/s/uoycfd38
//
// The source is a screen-space render2() pass added to u_pass1. DynamicBox
// has no runtime texture input, so the same Worley/FBM ingredients are used
// as detail inside an explicit 3D spiral-galaxy density field.

static float gfRand1(float2 co) {
    return fract(sin(dot(co, float2(12.9898f, 78.233f))) * 43758.5453f);
}

static float2 gfRand2(float2 p) {
    return fract(sin(float2(
        dot(p, float2(12.9898f, 78.233f)),
        dot(p, float2(54.321f, 12.345f)))) * 43758.5453f);
}

static float3 gfRand3(float2 p) {
    return fract(sin(float3(
        dot(p, float2(12.9898f, 78.233f)),
        dot(p, float2(39.3467f, 11.1359f)),
        dot(p, float2(73.5678f, 53.2234f)))) * 43758.5453f);
}

static float gfValueNoise(float2 p) {
    float2 cell = floor(p);
    float2 local = fract(p);
    local = local * local * (3.0f - 2.0f * local);

    float a = gfRand1(cell);
    float b = gfRand1(cell + float2(1.0f, 0.0f));
    float c = gfRand1(cell + float2(0.0f, 1.0f));
    float d = gfRand1(cell + float2(1.0f, 1.0f));
    return mix(mix(a, b, local.x), mix(c, d, local.x), local.y);
}

static float gfFBM(float2 p) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    for (int i = 0; i < 6; i++) {
        value += amplitude * gfValueNoise(p * frequency);
        frequency *= 2.0f;
        amplitude *= 0.5f;
    }
    return value;
}

// A softer Worley response than the source's bokeh exponent. It supplies
// compact cloud knots without reducing the whole galaxy to isolated dots.
static float gfWorley(float2 p, float2 shift, float seed) {
    p += shift;
    float2 cell = floor(p);
    float2 local = fract(p);
    float minimumDistance = 1.0f;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float2 neighbor = float2(float(x), float(y));
            float2 point = gfRand2(cell + neighbor + seed);
            float radius = min(gfRand1(point) * 0.9f + 0.1f, 0.3f);
            float2 difference = neighbor + point - local;
            float distanceValue = length(difference) * (radius / 1.1f);
            minimumDistance = min(minimumDistance, distanceValue);
        }
    }

    return pow(max(1.0f - minimumDistance, 0.0f), 10.0f);
}

static float3 gfGalaxyField(float3 point, float time) {
    // The disk is in XY and is viewed through its Z thickness.
    point.z += time * 0.025f;
    float radius = length(point.xy);
    float angle = atan2(point.y, point.x);

    // Three logarithmic spiral arms. The radial envelope keeps the center
    // bright while the arms remain clearly separated toward the rim.
    float spiralPhase = angle - time * 0.12f + log(radius + 0.075f) * 2.35f;
    float armWave = 0.5f + 0.5f * cos(spiralPhase * 3.0f);
    float arms = pow(clamp(armWave, 0.0f, 1.0f), 7.0f);

    float disk = exp(-abs(point.z) * 8.0f) * exp(-radius * 1.35f);
    float bulge = exp(-radius * radius * 10.0f) * exp(-abs(point.z) * 3.5f);

    float2 polarUV = float2(
        log(radius + 0.12f) * 2.2f,
        angle * 1.65f - time * 0.06f);
    float cloudNoise = gfFBM(polarUV * 1.15f + float2(time * 0.018f, 0.0f));
    float cloudCells = gfWorley(
        point.xy * 5.0f + float2(time * 0.04f, -time * 0.025f),
        float2(0.0f), 0.0f);

    float armDensity = disk * (0.08f + 1.85f * arms)
        * (0.34f + 0.82f * cloudNoise + 0.35f * cloudCells);
    float coreDensity = bulge * (0.9f + 0.35f * cloudNoise);

    // A second, softer arm layer gives the wispy structure visible between
    // the main arms instead of leaving a hard three-stripe pattern.
    float secondaryArms = pow(
        0.5f + 0.5f * cos(spiralPhase * 3.0f + 1.15f), 13.0f);
    float wispDensity = disk * secondaryArms * (0.12f + 0.35f * cloudNoise);

    float density = armDensity + coreDensity + wispDensity;
    float armHue = 0.56f + 0.10f * cloudNoise + 0.04f * sin(angle * 3.0f);
    float3 armColor = float3(0.18f, 0.38f, 1.0f)
        + float3(0.25f, 0.18f, 0.10f) * cloudNoise;
    float3 coreColor = float3(1.0f, 0.32f, 0.07f)
        * (0.75f + 0.25f * cloudNoise);

    float3 color = armColor * (armDensity + wispDensity)
        + coreColor * coreDensity * 1.45f;
    color += float3(0.04f, 0.12f, 0.45f) * density * armHue;
    return color;
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
    float3 background = float3(0.001f, 0.002f, 0.008f);

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

    float3 ro = (uniforms.patternTransform * float4(rayOrigin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRay, 0.0f)));

    // Frequency > 1 makes the galaxy smaller than the full DynamicBox, so
    // the complete disk and its three arms remain visible inside the portal.
    const float galaxyScale = 2.35f;
    float marchLength = (exitDistance + 0.65f) * galaxyScale;
    float3 patternOrigin = ro * galaxyScale;

    float3 accumulated = float3(0.0f);
    const int samples = 64;
    for (int i = 0; i < samples; i++) {
        float travel = marchLength * (float(i) + 0.5f) / float(samples);
        float3 point = patternOrigin + rd * travel;
        accumulated += gfGalaxyField(point, uniforms.time);
    }

    float3 color = accumulated / float(samples);
    color = 1.0f - exp(-color * 2.6f);
    color = pow(max(color, 0.0f), float3(0.72f));
    return float4(color + background, 1.0f);
}
