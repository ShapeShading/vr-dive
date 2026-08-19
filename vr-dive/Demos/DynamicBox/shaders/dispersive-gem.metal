// dispersive-gem.metal — spectral refractive gem
//
// Adapted from:
// https://fragcoord.xyz/s/sill88c6
//
// The source uses a synthetic camera and optional mouse rotation. DynamicBox
// supplies a real stereo ray; mouse rotation is replaced by the existing
// pattern navigation transform plus the source's time-driven golden-ratio
// rotation.

#define DG_DISPERSION 0.044f
#define DG_IOR_GEM 2.42f
#define DG_SPECTRAL_SAMPLES 2
#define DG_LIGHT_BOUNCES 16
#define DG_EXPOSURE 0.3f
#define DG_IOR_AIR 1.0f

#define DG_PHI 1.618033988749895f
#define DG_BASE_RATE 0.1f

constant float3 DG_GEM_PLANES[20] = {
    float3( 0.5773502692f,  0.5773502692f,  0.5773502692f),
    float3(-0.5773502692f,  0.5773502692f,  0.5773502692f),
    float3( 0.5773502692f, -0.5773502692f,  0.5773502692f),
    float3(-0.5773502692f, -0.5773502692f,  0.5773502692f),
    float3( 0.5773502692f,  0.5773502692f, -0.5773502692f),
    float3(-0.5773502692f,  0.5773502692f, -0.5773502692f),
    float3( 0.5773502692f, -0.5773502692f, -0.5773502692f),
    float3(-0.5773502692f, -0.5773502692f, -0.5773502692f),
    float3( 0.0f,  0.3568220898f,  0.9341723590f),
    float3( 0.0f, -0.3568220898f,  0.9341723590f),
    float3( 0.0f,  0.3568220898f, -0.9341723590f),
    float3( 0.0f, -0.3568220898f, -0.9341723590f),
    float3( 0.3568220898f,  0.9341723590f,  0.0f),
    float3(-0.3568220898f,  0.9341723590f,  0.0f),
    float3( 0.3568220898f, -0.9341723590f,  0.0f),
    float3(-0.3568220898f, -0.9341723590f,  0.0f),
    float3( 0.9341723590f,  0.0f,  0.3568220898f),
    float3(-0.9341723590f,  0.0f,  0.3568220898f),
    float3( 0.9341723590f,  0.0f, -0.3568220898f),
    float3(-0.9341723590f,  0.0f, -0.3568220898f)
};

constant float3 DG_XYZ_TO_RGB[3] = {
    float3( 8.09817f, -3.05248f,  0.18374f),
    float3(-3.84142f,  5.90964f, -0.67295f),
    float3(-1.24599f,  0.13074f,  3.48683f)
};

static float3 dgXYZToRGB(float3 xyz) {
    return float3(
        dot(DG_XYZ_TO_RGB[0], xyz),
        dot(DG_XYZ_TO_RGB[1], xyz),
        dot(DG_XYZ_TO_RGB[2], xyz));
}

static float3 dgAgxInset(float3 color) {
    return float3(
        dot(float3(0.8424790623f, 0.0423282423f, 0.0423756549f), color),
        dot(float3(0.0784336000f, 0.8784686365f, 0.0784336000f), color),
        dot(float3(0.0792237451f, 0.0791661275f, 0.8791429738f), color));
}

static float3 dgAgxOutset(float3 color) {
    return float3(
        dot(float3(1.196879006f, -0.052896852f, -0.052971636f), color),
        dot(float3(-0.098020881f, 1.151903130f, -0.098043451f), color),
        dot(float3(-0.099029745f, -0.098961175f, 1.151073673f), color));
}

static float3 dgAgxContrast(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4
        - 6.868f * x2 * x + 0.4298f * x2 + 0.1191f * x - 0.00232f;
}

static float3 dgTonemapAgx(float3 color) {
    const float minEV = -12.47393f;
    const float maxEV = 4.026069f;
    color = dgAgxInset(color);
    color = clamp(log2(max(color, float3(1e-10f))), minEV, maxEV);
    color = (color - minEV) / (maxEV - minEV);
    color = dgAgxContrast(color);
    return dgAgxOutset(color);
}

static float dgCieG(float wavelength, float mean, float left, float right) {
    float t = (wavelength - mean) * (wavelength < mean ? left : right);
    return exp(-0.5f * t * t);
}

static float3 dgWavelengthToXYZ(float wavelength) {
    float x = 1.056f * dgCieG(wavelength, 599.8f, 0.0264f, 0.0323f)
        + 0.362f * dgCieG(wavelength, 442.0f, 0.0624f, 0.0374f)
        - 0.065f * dgCieG(wavelength, 501.1f, 0.0490f, 0.0382f);
    float y = 0.821f * dgCieG(wavelength, 568.8f, 0.0213f, 0.0247f)
        + 0.286f * dgCieG(wavelength, 530.9f, 0.0613f, 0.0322f);
    float z = 1.217f * dgCieG(wavelength, 437.0f, 0.0845f, 0.0278f)
        + 0.681f * dgCieG(wavelength, 459.0f, 0.0385f, 0.0725f);
    return float3(x, y, z);
}

static float3x3 dgRotXZ(float angle) {
    float s = sin(angle), c = cos(angle);
    return float3x3(c, 0.0f, -s, 0.0f, 1.0f, 0.0f, s, 0.0f, c);
}

static float3x3 dgRotXY(float angle) {
    float s = sin(angle), c = cos(angle);
    return float3x3(c, -s, 0.0f, s, c, 0.0f, 0.0f, 0.0f, 1.0f);
}

static float3x3 dgRotYZ(float angle) {
    float s = sin(angle), c = cos(angle);
    return float3x3(1.0f, 0.0f, 0.0f, 0.0f, c, -s, 0.0f, s, c);
}

static float dgEnvironment(float3 direction) {
    return pow(max(-dot(direction, float3(0.0f, 0.0f, 1.0f)), 0.0f), 4.0f) * 3.0f;
}

static float dgFresnelDielectric(float cosI, float n1, float n2) {
    cosI = clamp(cosI, 0.0f, 1.0f);
    float eta = n1 / n2;
    float sinT2 = eta * eta * (1.0f - cosI * cosI);
    if (sinT2 >= 1.0f) return 1.0f;
    float cosT = sqrt(1.0f - sinT2);
    float rs = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT);
    float rp = (n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI);
    return 0.5f * (rs * rs + rp * rp);
}

static bool dgHitGem(float3 ro, float3 rd, thread float &tHit, thread float3 &nHit) {
    float tEnter = -1e30f;
    float tExit = 1e30f;
    float3 nEnter = float3(0.0f);

    for (int i = 0; i < 20; i++) {
        float3 normal = DG_GEM_PLANES[i];
        float denominator = dot(rd, normal);
        float numerator = 1.0f - dot(ro, normal);
        if (abs(denominator) < 1e-8f) {
            if (numerator < 0.0f) return false;
            continue;
        }
        float t = numerator / denominator;
        if (denominator < 0.0f) {
            if (t > tEnter) {
                tEnter = t;
                nEnter = normal;
            }
        } else {
            tExit = min(tExit, t);
        }
    }

    if (tEnter > tExit || tExit < 0.0f || tEnter < 0.0f) return false;
    tHit = tEnter;
    nHit = nEnter;
    return true;
}

static bool dgExitGem(float3 ro, float3 rd, thread float &tHit, thread float3 &nHit) {
    float tExit = 1e30f;
    float3 nExit = float3(0.0f);
    for (int i = 0; i < 20; i++) {
        float3 normal = DG_GEM_PLANES[i];
        float denominator = dot(rd, normal);
        if (denominator > 1e-8f) {
            float t = (1.0f - dot(ro, normal)) / denominator;
            if (t > 1e-4f && t < tExit) {
                tExit = t;
                nExit = normal;
            }
        }
    }
    tHit = tExit;
    nHit = nExit;
    return tExit < 1e29f;
}

static float dgTraceRefraction(
    float3 startPoint, float3 startDirection, float3 startNormal, float ior)
{
    float3 direction = refract(startDirection, startNormal, DG_IOR_AIR / ior);
    float3 point = startPoint;
    float accumulated = 0.0f;
    float throughput = 1.0f;

    for (int bounce = 0; bounce < DG_LIGHT_BOUNCES; bounce++) {
        float t;
        float3 normal;
        if (!dgExitGem(point, direction, t, normal)) break;
        point += direction * t;

        float3 orientedNormal = -normal;
        float3 refracted = refract(direction, orientedNormal, ior / DG_IOR_AIR);
        float cosI = abs(dot(direction, orientedNormal));
        float reflectance = dgFresnelDielectric(cosI, ior, DG_IOR_AIR);

        if (dot(refracted, refracted) > 1e-6f) {
            accumulated += throughput * (1.0f - reflectance)
                * dgEnvironment(refracted);
        }
        throughput *= reflectance;
        if (throughput < 0.04f) break;
        direction = reflect(direction, orientedNormal);
    }
    return accumulated;
}

static float3 dgDispersedRefraction(
    float3 hitPoint, float3 rayDirection, float3 surfaceNormal, float ditherValue)
{
    float3 reflectDirection = reflect(rayDirection, surfaceNormal);
    float reflectBrightness = dgEnvironment(reflectDirection);
    float cosI = max(dot(-rayDirection, surfaceNormal), 0.0f);
    float3 accumulatedXYZ = float3(0.0f);
    float3 whiteXYZ = float3(0.0f);

    for (int i = 0; i < DG_SPECTRAL_SAMPLES; i++) {
        float position = (float(i) + ditherValue) / float(DG_SPECTRAL_SAMPLES);
        float wavelength = mix(380.0f, 700.0f, position);
        float ior = DG_IOR_GEM + DG_DISPERSION * (0.5f - position);
        float3 xyz = dgWavelengthToXYZ(wavelength);

        float reflectance = dgFresnelDielectric(cosI, DG_IOR_AIR, ior);
        float refractBrightness = dgTraceRefraction(
            hitPoint, rayDirection, surfaceNormal, ior);
        float combined = reflectance * reflectBrightness
            + (1.0f - reflectance) * refractBrightness;
        accumulatedXYZ += combined * xyz;
        whiteXYZ += xyz;
    }

    return max(dgXYZToRGB(
        accumulatedXYZ / max(whiteXYZ, float3(1e-4f))), 0.0f);
}

static float dgDither(float3 point) {
    float2 cell = floor(point.xy * 97.0f + point.z * 13.0f);
    return fract(sin(dot(cell, float2(12.9898f, 78.233f))) * 43758.5453f);
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

    // Keep the gem comfortably inside the DynamicBox while retaining the
    // source's unit-distance plane construction.
    const float gemScale = 0.58f;
    float3 patternOrigin = (uniforms.patternTransform
        * float4(boxEye, 1.0f)).xyz / gemScale;
    float3 patternDirection = normalize(float3(uniforms.patternTransform
        * float4(boxRay, 0.0f)));

    float time = uniforms.time;
    float3x3 rotation = dgRotYZ(time * DG_BASE_RATE * DG_PHI * DG_PHI * DG_PHI)
        * dgRotXY(time * DG_BASE_RATE * DG_PHI * DG_PHI)
        * dgRotXZ(time * DG_BASE_RATE * DG_PHI);
    float3 rayOrigin = rotation * patternOrigin;
    float3 rayDirection = normalize(rotation * patternDirection);

    float t;
    float3 normal;
    if (!dgHitGem(rayOrigin, rayDirection, t, normal)) {
        return float4(0.0f, 0.0f, 0.002f, 1.0f);
    }

    float3 hitPoint = rayOrigin + rayDirection * t;
    float3 color = dgDispersedRefraction(
        hitPoint, rayDirection, normal, dgDither(rayOrigin));
    color *= DG_EXPOSURE;
    color = max(dgTonemapAgx(color), 0.0f);
    return float4(color, 1.0f);
}
