// mandelbox-fold.metal - 3D Mandelbox Fold Fractal
//
// A ray-marchable extension of planar iterative fractal ideas. A direct 2D
// Newton basin is not a suitable surface SDF: its root boundaries are
// discontinuous and do not provide conservative sphere-tracing steps. This
// Mandelbox instead extends repeated planar folding into three coordinates,
// combining box folds and spherical inversions into a stable distance estimate.
// The orbit trap coloring still gives neighboring attractor regions distinct
// colors, echoing the visual language of Newton basins without sacrificing
// reliable three-dimensional geometry.
//
// Design: 12 box/sphere-fold iterations, 88 ray steps, and six normal samples.
// This is intentionally a new fold/inversion family, distinct from the existing
// Menger removal rule and the periodic Apollonian sphere packing.
// The complete fractal, including its central cavity, is enlarged 4x in the
// box. Three extra fold iterations restore fine self-similar structure that
// would otherwise be lost when viewing the enlarged interior.

constant float MANDELBOX_VOLUME_SCALE = 4.0f;

static float3 rotateY(float3 p, float angle) {
    float s = sin(angle), c = cos(angle);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static float mandelboxDE(float3 p, float time, thread float3 &trap) {
    float3 z = p;
    float derivative = 1.0f;
    float scale = -1.72f + 0.06f * sin(time * 0.10f);
    trap = float3(1e6f);

    for (int i = 0; i < 12; i++) {
        z = clamp(z, -1.0f, 1.0f) * 2.0f - z;
        float radiusSquared = dot(z, z);
        float sphereScale = clamp(1.0f / max(radiusSquared, 0.25f), 1.0f, 4.0f);
        z *= sphereScale;
        derivative *= sphereScale;
        trap = min(trap, abs(z));
        z = z * scale + p;
        derivative = derivative * abs(scale) + 1.0f;
    }

    return length(z) / max(abs(derivative), 1e-5f);
}

static float mandelboxMap(float3 p, float time) {
    float3 unusedTrap;
    return mandelboxDE(p / MANDELBOX_VOLUME_SCALE, time, unusedTrap)
        * MANDELBOX_VOLUME_SCALE;
}

static float3 calcNormal(float3 p, float time) {
    float2 e = float2(0.0015f, 0.0f);
    return normalize(float3(
        mandelboxMap(p + e.xyy, time) - mandelboxMap(p - e.xyy, time),
        mandelboxMap(p + e.yxy, time) - mandelboxMap(p - e.yxy, time),
        mandelboxMap(p + e.yyx, time) - mandelboxMap(p - e.yyx, time)
    ));
}

static float3 palette(float t) {
    return float3(0.36f, 0.39f, 0.45f)
        + float3(0.52f, 0.46f, 0.38f)
        * cos(6.28318f * (float3(0.67f, 0.82f, 1.08f) * t
            + float3(0.08f, 0.31f, 0.57f)));
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut in [[stage_in]],
    constant DynamicBoxUniforms &uniforms [[buffer(0)]],
    constant float4x4 *v2wMats [[buffer(1)]],
    constant float4x4 *vpMatrices [[buffer(2)]])
{
    uint viewIndex = min(in.viewIndex, uniforms.viewCount - 1u);
    float3 cameraWorld = float3(v2wMats[viewIndex][3].x, v2wMats[viewIndex][3].y, v2wMats[viewIndex][3].z);
    float3 boxEye = (cameraWorld - uniforms.objectCenter.xyz) / uniforms.boxScale;
    float3 boxRay = normalize(in.worldPos - cameraWorld);
    float3 background = float3(0.004f, 0.006f, 0.018f);
    float3 origin;

    if (!all(abs(boxEye) < (DB_BOXDIMS - 1e-3f))) {
        float3 entryNormal;
        float entry = db_boxHit(boxEye, boxRay, DB_BOXDIMS, entryNormal, true);
        if (entry < 0.0f) return float4(background, 1.0f);
        origin = boxEye + boxRay * (entry + 1e-3f);
    } else {
        origin = boxEye;
    }

    float3 exitNormal;
    if (db_boxHit(origin, boxRay, DB_BOXDIMS, exitNormal, false) <= 0.0f) discard_fragment();

    float time = uniforms.time;
    float3 rayOrigin = (uniforms.patternTransform * float4(origin, 1.0f)).xyz;
    float3 rayDirection = normalize(float3(uniforms.patternTransform * float4(boxRay, 0.0f)));
    rayOrigin = rotateY(rayOrigin, time * 0.055f);
    rayDirection = rotateY(rayDirection, time * 0.055f);

    float distanceAlongRay = 0.0f;
    for (int i = 0; i < 88; i++) {
        float3 point = rayOrigin + rayDirection * distanceAlongRay;
        float3 trap;
        float distance = mandelboxDE(point / MANDELBOX_VOLUME_SCALE, time, trap)
            * MANDELBOX_VOLUME_SCALE;
        if (distance < 0.0016f) {
            float3 normal = calcNormal(point, time);
            float3 light = normalize(float3(-0.42f, 0.84f, 0.36f));
            float diffuse = max(dot(normal, light), 0.0f);
            float rim = pow(1.0f - max(dot(-rayDirection, normal), 0.0f), 3.0f);
            float attractor = dot(trap, float3(0.55f, 0.83f, 1.17f));
            float3 color = palette(attractor * 1.8f + length(point) * 0.24f)
                * (0.17f + diffuse * 1.05f);
            color += float3(0.20f, 0.62f, 1.0f) * rim * 0.55f;
            return float4(color * exp(-distanceAlongRay * 0.22f), 1.0f);
        }
        distanceAlongRay += max(distance * 0.82f, 0.0015f);
        if (distanceAlongRay > 25.0f) break;
    }

    return float4(background, 1.0f);
}