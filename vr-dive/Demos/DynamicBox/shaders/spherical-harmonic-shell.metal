// spherical-harmonic-shell.metal — animated real spherical-harmonic shell
//
// This original DynamicBox demo combines several low-order real spherical
// harmonic modes into the radius of a thin shell. Unlike the existing fractal,
// minimal-surface, particle, and primitive-composition demos, its structure is
// driven entirely by angular eigenmodes on the sphere.
//
// The field is not an exact Euclidean SDF because the radius varies with angle,
// so the marcher uses a conservative scale and remains bounded by the actual
// DynamicBox entry/exit interval.

static float2 shsRotate(float2 value, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float2(c * value.x - s * value.y,
                  s * value.x + c * value.y);
}

static float3 shsObjectPoint(float3 point, float time) {
    float3 q = point;
    q.xz = shsRotate(q.xz, time * 0.10f);
    q.xy = shsRotate(q.xy, 0.24f * sin(time * 0.13f));
    return q;
}

// Real spherical-harmonic-style polynomial modes. Normalization constants are
// folded into the weights because only their relative shape matters here.
static float shsAngularSignal(float3 direction, float time) {
    float x = direction.x;
    float y = direction.y;
    float z = direction.z;
    float y2 = y * y;

    float h32 = 5.2f * x * y * z;
    float h42 = 0.72f * (x * x - z * z) * (7.0f * y2 - 1.0f);
    float h50 = 0.125f * y * (63.0f * y2 * y2 - 70.0f * y2 + 15.0f);
    float h44 = 3.2f * x * z * (x * x - z * z);

    float blend = 0.5f + 0.5f * sin(time * 0.19f);
    float familyA = 0.68f * h32 + 0.32f * h50;
    float familyB = 0.58f * h42 + 0.42f * h44;
    return mix(familyA, familyB, blend);
}

static float shsSurfaceField(float3 point, float time) {
    float3 q = shsObjectPoint(point, time);
    float radialDistance = max(length(q), 1e-5f);
    float3 direction = q / radialDistance;
    float signal = shsAngularSignal(direction, time);
    float fineMode = 4.0f * direction.x * direction.z
        * (direction.x * direction.x - direction.z * direction.z);
    float radius = 0.56f + 0.125f * signal + 0.022f * fineMode;
    return radialDistance - radius;
}

static float shsDistance(float3 point, float time) {
    // Convert the radial zero-set into a thin shell. The 0.48 factor keeps
    // steps conservative under the angular deformation.
    return (abs(shsSurfaceField(point, time)) - 0.018f) * 0.48f;
}

static float3 shsNormal(float3 point, float time) {
    float2 e = float2(0.0015f, 0.0f);
    return normalize(float3(
        shsDistance(point + e.xyy, time) - shsDistance(point - e.xyy, time),
        shsDistance(point + e.yxy, time) - shsDistance(point - e.yxy, time),
        shsDistance(point + e.yyx, time) - shsDistance(point - e.yyx, time)
    ));
}

static float3 shsPalette(float phase) {
    return 0.50f + 0.50f * cos(
        6.2831853f * (phase + float3(0.02f, 0.34f, 0.67f))
    );
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut          in          [[stage_in]],
    constant DynamicBoxUniforms &uniforms    [[buffer(0)]],
    constant float4x4           *v2wMats     [[buffer(1)]],
    constant float4x4           *vpMatrices  [[buffer(2)]])
{
    uint viewIndex = min(in.viewIndex, max(uniforms.viewCount, 1u) - 1u);
    float3 cameraWorld = v2wMats[viewIndex][3].xyz;
    float3 boxEye = (cameraWorld - uniforms.objectCenter.xyz) / uniforms.boxScale;
    float3 boxRay = normalize(in.worldPos - cameraWorld);
    float3 background = float3(0.002f, 0.003f, 0.012f);

    float3 boxOrigin = boxEye;
    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    if (!insideBox) {
        float3 entryNormal;
        float entryDistance = db_boxHit(
            boxEye, boxRay, DB_BOXDIMS, entryNormal, true
        );
        if (entryDistance < 0.0f) return float4(background, 1.0f);
        boxOrigin = boxEye + boxRay * (entryDistance + 1e-3f);
    }

    float3 exitNormal;
    float exitDistance = db_boxHit(
        boxOrigin, boxRay, DB_BOXDIMS, exitNormal, false
    );
    if (exitDistance <= 0.0f) return float4(background, 1.0f);

    float3 rayOrigin = (
        uniforms.patternTransform * float4(boxOrigin, 1.0f)
    ).xyz;
    float3 rayDirection = normalize(float3(
        uniforms.patternTransform * float4(boxRay, 0.0f)
    ));

    float time = uniforms.time;
    float travel = 0.0f;
    float closestDistance = 1.0f;

    for (int step = 0; step < 112; ++step) {
        float3 point = rayOrigin + rayDirection * travel;
        float distance = shsDistance(point, time);
        closestDistance = min(closestDistance, abs(distance));

        float epsilon = 0.0008f + travel * 0.00012f;
        if (distance < epsilon) {
            float3 normal = shsNormal(point, time);
            float3 lightDirection = normalize(float3(-0.42f, 0.82f, 0.38f));
            float diffuse = max(dot(normal, lightDirection), 0.0f);
            float rim = pow(
                1.0f - max(dot(normal, -rayDirection), 0.0f), 2.8f
            );
            float specular = pow(
                max(dot(reflect(-lightDirection, normal), -rayDirection), 0.0f),
                42.0f
            );

            float3 objectPoint = shsObjectPoint(point, time);
            float3 direction = normalize(objectPoint);
            float signal = shsAngularSignal(direction, time);
            float phase = 0.50f + 0.22f * signal
                + 0.08f * dot(direction, float3(0.7f, 1.1f, 1.5f));
            float3 baseColor = shsPalette(phase);
            float3 color = baseColor * (0.16f + 1.05f * diffuse);
            color += rim * mix(float3(0.10f, 0.35f, 0.95f), baseColor, 0.35f);
            color += specular * float3(1.0f, 0.82f, 0.58f);
            color *= exp(-travel * 0.12f);
            return float4(color, 1.0f);
        }

        travel += clamp(max(distance, epsilon) * 0.72f, 0.0007f, 0.045f);
        if (travel > exitDistance + 0.01f) break;
    }

    float glow = exp(-closestDistance * 85.0f) * 0.06f;
    return float4(background + glow * float3(0.15f, 0.35f, 0.85f), 1.0f);
}
