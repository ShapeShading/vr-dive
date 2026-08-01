// newton-basins-3d.metal - Coupled 3D Newton Basin Shell
//
// A genuine Newton iteration in R3, rather than a 2D Newton image extruded
// along an axis. For the coupled polynomial system F:R3->R3, each sample uses
// z <- z - J_F(z)^-1 F(z). Its multiple root attractors create 3D basins.
// Basin boundaries themselves are discontinuous classifications and cannot be
// sphere-traced as a signed distance field. This shader instead renders a thin,
// conservative shell whose radius is gently displaced by near-singular Newton
// orbits; root attractors supply the basin color.
//
// Cost: 7 Newton steps per field evaluation, 76 march steps, and six normal
// samples only at a hit. The shell step is intentionally conservative to keep
// the basin-derived ridges stable under stereo ray marching.

struct NewtonOrbit {
    float3 root;
    float residual;
    float singularity;
};

static float3 rotateY(float3 p, float angle) {
    float s = sin(angle), c = cos(angle);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

// F is symmetric but coupled across all three coordinates, so its attraction
// basins are volumetric rather than a planar Newton pattern copied through z.
static float3 newtonFunction(float3 p, float a) {
    return float3(
        p.x * p.x - 0.78f * p.y * p.z - a,
        p.y * p.y - 0.78f * p.z * p.x - a * 0.91f,
        p.z * p.z - 0.78f * p.x * p.y - a * 1.09f
    );
}

static NewtonOrbit evaluateNewtonOrbit(float3 initial, float time) {
    float a = 0.34f + 0.035f * sin(time * 0.08f);
    float3 z = initial;
    float minDeterminant = 1e6f;
    float residual = 1e6f;

    for (int i = 0; i < 7; i++) {
        float3 f = newtonFunction(z, a);
        residual = length(f);
        if (residual < 0.0003f) break;

        float3 row0 = float3(2.0f * z.x, -0.78f * z.z, -0.78f * z.y);
        float3 row1 = float3(-0.78f * z.z, 2.0f * z.y, -0.78f * z.x);
        float3 row2 = float3(-0.78f * z.y, -0.78f * z.x, 2.0f * z.z);
        float3 cross12 = cross(row1, row2);
        float determinant = dot(row0, cross12);
        minDeterminant = min(minDeterminant, abs(determinant));

        if (abs(determinant) < 0.0001f) {
            z += float3(0.017f, -0.011f, 0.013f);
            continue;
        }

        float3 step = float3(
            dot(f, cross12),
            dot(f, cross(row2, row0)),
            dot(f, cross(row0, row1))
        ) / determinant;
        z -= step * 0.82f;
        if (dot(z, z) > 64.0f) break;
    }

    NewtonOrbit orbit;
    orbit.root = clamp(z, -3.0f, 3.0f);
    orbit.residual = min(residual, 20.0f);
    orbit.singularity = clamp(-log(max(minDeterminant, 1e-5f)) * 0.13f, 0.0f, 1.0f);
    return orbit;
}

static float3 palette(float t) {
    return float3(0.32f, 0.36f, 0.42f)
        + float3(0.50f, 0.45f, 0.38f)
        * cos(6.28318f * (float3(0.71f, 0.93f, 1.19f) * t
            + float3(0.07f, 0.35f, 0.61f)));
}

static float newtonShellMap(float3 p, float time) {
    NewtonOrbit orbit = evaluateNewtonOrbit(p * 1.28f, time);
    float rootPhase = dot(orbit.root, float3(0.71f, 1.13f, 1.67f));
    float ripple = sin(rootPhase * 2.7f + dot(p, float3(3.1f, -2.3f, 2.7f)));
    float radius = 0.68f + orbit.singularity * 0.13f + ripple * 0.018f;
    return abs(length(p) - radius) - 0.021f;
}

static float3 calcNormal(float3 p, float time) {
    float2 e = float2(0.0015f, 0.0f);
    return normalize(float3(
        newtonShellMap(p + e.xyy, time) - newtonShellMap(p - e.xyy, time),
        newtonShellMap(p + e.yxy, time) - newtonShellMap(p - e.yxy, time),
        newtonShellMap(p + e.yyx, time) - newtonShellMap(p - e.yyx, time)
    ));
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
    float3 background = float3(0.003f, 0.006f, 0.016f);
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
    rayOrigin = rotateY(rayOrigin, time * 0.045f);
    rayDirection = rotateY(rayDirection, time * 0.045f);

    float travel = 0.0f;
    for (int i = 0; i < 76; i++) {
        float3 point = rayOrigin + rayDirection * travel;
        float distance = newtonShellMap(point, time);
        if (distance < 0.002f) {
            NewtonOrbit orbit = evaluateNewtonOrbit(point * 1.28f, time);
            float3 normal = calcNormal(point, time);
            float3 light = normalize(float3(-0.38f, 0.86f, 0.42f));
            float diffuse = max(dot(normal, light), 0.0f);
            float rim = pow(1.0f - max(dot(-rayDirection, normal), 0.0f), 3.0f);
            float basin = dot(orbit.root, float3(0.19f, 0.37f, 0.61f));
            float3 color = palette(basin + orbit.singularity * 0.31f)
                * (0.20f + diffuse * 1.05f);
            color += float3(0.22f, 0.72f, 1.0f) * rim * 0.42f;
            color += float3(1.0f, 0.42f, 0.16f) * orbit.singularity * 0.18f;
            return float4(color * exp(-travel * 0.21f), 1.0f);
        }
        travel += max(distance * 0.48f, 0.003f);
        if (travel > 25.0f) break;
    }

    return float4(background, 1.0f);
}