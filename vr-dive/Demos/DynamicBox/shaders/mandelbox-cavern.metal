// mandelbox-cavern.metal — 镂空 Mandelbox 洞穴
//
// 用于观察内部的 Mandelbox 变体。保留 mandelbox-fold 的 12 步盒/球折叠
// 距离估计，然后通过 max(fractalDE, cavitySDF) 减去一个球形腔体。
// 该腔体是真实的 CSG 几何体，而非放大的相机空间空洞，因此其边缘
// 会暴露递归折叠结构，并在中心保留一个清晰的、可导航的负空间。
//
// 代价：12 次 DE 迭代、84 步行进、中心差分法线。

constant float CAVERN_SCALE = 4.0f;

static float3 rotateY(float3 p, float angle) {
    float s = sin(angle), c = cos(angle);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static float cavernFractalDE(float3 p, float time, thread float3 &trap) {
    float3 z = p;
    float derivative = 1.0f;
    float scale = -1.69f + 0.045f * sin(time * 0.09f);
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

static float cavernMap(float3 p, float time) {
    float3 unusedTrap;
    float fractal = cavernFractalDE(p / CAVERN_SCALE, time, unusedTrap) * CAVERN_SCALE;
    float3 chamberCenter = float3(0.05f * sin(time * 0.07f), -0.04f, 0.03f * cos(time * 0.06f));
    float cavity = 0.48f - length(p - chamberCenter);
    return max(fractal, cavity);
}

static float3 calcNormal(float3 p, float time) {
    float2 e = float2(0.0015f, 0.0f);
    return normalize(float3(
        cavernMap(p + e.xyy, time) - cavernMap(p - e.xyy, time),
        cavernMap(p + e.yxy, time) - cavernMap(p - e.yxy, time),
        cavernMap(p + e.yyx, time) - cavernMap(p - e.yyx, time)
    ));
}

static float3 palette(float t) {
    return float3(0.34f, 0.31f, 0.29f)
        + float3(0.52f, 0.44f, 0.35f)
        * cos(6.28318f * (float3(0.72f, 0.94f, 1.11f) * t
            + float3(0.03f, 0.24f, 0.49f)));
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
    float3 background = float3(0.006f, 0.003f, 0.010f);
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
    float3 rayOrigin = rotateY((uniforms.patternTransform * float4(origin, 1.0f)).xyz, time * 0.04f);
    float3 rayDirection = rotateY(normalize(float3(uniforms.patternTransform * float4(boxRay, 0.0f))), time * 0.04f);
    float travel = 0.0f;
    for (int i = 0; i < 84; i++) {
        float3 point = rayOrigin + rayDirection * travel;
        float distance = cavernMap(point, time);
        if (distance < 0.0017f) {
            float3 trap;
            cavernFractalDE(point / CAVERN_SCALE, time, trap);
            float3 normal = calcNormal(point, time);
            float3 light = normalize(float3(-0.40f, 0.78f, 0.48f));
            float diffuse = max(dot(normal, light), 0.0f);
            float rim = pow(1.0f - max(dot(-rayDirection, normal), 0.0f), 2.7f);
            float3 color = palette(dot(trap, float3(0.63f, 0.91f, 1.23f)) * 1.7f)
                * (0.19f + diffuse * 1.08f);
            color += float3(1.0f, 0.25f, 0.08f) * rim * 0.42f;
            return float4(color * exp(-travel * 0.20f), 1.0f);
        }
        travel += max(distance * 0.76f, 0.0015f);
        if (travel > 25.0f) break;
    }
    return float4(background, 1.0f);
}