// mandelbox-fast.metal — 轻量 Mandelbox
//
// 面向性能优化的 Mandelbox，属于同一盒/球折叠视觉家族。使用 8 次迭代
// 替代 12 次，60 步光线预算，略大的命中容差和较粗糙的法线 epsilon。
// 结构仍然保持可辨识的递归特征，同时大幅减少了最坏情况下的距离估计
// 计算量。
//
// 代价：8 次 DE 迭代、60 步行进、6 次粗略法线采样。

constant float FAST_SCALE = 4.0f;

static float3 rotateY(float3 p, float angle) {
    float s = sin(angle), c = cos(angle);
    return float3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

static float fastMandelboxDE(float3 p, float time, thread float3 &trap) {
    float3 z = p;
    float derivative = 1.0f;
    float scale = -1.70f + 0.04f * sin(time * 0.10f);
    trap = float3(1e6f);
    for (int i = 0; i < 8; i++) {
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

static float fastMap(float3 p, float time) {
    float3 unusedTrap;
    return fastMandelboxDE(p / FAST_SCALE, time, unusedTrap) * FAST_SCALE;
}

static float3 calcNormal(float3 p, float time) {
    float2 e = float2(0.0025f, 0.0f);
    return normalize(float3(
        fastMap(p + e.xyy, time) - fastMap(p - e.xyy, time),
        fastMap(p + e.yxy, time) - fastMap(p - e.yxy, time),
        fastMap(p + e.yyx, time) - fastMap(p - e.yyx, time)
    ));
}

static float3 palette(float t) {
    return float3(0.28f, 0.35f, 0.40f)
        + float3(0.50f, 0.44f, 0.36f)
        * cos(6.28318f * (float3(0.74f, 0.91f, 1.06f) * t
            + float3(0.12f, 0.36f, 0.58f)));
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
    float3 background = float3(0.004f, 0.008f, 0.015f);
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
    float3 rayOrigin = rotateY((uniforms.patternTransform * float4(origin, 1.0f)).xyz, time * 0.06f);
    float3 rayDirection = rotateY(normalize(float3(uniforms.patternTransform * float4(boxRay, 0.0f))), time * 0.06f);
    float travel = 0.0f;
    for (int i = 0; i < 60; i++) {
        float3 point = rayOrigin + rayDirection * travel;
        float3 trap;
        float distance = fastMandelboxDE(point / FAST_SCALE, time, trap) * FAST_SCALE;
        if (distance < 0.0028f) {
            float3 normal = calcNormal(point, time);
            float diffuse = max(dot(normal, normalize(float3(-0.36f, 0.84f, 0.40f))), 0.0f);
            float rim = pow(1.0f - max(dot(-rayDirection, normal), 0.0f), 2.5f);
            float3 color = palette(dot(trap, float3(0.61f, 0.85f, 1.13f)) * 1.5f)
                * (0.22f + diffuse * 1.0f);
            color += float3(0.20f, 0.64f, 1.0f) * rim * 0.38f;
            return float4(color * exp(-travel * 0.20f), 1.0f);
        }
        travel += max(distance * 0.90f, 0.0030f);
        if (travel > 25.0f) break;
    }
    return float4(background, 1.0f);
}