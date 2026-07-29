// gyroid.metal – Triply Periodic Minimal Surface
//
// Renders a gyroid – a triply periodic minimal surface found in
// butterfly wing scales and block copolymer nanostructures.
// The surface divides space into two interpenetrating labyrinths,
// creating a mesmerizing infinite connected mesh.
//
// sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0

static float gyroidSDF(float3 p, float t) {
    float g = sin(p.x) * cos(p.y) + sin(p.y) * cos(p.z) + sin(p.z) * cos(p.x);
    // Animated deformation
    g += 0.15f * sin(p.x * 0.5f + t * 0.3f) * cos(p.z * 0.7f + t * 0.2f);
    return g * 0.3f; // scale to reasonable DE value
}

static float3 calcNormal(float3 p, float t) {
    float2 e = float2(0.003f, 0.0f);
    return normalize(float3(
        gyroidSDF(p + e.xyy, t) - gyroidSDF(p - e.xyy, t),
        gyroidSDF(p + e.yxy, t) - gyroidSDF(p - e.yxy, t),
        gyroidSDF(p + e.yyx, t) - gyroidSDF(p - e.yyx, t)
    ));
}

fragment float4 dynamicBoxFragment(
    DynamicBoxVertexOut        in         [[stage_in]],
    constant DynamicBoxUniforms &uniforms [[buffer(0)]],
    constant float4x4         *v2wMats    [[buffer(1)]],
    constant float4x4         *vpMatrices [[buffer(2)]])
{
    uint vi = min(in.viewIndex, uniforms.viewCount - 1u);

    float4x4 v2w    = v2wMats[vi];
    float3 camWorld = float3(v2w[3].x, v2w[3].y, v2w[3].z);

    float3 center  = uniforms.objectCenter.xyz;
    float  sc      = uniforms.boxScale;
    float3 boxEye  = (camWorld - center) / sc;
    float3 boxRd   = normalize(in.worldPos - camWorld);

    bool insideBox = all(abs(boxEye) < (DB_BOXDIMS - 1e-3f));
    float3 marchOrigin;
    float3 bgColor = float3(0.0f, 0.01f, 0.02f);

    if (!insideBox) {
        float3 entryNormal;
        float  tEnter = db_boxHit(boxEye, boxRd, DB_BOXDIMS, entryNormal, true);
        if (tEnter < 0.0f) return float4(bgColor, 1.0f);
        marchOrigin = boxEye + boxRd * (tEnter + 1e-3f);
    } else {
        marchOrigin = boxEye;
    }

    float3 exitNormal;
    float  tExit = db_boxHit(marchOrigin, boxRd, DB_BOXDIMS, exitNormal, false);
    if (tExit <= 0.0f) discard_fragment();

    float3 ro = (uniforms.patternTransform * float4(marchOrigin, 1.0f)).xyz;
    float3 rd = normalize(float3(uniforms.patternTransform * float4(boxRd, 0.0f)));

    float t = uniforms.time;
    float march = 0.0f;
    float maxD = 25.0f;

    for (int i = 0; i < 80; i++) {
        float3 p = ro + rd * march;
        float d = gyroidSDF(p, t);
        if (fabs(d) < 0.005f) {
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.5f, 1.0f, 0.4f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * abs(n.y);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 3.0f) * 0.5f;

            // Iridescent: color shifts with surface normal
            float hue = fract(n.x * 0.5f + n.y * 0.3f + n.z * 0.2f + t * 0.03f);
            float3 col;
            { float4 K = float4(1,2/3.f,1/3.f,3); float3 pp = abs(fract(hue+K.xyz)*6-K.www); col = clamp(pp-K.xxx,0,1); }
            col = col * (dif * 1.2f + amb * 0.4f) + float3(0.2f, 0.5f, 1.0f) * rim;
            col *= exp(-march * 0.2f);
            return float4(col, 1.0f);
        }
        march += max(fabs(d), 0.02f);
        if (march > maxD) break;
    }
    return float4(bgColor, 1.0f);
}
