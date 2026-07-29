// apollonian.metal – Apollonian Gasket (3D Sphere Fractal)
//
// Renders an Apollonian gasket – a fractal of nested spheres – using
// ray marching. Each iteration inverts space through spheres, creating
// an infinitely detailed packing of spheres within spheres.
//
// Based on the standard Apollonian fractal DE popularized by
// Syntopia / FractalForums.

// ─── Apollonian DE ────────────────────────────────────────────────────────────
// Standard Apollonian gasket: iterative sphere inversion + fold + scale.
static float apollonianDE(float3 p, float t) {
    float3 q = p;
    float  dr = 1.0f;
    float  scale = 3.0f;

    for (int i = 0; i < 14; i++) {
        // Sphere inversion: reflect across unit sphere
        float m = dot(q, q);
        if (m < 1e-8f) break;

        q = q / m;
        dr = dr / m * 2.0f + 1.0f;

        // Symmetry fold
        q = abs(q);

        // Offset then scale
        q = q + float3(0.5f, 0.5f, 0.5f);
        q = q * scale;
        dr = dr * scale + 1.0f;
    }

    return length(q) / dr - 0.05f;
}

// ─── Normal ───────────────────────────────────────────────────────────────────
static float3 calcNormal(float3 p, float t) {
    float2 e = float2(0.002f, 0.0f);
    return normalize(float3(
        apollonianDE(p + e.xyy, t) - apollonianDE(p - e.xyy, t),
        apollonianDE(p + e.yxy, t) - apollonianDE(p - e.yxy, t),
        apollonianDE(p + e.yyx, t) - apollonianDE(p - e.yyx, t)
    ));
}

// ─── Color palette ────────────────────────────────────────────────────────────
static float3 palette(float t) {
    float3 a = float3(0.5f, 0.5f, 0.5f);
    float3 b = float3(0.5f, 0.5f, 0.5f);
    float3 c = float3(1.0f, 0.7f, 0.4f);
    float3 d = float3(0.00f, 0.15f, 0.20f);
    return a + b * cos(6.28318f * (c * t + d));
}

// ─── Fragment shader ──────────────────────────────────────────────────────────
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
    float3 bgColor = float3(0.0f, 0.0f, 0.02f);

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

    // Slow rotation for visual interest
    float t = uniforms.time;
    float ca = cos(t * 0.07f), sa = sin(t * 0.07f);
    ro.xz = float2(ro.x*ca - ro.z*sa, ro.x*sa + ro.z*ca);
    rd.xz = float2(rd.x*ca - rd.z*sa, rd.x*sa + rd.z*ca);
    float cb = cos(t * 0.05f), sb = sin(t * 0.05f);
    ro.xy = float2(ro.x*cb - ro.y*sb, ro.x*sb + ro.y*cb);
    rd.xy = float2(rd.x*cb - rd.y*sb, rd.x*sb + rd.y*cb);

    // ─── Ray march ─────────────────────────────────────────────────────────
    float march = 0.0f;
    float maxD  = 25.0f;

    for (int i = 0; i < 120; i++) {
        float3 p = ro + rd * march;
        float d = apollonianDE(p, t);
        if (d < 0.003f) {
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.5f, 1.0f, 0.3f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 3.0f) * 0.6f;

            // Color by iteration depth + position
            float colT = length(p) * 0.3f + t * 0.03f;
            float3 col = palette(colT) * (dif * 1.2f + amb * 0.4f) + float3(0.2f, 0.4f, 1.0f) * rim;
            col *= exp(-march * 0.25f);

            return float4(col, 1.0f);
        }
        march += d;
        if (march > maxD) break;
    }

    return float4(bgColor, 1.0f);
}
