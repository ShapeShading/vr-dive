// kaleidoscope.metal – Kaleidoscopic IFS Fractal
//
// A colorful kaleidoscopic fractal rendered with ray marching.
// Uses iterated fold-and-scale transformations with a sphere inversion,
// creating intricate self-symmetric 3D patterns.

// ─── Rotation matrix ──────────────────────────────────────────────────────────
static float3x3 rot(float a, float3 ax) {
    float s = sin(a), c = cos(a);
    float3 t = (1.0f - c) * ax;
    return float3x3(
        float3(c + t.x*ax.x, t.x*ax.y + s*ax.z, t.x*ax.z - s*ax.y),
        float3(t.y*ax.x - s*ax.z, c + t.y*ax.y, t.y*ax.z + s*ax.x),
        float3(t.z*ax.x + s*ax.y, t.z*ax.y - s*ax.x, c + t.z*ax.z)
    );
}

// ─── Color palette ────────────────────────────────────────────────────────────
static float3 palette(float t) {
    float3 a = float3(0.5f, 0.5f, 0.5f);
    float3 b = float3(0.5f, 0.5f, 0.5f);
    float3 c = float3(1.0f, 0.7f, 0.4f);
    float3 d = float3(0.00f, 0.15f, 0.20f);
    return a + b * cos(6.28318f * (c * t + d));
}

// ─── Kaleidoscopic IFS DE (KaliSet variant) ──────────────────────────────────
static float kifsDE(float3 p, float t) {
    float3 q = p;
    float  scale = 2.0f;
    float  dr    = 1.0f;
    float  foldR = 1.0f;

    // Slow rotation of the whole fractal
    float3 axis = normalize(float3(1.0f, 0.7f, 0.3f));
    float3x3 rotM = rot(t * 0.12f, axis);

    for (int i = 0; i < 14; i++) {
        // Box fold: reflect across +/- 1 planes
        q = abs(q);
        if (q.x > q.y) { float tmp = q.x; q.x = q.y; q.y = tmp; }
        if (q.x > q.z) { float tmp = q.x; q.x = q.z; q.z = tmp; }
        if (q.y > q.z) { float tmp = q.y; q.y = q.z; q.z = tmp; }

        // Sphere fold: invert points inside radius foldR
        float m = dot(q, q);
        if (m < foldR) {
            q *= foldR / max(m, 1e-8f);
            dr *= foldR / max(m, 1e-8f);
        }

        // Scale and shift
        q = q * scale + float3(-0.8f, -0.8f, -0.8f);
        dr = dr * scale + 1.0f;

        // Apply global rotation
        q = rotM * q;
    }

    return (length(q) - 0.05f) / dr;
}

// ─── Normal ───────────────────────────────────────────────────────────────────
static float3 calcNormal(float3 p, float t) {
    float2 e = float2(0.001f, 0.0f);
    return normalize(float3(
        kifsDE(p + e.xyy, t) - kifsDE(p - e.xyy, t),
        kifsDE(p + e.yxy, t) - kifsDE(p - e.yxy, t),
        kifsDE(p + e.yyx, t) - kifsDE(p - e.yyx, t)
    ));
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
    float3 bgColor = float3(0.01f, 0.0f, 0.02f);

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

    // ─── Ray march ─────────────────────────────────────────────────────────
    float t     = uniforms.time;
    float march = 0.0f;
    float maxD  = 25.0f;

    for (int i = 0; i < 100; i++) {
        float3 p = ro + rd * march;
        float d = kifsDE(p, t);
        if (d < 0.003f) {
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.5f, 1.0f, 0.3f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.3f + 0.7f * max(dot(n, float3(0,1,0)), 0.0f);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 3.0f) * 0.6f;

            // Iridescent coloring
            float colT = length(p) * 0.5f + t * 0.04f + n.x * 0.15f;
            float3 col = palette(colT) * (dif * 1.2f + amb * 0.4f) + float3(0.3f, 0.5f, 1.0f) * rim;
            col *= exp(-march * 0.3f);

            return float4(col, 1.0f);
        }
        march += d;
        if (march > maxD) break;
    }

    return float4(bgColor, 1.0f);
}
