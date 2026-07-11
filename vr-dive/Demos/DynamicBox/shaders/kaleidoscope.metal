// kaleidoscope.metal – Kaleidoscopic IFS Fractal
//
// A colorful iterated function system fractal that creates intricate
// kaleidoscopic patterns. Rendered with ray marching.
//
// Based on the "Kaleidoscopic IFS" family of fractals popularized by
// Knighty and Syntopia. Each iteration applies a rotation and a
// folding operation, creating infinitely detailed symmetric patterns.

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

// ─── Kaleidoscopic IFS DE ─────────────────────────────────────────────────────
static float kifsDE(float3 p, float t) {
    float3 q = p;
    float  scale = 2.5f;
    float  fold  = 1.0f;
    float  dr    = 1.0f;

    // Animated rotation axis
    float3 axis = normalize(float3(sin(t * 0.1f), cos(t * 0.13f), sin(t * 0.07f)));
    float3x3 r = rot(t * 0.2f, axis);

    for (int i = 0; i < 12; i++) {
        // Tetrahedral symmetry fold
        q = abs(q);
        float t0 = 2.0f * min(min(q.x, q.y), q.z);
        if (q.x < q.y && q.x < q.z) q.x = t0 - q.x;
        else if (q.y < q.z)          q.y = t0 - q.y;
        else                         q.z = t0 - q.z;

        // Sphere fold
        float m = dot(q, q);
        if (m < fold) q *= fold / m;

        // Scale and offset
        q = q * scale + float3(-0.5f, -0.5f, -0.5f);
        dr = dr * abs(scale) + 1.0f;

        // Rotation
        q = r * q;
    }

    return length(q) / dr - 0.05f;
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
    float maxD  = min(tExit, 3.0f);

    for (int i = 0; i < 120; i++) {
        float3 p = ro + rd * march;
        float d = kifsDE(p, t);
        if (d < 0.003f) {
            float3 n = calcNormal(p, t);
            float3 light = normalize(float3(0.6f, 0.8f, 0.4f));

            float dif = max(dot(n, light), 0.0f);
            float amb = 0.2f + 0.8f * max(dot(n, normalize(float3(0,1,1))), 0.0f);
            float rim = 1.0f - max(dot(-rd, n), 0.0f);
            rim = pow(rim, 3.0f) * 0.8f;

            // Iridescent color based on position + normal
            float hue = fract(length(p) * 0.6f + t * 0.03f + n.x * 0.2f);
            float3 col;
            {
                float4 K = float4(1.0f, 2.0f/3.0f, 1.0f/3.0f, 3.0f);
                float3 pp = abs(fract(hue + K.xyz) * 6.0f - K.www);
                col = clamp(pp - K.xxx, 0.0f, 1.0f);
            }
            col = col * (dif * 1.0f + amb * 0.5f) + float3(0.3f, 0.5f, 1.0f) * rim;
            col *= exp(-march * 0.35f);

            return float4(col, 1.0f);
        }
        march += d;
        if (march > maxD) break;
    }

    return float4(bgColor, 1.0f);
}
